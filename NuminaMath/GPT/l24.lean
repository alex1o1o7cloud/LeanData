import Mathlib

namespace value_of_x_plus_y_l24_24508

theorem value_of_x_plus_y 
  (x y : ℝ)
  (h1 : |x| = 3)
  (h2 : |y| = 2)
  (h3 : x > y) :
  x + y = 5 ∨ x + y = 1 := 
  sorry

end value_of_x_plus_y_l24_24508


namespace range_of_k_l24_24199

noncomputable def equation (k x : ℝ) : ℝ := 4^x - k * 2^x + k + 3

theorem range_of_k {x : ℝ} (h : ∀ k, equation k x = 0 → ∃! x : ℝ, equation k x = 0) :
  ∃ k : ℝ, (k = 6 ∨ k < -3)∧ (∀ y, equation k y ≠ 0 → (y ≠ x)) :=
sorry

end range_of_k_l24_24199


namespace fractional_eq_has_root_l24_24365

theorem fractional_eq_has_root (x : ℝ) (m : ℝ) (h : x ≠ 4) :
    (3 / (x - 4) + (x + m) / (4 - x) = 1) → m = -1 :=
by
    intros h_eq
    sorry

end fractional_eq_has_root_l24_24365


namespace john_total_water_usage_l24_24381

-- Define the basic conditions
def total_days_in_weeks (weeks : ℕ) : ℕ := weeks * 7
def showers_every_other_day (days : ℕ) : ℕ := days / 2
def total_minutes_shower (showers : ℕ) (minutes_per_shower : ℕ) : ℕ := showers * minutes_per_shower
def total_water_usage (total_minutes : ℕ) (water_per_minute : ℕ) : ℕ := total_minutes * water_per_minute

-- Main statement
theorem john_total_water_usage :
  total_water_usage (total_minutes_shower (showers_every_other_day (total_days_in_weeks 4)) 10) 2 = 280 :=
by
  sorry

end john_total_water_usage_l24_24381


namespace calculate_Delta_l24_24697

-- Define the Delta operation
def Delta (a b : ℚ) : ℚ := (a^2 + b^2) / (1 + a^2 * b^2)

-- Constants for the specific problem
def two := (2 : ℚ)
def three := (3 : ℚ)
def four := (4 : ℚ)

theorem calculate_Delta : Delta (Delta two three) four = 5945 / 4073 := by
  sorry

end calculate_Delta_l24_24697


namespace equivalent_proof_problem_l24_24497

theorem equivalent_proof_problem (x : ℤ) (h : (x + 2) * (x - 2) = 1221) :
    (x = 35 ∨ x = -35) ∧ ((x + 1) * (x - 1) = 1224) :=
sorry

end equivalent_proof_problem_l24_24497


namespace polygon_sides_eq_13_l24_24117

theorem polygon_sides_eq_13 (n : ℕ) (h : n * (n - 3) = 5 * n) : n = 13 := by
  sorry

end polygon_sides_eq_13_l24_24117


namespace least_possible_value_l24_24629

theorem least_possible_value (x y : ℝ) : (x + y - 1)^2 + (x * y)^2 ≥ 0 :=
by 
  sorry

end least_possible_value_l24_24629


namespace even_sum_exactly_one_even_l24_24618

theorem even_sum_exactly_one_even (a b c : ℕ) (h : (a + b + c) % 2 = 0) :
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) :=
by
  sorry

end even_sum_exactly_one_even_l24_24618


namespace sum_of_3digit_numbers_remainder_2_l24_24924

-- Define the smallest and largest three-digit numbers leaving remainder 2 when divided by 5
def smallest : ℕ := 102
def largest  : ℕ := 997
def common_diff : ℕ := 5

-- Define the arithmetic sequence
def seq_length : ℕ := ((largest - smallest) / common_diff) + 1
def sequence_sum : ℕ := seq_length * (smallest + largest) / 2

-- The theorem to be proven
theorem sum_of_3digit_numbers_remainder_2 : sequence_sum = 98910 :=
by
  sorry

end sum_of_3digit_numbers_remainder_2_l24_24924


namespace fraction_product_correct_l24_24800

theorem fraction_product_correct : (3 / 5) * (4 / 7) * (5 / 9) = 4 / 21 :=
by
  sorry

end fraction_product_correct_l24_24800


namespace axis_of_symmetry_l24_24860

theorem axis_of_symmetry {a b c : ℝ} (h1 : (2 : ℝ) * (a * 2 + b) + c = 5) (h2 : (4 : ℝ) * (a * 4 + b) + c = 5) : 
  (2 + 4) / 2 = 3 := 
by 
  sorry

end axis_of_symmetry_l24_24860


namespace evaluate_g_at_5_l24_24218

def g (x : ℝ) : ℝ := 5 * x + 2

theorem evaluate_g_at_5 : g 5 = 27 := by
  sorry

end evaluate_g_at_5_l24_24218


namespace sasha_questions_per_hour_l24_24741

-- Define the total questions and the time she worked, and the remaining questions
def total_questions : ℕ := 60
def time_worked : ℕ := 2
def remaining_questions : ℕ := 30

-- Define the number of questions she completed
def questions_completed := total_questions - remaining_questions

-- Define the rate at which she completes questions per hour
def questions_per_hour := questions_completed / time_worked

-- The theorem to prove
theorem sasha_questions_per_hour : questions_per_hour = 15 := 
by
  -- Here we would prove the theorem, but we're using sorry to skip the proof for now
  sorry

end sasha_questions_per_hour_l24_24741


namespace find_k_l24_24493

theorem find_k (k α β : ℝ)
  (h1 : (∀ x : ℝ, x^2 - (k-1) * x - 3*k - 2 = 0 → x = α ∨ x = β))
  (h2 : α^2 + β^2 = 17) :
  k = 2 :=
sorry

end find_k_l24_24493


namespace range_of_a1_l24_24834

noncomputable def sequence_a (n : ℕ) : ℤ := sorry
noncomputable def sum_S (n : ℕ) : ℤ := sorry

theorem range_of_a1 :
  (∀ n : ℕ, n > 0 → sum_S n + sum_S (n+1) = 2 * n^2 + n) ∧
  (∀ n : ℕ, n > 0 → sequence_a n < sequence_a (n+1)) →
  -1/4 < sequence_a 1 ∧ sequence_a 1 < 3/4 := sorry

end range_of_a1_l24_24834


namespace probability_of_composite_l24_24545

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l24_24545


namespace emily_stickers_l24_24771

theorem emily_stickers:
  ∃ S : ℕ, (S % 4 = 2) ∧
           (S % 6 = 2) ∧
           (S % 9 = 2) ∧
           (S % 10 = 2) ∧
           (S > 2) ∧
           (S = 182) :=
  sorry

end emily_stickers_l24_24771


namespace smallest_b_l24_24873

noncomputable def Q (b : ℤ) (x : ℤ) : ℤ := sorry -- Q is a polynomial, will be defined in proof

theorem smallest_b (b : ℤ) 
  (h1 : b > 0) 
  (h2 : ∀ x, x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 → Q b x = b) 
  (h3 : ∀ x, x = 1 ∨ x = 3 ∨ x = 5 ∨ x = 7 → Q b x = -b) 
  : b = 315 := sorry

end smallest_b_l24_24873


namespace zero_descriptions_l24_24225

-- Defining the descriptions of zero satisfying the given conditions.
def description1 : String := "The number corresponding to the origin on the number line."
def description2 : String := "The number that represents nothing."
def description3 : String := "The number that, when multiplied by any other number, equals itself."

-- Lean statement to prove the validity of the descriptions.
theorem zero_descriptions : 
  description1 = "The number corresponding to the origin on the number line." ∧
  description2 = "The number that represents nothing." ∧
  description3 = "The number that, when multiplied by any other number, equals itself." :=
by
  -- Proof omitted
  sorry

end zero_descriptions_l24_24225


namespace quadratic_roots_transformation_l24_24732

noncomputable def transformed_polynomial (p q r : ℝ) : Polynomial ℝ :=
  Polynomial.X^2 + (p*q + 2*q)*Polynomial.X + (p^3*r + p*q^2 + q^2)

noncomputable def original_polynomial (p q r : ℝ) : Polynomial ℝ :=
  p * Polynomial.X^2 + q * Polynomial.X + r

theorem quadratic_roots_transformation (p q r : ℝ) (u v : ℝ)
  (huv1 : u + v = -q / p)
  (huv2 : u * v = r / p) :
  transformed_polynomial p q r = Polynomial.monomial 2 1 +
    Polynomial.monomial 1 (p*q + 2*q) +
    Polynomial.monomial 0 (p^3*r + p*q^2 + q^2) :=
by {
  sorry
}

end quadratic_roots_transformation_l24_24732


namespace probability_composite_product_l24_24533

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l24_24533


namespace inscribed_rectangle_sides_l24_24863

theorem inscribed_rectangle_sides {a b c : ℕ} (h₀ : a = 3) (h₁ : b = 4) (h₂ : c = 5) (ratio : ℚ) (h_ratio : ratio = 1 / 3) :
  ∃ (x y : ℚ), x = 20 / 29 ∧ y = 60 / 29 ∧ x = ratio * y :=
by
  sorry

end inscribed_rectangle_sides_l24_24863


namespace inequality_inequal_pos_numbers_l24_24888

theorem inequality_inequal_pos_numbers {a b : ℝ} (h : a ≠ b) (ha : 0 < a) (hb : 0 < b) : 
  (2 / (1 / a + 1 / b)) < real.sqrt (a * b) ∧ real.sqrt (a * b) < (a + b) / 2 :=
by
  sorry

end inequality_inequal_pos_numbers_l24_24888


namespace roots_of_equation_l24_24916

theorem roots_of_equation (
  x y: ℝ
) (h1: x + y = 10) (h2: |x - y| = 12):
  (x = 11 ∧ y = -1) ∨ (x = -1 ∧ y = 11) ↔ ∃ (a b: ℝ), a = 11 ∧ b = -1 ∨ a = -1 ∧ b = 11 ∧ a^2 - 10*a - 22 = 0 ∧ b^2 - 10*b - 22 = 0 := 
by sorry

end roots_of_equation_l24_24916


namespace Lizzy_money_after_loan_l24_24399

theorem Lizzy_money_after_loan :
  let initial_savings := 30
  let loaned_amount := 15
  let interest_rate := 0.20
  let interest := loaned_amount * interest_rate
  let total_amount_returned := loaned_amount + interest
  let remaining_money := initial_savings - loaned_amount
  let total_money := remaining_money + total_amount_returned
  total_money = 33 :=
by
  sorry

end Lizzy_money_after_loan_l24_24399


namespace quadratic_roots_condition_l24_24071

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k*x^2 + 2*x + 1 = 0 ∧ k*y^2 + 2*y + 1 = 0) ↔ (k < 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_roots_condition_l24_24071


namespace olivia_quarters_left_l24_24247

-- Define the initial condition and action condition as parameters
def initial_quarters : ℕ := 11
def quarters_spent : ℕ := 4
def quarters_left : ℕ := initial_quarters - quarters_spent

-- The theorem to state the result
theorem olivia_quarters_left : quarters_left = 7 := by
  sorry

end olivia_quarters_left_l24_24247


namespace average_salary_of_all_workers_l24_24001

theorem average_salary_of_all_workers :
  let technicians := 7
  let technicians_avg_salary := 20000
  let rest := 49 - technicians
  let rest_avg_salary := 6000
  let total_workers := 49
  let total_tech_salary := technicians * technicians_avg_salary
  let total_rest_salary := rest * rest_avg_salary
  let total_salary := total_tech_salary + total_rest_salary
  (total_salary / total_workers) = 8000 := by
  sorry

end average_salary_of_all_workers_l24_24001


namespace journey_time_calculation_l24_24941

theorem journey_time_calculation (dist totalDistance : ℝ) (rate1 rate2 : ℝ)
  (firstHalfDistance secondHalfDistance : ℝ) (time1 time2 totalTime : ℝ) :
  totalDistance = 224 ∧ rate1 = 21 ∧ rate2 = 24 ∧
  firstHalfDistance = totalDistance / 2 ∧ secondHalfDistance = totalDistance / 2 ∧
  time1 = firstHalfDistance / rate1 ∧ time2 = secondHalfDistance / rate2 ∧
  totalTime = time1 + time2 →
  totalTime = 10 :=
sorry

end journey_time_calculation_l24_24941


namespace sufficient_but_not_necessary_condition_l24_24390

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.tan (ω * x + φ)
def P (f : ℝ → ℝ) : Prop := f 0 = 0
def Q (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem sufficient_but_not_necessary_condition (ω : ℝ) (φ : ℝ) (hω : ω > 0) :
  (P (f ω φ) → Q (f ω φ)) ∧ ¬(Q (f ω φ) → P (f ω φ)) := by
  sorry

end sufficient_but_not_necessary_condition_l24_24390


namespace find_multiple_of_q_l24_24856

variable (p q m : ℚ)

theorem find_multiple_of_q (h1 : p / q = 3 / 4) (h2 : 3 * p + m * q = 6.25) :
  m = 4 :=
sorry

end find_multiple_of_q_l24_24856


namespace lena_more_than_nicole_l24_24869

theorem lena_more_than_nicole :
  ∀ (L K N : ℝ),
    L = 37.5 →
    (L + 9.5) = 5 * K →
    K = N - 8.5 →
    (L - N) = 19.6 :=
by
  intros L K N hL hLK hK
  sorry

end lena_more_than_nicole_l24_24869


namespace treasures_found_second_level_l24_24169

theorem treasures_found_second_level:
  ∀ (P T1 S T2 : ℕ), 
    P = 4 → 
    T1 = 6 → 
    S = 32 → 
    S = P * T1 + P * T2 → 
    T2 = 2 := 
by
  intros P T1 S T2 hP hT1 hS hTotal
  sorry

end treasures_found_second_level_l24_24169


namespace linear_function_correct_max_profit_correct_min_selling_price_correct_l24_24785

-- Definition of the linear function
def linear_function (x : ℝ) : ℝ :=
  -2 * x + 360

-- Definition of monthly profit function
def profit_function (x : ℝ) : ℝ :=
  (-2 * x + 360) * (x - 30)

noncomputable def max_profit_statement : Prop :=
  ∃ x w, x = 105 ∧ w = 11250 ∧ profit_function x = w

noncomputable def min_selling_price (profit : ℝ) : Prop :=
  ∃ x, profit_function x ≥ profit ∧ x ≥ 80

-- The proof statements
theorem linear_function_correct : linear_function 30 = 300 ∧ linear_function 45 = 270 :=
  by
    sorry

theorem max_profit_correct : max_profit_statement :=
  by
    sorry

theorem min_selling_price_correct : min_selling_price 10000 :=
  by
    sorry

end linear_function_correct_max_profit_correct_min_selling_price_correct_l24_24785


namespace minimum_value_of_sum_of_squares_l24_24526

variable {x y : ℝ}

theorem minimum_value_of_sum_of_squares (h : x^2 + 2*x*y - y^2 = 7) : 
  x^2 + y^2 ≥ 7 * Real.sqrt 2 / 2 := by 
    sorry

end minimum_value_of_sum_of_squares_l24_24526


namespace problem_statement_l24_24339

variable (a b c d : ℝ)

noncomputable def circle_condition_1 : Prop := a = (1 : ℝ) / a
noncomputable def circle_condition_2 : Prop := b = (1 : ℝ) / b
noncomputable def circle_condition_3 : Prop := c = (1 : ℝ) / c
noncomputable def circle_condition_4 : Prop := d = (1 : ℝ) / d

theorem problem_statement (h1 : circle_condition_1 a)
                          (h2 : circle_condition_2 b)
                          (h3 : circle_condition_3 c)
                          (h4 : circle_condition_4 d) :
    2 * (a^2 + b^2 + c^2 + d^2) = (a + b + c + d)^2 := 
by
  sorry

end problem_statement_l24_24339


namespace binom_12_6_eq_924_l24_24807

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binom_12_6_eq_924_l24_24807


namespace find_a_l24_24308

noncomputable def point1 : ℝ × ℝ := (-3, 6)
noncomputable def point2 : ℝ × ℝ := (2, -1)

theorem find_a (a : ℝ) :
  let direction : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)
  direction = (5, -7) →
  let normalized_direction : ℝ × ℝ := (direction.1 / -7, direction.2 / -7)
  normalized_direction = (a, -1) →
  a = -5 / 7 :=
by 
  intros 
  sorry

end find_a_l24_24308


namespace angle_diff_l24_24049

-- Given conditions as definitions
def angle_A : ℝ := 120
def angle_B : ℝ := 50
def angle_D : ℝ := 60
def angle_E : ℝ := 140

-- Prove the difference between angle BCD and angle AFE is 10 degrees
theorem angle_diff (AB_parallel_DE : ∀ (A B D E : ℝ), AB_parallel_DE)
                 (angle_A_def : angle_A = 120)
                 (angle_B_def : angle_B = 50)
                 (angle_D_def : angle_D = 60)
                 (angle_E_def : angle_E = 140) :
    let angle_3 : ℝ := 180 - angle_A
    let angle_4 : ℝ := 180 - angle_E
    let angle_BCD : ℝ := angle_B + angle_D
    let angle_AFE : ℝ := angle_3 + angle_4
    angle_BCD - angle_AFE = 10 :=
by {
  sorry
}

end angle_diff_l24_24049


namespace sum_of_first_4n_integers_l24_24368

theorem sum_of_first_4n_integers (n : ℕ) 
  (h : (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 150) : 
  (4 * n * (4 * n + 1)) / 2 = 300 :=
by
  sorry

end sum_of_first_4n_integers_l24_24368


namespace find_a_b_l24_24772

theorem find_a_b :
  ∃ (a b : ℚ), 
    (∀ x : ℚ, x = 2 → (a * x^3 - 6 * x^2 + b * x - 5 - 3 = 0)) ∧
    (∀ x : ℚ, x = -1 → (a * x^3 - 6 * x^2 + b * x - 5 - 7 = 0)) ∧
    (a = -2/3 ∧ b = -52/3) :=
by {
  sorry
}

end find_a_b_l24_24772


namespace total_tank_capacity_l24_24654

-- Definitions based on conditions
def initial_condition (w c : ℝ) : Prop := w / c = 1 / 3
def after_adding_five (w c : ℝ) : Prop := (w + 5) / c = 1 / 2

-- The problem statement
theorem total_tank_capacity (w c : ℝ) (h1 : initial_condition w c) (h2 : after_adding_five w c) : c = 30 :=
sorry

end total_tank_capacity_l24_24654


namespace min_ab_value_l24_24362

variable (a b : ℝ) 

theorem min_ab_value (h : (4 / a) + (1 / b) = Real.sqrt (a * b)) (ha : a > 0) (hb : b > 0) : a * b = 4 :=
sorry

end min_ab_value_l24_24362


namespace solution_y_values_l24_24414
-- Import the necessary libraries

-- Define the system of equations and the necessary conditions
def equation1 (x : ℝ) := x^2 - 6*x + 8 = 0
def equation2 (x y : ℝ) := 2*x - y = 6

-- The main theorem to be proven
theorem solution_y_values : ∃ x1 x2 y1 y2 : ℝ, 
  (equation1 x1 ∧ equation1 x2 ∧ equation2 x1 y1 ∧ equation2 x2 y2 ∧ 
  y1 = 2 ∧ y2 = -2) :=
by
  -- Use the provided solutions in the problem statement
  use 4, 2, 2, -2
  sorry  -- The details of the proof are omitted.

end solution_y_values_l24_24414


namespace combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400_l24_24042

noncomputable def num_combinations_4_blocks_no_same_row_col :=
  (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

theorem combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400 :
  num_combinations_4_blocks_no_same_row_col = 5400 := 
by
  sorry

end combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400_l24_24042


namespace cuboid_unshaded_face_area_l24_24745

theorem cuboid_unshaded_face_area 
  (x : ℝ)
  (h1 : ∀ a  : ℝ, a = 4*x) -- Condition: each unshaded face area = 4 * shaded face area
  (h2 : 18*x = 72)         -- Condition: total surface area = 72 cm²
  : 4*x = 16 :=            -- Conclusion: area of one visible unshaded face is 16 cm²
by
  sorry

end cuboid_unshaded_face_area_l24_24745


namespace expression_approx_l24_24152

noncomputable def simplified_expression : ℝ :=
  (Real.sqrt 97 + 9 * Real.sqrt 6 + 5 * Real.sqrt 5) / (3 * Real.sqrt 6 + 7)

theorem expression_approx : abs (simplified_expression - 3.002) < 0.001 :=
by
  -- Proof omitted
  sorry

end expression_approx_l24_24152


namespace train_speed_l24_24465

theorem train_speed 
(length_of_train : ℕ) 
(time_to_cross_pole : ℕ) 
(h_length : length_of_train = 135) 
(h_time : time_to_cross_pole = 9) : 
  (length_of_train / time_to_cross_pole) * 3.6 = 54 :=
by 
  sorry

end train_speed_l24_24465


namespace saree_final_price_l24_24757

noncomputable def saree_original_price : ℝ := 5000
noncomputable def first_discount_rate : ℝ := 0.20
noncomputable def second_discount_rate : ℝ := 0.15
noncomputable def third_discount_rate : ℝ := 0.10
noncomputable def fourth_discount_rate : ℝ := 0.05
noncomputable def tax_rate : ℝ := 0.12
noncomputable def luxury_tax_rate : ℝ := 0.05
noncomputable def custom_fee : ℝ := 200
noncomputable def exchange_rate_to_usd : ℝ := 0.013

theorem saree_final_price :
  let price_after_first_discount := saree_original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  let price_after_third_discount := price_after_second_discount * (1 - third_discount_rate)
  let price_after_fourth_discount := price_after_third_discount * (1 - fourth_discount_rate)
  let tax := price_after_fourth_discount * tax_rate
  let luxury_tax := price_after_fourth_discount * luxury_tax_rate
  let total_charges := tax + luxury_tax + custom_fee
  let total_price_rs := price_after_fourth_discount + total_charges
  let final_price_usd := total_price_rs * exchange_rate_to_usd
  abs (final_price_usd - 46.82) < 0.01 :=
by sorry

end saree_final_price_l24_24757


namespace reflection_of_point_l24_24598

def reflect_across_y_neg_x (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (y, x)

theorem reflection_of_point
  (P : ℝ × ℝ)
  (h : P = (8, -3)) :
  reflect_across_y_neg_x P = (3, -8) :=
by
  rw [h]
  sorry

end reflection_of_point_l24_24598


namespace train_speed_A_to_B_l24_24167

-- Define the constants
def distance : ℝ := 480
def return_speed : ℝ := 120
def return_time_longer : ℝ := 1

-- Define the train's speed function on its way from A to B
noncomputable def train_speed : ℝ := distance / (4 - return_time_longer) -- This simplifies directly to 160 based on the provided conditions.

-- State the theorem
theorem train_speed_A_to_B :
  distance / train_speed + return_time_longer = distance / return_speed :=
by
  -- Result follows from the given conditions directly
  sorry

end train_speed_A_to_B_l24_24167


namespace find_z_l24_24602

theorem find_z (z : ℚ) : (7 + 11 + 23) / 3 = (15 + z) / 2 → z = 37 / 3 :=
by
  sorry

end find_z_l24_24602


namespace space_shuttle_speed_l24_24947

theorem space_shuttle_speed :
  ∀ (speed_kph : ℕ) (minutes_per_hour seconds_per_minute : ℕ),
    speed_kph = 32400 →
    minutes_per_hour = 60 →
    seconds_per_minute = 60 →
    (speed_kph / (minutes_per_hour * seconds_per_minute)) = 9 :=
by
  intros speed_kph minutes_per_hour seconds_per_minute
  intro h_speed
  intro h_minutes
  intro h_seconds
  sorry

end space_shuttle_speed_l24_24947


namespace downstream_speed_l24_24310

variable (Vu : ℝ) (Vs : ℝ)

theorem downstream_speed (h1 : Vu = 25) (h2 : Vs = 35) : (2 * Vs - Vu = 45) :=
by
  sorry

end downstream_speed_l24_24310


namespace task_assignments_count_l24_24765

theorem task_assignments_count (S : Finset (Fin 5)) :
  ∃ (assignments : Fin 5 → Fin 3),  
    (∀ t, assignments t ≠ t) ∧ 
    (∀ v, ∃ t, assignments t = v) ∧ 
    (∀ t, (t = 4 → assignments t = 1)) ∧ 
    S.card = 60 :=
by sorry

end task_assignments_count_l24_24765


namespace increasing_condition_l24_24093

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) + a * (Real.exp (-x))

theorem increasing_condition (a : ℝ) : (∀ x : ℝ, 0 ≤ (Real.exp (2 * x) - a) / (Real.exp x)) ↔ a ≤ 0 :=
by
  sorry

end increasing_condition_l24_24093


namespace composite_dice_product_probability_l24_24536

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l24_24536


namespace animal_market_problem_l24_24140

theorem animal_market_problem:
  ∃ (s c : ℕ), 0 < s ∧ 0 < c ∧ 28 * s + 27 * c = 1200 ∧ c > s :=
by
  sorry

end animal_market_problem_l24_24140


namespace parallelogram_side_length_l24_24789

-- We need trigonometric functions and operations with real numbers.
open Real

theorem parallelogram_side_length (s : ℝ) 
  (h_side_lengths : s > 0 ∧ 3 * s > 0) 
  (h_angle : sin (30 / 180 * π) = 1 / 2) 
  (h_area : 3 * s * (s * sin (30 / 180 * π)) = 9 * sqrt 3) :
  s = 3 * sqrt 2 :=
by
  sorry

end parallelogram_side_length_l24_24789


namespace least_number_to_subtract_l24_24633

theorem least_number_to_subtract (n : ℕ) : (n = 5) → (5000 - n) % 37 = 0 :=
by sorry

end least_number_to_subtract_l24_24633


namespace sum_a_cos_phi_eq_sum_a_sin_phi_eq_cosine_binomial_sum_sine_binomial_sum_l24_24323

noncomputable def sumacos (a : ℝ) (φ : ℝ) (h : |a| < 1) : ℝ :=
1 + a * cos φ + a^2 * cos (2 * φ) + ∑ k, a^(k+1) * cos ((k+1) * φ)

theorem sum_a_cos_phi_eq (a : ℝ) (φ : ℝ) (h : |a| < 1) :
  sumacos a φ h = (1 - a * cos φ) / (1 - 2 * a * cos φ + a^2) :=
by sorry

noncomputable def sumasin (a : ℝ) (φ : ℝ) (h : |a| < 1) : ℝ :=
a * sin φ + a^2 * sin (2 * φ) + ∑ k, a^((k+1)) * sin((k+1) * φ)

theorem sum_a_sin_phi_eq (a : ℝ) (φ : ℝ) (h : |a| < 1) :
  sumasin a φ h = (a * sin φ) / (1 - 2 * a * cos φ + a^2) :=
by sorry

def cosine_sum (n : ℕ) (φ : ℝ) : ℝ :=
cos φ + ∑ k in finset.range (n+1), nat.choose n k * cos ((k + 1) * φ)

theorem cosine_binomial_sum (n : ℕ) (φ : ℝ) :
  cosine_sum n φ = 2^n * cos^n (φ / 2) * cos ((n+2) * φ / 2) :=
by sorry

def sine_sum (n : ℕ) (φ : ℝ) : ℝ :=
sin φ + ∑ k in finset.range (n+1), nat.choose n k * sin ((k + 1) * φ)

theorem sine_binomial_sum (n : ℕ) (φ : ℝ) :
  sine_sum n φ = 2^n * cos^n (φ / 2) * sin ((n+2) * φ / 2) :=
by sorry

end sum_a_cos_phi_eq_sum_a_sin_phi_eq_cosine_binomial_sum_sine_binomial_sum_l24_24323


namespace binomial_12_6_eq_924_l24_24802

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_l24_24802


namespace find_M_coordinates_l24_24249

-- Definition of the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y ^ 2 = 2 * p * x

-- Definition to check if point M lies according to given conditions
def matchesCondition
  (p : ℝ) (M P O F : ℝ × ℝ) : Prop :=
  let xO := O.1
  let yO := O.2
  let xP := P.1
  let yP := P.2
  let xM := M.1
  let yM := M.2
  let xF := F.1
  let yF := F.2
  (xP = 2) ∧ (yP = 2 * p) ∧
  (xO = 0) ∧ (yO = 0) ∧
  (xF = p / 2) ∧ (yF = 0) ∧
  (Real.sqrt ((xM - xP) ^ 2 + (yM - yP) ^ 2) =
  Real.sqrt ((xM - xO) ^ 2 + (yM - yO) ^ 2)) ∧
  (Real.sqrt ((xM - xP) ^ 2 + (yM - yP) ^ 2) =
  Real.sqrt ((xM - xF) ^ 2 + (yM - yF) ^ 2))

-- Prove the coordinates of M satisfy the conditions
theorem find_M_coordinates :
  ∀ p : ℝ, p > 0 →
  matchesCondition p (1/4, 7/4) (2, 2 * p) (0, 0) (p / 2, 0) :=
by
  intros p hp
  simp [parabola, matchesCondition]
  sorry

end find_M_coordinates_l24_24249


namespace total_water_heaters_l24_24921

-- Define the conditions
variables (W C : ℕ) -- W: capacity of Wallace's water heater, C: capacity of Catherine's water heater
variable (wallace_3over4_full : W = 40 ∧ W * 3 / 4 ∧ C = W / 2 ∧ C * 3 / 4)

-- The proof problem
theorem total_water_heaters (wallace_3over4_full : W = 40 ∧ (W * 3 / 4 = 30) ∧ C = W / 2 ∧ (C * 3 / 4 = 15)) : W * 3 / 4 + C * 3 / 4 = 45 :=
sorry

end total_water_heaters_l24_24921


namespace hyperbola_a_unique_l24_24211

-- Definitions from the conditions
def hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 4 = 1
def foci (c : ℝ) : Prop := c = 2 * Real.sqrt 3
def a_positive (a : ℝ) : Prop := a > 0

-- Statement to prove
theorem hyperbola_a_unique (a : ℝ) (h : hyperbola 0 0 a ∧ foci (2 * Real.sqrt 3) ∧ a_positive a) : a = 2 * Real.sqrt 2 := 
sorry

end hyperbola_a_unique_l24_24211


namespace hall_ratio_l24_24272

theorem hall_ratio (w l : ℕ) (h1 : w * l = 450) (h2 : l - w = 15) : w / l = 1 / 2 :=
by sorry

end hall_ratio_l24_24272


namespace acme_cheaper_min_shirts_l24_24795

theorem acme_cheaper_min_shirts :
  ∃ x : ℕ, 60 + 11 * x < 10 + 16 * x ∧ x = 11 :=
by {
  sorry
}

end acme_cheaper_min_shirts_l24_24795


namespace fraction_problem_l24_24174

def fractions : List (ℚ) := [4/3, 7/5, 12/10, 23/20, 45/40, 89/80]
def subtracted_value : ℚ := -8

theorem fraction_problem :
  (fractions.sum - subtracted_value) = -163 / 240 := by
  sorry

end fraction_problem_l24_24174


namespace temp_below_zero_negative_l24_24704

theorem temp_below_zero_negative (temp_below_zero : ℤ) : temp_below_zero = -3 ↔ temp_below_zero < 0 := by
  sorry

end temp_below_zero_negative_l24_24704


namespace line_through_points_slope_intercept_sum_l24_24746

theorem line_through_points_slope_intercept_sum :
  ∃ m b : ℝ, (∀ x y : ℝ, (y = m * x + b) → ((((x, y) = (-3, 1)) ∨ ((x, y) = (1, 3))) ⇒ y = m * x + b)) ∧ (m + b = 3) :=
begin
  sorry
end

end line_through_points_slope_intercept_sum_l24_24746


namespace sequence_formula_l24_24261

theorem sequence_formula (a : ℕ → ℚ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = -1/2) (h3 : a 3 = 1/3) (h4 : a 4 = -1/4) :
  a n = (-1)^(n+1) * (1/n) :=
sorry

end sequence_formula_l24_24261


namespace greatest_three_digit_multiple_of_23_l24_24625

theorem greatest_three_digit_multiple_of_23 : ∃ n : ℕ, n % 23 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℕ, m % 23 = 0 ∧ 100 ≤ m ∧ m ≤ 999 → m ≤ n := 
by
  use 989
  split
  · -- 989 is a multiple of 23
    exact (by norm_num : 989 % 23 = 0)
  · split
    · -- 989 is at least 100
      exact (by norm_num : 100 ≤ 989)
    · split
      · -- 989 is at most 999
        exact (by norm_num : 989 ≤ 999)
      · -- 989 is the greatest such number within the range
        sorry

end greatest_three_digit_multiple_of_23_l24_24625


namespace intersection_complement_N_l24_24992

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {4, 5, 6}
def C_U_M : Set ℕ := U \ M

theorem intersection_complement_N : (C_U_M ∩ N) = {4, 6} :=
by
  sorry

end intersection_complement_N_l24_24992


namespace number_of_distinct_intersections_l24_24061

/-- The problem is to prove that the number of distinct intersection points
in the xy-plane for the graphs of the given equations is exactly 4. -/
theorem number_of_distinct_intersections :
  ∃ (S : Finset (ℝ × ℝ)), 
  (∀ p : ℝ × ℝ, p ∈ S ↔
    ((p.1 + p.2 = 7 ∨ 2 * p.1 - 3 * p.2 + 1 = 0) ∧
     (p.1 - p.2 - 2 = 0 ∨ 3 * p.1 + 2 * p.2 - 10 = 0))) ∧
  S.card = 4 :=
sorry

end number_of_distinct_intersections_l24_24061


namespace find_a₁_l24_24611

noncomputable def S_3 (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q^2

theorem find_a₁ (S₃_eq : S_3 a₁ q = a₁ + 3 * (a₁ * q)) (a₄_eq : a₁ * q^3 = 8) : a₁ = 1 :=
by
  -- proof skipped
  sorry

end find_a₁_l24_24611


namespace find_loan_amount_l24_24890

-- Define the conditions
def rate_of_interest : ℝ := 0.06
def time_period : ℝ := 6
def interest_paid : ℝ := 432

-- Define the simple interest formula
def simple_interest (P r t : ℝ) : ℝ := P * r * t

-- State the theorem to prove the loan amount
theorem find_loan_amount (P : ℝ) (h1 : rate_of_interest = 0.06) (h2 : time_period = 6) (h3 : interest_paid = 432) (h4 : simple_interest P rate_of_interest time_period = interest_paid) : P = 1200 :=
by
  -- Here should be the proof, but it's omitted for now
  sorry

end find_loan_amount_l24_24890


namespace num_students_taking_music_l24_24783

-- Definitions based on given conditions
def total_students : ℕ := 500
def students_taking_art : ℕ := 20
def students_taking_both_music_and_art : ℕ := 10
def students_taking_neither_music_nor_art : ℕ := 450

-- Theorem statement to prove the number of students taking music
theorem num_students_taking_music :
  ∃ (M : ℕ), M = 40 ∧ 
  (total_students - students_taking_neither_music_nor_art = M + students_taking_art - students_taking_both_music_and_art) := 
by
  sorry

end num_students_taking_music_l24_24783


namespace percent_not_crust_l24_24791

-- Definitions as conditions
def pie_total_weight : ℕ := 200
def crust_weight : ℕ := 50

-- The theorem to be proven
theorem percent_not_crust : (pie_total_weight - crust_weight) / pie_total_weight * 100 = 75 := 
by
  sorry

end percent_not_crust_l24_24791


namespace value_of_expression_l24_24832

theorem value_of_expression (a : ℝ) (h : a^2 + a = 0) : 4*a^2 + 4*a + 2011 = 2011 :=
by
  sorry

end value_of_expression_l24_24832


namespace complement_of_M_in_U_l24_24878

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}

theorem complement_of_M_in_U : U \ M = {2, 3, 5} := by
  sorry

end complement_of_M_in_U_l24_24878


namespace desired_value_l24_24725

noncomputable def find_sum (a b c : ℝ) (p q r : ℝ) : ℝ :=
  a / p + b / q + c / r

theorem desired_value (a b c : ℝ) (h1 : p = a / 2) (h2 : q = b / 2) (h3 : r = c / 2) :
  find_sum a b c p q r = 6 :=
by
  sorry

end desired_value_l24_24725


namespace num_sequences_to_initial_position_8_l24_24945

def validSequenceCount : ℕ := 4900

noncomputable def numberOfSequencesToInitialPosition (n : ℕ) : ℕ :=
if h : n = 8 then validSequenceCount else 0

theorem num_sequences_to_initial_position_8 :
  numberOfSequencesToInitialPosition 8 = 4900 :=
by
  sorry

end num_sequences_to_initial_position_8_l24_24945


namespace find_a13_l24_24207

variable (a_n : ℕ → ℝ)
variable (d : ℝ)
variable (h_arith : ∀ n, a_n (n + 1) = a_n n + d)
variable (h_geo : a_n 9 ^ 2 = a_n 1 * a_n 5)
variable (h_sum : a_n 1 + 3 * a_n 5 + a_n 9 = 20)

theorem find_a13 (h_non_zero_d : d ≠ 0):
  a_n 13 = 28 :=
sorry

end find_a13_l24_24207


namespace digits_problem_solution_l24_24749

def digits_proof_problem (E F G H : ℕ) : Prop :=
  (E, F, G) = (5, 0, 5) → H = 0

theorem digits_problem_solution 
  (E F G H : ℕ)
  (h1 : F + E = E ∨ F + E = E + 10)
  (h2 : E ≠ 0)
  (h3 : E = 5)
  (h4 : 5 + G = H)
  (h5 : 5 - G = 0) :
  H = 0 := 
by {
  sorry -- proof goes here
}

end digits_problem_solution_l24_24749


namespace common_difference_divisible_by_p_l24_24267

variable (a : ℕ → ℕ) (p : ℕ)

-- Define that the sequence a is an arithmetic progression with common difference d
def is_arithmetic_progression (d : ℕ) : Prop :=
  ∀ i : ℕ, a (i + 1) = a i + d

-- Define that the sequence a is strictly increasing
def is_increasing_arithmetic_progression : Prop :=
  ∀ i j : ℕ, i < j → a i < a j

-- Define that all elements a_i are prime numbers
def all_primes : Prop :=
  ∀ i : ℕ, Nat.Prime (a i)

-- Define that the first element of the sequence is greater than p
def first_element_greater_than_p : Prop :=
  a 1 > p

-- Combining all conditions
def conditions (d : ℕ) : Prop :=
  is_arithmetic_progression a d ∧ is_increasing_arithmetic_progression a ∧ all_primes a ∧ first_element_greater_than_p a p ∧ Nat.Prime p

-- Statement to prove: common difference is divisible by p
theorem common_difference_divisible_by_p (d : ℕ) (h : conditions a p d) : p ∣ d :=
sorry

end common_difference_divisible_by_p_l24_24267


namespace probability_f_leq_zero_l24_24688

noncomputable def f (k x : ℝ) : ℝ := k * x - 1

theorem probability_f_leq_zero : 
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) →
  (∀ k ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f k x ≤ 0) →
  (∃ k ∈ Set.Icc (-2 : ℝ) (1 : ℝ), f k x ≤ 0) →
  ((1 - (-2)) / (2 - (-2)) = 3 / 4) :=
by sorry

end probability_f_leq_zero_l24_24688


namespace average_expenditure_Feb_to_July_l24_24045

theorem average_expenditure_Feb_to_July (avg_Jan_to_Jun : ℝ) (spend_Jan : ℝ) (spend_July : ℝ) 
    (total_Jan_to_Jun : avg_Jan_to_Jun = 4200) (spend_Jan_eq : spend_Jan = 1200) (spend_July_eq : spend_July = 1500) :
    (4200 * 6 - 1200 + 1500) / 6 = 4250 :=
by
  sorry

end average_expenditure_Feb_to_July_l24_24045


namespace find_natural_numbers_l24_24195

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem find_natural_numbers (x : ℕ) :
  (x = 36 * sum_of_digits x) ↔ (x = 324 ∨ x = 648) :=
by
  sorry

end find_natural_numbers_l24_24195


namespace sonika_years_in_bank_l24_24897

variable (P A1 A2 : ℚ)
variables (r t : ℚ)

def simple_interest (P r t : ℚ) : ℚ := P * r * t / 100
def amount_with_interest (P r t : ℚ) : ℚ := P + simple_interest P r t

theorem sonika_years_in_bank :
  P = 9000 → A1 = 10200 → A2 = 10740 →
  amount_with_interest P r t = A1 →
  amount_with_interest P (r + 2) t = A2 →
  t = 3 :=
by
  intros hP hA1 hA2 hA1_eq hA2_eq
  sorry

end sonika_years_in_bank_l24_24897


namespace triangle_inequality_l24_24168

theorem triangle_inequality (x : ℕ) (hx : x > 0) :
  (x ≥ 34) ↔ (x + (10 + x) > 24) ∧ (x + 24 > 10 + x) ∧ ((10 + x) + 24 > x) := by
  sorry

end triangle_inequality_l24_24168


namespace proof_part_a_l24_24776

variable {α : Type} [LinearOrder α]

structure ConvexQuadrilateral (α : Type) :=
(a b c d : α)
(a'b'c'd' : α)
(ab_eq_a'b' : α)
(bc_eq_b'c' : α)
(cd_eq_c'd' : α)
(da_eq_d'a' : α)
(angle_A_gt_angle_A' : Prop)
(angle_B_lt_angle_B' : Prop)
(angle_C_gt_angle_C' : Prop)
(angle_D_lt_angle_D' : Prop)

theorem proof_part_a (Quad : ConvexQuadrilateral ℝ) : 
  Quad.angle_A_gt_angle_A' → 
  Quad.angle_B_lt_angle_B' ∧ Quad.angle_C_gt_angle_C' ∧ Quad.angle_D_lt_angle_D' := sorry

end proof_part_a_l24_24776


namespace prob_k_gnomes_fall_exp_gnomes_falling_l24_24712

variables (n k : ℕ) (p : ℝ)
hypotheses 
  (hn : 0 < n)
  (hp : 0 < p) (hp1 : p < 1)
  (hk : 0 ≤ k) (hk1 : k ≤ n)

open ProbabilityTheory
  
def probability_k_gnomes_fall := 
  p * (1 - p) ^ (n - k)

def expected_gnomes_fall :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p

theorem prob_k_gnomes_fall (hprob : 0 < p ∧ p < 1) : 
  ∀ n k : ℕ, 0 ≤ k ∧ k ≤ n → probability_k_gnomes_fall n k p = p * (1 - p) ^ (n - k) :=
by sorry

theorem exp_gnomes_falling (hprob : 0 < p ∧ p < 1) : 
  ∀ n : ℕ, 0 < n → expected_gnomes_fall n p = n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p :=
by sorry

end prob_k_gnomes_fall_exp_gnomes_falling_l24_24712


namespace jasonPears_l24_24720

-- Define the conditions
def keithPears : Nat := 47
def mikePears : Nat := 12
def totalPears : Nat := 105

-- Define the theorem stating the number of pears Jason picked
theorem jasonPears : (totalPears - keithPears - mikePears) = 46 :=
by 
  sorry

end jasonPears_l24_24720


namespace dice_product_composite_probability_l24_24541

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l24_24541


namespace max_number_of_different_ages_l24_24586

theorem max_number_of_different_ages
  (a : ℤ) (s : ℤ)
  (h1 : a = 31)
  (h2 : s = 5) :
  ∃ n : ℕ, n = (36 - 26 + 1) :=
by sorry

end max_number_of_different_ages_l24_24586


namespace equal_partitions_l24_24127

def weights : List ℕ := List.range (81 + 1) |>.map (λ n => n * n)

theorem equal_partitions (h : weights.sum = 178605) :
  ∃ P1 P2 P3 : List ℕ, P1.sum = 59535 ∧ P2.sum = 59535 ∧ P3.sum = 59535 ∧ P1 ++ P2 ++ P3 = weights := sorry

end equal_partitions_l24_24127


namespace goose_eggs_count_l24_24402

theorem goose_eggs_count (E : ℕ)
  (h1 : (2/3 : ℚ) * E ≥ 0)
  (h2 : (3/4 : ℚ) * (2/3 : ℚ) * E ≥ 0)
  (h3 : 100 = (2/5 : ℚ) * (3/4 : ℚ) * (2/3 : ℚ) * E) :
  E = 500 := by
  sorry

end goose_eggs_count_l24_24402


namespace probability_A_given_B_l24_24278

def roll_outcomes : ℕ := 6^3 -- Total number of possible outcomes when rolling three dice

def P_AB : ℚ := 60 / 216 -- Probability of both events A and B happening

def P_B : ℚ := 91 / 216 -- Probability of event B happening

theorem probability_A_given_B : (P_AB / P_B) = (60 / 91) := by
  sorry

end probability_A_given_B_l24_24278


namespace sin_beta_value_l24_24083

theorem sin_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos α = 4 / 5) 
  (h4 : Real.cos (α + β) = 5 / 13) : 
  Real.sin β = 33 / 65 := 
by 
  sorry

end sin_beta_value_l24_24083


namespace max_d_is_9_l24_24828

-- Define the 6-digit number of the form 8d8, 45e
def num (d e : ℕ) : ℕ :=
  800000 + 10000 * d + 800 + 450 + e

-- Define the conditions: the number is a multiple of 45, 0 ≤ d, e ≤ 9
def conditions (d e : ℕ) : Prop :=
  0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
  (num d e) % 45 = 0

-- Define the maximum value of d
noncomputable def max_d : ℕ :=
  9

-- The theorem statement to be proved
theorem max_d_is_9 :
  ∀ (d e : ℕ), conditions d e → d ≤ max_d :=
by
  sorry

end max_d_is_9_l24_24828


namespace area_of_square_l24_24264

theorem area_of_square (r s l : ℕ) (h1 : l = (2 * r) / 5) (h2 : r = s) (h3 : l * 10 = 240) : s * s = 3600 :=
by
  sorry

end area_of_square_l24_24264


namespace amount_B_l24_24640

noncomputable def A : ℝ := sorry -- Definition of A
noncomputable def B : ℝ := sorry -- Definition of B

-- Conditions
def condition1 : Prop := A + B = 100
def condition2 : Prop := (3 / 10) * A = (1 / 5) * B

-- Statement to prove
theorem amount_B : condition1 ∧ condition2 → B = 60 :=
by
  intros
  sorry

end amount_B_l24_24640


namespace find_N_product_l24_24661

variables (M L : ℤ) (N : ℤ)

theorem find_N_product
  (h1 : M = L + N)
  (h2 : M + 3 = (L + N + 3))
  (h3 : L - 5 = L - 5)
  (h4 : |(L + N + 3) - (L - 5)| = 4) :
  N = -4 ∨ N = -12 → (-4 * -12) = 48 :=
by sorry

end find_N_product_l24_24661


namespace probability_white_given_popped_is_7_over_12_l24_24647

noncomputable def probability_white_given_popped : ℚ :=
  let P_W := 0.4
  let P_Y := 0.4
  let P_R := 0.2
  let P_popped_given_W := 0.7
  let P_popped_given_Y := 0.5
  let P_popped_given_R := 0
  let P_popped := P_popped_given_W * P_W + P_popped_given_Y * P_Y + P_popped_given_R * P_R
  (P_popped_given_W * P_W) / P_popped

theorem probability_white_given_popped_is_7_over_12 : probability_white_given_popped = 7 / 12 := 
  by
    sorry

end probability_white_given_popped_is_7_over_12_l24_24647


namespace binomial_12_6_eq_924_l24_24803

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_l24_24803


namespace dice_product_composite_probability_l24_24547

theorem dice_product_composite_probability :
  (let total_outcomes := (6 : ℕ)^4,
       non_composite_outcomes := 1,
       composite_probability := 1 - (non_composite_outcomes / total_outcomes) in
   composite_probability = 1295 / 1296) :=
by sorry

end dice_product_composite_probability_l24_24547


namespace weight_of_b_l24_24033

theorem weight_of_b (a b c : ℝ) (h1 : (a + b + c) / 3 = 45) (h2 : (a + b) / 2 = 40) (h3 : (b + c) / 2 = 43) : b = 31 :=
by
  sorry

end weight_of_b_l24_24033


namespace greatest_odd_integer_x_l24_24619

theorem greatest_odd_integer_x (x : ℕ) (h1 : x % 2 = 1) (h2 : x^4 / x^2 < 50) : x ≤ 7 :=
sorry

end greatest_odd_integer_x_l24_24619


namespace exists_positive_integer_m_l24_24403

theorem exists_positive_integer_m (m : ℕ) (h_positive : m > 0) : 
  ∃ (m : ℕ), m > 0 ∧ ∃ k : ℕ, 8 * m = k^2 := 
sorry

end exists_positive_integer_m_l24_24403


namespace binomial_12_6_eq_924_l24_24813

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := sorry

end binomial_12_6_eq_924_l24_24813


namespace sum_last_two_digits_pow_mod_eq_zero_l24_24631

/-
Given condition: 
Sum of the last two digits of \( 9^{25} + 11^{25} \)
-/
theorem sum_last_two_digits_pow_mod_eq_zero : 
  let a := 9
  let b := 11
  let n := 25 
  (a ^ n + b ^ n) % 100 = 0 :=
by
  sorry

end sum_last_two_digits_pow_mod_eq_zero_l24_24631


namespace y_square_range_l24_24528

theorem y_square_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 16) ^ (1/3) = 4) : 
  230 ≤ y^2 ∧ y^2 < 240 :=
sorry

end y_square_range_l24_24528


namespace form_triangle_condition_right_angled_triangle_condition_l24_24849

def vector (α : Type*) := α × α
noncomputable def oa : vector ℝ := ⟨2, -1⟩
noncomputable def ob : vector ℝ := ⟨3, 2⟩
noncomputable def oc (m : ℝ) : vector ℝ := ⟨m, 2 * m + 1⟩

def vector_sub (v1 v2 : vector ℝ) : vector ℝ := ⟨v1.1 - v2.1, v1.2 - v2.2⟩
def vector_dot (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem form_triangle_condition (m : ℝ) : 
  ¬ ((vector_sub ob oa).1 * (vector_sub (oc m) oa).2 = (vector_sub ob oa).2 * (vector_sub (oc m) oa).1) ↔ m ≠ 8 :=
sorry

theorem right_angled_triangle_condition (m : ℝ) : 
  (vector_dot (vector_sub ob oa) (vector_sub (oc m) oa) = 0 ∨ 
   vector_dot (vector_sub ob oa) (vector_sub (oc m) ob) = 0 ∨ 
   vector_dot (vector_sub (oc m) oa) (vector_sub (oc m) ob) = 0) ↔ 
  (m = -4/7 ∨ m = 6/7) :=
sorry

end form_triangle_condition_right_angled_triangle_condition_l24_24849


namespace factorization_correct_l24_24774

theorem factorization_correct (x y : ℝ) : 
  x * (x - y) - y * (x - y) = (x - y) ^ 2 :=
by 
  sorry

end factorization_correct_l24_24774


namespace proposition_P_l24_24984

theorem proposition_P (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) := 
by 
  sorry

end proposition_P_l24_24984


namespace region_area_l24_24824

theorem region_area (x y : ℝ) : 
  (|2 * x - 16| + |3 * y + 9| ≤ 6) → ∃ A, A = 72 :=
sorry

end region_area_l24_24824


namespace initial_black_pieces_is_118_l24_24128

open Nat

-- Define the initial conditions and variables
variables (b w n : ℕ)

-- Hypotheses based on the conditions
axiom h1 : b = 2 * w
axiom h2 : w - 2 * n = 1
axiom h3 : b - 3 * n = 31

-- Goal to prove the initial number of black pieces were 118
theorem initial_black_pieces_is_118 : b = 118 :=
by 
  -- We only state the theorem, proof will be added as sorry
  sorry

end initial_black_pieces_is_118_l24_24128


namespace log2_ratio_squared_l24_24886

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log2_ratio_squared :
  ∀ (x y : ℝ), x ≠ 1 → y ≠ 1 → log_base 2 x = log_base y 25 → x * y = 81
  → (log_base 2 (x / y))^2 = 5.11 :=
by
  intros x y hx hy hlog hxy
  sorry

end log2_ratio_squared_l24_24886


namespace cos_7theta_l24_24854

theorem cos_7theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (7 * θ) = 49 / 2187 := 
  sorry

end cos_7theta_l24_24854


namespace calculate_principal_amount_l24_24034

theorem calculate_principal_amount (P : ℝ) (h1 : P * 0.1025 - P * 0.1 = 25) : 
  P = 10000 :=
by
  sorry

end calculate_principal_amount_l24_24034


namespace certain_amount_eq_3_l24_24773

theorem certain_amount_eq_3 (x A : ℕ) (hA : A = 5) (h : A + (11 + x) = 19) : x = 3 :=
by
  sorry

end certain_amount_eq_3_l24_24773


namespace cos_beta_value_cos_2alpha_plus_beta_value_l24_24200

-- Definitions of the conditions
variables (α β : ℝ)
variable (condition1 : 0 < α ∧ α < π / 2)
variable (condition2 : π / 2 < β ∧ β < π)
variable (condition3 : Real.cos (α + π / 4) = 1 / 3)
variable (condition4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3)

-- Proof problem (1)
theorem cos_beta_value :
  ∀ α β, (0 < α ∧ α < π / 2) →
  (π / 2 < β ∧ β < π) →
  (Real.cos (α + π / 4) = 1 / 3) →
  (Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) →
  Real.cos β = - 4 * Real.sqrt 2 / 9 :=
by
  intros α β condition1 condition2 condition3 condition4
  sorry

-- Proof problem (2)
theorem cos_2alpha_plus_beta_value :
  ∀ α β, (0 < α ∧ α < π / 2) →
  (π / 2 < β ∧ β < π) →
  (Real.cos (α + π / 4) = 1 / 3) →
  (Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) →
  Real.cos (2 * α + β) = -1 :=
by
  intros α β condition1 condition2 condition3 condition4
  sorry

end cos_beta_value_cos_2alpha_plus_beta_value_l24_24200


namespace max_sum_arith_seq_l24_24075

theorem max_sum_arith_seq :
  let a1 := 29
  let d := 2
  let a_n (n : ℕ) := a1 + (n - 1) * d
  let S_n (n : ℕ) := n / 2 * (a1 + a_n n)
  S_n 10 = S_n 20 → S_n 20 = 960 := by
sorry

end max_sum_arith_seq_l24_24075


namespace determine_n_l24_24959

theorem determine_n 
    (n : ℕ) (h2 : n ≥ 2) 
    (a : ℕ) (ha_div_n : a ∣ n) 
    (ha_min : ∀ d : ℕ, d ∣ n → d > 1 → d ≥ a) 
    (b : ℕ) (hb_div_n : b ∣ n)
    (h_eq : n = a^2 + b^2) : 
    n = 8 ∨ n = 20 :=
sorry

end determine_n_l24_24959


namespace distance_interval_l24_24318

-- Define the conditions based on the false statements:
variable (d : ℝ)

def false_by_alice : Prop := d < 8
def false_by_bob : Prop := d > 7
def false_by_charlie : Prop := d ≠ 6

theorem distance_interval (h_alice : false_by_alice d) (h_bob : false_by_bob d) (h_charlie : false_by_charlie d) :
  7 < d ∧ d < 8 :=
by
  sorry

end distance_interval_l24_24318


namespace inequality_A_if_ab_pos_inequality_D_if_ab_pos_l24_24361

variable (a b : ℝ)

theorem inequality_A_if_ab_pos (h : a * b > 0) : a^2 + b^2 ≥ 2 * a * b := 
sorry

theorem inequality_D_if_ab_pos (h : a * b > 0) : (b / a) + (a / b) ≥ 2 :=
sorry

end inequality_A_if_ab_pos_inequality_D_if_ab_pos_l24_24361


namespace largest_multiple_l24_24628

theorem largest_multiple (n : ℤ) (h8 : 8 ∣ n) (h : -n > -80) : n = 72 :=
by 
  sorry

end largest_multiple_l24_24628


namespace sum_of_n_and_k_l24_24110

open Nat

theorem sum_of_n_and_k (n k : ℕ) (h1 : (n.choose (k + 1)) = 3 * (n.choose k))
                      (h2 : (n.choose (k + 2)) = 2 * (n.choose (k + 1))) :
    n + k = 7 := by
  sorry

end sum_of_n_and_k_l24_24110


namespace range_of_a_l24_24836

variable (a : ℝ)
def p : Prop := a > 1/4
def q : Prop := a ≤ -1 ∨ a ≥ 1

theorem range_of_a :
  ((p a ∧ ¬ (q a)) ∨ (q a ∧ ¬ (p a))) ↔ (a > 1/4 ∧ a < 1) ∨ (a ≤ -1) :=
by
  sorry

end range_of_a_l24_24836


namespace probability_one_pair_same_color_l24_24510

theorem probability_one_pair_same_color :
  let n := 8
  let r := 4
  let total_combinations := Nat.choose n r
  let choose_colors := Nat.choose 4 3
  let choose_pair_color := Nat.choose 3 1
  let favorable_ways := choose_colors * choose_pair_color * 2 * 2
  let probability := (favorable_ways : ℚ) / (total_combinations : ℚ)
  probability = 24 / 35 :=
by
  sorry

end probability_one_pair_same_color_l24_24510


namespace compute_binom_12_6_eq_1848_l24_24821

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end compute_binom_12_6_eq_1848_l24_24821


namespace bc_sum_eq_twelve_l24_24084

theorem bc_sum_eq_twelve (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hb_lt : b < 12) (hc_lt : c < 12) 
  (h_eq : (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) : b + c = 12 :=
by
  sorry

end bc_sum_eq_twelve_l24_24084


namespace max_abs_z_2_2i_l24_24935

open Complex

theorem max_abs_z_2_2i (z : ℂ) (h : abs (z + 2 - 2 * I) = 1) : 
  ∃ w : ℂ, abs (w - 2 - 2 * I) = 5 :=
sorry

end max_abs_z_2_2i_l24_24935


namespace find_primes_l24_24488

theorem find_primes (p : ℕ) (x y : ℕ) (hx : x > 0) (hy : y > 0) (hp : Nat.Prime p) : 
  (x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ (p = 2 ∨ p = 3 ∨ p = 7) := sorry

end find_primes_l24_24488


namespace Vanya_two_digit_number_l24_24285

-- Define the conditions as a mathematical property
theorem Vanya_two_digit_number:
  ∃ (m n : ℕ), 0 ≤ m ∧ m ≤ 9 ∧ 0 ≤ n ∧ n ≤ 9 ∧ (10 * n + m) ^ 2 = 4 * (10 * m + n) ∧ (10 * m + n) = 81 :=
by
  -- Remember to replace the proof with 'sorry'
  sorry

end Vanya_two_digit_number_l24_24285


namespace birds_total_distance_l24_24047

-- Define the speeds of the birds
def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30

-- Define the flying time for each bird
def flying_time : ℕ := 2

-- Calculate the total distance flown by all birds
def total_distance_flown : ℕ := (eagle_speed * flying_time) +
                                 (falcon_speed * flying_time) +
                                 (pelican_speed * flying_time) +
                                 (hummingbird_speed * flying_time)

-- The goal is to prove that the total distance flown by all birds is 248 miles
theorem birds_total_distance : total_distance_flown = 248 := by
  -- Proof here
  sorry

end birds_total_distance_l24_24047


namespace al_original_amount_l24_24317

theorem al_original_amount : 
  ∃ (a b c : ℝ), 
    a + b + c = 1200 ∧ 
    (a - 200 + 3 * b + 4 * c) = 1800 ∧ 
    b = 2800 - 3 * a ∧ 
    c = 1200 - a - b ∧ 
    a = 860 := by
  sorry

end al_original_amount_l24_24317


namespace area_enclosed_by_graph_l24_24767

theorem area_enclosed_by_graph : 
  ∃ A : ℝ, (∀ x y : ℝ, |x| + |3 * y| = 9 ↔ (x = 9 ∨ x = -9 ∨ y = 3 ∨ y = -3)) → A = 54 :=
by
  sorry

end area_enclosed_by_graph_l24_24767


namespace anne_wandered_hours_l24_24048

noncomputable def speed : ℝ := 2 -- miles per hour
noncomputable def distance : ℝ := 6 -- miles

theorem anne_wandered_hours (t : ℝ) (h : distance = speed * t) : t = 3 := by
  sorry

end anne_wandered_hours_l24_24048


namespace excluded_numbers_range_l24_24259

theorem excluded_numbers_range (S S' E : ℕ) (h1 : S = 31 * 10) (h2 : S' = 28 * 8) (h3 : E = S - S') (h4 : E > 70) :
  ∀ (x y : ℕ), x + y = E → 1 ≤ x ∧ x ≤ 85 ∧ 1 ≤ y ∧ y ≤ 85 := by
  sorry

end excluded_numbers_range_l24_24259


namespace no_conclusions_deducible_l24_24214

open Set

variable {U : Type}  -- Universe of discourse

-- Conditions
variables (Bars Fins Grips : Set U)

def some_bars_are_not_fins := ∃ x, x ∈ Bars ∧ x ∉ Fins
def no_fins_are_grips := ∀ x, x ∈ Fins → x ∉ Grips

-- Lean statement
theorem no_conclusions_deducible 
  (h1 : some_bars_are_not_fins Bars Fins)
  (h2 : no_fins_are_grips Fins Grips) :
  ¬((∃ x, x ∈ Bars ∧ x ∉ Grips) ∨
    (∃ x, x ∈ Grips ∧ x ∉ Bars) ∨
    (∀ x, x ∈ Bars → x ∉ Grips) ∨
    (∃ x, x ∈ Bars ∧ x ∈ Grips)) :=
sorry

end no_conclusions_deducible_l24_24214


namespace lambs_goats_solution_l24_24031

theorem lambs_goats_solution : ∃ l g : ℕ, l > 0 ∧ g > 0 ∧ 30 * l + 32 * g = 1200 ∧ l = 24 ∧ g = 15 :=
by
  existsi 24
  existsi 15
  repeat { split }
  sorry

end lambs_goats_solution_l24_24031


namespace negation_of_existential_l24_24751

theorem negation_of_existential :
  (¬ (∃ x : ℝ, x^2 - x - 1 > 0)) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
sorry

end negation_of_existential_l24_24751


namespace simplify_and_evaluate_l24_24583

noncomputable def x := Real.tan (Real.pi / 4) + Real.cos (Real.pi / 6)

theorem simplify_and_evaluate :
  ((x / (x ^ 2 - 1)) * ((x - 1) / x - 2)) = - (2 * Real.sqrt 3) / 3 := 
sorry

end simplify_and_evaluate_l24_24583


namespace find_p_current_age_l24_24298

theorem find_p_current_age (x p q : ℕ) (h1 : p - 3 = 4 * x) (h2 : q - 3 = 3 * x) (h3 : (p + 6) / (q + 6) = 7 / 6) : p = 15 := 
sorry

end find_p_current_age_l24_24298


namespace sum_of_coefficients_l24_24982

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) :
  (1 - 2 * x)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + 
                  a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 = -1 :=
sorry

end sum_of_coefficients_l24_24982


namespace christine_savings_l24_24479

def commission_rate : ℝ := 0.12
def total_sales : ℝ := 24000
def personal_needs_percentage : ℝ := 0.60
def savings_percentage : ℝ := 1 - personal_needs_percentage

noncomputable def commission_earned : ℝ := total_sales * commission_rate
noncomputable def amount_saved : ℝ := commission_earned * savings_percentage

theorem christine_savings :
  amount_saved = 1152 :=
by
  sorry

end christine_savings_l24_24479


namespace possible_values_of_y_l24_24387

theorem possible_values_of_y (x : ℝ) (hx : x^2 + 5 * (x / (x - 3)) ^ 2 = 50) :
  ∃ (y : ℝ), y = (x - 3)^2 * (x + 4) / (3 * x - 4) ∧ (y = 0 ∨ y = 15 ∨ y = 49) :=
sorry

end possible_values_of_y_l24_24387


namespace find_m_of_quadratic_function_l24_24009

theorem find_m_of_quadratic_function :
  ∀ (m : ℝ), (m + 1 ≠ 0) → ((m + 1) * x ^ (m^2 + 1) + 5 = a * x^2 + b * x + c) → m = 1 :=
by
  intro m h h_quad
  -- Proof Here
  sorry

end find_m_of_quadratic_function_l24_24009


namespace total_water_in_heaters_l24_24919

theorem total_water_in_heaters (wallace_capacity : ℕ) (catherine_capacity : ℕ) 
(wallace_water : ℕ) (catherine_water : ℕ) :
  wallace_capacity = 40 →
  (wallace_water = (3 * wallace_capacity) / 4) →
  wallace_capacity = 2 * catherine_capacity →
  (catherine_water = (3 * catherine_capacity) / 4) →
  wallace_water + catherine_water = 45 :=
by
  sorry

end total_water_in_heaters_l24_24919


namespace trapezoid_height_l24_24555

theorem trapezoid_height (a b : ℝ) (A : ℝ) (h : ℝ) : a = 5 → b = 9 → A = 56 → A = (1 / 2) * (a + b) * h → h = 8 :=
by 
  intros ha hb hA eqn
  sorry

end trapezoid_height_l24_24555


namespace lottery_probability_correct_l24_24115

def number_of_winnerballs_ways : ℕ := Nat.choose 50 6

def probability_megaBall : ℚ := 1 / 30

def probability_winnerBalls : ℚ := 1 / number_of_winnerballs_ways

def combined_probability : ℚ := probability_megaBall * probability_winnerBalls

theorem lottery_probability_correct : combined_probability = 1 / 476721000 := by
  sorry

end lottery_probability_correct_l24_24115


namespace abraham_initial_budget_l24_24950

-- Definitions based on conditions
def shower_gel_price := 4
def shower_gel_quantity := 4
def toothpaste_price := 3
def laundry_detergent_price := 11
def remaining_budget := 30

-- Calculations based on the conditions
def spent_on_shower_gels := shower_gel_quantity * shower_gel_price
def spent_on_toothpaste := toothpaste_price
def spent_on_laundry_detergent := laundry_detergent_price
def total_spent := spent_on_shower_gels + spent_on_toothpaste + spent_on_laundry_detergent

-- The theorem to prove
theorem abraham_initial_budget :
  (total_spent + remaining_budget) = 60 :=
by
  sorry

end abraham_initial_budget_l24_24950


namespace compute_binom_12_6_eq_1848_l24_24819

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end compute_binom_12_6_eq_1848_l24_24819


namespace extra_food_needed_l24_24030

theorem extra_food_needed (f1 f2 : ℝ) (h1 : f1 = 0.5) (h2 : f2 = 0.9) :
  f2 - f1 = 0.4 :=
by sorry

end extra_food_needed_l24_24030


namespace trey_nail_usage_l24_24437

theorem trey_nail_usage (total_decorations nails thumbtacks sticky_strips : ℕ) 
  (h1 : nails = 2 * total_decorations / 3)
  (h2 : sticky_strips = 15)
  (h3 : sticky_strips = 3 * (total_decorations - 2 * total_decorations / 3) / 5) :
  nails = 50 :=
by
  sorry

end trey_nail_usage_l24_24437


namespace prank_helpers_combinations_l24_24129

theorem prank_helpers_combinations :
  let Monday := 1
  let Tuesday := 2
  let Wednesday := 3
  let Thursday := 4
  let Friday := 1
  (Monday * Tuesday * Wednesday * Thursday * Friday = 24) :=
by
  intros
  sorry

end prank_helpers_combinations_l24_24129


namespace solution_set_inequality_l24_24985

theorem solution_set_inequality (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  {x : ℝ | (x - a) * (x - (1 / a)) < 0} = {x : ℝ | a < x ∧ x < 1 / a} := sorry

end solution_set_inequality_l24_24985


namespace exists_positive_integer_m_l24_24404

theorem exists_positive_integer_m (m : ℕ) (hm : m > 0) : ∃ m : ℕ, m > 0 ∧ ∃ k : ℕ, 8 * m = k ^ 2 := 
by {
  let m := 2
  use m,
  dsimp,
  split,
  { exact hm },
  { use 4,
    calc 8 * m = 8 * 2 : by rfl
           ... = 16 : by norm_num
           ... = 4 ^ 2 : by norm_num }
}

end exists_positive_integer_m_l24_24404


namespace find_percentage_l24_24085

theorem find_percentage (x p : ℝ) (h1 : 0.25 * x = p * 10 - 30) (h2 : x = 680) : p = 20 := 
sorry

end find_percentage_l24_24085


namespace tetrahedron_condition_proof_l24_24094

/-- Define the conditions for the necessary and sufficient condition for each k -/
def tetrahedron_condition (a : ℝ) (k : ℕ) : Prop :=
  match k with
  | 1 => a < Real.sqrt 3
  | 2 => Real.sqrt (2 - Real.sqrt 3) < a ∧ a < Real.sqrt (2 + Real.sqrt 3)
  | 3 => a < Real.sqrt 3
  | 4 => a > Real.sqrt (2 - Real.sqrt 3)
  | 5 => a > 1 / Real.sqrt 3
  | _ => False -- not applicable for other values of k

/-- Prove that the condition is valid for given a and k -/
theorem tetrahedron_condition_proof (a : ℝ) (k : ℕ) : tetrahedron_condition a k := 
  by
  sorry

end tetrahedron_condition_proof_l24_24094


namespace inequality_solution_l24_24910

theorem inequality_solution (x : ℝ) (h₁ : 1 - x < 0) (h₂ : x - 3 ≤ 0) : 1 < x ∧ x ≤ 3 :=
by
  sorry

end inequality_solution_l24_24910


namespace remaining_black_cards_l24_24373

def total_black_cards_per_deck : ℕ := 26
def num_decks : ℕ := 5
def removed_black_face_cards : ℕ := 7
def removed_black_number_cards : ℕ := 12

theorem remaining_black_cards : total_black_cards_per_deck * num_decks - (removed_black_face_cards + removed_black_number_cards) = 111 :=
by
  -- proof will go here
  sorry

end remaining_black_cards_l24_24373


namespace initial_legos_500_l24_24719

-- Definitions and conditions from the problem
def initial_legos (x : ℕ) : Prop :=
  let used_pieces := x / 2
  let remaining_pieces := x - used_pieces
  let boxed_pieces := remaining_pieces - 5
  boxed_pieces = 245

-- Statement to be proven
theorem initial_legos_500 : initial_legos 500 :=
by
  -- Proof goes here
  sorry

end initial_legos_500_l24_24719


namespace sum_of_two_rel_prime_numbers_l24_24972

theorem sum_of_two_rel_prime_numbers (k : ℕ) : 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ Nat.gcd a b = 1 ∧ k = a + b) ↔ (k = 5 ∨ k ≥ 7) := sorry

end sum_of_two_rel_prime_numbers_l24_24972


namespace sum_tan_fourth_power_l24_24239

open Real

theorem sum_tan_fourth_power (S : Set ℝ) (hS : ∀ x ∈ S, 0 < x ∧ x < π/2 ∧ 
  (∃ a b c, a^2 + b^2 = c^2 ∧ (a, b, c) ∈ {(sin x)^2, (cos x)^2, (tan x)^2} × {(sin x)^2, (cos x)^2, (tan x)^2} × {(sin x)^2, (cos x)^2, (tan x)^2})) :
  ∑ x in S, (tan x)^4 = 4 - 2 * sqrt 2 :=
sorry

end sum_tan_fourth_power_l24_24239


namespace sum_of_first_10_common_elements_eq_13981000_l24_24977

def arithmetic_prog (n : ℕ) : ℕ := 4 + 3 * n
def geometric_prog (k : ℕ) : ℕ := 20 * 2 ^ k

theorem sum_of_first_10_common_elements_eq_13981000 :
  let common_elements : List ℕ := 
    [40, 160, 640, 2560, 10240, 40960, 163840, 655360, 2621440, 10485760]
  let sum_common_elements : ℕ := common_elements.sum
  sum_common_elements = 13981000 := by
  sorry

end sum_of_first_10_common_elements_eq_13981000_l24_24977


namespace simplify_expr_C_l24_24106

theorem simplify_expr_C (x y : ℝ) : 5 * x - (x - 2 * y) = 4 * x + 2 * y :=
by
  sorry

end simplify_expr_C_l24_24106


namespace ratio_children_to_adults_l24_24951

variable (male_adults : ℕ) (female_adults : ℕ) (total_people : ℕ)
variable (total_adults : ℕ) (children : ℕ)

theorem ratio_children_to_adults :
  male_adults = 100 →
  female_adults = male_adults + 50 →
  total_people = 750 →
  total_adults = male_adults + female_adults →
  children = total_people - total_adults →
  children / total_adults = 2 :=
by
  intros h_male h_female h_total h_adults h_children
  sorry

end ratio_children_to_adults_l24_24951


namespace brianne_savings_ratio_l24_24473

theorem brianne_savings_ratio
  (r : ℝ)
  (H1 : 10 * r^4 = 160) :
  r = 2 :=
by 
  sorry

end brianne_savings_ratio_l24_24473


namespace middle_digit_zero_l24_24788

theorem middle_digit_zero (a b c M : ℕ) (h1 : M = 36 * a + 6 * b + c) (h2 : M = 64 * a + 8 * b + c) (ha : 0 ≤ a ∧ a < 6) (hb : 0 ≤ b ∧ b < 6) (hc : 0 ≤ c ∧ c < 6) : 
  b = 0 := 
  by sorry

end middle_digit_zero_l24_24788


namespace min_points_game_12_l24_24228

noncomputable def player_scores := (18, 22, 9, 29)

def avg_after_eleven_games (scores: ℕ × ℕ × ℕ × ℕ) := 
  let s₁ := 78 -- Sum of the points in 8th, 9th, 10th, 11th games
  (s₁: ℕ) / 4

def points_twelve_game_cond (n: ℕ) : Prop :=
  let total_points := 78 + n
  total_points > (20 * 12)

theorem min_points_game_12 (points_in_first_7_games: ℕ) (score_12th_game: ℕ) 
  (H1: avg_after_eleven_games player_scores > (points_in_first_7_games / 7)) 
  (H2: points_twelve_game_cond score_12th_game):
  score_12th_game = 30 := by
  sorry

end min_points_game_12_l24_24228


namespace percentage_of_number_l24_24884

theorem percentage_of_number (n : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * n = 16) : 0.4 * n = 192 :=
by 
  sorry

end percentage_of_number_l24_24884


namespace question1_question2_l24_24081

-- Define the sets A and B as given in the problem
def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}
def B : Set ℝ := {x | x < -2 ∨ x > 5}

-- Lean statement for (1)
theorem question1 (m : ℝ) : 
  (A m ⊆ B) ↔ (m < 2 ∨ m > 4) :=
by
  sorry

-- Lean statement for (2)
theorem question2 (m : ℝ) : 
  (A m ∩ B = ∅) ↔ (m ≤ 3) :=
by
  sorry

end question1_question2_l24_24081


namespace PetrovFamilySavings_l24_24005

def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500

def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

def total_income : ℕ := parents_salary + grandmothers_pension + sons_scholarship
def total_expenses : ℕ := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses

def surplus : ℕ := total_income - total_expenses
def deposit : ℕ := surplus / 10

def amount_set_aside : ℕ := surplus - deposit

theorem PetrovFamilySavings : amount_set_aside = 16740 := by
  sorry

end PetrovFamilySavings_l24_24005


namespace solve_10_arithmetic_in_1_minute_l24_24287

-- Define the times required for each task
def time_math_class : Nat := 40 -- in minutes
def time_walk_kilometer : Nat := 20 -- in minutes
def time_solve_arithmetic : Nat := 1 -- in minutes

-- The question: Which task can be completed in 1 minute?
def task_completed_in_1_minute : Nat := 1

theorem solve_10_arithmetic_in_1_minute :
  time_solve_arithmetic = task_completed_in_1_minute :=
by
  sorry

end solve_10_arithmetic_in_1_minute_l24_24287


namespace geometric_sequence_identity_l24_24082

variables {b : ℕ → ℝ} {m n p : ℕ}

def is_geometric_sequence (b : ℕ → ℝ) :=
  ∀ i j k : ℕ, i < j → j < k → b j^2 = b i * b k

noncomputable def distinct_pos_ints (m n p : ℕ) :=
  0 < m ∧ 0 < n ∧ 0 < p ∧ m ≠ n ∧ n ≠ p ∧ p ≠ m

theorem geometric_sequence_identity 
  (h_geom : is_geometric_sequence b) 
  (h_distinct : distinct_pos_ints m n p) : 
  b p ^ (m - n) * b m ^ (n - p) * b n ^ (p - m) = 1 :=
sorry

end geometric_sequence_identity_l24_24082


namespace unique_two_digit_solution_l24_24914

theorem unique_two_digit_solution :
  ∃! (u : ℕ), 9 < u ∧ u < 100 ∧ 13 * u % 100 = 52 := 
sorry

end unique_two_digit_solution_l24_24914


namespace fraction_done_by_B_l24_24529

theorem fraction_done_by_B {A B : ℝ} (h : A = (2/5) * B) : (B / (A + B)) = (5/7) :=
by
  sorry

end fraction_done_by_B_l24_24529


namespace arithmetic_geometric_value_l24_24683

-- Definitions and annotations
variables {a1 a2 b1 b2 : ℝ}
variable {d : ℝ} -- common difference for the arithmetic sequence
variable {q : ℝ} -- common ratio for the geometric sequence

-- Assuming input values for the initial elements of the sequences
axiom h1 : -9 = -9
axiom h2 : -9 + 3 * d = -1
axiom h3 : b1 = -9 * q
axiom h4 : b2 = -9 * q^2

-- The desired equality to prove
theorem arithmetic_geometric_value :
  b2 * (a2 - a1) = -8 :=
sorry

end arithmetic_geometric_value_l24_24683


namespace steve_fraction_of_skylar_l24_24255

variables (S : ℤ) (Stacy Skylar Steve : ℤ)

-- Given conditions
axiom h1 : 32 = 3 * Steve + 2 -- Stacy's berries = 2 + 3 * Steve's berries
axiom h2 : Skylar = 20        -- Skylar has 20 berries
axiom h3 : Stacy = 32         -- Stacy has 32 berries

-- Final goal
theorem steve_fraction_of_skylar (h1: 32 = 3 * Steve + 2) (h2: 20 = Skylar) (h3: Stacy = 32) :
  Steve = Skylar / 2 := 
sorry

end steve_fraction_of_skylar_l24_24255


namespace diagonals_in_nine_sided_polygon_l24_24993

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l24_24993


namespace number_of_unit_fraction_pairs_l24_24902

/-- 
 The number of ways that 1/2007 can be expressed as the sum of two distinct positive unit fractions is 7.
-/
theorem number_of_unit_fraction_pairs : 
  ∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 ≠ p.2 ∧ (1 : ℚ) / 2007 = 1 / ↑p.1 + 1 / ↑p.2) ∧ 
    pairs.card = 7 :=
sorry

end number_of_unit_fraction_pairs_l24_24902


namespace frogs_meet_time_proven_l24_24908

-- Define the problem
def frogs_will_meet_at_time : Prop :=
  ∃ (meet_time : Nat),
    let initial_time := 12 * 60 -- 12:00 PM in minutes
    let initial_distance := 2015
    let green_frog_jump := 9
    let blue_frog_jump := 8 
    let combined_reduction := green_frog_jump + blue_frog_jump
    initial_distance % combined_reduction = 0 ∧
    meet_time == initial_time + (2 * (initial_distance / combined_reduction))

theorem frogs_meet_time_proven (h : frogs_will_meet_at_time) : meet_time = 15 * 60 + 56 :=
sorry

end frogs_meet_time_proven_l24_24908


namespace total_area_covered_is_60_l24_24615

-- Declare the dimensions of the strips
def length_strip : ℕ := 12
def width_strip : ℕ := 2
def num_strips : ℕ := 3

-- Define the total area covered without overlaps
def total_area_no_overlap := num_strips * (length_strip * width_strip)

-- Define the area of overlap for each pair of strips
def overlap_area_per_pair := width_strip * width_strip

-- Define the total overlap area given 3 pairs
def total_overlap_area := 3 * overlap_area_per_pair

-- Define the actual total covered area
def total_covered_area := total_area_no_overlap - total_overlap_area

-- Prove that the total covered area is 60 square units
theorem total_area_covered_is_60 : total_covered_area = 60 := by 
  sorry

end total_area_covered_is_60_l24_24615


namespace exponential_comparison_l24_24676

theorem exponential_comparison (a b c : ℝ) (h₁ : a = 0.5^((1:ℝ)/2))
                                          (h₂ : b = 0.5^((1:ℝ)/3))
                                          (h₃ : c = 0.5^((1:ℝ)/4)) : 
  a < b ∧ b < c := by
  sorry

end exponential_comparison_l24_24676


namespace number_of_real_solutions_l24_24516

theorem number_of_real_solutions : 
  set.countable {x : ℝ | ∃ n : ℕ, (sqrt (123 - sqrt x) = n)} = 12 :=
by
  sorry

end number_of_real_solutions_l24_24516


namespace fraction_reduction_by_11_l24_24411

theorem fraction_reduction_by_11 (k : ℕ) :
  (k^2 - 5 * k + 8) % 11 = 0 → 
  (k^2 + 6 * k + 19) % 11 = 0 :=
by
  sorry

end fraction_reduction_by_11_l24_24411


namespace balance_balls_l24_24883

-- Define the weights of the balls as variables
variables (B R O S : ℝ)

-- Given conditions
axiom h1 : R = 2 * B
axiom h2 : O = (7 / 3) * B
axiom h3 : S = (5 / 3) * B

-- Statement to prove
theorem balance_balls :
  (5 * R + 3 * O + 4 * S) = (71 / 3) * B :=
by {
  -- The proof is omitted
  sorry
}

end balance_balls_l24_24883


namespace eggs_per_chicken_per_week_l24_24235

-- Define the conditions
def chickens : ℕ := 10
def price_per_dozen : ℕ := 2  -- in dollars
def earnings_in_2_weeks : ℕ := 20  -- in dollars
def weeks : ℕ := 2
def eggs_per_dozen : ℕ := 12

-- Define the question as a theorem to be proved
theorem eggs_per_chicken_per_week : 
  (earnings_in_2_weeks / price_per_dozen) * eggs_per_dozen / (chickens * weeks) = 6 :=
by
  -- proof steps
  sorry

end eggs_per_chicken_per_week_l24_24235


namespace unit_digit_calc_l24_24671

theorem unit_digit_calc : (8 * 19 * 1981 - 8^3) % 10 = 0 := by
  sorry

end unit_digit_calc_l24_24671


namespace fraction_sum_l24_24475

theorem fraction_sum : (3 / 4 : ℚ) + (6 / 9 : ℚ) = 17 / 12 := 
by 
  -- Sorry placeholder to indicate proof is not provided.
  sorry

end fraction_sum_l24_24475


namespace sqrt_difference_l24_24141

theorem sqrt_difference (a b : ℝ) (ha : a = 7 + 4 * Real.sqrt 3) (hb : b = 7 - 4 * Real.sqrt 3) :
  Real.sqrt a - Real.sqrt b = 2 * Real.sqrt 3 :=
sorry

end sqrt_difference_l24_24141


namespace alice_sales_surplus_l24_24953

-- Define the constants
def adidas_cost : ℕ := 45
def nike_cost : ℕ := 60
def reebok_cost : ℕ := 35
def quota : ℕ := 1000

-- Define the quantities sold
def adidas_sold : ℕ := 6
def nike_sold : ℕ := 8
def reebok_sold : ℕ := 9

-- Calculate total sales
def total_sales : ℕ := adidas_sold * adidas_cost + nike_sold * nike_cost + reebok_sold * reebok_cost

-- Prove that Alice's total sales minus her quota is 65
theorem alice_sales_surplus : total_sales - quota = 65 := by
  -- Calculation is omitted here. Here is the mathematical fact to prove:
  sorry

end alice_sales_surplus_l24_24953


namespace intersection_line_l24_24334

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 3*x - y = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y = 0

-- Define the line that we need to prove as the intersection
def line (x y : ℝ) : Prop := x - 2*y = 0

-- The theorem to prove
theorem intersection_line (x y : ℝ) : circle1 x y ∧ circle2 x y → line x y :=
by
  sorry

end intersection_line_l24_24334


namespace four_digit_numbers_count_l24_24118

theorem four_digit_numbers_count : 
  (∀ d1 d2 d3 d4 : Fin 4, 
    (d1 = 1 ∨ d1 = 2 ∨ d1 = 3) ∧ 
    d2 ≠ d1 ∧ d2 ≠ 0 ∧ 
    d3 ≠ d1 ∧ d3 ≠ d2 ∧ 
    d4 ≠ d1 ∧ d4 ≠ d2 ∧ d4 ≠ d3) →
  3 * 6 = 18 := 
by
  sorry

end four_digit_numbers_count_l24_24118


namespace equilateral_triangle_area_l24_24726

theorem equilateral_triangle_area (A B C P : ℝ × ℝ)
  (hABC : ∃ a b c : ℝ, a = b ∧ b = c ∧ a = dist A B ∧ b = dist B C ∧ c = dist C A)
  (hPA : dist P A = 10)
  (hPB : dist P B = 8)
  (hPC : dist P C = 12) :
  ∃ (area : ℝ), area = 104 :=
by
  sorry

end equilateral_triangle_area_l24_24726


namespace fraction_remaining_distance_l24_24825

theorem fraction_remaining_distance
  (total_distance : ℕ)
  (first_stop_fraction : ℚ)
  (remaining_distance_after_second_stop : ℕ)
  (fraction_between_stops : ℚ) :
  total_distance = 280 →
  first_stop_fraction = 1/2 →
  remaining_distance_after_second_stop = 105 →
  (fraction_between_stops * (total_distance - (first_stop_fraction * total_distance)) + remaining_distance_after_second_stop = (total_distance - (first_stop_fraction * total_distance))) →
  fraction_between_stops = 1/4 :=
by
  sorry

end fraction_remaining_distance_l24_24825


namespace find_n_l24_24189

theorem find_n : ∃ n : ℕ, n < 2006 ∧ ∀ m : ℕ, 2006 * n = m * (2006 + n) ↔ n = 1475 := by
  sorry

end find_n_l24_24189


namespace exists_right_triangle_area_eq_perimeter_l24_24668

theorem exists_right_triangle_area_eq_perimeter :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a + b + c = (a * b) / 2 ∧ a ≠ b ∧ 
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 12 ∧ b = 5 ∧ c = 13) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10)) :=
by
  sorry

end exists_right_triangle_area_eq_perimeter_l24_24668


namespace prob_k_gnomes_fall_exp_gnomes_falling_l24_24711

variables (n k : ℕ) (p : ℝ)
hypotheses 
  (hn : 0 < n)
  (hp : 0 < p) (hp1 : p < 1)
  (hk : 0 ≤ k) (hk1 : k ≤ n)

open ProbabilityTheory
  
def probability_k_gnomes_fall := 
  p * (1 - p) ^ (n - k)

def expected_gnomes_fall :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p

theorem prob_k_gnomes_fall (hprob : 0 < p ∧ p < 1) : 
  ∀ n k : ℕ, 0 ≤ k ∧ k ≤ n → probability_k_gnomes_fall n k p = p * (1 - p) ^ (n - k) :=
by sorry

theorem exp_gnomes_falling (hprob : 0 < p ∧ p < 1) : 
  ∀ n : ℕ, 0 < n → expected_gnomes_fall n p = n + 1 - (1 / p) + ((1 - p) ^ (n + 1)) / p :=
by sorry

end prob_k_gnomes_fall_exp_gnomes_falling_l24_24711


namespace determinant_triangle_l24_24730

theorem determinant_triangle (A B C : ℝ) (h : A + B + C = Real.pi) :
  Matrix.det ![![Real.cos A ^ 2, Real.tan A, 1],
               ![Real.cos B ^ 2, Real.tan B, 1],
               ![Real.cos C ^ 2, Real.tan C, 1]] = 0 := by
  sorry

end determinant_triangle_l24_24730


namespace fraction_of_juan_chocolates_given_to_tito_l24_24171

variable (n : ℕ)
variable (Juan Angela Tito : ℕ)
variable (f : ℝ)

-- Conditions
def chocolates_Angela_Tito : Angela = 3 * Tito := 
by sorry

def chocolates_Juan_Angela : Juan = 4 * Angela := 
by sorry

def equal_distribution : (Juan + Angela + Tito) = 16 * n := 
by sorry

-- Theorem to prove
theorem fraction_of_juan_chocolates_given_to_tito (n : ℕ) 
  (H1 : Angela = 3 * Tito)
  (H2 : Juan = 4 * Angela)
  (H3 : Juan + Angela + Tito = 16 * n) :
  f = 13 / 36 :=
by sorry

end fraction_of_juan_chocolates_given_to_tito_l24_24171


namespace unknown_subtraction_problem_l24_24705

theorem unknown_subtraction_problem (x y : ℝ) (h1 : x = 40) (h2 : x / 4 * 5 + 10 - y = 48) : y = 12 :=
by
  sorry

end unknown_subtraction_problem_l24_24705


namespace maximize_QP_PR_QR_l24_24345

noncomputable def maximize_ratio (P Q : Point) (π : Plane) (R : Point) : Prop :=
  P ∈ π ∧ Q ∉ π ∧
  (forall S ∈ π, ∃ R, R = (line_through PQ).projection_onto π ∧
  (QP + PR) / QR >= (QP + PS) / QS )

theorem maximize_QP_PR_QR (P Q : Point) (π : Plane) : ∃ R : Point, maximize_ratio P Q π R :=
sorry

end maximize_QP_PR_QR_l24_24345


namespace percentage_calculation_l24_24065

theorem percentage_calculation (P : ℝ) : 
    (P / 100) * 24 + 0.10 * 40 = 5.92 ↔ P = 8 :=
by 
    sorry

end percentage_calculation_l24_24065


namespace step_count_initial_l24_24930

theorem step_count_initial :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (11 * y - x = 64) ∧ (10 * x + y = 26) :=
by
  sorry

end step_count_initial_l24_24930


namespace intersectionAandB_l24_24847

def setA (x : ℝ) : Prop := abs (x + 3) + abs (x - 4) ≤ 9
def setB (x : ℝ) : Prop := ∃ t : ℝ, 0 < t ∧ x = 4 * t + 1 / t - 6

theorem intersectionAandB : {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -2 ≤ x ∧ x ≤ 5} := 
by 
  sorry

end intersectionAandB_l24_24847


namespace polygon_sides_exterior_interior_sum_l24_24367

theorem polygon_sides_exterior_interior_sum (n : ℕ) (h : ((n - 2) * 180 = 360)) : n = 4 :=
by sorry

end polygon_sides_exterior_interior_sum_l24_24367


namespace vikki_tax_deduction_percentage_l24_24918

/- Definitions and conditions -/
def hours_worked : ℝ := 42
def hourly_pay_rate : ℝ := 10
def insurance_rate : ℝ := 0.05
def union_dues : ℝ := 5
def take_home_pay : ℝ := 310
def gross_earnings := hours_worked * hourly_pay_rate
def insurance_cover_deduction := insurance_rate * gross_earnings
def total_deductions_excl_tax := insurance_cover_deduction + union_dues
def total_deductions_incl_tax := gross_earnings - take_home_pay
def tax_deduction := total_deductions_incl_tax - total_deductions_excl_tax

/- The proof statement, asserting the main theorem -/
theorem vikki_tax_deduction_percentage :
  (tax_deduction / gross_earnings) * 100 = 20 :=
by
  sorry

end vikki_tax_deduction_percentage_l24_24918


namespace Lizzy_savings_after_loan_l24_24393

theorem Lizzy_savings_after_loan :
  ∀ (initial_amount loan_amount : ℕ) (interest_percent : ℕ),
  initial_amount = 30 →
  loan_amount = 15 →
  interest_percent = 20 →
  initial_amount - loan_amount + loan_amount + loan_amount * interest_percent / 100 = 33 :=
by
  intros initial_amount loan_amount interest_percent h1 h2 h3
  sorry

end Lizzy_savings_after_loan_l24_24393


namespace remainder_1235678_div_127_l24_24175

theorem remainder_1235678_div_127 : 1235678 % 127 = 69 := by
  sorry

end remainder_1235678_div_127_l24_24175


namespace Lizzy_money_after_loan_l24_24397

theorem Lizzy_money_after_loan :
  let initial_savings := 30
  let loaned_amount := 15
  let interest_rate := 0.20
  let interest := loaned_amount * interest_rate
  let total_amount_returned := loaned_amount + interest
  let remaining_money := initial_savings - loaned_amount
  let total_money := remaining_money + total_amount_returned
  total_money = 33 :=
by
  sorry

end Lizzy_money_after_loan_l24_24397


namespace x_range_l24_24122

theorem x_range (x : ℝ) : (x + 2) > 0 → (3 - x) ≥ 0 → (-2 < x ∧ x ≤ 3) :=
by
  intro h1 h2
  constructor
  { linarith }
  { linarith }

end x_range_l24_24122


namespace find_y_l24_24634

theorem find_y 
  (x y : ℕ) 
  (h1 : x % y = 9) 
  (h2 : x / y = 96) 
  (h3 : (x % y: ℝ) / y = 0.12) 
  : y = 75 := 
  by 
    sorry

end find_y_l24_24634


namespace highest_price_per_shirt_l24_24796

theorem highest_price_per_shirt (x : ℝ) 
  (num_shirts : ℕ := 20)
  (total_money : ℝ := 180)
  (entrance_fee : ℝ := 5)
  (sales_tax : ℝ := 0.08)
  (whole_number: ∀ p : ℝ, ∃ n : ℕ, p = n) :
  (∀ (price_per_shirt : ℕ), price_per_shirt ≤ 8) :=
by
  sorry

end highest_price_per_shirt_l24_24796


namespace men_left_bus_l24_24648

theorem men_left_bus (M W : ℕ) (initial_passengers : M + W = 72) 
  (women_half_men : W = M / 2) 
  (equal_men_women_after_changes : ∃ men_left : ℕ, ∀ W_new, W_new = W + 8 → M - men_left = W_new → M - men_left = 32) :
  ∃ men_left : ℕ, men_left = 16 :=
  sorry

end men_left_bus_l24_24648


namespace sequence_an_solution_l24_24204

theorem sequence_an_solution {a : ℕ → ℝ} (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, 0 < n → (1 / a (n + 1) = 1 / a n + 1)) : ∀ n : ℕ, 0 < n → (a n = 1 / n) :=
by
  sorry

end sequence_an_solution_l24_24204


namespace defective_chip_ratio_l24_24457

theorem defective_chip_ratio (defective_chips total_chips : ℕ)
  (h1 : defective_chips = 15)
  (h2 : total_chips = 60000) :
  defective_chips / total_chips = 1 / 4000 :=
by
  sorry

end defective_chip_ratio_l24_24457


namespace no_solution_abs_val_l24_24926

theorem no_solution_abs_val (x : ℝ) : ¬(∃ x : ℝ, |5 * x| + 7 = 0) :=
sorry

end no_solution_abs_val_l24_24926


namespace train_stoppages_l24_24280

variables (sA sA' sB sB' sC sC' : ℝ)
variables (x y z : ℝ)

-- Conditions
def conditions : Prop :=
  sA = 80 ∧ sA' = 60 ∧
  sB = 100 ∧ sB' = 75 ∧
  sC = 120 ∧ sC' = 90

-- Goal that we need to prove
def goal : Prop :=
  x = 15 ∧ y = 15 ∧ z = 15

-- Main statement
theorem train_stoppages : conditions sA sA' sB sB' sC sC' → goal x y z :=
by
  sorry

end train_stoppages_l24_24280


namespace pq_sum_l24_24840

theorem pq_sum {p q : ℤ}
  (h : ∀ x : ℤ, 36 * x^2 - 4 * (p^2 + 11) * x + 135 * (p + q) + 576 = 0) :
  p + q = 20 :=
sorry

end pq_sum_l24_24840


namespace number_of_integers_between_cubed_values_l24_24358

theorem number_of_integers_between_cubed_values :
  ∃ n : ℕ, n = (1278 - 1122 + 1) ∧ 
  ∀ x : ℤ, (1122 < x ∧ x < 1278) → (1123 ≤ x ∧ x ≤ 1277) := 
by
  sorry

end number_of_integers_between_cubed_values_l24_24358


namespace Lizzy_savings_after_loan_l24_24392

theorem Lizzy_savings_after_loan :
  ∀ (initial_amount loan_amount : ℕ) (interest_percent : ℕ),
  initial_amount = 30 →
  loan_amount = 15 →
  interest_percent = 20 →
  initial_amount - loan_amount + loan_amount + loan_amount * interest_percent / 100 = 33 :=
by
  intros initial_amount loan_amount interest_percent h1 h2 h3
  sorry

end Lizzy_savings_after_loan_l24_24392


namespace num_real_numbers_l24_24521

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end num_real_numbers_l24_24521


namespace count_distinct_x_l24_24514

theorem count_distinct_x :
  { x : ℝ | ∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ s = floor (sqrt (123 - sqrt x)) } = 12 :=
sorry

end count_distinct_x_l24_24514


namespace range_of_k_l24_24015

theorem range_of_k (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 → y^2 = 2 * x → (∃! (x₀ y₀ : ℝ), y₀ = k * x₀ + 1 ∧ y₀^2 = 2 * x₀)) ↔ 
  (k = 0 ∨ k ≥ 1/2) :=
sorry

end range_of_k_l24_24015


namespace solve_system_l24_24911

theorem solve_system (x y : ℝ) (h1 : 2 * x - y = 0) (h2 : x + 2 * y = 1) : 
  x = 1 / 5 ∧ y = 2 / 5 :=
by
  sorry

end solve_system_l24_24911


namespace relationship_between_x_and_y_l24_24073

variable (u : ℝ)

theorem relationship_between_x_and_y (h : u > 0) (hx : x = (u + 1)^(1 / u)) (hy : y = (u + 1)^((u + 1) / u)) :
  y^x = x^y :=
by
  sorry

end relationship_between_x_and_y_l24_24073


namespace diagonals_in_nine_sided_polygon_l24_24996

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l24_24996


namespace x_intercept_is_neg_three_halves_l24_24256

-- Definition of the points
def pointA : ℝ × ℝ := (-1, 1)
def pointB : ℝ × ℝ := (3, 9)

-- Statement of the theorem: The x-intercept of the line passing through the points is -3/2.
theorem x_intercept_is_neg_three_halves (A B : ℝ × ℝ)
    (hA : A = pointA)
    (hB : B = pointB) :
    ∃ x_intercept : ℝ, x_intercept = -3 / 2 := 
by
    sorry

end x_intercept_is_neg_three_halves_l24_24256


namespace problem_b_c_constants_l24_24217

theorem problem_b_c_constants (b c : ℝ) (h : ∀ x : ℝ, (x + 2) * (x + b) = x^2 + c * x + 6) : c = 5 := 
by sorry

end problem_b_c_constants_l24_24217


namespace diagonals_in_nine_sided_polygon_l24_24997

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l24_24997


namespace amount_of_bill_l24_24761

noncomputable def TD : ℝ := 360
noncomputable def BD : ℝ := 418.9090909090909
noncomputable def FV (TD BD : ℝ) : ℝ := TD * BD / (BD - TD)

theorem amount_of_bill :
  FV TD BD = 2568 :=
by
  sorry

end amount_of_bill_l24_24761


namespace sum_of_a2_and_a3_l24_24841

theorem sum_of_a2_and_a3 (S : ℕ → ℕ) (hS : ∀ n, S n = 3^n + 1) :
  S 3 - S 1 = 24 :=
by
  sorry

end sum_of_a2_and_a3_l24_24841


namespace select_4_officers_from_7_members_l24_24822

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Statement of the problem
theorem select_4_officers_from_7_members : binom 7 4 = 35 :=
by
  -- Proof not required, so we use sorry to skip it
  sorry

end select_4_officers_from_7_members_l24_24822


namespace pond_water_after_evaporation_l24_24305

theorem pond_water_after_evaporation 
  (I R D : ℕ) 
  (h_initial : I = 250)
  (h_evaporation_rate : R = 1)
  (h_days : D = 50) : 
  I - (R * D) = 200 := 
by 
  sorry

end pond_water_after_evaporation_l24_24305


namespace t_shirt_cost_l24_24054

theorem t_shirt_cost (total_amount_spent : ℝ) (number_of_t_shirts : ℕ) (cost_per_t_shirt : ℝ)
  (h0 : total_amount_spent = 201) 
  (h1 : number_of_t_shirts = 22)
  (h2 : cost_per_t_shirt = total_amount_spent / number_of_t_shirts) :
  cost_per_t_shirt = 9.14 := 
sorry

end t_shirt_cost_l24_24054


namespace binomial_12_6_eq_924_l24_24811

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := sorry

end binomial_12_6_eq_924_l24_24811


namespace solve_inequality_l24_24059

theorem solve_inequality (a : ℝ) : (∀ x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2 ↔ x = -a) ↔ (a = 1 ∨ a = 2) :=
sorry

end solve_inequality_l24_24059


namespace average_age_of_9_students_l24_24744

theorem average_age_of_9_students (avg_age_17_students : ℕ)
                                   (num_students : ℕ)
                                   (avg_age_5_students : ℕ)
                                   (num_5_students : ℕ)
                                   (age_17th_student : ℕ) :
    avg_age_17_students = 17 →
    num_students = 17 →
    avg_age_5_students = 14 →
    num_5_students = 5 →
    age_17th_student = 75 →
    (144 / 9) = 16 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_age_of_9_students_l24_24744


namespace sufficient_condition_for_increasing_l24_24206

theorem sufficient_condition_for_increasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^y < a^x) →
  (∀ x y : ℝ, x < y → (2 - a) * y ^ 3 > (2 - a) * x ^ 3) :=
sorry

end sufficient_condition_for_increasing_l24_24206


namespace smaller_circle_radius_is_6_l24_24327

-- Define the conditions of the problem
def large_circle_radius : ℝ := 2

def smaller_circles_touching_each_other (r : ℝ) : Prop :=
  let oa := large_circle_radius + r
  let ob := large_circle_radius + r
  let ab := 2 * r
  (oa^2 + ob^2 = ab^2)

def problem_statement : Prop :=
  ∃ r : ℝ, smaller_circles_touching_each_other r ∧ r = 6

theorem smaller_circle_radius_is_6 : problem_statement :=
sorry

end smaller_circle_radius_is_6_l24_24327


namespace problem_composite_for_n_geq_9_l24_24673

theorem problem_composite_for_n_geq_9 (n : ℤ) (h : n ≥ 9) : ∃ k m : ℤ, (2 ≤ k ∧ 2 ≤ m ∧ n + 7 = k * m) :=
by
  sorry

end problem_composite_for_n_geq_9_l24_24673


namespace total_cable_cost_l24_24322

theorem total_cable_cost 
    (num_east_west_streets : ℕ)
    (length_east_west_street : ℕ)
    (num_north_south_streets : ℕ)
    (length_north_south_street : ℕ)
    (cable_multiplier : ℕ)
    (cable_cost_per_mile : ℕ)
    (h1 : num_east_west_streets = 18)
    (h2 : length_east_west_street = 2)
    (h3 : num_north_south_streets = 10)
    (h4 : length_north_south_street = 4)
    (h5 : cable_multiplier = 5)
    (h6 : cable_cost_per_mile = 2000) :
    (num_east_west_streets * length_east_west_street + num_north_south_streets * length_north_south_street) * cable_multiplier * cable_cost_per_mile = 760000 := 
by
    sorry

end total_cable_cost_l24_24322


namespace find_coordinates_of_P_l24_24349

-- Define points N and M with given symmetries.
structure Point where
  x : ℝ
  y : ℝ

def symmetric_about_x (P1 P2 : Point) : Prop :=
  P1.x = P2.x ∧ P1.y = -P2.y

def symmetric_about_y (P1 P2 : Point) : Prop :=
  P1.x = -P2.x ∧ P1.y = P2.y

-- Given conditions
def N : Point := ⟨1, 2⟩
def M : Point := ⟨-1, 2⟩ -- derived from symmetry about y-axis with N
def P : Point := ⟨-1, -2⟩ -- derived from symmetry about x-axis with M

theorem find_coordinates_of_P :
  symmetric_about_x M P ∧ symmetric_about_y N M → P = ⟨-1, -2⟩ :=
by
  sorry

end find_coordinates_of_P_l24_24349


namespace pyramid_volume_l24_24986

theorem pyramid_volume (a : ℝ) (h : a > 0) : (1 / 6) * a^3 = 1 / 6 * a^3 :=
by
  sorry

end pyramid_volume_l24_24986


namespace santana_brothers_l24_24579

theorem santana_brothers (b : ℕ) (x : ℕ) (h1 : x + b = 7) (h2 : 3 + 8 = x + 1 + 2 + 7) : x = 1 :=
by
  -- Providing the necessary definitions and conditions
  let brothers := 7 -- Santana has 7 brothers
  let march_birthday := 3 -- 3 brothers have birthdays in March
  let november_birthday := 1 -- 1 brother has a birthday in November
  let december_birthday := 2 -- 2 brothers have birthdays in December
  let total_presents_first_half := 3 -- Total presents in the first half of the year is 3 (March)
  let x := x -- Number of brothers with birthdays in October to be proved
  let total_presents_second_half := x + 1 + 2 + 7 -- Total presents in the second half of the year
  have h3 : total_presents_first_half + 8 = total_presents_second_half := h2 -- Condition equation
  
  -- Start solving the proof
  sorry

end santana_brothers_l24_24579


namespace maximum_notebooks_maria_can_buy_l24_24879

def price_single : ℕ := 1
def price_pack_4 : ℕ := 3
def price_pack_7 : ℕ := 5
def total_budget : ℕ := 10

def max_notebooks (budget : ℕ) : ℕ :=
  if budget < price_single then 0
  else if budget < price_pack_4 then budget / price_single
  else if budget < price_pack_7 then max (budget / price_single) (4 * (budget / price_pack_4))
  else max (budget / price_single) (7 * (budget / price_pack_7))

theorem maximum_notebooks_maria_can_buy :
  max_notebooks total_budget = 14 := by
  sorry

end maximum_notebooks_maria_can_buy_l24_24879


namespace composite_dice_product_probability_l24_24535

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l24_24535


namespace num_real_numbers_l24_24522

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end num_real_numbers_l24_24522


namespace sequence_expression_l24_24089

theorem sequence_expression (a : ℕ → ℚ)
  (h1 : a 1 = 2 / 3)
  (h2 : ∀ n : ℕ, a (n + 1) = (n / (n + 1)) * a n) :
  ∀ n : ℕ, a n = 2 / (3 * n) :=
sorry

end sequence_expression_l24_24089


namespace probability_composite_l24_24553

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def non_composite_outcomes : ℕ := 1 + 12

theorem probability_composite :
  let total_outcomes := 6^4 in
  let non_composite := non_composite_outcomes in
  let composite_probability := (total_outcomes - non_composite) / total_outcomes.toRat in
  composite_probability = 1283 / 1296 := by
  let total_outcomes := 6^4
  let non_composite := 1 + 12
  let composite_probability := (total_outcomes - non_composite).toRat / total_outcomes.toRat
  sorry

end probability_composite_l24_24553


namespace Catriona_goldfish_count_l24_24476

theorem Catriona_goldfish_count (G : ℕ) (A : ℕ) (U : ℕ) 
    (h1 : A = G + 4) 
    (h2 : U = 2 * A) 
    (h3 : G + A + U = 44) : G = 8 :=
by
  -- Proof goes here
  sorry

end Catriona_goldfish_count_l24_24476


namespace next_shared_meeting_day_l24_24007

-- Definitions based on the conditions:
def dramaClubMeetingInterval : ℕ := 3
def choirMeetingInterval : ℕ := 5
def debateTeamMeetingInterval : ℕ := 7

-- Statement to prove:
theorem next_shared_meeting_day : Nat.lcm (Nat.lcm dramaClubMeetingInterval choirMeetingInterval) debateTeamMeetingInterval = 105 := by
  sorry

end next_shared_meeting_day_l24_24007


namespace diagonals_in_nine_sided_polygon_l24_24998

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l24_24998


namespace part1_A_union_B_when_a_eq_2_l24_24076

variable {a : ℝ}

def A : Set ℝ := {x | (x - 1) / (x - 2) ≤ 1 / 2}

def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2) * x + 2 * a ≤ 0}

def A_Union_B_when_a_eq_2 : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem part1_A_union_B_when_a_eq_2 : A ∪ B 2 = A_Union_B_when_a_eq_2 :=
  sorry

end part1_A_union_B_when_a_eq_2_l24_24076


namespace min_value_problem_inequality_solution_l24_24686

-- Definition of the function
noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Part (i): Minimum value problem
theorem min_value_problem (a : ℝ) (minF : ∀ x : ℝ, f x a ≥ 2) : a = 0 ∨ a = -4 :=
by
  sorry

-- Part (ii): Inequality solving problem
theorem inequality_solution (x : ℝ) (a : ℝ := 2) : f x a ≤ 6 ↔ -3 ≤ x ∧ x ≤ 3 :=
by
  sorry

end min_value_problem_inequality_solution_l24_24686


namespace halfway_fraction_between_l24_24197

theorem halfway_fraction_between (a b : ℚ) (h_a : a = 1/6) (h_b : b = 1/4) : (a + b) / 2 = 5 / 24 :=
by
  have h1 : a = (1 : ℚ) / 6 := h_a
  have h2 : b = (1 : ℚ) / 4 := h_b
  sorry

end halfway_fraction_between_l24_24197


namespace center_of_image_circle_l24_24596

def point := ℝ × ℝ

def reflect_about_y_eq_neg_x (p : point) : point :=
  let (a, b) := p
  (-b, -a)

theorem center_of_image_circle :
  reflect_about_y_eq_neg_x (8, -3) = (3, -8) :=
by
  sorry

end center_of_image_circle_l24_24596


namespace bike_growth_equation_l24_24895

-- Declare the parameters
variables (b1 b3 : ℕ) (x : ℝ)
-- Define the conditions
def condition1 : b1 = 1000 := sorry
def condition2 : b3 = b1 + 440 := sorry

-- Define the proposition to be proved
theorem bike_growth_equation (cond1 : b1 = 1000) (cond2 : b3 = b1 + 440) :
  b1 * (1 + x)^2 = b3 :=
sorry

end bike_growth_equation_l24_24895


namespace fraction_eq_l24_24359

theorem fraction_eq {x : ℝ} (h : 1 - 6 / x + 9 / x ^ 2 - 2 / x ^ 3 = 0) :
  3 / x = 3 / 2 ∨ 3 / x = 3 / (2 + Real.sqrt 3) ∨ 3 / x = 3 / (2 - Real.sqrt 3) :=
sorry

end fraction_eq_l24_24359


namespace alpha_centauri_puzzle_max_numbers_l24_24150

theorem alpha_centauri_puzzle_max_numbers (A B N : ℕ) (h1 : A = 1353) (h2 : B = 2134) (h3 : N = 11) : 
  ∃ S : set ℕ, S ⊆ {n | A ≤ n ∧ n ≤ B} ∧ (∀ x y ∈ S, x ≠ y → (x + y) % N ≠ 0) ∧ S.card = 356 :=
sorry

end alpha_centauri_puzzle_max_numbers_l24_24150


namespace problem_l24_24342

noncomputable def a : ℝ := Real.exp 1 - 2
noncomputable def b : ℝ := 1 - Real.log 2
noncomputable def c : ℝ := Real.exp (Real.exp 1) - Real.exp 2

theorem problem (a_def : a = Real.exp 1 - 2) 
                (b_def : b = 1 - Real.log 2) 
                (c_def : c = Real.exp (Real.exp 1) - Real.exp 2) : 
                c > a ∧ a > b := 
by 
  rw [a_def, b_def, c_def]
  sorry

end problem_l24_24342


namespace center_of_image_circle_l24_24595

def point := ℝ × ℝ

def reflect_about_y_eq_neg_x (p : point) : point :=
  let (a, b) := p
  (-b, -a)

theorem center_of_image_circle :
  reflect_about_y_eq_neg_x (8, -3) = (3, -8) :=
by
  sorry

end center_of_image_circle_l24_24595


namespace m_minus_n_is_square_l24_24642

theorem m_minus_n_is_square (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 2001 * m ^ 2 + m = 2002 * n ^ 2 + n) : ∃ k : ℕ, m - n = k ^ 2 :=
sorry

end m_minus_n_is_square_l24_24642


namespace people_who_like_both_l24_24706

-- Conditions
variables (total : ℕ) (a : ℕ) (b : ℕ) (none : ℕ)
-- Express the problem
theorem people_who_like_both : total = 50 → a = 23 → b = 20 → none = 14 → (a + b - (total - none) = 7) :=
by
  intros
  sorry

end people_who_like_both_l24_24706


namespace cheryl_used_total_amount_l24_24178

theorem cheryl_used_total_amount :
  let bought_A := (5 / 8 : ℚ)
  let bought_B := (2 / 9 : ℚ)
  let bought_C := (2 / 5 : ℚ)
  let leftover_A := (1 / 12 : ℚ)
  let leftover_B := (5 / 36 : ℚ)
  let leftover_C := (1 / 10 : ℚ)
  let used_A := bought_A - leftover_A
  let used_B := bought_B - leftover_B
  let used_C := bought_C - leftover_C
  used_A + used_B + used_C = 37 / 40 :=
by 
  sorry

end cheryl_used_total_amount_l24_24178


namespace dice_product_composite_probability_l24_24540

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l24_24540


namespace Q3_x_coords_sum_eq_Q1_x_coords_sum_l24_24300

-- Define a 40-gon and its x-coordinates sum
def Q1_x_coords_sum : ℝ := 120

-- Statement to prove
theorem Q3_x_coords_sum_eq_Q1_x_coords_sum (Q1_x_coords_sum: ℝ) (h: Q1_x_coords_sum = 120) : 
  (Q3_x_coords_sum: ℝ) = Q1_x_coords_sum :=
sorry

end Q3_x_coords_sum_eq_Q1_x_coords_sum_l24_24300


namespace design_height_lower_part_l24_24288

theorem design_height_lower_part (H : ℝ) (H_eq : H = 2) (L : ℝ) 
  (ratio : (H - L) / L = L / H) : L = Real.sqrt 5 - 1 :=
by {
  sorry
}

end design_height_lower_part_l24_24288


namespace sqrt_meaningful_range_l24_24695

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 2) : x ≥ 2 :=
sorry

end sqrt_meaningful_range_l24_24695


namespace probability_composite_l24_24550

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def non_composite_outcomes : ℕ := 1 + 12

theorem probability_composite :
  let total_outcomes := 6^4 in
  let non_composite := non_composite_outcomes in
  let composite_probability := (total_outcomes - non_composite) / total_outcomes.toRat in
  composite_probability = 1283 / 1296 := by
  let total_outcomes := 6^4
  let non_composite := 1 + 12
  let composite_probability := (total_outcomes - non_composite).toRat / total_outcomes.toRat
  sorry

end probability_composite_l24_24550


namespace distance_from_point_to_x_axis_l24_24226

theorem distance_from_point_to_x_axis (x y : ℝ) (P : ℝ × ℝ) (hP : P = (x, y)) :
  abs (y) = 3 :=
by
  -- Assuming the y-coordinate is given as -3
  have hy : y = -3 := sorry
  rw [hy]
  exact abs_neg 3

end distance_from_point_to_x_axis_l24_24226


namespace dessert_menu_count_is_192_l24_24651

-- Defining the set of desserts
inductive Dessert
| cake | pie | ice_cream

-- Function to count valid dessert menus (not repeating on consecutive days) with cake on Friday
def countDessertMenus : Nat :=
  -- Let's denote Sunday as day 1 and Saturday as day 7
  let sunday_choices := 3
  let weekday_choices := 2 -- for Monday to Thursday (no repeats consecutive)
  let weekend_choices := 2 -- for Saturday and Sunday after
  sunday_choices * weekday_choices^4 * 1 * weekend_choices^2

-- Theorem stating the number of valid dessert menus for the week
theorem dessert_menu_count_is_192 : countDessertMenus = 192 :=
  by
    -- Actual proof is omitted
    sorry

end dessert_menu_count_is_192_l24_24651


namespace downstream_rate_l24_24655

/--  
A man's rowing conditions and rates:
- The man's upstream rate is U = 12 kmph.
- The man's rate in still water is S = 7 kmph.
- We need to prove that the man's downstream rate D is 14 kmph.
-/
theorem downstream_rate (U S D : ℝ) (hU : U = 12) (hS : S = 7) : D = 14 :=
by
  -- Proof to be filled here
  sorry

end downstream_rate_l24_24655


namespace simplify_division_l24_24581

theorem simplify_division (a b c d : ℕ) (h1 : a = 27) (h2 : b = 10^12) (h3 : c = 9) (h4 : d = 10^4) :
  ((a * b) / (c * d) = 300000000) :=
by {
  sorry
}

end simplify_division_l24_24581


namespace dice_product_composite_probability_l24_24549

theorem dice_product_composite_probability :
  (let total_outcomes := (6 : ℕ)^4,
       non_composite_outcomes := 1,
       composite_probability := 1 - (non_composite_outcomes / total_outcomes) in
   composite_probability = 1295 / 1296) :=
by sorry

end dice_product_composite_probability_l24_24549


namespace right_angled_triangle_with_inscribed_circle_isosceles_triangle_with_inscribed_circle_l24_24917

structure Triangle :=
(base : ℝ)
(height : ℝ)
(hypotenuse : ℝ)

structure IsoscelesTriangle :=
(side : ℝ)
(base : ℝ)

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def perimeter_triangle (t : Triangle) : ℝ :=
t.base + t.height + t.hypotenuse

noncomputable def perimeter_isosceles_triangle (it : IsoscelesTriangle) : ℝ :=
2 * it.side + it.base

def angles_triangle := (α β : ℝ) -> Triangle -> α + β = 90

def angles_isosceles_triangle := (α β : ℝ) -> IsoscelesTriangle -> α = β

theorem right_angled_triangle_with_inscribed_circle (r : ℝ)
    (P : ℝ := 2 * circumference r) 
    (t : Triangle) 
    (α β : ℝ) 
    (h_perimeter : perimeter_triangle t = P)
    (h_angles : angles_triangle α β t) :
    (α ≈ 58 ∧ β ≈ 32) :=
sorry

theorem isosceles_triangle_with_inscribed_circle (r : ℝ)
    (P : ℝ := 2 * circumference r)
    (it : IsoscelesTriangle)
    (α β : ℝ)
    (h_perimeter : perimeter_isosceles_triangle it = P)
    (h_angles : angles_isosceles_triangle α β it) :
    (α ≈ 75 ∧ β ≈ 30) :=
sorry

end right_angled_triangle_with_inscribed_circle_isosceles_triangle_with_inscribed_circle_l24_24917


namespace people_attend_both_reunions_l24_24643

theorem people_attend_both_reunions (N D H x : ℕ) 
  (hN : N = 50)
  (hD : D = 50)
  (hH : H = 60)
  (h_total : N = D + H - x) : 
  x = 60 :=
by
  sorry

end people_attend_both_reunions_l24_24643


namespace arithmetic_sum_problem_l24_24494

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sum_problem (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : arithmetic_sequence a)
  (h_S_def : ∀ n : ℕ, S n = sum_of_first_n_terms a n)
  (h_S13 : S 13 = 52) : a 4 + a 8 + a 9 = 12 :=
sorry

end arithmetic_sum_problem_l24_24494


namespace circle_reflection_l24_24594

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3) : 
  let reflected_x := -y, reflected_y := -x in
  reflected_x = 3 ∧ reflected_y = -8 :=
by
  sorry

end circle_reflection_l24_24594


namespace find_n_value_l24_24943

theorem find_n_value (n a b : ℕ) 
    (h1 : n = 12 * b + a)
    (h2 : n = 10 * a + b)
    (h3 : 0 ≤ a ∧ a ≤ 11)
    (h4 : 0 ≤ b ∧ b ≤ 9) : 
    n = 119 :=
by
  sorry

end find_n_value_l24_24943


namespace ranking_sequences_l24_24248

theorem ranking_sequences
    (A D B E C : Type)
    (h_no_ties : ∀ (X Y : Type), X ≠ Y)
    (h_games : (W1 = A ∨ W1 = D) ∧ (W2 = B ∨ W2 = E) ∧ (W3 = W1 ∨ W3 = C)) :
  ∃! (n : ℕ), n = 48 := 
sorry

end ranking_sequences_l24_24248


namespace square_same_area_as_rectangle_l24_24944

theorem square_same_area_as_rectangle (l w : ℝ) (rect_area sq_side : ℝ) :
  l = 25 → w = 9 → rect_area = l * w → sq_side^2 = rect_area → sq_side = 15 :=
by
  intros h_l h_w h_rect_area h_sq_area
  rw [h_l, h_w] at h_rect_area
  sorry

end square_same_area_as_rectangle_l24_24944


namespace fraction_of_juniors_l24_24559

theorem fraction_of_juniors (J S : ℕ) (h1 : J > 0) (h2 : S > 0) (h : 1 / 2 * J = 2 / 3 * S) : J / (J + S) = 4 / 7 :=
by
  sorry

end fraction_of_juniors_l24_24559


namespace sufficient_not_necessary_condition_l24_24677

theorem sufficient_not_necessary_condition (x y : ℝ) (h1 : x ≥ 1) (h2 : y ≥ 2) : 
  x + y ≥ 3 ∧ (¬ (∀ x y : ℝ, x + y ≥ 3 → x ≥ 1 ∧ y ≥ 2)) := 
by {
  sorry -- The actual proof goes here.
}

end sufficient_not_necessary_condition_l24_24677


namespace math_problem_l24_24221

theorem math_problem {x y : ℕ} (h1 : 1059 % x = y) (h2 : 1417 % x = y) (h3 : 2312 % x = y) : x - y = 15 := by
  sorry

end math_problem_l24_24221


namespace center_of_image_circle_l24_24597

def point := ℝ × ℝ

def reflect_about_y_eq_neg_x (p : point) : point :=
  let (a, b) := p
  (-b, -a)

theorem center_of_image_circle :
  reflect_about_y_eq_neg_x (8, -3) = (3, -8) :=
by
  sorry

end center_of_image_circle_l24_24597


namespace hyperbola_asymptote_ratio_l24_24337

theorem hyperbola_asymptote_ratio
  (a b : ℝ) (h₁ : a ≠ b) (h₂ : (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1))
  (h₃ : ∀ m n: ℝ, m * n = -1 → ∃ θ: ℝ, θ = 90* (π / 180)): 
  a / b = 1 := 
sorry

end hyperbola_asymptote_ratio_l24_24337


namespace average_score_for_girls_at_both_schools_combined_l24_24050

/-
  The following conditions are given:
  - Average score for boys at Lincoln HS = 75
  - Average score for boys at Monroe HS = 85
  - Average score for boys at both schools combined = 82
  - Average score for girls at Lincoln HS = 78
  - Average score for girls at Monroe HS = 92
  - Average score for boys and girls combined at Lincoln HS = 76
  - Average score for boys and girls combined at Monroe HS = 88

  The goal is to prove that the average score for the girls at both schools combined is 89.
-/
theorem average_score_for_girls_at_both_schools_combined 
  (L l M m : ℕ)
  (h1 : (75 * L + 78 * l) / (L + l) = 76)
  (h2 : (85 * M + 92 * m) / (M + m) = 88)
  (h3 : (75 * L + 85 * M) / (L + M) = 82)
  : (78 * l + 92 * m) / (l + m) = 89 := 
sorry

end average_score_for_girls_at_both_schools_combined_l24_24050


namespace sum_of_drawn_numbers_is_26_l24_24487

theorem sum_of_drawn_numbers_is_26 :
  ∃ A B : ℕ, A > 1 ∧ A ≤ 50 ∧ B ≤ 50 ∧ A ≠ B ∧ Prime B ∧
           (150 * B + A = k^2) ∧ 1 ≤ B ∧ (B > 1 → A > 1 ∧ B = 2) ∧ A + B = 26 :=
by
  sorry

end sum_of_drawn_numbers_is_26_l24_24487


namespace pow_sum_ge_mul_l24_24889

theorem pow_sum_ge_mul (m n : ℕ) : 2^(m + n - 2) ≥ m * n := 
sorry

end pow_sum_ge_mul_l24_24889


namespace distance_between_trees_l24_24293

theorem distance_between_trees
  (num_trees : ℕ)
  (length_of_yard : ℝ)
  (one_tree_at_each_end : True)
  (h1 : num_trees = 26)
  (h2 : length_of_yard = 400) :
  length_of_yard / (num_trees - 1) = 16 :=
by
  sorry

end distance_between_trees_l24_24293


namespace cubic_roots_identity_l24_24740

theorem cubic_roots_identity (x1 x2 p q : ℝ) 
  (h1 : x1^2 + p * x1 + q = 0) 
  (h2 : x2^2 + p * x2 + q = 0) :
  (x1^3 + x2^3 = 3 * p * q - p^3) ∧ 
  (x1^3 - x2^3 = (p^2 - q) * Real.sqrt (p^2 - 4 * q) ∨ 
   x1^3 - x2^3 = -(p^2 - q) * Real.sqrt (p^2 - 4 * q)) :=
by
  sorry

end cubic_roots_identity_l24_24740


namespace greatest_three_digit_multiple_of_23_l24_24623

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of_23 (n : ℕ) : Prop :=
  n % 23 = 0

theorem greatest_three_digit_multiple_of_23 :
  ∀ n, is_three_digit n ∧ is_multiple_of_23 n → n ≤ 989 :=
by
  sorry

end greatest_three_digit_multiple_of_23_l24_24623


namespace greatest_three_digit_multiple_of_23_is_991_l24_24626

theorem greatest_three_digit_multiple_of_23_is_991 :
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 23 = 0) ∧ ∀ m : ℤ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 23 = 0) → m ≤ n :=
begin
  use 991,
  -- proof steps go here
  sorry
end

end greatest_three_digit_multiple_of_23_is_991_l24_24626


namespace johann_oranges_l24_24566

/-
  Johann had 60 oranges. He decided to eat 10.
  Once he ate them, half were stolen by Carson.
  Carson returned exactly 5. 
  How many oranges does Johann have now?
-/
theorem johann_oranges (initial_oranges : Nat) (eaten_oranges : Nat) (carson_returned : Nat) : 
  initial_oranges = 60 → eaten_oranges = 10 → carson_returned = 5 → 
  (initial_oranges - eaten_oranges) / 2 + carson_returned = 30 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end johann_oranges_l24_24566


namespace mat_inverse_sum_l24_24750

theorem mat_inverse_sum (a b c d : ℝ)
  (h1 : -2 * a + 3 * d = 1)
  (h2 : a * c - 12 = 0)
  (h3 : -8 + b * d = 0)
  (h4 : 4 * c - 4 * b = 0)
  (abc : a = 3 * Real.sqrt 2)
  (bb : b = 2 * Real.sqrt 2)
  (cc : c = 2 * Real.sqrt 2)
  (dd : d = (1 + 6 * Real.sqrt 2) / 3) :
  a + b + c + d = 9 * Real.sqrt 2 + 1 / 3 := by
  sorry

end mat_inverse_sum_l24_24750


namespace product_zero_when_a_is_2_l24_24331

theorem product_zero_when_a_is_2 : 
  ∀ (a : ℤ), a = 2 → (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  intros a ha
  sorry

end product_zero_when_a_is_2_l24_24331


namespace change_correct_l24_24693

def cost_gum : ℕ := 350
def cost_protractor : ℕ := 500
def amount_paid : ℕ := 1000

theorem change_correct : amount_paid - (cost_gum + cost_protractor) = 150 := by
  sorry

end change_correct_l24_24693


namespace distance_major_minor_axis_of_ellipse_l24_24182

noncomputable def ellipse_distance (x y : ℝ) : ℝ :=
  4 * (x-3)^2 + 16 * (y+2)^2

theorem distance_major_minor_axis_of_ellipse :
  (4 * (x-3)^2 + 16 * (y+2)^2 = 64) → 
  (distance (3 + 4, -2) (3, -2 + 2) = 2 * Real.sqrt 5) :=
by
  intros h
  sorry

end distance_major_minor_axis_of_ellipse_l24_24182


namespace smallest_7_digit_number_divisible_by_all_l24_24134

def smallest_7_digit_number : ℕ := 7207200

theorem smallest_7_digit_number_divisible_by_all :
  smallest_7_digit_number >= 1000000 ∧ smallest_7_digit_number < 10000000 ∧
  smallest_7_digit_number % 35 = 0 ∧ 
  smallest_7_digit_number % 112 = 0 ∧ 
  smallest_7_digit_number % 175 = 0 ∧ 
  smallest_7_digit_number % 288 = 0 ∧ 
  smallest_7_digit_number % 429 = 0 ∧ 
  smallest_7_digit_number % 528 = 0 :=
by
  sorry

end smallest_7_digit_number_divisible_by_all_l24_24134


namespace spherical_to_rectangular_coordinates_l24_24183

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ) (x y z : ℝ),
    ρ = 15 →
    θ = 5 * Real.pi / 6 →
    φ = Real.pi / 3 →
    x = ρ * Real.sin φ * Real.cos θ →
    y = ρ * Real.sin φ * Real.sin θ →
    z = ρ * Real.cos φ →
    x = -45 / 4 ∧ y = -15 * Real.sqrt 3 / 4 ∧ z = 7.5 := 
by
  intro ρ θ φ x y z
  intro hρ hθ hφ hx hy hz
  rw [hρ, hθ, hφ] at *
  rw [hx, hy, hz]
  sorry

end spherical_to_rectangular_coordinates_l24_24183


namespace minimum_value_l24_24068

noncomputable def smallest_value_expression (x y : ℝ) := x^4 + y^4 - x^2 * y - x * y^2

theorem minimum_value (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y ≤ 1) :
  (smallest_value_expression x y) ≥ -1 / 8 :=
sorry

end minimum_value_l24_24068


namespace race_course_length_l24_24649

theorem race_course_length (v : ℝ) (d : ℝ) (h1 : d = 7 * (d - 120)) : d = 140 :=
sorry

end race_course_length_l24_24649


namespace binomial_12_6_eq_1848_l24_24815

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end binomial_12_6_eq_1848_l24_24815


namespace triangle_ratio_l24_24839

variables (A B C : ℝ) (a b c : ℝ)

theorem triangle_ratio (h_cosB : Real.cos B = 4/5)
    (h_a : a = 5)
    (h_area : 1/2 * a * c * Real.sin B = 12) :
    (a + c) / (Real.sin A + Real.sin C) = 25 / 3 :=
sorry

end triangle_ratio_l24_24839


namespace number_of_real_solutions_l24_24515

theorem number_of_real_solutions : 
  set.countable {x : ℝ | ∃ n : ℕ, (sqrt (123 - sqrt x) = n)} = 12 :=
by
  sorry

end number_of_real_solutions_l24_24515


namespace arithmetic_sequence_problem_l24_24080

theorem arithmetic_sequence_problem 
  (a_n b_n : ℕ → ℕ) 
  (S_n T_n : ℕ → ℕ) 
  (h1: ∀ n, S_n n = (n * (a_n n + a_n (n-1))) / 2)
  (h2: ∀ n, T_n n = (n * (b_n n + b_n (n-1))) / 2)
  (h3: ∀ n, (S_n n) / (T_n n) = (7 * n + 2) / (n + 3)):
  (a_n 4) / (b_n 4) = 51 / 10 := 
sorry

end arithmetic_sequence_problem_l24_24080


namespace bus_stops_per_hour_l24_24144

-- Define the speeds as constants
def speed_excluding_stoppages : ℝ := 60
def speed_including_stoppages : ℝ := 50

-- Formulate the main theorem
theorem bus_stops_per_hour :
  (1 - speed_including_stoppages / speed_excluding_stoppages) * 60 = 10 := 
by
  sorry

end bus_stops_per_hour_l24_24144


namespace find_z_proportional_l24_24087

theorem find_z_proportional (k : ℝ) (y x z : ℝ) 
  (h₁ : y = 8) (h₂ : x = 2) (h₃ : z = 4) (relationship : y = (k * x^2) / z)
  (y' x' z' : ℝ) (h₄ : y' = 72) (h₅ : x' = 4) : 
  z' = 16 / 9 := by
  sorry

end find_z_proportional_l24_24087


namespace output_sequence_value_l24_24203

theorem output_sequence_value (x y : Int) (seq : List (Int × Int))
  (h : (x, y) ∈ seq) (h_y : y = -10) : x = 32 :=
by
  sorry

end output_sequence_value_l24_24203


namespace probability_of_pink_l24_24380

-- Given conditions
variables (B P : ℕ) (h : (B : ℚ) / (B + P) = 3 / 7)

-- To prove
theorem probability_of_pink (h_pow : (B : ℚ) ^ 2 / (B + P) ^ 2 = 9 / 49) :
  (P : ℚ) / (B + P) = 4 / 7 :=
sorry

end probability_of_pink_l24_24380


namespace tan_alpha_sub_2pi_over_3_two_sin_sq_alpha_sub_cos_sq_alpha_l24_24360

variable (α : ℝ)

theorem tan_alpha_sub_2pi_over_3 (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) :
    Real.tan (α - 2 * π / 3) = 2 * Real.sqrt 3 :=
sorry

theorem two_sin_sq_alpha_sub_cos_sq_alpha (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) :
    2 * (Real.sin α) ^ 2 - (Real.cos α) ^ 2 = -43 / 52 :=
sorry

end tan_alpha_sub_2pi_over_3_two_sin_sq_alpha_sub_cos_sq_alpha_l24_24360


namespace evaluate_expression_l24_24489

theorem evaluate_expression : (2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9) := by
    sorry

end evaluate_expression_l24_24489


namespace problem_statement_l24_24971

def are_collinear (A B C : Point) : Prop := sorry -- Definition for collinearity should be expanded.
def area (A B C : Point) : ℝ := sorry -- Definition for area must be provided.

theorem problem_statement :
  ∀ n : ℕ, (n > 3) →
  (∃ (A : Fin n → Point) (r : Fin n → ℝ),
    (∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i → ¬ are_collinear (A i) (A j) (A k)) ∧
    (∀ i j k : Fin n, area (A i) (A j) (A k) = r i + r j + r k)) →
  n = 4 :=
by sorry

end problem_statement_l24_24971


namespace foldable_positions_are_7_l24_24312

-- Define the initial polygon with 6 congruent squares forming a cross shape
def initial_polygon : Prop :=
  -- placeholder definition, in practice, this would be a more detailed geometrical model
  sorry

-- Define the positions where an additional square can be attached (11 positions in total)
def position (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 11

-- Define the resulting polygon when an additional square is attached at position n
def resulting_polygon (n : ℕ) : Prop :=
  position n ∧ initial_polygon

-- Define the condition that a polygon can be folded into a cube with one face missing
def can_fold_to_cube_with_missing_face (p : Prop) : Prop := sorry

-- The theorem that needs to be proved
theorem foldable_positions_are_7 : 
  ∃ (positions : Finset ℕ), 
    positions.card = 7 ∧ 
    ∀ n ∈ positions, can_fold_to_cube_with_missing_face (resulting_polygon n) :=
  sorry

end foldable_positions_are_7_l24_24312


namespace find_point_of_intersection_l24_24610
noncomputable def point_of_intersection_curve_line : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^y = y^x ∧ y = x ∧ x = Real.exp 1 ∧ y = Real.exp 1

theorem find_point_of_intersection : point_of_intersection_curve_line :=
sorry

end find_point_of_intersection_l24_24610


namespace distance_walked_is_4_point_6_l24_24580

-- Define the number of blocks Sarah walked in each direction
def blocks_west : ℕ := 8
def blocks_south : ℕ := 15

-- Define the length of each block in miles
def block_length : ℚ := 1 / 5

-- Calculate the total number of blocks
def total_blocks : ℕ := blocks_west + blocks_south

-- Calculate the total distance walked in miles
def total_distance_walked : ℚ := total_blocks * block_length

-- Statement to prove the total distance walked is 4.6 miles
theorem distance_walked_is_4_point_6 : total_distance_walked = 4.6 := sorry

end distance_walked_is_4_point_6_l24_24580


namespace unshaded_area_eq_20_l24_24281

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

end unshaded_area_eq_20_l24_24281


namespace circle_reflection_l24_24591

variable (x₀ y₀ : ℝ)

def reflect_over_line_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem circle_reflection 
  (h₁ : x₀ = 8)
  (h₂ : y₀ = -3) :
  reflect_over_line_y_eq_neg_x (x₀, y₀) = (3, -8) := by
  sorry

end circle_reflection_l24_24591


namespace eccentricity_of_hyperbola_l24_24213

noncomputable def hyperbola_eccentricity : ℝ → ℝ → ℝ → ℝ
| p, a, b => 
  let c := p / 2
  let e := c / a
  have h₁ : 9 * e^2 - 12 * e^2 / (e^2 - 1) = 1 := sorry
  e

theorem eccentricity_of_hyperbola (p a b : ℝ) (hp : p > 0) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity p a b = (Real.sqrt 7 + 2) / 3 :=
sorry

end eccentricity_of_hyperbola_l24_24213


namespace length_of_AE_l24_24716

theorem length_of_AE (AD AE EB EF: ℝ) (h_AD: AD = 80) (h_EB: EB = 40) (h_EF: EF = 30) 
  (h_eq_area: 2 * ((EB * EF) + (1 / 2) * (ED * (AD - EF))) = AD * (AD - AE)) : AE = 15 :=
  sorry

end length_of_AE_l24_24716


namespace original_price_of_article_l24_24314

theorem original_price_of_article (P : ℝ) : 
  (P - 0.30 * P) * (1 - 0.20) = 1120 → P = 2000 :=
by
  intro h
  -- h represents the given condition for the problem
  sorry  -- proof will go here

end original_price_of_article_l24_24314


namespace intersection_points_l24_24018

noncomputable def y1 := 2*((7 + Real.sqrt 61)/2)^2 - 3*((7 + Real.sqrt 61)/2) + 1
noncomputable def y2 := 2*((7 - Real.sqrt 61)/2)^2 - 3*((7 - Real.sqrt 61)/2) + 1

theorem intersection_points :
  ∃ (x y : ℝ), (y = 2*x^2 - 3*x + 1) ∧ (y = x^2 + 4*x + 4) ∧
                ((x = (7 + Real.sqrt 61)/2 ∧ y = y1) ∨
                 (x = (7 - Real.sqrt 61)/2 ∧ y = y2)) :=
by
  sorry

end intersection_points_l24_24018


namespace x_eq_one_l24_24384

theorem x_eq_one (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (div_cond : ∀ n : ℕ, 0 < n → (2^n * y + 1) ∣ (x^(2^n) - 1)) : x = 1 := by
  sorry

end x_eq_one_l24_24384


namespace ca_co3_to_ca_cl2_l24_24185

theorem ca_co3_to_ca_cl2 (caCO3 HCl : ℕ) (main_reaction : caCO3 = 1 ∧ HCl = 2) : ∃ CaCl2, CaCl2 = 1 :=
by
  -- The proof of the theorem will go here.
  sorry

end ca_co3_to_ca_cl2_l24_24185


namespace roots_of_polynomial_l24_24186

theorem roots_of_polynomial (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end roots_of_polynomial_l24_24186


namespace symmetric_about_line_periodic_function_l24_24352

section
variable {α : Type*} [LinearOrderedField α]

-- First proof problem
theorem symmetric_about_line (f : α → α) (a : α) (h : ∀ x, f (a + x) = f (a - x)) : 
  ∀ x, f (2 * a - x) = f x :=
sorry

-- Second proof problem
theorem periodic_function (f : α → α) (a b : α) (ha : a ≠ b)
  (hsymm_a : ∀ x, f (2 * a - x) = f x)
  (hsymm_b : ∀ x, f (2 * b - x) = f x) : 
  ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x :=
sorry
end

end symmetric_about_line_periodic_function_l24_24352


namespace sides_and_diagonals_l24_24502

def number_of_sides_of_polygon (n : ℕ) :=
  180 * (n - 2) = 360 + (1 / 4 : ℤ) * 360

def number_of_diagonals_of_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem sides_and_diagonals : 
  (∃ n : ℕ, number_of_sides_of_polygon n ∧ n = 12) ∧ number_of_diagonals_of_polygon 12 = 54 :=
by {
  -- Proof will be filled in later
  sorry
}

end sides_and_diagonals_l24_24502


namespace find_c_plus_d_l24_24605

theorem find_c_plus_d (c d : ℝ) (h1 : 2 * c = 6) (h2 : c^2 - d = 4) : c + d = 8 := by
  sorry

end find_c_plus_d_l24_24605


namespace pq_false_l24_24845

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := x > 3 ↔ x^2 > 9
def q (a b : ℝ) : Prop := a^2 > b^2 ↔ a > b

-- Theorem to prove that p ∨ q is false given the conditions
theorem pq_false (x a b : ℝ) (hp : ¬ p x) (hq : ¬ q a b) : ¬ (p x ∨ q a b) :=
by
  sorry

end pq_false_l24_24845


namespace candy_count_l24_24979

def initial_candy : ℕ := 47
def eaten_candy : ℕ := 25
def sister_candy : ℕ := 40
def final_candy : ℕ := 62

theorem candy_count : initial_candy - eaten_candy + sister_candy = final_candy := 
by
  sorry

end candy_count_l24_24979


namespace new_stamps_ratio_l24_24606

theorem new_stamps_ratio (x : ℕ) (h1 : 7 * x = P) (h2 : 4 * x = Q)
  (h3 : P - 8 = 8 + (Q + 8)) : (P - 8) / gcd (P - 8) (Q + 8) = 6 ∧ (Q + 8) / gcd (P - 8) (Q + 8) = 5 :=
by
  sorry

end new_stamps_ratio_l24_24606


namespace julia_money_remaining_l24_24722

theorem julia_money_remaining 
  (initial_amount : ℝ)
  (tablet_percentage : ℝ)
  (phone_percentage : ℝ)
  (game_percentage : ℝ)
  (case_percentage : ℝ) 
  (final_money : ℝ) :
  initial_amount = 120 → 
  tablet_percentage = 0.45 → 
  phone_percentage = 1/3 → 
  game_percentage = 0.25 → 
  case_percentage = 0.10 → 
  final_money = initial_amount * (1 - tablet_percentage) * (1 - phone_percentage) * (1 - game_percentage) * (1 - case_percentage) →
  final_money = 29.70 :=
by
  intros
  sorry

end julia_money_remaining_l24_24722


namespace calculate_expression_l24_24663

theorem calculate_expression :
  (2 ^ (1/3) * 8 ^ (1/3) + 18 / (3 * 3) - 8 ^ (5/3)) = 2 ^ (4/3) - 30 :=
by
  sorry

end calculate_expression_l24_24663


namespace fixed_monthly_fee_december_l24_24956

theorem fixed_monthly_fee_december (x y : ℝ) 
    (h1 : x + y = 15.00) 
    (h2 : x + 2 + 3 * y = 25.40) : 
    x = 10.80 :=
by
  sorry

end fixed_monthly_fee_december_l24_24956


namespace sum_x_y_z_eq_3_or_7_l24_24000

theorem sum_x_y_z_eq_3_or_7 (x y z : ℝ) (h1 : x + y / z = 2) (h2 : y + z / x = 2) (h3 : z + x / y = 2) : x + y + z = 3 ∨ x + y + z = 7 :=
by
  sorry

end sum_x_y_z_eq_3_or_7_l24_24000


namespace min_xsq_ysq_zsq_l24_24242

noncomputable def min_value_x_sq_y_sq_z_sq (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : ℝ :=
  (x^2 + y^2 + z^2)

theorem min_xsq_ysq_zsq (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  min_value_x_sq_y_sq_z_sq x y z h = 40 / 7 :=
  sorry

end min_xsq_ysq_zsq_l24_24242


namespace quadratic_root_shift_l24_24727

theorem quadratic_root_shift (r s : ℝ)
    (hr : 2 * r^2 - 8 * r + 6 = 0)
    (hs : 2 * s^2 - 8 * s + 6 = 0)
    (h_sum_roots : r + s = 4)
    (h_prod_roots : r * s = 3)
    (b : ℝ) (c : ℝ)
    (h_b : b = - (r - 3 + s - 3))
    (h_c : c = (r - 3) * (s - 3)) : c = 0 :=
  by sorry

end quadratic_root_shift_l24_24727


namespace vector_rotation_correct_l24_24016

def vector_rotate_z_90 (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := v
  ( -y, x, z )

theorem vector_rotation_correct :
  vector_rotate_z_90 (3, -1, 4) = (-3, 0, 4) := 
by 
  sorry

end vector_rotation_correct_l24_24016


namespace minimum_discount_l24_24459

variable (C P : ℝ) (r x : ℝ)

def microwave_conditions := 
  C = 1000 ∧ 
  P = 1500 ∧ 
  r = 0.02 ∧ 
  P * (x / 10) ≥ C * (1 + r)

theorem minimum_discount : ∃ x, microwave_conditions C P r x ∧ x ≥ 6.8 :=
by 
  sorry

end minimum_discount_l24_24459


namespace loan_principal_and_repayment_amount_l24_24482

theorem loan_principal_and_repayment_amount (P R : ℝ) (r : ℝ) (years : ℕ) (total_interest : ℝ)
    (h1: r = 0.12)
    (h2: years = 3)
    (h3: total_interest = 5400)
    (h4: total_interest / years = R)
    (h5: R = P * r) :
    P = 15000 ∧ R = 1800 :=
sorry

end loan_principal_and_repayment_amount_l24_24482


namespace vacation_savings_l24_24003

-- Definitions
def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500
def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

-- Prove the amount set aside for vacation
theorem vacation_savings :
  let 
    total_income := parents_salary + grandmothers_pension + sons_scholarship,
    total_expenses := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses,
    surplus := total_income - total_expenses,
    deposit := (10 * surplus) / 100, 
    vacation_money := surplus - deposit
  in
    vacation_money = 16740 := by
      -- Calculation steps skipped; proof not required
      sorry

end vacation_savings_l24_24003


namespace cross_number_puzzle_hundreds_digit_l24_24056

theorem cross_number_puzzle_hundreds_digit :
  ∃ a b : ℕ, a ≥ 5 ∧ a ≤ 6 ∧ b = 3 ∧ (3^a / 100 = 7 ∨ 7^b / 100 = 7) :=
sorry

end cross_number_puzzle_hundreds_digit_l24_24056


namespace alpha_centauri_puzzle_l24_24151

open Nat

def max_number_count (A B N : ℕ) : ℕ :=
  let pairs_count := ((B - A) / N) * (N / 2)
  pairs_count + 1  -- Adding the single remainder

theorem alpha_centauri_puzzle :
  let A := 1353
  let B := 2134
  let N := 11
  max_number_count A B N = 356 :=
by 
  let A := 1353
  let B := 2134
  let N := 11
  -- Using the helper function max_number_count defined above
  have h : max_number_count A B N = 356 := 
   by sorry  -- skips detailed proof for illustrative purposes
  exact h


end alpha_centauri_puzzle_l24_24151


namespace monthly_savings_correct_l24_24004

-- Define each component of the income and expenses
def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500

def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

-- Define the main theorem
theorem monthly_savings_correct :
  let I := parents_salary + grandmothers_pension + sons_scholarship in
  let E := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses in
  let Surplus := I - E in
  let Deposit := (Surplus * 10) / 100 in
  let AmountSetAside := Surplus - Deposit in
  AmountSetAside = 16740 :=
by sorry

end monthly_savings_correct_l24_24004


namespace average_value_continuous_l24_24499

noncomputable def average_value (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (1 / (b - a)) * ∫ x in a..b, f x

theorem average_value_continuous (f : ℝ → ℝ) (a b : ℝ) (h : ContinuousOn f (Set.Icc a b)) :
  (average_value f a b) = (1 / (b - a)) * (∫ x in a..b, f x) :=
by
  sorry

end average_value_continuous_l24_24499


namespace circle_reflection_l24_24590

variable (x₀ y₀ : ℝ)

def reflect_over_line_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem circle_reflection 
  (h₁ : x₀ = 8)
  (h₂ : y₀ = -3) :
  reflect_over_line_y_eq_neg_x (x₀, y₀) = (3, -8) := by
  sorry

end circle_reflection_l24_24590


namespace alice_sales_above_goal_l24_24952

theorem alice_sales_above_goal :
  let quota := 1000
  let nike_price := 60
  let adidas_price := 45
  let reebok_price := 35
  let nike_sold := 8
  let adidas_sold := 6
  let reebok_sold := 9
  let total_sales := nike_price * nike_sold + adidas_price * adidas_sold + reebok_price * reebok_sold
  in total_sales - quota = 65 :=
by
  let quota := 1000
  let nike_price := 60
  let adidas_price := 45
  let reebok_price := 35
  let nike_sold := 8
  let adidas_sold := 6
  let reebok_sold := 9
  let total_sales := nike_price * nike_sold + adidas_price * adidas_sold + reebok_price * reebok_sold
  show total_sales - quota = 65 from sorry

end alice_sales_above_goal_l24_24952


namespace x_minus_25_is_perfect_square_l24_24506

theorem x_minus_25_is_perfect_square (n : ℕ) : 
  let x := 10^(2*n + 4) + 10^(n + 3) + 50 in
  ∃ k : ℕ, x - 25 = k^2 := by
  sorry

end x_minus_25_is_perfect_square_l24_24506


namespace bike_growth_equation_l24_24894

-- Declare the parameters
variables (b1 b3 : ℕ) (x : ℝ)
-- Define the conditions
def condition1 : b1 = 1000 := sorry
def condition2 : b3 = b1 + 440 := sorry

-- Define the proposition to be proved
theorem bike_growth_equation (cond1 : b1 = 1000) (cond2 : b3 = b1 + 440) :
  b1 * (1 + x)^2 = b3 :=
sorry

end bike_growth_equation_l24_24894


namespace find_n_l24_24096

noncomputable def n : ℕ := sorry -- Explicitly define n as a variable, but the value is not yet provided.

theorem find_n (h₁ : n > 0)
    (h₂ : Real.sqrt 3 > (n + 4) / (n + 1))
    (h₃ : Real.sqrt 3 < (n + 3) / n) : 
    n = 4 :=
sorry

end find_n_l24_24096


namespace correct_time_fraction_is_11_over_15_l24_24460

/-!
  We define the conditions where a digital clock displays the time incorrectly:
  - The clock mistakenly displays a 5 whenever it should display a 2.
  - The hours range from 1 to 12.
  - The minutes range from 00 to 59.
-/

def correct_hours_fraction : ℚ :=
  11 / 12  -- 11 correct hours out of 12

def correct_minutes_fraction : ℚ :=
  11 / 15  -- 44 correct minutes out of 60, simplified to 11/15

def correct_time_fraction : ℚ :=
  correct_hours_fraction * correct_minutes_fraction

theorem correct_time_fraction_is_11_over_15 :
  correct_time_fraction = 11 / 15 :=
by
  -- this is the statement we're proving, proof is omitted
  sorry

end correct_time_fraction_is_11_over_15_l24_24460


namespace hyperbola_eccentricity_l24_24833

theorem hyperbola_eccentricity (a b c e : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1 → (x, y) ≠ (3, -4))
  (h2 : b / a = 4 / 3)
  (h3 : b^2 = c^2 - a^2)
  (h4 : c / a = e):
  e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l24_24833


namespace omega_bound_l24_24989

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - Real.sin (ω * x)

theorem omega_bound (ω : ℝ) (h₁ : ω > 0)
  (h₂ : ∀ x : ℝ, -π / 2 < x ∧ x < π / 2 → (f ω x) ≤ (f ω (-π / 2))) :
  ω ≤ 1 / 2 :=
sorry

end omega_bound_l24_24989


namespace range_of_t_l24_24406

theorem range_of_t (a b t : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 2 * a + b = 1) 
    (h_ineq : 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1 / 2):
    t = Real.sqrt 2 / 2 :=
sorry

end range_of_t_l24_24406


namespace distinct_real_x_l24_24524

theorem distinct_real_x (x : ℝ) :
  (∃! x, ∃ (s : ℝ), s ∈ set.Icc 0 11 ∧ s = real.sqrt (123 - real.sqrt x)) =
  12 := sorry

end distinct_real_x_l24_24524


namespace find_number_l24_24299

theorem find_number (x : ℝ) (h : 0.60 * 50 = 0.45 * x + 16.5) : x = 30 :=
by
  sorry

end find_number_l24_24299


namespace geometric_sum_thm_l24_24759

variable (S : ℕ → ℝ)

theorem geometric_sum_thm (h1 : S n = 48) (h2 : S (2 * n) = 60) : S (3 * n) = 63 :=
sorry

end geometric_sum_thm_l24_24759


namespace arithmetic_sequence_21st_term_l24_24262

theorem arithmetic_sequence_21st_term (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 13) (h3 : a 3 = 23) :
  a 21 = 203 :=
by
  sorry

end arithmetic_sequence_21st_term_l24_24262


namespace route_B_is_faster_by_7_5_minutes_l24_24880

def distance_A := 10  -- miles
def normal_speed_A := 30  -- mph
def construction_distance_A := 2  -- miles
def construction_speed_A := 15  -- mph
def distance_B := 8  -- miles
def normal_speed_B := 40  -- mph
def school_zone_distance_B := 1  -- miles
def school_zone_speed_B := 10  -- mph

noncomputable def time_for_normal_speed_A : ℝ := (distance_A - construction_distance_A) / normal_speed_A * 60  -- minutes
noncomputable def time_for_construction_A : ℝ := construction_distance_A / construction_speed_A * 60  -- minutes
noncomputable def total_time_A : ℝ := time_for_normal_speed_A + time_for_construction_A

noncomputable def time_for_normal_speed_B : ℝ := (distance_B - school_zone_distance_B) / normal_speed_B * 60  -- minutes
noncomputable def time_for_school_zone_B : ℝ := school_zone_distance_B / school_zone_speed_B * 60  -- minutes
noncomputable def total_time_B : ℝ := time_for_normal_speed_B + time_for_school_zone_B

theorem route_B_is_faster_by_7_5_minutes : total_time_B + 7.5 = total_time_A := by
  sorry

end route_B_is_faster_by_7_5_minutes_l24_24880


namespace ratio_of_areas_l24_24679

noncomputable def area_ratio (a : ℝ) : ℝ :=
  let side_triangle : ℝ := a
  let area_triangle : ℝ := (1 / 2) * side_triangle * side_triangle
  let height_rhombus : ℝ := side_triangle * Real.sin (Real.pi / 3)
  let area_rhombus : ℝ := height_rhombus * side_triangle
  area_rhombus / area_triangle

theorem ratio_of_areas (a : ℝ) (h : a > 0) : area_ratio a = 3 := by
  -- The proof would be here
  sorry

end ratio_of_areas_l24_24679


namespace rain_difference_l24_24377

variable (R : ℝ) -- Amount of rain in the second hour
variable (r1 : ℝ) -- Amount of rain in the first hour

-- Conditions
axiom h1 : r1 = 5
axiom h2 : R + r1 = 22

-- Theorem to prove
theorem rain_difference (R r1 : ℝ) (h1 : r1 = 5) (h2 : R + r1 = 22) : R - 2 * r1 = 7 := by
  sorry

end rain_difference_l24_24377


namespace find_k_and_b_l24_24848

noncomputable def setA := {p : ℝ × ℝ | p.2^2 - p.1 - 1 = 0}
noncomputable def setB := {p : ℝ × ℝ | 4 * p.1^2 + 2 * p.1 - 2 * p.2 + 5 = 0}
noncomputable def setC (k b : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + b}

theorem find_k_and_b (k b : ℕ) : 
  (setA ∪ setB) ∩ setC k b = ∅ ↔ (k = 1 ∧ b = 2) := 
sorry

end find_k_and_b_l24_24848


namespace pages_left_to_read_l24_24100

def total_pages : ℕ := 17
def pages_read : ℕ := 11

theorem pages_left_to_read : total_pages - pages_read = 6 := by
  sorry

end pages_left_to_read_l24_24100


namespace exists_f_lt_zero_l24_24343

noncomputable def f (x : ℝ) : ℝ := Real.sin x - x

theorem exists_f_lt_zero : ∃ x ∈ (set.Ioo 0 (Real.pi / 2)), f x < 0 :=
by
  sorry

end exists_f_lt_zero_l24_24343


namespace polynomial_coeff_sum_l24_24215

theorem polynomial_coeff_sum (a_4 a_3 a_2 a_1 a_0 : ℝ) :
  (∀ x: ℝ, (x - 1) ^ 4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_4 - a_3 + a_2 - a_1 + a_0 = 16 :=
by
  intro h
  sorry

end polynomial_coeff_sum_l24_24215


namespace charges_needed_to_vacuum_house_l24_24260

-- Conditions definitions
def battery_last_minutes : ℕ := 10
def vacuum_time_per_room : ℕ := 4
def number_of_bedrooms : ℕ := 3
def number_of_kitchens : ℕ := 1
def number_of_living_rooms : ℕ := 1

-- Question (proof problem statement)
theorem charges_needed_to_vacuum_house :
  ((number_of_bedrooms + number_of_kitchens + number_of_living_rooms) * vacuum_time_per_room) / battery_last_minutes = 2 :=
by
  sorry

end charges_needed_to_vacuum_house_l24_24260


namespace find_a_l24_24968

theorem find_a (r s a : ℚ) (h₁ : 2 * r * s = 18) (h₂ : s^2 = 16) (h₃ : a = r^2) : 
  a = 81 / 16 := 
sorry

end find_a_l24_24968


namespace b_bounded_l24_24870

open Real

-- Define sequences of real numbers
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry

-- Define initial conditions and properties
axiom a0_gt_half : a 0 > 1/2
axiom a_non_decreasing : ∀ n : ℕ, a (n + 1) ≥ a n
axiom b_recursive : ∀ n : ℕ, b (n + 1) = a n * (b n + b (n + 2))

-- Prove the sequence (b_n) is bounded
theorem b_bounded : ∃ M : ℝ, ∀ n : ℕ, b n ≤ M :=
by
  sorry

end b_bounded_l24_24870


namespace find_roots_l24_24066

theorem find_roots : 
  ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (x + 2) = 0 ↔ (x = -2 ∨ x = 2 ∨ x = 3) := by
  sorry

end find_roots_l24_24066


namespace net_investment_change_l24_24442

variable (I : ℝ)

def first_year_increase (I : ℝ) : ℝ := I * 1.75
def second_year_decrease (W : ℝ) : ℝ := W * 0.70

theorem net_investment_change : 
  let I' := first_year_increase 100 
  let I'' := second_year_decrease I' 
  I'' - 100 = 22.50 :=
by
  sorry

end net_investment_change_l24_24442


namespace james_total_distance_l24_24867

structure Segment where
  speed : ℝ -- speed in mph
  time : ℝ -- time in hours

def totalDistance (segments : List Segment) : ℝ :=
  segments.foldr (λ seg acc => seg.speed * seg.time + acc) 0

theorem james_total_distance :
  let segments := [
    Segment.mk 30 0.5,
    Segment.mk 60 0.75,
    Segment.mk 75 1.5,
    Segment.mk 60 2
  ]
  totalDistance segments = 292.5 :=
by
  sorry

end james_total_distance_l24_24867


namespace minimum_stamps_satisfying_congruences_l24_24891

theorem minimum_stamps_satisfying_congruences (n : ℕ) :
  (n % 4 = 3) ∧ (n % 5 = 2) ∧ (n % 7 = 1) → n = 107 :=
by
  sorry

end minimum_stamps_satisfying_congruences_l24_24891


namespace density_of_cone_in_mercury_l24_24462

variable {h : ℝ} -- height of the cone
variable {ρ : ℝ} -- density of the cone
variable {ρ_m : ℝ} -- density of the mercury
variable {k : ℝ} -- proportion factor

-- Archimedes' principle applied to the cone floating in mercury
theorem density_of_cone_in_mercury (stable_eq: ∀ (V V_sub: ℝ), (ρ * V) = (ρ_m * V_sub))
(h_sub: h / k = (k - 1) / k) :
  ρ = ρ_m * ((k - 1)^3 / k^3) :=
by
  sorry

end density_of_cone_in_mercury_l24_24462


namespace boys_camp_total_l24_24146

theorem boys_camp_total (T : ℕ) 
  (h1 : 0.20 * T = (0.20 : ℝ) * T) 
  (h2 : (0.30 : ℝ) * (0.20 * T) = (0.30 : ℝ) * (0.20 * T)) 
  (h3 : (0.70 : ℝ) * (0.20 * T) = 63) :
  T = 450 :=
by
  sorry

end boys_camp_total_l24_24146


namespace border_area_is_72_l24_24164

def livingRoomLength : ℝ := 12
def livingRoomWidth : ℝ := 10
def borderWidth : ℝ := 2

def livingRoomArea : ℝ := livingRoomLength * livingRoomWidth
def carpetLength : ℝ := livingRoomLength - 2 * borderWidth
def carpetWidth : ℝ := livingRoomWidth - 2 * borderWidth
def carpetArea : ℝ := carpetLength * carpetWidth
def borderArea : ℝ := livingRoomArea - carpetArea

theorem border_area_is_72 : borderArea = 72 := 
by
  sorry

end border_area_is_72_l24_24164


namespace tree_height_fraction_l24_24925

theorem tree_height_fraction :
  ∀ (initial_height growth_per_year : ℝ),
  initial_height = 4 ∧ growth_per_year = 0.5 →
  ((initial_height + 6 * growth_per_year) - (initial_height + 4 * growth_per_year)) / (initial_height + 4 * growth_per_year) = 1 / 6 :=
by
  intros initial_height growth_per_year h
  rcases h with ⟨h1, h2⟩
  sorry

end tree_height_fraction_l24_24925


namespace group_members_l24_24160

theorem group_members (n : ℕ) (hn : n * n = 1369) : n = 37 :=
by
  sorry

end group_members_l24_24160


namespace average_is_0_1667X_plus_3_l24_24058

noncomputable def average_of_three_numbers (X Y Z : ℝ) : ℝ := (X + Y + Z) / 3

theorem average_is_0_1667X_plus_3 (X Y Z : ℝ) 
  (h1 : 2001 * Z - 4002 * X = 8008) 
  (h2 : 2001 * Y + 5005 * X = 10010) : 
  average_of_three_numbers X Y Z = 0.1667 * X + 3 := 
sorry

end average_is_0_1667X_plus_3_l24_24058


namespace min_buses_l24_24313

theorem min_buses (n : ℕ) : (47 * n >= 625) → (n = 14) :=
by {
  -- Proof is omitted since the problem only asks for the Lean statement, not the solution steps.
  sorry
}

end min_buses_l24_24313


namespace triangle_inequality_1_triangle_inequality_2_l24_24872

variable (a b c : ℝ)

theorem triangle_inequality_1 (h1 : a + b + c = 2) (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) (h5 : a ≤ 1) (h6 : b ≤ 1) (h7 : c ≤ 1) : 
  a * b * c + 28 / 27 ≥ a * b + b * c + c * a :=
by
  sorry

theorem triangle_inequality_2 (h1 : a + b + c = 2) (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) (h5 : a ≤ 1) (h6 : b ≤ 1) (h7 : c ≤ 1) : 
  a * b + b * c + c * a ≥ a * b * c + 1 :=
by
  sorry

end triangle_inequality_1_triangle_inequality_2_l24_24872


namespace derivative_of_reciprocal_at_one_l24_24006

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem derivative_of_reciprocal_at_one : (deriv f 1) = -1 :=
by {
    sorry
}

end derivative_of_reciprocal_at_one_l24_24006


namespace nell_has_cards_left_l24_24401

def initial_cards : ℕ := 242
def cards_given_away : ℕ := 136

theorem nell_has_cards_left :
  initial_cards - cards_given_away = 106 :=
by
  sorry

end nell_has_cards_left_l24_24401


namespace exists_func_satisfies_condition_l24_24277

theorem exists_func_satisfies_condition :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2*x) = abs (x + 1) :=
sorry

end exists_func_satisfies_condition_l24_24277


namespace cost_price_of_table_l24_24036

theorem cost_price_of_table (SP : ℝ) (CP : ℝ) (h1 : SP = 1.20 * CP) (h2 : SP = 3600) : CP = 3000 :=
by
  sorry

end cost_price_of_table_l24_24036


namespace correct_statements_l24_24760

-- Definitions based on the conditions and question
def S (n : ℕ) : ℤ := -n^2 + 7 * n + 1

-- Definition of the sequence an
def a (n : ℕ) : ℤ := 
  if n = 1 then 7 
  else S n - S (n - 1)

-- Theorem statements based on the correct answers derived from solution
theorem correct_statements :
  (∀ n : ℕ, n > 4 → a n < 0) ∧ (S 3 = S 4 ∧ (∀ m : ℕ, S m ≤ S 3)) :=
by {
  sorry
}

end correct_statements_l24_24760


namespace reflection_of_point_l24_24599

def reflect_across_y_neg_x (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (y, x)

theorem reflection_of_point
  (P : ℝ × ℝ)
  (h : P = (8, -3)) :
  reflect_across_y_neg_x P = (3, -8) :=
by
  rw [h]
  sorry

end reflection_of_point_l24_24599


namespace nat_number_36_sum_of_digits_l24_24193

-- Define the function that represents the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main statement
theorem nat_number_36_sum_of_digits (x : ℕ) (hx : x = 36 * sum_of_digits x) : x = 324 ∨ x = 648 := 
by 
  sorry

end nat_number_36_sum_of_digits_l24_24193


namespace dispersion_is_variance_l24_24138

def Mean := "Mean"
def Variance := "Variance"
def Median := "Median"
def Mode := "Mode"

def dispersion_measure := Variance

theorem dispersion_is_variance (A B C D : String) (hA : A = Mean) (hB : B = Variance) (hC : C = Median) (hD : D = Mode) : 
  dispersion_measure = B :=
by
  rw [hB]
  exact sorry

end dispersion_is_variance_l24_24138


namespace problem_statement_l24_24243

theorem problem_statement (x1 x2 x3 : ℝ) 
  (h1 : x1 < x2)
  (h2 : x2 < x3)
  (h3 : (45*x1^3 - 4050*x1^2 - 4 = 0) ∧ 
        (45*x2^3 - 4050*x2^2 - 4 = 0) ∧ 
        (45*x3^3 - 4050*x3^2 - 4 = 0)) :
  x2 * (x1 + x3) = 0 :=
by
  sorry

end problem_statement_l24_24243


namespace parabola_equation_l24_24701

theorem parabola_equation (d : ℝ) (p : ℝ) (x y : ℝ) (h1 : d = 2) (h2 : y = 2) (h3 : x = 1) :
  y^2 = 4 * x :=
sorry

end parabola_equation_l24_24701


namespace total_visible_surface_area_l24_24826

-- Define the cubes by their volumes
def volumes : List ℝ := [1, 8, 27, 125, 216, 343, 512, 729]

-- Define the arrangement information as specified
def arrangement_conditions : Prop :=
  ∃ (s8 s7 s6 s5 s4 s3 s2 s1 : ℝ),
    s8^3 = 729 ∧ s7^3 = 512 ∧ s6^3 = 343 ∧ s5^3 = 216 ∧
    s4^3 = 125 ∧ s3^3 = 27 ∧ s2^3 = 8 ∧ s1^3 = 1 ∧
    5 * s8^2 + (5 * s7^2 + 4 * s6^2 + 4 * s5^2) + 
    (5 * s4^2 + 4 * s3^2 + 5 * s2^2 + 4 * s1^2) = 1250

-- The proof statement
theorem total_visible_surface_area : arrangement_conditions → 1250 = 1250 := by
  intro _ -- this stands for not proving the condition, taking it as assumption
  exact rfl


end total_visible_surface_area_l24_24826


namespace points_earned_l24_24850

-- Definition of the conditions explicitly stated in the problem
def points_per_bag := 8
def total_bags := 4
def bags_not_recycled := 2

-- Calculation of bags recycled
def bags_recycled := total_bags - bags_not_recycled

-- The main theorem stating the proof equivalent
theorem points_earned : points_per_bag * bags_recycled = 16 := 
by
  sorry

end points_earned_l24_24850


namespace part1_part2_part3_l24_24346

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + 2)

theorem part1 (h_odd : ∀ x : ℝ, f x b = -f (-x) b) : b = 1 :=
sorry

theorem part2 (h_b : b = 1) : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 1 > f x2 1 :=
sorry

theorem part3 (h_monotonic : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 1 > f x2 1) 
  : ∀ t : ℝ, f (t^2 - 2 * t) 1 + f (2 * t^2 - k) 1 < 0 → k < -1/3 :=
sorry

end part1_part2_part3_l24_24346


namespace max_distinct_terms_degree_6_l24_24131

-- Step 1: Define the variables and conditions
def polynomial_max_num_terms (deg : ℕ) (vars : ℕ) : ℕ :=
  Nat.choose (deg + vars - 1) (vars - 1)

-- Step 2: State the specific problem
theorem max_distinct_terms_degree_6 :
  polynomial_max_num_terms 6 5 = 210 :=
by
  sorry

end max_distinct_terms_degree_6_l24_24131


namespace expected_final_set_size_l24_24733

noncomputable def final_expected_set_size : ℚ :=
  let n := 8
  let initial_size := 255
  let steps := initial_size - 1
  n * (2^7 / initial_size)

theorem expected_final_set_size :
  final_expected_set_size = 1024 / 255 :=
by
  sorry

end expected_final_set_size_l24_24733


namespace Lizzy_money_after_loan_l24_24398

theorem Lizzy_money_after_loan :
  let initial_savings := 30
  let loaned_amount := 15
  let interest_rate := 0.20
  let interest := loaned_amount * interest_rate
  let total_amount_returned := loaned_amount + interest
  let remaining_money := initial_savings - loaned_amount
  let total_money := remaining_money + total_amount_returned
  total_money = 33 :=
by
  sorry

end Lizzy_money_after_loan_l24_24398


namespace circle_diameter_l24_24830

-- The problem statement in Lean 4

theorem circle_diameter
  (d α β : ℝ) :
  ∃ r: ℝ,
  r * 2 = d * (Real.sin α) * (Real.sin β) / (Real.cos ((α + β) / 2) * (Real.sin ((α - β) / 2))) :=
sorry

end circle_diameter_l24_24830


namespace marathon_speed_ratio_l24_24735

theorem marathon_speed_ratio (M D : ℝ) (J : ℝ) (H1 : D = 9) (H2 : J = 4/3 * M) (H3 : M + J + D = 23) :
  D / M = 3 / 2 :=
by
  sorry

end marathon_speed_ratio_l24_24735


namespace diagonals_in_nine_sided_polygon_l24_24995

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l24_24995


namespace divisible_by_91_l24_24407

theorem divisible_by_91 (n : ℕ) : 91 ∣ (5^n * (5^n + 1) - 6^n * (3^n + 2^n)) := 
by 
  sorry

end divisible_by_91_l24_24407


namespace function_takes_negative_values_l24_24296

def f (x a : ℝ) : ℝ := x^2 - a * x + 1

theorem function_takes_negative_values {a : ℝ} :
  (∃ x : ℝ, f x a < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end function_takes_negative_values_l24_24296


namespace expression_equals_12_l24_24029

-- Define the values of a, b, c, and k
def a : ℤ := 10
def b : ℤ := 15
def c : ℤ := 3
def k : ℤ := 2

-- Define the expression to be evaluated
def expr : ℤ := (a - (b - k * c)) - ((a - b) - k * c)

-- Prove that the expression equals 12
theorem expression_equals_12 : expr = 12 :=
by
  -- The proof will go here, leaving a placeholder for now
  sorry

end expression_equals_12_l24_24029


namespace solution_exists_l24_24440

theorem solution_exists (x y z a : ℝ) :
  (x + y + 2 * z = 4 * (a^2 + 1)) ∧ (z^2 - x * y = a^2) →
  ((x = a^2 + a + 1 ∧ y = a^2 - a + 1 ∧ z = a^2 + 1) ∨ 
   (x = a^2 - a + 1 ∧ y = a^2 + a + 1 ∧ z = a^2 + 1)) :=
by {
  intros h,
  sorry
}

end solution_exists_l24_24440


namespace pascal_sum_of_squares_of_interior_l24_24379

theorem pascal_sum_of_squares_of_interior (eighth_row_interior : List ℕ) 
    (h : eighth_row_interior = [7, 21, 35, 35, 21, 7]) : 
    (eighth_row_interior.map (λ x => x * x)).sum = 3430 := 
by
  sorry

end pascal_sum_of_squares_of_interior_l24_24379


namespace least_positive_integer_mod_conditions_l24_24023

theorem least_positive_integer_mod_conditions :
  ∃ N : ℕ, (N % 4 = 3) ∧ (N % 5 = 4) ∧ (N % 6 = 5) ∧ (N % 7 = 6) ∧ (N % 11 = 10) ∧ N = 4619 :=
by
  sorry

end least_positive_integer_mod_conditions_l24_24023


namespace function_identity_l24_24095

theorem function_identity (f : ℕ → ℕ) (h₁ : ∀ n, 0 < f n)
  (h₂ : ∀ n, f (n + 1) > f (f n)) :
∀ n, f n = n :=
sorry

end function_identity_l24_24095


namespace colorable_graph_l24_24371

variable (V : Type) [Fintype V] [DecidableEq V] (E : V → V → Prop) [DecidableRel E]

/-- Each city has at least one road leading out of it -/
def has_one_road (v : V) : Prop := ∃ w : V, E v w

/-- No city is connected by roads to all other cities -/
def not_connected_to_all (v : V) : Prop := ¬ ∀ w : V, E v w ↔ w ≠ v

/-- A set of cities D is dominating if every city not in D is connected by a road to at least one city in D -/
def is_dominating_set (D : Finset V) : Prop :=
  ∀ v : V, v ∉ D → ∃ d ∈ D, E v d

noncomputable def dominating_set_min_card (k : ℕ) : Prop :=
  ∀ D : Finset V, is_dominating_set V E D → D.card ≥ k

/-- Prove that the graph can be colored using 2001 - k colors such that no two adjacent vertices share the same color -/
theorem colorable_graph (k : ℕ) (hk : dominating_set_min_card V E k) :
    ∃ (colors : V → Fin (2001 - k)), ∀ v w : V, E v w → colors v ≠ colors w := 
by 
  sorry

end colorable_graph_l24_24371


namespace quadratic_ineq_solution_range_l24_24691

theorem quadratic_ineq_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2*x^2 - 8*x - 4 - a > 0) ↔ a < -4 :=
by
  sorry

end quadratic_ineq_solution_range_l24_24691


namespace parabola_focus_hyperbola_equation_l24_24154

-- Problem 1
theorem parabola_focus (p : ℝ) (h₀ : p > 0) (h₁ : 2 * p - 0 - 4 = 0) : p = 2 :=
by
  sorry

-- Problem 2
theorem hyperbola_equation (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : b / a = 3 / 4) (h₃ : a^2 / a = 16 / 5) (h₄ : a^2 + b^2 = 1) :
  (x^2 / 16) - (y^2 / 9) = 1 :=
by
  sorry

end parabola_focus_hyperbola_equation_l24_24154


namespace train_speed_is_45_km_per_hr_l24_24793

/-- 
  Given the length of the train (135 m), the time to cross a bridge (30 s),
  and the length of the bridge (240 m), we want to prove that the speed of the 
  train is 45 km/hr.
--/

def length_of_train : ℕ := 135
def time_to_cross_bridge : ℕ := 30
def length_of_bridge : ℕ := 240
def speed_of_train_in_km_per_hr (L_t t L_b : ℕ) : ℕ := 
  ((L_t + L_b) * 36 / 10) / t

theorem train_speed_is_45_km_per_hr : 
  speed_of_train_in_km_per_hr length_of_train time_to_cross_bridge length_of_bridge = 45 :=
by 
  -- Assuming the calculations are correct, the expected speed is provided here directly
  sorry

end train_speed_is_45_km_per_hr_l24_24793


namespace gcd_180_270_eq_90_l24_24021

-- Problem Statement
theorem gcd_180_270_eq_90 : Nat.gcd 180 270 = 90 := 
by 
  sorry

end gcd_180_270_eq_90_l24_24021


namespace find_natural_numbers_eq_36_sum_of_digits_l24_24190

-- Define the sum of digits function
def sum_of_digits (x : ℕ) : ℕ := 
  if x = 0 then 0
  else sum_of_digits (x / 10) + (x % 10)

-- Lean theorem statement proving the given problem
theorem find_natural_numbers_eq_36_sum_of_digits :
  {x : ℕ | x = 36 * (sum_of_digits x)} = {324, 648} :=
sorry

end find_natural_numbers_eq_36_sum_of_digits_l24_24190


namespace projects_count_minimize_time_l24_24448

-- Define the conditions as given in the problem
def total_projects := 15
def energy_transfer_condition (x y : ℕ) : Prop := x = 2 * y - 3

-- Define question 1 as a proof problem
theorem projects_count (x y : ℕ) (h1 : x + y = total_projects) (h2 : energy_transfer_condition x y) :
  x = 9 ∧ y = 6 :=
by
  sorry

-- Define conditions for question 2
def average_time (energy_transfer_time leaping_gate_time : ℕ) (m n total_time : ℕ) : Prop :=
  total_time = 6 * m + 8 * n

-- Define additional conditions needed for Question 2 regarding time
theorem minimize_time (m n total_time : ℕ)
  (h1 : m + n = 10)
  (h2 : 10 - m > n)
  (h3 : average_time 6 8 m n total_time)
  (h4 : m = 6) :
  total_time = 68 :=
by
  sorry

end projects_count_minimize_time_l24_24448


namespace invalid_prob_distribution_D_l24_24386

noncomputable def sum_of_probs_A : ℚ :=
  0 + 1/2 + 0 + 0 + 1/2

noncomputable def sum_of_probs_B : ℚ :=
  0.1 + 0.2 + 0.3 + 0.4

noncomputable def sum_of_probs_C (p : ℚ) (hp : 0 ≤ p ∧ p ≤ 1) : ℚ :=
  p + (1 - p)

noncomputable def sum_of_probs_D : ℚ :=
  (1/1*2) + (1/2*3) + (1/3*4) + (1/4*5) + (1/5*6) + (1/6*7) + (1/7*8)

theorem invalid_prob_distribution_D :
  sum_of_probs_D ≠ 1 := sorry

end invalid_prob_distribution_D_l24_24386


namespace william_ends_with_18_tickets_l24_24139

-- Define the initial number of tickets
def initialTickets : ℕ := 15

-- Define the tickets bought
def ticketsBought : ℕ := 3

-- Prove the total number of tickets William ends with
theorem william_ends_with_18_tickets : initialTickets + ticketsBought = 18 := by
  sorry

end william_ends_with_18_tickets_l24_24139


namespace rank_classmates_l24_24958

-- Definitions of the conditions
def emma_tallest (emma david fiona : ℕ) : Prop := emma > david ∧ emma > fiona
def fiona_not_shortest (david emma fiona : ℕ) : Prop := david > fiona ∧ emma > fiona
def david_not_tallest (david emma fiona : ℕ) : Prop := emma > david ∧ fiona > david

def exactly_one_true (david emma fiona : ℕ) : Prop :=
  (emma_tallest emma david fiona ∧ ¬fiona_not_shortest david emma fiona ∧ ¬david_not_tallest david emma fiona) ∨
  (¬emma_tallest emma david fiona ∧ fiona_not_shortest david emma fiona ∧ ¬david_not_tallest david emma fiona) ∨
  (¬emma_tallest emma david fiona ∧ ¬fiona_not_shortest david emma fiona ∧ david_not_tallest david emma fiona)

-- The final proof statement
theorem rank_classmates (david emma fiona : ℕ) (h : exactly_one_true david emma fiona) : david > fiona ∧ fiona > emma :=
  sorry

end rank_classmates_l24_24958


namespace least_integer_to_multiple_of_3_l24_24770

theorem least_integer_to_multiple_of_3 : ∃ n : ℕ, n > 0 ∧ (527 + n) % 3 = 0 ∧ ∀ m : ℕ, m > 0 → (527 + m) % 3 = 0 → m ≥ n :=
sorry

end least_integer_to_multiple_of_3_l24_24770


namespace even_function_a_is_neg_one_l24_24859

-- Define f and the condition that it is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the given function f(x) = (x-1)*(x-a)
def f (a x : ℝ) : ℝ := (x - 1) * (x - a)

-- Statement to prove that if f is an even function, then a must be -1
theorem even_function_a_is_neg_one (a : ℝ) :
  is_even (f a) → a = -1 :=
by 
  intro h,
  sorry

end even_function_a_is_neg_one_l24_24859


namespace length_of_other_train_is_correct_l24_24301

noncomputable def length_of_other_train
  (l1 : ℝ) -- length of the first train in meters
  (s1 : ℝ) -- speed of the first train in km/hr
  (s2 : ℝ) -- speed of the second train in km/hr
  (t : ℝ)  -- time in seconds
  (h1 : l1 = 500)
  (h2 : s1 = 240)
  (h3 : s2 = 180)
  (h4 : t = 12) :
  ℝ :=
  let s1_m_s := s1 * 1000 / 3600
  let s2_m_s := s2 * 1000 / 3600
  let relative_speed := s1_m_s + s2_m_s
  let total_distance := relative_speed * t
  total_distance - l1

theorem length_of_other_train_is_correct :
  length_of_other_train 500 240 180 12 rfl rfl rfl rfl = 900 := sorry

end length_of_other_train_is_correct_l24_24301


namespace cuboid_height_l24_24614

-- Given conditions
def volume_cuboid : ℝ := 1380 -- cubic meters
def base_area_cuboid : ℝ := 115 -- square meters

-- Prove that the height of the cuboid is 12 meters
theorem cuboid_height : volume_cuboid / base_area_cuboid = 12 := by
  sorry

end cuboid_height_l24_24614


namespace max_writers_and_editors_l24_24451

theorem max_writers_and_editors (total_people writers editors x : ℕ) (h_total_people : total_people = 100)
(h_writers : writers = 40) (h_editors : editors > 38) (h_both : 2 * x + (writers + editors - x) = total_people) :
x ≤ 21 := sorry

end max_writers_and_editors_l24_24451


namespace real_roots_if_and_only_if_m_leq_5_l24_24364

theorem real_roots_if_and_only_if_m_leq_5 (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x + 1 = 0) ↔ m ≤ 5 :=
by
  sorry

end real_roots_if_and_only_if_m_leq_5_l24_24364


namespace construction_cost_is_correct_l24_24041

def land_cost (cost_per_sqm : ℕ) (area : ℕ) : ℕ :=
  cost_per_sqm * area

def bricks_cost (cost_per_1000 : ℕ) (quantity : ℕ) : ℕ :=
  (cost_per_1000 * quantity) / 1000

def roof_tiles_cost (cost_per_tile : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_tile * quantity

def cement_bags_cost (cost_per_bag : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_bag * quantity

def wooden_beams_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def steel_bars_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def electrical_wiring_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def plumbing_pipes_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def total_cost : ℕ :=
  land_cost 60 2500 +
  bricks_cost 120 15000 +
  roof_tiles_cost 12 800 +
  cement_bags_cost 8 250 +
  wooden_beams_cost 25 1000 +
  steel_bars_cost 15 500 +
  electrical_wiring_cost 2 2000 +
  plumbing_pipes_cost 4 3000

theorem construction_cost_is_correct : total_cost = 212900 :=
  by
    sorry

end construction_cost_is_correct_l24_24041


namespace max_radius_of_circle_touching_graph_l24_24223

theorem max_radius_of_circle_touching_graph :
  ∃ r : ℝ, (∀ (x : ℝ), (x^2 + (x^4 - r)^2 = r^2) → r ≤ (3 * (2:ℝ)^(1/3)) / 4) ∧
           r = (3 * (2:ℝ)^(1/3)) / 4 :=
by
  sorry

end max_radius_of_circle_touching_graph_l24_24223


namespace intersecting_chords_second_length_l24_24450

theorem intersecting_chords_second_length (a b : ℕ) (k : ℕ) 
  (h_a : a = 12) (h_b : b = 18) (h_ratio : k ^ 2 = (a * b) / 24) 
  (x y : ℕ) (h_x : x = 3 * k) (h_y : y = 8 * k) :
  x + y = 33 :=
by
  sorry

end intersecting_chords_second_length_l24_24450


namespace probability_of_3_tails_in_8_flips_l24_24491

open ProbabilityTheory

/-- The probability of getting exactly 3 tails out of 8 flips of an unfair coin, where the probability of tails is 4/5 and the probability of heads is 1/5, is 3584/390625. -/
theorem probability_of_3_tails_in_8_flips :
  let p_heads := 1 / 5
  let p_tails := 4 / 5
  let n_trials := 8
  let k_successes := 3
  let binomial_coefficient := Nat.choose n_trials k_successes
  let probability := binomial_coefficient * (p_tails ^ k_successes) * (p_heads ^ (n_trials - k_successes))
  probability = (3584 : ℚ) / 390625 := 
by 
  sorry

end probability_of_3_tails_in_8_flips_l24_24491


namespace points_per_member_correct_l24_24949

noncomputable def points_per_member (total_members: ℝ) (absent_members: ℝ) (total_points: ℝ) :=
  (total_points / (total_members - absent_members))

theorem points_per_member_correct:
  points_per_member 5.0 2.0 6.0 = 2.0 :=
by 
  sorry

end points_per_member_correct_l24_24949


namespace initial_money_l24_24408

def cost_of_game : Nat := 47
def cost_of_toy : Nat := 7
def number_of_toys : Nat := 3

theorem initial_money (initial_amount : Nat) (remaining_amount : Nat) :
  initial_amount = cost_of_game + remaining_amount →
  remaining_amount = number_of_toys * cost_of_toy →
  initial_amount = 68 := by
    sorry

end initial_money_l24_24408


namespace shopkeeper_gain_l24_24315

noncomputable def overall_percentage_gain (P : ℝ) (increase_percentage : ℝ) (discount1_percentage : ℝ) (discount2_percentage : ℝ) : ℝ :=
  let increased_price := P * (1 + increase_percentage)
  let price_after_first_discount := increased_price * (1 - discount1_percentage)
  let final_price := price_after_first_discount * (1 - discount2_percentage)
  ((final_price - P) / P) * 100

theorem shopkeeper_gain : 
  overall_percentage_gain 100 0.32 0.10 0.15 = 0.98 :=
by
  sorry

end shopkeeper_gain_l24_24315


namespace minute_hand_angle_backward_l24_24471

theorem minute_hand_angle_backward (backward_minutes : ℝ) (h : backward_minutes = 10) :
  (backward_minutes / 60) * (2 * Real.pi) = Real.pi / 3 := by
  sorry

end minute_hand_angle_backward_l24_24471


namespace prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l24_24707

variables (n k : ℕ) (p : ℝ)
variables (h_pos : 0 < p) (h_lt_one : p < 1)

-- Probability that exactly k gnomes fall
theorem prob_exactly_k_gnomes_fall (h_k_le_n : k ≤ n) :
  prob_speed (exactly_k_gnomes_fall n k p) = p * (1 - p)^(n - k) := sorry

-- Expected number of fallen gnomes
theorem expected_fallen_gnomes : 
  expected_falls n p = n + 1 - 1/p + (1 - p)^(n + 1)/p := sorry

end prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l24_24707


namespace elevator_stop_time_l24_24573

def time_to_reach_top (stories time_per_story : Nat) : Nat := stories * time_per_story

def total_time_with_stops (stories time_per_story stop_time : Nat) : Nat :=
  stories * time_per_story + (stories - 1) * stop_time

theorem elevator_stop_time (stories : Nat) (lola_time_per_story elevator_time_per_story total_elevator_time_to_top stop_time_per_floor : Nat)
  (lola_total_time : Nat) (is_slower : Bool)
  (h_lola: lola_total_time = time_to_reach_top stories lola_time_per_story)
  (h_slower: total_elevator_time_to_top = if is_slower then lola_total_time else 220)
  (h_no_stops: time_to_reach_top stories elevator_time_per_story + (stories - 1) * stop_time_per_floor = total_elevator_time_to_top) :
  stop_time_per_floor = 3 := 
  sorry

end elevator_stop_time_l24_24573


namespace amount_paid_is_200_l24_24576

-- Definitions of the costs and change received
def cost_of_pants := 140
def cost_of_shirt := 43
def cost_of_tie := 15
def change_received := 2

-- Total cost calculation
def total_cost := cost_of_pants + cost_of_shirt + cost_of_tie

-- Lean proof statement
theorem amount_paid_is_200 : total_cost + change_received = 200 := by
  -- Definitions ensure the total cost and change received are used directly from conditions
  sorry

end amount_paid_is_200_l24_24576


namespace part1_part2_l24_24231

-- Definitions based on the conditions in the problem.

-- Probabilities
def P_A_scores : ℚ := 2/5
def P_B_scores : ℚ := 1/3
def P_A_not_scores : ℚ := 1 - P_A_scores
def P_B_not_scores : ℚ := 1 - P_B_scores

-- Probability distribution of X
def P_X_neg1 : ℚ := (1 - P_A_scores) * P_B_scores
def P_X_0 : ℚ := P_A_scores * P_B_scores + (1 - P_A_scores) * (1 - P_B_scores)
def P_X_1 : ℚ := P_A_scores * (1 - P_B_scores)

-- The probability that A's cumulative score is higher than B's after two rounds
def P_2 : ℚ :=
  P_X_0 * P_X_1 + P_X_1 * (P_X_0 + P_X_1)

-- The theorem for Part 1
theorem part1 : 
  (P_X_neg1 = 1/5) ∧
  (P_X_0 = 8/15) ∧
  (P_X_1 = 4/15) := 
by sorry

-- The theorem for Part 2
theorem part2 :
  P_2 = 16 / 45 :=
by sorry

end part1_part2_l24_24231


namespace problem1_problem2_l24_24078

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

-- 1st problem: Prove the solution set for f(x) ≤ 2 when a = -1 is { x | x = ± 1/2 }
theorem problem1 : (∀ x : ℝ, f x (-1) ≤ 2 ↔ x = 1/2 ∨ x = -1/2) :=
by sorry

-- 2nd problem: Prove the range of real number a is [0, 3]
theorem problem2 : (∃ a : ℝ, (∀ x ∈ Set.Icc (1/2:ℝ) 1, f x a ≤ |2 * x + 1| ) ↔ 0 ≤ a ∧ a ≤ 3) :=
by sorry

end problem1_problem2_l24_24078


namespace no_combination_of_five_coins_is_75_l24_24198

theorem no_combination_of_five_coins_is_75 :
  ∀ (a b c d e : ℕ), 
    (a + b + c + d + e = 5) →
    ∀ (v : ℤ), 
      v = a * 1 + b * 5 + c * 10 + d * 25 + e * 50 → 
      v ≠ 75 :=
by
  intro a b c d e h1 v h2
  sorry

end no_combination_of_five_coins_is_75_l24_24198


namespace min_f_case1_min_f_case2_min_f_l24_24572

open Real

-- Definitions
def f (a b m n : ℕ+) (x : ℝ) : ℝ :=
  (sin x) ^ (m : ℕ) / (a : ℝ) + (b : ℝ) / (sin x) ^ (n : ℕ)

-- Proving the minimum value of f(x)
theorem min_f_case1 (a b m n : ℕ+) (h : (↑a * ↑b * ↑n : ℝ) ≥ ↑m) :
  (∃ x ∈ Ioo 0 π, f a b m n x = 1 / a + b) :=
sorry

theorem min_f_case2 (a b m n : ℕ+) (h : (↑a * ↑b * ↑n : ℝ) < ↑m) :
  (∃ x ∈ Ioo 0 π, f a b m n x = (m + n) * (1 / (n * a : ℝ)) ^ n * (b / m : ℝ) ^ m) :=
sorry

-- The general minima theorem
theorem min_f (a b m n : ℕ+) :
  ( if (↑a * ↑b * ↑n : ℝ) ≥ ↑m then 
      ∃ x ∈ Ioo 0 π, f a b m n x = (1 / a : ℝ) + b 
    else 
      ∃ x ∈ Ioo 0 π, f a b m n x = (m + n) * (1 / (n * a : ℝ)) ^ n * (b / m : ℝ) ^ m ) :=
by
  by_cases (h : (↑a * ↑b * ↑n : ℝ) ≥ ↑m)
  · exact min_f_case1 a b m n h
  · exact min_f_case2 a b m n (not_le.mp h)

end min_f_case1_min_f_case2_min_f_l24_24572


namespace simplify_and_evaluate_expr_find_ab_l24_24827

theorem simplify_and_evaluate_expr (x y : ℝ) (hx : x = 0.5) (hy : y = -1) :
  (x - 5 * y) * (-x - 5 * y) - (-x + 5 * y)^2 = -5.5 :=
by
  rw [hx, hy]
  sorry

theorem find_ab (a b : ℝ) (h : a^2 - 2 * a + b^2 + 4 * b + 5 = 0) :
  (a + b) ^ 2013 = -1 :=
by
  sorry

end simplify_and_evaluate_expr_find_ab_l24_24827


namespace train_speed_l24_24468

theorem train_speed (time_seconds : ℕ) (length_meters : ℕ) (speed_kmph : ℕ)
  (h1 : time_seconds = 9) (h2 : length_meters = 135) : speed_kmph = 54 :=
sorry

end train_speed_l24_24468


namespace fractional_eq_k_l24_24070

open Real

theorem fractional_eq_k (x k : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) :
  (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0) ↔ k ≠ -3 ∧ k ≠ 5 := 
sorry

end fractional_eq_k_l24_24070


namespace logic_problem_l24_24366

theorem logic_problem (p q : Prop) (h1 : ¬p) (h2 : ¬(p ∧ q)) : ¬ (p ∨ q) :=
sorry

end logic_problem_l24_24366


namespace chromium_percentage_l24_24777

noncomputable def chromium_percentage_in_new_alloy 
    (chromium_percentage_first: ℝ) 
    (weight_first: ℝ) 
    (chromium_percentage_second: ℝ) 
    (weight_second: ℝ) : ℝ :=
    (((chromium_percentage_first * weight_first / 100) + (chromium_percentage_second * weight_second / 100)) 
    / (weight_first + weight_second)) * 100

theorem chromium_percentage 
    (chromium_percentage_first: ℝ) 
    (weight_first: ℝ) 
    (chromium_percentage_second: ℝ) 
    (weight_second: ℝ) 
    (h1 : chromium_percentage_first = 10) 
    (h2 : weight_first = 15) 
    (h3 : chromium_percentage_second = 8) 
    (h4 : weight_second = 35) :
    chromium_percentage_in_new_alloy chromium_percentage_first weight_first chromium_percentage_second weight_second = 8.6 :=
by 
  rw [h1, h2, h3, h4]
  simp [chromium_percentage_in_new_alloy]
  norm_num


end chromium_percentage_l24_24777


namespace proof_statements_l24_24291

theorem proof_statements :
  (∃ n : ℕ, 24 = 4 * n) ∧       -- corresponding to A
  ¬((∃ m : ℕ, 190 = 19 * m) ∧  ¬(∃ k : ℕ, 57 = 19 * k)) ∧  -- corresponding to B
  ¬((∃ p : ℕ, 90 = 30 * p) ∨ (∃ q : ℕ, 65 = 30 * q)) ∧     -- corresponding to C
  ¬((∃ r : ℕ, 33 = 11 * r) ∧ ¬(∃ s : ℕ, 55 = 11 * s)) ∧    -- corresponding to D
  (∃ t : ℕ, 162 = 9 * t) :=                                 -- corresponding to E
by {
  -- Proof steps would go here
  sorry
}

end proof_statements_l24_24291


namespace range_of_b_distance_when_b_eq_one_l24_24245

-- Definitions for conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def line (x y b : ℝ) : Prop := y = x + b
def intersect (x y b : ℝ) : Prop := ellipse x y ∧ line x y b

-- Prove the range of b for which there are two distinct intersection points
theorem range_of_b (b : ℝ) : (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ intersect x1 y1 b ∧ intersect x2 y2 b) ↔ (-Real.sqrt 3 < b ∧ b < Real.sqrt 3) :=
by sorry

-- Prove the distance between points A and B when b = 1
theorem distance_when_b_eq_one : 
  ∃ x1 y1 x2 y2, intersect x1 y1 1 ∧ intersect x2 y2 1 ∧ Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 * Real.sqrt 2 / 3 :=
by sorry

end range_of_b_distance_when_b_eq_one_l24_24245


namespace children_getting_on_bus_l24_24156

theorem children_getting_on_bus (a b c: ℕ) (ha : a = 64) (hb : b = 78) (hc : c = b - a) : c = 14 :=
by
  sorry

end children_getting_on_bus_l24_24156


namespace derek_lowest_score_l24_24666

theorem derek_lowest_score:
  ∀ (score1 score2 max_points target_avg min_score tests_needed last_test1 last_test2 : ℕ),
  score1 = 85 →
  score2 = 78 →
  max_points = 100 →
  target_avg = 84 →
  min_score = 60 →
  tests_needed = 4 →
  last_test1 >= min_score →
  last_test2 >= min_score →
  last_test1 <= max_points →
  last_test2 <= max_points →
  (score1 + score2 + last_test1 + last_test2) = target_avg * tests_needed →
  min last_test1 last_test2 = 73 :=
by
  sorry

end derek_lowest_score_l24_24666


namespace product_modulo_seven_l24_24014

theorem product_modulo_seven (a b c d : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3)
(h3 : c % 7 = 4) (h4 : d % 7 = 5) : (a * b * c * d) % 7 = 1 := 
sorry

end product_modulo_seven_l24_24014


namespace maximal_area_of_AMNQ_l24_24792

theorem maximal_area_of_AMNQ (s q : ℝ) (Hq1 : 0 ≤ q) (Hq2 : q ≤ s) :
  let Q := (s, q)
  ∃ M N : ℝ × ℝ, 
    (M.1 ∈ [0,s] ∧ M.2 = 0) ∧ 
    (N.1 = s ∧ N.2 ∈ [0,s]) ∧ 
    if q ≤ (2/3) * s 
    then 
      (M.1 * M.2 / 2 = (CQ/2)) 
    else 
      (N = (s, s)) :=
by sorry

end maximal_area_of_AMNQ_l24_24792


namespace nonnegative_fraction_iff_interval_l24_24674

theorem nonnegative_fraction_iff_interval (x : ℝ) : 
  0 ≤ x ∧ x < 3 ↔ 0 ≤ (x^2 - 12 * x^3 + 36 * x^4) / (9 - x^3) := by
  sorry

end nonnegative_fraction_iff_interval_l24_24674


namespace bankers_discount_is_correct_l24_24295

-- Define the given conditions
def TD := 45   -- True discount in Rs.
def FV := 270  -- Face value in Rs.

-- Calculate Present Value based on the given conditions
def PV := FV - TD

-- Define the formula for Banker's Discount
def BD := TD + (TD ^ 2 / PV)

-- Prove that the Banker's Discount is Rs. 54 given the conditions
theorem bankers_discount_is_correct : BD = 54 :=
by
  -- Steps to prove the theorem can be filled here
  -- Add "sorry" to skip the actual proof
  sorry

end bankers_discount_is_correct_l24_24295


namespace find_natural_numbers_l24_24194

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem find_natural_numbers (x : ℕ) :
  (x = 36 * sum_of_digits x) ↔ (x = 324 ∨ x = 648) :=
by
  sorry

end find_natural_numbers_l24_24194


namespace slant_asymptote_sum_l24_24330

theorem slant_asymptote_sum (m b : ℝ) 
  (h : ∀ x : ℝ, y = 3*x^2 + 4*x - 8 / (x - 4) → y = m*x + b) :
  m + b = 19 :=
sorry

end slant_asymptote_sum_l24_24330


namespace solve_inequality_l24_24909

theorem solve_inequality :
  ∀ x : ℝ, (3 * x^2 - 4 * x - 7 < 0) ↔ (-1 < x ∧ x < 7 / 3) :=
by
  sorry

end solve_inequality_l24_24909


namespace systematic_sampling_40th_number_l24_24372

theorem systematic_sampling_40th_number
  (total_students sample_size : ℕ)
  (first_group_start first_group_end selected_first_group_number steps : ℕ)
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_group_start = 1)
  (h4 : first_group_end = 20)
  (h5 : selected_first_group_number = 15)
  (h6 : steps = total_students / sample_size)
  (h7 : first_group_end - first_group_start + 1 = steps)
  : (selected_first_group_number + steps * (40 - 1)) = 795 :=
sorry

end systematic_sampling_40th_number_l24_24372


namespace find_value_at_l24_24241

-- Defining the function f
variable (f : ℝ → ℝ)

-- Conditions
-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Condition 2: f has a period of 4
def periodic_function (f : ℝ → ℝ) := ∀ x, f (x + 4) = f x

-- Condition 3: In the interval [0,1], f(x) = 3x
def definition_on_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 3 * x

-- Statement to prove
theorem find_value_at (f : ℝ → ℝ) 
  (odd_f : odd_function f) 
  (periodic_f : periodic_function f) 
  (def_on_interval : definition_on_interval f) :
  f 11.5 = -1.5 := by 
  sorry

end find_value_at_l24_24241


namespace initial_person_count_l24_24901

theorem initial_person_count
  (avg_weight_increase : ℝ)
  (weight_old_person : ℝ)
  (weight_new_person : ℝ)
  (h1 : avg_weight_increase = 4.2)
  (h2 : weight_old_person = 65)
  (h3 : weight_new_person = 98.6) :
  ∃ n : ℕ, weight_new_person - weight_old_person = avg_weight_increase * n ∧ n = 8 := 
by
  sorry

end initial_person_count_l24_24901


namespace batsman_average_l24_24109

theorem batsman_average
  (avg_20_matches : ℕ → ℕ → ℕ)
  (avg_10_matches : ℕ → ℕ → ℕ)
  (total_1st_20 : ℕ := avg_20_matches 20 30)
  (total_next_10 : ℕ := avg_10_matches 10 15) :
  (total_1st_20 + total_next_10) / 30 = 25 :=
by
  sorry

end batsman_average_l24_24109


namespace min_packages_l24_24055

theorem min_packages (p : ℕ) (N : ℕ) :
  (N = 19 * p) →
  (N % 7 = 4) →
  (N % 11 = 1) →
  p = 40 :=
by
  sorry

end min_packages_l24_24055


namespace smallest_palindromic_odd_integer_in_base2_and_4_l24_24325

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := n.digits base
  digits = digits.reverse

theorem smallest_palindromic_odd_integer_in_base2_and_4 :
  ∃ n : ℕ, n > 10 ∧ is_palindrome n 2 ∧ is_palindrome n 4 ∧ Odd n ∧ ∀ m : ℕ, (m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 4 ∧ Odd m) → n <= m :=
  sorry

end smallest_palindromic_odd_integer_in_base2_and_4_l24_24325


namespace contractor_daily_wage_l24_24458

theorem contractor_daily_wage (total_days : ℕ) (absent_days : ℕ) (fine_per_absent_day total_amount : ℝ) (daily_wage : ℝ)
  (h_total_days : total_days = 30)
  (h_absent_days : absent_days = 8)
  (h_fine : fine_per_absent_day = 7.50)
  (h_total_amount : total_amount = 490) 
  (h_work_days : total_days - absent_days = 22)
  (h_total_fined : fine_per_absent_day * absent_days = 60)
  (h_total_earned : 22 * daily_wage - 60 = 490) :
  daily_wage = 25 := 
by 
  sorry

end contractor_daily_wage_l24_24458


namespace sand_needed_l24_24236

def area_rectangular_patch : ℕ := 6 * 7
def area_square_patch : ℕ := 5 * 5
def sand_per_square_inch : ℕ := 3

theorem sand_needed : area_rectangular_patch + area_square_patch * sand_per_square_inch = 201 := sorry

end sand_needed_l24_24236


namespace adam_simon_distance_100_l24_24316

noncomputable def time_to_be_100_apart (x : ℝ) : Prop :=
  let distance_adam := 10 * x
  let distance_simon_east := 10 * x * (Real.sqrt 2 / 2)
  let distance_simon_south := 10 * x * (Real.sqrt 2 / 2)
  let total_eastward_separation := abs (distance_adam - distance_simon_east)
  let resultant_distance := Real.sqrt (total_eastward_separation^2 + distance_simon_south^2)
  resultant_distance = 100

theorem adam_simon_distance_100 : ∃ (x : ℝ), time_to_be_100_apart x ∧ x = 2 * Real.sqrt 2 := 
by
  sorry

end adam_simon_distance_100_l24_24316


namespace repeating_decimal_sum_l24_24187

noncomputable def a : ℚ := 0.66666667 -- Repeating decimal 0.666... corresponds to 2/3
noncomputable def b : ℚ := 0.22222223 -- Repeating decimal 0.222... corresponds to 2/9
noncomputable def c : ℚ := 0.44444445 -- Repeating decimal 0.444... corresponds to 4/9
noncomputable def d : ℚ := 0.99999999 -- Repeating decimal 0.999... corresponds to 1

theorem repeating_decimal_sum : a + b - c + d = 13 / 9 := by
  sorry

end repeating_decimal_sum_l24_24187


namespace distinct_real_x_l24_24523

theorem distinct_real_x (x : ℝ) :
  (∃! x, ∃ (s : ℝ), s ∈ set.Icc 0 11 ∧ s = real.sqrt (123 - real.sqrt x)) =
  12 := sorry

end distinct_real_x_l24_24523


namespace non_empty_solution_set_inequality_l24_24270

theorem non_empty_solution_set_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 3| < a) ↔ a > -4 := 
sorry

end non_empty_solution_set_inequality_l24_24270


namespace alyssa_spent_on_grapes_l24_24319

theorem alyssa_spent_on_grapes (t c g : ℝ) (h1 : t = 21.93) (h2 : c = 9.85) (h3 : t = g + c) : g = 12.08 :=
by
  sorry

end alyssa_spent_on_grapes_l24_24319


namespace necessary_but_not_sufficient_condition_l24_24527

noncomputable def necessary_but_not_sufficient (x : ℝ) : Prop :=
  (3 - x >= 0 → |x - 1| ≤ 2) ∧ ¬(3 - x >= 0 ↔ |x - 1| ≤ 2)

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  necessary_but_not_sufficient x :=
sorry

end necessary_but_not_sufficient_condition_l24_24527


namespace number_is_a_l24_24265

theorem number_is_a (x y z a : ℝ) (h1 : x + y + z = a) (h2 : (1 / x) + (1 / y) + (1 / z) = 1 / a) : 
  x = a ∨ y = a ∨ z = a :=
sorry

end number_is_a_l24_24265


namespace incorrect_proposition_l24_24667

theorem incorrect_proposition (p q : Prop) :
  ¬(¬(p ∧ q) → ¬p ∧ ¬q) := sorry

end incorrect_proposition_l24_24667


namespace exists_circular_chain_of_four_l24_24616

-- Let A and B be the two teams, each with a set of players.
variable {A B : Type}
-- Assume there exists a relation "beats" that determines match outcomes.
variable (beats : A → B → Prop)

-- Each player in both teams has at least one win and one loss against the opposite team.
axiom each_has_win_and_loss (a : A) : ∃ b1 b2 : B, beats a b1 ∧ ¬beats a b2 ∧ b1 ≠ b2
axiom each_has_win_and_loss' (b : B) : ∃ a1 a2 : A, beats a1 b ∧ ¬beats a2 b ∧ a1 ≠ a2

-- Main theorem: Exist four players forming a circular chain of victories.
theorem exists_circular_chain_of_four :
  ∃ (a1 a2 : A) (b1 b2 : B), beats a1 b1 ∧ ¬beats a1 b2 ∧ beats a2 b2 ∧ ¬beats a2 b1 ∧ b1 ≠ b2 ∧ a1 ≠ a2 :=
sorry

end exists_circular_chain_of_four_l24_24616


namespace train_speed_l24_24469

theorem train_speed (time_seconds : ℕ) (length_meters : ℕ) (speed_kmph : ℕ)
  (h1 : time_seconds = 9) (h2 : length_meters = 135) : speed_kmph = 54 :=
sorry

end train_speed_l24_24469


namespace range_of_a_l24_24355

variable (a : ℝ)

def proposition_p : Prop :=
  ∃ x₀ : ℝ, x₀^2 - a * x₀ + a = 0

def proposition_q : Prop :=
  ∀ x : ℝ, 1 < x → x + 1 / (x - 1) ≥ a

theorem range_of_a (h : ¬proposition_p a ∧ proposition_q a) : 0 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l24_24355


namespace max_value_of_x2_plus_y2_l24_24389

open Real

theorem max_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 = 2 * x - 2 * y + 2) : 
  x^2 + y^2 ≤ 6 + 4 * sqrt 2 :=
sorry

end max_value_of_x2_plus_y2_l24_24389


namespace union_sets_l24_24238

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5, 6}

theorem union_sets : (A ∪ B) = {1, 2, 3, 4, 5, 6} :=
by
  sorry

end union_sets_l24_24238


namespace rectangle_area_l24_24752

theorem rectangle_area (L W : ℕ) (h1 : 2 * L + 2 * W = 280) (h2 : L = 5 * (W / 2)) : L * W = 4000 :=
sorry

end rectangle_area_l24_24752


namespace interest_rate_compound_interest_l24_24121

theorem interest_rate_compound_interest :
  ∀ (P A : ℝ) (t n : ℕ), 
  P = 156.25 → A = 169 → t = 2 → n = 1 → 
  (∃ r : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ r * 100 = 4) :=
by
  intros P A t n hP hA ht hn
  use 0.04
  rw [hP, hA, ht, hn]
  sorry

end interest_rate_compound_interest_l24_24121


namespace shorter_steiner_network_l24_24276

-- Define the variables and inequality
noncomputable def side_length (a : ℝ) : ℝ := a
noncomputable def diagonal_network_length (a : ℝ) : ℝ := 2 * a * Real.sqrt 2
noncomputable def steiner_network_length (a : ℝ) : ℝ := a * (1 + Real.sqrt 3)

theorem shorter_steiner_network {a : ℝ} (h₀ : 0 < a) :
  diagonal_network_length a > steiner_network_length a :=
by
  -- Proof to be provided (skipping it with sorry)
  sorry

end shorter_steiner_network_l24_24276


namespace Vanya_original_number_l24_24283

theorem Vanya_original_number (m n : ℕ) (hm : m ≤ 9) (hn : n ≤ 9) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 10 * m + n = 81 := by
  sorry

end Vanya_original_number_l24_24283


namespace trigonometric_inequality_l24_24097

open Real

theorem trigonometric_inequality
  (x y z : ℝ)
  (h1 : 0 < x)
  (h2 : x < y)
  (h3 : y < z)
  (h4 : z < π / 2) :
  π / 2 + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) :=
by
  sorry

end trigonometric_inequality_l24_24097


namespace find_number_l24_24220

theorem find_number (x n : ℤ) (h1 : 5 * x + n = 10 * x - 17) (h2 : x = 4) : n = 3 := by
  sorry

end find_number_l24_24220


namespace permutations_with_k_in_first_position_l24_24013

noncomputable def numberOfPermutationsWithKInFirstPosition (N k : ℕ) (h : k < N) : ℕ :=
  (2 : ℕ)^(N-1)

theorem permutations_with_k_in_first_position (N k : ℕ) (h : k < N) :
  numberOfPermutationsWithKInFirstPosition N k h = (2 : ℕ)^(N-1) :=
sorry

end permutations_with_k_in_first_position_l24_24013


namespace problem_1_l24_24053

theorem problem_1 :
  (5 / ((1 / (1 * 2)) + (1 / (2 * 3)) + (1 / (3 * 4)) + (1 / (4 * 5)) + (1 / (5 * 6)))) = 6 := by
  sorry

end problem_1_l24_24053


namespace exists_twelve_distinct_x_l24_24520

theorem exists_twelve_distinct_x :
  ∃ S : Set ℝ, (S.card = 12) ∧ (∀ x ∈ S, ∃ k : ℤ, 0 ≤ k ∧ k ≤ 11 ∧ (sqrt (123 - sqrt x) = k)) :=
by
  sorry

end exists_twelve_distinct_x_l24_24520


namespace print_colored_pages_l24_24419

theorem print_colored_pages (cost_per_page : ℕ) (dollars : ℕ) (conversion_rate : ℕ) 
    (h_cost : cost_per_page = 4) (h_dollars : dollars = 30) (h_conversion : conversion_rate = 100) :
    (dollars * conversion_rate) / cost_per_page = 750 := 
by
  sorry

end print_colored_pages_l24_24419


namespace votes_lost_by_l24_24650

theorem votes_lost_by (total_votes : ℕ) (candidate_percentage : ℕ) : total_votes = 20000 → candidate_percentage = 10 → 
  (total_votes * candidate_percentage / 100 - total_votes * (100 - candidate_percentage) / 100 = 16000) :=
by
  intros h_total_votes h_candidate_percentage
  have vote_candidate := total_votes * candidate_percentage / 100
  have vote_rival := total_votes * (100 - candidate_percentage) / 100
  have votes_diff := vote_rival - vote_candidate
  rw [h_total_votes, h_candidate_percentage] at *
  sorry

end votes_lost_by_l24_24650


namespace correct_option_l24_24290

-- Definitions based on the conditions in step a
def option_a : Prop := (-3 - 1 = -2)
def option_b : Prop := (-2 * (-1 / 2) = 1)
def option_c : Prop := (16 / (-4 / 3) = 12)
def option_d : Prop := (- (3^2) / 4 = (9 / 4))

-- The proof problem statement asserting that only option B is correct.
theorem correct_option : option_b ∧ ¬ option_a ∧ ¬ option_c ∧ ¬ option_d :=
by sorry

end correct_option_l24_24290


namespace area_of_figure_eq_two_l24_24585

theorem area_of_figure_eq_two :
  ∫ x in (1 / Real.exp 1)..(Real.exp 1), 1 / x = 2 :=
by sorry

end area_of_figure_eq_two_l24_24585


namespace statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_incorrect_l24_24638

-- Statement A
theorem statement_A_incorrect (a b c d : ℝ) (ha : a < b) (hc : c < d) : ¬ (a * c < b * d) := by
  sorry

-- Statement B
theorem statement_B_correct (a b : ℝ) (ha : -2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) : -1 < a / b ∧ a / b < 3 := by
  sorry

-- Statement C
theorem statement_C_incorrect (m : ℝ) : ¬ (∀ x > 0, x / 2 + 2 / x ≥ m) ∧ (m ≤ 1) := by
  sorry

-- Statement D
theorem statement_D_incorrect : ∃ x : ℝ, (x^2 + 2) + 1 / (x^2 + 2) ≠ 2 := by
  sorry

end statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_incorrect_l24_24638


namespace common_tangent_and_inequality_l24_24844

noncomputable def f (x : ℝ) := Real.log (1 + x)
noncomputable def g (x : ℝ) := x - (1 / 2) * x^2 + (1 / 3) * x^3

theorem common_tangent_and_inequality :
  -- Condition: common tangent at (0, 0)
  (∀ x, deriv f x = deriv g x) →
  -- Condition: values of a and b found to be 0 and 1 respectively
  (∀ x, f x ≤ g x) :=
by
  intro h
  sorry

end common_tangent_and_inequality_l24_24844


namespace solve_for_q_l24_24483

theorem solve_for_q (q : ℝ) (p : ℝ) (h : p = 15 * q^2 - 5) : p = 40 → q = Real.sqrt 3 :=
by
  sorry

end solve_for_q_l24_24483


namespace diagonals_in_nine_sided_polygon_l24_24994

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l24_24994


namespace repeating_decimal_to_fraction_l24_24332

theorem repeating_decimal_to_fraction 
  (h : ∀ {x : ℝ}, (0.01 : ℝ) = 1 / 99 → x = 1.06 → (0.06 : ℝ) = 6 * 1 / 99): 
  1.06 = 35 / 33 :=
by sorry

end repeating_decimal_to_fraction_l24_24332


namespace Christine_savings_l24_24477

theorem Christine_savings 
  (commission_rate: ℝ) 
  (total_sales: ℝ) 
  (personal_needs_percentage: ℝ) 
  (savings: ℝ) 
  (h1: commission_rate = 0.12) 
  (h2: total_sales = 24000) 
  (h3: personal_needs_percentage = 0.60) 
  (h4: savings = total_sales * commission_rate * (1 - personal_needs_percentage)) : 
  savings = 1152 := by 
  sorry

end Christine_savings_l24_24477


namespace cats_on_ship_l24_24933

theorem cats_on_ship :
  ∃ (C S : ℕ), 
  (C + S + 1 + 1 = 16) ∧
  (4 * C + 2 * S + 2 * 1 + 1 * 1 = 41) ∧ 
  C = 5 :=
by
  sorry

end cats_on_ship_l24_24933


namespace find_speed_of_faster_train_l24_24017

noncomputable def speed_of_faster_train
  (length_each_train_m : ℝ)
  (speed_slower_kmph : ℝ)
  (time_pass_s : ℝ) : ℝ :=
  let distance_km := (2 * length_each_train_m / 1000)
  let time_pass_hr := (time_pass_s / 3600)
  let relative_speed_kmph := (distance_km / time_pass_hr)
  let speed_faster_kmph := (relative_speed_kmph - speed_slower_kmph)
  speed_faster_kmph

theorem find_speed_of_faster_train :
  speed_of_faster_train
    250   -- length_each_train_m
    30    -- speed_slower_kmph
    23.998080153587715 -- time_pass_s
  = 45 := sorry

end find_speed_of_faster_train_l24_24017


namespace eccentricity_equals_2_l24_24057

variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (A : ℝ × ℝ) (F : ℝ × ℝ) (B : ℝ × ℝ)
variables (eqn_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
variables (focus_F : F = (c, 0)) (imaginary_axis_B : B = (0, b))
variables (intersect_A : A = (c / 3, 2 * b / 3))
variables (vector_eqn : 3 * (A.1, A.2) = (F.1 + 2 * B.1, F.2 + 2 * B.2))
variables (asymptote_eqn : ∀ A1 A2 : ℝ, A2 = (b / a) * A1 → A = (A1, A2))

theorem eccentricity_equals_2 : (c / a = 2) :=
sorry

end eccentricity_equals_2_l24_24057


namespace elephant_weight_equivalence_l24_24561

variable (y : ℝ)
variable (porter_weight : ℝ := 120)
variable (blocks_1 : ℝ := 20)
variable (blocks_2 : ℝ := 21)
variable (porters_1 : ℝ := 3)
variable (porters_2 : ℝ := 1)

theorem elephant_weight_equivalence :
  (y - porters_1 * porter_weight) / blocks_1 = (y - porters_2 * porter_weight) / blocks_2 := 
sorry

end elephant_weight_equivalence_l24_24561


namespace petya_cannot_form_figure_c_l24_24102

-- Define the rhombus and its properties, including rotation
noncomputable def is_rotatable_rhombus (r : ℕ) : Prop := sorry

-- Define the larger shapes and their properties in terms of whether they can be formed using rotations of the rhombus.
noncomputable def can_form_figure_a (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_b (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_c (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_d (rhombus : ℕ) : Prop := sorry

-- Statement: Petya cannot form the figure (c) using the rhombus and allowed transformations.
theorem petya_cannot_form_figure_c (rhombus : ℕ) (h : is_rotatable_rhombus rhombus) :
  ¬ can_form_figure_c rhombus := sorry

end petya_cannot_form_figure_c_l24_24102


namespace fifteenth_entry_is_21_l24_24829

def r_9 (n : ℕ) : ℕ := n % 9

def condition (n : ℕ) : Prop := (7 * n) % 9 ≤ 5

def sequence_elements (k : ℕ) : ℕ := 
  if k = 0 then 0
  else if k = 1 then 2
  else if k = 2 then 3
  else if k = 3 then 4
  else if k = 4 then 7
  else if k = 5 then 8
  else if k = 6 then 9
  else if k = 7 then 11
  else if k = 8 then 12
  else if k = 9 then 13
  else if k = 10 then 16
  else if k = 11 then 17
  else if k = 12 then 18
  else if k = 13 then 20
  else if k = 14 then 21
  else 0 -- for the sake of ensuring completeness

theorem fifteenth_entry_is_21 : sequence_elements 14 = 21 :=
by
  -- Mathematical proof omitted.
  sorry

end fifteenth_entry_is_21_l24_24829


namespace sand_art_calculation_l24_24237

theorem sand_art_calculation :
  let rect_length := 6 in
  let rect_width := 7 in
  let square_side := 5 in
  let gram_per_sq_inch := 3 in
  let rect_area := rect_length * rect_width in
  let square_area := square_side * square_side in
  let rect_sand := rect_area * gram_per_sq_inch in
  let square_sand := square_area * gram_per_sq_inch in
  let total_sand_needed := rect_sand + square_sand in
  total_sand_needed = 201 :=
by
  -- sorry is used to skip the proof as instructed
  sorry

end sand_art_calculation_l24_24237


namespace days_c_worked_l24_24441

noncomputable def work_done_by_a_b := 1 / 10
noncomputable def work_done_by_b_c := 1 / 18
noncomputable def work_done_by_c_alone := 1 / 45

theorem days_c_worked
  (A B C : ℚ)
  (h1 : A + B = work_done_by_a_b)
  (h2 : B + C = work_done_by_b_c)
  (h3 : C = work_done_by_c_alone) :
  15 = (1/3) / work_done_by_c_alone :=
sorry

end days_c_worked_l24_24441


namespace smallest_n_square_partition_l24_24026

theorem smallest_n_square_partition (n : ℕ) (h : ∃ a b : ℕ, a ≥ 1 ∧ b ≥ 1 ∧ n = 40 * a + 49 * b) : n ≥ 2000 :=
by sorry

end smallest_n_square_partition_l24_24026


namespace original_fraction_l24_24702

theorem original_fraction (x y : ℝ) (hxy : x / y = 5 / 7)
  (hx : 1.20 * x / (0.90 * y) = 20 / 21) : x / y = 5 / 7 :=
by {
  sorry
}

end original_fraction_l24_24702


namespace original_number_of_employees_l24_24306

theorem original_number_of_employees (x : ℕ) 
  (h1 : 0.77 * (x : ℝ) = 328) : x = 427 :=
sorry

end original_number_of_employees_l24_24306


namespace Christine_savings_l24_24478

theorem Christine_savings 
  (commission_rate: ℝ) 
  (total_sales: ℝ) 
  (personal_needs_percentage: ℝ) 
  (savings: ℝ) 
  (h1: commission_rate = 0.12) 
  (h2: total_sales = 24000) 
  (h3: personal_needs_percentage = 0.60) 
  (h4: savings = total_sales * commission_rate * (1 - personal_needs_percentage)) : 
  savings = 1152 := by 
  sorry

end Christine_savings_l24_24478


namespace total_watermelon_weight_l24_24382

theorem total_watermelon_weight :
  let w1 := 9.91
  let w2 := 4.112
  let w3 := 6.059
  w1 + w2 + w3 = 20.081 :=
by
  sorry

end total_watermelon_weight_l24_24382


namespace graphs_intersect_at_one_point_l24_24079

theorem graphs_intersect_at_one_point (m : ℝ) (e := Real.exp 1) :
  (∀ f g : ℝ → ℝ,
    (∀ x, f x = x + Real.log x - 2 / e) ∧ (∀ x, g x = m / x) →
    ∃! x, f x = g x) ↔ (m ≥ 0 ∨ m = - (e + 1) / (e ^ 2)) :=
by sorry

end graphs_intersect_at_one_point_l24_24079


namespace annual_income_A_l24_24604

variable (A B C : ℝ)
variable (monthly_income_C : C = 17000)
variable (monthly_income_B : B = C + 0.12 * C)
variable (ratio_A_to_B : A / B = 5 / 2)

theorem annual_income_A (A B C : ℝ) 
    (hC : C = 17000) 
    (hB : B = C + 0.12 * C) 
    (hR : A / B = 5 / 2) : 
    A * 12 = 571200 :=
by
  sorry

end annual_income_A_l24_24604


namespace problem_I_problem_II_problem_III_l24_24843

-- Problem (I)
noncomputable def f (x a : ℝ) := Real.log x - a * (x - 1)
noncomputable def tangent_line (x a : ℝ) := (1 - a) * (x - 1)

theorem problem_I (a : ℝ) :
  ∃ y, tangent_line y a = f 1 a / (1 : ℝ) :=
sorry

-- Problem (II)
theorem problem_II (a : ℝ) (h : a ≥ 1 / 2) :
  ∀ x ≥ 1, f x a ≤ Real.log x / (x + 1) :=
sorry

-- Problem (III)
theorem problem_III (a : ℝ) :
  ∀ x ≥ 1, Real.exp (x - 1) - a * (x ^ 2 - x) ≥ x * f x a + 1 :=
sorry

end problem_I_problem_II_problem_III_l24_24843


namespace note_relationship_l24_24937

theorem note_relationship
  (x y z : ℕ) 
  (h1 : x + 5 * y + 10 * z = 480)
  (h2 : x + y + z = 90)
  (h3 : y = 2 * x)
  (h4 : z = 3 * x) : 
  x = 15 ∧ y = 30 ∧ z = 45 :=
by 
  sorry

end note_relationship_l24_24937


namespace greatest_three_digit_multiple_23_l24_24621

theorem greatest_three_digit_multiple_23 : 
  ∃ n : ℕ, n < 1000 ∧ n % 23 = 0 ∧ (∀ m : ℕ, m < 1000 ∧ m % 23 = 0 → m ≤ n) ∧ n = 989 :=
sorry

end greatest_three_digit_multiple_23_l24_24621


namespace Johann_oranges_l24_24565

-- Define the given conditions
def initial_oranges := 60
def eaten_oranges := 10
def half_remaining_oranges := (initial_oranges - eaten_oranges) / 2
def returned_oranges := 5

-- Define the statement to prove
theorem Johann_oranges :
  initial_oranges - eaten_oranges - half_remaining_oranges + returned_oranges = 30 := by
  sorry

end Johann_oranges_l24_24565


namespace fraction_tabs_closed_l24_24868

theorem fraction_tabs_closed (x : ℝ) (h₁ : 400 * (1 - x) * (3/5) * (1/2) = 90) : 
  x = 1 / 4 :=
by
  have := h₁
  sorry

end fraction_tabs_closed_l24_24868


namespace percentage_of_women_in_study_group_l24_24784

theorem percentage_of_women_in_study_group
  (W : ℝ)
  (H1 : 0 ≤ W ∧ W ≤ 1)
  (H2 : 0.60 * W = 0.54) :
  W = 0.9 :=
sorry

end percentage_of_women_in_study_group_l24_24784


namespace chord_line_parabola_l24_24063

theorem chord_line_parabola (x1 x2 y1 y2 : ℝ) (hx1 : y1^2 = 8*x1) (hx2 : y2^2 = 8*x2)
  (hmid : (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1) : 4*(1/2*(x1 + x2)) + (1/2*(y1 + y2)) - 3 = 0 :=
by
  sorry

end chord_line_parabola_l24_24063


namespace find_initial_days_provisions_last_l24_24786

def initial_days_provisions_last (initial_men reinforcements days_after_reinforcement : ℕ) (x : ℕ) : Prop :=
  initial_men * (x - 15) = (initial_men + reinforcements) * days_after_reinforcement

theorem find_initial_days_provisions_last
  (initial_men reinforcements days_after_reinforcement x : ℕ)
  (h1 : initial_men = 2000)
  (h2 : reinforcements = 1900)
  (h3 : days_after_reinforcement = 20)
  (h4 : initial_days_provisions_last initial_men reinforcements days_after_reinforcement x) :
  x = 54 :=
by
  sorry


end find_initial_days_provisions_last_l24_24786


namespace tire_cost_l24_24898

theorem tire_cost (total_cost : ℕ) (number_of_tires : ℕ) (cost_per_tire : ℕ) 
    (h1 : total_cost = 240) 
    (h2 : number_of_tires = 4)
    (h3 : cost_per_tire = total_cost / number_of_tires) : 
    cost_per_tire = 60 :=
sorry

end tire_cost_l24_24898


namespace probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l24_24713

noncomputable def probability_k_gnomes_fall (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p * (1 - p) ^ (n - k)

noncomputable def expected_number_of_fallen_gnomes (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p)

theorem probability_k_gnomes_fall_correct (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  probability_k_gnomes_fall n k p h = p * (1 - p) ^ (n - k) :=
by sorry

theorem expected_number_of_fallen_gnomes_correct (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  expected_number_of_fallen_gnomes n p h = n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p) :=
by sorry

end probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l24_24713


namespace borrowed_dimes_calculation_l24_24409

-- Define Sam's initial dimes and remaining dimes after borrowing
def original_dimes : ℕ := 8
def remaining_dimes : ℕ := 4

-- Statement to prove that the borrowed dimes is 4
theorem borrowed_dimes_calculation : (original_dimes - remaining_dimes) = 4 :=
by
  -- This is the proof section which follows by simple arithmetic computation
  sorry

end borrowed_dimes_calculation_l24_24409


namespace find_abc_l24_24961

theorem find_abc (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≤ b ∧ b ≤ c) (h5 : a + b + c + a * b + b * c + c * a = a * b * c + 1) :
  (a = 2 ∧ b = 5 ∧ c = 8) ∨ (a = 3 ∧ b = 4 ∧ c = 13) :=
sorry

end find_abc_l24_24961


namespace biased_die_probability_l24_24782

theorem biased_die_probability :
  let p := 1 / 3
  let P_e := 2 * p
  let P_o := p
  P_e ^ 3 + P_o ^ 3 + 3 * P_e ^ 2 * P_o = 7 / 9 :=
by
  simp [P_e, P_o]
  sorry

end biased_die_probability_l24_24782


namespace eval_expression_l24_24444

theorem eval_expression : 
  ( ( (476 * 100 + 424 * 100) * 2^3 - 4 * (476 * 100 * 424 * 100) ) * (376 - 150) ) / 250 = -7297340160 :=
by
  sorry

end eval_expression_l24_24444


namespace conic_sections_with_foci_at_F2_zero_l24_24682

theorem conic_sections_with_foci_at_F2_zero (a b m n: ℝ) (h1 : a > b) (h2: b > 0) (h3: m > 0) (h4: n > 0) (h5: a^2 - b^2 = 4) (h6: m^2 + n^2 = 4):
  (∀ x y: ℝ, x^2 / (a^2) + y^2 / (b^2) = 1) ∧ (∀ x y: ℝ, x^2 / (11/60) + y^2 / (11/16) = 1) ∧ 
  ∀ x y: ℝ, x^2 / (m^2) - y^2 / (n^2) = 1 ∧ ∀ x y: ℝ, 5*x^2 / 4 - 5*y^2 / 16 = 1 := 
sorry

end conic_sections_with_foci_at_F2_zero_l24_24682


namespace hyperbola_condition_l24_24430

theorem hyperbola_condition (m : ℝ) : (m > 0) ↔ (2 + m > 0 ∧ 1 + m > 0) :=
by sorry

end hyperbola_condition_l24_24430


namespace two_digit_sabroso_numbers_l24_24162

theorem two_digit_sabroso_numbers :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ (n + (10 * b + a) = k^2)} =
  {29, 38, 47, 56, 65, 74, 83, 92} :=
sorry

end two_digit_sabroso_numbers_l24_24162


namespace greatest_three_digit_multiple_of_23_l24_24622

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of_23 (n : ℕ) : Prop :=
  n % 23 = 0

theorem greatest_three_digit_multiple_of_23 :
  ∀ n, is_three_digit n ∧ is_multiple_of_23 n → n ≤ 989 :=
by
  sorry

end greatest_three_digit_multiple_of_23_l24_24622


namespace solve_trig_eq_l24_24253

open Real

theorem solve_trig_eq (n : ℤ) (x : ℝ) : 
  (sin x) ^ 4 + (cos x) ^ 4 = (sin (2 * x)) ^ 4 + (cos (2 * x)) ^ 4 ↔ x = (n : ℝ) * π / 6 :=
by
  sorry

end solve_trig_eq_l24_24253


namespace circle_tangent_to_y_axis_l24_24268

/-- The relationship between the circle with the focal radius |PF| of the parabola y^2 = 2px (where p > 0)
as its diameter and the y-axis -/
theorem circle_tangent_to_y_axis
  (p : ℝ) (hp : p > 0)
  (x1 y1 : ℝ)
  (focus : ℝ × ℝ := (p / 2, 0))
  (P : ℝ × ℝ := (x1, y1))
  (center : ℝ × ℝ := ((2 * x1 + p) / 4, y1 / 2))
  (radius : ℝ := (2 * x1 + p) / 4) :
  -- proof that the circle with PF as its diameter is tangent to the y-axis
  ∃ k : ℝ, k = radius ∧ (center.1 = k) :=
sorry

end circle_tangent_to_y_axis_l24_24268


namespace problem_1_problem_2_l24_24357

noncomputable def a (k : ℝ) : ℝ × ℝ := (2, k)
noncomputable def b : ℝ × ℝ := (1, 1)
noncomputable def a_minus_3b (k : ℝ) : ℝ × ℝ := (2 - 3 * 1, k - 3 * 1)

-- First problem: Prove that k = 4 given vectors a and b, and the condition that b is perpendicular to (a - 3b)
theorem problem_1 (k : ℝ) (h : b.1 * (a_minus_3b k).1 + b.2 * (a_minus_3b k).2 = 0) : k = 4 :=
sorry

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def cosine (v w : ℝ × ℝ) : ℝ := dot_product v w / (magnitude v * magnitude w)

-- Second problem: Prove that the cosine value of the angle between a and b is 3√10/10 when k is 4
theorem problem_2 (k : ℝ) (hk : k = 4) : cosine (a k) b = 3 * Real.sqrt 10 / 10 :=
sorry

end problem_1_problem_2_l24_24357


namespace iron_heating_time_l24_24124

-- Define the conditions as constants
def ironHeatingRate : ℝ := 9 -- degrees Celsius per 20 seconds
def ironCoolingRate : ℝ := 15 -- degrees Celsius per 30 seconds
def coolingTime : ℝ := 180 -- seconds

-- Define the theorem to prove the heating back time
theorem iron_heating_time :
  (coolingTime / 30) * ironCoolingRate = 90 →
  (90 / ironHeatingRate) * 20 = 200 :=
by
  sorry

end iron_heating_time_l24_24124


namespace find_speed_from_p_to_q_l24_24790

noncomputable def speed_from_p_to_q (v : ℝ) (d : ℝ) : Prop :=
  let return_speed := 1.5 * v
  let avg_speed := 75
  let total_distance := 2 * d
  let total_time := d / v + d / return_speed
  avg_speed = total_distance / total_time

theorem find_speed_from_p_to_q (v : ℝ) (d : ℝ) : speed_from_p_to_q v d → v = 62.5 :=
by
  intro h
  sorry

end find_speed_from_p_to_q_l24_24790


namespace average_student_headcount_l24_24962

def student_headcount_fall_0203 : ℕ := 11700
def student_headcount_fall_0304 : ℕ := 11500
def student_headcount_fall_0405 : ℕ := 11600

theorem average_student_headcount : 
  (student_headcount_fall_0203 + student_headcount_fall_0304 + student_headcount_fall_0405) / 3 = 11600 := by
  sorry

end average_student_headcount_l24_24962


namespace rooster_stamps_eq_two_l24_24257

variable (r d : ℕ) -- r is the number of rooster stamps, d is the number of daffodil stamps

theorem rooster_stamps_eq_two (h1 : d = 2) (h2 : r - d = 0) : r = 2 := by
  sorry

end rooster_stamps_eq_two_l24_24257


namespace composite_p_squared_plus_36_l24_24636

theorem composite_p_squared_plus_36 (p : ℕ) (h_prime : Prime p) : 
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ (k * m = p^2 + 36) :=
by {
  sorry
}

end composite_p_squared_plus_36_l24_24636


namespace gilled_mushrooms_count_l24_24454

theorem gilled_mushrooms_count : 
  ∀ (total_mushrooms gilled_mushrooms_ratio spotted_mushrooms_ratio : ℕ),
  (total_mushrooms = 30) →
  (gilled_mushrooms_ratio = 1) →
  (spotted_mushrooms_ratio = 9) →
  total_mushrooms / (gilled_mushrooms_ratio + spotted_mushrooms_ratio) = 3 :=
by
  intros total_mushrooms gilled_mushrooms_ratio spotted_mushrooms_ratio
  assume h_total h_gilled h_spotted
  rw [h_total, h_gilled, h_spotted]
  norm_num
  sorry

end gilled_mushrooms_count_l24_24454


namespace greatest_three_digit_multiple_of_23_is_991_l24_24627

theorem greatest_three_digit_multiple_of_23_is_991 :
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 23 = 0) ∧ ∀ m : ℤ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 23 = 0) → m ≤ n :=
begin
  use 991,
  -- proof steps go here
  sorry
end

end greatest_three_digit_multiple_of_23_is_991_l24_24627


namespace convert_base_five_to_ten_l24_24768

theorem convert_base_five_to_ten : ∃ n : ℕ, n = 38 ∧ (1 * 5^2 + 2 * 5^1 + 3 * 5^0 = n) :=
by
  sorry

end convert_base_five_to_ten_l24_24768


namespace difference_in_amount_paid_l24_24143

variable (P Q : ℝ)

theorem difference_in_amount_paid (hP : P > 0) (hQ : Q > 0) :
  (1.10 * P * 0.80 * Q - P * Q) = -0.12 * (P * Q) := 
by 
  sorry

end difference_in_amount_paid_l24_24143


namespace lizzy_wealth_after_loan_l24_24394

theorem lizzy_wealth_after_loan 
  (initial_wealth : ℕ)
  (loan : ℕ)
  (interest_rate : ℕ)
  (h1 : initial_wealth = 30)
  (h2 : loan = 15)
  (h3 : interest_rate = 20)
  : (initial_wealth - loan) + (loan + loan * interest_rate / 100) = 33 :=
by
  sorry

end lizzy_wealth_after_loan_l24_24394


namespace find_number_l24_24153

theorem find_number (x : ℝ) (h : -200 * x = 1600) : x = -8 :=
by sorry

end find_number_l24_24153


namespace trey_used_50_nails_l24_24436

-- Definitions based on conditions
def decorations_with_sticky_strips := 15
def fraction_nails := 2/3
def fraction_thumbtacks := 2/5

-- Define D as total number of decorations and use the given conditions
noncomputable def total_decorations : ℕ :=
  let D := decorations_with_sticky_strips / ((1:ℚ) - fraction_nails - (fraction_thumbtacks * (1 - fraction_nails))) in
  if h : 0 < D ∧ D.denom = 1 then D.num else 0

-- Nails used by Trey
noncomputable def nails_used : ℕ := (fraction_nails * total_decorations).toNat

theorem trey_used_50_nails : nails_used = 50 := by
  sorry

end trey_used_50_nails_l24_24436


namespace omega_value_l24_24500

theorem omega_value (ω : ℕ) (h : ω > 0) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = Real.sin (ω * x + Real.pi / 4)) 
  (h2 : ∀ x y, (Real.pi / 6 < x ∧ x < Real.pi / 3) → (Real.pi / 6 < y ∧ y < Real.pi / 3) → x < y → f y < f x) :
    ω = 2 ∨ ω = 3 := 
sorry

end omega_value_l24_24500


namespace grape_rate_per_kg_l24_24051

theorem grape_rate_per_kg (G : ℝ) : 
    (8 * G) + (9 * 55) = 1055 → G = 70 := by
  sorry

end grape_rate_per_kg_l24_24051


namespace students_in_both_clubs_l24_24320

theorem students_in_both_clubs (total_students drama_club art_club drama_or_art in_both_clubs : ℕ)
  (H1 : total_students = 300)
  (H2 : drama_club = 120)
  (H3 : art_club = 150)
  (H4 : drama_or_art = 220) :
  in_both_clubs = drama_club + art_club - drama_or_art :=
by
  -- this is the proof space
  sorry

end students_in_both_clubs_l24_24320


namespace find_missing_number_l24_24965

theorem find_missing_number (x : ℕ) (h : 10111 - x * 2 * 5 = 10011) : x = 5 := 
sorry

end find_missing_number_l24_24965


namespace isosceles_triangle_area_l24_24196

-- Define the conditions for the isosceles triangle
def is_isosceles_triangle (a b c : ℝ) : Prop := a = b ∨ b = c ∨ a = c 

-- Define the side lengths
def side_length_1 : ℝ := 15
def side_length_2 : ℝ := 15
def side_length_3 : ℝ := 24

-- State the theorem
theorem isosceles_triangle_area :
  is_isosceles_triangle side_length_1 side_length_2 side_length_3 →
  side_length_1 = 15 →
  side_length_2 = 15 →
  side_length_3 = 24 →
  ∃ A : ℝ, (A = (1 / 2) * 24 * 9) ∧ A = 108 :=
sorry

end isosceles_triangle_area_l24_24196


namespace sin_double_angle_of_tan_l24_24492

theorem sin_double_angle_of_tan (α : ℝ) (hα1 : Real.tan α = 2) (hα2 : 0 < α ∧ α < Real.pi / 2) : Real.sin (2 * α) = 4 / 5 := by
  sorry

end sin_double_angle_of_tan_l24_24492


namespace subtracted_value_l24_24043

-- Given conditions
def chosen_number : ℕ := 110
def result_number : ℕ := 110

-- Statement to prove
theorem subtracted_value : ∃ y : ℕ, 3 * chosen_number - y = result_number ∧ y = 220 :=
by
  sorry

end subtracted_value_l24_24043


namespace arithmetic_geometric_sequence_S30_l24_24681

variable (S : ℕ → ℝ)

theorem arithmetic_geometric_sequence_S30 :
  S 10 = 10 →
  S 20 = 30 →
  S 30 = 70 := by
  intros h1 h2
  -- proof steps go here
  sorry

end arithmetic_geometric_sequence_S30_l24_24681


namespace rs_value_l24_24874

theorem rs_value (r s : ℝ) (hr : 0 < r) (hs: 0 < s) (h1 : r^2 + s^2 = 1) (h2 : r^4 + s^4 = 3 / 4) :
  r * s = Real.sqrt 2 / 4 :=
sorry

end rs_value_l24_24874


namespace base_five_to_base_ten_l24_24769

theorem base_five_to_base_ten : 
  let b := 5 in 
  let x := 123 % b in
  let y := (123 / b) % b in
  let z := (123 / b) / b in
  z * b^2 + y * b + x = 38 :=
by {
  let b := 5,
  let x := 123 % b,  -- least significant digit
  let y := (123 / b) % b,  -- middle digit
  let z := (123 / b) / b,  -- most significant digit
  have hx : x = 3 := by norm_num,  -- 123 % 5 = 3
  have hy : y = 2 := by norm_num,  -- (123 / 5) % 5 = 2
  have hz : z = 1 := by norm_num,  -- (123 / 5) / 5 = 1
  rw [hx, hy, hz],
  norm_num
}

end base_five_to_base_ten_l24_24769


namespace a_16_value_l24_24230

-- Define the recurrence relation
def seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0       => 2
  | (n + 1) => (1 + a n) / (1 - a n)

-- State the theorem
theorem a_16_value :
  seq (a : ℕ → ℚ) 16 = -1/3 := 
sorry

end a_16_value_l24_24230


namespace half_time_score_30_l24_24964

-- Define sequence conditions
def arithmetic_sequence (a d : ℕ) : ℕ × ℕ × ℕ × ℕ := (a, a + d, a + 2 * d, a + 3 * d)
def geometric_sequence (b r : ℕ) : ℕ × ℕ × ℕ × ℕ := (b, b * r, b * r^2, b * r^3)

-- Define the sum of the first team
def first_team_sum (a d : ℕ) : ℕ := 4 * a + 6 * d

-- Define the sum of the second team
def second_team_sum (b r : ℕ) : ℕ := b * (1 + r + r^2 + r^3)

-- Define the winning condition
def winning_condition (a d b r : ℕ) : Prop := first_team_sum a d = second_team_sum b r + 2

-- Define the point sum constraint
def point_sum_constraint (a d b r : ℕ) : Prop := first_team_sum a d ≤ 100 ∧ second_team_sum b r ≤ 100

-- Define the constraints on r and d
def r_d_positive (r d : ℕ) : Prop := r > 1 ∧ d > 0

-- Define the half-time score for the first team
def first_half_first_team (a d : ℕ) : ℕ := a + (a + d)

-- Define the half-time score for the second team
def first_half_second_team (b r : ℕ) : ℕ := b + (b * r)

-- Define the total half-time score
def total_half_time_score (a d b r : ℕ) : ℕ := first_half_first_team a d + first_half_second_team b r

-- Main theorem: Total half-time score is 30 under given conditions
theorem half_time_score_30 (a d b r : ℕ) 
  (r_d_pos : r_d_positive r d) 
  (win_cond : winning_condition a d b r)
  (point_sum_cond : point_sum_constraint a d b r) : 
  total_half_time_score a d b r = 30 :=
sorry

end half_time_score_30_l24_24964


namespace new_average_weight_l24_24126

-- noncomputable theory can be enabled if necessary for real number calculations.
-- noncomputable theory

def original_players : Nat := 7
def original_avg_weight : Real := 103
def new_players : Nat := 2
def weight_first_new_player : Real := 110
def weight_second_new_player : Real := 60

theorem new_average_weight :
  let original_total_weight : Real := original_players * original_avg_weight
  let total_weight : Real := original_total_weight + weight_first_new_player + weight_second_new_player
  let total_players : Nat := original_players + new_players
  total_weight / total_players = 99 := by
  sorry

end new_average_weight_l24_24126


namespace single_elimination_games_l24_24166

theorem single_elimination_games (n : ℕ) (h : n = 23) : 
  ∃ g : ℕ, g = n - 1 :=
by
  sorry

end single_elimination_games_l24_24166


namespace growth_rate_inequality_l24_24159

theorem growth_rate_inequality (a b x : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_x_pos : x > 0) :
  x ≤ (a + b) / 2 :=
sorry

end growth_rate_inequality_l24_24159


namespace probability_composite_product_l24_24531

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l24_24531


namespace smallest_expression_l24_24698

theorem smallest_expression (a b : ℝ) (h : b < 0) : a + b < a ∧ a < a - b :=
by
  sorry

end smallest_expression_l24_24698


namespace midpoint_trajectory_l24_24940

theorem midpoint_trajectory (x y : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (8, 0) ∧ (B.1, B.2) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 } ∧ 
   ∃ P : ℝ × ℝ, P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ P = (x, y)) → (x - 4)^2 + y^2 = 1 :=
by sorry

end midpoint_trajectory_l24_24940


namespace sum_mean_median_mode_eq_l24_24052

open List

def the_list : List ℕ := [1, 2, 2, 4, 5, 5, 5, 7, 8]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def mode (l : List ℕ) : ℕ :=
  l.mode

def median (l : List ℕ) : ℕ :=
  l.median

theorem sum_mean_median_mode_eq :
  mean the_list + median the_list + mode the_list = 43 / 3 := by sorry

end sum_mean_median_mode_eq_l24_24052


namespace binomial_12_6_eq_924_l24_24810

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := sorry

end binomial_12_6_eq_924_l24_24810


namespace number_of_ways_to_select_numbers_with_even_sum_l24_24086

theorem number_of_ways_to_select_numbers_with_even_sum :
  ∃ (S : Finset ℕ) (n : ℕ), S ⊆ Finset.range 10 ∧ S.card = 4 ∧ (S.sum id) % 2 = 0 ∧ (Finset.card (Finset.filter (λ S, (S.sum id) % 2 = 0) (Finset.powersetLen 4 (Finset.range 10)))) = 66 :=
by
  let S := Finset.range 10
  existsi S, 4
  split
  { exact Finset.subset.refl _ }
  split
  { exact Finset.card_range 10 }
  split
  { sorry }
  { sorry }


end number_of_ways_to_select_numbers_with_even_sum_l24_24086


namespace min_value_expression_l24_24388

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 4*x + 2) * (y^2 + 4*y + 2) * (z^2 + 4*z + 2) / (x * y * z) ≥ 216 :=
by
  sorry

end min_value_expression_l24_24388


namespace gcd_of_sum_of_cubes_and_increment_l24_24672

theorem gcd_of_sum_of_cubes_and_increment {n : ℕ} (h : n > 3) : Nat.gcd (n^3 + 27) (n + 4) = 1 :=
by sorry

end gcd_of_sum_of_cubes_and_increment_l24_24672


namespace sarah_problem_solution_l24_24410

def two_digit_number := {x : ℕ // 10 ≤ x ∧ x < 100}
def three_digit_number := {y : ℕ // 100 ≤ y ∧ y < 1000}

theorem sarah_problem_solution (x : two_digit_number) (y : three_digit_number) 
    (h_eq : 1000 * x.1 + y.1 = 8 * x.1 * y.1) : 
    x.1 = 15 ∧ y.1 = 126 ∧ (x.1 + y.1 = 141) := 
by 
  sorry

end sarah_problem_solution_l24_24410


namespace part_I_solution_part_II_solution_l24_24689

-- Definitions for the problem
def f (x a : ℝ) : ℝ := |x - a| + |x - 1|

-- Part I: When a = 2, solve the inequality f(x) < 4
theorem part_I_solution (x : ℝ) : f x 2 < 4 ↔ x > -1/2 ∧ x < 7/2 :=
by sorry

-- Part II: Range of values for a such that f(x) ≥ 2 for all x
theorem part_II_solution (a : ℝ) : (∀ x, f x a ≥ 2) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by sorry

end part_I_solution_part_II_solution_l24_24689


namespace find_fraction_value_l24_24570

variable (a b : ℝ)
variable (h1 : b > a)
variable (h2 : a > 0)
variable (h3 : a / b + b / a = 4)

theorem find_fraction_value (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a / b + b / a = 4) : (a + b) / (a - b) = Real.sqrt 3 := by
  sorry

end find_fraction_value_l24_24570


namespace sqrt_meaningful_iff_ge_two_l24_24696

-- State the theorem according to the identified problem and conditions
theorem sqrt_meaningful_iff_ge_two (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 2)) → x ≥ 2 :=
by
  sorry  -- Proof placeholder

end sqrt_meaningful_iff_ge_two_l24_24696


namespace correct_growth_rate_equation_l24_24892

noncomputable def numberOfBikesFirstMonth : ℕ := 1000
noncomputable def additionalBikesThirdMonth : ℕ := 440
noncomputable def monthlyGrowthRate (x : ℝ) : Prop :=
  numberOfBikesFirstMonth * (1 + x)^2 = numberOfBikesFirstMonth + additionalBikesThirdMonth

theorem correct_growth_rate_equation (x : ℝ) : monthlyGrowthRate x :=
by
  sorry

end correct_growth_rate_equation_l24_24892


namespace calculate_expression_l24_24133

theorem calculate_expression : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end calculate_expression_l24_24133


namespace recurring_decimal_numerator_count_l24_24428

theorem recurring_decimal_numerator_count :
  let nums := {n | 1 ≤ n ∧ n ≤ 999};
  let rel_prime := {n | n ∈ nums ∧ Nat.gcd n 999 = 1};
  let additional := {n | n ∈ nums ∧ n % 81 = 0};
  (Card rel_prime + Card additional = 660) :=
by
  sorry

end recurring_decimal_numerator_count_l24_24428


namespace range_of_a_l24_24692

open Set

variable (a : ℝ)

noncomputable def I := univ ℝ
noncomputable def A := {x : ℝ | x ≤ a + 1}
noncomputable def B := {x : ℝ | x ≥ 1}
noncomputable def complement_B := {x : ℝ | x < 1}

theorem range_of_a (h : A a ⊆ complement_B) : a < 0 := sorry

end range_of_a_l24_24692


namespace compute_binom_12_6_eq_1848_l24_24820

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end compute_binom_12_6_eq_1848_l24_24820


namespace stating_sum_first_10_common_elements_l24_24976

/-- 
  Theorem stating that the sum of the first 10 elements that appear 
  in both the given arithmetic progression and the geometric progression 
  equals 13981000.
-/
theorem sum_first_10_common_elements :
  let AP := λ n : ℕ => 4 + 3 * n in
  let GP := λ k : ℕ => 20 * 2^k in
  let common_elements := (range 10).map (λ i => GP (2 * i + 1)) in
  (∑ i in (finset.range 10), common_elements[i]) = 13981000 :=
by
  let AP := λ n : ℕ => 4 + 3 * n
  let GP := λ k : ℕ => 20 * 2^k
  let common_elements := (finset.range 10).map (λ i => GP (2 * i + 1))
  have S : (∑ i in (finset.range 10), common_elements[i]) = 40 * (4^10 - 1) / 3,
  {
    sorry,
  }
  have : 40 * 349525 = 13981000 := by norm_num,
  exact this ▸ S

end stating_sum_first_10_common_elements_l24_24976


namespace solve_system_of_equations_l24_24742

theorem solve_system_of_equations :
  ∃ x y : ℝ, 4 * x - 6 * y = -3 ∧ 9 * x + 3 * y = 6.3 ∧ x = 0.436 ∧ y = 0.792 :=
by
  sorry

end solve_system_of_equations_l24_24742


namespace composite_dice_product_probability_l24_24537

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l24_24537


namespace sum_lent_out_l24_24657

theorem sum_lent_out (P R : ℝ) (h1 : 780 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 7) / 100) : P = 684 := 
  sorry

end sum_lent_out_l24_24657


namespace fewest_printers_l24_24456

theorem fewest_printers (x y : ℕ) (h1 : 375 * x = 150 * y) : x + y = 7 :=
  sorry

end fewest_printers_l24_24456


namespace andy_incorrect_l24_24472

theorem andy_incorrect (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 6) (h3 : c = 8) : a = 14 :=
by
  sorry

end andy_incorrect_l24_24472


namespace kabulek_four_digits_l24_24900

def isKabulekNumber (N: ℕ) : Prop :=
  let a := N / 100
  let b := N % 100
  (a + b) ^ 2 = N

theorem kabulek_four_digits :
  {N : ℕ | 1000 ≤ N ∧ N < 10000 ∧ isKabulekNumber N} = {2025, 3025, 9801} :=
by sorry

end kabulek_four_digits_l24_24900


namespace parallel_implies_eq_diagonals_l24_24383

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

structure Quadrilateral (A B C D O H1 H2 M N : V) : Prop :=
  (convex : Convex ℝ {x : V | x = A ∨ x = B ∨ x = C ∨ x = D})
  (diagonals_not_perpendicular : ∀ (u v : V), u ≠ 0 → v ≠ 0 → ∠ u v ≠ π / 2)
  (sides_not_parallel : ∀ (u v : V), ∥u∥ ≠ 0 → ∥v∥ ≠ 0 → u ≠ v)
  (O_intersection : ∀ (l1 l2 : Line V), line_through l1 A C ∧ line_through l1 B D → l2 = O)
  (midpoint_M : M = (A + B) / 2)
  (midpoint_N : N = (C + D) / 2)
  (orthocenter_H1 : H1 ⊆ Orthocenter A O B)
  (orthocenter_H2 : H2 ⊆ Orthocenter C O D)

theorem parallel_implies_eq_diagonals
  {A B C D O H1 H2 M N : V}
  (q : Quadrilateral A B C D O H1 H2 M N) :
  (∀ (u v : V), u || v → u = v) ↔ (∥A - C∥ = ∥B - D∥) :=
sorry

end parallel_implies_eq_diagonals_l24_24383


namespace initial_apples_count_l24_24609

variable (initial_apples : ℕ)
variable (used_apples : ℕ := 2)
variable (bought_apples : ℕ := 23)
variable (final_apples : ℕ := 38)

theorem initial_apples_count :
  initial_apples - used_apples + bought_apples = final_apples ↔ initial_apples = 17 := by
  sorry

end initial_apples_count_l24_24609


namespace max_area_curves_intersection_l24_24584

open Real

def C₁ (x : ℝ) : ℝ := x^3 - x
def C₂ (x a : ℝ) : ℝ := (x - a)^3 - (x - a)

theorem max_area_curves_intersection (a : ℝ) (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ C₁ x₁ = C₂ x₁ a ∧ C₁ x₂ = C₂ x₂ a) :
  ∃ A_max : ℝ, A_max = 3 / 4 :=
by
  -- TODO: Provide the proof here
  sorry

end max_area_curves_intersection_l24_24584


namespace ratio_of_segments_l24_24835

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end ratio_of_segments_l24_24835


namespace odd_even_divisors_ratio_l24_24385

theorem odd_even_divisors_ratio (M : ℕ) (h1 : M = 2^5 * 3^5 * 5 * 7^3) :
  let sum_odd_divisors := (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5) * (1 + 5) * (1 + 7 + 7^2 + 7^3)
  let sum_all_divisors := (1 + 2 + 4 + 8 + 16 + 32) * (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5) * (1 + 5) * (1 + 7 + 7^2 + 7^3)
  let sum_even_divisors := sum_all_divisors - sum_odd_divisors
  sum_odd_divisors / sum_even_divisors = 1 / 62 :=
by
  sorry

end odd_even_divisors_ratio_l24_24385


namespace greatest_three_digit_multiple_23_l24_24620

theorem greatest_three_digit_multiple_23 : 
  ∃ n : ℕ, n < 1000 ∧ n % 23 = 0 ∧ (∀ m : ℕ, m < 1000 ∧ m % 23 = 0 → m ≤ n) ∧ n = 989 :=
sorry

end greatest_three_digit_multiple_23_l24_24620


namespace part1_inequality_part2_inequality_l24_24107

theorem part1_inequality (x : ℝ) : 
  (3 * x - 2) / (x - 1) > 1 ↔ x > 1 ∨ x < 1 / 2 := 
by sorry

theorem part2_inequality (x a : ℝ) : 
  x^2 - a * x - 2 * a^2 < 0 ↔ 
  (a = 0 → False) ∧ 
  (a > 0 → -a < x ∧ x < 2 * a) ∧ 
  (a < 0 → 2 * a < x ∧ x < -a) := 
by sorry

end part1_inequality_part2_inequality_l24_24107


namespace taco_castle_trucks_l24_24912

theorem taco_castle_trucks :
  ∀ (ford_trucks toyota_trucks vw_bugs dodge_trucks : ℕ),
    (ford_trucks = 2 * toyota_trucks) →
    (vw_bugs = toyota_trucks / 2) →
    (vw_bugs = 5) →
    (dodge_trucks = 60) →
    ford_trucks / dodge_trucks = 1 / 3 := 
by
  intros ford_trucks toyota_trucks vw_bugs dodge_trucks
  intros h1 h2 h3 h4
  sorry

end taco_castle_trucks_l24_24912


namespace parakeets_in_each_cage_l24_24039

variable (num_cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ)

-- Given conditions
def total_parrots (num_cages parrots_per_cage : ℕ) : ℕ := num_cages * parrots_per_cage
def total_parakeets (total_birds total_parrots : ℕ) : ℕ := total_birds - total_parrots
def parakeets_per_cage (total_parakeets num_cages : ℕ) : ℕ := total_parakeets / num_cages

-- Theorem: Number of parakeets in each cage is 7
theorem parakeets_in_each_cage (h1 : num_cages = 8) (h2 : parrots_per_cage = 2) (h3 : total_birds = 72) : 
  parakeets_per_cage (total_parakeets total_birds (total_parrots num_cages parrots_per_cage)) num_cages = 7 :=
by
  sorry

end parakeets_in_each_cage_l24_24039


namespace shoes_ratio_l24_24104

theorem shoes_ratio (Scott_shoes : ℕ) (m : ℕ) (h1 : Scott_shoes = 7)
  (h2 : ∀ Anthony_shoes, Anthony_shoes = m * Scott_shoes)
  (h3 : ∀ Jim_shoes, Jim_shoes = Anthony_shoes - 2)
  (h4 : ∀ Anthony_shoes Jim_shoes, Anthony_shoes = Jim_shoes + 2) : 
  ∃ m : ℕ, (Anthony_shoes / Scott_shoes) = m := 
by 
  sorry

end shoes_ratio_l24_24104


namespace greatest_divisor_of_976543_and_897623_l24_24148

theorem greatest_divisor_of_976543_and_897623 :
  Nat.gcd (976543 - 7) (897623 - 11) = 4 := by
  sorry

end greatest_divisor_of_976543_and_897623_l24_24148


namespace fraction_painted_red_l24_24400

theorem fraction_painted_red :
  let matilda_section := (1:ℚ) / 2 -- Matilda's half section
  let ellie_section := (1:ℚ) / 2    -- Ellie's half section
  let matilda_painted := matilda_section / 2 -- Matilda's painted fraction
  let ellie_painted := ellie_section / 3    -- Ellie's painted fraction
  (matilda_painted + ellie_painted) = 5 / 12 := 
by
  sorry

end fraction_painted_red_l24_24400


namespace total_chairs_calc_l24_24069

-- Defining the condition of having 27 rows
def rows : ℕ := 27

-- Defining the condition of having 16 chairs per row
def chairs_per_row : ℕ := 16

-- Stating the theorem that the total number of chairs is 432
theorem total_chairs_calc : rows * chairs_per_row = 432 :=
by
  sorry

end total_chairs_calc_l24_24069


namespace integer_roots_iff_floor_square_l24_24090

variable (α β : ℝ)
variable (m n : ℕ)
variable (real_roots : α^2 - m*α + n = 0 ∧ β^2 - m*β + n = 0)

noncomputable def are_integers (α β : ℝ) : Prop := (∃ (a b : ℤ), α = a ∧ β = b)

theorem integer_roots_iff_floor_square (m n : ℕ) (α β : ℝ)
  (hmn : 0 ≤ m ∧ 0 ≤ n)
  (roots_real : α^2 - m*α + n = 0 ∧ β^2 - m*β + n = 0) :
  (are_integers α β) ↔ (∃ k : ℤ, (⌊m * α⌋ + ⌊m * β⌋) = k^2) :=
sorry

end integer_roots_iff_floor_square_l24_24090


namespace monotonic_decreasing_interval_l24_24603

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem monotonic_decreasing_interval : 
  ∀ x : ℝ, x < 2 → f' x < 0 :=
by
  intro x hx
  sorry

end monotonic_decreasing_interval_l24_24603


namespace selection_methods_l24_24644

theorem selection_methods :
  ∃ (ways_with_girls : ℕ), ways_with_girls = Nat.choose 6 4 - Nat.choose 4 4 ∧ ways_with_girls = 14 := by
  sorry

end selection_methods_l24_24644


namespace find_set_l24_24098

/-- Definition of set A -/
def setA : Set ℝ := { x : ℝ | abs x < 4 }

/-- Definition of set B -/
def setB : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 > 0 }

/-- Definition of the intersection A ∩ B -/
def intersectionAB : Set ℝ := { x : ℝ | abs x < 4 ∧ (x > 3 ∨ x < 1) }

/-- Definition of the set we want to find -/
def setDesired : Set ℝ := { x : ℝ | abs x < 4 ∧ ¬(abs x < 4 ∧ (x > 3 ∨ x < 1)) }

/-- The statement to prove -/
theorem find_set :
  setDesired = { x : ℝ | 1 ≤ x ∧ x ≤ 3 } :=
sorry

end find_set_l24_24098


namespace prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l24_24708

variables (n k : ℕ) (p : ℝ)
variables (h_pos : 0 < p) (h_lt_one : p < 1)

-- Probability that exactly k gnomes fall
theorem prob_exactly_k_gnomes_fall (h_k_le_n : k ≤ n) :
  prob_speed (exactly_k_gnomes_fall n k p) = p * (1 - p)^(n - k) := sorry

-- Expected number of fallen gnomes
theorem expected_fallen_gnomes : 
  expected_falls n p = n + 1 - 1/p + (1 - p)^(n + 1)/p := sorry

end prob_exactly_k_gnomes_fall_expected_fallen_gnomes_l24_24708


namespace amoeba_count_14_l24_24233

noncomputable def amoeba_count (day : ℕ) : ℕ :=
  if day = 1 then 1
  else if day = 2 then 2
  else 2^(day - 3) * 5

theorem amoeba_count_14 : amoeba_count 14 = 10240 := by
  sorry

end amoeba_count_14_l24_24233


namespace jake_earnings_per_hour_l24_24564

-- Definitions for conditions
def initialDebt : ℕ := 100
def payment : ℕ := 40
def hoursWorked : ℕ := 4
def remainingDebt : ℕ := initialDebt - payment

-- Theorem stating Jake's earnings per hour
theorem jake_earnings_per_hour : remainingDebt / hoursWorked = 15 := by
  sorry

end jake_earnings_per_hour_l24_24564


namespace find_natural_numbers_eq_36_sum_of_digits_l24_24191

-- Define the sum of digits function
def sum_of_digits (x : ℕ) : ℕ := 
  if x = 0 then 0
  else sum_of_digits (x / 10) + (x % 10)

-- Lean theorem statement proving the given problem
theorem find_natural_numbers_eq_36_sum_of_digits :
  {x : ℕ | x = 36 * (sum_of_digits x)} = {324, 648} :=
sorry

end find_natural_numbers_eq_36_sum_of_digits_l24_24191


namespace lizzy_wealth_after_loan_l24_24395

theorem lizzy_wealth_after_loan 
  (initial_wealth : ℕ)
  (loan : ℕ)
  (interest_rate : ℕ)
  (h1 : initial_wealth = 30)
  (h2 : loan = 15)
  (h3 : interest_rate = 20)
  : (initial_wealth - loan) + (loan + loan * interest_rate / 100) = 33 :=
by
  sorry

end lizzy_wealth_after_loan_l24_24395


namespace probability_composite_l24_24552

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def non_composite_outcomes : ℕ := 1 + 12

theorem probability_composite :
  let total_outcomes := 6^4 in
  let non_composite := non_composite_outcomes in
  let composite_probability := (total_outcomes - non_composite) / total_outcomes.toRat in
  composite_probability = 1283 / 1296 := by
  let total_outcomes := 6^4
  let non_composite := 1 + 12
  let composite_probability := (total_outcomes - non_composite).toRat / total_outcomes.toRat
  sorry

end probability_composite_l24_24552


namespace total_number_of_crickets_l24_24639

def initial_crickets : ℝ := 7.0
def additional_crickets : ℝ := 11.0
def total_crickets : ℝ := 18.0

theorem total_number_of_crickets :
  initial_crickets + additional_crickets = total_crickets :=
by
  sorry

end total_number_of_crickets_l24_24639


namespace annual_population_increase_l24_24266

theorem annual_population_increase (P₀ P₂ : ℝ) (r : ℝ) 
  (h0 : P₀ = 12000) 
  (h2 : P₂ = 18451.2) 
  (h_eq : P₂ = P₀ * (1 + r / 100)^2) :
  r = 24 :=
by
  sorry

end annual_population_increase_l24_24266


namespace train_speed_l24_24464

theorem train_speed 
(length_of_train : ℕ) 
(time_to_cross_pole : ℕ) 
(h_length : length_of_train = 135) 
(h_time : time_to_cross_pole = 9) : 
  (length_of_train / time_to_cross_pole) * 3.6 = 54 :=
by 
  sorry

end train_speed_l24_24464


namespace average_salary_of_all_workers_l24_24002

theorem average_salary_of_all_workers :
  let technicians := 7
  let technicians_avg_salary := 20000
  let rest := 49 - technicians
  let rest_avg_salary := 6000
  let total_workers := 49
  let total_tech_salary := technicians * technicians_avg_salary
  let total_rest_salary := rest * rest_avg_salary
  let total_salary := total_tech_salary + total_rest_salary
  (total_salary / total_workers) = 8000 := by
  sorry

end average_salary_of_all_workers_l24_24002


namespace function_intersects_line_at_most_once_l24_24119

variable {α β : Type} [Nonempty α]

def function_intersects_at_most_once (f : α → β) (a : α) : Prop :=
  ∀ (b b' : β), f a = b → f a = b' → b = b'

theorem function_intersects_line_at_most_once {α β : Type} [Nonempty α] (f : α → β) (a : α) :
  function_intersects_at_most_once f a :=
by
  sorry

end function_intersects_line_at_most_once_l24_24119


namespace calculate_result_l24_24176

theorem calculate_result :
  (-24) * ((5 / 6 : ℚ) - (4 / 3) + (5 / 8)) = -3 := 
by
  sorry

end calculate_result_l24_24176


namespace correct_growth_rate_equation_l24_24893

noncomputable def numberOfBikesFirstMonth : ℕ := 1000
noncomputable def additionalBikesThirdMonth : ℕ := 440
noncomputable def monthlyGrowthRate (x : ℝ) : Prop :=
  numberOfBikesFirstMonth * (1 + x)^2 = numberOfBikesFirstMonth + additionalBikesThirdMonth

theorem correct_growth_rate_equation (x : ℝ) : monthlyGrowthRate x :=
by
  sorry

end correct_growth_rate_equation_l24_24893


namespace horse_food_per_day_l24_24321

theorem horse_food_per_day (ratio_sh : ℕ) (ratio_h : ℕ) (sheep : ℕ) (total_food : ℕ) (sheep_count : sheep = 32) (ratio : ratio_sh = 4) (ratio_horses : ratio_h = 7) (total_food_need : total_food = 12880) :
  total_food / (sheep * ratio_h / ratio_sh) = 230 :=
by
  sorry

end horse_food_per_day_l24_24321


namespace polynomial_of_degree_2_l24_24568

noncomputable def polynomialSeq (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    ∃ (f_k f_k1 f_k2 : Polynomial ℝ),
      f_k ≠ Polynomial.C 0 ∧ (f_k * f_k1 = f_k1.comp f_k2)

theorem polynomial_of_degree_2 (n : ℕ) (h : n ≥ 3) :
  polynomialSeq n → 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    ∃ f : Polynomial ℝ, f = Polynomial.X ^ 2 :=
sorry

end polynomial_of_degree_2_l24_24568


namespace solution_l24_24973

noncomputable def problem (x : ℝ) : Prop :=
  (Real.sqrt (Real.sqrt (53 - 3 * x)) + Real.sqrt (Real.sqrt (39 + 3 * x))) = 5

theorem solution :
  ∀ x : ℝ, problem x → x = -23 / 3 :=
by
  intro x
  intro h
  sorry

end solution_l24_24973


namespace green_paint_mixture_l24_24978

theorem green_paint_mixture :
  ∀ (x : ℝ), 
    let light_green_paint := 5
    let darker_green_paint := x
    let final_paint := light_green_paint + darker_green_paint
    1 + 0.4 * darker_green_paint = 0.25 * final_paint -> x = 5 / 3 := 
by 
  intros x
  let light_green_paint := 5
  let darker_green_paint := x
  let final_paint := light_green_paint + darker_green_paint
  sorry

end green_paint_mixture_l24_24978


namespace birds_in_marsh_end_of_day_l24_24864

def geese_initial : Nat := 58
def ducks : Nat := 37
def geese_flew_away : Nat := 15
def swans : Nat := 22
def herons : Nat := 2

theorem birds_in_marsh_end_of_day : 
  58 - 15 + 37 + 22 + 2 = 104 := by
  sorry

end birds_in_marsh_end_of_day_l24_24864


namespace total_water_heaters_l24_24922

-- Define the conditions
variables (W C : ℕ) -- W: capacity of Wallace's water heater, C: capacity of Catherine's water heater
variable (wallace_3over4_full : W = 40 ∧ W * 3 / 4 ∧ C = W / 2 ∧ C * 3 / 4)

-- The proof problem
theorem total_water_heaters (wallace_3over4_full : W = 40 ∧ (W * 3 / 4 = 30) ∧ C = W / 2 ∧ (C * 3 / 4 = 15)) : W * 3 / 4 + C * 3 / 4 = 45 :=
sorry

end total_water_heaters_l24_24922


namespace binomial_coefficient_ratio_l24_24113

theorem binomial_coefficient_ratio (n k : ℕ) (h₁ : n = 4 * k + 3) (h₂ : n = 3 * k + 5) : n + k = 13 :=
by
  sorry

end binomial_coefficient_ratio_l24_24113


namespace average_speed_is_50_l24_24907

-- Defining the conditions
def totalDistance : ℕ := 250
def totalTime : ℕ := 5

-- Defining the average speed
def averageSpeed := totalDistance / totalTime

-- The theorem statement
theorem average_speed_is_50 : averageSpeed = 50 := sorry

end average_speed_is_50_l24_24907


namespace angle_P_measure_l24_24258

theorem angle_P_measure (P Q R S : ℝ) 
  (h1 : P = 3 * Q)
  (h2 : P = 4 * R)
  (h3 : P = 6 * S)
  (h_sum : P + Q + R + S = 360) : 
  P = 206 :=
by 
  sorry

end angle_P_measure_l24_24258


namespace worst_ranking_l24_24275

theorem worst_ranking (teams : Fin 25 → Nat) (A : Fin 25)
  (round_robin : ∀ i j, i ≠ j → teams i + teams j ≤ 4)
  (most_goals : ∀ i, i ≠ A → teams A > teams i)
  (fewest_goals : ∀ i, i ≠ A → teams i > teams A) :
  ∃ ranking : Fin 25 → Fin 25, ranking A = 24 :=
by
  sorry

end worst_ranking_l24_24275


namespace lizzy_wealth_after_loan_l24_24396

theorem lizzy_wealth_after_loan 
  (initial_wealth : ℕ)
  (loan : ℕ)
  (interest_rate : ℕ)
  (h1 : initial_wealth = 30)
  (h2 : loan = 15)
  (h3 : interest_rate = 20)
  : (initial_wealth - loan) + (loan + loan * interest_rate / 100) = 33 :=
by
  sorry

end lizzy_wealth_after_loan_l24_24396


namespace not_even_not_odd_neither_even_nor_odd_l24_24563

noncomputable def f (x : ℝ) : ℝ := ⌊x⌋ + 1 / 2

theorem not_even (x : ℝ) : f (-x) ≠ f x := sorry
theorem not_odd (x : ℝ) : f (0) ≠ 0 ∨ f (-x) ≠ -f x := sorry

theorem neither_even_nor_odd : ∀ x : ℝ, f (-x) ≠ f x ∧ (f (0) ≠ 0 ∨ f (-x) ≠ -f x) :=
by
  intros x
  exact ⟨not_even x, not_odd x⟩

end not_even_not_odd_neither_even_nor_odd_l24_24563


namespace find_value_of_A_l24_24421

theorem find_value_of_A (A B : ℤ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 :=
by
  sorry

end find_value_of_A_l24_24421


namespace no_sol_n4_minus_m4_eq_42_l24_24960

theorem no_sol_n4_minus_m4_eq_42 :
  ¬ ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ n^4 - m^4 = 42 :=
by
  sorry

end no_sol_n4_minus_m4_eq_42_l24_24960


namespace count_distinct_x_l24_24513

theorem count_distinct_x :
  { x : ℝ | ∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ s = floor (sqrt (123 - sqrt x)) } = 12 :=
sorry

end count_distinct_x_l24_24513


namespace tan_sum_sin_cos_conditions_l24_24525

theorem tan_sum_sin_cos_conditions {x y : ℝ} 
  (h1 : Real.sin x + Real.sin y = 1 / 2) 
  (h2 : Real.cos x + Real.cos y = Real.sqrt 3 / 2) :
  Real.tan x + Real.tan y = -Real.sqrt 3 := 
sorry

end tan_sum_sin_cos_conditions_l24_24525


namespace diagonals_in_nine_sided_polygon_l24_24999

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l24_24999


namespace minimize_x_plus_y_on_circle_l24_24496

theorem minimize_x_plus_y_on_circle (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) : x + y ≥ 2 :=
by
  sorry

end minimize_x_plus_y_on_circle_l24_24496


namespace problem_l24_24244

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x + x^2 - 3

theorem problem (a b : ℝ) (H1 : f a = 0) (H2 : g b = 0) : g a < 0 ∧ 0 < f b :=
by
  sorry

end problem_l24_24244


namespace function_monotonic_decreasing_interval_l24_24687

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6)

theorem function_monotonic_decreasing_interval :
  ∀ x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
  ∀ y ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
  (x ≤ y → f y ≤ f x) :=
by
  sorry

end function_monotonic_decreasing_interval_l24_24687


namespace limit_does_not_exist_l24_24335

noncomputable def does_not_exist_limit : Prop := 
  ¬ ∃ l : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    (0 < |x| ∧ 0 < |y| ∧ |x| < δ ∧ |y| < δ) →
    |(x^2 - y^2) / (x^2 + y^2) - l| < ε

theorem limit_does_not_exist :
  does_not_exist_limit :=
sorry

end limit_does_not_exist_l24_24335


namespace acute_triangle_angle_A_range_of_bc_l24_24715

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}
variable (Δ : ∃ (A B C : ℝ), a = sqrt 2 ∧ ∀ (a b c A B C : ℝ), 
  (a = sqrt 2) ∧ (b = b) ∧ (c = c) ∧ 
  (sin A * cos A / cos (A + C) = a * c / (b^2 - a^2 - c^2)))

-- Problem statement
theorem acute_triangle_angle_A (h : Δ) : A = π / 4 :=
sorry

theorem range_of_bc (h : Δ) : 0 < b * c ∧ b * c ≤ 2 + sqrt 2 :=
sorry

end acute_triangle_angle_A_range_of_bc_l24_24715


namespace solution_set_equivalence_l24_24981

theorem solution_set_equivalence (a : ℝ) : 
    (-1 < a ∧ a < 1) ∧ (3 * a^2 - 2 * a - 5 < 0) → 
    (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) :=
by
    sorry

end solution_set_equivalence_l24_24981


namespace sequence_value_238_l24_24946

theorem sequence_value_238 (a : ℕ → ℚ) :
  (a 1 = 1) ∧
  (∀ n, n ≥ 2 → (n % 2 = 0 → a n = a (n - 1) / 2 + 1) ∧ (n % 2 = 1 → a n = 1 / a (n - 1))) ∧
  (∃ n, a n = 30 / 19) → ∃ n, a n = 30 / 19 ∧ n = 238 :=
by
  sorry

end sequence_value_238_l24_24946


namespace sum_factors_of_30_l24_24632

theorem sum_factors_of_30 : (1 + 2 + 3 + 5 + 6 + 10 + 15 + 30) = 72 :=
by
  sorry

end sum_factors_of_30_l24_24632


namespace tricycles_count_l24_24955

-- Define the conditions
variable (b t s : ℕ)

def total_children := b + t + s = 10
def total_wheels := 2 * b + 3 * t + 2 * s = 29

-- Provide the theorem to prove
theorem tricycles_count (h1 : total_children b t s) (h2 : total_wheels b t s) : t = 9 := 
by
  sorry

end tricycles_count_l24_24955


namespace gilled_mushrooms_count_l24_24455

def mushrooms_problem (G S : ℕ) : Prop :=
  (S = 9 * G) ∧ (G + S = 30) → (G = 3)

-- The theorem statement corresponding to the problem
theorem gilled_mushrooms_count (G S : ℕ) : mushrooms_problem G S :=
by {
  sorry
}

end gilled_mushrooms_count_l24_24455


namespace smallest_b_perfect_fourth_power_l24_24155

theorem smallest_b_perfect_fourth_power:
  ∃ b : ℕ, (∀ n : ℕ, 5 * n = (7 * b^2 + 7 * b + 7) → ∃ x : ℕ, n = x^4) 
  ∧ b = 41 :=
sorry

end smallest_b_perfect_fourth_power_l24_24155


namespace find_alpha_l24_24201

theorem find_alpha (α : ℝ) (h1 : Real.tan α = -1) (h2 : 0 < α ∧ α ≤ Real.pi) : α = 3 * Real.pi / 4 :=
sorry

end find_alpha_l24_24201


namespace sufficient_not_necessary_l24_24694

theorem sufficient_not_necessary (x : ℝ) : (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 1) ∧ ¬((x ≠ 1) → (x^2 - 3 * x + 2 ≠ 0)) :=
by
  sorry

end sufficient_not_necessary_l24_24694


namespace sean_bought_two_soups_l24_24252

theorem sean_bought_two_soups :
  ∃ (number_of_soups : ℕ),
    let soda_cost := 1
    let total_soda_cost := 3 * soda_cost
    let soup_cost := total_soda_cost
    let sandwich_cost := 3 * soup_cost
    let total_cost := 3 * soda_cost + sandwich_cost + soup_cost * number_of_soups
    total_cost = 18 ∧ number_of_soups = 2 :=
by
  sorry

end sean_bought_two_soups_l24_24252


namespace probability_red_or_blue_is_713_l24_24304

-- Definition of area ratios
def area_ratio_red : ℕ := 6
def area_ratio_yellow : ℕ := 2
def area_ratio_blue : ℕ := 1
def area_ratio_black : ℕ := 4

-- Total area ratio
def total_area_ratio := area_ratio_red + area_ratio_yellow + area_ratio_blue + area_ratio_black

-- Probability of stopping on either red or blue
def probability_red_or_blue := (area_ratio_red + area_ratio_blue) / total_area_ratio

-- Theorem stating the probability is 7/13
theorem probability_red_or_blue_is_713 : probability_red_or_blue = 7 / 13 :=
by
  unfold probability_red_or_blue total_area_ratio area_ratio_red area_ratio_blue
  simp
  sorry

end probability_red_or_blue_is_713_l24_24304


namespace monty_hall_probability_l24_24369

noncomputable theory
open_locale classical

-- Representing the Monty Hall problem with appropriate probability
def monty_hall := {door : Type* // fintype door}

variables {door : Type*} [fintype door] [decidable_eq door] (car goat1 goat2 : door)

-- The doors are distinct
axiom car_goat1_distinct : car ≠ goat1
axiom car_goat2_distinct : car ≠ goat2
axiom goat1_goat2_distinct : goat1 ≠ goat2

-- Initial probabilities
def initial_probability (choose : door) (prize : door) : ℝ :=
  if choose = prize then 1 / 3 else 2 / 3

-- After host reveals a goat
def switch_probability (choose : door) (open : door) (prize : door) : ℝ :=
  if choose = prize then 1 / 3 else 2 / 3

theorem monty_hall_probability
  (choice : door) (revealed_goat : door)
  (h : revealed_goat ≠ choice) (h' : revealed_goat = goat1 ∨ revealed_goat = goat2) :
  initial_probability choice car = 1 / 3 ∧ switch_probability choice revealed_goat car = 2 / 3 :=
sorry

end monty_hall_probability_l24_24369


namespace ratio_of_quadratic_roots_l24_24412

theorem ratio_of_quadratic_roots (a b c : ℝ) (h : 2 * b^2 = 9 * a * c) : 
  ∃ (x₁ x₂ : ℝ), (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ (x₁ / x₂ = 2) :=
sorry

end ratio_of_quadratic_roots_l24_24412


namespace product_of_roots_l24_24974

theorem product_of_roots:
  let a : ℕ := 24,
      b : ℕ := 36,
      c : ℤ := -648,
      prod_roots : ℤ := c / (a : ℤ)
  in prod_roots = -27 :=
by 
  let a : ℤ := 24
  let c : ℤ := -648
  have product_of_roots : ℤ := c / a
  have h : product_of_roots = -27 := by norm_num
  exact h

end product_of_roots_l24_24974


namespace scientific_notation_correct_l24_24577

-- Define the given number
def given_number : ℕ := 138000

-- Define the scientific notation expression
def scientific_notation : ℝ := 1.38 * 10^5

-- The proof goal: Prove that 138,000 expressed in scientific notation is 1.38 * 10^5
theorem scientific_notation_correct : (given_number : ℝ) = scientific_notation := by
  -- Sorry is used to skip the proof
  sorry

end scientific_notation_correct_l24_24577


namespace number_of_people_who_didnt_do_both_l24_24375

def total_graduates : ℕ := 73
def graduates_both : ℕ := 13

theorem number_of_people_who_didnt_do_both : total_graduates - graduates_both = 60 :=
by
  sorry

end number_of_people_who_didnt_do_both_l24_24375


namespace smallest_n_modulo_l24_24024

theorem smallest_n_modulo (
  n : ℕ
) (h1 : 17 * n ≡ 5678 [MOD 11]) : n = 4 :=
by sorry

end smallest_n_modulo_l24_24024


namespace divisor_greater_than_2_l24_24289

theorem divisor_greater_than_2 (w n d : ℕ) (h1 : ∃ q1 : ℕ, w = d * q1 + 2)
                                       (h2 : n % 8 = 5)
                                       (h3 : n < 180) : 2 < d :=
sorry

end divisor_greater_than_2_l24_24289


namespace inverse_var_q_value_l24_24251

theorem inverse_var_q_value (p q : ℝ) (h1 : ∀ p q, (p * q = 400))
(p_init : p = 800) (q_init : q = 0.5) (new_p : p = 400) :
  q = 1 := by
  sorry

end inverse_var_q_value_l24_24251


namespace even_function_a_eq_neg_one_l24_24858

-- Definitions for the function f and the condition for it being an even function
def f (x a : ℝ) := (x - 1) * (x - a)

-- The theorem stating that if f is an even function, then a = -1
theorem even_function_a_eq_neg_one (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 :=
by
  sorry

end even_function_a_eq_neg_one_l24_24858


namespace find_a_l24_24838

theorem find_a (A B : Set ℝ) (a : ℝ)
  (hA : A = {1, 2})
  (hB : B = {a, a^2 + 1})
  (hUnion : A ∪ B = {0, 1, 2}) :
  a = 0 :=
sorry

end find_a_l24_24838


namespace OddPrimeDivisorCondition_l24_24328

theorem OddPrimeDivisorCondition (n : ℕ) (h_pos : 0 < n) (h_div : ∀ d : ℕ, d ∣ n → d + 1 ∣ n + 1) : 
  ∃ p : ℕ, Prime p ∧ n = p ∧ ¬ Even p :=
sorry

end OddPrimeDivisorCondition_l24_24328


namespace problem_statement_l24_24344

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem problem_statement : f (f (1/2)) = 1 :=
by
    sorry

end problem_statement_l24_24344


namespace find_b_for_continuity_at_2_l24_24731

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if h : x ≤ 2 then 4 * x^2 + 5 else b * x + 3

theorem find_b_for_continuity_at_2 (b : ℝ) : (∀ x, f x b = if x ≤ 2 then 4 * x^2 + 5 else b * x + 3) ∧ 
  (f 2 b = 21) ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x b - f 2 b| < ε) → 
  b = 9 :=
by
  sorry

end find_b_for_continuity_at_2_l24_24731


namespace train_speed_kph_l24_24466

-- Definitions based on conditions
def time_seconds : ℕ := 9
def length_meters : ℕ := 135
def conversion_factor : ℕ := 36 -- 3.6 represented as an integer by multiplying both sides by 10

-- The proof statement
theorem train_speed_kph : (length_meters * conversion_factor / 10 / time_seconds = 54) :=
by
  sorry

end train_speed_kph_l24_24466


namespace intersection_point_l24_24130

variable (x y : ℝ)

-- Definitions given by the conditions
def line1 (x y : ℝ) := 3 * y = -2 * x + 6
def line2 (x y : ℝ) := -2 * y = 6 * x + 4

-- The theorem we want to prove
theorem intersection_point : ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ x = -12/7 ∧ y = 22/7 := 
sorry

end intersection_point_l24_24130


namespace semicircle_triangle_l24_24420

variable (a b r : ℝ)

-- Conditions: 
-- (1) Semicircle of radius r inside a right-angled triangle
-- (2) Shorter edges of the triangle (tangents to the semicircle) have lengths a and b
-- (3) Diameter of the semicircle lies on the hypotenuse of the triangle

theorem semicircle_triangle (h1 : a > 0) (h2 : b > 0) (h3 : r > 0)
  (tangent_property : true) -- Assumed relevant tangent properties are true
  (angle_property : true) -- Assumed relevant angle properties are true
  (geom_configuration : true) -- Assumed specific geometric configuration is correct
  : 1 / r = 1 / a + 1 / b := 
  sorry

end semicircle_triangle_l24_24420


namespace coins_after_10_hours_l24_24763

def numberOfCoinsRemaining : Nat :=
  let hour1_coins := 20
  let hour2_coins := hour1_coins + 30
  let hour3_coins := hour2_coins + 30
  let hour4_coins := hour3_coins + 40
  let hour5_coins := hour4_coins - (hour4_coins * 20 / 100)
  let hour6_coins := hour5_coins + 50
  let hour7_coins := hour6_coins + 60
  let hour8_coins := hour7_coins - (hour7_coins / 5)
  let hour9_coins := hour8_coins + 70
  let hour10_coins := hour9_coins - (hour9_coins * 15 / 100)
  hour10_coins

theorem coins_after_10_hours : numberOfCoinsRemaining = 200 := by
  sorry

end coins_after_10_hours_l24_24763


namespace cos_squared_sum_gte_three_fourths_l24_24446

-- The following statement defines the mathematical problem
theorem cos_squared_sum_gte_three_fourths
  (α β γ : ℝ)
  (h : α + β + γ = 180) :
  (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 ≥ 3 / 4 := by
  sorry

end cos_squared_sum_gte_three_fourths_l24_24446


namespace infinite_series_fraction_l24_24179

theorem infinite_series_fraction:
  (∑' n : ℕ, (if n = 0 then 0 else ((2 : ℚ) / (3 * n) - (1 : ℚ) / (3 * (n + 1)) - (7 : ℚ) / (6 * (n + 3))))) =
  (1 : ℚ) / 3 := 
sorry

end infinite_series_fraction_l24_24179


namespace scientific_notation_of_sesame_mass_l24_24105

theorem scientific_notation_of_sesame_mass :
  0.00000201 = 2.01 * 10^(-6) :=
sorry

end scientific_notation_of_sesame_mass_l24_24105


namespace binomial_12_6_eq_924_l24_24804

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_l24_24804


namespace value_of_x_plus_y_l24_24509

theorem value_of_x_plus_y 
  (x y : ℝ)
  (h1 : |x| = 3)
  (h2 : |y| = 2)
  (h3 : x > y) :
  x + y = 5 ∨ x + y = 1 := 
  sorry

end value_of_x_plus_y_l24_24509


namespace cube_root_less_than_5_l24_24853

theorem cube_root_less_than_5 :
  {n : ℕ | n > 0 ∧ (∃ m : ℝ, m^3 = n ∧ m < 5)}.finite.card = 124 :=
by
  sorry

end cube_root_less_than_5_l24_24853


namespace binomial_12_6_eq_924_l24_24805

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_l24_24805


namespace paco_initial_sweet_cookies_l24_24250

theorem paco_initial_sweet_cookies
    (x : ℕ)  -- Paco's initial number of sweet cookies
    (eaten_sweet : ℕ)  -- number of sweet cookies Paco ate
    (left_sweet : ℕ)  -- number of sweet cookies Paco had left
    (h1 : eaten_sweet = 15)  -- Paco ate 15 sweet cookies
    (h2 : left_sweet = 19)  -- Paco had 19 sweet cookies left
    (h3 : x - eaten_sweet = left_sweet)  -- After eating, Paco had 19 sweet cookies left
    : x = 34 :=  -- Paco initially had 34 sweet cookies
sorry

end paco_initial_sweet_cookies_l24_24250


namespace vector_equation_proof_l24_24072

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C P : V)

/-- The given condition. -/
def given_condition : Prop :=
  (P - A) + 2 • (P - B) + 3 • (P - C) = 0

/-- The target equality we want to prove. -/
theorem vector_equation_proof (h : given_condition A B C P) :
  P - A = (1 / 3 : ℝ) • (B - A) + (1 / 2 : ℝ) • (C - A) :=
sorry

end vector_equation_proof_l24_24072


namespace distance_to_x_axis_l24_24227

def point_P : ℝ × ℝ := (2, -3)

theorem distance_to_x_axis : abs (point_P.snd) = 3 := by
  sorry

end distance_to_x_axis_l24_24227


namespace calculate_amount_l24_24934

-- Definitions
def principal_amount : ℝ := 51200
def rate_of_increase : ℝ := 1 / 8
def years : ℕ := 2

-- Theorem statement
theorem calculate_amount (principal_amount : ℝ) (rate_of_increase : ℝ) (years : ℕ) :
  principal_amount = 51200 → rate_of_increase = 1 / 8 → years = 2 →
  principal_amount * (1 + rate_of_increase) ^ years = 64800 :=
by
  intros h₁ h₂ h₃
  sorry

end calculate_amount_l24_24934


namespace intersection_of_A_and_B_l24_24846

-- Define set A
def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

-- Define set B
def B : Set ℤ := {2, 4, 6, 8}

-- Prove that the intersection of set A and set B is {2, 4}.
theorem intersection_of_A_and_B : A ∩ B = {2, 4} :=
by
  sorry

end intersection_of_A_and_B_l24_24846


namespace vertex_in_first_quadrant_l24_24607

theorem vertex_in_first_quadrant (a : ℝ) (h : a > 1) : 
  let x_vertex := (a + 1) / 2
  let y_vertex := (a + 3)^2 / 4
  x_vertex > 0 ∧ y_vertex > 0 := 
by
  sorry

end vertex_in_first_quadrant_l24_24607


namespace compute_binom_12_6_eq_1848_l24_24818

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end compute_binom_12_6_eq_1848_l24_24818


namespace pascal_triangle_ratios_l24_24558
open Nat

theorem pascal_triangle_ratios :
  ∃ n r : ℕ, 
  (choose n r) * 4 = (choose n (r + 1)) * 3 ∧ 
  (choose n (r + 1)) * 3 = (choose n (r + 2)) * 4 ∧ 
  n = 34 :=
by
  sorry

end pascal_triangle_ratios_l24_24558


namespace problem_solution_l24_24931

theorem problem_solution : (3.242 * 14) / 100 = 0.45388 := by
  sorry

end problem_solution_l24_24931


namespace tan_x0_eq_neg_sqrt3_l24_24990

noncomputable def f (x : ℝ) : ℝ := (1/2 : ℝ) * x - (1/4 : ℝ) * Real.sin x - (Real.sqrt 3 / 4 : ℝ) * Real.cos x

theorem tan_x0_eq_neg_sqrt3 (x₀ : ℝ) (h : HasDerivAt f (1 : ℝ) x₀) : Real.tan x₀ = -Real.sqrt 3 := by
  sorry

end tan_x0_eq_neg_sqrt3_l24_24990


namespace percentage_of_female_officers_on_duty_l24_24578

theorem percentage_of_female_officers_on_duty
    (on_duty : ℕ) (half_on_duty_female : on_duty / 2 = 100)
    (total_female_officers : ℕ)
    (total_female_officers_value : total_female_officers = 1000)
    : (100 / total_female_officers : ℝ) * 100 = 10 :=
by sorry

end percentage_of_female_officers_on_duty_l24_24578


namespace total_water_in_heaters_l24_24920

theorem total_water_in_heaters (wallace_capacity : ℕ) (catherine_capacity : ℕ) 
(wallace_water : ℕ) (catherine_water : ℕ) :
  wallace_capacity = 40 →
  (wallace_water = (3 * wallace_capacity) / 4) →
  wallace_capacity = 2 * catherine_capacity →
  (catherine_water = (3 * catherine_capacity) / 4) →
  wallace_water + catherine_water = 45 :=
by
  sorry

end total_water_in_heaters_l24_24920


namespace quotient_is_10_l24_24737

theorem quotient_is_10 (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 161)
  (h2 : divisor = 16)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) : 
  quotient = 10 := 
by
  sorry

end quotient_is_10_l24_24737


namespace hours_of_rain_l24_24340

def totalHours : ℕ := 9
def noRainHours : ℕ := 5
def rainHours : ℕ := totalHours - noRainHours

theorem hours_of_rain : rainHours = 4 := by
  sorry

end hours_of_rain_l24_24340


namespace mode_I_swaps_mode_II_swaps_l24_24718

-- Define the original and target strings
def original_sign := "MEGYEI TAKARÉKPÉNZTÁR R. T."
def target_sign := "TATÁR GYERMEK A PÉNZT KÉRI."

-- Define a function for adjacent swaps needed to convert original_sign to target_sign
def adjacent_swaps (orig : String) (target : String) : ℕ := sorry

-- Define a function for any distant swaps needed to convert original_sign to target_sign
def distant_swaps (orig : String) (target : String) : ℕ := sorry

-- The theorems we want to prove
theorem mode_I_swaps : adjacent_swaps original_sign target_sign = 85 := sorry

theorem mode_II_swaps : distant_swaps original_sign target_sign = 11 := sorry

end mode_I_swaps_mode_II_swaps_l24_24718


namespace square_of_1008_l24_24481

theorem square_of_1008 : 1008^2 = 1016064 := 
by sorry

end square_of_1008_l24_24481


namespace even_n_ineq_l24_24333

theorem even_n_ineq (n : ℕ) (h : ∀ x : ℝ, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2) : Even n :=
  sorry

end even_n_ineq_l24_24333


namespace intersecting_line_l24_24356

theorem intersecting_line {x y : ℝ} (h1 : x^2 + y^2 = 10) (h2 : (x - 1)^2 + (y - 3)^2 = 10) :
  x + 3 * y - 5 = 0 :=
sorry

end intersecting_line_l24_24356


namespace james_bags_l24_24234

theorem james_bags (total_marbles : ℕ) (remaining_marbles : ℕ) (b : ℕ) (m : ℕ) 
  (h1 : total_marbles = 28) 
  (h2 : remaining_marbles = 21) 
  (h3 : m = total_marbles - remaining_marbles) 
  (h4 : b = total_marbles / m) : 
  b = 4 :=
by
  sorry

end james_bags_l24_24234


namespace percentage_decrease_in_area_l24_24703

noncomputable def original_radius (r : ℝ) : ℝ := r
noncomputable def new_radius (r : ℝ) : ℝ := 0.5 * r
noncomputable def original_area (r : ℝ) : ℝ := Real.pi * r ^ 2
noncomputable def new_area (r : ℝ) : ℝ := Real.pi * (0.5 * r) ^ 2

theorem percentage_decrease_in_area (r : ℝ) (hr : 0 ≤ r) :
  ((original_area r - new_area r) / original_area r) * 100 = 75 :=
by
  sorry

end percentage_decrease_in_area_l24_24703


namespace find_possible_values_a_l24_24434

theorem find_possible_values_a :
  ∃ a : ℤ, ∃ b : ℤ, ∃ c : ℤ, 
  (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) ∧
  ((b + 5) * (c + 5) = 1 ∨ (b + 5) * (c + 5) = 4) ↔ 
  a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 7 :=
by
  sorry

end find_possible_values_a_l24_24434


namespace tory_sells_grandmother_l24_24490

theorem tory_sells_grandmother (G : ℕ)
    (total_goal : ℕ) (sold_to_uncle : ℕ) (sold_to_neighbor : ℕ) (remaining_to_sell : ℕ)
    (h_goal : total_goal = 50) (h_sold_to_uncle : sold_to_uncle = 7)
    (h_sold_to_neighbor : sold_to_neighbor = 5) (h_remaining_to_sell : remaining_to_sell = 26) :
    (G + sold_to_uncle + sold_to_neighbor + remaining_to_sell = total_goal) → G = 12 :=
by
    intros h
    -- Proof goes here
    sorry

end tory_sells_grandmother_l24_24490


namespace cos_angle_between_diagonals_l24_24161

/-- Definitions for the vectors that define the parallelogram -/
def a : ℝ × ℝ × ℝ := ⟨3, 2, 1⟩
def b : ℝ × ℝ × ℝ := ⟨1, 3, 2⟩

/-- Definitions for the diagonal vectors -/
def diagonal1 : ℝ × ℝ × ℝ := ⟨a.1 + b.1, a.2 + b.2, a.3 + b.3⟩
def diagonal2 : ℝ × ℝ × ℝ := ⟨b.1 - a.1, b.2 - a.2, b.3 - a.3⟩

/-- Cosine of the angle between the diagonals of the parallelogram -/
noncomputable def cos_theta : ℝ :=
(∑ i in (finset.range 3), (diagonal1.η i) * (diagonal2.η i)) /
(real.sqrt (∑ i in (finset.range 3), (diagonal1.η i)^2) *
 real.sqrt (∑ i in (finset.range 3), (diagonal2.η i)^2))

/-- Theorem stating the problem conclusion -/
theorem cos_angle_between_diagonals : cos_theta = 0 :=
by sorry

end cos_angle_between_diagonals_l24_24161


namespace evenness_oddness_of_f_min_value_of_f_l24_24240

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + |x - a| + 1

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem evenness_oddness_of_f (a : ℝ) :
  (is_even (f a) ↔ a = 0) ∧ (a ≠ 0 → ¬ is_even (f a) ∧ ¬ is_odd (f a)) :=
by
  sorry

theorem min_value_of_f (a x : ℝ) (h : x ≥ a) :
  (a ≤ -1 / 2 → f a x = 3 / 4 - a) ∧ (a > -1 / 2 → f a x = a^2 + 1) :=
by
  sorry

end evenness_oddness_of_f_min_value_of_f_l24_24240


namespace remainder_when_dividing_l24_24443

theorem remainder_when_dividing (a : ℕ) (h1 : a = 432 * 44) : a % 38 = 8 :=
by
  -- Proof goes here
  sorry

end remainder_when_dividing_l24_24443


namespace number_of_rocks_tossed_l24_24915

-- Conditions
def pebbles : ℕ := 6
def rocks : ℕ := 3
def boulders : ℕ := 2
def pebble_splash : ℚ := 1 / 4
def rock_splash : ℚ := 1 / 2
def boulder_splash : ℚ := 2

-- Total width of the splashes
def total_splash (R : ℕ) : ℚ := 
  pebbles * pebble_splash + R * rock_splash + boulders * boulder_splash

-- Given condition
def total_splash_condition : ℚ := 7

theorem number_of_rocks_tossed : 
  total_splash rocks = total_splash_condition → rocks = 3 :=
by
  intro h
  sorry

end number_of_rocks_tossed_l24_24915


namespace carlson_handkerchief_usage_l24_24801

def problem_statement : Prop :=
  let handkerchief_area := 25 * 25 -- Area in cm²
  let total_fabric_area := 3 * 10000 -- Total fabric area in cm²
  let days := 8
  let total_handkerchiefs := total_fabric_area / handkerchief_area
  let handkerchiefs_per_day := total_handkerchiefs / days
  handkerchiefs_per_day = 6

theorem carlson_handkerchief_usage : problem_statement := by
  sorry

end carlson_handkerchief_usage_l24_24801


namespace planar_graph_edge_bound_l24_24887

structure Graph :=
  (V E : ℕ) -- vertices and edges

def planar_connected (G : Graph) : Prop := 
  sorry -- Planarity and connectivity conditions are complex to formalize

def num_faces (G : Graph) : ℕ :=
  sorry -- Number of faces based on V, E and planarity

theorem planar_graph_edge_bound (G : Graph) (h_planar : planar_connected G) 
  (euler : G.V - G.E + num_faces G = 2) 
  (face_bound : 2 * G.E ≥ 3 * num_faces G) : 
  G.E ≤ 3 * G.V - 6 :=
sorry

end planar_graph_edge_bound_l24_24887


namespace min_initial_seeds_l24_24302

/-- Given conditions:
  - The farmer needs to sell at least 10,000 watermelons each year.
  - Each watermelon produces 250 seeds when used for seeds but cannot be sold if used for seeds.
  - We need to find the minimum number of initial seeds S the farmer must buy to never buy seeds again.
-/
theorem min_initial_seeds : ∃ (S : ℕ), S = 10041 ∧ ∀ (yearly_sales : ℕ), yearly_sales = 10000 →
  ∀ (seed_yield : ℕ), seed_yield = 250 →
  ∃ (x : ℕ), S = yearly_sales + x ∧ x * seed_yield ≥ S :=
sorry

end min_initial_seeds_l24_24302


namespace remainder_is_4_l24_24422

-- Definitions based on the given conditions
def dividend := 132
def divisor := 16
def quotient := 8

-- The theorem we aim to prove, stating the remainder
theorem remainder_is_4 : dividend = divisor * quotient + 4 := sorry

end remainder_is_4_l24_24422


namespace xiao_liang_reaches_museum_l24_24929

noncomputable def xiao_liang_distance_to_museum : ℝ :=
  let science_museum := (200 * Real.sqrt 2, 200 * Real.sqrt 2)
  let initial_mistake := (-300 * Real.sqrt 2, 300 * Real.sqrt 2)
  let to_supermarket := (-100 * Real.sqrt 2, 500 * Real.sqrt 2)
  Real.sqrt ((science_museum.1 - to_supermarket.1)^2 + (science_museum.2 - to_supermarket.2)^2)

theorem xiao_liang_reaches_museum :
  xiao_liang_distance_to_museum = 600 :=
sorry

end xiao_liang_reaches_museum_l24_24929


namespace bella_steps_l24_24798

/-- Bella begins to walk from her house toward her friend Ella's house. At the same time, Ella starts to skate toward Bella's house. They each maintain a constant speed, and Ella skates three times as fast as Bella walks. The distance between their houses is 10560 feet, and Bella covers 3 feet with each step. Prove that Bella will take 880 steps by the time she meets Ella. -/
theorem bella_steps 
  (d : ℝ)    -- distance between their houses in feet
  (s_bella : ℝ)    -- speed of Bella in feet per minute
  (s_ella : ℝ)    -- speed of Ella in feet per minute
  (steps_per_ft : ℝ)    -- feet per step of Bella
  (h1 : d = 10560)    -- distance between their houses is 10560 feet
  (h2 : s_ella = 3 * s_bella)    -- Ella skates three times as fast as Bella
  (h3 : steps_per_ft = 3)    -- Bella covers 3 feet with each step
  : (10560 / (4 * s_bella)) * s_bella / 3 = 880 :=
by
  -- proof here 
  sorry

end bella_steps_l24_24798


namespace inequality_solution_l24_24861

theorem inequality_solution (a : ℝ) (h : ∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) : a < -1 :=
sorry

end inequality_solution_l24_24861


namespace probability_three_at_marked_l24_24453

def sudoku_matrix := Matrix (Fin 9) (Fin 9) (Fin 9.succ)

def valid_sudoku (m : sudoku_matrix) : Prop :=
  ∀ i, (∀ j, (m i j).val ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (∀ i₁ i₂ j, i₁ ≠ i₂ → m i₁ j ≠ m i₂ j) ∧
  (∀ i j₁ j₂, j₁ ≠ j₂ → m i j₁ ≠ m i j₂) ∧
  (∀ k l n₁ n₂, (n₁ ≠ n₂) →
    m ⟨3 * (k / 3) + (n₁ / 3), _⟩ ⟨3 * (l / 3) + (n₁ % 3), _⟩ ≠
    m ⟨3 * (k / 3) + (n₂ / 3), _⟩ ⟨3 * (l / 3) + (n₂ % 3), _⟩)

noncomputable def probability_digit_at (m : sudoku_matrix) (d : Fin 9.succ) (i j : Fin 9) : ℚ :=
  if valid_sudoku m then (1 : ℚ) / 9 else 0

theorem probability_three_at_marked (m : sudoku_matrix) (marked : Fin 9 × Fin 9) :
  valid_sudoku m →
  probability_digit_at m (Fin.succ ⟨2, by norm_num⟩) marked.1 marked.2 = (1 : ℚ) / 9 :=
sorry

end probability_three_at_marked_l24_24453


namespace sum_of_roots_l24_24091

theorem sum_of_roots (a b c d : ℝ) (h : ∀ x : ℝ, 
  a * (x ^ 3 - x) ^ 3 + b * (x ^ 3 - x) ^ 2 + c * (x ^ 3 - x) + d 
  ≥ a * (x ^ 2 + x + 1) ^ 3 + b * (x ^ 2 + x + 1) ^ 2 + c * (x ^ 2 + x + 1) + d) :
  b / a = -6 :=
sorry

end sum_of_roots_l24_24091


namespace simplify_expression_l24_24896

-- Define the given expression
def given_expr (x y : ℝ) := 3 * x + 4 * y + 5 * x^2 + 2 - (8 - 5 * x - 3 * y - 2 * x^2)

-- Define the expected simplified expression
def simplified_expr (x y : ℝ) := 7 * x^2 + 8 * x + 7 * y - 6

-- Theorem statement to prove the equivalence of the expressions
theorem simplify_expression (x y : ℝ) : 
  given_expr x y = simplified_expr x y := sorry

end simplify_expression_l24_24896


namespace rectangle_width_l24_24425

theorem rectangle_width (w : ℝ) (h_length : w * 2 = l) (h_area : w * l = 50) : w = 5 :=
by
  sorry

end rectangle_width_l24_24425


namespace impossible_to_half_boys_sit_with_girls_l24_24612

theorem impossible_to_half_boys_sit_with_girls:
  ∀ (g b : ℕ), 
  (g + b = 30) → 
  (∃ k, g = 2 * k) →
  (∀ (d : ℕ), 2 * d = g) →
  ¬ ∃ m, (b = 2 * m) ∧ (∀ (d : ℕ), 2 * d = b) :=
by
  sorry

end impossible_to_half_boys_sit_with_girls_l24_24612


namespace inverse_proportion_point_passes_through_l24_24088

theorem inverse_proportion_point_passes_through
  (m : ℝ) (h1 : (4, 6) ∈ {p : ℝ × ℝ | p.snd = (m^2 + 2 * m - 1) / p.fst})
  : (-4, -6) ∈ {p : ℝ × ℝ | p.snd = (m^2 + 2 * m - 1) / p.fst} :=
sorry

end inverse_proportion_point_passes_through_l24_24088


namespace min_value_reciprocals_l24_24341

variable {a b : ℝ}

theorem min_value_reciprocals (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) :
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ∀ x y, x > 0 → y > 0 → x + y = 2 → 
  (1/a + 1/b) ≥ 2) :=
sorry

end min_value_reciprocals_l24_24341


namespace find_line_equation_l24_24423

-- Define the conditions for the x-intercept and inclination angle
def x_intercept (x : ℝ) (line : ℝ → ℝ) : Prop :=
  line x = 0

def inclination_angle (θ : ℝ) (k : ℝ) : Prop :=
  k = Real.tan θ

-- Define the properties of the line we're working with
def line (x : ℝ) : ℝ := -x + 5

theorem find_line_equation :
  x_intercept 5 line ∧ inclination_angle (3 * Real.pi / 4) (-1) → (∀ x, line x = -x + 5) :=
by
  intro h
  sorry

end find_line_equation_l24_24423


namespace tangent_line_and_curve_l24_24212

theorem tangent_line_and_curve (a x0 : ℝ) 
  (h1 : ∀ (x : ℝ), x0 + a = 1) 
  (h2 : ∀ (y : ℝ), y = x0 + 1) 
  (h3 : ∀ (y : ℝ), y = Real.log (x0 + a)) 
  : a = 2 := 
by 
  sorry

end tangent_line_and_curve_l24_24212


namespace pow_mod_eleven_l24_24630

theorem pow_mod_eleven : 
  ∀ (n : ℕ), (n ≡ 5 ^ 1 [MOD 11] → n ≡ 5 [MOD 11]) ∧ 
             (n ≡ 5 ^ 2 [MOD 11] → n ≡ 3 [MOD 11]) ∧ 
             (n ≡ 5 ^ 3 [MOD 11] → n ≡ 4 [MOD 11]) ∧ 
             (n ≡ 5 ^ 4 [MOD 11] → n ≡ 9 [MOD 11]) ∧ 
             (n ≡ 5 ^ 5 [MOD 11] → n ≡ 1 [MOD 11]) →
  5 ^ 1233 ≡ 4 [MOD 11] :=
by
  intro n h
  sorry

end pow_mod_eleven_l24_24630


namespace max_value_of_x_squared_plus_xy_plus_y_squared_l24_24108

theorem max_value_of_x_squared_plus_xy_plus_y_squared
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 - x * y + y^2 = 9) : 
  (x^2 + x * y + y^2) ≤ 27 :=
sorry

end max_value_of_x_squared_plus_xy_plus_y_squared_l24_24108


namespace number_of_projects_min_total_time_l24_24447

noncomputable def energy_transfer_projects (x y : ℕ) : Prop :=
  x + y = 15 ∧ x = 2 * y - 3

theorem number_of_projects (x y : ℕ) (h : energy_transfer_projects x y) :
  x = 9 ∧ y = 6 :=
sorry

noncomputable def minimize_time (m : ℕ) : Prop :=
  m + 10 - m = 10 ∧ 10 - m > m / 2 ∧ -2 * m + 80 = 68

theorem min_total_time (m : ℕ) (h : minimize_time m) :
  m = 6 ∧ 10 - m = 4 :=
sorry

end number_of_projects_min_total_time_l24_24447


namespace quadratic_equation_roots_l24_24857

-- Define the two numbers α and β such that their arithmetic and geometric means are given.
variables (α β : ℝ)

-- Arithmetic mean condition
def arithmetic_mean_condition : Prop := (α + β = 16)

-- Geometric mean condition
def geometric_mean_condition : Prop := (α * β = 225)

-- The quadratic equation with roots α and β
def quadratic_equation (x : ℝ) : ℝ := x^2 - 16 * x + 225

-- The proof statement
theorem quadratic_equation_roots (α β : ℝ) (h1 : arithmetic_mean_condition α β) (h2 : geometric_mean_condition α β) :
  ∃ x : ℝ, quadratic_equation x = 0 :=
sorry

end quadratic_equation_roots_l24_24857


namespace greatest_three_digit_multiple_of_23_l24_24624

theorem greatest_three_digit_multiple_of_23 : ∃ n : ℕ, n % 23 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ ∀ m : ℕ, m % 23 = 0 ∧ 100 ≤ m ∧ m ≤ 999 → m ≤ n := 
by
  use 989
  split
  · -- 989 is a multiple of 23
    exact (by norm_num : 989 % 23 = 0)
  · split
    · -- 989 is at least 100
      exact (by norm_num : 100 ≤ 989)
    · split
      · -- 989 is at most 999
        exact (by norm_num : 989 ≤ 999)
      · -- 989 is the greatest such number within the range
        sorry

end greatest_three_digit_multiple_of_23_l24_24624


namespace work_completion_time_l24_24780

noncomputable def work_done_by_woman_per_day : ℝ := 1 / 50
noncomputable def work_done_by_child_per_day : ℝ := 1 / 100
noncomputable def total_work_done_by_5_women_per_day : ℝ := 5 * work_done_by_woman_per_day
noncomputable def total_work_done_by_10_children_per_day : ℝ := 10 * work_done_by_child_per_day
noncomputable def combined_work_per_day : ℝ := total_work_done_by_5_women_per_day + total_work_done_by_10_children_per_day

theorem work_completion_time (h1 : 10 / 5 = 2) (h2 : 10 / 10 = 1) :
  1 / combined_work_per_day = 5 :=
by
  sorry

end work_completion_time_l24_24780


namespace find_c_l24_24210

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem find_c (a b m c : ℝ) (h1 : ∀ x, f x a b ≥ 0)
  (h2 : ∀ x, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
by
  sorry

end find_c_l24_24210


namespace total_distance_flown_l24_24046

/-- 
An eagle can fly 15 miles per hour; 
a falcon can fly 46 miles per hour; 
a pelican can fly 33 miles per hour; 
and a hummingbird can fly 30 miles per hour. 
All the birds flew for 2 hours straight.
Prove that the total distance flown by all the birds is 248 miles.
-/
theorem total_distance_flown :
  let eagle_speed := 15
      falcon_speed := 46
      pelican_speed := 33
      hummingbird_speed := 30
      hours_flown := 2
      eagle_distance := eagle_speed * hours_flown 
      falcon_distance := falcon_speed * hours_flown 
      pelican_distance := pelican_speed * hours_flown 
      hummingbird_distance := hummingbird_speed * hours_flown 
  in eagle_distance + falcon_distance + pelican_distance + hummingbird_distance = 248 := 
sorry

end total_distance_flown_l24_24046


namespace Vanya_original_number_l24_24284

theorem Vanya_original_number (m n : ℕ) (hm : m ≤ 9) (hn : n ≤ 9) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 10 * m + n = 81 := by
  sorry

end Vanya_original_number_l24_24284


namespace exterior_angle_DEG_l24_24415

-- Define the degree measures of angles in a square and a pentagon.
def square_interior_angle := 90
def pentagon_interior_angle := 108

-- Define the sum of the adjacent interior angles at D
def adjacent_interior_sum := square_interior_angle + pentagon_interior_angle

-- Statement to prove the exterior angle DEG
theorem exterior_angle_DEG :
  360 - adjacent_interior_sum = 162 := by
  sorry

end exterior_angle_DEG_l24_24415


namespace mod_11_residue_l24_24799

theorem mod_11_residue : 
  ((312 - 3 * 52 + 9 * 165 + 6 * 22) % 11) = 2 :=
by
  sorry

end mod_11_residue_l24_24799


namespace max_pairs_correct_l24_24876

def max_pairs (n : ℕ) : ℕ :=
  if h : n > 1 then (n * n) / 4 else 0

theorem max_pairs_correct (n : ℕ) (h : n ≥ 2) :
  (max_pairs n = (n * n) / 4) :=
by sorry

end max_pairs_correct_l24_24876


namespace probability_composite_product_l24_24532

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l24_24532


namespace evaluate_expression_l24_24338

variables (x : ℝ)

theorem evaluate_expression :
  x * (x * (x * (3 - x) - 5) + 13) + 1 = -x^4 + 3*x^3 - 5*x^2 + 13*x + 1 :=
by 
  sorry

end evaluate_expression_l24_24338


namespace probability_composite_l24_24551

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def non_composite_outcomes : ℕ := 1 + 12

theorem probability_composite :
  let total_outcomes := 6^4 in
  let non_composite := non_composite_outcomes in
  let composite_probability := (total_outcomes - non_composite) / total_outcomes.toRat in
  composite_probability = 1283 / 1296 := by
  let total_outcomes := 6^4
  let non_composite := 1 + 12
  let composite_probability := (total_outcomes - non_composite).toRat / total_outcomes.toRat
  sorry

end probability_composite_l24_24551


namespace min_sum_of_factors_240_l24_24025

theorem min_sum_of_factors_240 :
  ∃ a b : ℕ, a * b = 240 ∧ (∀ a' b' : ℕ, a' * b' = 240 → a + b ≤ a' + b') ∧ a + b = 31 :=
sorry

end min_sum_of_factors_240_l24_24025


namespace range_of_function_l24_24728

theorem range_of_function :
  ∀ x : ℝ,
  (0 < x ∧ x < (π / 2)) →
  ∃ y : ℝ, 
  y = (sin x - 2 * cos x + (32 / (125 * sin x * (1 - cos x)))) ∧ y ≥ 2 / 5 :=
sorry

end range_of_function_l24_24728


namespace exists_bound_for_expression_l24_24865

theorem exists_bound_for_expression :
  ∃ (C : ℝ), (∀ (k : ℤ), abs ((k^8 - 2*k + 1 : ℤ) / (k^4 - 3 : ℤ)) < C) := 
sorry

end exists_bound_for_expression_l24_24865


namespace smallest_r_for_B_in_C_l24_24571

def A : Set ℝ := {t | 0 < t ∧ t < 2 * Real.pi}

def B : Set (ℝ × ℝ) := 
  {p | ∃ t ∈ A, p.1 = Real.sin t ∧ p.2 = 2 * Real.sin t * Real.cos t}

def C (r : ℝ) : Set (ℝ × ℝ) := 
  {p | p.1^2 + p.2^2 ≤ r^2 ∧ r > 0}

theorem smallest_r_for_B_in_C : ∃ r, (B ⊆ C r ∧ ∀ r', r' < r → ¬ (B ⊆ C r')) :=
  sorry

end smallest_r_for_B_in_C_l24_24571


namespace trains_crossing_time_l24_24149

theorem trains_crossing_time (length : ℕ) (time1 time2 : ℕ) (h1 : length = 120) (h2 : time1 = 10) (h3 : time2 = 20) :
  (2 * length : ℚ) / (length / time1 + length / time2 : ℚ) = 13.33 :=
by
  sorry

end trains_crossing_time_l24_24149


namespace max_sum_of_distances_l24_24837

theorem max_sum_of_distances (x1 x2 y1 y2 : ℝ)
  (h1 : x1^2 + y1^2 = 1)
  (h2 : x2^2 + y2^2 = 1)
  (h3 : x1 * x2 + y1 * y2 = 1 / 2) :
  (|x1 + y1 - 1| / Real.sqrt 2 + |x2 + y2 - 1| / Real.sqrt 2) ≤ Real.sqrt 2 + Real.sqrt 3 :=
sorry

end max_sum_of_distances_l24_24837


namespace company_fund_initial_amount_l24_24116

theorem company_fund_initial_amount
  (n : ℕ) -- number of employees
  (initial_bonus_per_employee : ℕ := 60)
  (shortfall : ℕ := 10)
  (revised_bonus_per_employee : ℕ := 50)
  (fund_remaining : ℕ := 150)
  (initial_fund : ℕ := initial_bonus_per_employee * n - shortfall) -- condition that the fund was $10 short when planning the initial bonus
  (revised_fund : ℕ := revised_bonus_per_employee * n + fund_remaining) -- condition after distributing the $50 bonuses

  (eqn : initial_fund = revised_fund) -- equating initial and revised budget calculations
  
  : initial_fund = 950 := 
sorry

end company_fund_initial_amount_l24_24116


namespace missing_angle_correct_l24_24099

theorem missing_angle_correct (n : ℕ) (h1 : n ≥ 3) (angles_sum : ℕ) (h2 : angles_sum = 2017) 
    (sum_interior_angles : ℕ) (h3 : sum_interior_angles = 180 * (n - 2)) :
    (sum_interior_angles - angles_sum) = 143 :=
by
  sorry

end missing_angle_correct_l24_24099


namespace initial_average_mark_of_class_l24_24588

theorem initial_average_mark_of_class
  (avg_excluded : ℝ) (n_excluded : ℕ) (avg_remaining : ℝ)
  (n_total : ℕ) : 
  avg_excluded = 70 → 
  n_excluded = 5 → 
  avg_remaining = 90 → 
  n_total = 10 → 
  (10 * (10 / n_total + avg_excluded - avg_remaining) / 10) = 80 :=
by 
  intros 
  sorry

end initial_average_mark_of_class_l24_24588


namespace tangent_line_parallel_l24_24351

open Function

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x ^ 2 + 2 * b * x

theorem tangent_line_parallel {b : ℝ} (h : 2 * b = 1) :
  let f (x : ℝ) := x ^ 2 + x,
  S (n : ℕ) := ∑ i in Finset.range (n + 1), 1 / f (i : ℝ) in 
  S 2016 = 2016 / 2017 :=
by
  let f := λ x : ℝ, x ^ 2 + x
  sorry

end tangent_line_parallel_l24_24351


namespace cubics_sum_l24_24416

theorem cubics_sum (a b c : ℝ) (h₁ : a + b + c = 4) (h₂ : ab + ac + bc = 6) (h₃ : abc = -8) :
  a^3 + b^3 + c^3 = 8 :=
by {
  -- proof steps would go here
  sorry
}

end cubics_sum_l24_24416


namespace deduction_from_third_l24_24418

-- Define the conditions
def avg_10_consecutive_eq_20 (x : ℝ) : Prop :=
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 20

def new_avg_10_numbers_eq_15_5 (x y : ℝ) : Prop :=
  ((x - 9) + (x - 7) + (x + 2 - y) + (x - 3) + (x - 1) + (x + 1) + (x + 3) + (x + 5) + (x + 7) + (x + 9)) / 10 = 15.5

-- Define the theorem to be proved
theorem deduction_from_third (x y : ℝ) (h1 : avg_10_consecutive_eq_20 x) (h2 : new_avg_10_numbers_eq_15_5 x y) : y = 6 :=
sorry

end deduction_from_third_l24_24418


namespace range_of_function_l24_24729

open Real

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∃ y, y = sin x - 2 * cos x + 32 / (125 * sin x * (1 - cos x)) ∧ y ≥ 2 / 5 :=
sorry

end range_of_function_l24_24729


namespace remainder_when_subtract_div_by_6_l24_24294

theorem remainder_when_subtract_div_by_6 (m n : ℕ) (h1 : m % 6 = 2) (h2 : n % 6 = 3) (h3 : m > n) : (m - n) % 6 = 5 := 
by
  sorry

end remainder_when_subtract_div_by_6_l24_24294


namespace students_taking_both_courses_l24_24370

theorem students_taking_both_courses (n_total n_F n_G n_neither number_both : ℕ)
  (h_total : n_total = 79)
  (h_F : n_F = 41)
  (h_G : n_G = 22)
  (h_neither : n_neither = 25)
  (h_any_language : n_total - n_neither = 54)
  (h_sum_languages : n_F + n_G = 63)
  (h_both : n_F + n_G - (n_total - n_neither) = number_both) :
  number_both = 9 :=
by {
  sorry
}

end students_taking_both_courses_l24_24370


namespace num_real_x_l24_24518

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end num_real_x_l24_24518


namespace reflection_coefficient_l24_24040

theorem reflection_coefficient (I_0 : ℝ) (I_4 : ℝ) (k : ℝ) 
  (h1 : I_4 = I_0 * (1 - k)^4) 
  (h2 : I_4 = I_0 / 256) : 
  k = 0.75 :=
by 
  -- Proof omitted
  sorry

end reflection_coefficient_l24_24040


namespace avg_visitors_on_sundays_l24_24307

theorem avg_visitors_on_sundays (avg_other_days : ℕ) (avg_month : ℕ) (days_in_month sundays other_days : ℕ) (total_month_visitors : ℕ) (total_other_days_visitors : ℕ) (S : ℕ):
  avg_other_days = 240 →
  avg_month = 285 →
  days_in_month = 30 →
  sundays = 5 →
  other_days = 25 →
  total_month_visitors = avg_month * days_in_month →
  total_other_days_visitors = avg_other_days * other_days →
  5 * S + total_other_days_visitors = total_month_visitors →
  S = 510 :=
by
  intros _
          _
          _
          _
          _
          _
          _
          h
  -- Proof goes here
  sorry

end avg_visitors_on_sundays_l24_24307


namespace compute_binomial_sum_l24_24324

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem compute_binomial_sum :
  binomial 12 11 + binomial 12 1 = 24 :=
by
  sorry

end compute_binomial_sum_l24_24324


namespace product_of_three_numbers_l24_24432

theorem product_of_three_numbers (a b c : ℝ) 
  (h1 : a + b + c = 30) 
  (h2 : a = 5 * (b + c)) 
  (h3 : b = 9 * c) : 
  a * b * c = 56.25 := 
by 
  sorry

end product_of_three_numbers_l24_24432


namespace same_function_option_d_l24_24954

def f₁ (x : ℝ) := x
def g₁ (x : ℝ) := if x = 0 then 0 else x

def f₂ (x : ℝ) := 1
def g₂ (x : ℝ) := x^0

def f₃ (x : ℝ) := x
def g₃ (x : ℝ) := real.sqrt (x^2)

def f₄ (x : ℝ) := abs (x + 2)
def g₄ (x : ℝ) := if x ≥ -2 then x + 2 else -x - 2

theorem same_function_option_d : (∀ x, f₄ x = g₄ x) :=
by {
  -- The proof steps would go here, but they are omitted as per instructions.
  sorry
}

end same_function_option_d_l24_24954


namespace converse_opposite_l24_24637

theorem converse_opposite (x y : ℝ) : (x + y = 0) → (y = -x) :=
by
  sorry

end converse_opposite_l24_24637


namespace pow_neg_one_diff_l24_24664

theorem pow_neg_one_diff (n : ℤ) (h1 : n = 2010) (h2 : n + 1 = 2011) :
  (-1)^2010 - (-1)^2011 = 2 := 
by
  sorry

end pow_neg_one_diff_l24_24664


namespace scramble_words_count_l24_24748

-- Definitions based on the conditions
def alphabet_size : Nat := 25
def alphabet_size_no_B : Nat := 24

noncomputable def num_words_with_B : Nat :=
  let total_without_restriction := 25^1 + 25^2 + 25^3 + 25^4 + 25^5
  let total_without_B := 24^1 + 24^2 + 24^3 + 24^4 + 24^5
  total_without_restriction - total_without_B

-- Lean statement to prove the result
theorem scramble_words_count : num_words_with_B = 1692701 :=
by
  sorry

end scramble_words_count_l24_24748


namespace problem_inequality_l24_24208

open Real

theorem problem_inequality 
  (p q r x y theta: ℝ) :
  p * x ^ (q - y) + q * x ^ (r - y) + r * x ^ (y - theta)  ≥ p + q + r :=
sorry

end problem_inequality_l24_24208


namespace probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l24_24709

theorem probability_exactly_k_gnomes_fall (n k : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in p * q^(n - k) = p * (1 - p)^(n - k) := 
sorry

theorem expected_number_of_gnomes_fall (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in 
  (∑ j in finset.range n, (1 - q^(j+1))) = n + 1 - (1 / p) + ((1 - p)^(n+1) / p) :=
sorry

end probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l24_24709


namespace PQ_R_exist_l24_24062

theorem PQ_R_exist :
  ∃ P Q R : ℚ, 
    (P = -3/5) ∧ (Q = -1) ∧ (R = 13/5) ∧
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 → 
    (x^2 - 10)/((x - 1)*(x - 4)*(x - 6)) = P/(x - 1) + Q/(x - 4) + R/(x - 6)) :=
by
  sorry

end PQ_R_exist_l24_24062


namespace arithmetic_mean_of_fractions_l24_24662

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  (a + b) / 2 = 11 / 16 :=
by 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  show (a + b) / 2 = 11 / 16
  sorry

end arithmetic_mean_of_fractions_l24_24662


namespace percentage_increase_l24_24753

theorem percentage_increase (original_price new_price : ℝ) (h₁ : original_price = 300) (h₂ : new_price = 480) :
  ((new_price - original_price) / original_price) * 100 = 60 :=
by
  -- Proof goes here
  sorry

end percentage_increase_l24_24753


namespace erased_digit_is_4_l24_24787

def sum_of_digits (n : ℕ) : ℕ := 
  sorry -- definition of sum of digits

def D (N : ℕ) : ℕ := N - sum_of_digits N

theorem erased_digit_is_4 (N : ℕ) (x : ℕ) 
  (hD : D N % 9 = 0) 
  (h_sum : sum_of_digits (D N) - x = 131) 
  : x = 4 :=
by
  sorry

end erased_digit_is_4_l24_24787


namespace binom_12_6_eq_924_l24_24809

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binom_12_6_eq_924_l24_24809


namespace line_through_points_l24_24747

theorem line_through_points (x1 y1 x2 y2 : ℝ) (m b : ℝ) 
  (h1 : x1 = -3) (h2 : y1 = 1) (h3 : x2 = 1) (h4 : y2 = 3)
  (h5 : y1 = m * x1 + b) (h6 : y2 = m * x2 + b) :
  m + b = 3 := 
sorry

end line_through_points_l24_24747


namespace unit_cubes_fill_box_l24_24617

theorem unit_cubes_fill_box (p : ℕ) (hp : Nat.Prime p) :
  let length := p
  let width := 2 * p
  let height := 3 * p
  length * width * height = 6 * p^3 :=
by
  -- Proof here
  sorry

end unit_cubes_fill_box_l24_24617


namespace winner_C_l24_24560

noncomputable def votes_A : ℕ := 4500
noncomputable def votes_B : ℕ := 7000
noncomputable def votes_C : ℕ := 12000
noncomputable def votes_D : ℕ := 8500
noncomputable def votes_E : ℕ := 3500

noncomputable def total_votes : ℕ := votes_A + votes_B + votes_C + votes_D + votes_E

noncomputable def percentage (votes : ℕ) : ℚ :=
   (votes : ℚ) / (total_votes : ℚ) * 100

noncomputable def percentage_A : ℚ := percentage votes_A
noncomputable def percentage_B : ℚ := percentage votes_B
noncomputable def percentage_C : ℚ := percentage votes_C
noncomputable def percentage_D : ℚ := percentage votes_D
noncomputable def percentage_E : ℚ := percentage votes_E

theorem winner_C : (percentage_C = 33.803) := 
sorry

end winner_C_l24_24560


namespace find_correct_answer_l24_24326

theorem find_correct_answer (x : ℕ) (h : 3 * x = 135) : x / 3 = 15 :=
sorry

end find_correct_answer_l24_24326


namespace smallest_M_convex_quadrilateral_l24_24067

section ConvexQuadrilateral

-- Let a, b, c, d be the sides of a convex quadrilateral
variables {a b c d M : ℝ}

-- Condition to ensure that a, b, c, d are the sides of a convex quadrilateral
def is_convex_quadrilateral (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d < 360

-- The theorem statement
theorem smallest_M_convex_quadrilateral (hconvex : is_convex_quadrilateral a b c d) : ∃ M, (∀ a b c d, is_convex_quadrilateral a b c d → (a^2 + b^2) / (c^2 + d^2) > M) ∧ M = 1/2 :=
by sorry

end ConvexQuadrilateral

end smallest_M_convex_quadrilateral_l24_24067


namespace correct_measure_of_dispersion_l24_24137

theorem correct_measure_of_dispersion (mean variance median mode : Type) :
  ∃ d : Type, (d = variance) :=
by
  use variance
  sorry

end correct_measure_of_dispersion_l24_24137


namespace arithmetic_sequence_fifth_term_l24_24987

noncomputable def fifth_term_of_arithmetic_sequence (x y : ℝ)
  (h1 : 2 * x + y = 2 * x + y)
  (h2 : 2 * x - y = 2 * x + y - 2 * y)
  (h3 : 2 * x * y = 2 * x - 2 * y - 2 * y)
  (h4 : 2 * x / y = 2 * x * y - 5 * y^2 - 2 * y)
  : ℝ :=
(2 * x / y) - 2 * y

theorem arithmetic_sequence_fifth_term (x y : ℝ)
  (h1 : 2 * x + y = 2 * x + y)
  (h2 : 2 * x - y = 2 * x + y - 2 * y)
  (h3 : 2 * x * y = 2 * x - 2 * y - 2 * y)
  (h4 : 2 * x / y = 2 * x * y - 5 * y^2 - 2 * y)
  : fifth_term_of_arithmetic_sequence x y h1 h2 h3 h4 = -77 / 10 :=
sorry

end arithmetic_sequence_fifth_term_l24_24987


namespace dice_product_composite_probability_l24_24548

theorem dice_product_composite_probability :
  (let total_outcomes := (6 : ℕ)^4,
       non_composite_outcomes := 1,
       composite_probability := 1 - (non_composite_outcomes / total_outcomes) in
   composite_probability = 1295 / 1296) :=
by sorry

end dice_product_composite_probability_l24_24548


namespace num_real_satisfying_x_l24_24511

theorem num_real_satisfying_x : 
  (∃ (x : ℝ), ∃ (s : ℕ) (hs : s ∈ set.Ico 0 (floor (sqrt 123) + 1)), 
   s = int.of_nat (floor (sqrt (123 - sqrt x)))) →
   12 := sorry

end num_real_satisfying_x_l24_24511


namespace max_S_possible_l24_24347

theorem max_S_possible (nums : List ℝ) (h_nums_in_bound : ∀ n ∈ nums, 0 ≤ n ∧ n ≤ 1) (h_sum_leq_253_div_12 : nums.sum ≤ 253 / 12) :
  ∃ (A B : List ℝ), (∀ x ∈ A, x ∈ nums) ∧ (∀ y ∈ B, y ∈ nums) ∧ A.union B = nums ∧ A.sum ≤ 11 ∧ B.sum ≤ 11 :=
sorry

end max_S_possible_l24_24347


namespace function_increasing_in_range_l24_24353

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - m) * x - m else Real.log x / Real.log m

theorem function_increasing_in_range (m : ℝ) :
  (3 / 2 ≤ m ∧ m < 3) ↔ (∀ x y : ℝ, x < y → f m x < f m y) := by
  sorry

end function_increasing_in_range_l24_24353


namespace binom_12_6_eq_924_l24_24808

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binom_12_6_eq_924_l24_24808


namespace sum_of_interior_edges_l24_24461

-- Conditions
def width_of_frame_piece : ℝ := 1.5
def one_interior_edge : ℝ := 4.5
def total_frame_area : ℝ := 27

-- Statement of the problem as a theorem in Lean
theorem sum_of_interior_edges : 
  (∃ y : ℝ, (width_of_frame_piece * 2 + one_interior_edge) * (width_of_frame_piece * 2 + y) 
    - one_interior_edge * y = total_frame_area) →
  (4 * (one_interior_edge + y) = 12) :=
sorry

end sum_of_interior_edges_l24_24461


namespace radius_of_circle_l24_24180

noncomputable def radius (α : ℝ) : ℝ :=
  5 / Real.sin (α / 2)

theorem radius_of_circle (c α : ℝ) (h_c : c = 10) :
  (radius α) = 5 / Real.sin (α / 2) := by
  sorry

end radius_of_circle_l24_24180


namespace train_speed_in_km_per_hr_l24_24794

variables (L : ℕ) (t : ℕ) (train_speed : ℕ)

-- Conditions
def length_of_train : ℕ := 1050
def length_of_platform : ℕ := 1050
def crossing_time : ℕ := 1

-- Given calculation of speed in meters per minute
def speed_in_m_per_min : ℕ := (length_of_train + length_of_platform) / crossing_time

-- Conversion units
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000
def minutes_to_hours (min : ℕ) : ℕ := min / 60

-- Speed in km/hr
def speed_in_km_per_hr : ℕ := speed_in_m_per_min * (meters_to_kilometers 1000) * (minutes_to_hours 60)

theorem train_speed_in_km_per_hr : speed_in_km_per_hr = 35 :=
by {
  -- We will include the proof steps here, but for now, we just assert with sorry.
  sorry
}

end train_speed_in_km_per_hr_l24_24794


namespace set_difference_is_single_element_l24_24484

-- Define the sets M and N based on the given conditions
def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2002}
def N : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2003}

-- State the theorem that we need to prove
theorem set_difference_is_single_element : (N \ M) = {2003} :=
sorry

end set_difference_is_single_element_l24_24484


namespace monotonic_decreasing_interval_l24_24011

open Real

-- Definitions and conditions used to ensure the function is meaningful
def valid_domain (x : ℝ) : Prop := 4 + 3 * x - x^2 > 0

-- Define the function
noncomputable def f (x : ℝ) : ℝ := ln (4 + 3 * x - x^2)

-- The goal is to find the monotonic decreasing interval of f(x)
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, valid_domain x → x ∈ set.Icc (3 / 2) 4 → 
  ∀ y : ℝ, valid_domain y → y ∈ set.Icc (3 / 2) 4 → x < y → f x ≥ f y :=
sorry

end monotonic_decreasing_interval_l24_24011


namespace swim_team_girls_l24_24044

-- Definitions using the given conditions
variables (B G : ℕ)
theorem swim_team_girls (h1 : G = 5 * B) (h2 : G + B = 96) : G = 80 :=
sorry

end swim_team_girls_l24_24044


namespace perimeter_change_l24_24700

theorem perimeter_change (s h : ℝ) 
  (h1 : 2 * (1.3 * s + 0.8 * h) = 2 * (s + h)) :
  (2 * (0.8 * s + 1.3 * h) = 1.1 * (2 * (s + h))) :=
by
  sorry

end perimeter_change_l24_24700


namespace cauchy_problem_solution_l24_24779

noncomputable def x_solution (t : ℝ) : ℝ := 4 * Real.exp t + 2 * Real.exp (-t)
noncomputable def y_solution (t : ℝ) : ℝ := -Real.exp t - Real.exp (-t)

theorem cauchy_problem_solution :
  (∀ t : ℝ, (differential.differential (λ t, x_solution t) = (λ t, 3 * x_solution t + 8 * y_solution t) t) ∧
            (differential.differential (λ t, y_solution t) = (λ t, -(x_solution t) - 3 * y_solution t) t)) ∧
  (x_solution 0 = 6) ∧
  (y_solution 0 = -2) :=
by
  sorry

end cauchy_problem_solution_l24_24779


namespace fraction_not_integer_l24_24723

theorem fraction_not_integer (a b : ℕ) (h : a ≠ b) (parity: (a % 2 = b % 2)) 
(h_pos_a : 0 < a) (h_pos_b : 0 < b) : ¬ ∃ k : ℕ, (a! + b!) = k * 2^a := 
by sorry

end fraction_not_integer_l24_24723


namespace original_price_of_cycle_l24_24309

theorem original_price_of_cycle 
    (selling_price : ℝ) 
    (loss_percentage : ℝ) 
    (h1 : selling_price = 1120)
    (h2 : loss_percentage = 0.20) : 
    ∃ P : ℝ, P = 1400 :=
by
  sorry

end original_price_of_cycle_l24_24309


namespace calculate_S2018_l24_24717

def seq_a : ℕ → ℝ
| 1     := real.sqrt 2
| (n+2) := real.sqrt ((seq_a (n+1))^2 + 2)

def seq_b (n : ℕ) : ℝ :=
  4 / ((seq_a n)^2 * (seq_a (n+1))^2)

def seq_S (n : ℕ) : ℝ :=
  ∑ i in finset.range n, seq_b (i + 1)

theorem calculate_S2018 :
  seq_S 2018 = 2018 / 2019 := 
sorry

end calculate_S2018_l24_24717


namespace binom_12_6_eq_924_l24_24806

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binom_12_6_eq_924_l24_24806


namespace extreme_points_l24_24556

noncomputable def f (x : ℝ) : ℝ :=
  x + 4 / x

theorem extreme_points (P : ℝ × ℝ) :
  (P = (2, f 2) ∨ P = (-2, f (-2))) ↔ 
  ∃ x : ℝ, x ≠ 0 ∧ (P = (x, f x)) ∧ 
    (∀ ε > 0, f (x - ε) < f x ∧ f x > f (x + ε) ∨ f (x - ε) > f x ∧ f x < f (x + ε)) := 
sorry

end extreme_points_l24_24556


namespace white_square_area_l24_24734

theorem white_square_area
    (edge_length : ℝ)
    (total_paint : ℝ)
    (total_surface_area : ℝ)
    (green_paint_per_face : ℝ)
    (white_square_area_per_face: ℝ) :
    edge_length = 12 →
    total_paint = 432 →
    total_surface_area = 6 * (edge_length ^ 2) →
    green_paint_per_face = total_paint / 6 →
    white_square_area_per_face = (edge_length ^ 2) - green_paint_per_face →
    white_square_area_per_face = 72
:= sorry

end white_square_area_l24_24734


namespace product_gcd_lcm_15_9_l24_24336

theorem product_gcd_lcm_15_9 : Nat.gcd 15 9 * Nat.lcm 15 9 = 135 := 
by
  -- skipping proof as instructed
  sorry

end product_gcd_lcm_15_9_l24_24336


namespace flowers_per_day_l24_24574

-- Definitions for conditions
def total_flowers := 360
def days := 6

-- Proof that the number of flowers Miriam can take care of in one day is 60
theorem flowers_per_day : total_flowers / days = 60 := by
  sorry

end flowers_per_day_l24_24574


namespace largest_integer_divides_product_l24_24877

theorem largest_integer_divides_product (n : ℕ) : 
  ∃ m, ∀ k : ℕ, k = (2*n-1)*(2*n)*(2*n+2) → m ≥ 1 ∧ m = 8 ∧ m ∣ k :=
by
  sorry

end largest_integer_divides_product_l24_24877


namespace find_g_5_l24_24903

noncomputable def g : ℝ → ℝ := sorry

axiom g_property (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0

theorem find_g_5 : g 5 = 1 :=
by
  sorry

end find_g_5_l24_24903


namespace cistern_fill_time_l24_24932

theorem cistern_fill_time (F E : ℝ) (hF : F = 1/3) (hE : E = 1/6) : (1 / (F - E)) = 6 :=
by sorry

end cistern_fill_time_l24_24932


namespace common_difference_of_variance_is_half_l24_24495

variable {a : ℕ → ℝ} {d : ℝ}

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def mean (s : Fin 7 → ℝ) : ℝ :=
  (1 / 7) * (Finset.sum (Finset.univ : Finset (Fin 7)) s)

def variance (s : Fin 7 → ℝ) : ℝ :=
  mean (λ i : Fin 7, (s i - mean s)^2)

theorem common_difference_of_variance_is_half :
  arithmetic_seq a d →
  variance (λ i, a i.succ) = 1 →
  d = 1/2 ∨ d = -1/2 :=
by
  sorry

end common_difference_of_variance_is_half_l24_24495


namespace greatest_possible_value_of_y_l24_24743

-- Definitions according to problem conditions
variables {x y : ℤ}

-- The theorem statement to prove
theorem greatest_possible_value_of_y (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
sorry

end greatest_possible_value_of_y_l24_24743


namespace roots_of_quadratic_l24_24756

theorem roots_of_quadratic (x : ℝ) : (x - 3) ^ 2 = 25 ↔ (x = 8 ∨ x = -2) :=
by sorry

end roots_of_quadratic_l24_24756


namespace solve_system_l24_24254

noncomputable def system_solutions (x y z : ℤ) : Prop :=
  x^3 + y^3 + z^3 = 8 ∧
  x^2 + y^2 + z^2 = 22 ∧
  (1 / x + 1 / y + 1 / z = - (z / (x * y)))

theorem solve_system :
  ∀ (x y z : ℤ), system_solutions x y z ↔ 
    (x = 3 ∧ y = 2 ∧ z = -3) ∨
    (x = -3 ∧ y = 2 ∧ z = 3) ∨
    (x = 2 ∧ y = 3 ∧ z = -3) ∨
    (x = 2 ∧ y = -3 ∧ z = 3) := by
  sorry

end solve_system_l24_24254


namespace exists_twelve_distinct_x_l24_24519

theorem exists_twelve_distinct_x :
  ∃ S : Set ℝ, (S.card = 12) ∧ (∀ x ∈ S, ∃ k : ℤ, 0 ≤ k ∧ k ≤ 11 ∧ (sqrt (123 - sqrt x) = k)) :=
by
  sorry

end exists_twelve_distinct_x_l24_24519


namespace circle_reflection_l24_24592

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3) : 
  let reflected_x := -y, reflected_y := -x in
  reflected_x = 3 ∧ reflected_y = -8 :=
by
  sorry

end circle_reflection_l24_24592


namespace least_possible_value_of_squares_l24_24608

theorem least_possible_value_of_squares (a b x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h1 : 15 * a + 16 * b = x^2) (h2 : 16 * a - 15 * b = y^2) : 
  ∃ (x : ℕ) (y : ℕ), min (x^2) (y^2) = 231361 := 
sorry

end least_possible_value_of_squares_l24_24608


namespace simplify_sin_cos_expr_cos_pi_six_alpha_expr_l24_24449

open Real

-- Problem (1)
theorem simplify_sin_cos_expr (x : ℝ) :
  (sin x ^ 2 / (sin x - cos x)) - ((sin x + cos x) / (tan x ^ 2 - 1)) - sin x = cos x :=
sorry

-- Problem (2)
theorem cos_pi_six_alpha_expr (α : ℝ) (h : cos (π / 6 - α) = sqrt 3 / 3) :
  cos (5 * π / 6 + α) + cos (4 * π / 3 + α) ^ 2 = (2 - sqrt 3) / 3 :=
sorry

end simplify_sin_cos_expr_cos_pi_six_alpha_expr_l24_24449


namespace not_enough_info_sweets_l24_24738

theorem not_enough_info_sweets
    (S : ℕ)         -- Initial number of sweet cookies.
    (initial_salty : ℕ := 6)  -- Initial number of salty cookies given as 6.
    (eaten_sweets : ℕ := 20)   -- Number of sweet cookies Paco ate.
    (eaten_salty : ℕ := 34)    -- Number of salty cookies Paco ate.
    (diff_eaten : eaten_salty - eaten_sweets = 14) -- Paco ate 14 more salty cookies than sweet cookies.
    : (∃ S', S' = S) → False :=  -- Conclusion: Not enough information to determine initial number of sweet cookies S.
by
  sorry

end not_enough_info_sweets_l24_24738


namespace minimum_value_exists_l24_24906

theorem minimum_value_exists (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) ∧ (1 / m + 1 / n ≥ min_val) :=
by {
  -- Proof will be provided here.
  sorry
}

end minimum_value_exists_l24_24906


namespace vasya_can_win_l24_24019

noncomputable def initial_first : ℝ := 1 / 2009
noncomputable def initial_second : ℝ := 1 / 2008
noncomputable def increment : ℝ := 1 / (2008 * 2009)

theorem vasya_can_win :
  ∃ n : ℕ, ((2009 * n) * increment = 1) ∨ ((2008 * n) * increment = 1) :=
sorry

end vasya_can_win_l24_24019


namespace terminal_side_in_fourth_quadrant_l24_24216

theorem terminal_side_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  (θ ≥ 0 ∧ θ < Real.pi/2) ∨ (θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi) :=
sorry

end terminal_side_in_fourth_quadrant_l24_24216


namespace number_of_elements_l24_24913

theorem number_of_elements (n : ℕ) (S : ℕ) (sum_first_six : ℕ) (sum_last_six : ℕ) (sixth_number : ℕ)
    (h1 : S = 22 * n) 
    (h2 : sum_first_six = 6 * 19) 
    (h3 : sum_last_six = 6 * 27) 
    (h4 : sixth_number = 34) 
    (h5 : S = sum_first_six + sum_last_six - sixth_number) : 
    n = 11 := 
by
  sorry

end number_of_elements_l24_24913


namespace eq_m_neg_one_l24_24209

theorem eq_m_neg_one (m : ℝ) (x : ℝ) (h1 : (m-1) * x^(m^2 + 1) + 2*x - 3 = 0) (h2 : m - 1 ≠ 0) (h3 : m^2 + 1 = 2) : 
  m = -1 :=
sorry

end eq_m_neg_one_l24_24209


namespace find_x_l24_24135

variables (x : ℝ)
axiom h1 : (180 / x) + (5 * 12 / x) + 80 = 81

theorem find_x : x = 240 :=
by {
  sorry
}

end find_x_l24_24135


namespace law_of_motion_l24_24038

-- Input Conditions
variables (m ω : ℝ) (x : ℝ → ℝ)

-- Assumptions
axiom (h1 : ∃ C₁ C₂ t : ℝ, x t = C₁ * Real.cos (ω * t) + C₂ * Real.sin (ω * t))
axiom (h2 : m ≠ 0)
axiom (h3 : ∀ t, x'' t = - ω^2 * x t)

-- Required to Prove
theorem law_of_motion (x : ℝ → ℝ) (R α : ℝ) :
  x = (λ t, R * Real.sin (ω * t + α)) :=
sorry

end law_of_motion_l24_24038


namespace max_product_of_triangle_sides_l24_24680

theorem max_product_of_triangle_sides (a c : ℝ) (ha : a ≥ 0) (hc : c ≥ 0) :
  ∃ b : ℝ, b = 4 ∧ ∃ B : ℝ, B = 60 * (π / 180) ∧ a^2 + c^2 - a * c = b^2 ∧ a * c ≤ 16 :=
by
  sorry

end max_product_of_triangle_sides_l24_24680


namespace triangle_third_side_l24_24224

theorem triangle_third_side (a b : ℝ) (θ : ℝ) (cos_θ : ℝ) : 
  a = 9 → b = 12 → θ = 150 → cos_θ = - (Real.sqrt 3 / 2) → 
  (Real.sqrt (a^2 + b^2 - 2 * a * b * cos_θ)) = Real.sqrt (225 + 108 * Real.sqrt 3) := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end triangle_third_side_l24_24224


namespace part_a_l24_24445

theorem part_a (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) :
  ∃ (m : ℕ) (x1 x2 x3 x4 : ℤ), m < p ∧ (x1^2 + x2^2 + x3^2 + x4^2 = m * p) :=
sorry

end part_a_l24_24445


namespace circle_reflection_l24_24593

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3) : 
  let reflected_x := -y, reflected_y := -x in
  reflected_x = 3 ∧ reflected_y = -8 :=
by
  sorry

end circle_reflection_l24_24593


namespace camera_pics_l24_24438

-- Definitions of the given conditions
def phone_pictures := 22
def albums := 4
def pics_per_album := 6

-- The statement to prove the number of pictures uploaded from camera
theorem camera_pics : (albums * pics_per_album) - phone_pictures = 2 :=
by
  sorry

end camera_pics_l24_24438


namespace probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l24_24710

theorem probability_exactly_k_gnomes_fall (n k : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in p * q^(n - k) = p * (1 - p)^(n - k) := 
sorry

theorem expected_number_of_gnomes_fall (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) :
  let q := 1 - p in 
  (∑ j in finset.range n, (1 - q^(j+1))) = n + 1 - (1 / p) + ((1 - p)^(n+1) / p) :=
sorry

end probability_exactly_k_gnomes_fall_expected_number_of_gnomes_fall_l24_24710


namespace expected_rolls_in_non_leap_year_l24_24172

theorem expected_rolls_in_non_leap_year :
  let E := (1 : ℚ) + (1 / 10) * (E : ℚ) in
  E = 10 / 9 →
  (365 * E) = 3650 / 9 :=
by
  intro h
  sorry

end expected_rolls_in_non_leap_year_l24_24172


namespace Mary_is_10_years_younger_l24_24659

theorem Mary_is_10_years_younger
  (betty_age : ℕ)
  (albert_age : ℕ)
  (mary_age : ℕ)
  (h1 : albert_age = 2 * mary_age)
  (h2 : albert_age = 4 * betty_age)
  (h_betty : betty_age = 5) :
  (albert_age - mary_age) = 10 :=
  by
  sorry

end Mary_is_10_years_younger_l24_24659


namespace max_value_ln_x_plus_x_l24_24678

theorem max_value_ln_x_plus_x (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ Real.exp 1) : 
  ∃ y, y = Real.log x + x ∧ y ≤ Real.log (Real.exp 1) + Real.exp 1 :=
sorry

end max_value_ln_x_plus_x_l24_24678


namespace probability_B_given_A_l24_24762

-- Define the events A and B
def event_A (x y : ℕ) : Prop :=
  (x % 2 = 1) ∧ (y % 2 = 1)

def event_B (x y : ℕ) : Prop :=
  (x + y = 4)

-- Define the probability measure on the space of dice rolls
noncomputable def P (s : set (ℕ × ℕ)) : ℚ :=
  (s.to_finset.card : ℚ) / 36

-- Define the conditional probability P(B|A)
noncomputable def P_B_given_A : ℚ :=
  let A_outcomes := {pair | event_A pair.fst pair.snd} in
  let B_given_A_outcomes := {pair | event_A pair.fst pair.snd ∧ event_B pair.fst pair.snd} in
  (B_given_A_outcomes.to_finset.card : ℚ) / (A_outcomes.to_finset.card : ℚ)

-- Prove that P(B|A) is 2/9
theorem probability_B_given_A : P_B_given_A = 2 / 9 :=
by
  sorry

end probability_B_given_A_l24_24762


namespace p_value_for_roots_l24_24690

theorem p_value_for_roots (α β : ℝ) (h1 : 3 * α^2 + 5 * α + 2 = 0) (h2 : 3 * β^2 + 5 * β + 2 = 0)
  (hαβ : α + β = -5/3) (hαβ_prod : α * β = 2/3) : p = -49/9 :=
by
  sorry

end p_value_for_roots_l24_24690


namespace additional_license_plates_l24_24181

def original_license_plates : ℕ := 5 * 3 * 5
def new_license_plates : ℕ := 6 * 4 * 5

theorem additional_license_plates : new_license_plates - original_license_plates = 45 := by
  sorry

end additional_license_plates_l24_24181


namespace num_real_x_l24_24517

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end num_real_x_l24_24517


namespace even_factors_count_l24_24699

theorem even_factors_count (n : ℕ) (h : n = 2^4 * 3^2 * 5 * 7) : 
  ∃ k : ℕ, k = 48 ∧ ∃ a b c d : ℕ, 
  1 ≤ a ∧ a ≤ 4 ∧
  0 ≤ b ∧ b ≤ 2 ∧
  0 ≤ c ∧ c ≤ 1 ∧
  0 ≤ d ∧ d ≤ 1 ∧
  k = (4 - 1 + 1) * (2 + 1) * (1 + 1) * (1 + 1) := by
  sorry

end even_factors_count_l24_24699


namespace max_value_of_expression_l24_24554

theorem max_value_of_expression 
  (x y : ℝ) 
  (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  x^2 + y^2 + 2 * x ≤ 15 := sorry

end max_value_of_expression_l24_24554


namespace car_speed_l24_24037

-- Definitions based on the conditions
def distance : ℕ := 375
def time : ℕ := 5

-- Mathematically equivalent proof statement
theorem car_speed : distance / time = 75 := 
  by
  -- The actual proof will be placed here, but we'll skip it for now.
  sorry

end car_speed_l24_24037


namespace find_f_value_l24_24504

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^5 - b * x^3 + c * x - 3

theorem find_f_value (a b c : ℝ) (h : f a b c (-3) = 7) : f a b c 3 = -13 :=
by
  sorry

end find_f_value_l24_24504


namespace owls_joined_l24_24452

theorem owls_joined (initial_owls : ℕ) (total_owls : ℕ) (join_owls : ℕ) 
  (h_initial : initial_owls = 3) (h_total : total_owls = 5) : join_owls = 2 :=
by {
  -- Sorry is used to skip the proof
  sorry
}

end owls_joined_l24_24452


namespace fly_flies_more_than_10_meters_l24_24303

theorem fly_flies_more_than_10_meters :
  ∃ (fly_path_length : ℝ), 
  (∃ (c : ℝ) (a b : ℝ), c = 5 ∧ a^2 + b^2 = c^2) →
  (fly_path_length > 10) := 
by
  sorry

end fly_flies_more_than_10_meters_l24_24303


namespace focal_length_of_hyperbola_l24_24354

theorem focal_length_of_hyperbola (a b p: ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (p_pos : 0 < p) :
  (∃ (F V : ℝ × ℝ), 4 = dist F V ∧ F = (2, 0) ∧ V = (-2, 0)) ∧
  (∃ (P : ℝ × ℝ), P = (-2, -1) ∧ (∃ (d : ℝ), d = d / 2 ∧ P = (d, 0))) →
  2 * (Real.sqrt (a^2 + b^2)) = 2 * Real.sqrt 5 := 
sorry

end focal_length_of_hyperbola_l24_24354


namespace minimum_value_l24_24184

noncomputable def min_value_of_trig_function : ℝ :=
  Real.sin (5 * Real.pi / 12)

theorem minimum_value (x : ℝ) (h : -Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 6) :
  has_min_on (λ x, Real.cos (x + Real.pi / 3) - Real.cot (x + Real.pi / 3) + Real.sin (x + Real.pi / 4))
             (Set.Icc (-Real.pi / 12) (Real.pi / 6)) min_value_of_trig_function :=
sorry

end minimum_value_l24_24184


namespace Carlton_button_up_shirts_l24_24177

/-- 
Given that the number of sweater vests V is twice the number of button-up shirts S, 
and the total number of unique outfits (each combination of a sweater vest and a button-up shirt) is 18, 
prove that the number of button-up shirts S is 3. 
-/
theorem Carlton_button_up_shirts (V S : ℕ) (h1 : V = 2 * S) (h2 : V * S = 18) : S = 3 := by
  sorry

end Carlton_button_up_shirts_l24_24177


namespace number_of_rectangular_arrays_of_chairs_l24_24074

/-- 
Given a classroom that contains 45 chairs, prove that 
the number of rectangular arrays of chairs that can be made such that 
each row contains at least 3 chairs and each column contains at least 3 chairs is 4.
-/
theorem number_of_rectangular_arrays_of_chairs : 
  ∃ (n : ℕ), n = 4 ∧ 
    ∀ (a b : ℕ), (a * b = 45) → 
      (a ≥ 3) → (b ≥ 3) → 
      (n = 4) := 
sorry

end number_of_rectangular_arrays_of_chairs_l24_24074


namespace binomial_12_6_eq_1848_l24_24816

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end binomial_12_6_eq_1848_l24_24816


namespace erasers_left_in_the_box_l24_24613

-- Conditions expressed as definitions
def E0 : ℕ := 320
def E1 : ℕ := E0 - 67
def E2 : ℕ := E1 - 126
def E3 : ℕ := E2 + 30

-- Proof problem statement
theorem erasers_left_in_the_box : E3 = 157 := 
by sorry

end erasers_left_in_the_box_l24_24613


namespace train_speed_kph_l24_24467

-- Definitions based on conditions
def time_seconds : ℕ := 9
def length_meters : ℕ := 135
def conversion_factor : ℕ := 36 -- 3.6 represented as an integer by multiplying both sides by 10

-- The proof statement
theorem train_speed_kph : (length_meters * conversion_factor / 10 / time_seconds = 54) :=
by
  sorry

end train_speed_kph_l24_24467


namespace length_of_bridge_l24_24035

theorem length_of_bridge
  (length_of_train : ℕ)
  (speed_km_per_hr : ℕ)
  (crossing_time_sec : ℕ)
  (h_train_length : length_of_train = 100)
  (h_speed : speed_km_per_hr = 45)
  (h_time : crossing_time_sec = 30) :
  ∃ (length_of_bridge : ℕ), length_of_bridge = 275 :=
by
  -- Convert speed from km/hr to m/s
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600
  -- Total distance the train travels in crossing_time_sec
  let total_distance := speed_m_per_s * crossing_time_sec
  -- Length of the bridge
  let bridge_length := total_distance - length_of_train
  use bridge_length
  -- Skip the detailed proof steps
  sorry

end length_of_bridge_l24_24035


namespace trajectory_of_P_l24_24311

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (2 * x - 3) ^ 2 + 4 * y ^ 2 = 1

theorem trajectory_of_P (m n x y : ℝ) (hM_on_circle : m^2 + n^2 = 1)
  (hP_midpoint : 2 * x = 3 + m ∧ 2 * y = n) : trajectory_equation x y :=
by 
  sorry

end trajectory_of_P_l24_24311


namespace simplify_expression_l24_24582

theorem simplify_expression (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m / n - n / m) / (1 / m - 1 / n) = -(m + n) :=
by sorry

end simplify_expression_l24_24582


namespace students_in_front_of_Yuna_l24_24297

-- Defining the total number of students
def total_students : ℕ := 25

-- Defining the number of students behind Yuna
def students_behind_Yuna : ℕ := 9

-- Defining Yuna's position from the end of the line
def Yuna_position_from_end : ℕ := students_behind_Yuna + 1

-- Statement to prove the number of students in front of Yuna
theorem students_in_front_of_Yuna : (total_students - Yuna_position_from_end) = 15 := by
  sorry

end students_in_front_of_Yuna_l24_24297


namespace quadratic_inequality_solution_set_l24_24758

theorem quadratic_inequality_solution_set (x : ℝ) :
  (x^2 - 3 * x - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 4) :=
sorry

end quadratic_inequality_solution_set_l24_24758


namespace closest_points_to_A_l24_24101

noncomputable def distance_squared (x y : ℝ) : ℝ :=
  x^2 + (y + 3)^2

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 9

theorem closest_points_to_A :
  ∃ (x y : ℝ),
    hyperbola x y ∧
    (distance_squared x y = distance_squared (-3 * Real.sqrt 5 / 2) (-3/2) ∨
     distance_squared x y = distance_squared (3 * Real.sqrt 5 / 2) (-3/2)) :=
sorry

end closest_points_to_A_l24_24101


namespace pipe_filling_time_l24_24885

-- Definitions for the conditions
variables (A : ℝ) (h : 1 / A - 1 / 24 = 1 / 12)

-- The statement of the problem
theorem pipe_filling_time : A = 8 :=
by
  sorry

end pipe_filling_time_l24_24885


namespace Vanya_two_digit_number_l24_24286

-- Define the conditions as a mathematical property
theorem Vanya_two_digit_number:
  ∃ (m n : ℕ), 0 ≤ m ∧ m ≤ 9 ∧ 0 ≤ n ∧ n ≤ 9 ∧ (10 * n + m) ^ 2 = 4 * (10 * m + n) ∧ (10 * m + n) = 81 :=
by
  -- Remember to replace the proof with 'sorry'
  sorry

end Vanya_two_digit_number_l24_24286


namespace batsman_average_is_18_l24_24158
noncomputable def average_after_18_innings (score_18th: ℕ) (average_17th: ℕ) (innings: ℕ) : ℕ :=
  let total_runs_17 := average_17th * 17
  let total_runs_18 := total_runs_17 + score_18th
  total_runs_18 / innings

theorem batsman_average_is_18 {score_18th: ℕ} {average_17th: ℕ} {expected_average: ℕ} :
  score_18th = 1 → average_17th = 19 → expected_average = 18 →
  average_after_18_innings score_18th average_17th 18 = expected_average := by
  sorry

end batsman_average_is_18_l24_24158


namespace larger_angle_is_99_l24_24424

theorem larger_angle_is_99 (x : ℝ) (h1 : 2 * x + 18 = 180) : x + 18 = 99 :=
by
  sorry

end larger_angle_is_99_l24_24424


namespace math_problem_l24_24474

theorem math_problem : 
  27 * ((8/3 : ℚ) - (13/4 : ℚ)) / ((3/2 : ℚ) + (11/5 : ℚ)) = -4 - (43/74 : ℚ) :=
by
  sorry

end math_problem_l24_24474


namespace line_through_point_parallel_l24_24008

theorem line_through_point_parallel (p : ℝ × ℝ) (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0)
  (hp : a * p.1 + b * p.2 + c = 0) :
  ∃ k : ℝ, a * p.1 + b * p.2 + k = 0 :=
by
  use - (a * p.1 + b * p.2)
  sorry

end line_through_point_parallel_l24_24008


namespace jill_draws_spade_probability_l24_24866

noncomputable def probability_jill_draws_spade : ℚ :=
  ∑' (k : ℕ), ((3 / 4) * (3 / 4))^k * ((3 / 4) * (1 / 4))

theorem jill_draws_spade_probability : probability_jill_draws_spade = 3 / 7 :=
sorry

end jill_draws_spade_probability_l24_24866


namespace no_positive_int_solutions_l24_24103

theorem no_positive_int_solutions
  (x y z t : ℕ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (ht : 0 < t)
  (h1 : x^2 + 2 * y^2 = z^2)
  (h2 : 2 * x^2 + y^2 = t^2) : false :=
by
  sorry

end no_positive_int_solutions_l24_24103


namespace dice_product_composite_probability_l24_24539

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l24_24539


namespace point_outside_circle_l24_24501

theorem point_outside_circle
  (radius : ℝ) (distance : ℝ) (h_radius : radius = 8) (h_distance : distance = 10) :
  distance > radius :=
by sorry

end point_outside_circle_l24_24501


namespace no_first_quadrant_l24_24988

theorem no_first_quadrant (a b : ℝ) (h_a : a < 0) (h_b : b < 0) (h_am : (a - b) < 0) :
  ¬∃ x : ℝ, (a - b) * x + b > 0 ∧ x > 0 :=
sorry

end no_first_quadrant_l24_24988


namespace paint_per_door_l24_24957

variable (cost_per_pint : ℕ) (cost_per_gallon : ℕ) (num_doors : ℕ) (pints_per_gallon : ℕ) (savings : ℕ)

theorem paint_per_door :
  cost_per_pint = 8 →
  cost_per_gallon = 55 →
  num_doors = 8 →
  pints_per_gallon = 8 →
  savings = 9 →
  (pints_per_gallon / num_doors = 1) :=
by
  intros h_cpint h_cgallon h_nd h_pgallon h_savings
  sorry

end paint_per_door_l24_24957


namespace sum_of_n_and_k_l24_24111

open Nat

theorem sum_of_n_and_k (n k : ℕ) (h1 : (n.choose (k + 1)) = 3 * (n.choose k))
                      (h2 : (n.choose (k + 2)) = 2 * (n.choose (k + 1))) :
    n + k = 7 := by
  sorry

end sum_of_n_and_k_l24_24111


namespace range_of_a_l24_24724

theorem range_of_a {a : ℝ} : (∀ x : ℝ, (x^2 + 2 * (a + 1) * x + a^2 - 1 = 0) → (x = 0 ∨ x = -4)) → (a = 1 ∨ a ≤ -1) := 
by {
  sorry
}

end range_of_a_l24_24724


namespace composite_dice_product_probability_l24_24534

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l24_24534


namespace major_axis_length_l24_24170

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def foci_1 : ℝ × ℝ := (3, 5)
def foci_2 : ℝ × ℝ := (23, 40)
def reflected_foci_1 : ℝ × ℝ := (-3, 5)

theorem major_axis_length :
  distance (reflected_foci_1.1) (reflected_foci_1.2) (foci_2.1) (foci_2.2) = Real.sqrt 1921 :=
sorry

end major_axis_length_l24_24170


namespace calc_quotient_l24_24028

theorem calc_quotient (a b : ℕ) (h1 : a - b = 177) (h2 : 14^2 = 196) : (a - b)^2 / 196 = 144 := 
by sorry

end calc_quotient_l24_24028


namespace probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l24_24714

noncomputable def probability_k_gnomes_fall (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p * (1 - p) ^ (n - k)

noncomputable def expected_number_of_fallen_gnomes (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p)

theorem probability_k_gnomes_fall_correct (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  probability_k_gnomes_fall n k p h = p * (1 - p) ^ (n - k) :=
by sorry

theorem expected_number_of_fallen_gnomes_correct (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  expected_number_of_fallen_gnomes n p h = n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p) :=
by sorry

end probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l24_24714


namespace find_xy_plus_yz_plus_xz_l24_24842

theorem find_xy_plus_yz_plus_xz
  (x y z : ℝ)
  (h₁ : x > 0)
  (h₂ : y > 0)
  (h₃ : z > 0)
  (eq1 : x^2 + x * y + y^2 = 75)
  (eq2 : y^2 + y * z + z^2 = 64)
  (eq3 : z^2 + z * x + x^2 = 139) :
  x * y + y * z + z * x = 80 :=
by
  sorry

end find_xy_plus_yz_plus_xz_l24_24842


namespace intersection_A_B_l24_24675

noncomputable def A : Set ℝ := { y | ∃ x : ℝ, y = Real.sin x }
noncomputable def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_A_B : A ∩ B = { y | 0 ≤ y ∧ y ≤ 1 } :=
by 
  sorry

end intersection_A_B_l24_24675


namespace ratio_of_saramago_readers_l24_24862

theorem ratio_of_saramago_readers 
  (W : ℕ) (S K B N : ℕ)
  (h1 : W = 42)
  (h2 : K = W / 6)
  (h3 : B = 3)
  (h4 : N = (S - B) - 1)
  (h5 : W = (S - B) + (K - B) + B + N) :
  S / W = 1 / 2 :=
by
  sorry

end ratio_of_saramago_readers_l24_24862


namespace num_pos_whole_numbers_with_cube_roots_less_than_five_l24_24852

theorem num_pos_whole_numbers_with_cube_roots_less_than_five : 
  {n : ℕ | ∃ k : ℕ, k < 5 ∧ k^3 = n}.card = 124 :=
sorry

end num_pos_whole_numbers_with_cube_roots_less_than_five_l24_24852


namespace disease_cases_linear_decrease_l24_24222

theorem disease_cases_linear_decrease (cases_1970 cases_2010 cases_1995 cases_2005 : ℕ)
  (year_1970 year_2010 year_1995 year_2005 : ℕ)
  (h_cases_1970 : cases_1970 = 800000)
  (h_cases_2010 : cases_2010 = 200)
  (h_year_1970 : year_1970 = 1970)
  (h_year_2010 : year_2010 = 2010)
  (h_year_1995 : year_1995 = 1995)
  (h_year_2005 : year_2005 = 2005)
  (linear_decrease : ∀ t, cases_1970 - (cases_1970 - cases_2010) * (t - year_1970) / (year_2010 - year_1970) = cases_1970 - t * (cases_1970 - cases_2010) / (year_2010 - year_1970))
  : cases_1995 = 300125 ∧ cases_2005 = 100175 := sorry

end disease_cases_linear_decrease_l24_24222


namespace depth_of_water_in_cistern_l24_24652

-- Define the given constants
def length_cistern : ℝ := 6
def width_cistern : ℝ := 5
def total_wet_area : ℝ := 57.5

-- Define the area of the bottom of the cistern
def area_bottom (length : ℝ) (width : ℝ) : ℝ := length * width

-- Define the area of the longer sides of the cistern in contact with water
def area_long_sides (length : ℝ) (depth : ℝ) : ℝ := 2 * length * depth

-- Define the area of the shorter sides of the cistern in contact with water
def area_short_sides (width : ℝ) (depth : ℝ) : ℝ := 2 * width * depth

-- Define the total wet surface area based on depth of the water
def total_wet_surface_area (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ := 
    area_bottom length width + area_long_sides length depth + area_short_sides width depth

-- Define the proof statement
theorem depth_of_water_in_cistern : ∃ h : ℝ, h = 1.25 ∧ total_wet_surface_area length_cistern width_cistern h = total_wet_area := 
by
  use 1.25
  sorry

end depth_of_water_in_cistern_l24_24652


namespace digit_to_make_52B6_divisible_by_3_l24_24020

theorem digit_to_make_52B6_divisible_by_3 (B : ℕ) (hB : 0 ≤ B ∧ B ≤ 9) : 
  (5 + 2 + B + 6) % 3 = 0 ↔ (B = 2 ∨ B = 5 ∨ B = 8) := 
by
  sorry

end digit_to_make_52B6_divisible_by_3_l24_24020


namespace sum_first_nine_terms_arithmetic_sequence_l24_24562

theorem sum_first_nine_terms_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = (a 2 - a 1))
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 3 + a 6 + a 9 = 27) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) = 108 := 
sorry

end sum_first_nine_terms_arithmetic_sequence_l24_24562


namespace min_supreme_supervisors_l24_24125

-- Definitions
def num_employees : ℕ := 50000
def supervisors (e : ℕ) : ℕ := 7 - e

-- Theorem statement
theorem min_supreme_supervisors (k : ℕ) (num_employees_le_reached : ∀ n : ℕ, 50000 ≤ n) : 
  k ≥ 28 := 
sorry

end min_supreme_supervisors_l24_24125


namespace scientific_notation_of_000000301_l24_24966

/--
Expressing a small number in scientific notation:
Prove that \(0.000000301\) can be written as \(3.01 \times 10^{-7}\).
-/
theorem scientific_notation_of_000000301 :
  0.000000301 = 3.01 * 10 ^ (-7) :=
sorry

end scientific_notation_of_000000301_l24_24966


namespace Nicki_total_miles_run_l24_24246

theorem Nicki_total_miles_run:
  ∀ (miles_per_week_first_half miles_per_week_second_half weeks_in_year weeks_per_half_year : ℕ),
  miles_per_week_first_half = 20 →
  miles_per_week_second_half = 30 →
  weeks_in_year = 52 →
  weeks_per_half_year = weeks_in_year / 2 →
  (miles_per_week_first_half * weeks_per_half_year) + (miles_per_week_second_half * weeks_per_half_year) = 1300 :=
by
  intros miles_per_week_first_half miles_per_week_second_half weeks_in_year weeks_per_half_year
  intros h1 h2 h3 h4
  sorry

end Nicki_total_miles_run_l24_24246


namespace find_PR_in_triangle_l24_24378

theorem find_PR_in_triangle (P Q R M : ℝ) (PQ QR PM : ℝ):
  PQ = 7 →
  QR = 10 →
  PM = 5 →
  M = (Q + R) / 2 →
  PR = Real.sqrt 149 := 
sorry

end find_PR_in_triangle_l24_24378


namespace det_matrixB_eq_neg_one_l24_24092

variable (x y : ℝ)

def matrixB : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![x, 3],
  ![-4, y]
]

theorem det_matrixB_eq_neg_one 
  (h : matrixB x y - (matrixB x y)⁻¹ = 2 • (1 : Matrix (Fin 2) (Fin 2) ℝ)) :
  Matrix.det (matrixB x y) = -1 := sorry

end det_matrixB_eq_neg_one_l24_24092


namespace cost_sum_in_WD_l24_24658

def watch_cost_loss (W : ℝ) : ℝ := 0.9 * W
def watch_cost_gain (W : ℝ) : ℝ := 1.04 * W
def bracelet_cost_gain (B : ℝ) : ℝ := 1.08 * B
def bracelet_cost_reduced_gain (B : ℝ) : ℝ := 1.02 * B

theorem cost_sum_in_WD :
  ∃ W B : ℝ, 
    watch_cost_loss W + 196 = watch_cost_gain W ∧ 
    bracelet_cost_gain B - 100 = bracelet_cost_reduced_gain B ∧ 
    (W + B / 1.5 = 2511.11) :=
sorry

end cost_sum_in_WD_l24_24658


namespace general_equation_of_curve_l24_24904

variable (θ x y : ℝ)

theorem general_equation_of_curve
  (h1 : x = Real.cos θ - 1)
  (h2 : y = Real.sin θ + 1) :
  (x + 1)^2 + (y - 1)^2 = 1 := sorry

end general_equation_of_curve_l24_24904


namespace larger_number_of_hcf_and_lcm_factors_l24_24778

theorem larger_number_of_hcf_and_lcm_factors :
  ∃ (a b : ℕ), (∀ d, d ∣ a ∧ d ∣ b → d ≤ 20) ∧ (∃ x y, x * y * 20 = a * b ∧ x * 20 = a ∧ y * 20 = b ∧ x > y ∧ x = 15 ∧ y = 11) → max a b = 300 :=
by sorry

end larger_number_of_hcf_and_lcm_factors_l24_24778


namespace pumac_grader_remainder_l24_24645

/-- A PUMaC grader is grading the submissions of forty students s₁, s₂, ..., s₄₀ for the
    individual finals round, which has three problems.
    After grading a problem of student sᵢ, the grader either:
    * grades another problem of the same student, or
    * grades the same problem of the student sᵢ₋₁ or sᵢ₊₁ (if i > 1 and i < 40, respectively).
    He grades each problem exactly once, starting with the first problem of s₁
    and ending with the third problem of s₄₀.
    Let N be the number of different orders the grader may grade the students’ problems in this way.
    Prove: N ≡ 78 [MOD 100] -/

noncomputable def grading_orders_mod : ℕ := 2 * (3 ^ 38) % 100

theorem pumac_grader_remainder :
  grading_orders_mod = 78 :=
by
  sorry

end pumac_grader_remainder_l24_24645


namespace dice_product_composite_probability_l24_24546

theorem dice_product_composite_probability :
  (let total_outcomes := (6 : ℕ)^4,
       non_composite_outcomes := 1,
       composite_probability := 1 - (non_composite_outcomes / total_outcomes) in
   composite_probability = 1295 / 1296) :=
by sorry

end dice_product_composite_probability_l24_24546


namespace range_C_8_x_l24_24569

def greatestIntLessOrEqual (x : ℝ) : ℤ := ⌊x⌋

def C_n^x (n : ℕ) (x : ℝ) : ℝ := 
  (1:ℝ) * (n:ℝ) *
  ∏ i in Finset.range (greatestIntLessOrEqual x).toNat_pred, 
      (n - i : ℝ) /
  (x * ∏ i in Finset.range (greatestIntLessOrEqual x).toNat_pred,
      (x - (i : ℝ)))

theorem range_C_8_x : 
  ∀ (x : ℝ), (x ∈ Ico (3 / 2) 3) →
  (C_n^x 8 x ∈ Set.Ioo 4 (16 / 3) ∪ Set.Ioo (28 / 3) 28) :=
sorry

end range_C_8_x_l24_24569


namespace find_a_l24_24970

theorem find_a (a : ℚ) (h : ∃ r s : ℚ, (r*x + s)^2 = ax^2 + 18*x + 16) : a = 81 / 16 := 
by sorry 

end find_a_l24_24970


namespace sum_of_b_is_negative_twelve_l24_24963

-- Conditions: the quadratic equation and its property having exactly one solution
def quadratic_equation (b : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + b * x + 6 * x + 10 = 0

-- Statement to prove: sum of the values of b is -12, 
-- given the condition that the equation has exactly one solution
theorem sum_of_b_is_negative_twelve :
  ∀ b1 b2 : ℝ, (quadratic_equation b1 ∧ quadratic_equation b2) ∧
  (∀ x : ℝ, 3 * x^2 + (b1 + 6) * x + 10 = 0 ∧ 3 * x^2 + (b2 + 6) * x + 10 = 0) ∧
  (∀ b : ℝ, b = b1 ∨ b = b2) →
  b1 + b2 = -12 :=
by
  sorry

end sum_of_b_is_negative_twelve_l24_24963


namespace probability_composite_product_l24_24530

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l24_24530


namespace christine_savings_l24_24480

def commission_rate : ℝ := 0.12
def total_sales : ℝ := 24000
def personal_needs_percentage : ℝ := 0.60
def savings_percentage : ℝ := 1 - personal_needs_percentage

noncomputable def commission_earned : ℝ := total_sales * commission_rate
noncomputable def amount_saved : ℝ := commission_earned * savings_percentage

theorem christine_savings :
  amount_saved = 1152 :=
by
  sorry

end christine_savings_l24_24480


namespace probability_of_composite_l24_24543

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l24_24543


namespace perimeter_of_octagon_l24_24881

theorem perimeter_of_octagon :
  let base := 10
  let left_side := 9
  let right_side := 11
  let top_left_diagonal := 6
  let top_right_diagonal := 7
  let small_side1 := 2
  let small_side2 := 3
  let small_side3 := 4
  base + left_side + right_side + top_left_diagonal + top_right_diagonal + small_side1 + small_side2 + small_side3 = 52 :=
by
  -- This automatically assumes all the definitions and shows the equation
  sorry

end perimeter_of_octagon_l24_24881


namespace euler_conjecture_disproof_l24_24279

theorem euler_conjecture_disproof :
    ∃ (n : ℕ), 133^4 + 110^4 + 56^4 = n^4 ∧ n = 143 :=
by {
  use 143,
  sorry
}

end euler_conjecture_disproof_l24_24279


namespace calculate_expression_l24_24292

theorem calculate_expression : 
  (1007^2 - 995^2 - 1005^2 + 997^2) = 8008 := 
by {
  sorry
}

end calculate_expression_l24_24292


namespace carol_tom_combined_weight_mildred_heavier_than_carol_tom_combined_l24_24736

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9
def tom_weight : ℕ := 20

theorem carol_tom_combined_weight :
  carol_weight + tom_weight = 29 := by
  sorry

theorem mildred_heavier_than_carol_tom_combined :
  mildred_weight - (carol_weight + tom_weight) = 30 := by
  sorry

end carol_tom_combined_weight_mildred_heavier_than_carol_tom_combined_l24_24736


namespace prove_expression_l24_24871

theorem prove_expression (a b : ℕ) 
  (h1 : 180 % 2^a = 0 ∧ 180 % 2^(a+1) ≠ 0)
  (h2 : 180 % 3^b = 0 ∧ 180 % 3^(b+1) ≠ 0) :
  (1 / 4 : ℚ)^(b - a) = 1 := 
sorry

end prove_expression_l24_24871


namespace A_lt_B_l24_24927

def A (n : ℕ) : ℕ := Nat.pow 2 (Nat.pow 2 (Nat.pow 2 (Nat.pow 2 n)))
def B (m : ℕ) : ℕ := Nat.pow 3 (Nat.pow 3 (Nat.pow 3 m))

theorem A_lt_B : A 1001 < B 1000 := 
by
  simp [A, B]
  sorry

end A_lt_B_l24_24927


namespace roots_quadratic_l24_24498

theorem roots_quadratic (a b : ℝ) 
  (h1: a^2 + 3 * a - 2010 = 0) 
  (h2: b^2 + 3 * b - 2010 = 0)
  (h_roots: a + b = -3 ∧ a * b = -2010):
  a^2 - a - 4 * b = 2022 :=
by
  sorry

end roots_quadratic_l24_24498


namespace number_of_combinations_6x6_no_two_same_row_column_l24_24463

theorem number_of_combinations_6x6_no_two_same_row_column : 
  let grid_size := 6
  let blocks_to_select := 4
  let choose := λ n k => n.choose k
  let factorial := λ n => n.factorial
  choose grid_size blocks_to_select * choose grid_size blocks_to_select * factorial blocks_to_select = 5400 :=
by
  let grid_size := 6
  let blocks_to_select := 4
  let choose := λ n k => n.choose k
  let factorial := λ n => n.factorial
  sorry

end number_of_combinations_6x6_no_two_same_row_column_l24_24463


namespace find_A_l24_24665

def clubsuit (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

theorem find_A (A : ℝ) : clubsuit A 6 = 31 → A = 10.5 :=
by
  intro h
  sorry

end find_A_l24_24665


namespace max_value_proof_l24_24875

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  x + y^3 + z^4

theorem max_value_proof
  (x y z : ℝ)
  (hx : 0 ≤ x)
  (hy : 0 ≤ y)
  (hz : 0 ≤ z)
  (h1 : x + y + z = 1)
  (h2 : x^2 + y^2 + z^2 = 1) :
  maximum_value x y z ≤ 1 :=
sorry

end max_value_proof_l24_24875


namespace Lizzy_savings_after_loan_l24_24391

theorem Lizzy_savings_after_loan :
  ∀ (initial_amount loan_amount : ℕ) (interest_percent : ℕ),
  initial_amount = 30 →
  loan_amount = 15 →
  interest_percent = 20 →
  initial_amount - loan_amount + loan_amount + loan_amount * interest_percent / 100 = 33 :=
by
  intros initial_amount loan_amount interest_percent h1 h2 h3
  sorry

end Lizzy_savings_after_loan_l24_24391


namespace fewer_cans_today_l24_24775

variable (nc_sarah_yesterday : ℕ)
variable (nc_lara_yesterday : ℕ)
variable (nc_alex_yesterday : ℕ)
variable (nc_sarah_today : ℕ)
variable (nc_lara_today : ℕ)
variable (nc_alex_today : ℕ)

-- Given conditions
def yesterday_collected_cans : Prop :=
  nc_sarah_yesterday = 50 ∧
  nc_lara_yesterday = nc_sarah_yesterday + 30 ∧
  nc_alex_yesterday = 90

def today_collected_cans : Prop :=
  nc_sarah_today = 40 ∧
  nc_lara_today = 70 ∧
  nc_alex_today = 55

theorem fewer_cans_today :
  yesterday_collected_cans nc_sarah_yesterday nc_lara_yesterday nc_alex_yesterday →
  today_collected_cans nc_sarah_today nc_lara_today nc_alex_today →
  (nc_sarah_yesterday + nc_lara_yesterday + nc_alex_yesterday) -
  (nc_sarah_today + nc_lara_today + nc_alex_today) = 55 :=
by
  intros h1 h2
  sorry

end fewer_cans_today_l24_24775


namespace certain_percentage_of_1600_l24_24936

theorem certain_percentage_of_1600 (P : ℝ) 
  (h : 0.05 * (P / 100 * 1600) = 20) : 
  P = 25 :=
by 
  sorry

end certain_percentage_of_1600_l24_24936


namespace complex_multiplication_l24_24348

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : (2 + i) * (1 - 3 * i) = 5 - 5 * i := 
by
  sorry

end complex_multiplication_l24_24348


namespace trivia_team_points_l24_24470

theorem trivia_team_points 
    (total_members : ℕ) 
    (members_absent : ℕ) 
    (total_points : ℕ) 
    (members_present : ℕ := total_members - members_absent) 
    (points_per_member : ℕ := total_points / members_present) 
    (h1 : total_members = 7) 
    (h2 : members_absent = 2) 
    (h3 : total_points = 20) : 
    points_per_member = 4 :=
by
    sorry

end trivia_team_points_l24_24470


namespace find_value_of_m_l24_24205

theorem find_value_of_m (x m : ℤ) (h₁ : x = 2) (h₂ : y = m) (h₃ : 3 * x + 2 * y = 10) : m = 2 := 
by
  sorry

end find_value_of_m_l24_24205


namespace simplified_value_of_sum_l24_24923

theorem simplified_value_of_sum :
  (-1)^(2004) + (-1)^(2005) + 1^(2006) - 1^(2007) = -2 := by
  sorry

end simplified_value_of_sum_l24_24923


namespace journey_time_l24_24942

theorem journey_time :
  let total_distance := 224
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 21
  let speed_second_half := 24
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  total_time = 10 := by
  have first_half_distance_eq : first_half_distance = 112 := by norm_num
  have second_half_distance_eq : second_half_distance = 112 := by norm_num
  have time_first_half_eq : time_first_half = (112 / 21) := rfl
  have time_second_half_eq : time_second_half = (112 / 24) := rfl
  have total_time_eq : total_time = (112 / 21) + (112 / 24) := rfl
  have fraction_sum_eq : (112 / 21) + (112 / 24) = 10 := by norm_num
  show total_time = 10 from fraction_sum_eq

end journey_time_l24_24942


namespace age_difference_l24_24646

-- Define the hypothesis and statement
theorem age_difference (A B C : ℕ) 
  (h1 : A + B = B + C + 15)
  (h2 : C = A - 15) : 
  (A + B) - (B + C) = 15 :=
by
  sorry

end age_difference_l24_24646


namespace sum_of_first_10_common_elements_is_correct_l24_24975

-- Define arithmetic progression
def a (n : ℕ) : ℕ := 4 + 3 * n

-- Define geometric progression
def b (k : ℕ) : ℕ := 20 * (2 ^ k)

-- Define the sum of the first 10 common elements in both sequences
def sum_first_10_common_elements : ℕ := 13981000

-- Statement of the proof problem in Lean 4
theorem sum_of_first_10_common_elements_is_correct :
  ∑ i in (finset.range 10).image (λ k, b(2*k + 1)), id = sum_first_10_common_elements :=
by
  -- Proof omitted
  sorry

end sum_of_first_10_common_elements_is_correct_l24_24975


namespace solution_set_inequality_range_of_m_l24_24505

def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 1| - |x + 1|

-- Problem 1
theorem solution_set_inequality (x : ℝ) : 
  (f x 5 > 2) ↔ (-3 / 2 < x ∧ x < 3 / 2) :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (x^2 + 2 * x + 3) ∧ y = f x m) ↔ (m ≥ 4) :=
sorry

end solution_set_inequality_range_of_m_l24_24505


namespace most_colored_pencils_l24_24274

theorem most_colored_pencils (total red blue yellow : ℕ) 
  (h_total : total = 24)
  (h_red : red = total / 4)
  (h_blue : blue = red + 6)
  (h_yellow : yellow = total - (red + blue)) :
  blue = 12 :=
by
  sorry

end most_colored_pencils_l24_24274


namespace range_of_m_l24_24557

theorem range_of_m (m : ℝ) : (∀ x, 0 ≤ x ∧ x ≤ m → -6 ≤ x^2 - 4 * x - 2 ∧ x^2 - 4 * x - 2 ≤ -2) → 2 ≤ m ∧ m ≤ 4 :=
by
  sorry

end range_of_m_l24_24557


namespace math_problem_l24_24173

theorem math_problem :
    3 * 3^4 - (27 ^ 63 / 27 ^ 61) = -486 :=
by
  sorry

end math_problem_l24_24173


namespace Jill_llamas_count_l24_24721

theorem Jill_llamas_count :
  let initial_pregnant_with_one_calf := 9
  let initial_pregnant_with_twins := 5
  let total_calves_born := (initial_pregnant_with_one_calf * 1) + (initial_pregnant_with_twins * 2)
  let calves_after_trade := total_calves_born - 8
  let initial_pregnant_lamas := initial_pregnant_with_one_calf + initial_pregnant_with_twins
  let total_lamas_after_birth := initial_pregnant_lamas + total_calves_born
  let lamas_after_trade := total_lamas_after_birth - 8 + 2
  let lamas_sold := lamas_after_trade / 3
  let final_lamas := lamas_after_trade - lamas_sold
  final_lamas = 18 :=
by
  sorry

end Jill_llamas_count_l24_24721


namespace gift_certificate_value_is_correct_l24_24269

-- Define the conditions
def total_race_time_minutes : ℕ := 12
def one_lap_meters : ℕ := 100
def total_laps : ℕ := 24
def earning_rate_per_minute : ℕ := 7

-- The total distance run in meters
def total_distance_meters : ℕ := total_laps * one_lap_meters

-- The total earnings in dollars
def total_earnings_dollars : ℕ := earning_rate_per_minute * total_race_time_minutes

-- The worth of the gift certificate per 100 meters (to be proven as 3.50 dollars)
def gift_certificate_value : ℚ := total_earnings_dollars / (total_distance_meters / one_lap_meters)

-- Prove that the gift certificate value is $3.50
theorem gift_certificate_value_is_correct : 
    gift_certificate_value = 3.5 := by
  sorry

end gift_certificate_value_is_correct_l24_24269


namespace soccer_team_games_l24_24165

theorem soccer_team_games :
  ∃ G : ℕ, G % 2 = 0 ∧ 
           45 / 100 * 36 = 16 ∧ 
           ∀ R, R = G - 36 → (16 + 75 / 100 * R) = 62 / 100 * G ∧
           G = 84 :=
sorry

end soccer_team_games_l24_24165


namespace find_speed_of_stream_l24_24641

variable (b s : ℝ)

-- Equation derived from downstream condition
def downstream_equation := b + s = 24

-- Equation derived from upstream condition
def upstream_equation := b - s = 10

theorem find_speed_of_stream
  (b s : ℝ)
  (h1 : downstream_equation b s)
  (h2 : upstream_equation b s) :
  s = 7 := by
  -- placeholder for the proof
  sorry

end find_speed_of_stream_l24_24641


namespace cost_price_A_min_cost_bshelves_l24_24656

-- Define the cost price of type B bookshelf
def costB_bshelf : ℝ := 300

-- Define the cost price of type A bookshelf
def costA_bshelf : ℝ := 1.2 * costB_bshelf

-- Define the total number of bookshelves
def total_bshelves : ℕ := 60

-- Define the condition for type A and type B bookshelves count
def typeBshelves := λ (typeAshelves : ℕ) => total_bshelves - typeAshelves
def typeBshelves_constraints := λ (typeAshelves : ℕ) => total_bshelves - typeAshelves ≤ 2 * typeAshelves

-- Define the equation for the costs
noncomputable def total_cost (typeAshelves : ℕ) : ℝ :=
  360 * typeAshelves + 300 * (total_bshelves - typeAshelves)

-- Define the goal: cost price of type A bookshelf is 360 yuan
theorem cost_price_A : costA_bshelf = 360 :=
by 
  sorry

-- Define the goal: the school should buy 20 type A bookshelves and 40 type B bookshelves to minimize cost
theorem min_cost_bshelves : ∃ typeAshelves : ℕ, typeAshelves = 20 ∧ typeBshelves typeAshelves = 40 :=
by
  sorry

end cost_price_A_min_cost_bshelves_l24_24656


namespace at_least_one_bigger_than_44_9_l24_24429

noncomputable def x : ℕ → ℝ := sorry
noncomputable def y : ℕ → ℝ := sorry

axiom x_positive (n : ℕ) : 0 < x n
axiom y_positive (n : ℕ) : 0 < y n
axiom recurrence_x (n : ℕ) : x (n + 1) = x n + 1 / (2 * y n)
axiom recurrence_y (n : ℕ) : y (n + 1) = y n + 1 / (2 * x n)

theorem at_least_one_bigger_than_44_9 : x 2018 > 44.9 ∨ y 2018 > 44.9 :=
sorry

end at_least_one_bigger_than_44_9_l24_24429


namespace function_passes_through_fixed_point_l24_24685

noncomputable def f (a x : ℝ) := a^(x+1) - 1

theorem function_passes_through_fixed_point (a : ℝ) (h_pos : 0 < a) (h_not_one : a ≠ 1) :
  f a (-1) = 0 := by
  sorry

end function_passes_through_fixed_point_l24_24685


namespace sufficient_condition_not_necessary_condition_l24_24855

/--
\(a > 1\) is a sufficient but not necessary condition for \(\frac{1}{a} < 1\).
-/
theorem sufficient_condition (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by
  sorry

theorem not_necessary_condition (a : ℝ) (h : 1 / a < 1) : a > 1 ∨ a < 0 :=
by
  sorry

end sufficient_condition_not_necessary_condition_l24_24855


namespace probability_of_composite_l24_24542

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l24_24542


namespace find_multiple_l24_24188

theorem find_multiple (n m : ℕ) (h_n : n = 5) (h_ineq : (m * n - 15) > 2 * n) : m = 6 := 
by {
  sorry
}

end find_multiple_l24_24188


namespace clock_strikes_l24_24939

theorem clock_strikes (t n : ℕ) (h_t : 13 * t = 26) (h_n : 2 * n - 1 * t = 22) : n = 6 :=
by
  sorry

end clock_strikes_l24_24939


namespace percentage_alcohol_final_l24_24575

-- Let's define the given conditions
variable (A B totalVolume : ℝ)
variable (percentAlcoholA percentAlcoholB : ℝ)
variable (approxA : ℝ)

-- Assume the conditions
axiom condition1 : percentAlcoholA = 0.20
axiom condition2 : percentAlcoholB = 0.50
axiom condition3 : totalVolume = 15
axiom condition4 : approxA = 10
axiom condition5 : A = approxA
axiom condition6 : B = totalVolume - A

-- The proof statement
theorem percentage_alcohol_final : 
  (0.20 * A + 0.50 * B) / 15 * 100 = 30 :=
by 
  -- Introduce enough structure for Lean to handle the problem.
  sorry

end percentage_alcohol_final_l24_24575


namespace circle_reflection_l24_24589

variable (x₀ y₀ : ℝ)

def reflect_over_line_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem circle_reflection 
  (h₁ : x₀ = 8)
  (h₂ : y₀ = -3) :
  reflect_over_line_y_eq_neg_x (x₀, y₀) = (3, -8) := by
  sorry

end circle_reflection_l24_24589


namespace cube_root_less_than_five_count_l24_24851

theorem cube_root_less_than_five_count :
  (∃ n : ℕ, n = 124 ∧ ∀ x : ℕ, 1 ≤ x → x < 5^3 → x < 125) := 
sorry

end cube_root_less_than_five_count_l24_24851


namespace graph_of_direct_proportion_is_line_l24_24010

-- Define the direct proportion function
def direct_proportion (k : ℝ) (x : ℝ) : ℝ :=
  k * x

-- State the theorem to prove the graph of this function is a straight line
theorem graph_of_direct_proportion_is_line (k : ℝ) :
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, direct_proportion k x = a * x + b ∧ b = 0 := 
by 
  sorry

end graph_of_direct_proportion_is_line_l24_24010


namespace faster_train_speed_l24_24764

theorem faster_train_speed (v : ℝ) (h_total_length : 100 + 100 = 200) 
  (h_cross_time : 8 = 8) (h_speeds : 3 * v = 200 / 8) : 2 * v = 50 / 3 :=
sorry

end faster_train_speed_l24_24764


namespace binomial_12_6_eq_924_l24_24812

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := sorry

end binomial_12_6_eq_924_l24_24812


namespace binomial_12_6_eq_1848_l24_24814

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end binomial_12_6_eq_1848_l24_24814


namespace num_real_satisfying_x_l24_24512

theorem num_real_satisfying_x : 
  (∃ (x : ℝ), ∃ (s : ℕ) (hs : s ∈ set.Ico 0 (floor (sqrt 123) + 1)), 
   s = int.of_nat (floor (sqrt (123 - sqrt x)))) →
   12 := sorry

end num_real_satisfying_x_l24_24512


namespace find_a_l24_24969

theorem find_a (a : ℚ) (h : ∃ r s : ℚ, (r*x + s)^2 = ax^2 + 18*x + 16) : a = 81 / 16 := 
by sorry 

end find_a_l24_24969


namespace area_rhombus_abs_eq_9_l24_24766

open Real

theorem area_rhombus_abs_eq_9 :
  (∃ d1 d2 : ℝ, (d1 = 18) ∧ (d2 = 6) ∧ (1/2 * d1 * d2 = 54)) :=
begin
  use 18,
  use 6,
  split,
  { refl },
  split,
  { refl },
  { norm_num }
end

end area_rhombus_abs_eq_9_l24_24766


namespace average_percent_increase_per_year_l24_24120

def initial_population : ℕ := 175000
def final_population : ℕ := 262500
def years : ℕ := 10

theorem average_percent_increase_per_year :
  ( ( ( ( final_population - initial_population ) / years : ℝ ) / initial_population ) * 100 ) = 5 := by
  sorry

end average_percent_increase_per_year_l24_24120


namespace find_coefficients_l24_24329

theorem find_coefficients (a1 a2 : ℚ) :
  (4 * a1 + 5 * a2 = 9) ∧ (-a1 + 3 * a2 = 4) ↔ (a1 = 181 / 136) ∧ (a2 = 25 / 68) := 
sorry

end find_coefficients_l24_24329


namespace value_of_x_add_y_l24_24684

theorem value_of_x_add_y (x y : ℝ) 
  (h1 : x + Real.sin y = 2023)
  (h2 : x + 2023 * Real.cos y = 2021)
  (h3 : (Real.pi / 4) ≤ y ∧ y ≤ (3 * Real.pi / 4)) : 
  x + y = 2023 - (Real.sqrt 2) / 2 + (3 * Real.pi) / 4 := 
sorry

end value_of_x_add_y_l24_24684


namespace exists_m_sqrt_8m_integer_l24_24405

theorem exists_m_sqrt_8m_integer : ∃ (m : ℕ), (m > 0) ∧ (∃ k : ℕ, k^2 = 8 * m) :=
by
  use 2
  split
  · exact Nat.succ_pos 1
  · use 4
    exact Nat.succ_pos 1
    sorry

end exists_m_sqrt_8m_integer_l24_24405


namespace total_goals_proof_l24_24669

-- Definitions based on the conditions
def first_half_team_a := 8
def first_half_team_b := first_half_team_a / 2
def first_half_team_c := first_half_team_b * 2

def second_half_team_a := first_half_team_c
def second_half_team_b := first_half_team_a
def second_half_team_c := second_half_team_b + 3

-- Total scores for each team
def total_team_a := first_half_team_a + second_half_team_a
def total_team_b := first_half_team_b + second_half_team_b
def total_team_c := first_half_team_c + second_half_team_c

-- Total goals for all teams
def total_goals := total_team_a + total_team_b + total_team_c

-- The theorem to be proved
theorem total_goals_proof : total_goals = 47 := by
  sorry

end total_goals_proof_l24_24669


namespace recreation_percentage_correct_l24_24282

noncomputable def recreation_percentage (W : ℝ) : ℝ :=
  let recreation_two_weeks_ago := 0.25 * W
  let wages_last_week := 0.95 * W
  let recreation_last_week := 0.35 * (0.95 * W)
  let wages_this_week := 0.95 * W * 0.85
  let recreation_this_week := 0.45 * (0.95 * W * 0.85)
  (recreation_this_week / recreation_two_weeks_ago) * 100

theorem recreation_percentage_correct (W : ℝ) : recreation_percentage W = 145.35 :=
by
  sorry

end recreation_percentage_correct_l24_24282


namespace gcd_180_270_eq_90_l24_24022

-- Problem Statement
theorem gcd_180_270_eq_90 : Nat.gcd 180 270 = 90 := 
by 
  sorry

end gcd_180_270_eq_90_l24_24022


namespace f_1001_value_l24_24755

noncomputable def f : ℕ → ℝ := sorry

theorem f_1001_value :
  (∀ a b n : ℕ, a + b = 2^n → f a + f b = n^2) →
  f 1 = 1 →
  f 1001 = 83 :=
by
  intro h₁ h₂
  sorry

end f_1001_value_l24_24755


namespace union_set_when_m_neg3_range_of_m_for_intersection_l24_24991

def setA (x : ℝ) : Prop := x^2 - x - 12 ≤ 0
def setB (x m : ℝ) : Prop := 2*m - 1 ≤ x ∧ x ≤ m + 1

theorem union_set_when_m_neg3 : 
  (∀ x, setA x ∨ setB x (-3) ↔ -7 ≤ x ∧ x ≤ 4) := 
by sorry

theorem range_of_m_for_intersection :
  (∀ m x, (setA x ∧ setB x m ↔ setB x m) → m ≥ -1) := 
by sorry

end union_set_when_m_neg3_range_of_m_for_intersection_l24_24991


namespace min_commission_deputies_l24_24433

theorem min_commission_deputies 
  (members : ℕ) 
  (brawls : ℕ) 
  (brawl_participants : brawls = 200) 
  (member_count : members = 200) :
  ∃ minimal_commission_members : ℕ, minimal_commission_members = 67 := 
sorry

end min_commission_deputies_l24_24433


namespace will_jogged_for_30_minutes_l24_24928

theorem will_jogged_for_30_minutes 
  (calories_before : ℕ)
  (calories_per_minute : ℕ)
  (net_calories_after : ℕ)
  (h1 : calories_before = 900)
  (h2 : calories_per_minute = 10)
  (h3 : net_calories_after = 600) :
  let calories_burned := calories_before - net_calories_after
  let jogging_time := calories_burned / calories_per_minute
  jogging_time = 30 := by
  sorry

end will_jogged_for_30_minutes_l24_24928


namespace fraction_addition_l24_24132

theorem fraction_addition :
  (1 / 3 * 2 / 5) + 1 / 4 = 23 / 60 := 
  sorry

end fraction_addition_l24_24132


namespace polynomial_expansion_proof_l24_24486

variable (z : ℤ)

-- Define the polynomials p and q
noncomputable def p (z : ℤ) : ℤ := 3 * z^2 - 4 * z + 1
noncomputable def q (z : ℤ) : ℤ := 2 * z^3 + 3 * z^2 - 5 * z + 2

-- Define the expanded polynomial
noncomputable def expanded (z : ℤ) : ℤ :=
  6 * z^5 + z^4 - 25 * z^3 + 29 * z^2 - 13 * z + 2

-- The goal is to prove the equivalence of (p * q) == expanded 
theorem polynomial_expansion_proof :
  (p z) * (q z) = expanded z :=
by
  sorry

end polynomial_expansion_proof_l24_24486


namespace find_a_l24_24967

theorem find_a (r s a : ℚ) (h₁ : 2 * r * s = 18) (h₂ : s^2 = 16) (h₃ : a = r^2) : 
  a = 81 / 16 := 
sorry

end find_a_l24_24967


namespace find_13th_result_l24_24147

theorem find_13th_result (avg25 : ℕ) (avg12_first : ℕ) (avg12_last : ℕ)
  (h_avg25 : avg25 = 18) (h_avg12_first : avg12_first = 10) (h_avg12_last : avg12_last = 20) :
  ∃ r13 : ℕ, r13 = 90 := by
  sorry

end find_13th_result_l24_24147


namespace graph_does_not_pass_through_second_quadrant_l24_24263

theorem graph_does_not_pass_through_second_quadrant :
  ¬ ∃ x : ℝ, x < 0 ∧ 2 * x - 3 > 0 :=
by
  -- Include the necessary steps to complete the proof, but for now we provide a placeholder:
  sorry

end graph_does_not_pass_through_second_quadrant_l24_24263


namespace ilya_incorrect_l24_24739

theorem ilya_incorrect (s t : ℝ) : ¬ (s + t = s * t ∧ s * t = s / t) :=
by
  sorry

end ilya_incorrect_l24_24739


namespace value_of_fraction_l24_24077

variables {a b c : ℝ}

-- Conditions
def quadratic_has_no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

def person_A_roots (a' b c : ℝ) : Prop :=
  b = -6 * a' ∧ c = 8 * a'

def person_B_roots (a b' c : ℝ) : Prop :=
  b' = -3 * a ∧ c = -4 * a

-- Proof Statement
theorem value_of_fraction (a b c a' b' : ℝ)
  (hnr : quadratic_has_no_real_roots a b c)
  (hA : person_A_roots a' b c)
  (hB : person_B_roots a b' c) :
  (2 * b + 3 * c) / a = 6 :=
by
  sorry

end value_of_fraction_l24_24077


namespace marcy_fewer_tickets_l24_24435

theorem marcy_fewer_tickets (A M : ℕ) (h1 : A = 26) (h2 : M = 5 * A) (h3 : A + M = 150) : M - A = 104 :=
by
  sorry

end marcy_fewer_tickets_l24_24435


namespace geometric_series_common_ratio_l24_24431

theorem geometric_series_common_ratio (a S r : ℝ) 
  (hS : S = a / (1 - r)) 
  (h_modified : (a * r^2) / (1 - r) = S / 16) : 
  r = 1/4 ∨ r = -1/4 :=
by
  sorry

end geometric_series_common_ratio_l24_24431


namespace capacity_of_other_bottle_l24_24157

theorem capacity_of_other_bottle 
  (total_milk : ℕ) (capacity_bottle_one : ℕ) (fraction_filled_other_bottle : ℚ)
  (equal_fraction : ℚ) (other_bottle_milk : ℚ) (capacity_other_bottle : ℚ) : 
  total_milk = 8 ∧ capacity_bottle_one = 4 ∧ other_bottle_milk = 16/3 ∧ 
  (equal_fraction * capacity_bottle_one + equal_fraction * capacity_other_bottle = total_milk) ∧ 
  (fraction_filled_other_bottle = 5.333333333333333) → capacity_other_bottle = 8 :=
by
  intro h
  sorry

end capacity_of_other_bottle_l24_24157


namespace stratified_sampling_girls_count_l24_24653

theorem stratified_sampling_girls_count :
  (boys girls sampleSize totalSample : ℕ) →
  boys = 36 →
  girls = 18 →
  sampleSize = 6 →
  totalSample = boys + girls →
  (sampleSize * girls) / totalSample = 2 :=
by
  intros boys girls sampleSize totalSample h_boys h_girls h_sampleSize h_totalSample
  sorry

end stratified_sampling_girls_count_l24_24653


namespace compute_value_l24_24363

noncomputable def repeating_decimal_31 : ℝ := 31 / 100000
noncomputable def repeating_decimal_47 : ℝ := 47 / 100000
def term : ℝ := 10^5 - 10^3

theorem compute_value : (term * repeating_decimal_31 + term * repeating_decimal_47) = 77.22 := 
by
  sorry

end compute_value_l24_24363


namespace simplify_expression_l24_24413

theorem simplify_expression : 2 - 2 / (2 + Real.sqrt 5) + 2 / (2 - Real.sqrt 5) = 2 + 4 * Real.sqrt 5 :=
by sorry

end simplify_expression_l24_24413


namespace andrew_age_l24_24660

variable (a g s : ℝ)

theorem andrew_age :
  g = 10 * a ∧ g - s = a + 45 ∧ s = 5 → a = 50 / 9 := by
  sorry

end andrew_age_l24_24660


namespace find_a_b_solution_set_l24_24983

-- Given function
def f (x : ℝ) (a b : ℝ) := x^2 - (a + b) * x + 3 * a

-- Part 1: Prove the values of a and b given the solution set of the inequality
theorem find_a_b (a b : ℝ) 
  (h1 : 1^2 - (a + b) * 1 + 3 * 1 = 0)
  (h2 : 3^2 - (a + b) * 3 + 3 * 1 = 0) :
  a = 1 ∧ b = 3 :=
sorry

-- Part 2: Find the solution set of the inequality f(x) > 0 given b = 3
theorem solution_set (a : ℝ)
  (h : b = 3) :
  (a > 3 → (∀ x, f x a 3 > 0 ↔ x < 3 ∨ x > a)) ∧
  (a < 3 → (∀ x, f x a 3 > 0 ↔ x < a ∨ x > 3)) ∧
  (a = 3 → (∀ x, f x a 3 > 0 ↔ x ≠ 3)) :=
sorry

end find_a_b_solution_set_l24_24983


namespace geometric_sequence_arithmetic_l24_24202

theorem geometric_sequence_arithmetic (a₁ q : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) 
  (h2 : 2 * S 6 = S 3 + S 9) : 
  q^3 = -1 := 
sorry

end geometric_sequence_arithmetic_l24_24202


namespace find_radius_l24_24271

noncomputable def square_radius (r : ℝ) : Prop :=
  let s := (2 * r) / Real.sqrt 2  -- side length of the square derived from the radius
  let perimeter := 4 * s         -- perimeter of the square
  let area := Real.pi * r^2      -- area of the circumscribed circle
  perimeter = area               -- given condition

theorem find_radius (r : ℝ) (h : square_radius r) : r = (4 * Real.sqrt 2) / Real.pi :=
by
  sorry

end find_radius_l24_24271


namespace solve_quadratic_l24_24145

theorem solve_quadratic : ∀ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 → x = 1 :=
by
  intros x h
  sorry

end solve_quadratic_l24_24145


namespace hamiltonian_cycle_with_one_switch_l24_24273

open Finset Function SimpleGraph

theorem hamiltonian_cycle_with_one_switch (n : ℕ) (h : 3 ≤ n):
  ∃ (G : SimpleGraph (Fin n)) (c : G.Edge → Bool), 
  (G.isHamiltonian ∧ ∃ (cycle : Finset (Fin n)), 
  G.adj_circuit cycle ∧ (∀ e₁ e₂ ∈ cycle.edges, 
  c e₁ = c e₂ ∨ c e₁ ≠ c e₂)) := sorry

end hamiltonian_cycle_with_one_switch_l24_24273


namespace proof_problem_l24_24882

theorem proof_problem (x : ℝ) 
    (h1 : (x - 1) * (x + 1) = x^2 - 1)
    (h2 : (x - 1) * (x^2 + x + 1) = x^3 - 1)
    (h3 : (x - 1) * (x^3 + x^2 + x + 1) = x^4 - 1)
    (h4 : (x - 1) * (x^4 + x^3 + x^2 + x + 1) = -2) :
    x^2023 = -1 := 
by 
  sorry -- Proof is omitted

end proof_problem_l24_24882


namespace false_props_l24_24503

-- Definitions for conditions
def prop1 :=
  ∀ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ (a * d = b * c) → 
  (a / b = b / c ∧ b / c = c / d)

def prop2 :=
  ∀ (a : ℕ), (∃ k : ℕ, a = 2 * k) → (a % 2 = 0)

def prop3 :=
  ∀ (A : ℝ), (A > 30) → (Real.sin (A * Real.pi / 180) > 1 / 2)

-- Theorem statement
theorem false_props : (¬ prop1) ∧ (¬ prop3) :=
by sorry

end false_props_l24_24503


namespace new_member_money_l24_24587

variable (T M : ℝ)
variable (H1 : T / 7 = 20)
variable (H2 : (T + M) / 8 = 14)

theorem new_member_money : M = 756 :=
by
  sorry

end new_member_money_l24_24587


namespace find_base_side_length_l24_24601

-- Regular triangular pyramid properties and derived values
variables
  (a l h : ℝ) -- side length of the base, slant height, and height of the pyramid
  (V : ℝ) -- volume of the pyramid

-- Given conditions
def inclined_to_base_plane_at_angle (angle : ℝ) := angle = 45
def volume_of_pyramid (V : ℝ) := V = 18

-- Prove the side length of the base
theorem find_base_side_length
  (h_eq : h = a * Real.sqrt 3 / 3)
  (volume_eq : V = 1 / 3 * (a * a * Real.sqrt 3 / 4) * h)
  (volume_given : V = 18) :
  a = 6 := by
  sorry

end find_base_side_length_l24_24601


namespace multiplication_digit_sum_l24_24229

theorem multiplication_digit_sum :
  let a := 879
  let b := 492
  let product := a * b
  let sum_of_digits := (4 + 3 + 2 + 4 + 6 + 8)
  product = 432468 ∧ sum_of_digits = 27 := by
  -- Step 1: Set up the given numbers
  let a := 879
  let b := 492

  -- Step 2: Calculate the product
  let product := a * b
  have product_eq : product = 432468 := by
    sorry

  -- Step 3: Sum the digits of the product
  let sum_of_digits := (4 + 3 + 2 + 4 + 6 + 8)
  have sum_of_digits_eq : sum_of_digits = 27 := by
    sorry

  -- Conclusion
  exact ⟨product_eq, sum_of_digits_eq⟩

end multiplication_digit_sum_l24_24229


namespace area_of_kite_l24_24980

theorem area_of_kite (A B C D : ℝ × ℝ) (hA : A = (2, 3)) (hB : B = (6, 7)) (hC : C = (10, 3)) (hD : D = (6, 0)) : 
  let base := (C.1 - A.1)
  let height := (B.2 - D.2)
  let area := 2 * (1 / 2 * base * height)
  area = 56 := 
by
  sorry

end area_of_kite_l24_24980


namespace radius_scientific_notation_l24_24012

theorem radius_scientific_notation :
  696000 = 6.96 * 10^5 :=
sorry

end radius_scientific_notation_l24_24012


namespace nat_number_36_sum_of_digits_l24_24192

-- Define the function that represents the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main statement
theorem nat_number_36_sum_of_digits (x : ℕ) (hx : x = 36 * sum_of_digits x) : x = 324 ∨ x = 648 := 
by 
  sorry

end nat_number_36_sum_of_digits_l24_24192


namespace triangle_angle_R_measure_l24_24232

theorem triangle_angle_R_measure :
  ∀ (P Q R : ℝ),
  P + Q + R = 180 ∧ P = 70 ∧ Q = 2 * R + 15 → R = 95 / 3 :=
by
  intros P Q R h
  sorry

end triangle_angle_R_measure_l24_24232


namespace fraction_of_recipe_l24_24163

theorem fraction_of_recipe 
  (recipe_sugar recipe_milk recipe_flour : ℚ)
  (have_sugar have_milk have_flour : ℚ)
  (h1 : recipe_sugar = 3/4) (h2 : recipe_milk = 2/3) (h3 : recipe_flour = 3/8)
  (h4 : have_sugar = 2/4) (h5 : have_milk = 1/2) (h6 : have_flour = 1/4) : 
  (min ((have_sugar / recipe_sugar)) (min ((have_milk / recipe_milk)) (have_flour / recipe_flour)) = 2/3) := 
by sorry

end fraction_of_recipe_l24_24163


namespace secondary_spermatocytes_can_contain_two_y_chromosomes_l24_24376

-- Definitions corresponding to the conditions
def primary_spermatocytes_first_meiotic_division_contains_y (n : Nat) : Prop := n = 1
def spermatogonia_metaphase_mitosis_contains_y (n : Nat) : Prop := n = 1
def secondary_spermatocytes_second_meiotic_division_contains_y (n : Nat) : Prop := n = 0 ∨ n = 2
def spermatogonia_prophase_mitosis_contains_y (n : Nat) : Prop := n = 1

-- The theorem statement equivalent to the given math problem
theorem secondary_spermatocytes_can_contain_two_y_chromosomes :
  ∃ n, (secondary_spermatocytes_second_meiotic_division_contains_y n ∧ n = 2) :=
sorry

end secondary_spermatocytes_can_contain_two_y_chromosomes_l24_24376


namespace binomial_12_6_eq_1848_l24_24817

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end binomial_12_6_eq_1848_l24_24817


namespace common_ratio_neg_two_l24_24905

theorem common_ratio_neg_two (a : ℕ → ℝ) (q : ℝ) 
  (h : ∀ n, a (n + 1) = a n * q)
  (H : 8 * a 2 + a 5 = 0) : 
  q = -2 :=
sorry

end common_ratio_neg_two_l24_24905


namespace reflection_of_point_l24_24600

def reflect_across_y_neg_x (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (y, x)

theorem reflection_of_point
  (P : ℝ × ℝ)
  (h : P = (8, -3)) :
  reflect_across_y_neg_x P = (3, -8) :=
by
  rw [h]
  sorry

end reflection_of_point_l24_24600


namespace dice_product_composite_probability_l24_24538

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l24_24538


namespace simplify_expression_l24_24823

theorem simplify_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3 * x^2 + 2 * x) / ((x - 1) * (x + 3)) - (5 * x + 3) / ((x - 1) * (x + 3))
  = 3 * (x^2 - x - 1) / ((x - 1) * (x + 3)) :=
by
  sorry

end simplify_expression_l24_24823


namespace algebraic_expression_value_l24_24136

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a + 2 * b = 2) : 
  (-a * (-2) ^ 2 + b * (-2) + 1) = -1 :=
by
  sorry

end algebraic_expression_value_l24_24136


namespace valid_shirt_tie_combinations_l24_24899

theorem valid_shirt_tie_combinations
  (num_shirts : ℕ)
  (num_ties : ℕ)
  (restricted_shirts : ℕ)
  (restricted_ties : ℕ)
  (h_shirts : num_shirts = 8)
  (h_ties : num_ties = 7)
  (h_restricted_shirts : restricted_shirts = 3)
  (h_restricted_ties : restricted_ties = 2) :
  num_shirts * num_ties - restricted_shirts * restricted_ties = 50 := by
  sorry

end valid_shirt_tie_combinations_l24_24899


namespace product_of_y_values_l24_24485

theorem product_of_y_values (y : ℝ) (h : abs (2 * y * 3) + 5 = 47) :
  ∃ y1 y2, (abs (2 * y1 * 3) + 5 = 47) ∧ (abs (2 * y2 * 3) + 5 = 47) ∧ y1 * y2 = -49 :=
by 
  sorry

end product_of_y_values_l24_24485


namespace binomial_coefficient_ratio_l24_24112

theorem binomial_coefficient_ratio (n k : ℕ) (h₁ : n = 4 * k + 3) (h₂ : n = 3 * k + 5) : n + k = 13 :=
by
  sorry

end binomial_coefficient_ratio_l24_24112


namespace value_of_expression_l24_24635

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 7) : 3*x^2 + 9*x - 2 = 4 :=
by
  -- The proof will be filled here; it's currently skipped using 'sorry'
  sorry

end value_of_expression_l24_24635


namespace valid_parameterizations_l24_24114

noncomputable def is_scalar_multiple (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def lies_on_line (p : ℝ × ℝ) (m b : ℝ) : Prop :=
  p.2 = m * p.1 + b

def is_valid_parameterization (p d : ℝ × ℝ) (m b : ℝ) : Prop :=
  lies_on_line p m b ∧ is_scalar_multiple d (2, 1)

theorem valid_parameterizations :
  (is_valid_parameterization (7, 18) (-1, -2) 2 4) ∧
  (is_valid_parameterization (1, 6) (5, 10) 2 4) ∧
  (is_valid_parameterization (2, 8) (20, 40) 2 4) ∧
  ¬ (is_valid_parameterization (-4, -4) (1, -1) 2 4) ∧
  ¬ (is_valid_parameterization (-3, -2) (0.5, 1) 2 4) :=
by {
  sorry
}

end valid_parameterizations_l24_24114


namespace correct_option_C_l24_24507

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the complements of sets A and B in U
def complA : Set ℕ := {2, 4}
def complB : Set ℕ := {3, 4}

-- Define sets A and B using the complements
def A : Set ℕ := U \ complA
def B : Set ℕ := U \ complB

-- Mathematical proof problem statement
theorem correct_option_C : 3 ∈ A ∧ 3 ∉ B := by
  sorry

end correct_option_C_l24_24507


namespace ratio_of_expenditures_l24_24427

variable (Rajan_income Balan_income Rajan_expenditure Balan_expenditure Rajan_savings Balan_savings: ℤ)
variable (ratio_incomes: ℚ)
variable (savings_amount: ℤ)

-- Given conditions
def conditions : Prop :=
  Rajan_income = 7000 ∧
  ratio_incomes = 7 / 6 ∧
  savings_amount = 1000 ∧
  Rajan_savings = Rajan_income - Rajan_expenditure ∧
  Balan_savings = Balan_income - Balan_expenditure ∧
  Rajan_savings = savings_amount ∧
  Balan_savings = savings_amount

-- The theorem we want to prove
theorem ratio_of_expenditures :
  conditions Rajan_income Balan_income Rajan_expenditure Balan_expenditure Rajan_savings Balan_savings ratio_incomes savings_amount →
  (Rajan_expenditure : ℚ) / (Balan_expenditure : ℚ) = 6 / 5 :=
by
  sorry

end ratio_of_expenditures_l24_24427


namespace solution_set_of_inequality_l24_24123

theorem solution_set_of_inequality (x m : ℝ) : 
  (x^2 - (2 * m + 1) * x + m^2 + m < 0) ↔ m < x ∧ x < m + 1 := 
by
  sorry

end solution_set_of_inequality_l24_24123


namespace cab_time_l24_24938

theorem cab_time (d t : ℝ) (v : ℝ := d / t)
    (v1 : ℝ := (5 / 6) * v)
    (t1 : ℝ := d / v1)
    (v2 : ℝ := (2 / 3) * v)
    (t2 : ℝ := d / v2)
    (T : ℝ := t1 + t2)
    (delay : ℝ := 5) :
    let total_time := 2 * t + delay
    t * d ≠ 0 → T = total_time → t = 50 / 7 := by
    sorry

end cab_time_l24_24938


namespace find_x_l24_24781

theorem find_x (x : ℝ) : 9 - (x / (1 / 3)) + 3 = 3 → x = 3 := by
  intro h
  sorry

end find_x_l24_24781


namespace sum_interior_numbers_eighth_row_of_pascals_triangle_l24_24027

theorem sum_interior_numbers_eighth_row_of_pascals_triangle :
  let n := 8
  let sum_all_elements := 2 ^ (n - 1)
  let sum_interior_numbers := sum_all_elements - 2
  sum_interior_numbers = 126 :=
by
  let n := 8
  let sum_all_elements := 2 ^ (n - 1)
  let sum_interior_numbers := sum_all_elements - 2
  show sum_interior_numbers = 126
  sorry

end sum_interior_numbers_eighth_row_of_pascals_triangle_l24_24027


namespace johann_oranges_problem_l24_24567

theorem johann_oranges_problem :
  ∀ (initial_oranges johann_ate half_stolen carson_returned : ℕ),
  initial_oranges = 60 →
  johann_ate = 10 →
  half_stolen = (initial_oranges - johann_ate) / 2 →
  carson_returned = 5 →
  initial_oranges - johann_ate - half_stolen + carson_returned = 30 :=
begin
  intros initial_oranges johann_ate half_stolen carson_returned,
  sorry
end

end johann_oranges_problem_l24_24567


namespace systematic_sampling_third_group_draw_l24_24374

theorem systematic_sampling_third_group_draw
  (first_draw : ℕ) (second_draw : ℕ) (first_draw_eq : first_draw = 2)
  (second_draw_eq : second_draw = 12) :
  ∃ (third_draw : ℕ), third_draw = 22 :=
by
  sorry

end systematic_sampling_third_group_draw_l24_24374


namespace average_age_of_students_l24_24417

theorem average_age_of_students :
  (8 * 14 + 6 * 16 + 17) / 15 = 15 :=
by
  sorry

end average_age_of_students_l24_24417


namespace solve_trigonometric_eqn_l24_24032

theorem solve_trigonometric_eqn (x : ℝ) : 
  (∃ k : ℤ, x = 3 * (π / 4 * (4 * k + 1))) ∨ (∃ n : ℤ, x = π * (3 * n + 1) ∨ x = π * (3 * n - 1)) :=
by 
  sorry

end solve_trigonometric_eqn_l24_24032


namespace probability_of_composite_l24_24544

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end probability_of_composite_l24_24544


namespace increasing_interval_l24_24060

noncomputable def f (x : ℝ) := Real.logb 2 (5 - 4 * x - x^2)

theorem increasing_interval : ∀ {x : ℝ}, (-5 < x ∧ x ≤ -2) → f x = Real.logb 2 (5 - 4 * x - x^2) := by
  sorry

end increasing_interval_l24_24060


namespace not_solution_B_l24_24797

theorem not_solution_B : ¬ (1 + 6 = 5) := by
  sorry

end not_solution_B_l24_24797


namespace find_matrix_N_l24_24064

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![2, -5],
  ![4, -3]
]

def B : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![-20, -8],
  ![9, 4]
]

def N : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![(46 / 7 : ℚ), (-58 / 7 : ℚ)],
  ![(-43 / 14 : ℚ), (53 / 14 : ℚ)]
]

theorem find_matrix_N : N ⬝ A = B := by
  sorry

end find_matrix_N_l24_24064


namespace joined_after_8_months_l24_24142

theorem joined_after_8_months
  (investment_A investment_B : ℕ)
  (time_A time_B : ℕ)
  (profit_ratio : ℕ × ℕ)
  (h_A : investment_A = 36000)
  (h_B : investment_B = 54000)
  (h_ratio : profit_ratio = (2, 1))
  (h_time_A : time_A = 12)
  (h_eq : (investment_A * time_A) / (investment_B * time_B) = (profit_ratio.1 / profit_ratio.2)) :
  time_B = 4 := by
  sorry

end joined_after_8_months_l24_24142


namespace area_ratio_l24_24948

noncomputable def AreaOfTrapezoid (AD BC : ℝ) (R : ℝ) : ℝ :=
  let s_π := Real.pi
  let height1 := 2 -- One of the heights considered
  let height2 := 14 -- Another height considered
  (AD + BC) / 2 * height1  -- First case area
  -- Here we assume the area uses sine which is arc-related, but provide fixed coefficients for area representation

noncomputable def AreaOfRectangle (R : ℝ) : ℝ :=
  let d := 2 * R
  -- Using the equation for area discussed
  d * d / 2

theorem area_ratio (AD BC : ℝ) (R : ℝ) (hAD : AD = 16) (hBC : BC = 12) (hR : R = 10) :
  let area_trap := AreaOfTrapezoid AD BC R
  let area_rect := AreaOfRectangle R
  area_trap / area_rect = 1 / 2 ∨ area_trap / area_rect = 49 / 50 :=
by
  sorry

end area_ratio_l24_24948


namespace probability_not_snowing_l24_24754

theorem probability_not_snowing (p_snow : ℚ) (h : p_snow = 5 / 8) : 1 - p_snow = 3 / 8 :=
by
  rw [h]
  sorry

end probability_not_snowing_l24_24754


namespace eval_at_d_eq_4_l24_24670

theorem eval_at_d_eq_4 : ((4: ℕ) ^ 4 - (4: ℕ) * ((4: ℕ) - 2) ^ 4) ^ 4 = 136048896 :=
by
  sorry

end eval_at_d_eq_4_l24_24670


namespace angle_in_third_quadrant_l24_24831

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.cos α < 0) : 
  ∃ k : ℤ, α = (2 * k + 1) * Real.pi + β ∧ β ∈ Set.Ioo (0 : ℝ) Real.pi :=
by
  sorry

end angle_in_third_quadrant_l24_24831


namespace max_value_of_f_l24_24426

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_of_f : ∃ max, max ∈ Set.image f (Set.Icc (-1 : ℝ) 1) ∧ max = Real.exp 1 - 1 :=
by
  sorry

end max_value_of_f_l24_24426


namespace remainder_of_division_l24_24439

open Polynomial

noncomputable def p : Polynomial ℤ := 3 * X^3 - 20 * X^2 + 45 * X + 23
noncomputable def d : Polynomial ℤ := (X - 3)^2

theorem remainder_of_division :
  ∃ q r : Polynomial ℤ, p = q * d + r ∧ degree r < degree d ∧ r = 6 * X + 41 := sorry

end remainder_of_division_l24_24439


namespace sum_of_squares_positive_l24_24219

theorem sum_of_squares_positive (x_1 x_2 k : ℝ) (h : x_1 ≠ x_2) 
  (hx1 : x_1^2 + 2*x_1 - k = 0) (hx2 : x_2^2 + 2*x_2 - k = 0) :
  x_1^2 + x_2^2 > 0 :=
by
  sorry

end sum_of_squares_positive_l24_24219


namespace same_side_of_line_l24_24350

theorem same_side_of_line (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) > 0 ↔ a < -7 ∨ a > 24 :=
by
  sorry

end same_side_of_line_l24_24350
