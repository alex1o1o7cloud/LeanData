import Mathlib

namespace value_of_expression_l22_22963

theorem value_of_expression (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by {
   rw h,
   norm_num,
   sorry
}

end value_of_expression_l22_22963


namespace image_center_after_reflection_and_translation_l22_22194

def circle_center_before_translation : ℝ × ℝ := (3, -4)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-x, y)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x, y + d)

theorem image_center_after_reflection_and_translation :
  translate_up (reflect_y_axis circle_center_before_translation) 5 = (-3, 1) :=
by
  -- The detail proof goes here.
  sorry

end image_center_after_reflection_and_translation_l22_22194


namespace find_a_2b_3c_value_l22_22439

-- Problem statement and conditions
theorem find_a_2b_3c_value (a b c : ℝ)
  (h : ∀ x : ℝ, (x < -1 ∨ abs (x - 10) ≤ 2) ↔ (x - a) * (x - b) / (x - c) ≤ 0)
  (h_ab : a < b) : a + 2 * b + 3 * c = 29 := 
sorry

end find_a_2b_3c_value_l22_22439


namespace mean_of_six_numbers_l22_22330

theorem mean_of_six_numbers (sum_six_numbers : ℚ) (h : sum_six_numbers = 3/4) : 
  (sum_six_numbers / 6) = 1/8 := by
  -- proof can be filled in here
  sorry

end mean_of_six_numbers_l22_22330


namespace hyperbola_slope_reciprocals_l22_22801

theorem hyperbola_slope_reciprocals (P : ℝ × ℝ) (t : ℝ) :
  (P.1 = t ∧ P.2 = - (8 / 9) * t ∧ t ≠ 0 ∧  
    ∃ k1 k2: ℝ, k1 = - (8 * t) / (9 * (t + 3)) ∧ k2 = - (8 * t) / (9 * (t - 3)) ∧
    (1 / k1) + (1 / k2) = -9 / 4) ∧
    ((P = (9/5, -(8/5)) ∨ P = (-(9/5), 8/5)) →
        ∃ kOA kOB kOC kOD : ℝ, (kOA + kOB + kOC + kOD = 0)) := 
sorry

end hyperbola_slope_reciprocals_l22_22801


namespace scientific_notation_of_384000_l22_22862

theorem scientific_notation_of_384000 :
  (384000 : ℝ) = 3.84 * 10^5 :=
by
  sorry

end scientific_notation_of_384000_l22_22862


namespace sum_of_ages_l22_22833

-- Define the ages of Maggie, Juliet, and Ralph
def maggie_age : ℕ := by
  let juliet_age := 10
  let maggie_age := juliet_age - 3
  exact maggie_age

def ralph_age : ℕ := by
  let juliet_age := 10
  let ralph_age := juliet_age + 2
  exact ralph_age

-- The main theorem: The sum of Maggie's and Ralph's ages
theorem sum_of_ages : maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l22_22833


namespace find_n_l22_22171

theorem find_n (n : ℕ) (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^28) : n = 27 :=
sorry

end find_n_l22_22171


namespace num_two_digit_numbers_with_digit_less_than_35_l22_22064

theorem num_two_digit_numbers_with_digit_less_than_35 : 
  let total_two_digit := 90 in
  let unsuitable_numbers := 35 in
  let suitable_numbers := total_two_digit - unsuitable_numbers
  in suitable_numbers = 55 :=
by 
  let total_two_digit := 90
  let unsuitable_numbers := 35
  let suitable_numbers := total_two_digit - unsuitable_numbers
  show suitable_numbers = 55, from sorry

end num_two_digit_numbers_with_digit_less_than_35_l22_22064


namespace rainfall_ratio_l22_22375

theorem rainfall_ratio (R1 R2 : ℕ) (H1 : R2 = 18) (H2 : R1 + R2 = 30) : R2 / R1 = 3 / 2 := by
  sorry

end rainfall_ratio_l22_22375


namespace two_digit_numbers_less_than_35_l22_22075

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l22_22075


namespace product_of_two_numbers_l22_22708

variable (x y : ℝ)

-- conditions
def condition1 : Prop := x + y = 23
def condition2 : Prop := x - y = 7

-- target
theorem product_of_two_numbers {x y : ℝ} 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  x * y = 120 := 
sorry

end product_of_two_numbers_l22_22708


namespace equidistant_point_on_y_axis_l22_22340

theorem equidistant_point_on_y_axis :
  ∃ (y : ℝ), 0 < y ∧ 
  (dist (0, y) (-3, 0) = dist (0, y) (-2, 5)) ∧ 
  y = 2 :=
by
  sorry

end equidistant_point_on_y_axis_l22_22340


namespace students_in_class_l22_22492

theorem students_in_class (b n : ℕ) :
  6 * (b + 1) = n ∧ 9 * (b - 1) = n → n = 36 :=
by
  sorry

end students_in_class_l22_22492


namespace radius_of_circle_l22_22691

theorem radius_of_circle (r : ℝ) (C : ℝ) (A : ℝ) (h1 : 3 * C = 2 * A) 
  (h2 : C = 2 * Real.pi * r) (h3 : A = Real.pi * r^2) : 
  r = 3 :=
by 
  sorry

end radius_of_circle_l22_22691


namespace student_selection_problem_l22_22773

noncomputable def total_selections : ℕ :=
  let C := Nat.choose
  let A := Nat.factorial
  (C 3 1 * C 3 2 + C 3 2 * C 3 1 + C 3 3) * A 3

theorem student_selection_problem :
  total_selections = 114 :=
by
  sorry

end student_selection_problem_l22_22773


namespace cannot_factorize_using_difference_of_squares_l22_22980

theorem cannot_factorize_using_difference_of_squares (x y : ℝ) :
  ¬ ∃ a b : ℝ, -x^2 - y^2 = a^2 - b^2 :=
sorry

end cannot_factorize_using_difference_of_squares_l22_22980


namespace complete_square_solution_l22_22166

theorem complete_square_solution (x: ℝ) : (x^2 + 8 * x - 3 = 0) -> ((x + 4)^2 = 19) := 
by
  sorry

end complete_square_solution_l22_22166


namespace opposite_face_number_l22_22683

theorem opposite_face_number (sum_faces : ℕ → ℕ → ℕ) (face_number : ℕ → ℕ) :
  (face_number 1 = 6) ∧ (face_number 2 = 7) ∧ (face_number 3 = 8) ∧ 
  (face_number 4 = 9) ∧ (face_number 5 = 10) ∧ (face_number 6 = 11) →
  (sum_faces 1 2 + sum_faces 3 4 + sum_faces 5 6 = 33 + 18) →
  (sum_faces 1 2 + sum_faces 3 4 + sum_faces 5 6 = 35 + 16) →
  (face_number 2 ≠ 9 ∨ face_number 2 ≠ 11) → 
  face_number 2 = 9 ∨ face_number 2 = 11 :=
by
  intros hface_numbers hsum1 hsum2 hnot_possible
  sorry

end opposite_face_number_l22_22683


namespace problem_statement_l22_22806

def U := Set ℝ
def M := { x : ℝ | x^2 - 4 * x - 5 < 0 }
def N := { x : ℝ | 1 ≤ x }
def comp_U_N := { x : ℝ | x < 1 }
def intersection := { x : ℝ | -1 < x ∧ x < 1 }

theorem problem_statement : M ∩ comp_U_N = intersection := sorry

end problem_statement_l22_22806


namespace impossible_coins_l22_22600

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l22_22600


namespace cube_volume_is_216_l22_22205

-- Define the conditions
def total_edge_length : ℕ := 72
def num_edges_of_cube : ℕ := 12

-- The side length of the cube can be calculated as
def side_length (E : ℕ) (n : ℕ) : ℕ := E / n

-- The volume of the cube is the cube of its side length
def volume (s : ℕ) : ℕ := s ^ 3

theorem cube_volume_is_216 (E : ℕ) (n : ℕ) (V : ℕ) 
  (hE : E = total_edge_length) 
  (hn : n = num_edges_of_cube) 
  (hv : V = volume (side_length E n)) : 
  V = 216 := by
  sorry

end cube_volume_is_216_l22_22205


namespace unique_triple_solution_l22_22525

theorem unique_triple_solution {x y z : ℤ} (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (H1 : x ∣ y * z - 1) (H2 : y ∣ z * x - 1) (H3 : z ∣ x * y - 1) :
  (x, y, z) = (5, 3, 2) :=
sorry

end unique_triple_solution_l22_22525


namespace simplify_sqrt_7_pow_6_l22_22647

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l22_22647


namespace arctan_sum_l22_22011

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l22_22011


namespace fraction_subtraction_l22_22367

theorem fraction_subtraction (h : ((8 : ℚ) / 21 - (10 / 63) = (2 / 9))) : 
  8 / 21 - 10 / 63 = 2 / 9 :=
by
  sorry

end fraction_subtraction_l22_22367


namespace two_vertical_asymptotes_l22_22024

theorem two_vertical_asymptotes (k : ℝ) : 
  (∀ x : ℝ, (x ≠ 3 ∧ x ≠ -2) → 
           (∃ δ > 0, ∀ ε > 0, ∃ x' : ℝ, x + δ > x' ∧ x' > x - δ ∧ 
                             (x' ≠ 3 ∧ x' ≠ -2) → 
                             |(x'^2 + 2 * x' + k) / (x'^2 - x' - 6)| > 1/ε)) ↔ 
  (k ≠ -15 ∧ k ≠ 0) :=
sorry

end two_vertical_asymptotes_l22_22024


namespace probability_four_of_five_same_value_l22_22030

-- Define the conditions
def standard_six_sided_dice := {1, 2, 3, 4, 5, 6}
def initial_roll_condition (dices : Fin 5 → ℕ) : Prop :=
  ∃ num : ℕ, num ∈ standard_six_sided_dice ∧
             (∃ triplet : Finset (Fin 5), triplet.card = 3 ∧
               ∀ i ∈ triplet, dices i = num) ∧
             ∀ k : Fin 5, dices k ≠ num → ∃ j, dices j ≠ dices k

-- Define the probability space and the required outcome
noncomputable def probability_of_at_least_four_same_value :=
  @prob_set (Finset (Finset (Fin 6))) _ (λ outcomes, ∃ num ∈ standard_six_sided_dice, 
    filter (λ d, d = num) outcomes  ≥ 4)

-- State the theorem
theorem probability_four_of_five_same_value (dices : Fin 5 → ℕ) :
  initial_roll_condition dices →
  probability_of_at_least_four_same_value = 1/36 := by 
sorry

end probability_four_of_five_same_value_l22_22030


namespace impossible_coins_l22_22617

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l22_22617


namespace number_verification_l22_22361

def is_product_of_four_consecutive (n : ℕ) : Prop :=
  ∃ a : ℕ, n = a * (a + 1) * (a + 2) * (a + 3)

theorem number_verification (h1 : 1680 % 3 = 0) (h2 : ∃ a : ℕ, 1680 = a * (a + 1) * (a + 2) * (a + 3)) : 
  is_product_of_four_consecutive 1680 :=
by
  sorry

end number_verification_l22_22361


namespace find_sum_abc_l22_22585

noncomputable def f (x a b c : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x + c

theorem find_sum_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (habc_distinct : a ≠ b) (hfa : f a a b c = a^3) (hfb : f b a b c = b^3) : 
  a + b + c = 18 := 
sorry

end find_sum_abc_l22_22585


namespace shampoo_duration_l22_22279

-- Conditions
def rose_shampoo : ℚ := 1/3
def jasmine_shampoo : ℚ := 1/4
def daily_usage : ℚ := 1/12

-- Question
theorem shampoo_duration : (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end shampoo_duration_l22_22279


namespace maria_uses_666_blocks_l22_22110

theorem maria_uses_666_blocks :
  let original_volume := 15 * 12 * 7
  let interior_length := 15 - 2 * 1.5
  let interior_width := 12 - 2 * 1.5
  let interior_height := 7 - 1.5
  let interior_volume := interior_length * interior_width * interior_height
  let blocks_volume := original_volume - interior_volume
  blocks_volume = 666 :=
by
  sorry

end maria_uses_666_blocks_l22_22110


namespace max_point_f_l22_22042

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Maximum point of the function f is -2
theorem max_point_f : ∃ m, m = -2 ∧ ∀ x, f x ≤ f (-2) :=
by
  sorry

end max_point_f_l22_22042


namespace sean_whistles_l22_22312

def charles_whistles : ℕ := 13
def extra_whistles : ℕ := 32

theorem sean_whistles : charles_whistles + extra_whistles = 45 := by
  sorry

end sean_whistles_l22_22312


namespace ZhangSan_correct_probability_l22_22467

namespace ZhangSan

-- Define the total number of questions
def total_questions : ℕ := 4

-- Define the probability of Zhang San having an idea for a question
def P_A1 : ℚ := 3 / 4

-- Define the probability of Zhang San being unclear about a question
def P_A2 : ℚ := 1 / 4

-- Define the probability of answering correctly given an idea
def P_B_given_A1 : ℚ := 3 / 4

-- Define the probability of answering correctly given an unclear status
def P_B_given_A2 : ℚ := 1 / 4

-- Define the probability of answering a question correctly
def P_B : ℚ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2

theorem ZhangSan_correct_probability :
  P_B = 5 / 8 :=
by
  unfold P_B P_A1 P_A2 P_B_given_A1 P_B_given_A2
  sorry

end ZhangSan

end ZhangSan_correct_probability_l22_22467


namespace range_of_m_l22_22094

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ (x^3 - 3 * x + m = 0)) → (m ≥ -2 ∧ m ≤ 2) :=
sorry

end range_of_m_l22_22094


namespace simplify_sqrt_power_l22_22624

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l22_22624


namespace cost_of_apples_l22_22245

def cost_per_kilogram (m : ℝ) : ℝ := m
def number_of_kilograms : ℝ := 3

theorem cost_of_apples (m : ℝ) : cost_per_kilogram m * number_of_kilograms = 3 * m :=
by
  unfold cost_per_kilogram number_of_kilograms
  sorry

end cost_of_apples_l22_22245


namespace mia_spent_per_parent_l22_22848

theorem mia_spent_per_parent (amount_sibling : ℕ) (num_siblings : ℕ) (total_spent : ℕ) 
  (num_parents : ℕ) : 
  amount_sibling = 30 → num_siblings = 3 → total_spent = 150 → num_parents = 2 → 
  (total_spent - num_siblings * amount_sibling) / num_parents = 30 :=
by
  sorry

end mia_spent_per_parent_l22_22848


namespace batsman_average_l22_22899

theorem batsman_average (A : ℕ) (total_runs_before : ℕ) (new_score : ℕ) (increase : ℕ)
  (h1 : total_runs_before = 11 * A)
  (h2 : new_score = 70)
  (h3 : increase = 3)
  (h4 : 11 * A + new_score = 12 * (A + increase)) :
  (A + increase) = 37 :=
by
  -- skipping the proof with sorry
  sorry

end batsman_average_l22_22899


namespace cost_of_five_trip_ticket_l22_22520

-- Variables for the costs of the tickets
variables (x y z : ℕ)

-- Conditions from the problem
def condition1 : Prop := 5 * x > y
def condition2 : Prop := 4 * y > z
def condition3 : Prop := z + 3 * y = 33
def condition4 : Prop := 20 + 3 * 5 = 35

-- The theorem to prove
theorem cost_of_five_trip_ticket (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z y) (h4 : condition4) : y = 5 := 
by
  sorry

end cost_of_five_trip_ticket_l22_22520


namespace factor_expression_l22_22523

variable (x : ℝ)

theorem factor_expression : 
  (10 * x^3 + 50 * x^2 - 5) - (-5 * x^3 + 15 * x^2 - 5) = 5 * x^2 * (3 * x + 7) := 
by 
  sorry

end factor_expression_l22_22523


namespace total_interest_rate_l22_22742

theorem total_interest_rate (I_total I_11: ℝ) (r_9 r_11: ℝ) (h1: I_total = 100000) (h2: I_11 = 12499.999999999998) (h3: I_11 < I_total):
  r_9 = 0.09 →
  r_11 = 0.11 →
  ( ((I_total - I_11) * r_9 + I_11 * r_11) / I_total * 100 = 9.25 ) :=
by
  sorry

end total_interest_rate_l22_22742


namespace astronaut_days_on_orbius_l22_22119

noncomputable def days_in_year : ℕ := 250
noncomputable def seasons_in_year : ℕ := 5
noncomputable def seasons_stayed : ℕ := 3

theorem astronaut_days_on_orbius :
  (days_in_year / seasons_in_year) * seasons_stayed = 150 := by
  sorry

end astronaut_days_on_orbius_l22_22119


namespace find_x_l22_22990

def star (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_x (x : ℝ) (h : star 4 x = 52) : x = 8 :=
by
  sorry

end find_x_l22_22990


namespace draw_four_balls_in_order_l22_22175

theorem draw_four_balls_in_order :
  let total_balls := 15
  let color_sequence_length := 4
  let colors_sequence := ["Red", "Green", "Blue", "Yellow"]
  total_balls * (total_balls - 1) * (total_balls - 2) * (total_balls - 3) = 32760 :=
by 
  sorry

end draw_four_balls_in_order_l22_22175


namespace find_c_l22_22901

-- Let a, b, c, d, and e be positive consecutive integers.
variables {a b c d e : ℕ}

-- Conditions: 
def conditions (a b c d e : ℕ) : Prop :=
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a + b = e - 1 ∧
  a * b = d + 1

-- Proof statement
theorem find_c (h : conditions a b c d e) : c = 4 :=
by sorry

end find_c_l22_22901


namespace impossible_coins_l22_22599

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l22_22599


namespace rectangle_invalid_perimeter_l22_22916

-- Define conditions
def positive_integer (n : ℕ) : Prop := n > 0

-- Define the rectangle with given area
def area_24 (length width : ℕ) : Prop := length * width = 24

-- Define the function to calculate perimeter for given length and width
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- The theorem to prove
theorem rectangle_invalid_perimeter (length width : ℕ) (h₁ : positive_integer length) (h₂ : positive_integer width) (h₃ : area_24 length width) : 
  (perimeter length width) ≠ 36 :=
sorry

end rectangle_invalid_perimeter_l22_22916


namespace directrix_of_parabola_l22_22935

theorem directrix_of_parabola (h : ∀ x : ℝ, y = -3 * x ^ 2 + 6 * x - 5) : ∃ y : ℝ, y = -25 / 12 :=
by
  sorry

end directrix_of_parabola_l22_22935


namespace quoted_value_of_stock_l22_22908

theorem quoted_value_of_stock (F P : ℝ) (h1 : F > 0) (h2 : P = 1.25 * F) : 
  (0.10 * F) / P = 0.08 := 
sorry

end quoted_value_of_stock_l22_22908


namespace simplify_sqrt_7_pow_6_l22_22646

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l22_22646


namespace two_digit_numbers_less_than_35_l22_22076

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l22_22076


namespace product_increase_by_13_l22_22574

theorem product_increase_by_13 {
    a1 a2 a3 a4 a5 a6 a7 : ℕ
} : (a1 > 3) → (a2 > 3) → (a3 > 3) → (a4 > 3) → (a5 > 3) → (a6 > 3) → (a7 > 3) → 
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * a1 * a2 * a3 * a4 * a5 * a6 * a7) :=
        sorry

end product_increase_by_13_l22_22574


namespace factorize_expression_l22_22768

theorem factorize_expression (y a : ℝ) : 
  3 * y * a ^ 2 - 6 * y * a + 3 * y = 3 * y * (a - 1) ^ 2 :=
by
  sorry

end factorize_expression_l22_22768


namespace calculate_expression_l22_22366

theorem calculate_expression :
  (16^16 * 8^8) / 4^32 = 16777216 := by
  sorry

end calculate_expression_l22_22366


namespace students_per_bus_l22_22303

/-- The number of students who can be accommodated in each bus -/
theorem students_per_bus (total_students : ℕ) (students_in_cars : ℕ) (num_buses : ℕ) 
(h1 : total_students = 375) (h2 : students_in_cars = 4) (h3 : num_buses = 7) : 
(total_students - students_in_cars) / num_buses = 53 :=
by
  sorry

end students_per_bus_l22_22303


namespace frequency_of_a_is_3_l22_22823

def sentence : String := "Happy Teachers'Day!"

def frequency_of_a_in_sentence (s : String) : Nat :=
  s.foldl (λ acc c => if c = 'a' then acc + 1 else acc) 0

theorem frequency_of_a_is_3 : frequency_of_a_in_sentence sentence = 3 :=
  by
    sorry

end frequency_of_a_is_3_l22_22823


namespace card_pair_probability_l22_22734

theorem card_pair_probability :
  ∃ m n : ℕ, 
    0 < m ∧ 0 < n ∧ 
    (m.gcd n = 1) ∧ 
    (m = 73) ∧ 
    (n = 1225) ∧ 
    (m + n = 1298) := by
{
  sorry
}

end card_pair_probability_l22_22734


namespace four_disjoint_subsets_with_equal_sums_l22_22590

theorem four_disjoint_subsets_with_equal_sums :
  ∀ (S : Finset ℕ), 
  (∀ x ∈ S, 100 ≤ x ∧ x ≤ 999) ∧ S.card = 117 → 
  ∃ A B C D : Finset ℕ, 
    (A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧ D ⊆ S) ∧ 
    (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ A ∩ D = ∅ ∧ B ∩ C = ∅ ∧ B ∩ D = ∅ ∧ C ∩ D = ∅) ∧ 
    (A.sum id = B.sum id ∧ B.sum id = C.sum id ∧ C.sum id = D.sum id) := by
  sorry

end four_disjoint_subsets_with_equal_sums_l22_22590


namespace negation_of_existence_l22_22104

theorem negation_of_existence :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by
  sorry

end negation_of_existence_l22_22104


namespace simplify_sqrt_seven_pow_six_proof_l22_22629

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l22_22629


namespace ratio_tina_betsy_l22_22926

theorem ratio_tina_betsy :
  ∀ (t_cindy t_betsy t_tina : ℕ),
  t_cindy = 12 →
  t_betsy = t_cindy / 2 →
  t_tina = t_cindy + 6 →
  t_tina / t_betsy = 3 :=
by
  intros t_cindy t_betsy t_tina h_cindy h_betsy h_tina
  sorry

end ratio_tina_betsy_l22_22926


namespace roots_equality_l22_22388

variable {α β p q : ℝ}

theorem roots_equality (h1 : α ≠ β)
    (h2 : α * α + p * α + q = 0 ∧ β * β + p * β + q = 0)
    (h3 : α^3 - α^2 * β - α * β^2 + β^3 = 0) : 
  p = 0 ∧ q < 0 :=
by 
  sorry

end roots_equality_l22_22388


namespace solve_fractional_eq_l22_22648

-- Defining the fractional equation as a predicate
def fractional_eq (x : ℝ) : Prop :=
  (5 / x) = (7 / (x - 2))

-- The main theorem to be proven
theorem solve_fractional_eq : ∃ x : ℝ, fractional_eq x ∧ x = -5 := by
  sorry

end solve_fractional_eq_l22_22648


namespace partial_fraction_l22_22989

noncomputable def roots := 
  {p q r : ℝ // is_root (X^3 - 24*X^2 + 98*X - 75) p ∧ 
                  is_root (X^3 - 24*X^2 + 98*X - 75) q ∧ 
                  is_root (X^3 - 24*X^2 + 98*X - 75) r}

noncomputable def A (p q r s : ℝ) := 
  (5 / ((s - p) * (s - q) * (s - r)))

noncomputable def B (p q r s : ℝ) := 
  (5 / ((s - p) * (s - q) * (s - r)))

noncomputable def C (p q r s : ℝ) :=
  (5 / ((s - p) * (s - q) * (s - r)))

theorem partial_fraction (p q r A B C : ℝ) (hpqrs : roots p q r) 
  (hA : A (p q r s) = (dfrac{A}{s-p})
  (hB : B (p q r s) = (dfrac{B}{s-q})
  (hC : C (p q r s) = (dfrac{C}{s-r})
  : 1 / A + 1 / B + 1 / C = 256 :=
sorry

end partial_fraction_l22_22989


namespace min_banks_l22_22288

theorem min_banks (total_rubles : ℝ) (max_payout : ℝ) (total_rubles = 10000000) (max_payout = 1400000) : 
  (Real.ceil (total_rubles / max_payout) = 8) := by
  sorry

end min_banks_l22_22288


namespace pentagon_area_proof_l22_22760

noncomputable def area_of_pentagon : ℕ :=
  let side1 := 18
  let side2 := 25
  let side3 := 30
  let side4 := 28
  let side5 := 25
  -- Assuming the total area calculated from problem's conditions
  950

theorem pentagon_area_proof : area_of_pentagon = 950 := by
  sorry

end pentagon_area_proof_l22_22760


namespace ten_factorial_mod_thirteen_l22_22227

open Nat

theorem ten_factorial_mod_thirteen :
  (10! % 13) = 6 := by
  sorry

end ten_factorial_mod_thirteen_l22_22227


namespace possible_measures_of_angle_X_l22_22135

theorem possible_measures_of_angle_X : 
  ∃ n : ℕ, n = 17 ∧ (∀ (X Y : ℕ), 
    X > 0 ∧ Y > 0 ∧ X + Y = 180 ∧ 
    ∃ m : ℕ, m ≥ 1 ∧ X = m * Y) :=
sorry

end possible_measures_of_angle_X_l22_22135


namespace Martha_should_buy_84oz_of_apples_l22_22846

theorem Martha_should_buy_84oz_of_apples 
  (apple_weight : ℕ)
  (orange_weight : ℕ)
  (bag_capacity : ℕ)
  (num_bags : ℕ)
  (equal_fruits : Prop) 
  (total_weight : ℕ :=
    num_bags * bag_capacity)
  (pair_weight : ℕ :=
    apple_weight + orange_weight)
  (num_pairs : ℕ :=
    total_weight / pair_weight)
  (total_apple_weight : ℕ :=
    num_pairs * apple_weight) :
  apple_weight = 4 → 
  orange_weight = 3 → 
  bag_capacity = 49 → 
  num_bags = 3 → 
  equal_fruits → 
  total_apple_weight = 84 := 
by sorry

end Martha_should_buy_84oz_of_apples_l22_22846


namespace two_digit_numbers_less_than_35_l22_22077

theorem two_digit_numbers_less_than_35 : 
  ∃ n, n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
  (let tens_digit := (n + 10) / 10 in 
   let units_digit := (n + 10) % 10 in 
   tens_digit < 3 ∨ units_digit < 5) → 
  nat.card {n | n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
                    (let tens_digit := (n + 10) / 10 in 
                     let units_digit := (n + 10) % 10 in 
                     tens_digit < 3 ∨ units_digit < 5)} = 55
:= sorry

end two_digit_numbers_less_than_35_l22_22077


namespace initial_number_of_friends_is_six_l22_22735

theorem initial_number_of_friends_is_six
  (car_cost : ℕ)
  (car_wash_earnings : ℕ)
  (F : ℕ)
  (additional_cost_when_one_friend_leaves : ℕ)
  (h1 : car_cost = 1700)
  (h2 : car_wash_earnings = 500)
  (remaining_cost := car_cost - car_wash_earnings)
  (cost_per_friend_before := remaining_cost / F)
  (cost_per_friend_after := remaining_cost / (F - 1))
  (h3 : additional_cost_when_one_friend_leaves = 40)
  (h4 : cost_per_friend_after = cost_per_friend_before + additional_cost_when_one_friend_leaves) :
  F = 6 :=
by
  sorry

end initial_number_of_friends_is_six_l22_22735


namespace evaluate_expression_l22_22532

theorem evaluate_expression :
  ( ( ( 5 / 2 : ℚ ) / ( 7 / 12 : ℚ ) ) - ( 4 / 9 : ℚ ) ) = ( 242 / 63 : ℚ ) :=
by
  sorry

end evaluate_expression_l22_22532


namespace tetris_blocks_form_square_l22_22321

-- Definitions of Tetris blocks types
inductive TetrisBlock
| A | B | C | D | E | F | G

open TetrisBlock

-- Definition of a block's ability to form a square
def canFormSquare (block: TetrisBlock) : Prop :=
  block = A ∨ block = B ∨ block = C ∨ block = D ∨ block = G

-- The main theorem statement
theorem tetris_blocks_form_square : ∀ (block : TetrisBlock), canFormSquare block → block = A ∨ block = B ∨ block = C ∨ block = D ∨ block = G := 
by
  intros block h
  exact h

end tetris_blocks_form_square_l22_22321


namespace linear_equation_a_is_minus_one_l22_22092

theorem linear_equation_a_is_minus_one (a : ℝ) (x : ℝ) :
  ((a - 1) * x ^ (2 - |a|) + 5 = 0) → (2 - |a| = 1) → (a ≠ 1) → a = -1 :=
by
  intros h1 h2 h3
  sorry

end linear_equation_a_is_minus_one_l22_22092


namespace f_diff_l22_22541

-- Define the function f(n)
def f (n : ℕ) : ℚ :=
  (Finset.range (3 * n + 1 + 1)).sum (λ i => 1 / (n + i + 1))

-- The theorem stating the main problem
theorem f_diff (k : ℕ) : 
  f (k + 1) - f k = (1 / (3 * k + 2)) + (1 / (3 * k + 3)) + (1 / (3 * k + 4)) - (1 / (k + 1)) :=
by
  sorry

end f_diff_l22_22541


namespace arithmetic_sequence_a4_possible_values_l22_22786

theorem arithmetic_sequence_a4_possible_values (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 * a 5 = 9)
  (h3 : a 2 = 3) : 
  a 4 = 3 ∨ a 4 = 7 := 
by 
  sorry

end arithmetic_sequence_a4_possible_values_l22_22786


namespace arithmetic_sequence_a12_bound_l22_22430

theorem arithmetic_sequence_a12_bound (a_1 d : ℤ) (h8 : a_1 + 7 * d ≥ 15) (h9 : a_1 + 8 * d ≤ 13) : 
  a_1 + 11 * d ≤ 7 :=
by
  sorry

end arithmetic_sequence_a12_bound_l22_22430


namespace anna_gets_more_candy_l22_22518

theorem anna_gets_more_candy :
  let anna_pieces_per_house := 14
  let anna_houses := 60
  let billy_pieces_per_house := 11
  let billy_houses := 75
  let anna_total := anna_pieces_per_house * anna_houses
  let billy_total := billy_pieces_per_house * billy_houses
  anna_total - billy_total = 15 := by
    let anna_pieces_per_house := 14
    let anna_houses := 60
    let billy_pieces_per_house := 11
    let billy_houses := 75
    let anna_total := anna_pieces_per_house * anna_houses
    let billy_total := billy_pieces_per_house * billy_houses
    have h1 : anna_total = 14 * 60 := rfl
    have h2 : billy_total = 11 * 75 := rfl
    sorry

end anna_gets_more_candy_l22_22518


namespace area_of_square_l22_22323

theorem area_of_square (r s l b : ℝ) (h1 : l = (2/5) * r)
                               (h2 : r = s)
                               (h3 : b = 10)
                               (h4 : l * b = 220) :
  s^2 = 3025 :=
by
  -- proof goes here
  sorry

end area_of_square_l22_22323


namespace arithmetic_sqrt_of_sqrt_16_l22_22672

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22672


namespace sum_tetrahedral_formula_l22_22852

def tetrahedral_number (n : ℕ) : ℕ :=
  Nat.choose (n + 2) 3

def sum_tetrahedral (k : ℕ) : ℕ :=
  (Finset.range k).sum (λ i => tetrahedral_number (i + 1))

theorem sum_tetrahedral_formula (k : ℕ) : sum_tetrahedral (k + 1) = Nat.choose (k + 3) 4 :=
by
  sorry

end sum_tetrahedral_formula_l22_22852


namespace calculation_l22_22184

variable (x y z : ℕ)

theorem calculation (h1 : x + y + z = 20) (h2 : x + y - z = 8) :
  x + y = 14 :=
  sorry

end calculation_l22_22184


namespace inequality_proof_l22_22785

theorem inequality_proof
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ (2 * (a^3 + b^3 + c^3)) / (a * b * c) + 3 :=
by
  sorry

end inequality_proof_l22_22785


namespace liquidX_percentage_l22_22000

variable (wA wB : ℝ) (pA pB : ℝ) (mA mB : ℝ)

-- Conditions
def weightA : ℝ := 200
def weightB : ℝ := 700
def percentA : ℝ := 0.8
def percentB : ℝ := 1.8

-- The question and answer.
theorem liquidX_percentage :
  (percentA / 100 * weightA + percentB / 100 * weightB) / (weightA + weightB) * 100 = 1.58 := by
  sorry

end liquidX_percentage_l22_22000


namespace collinear_vectors_value_m_l22_22957

theorem collinear_vectors_value_m (m : ℝ) : 
  (∃ k : ℝ, (2*m = k * (m - 1)) ∧ (3 = k)) → m = 3 :=
by
  sorry

end collinear_vectors_value_m_l22_22957


namespace bus_prob_at_least_two_days_on_time_l22_22050

noncomputable def bus_on_time (p : ℚ) : ℚ :=
  let q := 1 - p
  let P_X_2 := (3.choose 2) * (p^2) * (q^1)
  let P_X_3 := (3.choose 3) * (p^3) * (q^0)
  P_X_2 + P_X_3

theorem bus_prob_at_least_two_days_on_time :
  bus_on_time (3/5) = 81/125 :=
by
  sorry

end bus_prob_at_least_two_days_on_time_l22_22050


namespace cos_beta_l22_22397

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
  (h_cos_α : Real.cos α = 3/5) (h_cos_alpha_plus_beta : Real.cos (α + β) = -5/13) : 
  Real.cos β = 33/65 :=
by
  sorry

end cos_beta_l22_22397


namespace total_coffee_consumed_l22_22434

def Ivory_hourly_coffee := 2
def Kimberly_hourly_coffee := Ivory_hourly_coffee
def Brayan_hourly_coffee := 4
def Raul_hourly_coffee := Brayan_hourly_coffee / 2
def duration_hours := 10

theorem total_coffee_consumed :
  (Brayan_hourly_coffee * duration_hours) + 
  (Ivory_hourly_coffee * duration_hours) + 
  (Kimberly_hourly_coffee * duration_hours) + 
  (Raul_hourly_coffee * duration_hours) = 100 :=
by sorry

end total_coffee_consumed_l22_22434


namespace jameson_badminton_medals_l22_22827

theorem jameson_badminton_medals :
  ∃ (b : ℕ),  (∀ (t s : ℕ), t = 5 → s = 2 * t → t + s + b = 20) ∧ b = 5 :=
by {
sorry
}

end jameson_badminton_medals_l22_22827


namespace min_value_of_expression_l22_22252

noncomputable def f (m : ℝ) : ℝ :=
  let x1 := -m - (m^2 + 3 * m - 2)
  let x2 := -2 * m - x1
  x1 * (x2 + x1) + x2^2

theorem min_value_of_expression :
  ∃ m : ℝ, f m = 3 * (m - 1/2)^2 + 5/4 ∧ f m ≥ f (1/2) := by
  sorry

end min_value_of_expression_l22_22252


namespace evaluate_expression_l22_22198

theorem evaluate_expression (a x : ℤ) (h : x = a + 5) : 2 * x - a + 4 = a + 14 :=
by
  sorry

end evaluate_expression_l22_22198


namespace problem_statement_l22_22272

-- Let's define the conditions
def num_blue_balls : ℕ := 8
def num_green_balls : ℕ := 7
def total_balls : ℕ := num_blue_balls + num_green_balls

-- Function to calculate combinations (binomial coefficients)
def combination (n r : ℕ) : ℕ :=
  n.choose r

-- Specific combinations for this problem
def blue_ball_ways : ℕ := combination num_blue_balls 3
def green_ball_ways : ℕ := combination num_green_balls 2
def total_ways : ℕ := combination total_balls 5

-- The number of favorable outcomes
def favorable_outcomes : ℕ := blue_ball_ways * green_ball_ways

-- The probability
def probability : ℚ := favorable_outcomes / total_ways

-- The theorem stating our result
theorem problem_statement : probability = 1176/3003 := by
  sorry

end problem_statement_l22_22272


namespace train_speed_correct_l22_22185

def train_length : ℝ := 110
def bridge_length : ℝ := 142
def crossing_time : ℝ := 12.598992080633549
def expected_speed : ℝ := 20.002

theorem train_speed_correct :
  (train_length + bridge_length) / crossing_time = expected_speed :=
by
  sorry

end train_speed_correct_l22_22185


namespace parallelogram_diagonal_square_l22_22437

theorem parallelogram_diagonal_square (A B C D P Q R S : Type)
    (area_ABCD : ℝ) (proj_A_P_BD proj_C_Q_BD proj_B_R_AC proj_D_S_AC : Prop)
    (PQ RS : ℝ) (d_squared : ℝ) 
    (h_area : area_ABCD = 24)
    (h_proj_A_P : proj_A_P_BD) (h_proj_C_Q : proj_C_Q_BD)
    (h_proj_B_R : proj_B_R_AC) (h_proj_D_S : proj_D_S_AC)
    (h_PQ_length : PQ = 8) (h_RS_length : RS = 10)
    : d_squared = 62 + 20*Real.sqrt 61 := sorry

end parallelogram_diagonal_square_l22_22437


namespace poly_remainder_l22_22534

theorem poly_remainder (x : ℤ) :
  (x^1012) % (x^3 - x^2 + x - 1) = 1 := by
  sorry

end poly_remainder_l22_22534


namespace draw_probability_l22_22909

variable (P_lose_a win_a : ℝ)
variable (not_lose_a : ℝ := 0.8)
variable (win_prob_a : ℝ := 0.6)

-- Given conditions
def A_not_losing : Prop := not_lose_a = win_prob_a + win_a

-- Main theorem to prove
theorem draw_probability : P_lose_a = 0.2 :=
by
  sorry

end draw_probability_l22_22909


namespace simplify_expression_l22_22856

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^3 + 2 * b^2) - 2 * b^2 + 5 = 9 * b^4 + 6 * b^3 - 2 * b^2 + 5 := sorry

end simplify_expression_l22_22856


namespace range_of_a_l22_22953

noncomputable def f (x a : ℝ) : ℝ := (Real.sqrt x) / (x^3 - 3 * x + a)

theorem range_of_a (a : ℝ) :
    (∀ x, 0 ≤ x → x^3 - 3 * x + a ≠ 0) ↔ 2 < a := 
by 
  sorry

end range_of_a_l22_22953


namespace value_of_y_at_x8_l22_22089

theorem value_of_y_at_x8
  (k : ℝ)
  (y : ℝ → ℝ)
  (hx64 : y 64 = 4 * Real.sqrt 3)
  (hy_def : ∀ x, y x = k * x^(1 / 3)) :
  y 8 = 2 * Real.sqrt 3 :=
by {
  sorry,
}

end value_of_y_at_x8_l22_22089


namespace total_spending_l22_22765

-- Define the condition of spending for each day
def friday_spending : ℝ := 20
def saturday_spending : ℝ := 2 * friday_spending
def sunday_spending : ℝ := 3 * friday_spending

-- Define the statement to be proven
theorem total_spending : friday_spending + saturday_spending + sunday_spending = 120 :=
by
  -- Provide conditions and calculations here (if needed)
  sorry

end total_spending_l22_22765


namespace Ashok_took_six_subjects_l22_22190

theorem Ashok_took_six_subjects
  (n : ℕ) -- number of subjects Ashok took
  (T : ℕ) -- total marks secured in those subjects
  (h_avg_n : T = n * 72) -- condition: average of marks in n subjects is 72
  (h_avg_5 : 5 * 74 = 370) -- condition: average of marks in 5 subjects is 74
  (h_6th_mark : 62 > 0) -- condition: the 6th subject's mark is 62
  (h_T : T = 370 + 62) -- condition: total marks including the 6th subject
  : n = 6 := 
sorry


end Ashok_took_six_subjects_l22_22190


namespace union_complement_l22_22107

open Set

-- Definitions for the universal set U and subsets A, B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 3, 4}
def B : Set ℕ := {1, 3}

-- Definition for the complement of A with respect to U
def CuA : Set ℕ := U \ A

-- Proof statement
theorem union_complement (U_def : U = {0, 1, 2, 3, 4})
                         (A_def : A = {0, 3, 4})
                         (B_def : B = {1, 3}) :
  (CuA ∪ B) = {1, 2, 3} := by
  sorry

end union_complement_l22_22107


namespace positive_value_of_a_l22_22725

-- Definitions based on the problem's conditions
variable (X : ℝ → Prop)
variable (normal_dist : X ∼ Normal 1 σ)
variable (a : ℝ)

-- The proposition that needs to be proven
theorem positive_value_of_a (h : P(X ≤ a ^ 2 - 1) = P(X > a - 3)) : a = 2 := by
  sorry

end positive_value_of_a_l22_22725


namespace simplify_sqrt_7_pow_6_l22_22644

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l22_22644


namespace arithmetic_sqrt_sqrt_16_l22_22666

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l22_22666


namespace arctan_sum_l22_22017

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l22_22017


namespace company_initial_bureaus_l22_22733

theorem company_initial_bureaus (B : ℕ) (offices : ℕ) (extra_bureaus : ℕ) 
  (h1 : offices = 14) 
  (h2 : extra_bureaus = 10) 
  (h3 : (B + extra_bureaus) % offices = 0) : 
  B = 8 := 
by
  sorry

end company_initial_bureaus_l22_22733


namespace solve_for_n_l22_22314

theorem solve_for_n (n : ℕ) : (3^n * 3^n * 3^n * 3^n = 81^2) → n = 2 :=
by
  sorry

end solve_for_n_l22_22314


namespace nine_digit_number_conditions_l22_22591

def nine_digit_number := 900900000

def remove_second_digit (n : ℕ) : ℕ := n / 100000000 * 10000000 + n % 10000000
def remove_third_digit (n : ℕ) : ℕ := n / 10000000 * 1000000 + n % 1000000
def remove_ninth_digit (n : ℕ) : ℕ := n / 10

theorem nine_digit_number_conditions :
  (remove_second_digit nine_digit_number) % 2 = 0 ∧
  (remove_third_digit nine_digit_number) % 3 = 0 ∧
  (remove_ninth_digit nine_digit_number) % 9 = 0 :=
by
  -- Proof steps would be included here.
  sorry

end nine_digit_number_conditions_l22_22591


namespace product_increase_by_13_exists_l22_22572

theorem product_increase_by_13_exists :
  ∃ a1 a2 a3 a4 a5 a6 a7 : ℕ,
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * (a1 * a2 * a3 * a4 * a5 * a6 * a7)) :=
by
  sorry

end product_increase_by_13_exists_l22_22572


namespace ellipse_foci_distance_l22_22394

theorem ellipse_foci_distance (a b : ℝ) (ha : a = 10) (hb : b = 8) :
  2 * Real.sqrt (a^2 - b^2) = 12 :=
by
  rw [ha, hb]
  -- Proof follows here, but we skip it using sorry.
  sorry

end ellipse_foci_distance_l22_22394


namespace length_of_AB_l22_22298

theorem length_of_AB {A B P Q : ℝ} (h1 : P = 3 / 5 * B)
                    (h2 : Q = 2 / 5 * A + 3 / 5 * B)
                    (h3 : dist P Q = 5) :
  dist A B = 25 :=
by sorry

end length_of_AB_l22_22298


namespace avg_price_two_returned_theorem_l22_22484

-- Defining the initial conditions given in the problem
def avg_price_of_five (price: ℕ) (packets: ℕ) : Prop :=
  packets = 5 ∧ price = 20

def avg_price_of_three_remaining (price: ℕ) (packets: ℕ) : Prop :=
  packets = 3 ∧ price = 12
  
def cost_of_packets (price packets: ℕ) := price * packets

noncomputable def avg_price_two_returned (total_initial_cost total_remaining_cost: ℕ) :=
  (total_initial_cost - total_remaining_cost) / 2

-- The Lean 4 proof statement
theorem avg_price_two_returned_theorem (p1 p2 p3 p4: ℕ):
  avg_price_of_five p1 5 →
  avg_price_of_three_remaining p2 3 →
  cost_of_packets p1 5 = 100 →
  cost_of_packets p2 3 = 36 →
  avg_price_two_returned 100 36 = 32 :=
by
  intros h1 h2 h3 h4
  sorry

end avg_price_two_returned_theorem_l22_22484


namespace arctan_sum_l22_22004

theorem arctan_sum : ∀ (a b : ℝ), 
  a = 3/7 → 
  b = 7/3 → 
  a * b = 1 → 
  a > 0 → 
  b > 0 → 
  Real.arctan a + Real.arctan b = Real.pi / 2 :=
by intros a b ha hb hab ha_pos hb_pos
   rw [ha, hb, hab]
   sorry

end arctan_sum_l22_22004


namespace find_four_digit_number_l22_22687

/-- 
  If there exists a positive integer M and M² both end in the same sequence of 
  five digits abcde in base 10 where a ≠ 0, 
  then the four-digit number abcd derived from M = 96876 is 9687.
-/
theorem find_four_digit_number
  (M : ℕ)
  (h_end_digits : (M % 100000) = (M * M % 100000))
  (h_first_digit_nonzero : 10000 ≤ M % 100000  ∧ M % 100000 < 100000)
  : (M = 96876 → (M / 10 % 10000 = 9687)) :=
by { sorry }

end find_four_digit_number_l22_22687


namespace simplify_sqrt_seven_pow_six_l22_22635

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l22_22635


namespace probability_two_8sided_dice_sum_perfect_square_l22_22144

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probabilityPerfectSquareSum (dice_sides : ℕ) (perfect_squares : List ℕ) : ℚ :=
  let outcomes := (dice_sides * dice_sides)
  let favorable_outcomes := perfect_squares.sum (λ ps, 
    (List.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd = ps) ((List.range dice_sides).product (List.range dice_sides))).length)
  favorable_outcomes /. outcomes

theorem probability_two_8sided_dice_sum_perfect_square :
  probabilityPerfectSquareSum 8 [4, 9, 16] = 3 / 16 := sorry

end probability_two_8sided_dice_sum_perfect_square_l22_22144


namespace sum_of_ages_l22_22838

theorem sum_of_ages (J M R : ℕ) (hJ : J = 10) (hM : M = J - 3) (hR : R = J + 2) : M + R = 19 :=
by
  sorry

end sum_of_ages_l22_22838


namespace investment_C_120000_l22_22921

noncomputable def investment_C (P_B P_A_difference : ℕ) (investment_A investment_B : ℕ) : ℕ :=
  let P_A := (P_B * investment_A) / investment_B
  let P_C := P_A + P_A_difference
  (P_C * investment_B) / P_B

theorem investment_C_120000
  (investment_A investment_B P_B P_A_difference : ℕ)
  (hA : investment_A = 8000)
  (hB : investment_B = 10000)
  (hPB : P_B = 1400)
  (hPA_difference : P_A_difference = 560) :
  investment_C P_B P_A_difference investment_A investment_B = 120000 :=
by
  sorry

end investment_C_120000_l22_22921


namespace problem_statement_l22_22209

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.range (n+1))

theorem problem_statement : ∃ r : ℕ, r < 13 ∧ (factorial 10) % 13 = r :=
by
  sorry

end problem_statement_l22_22209


namespace time_to_cross_bridge_l22_22920

noncomputable def train_crossing_time
  (length_train : ℕ)
  (length_bridge : ℕ)
  (speed_kmph : ℕ)
  (conversion_factor : ℚ) : ℚ :=
  (length_train + length_bridge) / (speed_kmph * conversion_factor)

theorem time_to_cross_bridge :
  train_crossing_time 135 240 45 (5 / 18) = 30 := by
  sorry

end time_to_cross_bridge_l22_22920


namespace locus_of_P_l22_22802

theorem locus_of_P
  (a b x y : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : (x ≠ 0 ∧ y ≠ 0))
  (h4 : x^2 / a^2 - y^2 / b^2 = 1) :
  (x / a)^2 - (y / b)^2 = ((a^2 + b^2) / (a^2 - b^2))^2 := by
  sorry

end locus_of_P_l22_22802


namespace black_white_ratio_extended_pattern_l22_22919

theorem black_white_ratio_extended_pattern
  (original_black : ℕ) (original_white : ℕ) (added_black : ℕ)
  (h1 : original_black = 10)
  (h2 : original_white = 26)
  (h3 : added_black = 20) :
  (original_black + added_black) / original_white = 30 / 26 :=
by sorry

end black_white_ratio_extended_pattern_l22_22919


namespace ending_number_of_multiples_l22_22468

theorem ending_number_of_multiples (n : ℤ) (h : 991 = (n - 100) / 10 + 1) : n = 10000 :=
by
  sorry

end ending_number_of_multiples_l22_22468


namespace sum_of_ages_l22_22834

-- Define the ages of Maggie, Juliet, and Ralph
def maggie_age : ℕ := by
  let juliet_age := 10
  let maggie_age := juliet_age - 3
  exact maggie_age

def ralph_age : ℕ := by
  let juliet_age := 10
  let ralph_age := juliet_age + 2
  exact ralph_age

-- The main theorem: The sum of Maggie's and Ralph's ages
theorem sum_of_ages : maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l22_22834


namespace rabbitAgeOrder_l22_22588

-- Define the ages of the rabbits as variables
variables (blue black red gray : ℕ)

-- Conditions based on the problem statement
noncomputable def rabbitConditions := 
  (blue ≠ max blue (max black (max red gray))) ∧  -- The blue-eyed rabbit is not the eldest
  (gray ≠ min blue (min black (min red gray))) ∧  -- The gray rabbit is not the youngest
  (red ≠ min blue (min black (min red gray))) ∧  -- The red-eyed rabbit is not the youngest
  (black > red) ∧ (gray > black)  -- The black rabbit is older than the red-eyed rabbit and younger than the gray rabbit

-- Required proof statement
theorem rabbitAgeOrder : rabbitConditions blue black red gray → gray > black ∧ black > red ∧ red > blue :=
by
  intro h
  sorry

end rabbitAgeOrder_l22_22588


namespace range_of_a_no_solution_inequality_l22_22959

theorem range_of_a_no_solution_inequality (a : ℝ) :
  (∀ x : ℝ, x + 2 > 3 → x < a) ↔ a ≤ 1 :=
by {
  sorry
}

end range_of_a_no_solution_inequality_l22_22959


namespace digit_in_ten_thousandths_place_of_five_over_thirty_two_l22_22893

def fractional_part_to_decimal (n d : ℕ) : ℚ := n / d

def ten_thousandths_place_digit (q : ℚ) : ℕ :=
  let decimal_str := (q * 10^5).round.to_string
  decimal_str.get_or_else (decimal_str.find_idx (>= ".") + 5) '0' - '0'

theorem digit_in_ten_thousandths_place_of_five_over_thirty_two :
  let q := fractional_part_to_decimal 5 32
  ten_thousandths_place_digit q = 2 :=
by
  -- The proof that computes the ten-thousandths place digit will be here.
  sorry

end digit_in_ten_thousandths_place_of_five_over_thirty_two_l22_22893


namespace masking_tape_problem_l22_22376

variable (width_other : ℕ)

theorem masking_tape_problem
  (h1 : ∀ w : ℕ, (2 * 4 + 2 * w) = 20)
  : width_other = 6 :=
by
  have h2 : 8 + 2 * width_other = 20 := h1 width_other
  sorry

end masking_tape_problem_l22_22376


namespace valid_orders_fraction_l22_22930

-- Define the conditions of the problem
variables (n : ℕ) (m : ℕ) (price : ℕ)

-- Initial conditions
constant eight_people : n = 8
constant four_100Ft : m = 4
constant four_200Ft : n - m = 4
constant ticket_price : price = 100
constant register_empty : ticket_price * 0 = 0

-- Define the main proof problem
theorem valid_orders_fraction : 
  ∀ (valid_orders : ℕ) (total_orders : ℕ),
  valid_orders = 14 → total_orders = 70 → 
  valid_orders * 5 = total_orders :=
by 
  intros ;
  simp only [valid_orders, total_orders] ;
  sorry

end valid_orders_fraction_l22_22930


namespace arctan_sum_l22_22005

theorem arctan_sum : ∀ (a b : ℝ), 
  a = 3/7 → 
  b = 7/3 → 
  a * b = 1 → 
  a > 0 → 
  b > 0 → 
  Real.arctan a + Real.arctan b = Real.pi / 2 :=
by intros a b ha hb hab ha_pos hb_pos
   rw [ha, hb, hab]
   sorry

end arctan_sum_l22_22005


namespace back_wheel_revolutions_calculation_l22_22999

def front_wheel_radius : ℝ := 3
def back_wheel_radius : ℝ := 0.5
def gear_ratio : ℝ := 2
def front_wheel_revolutions : ℕ := 50

noncomputable def back_wheel_revolutions (front_wheel_radius back_wheel_radius gear_ratio : ℝ) (front_wheel_revolutions : ℕ) : ℝ :=
  let front_circumference := 2 * Real.pi * front_wheel_radius
  let distance_traveled := front_circumference * front_wheel_revolutions
  let back_circumference := 2 * Real.pi * back_wheel_radius
  distance_traveled / back_circumference * gear_ratio

theorem back_wheel_revolutions_calculation :
  back_wheel_revolutions front_wheel_radius back_wheel_radius gear_ratio front_wheel_revolutions = 600 :=
sorry

end back_wheel_revolutions_calculation_l22_22999


namespace average_gpa_difference_2_l22_22450

def avg_gpa_6th_grader := 93
def avg_gpa_8th_grader := 91
def school_avg_gpa := 93

noncomputable def gpa_diff (gpa_7th_grader diff : ℝ) (avg6 avg8 school_avg : ℝ) := 
  gpa_7th_grader = avg6 + diff ∧ 
  (avg6 + gpa_7th_grader + avg8) / 3 = school_avg

theorem average_gpa_difference_2 (x : ℝ) : 
  (∃ G : ℝ, gpa_diff G x avg_gpa_6th_grader avg_gpa_8th_grader school_avg_gpa) → x = 2 :=
by
  sorry

end average_gpa_difference_2_l22_22450


namespace find_remainder_l22_22369

-- Definition of N based on given conditions
def N : ℕ := 44 * 432

-- Definition of next multiple of 432
def next_multiple_of_432 : ℕ := N + 432

-- Statement to prove the remainder when next_multiple_of_432 is divided by 39 is 12
theorem find_remainder : next_multiple_of_432 % 39 = 12 := 
by sorry

end find_remainder_l22_22369


namespace arithmetic_sqrt_of_sqrt_16_l22_22662

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22662


namespace factorial_mod_prime_l22_22230
-- Import all necessary libraries

-- State the conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The main problem statement
theorem factorial_mod_prime (n : ℕ) (h : n = 10) : factorial n % 13 = 7 := by
  sorry

end factorial_mod_prime_l22_22230


namespace fractions_integer_or_fractional_distinct_l22_22859

theorem fractions_integer_or_fractional_distinct (a b : Fin 6 → ℕ) (h_pos : ∀ i, 0 < a i ∧ 0 < b i)
  (h_irreducible : ∀ i, Nat.gcd (a i) (b i) = 1)
  (h_sum_eq : (Finset.univ : Finset (Fin 6)).sum a = (Finset.univ : Finset (Fin 6)).sum b) :
  ¬ ∀ i j : Fin 6, i ≠ j → ((a i / b i = a j / b j) ∨ (a i % b i / b i = a j % b j / b j)) :=
sorry

end fractions_integer_or_fractional_distinct_l22_22859


namespace coprime_boxes_equal_after_operations_not_coprime_boxes_never_equal_l22_22651

-- Defining the conditions
def m : ℕ := 5 -- Or any specific value you want to test
def n : ℕ := 3 -- Or any selected value less than m
def operation (boxes : vector ℕ m) (chosen : fin n → fin m) : vector ℕ m :=
  boxes.map_with_idx (λ i x, x + if i ∈ chosen then 1 else 0)

-- Part 1: Coprime case
theorem coprime_boxes_equal_after_operations
  (h_coprime : nat.coprime m n):
  ∃ t : ℕ, ∀ k : ℕ, ∀ boxes : vector ℕ m,
  ∃ boxes' : vector ℕ m, 
  (boxes' == vector.const (boxes.head + t * n) m) :=
sorry

-- Part 2: Not coprime case
theorem not_coprime_boxes_never_equal
  (h_not_coprime : ¬ nat.coprime m n):
  ∃ (initial_distribution : vector ℕ m),
  ∀ t : ℕ, ∃ boxes' : vector ℕ m, 
  ¬ ∀ i : fin m, boxes'.nth i = boxes'.nth 0 :=
sorry

end coprime_boxes_equal_after_operations_not_coprime_boxes_never_equal_l22_22651


namespace percentage_of_students_owning_cats_l22_22979

theorem percentage_of_students_owning_cats (N C : ℕ) (hN : N = 500) (hC : C = 75) :
  (C / N : ℚ) * 100 = 15 := by
  sorry

end percentage_of_students_owning_cats_l22_22979


namespace arithmetic_sequence_a12_bound_l22_22429

theorem arithmetic_sequence_a12_bound (a_1 d : ℤ) (h8 : a_1 + 7 * d ≥ 15) (h9 : a_1 + 8 * d ≤ 13) : 
  a_1 + 11 * d ≤ 7 :=
by
  sorry

end arithmetic_sequence_a12_bound_l22_22429


namespace negation_exists_x_squared_lt_zero_l22_22462

open Classical

theorem negation_exists_x_squared_lt_zero :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by 
  sorry

end negation_exists_x_squared_lt_zero_l22_22462


namespace arithmetic_square_root_of_sqrt_16_l22_22657

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l22_22657


namespace intersection_of_A_and_B_l22_22549

def set_A : Set ℝ := {x : ℝ | x^2 - 5 * x + 6 > 0}
def set_B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | x < 1} :=
sorry

end intersection_of_A_and_B_l22_22549


namespace scientific_notation_per_capita_GDP_l22_22360

theorem scientific_notation_per_capita_GDP (GDP : ℝ) (h : GDP = 104000): 
  GDP = 1.04 * 10^5 := 
by
  sorry

end scientific_notation_per_capita_GDP_l22_22360


namespace compare_31_17_compare_33_63_compare_82_26_compare_29_80_l22_22196

-- Definition and proof obligation for each comparison

theorem compare_31_17 : 31^11 < 17^14 := sorry

theorem compare_33_63 : 33^75 > 63^60 := sorry

theorem compare_82_26 : 82^33 > 26^44 := sorry

theorem compare_29_80 : 29^31 > 80^23 := sorry

end compare_31_17_compare_33_63_compare_82_26_compare_29_80_l22_22196


namespace union_sets_l22_22789

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}

theorem union_sets : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_sets_l22_22789


namespace platform_length_is_correct_l22_22723

noncomputable def length_of_platform (train1_speed_kmph : ℕ) (train2_speed_kmph : ℕ) (cross_time_s : ℕ) (platform_time_s : ℕ) : ℕ :=
  let train1_speed_mps := train1_speed_kmph * 5 / 18
  let train2_speed_mps := train2_speed_kmph * 5 / 18
  let relative_speed := train1_speed_mps + train2_speed_mps
  let total_distance := relative_speed * cross_time_s
  let train1_length := 2 * total_distance / 3
  let platform_length := train1_speed_mps * platform_time_s
  platform_length

theorem platform_length_is_correct : length_of_platform 48 42 12 45 = 600 :=
by
  sorry

end platform_length_is_correct_l22_22723


namespace probability_of_perfect_square_sum_l22_22153

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_l22_22153


namespace sum_a_n_eq_2014_l22_22954

def f (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n : ℤ)^2 else - (n : ℤ)^2

def a (n : ℕ) : ℤ :=
  f n + f (n + 1)

theorem sum_a_n_eq_2014 : (Finset.range 2014).sum a = 2014 :=
by
  sorry

end sum_a_n_eq_2014_l22_22954


namespace anna_more_candy_than_billy_l22_22516

theorem anna_more_candy_than_billy :
  let anna_candy_per_house := 14
  let billy_candy_per_house := 11
  let anna_houses := 60
  let billy_houses := 75
  let anna_total_candy := anna_candy_per_house * anna_houses
  let billy_total_candy := billy_candy_per_house * billy_houses
  anna_total_candy - billy_total_candy = 15 :=
by
  sorry

end anna_more_candy_than_billy_l22_22516


namespace janets_shampoo_days_l22_22283

-- Definitions from the problem conditions
def rose_shampoo := 1 / 3
def jasmine_shampoo := 1 / 4
def daily_usage := 1 / 12

-- Define the total shampoo and the days lasts
def total_shampoo := rose_shampoo + jasmine_shampoo
def days_lasts := total_shampoo / daily_usage

-- The theorem to be proved
theorem janets_shampoo_days : days_lasts = 7 :=
by sorry

end janets_shampoo_days_l22_22283


namespace arctan_sum_l22_22012

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l22_22012


namespace correct_option_D_l22_22417

noncomputable def total_students := 40
noncomputable def male_students := 25
noncomputable def female_students := 15
noncomputable def class_president := 1
noncomputable def prob_class_president := class_president / total_students
noncomputable def prob_class_president_from_females := 0

theorem correct_option_D
  (h1 : total_students = 40)
  (h2 : male_students = 25)
  (h3 : female_students = 15)
  (h4 : class_president = 1) :
  prob_class_president = 1 / 40 ∧ prob_class_president_from_females = 0 := 
by
  sorry

end correct_option_D_l22_22417


namespace find_enclosed_area_l22_22454

def area_square (side_length : ℕ) : ℕ :=
  side_length * side_length

def area_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem find_enclosed_area :
  let side1 := 3
  let side2 := 6
  let area1 := area_square side1
  let area2 := area_square side2
  let area_tri := 2 * area_triangle side1 side2
  area1 + area2 + area_tri = 63 :=
by
  sorry

end find_enclosed_area_l22_22454


namespace slower_speed_l22_22499

theorem slower_speed (x : ℝ) :
  (50 / x = 70 / 14) → x = 10 := by
  sorry

end slower_speed_l22_22499


namespace least_number_to_add_l22_22348

theorem least_number_to_add (a : ℕ) (b : ℕ) (n : ℕ) (h : a = 1056) (h1: b = 26) (h2 : n = 10) : 
  (a + n) % b = 0 := 
sorry

end least_number_to_add_l22_22348


namespace quadratic_roots_ratio_l22_22136

noncomputable def value_of_m (m : ℚ) : Prop :=
  ∃ r s : ℚ, r ≠ 0 ∧ s ≠ 0 ∧ (r / s = 3) ∧ (r + s = -9) ∧ (r * s = m)

theorem quadratic_roots_ratio (m : ℚ) (h : value_of_m m) : m = 243 / 16 :=
by
  sorry

end quadratic_roots_ratio_l22_22136


namespace exponent_rule_l22_22399

variable (a : ℝ) (m n : ℕ)

theorem exponent_rule (h1 : a^m = 3) (h2 : a^n = 2) : a^(m + n) = 6 :=
by
  sorry

end exponent_rule_l22_22399


namespace simplify_sqrt_seven_pow_six_proof_l22_22630

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l22_22630


namespace middle_digit_base_7_of_reversed_base_9_l22_22739

noncomputable def middle_digit_of_number_base_7 (N : ℕ) : ℕ :=
  let x := (N / 81) % 9  -- Extract the first digit in base-9
  let y := (N / 9) % 9   -- Extract the middle digit in base-9
  let z := N % 9         -- Extract the last digit in base-9
  -- Given condition: 81x + 9y + z = 49z + 7y + x
  let eq1 := 81 * x + 9 * y + z
  let eq2 := 49 * z + 7 * y + x
  let condition := eq1 = eq2 ∧ 0 ≤ y ∧ y < 7 -- y is a digit in base-7
  if condition then y else sorry

theorem middle_digit_base_7_of_reversed_base_9 (N : ℕ) :
  (∃ (x y z : ℕ), x < 9 ∧ y < 9 ∧ z < 9 ∧
  N = 81 * x + 9 * y + z ∧ N = 49 * z + 7 * y + x) → middle_digit_of_number_base_7 N = 0 :=
  by sorry

end middle_digit_base_7_of_reversed_base_9_l22_22739


namespace y_at_x_equals_8_l22_22086

theorem y_at_x_equals_8 (k : ℝ) (h1 : ∀ x y, y = k * x^(1/3))
    (h2 : 4 * real.sqrt 3 = k * 64^(1/3)) : k * 8^(1/3) = 2 * real.sqrt 3 :=
by
  sorry

end y_at_x_equals_8_l22_22086


namespace factorial_mod_prime_l22_22232
-- Import all necessary libraries

-- State the conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The main problem statement
theorem factorial_mod_prime (n : ℕ) (h : n = 10) : factorial n % 13 = 7 := by
  sorry

end factorial_mod_prime_l22_22232


namespace k3_to_fourth_equals_81_l22_22133

theorem k3_to_fourth_equals_81
  (h k : ℝ → ℝ)
  (h_cond : ∀ x, x ≥ 1 → h (k x) = x^3)
  (k_cond : ∀ x, x ≥ 1 → k (h x) = x^4)
  (k_81 : k 81 = 81) :
  k 3 ^ 4 = 81 :=
sorry

end k3_to_fourth_equals_81_l22_22133


namespace smallest_integer_square_l22_22480

theorem smallest_integer_square (x : ℤ) (h : x^2 = 2 * x + 75) : x = -7 :=
  sorry

end smallest_integer_square_l22_22480


namespace probability_of_perfect_square_sum_l22_22150

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_l22_22150


namespace daria_amount_owed_l22_22524

variable (savings : ℝ)
variable (couch_price : ℝ)
variable (table_price : ℝ)
variable (lamp_price : ℝ)
variable (total_cost : ℝ)
variable (amount_owed : ℝ)

theorem daria_amount_owed (h_savings : savings = 500)
                          (h_couch : couch_price = 750)
                          (h_table : table_price = 100)
                          (h_lamp : lamp_price = 50)
                          (h_total_cost : total_cost = couch_price + table_price + lamp_price)
                          (h_amount_owed : amount_owed = total_cost - savings) :
                          amount_owed = 400 :=
by
  sorry

end daria_amount_owed_l22_22524


namespace product_of_values_l22_22406

theorem product_of_values (x : ℚ) (hx : abs ((18 / x) + 4) = 3) :
  x = -18 ∨ x = -18 / 7 ∧ -18 * (-18 / 7) = 324 / 7 :=
by sorry

end product_of_values_l22_22406


namespace inequality_solution_set_is_correct_l22_22707

noncomputable def inequality_solution_set (x : ℝ) : Prop :=
  (3 * x - 1) / (2 - x) ≥ 1

theorem inequality_solution_set_is_correct :
  { x : ℝ | inequality_solution_set x } = { x : ℝ | 3 / 4 ≤ x ∧ x < 2 } :=
by sorry

end inequality_solution_set_is_correct_l22_22707


namespace lauren_annual_income_l22_22982

open Real

theorem lauren_annual_income (p : ℝ) (A : ℝ) (T : ℝ) :
  (T = (p + 0.45)/100 * A) →
  (T = (p/100) * 20000 + ((p + 1)/100) * 15000 + ((p + 3)/100) * (A - 35000)) →
  A = 36000 :=
by
  intros
  sorry

end lauren_annual_income_l22_22982


namespace three_x_plus_four_l22_22966

theorem three_x_plus_four (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  sorry

end three_x_plus_four_l22_22966


namespace trajectory_equation_l22_22240

variable (x y a b : ℝ)
variable (P : ℝ × ℝ := (0, -3))
variable (A : ℝ × ℝ := (a, 0))
variable (Q : ℝ × ℝ := (0, b))
variable (M : ℝ × ℝ := (x, y))

theorem trajectory_equation
  (h1 : A.1 = a)
  (h2 : A.2 = 0)
  (h3 : Q.1 = 0)
  (h4 : Q.2 > 0)
  (h5 : (P.1 - A.1) * (x - A.1) + (P.2 - A.2) * y = 0)
  (h6 : (x - A.1, y) = (-3/2 * (-x, b - y))) :
  y = (1 / 4) * x ^ 2 ∧ x ≠ 0 := by
    -- Sorry, proof omitted
    sorry

end trajectory_equation_l22_22240


namespace distinct_roots_condition_l22_22293

noncomputable def f (x c : ℝ) : ℝ := x^2 + 6*x + c

theorem distinct_roots_condition (c : ℝ) :
  (∀x : ℝ, f (f x c) = 0 → ∃ a b : ℝ, (a ≠ b) ∧ f x c = a * (x - b) * (x - c) ) →
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end distinct_roots_condition_l22_22293


namespace anna_gets_more_candy_l22_22517

theorem anna_gets_more_candy :
  let anna_pieces_per_house := 14
  let anna_houses := 60
  let billy_pieces_per_house := 11
  let billy_houses := 75
  let anna_total := anna_pieces_per_house * anna_houses
  let billy_total := billy_pieces_per_house * billy_houses
  anna_total - billy_total = 15 := by
    let anna_pieces_per_house := 14
    let anna_houses := 60
    let billy_pieces_per_house := 11
    let billy_houses := 75
    let anna_total := anna_pieces_per_house * anna_houses
    let billy_total := billy_pieces_per_house * billy_houses
    have h1 : anna_total = 14 * 60 := rfl
    have h2 : billy_total = 11 * 75 := rfl
    sorry

end anna_gets_more_candy_l22_22517


namespace tree_height_when_planted_l22_22472

def initial_height (current_height : ℕ) (growth_rate : ℕ) (current_age : ℕ) (initial_age : ℕ) : ℕ :=
  current_height - (current_age - initial_age) * growth_rate

theorem tree_height_when_planted :
  initial_height 23 3 7 1 = 5 :=
by
  sorry

end tree_height_when_planted_l22_22472


namespace fg_eq_neg7_l22_22081

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem fg_eq_neg7 : f (g 2) = -7 :=
  by
    sorry

end fg_eq_neg7_l22_22081


namespace weight_loss_percentage_l22_22357

theorem weight_loss_percentage 
  (weight_before weight_after : ℝ) 
  (h_before : weight_before = 800) 
  (h_after : weight_after = 640) : 
  (weight_before - weight_after) / weight_before * 100 = 20 := 
by
  sorry

end weight_loss_percentage_l22_22357


namespace sum_of_ages_l22_22837

theorem sum_of_ages (J M R : ℕ) (hJ : J = 10) (hM : M = J - 3) (hR : R = J + 2) : M + R = 19 :=
by
  sorry

end sum_of_ages_l22_22837


namespace micheal_item_count_l22_22849

theorem micheal_item_count : ∃ a b c : ℕ, a + b + c = 50 ∧ 60 * a + 500 * b + 400 * c = 10000 ∧ a = 30 :=
  by
    sorry

end micheal_item_count_l22_22849


namespace arithmetic_sqrt_of_sqrt_16_l22_22663

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22663


namespace circle_y_axis_intersection_range_l22_22090

theorem circle_y_axis_intersection_range (m : ℝ) : (4 - 4 * (m + 6) > 0) → (-2 < 0) → (m + 6 > 0) → (-6 < m ∧ m < -5) :=
by 
  intros h1 h2 h3 
  sorry

end circle_y_axis_intersection_range_l22_22090


namespace car_speed_in_mph_l22_22910

-- Defining the given conditions
def fuel_efficiency : ℚ := 56 -- kilometers per liter
def gallons_to_liters : ℚ := 3.8 -- liters per gallon
def kilometers_to_miles : ℚ := 1 / 1.6 -- miles per kilometer
def fuel_decrease_gallons : ℚ := 3.9 -- gallons
def time_hours : ℚ := 5.7 -- hours

-- Using definitions to compute the speed
theorem car_speed_in_mph :
  (fuel_decrease_gallons * gallons_to_liters * fuel_efficiency * kilometers_to_miles) / time_hours = 91 :=
sorry

end car_speed_in_mph_l22_22910


namespace expression_value_l22_22242

variable (m n : ℝ)

theorem expression_value (hm : 3 * m ^ 2 + 5 * m - 3 = 0)
                         (hn : 3 * n ^ 2 - 5 * n - 3 = 0)
                         (hneq : m * n ≠ 1) :
                         (1 / n ^ 2) + (m / n) - (5 / 3) * m = 25 / 9 :=
by {
  sorry
}

end expression_value_l22_22242


namespace units_digit_product_l22_22479

theorem units_digit_product :
  ((734^99 + 347^83) % 10) * ((956^75 - 214^61) % 10) % 10 = 4 := by
  sorry

end units_digit_product_l22_22479


namespace inscribed_circle_radius_square_l22_22732

theorem inscribed_circle_radius_square (ER RF GS SH : ℝ) (r : ℝ) 
  (hER : ER = 23) (hRF : RF = 34) (hGS : GS = 42) (hSH : SH = 28)
  (h_tangent : ∀ t, t = r * r * (70 * t - 87953)) :
  r^2 = 87953 / 70 :=
by
  sorry

end inscribed_circle_radius_square_l22_22732


namespace calculate_expression_l22_22192

theorem calculate_expression (a b c : ℤ) (ha : a = 3) (hb : b = 7) (hc : c = 2) :
  ((a * b - c) - (a + b * c)) - ((a * c - b) - (a - b * c)) = -8 :=
by
  rw [ha, hb, hc]  -- Substitute a, b, c with 3, 7, 2 respectively
  sorry  -- Placeholder for the proof

end calculate_expression_l22_22192


namespace correctness_of_solution_set_l22_22131

-- Define the set of real numbers satisfying the inequality
def solution_set : Set ℝ := { x | 3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9 }

-- Define the expected solution set derived from the problem
def expected_solution_set : Set ℝ := { x | -1 < x ∧ x ≤ 1 } ∪ { x | 2.5 < x ∧ x < 4.5 }

-- The proof statement
theorem correctness_of_solution_set : solution_set = expected_solution_set :=
  sorry

end correctness_of_solution_set_l22_22131


namespace root_equality_l22_22106

theorem root_equality (p q : ℝ) (h1 : 1 + p + q = (2 - 2 * q) / p) (h2 : 1 + p + q = (1 - p + q) / q) :
  p + q = 1 :=
sorry

end root_equality_l22_22106


namespace initial_boxes_l22_22446

theorem initial_boxes (x : ℕ) (h : x + 6 = 14) : x = 8 :=
by sorry

end initial_boxes_l22_22446


namespace function_has_zero_in_interval_l22_22869

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 2 * x - 2

theorem function_has_zero_in_interval (x : ℝ) (h1 : 2 < x) (h2 : x < 3) : ∃ c ∈ (2,3), f c = 0 :=
by
  sorry

end function_has_zero_in_interval_l22_22869


namespace purely_periodic_fraction_period_length_divisible_l22_22103

noncomputable def purely_periodic_fraction (p q n : ℕ) : Prop :=
  ∃ (r : ℕ), 10 ^ n - 1 = r * q ∧ (∃ (k : ℕ), q * (10 ^ (n * k)) ∣ p)

theorem purely_periodic_fraction_period_length_divisible
  (p q n : ℕ) (hq : ¬ (2 ∣ q) ∧ ¬ (5 ∣ q)) (hpq : p < q) (hn : 10 ^ n - 1 ∣ q) :
  purely_periodic_fraction p q n :=
by
  sorry

end purely_periodic_fraction_period_length_divisible_l22_22103


namespace optimal_solution_range_l22_22395

theorem optimal_solution_range (a : ℝ) (x y : ℝ) :
  (x + y - 4 ≥ 0) → (2 * x - y - 5 ≤ 0) → (x = 1) → (y = 3) →
  (-2 < a) ∧ (a < 1) :=
by
  intros h1 h2 hx hy
  sorry

end optimal_solution_range_l22_22395


namespace impossible_coins_l22_22595

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l22_22595


namespace probability_of_perfect_square_sum_on_two_8_sided_dice_l22_22157

-- Define the dice and possible sums
def is_sum_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def pairs_rolled : finset (ℕ × ℕ) :=
  finset.univ.filter (λ p, p.1 + p.2 ∈ {4, 9, 16})

theorem probability_of_perfect_square_sum_on_two_8_sided_dice :
  (pairs_rolled.card : ℚ) / 64 = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_on_two_8_sided_dice_l22_22157


namespace ice_cream_stacks_l22_22126

theorem ice_cream_stacks :
  let ice_cream_flavors := ["vanilla", "chocolate", "strawberry", "cherry", "banana"]
  let ways_to_stack := Nat.factorial ice_cream_flavors.length
  ways_to_stack = 120 :=
by
  let ice_cream_flavors := ["vanilla", "chocolate", "strawberry", "cherry", "banana"]
  let ways_to_stack := Nat.factorial ice_cream_flavors.length
  show (ways_to_stack = 120)
  sorry

end ice_cream_stacks_l22_22126


namespace num_solutions_l22_22385

-- Define the problem and the condition
def matrix_eq (x : ℝ) : Prop :=
  3 * x^2 - 4 * x = 7

-- Define the main theorem to prove the number of solutions
theorem num_solutions : ∃! x : ℝ, matrix_eq x :=
sorry

end num_solutions_l22_22385


namespace repeating_decimal_as_fraction_l22_22201

-- Define the repeating decimal
def repeating_decimal := 3 + (127 / 999)

-- State the goal
theorem repeating_decimal_as_fraction : repeating_decimal = (3124 / 999) := 
by 
  sorry

end repeating_decimal_as_fraction_l22_22201


namespace cube_expansion_l22_22342

theorem cube_expansion : 101^3 + 3 * 101^2 + 3 * 101 + 1 = 1061208 :=
by
  sorry

end cube_expansion_l22_22342


namespace average_time_correct_l22_22903

-- Define the times for each runner
def y_time : ℕ := 58
def z_time : ℕ := 26
def w_time : ℕ := 2 * z_time

-- Define the number of runners
def num_runners : ℕ := 3

-- Calculate the summed time of all runners
def total_time : ℕ := y_time + z_time + w_time

-- Calculate the average time
def average_time : ℚ := total_time / num_runners

-- Statement of the proof problem
theorem average_time_correct : average_time = 45.33 := by
  -- The proof would go here
  sorry

end average_time_correct_l22_22903


namespace average_weight_men_women_l22_22880

theorem average_weight_men_women (n_men n_women : ℕ) (avg_weight_men avg_weight_women : ℚ)
  (h_men : n_men = 8) (h_women : n_women = 6) (h_avg_weight_men : avg_weight_men = 190)
  (h_avg_weight_women : avg_weight_women = 120) :
  (n_men * avg_weight_men + n_women * avg_weight_women) / (n_men + n_women) = 160 := 
by
  sorry

end average_weight_men_women_l22_22880


namespace arithmetic_sqrt_of_sqrt_16_l22_22655

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22655


namespace fewer_cans_l22_22898

theorem fewer_cans (sarah_yesterday lara_more alex_yesterday sarah_today lara_today alex_today : ℝ)
  (H1 : sarah_yesterday = 50.5)
  (H2 : lara_more = 30.3)
  (H3 : alex_yesterday = 90.2)
  (H4 : sarah_today = 40.7)
  (H5 : lara_today = 70.5)
  (H6 : alex_today = 55.3) :
  (sarah_yesterday + (sarah_yesterday + lara_more) + alex_yesterday) - (sarah_today + lara_today + alex_today) = 55 :=
by {
  -- Sorry to skip the proof
  sorry
}

end fewer_cans_l22_22898


namespace maximize_profit_l22_22424

def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8
def total_devices : ℕ := 50

def profit (x : ℕ) : ℝ := (price_A - cost_A) * x + (price_B - cost_B) * (total_devices - x)

def functional_relationship (x : ℕ) : ℝ := -0.1 * x + 20

def purchase_condition (x : ℕ) : Prop := 4 * x ≥ total_devices - x

theorem maximize_profit :
    functional_relationship (10) = 19 ∧ 
    (∀ x : ℕ, purchase_condition x → functional_relationship x ≤ 19) :=
by {
    -- Proof omitted
    sorry
}

end maximize_profit_l22_22424


namespace simplify_sqrt_power_l22_22626

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l22_22626


namespace arithmetic_mean_inequality_l22_22291

variable (a b c : ℝ)

-- conditions
def m := (a + b + c) / 3

-- conjecture
theorem arithmetic_mean_inequality (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) :
  (Real.sqrt (a + Real.sqrt (b + Real.sqrt c)) +
   Real.sqrt (b + Real.sqrt (c + Real.sqrt a)) +
   Real.sqrt (c + Real.sqrt (a + Real.sqrt b))) ≤
  3 * Real.sqrt (m + Real.sqrt (m + Real.sqrt m)) :=
sorry

end arithmetic_mean_inequality_l22_22291


namespace tangent_line_to_circle_l22_22761

theorem tangent_line_to_circle : 
  ∀ (ρ θ : ℝ), (ρ = 4 * Real.sin θ) → (∃ ρ θ : ℝ, ρ * Real.cos θ = 2) :=
by
  sorry

end tangent_line_to_circle_l22_22761


namespace sum_of_ages_l22_22835

-- Define the ages of Maggie, Juliet, and Ralph
def maggie_age : ℕ := by
  let juliet_age := 10
  let maggie_age := juliet_age - 3
  exact maggie_age

def ralph_age : ℕ := by
  let juliet_age := 10
  let ralph_age := juliet_age + 2
  exact ralph_age

-- The main theorem: The sum of Maggie's and Ralph's ages
theorem sum_of_ages : maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l22_22835


namespace problem_inequality_l22_22871

theorem problem_inequality (a b : ℝ) (n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : 1 < a * b) (h_n : 2 ≤ n) :
  (a + b)^n > a^n + b^n + 2^n - 2 :=
sorry

end problem_inequality_l22_22871


namespace factorial_mod_10_eq_6_l22_22217

theorem factorial_mod_10_eq_6 : (10! % 13) = 6 := by
  sorry

end factorial_mod_10_eq_6_l22_22217


namespace compare_exponents_product_of_roots_l22_22247

noncomputable def f (x : ℝ) (a : ℝ) := (Real.log x) / (x + a)

theorem compare_exponents : (2016 : ℝ) ^ 2017 > (2017 : ℝ) ^ 2016 :=
sorry

theorem product_of_roots (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f x1 0 = k) (h3 : f x2 0 = k) : 
  x1 * x2 > Real.exp 2 :=
sorry

end compare_exponents_product_of_roots_l22_22247


namespace no_real_solution_l22_22019

theorem no_real_solution (x y : ℝ) (hx : x^2 = 1 + 1 / y^2) (hy : y^2 = 1 + 1 / x^2) : false :=
by
  sorry

end no_real_solution_l22_22019


namespace rectangle_length_is_4_l22_22356

theorem rectangle_length_is_4 (w l : ℝ) (h_length : l = w + 3) (h_area : l * w = 4) : l = 4 := 
sorry

end rectangle_length_is_4_l22_22356


namespace Amy_current_age_l22_22997

def Mark_age_in_5_years : ℕ := 27
def years_in_future : ℕ := 5
def age_difference : ℕ := 7

theorem Amy_current_age : ∃ (Amy_age : ℕ), Amy_age = 15 :=
  by
    let Mark_current_age := Mark_age_in_5_years - years_in_future
    let Amy_age := Mark_current_age - age_difference
    use Amy_age
    sorry

end Amy_current_age_l22_22997


namespace percentage_difference_l22_22270

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.70 * x) (h2 : z = 1.50 * y) :
   x / z = 39.22 / 100 :=
by
  sorry

end percentage_difference_l22_22270


namespace find_H2SO4_moles_l22_22384

-- Let KOH, H2SO4, and KHSO4 represent the moles of each substance in the reaction.
variable (KOH H2SO4 KHSO4 : ℕ)

-- Conditions provided in the problem
def KOH_moles : ℕ := 2
def KHSO4_moles (H2SO4 : ℕ) : ℕ := H2SO4

-- Main statement, we need to prove that given the conditions,
-- 2 moles of KOH and 2 moles of KHSO4 imply 2 moles of H2SO4.
theorem find_H2SO4_moles (KOH_sufficient : KOH = KOH_moles) 
  (KHSO4_produced : KHSO4 = KOH) : KHSO4_moles H2SO4 = 2 := 
sorry

end find_H2SO4_moles_l22_22384


namespace number_2digit_smaller_than_35_l22_22071

/--
Prove that the number of two-digit numbers where at least one digit
is smaller than the corresponding digit in 35 is exactly 55.
-/
theorem number_2digit_smaller_than_35 : 
  (Finset.filter (λ n : ℕ, let d1 := n / 10, d2 := n % 10 in d1 < 3 ∨ d2 < 5) (Finset.range' 10 100)).card = 55 := 
by
  sorry

end number_2digit_smaller_than_35_l22_22071


namespace probability_of_perfect_square_sum_l22_22161

-- Define the conditions
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def perfect_squares_in_range : Finset ℕ := {4, 9, 16}
def total_outcomes : ℕ := 8 * 8  -- total number of outcomes when rolling two dice

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  (dice_faces.product dice_faces).filter (λ pair => perfect_squares_in_range.member (pair.1 + pair.2)).card

-- Calculate the probability of getting a perfect square sum
noncomputable def probability_perfect_square_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- The statement to prove that the probability is 3/16
theorem probability_of_perfect_square_sum :
  probability_perfect_square_sum = 3 / 16 :=
begin
  sorry
end

end probability_of_perfect_square_sum_l22_22161


namespace oblique_prism_volume_l22_22134

noncomputable def volume_of_oblique_prism 
  (a b c : ℝ) (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  : ℝ :=
  a * b * c / Real.sqrt (1 + (Real.cos α / Real.sin α)^2 + (Real.cos β / Real.sin β)^2)

theorem oblique_prism_volume 
  (a b c : ℝ) (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  : volume_of_oblique_prism a b c α β hα hβ = a * b * c / Real.sqrt (1 + (Real.cos α / Real.sin α)^2 + (Real.cos β / Real.sin β)^2) := 
by
  -- Proof will be completed here
  sorry

end oblique_prism_volume_l22_22134


namespace fraction_of_calls_processed_by_team_B_l22_22170

theorem fraction_of_calls_processed_by_team_B
  (C_B : ℕ) -- the number of calls processed by each member of team B
  (B : ℕ)  -- the number of call center agents in team B
  (C_A : ℕ := C_B / 5) -- each member of team A processes 1/5 the number of calls as each member of team B
  (A : ℕ := 5 * B / 8) -- team A has 5/8 as many agents as team B
: 
  (B * C_B) / ((A * C_A) + (B * C_B)) = (8 / 9 : ℚ) :=
sorry

end fraction_of_calls_processed_by_team_B_l22_22170


namespace unique_a_exists_for_prime_p_l22_22758

theorem unique_a_exists_for_prime_p (p : ℕ) [Fact p.Prime] :
  (∃! (a : ℕ), a ∈ Finset.range (p + 1) ∧ (a^3 - 3*a + 1) % p = 0) ↔ p = 3 := by
  sorry

end unique_a_exists_for_prime_p_l22_22758


namespace probability_of_perfect_square_sum_l22_22160

-- Define the conditions
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def perfect_squares_in_range : Finset ℕ := {4, 9, 16}
def total_outcomes : ℕ := 8 * 8  -- total number of outcomes when rolling two dice

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  (dice_faces.product dice_faces).filter (λ pair => perfect_squares_in_range.member (pair.1 + pair.2)).card

-- Calculate the probability of getting a perfect square sum
noncomputable def probability_perfect_square_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- The statement to prove that the probability is 3/16
theorem probability_of_perfect_square_sum :
  probability_perfect_square_sum = 3 / 16 :=
begin
  sorry
end

end probability_of_perfect_square_sum_l22_22160


namespace value_of_a_plus_b_l22_22777

theorem value_of_a_plus_b (a b : Int) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
by
  sorry

end value_of_a_plus_b_l22_22777


namespace matrix_mult_7_l22_22941

theorem matrix_mult_7 (M : Matrix (Fin 3) (Fin 3) ℝ) (v : Fin 3 → ℝ) : 
  (∀ v, M.mulVec v = (7 : ℝ) • v) ↔ M = 7 • 1 :=
by
  sorry

end matrix_mult_7_l22_22941


namespace solve_alcohol_mixture_problem_l22_22711

theorem solve_alcohol_mixture_problem (x y : ℝ) 
(h1 : x + y = 18) 
(h2 : 0.75 * x + 0.15 * y = 9) 
: x = 10.5 ∧ y = 7.5 :=
by 
  sorry

end solve_alcohol_mixture_problem_l22_22711


namespace number_2digit_smaller_than_35_l22_22073

/--
Prove that the number of two-digit numbers where at least one digit
is smaller than the corresponding digit in 35 is exactly 55.
-/
theorem number_2digit_smaller_than_35 : 
  (Finset.filter (λ n : ℕ, let d1 := n / 10, d2 := n % 10 in d1 < 3 ∨ d2 < 5) (Finset.range' 10 100)).card = 55 := 
by
  sorry

end number_2digit_smaller_than_35_l22_22073


namespace ratio_a_c_l22_22463

theorem ratio_a_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 1) 
  (h3 : d / b = 2 / 5) : 
  a / c = 25 / 32 := 
by sorry

end ratio_a_c_l22_22463


namespace least_three_digit_product_18_l22_22719

theorem least_three_digit_product_18 : ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ (∃ (H T U : ℕ), H ≠ 0 ∧ N = 100 * H + 10 * T + U ∧ H * T * U = 18) ∧ ∀ M : ℕ, (100 ≤ M ∧ M ≤ 999 ∧ (∃ (H T U : ℕ), H ≠ 0 ∧ M = 100 * H + 10 * T + U ∧ H * T * U = 18)) → N ≤ M :=
    sorry

end least_three_digit_product_18_l22_22719


namespace circle_radius_l22_22699

theorem circle_radius (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end circle_radius_l22_22699


namespace kitten_length_after_4_months_l22_22508

theorem kitten_length_after_4_months
  (initial_length : ℕ)
  (doubled_length_2_weeks : ℕ)
  (final_length_4_months : ℕ)
  (h1 : initial_length = 4)
  (h2 : doubled_length_2_weeks = initial_length * 2)
  (h3 : final_length_4_months = doubled_length_2_weeks * 2) :
  final_length_4_months = 16 := 
by
  sorry

end kitten_length_after_4_months_l22_22508


namespace Cary_final_salary_l22_22752

def initial_salary : ℝ := 10
def raise_percentage : ℝ := 0.20
def cut_percentage : ℝ := 0.75

theorem Cary_final_salary :
  let raise_amount := raise_percentage * initial_salary in
  let new_salary_after_raise := initial_salary + raise_amount in
  let final_salary := cut_percentage * new_salary_after_raise in
  final_salary = 9 := by
  sorry

end Cary_final_salary_l22_22752


namespace optimal_selling_price_maximizes_profit_l22_22706

/-- The purchase price of a certain product is 40 yuan. -/
def cost_price : ℝ := 40

/-- At a selling price of 50 yuan, 50 units can be sold. -/
def initial_selling_price : ℝ := 50
def initial_quantity_sold : ℝ := 50

/-- If the selling price increases by 1 yuan, the sales volume decreases by 1 unit. -/
def price_increase_effect (x : ℝ) : ℝ := initial_selling_price + x
def quantity_decrease_effect (x : ℝ) : ℝ := initial_quantity_sold - x

/-- The revenue function. -/
def revenue (x : ℝ) : ℝ := (price_increase_effect x) * (quantity_decrease_effect x)

/-- The cost function. -/
def cost (x : ℝ) : ℝ := cost_price * (quantity_decrease_effect x)

/-- The profit function. -/
def profit (x : ℝ) : ℝ := revenue x - cost x

/-- The proof that the optimal selling price to maximize profit is 70 yuan. -/
theorem optimal_selling_price_maximizes_profit : price_increase_effect 20 = 70 :=
by
  sorry

end optimal_selling_price_maximizes_profit_l22_22706


namespace lina_walk_probability_l22_22487

/-- Total number of gates -/
def num_gates : ℕ := 20

/-- Distance between adjacent gates in feet -/
def gate_distance : ℕ := 50

/-- Maximum distance in feet Lina can walk to be within the desired range -/
def max_walk_distance : ℕ := 200

/-- Number of gates Lina can move within the max walk distance -/
def max_gates_within_distance : ℕ := max_walk_distance / gate_distance

/-- Total possible gate pairs for initial and new gate selection -/
def total_possible_pairs : ℕ := num_gates * (num_gates - 1)

/-- Total number of favorable gate pairs where walking distance is within the allowed range -/
def total_favorable_pairs : ℕ :=
  let edge_favorable (g : ℕ) := if g = 1 ∨ g = num_gates then 4
                                else if g = 2 ∨ g = num_gates - 1 then 5
                                else if g = 3 ∨ g = num_gates - 2 then 6
                                else if g = 4 ∨ g = num_gates - 3 then 7 else 8
  (edge_favorable 1) + (edge_favorable 2) + (edge_favorable 3) +
  (edge_favorable 4) + (num_gates - 8) * 8

/-- Probability that Lina walks 200 feet or less expressed as a reduced fraction -/
def probability_within_distance : ℚ :=
  (total_favorable_pairs : ℚ) / (total_possible_pairs : ℚ)

/-- p and q components of the fraction representing the probability -/
def p := 7
def q := 19

/-- Sum of p and q -/
def p_plus_q : ℕ := p + q

theorem lina_walk_probability : p_plus_q = 26 := by sorry

end lina_walk_probability_l22_22487


namespace total_spent_l22_22502

def spending (A B C : ℝ) : Prop :=
  (A = (13 / 10) * B) ∧
  (C = (4 / 5) * B) ∧
  (A = C + 15)

theorem total_spent (A B C : ℝ) (h : spending A B C) : A + B + C = 93 :=
by
  sorry

end total_spent_l22_22502


namespace molecular_weight_CaO_l22_22163

def atomic_weight_Ca : Float := 40.08
def atomic_weight_O : Float := 16.00

def molecular_weight (atoms : List (String × Float)) : Float :=
  atoms.foldr (fun (_, w) acc => w + acc) 0.0

theorem molecular_weight_CaO :
  molecular_weight [("Ca", atomic_weight_Ca), ("O", atomic_weight_O)] = 56.08 :=
by
  sorry

end molecular_weight_CaO_l22_22163


namespace at_least_one_of_p_or_q_true_l22_22415

variable (p q : Prop)

theorem at_least_one_of_p_or_q_true (h : ¬(p ∨ q) = false) : p ∨ q :=
by 
  sorry

end at_least_one_of_p_or_q_true_l22_22415


namespace boston_snow_l22_22364

noncomputable def initial_snow : ℝ := 0.5
noncomputable def second_day_snow_inch : ℝ := 8 / 12
noncomputable def next_two_days_melt_inch : ℝ := 2 / 12
noncomputable def fifth_day_snow_factor : ℝ := 2

theorem boston_snow : 
  let second_day_snow := initial_snow + second_day_snow_inch,
      snow_after_melt := second_day_snow - next_two_days_melt_inch,
      fifth_day_snow := fifth_day_snow_factor * initial_snow
  in snow_after_melt + fifth_day_snow = 2 := 
by
  sorry

end boston_snow_l22_22364


namespace finite_spheres_block_light_l22_22471

-- Defining the primary entities involved
def lamp_origin : ℝ^d := 0

def is_ray_blocked_by_sphere (r : ℝ) (c : ℝ^d) (ray : ℝ^d → Prop) : Prop :=
  ∀ x, ray x → ‖x - c‖ ≥ r

def finite_spheres_exists (r : ℝ) (S : set (ℝ^d)) : Prop :=
  ∃ (sphere_centers : finset (ℝ^d)), 
  ∀ ray, (∀ x, ¬is_ray_blocked_by_sphere r lamp_origin ray) → 
  ∃ c ∈ sphere_centers, is_ray_blocked_by_sphere r c ray

-- The main theorem statement
theorem finite_spheres_block_light (r : ℝ) (S : set (ℝ^d)) (hr : r < 1 / 2) :
  finite_spheres_exists r S := 
sorry

end finite_spheres_block_light_l22_22471


namespace number_of_cows_on_farm_l22_22116

theorem number_of_cows_on_farm :
  (∀ (cows_per_week : ℤ) (six_cows_milk : ℤ) (total_milk : ℤ) (weeks : ℤ),
    cows_per_week = 6 → 
    six_cows_milk = 108 →
    total_milk = 2160 →
    weeks = 5 →
    (total_milk / (six_cows_milk / cows_per_week * weeks)) = 24) :=
by
  intros cows_per_week six_cows_milk total_milk weeks h1 h2 h3 h4
  have h_cow_milk_per_week : six_cows_milk / cows_per_week = 18 := by sorry
  have h_cow_milk_per_five_weeks : (six_cows_milk / cows_per_week) * weeks = 90 := by sorry
  have h_total_cows : total_milk / ((six_cows_milk / cows_per_week) * weeks) = 24 := by sorry
  exact h_total_cows

end number_of_cows_on_farm_l22_22116


namespace radius_of_circle_l22_22692

theorem radius_of_circle (r : ℝ) (C : ℝ) (A : ℝ) (h1 : 3 * C = 2 * A) 
  (h2 : C = 2 * Real.pi * r) (h3 : A = Real.pi * r^2) : 
  r = 3 :=
by 
  sorry

end radius_of_circle_l22_22692


namespace sufficient_not_necessary_l22_22236

variable (a : ℝ)

theorem sufficient_not_necessary :
  (a > 1 → a^2 > a) ∧ (¬(a > 1) ∧ a^2 > a → a < 0) :=
by
  sorry

end sufficient_not_necessary_l22_22236


namespace impossible_coins_l22_22614

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l22_22614


namespace impossible_coins_l22_22616

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l22_22616


namespace problem1_problem2_l22_22239

open Real

noncomputable def f (x : ℝ) : ℝ :=
  cos (2 * x + 2 * π / 3) + 2 * (cos x)^2

theorem problem1 : (∀ x, f(x) ≤ 2) ∧ (∃ x, f(x) = 2) ∧
  (∀ x, f(x) = 2 → ∃ k : ℤ, x = k * π - π / 6) := 
by sorry

noncomputable def fA (A : ℝ) : ℝ :=
  cos (2 * A + 2 * π / 3) + 2 * (cos A)^2

theorem problem2 (A a b c : ℝ):
  fA A = 3 / 2 ∧ b + c = 2 →
  ∃ a, a = sqrt 3 := 
by sorry

end problem1_problem2_l22_22239


namespace problem_statement_l22_22772

noncomputable def M (x y : ℝ) : ℝ := max x y
noncomputable def m (x y : ℝ) : ℝ := min x y

theorem problem_statement {p q r s t : ℝ} (h1 : p < q) (h2 : q < r) (h3 : r < s) (h4 : s < t) :
  M (M p (m q r)) (m s (m p t)) = q :=
by
  sorry

end problem_statement_l22_22772


namespace kate_change_l22_22287

def candyCost : ℝ := 0.54
def amountGiven : ℝ := 1.00
def change (amountGiven candyCost : ℝ) : ℝ := amountGiven - candyCost

theorem kate_change : change amountGiven candyCost = 0.46 := by
  sorry

end kate_change_l22_22287


namespace carousel_seat_count_l22_22124

theorem carousel_seat_count
  (total_seats : ℕ)
  (colors : ℕ → Prop)
  (num_yellow num_blue num_red : ℕ)
  (num_colors : ∀ n, colors n → n = num_yellow ∨ n = num_blue ∨ n = num_red)
  (opposite_blue_red_7_3 : ∀ n, n = 7 ↔ n + 50 = 3)
  (opposite_yellow_red_7_23 : ∀ n, n = 7 ↔ n + 50 = 23)
  (total := 100)
 :
 (num_yellow = 34 ∧ num_blue = 20 ∧ num_red = 46) :=
by
  sorry

end carousel_seat_count_l22_22124


namespace find_value_l22_22176

theorem find_value (x v : ℝ) (h1 : 0.80 * x + v = x) (h2 : x = 100) : v = 20 := by
    sorry

end find_value_l22_22176


namespace domain_of_f_l22_22563

theorem domain_of_f (m : ℝ) : (∀ x : ℝ, mx^2 + mx + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end domain_of_f_l22_22563


namespace xy_product_l22_22650

theorem xy_product (x y : ℝ) (h : x^2 + y^2 - 22*x - 20*y + 221 = 0) : x * y = 110 := 
sorry

end xy_product_l22_22650


namespace prob_rain_all_days_l22_22325

/--
The probability of rain on Friday, Saturday, and Sunday is given by 
0.40, 0.60, and 0.35 respectively.
We want to prove that the combined probability of rain on all three days,
assuming independence, is 8.4%.
-/
theorem prob_rain_all_days :
  let p_friday := 0.40
  let p_saturday := 0.60
  let p_sunday := 0.35
  p_friday * p_saturday * p_sunday = 0.084 :=
by
  sorry

end prob_rain_all_days_l22_22325


namespace amplitude_combined_wave_l22_22421

noncomputable def y1 (t : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y2 (t : ℝ) : ℝ := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) : ℝ := y1 t + y2 t
noncomputable def amplitude : ℝ := 3 * Real.sqrt 5

theorem amplitude_combined_wave : ∀ t : ℝ, ∃ A : ℝ, A = 3 * Real.sqrt 5 :=
by
  intro t
  use amplitude
  exact sorry

end amplitude_combined_wave_l22_22421


namespace max_n_divisor_l22_22795

theorem max_n_divisor (k n : ℕ) (h1 : 81849 % n = k) (h2 : 106392 % n = k) (h3 : 124374 % n = k) : n = 243 := by
  sorry

end max_n_divisor_l22_22795


namespace reflect_point_across_x_axis_l22_22569

theorem reflect_point_across_x_axis {x y : ℝ} (h : (x, y) = (2, 3)) : (x, -y) = (2, -3) :=
by
  sorry

end reflect_point_across_x_axis_l22_22569


namespace range_of_a_l22_22249

theorem range_of_a (a : ℝ) :
  ((∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ (-4 < a ∧ a ≤ 0)) :=
sorry

end range_of_a_l22_22249


namespace product_of_possible_values_l22_22408

theorem product_of_possible_values :
  (∀ x : ℝ, abs (18 / x + 4) = 3 → x = -18 ∨ x = -18 / 7) →
  (∀ x1 x2 : ℝ, x1 = -18 → x2 = -18 / 7 → x1 * x2 = 324 / 7) :=
by
  intros h x1 x2 hx1 hx2
  rw [hx1, hx2]
  norm_num

end product_of_possible_values_l22_22408


namespace arctan_sum_l22_22014

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l22_22014


namespace radius_of_circle_l22_22689

theorem radius_of_circle (r : ℝ) (C : ℝ) (A : ℝ) (h1 : 3 * C = 2 * A) 
  (h2 : C = 2 * Real.pi * r) (h3 : A = Real.pi * r^2) : 
  r = 3 :=
by 
  sorry

end radius_of_circle_l22_22689


namespace set_equality_l22_22805

theorem set_equality (M P : Set (ℝ × ℝ))
  (hM : M = {p : ℝ × ℝ | p.1 + p.2 < 0 ∧ p.1 * p.2 > 0})
  (hP : P = {p : ℝ × ℝ | p.1 < 0 ∧ p.2 < 0}) : M = P :=
by
  sorry

end set_equality_l22_22805


namespace delivery_meals_l22_22362

theorem delivery_meals (M P : ℕ) 
  (h1 : P = 8 * M) 
  (h2 : M + P = 27) : 
  M = 3 := by
  sorry

end delivery_meals_l22_22362


namespace kitten_length_after_4_months_l22_22507

theorem kitten_length_after_4_months
  (initial_length : ℕ)
  (doubled_length_2_weeks : ℕ)
  (final_length_4_months : ℕ)
  (h1 : initial_length = 4)
  (h2 : doubled_length_2_weeks = initial_length * 2)
  (h3 : final_length_4_months = doubled_length_2_weeks * 2) :
  final_length_4_months = 16 := 
by
  sorry

end kitten_length_after_4_months_l22_22507


namespace arithmetic_sqrt_sqrt_16_eq_2_l22_22678

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l22_22678


namespace angle_sum_420_l22_22261

theorem angle_sum_420 (A B C D E F : ℝ) (hE : E = 30) : 
  A + B + C + D + E + F = 420 :=
by
  sorry

end angle_sum_420_l22_22261


namespace value_of_expression_l22_22968

theorem value_of_expression (x : ℤ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  exact rfl

end value_of_expression_l22_22968


namespace calculator_press_count_l22_22728

theorem calculator_press_count : 
  ∃ n : ℕ, n ≥ 4 ∧ (2 ^ (2 ^ n)) > 500 := 
by
  sorry

end calculator_press_count_l22_22728


namespace product_increase_by_13_exists_l22_22571

theorem product_increase_by_13_exists :
  ∃ a1 a2 a3 a4 a5 a6 a7 : ℕ,
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * (a1 * a2 * a3 * a4 * a5 * a6 * a7)) :=
by
  sorry

end product_increase_by_13_exists_l22_22571


namespace part1_part2_l22_22263

variables (a b : ℝ)

theorem part1 (h₀ : a > 0) (h₁ : b > 0) (h₂ : ab = a + b + 8) : ab ≥ 16 :=
sorry

theorem part2 (h₀ : a > 0) (h₁ : b > 0) (h₂ : ab = a + b + 8) :
  ∃ (a b : ℝ), a = 7 ∧ b = 5 / 2 ∧ a + 4 * b = 17 :=
sorry

end part1_part2_l22_22263


namespace sean_whistles_l22_22311

def charles_whistles : ℕ := 13
def extra_whistles : ℕ := 32

theorem sean_whistles : charles_whistles + extra_whistles = 45 := by
  sorry

end sean_whistles_l22_22311


namespace range_of_a_l22_22906

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = abs (x - 2) + abs (x + a) ∧ f x ≥ 3) : a ≤ -5 ∨ a ≥ 1 :=
sorry

end range_of_a_l22_22906


namespace impossible_coins_l22_22594

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l22_22594


namespace complex_quadrant_l22_22451

open Complex

theorem complex_quadrant (z : ℂ) (h : z = (2 - I) / (2 + I)) : 
  z.re > 0 ∧ z.im < 0 := 
by
  sorry

end complex_quadrant_l22_22451


namespace sum_real_imag_parts_l22_22251

open Complex

theorem sum_real_imag_parts (z : ℂ) (i : ℂ) (i_property : i * i = -1) (z_eq : z * i = -1 + i) :
  (z.re + z.im = 2) :=
  sorry

end sum_real_imag_parts_l22_22251


namespace container_volume_ratio_l22_22530

theorem container_volume_ratio (C D : ℝ) (hC: C > 0) (hD: D > 0)
  (h: (3/4) * C = (5/8) * D) : (C / D) = (5 / 6) :=
by
  sorry

end container_volume_ratio_l22_22530


namespace wilsons_theorem_l22_22988

theorem wilsons_theorem (p : ℕ) (hp : p ≥ 2) : Nat.Prime p ↔ (Nat.factorial (p - 1) + 1) % p = 0 := 
sorry

end wilsons_theorem_l22_22988


namespace angle_distance_between_CM_BK_l22_22420

noncomputable def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

def vector (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2, B.3 - A.3)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2 + u.3^2)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

theorem angle_distance_between_CM_BK :
  let A := (0, 0, 0) in
  let B := (a, 0, 0) in
  let C := (a/2, a*Real.sqrt 3 / 2, 0) in
  let D := (a/2, a*Real.sqrt 3 / 6, a*Real.sqrt 6 / 3) in
  let M := midpoint A B in
  let K := midpoint C D in
  let CM := vector C M in
  let BK := vector B K in
  let theta := Real.arccos ((dot_product CM BK) / ((magnitude CM) * (magnitude BK))) in
  theta = Real.arccos (Real.sqrt 6 / 3) ∧
  Real.sqrt ((magnitude (cross_product CM BK))^2 / (magnitude (vector C B))^2) = (Real.sqrt 3 / 10)
:=
by
  sorry

end angle_distance_between_CM_BK_l22_22420


namespace find_a1_a7_l22_22947

variable {a n : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k n, a (k + n) = a k + n * d

theorem find_a1_a7 
  (a1 : ℝ) (d : ℝ)
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h1 : a 3 + a 5 = 14)
  (h2 : a 2 * a 6 = 33) :
  a 1 * a 7 = 13 := 
sorry

end find_a1_a7_l22_22947


namespace max_ahead_distance_l22_22501

noncomputable def distance_run_by_alex (initial_distance ahead1 ahead_max_runs final_ahead : ℝ) : ℝ :=
  initial_distance + ahead1 + ahead_max_runs + final_ahead

theorem max_ahead_distance :
  let initial_distance := 200
  let ahead1 := 300
  let final_ahead := 440
  let total_road := 5000
  let distance_remaining := 3890
  let distance_run_alex := total_road - distance_remaining
  ∃ X : ℝ, distance_run_by_alex initial_distance ahead1 X final_ahead = distance_run_alex ∧ X = 170 :=
by
  intro initial_distance ahead1 final_ahead total_road distance_remaining distance_run_alex
  use 170
  simp [initial_distance, ahead1, final_ahead, total_road, distance_remaining, distance_run_alex, distance_run_by_alex]
  sorry

end max_ahead_distance_l22_22501


namespace simplify_sqrt_seven_pow_six_l22_22632

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l22_22632


namespace even_integers_in_form_3k_plus_4_l22_22551

theorem even_integers_in_form_3k_plus_4 (n : ℕ) :
  (20 ≤ n ∧ n ≤ 180 ∧ ∃ k : ℕ, n = 3 * k + 4) → 
  (∃ s : ℕ, s = 27) :=
by
  sorry

end even_integers_in_form_3k_plus_4_l22_22551


namespace variance_of_scores_l22_22183

open Real

def scores : List ℝ := [30, 26, 32, 27, 35]
noncomputable def average (s : List ℝ) : ℝ := s.sum / s.length
noncomputable def variance (s : List ℝ) : ℝ :=
  (s.map (λ x => (x - average s) ^ 2)).sum / s.length

theorem variance_of_scores :
  variance scores = 54 / 5 := 
by
  sorry

end variance_of_scores_l22_22183


namespace derivative_at_five_l22_22548

noncomputable def g (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

theorem derivative_at_five : deriv g 5 = 26 :=
sorry

end derivative_at_five_l22_22548


namespace radius_of_circle_l22_22702

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end radius_of_circle_l22_22702


namespace findNumberOfItemsSoldByStoreA_l22_22324

variable (P x : ℝ) -- P is the price of the product, x is the number of items Store A sells

-- Total sales amount for Store A (in yuan)
def totalSalesA := P * x = 7200

-- Total sales amount for Store B (in yuan)
def totalSalesB := 0.8 * P * (x + 15) = 7200

-- Same price in both stores
def samePriceInBothStores := (P > 0)

-- Proof Problem Statement
theorem findNumberOfItemsSoldByStoreA (storeASellsAtListedPrice : totalSalesA P x)
  (storeBSells15MoreItemsAndAt80PercentPrice : totalSalesB P x)
  (priceIsPositive : samePriceInBothStores P) :
  x = 60 :=
sorry

end findNumberOfItemsSoldByStoreA_l22_22324


namespace optimal_response_l22_22924

theorem optimal_response (n : ℕ) (m : ℕ) (s : ℕ) (a_1 : ℕ) (a_2 : ℕ -> ℕ) (a_opt : ℕ):
  n = 100 → 
  m = 107 →
  (∀ i, i ≥ 1 ∧ i ≤ 99 → a_2 i = a_opt) →
  a_1 = 7 :=
by
  sorry

end optimal_response_l22_22924


namespace total_minutes_to_finish_album_l22_22810

variable (initial_songs : ℕ) (additional_songs : ℕ) (duration : ℕ)

theorem total_minutes_to_finish_album 
  (h1: initial_songs = 25) 
  (h2: additional_songs = 10) 
  (h3: duration = 3) :
  (initial_songs + additional_songs) * duration = 105 :=
sorry

end total_minutes_to_finish_album_l22_22810


namespace cos_x_when_sin_x_is_given_l22_22390

theorem cos_x_when_sin_x_is_given (x : ℝ) (h : Real.sin x = (Real.sqrt 5) / 5) :
  Real.cos x = -(Real.sqrt 20) / 5 :=
sorry

end cos_x_when_sin_x_is_given_l22_22390


namespace no_real_roots_of_geom_seq_l22_22559

theorem no_real_roots_of_geom_seq (a b c : ℝ) (h_geom_seq : b^2 = a * c) : ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  -- You can assume the steps of proving here
  sorry

end no_real_roots_of_geom_seq_l22_22559


namespace fgf_3_equals_108_l22_22843

def f (x : ℕ) : ℕ := 2 * x + 4
def g (x : ℕ) : ℕ := 5 * x + 2

theorem fgf_3_equals_108 : f (g (f 3)) = 108 := 
by
  sorry

end fgf_3_equals_108_l22_22843


namespace necessary_and_sufficient_condition_l22_22250

def f (x a : ℝ) : ℝ := x^3 + 3 * a * x

def slope_tangent_at_one (a : ℝ) : ℝ := 3 * 1^2 + 3 * a

def are_perpendicular (a : ℝ) : Prop := -a = -1

theorem necessary_and_sufficient_condition (a : ℝ) :
  (slope_tangent_at_one a = 6) ↔ (are_perpendicular a) :=
by
  sorry

end necessary_and_sufficient_condition_l22_22250


namespace compute_a_l22_22949

theorem compute_a (a b : ℚ) 
  (h_root1 : (-1:ℚ) - 5 * (Real.sqrt 3) = -1 - 5 * (Real.sqrt 3))
  (h_rational1 : (-1:ℚ) + 5 * (Real.sqrt 3) = -1 + 5 * (Real.sqrt 3))
  (h_poly : ∀ x, x^3 + a*x^2 + b*x + 48 = 0) :
  a = 50 / 37 :=
by
  sorry

end compute_a_l22_22949


namespace impossible_coins_l22_22598

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l22_22598


namespace total_weight_of_apples_l22_22847

/-- Define the weight of an apple and an orange -/
def apple_weight := 4
def orange_weight := 3

/-- Define the maximum weight a bag can hold -/
def max_bag_weight := 49

/-- Define the number of bags Marta buys -/
def num_bags := 3

/-- Prove the total weight of apples Marta should buy -/
theorem total_weight_of_apples : 
    ∀ (A : ℕ), 4 * A + 3 * A ≤ 49 → A = 7 → 4 * A * 3 = 84 :=
by 
    intros A h1 h2
    rw [h2]
    norm_num 
    sorry

end total_weight_of_apples_l22_22847


namespace part_a_part_b_l22_22927

-- Part (a): Prove that for N = a^2 + 2, the equation has positive integral solutions for infinitely many a.
theorem part_a (N : ℕ) (a : ℕ) (x y z t : ℕ) (hx : x = a * (a^2 + 2)) (hy : y = a) (hz : z = 1) (ht : t = 1) :
  (∃ (N : ℕ), ∀ (a : ℕ), ∃ (x y z t : ℕ),
    x^2 + y^2 + z^2 + t^2 = N * x * y * z * t + N ∧
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0) :=
sorry

-- Part (b): Prove that for N = 4^k(8m + 7), the equation has no positive integral solutions.
theorem part_b (N : ℕ) (k m : ℕ) (x y z t : ℕ) (hN : N = 4^k * (8 * m + 7)) :
  ¬ (x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ x^2 + y^2 + z^2 + t^2 = N * x * y * z * t + N) :=
sorry

end part_a_part_b_l22_22927


namespace max_integer_values_correct_l22_22292

noncomputable def max_integer_values (a b c : ℝ) : ℕ :=
  if a > 100 then 2 else 0

theorem max_integer_values_correct (a b c : ℝ) (h : a > 100) :
  max_integer_values a b c = 2 :=
by sorry

end max_integer_values_correct_l22_22292


namespace geometric_sequence_result_l22_22274

-- Definitions representing the conditions
variables {a : ℕ → ℝ}

-- Conditions
axiom cond1 : a 7 * a 11 = 6
axiom cond2 : a 4 + a 14 = 5

theorem geometric_sequence_result :
  ∃ x, x = a 20 / a 10 ∧ (x = 2 / 3 ∨ x = 3 / 2) :=
by {
  sorry
}

end geometric_sequence_result_l22_22274


namespace find_other_integer_l22_22854

theorem find_other_integer (x y : ℤ) (h1 : 3*x + 4*y = 103) (h2 : x = 19 ∨ y = 19) : x = 9 ∨ y = 9 :=
by sorry

end find_other_integer_l22_22854


namespace solve_equation_l22_22138

theorem solve_equation (x : ℝ) (h1 : 2 * x + 1 ≠ 0) (h2 : 4 * x ≠ 0) : 
  (3 / (2 * x + 1) = 5 / (4 * x)) ↔ (x = 2.5) :=
by 
  sorry

end solve_equation_l22_22138


namespace total_height_of_sculpture_and_base_l22_22193

def height_of_sculpture_m : Float := 0.88
def height_of_base_cm : Float := 20
def meter_to_cm : Float := 100

theorem total_height_of_sculpture_and_base :
  (height_of_sculpture_m * meter_to_cm + height_of_base_cm) = 108 :=
by
  sorry

end total_height_of_sculpture_and_base_l22_22193


namespace find_sum_of_A_and_B_l22_22727

theorem find_sum_of_A_and_B :
  ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ B = A - 2 ∧ A = 5 + 3 ∧ A + B = 14 :=
by
  sorry

end find_sum_of_A_and_B_l22_22727


namespace arithmetic_square_root_of_sqrt_16_l22_22671

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l22_22671


namespace sum_of_squares_of_sides_l22_22204

-- Definition: A cyclic quadrilateral with perpendicular diagonals inscribed in a circle
structure CyclicQuadrilateral (R : ℝ) :=
  (m n k t : ℝ) -- sides of the quadrilateral
  (perpendicular_diagonals : true) -- diagonals are perpendicular (trivial placeholder)
  (radius : ℝ := R) -- Radius of the circumscribed circle

-- The theorem to prove: The sum of the squares of the sides of the quadrilateral is 8R^2
theorem sum_of_squares_of_sides (R : ℝ) (quad : CyclicQuadrilateral R) :
  quad.m ^ 2 + quad.n ^ 2 + quad.k ^ 2 + quad.t ^ 2 = 8 * R^2 := 
by sorry

end sum_of_squares_of_sides_l22_22204


namespace circle_radius_l22_22695

theorem circle_radius (r : ℝ) (hr : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 :=
by 
  sorry

end circle_radius_l22_22695


namespace arithmetic_sequence_seventh_term_l22_22476

variable (a1 a15 : ℚ)
variable (n : ℕ) (a7 : ℚ)

-- Given conditions
def first_term (a1 : ℚ) : Prop := a1 = 3
def last_term (a15 : ℚ) : Prop := a15 = 72
def total_terms (n : ℕ) : Prop := n = 15

-- Arithmetic sequence formula
def common_difference (d : ℚ) : Prop := d = (72 - 3) / (15 - 1)
def nth_term (a_n : ℚ) (a1 : ℚ) (n : ℕ) (d : ℚ) : Prop := a_n = a1 + (n - 1) * d

-- Prove that the 7th term is approximately 33
theorem arithmetic_sequence_seventh_term :
  ∀ (a1 a15 : ℚ) (n : ℕ), first_term a1 → last_term a15 → total_terms n → ∃ a7 : ℚ, 
  nth_term a7 a1 7 ((a15 - a1) / (n - 1)) ∧ (33 - 0.5) < a7 ∧ a7 < (33 + 0.5) :=
by {
  sorry
}

end arithmetic_sequence_seventh_term_l22_22476


namespace arithmetic_sqrt_of_sqrt_16_l22_22654

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22654


namespace proposition_4_l22_22840

variables {Line Plane : Type}
variables {a b : Line} {α β : Plane}

-- Definitions of parallel and perpendicular relationships
class Parallel (l : Line) (p : Plane) : Prop
class Perpendicular (l : Line) (p : Plane) : Prop
class Contains (p : Plane) (l : Line) : Prop

theorem proposition_4
  (h1: Perpendicular a β)
  (h2: Parallel a b)
  (h3: Contains α b) : Perpendicular α β :=
sorry

end proposition_4_l22_22840


namespace evaluate_expression_l22_22531

theorem evaluate_expression : 3002^3 - 3001 * 3002^2 - 3001^2 * 3002 + 3001^3 + 1 = 6004 :=
by
  sorry

end evaluate_expression_l22_22531


namespace coefficient_of_x3_in_expansion_l22_22717

theorem coefficient_of_x3_in_expansion :
  let coeff := 56 * 972 * Real.sqrt 2
  coeff = 54432 * Real.sqrt 2 :=
by
  let coeff := 56 * 972 * Real.sqrt 2
  have h : coeff = 54432 * Real.sqrt 2 := sorry
  exact h

end coefficient_of_x3_in_expansion_l22_22717


namespace minor_premise_is_wrong_l22_22474

theorem minor_premise_is_wrong (a : ℝ) : ¬ (0 < a^2) := by
  sorry

end minor_premise_is_wrong_l22_22474


namespace correct_statement_l22_22807

variables {Line Plane : Type}
variable (a b c : Line)
variable (M N : Plane)

/- Definitions for the conditions -/
def lies_on_plane (l : Line) (p : Plane) : Prop := sorry
def intersection (p1 p2 : Plane) : Line := sorry
def parallel (l1 l2 : Line) : Prop := sorry

/- Conditions -/
axiom h1 : lies_on_plane a M
axiom h2 : lies_on_plane b N
axiom h3 : intersection M N = c

/- The correct statement to be proved -/
theorem correct_statement : parallel a b → parallel a c :=
by sorry

end correct_statement_l22_22807


namespace shirt_cost_l22_22085

variables (J S : ℝ)

theorem shirt_cost :
  (3 * J + 2 * S = 69) ∧
  (2 * J + 3 * S = 86) →
  S = 24 :=
by
  sorry

end shirt_cost_l22_22085


namespace weight_of_a_l22_22877

-- Define conditions
def weight_of_b : ℕ := 750 -- weight of one liter of ghee packet of brand 'b' in grams
def ratio_a_to_b : ℕ × ℕ := (3, 2)
def total_volume_liters : ℕ := 4 -- total volume of the mixture in liters
def total_weight_grams : ℕ := 3360 -- total weight of the mixture in grams

-- Target proof statement
theorem weight_of_a (W_a : ℕ) 
  (h_ratio : (ratio_a_to_b.1 + ratio_a_to_b.2) = 5)
  (h_mix_vol_a : (ratio_a_to_b.1 * total_volume_liters) = 12)
  (h_mix_vol_b : (ratio_a_to_b.2 * total_volume_liters) = 8)
  (h_weight_eq : (ratio_a_to_b.1 * W_a * total_volume_liters + ratio_a_to_b.2 * weight_of_b * total_volume_liters) = total_weight_grams * 5) : 
  W_a = 900 :=
by {
  sorry
}

end weight_of_a_l22_22877


namespace directrix_of_parabola_l22_22937

-- Define the given parabola equation
def parabola_eq (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 5

-- Define the expected result for the directrix
def directrix_eq : ℝ := -23 / 12

-- State the problem in Lean
theorem directrix_of_parabola : 
  (∃ d : ℝ, (∀ x y : ℝ, y = parabola_eq x → y = d) → d = directrix_eq) :=
by
  sorry

end directrix_of_parabola_l22_22937


namespace find_a_plus_b_l22_22529

variable (r a b : ℝ)
variable (seq : ℕ → ℝ)

-- Conditions on the sequence
axiom seq_def : seq 0 = 4096
axiom seq_rule : ∀ n, seq (n + 1) = seq n * r

-- Given value
axiom r_value : r = 1 / 4

-- Given intermediate positions in the sequence
axiom seq_a : seq 3 = a
axiom seq_b : seq 4 = b
axiom seq_5 : seq 5 = 4

-- Theorem to prove
theorem find_a_plus_b : a + b = 80 := by
  sorry

end find_a_plus_b_l22_22529


namespace division_number_l22_22304

-- Definitions from conditions
def D : Nat := 3
def Q : Nat := 4
def R : Nat := 3

-- Theorem statement
theorem division_number : ∃ N : Nat, N = D * Q + R ∧ N = 15 :=
by
  sorry

end division_number_l22_22304


namespace function_range_l22_22684

-- Proving the range of the given function f(x) on the interval [-1, 2]
theorem function_range (a b : ℝ) (h1 : f(x) = a*x^2 + b*x - 2)
  (h2 : ∀ x ∈ Icc (-2) 2, f(-x) = f(x))
  (h3 : 1 + a = -2) :
  set.range (λ x : ℝ, a * x^2 + b * x - 2) ∩ Icc (-1 : ℝ) 2 = Icc (-14 : ℝ) (-2) := 
sorry

end function_range_l22_22684


namespace trajectory_eq_range_of_k_l22_22946

-- definitions based on the conditions:
def fixed_circle (x y : ℝ) := (x + 1)^2 + y^2 = 16
def moving_circle_passing_through_B (M : ℝ × ℝ) (B : ℝ × ℝ) := 
    B = (1, 0) ∧ M.1^2 / 4 + M.2^2 / 3 = 1 -- the ellipse trajectory equation

-- question 1: prove the equation of the ellipse
theorem trajectory_eq :
    ∀ M : ℝ × ℝ, (∃ B : ℝ × ℝ, moving_circle_passing_through_B M B)
    → (M.1^2 / 4 + M.2^2 / 3 = 1) :=
sorry

-- question 2: find the range of k which satisfies given area condition
theorem range_of_k (k : ℝ) :
    (∃ M : ℝ × ℝ, ∃ B : ℝ × ℝ, moving_circle_passing_through_B M B) → 
    (0 < k) → (¬ (k = 0)) →
    ((∃ m : ℝ, (4 * k^2 + 3 - m^2 > 0) ∧ 
    (1 / 2) * (|k| * m^2 / (4 * k^2 + 3)^2) = 1 / 14) → (3 / 4 < k ∧ k < 1) 
    ∨ (-1 < k ∧ k < -3 / 4)) :=
sorry

end trajectory_eq_range_of_k_l22_22946


namespace odd_function_min_periodic_3_l22_22792

noncomputable def f : ℝ → ℝ
| x => if -3/2 < x ∧ x < 0 then real.logb 2 (-3 * x + 1) else sorry

theorem odd_function_min_periodic_3 (f : ℝ → ℝ)
  (hf_odd: ∀ x, f (-x) = -f (x))
  (hf_period: ∀ x, f (x + 3) = f x)
  (hf_def: ∀ x : ℝ, -3/2 < x ∧ x < 0 → f x = real.logb 2 (-3 * x + 1)) :
  f 2011 = -2 := 
sorry

end odd_function_min_periodic_3_l22_22792


namespace two_digit_numbers_less_than_35_l22_22074

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l22_22074


namespace smallest_a_plus_b_l22_22951

theorem smallest_a_plus_b (a b : ℕ) (h1: 0 < a) (h2: 0 < b) (h3 : 2^10 * 7^3 = a^b) : a + b = 31 :=
sorry

end smallest_a_plus_b_l22_22951


namespace old_geometry_book_pages_l22_22914

def old_pages := 340
def new_pages := 450
def deluxe_pages := 915

theorem old_geometry_book_pages : 
  (new_pages = 2 * old_pages - 230) ∧ 
  (deluxe_pages = new_pages + old_pages + 125) ∧ 
  (deluxe_pages ≥ old_pages + old_pages / 10) 
  → old_pages = 340 := by
  sorry

end old_geometry_book_pages_l22_22914


namespace arithmetic_square_root_of_sqrt_16_l22_22656

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l22_22656


namespace simplify_fraction_l22_22128

theorem simplify_fraction :
  (21 / 25) * (35 / 45) * (75 / 63) = 35 / 9 :=
by
  sorry

end simplify_fraction_l22_22128


namespace count_two_digit_numbers_less_35_l22_22066

open Nat

theorem count_two_digit_numbers_less_35 : 
  let two_digit_numbers := (finset.Ico 10 100) -- The range of two-digit numbers
  let count_satisfying := two_digit_numbers.filter (λ n, (n / 10 < 3 ∨ n % 10 < 5)).card
  count_satisfying = 55 :=
by
  -- Placeholder for the actual proof.
  sorry

end count_two_digit_numbers_less_35_l22_22066


namespace average_weight_l22_22881

theorem average_weight (men women : ℕ) (avg_weight_men avg_weight_women : ℝ) (total_people : ℕ) (combined_avg_weight : ℝ) 
  (h1 : men = 8) (h2 : avg_weight_men = 190) (h3 : women = 6) (h4 : avg_weight_women = 120) (h5 : total_people = 14) 
  (h6 : (men * avg_weight_men + women * avg_weight_women) / total_people = combined_avg_weight) : combined_avg_weight = 160 := 
  sorry

end average_weight_l22_22881


namespace arithmetic_contains_geometric_l22_22174

theorem arithmetic_contains_geometric {a b : ℚ} (h : a^2 + b^2 ≠ 0) :
  ∃ (c q : ℚ) (f : ℕ → ℚ), (∀ n, f n = c * q^n) ∧ (∀ n, f n = a + b * n) := 
sorry

end arithmetic_contains_geometric_l22_22174


namespace sum_of_ages_l22_22830

/-
Juliet is 3 years older than her sister Maggie but 2 years younger than her elder brother Ralph.
If Juliet is 10 years old, the sum of Maggie's and Ralph's ages is 19 years.
-/
theorem sum_of_ages (juliet_age maggie_age ralph_age : ℕ) :
  juliet_age = 10 →
  juliet_age = maggie_age + 3 →
  ralph_age = juliet_age + 2 →
  maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l22_22830


namespace fraction_of_earnings_spent_on_candy_l22_22286

theorem fraction_of_earnings_spent_on_candy :
  let candy_bars_cost := 2 * 0.75
  let lollipops_cost := 4 * 0.25
  let total_candy_cost := candy_bars_cost + lollipops_cost
  let earnings_per_driveway := 1.5
  let total_earnings := 10 * earnings_per_driveway
  total_candy_cost / total_earnings = 1 / 6 :=
by
  let candy_bars_cost := 2 * 0.75
  let lollipops_cost := 4 * 0.25
  let total_candy_cost := candy_bars_cost + lollipops_cost
  let earnings_per_driveway := 1.5
  let total_earnings := 10 * earnings_per_driveway
  have h : total_candy_cost / total_earnings = 1 / 6 := by sorry
  exact h

end fraction_of_earnings_spent_on_candy_l22_22286


namespace directrix_of_parabola_l22_22939

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = -3 * x^2 + 6 * x - 5 → y = -35 / 18 := by
  sorry

end directrix_of_parabola_l22_22939


namespace simplify_sqrt_power_l22_22625

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l22_22625


namespace arctan_sum_l22_22008

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l22_22008


namespace impossible_coins_l22_22611

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l22_22611


namespace num_suitable_two_digit_numbers_l22_22061

/-- 
How many two-digit numbers have at least one digit that is smaller than the corresponding digit in the number 35?
-/
theorem num_suitable_two_digit_numbers : 
  let two_digit_numbers := { n : ℕ | 10 ≤ n ∧ n ≤ 99 },
      suitable_numbers := { n ∈ two_digit_numbers | (n / 10 < 3) ∨ (n % 10 < 5) } in
  suitable_numbers.card = 55 :=
by
  sorry

end num_suitable_two_digit_numbers_l22_22061


namespace calculation_correct_l22_22001

theorem calculation_correct :
  (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) * (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64) = 4^128 - 3^128 :=
by
  sorry

end calculation_correct_l22_22001


namespace arithmetic_sqrt_of_sqrt_16_l22_22653

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22653


namespace frosting_sugar_calc_l22_22178

theorem frosting_sugar_calc (total_sugar cake_sugar : ℝ) (h1 : total_sugar = 0.8) (h2 : cake_sugar = 0.2) : 
  total_sugar - cake_sugar = 0.6 :=
by
  rw [h1, h2]
  sorry  -- Proof should go here

end frosting_sugar_calc_l22_22178


namespace directrix_of_parabola_l22_22940

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = -3 * x^2 + 6 * x - 5 → y = -35 / 18 := by
  sorry

end directrix_of_parabola_l22_22940


namespace cows_on_farm_l22_22114

theorem cows_on_farm (weekly_production_per_6_cows : ℕ) 
                     (production_over_5_weeks : ℕ) 
                     (number_of_weeks : ℕ) 
                     (cows : ℕ) :
  weekly_production_per_6_cows = 108 →
  production_over_5_weeks = 2160 →
  number_of_weeks = 5 →
  (cows * (weekly_production_per_6_cows / 6) * number_of_weeks = production_over_5_weeks) →
  cows = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end cows_on_farm_l22_22114


namespace k_is_even_set_l22_22267

open Set -- using Set from Lean library

noncomputable def kSet (s : Set ℤ) :=
  (∀ g ∈ ({5, 8, 7, 1} : Set ℤ), ∀ k ∈ s, (g * k) % 2 = 0)

theorem k_is_even_set (s : Set ℤ) :
  (∀ g ∈ ({5, 8, 7, 1} : Set ℤ), ∀ k ∈ s, (g * k) % 2 = 0) →
  ∀ k ∈ s, k % 2 = 0 :=
by
  intro h
  sorry

end k_is_even_set_l22_22267


namespace cosine_evaluation_l22_22398

variable (α : ℝ)

theorem cosine_evaluation
  (h : Real.sin (Real.pi / 6 + α) = 1 / 3) :
  Real.cos (Real.pi / 3 - α) = 1 / 3 :=
sorry

end cosine_evaluation_l22_22398


namespace interval_of_x_l22_22496

theorem interval_of_x (x : ℝ) (h : x = ((-x)^2 / x) + 3) : 3 < x ∧ x ≤ 6 :=
by
  sorry

end interval_of_x_l22_22496


namespace repeating_decimal_sum_l22_22767

theorem repeating_decimal_sum :
  (0.12121212 + 0.003003003 + 0.0000500005 : ℚ) = 124215 / 999999 :=
by 
  have h1 : (0.12121212 : ℚ) = (0.12 + 0.0012) := sorry
  have h2 : (0.003003003 : ℚ) = (0.003 + 0.000003) := sorry
  have h3 : (0.0000500005 : ℚ) = (0.00005 + 0.0000000005) := sorry
  sorry


end repeating_decimal_sum_l22_22767


namespace grape_juice_percentage_l22_22266

theorem grape_juice_percentage
  (initial_volume : ℝ) (initial_percentage : ℝ) (added_juice : ℝ)
  (h_initial_volume : initial_volume = 50)
  (h_initial_percentage : initial_percentage = 0.10)
  (h_added_juice : added_juice = 10) :
  ((initial_percentage * initial_volume + added_juice) / (initial_volume + added_juice) * 100) = 25 := 
by
  sorry

end grape_juice_percentage_l22_22266


namespace factorial_mod_prime_l22_22231
-- Import all necessary libraries

-- State the conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The main problem statement
theorem factorial_mod_prime (n : ℕ) (h : n = 10) : factorial n % 13 = 7 := by
  sorry

end factorial_mod_prime_l22_22231


namespace number_of_milkshakes_l22_22188

-- Define the amounts and costs
def initial_money : ℕ := 132
def remaining_money : ℕ := 70
def hamburger_cost : ℕ := 4
def milkshake_cost : ℕ := 5
def hamburgers_bought : ℕ := 8

-- Defining the money spent calculations
def hamburgers_spent : ℕ := hamburgers_bought * hamburger_cost
def total_spent : ℕ := initial_money - remaining_money
def milkshake_spent : ℕ := total_spent - hamburgers_spent

-- The final theorem to prove
theorem number_of_milkshakes : (milkshake_spent / milkshake_cost) = 6 :=
by
  sorry

end number_of_milkshakes_l22_22188


namespace kitten_current_length_l22_22514

theorem kitten_current_length (initial_length : ℕ) (double_after_2_weeks : ℕ → ℕ) (double_after_4_months : ℕ → ℕ)
  (h1 : initial_length = 4)
  (h2 : double_after_2_weeks initial_length = 2 * initial_length)
  (h3 : double_after_4_months (double_after_2_weeks initial_length) = 2 * (double_after_2_weeks initial_length)) :
  double_after_4_months (double_after_2_weeks initial_length) = 16 := 
by
  sorry

end kitten_current_length_l22_22514


namespace num_pos_divisors_180_l22_22555

theorem num_pos_divisors_180 : 
  let n := 180 in
  let prime_factorization := [(2, 2), (3, 2), (5, 1)] in
  (prime_factorization.foldr (λ (p : ℕ × ℕ) te : ℕ, (p.2 + 1) * te) 1) = 18 :=
by 
  let n := 180
  let prime_factorization := [(2, 2), (3, 2), (5, 1)]
  have num_divisors := prime_factorization.foldr (λ (p : ℕ × ℕ) te : ℕ, (p.2 + 1) * te) 1 
  show num_divisors = 18
  sorry

end num_pos_divisors_180_l22_22555


namespace tanvi_min_candies_l22_22316

theorem tanvi_min_candies : 
  ∃ c : ℕ, 
  (c % 6 = 5) ∧ 
  (c % 8 = 7) ∧ 
  (c % 9 = 6) ∧ 
  (c % 11 = 0) ∧ 
  (∀ d : ℕ, 
    (d % 6 = 5) ∧ 
    (d % 8 = 7) ∧ 
    (d % 9 = 6) ∧ 
    (d % 11 = 0) → 
    c ≤ d) → 
  c = 359 :=
by sorry

end tanvi_min_candies_l22_22316


namespace arctan_sum_l22_22009

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l22_22009


namespace metropolis_hospital_babies_l22_22564

theorem metropolis_hospital_babies 
    (a b d : ℕ) 
    (h1 : a = 3 * b) 
    (h2 : b = 2 * d) 
    (h3 : 2 * a + 3 * b + 5 * d = 1200) : 
    5 * d = 260 := 
sorry

end metropolis_hospital_babies_l22_22564


namespace probability_of_multiple_2_3_7_l22_22923

open Nat

def count_multiples (n m : ℕ) : ℕ :=
  m / n

def inclusion_exclusion (a b c : ℕ) (ab ac bc abc : ℕ) : ℕ :=
  a + b + c - ab - ac - bc + abc

def probability_multiple_2_3_7 : ℚ :=
  let total_cards := 150
  let multiples_2 := count_multiples 2 total_cards
  let multiples_3 := count_multiples 3 total_cards
  let multiples_7 := count_multiples 7 total_cards
  let multiples_6 := count_multiples 6 total_cards
  let multiples_14 := count_multiples 14 total_cards
  let multiples_21 := count_multiples 21 total_cards
  let multiples_42 := count_multiples 42 total_cards
  let total_multiples := inclusion_exclusion multiples_2 multiples_3 multiples_7 multiples_6 multiples_14 multiples_21 multiples_42
  (total_multiples : ℚ) / (total_cards : ℚ)

theorem probability_of_multiple_2_3_7 :
  probability_multiple_2_3_7 = 107 / 150 :=
  sorry

end probability_of_multiple_2_3_7_l22_22923


namespace find_amount_l22_22343

def total_amount (A : ℝ) : Prop :=
  A / 20 = A / 25 + 100

theorem find_amount 
  (A : ℝ) 
  (h : total_amount A) : 
  A = 10000 := 
  sorry

end find_amount_l22_22343


namespace simplify_sqrt7_pow6_l22_22622

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l22_22622


namespace johns_cycling_speed_needed_l22_22577

theorem johns_cycling_speed_needed 
  (swim_speed : Float := 3)
  (swim_distance : Float := 0.5)
  (run_speed : Float := 8)
  (run_distance : Float := 4)
  (total_time : Float := 3)
  (bike_distance : Float := 20) :
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed))) = 60 / 7 := 
  by
  sorry

end johns_cycling_speed_needed_l22_22577


namespace domain_of_f_l22_22319

noncomputable def f (x : ℝ) := 1 / Real.log (x + 1) + Real.sqrt (9 - x^2)

theorem domain_of_f : {x : ℝ | (x > -1) ∧ (x ≠ 0) ∧ (x ∈ [-3, 3])} = 
  {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 3} :=
by
  sorry

end domain_of_f_l22_22319


namespace find_y_l22_22265

/-- Given (2 ^ x) - (2 ^ y) = 3 * (2 ^ 10) and x = 12, prove that y = 10 -/
theorem find_y (x y : ℕ) (h : (2 ^ x) - (2 ^ y) = 3 * (2 ^ 10)) (hx : x = 12) : y = 10 :=
by
  sorry

end find_y_l22_22265


namespace unique_c1_c2_exists_l22_22105

theorem unique_c1_c2_exists (a_0 a_1 x_1 x_2 : ℝ) (h_distinct : x_1 ≠ x_2) : 
  ∃! (c_1 c_2 : ℝ), ∀ n : ℕ, a_n = c_1 * x_1^n + c_2 * x_2^n :=
sorry

end unique_c1_c2_exists_l22_22105


namespace cube_construction_possible_l22_22490

theorem cube_construction_possible (n : ℕ) : (∃ k : ℕ, n = 12 * k) ↔ ∃ V : ℕ, (n ^ 3) = 12 * V := by
sorry

end cube_construction_possible_l22_22490


namespace sum_of_ages_l22_22831

/-
Juliet is 3 years older than her sister Maggie but 2 years younger than her elder brother Ralph.
If Juliet is 10 years old, the sum of Maggie's and Ralph's ages is 19 years.
-/
theorem sum_of_ages (juliet_age maggie_age ralph_age : ℕ) :
  juliet_age = 10 →
  juliet_age = maggie_age + 3 →
  ralph_age = juliet_age + 2 →
  maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l22_22831


namespace find_third_integer_l22_22705

theorem find_third_integer (a b c : ℕ) (h1 : a * b * c = 42) (h2 : a + b = 9) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : c = 3 :=
sorry

end find_third_integer_l22_22705


namespace probability_of_picking_letter_from_mathematics_l22_22815

-- Definition of the problem conditions
def extended_alphabet_size := 30
def distinct_letters_in_mathematics := 8

-- Theorem statement
theorem probability_of_picking_letter_from_mathematics :
  (distinct_letters_in_mathematics / extended_alphabet_size : ℚ) = 4 / 15 := 
by 
  sorry

end probability_of_picking_letter_from_mathematics_l22_22815


namespace factory_produces_correct_number_of_doors_l22_22350

variable (initial_planned_production : ℕ) (metal_shortage_decrease : ℕ) (pandemic_decrease_factor : ℕ)
variable (doors_per_car : ℕ)

theorem factory_produces_correct_number_of_doors
  (h1 : initial_planned_production = 200)
  (h2 : metal_shortage_decrease = 50)
  (h3 : pandemic_decrease_factor = 50)
  (h4 : doors_per_car = 5) :
  (initial_planned_production - metal_shortage_decrease) * (100 - pandemic_decrease_factor) * doors_per_car / 100 = 375 :=
by
  sorry

end factory_produces_correct_number_of_doors_l22_22350


namespace time_interval_for_birth_and_death_rates_l22_22097

theorem time_interval_for_birth_and_death_rates
  (birth_rate : ℝ)
  (death_rate : ℝ)
  (population_net_increase_per_day : ℝ)
  (number_of_minutes_per_day : ℝ)
  (net_increase_per_interval : ℝ)
  (time_intervals_per_day : ℝ)
  (time_interval_in_minutes : ℝ):

  birth_rate = 10 →
  death_rate = 2 →
  population_net_increase_per_day = 345600 →
  number_of_minutes_per_day = 1440 →
  net_increase_per_interval = birth_rate - death_rate →
  time_intervals_per_day = population_net_increase_per_day / net_increase_per_interval →
  time_interval_in_minutes = number_of_minutes_per_day / time_intervals_per_day →
  time_interval_in_minutes = 48 :=
by
  intros
  sorry

end time_interval_for_birth_and_death_rates_l22_22097


namespace solve_for_s_l22_22352

-- Definition of the given problem conditions
def parallelogram_sides_60_angle_sqrt_area (s : ℝ) :=
  ∃ (area : ℝ), (area = 27 * Real.sqrt 3) ∧
  (3 * s * s * Real.sqrt 3 = area)

-- Proof statement to demonstrate the equivalence of the theoretical and computed value of s
theorem solve_for_s (s : ℝ) : parallelogram_sides_60_angle_sqrt_area s → s = 3 :=
by
  intro h
  sorry

end solve_for_s_l22_22352


namespace power_function_k_values_l22_22819

theorem power_function_k_values (k : ℝ) :
  (∃ (a : ℝ), (k^2 - k - 5) = a ∧ (∀ x : ℝ, (k^2 - k - 5) * x^3 = a * x^3)) →
  (k = 3 ∨ k = -2) :=
by
  intro h
  sorry

end power_function_k_values_l22_22819


namespace impossible_coins_l22_22601

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l22_22601


namespace symmetric_points_y_axis_l22_22396

theorem symmetric_points_y_axis (a b : ℝ) (h₁ : ∃ M N : ℝ × ℝ, M = (a, 3) ∧ N = (4, b) ∧ M.1 = -N.1 ∧ M.2 = N.2) :
  (a + b) ^ 2012 = 1 :=
by 
  sorry

end symmetric_points_y_axis_l22_22396


namespace n_cube_plus_5n_divisible_by_6_l22_22851

theorem n_cube_plus_5n_divisible_by_6 (n : ℤ) : 6 ∣ (n^3 + 5 * n) := 
sorry

end n_cube_plus_5n_divisible_by_6_l22_22851


namespace simplify_sqrt7_pow6_l22_22637

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l22_22637


namespace marbles_game_winning_strategy_l22_22123

theorem marbles_game_winning_strategy :
  ∃ k : ℕ, 1 < k ∧ k < 1024 ∧ (k = 4 ∨ k = 24 ∨ k = 40) := sorry

end marbles_game_winning_strategy_l22_22123


namespace arctan_sum_l22_22002

theorem arctan_sum : ∀ (a b : ℝ), 
  a = 3/7 → 
  b = 7/3 → 
  a * b = 1 → 
  a > 0 → 
  b > 0 → 
  Real.arctan a + Real.arctan b = Real.pi / 2 :=
by intros a b ha hb hab ha_pos hb_pos
   rw [ha, hb, hab]
   sorry

end arctan_sum_l22_22002


namespace max_profit_l22_22422

/-- Define the cost and price of device A and device B -/
def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8

/-- Define the total number of devices -/
def total_devices : ℝ := 50

/-- Define the profits per device -/
def profit_per_A : ℝ := price_A - cost_A -- 0.3
def profit_per_B : ℝ := price_B - cost_B -- 0.4

/-- Define the function for total profit -/
def total_profit (x : ℝ) : ℝ :=
  profit_per_A * x + profit_per_B * (total_devices - x)

/-- Define the constraint -/
def constraint (x : ℝ) : Prop := 4 * x ≥ total_devices - x -- x ≥ 10

/-- The statement of the problem that needs to be proven -/
theorem max_profit :
  (total_profit x = -0.1 * x + 20) ∧ 
  ( ∀ x, constraint x → x ≥ 10 → x = 10 ∧ total_profit x = 19) :=
by
  sorry

end max_profit_l22_22422


namespace bob_and_jim_total_skips_l22_22191

-- Definitions based on conditions
def bob_skips_per_rock : Nat := 12
def jim_skips_per_rock : Nat := 15
def rocks_skipped_by_each : Nat := 10

-- Total skips calculation based on the given conditions
def bob_total_skips : Nat := bob_skips_per_rock * rocks_skipped_by_each
def jim_total_skips : Nat := jim_skips_per_rock * rocks_skipped_by_each
def total_skips : Nat := bob_total_skips + jim_total_skips

-- Theorem statement
theorem bob_and_jim_total_skips : total_skips = 270 := by
  sorry

end bob_and_jim_total_skips_l22_22191


namespace simplify_sqrt7_pow6_l22_22639

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l22_22639


namespace probability_green_ball_eq_l22_22020

noncomputable def prob_green_ball : ℚ := 
  1 / 3 * (5 / 18) + 1 / 3 * (1 / 2) + 1 / 3 * (1 / 2)

theorem probability_green_ball_eq : 
  prob_green_ball = 23 / 54 := 
  by
  sorry

end probability_green_ball_eq_l22_22020


namespace simplify_expression_l22_22858

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : (x^2)⁻¹ - 2 = (1 - 2 * x^2) / (x^2) :=
by
  -- proof here
  sorry

end simplify_expression_l22_22858


namespace find_y_when_x_is_4_l22_22465

variables (x y : ℕ)
def inversely_proportional (C : ℕ) (x y : ℕ) : Prop := x * y = C

theorem find_y_when_x_is_4 :
  inversely_proportional 240 x y → x = 4 → y = 60 :=
by
  sorry

end find_y_when_x_is_4_l22_22465


namespace edward_money_l22_22764

theorem edward_money (X : ℝ) (H1 : X - 130 - 0.25 * (X - 130) = 270) : X = 490 :=
by
  sorry

end edward_money_l22_22764


namespace sum_of_ages_l22_22832

/-
Juliet is 3 years older than her sister Maggie but 2 years younger than her elder brother Ralph.
If Juliet is 10 years old, the sum of Maggie's and Ralph's ages is 19 years.
-/
theorem sum_of_ages (juliet_age maggie_age ralph_age : ℕ) :
  juliet_age = 10 →
  juliet_age = maggie_age + 3 →
  ralph_age = juliet_age + 2 →
  maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l22_22832


namespace value_of_y_l22_22482

theorem value_of_y (y : ℝ) (h : |y| = |y - 3|) : y = 3 / 2 :=
sorry

end value_of_y_l22_22482


namespace maria_original_number_25_3_l22_22111

theorem maria_original_number_25_3 (x : ℚ) 
  (h : ((3 * (x + 3) - 4) / 3) = 10) : 
  x = 25 / 3 := 
by 
  sorry

end maria_original_number_25_3_l22_22111


namespace probability_of_perfect_square_sum_l22_22159

-- Define the conditions
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def perfect_squares_in_range : Finset ℕ := {4, 9, 16}
def total_outcomes : ℕ := 8 * 8  -- total number of outcomes when rolling two dice

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  (dice_faces.product dice_faces).filter (λ pair => perfect_squares_in_range.member (pair.1 + pair.2)).card

-- Calculate the probability of getting a perfect square sum
noncomputable def probability_perfect_square_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- The statement to prove that the probability is 3/16
theorem probability_of_perfect_square_sum :
  probability_perfect_square_sum = 3 / 16 :=
begin
  sorry
end

end probability_of_perfect_square_sum_l22_22159


namespace fraction_zero_iff_x_neg_one_l22_22820

theorem fraction_zero_iff_x_neg_one (x : ℝ) (h : 1 - |x| = 0) (h_non_zero : 1 - x ≠ 0) : x = -1 :=
sorry

end fraction_zero_iff_x_neg_one_l22_22820


namespace simplify_sqrt_seven_pow_six_proof_l22_22628

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l22_22628


namespace canoes_to_kayaks_ratio_l22_22162

theorem canoes_to_kayaks_ratio
  (canoe_cost kayak_cost total_revenue canoes_more_than_kayaks : ℕ)
  (H1 : canoe_cost = 14)
  (H2 : kayak_cost = 15)
  (H3 : total_revenue = 288)
  (H4 : ∃ C K : ℕ, C = K + canoes_more_than_kayaks ∧ 14 * C + 15 * K = 288) :
  ∃ (r : ℚ), r = 3 / 2 := by
  sorry

end canoes_to_kayaks_ratio_l22_22162


namespace find_values_l22_22586

theorem find_values (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x - 4 = 21 * (1 / x)) 
  (h2 : x + y^2 = 45) : 
  x = 7 ∧ y = Real.sqrt 38 :=
by
  sorry

end find_values_l22_22586


namespace solve_system_of_inequalities_l22_22132

theorem solve_system_of_inequalities {x : ℝ} :
  (x + 3 ≥ 2) ∧ (2 * (x + 4) > 4 * x + 2) ↔ (-1 ≤ x ∧ x < 3) :=
by
  sorry

end solve_system_of_inequalities_l22_22132


namespace range_of_m_l22_22775

noncomputable def f (m x : ℝ) : ℝ := m * (x - 2 * m) * (x + m + 3)
noncomputable def g (x : ℝ) : ℝ := 2^x - 2

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) ∧ (∃ x : ℝ, x < -4 ∧ f m x * g x < 0) → (-4 < m ∧ m < -2) :=
by
  sorry

end range_of_m_l22_22775


namespace line_intersects_ellipse_with_conditions_l22_22048

theorem line_intersects_ellipse_with_conditions :
  ∃ l : ℝ → ℝ, (∃ A B : ℝ × ℝ, 
  (A.fst^2/6 + A.snd^2/3 = 1 ∧ B.fst^2/6 + B.snd^2/3 = 1) ∧
  A.fst > 0 ∧ A.snd > 0 ∧ B.fst > 0 ∧ B.snd > 0 ∧
  (∃ M N : ℝ × ℝ, 
    M.snd = 0 ∧ N.fst = 0 ∧
    M.fst^2 + N.snd^2 = (2 * Real.sqrt 3)^2 ∧
    (M.snd - A.snd)^2 + (M.fst - A.fst)^2 = (N.fst - B.fst)^2 + (N.snd - B.snd)^2) ∧
    (∀ x, l x + Real.sqrt 2 * x - 2 * Real.sqrt 2 = 0)
) :=
sorry

end line_intersects_ellipse_with_conditions_l22_22048


namespace math_problem_l22_22297

variable {a b c : ℝ}

theorem math_problem
  (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0)
  (h : a + b + c = -a * b * c) :
  (a^2 * b^2 / ((a^2 + b * c) * (b^2 + a * c)) +
  a^2 * c^2 / ((a^2 + b * c) * (c^2 + a * b)) +
  b^2 * c^2 / ((b^2 + a * c) * (c^2 + a * b))) = 1 :=
by
  sorry

end math_problem_l22_22297


namespace find_sum_of_cubes_l22_22987

noncomputable def roots_of_polynomial := 
  ∃ a b c : ℝ, 
    (6 * a^3 + 500 * a + 1001 = 0) ∧ 
    (6 * b^3 + 500 * b + 1001 = 0) ∧ 
    (6 * c^3 + 500 * c + 1001 = 0)

theorem find_sum_of_cubes (a b c : ℝ) 
  (h : roots_of_polynomial) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 500.5 := 
sorry

end find_sum_of_cubes_l22_22987


namespace triangles_form_even_square_l22_22233

-- Given conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def triangle_area (b h : ℕ) : ℚ :=
  (b * h) / 2

-- Statement of the problem
theorem triangles_form_even_square (n : ℕ) :
  (∀ t : Fin n, is_right_triangle 3 4 5 ∧ triangle_area 3 4 = 6) →
  (∃ a : ℕ, a^2 = 6 * n) →
  Even n :=
by
  sorry

end triangles_form_even_square_l22_22233


namespace sum_of_areas_of_tangent_circles_l22_22318

theorem sum_of_areas_of_tangent_circles :
  ∀ (a b c : ℝ), 
    a + b = 5 →
    a + c = 12 →
    b + c = 13 →
    π * (a^2 + b^2 + c^2) = 113 * π :=
by
  intros a b c h₁ h₂ h₃
  sorry

end sum_of_areas_of_tangent_circles_l22_22318


namespace number_of_sheep_l22_22977

theorem number_of_sheep (s d : ℕ) 
  (h1 : s + d = 15)
  (h2 : 4 * s + 2 * d = 22 + 2 * (s + d)) : 
  s = 11 :=
by
  sorry

end number_of_sheep_l22_22977


namespace impossible_coins_l22_22615

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l22_22615


namespace percentage_small_bottles_sold_l22_22182

theorem percentage_small_bottles_sold :
  ∀ (x : ℕ), (6000 - (x * 60)) + 8500 = 13780 → x = 12 :=
by
  intro x h
  sorry

end percentage_small_bottles_sold_l22_22182


namespace cary_wage_after_two_years_l22_22751

theorem cary_wage_after_two_years (initial_wage raise_percentage cut_percentage : ℝ) (wage_after_first_year wage_after_second_year : ℝ) :
  initial_wage = 10 ∧ raise_percentage = 0.2 ∧ cut_percentage = 0.75 ∧ 
  wage_after_first_year = initial_wage * (1 + raise_percentage) ∧
  wage_after_second_year = wage_after_first_year * cut_percentage → 
  wage_after_second_year = 9 :=
by
  sorry

end cary_wage_after_two_years_l22_22751


namespace fraction_negative_iff_x_lt_2_l22_22416

theorem fraction_negative_iff_x_lt_2 (x : ℝ) :
  (-5) / (2 - x) < 0 ↔ x < 2 := by
  sorry

end fraction_negative_iff_x_lt_2_l22_22416


namespace product_increase_by_13_l22_22573

theorem product_increase_by_13 {
    a1 a2 a3 a4 a5 a6 a7 : ℕ
} : (a1 > 3) → (a2 > 3) → (a3 > 3) → (a4 > 3) → (a5 > 3) → (a6 > 3) → (a7 > 3) → 
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * a1 * a2 * a3 * a4 * a5 * a6 * a7) :=
        sorry

end product_increase_by_13_l22_22573


namespace students_owning_both_pets_l22_22565

theorem students_owning_both_pets:
  ∀ (students total students_dog students_cat : ℕ),
    total = 45 →
    students_dog = 28 →
    students_cat = 38 →
    -- Each student owning at least one pet means 
    -- total = students_dog ∪ students_cat
    total = students_dog + students_cat - students →
    students = 21 :=
by
  intros students total students_dog students_cat h_total h_dog h_cat h_union
  sorry

end students_owning_both_pets_l22_22565


namespace polynomial_remainder_x1012_l22_22536

theorem polynomial_remainder_x1012 (x : ℂ) : 
  (x^1012) % (x^3 - x^2 + x - 1) = 1 :=
sorry

end polynomial_remainder_x1012_l22_22536


namespace exponent_equality_l22_22083

theorem exponent_equality (s m : ℕ) (h : (2^16) * (25^s) = 5 * (10^m)) : m = 16 :=
by sorry

end exponent_equality_l22_22083


namespace speed_of_second_train_l22_22716

-- Definitions of conditions
def distance_train1 : ℝ := 200
def speed_train1 : ℝ := 50
def distance_train2 : ℝ := 240
def time_train1_and_train2 : ℝ := 4

-- Statement of the problem
theorem speed_of_second_train : (distance_train2 / time_train1_and_train2) = 60 := by
  sorry

end speed_of_second_train_l22_22716


namespace snow_total_inches_l22_22365

theorem snow_total_inches (initial_snow_ft : ℝ) (additional_snow_in : ℝ)
  (melted_snow_in : ℝ) (multiplier : ℝ) (days_after : ℕ) (conversion_rate : ℝ)
  (initial_snow_in : ℝ) (fifth_day_snow_in : ℝ) :
  initial_snow_ft = 0.5 →
  additional_snow_in = 8 →
  melted_snow_in = 2 →
  multiplier = 2 →
  days_after = 5 →
  conversion_rate = 12 →
  initial_snow_in = initial_snow_ft * conversion_rate →
  fifth_day_snow_in = multiplier * initial_snow_in →
  (initial_snow_in + additional_snow_in - melted_snow_in + fifth_day_snow_in) / conversion_rate = 2 :=
by
  sorry

end snow_total_inches_l22_22365


namespace domain_of_f_range_of_f_strictly_decreasing_on_positive_reals_range_of_f_on_interval_l22_22248

noncomputable def f (x : ℝ) := (x + 2) / x

theorem domain_of_f : {x : ℝ | x ≠ 0} = set_of (λ x, x ≠ 0) := 
sorry

theorem range_of_f : ∀ y : ℝ, y ≠ 1 ↔ ∃ x : ℝ, x ≠ 0 ∧ f x = y := 
sorry

theorem strictly_decreasing_on_positive_reals : ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ > f x₂ := 
sorry

theorem range_of_f_on_interval : 
  set.range (λ x, (f x)) (set.Icc (2 : ℝ) (8 : ℝ)) = set.Icc (5 / 4 : ℝ) 2 := 
sorry

end domain_of_f_range_of_f_strictly_decreasing_on_positive_reals_range_of_f_on_interval_l22_22248


namespace factorial_mod_10_eq_6_l22_22215

theorem factorial_mod_10_eq_6 : (10! % 13) = 6 := by
  sorry

end factorial_mod_10_eq_6_l22_22215


namespace find_quotient_l22_22118

-- Define the given conditions
def dividend : ℤ := 144
def divisor : ℤ := 11
def remainder : ℤ := 1

-- Define the quotient logically derived from the given conditions
def quotient : ℤ := dividend / divisor

-- The theorem we need to prove
theorem find_quotient : quotient = 13 := by
  sorry

end find_quotient_l22_22118


namespace ratio_of_DE_EC_l22_22317

noncomputable def ratio_DE_EC (a x : ℝ) : ℝ :=
  let DE := a - x
  x / DE

theorem ratio_of_DE_EC (a : ℝ) (H1 : ∀ x, x = 5 * a / 7) :
  ratio_DE_EC a (5 * a / 7) = 5 / 2 :=
by
  sorry

end ratio_of_DE_EC_l22_22317


namespace scholarship_total_l22_22445

-- Definitions of the money received by Wendy, Kelly, Nina, and Jason based on the given conditions
def wendy_scholarship : ℕ := 20000
def kelly_scholarship : ℕ := 2 * wendy_scholarship
def nina_scholarship : ℕ := kelly_scholarship - 8000
def jason_scholarship : ℕ := (3 * kelly_scholarship) / 4

-- Total amount of scholarships
def total_scholarship : ℕ := wendy_scholarship + kelly_scholarship + nina_scholarship + jason_scholarship

-- The proof statement that needs to be proven
theorem scholarship_total : total_scholarship = 122000 := by
  -- Here we use 'sorry' to indicate that the proof is not provided.
  sorry

end scholarship_total_l22_22445


namespace isosceles_triangle_perimeter_l22_22787

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 4) (h2 : b = 9) (h3 : c = 9) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : a + b + c = 22 := 
by 
  sorry

end isosceles_triangle_perimeter_l22_22787


namespace difference_between_new_and_original_l22_22543

variables (x y : ℤ) -- Declaring variables x and y as integers

-- The original number is represented as 10*x + y, and the new number after swapping is 10*y + x.
-- We need to prove that the difference between the new number and the original number is -9*x + 9*y.
theorem difference_between_new_and_original (x y : ℤ) :
  (10 * y + x) - (10 * x + y) = -9 * x + 9 * y :=
by
  sorry -- Proof placeholder

end difference_between_new_and_original_l22_22543


namespace sqrt_mul_power_expr_l22_22522

theorem sqrt_mul_power_expr : ( (Real.sqrt 3 + Real.sqrt 2) ^ 2023 * (Real.sqrt 3 - Real.sqrt 2) ^ 2022 ) = (Real.sqrt 3 + Real.sqrt 2) := 
  sorry

end sqrt_mul_power_expr_l22_22522


namespace y_at_x8_l22_22088

theorem y_at_x8 (k : ℝ) (y : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
by
  sorry

end y_at_x8_l22_22088


namespace no_such_k_l22_22839

theorem no_such_k (u : ℕ → ℝ) (v : ℕ → ℝ)
  (h1 : u 0 = 6) (h2 : v 0 = 4)
  (h3 : ∀ n, u (n + 1) = (3 / 5) * u n - (4 / 5) * v n)
  (h4 : ∀ n, v (n + 1) = (4 / 5) * u n + (3 / 5) * v n) :
  ¬ ∃ k, u k = 7 ∧ v k = 2 :=
by
  sorry

end no_such_k_l22_22839


namespace years_ago_l22_22649

theorem years_ago (M D X : ℕ) (hM : M = 41) (hD : D = 23) 
  (h_eq : M - X = 2 * (D - X)) : X = 5 := by 
  sorry

end years_ago_l22_22649


namespace radius_of_circle_l22_22690

theorem radius_of_circle (r : ℝ) (C : ℝ) (A : ℝ) (h1 : 3 * C = 2 * A) 
  (h2 : C = 2 * Real.pi * r) (h3 : A = Real.pi * r^2) : 
  r = 3 :=
by 
  sorry

end radius_of_circle_l22_22690


namespace min_value_of_sum_squares_l22_22812

theorem min_value_of_sum_squares (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 11) : 
  x^2 + y^2 + z^2 ≥ 121 / 29 := sorry

end min_value_of_sum_squares_l22_22812


namespace evaluate_i_powers_sum_l22_22379

-- Given conditions: i is the imaginary unit
def i : ℂ := Complex.I

-- Proof problem: Prove that i^2023 + i^2024 + i^2025 + i^2026 = 0
theorem evaluate_i_powers_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := 
by sorry

end evaluate_i_powers_sum_l22_22379


namespace find_line_l_l22_22046

theorem find_line_l (A B M N : ℝ × ℝ)
  (h1 : ∃ x1 y1 x2 y2, A = (x1, y1) ∧ B = (x2, y2) ∧ x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 > 0)
  (h2 : M = (x1, 0))
  (h3 : N = (0, y2))
  (h4 : abs (| M.1 - A.1 |) = abs (| N.2 - B.2 |))
  (h5 : dist M N = 2 * sqrt 3) :
  ∃ k m : ℝ, k < 0 ∧ m > 0 ∧ (∀ x y : ℝ, (y = k * x + m ↔ x + sqrt 2 * y - 2 * sqrt 2 = 0)) := sorry

end find_line_l_l22_22046


namespace intersecting_lines_c_plus_d_l22_22459

theorem intersecting_lines_c_plus_d (c d : ℝ) 
  (h1 : ∀ y, ∃ x, x = (1/3) * y + c) 
  (h2 : ∀ x, ∃ y, y = (1/3) * x + d)
  (P : (3:ℝ) = (1 / 3) * (3:ℝ) + c) 
  (Q : (3:ℝ) = (1 / 3) * (3:ℝ) + d) : 
  c + d = 4 := 
by
  sorry

end intersecting_lines_c_plus_d_l22_22459


namespace parabola_passes_through_fixed_point_l22_22842

theorem parabola_passes_through_fixed_point:
  ∀ t : ℝ, ∃ x y : ℝ, (y = 4 * x^2 + 2 * t * x - 3 * t ∧ (x = 3 ∧ y = 36)) :=
by
  intro t
  use 3
  use 36
  sorry

end parabola_passes_through_fixed_point_l22_22842


namespace find_a_minus_b_l22_22315

def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -3 * x + 5
def h (a b x : ℝ) : ℝ := f a b (g x)
def h_inv (x : ℝ) : ℝ := x + 7

theorem find_a_minus_b (a b : ℝ) :
  (∀ x : ℝ, h a b x = -3 * a * x + 5 * a + b) ∧
  (∀ x : ℝ, h_inv (h a b x) = x) ∧
  (∀ x : ℝ, h a b x = x - 7) →
  a - b = 5 :=
by
  sorry

end find_a_minus_b_l22_22315


namespace range_of_a_l22_22991

noncomputable def odd_function_periodic_real (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ -- odd function condition
  (∀ x, f (x + 5) = f x) ∧ -- periodic function condition
  (f 1 < -1) ∧ -- given condition
  (f 4 = Real.log a / Real.log 2) -- condition using log base 2

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h : odd_function_periodic_real f a) : a > 2 :=
by sorry 

end range_of_a_l22_22991


namespace union_of_A_and_B_is_R_l22_22039

open Set Real

def A := {x : ℝ | log x > 0}
def B := {x : ℝ | x ≤ 1}

theorem union_of_A_and_B_is_R : A ∪ B = univ := by
  sorry

end union_of_A_and_B_is_R_l22_22039


namespace arithmetic_square_root_of_sqrt_16_l22_22658

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l22_22658


namespace find_a_perpendicular_lines_l22_22788

theorem find_a_perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, ax + (a + 2) * y + 1 = 0 ∧ x + a * y + 2 = 0) → a = -3 :=
sorry

end find_a_perpendicular_lines_l22_22788


namespace probability_dice_sum_perfect_square_l22_22146

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end probability_dice_sum_perfect_square_l22_22146


namespace volume_of_circumscribed_sphere_of_cube_l22_22952

theorem volume_of_circumscribed_sphere_of_cube (a : ℝ) (h : a = 1) : 
  (4 / 3) * Real.pi * ((Real.sqrt 3 / 2) ^ 3) = (Real.sqrt 3 / 2) * Real.pi :=
by sorry

end volume_of_circumscribed_sphere_of_cube_l22_22952


namespace inequality_proof_l22_22992

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (c + a)) * (1 + 4 * c / (a + b)) > 25 := by
  sorry

end inequality_proof_l22_22992


namespace supplementary_angle_measure_l22_22870

theorem supplementary_angle_measure (a b : ℝ) 
  (h1 : a + b = 180) 
  (h2 : a / 5 = b / 4) : b = 80 :=
by
  sorry

end supplementary_angle_measure_l22_22870


namespace num_ordered_pairs_xy_eq_2200_l22_22688

/-- There are 24 ordered pairs (x, y) such that xy = 2200. -/
theorem num_ordered_pairs_xy_eq_2200 : 
  ∃ (n : ℕ), n = 24 ∧ (∃ divisors : Finset ℕ, 
    (∀ d ∈ divisors, 2200 % d = 0) ∧ 
    (divisors.card = 24)) := 
sorry

end num_ordered_pairs_xy_eq_2200_l22_22688


namespace find_value_l22_22540

theorem find_value (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 :=
sorry

end find_value_l22_22540


namespace percentage_reduction_price_increase_l22_22180

-- Part 1: Proof that the percentage reduction each time is 20%
theorem percentage_reduction (a : ℝ) (h1 : 50 * (1 - a)^2 = 32) : a = 0.2 := 
by
  have : 1 - a = √(32 / 50) := sorry
  have : 1 - a = 0.8 := sorry
  have : a = 1 - 0.8 := sorry
  exact this

-- Part 2: Proof that increasing the price by 5 yuan achieves the required profit
theorem price_increase 
  (x : ℝ)
  (h2 : (10 + x) * (500 - 20 * x) = 6000) 
  (h3 : ∀ y : ℝ, (10 + y) * (500 - 20 * y) < 6000 → y > x) 
  : x = 5 :=
by
  have : -20 * x^2 + 300 * x - 1000 = 0 := sorry
  have : x^2 - 15 * x + 50 = 0 := sorry
  have solution1 : x = 5 := sorry
  have solution2 : x = 10 := sorry
  have : x ≠ 10 := sorry
  exact solution1

end percentage_reduction_price_increase_l22_22180


namespace intersection_of_complements_l22_22845

open Set

variable (U A B : Set Nat)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
variable (hA : A = {2, 3, 4, 5})
variable (hB : B = {2, 4, 6, 8})

theorem intersection_of_complements :
  A ∩ (U \ B) = {3, 5} :=
by
  rw [hU, hA, hB]
  sorry

end intersection_of_complements_l22_22845


namespace expression_eqn_l22_22497

theorem expression_eqn (a : ℝ) (E : ℝ → ℝ)
  (h₁ : -6 * a^2 = 3 * (E a + 2))
  (h₂ : a = 1) : E a = -2 * a^2 - 2 :=
by
  sorry

end expression_eqn_l22_22497


namespace find_k_value_l22_22456

theorem find_k_value (k : ℝ) : (∃ k, ∀ x y, y = k * x + 3 ∧ (x, y) = (1, 2)) → k = -1 :=
by
  sorry

end find_k_value_l22_22456


namespace hansel_album_duration_l22_22809

theorem hansel_album_duration 
    (initial_songs : ℕ)
    (additional_songs : ℕ)
    (duration_per_song : ℕ)
    (h_initial : initial_songs = 25)
    (h_additional : additional_songs = 10)
    (h_duration : duration_per_song = 3):
    initial_songs * duration_per_song + additional_songs * duration_per_song = 105 := 
by
  sorry

end hansel_album_duration_l22_22809


namespace both_complementary_angles_acute_is_certain_event_l22_22904

def complementary_angles (A B : ℝ) : Prop :=
  A + B = 90

def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

theorem both_complementary_angles_acute_is_certain_event (A B : ℝ) (h1 : complementary_angles A B) (h2 : acute_angle A) (h3 : acute_angle B) : (A < 90) ∧ (B < 90) :=
by
  sorry

end both_complementary_angles_acute_is_certain_event_l22_22904


namespace arithmetic_sqrt_of_sqrt_16_l22_22675

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22675


namespace number_of_cows_on_farm_l22_22117

theorem number_of_cows_on_farm :
  (∀ (cows_per_week : ℤ) (six_cows_milk : ℤ) (total_milk : ℤ) (weeks : ℤ),
    cows_per_week = 6 → 
    six_cows_milk = 108 →
    total_milk = 2160 →
    weeks = 5 →
    (total_milk / (six_cows_milk / cows_per_week * weeks)) = 24) :=
by
  intros cows_per_week six_cows_milk total_milk weeks h1 h2 h3 h4
  have h_cow_milk_per_week : six_cows_milk / cows_per_week = 18 := by sorry
  have h_cow_milk_per_five_weeks : (six_cows_milk / cows_per_week) * weeks = 90 := by sorry
  have h_total_cows : total_milk / ((six_cows_milk / cows_per_week) * weeks) = 24 := by sorry
  exact h_total_cows

end number_of_cows_on_farm_l22_22117


namespace sequence_infinite_divisibility_l22_22102

theorem sequence_infinite_divisibility :
  ∃ (u : ℕ → ℤ), (∀ n, u (n + 2) = u (n + 1) ^ 2 - u n) ∧ u 1 = 39 ∧ u 2 = 45 ∧ (∀ N, ∃ k ≥ N, 1986 ∣ u k) := 
by
  sorry

end sequence_infinite_divisibility_l22_22102


namespace green_shirt_pairs_l22_22747

theorem green_shirt_pairs (r g : ℕ) (p total_pairs red_pairs : ℕ) :
  r = 63 → g = 69 → p = 66 → red_pairs = 25 → (g - (r - red_pairs * 2)) / 2 = 28 :=
by
  intros hr hg hp hred_pairs
  sorry

end green_shirt_pairs_l22_22747


namespace total_rides_correct_l22_22560

-- Definitions based on the conditions:
def billy_rides : ℕ := 17
def john_rides : ℕ := 2 * billy_rides
def mother_rides : ℕ := john_rides + 10
def total_rides : ℕ := billy_rides + john_rides + mother_rides

-- The theorem to prove their total bike rides.
theorem total_rides_correct : total_rides = 95 := by
  sorry

end total_rides_correct_l22_22560


namespace kho_kho_only_l22_22488

theorem kho_kho_only (kabaddi_total : ℕ) (both_games : ℕ) (total_players : ℕ) (kabaddi_only : ℕ) (kho_kho_only : ℕ) 
  (h1 : kabaddi_total = 10)
  (h2 : both_games = 5)
  (h3 : total_players = 50)
  (h4 : kabaddi_only = 10 - both_games)
  (h5 : kabaddi_only + kho_kho_only + both_games = total_players) :
  kho_kho_only = 40 :=
by
  -- Proof is not required
  sorry

end kho_kho_only_l22_22488


namespace number_of_students_l22_22447

theorem number_of_students (left_pos right_pos total_pos : ℕ) 
  (h₁ : left_pos = 5) 
  (h₂ : right_pos = 3) 
  (h₃ : total_pos = left_pos - 1 + 1 + (right_pos - 1)) : 
  total_pos = 7 :=
by
  rw [h₁, h₂] at h₃
  simp at h₃
  exact h₃

end number_of_students_l22_22447


namespace math_proof_problem_l22_22441

noncomputable def even_function (f : ℝ → ℝ) := 
∀ x, f x = f (-x)

noncomputable def satisfied_condition (f : ℝ → ℝ) := 
∀ x, f (x + 1) = - f x

noncomputable def increasing_on_interval (f : ℝ → ℝ) := 
∀ x ∈ Icc (-1 : ℝ) (0 : ℝ), ∀ y ∈ Icc (-1 : ℝ) (0 : ℝ), x < y → f x < f y

theorem math_proof_problem (f : ℝ → ℝ) :
  even_function f →
  satisfied_condition f →
  increasing_on_interval f →
  (∀ x, f (x + 2) = f x) ∧ (∀ x, f (2 - x) = f (2 + x)) ∧ (f 2 = f 0) :=
by
  intros h1 h2 h3
  split
  sorry
  split
  sorry
  sorry

end math_proof_problem_l22_22441


namespace original_salary_l22_22485

-- Given conditions as definitions
def salaryAfterRaise (x : ℝ) : ℝ := 1.10 * x
def salaryAfterReduction (x : ℝ) : ℝ := salaryAfterRaise x * 0.95
def finalSalary : ℝ := 1045

-- Statement to prove
theorem original_salary (x : ℝ) (h : salaryAfterReduction x = finalSalary) : x = 1000 :=
by
  sorry

end original_salary_l22_22485


namespace lions_deers_15_minutes_l22_22814

theorem lions_deers_15_minutes :
  ∀ (n : ℕ), (15 * n = 15 * 15 → n = 15 → ∀ t, t = 15) := by
  sorry

end lions_deers_15_minutes_l22_22814


namespace find_white_towels_l22_22995

variable (W : ℕ) -- Let W be the number of white towels Maria bought

def green_towels : ℕ := 40
def towels_given : ℕ := 65
def towels_left : ℕ := 19

theorem find_white_towels :
  green_towels + W - towels_given = towels_left →
  W = 44 :=
by
  intro h
  sorry

end find_white_towels_l22_22995


namespace no_common_root_l22_22368

theorem no_common_root 
  (a b : ℚ) 
  (α : ℂ) 
  (h1 : α^5 = α + 1) 
  (h2 : α^2 = -a * α - b) : 
  False :=
sorry

end no_common_root_l22_22368


namespace intersection_line_circle_diameter_l22_22053

noncomputable def length_of_AB : ℝ := 2

theorem intersection_line_circle_diameter 
  (x y : ℝ)
  (h_line : x - 2*y - 1 = 0)
  (h_circle : (x - 1)^2 + y^2 = 1) :
  |(length_of_AB)| = 2 := 
sorry

end intersection_line_circle_diameter_l22_22053


namespace sum_of_ages_l22_22836

theorem sum_of_ages (J M R : ℕ) (hJ : J = 10) (hM : M = J - 3) (hR : R = J + 2) : M + R = 19 :=
by
  sorry

end sum_of_ages_l22_22836


namespace f_expression_and_extrema_l22_22237

open Polynomial

-- Define a quadratic function f
noncomputable def f (x : ℝ) : ℝ := 6 * x^2 - 4

-- Conditions
def condition1 : f (-1) = 2 := by
  unfold f
  ring

def condition2 : deriv f 0 = 0 := by
  unfold f
  simp
  ring

def condition3 : ∫ x in 0..1, f x = -2 := by
  unfold f
  calc
    ∫ x in 0..1, (6 * x^2 - 4) 
    = (∫ x in 0..1, 6 * x^2) - (∫ x in 0..1, 4) : by norm_num
    ... = (6 / 3 * 1^3) - (4 * 1 - 4 * 0) : by simp [integral_poly]
    ... = 2 - 4 : by norm_num
    ... = -2 : by norm_num

-- Define the proof problem statement
theorem f_expression_and_extrema :
  (∀ x : ℝ, f x = 6 * x^2 - 4) ∧ (∀ x : ℝ, x ∈ set.Icc (-1) 1 → (-4 ≤ f x ∧ f x ≤ 2)) := by
  sorry

end f_expression_and_extrema_l22_22237


namespace union_sets_l22_22040

namespace Proof

def setA : Set ℝ := { x : ℝ | x * (x + 1) ≤ 0 }
def setB : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }

theorem union_sets : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
sorry

end Proof

end union_sets_l22_22040


namespace taxi_ride_distance_l22_22731

theorem taxi_ride_distance
  (initial_charge : ℝ) (additional_charge : ℝ) 
  (total_charge : ℝ) (initial_increment : ℝ) (distance_increment : ℝ)
  (initial_charge_eq : initial_charge = 2.10) 
  (additional_charge_eq : additional_charge = 0.40) 
  (total_charge_eq : total_charge = 17.70) 
  (initial_increment_eq : initial_increment = 1/5) 
  (distance_increment_eq : distance_increment = 1/5) : 
  (distance : ℝ) = 8 :=
by sorry

end taxi_ride_distance_l22_22731


namespace probability_dice_sum_perfect_square_l22_22147

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end probability_dice_sum_perfect_square_l22_22147


namespace radius_of_circle_l22_22703

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end radius_of_circle_l22_22703


namespace complement_M_l22_22550

noncomputable def U : Set ℝ := Set.univ

def M : Set ℝ := { x | x^2 - 4 ≤ 0 }

theorem complement_M : U \ M = { x | x < -2 ∨ x > 2 } :=
by 
  sorry

end complement_M_l22_22550


namespace pupils_like_both_l22_22710

theorem pupils_like_both (total_pupils : ℕ) (likes_pizza : ℕ) (likes_burgers : ℕ)
  (total := 200) (P := 125) (B := 115) :
  (P + B - total_pupils) = 40 :=
by
  sorry

end pupils_like_both_l22_22710


namespace f_f_has_three_distinct_real_roots_l22_22296

open Polynomial

noncomputable def f (c x : ℝ) : ℝ := x^2 + 6 * x + c

theorem f_f_has_three_distinct_real_roots (c : ℝ) :
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f c (f c x) = 0) ↔
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end f_f_has_three_distinct_real_roots_l22_22296


namespace sqrt_product_l22_22749

theorem sqrt_product (a b : ℝ) (ha : a = 20) (hb : b = 1/5) : Real.sqrt a * Real.sqrt b = 2 := 
by
  sorry

end sqrt_product_l22_22749


namespace triangle_equilateral_l22_22973

-- Assume we are given side lengths a, b, and c of a triangle and angles A, B, and C in radians.
variables {a b c : ℝ} {A B C : ℝ}

-- We'll use the assumption that (a + b + c) * (b + c - a) = 3 * b * c and sin A = 2 * sin B * cos C.
axiom triangle_condition1 : (a + b + c) * (b + c - a) = 3 * b * c
axiom triangle_condition2 : Real.sin A = 2 * Real.sin B * Real.cos C

-- We need to prove that the triangle is equilateral.
theorem triangle_equilateral : (a = b) ∧ (b = c) ∧ (c = a) := by
  sorry

end triangle_equilateral_l22_22973


namespace largest_among_options_l22_22897

theorem largest_among_options :
  let A := 15679 + (1 / 3579)
  let B := 15679 - (1 / 3579)
  let C := 15679 * (1 / 3579)
  let D := 15679 / (1 / 3579)
  let E := 15679 * 1.03
  D > A ∧ D > B ∧ D > C ∧ D > E := by
{
  let A := 15679 + (1 / 3579)
  let B := 15679 - (1 / 3579)
  let C := 15679 * (1 / 3579)
  let D := 15679 / (1 / 3579)
  let E := 15679 * 1.03
  sorry
}

end largest_among_options_l22_22897


namespace Louis_ate_whole_boxes_l22_22101

def package_size := 6
def total_lemon_heads := 54

def whole_boxes : ℕ := total_lemon_heads / package_size

theorem Louis_ate_whole_boxes :
  whole_boxes = 9 :=
by
  sorry

end Louis_ate_whole_boxes_l22_22101


namespace triangle_non_existence_no_solution_max_value_expression_l22_22433

-- Define sides and angles
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Corresponding opposite sides

-- Define the triangle conditions
def triangle_sides_angles (a b c A B C : ℝ) : Prop := 
  (a^2 = (1 - Real.cos A) / (1 - Real.cos B)) ∧ 
  (b = 1) ∧ 
  -- Additional properties ensuring we have a valid triangle can be added here
  (A ≠ B) -- Non-isosceles condition (equivalent to angles being different).

-- Prove non-existence under given conditions
theorem triangle_non_existence_no_solution (h : triangle_sides_angles a b c A B C) : false := 
sorry 

-- Define the maximization problem
theorem max_value_expression (h : a^2 = (1 - Real.cos A) / (1 - Real.cos B)) : 
(∃ b c, (b = 1) → ∀ a, a > 0 → (c > 0) ∧ ((1/c) * (1/b - 1/a)) ≤ (3 - 2 * Real.sqrt 2)) := 
sorry

end triangle_non_existence_no_solution_max_value_expression_l22_22433


namespace largest_five_digit_negative_int_congruent_mod_23_l22_22718

theorem largest_five_digit_negative_int_congruent_mod_23 :
  ∃ n : ℤ, 23 * n + 1 < -9999 ∧ 23 * n + 1 = -9994 := 
sorry

end largest_five_digit_negative_int_congruent_mod_23_l22_22718


namespace solution_exists_l22_22055

open Int

def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d ∣ p, d = 1 ∨ d = p

theorem solution_exists :
  ∃ (p : ℕ), isPrime p ∧ ∃ (x : ℤ), (2 * x^2 - x - 36 = p^2 ∧ x = 13) := sorry

end solution_exists_l22_22055


namespace largest_divisor_l22_22338

theorem largest_divisor (n : ℤ) (h1 : n > 0) (h2 : n % 2 = 1) : 
  (∃ k : ℤ, k > 0 ∧ (∀ n : ℤ, n > 0 → n % 2 = 1 → k ∣ (n * (n + 2) * (n + 4) * (n + 6) * (n + 8)))) → 
  k = 15 :=
by
  sorry

end largest_divisor_l22_22338


namespace sock_ratio_l22_22592

theorem sock_ratio (b : ℕ) (x : ℕ) (hx_pos : 0 < x)
  (h1 : 5 * x + 3 * b * x = k) -- Original cost is 5x + 3bx
  (h2 : b * x + 15 * x = 2 * k) -- Interchanged cost is doubled
  : b = 1 :=
by sorry

end sock_ratio_l22_22592


namespace power_addition_l22_22774

theorem power_addition {a m n : ℝ} (h1 : a^m = 2) (h2 : a^n = 8) : a^(m + n) = 16 :=
sorry

end power_addition_l22_22774


namespace trio_all_three_games_l22_22978

theorem trio_all_three_games (n : ℕ) :
  ∀ (G : SimpleGraph (Fin (3 * n + 1))),
    (∀ v : Fin (3 * n + 1), ∃ w1 w2 w3 : Fin (3 * n + 1),
      w1 ≠ v ∧ w2 ≠ v ∧ w3 ≠ v ∧
      G.Adj v w1 ∧ G.Adj v w2 ∧ G.Adj v w3 ∧ 
      ((G.Adj w1 w2 ∧ G.Adj w2 w3 ∧ G.Adj w3 w1) ∨ 
      (G.Adj w1 w3 ∧ G.Adj v w2 ∧ G.Adj v w1))) →
    ∃ (v1 v2 v3 : Fin (3 * n + 1)),
      v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
      G.Adj v1 v2 ∧ G.Adj v2 v3 ∧ G.Adj v3 v1 :=
sorry

end trio_all_three_games_l22_22978


namespace arithmetic_square_root_of_sqrt_16_l22_22670

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l22_22670


namespace evaluate_i_powers_sum_l22_22380

-- Given conditions: i is the imaginary unit
def i : ℂ := Complex.I

-- Proof problem: Prove that i^2023 + i^2024 + i^2025 + i^2026 = 0
theorem evaluate_i_powers_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := 
by sorry

end evaluate_i_powers_sum_l22_22380


namespace sufficient_but_not_necessary_condition_l22_22241

def P (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (1/x + 4 * x + 6 * m) ≥ 0

def Q (m : ℝ) : Prop :=
  m ≥ -5

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (P m → Q m) ∧ ¬(Q m → P m) := sorry

end sufficient_but_not_necessary_condition_l22_22241


namespace arctan_sum_l22_22015

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l22_22015


namespace num_ways_choose_pair_of_diff_color_socks_l22_22556

-- Define the numbers of socks of each color
def num_white := 5
def num_brown := 5
def num_blue := 3
def num_black := 3

-- Define the calculation for pairs of different colored socks
def num_pairs_white_brown := num_white * num_brown
def num_pairs_brown_blue := num_brown * num_blue
def num_pairs_white_blue := num_white * num_blue
def num_pairs_white_black := num_white * num_black
def num_pairs_brown_black := num_brown * num_black
def num_pairs_blue_black := num_blue * num_black

-- Define the total number of pairs
def total_pairs := num_pairs_white_brown + num_pairs_brown_blue + num_pairs_white_blue + num_pairs_white_black + num_pairs_brown_black + num_pairs_blue_black

-- The theorem to be proved
theorem num_ways_choose_pair_of_diff_color_socks : total_pairs = 94 := by
  -- Since we do not need to include the proof steps, we use sorry
  sorry

end num_ways_choose_pair_of_diff_color_socks_l22_22556


namespace trajectory_equation_l22_22682

theorem trajectory_equation : ∀ (x y : ℝ),
  (x + 3)^2 + y^2 + (x - 3)^2 + y^2 = 38 → x^2 + y^2 = 10 :=
by
  intros x y h
  sorry

end trajectory_equation_l22_22682


namespace total_rides_correct_l22_22561

-- Definitions based on the conditions:
def billy_rides : ℕ := 17
def john_rides : ℕ := 2 * billy_rides
def mother_rides : ℕ := john_rides + 10
def total_rides : ℕ := billy_rides + john_rides + mother_rides

-- The theorem to prove their total bike rides.
theorem total_rides_correct : total_rides = 95 := by
  sorry

end total_rides_correct_l22_22561


namespace impossible_coins_l22_22618

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end impossible_coins_l22_22618


namespace math_and_science_students_l22_22301

theorem math_and_science_students (x y : ℕ) 
  (h1 : x + y + 2 = 30)
  (h2 : y = 3 * x + 4) :
  y - 2 = 20 :=
by {
  sorry
}

end math_and_science_students_l22_22301


namespace charlie_age_l22_22285

variable (J C B : ℝ)

def problem_statement :=
  J = C + 12 ∧ C = B + 7 ∧ J = 3 * B → C = 18

theorem charlie_age : problem_statement J C B :=
by
  sorry

end charlie_age_l22_22285


namespace not_exists_k_eq_one_l22_22945

theorem not_exists_k_eq_one (k : ℝ) : (∃ x y : ℝ, y = k * x + 2 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 :=
by sorry

end not_exists_k_eq_one_l22_22945


namespace arithmetic_sqrt_sqrt_16_eq_2_l22_22677

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l22_22677


namespace arctan_sum_l22_22007

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l22_22007


namespace gas_pipe_probability_l22_22498

-- Define the conditions as Lean hypotheses
theorem gas_pipe_probability (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hxy : x + y ≤ 100) (h25x : 25 ≤ x) (h25y : 25 ≤ y)
    (h100xy : 75 ≥ x + y) :
  ∃ (p : ℝ), p = 1/16 :=
by
  sorry

end gas_pipe_probability_l22_22498


namespace grid_entirely_black_probability_l22_22177

noncomputable def probability_entire_grid_black : ℚ := 
  let p_center_squares_black := (1 / 2) ^ 4
  let p_outer_squares_black := (1 / 2) ^ 12
  p_center_squares_black * p_outer_squares_black

theorem grid_entirely_black_probability (p : ℚ) (h : p = probability_entire_grid_black) : p = 1 / 65536 :=
by
  simp [probability_entire_grid_black]
  sorry

end grid_entirely_black_probability_l22_22177


namespace find_ab_l22_22816

theorem find_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_area_9 : (1/2) * (12 / a) * (12 / b) = 9) : 
  a * b = 8 := 
by 
  sorry

end find_ab_l22_22816


namespace arctan_sum_l22_22006

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l22_22006


namespace value_of_expression_l22_22967

theorem value_of_expression (x : ℤ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  exact rfl

end value_of_expression_l22_22967


namespace find_possible_values_l22_22440

theorem find_possible_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y, (y = (x - 3)^2 * (x + 4) / (3 * x - 4)) ∧ (y = 36 / 11 ∨ y = 468 / 23) :=
by
  sorry

end find_possible_values_l22_22440


namespace probability_excluded_probability_selected_l22_22715

-- Define the population size and the sample size
def population_size : ℕ := 1005
def sample_size : ℕ := 50
def excluded_count : ℕ := 5

-- Use these values within the theorems
theorem probability_excluded : (excluded_count : ℚ) / (population_size : ℚ) = 5 / 1005 :=
by sorry

theorem probability_selected : (sample_size : ℚ) / (population_size : ℚ) = 50 / 1005 :=
by sorry

end probability_excluded_probability_selected_l22_22715


namespace arithmetic_sqrt_of_sqrt_16_l22_22652

-- Define the arithmetic square root function
def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (real.sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22652


namespace smallest_n_l22_22022

def is_perfect_fourth (m : ℕ) : Prop := ∃ x : ℕ, m = x^4
def is_perfect_fifth (m : ℕ) : Prop := ∃ y : ℕ, m = y^5

theorem smallest_n :
  ∃ n : ℕ, n > 0 ∧ is_perfect_fourth (3 * n) ∧ is_perfect_fifth (2 * n) ∧ n = 6912 :=
by {
  sorry
}

end smallest_n_l22_22022


namespace enemies_left_undefeated_l22_22822

theorem enemies_left_undefeated (points_per_enemy : ℕ) (total_enemies : ℕ) (total_points_earned : ℕ) 
  (h1: points_per_enemy = 9) (h2: total_enemies = 11) (h3: total_points_earned = 72):
  total_enemies - (total_points_earned / points_per_enemy) = 3 :=
by
  sorry

end enemies_left_undefeated_l22_22822


namespace arithmetic_square_root_of_sqrt_16_l22_22659

theorem arithmetic_square_root_of_sqrt_16 : real.sqrt (real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l22_22659


namespace count_two_digit_numbers_less_35_l22_22067

open Nat

theorem count_two_digit_numbers_less_35 : 
  let two_digit_numbers := (finset.Ico 10 100) -- The range of two-digit numbers
  let count_satisfying := two_digit_numbers.filter (λ n, (n / 10 < 3 ∨ n % 10 < 5)).card
  count_satisfying = 55 :=
by
  -- Placeholder for the actual proof.
  sorry

end count_two_digit_numbers_less_35_l22_22067


namespace num_digits_c_l22_22461

theorem num_digits_c (a b c : ℕ) (ha : 10 ^ 2010 ≤ a ∧ a < 10 ^ 2011)
  (hb : 10 ^ 2011 ≤ b ∧ b < 10 ^ 2012)
  (h1 : a < b) (h2 : b < c)
  (div1 : ∃ k : ℕ, b + a = k * (b - a))
  (div2 : ∃ m : ℕ, c + b = m * (c - b)) :
  10 ^ 4 ≤ c ∧ c < 10 ^ 5 :=
sorry

end num_digits_c_l22_22461


namespace directrix_of_parabola_l22_22938

-- Define the given parabola equation
def parabola_eq (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 5

-- Define the expected result for the directrix
def directrix_eq : ℝ := -23 / 12

-- State the problem in Lean
theorem directrix_of_parabola : 
  (∃ d : ℝ, (∀ x y : ℝ, y = parabola_eq x → y = d) → d = directrix_eq) :=
by
  sorry

end directrix_of_parabola_l22_22938


namespace original_ratio_white_yellow_l22_22918

-- Define the given conditions
variables (W Y : ℕ)
axiom total_balls : W + Y = 64
axiom erroneous_dispatch : W = 8 * (Y + 20) / 13

-- The theorem we need to prove
theorem original_ratio_white_yellow (W Y : ℕ) (h1 : W + Y = 64) (h2 : W = 8 * (Y + 20) / 13) : W = Y :=
by sorry

end original_ratio_white_yellow_l22_22918


namespace quadratic_inequality_solution_set_l22_22770

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - x - 2 < 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end quadratic_inequality_solution_set_l22_22770


namespace probability_dice_sum_perfect_square_l22_22148

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end probability_dice_sum_perfect_square_l22_22148


namespace fraction_notation_correct_reading_decimal_correct_l22_22345

-- Define the given conditions
def fraction_notation (num denom : ℕ) : Prop :=
  num / denom = num / denom  -- Essentially stating that in fraction notation, it holds

def reading_decimal (n : ℚ) (s : String) : Prop :=
  if n = 90.58 then s = "ninety point five eight" else false -- Defining the reading rule for this specific case

-- State the theorem using the defined conditions
theorem fraction_notation_correct : fraction_notation 8 9 := 
by 
  sorry

theorem reading_decimal_correct : reading_decimal 90.58 "ninety point five eight" :=
by 
  sorry

end fraction_notation_correct_reading_decimal_correct_l22_22345


namespace volume_of_solid_bounded_by_planes_l22_22714

theorem volume_of_solid_bounded_by_planes (a : ℝ) : 
  ∃ v, v = (a ^ 3) / 6 :=
by 
  sorry

end volume_of_solid_bounded_by_planes_l22_22714


namespace find_pairs_l22_22929

theorem find_pairs (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) :
  y ∣ x^2 + 1 ∧ x^2 ∣ y^3 + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 2) :=
by
  sorry

end find_pairs_l22_22929


namespace shampoo_duration_l22_22280

-- Conditions
def rose_shampoo : ℚ := 1/3
def jasmine_shampoo : ℚ := 1/4
def daily_usage : ℚ := 1/12

-- Question
theorem shampoo_duration : (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end shampoo_duration_l22_22280


namespace f_bound_l22_22442

theorem f_bound (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1) 
  (h2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) : ∀ x : ℝ, |f x| ≤ 2 + x^2 :=
by
  sorry

end f_bound_l22_22442


namespace profit_share_of_B_l22_22741

variable {capital_A capital_B capital_C : ℚ}
variable {total_profit : ℚ}

-- Definitions based on given conditions
def investment_ratio_A := capital_A / 2000
def investment_ratio_B := capital_B / 2000
def investment_ratio_C := capital_C / 2000

def profit_share_A (P : ℚ) := (investment_ratio_A / (investment_ratio_A + investment_ratio_B + investment_ratio_C)) * P
def profit_share_B (P : ℚ) := (investment_ratio_B / (investment_ratio_A + investment_ratio_B + investment_ratio_C)) * P
def profit_share_C (P : ℚ) := (investment_ratio_C / (investment_ratio_A + investment_ratio_B + investment_ratio_C)) * P

-- Given condition of profit shares difference
axiom profit_difference_condition : profit_share_C total_profit - profit_share_A total_profit = 500

-- Definition of capitals
def capital_A := 6000
def capital_B := 8000
def capital_C := 10000

-- Theorem statement to prove
theorem profit_share_of_B : profit_share_B total_profit = 1000 := by
  sorry

end profit_share_of_B_l22_22741


namespace max_profit_l22_22423

/-- Define the cost and price of device A and device B -/
def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8

/-- Define the total number of devices -/
def total_devices : ℝ := 50

/-- Define the profits per device -/
def profit_per_A : ℝ := price_A - cost_A -- 0.3
def profit_per_B : ℝ := price_B - cost_B -- 0.4

/-- Define the function for total profit -/
def total_profit (x : ℝ) : ℝ :=
  profit_per_A * x + profit_per_B * (total_devices - x)

/-- Define the constraint -/
def constraint (x : ℝ) : Prop := 4 * x ≥ total_devices - x -- x ≥ 10

/-- The statement of the problem that needs to be proven -/
theorem max_profit :
  (total_profit x = -0.1 * x + 20) ∧ 
  ( ∀ x, constraint x → x ≥ 10 → x = 10 ∧ total_profit x = 19) :=
by
  sorry

end max_profit_l22_22423


namespace total_weight_of_hay_bales_l22_22712

theorem total_weight_of_hay_bales
  (initial_bales : Nat) (weight_per_initial_bale : Nat)
  (total_bales_now : Nat) (weight_per_new_bale : Nat) : 
  (initial_bales = 73 ∧ weight_per_initial_bale = 45 ∧ 
   total_bales_now = 96 ∧ weight_per_new_bale = 50) →
  (73 * 45 + (96 - 73) * 50 = 4435) :=
by
  sorry

end total_weight_of_hay_bales_l22_22712


namespace set_subset_find_m_l22_22057

open Set

def A (m : ℝ) : Set ℝ := {1, 3, 2 * m + 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem set_subset_find_m (m : ℝ) : (B m ⊆ A m) → (m = 1 ∨ m = 3) :=
by 
  intro h
  sorry

end set_subset_find_m_l22_22057


namespace ratio_of_white_to_yellow_balls_l22_22500

theorem ratio_of_white_to_yellow_balls (original_white original_yellow extra_yellow : ℕ) 
(h1 : original_white = 32) 
(h2 : original_yellow = 32) 
(h3 : extra_yellow = 20) : 
(original_white : ℚ) / (original_yellow + extra_yellow) = 8 / 13 := 
by
  sorry

end ratio_of_white_to_yellow_balls_l22_22500


namespace Craig_walk_distance_l22_22371

/-- Craig walked some distance from school to David's house and 0.7 miles from David's house to his own house. 
In total, Craig walked 0.9 miles. Prove that the distance Craig walked from school to David's house is 0.2 miles. 
--/
theorem Craig_walk_distance (d_school_David d_David_Craig d_total : ℝ) 
  (h1 : d_David_Craig = 0.7) 
  (h2 : d_total = 0.9) : 
  d_school_David = 0.2 :=
by 
  sorry

end Craig_walk_distance_l22_22371


namespace kitten_length_doubling_l22_22510

theorem kitten_length_doubling (initial_length : ℕ) (week2_length : ℕ) (current_length : ℕ) 
  (h1 : initial_length = 4) 
  (h2 : week2_length = 2 * initial_length) 
  (h3 : current_length = 2 * week2_length) : 
    current_length = 16 := 
by 
  sorry

end kitten_length_doubling_l22_22510


namespace value_of_a_plus_b_l22_22780

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) :
  a + b = -1 ∨ a + b = -3 :=
sorry

end value_of_a_plus_b_l22_22780


namespace range_of_a_l22_22413

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * Real.sin x - (Real.cos x)^2 ≤ 3) : -3 ≤ a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l22_22413


namespace math_problem_l22_22753

/-- The proof problem: Calculate -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11. -/
theorem math_problem : -7 * 3 - (-4 * -2) + (-9 * -6) / 3 = -11 :=
by
  sorry

end math_problem_l22_22753


namespace class_A_has_neater_scores_l22_22568

-- Definitions for the given problem conditions
def mean_Class_A : ℝ := 120
def mean_Class_B : ℝ := 120
def variance_Class_A : ℝ := 42
def variance_Class_B : ℝ := 56

-- The theorem statement to prove Class A has neater scores
theorem class_A_has_neater_scores : (variance_Class_A < variance_Class_B) := by
  sorry

end class_A_has_neater_scores_l22_22568


namespace value_of_a_plus_b_l22_22781

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) :
  a + b = -1 ∨ a + b = -3 :=
sorry

end value_of_a_plus_b_l22_22781


namespace factorial_mod_prime_l22_22228
-- Import all necessary libraries

-- State the conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The main problem statement
theorem factorial_mod_prime (n : ℕ) (h : n = 10) : factorial n % 13 = 7 := by
  sorry

end factorial_mod_prime_l22_22228


namespace value_of_a_add_b_l22_22783

theorem value_of_a_add_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
sorry

end value_of_a_add_b_l22_22783


namespace digit_in_ten_thousandths_place_of_fraction_l22_22895

theorem digit_in_ten_thousandths_place_of_fraction (n d : ℕ) (h1 : n = 5) (h2 : d = 32) :
  (∀ {x : ℕ}, (decimalExpansion n d x 4 = 5) ↔ (x = 10^4)) :=
sorry

end digit_in_ten_thousandths_place_of_fraction_l22_22895


namespace solve_for_s_l22_22355

---
theorem solve_for_s (s : ℝ) (h₀: s > 0)
(h₁ : let θ := real.pi / 3 in area = s * 3 * (s * real.sin θ))
(h₂ : area = 27 * real.sqrt 3) : s = 3 * real.sqrt 2 := by
  sorry

end solve_for_s_l22_22355


namespace Gordons_heavier_bag_weight_l22_22473

theorem Gordons_heavier_bag_weight :
  ∀ (G : ℝ), (5 * 2 = 3 + G) → G = 7 :=
by
  intro G h
  sorry

end Gordons_heavier_bag_weight_l22_22473


namespace inequality_solution_l22_22130

theorem inequality_solution (x : ℝ) : x ∈ Set.Ioo (-7 : ℝ) (7 : ℝ) ↔ (x^2 - 49) / (x + 7) < 0 :=
by 
  sorry

end inequality_solution_l22_22130


namespace probability_of_perfect_square_sum_on_two_8_sided_dice_l22_22155

-- Define the dice and possible sums
def is_sum_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def pairs_rolled : finset (ℕ × ℕ) :=
  finset.univ.filter (λ p, p.1 + p.2 ∈ {4, 9, 16})

theorem probability_of_perfect_square_sum_on_two_8_sided_dice :
  (pairs_rolled.card : ℚ) / 64 = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_on_two_8_sided_dice_l22_22155


namespace michael_clean_times_in_one_year_l22_22333

-- Definitions from the conditions
def baths_per_week : ℕ := 2
def showers_per_week : ℕ := 1
def weeks_per_year : ℕ := 52

-- Theorem statement for the proof problem
theorem michael_clean_times_in_one_year :
  (baths_per_week + showers_per_week) * weeks_per_year = 156 :=
by
  sorry

end michael_clean_times_in_one_year_l22_22333


namespace impossible_coins_l22_22609

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l22_22609


namespace problem_statement_l22_22212

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.range (n+1))

theorem problem_statement : ∃ r : ℕ, r < 13 ∧ (factorial 10) % 13 = r :=
by
  sorry

end problem_statement_l22_22212


namespace range_of_m_for_false_p_and_q_l22_22994

theorem range_of_m_for_false_p_and_q (m : ℝ) :
  (¬ (∀ x y : ℝ, (x^2 / (1 - m) + y^2 / (m + 2) = 1) ∧ ∀ x y : ℝ, (x^2 / (2 * m) + y^2 / (2 - m) = 1))) →
  (m ≤ 1 ∨ m ≥ 2) :=
sorry

end range_of_m_for_false_p_and_q_l22_22994


namespace find_mass_of_aluminum_l22_22505

noncomputable def mass_of_aluminum 
  (rho_A : ℝ) (rho_M : ℝ) (delta_m : ℝ) : ℝ :=
  rho_A * delta_m / (rho_M - rho_A)

theorem find_mass_of_aluminum :
  mass_of_aluminum 2700 8900 0.06 = 26 := by
  sorry

end find_mass_of_aluminum_l22_22505


namespace smallest_b_for_perfect_square_l22_22341

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ ∃ (n : ℤ), 3 * b + 4 = n * n ∧ b = 7 := by
  sorry

end smallest_b_for_perfect_square_l22_22341


namespace arctan_sum_l22_22010

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l22_22010


namespace maximize_profit_l22_22425

def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8
def total_devices : ℕ := 50

def profit (x : ℕ) : ℝ := (price_A - cost_A) * x + (price_B - cost_B) * (total_devices - x)

def functional_relationship (x : ℕ) : ℝ := -0.1 * x + 20

def purchase_condition (x : ℕ) : Prop := 4 * x ≥ total_devices - x

theorem maximize_profit :
    functional_relationship (10) = 19 ∧ 
    (∀ x : ℕ, purchase_condition x → functional_relationship x ≤ 19) :=
by {
    -- Proof omitted
    sorry
}

end maximize_profit_l22_22425


namespace arithmetic_sqrt_sqrt_16_eq_2_l22_22676

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l22_22676


namespace eval_expression_l22_22766

def x : ℤ := 18 / 3 * 7^2 - 80 + 4 * 7

theorem eval_expression : -x = -242 := by
  sorry

end eval_expression_l22_22766


namespace arithmetic_sqrt_of_sqrt_16_l22_22660

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22660


namespace car_travel_distance_20_minutes_l22_22405

noncomputable def train_speed_in_mph : ℝ := 80
noncomputable def car_speed_ratio : ℝ := 3/4
noncomputable def car_speed_in_mph : ℝ := car_speed_ratio * train_speed_in_mph
noncomputable def travel_time_in_hours : ℝ := 20 / 60
noncomputable def distance_travelled_by_car : ℝ := car_speed_in_mph * travel_time_in_hours

theorem car_travel_distance_20_minutes : distance_travelled_by_car = 20 := 
by 
  sorry

end car_travel_distance_20_minutes_l22_22405


namespace find_divisor_l22_22269

-- Define the conditions as hypotheses and the main problem as a theorem
theorem find_divisor (x y : ℕ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 6) / y = 6) : y = 8 := sorry

end find_divisor_l22_22269


namespace simplify_sqrt7_pow6_l22_22636

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l22_22636


namespace f_f_has_three_distinct_real_roots_l22_22295

open Polynomial

noncomputable def f (c x : ℝ) : ℝ := x^2 + 6 * x + c

theorem f_f_has_three_distinct_real_roots (c : ℝ) :
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f c (f c x) = 0) ↔
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end f_f_has_three_distinct_real_roots_l22_22295


namespace house_total_volume_l22_22493

def room_volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

def bathroom_volume := room_volume 4 2 7
def bedroom_volume := room_volume 12 10 8
def livingroom_volume := room_volume 15 12 9

def total_volume := bathroom_volume + bedroom_volume + livingroom_volume

theorem house_total_volume : total_volume = 2636 := by
  sorry

end house_total_volume_l22_22493


namespace sum_first_15_terms_l22_22566

variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function from natural numbers to real numbers

-- Define the conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def a1_plus_a15_eq_three (a : ℕ → ℝ) : Prop :=
  a 1 + a 15 = 3

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a 1 + a n)

theorem sum_first_15_terms (a : ℕ → ℝ) (h_arith: arithmetic_sequence a) (h_sum: a1_plus_a15_eq_three a) :
  sum_first_n_terms a 15 = 22.5 := by
  sorry

end sum_first_15_terms_l22_22566


namespace digit_in_ten_thousandths_place_l22_22894

theorem digit_in_ten_thousandths_place (n : ℕ) (hn : n = 5) : 
    (decimal_expansion 5/32).get (4) = some 2 :=
by
  sorry

end digit_in_ten_thousandths_place_l22_22894


namespace last_two_digits_square_l22_22457

theorem last_two_digits_square (n : ℕ) (hnz : (n % 10 ≠ 0) ∧ ((n ^ 2) % 100 = n % 10 * 11)): ((n ^ 2) % 100 = 44) :=
sorry

end last_two_digits_square_l22_22457


namespace two_digit_numbers_less_than_35_l22_22078

theorem two_digit_numbers_less_than_35 : 
  ∃ n, n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
  (let tens_digit := (n + 10) / 10 in 
   let units_digit := (n + 10) % 10 in 
   tens_digit < 3 ∨ units_digit < 5) → 
  nat.card {n | n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
                    (let tens_digit := (n + 10) / 10 in 
                     let units_digit := (n + 10) % 10 in 
                     tens_digit < 3 ∨ units_digit < 5)} = 55
:= sorry

end two_digit_numbers_less_than_35_l22_22078


namespace ten_thousandths_digit_of_five_over_thirty_two_l22_22891

theorem ten_thousandths_digit_of_five_over_thirty_two : 
  (Rat.ofInt 5 / Rat.ofInt 32).toDecimalString 5 = "0.15625" := 
by 
  sorry

end ten_thousandths_digit_of_five_over_thirty_two_l22_22891


namespace find_x_for_prime_square_l22_22054

theorem find_x_for_prime_square (x p : ℤ) (hp : Prime p) (h : 2 * x^2 - x - 36 = p^2) : x = 13 ∧ p = 17 :=
by
  sorry

end find_x_for_prime_square_l22_22054


namespace equation_of_latus_rectum_l22_22320

theorem equation_of_latus_rectum (p : ℝ) (h1 : p = 6) :
  (∀ x y : ℝ, y ^ 2 = -12 * x → x = 3) :=
sorry

end equation_of_latus_rectum_l22_22320


namespace michael_cleanings_total_l22_22334

theorem michael_cleanings_total (baths_per_week : ℕ) (showers_per_week : ℕ) (weeks_in_year : ℕ) 
  (h_baths : baths_per_week = 2) (h_showers : showers_per_week = 1) (h_weeks : weeks_in_year = 52) :
  (baths_per_week + showers_per_week) * weeks_in_year = 156 :=
by 
  -- Omitting proof as instructed.
  sorry

end michael_cleanings_total_l22_22334


namespace distance_from_star_l22_22326

def speed_of_light : ℝ := 3 * 10^5 -- km/s
def time_years : ℝ := 4 -- years
def seconds_per_year : ℝ := 3 * 10^7 -- s

theorem distance_from_star :
  let distance := speed_of_light * (time_years * seconds_per_year)
  distance = 3.6 * 10^13 :=
by
  sorry

end distance_from_star_l22_22326


namespace directrix_of_parabola_l22_22936

theorem directrix_of_parabola (h : ∀ x : ℝ, y = -3 * x ^ 2 + 6 * x - 5) : ∃ y : ℝ, y = -25 / 12 :=
by
  sorry

end directrix_of_parabola_l22_22936


namespace unique_solution_abc_l22_22037

theorem unique_solution_abc (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
(h1 : b ∣ 2^a - 1) 
(h2 : c ∣ 2^b - 1) 
(h3 : a ∣ 2^c - 1) : 
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end unique_solution_abc_l22_22037


namespace right_handed_players_total_l22_22172

theorem right_handed_players_total (total_players throwers : ℕ) (non_throwers: ℕ := total_players - throwers)
  (left_handed_non_throwers : ℕ := non_throwers / 3)
  (right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers)
  (all_throwers_right_handed : throwers = 37)
  (total_players_55 : total_players = 55)
  (one_third_left_handed : left_handed_non_throwers = non_throwers / 3)
  (right_handed_total: ℕ := throwers + right_handed_non_throwers)
  : right_handed_total = 49 := by
  sorry

end right_handed_players_total_l22_22172


namespace michael_cleanings_total_l22_22335

theorem michael_cleanings_total (baths_per_week : ℕ) (showers_per_week : ℕ) (weeks_in_year : ℕ) 
  (h_baths : baths_per_week = 2) (h_showers : showers_per_week = 1) (h_weeks : weeks_in_year = 52) :
  (baths_per_week + showers_per_week) * weeks_in_year = 156 :=
by 
  -- Omitting proof as instructed.
  sorry

end michael_cleanings_total_l22_22335


namespace common_ratio_of_geometric_sequence_l22_22207

theorem common_ratio_of_geometric_sequence (a₁ : ℝ) (S : ℕ → ℝ) (q : ℝ) (h₁ : ∀ n, S (n + 1) = S n + a₁ * q ^ n) (h₂ : 2 * S n = S (n + 1) + S (n + 2)) :
  q = -2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l22_22207


namespace radius_of_circle_l22_22704

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end radius_of_circle_l22_22704


namespace simplify_sqrt_pow_six_l22_22643

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l22_22643


namespace ratio_of_new_time_to_previous_time_l22_22351

noncomputable def distance : ℝ := 420
noncomputable def previous_time : ℝ := 7
noncomputable def speed_increase : ℝ := 40

-- Original speed
noncomputable def original_speed : ℝ := distance / previous_time

-- New speed
noncomputable def new_speed : ℝ := original_speed + speed_increase

-- New time taken to cover the same distance at the new speed
noncomputable def new_time : ℝ := distance / new_speed

-- Ratio of new time to previous time
noncomputable def time_ratio : ℝ := new_time / previous_time

theorem ratio_of_new_time_to_previous_time :
  time_ratio = 0.6 :=
by sorry

end ratio_of_new_time_to_previous_time_l22_22351


namespace janets_shampoo_days_l22_22282

-- Definitions from the problem conditions
def rose_shampoo := 1 / 3
def jasmine_shampoo := 1 / 4
def daily_usage := 1 / 12

-- Define the total shampoo and the days lasts
def total_shampoo := rose_shampoo + jasmine_shampoo
def days_lasts := total_shampoo / daily_usage

-- The theorem to be proved
theorem janets_shampoo_days : days_lasts = 7 :=
by sorry

end janets_shampoo_days_l22_22282


namespace problem_statement_l22_22210

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.range (n+1))

theorem problem_statement : ∃ r : ℕ, r < 13 ∧ (factorial 10) % 13 = r :=
by
  sorry

end problem_statement_l22_22210


namespace circle_radius_l22_22696

theorem circle_radius (r : ℝ) (hr : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 :=
by 
  sorry

end circle_radius_l22_22696


namespace units_digit_sum_of_factorials_50_l22_22557

def units_digit (n : Nat) : Nat :=
  n % 10

def sum_of_factorials (n : Nat) : Nat :=
  (List.range' 1 n).map Nat.factorial |>.sum

theorem units_digit_sum_of_factorials_50 :
  units_digit (sum_of_factorials 51) = 3 := 
sorry

end units_digit_sum_of_factorials_50_l22_22557


namespace remainder_of_3n_mod_9_l22_22027

theorem remainder_of_3n_mod_9 (n : ℕ) (h : n % 9 = 7) : (3 * n) % 9 = 3 :=
by
  sorry

end remainder_of_3n_mod_9_l22_22027


namespace solve_for_s_l22_22353

-- Definition of the given problem conditions
def parallelogram_sides_60_angle_sqrt_area (s : ℝ) :=
  ∃ (area : ℝ), (area = 27 * Real.sqrt 3) ∧
  (3 * s * s * Real.sqrt 3 = area)

-- Proof statement to demonstrate the equivalence of the theoretical and computed value of s
theorem solve_for_s (s : ℝ) : parallelogram_sides_60_angle_sqrt_area s → s = 3 :=
by
  intro h
  sorry

end solve_for_s_l22_22353


namespace find_jack_euros_l22_22985

theorem find_jack_euros (E : ℕ) (h1 : 45 + 2 * E = 117) : E = 36 :=
by
  sorry

end find_jack_euros_l22_22985


namespace find_unit_vector_l22_22771

theorem find_unit_vector (a b : ℝ) : 
  a^2 + b^2 = 1 ∧ 3 * a + 4 * b = 0 →
  (a = 4 / 5 ∧ b = -3 / 5) ∨ (a = -4 / 5 ∧ b = 3 / 5) :=
by sorry

end find_unit_vector_l22_22771


namespace range_of_a_l22_22948

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x + 1) ^ 2 > 4 → x > a) → a ≥ 1 := sorry

end range_of_a_l22_22948


namespace positive_int_sum_square_l22_22327

theorem positive_int_sum_square (M : ℕ) (h_pos : 0 < M) (h_eq : M^2 + M = 12) : M = 3 :=
by
  sorry

end positive_int_sum_square_l22_22327


namespace height_relationship_height_at_90_l22_22452

noncomputable def f (x : ℝ) : ℝ := (1/2) * x

theorem height_relationship :
  (∀ x : ℝ, (x = 10 -> f x = 5) ∧ (x = 30 -> f x = 15) ∧ (x = 50 -> f x = 25) ∧ (x = 70 -> f x = 35)) → (∀ x : ℝ, f x = (1/2) * x) :=
by
  sorry

theorem height_at_90 :
  f 90 = 45 :=
by
  sorry

end height_relationship_height_at_90_l22_22452


namespace relationship_l22_22098

-- Define sequences
variable (a b : ℕ → ℝ)

-- Define conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 1 < m → m < n → a m = a 1 + (m - 1) * (a n - a 1) / (n - 1)

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 1 < m → m < n → b m = b 1 * (b n / b 1)^(m - 1) / (n - 1)

noncomputable def sequences_conditions : Prop :=
  a 1 = b 1 ∧ a 1 > 0 ∧ ∀ n, a n = b n ∧ b n > 0

-- The main theorem
theorem relationship (h: sequences_conditions a b) : ∀ m n : ℕ, 1 < m → m < n → a m ≥ b m := 
by
  sorry

end relationship_l22_22098


namespace ratio_both_basketball_volleyball_l22_22975

variable (total_students : ℕ) (play_basketball : ℕ) (play_volleyball : ℕ) (play_neither : ℕ) (play_both : ℕ)

theorem ratio_both_basketball_volleyball (h1 : total_students = 20)
    (h2 : play_basketball = 20 / 2)
    (h3 : play_volleyball = (2 * 20) / 5)
    (h4 : play_neither = 4)
    (h5 : total_students - play_neither = play_basketball + play_volleyball - play_both) :
    play_both / total_students = 1 / 10 :=
by
  sorry

end ratio_both_basketball_volleyball_l22_22975


namespace find_common_chord_l22_22238

variable (x y : ℝ)

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 3*y = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*x + 2*y + 1 = 0
def common_chord (x y : ℝ) := 6*x + y - 1 = 0

theorem find_common_chord (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : common_chord x y :=
by
  sorry

end find_common_chord_l22_22238


namespace isosceles_triangle_area_l22_22562

theorem isosceles_triangle_area 
  (x y : ℝ)
  (h_perimeter : 2*y + 2*x = 32)
  (h_height : ∃ h : ℝ, h = 8 ∧ y^2 = x^2 + h^2) :
  ∃ area : ℝ, area = 48 :=
by
  sorry

end isosceles_triangle_area_l22_22562


namespace gross_pay_is_450_l22_22169

def net_pay : ℤ := 315
def taxes : ℤ := 135
def gross_pay : ℤ := net_pay + taxes

theorem gross_pay_is_450 : gross_pay = 450 := by
  sorry

end gross_pay_is_450_l22_22169


namespace factorial_mod_10_l22_22222

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the problem statement
theorem factorial_mod_10 : factorial 10 % 13 = 7 :=
by sorry

end factorial_mod_10_l22_22222


namespace kitten_length_doubling_l22_22509

theorem kitten_length_doubling (initial_length : ℕ) (week2_length : ℕ) (current_length : ℕ) 
  (h1 : initial_length = 4) 
  (h2 : week2_length = 2 * initial_length) 
  (h3 : current_length = 2 * week2_length) : 
    current_length = 16 := 
by 
  sorry

end kitten_length_doubling_l22_22509


namespace contractor_absent_days_l22_22900

variable (x y : ℕ)  -- Number of days worked and absent, both are natural numbers

-- Conditions from the problem
def total_days (x y : ℕ) : Prop := x + y = 30
def total_payment (x y : ℕ) : Prop := 25 * x - 75 * y / 10 = 360

-- Main statement
theorem contractor_absent_days (h1 : total_days x y) (h2 : total_payment x y) : y = 12 :=
by
  sorry

end contractor_absent_days_l22_22900


namespace number_of_surjective_non_decreasing_functions_l22_22256

theorem number_of_surjective_non_decreasing_functions (A B : Finset ℝ) (hA : A.card = 100) (hB : B.card = 50) :
  let f : ℕ → ℕ := λ n, (Nat.choose 99 49) in ∃ f : ℝ → ℝ, (∀ x ∈ A, f(x) ∈ B) ∧ (∀ x y ∈ A, x ≤ y → f(x) ≤ f(y)) → f = Nat.choose 99 49 :=
by sorry

end number_of_surjective_non_decreasing_functions_l22_22256


namespace f_2007_l22_22962

noncomputable def f : ℕ → ℝ :=
  sorry

axiom functional_eq (x y : ℕ) : f (x + y) = f x * f y

axiom f_one : f 1 = 2

theorem f_2007 : f 2007 = 2 ^ 2007 :=
by
  sorry

end f_2007_l22_22962


namespace vicki_donated_fraction_l22_22435

/-- Given Jeff had 300 pencils and donated 30% of them, and Vicki had twice as many pencils as Jeff originally 
    had, and there are 360 pencils remaining altogether after both donations,
    prove that Vicki donated 3/4 of her pencils. -/
theorem vicki_donated_fraction : 
  let jeff_pencils := 300
  let jeff_donated := jeff_pencils * 0.30
  let jeff_remaining := jeff_pencils - jeff_donated
  let vicki_pencils := 2 * jeff_pencils
  let total_remaining := 360
  let vicki_remaining := total_remaining - jeff_remaining
  let vicki_donated := vicki_pencils - vicki_remaining
  vicki_donated / vicki_pencils = 3 / 4 :=
by
  -- Proof needs to be inserted here
  sorry

end vicki_donated_fraction_l22_22435


namespace factorial_mod_10_eq_6_l22_22213

theorem factorial_mod_10_eq_6 : (10! % 13) = 6 := by
  sorry

end factorial_mod_10_eq_6_l22_22213


namespace radius_of_spheres_in_cone_l22_22275

def base_radius := 8
def cone_height := 15
def num_spheres := 3
def spheres_are_tangent := true

theorem radius_of_spheres_in_cone :
  ∃ (r : ℝ), r = (280 - 100 * Real.sqrt 3) / 121 :=
sorry

end radius_of_spheres_in_cone_l22_22275


namespace parabola_vertex_l22_22681

-- Define the condition: the equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 4 * y + 3 * x + 1 = 0

-- Define the statement: prove that the vertex of the parabola is (1, -2)
theorem parabola_vertex :
  parabola_equation 1 (-2) :=
by
  sorry

end parabola_vertex_l22_22681


namespace sum_of_ages_l22_22125

-- Definitions for Robert's and Maria's current ages
variables (R M : ℕ)

-- Conditions based on the problem statement
theorem sum_of_ages
  (h1 : R = M + 8)
  (h2 : R + 5 = 3 * (M - 3)) :
  R + M = 30 :=
by
  sorry

end sum_of_ages_l22_22125


namespace num_suitable_two_digit_numbers_l22_22059

/-- 
How many two-digit numbers have at least one digit that is smaller than the corresponding digit in the number 35?
-/
theorem num_suitable_two_digit_numbers : 
  let two_digit_numbers := { n : ℕ | 10 ≤ n ∧ n ≤ 99 },
      suitable_numbers := { n ∈ two_digit_numbers | (n / 10 < 3) ∨ (n % 10 < 5) } in
  suitable_numbers.card = 55 :=
by
  sorry

end num_suitable_two_digit_numbers_l22_22059


namespace order_of_f0_f1_f_2_l22_22080

noncomputable def f (m x : ℝ) := (m-1) * x^2 + 6 * m * x + 2

theorem order_of_f0_f1_f_2 (m : ℝ) (h_even : ∀ x : ℝ, f m x = f m (-x)) :
  m = 0 → f m (-2) < f m 1 ∧ f m 1 < f m 0 :=
by 
  sorry

end order_of_f0_f1_f_2_l22_22080


namespace largest_pos_integer_binary_op_l22_22347

def binary_op (n : ℤ) : ℤ := n - n * 5

theorem largest_pos_integer_binary_op :
  ∃ n : ℕ, binary_op n < 14 ∧ ∀ m : ℕ, binary_op m < 14 → m ≤ 1 :=
sorry

end largest_pos_integer_binary_op_l22_22347


namespace count_two_digit_numbers_less_35_l22_22065

open Nat

theorem count_two_digit_numbers_less_35 : 
  let two_digit_numbers := (finset.Ico 10 100) -- The range of two-digit numbers
  let count_satisfying := two_digit_numbers.filter (λ n, (n / 10 < 3 ∨ n % 10 < 5)).card
  count_satisfying = 55 :=
by
  -- Placeholder for the actual proof.
  sorry

end count_two_digit_numbers_less_35_l22_22065


namespace probability_of_perfect_square_sum_on_two_8_sided_dice_l22_22156

-- Define the dice and possible sums
def is_sum_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def pairs_rolled : finset (ℕ × ℕ) :=
  finset.univ.filter (λ p, p.1 + p.2 ∈ {4, 9, 16})

theorem probability_of_perfect_square_sum_on_two_8_sided_dice :
  (pairs_rolled.card : ℚ) / 64 = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_on_two_8_sided_dice_l22_22156


namespace medicine_types_count_l22_22096

theorem medicine_types_count (n : ℕ) (hn : n = 5) : (Nat.choose n 2 = 10) :=
by
  sorry

end medicine_types_count_l22_22096


namespace impossible_coins_l22_22602

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l22_22602


namespace arithmetic_sqrt_sqrt_16_eq_2_l22_22679

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l22_22679


namespace algebraic_expression_value_l22_22268

-- Define the conditions given
variables {a b : ℝ}
axiom h1 : a ≠ b
axiom h2 : a^2 - 8 * a + 5 = 0
axiom h3 : b^2 - 8 * b + 5 = 0

-- Main theorem to prove the expression equals -20
theorem algebraic_expression_value:
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 :=
sorry

end algebraic_expression_value_l22_22268


namespace cooking_time_at_least_l22_22491

-- Definitions based on conditions
def total_potatoes : ℕ := 35
def cooked_potatoes : ℕ := 11
def time_per_potato : ℕ := 7 -- in minutes
def salad_time : ℕ := 15 -- in minutes

-- The statement to prove
theorem cooking_time_at_least (oven_capacity : ℕ) :
  ∃ t : ℕ, t ≥ salad_time :=
by
  sorry

end cooking_time_at_least_l22_22491


namespace units_digit_is_six_l22_22449

theorem units_digit_is_six (n : ℤ) (h : (n^2 / 10 % 10) = 7) : (n^2 % 10) = 6 :=
by sorry

end units_digit_is_six_l22_22449


namespace cauchy_normal_ratio_l22_22300

noncomputable section

open ProbabilityTheory MeasureTheory

theorem cauchy_normal_ratio {C X Y : ℝ} 
  (hC : C ~ Cauchy 0 1) 
  (hX : X ~ Normal 0 1) 
  (hY : Y ~ Normal 0 1) 
  (h_indep : Indep X Y) :
  C ∼ (X / Y) ∧ (X / Y) ∼ (X / abs Y) :=
sorry

end cauchy_normal_ratio_l22_22300


namespace complex_powers_sum_zero_l22_22378

theorem complex_powers_sum_zero (i : ℂ) (h : i^2 = -1) : i^2023 + i^2024 + i^2025 + i^2026 = 0 :=
by
  sorry

end complex_powers_sum_zero_l22_22378


namespace smaller_variance_stability_l22_22344

variable {α : Type*}
variable [Nonempty α]

def same_average (X Y : α → ℝ) (avg : ℝ) : Prop := 
  (∀ x, X x = avg) ∧ (∀ y, Y y = avg)

def smaller_variance_is_stable (X Y : α → ℝ) : Prop := 
  (X = Y)

theorem smaller_variance_stability {X Y : α → ℝ} (avg : ℝ) :
  same_average X Y avg → smaller_variance_is_stable X Y :=
by sorry

end smaller_variance_stability_l22_22344


namespace exists_real_m_l22_22052

noncomputable def f (a : ℝ) (x : ℝ) := 4 * x + a * x ^ 2 - (2 / 3) * x ^ 3
noncomputable def g (x : ℝ) := 2 * x + (1 / 3) * x ^ 3

theorem exists_real_m (a : ℝ) (t : ℝ) (x1 x2 : ℝ) :
  (-1 : ℝ) ≤ a ∧ a ≤ 1 →
  (-1 : ℝ) ≤ t ∧ t ≤ 1 →
  f a x1 = g x1 ∧ f a x2 = g x2 →
  x1 ≠ 0 ∧ x2 ≠ 0 →
  x1 ≠ x2 →
  ∃ m : ℝ, (m ≥ 2 ∨ m ≤ -2) ∧ m^2 + t * m + 1 ≥ |x1 - x2| :=
sorry

end exists_real_m_l22_22052


namespace final_answer_for_m_l22_22797

noncomputable def proof_condition_1 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

noncomputable def proof_condition_2 (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

noncomputable def proof_condition_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1*x2 + y1*y2 = 0

theorem final_answer_for_m :
  (∀ (x y m : ℝ), proof_condition_1 x y m) →
  (∀ (x y : ℝ), proof_condition_2 x y) →
  (∀ (x1 y1 x2 y2 : ℝ), proof_condition_perpendicular x1 y1 x2 y2) →
  m = 12 / 5 :=
sorry

end final_answer_for_m_l22_22797


namespace lower_limit_brother_opinion_l22_22519

variables (w B : ℝ)

-- Conditions
-- Arun's weight is between 61 and 72 kg
def arun_cond := 61 < w ∧ w < 72
-- Arun's brother's opinion: greater than B, less than 70
def brother_cond := B < w ∧ w < 70
-- Arun's mother's view: not greater than 64
def mother_cond :=  w ≤ 64

-- Given the average
def avg_weight := 63

theorem lower_limit_brother_opinion (h_arun : arun_cond w) (h_brother: brother_cond w B) (h_mother: mother_cond w) (h_avg: avg_weight = (B + 64)/2) : 
  B = 62 :=
sorry

end lower_limit_brother_opinion_l22_22519


namespace shampoo_duration_l22_22276

theorem shampoo_duration
  (rose_shampoo : ℚ := 1/3)
  (jasmine_shampoo : ℚ := 1/4)
  (daily_usage : ℚ := 1/12) :
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := 
by
  sorry

end shampoo_duration_l22_22276


namespace average_marks_l22_22745

theorem average_marks {n : ℕ} (h1 : 5 * 74 + 104 = n * 79) : n = 6 :=
by
  sorry

end average_marks_l22_22745


namespace unique_solution_c_min_l22_22944

theorem unique_solution_c_min (x y : ℝ) (c : ℝ)
  (h1 : 2 * (x+7)^2 + (y-4)^2 = c)
  (h2 : (x+4)^2 + 2 * (y-7)^2 = c) :
  c = 6 :=
sorry

end unique_solution_c_min_l22_22944


namespace sum_of_cosines_bounds_l22_22875

theorem sum_of_cosines_bounds (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ π / 2)
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ π / 2)
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ π / 2)
  (h₄ : 0 ≤ x₄ ∧ x₄ ≤ π / 2)
  (h₅ : 0 ≤ x₅ ∧ x₅ ≤ π / 2)
  (sum_sines_eq : Real.sin x₁ + Real.sin x₂ + Real.sin x₃ + Real.sin x₄ + Real.sin x₅ = 3) : 
  2 ≤ Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ ∧ 
      Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ ≤ 4 :=
by
  sorry

end sum_of_cosines_bounds_l22_22875


namespace arithmetic_sequence_suff_nec_straight_line_l22_22841

variable (n : ℕ) (P_n : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ m : ℕ, a (m + 1) = a m + d

def lies_on_straight_line (P : ℕ → ℝ) : Prop :=
  ∃ m b, ∀ n, P n = m * n + b

theorem arithmetic_sequence_suff_nec_straight_line
  (h_n : 0 < n)
  (h_arith : arithmetic_sequence P_n) :
  lies_on_straight_line P_n ↔ arithmetic_sequence P_n :=
sorry

end arithmetic_sequence_suff_nec_straight_line_l22_22841


namespace value_of_K_l22_22481

theorem value_of_K (K: ℕ) : 4^5 * 2^3 = 2^K → K = 13 := by
  sorry

end value_of_K_l22_22481


namespace equation_equivalence_and_rst_l22_22865

theorem equation_equivalence_and_rst 
  (a x y c : ℝ) 
  (r s t : ℤ) 
  (h1 : r = 3) 
  (h2 : s = 1) 
  (h3 : t = 5)
  (h_eq1 : a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1)) :
  (a^r * x - a^s) * (a^t * y - a^3) = a^5 * c^5 ∧ r * s * t = 15 :=
by
  sorry

end equation_equivalence_and_rst_l22_22865


namespace age_transition_l22_22888

theorem age_transition (initial_ages : List ℕ) : 
  initial_ages = [19, 34, 37, 42, 48] →
  (∃ x, 0 < x ∧ x < 10 ∧ 
  new_ages = List.map (fun age => age + x) initial_ages ∧ 
  new_ages = [25, 40, 43, 48, 54]) →
  x = 6 :=
by
  intros h_initial_ages h_exist_x
  sorry

end age_transition_l22_22888


namespace average_weight_l22_22883

theorem average_weight (men women : ℕ) (avg_weight_men avg_weight_women : ℝ) (total_people : ℕ) (combined_avg_weight : ℝ) 
  (h1 : men = 8) (h2 : avg_weight_men = 190) (h3 : women = 6) (h4 : avg_weight_women = 120) (h5 : total_people = 14) 
  (h6 : (men * avg_weight_men + women * avg_weight_women) / total_people = combined_avg_weight) : combined_avg_weight = 160 := 
  sorry

end average_weight_l22_22883


namespace pieces_from_sister_calculation_l22_22206

-- Definitions for the conditions
def pieces_from_neighbors : ℕ := 5
def pieces_per_day : ℕ := 9
def duration : ℕ := 2

-- Definition to calculate the total number of pieces Emily ate
def total_pieces : ℕ := pieces_per_day * duration

-- Proof Problem: Prove Emily received 13 pieces of candy from her older sister
theorem pieces_from_sister_calculation :
  ∃ (pieces_from_sister : ℕ), pieces_from_sister = total_pieces - pieces_from_neighbors ∧ pieces_from_sister = 13 :=
by
  sorry

end pieces_from_sister_calculation_l22_22206


namespace arithmetic_sequence_sum_l22_22443

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (hS3 : S 3 = 12) (hS6 : S 6 = 42) 
  (h_arith_seq : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) :
  a 10 + a 11 + a 12 = 66 :=
sorry

end arithmetic_sequence_sum_l22_22443


namespace unique_solution_t_interval_l22_22038

theorem unique_solution_t_interval (x y z v t : ℝ) :
  (x + y + z + v = 0) →
  ((x * y + y * z + z * v) + t * (x * z + x * v + y * v) = 0) →
  (t > (3 - Real.sqrt 5) / 2) ∧ (t < (3 + Real.sqrt 5) / 2) :=
by
  intro h1 h2
  sorry

end unique_solution_t_interval_l22_22038


namespace circle_radius_l22_22694

theorem circle_radius (r : ℝ) (hr : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 :=
by 
  sorry

end circle_radius_l22_22694


namespace number_of_b_values_l22_22033

-- Let's define the conditions and the final proof required.
def inequations (x b : ℤ) : Prop := 
  (3 * x > 4 * x - 4) ∧
  (4 * x - b > -8) ∧
  (5 * x < b + 13)

theorem number_of_b_values :
  (∀ x : ℤ, 1 ≤ x → x ≠ 3 → ¬ inequations x b) →
  (∃ (b_values : Finset ℤ), 
      (∀ b ∈ b_values, inequations 3 b) ∧ 
      (b_values.card = 7)) :=
sorry

end number_of_b_values_l22_22033


namespace range_of_m_l22_22972

noncomputable def proof_problem (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1/x + 2/y = 1) : Prop :=
  ∃ x y : ℝ, (0 < x) ∧ (0 < y) ∧ (1/x + 2/y = 1) ∧ (x + y / 2 < m^2 + 3 * m) ↔ (m < -4 ∨ m > 1)

theorem range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1/x + 2/y = 1) :
  proof_problem x y m hx hy hxy :=
sorry

end range_of_m_l22_22972


namespace poker_flush_probability_l22_22475

theorem poker_flush_probability :
  let total_ways := Nat.choose 52 5
  let flush_ways := 4 * Nat.choose 13 5
  (flush_ways : ℚ) / total_ways = 103 / 51980 :=
by
  sorry

end poker_flush_probability_l22_22475


namespace num_suitable_two_digit_numbers_l22_22060

/-- 
How many two-digit numbers have at least one digit that is smaller than the corresponding digit in the number 35?
-/
theorem num_suitable_two_digit_numbers : 
  let two_digit_numbers := { n : ℕ | 10 ≤ n ∧ n ≤ 99 },
      suitable_numbers := { n ∈ two_digit_numbers | (n / 10 < 3) ∨ (n % 10 < 5) } in
  suitable_numbers.card = 55 :=
by
  sorry

end num_suitable_two_digit_numbers_l22_22060


namespace simplify_sqrt_power_l22_22627

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l22_22627


namespace factorial_mod_prime_l22_22229
-- Import all necessary libraries

-- State the conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The main problem statement
theorem factorial_mod_prime (n : ℕ) (h : n = 10) : factorial n % 13 = 7 := by
  sorry

end factorial_mod_prime_l22_22229


namespace max_take_home_pay_income_l22_22419

theorem max_take_home_pay_income (x : ℤ) : 
  (1000 * 2 * 50) - 20 * 50^2 = 100000 := 
by 
  sorry

end max_take_home_pay_income_l22_22419


namespace shampoo_duration_l22_22277

theorem shampoo_duration
  (rose_shampoo : ℚ := 1/3)
  (jasmine_shampoo : ℚ := 1/4)
  (daily_usage : ℚ := 1/12) :
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := 
by
  sorry

end shampoo_duration_l22_22277


namespace vertex_x_coordinate_l22_22685

theorem vertex_x_coordinate (a b c : ℝ) :
  (∀ x, x = 0 ∨ x = 4 ∨ x = 7 →
    (0 ≤ x ∧ x ≤ 7 →
      (x = 0 → c = 1) ∧
      (x = 4 → 16 * a + 4 * b + c = 1) ∧
      (x = 7 → 49 * a + 7 * b + c = 5))) →
  (2 * x = 2 * 2 - b / a) ∧ (0 ≤ x ∧ x ≤ 7) :=
sorry

end vertex_x_coordinate_l22_22685


namespace part_a_part_b_l22_22907

-- Part (a)
theorem part_a (a b c : ℚ) (z : ℚ) (h : a * z^2 + b * z + c = 0) (n : ℕ) (hn : n > 0) :
  ∃ f : ℚ → ℚ, z = f (z^n) :=
sorry

-- Part (b)
theorem part_b (x : ℚ) (h : x ≠ 0) :
  x = (x^3 + (x + 1/x)) / ((x + 1/x)^2 - 1) :=
sorry

end part_a_part_b_l22_22907


namespace number_of_divisors_of_180_l22_22552

theorem number_of_divisors_of_180 : 
   (nat.coprime 2 3 ∧ nat.coprime 3 5 ∧ nat.coprime 5 2 ∧ 180 = 2^2 * 3^2 * 5^1) →
   (nat.divisors_count 180 = 18) :=
by
  sorry

end number_of_divisors_of_180_l22_22552


namespace compound_ratio_is_one_fourteenth_l22_22902

theorem compound_ratio_is_one_fourteenth :
  (2 / 3) * (6 / 7) * (1 / 3) * (3 / 8) = 1 / 14 :=
by sorry

end compound_ratio_is_one_fourteenth_l22_22902


namespace carSpeedIs52mpg_l22_22179

noncomputable def carSpeed (fuelConsumptionKMPL : ℕ) -- 32 kilometers per liter
                           (gallonToLiter : ℝ)        -- 1 gallon = 3.8 liters
                           (fuelDecreaseGallons : ℝ)  -- 3.9 gallons
                           (timeHours : ℝ)            -- 5.7 hours
                           (kmToMiles : ℝ)            -- 1 mile = 1.6 kilometers
                           : ℝ :=
  let totalLiters := fuelDecreaseGallons * gallonToLiter
  let totalKilometers := totalLiters * fuelConsumptionKMPL
  let totalMiles := totalKilometers / kmToMiles
  totalMiles / timeHours

theorem carSpeedIs52mpg : carSpeed 32 3.8 3.9 5.7 1.6 = 52 := sorry

end carSpeedIs52mpg_l22_22179


namespace rhombus_area_l22_22526

theorem rhombus_area 
  (a : ℝ) (d1 d2 : ℝ)
  (h_side : a = Real.sqrt 113)
  (h_diagonal_diff : abs (d1 - d2) = 8)
  (h_geq : d1 ≠ d2) : 
  (a^2 * d1 * d2 / 2 = 194) :=
sorry -- Proof to be completed

end rhombus_area_l22_22526


namespace ratio_problem_l22_22165

theorem ratio_problem (x : ℕ) : (20 / 1 : ℝ) = (x / 10 : ℝ) → x = 200 := by
  sorry

end ratio_problem_l22_22165


namespace arithmetic_square_root_of_sqrt_16_l22_22668

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l22_22668


namespace circle_center_and_radius_l22_22455

theorem circle_center_and_radius (x y : ℝ) (h : x^2 + y^2 - 6*x = 0) :
  (∃ c : ℝ × ℝ, c = (3, 0)) ∧ (∃ r : ℝ, r = 3) := 
sorry

end circle_center_and_radius_l22_22455


namespace theater_ticket_problem_l22_22521

noncomputable def total_cost_proof (x : ℝ) : Prop :=
  let cost_adult_tickets := 10 * x
  let cost_child_tickets := 8 * (x / 2)
  let cost_senior_tickets := 4 * (0.75 * x)
  cost_adult_tickets + cost_child_tickets + cost_senior_tickets = 58.65

theorem theater_ticket_problem (x : ℝ) (h : 6 * x + 5 * (x / 2) + 3 * (0.75 * x) = 42) : 
  total_cost_proof x :=
by
  sorry

end theater_ticket_problem_l22_22521


namespace profit_ratio_l22_22469

-- Definitions based on conditions
-- Let A_orig and B_orig represent the original profits of stores A and B
-- after increase and decrease respectively, they become equal

variable (A_orig B_orig : ℝ)
variable (h1 : (1.2 * A_orig) = (0.9 * B_orig))

-- Prove that the original profit of store A was 75% of the profit of store B
theorem profit_ratio (h1 : 1.2 * A_orig = 0.9 * B_orig) : A_orig = 0.75 * B_orig :=
by
  -- Insert proof here
  sorry

end profit_ratio_l22_22469


namespace probability_two_8sided_dice_sum_perfect_square_l22_22145

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probabilityPerfectSquareSum (dice_sides : ℕ) (perfect_squares : List ℕ) : ℚ :=
  let outcomes := (dice_sides * dice_sides)
  let favorable_outcomes := perfect_squares.sum (λ ps, 
    (List.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd = ps) ((List.range dice_sides).product (List.range dice_sides))).length)
  favorable_outcomes /. outcomes

theorem probability_two_8sided_dice_sum_perfect_square :
  probabilityPerfectSquareSum 8 [4, 9, 16] = 3 / 16 := sorry

end probability_two_8sided_dice_sum_perfect_square_l22_22145


namespace coins_with_specific_probabilities_impossible_l22_22606

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l22_22606


namespace factorial_mod_10_eq_6_l22_22214

theorem factorial_mod_10_eq_6 : (10! % 13) = 6 := by
  sorry

end factorial_mod_10_eq_6_l22_22214


namespace range_of_a12_l22_22427

variable (a : ℕ → ℝ)
variable (a1 d : ℝ)

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

variable (h_arith_seq : arithmetic_seq a a1 d)
variable (h_a8 : a 7 ≥ 15)
variable (h_a9 : a 8 ≤ 13)

theorem range_of_a12 : ∀ a1 d, (arithmetic_seq a a1 d) → (a 7 ≥ 15) → (a 8 ≤ 13) → (a 11 ≤ 7) :=
by
  intro a1 d h_arith_seq h_a8 h_a9
  sorry

end range_of_a12_l22_22427


namespace ten_factorial_mod_thirteen_l22_22223

open Nat

theorem ten_factorial_mod_thirteen :
  (10! % 13) = 6 := by
  sorry

end ten_factorial_mod_thirteen_l22_22223


namespace terminal_side_second_or_third_quadrant_l22_22960

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

end terminal_side_second_or_third_quadrant_l22_22960


namespace triangle_area_l22_22759

theorem triangle_area (h b : ℝ) (Hhb : h < b) :
  let P := (0, b)
  let B := (b, 0)
  let D := (h, h)
  let PD := b - h
  let DB := b - h
  1 / 2 * PD * DB = 1 / 2 * (b - h) ^ 2 := by 
  sorry

end triangle_area_l22_22759


namespace line_slope_l22_22021

theorem line_slope (x y : ℝ) : 3 * y - (1 / 2) * x = 9 → ∃ m, m = 1 / 6 :=
by
  sorry

end line_slope_l22_22021


namespace solve_equation_1_solve_equation_2_l22_22861

theorem solve_equation_1 :
  ∀ x : ℝ, 2 * x^2 - 4 * x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  intro x
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, x^2 - 6 * x - 6 = 0 ↔ (x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15) :=
by
  intro x
  sorry

end solve_equation_1_solve_equation_2_l22_22861


namespace graph_passes_through_quadrants_l22_22322

theorem graph_passes_through_quadrants :
  (∃ x, x > 0 ∧ -1/2 * x + 2 > 0) ∧  -- Quadrant I condition: x > 0, y > 0
  (∃ x, x < 0 ∧ -1/2 * x + 2 > 0) ∧  -- Quadrant II condition: x < 0, y > 0
  (∃ x, x > 0 ∧ -1/2 * x + 2 < 0) := -- Quadrant IV condition: x > 0, y < 0
by
  sorry

end graph_passes_through_quadrants_l22_22322


namespace reduction_percentage_price_increase_l22_22181

-- Proof Problem 1: Reduction Percentage
theorem reduction_percentage (a : ℝ) (h₁ : (50 * (1 - a)^2 = 32)) : a = 0.2 := by
  sorry

-- Proof Problem 2: Price Increase for Daily Profit
theorem price_increase 
  (x : ℝ)
  (profit_per_kg : ℝ := 10)
  (initial_sales : ℕ := 500)
  (sales_decrease_per_unit : ℝ := 20)
  (required_profit : ℝ := 6000)
  (h₁ : (10 + x) * (initial_sales - sales_decrease_per_unit * x) = required_profit) : 
  x = 5 := by
  sorry

end reduction_percentage_price_increase_l22_22181


namespace rectangular_garden_shorter_side_length_l22_22917

theorem rectangular_garden_shorter_side_length
  (a b : ℕ)
  (h1 : 2 * a + 2 * b = 46)
  (h2 : a * b = 108) :
  b = 9 :=
by 
  sorry

end rectangular_garden_shorter_side_length_l22_22917


namespace range_g_l22_22386

noncomputable def g (x : ℝ) : ℝ := x / (x^2 - 2 * x + 2)

theorem range_g : Set.Icc (-(1:ℝ)/2) (1/2) = {y : ℝ | ∃ x : ℝ, g x = y} := 
by
  sorry

end range_g_l22_22386


namespace company_blocks_l22_22821

noncomputable def number_of_blocks (workers_per_block total_budget gift_cost : ℕ) : ℕ :=
  (total_budget / gift_cost) / workers_per_block

theorem company_blocks :
  number_of_blocks 200 6000 2 = 15 :=
by
  sorry

end company_blocks_l22_22821


namespace distinct_roots_condition_l22_22294

noncomputable def f (x c : ℝ) : ℝ := x^2 + 6*x + c

theorem distinct_roots_condition (c : ℝ) :
  (∀x : ℝ, f (f x c) = 0 → ∃ a b : ℝ, (a ≠ b) ∧ f x c = a * (x - b) * (x - c) ) →
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end distinct_roots_condition_l22_22294


namespace unique_geometric_progression_12_a_b_ab_l22_22763

noncomputable def geometric_progression_12_a_b_ab : Prop :=
  ∃ (a b : ℝ), ∃ r : ℝ, a = 12 * r ∧ b = 12 * r^2 ∧ 12 * r * (12 * r^2) = 144 * r^3

theorem unique_geometric_progression_12_a_b_ab :
  ∃! (a b : ℝ), ∃ r : ℝ, a = 12 * r ∧ b = 12 * r^2 ∧ 12 * r * (12 * r^2) = 144 * r^3 :=
by
  sorry

end unique_geometric_progression_12_a_b_ab_l22_22763


namespace sequence_sum_l22_22984

-- Defining the sequence terms
variables (J K L M N O P Q R S : ℤ)
-- Condition N = 7
def N_value : Prop := N = 7
-- Condition sum of any four consecutive terms is 40
def sum_of_consecutive : Prop := 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40

-- The main theorem stating J + S = 40 given the conditions
theorem sequence_sum (N_value : N = 7) (sum_of_consecutive : 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40) : 
  J + S = 40 := sorry

end sequence_sum_l22_22984


namespace value_of_a_plus_b_l22_22776

theorem value_of_a_plus_b (a b : Int) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
by
  sorry

end value_of_a_plus_b_l22_22776


namespace determine_house_numbers_l22_22969

-- Definitions based on the conditions given
def even_numbered_side (n : ℕ) : Prop :=
  n % 2 = 0

def sum_balanced (n : ℕ) (house_numbers : List ℕ) : Prop :=
  let left_sum := house_numbers.take n |>.sum
  let right_sum := house_numbers.drop (n + 1) |>.sum
  left_sum = right_sum

def house_constraints (n : ℕ) : Prop :=
  50 < n ∧ n < 500

-- Main theorem statement
theorem determine_house_numbers : 
  ∃ (n : ℕ) (house_numbers : List ℕ), 
    even_numbered_side n ∧ 
    house_constraints n ∧ 
    sum_balanced n house_numbers :=
  sorry

end determine_house_numbers_l22_22969


namespace angle_A_is_120_degrees_l22_22308

theorem angle_A_is_120_degrees
  (b c l_a : ℝ)
  (h : (1 / b) + (1 / c) = 1 / l_a) :
  ∃ A : ℝ, A = 120 :=
by
  sorry

end angle_A_is_120_degrees_l22_22308


namespace complex_number_modulus_l22_22091

open Complex

theorem complex_number_modulus :
  ∀ x : ℂ, x + I = (2 - I) / I → abs x = Real.sqrt 10 := by
  sorry

end complex_number_modulus_l22_22091


namespace hansel_album_duration_l22_22808

theorem hansel_album_duration 
    (initial_songs : ℕ)
    (additional_songs : ℕ)
    (duration_per_song : ℕ)
    (h_initial : initial_songs = 25)
    (h_additional : additional_songs = 10)
    (h_duration : duration_per_song = 3):
    initial_songs * duration_per_song + additional_songs * duration_per_song = 105 := 
by
  sorry

end hansel_album_duration_l22_22808


namespace last_digit_of_prime_l22_22922

theorem last_digit_of_prime (n : ℕ) (h1 : 859433 = 214858 * 4 + 1) : (2 ^ 859433 - 1) % 10 = 1 := by
  sorry

end last_digit_of_prime_l22_22922


namespace value_of_x_plus_y_l22_22410

noncomputable def x : ℝ := 1 / 2
noncomputable def y : ℝ := 3

theorem value_of_x_plus_y
  (hx : 1 / x = 2)
  (hy : 1 / x + 3 / y = 3) :
  x + y = 7 / 2 :=
  sorry

end value_of_x_plus_y_l22_22410


namespace arithmetic_sqrt_of_sqrt_16_l22_22674

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22674


namespace complex_powers_sum_zero_l22_22377

theorem complex_powers_sum_zero (i : ℂ) (h : i^2 = -1) : i^2023 + i^2024 + i^2025 + i^2026 = 0 :=
by
  sorry

end complex_powers_sum_zero_l22_22377


namespace parabola_properties_l22_22570

noncomputable def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties
  (a b c t m n x₀ : ℝ)
  (ha : a > 0)
  (h1 : parabola a b c 1 = m)
  (h4 : parabola a b c 4 = n)
  (ht : t = -b / (2 * a))
  (h3ab : 3 * a + b = 0) 
  (hmnc : m < c ∧ c < n)
  (hx₀ym : parabola a b c x₀ = m) :
  m < n ∧ (1 / 2) < t ∧ t < 2 ∧ 0 < x₀ ∧ x₀ < 3 :=
  sorry

end parabola_properties_l22_22570


namespace cyclist_speed_l22_22141

theorem cyclist_speed (v : ℝ) (h : 0.7142857142857143 * (30 + v) = 50) : v = 40 :=
by
  sorry

end cyclist_speed_l22_22141


namespace find_numbers_in_progressions_l22_22140

theorem find_numbers_in_progressions (a b c d : ℝ) :
    (a + b + c = 114) ∧ -- Sum condition
    (b^2 = a * c) ∧ -- Geometric progression condition
    (b = a + 3 * d) ∧ -- Arithmetic progression first condition
    (c = a + 24 * d) -- Arithmetic progression second condition
    ↔ (a = 38 ∧ b = 38 ∧ c = 38) ∨ (a = 2 ∧ b = 14 ∧ c = 98) := by
  sorry

end find_numbers_in_progressions_l22_22140


namespace probability_of_perfect_square_sum_l22_22152

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_l22_22152


namespace smallest_internal_angle_l22_22246

theorem smallest_internal_angle (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : α = 2 * β) (h2 : α = 3 * γ)
  (h3 : α + β + γ = π) :
  α = π / 6 :=
by
  sorry

end smallest_internal_angle_l22_22246


namespace ratio_of_awards_l22_22436

theorem ratio_of_awards 
  (Scott_awards : ℕ) (Scott_awards_eq : Scott_awards = 4)
  (Jessie_awards : ℕ) (Jessie_awards_eq : Jessie_awards = 3 * Scott_awards)
  (rival_awards : ℕ) (rival_awards_eq : rival_awards = 24) :
  rival_awards / Jessie_awards = 2 :=
by sorry

end ratio_of_awards_l22_22436


namespace two_sum_fourth_power_square_l22_22464

-- Define the condition
def sum_zero (x y z : ℤ) : Prop := x + y + z = 0

-- The theorem to be proven
theorem two_sum_fourth_power_square (x y z : ℤ) (h : sum_zero x y z) : ∃ k : ℤ, 2 * (x^4 + y^4 + z^4) = k^2 :=
by
  -- skipping the proof
  sorry

end two_sum_fourth_power_square_l22_22464


namespace students_neither_math_physics_drama_exclusive_l22_22418

def total_students : ℕ := 75
def math_students : ℕ := 42
def physics_students : ℕ := 35
def both_students : ℕ := 25
def drama_exclusive_students : ℕ := 10

theorem students_neither_math_physics_drama_exclusive : 
  total_students - (math_students + physics_students - both_students + drama_exclusive_students) = 13 :=
by
  sorry

end students_neither_math_physics_drama_exclusive_l22_22418


namespace tan_theta_expr_l22_22581

variables {θ x : ℝ}

-- Let θ be an acute angle and let sin(θ/2) = sqrt((x - 2) / (3x)).
theorem tan_theta_expr (h₀ : 0 < θ) (h₁ : θ < (Real.pi / 2)) (h₂ : Real.sin (θ / 2) = Real.sqrt ((x - 2) / (3 * x))) :
  Real.tan θ = (3 * Real.sqrt (7 * x^2 - 8 * x - 16)) / (x + 4) :=
sorry

end tan_theta_expr_l22_22581


namespace sum_fractions_bounds_l22_22544

theorem sum_fractions_bounds {a b c : ℝ} (h : a * b * c = 1) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 < (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) ∧ 
  (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) < 2 :=
  sorry

end sum_fractions_bounds_l22_22544


namespace incorrect_statements_l22_22896

-- Definitions for the points
def A := (-2, -3) 
def P := (1, 1)
def pt := (1, 3)

-- Definitions for the equations in the statements
def equationA (x y : ℝ) := x + y + 5 = 0
def equationB (m x y : ℝ) := 2*(m+1)*x + (m-3)*y + 7 - 5*m = 0
def equationC (θ x y : ℝ) := y - 1 = Real.tan θ * (x - 1)
def equationD (x₁ y₁ x₂ y₂ x y : ℝ) := (x₂ - x₁)*(y - y₁) = (y₂ - y₁)*(x - x₁)

-- Points of interest
def xA : ℝ := -2
def yA : ℝ := -3
def xP : ℝ := 1
def yP : ℝ := 1
def pt_x : ℝ := 1
def pt_y : ℝ := 3

-- Main proof to show which statements are incorrect
theorem incorrect_statements :
  ¬ equationA xA yA ∨ ¬ (∀ m, equationB m pt_x pt_y) ∨ (θ = (Real.pi / 2) → ¬ equationC θ xP yP) ∨
  ∀ x₁ y₁ x₂ y₂ x y, equationD x₁ y₁ x₂ y₂ x y :=
by {
  sorry
}

end incorrect_statements_l22_22896


namespace machine_present_value_l22_22737

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

end machine_present_value_l22_22737


namespace simplify_sqrt7_pow6_l22_22620

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l22_22620


namespace days_spent_on_Orbius5_l22_22121

-- Define the conditions
def days_per_year : Nat := 250
def seasons_per_year : Nat := 5
def length_of_season : Nat := days_per_year / seasons_per_year
def seasons_stayed : Nat := 3

-- Theorem statement
theorem days_spent_on_Orbius5 : (length_of_season * seasons_stayed = 150) :=
by 
  -- Proof is skipped
  sorry

end days_spent_on_Orbius5_l22_22121


namespace exponential_function_f1_l22_22051

theorem exponential_function_f1 (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h3 : a^3 = 8) : a^1 = 2 := by
  sorry

end exponential_function_f1_l22_22051


namespace value_of_a_add_b_l22_22782

theorem value_of_a_add_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
sorry

end value_of_a_add_b_l22_22782


namespace newer_model_distance_l22_22915

theorem newer_model_distance (d_old : ℝ) (p_increase : ℝ) (d_new : ℝ) (h1 : d_old = 300) (h2 : p_increase = 0.30) (h3 : d_new = d_old * (1 + p_increase)) : d_new = 390 :=
by
  sorry

end newer_model_distance_l22_22915


namespace jan_keeps_on_hand_l22_22099

theorem jan_keeps_on_hand (total_length : ℕ) (section_length : ℕ) (friend_fraction : ℚ) (storage_fraction : ℚ) 
  (total_sections : ℕ) (sections_to_friend : ℕ) (remaining_sections : ℕ) (sections_in_storage : ℕ) (sections_on_hand : ℕ) :
  total_length = 1000 → section_length = 25 → friend_fraction = 1 / 4 → storage_fraction = 1 / 2 →
  total_sections = total_length / section_length →
  sections_to_friend = friend_fraction * total_sections →
  remaining_sections = total_sections - sections_to_friend →
  sections_in_storage = storage_fraction * remaining_sections →
  sections_on_hand = remaining_sections - sections_in_storage →
  sections_on_hand = 15 :=
by sorry

end jan_keeps_on_hand_l22_22099


namespace pentagonal_number_formula_l22_22748

def pentagonal_number (n : ℕ) : ℕ :=
  (n * (3 * n + 1)) / 2

theorem pentagonal_number_formula (n : ℕ) :
  pentagonal_number n = (n * (3 * n + 1)) / 2 :=
by
  sorry

end pentagonal_number_formula_l22_22748


namespace problem1_problem2_l22_22933

theorem problem1 (n : ℕ) : 2^n + 3 = k * k → n = 0 :=
by
  intros
  sorry 

theorem problem2 (n : ℕ) : 2^n + 1 = x * x → n = 3 :=
by
  intros
  sorry 

end problem1_problem2_l22_22933


namespace simplify_sqrt_seven_pow_six_proof_l22_22631

noncomputable def simplify_sqrt_seven_pow_six : Prop :=
  (real.sqrt 7)^6 = 343

theorem simplify_sqrt_seven_pow_six_proof : simplify_sqrt_seven_pow_six :=
by
  -- Proof will go here
  sorry

end simplify_sqrt_seven_pow_six_proof_l22_22631


namespace cows_on_farm_l22_22115

theorem cows_on_farm (weekly_production_per_6_cows : ℕ) 
                     (production_over_5_weeks : ℕ) 
                     (number_of_weeks : ℕ) 
                     (cows : ℕ) :
  weekly_production_per_6_cows = 108 →
  production_over_5_weeks = 2160 →
  number_of_weeks = 5 →
  (cows * (weekly_production_per_6_cows / 6) * number_of_weeks = production_over_5_weeks) →
  cows = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end cows_on_farm_l22_22115


namespace ratio_of_wealth_l22_22756

theorem ratio_of_wealth (W P : ℝ) 
  (h1 : 0 < P) (h2 : 0 < W) 
  (pop_X : ℝ := 0.4 * P) 
  (wealth_X : ℝ := 0.6 * W) 
  (top50_pop_X : ℝ := 0.5 * pop_X) 
  (top50_wealth_X : ℝ := 0.8 * wealth_X) 
  (pop_Y : ℝ := 0.2 * P) 
  (wealth_Y : ℝ := 0.3 * W) 
  (avg_wealth_top50_X : ℝ := top50_wealth_X / top50_pop_X) 
  (avg_wealth_Y : ℝ := wealth_Y / pop_Y) : 
  avg_wealth_top50_X / avg_wealth_Y = 1.6 := 
by sorry

end ratio_of_wealth_l22_22756


namespace math_proof_l22_22956

-- Definitions
def U := Set ℝ
def A : Set ℝ := {x | x ≥ 3}
def B : Set ℝ := {x | x^2 - 8*x + 7 ≤ 0}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem
theorem math_proof (a : ℝ) :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x | x ≥ 1}) ∧
  (C a ∪ A = A → a ≥ 4) :=
by
  sorry

end math_proof_l22_22956


namespace num_two_digit_numbers_with_digit_less_than_35_l22_22062

theorem num_two_digit_numbers_with_digit_less_than_35 : 
  let total_two_digit := 90 in
  let unsuitable_numbers := 35 in
  let suitable_numbers := total_two_digit - unsuitable_numbers
  in suitable_numbers = 55 :=
by 
  let total_two_digit := 90
  let unsuitable_numbers := 35
  let suitable_numbers := total_two_digit - unsuitable_numbers
  show suitable_numbers = 55, from sorry

end num_two_digit_numbers_with_digit_less_than_35_l22_22062


namespace distance_between_trees_l22_22489

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (num_spaces : ℕ) (distance : ℕ)
  (h1 : yard_length = 180)
  (h2 : num_trees = 11)
  (h3 : num_spaces = num_trees - 1)
  (h4 : distance = yard_length / num_spaces) :
  distance = 18 :=
by
  sorry

end distance_between_trees_l22_22489


namespace empty_set_a_gt_nine_over_eight_singleton_set_a_values_at_most_one_element_set_a_range_l22_22056

noncomputable def A (a : ℝ) : Set ℝ := { x | a*x^2 - 3*x + 2 = 0 }

theorem empty_set_a_gt_nine_over_eight (a : ℝ) : A a = ∅ ↔ a > 9 / 8 :=
by
  sorry

theorem singleton_set_a_values (a : ℝ) : (∃ x, A a = {x}) ↔ (a = 0 ∨ a = 9 / 8) :=
by
  sorry

theorem at_most_one_element_set_a_range (a : ℝ) : (∀ x y, x ∈ A a → y ∈ A a → x = y) →
  (A a = ∅ ∨ ∃ x, A a = {x}) ↔ (a = 0 ∨ a ≥ 9 / 8) :=
by
  sorry

end empty_set_a_gt_nine_over_eight_singleton_set_a_values_at_most_one_element_set_a_range_l22_22056


namespace total_minutes_to_finish_album_l22_22811

variable (initial_songs : ℕ) (additional_songs : ℕ) (duration : ℕ)

theorem total_minutes_to_finish_album 
  (h1: initial_songs = 25) 
  (h2: additional_songs = 10) 
  (h3: duration = 3) :
  (initial_songs + additional_songs) * duration = 105 :=
sorry

end total_minutes_to_finish_album_l22_22811


namespace burger_share_per_person_l22_22890

-- Definitions based on conditions
def foot_to_inches : ℕ := 12
def burger_length_foot : ℕ := 1
def burger_length_inches : ℕ := burger_length_foot * foot_to_inches

theorem burger_share_per_person : (burger_length_inches / 2) = 6 := by
  sorry

end burger_share_per_person_l22_22890


namespace complement_of_A_in_U_eq_l22_22058

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x ≥ Real.exp 1}
def complement_U_A : Set ℝ := {x | 0 < x ∧ x ≤ Real.exp 1}

theorem complement_of_A_in_U_eq : 
  (U \ A) = complement_U_A := 
by
  sorry

end complement_of_A_in_U_eq_l22_22058


namespace intersection_A_CRB_l22_22253

-- Definition of sets A and C_{R}B
def is_in_A (x: ℝ) := 0 < x ∧ x < 2

def is_in_CRB (x: ℝ) := x ≤ 1 ∨ x ≥ Real.exp 2

-- Proof that the intersection of A and C_{R}B is (0, 1]
theorem intersection_A_CRB : {x : ℝ | is_in_A x} ∩ {x : ℝ | is_in_CRB x} = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_A_CRB_l22_22253


namespace units_digit_calculation_l22_22876

theorem units_digit_calculation : 
  ((33 * (83 ^ 1001) * (7 ^ 1002) * (13 ^ 1003)) % 10) = 9 :=
by
  sorry

end units_digit_calculation_l22_22876


namespace arithmetic_sqrt_of_sqrt_16_l22_22673

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22673


namespace bars_per_set_correct_l22_22036

-- Define the total number of metal bars and the number of sets
def total_metal_bars : ℕ := 14
def number_of_sets : ℕ := 2

-- Define the function to compute bars per set
def bars_per_set (total_bars : ℕ) (sets : ℕ) : ℕ :=
  total_bars / sets

-- The proof statement
theorem bars_per_set_correct : bars_per_set total_metal_bars number_of_sets = 7 := by
  sorry

end bars_per_set_correct_l22_22036


namespace total_percentage_increase_l22_22829

noncomputable def initialSalary : ℝ := 60
noncomputable def firstRaisePercent : ℝ := 10
noncomputable def secondRaisePercent : ℝ := 15
noncomputable def promotionRaisePercent : ℝ := 20

theorem total_percentage_increase :
  let finalSalary := initialSalary * (1 + firstRaisePercent / 100) * (1 + secondRaisePercent / 100) * (1 + promotionRaisePercent / 100)
  let increase := finalSalary - initialSalary
  let percentageIncrease := (increase / initialSalary) * 100
  percentageIncrease = 51.8 := by
  sorry

end total_percentage_increase_l22_22829


namespace remainder_proof_l22_22738

theorem remainder_proof (n : ℤ) (h : n % 6 = 1) : (3 * (n + 1812)) % 6 = 3 := 
by 
  sorry

end remainder_proof_l22_22738


namespace distance_they_both_run_l22_22729

theorem distance_they_both_run
  (D : ℝ)
  (A_time : D / 28 = A_speed)
  (B_time : D / 32 = B_speed)
  (A_beats_B : A_speed * 28 = B_speed * 28 + 16) :
  D = 128 := 
sorry

end distance_they_both_run_l22_22729


namespace age_ratio_7_9_l22_22127

/-- Definition of Sachin and Rahul's ages -/
def sachin_age : ℝ := 24.5
def rahul_age : ℝ := sachin_age + 7

/-- The ratio of Sachin's age to Rahul's age is 7:9 -/
theorem age_ratio_7_9 : sachin_age / rahul_age = 7 / 9 := by
  sorry

end age_ratio_7_9_l22_22127


namespace contrapositive_example_l22_22863

theorem contrapositive_example (x : ℝ) : (x = 1 → x^2 - 3 * x + 2 = 0) ↔ (x^2 - 3 * x + 2 ≠ 0 → x ≠ 1) :=
by
  sorry

end contrapositive_example_l22_22863


namespace least_five_digit_congruent_to_5_mod_15_l22_22478

theorem least_five_digit_congruent_to_5_mod_15 : ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 15 = 5 ∧ n = 10010 := by
  sorry

end least_five_digit_congruent_to_5_mod_15_l22_22478


namespace rectangle_ratio_l22_22458

theorem rectangle_ratio (A L : ℝ) (hA : A = 100) (hL : L = 20) :
  ∃ W : ℝ, A = L * W ∧ (L / W) = 4 :=
by
  sorry

end rectangle_ratio_l22_22458


namespace factorial_mod_10_eq_6_l22_22216

theorem factorial_mod_10_eq_6 : (10! % 13) = 6 := by
  sorry

end factorial_mod_10_eq_6_l22_22216


namespace nora_nuts_problem_l22_22528

theorem nora_nuts_problem :
  ∃ n : ℕ, (∀ (a p c : ℕ), 30 * n = 18 * a ∧ 30 * n = 21 * p ∧ 30 * n = 16 * c) ∧ n = 34 :=
by
  -- Provided conditions and solution steps will go here.
  sorry

end nora_nuts_problem_l22_22528


namespace smallest_non_consecutive_product_not_factor_of_48_l22_22889

def is_factor (a b : ℕ) : Prop := b % a = 0

def non_consecutive_pairs (x y : ℕ) : Prop := (x ≠ y) ∧ (x + 1 ≠ y) ∧ (y + 1 ≠ x)

theorem smallest_non_consecutive_product_not_factor_of_48 :
  ∃ x y, x ∣ 48 ∧ y ∣ 48 ∧ non_consecutive_pairs x y ∧ ¬ (x * y ∣ 48) ∧ (∀ x' y', x' ∣ 48 ∧ y' ∣ 48 ∧ non_consecutive_pairs x' y' ∧ ¬ (x' * y' ∣ 48) → x' * y' ≥ 18) :=
by
  sorry

end smallest_non_consecutive_product_not_factor_of_48_l22_22889


namespace problem_statement_l22_22211

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.range (n+1))

theorem problem_statement : ∃ r : ℕ, r < 13 ∧ (factorial 10) % 13 = r :=
by
  sorry

end problem_statement_l22_22211


namespace num_two_digit_numbers_with_digit_less_than_35_l22_22063

theorem num_two_digit_numbers_with_digit_less_than_35 : 
  let total_two_digit := 90 in
  let unsuitable_numbers := 35 in
  let suitable_numbers := total_two_digit - unsuitable_numbers
  in suitable_numbers = 55 :=
by 
  let total_two_digit := 90
  let unsuitable_numbers := 35
  let suitable_numbers := total_two_digit - unsuitable_numbers
  show suitable_numbers = 55, from sorry

end num_two_digit_numbers_with_digit_less_than_35_l22_22063


namespace find_a2_b2_l22_22545

theorem find_a2_b2 (a b : ℝ) (h1 : a - b = 6) (h2 : a * b = 32) : a^2 + b^2 = 100 :=
by
  sorry

end find_a2_b2_l22_22545


namespace pythagorean_triple_9_12_15_l22_22503

theorem pythagorean_triple_9_12_15 : ∃ a b c : ℕ, a = 9 ∧ b = 12 ∧ c = 15 ∧ (a * a + b * b = c * c) :=
by 
  existsi (9, 12, 15)
  split
  rfl
  split
  rfl
  split
  rfl
  sorry

end pythagorean_triple_9_12_15_l22_22503


namespace largest_fraction_l22_22722

noncomputable def compare_fractions : List ℚ :=
  [5 / 11, 7 / 16, 9 / 20, 11 / 23, 111 / 245, 145 / 320, 185 / 409, 211 / 465, 233 / 514]

theorem largest_fraction :
  max (5 / 11) (max (7 / 16) (max (9 / 20) (max (11 / 23) (max (111 / 245) (max (145 / 320) (max (185 / 409) (max (211 / 465) (233 / 514)))))))) = 11 / 23 := 
  sorry

end largest_fraction_l22_22722


namespace x2004_y2004_l22_22082

theorem x2004_y2004 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) : 
  x^2004 + y^2004 = 2^2004 := 
by
  sorry

end x2004_y2004_l22_22082


namespace find_sequence_term_l22_22546

noncomputable def sequence_sum (n : ℕ) : ℚ :=
  (2 / 3) * n^2 - (1 / 3) * n

def sequence_term (n : ℕ) : ℚ :=
  if n = 1 then (1 / 3) else (4 / 3) * n - 1

theorem find_sequence_term (n : ℕ) : sequence_term n = (sequence_sum n - sequence_sum (n - 1)) :=
by
  unfold sequence_sum
  unfold sequence_term
  sorry

end find_sequence_term_l22_22546


namespace norma_cards_lost_l22_22302

def initial_cards : ℕ := 88
def final_cards : ℕ := 18
def cards_lost : ℕ := initial_cards - final_cards

theorem norma_cards_lost : cards_lost = 70 :=
by
  sorry

end norma_cards_lost_l22_22302


namespace joey_speed_return_l22_22100

/--
Joey the postman takes 1 hour to run a 5-mile-long route every day, delivering packages along the way.
On his return, he must climb a steep hill covering 3 miles and then navigate a rough, muddy terrain spanning 2 miles.
If the average speed of the entire round trip is 8 miles per hour, prove that the speed with which Joey returns along the path is 20 miles per hour.
-/
theorem joey_speed_return
  (dist_out : ℝ := 5)
  (time_out : ℝ := 1)
  (dist_hill : ℝ := 3)
  (dist_terrain : ℝ := 2)
  (avg_speed_round : ℝ := 8)
  (total_dist : ℝ := dist_out * 2)
  (total_time : ℝ := total_dist / avg_speed_round)
  (time_return : ℝ := total_time - time_out)
  (dist_return : ℝ := dist_hill + dist_terrain) :
  (dist_return / time_return = 20) := 
sorry

end joey_speed_return_l22_22100


namespace common_solutions_form_segment_length_one_l22_22028

theorem common_solutions_form_segment_length_one (a : ℝ) (h₁ : ∀ x : ℝ, x^2 - 4 * x + 2 - a ≤ 0) 
  (h₂ : ∀ x : ℝ, x^2 - 5 * x + 2 * a + 8 ≤ 0) : 
  (a = -1 ∨ a = -7 / 4) :=
by
  sorry

end common_solutions_form_segment_length_one_l22_22028


namespace sum_fractions_l22_22932

theorem sum_fractions :
  (1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)) = (2 / 9) :=
by
  sorry

end sum_fractions_l22_22932


namespace johns_cloth_cost_per_metre_l22_22828

noncomputable def calculate_cost_per_metre (total_cost : ℝ) (total_metres : ℝ) : ℝ :=
  total_cost / total_metres

def johns_cloth_purchasing_data : Prop :=
  calculate_cost_per_metre 444 9.25 = 48

theorem johns_cloth_cost_per_metre : johns_cloth_purchasing_data :=
  sorry

end johns_cloth_cost_per_metre_l22_22828


namespace kitten_current_length_l22_22512

theorem kitten_current_length (initial_length : ℕ) (double_after_2_weeks : ℕ → ℕ) (double_after_4_months : ℕ → ℕ)
  (h1 : initial_length = 4)
  (h2 : double_after_2_weeks initial_length = 2 * initial_length)
  (h3 : double_after_4_months (double_after_2_weeks initial_length) = 2 * (double_after_2_weeks initial_length)) :
  double_after_4_months (double_after_2_weeks initial_length) = 16 := 
by
  sorry

end kitten_current_length_l22_22512


namespace factorization_implies_k_l22_22542

theorem factorization_implies_k (x y k : ℝ) (h : ∃ (a b c d e f : ℝ), 
                            x^3 + 3 * x^2 - 2 * x * y - k * x - 4 * y = (a * x + b * y + c) * (d * x^2 + e * xy + f)) :
  k = -2 :=
sorry

end factorization_implies_k_l22_22542


namespace janets_shampoo_days_l22_22284

-- Definitions from the problem conditions
def rose_shampoo := 1 / 3
def jasmine_shampoo := 1 / 4
def daily_usage := 1 / 12

-- Define the total shampoo and the days lasts
def total_shampoo := rose_shampoo + jasmine_shampoo
def days_lasts := total_shampoo / daily_usage

-- The theorem to be proved
theorem janets_shampoo_days : days_lasts = 7 :=
by sorry

end janets_shampoo_days_l22_22284


namespace poly_remainder_l22_22535

theorem poly_remainder (x : ℤ) :
  (x^1012) % (x^3 - x^2 + x - 1) = 1 := by
  sorry

end poly_remainder_l22_22535


namespace sean_has_45_whistles_l22_22309

variable (Sean Charles : ℕ)

def sean_whistles (Charles : ℕ) : ℕ :=
  Charles + 32

theorem sean_has_45_whistles
    (Charles_whistles : Charles = 13) 
    (Sean_whistles_condition : Sean = sean_whistles Charles) :
    Sean = 45 := by
  sorry

end sean_has_45_whistles_l22_22309


namespace scientific_notation_of_3933_billion_l22_22271

-- Definitions and conditions
def is_scientific_notation (a : ℝ) (n : ℤ) :=
  1 ≤ |a| ∧ |a| < 10 ∧ (39.33 * 10^9 = a * 10^n)

-- Theorem (statement only)
theorem scientific_notation_of_3933_billion : 
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n ∧ a = 3.933 ∧ n = 10 :=
by
  sorry

end scientific_notation_of_3933_billion_l22_22271


namespace value_of_a_plus_b_l22_22779

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) :
  a + b = -1 ∨ a + b = -3 :=
sorry

end value_of_a_plus_b_l22_22779


namespace two_digit_numbers_less_than_35_l22_22079

theorem two_digit_numbers_less_than_35 : 
  ∃ n, n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
  (let tens_digit := (n + 10) / 10 in 
   let units_digit := (n + 10) % 10 in 
   tens_digit < 3 ∨ units_digit < 5) → 
  nat.card {n | n ∈ finset.range 90 ∧ (10 ≤ n + 10 ∧ n + 10 ≤ 99) ∧ 
                    (let tens_digit := (n + 10) / 10 in 
                     let units_digit := (n + 10) % 10 in 
                     tens_digit < 3 ∨ units_digit < 5)} = 55
:= sorry

end two_digit_numbers_less_than_35_l22_22079


namespace num_positive_integer_N_l22_22032

def num_valid_N : Nat := 7

theorem num_positive_integer_N (N : Nat) (h_pos : N > 0) :
  (∃ k : Nat, k > 3 ∧ N = k - 3 ∧ 48 % k = 0) ↔ (N < 45) ∧ (num_valid_N = 7) := 
by
sorry

end num_positive_integer_N_l22_22032


namespace initial_team_sizes_l22_22306

/-- 
On the first day of the sports competition, 1/6 of the boys' team and 1/7 of the girls' team 
did not meet the qualifying standards and were eliminated. During the rest of the competition, 
the same number of athletes from both teams were eliminated for not meeting the standards. 
By the end of the competition, a total of 48 boys and 50 girls did not meet the qualifying standards. 
Moreover, the number of girls who met the qualifying standards was twice the number of boys who did.
We are to prove the initial number of boys and girls in their respective teams.
-/

theorem initial_team_sizes (initial_boys initial_girls : ℕ) :
  (∃ (x : ℕ), 
    initial_boys = x + 48 ∧ 
    initial_girls = 2 * x + 50 ∧ 
    48 - (1 / 6 : ℚ) * (x + 48 : ℚ) = 50 - (1 / 7 : ℚ) * (2 * x + 50 : ℚ) ∧
    initial_girls - 2 * initial_boys = 98 - 2 * 72
  ) ↔ 
  initial_boys = 72 ∧ initial_girls = 98 := 
sorry

end initial_team_sizes_l22_22306


namespace valid_outfit_selections_l22_22257

-- Definitions based on the given conditions
def num_shirts : ℕ := 6
def num_pants : ℕ := 5
def num_hats : ℕ := 6
def num_colors : ℕ := 6

-- The total number of outfits without restrictions
def total_outfits : ℕ := num_shirts * num_pants * num_hats

-- The theorem statement to prove the final answer
theorem valid_outfit_selections : total_outfits = 150 :=
by
  have h1 : total_outfits = 6 * 5 * 6 := rfl
  have h2 : 6 * 5 * 6 = 180 := by norm_num
  have h3 : 180 = 150 := sorry -- Here you need to differentiate the invalid outfits using provided restrictions
  exact h3

end valid_outfit_selections_l22_22257


namespace sum_of_interior_angles_l22_22453

theorem sum_of_interior_angles (n : ℕ) 
  (h : 180 * (n - 2) = 3600) :
  180 * (n + 2 - 2) = 3960 ∧ 180 * (n - 2 - 2) = 3240 :=
by
  sorry

end sum_of_interior_angles_l22_22453


namespace circle_radius_l22_22693

theorem circle_radius (r : ℝ) (hr : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 :=
by 
  sorry

end circle_radius_l22_22693


namespace sequence_bounds_l22_22740

theorem sequence_bounds (n : ℕ) (hpos : 0 < n) :
  ∃ (a : ℕ → ℝ), (a 0 = 1/2) ∧
  (∀ k < n, a (k + 1) = a k + (1/n) * (a k)^2) ∧
  (1 - 1 / n < a n ∧ a n < 1) :=
sorry

end sequence_bounds_l22_22740


namespace simplify_expression_l22_22619

theorem simplify_expression (r : ℝ) : (2 * r^2 + 5 * r - 3) + (3 * r^2 - 4 * r + 2) = 5 * r^2 + r - 1 := 
by
  sorry

end simplify_expression_l22_22619


namespace num_valid_triples_l22_22412

theorem num_valid_triples : 
  (∃ (a1 a2 a3: ℕ),
    1 ≤ a1 ∧ a1 < a2 ∧ a2 < a3 ∧ a3 ≤ 14 ∧ 
    a2 - a1 ≥ 3 ∧ a3 - a2 ≥ 3) → 
  (fintype.card { t : ℕ × ℕ × ℕ // 
    let (a1, a2, a3) := t in 
    1 ≤ a1 ∧ a1 < a2 ∧ a2 < a3 ∧ a3 ≤ 14 ∧ 
    a2 - a1 ≥ 3 ∧ a3 - a2 ≥ 3 } = 120) := 
by 
  sorry

end num_valid_triples_l22_22412


namespace probability_is_one_fourteenth_l22_22887

-- Define the set of numbers
def num_set := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the function to determine smallest difference >= 3
def valid_triplet (a b c : ℕ) : Prop :=
  a ∈ num_set ∧ b ∈ num_set ∧ c ∈ num_set ∧
  a < b ∧ b < c ∧ (b - a) ≥ 3 ∧ (c - b) ≥ 3

-- Count the number of valid triplets
noncomputable def count_valid_triplets : ℕ :=
  (num_set.to_list.comb 3).countp (λ t, match t with
                                        | [a, b, c] => valid_triplet a b c
                                        | _         => false
                                        end)

-- Total combinations of three numbers
def total_combinations : ℕ := num_set.card.choose 3

-- Define the probability
noncomputable def probability : ℚ :=
  count_valid_triplets / total_combinations

-- Theorem statement
theorem probability_is_one_fourteenth :
  probability = 1 / 14 := by
    sorry

end probability_is_one_fourteenth_l22_22887


namespace at_least_one_not_less_than_2_l22_22264

-- Definitions for the problem
variables {a b c : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The Lean 4 statement for the problem
theorem at_least_one_not_less_than_2 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (2 ≤ a + 1/b) ∨ (2 ≤ b + 1/c) ∨ (2 ≤ c + 1/a) :=
sorry

end at_least_one_not_less_than_2_l22_22264


namespace find_number_l22_22864

theorem find_number : ∃ x : ℝ, 3 * x - 1 = 2 * x ∧ x = 1 := sorry

end find_number_l22_22864


namespace find_ax6_by6_l22_22961

variable {a b x y : ℝ}

theorem find_ax6_by6
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 12)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 80) :
  a * x^6 + b * y^6 = 1531.25 :=
sorry

end find_ax6_by6_l22_22961


namespace polynomial_remainder_x1012_l22_22537

theorem polynomial_remainder_x1012 (x : ℂ) : 
  (x^1012) % (x^3 - x^2 + x - 1) = 1 :=
sorry

end polynomial_remainder_x1012_l22_22537


namespace cos_identity_l22_22558

theorem cos_identity (α : ℝ) (h : Real.cos (Real.pi / 8 - α) = 1 / 6) :
  Real.cos (3 * Real.pi / 4 + 2 * α) = 17 / 18 :=
by
  sorry

end cos_identity_l22_22558


namespace quadratic_axis_of_symmetry_l22_22970

theorem quadratic_axis_of_symmetry (b c : ℝ) (h : -b / 2 = 3) : b = 6 :=
by
  sorry

end quadratic_axis_of_symmetry_l22_22970


namespace probability_of_perfect_square_sum_l22_22151

def is_perfect_square (n : ℕ) : Prop :=
  n = 1*1 ∨ n = 2*2 ∨ n = 3*3 ∨ n = 4*4

theorem probability_of_perfect_square_sum :
  let total_outcomes := 64
  let perfect_square_sums := 12
  (perfect_square_sums / total_outcomes : ℚ) = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_l22_22151


namespace arithmetic_sqrt_sqrt_16_l22_22664

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l22_22664


namespace exists_infinitely_many_gcd_condition_l22_22289

theorem exists_infinitely_many_gcd_condition (a : ℕ → ℕ) (h : ∀ n : ℕ, ∃ m : ℕ, a m = n) :
  ∃ᶠ i in at_top, Nat.gcd (a i) (a (i + 1)) ≤ (3 * i) / 4 :=
sorry

end exists_infinitely_many_gcd_condition_l22_22289


namespace sale_decrease_by_20_percent_l22_22305

theorem sale_decrease_by_20_percent (P Q : ℝ)
  (h1 : P > 0) (h2 : Q > 0)
  (price_increased : ∀ P', P' = 1.30 * P)
  (revenue_increase : ∀ R, R = P * Q → ∀ R', R' = 1.04 * R)
  (new_revenue : ∀ P' Q' R', P' = 1.30 * P → Q' = Q * (1 - x / 100) → R' = P' * Q' → R' = 1.04 * (P * Q)) :
  1 - (20 / 100) = 0.8 :=
by sorry

end sale_decrease_by_20_percent_l22_22305


namespace days_spent_on_Orbius5_l22_22122

-- Define the conditions
def days_per_year : Nat := 250
def seasons_per_year : Nat := 5
def length_of_season : Nat := days_per_year / seasons_per_year
def seasons_stayed : Nat := 3

-- Theorem statement
theorem days_spent_on_Orbius5 : (length_of_season * seasons_stayed = 150) :=
by 
  -- Proof is skipped
  sorry

end days_spent_on_Orbius5_l22_22122


namespace kaeli_problems_per_day_l22_22996

-- Definitions based on conditions
def problems_solved_per_day_marie_pascale : ℕ := 4
def total_problems_marie_pascale : ℕ := 72
def total_problems_kaeli : ℕ := 126

-- Number of days both took should be the same
def number_of_days : ℕ := total_problems_marie_pascale / problems_solved_per_day_marie_pascale

-- Kaeli solves 54 more problems than Marie-Pascale
def extra_problems_kaeli : ℕ := 54

-- Definition that Kaeli's total problems solved is that of Marie-Pascale plus 54
axiom kaeli_total_problems (h : total_problems_marie_pascale + extra_problems_kaeli = total_problems_kaeli) : True

-- Now to find x, the problems solved per day by Kaeli
def x : ℕ := total_problems_kaeli / number_of_days

-- Prove that x = 7
theorem kaeli_problems_per_day (h : total_problems_marie_pascale + extra_problems_kaeli = total_problems_kaeli) : x = 7 := by
  sorry

end kaeli_problems_per_day_l22_22996


namespace initial_average_production_l22_22387

theorem initial_average_production (n : ℕ) (today_production : ℕ) 
  (new_average : ℕ) (initial_average : ℕ) :
  n = 1 → today_production = 60 → new_average = 55 → initial_average = (new_average * (n + 1) - today_production) → initial_average = 50 :=
by
  intros h1 h2 h3 h4
  -- Insert further proof here
  sorry

end initial_average_production_l22_22387


namespace integer_solutions_for_xyz_eq_4_l22_22686

theorem integer_solutions_for_xyz_eq_4 :
  {n : ℕ // n = 48} :=
sorry

end integer_solutions_for_xyz_eq_4_l22_22686


namespace ratio_s_to_t_l22_22337

theorem ratio_s_to_t (b : ℝ) (s t : ℝ)
  (h1 : s = -b / 10)
  (h2 : t = -b / 6) :
  s / t = 3 / 5 :=
by sorry

end ratio_s_to_t_l22_22337


namespace range_of_a_l22_22402

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * x + 1 + a * Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) ↔ (0 < a ∧ a < 1/2) := by
  sorry

end range_of_a_l22_22402


namespace total_transportation_cost_l22_22736

def weights_in_grams : List ℕ := [300, 450, 600]
def cost_per_kg : ℕ := 15000

def convert_to_kg (w : ℕ) : ℚ :=
  w / 1000

def calculate_cost (weight_in_kg : ℚ) (cost_per_kg : ℕ) : ℚ :=
  weight_in_kg * cost_per_kg

def total_cost (weights_in_grams : List ℕ) (cost_per_kg : ℕ) : ℚ :=
  weights_in_grams.map (λ w => calculate_cost (convert_to_kg w) cost_per_kg) |>.sum

theorem total_transportation_cost :
  total_cost weights_in_grams cost_per_kg = 20250 := by
  sorry

end total_transportation_cost_l22_22736


namespace correct_statements_l22_22034

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - 13 / 4 * Real.pi)

theorem correct_statements :
    (f (Real.pi / 8) = 0) ∧ 
    (∀ x, 2 * Real.sin (2 * (x - 5 / 8 * Real.pi)) = f x) :=
by
  sorry

end correct_statements_l22_22034


namespace positive_integer_divisibility_by_3_l22_22202

theorem positive_integer_divisibility_by_3 (n : ℕ) (h : 0 < n) :
  (n * 2^n + 1) % 3 = 0 ↔ n % 6 = 1 ∨ n % 6 = 2 := 
sorry

end positive_integer_divisibility_by_3_l22_22202


namespace arithmetic_sqrt_of_sqrt_16_l22_22661

noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := real.sqrt x

theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt (arithmetic_sqrt 16) = 2 := by
  sorry

end arithmetic_sqrt_of_sqrt_16_l22_22661


namespace find_n_for_conditions_l22_22533

theorem find_n_for_conditions :
  ∀ (n : ℕ), n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 5 →
  ∃ (k : ℕ), k ≥ 2 ∧ ∃ (a : Fin k → ℚ), 
  (∀ i, 0 < a i) ∧
  (∑ i, a i = n) ∧
  (∏ i, a i = n) :=
by
  intros n hn
  sorry

end find_n_for_conditions_l22_22533


namespace f_value_5pi_over_3_l22_22757

noncomputable def f : ℝ → ℝ :=
sorry -- We need to define the function according to the given conditions but we skip this for now.

lemma f_property_1 (x : ℝ) : f (-x) = -f x :=
sorry -- as per the condition f(-x) = -f(x)

lemma f_property_2 (x : ℝ) : f (x + π/2) = f (x - π/2) :=
sorry -- as per the condition f(x + π/2) = f(x - π/2)

lemma f_property_3 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π/2) : f x = Real.sin x :=
sorry -- as per the condition f(x) = sin(x) for x in [0, π/2]

theorem f_value_5pi_over_3 : f (5 * π / 3) = - (Real.sin (π / 3)) :=
by
  -- The exact proof steps are omitted; this involves properties of periodicity and the odd function nature
  sorry

end f_value_5pi_over_3_l22_22757


namespace parity_expression_l22_22993

theorem parity_expression
  (a b c : ℕ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_odd : a % 2 = 1)
  (h_b_odd : b % 2 = 1) :
  (5^a + (b + 1)^2 * c) % 2 = 1 :=
by
  sorry

end parity_expression_l22_22993


namespace power_function_passes_point_l22_22803

noncomputable def f (k α x : ℝ) : ℝ := k * x^α

theorem power_function_passes_point (k α : ℝ) (h1 : f k α (1/2) = (Real.sqrt 2)/2) : 
  k + α = 3/2 :=
sorry

end power_function_passes_point_l22_22803


namespace average_weight_l22_22882

theorem average_weight (men women : ℕ) (avg_weight_men avg_weight_women : ℝ) (total_people : ℕ) (combined_avg_weight : ℝ) 
  (h1 : men = 8) (h2 : avg_weight_men = 190) (h3 : women = 6) (h4 : avg_weight_women = 120) (h5 : total_people = 14) 
  (h6 : (men * avg_weight_men + women * avg_weight_women) / total_people = combined_avg_weight) : combined_avg_weight = 160 := 
  sorry

end average_weight_l22_22882


namespace average_weight_correct_l22_22884

-- Define the number of men and women
def number_of_men : ℕ := 8
def number_of_women : ℕ := 6

-- Define the average weights of men and women
def average_weight_men : ℕ := 190
def average_weight_women : ℕ := 120

-- Define the total weight of men and women
def total_weight_men : ℕ := number_of_men * average_weight_men
def total_weight_women : ℕ := number_of_women * average_weight_women

-- Define the total number of individuals
def total_individuals : ℕ := number_of_men + number_of_women

-- Define the combined total weight
def total_weight : ℕ := total_weight_men + total_weight_women

-- Define the average weight of all individuals
def average_weight_all : ℕ := total_weight / total_individuals

theorem average_weight_correct :
  average_weight_all = 160 :=
  by sorry

end average_weight_correct_l22_22884


namespace sawyer_total_octopus_legs_l22_22855

-- Formalization of the problem conditions
def num_octopuses : Nat := 5
def legs_per_octopus : Nat := 8

-- Formalization of the question and answer
def total_legs : Nat := num_octopuses * legs_per_octopus

-- The proof statement
theorem sawyer_total_octopus_legs : total_legs = 40 :=
by
  sorry

end sawyer_total_octopus_legs_l22_22855


namespace hyperbola_asymptote_slopes_l22_22195

theorem hyperbola_asymptote_slopes :
  ∀ (x y : ℝ), 2 * (y^2 / 16) - 2 * (x^2 / 9) = 1 → (∃ m : ℝ, y = m * x ∨ y = -m * x) ∧ m = (Real.sqrt 80) / 3 :=
by
  sorry

end hyperbola_asymptote_slopes_l22_22195


namespace measurement_error_probability_l22_22998

noncomputable def normal_distribution_cdf (z : ℝ) : ℝ :=
  sorry -- Assume there is an existing function for the CDF of normal distribution

theorem measurement_error_probability :
  let σ := 10
  let δ := 15
  let phi := normal_distribution_cdf
  2 * phi (δ / σ) = 0.8664 :=
by
  intros
  -- The proof is omitted as per instruction
  sorry

end measurement_error_probability_l22_22998


namespace digit_in_ten_thousandths_place_of_fraction_is_two_l22_22892

theorem digit_in_ten_thousandths_place_of_fraction_is_two :
  (∃ d : ℕ, d = (Int.floor (5 / 32 * 10^4) % 10) ∧ d = 2) :=
by
  sorry

end digit_in_ten_thousandths_place_of_fraction_is_two_l22_22892


namespace find_pairs_l22_22769

-- Define predicative statements for the conditions
def is_integer (x : ℝ) : Prop :=
  ∃ (n : ℤ), x = n

def condition1 (m n : ℕ) : Prop := 
  (n^2 + 1) % (2 * m) = 0

def condition2 (m n : ℕ) : Prop := 
  is_integer (Real.sqrt (2^(n-1) + m + 4))

-- The goal is to find the pairs of positive integers
theorem find_pairs (m n : ℕ) (h1: condition1 m n) (h2: condition2 m n) : 
  (m = 61 ∧ n = 11) :=
sorry

end find_pairs_l22_22769


namespace perp_lines_a_value_l22_22255

theorem perp_lines_a_value :
  ∀ a : ℝ, ((a + 1) * 1 - 2 * (-a) = 0) → a = 1 :=
by
  intro a
  intro h
  -- We now state that a must satisfy the given condition and show that this leads to a = 1
  -- The proof is left as sorry
  sorry

end perp_lines_a_value_l22_22255


namespace function_property_l22_22244

def y (x : ℝ) : ℝ := x - 2

theorem function_property : y 1 = -1 :=
by
  -- place for proof
  sorry

end function_property_l22_22244


namespace average_weight_correct_l22_22885

-- Define the number of men and women
def number_of_men : ℕ := 8
def number_of_women : ℕ := 6

-- Define the average weights of men and women
def average_weight_men : ℕ := 190
def average_weight_women : ℕ := 120

-- Define the total weight of men and women
def total_weight_men : ℕ := number_of_men * average_weight_men
def total_weight_women : ℕ := number_of_women * average_weight_women

-- Define the total number of individuals
def total_individuals : ℕ := number_of_men + number_of_women

-- Define the combined total weight
def total_weight : ℕ := total_weight_men + total_weight_women

-- Define the average weight of all individuals
def average_weight_all : ℕ := total_weight / total_individuals

theorem average_weight_correct :
  average_weight_all = 160 :=
  by sorry

end average_weight_correct_l22_22885


namespace find_line_equation_l22_22049

def ellipse (x y : ℝ) : Prop := (x ^ 2) / 6 + (y ^ 2) / 3 = 1

def meets_first_quadrant (l : Line) : Prop :=
  ∃ A B : Point, ellipse A.x A.y ∧ ellipse B.x B.y ∧ 
  A.x > 0 ∧ A.y > 0 ∧ B.x > 0 ∧ B.y > 0 ∧ l.contains A ∧ l.contains B

def intersects_axes (l : Line) : Prop :=
  ∃ M N : Point, M.y = 0 ∧ N.x = 0 ∧ l.contains M ∧ l.contains N
  
def equal_distances (M N A B : Point) : Prop :=
  dist M A = dist N B

def distance_MN (M N : Point) : Prop :=
  dist M N = 2 * Real.sqrt 3

theorem find_line_equation (l : Line) (A B M N : Point)
  (h1 : meets_first_quadrant l)
  (h2 : intersects_axes l)
  (h3 : equal_distances M N A B)
  (h4 : distance_MN M N) :
  l.equation = "x + sqrt(2) * y - 2 * sqrt(2) = 0" :=
sorry

end find_line_equation_l22_22049


namespace function_equality_l22_22579

theorem function_equality (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, f n < f (n + 1) )
  (h2 : f 2 = 2)
  (h3 : ∀ m n : ℕ, f (m * n) = f m * f n) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_equality_l22_22579


namespace solve_abs_inequality_l22_22943

theorem solve_abs_inequality (x : ℝ) : abs ((7 - 2 * x) / 4) < 3 ↔ -2.5 < x ∧ x < 9.5 := by
  sorry

end solve_abs_inequality_l22_22943


namespace janet_better_condition_count_l22_22575

noncomputable def janet_initial := 10
noncomputable def janet_sells := 6
noncomputable def janet_remaining := janet_initial - janet_sells
noncomputable def brother_gives := 2 * janet_remaining
noncomputable def janet_after_brother := janet_remaining + brother_gives
noncomputable def janet_total := 24

theorem janet_better_condition_count : 
  janet_total - janet_after_brother = 12 := by
  sorry

end janet_better_condition_count_l22_22575


namespace find_y_value_l22_22087

-- Define the given conditions and the final question in Lean
theorem find_y_value (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = k * x ^ (1/3)) 
  (h2 : y = 4 * real.sqrt 3)
  (x1 : x = 64) 
  : ∃ k, y = 2 * real.sqrt 3 :=
sorry

end find_y_value_l22_22087


namespace shampoo_duration_l22_22278

theorem shampoo_duration
  (rose_shampoo : ℚ := 1/3)
  (jasmine_shampoo : ℚ := 1/4)
  (daily_usage : ℚ := 1/12) :
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := 
by
  sorry

end shampoo_duration_l22_22278


namespace cubic_roots_expression_l22_22583

theorem cubic_roots_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = -1) (h3 : a * b * c = 2) :
  2 * a * (b - c) ^ 2 + 2 * b * (c - a) ^ 2 + 2 * c * (a - b) ^ 2 = -36 :=
by
  sorry

end cubic_roots_expression_l22_22583


namespace two_digit_numbers_count_l22_22070

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  { n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ (tens_digit n < 3 ∨ units_digit n < 5) }.to_finset.card = 55 :=
by
  sorry

end two_digit_numbers_count_l22_22070


namespace area_triangle_ABC_l22_22796

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_triangle_ABC :
  area_of_triangle (2, 4) (-1, 1) (1, -1) = 6 :=
by
  sorry

end area_triangle_ABC_l22_22796


namespace number_2digit_smaller_than_35_l22_22072

/--
Prove that the number of two-digit numbers where at least one digit
is smaller than the corresponding digit in 35 is exactly 55.
-/
theorem number_2digit_smaller_than_35 : 
  (Finset.filter (λ n : ℕ, let d1 := n / 10, d2 := n % 10 in d1 < 3 ∨ d2 < 5) (Finset.range' 10 100)).card = 55 := 
by
  sorry

end number_2digit_smaller_than_35_l22_22072


namespace arctan_sum_l22_22013

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end arctan_sum_l22_22013


namespace unique_triangle_determination_l22_22168

-- Definitions for each type of triangle and their respective conditions
def isosceles_triangle (base_angle : ℝ) (altitude : ℝ) : Type := sorry
def vertex_base_isosceles_triangle (vertex_angle : ℝ) (base : ℝ) : Type := sorry
def circ_radius_side_equilateral_triangle (radius : ℝ) (side : ℝ) : Type := sorry
def leg_radius_right_triangle (leg : ℝ) (radius : ℝ) : Type := sorry
def angles_side_scalene_triangle (angle1 : ℝ) (angle2 : ℝ) (opp_side : ℝ) : Type := sorry

-- Condition: Option A does not uniquely determine a triangle
def option_A_does_not_uniquely_determine : Prop :=
  ∀ (base_angle altitude : ℝ), 
    (∃ t1 t2 : isosceles_triangle base_angle altitude, t1 ≠ t2)

-- Condition: Options B through E uniquely determine the triangle
def options_B_to_E_uniquely_determine : Prop :=
  (∀ (vertex_angle base : ℝ), ∃! t : vertex_base_isosceles_triangle vertex_angle base, true) ∧
  (∀ (radius side : ℝ), ∃! t : circ_radius_side_equilateral_triangle radius side, true) ∧
  (∀ (leg radius : ℝ), ∃! t : leg_radius_right_triangle leg radius, true) ∧
  (∀ (angle1 angle2 opp_side : ℝ), ∃! t : angles_side_scalene_triangle angle1 angle2 opp_side, true)

-- Main theorem combining both conditions
theorem unique_triangle_determination :
  option_A_does_not_uniquely_determine ∧ options_B_to_E_uniquely_determine :=
  sorry

end unique_triangle_determination_l22_22168


namespace mean_of_six_numbers_l22_22329

theorem mean_of_six_numbers (sum_of_six : ℚ) (h : sum_of_six = 3 / 4) : (sum_of_six / 6) = 1 / 8 :=
by
  sorry

end mean_of_six_numbers_l22_22329


namespace irreducible_fraction_l22_22593

theorem irreducible_fraction (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry

end irreducible_fraction_l22_22593


namespace sean_has_45_whistles_l22_22310

variable (Sean Charles : ℕ)

def sean_whistles (Charles : ℕ) : ℕ :=
  Charles + 32

theorem sean_has_45_whistles
    (Charles_whistles : Charles = 13) 
    (Sean_whistles_condition : Sean = sean_whistles Charles) :
    Sean = 45 := by
  sorry

end sean_has_45_whistles_l22_22310


namespace conditionD_necessary_not_sufficient_l22_22235

variable (a b : ℝ)

-- Define each of the conditions as separate variables
def conditionA : Prop := |a| < |b|
def conditionB : Prop := 2 * a < 2 * b
def conditionC : Prop := a < b - 1
def conditionD : Prop := a < b + 1

-- Prove that condition D is necessary but not sufficient for a < b
theorem conditionD_necessary_not_sufficient : conditionD a b → (¬ conditionA a b ∨ ¬ conditionB a b ∨ ¬ conditionC a b) ∧ ¬(conditionD a b ↔ a < b) :=
by sorry

end conditionD_necessary_not_sufficient_l22_22235


namespace factorial_mod_10_l22_22219

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the problem statement
theorem factorial_mod_10 : factorial 10 % 13 = 7 :=
by sorry

end factorial_mod_10_l22_22219


namespace intersection_A_B_l22_22254

def A : Set ℤ := {-2, -1, 0, 1, 2, 3}
def B : Set ℤ := {x | x^2 - 2 * x - 3 < 0}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l22_22254


namespace smallest_value_of_2a_plus_1_l22_22258

theorem smallest_value_of_2a_plus_1 (a : ℝ) 
  (h : 6 * a^2 + 5 * a + 4 = 3) : 
  ∃ b : ℝ, b = 2 * a + 1 ∧ b = 0 := 
sorry

end smallest_value_of_2a_plus_1_l22_22258


namespace solve_for_x_l22_22860

theorem solve_for_x (x : ℚ) (h : 2 / 3 + 1 / x = 7 / 9) : x = 9 :=
sorry

end solve_for_x_l22_22860


namespace unit_vector_norm_diff_l22_22470

noncomputable def sqrt42_sqrt3_div_2 : ℝ := (Real.sqrt 42 * Real.sqrt 3) / 2
noncomputable def sqrt17_div_sqrt2 : ℝ := (Real.sqrt 17) / Real.sqrt 2

theorem unit_vector_norm_diff {x1 y1 z1 x2 y2 z2 : ℝ}
  (h1 : x1^2 + y1^2 + z1^2 = 1)
  (h2 : 3*x1 + y1 + 2*z1 = sqrt42_sqrt3_div_2)
  (h3 : 2*x1 + 2*y1 + 3*z1 = sqrt17_div_sqrt2)
  (h4 : x2^2 + y2^2 + z2^2 = 1)
  (h5 : 3*x2 + y2 + 2*z2 = sqrt42_sqrt3_div_2)
  (h6 : 2*x2 + 2*y2 + 3*z2 = sqrt17_div_sqrt2)
  (h_distinct : (x1, y1, z1) ≠ (x2, y2, z2)) :
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2) = Real.sqrt 2 :=
by
  sorry

end unit_vector_norm_diff_l22_22470


namespace selling_price_is_correct_l22_22109

-- Define the constants used in the problem
noncomputable def cost_price : ℝ := 540
noncomputable def markup_percentage : ℝ := 0.15
noncomputable def discount_percentage : ℝ := 26.570048309178745 / 100

-- Define the conditions in the problem
noncomputable def marked_price : ℝ := cost_price * (1 + markup_percentage)
noncomputable def discount_amount : ℝ := marked_price * discount_percentage
noncomputable def selling_price : ℝ := marked_price - discount_amount

-- Theorem stating the problem
theorem selling_price_is_correct : selling_price = 456 := by 
  sorry

end selling_price_is_correct_l22_22109


namespace find_value_of_b_l22_22799

theorem find_value_of_b (a b : ℕ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 :=
sorry

end find_value_of_b_l22_22799


namespace find_line_equation_l22_22047

theorem find_line_equation (x y : ℝ) : 
  (∃ A B, (A.x^2 / 6 + A.y^2 / 3 = 1) ∧ (B.x^2 / 6 + B.y^2 / 3 = 1) ∧
  (A.x > 0 ∧ A.y > 0) ∧ (B.x > 0 ∧ B.y > 0) ∧
  let M := (-B.y, 0) in
  let N := (0, B.y) in 
  (abs (M.x - A.x) = abs (N.y - B.y)) ∧ 
  (abs (M.x - N.x + M.y - N.y) = 2 * sqrt 3)) →
  x + sqrt 2 * y - 2 * sqrt 2 = 0 := 
sorry

end find_line_equation_l22_22047


namespace gcd_m_n_l22_22290

def m : ℕ := 333333
def n : ℕ := 888888888

theorem gcd_m_n : Nat.gcd m n = 3 := by
  sorry

end gcd_m_n_l22_22290


namespace find_x_condition_l22_22813

theorem find_x_condition (x : ℝ) (h : 0.75 / x = 5 / 11) : x = 1.65 := 
by
  sorry

end find_x_condition_l22_22813


namespace total_heads_l22_22912

variables (H C : ℕ)

theorem total_heads (h_hens: H = 22) (h_feet: 2 * H + 4 * C = 140) : H + C = 46 :=
by
  sorry

end total_heads_l22_22912


namespace functional_equation_solution_l22_22372

def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (2 * x + f y) = x + y + f x) →
  (∀ x : ℝ, f x = x) :=
by
  intros h x
  sorry

end functional_equation_solution_l22_22372


namespace trapezoid_area_l22_22825

variable (A B C D K : Type)
variable [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty K]

-- Define the lengths as given in the conditions
def AK : ℝ := 16
def DK : ℝ := 4
def CD : ℝ := 6

-- Define the property that the trapezoid ABCD has an inscribed circle
axiom trapezoid_with_inscribed_circle (ABCD : Prop) : Prop

-- The Lean theorem statement
theorem trapezoid_area (ABCD : Prop) (AK DK CD : ℝ) 
  (H1 : trapezoid_with_inscribed_circle ABCD)
  (H2 : AK = 16)
  (H3 : DK = 4)
  (H4 : CD = 6) : 
  ∃ (area : ℝ), area = 432 :=
by
  sorry

end trapezoid_area_l22_22825


namespace pythagorean_triple_9_12_15_l22_22504

theorem pythagorean_triple_9_12_15 : 9^2 + 12^2 = 15^2 :=
by 
  sorry

end pythagorean_triple_9_12_15_l22_22504


namespace seq_positive_integers_no_m_exists_l22_22392

-- Definition of the sequence
def seq (n : ℕ) : ℕ :=
  Nat.recOn n
    1
    (λ n a_n => 3 * a_n + 2 * (2 * a_n * a_n - 1).sqrt)

-- Axiomatize the properties involved in the recurrence relation
axiom rec_sqrt_property (n : ℕ) : ∃ k : ℕ, (2 * seq n * seq n - 1) = k * k

-- Proof statement for the sequence of positive integers
theorem seq_positive_integers (n : ℕ) : seq n > 0 := sorry

-- Proof statement for non-existence of m such that 2015 divides seq(m)
theorem no_m_exists (m : ℕ) : ¬ (2015 ∣ seq m) := sorry

end seq_positive_integers_no_m_exists_l22_22392


namespace determine_b_l22_22197

theorem determine_b (b : ℤ) : (x - 5) ∣ (x^3 + 3 * x^2 + b * x + 5) → b = -41 :=
by
  sorry

end determine_b_l22_22197


namespace real_solution_unique_l22_22934

variable (x : ℝ)

theorem real_solution_unique :
  (x ≠ 2 ∧ (x^3 - 3 * x^2) / (x^2 - 4 * x + 4) + x = 3) ↔ x = 1 := 
by 
  sorry

end real_solution_unique_l22_22934


namespace ten_factorial_mod_thirteen_l22_22226

open Nat

theorem ten_factorial_mod_thirteen :
  (10! % 13) = 6 := by
  sorry

end ten_factorial_mod_thirteen_l22_22226


namespace a_minus_b_value_l22_22460

theorem a_minus_b_value (a b c : ℝ) (x : ℝ) 
    (h1 : (2 * x - 3) ^ 2 = a * x ^ 2 + b * x + c)
    (h2 : x = 0 → c = 9)
    (h3 : x = 1 → a + b + c = 1)
    (h4 : x = -1 → (2 * (-1) - 3) ^ 2 = a * (-1) ^ 2 + b * (-1) + c) : 
    a - b = 16 :=
by  
  sorry

end a_minus_b_value_l22_22460


namespace power_subtraction_divisibility_l22_22307

theorem power_subtraction_divisibility (N : ℕ) (h : N > 1) : 
  ∃ k : ℕ, (N^2)^2014 - (N^11)^106 = k * (N^6 + N^3 + 1) :=
by
  sorry

end power_subtraction_divisibility_l22_22307


namespace carpooling_plans_l22_22986

def last_digits (jia : ℕ) (friend1 : ℕ) (friend2 : ℕ) (friend3 : ℕ) (friend4 : ℕ) : Prop :=
  jia = 0 ∧ friend1 = 0 ∧ friend2 = 2 ∧ friend3 = 1 ∧ friend4 = 5

def total_car_plans : Prop :=
  ∀ (jia friend1 friend2 friend3 friend4 : ℕ),
    last_digits jia friend1 friend2 friend3 friend4 →
    (∃ num_ways : ℕ, num_ways = 64)

theorem carpooling_plans : total_car_plans :=
sorry

end carpooling_plans_l22_22986


namespace parallel_planes_mn_l22_22791

theorem parallel_planes_mn (m n : ℝ) (a b : ℝ × ℝ × ℝ) (α β : Type) (h1 : a = (0, 1, m)) (h2 : b = (0, n, -3)) 
  (h3 : ∃ k : ℝ, a = (k • b)) : m * n = -3 :=
by
  -- Proof would be here
  sorry

end parallel_planes_mn_l22_22791


namespace arithmetic_square_root_of_sqrt_16_l22_22669

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l22_22669


namespace kitten_length_doubling_l22_22511

theorem kitten_length_doubling (initial_length : ℕ) (week2_length : ℕ) (current_length : ℕ) 
  (h1 : initial_length = 4) 
  (h2 : week2_length = 2 * initial_length) 
  (h3 : current_length = 2 * week2_length) : 
    current_length = 16 := 
by 
  sorry

end kitten_length_doubling_l22_22511


namespace find_x0_l22_22547

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 2 then x^2 - 4 else 2 * x

theorem find_x0 (x0 : ℝ) (h : f x0 = 8) : x0 = 4 := by
  sorry

end find_x0_l22_22547


namespace symmetric_points_subtraction_l22_22793

theorem symmetric_points_subtraction (a b : ℝ) (h1 : -2 = -a) (h2 : b = -3) : a - b = 5 :=
by {
  sorry
}

end symmetric_points_subtraction_l22_22793


namespace min_value_expr_l22_22950

theorem min_value_expr (a : ℝ) (h₁ : 0 < a) (h₂ : a < 3) : 
  ∃ m : ℝ, (∀ x : ℝ, 0 < x → x < 3 → (1/x + 9/(3 - x)) ≥ m) ∧ m = 16 / 3 :=
sorry

end min_value_expr_l22_22950


namespace anna_more_candy_than_billy_l22_22515

theorem anna_more_candy_than_billy :
  let anna_candy_per_house := 14
  let billy_candy_per_house := 11
  let anna_houses := 60
  let billy_houses := 75
  let anna_total_candy := anna_candy_per_house * anna_houses
  let billy_total_candy := billy_candy_per_house * billy_houses
  anna_total_candy - billy_total_candy = 15 :=
by
  sorry

end anna_more_candy_than_billy_l22_22515


namespace sum_of_squares_of_roots_eq_226_l22_22023

theorem sum_of_squares_of_roots_eq_226 (s_1 s_2 : ℝ) (h_eq : ∀ x, x^2 - 16 * x + 15 = 0 → (x = s_1 ∨ x = s_2)) :
  s_1^2 + s_2^2 = 226 := by
  sorry

end sum_of_squares_of_roots_eq_226_l22_22023


namespace probability_of_perfect_square_sum_on_two_8_sided_dice_l22_22154

-- Define the dice and possible sums
def is_sum_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def pairs_rolled : finset (ℕ × ℕ) :=
  finset.univ.filter (λ p, p.1 + p.2 ∈ {4, 9, 16})

theorem probability_of_perfect_square_sum_on_two_8_sided_dice :
  (pairs_rolled.card : ℚ) / 64 = 3 / 16 :=
by
  sorry

end probability_of_perfect_square_sum_on_two_8_sided_dice_l22_22154


namespace combined_distance_correct_l22_22186

-- Define the conditions
def wheelA_rotations_per_minute := 20
def wheelA_distance_per_rotation_cm := 35
def wheelB_rotations_per_minute := 30
def wheelB_distance_per_rotation_cm := 50

-- Calculate distances in meters
def wheelA_distance_per_minute_m :=
  (wheelA_rotations_per_minute * wheelA_distance_per_rotation_cm) / 100

def wheelB_distance_per_minute_m :=
  (wheelB_rotations_per_minute * wheelB_distance_per_rotation_cm) / 100

def wheelA_distance_per_hour_m :=
  wheelA_distance_per_minute_m * 60

def wheelB_distance_per_hour_m :=
  wheelB_distance_per_minute_m * 60

def combined_distance_per_hour_m :=
  wheelA_distance_per_hour_m + wheelB_distance_per_hour_m

theorem combined_distance_correct : combined_distance_per_hour_m = 1320 := by
  -- skip the proof here with sorry
  sorry

end combined_distance_correct_l22_22186


namespace polynomial_simplification_l22_22483

variable (x : ℝ)

theorem polynomial_simplification :
  (3*x^3 + 4*x^2 + 12)*(x + 1) - (x + 1)*(2*x^3 + 6*x^2 - 42) + (6*x^2 - 28)*(x + 1)*(x - 2) = 
  7*x^4 - 7*x^3 - 42*x^2 + 82*x + 110 :=
by
  sorry

end polynomial_simplification_l22_22483


namespace arithmetic_sqrt_sqrt_16_l22_22667

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l22_22667


namespace apollonius_circle_equation_l22_22868

theorem apollonius_circle_equation (x y : ℝ) (A B : ℝ × ℝ) (hA : A = (2, 0)) (hB : B = (8, 0))
  (h : dist (x, y) A / dist (x, y) B = 1 / 2) : x^2 + y^2 = 16 := 
sorry

end apollonius_circle_equation_l22_22868


namespace percentage_of_a_added_to_get_x_l22_22137

variable (a b x m : ℝ) (P : ℝ) (k : ℝ)
variable (h1 : a / b = 4 / 5)
variable (h2 : x = a * (1 + P / 100))
variable (h3 : m = b * 0.2)
variable (h4 : m / x = 0.14285714285714285)

theorem percentage_of_a_added_to_get_x :
  P = 75 :=
by
  sorry

end percentage_of_a_added_to_get_x_l22_22137


namespace arctan_sum_l22_22003

theorem arctan_sum : ∀ (a b : ℝ), 
  a = 3/7 → 
  b = 7/3 → 
  a * b = 1 → 
  a > 0 → 
  b > 0 → 
  Real.arctan a + Real.arctan b = Real.pi / 2 :=
by intros a b ha hb hab ha_pos hb_pos
   rw [ha, hb, hab]
   sorry

end arctan_sum_l22_22003


namespace line_y_axis_intersect_l22_22494

theorem line_y_axis_intersect (x1 y1 x2 y2 : ℝ) (h1 : x1 = 3 ∧ y1 = 27) (h2 : x2 = -7 ∧ y2 = -1) :
  ∃ y : ℝ, (∀ x : ℝ, y = (y2 - y1) / (x2 - x1) * (x - x1) + y1) ∧ y = 18.6 :=
by
  sorry

end line_y_axis_intersect_l22_22494


namespace sum_of_coefficients_l22_22866

theorem sum_of_coefficients :
  ∃ a b c d e : ℤ, 
    27 * (x : ℝ)^3 + 64 = (a * x + b) * (c * x^2 + d * x + e) ∧ 
    a + b + c + d + e = 20 :=
by
  sorry

end sum_of_coefficients_l22_22866


namespace max_distinct_sums_l22_22139

/-- Given 3 boys and 20 girls standing in a row, each child counts the number of girls to their 
left and the number of boys to their right and adds these two counts together. Prove that 
the maximum number of different sums that the children could have obtained is 20. -/
theorem max_distinct_sums (boys girls : ℕ) (total_children : ℕ) 
  (h_boys : boys = 3) (h_girls : girls = 20) (h_total : total_children = boys + girls) : 
  ∃ (max_sums : ℕ), max_sums = 20 := 
by 
  sorry

end max_distinct_sums_l22_22139


namespace min_value_inequality_l22_22582

theorem min_value_inequality (a b c d e f : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_pos_e : 0 < e) (h_pos_f : 0 < f)
    (h_sum : a + b + c + d + e + f = 9) : 
    1 / a + 9 / b + 16 / c + 25 / d + 36 / e + 49 / f ≥ 676 / 9 := 
by 
  sorry

end min_value_inequality_l22_22582


namespace total_time_spent_l22_22826

variable (B I E M EE ST ME : ℝ)

def learn_basic_rules : ℝ := B
def learn_intermediate_level : ℝ := I
def learn_expert_level : ℝ := E
def learn_master_level : ℝ := M
def endgame_exercises : ℝ := EE
def middle_game_strategy_tactics : ℝ := ST
def mentoring : ℝ := ME

theorem total_time_spent :
  B = 2 →
  I = 75 * B →
  E = 50 * (B + I) →
  M = 30 * E →
  EE = 0.25 * I →
  ST = 2 * EE →
  ME = 0.5 * E →
  B + I + E + M + EE + ST + ME = 235664.5 :=
by
  intros hB hI hE hM hEE hST hME
  rw [hB, hI, hE, hM, hEE, hST, hME]
  sorry

end total_time_spent_l22_22826


namespace average_weight_men_women_l22_22879

theorem average_weight_men_women (n_men n_women : ℕ) (avg_weight_men avg_weight_women : ℚ)
  (h_men : n_men = 8) (h_women : n_women = 6) (h_avg_weight_men : avg_weight_men = 190)
  (h_avg_weight_women : avg_weight_women = 120) :
  (n_men * avg_weight_men + n_women * avg_weight_women) / (n_men + n_women) = 160 := 
by
  sorry

end average_weight_men_women_l22_22879


namespace orchard_produce_l22_22187

theorem orchard_produce (num_apple_trees num_orange_trees apple_baskets_per_tree apples_per_basket orange_baskets_per_tree oranges_per_basket : ℕ) 
  (h1 : num_apple_trees = 50) 
  (h2 : num_orange_trees = 30) 
  (h3 : apple_baskets_per_tree = 25) 
  (h4 : apples_per_basket = 18)
  (h5 : orange_baskets_per_tree = 15) 
  (h6 : oranges_per_basket = 12) 
: (num_apple_trees * (apple_baskets_per_tree * apples_per_basket) = 22500) ∧ 
  (num_orange_trees * (orange_baskets_per_tree * oranges_per_basket) = 5400) :=
  by 
  sorry

end orchard_produce_l22_22187


namespace find_number_of_girls_l22_22976

-- Define the ratio of boys to girls as 8:4.
def ratio_boys_to_girls : ℕ × ℕ := (8, 4)

-- Define the total number of students.
def total_students : ℕ := 600

-- Define what it means for the number of girls given a ratio and total students.
def number_of_girls (ratio : ℕ × ℕ) (total : ℕ) : ℕ :=
  let total_parts := (ratio.1 + ratio.2)
  let part_value := total / total_parts
  ratio.2 * part_value

-- State the goal to prove the number of girls is 200 given the conditions.
theorem find_number_of_girls :
  number_of_girls ratio_boys_to_girls total_students = 200 :=
sorry

end find_number_of_girls_l22_22976


namespace num_intersection_points_l22_22754

-- Define the equations of the lines as conditions
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 8

-- The theorem to prove the number of intersection points
theorem num_intersection_points :
  ∃! p : ℝ × ℝ, (line1 p.1 p.2 ∧ line2 p.1 p.2) ∨ (line2 p.1 p.2 ∧ line3 p.1 p.2) :=
sorry

end num_intersection_points_l22_22754


namespace circle_radius_l22_22697

theorem circle_radius (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end circle_radius_l22_22697


namespace intersection_of_sets_l22_22804

def setA : Set ℝ := {x | x^2 ≤ 4 * x}
def setB : Set ℝ := {x | x < 1}

theorem intersection_of_sets : setA ∩ setB = {x | x < 1} := by
  sorry

end intersection_of_sets_l22_22804


namespace problem_statement_l22_22208

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.range (n+1))

theorem problem_statement : ∃ r : ℕ, r < 13 ∧ (factorial 10) % 13 = r :=
by
  sorry

end problem_statement_l22_22208


namespace astronaut_days_on_orbius_l22_22120

noncomputable def days_in_year : ℕ := 250
noncomputable def seasons_in_year : ℕ := 5
noncomputable def seasons_stayed : ℕ := 3

theorem astronaut_days_on_orbius :
  (days_in_year / seasons_in_year) * seasons_stayed = 150 := by
  sorry

end astronaut_days_on_orbius_l22_22120


namespace complex_number_z_satisfies_l22_22790

theorem complex_number_z_satisfies (z : ℂ) : 
  (z * (1 + I) + (-I) * (1 - I) = 0) → z = -1 := 
by {
  sorry
}

end complex_number_z_satisfies_l22_22790


namespace problem1_problem2_l22_22173

namespace MathProofs

theorem problem1 : (-3 - (-8) + (-6) + 10) = 9 :=
by
  sorry

theorem problem2 : (-12 * ((1 : ℚ) / 6 - (1 : ℚ) / 3 - 3 / 4)) = 11 :=
by
  sorry

end MathProofs

end problem1_problem2_l22_22173


namespace ratio_wealth_citizen_XY_l22_22755

noncomputable def wealth_ratio_XY 
  (P W : ℝ) 
  (h1 : 0 < P) 
  (h2 : 0 < W) : ℝ :=
  let pop_X := 0.4 * P
  let wealth_X_before_tax := 0.5 * W
  let tax_X := 0.1 * wealth_X_before_tax
  let wealth_X_after_tax := wealth_X_before_tax - tax_X
  let wealth_per_citizen_X := wealth_X_after_tax / pop_X

  let pop_Y := 0.3 * P
  let wealth_Y := 0.6 * W
  let wealth_per_citizen_Y := wealth_Y / pop_Y

  wealth_per_citizen_X / wealth_per_citizen_Y

theorem ratio_wealth_citizen_XY 
  (P W : ℝ) 
  (h1 : 0 < P) 
  (h2 : 0 < W) : 
  wealth_ratio_XY P W h1 h2 = 9 / 16 := 
by
  sorry

end ratio_wealth_citizen_XY_l22_22755


namespace find_value_l22_22539

theorem find_value (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 :=
sorry

end find_value_l22_22539


namespace total_songs_performed_l22_22538

theorem total_songs_performed :
  ∃ N : ℕ, 
  (∃ e d o : ℕ, 
     (e > 3 ∧ e < 9) ∧ (d > 3 ∧ d < 9) ∧ (o > 3 ∧ o < 9)
      ∧ N = (9 + 3 + e + d + o) / 4) ∧ N = 6 :=
sorry

end total_songs_performed_l22_22538


namespace simplify_sqrt_seven_pow_six_l22_22634

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l22_22634


namespace not_cheap_is_necessary_condition_l22_22313

-- Define propositions for "good quality" and "not cheap"
variables {P: Prop} {Q: Prop} 

-- Statement "You get what you pay for" implies "good quality is not cheap"
axiom H : P → Q 

-- The proof problem
theorem not_cheap_is_necessary_condition (H : P → Q) : Q → P :=
by sorry

end not_cheap_is_necessary_condition_l22_22313


namespace circle_radius_l22_22700

theorem circle_radius (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end circle_radius_l22_22700


namespace fraction_value_l22_22580

theorem fraction_value (x y : ℝ) (h1 : 2 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 5) (h3 : (∃ m : ℤ, x = m * y)) : x / y = -2 :=
sorry

end fraction_value_l22_22580


namespace value_of_a_add_b_l22_22784

theorem value_of_a_add_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
sorry

end value_of_a_add_b_l22_22784


namespace algebraic_expression_l22_22389

variable (m n x y : ℤ)

theorem algebraic_expression (h1 : x = m) (h2 : y = n) (h3 : x - y = 2) : n - m = -2 := 
by
  sorry

end algebraic_expression_l22_22389


namespace impossible_coins_l22_22610

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l22_22610


namespace upper_bound_expression_4n_plus_7_l22_22234

theorem upper_bound_expression_4n_plus_7 (U : ℤ) :
  (∃ (n : ℕ),  4 * n + 7 > 1) ∧
  (∀ (n : ℕ), 4 * n + 7 < U → ∃ (k : ℕ), k ≤ 19 ∧ k = n) ∧
  (∃ (n_min n_max : ℕ), n_max = n_min + 19 ∧ 4 * n_max + 7 < U) →
  U = 84 := sorry

end upper_bound_expression_4n_plus_7_l22_22234


namespace james_jump_height_is_16_l22_22444

-- Define given conditions
def mark_jump_height : ℕ := 6
def lisa_jump_height : ℕ := 2 * mark_jump_height
def jacob_jump_height : ℕ := 2 * lisa_jump_height
def james_jump_height : ℕ := (2 * jacob_jump_height) / 3

-- Problem Statement to prove
theorem james_jump_height_is_16 : james_jump_height = 16 :=
by
  sorry

end james_jump_height_is_16_l22_22444


namespace factorial_mod_10_l22_22220

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the problem statement
theorem factorial_mod_10 : factorial 10 % 13 = 7 :=
by sorry

end factorial_mod_10_l22_22220


namespace parabola_translation_l22_22336

theorem parabola_translation :
  (∀ x, y = x^2) →
  (∀ x, y = (x + 1)^2 - 2) :=
by
  sorry

end parabola_translation_l22_22336


namespace vertex_of_parabola_find_shift_m_l22_22403

-- Problem 1: Vertex of the given parabola
theorem vertex_of_parabola : 
  ∃ x y: ℝ, (y = 2 * x^2 + 4 * x - 6) ∧ (x, y) = (-1, -8) := 
by
  -- Proof goes here
  sorry

-- Problem 2: Finding the shift m
theorem find_shift_m (m : ℝ) (h : m > 0) : 
  (∀ x (hx : (x = (x + m)) ∧ (2 * x^2 + 4 * x - 6 = 0)), x = 1 ∨ x = -3) ∧ 
  ((-3 + m) = 0) → m = 3 :=
by
  -- Proof goes here
  sorry

end vertex_of_parabola_find_shift_m_l22_22403


namespace proportion_of_salt_correct_l22_22527

def grams_of_salt := 50
def grams_of_water := 1000
def total_solution := grams_of_salt + grams_of_water
def proportion_of_salt : ℚ := grams_of_salt / total_solution

theorem proportion_of_salt_correct :
  proportion_of_salt = 1 / 21 := 
  by {
    sorry
  }

end proportion_of_salt_correct_l22_22527


namespace metallic_sheet_length_l22_22913

theorem metallic_sheet_length (w : ℝ) (s : ℝ) (v : ℝ) (L : ℝ) 
  (h_w : w = 38) 
  (h_s : s = 8) 
  (h_v : v = 5632) 
  (h_volume : (L - 2 * s) * (w - 2 * s) * s = v) : 
  L = 48 :=
by
  -- To complete the proof, follow the mathematical steps:
  -- (L - 2 * s) * (w - 2 * s) * s = v
  -- (L - 2 * 8) * (38 - 2 * 8) * 8 = 5632
  -- Simplify and solve for L
  sorry

end metallic_sheet_length_l22_22913


namespace value_of_other_number_l22_22084

theorem value_of_other_number (k : ℕ) (other_number : ℕ) (h1 : k = 2) (h2 : (5 + k) * (5 - k) = 5^2 - other_number) : other_number = 21 :=
  sorry

end value_of_other_number_l22_22084


namespace radius_of_circle_l22_22701

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end radius_of_circle_l22_22701


namespace intersection_sets_l22_22041

-- defining sets A and B
def A : Set ℤ := {-1, 2, 4}
def B : Set ℤ := {0, 2, 6}

-- the theorem to be proved
theorem intersection_sets:
  A ∩ B = {2} :=
sorry

end intersection_sets_l22_22041


namespace crow_eats_quarter_in_twenty_hours_l22_22349

-- Given: The crow eats 1/5 of the nuts in 4 hours
def crow_eating_rate (N : ℕ) : ℕ := N / 5 / 4

-- Prove: It will take 20 hours to eat 1/4 of the nuts
theorem crow_eats_quarter_in_twenty_hours (N : ℕ) (h : ℕ) (h_eq : h = 20) : 
  ((N / 5) / 4 : ℝ) = ((N / 4) / h : ℝ) :=
by
  sorry

end crow_eats_quarter_in_twenty_hours_l22_22349


namespace three_x_plus_four_l22_22965

theorem three_x_plus_four (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  sorry

end three_x_plus_four_l22_22965


namespace andrew_kept_correct_l22_22743

open Nat

def andrew_bought : ℕ := 750
def daniel_received : ℕ := 250
def fred_received : ℕ := daniel_received + 120
def total_shared : ℕ := daniel_received + fred_received
def andrew_kept : ℕ := andrew_bought - total_shared

theorem andrew_kept_correct : andrew_kept = 130 :=
by
  unfold andrew_kept andrew_bought total_shared fred_received daniel_received
  rfl

end andrew_kept_correct_l22_22743


namespace smallest_b_base_l22_22164

theorem smallest_b_base :
  ∃ b : ℕ, b^2 ≤ 25 ∧ 25 < b^3 ∧ (∀ c : ℕ, c < b → ¬(c^2 ≤ 25 ∧ 25 < c^3)) :=
sorry

end smallest_b_base_l22_22164


namespace kitten_length_after_4_months_l22_22506

theorem kitten_length_after_4_months
  (initial_length : ℕ)
  (doubled_length_2_weeks : ℕ)
  (final_length_4_months : ℕ)
  (h1 : initial_length = 4)
  (h2 : doubled_length_2_weeks = initial_length * 2)
  (h3 : final_length_4_months = doubled_length_2_weeks * 2) :
  final_length_4_months = 16 := 
by
  sorry

end kitten_length_after_4_months_l22_22506


namespace sector_perimeter_l22_22044

theorem sector_perimeter (R : ℝ) (α : ℝ) (A : ℝ) (P : ℝ) : 
  A = (1 / 2) * R^2 * α → 
  α = 4 → 
  A = 2 → 
  P = 2 * R + R * α → 
  P = 6 := 
by
  intros hArea hAlpha hA hP
  sorry

end sector_perimeter_l22_22044


namespace log_product_eq_one_l22_22905

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_product_eq_one :
  log_base 5 2 * log_base 4 25 = 1 := 
by
  sorry

end log_product_eq_one_l22_22905


namespace smaller_circle_radius_l22_22431

theorem smaller_circle_radius :
  ∀ (R r : ℝ), R = 10 ∧ (4 * r = 2 * R) → r = 5 :=
by
  intro R r
  intro h
  have h1 : R = 10 := h.1
  have h2 : 4 * r = 2 * R := h.2
  -- Use the conditions to eventually show r = 5
  sorry

end smaller_circle_radius_l22_22431


namespace sum_of_squares_largest_multiple_of_7_l22_22339

theorem sum_of_squares_largest_multiple_of_7
  (N : ℕ) (a : ℕ) (h1 : N = a^2 + (a + 1)^2 + (a + 2)^2)
  (h2 : N < 10000)
  (h3 : 7 ∣ N) :
  N = 8750 := sorry

end sum_of_squares_largest_multiple_of_7_l22_22339


namespace consecutive_sum_l22_22874

theorem consecutive_sum (m k : ℕ) (h : (k + 1) * (2 * m + k) = 2000) :
  (m = 1000 ∧ k = 0) ∨ 
  (m = 198 ∧ k = 4) ∨ 
  (m = 28 ∧ k = 24) ∨ 
  (m = 55 ∧ k = 15) :=
by sorry

end consecutive_sum_l22_22874


namespace find_a_l22_22818

theorem find_a (x y a : ℤ) (h1 : a * x + y = 40) (h2 : 2 * x - y = 20) (h3 : 3 * y^2 = 48) : a = 3 :=
sorry

end find_a_l22_22818


namespace two_digit_numbers_count_l22_22068

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  { n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ (tens_digit n < 3 ∨ units_digit n < 5) }.to_finset.card = 55 :=
by
  sorry

end two_digit_numbers_count_l22_22068


namespace mean_of_six_numbers_l22_22328

theorem mean_of_six_numbers (sum_of_six : ℚ) (h : sum_of_six = 3 / 4) : (sum_of_six / 6) = 1 / 8 :=
by
  sorry

end mean_of_six_numbers_l22_22328


namespace calculate_3_diamond_4_l22_22928

-- Define the operations
def op (a b : ℝ) : ℝ := a^2 + 2 * a * b
def diamond (a b : ℝ) : ℝ := 4 * a + 6 * b - op a b

-- State the theorem
theorem calculate_3_diamond_4 : diamond 3 4 = 3 := by
  sorry

end calculate_3_diamond_4_l22_22928


namespace average_weight_correct_l22_22886

-- Define the number of men and women
def number_of_men : ℕ := 8
def number_of_women : ℕ := 6

-- Define the average weights of men and women
def average_weight_men : ℕ := 190
def average_weight_women : ℕ := 120

-- Define the total weight of men and women
def total_weight_men : ℕ := number_of_men * average_weight_men
def total_weight_women : ℕ := number_of_women * average_weight_women

-- Define the total number of individuals
def total_individuals : ℕ := number_of_men + number_of_women

-- Define the combined total weight
def total_weight : ℕ := total_weight_men + total_weight_women

-- Define the average weight of all individuals
def average_weight_all : ℕ := total_weight / total_individuals

theorem average_weight_correct :
  average_weight_all = 160 :=
  by sorry

end average_weight_correct_l22_22886


namespace angle_sum_420_l22_22262

theorem angle_sum_420 (A B C D E F : ℝ) (hE : E = 30) : 
  A + B + C + D + E + F = 420 :=
by
  sorry

end angle_sum_420_l22_22262


namespace range_arcsin_x_squared_minus_x_l22_22872

noncomputable def range_of_arcsin : Set ℝ :=
  {x | -Real.arcsin (1 / 4) ≤ x ∧ x ≤ Real.pi / 2}

theorem range_arcsin_x_squared_minus_x :
  ∀ x : ℝ, ∃ y ∈ range_of_arcsin, y = Real.arcsin (x^2 - x) :=
by
  sorry

end range_arcsin_x_squared_minus_x_l22_22872


namespace binomial_coefficient_x_term_l22_22981

/-- 
In the binomial expansion of (x - 2 / sqrt x)^7, the coefficient of the x term is 560.
-/
theorem binomial_coefficient_x_term :
  let expr := (x - 2 / real.sqrt x) in
  let n := 7 in
  let k := 4 in
  let general_term := (λ k: ℕ, (-2)^k * (nat.choose n k) * x^((14 - 3 * k) / 2)) in
  general_term k = 560 := 
by
  sorry

end binomial_coefficient_x_term_l22_22981


namespace maximum_k_for_ray_below_f_l22_22401

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 3 * x - 2

theorem maximum_k_for_ray_below_f :
  let g (x : ℝ) : ℝ := (x * Real.log x + 3 * x - 2) / (x - 1)
  ∃ k : ℤ, ∀ x > 1, g x > k ∧ k = 5 :=
by sorry

end maximum_k_for_ray_below_f_l22_22401


namespace arithmetic_sqrt_sqrt_16_l22_22665

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l22_22665


namespace sum_invested_l22_22358

theorem sum_invested (P R: ℝ) (h1: SI₁ = P * R * 20 / 100) (h2: SI₂ = P * (R + 10) * 20 / 100) (h3: SI₂ = SI₁ + 3000) : P = 1500 :=
by
  sorry

end sum_invested_l22_22358


namespace probability_sum_8_9_10_l22_22867

/-- The faces of the first die -/
def first_die := [2, 2, 3, 3, 5, 5]

/-- The faces of the second die -/
def second_die := [1, 3, 4, 5, 6, 7]

/-- Predicate that checks if the sum of two numbers is either 8, 9, or 10 -/
def valid_sum (a b : ℕ) : Prop := a + b = 8 ∨ a + b = 9 ∨ a + b = 10

/-- Calculate the probability of a sum being 8, 9, or 10 according to the given dice setup -/
def calc_probability : ℚ := 
  let valid_pairs := [(2, 6), (2, 7), (3, 5), (3, 6), (3, 7), (5, 3), (5, 4), (5, 5)] 
  (valid_pairs.length : ℚ) / (first_die.length * second_die.length : ℚ)

theorem probability_sum_8_9_10 : calc_probability = 4 / 9 :=
by
  sorry

end probability_sum_8_9_10_l22_22867


namespace candle_height_after_half_time_l22_22730

-- Define the parameters
def initial_height : ℕ := 100
def first_cm_time : ℕ := 15
def time_increment : ℕ := 15
def total_cm : ℕ := 100

-- Arithmetic sequence sum function
def arithmetic_sum (n a d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- The total time to burn the candle completely
def total_time : ℕ :=
  arithmetic_sum total_cm first_cm_time time_increment

-- Half of the total burning time
def half_total_time : ℕ := total_time / 2

-- Find the height of the candle after half the burning time
theorem candle_height_after_half_time :
  initial_height - 70 = 30 := by
  sorry

end candle_height_after_half_time_l22_22730


namespace cameron_total_questions_l22_22750

def usual_questions : Nat := 2

def group_a_questions : Nat := 
  let q1 := 2 * 1 -- 2 people who asked a single question each
  let q2 := 3 * usual_questions -- 3 people who asked two questions as usual
  let q3 := 1 * 5 -- 1 person who asked 5 questions
  q1 + q2 + q3

def group_b_questions : Nat :=
  let q1 := 1 * 0 -- 1 person asked no questions
  let q2 := 6 * 3 -- 6 people asked 3 questions each
  let q3 := 4 * usual_questions -- 4 people asked the usual number of questions
  q1 + q2 + q3

def group_c_questions : Nat :=
  let q1 := 1 * (usual_questions * 3) -- 1 person asked three times as many questions as usual
  let q2 := 1 * 1 -- 1 person asked only one question
  let q3 := 2 * 0 -- 2 members asked no questions
  let q4 := 4 * usual_questions -- The remaining tourists asked the usual 2 questions each
  q1 + q2 + q3 + q4

def group_d_questions : Nat :=
  let q1 := 1 * (usual_questions * 4) -- 1 individual asked four times as many questions as normal
  let q2 := 1 * 0 -- 1 person asked no questions at all
  let q3 := 3 * usual_questions -- The remaining tourists asked the usual number of questions
  q1 + q2 + q3

def group_e_questions : Nat :=
  let q1 := 3 * (usual_questions * 2) -- 3 people asked double the average number of questions
  let q2 := 2 * 0 -- 2 people asked none
  let q3 := 1 * 5 -- 1 tourist asked 5 questions
  let q4 := 3 * usual_questions -- The remaining tourists asked the usual number
  q1 + q2 + q3 + q4

def group_f_questions : Nat :=
  let q1 := 2 * 3 -- 2 individuals asked three questions each
  let q2 := 1 * 0 -- 1 person asked no questions
  let q3 := 4 * usual_questions -- The remaining tourists asked the usual number
  q1 + q2 + q3

def total_questions : Nat :=
  group_a_questions + group_b_questions + group_c_questions + group_d_questions + group_e_questions + group_f_questions

theorem cameron_total_questions : total_questions = 105 := by
  sorry

end cameron_total_questions_l22_22750


namespace circle_radius_l22_22698

theorem circle_radius (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end circle_radius_l22_22698


namespace average_infection_per_round_l22_22025

theorem average_infection_per_round (x : ℝ) (h1 : 1 + x + x * (1 + x) = 100) : x = 9 :=
sorry

end average_infection_per_round_l22_22025


namespace simplify_expression_l22_22857

theorem simplify_expression (a b : ℤ) : 
  (18 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 40 * b) = 21 * a + 41 * b := 
by
  sorry

end simplify_expression_l22_22857


namespace sum_of_angles_l22_22259

variables (A B C D E F : ℝ)

theorem sum_of_angles 
  (h : E = 30) :
  A + B + C + D + E + F = 420 :=
sorry

end sum_of_angles_l22_22259


namespace weigh_80_grams_is_false_l22_22346

def XiaoGang_weight_grams : Nat := 80000  -- 80 kilograms in grams
def weight_claim : Nat := 80  -- 80 grams claim

theorem weigh_80_grams_is_false : weight_claim ≠ XiaoGang_weight_grams :=
by
  sorry

end weigh_80_grams_is_false_l22_22346


namespace trajectory_of_point_M_l22_22495

theorem trajectory_of_point_M (a x y : ℝ) (h: 0 < a) (A B M : ℝ × ℝ)
    (hA : A = (x, 0)) (hB : B = (0, y)) (hAB_length : Real.sqrt (x^2 + y^2) = 2 * a)
    (h_ratio : ∃ k, k ≠ 0 ∧ ∃ k', k' ≠ 0 ∧ A = k • M + k' • B ∧ (k + k' = 1) ∧ (k / k' = 1 / 2)) :
    (x / (4 / 3 * a))^2 + (y / (2 / 3 * a))^2 = 1 :=
sorry

end trajectory_of_point_M_l22_22495


namespace bea_glasses_sold_is_10_l22_22363

variable (B : ℕ)
variable (earnings_bea earnings_dawn : ℕ)

def bea_price_per_glass := 25
def dawn_price_per_glass := 28
def dawn_glasses_sold := 8
def earnings_diff := 26

def bea_earnings := bea_price_per_glass * B
def dawn_earnings := dawn_price_per_glass * dawn_glasses_sold

def bea_earnings_greater := bea_earnings = dawn_earnings + earnings_diff

theorem bea_glasses_sold_is_10 (h : bea_earnings_greater) : B = 10 :=
by sorry

end bea_glasses_sold_is_10_l22_22363


namespace count_correct_statements_l22_22426

theorem count_correct_statements :
  ∃ (M: ℚ) (M1: ℚ) (M2: ℚ) (M3: ℚ) (M4: ℚ)
    (a b c d e : ℚ) (hacb : c ≠ 0) (habc: a ≠ 0) (hbcb : b ≠ 0) (hdcb: d ≠ 0) (hec: e ≠ 0),
  M = (ac + bd - ce) / c 
  ∧ M1 = (-bc - ad - ce) / c 
  ∧ M2 = (-dc - ab - ce) / c 
  ∧ M3 = (-dc - ab - de) / d 
  ∧ M4 = (ce - bd - ac) / (-c)
  ∧ M4 = M
  ∧ (M ≠ M3)
  ∧ (∀ M1, M1 = (-bc - ad - ce) / c → ((a = c ∨ b = d) ↔ b = d))
  ∧ (M4 = (ac + bd - ce)/c) :=
sorry

end count_correct_statements_l22_22426


namespace trees_still_left_l22_22713

theorem trees_still_left 
  (initial_trees : ℕ) 
  (trees_died : ℕ) 
  (trees_cut : ℕ) 
  (initial_trees_eq : initial_trees = 86) 
  (trees_died_eq : trees_died = 15) 
  (trees_cut_eq : trees_cut = 23) 
  : initial_trees - (trees_died + trees_cut) = 48 :=
by
  sorry

end trees_still_left_l22_22713


namespace defective_bolt_probability_l22_22746

noncomputable def machine1_prob : ℝ := 0.30
noncomputable def machine2_prob : ℝ := 0.25
noncomputable def machine3_prob : ℝ := 0.45

noncomputable def defect_prob_machine1 : ℝ := 0.02
noncomputable def defect_prob_machine2 : ℝ := 0.01
noncomputable def defect_prob_machine3 : ℝ := 0.03

noncomputable def total_defect_prob : ℝ :=
  machine1_prob * defect_prob_machine1 +
  machine2_prob * defect_prob_machine2 +
  machine3_prob * defect_prob_machine3

theorem defective_bolt_probability : total_defect_prob = 0.022 := by
  sorry

end defective_bolt_probability_l22_22746


namespace probability_two_8sided_dice_sum_perfect_square_l22_22143

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probabilityPerfectSquareSum (dice_sides : ℕ) (perfect_squares : List ℕ) : ℚ :=
  let outcomes := (dice_sides * dice_sides)
  let favorable_outcomes := perfect_squares.sum (λ ps, 
    (List.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd = ps) ((List.range dice_sides).product (List.range dice_sides))).length)
  favorable_outcomes /. outcomes

theorem probability_two_8sided_dice_sum_perfect_square :
  probabilityPerfectSquareSum 8 [4, 9, 16] = 3 / 16 := sorry

end probability_two_8sided_dice_sum_perfect_square_l22_22143


namespace exists_base_for_part_a_not_exists_base_for_part_b_l22_22373

theorem exists_base_for_part_a : ∃ b : ℕ, (3 + 4 = b) ∧ (3 * 4 = 1 * b + 5) := 
by
  sorry

theorem not_exists_base_for_part_b : ¬ ∃ b : ℕ, (2 + 3 = b) ∧ (2 * 3 = 1 * b + 1) :=
by
  sorry

end exists_base_for_part_a_not_exists_base_for_part_b_l22_22373


namespace problem_equivalent_proof_statement_l22_22167

-- Definition of a line with a definite slope
def has_definite_slope (m : ℝ) : Prop :=
  ∃ slope : ℝ, slope = -m 

-- Definition of the equation of a line passing through two points being correct
def line_through_two_points (x1 y1 x2 y2 : ℝ) (h : x1 ≠ x2) : Prop :=
  ∀ x y : ℝ, (y - y1 = ((y2 - y1) / (x2 - x1)) * (x - x1)) ↔ y = ((y2 - y1) * (x - x1) / (x2 - x1)) + y1 

-- Formalizing and proving the given conditions
theorem problem_equivalent_proof_statement : 
  (∀ m : ℝ, has_definite_slope m) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ) (h : x1 ≠ x2), line_through_two_points x1 y1 x2 y2 h) :=
by 
  sorry

end problem_equivalent_proof_statement_l22_22167


namespace intersection_is_negative_real_l22_22299

def setA : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2 * x + 1}
def setB : Set ℝ := {y : ℝ | ∃ x : ℝ, y = - x ^ 2}

theorem intersection_is_negative_real :
  setA ∩ setB = {y : ℝ | y ≤ 0} := 
sorry

end intersection_is_negative_real_l22_22299


namespace new_profit_percentage_l22_22925

theorem new_profit_percentage (P SP NSP NP : ℝ) 
  (h1 : SP = 1.10 * P) 
  (h2 : SP = 879.9999999999993) 
  (h3 : NP = 0.90 * P) 
  (h4 : NSP = SP + 56) : 
  (NSP - NP) / NP * 100 = 30 := 
by
  sorry

end new_profit_percentage_l22_22925


namespace incorrect_statement_C_l22_22374

-- Lean 4 statement to verify correctness of problem translation
theorem incorrect_statement_C (n : ℕ) (w : ℕ → ℕ) :
  (w 1 = 55) ∧
  (w 2 = 110) ∧
  (w 3 = 160) ∧
  (w 4 = 200) ∧
  (w 5 = 254) ∧
  (w 6 = 300) ∧
  (w 7 = 350) →
  ¬(∀ n, w n = 55 * n) :=
by
  intros h
  sorry

end incorrect_statement_C_l22_22374


namespace sphere_radius_five_times_surface_area_l22_22709

theorem sphere_radius_five_times_surface_area (R : ℝ) (h₁ : (4 * π * R^3 / 3) = 5 * (4 * π * R^2)) : R = 15 :=
sorry

end sphere_radius_five_times_surface_area_l22_22709


namespace mean_of_six_numbers_l22_22331

theorem mean_of_six_numbers (sum_six_numbers : ℚ) (h : sum_six_numbers = 3/4) : 
  (sum_six_numbers / 6) = 1/8 := by
  -- proof can be filled in here
  sorry

end mean_of_six_numbers_l22_22331


namespace total_cost_correct_l22_22587

noncomputable def totalCost : ℝ :=
  let fuel_efficiences := [15, 12, 14, 10, 13, 15]
  let distances := [10, 6, 7, 5, 3, 9]
  let gas_prices := [3.5, 3.6, 3.4, 3.55, 3.55, 3.5]
  let gas_used := distances.zip fuel_efficiences |>.map (λ p => (p.1 : ℝ) / p.2)
  let costs := gas_prices.zip gas_used |>.map (λ p => p.1 * p.2)
  costs.sum

theorem total_cost_correct : abs (totalCost - 10.52884) < 0.01 := by
  sorry

end total_cost_correct_l22_22587


namespace hurleys_age_l22_22680

-- Definitions and conditions
variable (H R : ℕ)
variable (cond1 : R - H = 20)
variable (cond2 : (R + 40) + (H + 40) = 128)

-- Theorem statement
theorem hurleys_age (H R : ℕ) (cond1 : R - H = 20) (cond2 : (R + 40) + (H + 40) = 128) : H = 14 := 
by
  sorry

end hurleys_age_l22_22680


namespace number_four_units_away_from_neg_five_l22_22589

theorem number_four_units_away_from_neg_five (x : ℝ) : 
    abs (x + 5) = 4 ↔ x = -9 ∨ x = -1 :=
by 
  sorry

end number_four_units_away_from_neg_five_l22_22589


namespace ten_factorial_mod_thirteen_l22_22225

open Nat

theorem ten_factorial_mod_thirteen :
  (10! % 13) = 6 := by
  sorry

end ten_factorial_mod_thirteen_l22_22225


namespace problem_statement_l22_22045

noncomputable def f (x : ℝ) (b c : ℝ) := x^2 + b * x + c

theorem problem_statement (b c : ℝ) (h : ∀ x : ℝ, f (x - 1) b c = f (3 - x) b c) : f 0 b c < f (-2) b c ∧ f (-2) b c < f 5 b c := 
by sorry

end problem_statement_l22_22045


namespace rounded_diff_greater_l22_22567

variable (x y ε : ℝ)
variable (h1 : x > y)
variable (h2 : y > 0)
variable (h3 : ε > 0)

theorem rounded_diff_greater : (x + ε) - (y - ε) > x - y :=
  by
  sorry

end rounded_diff_greater_l22_22567


namespace x_sq_sub_y_sq_l22_22411

theorem x_sq_sub_y_sq (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 :=
by
  sorry

end x_sq_sub_y_sq_l22_22411


namespace expand_polynomial_product_l22_22200

variable (x : ℝ)

theorem expand_polynomial_product :
  (3 * x + 4) * (2 * x + 7) = 6 * x^2 + 29 * x + 28 := by
  sorry

end expand_polynomial_product_l22_22200


namespace simplify_sqrt7_pow6_l22_22623

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l22_22623


namespace num_divisors_180_l22_22553

-- Define a positive integer 180
def n : ℕ := 180

-- Define the function to calculate the number of divisors using prime factorization
def num_divisors (n : ℕ) : ℕ :=
  let factors := [(2, 2), (3, 2), (5, 1)] in
  factors.foldl (λ acc (p : ℕ × ℕ), acc * (p.snd + 1)) 1

-- The main theorem statement
theorem num_divisors_180 : num_divisors n = 18 :=
by
  sorry

end num_divisors_180_l22_22553


namespace solve_inequality_l22_22018

theorem solve_inequality (x : ℝ) :
  (x - 1)^2 < 12 - x ↔ 
  (Real.sqrt 5) ≠ 0 ∧
  (1 - 3 * (Real.sqrt 5)) / 2 < x ∧ 
  x < (1 + 3 * (Real.sqrt 5)) / 2 :=
sorry

end solve_inequality_l22_22018


namespace hyperbola_eccentricity_l22_22383

theorem hyperbola_eccentricity :
  let a := 2
  let b := 2 * Real.sqrt 2
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (e = Real.sqrt 3) :=
by {
  sorry
}

end hyperbola_eccentricity_l22_22383


namespace complement_A_complement_A_intersection_B_intersection_A_B_complement_intersection_A_B_l22_22108

def U : Set ℝ := {x | x ≥ -2}
def A : Set ℝ := {x | 2 < x ∧ x < 10}
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}

theorem complement_A :
  (U \ A) = {x | -2 ≤ x ∧ x ≤ 2 ∨ x ≥ 10} :=
by sorry

theorem complement_A_intersection_B :
  (U \ A) ∩ B = {2} :=
by sorry

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 8} :=
by sorry

theorem complement_intersection_A_B :
  U \ (A ∩ B) = {x | -2 ≤ x ∧ x ≤ 2 ∨ x > 8} :=
by sorry

end complement_A_complement_A_intersection_B_intersection_A_B_complement_intersection_A_B_l22_22108


namespace class_size_l22_22448

theorem class_size
  (S_society : ℕ) (S_music : ℕ) (S_both : ℕ) (S : ℕ)
  (h_society : S_society = 25)
  (h_music : S_music = 32)
  (h_both : S_both = 27)
  (h_total : S = S_society + S_music - S_both) :
  S = 30 :=
by
  rw [h_society, h_music, h_both] at h_total
  exact h_total

end class_size_l22_22448


namespace quadratic_intersects_at_3_points_l22_22095

theorem quadratic_intersects_at_3_points (m : ℝ) : 
  (exists x : ℝ, x^2 + 2*x + m = 0) ∧ (m ≠ 0) → m < 1 :=
by
  sorry

end quadratic_intersects_at_3_points_l22_22095


namespace P_72_l22_22438

def P (n : ℕ) : ℕ :=
  -- The definition of P(n) should enumerate the ways of expressing n as a product
  -- of integers greater than 1, considering the order of factors.
  sorry

theorem P_72 : P 72 = 17 :=
by
  sorry

end P_72_l22_22438


namespace def_integral_abs_x2_minus_2x_l22_22931

theorem def_integral_abs_x2_minus_2x :
  (∫ x in -2..2, |x^2 - 2*x|) = 8 :=
by
  -- Conditions specified in the text
  have h1 : ∀ x, x ∈ set.Icc (-2:ℝ) 0 → (x^2 - 2*x) ≥ 0 := by
    intros x hx
    linarith [hx.1, hx.2, pow_two_nonneg x]
  have h2 : ∀ x, x ∈ set.Ioc (0:ℝ) 2 → (x^2 - 2*x) < 0 := by
    intros x hx
    linarith [hx.1, hx.2, pow_two_nonneg x]
  sorry

end def_integral_abs_x2_minus_2x_l22_22931


namespace value_of_expression_l22_22964

theorem value_of_expression (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by {
   rw h,
   norm_num,
   sorry
}

end value_of_expression_l22_22964


namespace simplify_sqrt_pow_six_l22_22641

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l22_22641


namespace library_average_visitors_l22_22911

theorem library_average_visitors (V : ℝ) (h1 : (4 * 1000 + 26 * V = 750 * 30)) : V = 18500 / 26 := 
by 
  -- The actual proof is omitted and replaced by sorry.
  sorry

end library_average_visitors_l22_22911


namespace estimate_total_fish_l22_22113

theorem estimate_total_fish (marked : ℕ) (sample_size : ℕ) (marked_in_sample : ℕ) (x : ℝ) 
  (h1 : marked = 50) 
  (h2 : sample_size = 168) 
  (h3 : marked_in_sample = 8) 
  (h4 : sample_size * 50 = marked_in_sample * x) : 
  x = 1050 := 
sorry

end estimate_total_fish_l22_22113


namespace words_to_score_A_l22_22958

-- Define the total number of words
def total_words : ℕ := 600

-- Define the target percentage
def target_percentage : ℚ := 90 / 100

-- Define the minimum number of words to learn
def min_words_to_learn : ℕ := 540

-- Define the condition for scoring at least 90%
def meets_requirement (learned_words : ℕ) : Prop :=
  learned_words / total_words ≥ target_percentage

-- The goal is to prove that learning 540 words meets the requirement
theorem words_to_score_A : meets_requirement min_words_to_learn :=
by
  sorry

end words_to_score_A_l22_22958


namespace andrew_kept_correct_l22_22744

open Nat

def andrew_bought : ℕ := 750
def daniel_received : ℕ := 250
def fred_received : ℕ := daniel_received + 120
def total_shared : ℕ := daniel_received + fred_received
def andrew_kept : ℕ := andrew_bought - total_shared

theorem andrew_kept_correct : andrew_kept = 130 :=
by
  unfold andrew_kept andrew_bought total_shared fred_received daniel_received
  rfl

end andrew_kept_correct_l22_22744


namespace least_subtracted_to_divisible_by_10_l22_22203

theorem least_subtracted_to_divisible_by_10 (n : ℕ) (k : ℕ) (h : n = 724946) (div_cond : (n - k) % 10 = 0) : k = 6 :=
by
  sorry

end least_subtracted_to_divisible_by_10_l22_22203


namespace evaluate_K_l22_22026

theorem evaluate_K : ∃ K : ℕ, 32^2 * 4^4 = 2^K ∧ K = 18 := by
  use 18
  sorry

end evaluate_K_l22_22026


namespace current_failing_rate_l22_22273

def failing_student_rate := 28

def is_failing_student_rate (V : Prop) (n : ℕ) (rate : ℕ) : Prop :=
  (V ∧ rate = 24 ∧ n = 25) ∨ (¬V ∧ rate = 25 ∧ n - 1 = 24)

theorem current_failing_rate (V : Prop) (n : ℕ) (rate : ℕ) :
  is_failing_student_rate V n rate → rate = failing_student_rate :=
by
  sorry

end current_failing_rate_l22_22273


namespace best_fitting_model_is_model_3_l22_22824

-- Define models with their corresponding R^2 values
def R_squared_model_1 : ℝ := 0.72
def R_squared_model_2 : ℝ := 0.64
def R_squared_model_3 : ℝ := 0.98
def R_squared_model_4 : ℝ := 0.81

-- Define a proposition that model 3 has the best fitting effect
def best_fitting_model (R1 R2 R3 R4 : ℝ) : Prop :=
  R3 = max (max R1 R2) (max R3 R4)

-- State the theorem that we need to prove
theorem best_fitting_model_is_model_3 :
  best_fitting_model R_squared_model_1 R_squared_model_2 R_squared_model_3 R_squared_model_4 :=
by
  sorry

end best_fitting_model_is_model_3_l22_22824


namespace simplify_sqrt_seven_pow_six_l22_22633

theorem simplify_sqrt_seven_pow_six : (real.sqrt 7)^6 = 343 :=
by
  sorry

end simplify_sqrt_seven_pow_six_l22_22633


namespace correct_statements_about_f_l22_22035

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - (13 / 4) * Real.pi)

theorem correct_statements_about_f :
  ((f (Real.pi / 8) = 0) ∧ (f (Real.pi / 8) = f (-(Real.pi / 8)))) ∧  -- Statement A
  (f(2 * (Real.pi / 8)) = 2 * Real.sin(2 * (Real.pi / 8) - (13 / 4) * Real.pi)) ∧  -- Statement B
  (f (2 * (Real.pi / 8) + 5/8 * Real.pi) = 2 * Real.sin (2 * (Real.pi / 8 + 5/8 * Real.pi) - 13 / 4 * Real.pi)) ∧ -- Statement C
  (f (2 * (Real.pi / 8) - 5/8 * Real.pi) = 2 * Real.sin (2 * (Real.pi / 8 - 5/8 * Real.pi) - 13 / 4 * Real.pi))  -- Statement D
:= sorry

end correct_statements_about_f_l22_22035


namespace simplify_sqrt_pow_six_l22_22642

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l22_22642


namespace minimum_w_value_l22_22942

theorem minimum_w_value : 
  (∀ x y : ℝ, w = 2*x^2 + 3*y^2 - 12*x + 9*y + 35) → 
  ∃ w_min : ℝ, w_min = 41 / 4 ∧ 
  (∀ x y : ℝ, 2*x^2 + 3*y^2 - 12*x + 9*y + 35 ≥ w_min) :=
by
  sorry

end minimum_w_value_l22_22942


namespace simplify_sqrt7_pow6_l22_22638

theorem simplify_sqrt7_pow6 : (sqrt 7)^6 = 343 := 
by
  sorry

end simplify_sqrt7_pow6_l22_22638


namespace problem_equivalence_l22_22971

variables (P Q : Prop)

theorem problem_equivalence :
  (P ↔ Q) ↔ ((P → Q) ∧ (Q → P) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q)) :=
by sorry

end problem_equivalence_l22_22971


namespace transformed_curve_l22_22029

theorem transformed_curve (x y : ℝ) :
  (∃ (x1 y1 : ℝ), x1 = 3*x ∧ y1 = 2*y ∧ (x1^2 / 9 + y1^2 / 4 = 1)) →
  x^2 + y^2 = 1 :=
by
  sorry

end transformed_curve_l22_22029


namespace number_of_factors_n_l22_22762

-- Defining the value of n with its prime factorization
def n : ℕ := 2^5 * 3^9 * 5^5

-- Theorem stating the number of natural-number factors of n
theorem number_of_factors_n : 
  (Nat.divisors n).card = 360 := by
  -- Proof is omitted
  sorry

end number_of_factors_n_l22_22762


namespace and_15_and_l22_22031

def x_and (x : ℝ) : ℝ := 8 - x
def and_x (x : ℝ) : ℝ := x - 8

theorem and_15_and : and_x (x_and 15) = -15 :=
by
  sorry

end and_15_and_l22_22031


namespace simplify_sqrt_7_pow_6_l22_22645

theorem simplify_sqrt_7_pow_6 : (sqrt 7)^6 = 343 := by
  sorry

end simplify_sqrt_7_pow_6_l22_22645


namespace product_of_possible_values_l22_22409

theorem product_of_possible_values :
  (∀ x : ℝ, abs (18 / x + 4) = 3 → x = -18 ∨ x = -18 / 7) →
  (∀ x1 x2 : ℝ, x1 = -18 → x2 = -18 / 7 → x1 * x2 = 324 / 7) :=
by
  intros h x1 x2 hx1 hx2
  rw [hx1, hx2]
  norm_num

end product_of_possible_values_l22_22409


namespace probability_two_8sided_dice_sum_perfect_square_l22_22142

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probabilityPerfectSquareSum (dice_sides : ℕ) (perfect_squares : List ℕ) : ℚ :=
  let outcomes := (dice_sides * dice_sides)
  let favorable_outcomes := perfect_squares.sum (λ ps, 
    (List.filter (λ (pair : ℕ × ℕ), pair.fst + pair.snd = ps) ((List.range dice_sides).product (List.range dice_sides))).length)
  favorable_outcomes /. outcomes

theorem probability_two_8sided_dice_sum_perfect_square :
  probabilityPerfectSquareSum 8 [4, 9, 16] = 3 / 16 := sorry

end probability_two_8sided_dice_sum_perfect_square_l22_22142


namespace simplify_sqrt_pow_six_l22_22640

theorem simplify_sqrt_pow_six : (sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_pow_six_l22_22640


namespace parallelogram_lengths_l22_22391

noncomputable def parallelogram_data 
  (ABCD : Type) 
  (A B C D M E K : ABCD) 
  (diameter : ℝ) 
  (EM_length : ℝ) 
  (arc_AE : ℝ) 
  (arc_BM : ℝ) 
  (Ω : Type) :=
  diameter = 13 ∧
  EM_length = 12 ∧
  arc_AE = 2 * arc_BM ∧
  (Ω circumscribe (triangle ABC M))

theorem parallelogram_lengths 
  (ABCD : Type) 
  (A B C D M E K : ABCD) 
  (diameter : ℝ) 
  (EM_length : ℝ) 
  (arc_AE : ℝ) 
  (arc_BM : ℝ) 
  (Ω : Type)
  (h_parallelogram : parallelogram_data ABCD A B C D M E K diameter EM_length arc_AE arc_BM Ω):
  BC = 13 ∧ BK = 120 / 13 ∧ (AK + KM + MA) = 340 / 13 := sorry

end parallelogram_lengths_l22_22391


namespace impossible_coins_l22_22596

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l22_22596


namespace area_white_portion_l22_22466

/-- The dimensions of the sign --/
def sign_width : ℝ := 7
def sign_height : ℝ := 20

/-- The areas of letters "S", "A", "V", and "E" --/
def area_S : ℝ := 14
def area_A : ℝ := 16
def area_V : ℝ := 12
def area_E : ℝ := 12

/-- Calculate the total area of the sign --/
def total_area_sign : ℝ := sign_width * sign_height

/-- Calculate the total area covered by the letters --/
def total_area_letters : ℝ := area_S + area_A + area_V + area_E

/-- Calculate the area of the white portion of the sign --/
theorem area_white_portion : total_area_sign - total_area_letters = 86 := by
  sorry

end area_white_portion_l22_22466


namespace coins_with_specific_probabilities_impossible_l22_22605

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l22_22605


namespace impossible_coins_l22_22613

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l22_22613


namespace value_of_a_plus_b_l22_22778

theorem value_of_a_plus_b (a b : Int) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
by
  sorry

end value_of_a_plus_b_l22_22778


namespace sum_of_angles_l22_22260

variables (A B C D E F : ℝ)

theorem sum_of_angles 
  (h : E = 30) :
  A + B + C + D + E + F = 420 :=
sorry

end sum_of_angles_l22_22260


namespace ten_factorial_mod_thirteen_l22_22224

open Nat

theorem ten_factorial_mod_thirteen :
  (10! % 13) = 6 := by
  sorry

end ten_factorial_mod_thirteen_l22_22224


namespace hallie_hours_worked_on_tuesday_l22_22404

theorem hallie_hours_worked_on_tuesday
    (hourly_wage : ℝ := 10)
    (hours_monday : ℝ := 7)
    (tips_monday : ℝ := 18)
    (hours_wednesday : ℝ := 7)
    (tips_wednesday : ℝ := 20)
    (tips_tuesday : ℝ := 12)
    (total_earnings : ℝ := 240)
    (tuesday_hours : ℝ) :
    (hourly_wage * hours_monday + tips_monday) +
    (hourly_wage * hours_wednesday + tips_wednesday) +
    (hourly_wage * tuesday_hours + tips_tuesday) = total_earnings →
    tuesday_hours = 5 :=
by
  sorry

end hallie_hours_worked_on_tuesday_l22_22404


namespace bottle_caps_left_l22_22578

theorem bottle_caps_left {init_caps given_away_rebecca given_away_michael left_caps : ℝ} 
  (h1 : init_caps = 143.6)
  (h2 : given_away_rebecca = 89.2)
  (h3 : given_away_michael = 16.7)
  (h4 : left_caps = init_caps - (given_away_rebecca + given_away_michael)) :
  left_caps = 37.7 := by
  sorry

end bottle_caps_left_l22_22578


namespace solve_for_s_l22_22354

---
theorem solve_for_s (s : ℝ) (h₀: s > 0)
(h₁ : let θ := real.pi / 3 in area = s * 3 * (s * real.sin θ))
(h₂ : area = 27 * real.sqrt 3) : s = 3 * real.sqrt 2 := by
  sorry

end solve_for_s_l22_22354


namespace range_a_satisfies_l22_22414

theorem range_a_satisfies (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, f x = x^3) (h₂ : f 2 = 8) :
  (f (a - 3) > f (1 - a)) ↔ a > 2 :=
by
  sorry

end range_a_satisfies_l22_22414


namespace two_digit_numbers_count_l22_22069

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_numbers_count :
  { n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ (tens_digit n < 3 ∨ units_digit n < 5) }.to_finset.card = 55 :=
by
  sorry

end two_digit_numbers_count_l22_22069


namespace triangle_area_condition_l22_22794

theorem triangle_area_condition (m : ℝ) 
  (H_line : ∀ (x y : ℝ), x - m*y + 1 = 0)
  (H_circle : ∀ (x y : ℝ), (x - 1)^2 + y^2 = 4)
  (H_area : ∃ (A B C : (ℝ × ℝ)), (x - my + 1 = 0) ∧ (∃ C : (ℝ × ℝ), (x1 - 1)^2 + y1^2 = 4 ∨ (x2 - 1)^2 + y2^2 = 4))
  : m = 2 :=
sorry

end triangle_area_condition_l22_22794


namespace faye_initial_coloring_books_l22_22381

theorem faye_initial_coloring_books (gave_away1 gave_away2 remaining : ℝ) 
    (h1 : gave_away1 = 34.0) (h2 : gave_away2 = 3.0) (h3 : remaining = 11.0) :
    gave_away1 + gave_away2 + remaining = 48.0 := 
by
  sorry

end faye_initial_coloring_books_l22_22381


namespace sequence_sum_l22_22983

-- Defining the sequence terms
variables (J K L M N O P Q R S : ℤ)
-- Condition N = 7
def N_value : Prop := N = 7
-- Condition sum of any four consecutive terms is 40
def sum_of_consecutive : Prop := 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40

-- The main theorem stating J + S = 40 given the conditions
theorem sequence_sum (N_value : N = 7) (sum_of_consecutive : 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40) : 
  J + S = 40 := sorry

end sequence_sum_l22_22983


namespace max_product_is_negative_one_l22_22370

def f (x : ℝ) : ℝ := sorry    -- Assume some function f
def g (x : ℝ) : ℝ := sorry    -- Assume some function g

theorem max_product_is_negative_one (h_f_range : ∀ y, 1 ≤ y ∧ y ≤ 6 → ∃ x, f x = y) 
    (h_g_range : ∀ y, -4 ≤ y ∧ y ≤ -1 → ∃ x, g x = y) : 
    ∃ b, b = -1 ∧ ∀ x, f x * g x ≤ b :=
sorry

end max_product_is_negative_one_l22_22370


namespace relationship_among_abc_l22_22243

noncomputable def a : ℝ := 4^(1/3 : ℝ)
noncomputable def b : ℝ := Real.log 1/7 / Real.log 3
noncomputable def c : ℝ := (1/3 : ℝ)^(1/5 : ℝ)

theorem relationship_among_abc : a > c ∧ c > b := 
by 
  sorry

end relationship_among_abc_l22_22243


namespace part1_part2_l22_22800

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * x^2 - Real.log x

theorem part1 (a : ℝ) (h : 0 < a) (hf'1 : (1 - 2 * a * 1 - 1) = -2) :
  a = 1 ∧ (∀ x y : ℝ, y = -2 * (x - 1) → 2 * x + y - 2 = 0) :=
by
  sorry

theorem part2 {a : ℝ} (ha : a ≥ 1 / 8) :
  ∀ x : ℝ, (1 - 2 * a * x - 1 / x) ≤ 0 :=
by
  sorry

end part1_part2_l22_22800


namespace find_angle_B_l22_22393

-- Given definitions and conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variable (h1 : (a + b + c) * (a - b + c) = a * c )

-- Statement of the proof problem
theorem find_angle_B (h1 : (a + b + c) * (a - b + c) = a * c) :
  B = 2 * π / 3 :=
sorry

end find_angle_B_l22_22393


namespace part_a_part_b_l22_22726

-- Part (a)
theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (a b : ℝ), a ≠ b ∧ 0 < (a - b) / (1 + a * b) ∧ (a - b) / (1 + a * b) ≤ 1 := sorry

-- Part (b)
theorem part_b (x y z u : ℝ) :
  ∃ (a b : ℝ), a ≠ b ∧ 0 < (b - a) / (1 + a * b) ∧ (b - a) / (1 + a * b) ≤ 1 := sorry

end part_a_part_b_l22_22726


namespace range_of_m_l22_22955

theorem range_of_m (a m : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  m * (a + 1/a) / Real.sqrt 2 > 1 → m ≥ Real.sqrt 2 / 2 := by
  sorry

end range_of_m_l22_22955


namespace coins_with_specific_probabilities_impossible_l22_22608

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l22_22608


namespace number_of_divisors_180_l22_22554

theorem number_of_divisors_180 : (∃ (n : ℕ), n = 180 ∧ (∀ (e1 e2 e3 : ℕ), 180 = 2^e1 * 3^e2 * 5^e3 → (e1 + 1) * (e2 + 1) * (e3 + 1) = 18)) :=
  sorry

end number_of_divisors_180_l22_22554


namespace coins_with_specific_probabilities_impossible_l22_22604

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l22_22604


namespace problem_I3_1_l22_22043

theorem problem_I3_1 (w x y z : ℝ) (h1 : w * x * y * z = 4) (h2 : w - x * y * z = 3) (h3 : w > 0) : 
  w = 4 :=
by
  sorry

end problem_I3_1_l22_22043


namespace correct_calculation_l22_22721

theorem correct_calculation (a b : ℝ) :
  ((ab)^3 = a^3 * b^3) ∧ 
  ¬(a + 2 * a^2 = 3 * a^3) ∧ 
  ¬(a * (-a)^4 = -a^5) ∧ 
  ¬((a^3)^2 = a^5) :=
  by
  sorry

end correct_calculation_l22_22721


namespace second_derivative_at_x₀_l22_22844

noncomputable def f (x : ℝ) : ℝ := sorry
variables (x₀ a b : ℝ)

-- Condition: f(x₀ + Δx) - f(x₀) = a * Δx + b * (Δx)^2
axiom condition : ∀ Δx, f (x₀ + Δx) - f x₀ = a * Δx + b * (Δx)^2

theorem second_derivative_at_x₀ : deriv (deriv f) x₀ = 2 * b :=
sorry

end second_derivative_at_x₀_l22_22844


namespace not_prime_for_any_n_l22_22853

theorem not_prime_for_any_n (k : ℕ) (hk : 1 < k) (n : ℕ) : 
  ¬ Prime (n^4 + 4 * k^4) :=
sorry

end not_prime_for_any_n_l22_22853


namespace shampoo_duration_l22_22281

-- Conditions
def rose_shampoo : ℚ := 1/3
def jasmine_shampoo : ℚ := 1/4
def daily_usage : ℚ := 1/12

-- Question
theorem shampoo_duration : (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end shampoo_duration_l22_22281


namespace impossible_coins_l22_22597

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l22_22597


namespace n_energetic_all_n_specific_energetic_constraints_l22_22199

-- Proof Problem 1
theorem n_energetic_all_n (a b c : ℕ) (n : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : Nat.gcd a (Nat.gcd b c) = 1) 
(h4 : ∀ n ≥ 1, (a^n + b^n + c^n) % (a + b + c) = 0) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 4) := sorry

-- Proof Problem 2
theorem specific_energetic_constraints (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) 
(h3 : Nat.gcd a (Nat.gcd b c) = 1) 
(h4 : (a^2004 + b^2004 + c^2004) % (a + b + c) = 0)
(h5 : (a^2005 + b^2005 + c^2005) % (a + b + c) = 0) 
(h6 : (a^2007 + b^2007 + c^2007) % (a + b + c) ≠ 0) :
  false := sorry

end n_energetic_all_n_specific_energetic_constraints_l22_22199


namespace probability_dice_sum_perfect_square_l22_22149

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end probability_dice_sum_perfect_square_l22_22149


namespace impossible_coins_l22_22612

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l22_22612


namespace initial_pokemon_cards_l22_22576

variables (x : ℕ)

theorem initial_pokemon_cards (h : x - 2 = 1) : x = 3 := 
sorry

end initial_pokemon_cards_l22_22576


namespace factorial_mod_10_l22_22218

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the problem statement
theorem factorial_mod_10 : factorial 10 % 13 = 7 :=
by sorry

end factorial_mod_10_l22_22218


namespace factorial_mod_10_l22_22221

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the problem statement
theorem factorial_mod_10 : factorial 10 % 13 = 7 :=
by sorry

end factorial_mod_10_l22_22221


namespace find_length_of_first_train_l22_22359

noncomputable def length_of_first_train (speed_train1 speed_train2 : ℕ) (time_to_cross : ℕ) (length_train2 : ℚ) : ℚ :=
  let relative_speed := (speed_train1 + speed_train2) * 1000 / 3600
  let combined_length := relative_speed * time_to_cross
  combined_length - length_train2

theorem find_length_of_first_train :
  length_of_first_train 120 80 9 280.04 = 220 := sorry

end find_length_of_first_train_l22_22359


namespace anton_stationary_escalator_steps_l22_22189

theorem anton_stationary_escalator_steps
  (N : ℕ)
  (H1 : N = 30)
  (H2 : 5 * N = 150) :
  (stationary_steps : ℕ) = 50 :=
by
  sorry

end anton_stationary_escalator_steps_l22_22189


namespace largest_number_using_digits_l22_22720

theorem largest_number_using_digits (d1 d2 d3 : ℕ) (h1 : d1 = 7) (h2 : d2 = 1) (h3 : d3 = 0) : 
  ∃ n : ℕ, (n = 710) ∧ (∀ m : ℕ, (m = d1 * 100 + d2 * 10 + d3) ∨ (m = d1 * 100 + d3 * 10 + d2) ∨ (m = d2 * 100 + d1 * 10 + d3) ∨ 
  (m = d2 * 100 + d3 * 10 + d1) ∨ (m = d3 * 100 + d1 * 10 + d2) ∨ (m = d3 * 100 + d2 * 10 + d1) → n ≥ m) := 
by
  sorry

end largest_number_using_digits_l22_22720


namespace tank_fills_in_56_minutes_l22_22850

theorem tank_fills_in_56_minutes : 
  (∃ A B C : ℕ, (A = 40 ∧ B = 30 ∧ C = 20) ∧ 
                 ∃ capacity : ℕ, capacity = 950 ∧ 
                 ∃ time : ℕ, time = 56 ∧
                 ∀ cycle_time : ℕ, cycle_time = 3 ∧ 
                 ∀ net_water_per_cycle : ℕ, net_water_per_cycle = A + B - C ∧
                 ∀ total_cycles : ℕ, total_cycles = capacity / net_water_per_cycle ∧
                 ∀ total_time : ℕ, total_time = total_cycles * cycle_time - 1 ∧
                 total_time = time) :=
sorry

end tank_fills_in_56_minutes_l22_22850


namespace impossible_coins_l22_22603

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l22_22603


namespace coins_with_specific_probabilities_impossible_l22_22607

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l22_22607


namespace remainder_of_2_pow_33_mod_9_l22_22873

theorem remainder_of_2_pow_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end remainder_of_2_pow_33_mod_9_l22_22873


namespace mary_money_left_l22_22112

theorem mary_money_left (p : ℝ) : 50 - (4 * p + 2 * p + 4 * p) = 50 - 10 * p := 
by 
  sorry

end mary_money_left_l22_22112


namespace average_weight_men_women_l22_22878

theorem average_weight_men_women (n_men n_women : ℕ) (avg_weight_men avg_weight_women : ℚ)
  (h_men : n_men = 8) (h_women : n_women = 6) (h_avg_weight_men : avg_weight_men = 190)
  (h_avg_weight_women : avg_weight_women = 120) :
  (n_men * avg_weight_men + n_women * avg_weight_women) / (n_men + n_women) = 160 := 
by
  sorry

end average_weight_men_women_l22_22878


namespace solve_inequality1_solve_inequality2_l22_22129

-- Problem 1: Solve the inequality (1)
theorem solve_inequality1 (x : ℝ) (h : x ≠ -4) : 
  (2 - x) / (x + 4) ≤ 0 ↔ (x ≥ 2 ∨ x < -4) := sorry

-- Problem 2: Solve the inequality (2) for different cases of a
theorem solve_inequality2 (x a : ℝ) : 
  (x^2 - 3 * a * x + 2 * a^2 ≥ 0) ↔
  (if a > 0 then (x ≥ 2 * a ∨ x ≤ a) 
   else if a < 0 then (x ≥ a ∨ x ≤ 2 * a) 
   else true) := sorry

end solve_inequality1_solve_inequality2_l22_22129


namespace range_of_a12_l22_22428

variable (a : ℕ → ℝ)
variable (a1 d : ℝ)

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

variable (h_arith_seq : arithmetic_seq a a1 d)
variable (h_a8 : a 7 ≥ 15)
variable (h_a9 : a 8 ≤ 13)

theorem range_of_a12 : ∀ a1 d, (arithmetic_seq a a1 d) → (a 7 ≥ 15) → (a 8 ≤ 13) → (a 11 ≤ 7) :=
by
  intro a1 d h_arith_seq h_a8 h_a9
  sorry

end range_of_a12_l22_22428


namespace train_speeds_l22_22724

-- Definitions used in conditions
def initial_distance : ℝ := 300
def time_elapsed : ℝ := 2
def remaining_distance : ℝ := 40
def speed_difference : ℝ := 10

-- Stating the problem in Lean
theorem train_speeds :
  ∃ (v_fast v_slow : ℝ),
    v_slow + speed_difference = v_fast ∧
    (2 * (v_slow + v_fast)) = (initial_distance - remaining_distance) ∧
    v_slow = 60 ∧
    v_fast = 70 :=
by
  sorry

end train_speeds_l22_22724


namespace angle_in_second_quadrant_l22_22817

theorem angle_in_second_quadrant (α : ℝ) (h₁ : -2 * Real.pi < α) (h₂ : α < -Real.pi) : 
  α = -4 → (α > -3 * Real.pi / 2 ∧ α < -Real.pi / 2) :=
by
  intros hα
  sorry

end angle_in_second_quadrant_l22_22817


namespace trapezium_height_l22_22382

theorem trapezium_height (a b A h : ℝ) (ha : a = 12) (hb : b = 16) (ha_area : A = 196) :
  (A = 0.5 * (a + b) * h) → h = 14 :=
by
  intros h_eq
  rw [ha, hb, ha_area] at h_eq
  sorry

end trapezium_height_l22_22382


namespace average_speed_l22_22486

def dist1 : ℝ := 60
def dist2 : ℝ := 30
def time : ℝ := 2

theorem average_speed : (dist1 + dist2) / time = 45 := by
  sorry

end average_speed_l22_22486


namespace product_of_values_l22_22407

theorem product_of_values (x : ℚ) (hx : abs ((18 / x) + 4) = 3) :
  x = -18 ∨ x = -18 / 7 ∧ -18 * (-18 / 7) = 324 / 7 :=
by sorry

end product_of_values_l22_22407


namespace quadratic_roots_l22_22093

theorem quadratic_roots (m : ℝ) (h1 : m > 4) :
  (∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2 * (m + 2) * x + m = 0 ∧ (m-5) * y^2 - 2 * (m + 2) * y + m = 0)
  ∨ (m = 5 ∧ ∃ x : ℝ, (m-5) * x^2 - 2 * (m + 2) * x + m = 0)
  ∨ (¬((∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2 * (m + 2) * x + m = 0) ∨ (m = 5 ∧ ∃ x : ℝ, (m-5) * x^2 - 2 * (m + 2) * x + m = 0))) :=
by
  sorry

end quadratic_roots_l22_22093


namespace find_max_value_l22_22584

noncomputable def max_value (x y z : ℝ) : ℝ := (x + y) / (x * y * z)

theorem find_max_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 2) :
  max_value x y z ≤ 13.5 :=
sorry

end find_max_value_l22_22584


namespace gcd_180_308_l22_22477

theorem gcd_180_308 : Nat.gcd 180 308 = 4 :=
by
  sorry

end gcd_180_308_l22_22477


namespace michael_clean_times_in_one_year_l22_22332

-- Definitions from the conditions
def baths_per_week : ℕ := 2
def showers_per_week : ℕ := 1
def weeks_per_year : ℕ := 52

-- Theorem statement for the proof problem
theorem michael_clean_times_in_one_year :
  (baths_per_week + showers_per_week) * weeks_per_year = 156 :=
by
  sorry

end michael_clean_times_in_one_year_l22_22332


namespace find_bc_l22_22974

theorem find_bc (A : ℝ) (a : ℝ) (area : ℝ) (b c : ℝ) :
  A = 60 * (π / 180) → a = Real.sqrt 7 → area = (3 * Real.sqrt 3) / 2 →
  ((b = 3 ∧ c = 2) ∨ (b = 2 ∧ c = 3)) :=
by
  intros hA ha harea
  -- From the given area condition, derive bc = 6
  have h1 : b * c = 6 := sorry
  -- From the given conditions, derive b + c = 5
  have h2 : b + c = 5 := sorry
  -- Solve the system of equations to find possible values for b and c
  -- Using x² - S⋅x + P = 0 where x are roots, S = b + c, P = b⋅c
  have h3 : (b = 3 ∧ c = 2) ∨ (b = 2 ∧ c = 3) := sorry
  exact h3

end find_bc_l22_22974


namespace kitten_current_length_l22_22513

theorem kitten_current_length (initial_length : ℕ) (double_after_2_weeks : ℕ → ℕ) (double_after_4_months : ℕ → ℕ)
  (h1 : initial_length = 4)
  (h2 : double_after_2_weeks initial_length = 2 * initial_length)
  (h3 : double_after_4_months (double_after_2_weeks initial_length) = 2 * (double_after_2_weeks initial_length)) :
  double_after_4_months (double_after_2_weeks initial_length) = 16 := 
by
  sorry

end kitten_current_length_l22_22513


namespace simplify_sqrt7_pow6_l22_22621

theorem simplify_sqrt7_pow6 : (real.sqrt 7) ^ 6 = 343 :=
by
  sorry

end simplify_sqrt7_pow6_l22_22621


namespace statement_A_statement_B_statement_C_statement_D_l22_22400

-- Definitions based on the problem conditions
def curve (m : ℝ) (x y : ℝ) : Prop :=
  x^4 + y^4 + m * x^2 * y^2 = 1

def is_symmetric_about_origin (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y ↔ curve m (-x) (-y)

def enclosed_area_eq_pi (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y → (x^2 + y^2)^2 = 1

def does_not_intersect_y_eq_x (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y ∧ x = y → false

def no_common_points_with_region (m : ℝ) : Prop :=
  ∀ x y : ℝ, |x| + |y| < 1 → ¬ curve m x y

-- Statements to prove based on correct answers
theorem statement_A (m : ℝ) : is_symmetric_about_origin m :=
  sorry

theorem statement_B (m : ℝ) (h : m = 2) : enclosed_area_eq_pi m :=
  sorry

theorem statement_C (m : ℝ) (h : m = -2) : ¬ does_not_intersect_y_eq_x m :=
  sorry

theorem statement_D (m : ℝ) (h : m = -1) : no_common_points_with_region m :=
  sorry

end statement_A_statement_B_statement_C_statement_D_l22_22400


namespace dropping_more_than_eating_l22_22432

theorem dropping_more_than_eating (n : ℕ) : n = 20 → (n * (n + 1)) / 2 > 10 * n := by
  intros h
  rw [h]
  sorry

end dropping_more_than_eating_l22_22432


namespace probability_of_perfect_square_sum_l22_22158

-- Define the conditions
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def perfect_squares_in_range : Finset ℕ := {4, 9, 16}
def total_outcomes : ℕ := 8 * 8  -- total number of outcomes when rolling two dice

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  (dice_faces.product dice_faces).filter (λ pair => perfect_squares_in_range.member (pair.1 + pair.2)).card

-- Calculate the probability of getting a perfect square sum
noncomputable def probability_perfect_square_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- The statement to prove that the probability is 3/16
theorem probability_of_perfect_square_sum :
  probability_perfect_square_sum = 3 / 16 :=
begin
  sorry
end

end probability_of_perfect_square_sum_l22_22158


namespace problem_1_problem_2_l22_22798

noncomputable def f (a b x : ℝ) := a * (x - 1)^2 + b * Real.log x

theorem problem_1 (a : ℝ) (h_deriv : ∀ x ≥ 2, (2 * a * x^2 - 2 * a * x + 1) / x ≤ 0) : 
  a ≤ -1 / 4 :=
sorry

theorem problem_2 (a : ℝ) (h_ineq : ∀ x ≥ 1, a * (x - 1)^2 + Real.log x ≤ x - 1) : 
  a ≤ 0 :=
sorry

end problem_1_problem_2_l22_22798


namespace arctan_sum_l22_22016

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l22_22016
