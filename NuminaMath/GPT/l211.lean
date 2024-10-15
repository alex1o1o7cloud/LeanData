import Mathlib

namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l211_21133

theorem isosceles_triangle_perimeter {a b : ℝ} (h1 : a = 6) (h2 : b = 3) (h3 : a ≠ b) :
  (2 * b + a = 15) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l211_21133


namespace NUMINAMATH_GPT_max_at_zero_l211_21186

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem max_at_zero : ∀ x : ℝ, f x ≤ f 0 :=
by
  sorry

end NUMINAMATH_GPT_max_at_zero_l211_21186


namespace NUMINAMATH_GPT_mean_value_of_quadrilateral_angles_l211_21198

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end NUMINAMATH_GPT_mean_value_of_quadrilateral_angles_l211_21198


namespace NUMINAMATH_GPT_multiplication_problem_l211_21139

-- Definitions for different digits A, B, C, D
def is_digit (n : ℕ) := n < 10

theorem multiplication_problem 
  (A B C D : ℕ) 
  (hA : is_digit A) 
  (hB : is_digit B) 
  (hC : is_digit C) 
  (hD : is_digit D) 
  (h_diff : ∀ x y : ℕ, x ≠ y → is_digit x → is_digit y → x ≠ A → y ≠ B → x ≠ C → y ≠ D)
  (hD1 : D = 1)
  (h_mult : A * D = A) 
  (hC_eq : C = A + B) :
  A + C = 5 := sorry

end NUMINAMATH_GPT_multiplication_problem_l211_21139


namespace NUMINAMATH_GPT_percentage_given_away_l211_21117

theorem percentage_given_away
  (initial_bottles : ℕ)
  (drank_percentage : ℝ)
  (remaining_percentage : ℝ)
  (gave_away : ℝ):
  initial_bottles = 3 →
  drank_percentage = 0.90 →
  remaining_percentage = 0.70 →
  gave_away = initial_bottles - (drank_percentage * 1 + remaining_percentage) →
  (gave_away / 2) / 1 * 100 = 70 :=
by
  intros
  sorry

end NUMINAMATH_GPT_percentage_given_away_l211_21117


namespace NUMINAMATH_GPT_value_of_m_l211_21132

theorem value_of_m 
  (m : ℤ) 
  (h : ∀ x : ℤ, x^2 - 2 * (m + 1) * x + 16 = (x - 4)^2) : 
  m = 3 := 
sorry

end NUMINAMATH_GPT_value_of_m_l211_21132


namespace NUMINAMATH_GPT_colorful_family_total_children_l211_21105

theorem colorful_family_total_children (x : ℕ) (b : ℕ) :
  -- Initial equal number of white, blue, and striped children
  -- After some blue children become striped
  -- Total number of blue and white children was 10,
  -- Total number of white and striped children was 18
  -- We need to prove the total number of children is 21
  (x = 5) →
  (x + x = 10) →
  (10 + b = 18) →
  (3*x = 21) :=
by
  intros h1 h2 h3
  -- x initially represents the number of white, blue, and striped children
  -- We know x is 5 and satisfy the conditions
  sorry

end NUMINAMATH_GPT_colorful_family_total_children_l211_21105


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l211_21102

variable (a_n : ℕ → ℝ)

theorem arithmetic_sequence_common_difference
  (h_arith : ∀ n, a_n (n + 1) = a_n n + d)
  (h_non_zero : d ≠ 0)
  (h_sum : a_n 1 + a_n 2 + a_n 3 = 9)
  (h_geom : a_n 2 ^ 2 = a_n 1 * a_n 5) :
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l211_21102


namespace NUMINAMATH_GPT_carnival_days_l211_21195

-- Define the given conditions
def total_money := 3168
def daily_income := 144

-- Define the main theorem statement
theorem carnival_days : (total_money / daily_income) = 22 := by
  sorry

end NUMINAMATH_GPT_carnival_days_l211_21195


namespace NUMINAMATH_GPT_final_volume_of_water_in_tank_l211_21155

theorem final_volume_of_water_in_tank (capacity : ℕ) (initial_fraction full_volume : ℕ)
  (percent_empty percent_fill final_volume : ℕ) :
  capacity = 8000 →
  initial_fraction = 3 / 4 →
  percent_empty = 40 →
  percent_fill = 30 →
  full_volume = capacity * initial_fraction →
  final_volume = full_volume - (full_volume * percent_empty / 100) + ((full_volume - (full_volume * percent_empty / 100)) * percent_fill / 100) →
  final_volume = 4680 :=
by
  sorry

end NUMINAMATH_GPT_final_volume_of_water_in_tank_l211_21155


namespace NUMINAMATH_GPT_apples_in_pile_l211_21146

-- Define the initial number of apples in the pile
def initial_apples : ℕ := 8

-- Define the number of added apples
def added_apples : ℕ := 5

-- Define the total number of apples
def total_apples : ℕ := initial_apples + added_apples

-- Prove that the total number of apples is 13
theorem apples_in_pile : total_apples = 13 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_pile_l211_21146


namespace NUMINAMATH_GPT_height_of_water_in_cylinder_l211_21131

theorem height_of_water_in_cylinder
  (r_cone : ℝ) (h_cone : ℝ) (r_cylinder : ℝ) (V_cone : ℝ) (V_cylinder : ℝ) (h_cylinder : ℝ) :
  r_cone = 15 → h_cone = 25 → r_cylinder = 20 →
  V_cone = (1 / 3) * π * r_cone^2 * h_cone →
  V_cylinder = V_cone → V_cylinder = π * r_cylinder^2 * h_cylinder →
  h_cylinder = 4.7 :=
by
  intros r_cone_eq h_cone_eq r_cylinder_eq V_cone_eq V_cylinder_eq volume_eq
  sorry

end NUMINAMATH_GPT_height_of_water_in_cylinder_l211_21131


namespace NUMINAMATH_GPT_distribute_books_l211_21151

theorem distribute_books (m n : ℕ) (h1 : m = 3*n + 8) (h2 : ∃k, m = 5*k + r ∧ r < 5 ∧ r > 0) : 
  n = 5 ∨ n = 6 :=
by sorry

end NUMINAMATH_GPT_distribute_books_l211_21151


namespace NUMINAMATH_GPT_base4_division_l211_21154

/-- Given in base 4:
2023_4 div 13_4 = 155_4
We need to prove the quotient is equal to 155_4.
-/
theorem base4_division (n m q r : ℕ) (h1 : n = 2 * 4^3 + 0 * 4^2 + 2 * 4^1 + 3 * 4^0)
    (h2 : m = 1 * 4^1 + 3 * 4^0)
    (h3 : q = 1 * 4^2 + 5 * 4^1 + 5 * 4^0)
    (h4 : n = m * q + r)
    (h5 : 0 ≤ r ∧ r < m):
  q = 1 * 4^2 + 5 * 4^1 + 5 * 4^0 := 
by
  sorry

end NUMINAMATH_GPT_base4_division_l211_21154


namespace NUMINAMATH_GPT_term_5th_in_sequence_l211_21116

theorem term_5th_in_sequence : 
  ∃ n : ℕ, n = 5 ∧ ( ∃ t : ℕ, t = 28 ∧ 3^t ∈ { 3^(7 * (k - 1)) | k : ℕ } ) :=
by {
  sorry
}

end NUMINAMATH_GPT_term_5th_in_sequence_l211_21116


namespace NUMINAMATH_GPT_find_f_neg_a_l211_21194

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.tan x + 1

theorem find_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_a_l211_21194


namespace NUMINAMATH_GPT_ellipse_distance_CD_l211_21110

theorem ellipse_distance_CD :
  ∃ (CD : ℝ), 
    (∀ (x y : ℝ),
    4 * (x - 2)^2 + 16 * y^2 = 64) → 
      CD = 2*Real.sqrt 5 :=
by sorry

end NUMINAMATH_GPT_ellipse_distance_CD_l211_21110


namespace NUMINAMATH_GPT_expected_winnings_l211_21101

def probability_heads : ℚ := 1 / 3
def probability_tails : ℚ := 1 / 2
def probability_edge : ℚ := 1 / 6

def winning_heads : ℚ := 2
def winning_tails : ℚ := 2
def losing_edge : ℚ := -4

def expected_value : ℚ := probability_heads * winning_heads + probability_tails * winning_tails + probability_edge * losing_edge

theorem expected_winnings : expected_value = 1 := by
  sorry

end NUMINAMATH_GPT_expected_winnings_l211_21101


namespace NUMINAMATH_GPT_natural_number_40_times_smaller_l211_21141

-- Define the sum of the first (n-1) natural numbers
def sum_natural_numbers (n : ℕ) := (n * (n - 1)) / 2

-- Define the proof statement
theorem natural_number_40_times_smaller (n : ℕ) (h : sum_natural_numbers n = 40 * n) : n = 81 :=
by {
  -- The proof is left as an exercise
  sorry
}

end NUMINAMATH_GPT_natural_number_40_times_smaller_l211_21141


namespace NUMINAMATH_GPT_find_integers_l211_21153

theorem find_integers (a b : ℤ) (h1 : a * b = a + b) (h2 : a * b = a - b) : a = 0 ∧ b = 0 :=
by 
  sorry

end NUMINAMATH_GPT_find_integers_l211_21153


namespace NUMINAMATH_GPT_total_books_after_donations_l211_21107

variable (Boris_books : Nat := 24)
variable (Cameron_books : Nat := 30)

theorem total_books_after_donations :
  (Boris_books - Boris_books / 4) + (Cameron_books - Cameron_books / 3) = 38 := by
  sorry

end NUMINAMATH_GPT_total_books_after_donations_l211_21107


namespace NUMINAMATH_GPT_boxes_to_fill_l211_21172

theorem boxes_to_fill (total_boxes filled_boxes : ℝ) (h₁ : total_boxes = 25.75) (h₂ : filled_boxes = 17.5) : 
  total_boxes - filled_boxes = 8.25 := 
by
  sorry

end NUMINAMATH_GPT_boxes_to_fill_l211_21172


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l211_21142

theorem repeating_decimal_to_fraction :
  let x := 0.431431431 + 0.000431431431 + 0.000000431431431
  let y := 0.4 + x
  y = 427 / 990 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l211_21142


namespace NUMINAMATH_GPT_routes_from_A_to_B_in_4_by_3_grid_l211_21109

-- Problem: Given a 4 by 3 rectangular grid, and movement allowing only right (R) or down (D),
-- prove that the number of different routes from point A to point B is 35.
def routes_4_by_3 : ℕ :=
  let n_moves := 3 + 4  -- Total moves required are 3 Rs and 4 Ds
  let r_moves := 3      -- Number of Right moves (R)
  Nat.choose (n_moves) (r_moves) -- Number of ways to choose 3 Rs from 7 moves

theorem routes_from_A_to_B_in_4_by_3_grid : routes_4_by_3 = 35 := by {
  sorry -- Proof omitted
}

end NUMINAMATH_GPT_routes_from_A_to_B_in_4_by_3_grid_l211_21109


namespace NUMINAMATH_GPT_john_new_bench_press_l211_21183

theorem john_new_bench_press (initial_weight : ℕ) (decrease_percent : ℕ) (retain_percent : ℕ) (training_factor : ℕ) (final_weight : ℕ) 
  (h1 : initial_weight = 500)
  (h2 : decrease_percent = 80)
  (h3 : retain_percent = 20)
  (h4 : training_factor = 3)
  (h5 : final_weight = initial_weight * retain_percent / 100 * training_factor) : 
  final_weight = 300 := 
by sorry

end NUMINAMATH_GPT_john_new_bench_press_l211_21183


namespace NUMINAMATH_GPT_four_consecutive_none_multiple_of_5_l211_21135

theorem four_consecutive_none_multiple_of_5 (n : ℤ) :
  (∃ k : ℤ, n + (n + 1) + (n + 2) + (n + 3) = 5 * k) →
  ¬ (∃ m : ℤ, (n = 5 * m) ∨ (n + 1 = 5 * m) ∨ (n + 2 = 5 * m) ∨ (n + 3 = 5 * m)) :=
by sorry

end NUMINAMATH_GPT_four_consecutive_none_multiple_of_5_l211_21135


namespace NUMINAMATH_GPT_trapezoid_total_area_l211_21191

/-- 
Given a trapezoid with side lengths 4, 6, 8, and 10, where sides 4 and 8 are used as parallel bases, 
prove that the total area of the trapezoid in all possible configurations is 48√2.
-/
theorem trapezoid_total_area : 
  let a := 4
  let b := 8
  let c := 6
  let d := 10
  let h := 4 * Real.sqrt 2
  let Area := (1 / 2) * (a + b) * h
  (Area + Area) = 48 * Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_trapezoid_total_area_l211_21191


namespace NUMINAMATH_GPT_new_person_weight_l211_21130

theorem new_person_weight (W : ℝ) (N : ℝ)
  (h1 : ∀ avg_increase : ℝ, avg_increase = 2.5 → N = 55) 
  (h2 : ∀ original_weight : ℝ, original_weight = 35) 
  : N = 55 := 
by 
  sorry

end NUMINAMATH_GPT_new_person_weight_l211_21130


namespace NUMINAMATH_GPT_contractor_daily_amount_l211_21148

theorem contractor_daily_amount
  (days_worked : ℕ) (total_days : ℕ) (fine_per_absent_day : ℝ)
  (total_amount : ℝ) (days_absent : ℕ) (amount_received : ℝ) :
  days_worked = total_days - days_absent →
  (total_amount = (days_worked * amount_received - days_absent * fine_per_absent_day)) →
  total_days = 30 →
  fine_per_absent_day = 7.50 →
  total_amount = 685 →
  days_absent = 2 →
  amount_received = 25 :=
by
  sorry

end NUMINAMATH_GPT_contractor_daily_amount_l211_21148


namespace NUMINAMATH_GPT_tree_count_in_yard_l211_21185

-- Definitions from conditions
def yard_length : ℕ := 350
def tree_distance : ℕ := 14

-- Statement of the theorem
theorem tree_count_in_yard : (yard_length / tree_distance) + 1 = 26 := by
  sorry

end NUMINAMATH_GPT_tree_count_in_yard_l211_21185


namespace NUMINAMATH_GPT_average_time_per_other_class_l211_21167

theorem average_time_per_other_class (school_hours : ℚ) (num_classes : ℕ) (hist_chem_hours : ℚ)
  (total_school_time_minutes : ℕ) (hist_chem_time_minutes : ℕ) (num_other_classes : ℕ)
  (other_classes_time_minutes : ℕ) (average_time_other_classes : ℕ) :
  school_hours = 7.5 →
  num_classes = 7 →
  hist_chem_hours = 1.5 →
  total_school_time_minutes = school_hours * 60 →
  hist_chem_time_minutes = hist_chem_hours * 60 →
  other_classes_time_minutes = total_school_time_minutes - hist_chem_time_minutes →
  num_other_classes = num_classes - 2 →
  average_time_other_classes = other_classes_time_minutes / num_other_classes →
  average_time_other_classes = 72 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_time_per_other_class_l211_21167


namespace NUMINAMATH_GPT_particle_max_height_l211_21150

noncomputable def max_height (r ω g : ℝ) : ℝ :=
  (r * ω + g / ω) ^ 2 / (2 * g)

theorem particle_max_height (r ω g : ℝ) (h : ω > Real.sqrt (g / r)) :
    max_height r ω g = (r * ω + g / ω) ^ 2 / (2 * g) :=
sorry

end NUMINAMATH_GPT_particle_max_height_l211_21150


namespace NUMINAMATH_GPT_minimum_value_proof_l211_21174

noncomputable def minimum_value (a b c : ℝ) (h : a + b + c = 6) : ℝ :=
  9 / a + 4 / b + 1 / c

theorem minimum_value_proof (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 6) :
  (minimum_value a b c h₃) = 6 :=
sorry

end NUMINAMATH_GPT_minimum_value_proof_l211_21174


namespace NUMINAMATH_GPT_other_root_of_quadratic_l211_21152

theorem other_root_of_quadratic (m : ℝ) (h : (2:ℝ) * (t:ℝ) = -6 ): 
  ∃ t, t = -3 :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l211_21152


namespace NUMINAMATH_GPT_relation_1_relation_2_relation_3_general_relationship_l211_21158

theorem relation_1 (a b : ℝ) (h1: a = 3) (h2: b = 3) : a^2 + b^2 = 2 * a * b :=
by 
  have h : a = 3 := h1
  have h' : b = 3 := h2
  sorry

theorem relation_2 (a b : ℝ) (h1: a = 2) (h2: b = 1/2) : a^2 + b^2 > 2 * a * b :=
by 
  have h : a = 2 := h1
  have h' : b = 1/2 := h2
  sorry

theorem relation_3 (a b : ℝ) (h1: a = -2) (h2: b = 3) : a^2 + b^2 > 2 * a * b :=
by 
  have h : a = -2 := h1
  have h' : b = 3 := h2
  sorry

theorem general_relationship (a b : ℝ) : a^2 + b^2 ≥ 2 * a * b :=
by
  sorry

end NUMINAMATH_GPT_relation_1_relation_2_relation_3_general_relationship_l211_21158


namespace NUMINAMATH_GPT_relationship_between_a_b_l211_21163

theorem relationship_between_a_b (a b c : ℝ) (x y : ℝ) (h1 : x = -3) (h2 : y = -2)
  (h3 : a * x + c * y = 1) (h4 : c * x - b * y = 2) : 9 * a + 4 * b = 1 :=
sorry

end NUMINAMATH_GPT_relationship_between_a_b_l211_21163


namespace NUMINAMATH_GPT_find_k_value_l211_21169

theorem find_k_value :
  ∃ k : ℝ, (∀ x : ℝ, 1 ≤ x^2 - 3 * x + k ∧ x^2 - 3 * x + k ≤ 5) ∧ 
          (∃ a b : ℝ, b - a = 8 ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ b → 1 ≤ x^2 - 3 * x + k ∧ x^2 - 3 * x + k ≤ 5)) ∧ 
          k = 9 / 4 :=
sorry

end NUMINAMATH_GPT_find_k_value_l211_21169


namespace NUMINAMATH_GPT_last_two_digits_of_9_power_h_are_21_l211_21124

def a := 1
def b := 2^a
def c := 3^b
def d := 4^c
def e := 5^d
def f := 6^e
def g := 7^f
def h := 8^g

theorem last_two_digits_of_9_power_h_are_21 : (9^h) % 100 = 21 := by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_9_power_h_are_21_l211_21124


namespace NUMINAMATH_GPT_trader_profit_percentage_l211_21149

theorem trader_profit_percentage
  (P : ℝ)
  (h1 : P > 0)
  (buy_price : ℝ := 0.80 * P)
  (sell_price : ℝ := 1.60 * P) :
  (sell_price - P) / P * 100 = 60 := 
by sorry

end NUMINAMATH_GPT_trader_profit_percentage_l211_21149


namespace NUMINAMATH_GPT_gcd_m_n_l211_21173

def m := 122^2 + 234^2 + 345^2 + 10
def n := 123^2 + 233^2 + 347^2 + 10

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_m_n_l211_21173


namespace NUMINAMATH_GPT_geometric_mean_a_b_l211_21190

theorem geometric_mean_a_b : ∀ (a b : ℝ), a > 0 → b > 0 → Real.sqrt 3 = Real.sqrt (3^a * 3^b) → a + b = 1 :=
by
  intros a b ha hb hgeo
  sorry

end NUMINAMATH_GPT_geometric_mean_a_b_l211_21190


namespace NUMINAMATH_GPT_prove_Praveen_present_age_l211_21147

-- Definitions based on the conditions identified in a)
def PraveenAge (P : ℝ) := P + 10 = 3 * (P - 3)

-- The equivalent proof problem statement
theorem prove_Praveen_present_age : ∃ P : ℝ, PraveenAge P ∧ P = 9.5 :=
by
  sorry

end NUMINAMATH_GPT_prove_Praveen_present_age_l211_21147


namespace NUMINAMATH_GPT_number_of_red_balls_l211_21188

theorem number_of_red_balls (W R T : ℕ) (hW : W = 12) (h_freq : (R : ℝ) / (T : ℝ) = 0.25) (hT : T = W + R) : R = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_red_balls_l211_21188


namespace NUMINAMATH_GPT_math_olympiad_problem_l211_21144

theorem math_olympiad_problem (students : Fin 11 → Finset (Fin n)) (h_solved : ∀ i, (students i).card = 3)
  (h_distinct : ∀ i j, i ≠ j → ∃ p, p ∈ students i ∧ p ∉ students j) : 
  6 ≤ n := 
sorry

end NUMINAMATH_GPT_math_olympiad_problem_l211_21144


namespace NUMINAMATH_GPT_car_maintenance_fraction_l211_21170

variable (p : ℝ) (f : ℝ)

theorem car_maintenance_fraction (hp : p = 5200)
  (he : p - f * p - (p - 320) = 200) : f = 3 / 130 :=
by
  have hp_pos : p ≠ 0 := by linarith [hp]
  sorry

end NUMINAMATH_GPT_car_maintenance_fraction_l211_21170


namespace NUMINAMATH_GPT_factorial_expression_evaluation_l211_21114

theorem factorial_expression_evaluation : (Real.sqrt ((Nat.factorial 5 * Nat.factorial 4) / Nat.factorial 2))^2 = 1440 :=
by
  sorry

end NUMINAMATH_GPT_factorial_expression_evaluation_l211_21114


namespace NUMINAMATH_GPT_positive_rational_representation_l211_21187

theorem positive_rational_representation (q : ℚ) (h_pos_q : 0 < q) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ q = (a^2021 + b^2023) / (c^2022 + d^2024) :=
by
  sorry

end NUMINAMATH_GPT_positive_rational_representation_l211_21187


namespace NUMINAMATH_GPT_system_inequalities_1_l211_21108

theorem system_inequalities_1 (x : ℝ) (h1 : 2 * x ≥ x - 1) (h2 : 4 * x + 10 > x + 1) :
  x ≥ -1 :=
sorry

end NUMINAMATH_GPT_system_inequalities_1_l211_21108


namespace NUMINAMATH_GPT_part1_part2_l211_21119

-- Definitions for part 1
def total_souvenirs := 60
def price_a := 100
def price_b := 60
def total_cost_1 := 4600

-- Definitions for part 2
def max_total_cost := 4500
def twice (m : ℕ) := 2 * m

theorem part1 (x y : ℕ) (hx : x + y = total_souvenirs) (hc : price_a * x + price_b * y = total_cost_1) :
  x = 25 ∧ y = 35 :=
by
  -- You can provide the detailed proof here
  sorry

theorem part2 (m : ℕ) (hm1 : 20 ≤ m) (hm2 : m ≤ 22) (hc2 : price_a * m + price_b * (total_souvenirs - m) ≤ max_total_cost) :
  (m = 20 ∨ m = 21 ∨ m = 22) ∧ 
  ∃ W, W = min (40 * 20 + 3600) (min (40 * 21 + 3600) (40 * 22 + 3600)) ∧ W = 4400 :=
by
  -- You can provide the detailed proof here
  sorry

end NUMINAMATH_GPT_part1_part2_l211_21119


namespace NUMINAMATH_GPT_no_integer_solution_for_system_l211_21196

theorem no_integer_solution_for_system :
  ¬ ∃ (a b c d : ℤ), 
    (a * b * c * d - a = 1961) ∧ 
    (a * b * c * d - b = 961) ∧ 
    (a * b * c * d - c = 61) ∧ 
    (a * b * c * d - d = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_integer_solution_for_system_l211_21196


namespace NUMINAMATH_GPT_num_ordered_pairs_l211_21165

theorem num_ordered_pairs :
  ∃ (m n : ℤ), (m * n ≥ 0) ∧ (m^3 + n^3 + 99 * m * n = 33^3) ∧ (35 = 35) :=
by
  sorry

end NUMINAMATH_GPT_num_ordered_pairs_l211_21165


namespace NUMINAMATH_GPT_remainder_when_divided_by_15_l211_21192

theorem remainder_when_divided_by_15 (N : ℤ) (k : ℤ) 
  (h : N = 45 * k + 31) : (N % 15) = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_15_l211_21192


namespace NUMINAMATH_GPT_brittany_age_after_vacation_l211_21171

-- Definitions of the conditions
def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_years : ℕ := 4

-- Prove the main statement
theorem brittany_age_after_vacation : rebecca_age + age_difference + vacation_years = 32 := by
  sorry

end NUMINAMATH_GPT_brittany_age_after_vacation_l211_21171


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_l211_21118

noncomputable def convertToRectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular :
  let r := 10
  let θ := Real.pi / 3
  let z := 2
  let r' := 2 * r
  let z' := z + 1
  convertToRectangular r' θ z' = (10, 10 * Real.sqrt 3, 3) :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_l211_21118


namespace NUMINAMATH_GPT_fraction_identity_l211_21182

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := 2 * x - 3

theorem fraction_identity : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by
  sorry

end NUMINAMATH_GPT_fraction_identity_l211_21182


namespace NUMINAMATH_GPT_cubic_transform_l211_21181

theorem cubic_transform (A B C x z β : ℝ) (h₁ : z = x + β) (h₂ : 3 * β + A = 0) :
  z^3 + A * z^2 + B * z + C = 0 ↔ x^3 + (B - (A^2 / 3)) * x + (C - A * B / 3 + 2 * A^3 / 27) = 0 :=
sorry

end NUMINAMATH_GPT_cubic_transform_l211_21181


namespace NUMINAMATH_GPT_sequence_first_five_terms_l211_21145

noncomputable def a_n (n : ℕ) : ℤ := (-1) ^ n + (n : ℤ)

theorem sequence_first_five_terms :
  a_n 1 = 0 ∧
  a_n 2 = 3 ∧
  a_n 3 = 2 ∧
  a_n 4 = 5 ∧
  a_n 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_sequence_first_five_terms_l211_21145


namespace NUMINAMATH_GPT_polar_to_line_distance_l211_21134

theorem polar_to_line_distance : 
  let point_polar := (2, Real.pi / 3)
  let line_polar := (2, 0)  -- Corresponding (rho, theta) for the given line
  let point_rect := (2 * Real.cos (Real.pi / 3), 2 * Real.sin (Real.pi / 3))
  let line_rect := 2  -- x = 2 in rectangular coordinates
  let distance := abs (line_rect - point_rect.1)
  distance = 1 := by
{
  sorry
}

end NUMINAMATH_GPT_polar_to_line_distance_l211_21134


namespace NUMINAMATH_GPT_ratio_steel_iron_is_5_to_2_l211_21184

-- Definitions based on the given conditions
def amount_steel : ℕ := 35
def amount_iron : ℕ := 14

-- Main statement
theorem ratio_steel_iron_is_5_to_2 :
  (amount_steel / Nat.gcd amount_steel amount_iron) = 5 ∧
  (amount_iron / Nat.gcd amount_steel amount_iron) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_steel_iron_is_5_to_2_l211_21184


namespace NUMINAMATH_GPT_fuel_efficiency_problem_l211_21113

theorem fuel_efficiency_problem :
  let F_highway := 30
  let F_urban := 25
  let F_hill := 20
  let D_highway := 100
  let D_urban := 60
  let D_hill := 40
  let gallons_highway := D_highway / F_highway
  let gallons_urban := D_urban / F_urban
  let gallons_hill := D_hill / F_hill
  let total_gallons := gallons_highway + gallons_urban + gallons_hill
  total_gallons = 7.73 := 
by 
  sorry

end NUMINAMATH_GPT_fuel_efficiency_problem_l211_21113


namespace NUMINAMATH_GPT_distinct_sets_count_l211_21103

noncomputable def num_distinct_sets : ℕ :=
  let product : ℕ := 11 * 21 * 31 * 41 * 51 * 61
  728

theorem distinct_sets_count : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 11 * 21 * 31 * 41 * 51 * 61 ∧ num_distinct_sets = 728 :=
sorry

end NUMINAMATH_GPT_distinct_sets_count_l211_21103


namespace NUMINAMATH_GPT_dogs_in_academy_l211_21112

noncomputable def numberOfDogs : ℕ :=
  let allSit := 60
  let allStay := 35
  let allFetch := 40
  let allRollOver := 45
  let sitStay := 20
  let sitFetch := 15
  let sitRollOver := 18
  let stayFetch := 10
  let stayRollOver := 13
  let fetchRollOver := 12
  let sitStayFetch := 11
  let sitStayFetchRoll := 8
  let none := 15
  118 -- final count of dogs in the academy

theorem dogs_in_academy : numberOfDogs = 118 :=
by
  sorry

end NUMINAMATH_GPT_dogs_in_academy_l211_21112


namespace NUMINAMATH_GPT_find_ratio_l211_21166

variable {d : ℕ}
variable {a : ℕ → ℝ}

-- Conditions: arithmetic sequence with non-zero common difference, and geometric sequence terms
axiom arithmetic_sequence (n : ℕ) : a n = a 1 + (n - 1) * d
axiom non_zero_d : d ≠ 0
axiom geometric_sequence : (a 1 + 2*d)^2 = a 1 * (a 1 + 8*d)

-- Theorem to prove the desired ratio
theorem find_ratio : (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
sorry

end NUMINAMATH_GPT_find_ratio_l211_21166


namespace NUMINAMATH_GPT_smallest_square_condition_l211_21120

-- Definition of the conditions
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_last_digit_not_zero (n : ℕ) : Prop := n % 10 ≠ 0

def remove_last_two_digits (n : ℕ) : ℕ :=
  n / 100

-- The statement of the theorem we need to prove
theorem smallest_square_condition : 
  ∃ n : ℕ, is_square n ∧ has_last_digit_not_zero n ∧ is_square (remove_last_two_digits n) ∧ 121 ≤ n :=
sorry

end NUMINAMATH_GPT_smallest_square_condition_l211_21120


namespace NUMINAMATH_GPT_pool_water_after_eight_hours_l211_21136

-- Define the conditions
def hour1_fill_rate := 8
def hour2_and_hour3_fill_rate := 10
def hour4_and_hour5_fill_rate := 14
def hour6_fill_rate := 12
def hour7_fill_rate := 12
def hour8_fill_rate := 12
def hour7_leak := -8
def hour8_leak := -5

-- Calculate the water added in each time period
def water_added := hour1_fill_rate +
                   (hour2_and_hour3_fill_rate * 2) +
                   (hour4_and_hour5_fill_rate * 2) +
                   (hour6_fill_rate + hour7_fill_rate + hour8_fill_rate)

-- Calculate the water lost due to leaks
def water_lost := hour7_leak + hour8_leak  -- Note: Leaks are already negative

-- The final calculation: total water added minus total water lost
def final_water := water_added + water_lost

theorem pool_water_after_eight_hours : final_water = 79 :=
by {
  -- proof steps to check equality are omitted here
  sorry
}

end NUMINAMATH_GPT_pool_water_after_eight_hours_l211_21136


namespace NUMINAMATH_GPT_kenneth_fabric_amount_l211_21199

theorem kenneth_fabric_amount :
  ∃ K : ℤ, (∃ N : ℤ, N = 6 * K ∧ (K * 40 + 140000 = N * 40) ∧ K > 0) ∧ K = 700 :=
by
  sorry

end NUMINAMATH_GPT_kenneth_fabric_amount_l211_21199


namespace NUMINAMATH_GPT_cubic_polynomial_sum_l211_21164

noncomputable def Q (a b c m x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2 * m

theorem cubic_polynomial_sum (a b c m : ℝ) :
  Q a b c m 0 = 2 * m ∧ Q a b c m 1 = 3 * m ∧ Q a b c m (-1) = 5 * m →
  Q a b c m 2 + Q a b c m (-2) = 20 * m :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cubic_polynomial_sum_l211_21164


namespace NUMINAMATH_GPT_expressions_divisible_by_17_l211_21189

theorem expressions_divisible_by_17 (a b : ℤ) : 
  let x := 3 * b - 5 * a
  let y := 9 * a - 2 * b
  (∃ k : ℤ, (2 * x + 3 * y) = 17 * k) ∧ (∃ k : ℤ, (9 * x + 5 * y) = 17 * k) :=
by
  exact ⟨⟨a, by sorry⟩, ⟨b, by sorry⟩⟩

end NUMINAMATH_GPT_expressions_divisible_by_17_l211_21189


namespace NUMINAMATH_GPT_base_k_addition_is_ten_l211_21123

theorem base_k_addition_is_ten :
  ∃ k : ℕ, (k > 4) ∧ (5 * k^3 + 3 * k^2 + 4 * k + 2 + 6 * k^3 + 4 * k^2 + 2 * k + 1 = 1 * k^4 + 4 * k^3 + 1 * k^2 + 6 * k + 3) ∧ k = 10 :=
by
  sorry

end NUMINAMATH_GPT_base_k_addition_is_ten_l211_21123


namespace NUMINAMATH_GPT_mass_percentage_O_correct_l211_21177

noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_B : ℝ := 10.81
noncomputable def molar_mass_O : ℝ := 16.00

noncomputable def molar_mass_H3BO3 : ℝ := (3 * molar_mass_H) + (1 * molar_mass_B) + (3 * molar_mass_O)

noncomputable def mass_percentage_O_in_H3BO3 : ℝ := ((3 * molar_mass_O) / molar_mass_H3BO3) * 100

theorem mass_percentage_O_correct : abs (mass_percentage_O_in_H3BO3 - 77.59) < 0.01 := 
sorry

end NUMINAMATH_GPT_mass_percentage_O_correct_l211_21177


namespace NUMINAMATH_GPT_price_reduction_l211_21138

theorem price_reduction (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : 150 * (1 - x) * (1 - x) = 96 :=
sorry

end NUMINAMATH_GPT_price_reduction_l211_21138


namespace NUMINAMATH_GPT_sum_after_operations_l211_21162

theorem sum_after_operations (a b S : ℝ) (h : a + b = S) : 
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := 
by 
  sorry

end NUMINAMATH_GPT_sum_after_operations_l211_21162


namespace NUMINAMATH_GPT_calories_per_candy_bar_l211_21179

theorem calories_per_candy_bar (total_calories : ℕ) (number_of_bars : ℕ) 
  (h : total_calories = 341) (n : number_of_bars = 11) : (total_calories / number_of_bars = 31) :=
by
  sorry

end NUMINAMATH_GPT_calories_per_candy_bar_l211_21179


namespace NUMINAMATH_GPT_equation_has_seven_real_solutions_l211_21161

def f (x : ℝ) : ℝ := abs (x^2 - 1) - 1

theorem equation_has_seven_real_solutions (b c : ℝ) : 
  (c ≤ 0 ∧ 0 < b ∧ b < 1) ↔ 
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ), 
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₁ ≠ x₆ ∧ x₁ ≠ x₇ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₂ ≠ x₆ ∧ x₂ ≠ x₇ ∧
  x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₃ ≠ x₆ ∧ x₃ ≠ x₇ ∧
  x₄ ≠ x₅ ∧ x₄ ≠ x₆ ∧ x₄ ≠ x₇ ∧
  x₅ ≠ x₆ ∧ x₅ ≠ x₇ ∧
  x₆ ≠ x₇ ∧
  f x₁ ^ 2 - b * f x₁ + c = 0 ∧ f x₂ ^ 2 - b * f x₂ + c = 0 ∧
  f x₃ ^ 2 - b * f x₃ + c = 0 ∧ f x₄ ^ 2 - b * f x₄ + c = 0 ∧
  f x₅ ^ 2 - b * f x₅ + c = 0 ∧ f x₆ ^ 2 - b * f x₆ + c = 0 ∧
  f x₇ ^ 2 - b * f x₇ + c = 0 :=
sorry

end NUMINAMATH_GPT_equation_has_seven_real_solutions_l211_21161


namespace NUMINAMATH_GPT_cost_price_equals_selling_price_l211_21157

theorem cost_price_equals_selling_price (C : ℝ) (x : ℝ) (h1 : 20 * C = 1.25 * C * x) : x = 16 :=
by
  -- This proof is omitted at the moment
  sorry

end NUMINAMATH_GPT_cost_price_equals_selling_price_l211_21157


namespace NUMINAMATH_GPT_shopkeeper_total_profit_percentage_l211_21115

noncomputable def profit_percentage (actual_weight faulty_weight ratio : ℕ) : ℝ :=
  (actual_weight - faulty_weight) / actual_weight * 100 * ratio

noncomputable def total_profit_percentage (ratios profits : List ℝ) : ℝ :=
  (List.sum (List.zipWith (· * ·) ratios profits)) / (List.sum ratios)

theorem shopkeeper_total_profit_percentage :
  let actual_weight := 1000
  let faulty_weights := [900, 850, 950]
  let profit_percentages := [10, 15, 5]
  let ratios := [3, 2, 1]
  total_profit_percentage ratios profit_percentages = 10.83 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_total_profit_percentage_l211_21115


namespace NUMINAMATH_GPT_find_PS_eq_13point625_l211_21178

theorem find_PS_eq_13point625 (PQ PR QR : ℝ) (h : ℝ) (QS SR : ℝ)
  (h_QS : QS^2 = 225 - h^2)
  (h_SR : SR^2 = 400 - h^2)
  (h_ratio : QS / SR = 3 / 7) :
  PS = 13.625 :=
by
  sorry

end NUMINAMATH_GPT_find_PS_eq_13point625_l211_21178


namespace NUMINAMATH_GPT_heartsuit_ratio_l211_21143

-- Define the operation ⧡
def heartsuit (n m : ℕ) := n^(3+m) * m^(2+n)

-- The problem statement to prove
theorem heartsuit_ratio : heartsuit 2 4 / heartsuit 4 2 = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_heartsuit_ratio_l211_21143


namespace NUMINAMATH_GPT_circle_equation_l211_21126

theorem circle_equation :
  ∃ M : ℝ × ℝ, (2 * M.1 + M.2 - 1 = 0) ∧
    (∃ r : ℝ, r ≥ 0 ∧ 
      ((3 - M.1)^2 + (0 - M.2)^2 = r^2) ∧
      ((0 - M.1)^2 + (1 - M.2)^2 = r^2)) ∧
    (∃ x y : ℝ, ((x - 1)^2 + (y + 1)^2 = 5)) := 
sorry

end NUMINAMATH_GPT_circle_equation_l211_21126


namespace NUMINAMATH_GPT_inequality_no_real_solutions_l211_21111

theorem inequality_no_real_solutions (a b : ℝ) 
  (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) : 
  |b| ≤ 1 :=
sorry

end NUMINAMATH_GPT_inequality_no_real_solutions_l211_21111


namespace NUMINAMATH_GPT_distinct_configurations_l211_21180

/-- 
Define m, n, and the binomial coefficient function.
conditions:
  - integer grid dimensions m and n with m >= 1, n >= 1.
  - initially (m-1)(n-1) coins in the subgrid of size (m-1) x (n-1).
  - legal move conditions for coins.
question:
  - Prove the number of distinct configurations of coins equals the binomial coefficient.
-/
def number_of_distinct_configurations (m n : ℕ) : ℕ :=
  Nat.choose (m + n - 2) (m - 1)

theorem distinct_configurations (m n : ℕ) (h_m : 1 ≤ m) (h_n : 1 ≤ n) :
  number_of_distinct_configurations m n = Nat.choose (m + n - 2) (m - 1) :=
sorry

end NUMINAMATH_GPT_distinct_configurations_l211_21180


namespace NUMINAMATH_GPT_expected_steps_unit_interval_l211_21127

noncomputable def expected_steps_to_color_interval : ℝ := 
  -- Placeholder for the function calculating expected steps
  sorry 

theorem expected_steps_unit_interval : expected_steps_to_color_interval = 5 :=
  sorry

end NUMINAMATH_GPT_expected_steps_unit_interval_l211_21127


namespace NUMINAMATH_GPT_abby_potatoes_peeled_l211_21121

theorem abby_potatoes_peeled (total_potatoes : ℕ) (homers_rate : ℕ) (abbys_rate : ℕ) (time_alone : ℕ) (potatoes_peeled : ℕ) :
  (total_potatoes = 60) →
  (homers_rate = 4) →
  (abbys_rate = 6) →
  (time_alone = 6) →
  (potatoes_peeled = 22) :=
  sorry

end NUMINAMATH_GPT_abby_potatoes_peeled_l211_21121


namespace NUMINAMATH_GPT_algebraic_expression_value_l211_21159

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 2*a - 1 = 0) : 2*a^2 - 4*a + 2023 = 2025 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l211_21159


namespace NUMINAMATH_GPT_find_number_l211_21140

theorem find_number (x : ℝ) (h : 0.35 * x = 0.50 * x - 24) : x = 160 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l211_21140


namespace NUMINAMATH_GPT_fraction_powers_sum_l211_21168

theorem fraction_powers_sum : 
  ( (5:ℚ) / (3:ℚ) )^6 + ( (2:ℚ) / (3:ℚ) )^6 = (15689:ℚ) / (729:ℚ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_powers_sum_l211_21168


namespace NUMINAMATH_GPT_product_terms_l211_21197

variable (a_n : ℕ → ℝ)
variable (r : ℝ)

-- a1 = 1 and a10 = 3
axiom geom_seq  (h : ∀ n, a_n (n + 1) = r * a_n n) : a_n 1 = 1 → a_n 10 = 3

theorem product_terms :
  (∀ n, a_n (n + 1) = r * a_n n) → a_n 1 = 1 → a_n 10 = 3 → 
  a_n 2 * a_n 3 * a_n 4 * a_n 5 * a_n 6 * a_n 7 * a_n 8 * a_n 9 = 81 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_product_terms_l211_21197


namespace NUMINAMATH_GPT_fraction_simplification_l211_21104

theorem fraction_simplification (x : ℝ) (h : x = 0.5 * 106) : 18 / x = 18 / 53 := by
  rw [h]
  norm_num

end NUMINAMATH_GPT_fraction_simplification_l211_21104


namespace NUMINAMATH_GPT_inequality_proof_l211_21128

theorem inequality_proof (m n : ℝ) (h1 : m < n) (h2 : n < 0) : (n / m) + (m / n) > 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l211_21128


namespace NUMINAMATH_GPT_trains_meet_in_32_seconds_l211_21176

noncomputable def length_first_train : ℕ := 400
noncomputable def length_second_train : ℕ := 200
noncomputable def initial_distance : ℕ := 200

noncomputable def speed_first_train : ℕ := 15
noncomputable def speed_second_train : ℕ := 10

noncomputable def relative_speed : ℕ := speed_first_train + speed_second_train
noncomputable def total_distance : ℕ := length_first_train + length_second_train + initial_distance
noncomputable def time_to_meet := total_distance / relative_speed

theorem trains_meet_in_32_seconds : time_to_meet = 32 := by
  sorry

end NUMINAMATH_GPT_trains_meet_in_32_seconds_l211_21176


namespace NUMINAMATH_GPT_power_of_three_l211_21106

theorem power_of_three (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_mult : (3^a) * (3^b) = 81) : (3^a)^b = 81 :=
sorry

end NUMINAMATH_GPT_power_of_three_l211_21106


namespace NUMINAMATH_GPT_matchsticks_in_20th_stage_l211_21125

-- Define the first term and common difference
def first_term : ℕ := 4
def common_difference : ℕ := 3

-- Define the mathematical function for the n-th term of the arithmetic sequence
def num_matchsticks (n : ℕ) : ℕ :=
  first_term + (n - 1) * common_difference

-- State the theorem to prove the number of matchsticks in the 20th stage
theorem matchsticks_in_20th_stage : num_matchsticks 20 = 61 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_matchsticks_in_20th_stage_l211_21125


namespace NUMINAMATH_GPT_steven_erasers_l211_21129

theorem steven_erasers (skittles erasers groups items_per_group total_items : ℕ)
  (h1 : skittles = 4502)
  (h2 : groups = 154)
  (h3 : items_per_group = 57)
  (h4 : total_items = groups * items_per_group)
  (h5 : total_items - skittles = erasers) :
  erasers = 4276 :=
by
  sorry

end NUMINAMATH_GPT_steven_erasers_l211_21129


namespace NUMINAMATH_GPT_trip_distance_1200_miles_l211_21156

theorem trip_distance_1200_miles
    (D : ℕ)
    (H : D / 50 - D / 60 = 4) :
    D = 1200 :=
by
    sorry

end NUMINAMATH_GPT_trip_distance_1200_miles_l211_21156


namespace NUMINAMATH_GPT_find_alpha_l211_21175

theorem find_alpha (α : ℝ) (h : Real.sin α * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) = 1) :
  α = 13 * Real.pi / 18 :=
sorry

end NUMINAMATH_GPT_find_alpha_l211_21175


namespace NUMINAMATH_GPT_compare_inequalities_l211_21193

theorem compare_inequalities (a b c π : ℝ) (h1 : a > π) (h2 : π > b) (h3 : b > 1) (h4 : 1 > c) (h5 : c > 0) 
  (x := a^(1 / π)) (y := Real.log b / Real.log π) (z := Real.log π / Real.log c) : x > y ∧ y > z := 
sorry

end NUMINAMATH_GPT_compare_inequalities_l211_21193


namespace NUMINAMATH_GPT_weight_of_b_l211_21122

variable (Wa Wb Wc: ℝ)

-- Conditions
def avg_weight_abc : Prop := (Wa + Wb + Wc) / 3 = 45
def avg_weight_ab : Prop := (Wa + Wb) / 2 = 40
def avg_weight_bc : Prop := (Wb + Wc) / 2 = 43

-- Theorem to prove
theorem weight_of_b (Wa Wb Wc: ℝ) (h_avg_abc : avg_weight_abc Wa Wb Wc)
  (h_avg_ab : avg_weight_ab Wa Wb) (h_avg_bc : avg_weight_bc Wb Wc) : Wb = 31 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_b_l211_21122


namespace NUMINAMATH_GPT_repayment_is_correct_l211_21137

noncomputable def repayment_amount (a r : ℝ) : ℝ := a * r * (1 + r) ^ 5 / ((1 + r) ^ 5 - 1)

theorem repayment_is_correct (a r : ℝ) (h_a : a > 0) (h_r : r > 0) :
  repayment_amount a r = a * r * (1 + r) ^ 5 / ((1 + r) ^ 5 - 1) :=
by
  sorry

end NUMINAMATH_GPT_repayment_is_correct_l211_21137


namespace NUMINAMATH_GPT_absolute_value_sum_10_terms_l211_21100

def sequence_sum (n : ℕ) : ℤ := (n^2 - 4 * n + 2)

def term (n : ℕ) : ℤ := sequence_sum n - sequence_sum (n - 1)

-- Prove that the sum of the absolute values of the first 10 terms is 66.
theorem absolute_value_sum_10_terms : 
  (|term 1| + |term 2| + |term 3| + |term 4| + |term 5| + 
   |term 6| + |term 7| + |term 8| + |term 9| + |term 10| = 66) := 
by 
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_absolute_value_sum_10_terms_l211_21100


namespace NUMINAMATH_GPT_power_add_one_eq_twice_l211_21160

theorem power_add_one_eq_twice (a b : ℕ) (h : 2^a = b) : 2^(a + 1) = 2 * b := by
  sorry

end NUMINAMATH_GPT_power_add_one_eq_twice_l211_21160
