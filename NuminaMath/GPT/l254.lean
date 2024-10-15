import Mathlib

namespace NUMINAMATH_GPT_index_card_area_l254_25401

theorem index_card_area (length width : ℕ) (h_length : length = 5) (h_width : width = 7)
  (h_area_shortened_length : (length - 2) * width = 21) : (length * (width - 2)) = 25 := by
  sorry

end NUMINAMATH_GPT_index_card_area_l254_25401


namespace NUMINAMATH_GPT_factorization_correct_l254_25443

-- Define the expression
def expression (x : ℝ) : ℝ := x^2 + 2 * x

-- State the theorem to prove the factorized form is equal to the expression
theorem factorization_correct (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by {
  -- Lean will skip the proof because of sorry, ensuring the statement compiles correctly.
  sorry
}

end NUMINAMATH_GPT_factorization_correct_l254_25443


namespace NUMINAMATH_GPT_total_transport_cost_l254_25407

def cost_per_kg : ℝ := 25000
def mass_sensor_g : ℝ := 350
def mass_communication_g : ℝ := 150

theorem total_transport_cost : 
  (cost_per_kg * (mass_sensor_g / 1000) + cost_per_kg * (mass_communication_g / 1000)) = 12500 :=
by
  sorry

end NUMINAMATH_GPT_total_transport_cost_l254_25407


namespace NUMINAMATH_GPT_students_distribute_l254_25490

theorem students_distribute (x y : ℕ) (h₁ : x + y = 4200)
        (h₂ : x * 108 / 100 + y * 111 / 100 = 4620) :
    x = 1400 ∧ y = 2800 :=
by
  sorry

end NUMINAMATH_GPT_students_distribute_l254_25490


namespace NUMINAMATH_GPT_tan_960_eq_sqrt_3_l254_25438

theorem tan_960_eq_sqrt_3 : Real.tan (960 * Real.pi / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_960_eq_sqrt_3_l254_25438


namespace NUMINAMATH_GPT_prime_half_sum_l254_25473

theorem prime_half_sum
  (a b c : ℕ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : Nat.Prime (a.factorial + b + c))
  (h2 : Nat.Prime (b.factorial + c + a))
  (h3 : Nat.Prime (c.factorial + a + b)) :
  Nat.Prime ((a + b + c + 1) / 2) := 
sorry

end NUMINAMATH_GPT_prime_half_sum_l254_25473


namespace NUMINAMATH_GPT_probability_of_picking_grain_buds_l254_25445

theorem probability_of_picking_grain_buds :
  let num_stamps := 3
  let num_grain_buds := 1
  let probability := num_grain_buds / num_stamps
  probability = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_picking_grain_buds_l254_25445


namespace NUMINAMATH_GPT_eccentricity_range_l254_25437

noncomputable def ellipse_eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) : Prop :=
  ∃ c : ℝ, c^2 = a^2 - b^2 ∧ e = c / a ∧ (2 * ((-a) * (c + a / 2) - (b / 2) * b) + b^2 + c^2 ≥ 0)

theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ e : ℝ, ellipse_eccentricity_range a b h e ∧ (0 < e ∧ e ≤ -1 + Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_eccentricity_range_l254_25437


namespace NUMINAMATH_GPT_find_positive_integer_solutions_l254_25424

-- Define the problem conditions
variable {x y z : ℕ}

-- Main theorem statement
theorem find_positive_integer_solutions 
  (h1 : Prime y)
  (h2 : ¬ 3 ∣ z)
  (h3 : ¬ y ∣ z)
  (h4 : x^3 - y^3 = z^2) : 
  x = 8 ∧ y = 7 ∧ z = 13 := 
sorry

end NUMINAMATH_GPT_find_positive_integer_solutions_l254_25424


namespace NUMINAMATH_GPT_find_b_of_square_binomial_l254_25472

theorem find_b_of_square_binomial (b : ℚ) 
  (h : ∃ c : ℚ, ∀ x : ℚ, (3 * x + c) ^ 2 = 9 * x ^ 2 + 21 * x + b) : 
  b = 49 / 4 := 
sorry

end NUMINAMATH_GPT_find_b_of_square_binomial_l254_25472


namespace NUMINAMATH_GPT_age_difference_l254_25488

variable (A J : ℕ)
variable (h1 : A + 5 = 40)
variable (h2 : J = 31)

theorem age_difference (h1 : A + 5 = 40) (h2 : J = 31) : A - J = 4 := by
  sorry

end NUMINAMATH_GPT_age_difference_l254_25488


namespace NUMINAMATH_GPT_students_per_group_l254_25423

-- Definitions for conditions
def number_of_boys : ℕ := 28
def number_of_girls : ℕ := 4
def number_of_groups : ℕ := 8
def total_students : ℕ := number_of_boys + number_of_girls

-- The Theorem we want to prove
theorem students_per_group : total_students / number_of_groups = 4 := by
  sorry

end NUMINAMATH_GPT_students_per_group_l254_25423


namespace NUMINAMATH_GPT_arun_weight_average_l254_25448

theorem arun_weight_average :
  (∀ w : ℝ, 65 < w ∧ w < 72 → 60 < w ∧ w < 70 → w ≤ 68 → 66 ≤ w ∧ w ≤ 69 → 64 ≤ w ∧ w ≤ 67.5 → 
    (66.75 = (66 + 67.5) / 2)) := by
  sorry

end NUMINAMATH_GPT_arun_weight_average_l254_25448


namespace NUMINAMATH_GPT_difference_between_numbers_l254_25416

noncomputable def L : ℕ := 1614
noncomputable def Q : ℕ := 6
noncomputable def R : ℕ := 15

theorem difference_between_numbers (S : ℕ) (h : L = Q * S + R) : L - S = 1348 :=
by {
  -- proof skipped
  sorry
}

end NUMINAMATH_GPT_difference_between_numbers_l254_25416


namespace NUMINAMATH_GPT_cream_ratio_l254_25484

noncomputable def joe_coffee_initial := 14
noncomputable def joe_coffee_drank := 3
noncomputable def joe_cream_added := 3

noncomputable def joann_coffee_initial := 14
noncomputable def joann_cream_added := 3
noncomputable def joann_mixture_stirred := 17
noncomputable def joann_amount_drank := 3

theorem cream_ratio (joe_coffee_initial joe_coffee_drank joe_cream_added 
                     joann_coffee_initial joann_cream_added joann_mixture_stirred 
                     joann_amount_drank : ℝ) : 
  (joe_coffee_initial - joe_coffee_drank + joe_cream_added) / 
  (joann_cream_added - (joann_amount_drank * (joann_cream_added / joann_mixture_stirred))) = 17 / 14 :=
by
  -- Prove the theorem statement
  sorry

end NUMINAMATH_GPT_cream_ratio_l254_25484


namespace NUMINAMATH_GPT_doctors_to_lawyers_ratio_l254_25486

theorem doctors_to_lawyers_ratio
  (d l : ℕ)
  (h1 : (40 * d + 55 * l) / (d + l) = 45)
  (h2 : d + l = 20) :
  d / l = 2 :=
by sorry

end NUMINAMATH_GPT_doctors_to_lawyers_ratio_l254_25486


namespace NUMINAMATH_GPT_solve_linear_system_l254_25433

theorem solve_linear_system (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  2 * x₁ + 2 * x₂ - x₃ + x₄ + 4 * x₆ = 0 ∧
  x₁ + 2 * x₂ + 2 * x₃ + 3 * x₅ + x₆ = -2 ∧
  x₁ - 2 * x₂ + x₄ + 2 * x₅ = 0 →
  x₁ = -1 / 4 - 5 / 8 * x₄ - 9 / 8 * x₅ - 9 / 8 * x₆ ∧
  x₂ = -1 / 8 + 3 / 16 * x₄ - 7 / 16 * x₅ + 9 / 16 * x₆ ∧
  x₃ = -3 / 4 + 1 / 8 * x₄ - 11 / 8 * x₅ + 5 / 8 * x₆ :=
by
  sorry

end NUMINAMATH_GPT_solve_linear_system_l254_25433


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l254_25429

theorem equation1_solution (x : ℝ) : 4 * (2 * x - 1) ^ 2 = 36 ↔ x = 2 ∨ x = -1 :=
by sorry

theorem equation2_solution (x : ℝ) : (1 / 4) * (2 * x + 3) ^ 3 - 54 = 0 ↔ x = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l254_25429


namespace NUMINAMATH_GPT_triangle_angles_l254_25414

theorem triangle_angles (r_a r_b r_c R : ℝ) (h1 : r_a + r_b = 3 * R) (h2 : r_b + r_c = 2 * R) :
  ∃ (α β γ : ℝ), α = 90 ∧ γ = 60 ∧ β = 30 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angles_l254_25414


namespace NUMINAMATH_GPT_book_cost_l254_25477

theorem book_cost (n_5 n_3 : ℕ) (N : ℕ) :
  (N = n_5 + n_3) ∧ (N > 10) ∧ (N < 20) ∧ (5 * n_5 = 3 * n_3) →  5 * n_5 = 30 := 
sorry

end NUMINAMATH_GPT_book_cost_l254_25477


namespace NUMINAMATH_GPT_david_first_six_l254_25418

def prob_six := (1:ℚ) / 6
def prob_not_six := (5:ℚ) / 6

def prob_david_first_six_cycle : ℚ :=
  prob_not_six * prob_not_six * prob_not_six * prob_six

def prob_no_six_cycle : ℚ :=
  prob_not_six ^ 4

def infinite_series_sum (a r: ℚ) : ℚ := 
  a / (1 - r)

theorem david_first_six :
  infinite_series_sum prob_david_first_six_cycle prob_no_six_cycle = 125 / 671 :=
by
  sorry

end NUMINAMATH_GPT_david_first_six_l254_25418


namespace NUMINAMATH_GPT_rectangle_area_l254_25422

theorem rectangle_area {A_s A_r : ℕ} (s l w : ℕ) (h1 : A_s = 36) (h2 : A_s = s * s)
  (h3 : w = s) (h4 : l = 3 * w) (h5 : A_r = w * l) : A_r = 108 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l254_25422


namespace NUMINAMATH_GPT_find_third_card_value_l254_25404

noncomputable def point_values (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 13 ∧
  1 ≤ b ∧ b ≤ 13 ∧
  1 ≤ c ∧ c ≤ 13 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b = 25 ∧
  b + c = 13

theorem find_third_card_value :
  ∃ a b c : ℕ, point_values a b c ∧ c = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_third_card_value_l254_25404


namespace NUMINAMATH_GPT_greatest_2q_minus_r_l254_25460

theorem greatest_2q_minus_r :
  ∃ (q r : ℕ), 1027 = 21 * q + r ∧ q > 0 ∧ r > 0 ∧ 2 * q - r = 77 :=
by
  sorry

end NUMINAMATH_GPT_greatest_2q_minus_r_l254_25460


namespace NUMINAMATH_GPT_total_paths_A_to_D_l254_25455

-- Given conditions
def paths_from_A_to_B := 2
def paths_from_B_to_C := 2
def paths_from_C_to_D := 2
def direct_path_A_to_C := 1
def direct_path_B_to_D := 1

-- Proof statement
theorem total_paths_A_to_D : 
  paths_from_A_to_B * paths_from_B_to_C * paths_from_C_to_D + 
  direct_path_A_to_C * paths_from_C_to_D + 
  paths_from_A_to_B * direct_path_B_to_D = 12 := 
  by
    sorry

end NUMINAMATH_GPT_total_paths_A_to_D_l254_25455


namespace NUMINAMATH_GPT_minimum_value_inequality_l254_25483

noncomputable def min_value (a b : ℝ) (ha : 0 < a) (hb : 1 < b) (hab : a + b = 2) : ℝ :=
  (4 / a) + (1 / (b - 1))

theorem minimum_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 1 < b) (hab : a + b = 2) : 
  min_value a b ha hb hab ≥ 9 :=
  sorry

end NUMINAMATH_GPT_minimum_value_inequality_l254_25483


namespace NUMINAMATH_GPT_intersecting_lines_triangle_area_l254_25441

theorem intersecting_lines_triangle_area :
  let line1 := { p : ℝ × ℝ | p.2 = p.1 }
  let line2 := { p : ℝ × ℝ | p.1 = -6 }
  let intersection := (-6, -6)
  let base := 6
  let height := 6
  let area := (1 / 2 : ℝ) * base * height
  area = 18 := by
  sorry

end NUMINAMATH_GPT_intersecting_lines_triangle_area_l254_25441


namespace NUMINAMATH_GPT_gcd_10293_29384_l254_25457

theorem gcd_10293_29384 : Nat.gcd 10293 29384 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_10293_29384_l254_25457


namespace NUMINAMATH_GPT_no_integer_solutions_l254_25444

theorem no_integer_solutions (x y z : ℤ) :
  x^2 - 4 * x * y + 3 * y^2 - z^2 = 25 ∧
  -x^2 + 4 * y * z + 3 * z^2 = 36 ∧
  x^2 + 2 * x * y + 9 * z^2 = 121 → false :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l254_25444


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l254_25464

theorem quadratic_inequality_solution :
  { m : ℝ // ∀ x : ℝ, m * x^2 - 6 * m * x + 5 * m + 1 > 0 } = { m : ℝ // 0 ≤ m ∧ m < 1/4 } :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l254_25464


namespace NUMINAMATH_GPT_product_of_three_consecutive_natural_numbers_divisible_by_six_l254_25494

theorem product_of_three_consecutive_natural_numbers_divisible_by_six (n : ℕ) : 6 ∣ (n * (n + 1) * (n + 2)) :=
by
  sorry

end NUMINAMATH_GPT_product_of_three_consecutive_natural_numbers_divisible_by_six_l254_25494


namespace NUMINAMATH_GPT_solution_set_proof_l254_25487

theorem solution_set_proof {a b : ℝ} :
  (∀ x, 2 < x ∧ x < 3 → x^2 - a * x - b < 0) →
  (∀ x, bx^2 - a * x - 1 > 0) →
  (∀ x, -1 / 2 < x ∧ x < -1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_proof_l254_25487


namespace NUMINAMATH_GPT_four_digit_number_l254_25419

theorem four_digit_number (x : ℕ) (hx : 100 ≤ x ∧ x < 1000) (unit_digit : ℕ) (hu : unit_digit = 2) :
    (10 * x + unit_digit) - (2000 + x) = 108 → 10 * x + unit_digit = 2342 :=
by
  intros h
  sorry


end NUMINAMATH_GPT_four_digit_number_l254_25419


namespace NUMINAMATH_GPT_abc_inequality_l254_25478

-- Define a mathematical statement to encapsulate the problem
theorem abc_inequality (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by sorry

end NUMINAMATH_GPT_abc_inequality_l254_25478


namespace NUMINAMATH_GPT_lunch_break_duration_l254_25492

-- Definitions based on the conditions
variables (p h1 h2 L : ℝ)
-- Monday equation
def monday_eq : Prop := (9 - L/60) * (p + h1 + h2) = 0.55
-- Tuesday equation
def tuesday_eq : Prop := (7 - L/60) * (p + h2) = 0.35
-- Wednesday equation
def wednesday_eq : Prop := (5 - L/60) * (p + h1 + h2) = 0.25
-- Thursday equation
def thursday_eq : Prop := (4 - L/60) * p = 0.15

-- Combine all conditions
def all_conditions : Prop :=
  monday_eq p h1 h2 L ∧ tuesday_eq p h2 L ∧ wednesday_eq p h1 h2 L ∧ thursday_eq p L

-- Proof that the lunch break duration is 60 minutes
theorem lunch_break_duration : all_conditions p h1 h2 L → L = 60 :=
by
  sorry

end NUMINAMATH_GPT_lunch_break_duration_l254_25492


namespace NUMINAMATH_GPT_consecutive_numbers_l254_25426

theorem consecutive_numbers (x : ℕ) (h : (4 * x + 2) * (4 * x^2 + 6 * x + 6) = 3 * (4 * x^3 + 4 * x^2 + 18 * x + 8)) :
  x = 2 :=
sorry

end NUMINAMATH_GPT_consecutive_numbers_l254_25426


namespace NUMINAMATH_GPT_middle_integer_of_consecutive_odd_l254_25469

theorem middle_integer_of_consecutive_odd (n : ℕ)
  (h1 : n > 2)
  (h2 : n < 8)
  (h3 : (n-2) % 2 = 1)
  (h4 : n % 2 = 1)
  (h5 : (n+2) % 2 = 1)
  (h6 : (n-2) + n + (n+2) = (n-2) * n * (n+2) / 9) :
  n = 5 :=
by sorry

end NUMINAMATH_GPT_middle_integer_of_consecutive_odd_l254_25469


namespace NUMINAMATH_GPT_remainder_when_expr_divided_by_9_l254_25481

theorem remainder_when_expr_divided_by_9 (n m p : ℤ)
  (h1 : n % 18 = 10)
  (h2 : m % 27 = 16)
  (h3 : p % 6 = 4) :
  (2 * n + 3 * m - p) % 9 = 1 := 
sorry

end NUMINAMATH_GPT_remainder_when_expr_divided_by_9_l254_25481


namespace NUMINAMATH_GPT_percent_increase_jordan_alex_l254_25454

theorem percent_increase_jordan_alex :
  let pound_to_dollar := 1.5
  let alex_dollars := 600
  let jordan_pounds := 450
  let jordan_dollars := jordan_pounds * pound_to_dollar
  let percent_increase := ((jordan_dollars - alex_dollars) / alex_dollars) * 100
  percent_increase = 12.5 := 
by
  sorry

end NUMINAMATH_GPT_percent_increase_jordan_alex_l254_25454


namespace NUMINAMATH_GPT_exists_constant_a_l254_25496

theorem exists_constant_a (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : (m : ℝ) / n < Real.sqrt 7) :
  ∃ (a : ℝ), a > 1 ∧ (7 - (m^2 : ℝ) / (n^2 : ℝ) ≥ a / (n^2 : ℝ)) ∧ a = 3 :=
by
  sorry

end NUMINAMATH_GPT_exists_constant_a_l254_25496


namespace NUMINAMATH_GPT_light_intensity_at_10_m_l254_25409

theorem light_intensity_at_10_m (k : ℝ) (d1 d2 : ℝ) (I1 I2 : ℝ)
  (h1: I1 = k / d1^2) (h2: I1 = 200) (h3: d1 = 5) (h4: d2 = 10) :
  I2 = k / d2^2 → I2 = 50 :=
sorry

end NUMINAMATH_GPT_light_intensity_at_10_m_l254_25409


namespace NUMINAMATH_GPT_intersection_A_B_union_B_Ac_range_a_l254_25475

open Set

-- Conditions
def U : Set ℝ := univ
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def Ac : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}
def Bc : Set ℝ := {x | x < -2 ∨ x > 5}

-- Questions rewritten as Lean statements

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 5} := sorry

theorem union_B_Ac :
  B ∪ Ac = {x | x ≤ 5 ∨ x ≥ 9} := sorry

theorem range_a (a : ℝ) :
  {x | a ≤ x ∧ x ≤ a + 2} ⊆ Bc → a ∈ Iio (-4) ∪ Ioi 5 := sorry

end NUMINAMATH_GPT_intersection_A_B_union_B_Ac_range_a_l254_25475


namespace NUMINAMATH_GPT_lily_sees_leo_l254_25476

theorem lily_sees_leo : 
  ∀ (d₁ d₂ v₁ v₂ : ℝ), 
  d₁ = 0.75 → 
  d₂ = 0.75 → 
  v₁ = 15 → 
  v₂ = 9 → 
  (d₁ + d₂) / (v₁ - v₂) * 60 = 15 :=
by 
  intros d₁ d₂ v₁ v₂ h₁ h₂ h₃ h₄
  -- skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_lily_sees_leo_l254_25476


namespace NUMINAMATH_GPT_inequality_proof_l254_25499

theorem inequality_proof {k l m n : ℕ} (h_pos_k : 0 < k) (h_pos_l : 0 < l) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_klmn : k < l ∧ l < m ∧ m < n)
  (h_equation : k * n = l * m) : 
  (n - k) / 2 ^ 2 ≥ k + 2 := 
by sorry

end NUMINAMATH_GPT_inequality_proof_l254_25499


namespace NUMINAMATH_GPT_range_of_m_l254_25405

theorem range_of_m (m : ℝ) (h1 : (m - 3) < 0) (h2 : (m + 1) > 0) : -1 < m ∧ m < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l254_25405


namespace NUMINAMATH_GPT_min_f_eq_2_m_n_inequality_l254_25468

def f (x : ℝ) := abs (x + 1) + abs (x - 1)

theorem min_f_eq_2 : (∀ x, f x ≥ 2) ∧ (∃ x, f x = 2) :=
by
  sorry

theorem m_n_inequality (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m^3 + n^3 = 2) : m + n ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_min_f_eq_2_m_n_inequality_l254_25468


namespace NUMINAMATH_GPT_product_of_two_larger_numbers_is_115_l254_25461

noncomputable def proofProblem : Prop :=
  ∃ (A B C : ℝ), B = 10 ∧ (C - B = B - A) ∧ (A * B = 85) ∧ (B * C = 115)

theorem product_of_two_larger_numbers_is_115 : proofProblem :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_larger_numbers_is_115_l254_25461


namespace NUMINAMATH_GPT_unique_solution_quadratic_l254_25459

theorem unique_solution_quadratic (n : ℕ) : (∀ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_unique_solution_quadratic_l254_25459


namespace NUMINAMATH_GPT_arithmetic_example_l254_25412

theorem arithmetic_example : 3889 + 12.808 - 47.80600000000004 = 3854.002 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_example_l254_25412


namespace NUMINAMATH_GPT_n_divisible_by_100_l254_25450

theorem n_divisible_by_100 (n : ℤ) (h1 : n > 101) (h2 : 101 ∣ n)
  (h3 : ∀ d : ℤ, 1 < d ∧ d < n → d ∣ n → ∃ k m : ℤ, k ∣ n ∧ m ∣ n ∧ d = k - m) : 100 ∣ n :=
sorry

end NUMINAMATH_GPT_n_divisible_by_100_l254_25450


namespace NUMINAMATH_GPT_number_of_columns_per_section_l254_25425

variables (S C : ℕ)

-- Define the first condition: S * C + (S - 1) / 2 = 1223
def condition1 := S * C + (S - 1) / 2 = 1223

-- Define the second condition: S = 2 * C + 5
def condition2 := S = 2 * C + 5

-- Formulate the theorem that C = 23 given the two conditions
theorem number_of_columns_per_section
  (h1 : condition1 S C)
  (h2 : condition2 S C) :
  C = 23 :=
sorry

end NUMINAMATH_GPT_number_of_columns_per_section_l254_25425


namespace NUMINAMATH_GPT_domain_of_inverse_l254_25446

noncomputable def f (x : ℝ) : ℝ := 3 ^ x

theorem domain_of_inverse (x : ℝ) : f x > 0 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_inverse_l254_25446


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l254_25458

variable (v_m v_s : ℝ)

-- Conditions as definitions 
def downstream_distance_eq : Prop :=
  36 = (v_m + v_s) * 3

def upstream_distance_eq : Prop :=
  18 = (v_m - v_s) * 3

theorem speed_of_man_in_still_water (h1 : downstream_distance_eq v_m v_s) (h2 : upstream_distance_eq v_m v_s) : v_m = 9 := 
  by
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l254_25458


namespace NUMINAMATH_GPT_ab_equals_six_l254_25480

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_GPT_ab_equals_six_l254_25480


namespace NUMINAMATH_GPT_cos_neg_2theta_l254_25498

theorem cos_neg_2theta (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : Real.cos (-2 * θ) = -7 / 25 := 
by
  sorry

end NUMINAMATH_GPT_cos_neg_2theta_l254_25498


namespace NUMINAMATH_GPT_coffee_customers_l254_25406

theorem coffee_customers (C : ℕ) :
  let coffee_cost := 5
  let tea_ordered := 8
  let tea_cost := 4
  let total_revenue := 67
  (coffee_cost * C + tea_ordered * tea_cost = total_revenue) → C = 7 := by
  sorry

end NUMINAMATH_GPT_coffee_customers_l254_25406


namespace NUMINAMATH_GPT_evaluate_expression_l254_25456

theorem evaluate_expression : (5^2 - 4^2)^3 = 729 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l254_25456


namespace NUMINAMATH_GPT_coin_toss_probability_l254_25471

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem coin_toss_probability :
  binomial_probability 3 2 0.5 = 0.375 :=
by
  sorry

end NUMINAMATH_GPT_coin_toss_probability_l254_25471


namespace NUMINAMATH_GPT_reflect_across_y_axis_l254_25430

-- Definition of the original point A
def pointA : ℝ × ℝ := (2, 3)

-- Definition of the reflected point across the y-axis
def reflectedPoint (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- The theorem stating the reflection result
theorem reflect_across_y_axis : reflectedPoint pointA = (-2, 3) :=
by
  -- Proof (skipped)
  sorry

end NUMINAMATH_GPT_reflect_across_y_axis_l254_25430


namespace NUMINAMATH_GPT_triangle_sides_are_6_8_10_l254_25421

theorem triangle_sides_are_6_8_10 (a b c r r1 r2 r3 : ℕ) (hr_even : Even r) (hr1_even : Even r1) 
(hr2_even : Even r2) (hr3_even : Even r3) (relationship : r * r1 * r2 + r * r2 * r3 + r * r3 * r1 + r1 * r2 * r3 = r * r1 * r2 * r3) :
  (a, b, c) = (6, 8, 10) :=
sorry

end NUMINAMATH_GPT_triangle_sides_are_6_8_10_l254_25421


namespace NUMINAMATH_GPT_largest_number_among_given_l254_25466

theorem largest_number_among_given (
  A B C D E : ℝ
) (hA : A = 0.936)
  (hB : B = 0.9358)
  (hC : C = 0.9361)
  (hD : D = 0.935)
  (hE : E = 0.921):
  C = max A (max B (max C (max D E))) :=
by
  sorry

end NUMINAMATH_GPT_largest_number_among_given_l254_25466


namespace NUMINAMATH_GPT_negation_of_exists_abs_le_two_l254_25452

theorem negation_of_exists_abs_le_two :
  (¬ ∃ x : ℝ, |x| ≤ 2) ↔ (∀ x : ℝ, |x| > 2) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_abs_le_two_l254_25452


namespace NUMINAMATH_GPT_randy_gave_sally_l254_25440

-- Define the given conditions
def initial_amount_randy : ℕ := 3000
def smith_contribution : ℕ := 200
def amount_kept_by_randy : ℕ := 2000

-- The total amount Randy had after Smith's contribution
def total_amount_randy : ℕ := initial_amount_randy + smith_contribution

-- The amount of money Randy gave to Sally
def amount_given_to_sally : ℕ := total_amount_randy - amount_kept_by_randy

-- The theorem statement: Given the conditions, prove that Randy gave Sally $1,200
theorem randy_gave_sally : amount_given_to_sally = 1200 :=
by
  sorry

end NUMINAMATH_GPT_randy_gave_sally_l254_25440


namespace NUMINAMATH_GPT_like_terms_value_l254_25431

theorem like_terms_value (a b : ℤ) (h1 : a + b = 2) (h2 : a - 1 = 1) : a - b = 2 :=
sorry

end NUMINAMATH_GPT_like_terms_value_l254_25431


namespace NUMINAMATH_GPT_age_ratio_7_9_l254_25491

/-- Definition of Sachin and Rahul's ages -/
def sachin_age : ℝ := 24.5
def rahul_age : ℝ := sachin_age + 7

/-- The ratio of Sachin's age to Rahul's age is 7:9 -/
theorem age_ratio_7_9 : sachin_age / rahul_age = 7 / 9 := by
  sorry

end NUMINAMATH_GPT_age_ratio_7_9_l254_25491


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l254_25497

theorem eccentricity_of_ellipse (p q : ℕ) (hp : Nat.Coprime p q) (z : ℂ) :
  ((z - 2) * (z^2 + 3 * z + 5) * (z^2 + 5 * z + 8) = 0) →
  (∃ p q : ℕ, Nat.Coprime p q ∧ (∃ e : ℝ, e^2 = p / q ∧ p + q = 16)) :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l254_25497


namespace NUMINAMATH_GPT_k_value_l254_25470

theorem k_value (k : ℝ) :
    (∀ r s : ℝ, (r + s = -k ∧ r * s = 9) ∧ ((r + 3) + (s + 3) = k)) → k = -3 :=
by
    intro h
    sorry

end NUMINAMATH_GPT_k_value_l254_25470


namespace NUMINAMATH_GPT_angle_380_in_first_quadrant_l254_25432

theorem angle_380_in_first_quadrant : ∃ n : ℤ, 380 - 360 * n = 20 ∧ 0 ≤ 20 ∧ 20 ≤ 90 :=
by
  use 1 -- We use 1 because 380 = 20 + 360 * 1
  sorry

end NUMINAMATH_GPT_angle_380_in_first_quadrant_l254_25432


namespace NUMINAMATH_GPT_Tanya_bought_9_apples_l254_25462

def original_fruit_count : ℕ := 18
def remaining_fruit_count : ℕ := 9
def pears_count : ℕ := 6
def pineapples_count : ℕ := 2
def plums_basket_count : ℕ := 1

theorem Tanya_bought_9_apples : 
  remaining_fruit_count * 2 = original_fruit_count →
  original_fruit_count - (pears_count + pineapples_count + plums_basket_count) = 9 :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_Tanya_bought_9_apples_l254_25462


namespace NUMINAMATH_GPT_parallelogram_area_is_correct_l254_25439

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨0, 2, 3⟩
def B : Point3D := ⟨2, 5, 2⟩
def C : Point3D := ⟨-2, 3, 6⟩

noncomputable def vectorAB (A B : Point3D) : Point3D :=
  { x := B.x - A.x
  , y := B.y - A.y
  , z := B.z - A.z 
  }

noncomputable def vectorAC (A C : Point3D) : Point3D :=
  { x := C.x - A.x
  , y := C.y - A.y
  , z := C.z - A.z 
  }

noncomputable def dotProduct (u v : Point3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

noncomputable def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

noncomputable def sinAngle (u v : Point3D) : ℝ :=
  Real.sqrt (1 - (dotProduct u v / (magnitude u * magnitude v)) ^ 2)

noncomputable def parallelogramArea (u v : Point3D) : ℝ :=
  magnitude u * magnitude v * sinAngle u v

theorem parallelogram_area_is_correct :
  parallelogramArea (vectorAB A B) (vectorAC A C) = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_is_correct_l254_25439


namespace NUMINAMATH_GPT_fish_ratio_l254_25413

theorem fish_ratio (k : ℕ) (kendra_fish : ℕ) (home_fish : ℕ)
    (h1 : kendra_fish = 30)
    (h2 : home_fish = 87)
    (h3 : k - 3 + kendra_fish = home_fish) :
  k = 60 ∧ (k / 3, kendra_fish / 3) = (19, 10) :=
by
  sorry

end NUMINAMATH_GPT_fish_ratio_l254_25413


namespace NUMINAMATH_GPT_sin_double_angle_l254_25400

theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l254_25400


namespace NUMINAMATH_GPT_proof_problem_l254_25449

noncomputable def problem_statement (a b c d : ℝ) : Prop :=
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (a + b = 2 * c) ∧ (a * b = -5 * d) ∧ (c + d = 2 * a) ∧ (c * d = -5 * b)

theorem proof_problem (a b c d : ℝ) (h : problem_statement a b c d) : a + b + c + d = 30 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l254_25449


namespace NUMINAMATH_GPT_solve_equation_l254_25489

theorem solve_equation (x : ℝ) (h : ((x^2 + 3*x + 4) / (x + 5)) = x + 6) : x = -13 / 4 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l254_25489


namespace NUMINAMATH_GPT_common_ratio_of_arithmetic_sequence_l254_25474

variable {α : Type} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_ratio_of_arithmetic_sequence (a : ℕ → α) (q : α)
  (h1 : is_arithmetic_sequence a)
  (h2 : ∀ n : ℕ, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
  (h3 : a 1 > 0)
  (h4 : ∀ n : ℕ, a n < a (n + 1)) :
  q = 2 := 
sorry

end NUMINAMATH_GPT_common_ratio_of_arithmetic_sequence_l254_25474


namespace NUMINAMATH_GPT_markese_earnings_16_l254_25485

theorem markese_earnings_16 (E M : ℕ) (h1 : M = E - 5) (h2 : E + M = 37) : M = 16 :=
by
  sorry

end NUMINAMATH_GPT_markese_earnings_16_l254_25485


namespace NUMINAMATH_GPT_work_days_together_l254_25493

theorem work_days_together (A_days B_days : ℕ) (work_left_fraction : ℚ) 
  (hA : A_days = 15) (hB : B_days = 20) (h_fraction : work_left_fraction = 8 / 15) : 
  ∃ d : ℕ, d * (1 / 15 + 1 / 20) = 1 - 8 / 15 ∧ d = 4 :=
by
  sorry

end NUMINAMATH_GPT_work_days_together_l254_25493


namespace NUMINAMATH_GPT_piglets_each_ate_6_straws_l254_25415

theorem piglets_each_ate_6_straws (total_straws : ℕ) (fraction_for_adult_pigs : ℚ) (piglets : ℕ) 
  (h1 : total_straws = 300) 
  (h2 : fraction_for_adult_pigs = 3/5) 
  (h3 : piglets = 20) :
  (total_straws * (1 - fraction_for_adult_pigs) / piglets) = 6 :=
by
  sorry

end NUMINAMATH_GPT_piglets_each_ate_6_straws_l254_25415


namespace NUMINAMATH_GPT_chips_count_l254_25420

theorem chips_count (B G P R x : ℕ) 
  (hx1 : 5 < x) (hx2 : x < 11) 
  (h : 1^B * 5^G * x^P * 11^R = 28160) : 
  P = 2 :=
by 
  -- Hint: Prime factorize 28160 to apply constraints and identify corresponding exponents.
  have prime_factorization_28160 : 28160 = 2^6 * 5^1 * 7^2 := by sorry
  -- Given 5 < x < 11 and by prime factorization, x can only be 7 (since it factors into the count of 7)
  -- Complete the rest of the proof
  sorry

end NUMINAMATH_GPT_chips_count_l254_25420


namespace NUMINAMATH_GPT_letters_written_l254_25482

theorem letters_written (nathan_rate : ℕ) (jacob_rate : ℕ) (combined_rate : ℕ) (hours : ℕ) :
  nathan_rate = 25 →
  jacob_rate = 2 * nathan_rate →
  combined_rate = nathan_rate + jacob_rate →
  hours = 10 →
  combined_rate * hours = 750 :=
by
  intros
  sorry

end NUMINAMATH_GPT_letters_written_l254_25482


namespace NUMINAMATH_GPT_coins_problem_l254_25410

theorem coins_problem : 
  ∃ x : ℕ, 
  (x % 8 = 6) ∧ 
  (x % 7 = 5) ∧ 
  (x % 9 = 1) ∧ 
  (x % 11 = 0) := 
by
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_coins_problem_l254_25410


namespace NUMINAMATH_GPT_calculation_is_correct_l254_25417

-- Define the numbers involved in the calculation
def a : ℝ := 12.05
def b : ℝ := 5.4
def c : ℝ := 0.6

-- Expected result of the calculation
def expected_result : ℝ := 65.67

-- Prove that the calculation is correct
theorem calculation_is_correct : (a * b + c) = expected_result :=
by
  sorry

end NUMINAMATH_GPT_calculation_is_correct_l254_25417


namespace NUMINAMATH_GPT_ratio_of_perimeters_l254_25402

theorem ratio_of_perimeters (s d s' d': ℝ) (h1 : d = s * Real.sqrt 2) (h2 : d' = 2.5 * d) (h3 : d' = s' * Real.sqrt 2) : (4 * s') / (4 * s) = 5 / 2 :=
by
  -- Additional tactical details for completion, proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_ratio_of_perimeters_l254_25402


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l254_25451

open Real

/--
Given that 2013° can be represented as 213° + 5 * 360° and that 213° is a third quadrant angle,
we can deduce that 2013° is also a third quadrant angle.
-/
theorem angle_in_third_quadrant (h1 : 2013 = 213 + 5 * 360) (h2 : 180 < 213 ∧ 213 < 270) : 
  (540 < 2013 % 360 ∧ 2013 % 360 < 270) :=
sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l254_25451


namespace NUMINAMATH_GPT_correct_option_a_l254_25463

theorem correct_option_a (x y a b : ℝ) : 3 * x - 2 * x = x :=
by sorry

end NUMINAMATH_GPT_correct_option_a_l254_25463


namespace NUMINAMATH_GPT_julia_played_with_kids_on_Monday_l254_25442

theorem julia_played_with_kids_on_Monday (k_wednesday : ℕ) (k_monday : ℕ)
  (h1 : k_wednesday = 4) (h2 : k_monday = k_wednesday + 2) : k_monday = 6 := 
by
  sorry

end NUMINAMATH_GPT_julia_played_with_kids_on_Monday_l254_25442


namespace NUMINAMATH_GPT_range_of_f_is_0_2_3_l254_25465

def f (x : ℤ) : ℤ := x + 1
def S : Set ℤ := {-1, 1, 2}

theorem range_of_f_is_0_2_3 : Set.image f S = {0, 2, 3} := by
  sorry

end NUMINAMATH_GPT_range_of_f_is_0_2_3_l254_25465


namespace NUMINAMATH_GPT_geometric_series_first_term_l254_25453

theorem geometric_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h_r : r = 1/4) (h_S : S = 80)
  (h_sum : S = a / (1 - r)) : a = 60 :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_geometric_series_first_term_l254_25453


namespace NUMINAMATH_GPT_find_length_of_CE_l254_25408

theorem find_length_of_CE
  (triangle_ABE_right : ∀ A B E : Type, ∃ (angle_AEB : Real), angle_AEB = 45)
  (triangle_BCE_right : ∀ B C E : Type, ∃ (angle_BEC : Real), angle_BEC = 45)
  (triangle_CDE_right : ∀ C D E : Type, ∃ (angle_CED : Real), angle_CED = 45)
  (AE_is_32 : 32 = 32) :
  ∃ (CE : ℝ), CE = 16 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_length_of_CE_l254_25408


namespace NUMINAMATH_GPT_quadratic_trinomial_has_two_roots_l254_25436

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  4 * (a^2 - a * b + b^2 - 3 * a * c) > 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_trinomial_has_two_roots_l254_25436


namespace NUMINAMATH_GPT_power_vs_square_l254_25435

theorem power_vs_square (n : ℕ) (h : n ≥ 4) : 2^n ≥ n^2 := by
  sorry

end NUMINAMATH_GPT_power_vs_square_l254_25435


namespace NUMINAMATH_GPT_inequality_solution_l254_25495

noncomputable def solution_set (x : ℝ) : Prop := 
  (x < -1) ∨ (x > 3)

theorem inequality_solution :
  { x : ℝ | (3 - x) / (x + 1) < 0 } = { x : ℝ | solution_set x } :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l254_25495


namespace NUMINAMATH_GPT_simplify_expression_l254_25411

theorem simplify_expression : (2 * 3 * b * 4 * (b ^ 2) * 5 * (b ^ 3) * 6 * (b ^ 4)) = 720 * (b ^ 10) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l254_25411


namespace NUMINAMATH_GPT_ryegrass_percentage_l254_25467

theorem ryegrass_percentage (x_ryegrass_percent : ℝ) (y_ryegrass_percent : ℝ) (mixture_x_percent : ℝ)
  (hx : x_ryegrass_percent = 0.40)
  (hy : y_ryegrass_percent = 0.25)
  (hmx : mixture_x_percent = 0.8667) :
  (x_ryegrass_percent * mixture_x_percent + y_ryegrass_percent * (1 - mixture_x_percent)) * 100 = 38 :=
by
  sorry

end NUMINAMATH_GPT_ryegrass_percentage_l254_25467


namespace NUMINAMATH_GPT_g_difference_l254_25403

variable (g : ℝ → ℝ)

-- Condition: g is a linear function
axiom linear_g : ∃ a b : ℝ, ∀ x : ℝ, g x = a * x + b

-- Condition: g(10) - g(4) = 18
axiom g_condition : g 10 - g 4 = 18

theorem g_difference : g 16 - g 4 = 36 :=
by
  sorry

end NUMINAMATH_GPT_g_difference_l254_25403


namespace NUMINAMATH_GPT_least_possible_sum_l254_25447

theorem least_possible_sum (x y z : ℕ) (h1 : 2 * x = 5 * y) (h2 : 5 * y = 6 * z) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x + y + z = 26 :=
by sorry

end NUMINAMATH_GPT_least_possible_sum_l254_25447


namespace NUMINAMATH_GPT_last_two_digits_of_squared_expression_l254_25428

theorem last_two_digits_of_squared_expression (n : ℕ) :
  (n * 2 * 3 * 4 * 46 * 47 * 48 * 49) ^ 2 % 100 = 76 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_squared_expression_l254_25428


namespace NUMINAMATH_GPT_decreasing_function_l254_25479

noncomputable def f (a x : ℝ) : ℝ := a^(1 - x)

theorem decreasing_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : ∀ x > 1, f a x < 1) :
  ∀ x y : ℝ, x < y → f a x > f a y :=
sorry

end NUMINAMATH_GPT_decreasing_function_l254_25479


namespace NUMINAMATH_GPT_christopher_strolling_time_l254_25427

theorem christopher_strolling_time
  (initial_distance : ℝ) (initial_speed : ℝ) (break_time : ℝ)
  (continuation_distance : ℝ) (continuation_speed : ℝ)
  (H1 : initial_distance = 2) (H2 : initial_speed = 4)
  (H3 : break_time = 0.25) (H4 : continuation_distance = 3)
  (H5 : continuation_speed = 6) :
  (initial_distance / initial_speed + break_time + continuation_distance / continuation_speed) = 1.25 := 
  sorry

end NUMINAMATH_GPT_christopher_strolling_time_l254_25427


namespace NUMINAMATH_GPT_alloy_copper_percentage_l254_25434

theorem alloy_copper_percentage 
  (x : ℝ)
  (h1 : 0 ≤ x)
  (h2 : (30 / 100) * x + (70 / 100) * 27 = 24.9) :
  x = 20 :=
sorry

end NUMINAMATH_GPT_alloy_copper_percentage_l254_25434
