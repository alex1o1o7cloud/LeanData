import Mathlib

namespace NUMINAMATH_GPT_square_octagon_can_cover_ground_l1390_139048

def square_interior_angle := 90
def octagon_interior_angle := 135

theorem square_octagon_can_cover_ground :
  square_interior_angle + 2 * octagon_interior_angle = 360 :=
by
  -- Proof skipped with sorry
  sorry

end NUMINAMATH_GPT_square_octagon_can_cover_ground_l1390_139048


namespace NUMINAMATH_GPT_monotonic_decreasing_range_of_a_l1390_139021

-- Define the given function
def f (a x : ℝ) := a * x^2 - 3 * x + 4

-- State the proof problem
theorem monotonic_decreasing_range_of_a (a : ℝ) : (∀ x : ℝ, x < 6 → deriv (f a) x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 1/4 :=
sorry

end NUMINAMATH_GPT_monotonic_decreasing_range_of_a_l1390_139021


namespace NUMINAMATH_GPT_hexagon_rectangle_ratio_l1390_139043

theorem hexagon_rectangle_ratio:
  ∀ (h w : ℕ), 
  (6 * h = 24) → (2 * (2 * w + w) = 24) → 
  (h / w = 1) := by
  intros h w
  intro hex_condition
  intro rect_condition
  sorry

end NUMINAMATH_GPT_hexagon_rectangle_ratio_l1390_139043


namespace NUMINAMATH_GPT_inverse_proportional_l1390_139026

theorem inverse_proportional (p q : ℝ) (k : ℝ) 
  (h1 : ∀ (p q : ℝ), p * q = k)
  (h2 : p = 25)
  (h3 : q = 6) 
  (h4 : q = 15) : 
  p = 10 := 
by
  sorry

end NUMINAMATH_GPT_inverse_proportional_l1390_139026


namespace NUMINAMATH_GPT_expression_result_l1390_139099

theorem expression_result :
  ( (9 + (1 / 2)) + (7 + (1 / 6)) + (5 + (1 / 12)) + (3 + (1 / 20)) + (1 + (1 / 30)) ) * 12 = 310 := by
  sorry

end NUMINAMATH_GPT_expression_result_l1390_139099


namespace NUMINAMATH_GPT_negation_of_no_slow_learners_attend_school_l1390_139053

variable {α : Type}
variable (SlowLearner : α → Prop) (AttendsSchool : α → Prop)

-- The original statement
def original_statement : Prop := ∀ x, SlowLearner x → ¬ AttendsSchool x

-- The corresponding negation
def negation_statement : Prop := ∃ x, SlowLearner x ∧ AttendsSchool x

-- The proof problem statement
theorem negation_of_no_slow_learners_attend_school : 
  ¬ original_statement SlowLearner AttendsSchool ↔ negation_statement SlowLearner AttendsSchool := by
  sorry

end NUMINAMATH_GPT_negation_of_no_slow_learners_attend_school_l1390_139053


namespace NUMINAMATH_GPT_sum_f_a_seq_positive_l1390_139041

noncomputable def f (x : ℝ) : ℝ := sorry
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_monotone_decreasing_nonneg : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x ≤ y → f y ≤ f x
axiom a_seq : ∀ n : ℕ, ℝ
axiom a_arithmetic : ∀ m n k : ℕ, m + k = 2 * n → a_seq m + a_seq k = 2 * a_seq n
axiom a3_neg : a_seq 3 < 0

theorem sum_f_a_seq_positive :
    f (a_seq 1) + 
    f (a_seq 2) + 
    f (a_seq 3) + 
    f (a_seq 4) + 
    f (a_seq 5) > 0 :=
sorry

end NUMINAMATH_GPT_sum_f_a_seq_positive_l1390_139041


namespace NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l1390_139061

noncomputable def sum_of_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

def nth_term (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_tenth_term
  (a1 d : ℝ)
  (h1 : a1 + (a1 + d) + (a1 + 2 * d) = (a1 + 3 * d) + (a1 + 4 * d))
  (h2 : sum_of_arithmetic_sequence a1 d 5 = 60) :
  nth_term a1 d 10 = 26 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l1390_139061


namespace NUMINAMATH_GPT_Bella_bought_38_stamps_l1390_139087

def stamps (n t r : ℕ) : ℕ :=
  n + t + r

theorem Bella_bought_38_stamps :
  ∃ (n t r : ℕ),
    n = 11 ∧
    t = n + 9 ∧
    r = t - 13 ∧
    stamps n t r = 38 := 
  by
  sorry

end NUMINAMATH_GPT_Bella_bought_38_stamps_l1390_139087


namespace NUMINAMATH_GPT_sin_add_double_alpha_l1390_139036

open Real

theorem sin_add_double_alpha (alpha : ℝ) (h : sin (π / 6 - alpha) = 3 / 5) :
  sin (π / 6 + 2 * alpha) = 7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_add_double_alpha_l1390_139036


namespace NUMINAMATH_GPT_triangle_hypotenuse_segments_l1390_139080

theorem triangle_hypotenuse_segments :
  ∀ (x : ℝ) (BC AC : ℝ),
  BC / AC = 3 / 7 →
  ∃ (h : ℝ) (BD AD : ℝ),
    h = 42 ∧
    BD * AD = h^2 ∧
    BD / AD = 9 / 49 ∧
    BD = 18 ∧
    AD = 98 :=
by
  sorry

end NUMINAMATH_GPT_triangle_hypotenuse_segments_l1390_139080


namespace NUMINAMATH_GPT_minimize_square_sum_l1390_139028

theorem minimize_square_sum (x1 x2 x3 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) 
  (h4 : x1 + 3 * x2 + 5 * x3 = 100) : 
  x1^2 + x2^2 + x3^2 ≥ 2000 / 7 :=
sorry

end NUMINAMATH_GPT_minimize_square_sum_l1390_139028


namespace NUMINAMATH_GPT_priyas_speed_is_30_l1390_139093

noncomputable def find_priyas_speed (v : ℝ) : Prop :=
  let riya_speed := 20
  let time := 0.5  -- in hours
  let distance_apart := 25
  (riya_speed + v) * time = distance_apart

theorem priyas_speed_is_30 : ∃ v : ℝ, find_priyas_speed v ∧ v = 30 :=
by
  sorry

end NUMINAMATH_GPT_priyas_speed_is_30_l1390_139093


namespace NUMINAMATH_GPT_lucy_packs_of_cake_l1390_139039

theorem lucy_packs_of_cake (total_groceries cookies : ℕ) (h1 : total_groceries = 27) (h2 : cookies = 23) :
  total_groceries - cookies = 4 :=
by
  -- In Lean, we would provide the actual proof here, but we'll use sorry to skip the proof as instructed
  sorry

end NUMINAMATH_GPT_lucy_packs_of_cake_l1390_139039


namespace NUMINAMATH_GPT_fixed_point_coordinates_l1390_139078

noncomputable def fixed_point (A : Real × Real) : Prop :=
∀ (k : Real), ∃ (x y : Real), A = (x, y) ∧ (3 + k) * x + (1 - 2 * k) * y + 1 + 5 * k = 0

theorem fixed_point_coordinates :
  fixed_point (-1, 2) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_coordinates_l1390_139078


namespace NUMINAMATH_GPT_cube_surface_area_increase_l1390_139059

theorem cube_surface_area_increase (s : ℝ) : 
  let original_surface_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_surface_area := 6 * (new_edge)^2
  let percentage_increase := ((new_surface_area - original_surface_area) / original_surface_area) * 100
  percentage_increase = 69 := 
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_increase_l1390_139059


namespace NUMINAMATH_GPT_max_value_k_l1390_139086

theorem max_value_k (x y k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < k)
(h4 : 4 = k^2 * (x^2 / y^2 + 2 + y^2 / x^2) + k^3 * (x / y + y / x)) : 
k ≤ 4 * (Real.sqrt 2) - 4 :=
by sorry

end NUMINAMATH_GPT_max_value_k_l1390_139086


namespace NUMINAMATH_GPT_simplify_fraction_expression_l1390_139050

variable (d : ℤ)

theorem simplify_fraction_expression : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_expression_l1390_139050


namespace NUMINAMATH_GPT_probability_differ_by_three_is_one_sixth_l1390_139000

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end NUMINAMATH_GPT_probability_differ_by_three_is_one_sixth_l1390_139000


namespace NUMINAMATH_GPT_calc_expression_l1390_139022

theorem calc_expression : 2 * 0 * 1 + 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_l1390_139022


namespace NUMINAMATH_GPT_determine_x_2y_l1390_139015

theorem determine_x_2y (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : (x + y) / 3 = 5 / 3) : x + 2 * y = 8 :=
sorry

end NUMINAMATH_GPT_determine_x_2y_l1390_139015


namespace NUMINAMATH_GPT_grocery_store_more_expensive_l1390_139063

def bulk_price_per_can (total_price : ℚ) (num_cans : ℕ) : ℚ := total_price / num_cans

def grocery_price_per_can (total_price : ℚ) (num_cans : ℕ) : ℚ := total_price / num_cans

def price_difference_in_cents (price1 : ℚ) (price2 : ℚ) : ℚ := (price2 - price1) * 100

theorem grocery_store_more_expensive
  (bulk_total_price : ℚ)
  (bulk_cans : ℕ)
  (grocery_total_price : ℚ)
  (grocery_cans : ℕ)
  (difference_in_cents : ℚ) :
  bulk_total_price = 12.00 →
  bulk_cans = 48 →
  grocery_total_price = 6.00 →
  grocery_cans = 12 →
  difference_in_cents = 25 →
  price_difference_in_cents (bulk_price_per_can bulk_total_price bulk_cans) 
                            (grocery_price_per_can grocery_total_price grocery_cans) = difference_in_cents := by
  sorry

end NUMINAMATH_GPT_grocery_store_more_expensive_l1390_139063


namespace NUMINAMATH_GPT_f_at_5_l1390_139081

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition

axiom odd_function (f: ℝ → ℝ) : ∀ x : ℝ, f (-x) = -f x
axiom functional_equation (f: ℝ → ℝ) : ∀ x : ℝ, f (x + 1) + f x = 0

theorem f_at_5 : f 5 = 0 :=
by {
  -- Proof to be provided here
  sorry
}

end NUMINAMATH_GPT_f_at_5_l1390_139081


namespace NUMINAMATH_GPT_parabola_equation_focus_l1390_139045

theorem parabola_equation_focus (p : ℝ) (h₀ : p > 0)
  (h₁ : (p / 2 = 2)) : (y^2 = 2 * p * x) :=
  sorry

end NUMINAMATH_GPT_parabola_equation_focus_l1390_139045


namespace NUMINAMATH_GPT_eval_expr_at_values_l1390_139054

variable (x y : ℝ)

def expr := 2 * (3 * x^2 + x * y^2)- 3 * (2 * x * y^2 - x^2) - 10 * x^2

theorem eval_expr_at_values : x = -1 → y = 0.5 → expr x y = 0 :=
by
  intros hx hy
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_eval_expr_at_values_l1390_139054


namespace NUMINAMATH_GPT_problem_l1390_139082

theorem problem (a b : ℤ)
  (h1 : -2022 = -a)
  (h2 : -1 = -b) :
  a + b = 2023 :=
sorry

end NUMINAMATH_GPT_problem_l1390_139082


namespace NUMINAMATH_GPT_intersection_of_sets_l1390_139020

noncomputable def setA : Set ℝ := { x | (x + 2) / (x - 2) ≤ 0 }
noncomputable def setB : Set ℝ := { x | x ≥ 1 }
noncomputable def expectedSet : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem intersection_of_sets : (setA ∩ setB) = expectedSet := by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1390_139020


namespace NUMINAMATH_GPT_unique_n_for_50_percent_mark_l1390_139068

def exam_conditions (n : ℕ) : Prop :=
  let correct_first_20 : ℕ := 15
  let remaining : ℕ := n - 20
  let correct_remaining : ℕ := remaining / 3
  let total_correct : ℕ := correct_first_20 + correct_remaining
  total_correct * 2 = n

theorem unique_n_for_50_percent_mark : ∃! (n : ℕ), exam_conditions n := sorry

end NUMINAMATH_GPT_unique_n_for_50_percent_mark_l1390_139068


namespace NUMINAMATH_GPT_substitution_not_sufficient_for_identity_proof_l1390_139089

theorem substitution_not_sufficient_for_identity_proof {α : Type} (f g : α → α) :
  (∀ x : α, f x = g x) ↔ ¬ (∀ x, f x = g x ↔ (∃ (c : α), f c ≠ g c)) := by
  sorry

end NUMINAMATH_GPT_substitution_not_sufficient_for_identity_proof_l1390_139089


namespace NUMINAMATH_GPT_no_square_has_units_digit_seven_l1390_139038

theorem no_square_has_units_digit_seven :
  ¬ ∃ n : ℕ, n ≤ 9 ∧ (n^2 % 10) = 7 := by
  sorry

end NUMINAMATH_GPT_no_square_has_units_digit_seven_l1390_139038


namespace NUMINAMATH_GPT_triangle_area_is_correct_l1390_139071

-- Defining the vertices of the triangle
def vertexA : ℝ × ℝ := (0, 0)
def vertexB : ℝ × ℝ := (0, 6)
def vertexC : ℝ × ℝ := (8, 10)

-- Define a function to calculate the area of a triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Statement to prove
theorem triangle_area_is_correct : triangle_area vertexA vertexB vertexC = 24.0 := 
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_correct_l1390_139071


namespace NUMINAMATH_GPT_part1_option1_payment_part1_option2_payment_part2_cost_effective_part3_more_cost_effective_l1390_139095

def cost_option1 (x : ℕ) : ℕ :=
  20 * x + 1200

def cost_option2 (x : ℕ) : ℕ :=
  18 * x + 1440

theorem part1_option1_payment (x : ℕ) (h : x > 20) : cost_option1 x = 20 * x + 1200 :=
  by sorry

theorem part1_option2_payment (x : ℕ) (h : x > 20) : cost_option2 x = 18 * x + 1440 :=
  by sorry

theorem part2_cost_effective (x : ℕ) (h : x = 30) : cost_option1 x < cost_option2 x :=
  by sorry

theorem part3_more_cost_effective (x : ℕ) (h : x = 30) : 20 * 80 + 20 * 10 * 9 / 10 = 1780 :=
  by sorry

end NUMINAMATH_GPT_part1_option1_payment_part1_option2_payment_part2_cost_effective_part3_more_cost_effective_l1390_139095


namespace NUMINAMATH_GPT_find_x_values_l1390_139052

def f (x : ℝ) : ℝ := 3 * x^2 - 8

noncomputable def f_inv (y : ℝ) : ℝ := sorry  -- Placeholder for the inverse function

theorem find_x_values:
  ∃ x : ℝ, (f x = f_inv x) ↔ (x = (1 + Real.sqrt 97) / 6 ∨ x = (1 - Real.sqrt 97) / 6) := sorry

end NUMINAMATH_GPT_find_x_values_l1390_139052


namespace NUMINAMATH_GPT_product_sum_abcd_e_l1390_139085

-- Define the individual numbers
def a : ℕ := 12
def b : ℕ := 25
def c : ℕ := 52
def d : ℕ := 21
def e : ℕ := 32

-- Define the sum of the numbers a, b, c, and d
def sum_abcd : ℕ := a + b + c + d

-- Prove that multiplying the sum by e equals 3520
theorem product_sum_abcd_e : sum_abcd * e = 3520 := by
  sorry

end NUMINAMATH_GPT_product_sum_abcd_e_l1390_139085


namespace NUMINAMATH_GPT_steve_average_speed_l1390_139006

-- Define the conditions as constants
def hours1 := 5
def speed1 := 40
def hours2 := 3
def speed2 := 80
def hours3 := 2
def speed3 := 60

-- Define a theorem that calculates average speed and proves the result is 56
theorem steve_average_speed :
  (hours1 * speed1 + hours2 * speed2 + hours3 * speed3) / (hours1 + hours2 + hours3) = 56 := by
  sorry

end NUMINAMATH_GPT_steve_average_speed_l1390_139006


namespace NUMINAMATH_GPT_charcoal_drawings_count_l1390_139013

/-- Thomas' drawings problem
  Thomas has 25 drawings in total.
  14 drawings with colored pencils.
  7 drawings with blending markers.
  The rest drawings are made with charcoal.
  We assert that the number of charcoal drawings is 4.
-/
theorem charcoal_drawings_count 
  (total_drawings : ℕ) 
  (colored_pencil_drawings : ℕ) 
  (marker_drawings : ℕ) :
  total_drawings = 25 →
  colored_pencil_drawings = 14 →
  marker_drawings = 7 →
  total_drawings - (colored_pencil_drawings + marker_drawings) = 4 := 
  by
    sorry

end NUMINAMATH_GPT_charcoal_drawings_count_l1390_139013


namespace NUMINAMATH_GPT_arithmetic_sequence_s10_l1390_139083

noncomputable def arithmetic_sequence_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_s10 (a : ℤ) (d : ℤ)
  (h1 : a + (a + 8 * d) = 18)
  (h4 : a + 3 * d = 7) :
  arithmetic_sequence_sum 10 a d = 100 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_s10_l1390_139083


namespace NUMINAMATH_GPT_sam_original_puppies_count_l1390_139040

theorem sam_original_puppies_count 
  (spotted_puppies_start : ℕ)
  (non_spotted_puppies_start : ℕ)
  (spotted_puppies_given : ℕ)
  (non_spotted_puppies_given : ℕ)
  (spotted_puppies_left : ℕ)
  (non_spotted_puppies_left : ℕ)
  (h1 : spotted_puppies_start = 8)
  (h2 : non_spotted_puppies_start = 5)
  (h3 : spotted_puppies_given = 2)
  (h4 : non_spotted_puppies_given = 3)
  (h5 : spotted_puppies_left = spotted_puppies_start - spotted_puppies_given)
  (h6 : non_spotted_puppies_left = non_spotted_puppies_start - non_spotted_puppies_given)
  (h7 : spotted_puppies_left = 6)
  (h8 : non_spotted_puppies_left = 2) :
  spotted_puppies_start + non_spotted_puppies_start = 13 :=
by
  sorry

end NUMINAMATH_GPT_sam_original_puppies_count_l1390_139040


namespace NUMINAMATH_GPT_min_quotient_l1390_139070

def digits_distinct (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def quotient (a b c : ℕ) : ℚ := 
  (100 * a + 10 * b + c : ℚ) / (a + b + c : ℚ)

theorem min_quotient (a b c : ℕ) (h1 : b > 3) (h2 : c ≠ b) (h3: digits_distinct a b c) : 
  quotient a b c ≥ 19.62 :=
sorry

end NUMINAMATH_GPT_min_quotient_l1390_139070


namespace NUMINAMATH_GPT_wilson_hamburgers_l1390_139003

def hamburger_cost (H : ℕ) := 5 * H
def cola_cost := 6
def discount := 4
def total_cost (H : ℕ) := hamburger_cost H + cola_cost - discount

theorem wilson_hamburgers (H : ℕ) (h : total_cost H = 12) : H = 2 :=
sorry

end NUMINAMATH_GPT_wilson_hamburgers_l1390_139003


namespace NUMINAMATH_GPT_age_of_15th_student_l1390_139097

theorem age_of_15th_student : 
  let average_age_all_students := 15
  let number_of_students := 15
  let average_age_first_group := 13
  let number_of_students_first_group := 5
  let average_age_second_group := 16
  let number_of_students_second_group := 9
  let total_age_all_students := number_of_students * average_age_all_students
  let total_age_first_group := number_of_students_first_group * average_age_first_group
  let total_age_second_group := number_of_students_second_group * average_age_second_group
  total_age_all_students - (total_age_first_group + total_age_second_group) = 16 :=
by
  let average_age_all_students := 15
  let number_of_students := 15
  let average_age_first_group := 13
  let number_of_students_first_group := 5
  let average_age_second_group := 16
  let number_of_students_second_group := 9
  let total_age_all_students := number_of_students * average_age_all_students
  let total_age_first_group := number_of_students_first_group * average_age_first_group
  let total_age_second_group := number_of_students_second_group * average_age_second_group
  sorry

end NUMINAMATH_GPT_age_of_15th_student_l1390_139097


namespace NUMINAMATH_GPT_number_of_passed_candidates_l1390_139011

theorem number_of_passed_candidates :
  ∀ (P F : ℕ),
  (P + F = 500) →
  (P * 80 + F * 15 = 500 * 60) →
  P = 346 :=
by
  intros P F h1 h2
  sorry

end NUMINAMATH_GPT_number_of_passed_candidates_l1390_139011


namespace NUMINAMATH_GPT_base_edge_length_l1390_139067

theorem base_edge_length (x : ℕ) :
  (∃ (x : ℕ), 
    (∀ (sum_edges : ℕ), sum_edges = 6 * x + 48 → sum_edges = 120) →
    x = 12) := 
sorry

end NUMINAMATH_GPT_base_edge_length_l1390_139067


namespace NUMINAMATH_GPT_find_trajectory_l1390_139034

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (y - 1) * (y + 1) / ((x + 1) * (x - 1)) = -1 / 3

theorem find_trajectory (x y : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  trajectory_equation x y → x^2 + 3 * y^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_trajectory_l1390_139034


namespace NUMINAMATH_GPT_apple_cost_l1390_139030

theorem apple_cost (rate_cost : ℕ) (rate_weight total_weight : ℕ) (h_rate : rate_cost = 5) (h_weight : rate_weight = 7) (h_total : total_weight = 21) :
  ∃ total_cost : ℕ, total_cost = 15 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_apple_cost_l1390_139030


namespace NUMINAMATH_GPT_correct_choice_l1390_139084

theorem correct_choice (x y m : ℝ) (h1 : x > y) (h2 : m > 0) : x - y > 0 := by
  sorry

end NUMINAMATH_GPT_correct_choice_l1390_139084


namespace NUMINAMATH_GPT_total_birds_in_marsh_l1390_139073

-- Given conditions
def initial_geese := 58
def doubled_geese := initial_geese * 2
def ducks := 37
def swans := 15
def herons := 22

-- Prove that the total number of birds is 190
theorem total_birds_in_marsh : 
  doubled_geese + ducks + swans + herons = 190 := 
by
  sorry

end NUMINAMATH_GPT_total_birds_in_marsh_l1390_139073


namespace NUMINAMATH_GPT_ratio_of_books_to_pens_l1390_139010

theorem ratio_of_books_to_pens (total_stationery : ℕ) (books : ℕ) (pens : ℕ) 
    (h1 : total_stationery = 400) (h2 : books = 280) (h3 : pens = total_stationery - books) : 
    books / (Nat.gcd books pens) = 7 ∧ pens / (Nat.gcd books pens) = 3 := 
by 
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_ratio_of_books_to_pens_l1390_139010


namespace NUMINAMATH_GPT_remainder_seven_pow_two_thousand_mod_thirteen_l1390_139055

theorem remainder_seven_pow_two_thousand_mod_thirteen :
  7^2000 % 13 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_seven_pow_two_thousand_mod_thirteen_l1390_139055


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l1390_139032

theorem asymptotes_of_hyperbola (a : ℝ) :
  (∃ x y : ℝ, y^2 = 12 * x ∧ (x = 3) ∧ (y = 0)) →
  (a^2 = 9) →
  (∀ b c : ℝ, (b, c) ∈ ({(a, b) | (b = a/3 ∨ b = -a/3)})) :=
by
  intro h_focus_coincides vertex_condition
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l1390_139032


namespace NUMINAMATH_GPT_min_area_circle_equation_l1390_139024

theorem min_area_circle_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 3 / (2 + x) + 3 / (2 + y) = 1) : (x - 4)^2 + (y - 4)^2 = 256 :=
sorry

end NUMINAMATH_GPT_min_area_circle_equation_l1390_139024


namespace NUMINAMATH_GPT_real_roots_a_set_t_inequality_l1390_139035

noncomputable def set_of_a : Set ℝ := {a | -1 ≤ a ∧ a ≤ 7}

theorem real_roots_a_set (x a : ℝ) :
  (∃ x, x^2 - 4 * x + abs (a - 3) = 0) ↔ a ∈ set_of_a := 
by
  sorry

theorem t_inequality (t a : ℝ) (h : ∀ a ∈ set_of_a, t^2 - 2 * a * t + 12 < 0) :
  3 < t ∧ t < 4 := 
by
  sorry

end NUMINAMATH_GPT_real_roots_a_set_t_inequality_l1390_139035


namespace NUMINAMATH_GPT_no_intersection_at_roots_l1390_139072

theorem no_intersection_at_roots {f g : ℝ → ℝ} (h : ∀ x, f x = x ∧ g x = x - 3) :
  ¬ (∃ x, (x = 0 ∨ x = 3) ∧ (f x = g x)) :=
by
  intros 
  sorry

end NUMINAMATH_GPT_no_intersection_at_roots_l1390_139072


namespace NUMINAMATH_GPT_age_ratio_l1390_139051

noncomputable def ratio_of_ages (A M : ℕ) : ℕ × ℕ :=
if A = 30 ∧ (A + 15 + (M + 15)) / 2 = 50 then
  (A / Nat.gcd A M, M / Nat.gcd A M)
else
  (0, 0)

theorem age_ratio :
  (45 + (40 + 15)) / 2 = 50 → 30 = 3 * 10 ∧ 40 = 4 * 10 →
  ratio_of_ages 30 40 = (3, 4) :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l1390_139051


namespace NUMINAMATH_GPT_inverse_function_log_base_two_l1390_139025

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_log_base_two (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : f a (a^2) = a) : f a = fun x => Real.log x / Real.log 2 := 
by
  sorry

end NUMINAMATH_GPT_inverse_function_log_base_two_l1390_139025


namespace NUMINAMATH_GPT_solve_for_x_l1390_139079

theorem solve_for_x (x : ℚ) :  (1/2) * (12 * x + 3) = 3 * x + 2 → x = 1/6 := by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1390_139079


namespace NUMINAMATH_GPT_top_and_bottom_edges_same_color_l1390_139004

-- Define the vertices for top and bottom pentagonal faces
inductive Vertex
| A1 | A2 | A3 | A4 | A5
| B1 | B2 | B3 | B4 | B5

-- Define the edges
inductive Edge : Type
| TopEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) (h2 : v2 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) : Edge
| BottomEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) (h2 : v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) : Edge
| SideEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) (h2 : v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) : Edge

-- Define colors
inductive Color
| Red | Blue

-- Define a function that assigns a color to each edge
def edgeColor : Edge → Color := sorry

-- Define a function that checks if a triangle is monochromatic
def isMonochromatic (e1 e2 e3 : Edge) : Prop :=
  edgeColor e1 = edgeColor e2 ∧ edgeColor e2 = edgeColor e3

-- Define our main theorem statement
theorem top_and_bottom_edges_same_color (h : ∀ v1 v2 v3 : Vertex, ¬ isMonochromatic (Edge.TopEdge v1 v2 sorry sorry) (Edge.SideEdge v1 v3 sorry sorry) (Edge.BottomEdge v2 v3 sorry sorry)) : 
  (∀ (v1 v2 : Vertex), v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5] → v2 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5] → edgeColor (Edge.TopEdge v1 v2 sorry sorry) = edgeColor (Edge.TopEdge Vertex.A1 Vertex.A2 sorry sorry)) ∧
  (∀ (v1 v2 : Vertex), v1 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5] → v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5] → edgeColor (Edge.BottomEdge v1 v2 sorry sorry) = edgeColor (Edge.BottomEdge Vertex.B1 Vertex.B2 sorry sorry)) :=
sorry

end NUMINAMATH_GPT_top_and_bottom_edges_same_color_l1390_139004


namespace NUMINAMATH_GPT_xiaoxian_mistake_xiaoxuan_difference_l1390_139044

-- Define the initial expressions and conditions
def original_expr := (-9) * 3 - 5
def xiaoxian_expr (x : Int) := (-9) * 3 - x
def xiaoxuan_expr := (-9) / 3 - 5

-- Given conditions
variable (result_xiaoxian : Int)
variable (result_original : Int)

-- Proof statement
theorem xiaoxian_mistake (hx : xiaoxian_expr 2 = -29) : 
  xiaoxian_expr 5 = result_xiaoxian := sorry

theorem xiaoxuan_difference : 
  abs (xiaoxuan_expr - original_expr) = 24 := sorry

end NUMINAMATH_GPT_xiaoxian_mistake_xiaoxuan_difference_l1390_139044


namespace NUMINAMATH_GPT_universal_quantifiers_and_propositions_l1390_139077

-- Definitions based on conditions
def universal_quantifiers_phrases := ["for all", "for any"]
def universal_quantifier_symbol := "∀"
def universal_proposition := "Universal Proposition"
def universal_proposition_representation := "∀ x ∈ M, p(x)"

-- Main theorem
theorem universal_quantifiers_and_propositions :
  universal_quantifiers_phrases = ["for all", "for any"]
  ∧ universal_quantifier_symbol = "∀"
  ∧ universal_proposition = "Universal Proposition"
  ∧ universal_proposition_representation = "∀ x ∈ M, p(x)" :=
by
  sorry

end NUMINAMATH_GPT_universal_quantifiers_and_propositions_l1390_139077


namespace NUMINAMATH_GPT_inequality_a3_b3_c3_l1390_139098

theorem inequality_a3_b3_c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 + 3 * a * b * c > a * b * (a + b) + b * c * (b + c) + a * c * (a + c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_a3_b3_c3_l1390_139098


namespace NUMINAMATH_GPT_percentage_of_items_sold_l1390_139031

theorem percentage_of_items_sold (total_items price_per_item discount_rate debt creditors_balance remaining_balance : ℕ)
  (H1 : total_items = 2000)
  (H2 : price_per_item = 50)
  (H3 : discount_rate = 80)
  (H4 : debt = 15000)
  (H5 : remaining_balance = 3000) :
  (total_items * (price_per_item - (price_per_item * discount_rate / 100)) + remaining_balance = debt + remaining_balance) →
  (remaining_balance / (price_per_item - (price_per_item * discount_rate / 100)) / total_items * 100 = 90) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_items_sold_l1390_139031


namespace NUMINAMATH_GPT_roots_polynomial_equation_l1390_139094

noncomputable def rootsEquation (x y : ℝ) := x + y = 10 ∧ |x - y| = 12

theorem roots_polynomial_equation : ∃ (x y : ℝ), rootsEquation x y ∧ (x^2 - 10 * x - 11 = 0) := sorry

end NUMINAMATH_GPT_roots_polynomial_equation_l1390_139094


namespace NUMINAMATH_GPT_cubic_sum_of_roots_l1390_139058

theorem cubic_sum_of_roots :
  ∀ (r s : ℝ), (r + s = 5) → (r * s = 6) → (r^3 + s^3 = 35) :=
by
  intros r s h₁ h₂
  sorry

end NUMINAMATH_GPT_cubic_sum_of_roots_l1390_139058


namespace NUMINAMATH_GPT_parallelogram_side_length_l1390_139088

-- We need trigonometric functions and operations with real numbers.
open Real

theorem parallelogram_side_length (s : ℝ) 
  (h_side_lengths : s > 0 ∧ 3 * s > 0) 
  (h_angle : sin (30 / 180 * π) = 1 / 2) 
  (h_area : 3 * s * (s * sin (30 / 180 * π)) = 9 * sqrt 3) :
  s = 3 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_side_length_l1390_139088


namespace NUMINAMATH_GPT_fractional_eq_range_m_l1390_139049

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end NUMINAMATH_GPT_fractional_eq_range_m_l1390_139049


namespace NUMINAMATH_GPT_pastries_left_l1390_139047

def pastries_baked : ℕ := 4 + 29
def pastries_sold : ℕ := 9

theorem pastries_left : pastries_baked - pastries_sold = 24 :=
by
  -- assume pastries_baked = 33
  -- assume pastries_sold = 9
  -- prove 33 - 9 = 24
  sorry

end NUMINAMATH_GPT_pastries_left_l1390_139047


namespace NUMINAMATH_GPT_percentage_error_formula_l1390_139007

noncomputable def percentage_error_in_area (a b : ℝ) (x y : ℝ) :=
  let actual_area := a * b
  let measured_area := a * (1 + x / 100) * b * (1 + y / 100)
  let error_percentage := ((measured_area - actual_area) / actual_area) * 100
  error_percentage

theorem percentage_error_formula (a b x y : ℝ) :
  percentage_error_in_area a b x y = x + y + (x * y / 100) :=
by
  sorry

end NUMINAMATH_GPT_percentage_error_formula_l1390_139007


namespace NUMINAMATH_GPT_leah_daily_savings_l1390_139033

theorem leah_daily_savings 
  (L : ℝ)
  (h1 : 0.25 * 24 = 6)
  (h2 : ∀ (L : ℝ), (L * 20) = 20 * L)
  (h3 : ∀ (L : ℝ), 2 * L * 12 = 24 * L)
  (h4 :  6 + 20 * L + 24 * L = 28) 
: L = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_leah_daily_savings_l1390_139033


namespace NUMINAMATH_GPT_sum_of_transformed_roots_l1390_139023

theorem sum_of_transformed_roots (α β γ : ℂ) (h₁ : α^3 - α + 1 = 0) (h₂ : β^3 - β + 1 = 0) (h₃ : γ^3 - γ + 1 = 0) :
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_transformed_roots_l1390_139023


namespace NUMINAMATH_GPT_which_point_is_in_fourth_quadrant_l1390_139060

def point (x: ℝ) (y: ℝ) : Prop := x > 0 ∧ y < 0

theorem which_point_is_in_fourth_quadrant :
  point 5 (-4) :=
by {
  -- proofs for each condition can be added,
  sorry
}

end NUMINAMATH_GPT_which_point_is_in_fourth_quadrant_l1390_139060


namespace NUMINAMATH_GPT_definite_integral_example_l1390_139017

theorem definite_integral_example : ∫ x in (0 : ℝ)..(π/2), 2 * x = π^2 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_definite_integral_example_l1390_139017


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1390_139064

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x * (x - 1) < 0 → x < 1) ∧ ¬(x < 1 → x * (x - 1) < 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1390_139064


namespace NUMINAMATH_GPT_inverse_value_l1390_139009

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_value :
  g (-3) = -103 :=
by
  sorry

end NUMINAMATH_GPT_inverse_value_l1390_139009


namespace NUMINAMATH_GPT_percentage_of_female_employees_l1390_139008

theorem percentage_of_female_employees (E : ℕ) (hE : E = 1400) 
  (pct_computer_literate : ℚ) (hpct : pct_computer_literate = 0.62)
  (female_computer_literate : ℕ) (hfcl : female_computer_literate = 588)
  (pct_male_computer_literate : ℚ) (hmcl : pct_male_computer_literate = 0.5) :
  100 * (840 / 1400) = 60 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_female_employees_l1390_139008


namespace NUMINAMATH_GPT_skittles_students_division_l1390_139012

theorem skittles_students_division (n : ℕ) (h1 : 27 % 3 = 0) (h2 : 27 / 3 = n) : n = 9 := by
  sorry

end NUMINAMATH_GPT_skittles_students_division_l1390_139012


namespace NUMINAMATH_GPT_surface_area_ratio_l1390_139075

theorem surface_area_ratio (x : ℝ) (hx : x > 0) :
  let SA1 := 6 * (4 * x) ^ 2
  let SA2 := 6 * x ^ 2
  (SA1 / SA2) = 16 := by
  sorry

end NUMINAMATH_GPT_surface_area_ratio_l1390_139075


namespace NUMINAMATH_GPT_inequality_example_l1390_139037

variable {a b c : ℝ} -- Declare a, b, c as real numbers

theorem inequality_example
  (ha : 0 < a)  -- Condition: a is positive
  (hb : 0 < b)  -- Condition: b is positive
  (hc : 0 < c) :  -- Condition: c is positive
  (ab * (a + b) + ac * (a + c) + bc * (b + c)) / (abc) ≥ 6 := 
sorry  -- Proof is skipped

end NUMINAMATH_GPT_inequality_example_l1390_139037


namespace NUMINAMATH_GPT_perpendicular_lines_sufficient_l1390_139018

noncomputable def line1_slope (a : ℝ) : ℝ :=
-((a + 2) / (3 * a))

noncomputable def line2_slope (a : ℝ) : ℝ :=
-((a - 2) / (a + 2))

theorem perpendicular_lines_sufficient (a : ℝ) (h : a = -2) :
  line1_slope a * line2_slope a = -1 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_sufficient_l1390_139018


namespace NUMINAMATH_GPT_kittens_and_mice_count_l1390_139096

theorem kittens_and_mice_count :
  let children := 12
  let baskets_per_child := 3
  let cats_per_basket := 1
  let kittens_per_cat := 12
  let mice_per_kitten := 4
  let total_kittens := children * baskets_per_child * cats_per_basket * kittens_per_cat
  let total_mice := total_kittens * mice_per_kitten
  total_kittens + total_mice = 2160 :=
by
  sorry

end NUMINAMATH_GPT_kittens_and_mice_count_l1390_139096


namespace NUMINAMATH_GPT_right_triangle_area_l1390_139091

theorem right_triangle_area (a b : ℝ) (H₁ : a = 3) (H₂ : b = 5) : 
  1 / 2 * a * b = 7.5 := by
  rw [H₁, H₂]
  norm_num

end NUMINAMATH_GPT_right_triangle_area_l1390_139091


namespace NUMINAMATH_GPT_factorization_of_polynomial_l1390_139092

theorem factorization_of_polynomial : ∀ x : ℝ, x^2 - x - 42 = (x + 6) * (x - 7) :=
by
  sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l1390_139092


namespace NUMINAMATH_GPT_speed_of_first_train_l1390_139014

-- Define the problem conditions
def distance_between_stations : ℝ := 20
def speed_of_second_train : ℝ := 25
def meet_time : ℝ := 8
def start_time_first_train : ℝ := 7
def start_time_second_train : ℝ := 8
def travel_time_first_train : ℝ := meet_time - start_time_first_train

-- The actual proof statement in Lean
theorem speed_of_first_train : ∀ (v : ℝ),
  v * travel_time_first_train = distance_between_stations → v = 20 :=
by
  intro v
  intro h
  sorry

end NUMINAMATH_GPT_speed_of_first_train_l1390_139014


namespace NUMINAMATH_GPT_min_value_ineq_l1390_139019

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem min_value_ineq (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : f a = f b) :
  (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_min_value_ineq_l1390_139019


namespace NUMINAMATH_GPT_multiplication_333_111_l1390_139076

theorem multiplication_333_111: 333 * 111 = 36963 := 
by 
sorry

end NUMINAMATH_GPT_multiplication_333_111_l1390_139076


namespace NUMINAMATH_GPT_distance_traveled_on_foot_l1390_139066

theorem distance_traveled_on_foot (x y : ℝ) : x + y = 61 ∧ (x / 4 + y / 9 = 9) → x = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_traveled_on_foot_l1390_139066


namespace NUMINAMATH_GPT_isosceles_right_triangle_solution_l1390_139001

theorem isosceles_right_triangle_solution (a b : ℝ) (area : ℝ) 
  (h1 : a = b) (h2 : XY = a * Real.sqrt 2) (h3 : area = (1/2) * a * b) (h4 : area = 36) : 
  XY = 12 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_solution_l1390_139001


namespace NUMINAMATH_GPT_roofing_cost_per_foot_l1390_139042

theorem roofing_cost_per_foot:
  ∀ (total_feet needed_feet free_feet : ℕ) (total_cost : ℕ),
  needed_feet = 300 →
  free_feet = 250 →
  total_cost = 400 →
  needed_feet - free_feet = 50 →
  total_cost / (needed_feet - free_feet) = 8 :=
by sorry

end NUMINAMATH_GPT_roofing_cost_per_foot_l1390_139042


namespace NUMINAMATH_GPT_inequality_and_equality_equality_condition_l1390_139062

theorem inequality_and_equality (a b : ℕ) (ha : a > 1) (hb : b > 2) : a^b + 1 ≥ b * (a + 1) :=
by sorry

theorem equality_condition (a b : ℕ) : a = 2 ∧ b = 3 → a^b + 1 = b * (a + 1) :=
by
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_inequality_and_equality_equality_condition_l1390_139062


namespace NUMINAMATH_GPT_sum_xyz_eq_two_l1390_139057

-- Define the variables x, y, and z to be real numbers
variables (x y z : ℝ)

-- Given condition
def condition : Prop :=
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0

-- The theorem to prove
theorem sum_xyz_eq_two (h : condition x y z) : x + y + z = 2 :=
sorry

end NUMINAMATH_GPT_sum_xyz_eq_two_l1390_139057


namespace NUMINAMATH_GPT_Malou_average_is_correct_l1390_139002

def quiz1_score : ℕ := 91
def quiz2_score : ℕ := 90
def quiz3_score : ℕ := 92
def total_score : ℕ := quiz1_score + quiz2_score + quiz3_score
def number_of_quizzes : ℕ := 3

def Malous_average_score : ℕ := total_score / number_of_quizzes

theorem Malou_average_is_correct : Malous_average_score = 91 := by
  sorry

end NUMINAMATH_GPT_Malou_average_is_correct_l1390_139002


namespace NUMINAMATH_GPT_xy_cubed_identity_l1390_139069

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_xy_cubed_identity_l1390_139069


namespace NUMINAMATH_GPT_cost_of_shoes_l1390_139016

-- Define the conditions
def saved : Nat := 30
def earn_per_lawn : Nat := 5
def lawns_per_weekend : Nat := 3
def weekends_needed : Nat := 6

-- Prove the total amount saved is the cost of the shoes
theorem cost_of_shoes : saved + (earn_per_lawn * lawns_per_weekend * weekends_needed) = 120 := by
  sorry

end NUMINAMATH_GPT_cost_of_shoes_l1390_139016


namespace NUMINAMATH_GPT_simplified_expression_evaluates_to_2_l1390_139005

-- Definitions based on given conditions:
def x := 2 -- where x = (1/2)^(-1)
def y := 1 -- where y = (-2023)^0

-- Main statement to prove:
theorem simplified_expression_evaluates_to_2 :
  ((2 * x - y) / (x + y) - (x * x - 2 * x * y + y * y) / (x * x - y * y)) / (x - y) / (x + y) = 2 :=
by
  sorry

end NUMINAMATH_GPT_simplified_expression_evaluates_to_2_l1390_139005


namespace NUMINAMATH_GPT_complex_expression_l1390_139029

theorem complex_expression (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) :
  (x^9 + y^9) / (x^9 - y^9) + (x^9 - y^9) / (x^9 + y^9) = 3 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_complex_expression_l1390_139029


namespace NUMINAMATH_GPT_part1_part2_l1390_139056

def traditional_chinese_paintings : ℕ := 6
def oil_paintings : ℕ := 4
def watercolor_paintings : ℕ := 5

theorem part1 :
  traditional_chinese_paintings * oil_paintings * watercolor_paintings = 120 :=
by
  sorry

theorem part2 :
  (traditional_chinese_paintings * oil_paintings) + 
  (traditional_chinese_paintings * watercolor_paintings) + 
  (oil_paintings * watercolor_paintings) = 74 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1390_139056


namespace NUMINAMATH_GPT_range_of_m_l1390_139065

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + m * x + 2 * m - 3 ≥ 0) ↔ 2 ≤ m ∧ m ≤ 6 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1390_139065


namespace NUMINAMATH_GPT_quad_intersects_x_axis_l1390_139090

theorem quad_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x ^ 2 - 7 * x - 7 = 0) ↔ (k ≥ -7 / 4 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_GPT_quad_intersects_x_axis_l1390_139090


namespace NUMINAMATH_GPT_metallic_sphere_radius_l1390_139027

theorem metallic_sphere_radius 
  (r_wire : ℝ)
  (h_wire : ℝ)
  (r_sphere : ℝ) 
  (V_sphere : ℝ)
  (V_wire : ℝ)
  (h_wire_eq : h_wire = 16)
  (r_wire_eq : r_wire = 12)
  (V_wire_eq : V_wire = π * r_wire^2 * h_wire)
  (V_sphere_eq : V_sphere = (4/3) * π * r_sphere^3)
  (volume_eq : V_sphere = V_wire) :
  r_sphere = 12 :=
by
  sorry

end NUMINAMATH_GPT_metallic_sphere_radius_l1390_139027


namespace NUMINAMATH_GPT_unique_integer_for_P5_l1390_139046

-- Define the polynomial P with integer coefficients
variable (P : ℤ → ℤ)

-- The conditions given in the problem
variable (x1 x2 x3 : ℤ)
variable (Hx1 : P x1 = 1)
variable (Hx2 : P x2 = 2)
variable (Hx3 : P x3 = 3)

-- The main theorem to prove
theorem unique_integer_for_P5 {P : ℤ → ℤ} {x1 x2 x3 : ℤ}
(Hx1 : P x1 = 1) (Hx2 : P x2 = 2) (Hx3 : P x3 = 3) :
  ∃!(x : ℤ), P x = 5 := sorry

end NUMINAMATH_GPT_unique_integer_for_P5_l1390_139046


namespace NUMINAMATH_GPT_ethan_days_worked_per_week_l1390_139074

-- Define the conditions
def hourly_wage : ℕ := 18
def hours_per_day : ℕ := 8
def total_earnings : ℕ := 3600
def weeks_worked : ℕ := 5

-- Compute derived values
def daily_earnings : ℕ := hourly_wage * hours_per_day
def weekly_earnings : ℕ := total_earnings / weeks_worked

-- Define the proposition to be proved
theorem ethan_days_worked_per_week : ∃ d: ℕ, d * daily_earnings = weekly_earnings ∧ d = 5 :=
by
  use 5
  simp [daily_earnings, weekly_earnings]
  sorry

end NUMINAMATH_GPT_ethan_days_worked_per_week_l1390_139074
