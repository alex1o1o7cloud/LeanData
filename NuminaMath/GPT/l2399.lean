import Mathlib

namespace NUMINAMATH_GPT_square_no_remainder_5_mod_9_l2399_239984

theorem square_no_remainder_5_mod_9 (n : ℤ) : (n^2 % 9 ≠ 5) :=
by sorry

end NUMINAMATH_GPT_square_no_remainder_5_mod_9_l2399_239984


namespace NUMINAMATH_GPT_john_total_spent_l2399_239932

noncomputable def computer_cost : ℝ := 1500
noncomputable def peripherals_cost : ℝ := (1 / 4) * computer_cost
noncomputable def base_video_card_cost : ℝ := 300
noncomputable def upgraded_video_card_cost : ℝ := 2.5 * base_video_card_cost
noncomputable def discount_on_video_card : ℝ := 0.12 * upgraded_video_card_cost
noncomputable def video_card_cost_after_discount : ℝ := upgraded_video_card_cost - discount_on_video_card
noncomputable def sales_tax_on_peripherals : ℝ := 0.05 * peripherals_cost
noncomputable def total_spent : ℝ := computer_cost + peripherals_cost + video_card_cost_after_discount + sales_tax_on_peripherals

theorem john_total_spent : total_spent = 2553.75 := by
  sorry

end NUMINAMATH_GPT_john_total_spent_l2399_239932


namespace NUMINAMATH_GPT_probability_interval_l2399_239931

theorem probability_interval (P_A P_B p : ℝ) (hP_A : P_A = 2 / 3) (hP_B : P_B = 3 / 5) :
  4 / 15 ≤ p ∧ p ≤ 3 / 5 := sorry

end NUMINAMATH_GPT_probability_interval_l2399_239931


namespace NUMINAMATH_GPT_deck_length_is_30_l2399_239977

theorem deck_length_is_30
  (x : ℕ)
  (h1 : ∀ a : ℕ, a = 40 * x)
  (h2 : ∀ b : ℕ, b = 3 * a + 1 * a ∧ b = 4800) :
  x = 30 := by
  sorry

end NUMINAMATH_GPT_deck_length_is_30_l2399_239977


namespace NUMINAMATH_GPT_value_is_50_cents_l2399_239964

-- Define Leah's total number of coins and the condition on the number of nickels and pennies.
variables (p n : ℕ)

-- Leah has a total of 18 coins
def total_coins : Prop := n + p = 18

-- Condition for nickels and pennies
def condition : Prop := p = n + 2

-- Calculate the total value of Leah's coins and check if it equals 50 cents
def total_value : ℕ := 5 * n + p

-- Proposition stating that under given conditions, total value is 50 cents
theorem value_is_50_cents (h1 : total_coins p n) (h2 : condition p n) :
  total_value p n = 50 := sorry

end NUMINAMATH_GPT_value_is_50_cents_l2399_239964


namespace NUMINAMATH_GPT_percentage_y_less_than_x_l2399_239921

theorem percentage_y_less_than_x (x y : ℝ) (h : x = 4 * y) : (x - y) / x * 100 = 75 := by
  sorry

end NUMINAMATH_GPT_percentage_y_less_than_x_l2399_239921


namespace NUMINAMATH_GPT_certain_number_value_l2399_239901

theorem certain_number_value (x : ℕ) (p n : ℕ) (hp : Nat.Prime p) (hx : x = 44) (h : x / (n * p) = 2) : n = 2 := 
by
  sorry

end NUMINAMATH_GPT_certain_number_value_l2399_239901


namespace NUMINAMATH_GPT_domain_of_f_l2399_239939

noncomputable def f (x : ℝ) : ℝ := (5 * x + 2) / Real.sqrt (2 * x - 10)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f y = f x} = {x : ℝ | x > 5} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2399_239939


namespace NUMINAMATH_GPT_compare_sqrt_expression_l2399_239989

theorem compare_sqrt_expression : 2 * Real.sqrt 3 < 3 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_compare_sqrt_expression_l2399_239989


namespace NUMINAMATH_GPT_solve_x_l2399_239988

theorem solve_x :
  ∃ x : ℝ, 2.5 * ( ( x * 0.48 * 2.50 ) / ( 0.12 * 0.09 * 0.5 ) ) = 2000.0000000000002 ∧ x = 3.6 :=
by sorry

end NUMINAMATH_GPT_solve_x_l2399_239988


namespace NUMINAMATH_GPT_problem_1_problem_2_l2399_239951

open Set

variables (a x : ℝ)

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a ^ 2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem problem_1 (a : ℝ) (ha : a = 1) : 
  {x : ℝ | x^2 - 4 * a * x + 3 * a ^ 2 < 0} ∩ {x : ℝ | (x - 3) / (x - 2) ≤ 0} = Ioo 2 3 :=
sorry

theorem problem_2 (a : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 4 * a * x + 3 * a ^ 2 < 0) → ¬((x - 3) / (x - 2) ≤ 0)) →
  (∃ x : ℝ, ¬((x - 3) / (x - 2) ≤ 0) → ¬(x^2 - 4 * a * x + 3 * a ^ 2 < 0)) →
  1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2399_239951


namespace NUMINAMATH_GPT_delta_value_l2399_239944

noncomputable def delta : ℝ :=
  Real.arccos (
    (Finset.range 3600).sum (fun k => Real.sin ((2539 + k) * Real.pi / 180)) ^ Real.cos (2520 * Real.pi / 180) +
    (Finset.range 3599).sum (fun k => Real.cos ((2521 + k) * Real.pi / 180)) +
    Real.cos (6120 * Real.pi / 180)
  )

theorem delta_value : delta = 71 :=
by
  sorry

end NUMINAMATH_GPT_delta_value_l2399_239944


namespace NUMINAMATH_GPT_cost_of_adult_ticket_l2399_239965

theorem cost_of_adult_ticket (c : ℝ) 
  (h₁ : 2 * (c + 6) + 3 * c = 77)
  : c + 6 = 19 :=
sorry

end NUMINAMATH_GPT_cost_of_adult_ticket_l2399_239965


namespace NUMINAMATH_GPT_range_of_a_l2399_239909

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x - Real.exp (-x) + Real.log (x + Real.sqrt (x^2 + 1))

theorem range_of_a
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f (x^2 + 2) + f (-2 * a * x) ≥ 0) :
  -3/2 ≤ a ∧ a ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2399_239909


namespace NUMINAMATH_GPT_rectangle_area_l2399_239919

theorem rectangle_area (y : ℝ) (w : ℝ) : 
  (3 * w) ^ 2 + w ^ 2 = y ^ 2 → 
  3 * w * w = (3 / 10) * y ^ 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rectangle_area_l2399_239919


namespace NUMINAMATH_GPT_find_k_l2399_239929

theorem find_k (k : ℝ) (x₁ x₂ : ℝ)
  (h : x₁^2 + (2 * k - 1) * x₁ + k^2 - 1 = 0)
  (h' : x₂^2 + (2 * k - 1) * x₂ + k^2 - 1 = 0)
  (hx : x₁ ≠ x₂)
  (cond : x₁^2 + x₂^2 = 19) : k = -2 :=
sorry

end NUMINAMATH_GPT_find_k_l2399_239929


namespace NUMINAMATH_GPT_solve_f_lt_zero_solve_f_plus_4_abs_x_minus_4_gt_m_l2399_239905

def f (x : ℝ) : ℝ := |3 * x + 1| - |x - 4|

theorem solve_f_lt_zero :
  { x : ℝ | f x < 0 } = { x : ℝ | -5 / 2 < x ∧ x < 3 / 4 } := 
sorry

theorem solve_f_plus_4_abs_x_minus_4_gt_m (m : ℝ) :
  (∀ x : ℝ, f x + 4 * |x - 4| > m) → m < 15 :=
sorry

end NUMINAMATH_GPT_solve_f_lt_zero_solve_f_plus_4_abs_x_minus_4_gt_m_l2399_239905


namespace NUMINAMATH_GPT_find_a_l2399_239959

theorem find_a (a : ℤ) (h : ∃ x1 x2 : ℤ, (x - x1) * (x - x2) = (x - a) * (x - 8) - 1) : a = 8 :=
sorry

end NUMINAMATH_GPT_find_a_l2399_239959


namespace NUMINAMATH_GPT_find_constant_term_l2399_239902

theorem find_constant_term (q' : ℝ → ℝ) (c : ℝ) (h1 : ∀ q : ℝ, q' q = 3 * q - c)
  (h2 : q' (q' 7) = 306) : c = 252 :=
by
  sorry

end NUMINAMATH_GPT_find_constant_term_l2399_239902


namespace NUMINAMATH_GPT_invisible_dots_48_l2399_239911

theorem invisible_dots_48 (visible : Multiset ℕ) (hv : visible = [1, 2, 3, 3, 4, 5, 6, 6, 6]) :
  let total_dots := 4 * (1 + 2 + 3 + 4 + 5 + 6)
  let visible_sum := visible.sum
  total_dots - visible_sum = 48 :=
by
  sorry

end NUMINAMATH_GPT_invisible_dots_48_l2399_239911


namespace NUMINAMATH_GPT_distance_covered_downstream_l2399_239928

noncomputable def speed_in_still_water := 16 -- km/hr
noncomputable def speed_of_stream := 5 -- km/hr
noncomputable def time_taken := 5 -- hours
noncomputable def effective_speed_downstream := speed_in_still_water + speed_of_stream -- km/hr

theorem distance_covered_downstream :
  (effective_speed_downstream * time_taken = 105) :=
by
  sorry

end NUMINAMATH_GPT_distance_covered_downstream_l2399_239928


namespace NUMINAMATH_GPT_smallest_n_for_divisibility_l2399_239992

theorem smallest_n_for_divisibility (n : ℕ) : 
  (∀ m, m > 0 → (315^2 - m^2) ∣ (315^3 - m^3) → m ≥ n) → 
  (315^2 - n^2) ∣ (315^3 - n^3) → 
  n = 90 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_divisibility_l2399_239992


namespace NUMINAMATH_GPT_restore_original_expression_l2399_239950

-- Define the altered product and correct restored products
def original_expression_1 := 4 * 5 * 4 * 7 * 4
def original_expression_2 := 4 * 7 * 4 * 5 * 4
def altered_product := 2247
def corrected_product := 2240

-- Statement that proves the corrected restored product given the altered product
theorem restore_original_expression :
  (4 * 5 * 4 * 7 * 4 = corrected_product ∨ 4 * 7 * 4 * 5 * 4 = corrected_product) :=
sorry

end NUMINAMATH_GPT_restore_original_expression_l2399_239950


namespace NUMINAMATH_GPT_problem_l2399_239997

theorem problem (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -6) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 := 
by sorry

end NUMINAMATH_GPT_problem_l2399_239997


namespace NUMINAMATH_GPT_value_f2_f5_l2399_239953

variable {α : Type} [AddGroup α]

noncomputable def f : α → ℤ := sorry

axiom func_eq : ∀ x y, f (x + y) = f x + f y + 7 * x * y + 4

axiom f_one : f 1 = 4

theorem value_f2_f5 :
  f 2 + f 5 = 125 :=
sorry

end NUMINAMATH_GPT_value_f2_f5_l2399_239953


namespace NUMINAMATH_GPT_plane_distance_l2399_239904

theorem plane_distance (n : ℕ) : n % 45 = 0 ∧ (n / 10) % 100 = 39 ∧ n <= 5000 → n = 1395 := 
by
  sorry

end NUMINAMATH_GPT_plane_distance_l2399_239904


namespace NUMINAMATH_GPT_find_y_l2399_239982

theorem find_y (x y : ℤ) (q : ℤ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x = q * y + 6) (h4 : (x : ℚ) / y = 96.15) : y = 40 :=
sorry

end NUMINAMATH_GPT_find_y_l2399_239982


namespace NUMINAMATH_GPT_division_of_cookies_l2399_239970

theorem division_of_cookies (n p : Nat) (h1 : n = 24) (h2 : p = 6) : n / p = 4 :=
by sorry

end NUMINAMATH_GPT_division_of_cookies_l2399_239970


namespace NUMINAMATH_GPT_correct_operation_l2399_239958

theorem correct_operation (a m : ℝ) :
  ¬(a^5 / a^10 = a^2) ∧ 
  (-2 * a^3)^2 = 4 * a^6 ∧ 
  ¬((1 / (2 * m)) - (1 / m) = (1 / m)) ∧ 
  ¬(a^4 + a^3 = a^7) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l2399_239958


namespace NUMINAMATH_GPT_triangle_equilateral_l2399_239986

-- Assume we are given side lengths a, b, and c of a triangle and angles A, B, and C in radians.
variables {a b c : ℝ} {A B C : ℝ}

-- We'll use the assumption that (a + b + c) * (b + c - a) = 3 * b * c and sin A = 2 * sin B * cos C.
axiom triangle_condition1 : (a + b + c) * (b + c - a) = 3 * b * c
axiom triangle_condition2 : Real.sin A = 2 * Real.sin B * Real.cos C

-- We need to prove that the triangle is equilateral.
theorem triangle_equilateral : (a = b) ∧ (b = c) ∧ (c = a) := by
  sorry

end NUMINAMATH_GPT_triangle_equilateral_l2399_239986


namespace NUMINAMATH_GPT_condition_sufficient_not_necessary_l2399_239907

theorem condition_sufficient_not_necessary (x : ℝ) : (0 < x ∧ x < 5) → (|x - 2| < 3) ∧ (¬ ((|x - 2| < 3) → (0 < x ∧ x < 5))) :=
by
  sorry

end NUMINAMATH_GPT_condition_sufficient_not_necessary_l2399_239907


namespace NUMINAMATH_GPT_probability_non_black_ball_l2399_239957

/--
Given the odds of drawing a black ball as 5:3,
prove that the probability of drawing a non-black ball from the bag is 3/8.
-/
theorem probability_non_black_ball (n_black n_non_black : ℕ) (h : n_black = 5) (h' : n_non_black = 3) :
  (n_non_black : ℚ) / (n_black + n_non_black) = 3 / 8 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_probability_non_black_ball_l2399_239957


namespace NUMINAMATH_GPT_train_length_l2399_239927

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_conversion_factor : ℝ) (approx_length : ℝ) :
  speed_km_hr = 60 → time_seconds = 6 → speed_conversion_factor = (1000 / 3600) → approx_length = 100.02 →
  speed_km_hr * speed_conversion_factor * time_seconds = approx_length :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_train_length_l2399_239927


namespace NUMINAMATH_GPT_number_of_triples_l2399_239981

theorem number_of_triples : 
  ∃ n : ℕ, 
  n = 2 ∧
  ∀ (a b c : ℕ), 
    (2 ≤ a ∧ a ≤ b ∧ b ≤ c) →
    (a * b * c = 4 * (a * b + b * c + c * a)) →
    n = 2 :=
sorry

end NUMINAMATH_GPT_number_of_triples_l2399_239981


namespace NUMINAMATH_GPT_center_of_symmetry_l2399_239968

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * Real.tan (-7 * x + (Real.pi / 3))

theorem center_of_symmetry : f (Real.pi / 21) = 0 :=
by
  -- Mathematical proof goes here, skipping with sorry.
  sorry

end NUMINAMATH_GPT_center_of_symmetry_l2399_239968


namespace NUMINAMATH_GPT_parallel_lines_sufficient_not_necessary_l2399_239999

theorem parallel_lines_sufficient_not_necessary (a : ℝ) :
  ((a = 3) → (∀ x y : ℝ, (a * x + 2 * y + 1 = 0) → (3 * x + (a - 1) * y - 2 = 0)) ∧ 
  (∀ x y : ℝ, (a * x + 2 * y + 1 = 0) ∧ (3 * x + (a - 1) * y - 2 = 0) → (a = 3 ∨ a = -2))) :=
sorry

end NUMINAMATH_GPT_parallel_lines_sufficient_not_necessary_l2399_239999


namespace NUMINAMATH_GPT_students_per_van_correct_l2399_239926

-- Define the conditions.
def num_vans : Nat := 6
def num_minibuses : Nat := 4
def students_per_minibus : Nat := 24
def total_students : Nat := 156

-- Define the number of students on each van is 'V'
def V : Nat := sorry 

-- State the final question/proof.
theorem students_per_van_correct : V = 10 :=
  sorry


end NUMINAMATH_GPT_students_per_van_correct_l2399_239926


namespace NUMINAMATH_GPT_fuel_oil_used_l2399_239913

theorem fuel_oil_used (V_initial : ℕ) (V_jan : ℕ) (V_may : ℕ) : 
  (V_initial - V_jan) + (V_initial - V_may) = 4582 :=
by
  let V_initial := 3000
  let V_jan := 180
  let V_may := 1238
  sorry

end NUMINAMATH_GPT_fuel_oil_used_l2399_239913


namespace NUMINAMATH_GPT_circle_area_of_white_cube_l2399_239918

/-- 
Marla has a large white cube with an edge length of 12 feet and enough green paint to cover 432 square feet.
Marla paints a white circle centered on each face of the cube, surrounded by a green border.
Prove the area of one of the white circles is 72 square feet.
 -/
theorem circle_area_of_white_cube
  (edge_length : ℝ) (paint_area : ℝ) (faces : ℕ)
  (h_edge_length : edge_length = 12)
  (h_paint_area : paint_area = 432)
  (h_faces : faces = 6) :
  ∃ (circle_area : ℝ), circle_area = 72 :=
by
  sorry

end NUMINAMATH_GPT_circle_area_of_white_cube_l2399_239918


namespace NUMINAMATH_GPT_find_principal_sum_l2399_239978

theorem find_principal_sum (P R : ℝ) (SI CI : ℝ) 
  (h1 : SI = 10200) 
  (h2 : CI = 11730) 
  (h3 : SI = P * R * 2 / 100)
  (h4 : CI = P * (1 + R / 100)^2 - P) :
  P = 17000 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_sum_l2399_239978


namespace NUMINAMATH_GPT_pima_initial_investment_l2399_239972

/-- Pima's initial investment in Ethereum. The investment value gained 25% in the first week and 50% of its current value in the second week. The final investment value is $750. -/
theorem pima_initial_investment (I : ℝ) 
  (h1 : 1.25 * I * 1.5 = 750) : I = 400 :=
sorry

end NUMINAMATH_GPT_pima_initial_investment_l2399_239972


namespace NUMINAMATH_GPT_slope_of_line_l2399_239985

variable (x y : ℝ)

def line_equation : Prop := 4 * y = -5 * x + 8

theorem slope_of_line (h : line_equation x y) :
  ∃ m b, y = m * x + b ∧ m = -5/4 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l2399_239985


namespace NUMINAMATH_GPT_solve_equation_l2399_239998

theorem solve_equation (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x + 3 = 18 :=
sorry

end NUMINAMATH_GPT_solve_equation_l2399_239998


namespace NUMINAMATH_GPT_students_without_scholarships_l2399_239952

theorem students_without_scholarships :
  let total_students := 300
  let full_merit_percent := 0.05
  let half_merit_percent := 0.10
  let sports_percent := 0.03
  let need_based_percent := 0.07
  let full_merit_and_sports_percent := 0.01
  let half_merit_and_need_based_percent := 0.02
  let full_merit := full_merit_percent * total_students
  let half_merit := half_merit_percent * total_students
  let sports := sports_percent * total_students
  let need_based := need_based_percent * total_students
  let full_merit_and_sports := full_merit_and_sports_percent * total_students
  let half_merit_and_need_based := half_merit_and_need_based_percent * total_students
  let total_with_scholarships := (full_merit + half_merit + sports + need_based) - (full_merit_and_sports + half_merit_and_need_based)
  let students_without_scholarships := total_students - total_with_scholarships
  students_without_scholarships = 234 := 
by
  sorry

end NUMINAMATH_GPT_students_without_scholarships_l2399_239952


namespace NUMINAMATH_GPT_total_blue_marbles_l2399_239971

def jason_blue_marbles : Nat := 44
def tom_blue_marbles : Nat := 24

theorem total_blue_marbles : jason_blue_marbles + tom_blue_marbles = 68 := by
  sorry

end NUMINAMATH_GPT_total_blue_marbles_l2399_239971


namespace NUMINAMATH_GPT_artwork_collection_l2399_239948

theorem artwork_collection :
  ∀ (students quarters years artworks_per_student_per_quarter : ℕ), 
  students = 15 → quarters = 4 → years = 2 → artworks_per_student_per_quarter = 2 →
  students * artworks_per_student_per_quarter * quarters * years = 240 :=
by
  intros students quarters years artworks_per_student_per_quarter
  rintro (rfl : students = 15) (rfl : quarters = 4) (rfl : years = 2) (rfl : artworks_per_student_per_quarter = 2)
  sorry

end NUMINAMATH_GPT_artwork_collection_l2399_239948


namespace NUMINAMATH_GPT_find_sachin_age_l2399_239975

-- Define Sachin's and Rahul's ages as variables
variables (S R : ℝ)

-- Define the conditions
def rahul_age := S + 9
def age_ratio := (S / R) = (7 / 9)

-- State the theorem for Sachin's age
theorem find_sachin_age (h1 : R = rahul_age S) (h2 : age_ratio S R) : S = 31.5 :=
by sorry

end NUMINAMATH_GPT_find_sachin_age_l2399_239975


namespace NUMINAMATH_GPT_possible_b_value_l2399_239954

theorem possible_b_value (a b : ℤ) (h1 : a = 3^20) (h2 : a ≡ b [ZMOD 10]) : b = 2011 :=
by sorry

end NUMINAMATH_GPT_possible_b_value_l2399_239954


namespace NUMINAMATH_GPT_binom_divisible_by_prime_l2399_239969

theorem binom_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (hk1 : 1 ≤ k) (hk2 : k ≤ p - 1) : Nat.choose p k % p = 0 := 
  sorry

end NUMINAMATH_GPT_binom_divisible_by_prime_l2399_239969


namespace NUMINAMATH_GPT_atleast_one_alarm_rings_on_time_l2399_239966

def probability_alarm_A_rings := 0.80
def probability_alarm_B_rings := 0.90

def probability_atleast_one_rings := 1 - (1 - probability_alarm_A_rings) * (1 - probability_alarm_B_rings)

theorem atleast_one_alarm_rings_on_time :
  probability_atleast_one_rings = 0.98 :=
sorry

end NUMINAMATH_GPT_atleast_one_alarm_rings_on_time_l2399_239966


namespace NUMINAMATH_GPT_solve_eqs_l2399_239925

theorem solve_eqs (x y : ℤ) (h1 : 7 - x = 15) (h2 : y - 3 = 4 + x) : x = -8 ∧ y = -1 := 
by
  sorry

end NUMINAMATH_GPT_solve_eqs_l2399_239925


namespace NUMINAMATH_GPT_find_value_l2399_239979

def equation := ∃ x : ℝ, x^2 - 2 * x - 3 = 0
def expression (x : ℝ) := 2 * x^2 - 4 * x + 12

theorem find_value :
  (∃ x : ℝ, (x^2 - 2 * x - 3 = 0) ∧ (expression x = 18)) :=
by
  sorry

end NUMINAMATH_GPT_find_value_l2399_239979


namespace NUMINAMATH_GPT_sin_30_eq_half_l2399_239976

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_30_eq_half_l2399_239976


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l2399_239942

open Classical

noncomputable def f (x a : ℝ) := x + a / x

theorem necessary_and_sufficient_condition
  (a : ℝ) :
  (∀ x : ℝ, x > 0 → f x a ≥ 2) ↔ (a ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l2399_239942


namespace NUMINAMATH_GPT_x_finishes_work_alone_in_18_days_l2399_239934

theorem x_finishes_work_alone_in_18_days
  (y_days : ℕ) (y_worked : ℕ) (x_remaining_days : ℝ)
  (hy : y_days = 15) (hy_worked : y_worked = 10) 
  (hx_remaining : x_remaining_days = 6.000000000000001) :
  ∃ (x_days : ℝ), x_days = 18 :=
by 
  sorry

end NUMINAMATH_GPT_x_finishes_work_alone_in_18_days_l2399_239934


namespace NUMINAMATH_GPT_probability_sum_leq_12_l2399_239945

theorem probability_sum_leq_12 (dice1 dice2 : ℕ) (h1 : 1 ≤ dice1 ∧ dice1 ≤ 8) (h2 : 1 ≤ dice2 ∧ dice2 ≤ 8) :
  (∃ outcomes : ℕ, (outcomes = 64) ∧ 
   (∃ favorable : ℕ, (favorable = 54) ∧ 
   (favorable / outcomes = 27 / 32))) :=
sorry

end NUMINAMATH_GPT_probability_sum_leq_12_l2399_239945


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l2399_239947

theorem sum_of_consecutive_integers (n : ℕ) (h : n*(n + 1) = 2720) : n + (n + 1) = 103 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l2399_239947


namespace NUMINAMATH_GPT_smallest_x_abs_eq_15_l2399_239908

theorem smallest_x_abs_eq_15 :
  ∃ x : ℝ, |5 * x - 3| = 15 ∧ ∀ y : ℝ, |5 * y - 3| = 15 → x ≤ y :=
sorry

end NUMINAMATH_GPT_smallest_x_abs_eq_15_l2399_239908


namespace NUMINAMATH_GPT_similar_triangle_perimeter_l2399_239949

theorem similar_triangle_perimeter
  (a b c : ℕ)
  (h1 : a = 7)
  (h2 : b = 7)
  (h3 : c = 12)
  (similar_triangle_longest_side : ℕ)
  (h4 : similar_triangle_longest_side = 36)
  (h5 : c * similar_triangle_longest_side = 12 * 36) :
  ∃ P : ℕ, P = 78 := by
  sorry

end NUMINAMATH_GPT_similar_triangle_perimeter_l2399_239949


namespace NUMINAMATH_GPT_boat_travel_distance_along_stream_l2399_239912

theorem boat_travel_distance_along_stream :
  ∀ (v_s : ℝ), (5 - v_s = 2) → (5 + v_s) * 1 = 8 :=
by
  intro v_s
  intro h1
  have vs_value : v_s = 3 := by linarith
  rw [vs_value]
  norm_num

end NUMINAMATH_GPT_boat_travel_distance_along_stream_l2399_239912


namespace NUMINAMATH_GPT_profit_percentage_example_l2399_239914

noncomputable def selling_price : ℝ := 100
noncomputable def cost_price (sp : ℝ) : ℝ := 0.75 * sp
noncomputable def profit (sp cp : ℝ) : ℝ := sp - cp
noncomputable def profit_percentage (profit cp : ℝ) : ℝ := (profit / cp) * 100

theorem profit_percentage_example :
  profit_percentage (profit selling_price (cost_price selling_price)) (cost_price selling_price) = 33.33 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_profit_percentage_example_l2399_239914


namespace NUMINAMATH_GPT_domain_of_function_l2399_239973

theorem domain_of_function :
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) ↔ (1 - x ≥ 0 ∧ x ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l2399_239973


namespace NUMINAMATH_GPT_scientific_notation_integer_l2399_239995

theorem scientific_notation_integer (x : ℝ) (h1 : x > 10) :
  ∃ (A : ℝ) (N : ℤ), (1 ≤ A ∧ A < 10) ∧ x = A * 10^N :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_integer_l2399_239995


namespace NUMINAMATH_GPT_symmetric_periodic_l2399_239924

theorem symmetric_periodic
  (f : ℝ → ℝ) (a b : ℝ) (h1 : a ≠ b)
  (h2 : ∀ x : ℝ, f (a - x) = f (a + x))
  (h3 : ∀ x : ℝ, f (b - x) = f (b + x)) :
  ∀ x : ℝ, f x = f (x + 2 * (b - a)) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_periodic_l2399_239924


namespace NUMINAMATH_GPT_gcd_m_n_l2399_239938

def m : ℕ := 131^2 + 243^2 + 357^2
def n : ℕ := 130^2 + 242^2 + 358^2

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_m_n_l2399_239938


namespace NUMINAMATH_GPT_group_size_of_bananas_l2399_239910

theorem group_size_of_bananas (totalBananas numberOfGroups : ℕ) (h1 : totalBananas = 203) (h2 : numberOfGroups = 7) :
  totalBananas / numberOfGroups = 29 :=
sorry

end NUMINAMATH_GPT_group_size_of_bananas_l2399_239910


namespace NUMINAMATH_GPT_cyclist_pedestrian_meeting_distance_l2399_239936

theorem cyclist_pedestrian_meeting_distance :
  let A := 0 -- Representing point A at 0 km
  let B := 3 -- Representing point B at 3 km since AB = 3 km
  let C := 7 -- Representing point C at 7 km since AC = AB + BC = 3 + 4 = 7 km
  exists (x : ℝ), x > 0 ∧ x < 3 ∧ x = 2.1 :=
sorry

end NUMINAMATH_GPT_cyclist_pedestrian_meeting_distance_l2399_239936


namespace NUMINAMATH_GPT_ratio_of_number_to_ten_l2399_239906

theorem ratio_of_number_to_ten (n : ℕ) (h : n = 200) : n / 10 = 20 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_number_to_ten_l2399_239906


namespace NUMINAMATH_GPT_ordered_triples_54000_l2399_239923

theorem ordered_triples_54000 : 
  ∃ (count : ℕ), 
  count = 16 ∧ 
  ∀ (a b c : ℕ), 
  0 < a → 0 < b → 0 < c → a^4 * b^2 * c = 54000 → 
  count = 16 := 
sorry

end NUMINAMATH_GPT_ordered_triples_54000_l2399_239923


namespace NUMINAMATH_GPT_cubed_difference_l2399_239955

theorem cubed_difference (x : ℝ) (h : x - 1/x = 3) : (x^3 - 1/x^3 = 36) := 
by
  sorry

end NUMINAMATH_GPT_cubed_difference_l2399_239955


namespace NUMINAMATH_GPT_horror_movie_more_than_triple_romance_l2399_239930

-- Definitions and Conditions
def tickets_sold_romance : ℕ := 25
def tickets_sold_horror : ℕ := 93
def triple_tickets_romance := 3 * tickets_sold_romance

-- Theorem Statement
theorem horror_movie_more_than_triple_romance :
  (tickets_sold_horror - triple_tickets_romance) = 18 :=
by
  sorry

end NUMINAMATH_GPT_horror_movie_more_than_triple_romance_l2399_239930


namespace NUMINAMATH_GPT_triangle_inequality_equality_condition_l2399_239983

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_triangle_inequality_equality_condition_l2399_239983


namespace NUMINAMATH_GPT_trajectory_of_midpoint_l2399_239917

noncomputable section

open Real

-- Define the points and lines
def C : ℝ × ℝ := (-2, -2)
def A (x : ℝ) : ℝ × ℝ := (x, 0)
def B (y : ℝ) : ℝ × ℝ := (0, y)
def M (x y : ℝ) : ℝ × ℝ := ((x + 0) / 2, (0 + y) / 2)

theorem trajectory_of_midpoint (CA_dot_CB : (C.1 * (A 0).1 + (C.2 - (A 0).2)) * (C.1 * (B 0).1 + (C.2 - (B 0).2)) = 0) :
  ∀ (M : ℝ × ℝ), (M.1 = (A 0).1 / 2) ∧ (M.2 = (B 0).2 / 2) → (M.1 + M.2 + 2 = 0) :=
by
  -- here's where the proof would go
  sorry

end NUMINAMATH_GPT_trajectory_of_midpoint_l2399_239917


namespace NUMINAMATH_GPT_find_angle_B_l2399_239915

variable {a b c : ℝ}
variable {A B C : ℝ}
variable {m n : ℝ × ℝ}
variable (h1 : m = (Real.cos A, Real.sin A))
variable (h2 : n = (1, Real.sqrt 3))
variable (h3 : m.1 / n.1 = m.2 / n.2)
variable (h4 : a * Real.cos B + b * Real.cos A = c * Real.sin C)

theorem find_angle_B (h_conditions : a * Real.cos B + b * Real.cos A = c * Real.sin C) : B = Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_find_angle_B_l2399_239915


namespace NUMINAMATH_GPT_toothpicks_required_l2399_239994

noncomputable def total_small_triangles (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def total_initial_toothpicks (n : ℕ) : ℕ :=
  3 * total_small_triangles n

noncomputable def adjusted_toothpicks (n : ℕ) : ℕ :=
  total_initial_toothpicks n / 2

noncomputable def boundary_toothpicks (n : ℕ) : ℕ :=
  2 * n

noncomputable def total_toothpicks (n : ℕ) : ℕ :=
  adjusted_toothpicks n + boundary_toothpicks n

theorem toothpicks_required {n : ℕ} (h : n = 2500) : total_toothpicks n = 4694375 :=
by sorry

end NUMINAMATH_GPT_toothpicks_required_l2399_239994


namespace NUMINAMATH_GPT_roots_numerically_equal_opposite_signs_l2399_239991

theorem roots_numerically_equal_opposite_signs
  (a b d: ℝ) 
  (h: ∃ x : ℝ, (x^2 - (a + 1) * x) / ((b + 1) * x - d) = (n - 2) / (n + 2) ∧ x = -x)
  : n = 2 * (b - a) / (a + b + 2) := by
  sorry

end NUMINAMATH_GPT_roots_numerically_equal_opposite_signs_l2399_239991


namespace NUMINAMATH_GPT_scientific_notation_4947_66_billion_l2399_239922

theorem scientific_notation_4947_66_billion :
  4947.66 * 10^8 = 4.94766 * 10^11 :=
sorry

end NUMINAMATH_GPT_scientific_notation_4947_66_billion_l2399_239922


namespace NUMINAMATH_GPT_sum_values_of_cubes_eq_l2399_239980

theorem sum_values_of_cubes_eq :
  ∀ (a b : ℝ), a^3 + b^3 + 3 * a * b = 1 → a + b = 1 ∨ a + b = -2 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_sum_values_of_cubes_eq_l2399_239980


namespace NUMINAMATH_GPT_max_value_on_interval_l2399_239961

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + c) / Real.exp x
noncomputable def f' (a b c : ℝ) (x : ℝ) : ℝ := ((2 * a * x + b) * Real.exp x - (a * x^2 + b * x + c)) / Real.exp (2 * x)

variable (a b c : ℝ)

-- Given conditions
axiom pos_a : a > 0
axiom zero_point_neg3 : f' a b c (-3) = 0
axiom zero_point_0 : f' a b c 0 = 0
axiom min_value_neg3 : f a b c (-3) = -Real.exp 3

-- Goal: Maximum value of f(x) on the interval [-5, ∞) is 5e^5.
theorem max_value_on_interval : ∃ y ∈ Set.Ici (-5), f a b c y = 5 * Real.exp 5 := by
  sorry

end NUMINAMATH_GPT_max_value_on_interval_l2399_239961


namespace NUMINAMATH_GPT_total_wheels_l2399_239943

theorem total_wheels (n_bicycles n_tricycles n_unicycles n_four_wheelers : ℕ)
                     (w_bicycle w_tricycle w_unicycle w_four_wheeler : ℕ)
                     (h1 : n_bicycles = 16)
                     (h2 : n_tricycles = 7)
                     (h3 : n_unicycles = 10)
                     (h4 : n_four_wheelers = 5)
                     (h5 : w_bicycle = 2)
                     (h6 : w_tricycle = 3)
                     (h7 : w_unicycle = 1)
                     (h8 : w_four_wheeler = 4)
  : (n_bicycles * w_bicycle + n_tricycles * w_tricycle
     + n_unicycles * w_unicycle + n_four_wheelers * w_four_wheeler) = 83 := by
  sorry

end NUMINAMATH_GPT_total_wheels_l2399_239943


namespace NUMINAMATH_GPT_weight_of_b_l2399_239996

theorem weight_of_b (A B C : ℕ) 
  (h1 : A + B + C = 129) 
  (h2 : A + B = 80) 
  (h3 : B + C = 86) : 
  B = 37 := 
by 
  sorry

end NUMINAMATH_GPT_weight_of_b_l2399_239996


namespace NUMINAMATH_GPT_volume_in_cubic_yards_l2399_239940

-- Adding the conditions as definitions
def feet_to_yards : ℝ := 3 -- 3 feet in a yard
def cubic_feet_to_cubic_yards : ℝ := feet_to_yards^3 -- convert to cubic yards
def volume_in_cubic_feet : ℝ := 108 -- volume in cubic feet

-- The theorem to prove the equivalence
theorem volume_in_cubic_yards
  (h1 : feet_to_yards = 3)
  (h2 : volume_in_cubic_feet = 108)
  : (volume_in_cubic_feet / cubic_feet_to_cubic_yards) = 4 := 
sorry

end NUMINAMATH_GPT_volume_in_cubic_yards_l2399_239940


namespace NUMINAMATH_GPT_min_largest_value_in_set_l2399_239962

theorem min_largest_value_in_set (a b : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : (8:ℚ) / 19 * a * b ≤ (a - 1) * a / 2): a ≥ 13 :=
by
  sorry

end NUMINAMATH_GPT_min_largest_value_in_set_l2399_239962


namespace NUMINAMATH_GPT_visible_black_area_ratio_l2399_239937

-- Definitions for circle areas as nonnegative real numbers
variables (A_b A_g A_w : ℝ) (hA_b : 0 ≤ A_b) (hA_g : 0 ≤ A_g) (hA_w : 0 ≤ A_w)
-- Condition: Initial visible black area is 7 times the white area
axiom initial_visible_black_area : 7 * A_w = A_b

-- Definition of new visible black area after movement
def new_visible_black_area := A_b - A_w

-- Prove the ratio of the visible black regions before and after moving the circles
theorem visible_black_area_ratio :
  (7 * A_w) / ((7 * A_w) - A_w) = 7 / 6 :=
by { sorry }

end NUMINAMATH_GPT_visible_black_area_ratio_l2399_239937


namespace NUMINAMATH_GPT_range_of_a_l2399_239916

def p (a : ℝ) : Prop := 0 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 1 / 8

def resolution (a : ℝ) : Prop :=
(p a ∨ q a) ∧ ¬(p a ∧ q a) → (0 < a ∧ a ≤ 1 / 8) ∨ a ≥ 1

theorem range_of_a (a : ℝ) : resolution a := sorry

end NUMINAMATH_GPT_range_of_a_l2399_239916


namespace NUMINAMATH_GPT_merchant_profit_percentage_is_35_l2399_239900

noncomputable def cost_price : ℝ := 100
noncomputable def markup_percentage : ℝ := 0.80
noncomputable def discount_percentage : ℝ := 0.25

-- Marked price after 80% markup
noncomputable def marked_price (cp : ℝ) (markup_pct : ℝ) : ℝ :=
  cp + (markup_pct * cp)

-- Selling price after 25% discount on marked price
noncomputable def selling_price (mp : ℝ) (discount_pct : ℝ) : ℝ :=
  mp - (discount_pct * mp)

-- Profit as the difference between selling price and cost price
noncomputable def profit (sp cp : ℝ) : ℝ :=
  sp - cp

-- Profit percentage
noncomputable def profit_percentage (profit cp : ℝ) : ℝ :=
  (profit / cp) * 100

theorem merchant_profit_percentage_is_35 :
  let cp := cost_price
  let mp := marked_price cp markup_percentage
  let sp := selling_price mp discount_percentage
  let prof := profit sp cp
  profit_percentage prof cp = 35 :=
by
  let cp := cost_price
  let mp := marked_price cp markup_percentage
  let sp := selling_price mp discount_percentage
  let prof := profit sp cp
  show profit_percentage prof cp = 35
  sorry

end NUMINAMATH_GPT_merchant_profit_percentage_is_35_l2399_239900


namespace NUMINAMATH_GPT_evaluate_expression_l2399_239990

-- Definitions based on conditions
variables (b : ℤ) (x : ℤ)
def condition := x = 2 * b + 9

-- Statement of the problem
theorem evaluate_expression (b : ℤ) (x : ℤ) (h : condition b x) : x - 2 * b + 5 = 14 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l2399_239990


namespace NUMINAMATH_GPT_solution_interval_l2399_239967

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 x + x - 2

theorem solution_interval :
  ∃ x, (1 < x ∧ x < 1.5) ∧ f x = 0 :=
sorry

end NUMINAMATH_GPT_solution_interval_l2399_239967


namespace NUMINAMATH_GPT_over_limit_weight_l2399_239974

variable (hc_books : ℕ) (hc_weight : ℕ → ℝ)
variable (tex_books : ℕ) (tex_weight : ℕ → ℝ)
variable (knick_knacks : ℕ) (knick_weight : ℕ → ℝ)
variable (weight_limit : ℝ)

axiom hc_books_value : hc_books = 70
axiom hc_weight_value : hc_weight hc_books = 0.5
axiom tex_books_value : tex_books = 30
axiom tex_weight_value : tex_weight tex_books = 2
axiom knick_knacks_value : knick_knacks = 3
axiom knick_weight_value : knick_weight knick_knacks = 6
axiom weight_limit_value : weight_limit = 80

theorem over_limit_weight : 
  (hc_books * hc_weight hc_books + tex_books * tex_weight tex_books + knick_knacks * knick_weight knick_knacks) - weight_limit = 33 := by
  sorry

end NUMINAMATH_GPT_over_limit_weight_l2399_239974


namespace NUMINAMATH_GPT_notebook_price_l2399_239941

theorem notebook_price (x : ℝ) 
  (h1 : 3 * x + 1.50 + 1.70 = 6.80) : 
  x = 1.20 :=
by 
  sorry

end NUMINAMATH_GPT_notebook_price_l2399_239941


namespace NUMINAMATH_GPT_proof_problem_l2399_239920

-- Conditions
def a : ℤ := 1
def b : ℤ := 0
def c : ℤ := -1 + 3

-- Proof Statement
theorem proof_problem : (2 * a + 3 * c) * b = 0 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l2399_239920


namespace NUMINAMATH_GPT_geometric_sequence_fourth_term_l2399_239935

theorem geometric_sequence_fourth_term (x : ℚ) (r : ℚ)
  (h1 : x ≠ 0)
  (h2 : x ≠ -1)
  (h3 : 3 * x + 3 = r * x)
  (h4 : 5 * x + 5 = r * (3 * x + 3)) :
  r^3 * (5 * x + 5) = -125 / 12 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fourth_term_l2399_239935


namespace NUMINAMATH_GPT_Joey_age_is_six_l2399_239993

theorem Joey_age_is_six (ages: Finset ℕ) (a1 a2 a3 a4 : ℕ) (h1: ages = {4, 6, 8, 10})
  (h2: a1 + a2 = 14 ∨ a2 + a3 = 14 ∨ a3 + a4 = 14) (h3: a1 > 7 ∨ a2 > 7 ∨ a3 > 7 ∨ a4 > 7)
  (h4: (6 ∈ ages ∧ a1 ∈ ages) ∨ (6 ∈ ages ∧ a2 ∈ ages) ∨ 
      (6 ∈ ages ∧ a3 ∈ ages) ∨ (6 ∈ ages ∧ a4 ∈ ages)): 
  (a1 = 6 ∨ a2 = 6 ∨ a3 = 6 ∨ a4 = 6) :=
by
  sorry

end NUMINAMATH_GPT_Joey_age_is_six_l2399_239993


namespace NUMINAMATH_GPT_estimate_production_in_March_l2399_239903

theorem estimate_production_in_March 
  (monthly_production : ℕ → ℝ)
  (x y : ℝ)
  (hx : x = 3)
  (hy : y = x + 1) : y = 4 :=
by
  sorry

end NUMINAMATH_GPT_estimate_production_in_March_l2399_239903


namespace NUMINAMATH_GPT_problem_statement_l2399_239956

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 2)

def S : Set ℝ := {y | ∃ x ≥ 0, y = f x}

theorem problem_statement :
  (∀ y ∈ S, y ≤ 2) ∧ (¬ (2 ∈ S)) ∧ (∀ y ∈ S, y ≥ 3 / 2) ∧ (3 / 2 ∈ S) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2399_239956


namespace NUMINAMATH_GPT_find_cows_l2399_239960

-- Define the number of ducks (D) and cows (C)
variables (D C : ℕ)

-- Define the main condition given in the problem
def legs_eq_condition (D C : ℕ) : Prop :=
  2 * D + 4 * C = 2 * (D + C) + 36

-- State the theorem we wish to prove
theorem find_cows (D C : ℕ) (h : legs_eq_condition D C) : C = 18 :=
sorry

end NUMINAMATH_GPT_find_cows_l2399_239960


namespace NUMINAMATH_GPT_Benjie_is_older_by_5_l2399_239963

def BenjieAge : ℕ := 6
def MargoFutureAge : ℕ := 4
def YearsToFuture : ℕ := 3

theorem Benjie_is_older_by_5 :
  BenjieAge - (MargoFutureAge - YearsToFuture) = 5 :=
by
  sorry

end NUMINAMATH_GPT_Benjie_is_older_by_5_l2399_239963


namespace NUMINAMATH_GPT_original_group_size_l2399_239946

theorem original_group_size (M : ℕ) 
  (h1 : ∀ work_done_by_one, work_done_by_one = 1 / (6 * M))
  (h2 : ∀ work_done_by_one, work_done_by_one = 1 / (12 * (M - 4))) : 
  M = 8 :=
by
  sorry

end NUMINAMATH_GPT_original_group_size_l2399_239946


namespace NUMINAMATH_GPT_rebecca_bought_2_more_bottles_of_water_l2399_239933

noncomputable def number_of_more_bottles_of_water_than_tent_stakes
  (T D W : ℕ) 
  (hT : T = 4) 
  (hD : D = 3 * T) 
  (hTotal : T + D + W = 22) : Prop :=
  W - T = 2

theorem rebecca_bought_2_more_bottles_of_water
  (T D W : ℕ) 
  (hT : T = 4) 
  (hD : D = 3 * T) 
  (hTotal : T + D + W = 22) : 
  number_of_more_bottles_of_water_than_tent_stakes T D W hT hD hTotal :=
by 
  sorry

end NUMINAMATH_GPT_rebecca_bought_2_more_bottles_of_water_l2399_239933


namespace NUMINAMATH_GPT_zoo_children_tuesday_l2399_239987

theorem zoo_children_tuesday 
  (x : ℕ) 
  (child_ticket_cost adult_ticket_cost : ℕ) 
  (children_monday adults_monday adults_tuesday : ℕ)
  (total_revenue : ℕ) : 
  child_ticket_cost = 3 → 
  adult_ticket_cost = 4 → 
  children_monday = 7 → 
  adults_monday = 5 → 
  adults_tuesday = 2 → 
  total_revenue = 61 → 
  7 * 3 + 5 * 4 + x * 3 + 2 * 4 = total_revenue → 
  x = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_zoo_children_tuesday_l2399_239987
