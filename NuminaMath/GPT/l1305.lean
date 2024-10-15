import Mathlib

namespace NUMINAMATH_GPT_fraction_of_water_l1305_130582

theorem fraction_of_water (total_weight sand_ratio water_weight gravel_weight : ℝ)
  (htotal : total_weight = 49.99999999999999)
  (hsand_ratio : sand_ratio = 1/2)
  (hwater : water_weight = total_weight - total_weight * sand_ratio - gravel_weight)
  (hgravel : gravel_weight = 15)
  : (water_weight / total_weight) = 1/5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_water_l1305_130582


namespace NUMINAMATH_GPT_polynomial_divisibility_l1305_130576

theorem polynomial_divisibility (n : ℕ) : (∀ x : ℤ, (x^2 + x + 1 ∣ x^(2*n) + x^n + 1)) ↔ (3 ∣ n) := by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1305_130576


namespace NUMINAMATH_GPT_solve_system_of_equations_l1305_130566

theorem solve_system_of_equations
  (a b c : ℝ) (x y z : ℝ)
  (h1 : x + y = a)
  (h2 : y + z = b)
  (h3 : z + x = c) :
  x = (a + c - b) / 2 ∧ y = (a + b - c) / 2 ∧ z = (b + c - a) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1305_130566


namespace NUMINAMATH_GPT_no_five_coins_sum_to_43_l1305_130559

def coin_values : Set ℕ := {1, 5, 10, 25}

theorem no_five_coins_sum_to_43 :
  ¬ ∃ (a b c d e : ℕ), a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ d ∈ coin_values ∧ e ∈ coin_values ∧ (a + b + c + d + e = 43) :=
sorry

end NUMINAMATH_GPT_no_five_coins_sum_to_43_l1305_130559


namespace NUMINAMATH_GPT_race_distance_between_Sasha_and_Kolya_l1305_130568

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end NUMINAMATH_GPT_race_distance_between_Sasha_and_Kolya_l1305_130568


namespace NUMINAMATH_GPT_length_AM_is_correct_l1305_130541

-- Definitions of the problem conditions
def length_of_square : ℝ := 9

def ratio_AP_PB : ℝ × ℝ := (7, 2)

def radius_of_quarter_circle : ℝ := 9

-- The theorem to prove
theorem length_AM_is_correct
  (AP PB PE : ℝ)
  (x : ℝ)
  (AM : ℝ) 
  (H_AP_PB  : AP = 7 ∧ PB = 2 ∧ PE = 2)
  (H_QD_QE : x = 63 / 11)
  (H_PQ : PQ = 2 + x) :
  AM = 85 / 22 :=
by
  sorry

end NUMINAMATH_GPT_length_AM_is_correct_l1305_130541


namespace NUMINAMATH_GPT_distinct_solutions_eq_108_l1305_130533

theorem distinct_solutions_eq_108 {p q : ℝ} (h1 : (p - 6) * (3 * p + 10) = p^2 - 19 * p + 50)
  (h2 : (q - 6) * (3 * q + 10) = q^2 - 19 * q + 50)
  (h3 : p ≠ q) : (p + 2) * (q + 2) = 108 := 
by
  sorry

end NUMINAMATH_GPT_distinct_solutions_eq_108_l1305_130533


namespace NUMINAMATH_GPT_original_square_area_l1305_130591

-- Definitions based on the given problem conditions
variable (s : ℝ) (A : ℝ)
def is_square (s : ℝ) : Prop := s > 0
def oblique_projection (s : ℝ) (A : ℝ) : Prop :=
  (A = s^2 ∨ A = 4^2) ∧ s = 4

-- The theorem statement based on the problem question and correct answer
theorem original_square_area :
  is_square s →
  oblique_projection s A →
  ∃ A, A = 16 ∨ A = 64 := 
sorry

end NUMINAMATH_GPT_original_square_area_l1305_130591


namespace NUMINAMATH_GPT_percentage_greater_than_88_l1305_130590

theorem percentage_greater_than_88 (x : ℝ) (percentage : ℝ) (h : x = 88 + percentage * 88) (hx : x = 132) : 
  percentage = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_greater_than_88_l1305_130590


namespace NUMINAMATH_GPT_f_of_f_eq_f_l1305_130557

def f (x : ℝ) : ℝ := x^2 - 5 * x

theorem f_of_f_eq_f (x : ℝ) : f (f x) = f x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_f_of_f_eq_f_l1305_130557


namespace NUMINAMATH_GPT_solve_for_x_l1305_130596

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1305_130596


namespace NUMINAMATH_GPT_circle_through_point_and_same_center_l1305_130588

theorem circle_through_point_and_same_center :
  ∃ (x_0 y_0 r : ℝ),
    (∀ (x y : ℝ), (x - x_0)^2 + (y - y_0)^2 = r^2 ↔
      x^2 + y^2 - 4 * x + 6 * y - 3 = 0)
    ∧
    ∀ (x y : ℝ), (x - x_0)^2 + (y - y_0)^2 = r^2 ↔
      (x - 2)^2 + (y + 3)^2 = 25 := sorry

end NUMINAMATH_GPT_circle_through_point_and_same_center_l1305_130588


namespace NUMINAMATH_GPT_one_thirds_in_nine_halves_l1305_130530

theorem one_thirds_in_nine_halves : (9/2) / (1/3) = 27/2 := by
  sorry

end NUMINAMATH_GPT_one_thirds_in_nine_halves_l1305_130530


namespace NUMINAMATH_GPT_zachary_pushups_l1305_130597

theorem zachary_pushups (d z : ℕ) (h1 : d = z + 30) (h2 : d = 37) : z = 7 := by
  sorry

end NUMINAMATH_GPT_zachary_pushups_l1305_130597


namespace NUMINAMATH_GPT_range_b_intersects_ellipse_l1305_130517

open Real

noncomputable def line_intersects_ellipse (b : ℝ) : Prop :=
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < π → ∃ x y : ℝ, x = 2 * cos θ ∧ y = 4 * sin θ ∧ y = x + b

theorem range_b_intersects_ellipse :
  ∀ b : ℝ, line_intersects_ellipse b ↔ b ∈ Set.Icc (-2 : ℝ) (2 * sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_range_b_intersects_ellipse_l1305_130517


namespace NUMINAMATH_GPT_minimum_balls_to_draw_l1305_130509

theorem minimum_balls_to_draw
  (red green yellow blue white : ℕ)
  (h_red : red = 30)
  (h_green : green = 25)
  (h_yellow : yellow = 20)
  (h_blue : blue = 15)
  (h_white : white = 10) :
  ∃ (n : ℕ), n = 81 ∧
    (∀ (r g y b w : ℕ), 
       (r + g + y + b + w >= n) →
       ((r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ w ≥ 20) ∧ 
        (r ≥ 10 ∨ g ≥ 10 ∨ y ≥ 10 ∨ b ≥ 10 ∨ w ≥ 10))
    ) := sorry

end NUMINAMATH_GPT_minimum_balls_to_draw_l1305_130509


namespace NUMINAMATH_GPT_exist_rel_prime_k_l_divisible_l1305_130589

theorem exist_rel_prime_k_l_divisible (a b p : ℤ) : 
  ∃ (k l : ℤ), Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) := 
sorry

end NUMINAMATH_GPT_exist_rel_prime_k_l_divisible_l1305_130589


namespace NUMINAMATH_GPT_group_total_payment_l1305_130531

-- Declare the costs of the tickets as constants
def cost_adult : ℝ := 9.50
def cost_child : ℝ := 6.50

-- Conditions for the group
def total_moviegoers : ℕ := 7
def number_adults : ℕ := 3

-- Calculate the number of children
def number_children : ℕ := total_moviegoers - number_adults

-- Define the total cost paid by the group
def total_cost_paid : ℝ :=
  (number_adults * cost_adult) + (number_children * cost_child)

-- The proof problem: Prove that the total amount paid by the group is $54.50
theorem group_total_payment : total_cost_paid = 54.50 := by
  sorry

end NUMINAMATH_GPT_group_total_payment_l1305_130531


namespace NUMINAMATH_GPT_prism_volume_l1305_130551

theorem prism_volume 
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 50)
  (h3 : b * c = 75) : 
  a * b * c = 150 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_prism_volume_l1305_130551


namespace NUMINAMATH_GPT_prove_expression_l1305_130514

def given_expression : ℤ := -4 + 6 / (-2)

theorem prove_expression : given_expression = -7 := 
by 
  -- insert proof here
  sorry

end NUMINAMATH_GPT_prove_expression_l1305_130514


namespace NUMINAMATH_GPT_max_value_theorem_l1305_130526

open Real

noncomputable def max_value (x y : ℝ) : ℝ :=
  x * y * (75 - 5 * x - 3 * y)

theorem max_value_theorem :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 5 * x + 3 * y < 75 ∧ max_value x y = 3125 / 3 := by
  sorry

end NUMINAMATH_GPT_max_value_theorem_l1305_130526


namespace NUMINAMATH_GPT_stocking_stuffers_total_l1305_130577

-- Defining the number of items per category
def candy_canes := 4
def beanie_babies := 2
def books := 1
def small_toys := 3
def gift_cards := 1

-- Total number of stocking stuffers per child
def items_per_child := candy_canes + beanie_babies + books + small_toys + gift_cards

-- Number of children
def number_of_children := 3

-- Total number of stocking stuffers for all children
def total_stocking_stuffers := items_per_child * number_of_children

-- Statement to be proved
theorem stocking_stuffers_total : total_stocking_stuffers = 33 := by
  sorry

end NUMINAMATH_GPT_stocking_stuffers_total_l1305_130577


namespace NUMINAMATH_GPT_symmetric_point_xoz_plane_l1305_130540

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_xoz (M : Point3D) : Point3D :=
  ⟨M.x, -M.y, M.z⟩

theorem symmetric_point_xoz_plane :
  let M := Point3D.mk 5 1 (-2)
  symmetric_xoz M = Point3D.mk 5 (-1) (-2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_xoz_plane_l1305_130540


namespace NUMINAMATH_GPT_contradiction_in_triangle_l1305_130502

theorem contradiction_in_triangle :
  (∀ (A B C : ℝ), A + B + C = 180 ∧ A < 60 ∧ B < 60 ∧ C < 60 → false) :=
by
  sorry

end NUMINAMATH_GPT_contradiction_in_triangle_l1305_130502


namespace NUMINAMATH_GPT_land_percentage_relationship_l1305_130578

variable {V : ℝ} -- Total taxable value of all land in the village
variable {x y z : ℝ} -- Percentages of Mr. William's land in types A, B, C

-- Conditions
axiom total_tax_collected : 0.80 * (x / 100 * V) + 0.90 * (y / 100 * V) + 0.95 * (z / 100 * V) = 3840
axiom mr_william_tax : 0.80 * (x / 100 * V) + 0.90 * (y / 100 * V) + 0.95 * (z / 100 * V) = 480

-- Prove the relationship
theorem land_percentage_relationship : (0.80 * x + 0.90 * y + 0.95 * z = 48000 / V) → (x + y + z = 100) := by
  sorry

end NUMINAMATH_GPT_land_percentage_relationship_l1305_130578


namespace NUMINAMATH_GPT_measure_angle_YPZ_is_142_l1305_130550

variables (X Y Z : Type) [Inhabited X] [Inhabited Y] [Inhabited Z]
variables (XM YN ZO : Type) [Inhabited XM] [Inhabited YN] [Inhabited ZO]

noncomputable def angle_XYZ : ℝ := 65
noncomputable def angle_XZY : ℝ := 38
noncomputable def angle_YXZ : ℝ := 180 - angle_XYZ - angle_XZY
noncomputable def angle_YNZ : ℝ := 90 - angle_YXZ
noncomputable def angle_ZMY : ℝ := 90 - angle_XYZ
noncomputable def angle_YPZ : ℝ := 180 - angle_YNZ - angle_ZMY

theorem measure_angle_YPZ_is_142 :
  angle_YPZ = 142 := sorry

end NUMINAMATH_GPT_measure_angle_YPZ_is_142_l1305_130550


namespace NUMINAMATH_GPT_sheila_hourly_rate_is_6_l1305_130579

variable (weekly_earnings : ℕ) (hours_mwf : ℕ) (days_mwf : ℕ) (hours_tt: ℕ) (days_tt : ℕ)
variable [NeZero hours_mwf] [NeZero days_mwf] [NeZero hours_tt] [NeZero days_tt]

-- Define Sheila's working hours and weekly earnings as given conditions
def weekly_hours := (hours_mwf * days_mwf) + (hours_tt * days_tt)
def hourly_rate := weekly_earnings / weekly_hours

-- Specific values from the given problem
def sheila_weekly_earnings : ℕ := 216
def sheila_hours_mwf : ℕ := 8
def sheila_days_mwf : ℕ := 3
def sheila_hours_tt : ℕ := 6
def sheila_days_tt : ℕ := 2

-- The theorem to prove
theorem sheila_hourly_rate_is_6 :
  (sheila_weekly_earnings / ((sheila_hours_mwf * sheila_days_mwf) + (sheila_hours_tt * sheila_days_tt))) = 6 := by
  sorry

end NUMINAMATH_GPT_sheila_hourly_rate_is_6_l1305_130579


namespace NUMINAMATH_GPT_sequence_conditions_general_formulas_sum_of_first_n_terms_l1305_130527

noncomputable def arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n n = a_n 1 + d * (n - 1)

noncomputable def geometric_sequence (b_n : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, q > 0 ∧ ∀ n : ℕ, b_n (n + 1) = b_n n * q

variables {a_n b_n c_n : ℕ → ℤ}
variables (d q : ℤ) (d_pos : 0 < d) (hq : q > 0)
variables (S_n : ℕ → ℤ)

axiom initial_conditions : a_n 1 = 2 ∧ b_n 1 = 2 ∧ a_n 3 = 8 ∧ b_n 3 = 8

theorem sequence_conditions : arithmetic_sequence a_n ∧ geometric_sequence b_n := sorry

theorem general_formulas :
  (∀ n : ℕ, a_n n = 3 * n - 1) ∧
  (∀ n : ℕ, b_n n = 2^n) := sorry

theorem sum_of_first_n_terms :
  (∀ n : ℕ, S_n n = 3 * 2^(n+1) - n - 6) := sorry

end NUMINAMATH_GPT_sequence_conditions_general_formulas_sum_of_first_n_terms_l1305_130527


namespace NUMINAMATH_GPT_operations_on_S_l1305_130563

def is_element_of_S (x : ℤ) : Prop :=
  x = 0 ∨ ∃ n : ℤ, x = 2 * n

theorem operations_on_S (a b : ℤ) (ha : is_element_of_S a) (hb : is_element_of_S b) :
  (is_element_of_S (a + b)) ∧
  (is_element_of_S (a - b)) ∧
  (is_element_of_S (a * b)) ∧
  (¬ is_element_of_S (a / b)) ∧
  (¬ is_element_of_S ((a + b) / 2)) :=
by
  sorry

end NUMINAMATH_GPT_operations_on_S_l1305_130563


namespace NUMINAMATH_GPT_quadratic_roots_ratio_l1305_130593

theorem quadratic_roots_ratio (k : ℝ) (k1 k2 : ℝ) (a b : ℝ) 
  (h_roots : ∀ x : ℝ, k * x * x + (1 - 6 * k) * x + 8 = 0 ↔ (x = a ∨ x = b))
  (h_ab : a ≠ b)
  (h_cond : a / b + b / a = 3 / 7)
  (h_ks : k^1 - 6 * (k1 + k2) + 8 = 0)
  (h_vieta : k1 + k2 = 200 / 36 ∧ k1 * k2 = 49 / 36) : 
  (k1 / k2 + k2 / k1 = 6.25) :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_ratio_l1305_130593


namespace NUMINAMATH_GPT_sin_B_sin_C_l1305_130587

open Real

noncomputable def triangle_condition (A B C : ℝ) (a b c : ℝ) : Prop :=
  cos (2 * A) - 3 * cos (B + C) = 1 ∧
  (1 / 2) * b * c * sin A = 5 * sqrt 3 ∧
  b = 5

theorem sin_B_sin_C {A B C a b c : ℝ} (h : triangle_condition A B C a b c) :
  (sin B) * (sin C) = 5 / 7 := 
sorry

end NUMINAMATH_GPT_sin_B_sin_C_l1305_130587


namespace NUMINAMATH_GPT_inequality_proof_l1305_130544

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_condition : (a + b + c + d) * (1/a + 1/b + 1/c + 1/d) = 20) :
  (a^2 + b^2 + c^2 + d^2) * (1/(a^2) + 1/(b^2) + 1/(c^2) + 1/(d^2)) ≥ 36 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1305_130544


namespace NUMINAMATH_GPT_salt_percentage_l1305_130573

theorem salt_percentage (salt water : ℝ) (h_salt : salt = 10) (h_water : water = 40) : 
  salt / water = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_salt_percentage_l1305_130573


namespace NUMINAMATH_GPT_seven_digit_numbers_count_l1305_130510

/-- Given a six-digit phone number represented by six digits A, B, C, D, E, F:
- There are 7 positions where a new digit can be inserted: before A, between each pair of consecutive digits, and after F.
- Each of these positions can be occupied by any of the 10 digits (0 through 9).
The number of seven-digit numbers that can be formed by adding one digit to the six-digit phone number is 70. -/
theorem seven_digit_numbers_count (A B C D E F : ℕ) (hA : 0 ≤ A ∧ A < 10) (hB : 0 ≤ B ∧ B < 10) 
  (hC : 0 ≤ C ∧ C < 10) (hD : 0 ≤ D ∧ D < 10) (hE : 0 ≤ E ∧ E < 10) (hF : 0 ≤ F ∧ F < 10) : 
  ∃ n : ℕ, n = 70 :=
sorry

end NUMINAMATH_GPT_seven_digit_numbers_count_l1305_130510


namespace NUMINAMATH_GPT_perfect_square_m_value_l1305_130547

theorem perfect_square_m_value (M X : ℤ) (hM : M > 1) (hX_lt_max : X < 8000) (hX_gt_min : 1000 < X) (hX_eq : X = M^3) : 
  (∃ M : ℤ, M > 1 ∧ 1000 < M^3 ∧ M^3 < 8000 ∧ (∃ k : ℤ, X = k * k) ∧ M = 16) :=
by
  use 16
  -- Here, we would normally provide the proof steps to show that 1000 < 16^3 < 8000 and 16^3 is a perfect square
  sorry

end NUMINAMATH_GPT_perfect_square_m_value_l1305_130547


namespace NUMINAMATH_GPT_cos_pi_over_6_minus_2alpha_l1305_130511

open Real

noncomputable def tan_plus_pi_over_6 (α : ℝ) := tan (α + π / 6) = 2

theorem cos_pi_over_6_minus_2alpha (α : ℝ) 
  (h1 : π < α ∧ α < 2 * π) 
  (h2 : tan_plus_pi_over_6 α) : 
  cos (π / 6 - 2 * α) = 4 / 5 :=
sorry

end NUMINAMATH_GPT_cos_pi_over_6_minus_2alpha_l1305_130511


namespace NUMINAMATH_GPT_hypotenuse_of_right_triangle_l1305_130561

theorem hypotenuse_of_right_triangle (a b : ℕ) (ha : a = 140) (hb : b = 336) :
  Nat.sqrt (a * a + b * b) = 364 := by
  sorry

end NUMINAMATH_GPT_hypotenuse_of_right_triangle_l1305_130561


namespace NUMINAMATH_GPT_tom_seashells_l1305_130523

theorem tom_seashells (fred_seashells : ℕ) (total_seashells : ℕ) (tom_seashells : ℕ)
  (h1 : fred_seashells = 43)
  (h2 : total_seashells = 58)
  (h3 : total_seashells = fred_seashells + tom_seashells) : tom_seashells = 15 :=
by
  sorry

end NUMINAMATH_GPT_tom_seashells_l1305_130523


namespace NUMINAMATH_GPT_problem_solution_l1305_130554

def p1 : Prop := ∃ x : ℝ, x^2 + x + 1 < 0
def p2 : Prop := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - 1 ≥ 0

theorem problem_solution : (¬ p1) ∨ (¬ p2) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1305_130554


namespace NUMINAMATH_GPT_simplify_expression_l1305_130592

theorem simplify_expression (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  let x := q/r + r/q
  let y := p/r + r/p
  let z := p/q + q/p
  (x^2 + y^2 + z^2 - 2 * x * y * z) = 4 :=
by
  let x := q/r + r/q
  let y := p/r + r/p
  let z := p/q + q/p
  sorry

end NUMINAMATH_GPT_simplify_expression_l1305_130592


namespace NUMINAMATH_GPT_milk_left_l1305_130564

theorem milk_left (initial_milk : ℝ) (given_away : ℝ) (h_initial : initial_milk = 5) (h_given : given_away = 18 / 4) :
  ∃ remaining_milk : ℝ, remaining_milk = initial_milk - given_away ∧ remaining_milk = 1 / 2 :=
by
  use 1 / 2
  sorry

end NUMINAMATH_GPT_milk_left_l1305_130564


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1305_130534

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 5 + 1) : 
  ( ( (x^2 - 1) / x ) / (1 + 1 / x) ) = Real.sqrt 5 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1305_130534


namespace NUMINAMATH_GPT_chromium_percentage_in_second_alloy_l1305_130569

theorem chromium_percentage_in_second_alloy
  (x : ℝ)
  (h1 : chromium_percentage_in_first_alloy = 15)
  (h2 : weight_first_alloy = 15)
  (h3 : weight_second_alloy = 35)
  (h4 : chromium_percentage_in_new_alloy = 10.1)
  (h5 : total_weight = weight_first_alloy + weight_second_alloy)
  (h6 : chromium_in_new_alloy = chromium_percentage_in_new_alloy / 100 * total_weight)
  (h7 : chromium_in_first_alloy = chromium_percentage_in_first_alloy / 100 * weight_first_alloy)
  (h8 : chromium_in_second_alloy = x / 100 * weight_second_alloy)
  (h9 : chromium_in_new_alloy = chromium_in_first_alloy + chromium_in_second_alloy) :
  x = 8 := by
  sorry

end NUMINAMATH_GPT_chromium_percentage_in_second_alloy_l1305_130569


namespace NUMINAMATH_GPT_area_of_MNFK_l1305_130543

theorem area_of_MNFK (ABNF CMKD MNFK : ℝ) (BN : ℝ) (KD : ℝ) (ABMK : ℝ) (CDFN : ℝ)
  (h1 : BN = 8) (h2 : KD = 9) (h3 : ABMK = 25) (h4 : CDFN = 32) :
  MNFK = 31 :=
by
  have hx : 8 * (MNFK + 25) - 25 = 9 * (MNFK + 32) - 32 := sorry
  exact sorry

end NUMINAMATH_GPT_area_of_MNFK_l1305_130543


namespace NUMINAMATH_GPT_solve_for_t_l1305_130575

theorem solve_for_t (t : ℚ) :
  (t+2) * (4*t-4) = (4*t-6) * (t+3) + 3 → t = 7/2 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_t_l1305_130575


namespace NUMINAMATH_GPT_rebecca_eggs_l1305_130562

theorem rebecca_eggs (groups eggs_per_group : ℕ) (h1 : groups = 3) (h2 : eggs_per_group = 6) : 
  (groups * eggs_per_group = 18) :=
by
  sorry

end NUMINAMATH_GPT_rebecca_eggs_l1305_130562


namespace NUMINAMATH_GPT_seq_is_arithmetic_l1305_130584

-- Define the sequence sum S_n and the sequence a_n
noncomputable def S (a : ℕ) (n : ℕ) : ℕ := a * n^2 + n
noncomputable def a_n (a : ℕ) (n : ℕ) : ℕ := S a n - S a (n - 1)

-- Define the property of being an arithmetic sequence
def is_arithmetic_seq (a_n : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → (a_n (n + 1) : ℤ) - (a_n n : ℤ) = d

-- The theorem to be proven
theorem seq_is_arithmetic (a : ℕ) (h : 0 < a) : is_arithmetic_seq (a_n a) :=
by
  sorry

end NUMINAMATH_GPT_seq_is_arithmetic_l1305_130584


namespace NUMINAMATH_GPT_no_solution_exists_l1305_130501

theorem no_solution_exists :
  ¬ ∃ (x1 x2 x3 x4 : ℝ), 
    (x1 + x2 = 1) ∧
    (x2 + x3 - x4 = 1) ∧
    (0 ≤ x1) ∧
    (0 ≤ x2) ∧
    (0 ≤ x3) ∧
    (0 ≤ x4) ∧
    ∀ (F : ℝ), F = x1 - x2 + 2 * x3 - x4 → 
    ∀ (b : ℝ), F ≤ b :=
by sorry

end NUMINAMATH_GPT_no_solution_exists_l1305_130501


namespace NUMINAMATH_GPT_no_integer_solutions_l1305_130524

theorem no_integer_solutions (x y : ℤ) (hx : x ≠ 1) : (x^7 - 1) / (x - 1) ≠ y^5 - 1 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1305_130524


namespace NUMINAMATH_GPT_sum_of_first_7_terms_l1305_130532

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

theorem sum_of_first_7_terms (h1 : a 2 = 3) (h2 : a 6 = 11)
  (h3 : ∀ n, S n = n * (a 1 + a n) / 2) : S 7 = 49 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_first_7_terms_l1305_130532


namespace NUMINAMATH_GPT_share_equally_l1305_130567

variable (Emani Howard : ℕ)
axiom h1 : Emani = 150
axiom h2 : Emani = Howard + 30

theorem share_equally : (Emani + Howard) / 2 = 135 :=
by sorry

end NUMINAMATH_GPT_share_equally_l1305_130567


namespace NUMINAMATH_GPT_combined_list_correct_l1305_130520

def james_friends : ℕ := 75
def john_friends : ℕ := 3 * james_friends
def shared_friends : ℕ := 25
def combined_list : ℕ := james_friends + john_friends - shared_friends

theorem combined_list_correct :
  combined_list = 275 :=
by
  sorry

end NUMINAMATH_GPT_combined_list_correct_l1305_130520


namespace NUMINAMATH_GPT_total_heartbeats_during_race_l1305_130519

namespace Heartbeats

def avg_heart_beats_per_minute : ℕ := 160
def pace_minutes_per_mile : ℕ := 6
def race_distance_miles : ℕ := 20

theorem total_heartbeats_during_race :
  (race_distance_miles * pace_minutes_per_mile * avg_heart_beats_per_minute = 19200) :=
by
  sorry

end Heartbeats

end NUMINAMATH_GPT_total_heartbeats_during_race_l1305_130519


namespace NUMINAMATH_GPT_problem_statement_l1305_130555

theorem problem_statement :
  let pct := 208 / 100
  let initial_value := 1265
  let step1 := pct * initial_value
  let step2 := step1 ^ 2
  let answer := step2 / 12
  answer = 576857.87 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1305_130555


namespace NUMINAMATH_GPT_solution_of_xyz_l1305_130560

theorem solution_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : x + y + z = 48 := 
sorry

end NUMINAMATH_GPT_solution_of_xyz_l1305_130560


namespace NUMINAMATH_GPT_seq_expression_l1305_130548

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ := n^2 * a n

theorem seq_expression (a : ℕ → ℝ) (h₁ : a 1 = 2) (h₂ : ∀ n ≥ 1, S n a = n^2 * a n) :
  ∀ n ≥ 1, a n = 4 / (n * (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_seq_expression_l1305_130548


namespace NUMINAMATH_GPT_smallest_N_exists_l1305_130505

theorem smallest_N_exists (c1 c2 c3 c4 c5 c6 : ℕ) (N : ℕ) :
  (c1 = 6 * c3 - 2) →
  (N + c2 = 6 * c1 - 5) →
  (2 * N + c3 = 6 * c5 - 2) →
  (3 * N + c4 = 6 * c6 - 2) →
  (4 * N + c5 = 6 * c4 - 1) →
  (5 * N + c6 = 6 * c2 - 5) →
  N = 75 :=
by sorry

end NUMINAMATH_GPT_smallest_N_exists_l1305_130505


namespace NUMINAMATH_GPT_n_energetic_all_n_specific_energetic_constraints_l1305_130553

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

end NUMINAMATH_GPT_n_energetic_all_n_specific_energetic_constraints_l1305_130553


namespace NUMINAMATH_GPT_street_length_l1305_130516

theorem street_length
  (time_minutes : ℕ)
  (speed_kmph : ℕ)
  (length_meters : ℕ)
  (h1 : time_minutes = 12)
  (h2 : speed_kmph = 9)
  (h3 : length_meters = 1800) :
  length_meters = (speed_kmph * 1000 / 60) * time_minutes :=
by sorry

end NUMINAMATH_GPT_street_length_l1305_130516


namespace NUMINAMATH_GPT_largest_n_l1305_130585

noncomputable def is_multiple_of_seven (n : ℕ) : Prop :=
  (6 * (n-3)^3 - n^2 + 10 * n - 15) % 7 = 0

theorem largest_n (n : ℕ) : n < 50000 ∧ is_multiple_of_seven n → n = 49999 :=
by sorry

end NUMINAMATH_GPT_largest_n_l1305_130585


namespace NUMINAMATH_GPT_time_to_fill_remaining_l1305_130503

-- Define the rates at which pipes P and Q fill the cistern
def rate_P := 1 / 12
def rate_Q := 1 / 15

-- Define the time both pipes are open together
def time_both_open := 4

-- Calculate the combined rate when both pipes are open
def combined_rate := rate_P + rate_Q

-- Calculate the amount of the cistern filled in the time both pipes are open
def filled_amount_both_open := time_both_open * combined_rate

-- Calculate the remaining amount to fill after Pipe P is turned off
def remaining_amount := 1 - filled_amount_both_open

-- Calculate the time it will take for Pipe Q alone to fill the remaining amount
def time_Q_to_fill_remaining := remaining_amount / rate_Q

-- The final theorem
theorem time_to_fill_remaining : time_Q_to_fill_remaining = 6 := by
  sorry

end NUMINAMATH_GPT_time_to_fill_remaining_l1305_130503


namespace NUMINAMATH_GPT_inequality_abc_l1305_130528

theorem inequality_abc (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_inequality_abc_l1305_130528


namespace NUMINAMATH_GPT_find_r_l1305_130529

noncomputable def r_value (a b : ℝ) (h : a * b = 3) : ℝ :=
  let r := (a^2 + 1 / b^2) * (b^2 + 1 / a^2)
  r

theorem find_r (a b : ℝ) (h : a * b = 3) : r_value a b h = 100 / 9 := by
  sorry

end NUMINAMATH_GPT_find_r_l1305_130529


namespace NUMINAMATH_GPT_find_rate_per_kg_of_mangoes_l1305_130506

theorem find_rate_per_kg_of_mangoes (r : ℝ) 
  (total_units_paid : ℝ) (grapes_kg : ℝ) (grapes_rate : ℝ)
  (mangoes_kg : ℝ) (total_grapes_cost : ℝ)
  (total_mangoes_cost : ℝ) (total_cost : ℝ) :
  grapes_kg = 8 →
  grapes_rate = 70 →
  mangoes_kg = 10 →
  total_units_paid = 1110 →
  total_grapes_cost = grapes_kg * grapes_rate →
  total_mangoes_cost = total_units_paid - total_grapes_cost →
  r = total_mangoes_cost / mangoes_kg →
  r = 55 := by
  intros
  sorry

end NUMINAMATH_GPT_find_rate_per_kg_of_mangoes_l1305_130506


namespace NUMINAMATH_GPT_find_c_l1305_130574

theorem find_c (c : ℝ) (h : (-(c / 3) + -(c / 5) = 30)) : c = -56.25 :=
sorry

end NUMINAMATH_GPT_find_c_l1305_130574


namespace NUMINAMATH_GPT_derivative_at_one_l1305_130512

section

variable {f : ℝ → ℝ}

-- Define the condition
def limit_condition (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (1 + Δx) - f (1 - Δx)) / Δx + 6) < ε

-- State the main theorem
theorem derivative_at_one (h : limit_condition f) : deriv f 1 = -3 :=
by
  sorry

end

end NUMINAMATH_GPT_derivative_at_one_l1305_130512


namespace NUMINAMATH_GPT_initial_percentage_of_grape_juice_l1305_130538

theorem initial_percentage_of_grape_juice
  (P : ℝ)    -- P is the initial percentage in decimal
  (h₁ : 0 ≤ P ∧ P ≤ 1)    -- P is a valid probability
  (h₂ : 40 * P + 10 = 0.36 * 50):    -- Given condition from the problem
  P = 0.2 := 
sorry

end NUMINAMATH_GPT_initial_percentage_of_grape_juice_l1305_130538


namespace NUMINAMATH_GPT_combined_mixture_nuts_l1305_130552

def sue_percentage_nuts : ℝ := 0.30
def sue_percentage_dried_fruit : ℝ := 0.70

def jane_percentage_nuts : ℝ := 0.60
def combined_percentage_dried_fruit : ℝ := 0.35

theorem combined_mixture_nuts :
  let sue_contribution := 100.0
  let jane_contribution := 100.0
  let sue_nuts := sue_contribution * sue_percentage_nuts
  let jane_nuts := jane_contribution * jane_percentage_nuts
  let combined_nuts := sue_nuts + jane_nuts
  let total_weight := sue_contribution + jane_contribution
  (combined_nuts / total_weight) * 100 = 45 :=
by
  sorry

end NUMINAMATH_GPT_combined_mixture_nuts_l1305_130552


namespace NUMINAMATH_GPT_fans_received_all_items_l1305_130504

def multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, m = n * k

theorem fans_received_all_items :
  (∀ n, multiple_of 100 n → multiple_of 40 n ∧ multiple_of 60 n ∧ multiple_of 24 n ∧ n ≤ 7200 → ∃ k, n = 600 * k) →
  (∃ k : ℕ, 7200 / 600 = k ∧ k = 12) :=
by
  sorry

end NUMINAMATH_GPT_fans_received_all_items_l1305_130504


namespace NUMINAMATH_GPT_Ben_win_probability_l1305_130542

theorem Ben_win_probability (lose_prob : ℚ) (no_tie : ¬ ∃ (p : ℚ), p ≠ lose_prob ∧ p + lose_prob = 1) 
  (h : lose_prob = 5/8) : (1 - lose_prob) = 3/8 := by
  sorry

end NUMINAMATH_GPT_Ben_win_probability_l1305_130542


namespace NUMINAMATH_GPT_real_number_a_pure_imaginary_l1305_130558

-- Definition of an imaginary number
def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

-- Given conditions and the proof problem statement
theorem real_number_a_pure_imaginary (a : ℝ) :
  pure_imaginary (⟨(a + 1) / 2, (1 - a) / 2⟩) → a = -1 :=
by
  sorry

end NUMINAMATH_GPT_real_number_a_pure_imaginary_l1305_130558


namespace NUMINAMATH_GPT_market_survey_l1305_130539

theorem market_survey (X Y : ℕ) (h1 : X / Y = 9) (h2 : X + Y = 400) : X = 360 :=
by
  sorry

end NUMINAMATH_GPT_market_survey_l1305_130539


namespace NUMINAMATH_GPT_value_of_expression_l1305_130572

theorem value_of_expression (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : x * y - x = 9 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1305_130572


namespace NUMINAMATH_GPT_g_neither_even_nor_odd_l1305_130522

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 2) + 1/3

theorem g_neither_even_nor_odd : ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) :=
by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_g_neither_even_nor_odd_l1305_130522


namespace NUMINAMATH_GPT_largest_volume_sold_in_august_is_21_l1305_130500

def volumes : List ℕ := [13, 15, 16, 17, 19, 21]

theorem largest_volume_sold_in_august_is_21
  (sold_volumes_august : List ℕ)
  (sold_volumes_september : List ℕ) :
  sold_volumes_august.length = 3 ∧
  sold_volumes_september.length = 2 ∧
  2 * (sold_volumes_september.sum) = sold_volumes_august.sum ∧
  (sold_volumes_august ++ sold_volumes_september).sum = volumes.sum →
  21 ∈ sold_volumes_august :=
sorry

end NUMINAMATH_GPT_largest_volume_sold_in_august_is_21_l1305_130500


namespace NUMINAMATH_GPT_school_spent_on_grass_seeds_bottle_capacity_insufficient_l1305_130537

-- Problem 1: Cost Calculation
theorem school_spent_on_grass_seeds (kg_seeds : ℝ) (cost_per_kg : ℝ) (total_cost : ℝ) 
  (h1 : kg_seeds = 3.3) (h2 : cost_per_kg = 9.48) :
  total_cost = 31.284 :=
  by
    sorry

-- Problem 2: Bottle Capacity
theorem bottle_capacity_insufficient (total_seeds : ℝ) (max_capacity_per_bottle : ℝ) (num_bottles : ℕ)
  (h1 : total_seeds = 3.3) (h2 : max_capacity_per_bottle = 0.35) (h3 : num_bottles = 9) :
  3.3 > 0.35 * 9 :=
  by
    sorry

end NUMINAMATH_GPT_school_spent_on_grass_seeds_bottle_capacity_insufficient_l1305_130537


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_l1305_130594

variable (a : ℤ) 

def is_arithmetic_sequence (a1 a2 a3 : ℤ) : Prop :=
  2 * a2 = a1 + a3

theorem arithmetic_sequence_general_formula :
  ∀ {a1 a2 a3 : ℤ}, is_arithmetic_sequence a1 a2 a3 → a1 = a - 1 ∧ a2 = a + 1 ∧ a3 = 2 * a + 3 → 
  ∀ n : ℕ, a_n = 2 * n - 3
:= by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_l1305_130594


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1305_130518

noncomputable def p (k : ℝ) (x : ℝ) : ℝ :=
  4 * x^2 + 2 * k * x + 9

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, p k x = (2 * x + b)^2) → (k = 6 ∨ k = -6) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1305_130518


namespace NUMINAMATH_GPT_total_accepted_cartons_l1305_130525

-- Definitions for the number of cartons delivered and damaged for each customer
def cartons_delivered_first_two : Nat := 300
def cartons_delivered_last_three : Nat := 200

def cartons_damaged_first : Nat := 70
def cartons_damaged_second : Nat := 50
def cartons_damaged_third : Nat := 40
def cartons_damaged_fourth : Nat := 30
def cartons_damaged_fifth : Nat := 20

-- Statement to prove
theorem total_accepted_cartons :
  let accepted_first := cartons_delivered_first_two - cartons_damaged_first
  let accepted_second := cartons_delivered_first_two - cartons_damaged_second
  let accepted_third := cartons_delivered_last_three - cartons_damaged_third
  let accepted_fourth := cartons_delivered_last_three - cartons_damaged_fourth
  let accepted_fifth := cartons_delivered_last_three - cartons_damaged_fifth
  accepted_first + accepted_second + accepted_third + accepted_fourth + accepted_fifth = 990 :=
by
  sorry

end NUMINAMATH_GPT_total_accepted_cartons_l1305_130525


namespace NUMINAMATH_GPT_circles_internally_tangent_l1305_130581

theorem circles_internally_tangent :
  ∀ (x y : ℝ),
  (x - 6)^2 + y^2 = 1 → 
  (x - 3)^2 + (y - 4)^2 = 36 → 
  true := 
by 
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_circles_internally_tangent_l1305_130581


namespace NUMINAMATH_GPT_cricketer_stats_l1305_130507

theorem cricketer_stats :
  let total_runs := 225
  let total_balls := 120
  let boundaries := 4 * 15
  let sixes := 6 * 8
  let twos := 2 * 3
  let singles := 1 * 10
  let perc_boundaries := (boundaries / total_runs.toFloat) * 100
  let perc_sixes := (sixes / total_runs.toFloat) * 100
  let perc_twos := (twos / total_runs.toFloat) * 100
  let perc_singles := (singles / total_runs.toFloat) * 100
  let strike_rate := (total_runs.toFloat / total_balls.toFloat) * 100
  perc_boundaries = 26.67 ∧
  perc_sixes = 21.33 ∧
  perc_twos = 2.67 ∧
  perc_singles = 4.44 ∧
  strike_rate = 187.5 :=
by
  sorry

end NUMINAMATH_GPT_cricketer_stats_l1305_130507


namespace NUMINAMATH_GPT_mean_inequalities_l1305_130583

noncomputable def arith_mean (a : List ℝ) : ℝ := 
  (a.foldr (· + ·) 0) / a.length

noncomputable def geom_mean (a : List ℝ) : ℝ := 
  Real.exp ((a.foldr (λ x y => Real.log x + y) 0) / a.length)

noncomputable def harm_mean (a : List ℝ) : ℝ := 
  a.length / (a.foldr (λ x y => 1 / x + y) 0)

def is_positive (a : List ℝ) : Prop := 
  ∀ x ∈ a, x > 0

def bounds (a : List ℝ) (m g h : ℝ) : Prop := 
  let α := List.minimum a
  let β := List.maximum a
  α ≤ h ∧ h ≤ g ∧ g ≤ m ∧ m ≤ β

theorem mean_inequalities (a : List ℝ) (h g m : ℝ) (h_assoc: h = harm_mean a) (g_assoc: g = geom_mean a) (m_assoc: m = arith_mean a) :
  is_positive a → bounds a m g h :=
  
sorry

end NUMINAMATH_GPT_mean_inequalities_l1305_130583


namespace NUMINAMATH_GPT_combined_fractions_value_l1305_130599

theorem combined_fractions_value (N : ℝ) (h1 : 0.40 * N = 168) : 
  (1/4) * (1/3) * (2/5) * N = 14 :=
by
  sorry

end NUMINAMATH_GPT_combined_fractions_value_l1305_130599


namespace NUMINAMATH_GPT_special_day_jacket_price_l1305_130570

noncomputable def original_price : ℝ := 240
noncomputable def first_discount_rate : ℝ := 0.4
noncomputable def special_day_discount_rate : ℝ := 0.25

noncomputable def first_discounted_price : ℝ :=
  original_price * (1 - first_discount_rate)
  
noncomputable def special_day_price : ℝ :=
  first_discounted_price * (1 - special_day_discount_rate)

theorem special_day_jacket_price : special_day_price = 108 := by
  -- definitions and calculations go here
  sorry

end NUMINAMATH_GPT_special_day_jacket_price_l1305_130570


namespace NUMINAMATH_GPT_multiply_exponents_l1305_130521

theorem multiply_exponents (a : ℝ) : (6 * a^2) * (1/2 * a^3) = 3 * a^5 := by
  sorry

end NUMINAMATH_GPT_multiply_exponents_l1305_130521


namespace NUMINAMATH_GPT_CEMC_additional_employees_l1305_130535

variable (t : ℝ)

def initialEmployees (t : ℝ) := t + 40

def finalEmployeesMooseJaw (t : ℝ) := 1.25 * t

def finalEmployeesOkotoks : ℝ := 26

def finalEmployeesTotal (t : ℝ) := finalEmployeesMooseJaw t + finalEmployeesOkotoks

def netChangeInEmployees (t : ℝ) := finalEmployeesTotal t - initialEmployees t

theorem CEMC_additional_employees (t : ℝ) (h : t = 120) : 
    netChangeInEmployees t = 16 := 
by
    sorry

end NUMINAMATH_GPT_CEMC_additional_employees_l1305_130535


namespace NUMINAMATH_GPT_remainder_of_expression_l1305_130571

theorem remainder_of_expression (n : ℤ) (h : n % 100 = 99) : (n^2 + 2*n + 3 + n^3) % 100 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_expression_l1305_130571


namespace NUMINAMATH_GPT_orange_probability_l1305_130595

theorem orange_probability (total_apples : ℕ) (total_oranges : ℕ) (other_fruits : ℕ)
  (h1 : total_apples = 20) (h2 : total_oranges = 10) (h3 : other_fruits = 0) :
  (total_oranges : ℚ) / (total_apples + total_oranges + other_fruits) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_orange_probability_l1305_130595


namespace NUMINAMATH_GPT_average_jump_difference_l1305_130508

-- Define the total jumps and time
def total_jumps_liu_li : ℕ := 480
def total_jumps_zhang_hua : ℕ := 420
def time_minutes : ℕ := 5

-- Define the average jumps per minute
def average_jumps_per_minute (total_jumps : ℕ) (time : ℕ) : ℕ :=
  total_jumps / time

-- State the theorem
theorem average_jump_difference :
  average_jumps_per_minute total_jumps_liu_li time_minutes - 
  average_jumps_per_minute total_jumps_zhang_hua time_minutes = 12 := 
sorry


end NUMINAMATH_GPT_average_jump_difference_l1305_130508


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1305_130556

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem eccentricity_of_ellipse :
  let P := (2, 3)
  let F1 := (-2, 0)
  let F2 := (2, 0)
  let d1 := distance P F1
  let d2 := distance P F2
  let a := (d1 + d2) / 2
  let c := distance F1 F2 / 2
  let e := c / a
  e = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1305_130556


namespace NUMINAMATH_GPT_quadratic_equal_real_roots_l1305_130546

theorem quadratic_equal_real_roots :
  ∃ k : ℝ, (∀ x : ℝ, x^2 - 4 * x + k = 0) ∧ k = 4 := by
  sorry

end NUMINAMATH_GPT_quadratic_equal_real_roots_l1305_130546


namespace NUMINAMATH_GPT_distance_between_intersections_l1305_130565

def ellipse_eq (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 25 = 1

def is_focus_of_ellipse (fx fy : ℝ) : Prop := (fx = 0 ∧ (fy = 4 ∨ fy = -4))

def parabola_eq (x y : ℝ) : Prop := y = x^2 / 8 + 2

theorem distance_between_intersections :
  let d := 12 * Real.sqrt 2 / 5
  ∃ x1 x2 y1 y2 : ℝ, 
    ellipse_eq x1 y1 ∧ 
    parabola_eq x1 y1 ∧
    ellipse_eq x2 y2 ∧
    parabola_eq x2 y2 ∧ 
    (x2 - x1)^2 + (y2 - y1)^2 = d^2 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_intersections_l1305_130565


namespace NUMINAMATH_GPT_xy_range_l1305_130515

open Real

theorem xy_range (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 2 / x + 3 * y + 4 / y = 10) : 
  1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_xy_range_l1305_130515


namespace NUMINAMATH_GPT_solve_expression_l1305_130598

theorem solve_expression : 6 / 3 - 2 - 8 + 2 * 8 = 8 := 
by 
  sorry

end NUMINAMATH_GPT_solve_expression_l1305_130598


namespace NUMINAMATH_GPT_sum_of_three_primes_eq_86_l1305_130545

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_of_three_primes_eq_86 (a b c : ℕ) (ha : is_prime a) (hb : is_prime b) (hc : is_prime c) (h_sum : a + b + c = 86) :
  (a, b, c) = (2, 5, 79) ∨ (a, b, c) = (2, 11, 73) ∨ (a, b, c) = (2, 13, 71) ∨ (a, b, c) = (2, 17, 67) ∨
  (a, b, c) = (2, 23, 61) ∨ (a, b, c) = (2, 31, 53) ∨ (a, b, c) = (2, 37, 47) ∨ (a, b, c) = (2, 41, 43) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_primes_eq_86_l1305_130545


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l1305_130536

variable (a b hcf lcm : ℕ)

theorem lcm_of_two_numbers (ha : a = 330) (hb : b = 210) (hhcf : Nat.gcd a b = 30) :
  Nat.lcm a b = 2310 := by
  sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_l1305_130536


namespace NUMINAMATH_GPT_large_square_area_l1305_130513

theorem large_square_area (l w : ℕ) (h1 : 2 * (l + w) = 28) : (l + w) * (l + w) = 196 :=
by {
  sorry
}

end NUMINAMATH_GPT_large_square_area_l1305_130513


namespace NUMINAMATH_GPT_arithmetic_sequence_evaluation_l1305_130586

theorem arithmetic_sequence_evaluation :
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 337 :=
by 
-- Proof omitted
sorry

end NUMINAMATH_GPT_arithmetic_sequence_evaluation_l1305_130586


namespace NUMINAMATH_GPT_four_digit_number_with_divisors_l1305_130580

def is_four_digit (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_minimal_divisor (n p : Nat) : Prop :=
  p > 1 ∧ n % p = 0
  
def is_maximal_divisor (n q : Nat) : Prop :=
  q < n ∧ n % q = 0
  
theorem four_digit_number_with_divisors :
  ∃ (n p : Nat), is_four_digit n ∧ is_minimal_divisor n p ∧ n = 49 * p * p :=
by
  sorry

end NUMINAMATH_GPT_four_digit_number_with_divisors_l1305_130580


namespace NUMINAMATH_GPT_calculate_sum_l1305_130549

theorem calculate_sum : 5 * 12 + 7 * 15 + 13 * 4 + 6 * 9 = 271 :=
by
  sorry

end NUMINAMATH_GPT_calculate_sum_l1305_130549
