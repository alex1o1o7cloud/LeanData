import Mathlib

namespace NUMINAMATH_GPT_Julio_spent_on_limes_l1189_118946

theorem Julio_spent_on_limes
  (days : ℕ)
  (lime_cost_per_3 : ℕ)
  (mocktails_per_day : ℕ)
  (lime_juice_per_lime_tbsp : ℕ)
  (lime_juice_per_mocktail_tbsp : ℕ)
  (limes_per_set : ℕ)
  (days_eq_30 : days = 30)
  (lime_cost_per_3_eq_1 : lime_cost_per_3 = 1)
  (mocktails_per_day_eq_1 : mocktails_per_day = 1)
  (lime_juice_per_lime_tbsp_eq_2 : lime_juice_per_lime_tbsp = 2)
  (lime_juice_per_mocktail_tbsp_eq_1 : lime_juice_per_mocktail_tbsp = 1)
  (limes_per_set_eq_3 : limes_per_set = 3) :
  days * mocktails_per_day * lime_juice_per_mocktail_tbsp / lime_juice_per_lime_tbsp / limes_per_set * lime_cost_per_3 = 5 :=
sorry

end NUMINAMATH_GPT_Julio_spent_on_limes_l1189_118946


namespace NUMINAMATH_GPT_stamp_total_cost_l1189_118922

theorem stamp_total_cost :
  let price_A := 2
  let price_B := 3
  let price_C := 5
  let num_A := 150
  let num_B := 90
  let num_C := 60
  let discount_A := if num_A > 100 then 0.20 else 0
  let discount_B := if num_B > 50 then 0.15 else 0
  let discount_C := if num_C > 30 then 0.10 else 0
  let cost_A := num_A * price_A * (1 - discount_A)
  let cost_B := num_B * price_B * (1 - discount_B)
  let cost_C := num_C * price_C * (1 - discount_C)
  cost_A + cost_B + cost_C = 739.50 := sorry

end NUMINAMATH_GPT_stamp_total_cost_l1189_118922


namespace NUMINAMATH_GPT_grey_eyes_black_hair_l1189_118997

-- Definitions based on conditions
def num_students := 60
def num_black_hair := 36
def num_green_eyes_red_hair := 20
def num_grey_eyes := 24

-- Calculate number of students with red hair
def num_red_hair := num_students - num_black_hair

-- Calculate number of grey-eyed students with red hair
def num_grey_eyes_red_hair := num_red_hair - num_green_eyes_red_hair

-- Prove the number of grey-eyed students with black hair
theorem grey_eyes_black_hair:
  ∃ n, n = num_grey_eyes - num_grey_eyes_red_hair ∧ n = 20 :=
by
  sorry

end NUMINAMATH_GPT_grey_eyes_black_hair_l1189_118997


namespace NUMINAMATH_GPT_similar_triangles_legs_l1189_118968

theorem similar_triangles_legs (y : ℝ) (h : 12 / y = 9 / 7) : y = 84 / 9 := by
  sorry

end NUMINAMATH_GPT_similar_triangles_legs_l1189_118968


namespace NUMINAMATH_GPT_probability_of_selecting_specific_letters_l1189_118978

theorem probability_of_selecting_specific_letters :
  let total_cards := 15
  let amanda_cards := 6
  let chloe_or_ethan_cards := 9
  let prob_amanda_then_chloe_or_ethan := (amanda_cards / total_cards) * (chloe_or_ethan_cards / (total_cards - 1))
  let prob_chloe_or_ethan_then_amanda := (chloe_or_ethan_cards / total_cards) * (amanda_cards / (total_cards - 1))
  let total_prob := prob_amanda_then_chloe_or_ethan + prob_chloe_or_ethan_then_amanda
  total_prob = 18 / 35 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_specific_letters_l1189_118978


namespace NUMINAMATH_GPT_initial_red_marbles_l1189_118941

theorem initial_red_marbles (r g : ℕ) 
  (h1 : r = 5 * g / 3) 
  (h2 : (r - 20) * 5 = g + 40) : 
  r = 317 :=
by
  sorry

end NUMINAMATH_GPT_initial_red_marbles_l1189_118941


namespace NUMINAMATH_GPT_find_f_of_2_l1189_118947

variable (f : ℝ → ℝ)

def functional_equation_condition :=
  ∀ x : ℝ, f (f (f x)) + 3 * f (f x) + 9 * f x + 27 * x = 0

theorem find_f_of_2
  (h : functional_equation_condition f) :
  f (f (f (f 2))) = 162 :=
sorry

end NUMINAMATH_GPT_find_f_of_2_l1189_118947


namespace NUMINAMATH_GPT_green_eyed_snack_min_l1189_118970

variable {total_count green_eyes_count snack_bringers_count : ℕ}

def least_green_eyed_snack_bringers (total_count green_eyes_count snack_bringers_count : ℕ) : ℕ :=
  green_eyes_count - (total_count - snack_bringers_count)

theorem green_eyed_snack_min 
  (h_total : total_count = 35)
  (h_green_eyes : green_eyes_count = 18)
  (h_snack_bringers : snack_bringers_count = 24)
  : least_green_eyed_snack_bringers total_count green_eyes_count snack_bringers_count = 7 :=
by
  rw [h_total, h_green_eyes, h_snack_bringers]
  unfold least_green_eyed_snack_bringers
  norm_num

end NUMINAMATH_GPT_green_eyed_snack_min_l1189_118970


namespace NUMINAMATH_GPT_order_abc_l1189_118981

noncomputable def a : ℝ := (3 * (2 - Real.log 3)) / Real.exp 2
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := (Real.sqrt (Real.exp 1)) / (2 * Real.exp 1)

theorem order_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_GPT_order_abc_l1189_118981


namespace NUMINAMATH_GPT_angle_solution_exists_l1189_118973

theorem angle_solution_exists :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ 9 * (Real.sin x) * (Real.cos x)^4 - 9 * (Real.sin x)^4 * (Real.cos x) = 1 / 2 ∧ x = 30 :=
by
  sorry

end NUMINAMATH_GPT_angle_solution_exists_l1189_118973


namespace NUMINAMATH_GPT_hyperbola_asymptotes_identical_l1189_118983

theorem hyperbola_asymptotes_identical (x y M : ℝ) :
  (∃ (a b : ℝ), a = 3 ∧ b = 4 ∧ (y = (b/a) * x ∨ y = -(b/a) * x)) ∧
  (∃ (c d : ℝ), c = 5 ∧ y = (c / d) * x ∨ y = -(c / d) * x) →
  M = (225 / 16) :=
by sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_identical_l1189_118983


namespace NUMINAMATH_GPT_lcm_150_414_l1189_118982

theorem lcm_150_414 : Nat.lcm 150 414 = 10350 :=
by
  sorry

end NUMINAMATH_GPT_lcm_150_414_l1189_118982


namespace NUMINAMATH_GPT_solve_quadratic_equation_l1189_118958

theorem solve_quadratic_equation :
  ∀ (x : ℝ), ((x - 2) * (x + 3) = 0) ↔ (x = 2 ∨ x = -3) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l1189_118958


namespace NUMINAMATH_GPT_square_sum_inverse_eq_23_l1189_118945

theorem square_sum_inverse_eq_23 {x : ℝ} (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 :=
by
  sorry

end NUMINAMATH_GPT_square_sum_inverse_eq_23_l1189_118945


namespace NUMINAMATH_GPT_difference_between_blue_and_red_balls_l1189_118928

-- Definitions and conditions
def number_of_blue_balls := ℕ
def number_of_red_balls := ℕ
def difference_between_balls (m n : ℕ) := m - n

-- Problem statement: Prove that the difference between number_of_blue_balls and number_of_red_balls
-- can be any natural number greater than 1.
theorem difference_between_blue_and_red_balls (m n : ℕ) (h1 : m > n) (h2 : 
  let P_same := (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1))
  let P_diff := 2 * (n * m) / ((n + m) * (n + m - 1))
  P_same = P_diff
  ) : ∃ a : ℕ, a > 1 ∧ a = m - n :=
by
  sorry

end NUMINAMATH_GPT_difference_between_blue_and_red_balls_l1189_118928


namespace NUMINAMATH_GPT_geometric_seq_general_formula_sum_c_seq_terms_l1189_118992

noncomputable def a_seq (n : ℕ) : ℕ := 2 * 3 ^ (n - 1)

noncomputable def S_seq (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (a_seq n - 2) / 2

theorem geometric_seq_general_formula (n : ℕ) (h : n > 0) : 
  a_seq n = 2 * 3 ^ (n - 1) := 
by {
  sorry
}

noncomputable def d_n (n : ℕ) : ℕ :=
  (a_seq (n + 1) - a_seq n) / (n + 1)

noncomputable def c_seq (n : ℕ) : ℕ :=
  d_n n / (n * a_seq n)

noncomputable def T_n (n : ℕ) : ℕ :=
  2 * (1 - 1 / (n + 1)) * n

theorem sum_c_seq_terms (n : ℕ) (h : n > 0) : 
  T_n n = 2 * n / (n + 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_seq_general_formula_sum_c_seq_terms_l1189_118992


namespace NUMINAMATH_GPT_height_of_pole_l1189_118969

noncomputable section
open Real

theorem height_of_pole (α β γ : ℝ) (h xA xB xC : ℝ) 
  (hA : tan α = h / xA) (hB : tan β = h / xB) (hC : tan γ = h / xC) 
  (sum_angles : α + β + γ = π / 2) : h = 10 :=
by
  sorry

end NUMINAMATH_GPT_height_of_pole_l1189_118969


namespace NUMINAMATH_GPT_factor_quadratic_l1189_118962

theorem factor_quadratic (x : ℝ) : (16 * x^2 - 40 * x + 25) = (4 * x - 5)^2 :=
by 
  sorry

end NUMINAMATH_GPT_factor_quadratic_l1189_118962


namespace NUMINAMATH_GPT_ratio_alison_brittany_l1189_118907

def kent_money : ℕ := 1000
def brooke_money : ℕ := 2 * kent_money
def brittany_money : ℕ := 4 * brooke_money
def alison_money : ℕ := 4000

theorem ratio_alison_brittany : alison_money * 2 = brittany_money :=
by
  sorry

end NUMINAMATH_GPT_ratio_alison_brittany_l1189_118907


namespace NUMINAMATH_GPT_playground_area_l1189_118977

theorem playground_area (B : ℕ) (L : ℕ) (playground_area : ℕ) 
  (h1 : L = 8 * B) 
  (h2 : L = 240) 
  (h3 : playground_area = (1 / 6) * (L * B)) : 
  playground_area = 1200 :=
by
  sorry

end NUMINAMATH_GPT_playground_area_l1189_118977


namespace NUMINAMATH_GPT_hybrids_with_full_headlights_l1189_118996

theorem hybrids_with_full_headlights (total_cars hybrids_percentage one_headlight_percentage : ℝ) 
  (hc : total_cars = 600) (hp : hybrids_percentage = 0.60) (ho : one_headlight_percentage = 0.40) : 
  total_cars * hybrids_percentage - total_cars * hybrids_percentage * one_headlight_percentage = 216 := by
  sorry

end NUMINAMATH_GPT_hybrids_with_full_headlights_l1189_118996


namespace NUMINAMATH_GPT_tan_alpha_eq_three_sin_cos_l1189_118905

theorem tan_alpha_eq_three_sin_cos (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin α * Real.cos α = 3 / 10 :=
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_three_sin_cos_l1189_118905


namespace NUMINAMATH_GPT_father_walk_time_l1189_118944

-- Xiaoming's cycling speed is 4 times his father's walking speed.
-- Xiaoming continues for another 18 minutes to reach B after meeting his father.
-- Prove that Xiaoming's father needs 288 minutes to walk from the meeting point to A.
theorem father_walk_time {V : ℝ} (h₁ : V > 0) (h₂ : ∀ t : ℝ, t > 0 → 18 * V = (V / 4) * t) :
  288 = 4 * 72 :=
by
  sorry

end NUMINAMATH_GPT_father_walk_time_l1189_118944


namespace NUMINAMATH_GPT_part1_part2_l1189_118923

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := |2 * x| + |2 * x - 3|

-- Part 1: Proving the inequality solution
theorem part1 (x : ℝ) (h : f x ≤ 5) :
  -1/2 ≤ x ∧ x ≤ 2 :=
sorry

-- Part 2: Proving the range of m
theorem part2 (x₀ m : ℝ) (h1 : x₀ ∈ Set.Ici 1)
  (h2 : f x₀ + m ≤ x₀ + 3/x₀) :
  m ≤ 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1189_118923


namespace NUMINAMATH_GPT_max_value_of_sums_l1189_118991

noncomputable def max_of_sums (a b c d : ℝ) : ℝ :=
  a^4 + b^4 + c^4 + d^4

theorem max_value_of_sums (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) :
  max_of_sums a b c d ≤ 16 :=
sorry

end NUMINAMATH_GPT_max_value_of_sums_l1189_118991


namespace NUMINAMATH_GPT_angle_same_after_minutes_l1189_118936

def angle_between_hands (H M : ℝ) : ℝ :=
  abs (30 * H - 5.5 * M)

theorem angle_same_after_minutes (x : ℝ) :
  x = 54 + 6 / 11 → 
  angle_between_hands (5 + (x / 60)) x = 150 :=
by
  sorry

end NUMINAMATH_GPT_angle_same_after_minutes_l1189_118936


namespace NUMINAMATH_GPT_simplify_fraction_addition_l1189_118949

theorem simplify_fraction_addition (a b : ℚ) (h1 : a = 4 / 252) (h2 : b = 17 / 36) :
  a + b = 41 / 84 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_addition_l1189_118949


namespace NUMINAMATH_GPT_equilateral_triangle_of_altitude_sum_l1189_118964

def triangle (a b c : ℝ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def altitude (a b c : ℝ) (S : ℝ) : ℝ := 
  2 * S / a

noncomputable def inradius (S : ℝ) (s : ℝ) : ℝ := 
  S / s

def shape_equilateral (a b c : ℝ) : Prop := 
  a = b ∧ b = c

theorem equilateral_triangle_of_altitude_sum (a b c h_a h_b h_c r S s : ℝ) 
  (habc : triangle a b c)
  (ha : h_a = altitude a b c S)
  (hb : h_b = altitude b a c S)
  (hc : h_c = altitude c a b S)
  (hr : r = inradius S s)
  (h_sum : h_a + h_b + h_c = 9 * r)
  (h_area : S = s * r)
  (h_semi : s = (a + b + c) / 2) : 
  shape_equilateral a b c := 
sorry

end NUMINAMATH_GPT_equilateral_triangle_of_altitude_sum_l1189_118964


namespace NUMINAMATH_GPT_area_triangle_possible_values_l1189_118972

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem area_triangle_possible_values (a b c : ℝ) (A B C : ℝ) (ha : a = 2) (hc : c = 2 * Real.sqrt 3) (hA : A = Real.pi / 6) :
  ∃ S, S = 2 * Real.sqrt 3 ∨ S = Real.sqrt 3 :=
by
  -- Define the area using the given values
  sorry

end NUMINAMATH_GPT_area_triangle_possible_values_l1189_118972


namespace NUMINAMATH_GPT_parallel_lines_implies_value_of_a_l1189_118912

theorem parallel_lines_implies_value_of_a (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y = 0 ∧ x + (a-1)*y + (a^2-1) = 0 → 
  (- a / 2) = - (1 / (a-1))) → a = 2 :=
sorry

end NUMINAMATH_GPT_parallel_lines_implies_value_of_a_l1189_118912


namespace NUMINAMATH_GPT_mr_kishore_savings_l1189_118984

theorem mr_kishore_savings :
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let misc := 3940
  let total_expenses := rent + milk + groceries + education + petrol + misc
  let savings_percentage := 0.10
  let salary := total_expenses / (1 - savings_percentage)
  let savings := savings_percentage * salary
  savings = 1937.78 := by
  sorry

end NUMINAMATH_GPT_mr_kishore_savings_l1189_118984


namespace NUMINAMATH_GPT_geometric_sequence_a5_l1189_118971

-- Definitions from the conditions
def a1 : ℕ := 2
def a9 : ℕ := 8

-- The statement we need to prove
theorem geometric_sequence_a5 (q : ℝ) (h1 : a1 = 2) (h2 : a9 = a1 * q ^ 8) : a1 * q ^ 4 = 4 := by
  have h_q4 : q ^ 4 = 2 := sorry
  -- Proof continues...
  sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l1189_118971


namespace NUMINAMATH_GPT_even_fn_a_eq_zero_l1189_118987

def f (x a : ℝ) : ℝ := x^2 - |x + a|

theorem even_fn_a_eq_zero (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_even_fn_a_eq_zero_l1189_118987


namespace NUMINAMATH_GPT_part1_double_root_equation_part2_value_m_squared_2m_2_part3_value_m_l1189_118948

-- Part 1: Is x^2 - 3x + 2 = 0 a "double root equation"?
theorem part1_double_root_equation :
    ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂ ∧ x₁ * 2 = x₂) 
              ∧ (x^2 - 3 * x + 2 = 0) :=
sorry

-- Part 2: Given (x - 2)(x - m) = 0 is a "double root equation", find value of m^2 + 2m + 2.
theorem part2_value_m_squared_2m_2 (m : ℝ) :
    ∃ (v : ℝ), v = m^2 + 2 * m + 2 ∧ 
          (m = 1 ∨ m = 4) ∧
          (v = 5 ∨ v = 26) :=
sorry

-- Part 3: Determine m such that x^2 - (m-1)x + 32 = 0 is a "double root equation".
theorem part3_value_m (m : ℝ) :
    x^2 - (m - 1) * x + 32 = 0 ∧ 
    (m = 13 ∨ m = -11) :=
sorry

end NUMINAMATH_GPT_part1_double_root_equation_part2_value_m_squared_2m_2_part3_value_m_l1189_118948


namespace NUMINAMATH_GPT_walking_speed_l1189_118930

theorem walking_speed 
  (v : ℕ) -- v represents the man's walking speed in kmph
  (distance_formula : distance = speed * time)
  (distance_walking : distance = v * 9)
  (distance_running : distance = 24 * 3) : 
  v = 8 :=
by
  sorry

end NUMINAMATH_GPT_walking_speed_l1189_118930


namespace NUMINAMATH_GPT_min_value_expression_l1189_118995

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 = 4 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1189_118995


namespace NUMINAMATH_GPT_area_of_polygon_ABLFKJ_l1189_118919

theorem area_of_polygon_ABLFKJ 
  (side_length : ℝ) (area_square : ℝ) (midpoint_l : ℝ) (area_triangle : ℝ)
  (remaining_area_each_square : ℝ) (total_area : ℝ)
  (h1 : side_length = 6)
  (h2 : area_square = side_length * side_length)
  (h3 : midpoint_l = side_length / 2)
  (h4 : area_triangle = 0.5 * side_length * midpoint_l)
  (h5 : remaining_area_each_square = area_square - 2 * area_triangle)
  (h6 : total_area = 3 * remaining_area_each_square)
  : total_area = 54 :=
by
  sorry

end NUMINAMATH_GPT_area_of_polygon_ABLFKJ_l1189_118919


namespace NUMINAMATH_GPT_computers_built_per_month_l1189_118953

theorem computers_built_per_month (days_in_month : ℕ) (hours_per_day : ℕ) (computers_per_interval : ℚ) (intervals_per_hour : ℕ)
    (h_days : days_in_month = 28) (h_hours : hours_per_day = 24) (h_computers : computers_per_interval = 2.25) (h_intervals : intervals_per_hour = 2) :
    days_in_month * hours_per_day * intervals_per_hour * computers_per_interval = 3024 :=
by
  -- We would give the proof here, but it's omitted as per instructions.
  sorry

end NUMINAMATH_GPT_computers_built_per_month_l1189_118953


namespace NUMINAMATH_GPT_binary_sum_is_11_l1189_118924

-- Define the binary numbers
def b1 : ℕ := 5  -- equivalent to 101 in binary
def b2 : ℕ := 6  -- equivalent to 110 in binary

-- Define the expected sum in decimal
def expected_sum : ℕ := 11

-- The theorem statement
theorem binary_sum_is_11 : b1 + b2 = expected_sum := by
  sorry

end NUMINAMATH_GPT_binary_sum_is_11_l1189_118924


namespace NUMINAMATH_GPT_find_b_value_l1189_118988

theorem find_b_value (b : ℝ) : (∃ (x y : ℝ), (x, y) = ((2 + 4) / 2, (5 + 9) / 2) ∧ x + y = b) ↔ b = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_b_value_l1189_118988


namespace NUMINAMATH_GPT_sqrt2_times_sqrt5_eq_sqrt10_l1189_118926

theorem sqrt2_times_sqrt5_eq_sqrt10 : (Real.sqrt 2) * (Real.sqrt 5) = Real.sqrt 10 := 
by
  sorry

end NUMINAMATH_GPT_sqrt2_times_sqrt5_eq_sqrt10_l1189_118926


namespace NUMINAMATH_GPT_find_digit_B_l1189_118979

theorem find_digit_B (A B : ℕ) (h : 1 ≤ A ∧ A ≤ 9) (h' : 0 ≤ B ∧ B ≤ 9) (eqn : 10 * A + 22 = 9 * B) : B = 8 := 
  sorry

end NUMINAMATH_GPT_find_digit_B_l1189_118979


namespace NUMINAMATH_GPT_sixth_distance_l1189_118974

theorem sixth_distance (A B C D : Point)
  (dist_AB dist_AC dist_BC dist_AD dist_BD dist_CD : ℝ)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_lengths : (dist_AB = 1 ∧ dist_AC = 1 ∧ dist_BC = 1 ∧ dist_AD = 1) ∨
               (dist_AB = 1 ∧ dist_AC = 1 ∧ dist_BD = 1 ∧ dist_CD = 1) ∨
               (dist_AB = 1 ∧ dist_AD = 1 ∧ dist_BC = 1 ∧ dist_CD = 1) ∨
               (dist_AC = 1 ∧ dist_AD = 1 ∧ dist_BC = 1 ∧ dist_BD = 1) ∨
               (dist_AC = 1 ∧ dist_AD = 1 ∧ dist_BD = 1 ∧ dist_CD = 1) ∨
               (dist_AD = 1 ∧ dist_BC = 1 ∧ dist_BD = 1 ∧ dist_CD = 1))
  (h_one_point_two : dist_AB = 1.2 ∨ dist_AC = 1.2 ∨ dist_BC = 1.2 ∨ dist_AD = 1.2 ∨ dist_BD = 1.2 ∨ dist_CD = 1.2) :
  dist_AB = 1.84 ∨ dist_AB = 0.24 ∨ dist_AB = 1.6 ∨
  dist_AC = 1.84 ∨ dist_AC = 0.24 ∨ dist_AC = 1.6 ∨
  dist_BC = 1.84 ∨ dist_BC = 0.24 ∨ dist_BC = 1.6 ∨
  dist_AD = 1.84 ∨ dist_AD = 0.24 ∨ dist_AD = 1.6 ∨
  dist_BD = 1.84 ∨ dist_BD = 0.24 ∨ dist_BD = 1.6 ∨
  dist_CD = 1.84 ∨ dist_CD = 0.24 ∨ dist_CD = 1.6 :=
sorry

end NUMINAMATH_GPT_sixth_distance_l1189_118974


namespace NUMINAMATH_GPT_incorrect_tripling_radius_l1189_118915

-- Let r be the radius of a circle, and A be its area.
-- The claim is that tripling the radius quadruples the area.
-- We need to prove this claim is incorrect.

theorem incorrect_tripling_radius (r : ℝ) (A : ℝ) (π : ℝ) (hA : A = π * r^2) : 
    (π * (3 * r)^2) ≠ 4 * A :=
by
  sorry

end NUMINAMATH_GPT_incorrect_tripling_radius_l1189_118915


namespace NUMINAMATH_GPT_pi_minus_five_floor_value_l1189_118956

noncomputable def greatest_integer_function (x : ℝ) : ℤ := Int.floor x

theorem pi_minus_five_floor_value :
  greatest_integer_function (Real.pi - 5) = -2 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_pi_minus_five_floor_value_l1189_118956


namespace NUMINAMATH_GPT_cos_alpha_minus_pi_l1189_118989

theorem cos_alpha_minus_pi (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) (h3 : 3 * Real.sin (2 * α) = Real.sin α) : 
  Real.cos (α - Real.pi) = -1/6 := 
by
  sorry

end NUMINAMATH_GPT_cos_alpha_minus_pi_l1189_118989


namespace NUMINAMATH_GPT_arlo_books_l1189_118914

theorem arlo_books (total_items : ℕ) (books_ratio : ℕ) (pens_ratio : ℕ) (notebooks_ratio : ℕ) 
  (ratio_sum : ℕ) (items_per_part : ℕ) (parts_for_books : ℕ) (total_parts : ℕ) :
  total_items = 600 →
  books_ratio = 7 →
  pens_ratio = 3 →
  notebooks_ratio = 2 →
  total_parts = books_ratio + pens_ratio + notebooks_ratio →
  items_per_part = total_items / total_parts →
  parts_for_books = books_ratio →
  parts_for_books * items_per_part = 350 := by
  intros
  sorry

end NUMINAMATH_GPT_arlo_books_l1189_118914


namespace NUMINAMATH_GPT_michael_regular_hours_l1189_118950

-- Define the constants and conditions
def regular_rate : ℝ := 7
def overtime_rate : ℝ := 14
def total_earnings : ℝ := 320
def total_hours : ℝ := 42.857142857142854

-- Declare the proof problem
theorem michael_regular_hours :
  ∃ R O : ℝ, (regular_rate * R + overtime_rate * O = total_earnings) ∧ (R + O = total_hours) ∧ (R = 40) :=
by
  sorry

end NUMINAMATH_GPT_michael_regular_hours_l1189_118950


namespace NUMINAMATH_GPT_smallest_rectangles_cover_square_l1189_118927

theorem smallest_rectangles_cover_square :
  ∃ (n : ℕ), n = 8 ∧ ∀ (a : ℕ), ∀ (b : ℕ), (a = 2) ∧ (b = 4) → 
  ∃ (s : ℕ), s = 8 ∧ (s * s) / (a * b) = n :=
by
  sorry

end NUMINAMATH_GPT_smallest_rectangles_cover_square_l1189_118927


namespace NUMINAMATH_GPT_suitable_survey_l1189_118994

inductive Survey
| FavoriteTVPrograms : Survey
| PrintingErrors : Survey
| BatteryServiceLife : Survey
| InternetUsage : Survey

def is_suitable_for_census (s : Survey) : Prop :=
  match s with
  | Survey.PrintingErrors => True
  | _ => False

theorem suitable_survey : is_suitable_for_census Survey.PrintingErrors = True :=
by
  sorry

end NUMINAMATH_GPT_suitable_survey_l1189_118994


namespace NUMINAMATH_GPT_winning_cards_at_least_one_l1189_118993

def cyclicIndex (n : ℕ) (i : ℕ) : ℕ := (i % n + n) % n

theorem winning_cards_at_least_one (a : ℕ → ℕ) (h : ∀ i, (a (cyclicIndex 8 (i - 1)) + a i + a (cyclicIndex 8 (i + 1))) % 2 = 1) :
  ∀ i, 1 ≤ a i :=
by
  sorry

end NUMINAMATH_GPT_winning_cards_at_least_one_l1189_118993


namespace NUMINAMATH_GPT_prove_travel_cost_l1189_118980

noncomputable def least_expensive_travel_cost
  (a_cost_per_km : ℝ) (a_booking_fee : ℝ) (b_cost_per_km : ℝ)
  (DE DF EF : ℝ) :
  ℝ := by
  let a_cost_DE := DE * a_cost_per_km + a_booking_fee
  let b_cost_DE := DE * b_cost_per_km
  let cheaper_cost_DE := min a_cost_DE b_cost_DE

  let a_cost_EF := EF * a_cost_per_km + a_booking_fee
  let b_cost_EF := EF * b_cost_per_km
  let cheaper_cost_EF := min a_cost_EF b_cost_EF

  let a_cost_DF := DF * a_cost_per_km + a_booking_fee
  let b_cost_DF := DF * b_cost_per_km
  let cheaper_cost_DF := min a_cost_DF b_cost_DF

  exact cheaper_cost_DE + cheaper_cost_EF + cheaper_cost_DF

def travel_problem : Prop :=
  let DE := 5000
  let DF := 4000
  let EF := 2500 -- derived from the Pythagorean theorem
  least_expensive_travel_cost 0.12 120 0.20 DE DF EF = 1740

theorem prove_travel_cost : travel_problem := sorry

end NUMINAMATH_GPT_prove_travel_cost_l1189_118980


namespace NUMINAMATH_GPT_cylinder_surface_area_l1189_118918

theorem cylinder_surface_area (side : ℝ) (h : ℝ) (r : ℝ) : 
  side = 2 ∧ h = side ∧ r = side → 
  (2 * Real.pi * r^2 + 2 * Real.pi * r * h) = 16 * Real.pi := 
by
  intro h
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_l1189_118918


namespace NUMINAMATH_GPT_total_notebooks_correct_l1189_118961

-- Definitions based on conditions
def total_students : ℕ := 28
def half_students : ℕ := total_students / 2
def notebooks_per_student_group1 : ℕ := 5
def notebooks_per_student_group2 : ℕ := 3

-- Total notebooks calculation
def total_notebooks : ℕ :=
  (half_students * notebooks_per_student_group1) + (half_students * notebooks_per_student_group2)

-- Theorem to be proved
theorem total_notebooks_correct : total_notebooks = 112 := by
  sorry

end NUMINAMATH_GPT_total_notebooks_correct_l1189_118961


namespace NUMINAMATH_GPT_quadratic_real_roots_a_leq_2_l1189_118940

theorem quadratic_real_roots_a_leq_2
    (a : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 + 2*a = 0) ∧ (x2^2 - 4*x2 + 2*a = 0)) →
    a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_quadratic_real_roots_a_leq_2_l1189_118940


namespace NUMINAMATH_GPT_dot_product_equals_6_l1189_118902

-- Define the vectors
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)

-- Define the scalar multiplication and addition
def scaled_added_vector : ℝ × ℝ := (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2)

-- Define the dot product
def dot_product : ℝ := scaled_added_vector.1 * vec_a.1 + scaled_added_vector.2 * vec_a.2

-- Assertion that the dot product is equal to 6
theorem dot_product_equals_6 : dot_product = 6 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_equals_6_l1189_118902


namespace NUMINAMATH_GPT_total_apples_l1189_118910

theorem total_apples (x : ℕ) : 
    (x - x / 5 - x / 12 - x / 8 - x / 20 - x / 4 - x / 7 - x / 30 - 4 * (x / 30) - 300 ≤ 50) -> 
    x = 3360 :=
by
    sorry

end NUMINAMATH_GPT_total_apples_l1189_118910


namespace NUMINAMATH_GPT_initial_manufacturing_cost_l1189_118904

theorem initial_manufacturing_cost
  (P : ℝ) -- selling price
  (initial_cost new_cost : ℝ)
  (initial_profit new_profit : ℝ)
  (h1 : initial_profit = 0.25 * P)
  (h2 : new_profit = 0.50 * P)
  (h3 : new_cost = 50)
  (h4 : new_profit = P - new_cost)
  (h5 : initial_profit = P - initial_cost) :
  initial_cost = 75 := 
by
  sorry

end NUMINAMATH_GPT_initial_manufacturing_cost_l1189_118904


namespace NUMINAMATH_GPT_value_of_f_at_9_l1189_118957

def f (n : ℕ) : ℕ := n^3 + n^2 + n + 17

theorem value_of_f_at_9 : f 9 = 836 := sorry

end NUMINAMATH_GPT_value_of_f_at_9_l1189_118957


namespace NUMINAMATH_GPT_monotonic_decreasing_intervals_l1189_118933

theorem monotonic_decreasing_intervals (α : ℝ) (hα : α < 0) :
  (∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → x ^ α > y ^ α) ∧ 
  (∀ x y : ℝ, x < y ∧ 0 < x ∧ 0 < y → x ^ α > y ^ α) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_intervals_l1189_118933


namespace NUMINAMATH_GPT_new_man_weight_l1189_118916

theorem new_man_weight (avg_increase : ℝ) (crew_weight : ℝ) (new_man_weight : ℝ) 
(h_avg_increase : avg_increase = 1.8) (h_crew_weight : crew_weight = 53) :
  new_man_weight = crew_weight + 10 * avg_increase :=
by
  -- Here we will use the conditions to prove the theorem
  sorry

end NUMINAMATH_GPT_new_man_weight_l1189_118916


namespace NUMINAMATH_GPT_cameron_gold_tokens_l1189_118903

/-- Cameron starts with 90 red tokens and 60 blue tokens. 
  Booth 1 exchange: 3 red tokens for 1 gold token and 2 blue tokens.
  Booth 2 exchange: 2 blue tokens for 1 gold token and 1 red token.
  Cameron stops when fewer than 3 red tokens or 2 blue tokens remain.
  Prove that the number of gold tokens Cameron ends up with is 148.
-/
theorem cameron_gold_tokens :
  ∃ (x y : ℕ), 
    90 - 3 * x + y < 3 ∧
    60 + 2 * x - 2 * y < 2 ∧
    (x + y = 148) :=
  sorry

end NUMINAMATH_GPT_cameron_gold_tokens_l1189_118903


namespace NUMINAMATH_GPT_minimum_room_size_for_table_l1189_118985

theorem minimum_room_size_for_table (S : ℕ) :
  (∃ S, S ≥ 13) := sorry

end NUMINAMATH_GPT_minimum_room_size_for_table_l1189_118985


namespace NUMINAMATH_GPT_time_to_pass_platform_l1189_118998

-- Conditions of the problem
def length_of_train : ℕ := 1500
def time_to_cross_tree : ℕ := 100
def length_of_platform : ℕ := 500

-- Derived values according to solution steps
def speed_of_train : ℚ := length_of_train / time_to_cross_tree
def total_distance_to_pass_platform : ℕ := length_of_train + length_of_platform

-- The theorem to be proved
theorem time_to_pass_platform :
  (total_distance_to_pass_platform / speed_of_train : ℚ) = 133.33 := sorry

end NUMINAMATH_GPT_time_to_pass_platform_l1189_118998


namespace NUMINAMATH_GPT_smaller_angle_linear_pair_l1189_118960

theorem smaller_angle_linear_pair (a b : ℝ) (h1 : a + b = 180) (h2 : a = 5 * b) : b = 30 := by
  sorry

end NUMINAMATH_GPT_smaller_angle_linear_pair_l1189_118960


namespace NUMINAMATH_GPT_triangle_inequality_values_l1189_118908

theorem triangle_inequality_values (x : ℕ) :
  x ≥ 2 ∧ x < 10 ↔ (x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9) :=
by sorry

end NUMINAMATH_GPT_triangle_inequality_values_l1189_118908


namespace NUMINAMATH_GPT_find_f_20_l1189_118951

theorem find_f_20 (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = (1/2) * f (x + 2))
  (h2 : f 2 = 1) :
  f 20 = 512 :=
sorry

end NUMINAMATH_GPT_find_f_20_l1189_118951


namespace NUMINAMATH_GPT_lettuce_price_1_l1189_118967

theorem lettuce_price_1 (customers_per_month : ℕ) (lettuce_per_customer : ℕ) (tomatoes_per_customer : ℕ) 
(price_per_tomato : ℝ) (total_sales : ℝ)
  (h_customers : customers_per_month = 500)
  (h_lettuce_per_customer : lettuce_per_customer = 2)
  (h_tomatoes_per_customer : tomatoes_per_customer = 4)
  (h_price_per_tomato : price_per_tomato = 0.5)
  (h_total_sales : total_sales = 2000) :
  let heads_of_lettuce_sold := customers_per_month * lettuce_per_customer
  let tomato_sales := customers_per_month * tomatoes_per_customer * price_per_tomato
  let lettuce_sales := total_sales - tomato_sales
  let price_per_lettuce := lettuce_sales / heads_of_lettuce_sold
  price_per_lettuce = 1 := by
{
  sorry
}

end NUMINAMATH_GPT_lettuce_price_1_l1189_118967


namespace NUMINAMATH_GPT_correct_sampling_method_is_D_l1189_118959

def is_simple_random_sample (method : String) : Prop :=
  method = "drawing lots method to select 3 out of 10 products for quality inspection"

theorem correct_sampling_method_is_D : 
  is_simple_random_sample "drawing lots method to select 3 out of 10 products for quality inspection" :=
sorry

end NUMINAMATH_GPT_correct_sampling_method_is_D_l1189_118959


namespace NUMINAMATH_GPT_range_of_m_l1189_118999

-- Definitions given in the problem
def p (x : ℝ) : Prop := x < -2 ∨ x > 10
def q (x m : ℝ) : Prop := x^2 - 2*x - (m^2 - 1) ≥ 0
def neg_q_sufficient_for_neg_p : Prop :=
  ∀ {x m : ℝ}, (1 - m < x ∧ x < 1 + m) → (-2 ≤ x ∧ x ≤ 10)

-- The statement to prove
theorem range_of_m (m : ℝ) (h1 : m > 0) (h2 : 1 - m ≥ -2) (h3 : 1 + m ≤ 10) :
  0 < m ∧ m ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1189_118999


namespace NUMINAMATH_GPT_AM_GM_proof_equality_condition_l1189_118937

variable (a b : ℝ)
variable (ha : 0 < a) (hb : 0 < b)

theorem AM_GM_proof : (a + b)^3 / (a^2 * b) ≥ 27 / 4 :=
sorry

theorem equality_condition : (a + b)^3 / (a^2 * b) = 27 / 4 ↔ a = 2 * b :=
sorry

end NUMINAMATH_GPT_AM_GM_proof_equality_condition_l1189_118937


namespace NUMINAMATH_GPT_birds_on_fence_l1189_118929

theorem birds_on_fence :
  let i := 12           -- initial birds
  let added1 := 8       -- birds that land first
  let T := i + added1   -- total first stage birds
  
  let fly_away1 := 5
  let join1 := 3
  let W := T - fly_away1 + join1   -- birds after some fly away, others join
  
  let D := W * 2       -- birds doubles
  
  let fly_away2 := D * 0.25  -- 25% fly away
  let D_after_fly_away := D - fly_away2
  
  let return_birds := 2        -- 2.5 birds return, rounded down to 2
  let final_birds := D_after_fly_away + return_birds
  
  final_birds = 29 := 
by {
  sorry
}

end NUMINAMATH_GPT_birds_on_fence_l1189_118929


namespace NUMINAMATH_GPT_find_period_for_interest_l1189_118901

noncomputable def period_for_compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) : ℝ :=
  (Real.log A - Real.log P) / (n * Real.log (1 + r / n))

theorem find_period_for_interest :
  period_for_compound_interest 8000 0.15 1 11109 = 2 := 
sorry

end NUMINAMATH_GPT_find_period_for_interest_l1189_118901


namespace NUMINAMATH_GPT_gcd_cube_sum_condition_l1189_118976

theorem gcd_cube_sum_condition (n : ℕ) (hn : n > 32) : Nat.gcd (n^3 + 125) (n + 5) = 1 := 
  by 
  sorry

end NUMINAMATH_GPT_gcd_cube_sum_condition_l1189_118976


namespace NUMINAMATH_GPT_sum_of_squares_of_rates_l1189_118966

theorem sum_of_squares_of_rates (c j s : ℕ) (cond1 : 3 * c + 2 * j + 2 * s = 80) (cond2 : 2 * j + 2 * s + 4 * c = 104) : 
  c^2 + j^2 + s^2 = 592 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_rates_l1189_118966


namespace NUMINAMATH_GPT_kim_fraction_of_shirts_given_l1189_118921

open Nat

theorem kim_fraction_of_shirts_given (d : ℕ) (s_left : ℕ) (one_dozen := 12) 
  (original_shirts := 4 * one_dozen) 
  (given_shirts := original_shirts - s_left) 
  (fraction_given := given_shirts / original_shirts) 
  (hc1 : d = one_dozen) 
  (hc2 : s_left = 32) 
  : fraction_given = 1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_kim_fraction_of_shirts_given_l1189_118921


namespace NUMINAMATH_GPT_percentage_markup_on_cost_price_l1189_118986

theorem percentage_markup_on_cost_price 
  (SP : ℝ) (CP : ℝ) (hSP : SP = 6400) (hCP : CP = 5565.217391304348) : 
  ((SP - CP) / CP) * 100 = 15 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_percentage_markup_on_cost_price_l1189_118986


namespace NUMINAMATH_GPT_c_alone_finishes_job_in_7_5_days_l1189_118932

theorem c_alone_finishes_job_in_7_5_days (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : A + B + C = 1 / 5) :
  1 / C = 7.5 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_c_alone_finishes_job_in_7_5_days_l1189_118932


namespace NUMINAMATH_GPT_largest_square_side_length_largest_rectangle_dimensions_l1189_118911

variable (a b : ℝ) (h : a > 0) (k : b > 0)

-- Part (a): Side length of the largest possible square
theorem largest_square_side_length (h : a > 0) (k : b > 0) :
  ∃ (s : ℝ), s = (a * b) / (a + b) := sorry

-- Part (b): Dimensions of the largest possible rectangle
theorem largest_rectangle_dimensions (h : a > 0) (k : b > 0) :
  ∃ (x y : ℝ), x = a / 2 ∧ y = b / 2 := sorry

end NUMINAMATH_GPT_largest_square_side_length_largest_rectangle_dimensions_l1189_118911


namespace NUMINAMATH_GPT_smallest_k_l1189_118900

theorem smallest_k (k : ℕ) (h1 : k > 1) (h2 : k % 19 = 1) (h3 : k % 7 = 1) (h4 : k % 3 = 1) : k = 400 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_l1189_118900


namespace NUMINAMATH_GPT_book_total_pages_l1189_118925

-- Define the conditions given in the problem
def pages_per_night : ℕ := 12
def nights_to_finish : ℕ := 10

-- State that the total number of pages in the book is 120 given the conditions
theorem book_total_pages : (pages_per_night * nights_to_finish) = 120 :=
by sorry

end NUMINAMATH_GPT_book_total_pages_l1189_118925


namespace NUMINAMATH_GPT_sum_x_y_is_9_l1189_118934

-- Definitions of the conditions
variables (x y S : ℝ)
axiom h1 : x + y = S
axiom h2 : x - y = 3
axiom h3 : x^2 - y^2 = 27

-- The theorem to prove
theorem sum_x_y_is_9 : S = 9 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_sum_x_y_is_9_l1189_118934


namespace NUMINAMATH_GPT_increase_in_area_400ft2_l1189_118917

theorem increase_in_area_400ft2 (l w : ℝ) (h₁ : l = 60) (h₂ : w = 20)
  (h₃ : 4 * (l + w) = 4 * (4 * (l + w) / 4 / 4 )):
  (4 * (l + w) / 4) ^ 2 - l * w = 400 := by
  sorry

end NUMINAMATH_GPT_increase_in_area_400ft2_l1189_118917


namespace NUMINAMATH_GPT_sherman_total_weekly_driving_time_l1189_118913

def daily_commute_time : Nat := 1  -- 1 hour for daily round trip commute time
def work_days : Nat := 5  -- Sherman works 5 days a week
def weekend_day_driving_time : Nat := 2  -- 2 hours of driving each weekend day
def weekend_days : Nat := 2  -- There are 2 weekend days

theorem sherman_total_weekly_driving_time :
  daily_commute_time * work_days + weekend_day_driving_time * weekend_days = 9 := 
by
  sorry

end NUMINAMATH_GPT_sherman_total_weekly_driving_time_l1189_118913


namespace NUMINAMATH_GPT_exists_prime_seq_satisfying_condition_l1189_118938

theorem exists_prime_seq_satisfying_condition :
  ∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ (∀ m n, m < n → a m < a n) ∧ 
  (∀ i j, i ≠ j → (i * a j, j * a i) = (i, j)) :=
sorry

end NUMINAMATH_GPT_exists_prime_seq_satisfying_condition_l1189_118938


namespace NUMINAMATH_GPT_shifted_line_does_not_pass_third_quadrant_l1189_118943

def line_eq (x: ℝ) : ℝ := -2 * x - 1
def shifted_line_eq (x: ℝ) : ℝ := -2 * (x - 3) - 1

theorem shifted_line_does_not_pass_third_quadrant :
  ¬∃ x y : ℝ, shifted_line_eq x = y ∧ x < 0 ∧ y < 0 :=
sorry

end NUMINAMATH_GPT_shifted_line_does_not_pass_third_quadrant_l1189_118943


namespace NUMINAMATH_GPT_ambulance_reachable_area_l1189_118931

theorem ambulance_reachable_area :
  let travel_time_minutes := 8
  let travel_time_hours := (travel_time_minutes : ℝ) / 60
  let speed_on_road := 60 -- speed in miles per hour
  let speed_off_road := 10 -- speed in miles per hour
  let distance_on_road := speed_on_road * travel_time_hours
  distance_on_road = 8 → -- this verifies the distance covered on road
  let area := (2 * distance_on_road) ^ 2
  area = 256 := sorry

end NUMINAMATH_GPT_ambulance_reachable_area_l1189_118931


namespace NUMINAMATH_GPT_find_x_l1189_118939

variables (x : ℝ)
axiom h1 : (180 / x) + (5 * 12 / x) + 80 = 81

theorem find_x : x = 240 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_l1189_118939


namespace NUMINAMATH_GPT_product_of_numbers_l1189_118920

theorem product_of_numbers (a b : ℕ) (hcf : ℕ := 12) (lcm : ℕ := 205) (ha : Nat.gcd a b = hcf) (hb : Nat.lcm a b = lcm) : a * b = 2460 := by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l1189_118920


namespace NUMINAMATH_GPT_no_primes_in_Q_plus_m_l1189_118990

def Q : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_primes_in_Q_plus_m (m : ℕ) (hm : 2 ≤ m ∧ m ≤ 32) : ¬is_prime (Q + m) :=
by
  sorry  -- Proof would be provided here

end NUMINAMATH_GPT_no_primes_in_Q_plus_m_l1189_118990


namespace NUMINAMATH_GPT_print_papers_in_time_l1189_118906

theorem print_papers_in_time :
  ∃ (n : ℕ), 35 * 15 * n = 500000 * 21 * n := by
  sorry

end NUMINAMATH_GPT_print_papers_in_time_l1189_118906


namespace NUMINAMATH_GPT_dan_money_left_l1189_118952

def initial_money : ℝ := 50.00
def candy_bar_price : ℝ := 1.75
def candy_bar_count : ℕ := 3
def gum_price : ℝ := 0.85
def soda_price : ℝ := 2.25
def sales_tax_rate : ℝ := 0.08

theorem dan_money_left : 
  initial_money - (candy_bar_count * candy_bar_price + gum_price + soda_price) * (1 + sales_tax_rate) = 40.98 :=
by
  sorry

end NUMINAMATH_GPT_dan_money_left_l1189_118952


namespace NUMINAMATH_GPT_incorrect_conclusion_D_l1189_118909

-- Define lines and planes
variables (l m n : Type) -- lines
variables (α β γ : Type) -- planes

-- Define the conditions
def intersection_planes (p1 p2 : Type) : Type := sorry
def perpendicular (a b : Type) : Prop := sorry

-- Given conditions for option D
axiom h1 : intersection_planes α β = m
axiom h2 : intersection_planes β γ = l
axiom h3 : intersection_planes γ α = n
axiom h4 : perpendicular l m
axiom h5 : perpendicular l n

-- Theorem stating that the conclusion of option D is incorrect
theorem incorrect_conclusion_D : ¬ perpendicular m n :=
by sorry

end NUMINAMATH_GPT_incorrect_conclusion_D_l1189_118909


namespace NUMINAMATH_GPT_exists_k_l1189_118935

def satisfies_condition (a b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → (a n - a (n-1)) * (a n - a (n-2)) + (b n - b (n-1)) * (b n - b (n-2)) = 0

theorem exists_k (a b : ℕ → ℤ) 
  (h : satisfies_condition a b) : 
  ∃ k : ℕ, k > 0 ∧ a k = a (k + 2008) :=
sorry

end NUMINAMATH_GPT_exists_k_l1189_118935


namespace NUMINAMATH_GPT_range_of_m_l1189_118975

theorem range_of_m 
  (m : ℝ)
  (hM : -4 ≤ m ∧ m ≤ 4)
  (ellipse : ∀ (x y : ℝ), x^2 / 16 + y^2 / 12 = 1 → y = 0) :
  1 ≤ m ∧ m ≤ 4 := sorry

end NUMINAMATH_GPT_range_of_m_l1189_118975


namespace NUMINAMATH_GPT_original_number_is_twenty_l1189_118963

theorem original_number_is_twenty (x : ℕ) (h : 100 * x = x + 1980) : x = 20 :=
sorry

end NUMINAMATH_GPT_original_number_is_twenty_l1189_118963


namespace NUMINAMATH_GPT_P_72_l1189_118955

def P (n : ℕ) : ℕ :=
  -- The definition of P(n) should enumerate the ways of expressing n as a product
  -- of integers greater than 1, considering the order of factors.
  sorry

theorem P_72 : P 72 = 17 :=
by
  sorry

end NUMINAMATH_GPT_P_72_l1189_118955


namespace NUMINAMATH_GPT_percent_of_class_received_50_to_59_l1189_118954

-- Define the frequencies for each score range
def freq_90_to_100 := 5
def freq_80_to_89 := 7
def freq_70_to_79 := 9
def freq_60_to_69 := 8
def freq_50_to_59 := 4
def freq_below_50 := 3

-- Define the total number of students
def total_students := freq_90_to_100 + freq_80_to_89 + freq_70_to_79 + freq_60_to_69 + freq_50_to_59 + freq_below_50

-- Define the frequency of students scoring in the 50%-59% range
def freq_50_to_59_ratio := (freq_50_to_59 : ℚ) / total_students

-- Define the percentage calculation
def percent_50_to_59 := freq_50_to_59_ratio * 100

theorem percent_of_class_received_50_to_59 :
  percent_50_to_59 = 100 / 9 := 
by {
  sorry
}

end NUMINAMATH_GPT_percent_of_class_received_50_to_59_l1189_118954


namespace NUMINAMATH_GPT_product_gcd_lcm_l1189_118942

theorem product_gcd_lcm (a b : ℕ) (ha : a = 90) (hb : b = 150) :
  Nat.gcd a b * Nat.lcm a b = 13500 := by
  sorry

end NUMINAMATH_GPT_product_gcd_lcm_l1189_118942


namespace NUMINAMATH_GPT_prob_one_boy_one_girl_l1189_118965

-- Defining the probabilities of birth
def prob_boy := 2 / 3
def prob_girl := 1 / 3

-- Calculating the probability of all boys
def prob_all_boys := prob_boy ^ 4

-- Calculating the probability of all girls
def prob_all_girls := prob_girl ^ 4

-- Calculating the probability of having at least one boy and one girl
def prob_at_least_one_boy_and_one_girl := 1 - (prob_all_boys + prob_all_girls)

-- Proof statement
theorem prob_one_boy_one_girl : prob_at_least_one_boy_and_one_girl = 64 / 81 :=
by sorry

end NUMINAMATH_GPT_prob_one_boy_one_girl_l1189_118965
