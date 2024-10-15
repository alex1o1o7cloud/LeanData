import Mathlib

namespace NUMINAMATH_GPT_room_width_l1212_121213

theorem room_width (w : ℝ) (h1 : 21 > 0) (h2 : 2 > 0) 
  (h3 : (25 * (w + 4) - 21 * w = 148)) : w = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_room_width_l1212_121213


namespace NUMINAMATH_GPT_function_ordering_l1212_121225

-- Definitions for the function and conditions
variable (f : ℝ → ℝ)

-- Assuming properties of the function
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodicity : ∀ x, f (x + 4) = -f x
axiom increasing_on : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 2 → f x < f y

-- Main theorem statement
theorem function_ordering : f (-25) < f 80 ∧ f 80 < f 11 :=
by 
  sorry

end NUMINAMATH_GPT_function_ordering_l1212_121225


namespace NUMINAMATH_GPT_seating_arrangement_l1212_121218

theorem seating_arrangement (x y : ℕ) (h : 9 * x + 6 * y = 57) : x = 1 :=
sorry

end NUMINAMATH_GPT_seating_arrangement_l1212_121218


namespace NUMINAMATH_GPT_three_digit_number_l1212_121211

/-- 
Prove there exists three-digit number N such that 
1. N is of form 100a + 10b + c
2. 1 ≤ a ≤ 9
3. 0 ≤ b, c ≤ 9
4. N = 11 * (a + b + c)
--/
theorem three_digit_number (N a b c : ℕ) 
  (hN: N = 100 * a + 10 * b + c) 
  (h_a: 1 ≤ a ∧ a ≤ 9)
  (h_b_c: 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9)
  (h_condition: N = 11 * (a + b + c)) :
  N = 198 := 
sorry

end NUMINAMATH_GPT_three_digit_number_l1212_121211


namespace NUMINAMATH_GPT_domain_transform_l1212_121241

-- Definitions based on conditions
def domain_f_x_plus_1 : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
def domain_f_id : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def domain_f_2x_minus_1 : Set ℝ := { x | 0 ≤ x ∧ x ≤ 5/2 }

-- The theorem to prove the mathematically equivalent problem
theorem domain_transform :
  (∀ x, (x + 1) ∈ domain_f_x_plus_1) →
  (∀ y, y ∈ domain_f_2x_minus_1 ↔ 2 * y - 1 ∈ domain_f_id) :=
by
  sorry

end NUMINAMATH_GPT_domain_transform_l1212_121241


namespace NUMINAMATH_GPT_solve_for_x_l1212_121249

theorem solve_for_x {x : ℝ} (h : -3 * x - 10 = 4 * x + 5) : x = -15 / 7 :=
  sorry

end NUMINAMATH_GPT_solve_for_x_l1212_121249


namespace NUMINAMATH_GPT_flight_duration_sum_l1212_121290

theorem flight_duration_sum 
  (departure_time : ℕ×ℕ) (arrival_time : ℕ×ℕ) (delay : ℕ)
  (h m : ℕ)
  (h0 : 0 < m ∧ m < 60)
  (h1 : departure_time = (9, 20))
  (h2 : arrival_time = (13, 45)) -- using 13 for 1 PM, 24-hour format
  (h3 : delay = 25)
  (h4 : ((arrival_time.1 * 60 + arrival_time.2) - (departure_time.1 * 60 + departure_time.2) + delay) = h * 60 + m) :
  h + m = 29 :=
by {
  -- Proof is skipped
  sorry
}

end NUMINAMATH_GPT_flight_duration_sum_l1212_121290


namespace NUMINAMATH_GPT_bees_on_20th_day_l1212_121260

-- Define the conditions
def initial_bees : ℕ := 1

def companions_per_bee : ℕ := 4

-- Define the total number of bees on day n
def total_bees (n : ℕ) : ℕ :=
  (initial_bees + companions_per_bee) ^ n

-- Statement to prove
theorem bees_on_20th_day : total_bees 20 = 5^20 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_bees_on_20th_day_l1212_121260


namespace NUMINAMATH_GPT_min_overlap_percentage_l1212_121253

theorem min_overlap_percentage (A B : ℝ) (hA : A = 0.9) (hB : B = 0.8) : ∃ x, x = 0.7 := 
by sorry

end NUMINAMATH_GPT_min_overlap_percentage_l1212_121253


namespace NUMINAMATH_GPT_factorize_expr_l1212_121219

theorem factorize_expr (x y : ℝ) : x^3 - 4 * x * y^2 = x * (x + 2 * y) * (x - 2 * y) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expr_l1212_121219


namespace NUMINAMATH_GPT_digging_project_length_l1212_121209

theorem digging_project_length (L : ℝ) (V1 V2 : ℝ) (depth1 length1 depth2 breadth1 breadth2 : ℝ) 
  (h1 : depth1 = 100) (h2 : length1 = 25) (h3 : breadth1 = 30) (h4 : V1 = depth1 * length1 * breadth1)
  (h5 : depth2 = 75) (h6 : breadth2 = 50) (h7 : V2 = depth2 * L * breadth2) (h8 : V1 / V2 = 1) :
  L = 20 :=
by
  sorry

end NUMINAMATH_GPT_digging_project_length_l1212_121209


namespace NUMINAMATH_GPT_dice_probability_sum_three_l1212_121243

theorem dice_probability_sum_three (total_outcomes : ℕ := 36) (favorable_outcomes : ℕ := 2) :
  favorable_outcomes / total_outcomes = 1 / 18 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_sum_three_l1212_121243


namespace NUMINAMATH_GPT_polynomial_perfect_square_value_of_k_l1212_121232

noncomputable def is_perfect_square (p : Polynomial ℝ) : Prop :=
  ∃ (q : Polynomial ℝ), p = q^2

theorem polynomial_perfect_square_value_of_k {k : ℝ} :
  is_perfect_square (Polynomial.X^2 - Polynomial.C k * Polynomial.X + Polynomial.C 25) ↔ (k = 10 ∨ k = -10) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_perfect_square_value_of_k_l1212_121232


namespace NUMINAMATH_GPT_multiplication_72515_9999_l1212_121208

theorem multiplication_72515_9999 : 72515 * 9999 = 725077485 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_72515_9999_l1212_121208


namespace NUMINAMATH_GPT_problem_c_d_sum_l1212_121221

theorem problem_c_d_sum (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (C / (x - 3) + D * (x - 2) = (5 * x ^ 2 - 8 * x - 6) / (x - 3))) : C + D = 20 :=
sorry

end NUMINAMATH_GPT_problem_c_d_sum_l1212_121221


namespace NUMINAMATH_GPT_number_of_integers_with_6_or_7_as_digit_in_base9_l1212_121220

/-- 
  There are 729 smallest positive integers written in base 9.
  We want to determine how many of these integers use the digits 6 or 7 (or both) at least once.
-/
theorem number_of_integers_with_6_or_7_as_digit_in_base9 : 
  ∃ n : ℕ, n = 729 ∧ ∃ m : ℕ, m = n - 7^3 := sorry

end NUMINAMATH_GPT_number_of_integers_with_6_or_7_as_digit_in_base9_l1212_121220


namespace NUMINAMATH_GPT_greatest_divisor_of_arithmetic_sum_l1212_121268

theorem greatest_divisor_of_arithmetic_sum (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ k : ℕ, k = 6 ∧ ∀ a d : ℕ, 12 * a + 66 * d % k = 0 :=
by sorry

end NUMINAMATH_GPT_greatest_divisor_of_arithmetic_sum_l1212_121268


namespace NUMINAMATH_GPT_dot_product_u_v_l1212_121274

def u : ℝ × ℝ × ℝ × ℝ := (4, -3, 5, -2)
def v : ℝ × ℝ × ℝ × ℝ := (-6, 1, 2, 3)

theorem dot_product_u_v : (4 * -6 + -3 * 1 + 5 * 2 + -2 * 3) = -23 := by
  sorry

end NUMINAMATH_GPT_dot_product_u_v_l1212_121274


namespace NUMINAMATH_GPT_anya_more_erasers_l1212_121287

theorem anya_more_erasers (anya_erasers andrea_erasers : ℕ)
  (h1 : anya_erasers = 4 * andrea_erasers)
  (h2 : andrea_erasers = 4) :
  anya_erasers - andrea_erasers = 12 := by
  sorry

end NUMINAMATH_GPT_anya_more_erasers_l1212_121287


namespace NUMINAMATH_GPT_difference_between_x_and_y_l1212_121298

theorem difference_between_x_and_y (x y : ℕ) (h₁ : 3 ^ x * 4 ^ y = 59049) (h₂ : x = 10) : x - y = 10 := by
  sorry

end NUMINAMATH_GPT_difference_between_x_and_y_l1212_121298


namespace NUMINAMATH_GPT_sequence_term_position_l1212_121296

theorem sequence_term_position (n : ℕ) (h : 2 * Real.sqrt 5 = Real.sqrt (3 * n - 1)) : n = 7 :=
sorry

end NUMINAMATH_GPT_sequence_term_position_l1212_121296


namespace NUMINAMATH_GPT_probability_no_defective_pencils_l1212_121258

-- Definitions based on conditions
def total_pencils : ℕ := 11
def defective_pencils : ℕ := 2
def selected_pencils : ℕ := 3

-- Helper function to compute combinations
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The proof statement
theorem probability_no_defective_pencils :
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination (total_pencils - defective_pencils) selected_pencils
  total_ways ≠ 0 → 
  (non_defective_ways / total_ways : ℚ) = 28 / 55 := 
by
  sorry

end NUMINAMATH_GPT_probability_no_defective_pencils_l1212_121258


namespace NUMINAMATH_GPT_cube_less_than_triple_l1212_121265

theorem cube_less_than_triple : ∀ x : ℤ, (x^3 < 3*x) ↔ (x = 1 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_cube_less_than_triple_l1212_121265


namespace NUMINAMATH_GPT_passengers_on_board_l1212_121238

/-- 
Given the fractions of passengers from different continents and remaining 42 passengers,
show that the total number of passengers P is 240.
-/
theorem passengers_on_board :
  ∃ P : ℕ,
    (1 / 3) * (P : ℝ) + (1 / 8) * (P : ℝ) + (1 / 5) * (P : ℝ) + (1 / 6) * (P : ℝ) + 42 = (P : ℝ) ∧ P = 240 :=
by
  let P := 240
  have h : (1 / 3) * (P : ℝ) + (1 / 8) * (P : ℝ) + (1 / 5) * (P : ℝ) + (1 / 6) * (P : ℝ) + 42 = (P : ℝ) := sorry
  exact ⟨P, h, rfl⟩

end NUMINAMATH_GPT_passengers_on_board_l1212_121238


namespace NUMINAMATH_GPT_plant_lamp_arrangement_count_l1212_121226

theorem plant_lamp_arrangement_count :
  let basil_plants := 2
  let aloe_plants := 2
  let white_lamps := 3
  let red_lamps := 3
  (∀ plant, plant = basil_plants ∨ plant = aloe_plants)
  ∧ (∀ lamp, lamp = white_lamps ∨ lamp = red_lamps)
  → (∀ plant, ∃ lamp, plant → lamp)
  → ∃ count, count = 50 := 
by
  sorry

end NUMINAMATH_GPT_plant_lamp_arrangement_count_l1212_121226


namespace NUMINAMATH_GPT_lengths_of_angle_bisectors_areas_of_triangles_l1212_121223

-- Given conditions
variables (x y : ℝ) (S1 S2 : ℝ)
variables (hx1 : x + y = 15) (hx2 : x / y = 3 / 2)
variables (hS1 : S1 / S2 = 9 / 4) (hS2 : S1 - S2 = 6)

-- Prove the lengths of the angle bisectors
theorem lengths_of_angle_bisectors :
  x = 9 ∧ y = 6 :=
by sorry

-- Prove the areas of the triangles
theorem areas_of_triangles :
  S1 = 54 / 5 ∧ S2 = 24 / 5 :=
by sorry

end NUMINAMATH_GPT_lengths_of_angle_bisectors_areas_of_triangles_l1212_121223


namespace NUMINAMATH_GPT_sum_of_first_5_terms_is_55_l1212_121267

variable (a : ℕ → ℝ) -- the arithmetic sequence
variable (d : ℝ) -- the common difference
variable (a_2 : a 2 = 7)
variable (a_4 : a 4 = 15)
noncomputable def sum_of_first_5_terms : ℝ := (5 * (a 2 + a 4)) / 2

theorem sum_of_first_5_terms_is_55 :
  sum_of_first_5_terms a = 55 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_5_terms_is_55_l1212_121267


namespace NUMINAMATH_GPT_perfect_squares_m_l1212_121246

theorem perfect_squares_m (m : ℕ) (hm_pos : m > 0) (hm_min4_square : ∃ a : ℕ, m - 4 = a^2) (hm_plus5_square : ∃ b : ℕ, m + 5 = b^2) : m = 20 ∨ m = 4 :=
by
  sorry

end NUMINAMATH_GPT_perfect_squares_m_l1212_121246


namespace NUMINAMATH_GPT_standard_deviation_upper_bound_l1212_121203

theorem standard_deviation_upper_bound (Mean StdDev : ℝ) (h : Mean = 54) (h2 : 54 - 3 * StdDev > 47) : StdDev < 2.33 :=
by
  sorry

end NUMINAMATH_GPT_standard_deviation_upper_bound_l1212_121203


namespace NUMINAMATH_GPT_sqrt_inequality_l1212_121276

theorem sqrt_inequality : (Real.sqrt 3 + Real.sqrt 7) < 2 * Real.sqrt 5 := 
  sorry

end NUMINAMATH_GPT_sqrt_inequality_l1212_121276


namespace NUMINAMATH_GPT_num_men_in_second_group_l1212_121299

def total_work_hours_week (men: ℕ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℕ :=
  men * hours_per_day * days_per_week

def earnings_per_man_hour (total_earnings: ℕ) (total_work_hours: ℕ) : ℚ :=
  total_earnings / total_work_hours

def required_man_hours (total_earnings: ℕ) (earnings_per_hour: ℚ) : ℚ :=
  total_earnings / earnings_per_hour

def number_of_men (total_man_hours: ℚ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℚ :=
  total_man_hours / (hours_per_day * days_per_week)

theorem num_men_in_second_group :
  let hours_per_day_1 := 10
  let hours_per_day_2 := 6
  let days_per_week := 7
  let men_1 := 4
  let earnings_1 := 1000
  let earnings_2 := 1350
  let work_hours_1 := total_work_hours_week men_1 hours_per_day_1 days_per_week
  let rate_1 := earnings_per_man_hour earnings_1 work_hours_1
  let work_hours_2 := required_man_hours earnings_2 rate_1
  number_of_men work_hours_2 hours_per_day_2 days_per_week = 9 := by
  sorry

end NUMINAMATH_GPT_num_men_in_second_group_l1212_121299


namespace NUMINAMATH_GPT_fraction_simplification_l1212_121245

theorem fraction_simplification (b : ℝ) (hb : b = 3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b^2 = 10 / 81 :=
by
  rw [hb]
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1212_121245


namespace NUMINAMATH_GPT_length_of_faster_train_l1212_121248

-- Definitions for the given conditions
def speed_faster_train_kmh : ℝ := 50
def speed_slower_train_kmh : ℝ := 32
def time_seconds : ℝ := 15

theorem length_of_faster_train : 
  let speed_relative_kmh := speed_faster_train_kmh - speed_slower_train_kmh
  let speed_relative_mps := speed_relative_kmh * (1000 / 3600)
  let length_faster_train := speed_relative_mps * time_seconds
  length_faster_train = 75 := 
by 
  sorry 

end NUMINAMATH_GPT_length_of_faster_train_l1212_121248


namespace NUMINAMATH_GPT_length_of_second_train_is_approximately_159_98_l1212_121280

noncomputable def length_of_second_train : ℝ :=
  let length_first_train := 110 -- meters
  let speed_first_train := 60 -- km/hr
  let speed_second_train := 40 -- km/hr
  let time_to_cross := 9.719222462203025 -- seconds
  let km_per_hr_to_m_per_s := 5 / 18 -- conversion factor from km/hr to m/s
  let relative_speed := (speed_first_train + speed_second_train) * km_per_hr_to_m_per_s -- relative speed in m/s
  let total_distance := relative_speed * time_to_cross -- total distance covered
  total_distance - length_first_train -- length of the second train

theorem length_of_second_train_is_approximately_159_98 :
  abs (length_of_second_train - 159.98) < 0.01 := 
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_GPT_length_of_second_train_is_approximately_159_98_l1212_121280


namespace NUMINAMATH_GPT_confidence_95_implies_K2_gt_3_841_l1212_121285

-- Conditions
def confidence_no_relationship (K2 : ℝ) : Prop := K2 ≤ 3.841
def confidence_related_95 (K2 : ℝ) : Prop := K2 > 3.841
def confidence_related_99 (K2 : ℝ) : Prop := K2 > 6.635

theorem confidence_95_implies_K2_gt_3_841 (K2 : ℝ) :
  confidence_related_95 K2 ↔ K2 > 3.841 :=
by sorry

end NUMINAMATH_GPT_confidence_95_implies_K2_gt_3_841_l1212_121285


namespace NUMINAMATH_GPT_Y_3_2_eq_1_l1212_121230

def Y (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem Y_3_2_eq_1 : Y 3 2 = 1 := by
  sorry

end NUMINAMATH_GPT_Y_3_2_eq_1_l1212_121230


namespace NUMINAMATH_GPT_parabola_focus_coincides_hyperbola_focus_l1212_121278

theorem parabola_focus_coincides_hyperbola_focus (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2 * p * x -> (3,0) = (3,0)) → 
  (∀ x y : ℝ, x^2 / 6 - y^2 / 3 = 1 -> x = 3) → 
  p = 6 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_coincides_hyperbola_focus_l1212_121278


namespace NUMINAMATH_GPT_mean_of_other_four_l1212_121200

theorem mean_of_other_four (a b c d e : ℕ) (h_mean : (a + b + c + d + e + 90) / 6 = 75)
  (h_max : max a (max b (max c (max d (max e 90)))) = 90)
  (h_twice : b = 2 * a) :
  (a + c + d + e) / 4 = 60 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_other_four_l1212_121200


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1212_121255

theorem solution_set_of_inequality (x : ℝ) : 2 * x - 6 < 0 ↔ x < 3 := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1212_121255


namespace NUMINAMATH_GPT_no_integers_exist_l1212_121212

theorem no_integers_exist :
  ¬ (∃ x y : ℤ, (x + 2019) * (x + 2020) + (x + 2020) * (x + 2021) + (x + 2019) * (x + 2021) = y^2) :=
by
  sorry

end NUMINAMATH_GPT_no_integers_exist_l1212_121212


namespace NUMINAMATH_GPT_min_colors_needed_for_boxes_l1212_121283

noncomputable def min_colors_needed : Nat := 23

theorem min_colors_needed_for_boxes :
  ∀ (boxes : Fin 8 → Fin 6 → Nat), 
  (∀ i, ∀ j : Fin 6, boxes i j < min_colors_needed) → 
  (∀ i, (Function.Injective (boxes i))) → 
  (∀ c1 c2, c1 ≠ c2 → (∃! b, ∃ p1 p2, (p1 ≠ p2 ∧ boxes b p1 = c1 ∧ boxes b p2 = c2))) → 
  min_colors_needed = 23 := 
by sorry

end NUMINAMATH_GPT_min_colors_needed_for_boxes_l1212_121283


namespace NUMINAMATH_GPT_min_value_of_M_l1212_121205

noncomputable def f (p q x : ℝ) : ℝ := x^2 + p * x + q

theorem min_value_of_M (p q M : ℝ) :
  (M = max (|f p q 1|) (max (|f p q (-1)|) (|f p q 0|))) →
  (0 > f p q 1 → 0 > f p q (-1) → 0 > f p q 0 → M = 1 / 2) :=
sorry

end NUMINAMATH_GPT_min_value_of_M_l1212_121205


namespace NUMINAMATH_GPT_simplify_fraction_l1212_121264

theorem simplify_fraction (b : ℝ) (h : b ≠ 1) : 
  (b - 1) / (b + b / (b - 1)) = (b - 1) ^ 2 / b ^ 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_fraction_l1212_121264


namespace NUMINAMATH_GPT_mary_change_in_dollars_l1212_121270

theorem mary_change_in_dollars :
  let cost_berries_euros := 7.94
  let cost_peaches_dollars := 6.83
  let exchange_rate := 1.2
  let money_handed_euros := 20
  let money_handed_dollars := 10
  let cost_berries_dollars := cost_berries_euros * exchange_rate
  let total_cost_dollars := cost_berries_dollars + cost_peaches_dollars
  let total_handed_dollars := (money_handed_euros * exchange_rate) + money_handed_dollars
  total_handed_dollars - total_cost_dollars = 17.642 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mary_change_in_dollars_l1212_121270


namespace NUMINAMATH_GPT_ratio_of_sides_l1212_121289

theorem ratio_of_sides (s r : ℝ) (h : s^2 = 2 * r^2 * Real.sqrt 2) : r / s = 1 / Real.sqrt (2 * Real.sqrt 2) := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_sides_l1212_121289


namespace NUMINAMATH_GPT_ellipse_equation_range_of_M_x_coordinate_l1212_121202

-- Proof 1: Proving the equation of the ellipse
theorem ellipse_equation {a b : ℝ} (h_ab : a > b) (h_b0 : b > 0) (e : ℝ)
  (h_e : e = (Real.sqrt 3) / 3) (vertex : ℝ × ℝ) (h_vertex : vertex = (Real.sqrt 3, 0)) :
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ e = (Real.sqrt 3) / 3 ∧ vertex = (Real.sqrt 3, 0) ∧ (∀ (x y : ℝ), (x^2) / 3 + (y^2) / 2 = 1)) :=
sorry

-- Proof 2: Proving the range of x-coordinate of point M
theorem range_of_M_x_coordinate (k : ℝ) (h_k : k ≠ 0) :
  (∃ M_x : ℝ, by sorry) :=
sorry


end NUMINAMATH_GPT_ellipse_equation_range_of_M_x_coordinate_l1212_121202


namespace NUMINAMATH_GPT_value_is_twenty_l1212_121272

theorem value_is_twenty (n : ℕ) (h : n = 16) : 32 - 12 = 20 :=
by {
  -- Simplification of the proof process
  sorry
}

end NUMINAMATH_GPT_value_is_twenty_l1212_121272


namespace NUMINAMATH_GPT_foci_of_ellipse_l1212_121259

def ellipse_focus (x y : ℝ) : Prop :=
  (x = 0 ∧ (y = 12 ∨ y = -12))

theorem foci_of_ellipse :
  ∀ (x y : ℝ), (x^2)/25 + (y^2)/169 = 1 → ellipse_focus x y :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_foci_of_ellipse_l1212_121259


namespace NUMINAMATH_GPT_gum_needed_l1212_121204

-- Definitions based on problem conditions
def num_cousins : ℕ := 4
def gum_per_cousin : ℕ := 5

-- Proposition that we need to prove
theorem gum_needed : num_cousins * gum_per_cousin = 20 := by
  sorry

end NUMINAMATH_GPT_gum_needed_l1212_121204


namespace NUMINAMATH_GPT_least_6_digit_number_sum_of_digits_l1212_121222

-- Definitions based on conditions
def is_6_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def leaves_remainder2 (n : ℕ) (d : ℕ) : Prop := n % d = 2

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Problem statement
theorem least_6_digit_number_sum_of_digits :
  ∃ n : ℕ, is_6_digit n ∧ leaves_remainder2 n 4 ∧ leaves_remainder2 n 610 ∧ leaves_remainder2 n 15 ∧ sum_of_digits n = 17 :=
sorry

end NUMINAMATH_GPT_least_6_digit_number_sum_of_digits_l1212_121222


namespace NUMINAMATH_GPT_solve_y_l1212_121201

theorem solve_y (y : ℝ) : (12 - y)^2 = 4 * y^2 ↔ y = 4 ∨ y = -12 := by
  sorry

end NUMINAMATH_GPT_solve_y_l1212_121201


namespace NUMINAMATH_GPT_phase_shift_of_sine_l1212_121207

theorem phase_shift_of_sine :
  let a := 3
  let b := 4
  let c := - (Real.pi / 4)
  let phase_shift := -(c / b)
  phase_shift = Real.pi / 16 :=
by
  sorry

end NUMINAMATH_GPT_phase_shift_of_sine_l1212_121207


namespace NUMINAMATH_GPT_total_arms_collected_l1212_121269

-- Define the conditions as parameters
def arms_of_starfish := 7 * 5
def arms_of_seastar := 14

-- Define the theorem to prove total arms
theorem total_arms_collected : arms_of_starfish + arms_of_seastar = 49 := by
  sorry

end NUMINAMATH_GPT_total_arms_collected_l1212_121269


namespace NUMINAMATH_GPT_total_peaches_l1212_121273

theorem total_peaches (x : ℕ) (P : ℕ) 
(h1 : P = 6 * x + 57)
(h2 : 6 * x + 57 = 9 * x - 51) : 
  P = 273 :=
by
  sorry

end NUMINAMATH_GPT_total_peaches_l1212_121273


namespace NUMINAMATH_GPT_no_seven_sum_possible_l1212_121271

theorem no_seven_sum_possible :
  let outcomes := [-1, -3, -5, 2, 4, 6]
  ∀ (a b : Int), a ∈ outcomes → b ∈ outcomes → a + b ≠ 7 :=
by
  sorry

end NUMINAMATH_GPT_no_seven_sum_possible_l1212_121271


namespace NUMINAMATH_GPT_quadratic_roots_and_expression_value_l1212_121266

theorem quadratic_roots_and_expression_value :
  let a := 3 + Real.sqrt 21
  let b := 3 - Real.sqrt 21
  (a ≥ b) →
  (∃ x : ℝ, x^2 - 6 * x + 11 = 23) →
  3 * a + 2 * b = 15 + Real.sqrt 21 :=
by
  intros a b h1 h2
  sorry

end NUMINAMATH_GPT_quadratic_roots_and_expression_value_l1212_121266


namespace NUMINAMATH_GPT_set_intersection_l1212_121284

theorem set_intersection (A B : Set ℝ) 
  (hA : A = { x : ℝ | 0 < x ∧ x < 5 }) 
  (hB : B = { x : ℝ | -1 ≤ x ∧ x < 4 }) : 
  (A ∩ B) = { x : ℝ | 0 < x ∧ x < 4 } :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_l1212_121284


namespace NUMINAMATH_GPT_Anne_cleaning_time_l1212_121263

theorem Anne_cleaning_time (B A C : ℚ) 
  (h1 : B + A + C = 1 / 6) 
  (h2 : B + 2 * A + 3 * C = 1 / 2)
  (h3 : B + A = 1 / 4)
  (h4 : B + C = 1 / 3) : 
  A = 1 / 6 := 
sorry

end NUMINAMATH_GPT_Anne_cleaning_time_l1212_121263


namespace NUMINAMATH_GPT_inequality_proof_l1212_121242

theorem inequality_proof
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b) + (b^3 / c^2) + (c^4 / a^3) ≥ -a + 2*b + 2*c :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1212_121242


namespace NUMINAMATH_GPT_find_f_neg_two_l1212_121247

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (h1 : ∀ a b : ℝ, f (a + b) = f a * f b)
variable (h2 : ∀ x : ℝ, f x > 0)
variable (h3 : f 1 = 1 / 2)

-- State the theorem to prove that f(-2) = 4
theorem find_f_neg_two : f (-2) = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_two_l1212_121247


namespace NUMINAMATH_GPT_half_angle_second_quadrant_l1212_121240

theorem half_angle_second_quadrant (k : ℤ) (α : ℝ) (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
    ∃ j : ℤ, (j * π + π / 4 < α / 2 ∧ α / 2 < j * π + π / 2) ∨ (j * π + 5 * π / 4 < α / 2 ∧ α / 2 < (j + 1) * π / 2) :=
sorry

end NUMINAMATH_GPT_half_angle_second_quadrant_l1212_121240


namespace NUMINAMATH_GPT_solution_mixture_l1212_121233

/-
  Let X be a solution that is 10% alcohol by volume.
  Let Y be a solution that is 30% alcohol by volume.
  We define the final solution to be 22% alcohol by volume.
  We need to prove that the amount of solution Y that needs
  to be added to 300 milliliters of solution X to achieve this 
  concentration is 450 milliliters.
-/

theorem solution_mixture (y : ℝ) : 
  (0.10 * 300) + (0.30 * y) = 0.22 * (300 + y) → 
  y = 450 :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_mixture_l1212_121233


namespace NUMINAMATH_GPT_range_of_m_l1212_121261

theorem range_of_m (m : ℝ) :
  (3 * 1 - 2 + m) * (3 * 1 - 1 + m) < 0 →
  -2 < m ∧ m < -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l1212_121261


namespace NUMINAMATH_GPT_rectangle_area_l1212_121286

theorem rectangle_area (length : ℝ) (width_dm : ℝ) (width_m : ℝ) (h1 : length = 8) (h2 : width_dm = 50) (h3 : width_m = width_dm / 10) : 
  (length * width_m = 40) :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_area_l1212_121286


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1212_121237

theorem arithmetic_sequence_fifth_term (x : ℝ) (a₂ : ℝ := x) (a₃ : ℝ := 3) 
    (a₁ : ℝ := -1) (h₁ : a₂ = a₁ + (1*(x + 1))) (h₂ : a₃ = a₁ + 2*(x + 1)) : 
    a₁ + 4*(a₃ - a₂ + 1) = 7 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1212_121237


namespace NUMINAMATH_GPT_ratio_eq_one_l1212_121244

theorem ratio_eq_one {a b : ℝ} (h1 : 4 * a^2 = 5 * b^3) (h2 : a ≠ 0 ∧ b ≠ 0) : (a^2 / 5) / (b^3 / 4) = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_eq_one_l1212_121244


namespace NUMINAMATH_GPT_find_4_digit_number_l1212_121293

theorem find_4_digit_number (a b c d : ℕ) 
(h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182)
(h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
(h6 : 0 ≤ c) (h7 : c ≤ 9) (h8 : 1 ≤ d) (h9 : d ≤ 9) : 
1000 * a + 100 * b + 10 * c + d = 1909 :=
sorry

end NUMINAMATH_GPT_find_4_digit_number_l1212_121293


namespace NUMINAMATH_GPT_find_nearest_integer_x_minus_y_l1212_121231

variable (x y : ℝ)

theorem find_nearest_integer_x_minus_y
  (h1 : abs x + y = 5)
  (h2 : abs x * y - x^3 = 0)
  (hx : x ≠ 0)
  (hy : y ≠ 0) :
  |x - y| = 5 := sorry

end NUMINAMATH_GPT_find_nearest_integer_x_minus_y_l1212_121231


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l1212_121250

theorem trigonometric_identity_proof (alpha : Real)
(h1 : Real.tan (alpha + π / 4) = 1 / 2)
(h2 : -π / 2 < alpha ∧ alpha < 0) :
  (2 * Real.sin alpha ^ 2 + Real.sin (2 * alpha)) / Real.cos (alpha - π / 4) = - (2 * Real.sqrt 5) / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l1212_121250


namespace NUMINAMATH_GPT_find_u_plus_v_l1212_121262

-- Conditions: 3u - 4v = 17 and 5u - 2v = 1.
-- Question: Find the value of u + v.

theorem find_u_plus_v (u v : ℚ) (h1 : 3 * u - 4 * v = 17) (h2 : 5 * u - 2 * v = 1) : u + v = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_u_plus_v_l1212_121262


namespace NUMINAMATH_GPT_profit_maximization_problem_l1212_121288

-- Step 1: Define the data points and linear function
def data_points : List (ℝ × ℝ) := [(65, 70), (70, 60), (75, 50), (80, 40)]

-- Step 2: Define the linear function between y and x
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Step 3: Define cost and profit function
def cost_per_kg : ℝ := 60
def profit_function (y x : ℝ) : ℝ := y * (x - cost_per_kg)

-- Step 4: The main problem statement
theorem profit_maximization_problem :
  ∃ (k b : ℝ), 
  (∀ (x₁ x₂ : ℝ), (x₁, y₁) ∈ data_points ∧ (x₂, y₂) ∈ data_points → linear_function k b x₁ = y₁ ∧ linear_function k b x₂ = y₂) ∧
  ∃ (x : ℝ), profit_function (linear_function k b x) x = 600 ∧
  ∀ x : ℝ, -2 * x^2 + 320 * x - 12000 ≤ -2 * 80^2 + 320 * 80 - 12000
  :=
sorry

end NUMINAMATH_GPT_profit_maximization_problem_l1212_121288


namespace NUMINAMATH_GPT_Jana_taller_than_Kelly_l1212_121295

-- Definitions and given conditions
def Jess_height := 72
def Jana_height := 74
def Kelly_height := Jess_height - 3

-- Proof statement
theorem Jana_taller_than_Kelly : Jana_height - Kelly_height = 5 := by
  sorry

end NUMINAMATH_GPT_Jana_taller_than_Kelly_l1212_121295


namespace NUMINAMATH_GPT_ratio_of_areas_of_triangles_l1212_121206

theorem ratio_of_areas_of_triangles 
  (a b c d e f : ℕ)
  (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (h4 : d = 9) (h5 : e = 40) (h6 : f = 41) : 
  (84 : ℚ) / (180 : ℚ) = 7 / 15 := by
  have hPQR : a^2 + b^2 = c^2 := by
    rw [h1, h2, h3]
    norm_num
  have hSTU : d^2 + e^2 = f^2 := by
    rw [h4, h5, h6]
    norm_num
  have areaPQR : (1/2 : ℚ) * a * b = 84 := by
    rw [h1, h2]
    norm_num
  have areaSTU : (1/2 : ℚ) * d * e = 180 := by
    rw [h4, h5]
    norm_num
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_triangles_l1212_121206


namespace NUMINAMATH_GPT_students_disliked_menu_l1212_121256

theorem students_disliked_menu (total_students liked_students : ℕ) (h1 : total_students = 400) (h2 : liked_students = 235) : total_students - liked_students = 165 :=
by 
  sorry

end NUMINAMATH_GPT_students_disliked_menu_l1212_121256


namespace NUMINAMATH_GPT_Coe_speed_theorem_l1212_121214

-- Define the conditions
def Teena_speed : ℝ := 55
def initial_distance_behind : ℝ := 7.5
def time_hours : ℝ := 1.5
def distance_ahead : ℝ := 15

-- Define Coe's speed
def Coe_speed := 50

-- State the theorem
theorem Coe_speed_theorem : 
  let distance_Teena_covers := Teena_speed * time_hours
  let total_relative_distance := distance_Teena_covers + initial_distance_behind
  let distance_Coe_covers := total_relative_distance - distance_ahead
  let computed_Coe_speed := distance_Coe_covers / time_hours
  computed_Coe_speed = Coe_speed :=
by sorry

end NUMINAMATH_GPT_Coe_speed_theorem_l1212_121214


namespace NUMINAMATH_GPT_initial_chocolate_amount_l1212_121239

-- Define the problem conditions

def initial_dough (d : ℕ) := d = 36
def left_over_chocolate (lo_choc : ℕ) := lo_choc = 4
def chocolate_percentage (p : ℚ) := p = 0.20
def total_weight (d : ℕ) (c_choc : ℕ) := d + c_choc - 4
def chocolate_used (c_choc : ℕ) (lo_choc : ℕ) := c_choc - lo_choc

-- The main proof goal
theorem initial_chocolate_amount (d : ℕ) (lo_choc : ℕ) (p : ℚ) (C : ℕ) :
  initial_dough d → left_over_chocolate lo_choc → chocolate_percentage p →
  p * (total_weight d C) = chocolate_used C lo_choc → C = 13 :=
by
  intros hd hlc hp h
  sorry

end NUMINAMATH_GPT_initial_chocolate_amount_l1212_121239


namespace NUMINAMATH_GPT_quadratic_has_real_roots_iff_l1212_121229

theorem quadratic_has_real_roots_iff (k : ℝ) (hk : k ≠ 0) :
  (∃ x : ℝ, k * x^2 - x + 1 = 0) ↔ k ≤ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_iff_l1212_121229


namespace NUMINAMATH_GPT_annual_increase_rate_l1212_121252

theorem annual_increase_rate (PV FV : ℝ) (n : ℕ) (r : ℝ) :
  PV = 32000 ∧ FV = 40500 ∧ n = 2 ∧ FV = PV * (1 + r)^2 → r = 0.125 :=
by
  sorry

end NUMINAMATH_GPT_annual_increase_rate_l1212_121252


namespace NUMINAMATH_GPT_cos_squared_alpha_plus_pi_over_4_l1212_121292

theorem cos_squared_alpha_plus_pi_over_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + Real.pi / 4) ^ 2 = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_cos_squared_alpha_plus_pi_over_4_l1212_121292


namespace NUMINAMATH_GPT_smallest_prime_l1212_121217

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ , m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n 

theorem smallest_prime :
  ∃ n : ℕ, n = 29 ∧ 
  n >= 10 ∧ n < 100 ∧
  is_prime n ∧
  ((n / 10) = 3) ∧ 
  is_composite (n % 10 * 10 + n / 10) ∧
  (n % 10 * 10 + n / 10) % 5 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_prime_l1212_121217


namespace NUMINAMATH_GPT_stratified_sampling_first_level_l1212_121224

-- Definitions from the conditions
def num_senior_teachers : ℕ := 90
def num_first_level_teachers : ℕ := 120
def num_second_level_teachers : ℕ := 170
def total_teachers : ℕ := num_senior_teachers + num_first_level_teachers + num_second_level_teachers
def sample_size : ℕ := 38

-- Definition of the stratified sampling result
def num_first_level_selected : ℕ := (num_first_level_teachers * sample_size) / total_teachers

-- The statement to be proven
theorem stratified_sampling_first_level : num_first_level_selected = 12 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_first_level_l1212_121224


namespace NUMINAMATH_GPT_greatest_possible_q_minus_r_l1212_121291

theorem greatest_possible_q_minus_r :
  ∃ (q r : ℕ), 945 = 21 * q + r ∧ 0 ≤ r ∧ r < 21 ∧ q - r = 45 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_q_minus_r_l1212_121291


namespace NUMINAMATH_GPT_same_terminal_side_l1212_121281

theorem same_terminal_side : 
  let θ1 := 23 * Real.pi / 3
  let θ2 := 5 * Real.pi / 3
  (∃ k : ℤ, θ1 - 2 * k * Real.pi = θ2) :=
sorry

end NUMINAMATH_GPT_same_terminal_side_l1212_121281


namespace NUMINAMATH_GPT_repeating_pattern_sum_23_l1212_121257

def repeating_pattern_sum (n : ℕ) : ℤ :=
  let pattern := [4, -3, 2, -1, 0]
  let block_sum := List.sum pattern
  let complete_blocks := n / pattern.length
  let remainder := n % pattern.length
  complete_blocks * block_sum + List.sum (pattern.take remainder)

theorem repeating_pattern_sum_23 : repeating_pattern_sum 23 = 11 := 
  sorry

end NUMINAMATH_GPT_repeating_pattern_sum_23_l1212_121257


namespace NUMINAMATH_GPT_cost_of_outfit_l1212_121215

theorem cost_of_outfit (P T J : ℝ) 
  (h1 : 4 * P + 8 * T + 2 * J = 2400)
  (h2 : 2 * P + 14 * T + 3 * J = 2400)
  (h3 : 3 * P + 6 * T = 1500) :
  P + 4 * T + J = 860 := 
sorry

end NUMINAMATH_GPT_cost_of_outfit_l1212_121215


namespace NUMINAMATH_GPT_larger_integer_value_l1212_121294

theorem larger_integer_value (a b : ℕ) (h1 : a * b = 189) (h2 : a / gcd a b = 7 ∧ b / gcd a b = 3 ∨ a / gcd a b = 3 ∧ b / gcd a b = 7) : max a b = 21 :=
by
  sorry

end NUMINAMATH_GPT_larger_integer_value_l1212_121294


namespace NUMINAMATH_GPT_greatest_possible_average_speed_l1212_121279

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

theorem greatest_possible_average_speed :
  ∀ (o₁ o₂ : ℕ) (v_max t : ℝ), 
  is_palindrome o₁ → 
  is_palindrome o₂ → 
  o₁ = 12321 → 
  t = 2 ∧ v_max = 65 → 
  (∃ d, d = o₂ - o₁ ∧ d / t <= v_max) → 
  d / t = v_max :=
sorry

end NUMINAMATH_GPT_greatest_possible_average_speed_l1212_121279


namespace NUMINAMATH_GPT_xiaoming_problem_l1212_121275

theorem xiaoming_problem (a x : ℝ) 
  (h1 : 20.18 * a - 20.18 = x)
  (h2 : x = 2270.25) : 
  a = 113.5 := 
by 
  sorry

end NUMINAMATH_GPT_xiaoming_problem_l1212_121275


namespace NUMINAMATH_GPT_final_price_for_tiffany_l1212_121235

noncomputable def calculate_final_price (n : ℕ) (c : ℝ) (d : ℝ) (s : ℝ) : ℝ :=
  let total_cost := n * c
  let discount := d * total_cost
  let discounted_price := total_cost - discount
  let sales_tax := s * discounted_price
  let final_price := discounted_price + sales_tax
  final_price

theorem final_price_for_tiffany :
  calculate_final_price 9 4.50 0.20 0.07 = 34.67 :=
by
  sorry

end NUMINAMATH_GPT_final_price_for_tiffany_l1212_121235


namespace NUMINAMATH_GPT_regular_octahedron_has_4_pairs_l1212_121227

noncomputable def regular_octahedron_parallel_edges : ℕ :=
  4

theorem regular_octahedron_has_4_pairs
  (h : true) : regular_octahedron_parallel_edges = 4 :=
by
  sorry

end NUMINAMATH_GPT_regular_octahedron_has_4_pairs_l1212_121227


namespace NUMINAMATH_GPT_work_done_by_b_l1212_121251

theorem work_done_by_b (x : ℝ) (h1 : (1/6) + (1/13) = (1/x)) : x = 78/7 :=
  sorry

end NUMINAMATH_GPT_work_done_by_b_l1212_121251


namespace NUMINAMATH_GPT_unique_integer_solution_l1212_121210

theorem unique_integer_solution (a b : ℤ) : 
  ∀ x₁ x₂ : ℤ, (x₁ - a) * (x₁ - b) * (x₁ - 3) + 1 = 0 ∧ (x₂ - a) * (x₂ - b) * (x₂ - 3) + 1 = 0 → x₁ = x₂ :=
by
  sorry

end NUMINAMATH_GPT_unique_integer_solution_l1212_121210


namespace NUMINAMATH_GPT_fraction_square_eq_decimal_l1212_121216

theorem fraction_square_eq_decimal :
  ∃ (x : ℚ), x^2 = 0.04000000000000001 ∧ x = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_square_eq_decimal_l1212_121216


namespace NUMINAMATH_GPT_solve_for_x_l1212_121234

-- Definitions and conditions from a) directly 
def f (x : ℝ) : ℝ := 64 * (2 * x - 1) ^ 3

-- Lean 4 statement to prove the problem
theorem solve_for_x (x : ℝ) : f x = 27 → x = 7 / 8 :=
by
  intro h
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_solve_for_x_l1212_121234


namespace NUMINAMATH_GPT_binom_computation_l1212_121236

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k+1     => 0
| n+1, k+1   => binom n k + binom n (k+1)

theorem binom_computation :
  (binom 10 3) * (binom 8 3) = 6720 := by
  sorry

end NUMINAMATH_GPT_binom_computation_l1212_121236


namespace NUMINAMATH_GPT_store_profit_is_33_percent_l1212_121228

noncomputable def store_profit (C : ℝ) : ℝ :=
  let initial_markup := 1.20 * C
  let new_year_markup := initial_markup + 0.25 * initial_markup
  let february_discount := new_year_markup * 0.92
  let shipping_cost := C * 1.05
  (february_discount - shipping_cost)

theorem store_profit_is_33_percent (C : ℝ) : store_profit C = 0.33 * C :=
by
  sorry

end NUMINAMATH_GPT_store_profit_is_33_percent_l1212_121228


namespace NUMINAMATH_GPT_evaluate_K_l1212_121297

theorem evaluate_K : ∃ K : ℕ, 32^2 * 4^4 = 2^K ∧ K = 18 := by
  use 18
  sorry

end NUMINAMATH_GPT_evaluate_K_l1212_121297


namespace NUMINAMATH_GPT_equation_has_two_solutions_l1212_121254

theorem equation_has_two_solutions : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ∀ x : ℝ, ¬ ( |x - 1| = |x - 2| + |x - 3| ) ↔ (x ≠ x₁ ∧ x ≠ x₂) :=
sorry

end NUMINAMATH_GPT_equation_has_two_solutions_l1212_121254


namespace NUMINAMATH_GPT_prob_not_same_city_is_056_l1212_121277

def probability_not_same_city (P_A_cityA P_B_cityA : ℝ) : ℝ :=
  let P_A_cityB := 1 - P_A_cityA
  let P_B_cityB := 1 - P_B_cityA
  (P_A_cityA * P_B_cityB) + (P_A_cityB * P_B_cityA)

theorem prob_not_same_city_is_056 :
  probability_not_same_city 0.6 0.2 = 0.56 :=
by
  sorry

end NUMINAMATH_GPT_prob_not_same_city_is_056_l1212_121277


namespace NUMINAMATH_GPT_apples_per_bucket_l1212_121282

theorem apples_per_bucket (total_apples buckets : ℕ) (h1 : total_apples = 56) (h2 : buckets = 7) : 
  (total_apples / buckets) = 8 :=
by
  sorry

end NUMINAMATH_GPT_apples_per_bucket_l1212_121282
