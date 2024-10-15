import Mathlib

namespace NUMINAMATH_GPT_compare_b_d_l1072_107208

noncomputable def percentage_increase (x : ℝ) (p : ℝ) := x * (1 + p)
noncomputable def percentage_decrease (x : ℝ) (p : ℝ) := x * (1 - p)

theorem compare_b_d (a b c d : ℝ)
  (h1 : 0 < b)
  (h2 : a = percentage_increase b 0.02)
  (h3 : c = percentage_decrease a 0.01)
  (h4 : d = percentage_decrease c 0.01) :
  b > d :=
sorry

end NUMINAMATH_GPT_compare_b_d_l1072_107208


namespace NUMINAMATH_GPT_slope_of_line_l1072_107233

theorem slope_of_line (x y : ℝ) (h : 6 * x + 7 * y - 3 = 0) : - (6 / 7) = -6 / 7 := 
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l1072_107233


namespace NUMINAMATH_GPT_circumference_of_tire_l1072_107291

theorem circumference_of_tire (rotations_per_minute : ℕ) (speed_kmh : ℕ) 
  (h1 : rotations_per_minute = 400) (h2 : speed_kmh = 72) :
  let speed_mpm := speed_kmh * 1000 / 60
  let circumference := speed_mpm / rotations_per_minute
  circumference = 3 :=
by
  sorry

end NUMINAMATH_GPT_circumference_of_tire_l1072_107291


namespace NUMINAMATH_GPT_max_f_of_polynomial_l1072_107292

theorem max_f_of_polynomial (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (h_poly : ∃ p : Polynomial ℝ, ∀ x, f x = Polynomial.eval x p)
    (h1 : f 4 = 16)
    (h2 : f 16 = 512) :
    f 8 ≤ 64 :=
by
  sorry

end NUMINAMATH_GPT_max_f_of_polynomial_l1072_107292


namespace NUMINAMATH_GPT_sum_of_children_ages_l1072_107256

theorem sum_of_children_ages :
  ∃ E: ℕ, E = 12 ∧ 
  (∃ a b c d e : ℕ, a = E ∧ b = E - 2 ∧ c = E - 4 ∧ d = E - 6 ∧ e = E - 8 ∧ 
   a + b + c + d + e = 40) :=
sorry

end NUMINAMATH_GPT_sum_of_children_ages_l1072_107256


namespace NUMINAMATH_GPT_lisa_cleaning_time_l1072_107253

theorem lisa_cleaning_time (L : ℝ) (h1 : (1 / L) + (1 / 12) = 1 / 4.8) : L = 8 :=
sorry

end NUMINAMATH_GPT_lisa_cleaning_time_l1072_107253


namespace NUMINAMATH_GPT_range_of_n_l1072_107246

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.exp 1) * Real.exp x + (1 / 2) * x^2 - x

theorem range_of_n :
  (∃ m : ℝ, f m ≤ 2 * n^2 - n) ↔ (n ≤ -1/2 ∨ 1 ≤ n) :=
sorry

end NUMINAMATH_GPT_range_of_n_l1072_107246


namespace NUMINAMATH_GPT_fraction_divisible_by_1963_l1072_107275

theorem fraction_divisible_by_1963 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ,
    13 * 733^n + 1950 * 582^n = 1963 * k ∧
    ∃ m : ℤ,
      333^n - 733^n - 1068^n + 431^n = 1963 * m :=
by
  sorry

end NUMINAMATH_GPT_fraction_divisible_by_1963_l1072_107275


namespace NUMINAMATH_GPT_problem1_correct_solution_problem2_correct_solution_l1072_107216

noncomputable def g (x a : ℝ) : ℝ := |x| + 2 * |x + 2 - a|

/-- 
    Prove that the set {x | -2/3 ≤ x ≤ 2} satisfies g(x) ≤ 4 when a = 3 
--/
theorem problem1_correct_solution (x : ℝ) : g x 3 ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 :=
by
  sorry

noncomputable def f (x a : ℝ) : ℝ := g (x - 2) a

/-- 
    Prove that the range of a such that f(x) ≥ 1 for all x ∈ ℝ 
    is a ≤ 1 or a ≥ 3
--/
theorem problem2_correct_solution (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_correct_solution_problem2_correct_solution_l1072_107216


namespace NUMINAMATH_GPT_cistern_problem_l1072_107202

theorem cistern_problem (T : ℝ) (h1 : (1 / 2 - 1 / T) = 1 / 2.571428571428571) : T = 9 :=
by
  sorry

end NUMINAMATH_GPT_cistern_problem_l1072_107202


namespace NUMINAMATH_GPT_value_of_c_l1072_107223

theorem value_of_c (a b c : ℕ) (hab : b = 1) (hd : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_pow : (10 * a + b)^2 = 100 * c + 10 * c + b) (h_gt : 100 * c + 10 * c + b > 300) : 
  c = 4 :=
sorry

end NUMINAMATH_GPT_value_of_c_l1072_107223


namespace NUMINAMATH_GPT_penelope_saving_days_l1072_107265

theorem penelope_saving_days :
  ∀ (daily_savings total_saved : ℕ),
  daily_savings = 24 ∧ total_saved = 8760 →
    total_saved / daily_savings = 365 :=
by
  rintro _ _ ⟨rfl, rfl⟩
  sorry

end NUMINAMATH_GPT_penelope_saving_days_l1072_107265


namespace NUMINAMATH_GPT_angle_measure_l1072_107211

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end NUMINAMATH_GPT_angle_measure_l1072_107211


namespace NUMINAMATH_GPT_distinct_remainders_mod_3n_l1072_107288

open Nat

theorem distinct_remainders_mod_3n 
  (n : ℕ) 
  (hn_odd : Odd n)
  (ai : ℕ → ℕ)
  (bi : ℕ → ℕ)
  (ai_def : ∀ i, 1 ≤ i ∧ i ≤ n → ai i = 3*i - 2)
  (bi_def : ∀ i, 1 ≤ i ∧ i ≤ n → bi i = 3*i - 3)
  (k : ℕ) 
  (hk : 0 < k ∧ k < n)
  : ∀ i, 1 ≤ i ∧ i ≤ n → (∀ j, 1 ≤ j ∧ j ≤ n → i ≠ j →
     ∀ ⦃ r s t u v : ℕ ⦄, 
       (r = (ai i + ai (i % n + 1)) % (3*n) ∧ 
        s = (ai i + bi i) % (3*n) ∧ 
        t = (bi i + bi ((i + k) % n + 1)) % (3*n)) →
       r ≠ s ∧ s ≠ t ∧ t ≠ r) := 
sorry

end NUMINAMATH_GPT_distinct_remainders_mod_3n_l1072_107288


namespace NUMINAMATH_GPT_gravel_weight_is_correct_l1072_107270

def weight_of_gravel (total_weight : ℝ) (fraction_sand : ℝ) (fraction_water : ℝ) : ℝ :=
  total_weight - (fraction_sand * total_weight + fraction_water * total_weight)

theorem gravel_weight_is_correct :
  weight_of_gravel 23.999999999999996 (1 / 3) (1 / 4) = 10 :=
by
  sorry

end NUMINAMATH_GPT_gravel_weight_is_correct_l1072_107270


namespace NUMINAMATH_GPT_investment_final_value_l1072_107241

theorem investment_final_value 
  (original_investment : ℝ) 
  (increase_percentage : ℝ) 
  (original_investment_eq : original_investment = 12500)
  (increase_percentage_eq : increase_percentage = 2.15) : 
  original_investment * (1 + increase_percentage) = 39375 := 
by
  sorry

end NUMINAMATH_GPT_investment_final_value_l1072_107241


namespace NUMINAMATH_GPT_probability_first_green_then_blue_l1072_107269

variable {α : Type} [Fintype α]

noncomputable def prob_first_green_second_blue : ℚ := 
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := (green_marbles : ℚ) / total_marbles
  let prob_second_blue := (blue_marbles : ℚ) / (total_marbles - 1)
  (prob_first_green * prob_second_blue)

theorem probability_first_green_then_blue :
  prob_first_green_second_blue = 4 / 15 := by
  sorry

end NUMINAMATH_GPT_probability_first_green_then_blue_l1072_107269


namespace NUMINAMATH_GPT_expression_of_fn_l1072_107248

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
if n = 0 then x else f (n - 1) x / (1 + n * x)

theorem expression_of_fn (n : ℕ) (x : ℝ) (hn : 1 ≤ n) : f n x = x / (1 + n * x) :=
sorry

end NUMINAMATH_GPT_expression_of_fn_l1072_107248


namespace NUMINAMATH_GPT_magician_ball_count_l1072_107237

theorem magician_ball_count (k : ℕ) : ∃ k : ℕ, 6 * k + 7 = 1993 :=
by sorry

end NUMINAMATH_GPT_magician_ball_count_l1072_107237


namespace NUMINAMATH_GPT_find_x_l1072_107282

-- Given condition: 144 / x = 14.4 / 0.0144
theorem find_x (x : ℝ) (h : 144 / x = 14.4 / 0.0144) : x = 0.144 := by
  sorry

end NUMINAMATH_GPT_find_x_l1072_107282


namespace NUMINAMATH_GPT_find_other_number_l1072_107215

theorem find_other_number (x : ℕ) (h1 : 10 + x = 30) : x = 20 := by
  sorry

end NUMINAMATH_GPT_find_other_number_l1072_107215


namespace NUMINAMATH_GPT_correct_result_l1072_107264

theorem correct_result (x : ℝ) (h : x / 6 = 52) : x + 40 = 352 := by
  sorry

end NUMINAMATH_GPT_correct_result_l1072_107264


namespace NUMINAMATH_GPT_rectangle_side_lengths_l1072_107249

theorem rectangle_side_lengths (x y : ℝ) (h1 : 2 * x + 4 = 10) (h2 : 8 * y - 2 = 10) : x + y = 4.5 := by
  sorry

end NUMINAMATH_GPT_rectangle_side_lengths_l1072_107249


namespace NUMINAMATH_GPT_winnie_proof_l1072_107225

def winnie_problem : Prop :=
  let initial_count := 2017
  let multiples_of_3 := initial_count / 3
  let multiples_of_6 := initial_count / 6
  let multiples_of_27 := initial_count / 27
  let multiples_to_erase_3 := multiples_of_3
  let multiples_to_reinstate_6 := multiples_of_6
  let multiples_to_erase_27 := multiples_of_27
  let final_count := initial_count - multiples_to_erase_3 + multiples_to_reinstate_6 - multiples_to_erase_27
  initial_count - final_count = 373

theorem winnie_proof : winnie_problem := by
  sorry

end NUMINAMATH_GPT_winnie_proof_l1072_107225


namespace NUMINAMATH_GPT_square_roots_equal_implication_l1072_107293

theorem square_roots_equal_implication (b : ℝ) (h : 5 * b = 3 + 2 * b) : -b = -1 := 
by sorry

end NUMINAMATH_GPT_square_roots_equal_implication_l1072_107293


namespace NUMINAMATH_GPT_product_is_zero_l1072_107226

theorem product_is_zero (b : ℤ) (h : b = 3) :
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b * (b + 1) * (b + 2) = 0 :=
by {
  -- Substituting b = 3
  -- (3-5) * (3-4) * (3-3) * (3-2) * (3-1) * 3 * (3+1) * (3+2)
  -- = (-2) * (-1) * 0 * 1 * 2 * 3 * 4 * 5
  -- = 0
  sorry
}

end NUMINAMATH_GPT_product_is_zero_l1072_107226


namespace NUMINAMATH_GPT_sin_max_value_l1072_107227

open Real

theorem sin_max_value (a b : ℝ) (h : sin (a + b) = sin a + sin b) :
  sin a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_sin_max_value_l1072_107227


namespace NUMINAMATH_GPT_find_number_l1072_107245

theorem find_number (x n : ℤ) (h1 : 5 * x + n = 10 * x - 17) (h2 : x = 4) : n = 3 := by
  sorry

end NUMINAMATH_GPT_find_number_l1072_107245


namespace NUMINAMATH_GPT_radius_increase_is_0_31_l1072_107236

noncomputable def increase_in_radius (initial_radius : ℝ) (odometer_summer : ℝ) (odometer_winter : ℝ) (miles_to_inches : ℝ) : ℝ :=
  let circumference_summer := 2 * Real.pi * initial_radius
  let distance_per_rotation_summer := circumference_summer / miles_to_inches
  let rotations_summer := odometer_summer / distance_per_rotation_summer
  let rotations_winter := odometer_winter / distance_per_rotation_summer
  let distance_winter := rotations_winter * distance_per_rotation_summer
  let new_radius := (distance_winter * miles_to_inches) / (2 * rotations_winter * Real.pi)
  new_radius - initial_radius

theorem radius_increase_is_0_31 : 
    increase_in_radius 16 530 520 63360 = 0.31 := 
by
    sorry

end NUMINAMATH_GPT_radius_increase_is_0_31_l1072_107236


namespace NUMINAMATH_GPT_line_through_points_l1072_107276

theorem line_through_points (a b : ℝ) (h1 : 3 = a * 2 + b) (h2 : 19 = a * 6 + b) :
  a - b = 9 :=
sorry

end NUMINAMATH_GPT_line_through_points_l1072_107276


namespace NUMINAMATH_GPT_part_1_part_2_l1072_107219

theorem part_1 (a : ℕ → ℚ) (S : ℕ → ℚ) (h1 : ∀ n, S (n + 1) = 4 * a n - 2) (h2 : a 1 = 2) (n : ℕ) (hn_pos : 0 < n) : 
  a (n + 1) - 2 * a n = 0 :=
sorry

theorem part_2 (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ) (h1 : ∀ n, S (n + 1) = 4 * a n - 2) (h2 : a 1 = 2) :
  (∀ n, b n = 1 / (a n * a (n + 1))) → ∀ n, S n = (1/6) * (1 - (1/4)^n) :=
sorry

end NUMINAMATH_GPT_part_1_part_2_l1072_107219


namespace NUMINAMATH_GPT_cylinder_unoccupied_volume_l1072_107278

theorem cylinder_unoccupied_volume (r h_cylinder h_cone : ℝ) 
  (h : r = 10 ∧ h_cylinder = 30 ∧ h_cone = 15) :
  (π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π) :=
by
  rcases h with ⟨rfl, rfl, rfl⟩
  simp
  sorry

end NUMINAMATH_GPT_cylinder_unoccupied_volume_l1072_107278


namespace NUMINAMATH_GPT_product_586645_9999_l1072_107257

theorem product_586645_9999 :
  586645 * 9999 = 5865885355 :=
by
  sorry

end NUMINAMATH_GPT_product_586645_9999_l1072_107257


namespace NUMINAMATH_GPT_number_of_multiples_143_l1072_107235

theorem number_of_multiples_143
  (h1 : 143 = 11 * 13)
  (h2 : ∀ i j : ℕ, 10^j - 10^i = 10^i * (10^(j-i) - 1))
  (h3 : ∀ i : ℕ, gcd (10^i) 143 = 1)
  (h4 : ∀ k : ℕ, 143 ∣ 10^k - 1 ↔ k % 6 = 0)
  (h5 : ∀ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ 99)
  : ∃ n : ℕ, n = 784 :=
by
  sorry

end NUMINAMATH_GPT_number_of_multiples_143_l1072_107235


namespace NUMINAMATH_GPT_neg_five_power_zero_simplify_expression_l1072_107261

-- Proof statement for the first question.
theorem neg_five_power_zero : (-5 : ℝ)^0 = 1 := 
by sorry

-- Proof statement for the second question.
theorem simplify_expression (a b : ℝ) : ((-2 * a^2)^2) * (3 * a * b^2) = 12 * a^5 * b^2 := 
by sorry

end NUMINAMATH_GPT_neg_five_power_zero_simplify_expression_l1072_107261


namespace NUMINAMATH_GPT_sum_of_squares_of_real_solutions_l1072_107289

theorem sum_of_squares_of_real_solutions (x : ℝ) (h : x ^ 64 = 16 ^ 16) : 
  (x = 2 ∨ x = -2) → (x ^ 2 + (-x) ^ 2) = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_real_solutions_l1072_107289


namespace NUMINAMATH_GPT_octagon_side_length_l1072_107239

theorem octagon_side_length 
  (num_sides : ℕ) 
  (perimeter : ℝ) 
  (h_sides : num_sides = 8) 
  (h_perimeter : perimeter = 23.6) :
  (perimeter / num_sides) = 2.95 :=
by
  have h_valid_sides : num_sides = 8 := h_sides
  have h_valid_perimeter : perimeter = 23.6 := h_perimeter
  sorry

end NUMINAMATH_GPT_octagon_side_length_l1072_107239


namespace NUMINAMATH_GPT_distance_proof_l1072_107234

theorem distance_proof (d : ℝ) (h1 : d < 6) (h2 : d > 5) (h3 : d > 4) : d ∈ Set.Ioo 5 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_proof_l1072_107234


namespace NUMINAMATH_GPT_cost_percentage_l1072_107294

variable (t b : ℝ)

def C := t * b ^ 4
def R := t * (2 * b) ^ 4

theorem cost_percentage : R = 16 * C := by
  sorry

end NUMINAMATH_GPT_cost_percentage_l1072_107294


namespace NUMINAMATH_GPT_zeros_distance_l1072_107207

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3*x^2 + a

theorem zeros_distance (a x1 x2 : ℝ) 
  (hx1 : f a x1 = 0) (hx2 : f a x2 = 0) (h_order: x1 < x2) : 
  x2 - x1 = 3 := 
sorry

end NUMINAMATH_GPT_zeros_distance_l1072_107207


namespace NUMINAMATH_GPT_polygon_area_correct_l1072_107268

-- Define the coordinates of the vertices
def vertex1 := (2, 1)
def vertex2 := (4, 3)
def vertex3 := (6, 1)
def vertex4 := (4, 6)

-- Define a function to calculate the area using the Shoelace Theorem
noncomputable def shoelace_area (vertices : List (ℕ × ℕ)) : ℚ :=
  let xys := vertices ++ [vertices.head!]
  let sum1 := (xys.zip (xys.tail!)).map (fun ((x1, y1), (x2, y2)) => x1 * y2)
  let sum2 := (xys.zip (xys.tail!)).map (fun ((x1, y1), (x2, y2)) => y1 * x2)
  (sum1.sum - sum2.sum : ℚ) / 2

-- Instantiate the specific vertices
def polygon := [vertex1, vertex2, vertex3, vertex4]

-- The theorem statement
theorem polygon_area_correct : shoelace_area polygon = 6 := by
  sorry

end NUMINAMATH_GPT_polygon_area_correct_l1072_107268


namespace NUMINAMATH_GPT_number_of_pencil_cartons_l1072_107286

theorem number_of_pencil_cartons
  (P E : ℕ) 
  (h1 : P + E = 100)
  (h2 : 6 * P + 3 * E = 360) : 
  P = 20 := 
by
  sorry

end NUMINAMATH_GPT_number_of_pencil_cartons_l1072_107286


namespace NUMINAMATH_GPT_range_of_m_l1072_107229

theorem range_of_m (m : ℝ) (h1 : m + 3 > 0) (h2 : m - 1 < 0) : -3 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1072_107229


namespace NUMINAMATH_GPT_nonneg_real_inequality_l1072_107296

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 := 
by
  sorry

end NUMINAMATH_GPT_nonneg_real_inequality_l1072_107296


namespace NUMINAMATH_GPT_common_difference_unique_l1072_107254

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1 : ℝ, ∀ n : ℕ, a n = a1 + n * d

theorem common_difference_unique {a : ℕ → ℝ}
  (h1 : a 2 = 5)
  (h2 : a 3 + a 5 = 2) :
  ∃ d : ℝ, (∀ n : ℕ, a n = a 1 + (n - 1) * d) ∧ d = -2 :=
sorry

end NUMINAMATH_GPT_common_difference_unique_l1072_107254


namespace NUMINAMATH_GPT_initial_temperature_is_20_l1072_107210

-- Define the initial temperature, final temperature, rate of increase and time
def T_initial (T_final : ℕ) (rate_of_increase : ℕ) (time : ℕ) : ℕ :=
  T_final - rate_of_increase * time

-- Statement: The initial temperature is 20 degrees given the specified conditions.
theorem initial_temperature_is_20 :
  T_initial 100 5 16 = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_temperature_is_20_l1072_107210


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1072_107279

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : a 8 + a 10 = 2) : 
  (17 * (a 1 + a 17) / 2) = 17 := by
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1072_107279


namespace NUMINAMATH_GPT_stripe_area_is_640pi_l1072_107224

noncomputable def cylinder_stripe_area (diameter height stripe_width : ℝ) (revolutions : ℕ) : ℝ :=
  let circumference := Real.pi * diameter
  let length := circumference * (revolutions : ℝ)
  stripe_width * length

theorem stripe_area_is_640pi :
  cylinder_stripe_area 20 100 4 4 = 640 * Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_stripe_area_is_640pi_l1072_107224


namespace NUMINAMATH_GPT_quadractic_roots_value_l1072_107297

theorem quadractic_roots_value (c d : ℝ) (h₁ : 3*c^2 + 9*c - 21 = 0) (h₂ : 3*d^2 + 9*d - 21 = 0) :
  (3*c - 4) * (6*d - 8) = -22 := by
  sorry

end NUMINAMATH_GPT_quadractic_roots_value_l1072_107297


namespace NUMINAMATH_GPT_difference_of_squares_l1072_107280

variable (a b : ℝ)

theorem difference_of_squares (h1 : a + b = 2) (h2 : a - b = 3) : a^2 - b^2 = 6 := 
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1072_107280


namespace NUMINAMATH_GPT_contrapositive_iff_l1072_107287

theorem contrapositive_iff (a b : ℝ) :
  (a^2 - b^2 = 0 → a = b) ↔ (a ≠ b → a^2 - b^2 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_iff_l1072_107287


namespace NUMINAMATH_GPT_tips_collected_l1072_107240

-- Definitions based on conditions
def total_collected : ℕ := 240
def hourly_wage : ℕ := 10
def hours_worked : ℕ := 19

-- Correct answer translated into a proof problem
theorem tips_collected : total_collected - (hours_worked * hourly_wage) = 50 := by
  sorry

end NUMINAMATH_GPT_tips_collected_l1072_107240


namespace NUMINAMATH_GPT_sin_B_of_arithmetic_sequence_angles_l1072_107273

theorem sin_B_of_arithmetic_sequence_angles (A B C : ℝ) (h1 : A + C = 2 * B) (h2 : A + B + C = Real.pi) :
  Real.sin B = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_sin_B_of_arithmetic_sequence_angles_l1072_107273


namespace NUMINAMATH_GPT_regular_hexagon_area_inscribed_in_circle_l1072_107260

theorem regular_hexagon_area_inscribed_in_circle
  (h : Real.pi * r^2 = 100 * Real.pi) :
  6 * (r^2 * Real.sqrt 3 / 4) = 150 * Real.sqrt 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_regular_hexagon_area_inscribed_in_circle_l1072_107260


namespace NUMINAMATH_GPT_power_addition_l1072_107258

variable {R : Type*} [CommRing R]

theorem power_addition (x : R) (m n : ℕ) (h1 : x^m = 6) (h2 : x^n = 2) : x^(m + n) = 12 :=
by
  sorry

end NUMINAMATH_GPT_power_addition_l1072_107258


namespace NUMINAMATH_GPT_three_x_y_z_l1072_107262

variable (x y z : ℝ)

def equation1 : Prop := y + z = 17 - 2 * x
def equation2 : Prop := x + z = -11 - 2 * y
def equation3 : Prop := x + y = 9 - 2 * z

theorem three_x_y_z : equation1 x y z ∧ equation2 x y z ∧ equation3 x y z → 3 * x + 3 * y + 3 * z = 45 / 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_three_x_y_z_l1072_107262


namespace NUMINAMATH_GPT_johns_starting_elevation_l1072_107295

variable (horizontal_distance : ℝ) (final_elevation : ℝ) (initial_elevation : ℝ)
variable (vertical_ascent : ℝ)

-- Given conditions
axiom h1 : (vertical_ascent / horizontal_distance) = (1 / 2)
axiom h2 : final_elevation = 1450
axiom h3 : horizontal_distance = 2700

-- Prove that John's starting elevation is 100 feet
theorem johns_starting_elevation : initial_elevation = 100 := by
  sorry

end NUMINAMATH_GPT_johns_starting_elevation_l1072_107295


namespace NUMINAMATH_GPT_nelly_refrigerator_payment_l1072_107217

theorem nelly_refrigerator_payment (T : ℝ) (p1 p2 p3 : ℝ) (p1_percent p2_percent p3_percent : ℝ)
  (h1 : p1 = 875) (h2 : p2 = 650) (h3 : p3 = 1200)
  (h4 : p1_percent = 0.25) (h5 : p2_percent = 0.15) (h6 : p3_percent = 0.35)
  (total_paid := p1 + p2 + p3)
  (percent_paid := p1_percent + p2_percent + p3_percent)
  (total_cost := total_paid / percent_paid)
  (remaining := total_cost - total_paid) :
  remaining = 908.33 := by
  sorry

end NUMINAMATH_GPT_nelly_refrigerator_payment_l1072_107217


namespace NUMINAMATH_GPT_algebraic_expression_value_l1072_107251

theorem algebraic_expression_value {x : ℝ} (h : x * (x + 2) = 2023) : 2 * (x + 3) * (x - 1) - 2018 = 2022 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1072_107251


namespace NUMINAMATH_GPT_sum_of_areas_of_tangent_circles_l1072_107213

theorem sum_of_areas_of_tangent_circles :
  ∃ r s t : ℝ, r > 0 ∧ s > 0 ∧ t > 0 ∧
    (r + s = 3) ∧
    (r + t = 4) ∧
    (s + t = 5) ∧
    π * (r^2 + s^2 + t^2) = 14 * π :=
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_tangent_circles_l1072_107213


namespace NUMINAMATH_GPT_solve_for_x_l1072_107200

theorem solve_for_x (x : ℝ) (h : 0.20 * x = 0.15 * 1500 - 15) : x = 1050 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1072_107200


namespace NUMINAMATH_GPT_sarah_meals_count_l1072_107205

theorem sarah_meals_count :
  let main_courses := 4
  let sides := 3
  let drinks := 2
  let desserts := 2
  main_courses * sides * drinks * desserts = 48 := 
by
  let main_courses := 4
  let sides := 3
  let drinks := 2
  let desserts := 2
  calc
    4 * 3 * 2 * 2 = 48 := sorry

end NUMINAMATH_GPT_sarah_meals_count_l1072_107205


namespace NUMINAMATH_GPT_original_rectangle_area_l1072_107206

theorem original_rectangle_area
  (A : ℝ)
  (h1 : ∀ (a : ℝ), a = 2 * A)
  (h2 : 4 * A = 32) : 
  A = 8 := 
by
  sorry

end NUMINAMATH_GPT_original_rectangle_area_l1072_107206


namespace NUMINAMATH_GPT_sqrt_of_expression_l1072_107281

theorem sqrt_of_expression (x : ℝ) (h : x = 2) : Real.sqrt (2 * x - 3) = 1 :=
by
  rw [h]
  simp
  sorry

end NUMINAMATH_GPT_sqrt_of_expression_l1072_107281


namespace NUMINAMATH_GPT_tan_pi_div_four_l1072_107284

theorem tan_pi_div_four : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_pi_div_four_l1072_107284


namespace NUMINAMATH_GPT_probability_team_A_champions_l1072_107242

theorem probability_team_A_champions : 
  let p : ℚ := 1 / 2 
  let prob_team_A_win_next := p
  let prob_team_B_win_next_A_win_after := p * p
  prob_team_A_win_next + prob_team_B_win_next_A_win_after = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_team_A_champions_l1072_107242


namespace NUMINAMATH_GPT_field_length_l1072_107209

theorem field_length (w l : ℝ) (A_f A_p : ℝ) 
  (h1 : l = 3 * w)
  (h2 : A_p = 150) 
  (h3 : A_p = 0.4 * A_f)
  (h4 : A_f = l * w) : 
  l = 15 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_field_length_l1072_107209


namespace NUMINAMATH_GPT_xiaojuan_savings_l1072_107274

-- Define the conditions
def spent_on_novel (savings : ℝ) : ℝ := 0.5 * savings
def mother_gave : ℝ := 5
def spent_on_dictionary (amount_given : ℝ) : ℝ := 0.5 * amount_given + 0.4
def remaining_amount : ℝ := 7.2

-- Define the theorem stating the equivalence
theorem xiaojuan_savings : ∃ (savings: ℝ), spent_on_novel savings + mother_gave - spent_on_dictionary mother_gave - remaining_amount = savings / 2 ∧ savings = 20.4 :=
by {
  sorry
}

end NUMINAMATH_GPT_xiaojuan_savings_l1072_107274


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l1072_107250

theorem geometric_sequence_third_term (r : ℕ) (a : ℕ) (h1 : a = 6) (h2 : a * r^3 = 384) : a * r^2 = 96 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l1072_107250


namespace NUMINAMATH_GPT_keith_gave_away_p_l1072_107259

theorem keith_gave_away_p (k_init : Nat) (m_init : Nat) (final_pears : Nat) (k_gave_away : Nat) (total_init: Nat := k_init + m_init) :
  k_init = 47 →
  m_init = 12 →
  final_pears = 13 →
  k_gave_away = total_init - final_pears →
  k_gave_away = 46 :=
by
  -- Insert proof here (skip using sorry)
  sorry

end NUMINAMATH_GPT_keith_gave_away_p_l1072_107259


namespace NUMINAMATH_GPT_expected_number_of_defective_products_l1072_107267

theorem expected_number_of_defective_products 
  (N : ℕ) (D : ℕ) (n : ℕ) (hN : N = 15000) (hD : D = 1000) (hn : n = 150) :
  n * (D / N : ℚ) = 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_expected_number_of_defective_products_l1072_107267


namespace NUMINAMATH_GPT_drum_filled_capacity_l1072_107232

theorem drum_filled_capacity (C : ℝ) (h1 : 0 < C) :
    (4 / 5) * C + (1 / 2) * C = (13 / 10) * C :=
by
  sorry

end NUMINAMATH_GPT_drum_filled_capacity_l1072_107232


namespace NUMINAMATH_GPT_original_painting_width_l1072_107247

theorem original_painting_width {W : ℝ} 
  (orig_height : ℝ) (print_height : ℝ) (print_width : ℝ)
  (h1 : orig_height = 10) 
  (h2 : print_height = 25)
  (h3 : print_width = 37.5) :
  W = 15 :=
  sorry

end NUMINAMATH_GPT_original_painting_width_l1072_107247


namespace NUMINAMATH_GPT_smallest_prime_square_mod_six_l1072_107201

theorem smallest_prime_square_mod_six (p : ℕ) (h_prime : Nat.Prime p) (h_mod : p^2 % 6 = 1) : p = 5 :=
sorry

end NUMINAMATH_GPT_smallest_prime_square_mod_six_l1072_107201


namespace NUMINAMATH_GPT_min_value_theorem_l1072_107204

noncomputable def min_value (x y : ℝ) : ℝ :=
  (x + 2) * (2 * y + 1) / (x * y)

theorem min_value_theorem {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  min_value x y = 19 + 4 * Real.sqrt 15 :=
sorry

end NUMINAMATH_GPT_min_value_theorem_l1072_107204


namespace NUMINAMATH_GPT_problem_1_problem_2_l1072_107277

def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := -|x + 1| + 4

theorem problem_1:
  { x : ℝ // 0 ≤ x ∧ x ≤ 6 } = { x : ℝ // f x ≤ 1 } :=
sorry

theorem problem_2:
  { m : ℝ // m ≤ -3 } = { m : ℝ // ∀ x : ℝ, f x - g x ≥ m + 1 } :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1072_107277


namespace NUMINAMATH_GPT_larger_of_two_numbers_l1072_107228

theorem larger_of_two_numbers (A B : ℕ) (hcf lcm : ℕ) (h1 : hcf = 23)
                              (h2 : lcm = hcf * 14 * 15) 
                              (h3 : lcm = A * B) (h4 : A = 23 * 14) 
                              (h5 : B = 23 * 15) : max A B = 345 :=
    sorry

end NUMINAMATH_GPT_larger_of_two_numbers_l1072_107228


namespace NUMINAMATH_GPT_spellbook_cost_in_gold_l1072_107231

-- Define the constants
def num_spellbooks : ℕ := 5
def cost_potion_kit_in_silver : ℕ := 20
def num_potion_kits : ℕ := 3
def cost_owl_in_gold : ℕ := 28
def conversion_rate : ℕ := 9
def total_payment_in_silver : ℕ := 537

-- Define the problem to prove the cost of each spellbook in gold given the conditions
theorem spellbook_cost_in_gold : (total_payment_in_silver 
  - (cost_potion_kit_in_silver * num_potion_kits + cost_owl_in_gold * conversion_rate)) / num_spellbooks / conversion_rate = 5 := 
  by
  sorry

end NUMINAMATH_GPT_spellbook_cost_in_gold_l1072_107231


namespace NUMINAMATH_GPT_abs_neg_eq_five_l1072_107218

theorem abs_neg_eq_five (a : ℝ) : abs (-a) = 5 ↔ (a = 5 ∨ a = -5) :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_eq_five_l1072_107218


namespace NUMINAMATH_GPT_sum_of_coefficients_l1072_107243

theorem sum_of_coefficients :
  ∃ (A B C D E F G H J K : ℤ),
  (∀ x y : ℤ, 125 * x ^ 8 - 2401 * y ^ 8 = (A * x + B * y) * (C * x ^ 4 + D * x * y + E * y ^ 4) * (F * x + G * y) * (H * x ^ 4 + J * x * y + K * y ^ 4))
  ∧ A + B + C + D + E + F + G + H + J + K = 102 := 
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1072_107243


namespace NUMINAMATH_GPT_phoebe_dog_peanut_butter_l1072_107299

-- Definitions based on the conditions
def servings_per_jar : ℕ := 15
def jars_needed : ℕ := 4
def days : ℕ := 30

-- Problem statement
theorem phoebe_dog_peanut_butter :
  (jars_needed * servings_per_jar) / days / 2 = 1 :=
by sorry

end NUMINAMATH_GPT_phoebe_dog_peanut_butter_l1072_107299


namespace NUMINAMATH_GPT_jerry_money_left_after_shopping_l1072_107283

theorem jerry_money_left_after_shopping :
  let initial_money := 50
  let cost_mustard_oil := 2 * 13
  let cost_penne_pasta := 3 * 4
  let cost_pasta_sauce := 1 * 5
  let total_cost := cost_mustard_oil + cost_penne_pasta + cost_pasta_sauce
  let money_left := initial_money - total_cost
  money_left = 7 := 
sorry

end NUMINAMATH_GPT_jerry_money_left_after_shopping_l1072_107283


namespace NUMINAMATH_GPT_g_at_6_l1072_107263

noncomputable def g : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : g (x + y) = g x + g y
axiom g_at_3 : g 3 = 4

theorem g_at_6 : g 6 = 8 :=
by 
  sorry

end NUMINAMATH_GPT_g_at_6_l1072_107263


namespace NUMINAMATH_GPT_james_total_points_l1072_107298

def f : ℕ := 13
def s : ℕ := 20
def p_f : ℕ := 3
def p_s : ℕ := 2

def total_points : ℕ := (f * p_f) + (s * p_s)

theorem james_total_points : total_points = 79 := 
by
  -- Proof would go here.
  sorry

end NUMINAMATH_GPT_james_total_points_l1072_107298


namespace NUMINAMATH_GPT_number_of_players_l1072_107222

variable (total_socks : ℕ) (socks_per_player : ℕ)

theorem number_of_players (h1 : total_socks = 16) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 8 := by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_number_of_players_l1072_107222


namespace NUMINAMATH_GPT_bridget_poster_board_side_length_l1072_107266

theorem bridget_poster_board_side_length
  (num_cards : ℕ)
  (card_length : ℕ)
  (card_width : ℕ)
  (posterboard_area : ℕ)
  (posterboard_side_length_feet : ℕ)
  (posterboard_side_length_inches : ℕ)
  (cards_area : ℕ) :
  num_cards = 24 ∧
  card_length = 2 ∧
  card_width = 3 ∧
  posterboard_area = posterboard_side_length_inches ^ 2 ∧
  cards_area = num_cards * (card_length * card_width) ∧
  cards_area = posterboard_area ∧
  posterboard_side_length_inches = 12 ∧
  posterboard_side_length_feet = posterboard_side_length_inches / 12 →
  posterboard_side_length_feet = 1 :=
sorry

end NUMINAMATH_GPT_bridget_poster_board_side_length_l1072_107266


namespace NUMINAMATH_GPT_aram_fraction_of_fine_l1072_107203

theorem aram_fraction_of_fine
  (F : ℝ)
  (Joe_payment : ℝ := (1 / 4) * F + 3)
  (Peter_payment : ℝ := (1 / 3) * F - 3)
  (Aram_payment : ℝ := (1 / 2) * F - 4)
  (sum_payments_eq_F : Joe_payment + Peter_payment + Aram_payment = F):
  (Aram_payment / F) = (5 / 12) :=
by
  sorry

end NUMINAMATH_GPT_aram_fraction_of_fine_l1072_107203


namespace NUMINAMATH_GPT_track_length_l1072_107230

theorem track_length (h₁ : ∀ (x : ℕ), (exists y₁ y₂ : ℕ, y₁ = 120 ∧ y₂ = 180 ∧ y₁ + y₂ = x ∧ (y₂ - y₁ = 60) ∧ (y₂ = x - 120))) : 
  ∃ x : ℕ, x = 600 := by
  sorry

end NUMINAMATH_GPT_track_length_l1072_107230


namespace NUMINAMATH_GPT_find_prime_pairs_l1072_107271

theorem find_prime_pairs :
  ∃ p q : ℕ, Prime p ∧ Prime q ∧
    ∃ a b : ℕ, a^2 = p - q ∧ b^2 = pq - q ∧ (p = 3 ∧ q = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_prime_pairs_l1072_107271


namespace NUMINAMATH_GPT_line_through_points_has_sum_m_b_3_l1072_107252

-- Define the structure that two points are given
structure LineThroughPoints (P1 P2 : ℝ × ℝ) : Prop :=
  (slope_intercept_form : ∃ m b, (P1.snd = m * P1.fst + b) ∧ (P2.snd = b)) 

-- Define the particular points
def point1 : ℝ × ℝ := (-2, 0)
def point2 : ℝ × ℝ := (0, 2)

-- The theorem statement
theorem line_through_points_has_sum_m_b_3 
  (h : LineThroughPoints point1 point2) : 
  ∃ m b, (point1.snd = m * point1.fst + b) ∧ (point2.snd = b) ∧ (m + b = 3) :=
by
  sorry

end NUMINAMATH_GPT_line_through_points_has_sum_m_b_3_l1072_107252


namespace NUMINAMATH_GPT_h_is_decreasing_intervals_l1072_107214

noncomputable def f (x : ℝ) := if x >= 1 then x - 2 else 0
noncomputable def g (x : ℝ) := if x <= 2 then -2 * x + 3 else 0

noncomputable def h (x : ℝ) :=
  if x >= 1 ∧ x <= 2 then f x * g x
  else if x >= 1 then f x
  else if x <= 2 then g x
  else 0

theorem h_is_decreasing_intervals :
  (∀ x1 x2 : ℝ, x1 < x2 → x1 < 1 → h x1 > h x2) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → x1 ≥ 7 / 4 → x2 ≤ 2 → h x1 ≥ h x2) :=
by
  sorry

end NUMINAMATH_GPT_h_is_decreasing_intervals_l1072_107214


namespace NUMINAMATH_GPT_max_mogs_l1072_107255

theorem max_mogs : ∃ x y z : ℕ, 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ 3 * x + 4 * y + 8 * z = 100 ∧ z = 10 :=
by
  sorry

end NUMINAMATH_GPT_max_mogs_l1072_107255


namespace NUMINAMATH_GPT_value_of_X_when_S_reaches_15000_l1072_107244

def X : Nat → Nat
| 0       => 5
| (n + 1) => X n + 3

def S : Nat → Nat
| 0       => 0
| (n + 1) => S n + X (n + 1)

theorem value_of_X_when_S_reaches_15000 :
  ∃ n, S n ≥ 15000 ∧ X n = 299 := by
  sorry

end NUMINAMATH_GPT_value_of_X_when_S_reaches_15000_l1072_107244


namespace NUMINAMATH_GPT_determine_positive_intervals_l1072_107212

noncomputable def positive_intervals (x : ℝ) : Prop :=
  (x+1) * (x-1) * (x-3) > 0

theorem determine_positive_intervals :
  ∀ x : ℝ, (positive_intervals x ↔ (x ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) ∨ x ∈ Set.Ioi (3 : ℝ))) :=
by
  sorry

end NUMINAMATH_GPT_determine_positive_intervals_l1072_107212


namespace NUMINAMATH_GPT_term_transition_addition_l1072_107272

theorem term_transition_addition (k : Nat) :
  (2:ℚ) / ((k + 1) * (k + 2)) = ((2:ℚ) / ((k * (k + 1))) - ((2:ℚ) / ((k + 1) * (k + 2)))) := 
sorry

end NUMINAMATH_GPT_term_transition_addition_l1072_107272


namespace NUMINAMATH_GPT_unfenced_side_length_l1072_107221

-- Define the conditions
variables (L W : ℝ)
axiom area_condition : L * W = 480
axiom fence_condition : 2 * W + L = 64

-- Prove the unfenced side of the yard (L) is 40 feet
theorem unfenced_side_length : L = 40 :=
by
  -- Conditions, definitions, and properties go here.
  -- But we leave the proof as a placeholder since the statement is sufficient.
  sorry

end NUMINAMATH_GPT_unfenced_side_length_l1072_107221


namespace NUMINAMATH_GPT_acute_triangle_angle_measure_acute_triangle_side_range_l1072_107238

theorem acute_triangle_angle_measure (A B C a b c : ℝ) (h_acute : A + B + C = π) (h_acute_A : A < π / 2) (h_acute_B : B < π / 2) (h_acute_C : C < π / 2)
  (triangle_relation : (2 * a - c) / (Real.cos (A + B)) = b / (Real.cos (A + C))) : B = π / 3 :=
by
  sorry

theorem acute_triangle_side_range (A B C a b c : ℝ) (h_acute : A + B + C = π) (h_acute_A : A < π / 2) (h_acute_B : B < π / 2) (h_acute_C : C < π / 2)
  (triangle_relation : (2 * a - c) / (Real.cos (A + B)) = b / (Real.cos (A + C))) (hB : B = π / 3) (hb : b = 3) :
  3 * Real.sqrt 3 < a + c ∧ a + c ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_acute_triangle_angle_measure_acute_triangle_side_range_l1072_107238


namespace NUMINAMATH_GPT_possible_sets_C_l1072_107290

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def is_partition (A B C : Set ℕ) : Prop :=
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ A ∪ B ∪ C = M

def conditions (A B C : Set ℕ) : Prop :=
  is_partition A B C ∧ (∃ (a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 : ℕ), 
    A = {a1, a2, a3, a4} ∧
    B = {b1, b2, b3, b4} ∧
    C = {c1, c2, c3, c4} ∧
    c1 < c2 ∧ c2 < c3 ∧ c3 < c4 ∧
    a1 + b1 = c1 ∧ a2 + b2 = c2 ∧ a3 + b3 = c3 ∧ a4 + b4 = c4)

theorem possible_sets_C (A B C : Set ℕ) (h : conditions A B C) :
  C = {8, 9, 10, 12} ∨ C = {7, 9, 11, 12} ∨ C = {6, 10, 11, 12} :=
sorry

end NUMINAMATH_GPT_possible_sets_C_l1072_107290


namespace NUMINAMATH_GPT_point_not_in_second_quadrant_l1072_107220

theorem point_not_in_second_quadrant (m : ℝ) : ¬ (m^2 + m ≤ 0 ∧ m - 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_point_not_in_second_quadrant_l1072_107220


namespace NUMINAMATH_GPT_fifteen_percent_eq_135_l1072_107285

theorem fifteen_percent_eq_135 (x : ℝ) (h : (15 / 100) * x = 135) : x = 900 :=
sorry

end NUMINAMATH_GPT_fifteen_percent_eq_135_l1072_107285
