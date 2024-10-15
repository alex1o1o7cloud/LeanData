import Mathlib

namespace NUMINAMATH_GPT_inequality_solution_l2199_219969

theorem inequality_solution (x : ℝ) (h : x ≠ 5) :
    (15 ≤ x * (x - 2) / (x - 5) ^ 2) ↔ (4.1933 ≤ x ∧ x < 5 ∨ 5 < x ∧ x ≤ 6.3767) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2199_219969


namespace NUMINAMATH_GPT_sum_eq_prod_S1_sum_eq_prod_S2_l2199_219961

def S1 : List ℕ := [1, 1, 1, 1, 1, 1, 2, 8]
def S2 : List ℕ := [1, 1, 1, 1, 1, 2, 2, 3]

def sum_list (l : List ℕ) : ℕ := l.foldr Nat.add 0
def prod_list (l : List ℕ) : ℕ := l.foldr Nat.mul 1

theorem sum_eq_prod_S1 : sum_list S1 = prod_list S1 := 
by
  sorry

theorem sum_eq_prod_S2 : sum_list S2 = prod_list S2 := 
by
  sorry

end NUMINAMATH_GPT_sum_eq_prod_S1_sum_eq_prod_S2_l2199_219961


namespace NUMINAMATH_GPT_arithmetic_seq_property_l2199_219927

theorem arithmetic_seq_property (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_cond : a 1 + 3 * a 8 + a 15 = 120) :
  2 * a 9 - a 10 = 24 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_property_l2199_219927


namespace NUMINAMATH_GPT_solution_set_f_div_x_lt_zero_l2199_219924

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_div_x_lt_zero :
  (∀ x, f (2 + (2 - x)) = f x) ∧
  (∀ x1 x2 : ℝ, x1 < 2 ∧ x2 < 2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0) ∧
  f 4 = 0 →
  { x : ℝ | f x / x < 0 } = { x | x < 0 } ∪ { x | 0 < x ∧ x < 4 } :=
sorry

end NUMINAMATH_GPT_solution_set_f_div_x_lt_zero_l2199_219924


namespace NUMINAMATH_GPT_train_crosses_pole_in_time_l2199_219977

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  length / speed_ms

theorem train_crosses_pole_in_time :
  ∀ (length speed_kmh : ℝ), length = 240 → speed_kmh = 126 →
    time_to_cross_pole length speed_kmh = 6.8571 :=
by
  intros length speed_kmh h_length h_speed
  rw [h_length, h_speed, time_to_cross_pole]
  sorry

end NUMINAMATH_GPT_train_crosses_pole_in_time_l2199_219977


namespace NUMINAMATH_GPT_range_of_a_l2199_219925

noncomputable def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
noncomputable def B (a : ℝ) : Set ℝ := {a - 3, a - 1, a + 1}

theorem range_of_a (a : ℝ) : (A a ∩ B a = {-2}) ↔ (a = -1) :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l2199_219925


namespace NUMINAMATH_GPT_abs_value_identity_l2199_219951

theorem abs_value_identity (a : ℝ) (h : a + |a| = 0) : a - |2 * a| = 3 * a :=
by
  sorry

end NUMINAMATH_GPT_abs_value_identity_l2199_219951


namespace NUMINAMATH_GPT_binary_11011011_to_base4_is_3123_l2199_219979

def binary_to_base4 (b : Nat) : Nat :=
  -- Function to convert binary number to base 4
  -- This will skip implementation details
  sorry

theorem binary_11011011_to_base4_is_3123 :
  binary_to_base4 0b11011011 = 0x3123 := 
sorry

end NUMINAMATH_GPT_binary_11011011_to_base4_is_3123_l2199_219979


namespace NUMINAMATH_GPT_range_of_x_l2199_219985

theorem range_of_x (x : ℝ) (p : x^2 - 2 * x - 3 < 0) (q : 1 / (x - 2) < 0) : -1 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l2199_219985


namespace NUMINAMATH_GPT_total_books_on_shelves_l2199_219947

-- Definitions based on conditions
def num_shelves : Nat := 150
def books_per_shelf : Nat := 15

-- The statement to be proved
theorem total_books_on_shelves : num_shelves * books_per_shelf = 2250 := by
  sorry

end NUMINAMATH_GPT_total_books_on_shelves_l2199_219947


namespace NUMINAMATH_GPT_polynomial_solution_l2199_219960

theorem polynomial_solution (P : Polynomial ℝ) (h_0 : P.eval 0 = 0) (h_func : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) :
  P = Polynomial.X :=
sorry

end NUMINAMATH_GPT_polynomial_solution_l2199_219960


namespace NUMINAMATH_GPT_f_f_f_three_l2199_219908

def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 1

theorem f_f_f_three : f (f (f 3)) = 43 :=
by
  -- Introduction of definitions and further necessary steps here are skipped
  sorry

end NUMINAMATH_GPT_f_f_f_three_l2199_219908


namespace NUMINAMATH_GPT_population_of_seventh_village_l2199_219911

def village_populations : List ℕ := [803, 900, 1100, 1023, 945, 980]

def average_population : ℕ := 1000

theorem population_of_seventh_village 
  (h1 : List.length village_populations = 6)
  (h2 : 1000 * 7 = 7000)
  (h3 : village_populations.sum = 5751) : 
  7000 - village_populations.sum = 1249 := 
by {
  -- h1 ensures there's exactly 6 villages in the list
  -- h2 calculates the total population of 7 villages assuming the average population
  -- h3 calculates the sum of populations in the given list of 6 villages
  -- our goal is to show that 7000 - village_populations.sum = 1249
  -- this will be simplified in the proof
  sorry
}

end NUMINAMATH_GPT_population_of_seventh_village_l2199_219911


namespace NUMINAMATH_GPT_problem_1_problem_2_l2199_219949

noncomputable def O := (0, 0)
noncomputable def A := (1, 2)
noncomputable def B := (-3, 4)

noncomputable def vector_AB := (B.1 - A.1, B.2 - A.2)
noncomputable def magnitude_AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def dot_OA_OB := A.1 * B.1 + A.2 * B.2
noncomputable def magnitude_OA := Real.sqrt (A.1^2 + A.2^2)
noncomputable def magnitude_OB := Real.sqrt (B.1^2 + B.2^2)
noncomputable def cosine_angle := dot_OA_OB / (magnitude_OA * magnitude_OB)

theorem problem_1 : vector_AB = (-4, 2) ∧ magnitude_AB = 2 * Real.sqrt 5 := sorry

theorem problem_2 : cosine_angle = Real.sqrt 5 / 5 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2199_219949


namespace NUMINAMATH_GPT_gcd_max_two_digits_l2199_219933

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end NUMINAMATH_GPT_gcd_max_two_digits_l2199_219933


namespace NUMINAMATH_GPT_smallest_X_l2199_219937

noncomputable def T : ℕ := 1110
noncomputable def X : ℕ := T / 6

theorem smallest_X (hT_digits : (∀ d ∈ T.digits 10, d = 0 ∨ d = 1))
  (hT_positive : T > 0)
  (hT_div_6 : T % 6 = 0) :
  X = 185 := by
  sorry

end NUMINAMATH_GPT_smallest_X_l2199_219937


namespace NUMINAMATH_GPT_projection_matrix_solution_l2199_219935

theorem projection_matrix_solution 
  (a c : ℚ) 
  (P : Matrix (Fin 2) (Fin 2) ℚ := ![![a, 18/45], ![c, 27/45]])
  (hP : P * P = P) :
  (a, c) = (9/25, 12/25) :=
by
  sorry

end NUMINAMATH_GPT_projection_matrix_solution_l2199_219935


namespace NUMINAMATH_GPT_total_legs_of_all_animals_l2199_219955

def num_kangaroos : ℕ := 23
def num_goats : ℕ := 3 * num_kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_legs_of_all_animals : num_kangaroos * legs_of_kangaroo + num_goats * legs_of_goat = 322 :=
by 
  sorry

end NUMINAMATH_GPT_total_legs_of_all_animals_l2199_219955


namespace NUMINAMATH_GPT_cost_price_per_meter_of_cloth_l2199_219905

theorem cost_price_per_meter_of_cloth 
  (total_meters : ℕ)
  (selling_price : ℝ)
  (profit_per_meter : ℝ) 
  (total_profit : ℝ)
  (cp_45 : ℝ)
  (cp_per_meter: ℝ) :
  total_meters = 45 →
  selling_price = 4500 →
  profit_per_meter = 14 →
  total_profit = profit_per_meter * total_meters →
  cp_45 = selling_price - total_profit →
  cp_per_meter = cp_45 / total_meters →
  cp_per_meter = 86 :=
by
  -- your proof here
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_of_cloth_l2199_219905


namespace NUMINAMATH_GPT_candy_bar_sales_ratio_l2199_219992

theorem candy_bar_sales_ratio
    (candy_bar_cost : ℕ := 2)
    (marvin_candy_sold : ℕ := 35)
    (tina_extra_earnings : ℕ := 140)
    (marvin_earnings := marvin_candy_sold * candy_bar_cost)
    (tina_earnings := marvin_earnings + tina_extra_earnings)
    (tina_candy_sold := tina_earnings / candy_bar_cost):
  tina_candy_sold / marvin_candy_sold = 3 :=
by
  sorry

end NUMINAMATH_GPT_candy_bar_sales_ratio_l2199_219992


namespace NUMINAMATH_GPT_students_from_second_grade_l2199_219932

theorem students_from_second_grade (r1 r2 r3 : ℕ) (total_students sample_size : ℕ) (h_ratio: r1 = 3 ∧ r2 = 3 ∧ r3 = 4 ∧ r1 + r2 + r3 = 10) (h_sample_size: sample_size = 50) : 
  (r2 * sample_size / (r1 + r2 + r3)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_students_from_second_grade_l2199_219932


namespace NUMINAMATH_GPT_polygon_E_largest_area_l2199_219994

def unit_square_area : ℕ := 1
def right_triangle_area : ℚ := 1 / 2
def rectangle_area : ℕ := 2

def polygon_A_area : ℚ := 3 * unit_square_area + 2 * right_triangle_area
def polygon_B_area : ℚ := 2 * unit_square_area + 4 * right_triangle_area
def polygon_C_area : ℚ := 4 * unit_square_area + 1 * rectangle_area
def polygon_D_area : ℚ := 3 * rectangle_area
def polygon_E_area : ℚ := 2 * unit_square_area + 2 * right_triangle_area + 2 * rectangle_area

theorem polygon_E_largest_area :
  polygon_E_area = max polygon_A_area (max polygon_B_area (max polygon_C_area (max polygon_D_area polygon_E_area))) := by
  sorry

end NUMINAMATH_GPT_polygon_E_largest_area_l2199_219994


namespace NUMINAMATH_GPT_ratio_of_areas_l2199_219913

theorem ratio_of_areas 
  (lenA : ℕ) (brdA : ℕ) (lenB : ℕ) (brdB : ℕ)
  (h_lenA : lenA = 48) 
  (h_brdA : brdA = 30)
  (h_lenB : lenB = 60) 
  (h_brdB : brdB = 35) :
  (lenA * brdA : ℚ) / (lenB * brdB) = 24 / 35 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l2199_219913


namespace NUMINAMATH_GPT_arithmetic_mean_reciprocal_primes_l2199_219998

theorem arithmetic_mean_reciprocal_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := (1 : ℚ) / p1
  let r2 := (1 : ℚ) / p2
  let r3 := (1 : ℚ) / p3
  let r4 := (1 : ℚ) / p4
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 := by
sorry

end NUMINAMATH_GPT_arithmetic_mean_reciprocal_primes_l2199_219998


namespace NUMINAMATH_GPT_chocolates_total_l2199_219984

theorem chocolates_total (x : ℕ)
  (h1 : x - 12 + x - 18 + x - 20 = 2 * x) :
  x = 50 :=
  sorry

end NUMINAMATH_GPT_chocolates_total_l2199_219984


namespace NUMINAMATH_GPT_slopes_hyperbola_l2199_219983

theorem slopes_hyperbola 
  (x y : ℝ)
  (M : ℝ × ℝ) 
  (t m : ℝ) 
  (h_point_M_on_line: M = (9 / 5, t))
  (h_hyperbola : ∀ t: ℝ, (16 * m^2 - 9) * t^2 + 160 * m * t + 256 = 0)
  (k1 k2 k3 : ℝ)
  (h_k2 : k2 = -5 * t / 16) :
  k1 + k3 = 2 * k2 :=
sorry

end NUMINAMATH_GPT_slopes_hyperbola_l2199_219983


namespace NUMINAMATH_GPT_remaining_rectangle_area_l2199_219989

theorem remaining_rectangle_area (s a b : ℕ) (hs : s = a + b) (total_area_cut : a^2 + b^2 = 40) : s^2 - 40 = 24 :=
by
  sorry

end NUMINAMATH_GPT_remaining_rectangle_area_l2199_219989


namespace NUMINAMATH_GPT_gcd_50420_35313_l2199_219934

theorem gcd_50420_35313 : Int.gcd 50420 35313 = 19 := 
sorry

end NUMINAMATH_GPT_gcd_50420_35313_l2199_219934


namespace NUMINAMATH_GPT_base7_addition_l2199_219965

theorem base7_addition : (26:ℕ) + (245:ℕ) = 304 :=
  sorry

end NUMINAMATH_GPT_base7_addition_l2199_219965


namespace NUMINAMATH_GPT_dividend_in_terms_of_a_l2199_219973

variable (a Q R D : ℕ)

-- Given conditions as hypotheses
def condition1 : Prop := D = 25 * Q
def condition2 : Prop := D = 7 * R
def condition3 : Prop := Q - R = 15
def condition4 : Prop := R = 3 * a

-- Prove that the dividend given these conditions equals the expected expression
theorem dividend_in_terms_of_a (a : ℕ) (Q : ℕ) (R : ℕ) (D : ℕ) :
  condition1 D Q → condition2 D R → condition3 Q R → condition4 R a →
  (D * Q + R) = 225 * a^2 + 1128 * a + 5625 :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_dividend_in_terms_of_a_l2199_219973


namespace NUMINAMATH_GPT_y_run_time_l2199_219900

theorem y_run_time (t : ℕ) (h_avg : (t + 26) / 2 = 42) : t = 58 :=
by
  sorry

end NUMINAMATH_GPT_y_run_time_l2199_219900


namespace NUMINAMATH_GPT_find_m_value_l2199_219901

theorem find_m_value (m : ℝ) 
  (first_term : ℝ := 18) (second_term : ℝ := 6)
  (second_term_2 : ℝ := 6 + m) 
  (S1 : ℝ := first_term / (1 - second_term / first_term))
  (S2 : ℝ := first_term / (1 - second_term_2 / first_term))
  (eq_sum : S2 = 3 * S1) :
  m = 8 := by
  sorry

end NUMINAMATH_GPT_find_m_value_l2199_219901


namespace NUMINAMATH_GPT_find_y_l2199_219926

theorem find_y (a b y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : y > 0)
  (h4 : (2 * a)^(4 * b) = a^b * y^(3 * b)) : y = 2^(4 / 3) * a :=
by
  sorry

end NUMINAMATH_GPT_find_y_l2199_219926


namespace NUMINAMATH_GPT_arthur_first_day_spending_l2199_219902

-- Define the costs of hamburgers and hot dogs.
variable (H D : ℝ)
-- Given conditions
axiom hot_dog_cost : D = 1
axiom second_day_purchase : 2 * H + 3 * D = 7

-- Goal: How much did Arthur spend on the first day?
-- We need to verify that 3H + 4D = 10
theorem arthur_first_day_spending : 3 * H + 4 * D = 10 :=
by
  -- Validating given conditions
  have h1 := hot_dog_cost
  have h2 := second_day_purchase
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_arthur_first_day_spending_l2199_219902


namespace NUMINAMATH_GPT_price_of_sundae_l2199_219914

theorem price_of_sundae (total_ice_cream_bars : ℕ) (total_sundae_price : ℝ)
                        (total_price : ℝ) (price_per_ice_cream_bar : ℝ) (num_ice_cream_bars : ℕ) (num_sundaes : ℕ)
                        (h1 : total_ice_cream_bars = num_ice_cream_bars)
                        (h2 : total_price = 200)
                        (h3 : price_per_ice_cream_bar = 0.40)
                        (h4 : num_ice_cream_bars = 200)
                        (h5 : num_sundaes = 200)
                        (h6 : total_ice_cream_bars * price_per_ice_cream_bar + total_sundae_price = total_price) :
  total_sundae_price / num_sundaes = 0.60 :=
sorry

end NUMINAMATH_GPT_price_of_sundae_l2199_219914


namespace NUMINAMATH_GPT_minimum_period_tan_2x_l2199_219940

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem minimum_period_tan_2x : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_period_tan_2x_l2199_219940


namespace NUMINAMATH_GPT_max_abc_l2199_219991

theorem max_abc : ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a * b + b * c = 518) ∧ 
  (a * b - a * c = 360) ∧ 
  (a * b * c = 1008) := 
by {
  -- Definitions of a, b, c satisfying the given conditions.
  -- Proof of the maximum value will be placed here (not required as per instructions).
  sorry
}

end NUMINAMATH_GPT_max_abc_l2199_219991


namespace NUMINAMATH_GPT_gcd_pow_diff_l2199_219936

theorem gcd_pow_diff (m n: ℤ) (H1: m = 2^2025 - 1) (H2: n = 2^2016 - 1) : Int.gcd m n = 511 := by
  sorry

end NUMINAMATH_GPT_gcd_pow_diff_l2199_219936


namespace NUMINAMATH_GPT_hat_cost_l2199_219929

theorem hat_cost (total_hats blue_hat_cost green_hat_cost green_hats : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_hat_cost = 6)
  (h3 : green_hat_cost = 7)
  (h4 : green_hats = 20) :
  (total_hats - green_hats) * blue_hat_cost + green_hats * green_hat_cost = 530 := 
by sorry

end NUMINAMATH_GPT_hat_cost_l2199_219929


namespace NUMINAMATH_GPT_profit_percentage_of_cp_is_75_percent_of_sp_l2199_219904

/-- If the cost price (CP) is 75% of the selling price (SP), then the profit percentage is 33.33% -/
theorem profit_percentage_of_cp_is_75_percent_of_sp (SP : ℝ) (h : SP > 0) (CP : ℝ) (hCP : CP = 0.75 * SP) :
  (SP - CP) / CP * 100 = 33.33 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_of_cp_is_75_percent_of_sp_l2199_219904


namespace NUMINAMATH_GPT_remainder_div_1234567_256_l2199_219943

theorem remainder_div_1234567_256 : 1234567 % 256 = 45 :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_1234567_256_l2199_219943


namespace NUMINAMATH_GPT_angle_R_in_triangle_l2199_219907

theorem angle_R_in_triangle (P Q R : ℝ) 
  (hP : P = 90)
  (hQ : Q = 4 * R - 10)
  (angle_sum : P + Q + R = 180) 
  : R = 20 := by 
sorry

end NUMINAMATH_GPT_angle_R_in_triangle_l2199_219907


namespace NUMINAMATH_GPT_regular_polygon_interior_angle_ratio_l2199_219942

theorem regular_polygon_interior_angle_ratio (r k : ℕ) (h1 : 180 - 360 / r = (5 : ℚ) / (3 : ℚ) * (180 - 360 / k)) (h2 : r = 2 * k) :
  r = 8 ∧ k = 4 :=
sorry

end NUMINAMATH_GPT_regular_polygon_interior_angle_ratio_l2199_219942


namespace NUMINAMATH_GPT_third_competitor_hot_dogs_l2199_219939

theorem third_competitor_hot_dogs (first_ate : ℕ) (second_multiplier : ℕ) (third_percent_less : ℕ) (second_ate third_ate : ℕ) 
  (H1 : first_ate = 12)
  (H2 : second_multiplier = 2)
  (H3 : third_percent_less = 25)
  (H4 : second_ate = first_ate * second_multiplier)
  (H5 : third_ate = second_ate - (third_percent_less * second_ate / 100)) : 
  third_ate = 18 := 
by 
  sorry

end NUMINAMATH_GPT_third_competitor_hot_dogs_l2199_219939


namespace NUMINAMATH_GPT_parallel_lines_slope_eq_l2199_219976

theorem parallel_lines_slope_eq (k : ℚ) :
  (5 = 3 * k) → k = 5 / 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_eq_l2199_219976


namespace NUMINAMATH_GPT_find_m_l2199_219963

theorem find_m (m : ℤ) (h1 : -180 < m ∧ m < 180) : 
  ((m = 45) ∨ (m = -135)) ↔ (Real.tan (m * Real.pi / 180) = Real.tan (225 * Real.pi / 180)) := 
by 
  sorry

end NUMINAMATH_GPT_find_m_l2199_219963


namespace NUMINAMATH_GPT_intersection_point_proof_l2199_219922

def intersect_point : Prop := 
  ∃ x y : ℚ, (5 * x - 6 * y = 3) ∧ (8 * x + 2 * y = 22) ∧ x = 69 / 29 ∧ y = 43 / 29

theorem intersection_point_proof : intersect_point :=
  sorry

end NUMINAMATH_GPT_intersection_point_proof_l2199_219922


namespace NUMINAMATH_GPT_groups_partition_count_l2199_219995

-- Definitions based on the conditions
def num_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 6
def group3_size : ℕ := 2

-- Given specific names for groups based on problem statement
def Fluffy_group_size : ℕ := group1_size
def Nipper_group_size : ℕ := group2_size

-- The total number of ways to form the groups given the conditions
def total_ways (n k : ℕ) : ℕ := Nat.choose n k

-- Statement of the problem
theorem groups_partition_count :
  total_ways 10 3 * total_ways 7 5 = 2520 := sorry

end NUMINAMATH_GPT_groups_partition_count_l2199_219995


namespace NUMINAMATH_GPT_find_k_range_l2199_219946

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + x else Real.log x / Real.log (1 / 3)

def g (x k : ℝ) : ℝ :=
abs (x - k) + abs (x - 1)

theorem find_k_range (k : ℝ) :
  (∀ x1 x2 : ℝ, f x1 ≤ g x2 k) → (k ≤ 3 / 4 ∨ k ≥ 5 / 4) :=
by
  sorry

end NUMINAMATH_GPT_find_k_range_l2199_219946


namespace NUMINAMATH_GPT_solve_for_x_l2199_219909

theorem solve_for_x (x y z : ℝ) 
  (h1 : x * y + 3 * x + 2 * y = 12) 
  (h2 : y * z + 5 * y + 3 * z = 15) 
  (h3 : x * z + 5 * x + 4 * z = 40) :
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2199_219909


namespace NUMINAMATH_GPT_average_speed_of_trip_l2199_219919

theorem average_speed_of_trip :
  let total_distance := 50 -- in kilometers
  let distance1 := 25 -- in kilometers
  let speed1 := 66 -- in kilometers per hour
  let distance2 := 25 -- in kilometers
  let speed2 := 33 -- in kilometers per hour
  let time1 := distance1 / speed1 -- time taken for the first part
  let time2 := distance2 / speed2 -- time taken for the second part
  let total_time := time1 + time2 -- total time for the trip
  let average_speed := total_distance / total_time -- average speed of the trip
  average_speed = 44 := by
{
  sorry
}

end NUMINAMATH_GPT_average_speed_of_trip_l2199_219919


namespace NUMINAMATH_GPT_parabola_focus_l2199_219954

-- Definitions used in the conditions
def parabola_eq (p : ℝ) (x : ℝ) : ℝ := 2 * p * x^2
def passes_through (p : ℝ) : Prop := parabola_eq p 1 = 4

-- The proof that the coordinates of the focus are (0, 1/16) given the conditions
theorem parabola_focus (p : ℝ) (h : passes_through p) : p = 2 → (0, 1 / 16) = (0, 1 / (4 * p)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l2199_219954


namespace NUMINAMATH_GPT_other_root_of_quadratic_l2199_219945

theorem other_root_of_quadratic (a b : ℝ) (h₀ : a ≠ 0) (h₁ : ∃ x : ℝ, (a * x ^ 2 = b) ∧ (x = 2)) : 
  ∃ m : ℝ, (a * m ^ 2 = b) ∧ (m = -2) := 
sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l2199_219945


namespace NUMINAMATH_GPT_additional_men_joined_l2199_219956

noncomputable def solve_problem := 
  let M := 1000
  let days_initial := 17
  let days_new := 11.333333333333334
  let total_provisions := M * days_initial
  let additional_men := (total_provisions / days_new) - M
  additional_men

theorem additional_men_joined : solve_problem = 500 := by
  sorry

end NUMINAMATH_GPT_additional_men_joined_l2199_219956


namespace NUMINAMATH_GPT_solve_fraction_l2199_219971

theorem solve_fraction (x : ℝ) (h : 2 / (x - 3) = 2) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_l2199_219971


namespace NUMINAMATH_GPT_complement_U_M_l2199_219931

def U : Set ℕ := {x | x > 0 ∧ ∃ y : ℝ, y = Real.sqrt (5 - x)}
def M : Set ℕ := {x ∈ U | 4^x ≤ 16}

theorem complement_U_M : U \ M = {3, 4, 5} := by
  sorry

end NUMINAMATH_GPT_complement_U_M_l2199_219931


namespace NUMINAMATH_GPT_minimum_omega_l2199_219915

theorem minimum_omega (ω : ℝ) (h_pos : ω > 0) :
  (∃ k : ℤ, ω * (3 * π / 4) - ω * (π / 4) = k * π) → ω = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_omega_l2199_219915


namespace NUMINAMATH_GPT_ice_cream_volume_l2199_219993

-- Definitions based on Conditions
def radius_cone : Real := 3 -- radius at the opening of the cone
def height_cone : Real := 12 -- height of the cone

-- The proof statement
theorem ice_cream_volume :
  (1 / 3 * Real.pi * radius_cone^2 * height_cone) + (4 / 3 * Real.pi * radius_cone^3) = 72 * Real.pi := by
  sorry

end NUMINAMATH_GPT_ice_cream_volume_l2199_219993


namespace NUMINAMATH_GPT_tan_alpha_value_l2199_219990

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end NUMINAMATH_GPT_tan_alpha_value_l2199_219990


namespace NUMINAMATH_GPT_solve_for_m_l2199_219975

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 4

theorem solve_for_m (m : ℤ) (h : ∃ x : ℝ, 2^x + x = 4 ∧ m ≤ x ∧ x ≤ m + 1) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_m_l2199_219975


namespace NUMINAMATH_GPT_tan_equality_condition_l2199_219972

open Real

theorem tan_equality_condition (α β : ℝ) :
  (α = β) ↔ (tan α = tan β) :=
sorry

end NUMINAMATH_GPT_tan_equality_condition_l2199_219972


namespace NUMINAMATH_GPT_symmetric_about_z_correct_l2199_219921

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_about_z (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

theorem symmetric_about_z_correct (p : Point3D) :
  p = {x := 3, y := 4, z := 5} → symmetric_about_z p = {x := -3, y := -4, z := 5} :=
by
  sorry

end NUMINAMATH_GPT_symmetric_about_z_correct_l2199_219921


namespace NUMINAMATH_GPT_triangle_QR_length_l2199_219970

/-- Conditions for the triangles PQR and SQR sharing a side QR with given side lengths. -/
structure TriangleSetup where
  (PQ PR SR SQ QR : ℝ)
  (PQ_pos : PQ > 0)
  (PR_pos : PR > 0)
  (SR_pos : SR > 0)
  (SQ_pos : SQ > 0)
  (shared_side_QR : QR = QR)

/-- The problem statement asserting the least possible length of QR. -/
theorem triangle_QR_length (t : TriangleSetup) 
  (h1 : t.PQ = 8)
  (h2 : t.PR = 15)
  (h3 : t.SR = 10)
  (h4 : t.SQ = 25) :
  t.QR = 15 :=
by
  sorry

end NUMINAMATH_GPT_triangle_QR_length_l2199_219970


namespace NUMINAMATH_GPT_graph_must_pass_l2199_219920

variable (f : ℝ → ℝ)
variable (finv : ℝ → ℝ)
variable (h_inv : ∀ y, f (finv y) = y ∧ finv (f y) = y)
variable (h_point : (2 - f 2) = 5)

theorem graph_must_pass : finv (-3) + 3 = 5 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_graph_must_pass_l2199_219920


namespace NUMINAMATH_GPT_smaller_fraction_l2199_219966

variable (x y : ℚ)

theorem smaller_fraction (h1 : x + y = 13 / 14) (h2 : x * y = 1 / 8) : min x y = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_smaller_fraction_l2199_219966


namespace NUMINAMATH_GPT_cupcakes_difference_l2199_219906

theorem cupcakes_difference (h : ℕ) (betty_rate : ℕ) (dora_rate : ℕ) (betty_break : ℕ) 
  (cupcakes_difference : ℕ) 
  (H₁ : betty_rate = 10) 
  (H₂ : dora_rate = 8) 
  (H₃ : betty_break = 2) 
  (H₄ : cupcakes_difference = 10) : 
  8 * h - 10 * (h - 2) = 10 → h = 5 :=
by
  intro H
  sorry

end NUMINAMATH_GPT_cupcakes_difference_l2199_219906


namespace NUMINAMATH_GPT_quadrilateral_area_is_33_l2199_219938

-- Definitions for the points and their coordinates
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 4, y := 0}
def B : Point := {x := 0, y := 12}
def C : Point := {x := 10, y := 0}
def E : Point := {x := 3, y := 3}

-- Define the quadrilateral area computation
noncomputable def areaQuadrilateral (O B E C : Point) : ℝ :=
  let triangle_area (p1 p2 p3 : Point) :=
    abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2
  triangle_area O B E + triangle_area O E C

-- Statement to prove
theorem quadrilateral_area_is_33 : areaQuadrilateral {x := 0, y := 0} B E C = 33 := by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_is_33_l2199_219938


namespace NUMINAMATH_GPT_problem1_l2199_219928

theorem problem1 (x : ℝ) (hx : x < 5/4) : 
  ∃ y : ℝ, y = 4 * x - 2 + 1 / (4 * x - 5) ∧ y ≤ 1 := 
sorry

end NUMINAMATH_GPT_problem1_l2199_219928


namespace NUMINAMATH_GPT_part1_part2_l2199_219948

-- Definition of the branches of the hyperbola
def C1 (P : ℝ × ℝ) : Prop := P.1 * P.2 = 1
def C2 (P : ℝ × ℝ) : Prop := P.1 * P.2 = 1

-- Problem Part 1: Proving that P, Q, and R cannot lie on the same branch
theorem part1 (P Q R : ℝ × ℝ) (hP : C1 P) (hQ : C1 Q) (hR : C1 R) : False := by
  sorry

-- Problem Part 2: Finding the coordinates of Q and R
theorem part2 : 
  ∃ Q R : ℝ × ℝ, C1 Q ∧ C1 R ∧ 
                (Q = (2 - Real.sqrt 3, 1 / (2 - Real.sqrt 3))) ∧ 
                (R = (2 + Real.sqrt 3, 1 / (2 + Real.sqrt 3))) := 
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2199_219948


namespace NUMINAMATH_GPT_find_range_for_two_real_solutions_l2199_219967

noncomputable def f (k x : ℝ) := k * x
noncomputable def g (x : ℝ) := (Real.log x) / x

noncomputable def h (x : ℝ) := (Real.log x) / (x^2)

theorem find_range_for_two_real_solutions :
  (∃ k : ℝ, ∀ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 → (f k x = g x ↔ k ∈ Set.Icc (1 / Real.exp 2) (1 / (2 * Real.exp 1)))) :=
sorry

end NUMINAMATH_GPT_find_range_for_two_real_solutions_l2199_219967


namespace NUMINAMATH_GPT_max_value_of_expression_l2199_219999

theorem max_value_of_expression (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  ∃ m, m = 15 ∧ x^2 + y^2 + 2 * x ≤ m := 
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2199_219999


namespace NUMINAMATH_GPT_find_c_l2199_219912

theorem find_c (x c : ℤ) (h1 : 3 * x + 9 = 0) (h2 : c * x - 5 = -11) : c = 2 := by
  have x_eq : x = -3 := by
    linarith
  subst x_eq
  have c_eq : c = 2 := by
    linarith
  exact c_eq

end NUMINAMATH_GPT_find_c_l2199_219912


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_regular_polygon_l2199_219982

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h : 60 = 360 / n) : (n - 2) * 180 = 720 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_regular_polygon_l2199_219982


namespace NUMINAMATH_GPT_remainder_b100_mod_81_l2199_219988

def b (n : ℕ) := 7^n + 9^n

theorem remainder_b100_mod_81 : (b 100) % 81 = 38 := by
  sorry

end NUMINAMATH_GPT_remainder_b100_mod_81_l2199_219988


namespace NUMINAMATH_GPT_find_relationship_l2199_219964

variables (x y : ℝ)

def AB : ℝ × ℝ := (6, 1)
def BC : ℝ × ℝ := (x, y)
def CD : ℝ × ℝ := (-2, -3)
def DA : ℝ × ℝ := (4 - x, -2 - y)
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_relationship (h_parallel : parallel (x, y) (4 - x, -2 - y)) : x + 2 * y = 0 :=
sorry

end NUMINAMATH_GPT_find_relationship_l2199_219964


namespace NUMINAMATH_GPT_first_term_arithmetic_sum_l2199_219957

theorem first_term_arithmetic_sum 
  (T : ℕ → ℚ) (b : ℚ) (d : ℚ) (h₁ : ∀ n, T n = n * (2 * b + (n - 1) * d) / 2)
  (h₂ : d = 5)
  (h₃ : ∀ n, (T (4 * n)) / (T n) = (16 : ℚ)) : 
  b = 5 / 2 :=
sorry

end NUMINAMATH_GPT_first_term_arithmetic_sum_l2199_219957


namespace NUMINAMATH_GPT_part1_part2_l2199_219941

-- Part 1
theorem part1 (x y : ℝ) : (2 * x - 3 * y) ^ 2 - (y + 3 * x) * (3 * x - y) = -5 * x ^ 2 - 12 * x * y + 10 * y ^ 2 := 
sorry

-- Part 2
theorem part2 : (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) - 2 ^ 16 = -1 := 
sorry

end NUMINAMATH_GPT_part1_part2_l2199_219941


namespace NUMINAMATH_GPT_frank_problems_per_type_l2199_219959

-- Definitions based on the problem conditions
def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def types_of_problems : ℕ := 4

-- The proof statement equivalent to the math problem
theorem frank_problems_per_type : frank_problems / types_of_problems = 30 :=
by 
  -- skipping the proof steps
  sorry

end NUMINAMATH_GPT_frank_problems_per_type_l2199_219959


namespace NUMINAMATH_GPT_weston_academy_geography_players_l2199_219952

theorem weston_academy_geography_players
  (total_players : ℕ)
  (history_players : ℕ)
  (both_players : ℕ) :
  total_players = 18 →
  history_players = 10 →
  both_players = 6 →
  ∃ (geo_players : ℕ), geo_players = 14 := 
by 
  intros h1 h2 h3
  use 18 - (10 - 6) + 6
  sorry

end NUMINAMATH_GPT_weston_academy_geography_players_l2199_219952


namespace NUMINAMATH_GPT_corner_movement_l2199_219950

-- Definition of corner movement problem
def canMoveCornerToBottomRight (m n : ℕ) : Prop :=
  m ≥ 2 ∧ n ≥ 2 ∧ (m % 2 = 1 ∧ n % 2 = 1)

theorem corner_movement (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) :
  (canMoveCornerToBottomRight m n ↔ (m % 2 = 1 ∧ n % 2 = 1)) :=
by
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_corner_movement_l2199_219950


namespace NUMINAMATH_GPT_c_is_perfect_square_l2199_219987

theorem c_is_perfect_square (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : c = a + b / a - 1 / b) : ∃ m : ℕ, c = m * m :=
by
  sorry

end NUMINAMATH_GPT_c_is_perfect_square_l2199_219987


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_shifted_roots_l2199_219918

noncomputable def cubic_poly (x : ℝ) := 45 * x^3 - 75 * x^2 + 33 * x - 2

theorem sum_of_reciprocals_of_shifted_roots (a b c : ℝ) 
  (ha : cubic_poly a = 0) 
  (hb : cubic_poly b = 0) 
  (hc : cubic_poly c = 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_bounds_a : 0 < a ∧ a < 1)
  (h_bounds_b : 0 < b ∧ b < 1)
  (h_bounds_c : 0 < c ∧ c < 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 4 / 3 := 
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_shifted_roots_l2199_219918


namespace NUMINAMATH_GPT_abc_equality_l2199_219916

noncomputable def abc_value (a b c : ℝ) : ℝ := (11 + Real.sqrt 117) / 2

theorem abc_equality (a b c : ℝ) (h1 : a + 1/b = 5) (h2 : b + 1/c = 2) (h3 : (c + 1/a)^2 = 4) :
  a * b * c = abc_value a b c := 
sorry

end NUMINAMATH_GPT_abc_equality_l2199_219916


namespace NUMINAMATH_GPT_division_result_l2199_219997

theorem division_result : 210 / (15 + 12 * 3 - 6) = 210 / 45 :=
by
  sorry

end NUMINAMATH_GPT_division_result_l2199_219997


namespace NUMINAMATH_GPT_cos_2000_eq_neg_inv_sqrt_l2199_219962

theorem cos_2000_eq_neg_inv_sqrt (a : ℝ) (h : Real.tan (20 * Real.pi / 180) = a) :
  Real.cos (2000 * Real.pi / 180) = -1 / Real.sqrt (1 + a^2) :=
sorry

end NUMINAMATH_GPT_cos_2000_eq_neg_inv_sqrt_l2199_219962


namespace NUMINAMATH_GPT_manufacturing_section_degrees_l2199_219986

theorem manufacturing_section_degrees (percentage : ℝ) (total_degrees : ℝ) (h1 : total_degrees = 360) (h2 : percentage = 35) : 
  ((percentage / 100) * total_degrees) = 126 :=
by
  sorry

end NUMINAMATH_GPT_manufacturing_section_degrees_l2199_219986


namespace NUMINAMATH_GPT_P_lt_Q_l2199_219923

theorem P_lt_Q (x : ℝ) (hx : x > 0) (P : ℝ) (Q : ℝ) 
  (hP : P = Real.sqrt (1 + x)) 
  (hQ : Q = 1 + x / 2) : P < Q := 
by
  sorry

end NUMINAMATH_GPT_P_lt_Q_l2199_219923


namespace NUMINAMATH_GPT_value_of_S_l2199_219917

theorem value_of_S (x R S : ℝ) (h1 : x + 1/x = R) (h2 : R = 6) : x^3 + 1/x^3 = 198 :=
by
  sorry

end NUMINAMATH_GPT_value_of_S_l2199_219917


namespace NUMINAMATH_GPT_marcy_multiple_tickets_l2199_219930

theorem marcy_multiple_tickets (m : ℕ) : 
  (26 + (m * 26 - 6) = 150) → m = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_marcy_multiple_tickets_l2199_219930


namespace NUMINAMATH_GPT_subcommittee_count_l2199_219978

theorem subcommittee_count :
  let republicans := 10
  let democrats := 8
  let select_republicans := 4
  let select_democrats := 3
  let num_ways_republicans := Nat.choose republicans select_republicans
  let num_ways_democrats := Nat.choose democrats select_democrats
  let num_ways := num_ways_republicans * num_ways_democrats
  num_ways = 11760 :=
by
  sorry

end NUMINAMATH_GPT_subcommittee_count_l2199_219978


namespace NUMINAMATH_GPT_sqrt_sum_eq_ten_l2199_219980

theorem sqrt_sum_eq_ten :
  Real.sqrt ((5 - 4*Real.sqrt 2)^2) + Real.sqrt ((5 + 4*Real.sqrt 2)^2) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_sum_eq_ten_l2199_219980


namespace NUMINAMATH_GPT_find_optimal_addition_l2199_219944

theorem find_optimal_addition (m : ℝ) : 
  (1000 + (m - 1000) * 0.618 = 1618 ∨ 1000 + (m - 1000) * 0.618 = 2618) →
  (m = 2000 ∨ m = 2618) :=
sorry

end NUMINAMATH_GPT_find_optimal_addition_l2199_219944


namespace NUMINAMATH_GPT_number_of_sheep_l2199_219981

def ratio_sheep_horses (S H : ℕ) : Prop := S / H = 3 / 7
def horse_food_per_day := 230 -- ounces
def total_food_per_day := 12880 -- ounces

theorem number_of_sheep (S H : ℕ) 
  (h1 : ratio_sheep_horses S H) 
  (h2 : H * horse_food_per_day = total_food_per_day) 
  : S = 24 :=
sorry

end NUMINAMATH_GPT_number_of_sheep_l2199_219981


namespace NUMINAMATH_GPT_f_prime_neg1_l2199_219910

def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

def f' (a b c x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

theorem f_prime_neg1 (a b c : ℝ) (h : f' a b c 1 = 2) : f' a b c (-1) = -2 :=
by
  sorry

end NUMINAMATH_GPT_f_prime_neg1_l2199_219910


namespace NUMINAMATH_GPT_oliver_money_left_l2199_219958

-- Definitions based on the given conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def frisbee_cost : ℕ := 4
def puzzle_cost : ℕ := 3
def birthday_gift : ℕ := 8

-- The proof problem statement
theorem oliver_money_left : 
  initial_amount + allowance_saved - (frisbee_cost + puzzle_cost) + birthday_gift = 15 := 
by
  sorry

end NUMINAMATH_GPT_oliver_money_left_l2199_219958


namespace NUMINAMATH_GPT_largest_multiple_of_7_less_than_100_l2199_219968

theorem largest_multiple_of_7_less_than_100 : ∃ n : ℕ, 7 * n < 100 ∧ ∀ m : ℕ, 7 * m < 100 → 7 * m ≤ 7 * n := by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_7_less_than_100_l2199_219968


namespace NUMINAMATH_GPT_dolls_given_to_girls_correct_l2199_219903

-- Define the total number of toys given
def total_toys_given : ℕ := 403

-- Define the number of toy cars given to boys
def toy_cars_given_to_boys : ℕ := 134

-- Define the number of dolls given to girls
def dolls_given_to_girls : ℕ := total_toys_given - toy_cars_given_to_boys

-- State the theorem to prove the number of dolls given to girls
theorem dolls_given_to_girls_correct : dolls_given_to_girls = 269 := by
  sorry

end NUMINAMATH_GPT_dolls_given_to_girls_correct_l2199_219903


namespace NUMINAMATH_GPT_problem_l2199_219974

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + Real.pi / 4)

theorem problem
  (h1 : f (Real.pi / 8) = 2)
  (h2 : f (5 * Real.pi / 8) = -2) :
  (∀ x : ℝ, f x = 1 ↔ 
    (∃ k : ℤ, x = -Real.pi / 24 + k * Real.pi) ∨
    (∃ k : ℤ, x = 7 * Real.pi / 24 + k * Real.pi)) :=
by
  sorry

end NUMINAMATH_GPT_problem_l2199_219974


namespace NUMINAMATH_GPT_min_square_distance_l2199_219996

theorem min_square_distance (x y z w : ℝ) (h1 : x * y = 4) (h2 : z^2 + 4 * w^2 = 4) : (x - z)^2 + (y - w)^2 ≥ 1.6 :=
sorry

end NUMINAMATH_GPT_min_square_distance_l2199_219996


namespace NUMINAMATH_GPT_general_term_arithmetic_sequence_sum_first_n_terms_l2199_219953

noncomputable def a_n (n : ℕ) : ℤ :=
  3 * n - 1

def b_n (n : ℕ) (b : ℕ → ℚ) : Prop :=
  (b 1 = 1) ∧ (b 2 = 1 / 3) ∧ ∀ n : ℕ, a_n n * b (n + 1) = n * b n

def sum_b_n (n : ℕ) (b : ℕ → ℚ) : ℚ :=
  (3 / 2) - (1 / (2 * (3 ^ (n - 1))))

theorem general_term_arithmetic_sequence (n : ℕ) :
  a_n n = 3 * n - 1 := by sorry

theorem sum_first_n_terms (n : ℕ) (b : ℕ → ℚ) (h : b_n n b) :
  sum_b_n n b = (3 / 2) - (1 / (2 * (3 ^ (n - 1)))) := by sorry

end NUMINAMATH_GPT_general_term_arithmetic_sequence_sum_first_n_terms_l2199_219953
