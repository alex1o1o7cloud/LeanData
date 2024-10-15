import Mathlib

namespace NUMINAMATH_GPT_find_f_at_3_l709_70968

theorem find_f_at_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = x ^ 2 - 2 * x) : f 3 = -1 :=
by {
  -- Proof would go here.
  sorry
}

end NUMINAMATH_GPT_find_f_at_3_l709_70968


namespace NUMINAMATH_GPT_integer_solutions_to_quadratic_inequality_l709_70908

theorem integer_solutions_to_quadratic_inequality :
  {x : ℤ | (x^2 + 6 * x + 8) * (x^2 - 4 * x + 3) < 0} = {-3, 2} :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_to_quadratic_inequality_l709_70908


namespace NUMINAMATH_GPT_return_journey_time_l709_70919

-- Define the conditions
def walking_speed : ℕ := 100 -- meters per minute
def walking_time : ℕ := 36 -- minutes
def running_speed : ℕ := 3 -- meters per second

-- Define derived values from conditions
def distance_walked : ℕ := walking_speed * walking_time -- meters
def running_speed_minute : ℕ := running_speed * 60 -- meters per minute

-- Statement of the problem
theorem return_journey_time :
  (distance_walked / running_speed_minute) = 20 := by
  sorry

end NUMINAMATH_GPT_return_journey_time_l709_70919


namespace NUMINAMATH_GPT_fraction_upgraded_sensors_l709_70972

theorem fraction_upgraded_sensors (N U : ℕ) (h1 : N = U / 3) (h2 : U = 3 * N) : 
  (U : ℚ) / (24 * N + U) = 1 / 9 := by
  sorry

end NUMINAMATH_GPT_fraction_upgraded_sensors_l709_70972


namespace NUMINAMATH_GPT_geometric_sequence_n_value_l709_70930

theorem geometric_sequence_n_value (a : ℕ → ℝ) (q : ℝ) (n : ℕ) 
  (h1 : a 3 + a 6 = 36) 
  (h2 : a 4 + a 7 = 18)
  (h3 : a n = 1/2) :
  n = 9 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_n_value_l709_70930


namespace NUMINAMATH_GPT_stone_length_is_correct_l709_70903

variable (length_m width_m : ℕ)
variable (num_stones : ℕ)
variable (width_stone dm : ℕ)

def length_of_each_stone (length_m : ℕ) (width_m : ℕ) (num_stones : ℕ) (width_stone : ℕ) : ℕ :=
  let length_dm := length_m * 10
  let width_dm := width_m * 10
  let area_hall := length_dm * width_dm
  let area_stone := width_stone * 5
  (area_hall / num_stones) / width_stone

theorem stone_length_is_correct :
  length_of_each_stone 36 15 5400 5 = 2 := by
  sorry

end NUMINAMATH_GPT_stone_length_is_correct_l709_70903


namespace NUMINAMATH_GPT_arithmetic_sequence_30th_term_l709_70925

-- Definitions
def a₁ : ℤ := 8
def d : ℤ := -3
def n : ℕ := 30

-- The statement to be proved
theorem arithmetic_sequence_30th_term :
  a₁ + (n - 1) * d = -79 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_30th_term_l709_70925


namespace NUMINAMATH_GPT_weight_of_new_student_l709_70974

theorem weight_of_new_student (W x y z : ℝ) (h : (W - x - y + z = W - 40)) : z = 40 - (x + y) :=
by
  sorry

end NUMINAMATH_GPT_weight_of_new_student_l709_70974


namespace NUMINAMATH_GPT_solve_inequalities_l709_70935

theorem solve_inequalities (x : ℝ) : (x + 1 > 0 ∧ x - 3 < 2) ↔ (-1 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_GPT_solve_inequalities_l709_70935


namespace NUMINAMATH_GPT_MischiefConventionHandshakes_l709_70901

theorem MischiefConventionHandshakes :
  let gremlins := 30
  let imps := 25
  let reconciled_imps := 10
  let non_reconciled_imps := imps - reconciled_imps
  let handshakes_among_gremlins := (gremlins * (gremlins - 1)) / 2
  let handshakes_among_imps := (reconciled_imps * (reconciled_imps - 1)) / 2
  let handshakes_between_gremlins_and_imps := gremlins * imps
  handshakes_among_gremlins + handshakes_among_imps + handshakes_between_gremlins_and_imps = 1230 := by
  sorry

end NUMINAMATH_GPT_MischiefConventionHandshakes_l709_70901


namespace NUMINAMATH_GPT_parcel_cost_guangzhou_shanghai_l709_70993

theorem parcel_cost_guangzhou_shanghai (x y : ℕ) :
  (x + 2 * y = 10 ∧ x + 3 * (y + 3) + 2 = 23) →
  (x = 6 ∧ y = 2 ∧ (6 + 4 * 2 = 14)) := by
  sorry

end NUMINAMATH_GPT_parcel_cost_guangzhou_shanghai_l709_70993


namespace NUMINAMATH_GPT_max_satiated_pikes_l709_70962

-- Define the total number of pikes
def total_pikes : ℕ := 30

-- Define the condition for satiation
def satiated_condition (eats : ℕ) : Prop := eats ≥ 3

-- Define the number of pikes eaten by each satiated pike
def eaten_by_satiated_pike : ℕ := 3

-- Define the theorem to find the maximum number of satiated pikes
theorem max_satiated_pikes (s : ℕ) : 
  (s * eaten_by_satiated_pike < total_pikes) → s ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_max_satiated_pikes_l709_70962


namespace NUMINAMATH_GPT_distinct_integers_sum_441_l709_70990

-- Define the variables and conditions
variables (a b c d : ℕ)

-- State the conditions: a, b, c, d are distinct positive integers and their product is 441
def distinct_positive_integers (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
def positive_integers (a b c d : ℕ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

-- Define the main statement to be proved
theorem distinct_integers_sum_441 (a b c d : ℕ) (h_distinct : distinct_positive_integers a b c d) 
(h_positive : positive_integers a b c d) 
(h_product : a * b * c * d = 441) : a + b + c + d = 32 :=
by
  sorry

end NUMINAMATH_GPT_distinct_integers_sum_441_l709_70990


namespace NUMINAMATH_GPT_cube_greater_l709_70958

theorem cube_greater (a b : ℝ) (h : a > b) : a^3 > b^3 := 
sorry

end NUMINAMATH_GPT_cube_greater_l709_70958


namespace NUMINAMATH_GPT_find_side2_l709_70973

-- Define the given conditions
def perimeter : ℕ := 160
def side1 : ℕ := 40
def side3 : ℕ := 70

-- Define the second side as a variable
def side2 : ℕ := perimeter - side1 - side3

-- State the theorem to be proven
theorem find_side2 : side2 = 50 := by
  -- We skip the proof here with sorry
  sorry

end NUMINAMATH_GPT_find_side2_l709_70973


namespace NUMINAMATH_GPT_standard_circle_equation_l709_70952

theorem standard_circle_equation (x y : ℝ) :
  ∃ (h k r : ℝ), h = 2 ∧ k = -1 ∧ r = 3 ∧ (x - h)^2 + (y - k + 1)^2 = r^2 :=
by
  use 2, -1, 3
  simp
  sorry

end NUMINAMATH_GPT_standard_circle_equation_l709_70952


namespace NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l709_70929

theorem max_sum_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a3 : a 3 = 7)
  (h_a1_a7 : a 1 + a 7 = 10)
  (h_S : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 1 - a 0))) / 2) :
  ∃ n, S n = S 6 ∧ (∀ m, S m ≤ S 6) :=
sorry

end NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l709_70929


namespace NUMINAMATH_GPT_probability_of_triangle_segments_from_15gon_l709_70991

/-- A proof problem that calculates the probability that three randomly selected segments 
    from a regular 15-gon inscribed in a circle form a triangle with positive area. -/
theorem probability_of_triangle_segments_from_15gon : 
  let n := 15
  let total_segments := (n * (n - 1)) / 2 
  let total_combinations := total_segments * (total_segments - 1) * (total_segments - 2) / 6 
  let valid_probability := 943 / 1365
  valid_probability = (total_combinations - count_violating_combinations) / total_combinations :=
sorry

end NUMINAMATH_GPT_probability_of_triangle_segments_from_15gon_l709_70991


namespace NUMINAMATH_GPT_total_tiles_count_l709_70969

theorem total_tiles_count (n total_tiles: ℕ) 
  (h1: total_tiles - n^2 = 36) 
  (h2: total_tiles - (n + 1)^2 = 3) : total_tiles = 292 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_tiles_count_l709_70969


namespace NUMINAMATH_GPT_grade_point_average_l709_70943

theorem grade_point_average (X : ℝ) (GPA_rest : ℝ) (GPA_whole : ℝ) 
  (h1 : GPA_rest = 66) (h2 : GPA_whole = 64) 
  (h3 : (1 / 3) * X + (2 / 3) * GPA_rest = GPA_whole) : X = 60 :=
sorry

end NUMINAMATH_GPT_grade_point_average_l709_70943


namespace NUMINAMATH_GPT_traders_fabric_sales_l709_70913

theorem traders_fabric_sales (x y : ℕ) : 
  x + y = 85 ∧
  x = y + 5 ∧
  60 = x * (60 / y) ∧
  30 = y * (30 / x) →
  (x, y) = (25, 20) :=
by {
  sorry
}

end NUMINAMATH_GPT_traders_fabric_sales_l709_70913


namespace NUMINAMATH_GPT_math_proof_l709_70981

def exponentiation_result := -1 ^ 4
def negative_exponentiation_result := (-2) ^ 3
def absolute_value_result := abs (-3 - 1)
def division_result := 16 / negative_exponentiation_result
def multiplication_result := division_result * absolute_value_result
def final_result := exponentiation_result + multiplication_result

theorem math_proof : final_result = -9 := by
  -- To be proved
  sorry

end NUMINAMATH_GPT_math_proof_l709_70981


namespace NUMINAMATH_GPT_fraction_product_l709_70907

theorem fraction_product : (1 / 4 : ℚ) * (2 / 5) * (3 / 6) = (1 / 20) := by
  sorry

end NUMINAMATH_GPT_fraction_product_l709_70907


namespace NUMINAMATH_GPT_polynomial_identity_l709_70933

theorem polynomial_identity : 
  ∀ x : ℝ, 
    5 * x^3 - 32 * x^2 + 75 * x - 71 = 
    5 * (x - 2)^3 + (-2) * (x - 2)^2 + 7 * (x - 2) - 9 :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_identity_l709_70933


namespace NUMINAMATH_GPT_root_of_quadratic_l709_70953

theorem root_of_quadratic (m : ℝ) (h : 3*1^2 - 1 + m = 0) : m = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_root_of_quadratic_l709_70953


namespace NUMINAMATH_GPT_first_train_takes_4_hours_less_l709_70985

-- Definitions of conditions
def distance: ℝ := 425.80645161290323
def speed_first_train: ℝ := 75
def speed_second_train: ℝ := 44

-- Lean statement to prove the correct answer
theorem first_train_takes_4_hours_less:
  (distance / speed_second_train) - (distance / speed_first_train) = 4 := 
  by
    -- Skip the actual proof
    sorry

end NUMINAMATH_GPT_first_train_takes_4_hours_less_l709_70985


namespace NUMINAMATH_GPT_fraction_addition_l709_70924

def fraction_sum : ℚ := (2 : ℚ)/5 + (3 : ℚ)/8

theorem fraction_addition : fraction_sum = 31/40 := by
  sorry

end NUMINAMATH_GPT_fraction_addition_l709_70924


namespace NUMINAMATH_GPT_cos_alpha_in_fourth_quadrant_l709_70944

theorem cos_alpha_in_fourth_quadrant (α : ℝ) (P : ℝ × ℝ) (h_angle_quadrant : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi)
(h_point : P = (Real.sqrt 5, 2)) (h_sin : Real.sin α = (Real.sqrt 2 / 4) * 2) :
  Real.cos α = Real.sqrt 10 / 4 :=
sorry

end NUMINAMATH_GPT_cos_alpha_in_fourth_quadrant_l709_70944


namespace NUMINAMATH_GPT_value_of_x_l709_70910

theorem value_of_x (x y : ℝ) (h : x / (x - 1) = (y^2 + 2 * y + 3) / (y^2 + 2 * y + 2))  : 
  x = y^2 + 2 * y + 3 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_x_l709_70910


namespace NUMINAMATH_GPT_gcd_of_set_B_is_five_l709_70931

def is_in_set_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

theorem gcd_of_set_B_is_five : ∃ d, (∀ n, is_in_set_B n → d ∣ n) ∧ d = 5 :=
by 
  sorry

end NUMINAMATH_GPT_gcd_of_set_B_is_five_l709_70931


namespace NUMINAMATH_GPT_pieces_per_box_l709_70977

theorem pieces_per_box (total_pieces : ℕ) (boxes : ℕ) (h_total : total_pieces = 3000) (h_boxes : boxes = 6) :
  total_pieces / boxes = 500 := by
  sorry

end NUMINAMATH_GPT_pieces_per_box_l709_70977


namespace NUMINAMATH_GPT_problem_statement_l709_70938

theorem problem_statement (m : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * x + m ≤ 0)) → m > 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l709_70938


namespace NUMINAMATH_GPT_kw_price_approx_4266_percent_l709_70911

noncomputable def kw_price_percentage (A B C D E : ℝ) (hA : KW = 1.5 * A) (hB : KW = 2 * B) (hC : KW = 2.5 * C) (hD : KW = 2.25 * D) (hE : KW = 3 * E) : ℝ :=
  let total_assets := A + B + C + D + E
  let price_kw := 1.5 * A
  (price_kw / total_assets) * 100

theorem kw_price_approx_4266_percent (A B C D E KW : ℝ)
  (hA : KW = 1.5 * A) (hB : KW = 2 * B) (hC : KW = 2.5 * C) (hD : KW = 2.25 * D) (hE : KW = 3 * E)
  (hB_from_A : B = 0.75 * A) (hC_from_A : C = 0.6 * A) (hD_from_A : D = 0.6667 * A) (hE_from_A : E = 0.5 * A) :
  abs ((kw_price_percentage A B C D E hA hB hC hD hE) - 42.66) < 1 :=
by sorry

end NUMINAMATH_GPT_kw_price_approx_4266_percent_l709_70911


namespace NUMINAMATH_GPT_max_value_of_function_neg_x_l709_70964

theorem max_value_of_function_neg_x (x : ℝ) (h : x < 0) : 
  ∃ y, (y = 2 * x + 2 / x) ∧ y ≤ -4 := sorry

end NUMINAMATH_GPT_max_value_of_function_neg_x_l709_70964


namespace NUMINAMATH_GPT_unique_positive_b_for_discriminant_zero_l709_70922

theorem unique_positive_b_for_discriminant_zero (c : ℝ) : 
  (∃! b : ℝ, b > 0 ∧ (b^2 + 1/b^2)^2 - 4 * c = 0) → c = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_b_for_discriminant_zero_l709_70922


namespace NUMINAMATH_GPT_opposite_terminal_sides_l709_70914

theorem opposite_terminal_sides (α β : ℝ) (k : ℤ) (h : ∃ k : ℤ, α = β + 180 + k * 360) :
  α = β + 180 + k * 360 :=
by sorry

end NUMINAMATH_GPT_opposite_terminal_sides_l709_70914


namespace NUMINAMATH_GPT_carrie_savings_l709_70918

-- Define the original prices and discount rates
def deltaPrice : ℝ := 850
def deltaDiscount : ℝ := 0.20
def unitedPrice : ℝ := 1100
def unitedDiscount : ℝ := 0.30

-- Calculate discounted prices
def deltaDiscountAmount : ℝ := deltaPrice * deltaDiscount
def unitedDiscountAmount : ℝ := unitedPrice * unitedDiscount

def deltaDiscountedPrice : ℝ := deltaPrice - deltaDiscountAmount
def unitedDiscountedPrice : ℝ := unitedPrice - unitedDiscountAmount

-- Define the savings by choosing the cheaper flight
def savingsByChoosingCheaperFlight : ℝ := unitedDiscountedPrice - deltaDiscountedPrice

-- The theorem stating the amount saved
theorem carrie_savings : savingsByChoosingCheaperFlight = 90 :=
by
  sorry

end NUMINAMATH_GPT_carrie_savings_l709_70918


namespace NUMINAMATH_GPT_sum_of_parts_l709_70956

variable (x y : ℤ)
variable (h1 : x + y = 60)
variable (h2 : y = 45)

theorem sum_of_parts : 10 * x + 22 * y = 1140 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_parts_l709_70956


namespace NUMINAMATH_GPT_simplify_expression_and_evaluate_evaluate_expression_at_one_l709_70951

theorem simplify_expression_and_evaluate (x : ℝ)
  (h1 : x ≠ -2) (h2 : x ≠ 2) (h3 : x ≠ 3) :
  ( ((x^2 - 2*x) / (x^2 - 4*x + 4) - 3 / (x - 2)) / ((x - 3) / (x^2 - 4)) ) = x + 2 :=
by {
  sorry
}

theorem evaluate_expression_at_one :
  ( ((1^2 - 2*1) / (1^2 - 4*1 + 4) - 3 / (1 - 2)) / ((1 - 3) / (1^2 - 4)) ) = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_and_evaluate_evaluate_expression_at_one_l709_70951


namespace NUMINAMATH_GPT_f_zero_one_and_odd_l709_70975

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (a b : ℝ) : f (a * b) = a * f b + b * f a
axiom f_not_zero : ∃ x : ℝ, f x ≠ 0

theorem f_zero_one_and_odd :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end NUMINAMATH_GPT_f_zero_one_and_odd_l709_70975


namespace NUMINAMATH_GPT_prime_divides_sum_diff_l709_70939

theorem prime_divides_sum_diff
  (a b c p : ℕ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hp : p.Prime) 
  (h1 : p ∣ (100 * a + 10 * b + c)) 
  (h2 : p ∣ (100 * c + 10 * b + a)) 
  : p ∣ (a + b + c) ∨ p ∣ (a - b + c) ∨ p ∣ (a - c) :=
by
  sorry

end NUMINAMATH_GPT_prime_divides_sum_diff_l709_70939


namespace NUMINAMATH_GPT_smallest_two_digit_product_12_l709_70994

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end NUMINAMATH_GPT_smallest_two_digit_product_12_l709_70994


namespace NUMINAMATH_GPT_expression_value_l709_70980

theorem expression_value (x : ℤ) (hx : x = 1729) : abs (abs (abs x + x) + abs x) + x = 6916 :=
by
  rw [hx]
  sorry

end NUMINAMATH_GPT_expression_value_l709_70980


namespace NUMINAMATH_GPT_price_before_tax_l709_70960

theorem price_before_tax (P : ℝ) (h : 1.15 * P = 1955) : P = 1700 :=
by sorry

end NUMINAMATH_GPT_price_before_tax_l709_70960


namespace NUMINAMATH_GPT_calculate_value_expression_l709_70906

theorem calculate_value_expression :
  3000 * (3000 ^ 3000 + 3000 ^ 2999) = 3001 * 3000 ^ 3000 := 
by
  sorry

end NUMINAMATH_GPT_calculate_value_expression_l709_70906


namespace NUMINAMATH_GPT_smallest_arithmetic_mean_divisible_product_l709_70921

theorem smallest_arithmetic_mean_divisible_product :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → Nat.gcd ((n + i) * 1111) 1111 = 1111) ∧ (n + 4 = 97) := 
  sorry

end NUMINAMATH_GPT_smallest_arithmetic_mean_divisible_product_l709_70921


namespace NUMINAMATH_GPT_solve_for_k_l709_70900

theorem solve_for_k (k : ℚ) : 
  (∃ x : ℚ, (3 * x - 6 = 0) ∧ (2 * x - 5 * k = 11)) → k = -7/5 :=
by 
  intro h
  cases' h with x hx
  have hx1 : x = 2 := by linarith
  have hx2 : x = 11 / 2 + 5 / 2 * k := by linarith
  linarith

end NUMINAMATH_GPT_solve_for_k_l709_70900


namespace NUMINAMATH_GPT_real_roots_of_quadratic_l709_70971

theorem real_roots_of_quadratic (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 4 * x + 3 = 0) ↔ m ≤ 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_real_roots_of_quadratic_l709_70971


namespace NUMINAMATH_GPT_cos_double_angle_l709_70992

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l709_70992


namespace NUMINAMATH_GPT_antonella_toonies_l709_70970

theorem antonella_toonies (L T : ℕ) (h1 : L + T = 10) (h2 : L + 2 * T = 14) : T = 4 :=
by
  sorry

end NUMINAMATH_GPT_antonella_toonies_l709_70970


namespace NUMINAMATH_GPT_rectangle_length_l709_70928

theorem rectangle_length (P L W : ℕ) (h1 : P = 48) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : L = 16 := by
  sorry

end NUMINAMATH_GPT_rectangle_length_l709_70928


namespace NUMINAMATH_GPT_baseball_tickets_l709_70984

theorem baseball_tickets (B : ℕ) 
  (h1 : 25 = 2 * B + 6) : B = 9 :=
sorry

end NUMINAMATH_GPT_baseball_tickets_l709_70984


namespace NUMINAMATH_GPT_spinsters_count_l709_70940

theorem spinsters_count (S C : ℕ) (h1 : S / C = 2 / 9) (h2 : C = S + 42) : S = 12 := by
  sorry

end NUMINAMATH_GPT_spinsters_count_l709_70940


namespace NUMINAMATH_GPT_event_A_muffins_correct_event_B_muffins_correct_event_C_muffins_correct_l709_70986

-- Event A
def total_muffins_needed_A := 200
def arthur_muffins_A := 35
def beatrice_muffins_A := 48
def charles_muffins_A := 29
def total_muffins_baked_A := arthur_muffins_A + beatrice_muffins_A + charles_muffins_A
def additional_muffins_needed_A := total_muffins_needed_A - total_muffins_baked_A

-- Event B
def total_muffins_needed_B := 150
def arthur_muffins_B := 20
def beatrice_muffins_B := 35
def charles_muffins_B := 25
def total_muffins_baked_B := arthur_muffins_B + beatrice_muffins_B + charles_muffins_B
def additional_muffins_needed_B := total_muffins_needed_B - total_muffins_baked_B

-- Event C
def total_muffins_needed_C := 250
def arthur_muffins_C := 45
def beatrice_muffins_C := 60
def charles_muffins_C := 30
def total_muffins_baked_C := arthur_muffins_C + beatrice_muffins_C + charles_muffins_C
def additional_muffins_needed_C := total_muffins_needed_C - total_muffins_baked_C

-- Proof Statements
theorem event_A_muffins_correct : additional_muffins_needed_A = 88 := by
  sorry

theorem event_B_muffins_correct : additional_muffins_needed_B = 70 := by
  sorry

theorem event_C_muffins_correct : additional_muffins_needed_C = 115 := by
  sorry

end NUMINAMATH_GPT_event_A_muffins_correct_event_B_muffins_correct_event_C_muffins_correct_l709_70986


namespace NUMINAMATH_GPT_arccos_cos_three_l709_70967

-- Defining the problem conditions
def three_radians : ℝ := 3

-- Main statement to prove
theorem arccos_cos_three : Real.arccos (Real.cos three_radians) = three_radians := 
sorry

end NUMINAMATH_GPT_arccos_cos_three_l709_70967


namespace NUMINAMATH_GPT_smallest_b_l709_70949

theorem smallest_b
  (a b : ℕ)
  (h_pos : 0 < b)
  (h_diff : a - b = 8)
  (h_gcd : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) :
  b = 4 := sorry

end NUMINAMATH_GPT_smallest_b_l709_70949


namespace NUMINAMATH_GPT_find_number_l709_70982

theorem find_number (x : ℝ) (h : (3 / 4) * (1 / 2) * (2 / 5) * x = 753.0000000000001) : 
  x = 5020.000000000001 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l709_70982


namespace NUMINAMATH_GPT_num_white_balls_l709_70948

theorem num_white_balls (W : ℕ) (h : (W : ℝ) / (6 + W) = 0.45454545454545453) : W = 5 :=
by
  sorry

end NUMINAMATH_GPT_num_white_balls_l709_70948


namespace NUMINAMATH_GPT_distance_to_city_l709_70950

variable (d : ℝ)  -- Define d as a real number

theorem distance_to_city (h1 : ¬ (d ≥ 13)) (h2 : ¬ (d ≤ 10)) :
  10 < d ∧ d < 13 :=
by
  -- Here we will formalize the proof in Lean syntax
  sorry

end NUMINAMATH_GPT_distance_to_city_l709_70950


namespace NUMINAMATH_GPT_reflection_proof_l709_70995

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

noncomputable def initial_point : ℝ × ℝ := (3, -3)
noncomputable def reflected_over_y_axis := reflect_y initial_point
noncomputable def reflected_over_x_axis := reflect_x reflected_over_y_axis

theorem reflection_proof : reflected_over_x_axis = (-3, 3) :=
  by
    -- proof goes here
    sorry

end NUMINAMATH_GPT_reflection_proof_l709_70995


namespace NUMINAMATH_GPT_correct_operation_l709_70915

theorem correct_operation : (a : ℕ) →
  (a^2 * a^3 = a^5) ∧
  (2 * a + 4 ≠ 6 * a) ∧
  ((2 * a)^2 ≠ 2 * a^2) ∧
  (a^3 / a^3 ≠ a) := sorry

end NUMINAMATH_GPT_correct_operation_l709_70915


namespace NUMINAMATH_GPT_find_digits_l709_70937

def divisible_45z_by_8 (z : ℕ) : Prop :=
  45 * z % 8 = 0

def sum_digits_divisible_by_9 (x y z : ℕ) : Prop :=
  (1 + 3 + x + y + 4 + 5 + z) % 9 = 0

def alternating_sum_digits_divisible_by_11 (x y z : ℕ) : Prop :=
  (1 - 3 + x - y + 4 - 5 + z) % 11 = 0

theorem find_digits (x y z : ℕ) (h_div8 : divisible_45z_by_8 z) (h_div9 : sum_digits_divisible_by_9 x y z) (h_div11 : alternating_sum_digits_divisible_by_11 x y z) :
  x = 2 ∧ y = 3 ∧ z = 6 := 
sorry

end NUMINAMATH_GPT_find_digits_l709_70937


namespace NUMINAMATH_GPT_fernanda_total_time_to_finish_l709_70988

noncomputable def fernanda_days_to_finish_audiobooks
  (num_audiobooks : ℕ) (hours_per_audiobook : ℕ) (hours_listened_per_day : ℕ) : ℕ :=
num_audiobooks * hours_per_audiobook / hours_listened_per_day

-- Definitions based on the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Statement to prove
theorem fernanda_total_time_to_finish :
  fernanda_days_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 := 
sorry

end NUMINAMATH_GPT_fernanda_total_time_to_finish_l709_70988


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l709_70916

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 8 = 8)
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) (h3 : a 1 + a 15 = 2 * a 8) :
  S 15 = 120 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l709_70916


namespace NUMINAMATH_GPT_evaluate_fraction_l709_70957

theorem evaluate_fraction : (35 / 0.07) = 500 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l709_70957


namespace NUMINAMATH_GPT_train_length_equals_sixty_two_point_five_l709_70989

-- Defining the conditions
noncomputable def calculate_train_length (speed_faster_train : ℝ) (speed_slower_train : ℝ) (time_seconds : ℝ) : ℝ :=
  let relative_speed_kmh := speed_faster_train - speed_slower_train
  let relative_speed_ms := (relative_speed_kmh * 5) / 18
  let distance_covered := relative_speed_ms * time_seconds
  distance_covered / 2

theorem train_length_equals_sixty_two_point_five :
  calculate_train_length 46 36 45 = 62.5 :=
sorry

end NUMINAMATH_GPT_train_length_equals_sixty_two_point_five_l709_70989


namespace NUMINAMATH_GPT_min_distance_mn_l709_70902

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance_mn : ∃ m > 0, ∀ x > 0, |f x - g x| = 1/2 + 1/2 * Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_mn_l709_70902


namespace NUMINAMATH_GPT_centroid_sum_of_squares_l709_70961

theorem centroid_sum_of_squares (a b c : ℝ) 
  (h : 1 / Real.sqrt (1 / a^2 + 1 / b^2 + 1 / c^2) = 2) : 
  1 / (a / 3) ^ 2 + 1 / (b / 3) ^ 2 + 1 / (c / 3) ^ 2 = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_centroid_sum_of_squares_l709_70961


namespace NUMINAMATH_GPT_person_speed_l709_70912

namespace EscalatorProblem

/-- The speed of the person v_p walking on the moving escalator is 3 ft/sec given the conditions -/
theorem person_speed (v_p : ℝ) 
  (escalator_speed : ℝ := 12) 
  (escalator_length : ℝ := 150) 
  (time_taken : ℝ := 10) :
  escalator_length = (v_p + escalator_speed) * time_taken → v_p = 3 := 
by sorry

end EscalatorProblem

end NUMINAMATH_GPT_person_speed_l709_70912


namespace NUMINAMATH_GPT_cyclic_quadrilateral_diameter_l709_70905

theorem cyclic_quadrilateral_diameter
  (AB BC CD DA : ℝ)
  (h1 : AB = 25)
  (h2 : BC = 39)
  (h3 : CD = 52)
  (h4 : DA = 60) : 
  ∃ D : ℝ, D = 65 :=
by
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_diameter_l709_70905


namespace NUMINAMATH_GPT_work_completion_time_of_x_l709_70998

def totalWork := 1  -- We can normalize W to 1 unit to simplify the problem

theorem work_completion_time_of_x (W : ℝ) (Wx Wy : ℝ) 
  (hx : 8 * Wx + 16 * Wy = W)
  (hy : Wy = W / 20) :
  Wx = W / 40 :=
by
  -- The proof goes here, but we just put sorry for now.
  sorry

end NUMINAMATH_GPT_work_completion_time_of_x_l709_70998


namespace NUMINAMATH_GPT_count_valid_three_digit_numbers_l709_70926

theorem count_valid_three_digit_numbers : 
  let is_valid (a b c : ℕ) := 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ b = (a + c) / 2 ∧ (a + c) % 2 = 0
  ∃ n : ℕ, (∀ a b c : ℕ, is_valid a b c → n = 45) :=
sorry

end NUMINAMATH_GPT_count_valid_three_digit_numbers_l709_70926


namespace NUMINAMATH_GPT_combined_speed_in_still_water_l709_70909

theorem combined_speed_in_still_water 
  (U1 D1 U2 D2 : ℝ) 
  (hU1 : U1 = 30) 
  (hD1 : D1 = 60) 
  (hU2 : U2 = 40) 
  (hD2 : D2 = 80) 
  : (U1 + D1) / 2 + (U2 + D2) / 2 = 105 := 
by 
  sorry

end NUMINAMATH_GPT_combined_speed_in_still_water_l709_70909


namespace NUMINAMATH_GPT_simplify_expr_l709_70932

-- Define the expression
def expr (a : ℝ) := 4 * a ^ 2 * (3 * a - 1)

-- State the theorem
theorem simplify_expr (a : ℝ) : expr a = 12 * a ^ 3 - 4 * a ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expr_l709_70932


namespace NUMINAMATH_GPT_initial_candles_count_l709_70934

section

variable (C : ℝ)
variable (h_Alyssa : C / 2 = C / 2)
variable (h_Chelsea : C / 2 - 0.7 * (C / 2) = 6)

theorem initial_candles_count : C = 40 := 
by sorry

end

end NUMINAMATH_GPT_initial_candles_count_l709_70934


namespace NUMINAMATH_GPT_julian_notes_problem_l709_70946

theorem julian_notes_problem (x y : ℤ) (h1 : 3 * x + 4 * y = 151) (h2 : x = 19 ∨ y = 19) :
  x = 25 ∨ y = 25 := 
by
  sorry

end NUMINAMATH_GPT_julian_notes_problem_l709_70946


namespace NUMINAMATH_GPT_maximize_expression_l709_70983

theorem maximize_expression
  (a b c : ℝ)
  (h1 : a ≥ 0)
  (h2 : b ≥ 0)
  (h3 : c ≥ 0)
  (h_sum : a + b + c = 3) :
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 729 / 432 := 
sorry

end NUMINAMATH_GPT_maximize_expression_l709_70983


namespace NUMINAMATH_GPT_evaluate_expression_l709_70945

theorem evaluate_expression : 6^4 - 4 * 6^3 + 6^2 - 2 * 6 + 1 = 457 := 
  by
    sorry

end NUMINAMATH_GPT_evaluate_expression_l709_70945


namespace NUMINAMATH_GPT_train_speed_in_km_per_hr_l709_70954

/-- Given the length of a train and a bridge, and the time taken for the train to cross the bridge, prove the speed of the train in km/hr -/
theorem train_speed_in_km_per_hr
  (train_length : ℕ)  -- 100 meters
  (bridge_length : ℕ) -- 275 meters
  (crossing_time : ℕ) -- 30 seconds
  (conversion_factor : ℝ) -- 1 m/s = 3.6 km/hr
  (h_train_length : train_length = 100)
  (h_bridge_length : bridge_length = 275)
  (h_crossing_time : crossing_time = 30)
  (h_conversion_factor : conversion_factor = 3.6) : 
  (train_length + bridge_length) / crossing_time * conversion_factor = 45 := 
sorry

end NUMINAMATH_GPT_train_speed_in_km_per_hr_l709_70954


namespace NUMINAMATH_GPT_train_length_correct_l709_70904

noncomputable def train_length (v_kmph : ℝ) (t_sec : ℝ) (bridge_length : ℝ) : ℝ :=
  let v_mps := v_kmph / 3.6
  let total_distance := v_mps * t_sec
  total_distance - bridge_length

theorem train_length_correct : train_length 72 12.099 132 = 109.98 :=
by
  sorry

end NUMINAMATH_GPT_train_length_correct_l709_70904


namespace NUMINAMATH_GPT_sum_a_b_eq_4_l709_70965

-- Define the problem conditions
variables (a b : ℝ)

-- State the conditions
def condition1 : Prop := 2 * a = 8
def condition2 : Prop := a^2 - b = 16

-- State the theorem
theorem sum_a_b_eq_4 (h1 : condition1 a) (h2 : condition2 a b) : a + b = 4 :=
by sorry

end NUMINAMATH_GPT_sum_a_b_eq_4_l709_70965


namespace NUMINAMATH_GPT_abs_neg_three_l709_70963

theorem abs_neg_three : abs (-3) = 3 := 
by
  sorry

end NUMINAMATH_GPT_abs_neg_three_l709_70963


namespace NUMINAMATH_GPT_find_arithmetic_sequence_l709_70955

theorem find_arithmetic_sequence (a d : ℝ) : 
(a - d) + a + (a + d) = 6 ∧ (a - d) * a * (a + d) = -10 → 
  (a = 2 ∧ d = 3 ∨ a = 2 ∧ d = -3) :=
by
  sorry

end NUMINAMATH_GPT_find_arithmetic_sequence_l709_70955


namespace NUMINAMATH_GPT_linear_function_no_first_quadrant_l709_70987

theorem linear_function_no_first_quadrant : 
  ¬ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = -3 * x - 2 := by
  sorry

end NUMINAMATH_GPT_linear_function_no_first_quadrant_l709_70987


namespace NUMINAMATH_GPT_positive_difference_of_squares_and_product_l709_70999

theorem positive_difference_of_squares_and_product (x y : ℕ) 
  (h1 : x + y = 60) (h2 : x - y = 16) :
  x^2 - y^2 = 960 ∧ x * y = 836 :=
by sorry

end NUMINAMATH_GPT_positive_difference_of_squares_and_product_l709_70999


namespace NUMINAMATH_GPT_find_g_at_7_l709_70936

noncomputable def g (x : ℝ) (a b c : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 4

theorem find_g_at_7 (a b c : ℝ) (h_symm : ∀ x : ℝ, g x a b c + g (-x) a b c = -8) (h_neg7: g (-7) a b c = 12) :
  g 7 a b c = -20 :=
by
  sorry

end NUMINAMATH_GPT_find_g_at_7_l709_70936


namespace NUMINAMATH_GPT_range_of_m_l709_70978

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  x^3 - 6 * x^2 + 9 * x + m

theorem range_of_m (m : ℝ) :
  (∃ a b c : ℝ, a < b ∧ b < c ∧ f m a = 0 ∧ f m b = 0 ∧ f m c = 0) ↔ -4 < m ∧ m < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l709_70978


namespace NUMINAMATH_GPT_Bridget_Skittles_Final_l709_70927

def Bridget_Skittles_initial : Nat := 4
def Henry_Skittles_initial : Nat := 4

def Bridget_Receives_Henry_Skittles : Nat :=
  Bridget_Skittles_initial + Henry_Skittles_initial

theorem Bridget_Skittles_Final :
  Bridget_Receives_Henry_Skittles = 8 :=
by
  -- Proof steps here, assuming sorry for now
  sorry

end NUMINAMATH_GPT_Bridget_Skittles_Final_l709_70927


namespace NUMINAMATH_GPT_sum_tenth_powers_l709_70966

theorem sum_tenth_powers (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : a^10 + b^10 = 123 :=
  sorry

end NUMINAMATH_GPT_sum_tenth_powers_l709_70966


namespace NUMINAMATH_GPT_original_total_price_l709_70941

-- Definitions of the original prices
def original_price_candy_box : ℕ := 10
def original_price_soda : ℕ := 6
def original_price_chips : ℕ := 4
def original_price_chocolate_bar : ℕ := 2

-- Mathematical problem statement
theorem original_total_price :
  original_price_candy_box + original_price_soda + original_price_chips + original_price_chocolate_bar = 22 :=
by
  sorry

end NUMINAMATH_GPT_original_total_price_l709_70941


namespace NUMINAMATH_GPT_quadratic_value_range_l709_70997

theorem quadratic_value_range (y : ℝ) (h : y^3 - 6 * y^2 + 11 * y - 6 < 0) : 
  1 ≤ y^2 - 4 * y + 5 ∧ y^2 - 4 * y + 5 ≤ 2 := 
sorry

end NUMINAMATH_GPT_quadratic_value_range_l709_70997


namespace NUMINAMATH_GPT_scientific_notation_of_00000065_l709_70947

theorem scientific_notation_of_00000065:
  (6.5 * 10^(-7)) = 0.00000065 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_scientific_notation_of_00000065_l709_70947


namespace NUMINAMATH_GPT_forest_enclosure_l709_70920

theorem forest_enclosure
  (n : ℕ)
  (a : Fin n → ℝ)
  (h_a_lt_100 : ∀ i, a i < 100)
  (d : Fin n → Fin n → ℝ)
  (h_dist : ∀ i j, i < j → d i j ≤ (a i) - (a j)) :
  ∃ f : ℝ, f = 200 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_forest_enclosure_l709_70920


namespace NUMINAMATH_GPT_second_smallest_three_digit_in_pascal_triangle_l709_70976

theorem second_smallest_three_digit_in_pascal_triangle (m n : ℕ) :
  (∀ k : ℕ, ∃! r c : ℕ, r ≥ c ∧ r.choose c = k) →
  (∃! r : ℕ, r ≥ 2 ∧ 100 = r.choose 1) →
  (m = 101 ∧ n = 101) :=
by
  sorry

end NUMINAMATH_GPT_second_smallest_three_digit_in_pascal_triangle_l709_70976


namespace NUMINAMATH_GPT_remainder_3_pow_89_plus_5_mod_7_l709_70942

theorem remainder_3_pow_89_plus_5_mod_7 :
  (3^1 % 7 = 3) ∧ (3^2 % 7 = 2) ∧ (3^3 % 7 = 6) ∧ (3^4 % 7 = 4) ∧ (3^5 % 7 = 5) ∧ (3^6 % 7 = 1) →
  ((3^89 + 5) % 7 = 3) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_remainder_3_pow_89_plus_5_mod_7_l709_70942


namespace NUMINAMATH_GPT_average_salary_of_technicians_l709_70917

theorem average_salary_of_technicians
  (total_workers : ℕ)
  (average_salary_all : ℕ)
  (average_salary_non_technicians : ℕ)
  (num_technicians : ℕ)
  (num_non_technicians : ℕ)
  (h1 : total_workers = 21)
  (h2 : average_salary_all = 8000)
  (h3 : average_salary_non_technicians = 6000)
  (h4 : num_technicians = 7)
  (h5 : num_non_technicians = 14) :
  (average_salary_all * total_workers - average_salary_non_technicians * num_non_technicians) / num_technicians = 12000 :=
by
  sorry

end NUMINAMATH_GPT_average_salary_of_technicians_l709_70917


namespace NUMINAMATH_GPT_max_apples_discarded_l709_70996

theorem max_apples_discarded (n : ℕ) : n % 7 ≤ 6 := by
  sorry

end NUMINAMATH_GPT_max_apples_discarded_l709_70996


namespace NUMINAMATH_GPT_combine_syllables_to_computer_l709_70923

/-- Conditions provided in the problem -/
def first_syllable : String := "ком" -- A big piece of a snowman
def second_syllable : String := "пьют" -- Something done by elephants at a watering hole
def third_syllable : String := "ер" -- The old name of the hard sign

/-- The result obtained by combining the three syllables should be "компьютер" -/
theorem combine_syllables_to_computer :
  (first_syllable ++ second_syllable ++ third_syllable) = "компьютер" :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_combine_syllables_to_computer_l709_70923


namespace NUMINAMATH_GPT_printer_z_time_l709_70959

theorem printer_z_time (T_X T_Y T_Z : ℝ) (hZX_Y : T_X = 2.25 * (T_Y + T_Z)) 
  (hX : T_X = 15) (hY : T_Y = 10) : T_Z = 20 :=
by
  rw [hX, hY] at hZX_Y
  sorry

end NUMINAMATH_GPT_printer_z_time_l709_70959


namespace NUMINAMATH_GPT_min_value_2x_minus_y_l709_70979

open Real

theorem min_value_2x_minus_y : ∀ (x y : ℝ), |x| ≤ y ∧ y ≤ 2 → ∃ (c : ℝ), c = 2 * x - y ∧ ∀ z, z = 2 * x - y → z ≥ -6 := sorry

end NUMINAMATH_GPT_min_value_2x_minus_y_l709_70979
