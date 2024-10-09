import Mathlib

namespace fractional_exponent_equality_l89_8935

theorem fractional_exponent_equality :
  (3 / 4 : ℚ) ^ 2017 * (- ((1:ℚ) + 1 / 3)) ^ 2018 = 4 / 3 :=
by
  sorry

end fractional_exponent_equality_l89_8935


namespace customers_remaining_l89_8990

theorem customers_remaining (init : ℕ) (left : ℕ) (remaining : ℕ) :
  init = 21 → left = 9 → remaining = 12 → init - left = remaining :=
by sorry

end customers_remaining_l89_8990


namespace tan_600_eq_sqrt3_l89_8906

theorem tan_600_eq_sqrt3 : (Real.tan (600 * Real.pi / 180)) = Real.sqrt 3 := 
by 
  -- sorry to skip the actual proof steps
  sorry

end tan_600_eq_sqrt3_l89_8906


namespace origin_movement_by_dilation_l89_8986

/-- Given a dilation of the plane that maps a circle with radius 4 centered at (3,3) 
to a circle of radius 6 centered at (7,9), calculate the distance the origin (0,0)
moves under this transformation to be 0.5 * sqrt(10). -/
theorem origin_movement_by_dilation :
  let B := (3, 3)
  let B' := (7, 9)
  let radius_B := 4
  let radius_B' := 6
  let dilation_factor := radius_B' / radius_B
  let center_of_dilation := (-1, -3)
  let initial_distance := Real.sqrt ((-1)^2 + (-3)^2) 
  let moved_distance := dilation_factor * initial_distance
  moved_distance - initial_distance = 0.5 * Real.sqrt (10) := 
by
  sorry

end origin_movement_by_dilation_l89_8986


namespace statement_A_statement_B_statement_C_l89_8950

variable {α : Type}

-- Conditions for statement A
def angle_greater (A B : ℝ) : Prop := A > B
def sin_greater (A B : ℝ) : Prop := Real.sin A > Real.sin B

-- Conditions for statement B
def acute_triangle (A B C : ℝ) : Prop := A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2
def sin_greater_than_cos (A B : ℝ) : Prop := Real.sin A > Real.cos B

-- Conditions for statement C
def obtuse_triangle (C : ℝ) : Prop := C > Real.pi / 2

-- Statement A in Lean
theorem statement_A (A B : ℝ) : angle_greater A B → sin_greater A B :=
sorry

-- Statement B in Lean
theorem statement_B {A B C : ℝ} : acute_triangle A B C → sin_greater_than_cos A B :=
sorry

-- Statement C in Lean
theorem statement_C {a b c : ℝ} (h : a^2 + b^2 < c^2) : obtuse_triangle C :=
sorry

-- Statement D in Lean (proof not needed as it's incorrect)
-- Theorem is omitted since statement D is incorrect

end statement_A_statement_B_statement_C_l89_8950


namespace range_of_f_l89_8926

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem range_of_f :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), (f x) ∈ Set.Icc (-1 : ℝ) 2) :=
by
  sorry

end range_of_f_l89_8926


namespace evaluate_expression_l89_8937

theorem evaluate_expression (x b : ℝ) (h : x = b + 4) : 2 * x - b + 5 = b + 13 := by
  sorry

end evaluate_expression_l89_8937


namespace mod_inverse_non_existence_mod_inverse_existence_l89_8940

theorem mod_inverse_non_existence (a b c d : ℕ) (h1 : 1105 = a * b * c) (h2 : 15 = d * a) :
    ¬ ∃ x : ℕ, (15 * x) % 1105 = 1 := by sorry

theorem mod_inverse_existence (a b : ℕ) (h1 : 221 = a * b) :
    ∃ x : ℕ, (15 * x) % 221 = 59 := by sorry

end mod_inverse_non_existence_mod_inverse_existence_l89_8940


namespace pq_difference_l89_8916

theorem pq_difference (p q : ℝ) (h1 : 3 / p = 6) (h2 : 3 / q = 15) : p - q = 3 / 10 := by
  sorry

end pq_difference_l89_8916


namespace right_triangle_acute_angle_l89_8941

theorem right_triangle_acute_angle (x : ℝ) 
  (h1 : 5 * x = 90) : x = 18 :=
by sorry

end right_triangle_acute_angle_l89_8941


namespace x_gt_y_necessary_not_sufficient_for_x_gt_abs_y_l89_8971

variable {x : ℝ}
variable {y : ℝ}

theorem x_gt_y_necessary_not_sufficient_for_x_gt_abs_y
  (hx : x > 0) :
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|) := by
  sorry

end x_gt_y_necessary_not_sufficient_for_x_gt_abs_y_l89_8971


namespace factorable_polynomial_l89_8905

theorem factorable_polynomial (n : ℤ) :
  ∃ (a b c d e f : ℤ), 
    (a = 1) ∧ (d = 1) ∧ 
    (b + e = 2) ∧ 
    (f = b * e) ∧ 
    (c + f + b * e = 2) ∧ 
    (c * f + b * e = -n^2) ↔ 
    (n = 0 ∨ n = 2 ∨ n = -2) :=
by
  sorry

end factorable_polynomial_l89_8905


namespace mode_of_dataSet_is_3_l89_8951

-- Define the data set
def dataSet : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

-- Define what it means to be the mode of a list
def is_mode (l : List ℕ) (n : ℕ) : Prop :=
  ∀ m, l.count n ≥ l.count m

-- Prove the mode of the data set
theorem mode_of_dataSet_is_3 : is_mode dataSet 3 :=
by
  sorry

end mode_of_dataSet_is_3_l89_8951


namespace new_line_length_l89_8977

/-- Eli drew a line that was 1.5 meters long and then erased 37.5 centimeters of it.
    We need to prove that the length of the line now is 112.5 centimeters. -/
theorem new_line_length (initial_length_m : ℝ) (erased_length_cm : ℝ) 
    (h1 : initial_length_m = 1.5) (h2 : erased_length_cm = 37.5) :
    initial_length_m * 100 - erased_length_cm = 112.5 :=
by
  sorry

end new_line_length_l89_8977


namespace number_of_boxes_on_pallet_l89_8994

-- Define the total weight of the pallet.
def total_weight_of_pallet : ℤ := 267

-- Define the weight of each box.
def weight_of_each_box : ℤ := 89

-- The theorem states that given the total weight of the pallet and the weight of each box,
-- the number of boxes on the pallet is 3.
theorem number_of_boxes_on_pallet : total_weight_of_pallet / weight_of_each_box = 3 :=
by sorry

end number_of_boxes_on_pallet_l89_8994


namespace logarithm_identity_l89_8939

noncomputable section

open Real

theorem logarithm_identity : 
  log 10 = (log (sqrt 5) / log 10 + (1 / 2) * log 20) :=
sorry

end logarithm_identity_l89_8939


namespace unique_measures_of_A_l89_8915

theorem unique_measures_of_A : 
  ∃ n : ℕ, n = 17 ∧ 
    (∀ A B : ℕ, 
      (A > 0) ∧ (B > 0) ∧ (A + B = 180) ∧ (∃ k : ℕ, A = k * B) → 
      ∃! A : ℕ, A > 0 ∧ (A + B = 180)) :=
sorry

end unique_measures_of_A_l89_8915


namespace remaining_length_l89_8968

variable (L₁ L₂: ℝ)
variable (H₁: L₁ = 0.41)
variable (H₂: L₂ = 0.33)

theorem remaining_length (L₁ L₂: ℝ) (H₁: L₁ = 0.41) (H₂: L₂ = 0.33) : L₁ - L₂ = 0.08 :=
by
  sorry

end remaining_length_l89_8968


namespace aba_div_by_7_l89_8997

theorem aba_div_by_7 (a b : ℕ) (h : (a + b) % 7 = 0) : (101 * a + 10 * b) % 7 = 0 := 
sorry

end aba_div_by_7_l89_8997


namespace find_larger_number_l89_8924

theorem find_larger_number (x y : ℕ) (h1 : y - x = 1500) (h2 : y = 6 * x + 15) : y = 1797 := by
  sorry

end find_larger_number_l89_8924


namespace perpendicular_lines_l89_8996

theorem perpendicular_lines :
  (∀ (x y : ℝ), (4 * y - 3 * x = 16)) ∧ 
  (∀ (x y : ℝ), (3 * y + 4 * x = 15)) → 
  (∃ (m1 m2 : ℝ), m1 * m2 = -1) :=
by
  sorry

end perpendicular_lines_l89_8996


namespace remaining_amoeba_is_blue_l89_8959

-- Define the initial number of amoebas for red, blue, and yellow types.
def n1 := 47
def n2 := 40
def n3 := 53

-- Define the property that remains constant, i.e., the parity of differences
def parity_diff (a b : ℕ) : Bool := (a - b) % 2 == 1

-- Initial conditions based on the given problem
def initial_conditions : Prop :=
  parity_diff n1 n2 = true ∧  -- odd
  parity_diff n1 n3 = false ∧ -- even
  parity_diff n2 n3 = true    -- odd

-- Final statement: Prove that the remaining amoeba is blue
theorem remaining_amoeba_is_blue : Prop :=
  initial_conditions ∧ (∀ final : String, final = "Blue")

end remaining_amoeba_is_blue_l89_8959


namespace find_b_perpendicular_lines_l89_8910

theorem find_b_perpendicular_lines :
  ∀ (b : ℝ), (∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 ∧ b * x - 3 * y + 6 = 0 →
      (2 / 3) * (b / 3) = -1) → b = -9 / 2 :=
sorry

end find_b_perpendicular_lines_l89_8910


namespace intersection_A_B_l89_8902

def A := {x : ℝ | (x - 1) * (x - 4) < 0}
def B := {x : ℝ | x <= 2}

theorem intersection_A_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x <= 2} :=
sorry

end intersection_A_B_l89_8902


namespace max_mn_l89_8952

theorem max_mn (m n : ℝ) (h : m + n = 1) : mn ≤ 1 / 4 :=
by
  sorry

end max_mn_l89_8952


namespace combined_work_time_l89_8960

noncomputable def work_time_first_worker : ℤ := 5
noncomputable def work_time_second_worker : ℤ := 4

theorem combined_work_time :
  (1 / (1 / work_time_first_worker + 1 / work_time_second_worker)) = 20 / 9 :=
by
  unfold work_time_first_worker work_time_second_worker
  -- The detailed reasoning and computation would go here
  sorry

end combined_work_time_l89_8960


namespace problem_1_l89_8944

theorem problem_1 (m : ℝ) : (¬ ∃ x : ℝ, x^2 + 2 * x + m ≤ 0) ↔ m > 1 := sorry

end problem_1_l89_8944


namespace find_cost_of_books_l89_8975

theorem find_cost_of_books
  (C_L C_G1 C_G2 : ℝ)
  (h1 : C_L + C_G1 + C_G2 = 1080)
  (h2 : 0.9 * C_L = 1.15 * C_G1 + 1.25 * C_G2)
  (h3 : C_G1 + C_G2 = 1080 - C_L) :
  C_L = 784 :=
sorry

end find_cost_of_books_l89_8975


namespace quadratic_inequality_solution_l89_8901

theorem quadratic_inequality_solution :
  {x : ℝ | (x^2 - 50 * x + 576) ≤ 16} = {x : ℝ | 20 ≤ x ∧ x ≤ 28} :=
sorry

end quadratic_inequality_solution_l89_8901


namespace third_square_is_G_l89_8931

-- Conditions
-- Define eight 2x2 squares, where the last placed square is E
def squares : List String := ["F", "H", "G", "D", "A", "B", "C", "E"]

-- Let the third square be G
def third_square := "G"

-- Proof statement
theorem third_square_is_G : squares.get! 2 = third_square :=
by
  sorry

end third_square_is_G_l89_8931


namespace g_1993_at_2_l89_8922

def g (x : ℚ) : ℚ := (2 + x) / (1 - 4 * x^2)

def g_n : ℕ → ℚ → ℚ 
| 0     => id
| (n+1) => λ x => g (g_n n x)

theorem g_1993_at_2 : g_n 1993 2 = 65 / 53 := 
  sorry

end g_1993_at_2_l89_8922


namespace percentage_loss_is_25_l89_8936

def cost_price := 1400
def selling_price := 1050
def loss := cost_price - selling_price
def percentage_loss := (loss / cost_price) * 100

theorem percentage_loss_is_25 : percentage_loss = 25 := by
  sorry

end percentage_loss_is_25_l89_8936


namespace abs_expression_not_positive_l89_8989

theorem abs_expression_not_positive (x : ℝ) (h : |2 * x - 7| = 0) : x = 7 / 2 :=
by
  sorry

end abs_expression_not_positive_l89_8989


namespace lcm_of_3_8_9_12_l89_8958

theorem lcm_of_3_8_9_12 : Nat.lcm (Nat.lcm 3 8) (Nat.lcm 9 12) = 72 :=
by
  sorry

end lcm_of_3_8_9_12_l89_8958


namespace segment_ratios_l89_8956

theorem segment_ratios 
  (AB_parts BC_parts : ℝ) 
  (hAB: AB_parts = 3) 
  (hBC: BC_parts = 4) 
  : AB_parts / (AB_parts + BC_parts) = 3 / 7 ∧ BC_parts / (AB_parts + BC_parts) = 4 / 7 := 
  sorry

end segment_ratios_l89_8956


namespace pieces_1994_impossible_pieces_1997_possible_l89_8946

def P (n : ℕ) : ℕ := 1 + 4 * n

theorem pieces_1994_impossible : ∀ n : ℕ, P n ≠ 1994 := 
by sorry

theorem pieces_1997_possible : ∃ n : ℕ, P n = 1997 := 
by sorry

end pieces_1994_impossible_pieces_1997_possible_l89_8946


namespace like_terms_monomials_l89_8942

theorem like_terms_monomials (m n : ℕ) (h₁ : m = 2) (h₂ : n = 1) : m + n = 3 := 
by
  sorry

end like_terms_monomials_l89_8942


namespace sum_numbers_eq_432_l89_8917

theorem sum_numbers_eq_432 (n : ℕ) (h : (n * (n + 1)) / 2 = 432) : n = 28 :=
sorry

end sum_numbers_eq_432_l89_8917


namespace rope_length_third_post_l89_8987

theorem rope_length_third_post (total first second fourth : ℕ) (h_total : total = 70) 
    (h_first : first = 24) (h_second : second = 20) (h_fourth : fourth = 12) : 
    (total - first - second - fourth) = 14 :=
by
  -- Proof is skipped, but we can state that the theorem should follow from the given conditions.
  sorry

end rope_length_third_post_l89_8987


namespace scouts_earnings_over_weekend_l89_8984

def base_pay_per_hour : ℝ := 10.00
def tip_per_customer : ℝ := 5.00
def hours_worked_saturday : ℝ := 4.0
def customers_served_saturday : ℝ := 5.0
def hours_worked_sunday : ℝ := 5.0
def customers_served_sunday : ℝ := 8.0

def earnings_saturday : ℝ := (hours_worked_saturday * base_pay_per_hour) + (customers_served_saturday * tip_per_customer)
def earnings_sunday : ℝ := (hours_worked_sunday * base_pay_per_hour) + (customers_served_sunday * tip_per_customer)

def total_earnings : ℝ := earnings_saturday + earnings_sunday

theorem scouts_earnings_over_weekend : total_earnings = 155.00 := by
  sorry

end scouts_earnings_over_weekend_l89_8984


namespace no_integers_solution_l89_8973

theorem no_integers_solution (k : ℕ) (x y z : ℤ) (hx1 : 0 < x) (hx2 : x < k) (hy1 : 0 < y) (hy2 : y < k) (hz : z > 0) :
  x^k + y^k ≠ z^k :=
sorry

end no_integers_solution_l89_8973


namespace max_tiles_accommodated_l89_8963

/-- 
The rectangular tiles, each of size 40 cm by 28 cm, must be laid horizontally on a rectangular floor
of size 280 cm by 240 cm, such that the tiles do not overlap, and they are placed in an alternating
checkerboard pattern with edges jutting against each other on all edges. A tile can be placed in any
orientation so long as its edges are parallel to the edges of the floor, and it follows the required
checkerboard pattern. No tile should overshoot any edge of the floor. Determine the maximum number 
of tiles that can be accommodated on the floor while adhering to the placement pattern.
-/
theorem max_tiles_accommodated (tile_len tile_wid floor_len floor_wid : ℕ)
  (h_tile_len : tile_len = 40)
  (h_tile_wid : tile_wid = 28)
  (h_floor_len : floor_len = 280)
  (h_floor_wid : floor_wid = 240) :
  tile_len * tile_wid * 12 ≤ floor_len * floor_wid :=
by 
  sorry

end max_tiles_accommodated_l89_8963


namespace triangle_acute_angle_contradiction_l89_8920

theorem triangle_acute_angle_contradiction
  (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h_tri : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_at_most_one_acute : (α < 90 ∧ β ≥ 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β < 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β ≥ 90 ∧ γ < 90)) :
  false :=
by
  sorry

end triangle_acute_angle_contradiction_l89_8920


namespace represent_380000_in_scientific_notation_l89_8961

theorem represent_380000_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 380000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.8 ∧ n = 5 :=
by
  sorry

end represent_380000_in_scientific_notation_l89_8961


namespace intersection_of_M_and_N_l89_8954

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l89_8954


namespace sum_of_values_of_x_l89_8980

noncomputable def g (x : ℝ) : ℝ :=
if x < 3 then 7 * x + 10 else 3 * x - 18

theorem sum_of_values_of_x (h : ∃ x : ℝ, g x = 5) :
  (∃ x1 x2 : ℝ, g x1 = 5 ∧ g x2 = 5) → (x1 + x2 = 18 / 7) :=
sorry

end sum_of_values_of_x_l89_8980


namespace polynomial_division_result_q_neg1_r_1_sum_l89_8983

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 + 5 * x^3 - 4 * x^2 + 2 * x + 1
noncomputable def d (x : ℝ) : ℝ := x^2 + 2 * x - 3
noncomputable def q (x : ℝ) : ℝ := 3 * x^2 + x
noncomputable def r (x : ℝ) : ℝ := 7 * x + 4

theorem polynomial_division_result : f (-1) = q (-1) * d (-1) + r (-1)
  ∧ f 1 = q 1 * d 1 + r 1 :=
by sorry

theorem q_neg1_r_1_sum : (q (-1) + r 1) = 13 :=
by sorry

end polynomial_division_result_q_neg1_r_1_sum_l89_8983


namespace min_distance_value_l89_8921

theorem min_distance_value (x1 x2 y1 y2 : ℝ) 
  (h1 : (e ^ x1 + 2 * x1) / (3 * y1) = 1 / 3)
  (h2 : (x2 - 1) / y2 = 1 / 3) :
  ((x1 - x2)^2 + (y1 - y2)^2) = 8 / 5 :=
by
  sorry

end min_distance_value_l89_8921


namespace det_B_squared_minus_3IB_l89_8976

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 4], ![3, 1]]
def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem det_B_squared_minus_3IB :
  det (B * B - 3 * I * B) = 100 := by
  sorry

end det_B_squared_minus_3IB_l89_8976


namespace trams_required_l89_8955

theorem trams_required (initial_trams : ℕ) (initial_interval : ℚ) (reduction_fraction : ℚ) :
  initial_trams = 12 ∧ initial_interval = 5 ∧ reduction_fraction = 1/5 →
  (initial_trams + initial_trams * reduction_fraction - initial_trams) = 3 :=
by
  sorry

end trams_required_l89_8955


namespace incorrect_expression_l89_8995

theorem incorrect_expression (x y : ℝ) (h : x > y) : ¬ (3 - x > 3 - y) :=
by
  sorry

end incorrect_expression_l89_8995


namespace proof_stage_constancy_l89_8967

-- Definitions of stages
def Stage1 := "Fertilization and seed germination"
def Stage2 := "Flowering and pollination"
def Stage3 := "Meiosis and fertilization"
def Stage4 := "Formation of sperm and egg cells"

-- Question: Which stages maintain chromosome constancy and promote genetic recombination in plant life?
def Q := "Which stages maintain chromosome constancy and promote genetic recombination in plant life?"

-- Correct answer
def Answer := Stage3

-- Conditions
def s1 := Stage1
def s2 := Stage2
def s3 := Stage3
def s4 := Stage4

-- Theorem statement
theorem proof_stage_constancy : Q = Answer := by
  sorry

end proof_stage_constancy_l89_8967


namespace ones_digit_of_prime_p_l89_8999

theorem ones_digit_of_prime_p (p q r s : ℕ) (hp : p > 5) (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q) (prime_r : Nat.Prime r) (prime_s : Nat.Prime s)
  (hseq1 : q = p + 8) (hseq2 : r = p + 16) (hseq3 : s = p + 24) 
  : p % 10 = 3 := 
sorry

end ones_digit_of_prime_p_l89_8999


namespace find_s_l89_8904

section
variables {a b c p q s : ℕ}

-- Conditions given in the problem
variables (h1 : a + b = p)
variables (h2 : p + c = s)
variables (h3 : s + a = q)
variables (h4 : b + c + q = 18)
variables (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧ b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧ c ≠ p ∧ c ≠ q ∧ c ≠ s ∧ p ≠ q ∧ p ≠ s ∧ q ≠ s)
variables (h6 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0)

-- Statement of the problem
theorem find_s (h1 : a + b = p) (h2 : p + c = s) (h3 : s + a = q) (h4 : b + c + q = 18)
  (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧ b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧ c ≠ p ∧ c ≠ q ∧ c ≠ s ∧ p ≠ q ∧ p ≠ s ∧ q ≠ s)
  (h6 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0) :
  s = 9 :=
sorry
end

end find_s_l89_8904


namespace tan_sum_trig_identity_l89_8964

variable {α : ℝ}

-- Part (I)
theorem tan_sum (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
by
  sorry

-- Part (II)
theorem trig_identity (h : Real.tan α = 2) : 
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = 3 / 2 :=
by
  sorry

end tan_sum_trig_identity_l89_8964


namespace necessary_but_not_sufficient_condition_l89_8998

def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x ^ 2

theorem necessary_but_not_sufficient_condition :
  (∀ x, q x → p x) ∧ (¬ ∀ x, p x → q x) :=
by
  sorry

end necessary_but_not_sufficient_condition_l89_8998


namespace period2_students_is_8_l89_8943

-- Definitions according to conditions
def period1_students : Nat := 11
def relationship (x : Nat) := 2 * x - 5

-- Lean 4 statement
theorem period2_students_is_8 (x: Nat) (h: relationship x = period1_students) : x = 8 := 
by 
  -- Placeholder for the proof
  sorry

end period2_students_is_8_l89_8943


namespace tan_alpha_second_quadrant_l89_8903

noncomputable def tan_alpha (α : ℝ) : ℝ := Real.tan α

theorem tan_alpha_second_quadrant (α : ℝ) 
  (h1 : α > π / 2 ∧ α < π)
  (h2 : Real.cos (π / 2 - α) = 4 / 5) :
  tan_alpha α = -4 / 3 :=
by
  sorry

end tan_alpha_second_quadrant_l89_8903


namespace total_frisbees_l89_8988

-- Let x be the number of $3 frisbees and y be the number of $4 frisbees.
variables (x y : ℕ)

-- Condition 1: Total sales amount is 200 dollars.
def condition1 : Prop := 3 * x + 4 * y = 200

-- Condition 2: At least 8 $4 frisbees were sold.
def condition2 : Prop := y >= 8

-- Prove that the total number of frisbees sold is 64.
theorem total_frisbees (h1 : condition1 x y) (h2 : condition2 y) : x + y = 64 :=
by
  sorry

end total_frisbees_l89_8988


namespace good_horse_catchup_l89_8930

theorem good_horse_catchup 
  (x : ℕ) 
  (good_horse_speed : ℕ) (slow_horse_speed : ℕ) (head_start_days : ℕ) 
  (H1 : good_horse_speed = 240)
  (H2 : slow_horse_speed = 150)
  (H3 : head_start_days = 12) :
  good_horse_speed * x - slow_horse_speed * x = slow_horse_speed * head_start_days :=
by
  sorry

end good_horse_catchup_l89_8930


namespace combined_platforms_length_is_correct_l89_8972

noncomputable def combined_length_of_platforms (lengthA lengthB speedA_kmph speedB_kmph timeA_sec timeB_sec : ℝ) : ℝ :=
  let speedA := speedA_kmph * (1000 / 3600)
  let speedB := speedB_kmph * (1000 / 3600)
  let distanceA := speedA * timeA_sec
  let distanceB := speedB * timeB_sec
  let platformA := distanceA - lengthA
  let platformB := distanceB - lengthB
  platformA + platformB

theorem combined_platforms_length_is_correct :
  combined_length_of_platforms 650 450 115 108 30 25 = 608.32 := 
by 
  sorry

end combined_platforms_length_is_correct_l89_8972


namespace beehive_bee_count_l89_8962

theorem beehive_bee_count {a : ℕ → ℕ} (h₀ : a 0 = 1)
  (h₁ : a 1 = 6)
  (hn : ∀ n, a (n + 1) = a n + 5 * a n) :
  a 6 = 46656 :=
  sorry

end beehive_bee_count_l89_8962


namespace horner_v3_value_l89_8985

-- Define constants
def a_n : ℤ := 2 -- Leading coefficient of x^5
def a_3 : ℤ := -3 -- Coefficient of x^3
def a_2 : ℤ := 5 -- Coefficient of x^2
def a_0 : ℤ := -4 -- Constant term
def x : ℤ := 2 -- Given value of x

-- Horner's method sequence for the coefficients
def v_0 : ℤ := a_n -- Initial value v_0
def v_1 : ℤ := v_0 * x -- Calculated as v_0 * x
def v_2 : ℤ := v_1 * x + a_3 -- Calculated as v_1 * x + a_3 (coefficient of x^3)
def v_3 : ℤ := v_2 * x + a_2 -- Calculated as v_2 * x + a_2 (coefficient of x^2)

theorem horner_v3_value : v_3 = 15 := 
by
  -- Formal proof would go here, skipped due to problem specifications
  sorry

end horner_v3_value_l89_8985


namespace initial_number_of_peanuts_l89_8949

theorem initial_number_of_peanuts (x : ℕ) (h : x + 2 = 6) : x = 4 :=
sorry

end initial_number_of_peanuts_l89_8949


namespace rational_inequality_solution_l89_8927

open Set

theorem rational_inequality_solution (x : ℝ) :
  (x < -1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 5)) ↔ (x - 5) / ((x - 2) * (x^2 - 1)) < 0 := 
sorry

end rational_inequality_solution_l89_8927


namespace y_directly_varies_as_square_l89_8953

theorem y_directly_varies_as_square (k : ℚ) (y : ℚ) (x : ℚ) 
  (h1 : y = k * x ^ 2) (h2 : y = 18) (h3 : x = 3) : 
  ∃ y : ℚ, ∀ x : ℚ, x = 6 → y = 72 :=
by
  sorry

end y_directly_varies_as_square_l89_8953


namespace count_perfect_squares_l89_8909

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def E1 : ℕ := 1^3 + 2^3
def E2 : ℕ := 1^3 + 2^3 + 3^3
def E3 : ℕ := 1^3 + 2^3 + 3^3 + 4^3
def E4 : ℕ := 1^3 + 2^3 + 3^3 + 4^3 + 5^3

theorem count_perfect_squares :
  (is_perfect_square E1 → true) ∧
  (is_perfect_square E2 → true) ∧
  (is_perfect_square E3 → true) ∧
  (is_perfect_square E4 → true) →
  (∀ n : ℕ, (n = 4) ↔
    ∃ E1 E2 E3 E4, is_perfect_square E1 ∧ is_perfect_square E2 ∧ is_perfect_square E3 ∧ is_perfect_square E4) :=
by
  sorry

end count_perfect_squares_l89_8909


namespace remainder_of_sum_l89_8970

theorem remainder_of_sum (n : ℤ) : ((5 - n) + (n + 4)) % 5 = 4 := 
by 
  -- proof goes here
  sorry

end remainder_of_sum_l89_8970


namespace all_equal_l89_8913

variable (a : ℕ → ℝ)

axiom h1 : a 1 - 3 * a 2 + 2 * a 3 ≥ 0
axiom h2 : a 2 - 3 * a 3 + 2 * a 4 ≥ 0
axiom h3 : a 3 - 3 * a 4 + 2 * a 5 ≥ 0
axiom h4 : ∀ n, 4 ≤ n ∧ n ≤ 98 → a n - 3 * a (n + 1) + 2 * a (n + 2) ≥ 0
axiom h99 : a 99 - 3 * a 100 + 2 * a 1 ≥ 0
axiom h100 : a 100 - 3 * a 1 + 2 * a 2 ≥ 0

theorem all_equal : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 → a i = a j := by
  sorry

end all_equal_l89_8913


namespace shortest_distance_l89_8907

theorem shortest_distance 
  (C : ℝ × ℝ) (B : ℝ × ℝ) (stream : ℝ)
  (hC : C = (0, -3))
  (hB : B = (9, -8))
  (hStream : stream = 0) :
  ∃ d : ℝ, d = 3 + Real.sqrt 202 :=
by
  sorry

end shortest_distance_l89_8907


namespace number_of_games_can_buy_l89_8945

-- Definitions based on the conditions
def initial_money : ℕ := 42
def spent_money : ℕ := 10
def game_cost : ℕ := 8

-- The statement we need to prove: Mike can buy 4 games given the conditions
theorem number_of_games_can_buy : (initial_money - spent_money) / game_cost = 4 :=
by
  sorry

end number_of_games_can_buy_l89_8945


namespace quadrangular_pyramid_edge_length_l89_8992

theorem quadrangular_pyramid_edge_length :
  ∃ e : ℝ, 8 * e = 14.8 ∧ e = 1.85 :=
  sorry

end quadrangular_pyramid_edge_length_l89_8992


namespace weigh_1_to_10_kg_l89_8925

theorem weigh_1_to_10_kg (n : ℕ) : 1 ≤ n ∧ n ≤ 10 →
  ∃ (a b c : ℤ), 
    (abs a ≤ 1 ∧ abs b ≤ 1 ∧ abs c ≤ 1 ∧
    (n = a * 3 + b * 4 + c * 9)) :=
by sorry

end weigh_1_to_10_kg_l89_8925


namespace rearrangement_count_is_two_l89_8911

def is_adjacent (c1 c2 : Char) : Bool :=
  (c1 = 'a' ∧ c2 = 'b') ∨
  (c1 = 'b' ∧ c2 = 'c') ∨
  (c1 = 'c' ∧ c2 = 'd') ∨
  (c1 = 'd' ∧ c2 = 'e') ∨
  (c1 = 'b' ∧ c2 = 'a') ∨
  (c1 = 'c' ∧ c2 = 'b') ∨
  (c1 = 'd' ∧ c2 = 'c') ∨
  (c1 = 'e' ∧ c2 = 'd')

def no_adjacent_letters (s : List Char) : Bool :=
  match s with
  | [] => true
  | [_] => true
  | c1 :: c2 :: cs => 
    ¬ is_adjacent c1 c2 ∧ no_adjacent_letters (c2 :: cs)

def valid_rearrangements_count : Nat :=
  let perms := List.permutations ['a', 'b', 'c', 'd', 'e']
  perms.filter no_adjacent_letters |>.length

theorem rearrangement_count_is_two :
  valid_rearrangements_count = 2 :=
by sorry

end rearrangement_count_is_two_l89_8911


namespace stratified_sampling_male_athletes_l89_8938

theorem stratified_sampling_male_athletes (total_males : ℕ) (total_females : ℕ) (sample_size : ℕ)
  (total_population : ℕ) (male_sample_fraction : ℚ) (n_sample_males : ℕ) :
  total_males = 56 →
  total_females = 42 →
  sample_size = 28 →
  total_population = total_males + total_females →
  male_sample_fraction = (sample_size : ℚ) / (total_population : ℚ) →
  n_sample_males = (total_males : ℚ) * male_sample_fraction →
  n_sample_males = 16 := by
  intros h_males h_females h_samples h_population h_fraction h_final
  sorry

end stratified_sampling_male_athletes_l89_8938


namespace xy_sum_possible_values_l89_8948

theorem xy_sum_possible_values (x y : ℕ) (h1 : x < 20) (h2 : y < 20) (h3 : 0 < x) (h4 : 0 < y) (h5 : x + y + x * y = 95) :
  x + y = 18 ∨ x + y = 20 :=
by {
  sorry
}

end xy_sum_possible_values_l89_8948


namespace JiaZi_second_column_l89_8947

theorem JiaZi_second_column :
  let heavenlyStemsCycle := 10
  let earthlyBranchesCycle := 12
  let firstOccurrence := 1
  let lcmCycle := Nat.lcm heavenlyStemsCycle earthlyBranchesCycle
  let secondOccurrence := firstOccurrence + lcmCycle
  secondOccurrence = 61 :=
by
  sorry

end JiaZi_second_column_l89_8947


namespace set_intersection_problem_l89_8919

def set_product (A B : Set ℕ) : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}
def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 3}
def C : Set ℕ := {x | x^2 - 3 * x + 2 = 0}

theorem set_intersection_problem :
  (set_product A B) ∩ (set_product B C) = {2, 6} :=
by
  sorry

end set_intersection_problem_l89_8919


namespace messages_on_monday_l89_8929

theorem messages_on_monday (M : ℕ) (h0 : 200 + 500 + 1000 = 1700) (h1 : M + 1700 = 2000) : M = 300 :=
by
  -- Maths proof step here
  sorry

end messages_on_monday_l89_8929


namespace cos_90_eq_0_l89_8974

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l89_8974


namespace royal_family_children_l89_8932

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end royal_family_children_l89_8932


namespace price_per_package_l89_8908

theorem price_per_package (P : ℝ) (hp1 : 10 * P + 50 * (4 / 5 * P) = 1096) :
  P = 21.92 :=
by 
  sorry

end price_per_package_l89_8908


namespace sasha_remainder_l89_8928

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l89_8928


namespace convert_decimal_to_fraction_l89_8934

theorem convert_decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := 
by
  sorry

end convert_decimal_to_fraction_l89_8934


namespace complete_the_square_transforms_l89_8966

theorem complete_the_square_transforms (x : ℝ) :
  (x^2 + 8 * x + 7 = 0) → ((x + 4) ^ 2 = 9) :=
by
  intro h
  have step1 : x^2 + 8 * x = -7 := by sorry
  have step2 : x^2 + 8 * x + 16 = -7 + 16 := by sorry
  have step3 : (x + 4) ^ 2 = 9 := by sorry
  exact step3

end complete_the_square_transforms_l89_8966


namespace range_of_a_no_solution_inequality_l89_8965

theorem range_of_a_no_solution_inequality (a : ℝ) :
  (∀ x : ℝ, x + 2 > 3 → x < a) ↔ a ≤ 1 :=
by {
  sorry
}

end range_of_a_no_solution_inequality_l89_8965


namespace grading_ratio_l89_8969

noncomputable def num_questions : ℕ := 100
noncomputable def correct_answers : ℕ := 91
noncomputable def score_received : ℕ := 73
noncomputable def incorrect_answers : ℕ := num_questions - correct_answers
noncomputable def total_points_subtracted : ℕ := correct_answers - score_received
noncomputable def points_per_incorrect : ℚ := total_points_subtracted / incorrect_answers

theorem grading_ratio (h: (points_per_incorrect : ℚ) = 2) :
  2 / 1 = points_per_incorrect / 1 :=
by sorry

end grading_ratio_l89_8969


namespace room_breadth_l89_8991

theorem room_breadth (length height diagonal : ℕ) (h_length : length = 12) (h_height : height = 9) (h_diagonal : diagonal = 17) : 
  ∃ breadth : ℕ, breadth = 8 :=
by
  -- Using the three-dimensional Pythagorean theorem:
  -- d² = length² + breadth² + height²
  -- 17² = 12² + b² + 9²
  -- 289 = 144 + b² + 81
  -- 289 = 225 + b²
  -- b² = 289 - 225
  -- b² = 64
  -- Taking the square root of both sides, we find:
  -- b = √64
  -- b = 8
  let b := 8
  existsi b
  -- This is a skip step, where we assert the breadth equals 8
  sorry

end room_breadth_l89_8991


namespace min_distance_from_curve_to_line_l89_8957

open Real

-- Definitions and conditions
def curve_eq (x y: ℝ) : Prop := (x^2 - y - 2 * log (sqrt x) = 0)
def line_eq (x y: ℝ) : Prop := (4 * x + 4 * y + 1 = 0)

-- The main statement
theorem min_distance_from_curve_to_line :
  ∃ (x y : ℝ), curve_eq x y ∧ y = x^2 - 2 * log (sqrt x) ∧ line_eq x y ∧ y = -x - 1/4 ∧ 
               |4 * (1/2) + 4 * ((1/4) + log 2) + 1| / sqrt 32 = sqrt 2 / 2 * (1 + log 2) :=
by
  -- We skip the proof as requested, using sorry:
  sorry

end min_distance_from_curve_to_line_l89_8957


namespace seashells_given_to_Jessica_l89_8993

-- Define the initial number of seashells Dan had
def initialSeashells : ℕ := 56

-- Define the number of seashells Dan has left
def seashellsLeft : ℕ := 22

-- Define the number of seashells Dan gave to Jessica
def seashellsGiven : ℕ := initialSeashells - seashellsLeft

-- State the theorem to prove
theorem seashells_given_to_Jessica :
  seashellsGiven = 34 :=
by
  -- Begin the proof here
  sorry

end seashells_given_to_Jessica_l89_8993


namespace sin_cos_ratio_l89_8981

theorem sin_cos_ratio (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2)
  (h2 : Real.tan (α - β) = 3) : 
  Real.sin (2 * α) / Real.cos (2 * β) = (Real.sqrt 5 + 3 * Real.sqrt 2) / 20 := 
by
  sorry

end sin_cos_ratio_l89_8981


namespace boxes_needed_to_pack_all_muffins_l89_8979

theorem boxes_needed_to_pack_all_muffins
  (total_muffins : ℕ := 95)
  (muffins_per_box : ℕ := 5)
  (available_boxes : ℕ := 10) :
  (total_muffins / muffins_per_box) - available_boxes = 9 :=
by
  sorry

end boxes_needed_to_pack_all_muffins_l89_8979


namespace equivar_proof_l89_8918

variable {x : ℝ} {m : ℝ}

def p (m : ℝ) : Prop := m > 2

def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 4 * m * x + 4 * m - 3 ≥ 0

theorem equivar_proof (m : ℝ) (h : ¬p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 := by
  sorry

end equivar_proof_l89_8918


namespace a_n_value_l89_8914

theorem a_n_value (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 3) (h2 : ∀ n, S (n + 1) = 2 * S n) (h3 : S 1 = a 1)
  (h4 : ∀ n, S n = 3 * 2^(n - 1)) : a 4 = 12 :=
sorry

end a_n_value_l89_8914


namespace part1_l89_8982

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end part1_l89_8982


namespace sum_gcd_lcm_l89_8933

theorem sum_gcd_lcm (A B : ℕ) (hA : A = Nat.gcd 10 (Nat.gcd 15 25)) (hB : B = Nat.lcm 10 (Nat.lcm 15 25)) :
  A + B = 155 :=
by
  sorry

end sum_gcd_lcm_l89_8933


namespace average_cost_parking_l89_8912

theorem average_cost_parking :
  let cost_first_2_hours := 12.00
  let cost_per_additional_hour := 1.75
  let total_hours := 9
  let total_cost := cost_first_2_hours + cost_per_additional_hour * (total_hours - 2)
  let average_cost_per_hour := total_cost / total_hours
  average_cost_per_hour = 2.69 :=
by
  sorry

end average_cost_parking_l89_8912


namespace walter_exceptional_days_l89_8978

theorem walter_exceptional_days :
  ∃ (w b : ℕ), 
  b + w = 10 ∧ 
  3 * b + 5 * w = 36 ∧ 
  w = 3 :=
by
  sorry

end walter_exceptional_days_l89_8978


namespace find_k_l89_8923

theorem find_k {k : ℚ} :
    (∃ x y : ℚ, y = 3 * x + 6 ∧ y = -4 * x - 20 ∧ y = 2 * x + k) →
    k = 16 / 7 := 
  sorry

end find_k_l89_8923


namespace sqrt_expression_l89_8900

theorem sqrt_expression : 2 * Real.sqrt 3 - (3 * Real.sqrt 2 + Real.sqrt 3) = Real.sqrt 3 - 3 * Real.sqrt 2 :=
by
  sorry

end sqrt_expression_l89_8900
