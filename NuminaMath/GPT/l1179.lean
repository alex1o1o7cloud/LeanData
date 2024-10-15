import Mathlib

namespace NUMINAMATH_GPT_investor_share_purchase_price_l1179_117975

theorem investor_share_purchase_price 
  (dividend_rate : ℝ) 
  (face_value : ℝ) 
  (roi : ℝ) 
  (purchase_price : ℝ)
  (h1 : dividend_rate = 0.125)
  (h2 : face_value = 60)
  (h3 : roi = 0.25)
  (h4 : 0.25 = (0.125 * 60) / purchase_price) 
  : purchase_price = 30 := 
sorry

end NUMINAMATH_GPT_investor_share_purchase_price_l1179_117975


namespace NUMINAMATH_GPT_sum_three_times_m_and_half_n_square_diff_minus_square_sum_l1179_117946

-- Problem (1) Statement
theorem sum_three_times_m_and_half_n (m n : ℝ) : 3 * m + 1 / 2 * n = 3 * m + 1 / 2 * n :=
by
  sorry

-- Problem (2) Statement
theorem square_diff_minus_square_sum (a b : ℝ) : (a - b) ^ 2 - (a + b) ^ 2 = (a - b) ^ 2 - (a + b) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_three_times_m_and_half_n_square_diff_minus_square_sum_l1179_117946


namespace NUMINAMATH_GPT_solve_z_plus_inv_y_l1179_117932

theorem solve_z_plus_inv_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 4) (h3 : y + 1 / x = 30) :
  z + 1 / y = 36 / 119 :=
sorry

end NUMINAMATH_GPT_solve_z_plus_inv_y_l1179_117932


namespace NUMINAMATH_GPT_root_interval_l1179_117929

def f (x : ℝ) : ℝ := 5 * x - 7

theorem root_interval : ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- Proof steps should be here
  sorry

end NUMINAMATH_GPT_root_interval_l1179_117929


namespace NUMINAMATH_GPT_total_weekly_pay_proof_l1179_117987

-- Define the weekly pay for employees X and Y
def weekly_pay_employee_y : ℝ := 260
def weekly_pay_employee_x : ℝ := 1.2 * weekly_pay_employee_y

-- Definition of total weekly pay
def total_weekly_pay : ℝ := weekly_pay_employee_x + weekly_pay_employee_y

-- Theorem stating the total weekly pay equals 572
theorem total_weekly_pay_proof : total_weekly_pay = 572 := by
  sorry

end NUMINAMATH_GPT_total_weekly_pay_proof_l1179_117987


namespace NUMINAMATH_GPT_max_value_sum_faces_edges_vertices_l1179_117922

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

def pyramid_faces_added : ℕ := 4
def pyramid_base_faces_covered : ℕ := 1
def pyramid_edges_added : ℕ := 4
def pyramid_vertices_added : ℕ := 1

def resulting_faces : ℕ := rectangular_prism_faces - pyramid_base_faces_covered + pyramid_faces_added
def resulting_edges : ℕ := rectangular_prism_edges + pyramid_edges_added
def resulting_vertices : ℕ := rectangular_prism_vertices + pyramid_vertices_added

def sum_resulting_faces_edges_vertices : ℕ := resulting_faces + resulting_edges + resulting_vertices

theorem max_value_sum_faces_edges_vertices : sum_resulting_faces_edges_vertices = 34 :=
by
  sorry

end NUMINAMATH_GPT_max_value_sum_faces_edges_vertices_l1179_117922


namespace NUMINAMATH_GPT_baker_cake_count_l1179_117902

theorem baker_cake_count :
  let initial_cakes := 62
  let additional_cakes := 149
  let sold_cakes := 144
  initial_cakes + additional_cakes - sold_cakes = 67 :=
by
  sorry

end NUMINAMATH_GPT_baker_cake_count_l1179_117902


namespace NUMINAMATH_GPT_range_of_k_l1179_117961

noncomputable def quadratic_inequality (k : ℝ) := 
  ∀ x : ℝ, 2 * k * x^2 + k * x - (3 / 8) < 0

theorem range_of_k (k : ℝ) :
  (quadratic_inequality k) → -3 < k ∧ k < 0 := sorry

end NUMINAMATH_GPT_range_of_k_l1179_117961


namespace NUMINAMATH_GPT_smallest_n_division_l1179_117924

-- Lean statement equivalent to the mathematical problem
theorem smallest_n_division (n : ℕ) (hn : n ≥ 3) : 
  (∃ (s : Finset ℕ), (∀ m ∈ s, 3 ≤ m ∧ m ≤ 2006) ∧ s.card = n - 2) ↔ n = 3 := 
sorry

end NUMINAMATH_GPT_smallest_n_division_l1179_117924


namespace NUMINAMATH_GPT_range_of_sum_l1179_117957

theorem range_of_sum (x y : ℝ) (h : x^2 + x + y^2 + y = 0) : 
  -2 ≤ x + y ∧ x + y ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_sum_l1179_117957


namespace NUMINAMATH_GPT_band_members_count_l1179_117952

theorem band_members_count :
  ∃ n k m : ℤ, n = 10 * k + 4 ∧ n = 12 * m + 6 ∧ 200 ≤ n ∧ n ≤ 300 ∧ n = 254 :=
by
  -- Declaration of the theorem properties
  sorry

end NUMINAMATH_GPT_band_members_count_l1179_117952


namespace NUMINAMATH_GPT_xyz_value_l1179_117948

theorem xyz_value (x y z : ℝ)
    (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14) :
    x * y * z = 16 / 3 := by
    sorry

end NUMINAMATH_GPT_xyz_value_l1179_117948


namespace NUMINAMATH_GPT_green_pairs_count_l1179_117951

variable (blueShirtedStudents : Nat)
variable (yellowShirtedStudents : Nat)
variable (greenShirtedStudents : Nat)
variable (totalStudents : Nat)
variable (totalPairs : Nat)
variable (blueBluePairs : Nat)

def green_green_pairs (blueShirtedStudents yellowShirtedStudents greenShirtedStudents totalStudents totalPairs blueBluePairs : Nat) : Nat := 
  greenShirtedStudents / 2

theorem green_pairs_count
  (h1 : blueShirtedStudents = 70)
  (h2 : yellowShirtedStudents = 80)
  (h3 : greenShirtedStudents = 50)
  (h4 : totalStudents = 200)
  (h5 : totalPairs = 100)
  (h6 : blueBluePairs = 30) : 
  green_green_pairs blueShirtedStudents yellowShirtedStudents greenShirtedStudents totalStudents totalPairs blueBluePairs = 25 := by
  sorry

end NUMINAMATH_GPT_green_pairs_count_l1179_117951


namespace NUMINAMATH_GPT_max_d_n_is_one_l1179_117916

open Int

/-- The sequence definition -/
def seq (n : ℕ) : ℤ := 100 + n^3

/-- The definition of d_n -/
def d_n (n : ℕ) : ℤ := gcd (seq n) (seq (n + 1))

/-- The theorem stating the maximum value of d_n for positive integers is 1 -/
theorem max_d_n_is_one : ∀ (n : ℕ), 1 ≤ n → d_n n = 1 := by
  sorry

end NUMINAMATH_GPT_max_d_n_is_one_l1179_117916


namespace NUMINAMATH_GPT_problem_a_problem_c_problem_d_l1179_117940

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end NUMINAMATH_GPT_problem_a_problem_c_problem_d_l1179_117940


namespace NUMINAMATH_GPT_isosceles_if_interior_angles_equal_l1179_117986

-- Definition for a triangle
structure Triangle :=
  (A B C : Type)

-- Defining isosceles triangle condition
def is_isosceles (T : Triangle) :=
  ∃ a b c : ℝ, (a = b) ∨ (b = c) ∨ (a = c)

-- Defining the angle equality condition
def interior_angles_equal (T : Triangle) :=
  ∃ a b c : ℝ, (a = b) ∨ (b = c) ∨ (a = c)

-- Main theorem stating the contrapositive
theorem isosceles_if_interior_angles_equal (T : Triangle) : 
  interior_angles_equal T → is_isosceles T :=
by sorry

end NUMINAMATH_GPT_isosceles_if_interior_angles_equal_l1179_117986


namespace NUMINAMATH_GPT_reduced_price_l1179_117941

theorem reduced_price (P R : ℝ) (h1 : R = 0.8 * P) (h2 : 600 = (600 / P + 4) * R) : R = 30 := 
by
  sorry

end NUMINAMATH_GPT_reduced_price_l1179_117941


namespace NUMINAMATH_GPT_max_S_value_l1179_117997

noncomputable def max_S (A C : ℝ) [DecidableEq ℝ] : ℝ :=
  if h : 0 < A ∧ A < 2 * Real.pi / 3 ∧ A + C = 2 * Real.pi / 3 then
    (Real.sqrt 3 / 6) * Real.sin (2 * A - Real.pi / 3) + (Real.sqrt 3 / 12)
  else
    0

theorem max_S_value :
  ∃ (A C : ℝ), A + C = 2 * Real.pi / 3 ∧
    (S = (Real.sqrt 3 / 3) * Real.sin A * Real.sin C) ∧
    (max_S A C = Real.sqrt 3 / 4) := 
sorry

end NUMINAMATH_GPT_max_S_value_l1179_117997


namespace NUMINAMATH_GPT_monotonic_f_inequality_f_over_h_l1179_117905

noncomputable def f (x : ℝ) : ℝ := 1 + (1 / x) + Real.log x + (Real.log x / x)

theorem monotonic_f :
  ∀ x : ℝ, x > 0 → ∃ I : Set ℝ, (I = Set.Ioo 0 x ∨ I = Set.Icc 0 x) ∧ (∀ y ∈ I, y > 0 → f y = f x) :=
by
  sorry

theorem inequality_f_over_h :
  ∀ x : ℝ, x > 1 → (f x) / (Real.exp 1 + 1) > (2 * Real.exp (x - 1)) / (x * Real.exp x + 1) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_f_inequality_f_over_h_l1179_117905


namespace NUMINAMATH_GPT_problem_statement_l1179_117917

def f (x : ℤ) : ℤ := 2 * x ^ 2 + 3 * x - 1

theorem problem_statement : f (f 3) = 1429 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1179_117917


namespace NUMINAMATH_GPT_arnold_protein_intake_l1179_117998

def protein_in_collagen_powder (scoops : ℕ) : ℕ := if scoops = 1 then 9 else 18

def protein_in_protein_powder (scoops : ℕ) : ℕ := 21 * scoops

def protein_in_steak : ℕ := 56

def protein_in_greek_yogurt : ℕ := 15

def protein_in_almonds (cups : ℕ) : ℕ := 6 * cups

theorem arnold_protein_intake :
  protein_in_collagen_powder 1 + 
  protein_in_protein_powder 2 + 
  protein_in_steak + 
  protein_in_greek_yogurt + 
  protein_in_almonds 2 = 134 :=
by
  -- Sorry, the proof is omitted intentionally
  sorry

end NUMINAMATH_GPT_arnold_protein_intake_l1179_117998


namespace NUMINAMATH_GPT_stick_length_l1179_117973

theorem stick_length (x : ℕ) (h1 : 2 * x + (2 * x - 1) = 14) : x = 3 := sorry

end NUMINAMATH_GPT_stick_length_l1179_117973


namespace NUMINAMATH_GPT_three_digit_int_one_less_than_lcm_mult_l1179_117967

theorem three_digit_int_one_less_than_lcm_mult : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ (n + 1) % Nat.lcm (Nat.lcm (Nat.lcm 3 5) 7) 9 = 0 :=
sorry

end NUMINAMATH_GPT_three_digit_int_one_less_than_lcm_mult_l1179_117967


namespace NUMINAMATH_GPT_smallest_X_divisible_15_l1179_117914

theorem smallest_X_divisible_15 (T X : ℕ) 
  (h1 : T > 0) 
  (h2 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) 
  (h3 : T % 15 = 0) 
  (h4 : X = T / 15) : 
  X = 74 :=
sorry

end NUMINAMATH_GPT_smallest_X_divisible_15_l1179_117914


namespace NUMINAMATH_GPT_solve_xyz_eq_x_plus_y_l1179_117919

theorem solve_xyz_eq_x_plus_y (x y z : ℕ) (h1 : x * y * z = x + y) (h2 : x ≤ y) : (x = 2 ∧ y = 2 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 2) :=
by {
    sorry -- The actual proof goes here
}

end NUMINAMATH_GPT_solve_xyz_eq_x_plus_y_l1179_117919


namespace NUMINAMATH_GPT_num_words_with_consonant_l1179_117979

-- Definitions
def letters : List Char := ['A', 'B', 'C', 'D', 'E']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D']

-- Total number of 4-letter words without restrictions
def total_words : Nat := 5 ^ 4

-- Number of 4-letter words with only vowels
def vowels_only_words : Nat := 2 ^ 4

-- Number of 4-letter words with at least one consonant
def words_with_consonant : Nat := total_words - vowels_only_words

theorem num_words_with_consonant : words_with_consonant = 609 := by
  -- Add proof steps
  sorry

end NUMINAMATH_GPT_num_words_with_consonant_l1179_117979


namespace NUMINAMATH_GPT_richard_cleans_in_45_minutes_l1179_117904
noncomputable def richard_time (R : ℝ) := 
  let cory_time := R + 3
  let blake_time := (R + 3) - 4
  (R + cory_time + blake_time = 136) -> R = 45

theorem richard_cleans_in_45_minutes : 
  ∃ R : ℝ, richard_time R := 
sorry

end NUMINAMATH_GPT_richard_cleans_in_45_minutes_l1179_117904


namespace NUMINAMATH_GPT_price_reduction_achieves_profit_l1179_117947

theorem price_reduction_achieves_profit :
  ∃ x : ℝ, (40 - x) * (20 + 2 * (x / 4) * 8) = 1200 ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_price_reduction_achieves_profit_l1179_117947


namespace NUMINAMATH_GPT_train_speed_is_5400432_kmh_l1179_117939

noncomputable def train_speed_kmh (time_to_pass_platform : ℝ) (time_to_pass_man : ℝ) (length_platform : ℝ) : ℝ :=
  let speed_m_per_s := length_platform / (time_to_pass_platform - time_to_pass_man)
  speed_m_per_s * 3.6

theorem train_speed_is_5400432_kmh :
  train_speed_kmh 35 20 225.018 = 54.00432 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_5400432_kmh_l1179_117939


namespace NUMINAMATH_GPT_find_divisor_l1179_117910

theorem find_divisor 
  (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  (dividend = 172) → (quotient = 10) → (remainder = 2) → (dividend = (divisor * quotient) + remainder) → divisor = 17 :=
by 
  sorry

end NUMINAMATH_GPT_find_divisor_l1179_117910


namespace NUMINAMATH_GPT_winning_strategy_for_pawns_l1179_117980

def wiit_or_siti_wins (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 3 * k + 2) ∨ (∃ k : ℕ, n ≠ 3 * k + 2)

theorem winning_strategy_for_pawns (n : ℕ) : wiit_or_siti_wins n :=
sorry

end NUMINAMATH_GPT_winning_strategy_for_pawns_l1179_117980


namespace NUMINAMATH_GPT_max_tickets_l1179_117966

theorem max_tickets (cost : ℝ) (budget : ℝ) (max_tickets : ℕ) (h1 : cost = 15.25) (h2 : budget = 200) :
  max_tickets = 13 :=
by
  sorry

end NUMINAMATH_GPT_max_tickets_l1179_117966


namespace NUMINAMATH_GPT_remainder_consec_even_div12_l1179_117953

theorem remainder_consec_even_div12 (n : ℕ) (h: n % 2 = 0)
  (h1: 11234 ≤ n ∧ n + 12 ≥ 11246) : 
  (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) + (n + 12)) % 12 = 6 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_consec_even_div12_l1179_117953


namespace NUMINAMATH_GPT_sara_cakes_sales_l1179_117920

theorem sara_cakes_sales :
  let cakes_per_day := 4
  let days_per_week := 5
  let weeks := 4
  let price_per_cake := 8
  let cakes_per_week := cakes_per_day * days_per_week
  let total_cakes := cakes_per_week * weeks
  let total_money := total_cakes * price_per_cake
  total_money = 640 := 
by
  sorry

end NUMINAMATH_GPT_sara_cakes_sales_l1179_117920


namespace NUMINAMATH_GPT_tan_identity_proof_l1179_117949

theorem tan_identity_proof
  (α β : ℝ)
  (h₁ : Real.tan (α + β) = 3)
  (h₂ : Real.tan (α + π / 4) = -3) :
  Real.tan (β - π / 4) = -3 / 4 := 
sorry

end NUMINAMATH_GPT_tan_identity_proof_l1179_117949


namespace NUMINAMATH_GPT_seeds_per_flowerbed_l1179_117995

theorem seeds_per_flowerbed :
  ∀ (total_seeds flowerbeds seeds_per_bed : ℕ), 
  total_seeds = 32 → 
  flowerbeds = 8 → 
  seeds_per_bed = total_seeds / flowerbeds → 
  seeds_per_bed = 4 :=
  by 
    intros total_seeds flowerbeds seeds_per_bed h_total h_flowerbeds h_calc
    rw [h_total, h_flowerbeds] at h_calc
    exact h_calc

end NUMINAMATH_GPT_seeds_per_flowerbed_l1179_117995


namespace NUMINAMATH_GPT_wood_rope_equations_l1179_117984

theorem wood_rope_equations (x y : ℝ) (h1 : y - x = 4.5) (h2 : 0.5 * y = x - 1) :
  (y - x = 4.5) ∧ (0.5 * y = x - 1) :=
by
  sorry

end NUMINAMATH_GPT_wood_rope_equations_l1179_117984


namespace NUMINAMATH_GPT_max_f_l1179_117994

noncomputable def f (θ : ℝ) : ℝ :=
  Real.cos (θ / 2) * (1 + Real.sin θ)

theorem max_f : ∀ (θ : ℝ), 0 < θ ∧ θ < π → f θ ≤ (4 * Real.sqrt 3) / 9 :=
by
  sorry

end NUMINAMATH_GPT_max_f_l1179_117994


namespace NUMINAMATH_GPT_g_inv_g_inv_14_l1179_117900

noncomputable def g (x : ℝ) := 3 * x - 4

noncomputable def g_inv (x : ℝ) := (x + 4) / 3

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by sorry

end NUMINAMATH_GPT_g_inv_g_inv_14_l1179_117900


namespace NUMINAMATH_GPT_problem_l1179_117931

noncomputable def a_seq (n : ℕ) : ℚ := sorry

def is_geometric_sequence (seq : ℕ → ℚ) (q : ℚ) : Prop :=
  ∀ n : ℕ, seq (n + 1) = q * seq n

theorem problem (h_positive : ∀ n : ℕ, 0 < a_seq n)
                (h_ratio : ∀ n : ℕ, 2 * a_seq n = 3 * a_seq (n + 1))
                (h_product : a_seq 1 * a_seq 4 = 8 / 27) :
  is_geometric_sequence a_seq (2 / 3) ∧ 
  (∃ n : ℕ, a_seq n = 16 / 81 ∧ n = 6) :=
by
  sorry

end NUMINAMATH_GPT_problem_l1179_117931


namespace NUMINAMATH_GPT_min_cos_C_l1179_117992

theorem min_cos_C (A B C : ℝ) (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
  (h1 : (1 / Real.sin A) + (2 / Real.sin B) = 3 * ((1 / Real.tan A) + (1 / Real.tan B))) :
  Real.cos C ≥ (2 * Real.sqrt 10 - 2) / 9 := 
sorry

end NUMINAMATH_GPT_min_cos_C_l1179_117992


namespace NUMINAMATH_GPT_triangle_arithmetic_progression_l1179_117921

theorem triangle_arithmetic_progression (a d : ℝ) 
(h1 : (a-2*d)^2 + a^2 = (a+2*d)^2) 
(h2 : ∃ x : ℝ, (a = x * d) ∨ (d = x * a))
: (6 ∣ 6*d) ∧ (12 ∣ 6*d) ∧ (18 ∣ 6*d) ∧ (24 ∣ 6*d) ∧ (30 ∣ 6*d)
:= by
  sorry

end NUMINAMATH_GPT_triangle_arithmetic_progression_l1179_117921


namespace NUMINAMATH_GPT_range_of_z_l1179_117954

theorem range_of_z (x y : ℝ) (h : x^2 + 2 * x * y + 4 * y^2 = 6) :
  4 ≤ x^2 + 4 * y^2 ∧ x^2 + 4 * y^2 ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_range_of_z_l1179_117954


namespace NUMINAMATH_GPT_smallest_other_number_l1179_117943

theorem smallest_other_number (x : ℕ)  (h_pos : 0 < x) (n : ℕ)
  (h_gcd : Nat.gcd 60 n = x + 3)
  (h_lcm : Nat.lcm 60 n = x * (x + 3)) :
  n = 45 :=
sorry

end NUMINAMATH_GPT_smallest_other_number_l1179_117943


namespace NUMINAMATH_GPT_predicted_sales_volume_l1179_117962

-- Define the linear regression equation
def regression_equation (x : ℝ) : ℝ := 2 * x + 60

-- Use the given condition x = 34
def temperature_value : ℝ := 34

-- State the theorem that the predicted sales volume is 128
theorem predicted_sales_volume : regression_equation temperature_value = 128 :=
by
  sorry

end NUMINAMATH_GPT_predicted_sales_volume_l1179_117962


namespace NUMINAMATH_GPT_f_inequality_solution_set_l1179_117964

noncomputable
def f : ℝ → ℝ := sorry

axiom f_at_1 : f 1 = 1
axiom f_deriv : ∀ x : ℝ, deriv f x < 1/3

theorem f_inequality_solution_set :
  {x : ℝ | f (x^2) > (x^2 / 3) + 2 / 3} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_f_inequality_solution_set_l1179_117964


namespace NUMINAMATH_GPT_pulley_weight_l1179_117944

theorem pulley_weight (M g : ℝ) (hM_pos : 0 < M) (F : ℝ := 50) :
  (g ≠ 0) → (M * g = 100) :=
by
  sorry

end NUMINAMATH_GPT_pulley_weight_l1179_117944


namespace NUMINAMATH_GPT_intersection_M_N_l1179_117956

/-- Define the set M as pairs (x, y) such that x + y = 2. -/
def M : Set (ℝ × ℝ) := { p | p.1 + p.2 = 2 }

/-- Define the set N as pairs (x, y) such that x - y = 2. -/
def N : Set (ℝ × ℝ) := { p | p.1 - p.2 = 2 }

/-- The intersection of sets M and N is the single point (2, 0). -/
theorem intersection_M_N : M ∩ N = { (2, 0) } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1179_117956


namespace NUMINAMATH_GPT_profit_percentage_is_25_l1179_117963

variable (CP MP : ℝ) (d : ℝ)

/-- Given an article with a cost price of Rs. 85.5, a marked price of Rs. 112.5, 
    and a 5% discount on the marked price, the profit percentage on the cost 
    price is 25%. -/
theorem profit_percentage_is_25
  (hCP : CP = 85.5)
  (hMP : MP = 112.5)
  (hd : d = 0.05) :
  ((MP - (MP * d) - CP) / CP * 100) = 25 := 
sorry

end NUMINAMATH_GPT_profit_percentage_is_25_l1179_117963


namespace NUMINAMATH_GPT_sadies_average_speed_l1179_117906

def sadie_time : ℝ := 2
def ariana_speed : ℝ := 6
def ariana_time : ℝ := 0.5
def sarah_speed : ℝ := 4
def total_time : ℝ := 4.5
def total_distance : ℝ := 17

theorem sadies_average_speed :
  ((total_distance - ((ariana_speed * ariana_time) + (sarah_speed * (total_time - sadie_time - ariana_time)))) / sadie_time) = 3 := 
by sorry

end NUMINAMATH_GPT_sadies_average_speed_l1179_117906


namespace NUMINAMATH_GPT_geometric_arithmetic_sequence_sum_l1179_117988

theorem geometric_arithmetic_sequence_sum {a b : ℕ → ℝ} (q : ℝ) (n : ℕ) 
(h1 : a 2 = 2)
(h2 : a 2 = 2)
(h3 : 2 * (a 3 + 1) = a 2 + a 4)
(h4 : ∀ (n : ℕ), (a (n + 1)) = a 0 * q ^ (n + 1))
(h5 : b n = n * (n + 1)) :
a 8 + (b 8 - b 7) = 144 :=
by { sorry }

end NUMINAMATH_GPT_geometric_arithmetic_sequence_sum_l1179_117988


namespace NUMINAMATH_GPT_probability_of_drawing_two_white_balls_l1179_117933

-- Define the total number of balls and their colors
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def total_balls : ℕ := red_balls + white_balls

-- Define the total number of ways to draw 2 balls from 4
def total_draw_ways : ℕ := (total_balls.choose 2)

-- Define the number of ways to draw 2 white balls
def white_draw_ways : ℕ := (white_balls.choose 2)

-- Define the probability of drawing 2 white balls
def probability_white_draw : ℚ := white_draw_ways / total_draw_ways

-- The main theorem statement to prove
theorem probability_of_drawing_two_white_balls :
  probability_white_draw = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_two_white_balls_l1179_117933


namespace NUMINAMATH_GPT_sin_neg_765_eq_neg_sqrt2_div_2_l1179_117912

theorem sin_neg_765_eq_neg_sqrt2_div_2 :
  Real.sin (-765 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_neg_765_eq_neg_sqrt2_div_2_l1179_117912


namespace NUMINAMATH_GPT_fraction_simplified_l1179_117959

-- Define the fraction function
def fraction (n : ℕ) := (21 * n + 4, 14 * n + 3)

-- Define the gcd function to check if fractions are simplified.
def is_simplified (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Main theorem
theorem fraction_simplified (n : ℕ) : is_simplified (21 * n + 4) (14 * n + 3) :=
by
  -- Rest of the proof
  sorry

end NUMINAMATH_GPT_fraction_simplified_l1179_117959


namespace NUMINAMATH_GPT_incorrect_calculation_d_l1179_117999

theorem incorrect_calculation_d : (1 / 3) / (-1) ≠ 3 * (-1) := 
by {
  -- we'll leave the body of the proof as sorry.
  sorry
}

end NUMINAMATH_GPT_incorrect_calculation_d_l1179_117999


namespace NUMINAMATH_GPT_pleasant_goat_paths_l1179_117907

-- Define the grid points A, B, and C
structure Point :=
  (x : ℕ)
  (y : ℕ)

def A : Point := { x := 0, y := 0 }
def C : Point := { x := 3, y := 3 }  -- assuming some grid layout
def B : Point := { x := 1, y := 1 }

-- Define a statement to count the number of shortest paths
def shortest_paths_count (A B C : Point) : ℕ := sorry

-- Proving the shortest paths from A to C avoiding B is 81
theorem pleasant_goat_paths : shortest_paths_count A B C = 81 := 
sorry

end NUMINAMATH_GPT_pleasant_goat_paths_l1179_117907


namespace NUMINAMATH_GPT_determine_polynomial_l1179_117938

theorem determine_polynomial (p : ℝ → ℝ) (h : ∀ x : ℝ, 1 + p x = (p (x - 1) + p (x + 1)) / 2) :
  ∃ b c : ℝ, ∀ x : ℝ, p x = x^2 + b * x + c := by
  sorry

end NUMINAMATH_GPT_determine_polynomial_l1179_117938


namespace NUMINAMATH_GPT_sum_diff_l1179_117955

-- Define the lengths of the ropes
def shortest_rope_length := 80
def ratio_shortest := 4
def ratio_middle := 5
def ratio_longest := 6

-- Use the given ratio to find the common multiple x.
def x := shortest_rope_length / ratio_shortest

-- Find the lengths of the other ropes
def middle_rope_length := ratio_middle * x
def longest_rope_length := ratio_longest * x

-- Define the sum of the longest and shortest ropes
def sum_of_longest_and_shortest := longest_rope_length + shortest_rope_length

-- Define the difference between the sum of the longest and shortest rope and the middle rope
def difference := sum_of_longest_and_shortest - middle_rope_length

-- Theorem statement
theorem sum_diff : difference = 100 := by
  sorry

end NUMINAMATH_GPT_sum_diff_l1179_117955


namespace NUMINAMATH_GPT_pen_price_l1179_117965

theorem pen_price (x y : ℝ) (h1 : 2 * x + 3 * y = 49) (h2 : 3 * x + y = 49) : x = 14 :=
by
  -- Proof required here
  sorry

end NUMINAMATH_GPT_pen_price_l1179_117965


namespace NUMINAMATH_GPT_arith_expression_evaluation_l1179_117993

theorem arith_expression_evaluation :
  2 + (1/6:ℚ) + (((4.32:ℚ) - 1.68 - (1 + 8/25:ℚ)) * (5/11:ℚ) - (2/7:ℚ)) / (1 + 9/35:ℚ) = 2 + 101/210 := by
  sorry

end NUMINAMATH_GPT_arith_expression_evaluation_l1179_117993


namespace NUMINAMATH_GPT_sum_of_interior_angles_l1179_117989

def f (n : ℕ) : ℚ := (n - 2) * 180

theorem sum_of_interior_angles (n : ℕ) : f (n + 1) = f n + 180 :=
by
  unfold f
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_l1179_117989


namespace NUMINAMATH_GPT_fraction_of_girls_is_half_l1179_117903

variables (T G B : ℝ)
def fraction_x_of_girls (x : ℝ) : Prop :=
  x * G = (1/5) * T ∧ B / G = 1.5 ∧ T = B + G

theorem fraction_of_girls_is_half (x : ℝ) (h : fraction_x_of_girls T G B x) : x = 0.5 :=
sorry

end NUMINAMATH_GPT_fraction_of_girls_is_half_l1179_117903


namespace NUMINAMATH_GPT_maria_paid_9_l1179_117918

-- Define the conditions as variables/constants
def regular_price : ℝ := 15
def discount_rate : ℝ := 0.40

-- Calculate the discount amount
def discount_amount := discount_rate * regular_price

-- Calculate the final price after discount
def final_price := regular_price - discount_amount

-- The goal is to show that the final price is equal to 9
theorem maria_paid_9 : final_price = 9 := 
by
  -- put your proof here
  sorry

end NUMINAMATH_GPT_maria_paid_9_l1179_117918


namespace NUMINAMATH_GPT_marta_candies_received_l1179_117908

theorem marta_candies_received:
  ∃ x y : ℕ, x + y = 200 ∧ x < 100 ∧ x > (4 * y) / 5 ∧ (x % 8 = 0) ∧ (y % 8 = 0) ∧ x = 96 ∧ y = 104 := 
sorry

end NUMINAMATH_GPT_marta_candies_received_l1179_117908


namespace NUMINAMATH_GPT_sqrt_x_minus_2_meaningful_in_reals_l1179_117945

theorem sqrt_x_minus_2_meaningful_in_reals (x : ℝ) : (∃ (y : ℝ), y * y = x - 2) → x ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_x_minus_2_meaningful_in_reals_l1179_117945


namespace NUMINAMATH_GPT_parabola_focus_distance_l1179_117976

-- defining the problem in Lean
theorem parabola_focus_distance
  (A : ℝ × ℝ)
  (hA : A.2^2 = 4 * A.1)
  (h_distance : |A.1| = 3)
  (F : ℝ × ℝ)
  (hF : F = (1, 0)) :
  |(A.1 - F.1)^2 + (A.2 - F.2)^2| = 4 := 
sorry

end NUMINAMATH_GPT_parabola_focus_distance_l1179_117976


namespace NUMINAMATH_GPT_find_integer_n_l1179_117942

open Int

theorem find_integer_n (n a b : ℤ) :
  (4 * n + 1 = a^2) ∧ (9 * n + 1 = b^2) → n = 0 := by
sorry

end NUMINAMATH_GPT_find_integer_n_l1179_117942


namespace NUMINAMATH_GPT_female_democrats_count_l1179_117981

theorem female_democrats_count 
  (F M : ℕ) 
  (total_participants : F + M = 750)
  (female_democrats : ℕ := F / 2) 
  (male_democrats : ℕ := M / 4)
  (total_democrats : female_democrats + male_democrats = 250) :
  female_democrats = 125 := 
sorry

end NUMINAMATH_GPT_female_democrats_count_l1179_117981


namespace NUMINAMATH_GPT_gym_distance_l1179_117909

def distance_to_work : ℕ := 10
def distance_to_gym (dist : ℕ) : ℕ := (dist / 2) + 2

theorem gym_distance :
  distance_to_gym distance_to_work = 7 :=
sorry

end NUMINAMATH_GPT_gym_distance_l1179_117909


namespace NUMINAMATH_GPT_factorization_identity_l1179_117983

theorem factorization_identity (a : ℝ) : (a + 3) * (a - 7) + 25 = (a - 2) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_identity_l1179_117983


namespace NUMINAMATH_GPT_largest_square_area_with_4_interior_lattice_points_l1179_117968

/-- 
A point (x, y) in the plane is called a lattice point if both x and y are integers.
The largest square that contains exactly four lattice points solely in its interior
has an area of 9.
-/
theorem largest_square_area_with_4_interior_lattice_points : 
  ∃ s : ℝ, ∀ (x y : ℤ), 
  (1 ≤ x ∧ x < s ∧ 1 ≤ y ∧ y < s) → s^2 = 9 := 
sorry

end NUMINAMATH_GPT_largest_square_area_with_4_interior_lattice_points_l1179_117968


namespace NUMINAMATH_GPT_inequality_with_equality_condition_l1179_117972

variable {a b c d : ℝ}

theorem inequality_with_equality_condition (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : 
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧ 
  ((a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d) := sorry

end NUMINAMATH_GPT_inequality_with_equality_condition_l1179_117972


namespace NUMINAMATH_GPT_jellybean_mass_l1179_117901

noncomputable def cost_per_gram : ℚ := 7.50 / 250
noncomputable def mass_for_180_cents : ℚ := 1.80 / cost_per_gram

theorem jellybean_mass :
  mass_for_180_cents = 60 := 
  sorry

end NUMINAMATH_GPT_jellybean_mass_l1179_117901


namespace NUMINAMATH_GPT_larger_of_two_numbers_l1179_117935

-- Define necessary conditions
def hcf : ℕ := 23
def factor1 : ℕ := 11
def factor2 : ℕ := 12
def lcm : ℕ := hcf * factor1 * factor2

-- Define the problem statement in Lean
theorem larger_of_two_numbers : ∃ (a b : ℕ), a = hcf * factor1 ∧ b = hcf * factor2 ∧ max a b = 276 := by
  sorry

end NUMINAMATH_GPT_larger_of_two_numbers_l1179_117935


namespace NUMINAMATH_GPT_smallest_natural_greater_than_12_l1179_117936

def smallest_greater_than (n : ℕ) : ℕ := n + 1

theorem smallest_natural_greater_than_12 : smallest_greater_than 12 = 13 :=
by
  sorry

end NUMINAMATH_GPT_smallest_natural_greater_than_12_l1179_117936


namespace NUMINAMATH_GPT_cost_of_book_sold_at_loss_l1179_117969

theorem cost_of_book_sold_at_loss:
  ∃ (C1 C2 : ℝ), 
    C1 + C2 = 490 ∧ 
    C1 * 0.85 = C2 * 1.19 ∧ 
    C1 = 285.93 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_book_sold_at_loss_l1179_117969


namespace NUMINAMATH_GPT_bisection_next_interval_l1179_117934

def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_next_interval (h₀ : f 2.5 > 0) (h₁ : f 2 < 0) :
  ∃ a b, (2 < 2.5) ∧ f 2 < 0 ∧ f 2.5 > 0 ∧ a = 2 ∧ b = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_bisection_next_interval_l1179_117934


namespace NUMINAMATH_GPT_impossible_configuration_l1179_117991

theorem impossible_configuration : 
  ¬∃ (f : ℕ → ℕ) (h : ∀n, 1 ≤ f n ∧ f n ≤ 5) (perm : ∀i j, if i < j then f i ≠ f j else true), 
  (f 0 = 3) ∧ (f 1 = 4) ∧ (f 2 = 2) ∧ (f 3 = 1) ∧ (f 4 = 5) :=
sorry

end NUMINAMATH_GPT_impossible_configuration_l1179_117991


namespace NUMINAMATH_GPT_solve_absolute_value_eq_l1179_117927

theorem solve_absolute_value_eq (x : ℝ) : |x - 5| = 3 * x - 2 ↔ x = 7 / 4 :=
sorry

end NUMINAMATH_GPT_solve_absolute_value_eq_l1179_117927


namespace NUMINAMATH_GPT_sum_first_11_terms_eq_99_l1179_117985

variable {a_n : ℕ → ℝ} -- assuming the sequence values are real numbers
variable (S : ℕ → ℝ) -- sum of the first n terms
variable (a₃ a₆ a₉ : ℝ)
variable (h_sequence : ∀ n, a_n n = aₙ 1 + (n - 1) * (a_n 2 - aₙ 1)) -- sequence is arithmetic
variable (h_condition : a₃ + a₉ = 27 - a₆) -- given condition

theorem sum_first_11_terms_eq_99 
  (h_a₃ : a₃ = a_n 3) 
  (h_a₆ : a₆ = a_n 6) 
  (h_a₉ : a₉ = a_n 9) 
  (h_S : S 11 = 11 * a₆) : 
  S 11 = 99 := 
by 
  sorry


end NUMINAMATH_GPT_sum_first_11_terms_eq_99_l1179_117985


namespace NUMINAMATH_GPT_count_three_digit_numbers_using_1_and_2_l1179_117926

theorem count_three_digit_numbers_using_1_and_2 : 
  let n := 3
  let d := [1, 2]
  let total_combinations := (List.length d)^n
  let invalid_combinations := 2
  total_combinations - invalid_combinations = 6 :=
by
  let n := 3
  let d := [1, 2]
  let total_combinations := (List.length d)^n
  let invalid_combinations := 2
  show total_combinations - invalid_combinations = 6
  sorry

end NUMINAMATH_GPT_count_three_digit_numbers_using_1_and_2_l1179_117926


namespace NUMINAMATH_GPT_expected_yolks_correct_l1179_117974

-- Define the conditions
def total_eggs : ℕ := 15
def double_yolk_eggs : ℕ := 5
def triple_yolk_eggs : ℕ := 3
def single_yolk_eggs : ℕ := total_eggs - double_yolk_eggs - triple_yolk_eggs
def extra_yolk_prob : ℝ := 0.10

-- Define the expected number of yolks calculation
noncomputable def expected_yolks : ℝ :=
  (single_yolk_eggs * 1) + 
  (double_yolk_eggs * 2) + 
  (triple_yolk_eggs * 3) + 
  (double_yolk_eggs * extra_yolk_prob) + 
  (triple_yolk_eggs * extra_yolk_prob)

-- State that the expected number of total yolks is 26.8
theorem expected_yolks_correct : expected_yolks = 26.8 := by
  -- solution would go here
  sorry

end NUMINAMATH_GPT_expected_yolks_correct_l1179_117974


namespace NUMINAMATH_GPT_cookies_with_five_cups_l1179_117977

-- Define the initial condition: Lee can make 24 cookies with 3 cups of flour
def cookies_per_cup := 24 / 3

-- Theorem stating Lee can make 40 cookies with 5 cups of flour
theorem cookies_with_five_cups : 5 * cookies_per_cup = 40 :=
by
  sorry

end NUMINAMATH_GPT_cookies_with_five_cups_l1179_117977


namespace NUMINAMATH_GPT_major_axis_of_ellipse_l1179_117996

structure Ellipse :=
(center : ℝ × ℝ)
(tangent_y_axis : Bool)
(tangent_y_eq_3 : Bool)
(focus_1 : ℝ × ℝ)
(focus_2 : ℝ × ℝ)

noncomputable def major_axis_length (e : Ellipse) : ℝ :=
  2 * (e.focus_1.2 - e.center.2)

theorem major_axis_of_ellipse : 
  ∀ (e : Ellipse), 
    e.center = (3, 0) ∧
    e.tangent_y_axis = true ∧
    e.tangent_y_eq_3 = true ∧
    e.focus_1 = (3, 2 + Real.sqrt 2) ∧
    e.focus_2 = (3, -2 - Real.sqrt 2) →
      major_axis_length e = 4 + 2 * Real.sqrt 2 :=
by
  intro e
  intro h
  sorry

end NUMINAMATH_GPT_major_axis_of_ellipse_l1179_117996


namespace NUMINAMATH_GPT_expression_evaluation_l1179_117960

theorem expression_evaluation : 4 * 10 + 5 * 11 + 12 * 4 + 4 * 9 = 179 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1179_117960


namespace NUMINAMATH_GPT_prove_additional_minutes_needed_l1179_117925

-- Assume the given conditions as definitions in Lean 4
def number_of_classmates := 30
def initial_gathering_time := 120   -- in minutes (2 hours)
def time_per_flower := 10           -- in minutes
def flowers_lost := 3

-- Calculate the flowers gathered initially
def initial_flowers_gathered := initial_gathering_time / time_per_flower

-- Calculate flowers remaining after loss
def flowers_remaining := initial_flowers_gathered - flowers_lost

-- Calculate additional flowers needed
def additional_flowers_needed := number_of_classmates - flowers_remaining

-- Therefore, calculate the additional minutes required to gather the remaining flowers
def additional_minutes_needed := additional_flowers_needed * time_per_flower

theorem prove_additional_minutes_needed :
  additional_minutes_needed = 210 :=
by 
  unfold additional_minutes_needed additional_flowers_needed flowers_remaining initial_flowers_gathered
  sorry

end NUMINAMATH_GPT_prove_additional_minutes_needed_l1179_117925


namespace NUMINAMATH_GPT_winner_C_l1179_117990

noncomputable def votes_A : ℕ := 4500
noncomputable def votes_B : ℕ := 7000
noncomputable def votes_C : ℕ := 12000
noncomputable def votes_D : ℕ := 8500
noncomputable def votes_E : ℕ := 3500

noncomputable def total_votes : ℕ := votes_A + votes_B + votes_C + votes_D + votes_E

noncomputable def percentage (votes : ℕ) : ℚ :=
   (votes : ℚ) / (total_votes : ℚ) * 100

noncomputable def percentage_A : ℚ := percentage votes_A
noncomputable def percentage_B : ℚ := percentage votes_B
noncomputable def percentage_C : ℚ := percentage votes_C
noncomputable def percentage_D : ℚ := percentage votes_D
noncomputable def percentage_E : ℚ := percentage votes_E

theorem winner_C : (percentage_C = 33.803) := 
sorry

end NUMINAMATH_GPT_winner_C_l1179_117990


namespace NUMINAMATH_GPT_number_of_words_with_at_least_one_consonant_l1179_117971

def total_5_letter_words : ℕ := 6 ^ 5

def total_5_letter_vowel_words : ℕ := 2 ^ 5

def total_5_letter_words_with_consonant : ℕ := total_5_letter_words - total_5_letter_vowel_words

theorem number_of_words_with_at_least_one_consonant :
  total_5_letter_words_with_consonant = 7744 :=
  by
    -- We assert the calculation follows correctly:
    -- total_5_letter_words == 6^5 = 7776
    -- total_5_letter_vowel_words == 2^5 = 32
    -- 7776 - 32 == 7744
    sorry

end NUMINAMATH_GPT_number_of_words_with_at_least_one_consonant_l1179_117971


namespace NUMINAMATH_GPT_total_weight_of_hay_bales_l1179_117930

theorem total_weight_of_hay_bales
  (initial_bales : Nat) (weight_per_initial_bale : Nat)
  (total_bales_now : Nat) (weight_per_new_bale : Nat) : 
  (initial_bales = 73 ∧ weight_per_initial_bale = 45 ∧ 
   total_bales_now = 96 ∧ weight_per_new_bale = 50) →
  (73 * 45 + (96 - 73) * 50 = 4435) :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_hay_bales_l1179_117930


namespace NUMINAMATH_GPT_inequality_xyz_l1179_117982

theorem inequality_xyz (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) : 
    x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := 
    sorry

end NUMINAMATH_GPT_inequality_xyz_l1179_117982


namespace NUMINAMATH_GPT_circle_polar_equation_l1179_117950

-- Definitions and conditions
def circle_equation_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * y = 0

def polar_coordinates (ρ θ : ℝ) (x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem to be proven
theorem circle_polar_equation (ρ θ : ℝ) :
  (∀ x y : ℝ, circle_equation_cartesian x y → 
  polar_coordinates ρ θ x y) → ρ = 2 * Real.sin θ :=
by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_circle_polar_equation_l1179_117950


namespace NUMINAMATH_GPT_B_is_not_15_percent_less_than_A_l1179_117978

noncomputable def A (B : ℝ) : ℝ := 1.15 * B

theorem B_is_not_15_percent_less_than_A (B : ℝ) (h : B > 0) : A B ≠ 0.85 * (A B) :=
by
  unfold A
  suffices 1.15 * B ≠ 0.85 * (1.15 * B) by
    intro h1
    exact this h1
  sorry

end NUMINAMATH_GPT_B_is_not_15_percent_less_than_A_l1179_117978


namespace NUMINAMATH_GPT_required_total_money_l1179_117928

def bundle_count := 100
def number_of_bundles := 10
def bill_5_value := 5
def bill_10_value := 10
def bill_20_value := 20

-- Sum up the total money required to fill the machine
theorem required_total_money : 
  (bundle_count * bill_5_value * number_of_bundles) + 
  (bundle_count * bill_10_value * number_of_bundles) + 
  (bundle_count * bill_20_value * number_of_bundles) = 35000 := 
by 
  sorry

end NUMINAMATH_GPT_required_total_money_l1179_117928


namespace NUMINAMATH_GPT_tan_double_angle_identity_l1179_117911

theorem tan_double_angle_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) : 
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := 
  sorry

end NUMINAMATH_GPT_tan_double_angle_identity_l1179_117911


namespace NUMINAMATH_GPT_relationship_a_b_l1179_117923

-- Definitions of the two quadratic equations having a single common root
def has_common_root (a b : ℝ) : Prop :=
  ∃ t : ℝ, (t^2 + a * t + b = 0) ∧ (t^2 + b * t + a = 0)

-- Theorem stating the relationship between a and b
theorem relationship_a_b (a b : ℝ) (h : has_common_root a b) : a ≠ b → a + b + 1 = 0 :=
by sorry

end NUMINAMATH_GPT_relationship_a_b_l1179_117923


namespace NUMINAMATH_GPT_isosceles_triangle_EF_length_l1179_117970

theorem isosceles_triangle_EF_length (DE DF EF DK EK KF : ℝ)
  (h1 : DE = 5) (h2 : DF = 5) (h3 : DK^2 + EK^2 = DE^2) (h4 : DK^2 + KF^2 = EF^2)
  (h5 : EK + KF = EF) (h6 : EK = 4 * KF) :
  EF = Real.sqrt 10 :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_EF_length_l1179_117970


namespace NUMINAMATH_GPT_F_transformed_l1179_117915

-- Define the coordinates of point F
def F : ℝ × ℝ := (1, 0)

-- Reflection over the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Reflection over the y-axis
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

-- Reflection over the line y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

-- Point F after all transformations
def F_final : ℝ × ℝ :=
  reflect_y_eq_x (reflect_y (reflect_x F))

-- Statement to prove
theorem F_transformed : F_final = (0, -1) :=
  sorry

end NUMINAMATH_GPT_F_transformed_l1179_117915


namespace NUMINAMATH_GPT_find_x_l1179_117937

variable {a b x : ℝ}
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(2 * b) = a^b * x^b)

theorem find_x (h₀ : b ≠ 0) (h₁ : (3 * a)^(2 * b) = a^b * x^b) : x = 9 * a :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1179_117937


namespace NUMINAMATH_GPT_jacket_total_price_correct_l1179_117958

/-- The original price of the jacket -/
def original_price : ℝ := 120

/-- The initial discount rate -/
def initial_discount_rate : ℝ := 0.15

/-- The additional discount in dollars -/
def additional_discount : ℝ := 10

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.10

/-- The calculated total amount the shopper pays for the jacket including all discounts and tax -/
def total_amount_paid : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_rate)
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  price_after_additional_discount * (1 + sales_tax_rate)

theorem jacket_total_price_correct : total_amount_paid = 101.20 :=
  sorry

end NUMINAMATH_GPT_jacket_total_price_correct_l1179_117958


namespace NUMINAMATH_GPT_no_solution_implies_b_positive_l1179_117913

theorem no_solution_implies_b_positive (a b : ℝ) :
  (¬ ∃ x y : ℝ, y = x^2 + a * x + b ∧ x = y^2 + a * y + b) → b > 0 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_implies_b_positive_l1179_117913
