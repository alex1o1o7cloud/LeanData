import Mathlib

namespace NUMINAMATH_GPT_heidi_zoe_paint_wall_l1337_133793

theorem heidi_zoe_paint_wall :
  let heidi_rate := (1 : ℚ) / 60
  let zoe_rate := (1 : ℚ) / 90
  let combined_rate := heidi_rate + zoe_rate
  let painted_fraction_15_minutes := combined_rate * 15
  painted_fraction_15_minutes = (5 : ℚ) / 12 :=
by
  sorry

end NUMINAMATH_GPT_heidi_zoe_paint_wall_l1337_133793


namespace NUMINAMATH_GPT_amount_r_has_l1337_133791

variable (p q r : ℕ)
variable (total_amount : ℕ)
variable (two_thirdsOf_pq : ℕ)

def total_money : Prop := (p + q + r = 4000)
def two_thirds_of_pq : Prop := (r = 2 * (p + q) / 3)

theorem amount_r_has : total_money p q r → two_thirds_of_pq p q r → r = 1600 := by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_amount_r_has_l1337_133791


namespace NUMINAMATH_GPT_find_g_of_3_l1337_133759

theorem find_g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 2 * g x - 5 * g (1 / x) = 2 * x) : g 3 = -32 / 63 :=
by sorry

end NUMINAMATH_GPT_find_g_of_3_l1337_133759


namespace NUMINAMATH_GPT_find_m_given_solution_l1337_133743

theorem find_m_given_solution (m x y : ℚ) (h₁ : x = 4) (h₂ : y = 3) (h₃ : m * x - y = 4) : m = 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_given_solution_l1337_133743


namespace NUMINAMATH_GPT_mike_spent_on_speakers_l1337_133746

-- Definitions of the conditions:
def total_car_parts_cost : ℝ := 224.87
def new_tires_cost : ℝ := 106.33

-- Statement of the proof problem:
theorem mike_spent_on_speakers : total_car_parts_cost - new_tires_cost = 118.54 :=
by
  sorry

end NUMINAMATH_GPT_mike_spent_on_speakers_l1337_133746


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l1337_133726

theorem algebraic_expression_evaluation (x m : ℝ) (h1 : 5 * (2 - 1) + 3 * m * 2 = -7) (h2 : m = -2) :
  5 * (x - 1) + 3 * m * x = -1 ↔ x = -4 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l1337_133726


namespace NUMINAMATH_GPT_find_principal_l1337_133796

-- Define the conditions
def simple_interest (P R T : ℕ) : ℕ :=
  (P * R * T) / 100

-- Given values
def SI : ℕ := 750
def R : ℕ := 6
def T : ℕ := 5

-- Proof statement
theorem find_principal : ∃ P : ℕ, simple_interest P R T = SI ∧ P = 2500 := by
  aesop

end NUMINAMATH_GPT_find_principal_l1337_133796


namespace NUMINAMATH_GPT_gray_eyed_brunettes_l1337_133786

-- Given conditions
def total_students : ℕ := 60
def brunettes : ℕ := 35
def green_eyed_blondes : ℕ := 20
def gray_eyed_total : ℕ := 25

-- Conclude that the number of gray-eyed brunettes is 20
theorem gray_eyed_brunettes :
    (gray_eyed_total - (total_students - brunettes - green_eyed_blondes)) = 20 := by
    sorry

end NUMINAMATH_GPT_gray_eyed_brunettes_l1337_133786


namespace NUMINAMATH_GPT_frosting_cupcakes_l1337_133774

noncomputable def Cagney_rate := 1 / 20 -- cupcakes per second
noncomputable def Lacey_rate := 1 / 30 -- cupcakes per second
noncomputable def Hardy_rate := 1 / 40 -- cupcakes per second

noncomputable def combined_rate := Cagney_rate + Lacey_rate + Hardy_rate
noncomputable def total_time := 600 -- seconds (10 minutes)

theorem frosting_cupcakes :
  total_time * combined_rate = 65 := 
by 
  sorry

end NUMINAMATH_GPT_frosting_cupcakes_l1337_133774


namespace NUMINAMATH_GPT_x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2_l1337_133766

theorem x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2 
  (x : ℤ) (p m n : ℕ) (hp : 0 < p) (hm : 0 < m) (hn : 0 < n) :
  (x^2 + x + 1) ∣ (x^(3 * p) + x^(3 * m + 1) + x^(3 * n + 2)) :=
by
  sorry

end NUMINAMATH_GPT_x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2_l1337_133766


namespace NUMINAMATH_GPT_distance_of_each_race_l1337_133729

theorem distance_of_each_race (d : ℝ) : 
  (∃ (d : ℝ), 
    let lake_speed := 3 
    let ocean_speed := 2.5 
    let num_races := 10 
    let total_time := 11
    let num_lake_races := num_races / 2
    let num_ocean_races := num_races / 2
    (num_lake_races * (d / lake_speed) + num_ocean_races * (d / ocean_speed) = total_time)) →
  d = 3 :=
sorry

end NUMINAMATH_GPT_distance_of_each_race_l1337_133729


namespace NUMINAMATH_GPT_tim_total_score_l1337_133711

-- Definitions from conditions
def single_line_points : ℕ := 1000
def tetris_points : ℕ := 8 * single_line_points
def doubled_tetris_points : ℕ := 2 * tetris_points
def num_singles : ℕ := 6
def num_tetrises : ℕ := 4
def consecutive_tetrises : ℕ := 2
def regular_tetrises : ℕ := num_tetrises - consecutive_tetrises

-- Total score calculation
def total_score : ℕ :=
  num_singles * single_line_points +
  regular_tetrises * tetris_points +
  consecutive_tetrises * doubled_tetris_points

-- Prove that Tim's total score is 54000
theorem tim_total_score : total_score = 54000 :=
by 
  sorry

end NUMINAMATH_GPT_tim_total_score_l1337_133711


namespace NUMINAMATH_GPT_total_chocolate_sold_total_vanilla_sold_total_strawberry_sold_l1337_133724

def chocolate_sold : ℕ := 6 + 7 + 4 + 8 + 9 + 10 + 5
def vanilla_sold : ℕ := 4 + 5 + 3 + 7 + 6 + 8 + 4
def strawberry_sold : ℕ := 3 + 2 + 6 + 4 + 5 + 7 + 4

theorem total_chocolate_sold : chocolate_sold = 49 :=
by
  unfold chocolate_sold
  rfl

theorem total_vanilla_sold : vanilla_sold = 37 :=
by
  unfold vanilla_sold
  rfl

theorem total_strawberry_sold : strawberry_sold = 31 :=
by
  unfold strawberry_sold
  rfl

end NUMINAMATH_GPT_total_chocolate_sold_total_vanilla_sold_total_strawberry_sold_l1337_133724


namespace NUMINAMATH_GPT_pencils_placed_by_sara_l1337_133797

theorem pencils_placed_by_sara (initial_pencils final_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : final_pencils = 215) : final_pencils - initial_pencils = 100 := by
  sorry

end NUMINAMATH_GPT_pencils_placed_by_sara_l1337_133797


namespace NUMINAMATH_GPT_abcd_product_l1337_133740

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

axiom a_eq : a = Real.sqrt (4 - Real.sqrt (5 - a))
axiom b_eq : b = Real.sqrt (4 + Real.sqrt (5 - b))
axiom c_eq : c = Real.sqrt (4 - Real.sqrt (5 + c))
axiom d_eq : d = Real.sqrt (4 + Real.sqrt (5 + d))

theorem abcd_product : a * b * c * d = 11 := sorry

end NUMINAMATH_GPT_abcd_product_l1337_133740


namespace NUMINAMATH_GPT_Jesse_pages_left_to_read_l1337_133775

def pages_read := [10, 15, 27, 12, 19]
def total_pages_read := pages_read.sum
def fraction_read : ℚ := 1 / 3
def total_pages : ℚ := total_pages_read / fraction_read
def pages_left_to_read : ℚ := total_pages - total_pages_read

theorem Jesse_pages_left_to_read :
  pages_left_to_read = 166 := by
  sorry

end NUMINAMATH_GPT_Jesse_pages_left_to_read_l1337_133775


namespace NUMINAMATH_GPT_complex_number_solution_l1337_133773

theorem complex_number_solution
  (z : ℂ)
  (h : i * (z - 1) = 1 + i) :
  z = 2 - i :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l1337_133773


namespace NUMINAMATH_GPT_max_a_condition_l1337_133730

theorem max_a_condition (a : ℝ) : 
  (∀ x : ℝ, x < a → x^2 - 2*x - 3 > 0) ∧ (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) → a = -1 :=
by
  sorry

end NUMINAMATH_GPT_max_a_condition_l1337_133730


namespace NUMINAMATH_GPT_evaluation_result_l1337_133710

noncomputable def evaluate_expression : ℝ :=
  let a := 210
  let b := 206
  let numerator := 980 ^ 2
  let denominator := a^2 - b^2
  numerator / denominator

theorem evaluation_result : evaluate_expression = 577.5 := 
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_evaluation_result_l1337_133710


namespace NUMINAMATH_GPT_find_xyz_l1337_133758

variable (x y z : ℝ)

theorem find_xyz (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * (y + z) = 168)
  (h2 : y * (z + x) = 180)
  (h3 : z * (x + y) = 192) : x * y * z = 842 :=
sorry

end NUMINAMATH_GPT_find_xyz_l1337_133758


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_first_three_terms_l1337_133752

theorem arithmetic_sequence_sum_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 4) (h5 : a 5 = 7) (h6 : a 6 = 10) : a 1 + a 2 + a 3 = -6 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_first_three_terms_l1337_133752


namespace NUMINAMATH_GPT_employees_use_public_transportation_l1337_133737

theorem employees_use_public_transportation
    (total_employees : ℕ)
    (drive_percentage : ℝ)
    (public_transportation_fraction : ℝ)
    (h1 : total_employees = 100)
    (h2 : drive_percentage = 0.60)
    (h3 : public_transportation_fraction = 0.50) :
    ((total_employees * (1 - drive_percentage)) * public_transportation_fraction) = 20 :=
by
    sorry

end NUMINAMATH_GPT_employees_use_public_transportation_l1337_133737


namespace NUMINAMATH_GPT_investment_problem_l1337_133765

theorem investment_problem :
  ∃ (S G : ℝ), S + G = 10000 ∧ 0.06 * G = 0.05 * S + 160 ∧ S = 4000 :=
by
  sorry

end NUMINAMATH_GPT_investment_problem_l1337_133765


namespace NUMINAMATH_GPT_Problem_l1337_133755

theorem Problem (N : ℕ) (hn : N = 16) :
  (Nat.choose N 5) = 2002 := 
by 
  rw [hn] 
  sorry

end NUMINAMATH_GPT_Problem_l1337_133755


namespace NUMINAMATH_GPT_diameter_increase_l1337_133781

theorem diameter_increase (π : ℝ) (D : ℝ) (A A' D' : ℝ)
  (hA : A = (π / 4) * D^2)
  (hA' : A' = 4 * A)
  (hA'_def : A' = (π / 4) * D'^2) :
  D' = 2 * D :=
by
  sorry

end NUMINAMATH_GPT_diameter_increase_l1337_133781


namespace NUMINAMATH_GPT_pyramid_volume_pyramid_surface_area_l1337_133716

noncomputable def volume_of_pyramid (l : ℝ) := (l^3 * Real.sqrt 2) / 12

noncomputable def surface_area_of_pyramid (l : ℝ) := (l^2 * (2 + Real.sqrt 2)) / 2

theorem pyramid_volume (l : ℝ) :
  volume_of_pyramid l = (l^3 * Real.sqrt 2) / 12 :=
sorry

theorem pyramid_surface_area (l : ℝ) :
  surface_area_of_pyramid l = (l^2 * (2 + Real.sqrt 2)) / 2 :=
sorry

end NUMINAMATH_GPT_pyramid_volume_pyramid_surface_area_l1337_133716


namespace NUMINAMATH_GPT_arithmetic_sequence_max_value_l1337_133794

theorem arithmetic_sequence_max_value 
  (S : ℕ → ℤ)
  (k : ℕ)
  (h1 : 2 ≤ k)
  (h2 : S (k - 1) = 8)
  (h3 : S k = 0)
  (h4 : S (k + 1) = -10) :
  ∃ n, S n = 20 ∧ (∀ m, S m ≤ 20) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_max_value_l1337_133794


namespace NUMINAMATH_GPT_ab_bc_ca_fraction_l1337_133715

theorem ab_bc_ca_fraction (a b c : ℝ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 10) (h3 : a * b * c = 12) :
    (a * b / c) + (b * c / a) + (c * a / b) = -17 / 3 := 
    sorry

end NUMINAMATH_GPT_ab_bc_ca_fraction_l1337_133715


namespace NUMINAMATH_GPT_find_x_l1337_133762

noncomputable section

open Real

theorem find_x (x : ℝ) (hx : 0 < x ∧ x < 180) : 
  tan (120 * π / 180 - x * π / 180) = (sin (120 * π / 180) - sin (x * π / 180)) / (cos (120 * π / 180) - cos (x * π / 180)) →
  x = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1337_133762


namespace NUMINAMATH_GPT_inheritance_amount_l1337_133761

theorem inheritance_amount (x : ℝ) (hx1 : 0.25 * x + 0.1 * x = 15000) : x = 42857 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_inheritance_amount_l1337_133761


namespace NUMINAMATH_GPT_speed_difference_l1337_133795

theorem speed_difference (distance : ℝ) (time_heavy : ℝ) (time_no_traffic : ℝ) (d : distance = 200) (th : time_heavy = 5) (tn : time_no_traffic = 4) :
  (distance / time_no_traffic) - (distance / time_heavy) = 10 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_speed_difference_l1337_133795


namespace NUMINAMATH_GPT_cube_edge_length_l1337_133771

theorem cube_edge_length (sum_of_edges : ℕ) (num_edges : ℕ) (h : sum_of_edges = 144) (num_edges_h : num_edges = 12) :
  sum_of_edges / num_edges = 12 :=
by
  -- The proof is skipped.
  sorry

end NUMINAMATH_GPT_cube_edge_length_l1337_133771


namespace NUMINAMATH_GPT_find_b_of_quadratic_eq_l1337_133707

theorem find_b_of_quadratic_eq (a b c y1 y2 : ℝ) 
    (h1 : y1 = a * (2:ℝ)^2 + b * (2:ℝ) + c) 
    (h2 : y2 = a * (-2:ℝ)^2 + b * (-2:ℝ) + c) 
    (h_diff : y1 - y2 = 4) : b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_b_of_quadratic_eq_l1337_133707


namespace NUMINAMATH_GPT_value_of_square_l1337_133798

variable (x y : ℝ)

theorem value_of_square (h1 : x * (x + y) = 30) (h2 : y * (x + y) = 60) :
  (x + y) ^ 2 = 90 := sorry

end NUMINAMATH_GPT_value_of_square_l1337_133798


namespace NUMINAMATH_GPT_divisibility_by_seven_l1337_133799

theorem divisibility_by_seven (n : ℤ) (b : ℤ) (a : ℤ) (h : n = 10 * a + b) 
  (hb : 0 ≤ b) (hb9 : b ≤ 9) (ha : 0 ≤ a) (d : ℤ) (hd : d = a - 2 * b) :
  (2 * n + d) % 7 = 0 ↔ n % 7 = 0 := 
by
  sorry

end NUMINAMATH_GPT_divisibility_by_seven_l1337_133799


namespace NUMINAMATH_GPT_solve_eq_l1337_133753

theorem solve_eq {x : ℝ} (h : x + 2 * Real.sqrt x - 8 = 0) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_l1337_133753


namespace NUMINAMATH_GPT_min_m_plus_n_l1337_133722

open Nat

theorem min_m_plus_n (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_eq : 45 * m = n^3) (h_mult_of_five : 5 ∣ n) :
  m + n = 90 :=
sorry

end NUMINAMATH_GPT_min_m_plus_n_l1337_133722


namespace NUMINAMATH_GPT_double_acute_angle_is_positive_and_less_than_180_l1337_133778

variable (α : ℝ) (h : 0 < α ∧ α < π / 2)

theorem double_acute_angle_is_positive_and_less_than_180 :
  0 < 2 * α ∧ 2 * α < π :=
by
  sorry

end NUMINAMATH_GPT_double_acute_angle_is_positive_and_less_than_180_l1337_133778


namespace NUMINAMATH_GPT_solution_set_l1337_133714

-- Defining the system of equations as conditions
def equation1 (x y : ℝ) : Prop := x - 2 * y = 1
def equation2 (x y : ℝ) : Prop := x^3 - 6 * x * y - 8 * y^3 = 1

-- The main theorem
theorem solution_set (x y : ℝ) 
  (h1 : equation1 x y) 
  (h2 : equation2 x y) : 
  y = (x - 1) / 2 :=
sorry

end NUMINAMATH_GPT_solution_set_l1337_133714


namespace NUMINAMATH_GPT_find_m_l1337_133721

variable {m : ℝ}

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)
def vector_diff (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_m (hm: dot_product vector_a (vector_diff vector_a (vector_b m)) = 0) : m = 3 :=
  by
  sorry

end NUMINAMATH_GPT_find_m_l1337_133721


namespace NUMINAMATH_GPT_initial_average_mark_of_class_l1337_133745

theorem initial_average_mark_of_class
  (avg_excluded : ℝ) (n_excluded : ℕ) (avg_remaining : ℝ)
  (n_total : ℕ) : 
  avg_excluded = 70 → 
  n_excluded = 5 → 
  avg_remaining = 90 → 
  n_total = 10 → 
  (10 * (10 / n_total + avg_excluded - avg_remaining) / 10) = 80 :=
by 
  intros 
  sorry

end NUMINAMATH_GPT_initial_average_mark_of_class_l1337_133745


namespace NUMINAMATH_GPT_max_value_trig_formula_l1337_133763

theorem max_value_trig_formula (x : ℝ) : ∃ (M : ℝ), M = 5 ∧ ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ M := 
sorry

end NUMINAMATH_GPT_max_value_trig_formula_l1337_133763


namespace NUMINAMATH_GPT_acute_triangle_on_perpendicular_lines_l1337_133744

theorem acute_triangle_on_perpendicular_lines :
  ∀ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2) →
  ∃ (x y z : ℝ), (x^2 = (b^2 + c^2 - a^2) / 2) ∧ (y^2 = (a^2 + c^2 - b^2) / 2) ∧ (z^2 = (a^2 + b^2 - c^2) / 2) ∧ (x > 0) ∧ (y > 0) ∧ (z > 0) :=
by
  sorry

end NUMINAMATH_GPT_acute_triangle_on_perpendicular_lines_l1337_133744


namespace NUMINAMATH_GPT_area_of_square_l1337_133725

noncomputable def square_area (s : ℝ) : ℝ := s ^ 2

theorem area_of_square
  {E F G H : Type}
  (ABCD : Type)
  (on_segments : E → F → G → H → Prop)
  (EG FH : ℝ)
  (angle_intersection : ℝ)
  (hEG : EG = 7)
  (hFH : FH = 8)
  (hangle : angle_intersection = 30) :
  ∃ s : ℝ, square_area s = 147 / 4 :=
sorry

end NUMINAMATH_GPT_area_of_square_l1337_133725


namespace NUMINAMATH_GPT_find_value_l1337_133769

theorem find_value (x : ℝ) (h : 0.20 * x = 80) : 0.40 * x = 160 := 
by
  sorry

end NUMINAMATH_GPT_find_value_l1337_133769


namespace NUMINAMATH_GPT_find_pairs_l1337_133779

def sequence_a : Nat → Int
| 0 => 0
| 1 => 0
| n+2 => 2 * sequence_a (n+1) - sequence_a n + 2

def sequence_b : Nat → Int
| 0 => 8
| 1 => 8
| n+2 => 2 * sequence_b (n+1) - sequence_b n

theorem find_pairs :
  (sequence_a 1992 = 31872 ∧ sequence_b 1992 = 31880) ∨
  (sequence_a 1992 = -31872 ∧ sequence_b 1992 = -31864) :=
sorry

end NUMINAMATH_GPT_find_pairs_l1337_133779


namespace NUMINAMATH_GPT_max_ratio_l1337_133747

theorem max_ratio {a b c d : ℝ} 
  (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d > 0) 
  (h2 : a^2 + b^2 + c^2 + d^2 = (a + b + c + d)^2 / 3) : 
  ∃ x, x = (7 + 2 * Real.sqrt 6) / 5 ∧ x = (a + c) / (b + d) :=
by
  sorry

end NUMINAMATH_GPT_max_ratio_l1337_133747


namespace NUMINAMATH_GPT_least_pos_int_div_by_four_distinct_primes_l1337_133767

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end NUMINAMATH_GPT_least_pos_int_div_by_four_distinct_primes_l1337_133767


namespace NUMINAMATH_GPT_semicircle_radius_l1337_133792

theorem semicircle_radius (π : ℝ) (P : ℝ) (r : ℝ) (hπ : π ≠ 0) (hP : P = 162) (hPerimeter : P = π * r + 2 * r) : r = 162 / (π + 2) :=
by
  sorry

end NUMINAMATH_GPT_semicircle_radius_l1337_133792


namespace NUMINAMATH_GPT_calc_square_uncovered_area_l1337_133787

theorem calc_square_uncovered_area :
  ∀ (side_length : ℕ) (circle_diameter : ℝ) (num_circles : ℕ),
    side_length = 16 →
    circle_diameter = (16 / 3) →
    num_circles = 9 →
    (side_length ^ 2) - num_circles * (Real.pi * (circle_diameter / 2) ^ 2) = 256 - 64 * Real.pi :=
by
  intros side_length circle_diameter num_circles h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_calc_square_uncovered_area_l1337_133787


namespace NUMINAMATH_GPT_determine_clothes_l1337_133731

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end NUMINAMATH_GPT_determine_clothes_l1337_133731


namespace NUMINAMATH_GPT_cos_of_double_angles_l1337_133751

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_of_double_angles_l1337_133751


namespace NUMINAMATH_GPT_tan_alpha_sqrt3_l1337_133732

theorem tan_alpha_sqrt3 (α : ℝ) (h : Real.sin (α + 20 * Real.pi / 180) = Real.cos (α + 10 * Real.pi / 180) + Real.cos (α - 10 * Real.pi / 180)) :
  Real.tan α = Real.sqrt 3 := 
  sorry

end NUMINAMATH_GPT_tan_alpha_sqrt3_l1337_133732


namespace NUMINAMATH_GPT_apples_per_box_l1337_133741

variable (A : ℕ) -- Number of apples packed in a box

-- Conditions
def normal_boxes_per_day := 50
def days_per_week := 7
def boxes_first_week := normal_boxes_per_day * days_per_week * A
def boxes_second_week := (normal_boxes_per_day * A - 500) * days_per_week
def total_apples := 24500

-- Theorem
theorem apples_per_box : boxes_first_week + boxes_second_week = total_apples → A = 40 :=
by
  sorry

end NUMINAMATH_GPT_apples_per_box_l1337_133741


namespace NUMINAMATH_GPT_solve_for_x_l1337_133788

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.1 * (30 + x) = 15.5 → x = 83 := by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1337_133788


namespace NUMINAMATH_GPT_canyon_trail_length_l1337_133780

theorem canyon_trail_length
  (a b c d e : ℝ)
  (h1 : a + b + c = 36)
  (h2 : b + c + d = 42)
  (h3 : c + d + e = 45)
  (h4 : a + d = 29) :
  a + b + c + d + e = 71 :=
by sorry

end NUMINAMATH_GPT_canyon_trail_length_l1337_133780


namespace NUMINAMATH_GPT_prism_faces_l1337_133706

-- Define conditions based on the problem
def num_edges_of_prism (L : ℕ) : ℕ := 3 * L

theorem prism_faces (L : ℕ) (hL : num_edges_of_prism L = 18) : L + 2 = 8 :=
by
  -- Since the number of edges of the prism is given by 3L,
  -- if num_edges_of_prism L = 18, then 3L = 18 leading to L = 6.
  -- Thus, the total number of faces (L + 2) = 6 + 2 = 8.
  sorry

end NUMINAMATH_GPT_prism_faces_l1337_133706


namespace NUMINAMATH_GPT_q_can_complete_work_in_25_days_l1337_133742

-- Define work rates for p, q, and r
variables (W_p W_q W_r : ℝ)

-- Define total work
variable (W : ℝ)

-- Prove that q can complete the work in 25 days under given conditions
theorem q_can_complete_work_in_25_days
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = W / 10)
  (h3 : W_r = W / 50) :
  W_q = W / 25 :=
by
  -- Given: W_p = W_q + W_r
  -- Given: W_p + W_q = W / 10
  -- Given: W_r = W / 50
  -- We need to prove: W_q = W / 25
  sorry

end NUMINAMATH_GPT_q_can_complete_work_in_25_days_l1337_133742


namespace NUMINAMATH_GPT_quadratic_inequality_l1337_133790

-- Defining the quadratic expression
def quadratic_expr (a x : ℝ) : ℝ :=
  (a + 2) * x^2 + 2 * (a + 2) * x + 4

-- Statement to be proven
theorem quadratic_inequality {a : ℝ} :
  (∀ x : ℝ, quadratic_expr a x > 0) ↔ -2 ≤ a ∧ a < 2 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_quadratic_inequality_l1337_133790


namespace NUMINAMATH_GPT_mn_values_l1337_133702

theorem mn_values (m n : ℤ) (h : m^2 * n^2 + m^2 + n^2 + 10 * m * n + 16 = 0) : 
  (m = 2 ∧ n = -2) ∨ (m = -2 ∧ n = 2) :=
  sorry

end NUMINAMATH_GPT_mn_values_l1337_133702


namespace NUMINAMATH_GPT_vector_subtraction_magnitude_l1337_133720

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def condition1 : Real := 3 -- |a|
def condition2 : Real := 2 -- |b|
def condition3 : Real := 4 -- |a + b|

-- Proving the statement
theorem vector_subtraction_magnitude (h1 : ‖a‖ = condition1) (h2 : ‖b‖ = condition2) (h3 : ‖a + b‖ = condition3) :
  ‖a - b‖ = Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_vector_subtraction_magnitude_l1337_133720


namespace NUMINAMATH_GPT_selling_price_of_cycle_l1337_133777

theorem selling_price_of_cycle (cost_price : ℕ) (loss_percent : ℕ) (selling_price : ℕ) :
  cost_price = 1400 → loss_percent = 25 → selling_price = 1050 := by
  sorry

end NUMINAMATH_GPT_selling_price_of_cycle_l1337_133777


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_l1337_133733

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)
noncomputable def sum_known_angles : ℝ := 3780

theorem number_of_sides_of_polygon
  (n : ℕ)
  (h1 : sum_known_angles + missing_angle = sum_of_interior_angles n)
  (h2 : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a = 3 * c ∧ b = 3 * c ∧ a + b + c ≤ sum_known_angles) :
  n = 23 :=
sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_l1337_133733


namespace NUMINAMATH_GPT_bowl_weight_after_refill_l1337_133738

-- Define the problem conditions
def empty_bowl_weight : ℕ := 420
def day1_consumption : ℕ := 53
def day2_consumption : ℕ := 76
def day3_consumption : ℕ := 65
def day4_consumption : ℕ := 14

-- Define the total consumption over 4 days
def total_consumption : ℕ :=
  day1_consumption + day2_consumption + day3_consumption + day4_consumption

-- Define the final weight of the bowl after refilling
def final_bowl_weight : ℕ :=
  empty_bowl_weight + total_consumption

-- Statement to prove
theorem bowl_weight_after_refill : final_bowl_weight = 628 := by
  sorry

end NUMINAMATH_GPT_bowl_weight_after_refill_l1337_133738


namespace NUMINAMATH_GPT_nat_square_not_div_factorial_l1337_133754

-- Define n as a natural number
def n : Nat := sorry  -- We assume n is given somewhere

-- Define a function to check if a number is prime
def is_prime (p : Nat) : Prop := sorry  -- Placeholder for prime checking function

-- The main theorem to prove
theorem nat_square_not_div_factorial (n : Nat) : (n = 4 ∨ is_prime n) → ¬ ((n * n) ∣ Nat.factorial n) := by
  sorry

end NUMINAMATH_GPT_nat_square_not_div_factorial_l1337_133754


namespace NUMINAMATH_GPT_percentage_for_x_plus_y_l1337_133736

theorem percentage_for_x_plus_y (x y : Real) (P : Real) 
  (h1 : 0.60 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.5 * x) : 
  P = 20 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_for_x_plus_y_l1337_133736


namespace NUMINAMATH_GPT_rhombus_area_l1337_133772

theorem rhombus_area (R1 R2 : ℝ) (x y : ℝ)
  (hR1 : R1 = 15) (hR2 : R2 = 30)
  (hx : x = 15) (hy : y = 2 * x):
  (x * y / 2 = 225) :=
by 
  -- Lean 4 proof not required here
  sorry

end NUMINAMATH_GPT_rhombus_area_l1337_133772


namespace NUMINAMATH_GPT_elise_spent_on_comic_book_l1337_133717

-- Define the initial amount of money Elise had
def initial_amount : ℤ := 8

-- Define the amount saved from allowance
def saved_amount : ℤ := 13

-- Define the amount spent on puzzle
def spent_on_puzzle : ℤ := 18

-- Define the amount left after all expenditures
def amount_left : ℤ := 1

-- Define the total amount of money Elise had after saving
def total_amount : ℤ := initial_amount + saved_amount

-- Define the total amount spent which equals
-- the sum of amount spent on the comic book and the puzzle
def total_spent : ℤ := total_amount - amount_left

-- Define the amount spent on the comic book as the proposition to be proved
def spent_on_comic_book : ℤ := total_spent - spent_on_puzzle

-- State the theorem to prove how much Elise spent on the comic book
theorem elise_spent_on_comic_book : spent_on_comic_book = 2 :=
by
  sorry

end NUMINAMATH_GPT_elise_spent_on_comic_book_l1337_133717


namespace NUMINAMATH_GPT_quadratic_functions_count_correct_even_functions_count_correct_l1337_133748

def num_coefficients := 4
def valid_coefficients := [-1, 0, 1, 2]

def count_quadratic_functions : ℕ :=
  num_coefficients * num_coefficients * (num_coefficients - 1)

def count_even_functions : ℕ :=
  (num_coefficients - 1) * (num_coefficients - 2)

def total_quad_functions_correct : Prop := count_quadratic_functions = 18
def total_even_functions_correct : Prop := count_even_functions = 6

theorem quadratic_functions_count_correct : total_quad_functions_correct :=
by sorry

theorem even_functions_count_correct : total_even_functions_correct :=
by sorry

end NUMINAMATH_GPT_quadratic_functions_count_correct_even_functions_count_correct_l1337_133748


namespace NUMINAMATH_GPT_find_m_l1337_133713

theorem find_m (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 1/2) : 
  m = 100 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l1337_133713


namespace NUMINAMATH_GPT_tammy_speed_proof_l1337_133723

noncomputable def tammy_average_speed_second_day (v t : ℝ) :=
  v + 0.5

theorem tammy_speed_proof :
  ∃ v t : ℝ, 
    t + (t - 2) = 14 ∧
    v * t + (v + 0.5) * (t - 2) = 52 ∧
    tammy_average_speed_second_day v t = 4 :=
by
  sorry

end NUMINAMATH_GPT_tammy_speed_proof_l1337_133723


namespace NUMINAMATH_GPT_probability_of_A_l1337_133789

variable (A B : Prop)
variable (P : Prop → ℝ)

-- Given conditions
variable (h1 : P (A ∧ B) = 0.72)
variable (h2 : P (A ∧ ¬B) = 0.18)

theorem probability_of_A: P A = 0.90 := sorry

end NUMINAMATH_GPT_probability_of_A_l1337_133789


namespace NUMINAMATH_GPT_sequence_total_sum_is_correct_l1337_133760

-- Define the sequence pattern
def sequence_sum : ℕ → ℤ
| 0       => 1
| 1       => -2
| 2       => -4
| 3       => 8
| (n + 4) => sequence_sum n + 4

-- Define the number of groups in the sequence
def num_groups : ℕ := 319

-- Define the sum of each individual group
def group_sum : ℤ := 3

-- Define the total sum of the sequence
def total_sum : ℤ := num_groups * group_sum

theorem sequence_total_sum_is_correct : total_sum = 957 := by
  sorry

end NUMINAMATH_GPT_sequence_total_sum_is_correct_l1337_133760


namespace NUMINAMATH_GPT_overall_average_is_63_point_4_l1337_133776

theorem overall_average_is_63_point_4 : 
  ∃ (n total_marks : ℕ) (avg_marks : ℚ), 
  n = 50 ∧ 
  (∃ (marks_group1 marks_group2 marks_group3 marks_remaining : ℕ), 
    marks_group1 = 6 * 95 ∧
    marks_group2 = 4 * 0 ∧
    marks_group3 = 10 * 80 ∧
    marks_remaining = (n - 20) * 60 ∧
    total_marks = marks_group1 + marks_group2 + marks_group3 + marks_remaining) ∧ 
  avg_marks = total_marks / n ∧ 
  avg_marks = 63.4 := 
by 
  sorry

end NUMINAMATH_GPT_overall_average_is_63_point_4_l1337_133776


namespace NUMINAMATH_GPT_largest_four_digit_integer_congruent_to_17_mod_26_l1337_133785

theorem largest_four_digit_integer_congruent_to_17_mod_26 :
  ∃ x : ℤ, 1000 ≤ x ∧ x < 10000 ∧ x % 26 = 17 ∧ x = 9978 :=
by
  sorry

end NUMINAMATH_GPT_largest_four_digit_integer_congruent_to_17_mod_26_l1337_133785


namespace NUMINAMATH_GPT_intersection_A_B_l1337_133734

def set_A : Set ℝ := { x | abs (x - 1) < 2 }
def set_B : Set ℝ := { x | Real.log x / Real.log 2 > Real.log x / Real.log 3 }

theorem intersection_A_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1337_133734


namespace NUMINAMATH_GPT_system_of_equations_solution_system_of_inequalities_no_solution_l1337_133735

-- Problem 1: Solving system of linear equations
theorem system_of_equations_solution :
  ∃ x y : ℝ, x - 3*y = -5 ∧ 2*x + 2*y = 6 ∧ x = 1 ∧ y = 2 := by
  sorry

-- Problem 2: Solving the system of inequalities
theorem system_of_inequalities_no_solution :
  ¬ (∃ x : ℝ, 2*x < -4 ∧ (1/2)*x - 5 > 1 - (3/2)*x) := by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_system_of_inequalities_no_solution_l1337_133735


namespace NUMINAMATH_GPT_quadratic_properties_l1337_133703

theorem quadratic_properties (d e f : ℝ)
  (h1 : d * 1^2 + e * 1 + f = 3)
  (h2 : d * 2^2 + e * 2 + f = 0)
  (h3 : d * 9 + e * 3 + f = -3) :
  d + e + 2 * f = 19.5 :=
sorry

end NUMINAMATH_GPT_quadratic_properties_l1337_133703


namespace NUMINAMATH_GPT_max_volumes_on_fedor_shelf_l1337_133749

theorem max_volumes_on_fedor_shelf 
  (S s1 s2 n : ℕ) 
  (h1 : S + s1 ≥ (n - 2) / 2) 
  (h2 : S + s2 < (n - 2) / 3) 
  : n = 12 := 
sorry

end NUMINAMATH_GPT_max_volumes_on_fedor_shelf_l1337_133749


namespace NUMINAMATH_GPT_gcd_a_b_l1337_133768

-- Define a and b
def a : ℕ := 333333
def b : ℕ := 9999999

-- Prove that gcd(a, b) = 3
theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_a_b_l1337_133768


namespace NUMINAMATH_GPT_lillian_candies_addition_l1337_133704

noncomputable def lillian_initial_candies : ℕ := 88
noncomputable def lillian_father_candies : ℕ := 5
noncomputable def lillian_total_candies : ℕ := 93

theorem lillian_candies_addition : lillian_initial_candies + lillian_father_candies = lillian_total_candies := by
  sorry

end NUMINAMATH_GPT_lillian_candies_addition_l1337_133704


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l1337_133727

theorem infinite_geometric_series_sum :
  ∑' (n : ℕ), (1 : ℚ) * (-1 / 4 : ℚ) ^ n = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l1337_133727


namespace NUMINAMATH_GPT_binomial_expansion_coefficients_equal_l1337_133756

theorem binomial_expansion_coefficients_equal (n : ℕ) (h : n ≥ 6)
  (h_eq : 3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) : n = 7 := by
  sorry

end NUMINAMATH_GPT_binomial_expansion_coefficients_equal_l1337_133756


namespace NUMINAMATH_GPT_initial_number_of_girls_l1337_133719

theorem initial_number_of_girls (p : ℝ) (h : (0.4 * p - 2) / p = 0.3) : 0.4 * p = 8 := 
by
  sorry

end NUMINAMATH_GPT_initial_number_of_girls_l1337_133719


namespace NUMINAMATH_GPT_find_smallest_sphere_radius_squared_l1337_133770

noncomputable def smallest_sphere_radius_squared
  (AB CD AD BC : ℝ) (angle_ABC : ℝ) (radius_AC_squared : ℝ) : ℝ :=
if AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120 then radius_AC_squared else 0

theorem find_smallest_sphere_radius_squared
  (AB CD AD BC : ℝ) (angle_ABC : ℝ) (radius_AC_squared : ℝ) :
  (AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120) →
  radius_AC_squared = 49 :=
by
  intros h
  have h_ABCD : AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120 := h
  sorry -- The proof steps would be filled in here

end NUMINAMATH_GPT_find_smallest_sphere_radius_squared_l1337_133770


namespace NUMINAMATH_GPT_total_number_of_values_l1337_133705

theorem total_number_of_values (S n : ℕ) (h1 : (S - 165 + 135) / n = 150) (h2 : S / n = 151) : n = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_number_of_values_l1337_133705


namespace NUMINAMATH_GPT_nancy_hourly_wage_l1337_133784

-- Definitions based on conditions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def scholarship_amount : ℕ := 3000
def loan_amount : ℕ := 2 * scholarship_amount
def nancy_contributions : ℕ := parents_contribution + scholarship_amount + loan_amount
def remaining_tuition : ℕ := tuition_per_semester - nancy_contributions
def total_working_hours : ℕ := 200

-- Theorem to prove based on the formulated problem
theorem nancy_hourly_wage :
  (remaining_tuition / total_working_hours) = 10 :=
by
  sorry

end NUMINAMATH_GPT_nancy_hourly_wage_l1337_133784


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1337_133750

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + x - 12 ≥ 0} = {x : ℝ | x ≤ -4 ∨ x ≥ 3} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1337_133750


namespace NUMINAMATH_GPT_gcd_lcm_sum_l1337_133708

theorem gcd_lcm_sum (a b : ℕ) (h : a = 1999 * b) : Nat.gcd a b + Nat.lcm a b = 2000 * b := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l1337_133708


namespace NUMINAMATH_GPT_min_sum_abc_l1337_133701

theorem min_sum_abc (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hprod : a * b * c = 2550) : a + b + c ≥ 48 :=
by sorry

end NUMINAMATH_GPT_min_sum_abc_l1337_133701


namespace NUMINAMATH_GPT_carly_butterfly_days_l1337_133739

-- Define the conditions
variable (x : ℕ) -- number of days Carly practices her butterfly stroke
def butterfly_hours_per_day := 3  -- hours per day for butterfly stroke
def backstroke_hours_per_day := 2  -- hours per day for backstroke stroke
def backstroke_days_per_week := 6  -- days per week for backstroke stroke
def total_hours_per_month := 96  -- total hours practicing swimming in a month
def weeks_in_month := 4  -- number of weeks in a month

-- The proof problem
theorem carly_butterfly_days :
  (butterfly_hours_per_day * x + backstroke_hours_per_day * backstroke_days_per_week) * weeks_in_month = total_hours_per_month
  → x = 4 := 
by
  sorry

end NUMINAMATH_GPT_carly_butterfly_days_l1337_133739


namespace NUMINAMATH_GPT_sum_three_ways_l1337_133728

theorem sum_three_ways (n : ℕ) (h : n > 0) : 
  ∃ k, k = (n^2) / 12 ∧ k = (n^2) / 12 :=
sorry

end NUMINAMATH_GPT_sum_three_ways_l1337_133728


namespace NUMINAMATH_GPT_no_perfect_powers_in_sequence_l1337_133757

noncomputable def nth_triplet (n : Nat) : Nat × Nat × Nat :=
  Nat.recOn n (2, 3, 5) (λ _ ⟨a, b, c⟩ => (a + c, a + b, b + c))

def is_perfect_power (x : Nat) : Prop :=
  ∃ (m : Nat) (k : Nat), k ≥ 2 ∧ m^k = x

theorem no_perfect_powers_in_sequence : ∀ (n : Nat), ∀ (a b c : Nat),
  nth_triplet n = (a, b, c) →
  ¬(is_perfect_power a ∨ is_perfect_power b ∨ is_perfect_power c) :=
by
  intros
  sorry

end NUMINAMATH_GPT_no_perfect_powers_in_sequence_l1337_133757


namespace NUMINAMATH_GPT_flowers_per_bouquet_l1337_133783

theorem flowers_per_bouquet (total_flowers wilted_flowers : ℕ) (bouquets : ℕ) (remaining_flowers : ℕ)
    (h1 : total_flowers = 45)
    (h2 : wilted_flowers = 35)
    (h3 : bouquets = 2)
    (h4 : remaining_flowers = total_flowers - wilted_flowers)
    (h5 : bouquets * (remaining_flowers / bouquets) = remaining_flowers) :
  remaining_flowers / bouquets = 5 :=
by
  sorry

end NUMINAMATH_GPT_flowers_per_bouquet_l1337_133783


namespace NUMINAMATH_GPT_smallest_positive_x_for_maximum_sine_sum_l1337_133709

theorem smallest_positive_x_for_maximum_sine_sum :
  ∃ x : ℝ, (0 < x) ∧ (∃ k m : ℕ, x = 450 + 1800 * k ∧ x = 630 + 2520 * m ∧ x = 12690) := by
  sorry

end NUMINAMATH_GPT_smallest_positive_x_for_maximum_sine_sum_l1337_133709


namespace NUMINAMATH_GPT_square_area_eq_1296_l1337_133782

theorem square_area_eq_1296 (x : ℝ) (side : ℝ) (h1 : side = 6 * x - 18) (h2 : side = 3 * x + 9) : side ^ 2 = 1296 := sorry

end NUMINAMATH_GPT_square_area_eq_1296_l1337_133782


namespace NUMINAMATH_GPT_smallest_n_l1337_133712

theorem smallest_n
  (n : ℕ)
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 1)
  (h4 : n % 5 = 1)
  (h5 : n % 6 = 1)
  (h6 : n % 7 = 1)
  (h7 : 8 ∣ n) :
  n = 1681 :=
  sorry

end NUMINAMATH_GPT_smallest_n_l1337_133712


namespace NUMINAMATH_GPT_ferrisWheelPeopleCount_l1337_133700

/-!
# Problem Description

We are given the following conditions:
- The ferris wheel has 6.0 seats.
- It has to run 2.333333333 times for everyone to get a turn.

We need to prove that the total number of people who want to ride the ferris wheel is 14.
-/

def ferrisWheelSeats : ℕ := 6
def ferrisWheelRuns : ℚ := 2333333333 / 1000000000

theorem ferrisWheelPeopleCount :
  (ferrisWheelSeats : ℚ) * ferrisWheelRuns = 14 :=
by
  sorry

end NUMINAMATH_GPT_ferrisWheelPeopleCount_l1337_133700


namespace NUMINAMATH_GPT_floor_sqrt_equality_l1337_133718

theorem floor_sqrt_equality (n : ℕ) : 
  (Int.floor (Real.sqrt (4 * n + 1))) = (Int.floor (Real.sqrt (4 * n + 3))) := 
by 
  sorry

end NUMINAMATH_GPT_floor_sqrt_equality_l1337_133718


namespace NUMINAMATH_GPT_square_of_binomial_example_l1337_133764

theorem square_of_binomial_example : (23^2 + 2 * 23 * 2 + 2^2 = 625) :=
by
  sorry

end NUMINAMATH_GPT_square_of_binomial_example_l1337_133764
