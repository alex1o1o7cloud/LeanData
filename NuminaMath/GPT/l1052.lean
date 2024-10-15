import Mathlib

namespace NUMINAMATH_GPT_not_perfect_square_4n_squared_plus_4n_plus_4_l1052_105288

theorem not_perfect_square_4n_squared_plus_4n_plus_4 :
  ¬ ∃ m n : ℕ, m^2 = 4 * n^2 + 4 * n + 4 := 
by
  sorry

end NUMINAMATH_GPT_not_perfect_square_4n_squared_plus_4n_plus_4_l1052_105288


namespace NUMINAMATH_GPT_point_on_circle_x_value_l1052_105296

/-
In the xy-plane, the segment with endpoints (-3,0) and (21,0) is the diameter of a circle.
If the point (x,12) is on the circle, then x = 9.
-/
theorem point_on_circle_x_value :
  let c := (9, 0) -- center of the circle
  let r := 12 -- radius of the circle
  let circle := {p | (p.1 - 9)^2 + p.2^2 = 144} -- equation of the circle
  ∀ x : Real, (x, 12) ∈ circle → x = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_point_on_circle_x_value_l1052_105296


namespace NUMINAMATH_GPT_millennium_run_time_l1052_105269

theorem millennium_run_time (M A B : ℕ) (h1 : B = 100) (h2 : B = A + 10) (h3 : A = M - 30) : M = 120 := by
  sorry

end NUMINAMATH_GPT_millennium_run_time_l1052_105269


namespace NUMINAMATH_GPT_solve_combinations_l1052_105209

-- This function calculates combinations
noncomputable def C (n k : ℕ) : ℕ := if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem solve_combinations (x : ℤ) :
  C 16 (x^2 - x).natAbs = C 16 (5*x - 5).natAbs → x = 1 ∨ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_combinations_l1052_105209


namespace NUMINAMATH_GPT_copper_production_is_correct_l1052_105289

-- Define the percentages of copper production for each mine
def percentage_copper_mine_a : ℝ := 0.05
def percentage_copper_mine_b : ℝ := 0.10
def percentage_copper_mine_c : ℝ := 0.15

-- Define the daily production of each mine in tons
def daily_production_mine_a : ℕ := 3000
def daily_production_mine_b : ℕ := 4000
def daily_production_mine_c : ℕ := 3500

-- Define the total copper produced from all mines
def total_copper_produced : ℝ :=
  percentage_copper_mine_a * daily_production_mine_a +
  percentage_copper_mine_b * daily_production_mine_b +
  percentage_copper_mine_c * daily_production_mine_c

-- Prove that the total daily copper production is 1075 tons
theorem copper_production_is_correct :
  total_copper_produced = 1075 := 
sorry

end NUMINAMATH_GPT_copper_production_is_correct_l1052_105289


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1052_105277

def A : Set ℝ := { x | x^2 - x > 0 }
def B : Set ℝ := { x | Real.log x / Real.log 2 < 2 }

theorem intersection_of_A_and_B : A ∩ B = { x | 1 < x ∧ x < 4 } :=
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1052_105277


namespace NUMINAMATH_GPT_loan_amount_l1052_105213

theorem loan_amount (R T SI : ℕ) (hR : R = 7) (hT : T = 7) (hSI : SI = 735) : 
  ∃ P : ℕ, P = 1500 := 
by 
  sorry

end NUMINAMATH_GPT_loan_amount_l1052_105213


namespace NUMINAMATH_GPT_alcohol_solution_problem_l1052_105237

theorem alcohol_solution_problem (x_vol y_vol : ℚ) (x_alcohol y_alcohol target_alcohol : ℚ) (target_vol : ℚ) :
  x_vol = 250 ∧ x_alcohol = 10/100 ∧ y_alcohol = 30/100 ∧ target_alcohol = 25/100 ∧ target_vol = 250 + y_vol →
  (x_alcohol * x_vol + y_alcohol * y_vol = target_alcohol * target_vol) →
  y_vol = 750 :=
by
  sorry

end NUMINAMATH_GPT_alcohol_solution_problem_l1052_105237


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1052_105239

-- Definitions of the conditions
def with_stream_speed : ℝ := 36
def against_stream_speed : ℝ := 8

-- Let Vb be the speed of the boat in still water, and Vs be the speed of the stream.
variable (Vb Vs : ℝ)

-- Conditions given in the problem
axiom h1 : Vb + Vs = with_stream_speed
axiom h2 : Vb - Vs = against_stream_speed

-- The statement to prove: the speed of the boat in still water is 22 km/h.
theorem boat_speed_in_still_water : Vb = 22 := by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1052_105239


namespace NUMINAMATH_GPT_high_school_total_students_l1052_105263

theorem high_school_total_students (N_seniors N_sample N_freshmen_sample N_sophomores_sample N_total : ℕ)
  (h_seniors : N_seniors = 1000)
  (h_sample : N_sample = 185)
  (h_freshmen_sample : N_freshmen_sample = 75)
  (h_sophomores_sample : N_sophomores_sample = 60)
  (h_proportion : N_seniors * (N_sample - (N_freshmen_sample + N_sophomores_sample)) = N_total * (N_sample - N_freshmen_sample - N_sophomores_sample)) :
  N_total = 3700 :=
by
  sorry

end NUMINAMATH_GPT_high_school_total_students_l1052_105263


namespace NUMINAMATH_GPT_number_of_possible_values_l1052_105238

-- Define the decimal number s and its representation
def s (e f g h : ℕ) : ℚ := e / 10 + f / 100 + g / 1000 + h / 10000

-- Define the condition that the closest fraction is 2/9
def closest_to_2_9 (s : ℚ) : Prop :=
  abs (s - 2 / 9) < min (abs (s - 1 / 5)) (abs (s - 1 / 6)) ∧
  abs (s - 2 / 9) < min (abs (s - 1 / 5)) (abs (s - 2 / 11))

-- The main theorem stating the number of possible values for s
theorem number_of_possible_values :
  (∃ e f g h : ℕ, 0 ≤ e ∧ e ≤ 9 ∧ 0 ≤ f ∧ f ≤ 9 ∧ 0 ≤ g ∧ g ≤ 9 ∧ 0 ≤ h ∧ h ≤ 9 ∧
    closest_to_2_9 (s e f g h)) → (∃ n : ℕ, n = 169) :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_values_l1052_105238


namespace NUMINAMATH_GPT_flour_per_cake_l1052_105275

theorem flour_per_cake (traci_flour harris_flour : ℕ) (cakes_each : ℕ)
  (h_traci_flour : traci_flour = 500)
  (h_harris_flour : harris_flour = 400)
  (h_cakes_each : cakes_each = 9) :
  (traci_flour + harris_flour) / (2 * cakes_each) = 50 := by
  sorry

end NUMINAMATH_GPT_flour_per_cake_l1052_105275


namespace NUMINAMATH_GPT_profit_benny_wants_to_make_l1052_105249

noncomputable def pumpkin_pies : ℕ := 10
noncomputable def cherry_pies : ℕ := 12
noncomputable def cost_pumpkin_pie : ℝ := 3
noncomputable def cost_cherry_pie : ℝ := 5
noncomputable def price_per_pie : ℝ := 5

theorem profit_benny_wants_to_make : 5 * (pumpkin_pies + cherry_pies) - (pumpkin_pies * cost_pumpkin_pie + cherry_pies * cost_cherry_pie) = 20 :=
by
  sorry

end NUMINAMATH_GPT_profit_benny_wants_to_make_l1052_105249


namespace NUMINAMATH_GPT_determine_x_l1052_105223

variable {x y : ℝ}

theorem determine_x (h : (x - 1) / x = (y^3 + 3 * y^2 - 4) / (y^3 + 3 * y^2 - 5)) : 
  x = y^3 + 3 * y^2 - 5 := 
sorry

end NUMINAMATH_GPT_determine_x_l1052_105223


namespace NUMINAMATH_GPT_simplify_fraction_l1052_105270

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end NUMINAMATH_GPT_simplify_fraction_l1052_105270


namespace NUMINAMATH_GPT_positive_iff_triangle_l1052_105210

def is_triangle_inequality (x y z : ℝ) : Prop :=
  (x + y > z) ∧ (x + z > y) ∧ (y + z > x)

noncomputable def poly (x y z : ℝ) : ℝ :=
  (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

theorem positive_iff_triangle (x y z : ℝ) : 
  poly |x| |y| |z| > 0 ↔ is_triangle_inequality |x| |y| |z| :=
sorry

end NUMINAMATH_GPT_positive_iff_triangle_l1052_105210


namespace NUMINAMATH_GPT_cos420_add_sin330_l1052_105282

theorem cos420_add_sin330 : Real.cos (420 * Real.pi / 180) + Real.sin (330 * Real.pi / 180) = 0 := 
by
  sorry

end NUMINAMATH_GPT_cos420_add_sin330_l1052_105282


namespace NUMINAMATH_GPT_tangent_line_at_1_f_geq_x_minus_1_min_value_a_l1052_105231

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- 1. Proof that the equation of the tangent line at the point (1, f(1)) is y = x - 1
theorem tangent_line_at_1 :
  ∃ k b, (k = 1 ∧ b = -1 ∧ (∀ x, (f x - k * x - b) = 0)) :=
sorry

-- 2. Proof that f(x) ≥ x - 1 for all x in (0, +∞)
theorem f_geq_x_minus_1 :
  ∀ x, 0 < x → f x ≥ x - 1 :=
sorry

-- 3. Proof that the minimum value of a such that f(x) ≥ ax² + 2/a for all x in (0, +∞) is -e³
theorem min_value_a :
  ∃ a, (∀ x, 0 < x → f x ≥ a * x^2 + 2 / a) ∧ (a = -Real.exp 3) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_1_f_geq_x_minus_1_min_value_a_l1052_105231


namespace NUMINAMATH_GPT_find_number_l1052_105242

theorem find_number (x : ℝ) (h : x * 2 + (12 + 4) * (1/8) = 602) : x = 300 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1052_105242


namespace NUMINAMATH_GPT_complex_division_l1052_105248

-- Define i as the imaginary unit
def i : Complex := Complex.I

-- Define the problem statement to prove that 2i / (1 - i) equals -1 + i
theorem complex_division : (2 * i) / (1 - i) = -1 + i :=
by
  -- Since we are focusing on the statement, we use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_complex_division_l1052_105248


namespace NUMINAMATH_GPT_monotonic_increasing_range_l1052_105235

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) * (x + a) / x

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, x > 0 → (∀ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 < x2 → f x1 a ≤ f x2 a)) ↔ -4 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_range_l1052_105235


namespace NUMINAMATH_GPT_sum_of_first_45_natural_numbers_l1052_105256

theorem sum_of_first_45_natural_numbers : (45 * (45 + 1)) / 2 = 1035 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_45_natural_numbers_l1052_105256


namespace NUMINAMATH_GPT_problem_l1052_105247

theorem problem (f : ℝ → ℝ) (h : ∀ x, (x - 3) * (deriv f x) ≤ 0) : 
  f 0 + f 6 ≤ 2 * f 3 := 
sorry

end NUMINAMATH_GPT_problem_l1052_105247


namespace NUMINAMATH_GPT_alexei_loss_per_week_l1052_105279

-- Definitions
def aleesia_loss_per_week : ℝ := 1.5
def aleesia_total_weeks : ℕ := 10
def total_loss : ℝ := 35
def alexei_total_weeks : ℕ := 8

-- The statement to prove
theorem alexei_loss_per_week :
  (total_loss - aleesia_loss_per_week * aleesia_total_weeks) / alexei_total_weeks = 2.5 := 
by sorry

end NUMINAMATH_GPT_alexei_loss_per_week_l1052_105279


namespace NUMINAMATH_GPT_option_D_correct_l1052_105294

theorem option_D_correct (f : ℕ+ → ℕ) (h : ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2) 
  (hf : f 4 ≥ 25) : ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 :=
by
  sorry

end NUMINAMATH_GPT_option_D_correct_l1052_105294


namespace NUMINAMATH_GPT_total_fat_l1052_105207

def herring_fat := 40
def eel_fat := 20
def pike_fat := eel_fat + 10

def herrings := 40
def eels := 40
def pikes := 40

theorem total_fat :
  (herrings * herring_fat) + (eels * eel_fat) + (pikes * pike_fat) = 3600 :=
by
  sorry

end NUMINAMATH_GPT_total_fat_l1052_105207


namespace NUMINAMATH_GPT_range_of_a_increasing_f_on_interval_l1052_105295

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- Define the condition that f(x) is increasing on [4, +∞)
def isIncreasingOnInterval (a : ℝ) : Prop :=
  ∀ x y : ℝ, 4 ≤ x → x ≤ y → f a x ≤ f a y

theorem range_of_a_increasing_f_on_interval :
  (∀ a : ℝ, isIncreasingOnInterval a → a ≥ -3) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_increasing_f_on_interval_l1052_105295


namespace NUMINAMATH_GPT_parabola_equation_l1052_105297

-- Definitions for the given conditions
def parabola_vertex_origin (y x : ℝ) : Prop := y = 0 ↔ x = 0
def axis_of_symmetry_x (y x : ℝ) : Prop := (x = -y) ↔ (x = y)
def focus_on_line (y x : ℝ) : Prop := 3 * x - 4 * y - 12 = 0

-- The statement to be proved
theorem parabola_equation :
  ∀ (y x : ℝ),
  (parabola_vertex_origin y x) ∧ (axis_of_symmetry_x y x) ∧ (focus_on_line y x) →
  y^2 = 16 * x :=
by
  intros y x h
  sorry

end NUMINAMATH_GPT_parabola_equation_l1052_105297


namespace NUMINAMATH_GPT_cookies_left_l1052_105233

def initial_cookies : ℕ := 93
def eaten_cookies : ℕ := 15

theorem cookies_left : initial_cookies - eaten_cookies = 78 := by
  sorry

end NUMINAMATH_GPT_cookies_left_l1052_105233


namespace NUMINAMATH_GPT_total_number_of_girls_in_school_l1052_105246

theorem total_number_of_girls_in_school 
  (students_sampled : ℕ) 
  (students_total : ℕ) 
  (sample_girls : ℕ) 
  (sample_boys : ℕ)
  (h_sample_size : students_sampled = 200)
  (h_total_students : students_total = 2000)
  (h_diff_girls_boys : sample_boys = sample_girls + 6)
  (h_stratified_sampling : students_sampled / students_total = 200 / 2000) :
  sample_girls * (students_total / students_sampled) = 970 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_girls_in_school_l1052_105246


namespace NUMINAMATH_GPT_bounded_sequence_range_l1052_105205

theorem bounded_sequence_range (a : ℝ) (a_n : ℕ → ℝ) (h1 : a_n 1 = a)
    (hrec : ∀ n : ℕ, a_n (n + 1) = 3 * (a_n n)^3 - 7 * (a_n n)^2 + 5 * (a_n n))
    (bounded : ∃ M : ℝ, ∀ n : ℕ, abs (a_n n) ≤ M) :
    0 ≤ a ∧ a ≤ 4/3 :=
by
  sorry

end NUMINAMATH_GPT_bounded_sequence_range_l1052_105205


namespace NUMINAMATH_GPT_therapy_charge_l1052_105287

-- Defining the conditions
variables (A F : ℝ)
variables (h1 : F = A + 25)
variables (h2 : F + 4*A = 250)

-- The statement we need to prove
theorem therapy_charge : F + A = 115 := 
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_therapy_charge_l1052_105287


namespace NUMINAMATH_GPT_complement_of_beta_l1052_105216

theorem complement_of_beta (α β : ℝ) (h₀ : α + β = 180) (h₁ : α > β) : 
  90 - β = 1/2 * (α - β) :=
by
  sorry

end NUMINAMATH_GPT_complement_of_beta_l1052_105216


namespace NUMINAMATH_GPT_solution_positive_then_opposite_signs_l1052_105236

theorem solution_positive_then_opposite_signs
  (a b : ℝ) (h : a ≠ 0) (x : ℝ) (hx : ax + b = 0) (x_pos : x > 0) :
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) :=
by
  sorry

end NUMINAMATH_GPT_solution_positive_then_opposite_signs_l1052_105236


namespace NUMINAMATH_GPT_apples_on_tree_l1052_105281

-- Defining initial number of apples on the tree
def initial_apples : ℕ := 4

-- Defining apples picked from the tree
def apples_picked : ℕ := 2

-- Defining new apples grown on the tree
def new_apples : ℕ := 3

-- Prove the final number of apples on the tree is 5
theorem apples_on_tree : initial_apples - apples_picked + new_apples = 5 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_apples_on_tree_l1052_105281


namespace NUMINAMATH_GPT_part1_part2_1_part2_2_l1052_105251

theorem part1 (n : ℚ) :
  (2 / 2 + n / 5 = (2 + n) / 7) → n = -25 / 2 :=
by sorry

theorem part2_1 (m n : ℚ) :
  (m / 2 + n / 5 = (m + n) / 7) → m = -4 / 25 * n :=
by sorry

theorem part2_2 (m n: ℚ) :
  (m = -4 / 25 * n) → (25 * m + n = 6) → (m = 8 / 25 ∧ n = -2) :=
by sorry

end NUMINAMATH_GPT_part1_part2_1_part2_2_l1052_105251


namespace NUMINAMATH_GPT_range_of_m_l1052_105274

theorem range_of_m (m : ℝ) : (∃ x1 x2 x3 : ℝ, 
    (x1 - 1) * (x1^2 - 2*x1 + m) = 0 ∧ 
    (x2 - 1) * (x2^2 - 2*x2 + m) = 0 ∧ 
    (x3 - 1) * (x3^2 - 2*x3 + m) = 0 ∧ 
    x1 = 1 ∧ 
    x2^2 - 2*x2 + m = 0 ∧ 
    x3^2 - 2*x3 + m = 0 ∧ 
    x1 + x2 > x3 ∧ x1 + x3 > x2 ∧ x2 + x3 > x1 ∧ 
    x1 > 0 ∧ x2 > 0 ∧ x3 > 0) ↔ 3 / 4 < m ∧ m ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1052_105274


namespace NUMINAMATH_GPT_tan_subtraction_l1052_105259

theorem tan_subtraction (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_tan_subtraction_l1052_105259


namespace NUMINAMATH_GPT_solve_inequality_l1052_105276

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem solve_inequality {x : ℝ} (hx : 0 < x) : 
  f (Real.log x / Real.log 2) < f 2 ↔ (0 < x ∧ x < 1) ∨ (4 < x) :=
by
sorry

end NUMINAMATH_GPT_solve_inequality_l1052_105276


namespace NUMINAMATH_GPT_largest_sum_fraction_l1052_105258

theorem largest_sum_fraction :
  max (max (max (max ((1/3) + (1/2)) ((1/3) + (1/4))) ((1/3) + (1/5))) ((1/3) + (1/7))) ((1/3) + (1/9)) = 5/6 :=
by
  sorry

end NUMINAMATH_GPT_largest_sum_fraction_l1052_105258


namespace NUMINAMATH_GPT_h_evaluation_l1052_105241

variables {a b c : ℝ}

-- Definitions and conditions
def p (x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 3 * b * x + 4 * c
def h (x : ℝ) : ℝ := sorry -- Definition of h(x) in terms of the roots of p(x)

theorem h_evaluation (ha : a < b) (hb : b < c) : h 2 = (2 + 2 * a + 3 * b + c) / (c^2) :=
sorry

end NUMINAMATH_GPT_h_evaluation_l1052_105241


namespace NUMINAMATH_GPT_max_value_exponential_and_power_functions_l1052_105243

variable (a b : ℝ)

-- Given conditions
axiom condition : 0 < b ∧ b < a ∧ a < 1

-- Problem statement
theorem max_value_exponential_and_power_functions : 
  a^b = max (max (a^b) (b^a)) (max (a^a) (b^b)) :=
by
  sorry

end NUMINAMATH_GPT_max_value_exponential_and_power_functions_l1052_105243


namespace NUMINAMATH_GPT_part1_part2_l1052_105252

variable (x : ℝ)

def A := {x : ℝ | 1 < x ∧ x < 3}
def B := {x : ℝ | x < -3 ∨ 2 < x}

theorem part1 : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

theorem part2 (a b : ℝ) : (∀ x, 2 < x ∧ x < 3 → x^2 + a * x + b < 0) → a = -5 ∧ b = 6 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1052_105252


namespace NUMINAMATH_GPT_siamese_cats_initial_l1052_105203

theorem siamese_cats_initial (S : ℕ) (h1 : 20 + S - 20 = 12) : S = 12 :=
by
  sorry

end NUMINAMATH_GPT_siamese_cats_initial_l1052_105203


namespace NUMINAMATH_GPT_triangle_inequality_l1052_105257

variable (a b c p : ℝ)
variable (triangle : a + b > c ∧ a + c > b ∧ b + c > a)
variable (h_p : p = (a + b + c) / 2)

theorem triangle_inequality : 2 * Real.sqrt ((p - b) * (p - c)) ≤ a :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1052_105257


namespace NUMINAMATH_GPT_third_angle_is_90_triangle_is_right_l1052_105291

-- Define the given angles
def angle1 : ℝ := 56
def angle2 : ℝ := 34

-- Define the sum of angles in a triangle
def angle_sum : ℝ := 180

-- Define the third angle
def third_angle : ℝ := angle_sum - angle1 - angle2

-- Prove that the third angle is 90 degrees
theorem third_angle_is_90 : third_angle = 90 := by
  sorry

-- Define the type of the triangle based on the largest angle
def is_right_triangle : Prop := third_angle = 90

-- Prove that the triangle is a right triangle
theorem triangle_is_right : is_right_triangle := by
  sorry

end NUMINAMATH_GPT_third_angle_is_90_triangle_is_right_l1052_105291


namespace NUMINAMATH_GPT_total_weight_of_packages_l1052_105204

theorem total_weight_of_packages (x y z w : ℕ) (h1 : x + y + z = 150) (h2 : y + z + w = 160) (h3 : z + w + x = 170) :
  x + y + z + w = 160 :=
by sorry

end NUMINAMATH_GPT_total_weight_of_packages_l1052_105204


namespace NUMINAMATH_GPT_prism_volume_l1052_105232

noncomputable def volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : c * a = 84) : 
  abs (volume a b c - 594) < 1 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_prism_volume_l1052_105232


namespace NUMINAMATH_GPT_find_tents_l1052_105278

theorem find_tents (x y : ℕ) (hx : x + y = 600) (hy : 1700 * x + 1300 * y = 940000) : x = 400 ∧ y = 200 :=
by
  sorry

end NUMINAMATH_GPT_find_tents_l1052_105278


namespace NUMINAMATH_GPT_sum_of_four_interior_edges_l1052_105285

-- Define the given conditions
def is_two_inch_frame (w : ℕ) := w = 2
def frame_area (A : ℕ) := A = 68
def outer_edge_length (L : ℕ) := L = 15

-- Define the inner dimensions calculation function
def inner_dimensions (outerL outerH frameW : ℕ) := 
  (outerL - 2 * frameW, outerH - 2 * frameW)

-- Define the final question in Lean 4 reflective of the equivalent proof problem
theorem sum_of_four_interior_edges (w A L y : ℕ) 
  (h1 : is_two_inch_frame w) 
  (h2 : frame_area A)
  (h3 : outer_edge_length L)
  (h4 : 15 * y - (15 - 2 * w) * (y - 2 * w) = A)
  : 2 * (15 - 2 * w) + 2 * (y - 2 * w) = 26 := 
sorry

end NUMINAMATH_GPT_sum_of_four_interior_edges_l1052_105285


namespace NUMINAMATH_GPT_general_term_of_arithmetic_seq_l1052_105214

variable {a : ℕ → ℕ} 
variable {S : ℕ → ℕ}

/-- Definition of sum of first n terms of an arithmetic sequence -/
def sum_of_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

/-- Definition of arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n, a (n + 1) = a n + d

theorem general_term_of_arithmetic_seq
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 6 = 12)
  (h3 : S 3 = 12)
  (h4 : sum_of_arithmetic_sequence S a) :
  ∀ n, a n = 2 * n := 
sorry

end NUMINAMATH_GPT_general_term_of_arithmetic_seq_l1052_105214


namespace NUMINAMATH_GPT_largest_unattainable_sum_l1052_105224

theorem largest_unattainable_sum (n : ℕ) : ∃ s, s = 12 * n^2 + 8 * n - 1 ∧ 
  ∀ (k : ℕ), k ≤ s → ¬ ∃ a b c d, 
    k = (6 * n + 1) * a + (6 * n + 3) * b + (6 * n + 5) * c + (6 * n + 7) * d := 
sorry

end NUMINAMATH_GPT_largest_unattainable_sum_l1052_105224


namespace NUMINAMATH_GPT_smallest_positive_leading_coefficient_l1052_105221

variable {a b c : ℚ} -- Define variables a, b, c that are rational numbers
variable (P : ℤ → ℚ) -- Define the polynomial P as a function from integers to rationals

-- State that P(x) is in the form of ax^2 + bx + c
def is_quadratic_polynomial (P : ℤ → ℚ) (a b c : ℚ) :=
  ∀ x : ℤ, P x = a * x^2 + b * x + c

-- State that P(x) takes integer values for all integer x
def takes_integer_values (P : ℤ → ℚ) :=
  ∀ x : ℤ, ∃ k : ℤ, P x = k

-- The statement we want to prove
theorem smallest_positive_leading_coefficient (h1 : is_quadratic_polynomial P a b c)
                                              (h2 : takes_integer_values P) :
  ∃ a : ℚ, 0 < a ∧ ∀ b c : ℚ, is_quadratic_polynomial P a b c → takes_integer_values P → a = 1/2 :=
sorry

end NUMINAMATH_GPT_smallest_positive_leading_coefficient_l1052_105221


namespace NUMINAMATH_GPT_solution_interval_log_eq_l1052_105230

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2) + x - 3

theorem solution_interval_log_eq (h_mono : ∀ x y, (0 < x ∧ x < y) → f x < f y)
  (h_f2 : f 2 = 0)
  (h_f3 : f 3 > 0) :
  ∃ x, (2 ≤ x ∧ x < 3 ∧ f x = 0) :=
by
  sorry

end NUMINAMATH_GPT_solution_interval_log_eq_l1052_105230


namespace NUMINAMATH_GPT_alpha_in_second_quadrant_l1052_105280

theorem alpha_in_second_quadrant (α : ℝ) 
  (h1 : Real.sin α > Real.cos α)
  (h2 : Real.sin α * Real.cos α < 0) : 
  (Real.sin α > 0) ∧ (Real.cos α < 0) :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_alpha_in_second_quadrant_l1052_105280


namespace NUMINAMATH_GPT_rebus_solution_l1052_105266

theorem rebus_solution (A B C D : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (equation : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
sorry

end NUMINAMATH_GPT_rebus_solution_l1052_105266


namespace NUMINAMATH_GPT_laura_five_dollar_bills_l1052_105200

theorem laura_five_dollar_bills (x y z : ℕ) 
  (h1 : x + y + z = 40) 
  (h2 : x + 2 * y + 5 * z = 120) 
  (h3 : y = 2 * x) : 
  z = 16 := 
by
  sorry

end NUMINAMATH_GPT_laura_five_dollar_bills_l1052_105200


namespace NUMINAMATH_GPT_player_A_min_score_l1052_105220

theorem player_A_min_score (A B : ℕ) (hA_first_move : A = 1) (hB_next_move : B = 2) : 
  ∃ k : ℕ, k = 64 :=
by
  sorry

end NUMINAMATH_GPT_player_A_min_score_l1052_105220


namespace NUMINAMATH_GPT_find_d_in_polynomial_l1052_105208

theorem find_d_in_polynomial 
  (a b c d : ℤ) 
  (x1 x2 x3 x4 : ℤ)
  (roots_neg : x1 < 0 ∧ x2 < 0 ∧ x3 < 0 ∧ x4 < 0)
  (h_poly : ∀ x, 
    (x + x1) * (x + x2) * (x + x3) * (x + x4) = 
    x^4 + a * x^3 + b * x^2 + c * x + d)
  (h_sum_eq : a + b + c + d = 2009) :
  d = (x1 * x2 * x3 * x4) :=
by
  sorry

end NUMINAMATH_GPT_find_d_in_polynomial_l1052_105208


namespace NUMINAMATH_GPT_esteban_exercise_each_day_l1052_105254

theorem esteban_exercise_each_day (natasha_daily : ℕ) (natasha_days : ℕ) (esteban_days : ℕ) (total_hours : ℕ) :
  let total_minutes := total_hours * 60
  let natasha_total := natasha_daily * natasha_days
  let esteban_total := total_minutes - natasha_total
  esteban_days ≠ 0 →
  natasha_daily = 30 →
  natasha_days = 7 →
  esteban_days = 9 →
  total_hours = 5 →
  esteban_total / esteban_days = 10 := 
by
  intros
  sorry

end NUMINAMATH_GPT_esteban_exercise_each_day_l1052_105254


namespace NUMINAMATH_GPT_measure_of_RPS_l1052_105222

-- Assume the elements of the problem
variables {Q R P S : Type}

-- Angles in degrees
def angle_PQS := 35
def angle_QPR := 80
def angle_PSQ := 40

-- Define the angles and the straight line condition
def QRS_straight_line : Prop := true  -- This definition is trivial for a straight line

-- Measure of angle QPS using sum of angles in triangle
noncomputable def angle_QPS : ℝ := 180 - angle_PQS - angle_PSQ

-- Measure of angle RPS derived from the previous steps
noncomputable def angle_RPS : ℝ := angle_QPS - angle_QPR

-- The statement of the problem in Lean
theorem measure_of_RPS : angle_RPS = 25 := by
  sorry

end NUMINAMATH_GPT_measure_of_RPS_l1052_105222


namespace NUMINAMATH_GPT_cannot_be_sum_of_six_consecutive_odds_l1052_105267

def is_sum_of_six_consecutive_odds (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (6 * k + 30)

theorem cannot_be_sum_of_six_consecutive_odds :
  ¬ is_sum_of_six_consecutive_odds 198 ∧ ¬ is_sum_of_six_consecutive_odds 390 := 
sorry

end NUMINAMATH_GPT_cannot_be_sum_of_six_consecutive_odds_l1052_105267


namespace NUMINAMATH_GPT_tori_passing_question_l1052_105262

def arithmetic_questions : ℕ := 20
def algebra_questions : ℕ := 40
def geometry_questions : ℕ := 40
def total_questions : ℕ := arithmetic_questions + algebra_questions + geometry_questions
def arithmetic_correct_pct : ℕ := 80
def algebra_correct_pct : ℕ := 50
def geometry_correct_pct : ℕ := 70
def passing_grade_pct : ℕ := 65

theorem tori_passing_question (questions_needed_to_pass : ℕ) (arithmetic_correct : ℕ) (algebra_correct : ℕ) (geometry_correct : ℕ) : 
  questions_needed_to_pass = 1 :=
by
  let arithmetic_correct : ℕ := (arithmetic_correct_pct * arithmetic_questions / 100)
  let algebra_correct : ℕ := (algebra_correct_pct * algebra_questions / 100)
  let geometry_correct : ℕ := (geometry_correct_pct * geometry_questions / 100)
  let total_correct : ℕ := arithmetic_correct + algebra_correct + geometry_correct
  let passing_grade : ℕ := (passing_grade_pct * total_questions / 100)
  let questions_needed_to_pass : ℕ := passing_grade - total_correct
  exact sorry

end NUMINAMATH_GPT_tori_passing_question_l1052_105262


namespace NUMINAMATH_GPT_final_share_approx_equal_l1052_105264

noncomputable def total_bill : ℝ := 211.0
noncomputable def number_of_people : ℝ := 6.0
noncomputable def tip_percentage : ℝ := 0.15
noncomputable def tip_amount : ℝ := tip_percentage * total_bill
noncomputable def total_amount : ℝ := total_bill + tip_amount
noncomputable def each_person_share : ℝ := total_amount / number_of_people

theorem final_share_approx_equal :
  abs (each_person_share - 40.44) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_final_share_approx_equal_l1052_105264


namespace NUMINAMATH_GPT_period_ending_time_l1052_105299

theorem period_ending_time (start_time : ℕ) (rain_duration : ℕ) (no_rain_duration : ℕ) (end_time : ℕ) :
  start_time = 8 ∧ rain_duration = 4 ∧ no_rain_duration = 5 ∧ end_time = 8 + rain_duration + no_rain_duration
  → end_time = 17 :=
by
  sorry

end NUMINAMATH_GPT_period_ending_time_l1052_105299


namespace NUMINAMATH_GPT_tax_diminished_percentage_l1052_105255

theorem tax_diminished_percentage (T C : ℝ) (x : ℝ) (h : (T * (1 - x / 100)) * (C * 1.10) = T * C * 0.88) :
  x = 20 :=
sorry

end NUMINAMATH_GPT_tax_diminished_percentage_l1052_105255


namespace NUMINAMATH_GPT_product_of_five_integers_l1052_105226

theorem product_of_five_integers (E F G H I : ℚ)
  (h1 : E + F + G + H + I = 110)
  (h2 : E / 2 = F / 3 ∧ F / 3 = G * 4 ∧ G * 4 = H * 2 ∧ H * 2 = I - 5) :
  E * F * G * H * I = 623400000 / 371293 := by
  sorry

end NUMINAMATH_GPT_product_of_five_integers_l1052_105226


namespace NUMINAMATH_GPT_city_division_exists_l1052_105253

-- Define the problem conditions and prove the required statement
theorem city_division_exists (squares : Type) (streets : squares → squares → Prop)
  (h_outgoing: ∀ (s : squares), ∃ t u : squares, streets s t ∧ streets s u) :
  ∃ (districts : squares → ℕ), (∀ (s t : squares), districts s ≠ districts t → streets s t ∨ streets t s) ∧
  (∀ (i j : ℕ), i ≠ j → ∀ (s t : squares), districts s = i → districts t = j → streets s t ∨ streets t s) ∧
  (∃ m : ℕ, m = 1014) :=
sorry

end NUMINAMATH_GPT_city_division_exists_l1052_105253


namespace NUMINAMATH_GPT_sum_slope_y_intercept_eq_l1052_105211

noncomputable def J : ℝ × ℝ := (0, 8)
noncomputable def K : ℝ × ℝ := (0, 0)
noncomputable def L : ℝ × ℝ := (10, 0)
noncomputable def G : ℝ × ℝ := ((J.1 + K.1) / 2, (J.2 + K.2) / 2)

theorem sum_slope_y_intercept_eq :
  let L := (10, 0)
  let G := (0, 4)
  let slope := (G.2 - L.2) / (G.1 - L.1)
  let y_intercept := G.2
  slope + y_intercept = 18 / 5 :=
by
  -- Place the conditions and setup here
  let L := (10, 0)
  let G := (0, 4)
  let slope := (G.2 - L.2) / (G.1 - L.1)
  let y_intercept := G.2
  -- Proof will be provided here eventually
  sorry

end NUMINAMATH_GPT_sum_slope_y_intercept_eq_l1052_105211


namespace NUMINAMATH_GPT_line_through_points_l1052_105290

theorem line_through_points (a b : ℝ)
  (h1 : 2 = a * 1 + b)
  (h2 : 14 = a * 5 + b) :
  a - b = 4 := 
  sorry

end NUMINAMATH_GPT_line_through_points_l1052_105290


namespace NUMINAMATH_GPT_roots_quadratic_eq_k_l1052_105261

theorem roots_quadratic_eq_k (k : ℝ) :
  (∀ x : ℝ, (5 * x^2 + 20 * x + k = 0) ↔ (x = (-20 + Real.sqrt 60) / 10 ∨ x = (-20 - Real.sqrt 60) / 10)) →
  k = 17 := by
  intro h
  sorry

end NUMINAMATH_GPT_roots_quadratic_eq_k_l1052_105261


namespace NUMINAMATH_GPT_stratified_sampling_elderly_employees_l1052_105217

-- Definitions for the conditions
def total_employees : ℕ := 430
def young_employees : ℕ := 160
def middle_aged_employees : ℕ := 180
def elderly_employees : ℕ := 90
def sample_young_employees : ℕ := 32

-- The property we want to prove
theorem stratified_sampling_elderly_employees :
  (sample_young_employees / young_employees) * elderly_employees = 18 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_elderly_employees_l1052_105217


namespace NUMINAMATH_GPT_arith_seq_seventh_term_l1052_105292

theorem arith_seq_seventh_term (a1 a25 : ℝ) (n : ℕ) (d : ℝ) (a7 : ℝ) :
  a1 = 5 → a25 = 80 → n = 25 → d = (a25 - a1) / (n - 1) → a7 = a1 + (7 - 1) * d → a7 = 23.75 :=
by
  intros h1 h2 h3 hd ha7
  sorry

end NUMINAMATH_GPT_arith_seq_seventh_term_l1052_105292


namespace NUMINAMATH_GPT_linear_function_change_l1052_105272

-- Define a linear function g
variable (g : ℝ → ℝ)

-- Define and assume the conditions
def linear_function (g : ℝ → ℝ) : Prop := ∀ x y, g (x + y) = g x + g y ∧ g (x - y) = g x - g y
def condition_g_at_points : Prop := g 3 - g (-1) = 20

-- Prove that g(10) - g(2) = 40
theorem linear_function_change (g : ℝ → ℝ) 
  (linear_g : linear_function g) 
  (cond_g : condition_g_at_points g) : 
  g 10 - g 2 = 40 :=
sorry

end NUMINAMATH_GPT_linear_function_change_l1052_105272


namespace NUMINAMATH_GPT_divisible_iff_exists_t_l1052_105298

theorem divisible_iff_exists_t (a b m α : ℤ) (h_coprime : Int.gcd a m = 1) (h_divisible : a * α + b ≡ 0 [ZMOD m]):
  ∀ x : ℤ, (a * x + b ≡ 0 [ZMOD m]) ↔ ∃ t : ℤ, x = α + m * t :=
sorry

end NUMINAMATH_GPT_divisible_iff_exists_t_l1052_105298


namespace NUMINAMATH_GPT_crayons_left_is_4_l1052_105286

-- Define initial number of crayons in the drawer
def initial_crayons : Nat := 7

-- Define number of crayons Mary took out
def taken_by_mary : Nat := 3

-- Define the number of crayons left in the drawer
def crayons_left (initial : Nat) (taken : Nat) : Nat :=
  initial - taken

-- Prove the number of crayons left in the drawer is 4
theorem crayons_left_is_4 : crayons_left initial_crayons taken_by_mary = 4 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end NUMINAMATH_GPT_crayons_left_is_4_l1052_105286


namespace NUMINAMATH_GPT_simplify_polynomial_subtraction_l1052_105265

variable (x : ℝ)

def P1 : ℝ := 2*x^6 + x^5 + 3*x^4 + x^3 + 5
def P2 : ℝ := x^6 + 2*x^5 + x^4 - x^3 + 7
def P3 : ℝ := x^6 - x^5 + 2*x^4 + 2*x^3 - 2

theorem simplify_polynomial_subtraction : (P1 x - P2 x) = P3 x :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_subtraction_l1052_105265


namespace NUMINAMATH_GPT_circle_k_range_l1052_105225

theorem circle_k_range {k : ℝ}
  (h : ∀ x y : ℝ, x^2 + y^2 - 2*x + y + k = 0) :
  k < 5 / 4 :=
sorry

end NUMINAMATH_GPT_circle_k_range_l1052_105225


namespace NUMINAMATH_GPT_bouquet_cost_l1052_105234

theorem bouquet_cost (c : ℕ) : (c / 25 = 30 / 15) → c = 50 := by
  sorry

end NUMINAMATH_GPT_bouquet_cost_l1052_105234


namespace NUMINAMATH_GPT_find_multiplier_l1052_105244

theorem find_multiplier (x : ℕ) (h₁ : 3 * x = (26 - x) + 26) (h₂ : x = 13) : 3 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_multiplier_l1052_105244


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_l1052_105271

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_l1052_105271


namespace NUMINAMATH_GPT_abs_pi_sub_abs_pi_sub_three_l1052_105215

theorem abs_pi_sub_abs_pi_sub_three (h : Real.pi > 3) : 
  abs (Real.pi - abs (Real.pi - 3)) = 2 * Real.pi - 3 := 
by
  sorry

end NUMINAMATH_GPT_abs_pi_sub_abs_pi_sub_three_l1052_105215


namespace NUMINAMATH_GPT_Toph_caught_12_fish_l1052_105240

-- Define the number of fish each person caught
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def average_fish : ℕ := 8
def num_people : ℕ := 3

-- The total number of fish based on the average
def total_fish : ℕ := average_fish * num_people

-- Define the number of fish Toph caught
def Toph_fish : ℕ := total_fish - Aang_fish - Sokka_fish

-- Prove that Toph caught the correct number of fish
theorem Toph_caught_12_fish : Toph_fish = 12 := sorry

end NUMINAMATH_GPT_Toph_caught_12_fish_l1052_105240


namespace NUMINAMATH_GPT_unripe_oranges_per_day_l1052_105260

/-
Problem: Prove that if after 6 days, they will have 390 sacks of unripe oranges, then the number of sacks of unripe oranges harvested per day is 65.
-/

theorem unripe_oranges_per_day (total_sacks : ℕ) (days : ℕ) (harvest_per_day : ℕ)
  (h1 : days = 6)
  (h2 : total_sacks = 390)
  (h3 : harvest_per_day = total_sacks / days) :
  harvest_per_day = 65 :=
by
  sorry

end NUMINAMATH_GPT_unripe_oranges_per_day_l1052_105260


namespace NUMINAMATH_GPT_a_lt_sqrt3b_l1052_105206

open Int

theorem a_lt_sqrt3b (a b : ℤ) (h1 : a > b) (h2 : b > 1) 
    (h3 : a + b ∣ a * b + 1) (h4 : a - b ∣ a * b - 1) : a < sqrt 3 * b :=
  sorry

end NUMINAMATH_GPT_a_lt_sqrt3b_l1052_105206


namespace NUMINAMATH_GPT_mooncake_packaging_problem_l1052_105219

theorem mooncake_packaging_problem
  (x y : ℕ)
  (L : ℕ := 9)
  (S : ℕ := 4)
  (M : ℕ := 35)
  (h1 : L = 9)
  (h2 : S = 4)
  (h3 : M = 35) :
  9 * x + 4 * y = 35 ∧ x + y = 5 := 
by
  sorry

end NUMINAMATH_GPT_mooncake_packaging_problem_l1052_105219


namespace NUMINAMATH_GPT_sin_45_eq_sqrt2_div_2_l1052_105201

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = (Real.sqrt 2) / 2 := 
  sorry

end NUMINAMATH_GPT_sin_45_eq_sqrt2_div_2_l1052_105201


namespace NUMINAMATH_GPT_circles_radii_divide_regions_l1052_105245

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end NUMINAMATH_GPT_circles_radii_divide_regions_l1052_105245


namespace NUMINAMATH_GPT_exists_prime_q_not_div_n_p_minus_p_l1052_105202

variable (p : ℕ) [Fact (Nat.Prime p)]

theorem exists_prime_q_not_div_n_p_minus_p :
  ∃ q : ℕ, Nat.Prime q ∧ q ≠ p ∧ ∀ n : ℕ, ¬ q ∣ (n ^ p - p) :=
sorry

end NUMINAMATH_GPT_exists_prime_q_not_div_n_p_minus_p_l1052_105202


namespace NUMINAMATH_GPT_employees_count_l1052_105284

theorem employees_count (E M : ℝ) (h1 : M = 0.99 * E) (h2 : M - 299.9999999999997 = 0.98 * E) :
  E = 30000 :=
by sorry

end NUMINAMATH_GPT_employees_count_l1052_105284


namespace NUMINAMATH_GPT_simplify_div_expression_evaluate_at_2_l1052_105229

variable (a : ℝ)

theorem simplify_div_expression (h0 : a ≠ 0) (h1 : a ≠ 1) :
  (1 - 1 / a) / ((a^2 - 2 * a + 1) / a) = 1 / (a - 1) :=
by
  sorry

theorem evaluate_at_2 : (1 - 1 / 2) / ((2^2 - 2 * 2 + 1) / 2) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_div_expression_evaluate_at_2_l1052_105229


namespace NUMINAMATH_GPT_directrix_of_parabola_l1052_105268

theorem directrix_of_parabola :
  ∀ (a h k : ℝ), (a < 0) → (∀ x, y = a * (x - h) ^ 2 + k) → (h = 0) → (k = 0) → 
  (directrix = 1 / (4 * a)) → (directrix = 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1052_105268


namespace NUMINAMATH_GPT_ratio_Sarah_to_Eli_is_2_l1052_105250

variable (Kaylin_age : ℕ := 33)
variable (Freyja_age : ℕ := 10)
variable (Eli_age : ℕ := Freyja_age + 9)
variable (Sarah_age : ℕ := Kaylin_age + 5)

theorem ratio_Sarah_to_Eli_is_2 : (Sarah_age : ℚ) / Eli_age = 2 := 
by 
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_ratio_Sarah_to_Eli_is_2_l1052_105250


namespace NUMINAMATH_GPT_distance_between_bus_stops_l1052_105283

theorem distance_between_bus_stops (d : ℕ) (unit : String) 
  (h: d = 3000 ∧ unit = "meters") : unit = "C" := 
by 
  sorry

end NUMINAMATH_GPT_distance_between_bus_stops_l1052_105283


namespace NUMINAMATH_GPT_peter_money_l1052_105212

theorem peter_money (cost_per_ounce : ℝ) (amount_bought : ℝ) (leftover_money : ℝ) (total_money : ℝ) :
  cost_per_ounce = 0.25 ∧ amount_bought = 6 ∧ leftover_money = 0.50 → total_money = 2 :=
by
  intros h
  let h1 := h.1
  let h2 := h.2.1
  let h3 := h.2.2
  sorry

end NUMINAMATH_GPT_peter_money_l1052_105212


namespace NUMINAMATH_GPT_cos_alpha_value_l1052_105218

noncomputable def cos_alpha (α : ℝ) : ℝ :=
  (3 - 4 * Real.sqrt 3) / 10

theorem cos_alpha_value (α : ℝ) (h1 : Real.sin (Real.pi / 6 + α) = 3 / 5) (h2 : Real.pi / 3 < α ∧ α < 5 * Real.pi / 6) :
  Real.cos α = cos_alpha α :=
by
sorry

end NUMINAMATH_GPT_cos_alpha_value_l1052_105218


namespace NUMINAMATH_GPT_assistant_stop_time_l1052_105227

-- Define the start time for the craftsman
def craftsmanStartTime : Nat := 8 * 60 -- in minutes

-- Craftsman starts at 8:00 AM and stops at 12:00 PM
def craftsmanEndTime : Nat := 12 * 60 -- in minutes

-- Craftsman produces 6 bracelets every 20 minutes
def craftsmanProductionPerMinute : Nat := 6 / 20

-- Assistant starts working at 9:00 AM
def assistantStartTime : Nat := 9 * 60 -- in minutes

-- Assistant produces 8 bracelets every 30 minutes
def assistantProductionPerMinute : Nat := 8 / 30

-- Total production duration for craftsman in minutes
def craftsmanWorkDuration : Nat := craftsmanEndTime - craftsmanStartTime

-- Total bracelets produced by craftsman
def totalBraceletsCraftsman : Nat := craftsmanWorkDuration * craftsmanProductionPerMinute

-- Time it takes for the assistant to produce the same number of bracelets
def assistantWorkDuration : Nat := totalBraceletsCraftsman / assistantProductionPerMinute

-- Time the assistant will stop working
def assistantEndTime : Nat := assistantStartTime + assistantWorkDuration

-- Convert time in minutes to hours and minutes format (output as a string for clarity)
def formatTime (timeInMinutes: Nat) : String :=
  let hours := timeInMinutes / 60
  let minutes := timeInMinutes % 60
  s! "{hours}:{if minutes < 10 then "0" else ""}{minutes}"

-- Proof goal: assistant will stop working at "13:30" (or 1:30 PM)
theorem assistant_stop_time : 
  formatTime assistantEndTime = "13:30" := 
by
  sorry

end NUMINAMATH_GPT_assistant_stop_time_l1052_105227


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1052_105228

theorem sufficient_but_not_necessary_condition (x : ℝ) : (0 < x ∧ x < 5) → |x - 2| < 3 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1052_105228


namespace NUMINAMATH_GPT_break_even_point_l1052_105293

def cost_of_commodity (a : ℝ) : ℝ := a

def profit_beginning_of_month (a : ℝ) : ℝ := 100 + (a + 100) * 0.024

def profit_end_of_month : ℝ := 115

theorem break_even_point (a : ℝ) : profit_end_of_month - profit_beginning_of_month a = 0 → a = 525 := 
by sorry

end NUMINAMATH_GPT_break_even_point_l1052_105293


namespace NUMINAMATH_GPT_evaluate_expression_l1052_105273

noncomputable def w := Complex.exp (2 * Real.pi * Complex.I / 11)

theorem evaluate_expression : (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) = 88573 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1052_105273
