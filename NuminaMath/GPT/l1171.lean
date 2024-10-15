import Mathlib

namespace NUMINAMATH_GPT_sum_of_edges_proof_l1171_117120

noncomputable def sum_of_edges (a r : ℝ) : ℝ :=
  let l1 := a / r
  let l2 := a
  let l3 := a * r
  4 * (l1 + l2 + l3)

theorem sum_of_edges_proof : 
  ∀ (a r : ℝ), 
  (a > 0 ∧ r > 0 ∧ (a / r) * a * (a * r) = 512 ∧ 2 * ((a^2 / r) + a^2 + a^2 * r) = 384) → sum_of_edges a r = 96 :=
by
  intros a r h
  -- We skip the proof here with sorry
  sorry

end NUMINAMATH_GPT_sum_of_edges_proof_l1171_117120


namespace NUMINAMATH_GPT_second_discount_is_5_percent_l1171_117156

noncomputable def salePriceSecondDiscount (initialPrice finalPrice priceAfterFirstDiscount: ℝ) : ℝ :=
  (initialPrice - priceAfterFirstDiscount) + (priceAfterFirstDiscount - finalPrice)

noncomputable def secondDiscountPercentage (initialPrice finalPrice priceAfterFirstDiscount: ℝ) : ℝ :=
  (priceAfterFirstDiscount - finalPrice) / priceAfterFirstDiscount * 100

theorem second_discount_is_5_percent :
  ∀ (initialPrice finalPrice priceAfterFirstDiscount: ℝ),
    initialPrice = 600 ∧
    finalPrice = 456 ∧
    priceAfterFirstDiscount = initialPrice * 0.80 →
    secondDiscountPercentage initialPrice finalPrice priceAfterFirstDiscount = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_second_discount_is_5_percent_l1171_117156


namespace NUMINAMATH_GPT_min_value_of_f_l1171_117119

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (9 / (1 - x))

theorem min_value_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : f x = 16 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l1171_117119


namespace NUMINAMATH_GPT_students_called_back_l1171_117155

theorem students_called_back (g b d t c : ℕ) (h1 : g = 9) (h2 : b = 14) (h3 : d = 21) (h4 : t = g + b) (h5 : c = t - d) : c = 2 := by 
  sorry

end NUMINAMATH_GPT_students_called_back_l1171_117155


namespace NUMINAMATH_GPT_correct_X_Y_Z_l1171_117113

def nucleotide_types (A_types C_types T_types : ℕ) : ℕ :=
  A_types + C_types + T_types

def lowest_stability_period := "interphase"

def separation_period := "late meiosis I or late meiosis II"

theorem correct_X_Y_Z :
  nucleotide_types 2 2 1 = 3 ∧ 
  lowest_stability_period = "interphase" ∧ 
  separation_period = "late meiosis I or late meiosis II" :=
by
  sorry

end NUMINAMATH_GPT_correct_X_Y_Z_l1171_117113


namespace NUMINAMATH_GPT_fixed_errors_correct_l1171_117128

-- Conditions
def total_lines_of_code : ℕ := 4300
def lines_per_debug : ℕ := 100
def errors_per_debug : ℕ := 3

-- Question: How many errors has she fixed so far?
theorem fixed_errors_correct :
  (total_lines_of_code / lines_per_debug) * errors_per_debug = 129 := 
by 
  sorry

end NUMINAMATH_GPT_fixed_errors_correct_l1171_117128


namespace NUMINAMATH_GPT_program_output_l1171_117140

theorem program_output (a : ℕ) (h : a = 3) : (if a < 10 then 2 * a else a * a) = 6 :=
by
  rw [h]
  norm_num

end NUMINAMATH_GPT_program_output_l1171_117140


namespace NUMINAMATH_GPT_max_value_of_quadratic_function_l1171_117188

noncomputable def quadratic_function (x : ℝ) : ℝ := -5*x^2 + 25*x - 15

theorem max_value_of_quadratic_function : ∃ x : ℝ, quadratic_function x = 750 :=
by
-- maximum value
sorry

end NUMINAMATH_GPT_max_value_of_quadratic_function_l1171_117188


namespace NUMINAMATH_GPT_solve_trigonometric_equation_l1171_117125

theorem solve_trigonometric_equation (x : ℝ) : 
  (2 * (Real.sin x)^6 + 2 * (Real.cos x)^6 - 3 * (Real.sin x)^4 - 3 * (Real.cos x)^4) = Real.cos (2 * x) ↔ 
  ∃ (k : ℤ), x = (π / 2) * (2 * k + 1) :=
sorry

end NUMINAMATH_GPT_solve_trigonometric_equation_l1171_117125


namespace NUMINAMATH_GPT_find_x_y_l1171_117117

theorem find_x_y (x y : ℝ) : 
  (x - 12) ^ 2 + (y - 13) ^ 2 + (x - y) ^ 2 = 1 / 3 ↔ (x = 37 / 3 ∧ y = 38 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_x_y_l1171_117117


namespace NUMINAMATH_GPT_units_digit_n_l1171_117184

theorem units_digit_n (m n : ℕ) (h1 : m * n = 31 ^ 6) (h2 : m % 10 = 9) : n % 10 = 2 := 
sorry

end NUMINAMATH_GPT_units_digit_n_l1171_117184


namespace NUMINAMATH_GPT_trisha_total_distance_l1171_117150

theorem trisha_total_distance :
  let d1 := 0.1111111111111111
  let d2 := 0.1111111111111111
  let d3 := 0.6666666666666666
  d1 + d2 + d3 = 0.8888888888888888 := 
by
  sorry

end NUMINAMATH_GPT_trisha_total_distance_l1171_117150


namespace NUMINAMATH_GPT_complement_of_A_in_U_eq_l1171_117103

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x ≥ Real.exp 1}
def complement_U_A : Set ℝ := {x | 0 < x ∧ x ≤ Real.exp 1}

theorem complement_of_A_in_U_eq : 
  (U \ A) = complement_U_A := 
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_eq_l1171_117103


namespace NUMINAMATH_GPT_problem_statement_l1171_117189

noncomputable def f : ℝ → ℝ := sorry  -- Define f as a noncomputable function to accommodate the problem constraints

variables (a : ℝ)

theorem problem_statement (periodic_f : ∀ x, f (x + 3) = f x)
    (odd_f : ∀ x, f (-x) = -f x)
    (ineq_f1 : f 1 < 1)
    (eq_f2 : f 2 = (2*a-1)/(a+1)) :
    a < -1 ∨ 0 < a :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1171_117189


namespace NUMINAMATH_GPT_area_of_walkways_l1171_117174

-- Define the dimensions of the individual flower bed
def flower_bed_width : ℕ := 8
def flower_bed_height : ℕ := 3

-- Define the number of rows and columns of flower beds
def rows_of_beds : ℕ := 4
def cols_of_beds : ℕ := 3

-- Define the width of the walkways
def walkway_width : ℕ := 2

-- Calculate the total width and height of the garden including walkways
def total_width : ℕ := (cols_of_beds * flower_bed_width) + (cols_of_beds + 1) * walkway_width
def total_height : ℕ := (rows_of_beds * flower_bed_height) + (rows_of_beds + 1) * walkway_width

-- Calculate the area of the garden including walkways
def total_area : ℕ := total_width * total_height

-- Calculate the total area of all the flower beds
def total_beds_area : ℕ := (rows_of_beds * cols_of_beds) * (flower_bed_width * flower_bed_height)

-- Prove the area of walkways
theorem area_of_walkways : total_area - total_beds_area = 416 := by
  sorry

end NUMINAMATH_GPT_area_of_walkways_l1171_117174


namespace NUMINAMATH_GPT_probability_of_first_four_cards_each_suit_l1171_117147

noncomputable def probability_first_four_different_suits : ℚ := 3 / 32

theorem probability_of_first_four_cards_each_suit :
  let n := 52
  let k := 5
  let suits := 4
  (probability_first_four_different_suits = (3 / 32)) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_first_four_cards_each_suit_l1171_117147


namespace NUMINAMATH_GPT_empty_tank_time_l1171_117149

-- Definitions based on problem conditions
def tank_full_fraction := 1 / 5
def pipeA_fill_time := 15
def pipeB_empty_time := 6

-- Derived definitions
def rate_of_pipeA := 1 / pipeA_fill_time
def rate_of_pipeB := 1 / pipeB_empty_time
def combined_rate := rate_of_pipeA - rate_of_pipeB 

-- The time to empty the tank when both pipes are open
def time_to_empty (initial_fraction : ℚ) (combined_rate : ℚ) : ℚ :=
  initial_fraction / -combined_rate

-- The main theorem to prove
theorem empty_tank_time
  (initial_fraction : ℚ := tank_full_fraction)
  (combined_rate : ℚ := combined_rate)
  (time : ℚ := time_to_empty initial_fraction combined_rate) :
  time = 2 :=
by
  sorry

end NUMINAMATH_GPT_empty_tank_time_l1171_117149


namespace NUMINAMATH_GPT_sum_of_fractions_decimal_equivalence_l1171_117185

theorem sum_of_fractions :
  (2 / 15 : ℚ) + (4 / 20) + (5 / 45) = 4 / 9 := 
sorry

theorem decimal_equivalence :
  (4 / 9 : ℚ) = 0.444 := 
sorry

end NUMINAMATH_GPT_sum_of_fractions_decimal_equivalence_l1171_117185


namespace NUMINAMATH_GPT_unique_solution_condition_l1171_117170

-- Define p and q as real numbers
variables (p q : ℝ)

-- The Lean statement to prove a unique solution when q ≠ 4
theorem unique_solution_condition : (∀ x : ℝ, (4 * x - 7 + p = q * x + 2) ↔ (q ≠ 4)) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_condition_l1171_117170


namespace NUMINAMATH_GPT_distinct_arrangements_apple_l1171_117137

theorem distinct_arrangements_apple : 
  let n := 5
  let freq_p := 2
  let freq_a := 1
  let freq_l := 1
  let freq_e := 1
  (Nat.factorial n) / (Nat.factorial freq_p * Nat.factorial freq_a * Nat.factorial freq_l * Nat.factorial freq_e) = 60 :=
by
  sorry

end NUMINAMATH_GPT_distinct_arrangements_apple_l1171_117137


namespace NUMINAMATH_GPT_find_a_l1171_117167

theorem find_a (a : ℝ) : 
  (∀ (i : ℂ), i^2 = -1 → (a * i / (2 - i) + 1 = 2 * i)) → a = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l1171_117167


namespace NUMINAMATH_GPT_smallest_n_for_cube_root_form_l1171_117115

theorem smallest_n_for_cube_root_form
  (m n : ℕ) (r : ℝ)
  (h_pos_n : n > 0)
  (h_pos_r : r > 0)
  (h_r_bound : r < 1/500)
  (h_m : m = (n + r)^3)
  (h_min_m : ∀ k : ℕ, k = (n + r)^3 → k ≥ m) :
  n = 13 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_smallest_n_for_cube_root_form_l1171_117115


namespace NUMINAMATH_GPT_find_k_intersection_l1171_117145

theorem find_k_intersection :
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), y = 2 * x + 3 → y = k * x + 1 → (x = 1 ∧ y = 5) → k = 4) :=
sorry

end NUMINAMATH_GPT_find_k_intersection_l1171_117145


namespace NUMINAMATH_GPT_find_initial_balance_l1171_117199

-- Define the initial balance
variable (X : ℝ)

-- Conditions
def balance_tripled (X : ℝ) : ℝ := 3 * X
def balance_after_withdrawal (X : ℝ) : ℝ := balance_tripled X - 250

-- The problem statement to prove
theorem find_initial_balance (h : balance_after_withdrawal X = 950) : X = 400 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_balance_l1171_117199


namespace NUMINAMATH_GPT_Xiaohong_wins_5_times_l1171_117175

theorem Xiaohong_wins_5_times :
  ∃ W L : ℕ, (3 * W - 2 * L = 1) ∧ (W + L = 12) ∧ W = 5 :=
by
  sorry

end NUMINAMATH_GPT_Xiaohong_wins_5_times_l1171_117175


namespace NUMINAMATH_GPT_solve_system_l1171_117138

noncomputable def system_solution (x y : ℝ) :=
  x + y = 20 ∧ x * y = 36

theorem solve_system :
  (system_solution 18 2) ∧ (system_solution 2 18) :=
  sorry

end NUMINAMATH_GPT_solve_system_l1171_117138


namespace NUMINAMATH_GPT_eval_expression_l1171_117139

theorem eval_expression : (Real.pi + 2023)^0 + 2 * Real.sin (45 * Real.pi / 180) - (1 / 2)^(-1 : ℤ) + abs (Real.sqrt 2 - 2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1171_117139


namespace NUMINAMATH_GPT_decreasing_function_implies_inequality_l1171_117166

theorem decreasing_function_implies_inequality (k b : ℝ) (h : ∀ x : ℝ, (2 * k + 1) * x + b = (2 * k + 1) * x + b) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k + 1) * x1 + b > (2 * k + 1) * x2 + b) → k < -1/2 :=
by sorry

end NUMINAMATH_GPT_decreasing_function_implies_inequality_l1171_117166


namespace NUMINAMATH_GPT_ganpat_paint_time_l1171_117106

theorem ganpat_paint_time (H_rate G_rate : ℝ) (together_time H_time : ℝ) (h₁ : H_time = 3)
  (h₂ : together_time = 2) (h₃ : H_rate = 1 / H_time) (h₄ : G_rate = 1 / G_time)
  (h₅ : 1/H_time + 1/G_rate = 1/together_time) : G_time = 3 := 
by 
  sorry

end NUMINAMATH_GPT_ganpat_paint_time_l1171_117106


namespace NUMINAMATH_GPT_students_on_bus_after_stops_l1171_117111

-- Definitions
def initial_students : ℕ := 10
def first_stop_off : ℕ := 3
def first_stop_on : ℕ := 2
def second_stop_off : ℕ := 1
def second_stop_on : ℕ := 4
def third_stop_off : ℕ := 2
def third_stop_on : ℕ := 3

-- Theorem statement
theorem students_on_bus_after_stops :
  let after_first_stop := initial_students - first_stop_off + first_stop_on
  let after_second_stop := after_first_stop - second_stop_off + second_stop_on
  let after_third_stop := after_second_stop - third_stop_off + third_stop_on
  after_third_stop = 13 := 
by
  sorry

end NUMINAMATH_GPT_students_on_bus_after_stops_l1171_117111


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1171_117176

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - 3 * x < 0) → (0 < x ∧ x < 4) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1171_117176


namespace NUMINAMATH_GPT_find_s_l1171_117190

theorem find_s (x y : Real -> Real) : 
  (x 2 = 2 ∧ y 2 = 5) ∧ 
  (x 6 = 6 ∧ y 6 = 17) ∧ 
  (x 10 = 10 ∧ y 10 = 29) ∧ 
  (∀ x, y x = 3 * x - 1) -> 
  (y 34 = 101) := 
by 
  sorry

end NUMINAMATH_GPT_find_s_l1171_117190


namespace NUMINAMATH_GPT_triangle_ABC_right_angle_l1171_117178

def point := (ℝ × ℝ)
def line (P: point) := P.1 = 5 ∨ ∃ a: ℝ, P.1 - 5 = a * (P.2 + 2)
def parabola (P: point) := P.2 ^ 2 = 4 * P.1
def perpendicular_slopes (k1 k2: ℝ) := k1 * k2 = -1

theorem triangle_ABC_right_angle (A B C: point) (P: point) 
  (hA: A = (1, 2))
  (hP: P = (5, -2))
  (h_line: line B ∧ line C)
  (h_parabola: parabola B ∧ parabola C):
  (∃ k_AB k_AC: ℝ, perpendicular_slopes k_AB k_AC) →
  ∃k_AB k_AC: ℝ, k_AB * k_AC = -1 :=
by sorry

end NUMINAMATH_GPT_triangle_ABC_right_angle_l1171_117178


namespace NUMINAMATH_GPT_inequality_solution_l1171_117141

theorem inequality_solution :
  {x : ℝ | (3 * x - 8) * (x - 4) / (x - 1) ≥ 0 } = { x : ℝ | x < 1 } ∪ { x : ℝ | x ≥ 4 } :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_solution_l1171_117141


namespace NUMINAMATH_GPT_curve1_line_and_circle_curve2_two_points_l1171_117144

-- Define the first condition: x(x^2 + y^2 - 4) = 0
def curve1 (x y : ℝ) : Prop := x * (x^2 + y^2 - 4) = 0

-- Define the second condition: x^2 + (x^2 + y^2 - 4)^2 = 0
def curve2 (x y : ℝ) : Prop := x^2 + (x^2 + y^2 - 4)^2 = 0

-- The corresponding theorem statements
theorem curve1_line_and_circle : ∀ x y : ℝ, curve1 x y ↔ (x = 0 ∨ (x^2 + y^2 = 4)) := 
sorry 

theorem curve2_two_points : ∀ x y : ℝ, curve2 x y ↔ (x = 0 ∧ (y = 2 ∨ y = -2)) := 
sorry 

end NUMINAMATH_GPT_curve1_line_and_circle_curve2_two_points_l1171_117144


namespace NUMINAMATH_GPT_number_of_blue_tiles_is_16_l1171_117172

def length_of_floor : ℕ := 20
def breadth_of_floor : ℕ := 10
def tile_length : ℕ := 2

def total_tiles : ℕ := (length_of_floor / tile_length) * (breadth_of_floor / tile_length)

def black_tiles : ℕ :=
  let rows_length := 2 * (length_of_floor / tile_length)
  let rows_breadth := 2 * (breadth_of_floor / tile_length)
  (rows_length + rows_breadth) - 4

def remaining_tiles : ℕ := total_tiles - black_tiles
def white_tiles : ℕ := remaining_tiles / 3
def blue_tiles : ℕ := remaining_tiles - white_tiles

theorem number_of_blue_tiles_is_16 :
  blue_tiles = 16 :=
by
  sorry

end NUMINAMATH_GPT_number_of_blue_tiles_is_16_l1171_117172


namespace NUMINAMATH_GPT_amount_spent_on_marbles_l1171_117158

/-- A theorem to determine the amount Mike spent on marbles. -/
theorem amount_spent_on_marbles 
  (total_amount : ℝ) 
  (cost_football : ℝ) 
  (cost_baseball : ℝ) 
  (total_amount_eq : total_amount = 20.52)
  (cost_football_eq : cost_football = 4.95)
  (cost_baseball_eq : cost_baseball = 6.52) :
  ∃ (cost_marbles : ℝ), cost_marbles = total_amount - (cost_football + cost_baseball) 
  ∧ cost_marbles = 9.05 := 
by
  sorry

end NUMINAMATH_GPT_amount_spent_on_marbles_l1171_117158


namespace NUMINAMATH_GPT_find_y_l1171_117127

theorem find_y (x y: ℤ) (h1: x^2 - 3 * x + 2 = y + 6) (h2: x = -4) : y = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1171_117127


namespace NUMINAMATH_GPT_temperature_below_75_l1171_117107

theorem temperature_below_75
  (T : ℝ)
  (H1 : ∀ T, T ≥ 75 → swimming_area_open)
  (H2 : ¬swimming_area_open) : 
  T < 75 :=
sorry

end NUMINAMATH_GPT_temperature_below_75_l1171_117107


namespace NUMINAMATH_GPT_solve_system_of_equations_l1171_117110

theorem solve_system_of_equations :
  ∀ x y z : ℝ,
  (3 * x * y - 5 * y * z - x * z = 3 * y) →
  (x * y + y * z = -y) →
  (-5 * x * y + 4 * y * z + x * z = -4 * y) →
  (x = 2 ∧ y = -1 / 3 ∧ z = -3) ∨ 
  (y = 0 ∧ z = 0) ∨ 
  (x = 0 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1171_117110


namespace NUMINAMATH_GPT_inequality_solution_l1171_117135

theorem inequality_solution (x : ℝ) : x^2 - 2 * x - 5 > 2 * x ↔ x > 5 ∨ x < -1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1171_117135


namespace NUMINAMATH_GPT_largest_n_value_l1171_117198

theorem largest_n_value (n : ℕ) (h : (1 / 5 : ℝ) + (n / 8 : ℝ) + 1 < 2) : n ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_value_l1171_117198


namespace NUMINAMATH_GPT_largest_and_smallest_multiples_of_12_l1171_117195

theorem largest_and_smallest_multiples_of_12 (k : ℤ) (n₁ n₂ : ℤ) (h₁ : k = -150) (h₂ : n₁ = -156) (h₃ : n₂ = -144) :
  (∃ m1 : ℤ, m1 * 12 = n₁ ∧ n₁ < k) ∧ (¬ (∃ m2 : ℤ, m2 * 12 = n₂ ∧ n₂ > k ∧ ∃ m2' : ℤ, m2' * 12 > k ∧ m2' * 12 < n₂)) :=
by
  sorry

end NUMINAMATH_GPT_largest_and_smallest_multiples_of_12_l1171_117195


namespace NUMINAMATH_GPT_vector_simplification_l1171_117118

-- Define vectors AB, CD, AC, and BD
variables {V : Type*} [AddCommGroup V]

-- Given vectors
variables (AB CD AC BD : V)

-- Theorem to be proven
theorem vector_simplification :
  (AB - CD) - (AC - BD) = (0 : V) :=
sorry

end NUMINAMATH_GPT_vector_simplification_l1171_117118


namespace NUMINAMATH_GPT_average_of_hidden_primes_l1171_117191

theorem average_of_hidden_primes (p₁ p₂ : ℕ) (h₁ : Nat.Prime p₁) (h₂ : Nat.Prime p₂) (h₃ : p₁ + 37 = p₂ + 53) : 
  (p₁ + p₂) / 2 = 11 := 
by
  sorry

end NUMINAMATH_GPT_average_of_hidden_primes_l1171_117191


namespace NUMINAMATH_GPT_number_of_violinists_l1171_117104

open Nat

/-- There are 3 violinists in the orchestra, based on given conditions. -/
theorem number_of_violinists
  (total : ℕ)
  (percussion : ℕ)
  (brass : ℕ)
  (cellist : ℕ)
  (contrabassist : ℕ)
  (woodwinds : ℕ)
  (maestro : ℕ)
  (total_eq : total = 21)
  (percussion_eq : percussion = 1)
  (brass_eq : brass = 7)
  (strings_excluding_violinists : ℕ)
  (cellist_eq : cellist = 1)
  (contrabassist_eq : contrabassist = 1)
  (woodwinds_eq : woodwinds = 7)
  (maestro_eq : maestro = 1) :
  (total - (percussion + brass + (cellist + contrabassist) + woodwinds + maestro)) = 3 := 
by
  sorry

end NUMINAMATH_GPT_number_of_violinists_l1171_117104


namespace NUMINAMATH_GPT_intersection_union_complement_union_l1171_117179

open Set

variable (U : Set ℝ) (A B : Set ℝ)
variable [Inhabited (Set ℝ)]

noncomputable def setA : Set ℝ := { x : ℝ | abs (x - 2) > 1 }
noncomputable def setB : Set ℝ := { x : ℝ | x ≥ 0 }

theorem intersection (U : Set ℝ) : 
  (setA ∩ setB) = { x : ℝ | (0 < x ∧ x < 1) ∨ x > 3 } := 
  sorry

theorem union (U : Set ℝ) : 
  (setA ∪ setB) = univ := 
  sorry

theorem complement_union (U : Set ℝ) : 
  ((U \ setA) ∪ setB) = { x : ℝ | x ≥ 0 } := 
  sorry

end NUMINAMATH_GPT_intersection_union_complement_union_l1171_117179


namespace NUMINAMATH_GPT_find_larger_integer_l1171_117164

-- Defining the problem statement with the given conditions
theorem find_larger_integer (x : ℕ) (h : (x + 6) * 2 = 4 * x) : 4 * x = 24 :=
sorry

end NUMINAMATH_GPT_find_larger_integer_l1171_117164


namespace NUMINAMATH_GPT_stable_performance_l1171_117132

/-- The variance of student A's scores is 0.4 --/
def variance_A : ℝ := 0.4

/-- The variance of student B's scores is 0.3 --/
def variance_B : ℝ := 0.3

/-- Prove that student B has more stable performance given the variances --/
theorem stable_performance (h1 : variance_A = 0.4) (h2 : variance_B = 0.3) : variance_B < variance_A :=
by
  rw [h1, h2]
  exact sorry

end NUMINAMATH_GPT_stable_performance_l1171_117132


namespace NUMINAMATH_GPT_distance_24_km_l1171_117134

noncomputable def distance_between_house_and_school (D : ℝ) :=
  let speed_to_school := 6
  let speed_to_home := 4
  let total_time := 10
  total_time = (D / speed_to_school) + (D / speed_to_home)

theorem distance_24_km : ∃ D : ℝ, distance_between_house_and_school D ∧ D = 24 :=
by
  use 24
  unfold distance_between_house_and_school
  sorry

end NUMINAMATH_GPT_distance_24_km_l1171_117134


namespace NUMINAMATH_GPT_solve_for_x_l1171_117124

theorem solve_for_x (x : ℝ) : 4 * x - 8 + 3 * x = 12 + 5 * x → x = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1171_117124


namespace NUMINAMATH_GPT_jane_bought_two_bagels_l1171_117194

variable (b m d k : ℕ)

def problem_conditions : Prop :=
  b + m + d = 6 ∧ 
  (60 * b + 45 * m + 30 * d) = 100 * k

theorem jane_bought_two_bagels (hb : problem_conditions b m d k) : b = 2 :=
  sorry

end NUMINAMATH_GPT_jane_bought_two_bagels_l1171_117194


namespace NUMINAMATH_GPT_correlations_are_1_3_4_l1171_117165

def relation1 : Prop := ∃ (age wealth : ℝ), true
def relation2 : Prop := ∀ (point : ℝ × ℝ), ∃ (coords : ℝ × ℝ), coords = point
def relation3 : Prop := ∃ (yield : ℝ) (climate : ℝ), true
def relation4 : Prop := ∃ (diameter height : ℝ), true
def relation5 : Prop := ∃ (student : Type) (school : Type), true

theorem correlations_are_1_3_4 :
  (relation1 ∨ relation3 ∨ relation4) ∧ ¬ (relation2 ∨ relation5) :=
sorry

end NUMINAMATH_GPT_correlations_are_1_3_4_l1171_117165


namespace NUMINAMATH_GPT_y_axis_symmetry_l1171_117154

theorem y_axis_symmetry (x y : ℝ) (P : ℝ × ℝ) (hx : P = (-5, 3)) : 
  (P.1 = -5 ∧ P.2 = 3) → (P.1 * -1, P.2) = (5, 3) :=
by
  intro h
  rw [hx]
  simp [Neg.neg, h]
  sorry

end NUMINAMATH_GPT_y_axis_symmetry_l1171_117154


namespace NUMINAMATH_GPT_joyce_apples_l1171_117186

theorem joyce_apples (initial_apples given_apples remaining_apples : ℕ) (h1 : initial_apples = 75) (h2 : given_apples = 52) (h3 : remaining_apples = initial_apples - given_apples) : remaining_apples = 23 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_joyce_apples_l1171_117186


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1171_117102

theorem necessary_but_not_sufficient_condition (a : ℝ)
    (h : -2 ≤ a ∧ a ≤ 2)
    (hq : ∃ x y : ℂ, x ≠ y ∧ (x ^ 2 + (a : ℂ) * x + 1 = 0) ∧ (y ^ 2 + (a : ℂ) * y + 1 = 0)) :
    ∃ z : ℂ, z ^ 2 + (a : ℂ) * z + 1 = 0 ∧ (¬ ∀ b, -2 < b ∧ b < 2 → b = a) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1171_117102


namespace NUMINAMATH_GPT_seashells_collected_l1171_117129

theorem seashells_collected (x y z : ℕ) (hyp : x + y / 2 + z + 5 = 76) : x + y + z = 71 := 
by {
  sorry
}

end NUMINAMATH_GPT_seashells_collected_l1171_117129


namespace NUMINAMATH_GPT_martha_weight_l1171_117169

theorem martha_weight :
  ∀ (Bridget_weight : ℕ) (difference : ℕ) (Martha_weight : ℕ),
  Bridget_weight = 39 → difference = 37 →
  Bridget_weight = Martha_weight + difference →
  Martha_weight = 2 :=
by
  intros Bridget_weight difference Martha_weight hBridget hDifference hRelation
  sorry

end NUMINAMATH_GPT_martha_weight_l1171_117169


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l1171_117197

theorem trigonometric_identity_proof :
  3.438 * (Real.sin (84 * Real.pi / 180)) * (Real.sin (24 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * (Real.sin (12 * Real.pi / 180)) = 1 / 16 :=
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l1171_117197


namespace NUMINAMATH_GPT_find_lunch_break_duration_l1171_117161

def lunch_break_duration : ℝ → ℝ → ℝ → ℝ
  | s, a, L => L

theorem find_lunch_break_duration (s a L : ℝ) :
  (8 - L) * (s + a) = 0.6 ∧ (6.4 - L) * a = 0.28 ∧ (9.6 - L) * s = 0.12 →
  lunch_break_duration s a L = 1 :=
  by
    sorry

end NUMINAMATH_GPT_find_lunch_break_duration_l1171_117161


namespace NUMINAMATH_GPT_range_of_a_l1171_117108

open Real

theorem range_of_a (a : ℝ) (H : ∀ b : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs (x^2 + a * x + b) ≥ 1)) : a ≥ 1 ∨ a ≤ -3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1171_117108


namespace NUMINAMATH_GPT_solve_for_x_l1171_117131

theorem solve_for_x (x y : ℚ) (h1 : 3 * x - y = 7) (h2 : x + 3 * y = 2) : x = 23 / 10 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_solve_for_x_l1171_117131


namespace NUMINAMATH_GPT_fraction_sum_to_decimal_l1171_117126

theorem fraction_sum_to_decimal :
  (3 / 20 : ℝ) + (5 / 200 : ℝ) + (7 / 2000 : ℝ) = 0.1785 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_sum_to_decimal_l1171_117126


namespace NUMINAMATH_GPT_region_ratio_l1171_117136

theorem region_ratio (side_length : ℝ) (s r : ℝ) 
  (h1 : side_length = 2)
  (h2 : s = (1 / 2) * (1 : ℝ) * (1 : ℝ))
  (h3 : r = (1 / 2) * (Real.sqrt 2) * (Real.sqrt 2)) :
  r / s = 2 :=
by
  sorry

end NUMINAMATH_GPT_region_ratio_l1171_117136


namespace NUMINAMATH_GPT_right_triangle_property_l1171_117112

-- Variables representing the lengths of the sides and the height of the right triangle
variables (a b c h : ℝ)

-- Hypotheses from the conditions
-- 1. a and b are the lengths of the legs of the right triangle
-- 2. c is the length of the hypotenuse
-- 3. h is the height to the hypotenuse
-- Given equation: 1/2 * a * b = 1/2 * c * h
def given_equation (a b c h : ℝ) : Prop := (1 / 2) * a * b = (1 / 2) * c * h

-- The theorem to prove
theorem right_triangle_property (a b c h : ℝ) (h_eq : given_equation a b c h) : (1 / a^2 + 1 / b^2) = 1 / h^2 :=
sorry

end NUMINAMATH_GPT_right_triangle_property_l1171_117112


namespace NUMINAMATH_GPT_part_I_part_II_part_III_no_zeros_part_III_one_zero_part_III_two_zeros_l1171_117193

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a / x + Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 1 - a / (x^2) + 1 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f' x a - x

-- Problem (I)
theorem part_I (a : ℝ) : f' 1 a = 0 → a = 2 := sorry

-- Problem (II)
theorem part_II (a : ℝ) : (∀ x, 1 < x ∧ x < 2 → f' x a ≥ 0) → a ≤ 2 := sorry

-- Problem (III)
theorem part_III_no_zeros (a : ℝ) : a > 1 → ∀ x, g x a ≠ 0 := sorry
theorem part_III_one_zero (a : ℝ) : (a = 1 ∨ a ≤ 0) → ∃! x, g x a = 0 := sorry
theorem part_III_two_zeros (a : ℝ) : 0 < a ∧ a < 1 → ∃ x1 x2, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0 := sorry

end NUMINAMATH_GPT_part_I_part_II_part_III_no_zeros_part_III_one_zero_part_III_two_zeros_l1171_117193


namespace NUMINAMATH_GPT_task_completion_l1171_117109

theorem task_completion (x y z : ℝ) 
  (h1 : 1 / x + 1 / y = 1 / 2)
  (h2 : 1 / y + 1 / z = 1 / 4)
  (h3 : 1 / z + 1 / x = 5 / 12) :
  x = 3 := 
sorry

end NUMINAMATH_GPT_task_completion_l1171_117109


namespace NUMINAMATH_GPT_complete_the_square_l1171_117157

theorem complete_the_square (z : ℤ) : 
    z^2 - 6*z + 17 = (z - 3)^2 + 8 :=
sorry

end NUMINAMATH_GPT_complete_the_square_l1171_117157


namespace NUMINAMATH_GPT_intersection_A_B_range_m_l1171_117146

-- Definitions for Sets A, B, and C
def SetA : Set ℝ := { x | -2 ≤ x ∧ x < 5 }
def SetB : Set ℝ := { x | 3 * x - 5 ≥ x - 1 }
def SetC (m : ℝ) : Set ℝ := { x | -x + m > 0 }

-- Problem 1: Prove \( A \cap B = \{ x \mid 2 \leq x < 5 \} \)
theorem intersection_A_B : SetA ∩ SetB = { x : ℝ | 2 ≤ x ∧ x < 5 } :=
by
  sorry

-- Problem 2: Prove \( m \in [5, +\infty) \) given \( A \cup C = C \)
theorem range_m (m : ℝ) : (SetA ∪ SetC m = SetC m) → m ∈ Set.Ici 5 :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_range_m_l1171_117146


namespace NUMINAMATH_GPT_map_length_to_reality_l1171_117116

def scale : ℝ := 500
def length_map : ℝ := 7.2
def length_actual : ℝ := 3600

theorem map_length_to_reality : length_actual = length_map * scale :=
by
  sorry

end NUMINAMATH_GPT_map_length_to_reality_l1171_117116


namespace NUMINAMATH_GPT_zack_initial_marbles_l1171_117173

noncomputable def total_initial_marbles (x : ℕ) : ℕ :=
  81 * x + 27

theorem zack_initial_marbles :
  ∃ x : ℕ, total_initial_marbles x = 270 :=
by
  use 3
  sorry

end NUMINAMATH_GPT_zack_initial_marbles_l1171_117173


namespace NUMINAMATH_GPT_pat_mark_ratio_l1171_117123

theorem pat_mark_ratio :
  ∃ K P M : ℕ, P + K + M = 189 ∧ P = 2 * K ∧ M = K + 105 ∧ P / gcd P M = 1 ∧ M / gcd P M = 3 :=
by
  sorry

end NUMINAMATH_GPT_pat_mark_ratio_l1171_117123


namespace NUMINAMATH_GPT_completing_the_square_x_squared_plus_4x_plus_3_eq_0_l1171_117143

theorem completing_the_square_x_squared_plus_4x_plus_3_eq_0 :
  (x : ℝ) → x^2 + 4 * x + 3 = 0 → (x + 2)^2 = 1 :=
by
  intros x h
  -- The actual proof will be provided here
  sorry

end NUMINAMATH_GPT_completing_the_square_x_squared_plus_4x_plus_3_eq_0_l1171_117143


namespace NUMINAMATH_GPT_range_of_m_l1171_117101

open Real

theorem range_of_m (m : ℝ) : (m^2 > 2 + m ∧ 2 + m > 0) ↔ (m > 2 ∨ -2 < m ∧ m < -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1171_117101


namespace NUMINAMATH_GPT_coefficient_x2_in_expansion_l1171_117192

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Statement to prove the coefficient of the x^2 term in (x + 1)^42 is 861
theorem coefficient_x2_in_expansion :
  (binomial 42 2) = 861 := by
  sorry

end NUMINAMATH_GPT_coefficient_x2_in_expansion_l1171_117192


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1171_117196

theorem quadratic_inequality_solution (x : ℝ) :
  (-3 * x^2 + 8 * x + 3 > 0) ↔ (x < -1/3 ∨ x > 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1171_117196


namespace NUMINAMATH_GPT_sue_votes_correct_l1171_117148

def total_votes : ℕ := 1000
def percentage_others : ℝ := 0.65
def sue_votes : ℕ := 350

theorem sue_votes_correct :
  sue_votes = (total_votes : ℝ) * (1 - percentage_others) :=
by
  sorry

end NUMINAMATH_GPT_sue_votes_correct_l1171_117148


namespace NUMINAMATH_GPT_solve_real_equation_l1171_117180

theorem solve_real_equation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -3) :
  (x ^ 3 + 3 * x ^ 2 - x) / (x ^ 2 + 4 * x + 3) + x = -7 ↔ x = -5 / 2 ∨ x = -4 := 
by
  sorry

end NUMINAMATH_GPT_solve_real_equation_l1171_117180


namespace NUMINAMATH_GPT_tesla_ratio_l1171_117182

variables (s c e : ℕ)
variables (h1 : e = s + 10) (h2 : c = 6) (h3 : e = 13)

theorem tesla_ratio : s / c = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tesla_ratio_l1171_117182


namespace NUMINAMATH_GPT_quadratic_root_relation_l1171_117121

theorem quadratic_root_relation (x₁ x₂ : ℝ) (h₁ : x₁ ^ 2 - 3 * x₁ + 2 = 0) (h₂ : x₂ ^ 2 - 3 * x₂ + 2 = 0) :
  x₁ + x₂ - x₁ * x₂ = 1 := by
sorry

end NUMINAMATH_GPT_quadratic_root_relation_l1171_117121


namespace NUMINAMATH_GPT_decreasing_interval_ln_quadratic_l1171_117159

theorem decreasing_interval_ln_quadratic :
  ∀ x : ℝ, (x < 1 ∨ x > 3) → (∀ a b : ℝ, (a ≤ b) → (a < 1 ∨ a > 3) → (b < 1 ∨ b > 3) → (a ≤ x ∧ x ≤ b → (x^2 - 4 * x + 3) ≥ (b^2 - 4 * b + 3))) :=
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_ln_quadratic_l1171_117159


namespace NUMINAMATH_GPT_hexagon_area_ratio_l1171_117171

open Real

theorem hexagon_area_ratio (r s : ℝ) (h_eq_diam : s = r * sqrt 3) :
    (let a1 := (3 * sqrt 3 / 2) * ((3 * r / 4) ^ 2)
     let a2 := (3 * sqrt 3 / 2) * r^2
     a1 / a2 = 9 / 16) :=
by
  sorry

end NUMINAMATH_GPT_hexagon_area_ratio_l1171_117171


namespace NUMINAMATH_GPT_pyramid_volume_l1171_117114

noncomputable def volume_of_pyramid (a b c d: ℝ) (diagonal: ℝ) (angle: ℝ) : ℝ :=
  if (a = 10 ∧ d = 10 ∧ b = 5 ∧ c = 5 ∧ diagonal = 4 * Real.sqrt 5 ∧ angle = 45) then
    let base_area := 1 / 2 * (diagonal) * (Real.sqrt ((c * c) + (b * b)))
    let height := 10 / 3
    let volume := 1 / 3 * base_area * height
    volume
  else 0

theorem pyramid_volume :
  volume_of_pyramid 10 5 5 10 (4 * Real.sqrt 5) 45 = 500 / 9 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_volume_l1171_117114


namespace NUMINAMATH_GPT_triangle_ABC_is_acute_l1171_117152

noncomputable def arithmeticSeqTerm (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def geometricSeqTerm (a1 r : ℝ) (n : ℕ) : ℝ :=
  a1 * r^(n - 1)

def tanA_condition (a1 d : ℝ) :=
  arithmeticSeqTerm a1 d 3 = -4 ∧ arithmeticSeqTerm a1 d 7 = 4

def tanB_condition (a1 r : ℝ) :=
  geometricSeqTerm a1 r 3 = 1/3 ∧ geometricSeqTerm a1 r 6 = 9

theorem triangle_ABC_is_acute {A B : ℝ} (a1a da a1b rb : ℝ) 
  (hA : tanA_condition a1a da) 
  (hB : tanB_condition a1b rb) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ (A + B) < π :=
  sorry

end NUMINAMATH_GPT_triangle_ABC_is_acute_l1171_117152


namespace NUMINAMATH_GPT_roxy_bought_flowering_plants_l1171_117100

-- Definitions based on conditions
def initial_flowering_plants : ℕ := 7
def initial_fruiting_plants : ℕ := 2 * initial_flowering_plants
def plants_after_saturday (F : ℕ) : ℕ := initial_flowering_plants + F + initial_fruiting_plants + 2
def plants_after_sunday (F : ℕ) : ℕ := (initial_flowering_plants + F - 1) + (initial_fruiting_plants + 2 - 4)
def final_plants_in_garden : ℕ := 21

-- The proof statement
theorem roxy_bought_flowering_plants (F : ℕ) :
  plants_after_sunday F = final_plants_in_garden → F = 3 := 
sorry

end NUMINAMATH_GPT_roxy_bought_flowering_plants_l1171_117100


namespace NUMINAMATH_GPT_set_list_method_l1171_117142

theorem set_list_method : 
  {x : ℝ | x^2 - 2 * x + 1 = 0} = {1} :=
sorry

end NUMINAMATH_GPT_set_list_method_l1171_117142


namespace NUMINAMATH_GPT_time_to_wash_car_l1171_117133

theorem time_to_wash_car (W : ℕ) 
    (t_oil : ℕ := 15) 
    (t_tires : ℕ := 30) 
    (n_wash : ℕ := 9) 
    (n_oil : ℕ := 6) 
    (n_tires : ℕ := 2) 
    (total_time : ℕ := 240) 
    (h : n_wash * W + n_oil * t_oil + n_tires * t_tires = total_time) 
    : W = 10 := by
  sorry

end NUMINAMATH_GPT_time_to_wash_car_l1171_117133


namespace NUMINAMATH_GPT_find_PB_l1171_117168

variables (P A B C D : Point) (PA PD PC PB : ℝ)
-- Assume P is interior to rectangle ABCD
-- Conditions
axiom hPA : PA = 3
axiom hPD : PD = 4
axiom hPC : PC = 5

-- The main statement to prove
theorem find_PB (P A B C D : Point) (PA PD PC PB : ℝ)
  (hPA : PA = 3) (hPD : PD = 4) (hPC : PC = 5) : PB = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_PB_l1171_117168


namespace NUMINAMATH_GPT_actual_time_l1171_117177

variables (m_pos : ℕ) (h_pos : ℕ)

-- The mirrored positions
def minute_hand_in_mirror : ℕ := 10
def hour_hand_in_mirror : ℕ := 5

theorem actual_time (m_pos h_pos : ℕ) 
  (hm : m_pos = 2) 
  (hh : h_pos < 7 ∧ h_pos ≥ 6) : 
  m_pos = 10 ∧ h_pos < 7 ∧ h_pos ≥ 6 :=
sorry

end NUMINAMATH_GPT_actual_time_l1171_117177


namespace NUMINAMATH_GPT_combined_average_mark_l1171_117187

theorem combined_average_mark 
  (n_A n_B n_C n_D n_E : ℕ) 
  (avg_A avg_B avg_C avg_D avg_E : ℕ)
  (students_A : n_A = 22) (students_B : n_B = 28)
  (students_C : n_C = 15) (students_D : n_D = 35)
  (students_E : n_E = 25)
  (avg_marks_A : avg_A = 40) (avg_marks_B : avg_B = 60)
  (avg_marks_C : avg_C = 55) (avg_marks_D : avg_D = 75)
  (avg_marks_E : avg_E = 50) : 
  (22 * 40 + 28 * 60 + 15 * 55 + 35 * 75 + 25 * 50) / (22 + 28 + 15 + 35 + 25) = 58.08 := 
  by 
    sorry

end NUMINAMATH_GPT_combined_average_mark_l1171_117187


namespace NUMINAMATH_GPT_range_of_numbers_l1171_117151

theorem range_of_numbers (a b c : ℕ) (h_mean : (a + b + c) / 3 = 4) (h_median : b = 4) (h_smallest : a = 1) :
  c - a = 6 :=
sorry

end NUMINAMATH_GPT_range_of_numbers_l1171_117151


namespace NUMINAMATH_GPT_solve_xyz_l1171_117163

theorem solve_xyz (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : x + y + z = a + b + c)
  (h2 : 4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c) :
  (x, y, z) = ( (b + c) / 2, (c + a) / 2, (a + b) / 2 ) :=
sorry

end NUMINAMATH_GPT_solve_xyz_l1171_117163


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1171_117160

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x^2 - a ≤ 0)) → (a ≥ 5) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1171_117160


namespace NUMINAMATH_GPT_determine_number_of_shelves_l1171_117181

-- Define the total distance Karen bikes round trip
def total_distance : ℕ := 3200

-- Define the number of books per shelf
def books_per_shelf : ℕ := 400

-- Calculate the one-way distance from Karen's home to the library
def one_way_distance (total_distance : ℕ) : ℕ := total_distance / 2

-- Define the total number of books, which is the same as the one-way distance
def total_books (one_way_distance : ℕ) : ℕ := one_way_distance

-- Calculate the number of shelves
def number_of_shelves (total_books : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books / books_per_shelf

theorem determine_number_of_shelves :
  number_of_shelves (total_books (one_way_distance total_distance)) books_per_shelf = 4 :=
by 
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_determine_number_of_shelves_l1171_117181


namespace NUMINAMATH_GPT_odd_f_l1171_117153

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2^x else if x < 0 then -x^2 + 2^(-x) else 0

theorem odd_f (x : ℝ) : (f (-x) = -f x) :=
by
  sorry

end NUMINAMATH_GPT_odd_f_l1171_117153


namespace NUMINAMATH_GPT_find_number_l1171_117162

theorem find_number (x : ℕ) (h : x + 18 = 44) : x = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1171_117162


namespace NUMINAMATH_GPT_pencils_and_pens_cost_l1171_117130

theorem pencils_and_pens_cost (p q : ℝ)
  (h1 : 8 * p + 3 * q = 5.60)
  (h2 : 2 * p + 5 * q = 4.25) :
  3 * p + 4 * q = 9.68 :=
sorry

end NUMINAMATH_GPT_pencils_and_pens_cost_l1171_117130


namespace NUMINAMATH_GPT_sum_x_coordinates_midpoints_l1171_117105

theorem sum_x_coordinates_midpoints (a b c : ℝ) (h : a + b + c = 12) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_x_coordinates_midpoints_l1171_117105


namespace NUMINAMATH_GPT_minimize_x_2y_l1171_117183

noncomputable def minimum_value_x_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 3 / (x + 2) + 3 / (y + 2) = 1) : ℝ :=
  x + 2 * y

theorem minimize_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / (x + 2) + 3 / (y + 2) = 1) :
  minimum_value_x_2y x y hx hy h = 3 + 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimize_x_2y_l1171_117183


namespace NUMINAMATH_GPT_robbie_weekly_fat_intake_l1171_117122

theorem robbie_weekly_fat_intake
  (morning_cups : ℕ) (afternoon_cups : ℕ) (evening_cups : ℕ)
  (fat_per_cup : ℕ) (days_per_week : ℕ) :
  morning_cups = 3 →
  afternoon_cups = 2 →
  evening_cups = 5 →
  fat_per_cup = 10 →
  days_per_week = 7 →
  (morning_cups * fat_per_cup + afternoon_cups * fat_per_cup + evening_cups * fat_per_cup) * days_per_week = 700 :=
by
  intros
  sorry

end NUMINAMATH_GPT_robbie_weekly_fat_intake_l1171_117122
