import Mathlib

namespace NUMINAMATH_GPT_upper_limit_of_range_l2174_217402

theorem upper_limit_of_range (N : ℕ) :
  (∀ n : ℕ, (20 + n * 10 ≤ N) = (n < 198)) → N = 1990 :=
by
  sorry

end NUMINAMATH_GPT_upper_limit_of_range_l2174_217402


namespace NUMINAMATH_GPT_smallest_x_l2174_217412

noncomputable def f (x : ℝ) : ℝ :=
if 1 ≤ x ∧ x ≤ 4 then x^2 - 4 * x + 5 else sorry

theorem smallest_x (x : ℝ) (h₁ : ∀ x > 0, f (4 * x) = 4 * f x)
  (h₂ : ∀ x, (1 ≤ x ∧ x ≤ 4) → f x = x^2 - 4 * x + 5) :
  ∃ x₀, x₀ > 0 ∧ f x₀ = 1024 ∧ (∀ y, y > 0 ∧ f y = 1024 → y ≥ x₀) :=
sorry

end NUMINAMATH_GPT_smallest_x_l2174_217412


namespace NUMINAMATH_GPT_game_positions_l2174_217443

def spots := ["top-left", "top-right", "bottom-right", "bottom-left"]
def segments := ["top-left", "top-middle-left", "top-middle-right", "top-right", "right-top", "right-middle-top", "right-middle-bottom", "right-bottom", "bottom-right", "bottom-middle-right", "bottom-middle-left", "bottom-left", "left-top", "left-middle-top", "left-middle-bottom", "left-bottom"]

def cat_position_after_moves (n : Nat) : String :=
  spots.get! (n % 4)

def mouse_position_after_moves (n : Nat) : String :=
  segments.get! ((12 - (n % 12)) % 12)

theorem game_positions :
  cat_position_after_moves 359 = "bottom-right" ∧ 
  mouse_position_after_moves 359 = "left-middle-bottom" :=
by
  sorry

end NUMINAMATH_GPT_game_positions_l2174_217443


namespace NUMINAMATH_GPT_marble_cut_percentage_first_week_l2174_217476

theorem marble_cut_percentage_first_week :
  ∀ (W1 W2 : ℝ), 
  W1 = W2 / 0.70 → 
  W2 = 124.95 / 0.85 → 
  (300 - W1) / 300 * 100 = 30 :=
by
  intros W1 W2 h1 h2
  sorry

end NUMINAMATH_GPT_marble_cut_percentage_first_week_l2174_217476


namespace NUMINAMATH_GPT_extreme_value_and_tangent_line_l2174_217455

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2 - 3 * x

theorem extreme_value_and_tangent_line (a b : ℝ) (h1 : f a b 1 = 0) (h2 : f a b (-1) = 0) :
  (f 1 0 (-1) = 2) ∧ (f 1 0 1 = -2) ∧ (∀ x : ℝ, x = -2 → (9 * x - (x^3 - 3 * x) + 16 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_extreme_value_and_tangent_line_l2174_217455


namespace NUMINAMATH_GPT_solve_x_l2174_217425

theorem solve_x (x : ℝ) (h : (x + 1) ^ 2 = 9) : x = 2 ∨ x = -4 :=
sorry

end NUMINAMATH_GPT_solve_x_l2174_217425


namespace NUMINAMATH_GPT_steven_more_peaches_than_apples_l2174_217449

def steven_peaches : Nat := 17
def steven_apples : Nat := 16

theorem steven_more_peaches_than_apples : steven_peaches - steven_apples = 1 := by
  sorry

end NUMINAMATH_GPT_steven_more_peaches_than_apples_l2174_217449


namespace NUMINAMATH_GPT_no_snow_probability_l2174_217442

noncomputable def probability_of_no_snow (p_snow : ℚ) : ℚ :=
  1 - p_snow

theorem no_snow_probability : probability_of_no_snow (2/5) = 3/5 :=
  sorry

end NUMINAMATH_GPT_no_snow_probability_l2174_217442


namespace NUMINAMATH_GPT_number_of_small_spheres_l2174_217438

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem number_of_small_spheres
  (d_large : ℝ) (d_small : ℝ)
  (h1 : d_large = 6) (h2 : d_small = 2) :
  let V_large := volume_of_sphere (d_large / 2)
  let V_small := volume_of_sphere (d_small / 2)
  V_large / V_small = 27 := 
by
  sorry

end NUMINAMATH_GPT_number_of_small_spheres_l2174_217438


namespace NUMINAMATH_GPT_sufficiency_s_for_q_l2174_217465

variables {q r s : Prop}

theorem sufficiency_s_for_q (h₁ : r → q) (h₂ : ¬(q → r)) (h₃ : r ↔ s) : s → q ∧ ¬(q → s) :=
by
  sorry

end NUMINAMATH_GPT_sufficiency_s_for_q_l2174_217465


namespace NUMINAMATH_GPT_num_prime_divisors_50_factorial_eq_15_l2174_217421

theorem num_prime_divisors_50_factorial_eq_15 :
  (Finset.filter Nat.Prime (Finset.range 51)).card = 15 :=
by
  sorry

end NUMINAMATH_GPT_num_prime_divisors_50_factorial_eq_15_l2174_217421


namespace NUMINAMATH_GPT_problem_a_problem_b_l2174_217464

section ProblemA

variable (x : ℝ)

theorem problem_a :
  x ≠ 0 ∧ x ≠ -3/8 ∧ x ≠ 3/7 →
  2 + 5 / (4 * x) - 15 / (4 * x * (8 * x + 3)) = 2 * (7 * x + 1) / (7 * x - 3) →
  x = 9 := by
  sorry

end ProblemA

section ProblemB

variable (x : ℝ)

theorem problem_b :
  x ≠ 0 →
  2 / x + 1 / x^2 - (7 + 10 * x) / (x^2 * (x^2 + 7)) = 2 / (x + 3 / (x + 4 / x)) →
  x = 4 := by
  sorry

end ProblemB

end NUMINAMATH_GPT_problem_a_problem_b_l2174_217464


namespace NUMINAMATH_GPT_border_area_correct_l2174_217429

-- Define the dimensions of the photograph
def photograph_height : ℕ := 12
def photograph_width : ℕ := 15

-- Define the width of the border
def border_width : ℕ := 3

-- Define the area of the photograph
def area_photograph : ℕ := photograph_height * photograph_width

-- Define the total dimensions including the frame
def total_height : ℕ := photograph_height + 2 * border_width
def total_width : ℕ := photograph_width + 2 * border_width

-- Define the area of the framed area
def area_framed : ℕ := total_height * total_width

-- Define the area of the border
def area_border : ℕ := area_framed - area_photograph

theorem border_area_correct : area_border = 198 := by
  sorry

end NUMINAMATH_GPT_border_area_correct_l2174_217429


namespace NUMINAMATH_GPT_equation_of_parallel_line_l2174_217430

theorem equation_of_parallel_line (c : ℕ) :
  (∃ c, x + 2 * y + c = 0) ∧ (1 + 2 * 1 + c = 0) -> x + 2 * y - 3 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_equation_of_parallel_line_l2174_217430


namespace NUMINAMATH_GPT_minimum_value_function_l2174_217479

theorem minimum_value_function :
  ∀ x : ℝ, x ≥ 0 → (∃ y : ℝ, y = (3 * x^2 + 9 * x + 20) / (7 * (2 + x)) ∧
    (∀ z : ℝ, z ≥ 0 → (3 * z^2 + 9 * z + 20) / (7 * (2 + z)) ≥ y)) ∧
    (∃ x0 : ℝ, x0 = 0 ∧ y = (3 * x0^2 + 9 * x0 + 20) / (7 * (2 + x0)) ∧ y = 10 / 7) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_function_l2174_217479


namespace NUMINAMATH_GPT_solution_set_inequality_f_solution_range_a_l2174_217422

-- Define the function f 
def f (x : ℝ) := |x + 1| + |x - 3|

-- Statement for question 1
theorem solution_set_inequality_f (x : ℝ) : f x < 6 ↔ -2 < x ∧ x < 4 :=
sorry

-- Statement for question 2
theorem solution_range_a (a : ℝ) (h : ∃ x : ℝ, f x = |a - 2|) : a ≥ 6 ∨ a ≤ -2 :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_f_solution_range_a_l2174_217422


namespace NUMINAMATH_GPT_possible_values_of_n_l2174_217466

theorem possible_values_of_n :
  let a := 1500
  let max_r2 := 562499
  let total := max_r2
  let perfect_squares := (750 : Nat)
  total - perfect_squares = 561749 := by
    sorry

end NUMINAMATH_GPT_possible_values_of_n_l2174_217466


namespace NUMINAMATH_GPT_cos_4_arccos_l2174_217492

theorem cos_4_arccos (y : ℝ) (hy1 : y = Real.arccos (2/5)) (hy2 : Real.cos y = 2/5) : 
  Real.cos (4 * y) = -47 / 625 := 
by 
  sorry

end NUMINAMATH_GPT_cos_4_arccos_l2174_217492


namespace NUMINAMATH_GPT_weight_units_correct_l2174_217435

-- Definitions of weights
def weight_peanut_kernel := 1 -- gram
def weight_truck_capacity := 8 -- ton
def weight_xiao_ming := 30 -- kilogram
def weight_basketball := 580 -- gram

-- Proof that the weights have correct units
theorem weight_units_correct :
  (weight_peanut_kernel = 1 ∧ weight_truck_capacity = 8 ∧ weight_xiao_ming = 30 ∧ weight_basketball = 580) :=
by {
  sorry
}

end NUMINAMATH_GPT_weight_units_correct_l2174_217435


namespace NUMINAMATH_GPT_tie_to_shirt_ratio_l2174_217410

-- Definitions for the conditions
def pants_cost : ℝ := 20
def shirt_cost : ℝ := 2 * pants_cost
def socks_cost : ℝ := 3
def r : ℝ := sorry -- This will be proved
def tie_cost : ℝ := r * shirt_cost
def uniform_cost : ℝ := pants_cost + shirt_cost + tie_cost + socks_cost

-- The total cost for five uniforms
def total_cost : ℝ := 5 * uniform_cost

-- The given total cost
def given_total_cost : ℝ := 355

-- The theorem to be proved
theorem tie_to_shirt_ratio :
  total_cost = given_total_cost → r = 1 / 5 := 
sorry

end NUMINAMATH_GPT_tie_to_shirt_ratio_l2174_217410


namespace NUMINAMATH_GPT_math_problem_l2174_217431

noncomputable def compute_value (c d : ℝ) : ℝ := 100 * c + d

-- Problem statement as a theorem
theorem math_problem
  (c d : ℝ)
  (H1 : ∀ x : ℝ, (x + c) * (x + d) * (x + 10) = 0 → x = -c ∨ x = -d ∨ x = -10)
  (H2 : ∀ x : ℝ, (x + 3 * c) * (x + 5) * (x + 8) = 0 → (x = -4 ∧ ∀ y : ℝ, y ≠ -4 → (y + d) * (y + 10) ≠ 0))
  (H3 : c ≠ 4 / 3 → 3 * c = d ∨ 3 * c = 10) :
  compute_value c d = 141.33 :=
by sorry

end NUMINAMATH_GPT_math_problem_l2174_217431


namespace NUMINAMATH_GPT_factorize_square_difference_l2174_217489

open Real

theorem factorize_square_difference (m n : ℝ) :
  m ^ 2 - 4 * n ^ 2 = (m + 2 * n) * (m - 2 * n) :=
sorry

end NUMINAMATH_GPT_factorize_square_difference_l2174_217489


namespace NUMINAMATH_GPT_sum_of_constants_eq_17_l2174_217415

theorem sum_of_constants_eq_17
  (x y : ℝ)
  (a b c d : ℕ)
  (ha : a = 6)
  (hb : b = 2)
  (hc : c = 3)
  (hd : d = 3)
  (h1 : x + y = 4)
  (h2 : 3 * x * y = 4)
  (h3 : x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d) :
  a + b + c + d = 17 :=
sorry

end NUMINAMATH_GPT_sum_of_constants_eq_17_l2174_217415


namespace NUMINAMATH_GPT_average_income_l2174_217446

theorem average_income (income1 income2 income3 income4 income5 : ℝ)
    (h1 : income1 = 600) (h2 : income2 = 250) (h3 : income3 = 450) (h4 : income4 = 400) (h5 : income5 = 800) :
    (income1 + income2 + income3 + income4 + income5) / 5 = 500 := by
    sorry

end NUMINAMATH_GPT_average_income_l2174_217446


namespace NUMINAMATH_GPT_raghu_investment_approx_l2174_217490

-- Define the investments
def investments (R : ℝ) : Prop :=
  let Trishul := 0.9 * R
  let Vishal := 0.99 * R
  let Deepak := 1.188 * R
  R + Trishul + Vishal + Deepak = 8578

-- State the theorem to prove that Raghu invested approximately Rs. 2103.96
theorem raghu_investment_approx : 
  ∃ R : ℝ, investments R ∧ abs (R - 2103.96) < 1 :=
by
  sorry

end NUMINAMATH_GPT_raghu_investment_approx_l2174_217490


namespace NUMINAMATH_GPT_positive_solution_x_l2174_217411

theorem positive_solution_x (x y z : ℝ) (h1 : x * y = 10 - 3 * x - 2 * y) 
(h2 : y * z = 10 - 5 * y - 3 * z) 
(h3 : x * z = 40 - 5 * x - 2 * z) 
(h_pos : x > 0) : 
  x = 8 :=
sorry

end NUMINAMATH_GPT_positive_solution_x_l2174_217411


namespace NUMINAMATH_GPT_exists_rectangle_with_properties_l2174_217494

variables {e a φ : ℝ}

-- Define the given conditions
def diagonal_diff (e a : ℝ) := e - a
def angle_between_diagonals (φ : ℝ) := φ

-- The problem to prove
theorem exists_rectangle_with_properties (e a φ : ℝ) 
  (h_diff : diagonal_diff e a = e - a) 
  (h_angle : angle_between_diagonals φ = φ) : 
  ∃ (rectangle : Type) (A B C D : rectangle), 
    (e - a = e - a) ∧ 
    (φ = φ) := 
sorry

end NUMINAMATH_GPT_exists_rectangle_with_properties_l2174_217494


namespace NUMINAMATH_GPT_probability_T_H_E_equal_L_A_V_A_l2174_217414

noncomputable def probability_condition : ℚ :=
  -- Number of total sample space (3^6)
  (3 ^ 6 : ℚ)

noncomputable def favorable_events_0 : ℚ :=
  -- Number of favorable outcomes where 𝑻 ⋅ 𝑯 ⋅ 𝑬 is 0 and 𝑳 ⋅ 𝑨 ⋅ 𝑽 ⋅ 𝑨 is 0
  26 * 19

noncomputable def favorable_events_1 : ℚ :=
  -- Number of favorable outcomes where 𝑻 ⋅ 𝑯 ⋅ 𝑬 is 1 and 𝑳 ⋅ 𝑨 ⋅ 𝑽 ⋅ 𝑨 is 1
  1

noncomputable def total_favorable_events : ℚ :=
  favorable_events_0 + favorable_events_1

theorem probability_T_H_E_equal_L_A_V_A :
  (total_favorable_events / probability_condition) = 55 / 81 :=
sorry

end NUMINAMATH_GPT_probability_T_H_E_equal_L_A_V_A_l2174_217414


namespace NUMINAMATH_GPT_manny_had_3_pies_l2174_217480

-- Definitions of the conditions
def number_of_classmates : ℕ := 24
def number_of_teachers : ℕ := 1
def slices_per_pie : ℕ := 10
def slices_left : ℕ := 4

-- Number of people including Manny
def number_of_people : ℕ := number_of_classmates + number_of_teachers + 1

-- Total number of slices eaten
def slices_eaten : ℕ := number_of_people

-- Total number of slices initially
def total_slices : ℕ := slices_eaten + slices_left

-- Number of pies Manny had
def number_of_pies : ℕ := (total_slices / slices_per_pie) + 1

-- Theorem statement
theorem manny_had_3_pies : number_of_pies = 3 := by
  sorry

end NUMINAMATH_GPT_manny_had_3_pies_l2174_217480


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l2174_217456

theorem sum_of_squares_of_roots (s1 s2 : ℝ) (h1 : s1 * s2 = 4) (h2 : s1 + s2 = 16) : s1^2 + s2^2 = 248 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l2174_217456


namespace NUMINAMATH_GPT_proof_problem_l2174_217467

-- Conditions: p and q are solutions to the quadratic equation 3x^2 - 5x - 8 = 0
def is_solution (p q : ℝ) : Prop := (3 * p^2 - 5 * p - 8 = 0) ∧ (3 * q^2 - 5 * q - 8 = 0)

-- Question: Compute the value of (3 * p^2 - 3 * q^2) / (p - q) given the conditions
theorem proof_problem (p q : ℝ) (h : is_solution p q) :
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := sorry

end NUMINAMATH_GPT_proof_problem_l2174_217467


namespace NUMINAMATH_GPT_gcd_ab_a2b2_eq_1_or_2_l2174_217458

theorem gcd_ab_a2b2_eq_1_or_2
  (a b : Nat)
  (h_coprime : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_gcd_ab_a2b2_eq_1_or_2_l2174_217458


namespace NUMINAMATH_GPT_people_behind_yuna_l2174_217448

theorem people_behind_yuna (total_people : ℕ) (people_in_front : ℕ) (yuna : ℕ)
  (h1 : total_people = 7) (h2 : people_in_front = 2) (h3 : yuna = 1) :
  total_people - people_in_front - yuna = 4 :=
by
  sorry

end NUMINAMATH_GPT_people_behind_yuna_l2174_217448


namespace NUMINAMATH_GPT_square_of_positive_difference_l2174_217460

theorem square_of_positive_difference {y : ℝ}
  (h : (45 + y) / 2 = 50) :
  (|y - 45|)^2 = 100 :=
by
  sorry

end NUMINAMATH_GPT_square_of_positive_difference_l2174_217460


namespace NUMINAMATH_GPT_bob_shuck_2_hours_l2174_217462

def shucking_rate : ℕ := 10  -- oysters per 5 minutes
def minutes_per_hour : ℕ := 60
def hours : ℕ := 2
def minutes : ℕ := hours * minutes_per_hour
def interval : ℕ := 5  -- minutes per interval
def intervals : ℕ := minutes / interval
def num_oysters (intervals : ℕ) : ℕ := intervals * shucking_rate

theorem bob_shuck_2_hours : num_oysters intervals = 240 := by
  -- leave the proof as an exercise
  sorry

end NUMINAMATH_GPT_bob_shuck_2_hours_l2174_217462


namespace NUMINAMATH_GPT_binom_13_10_eq_286_l2174_217468

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_13_10_eq_286 : binomial 13 10 = 286 := by
  sorry

end NUMINAMATH_GPT_binom_13_10_eq_286_l2174_217468


namespace NUMINAMATH_GPT_problem_statement_l2174_217409

def g (x : ℝ) : ℝ :=
  x^2 - 5 * x

theorem problem_statement (x : ℝ) :
  (g (g x) = g x) ↔ (x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2174_217409


namespace NUMINAMATH_GPT_win_sector_area_l2174_217428

theorem win_sector_area (r : ℝ) (p_win : ℝ) (area_total : ℝ) 
  (h1 : r = 8)
  (h2 : p_win = 3 / 8)
  (h3 : area_total = π * r^2) :
  ∃ area_win, area_win = 24 * π ∧ area_win = p_win * area_total :=
by
  sorry

end NUMINAMATH_GPT_win_sector_area_l2174_217428


namespace NUMINAMATH_GPT_sphere_surface_area_from_volume_l2174_217433

theorem sphere_surface_area_from_volume 
  (V : ℝ) (h : V = 72 * Real.pi) :
  ∃ (A : ℝ), A = 36 * Real.pi * 2^(2/3) :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_from_volume_l2174_217433


namespace NUMINAMATH_GPT_jill_total_watch_time_l2174_217405

theorem jill_total_watch_time :
  ∀ (length_first_show length_second_show total_watch_time : ℕ),
    length_first_show = 30 →
    length_second_show = 4 * length_first_show →
    total_watch_time = length_first_show + length_second_show →
    total_watch_time = 150 :=
by
  sorry

end NUMINAMATH_GPT_jill_total_watch_time_l2174_217405


namespace NUMINAMATH_GPT_helga_tried_on_66_pairs_of_shoes_l2174_217474

variables 
  (n1 n2 n3 n4 n5 n6 : ℕ)
  (h1 : n1 = 7)
  (h2 : n2 = n1 + 2)
  (h3 : n3 = 0)
  (h4 : n4 = 2 * (n1 + n2 + n3))
  (h5 : n5 = n2 - 3)
  (h6 : n6 = n1 + 5)
  (total : ℕ := n1 + n2 + n3 + n4 + n5 + n6)

theorem helga_tried_on_66_pairs_of_shoes : total = 66 :=
by sorry

end NUMINAMATH_GPT_helga_tried_on_66_pairs_of_shoes_l2174_217474


namespace NUMINAMATH_GPT_chocolate_bar_cost_l2174_217439

theorem chocolate_bar_cost (total_bars : ℕ) (sold_bars : ℕ) (total_money : ℕ) (cost : ℕ) 
  (h1 : total_bars = 13)
  (h2 : sold_bars = total_bars - 4)
  (h3 : total_money = 18)
  (h4 : total_money = sold_bars * cost) :
  cost = 2 :=
by sorry

end NUMINAMATH_GPT_chocolate_bar_cost_l2174_217439


namespace NUMINAMATH_GPT_length_GH_l2174_217423

def length_AB : ℕ := 11
def length_FE : ℕ := 13
def length_CD : ℕ := 5

theorem length_GH : length_AB + length_CD + length_FE = 29 :=
by
  refine rfl -- This will unroll the constants and perform arithmetic

end NUMINAMATH_GPT_length_GH_l2174_217423


namespace NUMINAMATH_GPT_tan_double_angle_l2174_217445

theorem tan_double_angle (α β : ℝ) (h1 : Real.tan (α + β) = 7) (h2 : Real.tan (α - β) = 1) : 
  Real.tan (2 * α) = -4/3 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l2174_217445


namespace NUMINAMATH_GPT_five_nat_numbers_product_1000_l2174_217483

theorem five_nat_numbers_product_1000 :
  ∃ (a b c d e : ℕ), 
    a * b * c * d * e = 1000 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
    c ≠ d ∧ c ≠ e ∧ 
    d ≠ e := 
by
  sorry

end NUMINAMATH_GPT_five_nat_numbers_product_1000_l2174_217483


namespace NUMINAMATH_GPT_find_sum_l2174_217408

variables (a b c d : ℕ)

axiom h1 : 6 * a + 2 * b = 3848
axiom h2 : 6 * c + 3 * d = 4410
axiom h3 : a + 3 * b + 2 * d = 3080

theorem find_sum : a + b + c + d = 1986 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_l2174_217408


namespace NUMINAMATH_GPT_television_hours_watched_l2174_217491

theorem television_hours_watched (minutes_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)
  (h1 : minutes_per_day = 45) (h2 : days_per_week = 4) (h3 : weeks = 2):
  (minutes_per_day * days_per_week / 60) * weeks = 6 :=
by
  sorry

end NUMINAMATH_GPT_television_hours_watched_l2174_217491


namespace NUMINAMATH_GPT_megan_files_in_folder_l2174_217403

theorem megan_files_in_folder :
  let initial_files := 93.0
  let added_files := 21.0
  let total_files := initial_files + added_files
  let total_folders := 14.25
  (total_files / total_folders) = 8.0 :=
by
  let initial_files := 93.0
  let added_files := 21.0
  let total_files := initial_files + added_files
  let total_folders := 14.25
  have h1 : total_files = initial_files + added_files := rfl
  have h2 : total_files = 114.0 := by sorry -- 93.0 + 21.0 = 114.0
  have h3 : total_files / total_folders = 8.0 := by sorry -- 114.0 / 14.25 = 8.0
  exact h3

end NUMINAMATH_GPT_megan_files_in_folder_l2174_217403


namespace NUMINAMATH_GPT_percentage_of_mortality_l2174_217406

theorem percentage_of_mortality
  (P : ℝ) -- The population size could be represented as a real number
  (affected_fraction : ℝ) (dead_fraction : ℝ)
  (h1 : affected_fraction = 0.15) -- 15% of the population is affected
  (h2 : dead_fraction = 0.08) -- 8% of the affected population died
: (affected_fraction * dead_fraction) * 100 = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_mortality_l2174_217406


namespace NUMINAMATH_GPT_exists_positive_integers_abcd_l2174_217496

theorem exists_positive_integers_abcd (m : ℤ) : ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ (a * b - c * d = m) := by
  sorry

end NUMINAMATH_GPT_exists_positive_integers_abcd_l2174_217496


namespace NUMINAMATH_GPT_bob_second_third_lap_time_l2174_217418

theorem bob_second_third_lap_time :
  ∀ (lap_length : ℕ) (first_lap_time : ℕ) (average_speed : ℕ),
  lap_length = 400 →
  first_lap_time = 70 →
  average_speed = 5 →
  ∃ (second_third_lap_time : ℕ), second_third_lap_time = 85 :=
by
  intros lap_length first_lap_time average_speed lap_length_eq first_lap_time_eq average_speed_eq
  sorry

end NUMINAMATH_GPT_bob_second_third_lap_time_l2174_217418


namespace NUMINAMATH_GPT_square_three_times_side_length_l2174_217400

theorem square_three_times_side_length (a : ℝ) : 
  ∃ s, s = a * Real.sqrt 3 ∧ s ^ 2 = 3 * a ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_square_three_times_side_length_l2174_217400


namespace NUMINAMATH_GPT_ratio_of_friends_l2174_217407

theorem ratio_of_friends (friends_in_classes friends_in_clubs : ℕ) (thread_per_keychain total_thread : ℕ) 
  (h1 : thread_per_keychain = 12) (h2 : friends_in_classes = 6) (h3 : total_thread = 108)
  (keychains_total : total_thread / thread_per_keychain = 9) 
  (keychains_clubs : (total_thread / thread_per_keychain) - friends_in_classes = friends_in_clubs) :
  friends_in_clubs / friends_in_classes = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_friends_l2174_217407


namespace NUMINAMATH_GPT_determine_f_function_l2174_217481

variable (f : ℝ → ℝ)

theorem determine_f_function (x : ℝ) (h : f (1 - x) = 1 + x) : f x = 2 - x := 
sorry

end NUMINAMATH_GPT_determine_f_function_l2174_217481


namespace NUMINAMATH_GPT_volume_ratio_john_emma_l2174_217437

theorem volume_ratio_john_emma (r_J h_J r_E h_E : ℝ) (diam_J diam_E : ℝ)
  (h_diam_J : diam_J = 8) (h_r_J : r_J = diam_J / 2) (h_h_J : h_J = 15)
  (h_diam_E : diam_E = 10) (h_r_E : r_E = diam_E / 2) (h_h_E : h_E = 12) :
  (π * r_J^2 * h_J) / (π * r_E^2 * h_E) = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_volume_ratio_john_emma_l2174_217437


namespace NUMINAMATH_GPT_maximum_value_of_f_l2174_217461

noncomputable def f : ℝ → ℝ :=
  fun x => -x^2 * (x^2 + 4*x + 4)

theorem maximum_value_of_f :
  ∀ x : ℝ, x ≠ 0 → x ≠ -2 → x ≠ 1 → x ≠ -3 → f x ≤ 0 ∧ f 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_l2174_217461


namespace NUMINAMATH_GPT_num_7_digit_integers_correct_l2174_217413

-- Define the number of choices for each digit
def first_digit_choices : ℕ := 9
def other_digit_choices : ℕ := 10

-- Define the number of 7-digit positive integers
def num_7_digit_integers : ℕ := first_digit_choices * other_digit_choices^6

-- State the theorem to prove
theorem num_7_digit_integers_correct : num_7_digit_integers = 9000000 :=
by
  sorry

end NUMINAMATH_GPT_num_7_digit_integers_correct_l2174_217413


namespace NUMINAMATH_GPT_sum_max_min_on_interval_l2174_217419

-- Defining the function f
def f (x : ℝ) : ℝ := x + 2

-- The proof statement
theorem sum_max_min_on_interval : 
  let M := max (f 0) (f 4)
  let N := min (f 0) (f 4)
  M + N = 8 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_sum_max_min_on_interval_l2174_217419


namespace NUMINAMATH_GPT_appropriate_sampling_method_l2174_217441

/--
Given there are 40 products in total, consisting of 10 first-class products,
25 second-class products, and 5 defective products, if we need to select
8 products for quality analysis, then the appropriate sampling method is
the stratified sampling method.
-/
theorem appropriate_sampling_method
  (total_products : ℕ)
  (first_class_products : ℕ)
  (second_class_products : ℕ)
  (defective_products : ℕ)
  (selected_products : ℕ)
  (stratified_sampling : ℕ → ℕ → ℕ → ℕ → Prop) :
  total_products = 40 →
  first_class_products = 10 →
  second_class_products = 25 →
  defective_products = 5 →
  selected_products = 8 →
  stratified_sampling total_products first_class_products second_class_products defective_products →
  stratified_sampling total_products first_class_products second_class_products defective_products :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_appropriate_sampling_method_l2174_217441


namespace NUMINAMATH_GPT_perpendicular_line_through_P_l2174_217486

open Real

/-- Define the point P as (-1, 3) -/
def P : ℝ × ℝ := (-1, 3)

/-- Define the line equation -/
def line1 (x y : ℝ) : Prop := x + 2 * y - 3 = 0

/-- Define the perpendicular line equation -/
def perpendicular_line (x y : ℝ) : Prop := 2 * x - y + 5 = 0

/-- The theorem stating that P lies on the perpendicular line to the given line -/
theorem perpendicular_line_through_P : ∀ x y, P = (x, y) → line1 x y → perpendicular_line x y :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_through_P_l2174_217486


namespace NUMINAMATH_GPT_part1_part2_l2174_217457

variables {A B C a b c : ℝ}

-- Condition: sides opposite to angles A, B, and C are a, b, and c respectively and 4b * sin A = sqrt 7 * a
def condition1 : 4 * b * Real.sin A = Real.sqrt 7 * a := sorry

-- Prove that sin B = sqrt 7 / 4
theorem part1 (h : 4 * b * Real.sin A = Real.sqrt 7 * a) :
  Real.sin B = Real.sqrt 7 / 4 := sorry

-- Condition: a, b, and c form an arithmetic sequence with a common difference greater than 0
def condition2 : 2 * b = a + c := sorry

-- Prove that cos A - cos C = sqrt 7 / 2
theorem part2 (h1 : 4 * b * Real.sin A = Real.sqrt 7 * a) (h2 : 2 * b = a + c) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := sorry

end NUMINAMATH_GPT_part1_part2_l2174_217457


namespace NUMINAMATH_GPT_part1_part2_l2174_217452

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Statement for part 1
theorem part1 (a b : ℝ) (h1 : f a b (-1) = 0) (h2 : f a b 3 = 0) (h3 : a ≠ 0) :
  a = -1 ∧ b = 4 :=
sorry

-- Statement for part 2
theorem part2 (a b : ℝ) (h1 : f a b 1 = 2) (h2 : a + b = 1) (h3 : a > 0) (h4 : b > 0) :
  (∀ x > 0, 1 / a + 4 / b ≥ 9) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2174_217452


namespace NUMINAMATH_GPT_find_principal_l2174_217497

theorem find_principal (x y : ℝ) : 
  (2 * x * y / 100 = 400) → 
  (2 * x * y + x * y^2 / 100 = 41000) → 
  x = 4000 := 
by
  sorry

end NUMINAMATH_GPT_find_principal_l2174_217497


namespace NUMINAMATH_GPT_spiders_hired_l2174_217482

theorem spiders_hired (total_workers beavers : ℕ) (h_total : total_workers = 862) (h_beavers : beavers = 318) : (total_workers - beavers) = 544 := by
  sorry

end NUMINAMATH_GPT_spiders_hired_l2174_217482


namespace NUMINAMATH_GPT_second_smallest_packs_hot_dogs_l2174_217471

theorem second_smallest_packs_hot_dogs 
    (n : ℕ) 
    (k : ℤ) 
    (h1 : 10 * n ≡ 4 [MOD 8]) 
    (h2 : n = 4 * k + 2) : 
    n = 6 :=
by sorry

end NUMINAMATH_GPT_second_smallest_packs_hot_dogs_l2174_217471


namespace NUMINAMATH_GPT_det_B_squared_minus_3B_l2174_217469

open Matrix
open Real

variable {α : Type*} [Fintype α] {n : ℕ}
variable [DecidableEq α]

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 4],
  ![1, 3]
]

theorem det_B_squared_minus_3B : det (B * B - 3 • B) = -8 := sorry

end NUMINAMATH_GPT_det_B_squared_minus_3B_l2174_217469


namespace NUMINAMATH_GPT_collinear_vector_l2174_217451

theorem collinear_vector (c R : ℝ) (A B : ℝ × ℝ) (hA: A.1 ^ 2 + A.2 ^ 2 = R ^ 2) (hB: B.1 ^ 2 + B.2 ^ 2 = R ^ 2) 
                         (h_line_A: 2 * A.1 + A.2 = c) (h_line_B: 2 * B.1 + B.2 = c) :
                         ∃ k : ℝ, (4, 2) = (k * (A.1 + B.1), k * (A.2 + B.2)) :=
sorry

end NUMINAMATH_GPT_collinear_vector_l2174_217451


namespace NUMINAMATH_GPT_find_b_and_area_l2174_217426

open Real

variables (a c : ℝ) (A b S : ℝ)

theorem find_b_and_area 
  (h1 : a = sqrt 7) 
  (h2 : c = 3) 
  (h3 : A = π / 3) :
  (b = 1 ∨ b = 2) ∧ (S = 3 * sqrt 3 / 4 ∨ S = 3 * sqrt 3 / 2) := 
by sorry

end NUMINAMATH_GPT_find_b_and_area_l2174_217426


namespace NUMINAMATH_GPT_inequality_solution_l2174_217498

theorem inequality_solution (x : ℝ) :
  (x > -4 ∧ x < -5 / 3) ↔ 
  (2 * x + 3) / (3 * x + 5) > (4 * x + 1) / (x + 4) := 
sorry

end NUMINAMATH_GPT_inequality_solution_l2174_217498


namespace NUMINAMATH_GPT_solve_for_x_l2174_217447

theorem solve_for_x (x y : ℝ) 
  (h1 : 3 * x - 2 * y = 8) 
  (h2 : 2 * x + 3 * y = 11) :
  x = 46 / 13 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2174_217447


namespace NUMINAMATH_GPT_percentage_assigned_exam_l2174_217454

-- Define the conditions of the problem
def total_students : ℕ := 100
def average_assigned : ℝ := 0.55
def average_makeup : ℝ := 0.95
def average_total : ℝ := 0.67

-- Define the proof problem statement
theorem percentage_assigned_exam :
  ∃ (x : ℝ), (x / total_students) * average_assigned + ((total_students - x) / total_students) * average_makeup = average_total ∧ x = 70 :=
by
  sorry

end NUMINAMATH_GPT_percentage_assigned_exam_l2174_217454


namespace NUMINAMATH_GPT_dog_food_weighs_more_l2174_217450

def weight_in_ounces (weight_in_pounds: ℕ) := weight_in_pounds * 16
def total_food_weight (cat_food_bags dog_food_bags: ℕ) (cat_food_pounds dog_food_pounds: ℕ) :=
  (cat_food_bags * weight_in_ounces cat_food_pounds) + (dog_food_bags * weight_in_ounces dog_food_pounds)

theorem dog_food_weighs_more
  (cat_food_bags: ℕ) (cat_food_pounds: ℕ) (dog_food_bags: ℕ) (total_weight_ounces: ℕ) (ounces_in_pound: ℕ)
  (H1: cat_food_bags * weight_in_ounces cat_food_pounds = 96)
  (H2: total_food_weight cat_food_bags dog_food_bags cat_food_pounds dog_food_pounds = total_weight_ounces)
  (H3: ounces_in_pound = 16) :
  dog_food_pounds - cat_food_pounds = 2 := 
by sorry

end NUMINAMATH_GPT_dog_food_weighs_more_l2174_217450


namespace NUMINAMATH_GPT_number_of_possible_values_for_b_l2174_217434

theorem number_of_possible_values_for_b : 
  ∃ (n : ℕ), n = 10 ∧ ∀ (b : ℕ), (2 ≤ b) ∧ (b^2 ≤ 256) ∧ (256 < b^3) ↔ (7 ≤ b ∧ b ≤ 16) :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_possible_values_for_b_l2174_217434


namespace NUMINAMATH_GPT_sum_of_digits_in_7_pow_1500_l2174_217432

-- Define the problem and conditions
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem sum_of_digits_in_7_pow_1500 :
  sum_of_digits (7^1500) = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_in_7_pow_1500_l2174_217432


namespace NUMINAMATH_GPT_gcd_of_powers_l2174_217485

theorem gcd_of_powers (a b c : ℕ) (h1 : a = 2^105 - 1) (h2 : b = 2^115 - 1) (h3 : c = 1023) :
  Nat.gcd a b = c :=
by sorry

end NUMINAMATH_GPT_gcd_of_powers_l2174_217485


namespace NUMINAMATH_GPT_divisor_unique_l2174_217477

theorem divisor_unique {b : ℕ} (h1 : 826 % b = 7) (h2 : 4373 % b = 8) : b = 9 :=
sorry

end NUMINAMATH_GPT_divisor_unique_l2174_217477


namespace NUMINAMATH_GPT_calculate_division_of_powers_l2174_217404

theorem calculate_division_of_powers (a : ℝ) : a^8 / a^2 = a^6 :=
by sorry

end NUMINAMATH_GPT_calculate_division_of_powers_l2174_217404


namespace NUMINAMATH_GPT_find_cosine_l2174_217416
open Real

noncomputable def alpha (α : ℝ) : Prop := 0 < α ∧ α < π / 2 ∧ sin α = 3 / 5

theorem find_cosine (α : ℝ) (h : alpha α) :
  cos (π - α / 2) = - (3 * sqrt 10) / 10 :=
by sorry

end NUMINAMATH_GPT_find_cosine_l2174_217416


namespace NUMINAMATH_GPT_cube_root_opposite_zero_l2174_217420

theorem cube_root_opposite_zero (x : ℝ) (h : x^(1/3) = -x) : x = 0 :=
sorry

end NUMINAMATH_GPT_cube_root_opposite_zero_l2174_217420


namespace NUMINAMATH_GPT_license_plate_palindrome_probability_l2174_217487

-- Definitions for the problem conditions
def count_letter_palindromes : ℕ := 26 * 26
def total_letter_combinations : ℕ := 26 ^ 4

def count_digit_palindromes : ℕ := 10 * 10
def total_digit_combinations : ℕ := 10 ^ 4

def prob_letter_palindrome : ℚ := count_letter_palindromes / total_letter_combinations
def prob_digit_palindrome : ℚ := count_digit_palindromes / total_digit_combinations
def prob_both_palindrome : ℚ := (count_letter_palindromes * count_digit_palindromes) / (total_letter_combinations * total_digit_combinations)

def prob_atleast_one_palindrome : ℚ :=
  prob_letter_palindrome + prob_digit_palindrome - prob_both_palindrome

def p_q_sum : ℕ := 775 + 67600

-- Statement of the problem to be proved
theorem license_plate_palindrome_probability :
  prob_atleast_one_palindrome = 775 / 67600 ∧ p_q_sum = 68375 :=
by { sorry }

end NUMINAMATH_GPT_license_plate_palindrome_probability_l2174_217487


namespace NUMINAMATH_GPT_domain_not_neg1_increasing_on_neg1_infty_min_max_on_3_5_l2174_217444

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

namespace f_props

theorem domain_not_neg1 : ∀ x : ℝ, x ≠ -1 ↔ x ∈ {y | y ≠ -1} :=
by simp [f]

theorem increasing_on_neg1_infty : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → -1 < x2 → f x1 < f x2 :=
sorry

theorem min_max_on_3_5 : (∀ y : ℝ, y = f 3 → y = 5 / 4) ∧ (∀ y : ℝ, y = f 5 → y = 3 / 2) :=
sorry

end f_props

end NUMINAMATH_GPT_domain_not_neg1_increasing_on_neg1_infty_min_max_on_3_5_l2174_217444


namespace NUMINAMATH_GPT_kitten_current_length_l2174_217424

theorem kitten_current_length (initial_length : ℕ) (double_after_2_weeks : ℕ → ℕ) (double_after_4_months : ℕ → ℕ)
  (h1 : initial_length = 4)
  (h2 : double_after_2_weeks initial_length = 2 * initial_length)
  (h3 : double_after_4_months (double_after_2_weeks initial_length) = 2 * (double_after_2_weeks initial_length)) :
  double_after_4_months (double_after_2_weeks initial_length) = 16 := 
by
  sorry

end NUMINAMATH_GPT_kitten_current_length_l2174_217424


namespace NUMINAMATH_GPT_collinear_points_value_l2174_217459

/-- 
If the points (2, a, b), (a, 3, b), and (a, b, 4) are collinear, 
then the value of a + b is 7.
-/
theorem collinear_points_value (a b : ℝ) (h_collinear : ∃ l : ℝ → ℝ × ℝ × ℝ, 
  l 0 = (2, a, b) ∧ l 1 = (a, 3, b) ∧ l 2 = (a, b, 4) ∧ 
  ∀ t s : ℝ, l t = l s → t = s) :
  a + b = 7 :=
sorry

end NUMINAMATH_GPT_collinear_points_value_l2174_217459


namespace NUMINAMATH_GPT_sin_alpha_plus_3pi_div_2_l2174_217401

theorem sin_alpha_plus_3pi_div_2 (α : ℝ) (h : Real.cos α = 1 / 3) : 
  Real.sin (α + 3 * Real.pi / 2) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_plus_3pi_div_2_l2174_217401


namespace NUMINAMATH_GPT_fewest_coach_handshakes_l2174_217417

theorem fewest_coach_handshakes (n k : ℕ) (h1 : (n * (n - 1)) / 2 + k = 281) : k = 5 :=
sorry

end NUMINAMATH_GPT_fewest_coach_handshakes_l2174_217417


namespace NUMINAMATH_GPT_linear_eq_rewrite_l2174_217463

theorem linear_eq_rewrite (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  sorry

end NUMINAMATH_GPT_linear_eq_rewrite_l2174_217463


namespace NUMINAMATH_GPT_quadratic_roots_distinct_l2174_217488

theorem quadratic_roots_distinct (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2*x1 + m = 0 ∧ x2^2 + 2*x2 + m = 0) →
  m < 1 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_distinct_l2174_217488


namespace NUMINAMATH_GPT_difference_max_min_y_l2174_217440

theorem difference_max_min_y {total_students : ℕ} (initial_yes_pct initial_no_pct final_yes_pct final_no_pct : ℝ)
  (initial_conditions : initial_yes_pct = 0.4 ∧ initial_no_pct = 0.6)
  (final_conditions : final_yes_pct = 0.8 ∧ final_no_pct = 0.2) :
  ∃ (min_change max_change : ℝ), max_change - min_change = 0.2 := by
  sorry

end NUMINAMATH_GPT_difference_max_min_y_l2174_217440


namespace NUMINAMATH_GPT_ben_points_l2174_217427

theorem ben_points (B : ℕ) 
  (h1 : 42 = B + 21) : B = 21 := 
by 
-- Proof can be filled in here
sorry

end NUMINAMATH_GPT_ben_points_l2174_217427


namespace NUMINAMATH_GPT_simplify_expression_l2174_217453

theorem simplify_expression : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
    ((Real.sqrt 3 - 2 * Real.sqrt 5 - 1) * (-16 - 2 * Real.sqrt 3)) / 244 := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2174_217453


namespace NUMINAMATH_GPT_f_at_3_l2174_217495

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x - 1

-- The theorem to prove
theorem f_at_3 : f 3 = 5 := sorry

end NUMINAMATH_GPT_f_at_3_l2174_217495


namespace NUMINAMATH_GPT_marble_price_proof_l2174_217478

noncomputable def price_per_colored_marble (total_marbles white_percentage black_percentage white_price black_price total_earnings : ℕ) : ℕ :=
  let white_marbles := total_marbles * white_percentage / 100
  let black_marbles := total_marbles * black_percentage / 100
  let colored_marbles := total_marbles - (white_marbles + black_marbles)
  let earnings_from_white := white_marbles * white_price
  let earnings_from_black := black_marbles * black_price
  let earnings_from_colored := total_earnings - (earnings_from_white + earnings_from_black)
  earnings_from_colored / colored_marbles

theorem marble_price_proof : price_per_colored_marble 100 20 30 5 10 1400 = 20 := 
sorry

end NUMINAMATH_GPT_marble_price_proof_l2174_217478


namespace NUMINAMATH_GPT_ratio_boys_to_girls_l2174_217484

theorem ratio_boys_to_girls
  (b g : ℕ) 
  (h1 : b = g + 6) 
  (h2 : b + g = 36) : b / g = 7 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_boys_to_girls_l2174_217484


namespace NUMINAMATH_GPT_total_pencils_l2174_217472

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (hp : pencils_per_child = 2) (hc : children = 8) :
  pencils_per_child * children = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_pencils_l2174_217472


namespace NUMINAMATH_GPT_calculate_div_expression_l2174_217499

variable (x y : ℝ)

theorem calculate_div_expression : (6 * x^3 * y^2) / (-3 * x * y) = -2 * x^2 * y := by
  sorry

end NUMINAMATH_GPT_calculate_div_expression_l2174_217499


namespace NUMINAMATH_GPT_even_function_expression_l2174_217436

theorem even_function_expression (f : ℝ → ℝ)
  (h₀ : ∀ x, x ≥ 0 → f x = x^2 - 3 * x + 4)
  (h_even : ∀ x, f x = f (-x)) :
  ∀ x, f x = if x < 0 then x^2 + 3 * x + 4 else x^2 - 3 * x + 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_even_function_expression_l2174_217436


namespace NUMINAMATH_GPT_least_integer_gt_square_l2174_217473

theorem least_integer_gt_square (x : ℝ) (y : ℝ) (h1 : x = 2) (h2 : y = Real.sqrt 3) :
  ∃ (n : ℤ), n = 14 ∧ n > (x + y) ^ 2 := by
  sorry

end NUMINAMATH_GPT_least_integer_gt_square_l2174_217473


namespace NUMINAMATH_GPT_discount_equation_l2174_217470

theorem discount_equation (x : ℝ) : 280 * (1 - x) ^ 2 = 177 := 
by 
  sorry

end NUMINAMATH_GPT_discount_equation_l2174_217470


namespace NUMINAMATH_GPT_votes_switched_l2174_217493

theorem votes_switched (x : ℕ) (total_votes : ℕ) (half_votes : ℕ) 
  (votes_first_round : ℕ) (votes_second_round_winner : ℕ) (votes_second_round_loser : ℕ)
  (cond1 : total_votes = 48000)
  (cond2 : half_votes = total_votes / 2)
  (cond3 : votes_first_round = half_votes)
  (cond4 : votes_second_round_winner = half_votes + x)
  (cond5 : votes_second_round_loser = half_votes - x)
  (cond6 : votes_second_round_winner = 5 * votes_second_round_loser) :
  x = 16000 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_votes_switched_l2174_217493


namespace NUMINAMATH_GPT_problem_statement_l2174_217475

noncomputable def a (k : ℕ) : ℝ := 2^k / (3^(2^k) + 1)
noncomputable def A : ℝ := (Finset.range 10).sum (λ k => a k)
noncomputable def B : ℝ := (Finset.range 10).prod (λ k => a k)

theorem problem_statement : A / B = (3^(2^10) - 1) / 2^47 - 1 / 2^36 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2174_217475
