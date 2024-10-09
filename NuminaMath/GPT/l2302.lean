import Mathlib

namespace time_for_C_to_complete_work_l2302_230232

variable (A B C : ℕ) (R : ℚ)

def work_completion_in_days (days : ℕ) (portion : ℚ) :=
  portion = 1 / days

theorem time_for_C_to_complete_work :
  work_completion_in_days A 8 →
  work_completion_in_days B 12 →
  work_completion_in_days (A + B + C) 4 →
  C = 24 :=
by
  sorry

end time_for_C_to_complete_work_l2302_230232


namespace simplify_and_evaluate_l2302_230210

variable (a : ℕ)

theorem simplify_and_evaluate :
  (a^2 - 4) / a^2 / (1 - 2 / a) = 7 / 5 :=
by
  -- Assign the condition
  let a := 5
  sorry -- skip the proof

end simplify_and_evaluate_l2302_230210


namespace xyz_inequality_l2302_230294

theorem xyz_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ (3/4) :=
sorry

end xyz_inequality_l2302_230294


namespace good_students_options_l2302_230235

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end good_students_options_l2302_230235


namespace perfect_squares_digit_4_5_6_l2302_230248

theorem perfect_squares_digit_4_5_6 (n : ℕ) (hn : n^2 < 2000) : 
  (∃ k : ℕ, k = 18) :=
  sorry

end perfect_squares_digit_4_5_6_l2302_230248


namespace sqrt_factorial_div_l2302_230219

theorem sqrt_factorial_div:
  Real.sqrt (↑(Nat.factorial 9) / 90) = 4 * Real.sqrt 42 := 
by
  -- Steps of the proof
  sorry

end sqrt_factorial_div_l2302_230219


namespace trees_total_count_l2302_230270

theorem trees_total_count (D P : ℕ) 
  (h1 : D = 350 ∨ P = 350)
  (h2 : 300 * D + 225 * P = 217500) :
  D + P = 850 :=
by
  sorry

end trees_total_count_l2302_230270


namespace turnover_threshold_l2302_230292

-- Definitions based on the problem conditions
def valid_domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def daily_turnover (x : ℝ) : ℝ := 20 * (10 - x) * (50 + 8 * x)

-- Lean 4 statement equivalent to mathematical proof problem
theorem turnover_threshold (x : ℝ) (hx : valid_domain x) (h_turnover : daily_turnover x ≥ 10260) :
  x ≥ 1 / 2 ∧ x ≤ 2 :=
sorry

end turnover_threshold_l2302_230292


namespace orchids_initially_l2302_230255

-- Definitions and Conditions
def initial_orchids (current_orchids: ℕ) (cut_orchids: ℕ) : ℕ :=
  current_orchids + cut_orchids

-- Proof statement
theorem orchids_initially (current_orchids: ℕ) (cut_orchids: ℕ) : initial_orchids current_orchids cut_orchids = 3 :=
by 
  have h1 : current_orchids = 7 := sorry
  have h2 : cut_orchids = 4 := sorry
  have h3 : initial_orchids current_orchids cut_orchids = 7 + 4 := sorry
  have h4 : initial_orchids current_orchids cut_orchids = 3 := sorry
  sorry

end orchids_initially_l2302_230255


namespace distinct_bead_arrangements_on_bracelet_l2302_230209

open Nat

-- Definition of factorial
def fact : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * fact n

-- Theorem stating the number of distinct arrangements of 7 beads on a bracelet
theorem distinct_bead_arrangements_on_bracelet : 
  fact 7 / 14 = 360 := 
by 
  sorry

end distinct_bead_arrangements_on_bracelet_l2302_230209


namespace correct_option_l2302_230285

def monomial_structure_same (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

def monomial1 : ℕ → ℕ
| 0 => 1 -- Exponent of a in 3ab^2
| 1 => 2 -- Exponent of b in 3ab^2
| _ => 0

def monomial2 : ℕ → ℕ
| 0 => 1 -- Exponent of a in 4ab^2
| 1 => 2 -- Exponent of b in 4ab^2
| _ => 0

theorem correct_option :
  monomial_structure_same monomial1 monomial2 := sorry

end correct_option_l2302_230285


namespace austin_pairs_of_shoes_l2302_230200

theorem austin_pairs_of_shoes (S : ℕ) :
  0.45 * (S : ℝ) + 11 = S → S / 2 = 10 :=
by
  sorry

end austin_pairs_of_shoes_l2302_230200


namespace min_time_to_shoe_horses_l2302_230289

-- Definitions based on the conditions
def n_blacksmiths : ℕ := 48
def n_horses : ℕ := 60
def t_hoof : ℕ := 5 -- minutes per hoof
def n_hooves : ℕ := n_horses * 4
def total_time : ℕ := n_hooves * t_hoof
def t_min : ℕ := total_time / n_blacksmiths

-- The theorem states that the minimal time required is 25 minutes
theorem min_time_to_shoe_horses : t_min = 25 := by
  sorry

end min_time_to_shoe_horses_l2302_230289


namespace Aunt_Lucy_gift_correct_l2302_230238

def Jade_initial : ℕ := 38
def Julia_initial : ℕ := Jade_initial / 2
def Jack_initial : ℕ := 12
def John_initial : ℕ := 15
def Jane_initial : ℕ := 20

def Aunt_Mary_gift : ℕ := 65
def Aunt_Susan_gift : ℕ := 70

def total_initial : ℕ :=
  Jade_initial + Julia_initial + Jack_initial + John_initial + Jane_initial

def total_after_gifts : ℕ := 225
def total_gifts : ℕ := total_after_gifts - total_initial
def Aunt_Lucy_gift : ℕ := total_gifts - (Aunt_Mary_gift + Aunt_Susan_gift)

theorem Aunt_Lucy_gift_correct :
  Aunt_Lucy_gift = total_after_gifts - total_initial - (Aunt_Mary_gift + Aunt_Susan_gift) := by
  sorry

end Aunt_Lucy_gift_correct_l2302_230238


namespace tiling_problem_l2302_230230

theorem tiling_problem (n : ℕ) : 
  (∃ (k : ℕ), k > 1 ∧ n = 4 * k) 
  ↔ (∃ (L_tile T_tile : ℕ), n * n = 3 * L_tile + 4 * T_tile) :=
by
  sorry

end tiling_problem_l2302_230230


namespace trench_digging_l2302_230243

theorem trench_digging 
  (t : ℝ) (T : ℝ) (work_units : ℝ)
  (h1 : 4 * t = 10)
  (h2 : T = 5 * t) :
  work_units = 80 :=
by
  sorry

end trench_digging_l2302_230243


namespace trig_identity_l2302_230213

theorem trig_identity : 4 * Real.sin (15 * Real.pi / 180) * Real.sin (105 * Real.pi / 180) = 1 :=
by
  sorry

end trig_identity_l2302_230213


namespace rabbit_weight_l2302_230261

variable (k r p : ℝ)

theorem rabbit_weight :
  k + r + p = 39 →
  r + p = 3 * k →
  r + k = 1.5 * p →
  r = 13.65 :=
by
  intros h1 h2 h3
  sorry

end rabbit_weight_l2302_230261


namespace simplify_expression_l2302_230201

variable (x y : ℕ)

theorem simplify_expression :
  7 * x + 9 * y + 3 - x + 12 * y + 15 = 6 * x + 21 * y + 18 :=
by
  sorry

end simplify_expression_l2302_230201


namespace range_of_a_l2302_230256

noncomputable def f (x a : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x
noncomputable def f' (x a : ℝ) : ℝ := 1 - (2 / 3) * Real.cos (2 * x) + a * Real.cos x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f' x a ≥ 0) ↔ -1 / 3 ≤ a ∧ a ≤ 1 / 3 :=
sorry

end range_of_a_l2302_230256


namespace soda_price_before_increase_l2302_230206

theorem soda_price_before_increase
  (candy_box_after : ℝ)
  (soda_after : ℝ)
  (candy_box_increase : ℝ)
  (soda_increase : ℝ)
  (new_price_soda : soda_after = 9)
  (new_price_candy_box : candy_box_after = 10)
  (percent_candy_box_increase : candy_box_increase = 0.25)
  (percent_soda_increase : soda_increase = 0.50) :
  ∃ P : ℝ, 1.5 * P = 9 ∧ P = 6 := 
by
  sorry

end soda_price_before_increase_l2302_230206


namespace lisa_speed_correct_l2302_230217

def eugene_speed := 5

def carlos_speed := (3 / 4) * eugene_speed

def lisa_speed := (4 / 3) * carlos_speed

theorem lisa_speed_correct : lisa_speed = 5 := by
  sorry

end lisa_speed_correct_l2302_230217


namespace first_reduction_is_12_percent_l2302_230244

theorem first_reduction_is_12_percent (P : ℝ) (x : ℝ) (h1 : (1 - x / 100) * 0.9 * P = 0.792 * P) : x = 12 :=
by
  sorry

end first_reduction_is_12_percent_l2302_230244


namespace solve_equation_l2302_230273

theorem solve_equation :
  ∀ x : ℝ, 4 * x * (6 * x - 1) = 1 - 6 * x ↔ (x = 1/6 ∨ x = -1/4) := 
by
  sorry

end solve_equation_l2302_230273


namespace time_to_cover_length_l2302_230267

def escalator_rate : ℝ := 12 -- rate of the escalator in feet per second
def person_rate : ℝ := 8 -- rate of the person in feet per second
def escalator_length : ℝ := 160 -- length of the escalator in feet

theorem time_to_cover_length : escalator_length / (escalator_rate + person_rate) = 8 := by
  sorry

end time_to_cover_length_l2302_230267


namespace smallest_M_satisfying_conditions_l2302_230276

theorem smallest_M_satisfying_conditions :
  ∃ M : ℕ, M > 0 ∧ M = 250 ∧
    ( (M % 125 = 0 ∧ ((M + 1) % 8 = 0 ∧ (M + 2) % 9 = 0) ∨ ((M + 1) % 9 = 0 ∧ (M + 2) % 8 = 0)) ∨
      (M % 8 = 0 ∧ ((M + 1) % 125 = 0 ∧ (M + 2) % 9 = 0) ∨ ((M + 1) % 9 = 0 ∧ (M + 2) % 125 = 0)) ∨
      (M % 9 = 0 ∧ ((M + 1) % 8 = 0 ∧ (M + 2) % 125 = 0) ∨ ((M + 1) % 125 = 0 ∧ (M + 2) % 8 = 0)) ) :=
by
  sorry

end smallest_M_satisfying_conditions_l2302_230276


namespace vertical_axis_residuals_of_residual_plot_l2302_230205

theorem vertical_axis_residuals_of_residual_plot :
  ∀ (vertical_axis : Type), 
  (vertical_axis = Residuals ∨ 
   vertical_axis = SampleNumber ∨ 
   vertical_axis = EstimatedValue) →
  (vertical_axis = Residuals) :=
by
  sorry

end vertical_axis_residuals_of_residual_plot_l2302_230205


namespace line_intersects_ellipse_slopes_l2302_230291

theorem line_intersects_ellipse_slopes (m : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ 
  m ∈ Set.Iic (-Real.sqrt (1/5)) ∨ m ∈ Set.Ici (Real.sqrt (1/5)) :=
by
  sorry

end line_intersects_ellipse_slopes_l2302_230291


namespace courtyard_length_proof_l2302_230298

noncomputable def paving_stone_area (length width : ℝ) : ℝ := length * width

noncomputable def total_area_stones (stone_area : ℝ) (num_stones : ℝ) : ℝ := stone_area * num_stones

noncomputable def courtyard_length (total_area width : ℝ) : ℝ := total_area / width

theorem courtyard_length_proof :
  let stone_length := 2.5
  let stone_width := 2
  let courtyard_width := 16.5
  let num_stones := 99
  let stone_area := paving_stone_area stone_length stone_width
  let total_area := total_area_stones stone_area num_stones
  courtyard_length total_area courtyard_width = 30 :=
by
  sorry

end courtyard_length_proof_l2302_230298


namespace exists_q_no_zero_in_decimal_l2302_230226

theorem exists_q_no_zero_in_decimal : ∃ q : ℕ, ∀ (d : ℕ), q * 2 ^ 1967 ≠ 10 * d := 
sorry

end exists_q_no_zero_in_decimal_l2302_230226


namespace smallest_same_terminal_1000_l2302_230253

def has_same_terminal_side (theta phi : ℝ) : Prop :=
  ∃ n : ℤ, theta = phi + n * 360

theorem smallest_same_terminal_1000 : ∀ θ : ℝ,
  θ ≥ 0 → θ < 360 → has_same_terminal_side θ 1000 → θ = 280 :=
by
  sorry

end smallest_same_terminal_1000_l2302_230253


namespace sum_of_c_l2302_230224

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * n - 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℕ :=
  2^(n - 1)

-- Define the sequence c_n
def c (n : ℕ) : ℕ :=
  a n * b n

-- Define the sum S_n of the first n terms of c_n
def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => c (i + 1))

-- The main Lean statement
theorem sum_of_c (n : ℕ) : S n = 3 + (n - 1) * 2^(n + 1) :=
  sorry

end sum_of_c_l2302_230224


namespace find_prime_p_l2302_230239

def f (x : ℕ) : ℕ :=
  (x^4 + 2 * x^3 + 4 * x^2 + 2 * x + 1)^5

theorem find_prime_p : ∃! p, Nat.Prime p ∧ f p = 418195493 := by
  sorry

end find_prime_p_l2302_230239


namespace sum_of_three_squares_l2302_230222

theorem sum_of_three_squares (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 28) :
  x + y + z = 6 * Real.sqrt 3 := by
  sorry

end sum_of_three_squares_l2302_230222


namespace evaluate_polynomial_at_6_l2302_230263

def polynomial (x : ℝ) : ℝ := 2 * x^4 + 5 * x^3 - x^2 + 3 * x + 4

theorem evaluate_polynomial_at_6 : polynomial 6 = 3658 :=
by 
  sorry

end evaluate_polynomial_at_6_l2302_230263


namespace find_a1_l2302_230227

noncomputable def a (n : ℕ) : ℤ := sorry -- the definition of sequence a_n is not computable without initial terms
noncomputable def S (n : ℕ) : ℤ := sorry -- similarly, the definition of S_n without initial terms isn't given

axiom recurrence_relation (n : ℕ) (h : n ≥ 3): 
  a (n) = a (n - 1) - a (n - 2)

axiom S9 : S 9 = 6
axiom S10 : S 10 = 5

theorem find_a1 : a 1 = 1 :=
by
  sorry

end find_a1_l2302_230227


namespace problem_solution_l2302_230293

-- Definitions of sets A and B
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 }
def B : Set ℝ := {-2, -1, 1, 2}

-- Complement of set A in reals
def C_A : Set ℝ := {x | x < 0}

-- Lean theorem statement
theorem problem_solution : (C_A ∩ B) = {-2, -1} :=
by sorry

end problem_solution_l2302_230293


namespace derivative_at_pi_over_six_l2302_230274

-- Define the function f(x) = cos(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- State the theorem: the derivative of f at π/6 is -1/2
theorem derivative_at_pi_over_six : deriv f (Real.pi / 6) = -1 / 2 :=
by sorry

end derivative_at_pi_over_six_l2302_230274


namespace anthony_ate_total_l2302_230297

def slices := 16

def ate_alone := 1 / slices
def shared_with_ben := (1 / 2) * (1 / slices)
def shared_with_chris := (1 / 2) * (1 / slices)

theorem anthony_ate_total :
  ate_alone + shared_with_ben + shared_with_chris = 1 / 8 :=
by
  sorry

end anthony_ate_total_l2302_230297


namespace trainers_hours_split_equally_l2302_230221

noncomputable def dolphins := 12
noncomputable def hours_per_dolphin := 5
noncomputable def trainers := 4

theorem trainers_hours_split_equally :
  (dolphins * hours_per_dolphin) / trainers = 15 :=
by
  sorry

end trainers_hours_split_equally_l2302_230221


namespace M_inter_N_l2302_230265

def M : Set ℝ := {x | x^2 - x = 0}
def N : Set ℝ := {-1, 0}

theorem M_inter_N :
  M ∩ N = {0} :=
by
  sorry

end M_inter_N_l2302_230265


namespace sin_15_mul_sin_75_l2302_230252

theorem sin_15_mul_sin_75 : Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 4 := 
by
  sorry

end sin_15_mul_sin_75_l2302_230252


namespace circle_radius_l2302_230203

theorem circle_radius (M N r : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 10) : r = 20 :=
by
  sorry

end circle_radius_l2302_230203


namespace range_of_a_l2302_230202

theorem range_of_a (a : ℝ) : 
  ( ∃ x y : ℝ, (x^2 + 4 * (y - a)^2 = 4) ∧ (x^2 = 4 * y)) ↔ a ∈ Set.Ico (-1 : ℝ) (5 / 4 : ℝ) := 
sorry

end range_of_a_l2302_230202


namespace midpoint_trajectory_of_chord_l2302_230254

theorem midpoint_trajectory_of_chord {x y : ℝ} :
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 / 3 + A.2^2 = 1) ∧ 
    (B.1^2 / 3 + B.2^2 = 1) ∧ 
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (x, y) ∧ 
    ∃ t : ℝ, ((-1, 0) = ((1 - t) * A.1 + t * B.1, (1 - t) * A.2 + t * B.2))) -> 
  x^2 + x + 3 * y^2 = 0 :=
by sorry

end midpoint_trajectory_of_chord_l2302_230254


namespace percentage_of_nine_hundred_l2302_230288

theorem percentage_of_nine_hundred : (45 * 8 = 360) ∧ ((360 / 900) * 100 = 40) :=
by
  have h1 : 45 * 8 = 360 := by sorry
  have h2 : (360 / 900) * 100 = 40 := by sorry
  exact ⟨h1, h2⟩

end percentage_of_nine_hundred_l2302_230288


namespace reciprocal_square_inequality_l2302_230282

variable (x y : ℝ)
variable (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x ≤ y)

theorem reciprocal_square_inequality :
  (1 / y^2) ≤ (1 / x^2) :=
sorry

end reciprocal_square_inequality_l2302_230282


namespace prove_cartesian_eq_C1_prove_cartesian_eq_C2_prove_min_distance_C1_C2_l2302_230204

noncomputable def cartesian_eq_C1 (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 1)^2 = 4

noncomputable def cartesian_eq_C2 (x y : ℝ) : Prop :=
  (4 * x - y - 1 = 0)

noncomputable def min_distance_C1_C2 : ℝ :=
  (10 * Real.sqrt 17 / 17) - 2

theorem prove_cartesian_eq_C1 (x y t : ℝ) (h : x = -2 + 2 * Real.cos t ∧ y = 1 + 2 * Real.sin t) :
  cartesian_eq_C1 x y :=
sorry

theorem prove_cartesian_eq_C2 (ρ θ : ℝ) (h : 4 * ρ * Real.cos θ - ρ * Real.sin θ - 1 = 0) :
  cartesian_eq_C2 (ρ * Real.cos θ) (ρ * Real.sin θ) :=
sorry

theorem prove_min_distance_C1_C2 (h1 : ∀ x y, cartesian_eq_C1 x y) (h2 : ∀ x y, cartesian_eq_C2 x y) :
  ∀ P Q : ℝ × ℝ, (cartesian_eq_C1 P.1 P.2) → (cartesian_eq_C2 Q.1 Q.2) →
  (min_distance_C1_C2 = (Real.sqrt (4^2 + (-1)^2) / Real.sqrt 17) - 2) :=
sorry

end prove_cartesian_eq_C1_prove_cartesian_eq_C2_prove_min_distance_C1_C2_l2302_230204


namespace min_speed_A_l2302_230241

theorem min_speed_A (V_B V_C V_A : ℕ) (d_AB d_AC wind extra_speed : ℕ) :
  V_B = 50 →
  V_C = 70 →
  d_AB = 40 →
  d_AC = 280 →
  wind = 5 →
  V_A > ((d_AB * (V_A + wind + extra_speed)) / (d_AC - d_AB) - wind) :=
sorry

end min_speed_A_l2302_230241


namespace depth_of_well_l2302_230233

theorem depth_of_well 
  (t1 t2 : ℝ) 
  (d : ℝ) 
  (h1: t1 + t2 = 8) 
  (h2: d = 32 * t1^2) 
  (h3: t2 = d / 1100) 
  : d = 1348 := 
  sorry

end depth_of_well_l2302_230233


namespace no_integer_solutions_l2302_230295

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by {
  sorry
}

end no_integer_solutions_l2302_230295


namespace triangle_problem_l2302_230218

open Real

theorem triangle_problem (a b S : ℝ) (A B : ℝ) (hA_cos : cos A = (sqrt 6) / 3) (hA_val : a = 3) (hB_val : B = A + π / 2):
  b = 3 * sqrt 2 ∧
  S = (3 * sqrt 2) / 2 :=
by
  sorry

end triangle_problem_l2302_230218


namespace bianca_drawing_time_at_home_l2302_230214

-- Define the conditions
def drawing_time_at_school : ℕ := 22
def total_drawing_time : ℕ := 41

-- Define the calculation for drawing time at home
def drawing_time_at_home : ℕ := total_drawing_time - drawing_time_at_school

-- The proof goal
theorem bianca_drawing_time_at_home : drawing_time_at_home = 19 := by
  sorry

end bianca_drawing_time_at_home_l2302_230214


namespace min_sine_difference_l2302_230279

theorem min_sine_difference (N : ℕ) (hN : 0 < N) :
  ∃ (n k : ℕ), (1 ≤ n ∧ n ≤ N + 1) ∧ (1 ≤ k ∧ k ≤ N + 1) ∧ (n ≠ k) ∧ 
    (|Real.sin n - Real.sin k| < 2 / N) := 
sorry

end min_sine_difference_l2302_230279


namespace time_interval_for_birth_and_death_rates_l2302_230258

theorem time_interval_for_birth_and_death_rates
  (birth_rate : ℝ)
  (death_rate : ℝ)
  (population_net_increase_per_day : ℝ)
  (number_of_minutes_per_day : ℝ)
  (net_increase_per_interval : ℝ)
  (time_intervals_per_day : ℝ)
  (time_interval_in_minutes : ℝ):

  birth_rate = 10 →
  death_rate = 2 →
  population_net_increase_per_day = 345600 →
  number_of_minutes_per_day = 1440 →
  net_increase_per_interval = birth_rate - death_rate →
  time_intervals_per_day = population_net_increase_per_day / net_increase_per_interval →
  time_interval_in_minutes = number_of_minutes_per_day / time_intervals_per_day →
  time_interval_in_minutes = 48 :=
by
  intros
  sorry

end time_interval_for_birth_and_death_rates_l2302_230258


namespace dividend_rate_is_16_l2302_230281

noncomputable def dividend_rate_of_shares : ℝ :=
  let share_value := 48
  let interest_rate := 0.12
  let market_value := 36.00000000000001
  (interest_rate * share_value) / market_value * 100

theorem dividend_rate_is_16 :
  dividend_rate_of_shares = 16 := by
  sorry

end dividend_rate_is_16_l2302_230281


namespace proposition_relationship_l2302_230246
-- Import library

-- Statement of the problem
theorem proposition_relationship (p q : Prop) (hpq : p ∨ q) (hnp : ¬p) : ¬p ∧ q :=
  by
  sorry

end proposition_relationship_l2302_230246


namespace simplify_and_evaluate_expression_l2302_230228

theorem simplify_and_evaluate_expression 
  (x y : ℤ) (hx : x = -3) (hy : y = -2) :
  3 * x^2 * y - (2 * x^2 * y - (2 * x * y - x^2 * y) - 4 * x^2 * y) - x * y = -66 :=
by
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l2302_230228


namespace model_height_l2302_230290

noncomputable def H_actual : ℝ := 50
noncomputable def A_actual : ℝ := 25
noncomputable def A_model : ℝ := 0.025

theorem model_height : 
  let ratio := (A_actual / A_model)
  ∃ h : ℝ, h = H_actual / (Real.sqrt ratio) ∧ h = 5 * Real.sqrt 10 := 
by 
  sorry

end model_height_l2302_230290


namespace odd_function_value_l2302_230283

theorem odd_function_value (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_fx : ∀ x : ℝ, x ≤ 0 → f x = 2 * x ^ 2 - x) :
  f 1 = -3 := 
sorry

end odd_function_value_l2302_230283


namespace sum_of_angles_in_figure_l2302_230220

theorem sum_of_angles_in_figure : 
  let triangles := 3
  let angles_in_triangle := 180
  let square_angles := 4 * 90
  (triangles * angles_in_triangle + square_angles) = 900 := by
  sorry

end sum_of_angles_in_figure_l2302_230220


namespace gum_distribution_l2302_230211

theorem gum_distribution : 
  ∀ (John Cole Aubrey: ℕ), 
    John = 54 → 
    Cole = 45 → 
    Aubrey = 0 → 
    ((John + Cole + Aubrey) / 3) = 33 := 
by
  intros John Cole Aubrey hJohn hCole hAubrey
  sorry

end gum_distribution_l2302_230211


namespace merchant_profit_percentage_l2302_230275

theorem merchant_profit_percentage 
    (cost_price : ℝ) 
    (markup_percentage : ℝ) 
    (discount_percentage : ℝ) 
    (h1 : cost_price = 100) 
    (h2 : markup_percentage = 0.20) 
    (h3 : discount_percentage = 0.05) 
    : ((cost_price * (1 + markup_percentage) * (1 - discount_percentage) - cost_price) / cost_price * 100) = 14 := 
by 
    sorry

end merchant_profit_percentage_l2302_230275


namespace solve_system_l2302_230286

theorem solve_system (X Y : ℝ) : 
  (X + (X + 2 * Y) / (X^2 + Y^2) = 2 ∧ Y + (2 * X - Y) / (X^2 + Y^2) = 0) ↔ (X = 0 ∧ Y = 1) ∨ (X = 2 ∧ Y = -1) :=
by
  sorry

end solve_system_l2302_230286


namespace sum_of_cubes_l2302_230250

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l2302_230250


namespace Ram_has_amount_l2302_230212

theorem Ram_has_amount (R G K : ℕ)
    (h1 : R = 7 * G / 17)
    (h2 : G = 7 * K / 17)
    (h3 : K = 3757) : R = 637 := by
  sorry

end Ram_has_amount_l2302_230212


namespace two_digit_number_count_four_digit_number_count_l2302_230280

-- Defining the set of digits
def digits : Finset ℕ := {1, 2, 3, 4}

-- Problem 1 condition and question
def two_digit_count := Nat.choose 4 2 * 2

-- Problem 2 condition and question
def four_digit_count := Nat.choose 4 4 * 24

-- Theorem statement for Problem 1
theorem two_digit_number_count : two_digit_count = 12 :=
sorry

-- Theorem statement for Problem 2
theorem four_digit_number_count : four_digit_count = 24 :=
sorry

end two_digit_number_count_four_digit_number_count_l2302_230280


namespace mrs_generous_jelly_beans_l2302_230216

-- Define necessary terms and state the problem
def total_children (x : ℤ) : ℤ := x + (x + 3)

theorem mrs_generous_jelly_beans :
  ∃ x : ℤ, x^2 + (x + 3)^2 = 490 ∧ total_children x = 31 :=
by {
  sorry
}

end mrs_generous_jelly_beans_l2302_230216


namespace sum_central_square_l2302_230257

noncomputable def table_sum : ℕ := 10200
noncomputable def a : ℕ := 1200
noncomputable def central_sum : ℕ := 720

theorem sum_central_square :
  ∃ (a : ℕ), table_sum = a * (1 + (1 / 3) + (1 / 9) + (1 / 27)) * (1 + (1 / 4) + (1 / 16) + (1 / 64)) ∧ 
              central_sum = (a / 3) + (a / 12) + (a / 9) + (a / 36) :=
by
  sorry

end sum_central_square_l2302_230257


namespace max_f_l2302_230272

noncomputable def S_n (n : ℕ) : ℚ :=
  n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℚ :=
  S_n n / ((n + 32) * S_n (n + 1))

theorem max_f (n : ℕ) : f n ≤ 1 / 50 := sorry

-- Verify the bound is achieved for n = 8
example : f 8 = 1 / 50 := by
  unfold f S_n
  norm_num

end max_f_l2302_230272


namespace waiters_hired_l2302_230264

theorem waiters_hired (W H : ℕ) (h1 : 3 * W = 90) (h2 : 3 * (W + H) = 126) : H = 12 :=
sorry

end waiters_hired_l2302_230264


namespace Jason_seashells_l2302_230245

theorem Jason_seashells (initial_seashells given_to_Tim remaining_seashells : ℕ) :
  initial_seashells = 49 → given_to_Tim = 13 → remaining_seashells = initial_seashells - given_to_Tim →
  remaining_seashells = 36 :=
by intros; sorry

end Jason_seashells_l2302_230245


namespace reciprocal_of_negative_2023_l2302_230277

theorem reciprocal_of_negative_2023 : (-2023) * (-1 / 2023) = 1 :=
by
  sorry

end reciprocal_of_negative_2023_l2302_230277


namespace correct_calculation_l2302_230271

theorem correct_calculation (N : ℤ) (h : 41 - N = 12) : 41 + N = 70 := 
by 
  sorry

end correct_calculation_l2302_230271


namespace earnings_of_r_l2302_230237

theorem earnings_of_r (P Q R : ℕ) (h1 : 9 * (P + Q + R) = 1710) (h2 : 5 * (P + R) = 600) (h3 : 7 * (Q + R) = 910) : 
  R = 60 :=
by
  -- proof will be provided here
  sorry

end earnings_of_r_l2302_230237


namespace simplify_expression_correct_l2302_230268

-- Defining the problem conditions and required proof
def simplify_expression (x : ℝ) (h : x ≠ 2) : Prop :=
  (x / (x - 2) + 2 / (2 - x) = 1)

-- Stating the theorem
theorem simplify_expression_correct (x : ℝ) (h : x ≠ 2) : simplify_expression x h :=
  by sorry

end simplify_expression_correct_l2302_230268


namespace problem_1_problem_2_problem_3_l2302_230262

def range_1 : Set ℝ :=
  { y | ∃ x : ℝ, y = 1 / (x - 1) ∧ x ≠ 1 }

def range_2 : Set ℝ :=
  { y | ∃ x : ℝ, y = x^2 + 4 * x - 1 }

def range_3 : Set ℝ :=
  { y | ∃ x : ℝ, y = x + Real.sqrt (x + 1) ∧ x ≥ 0 }

theorem problem_1 : range_1 = {y | y < 0 ∨ y > 0} :=
by 
  sorry

theorem problem_2 : range_2 = {y | y ≥ -5} :=
by 
  sorry

theorem problem_3 : range_3 = {y | y ≥ -1} :=
by 
  sorry

end problem_1_problem_2_problem_3_l2302_230262


namespace radius_of_circumscribed_sphere_l2302_230284

-- Condition: SA = 2
def SA : ℝ := 2

-- Condition: SB = 4
def SB : ℝ := 4

-- Condition: SC = 4
def SC : ℝ := 4

-- Condition: The three side edges are pairwise perpendicular.
def pairwise_perpendicular : Prop := true -- This condition is described but would require geometric definition.

-- To prove: Radius of circumscribed sphere is 3
theorem radius_of_circumscribed_sphere : 
  ∀ (SA SB SC : ℝ) (pairwise_perpendicular : Prop), SA = 2 → SB = 4 → SC = 4 → pairwise_perpendicular → 
  (3 : ℝ) = 3 := by 
  intros SA SB SC pairwise_perpendicular h1 h2 h3 h4
  sorry

end radius_of_circumscribed_sphere_l2302_230284


namespace circle_equation_l2302_230278

def circle_center : (ℝ × ℝ) := (1, 2)
def radius : ℝ := 3

theorem circle_equation : 
  (∀ x y : ℝ, (x - circle_center.1) ^ 2 + (y - circle_center.2) ^ 2 = radius ^ 2 ↔ 
  (x - 1) ^ 2 + (y - 2) ^ 2 = 9) := 
by
  sorry

end circle_equation_l2302_230278


namespace inequality_holds_l2302_230242

variable (a b c : ℝ)

theorem inequality_holds : 
  (a * b + b * c + c * a - 1)^2 ≤ (a^2 + 1) * (b^2 + 1) * (c^2 + 1) := 
by 
  sorry

end inequality_holds_l2302_230242


namespace markup_calculation_l2302_230234

def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.25
def net_profit : ℝ := 12

def overhead := purchase_price * overhead_percentage
def total_cost := purchase_price + overhead
def selling_price := total_cost + net_profit
def markup := selling_price - purchase_price

theorem markup_calculation : markup = 24 := by
  sorry

end markup_calculation_l2302_230234


namespace pyramid_base_side_length_l2302_230266

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l2302_230266


namespace fractions_sum_equals_one_l2302_230269

variable {a b c x y z : ℝ}

variables (h1 : 17 * x + b * y + c * z = 0)
variables (h2 : a * x + 29 * y + c * z = 0)
variables (h3 : a * x + b * y + 53 * z = 0)
variables (ha : a ≠ 17)
variables (hx : x ≠ 0)

theorem fractions_sum_equals_one (a b c x y z : ℝ) 
  (h1 : 17 * x + b * y + c * z = 0)
  (h2 : a * x + 29 * y + c * z = 0)
  (h3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 := by 
  sorry

end fractions_sum_equals_one_l2302_230269


namespace find_x_l2302_230251

def vec := (ℝ × ℝ)

def a : vec := (1, 1)
def b (x : ℝ) : vec := (3, x)

def add_vec (v1 v2 : vec) : vec := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : vec) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (x : ℝ) (h : dot_product a (add_vec a (b x)) = 0) : x = -5 :=
by
  -- Proof steps (irrelevant for now)
  sorry

end find_x_l2302_230251


namespace positive_integer_sum_representation_l2302_230223

theorem positive_integer_sum_representation :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → ∃ (a : Fin 2004 → ℕ), 
    (∀ i j : Fin 2004, i < j → a i < a j) ∧ 
    (∀ i : Fin 2003, a i ∣ a (i + 1)) ∧
    (n = (Finset.univ.sum a)) := 
sorry

end positive_integer_sum_representation_l2302_230223


namespace kelly_points_l2302_230225

theorem kelly_points (K : ℕ) 
  (h1 : 12 + 2 * 12 + K + 2 * K + 12 / 2 = 69) : K = 9 := by
  sorry

end kelly_points_l2302_230225


namespace vasya_did_not_buy_anything_days_l2302_230287

theorem vasya_did_not_buy_anything_days :
  ∃ (x y z w : ℕ), 
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 ∧
    w = 7 :=
by sorry

end vasya_did_not_buy_anything_days_l2302_230287


namespace geom_sequence_third_term_l2302_230247

theorem geom_sequence_third_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a n = a 1 * r ^ (n - 1)) (h_cond : a 1 * a 5 = a 3) : a 3 = 1 :=
sorry

end geom_sequence_third_term_l2302_230247


namespace eggs_collected_week_l2302_230207

def num_chickens : ℕ := 6
def num_ducks : ℕ := 4
def num_geese : ℕ := 2
def eggs_per_chicken : ℕ := 3
def eggs_per_duck : ℕ := 2
def eggs_per_goose : ℕ := 1

def eggs_per_day (num_birds eggs_per_bird : ℕ) : ℕ := num_birds * eggs_per_bird

def eggs_collected_monday_to_saturday : ℕ :=
  6 * (eggs_per_day num_chickens eggs_per_chicken +
       eggs_per_day num_ducks eggs_per_duck +
       eggs_per_day num_geese eggs_per_goose)

def eggs_collected_sunday : ℕ :=
  eggs_per_day num_chickens (eggs_per_chicken - 1) +
  eggs_per_day num_ducks (eggs_per_duck - 1) +
  eggs_per_day num_geese (eggs_per_goose - 1)

def total_eggs_collected : ℕ :=
  eggs_collected_monday_to_saturday + eggs_collected_sunday

theorem eggs_collected_week : total_eggs_collected = 184 :=
by sorry

end eggs_collected_week_l2302_230207


namespace only_1996_is_leap_l2302_230208

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0))

def is_leap_year_1996 := is_leap_year 1996
def is_leap_year_1998 := is_leap_year 1998
def is_leap_year_2010 := is_leap_year 2010
def is_leap_year_2100 := is_leap_year 2100

theorem only_1996_is_leap : 
  is_leap_year_1996 ∧ ¬is_leap_year_1998 ∧ ¬is_leap_year_2010 ∧ ¬is_leap_year_2100 :=
by 
  -- proof will be added here later
  sorry

end only_1996_is_leap_l2302_230208


namespace islanders_liars_l2302_230229

inductive Person
| A
| B

open Person

def is_liar (p : Person) : Prop :=
  sorry -- placeholder for the actual definition

def makes_statement (p : Person) (statement : Prop) : Prop :=
  sorry -- placeholder for the actual definition

theorem islanders_liars :
  makes_statement A (is_liar A ∧ ¬ is_liar B) →
  is_liar A ∧ is_liar B :=
by
  sorry

end islanders_liars_l2302_230229


namespace rectangle_ratio_l2302_230299

-- Given conditions
variable (w : ℕ) -- width is a natural number

-- Definitions based on conditions 
def length := 10
def perimeter := 30

-- Theorem to prove
theorem rectangle_ratio (h : 2 * length + 2 * w = perimeter) : w = 5 ∧ 1 = 1 ∧ 2 = 2 :=
by
  sorry

end rectangle_ratio_l2302_230299


namespace drums_of_grapes_per_day_l2302_230249

-- Definitions derived from conditions
def pickers := 235
def raspberry_drums_per_day := 100
def total_days := 77
def total_drums := 17017

-- Prove the main theorem
theorem drums_of_grapes_per_day : (total_drums - total_days * raspberry_drums_per_day) / total_days = 121 := by
  sorry

end drums_of_grapes_per_day_l2302_230249


namespace polynomial_evaluation_l2302_230215

noncomputable def Q (x : ℝ) : ℝ :=
  x^4 + x^3 + 2 * x

theorem polynomial_evaluation :
  Q (3) = 114 := by
  -- We assume the conditions implicitly in this equivalence.
  sorry

end polynomial_evaluation_l2302_230215


namespace no_integer_pairs_satisfy_equation_l2302_230240

theorem no_integer_pairs_satisfy_equation :
  ¬ ∃ m n : ℤ, m^3 + 8 * m^2 + 17 * m = 8 * n^3 + 12 * n^2 + 6 * n + 1 :=
sorry

end no_integer_pairs_satisfy_equation_l2302_230240


namespace valid_parameterizations_l2302_230260

noncomputable def is_scalar_multiple (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def lies_on_line (p : ℝ × ℝ) (m b : ℝ) : Prop :=
  p.2 = m * p.1 + b

def is_valid_parameterization (p d : ℝ × ℝ) (m b : ℝ) : Prop :=
  lies_on_line p m b ∧ is_scalar_multiple d (2, 1)

theorem valid_parameterizations :
  (is_valid_parameterization (7, 18) (-1, -2) 2 4) ∧
  (is_valid_parameterization (1, 6) (5, 10) 2 4) ∧
  (is_valid_parameterization (2, 8) (20, 40) 2 4) ∧
  ¬ (is_valid_parameterization (-4, -4) (1, -1) 2 4) ∧
  ¬ (is_valid_parameterization (-3, -2) (0.5, 1) 2 4) :=
by {
  sorry
}

end valid_parameterizations_l2302_230260


namespace insulation_cost_per_sq_ft_l2302_230296

theorem insulation_cost_per_sq_ft 
  (l w h : ℤ) 
  (surface_area : ℤ := (2 * l * w) + (2 * l * h) + (2 * w * h))
  (total_cost : ℤ)
  (cost_per_sq_ft : ℤ := total_cost / surface_area)
  (h_l : l = 3)
  (h_w : w = 5)
  (h_h : h = 2)
  (h_total_cost : total_cost = 1240) :
  cost_per_sq_ft = 20 := 
by
  sorry

end insulation_cost_per_sq_ft_l2302_230296


namespace derivative_at_one_l2302_230231

-- Definition of the function
def f (x : ℝ) : ℝ := x^2

-- Condition
def x₀ : ℝ := 1

-- Problem statement
theorem derivative_at_one : (deriv f x₀) = 2 :=
sorry

end derivative_at_one_l2302_230231


namespace smallest_non_unit_digit_multiple_of_five_l2302_230259

theorem smallest_non_unit_digit_multiple_of_five :
  ∀ (d : ℕ), ((d = 0) ∨ (d = 5)) → (d ≠ 1 ∧ d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 6 ∧ d ≠ 7 ∧ d ≠ 8 ∧ d ≠ 9) :=
by {
  sorry
}

end smallest_non_unit_digit_multiple_of_five_l2302_230259


namespace fred_final_baseball_cards_l2302_230236

-- Conditions
def initial_cards : ℕ := 25
def sold_to_melanie : ℕ := 7
def traded_with_kevin : ℕ := 3
def bought_from_alex : ℕ := 5

-- Proof statement (Lean theorem)
theorem fred_final_baseball_cards : initial_cards - sold_to_melanie - traded_with_kevin + bought_from_alex = 20 := by
  sorry

end fred_final_baseball_cards_l2302_230236
