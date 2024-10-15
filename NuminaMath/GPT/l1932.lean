import Mathlib

namespace NUMINAMATH_GPT_solution_set_of_inequality_l1932_193251

variable {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f x

theorem solution_set_of_inequality
  (f : R → R)
  (odd_f : odd_function f)
  (h1 : f (-2) = 0)
  (h2 : ∀ (x1 x2 : R), x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 → (x2 * f x1 - x1 * f x2) / (x1 - x2) < 0) :
  { x : R | (f x) / x < 0 } = { x : R | x < -2 } ∪ { x : R | x > 2 } := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1932_193251


namespace NUMINAMATH_GPT_fraction_of_integer_l1932_193280

theorem fraction_of_integer :
  (5 / 6) * 30 = 25 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_integer_l1932_193280


namespace NUMINAMATH_GPT_movie_duration_l1932_193254

theorem movie_duration :
  let start_time := (13, 30)
  let end_time := (14, 50)
  let hours := end_time.1 - start_time.1
  let minutes := end_time.2 - start_time.2
  (if minutes < 0 then (hours - 1, minutes + 60) else (hours, minutes)) = (1, 20) := by
    sorry

end NUMINAMATH_GPT_movie_duration_l1932_193254


namespace NUMINAMATH_GPT_geometric_sequence_a3_l1932_193299

theorem geometric_sequence_a3 (a : ℕ → ℝ)
  (h : ∀ n m : ℕ, a (n + m) = a n * a m)
  (pos : ∀ n, 0 < a n)
  (a1 : a 1 = 1)
  (a5 : a 5 = 9) :
  a 3 = 3 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_l1932_193299


namespace NUMINAMATH_GPT_peanut_mixture_l1932_193215

-- Definitions of given conditions
def virginia_peanuts_weight : ℝ := 10
def virginia_peanuts_cost_per_pound : ℝ := 3.50
def spanish_peanuts_cost_per_pound : ℝ := 3.00
def texan_peanuts_cost_per_pound : ℝ := 4.00
def desired_cost_per_pound : ℝ := 3.60

-- Definitions of unknowns S (Spanish peanuts) and T (Texan peanuts)
variable (S T : ℝ)

-- Equation derived from given conditions
theorem peanut_mixture :
  (0.40 * T) - (0.60 * S) = 1 := sorry

end NUMINAMATH_GPT_peanut_mixture_l1932_193215


namespace NUMINAMATH_GPT_length_of_diagonal_l1932_193278

open Real

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, -a^2)
noncomputable def B (a : ℝ) : ℝ × ℝ := (-a, -a^2)
noncomputable def C (a : ℝ) : ℝ × ℝ := (a, -a^2)
def O : ℝ × ℝ := (0, 0)

noncomputable def is_square (A B O C : ℝ × ℝ) : Prop :=
  dist A B = dist B O ∧ dist B O = dist O C ∧ dist O C = dist C A

theorem length_of_diagonal (a : ℝ) (h_square : is_square (A a) (B a) O (C a)) : 
  dist (A a) (C a) = 2 * abs a :=
sorry

end NUMINAMATH_GPT_length_of_diagonal_l1932_193278


namespace NUMINAMATH_GPT_total_amount_received_l1932_193289

theorem total_amount_received (P R CI: ℝ) (T: ℕ) 
  (compound_interest_eq: CI = P * ((1 + R / 100) ^ T - 1)) 
  (P_eq: P = 2828.80 / 0.1664) 
  (R_eq: R = 8) 
  (T_eq: T = 2) : 
  P + CI = 19828.80 := 
by 
  sorry

end NUMINAMATH_GPT_total_amount_received_l1932_193289


namespace NUMINAMATH_GPT_product_of_roots_l1932_193261

noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

noncomputable def f_prime (a b c : ℝ) (x : ℝ) : ℝ :=
  3 * a * x^2 + 2 * b * x + c

theorem product_of_roots (a b c d x₁ x₂ : ℝ) 
  (h1 : f a b c d 0 = 0)
  (h2 : f a b c d x₁ = 0)
  (h3 : f a b c d x₂ = 0)
  (h_ext1 : f_prime a b c 1 = 0)
  (h_ext2 : f_prime a b c 2 = 0) :
  x₁ * x₂ = 6 :=
sorry

end NUMINAMATH_GPT_product_of_roots_l1932_193261


namespace NUMINAMATH_GPT_tangent_line_equation_l1932_193264

theorem tangent_line_equation (x y : ℝ) :
  (y = Real.exp x + 2) →
  (x = 0) →
  (y = 3) →
  (Real.exp x = 1) →
  (x - y + 3 = 0) :=
by
  intros h_eq h_x h_y h_slope
  -- The following proof will use the conditions to show the tangent line equation.
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l1932_193264


namespace NUMINAMATH_GPT_minimum_distance_from_parabola_to_circle_l1932_193209

noncomputable def minimum_distance_sum : ℝ :=
  let focus : ℝ × ℝ := (1, 0)
  let center : ℝ × ℝ := (0, 4)
  let radius : ℝ := 1
  let distance_from_focus_to_center : ℝ := Real.sqrt ((focus.1 - center.1)^2 + (focus.2 - center.2)^2)
  distance_from_focus_to_center - radius

theorem minimum_distance_from_parabola_to_circle : minimum_distance_sum = Real.sqrt 17 - 1 := by
  sorry

end NUMINAMATH_GPT_minimum_distance_from_parabola_to_circle_l1932_193209


namespace NUMINAMATH_GPT_integer_add_results_in_perfect_square_l1932_193200

theorem integer_add_results_in_perfect_square (x a b : ℤ) :
  (x + 100 = a^2 ∧ x + 164 = b^2) → (x = 125 ∨ x = -64 ∨ x = -100) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_integer_add_results_in_perfect_square_l1932_193200


namespace NUMINAMATH_GPT_propA_propB_relation_l1932_193291

variable (x y : ℤ)

theorem propA_propB_relation :
  (x + y ≠ 5 → x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → x + y ≠ 5) :=
by
  sorry

end NUMINAMATH_GPT_propA_propB_relation_l1932_193291


namespace NUMINAMATH_GPT_weekly_milk_production_l1932_193253

-- Conditions
def number_of_cows : ℕ := 52
def milk_per_cow_per_day : ℕ := 1000
def days_in_week : ℕ := 7

-- Statement to prove
theorem weekly_milk_production : (number_of_cows * milk_per_cow_per_day * days_in_week) = 364000 := by
  sorry

end NUMINAMATH_GPT_weekly_milk_production_l1932_193253


namespace NUMINAMATH_GPT_steve_paid_18_l1932_193225

-- Define the conditions
def mike_price : ℝ := 5
def steve_multiplier : ℝ := 2
def shipping_rate : ℝ := 0.8

-- Define Steve's cost calculation
def steve_total_cost : ℝ :=
  let steve_dvd_price := steve_multiplier * mike_price
  let shipping_cost := shipping_rate * steve_dvd_price
  steve_dvd_price + shipping_cost

-- Prove that Steve's total payment is 18.
theorem steve_paid_18 : steve_total_cost = 18 := by
  -- Provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_steve_paid_18_l1932_193225


namespace NUMINAMATH_GPT_travel_time_l1932_193277

noncomputable def distance (time: ℝ) (rate: ℝ) : ℝ := time * rate

theorem travel_time
  (initial_time: ℝ)
  (initial_speed: ℝ)
  (reduced_speed: ℝ)
  (stopover: ℝ)
  (h1: initial_time = 4)
  (h2: initial_speed = 80)
  (h3: reduced_speed = 50)
  (h4: stopover = 0.5) :
  (distance initial_time initial_speed) / reduced_speed + stopover = 6.9 := 
by
  sorry

end NUMINAMATH_GPT_travel_time_l1932_193277


namespace NUMINAMATH_GPT_equal_roots_m_eq_minus_half_l1932_193242

theorem equal_roots_m_eq_minus_half (x m : ℝ) 
  (h_eq: ∀ x, ( (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m )) :
  m = -1/2 := by 
  sorry

end NUMINAMATH_GPT_equal_roots_m_eq_minus_half_l1932_193242


namespace NUMINAMATH_GPT_proof_x_exists_l1932_193206

noncomputable def find_x : ℝ := 33.33

theorem proof_x_exists (A B C : ℝ) (h1 : A = (1 + find_x / 100) * B) (h2 : C = 0.75 * A) (h3 : A > C) (h4 : C > B) :
  find_x = 33.33 := 
by
  -- Proof steps
  sorry

end NUMINAMATH_GPT_proof_x_exists_l1932_193206


namespace NUMINAMATH_GPT_distance_equals_absolute_value_l1932_193296

def distance_from_origin (x : ℝ) : ℝ := abs x

theorem distance_equals_absolute_value (x : ℝ) : distance_from_origin x = abs x :=
by
  sorry

end NUMINAMATH_GPT_distance_equals_absolute_value_l1932_193296


namespace NUMINAMATH_GPT_perfect_squares_of_diophantine_l1932_193235

theorem perfect_squares_of_diophantine (a b : ℤ) (h : 2 * a^2 + a = 3 * b^2 + b) :
  ∃ k m : ℤ, (a - b) = k^2 ∧ (2 * a + 2 * b + 1) = m^2 := by
  sorry

end NUMINAMATH_GPT_perfect_squares_of_diophantine_l1932_193235


namespace NUMINAMATH_GPT_time_taken_by_A_l1932_193257

-- Definitions for the problem conditions
def race_distance : ℕ := 1000  -- in meters
def A_beats_B_by_distance : ℕ := 48  -- in meters
def A_beats_B_by_time : ℕ := 12  -- in seconds

-- The formal statement to prove in Lean
theorem time_taken_by_A :
  ∃ T_a : ℕ, (1000 * (T_a + 12) = 952 * T_a) ∧ T_a = 250 :=
by
  sorry

end NUMINAMATH_GPT_time_taken_by_A_l1932_193257


namespace NUMINAMATH_GPT_solve_fraction_equation_l1932_193203

theorem solve_fraction_equation (x : ℝ) (h : (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2) : x = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l1932_193203


namespace NUMINAMATH_GPT_sum_of_remainders_correct_l1932_193281

def sum_of_remainders : ℕ :=
  let remainders := [43210 % 37, 54321 % 37, 65432 % 37, 76543 % 37, 87654 % 37, 98765 % 37]
  remainders.sum

theorem sum_of_remainders_correct : sum_of_remainders = 36 :=
by sorry

end NUMINAMATH_GPT_sum_of_remainders_correct_l1932_193281


namespace NUMINAMATH_GPT_players_without_cautions_l1932_193201

theorem players_without_cautions (Y N : ℕ) (h1 : Y + N = 11) (h2 : Y = 6) : N = 5 :=
by
  sorry

end NUMINAMATH_GPT_players_without_cautions_l1932_193201


namespace NUMINAMATH_GPT_find_ab_l1932_193249

noncomputable def perpendicular_condition (a b : ℝ) :=
  a * (a - 1) - b = 0

noncomputable def point_on_l1_condition (a b : ℝ) :=
  -3 * a + b + 4 = 0

noncomputable def parallel_condition (a b : ℝ) :=
  a + b * (a - 1) = 0

noncomputable def distance_condition (a : ℝ) :=
  4 = abs ((-a) / (a - 1))

theorem find_ab (a b : ℝ) :
  (perpendicular_condition a b ∧ point_on_l1_condition a b ∧
   parallel_condition a b ∧ distance_condition a) →
  ((a = 2 ∧ b = 2) ∨ (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = 2)) :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l1932_193249


namespace NUMINAMATH_GPT_infinite_geometric_subsequence_exists_l1932_193202

theorem infinite_geometric_subsequence_exists
  (a : ℕ) (d : ℕ) (h_d_pos : d > 0)
  (a_n : ℕ → ℕ)
  (h_arith_prog : ∀ n, a_n n = a + n * d) :
  ∃ (g : ℕ → ℕ), (∀ m n, m < n → g m < g n) ∧ (∃ r : ℕ, ∀ n, g (n+1) = g n * r) ∧ (∀ n, ∃ m, a_n m = g n) :=
sorry

end NUMINAMATH_GPT_infinite_geometric_subsequence_exists_l1932_193202


namespace NUMINAMATH_GPT_memorable_numbers_count_l1932_193221

def is_memorable_number (d : Fin 10 → Fin 8 → ℕ) : Prop :=
  d 0 0 = d 1 0 ∧ d 0 1 = d 1 1 ∧ d 0 2 = d 1 2 ∧ d 0 3 = d 1 3

theorem memorable_numbers_count : 
  ∃ n : ℕ, n = 10000 ∧ ∀ (d : Fin 10 → Fin 8 → ℕ), is_memorable_number d → n = 10000 :=
sorry

end NUMINAMATH_GPT_memorable_numbers_count_l1932_193221


namespace NUMINAMATH_GPT_ants_in_park_l1932_193218

theorem ants_in_park:
  let width_meters := 100
  let length_meters := 130
  let cm_per_meter := 100
  let ants_per_sq_cm := 1.2
  let width_cm := width_meters * cm_per_meter
  let length_cm := length_meters * cm_per_meter
  let area_sq_cm := width_cm * length_cm
  let total_ants := ants_per_sq_cm * area_sq_cm
  total_ants = 156000000 := by
  sorry

end NUMINAMATH_GPT_ants_in_park_l1932_193218


namespace NUMINAMATH_GPT_instantaneous_velocity_at_2_l1932_193274

-- Define the motion equation
def s (t : ℝ) : ℝ := 3 + t^2

-- State the problem: Prove the instantaneous velocity at t = 2 is 4
theorem instantaneous_velocity_at_2 : (deriv s) 2 = 4 := by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_2_l1932_193274


namespace NUMINAMATH_GPT_volume_of_cylinder_cut_l1932_193279

open Real

noncomputable def cylinder_cut_volume (R α : ℝ) : ℝ :=
  (2 / 3) * R^3 * tan α

theorem volume_of_cylinder_cut (R α : ℝ) :
  cylinder_cut_volume R α = (2 / 3) * R^3 * tan α :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cylinder_cut_l1932_193279


namespace NUMINAMATH_GPT_parabola_translation_l1932_193286

theorem parabola_translation :
  ∀ f g : ℝ → ℝ,
    (∀ x, f x = - (x - 1) ^ 2) →
    (∀ x, g x = f (x - 1) + 2) →
    ∀ x, g x = - (x - 2) ^ 2 + 2 :=
by
  -- Add the proof steps here if needed
  sorry

end NUMINAMATH_GPT_parabola_translation_l1932_193286


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1932_193216

variable (x y : ℚ)
variable (expr : ℚ := 3 * x * y^2 - (x * y - 2 * (2 * x * y - 3 / 2 * x^2 * y) + 3 * x * y^2) + 3 * x^2 * y)

theorem simplify_and_evaluate (h1 : x = 3) (h2 : y = -1 / 3) : expr = -3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1932_193216


namespace NUMINAMATH_GPT_given_conditions_implies_correct_answer_l1932_193290

noncomputable def is_binomial_coefficient_equal (n : ℕ) : Prop := 
  Nat.choose n 2 = Nat.choose n 6

noncomputable def sum_of_odd_terms (n : ℕ) : ℕ :=
  2 ^ (n - 1)

theorem given_conditions_implies_correct_answer (n : ℕ) (h : is_binomial_coefficient_equal n) : 
  n = 8 ∧ sum_of_odd_terms n = 128 := by 
  sorry

end NUMINAMATH_GPT_given_conditions_implies_correct_answer_l1932_193290


namespace NUMINAMATH_GPT_smallest_whole_number_greater_than_sum_l1932_193258

theorem smallest_whole_number_greater_than_sum : 
  (3 + (1 / 3) + 4 + (1 / 4) + 6 + (1 / 6) + 7 + (1 / 7)) < 21 :=
sorry

end NUMINAMATH_GPT_smallest_whole_number_greater_than_sum_l1932_193258


namespace NUMINAMATH_GPT_cannot_form_isosceles_triangle_l1932_193248

theorem cannot_form_isosceles_triangle :
  ¬ ∃ (sticks : Finset ℕ) (a b c : ℕ), a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧
  a + b > c ∧ a + c > b ∧ b + c > a ∧ -- Triangle inequality
  (a = b ∨ b = c ∨ a = c) ∧ -- Isosceles condition
  sticks ⊆ {1, 2, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9} := sorry

end NUMINAMATH_GPT_cannot_form_isosceles_triangle_l1932_193248


namespace NUMINAMATH_GPT_ones_digit_of_prime_sequence_l1932_193247

theorem ones_digit_of_prime_sequence (p q r s : ℕ) (h1 : p > 5) 
    (h2 : p < q ∧ q < r ∧ r < s) (h3 : q - p = 8 ∧ r - q = 8 ∧ s - r = 8) 
    (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s) : 
    p % 10 = 3 :=
by
  sorry

end NUMINAMATH_GPT_ones_digit_of_prime_sequence_l1932_193247


namespace NUMINAMATH_GPT_f_monotonic_intervals_g_greater_than_4_3_l1932_193284

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x - Real.log x

theorem f_monotonic_intervals :
  (∀ x < -1, ∀ y < -1, x < y → f x > f y) ∧ 
  (∀ x > -1, ∀ y > -1, x < y → f x < f y) :=
sorry

theorem g_greater_than_4_3 (x : ℝ) (h : x > 0) : g x > (4 / 3) :=
sorry

end NUMINAMATH_GPT_f_monotonic_intervals_g_greater_than_4_3_l1932_193284


namespace NUMINAMATH_GPT_phone_numbers_divisible_by_13_l1932_193266

theorem phone_numbers_divisible_by_13 :
  ∃ (x y z : ℕ), (x < 10) ∧ (y < 10) ∧ (z < 10) ∧ (100 * x + 10 * y + z) % 13 = 0 ∧ (2 * y = x + z) :=
  sorry

end NUMINAMATH_GPT_phone_numbers_divisible_by_13_l1932_193266


namespace NUMINAMATH_GPT_dinner_time_correct_l1932_193226

-- Definitions based on the conditions in the problem
def pounds_per_turkey : Nat := 16
def roasting_time_per_pound : Nat := 15  -- minutes
def num_turkeys : Nat := 2
def minutes_per_hour : Nat := 60
def latest_start_time_hours : Nat := 10

-- The total roasting time in hours
def total_roasting_time_hours : Nat := 
  (roasting_time_per_pound * pounds_per_turkey * num_turkeys) / minutes_per_hour

-- The expected dinner time
def expected_dinner_time_hours : Nat := latest_start_time_hours + total_roasting_time_hours

-- The proof problem
theorem dinner_time_correct : expected_dinner_time_hours = 18 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_dinner_time_correct_l1932_193226


namespace NUMINAMATH_GPT_equal_cookies_per_person_l1932_193272

theorem equal_cookies_per_person 
  (boxes : ℕ) (cookies_per_box : ℕ) (people : ℕ)
  (h1 : boxes = 7) (h2 : cookies_per_box = 10) (h3 : people = 5) :
  (boxes * cookies_per_box) / people = 14 :=
by sorry

end NUMINAMATH_GPT_equal_cookies_per_person_l1932_193272


namespace NUMINAMATH_GPT_solve_for_x_l1932_193285

theorem solve_for_x (x : ℤ) (h : 24 - 6 = 3 + x) : x = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1932_193285


namespace NUMINAMATH_GPT_polynomial_root_range_l1932_193255

variable (a : ℝ)

theorem polynomial_root_range (h : ∀ x : ℂ, (2 * x^4 + a * x^3 + 9 * x^2 + a * x + 2 = 0) →
  ((x.re^2 + x.im^2 ≠ 1) ∧ x.im ≠ 0)) : (-2 * Real.sqrt 10 < a ∧ a < 2 * Real.sqrt 10) :=
sorry

end NUMINAMATH_GPT_polynomial_root_range_l1932_193255


namespace NUMINAMATH_GPT_intersection_empty_implies_range_l1932_193233

-- Define the sets A and B
def setA := {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 3}
def setB (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 1}

-- Prove that if A ∩ B = ∅, then 1 < a < 2
theorem intersection_empty_implies_range (a : ℝ) (h : setA ∩ setB a = ∅) : 1 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_empty_implies_range_l1932_193233


namespace NUMINAMATH_GPT_geometry_problem_l1932_193288

-- Definitions for points and segments based on given conditions
variables {O A B C D E F G : Type} [Inhabited O] [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited G]

-- Lengths of segments based on given conditions
variables (DE EG : ℝ)
variable (BG : ℝ)

-- Given lengths
def given_lengths : Prop :=
  DE = 5 ∧ EG = 3

-- Goal to prove
def goal : Prop :=
  BG = 12

-- The theorem combining conditions and the goal
theorem geometry_problem (h : given_lengths DE EG) : goal BG :=
  sorry

end NUMINAMATH_GPT_geometry_problem_l1932_193288


namespace NUMINAMATH_GPT_simplify_expression_l1932_193214

theorem simplify_expression : |(-4 : Int)^2 - (3 : Int)^2 + 2| = 9 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1932_193214


namespace NUMINAMATH_GPT_worker_y_defective_rate_l1932_193223

noncomputable def y_f : ℚ := 0.1666666666666668
noncomputable def d_x : ℚ := 0.005 -- converting percentage to decimal
noncomputable def d_total : ℚ := 0.0055 -- converting percentage to decimal

theorem worker_y_defective_rate :
  ∃ d_y : ℚ, d_y = 0.008 ∧ d_total = ((1 - y_f) * d_x + y_f * d_y) :=
by
  sorry

end NUMINAMATH_GPT_worker_y_defective_rate_l1932_193223


namespace NUMINAMATH_GPT_width_of_field_l1932_193245

theorem width_of_field (W L : ℝ) (h1 : L = (7 / 5) * W) (h2 : 2 * L + 2 * W = 360) : W = 75 :=
sorry

end NUMINAMATH_GPT_width_of_field_l1932_193245


namespace NUMINAMATH_GPT_probability_snow_once_first_week_l1932_193212

theorem probability_snow_once_first_week :
  let p_first_two_days := (3 / 4) * (3 / 4)
  let p_next_three_days := (1 / 2) * (1 / 2) * (1 / 2)
  let p_last_two_days := (2 / 3) * (2 / 3)
  let p_no_snow := p_first_two_days * p_next_three_days * p_last_two_days
  let p_at_least_once := 1 - p_no_snow
  p_at_least_once = 31 / 32 :=
by
  sorry

end NUMINAMATH_GPT_probability_snow_once_first_week_l1932_193212


namespace NUMINAMATH_GPT_range_of_a_l1932_193205

noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem range_of_a (a : ℝ) : (A a ∪ B a = Set.univ) → a ∈ Set.Iic 2 := by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1932_193205


namespace NUMINAMATH_GPT_sqrt_fraction_difference_l1932_193210

theorem sqrt_fraction_difference : 
  (Real.sqrt (16 / 9) - Real.sqrt (9 / 16)) = 7 / 12 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_difference_l1932_193210


namespace NUMINAMATH_GPT_fraction_equality_l1932_193208

theorem fraction_equality (p q x y : ℚ) (hpq : p / q = 4 / 5) (hx : x / y + (2 * q - p) / (2 * q + p) = 1) :
  x / y = 4 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_equality_l1932_193208


namespace NUMINAMATH_GPT_calc_f_7_2_l1932_193227

variable {f : ℝ → ℝ}

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_sqrt_on_interval : ∀ x, 0 < x ∧ x ≤ 1 → f x = Real.sqrt x

theorem calc_f_7_2 : f (7 / 2) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_calc_f_7_2_l1932_193227


namespace NUMINAMATH_GPT_ball_bounce_height_l1932_193239

theorem ball_bounce_height :
  ∃ (k : ℕ), 10 * (1 / 2) ^ k < 1 ∧ (∀ m < k, 10 * (1 / 2) ^ m ≥ 1) :=
sorry

end NUMINAMATH_GPT_ball_bounce_height_l1932_193239


namespace NUMINAMATH_GPT_robot_min_steps_l1932_193232

theorem robot_min_steps {a b : ℕ} (ha : 0 < a) (hb : 0 < b) : ∃ n, n = a + b - Nat.gcd a b :=
by
  sorry

end NUMINAMATH_GPT_robot_min_steps_l1932_193232


namespace NUMINAMATH_GPT_trig_identity_problem_l1932_193292

theorem trig_identity_problem 
  (t m n k : ℕ) 
  (h_rel_prime : Nat.gcd m n = 1) 
  (h_condition1 : (1 + Real.sin t) * (1 + Real.cos t) = 8 / 9) 
  (h_condition2 : (1 - Real.sin t) * (1 - Real.cos t) = m / n - Real.sqrt k) 
  (h_pos_int_m : 0 < m) 
  (h_pos_int_n : 0 < n) 
  (h_pos_int_k : 0 < k) :
  k + m + n = 15 := 
sorry

end NUMINAMATH_GPT_trig_identity_problem_l1932_193292


namespace NUMINAMATH_GPT_therapy_sessions_l1932_193237

theorem therapy_sessions (F A n : ℕ) 
  (h1 : F = A + 25)
  (h2 : F + A = 115)
  (h3 : F + (n - 1) * A = 250) : 
  n = 5 := 
by sorry

end NUMINAMATH_GPT_therapy_sessions_l1932_193237


namespace NUMINAMATH_GPT_problem_a_b_c_ge_neg2_l1932_193244

theorem problem_a_b_c_ge_neg2 {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1 / b > -2) ∨ (b + 1 / c > -2) ∨ (c + 1 / a > -2) → False :=
by
  sorry

end NUMINAMATH_GPT_problem_a_b_c_ge_neg2_l1932_193244


namespace NUMINAMATH_GPT_danielle_rooms_is_6_l1932_193294

def heidi_rooms (danielle_rooms : ℕ) : ℕ := 3 * danielle_rooms
def grant_rooms (heidi_rooms : ℕ) : ℕ := heidi_rooms / 9

theorem danielle_rooms_is_6 (danielle_rooms : ℕ) (h1 : heidi_rooms danielle_rooms = 18) (h2 : grant_rooms (heidi_rooms danielle_rooms) = 2) :
  danielle_rooms = 6 :=
by 
  sorry

end NUMINAMATH_GPT_danielle_rooms_is_6_l1932_193294


namespace NUMINAMATH_GPT_smallest_digit_never_in_units_place_of_odd_number_l1932_193229

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end NUMINAMATH_GPT_smallest_digit_never_in_units_place_of_odd_number_l1932_193229


namespace NUMINAMATH_GPT_problem1_l1932_193297

theorem problem1 (n : ℕ) (hn : 0 < n) : 20 ∣ (4 * 6^n + 5^(n+1) - 9) := 
  sorry

end NUMINAMATH_GPT_problem1_l1932_193297


namespace NUMINAMATH_GPT_plane_equation_l1932_193267

theorem plane_equation 
  (P Q : ℝ×ℝ×ℝ) (A B : ℝ×ℝ×ℝ)
  (hp : P = (-1, 2, 5))
  (hq : Q = (3, -4, 1))
  (ha : A = (0, -2, -1))
  (hb : B = (3, 2, -1)) :
  ∃ (a b c d : ℝ), (a = 3 ∧ b = 4 ∧ c = 0 ∧ d = 1) ∧ (∀ x y z : ℝ, a * (x - 1) + b * (y + 1) + c * (z - 3) = d) :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_l1932_193267


namespace NUMINAMATH_GPT_algebraic_expression_value_l1932_193224

theorem algebraic_expression_value (a : ℝ) (h : 2 * a^2 + 3 * a - 5 = 0) : 6 * a^2 + 9 * a - 5 = 10 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1932_193224


namespace NUMINAMATH_GPT_evaluate_expression_l1932_193298

variable (x y : ℝ)

theorem evaluate_expression
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hsum_sq : x^2 + y^2 ≠ 0)
  (hsum : x + y ≠ 0) :
    (x^2 + y^2)⁻¹ * ((x + y)⁻¹ + (x / y)⁻¹) = (1 + y) / ((x^2 + y^2) * (x + y)) :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l1932_193298


namespace NUMINAMATH_GPT_sin_C_eq_sin_A_minus_B_eq_l1932_193268

open Real

-- Problem 1
theorem sin_C_eq (A B C : ℝ) (a b c : ℝ)
  (hB : B = π / 3) 
  (h3a2b : 3 * a = 2 * b) 
  (hA_sum_B_C : A + B + C = π) 
  (h_sin_law_a : sin A / a = sin B / b) 
  (h_sin_law_b : sin B / b = sin C / c) :
  sin C = (sqrt 3 + 3 * sqrt 2) / 6 :=
sorry

-- Problem 2
theorem sin_A_minus_B_eq (A B C : ℝ) (a b c : ℝ)
  (h_cosC : cos C = 2 / 3) 
  (h3a2b : 3 * a = 2 * b) 
  (hA_sum_B_C : A + B + C = π) 
  (h_sin_law_a : sin A / a = sin B / b) 
  (h_sin_law_b : sin B / b = sin C / c) 
  (hA_acute : 0 < A ∧ A < π / 2)
  (hB_acute : 0 < B ∧ B < π / 2) :
  sin (A - B) = -sqrt 5 / 3 :=
sorry

end NUMINAMATH_GPT_sin_C_eq_sin_A_minus_B_eq_l1932_193268


namespace NUMINAMATH_GPT_net_percentage_change_l1932_193243

theorem net_percentage_change (k m : ℝ) : 
  let scale_factor_1 := 1 - k / 100
  let scale_factor_2 := 1 + m / 100
  let overall_scale_factor := scale_factor_1 * scale_factor_2
  let percentage_change := (overall_scale_factor - 1) * 100
  percentage_change = m - k - k * m / 100 := 
by 
  sorry

end NUMINAMATH_GPT_net_percentage_change_l1932_193243


namespace NUMINAMATH_GPT_probability_at_least_four_same_face_l1932_193207

-- Define the total number of outcomes for flipping five coins
def total_outcomes : ℕ := 2^5

-- Define the number of favorable outcomes where at least four coins show the same face
def favorable_outcomes : ℕ := 2 + 5 + 5

-- Define the probability of getting at least four heads or four tails out of five flips
def probability : ℚ := favorable_outcomes / total_outcomes

-- Theorem statement to prove the probability calculation
theorem probability_at_least_four_same_face : 
  probability = 3 / 8 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_probability_at_least_four_same_face_l1932_193207


namespace NUMINAMATH_GPT_number_of_sampled_medium_stores_is_five_l1932_193256

-- Definitions based on the conditions
def total_stores : ℕ := 300
def large_stores : ℕ := 30
def medium_stores : ℕ := 75
def small_stores : ℕ := 195
def sample_size : ℕ := 20

-- Proportion calculation function
def medium_store_proportion := (medium_stores : ℚ) / (total_stores : ℚ)

-- Sampled medium stores calculation
def sampled_medium_stores := medium_store_proportion * (sample_size : ℚ)

-- Theorem stating the number of medium stores drawn using stratified sampling
theorem number_of_sampled_medium_stores_is_five :
  sampled_medium_stores = 5 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_sampled_medium_stores_is_five_l1932_193256


namespace NUMINAMATH_GPT_system_of_equations_solution_l1932_193222

theorem system_of_equations_solution (x1 x2 x3 x4 x5 : ℝ) (h1 : x1 + x2 = x3^2) (h2 : x2 + x3 = x4^2)
  (h3 : x3 + x4 = x5^2) (h4 : x4 + x5 = x1^2) (h5 : x5 + x1 = x2^2) :
  x1 = 2 ∧ x2 = 2 ∧ x3 = 2 ∧ x4 = 2 ∧ x5 = 2 := 
sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1932_193222


namespace NUMINAMATH_GPT_average_speeds_l1932_193271

theorem average_speeds (x y : ℝ) (h1 : 4 * x + 5 * y = 98) (h2 : 4 * x = 5 * y - 2) : 
  x = 12 ∧ y = 10 :=
by sorry

end NUMINAMATH_GPT_average_speeds_l1932_193271


namespace NUMINAMATH_GPT_prove_correct_statement_l1932_193250

-- Define the conditions; we use the negation of incorrect statements
def condition1 (a b : ℝ) : Prop := a ≠ b → ¬((a - b > 0) → (a > 0 ∧ b > 0))
def condition2 (x : ℝ) : Prop := ¬(|x| > 0)
def condition4 (x : ℝ) : Prop := x ≠ 0 → (¬(∃ y, y = 1 / x))

-- Define the statement we want to prove as the correct one
def correct_statement (q : ℚ) : Prop := 0 - q = -q

-- The main theorem that combines conditions and proves the correct statement
theorem prove_correct_statement (a b : ℝ) (q : ℚ) :
  condition1 a b →
  condition2 a →
  condition4 a →
  correct_statement q :=
  by
  intros h1 h2 h4
  unfold correct_statement
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_prove_correct_statement_l1932_193250


namespace NUMINAMATH_GPT_cone_tangent_min_lateral_area_l1932_193246

/-- 
Given a cone with volume π / 6, prove that when the lateral area of the cone is minimized,
the tangent of the angle between the slant height and the base is sqrt(2).
-/
theorem cone_tangent_min_lateral_area :
  ∀ (r h l : ℝ), (π / 6 = (1 / 3) * π * r^2 * h) →
    (h = 1 / (2 * r^2)) →
    (l = Real.sqrt (r^2 + h^2)) →
    ((π * r * l) ≥ (3 / 4 * π)) →
    (r = Real.sqrt (2) / 2) →
    (h / r = Real.sqrt (2)) :=
by
  intro r h l V_cond h_cond l_def min_lateral_area r_val
  -- Proof steps go here (omitted as per the instruction)
  sorry

end NUMINAMATH_GPT_cone_tangent_min_lateral_area_l1932_193246


namespace NUMINAMATH_GPT_square_area_in_right_triangle_l1932_193260

theorem square_area_in_right_triangle (XY ZC : ℝ) (hXY : XY = 40) (hZC : ZC = 70) : 
  ∃ s : ℝ, s^2 = 2800 ∧ s = (40 * 70) / (XY + ZC) := 
by
  sorry

end NUMINAMATH_GPT_square_area_in_right_triangle_l1932_193260


namespace NUMINAMATH_GPT_abc_zero_iff_quadratic_identities_l1932_193276

variable {a b c : ℝ}

theorem abc_zero_iff_quadratic_identities (h : ¬(a = b ∧ b = c ∧ c = a)) : 
  a + b + c = 0 ↔ a^2 + ab + b^2 = b^2 + bc + c^2 ∧ b^2 + bc + c^2 = c^2 + ca + a^2 :=
by
  sorry

end NUMINAMATH_GPT_abc_zero_iff_quadratic_identities_l1932_193276


namespace NUMINAMATH_GPT_problem_2_l1932_193241

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x ^ 2 + a * Real.log (1 - x)

theorem problem_2 (a : ℝ) (x₁ x₂ : ℝ) (h₀ : 0 < a) (h₁ : a < 1/4) (h₂ : f x₂ a = 0) 
  (h₃ : f x₁ a = 0) (hx₁ : 0 < x₁) (hx₂ : x₁ < 1/2) (h₄ : x₁ < x₂) :
  f x₂ a - x₁ > - (3 + Real.log 4) / 8 := sorry

end NUMINAMATH_GPT_problem_2_l1932_193241


namespace NUMINAMATH_GPT_income_ratio_l1932_193236

theorem income_ratio (I1 I2 E1 E2 : ℝ) (h1 : I1 = 5500) (h2 : E1 = I1 - 2200) (h3 : E2 = I2 - 2200) (h4 : E1 / E2 = 3 / 2) : I1 / I2 = 5 / 4 := by
  -- This is where the proof would go, but it's omitted for brevity.
  sorry

end NUMINAMATH_GPT_income_ratio_l1932_193236


namespace NUMINAMATH_GPT_work_time_A_and_C_together_l1932_193228

theorem work_time_A_and_C_together
  (A_work B_work C_work : ℝ)
  (hA : A_work = 1/3)
  (hB : B_work = 1/6)
  (hBC : B_work + C_work = 1/3) :
  1 / (A_work + C_work) = 2 := by
  sorry

end NUMINAMATH_GPT_work_time_A_and_C_together_l1932_193228


namespace NUMINAMATH_GPT_total_tiles_l1932_193204

-- Define the dimensions
def length : ℕ := 16
def width : ℕ := 12

-- Define the number of 1-foot by 1-foot tiles for the border
def tiles_border : ℕ := (2 * length + 2 * width - 4)

-- Define the inner dimensions
def inner_length : ℕ := length - 2
def inner_width : ℕ := width - 2

-- Define the number of 2-foot by 2-foot tiles for the interior
def tiles_interior : ℕ := (inner_length * inner_width) / 4

-- Prove that the total number of tiles is 87
theorem total_tiles : tiles_border + tiles_interior = 87 := by
  sorry

end NUMINAMATH_GPT_total_tiles_l1932_193204


namespace NUMINAMATH_GPT_second_horse_revolutions_l1932_193283

-- Define the parameters and conditions:
def r₁ : ℝ := 30  -- Distance of the first horse from the center
def revolutions₁ : ℕ := 15  -- Number of revolutions by the first horse
def r₂ : ℝ := 5  -- Distance of the second horse from the center

-- Define the statement to prove:
theorem second_horse_revolutions : r₂ * (↑revolutions₁ * r₁⁻¹) * (↑revolutions₁) = 90 := 
by sorry

end NUMINAMATH_GPT_second_horse_revolutions_l1932_193283


namespace NUMINAMATH_GPT_root_value_algebraic_expression_l1932_193219

theorem root_value_algebraic_expression {a : ℝ} (h : a^2 + 3 * a + 2 = 0) : a^2 + 3 * a = -2 :=
by
  sorry

end NUMINAMATH_GPT_root_value_algebraic_expression_l1932_193219


namespace NUMINAMATH_GPT_tangent_line_at_origin_l1932_193282

noncomputable def curve (x : ℝ) : ℝ := x * Real.exp x + 2 * x - 1

def tangent_line (x₀ y₀ : ℝ) (k : ℝ) (x : ℝ) := y₀ + k * (x - x₀)

theorem tangent_line_at_origin : 
  tangent_line 0 (-1) 3 = λ x => 3 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_origin_l1932_193282


namespace NUMINAMATH_GPT_size_of_coffee_cup_l1932_193262

-- Define the conditions and the final proof statement
variable (C : ℝ) (h1 : (1/4) * C) (h2 : (1/2) * C) (remaining_after_cold : (1/4) * C - 1 = 2)

theorem size_of_coffee_cup : C = 6 := by
  -- Here the proof would go, but we omit it with sorry
  sorry

end NUMINAMATH_GPT_size_of_coffee_cup_l1932_193262


namespace NUMINAMATH_GPT_other_endpoint_product_l1932_193259

theorem other_endpoint_product :
  ∀ (x y : ℤ), 
    (3 = (x + 7) / 2) → 
    (-5 = (y - 1) / 2) → 
    x * y = 9 :=
by
  intro x y h1 h2
  sorry

end NUMINAMATH_GPT_other_endpoint_product_l1932_193259


namespace NUMINAMATH_GPT_find_value_b_in_geometric_sequence_l1932_193217

theorem find_value_b_in_geometric_sequence
  (b : ℝ)
  (h1 : 15 ≠ 0) -- to ensure division by zero does not occur
  (h2 : b ≠ 0)  -- to ensure division by zero does not occur
  (h3 : 15 * (b / 15) = b) -- 15 * r = b
  (h4 : b * (b / 15) = 45 / 4) -- b * r = 45 / 4
  : b = 15 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_value_b_in_geometric_sequence_l1932_193217


namespace NUMINAMATH_GPT_mouse_cannot_eat_entire_cheese_l1932_193238

-- Defining the conditions of the problem
structure Cheese :=
  (size : ℕ := 3)  -- The cube size is 3x3x3
  (central_cube_removed : Bool := true)  -- The central cube is removed

inductive CubeColor
| black
| white

structure Mouse :=
  (can_eat : CubeColor -> CubeColor -> Bool)
  (adjacency : Nat -> Nat -> Bool)

def cheese_problem (c : Cheese) (m : Mouse) : Bool := sorry

-- The main theorem: It is impossible for the mouse to eat the entire piece of cheese.
theorem mouse_cannot_eat_entire_cheese : ∀ (c : Cheese) (m : Mouse),
  cheese_problem c m = false := sorry

end NUMINAMATH_GPT_mouse_cannot_eat_entire_cheese_l1932_193238


namespace NUMINAMATH_GPT_students_not_examined_l1932_193270

theorem students_not_examined (boys girls examined : ℕ) (h1 : boys = 121) (h2 : girls = 83) (h3 : examined = 150) : 
  (boys + girls - examined = 54) := by
  sorry

end NUMINAMATH_GPT_students_not_examined_l1932_193270


namespace NUMINAMATH_GPT_available_milk_for_me_l1932_193211

def initial_milk_litres : ℝ := 1
def myeongseok_milk_litres : ℝ := 0.1
def mingu_milk_litres : ℝ := myeongseok_milk_litres + 0.2
def minjae_milk_litres : ℝ := 0.3

theorem available_milk_for_me :
  initial_milk_litres - (myeongseok_milk_litres + mingu_milk_litres + minjae_milk_litres) = 0.3 :=
by sorry

end NUMINAMATH_GPT_available_milk_for_me_l1932_193211


namespace NUMINAMATH_GPT_forty_percent_of_number_l1932_193234

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 15) :
  0.40 * N = 180 :=
by
  sorry

end NUMINAMATH_GPT_forty_percent_of_number_l1932_193234


namespace NUMINAMATH_GPT_probability_four_vertices_same_plane_proof_l1932_193269

noncomputable def probability_four_vertices_same_plane : ℚ := 
  let total_ways := Nat.choose 8 4
  let favorable_ways := 12
  favorable_ways / total_ways

theorem probability_four_vertices_same_plane_proof : 
  probability_four_vertices_same_plane = 6 / 35 :=
by
  -- include necessary definitions and calculations for the actual proof
  sorry

end NUMINAMATH_GPT_probability_four_vertices_same_plane_proof_l1932_193269


namespace NUMINAMATH_GPT_ben_chairs_in_10_days_l1932_193220

def number_of_chairs (days hours_per_shift hours_rocking_chair hours_dining_chair hours_armchair : ℕ) : ℕ × ℕ × ℕ :=
  let rocking_chairs_per_day := hours_per_shift / hours_rocking_chair
  let remaining_hours_after_rocking_chairs := hours_per_shift % hours_rocking_chair
  let dining_chairs_per_day := remaining_hours_after_rocking_chairs / hours_dining_chair
  let remaining_hours_after_dining_chairs := remaining_hours_after_rocking_chairs % hours_dining_chair
  if remaining_hours_after_dining_chairs >= hours_armchair then
    (days * rocking_chairs_per_day, days * dining_chairs_per_day, days * (remaining_hours_after_dining_chairs / hours_armchair))
  else
    (days * rocking_chairs_per_day, days * dining_chairs_per_day, 0)

theorem ben_chairs_in_10_days :
  number_of_chairs 10 8 5 3 6 = (10, 10, 0) :=
by 
  sorry

end NUMINAMATH_GPT_ben_chairs_in_10_days_l1932_193220


namespace NUMINAMATH_GPT_set_equality_l1932_193295

theorem set_equality (a : ℤ) : 
  {z : ℤ | ∃ x : ℤ, (x - a = z ∧ a - 1 ≤ x ∧ x ≤ a + 1)} = {-1, 0, 1} :=
by {
  sorry
}

end NUMINAMATH_GPT_set_equality_l1932_193295


namespace NUMINAMATH_GPT_order_of_xyz_l1932_193293

variable (a b c d : ℝ)

noncomputable def x : ℝ := Real.sqrt (a * b) + Real.sqrt (c * d)
noncomputable def y : ℝ := Real.sqrt (a * c) + Real.sqrt (b * d)
noncomputable def z : ℝ := Real.sqrt (a * d) + Real.sqrt (b * c)

theorem order_of_xyz (h₁ : a > b) (h₂ : b > c) (h₃ : c > d) (h₄ : d > 0) : x a b c d > y a b c d ∧ y a b c d > z a b c d :=
by
  sorry

end NUMINAMATH_GPT_order_of_xyz_l1932_193293


namespace NUMINAMATH_GPT_percentage_of_acid_is_18_18_percent_l1932_193275

noncomputable def percentage_of_acid_in_original_mixture
  (a w : ℝ) (h1 : (a + 1) / (a + w + 1) = 1 / 4) (h2 : (a + 1) / (a + w + 2) = 1 / 5) : ℝ :=
  a / (a + w) 

theorem percentage_of_acid_is_18_18_percent :
  ∃ (a w : ℝ), (a + 1) / (a + w + 1) = 1 / 4 ∧ (a + 1) / (a + w + 2) = 1 / 5 ∧ percentage_of_acid_in_original_mixture a w (by sorry) (by sorry) = 18.18 := by
  sorry

end NUMINAMATH_GPT_percentage_of_acid_is_18_18_percent_l1932_193275


namespace NUMINAMATH_GPT_average_temperature_week_l1932_193240

theorem average_temperature_week :
  let sunday := 99.1
  let monday := 98.2
  let tuesday := 98.7
  let wednesday := 99.3
  let thursday := 99.8
  let friday := 99.0
  let saturday := 98.9
  (sunday + monday + tuesday + wednesday + thursday + friday + saturday) / 7 = 99.0 :=
by
  sorry

end NUMINAMATH_GPT_average_temperature_week_l1932_193240


namespace NUMINAMATH_GPT_george_speed_to_school_l1932_193252

theorem george_speed_to_school :
  ∀ (D S_1 S_2 D_1 S_x : ℝ),
  D = 1.5 ∧ S_1 = 3 ∧ S_2 = 2 ∧ D_1 = 0.75 →
  S_x = (D - D_1) / ((D / S_1) - (D_1 / S_2)) →
  S_x = 6 :=
by
  intros D S_1 S_2 D_1 S_x h1 h2
  rw [h1.1, h1.2.1, h1.2.2.1, h1.2.2.2] at *
  sorry

end NUMINAMATH_GPT_george_speed_to_school_l1932_193252


namespace NUMINAMATH_GPT_pure_imaginary_solution_l1932_193265

theorem pure_imaginary_solution (a : ℝ) (i : ℂ) (h : i*i = -1) : (∀ z : ℂ, z = 1 + a * i → (z ^ 2).re = 0) → (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_GPT_pure_imaginary_solution_l1932_193265


namespace NUMINAMATH_GPT_cos_A_minus_B_l1932_193287

variable {A B : ℝ}

-- Conditions
def cos_conditions (A B : ℝ) : Prop :=
  (Real.cos A + Real.cos B = 1 / 2)

def sin_conditions (A B : ℝ) : Prop :=
  (Real.sin A + Real.sin B = 3 / 2)

-- Mathematically equivalent proof problem
theorem cos_A_minus_B (h1 : cos_conditions A B) (h2 : sin_conditions A B) :
  Real.cos (A - B) = 1 / 4 := 
sorry

end NUMINAMATH_GPT_cos_A_minus_B_l1932_193287


namespace NUMINAMATH_GPT_digits_in_2_pow_120_l1932_193273

theorem digits_in_2_pow_120 {a b : ℕ} (h : 10^a ≤ 2^200 ∧ 2^200 < 10^b) (ha : a = 60) (hb : b = 61) : 
  ∃ n : ℕ, 10^(n-1) ≤ 2^120 ∧ 2^120 < 10^n ∧ n = 37 :=
by {
  sorry
}

end NUMINAMATH_GPT_digits_in_2_pow_120_l1932_193273


namespace NUMINAMATH_GPT_quadratic_ineq_solution_range_of_b_for_any_a_l1932_193230

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a b x : α) : α := -3 * x^2 + a * (5 - a) * x + b

theorem quadratic_ineq_solution (a b : α) : 
  (∀ x ∈ Set.Ioo (-1 : α) 3, f a b x > 0) →
  ((a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9)) := 
  sorry

theorem range_of_b_for_any_a (a b : α) :
  (∀ a : α, f a b 2 < 0) → 
  b < -1 / 2 := 
  sorry

end NUMINAMATH_GPT_quadratic_ineq_solution_range_of_b_for_any_a_l1932_193230


namespace NUMINAMATH_GPT_eq_a_sub_b_l1932_193213

theorem eq_a_sub_b (a b : ℝ) (i : ℂ) (hi : i * i = -1) (h1 : (a + 4 * i) * i = b + i) : a - b = 5 :=
by
  have := hi
  have := h1
  sorry

end NUMINAMATH_GPT_eq_a_sub_b_l1932_193213


namespace NUMINAMATH_GPT_ducks_arrival_quantity_l1932_193231

variable {initial_ducks : ℕ} (arrival_ducks : ℕ)

def initial_geese (initial_ducks : ℕ) := 2 * initial_ducks - 10

def remaining_geese (initial_ducks : ℕ) := initial_geese initial_ducks - 10

def remaining_ducks (initial_ducks arrival_ducks : ℕ) := initial_ducks + arrival_ducks

theorem ducks_arrival_quantity :
  initial_ducks = 25 →
  remaining_geese initial_ducks = 30 →
  remaining_geese initial_ducks = remaining_ducks initial_ducks arrival_ducks + 1 →
  arrival_ducks = 4 :=
by
sorry

end NUMINAMATH_GPT_ducks_arrival_quantity_l1932_193231


namespace NUMINAMATH_GPT_initial_volume_of_mixture_l1932_193263

theorem initial_volume_of_mixture
  (x : ℕ)
  (h1 : 3 * x / (2 * x + 1) = 4 / 3)
  (h2 : x = 4) :
  5 * x = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_volume_of_mixture_l1932_193263
