import Mathlib

namespace NUMINAMATH_GPT_evaluate_expression_zero_l1962_196255

-- Main proof statement
theorem evaluate_expression_zero :
  ∀ (a d c b : ℤ),
    d = c + 5 →
    c = b - 8 →
    b = a + 3 →
    a = 3 →
    a - 1 ≠ 0 →
    d - 6 ≠ 0 →
    c + 4 ≠ 0 →
    (a + 3) * (d - 3) * (c + 9) = 0 :=
by
  intros a d c b hd hc hb ha h1 h2 h3
  sorry -- The proof goes here

end NUMINAMATH_GPT_evaluate_expression_zero_l1962_196255


namespace NUMINAMATH_GPT_sin_sum_triangle_inequality_l1962_196206

theorem sin_sum_triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_sin_sum_triangle_inequality_l1962_196206


namespace NUMINAMATH_GPT_modulo_calculation_l1962_196282

theorem modulo_calculation (n : ℕ) (hn : 0 ≤ n ∧ n < 19) (hmod : 5 * n % 19 = 1) : 
  ((3^n)^2 - 3) % 19 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_modulo_calculation_l1962_196282


namespace NUMINAMATH_GPT_min_minutes_to_make_B_cheaper_l1962_196293

def costA (x : ℕ) : ℕ :=
  if x ≤ 300 then 8 * x else 2400 + 7 * (x - 300)

def costB (x : ℕ) : ℕ := 2500 + 4 * x

theorem min_minutes_to_make_B_cheaper : ∃ (x : ℕ), x ≥ 301 ∧ costB x < costA x :=
by
  use 301
  sorry

end NUMINAMATH_GPT_min_minutes_to_make_B_cheaper_l1962_196293


namespace NUMINAMATH_GPT_solution_l1962_196205

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)

axiom periodic_f : ∀ x : ℝ, f (x - 3) = - f x

axiom increasing_f_on_interval : ∀ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ 3 ∧ 0 ≤ x2 ∧ x2 ≤ 3 ∧ x1 ≠ x2) → (f x1 - f x2) / (x1 - x2) > 0

theorem solution : f 49 < f 64 ∧ f 64 < f 81 :=
by
  sorry

end NUMINAMATH_GPT_solution_l1962_196205


namespace NUMINAMATH_GPT_initial_number_l1962_196251

theorem initial_number (N : ℤ) 
  (h : (N + 3) % 24 = 0) : N = 21 := 
sorry

end NUMINAMATH_GPT_initial_number_l1962_196251


namespace NUMINAMATH_GPT_numbers_unchanged_by_powers_of_n_l1962_196288

-- Definitions and conditions
def unchanged_when_raised (x : ℂ) (n : ℕ) : Prop :=
  x^n = x

def modulus_one (z : ℂ) : Prop :=
  Complex.abs z = 1

-- Proof statements
theorem numbers_unchanged_by_powers_of_n :
  (∀ x : ℂ, (∀ n : ℕ, n > 0 → unchanged_when_raised x n → x = 0 ∨ x = 1)) ∧
  (∀ z : ℂ, modulus_one z → (∀ n : ℕ, n > 0 → Complex.abs (z^n) = 1)) :=
by
  sorry

end NUMINAMATH_GPT_numbers_unchanged_by_powers_of_n_l1962_196288


namespace NUMINAMATH_GPT_find_a_prove_f_pos_l1962_196213

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.log x + (1 / 2) * x

theorem find_a (a x0 : ℝ) (hx0 : x0 > 0) (h_tangent : (x0 - a) * Real.log x0 + (1 / 2) * x0 = (1 / 2) * x0 ∧ Real.log x0 - a / x0 + 3 / 2 = 1 / 2) :
  a = 1 :=
sorry

theorem prove_f_pos (a : ℝ) (h_range : 1 / (2 * Real.exp 1) < a ∧ a < 2 * Real.sqrt (Real.exp 1)) (x : ℝ) (hx : x > 0) :
  f x a > 0 :=
sorry

end NUMINAMATH_GPT_find_a_prove_f_pos_l1962_196213


namespace NUMINAMATH_GPT_solve_fraction_x_l1962_196250

theorem solve_fraction_x (a b c d : ℤ) (hb : b ≠ 0) (hdc : d + c ≠ 0) 
: (2 * a + (bc - 2 * a * d) / (d + c)) / (b - (bc - 2 * a * d) / (d + c)) = c / d := 
sorry

end NUMINAMATH_GPT_solve_fraction_x_l1962_196250


namespace NUMINAMATH_GPT_sum_first_7_l1962_196268

variable {α : Type*} [LinearOrderedField α]

-- Definitions for the arithmetic sequence
noncomputable def arithmetic_sequence (a d : α) (n : ℕ) : α :=
  a + d * (n - 1)

noncomputable def sum_of_first_n_terms (a d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

-- Conditions
variable {a d : α} -- Initial term and common difference of the arithmetic sequence
variable (h : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 + arithmetic_sequence a d 6 = 12)

-- Proof statement
theorem sum_first_7 (a d : α) (h : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 + arithmetic_sequence a d 6 = 12) : 
  sum_of_first_n_terms a d 7 = 28 := 
by 
  sorry

end NUMINAMATH_GPT_sum_first_7_l1962_196268


namespace NUMINAMATH_GPT_hidden_dots_are_32_l1962_196202

theorem hidden_dots_are_32 
  (visible_faces : List ℕ)
  (h_visible : visible_faces = [1, 2, 3, 4, 4, 5, 6, 6])
  (num_dice : ℕ)
  (h_num_dice : num_dice = 3)
  (faces_per_die : List ℕ)
  (h_faces_per_die : faces_per_die = [1, 2, 3, 4, 5, 6]) :
  63 - visible_faces.sum = 32 := by
  sorry

end NUMINAMATH_GPT_hidden_dots_are_32_l1962_196202


namespace NUMINAMATH_GPT_calculate_height_l1962_196214

def base_length : ℝ := 2 -- in cm
def base_width : ℝ := 5 -- in cm
def volume : ℝ := 30 -- in cm^3

theorem calculate_height: base_length * base_width * 3 = volume :=
by
  -- base_length * base_width = 10
  -- 10 * 3 = 30
  sorry

end NUMINAMATH_GPT_calculate_height_l1962_196214


namespace NUMINAMATH_GPT_stratified_sampling_correct_l1962_196256

-- Defining the conditions
def total_students : ℕ := 900
def freshmen : ℕ := 300
def sophomores : ℕ := 200
def juniors : ℕ := 400
def sample_size : ℕ := 45

-- Defining the target sample numbers
def freshmen_sample : ℕ := 15
def sophomores_sample : ℕ := 10
def juniors_sample : ℕ := 20

-- The proof problem statement
theorem stratified_sampling_correct :
  freshmen_sample = (freshmen * sample_size / total_students) ∧
  sophomores_sample = (sophomores * sample_size / total_students) ∧
  juniors_sample = (juniors * sample_size / total_students) :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_correct_l1962_196256


namespace NUMINAMATH_GPT_trigonometric_identity_l1962_196262

noncomputable def alpha := -35 / 6 * Real.pi

theorem trigonometric_identity :
  (2 * Real.sin (Real.pi + alpha) * Real.cos (Real.pi - alpha)
    - Real.sin (3 * Real.pi / 2 + alpha)) /
  (1 + Real.sin (alpha) ^ 2 - Real.cos (Real.pi / 2 + alpha)
    - Real.cos (Real.pi + alpha) ^ 2) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1962_196262


namespace NUMINAMATH_GPT_distance_Reims_to_Chaumont_l1962_196297

noncomputable def distance_Chalons_Vitry : ℝ := 30
noncomputable def distance_Vitry_Chaumont : ℝ := 80
noncomputable def distance_Chaumont_SaintQuentin : ℝ := 236
noncomputable def distance_SaintQuentin_Reims : ℝ := 86
noncomputable def distance_Reims_Chalons : ℝ := 40

theorem distance_Reims_to_Chaumont :
  distance_Reims_Chalons + 
  distance_Chalons_Vitry + 
  distance_Vitry_Chaumont = 150 :=
sorry

end NUMINAMATH_GPT_distance_Reims_to_Chaumont_l1962_196297


namespace NUMINAMATH_GPT_remainder_of_f_div_x_minus_2_is_48_l1962_196230

-- Define the polynomial f(x)
noncomputable def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 8 * x^3 + 25 * x^2 - 14 * x - 40

-- State the theorem to prove that the remainder of f(x) when divided by x - 2 is 48
theorem remainder_of_f_div_x_minus_2_is_48 : f 2 = 48 :=
by sorry

end NUMINAMATH_GPT_remainder_of_f_div_x_minus_2_is_48_l1962_196230


namespace NUMINAMATH_GPT_remaining_pieces_l1962_196280

/-- Define the initial number of pieces on a standard chessboard. -/
def initial_pieces : Nat := 32

/-- Define the number of pieces lost by Audrey. -/
def audrey_lost : Nat := 6

/-- Define the number of pieces lost by Thomas. -/
def thomas_lost : Nat := 5

/-- Proof that the remaining number of pieces on the chessboard is 21. -/
theorem remaining_pieces : initial_pieces - (audrey_lost + thomas_lost) = 21 := by
  -- Mathematical equivalence to 32 - (6 + 5) = 21
  sorry

end NUMINAMATH_GPT_remaining_pieces_l1962_196280


namespace NUMINAMATH_GPT_temperature_on_friday_l1962_196228

variables {M T W Th F : ℝ}

theorem temperature_on_friday
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 41) :
  F = 33 :=
  sorry

end NUMINAMATH_GPT_temperature_on_friday_l1962_196228


namespace NUMINAMATH_GPT_average_age_of_team_l1962_196212

def total_age (A : ℕ) (N : ℕ) := A * N
def wicket_keeper_age (A : ℕ) := A + 3
def remaining_players_age (A : ℕ) (N : ℕ) (W : ℕ) := (total_age A N) - (A + W)

theorem average_age_of_team
  (A : ℕ)
  (N : ℕ)
  (H1 : N = 11)
  (H2 : A = 28)
  (W : ℕ)
  (H3 : W = wicket_keeper_age A)
  (H4 : (wicket_keeper_age A) = A + 3)
  : (remaining_players_age A N W) / (N - 2) = A - 1 :=
by
  rw [H1, H2, H3, H4]; sorry

end NUMINAMATH_GPT_average_age_of_team_l1962_196212


namespace NUMINAMATH_GPT_smallest_range_mean_2017_l1962_196248

theorem smallest_range_mean_2017 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (a + b + c + d) / 4 = 2017 ∧ (max (max a b) (max c d) - min (min a b) (min c d)) = 4 := 
sorry

end NUMINAMATH_GPT_smallest_range_mean_2017_l1962_196248


namespace NUMINAMATH_GPT_tangent_parallel_x_axis_coordinates_l1962_196220

theorem tangent_parallel_x_axis_coordinates :
  ∃ (x y : ℝ), (y = x^2 - 3 * x) ∧ (2 * x - 3 = 0) ∧ (x = 3 / 2) ∧ (y = -9 / 4) :=
by
  use (3 / 2)
  use (-9 / 4)
  sorry

end NUMINAMATH_GPT_tangent_parallel_x_axis_coordinates_l1962_196220


namespace NUMINAMATH_GPT_intersection_M_N_l1962_196211

def M : Set ℝ := { x | |x - 2| ≤ 1 }
def N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem intersection_M_N : M ∩ N = {3} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1962_196211


namespace NUMINAMATH_GPT_total_rainfall_l1962_196235

theorem total_rainfall :
  let monday := 0.12962962962962962
  let tuesday := 0.35185185185185186
  let wednesday := 0.09259259259259259
  let thursday := 0.25925925925925924
  let friday := 0.48148148148148145
  let saturday := 0.2222222222222222
  let sunday := 0.4444444444444444
  (monday + tuesday + wednesday + thursday + friday + saturday + sunday) = 1.9814814814814815 :=
by
  -- proof to be filled here
  sorry

end NUMINAMATH_GPT_total_rainfall_l1962_196235


namespace NUMINAMATH_GPT_additional_track_length_l1962_196253

theorem additional_track_length (h : ℝ) (g1 g2 : ℝ) (L1 L2 : ℝ)
  (rise_eq : h = 800) 
  (orig_grade : g1 = 0.04) 
  (new_grade : g2 = 0.025) 
  (L1_eq : L1 = h / g1) 
  (L2_eq : L2 = h / g2)
  : (L2 - L1 = 12000) := 
sorry

end NUMINAMATH_GPT_additional_track_length_l1962_196253


namespace NUMINAMATH_GPT_dot_product_is_4_l1962_196229

-- Define vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2, x)
def b : ℝ × ℝ := (1, -1)

-- Define the condition that a is parallel to (a + b)
def is_parallel (u v : ℝ × ℝ) : Prop := 
  (u.1 * v.2 - u.2 * v.1) = 0

theorem dot_product_is_4 (x : ℝ) (h_parallel : is_parallel (a x) (a x + b)) : 
  (a x).1 * b.1 + (a x).2 * b.2 = 4 :=
sorry

end NUMINAMATH_GPT_dot_product_is_4_l1962_196229


namespace NUMINAMATH_GPT_smallest_y_l1962_196281

theorem smallest_y (y : ℕ) : 
    (y % 5 = 4) ∧ 
    (y % 7 = 6) ∧ 
    (y % 8 = 7) → 
    y = 279 :=
sorry

end NUMINAMATH_GPT_smallest_y_l1962_196281


namespace NUMINAMATH_GPT_sumNats_l1962_196233

-- Define the set of natural numbers between 29 and 31 inclusive
def NatRange : List ℕ := [29, 30, 31]

-- Define the condition that checks the elements in the range
def isValidNumbers (n : ℕ) : Prop := n ≤ 31 ∧ n > 28

-- Check if all numbers in NatRange are valid
def allValidNumbers : Prop := ∀ n, n ∈ NatRange → isValidNumbers n

-- Define the sum function for the list
def sumList (lst : List ℕ) : ℕ := lst.foldr (.+.) 0

-- The main theorem
theorem sumNats : (allValidNumbers → (sumList NatRange) = 90) :=
by
  sorry

end NUMINAMATH_GPT_sumNats_l1962_196233


namespace NUMINAMATH_GPT_smallest_integer_with_eight_factors_l1962_196218

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end NUMINAMATH_GPT_smallest_integer_with_eight_factors_l1962_196218


namespace NUMINAMATH_GPT_complement_P_correct_l1962_196249

def is_solution (x : ℝ) : Prop := |x + 3| + |x + 6| = 3

def P : Set ℝ := {x | is_solution x}

def C_R (P : Set ℝ) : Set ℝ := {x | x ∉ P}

theorem complement_P_correct : C_R P = {x | x < -6 ∨ x > -3} :=
by
  sorry

end NUMINAMATH_GPT_complement_P_correct_l1962_196249


namespace NUMINAMATH_GPT_find_triples_l1962_196265

theorem find_triples : 
  { (a, b, k) : ℕ × ℕ × ℕ | 2^a * 3^b = k * (k + 1) } = 
  { (1, 0, 1), (1, 1, 2), (3, 2, 8), (2, 1, 3) } := 
by
  sorry

end NUMINAMATH_GPT_find_triples_l1962_196265


namespace NUMINAMATH_GPT_find_m_l1962_196274

-- Let's define the sets A and B.
def A : Set ℝ := {-1, 1, 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

-- We'll state the problem as a theorem
theorem find_m (m : ℝ) (h : B m ⊆ A) : m = 1 ∨ m = -1 :=
by sorry

end NUMINAMATH_GPT_find_m_l1962_196274


namespace NUMINAMATH_GPT_rational_sign_product_l1962_196226

theorem rational_sign_product (a b c : ℚ) (h : |a| / a + |b| / b + |c| / c = 1) : abc / |abc| = -1 := 
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_rational_sign_product_l1962_196226


namespace NUMINAMATH_GPT_correct_statements_l1962_196238

def problem_statements :=
  [ "The negation of the statement 'There exists an x ∈ ℝ such that x^2 - 3x + 3 = 0' is true.",
    "The statement '-1/2 < x < 0' is a necessary but not sufficient condition for '2x^2 - 5x - 3 < 0'.",
    "The negation of the statement 'If xy = 0, then at least one of x or y is equal to 0' is true.",
    "The curves x^2/25 + y^2/9 = 1 and x^2/(25 − k) + y^2/(9 − k) = 1 (9 < k < 25) share the same foci.",
    "There exists a unique line that passes through the point (1,3) and is tangent to the parabola y^2 = 4x."
  ]

theorem correct_statements :
  (∀ x : ℝ, ¬(x^2 - 3 * x + 3 = 0)) ∧ 
  ¬ (¬-1/2 < x ∧ x < 0 → 2 * x^2 - 5*x - 3 < 0) ∧ 
  (∀ x y : ℝ, xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧ 
  (∀ k : ℝ, 9 < k ∧ k < 25 → ∀ x y : ℝ, (x^2 / (25 - k) + y^2 / (9 - k) = 1) → (x^2 / 25 + y^2 / 9 = 1) → (x ≠ 0 ∨ y ≠ 0)) ∧ 
  ¬ (∃ l : ℝ, ∀ pt : ℝ × ℝ, pt = (1, 3) → ∀ y : ℝ, y^2 = 4 * pt.1 → y = 2 * pt.2)
:= 
  sorry

end NUMINAMATH_GPT_correct_statements_l1962_196238


namespace NUMINAMATH_GPT_passengers_on_third_plane_l1962_196231

theorem passengers_on_third_plane (
  P : ℕ
) (h1 : 600 - 2 * 50 = 500) -- Speed of the first plane
  (h2 : 600 - 2 * 60 = 480) -- Speed of the second plane
  (h_avg : (500 + 480 + (600 - 2 * P)) / 3 = 500) -- Average speed condition
  : P = 40 := by sorry

end NUMINAMATH_GPT_passengers_on_third_plane_l1962_196231


namespace NUMINAMATH_GPT_center_of_hyperbola_l1962_196237

theorem center_of_hyperbola :
  (∃ h k : ℝ, ∀ x y : ℝ, (3*y + 3)^2 / 49 - (2*x - 5)^2 / 9 = 1 ↔ x = h ∧ y = k) → 
  h = 5 / 2 ∧ k = -1 :=
by
  sorry

end NUMINAMATH_GPT_center_of_hyperbola_l1962_196237


namespace NUMINAMATH_GPT_sum_of_transformed_numbers_l1962_196271

variables (a b x k S : ℝ)

-- Define the condition that a + b = S
def sum_condition : Prop := a + b = S

-- Define the function that represents the final sum after transformations
def final_sum (a b x k : ℝ) : ℝ :=
  k * (a + x) + k * (b + x)

-- The theorem statement to prove
theorem sum_of_transformed_numbers (h : sum_condition a b S) : 
  final_sum a b x k = k * S + 2 * k * x :=
by
  sorry

end NUMINAMATH_GPT_sum_of_transformed_numbers_l1962_196271


namespace NUMINAMATH_GPT_problem_statement_l1962_196279

theorem problem_statement (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : abc = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) := 
  sorry

end NUMINAMATH_GPT_problem_statement_l1962_196279


namespace NUMINAMATH_GPT_cost_per_amulet_is_30_l1962_196223

variable (days_sold : ℕ := 2)
variable (amulets_per_day : ℕ := 25)
variable (price_per_amulet : ℕ := 40)
variable (faire_percentage : ℕ := 10)
variable (profit : ℕ := 300)

def total_amulets_sold := days_sold * amulets_per_day
def total_revenue := total_amulets_sold * price_per_amulet
def faire_cut := total_revenue * faire_percentage / 100
def revenue_after_faire := total_revenue - faire_cut
def total_cost := revenue_after_faire - profit
def cost_per_amulet := total_cost / total_amulets_sold

theorem cost_per_amulet_is_30 : cost_per_amulet = 30 := by
  sorry

end NUMINAMATH_GPT_cost_per_amulet_is_30_l1962_196223


namespace NUMINAMATH_GPT_parabola_intersection_value_l1962_196242

theorem parabola_intersection_value (a : ℝ) (h : a^2 - a - 1 = 0) : a^2 - a + 2014 = 2015 :=
by
  sorry

end NUMINAMATH_GPT_parabola_intersection_value_l1962_196242


namespace NUMINAMATH_GPT_initial_distance_between_trains_l1962_196272

theorem initial_distance_between_trains :
  let length_train1 := 100 -- meters
  let length_train2 := 200 -- meters
  let speed_train1_kmph := 54 -- km/h
  let speed_train2_kmph := 72 -- km/h
  let time_hours := 1.999840012798976 -- hours
  
  -- Conversion to meters per second
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600 -- 15 m/s
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600 -- 20 m/s

  -- Conversion of time to seconds
  let time_seconds := time_hours * 3600 -- 7199.4240460755136 seconds

  -- Relative speed in meters per second
  let relative_speed := speed_train1_mps + speed_train2_mps -- 35 m/s

  -- Distance covered by both trains
  let distance_covered := relative_speed * time_seconds -- 251980.84161264498 meters

  -- Initial distance between the trains
  let initial_distance := distance_covered - (length_train1 + length_train2) -- 251680.84161264498 meters

  initial_distance = 251680.84161264498 := 
by
  sorry

end NUMINAMATH_GPT_initial_distance_between_trains_l1962_196272


namespace NUMINAMATH_GPT_even_stones_fraction_odd_stones_fraction_l1962_196285

/-- The fraction of the distributions of 12 indistinguishable stones to 4 distinguishable boxes where every box contains an even number of stones is 12/65. -/
theorem even_stones_fraction : (∀ (B1 B2 B3 B4 : ℕ), B1 % 2 = 0 ∧ B2 % 2 = 0 ∧ B3 % 2 = 0 ∧ B4 % 2 = 0 ∧ B1 + B2 + B3 + B4 = 12) → (84 / 455 = 12 / 65) := 
by sorry

/-- The fraction of the distributions of 12 indistinguishable stones to 4 distinguishable boxes where every box contains an odd number of stones is 1/13. -/
theorem odd_stones_fraction : (∀ (B1 B2 B3 B4 : ℕ), B1 % 2 = 1 ∧ B2 % 2 = 1 ∧ B3 % 2 = 1 ∧ B4 % 2 = 1 ∧ B1 + B2 + B3 + B4 = 12) → (35 / 455 = 1 / 13) := 
by sorry

end NUMINAMATH_GPT_even_stones_fraction_odd_stones_fraction_l1962_196285


namespace NUMINAMATH_GPT_required_speed_remaining_l1962_196295

theorem required_speed_remaining (total_distance : ℕ) (total_time : ℕ) (initial_speed : ℕ) (initial_time : ℕ) 
  (h1 : total_distance = 24) (h2 : total_time = 8) (h3 : initial_speed = 4) (h4 : initial_time = 4) :
  (total_distance - initial_speed * initial_time) / (total_time - initial_time) = 2 := by
  sorry

end NUMINAMATH_GPT_required_speed_remaining_l1962_196295


namespace NUMINAMATH_GPT_remainder_sum_1_to_12_div_9_l1962_196207

-- Define the sum of the first n natural numbers
def sum_natural (n : Nat) : Nat := n * (n + 1) / 2

-- Define the sum of the numbers from 1 to 12
def sum_1_to_12 := sum_natural 12

-- Define the remainder function
def remainder (a b : Nat) : Nat := a % b

-- Prove that the remainder when the sum of the numbers from 1 to 12 is divided by 9 is 6
theorem remainder_sum_1_to_12_div_9 : remainder sum_1_to_12 9 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_sum_1_to_12_div_9_l1962_196207


namespace NUMINAMATH_GPT_work_completion_days_l1962_196208

-- Define the work rates
def john_work_rate : ℚ := 1/8
def rose_work_rate : ℚ := 1/16
def dave_work_rate : ℚ := 1/12

-- Define the combined work rate
def combined_work_rate : ℚ := john_work_rate + rose_work_rate + dave_work_rate

-- Define the required number of days to complete the work together
def days_to_complete_work : ℚ := 1 / combined_work_rate

-- Prove that the total number of days required to complete the work is 48/13
theorem work_completion_days : days_to_complete_work = 48 / 13 :=
by 
  -- Here is where the actual proof would be, but it is not needed as per instructions
  sorry

end NUMINAMATH_GPT_work_completion_days_l1962_196208


namespace NUMINAMATH_GPT_four_thirds_of_product_eq_25_div_2_l1962_196270

noncomputable def a : ℚ := 15 / 4
noncomputable def b : ℚ := 5 / 2
noncomputable def c : ℚ := 4 / 3
noncomputable def d : ℚ := a * b
noncomputable def e : ℚ := c * d

theorem four_thirds_of_product_eq_25_div_2 : e = 25 / 2 := 
sorry

end NUMINAMATH_GPT_four_thirds_of_product_eq_25_div_2_l1962_196270


namespace NUMINAMATH_GPT_average_time_per_leg_l1962_196258

-- Conditions
def time_y : ℕ := 58
def time_z : ℕ := 26
def total_time : ℕ := time_y + time_z
def number_of_legs : ℕ := 2

-- Theorem stating the average time per leg
theorem average_time_per_leg : total_time / number_of_legs = 42 := by
  sorry

end NUMINAMATH_GPT_average_time_per_leg_l1962_196258


namespace NUMINAMATH_GPT_daily_sales_volume_selling_price_for_profit_l1962_196246

noncomputable def cost_price : ℝ := 40
noncomputable def initial_selling_price : ℝ := 60
noncomputable def initial_sales_volume : ℝ := 20
noncomputable def price_decrease_per_increase : ℝ := 5
noncomputable def volume_increase_per_decrease : ℝ := 10

theorem daily_sales_volume (p : ℝ) (v : ℝ) :
  v = initial_sales_volume + ((initial_selling_price - p) / price_decrease_per_increase) * volume_increase_per_decrease :=
sorry

theorem selling_price_for_profit (p : ℝ) (profit : ℝ) :
  profit = (p - cost_price) * (initial_sales_volume + ((initial_selling_price - p) / price_decrease_per_increase) * volume_increase_per_decrease) → p = 54 :=
sorry

end NUMINAMATH_GPT_daily_sales_volume_selling_price_for_profit_l1962_196246


namespace NUMINAMATH_GPT_infinite_series_sum_l1962_196221

noncomputable def partial_sum (n : ℕ) : ℚ := (2 * n - 1) / (n * (n + 1) * (n + 2))

theorem infinite_series_sum : (∑' n, partial_sum (n + 1)) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l1962_196221


namespace NUMINAMATH_GPT_circle_area_of_circumscribed_triangle_l1962_196266

theorem circle_area_of_circumscribed_triangle :
  let a := 12
  let b := 12
  let c := 10
  let height := Real.sqrt (a^2 - (c / 2)^2)
  let A := (1 / 2) * c * height
  let R := (a * b * c) / (4 * A)
  π * R^2 = (5184 / 119) * π := 
by
  let a := 12
  let b := 12
  let c := 10
  let height := Real.sqrt (a^2 - (c / 2)^2)
  let A := (1 / 2) * c * height
  let R := (a * b * c) / (4 * A)
  have h1 : height = Real.sqrt (a^2 - (c / 2)^2) := by sorry
  have h2 : A = (1 / 2) * c * height := by sorry
  have h3 : R = (a * b * c) / (4 * A) := by sorry
  have h4 : π * R^2 = (5184 / 119) * π := by sorry
  exact h4

end NUMINAMATH_GPT_circle_area_of_circumscribed_triangle_l1962_196266


namespace NUMINAMATH_GPT_rectangular_solid_surface_area_l1962_196203

theorem rectangular_solid_surface_area
  (length : ℕ) (width : ℕ) (depth : ℕ)
  (h_length : length = 9) (h_width : width = 8) (h_depth : depth = 5) :
  2 * (length * width + width * depth + length * depth) = 314 := 
  by
  sorry

end NUMINAMATH_GPT_rectangular_solid_surface_area_l1962_196203


namespace NUMINAMATH_GPT_value_of_sum_plus_five_l1962_196259

theorem value_of_sum_plus_five (a b : ℕ) (h : 4 * a^2 + 4 * b^2 + 8 * a * b = 100) :
  (a + b) + 5 = 10 :=
sorry

end NUMINAMATH_GPT_value_of_sum_plus_five_l1962_196259


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1962_196215

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2) :
  ( (1 + x) / (1 - x) / (x - (2 * x / (1 - x))) = - (Real.sqrt 2 + 2) / 2) :=
by
  rw [h]
  simp
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1962_196215


namespace NUMINAMATH_GPT_expression_behavior_l1962_196234

theorem expression_behavior (x : ℝ) (h1 : -3 < x) (h2 : x < 2) :
  ¬∃ m, ∀ y : ℝ, (h3 : -3 < y) → (h4 : y < 2) → (x ≠ 1) → (y ≠ 1) → 
    (m <= (y^2 - 3*y + 3) / (y - 1)) ∧ 
    (m >= (y^2 - 3*y + 3) / (y - 1)) :=
sorry

end NUMINAMATH_GPT_expression_behavior_l1962_196234


namespace NUMINAMATH_GPT_cost_of_each_croissant_l1962_196291

theorem cost_of_each_croissant 
  (quiches_price : ℝ) (num_quiches : ℕ) (each_quiche_cost : ℝ)
  (buttermilk_biscuits_price : ℝ) (num_biscuits : ℕ) (each_biscuit_cost : ℝ)
  (total_cost_with_discount : ℝ) (discount_rate : ℝ)
  (num_croissants : ℕ) (croissant_price : ℝ) :
  quiches_price = num_quiches * each_quiche_cost →
  each_quiche_cost = 15 →
  num_quiches = 2 →
  buttermilk_biscuits_price = num_biscuits * each_biscuit_cost →
  each_biscuit_cost = 2 →
  num_biscuits = 6 →
  discount_rate = 0.10 →
  (quiches_price + buttermilk_biscuits_price + (num_croissants * croissant_price)) * (1 - discount_rate) = total_cost_with_discount →
  total_cost_with_discount = 54 →
  num_croissants = 6 →
  croissant_price = 3 :=
sorry

end NUMINAMATH_GPT_cost_of_each_croissant_l1962_196291


namespace NUMINAMATH_GPT_best_purchase_option_l1962_196236

-- Define the prices and discount conditions for each store
def technik_city_price_before_discount : ℝ := 2000 + 4000
def technomarket_price_before_discount : ℝ := 1500 + 4800

def technik_city_discount : ℝ := technik_city_price_before_discount * 0.10
def technomarket_bonus : ℝ := technomarket_price_before_discount * 0.20

def technik_city_final_price : ℝ := technik_city_price_before_discount - technik_city_discount
def technomarket_final_price : ℝ := technomarket_price_before_discount

-- The theorem stating the ultimate proof problem
theorem best_purchase_option : technik_city_final_price < technomarket_final_price :=
by
  -- Replace 'sorry' with the actual proof if required
  sorry

end NUMINAMATH_GPT_best_purchase_option_l1962_196236


namespace NUMINAMATH_GPT_paula_aunt_gave_her_total_money_l1962_196224

theorem paula_aunt_gave_her_total_money :
  let shirt_price := 11
  let pants_price := 13
  let shirts_bought := 2
  let money_left := 74
  let total_spent := shirts_bought * shirt_price + pants_price
  total_spent + money_left = 109 :=
by
  let shirt_price := 11
  let pants_price := 13
  let shirts_bought := 2
  let money_left := 74
  let total_spent := shirts_bought * shirt_price + pants_price
  show total_spent + money_left = 109
  sorry

end NUMINAMATH_GPT_paula_aunt_gave_her_total_money_l1962_196224


namespace NUMINAMATH_GPT_trajectory_of_G_l1962_196209

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop :=
  9 * x^2 / 4 + 3 * y^2 = 1

-- State the theorem
theorem trajectory_of_G (P G : ℝ × ℝ) (hP : ellipse P.1 P.2) (hG_relation : ∃ k : ℝ, k = 2 ∧ P = (3 * G.1, 3 * G.2)) :
  trajectory G.1 G.2 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_G_l1962_196209


namespace NUMINAMATH_GPT_scientific_notation_of_18500000_l1962_196240

-- Definition of scientific notation function
def scientific_notation (n : ℕ) : string := sorry

-- Problem statement
theorem scientific_notation_of_18500000 : 
  scientific_notation 18500000 = "1.85 × 10^7" :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_18500000_l1962_196240


namespace NUMINAMATH_GPT_all_statements_correct_l1962_196257

-- Definitions based on the problem conditions
def population_size : ℕ := 60000
def sample_size : ℕ := 1000
def is_sampling_survey (population_size sample_size : ℕ) : Prop := sample_size < population_size
def is_population (n : ℕ) : Prop := n = 60000
def is_sample (population_size sample_size : ℕ) : Prop := sample_size < population_size
def matches_sample_size (n : ℕ) : Prop := n = 1000

-- Lean problem statement representing the proof that all statements are correct
theorem all_statements_correct :
  is_sampling_survey population_size sample_size ∧
  is_population population_size ∧ 
  is_sample population_size sample_size ∧
  matches_sample_size sample_size := by
  sorry

end NUMINAMATH_GPT_all_statements_correct_l1962_196257


namespace NUMINAMATH_GPT_find_m_n_and_max_value_l1962_196247

-- Define the function f
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + 3 * m + n

-- Define a predicate for the function being even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define the conditions and what we want to prove
theorem find_m_n_and_max_value :
  ∀ m n : ℝ,
    is_even_function (f m n) →
    (m - 1 ≤ 2 * m) →
      (m = 1 / 3 ∧ n = 0) ∧ 
      (∀ x : ℝ, -2 / 3 ≤ x ∧ x ≤ 2 / 3 → f (1/3) 0 x ≤ 31 / 27) :=
by
  sorry

end NUMINAMATH_GPT_find_m_n_and_max_value_l1962_196247


namespace NUMINAMATH_GPT_find_b_l1962_196217

noncomputable def Q (x d b e : ℝ) : ℝ := x^3 + d*x^2 + b*x + e

theorem find_b (d b e : ℝ) (h1 : -d / 3 = -e) (h2 : -e = 1 + d + b + e) (h3 : e = 6) : b = -31 :=
by sorry

end NUMINAMATH_GPT_find_b_l1962_196217


namespace NUMINAMATH_GPT_odd_function_increasing_l1962_196277

variables {f : ℝ → ℝ}

/-- Let f be an odd function defined on (-∞, 0) ∪ (0, ∞). 
If ∀ y z ∈ (0, ∞), y ≠ z → (f y - f z) / (y - z) > 0, then f(-3) > f(-5). -/
theorem odd_function_increasing {f : ℝ → ℝ} 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ y z : ℝ, y > 0 → z > 0 → y ≠ z → (f y - f z) / (y - z) > 0) :
  f (-3) > f (-5) :=
sorry

end NUMINAMATH_GPT_odd_function_increasing_l1962_196277


namespace NUMINAMATH_GPT_rectangle_ABCD_area_l1962_196260

def rectangle_area (x : ℕ) : ℕ :=
  let side_lengths := [x, x+1, x+2, x+3];
  let width := side_lengths.sum;
  let height := width - x;
  width * height

theorem rectangle_ABCD_area : rectangle_area 1 = 143 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ABCD_area_l1962_196260


namespace NUMINAMATH_GPT_median_song_length_l1962_196287

-- Define the list of song lengths in seconds
def song_lengths : List ℕ := [32, 43, 58, 65, 70, 72, 75, 80, 145, 150, 175, 180, 195, 210, 215, 225, 250, 252]

-- Define the statement that the median length of the songs is 147.5 seconds
theorem median_song_length : ∃ median : ℕ, median = 147 ∧ (median : ℚ) + 0.5 = 147.5 := by
  sorry

end NUMINAMATH_GPT_median_song_length_l1962_196287


namespace NUMINAMATH_GPT_polynomial_root_divisibility_l1962_196210

noncomputable def p (x : ℤ) (a b c : ℤ) : ℤ := x^3 + a * x^2 + b * x + c

theorem polynomial_root_divisibility (a b c : ℤ) (h : ∃ u v : ℤ, p 0 a b c = (u * v * u * v)) :
  2 * (p (-1) a b c) ∣ (p 1 a b c + p (-1) a b c - 2 * (1 + p 0 a b c)) :=
sorry

end NUMINAMATH_GPT_polynomial_root_divisibility_l1962_196210


namespace NUMINAMATH_GPT_scientific_notation_of_188_million_l1962_196276

theorem scientific_notation_of_188_million : 
  (188000000 : ℝ) = 1.88 * 10^8 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_188_million_l1962_196276


namespace NUMINAMATH_GPT_cheryl_tournament_cost_is_1440_l1962_196283

noncomputable def cheryl_electricity_bill : ℝ := 800
noncomputable def additional_for_cell_phone : ℝ := 400
noncomputable def cheryl_cell_phone_expenses : ℝ := cheryl_electricity_bill + additional_for_cell_phone
noncomputable def tournament_cost_percentage : ℝ := 0.2
noncomputable def additional_tournament_cost : ℝ := tournament_cost_percentage * cheryl_cell_phone_expenses
noncomputable def total_tournament_cost : ℝ := cheryl_cell_phone_expenses + additional_tournament_cost

theorem cheryl_tournament_cost_is_1440 : total_tournament_cost = 1440 := by
  sorry

end NUMINAMATH_GPT_cheryl_tournament_cost_is_1440_l1962_196283


namespace NUMINAMATH_GPT_tan_theta_values_l1962_196275

theorem tan_theta_values (θ : ℝ) (h₁ : 0 < θ ∧ θ < Real.pi / 2) (h₂ : 12 / Real.sin θ + 12 / Real.cos θ = 35) : 
  Real.tan θ = 4 / 3 ∨ Real.tan θ = 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_tan_theta_values_l1962_196275


namespace NUMINAMATH_GPT_minimum_perimeter_area_l1962_196227

-- Define the focus point F of the parabola and point A
def F : ℝ × ℝ := (1, 0)  -- Focus for the parabola y² = 4x is (1, 0)
def A : ℝ × ℝ := (5, 4)

-- Parabola definition as a set of points (x, y) such that y² = 4x
def is_on_parabola (B : ℝ × ℝ) : Prop := B.2 * B.2 = 4 * B.1

-- The area of triangle ABF
def triangle_area (A B F : ℝ × ℝ) : ℝ := 
  0.5 * abs ((A.1 - B.1) * (A.2 - F.2) - (A.1 - F.1) * (A.2 - B.2))

-- Statement: The area of ∆ABF is 2 when the perimeter of ∆ABF is minimum
theorem minimum_perimeter_area (B : ℝ × ℝ) (hB : is_on_parabola B) 
  (hA_B_perimeter_min : ∀ (C : ℝ × ℝ), is_on_parabola C → 
                        (dist A C + dist C F ≥ dist A B + dist B F)) : 
  triangle_area A B F = 2 := 
sorry

end NUMINAMATH_GPT_minimum_perimeter_area_l1962_196227


namespace NUMINAMATH_GPT_percentage_of_students_play_sports_l1962_196292

def total_students : ℕ := 400
def soccer_percentage : ℝ := 0.125
def soccer_players : ℕ := 26

theorem percentage_of_students_play_sports : 
  ∃ P : ℝ, (soccer_percentage * P = soccer_players) → (P / total_students * 100 = 52) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_students_play_sports_l1962_196292


namespace NUMINAMATH_GPT_roots_ratio_sum_eq_six_l1962_196232

theorem roots_ratio_sum_eq_six (x1 x2 : ℝ) (h1 : 2 * x1^2 - 4 * x1 + 1 = 0) (h2 : 2 * x2^2 - 4 * x2 + 1 = 0) :
  (x1 / x2) + (x2 / x1) = 6 :=
sorry

end NUMINAMATH_GPT_roots_ratio_sum_eq_six_l1962_196232


namespace NUMINAMATH_GPT_male_students_in_grade_l1962_196219

-- Define the total number of students and the number of students in the sample
def total_students : ℕ := 1200
def sample_students : ℕ := 30

-- Define the number of female students in the sample
def female_students_sample : ℕ := 14

-- Calculate the number of male students in the sample
def male_students_sample := sample_students - female_students_sample

-- State the main theorem
theorem male_students_in_grade :
  (male_students_sample : ℕ) * total_students / sample_students = 640 :=
by
  -- placeholder for calculations based on provided conditions
  sorry

end NUMINAMATH_GPT_male_students_in_grade_l1962_196219


namespace NUMINAMATH_GPT_feet_per_inch_of_model_l1962_196252

theorem feet_per_inch_of_model 
  (height_tower : ℝ)
  (height_model : ℝ)
  (height_tower_eq : height_tower = 984)
  (height_model_eq : height_model = 6)
  : (height_tower / height_model) = 164 :=
by
  -- Assume the proof here
  sorry

end NUMINAMATH_GPT_feet_per_inch_of_model_l1962_196252


namespace NUMINAMATH_GPT_compute_f_2_neg3_neg1_l1962_196225

def f (p q r : ℤ) : ℚ := (r + p : ℚ) / (r - q + 1 : ℚ)

theorem compute_f_2_neg3_neg1 : f 2 (-3) (-1) = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_compute_f_2_neg3_neg1_l1962_196225


namespace NUMINAMATH_GPT_inequality_solution_set_nonempty_l1962_196201

-- Define the statement
theorem inequality_solution_set_nonempty (m : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - 1| < m) ↔ m > 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_nonempty_l1962_196201


namespace NUMINAMATH_GPT_casey_nail_decorating_time_l1962_196273

theorem casey_nail_decorating_time 
  (n_toenails n_fingernails : ℕ)
  (t_apply t_dry : ℕ)
  (coats : ℕ)
  (h1 : n_toenails = 10)
  (h2 : n_fingernails = 10)
  (h3 : t_apply = 20)
  (h4 : t_dry = 20)
  (h5 : coats = 3) :
  20 * (t_apply + t_dry) * coats = 120 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_casey_nail_decorating_time_l1962_196273


namespace NUMINAMATH_GPT_line_parallel_not_coincident_l1962_196278

theorem line_parallel_not_coincident (a : ℝ) :
  (a = 3) ↔ (∀ x y, (a * x + 2 * y + 3 * a = 0) ∧ (3 * x + (a - 1) * y + 7 - a = 0) → 
              (∃ k : Real, a / 3 = k ∧ k ≠ 3 * a / (7 - a))) :=
by
  sorry

end NUMINAMATH_GPT_line_parallel_not_coincident_l1962_196278


namespace NUMINAMATH_GPT_cost_price_of_book_l1962_196290

theorem cost_price_of_book 
  (C : ℝ)
  (h1 : ∃ C, C > 0)
  (h2 : 1.10 * C = 1.15 * C - 120) :
  C = 2400 :=
sorry

end NUMINAMATH_GPT_cost_price_of_book_l1962_196290


namespace NUMINAMATH_GPT_ratio_of_areas_l1962_196284

theorem ratio_of_areas
  (s: ℝ) (h₁: s > 0)
  (large_square_area: ℝ)
  (inscribed_square_area: ℝ)
  (harea₁: large_square_area = s * s)
  (harea₂: inscribed_square_area = (s / 2) * (s / 2)) :
  inscribed_square_area / large_square_area = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1962_196284


namespace NUMINAMATH_GPT_min_star_value_l1962_196269

theorem min_star_value :
  ∃ (star : ℕ), (98348 * 10 + star) % 72 = 0 ∧ (∀ (x : ℕ), (98348 * 10 + x) % 72 = 0 → star ≤ x) := sorry

end NUMINAMATH_GPT_min_star_value_l1962_196269


namespace NUMINAMATH_GPT_value_of_y_l1962_196294

theorem value_of_y (x y : ℝ) (h1 : x = 2) (h2 : x^(2*y) = 4) : y = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l1962_196294


namespace NUMINAMATH_GPT_total_movies_correct_l1962_196239

def num_movies_Screen1 : Nat := 3
def num_movies_Screen2 : Nat := 4
def num_movies_Screen3 : Nat := 2
def num_movies_Screen4 : Nat := 3
def num_movies_Screen5 : Nat := 5
def num_movies_Screen6 : Nat := 2

def total_movies : Nat :=
  num_movies_Screen1 + num_movies_Screen2 + num_movies_Screen3 + num_movies_Screen4 + num_movies_Screen5 + num_movies_Screen6

theorem total_movies_correct :
  total_movies = 19 :=
by 
  sorry

end NUMINAMATH_GPT_total_movies_correct_l1962_196239


namespace NUMINAMATH_GPT_area_quadrilateral_l1962_196243

theorem area_quadrilateral (EF GH: ℝ) (EHG: ℝ) 
  (h1 : EF = 9) (h2 : GH = 12) (h3 : GH = EH) (h4 : EHG = 75) 
  (a b c : ℕ)
  : 
  (∀ (a b c : ℕ), 
  a = 26 ∧ b = 18 ∧ c = 6 → 
  a + b + c = 50) := 
sorry

end NUMINAMATH_GPT_area_quadrilateral_l1962_196243


namespace NUMINAMATH_GPT_perpendicular_vectors_x_value_l1962_196200

theorem perpendicular_vectors_x_value 
  (x : ℝ) (a b : ℝ × ℝ) (hₐ : a = (1, -2)) (hᵦ : b = (3, x)) (h_perpendicular : a.1 * b.1 + a.2 * b.2 = 0) : 
  x = 3 / 2 :=
by
  -- The proof is not required, hence we use 'sorry'
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_x_value_l1962_196200


namespace NUMINAMATH_GPT_x_proportionality_find_x_value_l1962_196216

theorem x_proportionality (m n : ℝ) (x z : ℝ) (h1 : ∀ y, x = m * y^4) (h2 : ∀ z, y = n / z^2) (h3 : x = 4) (h4 : z = 8) :
  ∃ k, ∀ z : ℝ, x = k / z^8 := 
sorry

theorem find_x_value (m n : ℝ) (k : ℝ) (h1 : ∀ y, x = m * y^4) (h2 : ∀ z, y = n / z^2) (h5 : k = 67108864) :
  ∀ z, (z = 32 → x = 1 / 16) :=
sorry

end NUMINAMATH_GPT_x_proportionality_find_x_value_l1962_196216


namespace NUMINAMATH_GPT_percentage_markup_l1962_196298

theorem percentage_markup (SP CP : ℕ) (h1 : SP = 8340) (h2 : CP = 6672) :
  ((SP - CP) / CP * 100) = 25 :=
by
  -- Before proving, we state our assumptions
  sorry

end NUMINAMATH_GPT_percentage_markup_l1962_196298


namespace NUMINAMATH_GPT_total_remaining_books_l1962_196299

-- Define the initial conditions as constants
def total_books_crazy_silly_school : ℕ := 14
def read_books_crazy_silly_school : ℕ := 8
def total_books_mystical_adventures : ℕ := 10
def read_books_mystical_adventures : ℕ := 5
def total_books_sci_fi_universe : ℕ := 18
def read_books_sci_fi_universe : ℕ := 12

-- Define the remaining books calculation
def remaining_books_crazy_silly_school : ℕ :=
  total_books_crazy_silly_school - read_books_crazy_silly_school

def remaining_books_mystical_adventures : ℕ :=
  total_books_mystical_adventures - read_books_mystical_adventures

def remaining_books_sci_fi_universe : ℕ :=
  total_books_sci_fi_universe - read_books_sci_fi_universe

-- Define the proof statement
theorem total_remaining_books : 
  remaining_books_crazy_silly_school + remaining_books_mystical_adventures + remaining_books_sci_fi_universe = 17 := by
  sorry

end NUMINAMATH_GPT_total_remaining_books_l1962_196299


namespace NUMINAMATH_GPT_license_plates_count_l1962_196289

theorem license_plates_count : (6 * 10^5 * 26^3) = 10584576000 := by
  sorry

end NUMINAMATH_GPT_license_plates_count_l1962_196289


namespace NUMINAMATH_GPT_final_position_A_final_position_B_fuel_consumption_A_fuel_consumption_B_less_fuel_consumption_l1962_196263

-- Definitions of the driving records for trainee A and B
def driving_record_A : List Int := [15, -2, 5, -1, 10, -3, -2, 12, 4, -5, 6]
def driving_record_B : List Int := [-17, 9, -2, 8, 6, 9, -5, -1, 4, -7, -8]

-- Fuel consumption rate per kilometer
variable (a : ℝ)

-- Proof statements in Lean
theorem final_position_A : driving_record_A.sum = 39 := by sorry
theorem final_position_B : driving_record_B.sum = -4 := by sorry
theorem fuel_consumption_A : (driving_record_A.map (abs)).sum * a = 65 * a := by sorry
theorem fuel_consumption_B : (driving_record_B.map (abs)).sum * a = 76 * a := by sorry
theorem less_fuel_consumption : (driving_record_A.map (abs)).sum * a < (driving_record_B.map (abs)).sum * a := by sorry

end NUMINAMATH_GPT_final_position_A_final_position_B_fuel_consumption_A_fuel_consumption_B_less_fuel_consumption_l1962_196263


namespace NUMINAMATH_GPT_mondays_in_first_70_days_l1962_196245

theorem mondays_in_first_70_days (days : ℕ) (h1 : days = 70) (mondays_per_week : ℕ) (h2 : mondays_per_week = 1) : 
  ∃ (mondays : ℕ), mondays = 10 := 
by
  sorry

end NUMINAMATH_GPT_mondays_in_first_70_days_l1962_196245


namespace NUMINAMATH_GPT_restaurant_total_tables_l1962_196244

theorem restaurant_total_tables (N O : ℕ) (h1 : 6 * N + 4 * O = 212) (h2 : N = O + 12) : N + O = 40 :=
sorry

end NUMINAMATH_GPT_restaurant_total_tables_l1962_196244


namespace NUMINAMATH_GPT_Q_root_l1962_196296

def Q (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 11

theorem Q_root : Q (3^(1 / 3 : ℝ) + 2) = 0 := sorry

end NUMINAMATH_GPT_Q_root_l1962_196296


namespace NUMINAMATH_GPT_solve_inequality_l1962_196204

theorem solve_inequality (x : ℝ) : (|x - 3| + |x - 5| ≥ 4) ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1962_196204


namespace NUMINAMATH_GPT_shorter_leg_of_right_triangle_l1962_196254

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end NUMINAMATH_GPT_shorter_leg_of_right_triangle_l1962_196254


namespace NUMINAMATH_GPT_black_to_brown_ratio_l1962_196222

-- Definitions of the given conditions
def total_shoes : ℕ := 66
def brown_shoes : ℕ := 22
def black_shoes : ℕ := total_shoes - brown_shoes

-- Lean 4 problem statement: Prove the ratio of black shoes to brown shoes is 2:1
theorem black_to_brown_ratio :
  (black_shoes / Nat.gcd black_shoes brown_shoes) = 2 ∧ (brown_shoes / Nat.gcd black_shoes brown_shoes) = 1 := by
sorry

end NUMINAMATH_GPT_black_to_brown_ratio_l1962_196222


namespace NUMINAMATH_GPT_initial_earning_members_l1962_196264

theorem initial_earning_members (n : ℕ)
  (avg_income_initial : ℕ) (avg_income_after : ℕ) (income_deceased : ℕ)
  (h1 : avg_income_initial = 735)
  (h2 : avg_income_after = 590)
  (h3 : income_deceased = 1170)
  (h4 : 735 * n - 1170 = 590 * (n - 1)) :
  n = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_earning_members_l1962_196264


namespace NUMINAMATH_GPT_find_x_if_vectors_parallel_l1962_196261

/--
Given the vectors a = (2 * x + 1, 3) and b = (2 - x, 1), if a is parallel to b, 
then x must be equal to 1.
-/
theorem find_x_if_vectors_parallel (x : ℝ) :
  let a := (2 * x + 1, 3)
  let b := (2 - x, 1)
  (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_if_vectors_parallel_l1962_196261


namespace NUMINAMATH_GPT_rectangle_area_unchanged_l1962_196267

theorem rectangle_area_unchanged (l w : ℝ) (h : l * w = 432) : 
  0.8 * l * 1.25 * w = 432 := 
by {
  -- The proof goes here
  sorry
}

end NUMINAMATH_GPT_rectangle_area_unchanged_l1962_196267


namespace NUMINAMATH_GPT_distinct_roots_implies_m_greater_than_half_find_m_given_condition_l1962_196241

-- Define the quadratic equation with a free parameter m
def quadratic_eq (x : ℝ) (m : ℝ) : Prop :=
  x^2 - 4 * x - 2 * m + 5 = 0

-- Prove that if the quadratic equation has distinct roots, then m > 1/2
theorem distinct_roots_implies_m_greater_than_half (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂) →
  m > 1 / 2 :=
by
  sorry

-- Given that x₁ and x₂ satisfy both the quadratic equation and the sum-product condition, find the value of m
theorem find_m_given_condition (m : ℝ) (x₁ x₂ : ℝ) :
  quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ + x₁ + x₂ = m^2 + 6) → 
  m = 1 :=
by
  sorry

end NUMINAMATH_GPT_distinct_roots_implies_m_greater_than_half_find_m_given_condition_l1962_196241


namespace NUMINAMATH_GPT_cubic_roots_inequality_l1962_196286

theorem cubic_roots_inequality (a b c : ℝ) (h : ∃ (α β γ : ℝ), (x : ℝ) → x^3 + a * x^2 + b * x + c = (x - α) * (x - β) * (x - γ)) :
  3 * b ≤ a^2 :=
sorry

end NUMINAMATH_GPT_cubic_roots_inequality_l1962_196286
