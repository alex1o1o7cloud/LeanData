import Mathlib

namespace NUMINAMATH_GPT_find_n_l212_21295

def binomial_coefficient_sum (n : ℕ) (a b : ℝ) : ℝ :=
  (a + b) ^ n

def expanded_coefficient_sum (n : ℕ) (a b : ℝ) : ℝ :=
  (a + 3 * b) ^ n

theorem find_n (n : ℕ) :
  (expanded_coefficient_sum n 1 1) / (binomial_coefficient_sum n 1 1) = 64 → n = 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_n_l212_21295


namespace NUMINAMATH_GPT_ratio_of_sequence_l212_21219

variables (a b c : ℝ)

-- Condition 1: arithmetic sequence
def arithmetic_sequence : Prop := 2 * b = a + c

-- Condition 2: geometric sequence
def geometric_sequence : Prop := c^2 = a * b

-- Theorem stating the ratio of a:b:c
theorem ratio_of_sequence (h1 : arithmetic_sequence a b c) (h2 : geometric_sequence a b c) : 
  (a = 4 * b) ∧ (c = -2 * b) :=
sorry

end NUMINAMATH_GPT_ratio_of_sequence_l212_21219


namespace NUMINAMATH_GPT_sum_of_vertices_l212_21233

theorem sum_of_vertices (rect_verts: Nat) (pent_verts: Nat) (h1: rect_verts = 4) (h2: pent_verts = 5) : rect_verts + pent_verts = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_vertices_l212_21233


namespace NUMINAMATH_GPT_monotonicity_and_zeros_l212_21255

open Real

noncomputable def f (x k : ℝ) : ℝ := exp x - k * x + k

theorem monotonicity_and_zeros
  (k : ℝ)
  (h₁ : k > exp 2)
  (x₁ x₂ : ℝ)
  (h₂ : f x₁ k = 0)
  (h₃ : f x₂ k = 0)
  (h₄ : x₁ ≠ x₂) :
  x₁ + x₂ > 4 := 
sorry

end NUMINAMATH_GPT_monotonicity_and_zeros_l212_21255


namespace NUMINAMATH_GPT_numbers_distance_one_neg_two_l212_21216

theorem numbers_distance_one_neg_two (x : ℝ) (h : abs (x + 2) = 1) : x = -1 ∨ x = -3 := 
sorry

end NUMINAMATH_GPT_numbers_distance_one_neg_two_l212_21216


namespace NUMINAMATH_GPT_final_weight_of_box_l212_21274

theorem final_weight_of_box (w1 w2 w3 w4 : ℝ) (h1 : w1 = 2) (h2 : w2 = 3 * w1) (h3 : w3 = w2 + 2) (h4 : w4 = 2 * w3) : w4 = 16 :=
by
  sorry

end NUMINAMATH_GPT_final_weight_of_box_l212_21274


namespace NUMINAMATH_GPT_odd_function_f_l212_21217

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_f (f_odd : ∀ x : ℝ, f (-x) = - f x)
                       (f_lt_0 : ∀ x : ℝ, x < 0 → f x = x * (x - 1)) :
  ∀ x : ℝ, x > 0 → f x = - x * (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_odd_function_f_l212_21217


namespace NUMINAMATH_GPT_player_B_wins_l212_21232

-- Here we define the scenario and properties from the problem statement.
def initial_pile1 := 100
def initial_pile2 := 252

-- Definition of a turn, conditions and the win condition based on the problem
structure Turn :=
  (pile1 : ℕ)
  (pile2 : ℕ)
  (player_A_turn : Bool)  -- True if it's player A's turn, False if it's player B's turn

-- The game conditions and strategy for determining the winner
def will_player_B_win (initial_pile1 initial_pile2 : ℕ) : Bool :=
  -- assuming the conditions are provided and correctly analyzed, 
  -- we directly state the known result according to the optimal strategies from the solution
  true  -- B wins as per the solution's analysis if both play optimally.

-- The final theorem stating Player B wins given the initial conditions with both playing optimally and A going first.
theorem player_B_wins : will_player_B_win initial_pile1 initial_pile2 = true :=
  sorry  -- Proof omitted.

end NUMINAMATH_GPT_player_B_wins_l212_21232


namespace NUMINAMATH_GPT_total_students_in_lunchroom_l212_21226

theorem total_students_in_lunchroom (students_per_table : ℕ) (num_tables : ℕ) (total_students : ℕ) :
  students_per_table = 6 → 
  num_tables = 34 → 
  total_students = students_per_table * num_tables → 
  total_students = 204 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_total_students_in_lunchroom_l212_21226


namespace NUMINAMATH_GPT_tan_alpha_eq_neg2_complex_expression_eq_neg5_l212_21228

variables (α : ℝ)
variables (h_sin : Real.sin α = - (2 * Real.sqrt 5) / 5)
variables (h_tan_neg : Real.tan α < 0)

theorem tan_alpha_eq_neg2 :
  Real.tan α = -2 :=
sorry

theorem complex_expression_eq_neg5 :
  (2 * Real.sin (α + Real.pi) + Real.cos (2 * Real.pi - α)) /
  (Real.cos (α - Real.pi / 2) - Real.sin (3 * Real.pi / 2 + α)) = -5 :=
sorry

end NUMINAMATH_GPT_tan_alpha_eq_neg2_complex_expression_eq_neg5_l212_21228


namespace NUMINAMATH_GPT_A_50_correct_l212_21293

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := 
  ![![3, 2], 
    ![-8, -5]]

-- The theorem to prove
theorem A_50_correct : A^50 = ![![(-199 : ℤ), -100], 
                                 ![400, 201]] := 
by
  sorry

end NUMINAMATH_GPT_A_50_correct_l212_21293


namespace NUMINAMATH_GPT_find_k_inverse_proportion_l212_21278

theorem find_k_inverse_proportion :
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, ∀ y : ℝ, (x = 1 ∧ y = 3) → (y = k / x)) ∧ k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_inverse_proportion_l212_21278


namespace NUMINAMATH_GPT_lily_pad_half_lake_l212_21205

theorem lily_pad_half_lake
  (P : ℕ → ℝ) -- Define a function P(n) which represents the size of the patch on day n.
  (h1 : ∀ n, P n = P (n - 1) * 2) -- Every day, the patch doubles in size.
  (h2 : P 58 = 1) -- It takes 58 days for the patch to cover the entire lake (normalized to 1).
  : P 57 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_lily_pad_half_lake_l212_21205


namespace NUMINAMATH_GPT_ellipse_properties_l212_21213

theorem ellipse_properties :
  ∀ {x y : ℝ}, 4 * x^2 + 2 * y^2 = 16 →
    (∃ a b e c, a = 2 * Real.sqrt 2 ∧ b = 2 ∧ e = Real.sqrt 2 / 2 ∧ c = 2) ∧
    (∃ f1 f2, f1 = (0, 2) ∧ f2 = (0, -2)) ∧
    (∃ v1 v2 v3 v4, v1 = (0, 2 * Real.sqrt 2) ∧ v2 = (0, -2 * Real.sqrt 2) ∧ v3 = (2, 0) ∧ v4 = (-2, 0)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_properties_l212_21213


namespace NUMINAMATH_GPT_speed_conversion_l212_21284

theorem speed_conversion (s : ℝ) (h1 : s = 1 / 3) : s * 3.6 = 1.2 := by
  -- Proof follows from the conditions given
  sorry

end NUMINAMATH_GPT_speed_conversion_l212_21284


namespace NUMINAMATH_GPT_circle_equation_passing_through_points_symmetric_circle_equation_midpoint_trajectory_equation_l212_21222

-- Prove the equation of the circle passing through points A and B with center on a specified line
theorem circle_equation_passing_through_points
  (A B : ℝ × ℝ) (line : ℝ → ℝ → Prop)
  (N : ℝ → ℝ → Prop) :
  A = (3, 1) →
  B = (-1, 3) →
  (∀ x y, line x y ↔ 3 * x - y - 2 = 0) →
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  sorry :=
sorry

-- Prove the symmetric circle equation regarding a specified line
theorem symmetric_circle_equation
  (N N' : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) :
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  (∀ x y, N' x y ↔ (x - 1)^2 + (y - 5)^2 = 10) →
  (∀ x y, line x y ↔ x - y + 3 = 0) →
  sorry :=
sorry

-- Prove the trajectory equation of the midpoint
theorem midpoint_trajectory_equation
  (C : ℝ × ℝ) (N : ℝ → ℝ → Prop) (M_trajectory : ℝ → ℝ → Prop) :
  C = (3, 0) →
  (∀ x y, N x y ↔ (x - 2)^2 + (y - 4)^2 = 10) →
  (∀ x y, M_trajectory x y ↔ (x - 5 / 2)^2 + (y - 2)^2 = 5 / 2) →
  sorry :=
sorry

end NUMINAMATH_GPT_circle_equation_passing_through_points_symmetric_circle_equation_midpoint_trajectory_equation_l212_21222


namespace NUMINAMATH_GPT_angle_between_line_and_plane_l212_21246

noncomputable def vector_angle (m n : ℝ) : ℝ := 120

theorem angle_between_line_and_plane (m n : ℝ) : 
  (vector_angle m n = 120) → (90 - (vector_angle m n - 90) = 30) :=
by sorry

end NUMINAMATH_GPT_angle_between_line_and_plane_l212_21246


namespace NUMINAMATH_GPT_theater_ticket_cost_l212_21298

theorem theater_ticket_cost
  (O B : ℕ)
  (h1 : O + B = 370)
  (h2 : B = O + 190) 
  : 12 * O + 8 * B = 3320 :=
by
  sorry

end NUMINAMATH_GPT_theater_ticket_cost_l212_21298


namespace NUMINAMATH_GPT_part1_solution_l212_21245

def f (x m : ℝ) := |x + m| + |2 * x + 1|

theorem part1_solution (x : ℝ) : f x (-1) ≤ 3 → -1 ≤ x ∧ x ≤ 1 := 
sorry

end NUMINAMATH_GPT_part1_solution_l212_21245


namespace NUMINAMATH_GPT_range_of_m_l212_21281

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) ↔ (m ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l212_21281


namespace NUMINAMATH_GPT_cookout_ratio_l212_21220

theorem cookout_ratio (K_2004 K_2005 : ℕ) (h1 : K_2004 = 60) (h2 : (2 / 3) * K_2005 = 20) :
  K_2005 / K_2004 = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_cookout_ratio_l212_21220


namespace NUMINAMATH_GPT_rectangle_pentagon_ratio_l212_21242

theorem rectangle_pentagon_ratio
  (l w p : ℝ)
  (h1 : l = 2 * w)
  (h2 : 2 * (l + w) = 30)
  (h3 : 5 * p = 30) :
  l / p = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_pentagon_ratio_l212_21242


namespace NUMINAMATH_GPT_jane_last_day_vases_l212_21267

def vasesPerDay : Nat := 16
def totalVases : Nat := 248

theorem jane_last_day_vases : totalVases % vasesPerDay = 8 := by
  sorry

end NUMINAMATH_GPT_jane_last_day_vases_l212_21267


namespace NUMINAMATH_GPT_required_speed_l212_21223

theorem required_speed
  (D T : ℝ) (h1 : 30 = D / T) 
  (h2 : 2 * D / 3 = 30 * (T / 3)) :
  (D / 3) / (2 * T / 3) = 15 :=
by
  sorry

end NUMINAMATH_GPT_required_speed_l212_21223


namespace NUMINAMATH_GPT_arithmetic_seq_a6_l212_21294

open Real

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, (a (n + m) = a n + a m - a 0)

-- Given conditions
def condition_1 (a : ℕ → ℝ) : Prop :=
  a 2 = 4

def condition_2 (a : ℕ → ℝ) : Prop :=
  a 4 = 2

-- Mathematical statement
theorem arithmetic_seq_a6 
  (a : ℕ → ℝ)
  (h_seq: arithmetic_sequence a)
  (h_cond1 : condition_1 a)
  (h_cond2 : condition_2 a) : 
  a 6 = 0 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_a6_l212_21294


namespace NUMINAMATH_GPT_scale_reading_l212_21290

theorem scale_reading (a b c : ℝ) (h₁ : 10.15 < a ∧ a < 10.4) (h₂ : 10.275 = (10.15 + 10.4) / 2) : a = 10.3 := 
by 
  sorry

end NUMINAMATH_GPT_scale_reading_l212_21290


namespace NUMINAMATH_GPT_similar_triangles_perimeter_l212_21258

open Real

-- Defining the similar triangles and their associated conditions
noncomputable def triangle1 := (4, 6, 8)
noncomputable def side2 := 2

-- Define the possible perimeters of the other triangle
theorem similar_triangles_perimeter (h : True) :
  (∃ x, x = 4.5 ∨ x = 6 ∨ x = 9) :=
sorry

end NUMINAMATH_GPT_similar_triangles_perimeter_l212_21258


namespace NUMINAMATH_GPT_value_of_expression_l212_21237

theorem value_of_expression (x y : ℝ) (h1 : 4 * x + y = 20) (h2 : x + 4 * y = 16) : 
  17 * x ^ 2 + 20 * x * y + 17 * y ^ 2 = 656 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l212_21237


namespace NUMINAMATH_GPT_total_candies_l212_21209

-- Condition definitions
def lindaCandies : ℕ := 34
def chloeCandies : ℕ := 28

-- Proof statement to show their total candies
theorem total_candies : lindaCandies + chloeCandies = 62 := 
by
  sorry

end NUMINAMATH_GPT_total_candies_l212_21209


namespace NUMINAMATH_GPT_lucille_house_difference_l212_21234

def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

def average_height (h1 h2 h3 : ℕ) : ℕ := (h1 + h2 + h3) / 3

def difference (h_average h_actual : ℕ) : ℕ := h_average - h_actual

theorem lucille_house_difference :
  difference (average_height height_lucille height_neighbor1 height_neighbor2) height_lucille = 3 :=
by
  unfold difference
  unfold average_height
  sorry

end NUMINAMATH_GPT_lucille_house_difference_l212_21234


namespace NUMINAMATH_GPT_log_comparison_l212_21241

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4
noncomputable def log6 (x : ℝ) : ℝ := Real.log x / Real.log 6

theorem log_comparison :
  let a := log2 6
  let b := log4 12
  let c := log6 18
  a > b ∧ b > c :=
by 
  sorry

end NUMINAMATH_GPT_log_comparison_l212_21241


namespace NUMINAMATH_GPT_jimin_class_students_l212_21206

theorem jimin_class_students 
    (total_distance : ℝ)
    (interval_distance : ℝ)
    (h1 : total_distance = 242)
    (h2 : interval_distance = 5.5) :
    (total_distance / interval_distance) + 1 = 45 :=
by sorry

end NUMINAMATH_GPT_jimin_class_students_l212_21206


namespace NUMINAMATH_GPT_ben_is_10_l212_21249

-- Define the ages of the cousins
def ages : List ℕ := [6, 8, 10, 12, 14]

-- Define the conditions
def wentToPark (x y : ℕ) : Prop := x + y = 18
def wentToLibrary (x y : ℕ) : Prop := x + y < 20
def stayedHome (ben young : ℕ) : Prop := young = 6 ∧ ben ∈ ages ∧ ben ≠ 6 ∧ ben ≠ 12

-- The main theorem stating Ben's age
theorem ben_is_10 : ∃ ben, stayedHome ben 6 ∧ 
  (∃ x y, wentToPark x y ∧ x ∈ ages ∧ y ∈ ages ∧ x ≠ y ∧ x ≠ ben ∧ y ≠ ben) ∧
  (∃ x y, wentToLibrary x y ∧ x ∈ ages ∧ y ∈ ages ∧ x ≠ y ∧ x ≠ ben ∧ y ≠ ben) :=
by
  use 10
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_ben_is_10_l212_21249


namespace NUMINAMATH_GPT_ammeter_sum_l212_21261

variable (A1 A2 A3 A4 A5 : ℝ)
variable (I2 : ℝ)
variable (h1 : I2 = 4)
variable (h2 : A1 = I2)
variable (h3 : A3 = 2 * A1)
variable (h4 : A5 = A3 + A1)
variable (h5 : A4 = (5 / 3) * A5)

theorem ammeter_sum (A1 A2 A3 A4 A5 I2 : ℝ) (h1 : I2 = 4) (h2 : A1 = I2) (h3 : A3 = 2 * A1)
                   (h4 : A5 = A3 + A1) (h5 : A4 = (5 / 3) * A5) :
  A1 + I2 + A3 + A4 + A5 = 48 := 
sorry

end NUMINAMATH_GPT_ammeter_sum_l212_21261


namespace NUMINAMATH_GPT_arithmetic_geometric_seq_l212_21287

theorem arithmetic_geometric_seq (a : ℕ → ℝ) (d a_1 : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_nonzero : d ≠ 0) (h_geom : (a 0, a 1, a 4) = (a_1, a_1 + d, a_1 + 4 * d) ∧ (a 1)^2 = a 0 * a 4)
  (h_sum : a 0 + a 1 + a 4 > 13) : a_1 > 1 :=
by sorry

end NUMINAMATH_GPT_arithmetic_geometric_seq_l212_21287


namespace NUMINAMATH_GPT_eggs_per_snake_l212_21265

-- Define the conditions
def num_snakes : ℕ := 3
def price_regular : ℕ := 250
def price_super_rare : ℕ := 1000
def total_revenue : ℕ := 2250

-- Prove for the number of eggs each snake lays
theorem eggs_per_snake (E : ℕ) 
  (h1 : E * (num_snakes - 1) * price_regular + E * price_super_rare = total_revenue) : 
  E = 2 :=
sorry

end NUMINAMATH_GPT_eggs_per_snake_l212_21265


namespace NUMINAMATH_GPT_transform_roots_to_quadratic_l212_21263

noncomputable def quadratic_formula (p q : ℝ) (x : ℝ) : ℝ :=
  x^2 + p * x + q

theorem transform_roots_to_quadratic (x₁ x₂ y₁ y₂ p q : ℝ)
  (h₁ : quadratic_formula p q x₁ = 0)
  (h₂ : quadratic_formula p q x₂ = 0)
  (h₃ : x₁ ≠ 1)
  (h₄ : x₂ ≠ 1)
  (hy₁ : y₁ = (x₁ + 1) / (x₁ - 1))
  (hy₂ : y₂ = (x₂ + 1) / (x₂ - 1)) :
  (1 + p + q) * y₁^2 + 2 * (1 - q) * y₁ + (1 - p + q) = 0 ∧
  (1 + p + q) * y₂^2 + 2 * (1 - q) * y₂ + (1 - p + q) = 0 := 
sorry

end NUMINAMATH_GPT_transform_roots_to_quadratic_l212_21263


namespace NUMINAMATH_GPT_quadratic_y_axis_intersection_l212_21230

theorem quadratic_y_axis_intersection :
  (∃ y, (y = (0 - 1) ^ 2 + 2) ∧ (0, y) = (0, 3)) :=
sorry

end NUMINAMATH_GPT_quadratic_y_axis_intersection_l212_21230


namespace NUMINAMATH_GPT_series_proof_l212_21273

theorem series_proof (a b : ℝ) (h : (∑' n : ℕ, (-1)^n * a / b^(n+1)) = 6) : 
  (∑' n : ℕ, (-1)^n * a / (a - b)^(n+1)) = 6 / 7 := 
sorry

end NUMINAMATH_GPT_series_proof_l212_21273


namespace NUMINAMATH_GPT_length_of_AD_l212_21235

theorem length_of_AD (AB BC AC AD DC : ℝ)
    (h1 : AB = BC)
    (h2 : AD = 2 * DC)
    (h3 : AC = AD + DC)
    (h4 : AC = 27) : AD = 18 := 
by
  sorry

end NUMINAMATH_GPT_length_of_AD_l212_21235


namespace NUMINAMATH_GPT_midpoint_in_polar_coordinates_l212_21236

-- Define the problem as a theorem in Lean 4
theorem midpoint_in_polar_coordinates :
  let A := (10, Real.pi / 4)
  let B := (10, 3 * Real.pi / 4)
  ∃ r θ, (r = 5 * Real.sqrt 2) ∧ (θ = Real.pi / 2) ∧
         0 ≤ θ ∧ θ < 2 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_midpoint_in_polar_coordinates_l212_21236


namespace NUMINAMATH_GPT_min_value_expression_l212_21240

open Real

theorem min_value_expression (x y z: ℝ) (h1: 0 < x) (h2: 0 < y) (h3: 0 < z)
    (h4: (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10):
    (x / y + y / z + z / x) * (y / x + z / y + x / z) = 25 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l212_21240


namespace NUMINAMATH_GPT_trig_identity_proof_l212_21243

noncomputable def check_trig_identities (α β : ℝ) : Prop :=
  3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2

theorem trig_identity_proof (α β : ℝ) (h : check_trig_identities α β) :
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l212_21243


namespace NUMINAMATH_GPT_min_ω_value_l212_21297

def min_ω (ω : Real) : Prop :=
  ω > 0 ∧ (∃ k : Int, ω = 2 * k + 2 / 3)

theorem min_ω_value : ∃ ω : Real, min_ω ω ∧ ω = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_min_ω_value_l212_21297


namespace NUMINAMATH_GPT_objective_function_range_l212_21244

theorem objective_function_range:
  (∃ x y : ℝ, x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1) ∧
  (∀ x y : ℝ, (x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ 1) →
  (3*x + y ≥ (19:ℝ) / 9 ∧ 3*x + y ≤ 6)) :=
sorry

-- We have defined the conditions, the objective function, and the assertion in Lean 4.

end NUMINAMATH_GPT_objective_function_range_l212_21244


namespace NUMINAMATH_GPT_rightmost_three_digits_seven_pow_1983_add_123_l212_21270

theorem rightmost_three_digits_seven_pow_1983_add_123 :
  (7 ^ 1983 + 123) % 1000 = 466 := 
by 
  -- Proof steps are omitted
  sorry 

end NUMINAMATH_GPT_rightmost_three_digits_seven_pow_1983_add_123_l212_21270


namespace NUMINAMATH_GPT_boxes_in_carton_of_pencils_l212_21271

def cost_per_box_pencil : ℕ := 2
def cost_per_box_marker : ℕ := 4
def boxes_per_carton_marker : ℕ := 5
def cartons_of_pencils : ℕ := 20
def cartons_of_markers : ℕ := 10
def total_spent : ℕ := 600

theorem boxes_in_carton_of_pencils : ∃ x : ℕ, 20 * (2 * x) + 10 * (5 * 4) = 600 :=
by
  sorry

end NUMINAMATH_GPT_boxes_in_carton_of_pencils_l212_21271


namespace NUMINAMATH_GPT_lemonade_percentage_l212_21283

theorem lemonade_percentage (V : ℝ) (L : ℝ) :
  (0.80 * 0.40 * V + (100 - L) / 100 * 0.60 * V = 0.65 * V) →
  L = 99.45 :=
by
  intro h
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_lemonade_percentage_l212_21283


namespace NUMINAMATH_GPT_div_val_is_2_l212_21218

theorem div_val_is_2 (x : ℤ) (h : 5 * x = 100) : x / 10 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_div_val_is_2_l212_21218


namespace NUMINAMATH_GPT_pool_filling_time_l212_21225

noncomputable def fill_pool_time (hose_rate : ℕ) (cost_per_10_gallons : ℚ) (total_cost : ℚ) : ℚ :=
  let cost_per_gallon := cost_per_10_gallons / 10
  let total_gallons := total_cost / cost_per_gallon
  total_gallons / hose_rate

theorem pool_filling_time :
  fill_pool_time 100 (1 / 100) 5 = 50 := 
by
  sorry

end NUMINAMATH_GPT_pool_filling_time_l212_21225


namespace NUMINAMATH_GPT_solution_set_of_f_x_gt_2_minimum_value_of_f_l212_21231

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem solution_set_of_f_x_gt_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -7} ∪ {x : ℝ | x > 5 / 3} :=
by 
  sorry

theorem minimum_value_of_f : ∃ x : ℝ, f x = -9 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_f_x_gt_2_minimum_value_of_f_l212_21231


namespace NUMINAMATH_GPT_toms_age_l212_21269

variable (T J : ℕ)

theorem toms_age :
  (J - 6 = 3 * (T - 6)) ∧ (J + 4 = 2 * (T + 4)) → T = 16 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_toms_age_l212_21269


namespace NUMINAMATH_GPT_fraction_arithmetic_l212_21260

theorem fraction_arithmetic : 
  (2 / 5 + 3 / 7) / (4 / 9 * 1 / 8) = 522 / 35 := by
  sorry

end NUMINAMATH_GPT_fraction_arithmetic_l212_21260


namespace NUMINAMATH_GPT_roots_expression_eval_l212_21285

theorem roots_expression_eval (p q r : ℝ) 
  (h1 : p + q + r = 2)
  (h2 : p * q + q * r + r * p = -1)
  (h3 : p * q * r = -2)
  (hp : p^3 - 2 * p^2 - p + 2 = 0)
  (hq : q^3 - 2 * q^2 - q + 2 = 0)
  (hr : r^3 - 2 * r^2 - r + 2 = 0) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = 16 :=
sorry

end NUMINAMATH_GPT_roots_expression_eval_l212_21285


namespace NUMINAMATH_GPT_johns_salary_percentage_increase_l212_21224

theorem johns_salary_percentage_increase (initial_salary final_salary : ℕ) (h1 : initial_salary = 50) (h2 : final_salary = 90) :
  ((final_salary - initial_salary : ℕ) / initial_salary : ℚ) * 100 = 80 := by
  sorry

end NUMINAMATH_GPT_johns_salary_percentage_increase_l212_21224


namespace NUMINAMATH_GPT_average_of_four_digits_l212_21256

theorem average_of_four_digits (sum9 : ℤ) (avg9 : ℤ) (avg5 : ℤ) (sum4 : ℤ) (n : ℤ) :
  avg9 = 18 →
  n = 9 →
  sum9 = avg9 * n →
  avg5 = 26 →
  sum4 = sum9 - (avg5 * 5) →
  avg4 = sum4 / 4 →
  avg4 = 8 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_average_of_four_digits_l212_21256


namespace NUMINAMATH_GPT_time_b_started_walking_l212_21259

/-- A's speed is 7 kmph, B's speed is 7.555555555555555 kmph, and B overtakes A after 1.8 hours. -/
theorem time_b_started_walking (t : ℝ) (A_speed : ℝ) (B_speed : ℝ) (overtake_time : ℝ)
    (hA : A_speed = 7) (hB : B_speed = 7.555555555555555) (hOvertake : overtake_time = 1.8) 
    (distance_A : ℝ) (distance_B : ℝ)
    (hDistanceA : distance_A = (t + overtake_time) * A_speed)
    (hDistanceB : distance_B = B_speed * overtake_time) :
  t = 8.57 / 60 := by
  sorry

end NUMINAMATH_GPT_time_b_started_walking_l212_21259


namespace NUMINAMATH_GPT_Rohit_is_to_the_east_of_starting_point_l212_21266

-- Define the conditions and the problem statement.
def Rohit's_movements_proof
  (distance_south : ℕ) (distance_first_left : ℕ) (distance_second_left : ℕ) (distance_right : ℕ)
  (final_distance : ℕ) : Prop :=
  distance_south = 25 ∧
  distance_first_left = 20 ∧
  distance_second_left = 25 ∧
  distance_right = 15 ∧
  final_distance = 35 →
  (direction : String) → (distance : ℕ) →
  direction = "east" ∧ distance = final_distance

-- We can now state the theorem
theorem Rohit_is_to_the_east_of_starting_point :
  Rohit's_movements_proof 25 20 25 15 35 :=
by
  sorry

end NUMINAMATH_GPT_Rohit_is_to_the_east_of_starting_point_l212_21266


namespace NUMINAMATH_GPT_xyz_neg_l212_21208

theorem xyz_neg {a b c x y z : ℝ} 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) 
  (h : |x - a| + |y - b| + |z - c| = 0) : 
  x * y * z < 0 :=
by 
  -- to be proven
  sorry

end NUMINAMATH_GPT_xyz_neg_l212_21208


namespace NUMINAMATH_GPT_arithmetic_geometric_seq_l212_21204

variable {a_n : ℕ → ℝ}
variable {a_1 a_3 a_5 a_6 a_11 : ℝ}

theorem arithmetic_geometric_seq (h₁ : a_1 * a_5 + 2 * a_3 * a_6 + a_1 * a_11 = 16) 
                                  (h₂ : a_1 * a_5 = a_3^2) 
                                  (h₃ : a_1 * a_11 = a_6^2) 
                                  (h₄ : a_3 > 0)
                                  (h₅ : a_6 > 0) : 
    a_3 + a_6 = 4 := 
by {
    sorry
}

end NUMINAMATH_GPT_arithmetic_geometric_seq_l212_21204


namespace NUMINAMATH_GPT_odd_lattice_points_on_BC_l212_21207

theorem odd_lattice_points_on_BC
  (A B C : ℤ × ℤ)
  (odd_lattice_points_AB : Odd ((B.1 - A.1) * (B.2 - A.2)))
  (odd_lattice_points_AC : Odd ((C.1 - A.1) * (C.2 - A.2))) :
  Odd ((C.1 - B.1) * (C.2 - B.2)) :=
sorry

end NUMINAMATH_GPT_odd_lattice_points_on_BC_l212_21207


namespace NUMINAMATH_GPT_correct_operation_B_l212_21282

theorem correct_operation_B (a b : ℝ) : - (a - b) = -a + b := 
by sorry

end NUMINAMATH_GPT_correct_operation_B_l212_21282


namespace NUMINAMATH_GPT_cylindrical_pipe_height_l212_21276

theorem cylindrical_pipe_height (r_outer r_inner : ℝ) (SA : ℝ) (h : ℝ) 
  (h_outer : r_outer = 5)
  (h_inner : r_inner = 3)
  (h_SA : SA = 50 * Real.pi)
  (surface_area_eq: SA = 2 * Real.pi * (r_outer + r_inner) * h) : 
  h = 25 / 8 := 
by
  {
    sorry
  }

end NUMINAMATH_GPT_cylindrical_pipe_height_l212_21276


namespace NUMINAMATH_GPT_sum_c_d_eq_neg11_l212_21292

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := (x + 6) / (x^2 + c * x + d)

theorem sum_c_d_eq_neg11 (c d : ℝ) 
    (h₀ : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = 3 ∨ x = -4)) :
    c + d = -11 := 
sorry

end NUMINAMATH_GPT_sum_c_d_eq_neg11_l212_21292


namespace NUMINAMATH_GPT_largest_number_in_ratio_l212_21254

theorem largest_number_in_ratio (x : ℕ) (h : ((4 * x + 5 * x + 6 * x) / 3 : ℝ) = 20) : 6 * x = 24 := 
by 
  sorry

end NUMINAMATH_GPT_largest_number_in_ratio_l212_21254


namespace NUMINAMATH_GPT_negation_of_existential_proposition_l212_21253

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_GPT_negation_of_existential_proposition_l212_21253


namespace NUMINAMATH_GPT_maximum_of_function_l212_21221

theorem maximum_of_function :
  ∃ x y : ℝ, 
    (1/3 ≤ x ∧ x ≤ 2/5 ∧ 1/4 ≤ y ∧ y ≤ 5/12) ∧ 
    (∀ x' y' : ℝ, 1/3 ≤ x' ∧ x' ≤ 2/5 ∧ 1/4 ≤ y' ∧ y' ≤ 5/12 → 
                (xy / (x^2 + y^2) ≤ x' * y' / (x'^2 + y'^2))) ∧ 
    (xy / (x^2 + y^2) = 20 / 41) := 
sorry

end NUMINAMATH_GPT_maximum_of_function_l212_21221


namespace NUMINAMATH_GPT_jeremy_school_distance_l212_21277

theorem jeremy_school_distance (d : ℝ) (v : ℝ) :
  (d = v * 0.5) ∧
  (d = (v + 15) * 0.3) ∧
  (d = (v - 10) * (2 / 3)) →
  d = 15 :=
by 
  sorry

end NUMINAMATH_GPT_jeremy_school_distance_l212_21277


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l212_21280

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l212_21280


namespace NUMINAMATH_GPT_star_5_3_eq_31_l212_21264

def star (a b : ℤ) : ℤ := a^2 + a * b - b^2

theorem star_5_3_eq_31 : star 5 3 = 31 :=
by
  sorry

end NUMINAMATH_GPT_star_5_3_eq_31_l212_21264


namespace NUMINAMATH_GPT_find_third_polygon_sides_l212_21291

def interior_angle (n : ℕ) : ℚ :=
  (n - 2) * 180 / n

theorem find_third_polygon_sides :
  let square_angle := interior_angle 4
  let pentagon_angle := interior_angle 5
  let third_polygon_angle := 360 - (square_angle + pentagon_angle)
  ∃ (m : ℕ), interior_angle m = third_polygon_angle ∧ m = 20 :=
by
  let square_angle := interior_angle 4
  let pentagon_angle := interior_angle 5
  let third_polygon_angle := 360 - (square_angle + pentagon_angle)
  use 20
  sorry

end NUMINAMATH_GPT_find_third_polygon_sides_l212_21291


namespace NUMINAMATH_GPT_total_baseball_cards_l212_21200

/-- 
Given that you have 5 friends and each friend gets 91 baseball cards, 
prove that the total number of baseball cards you have is 455.
-/
def baseball_cards (f c : Nat) (t : Nat) : Prop :=
  (t = f * c)

theorem total_baseball_cards:
  ∀ (f c t : Nat), f = 5 → c = 91 → t = 455 → baseball_cards f c t :=
by
  intros f c t hf hc ht
  sorry

end NUMINAMATH_GPT_total_baseball_cards_l212_21200


namespace NUMINAMATH_GPT_jasmine_coffee_beans_purchase_l212_21250

theorem jasmine_coffee_beans_purchase (x : ℝ) (coffee_cost per_pound milk_cost per_gallon total_cost : ℝ)
  (h1 : coffee_cost = 2.50)
  (h2 : milk_cost = 3.50)
  (h3 : total_cost = 17)
  (h4 : milk_purchased = 2)
  (h_equation : coffee_cost * x + milk_cost * milk_purchased = total_cost) :
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_jasmine_coffee_beans_purchase_l212_21250


namespace NUMINAMATH_GPT_total_ticket_sales_l212_21296

-- Define the parameters and the theorem to be proven.
theorem total_ticket_sales (total_people : ℕ) (kids : ℕ) (adult_ticket_price : ℕ) (kid_ticket_price : ℕ) 
  (adult_tickets := total_people - kids) 
  (adult_ticket_sales := adult_tickets * adult_ticket_price) 
  (kid_ticket_sales := kids * kid_ticket_price) : 
  total_people = 254 → kids = 203 → adult_ticket_price = 28 → kid_ticket_price = 12 → 
  adult_ticket_sales + kid_ticket_sales = 3864 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_ticket_sales_l212_21296


namespace NUMINAMATH_GPT_grain_demand_l212_21272

variable (F : ℝ)
def S0 : ℝ := 1800000 -- base supply value

theorem grain_demand : ∃ D : ℝ, S = 0.75 * D ∧ S = S0 * (1 + F) ∧ D = (1800000 * (1 + F) / 0.75) :=
by
  sorry

end NUMINAMATH_GPT_grain_demand_l212_21272


namespace NUMINAMATH_GPT_number_of_distinct_values_l212_21203

theorem number_of_distinct_values (n : ℕ) (mode_count : ℕ) (second_count : ℕ) (total_count : ℕ) 
    (h1 : n = 3000) (h2 : mode_count = 15) (h3 : second_count = 14) : 
    (n - mode_count - second_count) / 13 + 2 ≥ 232 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_distinct_values_l212_21203


namespace NUMINAMATH_GPT_factorization_of_w4_minus_81_l212_21214

theorem factorization_of_w4_minus_81 (w : ℝ) : 
  (w^4 - 81) = (w - 3) * (w + 3) * (w^2 + 9) :=
by sorry

end NUMINAMATH_GPT_factorization_of_w4_minus_81_l212_21214


namespace NUMINAMATH_GPT_cucumber_to_tomato_ratio_l212_21247

variable (total_rows : ℕ) (space_per_row_tomato : ℕ) (tomatoes_per_plant : ℕ) (total_tomatoes : ℕ)

/-- Aubrey's Garden -/
theorem cucumber_to_tomato_ratio (total_rows_eq : total_rows = 15)
  (space_per_row_tomato_eq : space_per_row_tomato = 8)
  (tomatoes_per_plant_eq : tomatoes_per_plant = 3)
  (total_tomatoes_eq : total_tomatoes = 120) :
  let total_tomato_plants := total_tomatoes / tomatoes_per_plant
  let rows_tomato := total_tomato_plants / space_per_row_tomato
  let rows_cucumber := total_rows - rows_tomato
  (2 * rows_tomato = rows_cucumber)
:=
by
  sorry

end NUMINAMATH_GPT_cucumber_to_tomato_ratio_l212_21247


namespace NUMINAMATH_GPT_factorize_expression_l212_21212

-- The primary goal is to prove that -2xy^2 + 4xy - 2x = -2x(y - 1)^2
theorem factorize_expression (x y : ℝ) : 
  -2 * x * y^2 + 4 * x * y - 2 * x = -2 * x * (y - 1)^2 := 
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l212_21212


namespace NUMINAMATH_GPT_min_value_a_plus_b_l212_21229

open Real

theorem min_value_a_plus_b (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h : 1 / a + 2 / b = 1) :
  a + b = 3 + 2 * sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_a_plus_b_l212_21229


namespace NUMINAMATH_GPT_group_weight_problem_l212_21238

theorem group_weight_problem (n : ℕ) (avg_weight_increase : ℕ) (weight_diff : ℕ) (total_weight_increase : ℕ) 
  (h1 : avg_weight_increase = 3) (h2 : weight_diff = 75 - 45) (h3 : total_weight_increase = avg_weight_increase * n)
  (h4 : total_weight_increase = weight_diff) : n = 10 := by
  sorry

end NUMINAMATH_GPT_group_weight_problem_l212_21238


namespace NUMINAMATH_GPT_egg_whites_per_cake_l212_21202

-- Define the conversion ratio between tablespoons of aquafaba and egg whites
def tablespoons_per_egg_white : ℕ := 2

-- Define the total amount of aquafaba used for two cakes
def total_tablespoons_for_two_cakes : ℕ := 32

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Prove the number of egg whites needed per cake
theorem egg_whites_per_cake :
  (total_tablespoons_for_two_cakes / tablespoons_per_egg_white) / number_of_cakes = 8 := by
  sorry

end NUMINAMATH_GPT_egg_whites_per_cake_l212_21202


namespace NUMINAMATH_GPT_marble_count_l212_21286

theorem marble_count (p y v : ℝ) (h1 : y + v = 10) (h2 : p + v = 12) (h3 : p + y = 5) :
  p + y + v = 13.5 :=
sorry

end NUMINAMATH_GPT_marble_count_l212_21286


namespace NUMINAMATH_GPT_Ron_eats_24_pickle_slices_l212_21201

theorem Ron_eats_24_pickle_slices : 
  ∀ (pickle_slices_Sammy Tammy Ron : ℕ), 
    pickle_slices_Sammy = 15 → 
    Tammy = 2 * pickle_slices_Sammy → 
    Ron = Tammy - (20 * Tammy / 100) → 
    Ron = 24 := by
  intros pickle_slices_Sammy Tammy Ron h_sammy h_tammy h_ron
  sorry

end NUMINAMATH_GPT_Ron_eats_24_pickle_slices_l212_21201


namespace NUMINAMATH_GPT_train_length_72kmphr_9sec_180m_l212_21239

/-- Given speed in km/hr and time in seconds, calculate the length of the train in meters -/
theorem train_length_72kmphr_9sec_180m : ∀ (speed_kmph : ℕ) (time_sec : ℕ),
  speed_kmph = 72 → time_sec = 9 → 
  (speed_kmph * 1000 / 3600) * time_sec = 180 :=
by
  intros speed_kmph time_sec h1 h2
  sorry

end NUMINAMATH_GPT_train_length_72kmphr_9sec_180m_l212_21239


namespace NUMINAMATH_GPT_value_of_z_l212_21227

theorem value_of_z {x y z : ℤ} (h1 : x = 2) (h2 : y = x^2 - 5) (h3 : z = y^2 - 5) : z = -4 := by
  sorry

end NUMINAMATH_GPT_value_of_z_l212_21227


namespace NUMINAMATH_GPT_total_possible_arrangements_l212_21275

-- Define the subjects
inductive Subject : Type
| PoliticalScience
| Chinese
| Mathematics
| English
| PhysicalEducation
| Physics

open Subject

-- Define the condition that the first period cannot be Chinese
def first_period_cannot_be_chinese (schedule : Fin 6 → Subject) : Prop :=
  schedule 0 ≠ Chinese

-- Define the condition that the fifth period cannot be English
def fifth_period_cannot_be_english (schedule : Fin 6 → Subject) : Prop :=
  schedule 4 ≠ English

-- Define the schedule includes six unique subjects
def schedule_includes_all_subjects (schedule : Fin 6 → Subject) : Prop :=
  ∀ s : Subject, ∃ i : Fin 6, schedule i = s

-- Define the main theorem to prove the total number of possible arrangements
theorem total_possible_arrangements : 
  ∃ (schedules : List (Fin 6 → Subject)), 
  (∀ schedule, schedule ∈ schedules → 
    first_period_cannot_be_chinese schedule ∧ 
    fifth_period_cannot_be_english schedule ∧ 
    schedule_includes_all_subjects schedule) ∧ 
  schedules.length = 600 :=
sorry

end NUMINAMATH_GPT_total_possible_arrangements_l212_21275


namespace NUMINAMATH_GPT_four_digit_arithmetic_sequence_l212_21252

theorem four_digit_arithmetic_sequence :
  ∃ (a b c d : ℕ), 1000 * a + 100 * b + 10 * c + d = 5555 ∨ 1000 * a + 100 * b + 10 * c + d = 2468 ∧
  (a + d = 10) ∧ (b + c = 10) ∧ (2 * b = a + c) ∧ (c - b = b - a) ∧ (d - c = c - b) ∧
  (1000 * d + 100 * c + 10 * b + a + 1000 * a + 100 * b + 10 * c + d = 11110) :=
sorry

end NUMINAMATH_GPT_four_digit_arithmetic_sequence_l212_21252


namespace NUMINAMATH_GPT_intersection_complement_l212_21251

universe u

-- Define the universal set U, and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement (U A : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The main theorem to be proved
theorem intersection_complement :
  B ∩ (complement U A) = {3, 4} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l212_21251


namespace NUMINAMATH_GPT_plane_split_four_regions_l212_21299

theorem plane_split_four_regions :
  (∀ x y : ℝ, y = 3 * x ∨ x = 3 * y) → (exists regions : ℕ, regions = 4) :=
by
  sorry

end NUMINAMATH_GPT_plane_split_four_regions_l212_21299


namespace NUMINAMATH_GPT_ray_inequality_l212_21279

theorem ray_inequality (a : ℝ) :
  (∀ x : ℝ, x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3 ≥ 0 ↔ x ≥ 1)
  ∨ (∀ x : ℝ, x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3 ≥ 0 ↔ x ≥ -1) :=
sorry

end NUMINAMATH_GPT_ray_inequality_l212_21279


namespace NUMINAMATH_GPT_cylinder_height_l212_21268

   theorem cylinder_height (r h : ℝ) (SA : ℝ) (π : ℝ) :
     r = 3 → SA = 30 * π → SA = 2 * π * r^2 + 2 * π * r * h → h = 2 :=
   by
     intros hr hSA hSA_formula
     rw [hr] at hSA_formula
     rw [hSA] at hSA_formula
     sorry
   
end NUMINAMATH_GPT_cylinder_height_l212_21268


namespace NUMINAMATH_GPT_highest_percentage_without_car_l212_21211

noncomputable def percentage_without_car (total_percentage : ℝ) (car_percentage : ℝ) : ℝ :=
  total_percentage - total_percentage * car_percentage / 100

theorem highest_percentage_without_car :
  let A_total := 30
  let A_with_car := 25
  let B_total := 50
  let B_with_car := 15
  let C_total := 20
  let C_with_car := 35

  percentage_without_car A_total A_with_car = 22.5 /\
  percentage_without_car B_total B_with_car = 42.5 /\
  percentage_without_car C_total C_with_car = 13 /\
  percentage_without_car B_total B_with_car = max (percentage_without_car A_total A_with_car) (max (percentage_without_car B_total B_with_car) (percentage_without_car C_total C_with_car)) :=
by
  sorry

end NUMINAMATH_GPT_highest_percentage_without_car_l212_21211


namespace NUMINAMATH_GPT_ruby_shares_with_9_friends_l212_21288

theorem ruby_shares_with_9_friends
    (total_candies : ℕ) (candies_per_friend : ℕ)
    (h1 : total_candies = 36) (h2 : candies_per_friend = 4) :
    total_candies / candies_per_friend = 9 := by
  sorry

end NUMINAMATH_GPT_ruby_shares_with_9_friends_l212_21288


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l212_21248

theorem number_of_terms_in_arithmetic_sequence : 
  ∀ (a d l : ℕ), a = 20 → d = 5 → l = 150 → 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 27 :=
by
  intros a d l ha hd hl
  use 27
  rw [ha, hd, hl]
  sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l212_21248


namespace NUMINAMATH_GPT_prove_value_of_expression_l212_21262

theorem prove_value_of_expression (x : ℝ) (h : 10000 * x + 2 = 4) : 5000 * x + 1 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_prove_value_of_expression_l212_21262


namespace NUMINAMATH_GPT_find_number_l212_21215

theorem find_number (x : ℝ) (h : 0.80 * 40 = (4/5) * x + 16) : x = 20 :=
by sorry

end NUMINAMATH_GPT_find_number_l212_21215


namespace NUMINAMATH_GPT_right_triangle_angle_ratio_l212_21257

theorem right_triangle_angle_ratio
  (a b : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) 
  (h : a / b = 5 / 4)
  (h3 : a + b = 90) :
  (a = 50) ∧ (b = 40) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_angle_ratio_l212_21257


namespace NUMINAMATH_GPT_find_abc_sum_l212_21210

noncomputable def x := Real.sqrt ((Real.sqrt 105) / 2 + 7 / 2)

theorem find_abc_sum :
  ∃ (a b c : ℕ), a + b + c = 5824 ∧
  x ^ 100 = 3 * x ^ 98 + 15 * x ^ 96 + 12 * x ^ 94 - x ^ 50 + a * x ^ 46 + b * x ^ 44 + c * x ^ 40 :=
  sorry

end NUMINAMATH_GPT_find_abc_sum_l212_21210


namespace NUMINAMATH_GPT_cost_per_meal_is_8_l212_21289

-- Define the conditions
def number_of_adults := 2
def number_of_children := 5
def total_bill := 56
def total_people := number_of_adults + number_of_children

-- Define the cost per meal
def cost_per_meal := total_bill / total_people

-- State the theorem we want to prove
theorem cost_per_meal_is_8 : cost_per_meal = 8 := 
by
  -- The proof would go here, but we'll use sorry to skip it
  sorry

end NUMINAMATH_GPT_cost_per_meal_is_8_l212_21289
