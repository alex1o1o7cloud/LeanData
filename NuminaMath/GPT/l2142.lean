import Mathlib

namespace quadratic_factor_transformation_l2142_214271

theorem quadratic_factor_transformation (x : ℝ) :
  x^2 - 6 * x + 5 = 0 → (x - 3)^2 = 14 := 
by
  sorry

end quadratic_factor_transformation_l2142_214271


namespace how_many_one_halves_in_two_sevenths_l2142_214290

theorem how_many_one_halves_in_two_sevenths : (2 / 7) / (1 / 2) = 4 / 7 := by 
  sorry

end how_many_one_halves_in_two_sevenths_l2142_214290


namespace floor_area_cannot_exceed_10_square_meters_l2142_214222

theorem floor_area_cannot_exceed_10_square_meters
  (a b : ℝ)
  (h : 3 > 0)
  (floor_lt_wall1 : a * b < 3 * a)
  (floor_lt_wall2 : a * b < 3 * b) :
  a * b ≤ 9 :=
by
  -- This is where the proof would go
  sorry

end floor_area_cannot_exceed_10_square_meters_l2142_214222


namespace barbara_spent_on_other_goods_l2142_214260

theorem barbara_spent_on_other_goods
  (cost_tuna : ℝ := 5 * 2)
  (cost_water : ℝ := 4 * 1.5)
  (total_paid : ℝ := 56) :
  total_paid - (cost_tuna + cost_water) = 40 := by
  sorry

end barbara_spent_on_other_goods_l2142_214260


namespace expression_undefined_count_l2142_214298

theorem expression_undefined_count (x : ℝ) :
  ∃! x, (x - 1) * (x + 3) * (x - 3) = 0 :=
sorry

end expression_undefined_count_l2142_214298


namespace find_x_l2142_214258

theorem find_x (a b x: ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0)
    (h4 : (4 * a)^(4 * b) = a^b * x^(2 * b)) : x = 16 * a^(3 / 2) := by
  sorry

end find_x_l2142_214258


namespace find_d_l2142_214211

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
∀ n, a n = a₁ + d * (n - 1)

theorem find_d
  (a : ℕ → ℝ)
  (a₁ d : ℝ)
  (h_arith : arithmetic_sequence a a₁ d)
  (h₁ : a₁ = 1)
  (h_geom_mean : a 2 ^ 2 = a 1 * a 4)
  (h_d_neq_zero : d ≠ 0):
  d = 1 :=
sorry

end find_d_l2142_214211


namespace sum_first_3k_plus_2_terms_l2142_214264

variable (k : ℕ)

def first_term : ℕ := k^2 + 1

def sum_of_sequence (n : ℕ) : ℕ :=
  let a₁ := first_term k
  let aₙ := a₁ + (n - 1)
  n * (a₁ + aₙ) / 2

theorem sum_first_3k_plus_2_terms :
  sum_of_sequence k (3 * k + 2) = 3 * k^3 + 8 * k^2 + 6 * k + 3 :=
by
  -- Here we define the sequence and compute the sum
  sorry

end sum_first_3k_plus_2_terms_l2142_214264


namespace no_unfenced_area_l2142_214208

noncomputable def area : ℝ := 5000
noncomputable def cost_per_foot : ℝ := 30
noncomputable def budget : ℝ := 120000

theorem no_unfenced_area (area : ℝ) (cost_per_foot : ℝ) (budget : ℝ) :
  (budget / cost_per_foot) >= 4 * (Real.sqrt (area)) → 0 = 0 :=
by
  intro h
  sorry

end no_unfenced_area_l2142_214208


namespace flower_beds_fraction_l2142_214236

-- Definitions based on given conditions
def yard_length := 30
def yard_width := 6
def trapezoid_parallel_side1 := 20
def trapezoid_parallel_side2 := 30
def flower_bed_leg := (trapezoid_parallel_side2 - trapezoid_parallel_side1) / 2
def flower_bed_area := (1 / 2) * flower_bed_leg ^ 2
def total_flower_bed_area := 2 * flower_bed_area
def yard_area := yard_length * yard_width
def occupied_fraction := total_flower_bed_area / yard_area

-- Statement to prove
theorem flower_beds_fraction :
  occupied_fraction = 5 / 36 :=
by
  -- sorries to skip the proofs
  sorry

end flower_beds_fraction_l2142_214236


namespace trajectory_of_T_l2142_214200

-- Define coordinates for points A, T, and M
variables {x x0 y y0 : ℝ}
def A (x0: ℝ) (y0: ℝ) := (x0, y0)
def T (x: ℝ) (y: ℝ) := (x, y)
def M : ℝ × ℝ := (-2, 0)

-- Conditions
def curve (x : ℝ) (y : ℝ) := 4 * x^2 - y + 1 = 0
def vector_condition (x x0 y y0 : ℝ) := (x - x0, y - y0) = 2 * (-2 - x, -y)

theorem trajectory_of_T (x y x0 y0 : ℝ) (hA : curve x0 y0) (hV : vector_condition x x0 y y0) :
  4 * (3 * x + 4)^2 - 3 * y + 1 = 0 :=
by
  sorry

end trajectory_of_T_l2142_214200


namespace find_x_l2142_214249

theorem find_x 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (dot_product : ℝ)
  (ha : a = (1, 2)) 
  (hb : b = (x, 3)) 
  (hdot : a.1 * b.1 + a.2 * b.2 = dot_product) 
  (hdot_val : dot_product = 4) : 
  x = -2 :=
by 
  sorry

end find_x_l2142_214249


namespace solve_equation_l2142_214220

theorem solve_equation (x : ℝ) (h : x > 0) :
  25^(Real.log x / Real.log 4) - 5^(Real.log (x^2) / Real.log 16 + 1) = Real.log (9 * Real.sqrt 3) / Real.log (Real.sqrt 3) - 25^(Real.log x / Real.log 16) ->
  x = 4 :=
by
  sorry

end solve_equation_l2142_214220


namespace angle_between_slant_height_and_base_l2142_214265

theorem angle_between_slant_height_and_base (R : ℝ) (diam_base_upper diam_base_lower : ℝ) 
(h1 : diam_base_upper + diam_base_lower = 5 * R)
: ∃ θ : ℝ, θ = Real.arcsin (4 / 5) := 
sorry

end angle_between_slant_height_and_base_l2142_214265


namespace compute_xy_l2142_214281

variable (x y : ℝ)

-- Conditions from the problem
def condition1 : Prop := x + y = 10
def condition2 : Prop := x^3 + y^3 = 172

-- Theorem statement to prove the answer
theorem compute_xy (h1 : condition1 x y) (h2 : condition2 x y) : x * y = 41.4 :=
sorry

end compute_xy_l2142_214281


namespace pie_not_crust_percentage_l2142_214268

theorem pie_not_crust_percentage (total_weight crust_weight : ℝ) 
  (h1 : total_weight = 200) (h2 : crust_weight = 50) : 
  (total_weight - crust_weight) / total_weight * 100 = 75 :=
by
  sorry

end pie_not_crust_percentage_l2142_214268


namespace person_age_l2142_214218

theorem person_age (A : ℕ) (h : 6 * (A + 6) - 6 * (A - 6) = A) : A = 72 := 
by
  sorry

end person_age_l2142_214218


namespace tenth_term_arithmetic_seq_l2142_214283

theorem tenth_term_arithmetic_seq : 
  ∀ (first_term common_diff : ℤ) (n : ℕ), 
    first_term = 10 → common_diff = -2 → n = 10 → 
    (first_term + (n - 1) * common_diff) = -8 :=
by
  sorry

end tenth_term_arithmetic_seq_l2142_214283


namespace tangent_and_normal_lines_l2142_214224

theorem tangent_and_normal_lines (x y : ℝ → ℝ) (t : ℝ) (t₀ : ℝ) 
  (h0 : t₀ = 0) 
  (h1 : ∀ t, x t = (1/2) * t^2 - (1/4) * t^4) 
  (h2 : ∀ t, y t = (1/2) * t^2 + (1/3) * t^3) :
  (∃ m : ℝ, y (x t₀) = m * (x t₀) ∧ m = 1) ∧
  (∃ n : ℝ, y (x t₀) = n * (x t₀) ∧ n = -1) :=
by 
  sorry

end tangent_and_normal_lines_l2142_214224


namespace find_original_acid_amount_l2142_214246

noncomputable def original_amount_of_acid (a w : ℝ) : Prop :=
  3 * a = w + 2 ∧ 5 * a = 3 * w - 10

theorem find_original_acid_amount (a w : ℝ) (h : original_amount_of_acid a w) : a = 4 :=
by
  sorry

end find_original_acid_amount_l2142_214246


namespace sum_of_radii_tangent_circles_l2142_214210

theorem sum_of_radii_tangent_circles :
  ∃ (r1 r2 : ℝ), 
  (∀ r, (r = (6 + 2*Real.sqrt 6) ∨ r = (6 - 2*Real.sqrt 6)) → (r = r1 ∨ r = r2)) ∧ 
  ((r1 - 4)^2 + r1^2 = (r1 + 2)^2) ∧ 
  ((r2 - 4)^2 + r2^2 = (r2 + 2)^2) ∧ 
  (r1 + r2 = 12) :=
by
  sorry

end sum_of_radii_tangent_circles_l2142_214210


namespace triangle_height_l2142_214230

theorem triangle_height (area base : ℝ) (h_area : area = 9.31) (h_base : base = 4.9) : (2 * area) / base = 3.8 :=
by
  sorry

end triangle_height_l2142_214230


namespace gino_popsicle_sticks_left_l2142_214247

-- Define the initial number of popsicle sticks Gino has
def initial_popsicle_sticks : ℝ := 63.0

-- Define the number of popsicle sticks Gino gives away
def given_away_popsicle_sticks : ℝ := 50.0

-- Expected number of popsicle sticks Gino has left
def expected_remaining_popsicle_sticks : ℝ := 13.0

-- Main theorem to be proven
theorem gino_popsicle_sticks_left :
  initial_popsicle_sticks - given_away_popsicle_sticks = expected_remaining_popsicle_sticks := 
by
  -- This is where the proof would go, but we leave it as 'sorry' for now
  sorry

end gino_popsicle_sticks_left_l2142_214247


namespace part_a_solution_l2142_214250

theorem part_a_solution (x y : ℤ) : xy + 3 * x - 5 * y = -3 ↔ 
  (x = 6 ∧ y = -21) ∨ 
  (x = -13 ∧ y = -2) ∨ 
  (x = 4 ∧ y = 15) ∨ 
  (x = 23 ∧ y = -4) ∨ 
  (x = 7 ∧ y = -12) ∨ 
  (x = -4 ∧ y = -1) ∨ 
  (x = 3 ∧ y = 6) ∨ 
  (x = 14 ∧ y = -5) ∨ 
  (x = 8 ∧ y = -9) ∨ 
  (x = -1 ∧ y = 0) ∨ 
  (x = 2 ∧ y = 3) ∨ 
  (x = 11 ∧ y = -6) := 
by sorry

end part_a_solution_l2142_214250


namespace A_inter_B_empty_iff_l2142_214204

variable (m : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem A_inter_B_empty_iff : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by
  sorry

end A_inter_B_empty_iff_l2142_214204


namespace green_balls_more_than_red_l2142_214280

theorem green_balls_more_than_red
  (total_balls : ℕ) (red_balls : ℕ) (green_balls : ℕ)
  (h1 : total_balls = 66)
  (h2 : red_balls = 30)
  (h3 : green_balls = total_balls - red_balls) : green_balls - red_balls = 6 :=
by
  sorry

end green_balls_more_than_red_l2142_214280


namespace area_Q1RQ3Q5_of_regular_hexagon_l2142_214275

noncomputable def area_quadrilateral (s : ℝ) (θ : ℝ) : ℝ := s^2 * Real.sin θ / 2

theorem area_Q1RQ3Q5_of_regular_hexagon :
  let apothem := 3
  let side_length := 6 * Real.sqrt 3
  let θ := Real.pi / 3  -- 60 degrees in radians
  area_quadrilateral (3 * Real.sqrt 3) θ = 27 * Real.sqrt 3 / 2 :=
by
  sorry

end area_Q1RQ3Q5_of_regular_hexagon_l2142_214275


namespace checkerboard_problem_l2142_214261

def is_valid_square (size : ℕ) : Prop :=
  size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10

def check_10_by_10 : ℕ :=
  24 + 36 + 25 + 16 + 9 + 4 + 1

theorem checkerboard_problem :
  ∀ size : ℕ, ( size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10 ) →
  check_10_by_10 = 115 := 
sorry

end checkerboard_problem_l2142_214261


namespace angela_age_in_fifteen_years_l2142_214256

-- Condition 1: Angela is currently 3 times as old as Beth
def angela_age_three_times_beth (A B : ℕ) := A = 3 * B

-- Condition 2: Angela is half as old as Derek
def angela_half_derek (A D : ℕ) := A = D / 2

-- Condition 3: Twenty years ago, the sum of their ages was equal to Derek's current age
def sum_ages_twenty_years_ago (A B D : ℕ) := (A - 20) + (B - 20) + (D - 20) = D

-- Condition 4: In seven years, the difference in the square root of Angela's age and one-third of Beth's age is a quarter of Derek's age
def age_diff_seven_years (A B D : ℕ) := Real.sqrt (A + 7) - (B + 7) / 3 = D / 4

-- Define the main theorem to be proven
theorem angela_age_in_fifteen_years (A B D : ℕ) 
  (h1 : angela_age_three_times_beth A B)
  (h2 : angela_half_derek A D) 
  (h3 : sum_ages_twenty_years_ago A B D) 
  (h4 : age_diff_seven_years A B D) :
  A + 15 = 60 := 
  sorry

end angela_age_in_fifteen_years_l2142_214256


namespace find_C_l2142_214277

theorem find_C (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 350) : C = 50 :=
by
  sorry

end find_C_l2142_214277


namespace find_number_ge_40_l2142_214241

theorem find_number_ge_40 (x : ℝ) : 0.90 * x > 0.80 * 30 + 12 → x > 40 :=
by sorry

end find_number_ge_40_l2142_214241


namespace min_sum_of_integers_cauchy_schwarz_l2142_214228

theorem min_sum_of_integers_cauchy_schwarz :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 
  (1 / x + 4 / y + 9 / z = 1) ∧ 
  ((x + y + z) = 36) :=
  sorry

end min_sum_of_integers_cauchy_schwarz_l2142_214228


namespace find_k_range_of_m_l2142_214292

-- Given conditions and function definition
def f (x k : ℝ) : ℝ := x^2 + (2*k-3)*x + k^2 - 7

-- Prove that k = 3 when the zeros of f(x) are -1 and -2
theorem find_k (k : ℝ) (h₁ : f (-1) k = 0) (h₂ : f (-2) k = 0) : k = 3 := 
by sorry

-- Prove the range of m such that f(x) < m for x in [-2, 2]
theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + 3*x + 2 < m) ↔ 12 < m :=
by sorry

end find_k_range_of_m_l2142_214292


namespace smallest_base_for_100_l2142_214266

theorem smallest_base_for_100 : ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
by
  use 5
  sorry

end smallest_base_for_100_l2142_214266


namespace range_3a_2b_l2142_214254

theorem range_3a_2b (a b : ℝ) (h : a^2 + b^2 = 4) : 
  -2 * Real.sqrt 13 ≤ 3 * a + 2 * b ∧ 3 * a + 2 * b ≤ 2 * Real.sqrt 13 := 
by 
  sorry

end range_3a_2b_l2142_214254


namespace ratio_of_numbers_l2142_214229

theorem ratio_of_numbers (a b : ℕ) (hHCF : Nat.gcd a b = 4) (hLCM : Nat.lcm a b = 48) : a / b = 3 / 4 :=
by
  sorry

end ratio_of_numbers_l2142_214229


namespace paul_runs_41_miles_l2142_214201

-- Conditions as Definitions
def movie1_length : ℕ := (1 * 60) + 36
def movie2_length : ℕ := (2 * 60) + 18
def movie3_length : ℕ := (1 * 60) + 48
def movie4_length : ℕ := (2 * 60) + 30
def total_watch_time : ℕ := movie1_length + movie2_length + movie3_length + movie4_length
def time_per_mile : ℕ := 12

-- Proof Statement
theorem paul_runs_41_miles : total_watch_time / time_per_mile = 41 :=
by
  -- Proof would be provided here
  sorry 

end paul_runs_41_miles_l2142_214201


namespace bet_strategy_possible_l2142_214225

def betting_possibility : Prop :=
  (1 / 6 + 1 / 2 + 1 / 9 + 1 / 8 <= 1)

theorem bet_strategy_possible : betting_possibility :=
by
  -- Proof is intentionally omitted
  sorry

end bet_strategy_possible_l2142_214225


namespace quotient_of_1575_210_l2142_214272

theorem quotient_of_1575_210 (a b q : ℕ) (h1 : a = 1575) (h2 : b = a - 1365) (h3 : a % b = 15) : q = 7 :=
by {
  sorry
}

end quotient_of_1575_210_l2142_214272


namespace arithmetic_geometric_sequences_l2142_214235

variable {S T : ℕ → ℝ}
variable {a b : ℕ → ℝ}

theorem arithmetic_geometric_sequences (h1 : a 3 = b 3)
  (h2 : a 4 = b 4)
  (h3 : (S 5 - S 3) / (T 4 - T 2) = 5) :
  (a 5 + a 3) / (b 5 + b 3) = - (3 / 5) := by
  sorry

end arithmetic_geometric_sequences_l2142_214235


namespace train_crosses_pole_time_l2142_214282

theorem train_crosses_pole_time
  (l : ℕ) (v_kmh : ℕ) (v_ms : ℚ) (t : ℕ)
  (h_l : l = 100)
  (h_v_kmh : v_kmh = 180)
  (h_v_ms_conversion : v_ms = v_kmh * 1000 / 3600)
  (h_v_ms : v_ms = 50) :
  t = l / v_ms := by
  sorry

end train_crosses_pole_time_l2142_214282


namespace center_of_circle_is_1_2_l2142_214276

theorem center_of_circle_is_1_2 :
  ∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y = 0 ↔ ∃ (r : ℝ), (x - 1)^2 + (y - 2)^2 = r^2 := by
  sorry

end center_of_circle_is_1_2_l2142_214276


namespace problem_rational_sum_of_powers_l2142_214239

theorem problem_rational_sum_of_powers :
  ∃ (a b : ℚ), (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 ∧ a + b = 70 :=
by
  sorry

end problem_rational_sum_of_powers_l2142_214239


namespace bottle_caps_per_group_l2142_214286

theorem bottle_caps_per_group (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) 
  (h1 : total_caps = 12) (h2 : num_groups = 6) : 
  total_caps / num_groups = caps_per_group := by
  sorry

end bottle_caps_per_group_l2142_214286


namespace distance_along_stream_1_hour_l2142_214209

noncomputable def boat_speed_still_water : ℝ := 4
noncomputable def stream_speed : ℝ := 2
noncomputable def effective_speed_against_stream : ℝ := boat_speed_still_water - stream_speed
noncomputable def effective_speed_along_stream : ℝ := boat_speed_still_water + stream_speed

theorem distance_along_stream_1_hour : 
  effective_speed_agains_stream = 2 → effective_speed_along_stream * 1 = 6 :=
by
  sorry

end distance_along_stream_1_hour_l2142_214209


namespace not_all_odd_l2142_214234

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1
def divides (a b c d : ℕ) : Prop := a = b * c + d ∧ 0 ≤ d ∧ d < b

theorem not_all_odd (a b c d : ℕ) 
  (h_div : divides a b c d)
  (h_odd_a : is_odd a)
  (h_odd_b : is_odd b)
  (h_odd_c : is_odd c)
  (h_odd_d : is_odd d) :
  False :=
sorry

end not_all_odd_l2142_214234


namespace incorrect_option_D_l2142_214253

variable (AB BC BO DO AO CO : ℝ)
variable (DAB : ℝ)
variable (ABCD_is_rectangle ABCD_is_rhombus ABCD_is_square: Prop)

def conditions_statement :=
  AB = BC ∧
  DAB = 90 ∧
  BO = DO ∧
  AO = CO ∧
  (ABCD_is_rectangle ↔ (AB = BC ∧ AB ≠ BC)) ∧
  (ABCD_is_rhombus ↔ AB = BC ∧ AB ≠ BC) ∧
  (ABCD_is_square ↔ ABCD_is_rectangle ∧ ABCD_is_rhombus)

theorem incorrect_option_D
  (h1: BO = DO)
  (h2: AO = CO)
  (h3: ABCD_is_rectangle)
  (h4: conditions_statement AB BC BO DO AO CO DAB ABCD_is_rectangle ABCD_is_rhombus ABCD_is_square):
  ¬ ABCD_is_square :=
by
  sorry
  -- Proof omitted

end incorrect_option_D_l2142_214253


namespace sum_of_roots_l2142_214278

theorem sum_of_roots (x y : ℝ) (h : ∀ z, z^2 + 2023 * z - 2024 = 0 → z = x ∨ z = y) : x + y = -2023 := 
by
  sorry

end sum_of_roots_l2142_214278


namespace solve_abs_eq_l2142_214231

theorem solve_abs_eq (x : ℝ) : (|x + 4| = 3 - x) → (x = -1/2) := by
  intro h
  sorry

end solve_abs_eq_l2142_214231


namespace molecular_weight_of_compound_l2142_214259

theorem molecular_weight_of_compound (total_weight_of_3_moles : ℝ) (n_moles : ℝ) 
  (h1 : total_weight_of_3_moles = 528) (h2 : n_moles = 3) : 
  (total_weight_of_3_moles / n_moles) = 176 :=
by
  sorry

end molecular_weight_of_compound_l2142_214259


namespace right_isosceles_triangle_acute_angle_45_l2142_214244

theorem right_isosceles_triangle_acute_angle_45
    (a : ℝ)
    (h_leg_conditions : ∀ b : ℝ, a = b)
    (h_hypotenuse_condition : ∀ c : ℝ, c^2 = 2 * (a * a)) :
    ∃ θ : ℝ, θ = 45 :=
by
    sorry

end right_isosceles_triangle_acute_angle_45_l2142_214244


namespace machine_a_produces_18_sprockets_per_hour_l2142_214212

theorem machine_a_produces_18_sprockets_per_hour :
  ∃ (A : ℝ), (∀ (B C : ℝ),
  B = 1.10 * A ∧
  B = 1.20 * C ∧
  990 / A = 990 / B + 10 ∧
  990 / C = 990 / A - 5) →
  A = 18 :=
by { sorry }

end machine_a_produces_18_sprockets_per_hour_l2142_214212


namespace find_rope_costs_l2142_214214

theorem find_rope_costs (x y : ℕ) (h1 : 10 * x + 5 * y = 175) (h2 : 15 * x + 10 * y = 300) : x = 10 ∧ y = 15 :=
    sorry

end find_rope_costs_l2142_214214


namespace bananas_to_mush_l2142_214206

theorem bananas_to_mush (x : ℕ) (h1 : 3 * (20 / x) = 15) : x = 4 :=
by
  sorry

end bananas_to_mush_l2142_214206


namespace window_area_properties_l2142_214255

theorem window_area_properties
  (AB : ℝ) (AD : ℝ) (ratio : ℝ)
  (h1 : ratio = 3 / 1)
  (h2 : AB = 40)
  (h3 : AD = 3 * AB) :
  (AD * AB / (π * (AB / 2) ^ 2) = 12 / π) ∧
  (AD * AB + π * (AB / 2) ^ 2 = 4800 + 400 * π) :=
by
  -- Proof will go here
  sorry

end window_area_properties_l2142_214255


namespace sequence_general_term_l2142_214294

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n ≥ 1 → a n = n * (a (n + 1) - a n)) : 
  ∀ n : ℕ, n ≥ 1 → a n = n := 
by 
  sorry

end sequence_general_term_l2142_214294


namespace volume_of_prism_l2142_214242

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 45) (h3 : b * c = 54) : 
  a * b * c = 270 :=
by
  sorry

end volume_of_prism_l2142_214242


namespace xy_squared_value_l2142_214263

theorem xy_squared_value (x y : ℝ) (h1 : x * (x + y) = 22) (h2 : y * (x + y) = 78 - y) :
  (x + y) ^ 2 = 100 :=
  sorry

end xy_squared_value_l2142_214263


namespace problem1_problem2_problem3_l2142_214297

noncomputable def f (x a : ℝ) : ℝ := abs x * (x - a)

-- 1. Prove a = 0 if f(x) is odd
theorem problem1 (h: ∀ x : ℝ, f (-x) a = -f x a) : a = 0 :=
sorry

-- 2. Prove a ≤ 0 if f(x) is increasing on the interval [0, 2]
theorem problem2 (h: ∀ x y : ℝ, 0 ≤ x → x ≤ y → y ≤ 2 → f x a ≤ f y a) : a ≤ 0 :=
sorry

-- 3. Prove there exists an a < 0 such that the maximum value of f(x) on [-1, 1/2] is 2, and find a = -3
theorem problem3 (h: ∃ a : ℝ, a < 0 ∧ ∀ x : ℝ, -1 ≤ x → x ≤ 1/2 → f x a ≤ 2 ∧ ∃ x : ℝ, -1 ≤ x → x ≤ 1/2 → f x a = 2) : a = -3 :=
sorry

end problem1_problem2_problem3_l2142_214297


namespace five_student_committees_from_ten_select_two_committees_with_three_overlap_l2142_214233

-- Lean statement for the first part: number of different five-student committees from ten students.
theorem five_student_committees_from_ten : 
  (Nat.choose 10 5) = 252 := 
by
  sorry

-- Lean statement for the second part: number of ways to choose two five-student committees with exactly three overlapping members.
theorem select_two_committees_with_three_overlap :
  ( (Nat.choose 10 5) * ( (Nat.choose 5 3) * (Nat.choose 5 2) ) ) / 2 = 12600 := 
by
  sorry

end five_student_committees_from_ten_select_two_committees_with_three_overlap_l2142_214233


namespace TinaTotalPens_l2142_214269

variable (p g b : ℕ)
axiom H1 : p = 12
axiom H2 : g = p - 9
axiom H3 : b = g + 3

theorem TinaTotalPens : p + g + b = 21 := by
  sorry

end TinaTotalPens_l2142_214269


namespace Brenda_mice_left_l2142_214219

theorem Brenda_mice_left :
  ∀ (total_litters total_each sixth factor remaining : ℕ),
    total_litters = 3 → 
    total_each = 8 →
    sixth = total_litters * total_each / 6 →
    factor = 3 * (total_litters * total_each / 6) →
    remaining = total_litters * total_each - sixth - factor →
    remaining / 2 = ((total_litters * total_each - sixth - factor) / 2) →
    total_litters * total_each - sixth - factor - ((total_litters * total_each - sixth - factor) / 2) = 4 :=
by
  intros total_litters total_each sixth factor remaining h_litters h_each h_sixth h_factor h_remaining h_half
  sorry

end Brenda_mice_left_l2142_214219


namespace all_numbers_even_l2142_214267

theorem all_numbers_even
  (A B C D E : ℤ)
  (h1 : (A + B + C) % 2 = 0)
  (h2 : (A + B + D) % 2 = 0)
  (h3 : (A + B + E) % 2 = 0)
  (h4 : (A + C + D) % 2 = 0)
  (h5 : (A + C + E) % 2 = 0)
  (h6 : (A + D + E) % 2 = 0)
  (h7 : (B + C + D) % 2 = 0)
  (h8 : (B + C + E) % 2 = 0)
  (h9 : (B + D + E) % 2 = 0)
  (h10 : (C + D + E) % 2 = 0) :
  (A % 2 = 0) ∧ (B % 2 = 0) ∧ (C % 2 = 0) ∧ (D % 2 = 0) ∧ (E % 2 = 0) :=
sorry

end all_numbers_even_l2142_214267


namespace diameter_outer_boundary_correct_l2142_214217

noncomputable def diameter_outer_boundary 
  (D_fountain : ℝ)
  (w_gardenRing : ℝ)
  (w_innerPath : ℝ)
  (w_outerPath : ℝ) : ℝ :=
  let R_fountain := D_fountain / 2
  let R_innerPath := R_fountain + w_gardenRing
  let R_outerPathInner := R_innerPath + w_innerPath
  let R_outerPathOuter := R_outerPathInner + w_outerPath
  2 * R_outerPathOuter

theorem diameter_outer_boundary_correct :
  diameter_outer_boundary 10 12 3 4 = 48 := by
  -- skipping proof
  sorry

end diameter_outer_boundary_correct_l2142_214217


namespace smallest_sum_squares_edges_is_cube_l2142_214295

theorem smallest_sum_squares_edges_is_cube (V : ℝ) (a b c : ℝ)
  (h_vol : a * b * c = V) :
  a^2 + b^2 + c^2 ≥ 3 * (V^(2/3)) := 
sorry

end smallest_sum_squares_edges_is_cube_l2142_214295


namespace solution_set_for_inequality_l2142_214296

theorem solution_set_for_inequality (f : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_decreasing : ∀ ⦃x y⦄, 0 < x → x < y → f y < f x)
  (h_f_neg3 : f (-3) = 1) :
  { x | f x < 1 } = { x | x < -3 ∨ 3 < x } := 
by
  -- TODO: Prove this theorem
  sorry

end solution_set_for_inequality_l2142_214296


namespace total_mission_days_l2142_214202

variable (initial_days_first_mission : ℝ := 5)
variable (percentage_longer : ℝ := 0.60)
variable (days_second_mission : ℝ := 3)

theorem total_mission_days : 
  let days_first_mission_extra := initial_days_first_mission * percentage_longer
  let total_days_first_mission := initial_days_first_mission + days_first_mission_extra
  (total_days_first_mission + days_second_mission) = 11 := by
  sorry

end total_mission_days_l2142_214202


namespace laura_running_speed_l2142_214205

theorem laura_running_speed (x : ℝ) (hx : 3 * x + 1 > 0) : 
    (30 / (3 * x + 1)) + (10 / x) = 31 / 12 → x = 7.57 := 
by 
  sorry

end laura_running_speed_l2142_214205


namespace remainder_97_pow_103_mul_7_mod_17_l2142_214221

theorem remainder_97_pow_103_mul_7_mod_17 :
  (97 ^ 103 * 7) % 17 = 13 := by
  have h1 : 97 % 17 = -3 % 17 := by sorry
  have h2 : 9 % 17 = -8 % 17 := by sorry
  have h3 : 64 % 17 = 13 % 17 := by sorry
  have h4 : -21 % 17 = 13 % 17 := by sorry
  sorry

end remainder_97_pow_103_mul_7_mod_17_l2142_214221


namespace no_solution_x_l2142_214248

theorem no_solution_x : ¬ ∃ x : ℝ, x * (x - 1) * (x - 2) + (100 - x) * (99 - x) * (98 - x) = 0 := 
sorry

end no_solution_x_l2142_214248


namespace largest_perfect_square_factor_9240_l2142_214273

theorem largest_perfect_square_factor_9240 :
  ∃ n : ℕ, n * n = 36 ∧ ∃ m : ℕ, m ∣ 9240 ∧ m = n * n :=
by
  -- We will construct the proof here using the prime factorization
  sorry

end largest_perfect_square_factor_9240_l2142_214273


namespace keiths_total_spending_l2142_214243

theorem keiths_total_spending :
  let digimon_cost := 4 * 4.45
  let pokemon_cost := 3 * 5.25
  let yugioh_cost := 6 * 3.99
  let mtg_cost := 2 * 6.75
  let baseball_cost := 1 * 6.06
  let total_cost := digimon_cost + pokemon_cost + yugioh_cost + mtg_cost + baseball_cost
  total_cost = 77.05 :=
by
  let digimon_cost := 4 * 4.45
  let pokemon_cost := 3 * 5.25
  let yugioh_cost := 6 * 3.99
  let mtg_cost := 2 * 6.75
  let baseball_cost := 1 * 6.06
  let total_cost := digimon_cost + pokemon_cost + yugioh_cost + mtg_cost + baseball_cost
  have h : total_cost = 77.05 := sorry
  exact h

end keiths_total_spending_l2142_214243


namespace temperature_on_friday_l2142_214203

theorem temperature_on_friday 
  (M T W Th F : ℝ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 42) : 
  F = 34 :=
by
  sorry

end temperature_on_friday_l2142_214203


namespace original_number_is_85_l2142_214232

theorem original_number_is_85
  (x : ℤ) (h_sum : 10 ≤ x ∧ x < 100) 
  (h_condition1 : (x / 10) + (x % 10) = 13)
  (h_condition2 : 10 * (x % 10) + (x / 10) = x - 27) :
  x = 85 :=
by
  sorry

end original_number_is_85_l2142_214232


namespace intersection_of_A_and_B_l2142_214226

-- Define set A
def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

-- Define set B
def B : Set ℤ := {2, 4, 6, 8}

-- Prove that the intersection of set A and set B is {2, 4}.
theorem intersection_of_A_and_B : A ∩ B = {2, 4} :=
by
  sorry

end intersection_of_A_and_B_l2142_214226


namespace polynomial_coefficient_sum_l2142_214293

theorem polynomial_coefficient_sum
  (a b c d : ℤ)
  (h1 : (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + 2 * x^3 - 5 * x^2 + 8 * x - 12) :
  a + b + c + d = 6 := 
sorry

end polynomial_coefficient_sum_l2142_214293


namespace set_union_is_all_real_l2142_214289

-- Define the universal set U as the real numbers
def U := ℝ

-- Define the set M as {x | x > 0}
def M : Set ℝ := {x | x > 0}

-- Define the set N as {x | x^2 ≥ x}
def N : Set ℝ := {x | x^2 ≥ x}

-- Prove the relationship M ∪ N = ℝ
theorem set_union_is_all_real : M ∪ N = U := by
  sorry

end set_union_is_all_real_l2142_214289


namespace range_of_a_l2142_214207

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (1 - 2 * a) * x + 3 * a else Real.log x

theorem range_of_a (a : ℝ) : (-1 ≤ a ∧ a < 1/2) ↔
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) :=
by
  sorry

end range_of_a_l2142_214207


namespace last_two_digits_of_sum_of_first_15_factorials_eq_13_l2142_214238

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits_sum : ℕ :=
  let partial_sum := (factorial 1 % 100) + (factorial 2 % 100) + (factorial 3 % 100) +
                     (factorial 4 % 100) + (factorial 5 % 100) + (factorial 6 % 100) +
                     (factorial 7 % 100) + (factorial 8 % 100) + (factorial 9 % 100)
  partial_sum % 100

theorem last_two_digits_of_sum_of_first_15_factorials_eq_13 : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_eq_13_l2142_214238


namespace probability_of_exactly_one_solves_l2142_214251

variable (p1 p2 : ℝ)

theorem probability_of_exactly_one_solves (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end probability_of_exactly_one_solves_l2142_214251


namespace solve_cubic_equation_l2142_214291

theorem solve_cubic_equation (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by sorry

end solve_cubic_equation_l2142_214291


namespace nathan_correct_answers_l2142_214227

theorem nathan_correct_answers (c w : ℤ) (h1 : c + w = 15) (h2 : 6 * c - 3 * w = 45) : c = 10 := 
by sorry

end nathan_correct_answers_l2142_214227


namespace main_inequality_l2142_214237

noncomputable def b (c : ℝ) : ℝ := (1 + c) / (2 + c)

def f (c : ℝ) (x : ℝ) : ℝ := sorry

lemma f_continuous (c : ℝ) (h_c : 0 < c) : Continuous (f c) := sorry

lemma condition1 (c : ℝ) (h_c : 0 < c) (x : ℝ) (h_x : 0 ≤ x ∧ x ≤ 1/2) : 
  b c * f c (2 * x) = f c x := sorry

lemma condition2 (c : ℝ) (h_c : 0 < c) (x : ℝ) (h_x : 1/2 ≤ x ∧ x ≤ 1) : 
  f c x = b c + (1 - b c) * f c (2 * x - 1) := sorry

theorem main_inequality (c : ℝ) (h_c : 0 < c) : 
  ∀ x : ℝ, (0 < x ∧ x < 1) → (0 < f c x - x ∧ f c x - x < c) := sorry

end main_inequality_l2142_214237


namespace percentage_increase_in_items_sold_l2142_214215

-- Definitions
variables (P N M : ℝ)
-- Given conditions:
-- The new price of an item
def new_price := P * 0.90
-- The relationship between incomes
def income_increase := (P * 0.90) * M = P * N * 1.125

-- The problem statement
theorem percentage_increase_in_items_sold (h : income_increase P N M) :
  M = N * 1.25 :=
sorry

end percentage_increase_in_items_sold_l2142_214215


namespace total_stones_l2142_214284

theorem total_stones (sent_away kept total : ℕ) (h1 : sent_away = 63) (h2 : kept = 15) (h3 : total = sent_away + kept) : total = 78 :=
by
  sorry

end total_stones_l2142_214284


namespace impossibility_of_quadratic_conditions_l2142_214285

open Real

theorem impossibility_of_quadratic_conditions :
  ∀ (a b c t : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ t ∧ b ≠ t ∧ c ≠ t →
  (b * t) ^ 2 - 4 * a * c > 0 →
  c ^ 2 - 4 * b * a > 0 →
  (a * t) ^ 2 - 4 * b * c > 0 →
  false :=
by sorry

end impossibility_of_quadratic_conditions_l2142_214285


namespace sufficient_not_necessary_condition_l2142_214216

-- Definition of the proposition p
def prop_p (m : ℝ) := ∀ x : ℝ, x^2 - 4 * x + 2 * m ≥ 0

-- Statement of the proof problem
theorem sufficient_not_necessary_condition (m : ℝ) : 
  (m ≥ 3 → prop_p m) ∧ ¬(m ≥ 3 → m ≥ 2) ∧ (m ≥ 2 → prop_p m) → (m ≥ 3 → prop_p m) ∧ ¬(m ≥ 3 ↔ prop_p m) :=
sorry

end sufficient_not_necessary_condition_l2142_214216


namespace rectangle_long_side_eq_12_l2142_214240

theorem rectangle_long_side_eq_12 (s : ℕ) (a b : ℕ) (congruent_triangles : true) (h : a + b = s) (short_side_is_8 : s = 8) : a + b + 4 = 12 :=
by
  sorry

end rectangle_long_side_eq_12_l2142_214240


namespace num_colors_l2142_214245

def total_balls := 350
def balls_per_color := 35

theorem num_colors :
  total_balls / balls_per_color = 10 := 
by
  sorry

end num_colors_l2142_214245


namespace analytical_expression_of_f_l2142_214279

theorem analytical_expression_of_f (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f (x + 1 / x) = x^2 + 1 / x^2) →
  (∀ y : ℝ, (y ≥ 2 ∨ y ≤ -2) → f y = y^2 - 2) :=
by
  intro h1 y hy
  sorry

end analytical_expression_of_f_l2142_214279


namespace products_B_correct_l2142_214213

-- Define the total number of products
def total_products : ℕ := 4800

-- Define the sample size and the number of pieces from equipment A in the sample
def sample_size : ℕ := 80
def sample_A : ℕ := 50

-- Define the number of products produced by equipment A and B
def products_A : ℕ := 3000
def products_B : ℕ := total_products - products_A

-- The target number of products produced by equipment B
def target_products_B : ℕ := 1800

-- The theorem we need to prove
theorem products_B_correct :
  products_B = target_products_B := by
  sorry

end products_B_correct_l2142_214213


namespace delores_initial_money_l2142_214262

-- Definitions and conditions based on the given problem
def original_computer_price : ℝ := 400
def original_printer_price : ℝ := 40
def original_headphones_price : ℝ := 60

def computer_discount : ℝ := 0.10
def computer_tax : ℝ := 0.08
def printer_tax : ℝ := 0.05
def headphones_tax : ℝ := 0.06

def leftover_money : ℝ := 10

-- Final proof problem statement
theorem delores_initial_money :
  original_computer_price * (1 - computer_discount) * (1 + computer_tax) +
  original_printer_price * (1 + printer_tax) +
  original_headphones_price * (1 + headphones_tax) + leftover_money = 504.40 := by
  sorry -- Proof is not required

end delores_initial_money_l2142_214262


namespace expression_factorization_l2142_214299

variables (a b c : ℝ)

theorem expression_factorization :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3)
  = (a - b) * (b - c) * (c - a) * (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
sorry

end expression_factorization_l2142_214299


namespace sqrt_expression_equality_l2142_214288

theorem sqrt_expression_equality :
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * 5^(3/4) :=
by
  sorry

end sqrt_expression_equality_l2142_214288


namespace converse_example_l2142_214257

theorem converse_example (x : ℝ) (h : x^2 = 1) : x = 1 :=
sorry

end converse_example_l2142_214257


namespace percentage_decrease_of_b_l2142_214223

variables (a b x m : ℝ) (p : ℝ)

-- Given conditions
def ratio_ab : Prop := a / b = 4 / 5
def expression_x : Prop := x = 1.25 * a
def expression_m : Prop := m = b * (1 - p / 100)
def ratio_mx : Prop := m / x = 0.6

-- The theorem to be proved
theorem percentage_decrease_of_b 
  (h1 : ratio_ab a b)
  (h2 : expression_x a x)
  (h3 : expression_m b m p)
  (h4 : ratio_mx m x) 
  : p = 40 :=
sorry

end percentage_decrease_of_b_l2142_214223


namespace ivan_speed_ratio_l2142_214274

/-- 
A group of tourists started a hike from a campsite. Fifteen minutes later, Ivan returned to the campsite for a flashlight 
and started catching up with the group at a faster constant speed. He reached them 2.5 hours after initially leaving. 
Prove Ivan's speed is 1.2 times the group's speed.
-/
theorem ivan_speed_ratio (d_g d_i : ℝ) (t_g t_i : ℝ) (v_g v_i : ℝ)
    (h1 : t_g = 2.25)       -- Group's travel time (2.25 hours after initial 15 minutes)
    (h2 : t_i = 2.5)        -- Ivan's total travel time
    (h3 : d_g = t_g * v_g)  -- Distance covered by group
    (h4 : d_i = 3 * (v_g * (15 / 60))) -- Ivan's distance covered
    (h5 : d_g = d_i)        -- Ivan eventually catches up with the group
  : v_i / v_g = 1.2 := sorry

end ivan_speed_ratio_l2142_214274


namespace pages_in_each_book_l2142_214287

variable (BooksRead DaysPerBook TotalDays : ℕ)

theorem pages_in_each_book (h1 : BooksRead = 41) (h2 : DaysPerBook = 12) (h3 : TotalDays = 492) : (TotalDays / DaysPerBook) * DaysPerBook = 492 :=
by
  sorry

end pages_in_each_book_l2142_214287


namespace quadratic_has_real_solutions_l2142_214270

theorem quadratic_has_real_solutions (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 := 
by
  sorry

end quadratic_has_real_solutions_l2142_214270


namespace candy_box_original_price_l2142_214252

theorem candy_box_original_price (P : ℝ) (h1 : 1.25 * P = 20) : P = 16 :=
sorry

end candy_box_original_price_l2142_214252
