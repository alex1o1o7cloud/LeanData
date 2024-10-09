import Mathlib

namespace solve_inequality_system_l674_67488

theorem solve_inequality_system : 
  (∀ x : ℝ, (1 / 3 * x - 1 ≤ 1 / 2 * x + 1) ∧ ((3 * x - (x - 2) ≥ 6) ∧ (x + 1 > (4 * x - 1) / 3)) → (2 ≤ x ∧ x < 4)) := 
by
  intro x h
  sorry

end solve_inequality_system_l674_67488


namespace fraction_weevils_25_percent_l674_67414

-- Define the probabilities
def prob_good_milk : ℝ := 0.8
def prob_good_egg : ℝ := 0.4
def prob_all_good : ℝ := 0.24

-- The problem definition and statement
def fraction_weevils (F : ℝ) : Prop :=
  0.32 * (1 - F) = 0.24

theorem fraction_weevils_25_percent : fraction_weevils 0.25 :=
by sorry

end fraction_weevils_25_percent_l674_67414


namespace correct_average_of_20_numbers_l674_67407

theorem correct_average_of_20_numbers 
  (incorrect_avg : ℕ) 
  (n : ℕ) 
  (incorrectly_read : ℕ) 
  (correction : ℕ) 
  (a b c d e f g h i j : ℤ) 
  (sum_a_b_c_d_e : ℤ)
  (sum_f_g_h_i_j : ℤ)
  (incorrect_sum : ℤ)
  (correction_sum : ℤ) 
  (corrected_sum : ℤ)
  (correct_avg : ℤ) : 
  incorrect_avg = 35 ∧ 
  n = 20 ∧ 
  incorrectly_read = 5 ∧ 
  correction = 136 ∧ 
  a = 90 ∧ b = 73 ∧ c = 85 ∧ d = -45 ∧ e = 64 ∧ 
  f = 45 ∧ g = 36 ∧ h = 42 ∧ i = -27 ∧ j = 35 ∧ 
  sum_a_b_c_d_e = a + b + c + d + e ∧
  sum_f_g_h_i_j = f + g + h + i + j ∧
  incorrect_sum = incorrect_avg * n ∧ 
  correction_sum = sum_a_b_c_d_e - sum_f_g_h_i_j ∧ 
  corrected_sum = incorrect_sum + correction_sum → correct_avg = corrected_sum / n := 
  by sorry

end correct_average_of_20_numbers_l674_67407


namespace zero_a_if_square_every_n_l674_67402

theorem zero_a_if_square_every_n (a b : ℤ) (h : ∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) : a = 0 := 
sorry

end zero_a_if_square_every_n_l674_67402


namespace solution_x_alcohol_percentage_l674_67419

theorem solution_x_alcohol_percentage (P : ℝ) :
  let y_percentage := 0.30
  let mixture_percentage := 0.25
  let y_volume := 600
  let x_volume := 200
  let mixture_volume := y_volume + x_volume
  let y_alcohol_content := y_volume * y_percentage
  let mixture_alcohol_content := mixture_volume * mixture_percentage
  P * x_volume + y_alcohol_content = mixture_alcohol_content →
  P = 0.10 :=
by
  intros
  sorry

end solution_x_alcohol_percentage_l674_67419


namespace arithmetic_sequence_properties_l674_67491

theorem arithmetic_sequence_properties
    (a_1 : ℕ)
    (d : ℕ)
    (sequence : Fin 240 → ℕ)
    (h1 : ∀ n, sequence n = a_1 + n * d)
    (h2 : sequence 0 = a_1)
    (h3 : 1 ≤ a_1 ∧ a_1 ≤ 9)
    (h4 : ∃ n₁, sequence n₁ = 100)
    (h5 : ∃ n₂, sequence n₂ = 3103) :
  (a_1 = 9 ∧ d = 13) ∨ (a_1 = 1 ∧ d = 33) ∨ (a_1 = 9 ∧ d = 91) :=
sorry

end arithmetic_sequence_properties_l674_67491


namespace find_angle_A_l674_67498

theorem find_angle_A (a b : ℝ) (B A : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 2) (hB : B = Real.pi / 4) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_angle_A_l674_67498


namespace michael_choices_l674_67426

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem michael_choices : combination 10 4 = 210 := by
  sorry

end michael_choices_l674_67426


namespace equivalent_forms_l674_67441

-- Given line equation
def given_line_eq (x y : ℝ) : Prop :=
  (3 * x - 2) / 4 - (2 * y - 1) / 2 = 1

-- General form of the line
def general_form (x y : ℝ) : Prop :=
  3 * x - 8 * y - 2 = 0

-- Slope-intercept form of the line
def slope_intercept_form (x y : ℝ) : Prop := 
  y = (3 / 8) * x - 1 / 4

-- Intercept form of the line
def intercept_form (x y : ℝ) : Prop :=
  x / (2 / 3) + y / (-1 / 4) = 1

-- Normal form of the line
def normal_form (x y : ℝ) : Prop :=
  3 / Real.sqrt 73 * x - 8 / Real.sqrt 73 * y - 2 / Real.sqrt 73 = 0

-- Proof problem: Prove that the given line equation is equivalent to the derived forms
theorem equivalent_forms (x y : ℝ) :
  given_line_eq x y ↔ (general_form x y ∧ slope_intercept_form x y ∧ intercept_form x y ∧ normal_form x y) :=
sorry

end equivalent_forms_l674_67441


namespace floor_x_floor_x_eq_44_iff_l674_67476

theorem floor_x_floor_x_eq_44_iff (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 44) ↔ (7.333 ≤ x ∧ x < 7.5) :=
by
  sorry

end floor_x_floor_x_eq_44_iff_l674_67476


namespace women_attended_l674_67472

theorem women_attended (m w : ℕ) 
  (h_danced_with_4_women : ∀ (k : ℕ), k < m → k * 4 = 60)
  (h_danced_with_3_men : ∀ (k : ℕ), k < w → 3 * (k * (m / 3)) = 60)
  (h_men_count : m = 15) : 
  w = 20 := 
sorry

end women_attended_l674_67472


namespace arithmetic_sequence_general_formula_inequality_satisfaction_l674_67473

namespace Problem

-- Definitions for the sequences and the sum of terms
def a (n : ℕ) : ℕ := sorry -- define based on conditions
def S (n : ℕ) : ℕ := sorry -- sum of first n terms of {a_n}
def b (n : ℕ) : ℕ := 2 * (S (n + 1) - S n) * S n - n * (S (n + 1) + S n)

-- Part 1: Prove the general formula for the arithmetic sequence
theorem arithmetic_sequence_general_formula :
  (∀ n : ℕ, b n = 0) → (∀ n : ℕ, a n = 0 ∨ a n = n) :=
sorry

-- Part 2: Conditions for geometric sequences and inequality
def a_2n_minus_1 (n : ℕ) : ℕ := 2 ^ n
def a_2n (n : ℕ) : ℕ := 3 * 2 ^ (n - 1)
def b_2n (n : ℕ) : ℕ := sorry -- compute based on conditions
def b_2n_minus_1 (n : ℕ) : ℕ := sorry -- compute based on conditions

def b_condition (n : ℕ) : Prop := b_2n n < b_2n_minus_1 n

-- Prove the set of all positive integers n that satisfy the inequality
theorem inequality_satisfaction :
  { n : ℕ | b_condition n } = {1, 2, 3, 4, 5, 6} :=
sorry

end Problem

end arithmetic_sequence_general_formula_inequality_satisfaction_l674_67473


namespace total_silk_dyed_correct_l674_67404

-- Define the conditions
def green_silk_yards : ℕ := 61921
def pink_silk_yards : ℕ := 49500
def total_silk_yards : ℕ := green_silk_yards + pink_silk_yards

-- State the theorem to be proved
theorem total_silk_dyed_correct : total_silk_yards = 111421 := by
  sorry

end total_silk_dyed_correct_l674_67404


namespace acres_used_for_corn_l674_67425

noncomputable def total_acres : ℝ := 1634
noncomputable def beans_ratio : ℝ := 4.5
noncomputable def wheat_ratio : ℝ := 2.3
noncomputable def corn_ratio : ℝ := 3.8
noncomputable def barley_ratio : ℝ := 3.4

noncomputable def total_parts : ℝ := beans_ratio + wheat_ratio + corn_ratio + barley_ratio
noncomputable def acres_per_part : ℝ := total_acres / total_parts
noncomputable def corn_acres : ℝ := corn_ratio * acres_per_part

theorem acres_used_for_corn :
  corn_acres = 443.51 := by
  sorry

end acres_used_for_corn_l674_67425


namespace audit_sampling_is_systematic_l674_67447

def is_systematic_sampling (population_size : Nat) (step : Nat) (initial_index : Nat) : Prop :=
  ∃ (k : Nat), ∀ (n : Nat), n ≠ 0 → initial_index + step * (n - 1) ≤ population_size

theorem audit_sampling_is_systematic :
  ∀ (population_size : Nat) (random_index : Nat),
  population_size = 50 * 50 →  -- This represents the total number of invoices (50% of a larger population segment)
  random_index < 50 →         -- Randomly selected index from the first 50 invoices
  is_systematic_sampling population_size 50 random_index := 
by
  intros
  sorry

end audit_sampling_is_systematic_l674_67447


namespace unpainted_unit_cubes_l674_67469

theorem unpainted_unit_cubes (total_cubes painted_faces edge_overlaps corner_overlaps : ℕ) :
  total_cubes = 6 * 6 * 6 ∧
  painted_faces = 6 * (2 * 6) ∧
  edge_overlaps = 12 * 3 / 2 ∧
  corner_overlaps = 8 ∧
  total_cubes - (painted_faces - edge_overlaps - corner_overlaps) = 170 :=
by
  sorry

end unpainted_unit_cubes_l674_67469


namespace price_of_when_you_rescind_cd_l674_67434

variable (W : ℕ) -- Defining W as a natural number since prices can't be negative

theorem price_of_when_you_rescind_cd
  (price_life_journey : ℕ := 100)
  (price_day_life : ℕ := 50)
  (num_cds_each : ℕ := 3)
  (total_spent : ℕ := 705) :
  3 * price_life_journey + 3 * price_day_life + 3 * W = total_spent → 
  W = 85 :=
by
  intros h
  sorry

end price_of_when_you_rescind_cd_l674_67434


namespace third_speed_is_9_kmph_l674_67435

/-- Problem Statement: Given the total travel time, total distance, and two speeds, 
    prove that the third speed is 9 km/hr when distances are equal. -/
theorem third_speed_is_9_kmph (t : ℕ) (d_total : ℕ) (v1 v2 : ℕ) (d1 d2 d3 : ℕ) 
(h_t : t = 11)
(h_d_total : d_total = 900)
(h_v1 : v1 = 3)
(h_v2 : v2 = 6)
(h_d_eq : d1 = 300 ∧ d2 = 300 ∧ d3 = 300)
(h_sum_t : d1 / (v1 * 1000 / 60) + d2 / (v2 * 1000 / 60) + d3 / (v3 * 1000 / 60) = t) 
: (v3 = 9) :=
by 
  sorry

end third_speed_is_9_kmph_l674_67435


namespace c_share_of_profit_l674_67482

theorem c_share_of_profit
  (a_investment : ℝ)
  (b_investment : ℝ)
  (c_investment : ℝ)
  (total_profit : ℝ)
  (ha : a_investment = 30000)
  (hb : b_investment = 45000)
  (hc : c_investment = 50000)
  (hp : total_profit = 90000) :
  (c_investment / (a_investment + b_investment + c_investment)) * total_profit = 36000 := 
by
  sorry

end c_share_of_profit_l674_67482


namespace height_of_oil_truck_tank_l674_67428

/-- 
Given that a stationary oil tank is a right circular cylinder 
with a radius of 100 feet and its oil level dropped by 0.025 feet,
proving that if this oil is transferred to a right circular 
cylindrical oil truck's tank with a radius of 5 feet, then the 
height of the oil in the truck's tank will be 10 feet. 
-/
theorem height_of_oil_truck_tank
    (radius_stationary : ℝ) (height_drop_stationary : ℝ) (radius_truck : ℝ) 
    (height_truck : ℝ) (π : ℝ)
    (h1 : radius_stationary = 100)
    (h2 : height_drop_stationary = 0.025)
    (h3 : radius_truck = 5)
    (pi_approx : π = 3.14159265) :
    height_truck = 10 :=
by
    sorry

end height_of_oil_truck_tank_l674_67428


namespace quadratic_sum_l674_67474

theorem quadratic_sum (a b c : ℝ) (h : ∀ x : ℝ, 5 * x^2 - 30 * x - 45 = a * (x + b)^2 + c) :
  a + b + c = -88 := by
  sorry

end quadratic_sum_l674_67474


namespace jake_present_weight_l674_67468

theorem jake_present_weight (J S B : ℝ) (h1 : J - 20 = 2 * S) (h2 : B = 0.5 * J) (h3 : J + S + B = 330) :
  J = 170 :=
by sorry

end jake_present_weight_l674_67468


namespace binom_60_3_eq_34220_l674_67462

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l674_67462


namespace period_of_time_l674_67429

-- We define the annual expense and total amount spent as constants
def annual_expense : ℝ := 2
def total_amount_spent : ℝ := 20

-- Theorem to prove the period of time (in years)
theorem period_of_time : total_amount_spent / annual_expense = 10 :=
by 
  -- Placeholder proof
  sorry

end period_of_time_l674_67429


namespace maximum_triangle_area_le_8_l674_67432

def lengths : List ℝ := [2, 3, 4, 5, 6]

-- Function to determine if three lengths can form a valid triangle
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a 

-- Heron's formula to compute the area of a triangle given its sides
noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Statement to prove that the maximum possible area with given stick lengths is less than or equal to 8 cm²
theorem maximum_triangle_area_le_8 :
  ∃ (a b c : ℝ), a ∈ lengths ∧ b ∈ lengths ∧ c ∈ lengths ∧ 
  is_valid_triangle a b c ∧ heron_area a b c ≤ 8 :=
sorry

end maximum_triangle_area_le_8_l674_67432


namespace minimum_cuts_for_48_rectangles_l674_67424

theorem minimum_cuts_for_48_rectangles : 
  ∃ n : ℕ, n = 6 ∧ (∀ m < 6, 2 ^ m < 48) ∧ 2 ^ n ≥ 48 :=
by
  sorry

end minimum_cuts_for_48_rectangles_l674_67424


namespace charles_total_earnings_l674_67409

def charles_earnings (house_rate dog_rate : ℝ) (house_hours dog_count dog_hours : ℝ) : ℝ :=
  (house_rate * house_hours) + (dog_rate * dog_count * dog_hours)

theorem charles_total_earnings :
  charles_earnings 15 22 10 3 1 = 216 := by
  sorry

end charles_total_earnings_l674_67409


namespace problem_solution_l674_67470

theorem problem_solution (x : ℝ) : (∃ (x : ℝ), 5 < x ∧ x ≤ 6) ↔ (∃ (x : ℝ), (x - 3) / (x - 5) ≥ 3) :=
sorry

end problem_solution_l674_67470


namespace inverse_proposition_equivalence_l674_67416

theorem inverse_proposition_equivalence (x y : ℝ) :
  (x = y → abs x = abs y) ↔ (abs x = abs y → x = y) :=
sorry

end inverse_proposition_equivalence_l674_67416


namespace contrapositive_l674_67480

variable (P Q : Prop)

theorem contrapositive (h : P → Q) : ¬Q → ¬P :=
sorry

end contrapositive_l674_67480


namespace biography_increase_l674_67467

theorem biography_increase (B N : ℝ) (hN : N = 0.35 * (B + N) - 0.20 * B):
  (N / (0.20 * B) * 100) = 115.38 :=
by
  sorry

end biography_increase_l674_67467


namespace imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half_l674_67492

theorem imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half :
  (Complex.exp (-Complex.I * Real.pi / 6)).im = -1/2 := by
sorry

end imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half_l674_67492


namespace student_A_more_stable_performance_l674_67413

theorem student_A_more_stable_performance
    (mean : ℝ)
    (n : ℕ)
    (variance_A variance_B : ℝ)
    (h1 : mean = 1.6)
    (h2 : n = 10)
    (h3 : variance_A = 1.4)
    (h4 : variance_B = 2.5) :
    variance_A < variance_B :=
by {
    -- The proof is omitted as we are only writing the statement here.
    sorry
}

end student_A_more_stable_performance_l674_67413


namespace inequality_squares_l674_67493

theorem inequality_squares (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end inequality_squares_l674_67493


namespace abs_eq_5_iff_l674_67420

theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end abs_eq_5_iff_l674_67420


namespace odd_integer_solution_l674_67464

theorem odd_integer_solution
  (y : ℤ) (hy_odd : y % 2 = 1)
  (h : ∃ x : ℤ, x^2 + 2*y^2 = y*x^2 + y + 1) :
  y = 1 :=
sorry

end odd_integer_solution_l674_67464


namespace josh_initial_marbles_l674_67458

def marbles_initial (lost : ℕ) (left : ℕ) : ℕ := lost + left

theorem josh_initial_marbles :
  marbles_initial 5 4 = 9 :=
by sorry

end josh_initial_marbles_l674_67458


namespace minimum_x2_y2_z2_l674_67433

theorem minimum_x2_y2_z2 :
  ∀ x y z : ℝ, (x^3 + y^3 + z^3 - 3 * x * y * z = 1) → (∃ a b c : ℝ, a = x ∨ a = y ∨ a = z ∧ b = x ∨ b = y ∨ b = z ∧ c = x ∨ c = y ∨ a ≤ b ∨ a ≤ c ∧ b ≤ c) → (x^2 + y^2 + z^2 ≥ 1) :=
by
  sorry

end minimum_x2_y2_z2_l674_67433


namespace polar_circle_l674_67403

def is_circle (ρ θ : ℝ) : Prop :=
  ρ = Real.cos (Real.pi / 4 - θ)

theorem polar_circle : 
  ∀ ρ θ : ℝ, is_circle ρ θ ↔ ∃ (x y : ℝ), (x - 1/(2 * Real.sqrt 2))^2 + (y - 1/(2 * Real.sqrt 2))^2 = (1/(2 * Real.sqrt 2))^2 :=
by
  intro ρ θ
  sorry

end polar_circle_l674_67403


namespace smallest_five_digit_multiple_of_53_l674_67422

theorem smallest_five_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 53 = 0 ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_multiple_of_53_l674_67422


namespace cost_per_quart_l674_67461

theorem cost_per_quart (paint_cost : ℝ) (coverage : ℝ) (cost_to_paint_cube : ℝ) (cube_edge : ℝ) 
    (h_coverage : coverage = 1200) (h_cost_to_paint_cube : cost_to_paint_cube = 1.60) 
    (h_cube_edge : cube_edge = 10) : paint_cost = 3.20 := by 
  sorry

end cost_per_quart_l674_67461


namespace gas_station_constant_l674_67408

structure GasStationData where
  amount : ℝ
  unit_price : ℝ
  price_per_yuan_per_liter : ℝ

theorem gas_station_constant (data : GasStationData) (h1 : data.amount = 116.64) (h2 : data.unit_price = 18) (h3 : data.price_per_yuan_per_liter = 6.48) : data.unit_price = 18 :=
sorry

end gas_station_constant_l674_67408


namespace max_sum_when_product_is_399_l674_67401

theorem max_sum_when_product_is_399 :
  ∃ (X Y Z : ℕ), X * Y * Z = 399 ∧ X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X ∧ X + Y + Z = 29 :=
by
  sorry

end max_sum_when_product_is_399_l674_67401


namespace pentagon_diagl_sum_pentagon_diagonal_391_l674_67421

noncomputable def diagonal_sum (AB CD BC DE AE : ℕ) 
  (AC : ℚ) (BD : ℚ) (CE : ℚ) (AD : ℚ) (BE : ℚ) : ℚ :=
  3 * AC + AD + BE

theorem pentagon_diagl_sum (AB CD BC DE AE : ℕ)
  (hAB : AB = 3) (hCD : CD = 3) 
  (hBC : BC = 10) (hDE : DE = 10) 
  (hAE : AE = 14)
  (AC BD CE AD BE : ℚ)
  (hACBC : AC = 12) 
  (hADBC: AD = 13.5)
  (hCEBE: BE = 44 / 3) :
  diagonal_sum AB CD BC DE AE AC BD CE AD BE = 385 / 6 := sorry

theorem pentagon_diagonal_391 (AB CD BC DE AE : ℕ)
  (hAB : AB = 3) (hCD : CD = 3) 
  (hBC : BC = 10) (hDE : DE = 10) 
  (hAE : AE = 14)
  (AC BD CE AD BE : ℚ)
  (hACBC : AC = 12) 
  (hADBC: AD = 13.5)
  (hCEBE: BE = 44 / 3) :
  ∃ m n : ℕ, 
    m.gcd n = 1 ∧
    m / n = 385 / 6 ∧
    m + n = 391 := sorry

end pentagon_diagl_sum_pentagon_diagonal_391_l674_67421


namespace rectangular_prism_volume_increase_l674_67484

theorem rectangular_prism_volume_increase (L B H : ℝ) :
  let V_original := L * B * H
  let L_new := L * 1.07
  let B_new := B * 1.18
  let H_new := H * 1.25
  let V_new := L_new * B_new * H_new
  let increase_in_volume := (V_new - V_original) / V_original * 100
  increase_in_volume = 56.415 :=
by
  sorry

end rectangular_prism_volume_increase_l674_67484


namespace log_12_eq_2a_plus_b_l674_67487

variable (lg : ℝ → ℝ)
variable (lg_2_eq_a : lg 2 = a)
variable (lg_3_eq_b : lg 3 = b)

theorem log_12_eq_2a_plus_b : lg 12 = 2 * a + b :=
by
  sorry

end log_12_eq_2a_plus_b_l674_67487


namespace sin_product_eq_one_sixteenth_l674_67460

theorem sin_product_eq_one_sixteenth : 
  (Real.sin (12 * Real.pi / 180)) * 
  (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (54 * Real.pi / 180)) * 
  (Real.sin (78 * Real.pi / 180)) = 
  1 / 16 := 
sorry

end sin_product_eq_one_sixteenth_l674_67460


namespace simplified_evaluated_expression_l674_67410

noncomputable def a : ℚ := 1 / 3
noncomputable def b : ℚ := 1 / 2
noncomputable def c : ℚ := 1

def expression (a b c : ℚ) : ℚ := a^2 + 2 * b - c

theorem simplified_evaluated_expression :
  expression a b c = 1 / 9 :=
by
  sorry

end simplified_evaluated_expression_l674_67410


namespace nancy_carrots_l674_67497

def carrots_total 
  (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

theorem nancy_carrots : 
  carrots_total 12 2 21 = 31 :=
by
  -- Add the proof here
  sorry

end nancy_carrots_l674_67497


namespace total_study_hours_during_semester_l674_67465

-- Definitions of the given conditions
def semester_weeks : ℕ := 15
def weekday_study_hours_per_day : ℕ := 3
def saturday_study_hours : ℕ := 4
def sunday_study_hours : ℕ := 5

-- Theorem statement to prove the total study hours during the semester
theorem total_study_hours_during_semester : 
  (semester_weeks * ((5 * weekday_study_hours_per_day) + saturday_study_hours + sunday_study_hours)) = 360 := by
  -- We are skipping the proof step and adding a placeholder
  sorry

end total_study_hours_during_semester_l674_67465


namespace ellipse_problem_part1_ellipse_problem_part2_l674_67444

-- Statement of the problem
theorem ellipse_problem_part1 :
  ∃ k : ℝ, (∀ x y : ℝ, (x^2 / 2) + y^2 = 1 → (
    (∃ t > 0, x = t * y + 1) → k = (Real.sqrt 2) / 2)) :=
sorry

theorem ellipse_problem_part2 :
  ∃ S_max : ℝ, ∀ (t : ℝ), (t > 0 → (S_max = (4 * (t^2 + 1)^2) / ((t^2 + 2) * (2 * t^2 + 1)))) → t^2 = 1 → S_max = 16 / 9 :=
sorry

end ellipse_problem_part1_ellipse_problem_part2_l674_67444


namespace ideal_type_circle_D_l674_67456

-- Define the line equation
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the distance condition for circles
def ideal_type_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ (P Q : ℝ × ℝ), 
    line_l P.1 P.2 ∧ line_l Q.1 Q.2 ∧
    dist P (0, 0) = radius ∧
    dist Q (0, 0) = radius ∧
    dist (P, Q) = 1

-- Definition of given circles A, B, C, D
def circle_A (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_B (x y : ℝ) : Prop := x^2 + y^2 = 16
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + (y - 4)^2 = 1
def circle_D (x y : ℝ) : Prop := (x - 4)^2 + (y - 4)^2 = 16

-- Define circle centers and radii for A, B, C, D
def center_A : ℝ × ℝ := (0, 0)
def radius_A : ℝ := 1
def center_B : ℝ × ℝ := (0, 0)
def radius_B : ℝ := 4
def center_C : ℝ × ℝ := (4, 4)
def radius_C : ℝ := 1
def center_D : ℝ × ℝ := (4, 4)
def radius_D : ℝ := 4

-- Problem Statement: Prove that option D is the "ideal type" circle
theorem ideal_type_circle_D : 
  ideal_type_circle center_D radius_D :=
sorry

end ideal_type_circle_D_l674_67456


namespace find_f2_l674_67477

-- Definitions based on the given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable (f g : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_g_def : ∀ x, g x = f x + 9)
variable (h_g_val : g (-2) = 3)

-- Prove the required goal
theorem find_f2 : f 2 = 6 :=
by
  sorry

end find_f2_l674_67477


namespace goods_train_speed_l674_67452

theorem goods_train_speed (man_train_speed_kmh : Float) 
    (goods_train_length_m : Float) 
    (passing_time_s : Float) 
    (kmh_to_ms : Float := 1000 / 3600) : 
    man_train_speed_kmh = 50 → 
    goods_train_length_m = 280 → 
    passing_time_s = 9 → 
    Float.round ((goods_train_length_m / passing_time_s + man_train_speed_kmh * kmh_to_ms) * 3600 / 1000) = 61.99
:= by
  sorry

end goods_train_speed_l674_67452


namespace moles_of_water_l674_67459

-- Definitions related to the reaction conditions.
def HCl : Type := sorry
def NaHCO3 : Type := sorry
def NaCl : Type := sorry
def H2O : Type := sorry
def CO2 : Type := sorry

def reaction (h : HCl) (n : NaHCO3) : Nat := sorry -- Represents the balanced reaction

-- Given conditions in Lean.
axiom one_mole_HCl : HCl
axiom one_mole_NaHCO3 : NaHCO3
axiom balanced_equation : reaction one_mole_HCl one_mole_NaHCO3 = 1 -- 1 mole of water is produced

-- The theorem to prove.
theorem moles_of_water : reaction one_mole_HCl one_mole_NaHCO3 = 1 :=
by
  -- The proof would go here
  sorry

end moles_of_water_l674_67459


namespace average_a_b_l674_67427

theorem average_a_b (a b : ℝ) (h : (4 + 6 + 8 + a + b) / 5 = 20) : (a + b) / 2 = 41 :=
by
  sorry

end average_a_b_l674_67427


namespace optionA_optionB_optionC_optionD_l674_67483

-- Statement for option A
theorem optionA : (∀ x : ℝ, x ≠ 3 → x^2 - 4 * x + 3 ≠ 0) ↔ (x^2 - 4 * x + 3 = 0 → x = 3) := sorry

-- Statement for option B
theorem optionB : (¬ (∀ x : ℝ, x^2 - x + 2 > 0) ↔ ∃ x0 : ℝ, x0^2 - x0 + 2 ≤ 0) := sorry

-- Statement for option C
theorem optionC (p q : Prop) : p ∧ q → p ∧ q := sorry

-- Statement for option D
theorem optionD (x : ℝ) : (x > -1 → x^2 + 4 * x + 3 > 0) ∧ ¬ (∀ x : ℝ, x^2 + 4 * x + 3 > 0 → x > -1) := sorry

end optionA_optionB_optionC_optionD_l674_67483


namespace find_k_l674_67406

-- Definitions based on given conditions
def ellipse_equation (x y : ℝ) (k : ℝ) : Prop :=
  5 * x^2 + k * y^2 = 5

def is_focus (x y : ℝ) : Prop :=
  x = 0 ∧ y = 2

-- Statement of the problem
theorem find_k (k : ℝ) :
  (∀ x y, ellipse_equation x y k) →
  is_focus 0 2 →
  k = 1 :=
sorry

end find_k_l674_67406


namespace b_minus_a_l674_67466

theorem b_minus_a :
  ∃ (a b : ℝ), (2 + 4 = -a) ∧ (2 * 4 = b) ∧ (b - a = 14) :=
by
  use (-6 : ℝ)
  use (8 : ℝ)
  simp
  sorry

end b_minus_a_l674_67466


namespace tom_saves_80_dollars_l674_67495

def normal_doctor_cost : ℝ := 200
def discount_percentage : ℝ := 0.7
def discount_clinic_cost_per_visit : ℝ := normal_doctor_cost * (1 - discount_percentage)
def number_of_visits : ℝ := 2
def total_discount_clinic_cost : ℝ := discount_clinic_cost_per_visit * number_of_visits
def savings : ℝ := normal_doctor_cost - total_discount_clinic_cost

theorem tom_saves_80_dollars : savings = 80 := by
  sorry

end tom_saves_80_dollars_l674_67495


namespace chocolate_distribution_l674_67454

theorem chocolate_distribution
  (total_chocolate : ℚ)
  (num_piles : ℕ)
  (piles_given_to_shaina : ℕ)
  (weight_each_pile : ℚ)
  (weight_of_shaina_piles : ℚ)
  (h1 : total_chocolate = 72 / 7)
  (h2 : num_piles = 6)
  (h3 : piles_given_to_shaina = 2)
  (h4 : weight_each_pile = total_chocolate / num_piles)
  (h5 : weight_of_shaina_piles = piles_given_to_shaina * weight_each_pile) :
  weight_of_shaina_piles = 24 / 7 := by
  sorry

end chocolate_distribution_l674_67454


namespace find_p_l674_67453

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_p (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) (h1 : p + q = r + 2) (h2 : 1 < p) (h3 : p < q) :
  p = 2 := 
sorry

end find_p_l674_67453


namespace ratio_greater_than_two_ninths_l674_67455

-- Define the conditions
def M : ℕ := 8
def N : ℕ := 36

-- State the theorem
theorem ratio_greater_than_two_ninths : (M : ℚ) / (N : ℚ) > 2 / 9 := 
by {
    -- skipping the proof with sorry
    sorry
}

end ratio_greater_than_two_ninths_l674_67455


namespace gcd_101_power_l674_67440

theorem gcd_101_power (a b : ℕ) (h1 : a = 101^6 + 1) (h2 : b = 3 * 101^6 + 101^3 + 1) (h_prime : Nat.Prime 101) : Nat.gcd a b = 1 :=
by
  -- proof goes here
  sorry

end gcd_101_power_l674_67440


namespace rebecca_eggs_l674_67443

/-- Rebecca wants to split a collection of eggs into 4 groups. Each group will have 2 eggs. -/
def number_of_groups : Nat := 4

def eggs_per_group : Nat := 2

theorem rebecca_eggs : (number_of_groups * eggs_per_group) = 8 := by
  sorry

end rebecca_eggs_l674_67443


namespace largest_angle_in_ratio_3_4_5_l674_67448

theorem largest_angle_in_ratio_3_4_5 : ∃ (A B C : ℝ), (A / 3 = B / 4 ∧ B / 4 = C / 5) ∧ (A + B + C = 180) ∧ (C = 75) :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l674_67448


namespace zero_function_solution_l674_67445

theorem zero_function_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^3 + y^3) = f (x^3) + 3 * x^2 * f (x) * f (y) + 3 * (f (x) * f (y))^2 + y^6 * f (y)) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end zero_function_solution_l674_67445


namespace cookies_number_l674_67490

-- Define all conditions in the problem
def number_of_chips_per_cookie := 7
def number_of_cookies_per_dozen := 12
def number_of_uneaten_chips := 168

-- Define D as the number of dozens of cookies
variable (D : ℕ)

-- Prove the Lean theorem
theorem cookies_number (h : 7 * 6 * D = 168) : D = 4 :=
by
  sorry

end cookies_number_l674_67490


namespace original_savings_eq_920_l674_67478

variable (S : ℝ) -- Define S as a real number representing Linda's savings
variable (h1 : S * (1 / 4) = 230) -- Given condition

theorem original_savings_eq_920 :
  S = 920 :=
by
  sorry

end original_savings_eq_920_l674_67478


namespace fractions_addition_l674_67496

theorem fractions_addition :
  (1 / 3) * (3 / 4) * (1 / 5) + (1 / 6) = 13 / 60 :=
by 
  sorry

end fractions_addition_l674_67496


namespace sin_A_value_of_triangle_l674_67450

theorem sin_A_value_of_triangle 
  (a b : ℝ) (A B C : ℝ) (h_triangle : a = 2) (h_b : b = 3) (h_tanB : Real.tan B = 3) :
  Real.sin A = Real.sqrt 10 / 5 :=
sorry

end sin_A_value_of_triangle_l674_67450


namespace diagonals_of_60_sided_polygon_exterior_angle_of_60_sided_polygon_l674_67489

noncomputable def diagonals_in_regular_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

noncomputable def exterior_angle (n : ℕ) : ℝ :=
  360.0 / n

theorem diagonals_of_60_sided_polygon :
  diagonals_in_regular_polygon 60 = 1710 :=
by
  sorry

theorem exterior_angle_of_60_sided_polygon :
  exterior_angle 60 = 6.0 :=
by
  sorry

end diagonals_of_60_sided_polygon_exterior_angle_of_60_sided_polygon_l674_67489


namespace range_of_a_l674_67486

-- Define the inequality condition
def condition (a : ℝ) (x : ℝ) : Prop := abs (a - 2 * x) > x - 1

-- Define the range for x
def in_range (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the main theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, in_range x → condition a x) ↔ (a < 2 ∨ 5 < a) := 
by
  sorry

end range_of_a_l674_67486


namespace winnie_the_pooh_wins_l674_67439

variable (cones : ℕ)

def can_guarantee_win (initial_cones : ℕ) : Prop :=
  ∃ strategy : (ℕ → ℕ), 
    (strategy initial_cones = 4 ∨ strategy initial_cones = 1) ∧ 
    ∀ n, (strategy n = 1 → (n = 2012 - 4 ∨ n = 2007 - 1 ∨ n = 2005 - 1)) ∧
         (strategy n = 4 → n = 2012)

theorem winnie_the_pooh_wins : can_guarantee_win 2012 :=
sorry

end winnie_the_pooh_wins_l674_67439


namespace no_real_solutions_l674_67411

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else (2 - x^2) / x

theorem no_real_solutions :
  (∀ x : ℝ, x ≠ 0 → (f x + 2 * f (1 / x) = 3 * x)) →
  (∀ x : ℝ, f x = f (-x) → false) :=
by
  intro h1 h2
  sorry

end no_real_solutions_l674_67411


namespace combination_lock_code_l674_67481

theorem combination_lock_code :
  ∀ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (x + y + x * y = 10 * x + y) →
  10 * x + y = 19 ∨ 10 * x + y = 29 ∨ 10 * x + y = 39 ∨ 10 * x + y = 49 ∨
  10 * x + y = 59 ∨ 10 * x + y = 69 ∨ 10 * x + y = 79 ∨ 10 * x + y = 89 ∨
  10 * x + y = 99 :=
by
  sorry

end combination_lock_code_l674_67481


namespace sarah_trucks_l674_67499

-- Define the initial number of trucks denoted by T
def initial_trucks (T : ℝ) : Prop :=
  let left_after_jeff := T - 13.5
  let left_after_ashley := left_after_jeff - 0.25 * left_after_jeff
  left_after_ashley = 38

-- Theorem stating the initial number of trucks Sarah had is 64
theorem sarah_trucks : ∃ T : ℝ, initial_trucks T ∧ T = 64 :=
by
  sorry

end sarah_trucks_l674_67499


namespace statement_A_statement_C_statement_D_l674_67431

theorem statement_A (x : ℝ) :
  (¬ (∀ x ≥ 3, 2 * x - 10 ≥ 0)) ↔ (∃ x0 ≥ 3, 2 * x0 - 10 < 0) := 
sorry

theorem statement_C {a b c : ℝ} (h1 : c > a) (h2 : a > b) (h3 : b > 0) :
  (a / (c - a)) > (b / (c - b)) := 
sorry

theorem statement_D {a b m : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  (a / b) > ((a + m) / (b + m)) := 
sorry

end statement_A_statement_C_statement_D_l674_67431


namespace intersection_M_N_eq_neg2_l674_67463

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l674_67463


namespace smallest_base10_integer_l674_67457

theorem smallest_base10_integer :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (2 * a + 2 = 3 * b + 3) ∧ (2 * a + 2 = 18) :=
by
  existsi 8 -- assign specific solutions to a
  existsi 5 -- assign specific solutions to b
  exact sorry -- follows from the validations done above

end smallest_base10_integer_l674_67457


namespace no_positive_solution_l674_67475

theorem no_positive_solution (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) :
  ¬ (∀ n ≥ 2, a (n + 2) = a n - a (n - 1)) :=
sorry

end no_positive_solution_l674_67475


namespace number_of_bird_cages_l674_67415

-- Definitions for the problem conditions
def birds_per_cage : ℕ := 2 + 7
def total_birds : ℕ := 72

-- The theorem to prove the number of bird cages is 8
theorem number_of_bird_cages : total_birds / birds_per_cage = 8 := by
  sorry

end number_of_bird_cages_l674_67415


namespace bc_sum_l674_67417

theorem bc_sum (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 10) : B + C = 310 := by
  sorry

end bc_sum_l674_67417


namespace find_height_of_cylinder_l674_67423

theorem find_height_of_cylinder (h r : ℝ) (π : ℝ) (SA : ℝ) (r_val : r = 3) (SA_val : SA = 36 * π) 
  (SA_formula : SA = 2 * π * r^2 + 2 * π * r * h) : h = 3 := 
by
  sorry

end find_height_of_cylinder_l674_67423


namespace cylinder_volume_l674_67418

-- Define the volume of the cone
def V_cone : ℝ := 18.84

-- Define the volume of the cylinder
def V_cylinder : ℝ := 3 * V_cone

-- Prove that the volume of the cylinder is 56.52 cubic meters
theorem cylinder_volume :
  V_cylinder = 56.52 := 
by 
  -- the proof will go here
  sorry

end cylinder_volume_l674_67418


namespace find_vertex_A_l674_67449

variables (B C: ℝ × ℝ × ℝ)

-- Defining midpoints conditions
def midpoint_BC : ℝ × ℝ × ℝ := (1, 5, -1)
def midpoint_AC : ℝ × ℝ × ℝ := (0, 4, -2)
def midpoint_AB : ℝ × ℝ × ℝ := (2, 3, 4)

-- The coordinates of point A we need to prove
def target_A : ℝ × ℝ × ℝ := (1, 2, 3)

-- Lean statement proving the coordinates of A
theorem find_vertex_A (A B C : ℝ × ℝ × ℝ)
  (hBC : midpoint_BC = (1, 5, -1))
  (hAC : midpoint_AC = (0, 4, -2))
  (hAB : midpoint_AB = (2, 3, 4)) :
  A = (1, 2, 3) := 
sorry

end find_vertex_A_l674_67449


namespace problem_inequality_l674_67430

theorem problem_inequality 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b)
  (c_pos : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 := 
sorry

end problem_inequality_l674_67430


namespace Lisa_pay_per_hour_is_15_l674_67451

-- Given conditions:
def Greta_hours : ℕ := 40
def Greta_pay_per_hour : ℕ := 12
def Lisa_hours : ℕ := 32

-- Define Greta's earnings based on the given conditions:
def Greta_earnings : ℕ := Greta_hours * Greta_pay_per_hour

-- The main statement to prove:
theorem Lisa_pay_per_hour_is_15 (h1 : Greta_earnings = Greta_hours * Greta_pay_per_hour) 
                                (h2 : Greta_earnings = Lisa_hours * L) :
  L = 15 :=
by sorry

end Lisa_pay_per_hour_is_15_l674_67451


namespace train_length_l674_67494

theorem train_length (speed_faster speed_slower : ℝ) (time_sec : ℝ) (length_each_train : ℝ) :
  speed_faster = 47 ∧ speed_slower = 36 ∧ time_sec = 36 ∧ 
  (length_each_train = 55 ↔ 2 * length_each_train = ((speed_faster - speed_slower) * (1000/3600) * time_sec)) :=
by {
  -- We declare the speeds in km/hr and convert the relative speed to m/s for calculation.
  sorry
}

end train_length_l674_67494


namespace pens_in_shop_l674_67442

theorem pens_in_shop (P Pe E : ℕ) (h_ratio : 14 * Pe = 4 * P) (h_ratio2 : 14 * E = 14 * 3 + 11) (h_P : P = 140) (h_E : E = 30) : Pe = 40 :=
sorry

end pens_in_shop_l674_67442


namespace find_a_l674_67446

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a - 2, a^2 + 4*a, 10}) (h : -3 ∈ A) : a = -3 := 
by
  -- placeholder proof
  sorry

end find_a_l674_67446


namespace number_of_pots_of_rosemary_l674_67437

-- Definitions based on the conditions
def total_leaves_basil (pots_basil : ℕ) (leaves_per_basil : ℕ) : ℕ := pots_basil * leaves_per_basil
def total_leaves_rosemary (pots_rosemary : ℕ) (leaves_per_rosemary : ℕ) : ℕ := pots_rosemary * leaves_per_rosemary
def total_leaves_thyme (pots_thyme : ℕ) (leaves_per_thyme : ℕ) : ℕ := pots_thyme * leaves_per_thyme

-- The given problem conditions
def pots_basil : ℕ := 3
def leaves_per_basil : ℕ := 4
def leaves_per_rosemary : ℕ := 18
def pots_thyme : ℕ := 6
def leaves_per_thyme : ℕ := 30
def total_leaves : ℕ := 354

-- Proving the number of pots of rosemary
theorem number_of_pots_of_rosemary : 
  ∃ (pots_rosemary : ℕ), 
  total_leaves_basil pots_basil leaves_per_basil + 
  total_leaves_rosemary pots_rosemary leaves_per_rosemary + 
  total_leaves_thyme pots_thyme leaves_per_thyme = 
  total_leaves ∧ pots_rosemary = 9 :=
by
  sorry  -- proof is omitted

end number_of_pots_of_rosemary_l674_67437


namespace charity_meaning_l674_67479

theorem charity_meaning (noun_charity : String) (h : noun_charity = "charity") : 
  (noun_charity = "charity" → "charity" = "charitable organization") :=
by
  sorry

end charity_meaning_l674_67479


namespace coffee_mix_price_per_pound_l674_67485

-- Definitions based on conditions
def total_weight : ℝ := 100
def columbian_price_per_pound : ℝ := 8.75
def brazilian_price_per_pound : ℝ := 3.75
def columbian_weight : ℝ := 52
def brazilian_weight : ℝ := total_weight - columbian_weight

-- Goal to prove
theorem coffee_mix_price_per_pound :
  (columbian_weight * columbian_price_per_pound + brazilian_weight * brazilian_price_per_pound) / total_weight = 6.35 :=
by
  sorry

end coffee_mix_price_per_pound_l674_67485


namespace division_remainder_3012_97_l674_67400

theorem division_remainder_3012_97 : 3012 % 97 = 5 := 
by 
  sorry

end division_remainder_3012_97_l674_67400


namespace sphere_surface_area_l674_67412

theorem sphere_surface_area (a b c : ℝ)
  (h1 : a * b * c = Real.sqrt 6)
  (h2 : a * b = Real.sqrt 2)
  (h3 : b * c = Real.sqrt 3) :
  4 * Real.pi * (Real.sqrt (a^2 + b^2 + c^2) / 2) ^ 2 = 6 * Real.pi :=
sorry

end sphere_surface_area_l674_67412


namespace total_towels_l674_67405

theorem total_towels (packs : ℕ) (towels_per_pack : ℕ) (h1 : packs = 9) (h2 : towels_per_pack = 3) : packs * towels_per_pack = 27 := by
  sorry

end total_towels_l674_67405


namespace price_of_other_pieces_l674_67436

theorem price_of_other_pieces (total_spent : ℕ) (total_pieces : ℕ) (price_piece1 : ℕ) (price_piece2 : ℕ) 
  (remaining_pieces : ℕ) (price_remaining_piece : ℕ) (h1 : total_spent = 610) (h2 : total_pieces = 7)
  (h3 : price_piece1 = 49) (h4 : price_piece2 = 81) (h5 : remaining_pieces = (total_pieces - 2))
  (h6 : total_spent - price_piece1 - price_piece2 = remaining_pieces * price_remaining_piece) :
  price_remaining_piece = 96 := 
by
  sorry

end price_of_other_pieces_l674_67436


namespace max_k_consecutive_sum_l674_67438

theorem max_k_consecutive_sum :
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, k * (2 * n + k - 1) = 2^2 * 3^8 ∧ ∀ k' > k, ¬ ∃ n', n' > 0 ∧ k' * (2 * n' + k' - 1) = 2^2 * 3^8 := sorry

end max_k_consecutive_sum_l674_67438


namespace find_other_endpoint_of_diameter_l674_67471

theorem find_other_endpoint_of_diameter 
    (center endpoint : ℝ × ℝ) 
    (h_center : center = (5, -2)) 
    (h_endpoint : endpoint = (2, 3))
    : (center.1 + (center.1 - endpoint.1), center.2 + (center.2 - endpoint.2)) = (8, -7) := 
by
  sorry

end find_other_endpoint_of_diameter_l674_67471
