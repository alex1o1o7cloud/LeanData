import Mathlib

namespace circle_standard_equation_l262_26204

theorem circle_standard_equation (a : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - 1)^2 = (x - 1 + y - 1)^2) ∧
  (∀ x y : ℝ, (x - a)^2 + (y - 1)^2 = (x - 1 + y + 2)^2) →
  (∃ x y : ℝ, (x - 2) ^ 2 + (y - 1) ^ 2 = 2) :=
sorry

end circle_standard_equation_l262_26204


namespace imaginary_unit_multiplication_l262_26272

-- Statement of the problem   
theorem imaginary_unit_multiplication (i : ℂ) (hi : i ^ 2 = -1) : i * (1 + i) = -1 + i :=
by sorry

end imaginary_unit_multiplication_l262_26272


namespace evaluate_expression_l262_26279

theorem evaluate_expression : (900^2 / (153^2 - 147^2)) = 450 := by
  sorry

end evaluate_expression_l262_26279


namespace closest_square_to_350_l262_26271

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l262_26271


namespace focus_of_parabola_l262_26290

def parabola (x : ℝ) : ℝ := (x - 3) ^ 2

theorem focus_of_parabola :
  ∃ f : ℝ × ℝ, f = (3, 1 / 4) ∧
  ∀ x : ℝ, parabola x = (x - 3)^2 :=
sorry

end focus_of_parabola_l262_26290


namespace units_digit_13_times_41_l262_26286

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_13_times_41 :
  units_digit (13 * 41) = 3 :=
sorry

end units_digit_13_times_41_l262_26286


namespace min_value_expression_l262_26241

theorem min_value_expression (m n : ℝ) (h : m > 2 * n) :
  m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) ≥ 6 :=
by
  sorry

end min_value_expression_l262_26241


namespace inverse_graph_pass_point_l262_26256

variable {f : ℝ → ℝ}
variable {f_inv : ℝ → ℝ}

noncomputable def satisfies_inverse (f f_inv : ℝ → ℝ) : Prop :=
∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

theorem inverse_graph_pass_point
  (hf : satisfies_inverse f f_inv)
  (h_point : (1 : ℝ) - f 1 = 3) :
  f_inv (-2) + 3 = 4 :=
by
  sorry

end inverse_graph_pass_point_l262_26256


namespace geometric_sequence_a8_l262_26230

theorem geometric_sequence_a8 {a : ℕ → ℝ} (h1 : a 1 * a 3 = 4) (h9 : a 9 = 256) :
  a 8 = 128 ∨ a 8 = -128 :=
sorry

end geometric_sequence_a8_l262_26230


namespace candy_cooking_time_l262_26273

def initial_temperature : ℝ := 60
def peak_temperature : ℝ := 240
def final_temperature : ℝ := 170
def heating_rate : ℝ := 5
def cooling_rate : ℝ := 7

theorem candy_cooking_time : ( (peak_temperature - initial_temperature) / heating_rate + (peak_temperature - final_temperature) / cooling_rate ) = 46 := by
  sorry

end candy_cooking_time_l262_26273


namespace total_pennies_l262_26268

theorem total_pennies (R G K : ℕ) (h1 : R = 180) (h2 : G = R / 2) (h3 : K = G / 3) : R + G + K = 300 := by
  sorry

end total_pennies_l262_26268


namespace nancy_target_amount_l262_26295

theorem nancy_target_amount {rate : ℝ} {hours : ℝ} (h1 : rate = 28 / 4) (h2 : hours = 10) : 28 / 4 * 10 = 70 :=
by
  sorry

end nancy_target_amount_l262_26295


namespace max_result_of_operation_l262_26267

theorem max_result_of_operation : ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 → 3 * (300 - m) ≤ 870) ∧ 3 * (300 - n) = 870 :=
by
  sorry

end max_result_of_operation_l262_26267


namespace water_charge_rel_water_usage_from_charge_l262_26282

-- Define the conditions and functional relationship
theorem water_charge_rel (x : ℝ) (hx : x > 5) : y = 3.5 * x - 7.5 :=
  sorry

-- Prove the specific case where the charge y is 17 yuan
theorem water_usage_from_charge (h : 17 = 3.5 * x - 7.5) :
  x = 7 :=
  sorry

end water_charge_rel_water_usage_from_charge_l262_26282


namespace isosceles_right_triangle_angle_l262_26254

-- Define the conditions given in the problem
def is_isosceles (a b c : ℝ) : Prop := 
(a = b ∨ b = c ∨ c = a)

def is_right_triangle (a b c : ℝ) : Prop := 
(a = 90 ∨ b = 90 ∨ c = 90)

def angles_sum_to_180 (a b c : ℝ) : Prop :=
a + b + c = 180

-- The Proof Problem
theorem isosceles_right_triangle_angle :
  ∀ (a b c x : ℝ), (is_isosceles a b c) → (is_right_triangle a b c) → (angles_sum_to_180 a b c) → (x = a ∨ x = b ∨ x = c) → x = 45 :=
by
  intros a b c x h_isosceles h_right h_sum h_x
  -- Proof is omitted with sorry
  sorry

end isosceles_right_triangle_angle_l262_26254


namespace geometric_sequence_nec_suff_l262_26270

theorem geometric_sequence_nec_suff (a b c : ℝ) : (b^2 = a * c) ↔ (∃ r : ℝ, b = a * r ∧ c = b * r) :=
sorry

end geometric_sequence_nec_suff_l262_26270


namespace bottles_per_case_correct_l262_26251

-- Define the conditions given in the problem
def daily_bottle_production : ℕ := 120000
def number_of_cases_needed : ℕ := 10000

-- Define the expected answer
def bottles_per_case : ℕ := 12

-- The statement we need to prove
theorem bottles_per_case_correct :
  daily_bottle_production / number_of_cases_needed = bottles_per_case :=
by
  -- Leap of logic: actually solving this for correctness is here considered a leap
  sorry

end bottles_per_case_correct_l262_26251


namespace smallest_6_digit_div_by_111_l262_26253

theorem smallest_6_digit_div_by_111 : ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 ∧ n = 100011 := by
  sorry

end smallest_6_digit_div_by_111_l262_26253


namespace extreme_points_l262_26228

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem extreme_points (a b : ℝ) 
  (h1 : 3*(-2)^2 + 2*a*(-2) + b = 0) 
  (h2 : 3*(4)^2 + 2*a*(4) + b = 0) : 
  a - b = 21 :=
by sorry

end extreme_points_l262_26228


namespace rectangle_dimension_correct_l262_26238

-- Definition of the Width and Length based on given conditions
def width := 3 / 2
def length := 3

-- Perimeter and Area conditions
def perimeter_condition (w l : ℝ) := 2 * (w + l) = 2 * (w * l)
def length_condition (w l : ℝ) := l = 2 * w

-- Main theorem statement
theorem rectangle_dimension_correct :
  ∃ (w l : ℝ), perimeter_condition w l ∧ length_condition w l ∧ w = width ∧ l = length :=
by {
  -- add sorry to skip the proof
  sorry
}

end rectangle_dimension_correct_l262_26238


namespace angle_sum_l262_26260

theorem angle_sum (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 3 / 4)
  (sin_β : Real.sin β = 3 / 5) :
  α + 3 * β = 5 * Real.pi / 4 := 
sorry

end angle_sum_l262_26260


namespace b_payment_l262_26293

theorem b_payment (b_days : ℕ) (a_days : ℕ) (total_wages : ℕ) (b_payment : ℕ) :
  b_days = 10 →
  a_days = 15 →
  total_wages = 5000 →
  b_payment = 3000 :=
by
  intros h1 h2 h3
  -- conditions
  have hb := h1
  have ha := h2
  have ht := h3
  -- skipping proof
  sorry

end b_payment_l262_26293


namespace min_value_binom_l262_26294

theorem min_value_binom
  (a b : ℕ → ℕ)
  (n : ℕ) (hn : 0 < n)
  (h1 : ∀ n, a n = 2^n)
  (h2 : ∀ n, b n = 4^n) :
  ∃ n, 2^n + (1 / 2^n) = 5 / 2 :=
sorry

end min_value_binom_l262_26294


namespace trigonometric_identity_proof_l262_26200

noncomputable def four_sin_40_minus_tan_40 : ℝ :=
  4 * Real.sin (40 * Real.pi / 180) - Real.tan (40 * Real.pi / 180)

theorem trigonometric_identity_proof : four_sin_40_minus_tan_40 = Real.sqrt 3 := by
  sorry

end trigonometric_identity_proof_l262_26200


namespace perp_bisector_eq_l262_26246

noncomputable def C1 := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 6 * p.1 - 7 = 0 }
noncomputable def C2 := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 6 * p.2 - 27 = 0 }

theorem perp_bisector_eq :
  ∃ x y, ( (x, y) ∈ C1 ∧ (x, y) ∈ C2 ) -> ( x - y = 0 ) :=
by
  sorry

end perp_bisector_eq_l262_26246


namespace minimize_sum_m_n_l262_26244

-- Definitions of the given conditions
def last_three_digits_equal (a b : ℕ) : Prop :=
  (a % 1000) = (b % 1000)

-- The main statement to prove
theorem minimize_sum_m_n (m n : ℕ) (h1 : n > m) (h2 : 1 ≤ m) 
  (h3 : last_three_digits_equal (1978^n) (1978^m)) : m + n = 106 :=
sorry

end minimize_sum_m_n_l262_26244


namespace train_cross_time_l262_26237

noncomputable def train_length : ℝ := 130
noncomputable def train_speed_kph : ℝ := 45
noncomputable def total_length : ℝ := 375

noncomputable def speed_mps := train_speed_kph * 1000 / 3600
noncomputable def distance := train_length + total_length

theorem train_cross_time : (distance / speed_mps) = 30 := by
  sorry

end train_cross_time_l262_26237


namespace sumOddDivisorsOf90_l262_26231

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l262_26231


namespace apples_used_l262_26285

theorem apples_used (x : ℕ) 
  (initial_apples : ℕ := 23) 
  (bought_apples : ℕ := 6) 
  (final_apples : ℕ := 9) 
  (h : (initial_apples - x) + bought_apples = final_apples) : 
  x = 20 :=
by
  sorry

end apples_used_l262_26285


namespace ratio_of_juniors_to_freshmen_l262_26245

variables (f j : ℕ) 

theorem ratio_of_juniors_to_freshmen (h1 : (1/4 : ℚ) * f = (1/2 : ℚ) * j) :
  j = f / 2 :=
by
  sorry

end ratio_of_juniors_to_freshmen_l262_26245


namespace expression_defined_if_x_not_3_l262_26249

theorem expression_defined_if_x_not_3 (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end expression_defined_if_x_not_3_l262_26249


namespace rectangle_area_y_l262_26214

theorem rectangle_area_y (y : ℝ) (h_y_pos : y > 0)
  (h_area : (3 * y = 21)) : y = 7 :=
by
  sorry

end rectangle_area_y_l262_26214


namespace jane_ate_four_pieces_l262_26218

def total_pieces : ℝ := 12.0
def num_people : ℝ := 3.0
def pieces_per_person : ℝ := 4.0

theorem jane_ate_four_pieces :
  total_pieces / num_people = pieces_per_person := 
  by
    sorry

end jane_ate_four_pieces_l262_26218


namespace no_integer_k_sq_plus_k_plus_one_divisible_by_101_l262_26278

theorem no_integer_k_sq_plus_k_plus_one_divisible_by_101 (k : ℤ) : 
  (k^2 + k + 1) % 101 ≠ 0 := 
by
  sorry

end no_integer_k_sq_plus_k_plus_one_divisible_by_101_l262_26278


namespace arccos_zero_l262_26216

theorem arccos_zero : Real.arccos 0 = Real.pi / 2 := 
by 
  sorry

end arccos_zero_l262_26216


namespace tenth_pair_in_twentieth_row_l262_26276

def nth_pair_in_row (n k : ℕ) : ℕ × ℕ :=
  if h : n > 0 ∧ k > 0 ∧ n >= k then (k, n + 1 - k)
  else (0, 0) -- define (0,0) as a default for invalid inputs

theorem tenth_pair_in_twentieth_row : nth_pair_in_row 20 10 = (10, 11) :=
by sorry

end tenth_pair_in_twentieth_row_l262_26276


namespace distance_between_points_l262_26283

open Real -- opening real number namespace

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * cos θ, r * sin θ)

theorem distance_between_points :
  let A := polar_to_cartesian 2 (π / 3)
  let B := polar_to_cartesian 2 (2 * π / 3)
  dist A B = 2 :=
by
  sorry

end distance_between_points_l262_26283


namespace paths_E_to_G_through_F_and_H_l262_26203

-- Define positions of E, F, H, and G on the grid.
structure Point where
  x : ℕ
  y : ℕ

def E : Point := { x := 0, y := 0 }
def F : Point := { x := 3, y := 2 }
def H : Point := { x := 5, y := 4 }
def G : Point := { x := 8, y := 4 }

-- Function to calculate number of paths from one point to another given the number of right and down steps
def paths (start goal : Point) : ℕ :=
  let right_steps := goal.x - start.x
  let down_steps := goal.y - start.y
  Nat.choose (right_steps + down_steps) right_steps

theorem paths_E_to_G_through_F_and_H : paths E F * paths F H * paths H G = 60 := by
  sorry

end paths_E_to_G_through_F_and_H_l262_26203


namespace parabola_equation_l262_26222

theorem parabola_equation (vertex focus : ℝ × ℝ) 
  (h_vertex : vertex = (0, 0)) 
  (h_focus_line : ∃ x y : ℝ, focus = (x, y) ∧ x - y + 2 = 0) 
  (h_symmetry_axis : ∃ axis : ℝ × ℝ → ℝ, ∀ p : ℝ × ℝ, axis p = 0): 
  ∃ k : ℝ, k > 0 ∧ (∀ x y : ℝ, y^2 = -8*x ∨ x^2 = 8*y) :=
by {
  sorry
}

end parabola_equation_l262_26222


namespace rhombus_diagonals_l262_26297

theorem rhombus_diagonals (x y : ℝ) 
  (h1 : x * y = 234)
  (h2 : x + y = 31) :
  (x = 18 ∧ y = 13) ∨ (x = 13 ∧ y = 18) := by
sorry

end rhombus_diagonals_l262_26297


namespace odd_square_sum_of_consecutive_l262_26265

theorem odd_square_sum_of_consecutive (n : ℤ) (h_odd : n % 2 = 1) (h_gt : n > 1) : 
  ∃ (j : ℤ), n^2 = j + (j + 1) :=
by
  sorry

end odd_square_sum_of_consecutive_l262_26265


namespace problem1_problem2_l262_26263

noncomputable def part1 (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4} ∩ {x | x ≤ 2 * a - 5}
noncomputable def part2 (a : ℝ) : Prop := ∀ x : ℝ, (-2 ≤ x ∧ x ≤ 4) → (x ≤ 2 * a - 5)

theorem problem1 : part1 3 = {x | -2 ≤ x ∧ x ≤ 1} :=
by { sorry }

theorem problem2 : ∀ a : ℝ, (part2 a) ↔ (a ≥ 9/2) :=
by { sorry }

end problem1_problem2_l262_26263


namespace hot_dogs_remainder_l262_26201

theorem hot_dogs_remainder :
  let n := 16789537
  let d := 5
  n % d = 2 :=
by
  sorry

end hot_dogs_remainder_l262_26201


namespace g_ab_eq_zero_l262_26262

def g (x : ℤ) : ℤ := x^2 - 2013 * x

theorem g_ab_eq_zero (a b : ℤ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 :=
by
  sorry

end g_ab_eq_zero_l262_26262


namespace sparrows_among_non_pigeons_l262_26224

theorem sparrows_among_non_pigeons (perc_sparrows perc_pigeons perc_parrots perc_crows : ℝ)
  (h_sparrows : perc_sparrows = 0.40)
  (h_pigeons : perc_pigeons = 0.20)
  (h_parrots : perc_parrots = 0.15)
  (h_crows : perc_crows = 0.25) :
  (perc_sparrows / (1 - perc_pigeons) * 100) = 50 :=
by
  sorry

end sparrows_among_non_pigeons_l262_26224


namespace sector_angle_l262_26266

theorem sector_angle (r L : ℝ) (h1 : r = 1) (h2 : L = 4) : abs (L - 2 * r) = 2 :=
by 
  -- This is the statement of our proof problem
  -- and does not include the proof itself.
  sorry

end sector_angle_l262_26266


namespace find_t_l262_26207

-- Given: (1) g(x) = x^5 + px^4 + qx^3 + rx^2 + sx + t with all roots being negative integers
--        (2) p + q + r + s + t = 3024
-- Prove: t = 1600

noncomputable def poly (x : ℝ) (p q r s t : ℝ) := 
  x^5 + p*x^4 + q*x^3 + r*x^2 + s*x + t

theorem find_t
  (p q r s t : ℝ)
  (roots_neg_int : ∀ root, root ∈ [-s1, -s2, -s3, -s4, -s5] → (root : ℤ) < 0)
  (sum_coeffs : p + q + r + s + t = 3024)
  (poly_1_eq : poly 1 p q r s t = 3025) :
  t = 1600 := 
sorry

end find_t_l262_26207


namespace metallic_sheet_dimension_l262_26233

theorem metallic_sheet_dimension
  (length_cut : ℕ) (other_dim : ℕ) (volume : ℕ) (x : ℕ)
  (length_cut_eq : length_cut = 8)
  (other_dim_eq : other_dim = 36)
  (volume_eq : volume = 4800)
  (volume_formula : volume = (x - 2 * length_cut) * (other_dim - 2 * length_cut) * length_cut) :
  x = 46 :=
by
  sorry

end metallic_sheet_dimension_l262_26233


namespace initial_profit_percentage_l262_26232

-- Definitions of conditions
variables {x y : ℝ} (h1 : y > x) (h2 : 2 * y - x = 1.4 * x)

-- Proof statement in Lean
theorem initial_profit_percentage (x y : ℝ) (h1 : y > x) (h2 : 2 * y - x = 1.4 * x) :
  ((y - x) / x) * 100 = 20 :=
by sorry

end initial_profit_percentage_l262_26232


namespace nat_representation_l262_26212

theorem nat_representation (k : ℕ) : ∃ n r : ℕ, (r = 0 ∨ r = 1 ∨ r = 2) ∧ k = 3 * n + r :=
by
  sorry

end nat_representation_l262_26212


namespace least_positive_integer_x_l262_26287

theorem least_positive_integer_x (x : ℕ) (h1 : x + 3721 ≡ 1547 [MOD 12]) (h2 : x % 2 = 0) : x = 2 :=
sorry

end least_positive_integer_x_l262_26287


namespace num_five_ruble_coins_l262_26213

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l262_26213


namespace parabola_tangent_sum_l262_26289

theorem parabola_tangent_sum (m n : ℕ) (hmn_coprime : Nat.gcd m n = 1)
    (h_tangent : ∃ (k : ℝ), ∀ (x y : ℝ), y = 4 * x^2 ↔ x = y^2 + (m / n)) :
    m + n = 19 :=
by
  sorry

end parabola_tangent_sum_l262_26289


namespace beta_still_water_speed_l262_26280

-- Definitions that are used in the conditions
def alpha_speed_still_water : ℝ := 56 
def beta_speed_still_water : ℝ := 52  
def water_current_speed : ℝ := 4

-- The main theorem statement 
theorem beta_still_water_speed : β_speed_still_water = 61 := 
  sorry -- the proof goes here

end beta_still_water_speed_l262_26280


namespace dinner_handshakes_l262_26220

def num_couples := 8
def num_people_per_couple := 2
def num_attendees := num_couples * num_people_per_couple

def shakes_per_person (n : Nat) := n - 2
def total_possible_shakes (n : Nat) := (n * shakes_per_person n) / 2

theorem dinner_handshakes : total_possible_shakes num_attendees = 112 :=
by
  sorry

end dinner_handshakes_l262_26220


namespace inscribed_circle_radius_A_B_D_l262_26299

theorem inscribed_circle_radius_A_B_D (AB CD: ℝ) (angleA acuteAngleD: Prop)
  (M N: Type) (MN: ℝ) (area_trapezoid: ℝ)
  (radius: ℝ) : 
  AB = 2 ∧ CD = 3 ∧ angleA ∧ acuteAngleD ∧ MN = 4 ∧ area_trapezoid = (26 * Real.sqrt 2) / 3 
  → radius = (16 * Real.sqrt 2) / (15 + Real.sqrt 129) :=
by
  intro h
  sorry

end inscribed_circle_radius_A_B_D_l262_26299


namespace prob_B_hired_is_3_4_prob_at_least_two_hired_l262_26211

-- Definitions for the conditions
def prob_A_hired : ℚ := 2 / 3
def prob_neither_A_nor_B_hired : ℚ := 1 / 12
def prob_B_and_C_hired : ℚ := 3 / 8

-- Targets to prove
theorem prob_B_hired_is_3_4 (P_A_hired : ℚ) (P_neither_A_nor_B_hired : ℚ) (P_B_and_C_hired : ℚ)
    (P_A_hired_eq : P_A_hired = prob_A_hired)
    (P_neither_A_nor_B_hired_eq : P_neither_A_nor_B_hired = prob_neither_A_nor_B_hired)
    (P_B_and_C_hired_eq : P_B_and_C_hired = prob_B_and_C_hired)
    : ∃ x y : ℚ, y = 1 / 2 ∧ x = 3 / 4 :=
by
  sorry
  
theorem prob_at_least_two_hired (P_A_hired : ℚ) (P_B_hired : ℚ) (P_C_hired : ℚ)
    (P_A_hired_eq : P_A_hired = prob_A_hired)
    (P_B_hired_eq : P_B_hired = 3 / 4)
    (P_C_hired_eq : P_C_hired = 1 / 2)
    : (P_A_hired * P_B_hired * P_C_hired) + 
      ((1 - P_A_hired) * P_B_hired * P_C_hired) + 
      (P_A_hired * (1 - P_B_hired) * P_C_hired) + 
      (P_A_hired * P_B_hired * (1 - P_C_hired)) = 2 / 3 :=
by
  sorry

end prob_B_hired_is_3_4_prob_at_least_two_hired_l262_26211


namespace solve_inequality_l262_26219

theorem solve_inequality (x : ℝ) (h : 5 * x - 12 ≤ 2 * (4 * x - 3)) : x ≥ -2 :=
sorry

end solve_inequality_l262_26219


namespace square_of_binomial_l262_26243

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, (x^2 - 18 * x + k) = (x + b)^2) ↔ k = 81 :=
by
  sorry

end square_of_binomial_l262_26243


namespace jerry_initial_candy_l262_26281

theorem jerry_initial_candy
  (total_bags : ℕ)
  (chocolate_hearts_bags : ℕ)
  (chocolate_kisses_bags : ℕ)
  (nonchocolate_pieces : ℕ)
  (remaining_bags : ℕ)
  (pieces_per_bag : ℕ)
  (initial_candy : ℕ)
  (h_total_bags : total_bags = 9)
  (h_chocolate_hearts_bags : chocolate_hearts_bags = 2)
  (h_chocolate_kisses_bags : chocolate_kisses_bags = 3)
  (h_nonchocolate_pieces : nonchocolate_pieces = 28)
  (h_remaining_bags : remaining_bags = total_bags - chocolate_hearts_bags - chocolate_kisses_bags)
  (h_pieces_per_bag : pieces_per_bag = nonchocolate_pieces / remaining_bags)
  (h_initial_candy : initial_candy = total_bags * pieces_per_bag) :
  initial_candy = 63 := by
  sorry

end jerry_initial_candy_l262_26281


namespace range_of_a_l262_26259

noncomputable def setA : Set ℝ := {x | 3 + 2 * x - x^2 >= 0}
noncomputable def setB (a : ℝ) : Set ℝ := {x | x > a}

theorem range_of_a (a : ℝ) : (setA ∩ setB a).Nonempty → a < 3 :=
by
  sorry

end range_of_a_l262_26259


namespace abc_area_l262_26277

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

theorem abc_area :
  let smaller_side := 7
  let longer_side := 2 * smaller_side
  let length := 3 * longer_side -- since there are 3 identical rectangles placed side by side
  let width := smaller_side
  rectangle_area length width = 294 :=
by
  sorry

end abc_area_l262_26277


namespace intersection_point_on_square_diagonal_l262_26240

theorem intersection_point_on_square_diagonal (a b c : ℝ) (h : c = (a + b) / 2) :
  (b / 2) = (-a / 2) + c :=
by
  sorry

end intersection_point_on_square_diagonal_l262_26240


namespace volume_of_stone_l262_26205

theorem volume_of_stone 
  (width length initial_height final_height : ℕ)
  (h_width : width = 15)
  (h_length : length = 20)
  (h_initial_height : initial_height = 10)
  (h_final_height : final_height = 15)
  : (width * length * final_height - width * length * initial_height = 1500) :=
by
  sorry

end volume_of_stone_l262_26205


namespace daphne_necklaces_l262_26261

/--
Given:
1. Total cost of necklaces and earrings is $240,000.
2. Necklaces are equal in price.
3. Earrings were three times as expensive as any one necklace.
4. Cost of a single necklace is $40,000.

Prove:
Princess Daphne bought 3 necklaces.
-/
theorem daphne_necklaces (total_cost : ℤ) (price_necklace : ℤ) (price_earrings : ℤ) (n : ℤ)
  (h1 : total_cost = 240000)
  (h2 : price_necklace = 40000)
  (h3 : price_earrings = 3 * price_necklace)
  (h4 : total_cost = n * price_necklace + price_earrings) : n = 3 :=
by
  sorry

end daphne_necklaces_l262_26261


namespace state_A_selection_percentage_l262_26269

theorem state_A_selection_percentage
  (candidates_A : ℕ)
  (candidates_B : ℕ)
  (x : ℕ)
  (selected_B_ratio : ℚ)
  (extra_B : ℕ)
  (h1 : candidates_A = 7900)
  (h2 : candidates_B = 7900)
  (h3 : selected_B_ratio = 0.07)
  (h4 : extra_B = 79)
  (h5 : 7900 * (7 / 100) + 79 = 7900 * (x / 100) + 79) :
  x = 7 := by
  sorry

end state_A_selection_percentage_l262_26269


namespace largest_x_not_defined_l262_26298

theorem largest_x_not_defined : 
  (∀ x, (6 * x ^ 2 - 17 * x + 5 = 0) → x ≤ 2.5) ∧
  (∃ x, (6 * x ^ 2 - 17 * x + 5 = 0) ∧ x = 2.5) :=
by
  sorry

end largest_x_not_defined_l262_26298


namespace find_z_value_l262_26236

theorem find_z_value (z w : ℝ) (hz : z ≠ 0) (hw : w ≠ 0)
  (h1 : z + 1/w = 15) (h2 : w^2 + 1/z = 3) : z = 44/3 := 
by 
  sorry

end find_z_value_l262_26236


namespace grandfather_age_l262_26258

variables (M G y z : ℕ)

-- Conditions
def condition1 : Prop := G = 6 * M
def condition2 : Prop := G + y = 5 * (M + y)
def condition3 : Prop := G + y + z = 4 * (M + y + z)

-- Theorem to prove Grandfather's current age is 72
theorem grandfather_age : 
  condition1 M G → 
  condition2 M G y → 
  condition3 M G y z → 
  G = 72 :=
by
  intros h1 h2 h3
  unfold condition1 at h1
  unfold condition2 at h2
  unfold condition3 at h3
  sorry

end grandfather_age_l262_26258


namespace dice_probability_l262_26208

theorem dice_probability :
  let num_dice := 6
  let prob_one_digit := 9 / 20
  let prob_two_digit := 11 / 20
  let num_combinations := Nat.choose num_dice (num_dice / 2)
  let prob_each_combination := (prob_one_digit ^ 3) * (prob_two_digit ^ 3)
  let total_probability := num_combinations * prob_each_combination
  total_probability = 4851495 / 16000000 := by
    let num_dice := 6
    let prob_one_digit := 9 / 20
    let prob_two_digit := 11 / 20
    let num_combinations := Nat.choose num_dice (num_dice / 2)
    let prob_each_combination := (prob_one_digit ^ 3) * (prob_two_digit ^ 3)
    let total_probability := num_combinations * prob_each_combination
    sorry

end dice_probability_l262_26208


namespace minimum_number_of_apples_l262_26284

-- Define the problem conditions and the proof statement
theorem minimum_number_of_apples :
  ∃ p : Fin 6 → ℕ, (∀ i, p i > 0) ∧ (Function.Injective p) ∧ (Finset.univ.sum p * 4 = 100) ∧ (Finset.univ.sum p = 25 / 4) := 
sorry

end minimum_number_of_apples_l262_26284


namespace milan_minutes_billed_l262_26215

noncomputable def total_bill : ℝ := 23.36
noncomputable def monthly_fee : ℝ := 2.00
noncomputable def cost_per_minute : ℝ := 0.12

theorem milan_minutes_billed :
  (total_bill - monthly_fee) / cost_per_minute = 178 := 
sorry

end milan_minutes_billed_l262_26215


namespace expected_value_of_expression_is_50_l262_26288

def expected_value_single_digit : ℚ := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) / 9

def expected_value_expression : ℚ :=
  (expected_value_single_digit + expected_value_single_digit + expected_value_single_digit +
   (expected_value_single_digit + expected_value_single_digit * expected_value_single_digit) +
   (expected_value_single_digit * expected_value_single_digit + expected_value_single_digit) +
   (expected_value_single_digit * expected_value_single_digit * expected_value_single_digit)) / 4

theorem expected_value_of_expression_is_50 :
  expected_value_expression = 50 := sorry

end expected_value_of_expression_is_50_l262_26288


namespace proof_M_inter_N_eq_01_l262_26242
open Set

theorem proof_M_inter_N_eq_01 :
  let M := {x : ℤ | x^2 = x}
  let N := {-1, 0, 1}
  M ∩ N = {0, 1} := by
  sorry

end proof_M_inter_N_eq_01_l262_26242


namespace Jerry_has_36_stickers_l262_26292

variable (FredStickers GeorgeStickers JerryStickers CarlaStickers : ℕ)
variable (h1 : FredStickers = 18)
variable (h2 : GeorgeStickers = FredStickers - 6)
variable (h3 : JerryStickers = 3 * GeorgeStickers)
variable (h4 : CarlaStickers = JerryStickers + JerryStickers / 4)
variable (h5 : GeorgeStickers + FredStickers = CarlaStickers ^ 2)

theorem Jerry_has_36_stickers : JerryStickers = 36 := by
  sorry

end Jerry_has_36_stickers_l262_26292


namespace remaining_rice_l262_26247

theorem remaining_rice {q_0 : ℕ} {c : ℕ} {d : ℕ} 
    (h_q0 : q_0 = 52) 
    (h_c : c = 9) 
    (h_d : d = 3) : 
    q_0 - (c * d) = 25 := 
  by 
    -- Proof to be written here
    sorry

end remaining_rice_l262_26247


namespace total_packages_sold_l262_26206

variable (P : ℕ)

/-- An automobile parts supplier charges 25 per package of gaskets. 
    When a customer orders more than 10 packages of gaskets, the supplier charges 4/5 the price for each package in excess of 10.
    During a certain week, the supplier received 1150 in payment for the gaskets. --/
def cost (P : ℕ) : ℕ :=
  if P > 10 then 250 + (P - 10) * 20 else P * 25

theorem total_packages_sold :
  cost P = 1150 → P = 55 := by
  sorry

end total_packages_sold_l262_26206


namespace fifth_scroll_age_l262_26255

def scrolls_age (n : ℕ) : ℕ :=
  match n with
  | 0 => 4080
  | k+1 => (3 * scrolls_age k) / 2

theorem fifth_scroll_age : scrolls_age 4 = 20655 := sorry

end fifth_scroll_age_l262_26255


namespace coefficient_a2_l262_26275

theorem coefficient_a2 :
  ∀ (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ),
  (x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
  a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + 
  a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + a_9 * (x + 1)^9 + 
  a_10 * (x + 1)^10) →
  a_2 = 45 :=
by
  sorry

end coefficient_a2_l262_26275


namespace combined_weight_of_new_students_l262_26248

theorem combined_weight_of_new_students 
  (avg_weight_orig : ℝ) (num_students_orig : ℝ) 
  (new_avg_weight : ℝ) (num_new_students : ℝ) 
  (total_weight_gain_orig : ℝ) (total_weight_loss_orig : ℝ)
  (total_weight_orig : ℝ := avg_weight_orig * num_students_orig) 
  (net_weight_change_orig : ℝ := total_weight_gain_orig - total_weight_loss_orig)
  (total_weight_after_change_orig : ℝ := total_weight_orig + net_weight_change_orig) 
  (total_students_after : ℝ := num_students_orig + num_new_students) 
  (total_weight_class_after : ℝ := new_avg_weight * total_students_after) : 
  total_weight_class_after - total_weight_after_change_orig = 586 :=
by
  sorry

end combined_weight_of_new_students_l262_26248


namespace find_ab_l262_26225

theorem find_ab (a b c : ℤ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) (h3 : a + b + c = 21) : a * b = 10 := 
sorry

end find_ab_l262_26225


namespace find_center_and_tangent_slope_l262_26250

theorem find_center_and_tangent_slope :
  let C := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 6 * p.1 + 8 = 0 }
  let center := (3, 0)
  let k := - (Real.sqrt 2 / 4)
  (∃ c ∈ C, c = center) ∧
  (∃ q ∈ C, q.2 < 0 ∧ q.2 = k * q.1 ∧
             |3 * k| / Real.sqrt (k ^ 2 + 1) = 1) :=
by
  sorry

end find_center_and_tangent_slope_l262_26250


namespace line_MN_parallel_to_y_axis_l262_26296

-- Definition of points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of vector between two points
def vector_between (P Q : Point) : Point :=
  { x := Q.x - P.x,
    y := Q.y - P.y,
    z := Q.z - P.z }

-- Given points M and N
def M : Point := { x := 3, y := -2, z := 1 }
def N : Point := { x := 3, y := 2, z := 1 }

-- The vector \overrightarrow{MN}
def vec_MN : Point := vector_between M N

-- Theorem: The vector between points M and N is parallel to the y-axis
theorem line_MN_parallel_to_y_axis : vec_MN = {x := 0, y := 4, z := 0} := by
  sorry

end line_MN_parallel_to_y_axis_l262_26296


namespace shoe_price_calculation_l262_26252

theorem shoe_price_calculation :
  let initialPrice : ℕ := 50
  let increasedPrice : ℕ := 60  -- initialPrice * 1.2
  let discountAmount : ℕ := 9    -- increasedPrice * 0.15
  increasedPrice - discountAmount = 51 := 
by
  sorry

end shoe_price_calculation_l262_26252


namespace remainder_of_sums_modulo_l262_26209

theorem remainder_of_sums_modulo :
  (2 * (8735 + 8736 + 8737 + 8738 + 8739)) % 11 = 8 :=
by
  sorry

end remainder_of_sums_modulo_l262_26209


namespace common_divisors_count_l262_26264

-- Given conditions
def num1 : ℕ := 9240
def num2 : ℕ := 8000

-- Prime factorizations from conditions
def fact_num1 : List ℕ := [2^3, 3^1, 5^1, 7^2]
def fact_num2 : List ℕ := [2^6, 5^3]

-- Computing gcd based on factorizations
def gcd : ℕ := 2^3 * 5^1

-- The goal is to prove the number of divisors of gcd is 8
theorem common_divisors_count : 
  ∃ d, d = (3+1)*(1+1) ∧ d = 8 := 
by
  sorry

end common_divisors_count_l262_26264


namespace distance_formula_proof_l262_26291

open Real

noncomputable def distance_between_points_on_curve
  (a b c d m k : ℝ)
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  ℝ :=
  |c - a| * sqrt (1 + m^2 * (c + a)^2)

theorem distance_formula_proof
  (a b c d m k : ℝ)
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  distance_between_points_on_curve a b c d m k h1 h2 = |c - a| * sqrt (1 + m^2 * (c + a)^2) :=
by
  sorry

end distance_formula_proof_l262_26291


namespace equivalent_proposition_l262_26202

variable (M : Set α) (m n : α)

theorem equivalent_proposition :
  (m ∈ M → n ∉ M) ↔ (n ∈ M → m ∉ M) := by
  sorry

end equivalent_proposition_l262_26202


namespace sqrt_inequalities_l262_26274

theorem sqrt_inequalities
  (a b c d e : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 1)
  (hb : 0 ≤ b ∧ b ≤ 1)
  (hc : 0 ≤ c ∧ c ≤ 1)
  (hd : 0 ≤ d ∧ d ≤ 1)
  (he : 0 ≤ e ∧ e ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ∧
  Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ≤ 5 * Real.sqrt 2 :=
by {
  sorry
}

end sqrt_inequalities_l262_26274


namespace total_books_sold_amount_l262_26223

def num_fiction_books := 60
def num_non_fiction_books := 84
def num_children_books := 42

def fiction_books_sold := 3 / 4 * num_fiction_books
def non_fiction_books_sold := 5 / 6 * num_non_fiction_books
def children_books_sold := 2 / 3 * num_children_books

def price_fiction := 5
def price_non_fiction := 7
def price_children := 3

def total_amount_fiction := fiction_books_sold * price_fiction
def total_amount_non_fiction := non_fiction_books_sold * price_non_fiction
def total_amount_children := children_books_sold * price_children

def total_amount_received := total_amount_fiction + total_amount_non_fiction + total_amount_children

theorem total_books_sold_amount :
  total_amount_received = 799 :=
sorry

end total_books_sold_amount_l262_26223


namespace freddy_total_call_cost_l262_26226

def lm : ℕ := 45
def im : ℕ := 31
def lc : ℝ := 0.05
def ic : ℝ := 0.25

theorem freddy_total_call_cost : lm * lc + im * ic = 10.00 := by
  sorry

end freddy_total_call_cost_l262_26226


namespace factor_polynomial_l262_26210

def Polynomial_Factorization (x : ℝ) : Prop := 
  let P := x^2 - 6*x + 9 - 64*x^4
  P = (8*x^2 + x - 3) * (-8*x^2 + x - 3)

theorem factor_polynomial : ∀ x : ℝ, Polynomial_Factorization x :=
by 
  intro x
  unfold Polynomial_Factorization
  sorry

end factor_polynomial_l262_26210


namespace journey_distance_l262_26221

theorem journey_distance :
  ∃ D : ℝ, ((D / 2) / 21 + (D / 2) / 24 = 10) ∧ D = 224 :=
by
  use 224
  sorry

end journey_distance_l262_26221


namespace water_to_milk_ratio_l262_26239

theorem water_to_milk_ratio 
  (V : ℝ) 
  (hV : V > 0) 
  (milk_volume1 : ℝ := (3 / 5) * V) 
  (water_volume1 : ℝ := (2 / 5) * V) 
  (milk_volume2 : ℝ := (4 / 5) * V) 
  (water_volume2 : ℝ := (1 / 5) * V)
  (total_milk_volume : ℝ := milk_volume1 + milk_volume2)
  (total_water_volume : ℝ := water_volume1 + water_volume2) :
  total_water_volume / total_milk_volume = (3 / 7) := 
  sorry

end water_to_milk_ratio_l262_26239


namespace g_18_66_l262_26229

def g (x y : ℕ) : ℕ := sorry

axiom g_prop1 : ∀ x, g x x = x
axiom g_prop2 : ∀ x y, g x y = g y x
axiom g_prop3 : ∀ x y, (x + 2 * y) * g x y = y * g x (x + 2 * y)

theorem g_18_66 : g 18 66 = 198 :=
by
  sorry

end g_18_66_l262_26229


namespace consecutive_integers_sum_l262_26235

theorem consecutive_integers_sum (a b : ℤ) (sqrt_33 : ℝ) (h1 : a < sqrt_33) (h2 : sqrt_33 < b) (h3 : b = a + 1) (h4 : sqrt_33 = Real.sqrt 33) : a + b = 11 :=
  sorry

end consecutive_integers_sum_l262_26235


namespace find_t_given_conditions_l262_26257

variables (p t j x y : ℝ)

theorem find_t_given_conditions
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p * (1 - t / 100))
  (h4 : x = 0.10 * t)
  (h5 : y = 0.50 * j)
  (h6 : x + y = 12) :
  t = 24 :=
by sorry

end find_t_given_conditions_l262_26257


namespace difference_pencils_l262_26234

theorem difference_pencils (x : ℕ) (h1 : 162 = x * n_g) (h2 : 216 = x * n_f) : n_f - n_g = 3 :=
by
  sorry

end difference_pencils_l262_26234


namespace seventh_grade_problem_l262_26217

theorem seventh_grade_problem (x y : ℕ) (h1 : x + y = 12) (h2 : 6 * x = 3 * 4 * y) :
  (x + y = 12 ∧ 6 * x = 3 * 4 * y) :=
by
  apply And.intro
  . exact h1
  . exact h2

end seventh_grade_problem_l262_26217


namespace system1_solution_l262_26227

theorem system1_solution (x y : ℝ) (h1 : 2 * x - y = 1) (h2 : 7 * x - 3 * y = 4) : x = 1 ∧ y = 1 :=
by sorry

end system1_solution_l262_26227
