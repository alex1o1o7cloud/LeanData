import Mathlib

namespace NUMINAMATH_GPT_extremum_points_l362_36277

noncomputable def f (x1 x2 : ℝ) : ℝ := x1 * x2 / (1 + x1^2 * x2^2)

theorem extremum_points :
  (f 0 0 = 0) ∧
  (∀ x1 : ℝ, f x1 (-1 / x1) = -1 / 2) ∧
  (∀ x1 : ℝ, f x1 (1 / x1) = 1 / 2) ∧
  ∀ y1 y2 : ℝ, (f 0 0 < f y1 y2 → (0 < y1 ∧ 0 < y2)) ∧ 
             (f 0 0 > f y1 y2 → (0 > y1 ∧ 0 > y2)) :=
by
  sorry

end NUMINAMATH_GPT_extremum_points_l362_36277


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l362_36232

-- Define the first equation
def eq1 (x : ℝ) : Prop := x^2 - 2 * x - 1 = 0

-- Define the second equation
def eq2 (x : ℝ) : Prop := (x - 2)^2 = 2 * x - 4

-- State the first theorem
theorem solve_eq1 (x : ℝ) : eq1 x ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by sorry

-- State the second theorem
theorem solve_eq2 (x : ℝ) : eq2 x ↔ (x = 2 ∨ x = 4) :=
by sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l362_36232


namespace NUMINAMATH_GPT_inclination_angle_of_line_l362_36228

theorem inclination_angle_of_line (α : ℝ) (h_eq : ∀ x y, x - y + 1 = 0 ↔ y = x + 1) (h_range : 0 < α ∧ α < 180) :
  α = 45 :=
by
  -- α is the inclination angle satisfying tan α = 1 and 0 < α < 180
  sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l362_36228


namespace NUMINAMATH_GPT_lesser_number_l362_36237

theorem lesser_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : y = 25 :=
by
  have h3 : x = 35 := sorry
  exact sorry

end NUMINAMATH_GPT_lesser_number_l362_36237


namespace NUMINAMATH_GPT_sum_of_coordinates_eq_nine_halves_l362_36248

theorem sum_of_coordinates_eq_nine_halves {f : ℝ → ℝ} 
  (h₁ : 2 = (f 1) / 2) :
  (4 + (1 / 2) = 9 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_eq_nine_halves_l362_36248


namespace NUMINAMATH_GPT_distribution_properties_l362_36267

theorem distribution_properties (m d j s k : ℝ) (h1 : True)
  (h2 : True)
  (h3 : True)
  (h4 : 68 ≤ 100 ∧ 68 ≥ 0) -- 68% being a valid percentage
  : j = 84 ∧ s = s ∧ k = k :=
by
  -- sorry is used to highlight the proof is not included
  sorry

end NUMINAMATH_GPT_distribution_properties_l362_36267


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l362_36224

noncomputable def M : Set ℝ := { y : ℝ | ∃ x : ℝ, y = x^2 }
noncomputable def N : Set ℝ := { y : ℝ | ∃ x : ℝ, x^2 + y^2 = 1 }

theorem intersection_of_M_and_N : M ∩ N = { y : ℝ | 0 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l362_36224


namespace NUMINAMATH_GPT_john_has_dollars_left_l362_36204

-- Definitions based on the conditions
def john_savings_octal : ℕ := 5273
def rental_car_cost_decimal : ℕ := 1500

-- Define the function to convert octal to decimal
def octal_to_decimal (n : ℕ) : ℕ := -- Conversion logic
sorry

-- Statements for the conversion and subtraction
def john_savings_decimal : ℕ := octal_to_decimal john_savings_octal
def amount_left_for_gas_and_accommodations : ℕ :=
  john_savings_decimal - rental_car_cost_decimal

-- Theorem statement equivalent to the correct answer
theorem john_has_dollars_left :
  amount_left_for_gas_and_accommodations = 1247 :=
by sorry

end NUMINAMATH_GPT_john_has_dollars_left_l362_36204


namespace NUMINAMATH_GPT_positive_integer_expression_iff_l362_36217

theorem positive_integer_expression_iff (p : ℕ) : (0 < p) ∧ (∃ k : ℕ, 0 < k ∧ 4 * p + 35 = k * (3 * p - 8)) ↔ p = 3 :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_expression_iff_l362_36217


namespace NUMINAMATH_GPT_sum_of_factors_coefficients_l362_36249

theorem sum_of_factors_coefficients (a b c d e f g h i j k l m n o p : ℤ) :
  (81 * x^8 - 256 * y^8 = (a * x + b * y) *
                        (c * x^2 + d * x * y + e * y^2) *
                        (f * x^3 + g * x * y^2 + h * y^3) *
                        (i * x + j * y) *
                        (k * x^2 + l * x * y + m * y^2) *
                        (n * x^3 + o * x * y^2 + p * y^3)) →
  a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p = 40 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_factors_coefficients_l362_36249


namespace NUMINAMATH_GPT_volume_of_prism_l362_36278

theorem volume_of_prism (l w h : ℝ) (hlw : l * w = 10) (hwh : w * h = 15) (hlh : l * h = 18) :
  l * w * h = 30 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l362_36278


namespace NUMINAMATH_GPT_exists_palindromic_product_l362_36206

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let digits := toDigits 10 n
  digits = digits.reverse

theorem exists_palindromic_product (x : ℕ) (hx : ¬ (10 ∣ x)) : ∃ y : ℕ, is_palindrome (x * y) :=
by
  -- Prove that there exists a natural number y such that x * y is a palindromic number
  sorry

end NUMINAMATH_GPT_exists_palindromic_product_l362_36206


namespace NUMINAMATH_GPT_rotation_150_positions_l362_36229

/-
Define the initial positions and the shapes involved.
-/
noncomputable def initial_positions := ["A", "B", "C", "D"]
noncomputable def initial_order := ["triangle", "smaller_circle", "square", "pentagon"]

def rotate_clockwise_150 (pos : List String) : List String :=
  -- 1 full position and two-thirds into the next position
  [pos.get! 1, pos.get! 2, pos.get! 3, pos.get! 0]

theorem rotation_150_positions :
  rotate_clockwise_150 initial_positions = ["Triangle between B and C", 
                                            "Smaller circle between C and D", 
                                            "Square between D and A", 
                                            "Pentagon between A and B"] :=
by sorry

end NUMINAMATH_GPT_rotation_150_positions_l362_36229


namespace NUMINAMATH_GPT_common_difference_l362_36255

theorem common_difference (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1)
  (h2 : a 2 = 1 + d) (h4 : a 4 = 1 + 3 * d) (h5 : a 5 = 1 + 4 * d) 
  (h_geometric : (a 4)^2 = a 2 * a 5) 
  (h_nonzero : d ≠ 0) : 
  d = 1 / 5 :=
by sorry

end NUMINAMATH_GPT_common_difference_l362_36255


namespace NUMINAMATH_GPT_count_factors_multiple_of_150_l362_36292

theorem count_factors_multiple_of_150 (n : ℕ) (h : n = 2^10 * 3^14 * 5^8) : 
  ∃ k, k = 980 ∧ ∀ d : ℕ, d ∣ n → 150 ∣ d → (d.factors.card = k) := sorry

end NUMINAMATH_GPT_count_factors_multiple_of_150_l362_36292


namespace NUMINAMATH_GPT_binomial_parameters_l362_36289

theorem binomial_parameters
  (n : ℕ) (p : ℚ)
  (hE : n * p = 12) (hD : n * p * (1 - p) = 2.4) :
  n = 15 ∧ p = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_binomial_parameters_l362_36289


namespace NUMINAMATH_GPT_polygon_sides_eq_six_l362_36288

theorem polygon_sides_eq_six (n : ℕ) (S_i S_e : ℕ) :
  S_i = 2 * S_e →
  S_e = 360 →
  (n - 2) * 180 = S_i →
  n = 6 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_six_l362_36288


namespace NUMINAMATH_GPT_Genevieve_drinks_pints_l362_36256

theorem Genevieve_drinks_pints :
  ∀ (total_gallons : ℝ) (num_thermoses : ℕ) (pints_per_gallon : ℝ) (genevieve_thermoses : ℕ),
  total_gallons = 4.5 → num_thermoses = 18 → pints_per_gallon = 8 → genevieve_thermoses = 3 →
  (genevieve_thermoses * ((total_gallons / num_thermoses) * pints_per_gallon) = 6) :=
by
  intros total_gallons num_thermoses pints_per_gallon genevieve_thermoses
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_Genevieve_drinks_pints_l362_36256


namespace NUMINAMATH_GPT_stream_current_rate_l362_36245

theorem stream_current_rate (r w : ℝ) : 
  (15 / (r + w) + 5 = 15 / (r - w)) → 
  (15 / (2 * r + w) + 1 = 15 / (2 * r - w)) →
  w = 2 := 
by
  sorry

end NUMINAMATH_GPT_stream_current_rate_l362_36245


namespace NUMINAMATH_GPT_tiling_possible_values_of_n_l362_36287

-- Define the sizes of the grid and the tiles
def grid_size : ℕ × ℕ := (9, 7)
def l_tile_size : ℕ := 3  -- L-shaped tile composed of three unit squares
def square_tile_size : ℕ := 4  -- square tile composed of four unit squares

-- Formalize the properties of the grid and the constraints for the tiling
def total_squares : ℕ := grid_size.1 * grid_size.2
def white_squares (n : ℕ) : ℕ := 3 * n
def black_squares (n : ℕ) : ℕ := n
def total_black_squares : ℕ := 20
def total_white_squares : ℕ := total_squares - total_black_squares

-- The main theorem statement
theorem tiling_possible_values_of_n (n : ℕ) : 
  (n = 2 ∨ n = 5 ∨ n = 8 ∨ n = 11 ∨ n = 14 ∨ n = 17 ∨ n = 20) ↔
  (3 * (total_white_squares - 2 * (20 - n)) / 3 + n = 23 ∧ n + (total_black_squares - n) = 20) :=
sorry

end NUMINAMATH_GPT_tiling_possible_values_of_n_l362_36287


namespace NUMINAMATH_GPT_solve_x_values_l362_36231

theorem solve_x_values (x : ℝ) :
  (5 + x) / (7 + x) = (2 + x^2) / (4 + x) ↔ x = 1 ∨ x = -2 ∨ x = -3 := 
sorry

end NUMINAMATH_GPT_solve_x_values_l362_36231


namespace NUMINAMATH_GPT_paco_ate_more_cookies_l362_36222

-- Define the number of cookies Paco originally had
def original_cookies : ℕ := 25

-- Define the number of cookies Paco ate
def eaten_cookies : ℕ := 5

-- Define the number of cookies Paco bought
def bought_cookies : ℕ := 3

-- Define the number of more cookies Paco ate than bought
def more_cookies_eaten_than_bought : ℕ := eaten_cookies - bought_cookies

-- Prove that Paco ate 2 more cookies than he bought
theorem paco_ate_more_cookies : more_cookies_eaten_than_bought = 2 := by
  sorry

end NUMINAMATH_GPT_paco_ate_more_cookies_l362_36222


namespace NUMINAMATH_GPT_intersection_of_P_and_Q_l362_36282

def P : Set ℤ := {x | -4 ≤ x ∧ x ≤ 2 ∧ x ∈ Set.univ}
def Q : Set ℤ := {x | -3 < x ∧ x < 1}

theorem intersection_of_P_and_Q :
  P ∩ Q = {-2, -1, 0} :=
sorry

end NUMINAMATH_GPT_intersection_of_P_and_Q_l362_36282


namespace NUMINAMATH_GPT_compute_g_five_times_l362_36286

def g (x : ℤ) : ℤ :=
  if x ≥ 0 then - x^3 else x + 10

theorem compute_g_five_times (x : ℤ) (h : x = 2) : g (g (g (g (g x)))) = -8 := by
  sorry

end NUMINAMATH_GPT_compute_g_five_times_l362_36286


namespace NUMINAMATH_GPT_ice_cream_melt_l362_36203

theorem ice_cream_melt (r_sphere r_cylinder : ℝ) (h : ℝ)
  (V_sphere : ℝ := (4 / 3) * Real.pi * r_sphere^3)
  (V_cylinder : ℝ := Real.pi * r_cylinder^2 * h)
  (H_equal_volumes : V_sphere = V_cylinder) :
  h = 4 / 9 := by
  sorry

end NUMINAMATH_GPT_ice_cream_melt_l362_36203


namespace NUMINAMATH_GPT_solution_set_of_bx2_minus_ax_minus_1_gt_0_l362_36275

theorem solution_set_of_bx2_minus_ax_minus_1_gt_0
  (a b : ℝ)
  (h1 : ∀ (x : ℝ), 2 < x ∧ x < 3 ↔ x^2 - a * x - b < 0) :
  ∀ (x : ℝ), -1 / 2 < x ∧ x < -1 / 3 ↔ b * x^2 - a * x - 1 > 0 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_bx2_minus_ax_minus_1_gt_0_l362_36275


namespace NUMINAMATH_GPT_number_of_possible_third_side_lengths_l362_36213

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_third_side_lengths_l362_36213


namespace NUMINAMATH_GPT_checkered_square_division_l362_36259

theorem checkered_square_division (m n k d m1 n1 : ℕ) (h1 : m^2 = n * k)
  (h2 : d = Nat.gcd m n) (hm : m = m1 * d) (hn : n = n1 * d)
  (h3 : Nat.gcd m1 n1 = 1) : 
  ∃ (part_size : ℕ), 
    part_size = n ∧ (∃ (pieces : ℕ), pieces = k) ∧ m^2 = pieces * part_size := 
sorry

end NUMINAMATH_GPT_checkered_square_division_l362_36259


namespace NUMINAMATH_GPT_max_value_of_a_l362_36235

theorem max_value_of_a 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 - a * x) 
  (h2 : ∀ x ≥ 1, ∀ y ≥ 1, x ≤ y → f x ≤ f y) : 
  a ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_of_a_l362_36235


namespace NUMINAMATH_GPT_reciprocal_of_sum_frac_is_correct_l362_36226

/-- The reciprocal of the sum of the fractions 1/4 and 1/6 is 12/5. -/
theorem reciprocal_of_sum_frac_is_correct:
  (1 / (1 / 4 + 1 / 6)) = (12 / 5) :=
by 
  sorry

end NUMINAMATH_GPT_reciprocal_of_sum_frac_is_correct_l362_36226


namespace NUMINAMATH_GPT_length_of_place_mat_l362_36254

noncomputable def length_of_mat
  (R : ℝ)
  (w : ℝ)
  (n : ℕ)
  (θ : ℝ) : ℝ :=
  2 * R * Real.sin (θ / 2)

theorem length_of_place_mat :
  ∃ y : ℝ, y = length_of_mat 5 1 7 (360 / 7) := by
  use 4.38
  sorry

end NUMINAMATH_GPT_length_of_place_mat_l362_36254


namespace NUMINAMATH_GPT_natural_pairs_l362_36211

theorem natural_pairs (x y : ℕ) : 2^(2 * x + 1) + 2^x + 1 = y^2 ↔ (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by sorry

end NUMINAMATH_GPT_natural_pairs_l362_36211


namespace NUMINAMATH_GPT_graph_passes_fixed_point_l362_36202

-- Mathematical conditions
variables (a : ℝ)

-- Real numbers and conditions
def is_fixed_point (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧ ∃ x y, (x, y) = (2, 2) ∧ y = a^(x-2) + 1

-- Lean statement for the problem
theorem graph_passes_fixed_point : is_fixed_point a :=
  sorry

end NUMINAMATH_GPT_graph_passes_fixed_point_l362_36202


namespace NUMINAMATH_GPT_remainder_of_50_pow_2019_plus_1_mod_7_l362_36263

theorem remainder_of_50_pow_2019_plus_1_mod_7 :
  (50 ^ 2019 + 1) % 7 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_50_pow_2019_plus_1_mod_7_l362_36263


namespace NUMINAMATH_GPT_suit_price_after_discount_l362_36268

-- Definitions based on given conditions 
def original_price : ℝ := 200
def price_increase : ℝ := 0.30 * original_price
def new_price : ℝ := original_price + price_increase
def discount : ℝ := 0.30 * new_price
def final_price : ℝ := new_price - discount

-- The theorem
theorem suit_price_after_discount :
  final_price = 182 :=
by
  -- Here we would provide the proof if required
  sorry

end NUMINAMATH_GPT_suit_price_after_discount_l362_36268


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_37_l362_36240

theorem smallest_four_digit_multiple_of_37 : ∃ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 ∧ 37 ∣ n ∧ (∀ m : ℕ, m ≥ 1000 ∧ m ≤ 9999 ∧ 37 ∣ m → n ≤ m) ∧ n = 1036 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_37_l362_36240


namespace NUMINAMATH_GPT_second_smallest_integer_l362_36291

theorem second_smallest_integer (x y z w v : ℤ) (h_avg : (x + y + z + w + v) / 5 = 69)
  (h_median : z = 83) (h_mode : w = 85 ∧ v = 85) (h_range : 85 - x = 70) :
  y = 77 :=
by
  sorry

end NUMINAMATH_GPT_second_smallest_integer_l362_36291


namespace NUMINAMATH_GPT_monotonicity_of_f_f_greater_than_2_ln_a_plus_3_div_2_l362_36221

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x y, x < y → f x a > f y a) ∧
  (a > 0 →
    (∀ x, x < Real.log (1 / a) → f x a > f (Real.log (1 / a)) a) ∧
    (∀ x, x > Real.log (1 / a) → f x a > f (Real.log (1 / a)) a)) :=
sorry

theorem f_greater_than_2_ln_a_plus_3_div_2 (a : ℝ) (h : a > 0) (x : ℝ) :
  f x a > 2 * Real.log a + 3 / 2 :=
sorry

end NUMINAMATH_GPT_monotonicity_of_f_f_greater_than_2_ln_a_plus_3_div_2_l362_36221


namespace NUMINAMATH_GPT_remainder_of_2n_div_11_l362_36243

theorem remainder_of_2n_div_11 (n k : ℤ) (h : n = 22 * k + 12) : (2 * n) % 11 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_2n_div_11_l362_36243


namespace NUMINAMATH_GPT_amoeba_count_after_ten_days_l362_36205

theorem amoeba_count_after_ten_days : 
  let initial_amoebas := 1
  let splits_per_day := 3
  let days := 10
  (initial_amoebas * splits_per_day ^ days) = 59049 := 
by 
  let initial_amoebas := 1
  let splits_per_day := 3
  let days := 10
  show (initial_amoebas * splits_per_day ^ days) = 59049
  sorry

end NUMINAMATH_GPT_amoeba_count_after_ten_days_l362_36205


namespace NUMINAMATH_GPT_solution_set_l362_36238

variable (x : ℝ)

def condition_1 : Prop := 2 * x - 4 ≤ 0
def condition_2 : Prop := -x + 1 < 0

theorem solution_set : (condition_1 x ∧ condition_2 x) ↔ (1 < x ∧ x ≤ 2) := by
sorry

end NUMINAMATH_GPT_solution_set_l362_36238


namespace NUMINAMATH_GPT_part1_part2_l362_36230

-- Part 1: Determining the number of toys A and ornaments B wholesaled
theorem part1 (x y : ℕ) (h₁ : x + y = 100) (h₂ : 60 * x + 50 * y = 5650) : 
  x = 65 ∧ y = 35 := by
  sorry

-- Part 2: Determining the minimum number of toys A to wholesale for a 1400元 profit
theorem part2 (m : ℕ) (h₁ : m ≤ 100) (h₂ : (80 - 60) * m + (60 - 50) * (100 - m) ≥ 1400) : 
  m ≥ 40 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l362_36230


namespace NUMINAMATH_GPT_result_of_dividing_295_by_5_and_adding_6_is_65_l362_36298

theorem result_of_dividing_295_by_5_and_adding_6_is_65 : (295 / 5) + 6 = 65 := by
  sorry

end NUMINAMATH_GPT_result_of_dividing_295_by_5_and_adding_6_is_65_l362_36298


namespace NUMINAMATH_GPT_eagle_speed_l362_36296

theorem eagle_speed (E : ℕ) 
  (falcon_speed : ℕ := 46)
  (pelican_speed : ℕ := 33)
  (hummingbird_speed : ℕ := 30)
  (total_distance : ℕ := 248)
  (flight_time : ℕ := 2)
  (falcon_distance := falcon_speed * flight_time)
  (pelican_distance := pelican_speed * flight_time)
  (hummingbird_distance := hummingbird_speed * flight_time) :
  2 * E + falcon_distance + pelican_distance + hummingbird_distance = total_distance →
  E = 15 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_eagle_speed_l362_36296


namespace NUMINAMATH_GPT_compare_fractions_l362_36220

-- Define the fractions
def frac1 : ℚ := -2/3
def frac2 : ℚ := -3/4

-- Prove that -2/3 > -3/4
theorem compare_fractions : frac1 > frac2 :=
by {
  sorry
}

end NUMINAMATH_GPT_compare_fractions_l362_36220


namespace NUMINAMATH_GPT_find_symmetric_point_l362_36241

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def M : Point := ⟨3, -3, -1⟩

def line (x y z : ℝ) : Prop := 
  (x - 6) / 5 = (y - 3.5) / 4 ∧ (x - 6) / 5 = (z + 0.5) / 0

theorem find_symmetric_point (M' : Point) :
  (line M.x M.y M.z) →
  M' = ⟨-1, 2, 0⟩ := by
  sorry

end NUMINAMATH_GPT_find_symmetric_point_l362_36241


namespace NUMINAMATH_GPT_parabola_hyperbola_focus_l362_36271

noncomputable def focus_left (p : ℝ) : ℝ × ℝ :=
  (-p / 2, 0)

theorem parabola_hyperbola_focus (p : ℝ) (hp : p > 0) : 
  focus_left p = (-2, 0) ↔ p = 4 :=
by 
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_focus_l362_36271


namespace NUMINAMATH_GPT_jade_more_transactions_l362_36295

theorem jade_more_transactions 
    (mabel_transactions : ℕ) 
    (anthony_transactions : ℕ)
    (cal_transactions : ℕ)
    (jade_transactions : ℕ)
    (h_mabel : mabel_transactions = 90)
    (h_anthony : anthony_transactions = mabel_transactions + (mabel_transactions / 10))
    (h_cal : cal_transactions = 2 * anthony_transactions / 3)
    (h_jade : jade_transactions = 82) :
    jade_transactions - cal_transactions = 16 :=
sorry

end NUMINAMATH_GPT_jade_more_transactions_l362_36295


namespace NUMINAMATH_GPT_find_f2_l362_36280

theorem find_f2 :
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → 2 * f x - 3 * f (1 / x) = x ^ 2) ∧ f 2 = 93 / 32) :=
sorry

end NUMINAMATH_GPT_find_f2_l362_36280


namespace NUMINAMATH_GPT_wait_time_difference_l362_36250

noncomputable def kids_waiting_for_swings : ℕ := 3
noncomputable def kids_waiting_for_slide : ℕ := 2 * kids_waiting_for_swings
noncomputable def wait_per_kid_swings : ℕ := 2 * 60 -- 2 minutes in seconds
noncomputable def wait_per_kid_slide : ℕ := 15 -- in seconds

noncomputable def total_wait_swings : ℕ := kids_waiting_for_swings * wait_per_kid_swings
noncomputable def total_wait_slide : ℕ := kids_waiting_for_slide * wait_per_kid_slide

theorem wait_time_difference : total_wait_swings - total_wait_slide = 270 := by
  sorry

end NUMINAMATH_GPT_wait_time_difference_l362_36250


namespace NUMINAMATH_GPT_sum_of_first_10_terms_is_350_l362_36210

-- Define the terms and conditions for the arithmetic sequence
variables (a d : ℤ)

-- Define the 4th and 8th terms of the sequence
def fourth_term := a + 3*d
def eighth_term := a + 7*d

-- Given conditions
axiom h1 : fourth_term a d = 23
axiom h2 : eighth_term a d = 55

-- Sum of the first 10 terms of the sequence
def sum_first_10_terms := 10 / 2 * (2*a + (10 - 1)*d)

-- Theorem to prove
theorem sum_of_first_10_terms_is_350 : sum_first_10_terms a d = 350 :=
by sorry

end NUMINAMATH_GPT_sum_of_first_10_terms_is_350_l362_36210


namespace NUMINAMATH_GPT_Aryan_owes_1200_l362_36264

variables (A K : ℝ) -- A represents Aryan's debt, K represents Kyro's debt

-- Condition 1: Aryan's debt is twice Kyro's debt
axiom condition1 : A = 2 * K

-- Condition 2: Aryan pays 60% of her debt
axiom condition2 : (0.60 * A) + (0.80 * K) = 1500 - 300

theorem Aryan_owes_1200 : A = 1200 :=
by
  sorry

end NUMINAMATH_GPT_Aryan_owes_1200_l362_36264


namespace NUMINAMATH_GPT_episodes_first_season_l362_36234

theorem episodes_first_season :
  ∃ (E : ℕ), (100000 * E + 200000 * (3 / 2) * E + 200000 * (3 / 2)^2 * E + 200000 * (3 / 2)^3 * E + 200000 * 24 = 16800000) ∧ E = 8 := 
by {
  sorry
}

end NUMINAMATH_GPT_episodes_first_season_l362_36234


namespace NUMINAMATH_GPT_halfway_fraction_l362_36273

theorem halfway_fraction : 
  ∃ (x : ℚ), x = 1/2 * ((2/3) + (4/5)) ∧ x = 11/15 :=
by
  sorry

end NUMINAMATH_GPT_halfway_fraction_l362_36273


namespace NUMINAMATH_GPT_average_book_width_l362_36270

noncomputable def book_widths : List ℚ := [7, 3/4, 1.25, 3, 8, 2.5, 12]
def number_of_books : ℕ := 7
def total_sum_of_widths : ℚ := 34.5

theorem average_book_width :
  ((book_widths.sum) / number_of_books) = 241/49 :=
by
  sorry

end NUMINAMATH_GPT_average_book_width_l362_36270


namespace NUMINAMATH_GPT_area_ratio_of_region_A_and_C_l362_36260

theorem area_ratio_of_region_A_and_C
  (pA : ℕ) (pC : ℕ) 
  (hA : pA = 16)
  (hC : pC = 24) :
  let sA := pA / 4
  let sC := pC / 6
  let areaA := sA * sA
  let areaC := (3 * Real.sqrt 3 / 2) * sC * sC
  (areaA / areaC) = (2 * Real.sqrt 3 / 9) :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_of_region_A_and_C_l362_36260


namespace NUMINAMATH_GPT_shorter_leg_length_l362_36242

theorem shorter_leg_length (m h x : ℝ) (H1 : m = 15) (H2 : h = 3 * x) (H3 : m = 0.5 * h) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_shorter_leg_length_l362_36242


namespace NUMINAMATH_GPT_sandy_savings_l362_36285

-- Definition and conditions
def last_year_savings (S : ℝ) : ℝ := 0.06 * S
def this_year_salary (S : ℝ) : ℝ := 1.10 * S
def this_year_savings (S : ℝ) : ℝ := 1.8333333333333333 * last_year_savings S

-- The percentage P of this year's salary that Sandy saved
def this_year_savings_perc (S : ℝ) (P : ℝ) : Prop :=
  P * this_year_salary S = this_year_savings S

-- The proof statement: Sandy saved 10% of her salary this year
theorem sandy_savings (S : ℝ) (P : ℝ) (h: this_year_savings_perc S P) : P = 0.10 :=
  sorry

end NUMINAMATH_GPT_sandy_savings_l362_36285


namespace NUMINAMATH_GPT_integral_sin_pi_over_2_to_pi_l362_36212

theorem integral_sin_pi_over_2_to_pi : ∫ x in (Real.pi / 2)..Real.pi, Real.sin x = 1 := by
  sorry

end NUMINAMATH_GPT_integral_sin_pi_over_2_to_pi_l362_36212


namespace NUMINAMATH_GPT_MsSatosClassRatioProof_l362_36269

variable (g b : ℕ) -- g is the number of girls, b is the number of boys

def MsSatosClassRatioProblem : Prop :=
  (g = b + 6) ∧ (g + b = 32) → g / b = 19 / 13

theorem MsSatosClassRatioProof : MsSatosClassRatioProblem g b := by
  sorry

end NUMINAMATH_GPT_MsSatosClassRatioProof_l362_36269


namespace NUMINAMATH_GPT_pascal_triangle_fifth_number_l362_36251

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end NUMINAMATH_GPT_pascal_triangle_fifth_number_l362_36251


namespace NUMINAMATH_GPT_find_number_l362_36207

theorem find_number (x : ℝ) (h : 2994 / x = 173) : x = 17.3 := 
sorry

end NUMINAMATH_GPT_find_number_l362_36207


namespace NUMINAMATH_GPT_parabola_properties_l362_36214

open Real 

theorem parabola_properties 
  (a : ℝ) 
  (h₀ : a ≠ 0)
  (h₁ : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0)) :
  (a < 1 / 4 ∧ ∀ x₁ x₂, (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0) → x₁ < 0 ∧ x₂ < 0) ∧
  (∀ (x₁ x₂ C : ℝ), (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0) 
   ∧ (C = a^2) ∧ (-x₁ - x₂ = C - 2) → a = -3) :=
by
  sorry

end NUMINAMATH_GPT_parabola_properties_l362_36214


namespace NUMINAMATH_GPT_right_angled_triangle_l362_36284
  
theorem right_angled_triangle (x : ℝ) (hx : 0 < x) :
  let a := 5 * x
  let b := 12 * x
  let c := 13 * x
  a^2 + b^2 = c^2 :=
by
  let a := 5 * x
  let b := 12 * x
  let c := 13 * x
  sorry

end NUMINAMATH_GPT_right_angled_triangle_l362_36284


namespace NUMINAMATH_GPT_polyomino_count_5_l362_36276

-- Definition of distinct polyomino counts for n = 2, 3, and 4.
def polyomino_count (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 5
  else 0

-- Theorem stating the distinct polyomino count for n = 5
theorem polyomino_count_5 : polyomino_count 5 = 12 :=
by {
  -- Proof steps would go here, but for now we use sorry.
  sorry
}

end NUMINAMATH_GPT_polyomino_count_5_l362_36276


namespace NUMINAMATH_GPT_option_B_is_linear_inequality_with_one_var_l362_36225

noncomputable def is_linear_inequality_with_one_var (in_eq : String) : Prop :=
  match in_eq with
  | "3x^2 > 45 - 9x" => false
  | "3x - 2 < 4" => true
  | "1 / x < 2" => false
  | "4x - 3 < 2y - 7" => false
  | _ => false

theorem option_B_is_linear_inequality_with_one_var :
  is_linear_inequality_with_one_var "3x - 2 < 4" = true :=
by
  -- Add proof steps here
  sorry

end NUMINAMATH_GPT_option_B_is_linear_inequality_with_one_var_l362_36225


namespace NUMINAMATH_GPT_f_neg_l362_36262

/-- Define f(x) as an odd function --/
def f : ℝ → ℝ := sorry

/-- The property of odd functions: f(-x) = -f(x) --/
axiom odd_fn_property (x : ℝ) : f (-x) = -f x

/-- Define the function for non-negative x --/
axiom f_nonneg (x : ℝ) (hx : 0 ≤ x) : f x = x + 1

/-- The goal is to determine f(x) when x < 0 --/
theorem f_neg (x : ℝ) (h : x < 0) : f x = x - 1 :=
by
  sorry

end NUMINAMATH_GPT_f_neg_l362_36262


namespace NUMINAMATH_GPT_round_robin_teams_l362_36252

theorem round_robin_teams (x : ℕ) (h : x * (x - 1) / 2 = 28) : x = 8 :=
sorry

end NUMINAMATH_GPT_round_robin_teams_l362_36252


namespace NUMINAMATH_GPT_average_goals_per_game_l362_36297

theorem average_goals_per_game
  (number_of_pizzas : ℕ)
  (slices_per_pizza : ℕ)
  (number_of_games : ℕ)
  (h1 : number_of_pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : number_of_games = 8) : 
  (number_of_pizzas * slices_per_pizza) / number_of_games = 9 :=
by
  sorry

end NUMINAMATH_GPT_average_goals_per_game_l362_36297


namespace NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l362_36281

section
variables (a b : ℚ)

-- Define the operation
def otimes (a b : ℚ) : ℚ := a * b + abs a - b

-- Prove the three statements
theorem problem_part1 : otimes (-5) 4 = -19 :=
sorry

theorem problem_part2 : otimes (otimes 2 (-3)) 4 = -7 :=
sorry

theorem problem_part3 : otimes 3 (-2) > otimes (-2) 3 :=
sorry
end

end NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l362_36281


namespace NUMINAMATH_GPT_rectangle_area_l362_36279

theorem rectangle_area (b l : ℕ) (h1: l = 3 * b) (h2: 2 * (l + b) = 120) : l * b = 675 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l362_36279


namespace NUMINAMATH_GPT_proof_math_problem_l362_36266

noncomputable def math_problem (a b c d : ℝ) (ω : ℂ) : Prop :=
  a ≠ -1 ∧ b ≠ -1 ∧ c ≠ -1 ∧ d ≠ -1 ∧ 
  ω^4 = 1 ∧ ω ≠ 1 ∧ 
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω^2)

theorem proof_math_problem (a b c d : ℝ) (ω : ℂ) (h: math_problem a b c d ω) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 :=
sorry

end NUMINAMATH_GPT_proof_math_problem_l362_36266


namespace NUMINAMATH_GPT_complete_square_transform_l362_36227

theorem complete_square_transform (x : ℝ) :
  x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_complete_square_transform_l362_36227


namespace NUMINAMATH_GPT_annie_initial_money_l362_36201

theorem annie_initial_money
  (hamburger_price : ℕ := 4)
  (milkshake_price : ℕ := 3)
  (num_hamburgers : ℕ := 8)
  (num_milkshakes : ℕ := 6)
  (money_left : ℕ := 70)
  (total_cost_hamburgers : ℕ := num_hamburgers * hamburger_price)
  (total_cost_milkshakes : ℕ := num_milkshakes * milkshake_price)
  (total_cost : ℕ := total_cost_hamburgers + total_cost_milkshakes)
  : num_hamburgers * hamburger_price + num_milkshakes * milkshake_price + money_left = 120 :=
by
  -- proof part skipped
  sorry

end NUMINAMATH_GPT_annie_initial_money_l362_36201


namespace NUMINAMATH_GPT_simplify_expression_l362_36223

variable (b : ℤ)

theorem simplify_expression :
  (3 * b + 6 - 6 * b) / 3 = -b + 2 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l362_36223


namespace NUMINAMATH_GPT_inequality_proof_l362_36258

theorem inequality_proof 
  (a b c : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c > 0) :
  (a^2 - b^2) / c + (c^2 - b^2) / a + (a^2 - c^2) / b ≥ 3 * a - 4 * b + c :=
  sorry

end NUMINAMATH_GPT_inequality_proof_l362_36258


namespace NUMINAMATH_GPT_intersection_point_lines_distance_point_to_line_l362_36239

-- Problem 1
theorem intersection_point_lines :
  ∃ (x y : ℝ), (x - y + 2 = 0) ∧ (x - 2 * y + 3 = 0) ∧ (x = -1) ∧ (y = 1) :=
sorry

-- Problem 2
theorem distance_point_to_line :
  ∀ (x y : ℝ), (x = 1) ∧ (y = -2) → ∃ d : ℝ, d = 3 ∧ (d = abs (3 * x + 4 * y - 10) / (Real.sqrt (3^2 + 4^2))) :=
sorry

end NUMINAMATH_GPT_intersection_point_lines_distance_point_to_line_l362_36239


namespace NUMINAMATH_GPT_smallest_n_for_nonzero_constant_term_l362_36274

theorem smallest_n_for_nonzero_constant_term : 
  ∃ n : ℕ, (∃ r : ℕ, n = 5 * r / 3) ∧ (n > 0) ∧ ∀ m : ℕ, (m > 0) → (∃ s : ℕ, m = 5 * s / 3) → n ≤ m :=
by sorry

end NUMINAMATH_GPT_smallest_n_for_nonzero_constant_term_l362_36274


namespace NUMINAMATH_GPT_ellipse_minor_axis_length_l362_36253

theorem ellipse_minor_axis_length
  (semi_focal_distance : ℝ)
  (eccentricity : ℝ)
  (semi_focal_distance_eq : semi_focal_distance = 2)
  (eccentricity_eq : eccentricity = 2 / 3) :
  ∃ minor_axis_length : ℝ, minor_axis_length = 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_minor_axis_length_l362_36253


namespace NUMINAMATH_GPT_hoseok_position_l362_36233

variable (total_people : ℕ) (pos_from_back : ℕ)

theorem hoseok_position (h₁ : total_people = 9) (h₂ : pos_from_back = 5) :
  (total_people - pos_from_back + 1) = 5 :=
by
  sorry

end NUMINAMATH_GPT_hoseok_position_l362_36233


namespace NUMINAMATH_GPT_xyz_sum_eq_40_l362_36283

theorem xyz_sum_eq_40
  (x y z : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (h1 : x^2 + x * y + y^2 = 75)
  (h2 : y^2 + y * z + z^2 = 16)
  (h3 : z^2 + x * z + x^2 = 91) :
  x * y + y * z + z * x = 40 :=
sorry

end NUMINAMATH_GPT_xyz_sum_eq_40_l362_36283


namespace NUMINAMATH_GPT_star_polygon_edges_congruent_l362_36261

theorem star_polygon_edges_congruent
  (n : ℕ)
  (α β : ℝ)
  (h1 : ∀ i j : ℕ, i ≠ j → (n = 133))
  (h2 : α = (5 / 14) * β)
  (h3 : n * (α + β) = 360) :
n = 133 :=
by sorry

end NUMINAMATH_GPT_star_polygon_edges_congruent_l362_36261


namespace NUMINAMATH_GPT_positive_int_solution_is_perfect_square_l362_36272

variable (t n : ℤ)

theorem positive_int_solution_is_perfect_square (ht : ∃ n : ℕ, n > 0 ∧ n^2 + (4 * t - 1) * n + 4 * t^2 = 0) : ∃ k : ℕ, n = k^2 :=
  sorry

end NUMINAMATH_GPT_positive_int_solution_is_perfect_square_l362_36272


namespace NUMINAMATH_GPT_staffing_ways_l362_36290

def total_resumes : ℕ := 30
def unsuitable_resumes : ℕ := 10
def suitable_resumes : ℕ := total_resumes - unsuitable_resumes
def position_count : ℕ := 5

theorem staffing_ways :
  20 * 19 * 18 * 17 * 16 = 1860480 := by
  sorry

end NUMINAMATH_GPT_staffing_ways_l362_36290


namespace NUMINAMATH_GPT_bianca_total_pictures_l362_36246

def album1_pictures : Nat := 27
def album2_3_4_pictures : Nat := 3 * 2

theorem bianca_total_pictures : album1_pictures + album2_3_4_pictures = 33 := by
  sorry

end NUMINAMATH_GPT_bianca_total_pictures_l362_36246


namespace NUMINAMATH_GPT_tetrahedron_altitude_exsphere_eq_l362_36294

variable (h₁ h₂ h₃ h₄ r₁ r₂ r₃ r₄ : ℝ)

/-- The equality of the sum of the reciprocals of the heights and the radii of the exspheres of 
a tetrahedron -/
theorem tetrahedron_altitude_exsphere_eq :
  2 * (1 / h₁ + 1 / h₂ + 1 / h₃ + 1 / h₄) = (1 / r₁ + 1 / r₂ + 1 / r₃ + 1 / r₄) :=
sorry

end NUMINAMATH_GPT_tetrahedron_altitude_exsphere_eq_l362_36294


namespace NUMINAMATH_GPT_line_intersects_x_axis_at_point_l362_36236

theorem line_intersects_x_axis_at_point :
  ∃ x, (4 * x - 2 * 0 = 6) ∧ (2 - 0 = 2 * (0 - x)) → x = 2 := 
by
  sorry

end NUMINAMATH_GPT_line_intersects_x_axis_at_point_l362_36236


namespace NUMINAMATH_GPT_lcm_of_a_c_l362_36218

theorem lcm_of_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) : Nat.lcm a c = 30 := by
  sorry

end NUMINAMATH_GPT_lcm_of_a_c_l362_36218


namespace NUMINAMATH_GPT_calculate_expression_l362_36265

theorem calculate_expression : (-1 : ℝ) ^ 2 + (1 / 3 : ℝ) ^ 0 = 2 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l362_36265


namespace NUMINAMATH_GPT_solution_set_of_f_neg_2x_l362_36244

def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem solution_set_of_f_neg_2x (a b : ℝ) (hf_sol : ∀ x : ℝ, (a * x - 1) * (x + b) > 0 ↔ -1 < x ∧ x < 3) :
  ∀ x : ℝ, f a b (-2 * x) < 0 ↔ (x < -3/2 ∨ x > 1/2) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_f_neg_2x_l362_36244


namespace NUMINAMATH_GPT_find_y_l362_36208

theorem find_y (y : ℚ) : (3 / y - (3 / y) * (y / 5) = 1.2) → y = 5 / 3 :=
sorry

end NUMINAMATH_GPT_find_y_l362_36208


namespace NUMINAMATH_GPT_series_sum_eq_4_over_9_l362_36216

noncomputable def sum_series : ℝ := ∑' (k : ℕ), (k+1) / 4^(k+1)

theorem series_sum_eq_4_over_9 : sum_series = 4 / 9 := 
sorry

end NUMINAMATH_GPT_series_sum_eq_4_over_9_l362_36216


namespace NUMINAMATH_GPT_greatest_number_dividing_1642_and_1856_l362_36200

theorem greatest_number_dividing_1642_and_1856 (a b r1 r2 k : ℤ) (h_intro : a = 1642) (h_intro2 : b = 1856) 
    (h_r1 : r1 = 6) (h_r2 : r2 = 4) (h_k1 : k = Int.gcd (a - r1) (b - r2)) :
    k = 4 :=
by
  sorry

end NUMINAMATH_GPT_greatest_number_dividing_1642_and_1856_l362_36200


namespace NUMINAMATH_GPT_F_2457_find_Q_l362_36209

-- Define the properties of a "rising number"
def is_rising_number (m : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    m = 1000 * a + 100 * b + 10 * c + d ∧
    a < b ∧ b < c ∧ c < d ∧
    a + d = b + c

-- Define F(m) as specified
def F (m : ℕ) : ℤ :=
  let a := m / 1000
  let b := (m / 100) % 10
  let c := (m / 10) % 10
  let d := m % 10
  let m' := 1000 * c + 100 * b + 10 * a + d
  (m' - m) / 99

-- Problem statement for F(2457)
theorem F_2457 : F 2457 = 30 := sorry

-- Properties given in the problem statement for P and Q
def is_specific_rising_number (P Q : ℕ) : Prop :=
  ∃ (x y z t : ℕ),
    P = 1000 + 100 * x + 10 * y + z ∧
    Q = 1000 * x + 100 * t + 60 + z ∧
    1 < x ∧ x < t ∧ t < 6 ∧ 6 < z ∧
    1 + z = x + y ∧
    x + z = t + 6 ∧
    F P + F Q % 7 = 0

-- Problem statement to find the value of Q
theorem find_Q (Q : ℕ) : 
  ∃ (P : ℕ), is_specific_rising_number P Q ∧ Q = 3467 := sorry

end NUMINAMATH_GPT_F_2457_find_Q_l362_36209


namespace NUMINAMATH_GPT_minimize_expression_l362_36257

theorem minimize_expression (x : ℝ) : 3 * x^2 - 12 * x + 1 ≥ 3 * 2^2 - 12 * 2 + 1 :=
by sorry

end NUMINAMATH_GPT_minimize_expression_l362_36257


namespace NUMINAMATH_GPT_find_m_range_l362_36219

theorem find_m_range (m : ℝ) (x : ℝ) (h : ∃ c d : ℝ, (c ≠ 0) ∧ (∀ x, (c * x + d)^2 = x^2 + (12 / 5) * x + (2 * m / 5))) : 3.5 ≤ m ∧ m ≤ 3.7 :=
by
  sorry

end NUMINAMATH_GPT_find_m_range_l362_36219


namespace NUMINAMATH_GPT_powers_of_i_sum_l362_36299

theorem powers_of_i_sum :
  ∀ (i : ℂ), 
  (i^1 = i) ∧ (i^2 = -1) ∧ (i^3 = -i) ∧ (i^4 = 1) →
  i^8621 + i^8622 + i^8623 + i^8624 + i^8625 = 0 :=
by
  intros i h
  sorry

end NUMINAMATH_GPT_powers_of_i_sum_l362_36299


namespace NUMINAMATH_GPT_inequality_proof_l362_36293

theorem inequality_proof (x y : ℝ) :
  abs ((x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2))) ≤ 1 / 2 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l362_36293


namespace NUMINAMATH_GPT_rhombus_area_l362_36215

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 120 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l362_36215


namespace NUMINAMATH_GPT_total_six_letter_words_l362_36247

def num_vowels := 6
def vowel_count := 5
def word_length := 6

theorem total_six_letter_words : (num_vowels ^ word_length) = 46656 :=
by sorry

end NUMINAMATH_GPT_total_six_letter_words_l362_36247
