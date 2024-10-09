import Mathlib

namespace tickets_difference_is_cost_l2192_219209

def tickets_won : ℝ := 48.5
def yoyo_cost : ℝ := 11.7
def tickets_left (w : ℝ) (c : ℝ) : ℝ := w - c
def difference (w : ℝ) (l : ℝ) : ℝ := w - l

theorem tickets_difference_is_cost :
  difference tickets_won (tickets_left tickets_won yoyo_cost) = yoyo_cost :=
by
  -- Proof will be written here
  sorry

end tickets_difference_is_cost_l2192_219209


namespace fraction_identity_l2192_219291

theorem fraction_identity (x y z : ℤ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) :
  (x + y) / (3 * y - 2 * z) = 5 :=
by
  sorry

end fraction_identity_l2192_219291


namespace hot_water_bottles_sold_l2192_219266

theorem hot_water_bottles_sold (T H : ℕ) (h1 : 2 * T + 6 * H = 1200) (h2 : T = 7 * H) : H = 60 := 
by 
  sorry

end hot_water_bottles_sold_l2192_219266


namespace tan_alpha_plus_pi_over_4_l2192_219284

theorem tan_alpha_plus_pi_over_4 (x y : ℝ) (h1 : 3 * x + 4 * y = 0) : 
  Real.tan ((Real.arctan (- 3 / 4)) + π / 4) = 1 / 7 := 
by
  sorry

end tan_alpha_plus_pi_over_4_l2192_219284


namespace range_of_m_l2192_219295

noncomputable def has_two_solutions (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 = x₁ + m ∧ x₂^2 = x₂ + m 

theorem range_of_m (m : ℝ) : has_two_solutions m ↔ m > -(1/4) :=
sorry

end range_of_m_l2192_219295


namespace maximum_value_of_linear_expression_l2192_219287

theorem maximum_value_of_linear_expression (m n : ℕ) (h_sum : (m*(m + 1) + n^2 = 1987)) : 3 * m + 4 * n ≤ 221 :=
sorry

end maximum_value_of_linear_expression_l2192_219287


namespace smallest_composite_no_prime_factors_lt_20_l2192_219214

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n : ℕ, (n > 1 ∧ ¬ Prime n ∧ (∀ p : ℕ, Prime p → p < 20 → p ∣ n → False)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_lt_20_l2192_219214


namespace max_term_of_sequence_l2192_219258

noncomputable def a_n (n : ℕ) : ℚ := (n^2 : ℚ) / (2^n : ℚ)

theorem max_term_of_sequence :
  ∃ n : ℕ, (∀ m : ℕ, a_n n ≥ a_n m) ∧ a_n n = 9 / 8 :=
sorry

end max_term_of_sequence_l2192_219258


namespace minimum_value_of_f_l2192_219239

open Real

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sin x

theorem minimum_value_of_f (x : ℝ) (h : abs x ≤ π / 4) : 
  ∃ m : ℝ, (∀ y : ℝ, f y ≥ m) ∧ m = 1 / 2 - sqrt 2 / 2 :=
sorry

end minimum_value_of_f_l2192_219239


namespace complement_intersection_l2192_219213

open Set -- Open the Set namespace

variable (U : Set ℝ := univ)
variable (A : Set ℝ := {x | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2})
variable (B : Set ℝ := {x | x ≤ -1 ∨ x > 2})

theorem complement_intersection :
  (U \ B) ∩ A = {x | x = 0 ∨ x = 1 ∨ x = 2} :=
by
  sorry -- Proof not required as per the instructions

end complement_intersection_l2192_219213


namespace inequality_proof_l2192_219283

theorem inequality_proof (x : ℝ) : 
  (x + 1) / 2 > 1 - (2 * x - 1) / 3 → x > 5 / 7 := 
by
  sorry

end inequality_proof_l2192_219283


namespace trolley_length_l2192_219218

theorem trolley_length (L F : ℝ) (h1 : 4 * L + 3 * F = 108) (h2 : 10 * L + 9 * F = 168) : L = 78 := 
by
  sorry

end trolley_length_l2192_219218


namespace thrown_away_oranges_l2192_219264

theorem thrown_away_oranges (x : ℕ) (h : 40 - x + 7 = 10) : x = 37 :=
by sorry

end thrown_away_oranges_l2192_219264


namespace total_tiles_correct_l2192_219299

-- Definitions for room dimensions
def room_length : ℕ := 24
def room_width : ℕ := 18

-- Definitions for tile dimensions
def border_tile_side : ℕ := 2
def inner_tile_side : ℕ := 1

-- Definitions for border and inner area calculations
def border_width : ℕ := 2 * border_tile_side
def inner_length : ℕ := room_length - border_width
def inner_width : ℕ := room_width - border_width

-- Calculation of the number of tiles needed
def border_area : ℕ := (room_length * room_width) - (inner_length * inner_width)
def num_border_tiles : ℕ := border_area / (border_tile_side * border_tile_side)
def inner_area : ℕ := inner_length * inner_width
def num_inner_tiles : ℕ := inner_area / (inner_tile_side * inner_tile_side)

-- Total number of tiles
def total_tiles : ℕ := num_border_tiles + num_inner_tiles

-- The proof statement
theorem total_tiles_correct : total_tiles = 318 := by
  -- Lean code to check the calculations, proof is omitted.
  sorry

end total_tiles_correct_l2192_219299


namespace g_of_x_l2192_219233

theorem g_of_x (f g : ℕ → ℕ) (h1 : ∀ x, f x = 2 * x + 3)
  (h2 : ∀ x, g (x + 2) = f x) : ∀ x, g x = 2 * x - 1 :=
by
  sorry

end g_of_x_l2192_219233


namespace curve_symmetry_l2192_219223

-- Define the curve as a predicate
def curve (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0

-- Define the point symmetry condition for a line
def is_symmetric_about_line (curve : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, curve x y → line x y

-- Define the line x + y = 0
def line_x_plus_y_eq_0 (x y : ℝ) : Prop := x + y = 0

-- Main theorem stating the curve is symmetrical about the line x + y = 0
theorem curve_symmetry : is_symmetric_about_line curve line_x_plus_y_eq_0 := 
sorry

end curve_symmetry_l2192_219223


namespace john_sells_percentage_of_newspapers_l2192_219274

theorem john_sells_percentage_of_newspapers
    (n_newspapers : ℕ)
    (selling_price : ℝ)
    (cost_price_discount : ℝ)
    (profit : ℝ)
    (sold_percentage : ℝ)
    (h1 : n_newspapers = 500)
    (h2 : selling_price = 2)
    (h3 : cost_price_discount = 0.75)
    (h4 : profit = 550)
    (h5 : sold_percentage = 80) : 
    ( ∃ (sold_n : ℕ), 
      sold_n / n_newspapers * 100 = sold_percentage ∧
      sold_n * selling_price = 
        n_newspapers * selling_price * (1 - cost_price_discount) + profit) :=
by
  sorry

end john_sells_percentage_of_newspapers_l2192_219274


namespace base_of_second_exponent_l2192_219251

theorem base_of_second_exponent (a b : ℕ) (x : ℕ) 
  (h1 : (18^a) * (x^(3 * a - 1)) = (2^6) * (3^b)) 
  (h2 : a = 6) 
  (h3 : 0 < a)
  (h4 : 0 < b) : x = 3 := 
by
  sorry

end base_of_second_exponent_l2192_219251


namespace find_divisor_l2192_219255

theorem find_divisor (x : ℕ) (h : 172 = 10 * x + 2) : x = 17 :=
sorry

end find_divisor_l2192_219255


namespace num_undefined_values_l2192_219216

-- Condition: Denominator is given as (x^2 + 2x - 3)(x - 3)(x + 1)
def denominator (x : ℝ) : ℝ := (x^2 + 2 * x - 3) * (x - 3) * (x + 1)

-- The Lean statement to prove the number of values of x for which the expression is undefined
theorem num_undefined_values : 
  ∃ (n : ℕ), (∀ x : ℝ, denominator x = 0 → (x = 1 ∨ x = -3 ∨ x = 3 ∨ x = -1)) ∧ n = 4 :=
by
  sorry

end num_undefined_values_l2192_219216


namespace sum_of_consecutive_odds_l2192_219279

theorem sum_of_consecutive_odds (N1 N2 N3 : ℕ) (h1 : N1 % 2 = 1) (h2 : N2 % 2 = 1) (h3 : N3 % 2 = 1)
  (h_consec1 : N2 = N1 + 2) (h_consec2 : N3 = N2 + 2) (h_max : N3 = 27) : 
  N1 + N2 + N3 = 75 := by
  sorry

end sum_of_consecutive_odds_l2192_219279


namespace line_intersects_curve_l2192_219292

theorem line_intersects_curve (k : ℝ) :
  (∃ x y : ℝ, y + k * x + 2 = 0 ∧ x^2 + y^2 = 2 * x) ↔ k ≤ -3/4 := by
  sorry

end line_intersects_curve_l2192_219292


namespace stamps_sum_to_n_l2192_219230

noncomputable def selectStamps : Prop :=
  ∀ (n : ℕ) (k : ℕ), n > 0 → 
                      ∃ stamps : List ℕ, 
                      stamps.length = k ∧ 
                      n ≤ stamps.sum ∧ stamps.sum < 2 * k → 
                      ∃ (subset : List ℕ), 
                      subset.sum = n

theorem stamps_sum_to_n : selectStamps := sorry

end stamps_sum_to_n_l2192_219230


namespace evaluate_complex_ratio_l2192_219257

noncomputable def complex_ratio (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^3 + a^2 * b + a * b^2 + b^3 = 0) : ℂ :=
(a^12 + b^12) / (a + b)^12

theorem evaluate_complex_ratio (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^3 + a^2 * b + a * b^2 + b^3 = 0) :
  complex_ratio a b h1 h2 h3 = 1 / 32 :=
by
  sorry

end evaluate_complex_ratio_l2192_219257


namespace find_r_s_l2192_219234

def N : Matrix (Fin 2) (Fin 2) Int := ![![3, 4], ![-2, 0]]
def I : Matrix (Fin 2) (Fin 2) Int := ![![1, 0], ![0, 1]]

theorem find_r_s :
  ∃ (r s : Int), (N * N = r • N + s • I) ∧ (r = 3) ∧ (s = 16) :=
by
  sorry

end find_r_s_l2192_219234


namespace jony_speed_l2192_219273

theorem jony_speed :
  let start_block := 10
  let end_block := 90
  let turn_around_block := 70
  let block_length := 40 -- meters
  let start_time := 0 -- 07:00 in minutes from the start of his walk
  let end_time := 40 -- 07:40 in minutes from the start of his walk
  let total_blocks_walked := (end_block - start_block) + (end_block - turn_around_block)
  let total_distance := total_blocks_walked * block_length
  let total_time := end_time - start_time
  total_distance / total_time = 100 :=
by
  sorry

end jony_speed_l2192_219273


namespace number_of_members_l2192_219215

theorem number_of_members (n : ℕ) (H : n * n = 5776) : n = 76 :=
by
  sorry

end number_of_members_l2192_219215


namespace simplify_expression_l2192_219245

theorem simplify_expression (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c = 0) :
  |a - c| + |c - b| - |a - b| = 0 :=
by
  sorry

end simplify_expression_l2192_219245


namespace area_of_field_l2192_219240

-- Define the conditions: length, width, and total fencing
def length : ℕ := 40
def fencing : ℕ := 74

-- Define the property being proved: the area of the field
theorem area_of_field : ∃ (width : ℕ), 2 * width + length = fencing ∧ length * width = 680 :=
by
  -- Proof omitted
  sorry

end area_of_field_l2192_219240


namespace exists_monochromatic_triangle_l2192_219242

theorem exists_monochromatic_triangle (points : Fin 6 → Point) (color : (Point × Point) → Color) :
  ∃ (a b c : Point), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (color (a, b) = color (b, c) ∧ color (b, c) = color (c, a)) :=
by
  sorry

end exists_monochromatic_triangle_l2192_219242


namespace solve_for_x_l2192_219204

noncomputable def vec (x y : ℝ) : ℝ × ℝ := (x, y)

theorem solve_for_x (x : ℝ) :
  let a := vec 1 2
  let b := vec x 1
  let u := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let v := (2 * a.1 - 2 * b.1, 2 * a.2 - 2 * b.2)
  (u.1 * v.2 = u.2 * v.1) → x = 1 / 2 := by
  sorry

end solve_for_x_l2192_219204


namespace number_of_cats_l2192_219288

-- Defining the context and conditions
variables (x y z : Nat)
variables (h1 : x + y + z = 29) (h2 : x = z)

-- Proving the number of cats
theorem number_of_cats (x y z : Nat) (h1 : x + y + z = 29) (h2 : x = z) :
  6 * x + 3 * y = 87 := by
  sorry

end number_of_cats_l2192_219288


namespace bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3_l2192_219208

variables {a b c : ℝ}
-- Given conditions from Vieta's formulas for the polynomial x^3 - 20x^2 + 22
axiom vieta1 : a + b + c = 20
axiom vieta2 : a * b + b * c + c * a = 0
axiom vieta3 : a * b * c = -22

theorem bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3 (a b c : ℝ)
  (h1 : a + b + c = 20)
  (h2 : a * b + b * c + c * a = 0)
  (h3 : a * b * c = -22) :
  (b * c / a^2) + (a * c / b^2) + (a * b / c^2) = 3 := 
  sorry

end bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3_l2192_219208


namespace JackBuckets_l2192_219241

theorem JackBuckets (tank_capacity buckets_per_trip_jill trips_jill time_ratio trip_buckets_jack : ℕ) :
  tank_capacity = 600 → buckets_per_trip_jill = 5 → trips_jill = 30 →
  time_ratio = 3 / 2 → trip_buckets_jack = 2 :=
  sorry

end JackBuckets_l2192_219241


namespace solve_quadratic_equation_l2192_219243

theorem solve_quadratic_equation (x : ℝ) : x^2 + 4 * x = 5 ↔ x = 1 ∨ x = -5 := sorry

end solve_quadratic_equation_l2192_219243


namespace combined_rocket_height_l2192_219224

variable (h1 : ℕ) (h2 : ℕ)

-- Given conditions
def first_rocket_height : ℕ := 500
def second_rocket_height : ℕ := first_rocket_height * 2

-- Prove that the combined height is 1500 ft
theorem combined_rocket_height : first_rocket_height + second_rocket_height = 1500 := by
  sorry

end combined_rocket_height_l2192_219224


namespace red_suit_top_card_probability_l2192_219203

theorem red_suit_top_card_probability :
  let num_cards := 104
  let num_red_suits := 4
  let cards_per_suit := 26
  let num_red_cards := num_red_suits * cards_per_suit
  let top_card_is_red_probability := num_red_cards / num_cards
  top_card_is_red_probability = 1 := by
  sorry

end red_suit_top_card_probability_l2192_219203


namespace area_of_rhombus_l2192_219247

theorem area_of_rhombus (P D : ℕ) (area : ℝ) (hP : P = 48) (hD : D = 26) :
  area = 25 := by
  sorry

end area_of_rhombus_l2192_219247


namespace geometric_progression_terms_l2192_219272

theorem geometric_progression_terms (b1 b2 bn : ℕ) (q n : ℕ)
  (h1 : b1 = 3) 
  (h2 : b2 = 12)
  (h3 : bn = 3072)
  (h4 : b2 = b1 * q)
  (h5 : bn = b1 * q^(n-1)) : 
  n = 6 := 
by 
  sorry

end geometric_progression_terms_l2192_219272


namespace equal_perimeter_triangle_side_length_l2192_219249

theorem equal_perimeter_triangle_side_length (s: ℝ) : 
    ∀ (pentagon_perimeter triangle_perimeter: ℝ), 
    (pentagon_perimeter = 5 * 5) → 
    (triangle_perimeter = 3 * s) → 
    (pentagon_perimeter = triangle_perimeter) → 
    s = 25 / 3 :=
by
  intro pentagon_perimeter triangle_perimeter h1 h2 h3
  sorry

end equal_perimeter_triangle_side_length_l2192_219249


namespace probability_of_selecting_product_not_less_than_4_l2192_219263

theorem probability_of_selecting_product_not_less_than_4 :
  let total_products := 5 
  let favorable_outcomes := 2 
  (favorable_outcomes : ℚ) / total_products = 2 / 5 := 
by 
  sorry

end probability_of_selecting_product_not_less_than_4_l2192_219263


namespace simplify_fraction_l2192_219246

noncomputable def simplified_expression (x y : ℝ) : ℝ :=
  (x^2 - (4 / y)) / (y^2 - (4 / x))

theorem simplify_fraction {x y : ℝ} (h : x * y ≠ 4) :
  simplified_expression x y = x / y := 
by 
  sorry

end simplify_fraction_l2192_219246


namespace evaluate_expression_l2192_219286

theorem evaluate_expression (x y z : ℤ) (hx : x = 5) (hy : y = x + 3) (hz : z = y - 11) 
  (h₁ : x + 2 ≠ 0) (h₂ : y - 3 ≠ 0) (h₃ : z + 7 ≠ 0) : 
  ((x + 3) / (x + 2)) * ((y - 2) / (y - 3)) * ((z + 9) / (z + 7)) = 72 / 35 := 
by 
  sorry

end evaluate_expression_l2192_219286


namespace oldest_child_age_l2192_219238

theorem oldest_child_age (x : ℕ) (h : (6 + 8 + x) / 3 = 9) : x = 13 :=
by
  sorry

end oldest_child_age_l2192_219238


namespace sequence_is_k_plus_n_l2192_219220

theorem sequence_is_k_plus_n (a : ℕ → ℕ) (k : ℕ) (h : ∀ n : ℕ, a (n + 2) * (a (n + 1) - 1) = a n * (a (n + 1) + 1))
  (pos: ∀ n: ℕ, a n > 0) : ∀ n: ℕ, a n = k + n := 
sorry

end sequence_is_k_plus_n_l2192_219220


namespace exponentiation_condition_l2192_219267

theorem exponentiation_condition (a b : ℝ) (h0 : a > 0) (h1 : a ≠ 1) : 
  (a ^ b > 1 ↔ (a - 1) * b > 0) :=
sorry

end exponentiation_condition_l2192_219267


namespace solve_equation_l2192_219290

theorem solve_equation : 
  ∀ x : ℝ, (x^2 + 2*x + 3)/(x + 2) = x + 4 → x = -(5/4) := by
  sorry

end solve_equation_l2192_219290


namespace sum_of_real_roots_of_even_function_l2192_219271

noncomputable def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem sum_of_real_roots_of_even_function (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_intersects : ∃ a b c d, f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d) :
  a + b + c + d = 0 :=
sorry

end sum_of_real_roots_of_even_function_l2192_219271


namespace α_in_quadrants_l2192_219244

def α (k : ℤ) : ℝ := k * 180 + 45

theorem α_in_quadrants (k : ℤ) : 
  (0 ≤ α k ∧ α k < 90) ∨ (180 < α k ∧ α k ≤ 270) :=
sorry

end α_in_quadrants_l2192_219244


namespace total_area_of_triangles_l2192_219229

theorem total_area_of_triangles :
    let AB := 12
    let DE := 8 * Real.sqrt 2
    let area_ABC := (1 / 2) * AB * AB
    let area_DEF := (1 / 2) * DE * DE * 2
    area_ABC + area_DEF = 136 := by
  sorry

end total_area_of_triangles_l2192_219229


namespace part_a_part_b_part_c_part_d_part_e_l2192_219270

variable (n : ℤ)

theorem part_a : (n^3 - n) % 3 = 0 :=
  sorry

theorem part_b : (n^5 - n) % 5 = 0 :=
  sorry

theorem part_c : (n^7 - n) % 7 = 0 :=
  sorry

theorem part_d : (n^11 - n) % 11 = 0 :=
  sorry

theorem part_e : (n^13 - n) % 13 = 0 :=
  sorry

end part_a_part_b_part_c_part_d_part_e_l2192_219270


namespace tickets_spent_correct_l2192_219260

/-- Tom won 32 tickets playing 'whack a mole'. -/
def tickets_whack_mole : ℕ := 32

/-- Tom won 25 tickets playing 'skee ball'. -/
def tickets_skee_ball : ℕ := 25

/-- Tom is left with 50 tickets after spending some on a hat. -/
def tickets_left : ℕ := 50

/-- The total number of tickets Tom won from both games. -/
def tickets_total : ℕ := tickets_whack_mole + tickets_skee_ball

/-- The number of tickets Tom spent on the hat. -/
def tickets_spent : ℕ := tickets_total - tickets_left

-- Prove that the number of tickets Tom spent on the hat is 7.
theorem tickets_spent_correct : tickets_spent = 7 := by
  -- Proof goes here
  sorry

end tickets_spent_correct_l2192_219260


namespace pies_and_leftover_apples_l2192_219275

theorem pies_and_leftover_apples 
  (apples : ℕ) 
  (h : apples = 55) 
  (h1 : 15/3 = 5) :
  (apples / 5 = 11) ∧ (apples - 11 * 5 = 0) :=
by
  sorry

end pies_and_leftover_apples_l2192_219275


namespace infinitely_many_primes_of_form_6n_plus_5_l2192_219254

theorem infinitely_many_primes_of_form_6n_plus_5 :
  ∃ᶠ p in Filter.atTop, Prime p ∧ p % 6 = 5 :=
sorry

end infinitely_many_primes_of_form_6n_plus_5_l2192_219254


namespace solve_system_equations_l2192_219261

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem solve_system_equations :
  ∃ x y : ℝ, (y = 10^((log10 x)^(log10 x)) ∧ (log10 x)^(log10 (2 * x)) = (log10 y) * 10^((log10 (log10 x))^2))
  → ((x = 10 ∧ y = 10) ∨ (x = 100 ∧ y = 10000)) :=
by
  sorry

end solve_system_equations_l2192_219261


namespace ratio_nine_years_ago_correct_l2192_219206

-- Conditions
def C : ℕ := 24
def G : ℕ := C / 2

-- Question and expected answer
def ratio_nine_years_ago : ℕ := (C - 9) / (G - 9)

theorem ratio_nine_years_ago_correct : ratio_nine_years_ago = 5 := by
  sorry

end ratio_nine_years_ago_correct_l2192_219206


namespace simplify_expression_l2192_219205

theorem simplify_expression (a : ℝ) (h : a ≠ -1) : a - 1 + 1 / (a + 1) = a^2 / (a + 1) :=
  sorry

end simplify_expression_l2192_219205


namespace find_distinct_natural_numbers_l2192_219282

theorem find_distinct_natural_numbers :
  ∃ (x y : ℕ), x ≥ 10 ∧ y ≠ 1 ∧
  (x * y + x) + (x * y - x) + (x * y * x) + (x * y / x) = 576 :=
by
  sorry

end find_distinct_natural_numbers_l2192_219282


namespace sandy_siding_cost_l2192_219297

theorem sandy_siding_cost:
  let wall_width := 8
  let wall_height := 8
  let roof_width := 8
  let roof_height := 5
  let siding_width := 10
  let siding_height := 12
  let siding_cost := 30
  let wall_area := wall_width * wall_height
  let roof_side_area := roof_width * roof_height
  let roof_area := 2 * roof_side_area
  let total_area := wall_area + roof_area
  let siding_area := siding_width * siding_height
  let required_sections := (total_area + siding_area - 1) / siding_area -- ceiling division
  let total_cost := required_sections * siding_cost
  total_cost = 60 :=
by
  sorry

end sandy_siding_cost_l2192_219297


namespace find_weight_of_A_l2192_219262

theorem find_weight_of_A 
  (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : E = D + 5) 
  (h4 : (B + C + D + E) / 4 = 79) 
  : A = 77 := 
sorry

end find_weight_of_A_l2192_219262


namespace correct_system_of_equations_l2192_219281

-- Define the given problem conditions.
def cost_doll : ℝ := 60
def cost_keychain : ℝ := 20
def total_cost : ℝ := 5000

-- Define the condition that each gift set needs 1 doll and 2 keychains.
def gift_set_relation (x y : ℝ) : Prop := 2 * x = y

-- Define the system of equations representing the problem.
def system_of_equations (x y : ℝ) : Prop :=
  2 * x = y ∧
  60 * x + 20 * y = total_cost

-- State the theorem to prove that the given system correctly models the problem.
theorem correct_system_of_equations (x y : ℝ) :
  system_of_equations x y ↔ (2 * x = y ∧ 60 * x + 20 * y = 5000) :=
by sorry

end correct_system_of_equations_l2192_219281


namespace find_z_l2192_219252

open Complex

theorem find_z (z : ℂ) (h : (1 - I) * z = 2 * I) : z = -1 + I := by
  sorry

end find_z_l2192_219252


namespace cleared_land_with_corn_is_630_acres_l2192_219298

-- Definitions based on given conditions
def total_land : ℝ := 6999.999999999999
def cleared_fraction : ℝ := 0.90
def potato_fraction : ℝ := 0.20
def tomato_fraction : ℝ := 0.70

-- Calculate the cleared land
def cleared_land : ℝ := cleared_fraction * total_land

-- Calculate the land used for potato and tomato
def potato_land : ℝ := potato_fraction * cleared_land
def tomato_land : ℝ := tomato_fraction * cleared_land

-- Define the land planted with corn
def corn_land : ℝ := cleared_land - (potato_land + tomato_land)

-- The theorem to be proved
theorem cleared_land_with_corn_is_630_acres : corn_land = 630 := by
  sorry

end cleared_land_with_corn_is_630_acres_l2192_219298


namespace train_times_l2192_219225

theorem train_times (t x : ℝ) : 
  (30 * t = 360) ∧ (36 * (t - x) = 360) → x = 2 :=
by
  sorry

end train_times_l2192_219225


namespace transmission_time_l2192_219210

theorem transmission_time :
  let regular_blocks := 70
  let large_blocks := 30
  let chunks_per_regular_block := 800
  let chunks_per_large_block := 1600
  let channel_rate := 200
  let total_chunks := (regular_blocks * chunks_per_regular_block) + (large_blocks * chunks_per_large_block)
  let total_time_seconds := total_chunks / channel_rate
  let total_time_minutes := total_time_seconds / 60
  total_time_minutes = 8.67 := 
by 
  sorry

end transmission_time_l2192_219210


namespace birthday_candles_l2192_219259

def number_of_red_candles : ℕ := 18
def number_of_green_candles : ℕ := 37
def number_of_yellow_candles := number_of_red_candles / 2
def total_age : ℕ := 85
def total_candles_so_far := number_of_red_candles + number_of_yellow_candles + number_of_green_candles
def number_of_blue_candles := total_age - total_candles_so_far

theorem birthday_candles :
  number_of_yellow_candles = 9 ∧
  number_of_blue_candles = 21 ∧
  (number_of_red_candles + number_of_yellow_candles + number_of_green_candles + number_of_blue_candles) = total_age :=
by
  sorry

end birthday_candles_l2192_219259


namespace problem_statement_l2192_219285

variable {a : ℕ → ℝ} -- the arithmetic sequence
variable {S : ℕ → ℝ} -- the sum of the first n terms of the sequence
variable (d : ℝ) -- the common difference
variable (a1 : ℝ) -- the first term

-- Conditions
axiom arithmetic_sequence (n : ℕ) : a n = a1 + (n - 1) * d
axiom sum_of_first_n_terms (n : ℕ) : S n = n * (a1 + a n) / 2
axiom S_15_eq_45 : S 15 = 45

-- The statement to prove
theorem problem_statement : 2 * a 12 - a 16 = 3 :=
by
  sorry

end problem_statement_l2192_219285


namespace range_of_a_l2192_219227

def f (x a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ a < -3 :=
by sorry

end range_of_a_l2192_219227


namespace probability_odd_product_l2192_219212

theorem probability_odd_product :
  let box1 := [1, 2, 3, 4]
  let box2 := [1, 2, 3, 4]
  let total_outcomes := 4 * 4
  let favorable_outcomes := [(1,1), (1,3), (3,1), (3,3)]
  let num_favorable := favorable_outcomes.length
  (num_favorable / total_outcomes : ℚ) = 1 / 4 := 
by
  sorry

end probability_odd_product_l2192_219212


namespace students_in_classroom_l2192_219248

theorem students_in_classroom :
  ∃ n : ℕ, (n < 50) ∧ (n % 6 = 5) ∧ (n % 3 = 2) ∧ 
  (n = 5 ∨ n = 11 ∨ n = 17 ∨ n = 23 ∨ n = 29 ∨ n = 35 ∨ n = 41 ∨ n = 47) :=
by
  sorry

end students_in_classroom_l2192_219248


namespace f_increasing_interval_l2192_219250

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3 * x - 4)

def domain_f (x : ℝ) : Prop := (x < -1) ∨ (x > 4)

def increasing_g (a b : ℝ) : Prop := ∀ x y, a < x → x < y → y < b → (x^2 - 3 * x - 4 < y^2 - 3 * y - 4)

theorem f_increasing_interval :
  ∀ x, domain_f x → increasing_g 4 (a) → increasing_g 4 (b) → 
    (4 < x ∧ x < b) → (f x < f (b - 0.1)) := sorry

end f_increasing_interval_l2192_219250


namespace inequality_holds_for_positive_x_l2192_219219

theorem inequality_holds_for_positive_x (x : ℝ) (h : 0 < x) :
  (1 + x + x^2) * (1 + x + x^2 + x^3 + x^4) ≤ (1 + x + x^2 + x^3)^2 :=
sorry

end inequality_holds_for_positive_x_l2192_219219


namespace socks_impossible_l2192_219217

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end socks_impossible_l2192_219217


namespace binary_representation_of_28_l2192_219237

-- Define a function to convert a number to binary representation.
def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else 
    let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux n []

theorem binary_representation_of_28 : decimalToBinary 28 = [1, 1, 1, 0, 0] := 
  sorry

end binary_representation_of_28_l2192_219237


namespace prime_arithmetic_progression_difference_divisible_by_6_l2192_219277

theorem prime_arithmetic_progression_difference_divisible_by_6
    (p d : ℕ) (h₀ : Prime p) (h₁ : Prime (p - d)) (h₂ : Prime (p + d))
    (p_neq_3 : p ≠ 3) :
    ∃ (k : ℕ), d = 6 * k := by
  sorry

end prime_arithmetic_progression_difference_divisible_by_6_l2192_219277


namespace correct_option_D_l2192_219202

theorem correct_option_D (a : ℝ) (h : a ≠ 0) : a^0 = 1 :=
by sorry

end correct_option_D_l2192_219202


namespace base_conversion_l2192_219222

theorem base_conversion (k : ℕ) : (5 * 8^2 + 2 * 8^1 + 4 * 8^0 = 6 * k^2 + 6 * k + 4) → k = 7 :=
by 
  let x := 5 * 8^2 + 2 * 8^1 + 4 * 8^0
  have h : x = 340 := by sorry
  have hk : 6 * k^2 + 6 * k + 4 = 340 := by sorry
  sorry

end base_conversion_l2192_219222


namespace three_digit_repeated_digits_percentage_l2192_219226

noncomputable def percentage_repeated_digits : ℝ :=
  let total_numbers := 900
  let non_repeated := 9 * 9 * 8
  let repeated := total_numbers - non_repeated
  (repeated / total_numbers) * 100

theorem three_digit_repeated_digits_percentage :
  percentage_repeated_digits = 28.0 := by
  sorry

end three_digit_repeated_digits_percentage_l2192_219226


namespace log_ride_cost_l2192_219207

noncomputable def cost_of_log_ride (ferris_wheel : ℕ) (roller_coaster : ℕ) (initial_tickets : ℕ) (additional_tickets : ℕ) : ℕ :=
  let total_needed := initial_tickets + additional_tickets
  let total_known := ferris_wheel + roller_coaster
  total_needed - total_known

theorem log_ride_cost :
  cost_of_log_ride 6 5 2 16 = 7 :=
by
  -- specify the values for ferris_wheel, roller_coaster, initial_tickets, additional_tickets
  let ferris_wheel := 6
  let roller_coaster := 5
  let initial_tickets := 2
  let additional_tickets := 16
  -- calculate the cost of the log ride
  let total_needed := initial_tickets + additional_tickets
  let total_known := ferris_wheel + roller_coaster
  let log_ride := total_needed - total_known
  -- assert that the cost of the log ride is 7
  have : log_ride = 7 := by
    -- use arithmetic to justify the answer
    sorry
  exact this

end log_ride_cost_l2192_219207


namespace initial_number_of_persons_l2192_219268

/-- The average weight of some persons increases by 3 kg when a new person comes in place of one of them weighing 65 kg. 
    The weight of the new person might be 89 kg.
    Prove that the number of persons initially was 8.
-/
theorem initial_number_of_persons (n : ℕ) (h1 : (89 - 65 = 3 * n)) : n = 8 := by
  sorry

end initial_number_of_persons_l2192_219268


namespace Marcus_pretzels_l2192_219289

theorem Marcus_pretzels (John_pretzels : ℕ) (Marcus_more_than_John : ℕ) (h1 : John_pretzels = 28) (h2 : Marcus_more_than_John = 12) : Marcus_more_than_John + John_pretzels = 40 :=
by
  sorry

end Marcus_pretzels_l2192_219289


namespace var_power_eight_l2192_219231

variable (k j : ℝ)
variable {x y z : ℝ}

theorem var_power_eight (hx : x = k * y^4) (hy : y = j * z^2) : ∃ c : ℝ, x = c * z^8 :=
by
  sorry

end var_power_eight_l2192_219231


namespace find_a_b_extreme_points_l2192_219280

noncomputable def f (a b x : ℝ) : ℝ := x^3 - 3 * a * x + b

theorem find_a_b (a b : ℝ) (h₁ : a ≠ 0) (h₂ : deriv (f a b) 2 = 0) (h₃ : f a b 2 = 8) : 
  a = 4 ∧ b = 24 :=
by
  sorry

noncomputable def f_deriv (a x : ℝ) : ℝ := 3 * x^2 - 3 * a

theorem extreme_points (a : ℝ) (h₁ : a > 0) : 
  (∃ x: ℝ, f_deriv a x = 0 ∧ 
      ((x = -Real.sqrt a ∧ f a 24 x = 40) ∨ 
       (x = Real.sqrt a ∧ f a 24 x = 16))) := 
by
  sorry

end find_a_b_extreme_points_l2192_219280


namespace worm_length_difference_l2192_219294

def worm_1_length : ℝ := 0.8
def worm_2_length : ℝ := 0.1
def difference := worm_1_length - worm_2_length

theorem worm_length_difference : difference = 0.7 := by
  sorry

end worm_length_difference_l2192_219294


namespace probability_heads_l2192_219228

variable (p : ℝ)
variable (h1 : 0 ≤ p)
variable (h2 : p ≤ 1)
variable (h3 : p * (1 - p) ^ 4 = 0.03125)

theorem probability_heads :
  p = 0.5 :=
sorry

end probability_heads_l2192_219228


namespace intersection_eq_l2192_219201

namespace SetIntersection

open Set

-- Definitions of sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Prove the intersection of A and B is {1, 2}
theorem intersection_eq : A ∩ B = {1, 2} :=
by
  sorry

end SetIntersection

end intersection_eq_l2192_219201


namespace proportional_segments_l2192_219236

theorem proportional_segments (a b c d : ℝ) (h1 : a = 4) (h2 : b = 2) (h3 : c = 3) (h4 : a / b = c / d) : d = 3 / 2 :=
by
  -- proof steps here
  sorry

end proportional_segments_l2192_219236


namespace find_value_of_m_l2192_219211

noncomputable def m : ℤ := -2

theorem find_value_of_m (m : ℤ) :
  (m-2) ≠ 0 ∧ (m^2 - 3 = 1) → m = -2 :=
by
  intros h
  sorry

end find_value_of_m_l2192_219211


namespace sum_of_factors_30_l2192_219200

def sum_of_factors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (λ x => n % x = 0) |>.sum

theorem sum_of_factors_30 : sum_of_factors 30 = 72 := by
  sorry

end sum_of_factors_30_l2192_219200


namespace layla_more_points_than_nahima_l2192_219293

theorem layla_more_points_than_nahima (layla_points : ℕ) (total_points : ℕ) (h1 : layla_points = 70) (h2 : total_points = 112) :
  layla_points - (total_points - layla_points) = 28 :=
by
  sorry

end layla_more_points_than_nahima_l2192_219293


namespace expression_value_l2192_219276

theorem expression_value :
  (6^2 - 3^2)^4 = 531441 := by
  -- Proof steps were omitted
  sorry

end expression_value_l2192_219276


namespace least_number_to_subtract_l2192_219269

theorem least_number_to_subtract (n : ℕ) (h : n = 427398) : 
  ∃ k, (n - k) % 10 = 0 ∧ k = 8 :=
by
  sorry

end least_number_to_subtract_l2192_219269


namespace certain_number_value_l2192_219232

theorem certain_number_value :
  let D := 20
  let S := 55
  3 * D - 5 + (D - S) = 15 :=
by
  -- Definitions for D and S
  let D := 20
  let S := 55
  -- The main assertion
  show 3 * D - 5 + (D - S) = 15
  sorry

end certain_number_value_l2192_219232


namespace find_angle_C_l2192_219235

theorem find_angle_C (a b c A B C : ℝ) (h₀ : 0 < C) (h₁ : C < Real.pi)
  (h₂ : 2 * c * Real.sin A = a * Real.tan C) :
  C = Real.pi / 3 :=
sorry

end find_angle_C_l2192_219235


namespace digital_earth_correct_purposes_l2192_219221

def Purpose : Type := String

def P1 : Purpose := "To deal with natural and social issues of the entire Earth using digital means."
def P2 : Purpose := "To maximize the utilization of natural resources."
def P3 : Purpose := "To conveniently obtain information about the Earth."
def P4 : Purpose := "To provide precise locations, directions of movement, and speeds of moving objects."

def correct_purposes : Set Purpose := {P1, P2, P3}

theorem digital_earth_correct_purposes :
  {P1, P2, P3} = correct_purposes :=
by 
  sorry

end digital_earth_correct_purposes_l2192_219221


namespace domain_all_real_numbers_l2192_219296

theorem domain_all_real_numbers (k : ℝ) :
  (∀ x : ℝ, -7 * x^2 - 4 * x + k ≠ 0) ↔ k < -4 / 7 := by
  sorry

end domain_all_real_numbers_l2192_219296


namespace factorize_expression_l2192_219265

theorem factorize_expression (x : ℝ) : 
  x^4 + 324 = (x^2 - 18 * x + 162) * (x^2 + 18 * x + 162) := 
sorry

end factorize_expression_l2192_219265


namespace ratio_M_N_l2192_219256

theorem ratio_M_N (M Q P N : ℝ) (hM : M = 0.40 * Q) (hQ : Q = 0.25 * P) (hN : N = 0.60 * P) (hP : P ≠ 0) : 
  (M / N) = (1 / 6) := 
by 
  sorry

end ratio_M_N_l2192_219256


namespace elena_marco_sum_ratio_l2192_219253

noncomputable def sum_odds (n : Nat) : Nat := (n / 2 + 1) * n

noncomputable def sum_integers (n : Nat) : Nat := n * (n + 1) / 2

theorem elena_marco_sum_ratio :
  (sum_odds 499) / (sum_integers 250) = 2 :=
by
  sorry

end elena_marco_sum_ratio_l2192_219253


namespace part1_coordinates_of_P_if_AB_perp_PB_part2_coordinates_of_P_area_ABP_10_l2192_219278

-- Part (Ⅰ)
theorem part1_coordinates_of_P_if_AB_perp_PB :
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ (P = (7, 0)) :=
by
  sorry

-- Part (Ⅱ)
theorem part2_coordinates_of_P_area_ABP_10 :
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ (P = (9, 0) ∨ P = (-11, 0)) :=
by
  sorry

end part1_coordinates_of_P_if_AB_perp_PB_part2_coordinates_of_P_area_ABP_10_l2192_219278
