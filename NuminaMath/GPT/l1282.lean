import Mathlib

namespace NUMINAMATH_GPT_vec_eqn_solution_l1282_128215

theorem vec_eqn_solution :
  ∀ m : ℝ, let a : ℝ × ℝ := (1, -2) 
           let b : ℝ × ℝ := (m, 4) 
           (a.1 * b.2 = a.2 * b.1) → 2 • a - b = (4, -8) :=
by
  intro m a b h_parallel
  sorry

end NUMINAMATH_GPT_vec_eqn_solution_l1282_128215


namespace NUMINAMATH_GPT_sum_of_roots_l1282_128210

-- Define the quadratic equation whose roots are the excluded domain values C and D
def quadratic_eq (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0

-- Define C and D as the roots of the quadratic equation
def is_root (x : ℝ) : Prop := quadratic_eq x

-- Define C and D as the specific roots of the given quadratic equation
axiom C : ℝ
axiom D : ℝ

-- Assert that C and D are the roots of the quadratic equation
axiom hC : is_root C
axiom hD : is_root D

-- Statement to prove
theorem sum_of_roots : C + D = 3 :=
by sorry

end NUMINAMATH_GPT_sum_of_roots_l1282_128210


namespace NUMINAMATH_GPT_maximum_xyz_l1282_128274

-- Given conditions
variables {x y z : ℝ}

-- Lean 4 statement with the conditions
theorem maximum_xyz (h₁ : x * y + 2 * z = (x + z) * (y + z))
  (h₂ : x + y + 2 * z = 2)
  (h₃ : 0 < x) (h₄ : 0 < y) (h₅ : 0 < z) :
  xyz = 0 :=
sorry

end NUMINAMATH_GPT_maximum_xyz_l1282_128274


namespace NUMINAMATH_GPT_total_tires_mike_changed_l1282_128264

theorem total_tires_mike_changed (num_motorcycles : ℕ) (tires_per_motorcycle : ℕ)
                                (num_cars : ℕ) (tires_per_car : ℕ)
                                (total_tires : ℕ) :
  num_motorcycles = 12 →
  tires_per_motorcycle = 2 →
  num_cars = 10 →
  tires_per_car = 4 →
  total_tires = num_motorcycles * tires_per_motorcycle + num_cars * tires_per_car →
  total_tires = 64 := by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_tires_mike_changed_l1282_128264


namespace NUMINAMATH_GPT_relationship_abc_l1282_128261

theorem relationship_abc (a b c : ℝ) (ha : a = Real.exp 0.1 - 1) (hb : b = 0.1) (hc : c = Real.log 1.1) :
  c < b ∧ b < a :=
by
  sorry

end NUMINAMATH_GPT_relationship_abc_l1282_128261


namespace NUMINAMATH_GPT_sum_last_two_digits_of_x2012_l1282_128278

def sequence_defined (x : ℕ → ℕ) : Prop :=
  (x 1 = 5 ∨ x 1 = 7) ∧ ∀ k ≥ 1, (x (k+1) = 5^(x k) ∨ x (k+1) = 7^(x k))

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def possible_values : List ℕ :=
  [25, 7, 43]

theorem sum_last_two_digits_of_x2012 {x : ℕ → ℕ} (h : sequence_defined x) :
  List.sum (List.map last_two_digits [25, 7, 43]) = 75 :=
  by
    sorry

end NUMINAMATH_GPT_sum_last_two_digits_of_x2012_l1282_128278


namespace NUMINAMATH_GPT_james_pitbull_count_l1282_128289

-- Defining the conditions
def husky_count : ℕ := 5
def retriever_count : ℕ := 4
def retriever_pups_per_retriever (husky_pups_per_husky : ℕ) : ℕ := husky_pups_per_husky + 2
def husky_pups := husky_count * 3
def retriever_pups := retriever_count * (retriever_pups_per_retriever 3)
def pitbull_pups (P : ℕ) : ℕ := P * 3
def total_pups (P : ℕ) : ℕ := husky_pups + retriever_pups + pitbull_pups P
def total_adults (P : ℕ) : ℕ := husky_count + retriever_count + P
def condition (P : ℕ) : Prop := total_pups P = total_adults P + 30

-- The proof objective
theorem james_pitbull_count : ∃ P : ℕ, condition P → P = 2 := by
  sorry

end NUMINAMATH_GPT_james_pitbull_count_l1282_128289


namespace NUMINAMATH_GPT_mean_weight_correct_l1282_128214

def weights := [51, 60, 62, 64, 64, 65, 67, 73, 74, 74, 75, 76, 77, 78, 79]

noncomputable def mean_weight (weights : List ℕ) : ℚ :=
  (weights.sum : ℚ) / weights.length

theorem mean_weight_correct :
  mean_weight weights = 69.27 := by
  sorry

end NUMINAMATH_GPT_mean_weight_correct_l1282_128214


namespace NUMINAMATH_GPT_radius_of_sphere_touching_four_l1282_128218

noncomputable def r_sphere_internally_touching_four := Real.sqrt (3 / 2) + 1
noncomputable def r_sphere_externally_touching_four := Real.sqrt (3 / 2) - 1

theorem radius_of_sphere_touching_four (r : ℝ) (R := Real.sqrt (3 / 2)) :
  r = R + 1 ∨ r = R - 1 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_sphere_touching_four_l1282_128218


namespace NUMINAMATH_GPT_dennis_pants_purchase_l1282_128213

theorem dennis_pants_purchase
  (pants_cost : ℝ) 
  (pants_discount : ℝ) 
  (socks_cost : ℝ) 
  (socks_discount : ℝ) 
  (socks_quantity : ℕ)
  (total_spent : ℝ)
  (discounted_pants_cost : ℝ)
  (discounted_socks_cost : ℝ)
  (pants_quantity : ℕ) :
  pants_cost = 110.00 →
  pants_discount = 0.30 →
  socks_cost = 60.00 →
  socks_discount = 0.30 →
  socks_quantity = 2 →
  total_spent = 392.00 →
  discounted_pants_cost = pants_cost * (1 - pants_discount) →
  discounted_socks_cost = socks_cost * (1 - socks_discount) →
  total_spent = socks_quantity * discounted_socks_cost + pants_quantity * discounted_ppants_cost →
  pants_quantity = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_dennis_pants_purchase_l1282_128213


namespace NUMINAMATH_GPT_percentage_difference_l1282_128245

theorem percentage_difference :
  let x := 50
  let y := 30
  let p1 := 60
  let p2 := 30
  (p1 / 100 * x) - (p2 / 100 * y) = 21 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1282_128245


namespace NUMINAMATH_GPT_probability_one_white_one_black_l1282_128220

def white_ball_count : ℕ := 8
def black_ball_count : ℕ := 7
def total_ball_count : ℕ := white_ball_count + black_ball_count
def total_ways_to_choose_2_balls : ℕ := total_ball_count.choose 2
def favorable_ways : ℕ := white_ball_count * black_ball_count

theorem probability_one_white_one_black : 
  (favorable_ways : ℚ) / (total_ways_to_choose_2_balls : ℚ) = 8 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_white_one_black_l1282_128220


namespace NUMINAMATH_GPT_expression_even_l1282_128293

theorem expression_even (a b c : ℕ) (ha : a % 2 = 0) (hb : b % 2 = 1) :
  ∃ k : ℕ, 2^a * (b+1) ^ 2 * c = 2 * k :=
by
sorry

end NUMINAMATH_GPT_expression_even_l1282_128293


namespace NUMINAMATH_GPT_minimum_value_frac_abc_l1282_128269

variable (a b c : ℝ)

theorem minimum_value_frac_abc
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + 2 * c = 2) :
  (a + b) / (a * b * c) ≥ 8 :=
sorry

end NUMINAMATH_GPT_minimum_value_frac_abc_l1282_128269


namespace NUMINAMATH_GPT_one_kid_six_whiteboards_l1282_128268

theorem one_kid_six_whiteboards (k: ℝ) (b1 b2: ℝ) (t1 t2: ℝ) 
  (hk: k = 1) (hb1: b1 = 3) (hb2: b2 = 6) 
  (ht1: t1 = 20) 
  (H: 4 * t1 / b1 = t2 / b2) : 
  t2 = 160 := 
by
  -- provide the proof here
  sorry

end NUMINAMATH_GPT_one_kid_six_whiteboards_l1282_128268


namespace NUMINAMATH_GPT_students_behind_yoongi_l1282_128259

theorem students_behind_yoongi (n k : ℕ) (hn : n = 30) (hk : k = 20) : n - (k + 1) = 9 := by
  sorry

end NUMINAMATH_GPT_students_behind_yoongi_l1282_128259


namespace NUMINAMATH_GPT_problem_statement_l1282_128255

theorem problem_statement (x : ℝ) (h : x ≠ 2) :
  (x * (x + 1)) / ((x - 2)^2) ≥ 8 ↔ (1 ≤ x ∧ x < 2) ∨ (32/7 < x) :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1282_128255


namespace NUMINAMATH_GPT_purple_gumdrops_after_replacement_l1282_128225

def total_gumdrops : Nat := 200
def orange_percentage : Nat := 40
def purple_percentage : Nat := 10
def yellow_percentage : Nat := 25
def white_percentage : Nat := 15
def black_percentage : Nat := 10

def initial_orange_gumdrops := (orange_percentage * total_gumdrops) / 100
def initial_purple_gumdrops := (purple_percentage * total_gumdrops) / 100
def orange_to_purple := initial_orange_gumdrops / 3
def final_purple_gumdrops := initial_purple_gumdrops + orange_to_purple

theorem purple_gumdrops_after_replacement : final_purple_gumdrops = 47 := by
  sorry

end NUMINAMATH_GPT_purple_gumdrops_after_replacement_l1282_128225


namespace NUMINAMATH_GPT_classify_numbers_l1282_128211

def isDecimal (n : ℝ) : Prop :=
  ∃ (i : ℤ) (f : ℚ), n = i + f ∧ i ≠ 0

def isNatural (n : ℕ) : Prop :=
  n ≥ 0

theorem classify_numbers :
  (isDecimal 7.42) ∧ (isDecimal 3.6) ∧ (isDecimal 5.23) ∧ (isDecimal 37.8) ∧
  (isNatural 5) ∧ (isNatural 100) ∧ (isNatural 502) ∧ (isNatural 460) :=
by
  sorry

end NUMINAMATH_GPT_classify_numbers_l1282_128211


namespace NUMINAMATH_GPT_candle_problem_l1282_128277

-- Define the initial heights and burn rates of the candles
def heightA (t : ℝ) : ℝ := 12 - 2 * t
def heightB (t : ℝ) : ℝ := 15 - 3 * t

-- Lean theorem statement for the given problem
theorem candle_problem : ∃ t : ℝ, (heightA t = (1/3) * heightB t) ∧ t = 7 :=
by
  -- This is to keep the theorem statement valid without the proof
  sorry

end NUMINAMATH_GPT_candle_problem_l1282_128277


namespace NUMINAMATH_GPT_rahul_batting_average_before_match_l1282_128236

open Nat

theorem rahul_batting_average_before_match (R : ℕ) (A : ℕ) :
  (R + 69 = 6 * 54) ∧ (A = R / 5) → (A = 51) :=
by
  sorry

end NUMINAMATH_GPT_rahul_batting_average_before_match_l1282_128236


namespace NUMINAMATH_GPT_min_num_edges_chromatic_l1282_128230

-- Definition of chromatic number.
def chromatic_number (G : SimpleGraph V) : ℕ := sorry

-- Definition of the number of edges in a graph as a function.
def num_edges (G : SimpleGraph V) : ℕ := sorry

-- Statement of the theorem.
theorem min_num_edges_chromatic (G : SimpleGraph V) (n : ℕ) 
  (chrom_num_G : chromatic_number G = n) : 
  num_edges G ≥ n * (n - 1) / 2 :=
sorry

end NUMINAMATH_GPT_min_num_edges_chromatic_l1282_128230


namespace NUMINAMATH_GPT_cube_face_problem_l1282_128234

theorem cube_face_problem (n : ℕ) (h : 0 < n) :
  ((6 * n^2) : ℚ) / (6 * n^3) = 1 / 3 → n = 3 :=
by
  sorry

end NUMINAMATH_GPT_cube_face_problem_l1282_128234


namespace NUMINAMATH_GPT_quadratic_roots_m_value_l1282_128222

theorem quadratic_roots_m_value
  (x1 x2 m : ℝ)
  (h1 : x1^2 + 2 * x1 + m = 0)
  (h2 : x2^2 + 2 * x2 + m = 0)
  (h3 : x1 + x2 = x1 * x2 - 1) :
  m = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_m_value_l1282_128222


namespace NUMINAMATH_GPT_increasing_intervals_decreasing_interval_l1282_128271

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem increasing_intervals : 
  (∀ x, x < -1/3 → deriv f x > 0) ∧ 
  (∀ x, x > 1 → deriv f x > 0) :=
sorry

theorem decreasing_interval : 
  ∀ x, -1/3 < x ∧ x < 1 → deriv f x < 0 :=
sorry

end NUMINAMATH_GPT_increasing_intervals_decreasing_interval_l1282_128271


namespace NUMINAMATH_GPT_sum_contains_even_digit_l1282_128292

-- Define the five-digit integer and its reversed form
def reversed_digits (n : ℕ) : ℕ := 
  let a := n % 10
  let b := (n / 10) % 10
  let c := (n / 100) % 10
  let d := (n / 1000) % 10
  let e := (n / 10000) % 10
  a * 10000 + b * 1000 + c * 100 + d * 10 + e

theorem sum_contains_even_digit (n m : ℕ) (h1 : n >= 10000) (h2 : n < 100000) (h3 : m = reversed_digits n) : 
  ∃ d : ℕ, d < 10 ∧ d % 2 = 0 ∧ (n + m) % 10 = d ∨ (n + m) / 10 % 10 = d ∨ (n + m) / 100 % 10 = d ∨ (n + m) / 1000 % 10 = d ∨ (n + m) / 10000 % 10 = d := 
sorry

end NUMINAMATH_GPT_sum_contains_even_digit_l1282_128292


namespace NUMINAMATH_GPT_original_price_of_shoes_l1282_128276

theorem original_price_of_shoes (P : ℝ) (h : 0.08 * P = 16) : P = 200 :=
sorry

end NUMINAMATH_GPT_original_price_of_shoes_l1282_128276


namespace NUMINAMATH_GPT_lowest_price_eq_195_l1282_128206

def cost_per_component : ℕ := 80
def shipping_cost_per_unit : ℕ := 5
def fixed_monthly_costs : ℕ := 16500
def num_components : ℕ := 150

theorem lowest_price_eq_195 
  (cost_per_component shipping_cost_per_unit fixed_monthly_costs num_components : ℕ)
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 5)
  (h3 : fixed_monthly_costs = 16500)
  (h4 : num_components = 150) :
  (fixed_monthly_costs + num_components * (cost_per_component + shipping_cost_per_unit)) / num_components = 195 :=
by
  sorry

end NUMINAMATH_GPT_lowest_price_eq_195_l1282_128206


namespace NUMINAMATH_GPT_largest_int_lt_100_div_9_rem_5_l1282_128208

theorem largest_int_lt_100_div_9_rem_5 :
  ∃ a, a < 100 ∧ (a % 9 = 5) ∧ ∀ b, b < 100 ∧ (b % 9 = 5) → b ≤ 95 := by
sorry

end NUMINAMATH_GPT_largest_int_lt_100_div_9_rem_5_l1282_128208


namespace NUMINAMATH_GPT_b_must_be_one_l1282_128237

theorem b_must_be_one (a b : ℝ) (h1 : a + b - a * b = 1) (h2 : ∀ n : ℤ, a ≠ n) : b = 1 :=
sorry

end NUMINAMATH_GPT_b_must_be_one_l1282_128237


namespace NUMINAMATH_GPT_boundary_length_of_divided_rectangle_l1282_128279

/-- Suppose a rectangle is divided into three equal parts along its length and two equal parts along its width, 
creating semicircle arcs connecting points on adjacent sides. Given the rectangle has an area of 72 square units, 
we aim to prove that the total length of the boundary of the resulting figure is 36.0. -/
theorem boundary_length_of_divided_rectangle 
(area_of_rectangle : ℝ)
(length_divisions : ℕ)
(width_divisions : ℕ)
(semicircle_arcs_length : ℝ)
(straight_segments_length : ℝ) :
  area_of_rectangle = 72 →
  length_divisions = 3 →
  width_divisions = 2 →
  semicircle_arcs_length = 7 * Real.pi →
  straight_segments_length = 14 →
  semicircle_arcs_length + straight_segments_length = 36 :=
by
  intros h_area h_length_div h_width_div h_arc_length h_straight_length
  sorry

end NUMINAMATH_GPT_boundary_length_of_divided_rectangle_l1282_128279


namespace NUMINAMATH_GPT_expenditure_ratio_l1282_128250

/-- A man saves 35% of his income in the first year. -/
def saving_rate_first_year : ℝ := 0.35

/-- His income increases by 35% in the second year. -/
def income_increase_rate : ℝ := 0.35

/-- His savings increase by 100% in the second year. -/
def savings_increase_rate : ℝ := 1.0

theorem expenditure_ratio
  (I : ℝ)  -- first year income
  (S1 : ℝ := saving_rate_first_year * I)  -- first year saving
  (E1 : ℝ := I - S1)  -- first year expenditure
  (I2 : ℝ := I + income_increase_rate * I)  -- second year income
  (S2 : ℝ := 2 * S1)  -- second year saving (increases by 100%)
  (E2 : ℝ := I2 - S2)  -- second year expenditure
  :
  (E1 + E2) / E1 = 2
  :=
  sorry

end NUMINAMATH_GPT_expenditure_ratio_l1282_128250


namespace NUMINAMATH_GPT_numerical_value_expression_l1282_128251

theorem numerical_value_expression (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 1 / (a^2 + 1) + 1 / (b^2 + 1) = 2 / (ab + 1)) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 2 / (ab + 1) = 2 := 
by 
  -- Proof outline provided in the solution section, but actual proof is omitted
  sorry

end NUMINAMATH_GPT_numerical_value_expression_l1282_128251


namespace NUMINAMATH_GPT_fizz_preference_count_l1282_128282

-- Definitions from conditions
def total_people : ℕ := 500
def fizz_angle : ℕ := 270
def total_angle : ℕ := 360
def fizz_fraction : ℚ := fizz_angle / total_angle

-- The target proof statement
theorem fizz_preference_count (hp : total_people = 500) 
                              (ha : fizz_angle = 270) 
                              (ht : total_angle = 360)
                              (hf : fizz_fraction = 3 / 4) : 
    total_people * fizz_fraction = 375 := by
    sorry

end NUMINAMATH_GPT_fizz_preference_count_l1282_128282


namespace NUMINAMATH_GPT_num_cars_in_parking_lot_l1282_128221

-- Define the conditions
variable (C : ℕ) -- Number of cars
def number_of_bikes := 5 -- Number of bikes given
def total_wheels := 66 -- Total number of wheels given
def wheels_per_bike := 2 -- Number of wheels per bike
def wheels_per_car := 4 -- Number of wheels per car

-- Define the proof statement
theorem num_cars_in_parking_lot 
  (h1 : total_wheels = 66) 
  (h2 : number_of_bikes = 5) 
  (h3 : wheels_per_bike = 2)
  (h4 : wheels_per_car = 4) 
  (h5 : C * wheels_per_car + number_of_bikes * wheels_per_bike = total_wheels) :
  C = 14 :=
by
  sorry

end NUMINAMATH_GPT_num_cars_in_parking_lot_l1282_128221


namespace NUMINAMATH_GPT_smallest_number_divisible_by_set_l1282_128247

theorem smallest_number_divisible_by_set : ∃ x : ℕ, (∀ d ∈ [12, 24, 36, 48, 56, 72, 84], (x - 24) % d = 0) ∧ x = 1032 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_divisible_by_set_l1282_128247


namespace NUMINAMATH_GPT_room_length_l1282_128262

theorem room_length (L : ℕ) (h : 72 * L + 918 = 2718) : L = 25 := by
  sorry

end NUMINAMATH_GPT_room_length_l1282_128262


namespace NUMINAMATH_GPT_find_special_four_digit_square_l1282_128281

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ (x : ℕ), x * x = n

def same_first_two_digits (n : ℕ) : Prop := (n / 1000) = (n / 100 % 10)

def same_last_two_digits (n : ℕ) : Prop := (n % 100 / 10) = (n % 10)

theorem find_special_four_digit_square :
  ∃ (n : ℕ), is_four_digit n ∧ is_perfect_square n ∧ same_first_two_digits n ∧ same_last_two_digits n ∧ n = 7744 := 
sorry

end NUMINAMATH_GPT_find_special_four_digit_square_l1282_128281


namespace NUMINAMATH_GPT_find_digits_l1282_128288

/-- 
  Find distinct digits A, B, C, and D such that 9 * (100 * A + 10 * B + C) = B * (1000 * B + 100 * C + 10 * D + B).
 -/
theorem find_digits
  (A B C D : ℕ)
  (hA : A ≠ B) (hA : A ≠ C) (hA : A ≠ D)
  (hB : B ≠ C) (hB : B ≠ D)
  (hC : C ≠ D)
  (hNonZeroB : B ≠ 0) :
  9 * (100 * A + 10 * B + C) = B * (1000 * B + 100 * C + 10 * D + B) ↔ (A = 2 ∧ B = 1 ∧ C = 9 ∧ D = 7) := by
  sorry

end NUMINAMATH_GPT_find_digits_l1282_128288


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1282_128286

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∃ x, x > 2 ∧ ¬ (x > 3)) ∧ 
  (∀ x, x > 3 → x > 2) := by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1282_128286


namespace NUMINAMATH_GPT_expression_eval_neg_sqrt_l1282_128265

variable (a : ℝ)

theorem expression_eval_neg_sqrt (ha : a < 0) : a * Real.sqrt (-1 / a) = -Real.sqrt (-a) :=
by
  sorry

end NUMINAMATH_GPT_expression_eval_neg_sqrt_l1282_128265


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l1282_128248

open Real

theorem largest_angle_in_triangle
  (A B C : ℝ)
  (h : sin A / sin B / sin C = 1 / sqrt 2 / sqrt 5) :
  A ≤ B ∧ B ≤ C → C = 3 * π / 4 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l1282_128248


namespace NUMINAMATH_GPT_function_is_zero_l1282_128256

variable (n : ℕ) (a : Fin n → ℤ) (f : ℤ → ℝ)

axiom condition : ∀ (k l : ℤ), l ≠ 0 → (Finset.univ.sum (λ i => f (k + a i * l)) = 0)

theorem function_is_zero : ∀ x : ℤ, f x = 0 := by
  sorry

end NUMINAMATH_GPT_function_is_zero_l1282_128256


namespace NUMINAMATH_GPT_words_per_page_l1282_128239

theorem words_per_page (p : ℕ) :
  (p ≤ 120) ∧ (154 * p % 221 = 145) → p = 96 := by
  sorry

end NUMINAMATH_GPT_words_per_page_l1282_128239


namespace NUMINAMATH_GPT_eighth_term_geometric_sequence_l1282_128240

theorem eighth_term_geometric_sequence (a r : ℝ) (n : ℕ) (h_a : a = 12) (h_r : r = 1/4) (h_n : n = 8) :
  a * r^(n - 1) = 3 / 4096 := 
by 
  sorry

end NUMINAMATH_GPT_eighth_term_geometric_sequence_l1282_128240


namespace NUMINAMATH_GPT_messages_per_member_per_day_l1282_128209

theorem messages_per_member_per_day (initial_members removed_members remaining_members total_weekly_messages total_daily_messages : ℕ)
  (h1 : initial_members = 150)
  (h2 : removed_members = 20)
  (h3 : remaining_members = initial_members - removed_members)
  (h4 : total_weekly_messages = 45500)
  (h5 : total_daily_messages = total_weekly_messages / 7)
  (h6 : 7 * total_daily_messages = total_weekly_messages) -- ensures that total_daily_messages calculated is correct
  : total_daily_messages / remaining_members = 50 := 
by
  sorry

end NUMINAMATH_GPT_messages_per_member_per_day_l1282_128209


namespace NUMINAMATH_GPT_saved_money_is_30_l1282_128224

def week_payout : ℕ := 5 * 3
def total_payout (weeks: ℕ) : ℕ := weeks * week_payout
def shoes_cost : ℕ := 120
def remaining_weeks : ℕ := 6
def remaining_earnings : ℕ := total_payout remaining_weeks
def saved_money : ℕ := shoes_cost - remaining_earnings

theorem saved_money_is_30 : saved_money = 30 := by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_saved_money_is_30_l1282_128224


namespace NUMINAMATH_GPT_total_number_of_fish_l1282_128290

def number_of_tuna : Nat := 5
def number_of_spearfish : Nat := 2

theorem total_number_of_fish : number_of_tuna + number_of_spearfish = 7 := by
  sorry

end NUMINAMATH_GPT_total_number_of_fish_l1282_128290


namespace NUMINAMATH_GPT_y_intercept_of_line_with_slope_3_and_x_intercept_7_0_l1282_128275

def line_equation (m x1 y1 x y : ℝ) : Prop :=
  y - y1 = m * (x - x1)

theorem y_intercept_of_line_with_slope_3_and_x_intercept_7_0 :
  ∃ b : ℝ, line_equation 3 7 0 0 b ∧ b = -21 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_with_slope_3_and_x_intercept_7_0_l1282_128275


namespace NUMINAMATH_GPT_explicit_formula_of_odd_function_monotonicity_in_interval_l1282_128294

-- Using Noncomputable because divisions are involved.
noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := (p * x^2 + 2) / (q - 3 * x)

theorem explicit_formula_of_odd_function (p q : ℝ) 
  (h_odd : ∀ x : ℝ, f x p q = - f (-x) p q) 
  (h_value : f 2 p q = -5/3) : 
  f x 2 0 = -2/3 * (x + 1/x) :=
by sorry

theorem monotonicity_in_interval {x : ℝ} (h_domain : 0 < x ∧ x < 1) : 
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 -> f x1 2 0 < f x2 2 0 :=
by sorry

end NUMINAMATH_GPT_explicit_formula_of_odd_function_monotonicity_in_interval_l1282_128294


namespace NUMINAMATH_GPT_min_solution_of_x_abs_x_eq_3x_plus_4_l1282_128232

theorem min_solution_of_x_abs_x_eq_3x_plus_4 : 
  ∃ x : ℝ, (x * |x| = 3 * x + 4) ∧ ∀ y : ℝ, (y * |y| = 3 * y + 4) → x ≤ y :=
sorry

end NUMINAMATH_GPT_min_solution_of_x_abs_x_eq_3x_plus_4_l1282_128232


namespace NUMINAMATH_GPT_range_of_k_l1282_128287

noncomputable def f (k x : ℝ) := (k * x + 7) / (k * x^2 + 4 * k * x + 3)

theorem range_of_k (k : ℝ) : (∀ x : ℝ, k * x^2 + 4 * k * x + 3 ≠ 0) ↔ 0 ≤ k ∧ k < 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1282_128287


namespace NUMINAMATH_GPT_arithmetic_mean_of_multiples_of_6_l1282_128299

/-- The smallest three-digit multiple of 6 is 102. -/
def smallest_multiple_of_6 : ℕ := 102

/-- The largest three-digit multiple of 6 is 996. -/
def largest_multiple_of_6 : ℕ := 996

/-- The common difference in the arithmetic sequence of multiples of 6 is 6. -/
def common_difference_of_sequence : ℕ := 6

/-- The number of terms in the arithmetic sequence of three-digit multiples of 6. -/
def number_of_terms : ℕ := (largest_multiple_of_6 - smallest_multiple_of_6) / common_difference_of_sequence + 1

/-- The sum of the arithmetic sequence of three-digit multiples of 6. -/
def sum_of_sequence : ℕ := number_of_terms * (smallest_multiple_of_6 + largest_multiple_of_6) / 2

/-- The arithmetic mean of all positive three-digit multiples of 6 is 549. -/
theorem arithmetic_mean_of_multiples_of_6 : 
  let mean := sum_of_sequence / number_of_terms
  mean = 549 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_multiples_of_6_l1282_128299


namespace NUMINAMATH_GPT_arithmetic_mean_of_remaining_numbers_l1282_128252

-- Definitions and conditions
def initial_set_size : ℕ := 60
def initial_arithmetic_mean : ℕ := 45
def numbers_to_remove : List ℕ := [50, 55, 60]

-- Calculation of the total sum
def total_sum : ℕ := initial_arithmetic_mean * initial_set_size

-- Calculation of the sum of the numbers to remove
def sum_of_removed_numbers : ℕ := numbers_to_remove.sum

-- Sum of the remaining numbers
def new_sum : ℕ := total_sum - sum_of_removed_numbers

-- Size of the remaining set
def remaining_set_size : ℕ := initial_set_size - numbers_to_remove.length

-- The arithmetic mean of the remaining numbers
def new_arithmetic_mean : ℚ := new_sum / remaining_set_size

-- The proof statement
theorem arithmetic_mean_of_remaining_numbers :
  new_arithmetic_mean = 2535 / 57 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_remaining_numbers_l1282_128252


namespace NUMINAMATH_GPT_unused_types_l1282_128298

theorem unused_types (total_resources : ℕ) (used_types : ℕ) (valid_types : ℕ) :
  total_resources = 6 → used_types = 23 → valid_types = 2^total_resources - 1 - used_types → valid_types = 40 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end NUMINAMATH_GPT_unused_types_l1282_128298


namespace NUMINAMATH_GPT_no_partition_of_six_consecutive_numbers_product_equal_l1282_128231

theorem no_partition_of_six_consecutive_numbers_product_equal (n : ℕ) :
  ¬ ∃ (A B : Finset ℕ), 
    A ∪ B = Finset.range (n+6) ∧ 
    A ∩ B = ∅ ∧ 
    A.prod id = B.prod id :=
by
  sorry

end NUMINAMATH_GPT_no_partition_of_six_consecutive_numbers_product_equal_l1282_128231


namespace NUMINAMATH_GPT_total_distance_in_land_miles_l1282_128238

-- Definitions based on conditions
def speed_one_sail : ℕ := 25
def time_one_sail : ℕ := 4
def distance_one_sail := speed_one_sail * time_one_sail

def speed_two_sails : ℕ := 50
def time_two_sails : ℕ := 4
def distance_two_sails := speed_two_sails * time_two_sails

def conversion_factor : ℕ := 115  -- Note: 1.15 * 100 for simplicity with integers

-- Theorem to prove the total distance in land miles
theorem total_distance_in_land_miles : (distance_one_sail + distance_two_sails) * conversion_factor / 100 = 345 := by
  sorry

end NUMINAMATH_GPT_total_distance_in_land_miles_l1282_128238


namespace NUMINAMATH_GPT_commutative_star_l1282_128223

def star (a b : ℤ) : ℤ := a^2 + b^2

theorem commutative_star (a b : ℤ) : star a b = star b a :=
by sorry

end NUMINAMATH_GPT_commutative_star_l1282_128223


namespace NUMINAMATH_GPT_same_solution_implies_value_of_m_l1282_128284

theorem same_solution_implies_value_of_m (x m : ℤ) (h₁ : -5 * x - 6 = 3 * x + 10) (h₂ : -2 * m - 3 * x = 10) : m = -2 :=
by
  sorry

end NUMINAMATH_GPT_same_solution_implies_value_of_m_l1282_128284


namespace NUMINAMATH_GPT_smallest_four_digit_number_divisible_by_35_l1282_128226

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_number_divisible_by_35_l1282_128226


namespace NUMINAMATH_GPT_volume_removed_percentage_l1282_128242

noncomputable def original_volume : ℕ := 20 * 15 * 10

noncomputable def cube_volume : ℕ := 4 * 4 * 4

noncomputable def total_volume_removed : ℕ := 8 * cube_volume

noncomputable def percentage_volume_removed : ℝ :=
  (total_volume_removed : ℝ) / (original_volume : ℝ) * 100

theorem volume_removed_percentage :
  percentage_volume_removed = 512 / 30 := sorry

end NUMINAMATH_GPT_volume_removed_percentage_l1282_128242


namespace NUMINAMATH_GPT_intersection_nonempty_range_b_l1282_128200

noncomputable def A : Set ℝ := {x | (x - 1) / (x + 1) < 0}
noncomputable def B (b : ℝ) (a : ℝ) : Set ℝ := {x | (x - b)^2 < a}

theorem intersection_nonempty_range_b (b : ℝ) : 
  A ∩ B b 1 ≠ ∅ ↔ -2 < b ∧ b < 2 := 
by
  sorry

end NUMINAMATH_GPT_intersection_nonempty_range_b_l1282_128200


namespace NUMINAMATH_GPT_percentage_calculation_l1282_128249

theorem percentage_calculation (percentage : ℝ) (h : percentage * 50 = 0.15) : percentage = 0.003 :=
by
  sorry

end NUMINAMATH_GPT_percentage_calculation_l1282_128249


namespace NUMINAMATH_GPT_Jim_catches_Bob_in_20_minutes_l1282_128285

theorem Jim_catches_Bob_in_20_minutes
  (Bob_Speed : ℕ := 6)
  (Jim_Speed : ℕ := 9)
  (Head_Start : ℕ := 1) :
  (Head_Start / (Jim_Speed - Bob_Speed) * 60 = 20) :=
by
  sorry

end NUMINAMATH_GPT_Jim_catches_Bob_in_20_minutes_l1282_128285


namespace NUMINAMATH_GPT_find_f_2_l1282_128246

theorem find_f_2 (f : ℝ → ℝ) (h₁ : f 1 = 0)
  (h₂ : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)) :
  f 2 = 0 :=
sorry

end NUMINAMATH_GPT_find_f_2_l1282_128246


namespace NUMINAMATH_GPT_proof_problem_l1282_128267

noncomputable def a : ℝ := 3.54
noncomputable def b : ℝ := 1.32
noncomputable def result : ℝ := (a - b) * 2

theorem proof_problem : result = 4.44 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1282_128267


namespace NUMINAMATH_GPT_fraction_sum_l1282_128291

theorem fraction_sum : (3 / 8) + (9 / 14) = (57 / 56) := by
  sorry

end NUMINAMATH_GPT_fraction_sum_l1282_128291


namespace NUMINAMATH_GPT_linear_relation_is_correct_maximum_profit_l1282_128297

-- Define the given data points
structure DataPoints where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

-- Define the given conditions
def conditions : DataPoints := ⟨50, 100, 60, 90⟩

-- Define the cost and sell price range conditions
def cost_per_kg : ℝ := 20
def max_selling_price : ℝ := 90

-- Define the linear relationship function
def linear_relationship (k b x : ℝ) : ℝ := k * x + b

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - cost_per_kg) * (linear_relationship (-1) 150 x)

-- Statements to Prove
theorem linear_relation_is_correct (k b : ℝ) :
  linear_relationship k b 50 = 100 ∧
  linear_relationship k b 60 = 90 →
  (b = 150 ∧ k = -1) := by
  intros h
  sorry

theorem maximum_profit :
  ∃ x : ℝ, 20 ≤ x ∧ x ≤ max_selling_price ∧ profit_function x = 4225 := by
  use 85
  sorry

end NUMINAMATH_GPT_linear_relation_is_correct_maximum_profit_l1282_128297


namespace NUMINAMATH_GPT_all_stones_weigh_the_same_l1282_128244

theorem all_stones_weigh_the_same (x : Fin 13 → ℕ)
  (h : ∀ (i : Fin 13), ∃ (A B : Finset (Fin 13)), A.card = 6 ∧ B.card = 6 ∧
    i ∉ A ∧ i ∉ B ∧ ∀ (j k : Fin 13), j ∈ A → k ∈ B → x j = x k): 
  ∀ i j : Fin 13, x i = x j := 
sorry

end NUMINAMATH_GPT_all_stones_weigh_the_same_l1282_128244


namespace NUMINAMATH_GPT_problem_l1282_128273

theorem problem (x y z : ℝ) (h1 : x = y + z) (h2 : x = 2) : 
  x^3 + 3 * y^2 + 3 * z^2 + 3 * x * y * z = 20 := by
sorry

end NUMINAMATH_GPT_problem_l1282_128273


namespace NUMINAMATH_GPT_intercepts_equal_lines_parallel_l1282_128295

-- Definition of the conditions: line equations
def line_l (a : ℝ) : Prop := ∀ x y : ℝ, a * x + 3 * y + 1 = 0

-- Problem (1) : The intercepts of the line on the two coordinate axes are equal
theorem intercepts_equal (a : ℝ) (h : line_l a) : a = 3 := by
  sorry

-- Problem (2): The line is parallel to x + (a-2)y + a = 0
theorem lines_parallel (a : ℝ) (h : line_l a) : (∀ x y : ℝ, x + (a-2) * y + a = 0) → a = 3 := by
  sorry

end NUMINAMATH_GPT_intercepts_equal_lines_parallel_l1282_128295


namespace NUMINAMATH_GPT_cos_double_angle_l1282_128217

theorem cos_double_angle (x : ℝ) (h : 2 * Real.sin (Real.pi - x) + 1 = 0) : Real.cos (2 * x) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_l1282_128217


namespace NUMINAMATH_GPT_invalid_votes_percentage_l1282_128219

def total_votes : ℕ := 560000
def valid_votes_A : ℕ := 357000
def percentage_A : ℝ := 0.75
def invalid_percentage (x : ℝ) : Prop := (percentage_A * (1 - x / 100) * total_votes = valid_votes_A)

theorem invalid_votes_percentage : ∃ x : ℝ, invalid_percentage x ∧ x = 15 :=
by 
  use 15
  unfold invalid_percentage
  sorry

end NUMINAMATH_GPT_invalid_votes_percentage_l1282_128219


namespace NUMINAMATH_GPT_hike_down_distance_l1282_128233

theorem hike_down_distance :
  let rate_up := 4 -- rate going up in miles per day
  let time := 2    -- time in days
  let rate_down := 1.5 * rate_up -- rate going down in miles per day
  let distance_down := rate_down * time -- distance going down in miles
  distance_down = 12 :=
by
  sorry

end NUMINAMATH_GPT_hike_down_distance_l1282_128233


namespace NUMINAMATH_GPT_inequality_triangle_area_l1282_128203

-- Define the triangles and their properties
variables {α β γ : Real} -- Internal angles of triangle ABC
variables {r : Real} -- Circumradius of triangle ABC
variables {P Q : Real} -- Areas of triangles ABC and A'B'C' respectively

-- Define the bisectors and intersect points
-- Note: For the purpose of this proof, we're not explicitly defining the geometry
-- of the inner bisectors and intersect points but working from the given conditions.

theorem inequality_triangle_area
  (h1 : P = r^2 * (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) / 2)
  (h2 : Q = r^2 * (Real.sin (β + γ) + Real.sin (γ + α) + Real.sin (α + β)) / 2) :
  16 * Q^3 ≥ 27 * r^4 * P :=
sorry

end NUMINAMATH_GPT_inequality_triangle_area_l1282_128203


namespace NUMINAMATH_GPT_problem_l1282_128204

variable (a b : ℝ)

theorem problem (h : a = 1.25 * b) : (4 * b) / a = 3.2 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1282_128204


namespace NUMINAMATH_GPT_find_time_eating_dinner_l1282_128207

def total_flight_time : ℕ := 11 * 60 + 20
def time_reading : ℕ := 2 * 60
def time_watching_movies : ℕ := 4 * 60
def time_listening_radio : ℕ := 40
def time_playing_games : ℕ := 1 * 60 + 10
def time_nap : ℕ := 3 * 60

theorem find_time_eating_dinner : 
  total_flight_time - (time_reading + time_watching_movies + time_listening_radio + time_playing_games + time_nap) = 30 := 
by
  sorry

end NUMINAMATH_GPT_find_time_eating_dinner_l1282_128207


namespace NUMINAMATH_GPT_optimal_position_theorem_l1282_128235

noncomputable def optimal_position (a b a1 b1 : ℝ) : ℝ :=
  (b / 2) + (b1 / (2 * a1)) * (a - a1)

theorem optimal_position_theorem 
  (a b a1 b1 : ℝ) (ha1 : a1 > 0) (hb1 : b1 > 0) :
  ∃ x, x = optimal_position a b a1 b1 := by
  sorry

end NUMINAMATH_GPT_optimal_position_theorem_l1282_128235


namespace NUMINAMATH_GPT_max_value_a_l1282_128266

-- Define the variables and the constraint on the circle
def circular_arrangement_condition (x: ℕ → ℕ) : Prop :=
  ∀ i: ℕ, 1 ≤ x i ∧ x i ≤ 10 ∧ x i ≠ x (i + 1)

-- Define the existence of three consecutive numbers summing to at least 18
def three_consecutive_sum_ge_18 (x: ℕ → ℕ) : Prop :=
  ∃ i: ℕ, x i + x (i + 1) + x (i + 2) ≥ 18

-- The main theorem we aim to prove
theorem max_value_a : ∀ (x: ℕ → ℕ), circular_arrangement_condition x → three_consecutive_sum_ge_18 x :=
  by sorry

end NUMINAMATH_GPT_max_value_a_l1282_128266


namespace NUMINAMATH_GPT_smallest_gcd_bc_l1282_128202

theorem smallest_gcd_bc (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (gcd_ab : Nat.gcd a b = 168) (gcd_ac : Nat.gcd a c = 693) : Nat.gcd b c = 21 := 
sorry

end NUMINAMATH_GPT_smallest_gcd_bc_l1282_128202


namespace NUMINAMATH_GPT_increasing_range_of_a_l1282_128260

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 1 / x

theorem increasing_range_of_a (a : ℝ) : (∀ x > (1/2), (3 * x^2 + a - 1 / x^2) ≥ 0) ↔ a ≥ (13 / 4) :=
by sorry

end NUMINAMATH_GPT_increasing_range_of_a_l1282_128260


namespace NUMINAMATH_GPT_minimum_value_of_f_l1282_128212

def f (x : ℝ) : ℝ := |3 - x| + |x - 2|

theorem minimum_value_of_f : ∃ x0 : ℝ, (∀ x : ℝ, f x0 ≤ f x) ∧ f x0 = 1 := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1282_128212


namespace NUMINAMATH_GPT_sin_cos_identity_l1282_128280

theorem sin_cos_identity (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) : Real.sin x + 5 * Real.cos x = -28 / 13 := 
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l1282_128280


namespace NUMINAMATH_GPT_third_height_less_than_30_l1282_128257

theorem third_height_less_than_30 (h_a h_b : ℝ) (h_a_pos : h_a = 12) (h_b_pos : h_b = 20) : 
    ∃ (h_c : ℝ), h_c < 30 :=
by
  sorry

end NUMINAMATH_GPT_third_height_less_than_30_l1282_128257


namespace NUMINAMATH_GPT_evaluate_expression_at_x_eq_3_l1282_128270

theorem evaluate_expression_at_x_eq_3 : (3 ^ 3) ^ (3 ^ 3) = 27 ^ 27 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_x_eq_3_l1282_128270


namespace NUMINAMATH_GPT_avg_first_six_results_l1282_128296

theorem avg_first_six_results (A : ℝ) :
  (∀ (results : Fin 12 → ℝ), 
    (results 0 + results 1 + results 2 + results 3 + results 4 + results 5 + 
     results 6 + results 7 + results 8 + results 9 + results 10 + results 11) / 11 = 60 → 
    (results 0 + results 1 + results 2 + results 3 + results 4 + results 5) / 6 = A → 
    (results 5 + results 6 + results 7 + results 8 + results 9 + results 10) / 6 = 63 → 
    results 5 = 66) → 
  A = 58 :=
by
  sorry

end NUMINAMATH_GPT_avg_first_six_results_l1282_128296


namespace NUMINAMATH_GPT_die_roll_probability_div_3_l1282_128227

noncomputable def probability_divisible_by_3 : ℚ :=
  1 - ((2 : ℚ) / 3) ^ 8

theorem die_roll_probability_div_3 :
  probability_divisible_by_3 = 6305 / 6561 :=
by
  sorry

end NUMINAMATH_GPT_die_roll_probability_div_3_l1282_128227


namespace NUMINAMATH_GPT_number_of_cars_in_train_l1282_128258

theorem number_of_cars_in_train
  (constant_speed : Prop)
  (cars_in_12_seconds : ℕ)
  (time_to_clear : ℕ)
  (cars_per_second : ℕ → ℕ → ℚ)
  (total_time_seconds : ℕ) :
  cars_in_12_seconds = 8 →
  time_to_clear = 180 →
  cars_per_second cars_in_12_seconds 12 = 2 / 3 →
  total_time_seconds = 180 →
  cars_per_second cars_in_12_seconds 12 * total_time_seconds = 120 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cars_in_train_l1282_128258


namespace NUMINAMATH_GPT_increasing_interval_f_l1282_128263

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3)

theorem increasing_interval_f :
  (∀ x, x ∈ Set.Ioi 3 → f x ∈ Set.Ioi 3) := sorry

end NUMINAMATH_GPT_increasing_interval_f_l1282_128263


namespace NUMINAMATH_GPT_matrix_determinant_6_l1282_128205

theorem matrix_determinant_6 (x y z w : ℝ)
  (h : x * w - y * z = 3) :
  (x * (5 * z + 2 * w) - z * (5 * x + 2 * y)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_matrix_determinant_6_l1282_128205


namespace NUMINAMATH_GPT_problem_f_2005_value_l1282_128243

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_2005_value (h_even : ∀ x : ℝ, f (-x) = f x)
                            (h_periodic : ∀ x : ℝ, f (x + 8) = f x + f 4)
                            (h_initial : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x = 4 - x) :
  f 2005 = 0 :=
sorry

end NUMINAMATH_GPT_problem_f_2005_value_l1282_128243


namespace NUMINAMATH_GPT_triangle_perimeter_l1282_128253

-- Definitions and given conditions
def side_length_a (a : ℝ) : Prop := a = 6
def inradius (r : ℝ) : Prop := r = 2
def circumradius (R : ℝ) : Prop := R = 5

-- The final proof statement to be proven
theorem triangle_perimeter (a r R : ℝ) (b c P : ℝ) 
  (h1 : side_length_a a)
  (h2 : inradius r)
  (h3 : circumradius R)
  (h4 : P = 2 * ((a + b + c) / 2)) :
  P = 24 :=
sorry

end NUMINAMATH_GPT_triangle_perimeter_l1282_128253


namespace NUMINAMATH_GPT_inverse_function_property_l1282_128272

noncomputable def f (a x : ℝ) : ℝ := (x - a) * |x|

theorem inverse_function_property (a : ℝ) :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, f a (g x) = x) ↔ a = 0 :=
by sorry

end NUMINAMATH_GPT_inverse_function_property_l1282_128272


namespace NUMINAMATH_GPT_find_m_l1282_128229

-- Define the conditions
def ellipse_eq (x y : ℝ) (m : ℝ) : Prop := (x^2) / m + (y^2) / 4 = 1
def eccentricity (e : ℝ) : Prop := e = 2

-- Statement of the problem
theorem find_m (m : ℝ) (e : ℝ) (h1 : eccentricity e) (h2 : ∀ x y : ℝ, ellipse_eq x y m) :
  m = 3 ∨ m = 5 :=
sorry

end NUMINAMATH_GPT_find_m_l1282_128229


namespace NUMINAMATH_GPT_elementary_school_classes_count_l1282_128228

theorem elementary_school_classes_count (E : ℕ) (donate_per_class : ℕ) (middle_school_classes : ℕ) (total_balls : ℕ) :
  donate_per_class = 5 →
  middle_school_classes = 5 →
  total_balls = 90 →
  5 * 2 * E + 5 * 2 * middle_school_classes = total_balls →
  E = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_elementary_school_classes_count_l1282_128228


namespace NUMINAMATH_GPT_find_d_l1282_128254

variables {x y z k d : ℝ}
variables {a : ℝ} (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
variables (h_ap : x * (y - z) + y * (z - x) + z * (x - y) = 0)
variables (h_sum : x * (y - z) + (y * (z - x) + d) + (z * (x - y) + 2 * d) = k)

theorem find_d : d = k / 3 :=
sorry

end NUMINAMATH_GPT_find_d_l1282_128254


namespace NUMINAMATH_GPT_number_of_articles_l1282_128283

-- Conditions
variables (C S : ℚ)
-- Given that the cost price of 50 articles is equal to the selling price of some number of articles N.
variables (N : ℚ) (h1 : 50 * C = N * S)
-- Given that the gain is 11.11111111111111 percent.
variables (gain : ℚ := 1/9) (h2 : S = C * (1 + gain))

-- Prove that N = 45
theorem number_of_articles (C S : ℚ) (N : ℚ) (h1 : 50 * C = N * S)
    (gain : ℚ := 1/9) (h2 : S = C * (1 + gain)) : N = 45 :=
by
  sorry

end NUMINAMATH_GPT_number_of_articles_l1282_128283


namespace NUMINAMATH_GPT_solve_for_x_l1282_128216

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
    5 * y ^ 2 + 2 * y + 3 = 3 * (9 * x ^ 2 + y + 1) ↔ x = 0 ∨ x = 1 / 6 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1282_128216


namespace NUMINAMATH_GPT_range_of_x_l1282_128201

theorem range_of_x (total_students math_club chemistry_club : ℕ) (h_total : total_students = 45) 
(h_math : math_club = 28) (h_chemistry : chemistry_club = 21) (x : ℕ) :
  4 ≤ x ∧ x ≤ 21 ↔ (28 + 21 - x ≤ 45) :=
by sorry

end NUMINAMATH_GPT_range_of_x_l1282_128201


namespace NUMINAMATH_GPT_batsman_average_after_17th_innings_l1282_128241

theorem batsman_average_after_17th_innings :
  ∀ (A : ℕ), (80 + 16 * A) = 17 * (A + 2) → A + 2 = 48 := by
  intro A h
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_innings_l1282_128241
