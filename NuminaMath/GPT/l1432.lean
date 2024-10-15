import Mathlib

namespace NUMINAMATH_GPT_Jerry_needs_72_dollars_l1432_143213

def action_figures_current : ℕ := 7
def action_figures_total : ℕ := 16
def cost_per_figure : ℕ := 8
def money_needed : ℕ := 72

theorem Jerry_needs_72_dollars : 
  (action_figures_total - action_figures_current) * cost_per_figure = money_needed :=
by
  sorry

end NUMINAMATH_GPT_Jerry_needs_72_dollars_l1432_143213


namespace NUMINAMATH_GPT_proof_of_arithmetic_sequence_l1432_143273

theorem proof_of_arithmetic_sequence 
  (x y z : ℕ) 
  (h1 : x + y + z = 15) 
  (h2 : x < y) 
  (h3 : y < z)
  (h4 : (x + 1) * (z + 9) = (y + 3) ^ 2) : 
  (x, y, z) = (3, 5, 7) :=
sorry

end NUMINAMATH_GPT_proof_of_arithmetic_sequence_l1432_143273


namespace NUMINAMATH_GPT_car_catch_truck_l1432_143233

theorem car_catch_truck (truck_speed car_speed : ℕ) (time_head_start : ℕ) (t : ℕ)
  (h1 : truck_speed = 45) (h2 : car_speed = 60) (h3 : time_head_start = 1) :
  45 * t + 45 = 60 * t → t = 3 := by
  intro h
  sorry

end NUMINAMATH_GPT_car_catch_truck_l1432_143233


namespace NUMINAMATH_GPT_find_b_l1432_143220

noncomputable def circle1 (x y a : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 5 - a^2 = 0
noncomputable def circle2 (x y b : ℝ) : Prop := x^2 + y^2 - (2*b - 10)*x - 2*b*y + 2*b^2 - 10*b + 16 = 0
def is_intersection (x1 y1 x2 y2 : ℝ) : Prop := x1^2 + y1^2 = x2^2 + y2^2

theorem find_b (a x1 y1 x2 y2 : ℝ) (b : ℝ) :
  (circle1 x1 y1 a) ∧ (circle1 x2 y2 a) ∧ 
  (circle2 x1 y1 b) ∧ (circle2 x2 y2 b) ∧ 
  is_intersection x1 y1 x2 y2 →
  b = 5 / 3 :=
sorry

end NUMINAMATH_GPT_find_b_l1432_143220


namespace NUMINAMATH_GPT_geometric_progression_ratio_l1432_143255

theorem geometric_progression_ratio (q : ℝ) (h : |q| < 1 ∧ ∀a : ℝ, a = 4 * (a * q / (1 - q) - a * q)) :
  q = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_ratio_l1432_143255


namespace NUMINAMATH_GPT_matrix_mult_3I_l1432_143222

variable (N : Matrix (Fin 3) (Fin 3) ℝ)

theorem matrix_mult_3I (w : Fin 3 → ℝ):
  (∀ (w : Fin 3 → ℝ), N.mulVec w = 3 * w) ↔ (N = 3 • (1 : Matrix (Fin 3) (Fin 3) ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_matrix_mult_3I_l1432_143222


namespace NUMINAMATH_GPT_find_x_l1432_143276

def operation (x y : ℕ) : ℕ := 2 * x * y

theorem find_x : 
  (operation 4 5 = 40) ∧ (operation x 40 = 480) → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1432_143276


namespace NUMINAMATH_GPT_root_of_quadratic_gives_value_l1432_143261

theorem root_of_quadratic_gives_value (a : ℝ) (h : a^2 + 3 * a - 5 = 0) : a^2 + 3 * a + 2021 = 2026 :=
by {
  -- We will skip the proof here.
  sorry
}

end NUMINAMATH_GPT_root_of_quadratic_gives_value_l1432_143261


namespace NUMINAMATH_GPT_TV_height_l1432_143278

theorem TV_height (area : ℝ) (width : ℝ) (height : ℝ) (h1 : area = 21) (h2 : width = 3) : height = 7 :=
  by
  sorry

end NUMINAMATH_GPT_TV_height_l1432_143278


namespace NUMINAMATH_GPT_thomas_needs_more_money_l1432_143288

-- Define the conditions in Lean
def weeklyAllowance : ℕ := 50
def hourlyWage : ℕ := 9
def hoursPerWeek : ℕ := 30
def weeklyExpenses : ℕ := 35
def weeksInYear : ℕ := 52
def carCost : ℕ := 15000

-- Define the total earnings for the first year
def firstYearEarnings : ℕ :=
  weeklyAllowance * weeksInYear

-- Define the weekly earnings from the second year job
def secondYearWeeklyEarnings : ℕ :=
  hourlyWage * hoursPerWeek

-- Define the total earnings for the second year
def secondYearEarnings : ℕ :=
  secondYearWeeklyEarnings * weeksInYear

-- Define the total earnings over two years
def totalEarnings : ℕ :=
  firstYearEarnings + secondYearEarnings

-- Define the total expenses over two years
def totalExpenses : ℕ :=
  weeklyExpenses * (2 * weeksInYear)

-- Define the net savings after two years
def netSavings : ℕ :=
  totalEarnings - totalExpenses

-- Define the amount more needed for the car
def amountMoreNeeded : ℕ :=
  carCost - netSavings

-- The theorem to prove
theorem thomas_needs_more_money : amountMoreNeeded = 2000 := by
  sorry

end NUMINAMATH_GPT_thomas_needs_more_money_l1432_143288


namespace NUMINAMATH_GPT_solve_for_m_l1432_143266

-- Define the conditions for the lines being parallel
def condition_one (m : ℝ) : Prop :=
  ∃ x y : ℝ, x + m * y + 3 = 0

def condition_two (m : ℝ) : Prop :=
  ∃ x y : ℝ, (m - 1) * x + 2 * m * y + 2 * m = 0

def are_parallel (A B C D : ℝ) : Prop :=
  A * D = B * C

theorem solve_for_m :
  ∀ (m : ℝ),
    (condition_one m) → 
    (condition_two m) → 
    (are_parallel 1 m 3 (2 * m)) →
    (m = 0) :=
by
  intro m h1 h2 h_parallel
  sorry

end NUMINAMATH_GPT_solve_for_m_l1432_143266


namespace NUMINAMATH_GPT_gain_percent_l1432_143250

theorem gain_percent (CP SP : ℝ) (hCP : CP = 110) (hSP : SP = 125) : 
  (SP - CP) / CP * 100 = 13.64 := by
  sorry

end NUMINAMATH_GPT_gain_percent_l1432_143250


namespace NUMINAMATH_GPT_range_of_k_l1432_143246

def P (x k : ℝ) : Prop := x^2 + k*x + 1 > 0
def Q (x k : ℝ) : Prop := k*x^2 + x + 2 < 0

theorem range_of_k (k : ℝ) : (¬ (P 2 k ∧ Q 2 k)) ↔ k ∈ (Set.Iic (-5/2) ∪ Set.Ici (-1)) := 
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1432_143246


namespace NUMINAMATH_GPT_fraction_replaced_l1432_143205

theorem fraction_replaced (x : ℝ) (h₁ : 0.15 * (1 - x) + 0.19000000000000007 * x = 0.16) : x = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_fraction_replaced_l1432_143205


namespace NUMINAMATH_GPT_integer_solutions_l1432_143298

theorem integer_solutions (x y : ℤ) : 
  x^2 * y = 10000 * x + y ↔ 
  (x, y) = (-9, -1125) ∨ 
  (x, y) = (-3, -3750) ∨ 
  (x, y) = (0, 0) ∨ 
  (x, y) = (3, 3750) ∨ 
  (x, y) = (9, 1125) := 
by
  sorry

end NUMINAMATH_GPT_integer_solutions_l1432_143298


namespace NUMINAMATH_GPT_find_f_f_neg1_l1432_143214

def f (x : Int) : Int :=
  if x >= 0 then x + 2 else 1

theorem find_f_f_neg1 : f (f (-1)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_f_f_neg1_l1432_143214


namespace NUMINAMATH_GPT_Holly_throws_5_times_l1432_143228

def Bess.throw_distance := 20
def Bess.throw_times := 4
def Holly.throw_distance := 8
def total_distance := 200

theorem Holly_throws_5_times : 
  (total_distance - Bess.throw_times * 2 * Bess.throw_distance) / Holly.throw_distance = 5 :=
by 
  sorry

end NUMINAMATH_GPT_Holly_throws_5_times_l1432_143228


namespace NUMINAMATH_GPT_flat_fee_is_65_l1432_143269

-- Define the problem constants
def George_nights : ℕ := 3
def Noah_nights : ℕ := 6
def George_cost : ℤ := 155
def Noah_cost : ℤ := 290

-- Prove that the flat fee for the first night is 65, given the costs and number of nights stayed.
theorem flat_fee_is_65 
  (f n : ℤ)
  (h1 : f + (George_nights - 1) * n = George_cost)
  (h2 : f + (Noah_nights - 1) * n = Noah_cost) :
  f = 65 := 
sorry

end NUMINAMATH_GPT_flat_fee_is_65_l1432_143269


namespace NUMINAMATH_GPT_find_B_l1432_143225

variable {A B C a b c : Real}

noncomputable def B_value (A B C a b c : Real) : Prop :=
  B = 2 * Real.pi / 3

theorem find_B 
  (h_triangle: a^2 + b^2 + c^2 = 2*a*b*Real.cos C)
  (h_cos_eq: (2 * a + c) * Real.cos B + b * Real.cos C = 0) : 
  B_value A B C a b c :=
by
  sorry

end NUMINAMATH_GPT_find_B_l1432_143225


namespace NUMINAMATH_GPT_ratio_of_members_l1432_143286

theorem ratio_of_members (f m c : ℕ) 
  (h1 : (35 * f + 30 * m + 10 * c) / (f + m + c) = 25) :
  2 * f + m = 3 * c :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_members_l1432_143286


namespace NUMINAMATH_GPT_malcolm_joshua_time_difference_l1432_143238

-- Define the constants
def malcolm_speed : ℕ := 5 -- minutes per mile
def joshua_speed : ℕ := 8 -- minutes per mile
def race_distance : ℕ := 12 -- miles

-- Define the times it takes each runner to finish
def malcolm_time : ℕ := malcolm_speed * race_distance
def joshua_time : ℕ := joshua_speed * race_distance

-- Define the time difference and the proof statement
def time_difference : ℕ := joshua_time - malcolm_time

theorem malcolm_joshua_time_difference : time_difference = 36 := by
  sorry

end NUMINAMATH_GPT_malcolm_joshua_time_difference_l1432_143238


namespace NUMINAMATH_GPT_geometric_sequence_a4_a5_l1432_143282

open BigOperators

theorem geometric_sequence_a4_a5 (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 + a 2 = 1)
  (h3 : a 3 + a 4 = 9) : 
  a 4 + a 5 = 27 ∨ a 4 + a 5 = -27 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a4_a5_l1432_143282


namespace NUMINAMATH_GPT_sum_of_distinct_roots_l1432_143271

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_distinct_roots_l1432_143271


namespace NUMINAMATH_GPT_wholesale_price_l1432_143207

theorem wholesale_price (R : ℝ) (W : ℝ)
  (hR : R = 120)
  (h_discount : ∀ SP : ℝ, SP = R - (0.10 * R))
  (h_profit : ∀ P : ℝ, P = 0.20 * W)
  (h_SP_eq_W_P : ∀ SP P : ℝ, SP = W + P) :
  W = 90 := by
  sorry

end NUMINAMATH_GPT_wholesale_price_l1432_143207


namespace NUMINAMATH_GPT_angelina_speed_l1432_143247

theorem angelina_speed (v : ℝ) (h1 : 200 / v - 50 = 300 / (2 * v)) : 2 * v = 2 := 
by
  sorry

end NUMINAMATH_GPT_angelina_speed_l1432_143247


namespace NUMINAMATH_GPT_train_stops_one_minute_per_hour_l1432_143211

theorem train_stops_one_minute_per_hour (D : ℝ) (h1 : D / 400 = T₁) (h2 : D / 360 = T₂) : 
  (T₂ - T₁) * 60 = 1 :=
by
  sorry

end NUMINAMATH_GPT_train_stops_one_minute_per_hour_l1432_143211


namespace NUMINAMATH_GPT_arithmetic_mean_probability_l1432_143203

theorem arithmetic_mean_probability
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : b = (a + c) / 2) :
  b = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_probability_l1432_143203


namespace NUMINAMATH_GPT_find_general_formula_l1432_143232

section sequence

variables {R : Type*} [LinearOrderedField R]
variable (c : R)
variable (h_c : c ≠ 0)

def seq (a : Nat → R) : Prop :=
  a 1 = 1 ∧ ∀ n : Nat, n > 0 → a (n + 1) = c * a n + c^(n + 1) * (2 * n + 1)

def general_formula (a : Nat → R) : Prop :=
  ∀ n : Nat, n > 0 → a n = (n^2 - 1) * c^n + c^(n - 1)

theorem find_general_formula :
  ∃ a : Nat → R, seq c a ∧ general_formula c a :=
by
  sorry

end sequence

end NUMINAMATH_GPT_find_general_formula_l1432_143232


namespace NUMINAMATH_GPT_value_of_x_l1432_143235

theorem value_of_x (x : ℕ) : (8^4 + 8^4 + 8^4 = 2^x) → x = 13 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1432_143235


namespace NUMINAMATH_GPT_original_number_of_boys_l1432_143201

theorem original_number_of_boys (n : ℕ) (W : ℕ) 
  (h1 : W = n * 35) 
  (h2 : W + 135 = (n + 3) * 36) : 
  n = 27 := 
by 
  sorry

end NUMINAMATH_GPT_original_number_of_boys_l1432_143201


namespace NUMINAMATH_GPT_f_sqrt_2_l1432_143218

noncomputable def f : ℝ → ℝ :=
sorry

axiom domain_f : ∀ x, 0 < x → 0 < f x
axiom add_property : ∀ x y, f (x * y) = f x + f y
axiom f_at_8 : f 8 = 6

theorem f_sqrt_2 : f (Real.sqrt 2) = 1 :=
by
  have sqrt2pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  sorry

end NUMINAMATH_GPT_f_sqrt_2_l1432_143218


namespace NUMINAMATH_GPT_find_x_l1432_143234

theorem find_x (n : ℕ) (h1 : x = 8^n - 1) (h2 : Nat.Prime 31) 
  (h3 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 = 31 ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ x → (p = p1 ∨ p = p2 ∨ p = p3))) : 
  x = 32767 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1432_143234


namespace NUMINAMATH_GPT_distinct_shell_arrangements_l1432_143229

/--
John draws a regular five pointed star and places one of ten different sea shells at each of the 5 outward-pointing points and 5 inward-pointing points. 
Considering rotations and reflections of an arrangement as equivalent, prove that the number of ways he can place the shells is 362880.
-/
theorem distinct_shell_arrangements : 
  let total_arrangements := Nat.factorial 10
  let symmetries := 10
  total_arrangements / symmetries = 362880 :=
by
  sorry

end NUMINAMATH_GPT_distinct_shell_arrangements_l1432_143229


namespace NUMINAMATH_GPT_age_difference_l1432_143212

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 16) : A - C = 16 :=
sorry

end NUMINAMATH_GPT_age_difference_l1432_143212


namespace NUMINAMATH_GPT_points_on_parabola_l1432_143245

theorem points_on_parabola (a : ℝ) (y1 y2 y3 : ℝ) 
  (h_a : a < -1) 
  (h1 : y1 = (a - 1)^2) 
  (h2 : y2 = a^2) 
  (h3 : y3 = (a + 1)^2) : 
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end NUMINAMATH_GPT_points_on_parabola_l1432_143245


namespace NUMINAMATH_GPT_remainder_98_mul_102_div_11_l1432_143295

theorem remainder_98_mul_102_div_11 : (98 * 102) % 11 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_98_mul_102_div_11_l1432_143295


namespace NUMINAMATH_GPT_triangle_altitude_l1432_143248

theorem triangle_altitude
  (base : ℝ) (height : ℝ) (side : ℝ)
  (h_base : base = 6)
  (h_side : side = 6)
  (area_triangle : ℝ) (area_square : ℝ)
  (h_area_square : area_square = side ^ 2)
  (h_area_equal : area_triangle = area_square)
  (h_area_triangle : area_triangle = (base * height) / 2) :
  height = 12 := 
by
  sorry

end NUMINAMATH_GPT_triangle_altitude_l1432_143248


namespace NUMINAMATH_GPT_problem_l1432_143202

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≥ f y
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def max_value_in (f : ℝ → ℝ) (a b : ℝ) (v : ℝ) : Prop := ∀ x, a ≤ x → x ≤ b → f x ≤ v ∧ (∃ z, a ≤ z ∧ z ≤ b ∧ f z = v)

theorem problem
  (h_even : even_function f)
  (h_decreasing : decreasing_on f (-5) (-2))
  (h_max : max_value_in f (-5) (-2) 7) :
  increasing_on f 2 5 ∧ max_value_in f 2 5 7 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1432_143202


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l1432_143275

theorem roots_of_quadratic_eq (h : ∀ x : ℝ, x^2 - 3 * x = 0 → x = 0 ∨ x = 3) :
  ∀ x : ℝ, x^2 - 3 * x = 0 → x = 0 ∨ x = 3 :=
by sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l1432_143275


namespace NUMINAMATH_GPT_binomial_coeff_and_coeff_of_x8_l1432_143277

theorem binomial_coeff_and_coeff_of_x8 (x : ℂ) :
  let expr := (x^2 + 4*x + 4)^5
  let expansion := (x + 2)^10
  ∃ (binom_coeff_x8 coeff_x8 : ℤ),
    binom_coeff_x8 = 45 ∧ coeff_x8 = 180 :=
by
  sorry

end NUMINAMATH_GPT_binomial_coeff_and_coeff_of_x8_l1432_143277


namespace NUMINAMATH_GPT_v2_correct_at_2_l1432_143291

def poly (x : ℕ) : ℕ := x^5 + x^4 + 2 * x^3 + 3 * x^2 + 4 * x + 1

def horner_v2 (x : ℕ) : ℕ :=
  let v0 := 1
  let v1 := v0 * x + 4
  let v2 := v1 * x + 3
  v2

theorem v2_correct_at_2 : horner_v2 2 = 15 := by
  sorry

end NUMINAMATH_GPT_v2_correct_at_2_l1432_143291


namespace NUMINAMATH_GPT_line_eq_l1432_143242

theorem line_eq (x y : ℝ) (point eq_direction_vector) (h₀ : point = (3, -2))
    (h₁ : eq_direction_vector = (-5, 3)) :
    3 * x + 5 * y + 1 = 0 := by sorry

end NUMINAMATH_GPT_line_eq_l1432_143242


namespace NUMINAMATH_GPT_solve_number_puzzle_l1432_143251

def number_puzzle (N : ℕ) : Prop :=
  (1/4) * (1/3) * (2/5) * N = 14 → (40/100) * N = 168

theorem solve_number_puzzle : ∃ N, number_puzzle N := by
  sorry

end NUMINAMATH_GPT_solve_number_puzzle_l1432_143251


namespace NUMINAMATH_GPT_quadratic_polynomial_divisible_by_3_l1432_143265

theorem quadratic_polynomial_divisible_by_3
  (a b c : ℤ)
  (h : ∀ x : ℤ, 3 ∣ (a * x^2 + b * x + c)) :
  3 ∣ a ∧ 3 ∣ b ∧ 3 ∣ c :=
sorry

end NUMINAMATH_GPT_quadratic_polynomial_divisible_by_3_l1432_143265


namespace NUMINAMATH_GPT_min_value_of_c_l1432_143221

noncomputable def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

theorem min_value_of_c (c : ℕ) (n m : ℕ) (h1 : 5 * c = n^3) (h2 : 3 * c = m^2) : c = 675 := by
  sorry

end NUMINAMATH_GPT_min_value_of_c_l1432_143221


namespace NUMINAMATH_GPT_roots_abs_lt_one_l1432_143236

theorem roots_abs_lt_one
  (a b : ℝ)
  (h1 : |a| + |b| < 1)
  (h2 : a^2 - 4 * b ≥ 0) :
  ∀ (x : ℝ), x^2 + a * x + b = 0 → |x| < 1 :=
sorry

end NUMINAMATH_GPT_roots_abs_lt_one_l1432_143236


namespace NUMINAMATH_GPT_midpoint_condition_l1432_143240

theorem midpoint_condition (c : ℝ) :
  (∃ A B : ℝ × ℝ,
    A ∈ { p : ℝ × ℝ | p.2 = p.1^2 - 2 * p.1 - 3 } ∧
    B ∈ { p : ℝ × ℝ | p.2 = p.1^2 - 2 * p.1 - 3 } ∧
    A ∈ { p : ℝ × ℝ | p.2 = -p.1^2 + 4 * p.1 + c } ∧
    B ∈ { p : ℝ × ℝ | p.2 = -p.1^2 + 4 * p.1 + c } ∧
    ((A.1 + B.1) / 2 + (A.2 + B.2) / 2) = 2017
  ) ↔
  c = 4031 := sorry

end NUMINAMATH_GPT_midpoint_condition_l1432_143240


namespace NUMINAMATH_GPT_trig_inequality_l1432_143231

theorem trig_inequality (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.cos β)^2 * (Real.sin β)^2) ≥ 9) := by
  sorry

end NUMINAMATH_GPT_trig_inequality_l1432_143231


namespace NUMINAMATH_GPT_intersection_A_B_l1432_143219

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

def setA : Set ℝ := { x | Real.log x > 0 }
def setB : Set ℝ := { x | Real.exp x * Real.exp x < 3 }

theorem intersection_A_B : setA ∩ setB = { x | 1 < x ∧ x < log2 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1432_143219


namespace NUMINAMATH_GPT_sugar_flour_ratio_10_l1432_143259

noncomputable def sugar_to_flour_ratio (sugar flour : ℕ) : ℕ :=
  sugar / flour

theorem sugar_flour_ratio_10 (sugar flour : ℕ) (hs : sugar = 50) (hf : flour = 5) : sugar_to_flour_ratio sugar flour = 10 :=
by
  rw [hs, hf]
  unfold sugar_to_flour_ratio
  norm_num
  -- sorry

end NUMINAMATH_GPT_sugar_flour_ratio_10_l1432_143259


namespace NUMINAMATH_GPT_red_card_value_l1432_143279

theorem red_card_value (credits : ℕ) (total_cards : ℕ) (blue_card_value : ℕ) (red_cards : ℕ) (blue_cards : ℕ) 
    (condition1 : blue_card_value = 5)
    (condition2 : total_cards = 20)
    (condition3 : credits = 84)
    (condition4 : red_cards = 8)
    (condition5 : blue_cards = total_cards - red_cards) :
  (credits - blue_cards * blue_card_value) / red_cards = 3 :=
by
  sorry

end NUMINAMATH_GPT_red_card_value_l1432_143279


namespace NUMINAMATH_GPT_arithmetic_seq_a1_l1432_143280

theorem arithmetic_seq_a1 (a_1 d : ℝ) (h1 : a_1 + 4 * d = 9) (h2 : 2 * (a_1 + 2 * d) = (a_1 + d) + 6) : a_1 = -3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a1_l1432_143280


namespace NUMINAMATH_GPT_tire_circumference_l1432_143285

theorem tire_circumference 
  (rev_per_min : ℝ) -- revolutions per minute
  (car_speed_kmh : ℝ) -- car speed in km/h
  (conversion_factor : ℝ) -- conversion factor for speed from km/h to m/min
  (min_to_meter : ℝ) -- multiplier to convert minutes to meters
  (C : ℝ) -- circumference of the tire in meters
  : rev_per_min = 400 ∧ car_speed_kmh = 120 ∧ conversion_factor = 1000 / 60 ∧ min_to_meter = 1000 / 60 ∧ (C * rev_per_min = car_speed_kmh * min_to_meter) → C = 5 :=
by
  sorry

end NUMINAMATH_GPT_tire_circumference_l1432_143285


namespace NUMINAMATH_GPT_nails_needed_for_house_wall_l1432_143254

theorem nails_needed_for_house_wall
    (large_planks : ℕ)
    (small_planks : ℕ)
    (nails_for_large_planks : ℕ)
    (nails_for_small_planks : ℕ)
    (H1 : large_planks = 12)
    (H2 : small_planks = 10)
    (H3 : nails_for_large_planks = 15)
    (H4 : nails_for_small_planks = 5) :
    (nails_for_large_planks + nails_for_small_planks) = 20 := by
  sorry

end NUMINAMATH_GPT_nails_needed_for_house_wall_l1432_143254


namespace NUMINAMATH_GPT_largest_multiple_of_7_negation_greater_than_neg_150_l1432_143226

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end NUMINAMATH_GPT_largest_multiple_of_7_negation_greater_than_neg_150_l1432_143226


namespace NUMINAMATH_GPT_circle_ring_ratio_l1432_143293

theorem circle_ring_ratio
  (r R c d : ℝ)
  (hr : 0 < r)
  (hR : 0 < R)
  (hc : 0 < c)
  (hd : 0 < d)
  (h_areas : π * R^2 = (c / d) * (π * R^2 - π * r^2)) :
  R / r = (Real.sqrt c) / (Real.sqrt (d - c)) := 
by 
  sorry

end NUMINAMATH_GPT_circle_ring_ratio_l1432_143293


namespace NUMINAMATH_GPT_find_number_of_girls_l1432_143253

variable (B G : ℕ)

theorem find_number_of_girls
  (h1 : B = G / 2)
  (h2 : B + G = 90)
  : G = 60 :=
sorry

end NUMINAMATH_GPT_find_number_of_girls_l1432_143253


namespace NUMINAMATH_GPT_possible_initial_triangles_l1432_143274

-- Define the triangle types by their angles in degrees
inductive TriangleType
| T45T45T90
| T30T60T90
| T30T30T120
| T60T60T60

-- Define a Lean statement to express the problem
theorem possible_initial_triangles (T : TriangleType) :
  T = TriangleType.T45T45T90 ∨
  T = TriangleType.T30T60T90 ∨
  T = TriangleType.T30T30T120 ∨
  T = TriangleType.T60T60T60 :=
sorry

end NUMINAMATH_GPT_possible_initial_triangles_l1432_143274


namespace NUMINAMATH_GPT_largest_non_representable_integer_l1432_143299

theorem largest_non_representable_integer (n a b : ℕ) (h₁ : n = 42 * a + b)
  (h₂ : 0 ≤ b) (h₃ : b < 42) (h₄ : ¬ (b % 6 = 0)) :
  n ≤ 252 :=
sorry

end NUMINAMATH_GPT_largest_non_representable_integer_l1432_143299


namespace NUMINAMATH_GPT_mary_needs_10_charges_to_vacuum_house_l1432_143217

theorem mary_needs_10_charges_to_vacuum_house :
  (let bedroom_time := 10
   let kitchen_time := 12
   let living_room_time := 8
   let dining_room_time := 6
   let office_time := 9
   let bathroom_time := 5
   let battery_duration := 8
   3 * bedroom_time + kitchen_time + living_room_time + dining_room_time + office_time + 2 * bathroom_time) / battery_duration = 10 :=
by sorry

end NUMINAMATH_GPT_mary_needs_10_charges_to_vacuum_house_l1432_143217


namespace NUMINAMATH_GPT_quadratic_equation_reciprocal_integer_roots_l1432_143223

noncomputable def quadratic_equation_conditions (a b c : ℝ) : Prop :=
  (∃ r : ℝ, (r * (1/r) = 1) ∧ (r + (1/r) = 4)) ∧ 
  (c = a) ∧ 
  (b = -4 * a)

theorem quadratic_equation_reciprocal_integer_roots (a b c : ℝ) (h1 : quadratic_equation_conditions a b c) : 
  c = a ∧ b = -4 * a :=
by
  obtain ⟨r, hr₁, hr₂⟩ := h1.1
  sorry

end NUMINAMATH_GPT_quadratic_equation_reciprocal_integer_roots_l1432_143223


namespace NUMINAMATH_GPT_complex_division_l1432_143292

-- Define the complex numbers in Lean
def i : ℂ := Complex.I

-- Claim to be proved
theorem complex_division :
  (1 + i) / (3 - i) = (1 + 2 * i) / 5 :=
by
  sorry

end NUMINAMATH_GPT_complex_division_l1432_143292


namespace NUMINAMATH_GPT_max_third_side_l1432_143239

open Real

variables {A B C : ℝ} {a b c : ℝ} 

theorem max_third_side (h : cos (4 * A) + cos (4 * B) + cos (4 * C) = 1) 
                       (ha : a = 8) (hb : b = 15) : c = 17 :=
 by
  sorry 

end NUMINAMATH_GPT_max_third_side_l1432_143239


namespace NUMINAMATH_GPT_product_approximation_l1432_143256

theorem product_approximation :
  (3.05 * 7.95 * (6.05 + 3.95)) = 240 := by
  sorry

end NUMINAMATH_GPT_product_approximation_l1432_143256


namespace NUMINAMATH_GPT_solve_for_r_l1432_143264

variable (n : ℝ) (r : ℝ)

theorem solve_for_r (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) : r = (n * (1 + Real.sqrt 3)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_r_l1432_143264


namespace NUMINAMATH_GPT_circle_intersects_cells_l1432_143272

/-- On a grid with 1 cm x 1 cm cells, a circle with a radius of 100 cm is drawn.
    The circle does not pass through any vertices of the cells and does not touch the sides of the cells.
    Prove that the number of cells the circle can intersect is either 800 or 799. -/
theorem circle_intersects_cells (r : ℝ) (gsize : ℝ) (cells : ℕ) :
  r = 100 ∧ gsize = 1 ∧ cells = 800 ∨ cells = 799 :=
by
  sorry

end NUMINAMATH_GPT_circle_intersects_cells_l1432_143272


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l1432_143287

-- Conditions
def condition_1 (x : ℝ) : Prop := x > 3
def condition_2 (x : ℝ) : Prop := x^2 - 5 * x + 6 > 0

-- Theorem statement
theorem sufficient_but_not_necessary (x : ℝ) : condition_1 x → condition_2 x :=
sorry

theorem not_necessary (x : ℝ) : condition_2 x → ∃ y : ℝ, ¬ condition_1 y ∧ condition_2 y :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l1432_143287


namespace NUMINAMATH_GPT_range_of_k_decreasing_l1432_143200

theorem range_of_k_decreasing (k b : ℝ) (h : ∀ x₁ x₂, x₁ < x₂ → (k^2 - 3*k + 2) * x₁ + b > (k^2 - 3*k + 2) * x₂ + b) : 1 < k ∧ k < 2 :=
by
  -- Proof 
  sorry

end NUMINAMATH_GPT_range_of_k_decreasing_l1432_143200


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1432_143244

theorem sum_of_two_numbers (x : ℤ) (sum certain value : ℤ) (h₁ : 25 - x = 5) : 25 + x = 45 := by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1432_143244


namespace NUMINAMATH_GPT_find_b10_l1432_143297

def seq (b : ℕ → ℕ) :=
  (b 1 = 2)
  ∧ (∀ m n, b (m + n) = b m + b n + 2 * m * n)

theorem find_b10 (b : ℕ → ℕ) (h : seq b) : b 10 = 110 :=
by 
  -- Proof omitted, as requested.
  sorry

end NUMINAMATH_GPT_find_b10_l1432_143297


namespace NUMINAMATH_GPT_value_of_5a_l1432_143281

variable (a : ℕ)

theorem value_of_5a (h : 5 * (a - 3) = 25) : 5 * a = 40 :=
sorry

end NUMINAMATH_GPT_value_of_5a_l1432_143281


namespace NUMINAMATH_GPT_m_value_for_positive_root_eq_l1432_143210

-- We start by defining the problem:
-- Given the condition that the equation (3x - 1)/(x + 1) - m/(x + 1) = 1 has a positive root,
-- we need to prove that m = -4.

theorem m_value_for_positive_root_eq (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 * x - 1) / (x + 1) - m / (x + 1) = 1) → m = -4 :=
by
  sorry

end NUMINAMATH_GPT_m_value_for_positive_root_eq_l1432_143210


namespace NUMINAMATH_GPT_line_intersect_yaxis_at_l1432_143208

theorem line_intersect_yaxis_at
  (x1 y1 x2 y2 : ℝ) : (x1 = 3) → (y1 = 19) → (x2 = -7) → (y2 = -1) →
  ∃ y : ℝ, (0, y) = (0, 13) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_line_intersect_yaxis_at_l1432_143208


namespace NUMINAMATH_GPT_son_father_age_sum_l1432_143257

theorem son_father_age_sum
    (S F : ℕ)
    (h1 : F - 6 = 3 * (S - 6))
    (h2 : F = 2 * S) :
    S + F = 36 :=
sorry

end NUMINAMATH_GPT_son_father_age_sum_l1432_143257


namespace NUMINAMATH_GPT_combinedAgeIn5Years_l1432_143268

variable (Amy Mark Emily : ℕ)

-- Conditions
def amyAge : ℕ := 15
def markAge : ℕ := amyAge + 7
def emilyAge : ℕ := 2 * amyAge

-- Proposition to be proved
theorem combinedAgeIn5Years :
  Amy = amyAge →
  Mark = markAge →
  Emily = emilyAge →
  (Amy + 5) + (Mark + 5) + (Emily + 5) = 82 :=
by
  intros hAmy hMark hEmily
  sorry

end NUMINAMATH_GPT_combinedAgeIn5Years_l1432_143268


namespace NUMINAMATH_GPT_find_x_l1432_143230

variable (x : ℤ)
def A : Set ℤ := {x^2, x + 1, -3}
def B : Set ℤ := {x - 5, 2 * x - 1, x^2 + 1}

theorem find_x (h : A x ∩ B x = {-3}) : x = -1 :=
sorry

end NUMINAMATH_GPT_find_x_l1432_143230


namespace NUMINAMATH_GPT_area_between_chords_is_correct_l1432_143241

noncomputable def circle_radius : ℝ := 10
noncomputable def chord_distance_apart : ℝ := 12
noncomputable def area_between_chords : ℝ := 44.73

theorem area_between_chords_is_correct 
    (r : ℝ) (d : ℝ) (A : ℝ) 
    (hr : r = circle_radius) 
    (hd : d = chord_distance_apart) 
    (hA : A = area_between_chords) : 
    ∃ area : ℝ, area = A := by 
  sorry

end NUMINAMATH_GPT_area_between_chords_is_correct_l1432_143241


namespace NUMINAMATH_GPT_no_right_obtuse_triangle_l1432_143227

theorem no_right_obtuse_triangle :
  ∀ (α β γ : ℝ),
  (α + β + γ = 180) →
  (α = 90 ∨ β = 90 ∨ γ = 90) →
  (α > 90 ∨ β > 90 ∨ γ > 90) →
  false :=
by
  sorry

end NUMINAMATH_GPT_no_right_obtuse_triangle_l1432_143227


namespace NUMINAMATH_GPT_triple_square_side_area_l1432_143283

theorem triple_square_side_area (s : ℝ) : (3 * s) ^ 2 ≠ 3 * (s ^ 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_triple_square_side_area_l1432_143283


namespace NUMINAMATH_GPT_fraction_to_decimal_l1432_143294

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1432_143294


namespace NUMINAMATH_GPT_local_value_of_7_in_diff_l1432_143237

-- Definitions based on conditions
def local_value (n : ℕ) (d : ℕ) : ℕ :=
  if h : d < 10 ∧ (n / Nat.pow 10 (Nat.log 10 n - Nat.log 10 d)) % 10 = d then
    d * Nat.pow 10 (Nat.log 10 n - Nat.log 10 d)
  else
    0

def diff (a b : ℕ) : ℕ := a - b

-- Question translated to Lean 4 statement
theorem local_value_of_7_in_diff :
  local_value (diff 100889 (local_value 28943712 3)) 7 = 70000 :=
by sorry

end NUMINAMATH_GPT_local_value_of_7_in_diff_l1432_143237


namespace NUMINAMATH_GPT_min_value_fraction_l1432_143296

theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : 
  ∀ x, (x = (1 / m + 8 / n)) → x ≥ 18 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l1432_143296


namespace NUMINAMATH_GPT_poly_a_roots_poly_b_roots_l1432_143252

-- Define the polynomials
def poly_a (x : ℤ) : ℤ := 2 * x ^ 3 - 3 * x ^ 2 - 11 * x + 6
def poly_b (x : ℤ) : ℤ := x ^ 4 + 4 * x ^ 3 - 9 * x ^ 2 - 16 * x + 20

-- Assert the integer roots for poly_a
theorem poly_a_roots : {x : ℤ | poly_a x = 0} = {-2, 3} := sorry

-- Assert the integer roots for poly_b
theorem poly_b_roots : {x : ℤ | poly_b x = 0} = {1, 2, -2, -5} := sorry

end NUMINAMATH_GPT_poly_a_roots_poly_b_roots_l1432_143252


namespace NUMINAMATH_GPT_minimum_people_l1432_143258

def num_photos : ℕ := 10
def num_center_men : ℕ := 10
def num_people_per_photo : ℕ := 3

theorem minimum_people (n : ℕ) (h : n = num_photos) :
  (∃ total_people, total_people = 16) :=
sorry

end NUMINAMATH_GPT_minimum_people_l1432_143258


namespace NUMINAMATH_GPT_trigonometric_expression_l1432_143224

variable (α : Real)
open Real

theorem trigonometric_expression (h : tan α = 3) : 
  (2 * sin α - cos α) / (sin α + 3 * cos α) = 5 / 6 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_l1432_143224


namespace NUMINAMATH_GPT_exists_positive_real_u_l1432_143267

theorem exists_positive_real_u (n : ℕ) (h_pos : n > 0) : 
  ∃ u : ℝ, u > 0 ∧ ∀ n : ℕ, n > 0 → (⌊u^n⌋ - n) % 2 = 0 :=
sorry

end NUMINAMATH_GPT_exists_positive_real_u_l1432_143267


namespace NUMINAMATH_GPT_monotonicity_f_a_eq_1_domain_condition_inequality_condition_l1432_143216

noncomputable def f (x a : ℝ) := (Real.log (x^2 - 2 * x + a)) / (x - 1)

theorem monotonicity_f_a_eq_1 :
  ∀ x : ℝ, 1 < x → 
  (f x 1 < f (e + 1) 1 → 
   ∀ y, 1 < y ∧ y < e + 1 → f y 1 < f (e + 1) 1) ∧ 
  (f (e + 1) 1 < f x 1 → 
   ∀ z, e + 1 < z → f (e + 1) 1 < f z 1) :=
sorry

theorem domain_condition (a : ℝ) :
  (∀ x : ℝ, (x < 1 ∨ x > 1) → x^2 - 2 * x + a > 0) ↔ a ≥ 1 :=
sorry

theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f x a < (x - 1) * Real.exp x)) ↔ (1 + 1 / Real.exp 1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_monotonicity_f_a_eq_1_domain_condition_inequality_condition_l1432_143216


namespace NUMINAMATH_GPT_final_game_deficit_l1432_143249

-- Define the points for each scoring action
def free_throw_points := 1
def three_pointer_points := 3
def jump_shot_points := 2
def layup_points := 2
def and_one_points := layup_points + free_throw_points

-- Define the points scored by Liz
def liz_free_throws := 5 * free_throw_points
def liz_three_pointers := 4 * three_pointer_points
def liz_jump_shots := 5 * jump_shot_points
def liz_and_one := and_one_points

def liz_points := liz_free_throws + liz_three_pointers + liz_jump_shots + liz_and_one

-- Define the points scored by Taylor
def taylor_three_pointers := 2 * three_pointer_points
def taylor_jump_shots := 3 * jump_shot_points

def taylor_points := taylor_three_pointers + taylor_jump_shots

-- Define the points for Liz's team
def team_points := liz_points + taylor_points

-- Define the points scored by the opposing team players
def opponent_player1_points := 4 * three_pointer_points

def opponent_player2_jump_shots := 4 * jump_shot_points
def opponent_player2_free_throws := 2 * free_throw_points
def opponent_player2_points := opponent_player2_jump_shots + opponent_player2_free_throws

def opponent_player3_jump_shots := 2 * jump_shot_points
def opponent_player3_three_pointer := 1 * three_pointer_points
def opponent_player3_points := opponent_player3_jump_shots + opponent_player3_three_pointer

-- Define the points for the opposing team
def opponent_team_points := opponent_player1_points + opponent_player2_points + opponent_player3_points

-- Initial deficit
def initial_deficit := 25

-- Final net scoring in the final quarter
def net_quarter_scoring := team_points - opponent_team_points

-- Final deficit
def final_deficit := initial_deficit - net_quarter_scoring

theorem final_game_deficit : final_deficit = 12 := by
  sorry

end NUMINAMATH_GPT_final_game_deficit_l1432_143249


namespace NUMINAMATH_GPT_angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l1432_143289

-- Define the concept of a cube and the diagonals of its faces.
structure Cube :=
  (faces : Fin 6 → (Fin 4 → ℝ × ℝ × ℝ))    -- Representing each face as a set of four vertices in 3D space

def is_square_face (face : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  -- A function that checks if a given set of four vertices forms a square face.
  sorry

def are_adjacent_faces_perpendicular_diagonals 
  (face1 face2 : Fin 4 → ℝ × ℝ × ℝ) (c : Cube) : Prop :=
  -- A function that checks if the diagonals of two given adjacent square faces of a cube are perpendicular.
  sorry

-- The theorem stating the required proof:
theorem angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees
  (c : Cube)
  (h1 : is_square_face (c.faces 0))
  (h2 : is_square_face (c.faces 1))
  (h_adj: are_adjacent_faces_perpendicular_diagonals (c.faces 0) (c.faces 1) c) :
  ∃ q : ℝ, q = 90 :=
by
  sorry

end NUMINAMATH_GPT_angle_between_diagonals_of_adjacent_faces_of_cube_is_90_degrees_l1432_143289


namespace NUMINAMATH_GPT_minvalue_expression_l1432_143204

theorem minvalue_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
    9 * z / (3 * x + y) + 9 * x / (y + 3 * z) + 4 * y / (x + z) ≥ 3 := 
by
  sorry

end NUMINAMATH_GPT_minvalue_expression_l1432_143204


namespace NUMINAMATH_GPT_smallest_angle_in_convex_20_gon_seq_l1432_143215

theorem smallest_angle_in_convex_20_gon_seq :
  ∃ (α : ℕ), (α + 19 * (1:ℕ) = 180 ∧ α < 180 ∧ ∀ n, 1 ≤ n ∧ n ≤ 20 → α + (n - 1) * 1 < 180) ∧ α = 161 := 
by
  sorry

end NUMINAMATH_GPT_smallest_angle_in_convex_20_gon_seq_l1432_143215


namespace NUMINAMATH_GPT_smallest_sum_xy_min_45_l1432_143284

theorem smallest_sum_xy_min_45 (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y) (h4 : 1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 10) :
  x + y = 45 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_xy_min_45_l1432_143284


namespace NUMINAMATH_GPT_tangent_eq_tangent_intersect_other_l1432_143260

noncomputable def curve (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 - 9 * x^2 + 4

/-- Equation of the tangent line to curve C at x = 1 is y = -12x + 8 --/
theorem tangent_eq (tangent_line : ℝ → ℝ) (x : ℝ):
  tangent_line x = -12 * x + 8 :=
by
  sorry

/-- Apart from the tangent point (1, -4), the tangent line intersects the curve C at the points
    (-2, 32) and (2 / 3, 0) --/
theorem tangent_intersect_other (tangent_line : ℝ → ℝ) x:
  curve x = tangent_line x →
  (x = -2 ∧ curve (-2) = 32) ∨ (x = 2 / 3 ∧ curve (2 / 3) = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_eq_tangent_intersect_other_l1432_143260


namespace NUMINAMATH_GPT_polygon_sides_from_diagonals_l1432_143206

/-- A theorem to prove that a regular polygon with 740 diagonals has 40 sides. -/
theorem polygon_sides_from_diagonals (n : ℕ) (h : (n * (n - 3)) / 2 = 740) : n = 40 := sorry

end NUMINAMATH_GPT_polygon_sides_from_diagonals_l1432_143206


namespace NUMINAMATH_GPT_people_in_each_van_l1432_143290

theorem people_in_each_van
  (cars : ℕ) (taxis : ℕ) (vans : ℕ)
  (people_per_car : ℕ) (people_per_taxi : ℕ) (total_people : ℕ) 
  (people_per_van : ℕ) :
  cars = 3 → taxis = 6 → vans = 2 →
  people_per_car = 4 → people_per_taxi = 6 → total_people = 58 →
  3 * people_per_car + 6 * people_per_taxi + 2 * people_per_van = total_people →
  people_per_van = 5 :=
by sorry

end NUMINAMATH_GPT_people_in_each_van_l1432_143290


namespace NUMINAMATH_GPT_area_triangle_QCA_l1432_143270

noncomputable def area_of_triangle_QCA (p : ℝ) : ℝ :=
  let Q := (0, 12)
  let A := (3, 12)
  let C := (0, p)
  let QA := 3
  let QC := 12 - p
  (1/2) * QA * QC

theorem area_triangle_QCA (p : ℝ) : area_of_triangle_QCA p = (3/2) * (12 - p) :=
  sorry

end NUMINAMATH_GPT_area_triangle_QCA_l1432_143270


namespace NUMINAMATH_GPT_triangle_side_difference_l1432_143209

theorem triangle_side_difference (y : ℝ) (h : y > 6) :
  max (y + 6) (y + 3) - min (y + 6) (y + 3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_difference_l1432_143209


namespace NUMINAMATH_GPT_correct_answer_l1432_143243

theorem correct_answer (m n : ℤ) (h : 3 * m * n + 3 * m = n + 2) : 3 * m + n = -2 := 
by
  sorry

end NUMINAMATH_GPT_correct_answer_l1432_143243


namespace NUMINAMATH_GPT_race_distance_l1432_143262

/-
In a race, the ratio of the speeds of two contestants A and B is 3 : 4.
A has a start of 140 m.
A wins by 20 m.
Prove that the total distance of the race is 360 times the common speed factor.
-/
theorem race_distance (x D : ℕ)
  (ratio_A_B : ∀ (speed_A speed_B : ℕ), speed_A / speed_B = 3 / 4)
  (start_A : ∀ (start : ℕ), start = 140) 
  (win_A : ∀ (margin : ℕ), margin = 20) :
  D = 360 * x := 
sorry

end NUMINAMATH_GPT_race_distance_l1432_143262


namespace NUMINAMATH_GPT_time_ratio_l1432_143263

theorem time_ratio (A : ℝ) (B : ℝ) (h1 : B = 18) (h2 : 1 / A + 1 / B = 1 / 3) : A / B = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_time_ratio_l1432_143263
