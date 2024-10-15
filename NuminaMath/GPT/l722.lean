import Mathlib

namespace NUMINAMATH_GPT_hydroflow_rate_30_minutes_l722_72294

def hydroflow_pumped (rate_per_hour: ℕ) (minutes: ℕ) : ℕ :=
  let hours := minutes / 60
  rate_per_hour * hours

theorem hydroflow_rate_30_minutes : 
  hydroflow_pumped 500 30 = 250 :=
by 
  -- place the proof here
  sorry

end NUMINAMATH_GPT_hydroflow_rate_30_minutes_l722_72294


namespace NUMINAMATH_GPT_new_price_after_increase_l722_72280

def original_price : ℝ := 220
def percentage_increase : ℝ := 0.15

def new_price (original_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  original_price + (original_price * percentage_increase)

theorem new_price_after_increase : new_price original_price percentage_increase = 253 := 
by
  sorry

end NUMINAMATH_GPT_new_price_after_increase_l722_72280


namespace NUMINAMATH_GPT_vector_addition_proof_l722_72246

variables {Point : Type} [AddCommGroup Point]

variables (A B C D : Point)

theorem vector_addition_proof :
  (D - A) + (C - D) - (C - B) = B - A :=
by
  sorry

end NUMINAMATH_GPT_vector_addition_proof_l722_72246


namespace NUMINAMATH_GPT_only_integer_solution_l722_72210

theorem only_integer_solution (n : ℕ) (h1 : n > 1) (h2 : (2 * n + 1) % n ^ 2 = 0) : n = 3 := 
sorry

end NUMINAMATH_GPT_only_integer_solution_l722_72210


namespace NUMINAMATH_GPT_angle_B_is_pi_over_3_l722_72243

theorem angle_B_is_pi_over_3
  (A B C a b c : ℝ)
  (h1 : b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2)
  (h2 : 0 < B)
  (h3 : B < Real.pi)
  (h4 : 0 < A)
  (h5 : A < Real.pi)
  (h6 : 0 < C)
  (h7 : C < Real.pi) :
  B = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_B_is_pi_over_3_l722_72243


namespace NUMINAMATH_GPT_domain_f_l722_72236

def domain_of_f (x : ℝ) : Prop :=
  (2 ≤ x ∧ x < 3) ∨ (3 < x ∧ x < 4)

theorem domain_f :
  ∀ x, domain_of_f x ↔ (x ≥ 2 ∧ x < 4) ∧ x ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_domain_f_l722_72236


namespace NUMINAMATH_GPT_cycling_time_difference_l722_72250

-- Definitions from the conditions
def youth_miles : ℤ := 20
def youth_hours : ℤ := 2
def adult_miles : ℤ := 12
def adult_hours : ℤ := 3

-- Conversion from hours to minutes
def hours_to_minutes (hours : ℤ) : ℤ := hours * 60

-- Time per mile calculations
def youth_time_per_mile : ℤ := hours_to_minutes youth_hours / youth_miles
def adult_time_per_mile : ℤ := hours_to_minutes adult_hours / adult_miles

-- The difference in time per mile
def time_difference : ℤ := adult_time_per_mile - youth_time_per_mile

-- Theorem to prove the difference is 9 minutes
theorem cycling_time_difference : time_difference = 9 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_cycling_time_difference_l722_72250


namespace NUMINAMATH_GPT_tessa_initial_apples_l722_72224

theorem tessa_initial_apples (x : ℝ) (h : x + 5.0 - 4.0 = 11) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_tessa_initial_apples_l722_72224


namespace NUMINAMATH_GPT_mobius_trip_proof_l722_72258

noncomputable def mobius_trip_time : ℝ :=
  let speed_no_load := 13
  let speed_light_load := 12
  let speed_typical_load := 11
  let distance_total := 257
  let distance_typical := 120
  let distance_light := distance_total - distance_typical
  let time_typical := distance_typical / speed_typical_load
  let time_light := distance_light / speed_light_load
  let time_return := distance_total / speed_no_load
  let rest_first := (20 + 25 + 35) / 60.0
  let rest_second := (45 + 30) / 60.0
  time_typical + time_light + time_return + rest_first + rest_second

theorem mobius_trip_proof : mobius_trip_time = 44.6783 :=
  by sorry

end NUMINAMATH_GPT_mobius_trip_proof_l722_72258


namespace NUMINAMATH_GPT_ab_zero_proof_l722_72220

-- Given conditions
def square_side : ℝ := 3
def rect_short_side : ℝ := 3
def rect_long_side : ℝ := 6
def rect_area : ℝ := rect_short_side * rect_long_side
def split_side_proof (a b : ℝ) : Prop := a + b = rect_short_side

-- Lean theorem proving that ab = 0 given the conditions
theorem ab_zero_proof (a b : ℝ) 
  (h1 : square_side = 3)
  (h2 : rect_short_side = 3)
  (h3 : rect_long_side = 6)
  (h4 : rect_area = 18)
  (h5 : split_side_proof a b) : a * b = 0 := by
  sorry

end NUMINAMATH_GPT_ab_zero_proof_l722_72220


namespace NUMINAMATH_GPT_range_of_a_l722_72201

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x > 2 then 2^x + a else x + a^2

theorem range_of_a (a : ℝ) : (∀ y : ℝ, ∃ x : ℝ, f x a = y) ↔ (a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l722_72201


namespace NUMINAMATH_GPT_part1_part2_l722_72267

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) (b : ℝ) := 0.5 * x^2 - b * x
noncomputable def h (x : ℝ) (b : ℝ) := f x + g x b

theorem part1 (b : ℝ) :
  (∃ (tangent_point : ℝ),
    tangent_point = 1 ∧
    deriv f tangent_point = 1 ∧
    f tangent_point = 0 ∧
    ∃ (y_tangent : ℝ → ℝ), (∀ (x : ℝ), y_tangent x = x - 1) ∧
    ∃ (tangent_for_g : ℝ), (∀ (x : ℝ), y_tangent x = g x b)
  ) → false :=
sorry 

theorem part2 (b : ℝ) :
  ¬ (∀ (x : ℝ) (hx : 0 < x), deriv (h x) b = 0 → deriv (h x) b < 0) →
  2 < b :=
sorry

end NUMINAMATH_GPT_part1_part2_l722_72267


namespace NUMINAMATH_GPT_find_S7_l722_72208

variable {a : ℕ → ℚ} {S : ℕ → ℚ}

axiom a1_def : a 1 = 1 / 2
axiom a_next_def : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * S n + 1
axiom S_def : ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

theorem find_S7 : S 7 = 1457 / 2 := by
  sorry

end NUMINAMATH_GPT_find_S7_l722_72208


namespace NUMINAMATH_GPT_non_zero_const_c_l722_72297

theorem non_zero_const_c (a b c x1 x2 : ℝ) (h1 : x1 ≠ 0) (h2 : x2 ≠ 0) 
(h3 : (a - 1) * x1 ^ 2 + b * x1 + c = 0) 
(h4 : (a - 1) * x2 ^ 2 + b * x2 + c = 0)
(h5 : x1 * x2 = -1) 
(h6 : x1 ≠ x2) 
(h7 : x1 * x2 < 0): c ≠ 0 :=
sorry

end NUMINAMATH_GPT_non_zero_const_c_l722_72297


namespace NUMINAMATH_GPT_Mina_age_is_10_l722_72275

-- Define the conditions as Lean definitions
variable (S : ℕ)

def Minho_age := 3 * S
def Mina_age := 2 * S - 2

-- State the main problem as a theorem
theorem Mina_age_is_10 (h_sum : S + Minho_age S + Mina_age S = 34) : Mina_age S = 10 :=
by
  sorry

end NUMINAMATH_GPT_Mina_age_is_10_l722_72275


namespace NUMINAMATH_GPT_min_x8_x9_x10_eq_618_l722_72233

theorem min_x8_x9_x10_eq_618 (x : ℕ → ℕ) (h1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 10 → x i < x j)
  (h2 : x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9 + x 10 = 2023) :
  x 8 + x 9 + x 10 = 618 :=
sorry

end NUMINAMATH_GPT_min_x8_x9_x10_eq_618_l722_72233


namespace NUMINAMATH_GPT_total_monthly_bill_working_from_home_l722_72299

-- Definitions based on conditions
def original_bill : ℝ := 60
def increase_rate : ℝ := 0.45
def additional_internet_cost : ℝ := 25
def additional_cloud_cost : ℝ := 15

-- The theorem to prove
theorem total_monthly_bill_working_from_home : 
  original_bill * (1 + increase_rate) + additional_internet_cost + additional_cloud_cost = 127 := by
  sorry

end NUMINAMATH_GPT_total_monthly_bill_working_from_home_l722_72299


namespace NUMINAMATH_GPT_problem_1_problem_2_l722_72200

open Real

theorem problem_1
  (a b m n : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hm : m > 0)
  (hn : n > 0) :
  (m ^ 2 / a + n ^ 2 / b) ≥ ((m + n) ^ 2 / (a + b)) :=
sorry

theorem problem_2
  (x : ℝ)
  (hx1 : 0 < x)
  (hx2 : x < 1 / 2) :
  (2 / x + 9 / (1 - 2 * x)) ≥ 25 ∧ (2 / x + 9 / (1 - 2 * x)) = 25 ↔ x = 1 / 5 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l722_72200


namespace NUMINAMATH_GPT_parabola_conditions_l722_72202

-- Definitions based on conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - 4*x - 3 + a

def passes_through (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y

def intersects_at_2_points (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Proof Problem Statement
theorem parabola_conditions (a : ℝ) :
  (passes_through (quadratic_function a) 0 1 → a = 4) ∧
  (intersects_at_2_points (quadratic_function a) → (a = 3 ∨ a = 7)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_conditions_l722_72202


namespace NUMINAMATH_GPT_tangent_line_eq_l722_72256

theorem tangent_line_eq (x y : ℝ) (h_curve : y = x^3 + x + 1) (h_point : x = 1 ∧ y = 3) : 
  y = 4 * x - 1 := 
sorry

end NUMINAMATH_GPT_tangent_line_eq_l722_72256


namespace NUMINAMATH_GPT_train_length_l722_72234

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (h_speed : speed_kmh = 60) (h_time : time_s = 21) :
  (speed_kmh * (1000 / 3600) * time_s) = 350.07 := 
by
  sorry

end NUMINAMATH_GPT_train_length_l722_72234


namespace NUMINAMATH_GPT_arithmetic_sequence_S30_l722_72254

variable {α : Type*} [OrderedAddCommGroup α]

-- Definitions from the conditions
def arithmetic_sum (n : ℕ) : α :=
  sorry -- Placeholder for the sequence sum definition

axiom S10 : arithmetic_sum 10 = 20
axiom S20 : arithmetic_sum 20 = 15

-- The theorem to prove
theorem arithmetic_sequence_S30 : arithmetic_sum 30 = -15 :=
  sorry -- Proof will be completed here

end NUMINAMATH_GPT_arithmetic_sequence_S30_l722_72254


namespace NUMINAMATH_GPT_combined_avg_score_l722_72248

noncomputable def classA_student_count := 45
noncomputable def classB_student_count := 55
noncomputable def classA_avg_score := 110
noncomputable def classB_avg_score := 90

theorem combined_avg_score (nA nB : ℕ) (avgA avgB : ℕ) 
  (h1 : nA = classA_student_count) 
  (h2 : nB = classB_student_count) 
  (h3 : avgA = classA_avg_score) 
  (h4 : avgB = classB_avg_score) : 
  (nA * avgA + nB * avgB) / (nA + nB) = 99 := 
by 
  rw [h1, h2, h3, h4]
  -- Substitute the values to get:
  -- (45 * 110 + 55 * 90) / (45 + 55) 
  -- = (4950 + 4950) / 100 
  -- = 9900 / 100 
  -- = 99
  sorry

end NUMINAMATH_GPT_combined_avg_score_l722_72248


namespace NUMINAMATH_GPT_inequality_solution_l722_72259

theorem inequality_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b + c = 1) : (1 / (b * c + a + 1 / a) + 1 / (c * a + b + 1 / b) + 1 / (a * b + c + 1 / c) ≤ 27 / 31) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l722_72259


namespace NUMINAMATH_GPT_savings_for_23_students_is_30_yuan_l722_72295

-- Define the number of students
def number_of_students : ℕ := 23

-- Define the price per ticket in yuan
def price_per_ticket : ℕ := 10

-- Define the discount rate for the group ticket
def discount_rate : ℝ := 0.8

-- Define the group size that is eligible for the discount
def group_size_discount : ℕ := 25

-- Define the cost without ticket discount
def cost_without_discount : ℕ := number_of_students * price_per_ticket

-- Define the cost with the group ticket discount
def cost_with_discount : ℝ := price_per_ticket * discount_rate * group_size_discount

-- Define the expected amount saved by using the group discount
def expected_savings : ℝ := cost_without_discount - cost_with_discount

-- Theorem statement that the expected_savings is 30 yuan
theorem savings_for_23_students_is_30_yuan :
  expected_savings = 30 := 
sorry

end NUMINAMATH_GPT_savings_for_23_students_is_30_yuan_l722_72295


namespace NUMINAMATH_GPT_average_of_21_numbers_l722_72230

theorem average_of_21_numbers (n₁ n₂ : ℕ) (a b c : ℕ)
  (h₁ : n₁ = 11 * 48) -- Sum of the first 11 numbers
  (h₂ : n₂ = 11 * 41) -- Sum of the last 11 numbers
  (h₃ : c = 55) -- The 11th number
  : (n₁ + n₂ - c) / 21 = 44 := -- Average of all 21 numbers
by
  sorry

end NUMINAMATH_GPT_average_of_21_numbers_l722_72230


namespace NUMINAMATH_GPT_part1_part2_l722_72219

def my_mul (x y : Int) : Int :=
  if x = 0 then abs y
  else if y = 0 then abs x
  else if (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) then abs x + abs y
  else - (abs x + abs y)

theorem part1 : my_mul (-15) (my_mul 3 0) = -18 := 
  by
  sorry

theorem part2 (a : Int) : 
  my_mul 3 a + a = 
  if a < 0 then 2 * a - 3 
  else if a = 0 then 3
  else 2 * a + 3 :=
  by
  sorry

end NUMINAMATH_GPT_part1_part2_l722_72219


namespace NUMINAMATH_GPT_counterexample_exists_l722_72271

theorem counterexample_exists : 
  ∃ (m : ℤ), (∃ (k1 : ℤ), m = 2 * k1) ∧ ¬(∃ (k2 : ℤ), m = 4 * k2) := 
sorry

end NUMINAMATH_GPT_counterexample_exists_l722_72271


namespace NUMINAMATH_GPT_max_value_expression_l722_72240

theorem max_value_expression (x y : ℤ) (h : 3 * x^2 + 5 * y^2 = 345) : 
  ∃ (x y : ℤ), 3 * x^2 + 5 * y^2 = 345 ∧ (x + y = 13) := 
sorry

end NUMINAMATH_GPT_max_value_expression_l722_72240


namespace NUMINAMATH_GPT_find_three_digit_number_l722_72282

-- Definitions of digit constraints and the number representation
def is_three_digit_number (N : ℕ) (a b c : ℕ) : Prop :=
  N = 100 * a + 10 * b + c ∧ 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9

-- Definition of the problem condition
def sum_of_digits_condition (N : ℕ) (a b c : ℕ) : Prop :=
  a + b + c = N / 11

-- Lean theorem statement
theorem find_three_digit_number (N a b c : ℕ) :
  is_three_digit_number N a b c ∧ sum_of_digits_condition N a b c → N = 198 :=
by
  sorry

end NUMINAMATH_GPT_find_three_digit_number_l722_72282


namespace NUMINAMATH_GPT_largest_distinct_arithmetic_sequence_number_l722_72216

theorem largest_distinct_arithmetic_sequence_number :
  ∃ a b c d : ℕ, 
    (100 * a + 10 * b + c = 789) ∧ 
    (b - a = d) ∧ 
    (c - b = d) ∧ 
    (a ≠ b) ∧ 
    (b ≠ c) ∧ 
    (a ≠ c) ∧ 
    (a < 10) ∧ 
    (b < 10) ∧ 
    (c < 10) :=
sorry

end NUMINAMATH_GPT_largest_distinct_arithmetic_sequence_number_l722_72216


namespace NUMINAMATH_GPT_no_solution_exists_l722_72231

theorem no_solution_exists :
  ¬ ∃ x : ℝ, (x - 2) / (x + 2) - 16 / (x^2 - 4) = (x + 2) / (x - 2) :=
by sorry

end NUMINAMATH_GPT_no_solution_exists_l722_72231


namespace NUMINAMATH_GPT_iterated_kernels_l722_72211

noncomputable def K (x t : ℝ) : ℝ := 
  if 0 ≤ x ∧ x < t then 
    x + t 
  else if t < x ∧ x ≤ 1 then 
    x - t 
  else 
    0

noncomputable def K1 (x t : ℝ) : ℝ := K x t

noncomputable def K2 (x t : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < t then 
    (-2 / 3) * x^3 + t^3 - x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else if t < x ∧ x ≤ 1 then 
    (-2 / 3) * x^3 - t^3 + x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else
    0

theorem iterated_kernels (x t : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) :
  K1 x t = K x t ∧
  K2 x t = 
  if 0 ≤ x ∧ x < t then 
    (-2 / 3) * x^3 + t^3 - x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else if t < x ∧ x ≤ 1 then 
    (-2 / 3) * x^3 - t^3 + x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else
    0 := by
  sorry

end NUMINAMATH_GPT_iterated_kernels_l722_72211


namespace NUMINAMATH_GPT_volume_to_surface_area_ratio_l722_72283

theorem volume_to_surface_area_ratio (base_layer: ℕ) (top_layer: ℕ) (unit_cube_volume: ℕ) (unit_cube_faces_exposed_base: ℕ) (unit_cube_faces_exposed_top: ℕ) 
  (V : ℕ := base_layer * top_layer * unit_cube_volume) 
  (S : ℕ := base_layer * unit_cube_faces_exposed_base + top_layer * unit_cube_faces_exposed_top) 
  (ratio := V / S) : ratio = 1 / 2 :=
by
  -- Base Layer: 4 cubes, 3 faces exposed per cube
  have base_layer_faces : ℕ := 4 * 3
  -- Top Layer: 4 cubes, 1 face exposed per cube
  have top_layer_faces : ℕ := 4 * 1
  -- Total volume is 8
  have V : ℕ := 4 * 2
  -- Total surface area is 16
  have S : ℕ := base_layer_faces + top_layer_faces
  -- Volume to surface area ratio computation
  have ratio : ℕ := V / S
  sorry

end NUMINAMATH_GPT_volume_to_surface_area_ratio_l722_72283


namespace NUMINAMATH_GPT_ratio_of_tshirts_l722_72213

def spending_on_tshirts (Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats : ℝ) : Prop :=
  Lisa_tshirts = 40 ∧
  Lisa_jeans = Lisa_tshirts / 2 ∧
  Lisa_coats = 2 * Lisa_tshirts ∧
  Carly_jeans = 3 * Lisa_jeans ∧
  Carly_coats = Lisa_coats / 4 ∧
  Lisa_tshirts + Lisa_jeans + Lisa_coats + Carly_tshirts + Carly_jeans + Carly_coats = 230

theorem ratio_of_tshirts 
  (Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats : ℝ)
  (h : spending_on_tshirts Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats)
  : Carly_tshirts / Lisa_tshirts = 1 / 4 := 
sorry

end NUMINAMATH_GPT_ratio_of_tshirts_l722_72213


namespace NUMINAMATH_GPT_pqr_value_l722_72260

theorem pqr_value (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h1 : p + q + r = 30) 
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + (420 : ℚ) / (p * q * r) = 1) : 
  p * q * r = 1800 := 
sorry

end NUMINAMATH_GPT_pqr_value_l722_72260


namespace NUMINAMATH_GPT_greatest_radius_l722_72253

theorem greatest_radius (r : ℕ) (h : π * (r : ℝ)^2 < 50 * π) : r = 7 :=
sorry

end NUMINAMATH_GPT_greatest_radius_l722_72253


namespace NUMINAMATH_GPT_exists_nested_rectangles_l722_72214

theorem exists_nested_rectangles (rectangles : ℕ × ℕ → Prop) :
  (∀ n m : ℕ, rectangles (n, m)) → ∃ (n1 m1 n2 m2 : ℕ), n1 ≤ n2 ∧ m1 ≤ m2 ∧ rectangles (n1, m1) ∧ rectangles (n2, m2) :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_nested_rectangles_l722_72214


namespace NUMINAMATH_GPT_binomial_coeff_arith_seq_expansion_l722_72206

open BigOperators

-- Given the binomial expansion of (sqrt(x) + 2/sqrt(x))^n
-- we need to prove that the condition on binomial coefficients
-- implies that n = 7, and the expansion contains no constant term.
theorem binomial_coeff_arith_seq_expansion (x : ℝ) (n : ℕ) :
  (2 * Nat.choose n 2 = Nat.choose n 1 + Nat.choose n 3) ↔ n = 7 ∧ ∀ r : ℕ, x ^ (7 - 2 * r) / 2 ≠ x ^ 0 := by
  sorry

end NUMINAMATH_GPT_binomial_coeff_arith_seq_expansion_l722_72206


namespace NUMINAMATH_GPT_g_g_g_g_3_l722_72298

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_g_g_g_3 : g (g (g (g 3))) = 24 := by
  sorry

end NUMINAMATH_GPT_g_g_g_g_3_l722_72298


namespace NUMINAMATH_GPT_tara_dad_second_year_attendance_l722_72252

theorem tara_dad_second_year_attendance :
  let games_played_per_year := 20
  let attendance_rate := 0.90
  let first_year_games_attended := attendance_rate * games_played_per_year
  let second_year_games_difference := 4
  first_year_games_attended - second_year_games_difference = 14 :=
by
  -- We skip the proof here
  sorry

end NUMINAMATH_GPT_tara_dad_second_year_attendance_l722_72252


namespace NUMINAMATH_GPT_adam_money_ratio_l722_72262

theorem adam_money_ratio 
  (initial_dollars: ℕ) 
  (spent_dollars: ℕ) 
  (remaining_dollars: ℕ := initial_dollars - spent_dollars) 
  (ratio_numerator: ℕ := remaining_dollars / Nat.gcd remaining_dollars spent_dollars) 
  (ratio_denominator: ℕ := spent_dollars / Nat.gcd remaining_dollars spent_dollars) 
  (h_initial: initial_dollars = 91) 
  (h_spent: spent_dollars = 21) 
  (h_gcd: Nat.gcd (initial_dollars - spent_dollars) spent_dollars = 7) :
  ratio_numerator = 10 ∧ ratio_denominator = 3 := by
  sorry

end NUMINAMATH_GPT_adam_money_ratio_l722_72262


namespace NUMINAMATH_GPT_percentage_difference_l722_72273

theorem percentage_difference (x : ℝ) (h1 : 0.38 * 80 = 30.4) (h2 : 30.4 - (x / 100) * 160 = 11.2) :
    x = 12 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l722_72273


namespace NUMINAMATH_GPT_price_jemma_sells_each_frame_is_5_l722_72284

noncomputable def jemma_price_per_frame : ℝ :=
  let num_frames_jemma := 400
  let num_frames_dorothy := num_frames_jemma / 2
  let total_income := 2500
  let P_jemma := total_income / (num_frames_jemma + num_frames_dorothy / 2)
  P_jemma

theorem price_jemma_sells_each_frame_is_5 :
  jemma_price_per_frame = 5 := by
  sorry

end NUMINAMATH_GPT_price_jemma_sells_each_frame_is_5_l722_72284


namespace NUMINAMATH_GPT_cost_of_bananas_and_cantaloupe_l722_72241

theorem cost_of_bananas_and_cantaloupe (a b c d h : ℚ) 
  (h1: a + b + c + d + h = 30)
  (h2: d = 4 * a)
  (h3: c = 2 * a - b) :
  b + c = 50 / 7 := 
sorry

end NUMINAMATH_GPT_cost_of_bananas_and_cantaloupe_l722_72241


namespace NUMINAMATH_GPT_acquaintances_condition_l722_72223

theorem acquaintances_condition (n : ℕ) (hn : n > 1) (acquainted : ℕ → ℕ → Prop) :
  (∀ X Y, acquainted X Y → acquainted Y X) ∧
  (∀ X, ¬acquainted X X) →
  (∀ n, n ≠ 2 → n ≠ 4 → ∃ (A B : ℕ), (∃ (C : ℕ), acquainted C A ∧ acquainted C B) ∨ (∃ (D : ℕ), ¬acquainted D A ∧ ¬acquainted D B)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_acquaintances_condition_l722_72223


namespace NUMINAMATH_GPT_increased_numerator_value_l722_72264

theorem increased_numerator_value (x y a : ℝ) (h1 : x / y = 2 / 5) (h2 : (x + a) / (2 * y) = 1 / 3) (h3 : x + y = 5.25) : a = 1 :=
by
  -- skipped proof: sorry
  sorry

end NUMINAMATH_GPT_increased_numerator_value_l722_72264


namespace NUMINAMATH_GPT_sum_possible_values_l722_72296

theorem sum_possible_values (M : ℝ) (h : M * (M - 6) = -5) : ∀ x ∈ {M | M * (M - 6) = -5}, x + (-x) = 6 :=
by sorry

end NUMINAMATH_GPT_sum_possible_values_l722_72296


namespace NUMINAMATH_GPT_distance_home_gym_l722_72228

theorem distance_home_gym 
  (v_WangLei v_ElderSister : ℕ)  -- speeds in meters per minute
  (d_meeting : ℕ)                -- distance in meters from the gym to the meeting point
  (t_gym : ℕ)                    -- time in minutes for the older sister to the gym
  (speed_diff : v_ElderSister = v_WangLei + 20)  -- speed difference
  (t_gym_reached : d_meeting / 2 = (25 * (v_WangLei + 20)) - d_meeting): 
  v_WangLei * t_gym = 1500 :=
by
  sorry

end NUMINAMATH_GPT_distance_home_gym_l722_72228


namespace NUMINAMATH_GPT_determine_c_l722_72215

theorem determine_c (a c : ℝ) (h : (2 * a - 1) / -3 < - (c + 1) / -4) : c ≠ -1 ∧ (c > 0 ∨ c < 0) :=
by sorry

end NUMINAMATH_GPT_determine_c_l722_72215


namespace NUMINAMATH_GPT_percentage_x_equals_twenty_percent_of_487_50_is_65_l722_72261

theorem percentage_x_equals_twenty_percent_of_487_50_is_65
    (x : ℝ)
    (hx : x = 150)
    (y : ℝ)
    (hy : y = 487.50) :
    (∃ (P : ℝ), P * x = 0.20 * y ∧ P * 100 = 65) :=
by
  sorry

end NUMINAMATH_GPT_percentage_x_equals_twenty_percent_of_487_50_is_65_l722_72261


namespace NUMINAMATH_GPT_december_sales_fraction_l722_72222

variable (A : ℝ)

-- Define the total sales for January through November
def total_sales_jan_to_nov := 11 * A

-- Define the sales total for December, which is given as 5 times the average monthly sales from January to November
def sales_dec := 5 * A

-- Define the total sales for the year as the sum of January-November sales and December sales
def total_sales_year := total_sales_jan_to_nov + sales_dec

-- We need to prove that the fraction of the December sales to the total annual sales is 5/16
theorem december_sales_fraction : sales_dec / total_sales_year = 5 / 16 := by
  sorry

end NUMINAMATH_GPT_december_sales_fraction_l722_72222


namespace NUMINAMATH_GPT_ab_value_l722_72255

theorem ab_value (a b : ℝ) (h1 : b^2 - a^2 = 4) (h2 : a^2 + b^2 = 25) : abs (a * b) = Real.sqrt (609 / 4) := 
sorry

end NUMINAMATH_GPT_ab_value_l722_72255


namespace NUMINAMATH_GPT_number_of_adult_tickets_l722_72212

-- Let's define our conditions and the theorem to prove.
theorem number_of_adult_tickets (A C : ℕ) (h₁ : A + C = 522) (h₂ : 15 * A + 8 * C = 5086) : A = 131 :=
by
  sorry

end NUMINAMATH_GPT_number_of_adult_tickets_l722_72212


namespace NUMINAMATH_GPT_expand_polynomials_eq_l722_72242

-- Define the polynomials P(z) and Q(z)
def P (z : ℝ) : ℝ := 3 * z^3 + 2 * z^2 - 4 * z + 1
def Q (z : ℝ) : ℝ := 4 * z^4 - 3 * z^2 + 2

-- Define the result polynomial R(z)
def R (z : ℝ) : ℝ := 12 * z^7 + 8 * z^6 - 25 * z^5 - 2 * z^4 + 18 * z^3 + z^2 - 8 * z + 2

-- State the theorem that proves P(z) * Q(z) = R(z)
theorem expand_polynomials_eq :
  ∀ (z : ℝ), (P z) * (Q z) = R z :=
by
  intros z
  sorry

end NUMINAMATH_GPT_expand_polynomials_eq_l722_72242


namespace NUMINAMATH_GPT_probability_white_or_red_l722_72227

theorem probability_white_or_red (a b c : ℕ) : 
  (a + b) / (a + b + c) = (a + b) / (a + b + c) := by
  -- Conditions
  let total_balls := a + b + c
  let white_red_balls := a + b
  -- Goal
  have prob_white_or_red := white_red_balls / total_balls
  exact rfl

end NUMINAMATH_GPT_probability_white_or_red_l722_72227


namespace NUMINAMATH_GPT_simplify_complex_fraction_l722_72277

open Complex

theorem simplify_complex_fraction :
  (⟨2, 2⟩ : ℂ) / (⟨-3, 4⟩ : ℂ) = (⟨-14 / 25, -14 / 25⟩ : ℂ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_complex_fraction_l722_72277


namespace NUMINAMATH_GPT_complement_union_l722_72226

open Set

-- Define U to be the set of all real numbers
def U := @univ ℝ

-- Define the domain A for the function y = sqrt(x-2) + sqrt(x+1)
def A := {x : ℝ | x ≥ 2}

-- Define the domain B for the function y = sqrt(2x+4) / (x-3)
def B := {x : ℝ | x ≥ -2 ∧ x ≠ 3}

-- Theorem about the union of the complements
theorem complement_union : (U \ A ∪ U \ B) = {x : ℝ | x < 2 ∨ x = 3} := 
by
  sorry

end NUMINAMATH_GPT_complement_union_l722_72226


namespace NUMINAMATH_GPT_complement_union_eq_ge_two_l722_72245

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end NUMINAMATH_GPT_complement_union_eq_ge_two_l722_72245


namespace NUMINAMATH_GPT_find_smaller_number_l722_72239

-- Define the conditions
def condition1 (x y : ℤ) : Prop := x + y = 30
def condition2 (x y : ℤ) : Prop := x - y = 10

-- Define the theorem to prove the smaller number is 10
theorem find_smaller_number (x y : ℤ) (h1 : condition1 x y) (h2 : condition2 x y) : y = 10 := 
sorry

end NUMINAMATH_GPT_find_smaller_number_l722_72239


namespace NUMINAMATH_GPT_rectangle_k_value_l722_72274

theorem rectangle_k_value (a k : ℝ) (h1 : k > 0) (h2 : 2 * (3 * a + a) = k) (h3 : 3 * a^2 = k) : k = 64 / 3 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_k_value_l722_72274


namespace NUMINAMATH_GPT_sum_of_sequence_l722_72293

-- Definitions based on conditions
def a (n : ℕ) := 2 * n - 1
def b (n : ℕ) := 2^(a n) + n
def S (n : ℕ) := (Finset.range n).sum (λ i => b (i + 1))

-- The theorem assertion / problem statement
theorem sum_of_sequence (n : ℕ) : 
  S n = (2 * (4^n - 1)) / 3 + n * (n + 1) / 2 := 
sorry

end NUMINAMATH_GPT_sum_of_sequence_l722_72293


namespace NUMINAMATH_GPT_homer_total_points_l722_72285

noncomputable def first_try_points : ℕ := 400
noncomputable def second_try_points : ℕ := first_try_points - 70
noncomputable def third_try_points : ℕ := 2 * second_try_points
noncomputable def total_points : ℕ := first_try_points + second_try_points + third_try_points

theorem homer_total_points : total_points = 1390 :=
by
  -- Using the definitions above, we need to show that total_points = 1390
  sorry

end NUMINAMATH_GPT_homer_total_points_l722_72285


namespace NUMINAMATH_GPT_factorize_problem_1_factorize_problem_2_l722_72292

-- Problem 1 Statement
theorem factorize_problem_1 (a m : ℝ) : 2 * a * m^2 - 8 * a = 2 * a * (m + 2) * (m - 2) := 
sorry

-- Problem 2 Statement
theorem factorize_problem_2 (x y : ℝ) : (x - y)^2 + 4 * (x * y) = (x + y)^2 := 
sorry

end NUMINAMATH_GPT_factorize_problem_1_factorize_problem_2_l722_72292


namespace NUMINAMATH_GPT_min_students_with_both_l722_72209

-- Given conditions
def total_students : ℕ := 35
def students_with_brown_eyes : ℕ := 18
def students_with_lunch_box : ℕ := 25

-- Mathematical statement to prove the least number of students with both attributes
theorem min_students_with_both :
  ∃ x : ℕ, students_with_brown_eyes + students_with_lunch_box - total_students ≤ x ∧ x = 8 :=
sorry

end NUMINAMATH_GPT_min_students_with_both_l722_72209


namespace NUMINAMATH_GPT_pauline_total_spent_l722_72229

variable {items_total : ℝ} (discount_rate : ℝ) (discount_limit : ℝ) (sales_tax_rate : ℝ)

def total_spent (items_total discount_rate discount_limit sales_tax_rate : ℝ) : ℝ :=
  let discount_amount := discount_rate * discount_limit
  let discounted_total := discount_limit - discount_amount
  let non_discounted_total := items_total - discount_limit
  let subtotal := discounted_total + non_discounted_total
  let sales_tax := sales_tax_rate * subtotal
  subtotal + sales_tax

theorem pauline_total_spent :
  total_spent 250 0.15 100 0.08 = 253.80 :=
by
  sorry

end NUMINAMATH_GPT_pauline_total_spent_l722_72229


namespace NUMINAMATH_GPT_parking_lot_total_spaces_l722_72257

-- Given conditions
def section1_spaces := 320
def section2_spaces := 440
def section3_spaces := section2_spaces - 200
def total_spaces := section1_spaces + section2_spaces + section3_spaces

-- Problem statement to be proved
theorem parking_lot_total_spaces : total_spaces = 1000 :=
by
  sorry

end NUMINAMATH_GPT_parking_lot_total_spaces_l722_72257


namespace NUMINAMATH_GPT_arc_length_l722_72269

theorem arc_length (r α : ℝ) (h1 : r = 3) (h2 : α = π / 3) : r * α = π :=
by
  rw [h1, h2]
  norm_num
  sorry -- This is the step where actual simplification and calculation will happen

end NUMINAMATH_GPT_arc_length_l722_72269


namespace NUMINAMATH_GPT_coefficients_sum_l722_72205

theorem coefficients_sum:
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (1+x)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h_eq
  have h0 : a_0 = 1
  sorry -- proof when x=0
  have h1 : a_1 + a_2 + a_3 + a_4 + a_5 = 31
  sorry -- proof when x=1
  exact h1

end NUMINAMATH_GPT_coefficients_sum_l722_72205


namespace NUMINAMATH_GPT_smallest_N_circular_table_l722_72272

theorem smallest_N_circular_table (N chairs : ℕ) (circular_seating : N < chairs) :
  (∀ new_person_reserved : ℕ, new_person_reserved < chairs →
    (∃ i : ℕ, (i < N) ∧ (new_person_reserved = (i + 1) % chairs ∨ 
                           new_person_reserved = (i - 1) % chairs))) ↔ N = 18 := by
sorry

end NUMINAMATH_GPT_smallest_N_circular_table_l722_72272


namespace NUMINAMATH_GPT_average_temperature_is_95_l722_72237

noncomputable def tempNY := 80
noncomputable def tempMiami := tempNY + 10
noncomputable def tempSD := tempMiami + 25
noncomputable def avg_temp := (tempNY + tempMiami + tempSD) / 3

theorem average_temperature_is_95 :
  avg_temp = 95 :=
by
  sorry

end NUMINAMATH_GPT_average_temperature_is_95_l722_72237


namespace NUMINAMATH_GPT_solution_to_system_l722_72244

theorem solution_to_system :
  (∀ (x y : ℚ), (y - x - 1 = 0) ∧ (y + x - 2 = 0) ↔ (x = 1/2 ∧ y = 3/2)) :=
by
  sorry

end NUMINAMATH_GPT_solution_to_system_l722_72244


namespace NUMINAMATH_GPT_pizza_slices_l722_72265

theorem pizza_slices (S L : ℕ) (h1 : S + L = 36) (h2 : L = 2 * S) :
  (8 * S + 12 * L) = 384 :=
by
  sorry

end NUMINAMATH_GPT_pizza_slices_l722_72265


namespace NUMINAMATH_GPT_total_cost_is_correct_l722_72249

-- Conditions
def cost_per_object : ℕ := 11
def objects_per_person : ℕ := 5  -- 2 shoes, 2 socks, 1 mobile per person
def number_of_people : ℕ := 3

-- Expected total cost
def expected_total_cost : ℕ := 165

-- Proof problem: Prove that the total cost for storing all objects is 165 dollars
theorem total_cost_is_correct :
  (number_of_people * objects_per_person * cost_per_object) = expected_total_cost :=
by
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l722_72249


namespace NUMINAMATH_GPT_a_n_less_than_inverse_n_minus_1_l722_72218

theorem a_n_less_than_inverse_n_minus_1 
  (n : ℕ) (h1 : 2 ≤ n) 
  (a : ℕ → ℝ) 
  (h2 : ∀ k : ℕ, 1 ≤ k → k ≤ n-1 → (a (k-1) + a k) * (a k + a (k+1)) = a (k-1) - a (k+1)) 
  (h3 : ∀ m : ℕ, m ≤ n → 0 < a m) : 
  a n < 1 / (n - 1) :=
sorry

end NUMINAMATH_GPT_a_n_less_than_inverse_n_minus_1_l722_72218


namespace NUMINAMATH_GPT_g_triple_application_l722_72238

def g (x : ℤ) : ℤ := 7 * x - 3

theorem g_triple_application : g (g (g 3)) = 858 := by
  sorry

end NUMINAMATH_GPT_g_triple_application_l722_72238


namespace NUMINAMATH_GPT_cos_960_eq_neg_half_l722_72278

theorem cos_960_eq_neg_half (cos : ℝ → ℝ) (h1 : ∀ x, cos (x + 360) = cos x) 
  (h_even : ∀ x, cos (-x) = cos x) (h_cos120 : cos 120 = - cos 60)
  (h_cos60 : cos 60 = 1 / 2) : cos 960 = -(1 / 2) := by
  sorry

end NUMINAMATH_GPT_cos_960_eq_neg_half_l722_72278


namespace NUMINAMATH_GPT_observable_sea_creatures_l722_72247

theorem observable_sea_creatures (P_shark : ℝ) (P_truth : ℝ) (n : ℕ)
  (h1 : P_shark = 0.027777777777777773)
  (h2 : P_truth = 1/6)
  (h3 : P_shark = P_truth * (1/n : ℝ)) : 
  n = 6 := 
  sorry

end NUMINAMATH_GPT_observable_sea_creatures_l722_72247


namespace NUMINAMATH_GPT_problem_statement_l722_72291

theorem problem_statement (x y : ℝ) (M N P : ℝ) 
  (hM_def : M = 2 * x + y)
  (hN_def : N = 2 * x - y)
  (hP_def : P = x * y)
  (hM : M = 4)
  (hN : N = 2) : P = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l722_72291


namespace NUMINAMATH_GPT_bananas_per_monkey_l722_72263

-- Define the given conditions
def total_monkeys : ℕ := 12
def piles_with_9hands : ℕ := 6
def hands_per_pile_9hands : ℕ := 9
def bananas_per_hand_9hands : ℕ := 14
def piles_with_12hands : ℕ := 4
def hands_per_pile_12hands : ℕ := 12
def bananas_per_hand_12hands : ℕ := 9

-- Calculate the total number of bananas from each type of pile
def total_bananas_9hands : ℕ := piles_with_9hands * hands_per_pile_9hands * bananas_per_hand_9hands
def total_bananas_12hands : ℕ := piles_with_12hands * hands_per_pile_12hands * bananas_per_hand_12hands

-- Sum the total number of bananas
def total_bananas : ℕ := total_bananas_9hands + total_bananas_12hands

-- Prove that each monkey gets 99 bananas
theorem bananas_per_monkey : total_bananas / total_monkeys = 99 := by
  sorry

end NUMINAMATH_GPT_bananas_per_monkey_l722_72263


namespace NUMINAMATH_GPT_find_w_l722_72266

variables {x y z w : ℝ}

theorem find_w (h : (1 / x) + (1 / y) + (1 / z) = 1 / w) :
  w = (x * y * z) / (y * z + x * z + x * y) := by
  sorry

end NUMINAMATH_GPT_find_w_l722_72266


namespace NUMINAMATH_GPT_parts_purchased_l722_72251

noncomputable def price_per_part : ℕ := 80
noncomputable def total_paid_after_discount : ℕ := 439
noncomputable def total_discount : ℕ := 121

theorem parts_purchased : 
  ∃ n : ℕ, price_per_part * n - total_discount = total_paid_after_discount → n = 7 :=
by
  sorry

end NUMINAMATH_GPT_parts_purchased_l722_72251


namespace NUMINAMATH_GPT_domain_of_function_l722_72288

def function_undefined_at (x : ℝ) : Prop :=
  ∃ y : ℝ, y = (x - 3) / (x - 2)

theorem domain_of_function (x : ℝ) : ¬(x = 2) ↔ function_undefined_at x :=
sorry

end NUMINAMATH_GPT_domain_of_function_l722_72288


namespace NUMINAMATH_GPT_container_alcohol_amount_l722_72235

theorem container_alcohol_amount
  (A : ℚ) -- Amount of alcohol in quarts
  (initial_water : ℚ) -- Initial amount of water in quarts
  (added_water : ℚ) -- Amount of water added in quarts
  (final_ratio_alcohol_to_water : ℚ) -- Final ratio of alcohol to water
  (h_initial_water : initial_water = 4) -- Container initially contains 4 quarts of water.
  (h_added_water : added_water = 8/3) -- 2.666666666666667 quarts of water added.
  (h_final_ratio : final_ratio_alcohol_to_water = 3/5) -- Final ratio is 3 parts alcohol to 5 parts water.
  (h_final_water : initial_water + added_water = 20/3) -- Total final water quarts after addition.
  : A = 4 := 
sorry

end NUMINAMATH_GPT_container_alcohol_amount_l722_72235


namespace NUMINAMATH_GPT_solution_l722_72286

theorem solution (x : ℝ) : (x = -2/5) → (x < x^3 ∧ x^3 < x^2) :=
by
  intro h
  rw [h]
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_solution_l722_72286


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l722_72204

def f (a x : ℝ) : ℝ := x^2 - a * x + 1

theorem necessary_and_sufficient_condition (a : ℝ) : 
  (∃ x : ℝ, f a x < 0) ↔ |a| > 2 :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l722_72204


namespace NUMINAMATH_GPT_questionnaires_drawn_from_D_l722_72268

theorem questionnaires_drawn_from_D (a1 a2 a3 a4 total sample_b sample_total sample_d : ℕ)
  (h1 : a2 - a1 = a3 - a2)
  (h2 : a3 - a2 = a4 - a3)
  (h3 : a1 + a2 + a3 + a4 = total)
  (h4 : total = 1000)
  (h5 : sample_b = 30)
  (h6 : a2 = 200)
  (h7 : sample_total = 150)
  (h8 : sample_d * total = sample_total * a4) :
  sample_d = 60 :=
by sorry

end NUMINAMATH_GPT_questionnaires_drawn_from_D_l722_72268


namespace NUMINAMATH_GPT_scientific_notation_361000000_l722_72203

theorem scientific_notation_361000000 :
  361000000 = 3.61 * 10^8 :=
sorry

end NUMINAMATH_GPT_scientific_notation_361000000_l722_72203


namespace NUMINAMATH_GPT_find_a8_l722_72207

theorem find_a8 (a : ℕ → ℤ) (x : ℤ) :
  (1 + x)^10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 +
               a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 +
               a 7 * (1 - x)^7 + a 8 * (1 - x)^8 + a 9 * (1 - x)^9 +
               a 10 * (1 - x)^10 → a 8 = 180 := by
  sorry

end NUMINAMATH_GPT_find_a8_l722_72207


namespace NUMINAMATH_GPT_sum_greater_l722_72276

theorem sum_greater {a b c d : ℝ} (h1 : b + Real.sin a > d + Real.sin c) (h2 : a + Real.sin b > c + Real.sin d) : a + b > c + d := by
  sorry

end NUMINAMATH_GPT_sum_greater_l722_72276


namespace NUMINAMATH_GPT_double_root_conditions_l722_72290

theorem double_root_conditions (k : ℝ) :
  (∃ x, (k - 1)/(x^2 - 1) - 1/(x - 1) = k/(x + 1) ∧ (∀ ε > 0, (∃ δ > 0, (∀ y, |y - x| < δ → (k - 1)/(y^2 - 1) - 1/(y - 1) = k/(y + 1)))))
  → k = 3 ∨ k = 1/3 :=
sorry

end NUMINAMATH_GPT_double_root_conditions_l722_72290


namespace NUMINAMATH_GPT_proof_problem_l722_72217

theorem proof_problem 
  {a b c : ℝ} (h_cond : 1/a + 1/b + 1/c = 1/(a + b + c))
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (n : ℕ) :
  1/a^(2*n+1) + 1/b^(2*n+1) + 1/c^(2*n+1) = 1/(a^(2*n+1) + b^(2*n+1) + c^(2*n+1)) :=
sorry

end NUMINAMATH_GPT_proof_problem_l722_72217


namespace NUMINAMATH_GPT_sin_eq_sqrt3_div_2_l722_72279

open Real

theorem sin_eq_sqrt3_div_2 (theta : ℝ) :
  sin theta = (sqrt 3) / 2 ↔ (∃ k : ℤ, theta = π/3 + 2*k*π ∨ theta = 2*π/3 + 2*k*π) :=
by
  sorry

end NUMINAMATH_GPT_sin_eq_sqrt3_div_2_l722_72279


namespace NUMINAMATH_GPT_relative_complement_correct_l722_72221

noncomputable def M : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}
def complement_M_N : Set ℤ := {x ∈ M | x ∉ N}

theorem relative_complement_correct : complement_M_N = {-1, 0, 3} := 
by
  sorry

end NUMINAMATH_GPT_relative_complement_correct_l722_72221


namespace NUMINAMATH_GPT_solve_for_a_l722_72289

-- Define the line equation and the condition of equal intercepts
def line_eq (a x y : ℝ) : Prop :=
  a * x + y - 2 - a = 0

def equal_intercepts (a : ℝ) : Prop :=
  (∀ x, line_eq a x 0 → x = 2 + a) ∧ (∀ y, line_eq a 0 y → y = 2 + a)

-- State the problem to prove the value of 'a'
theorem solve_for_a (a : ℝ) : equal_intercepts a → (a = -2 ∨ a = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l722_72289


namespace NUMINAMATH_GPT_area_of_EFGH_l722_72270

-- Define the dimensions of the smaller rectangles
def smaller_rectangle_short_side : ℕ := 7
def smaller_rectangle_long_side : ℕ := 2 * smaller_rectangle_short_side

-- Define the configuration of rectangles
def width_EFGH : ℕ := 2 * smaller_rectangle_short_side
def length_EFGH : ℕ := smaller_rectangle_long_side

-- Prove that the area of rectangle EFGH is 196 square feet
theorem area_of_EFGH : width_EFGH * length_EFGH = 196 := by
  sorry

end NUMINAMATH_GPT_area_of_EFGH_l722_72270


namespace NUMINAMATH_GPT_bike_race_difference_l722_72225

-- Define the conditions
def carlos_miles : ℕ := 70
def dana_miles : ℕ := 50
def time_period : ℕ := 5

-- State the theorem to prove the difference in miles biked
theorem bike_race_difference :
  carlos_miles - dana_miles = 20 := 
sorry

end NUMINAMATH_GPT_bike_race_difference_l722_72225


namespace NUMINAMATH_GPT_units_digit_17_pow_28_l722_72232

theorem units_digit_17_pow_28 : (17 ^ 28) % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_17_pow_28_l722_72232


namespace NUMINAMATH_GPT_find_vertex_D_l722_72287

noncomputable def quadrilateral_vertices : Prop :=
  let A : (ℤ × ℤ) := (-1, -2)
  let B : (ℤ × ℤ) := (3, 1)
  let C : (ℤ × ℤ) := (0, 2)
  A ≠ B ∧ A ≠ C ∧ B ≠ C

theorem find_vertex_D (A B C D : ℤ × ℤ) (h_quad : quadrilateral_vertices) :
    (A = (-1, -2)) →
    (B = (3, 1)) →
    (C = (0, 2)) →
    (B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2) →
    D = (-4, -1) :=
by
  sorry

end NUMINAMATH_GPT_find_vertex_D_l722_72287


namespace NUMINAMATH_GPT_probability_of_neither_tamil_nor_english_l722_72281

-- Definitions based on the conditions
def TotalPopulation := 1500
def SpeakTamil := 800
def SpeakEnglish := 650
def SpeakTamilAndEnglish := 250

-- Use Inclusion-Exclusion Principle
def SpeakTamilOrEnglish : ℕ := SpeakTamil + SpeakEnglish - SpeakTamilAndEnglish

-- Number of people who speak neither Tamil nor English
def SpeakNeitherTamilNorEnglish : ℕ := TotalPopulation - SpeakTamilOrEnglish

-- The probability calculation
def Probability := (SpeakNeitherTamilNorEnglish : ℚ) / (TotalPopulation : ℚ)

-- Theorem to prove
theorem probability_of_neither_tamil_nor_english : Probability = (1/5 : ℚ) :=
sorry

end NUMINAMATH_GPT_probability_of_neither_tamil_nor_english_l722_72281
