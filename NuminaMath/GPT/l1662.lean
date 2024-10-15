import Mathlib

namespace NUMINAMATH_GPT_trapezoid_area_condition_l1662_166228

theorem trapezoid_area_condition
  (a x y z : ℝ)
  (h_sq  : ∀ (ABCD : ℝ), ABCD = a * a)
  (h_trap: ∀ (EBCF : ℝ), EBCF = x * a)
  (h_rec : ∀ (JKHG : ℝ), JKHG = y * z)
  (h_sum : y + z = a)
  (h_area : x * a = a * a - 2 * y * z) :
  x = a / 2 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_condition_l1662_166228


namespace NUMINAMATH_GPT_triangle_solutions_l1662_166269

theorem triangle_solutions :
  ∀ (a b c : ℝ) (A B C : ℝ),
  a = 7.012 ∧
  c - b = 1.753 ∧
  B = 38 + 12/60 + 48/3600 ∧
  A = 81 + 47/60 + 12.5/3600 ∧
  C = 60 ∧
  b = 4.3825 ∧
  c = 6.1355 :=
sorry -- Proof goes here

end NUMINAMATH_GPT_triangle_solutions_l1662_166269


namespace NUMINAMATH_GPT_arithmetic_geometric_sum_l1662_166299

noncomputable def a_n (n : ℕ) := 3 * n - 2
noncomputable def b_n (n : ℕ) := 4 ^ (n - 1)

theorem arithmetic_geometric_sum (n : ℕ) :
    a_n 1 = 1 ∧ a_n 2 = b_n 2 ∧ a_n 6 = b_n 3 ∧ S_n = 1 + (n - 1) * 4 ^ n :=
by sorry

end NUMINAMATH_GPT_arithmetic_geometric_sum_l1662_166299


namespace NUMINAMATH_GPT_income_percentage_l1662_166267

theorem income_percentage (J T M : ℝ) 
  (h1 : T = 0.5 * J) 
  (h2 : M = 1.6 * T) : 
  M = 0.8 * J :=
by 
  sorry

end NUMINAMATH_GPT_income_percentage_l1662_166267


namespace NUMINAMATH_GPT_smallest_b_in_ap_l1662_166224

-- Definition of an arithmetic progression
def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

-- Problem statement in Lean
theorem smallest_b_in_ap (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h_ap : is_arithmetic_progression a b c) 
  (h_prod : a * b * c = 216) : 
  b ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_in_ap_l1662_166224


namespace NUMINAMATH_GPT_sec_150_eq_neg_2_sqrt_3_div_3_csc_150_eq_2_l1662_166210

noncomputable def sec (x : ℝ) := 1 / Real.cos x
noncomputable def csc (x : ℝ) := 1 / Real.sin x

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := by
  sorry

theorem csc_150_eq_2 : csc (150 * Real.pi / 180) = 2 := by
  sorry

end NUMINAMATH_GPT_sec_150_eq_neg_2_sqrt_3_div_3_csc_150_eq_2_l1662_166210


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1662_166223

-- Define the conditions as seen in the problem statement
def condition_x (x : ℝ) : Prop := x < 0
def condition_ln (x : ℝ) : Prop := Real.log (x + 1) < 0

-- State that the condition "x < 0" is necessary but not sufficient for "ln(x + 1) < 0"
theorem necessary_but_not_sufficient :
  ∀ (x : ℝ), (condition_ln x → condition_x x) ∧ ¬(condition_x x → condition_ln x) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1662_166223


namespace NUMINAMATH_GPT_calculate_sum_of_squares_l1662_166285

variables {a b : ℤ}
theorem calculate_sum_of_squares (h1 : (a + b)^2 = 17) (h2 : (a - b)^2 = 11) : a^2 + b^2 = 14 :=
by
  sorry

end NUMINAMATH_GPT_calculate_sum_of_squares_l1662_166285


namespace NUMINAMATH_GPT_negation_of_p_correct_l1662_166261

def p := ∀ x : ℝ, x > Real.sin x

theorem negation_of_p_correct :
  (¬ p) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by
  sorry

end NUMINAMATH_GPT_negation_of_p_correct_l1662_166261


namespace NUMINAMATH_GPT_polynomial_solution_l1662_166294

variable (P : ℝ → ℝ → ℝ)

theorem polynomial_solution :
  (∀ x y : ℝ, P (x + y) (x - y) = 2 * P x y) →
  (∃ b c d : ℝ, ∀ x y : ℝ, P x y = b * x^2 + c * x * y + d * y^2) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l1662_166294


namespace NUMINAMATH_GPT_temperature_at_6_km_l1662_166251

-- Define the initial conditions
def groundTemperature : ℝ := 25
def temperatureDropPerKilometer : ℝ := 5

-- Define the question which is the temperature at a height of 6 kilometers
def temperatureAtHeight (height : ℝ) : ℝ :=
  groundTemperature - temperatureDropPerKilometer * height

-- Prove that the temperature at 6 kilometers is -5 degrees Celsius
theorem temperature_at_6_km : temperatureAtHeight 6 = -5 := by
  -- Use expected proof  
  simp [temperatureAtHeight, groundTemperature, temperatureDropPerKilometer]
  sorry

end NUMINAMATH_GPT_temperature_at_6_km_l1662_166251


namespace NUMINAMATH_GPT_rate_per_kg_mangoes_l1662_166246

theorem rate_per_kg_mangoes (kg_apples kg_mangoes total_cost rate_apples total_payment rate_mangoes : ℕ) 
  (h1 : kg_apples = 8) 
  (h2 : rate_apples = 70)
  (h3 : kg_mangoes = 9)
  (h4 : total_payment = 965) :
  rate_mangoes = 45 := 
by
  sorry

end NUMINAMATH_GPT_rate_per_kg_mangoes_l1662_166246


namespace NUMINAMATH_GPT_worker_assignment_l1662_166234

theorem worker_assignment :
  ∃ (x y : ℕ), x + y = 85 ∧
  (16 * x) / 2 = (10 * y) / 3 ∧
  x = 25 ∧ y = 60 :=
by
  sorry

end NUMINAMATH_GPT_worker_assignment_l1662_166234


namespace NUMINAMATH_GPT_find_f_neg4_l1662_166282

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^2 - a * x + b

theorem find_f_neg4 (a b : ℝ) (h1 : f 1 a b = -1) (h2 : f 2 a b = 2) : 
  f (-4) a b = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg4_l1662_166282


namespace NUMINAMATH_GPT_twelve_sided_die_expected_value_l1662_166241

theorem twelve_sided_die_expected_value : 
  ∃ (E : ℝ), (E = 6.5) :=
by
  let n := 12
  let numerator := n * (n + 1) / 2
  let expected_value := numerator / n
  use expected_value
  sorry

end NUMINAMATH_GPT_twelve_sided_die_expected_value_l1662_166241


namespace NUMINAMATH_GPT_Janice_earnings_after_deductions_l1662_166227

def dailyEarnings : ℕ := 30
def daysWorked : ℕ := 6
def weekdayOvertimeRate : ℕ := 15
def weekendOvertimeRate : ℕ := 20
def weekdayOvertimeShifts : ℕ := 2
def weekendOvertimeShifts : ℕ := 1
def tipsReceived : ℕ := 10
def taxRate : ℝ := 0.10

noncomputable def calculateEarnings : ℝ :=
  let regularEarnings := dailyEarnings * daysWorked
  let overtimeEarnings := (weekdayOvertimeRate * weekdayOvertimeShifts) + (weekendOvertimeRate * weekendOvertimeShifts)
  let totalEarningsBeforeTax := regularEarnings + overtimeEarnings + tipsReceived
  let taxAmount := totalEarningsBeforeTax * taxRate
  totalEarningsBeforeTax - taxAmount

theorem Janice_earnings_after_deductions :
  calculateEarnings = 216 := by
  sorry

end NUMINAMATH_GPT_Janice_earnings_after_deductions_l1662_166227


namespace NUMINAMATH_GPT_larger_square_side_length_l1662_166221

theorem larger_square_side_length (x y H : ℝ) 
  (smaller_square_perimeter : 4 * x = H - 20)
  (larger_square_perimeter : 4 * y = H) :
  y = x + 5 :=
by
  sorry

end NUMINAMATH_GPT_larger_square_side_length_l1662_166221


namespace NUMINAMATH_GPT_opposite_of_negative_2020_is_2020_l1662_166287

theorem opposite_of_negative_2020_is_2020 :
  ∃ x : ℤ, -2020 + x = 0 :=
by
  use 2020
  sorry

end NUMINAMATH_GPT_opposite_of_negative_2020_is_2020_l1662_166287


namespace NUMINAMATH_GPT_usual_time_eq_three_l1662_166200

variable (S T : ℝ)
variable (usual_speed : S > 0)
variable (usual_time : T > 0)
variable (reduced_speed : S' = 6/7 * S)
variable (reduced_time : T' = T + 0.5)

theorem usual_time_eq_three (h : 7/6 = T' / T) : T = 3 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_usual_time_eq_three_l1662_166200


namespace NUMINAMATH_GPT_circle_equation_tangent_x_axis_l1662_166254

theorem circle_equation_tangent_x_axis (x y : ℝ) (center : ℝ × ℝ) (r : ℝ) 
  (h_center : center = (-1, 2)) 
  (h_tangent : r = |2 - 0|) :
  (x + 1)^2 + (y - 2)^2 = 4 := 
sorry

end NUMINAMATH_GPT_circle_equation_tangent_x_axis_l1662_166254


namespace NUMINAMATH_GPT_maximize_hotel_profit_l1662_166226

theorem maximize_hotel_profit :
  let rooms := 50
  let base_price := 180
  let increase_per_vacancy := 10
  let maintenance_cost := 20
  ∃ (x : ℕ), ((base_price + increase_per_vacancy * x) * (rooms - x) 
    - maintenance_cost * (rooms - x) = 10890) ∧ (base_price + increase_per_vacancy * x = 350) :=
by
  sorry

end NUMINAMATH_GPT_maximize_hotel_profit_l1662_166226


namespace NUMINAMATH_GPT_problem_l1662_166213

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f (-x) = f x)  -- f is an even function
variable (h_mono : ∀ x y : ℝ, 0 < x → x < y → f y < f x)  -- f is monotonically decreasing on (0, +∞)

theorem problem : f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1662_166213


namespace NUMINAMATH_GPT_yogurt_production_cost_l1662_166266

-- Define the conditions
def milk_cost_per_liter : ℝ := 1.5
def fruit_cost_per_kg : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Define the theorem statement
theorem yogurt_production_cost : 
  (milk_cost_per_liter * milk_needed_per_batch + fruit_cost_per_kg * fruit_needed_per_batch) * batches = 63 := 
  by 
  sorry

end NUMINAMATH_GPT_yogurt_production_cost_l1662_166266


namespace NUMINAMATH_GPT_jordan_has_11_oreos_l1662_166296

-- Define the conditions
def jamesOreos (x : ℕ) : ℕ := 3 + 2 * x
def totalOreos (jordanOreos : ℕ) : ℕ := 36

-- Theorem stating the problem that Jordan has 11 Oreos given the conditions
theorem jordan_has_11_oreos (x : ℕ) (h1 : jamesOreos x + x = totalOreos x) : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_jordan_has_11_oreos_l1662_166296


namespace NUMINAMATH_GPT_Joe_time_from_home_to_school_l1662_166263

-- Define the parameters
def walking_time := 4 -- minutes
def waiting_time := 2 -- minutes
def running_speed_ratio := 2 -- Joe's running speed is twice his walking speed

-- Define the walking and running times
def running_time (walking_time : ℕ) (running_speed_ratio : ℕ) : ℕ :=
  walking_time / running_speed_ratio

-- Total time it takes Joe to get from home to school
def total_time (walking_time waiting_time : ℕ) (running_speed_ratio : ℕ) : ℕ :=
  walking_time + waiting_time + running_time walking_time running_speed_ratio

-- Conjecture to be proved
theorem Joe_time_from_home_to_school :
  total_time walking_time waiting_time running_speed_ratio = 10 := by
  sorry

end NUMINAMATH_GPT_Joe_time_from_home_to_school_l1662_166263


namespace NUMINAMATH_GPT_length_of_jordans_rectangle_l1662_166231

theorem length_of_jordans_rectangle
  (carol_length : ℕ) (carol_width : ℕ) (jordan_width : ℕ) (equal_area : (carol_length * carol_width) = (jordan_width * 2)) :
  (2 = 120 / 60) := by
  sorry

end NUMINAMATH_GPT_length_of_jordans_rectangle_l1662_166231


namespace NUMINAMATH_GPT_range_of_a_l1662_166222

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → (x < a ∨ x > a + 4)) ∧ ¬(∀ x : ℝ, (x < a ∨ x > a + 4) → -2 ≤ x ∧ x ≤ 1) ↔
  a > 1 ∨ a < -6 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l1662_166222


namespace NUMINAMATH_GPT_central_angle_of_sector_l1662_166219

/-- The central angle of the sector obtained by unfolding the lateral surface of a cone with
    base radius 1 and slant height 2 is \(\pi\). -/
theorem central_angle_of_sector (r_base : ℝ) (r_slant : ℝ) (α : ℝ)
  (h1 : r_base = 1) (h2 : r_slant = 2) (h3 : 2 * π = α * r_slant) : α = π :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l1662_166219


namespace NUMINAMATH_GPT_rectangle_in_right_triangle_dimensions_l1662_166288

theorem rectangle_in_right_triangle_dimensions :
  ∀ (DE EF DF x y : ℝ),
  DE = 6 → EF = 8 → DF = 10 →
  -- Assuming isosceles right triangle (interchange sides for the proof)
  ∃ (G H I J : ℝ),
  (G = 0 ∧ H = 0 ∧ I = y ∧ J = x ∧ x * y = GH * GI) → -- Rectangle GH parallel to DE
  (x = 10 / 8 * y) →
  ∃ (GH GI : ℝ), 
  GH = 8 / 8.33 ∧ GI = 6.67 / 8.33 →
  (x = 25 / 3 ∧ y = 40 / 6) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_in_right_triangle_dimensions_l1662_166288


namespace NUMINAMATH_GPT_mb_range_l1662_166276

theorem mb_range (m b : ℝ) (hm : m = 3 / 4) (hb : b = -2 / 3) :
  -1 < m * b ∧ m * b < 0 :=
by
  rw [hm, hb]
  sorry

end NUMINAMATH_GPT_mb_range_l1662_166276


namespace NUMINAMATH_GPT_sum_due_is_correct_l1662_166286

theorem sum_due_is_correct (BD TD PV : ℝ) (h1 : BD = 80) (h2 : TD = 70) (h_relation : BD = TD + (TD^2) / PV) : PV = 490 :=
by sorry

end NUMINAMATH_GPT_sum_due_is_correct_l1662_166286


namespace NUMINAMATH_GPT_swimming_pool_paint_area_l1662_166242

theorem swimming_pool_paint_area :
  let length := 20 -- The pool is 20 meters long
  let width := 12  -- The pool is 12 meters wide
  let depth := 2   -- The pool is 2 meters deep
  let area_longer_walls := 2 * length * depth
  let area_shorter_walls := 2 * width * depth
  let total_side_wall_area := area_longer_walls + area_shorter_walls
  let floor_area := length * width
  let total_area_to_paint := total_side_wall_area + floor_area
  total_area_to_paint = 368 :=
by
  sorry

end NUMINAMATH_GPT_swimming_pool_paint_area_l1662_166242


namespace NUMINAMATH_GPT_given_even_function_and_monotonic_increasing_l1662_166284

-- Define f as an even function on ℝ
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

-- Define that f is monotonically increasing on (-∞, 0)
def is_monotonically_increasing_on_negatives (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < 0 → f (x) < f (y)

-- Theorem statement
theorem given_even_function_and_monotonic_increasing {
  f : ℝ → ℝ
} (h_even : is_even_function f)
  (h_monotonic : is_monotonically_increasing_on_negatives f) :
  f (1) > f (-2) :=
sorry

end NUMINAMATH_GPT_given_even_function_and_monotonic_increasing_l1662_166284


namespace NUMINAMATH_GPT_total_weight_correct_l1662_166215

-- Definitions of the given weights of materials
def weight_concrete : ℝ := 0.17
def weight_bricks : ℝ := 0.237
def weight_sand : ℝ := 0.646
def weight_stone : ℝ := 0.5
def weight_steel : ℝ := 1.73
def weight_wood : ℝ := 0.894

-- Total weight of all materials
def total_weight : ℝ := 
  weight_concrete + weight_bricks + weight_sand + weight_stone + weight_steel + weight_wood

-- The proof statement
theorem total_weight_correct : total_weight = 4.177 := by
  sorry

end NUMINAMATH_GPT_total_weight_correct_l1662_166215


namespace NUMINAMATH_GPT_mineral_age_possibilities_l1662_166289

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def count_permutations_with_repeats (n : ℕ) (repeats : List ℕ) : ℕ :=
  factorial n / List.foldl (· * factorial ·) 1 repeats

theorem mineral_age_possibilities : 
  let digits := [2, 2, 4, 4, 7, 9]
  let odd_digits := [7, 9]
  let remaining_digits := [2, 2, 4, 4]
  2 * count_permutations_with_repeats 5 [2,2] = 60 :=
by
  sorry

end NUMINAMATH_GPT_mineral_age_possibilities_l1662_166289


namespace NUMINAMATH_GPT_ratio_spaghetti_to_fettuccine_l1662_166270

def spg : Nat := 300
def fet : Nat := 80

theorem ratio_spaghetti_to_fettuccine : spg / gcd spg fet = 300 / 20 ∧ fet / gcd spg fet = 80 / 20 ∧ (spg / gcd spg fet) / (fet / gcd spg fet) = 15 / 4 := by
  sorry

end NUMINAMATH_GPT_ratio_spaghetti_to_fettuccine_l1662_166270


namespace NUMINAMATH_GPT_maggie_sold_2_subscriptions_to_neighbor_l1662_166204

-- Definition of the problem conditions
def maggie_pays_per_subscription : Int := 5
def maggie_subscriptions_to_parents : Int := 4
def maggie_subscriptions_to_grandfather : Int := 1
def maggie_earned_total : Int := 55

-- Define the function to be proven
def subscriptions_sold_to_neighbor (x : Int) : Prop :=
  maggie_pays_per_subscription * (maggie_subscriptions_to_parents + maggie_subscriptions_to_grandfather + x + 2*x) = maggie_earned_total

-- The statement we need to prove
theorem maggie_sold_2_subscriptions_to_neighbor :
  subscriptions_sold_to_neighbor 2 :=
sorry

end NUMINAMATH_GPT_maggie_sold_2_subscriptions_to_neighbor_l1662_166204


namespace NUMINAMATH_GPT_tangent_line_eq_l1662_166292

theorem tangent_line_eq
    (f : ℝ → ℝ) (f_def : ∀ x, f x = x ^ 2)
    (tangent_point : ℝ × ℝ) (tangent_point_def : tangent_point = (1, 1))
    (f' : ℝ → ℝ) (f'_def : ∀ x, f' x = 2 * x)
    (slope_at_1 : f' 1 = 2) :
    ∃ (a b : ℝ), a = 2 ∧ b = -1 ∧ ∀ x y, y = a * x + b ↔ (2 * x - y - 1 = 0) :=
sorry

end NUMINAMATH_GPT_tangent_line_eq_l1662_166292


namespace NUMINAMATH_GPT_first_player_wins_l1662_166208

-- Define the set of points S
def S : Set (ℤ × ℤ) := { p | ∃ x y : ℤ, p = (x, y) ∧ x^2 + y^2 ≤ 1010 }

-- Define the game properties and conditions
def game_property :=
  ∀ (p : ℤ × ℤ), p ∈ S →
  ∀ (q : ℤ × ℤ), q ∈ S →
  p ≠ q →
  -- Forbidden to move to a point symmetric to the current one relative to the origin
  q ≠ (-p.fst, -p.snd) →
  -- Distances of moves must strictly increase
  dist p q > dist q (q.fst, q.snd)

-- The first player always guarantees a win
theorem first_player_wins : game_property → true :=
by
  sorry

end NUMINAMATH_GPT_first_player_wins_l1662_166208


namespace NUMINAMATH_GPT_find_ice_cream_cost_l1662_166250

def chapatis_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def rice_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def mixed_vegetable_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def soup_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def dessert_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def soft_drink_cost (num: ℕ) (price: ℝ) (discount: ℝ) : ℝ := num * price * (1 - discount)
def total_cost (chap: ℝ) (rice: ℝ) (veg: ℝ) (soup: ℝ) (dessert: ℝ) (drink: ℝ) : ℝ := chap + rice + veg + soup + dessert + drink
def total_cost_with_tax (base_cost: ℝ) (tax_rate: ℝ) : ℝ := base_cost * (1 + tax_rate)

theorem find_ice_cream_cost :
  let chapatis := chapatis_cost 16 6
  let rice := rice_cost 5 45
  let veg := mixed_vegetable_cost 7 70
  let soup := soup_cost 4 30
  let dessert := dessert_cost 3 85
  let drinks := soft_drink_cost 2 50 0.1
  let base_cost := total_cost chapatis rice veg soup dessert drinks
  let final_cost := total_cost_with_tax base_cost 0.18
  final_cost + 6 * 108.89 = 2159 := 
  by sorry

end NUMINAMATH_GPT_find_ice_cream_cost_l1662_166250


namespace NUMINAMATH_GPT_trisha_spending_l1662_166277

theorem trisha_spending :
  let initial_amount := 167
  let spent_meat := 17
  let spent_veggies := 43
  let spent_eggs := 5
  let spent_dog_food := 45
  let remaining_amount := 35
  let total_spent := initial_amount - remaining_amount
  let other_spending := spent_meat + spent_veggies + spent_eggs + spent_dog_food
  total_spent - other_spending = 22 :=
by
  let initial_amount := 167
  let spent_meat := 17
  let spent_veggies := 43
  let spent_eggs := 5
  let spent_dog_food := 45
  let remaining_amount := 35
  -- Calculate total spent
  let total_spent := initial_amount - remaining_amount
  -- Calculate spending on other items
  let other_spending := spent_meat + spent_veggies + spent_eggs + spent_dog_food
  -- Statement to prove
  show total_spent - other_spending = 22
  sorry

end NUMINAMATH_GPT_trisha_spending_l1662_166277


namespace NUMINAMATH_GPT_total_eggs_found_l1662_166225

def eggs_from_club_house : ℕ := 40
def eggs_from_park : ℕ := 25
def eggs_from_town_hall : ℕ := 15

theorem total_eggs_found : eggs_from_club_house + eggs_from_park + eggs_from_town_hall = 80 := by
  -- Proof of this theorem
  sorry

end NUMINAMATH_GPT_total_eggs_found_l1662_166225


namespace NUMINAMATH_GPT_max_red_socks_l1662_166244

theorem max_red_socks (x y : ℕ) 
  (h1 : x + y ≤ 2017) 
  (h2 : (x * (x - 1) + y * (y - 1)) = (x + y) * (x + y - 1) / 2) : 
  x ≤ 990 := 
sorry

end NUMINAMATH_GPT_max_red_socks_l1662_166244


namespace NUMINAMATH_GPT_sum_and_gap_l1662_166236

-- Define the gap condition
def gap_condition (x : ℝ) : Prop :=
  |5.46 - x| = 3.97

-- Define the main theorem to be proved 
theorem sum_and_gap :
  ∀ (x : ℝ), gap_condition x → x < 5.46 → x + 5.46 = 6.95 := 
by 
  intros x hx hlt
  sorry

end NUMINAMATH_GPT_sum_and_gap_l1662_166236


namespace NUMINAMATH_GPT_count_of_sequence_l1662_166202

theorem count_of_sequence : 
  let a := 156
  let d := -6
  let final_term := 36
  (∃ n, a + (n - 1) * d = final_term) -> n = 21 := 
by
  sorry

end NUMINAMATH_GPT_count_of_sequence_l1662_166202


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1662_166212

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = 1 / 4) :
  a 1 * a 2 + a 2 * a 3 + a 3 * a 4 + a 4 * a 5 + a 5 * a 6 = 341 / 32 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1662_166212


namespace NUMINAMATH_GPT_tangent_line_eq_f_positive_find_a_l1662_166229

noncomputable def f (x a : ℝ) : ℝ := 1 - (a * x^2) / (Real.exp x)
noncomputable def f' (x a : ℝ) : ℝ := (a * x * (x - 2)) / (Real.exp x)

-- Part 1: equation of tangent line
theorem tangent_line_eq (a : ℝ) (h1 : f' 1 a = 1) (hx : f 1 a = 2) : ∀ x, f 1 a + f' 1 a * (x - 1) = x + 1 :=
sorry

-- Part 2: f(x) > 0 for x > 0 when a = 1
theorem f_positive (x : ℝ) (h : x > 0) : f x 1 > 0 :=
sorry

-- Part 3: minimum value of f(x) is -3, find a
theorem find_a (a : ℝ) (h : ∀ x, f x a ≥ -3) : a = Real.exp 2 :=
sorry

end NUMINAMATH_GPT_tangent_line_eq_f_positive_find_a_l1662_166229


namespace NUMINAMATH_GPT_find_k_l1662_166211

noncomputable def polynomial1 : Polynomial Int := sorry

theorem find_k :
  ∃ P : Polynomial Int,
  (P.eval 1 = 2013) ∧
  (P.eval 2013 = 1) ∧
  (∃ k : Int, P.eval k = k) →
  ∃ k : Int, P.eval k = k ∧ k = 1007 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1662_166211


namespace NUMINAMATH_GPT_perimeter_of_C_l1662_166295

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end NUMINAMATH_GPT_perimeter_of_C_l1662_166295


namespace NUMINAMATH_GPT_percent_is_50_l1662_166298

variable (cats hogs percent : ℕ)
variable (hogs_eq_3cats : hogs = 3 * cats)
variable (hogs_eq_75 : hogs = 75)

theorem percent_is_50
  (cats_minus_5_percent_eq_10 : (cats - 5) * percent = 1000)
  (cats_eq_25 : cats = 25) :
  percent = 50 := by
  sorry

end NUMINAMATH_GPT_percent_is_50_l1662_166298


namespace NUMINAMATH_GPT_large_box_total_chocolate_bars_l1662_166216

def number_of_small_boxes : ℕ := 15
def chocolate_bars_per_small_box : ℕ := 20
def total_chocolate_bars (n : ℕ) (m : ℕ) : ℕ := n * m

theorem large_box_total_chocolate_bars :
  total_chocolate_bars number_of_small_boxes chocolate_bars_per_small_box = 300 :=
by
  sorry

end NUMINAMATH_GPT_large_box_total_chocolate_bars_l1662_166216


namespace NUMINAMATH_GPT_alex_received_12_cookies_l1662_166273

theorem alex_received_12_cookies :
  ∃ y: ℕ, (∀ s: ℕ, y = s + 8 ∧ s = y / 3) → y = 12 := by
  sorry

end NUMINAMATH_GPT_alex_received_12_cookies_l1662_166273


namespace NUMINAMATH_GPT_cost_per_crayon_l1662_166232

-- Definitions for conditions
def half_dozen := 6
def total_crayons := 4 * half_dozen
def total_cost := 48

-- Problem statement
theorem cost_per_crayon :
  (total_cost / total_crayons) = 2 := 
  by
    sorry

end NUMINAMATH_GPT_cost_per_crayon_l1662_166232


namespace NUMINAMATH_GPT_combined_area_of_removed_triangles_l1662_166218

theorem combined_area_of_removed_triangles (s : ℝ) (x : ℝ) (h : 15 = ((s - 2 * x) ^ 2 + (s - 2 * x) ^ 2) ^ (1/2)) :
  2 * x ^ 2 = 28.125 :=
by
  -- The necessary proof will go here
  sorry

end NUMINAMATH_GPT_combined_area_of_removed_triangles_l1662_166218


namespace NUMINAMATH_GPT_steve_travel_time_l1662_166240

noncomputable def total_travel_time (distance: ℕ) (speed_to_work: ℕ) (speed_back: ℕ) : ℕ :=
  (distance / speed_to_work) + (distance / speed_back)

theorem steve_travel_time : 
  ∀ (distance speed_back speed_to_work : ℕ), 
  (speed_to_work = speed_back / 2) → 
  speed_back = 15 → 
  distance = 30 → 
  total_travel_time distance speed_to_work speed_back = 6 := 
by
  intros
  rw [total_travel_time]
  sorry

end NUMINAMATH_GPT_steve_travel_time_l1662_166240


namespace NUMINAMATH_GPT_tony_lego_sets_l1662_166238

theorem tony_lego_sets
  (price_lego price_sword price_dough : ℕ)
  (num_sword num_dough total_cost : ℕ)
  (L : ℕ)
  (h1 : price_lego = 250)
  (h2 : price_sword = 120)
  (h3 : price_dough = 35)
  (h4 : num_sword = 7)
  (h5 : num_dough = 10)
  (h6 : total_cost = 1940)
  (h7 : total_cost = price_lego * L + price_sword * num_sword + price_dough * num_dough) :
  L = 3 := 
by
  sorry

end NUMINAMATH_GPT_tony_lego_sets_l1662_166238


namespace NUMINAMATH_GPT_derivative_at_2_l1662_166230

noncomputable def f (x : ℝ) : ℝ := x

theorem derivative_at_2 : (deriv f 2) = 1 :=
by
  -- sorry, proof not included
  sorry

end NUMINAMATH_GPT_derivative_at_2_l1662_166230


namespace NUMINAMATH_GPT_number_of_fours_is_even_l1662_166272

theorem number_of_fours_is_even (n3 n4 n5 : ℕ) 
  (h1 : n3 + n4 + n5 = 80)
  (h2 : 3 * n3 + 4 * n4 + 5 * n5 = 276) : Even n4 := 
sorry

end NUMINAMATH_GPT_number_of_fours_is_even_l1662_166272


namespace NUMINAMATH_GPT_train_speed_l1662_166260

def train_length : ℕ := 110
def bridge_length : ℕ := 265
def crossing_time : ℕ := 30

def speed_in_m_per_s (d t : ℕ) : ℕ := d / t
def speed_in_km_per_hr (s : ℕ) : ℕ := s * 36 / 10

theorem train_speed :
  speed_in_km_per_hr (speed_in_m_per_s (train_length + bridge_length) crossing_time) = 45 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1662_166260


namespace NUMINAMATH_GPT_mark_more_than_kate_l1662_166209

variables {K P M : ℕ}

-- Conditions
def total_hours (P K M : ℕ) : Prop := P + K + M = 189
def pat_as_kate (P K : ℕ) : Prop := P = 2 * K
def pat_as_mark (P M : ℕ) : Prop := P = M / 3

-- Statement
theorem mark_more_than_kate (K P M : ℕ) (h1 : total_hours P K M)
  (h2 : pat_as_kate P K) (h3 : pat_as_mark P M) : M - K = 105 :=
by {
  sorry
}

end NUMINAMATH_GPT_mark_more_than_kate_l1662_166209


namespace NUMINAMATH_GPT_area_of_50th_ring_l1662_166201

-- Definitions based on conditions:
def garden_area : ℕ := 9
def ring_area (n : ℕ) : ℕ := 9 * ((2 * n + 1) ^ 2 - (2 * (n - 1) + 1) ^ 2) / 2

-- Theorem to prove:
theorem area_of_50th_ring : ring_area 50 = 1800 := by sorry

end NUMINAMATH_GPT_area_of_50th_ring_l1662_166201


namespace NUMINAMATH_GPT_find_f_neg2_l1662_166253

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2^x + 3*x - 1 else -(2^(-x) + 3*(-x) - 1)

theorem find_f_neg2 : f (-2) = -9 :=
by sorry

end NUMINAMATH_GPT_find_f_neg2_l1662_166253


namespace NUMINAMATH_GPT_exist_integers_not_div_by_7_l1662_166207

theorem exist_integers_not_div_by_7 (k : ℕ) (hk : 0 < k) :
  ∃ (x y : ℤ), (¬ (7 ∣ x)) ∧ (¬ (7 ∣ y)) ∧ (x^2 + 6 * y^2 = 7^k) :=
sorry

end NUMINAMATH_GPT_exist_integers_not_div_by_7_l1662_166207


namespace NUMINAMATH_GPT_temperature_representation_l1662_166257

theorem temperature_representation (a : ℤ) (b : ℤ) (h1 : a = 8) (h2 : b = -5) :
    b < 0 → b = -5 :=
by
  sorry

end NUMINAMATH_GPT_temperature_representation_l1662_166257


namespace NUMINAMATH_GPT_complete_the_square_l1662_166274

theorem complete_the_square (x : ℝ) : (x^2 - 8*x + 15 = 0) → ((x - 4)^2 = 1) :=
by
  intro h
  have eq1 : x^2 - 8*x + 15 = 0 := h
  sorry

end NUMINAMATH_GPT_complete_the_square_l1662_166274


namespace NUMINAMATH_GPT_kitty_cleaning_weeks_l1662_166203

def time_spent_per_week (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust_furniture: ℕ) : ℕ :=
  pick_up + vacuum + clean_windows + dust_furniture

def total_weeks (total_time: ℕ) (time_per_week: ℕ) : ℕ :=
  total_time / time_per_week

theorem kitty_cleaning_weeks
  (pick_up_time : ℕ := 5)
  (vacuum_time : ℕ := 20)
  (clean_windows_time : ℕ := 15)
  (dust_furniture_time : ℕ := 10)
  (total_cleaning_time : ℕ := 200)
  : total_weeks total_cleaning_time (time_spent_per_week pick_up_time vacuum_time clean_windows_time dust_furniture_time) = 4 :=
by
  sorry

end NUMINAMATH_GPT_kitty_cleaning_weeks_l1662_166203


namespace NUMINAMATH_GPT_sin_double_angle_l1662_166283

variable (θ : ℝ)

-- Given condition: tan(θ) = -3/5
def tan_theta : Prop := Real.tan θ = -3/5

-- Target to prove: sin(2θ) = -15/17
theorem sin_double_angle : tan_theta θ → Real.sin (2*θ) = -15/17 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1662_166283


namespace NUMINAMATH_GPT_union_of_sets_l1662_166256

def A : Set ℝ := {x | x < -1 ∨ x > 3}
def B : Set ℝ := {x | x ≥ 2}

theorem union_of_sets : A ∪ B = {x | x < -1 ∨ x ≥ 2} :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l1662_166256


namespace NUMINAMATH_GPT_ratio_of_pieces_l1662_166206

theorem ratio_of_pieces (total_length : ℝ) (short_piece : ℝ) (total_length_eq : total_length = 70) (short_piece_eq : short_piece = 27.999999999999993) :
  let long_piece := total_length - short_piece
  let ratio := short_piece / long_piece
  ratio = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_pieces_l1662_166206


namespace NUMINAMATH_GPT_not_sophomores_percentage_l1662_166245

theorem not_sophomores_percentage (total_students : ℕ)
    (juniors_percentage : ℚ) (juniors : ℕ)
    (seniors : ℕ) (freshmen sophomores : ℕ)
    (h1 : total_students = 800)
    (h2 : juniors_percentage = 0.22)
    (h3 : juniors = juniors_percentage * total_students)
    (h4 : seniors = 160)
    (h5 : freshmen = sophomores + 48)
    (h6 : freshmen + sophomores + juniors + seniors = total_students) :
    ((total_students - sophomores : ℚ) / total_students) * 100 = 74 := by
  sorry

end NUMINAMATH_GPT_not_sophomores_percentage_l1662_166245


namespace NUMINAMATH_GPT_evaluate_expression_l1662_166237

theorem evaluate_expression :
  (3025^2 : ℝ) / ((305^2 : ℝ) - (295^2 : ℝ)) = 1525.10417 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1662_166237


namespace NUMINAMATH_GPT_induction_inequality_term_added_l1662_166247

theorem induction_inequality_term_added (k : ℕ) (h : k > 0) :
  let termAdded := (1 / (2 * (k + 1) - 1 : ℝ)) + (1 / (2 * (k + 1) : ℝ)) - (1 / (k + 1 : ℝ))
  ∃ h : ℝ, termAdded = h :=
by
  sorry

end NUMINAMATH_GPT_induction_inequality_term_added_l1662_166247


namespace NUMINAMATH_GPT_calc_subtract_l1662_166281

-- Define the repeating decimal
def repeating_decimal := (11 : ℚ) / 9

-- Define the problem statement
theorem calc_subtract : 3 - repeating_decimal = (16 : ℚ) / 9 := by
  sorry

end NUMINAMATH_GPT_calc_subtract_l1662_166281


namespace NUMINAMATH_GPT_kamal_chemistry_marks_l1662_166278

-- Definitions of the marks
def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 72
def biology_marks : ℕ := 82
def average_marks : ℕ := 71
def num_subjects : ℕ := 5

-- Statement to be proved
theorem kamal_chemistry_marks : ∃ (chemistry_marks : ℕ), 
  76 + 60 + 72 + 82 + chemistry_marks = 71 * 5 :=
by
sorry

end NUMINAMATH_GPT_kamal_chemistry_marks_l1662_166278


namespace NUMINAMATH_GPT_days_not_worked_correct_l1662_166265

def total_days : ℕ := 20
def earnings_for_work (days_worked : ℕ) : ℤ := 80 * days_worked
def penalty_for_no_work (days_not_worked : ℕ) : ℤ := -40 * days_not_worked
def final_earnings (days_worked days_not_worked : ℕ) : ℤ := 
  (earnings_for_work days_worked) + (penalty_for_no_work days_not_worked)
def received_amount : ℤ := 880

theorem days_not_worked_correct {y x : ℕ} 
  (h1 : x + y = total_days) 
  (h2 : final_earnings x y = received_amount) :
  y = 6 :=
sorry

end NUMINAMATH_GPT_days_not_worked_correct_l1662_166265


namespace NUMINAMATH_GPT_total_pumpkins_l1662_166264

-- Define the number of pumpkins grown by Sandy and Mike
def pumpkinsSandy : ℕ := 51
def pumpkinsMike : ℕ := 23

-- Prove that their total is 74
theorem total_pumpkins : pumpkinsSandy + pumpkinsMike = 74 := by
  sorry

end NUMINAMATH_GPT_total_pumpkins_l1662_166264


namespace NUMINAMATH_GPT_eggs_in_seven_boxes_l1662_166279

-- define the conditions
def eggs_per_box : Nat := 15
def number_of_boxes : Nat := 7

-- state the main theorem to prove
theorem eggs_in_seven_boxes : eggs_per_box * number_of_boxes = 105 := by
  sorry

end NUMINAMATH_GPT_eggs_in_seven_boxes_l1662_166279


namespace NUMINAMATH_GPT_function_value_l1662_166214

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem function_value (a b : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log_base a (2 + b) = 1) (h₃ : log_base a (8 + b) = 2) : a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_function_value_l1662_166214


namespace NUMINAMATH_GPT_hexagonal_prism_min_cut_l1662_166205

-- We formulate the problem conditions and the desired proof
def minimum_edges_to_cut (total_edges : ℕ) (uncut_edges : ℕ) : ℕ :=
  total_edges - uncut_edges

theorem hexagonal_prism_min_cut :
  minimum_edges_to_cut 18 7 = 11 :=
by
  sorry

end NUMINAMATH_GPT_hexagonal_prism_min_cut_l1662_166205


namespace NUMINAMATH_GPT_hair_cut_second_day_l1662_166248

variable (hair_first_day : ℝ) (total_hair_cut : ℝ)

theorem hair_cut_second_day (h1 : hair_first_day = 0.375) (h2 : total_hair_cut = 0.875) :
  total_hair_cut - hair_first_day = 0.500 :=
by sorry

end NUMINAMATH_GPT_hair_cut_second_day_l1662_166248


namespace NUMINAMATH_GPT_range_of_x_l1662_166252

noncomputable def A (x : ℝ) : ℤ := Int.ceil x

theorem range_of_x (x : ℝ) (h₁ : x > 0) (h₂ : A (2 * x * A x) = 5) : x ∈ Set.Ioc 1 (5 / 4 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_of_x_l1662_166252


namespace NUMINAMATH_GPT_find_n_l1662_166297

theorem find_n (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h1 : ∃ n : ℕ, n - 76 = a^3) (h2 : ∃ n : ℕ, n + 76 = b^3) : ∃ n : ℕ, n = 140 :=
by 
  sorry

end NUMINAMATH_GPT_find_n_l1662_166297


namespace NUMINAMATH_GPT_debt_calculation_correct_l1662_166271

-- Conditions
def initial_debt : ℤ := 40
def repayment : ℤ := initial_debt / 2
def additional_borrowing : ℤ := 10

-- Final Debt Calculation
def remaining_debt : ℤ := initial_debt - repayment
def final_debt : ℤ := remaining_debt + additional_borrowing

-- Proof Statement
theorem debt_calculation_correct : final_debt = 30 := 
by 
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_debt_calculation_correct_l1662_166271


namespace NUMINAMATH_GPT_final_amoeba_is_blue_l1662_166262

theorem final_amoeba_is_blue
  (n1 : ℕ) (n2 : ℕ) (n3 : ℕ)
  (merge : ∀ (a b : ℕ), a ≠ b → ∃ c, a + b - c = a ∧ a + b - c = b ∧ a + b - c = c)
  (initial_counts : n1 = 26 ∧ n2 = 31 ∧ n3 = 16)
  (final_count : ∃ a, a = 1) :
  ∃ color, color = "blue" := sorry

end NUMINAMATH_GPT_final_amoeba_is_blue_l1662_166262


namespace NUMINAMATH_GPT_find_b_l1662_166239

variable (a b : Prod ℝ ℝ)
variable (x y : ℝ)

theorem find_b (h1 : (Prod.fst a + Prod.fst b = 0) ∧
                    (Real.sqrt ((Prod.snd a + Prod.snd b) ^ 2) = 1))
                    (h2 : a = (2, -1)) :
                    b = (-2, 2) ∨ b = (-2, 0) :=
by sorry

end NUMINAMATH_GPT_find_b_l1662_166239


namespace NUMINAMATH_GPT_jogging_time_l1662_166291

theorem jogging_time (distance : ℝ) (speed : ℝ) (h1 : distance = 25) (h2 : speed = 5) : (distance / speed) = 5 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_jogging_time_l1662_166291


namespace NUMINAMATH_GPT_tim_change_l1662_166255

theorem tim_change :
  ∀ (initial_amount : ℕ) (amount_paid : ℕ),
  initial_amount = 50 →
  amount_paid = 45 →
  initial_amount - amount_paid = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tim_change_l1662_166255


namespace NUMINAMATH_GPT_people_per_car_l1662_166235

theorem people_per_car (total_people : ℕ) (total_cars : ℕ) (h_people : total_people = 63) (h_cars : total_cars = 3) : 
  total_people / total_cars = 21 := by
  sorry

end NUMINAMATH_GPT_people_per_car_l1662_166235


namespace NUMINAMATH_GPT_vector_operation_l1662_166258

open Matrix

def u : Matrix (Fin 2) (Fin 1) ℝ := ![![3], ![-6]]
def v : Matrix (Fin 2) (Fin 1) ℝ := ![![1], ![-9]]
def w : Matrix (Fin 2) (Fin 1) ℝ := ![![-1], ![4]]

--\mathbf{u} - 5\mathbf{v} + \mathbf{w} = \begin{pmatrix} = \begin{pmatrix} -3 \\ 43 \end{pmatrix}
theorem vector_operation : u - (5 : ℝ) • v + w = ![![-3], ![43]] :=
by
  sorry

end NUMINAMATH_GPT_vector_operation_l1662_166258


namespace NUMINAMATH_GPT_rooms_in_second_wing_each_hall_l1662_166243

theorem rooms_in_second_wing_each_hall
  (floors_first_wing : ℕ)
  (halls_per_floor_first_wing : ℕ)
  (rooms_per_hall_first_wing : ℕ)
  (floors_second_wing : ℕ)
  (halls_per_floor_second_wing : ℕ)
  (total_rooms : ℕ)
  (h1 : floors_first_wing = 9)
  (h2 : halls_per_floor_first_wing = 6)
  (h3 : rooms_per_hall_first_wing = 32)
  (h4 : floors_second_wing = 7)
  (h5 : halls_per_floor_second_wing = 9)
  (h6 : total_rooms = 4248) :
  (total_rooms - floors_first_wing * halls_per_floor_first_wing * rooms_per_hall_first_wing) / 
  (floors_second_wing * halls_per_floor_second_wing) = 40 :=
  by {
  sorry
}

end NUMINAMATH_GPT_rooms_in_second_wing_each_hall_l1662_166243


namespace NUMINAMATH_GPT_total_number_of_pipes_l1662_166275

theorem total_number_of_pipes (bottom_layer top_layer layers : ℕ) 
  (h_bottom_layer : bottom_layer = 13) 
  (h_top_layer : top_layer = 3) 
  (h_layers : layers = 11) : 
  bottom_layer + top_layer = 16 → 
  (bottom_layer + top_layer) * layers / 2 = 88 := 
by
  intro h_sum
  sorry

end NUMINAMATH_GPT_total_number_of_pipes_l1662_166275


namespace NUMINAMATH_GPT_average_of_k_l1662_166233

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end NUMINAMATH_GPT_average_of_k_l1662_166233


namespace NUMINAMATH_GPT_area_of_quadrilateral_l1662_166268

theorem area_of_quadrilateral (A B C : ℝ) (triangle1 triangle2 triangle3 quadrilateral : ℝ)
  (hA : A = 5) (hB : B = 9) (hC : C = 9)
  (h_sum : quadrilateral = triangle1 + triangle2 + triangle3)
  (h1 : triangle1 = A)
  (h2 : triangle2 = B)
  (h3 : triangle3 = C) :
  quadrilateral = 40 :=
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_l1662_166268


namespace NUMINAMATH_GPT_ratio_of_ian_to_jessica_l1662_166293

/-- 
Rodney has 35 dollars more than Ian. 
Jessica has 100 dollars. 
Jessica has 15 dollars more than Rodney. 
Prove that the ratio of Ian's money to Jessica's money is 1/2.
-/
theorem ratio_of_ian_to_jessica (I R J : ℕ) (h1 : R = I + 35) (h2 : J = 100) (h3 : J = R + 15) :
  I / J = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ian_to_jessica_l1662_166293


namespace NUMINAMATH_GPT_tan_a4_a12_eq_neg_sqrt3_l1662_166259

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

variables {a : ℕ → ℝ} (h_arith : is_arithmetic_sequence a)
          (h_sum : a 1 + a 8 + a 15 = Real.pi)

-- The main statement to prove
theorem tan_a4_a12_eq_neg_sqrt3 : 
  Real.tan (a 4 + a 12) = -Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_a4_a12_eq_neg_sqrt3_l1662_166259


namespace NUMINAMATH_GPT_train_length_proof_l1662_166217

-- Definitions based on the conditions given in the problem
def speed_km_per_hr := 45 -- speed of the train in km/hr
def time_seconds := 60 -- time taken to pass the platform in seconds
def length_platform_m := 390 -- length of the platform in meters

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (speed : ℕ) : ℕ := (speed * 1000) / 3600

-- Calculate the speed in m/s
def speed_m_per_s : ℕ := km_per_hr_to_m_per_s speed_km_per_hr

-- Calculate the total distance covered by the train while passing the platform
def total_distance_m : ℕ := speed_m_per_s * time_seconds

-- Total distance is the sum of the length of the train and the length of the platform
def length_train_m := total_distance_m - length_platform_m

-- The statement to prove the length of the train
theorem train_length_proof : length_train_m = 360 :=
by
  sorry

end NUMINAMATH_GPT_train_length_proof_l1662_166217


namespace NUMINAMATH_GPT_cost_comparison_l1662_166220

def cost_function_A (x : ℕ) : ℕ := 450 * x + 1000
def cost_function_B (x : ℕ) : ℕ := 500 * x

theorem cost_comparison (x : ℕ) : 
  if x = 20 then cost_function_A x = cost_function_B x 
  else if x < 20 then cost_function_A x > cost_function_B x 
  else cost_function_A x < cost_function_B x :=
sorry

end NUMINAMATH_GPT_cost_comparison_l1662_166220


namespace NUMINAMATH_GPT_store_loss_90_l1662_166280

theorem store_loss_90 (x y : ℝ) (h1 : x * (1 + 0.12) = 3080) (h2 : y * (1 - 0.12) = 3080) :
  2 * 3080 - x - y = -90 :=
by
  sorry

end NUMINAMATH_GPT_store_loss_90_l1662_166280


namespace NUMINAMATH_GPT_karen_has_32_quarters_l1662_166249

variable (k : ℕ)  -- the number of quarters Karen has

-- Define the number of quarters Christopher has
def christopher_quarters : ℕ := 64

-- Define the value of a single quarter in dollars
def quarter_value : ℚ := 0.25

-- Define the amount of money Christopher has
def christopher_money : ℚ := christopher_quarters * quarter_value

-- Define the monetary difference between Christopher and Karen
def money_difference : ℚ := 8

-- Define the amount of money Karen has
def karen_money : ℚ := christopher_money - money_difference

-- Define the number of quarters Karen has
def karen_quarters := karen_money / quarter_value

-- The theorem we need to prove
theorem karen_has_32_quarters : k = 32 :=
by
  sorry

end NUMINAMATH_GPT_karen_has_32_quarters_l1662_166249


namespace NUMINAMATH_GPT_product_of_primes_l1662_166290

theorem product_of_primes : (2 * 3 * 11) = 66 := by 
  sorry

end NUMINAMATH_GPT_product_of_primes_l1662_166290
