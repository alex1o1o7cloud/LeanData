import Mathlib

namespace NUMINAMATH_CALUDE_order_of_numbers_l642_64206

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def a : Nat := base_to_decimal [1,1,1,1,1,1] 2
def b : Nat := base_to_decimal [0,1,2] 6
def c : Nat := base_to_decimal [0,0,0,1] 4
def d : Nat := base_to_decimal [0,1,1] 8

theorem order_of_numbers : b > d ∧ d > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l642_64206


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l642_64287

theorem z_in_first_quadrant :
  ∀ z : ℂ, (1 + Complex.I)^2 * z = -1 + Complex.I →
  (z.re > 0 ∧ z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l642_64287


namespace NUMINAMATH_CALUDE_routes_on_grid_l642_64294

/-- The number of routes on a 3x3 grid from top-left to bottom-right -/
def num_routes : ℕ := 20

/-- The size of the grid -/
def grid_size : ℕ := 3

/-- The total number of moves required -/
def total_moves : ℕ := 2 * grid_size

/-- The number of moves in each direction -/
def moves_per_direction : ℕ := grid_size

theorem routes_on_grid : 
  num_routes = Nat.choose total_moves moves_per_direction :=
sorry

end NUMINAMATH_CALUDE_routes_on_grid_l642_64294


namespace NUMINAMATH_CALUDE_beth_and_jan_total_money_l642_64274

def beth_money : ℕ := 70
def jan_money : ℕ := 80

theorem beth_and_jan_total_money :
  (beth_money + 35 = 105) ∧
  (jan_money - 10 = beth_money) →
  beth_money + jan_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_beth_and_jan_total_money_l642_64274


namespace NUMINAMATH_CALUDE_birthday_attendees_l642_64215

theorem birthday_attendees : ∃ (n : ℕ), 
  (12 * (n + 2) = 16 * n) ∧ 
  (n = 6) := by
sorry

end NUMINAMATH_CALUDE_birthday_attendees_l642_64215


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l642_64296

-- Problem 1
theorem problem_1 :
  2⁻¹ + |Real.sqrt 6 - 3| + 2 * Real.sqrt 3 * Real.sin (45 * π / 180) - (-2)^2023 * (1/2)^2023 = 9/2 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a = 3) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4) / (a^2 + 2*a + 1)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l642_64296


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l642_64270

/-- The complex number z = i / (1 + i) is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := Complex.I / (1 + Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l642_64270


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l642_64293

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 20 + 1) +
  1 / (Real.log 4 / Real.log 15 + 1) +
  1 / (Real.log 7 / Real.log 12 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l642_64293


namespace NUMINAMATH_CALUDE_simplify_expression_l642_64259

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (125 : ℝ) ^ (1/2) = 20 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l642_64259


namespace NUMINAMATH_CALUDE_quadratic_roots_unique_l642_64262

theorem quadratic_roots_unique (b c : ℝ) : 
  ({1, 2} : Set ℝ) = {x | x^2 + b*x + c = 0} → b = -3 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_unique_l642_64262


namespace NUMINAMATH_CALUDE_sin_monotone_increasing_interval_l642_64237

/-- The function f(x) = sin(2π/3 - 2x) is monotonically increasing on the interval [7π/12, 13π/12] -/
theorem sin_monotone_increasing_interval :
  let f : ℝ → ℝ := λ x => Real.sin (2 * Real.pi / 3 - 2 * x)
  ∀ x y, 7 * Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ 13 * Real.pi / 12 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_sin_monotone_increasing_interval_l642_64237


namespace NUMINAMATH_CALUDE_product_draw_probabilities_l642_64256

/-- Represents the probability space for drawing products -/
structure ProductDraw where
  total : Nat
  defective : Nat
  nonDefective : Nat
  hTotal : total = defective + nonDefective

/-- The probability of drawing a defective product on the first draw -/
def probFirstDefective (pd : ProductDraw) : Rat :=
  pd.defective / pd.total

/-- The probability of drawing defective products on both draws -/
def probBothDefective (pd : ProductDraw) : Rat :=
  (pd.defective / pd.total) * ((pd.defective - 1) / (pd.total - 1))

/-- The probability of drawing a defective product on the second draw, given the first was defective -/
def probSecondDefectiveGivenFirst (pd : ProductDraw) : Rat :=
  (pd.defective - 1) / (pd.total - 1)

theorem product_draw_probabilities (pd : ProductDraw) 
  (h1 : pd.total = 20) 
  (h2 : pd.defective = 5) 
  (h3 : pd.nonDefective = 15) : 
  probFirstDefective pd = 1/4 ∧ 
  probBothDefective pd = 1/19 ∧ 
  probSecondDefectiveGivenFirst pd = 4/19 := by
  sorry

end NUMINAMATH_CALUDE_product_draw_probabilities_l642_64256


namespace NUMINAMATH_CALUDE_fraction_subtraction_l642_64220

theorem fraction_subtraction : (18 : ℚ) / 42 - 2 / 9 = 13 / 63 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l642_64220


namespace NUMINAMATH_CALUDE_reciprocal_abs_eq_neg_self_l642_64289

theorem reciprocal_abs_eq_neg_self :
  ∃! (a : ℝ), |1 / a| = -a :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_reciprocal_abs_eq_neg_self_l642_64289


namespace NUMINAMATH_CALUDE_inequality_proof_l642_64239

theorem inequality_proof (a b : ℝ) (h : a * b > 0) : b / a + a / b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l642_64239


namespace NUMINAMATH_CALUDE_estimate_fish_population_l642_64288

/-- Estimates the total number of fish in a pond using the capture-recapture method. -/
theorem estimate_fish_population
  (initial_catch : ℕ)
  (second_catch : ℕ)
  (marked_recaught : ℕ)
  (h1 : initial_catch = 100)
  (h2 : second_catch = 200)
  (h3 : marked_recaught = 5) :
  (initial_catch * second_catch) / marked_recaught = 4000 :=
sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l642_64288


namespace NUMINAMATH_CALUDE_root_implies_a_value_l642_64261

theorem root_implies_a_value (a : ℝ) : (1 : ℝ)^2 - 2*(1 : ℝ) + a = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l642_64261


namespace NUMINAMATH_CALUDE_no_arithmetic_progression_40_terms_l642_64222

theorem no_arithmetic_progression_40_terms : ¬ ∃ (a d : ℕ) (f : ℕ → ℕ × ℕ),
  (∀ i : ℕ, i < 40 → ∃ (m n : ℕ), f i = (m, n) ∧ a + i * d = 2^m + 3^n) :=
sorry

end NUMINAMATH_CALUDE_no_arithmetic_progression_40_terms_l642_64222


namespace NUMINAMATH_CALUDE_alpha_values_l642_64282

theorem alpha_values (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^4 - 1) = 5 * Complex.abs (α - 1)) :
  ∃ (x y : ℝ), α = Complex.mk x y ∧ 
    ((x = (1 + 8*Real.sqrt 2/9)/2 ∨ x = (1 - 8*Real.sqrt 2/9)/2) ∧
     y^2 = 9 - ((x + 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_alpha_values_l642_64282


namespace NUMINAMATH_CALUDE_system_solution_l642_64209

theorem system_solution : ∃! (u v : ℝ), 5 * u = -7 - 2 * v ∧ 3 * u = 4 * v - 25 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l642_64209


namespace NUMINAMATH_CALUDE_abs_one_minus_sqrt_three_l642_64258

theorem abs_one_minus_sqrt_three (h : Real.sqrt 3 > 1) :
  |1 - Real.sqrt 3| = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_one_minus_sqrt_three_l642_64258


namespace NUMINAMATH_CALUDE_fifth_island_not_maya_l642_64263

-- Define the types of residents
inductive Resident
| Knight
| Liar

-- Define the possible island names
inductive IslandName
| Maya
| NotMaya

-- Define the statements made by A and B
def statement_A (resident_A resident_B : Resident) (island : IslandName) : Prop :=
  (resident_A = Resident.Liar ∧ resident_B = Resident.Liar) ∧ island = IslandName.Maya

def statement_B (resident_A resident_B : Resident) (island : IslandName) : Prop :=
  (resident_A = Resident.Knight ∨ resident_B = Resident.Knight) ∧ island = IslandName.NotMaya

-- Define the truthfulness of statements based on the resident type
def is_truthful (r : Resident) (s : Prop) : Prop :=
  (r = Resident.Knight ∧ s) ∨ (r = Resident.Liar ∧ ¬s)

-- Theorem statement
theorem fifth_island_not_maya :
  ∀ (resident_A resident_B : Resident) (island : IslandName),
    is_truthful resident_A (statement_A resident_A resident_B island) →
    is_truthful resident_B (statement_B resident_A resident_B island) →
    island = IslandName.NotMaya :=
sorry

end NUMINAMATH_CALUDE_fifth_island_not_maya_l642_64263


namespace NUMINAMATH_CALUDE_gas_price_and_distance_l642_64213

-- Define the problem parameters
def expected_gallons : ℝ := 12
def actual_gallons : ℝ := 10
def price_increase : ℝ := 0.3
def fuel_efficiency : ℝ := 25

-- Define the theorem
theorem gas_price_and_distance :
  ∃ (original_price : ℝ) (new_distance : ℝ),
    -- The total cost remains the same
    expected_gallons * original_price = actual_gallons * (original_price + price_increase) ∧
    -- Calculate the new distance
    new_distance = actual_gallons * fuel_efficiency ∧
    -- The original price is $1.50
    original_price = 1.5 ∧
    -- The new distance is 250 miles
    new_distance = 250 := by
  sorry

end NUMINAMATH_CALUDE_gas_price_and_distance_l642_64213


namespace NUMINAMATH_CALUDE_solve_for_q_l642_64275

theorem solve_for_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p*q = 8) :
  q = 4 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l642_64275


namespace NUMINAMATH_CALUDE_percentage_of_boys_l642_64268

theorem percentage_of_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boy_ratio = 3)
  (h3 : girl_ratio = 4) :
  (boy_ratio : ℚ) / (boy_ratio + girl_ratio) * 100 = 42.86 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boys_l642_64268


namespace NUMINAMATH_CALUDE_percentage_relation_l642_64269

theorem percentage_relation (x y c : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 2.5 * y) (h2 : 2 * y = (c / 100) * x) : c = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l642_64269


namespace NUMINAMATH_CALUDE_smallest_common_multiple_l642_64224

theorem smallest_common_multiple (h : ℕ) (d : ℕ) : 
  (∀ k : ℕ, k > 0 ∧ 10 * k % 15 = 0 → k ≥ 3) ∧ 
  (10 * 3 % 15 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_l642_64224


namespace NUMINAMATH_CALUDE_fencing_problem_l642_64212

theorem fencing_problem (length width : ℝ) : 
  width = 40 →
  length * width = 200 →
  2 * length + width = 50 :=
by sorry

end NUMINAMATH_CALUDE_fencing_problem_l642_64212


namespace NUMINAMATH_CALUDE_not_p_and_not_p_and_q_implies_p_or_q_not_necessarily_true_l642_64200

theorem not_p_and_not_p_and_q_implies_p_or_q_not_necessarily_true
  (h1 : ¬p)
  (h2 : ¬(p ∧ q)) :
  ¬∀ (p q : Prop), p ∨ q :=
by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_p_and_q_implies_p_or_q_not_necessarily_true_l642_64200


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l642_64240

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 3 * Nat.factorial 3 + Nat.factorial 3 = 35904 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l642_64240


namespace NUMINAMATH_CALUDE_p_suff_not_nec_q_l642_64223

-- Define propositions p, q, and r
variable (p q r : Prop)

-- Define the conditions
axiom p_suff_not_nec_r : (p → r) ∧ ¬(r → p)
axiom q_nec_r : r → q

-- Theorem to prove
theorem p_suff_not_nec_q : (p → q) ∧ ¬(q → p) := by
  sorry

end NUMINAMATH_CALUDE_p_suff_not_nec_q_l642_64223


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l642_64266

theorem parabola_hyperbola_tangent (a b p k : ℝ) : 
  a > 0 → 
  b > 0 → 
  p > 0 → 
  (2 * a = 4 * Real.sqrt 2) → 
  (b = p / 2) → 
  (k = p / (4 * Real.sqrt 2)) → 
  (∀ x y : ℝ, y = k * x - 1 → x^2 = 2 * p * y) →
  p = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l642_64266


namespace NUMINAMATH_CALUDE_cubic_root_in_interval_l642_64250

theorem cubic_root_in_interval (a b c : ℝ) 
  (h_roots : ∃ (r₁ r₂ r₃ : ℝ), ∀ x, x^3 + a*x^2 + b*x + c = (x - r₁) * (x - r₂) * (x - r₃))
  (h_sum : -2 ≤ a + b + c ∧ a + b + c ≤ 0) :
  ∃ r, (r^3 + a*r^2 + b*r + c = 0) ∧ (0 ≤ r ∧ r ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_in_interval_l642_64250


namespace NUMINAMATH_CALUDE_abby_and_damon_weight_l642_64292

theorem abby_and_damon_weight (a b c d : ℝ) 
  (h1 : a + b = 260)
  (h2 : b + c = 245)
  (h3 : c + d = 270)
  (h4 : a + c = 220) :
  a + d = 285 := by
  sorry

end NUMINAMATH_CALUDE_abby_and_damon_weight_l642_64292


namespace NUMINAMATH_CALUDE_cost_difference_l642_64267

def rental_initial_cost : ℕ := 20
def rental_monthly_increase : ℕ := 5
def rental_insurance : ℕ := 15
def rental_maintenance : ℕ := 10

def new_car_monthly_payment : ℕ := 30
def new_car_down_payment : ℕ := 1500
def new_car_insurance : ℕ := 20
def new_car_maintenance_first_half : ℕ := 5
def new_car_maintenance_second_half : ℕ := 10

def months : ℕ := 12

def rental_total_cost : ℕ := 
  rental_initial_cost + rental_insurance + rental_maintenance + 
  (rental_initial_cost + rental_monthly_increase + rental_insurance + rental_maintenance) * (months - 1)

def new_car_total_cost : ℕ := 
  new_car_down_payment + 
  (new_car_monthly_payment + new_car_insurance + new_car_maintenance_first_half) * (months / 2) +
  (new_car_monthly_payment + new_car_insurance + new_car_maintenance_second_half) * (months / 2)

theorem cost_difference : new_car_total_cost - rental_total_cost = 1595 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l642_64267


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l642_64254

/-- A quadratic function passing through (-3,0) and (5,0) with maximum value 76 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_minus_three : a * (-3)^2 + b * (-3) + c = 0
  passes_through_five : a * 5^2 + b * 5 + c = 0
  max_value : ∃ x, a * x^2 + b * x + c = 76
  is_max : ∀ x, a * x^2 + b * x + c ≤ 76

/-- The sum of coefficients of the quadratic function is 76 -/
theorem sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = 76 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_coefficients_l642_64254


namespace NUMINAMATH_CALUDE_polygon_sides_l642_64299

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 + 360 = 1800 → n = 10 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l642_64299


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_l642_64244

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -6*x := by sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_l642_64244


namespace NUMINAMATH_CALUDE_change_in_math_preference_l642_64227

theorem change_in_math_preference (initial_yes initial_no final_yes final_no absentee_rate : ℝ) :
  initial_yes = 0.4 →
  initial_no = 0.6 →
  final_yes = 0.8 →
  final_no = 0.2 →
  absentee_rate = 0.1 →
  ∃ (min_change max_change : ℝ),
    min_change ≥ 0 ∧
    max_change ≤ 1 ∧
    max_change - min_change = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_change_in_math_preference_l642_64227


namespace NUMINAMATH_CALUDE_nine_solutions_mod_455_l642_64231

theorem nine_solutions_mod_455 : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 1 ≤ n ∧ n ≤ 455 ∧ n^3 % 455 = 1) ∧ 
    (∀ n, 1 ≤ n ∧ n ≤ 455 ∧ n^3 % 455 = 1 → n ∈ s) ∧ 
    s.card = 9 :=
by sorry

end NUMINAMATH_CALUDE_nine_solutions_mod_455_l642_64231


namespace NUMINAMATH_CALUDE_jane_tom_sum_difference_l642_64214

/-- The sum of numbers from 1 to 50 -/
def janeSum : ℕ := (List.range 50).map (· + 1) |>.sum

/-- Function to replace 3 with 2 in a number -/
def replace3With2 (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

/-- The sum of numbers from 1 to 50 with 3 replaced by 2 -/
def tomSum : ℕ := (List.range 50).map (· + 1) |>.map replace3With2 |>.sum

/-- Theorem stating the difference between Jane's and Tom's sums -/
theorem jane_tom_sum_difference : janeSum - tomSum = 105 := by
  sorry

end NUMINAMATH_CALUDE_jane_tom_sum_difference_l642_64214


namespace NUMINAMATH_CALUDE_power_function_property_l642_64233

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 4 / f 2 = 3) :
  f (1/2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l642_64233


namespace NUMINAMATH_CALUDE_l₃_symmetric_to_l₁_wrt_l₂_l642_64229

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def l₂ (x y : ℝ) : Prop := x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (f g h : ℝ → ℝ → Prop) : Prop :=
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    f x₁ y₁ → h x₂ y₂ → 
    ∃ (x₀ y₀ : ℝ), g x₀ y₀ ∧ 
      x₀ = (x₁ + x₂) / 2 ∧ 
      y₀ = (y₁ + y₂) / 2

-- Theorem statement
theorem l₃_symmetric_to_l₁_wrt_l₂ : symmetric_wrt l₁ l₂ l₃ := by
  sorry

end NUMINAMATH_CALUDE_l₃_symmetric_to_l₁_wrt_l₂_l642_64229


namespace NUMINAMATH_CALUDE_sqrt_4_squared_times_5_to_6th_l642_64208

theorem sqrt_4_squared_times_5_to_6th : Real.sqrt (4^2 * 5^6) = 500 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_4_squared_times_5_to_6th_l642_64208


namespace NUMINAMATH_CALUDE_multiple_exists_l642_64264

theorem multiple_exists (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h : ∀ i, 0 < a i ∧ a i ≤ 2*n) : 
  ∃ i j, i ≠ j ∧ (a i ∣ a j ∨ a j ∣ a i) := by
  sorry

end NUMINAMATH_CALUDE_multiple_exists_l642_64264


namespace NUMINAMATH_CALUDE_nicky_running_time_l642_64278

/-- Proves that Nicky runs for 60 seconds before Cristina catches up to him in a race with given conditions -/
theorem nicky_running_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 1500)
  (h2 : head_start = 25)
  (h3 : cristina_speed = 6)
  (h4 : nicky_speed = 3.5) :
  ∃ (t : ℝ), t = 60 ∧ cristina_speed * (t - head_start) = nicky_speed * t :=
by sorry

end NUMINAMATH_CALUDE_nicky_running_time_l642_64278


namespace NUMINAMATH_CALUDE_all_transformations_correct_l642_64211

-- Define the transformations
def transformation_A (a b : ℝ) : Prop := a = b → a + 5 = b + 5

def transformation_B (x y a : ℝ) : Prop := x = y → x / a = y / a

def transformation_C (m n : ℝ) : Prop := m = n → 1 - 3 * m = 1 - 3 * n

def transformation_D (x y c : ℝ) : Prop := x = y → x * c = y * c

-- Theorem stating all transformations are correct
theorem all_transformations_correct :
  (∀ a b : ℝ, transformation_A a b) ∧
  (∀ x y a : ℝ, a ≠ 0 → transformation_B x y a) ∧
  (∀ m n : ℝ, transformation_C m n) ∧
  (∀ x y c : ℝ, transformation_D x y c) :=
sorry

end NUMINAMATH_CALUDE_all_transformations_correct_l642_64211


namespace NUMINAMATH_CALUDE_equation_solution_l642_64201

theorem equation_solution : 
  ∃! x : ℝ, (x / (x + 2) + 3 / (x + 2) + 2 * x / (x + 2) = 4) ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l642_64201


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l642_64228

/-- Given a hyperbola xy = 2 and three points on this curve that also lie on a circle,
    prove that the fourth intersection point has specific coordinates. -/
theorem fourth_intersection_point (P₁ P₂ P₃ P₄ : ℝ × ℝ) : 
  P₁.1 * P₁.2 = 2 ∧ P₂.1 * P₂.2 = 2 ∧ P₃.1 * P₃.2 = 2 ∧ P₄.1 * P₄.2 = 2 →
  P₁ = (3, 2/3) ∧ P₂ = (-4, -1/2) ∧ P₃ = (1/2, 4) →
  ∃ (a b r : ℝ), 
    (P₁.1 - a)^2 + (P₁.2 - b)^2 = r^2 ∧
    (P₂.1 - a)^2 + (P₂.2 - b)^2 = r^2 ∧
    (P₃.1 - a)^2 + (P₃.2 - b)^2 = r^2 ∧
    (P₄.1 - a)^2 + (P₄.2 - b)^2 = r^2 →
  P₄ = (-2/3, -3) := by
sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l642_64228


namespace NUMINAMATH_CALUDE_expected_winning_percentage_approx_l642_64232

/-- Represents the political parties --/
inductive Party
  | Republican
  | Democrat
  | Independent

/-- Represents a candidate in the election --/
inductive Candidate
  | X
  | Y

/-- The ratio of registered voters for each party --/
def partyRatio : Party → ℚ
  | Party.Republican => 3
  | Party.Democrat => 2
  | Party.Independent => 1

/-- The percentage of voters from each party expected to vote for Candidate X --/
def votePercentageForX : Party → ℚ
  | Party.Republican => 85 / 100
  | Party.Democrat => 60 / 100
  | Party.Independent => 40 / 100

/-- The total number of registered voters (assumed to be 6n for some positive integer n) --/
def totalVoters : ℚ := 6

/-- Calculate the expected winning percentage for Candidate X --/
def expectedWinningPercentage : ℚ :=
  let votesForX := (partyRatio Party.Republican * votePercentageForX Party.Republican +
                    partyRatio Party.Democrat * votePercentageForX Party.Democrat +
                    partyRatio Party.Independent * votePercentageForX Party.Independent)
  let votesForY := totalVoters - votesForX
  (votesForX - votesForY) / totalVoters * 100

/-- Theorem stating that the expected winning percentage for Candidate X is approximately 38.33% --/
theorem expected_winning_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 / 100 ∧ |expectedWinningPercentage - 3833 / 100| < ε :=
sorry

end NUMINAMATH_CALUDE_expected_winning_percentage_approx_l642_64232


namespace NUMINAMATH_CALUDE_scientific_notation_of_7413000000_l642_64286

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_7413000000 :
  toScientificNotation 7413000000 = ScientificNotation.mk 7.413 9 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_7413000000_l642_64286


namespace NUMINAMATH_CALUDE_columns_in_first_arrangement_l642_64243

/-- Given a group of people, prove the number of columns formed when 30 people stand in each column. -/
theorem columns_in_first_arrangement 
  (total_people : ℕ) 
  (people_per_column_second : ℕ) 
  (columns_second : ℕ) 
  (people_per_column_first : ℕ) 
  (h1 : people_per_column_second = 32) 
  (h2 : columns_second = 15) 
  (h3 : people_per_column_first = 30) 
  (h4 : total_people = people_per_column_second * columns_second) :
  total_people / people_per_column_first = 16 :=
by sorry

end NUMINAMATH_CALUDE_columns_in_first_arrangement_l642_64243


namespace NUMINAMATH_CALUDE_complex_equation_solution_l642_64246

theorem complex_equation_solution (z : ℂ) (h : (2 - Complex.I) * z = 5) : z = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l642_64246


namespace NUMINAMATH_CALUDE_parallel_planes_theorem_l642_64290

-- Define the types for our objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (subset : Line → Plane → Prop)
variable (intersect : Line → Line → Point → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_theorem 
  (α β : Plane) (a b : Line) (A : Point) :
  subset a α →
  subset b α →
  intersect a b A →
  parallel_line_plane a β →
  parallel_line_plane b β →
  parallel_plane α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_theorem_l642_64290


namespace NUMINAMATH_CALUDE_class_average_increase_l642_64234

/-- Proves that adding a 50-year-old student to a class of 19 students with an average age of 10 years
    increases the overall average age by 2 years. -/
theorem class_average_increase (n : ℕ) (original_avg : ℝ) (new_student_age : ℝ) :
  n = 19 →
  original_avg = 10 →
  new_student_age = 50 →
  (n * original_avg + new_student_age) / (n + 1) - original_avg = 2 := by
  sorry

#check class_average_increase

end NUMINAMATH_CALUDE_class_average_increase_l642_64234


namespace NUMINAMATH_CALUDE_wendy_walking_distance_l642_64204

theorem wendy_walking_distance
  (ran_distance : ℝ)
  (difference : ℝ)
  (h1 : ran_distance = 19.833333333333332)
  (h2 : difference = 10.666666666666666)
  (h3 : ran_distance = walked_distance + difference) :
  walked_distance = 9.166666666666666 :=
by
  sorry

end NUMINAMATH_CALUDE_wendy_walking_distance_l642_64204


namespace NUMINAMATH_CALUDE_cookies_difference_l642_64245

theorem cookies_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) :
  initial_sweet = 8 →
  initial_salty = 6 →
  eaten_sweet = 20 →
  eaten_salty = 34 →
  eaten_salty - eaten_sweet = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_difference_l642_64245


namespace NUMINAMATH_CALUDE_set_intersection_equiv_a_range_l642_64225

/-- Given real number a, define set A -/
def A (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}

/-- Define set B based on set A -/
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 2 * x + 3}

/-- Define set C based on set A -/
def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = x^2}

/-- Theorem stating the equivalence of the set intersection condition and the range of a -/
theorem set_intersection_equiv_a_range (a : ℝ) :
  (B a ∩ C a = C a) ↔ (a < -2 ∨ (1/2 ≤ a ∧ a ≤ 3)) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equiv_a_range_l642_64225


namespace NUMINAMATH_CALUDE_intersection_point_in_zero_one_l642_64281

-- Define the function f(x) = x^3 - (1/2)^x
noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)^x

-- State the theorem
theorem intersection_point_in_zero_one :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1 ∧ f x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_in_zero_one_l642_64281


namespace NUMINAMATH_CALUDE_jacks_total_yen_l642_64236

/-- Represents the amount of money Jack has in different currencies -/
structure JacksMoney where
  pounds : ℕ
  euros : ℕ
  yen : ℕ

/-- Represents the exchange rates between currencies -/
structure ExchangeRates where
  pounds_per_euro : ℕ
  yen_per_pound : ℕ

/-- Calculates the total amount of yen Jack has -/
def total_yen (money : JacksMoney) (rates : ExchangeRates) : ℕ :=
  money.yen +
  money.pounds * rates.yen_per_pound +
  money.euros * rates.pounds_per_euro * rates.yen_per_pound

/-- Theorem stating that Jack's total amount in yen is 9400 -/
theorem jacks_total_yen :
  let money := JacksMoney.mk 42 11 3000
  let rates := ExchangeRates.mk 2 100
  total_yen money rates = 9400 := by
  sorry

end NUMINAMATH_CALUDE_jacks_total_yen_l642_64236


namespace NUMINAMATH_CALUDE_length_of_24_l642_64226

def length (k : ℕ) : ℕ :=
  (Nat.factors k).length

theorem length_of_24 : length 24 = 4 := by
  sorry

end NUMINAMATH_CALUDE_length_of_24_l642_64226


namespace NUMINAMATH_CALUDE_average_tickets_sold_l642_64260

/-- Proves that the average number of tickets sold per day is 80 -/
theorem average_tickets_sold (total_days : ℕ) (total_worth : ℕ) (ticket_cost : ℕ) :
  total_days = 3 →
  total_worth = 960 →
  ticket_cost = 4 →
  (total_worth / ticket_cost) / total_days = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_tickets_sold_l642_64260


namespace NUMINAMATH_CALUDE_triangle_shift_area_ratio_l642_64291

theorem triangle_shift_area_ratio (L α : ℝ) (h1 : 0 < α) (h2 : α < L) :
  let x := α / L
  (x * (2 * L^2 / 2) = (L^2 / 2 - (L - α)^2 / 2)) → x = (3 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_shift_area_ratio_l642_64291


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l642_64249

theorem solve_exponential_equation :
  ∃ x : ℝ, 5^(3*x) = (125 : ℝ)^(1/3) ∧ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l642_64249


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l642_64205

theorem smallest_solution_of_equation (x : ℚ) :
  (7 * (10 * x^2 + 10 * x + 11) = x * (10 * x - 45)) →
  x ≥ -7/5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l642_64205


namespace NUMINAMATH_CALUDE_storks_joined_l642_64217

theorem storks_joined (initial_birds : ℕ) (initial_storks : ℕ) (final_total : ℕ) : 
  initial_birds = 3 → initial_storks = 4 → final_total = 13 →
  final_total - (initial_birds + initial_storks) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_joined_l642_64217


namespace NUMINAMATH_CALUDE_smallest_number_l642_64216

theorem smallest_number (a b c d : ℝ) (h1 : a = 1) (h2 : b = 0) (h3 : c = -2 * Real.sqrt 2) (h4 : d = -3) :
  min a (min b (min c d)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l642_64216


namespace NUMINAMATH_CALUDE_hawk_percentage_l642_64242

theorem hawk_percentage (total : ℝ) (hawk paddyfield kingfisher other : ℝ) : 
  total > 0 ∧
  hawk ≥ 0 ∧ paddyfield ≥ 0 ∧ kingfisher ≥ 0 ∧ other ≥ 0 ∧
  hawk + paddyfield + kingfisher + other = total ∧
  paddyfield = 0.4 * (total - hawk) ∧
  kingfisher = 0.25 * paddyfield ∧
  other = 0.35 * total →
  hawk = 0.3 * total :=
by sorry

end NUMINAMATH_CALUDE_hawk_percentage_l642_64242


namespace NUMINAMATH_CALUDE_oil_cylinder_capacity_l642_64251

theorem oil_cylinder_capacity : ∀ (C : ℚ),
  (4 / 5 : ℚ) * C - (3 / 4 : ℚ) * C = 4 →
  C = 80 := by
sorry

end NUMINAMATH_CALUDE_oil_cylinder_capacity_l642_64251


namespace NUMINAMATH_CALUDE_consecutive_product_not_25k_plus_1_l642_64295

theorem consecutive_product_not_25k_plus_1 (k n : ℕ) : n * (n + 1) ≠ 25 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_25k_plus_1_l642_64295


namespace NUMINAMATH_CALUDE_correct_match_l642_64257

/-- Represents a philosophical statement --/
structure PhilosophicalStatement :=
  (text : String)
  (interpretation : String)

/-- Checks if a statement represents seizing opportunity for qualitative change --/
def representsQualitativeChange (statement : PhilosophicalStatement) : Prop :=
  statement.interpretation = "Decisively seize the opportunity to promote qualitative change"

/-- Checks if a statement represents forward development --/
def representsForwardDevelopment (statement : PhilosophicalStatement) : Prop :=
  statement.interpretation = "The future is bright"

/-- The four given statements --/
def statement1 : PhilosophicalStatement :=
  { text := "As cold comes and heat goes, the four seasons change"
  , interpretation := "Things are developing" }

def statement2 : PhilosophicalStatement :=
  { text := "Thousands of flowers arranged, just waiting for the first thunder"
  , interpretation := "Decisively seize the opportunity to promote qualitative change" }

def statement3 : PhilosophicalStatement :=
  { text := "Despite the intention to plant flowers, they don't bloom; unintentionally planting willows, they grow into shade"
  , interpretation := "The unity of contradictions" }

def statement4 : PhilosophicalStatement :=
  { text := "There will be times when the strong winds break the waves, and we will sail across the sea with clouds"
  , interpretation := "The future is bright" }

/-- Theorem stating that statements 2 and 4 correctly match the required interpretations --/
theorem correct_match :
  representsQualitativeChange statement2 ∧
  representsForwardDevelopment statement4 :=
by sorry

end NUMINAMATH_CALUDE_correct_match_l642_64257


namespace NUMINAMATH_CALUDE_vasyas_numbers_l642_64297

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x + y = x / y) :
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l642_64297


namespace NUMINAMATH_CALUDE_billy_wins_l642_64202

/-- Represents the swimming times for Billy and Margaret -/
structure SwimmingTimes where
  billy_first_5_laps : ℕ  -- in minutes
  billy_next_3_laps : ℕ  -- in minutes
  billy_next_lap : ℕ     -- in minutes
  billy_final_lap : ℕ    -- in seconds
  margaret_total : ℕ     -- in minutes

/-- Calculates the time difference between Billy and Margaret's finish times -/
def timeDifference (times : SwimmingTimes) : ℕ :=
  let billy_total_seconds := 
    (times.billy_first_5_laps + times.billy_next_3_laps + times.billy_next_lap) * 60 + times.billy_final_lap
  let margaret_total_seconds := times.margaret_total * 60
  margaret_total_seconds - billy_total_seconds

/-- Theorem stating that Billy finishes 30 seconds before Margaret -/
theorem billy_wins (times : SwimmingTimes) 
    (h1 : times.billy_first_5_laps = 2)
    (h2 : times.billy_next_3_laps = 4)
    (h3 : times.billy_next_lap = 1)
    (h4 : times.billy_final_lap = 150)
    (h5 : times.margaret_total = 10) : 
  timeDifference times = 30 := by
  sorry


end NUMINAMATH_CALUDE_billy_wins_l642_64202


namespace NUMINAMATH_CALUDE_dans_car_mpg_l642_64241

/-- Given the cost of gas and the distance a car can travel on a certain amount of gas,
    calculate the miles per gallon of the car. -/
theorem dans_car_mpg (gas_cost : ℝ) (miles : ℝ) (gas_expense : ℝ) (mpg : ℝ) : 
  gas_cost = 4 →
  miles = 464 →
  gas_expense = 58 →
  mpg = miles / (gas_expense / gas_cost) →
  mpg = 32 := by
sorry

end NUMINAMATH_CALUDE_dans_car_mpg_l642_64241


namespace NUMINAMATH_CALUDE_kindergarten_gifts_l642_64221

theorem kindergarten_gifts :
  ∀ (n : ℕ) (total_gifts : ℕ),
    (2 * 4 + (n - 2) * 3 + 11 = total_gifts) →
    (4 * 3 + (n - 4) * 6 + 10 = total_gifts) →
    total_gifts = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_kindergarten_gifts_l642_64221


namespace NUMINAMATH_CALUDE_eight_digit_integers_count_l642_64277

theorem eight_digit_integers_count : 
  (Finset.range 8).card * (10 ^ 7) = 80000000 := by sorry

end NUMINAMATH_CALUDE_eight_digit_integers_count_l642_64277


namespace NUMINAMATH_CALUDE_convex_polygon_25_sides_l642_64279

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3

/-- Number of diagonals in a polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Sum of interior angles in a polygon with n sides (in degrees) -/
def sumInteriorAngles (n : ℕ) : ℕ := (n - 2) * 180

theorem convex_polygon_25_sides :
  let p : ConvexPolygon 25 := ⟨by norm_num⟩
  numDiagonals 25 = 275 ∧ sumInteriorAngles 25 = 4140 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_25_sides_l642_64279


namespace NUMINAMATH_CALUDE_second_neighbor_brought_fewer_hotdog_difference_l642_64284

/-- The number of hotdogs brought by the first neighbor -/
def first_neighbor_hotdogs : ℕ := 75

/-- The total number of hotdogs brought by both neighbors -/
def total_hotdogs : ℕ := 125

/-- The number of hotdogs brought by the second neighbor -/
def second_neighbor_hotdogs : ℕ := total_hotdogs - first_neighbor_hotdogs

/-- The second neighbor brought fewer hotdogs than the first -/
theorem second_neighbor_brought_fewer :
  second_neighbor_hotdogs < first_neighbor_hotdogs := by sorry

/-- The difference in hotdogs between the first and second neighbor is 25 -/
theorem hotdog_difference :
  first_neighbor_hotdogs - second_neighbor_hotdogs = 25 := by sorry

end NUMINAMATH_CALUDE_second_neighbor_brought_fewer_hotdog_difference_l642_64284


namespace NUMINAMATH_CALUDE_complement_of_M_l642_64283

-- Define the universal set U as ℝ
def U := ℝ

-- Define the set M
def M : Set ℝ := {x | Real.log (1 - x) > 0}

-- State the theorem
theorem complement_of_M : 
  (Mᶜ : Set ℝ) = {x | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l642_64283


namespace NUMINAMATH_CALUDE_hostel_expenditure_l642_64203

theorem hostel_expenditure (original_students : ℕ) (new_students : ℕ) (expense_increase : ℚ) (average_decrease : ℚ) 
  (h1 : original_students = 35)
  (h2 : new_students = 7)
  (h3 : expense_increase = 42)
  (h4 : average_decrease = 1) :
  ∃ (original_expenditure : ℚ),
    original_expenditure = original_students * 
      ((expense_increase + (original_students + new_students) * average_decrease) / new_students) := by
  sorry

end NUMINAMATH_CALUDE_hostel_expenditure_l642_64203


namespace NUMINAMATH_CALUDE_inequality_theorem_l642_64248

theorem inequality_theorem (p q r s t u : ℝ) 
  (h1 : p^2 < s^2) (h2 : q^2 < t^2) (h3 : r^2 < u^2) :
  p^2 * q^2 + q^2 * r^2 + r^2 * p^2 < s^2 * t^2 + t^2 * u^2 + u^2 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l642_64248


namespace NUMINAMATH_CALUDE_professor_chair_choices_l642_64271

/-- The number of chairs in a row -/
def num_chairs : ℕ := 13

/-- The number of professors -/
def num_professors : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 9

/-- A function that returns true if a chair position is valid for a professor -/
def is_valid_chair (pos : ℕ) : Prop :=
  1 < pos ∧ pos < num_chairs

/-- A function that returns true if two chair positions are valid for two professors -/
def are_valid_chairs (pos1 pos2 : ℕ) : Prop :=
  is_valid_chair pos1 ∧ is_valid_chair pos2 ∧ pos1 + 1 < pos2

/-- The total number of ways professors can choose their chairs -/
def num_ways : ℕ := 45

/-- Theorem stating that the number of ways professors can choose their chairs is 45 -/
theorem professor_chair_choices :
  (Finset.sum (Finset.range (num_chairs - 3))
    (λ k => num_chairs - (k + 3))) = num_ways :=
sorry

end NUMINAMATH_CALUDE_professor_chair_choices_l642_64271


namespace NUMINAMATH_CALUDE_bryan_shelves_count_l642_64230

/-- The number of mineral samples per shelf -/
def samples_per_shelf : ℕ := 65

/-- The total number of mineral samples -/
def total_samples : ℕ := 455

/-- The number of shelves Bryan has -/
def number_of_shelves : ℕ := total_samples / samples_per_shelf

theorem bryan_shelves_count : number_of_shelves = 7 := by
  sorry

end NUMINAMATH_CALUDE_bryan_shelves_count_l642_64230


namespace NUMINAMATH_CALUDE_prob_sum_odd_is_13_27_l642_64252

/-- Represents an unfair die where even numbers are twice as likely as odd numbers -/
structure UnfairDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℝ
  /-- Probability of rolling an even number -/
  even_prob : ℝ
  /-- The probabilities sum to 1 -/
  prob_sum : odd_prob + even_prob = 1
  /-- Even numbers are twice as likely as odd numbers -/
  even_twice_odd : even_prob = 2 * odd_prob

/-- The probability of rolling a sum of three rolls being odd -/
def prob_sum_odd (d : UnfairDie) : ℝ :=
  3 * d.odd_prob * d.even_prob^2 + d.odd_prob^3

/-- Theorem stating the probability of rolling a sum of three rolls being odd is 13/27 -/
theorem prob_sum_odd_is_13_27 (d : UnfairDie) : prob_sum_odd d = 13/27 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_odd_is_13_27_l642_64252


namespace NUMINAMATH_CALUDE_budget_allocation_l642_64272

theorem budget_allocation (research_dev : ℝ) (utilities : ℝ) (equipment : ℝ) (supplies : ℝ) 
  (transportation_degrees : ℝ) (total_degrees : ℝ) :
  research_dev = 9 →
  utilities = 5 →
  equipment = 4 →
  supplies = 2 →
  transportation_degrees = 72 →
  total_degrees = 360 →
  let transportation := (transportation_degrees / total_degrees) * 100
  let other_categories := research_dev + utilities + equipment + supplies + transportation
  let salaries := 100 - other_categories
  salaries = 60 := by
sorry

end NUMINAMATH_CALUDE_budget_allocation_l642_64272


namespace NUMINAMATH_CALUDE_henrys_initial_games_henrys_initial_games_is_58_l642_64235

/-- Proves the number of games Henry had at first -/
theorem henrys_initial_games : ℕ → Prop := fun h =>
  let neil_initial := 7
  let henry_to_neil := 6
  let neil_final := neil_initial + henry_to_neil
  let henry_final := h - henry_to_neil
  (henry_final = 4 * neil_final) → h = 58

/-- The theorem holds for 58 -/
theorem henrys_initial_games_is_58 : henrys_initial_games 58 := by
  sorry

end NUMINAMATH_CALUDE_henrys_initial_games_henrys_initial_games_is_58_l642_64235


namespace NUMINAMATH_CALUDE_set_relations_theorem_l642_64253

universe u

theorem set_relations_theorem (U : Type u) (A B : Set U) : 
  (A ∩ B = ∅ → (Set.compl A ∪ Set.compl B) = Set.univ) ∧
  (A ∪ B = Set.univ → (Set.compl A ∩ Set.compl B) = ∅) ∧
  (A ∪ B = ∅ → A = ∅ ∧ B = ∅) := by
  sorry

end NUMINAMATH_CALUDE_set_relations_theorem_l642_64253


namespace NUMINAMATH_CALUDE_cookies_removed_theorem_l642_64238

/-- Calculates the number of cookies removed in 4 days given initial and final cookie counts over a week -/
def cookies_removed_in_four_days (initial_cookies : ℕ) (remaining_cookies : ℕ) : ℕ :=
  let total_removed : ℕ := initial_cookies - remaining_cookies
  let daily_removal : ℕ := total_removed / 7
  4 * daily_removal

/-- Theorem stating that given 70 initial cookies and 28 remaining after a week, 24 cookies are removed in 4 days -/
theorem cookies_removed_theorem :
  cookies_removed_in_four_days 70 28 = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookies_removed_theorem_l642_64238


namespace NUMINAMATH_CALUDE_abs_value_sum_l642_64207

theorem abs_value_sum (a b c : ℝ) : 
  abs a = 1 → abs b = 2 → abs c = 3 → a > b → b > c → a + b - c = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_sum_l642_64207


namespace NUMINAMATH_CALUDE_parabola_intersection_kite_coefficient_sum_l642_64276

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Represents a kite formed by the intersection points of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

/-- The area of a kite -/
def kiteArea (k : Kite) : ℝ := sorry

/-- The sum of the coefficients of the x^2 terms in the two parabolas forming the kite -/
def coefficientSum (k : Kite) : ℝ := k.p1.a + k.p2.a

theorem parabola_intersection_kite_coefficient_sum :
  ∀ k : Kite,
    k.p1 = Parabola.mk k.p1.a (-3) →
    k.p2 = Parabola.mk (-k.p2.a) 5 →
    kiteArea k = 15 →
    ∃ ε > 0, |coefficientSum k - 2.3| < ε :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_kite_coefficient_sum_l642_64276


namespace NUMINAMATH_CALUDE_total_missed_pitches_l642_64219

/-- The number of pitches per token -/
def pitches_per_token : ℕ := 15

/-- The number of tokens Macy used -/
def macy_tokens : ℕ := 11

/-- The number of tokens Piper used -/
def piper_tokens : ℕ := 17

/-- The number of times Macy hit the ball -/
def macy_hits : ℕ := 50

/-- The number of times Piper hit the ball -/
def piper_hits : ℕ := 55

/-- Theorem stating the total number of missed pitches -/
theorem total_missed_pitches :
  (macy_tokens * pitches_per_token - macy_hits) +
  (piper_tokens * pitches_per_token - piper_hits) = 315 := by
  sorry

end NUMINAMATH_CALUDE_total_missed_pitches_l642_64219


namespace NUMINAMATH_CALUDE_functional_equation_solution_l642_64210

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = y + (f x)^2

/-- The main theorem stating that any function satisfying the equation must be either the identity function or its negation -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l642_64210


namespace NUMINAMATH_CALUDE_number_of_kids_l642_64247

theorem number_of_kids (total_money : ℕ) (apple_cost : ℕ) (apples_per_kid : ℕ) : 
  total_money = 360 → 
  apple_cost = 4 → 
  apples_per_kid = 5 → 
  (total_money / apple_cost) / apples_per_kid = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_kids_l642_64247


namespace NUMINAMATH_CALUDE_supermarket_fruit_prices_l642_64218

theorem supermarket_fruit_prices 
  (strawberry_pints : ℕ) 
  (strawberry_sale_revenue : ℕ) 
  (strawberry_revenue_difference : ℕ)
  (blueberry_pints : ℕ) 
  (blueberry_sale_revenue : ℕ) 
  (blueberry_revenue_difference : ℕ)
  (h1 : strawberry_pints = 54)
  (h2 : strawberry_sale_revenue = 216)
  (h3 : strawberry_revenue_difference = 108)
  (h4 : blueberry_pints = 36)
  (h5 : blueberry_sale_revenue = 144)
  (h6 : blueberry_revenue_difference = 72) :
  (strawberry_sale_revenue + strawberry_revenue_difference) / strawberry_pints = 
  (blueberry_sale_revenue + blueberry_revenue_difference) / blueberry_pints :=
by sorry

end NUMINAMATH_CALUDE_supermarket_fruit_prices_l642_64218


namespace NUMINAMATH_CALUDE_perpendicular_to_countless_lines_iff_perpendicular_to_plane_l642_64280

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Defines when a line is perpendicular to a plane -/
def Line.perpendicular_to_plane (l : Line) (a : Plane) : Prop :=
  sorry

/-- Defines when a line is perpendicular to countless lines within a plane -/
def Line.perpendicular_to_countless_lines_in_plane (l : Line) (a : Plane) : Prop :=
  sorry

/-- 
  The statement that a line being perpendicular to countless lines within a plane
  is a necessary and sufficient condition for the line being perpendicular to the plane
-/
theorem perpendicular_to_countless_lines_iff_perpendicular_to_plane
  (l : Line) (a : Plane) :
  Line.perpendicular_to_countless_lines_in_plane l a ↔ Line.perpendicular_to_plane l a :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_countless_lines_iff_perpendicular_to_plane_l642_64280


namespace NUMINAMATH_CALUDE_ratio_and_mean_determine_a_l642_64255

theorem ratio_and_mean_determine_a (a b c : ℕ+) : 
  (a : ℚ) / b = 2 / 3 →
  (a : ℚ) / c = 2 / 4 →
  (b : ℚ) / c = 3 / 4 →
  (a + b + c : ℚ) / 3 = 42 →
  a = 28 := by
sorry

end NUMINAMATH_CALUDE_ratio_and_mean_determine_a_l642_64255


namespace NUMINAMATH_CALUDE_dice_sides_for_given_probability_l642_64298

theorem dice_sides_for_given_probability (n : ℕ+) : 
  (((6 : ℝ) / (n : ℝ)^2)^2 = 0.027777777777777776) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_dice_sides_for_given_probability_l642_64298


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_specific_quadratic_roots_difference_l642_64265

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * (x^2) + b * x + c = 0 → |r₁ - r₂| = Real.sqrt ((b^2 - 4*a*c) / (a^2)) :=
by sorry

theorem specific_quadratic_roots_difference :
  let r₁ := (-(-7) + Real.sqrt ((-7)^2 - 4*1*12)) / (2*1)
  let r₂ := (-(-7) - Real.sqrt ((-7)^2 - 4*1*12)) / (2*1)
  |r₁ - r₂| = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_specific_quadratic_roots_difference_l642_64265


namespace NUMINAMATH_CALUDE_mike_video_game_days_l642_64273

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of hours Mike watches TV per day -/
def tv_hours_per_day : ℕ := 4

/-- The total hours Mike spends on TV and video games in a week -/
def total_hours_per_week : ℕ := 34

/-- The number of days Mike plays video games in a week -/
def video_game_days : ℕ := 3

theorem mike_video_game_days :
  ∃ (video_game_hours_per_day : ℕ),
    video_game_hours_per_day = tv_hours_per_day / 2 ∧
    video_game_days * video_game_hours_per_day =
      total_hours_per_week - (days_in_week * tv_hours_per_day) :=
by
  sorry

end NUMINAMATH_CALUDE_mike_video_game_days_l642_64273


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l642_64285

theorem smallest_whole_number_above_sum : 
  ⌈(3 + 1/7 : ℚ) + (4 + 1/8 : ℚ) + (5 + 1/9 : ℚ) + (6 + 1/10 : ℚ)⌉ = 19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l642_64285
