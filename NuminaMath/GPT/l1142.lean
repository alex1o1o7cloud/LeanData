import Mathlib

namespace NUMINAMATH_GPT_islands_not_connected_by_bridges_for_infinitely_many_primes_l1142_114265

open Nat

theorem islands_not_connected_by_bridges_for_infinitely_many_primes :
  ∃ᶠ p in at_top, ∃ n m : ℕ, n ≠ m ∧ ¬(p ∣ (n^2 - m + 1) * (m^2 - n + 1)) :=
sorry

end NUMINAMATH_GPT_islands_not_connected_by_bridges_for_infinitely_many_primes_l1142_114265


namespace NUMINAMATH_GPT_max_modulus_l1142_114209

open Complex

theorem max_modulus (z : ℂ) (h : abs z = 1) : ∃ M, M = 6 ∧ ∀ w, abs (z - w) ≤ M :=
by
  use 6
  sorry

end NUMINAMATH_GPT_max_modulus_l1142_114209


namespace NUMINAMATH_GPT_terrell_weight_lifting_l1142_114231

theorem terrell_weight_lifting (n : ℝ) : 
  (2 * 25 * 10 = 500) → (2 * 20 * n = 500) → n = 12.5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_terrell_weight_lifting_l1142_114231


namespace NUMINAMATH_GPT_extremum_of_f_unique_solution_of_equation_l1142_114283

noncomputable def f (x m : ℝ) : ℝ := (1/2) * x^2 - m * Real.log x

theorem extremum_of_f (m : ℝ) (h_pos : 0 < m) :
  ∃ x_min : ℝ, x_min = Real.sqrt m ∧
  ∀ x : ℝ, 0 < x → f x m ≥ f (Real.sqrt m) m :=
sorry

theorem unique_solution_of_equation (m : ℝ) (h_ge_one : 1 ≤ m) :
  ∃! x : ℝ, 0 < x ∧ f x m = x^2 - (m + 1) * x :=
sorry

#check extremum_of_f -- Ensure it can be checked
#check unique_solution_of_equation -- Ensure it can be checked

end NUMINAMATH_GPT_extremum_of_f_unique_solution_of_equation_l1142_114283


namespace NUMINAMATH_GPT_complement_union_eq_l1142_114213

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3}
def N : Set ℕ := {3, 5}

theorem complement_union_eq :
  (U \ (M ∪ N)) = {2, 4} := by
  sorry

end NUMINAMATH_GPT_complement_union_eq_l1142_114213


namespace NUMINAMATH_GPT_shaded_rectangle_area_l1142_114299

theorem shaded_rectangle_area (side_length : ℝ) (x y : ℝ) 
  (h1 : side_length = 42) 
  (h2 : 4 * x + 2 * y = 168 - 4 * x) 
  (h3 : 2 * (side_length - y) + 2 * x = 168 - 4 * x)
  (h4 : 2 * (2 * x + y) = 168 - 4 * x) 
  (h5 : x = 18) :
  (2 * x) * (4 * x - (side_length - y)) = 540 := 
by
  sorry

end NUMINAMATH_GPT_shaded_rectangle_area_l1142_114299


namespace NUMINAMATH_GPT_average_length_of_ropes_l1142_114207

def length_rope_1 : ℝ := 2
def length_rope_2 : ℝ := 6

theorem average_length_of_ropes :
  (length_rope_1 + length_rope_2) / 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_length_of_ropes_l1142_114207


namespace NUMINAMATH_GPT_parabola_transform_correct_l1142_114234

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := -2 * x^2 + 1

-- Define the transformation of moving the parabola one unit to the right and one unit up
def transformed_parabola (x : ℝ) : ℝ := -2 * (x - 1)^2 + 2

-- The theorem to prove
theorem parabola_transform_correct :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 1) + 1 :=
by
  intros x
  sorry

end NUMINAMATH_GPT_parabola_transform_correct_l1142_114234


namespace NUMINAMATH_GPT_win_sector_area_l1142_114225

/-- Given a circular spinner with a radius of 8 cm and the probability of winning being 3/8,
    prove that the area of the WIN sector is 24π square centimeters. -/
theorem win_sector_area (r : ℝ) (P_win : ℝ) (area_WIN : ℝ) :
  r = 8 → P_win = 3 / 8 → area_WIN = 24 * Real.pi := by
sorry

end NUMINAMATH_GPT_win_sector_area_l1142_114225


namespace NUMINAMATH_GPT_log_base_4_of_8_l1142_114268

noncomputable def log_base_change (b a c : ℝ) : ℝ :=
  Real.log a / Real.log b

theorem log_base_4_of_8 : log_base_change 4 8 10 = 3 / 2 :=
by
  have h1 : Real.log 8 = 3 * Real.log 2 := by
    sorry  -- Use properties of logarithms: 8 = 2^3
  have h2 : Real.log 4 = 2 * Real.log 2 := by
    sorry  -- Use properties of logarithms: 4 = 2^2
  have h3 : log_base_change 4 8 10 = (3 * Real.log 2) / (2 * Real.log 2) := by
    rw [log_base_change, h1, h2]
  have h4 : (3 * Real.log 2) / (2 * Real.log 2) = 3 / 2 := by
    sorry  -- Simplify the fraction
  rw [h3, h4]

end NUMINAMATH_GPT_log_base_4_of_8_l1142_114268


namespace NUMINAMATH_GPT_expression_evaluation_l1142_114246

def e : Int := -(-1) + 3^2 / (1 - 4) * 2

theorem expression_evaluation : e = -5 := 
by
  unfold e
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1142_114246


namespace NUMINAMATH_GPT_symmetric_slope_angle_l1142_114212

theorem symmetric_slope_angle (α₁ : ℝ)
  (hα₁ : 0 ≤ α₁ ∧ α₁ < Real.pi) :
  ∃ α₂ : ℝ, (α₁ < Real.pi / 2 → α₂ = Real.pi - α₁) ∧
            (α₁ = Real.pi / 2 → α₂ = 0) :=
sorry

end NUMINAMATH_GPT_symmetric_slope_angle_l1142_114212


namespace NUMINAMATH_GPT_max_product_of_sum_2016_l1142_114256

theorem max_product_of_sum_2016 (x y : ℤ) (h : x + y = 2016) : x * y ≤ 1016064 :=
by
  -- Proof goes here, but is not needed as per instructions
  sorry

end NUMINAMATH_GPT_max_product_of_sum_2016_l1142_114256


namespace NUMINAMATH_GPT_inequality_solution_set_l1142_114235

theorem inequality_solution_set :
  {x : ℝ | (3 * x + 1) / (1 - 2 * x) ≥ 0} = {x : ℝ | -1 / 3 ≤ x ∧ x < 1 / 2} := by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1142_114235


namespace NUMINAMATH_GPT_evaluate_expr_l1142_114224

theorem evaluate_expr (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 5 * x^(y+1) + 6 * y^(x+1) = 2751 :=
by
  rw [h₁, h₂]
  rfl

end NUMINAMATH_GPT_evaluate_expr_l1142_114224


namespace NUMINAMATH_GPT_trapezoid_height_ratios_l1142_114232

theorem trapezoid_height_ratios (A B C D O M N K L : ℝ) (h : ℝ) (h_AD : D = 2 * B) 
  (h_OK : K = h / 3) (h_OL : L = (2 * h) / 3) :
  (K / h = 1 / 3) ∧ (L / h = 2 / 3) := by
  sorry

end NUMINAMATH_GPT_trapezoid_height_ratios_l1142_114232


namespace NUMINAMATH_GPT_evaluate_expression_l1142_114269

theorem evaluate_expression : 
  ((-4 : ℤ) ^ 6) / (4 ^ 4) + (2 ^ 5) * (5 : ℤ) - (7 ^ 2) = 127 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1142_114269


namespace NUMINAMATH_GPT_calculate_r_l1142_114267

def a := 0.24 * 450
def b := 0.62 * 250
def c := 0.37 * 720
def d := 0.38 * 100
def sum_bc := b + c
def diff := sum_bc - a
def r := diff / d

theorem calculate_r : r = 8.25 := by
  sorry

end NUMINAMATH_GPT_calculate_r_l1142_114267


namespace NUMINAMATH_GPT_mural_width_l1142_114201

theorem mural_width (l p r c t w : ℝ) (h₁ : l = 6) (h₂ : p = 4) (h₃ : r = 1.5) (h₄ : c = 10) (h₅ : t = 192) :
  4 * 6 * w + 10 * (6 * w / 1.5) = 192 → w = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mural_width_l1142_114201


namespace NUMINAMATH_GPT_edward_lawns_forgotten_l1142_114208

theorem edward_lawns_forgotten (dollars_per_lawn : ℕ) (total_lawns : ℕ) (total_earned : ℕ) (lawns_mowed : ℕ) (lawns_forgotten : ℕ) :
  dollars_per_lawn = 4 →
  total_lawns = 17 →
  total_earned = 32 →
  lawns_mowed = total_earned / dollars_per_lawn →
  lawns_forgotten = total_lawns - lawns_mowed →
  lawns_forgotten = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_edward_lawns_forgotten_l1142_114208


namespace NUMINAMATH_GPT_egg_production_difference_l1142_114200

def eggs_last_year : ℕ := 1416
def eggs_this_year : ℕ := 4636
def eggs_difference (a b : ℕ) : ℕ := a - b

theorem egg_production_difference : eggs_difference eggs_this_year eggs_last_year = 3220 := 
by
  sorry

end NUMINAMATH_GPT_egg_production_difference_l1142_114200


namespace NUMINAMATH_GPT_no_such_polynomial_exists_l1142_114289

theorem no_such_polynomial_exists :
  ∀ (P : ℤ → ℤ), (∃ a b c d : ℤ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
                  P a = 3 ∧ P b = 3 ∧ P c = 3 ∧ P d = 4) → false :=
by
  sorry

end NUMINAMATH_GPT_no_such_polynomial_exists_l1142_114289


namespace NUMINAMATH_GPT_total_money_9pennies_4nickels_3dimes_l1142_114291

def value_of_pennies (num_pennies : ℕ) : ℝ := num_pennies * 0.01
def value_of_nickels (num_nickels : ℕ) : ℝ := num_nickels * 0.05
def value_of_dimes (num_dimes : ℕ) : ℝ := num_dimes * 0.10

def total_value (pennies nickels dimes : ℕ) : ℝ :=
  value_of_pennies pennies + value_of_nickels nickels + value_of_dimes dimes

theorem total_money_9pennies_4nickels_3dimes :
  total_value 9 4 3 = 0.59 :=
by 
  sorry

end NUMINAMATH_GPT_total_money_9pennies_4nickels_3dimes_l1142_114291


namespace NUMINAMATH_GPT_hyperbola_slope_condition_l1142_114292

-- Define the setup
variables (a b : ℝ) (P F1 F2 : ℝ × ℝ)
variables (h : a > 0) (k : b > 0)
variables (hyperbola : (∀ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1)))

-- Define the condition
variables (cond : ∃ (P : ℝ × ℝ), 3 * abs (dist P F1 + dist P F2) ≤ 2 * dist F1 F2)

-- The proof goal
theorem hyperbola_slope_condition : (b / a) ≥ (Real.sqrt 5 / 2) :=
sorry

end NUMINAMATH_GPT_hyperbola_slope_condition_l1142_114292


namespace NUMINAMATH_GPT_showUpPeopleFirstDay_l1142_114260

def cansFood := 2000
def people1stDay (cansTaken_1stDay : ℕ) := cansFood - 1500 = cansTaken_1stDay
def peopleSnapped_1stDay := 500

theorem showUpPeopleFirstDay :
  (people1stDay peopleSnapped_1stDay) → (peopleSnapped_1stDay / 1) = 500 := 
by 
  sorry

end NUMINAMATH_GPT_showUpPeopleFirstDay_l1142_114260


namespace NUMINAMATH_GPT_total_grains_in_grey_parts_l1142_114287

theorem total_grains_in_grey_parts 
  (total_grains_each_circle : ℕ)
  (white_grains_first_circle : ℕ)
  (white_grains_second_circle : ℕ)
  (common_white_grains : ℕ) 
  (h1 : white_grains_first_circle = 87)
  (h2 : white_grains_second_circle = 110)
  (h3 : common_white_grains = 68) :
  (white_grains_first_circle - common_white_grains) +
  (white_grains_second_circle - common_white_grains) = 61 :=
by
  sorry

end NUMINAMATH_GPT_total_grains_in_grey_parts_l1142_114287


namespace NUMINAMATH_GPT_cookies_milk_conversion_l1142_114298

theorem cookies_milk_conversion :
  (18 : ℕ) / (3 * 2 : ℕ) / (18 : ℕ) * (9 : ℕ) = (3 : ℕ) :=
by
  sorry

end NUMINAMATH_GPT_cookies_milk_conversion_l1142_114298


namespace NUMINAMATH_GPT_quadratic_completion_l1142_114216

theorem quadratic_completion (x : ℝ) : 
  (2 * x^2 + 3 * x - 1) = 2 * (x + 3 / 4)^2 - 17 / 8 := 
by 
  -- Proof isn't required, we just state the theorem.
  sorry

end NUMINAMATH_GPT_quadratic_completion_l1142_114216


namespace NUMINAMATH_GPT_fuel_A_added_l1142_114239

noncomputable def total_tank_capacity : ℝ := 218

noncomputable def ethanol_fraction_A : ℝ := 0.12
noncomputable def ethanol_fraction_B : ℝ := 0.16

noncomputable def total_ethanol : ℝ := 30

theorem fuel_A_added (x : ℝ) 
    (hA : 0 ≤ x) 
    (hA_le_capacity : x ≤ total_tank_capacity) 
    (h_eq : 0.12 * x + 0.16 * (total_tank_capacity - x) = total_ethanol) : 
    x = 122 := 
sorry

end NUMINAMATH_GPT_fuel_A_added_l1142_114239


namespace NUMINAMATH_GPT_B_initial_investment_l1142_114255

theorem B_initial_investment (B : ℝ) :
  let A_initial := 2000
  let A_months := 12
  let A_withdraw := 1000
  let B_advanced := 1000
  let months_before_change := 8
  let months_after_change := 4
  let total_profit := 630
  let A_share := 175
  let B_share := total_profit - A_share
  let A_investment := A_initial * A_months
  let B_investment := (B * months_before_change) + ((B + B_advanced) * months_after_change)
  (B_share / A_share = B_investment / A_investment) →
  B = 4866.67 :=
sorry

end NUMINAMATH_GPT_B_initial_investment_l1142_114255


namespace NUMINAMATH_GPT_find_Y_payment_l1142_114250

theorem find_Y_payment 
  (P X Z : ℝ)
  (total_payment : ℝ)
  (h1 : P + X + Z = total_payment)
  (h2 : X = 1.2 * P)
  (h3 : Z = 0.96 * P) :
  P = 332.28 := by
  sorry

end NUMINAMATH_GPT_find_Y_payment_l1142_114250


namespace NUMINAMATH_GPT_shift_parabola_5_units_right_l1142_114266

def original_parabola (x : ℝ) : ℝ := x^2 + 3
def shifted_parabola (x : ℝ) : ℝ := (x-5)^2 + 3

theorem shift_parabola_5_units_right : ∀ x : ℝ, shifted_parabola x = original_parabola (x - 5) :=
by {
  -- This is the mathematical equivalence that we're proving
  sorry
}

end NUMINAMATH_GPT_shift_parabola_5_units_right_l1142_114266


namespace NUMINAMATH_GPT_inequality_of_f_l1142_114259

def f (x : ℝ) : ℝ := 3 * (x - 2)^2 + 5

theorem inequality_of_f (x₁ x₂ : ℝ) (h : |x₁ - 2| > |x₂ - 2|) : f x₁ > f x₂ :=
by
  -- sorry placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_inequality_of_f_l1142_114259


namespace NUMINAMATH_GPT_projection_is_correct_l1142_114243

theorem projection_is_correct :
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (4, -1)
  let p : ℝ × ℝ := (15/58, 35/58)
  let d : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)
  ∃ v : ℝ × ℝ, 
    (a.1 * v.1 + a.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧
    (b.1 * v.1 + b.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧ 
    (p.1 * d.1 + p.2 * d.2 = 0) :=
sorry

end NUMINAMATH_GPT_projection_is_correct_l1142_114243


namespace NUMINAMATH_GPT_type1_pieces_count_l1142_114257

theorem type1_pieces_count (n : ℕ) (pieces : ℕ → ℕ)  (nonNegative : ∀ i, pieces i ≥ 0) :
  pieces 1 ≥ 4 * n - 1 :=
sorry

end NUMINAMATH_GPT_type1_pieces_count_l1142_114257


namespace NUMINAMATH_GPT_find_ec_l1142_114219

theorem find_ec (angle_A : ℝ) (BC : ℝ) (BD_perp_AC : Prop) (CE_perp_AB : Prop)
  (angle_DBC_2_angle_ECB : Prop) :
  angle_A = 45 ∧ 
  BC = 8 ∧
  BD_perp_AC ∧
  CE_perp_AB ∧
  angle_DBC_2_angle_ECB → 
  ∃ (a b c : ℕ), a = 3 ∧ b = 2 ∧ c = 2 ∧ a + b + c = 7 :=
sorry

end NUMINAMATH_GPT_find_ec_l1142_114219


namespace NUMINAMATH_GPT_number_of_games_l1142_114254

theorem number_of_games (total_points points_per_game : ℕ) (h1 : total_points = 21) (h2 : points_per_game = 7) : total_points / points_per_game = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_games_l1142_114254


namespace NUMINAMATH_GPT_cars_equilibrium_l1142_114281

variable (days : ℕ) -- number of days after which we need the condition to hold
variable (carsA_init carsB_init carsA_to_B carsB_to_A : ℕ) -- initial conditions and parameters

theorem cars_equilibrium :
  let cars_total := 192 + 48
  let carsA := carsA_init + (carsB_to_A - carsA_to_B) * days
  let carsB := carsB_init + (carsA_to_B - carsB_to_A) * days
  carsA_init = 192 -> carsB_init = 48 ->
  carsA_to_B = 21 -> carsB_to_A = 24 ->
  cars_total = 192 + 48 ->
  days = 6 ->
  cars_total = carsA + carsB -> carsA = 7 * carsB :=
by
  intros
  sorry

end NUMINAMATH_GPT_cars_equilibrium_l1142_114281


namespace NUMINAMATH_GPT_parallelogram_area_l1142_114293

theorem parallelogram_area (base height : ℝ) (h_base : base = 22) (h_height : height = 14) :
  base * height = 308 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1142_114293


namespace NUMINAMATH_GPT_nancy_total_money_l1142_114205

def total_money (n_five n_ten n_one : ℕ) : ℕ :=
  (n_five * 5) + (n_ten * 10) + (n_one * 1)

theorem nancy_total_money :
  total_money 9 4 7 = 92 :=
by
  sorry

end NUMINAMATH_GPT_nancy_total_money_l1142_114205


namespace NUMINAMATH_GPT_find_a_plus_b_l1142_114249

theorem find_a_plus_b (a b : ℝ) (h1 : 2 * a = -6) (h2 : a^2 - b = 4) : a + b = 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1142_114249


namespace NUMINAMATH_GPT_alice_walks_distance_l1142_114222

theorem alice_walks_distance :
  let blocks_south := 5
  let blocks_west := 8
  let distance_per_block := 1 / 4
  let total_blocks := blocks_south + blocks_west
  let total_distance := total_blocks * distance_per_block
  total_distance = 3.25 :=
by
  sorry

end NUMINAMATH_GPT_alice_walks_distance_l1142_114222


namespace NUMINAMATH_GPT_quadratic_roots_problem_l1142_114227

theorem quadratic_roots_problem 
  (x y : ℤ) 
  (h1 : x + y = 10)
  (h2 : |x - y| = 12) :
  (x - 11) * (x + 1) = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_problem_l1142_114227


namespace NUMINAMATH_GPT_right_triangle_isosceles_l1142_114251

-- Define the conditions for a right-angled triangle inscribed in a circle
variables (a b : ℝ)

-- Conditions provided in the problem
def right_triangle_inscribed (a b : ℝ) : Prop :=
  ∃ h : a ≠ 0 ∧ b ≠ 0, 2 * (a^2 + b^2) = (a + 2*b)^2 + b^2 ∧ 2 * (a^2 + b^2) = (2 * a + b)^2 + a^2

-- The theorem to prove based on the conditions
theorem right_triangle_isosceles (a b : ℝ) (h : right_triangle_inscribed a b) : a = b :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_isosceles_l1142_114251


namespace NUMINAMATH_GPT_find_largest_integer_l1142_114296

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end NUMINAMATH_GPT_find_largest_integer_l1142_114296


namespace NUMINAMATH_GPT_custom_op_example_l1142_114270

-- Definition of the custom operation
def custom_op (a b : ℕ) : ℕ := 4 * a + 5 * b - a * b

-- The proof statement
theorem custom_op_example : custom_op 7 3 = 22 := by
  sorry

end NUMINAMATH_GPT_custom_op_example_l1142_114270


namespace NUMINAMATH_GPT_interior_angle_of_regular_polygon_l1142_114294

theorem interior_angle_of_regular_polygon (n : ℕ) (h_diagonals : n * (n - 3) / 2 = n) :
    n = 5 ∧ (5 - 2) * 180 / 5 = 108 := by
  sorry

end NUMINAMATH_GPT_interior_angle_of_regular_polygon_l1142_114294


namespace NUMINAMATH_GPT_smallest_k_for_mutual_criticism_l1142_114285

-- Define a predicate that checks if a given configuration of criticisms lead to mutual criticism
def mutual_criticism_exists (deputies : ℕ) (k : ℕ) : Prop :=
  k ≥ 8 -- This is derived from the problem where k = 8 is the smallest k ensuring a mutual criticism

theorem smallest_k_for_mutual_criticism:
  mutual_criticism_exists 15 8 :=
by
  -- This is the theorem statement with the conditions and correct answer. The proof is omitted.
  sorry

end NUMINAMATH_GPT_smallest_k_for_mutual_criticism_l1142_114285


namespace NUMINAMATH_GPT_smallest_prime_divides_polynomial_l1142_114226

theorem smallest_prime_divides_polynomial : 
  ∃ n : ℤ, n^2 + 5 * n + 23 = 17 := 
sorry

end NUMINAMATH_GPT_smallest_prime_divides_polynomial_l1142_114226


namespace NUMINAMATH_GPT_find_first_offset_l1142_114203

theorem find_first_offset {area diagonal offset₁ offset₂ : ℝ}
  (h_area : area = 150)
  (h_diagonal : diagonal = 20)
  (h_offset₂ : offset₂ = 6) :
  2 * area = diagonal * (offset₁ + offset₂) → offset₁ = 9 := by
  sorry

end NUMINAMATH_GPT_find_first_offset_l1142_114203


namespace NUMINAMATH_GPT_Allan_more_balloons_l1142_114280

-- Define the number of balloons that Allan and Jake brought
def Allan_balloons := 5
def Jake_balloons := 3

-- Prove that the number of more balloons that Allan had than Jake is 2
theorem Allan_more_balloons : (Allan_balloons - Jake_balloons) = 2 := by sorry

end NUMINAMATH_GPT_Allan_more_balloons_l1142_114280


namespace NUMINAMATH_GPT_min_value_f_l1142_114271

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + 4 * x + 20) + Real.sqrt (x^2 + 2 * x + 10)

theorem min_value_f : ∃ x : ℝ, f x = 5 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_l1142_114271


namespace NUMINAMATH_GPT_lucy_crayons_l1142_114240

theorem lucy_crayons (W L : ℕ) (h1 : W = 1400) (h2 : W = L + 1110) : L = 290 :=
by {
  sorry
}

end NUMINAMATH_GPT_lucy_crayons_l1142_114240


namespace NUMINAMATH_GPT_arithmetic_sequence_50th_term_l1142_114223

-- Definitions based on the conditions stated
def first_term := 3
def common_difference := 5
def n := 50

-- Function to calculate the n-th term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- The theorem that needs to be proven
theorem arithmetic_sequence_50th_term : nth_term first_term common_difference n = 248 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_50th_term_l1142_114223


namespace NUMINAMATH_GPT_N_10_first_player_wins_N_12_first_player_wins_N_15_second_player_wins_N_30_first_player_wins_l1142_114221

open Nat -- Natural numbers framework

-- Definitions for game conditions would go here. We assume them to be defined as:
-- structure GameCondition (N : ℕ) :=
-- (players_take_turns_to_circle_numbers_from_1_to_N : Prop)
-- (any_two_circled_numbers_must_be_coprime : Prop)
-- (a_number_cannot_be_circled_twice : Prop)
-- (player_who_cannot_move_loses : Prop)

inductive Player
| first
| second

-- Definitions indicating which player wins for a given N
def first_player_wins (N : ℕ) : Prop := sorry
def second_player_wins (N : ℕ) : Prop := sorry

-- For N = 10
theorem N_10_first_player_wins : first_player_wins 10 := sorry

-- For N = 12
theorem N_12_first_player_wins : first_player_wins 12 := sorry

-- For N = 15
theorem N_15_second_player_wins : second_player_wins 15 := sorry

-- For N = 30
theorem N_30_first_player_wins : first_player_wins 30 := sorry

end NUMINAMATH_GPT_N_10_first_player_wins_N_12_first_player_wins_N_15_second_player_wins_N_30_first_player_wins_l1142_114221


namespace NUMINAMATH_GPT_correct_division_result_l1142_114274

theorem correct_division_result {x : ℕ} (h : 3 * x = 90) : x / 3 = 10 :=
by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_correct_division_result_l1142_114274


namespace NUMINAMATH_GPT_tory_needs_to_raise_more_l1142_114286

variable (goal : ℕ) (pricePerChocolateChip pricePerOatmealRaisin pricePerSugarCookie : ℕ)
variable (soldChocolateChip soldOatmealRaisin soldSugarCookie : ℕ)

def remainingAmount (goal : ℕ) 
                    (pricePerChocolateChip pricePerOatmealRaisin pricePerSugarCookie : ℕ)
                    (soldChocolateChip soldOatmealRaisin soldSugarCookie : ℕ) : ℕ :=
  let profitFromChocolateChip := soldChocolateChip * pricePerChocolateChip
  let profitFromOatmealRaisin := soldOatmealRaisin * pricePerOatmealRaisin
  let profitFromSugarCookie := soldSugarCookie * pricePerSugarCookie
  let totalProfit := profitFromChocolateChip + profitFromOatmealRaisin + profitFromSugarCookie
  goal - totalProfit

theorem tory_needs_to_raise_more : 
  remainingAmount 250 6 5 4 5 10 15 = 110 :=
by
  -- Proof omitted 
  sorry

end NUMINAMATH_GPT_tory_needs_to_raise_more_l1142_114286


namespace NUMINAMATH_GPT_polygon_sides_sum_l1142_114264

theorem polygon_sides_sum (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_sum_l1142_114264


namespace NUMINAMATH_GPT_clock_in_probability_l1142_114242

-- Definitions
def start_time := 510 -- 8:30 in minutes from 00:00 (510 minutes)
def valid_clock_in_start := 495 -- 8:15 in minutes from 00:00 (495 minutes)
def arrival_start := 470 -- 7:50 in minutes from 00:00 (470 minutes)
def arrival_end := 510 -- 8:30 in minutes from 00:00 (510 minutes)
def valid_clock_in_end := 510 -- 8:30 in minutes from 00:00 (510 minutes)

-- Conditions
def arrival_window := arrival_end - arrival_start -- 40 minutes window
def valid_clock_in_window := valid_clock_in_end - valid_clock_in_start -- 15 minutes window

-- Required proof statement
theorem clock_in_probability :
  (valid_clock_in_window : ℚ) / (arrival_window : ℚ) = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_clock_in_probability_l1142_114242


namespace NUMINAMATH_GPT_function_sqrt_plus_one_l1142_114261

variable (f : ℝ → ℝ)
variable (x : ℝ)

theorem function_sqrt_plus_one (h1 : ∀ x : ℝ, f x = 3) (h2 : x ≥ 0) : f (Real.sqrt x) + 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_function_sqrt_plus_one_l1142_114261


namespace NUMINAMATH_GPT_solution_inequalities_l1142_114237

theorem solution_inequalities (x : ℝ) :
  (x^2 - 12 * x + 32 > 0) ∧ (x^2 - 13 * x + 22 < 0) → 2 < x ∧ x < 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solution_inequalities_l1142_114237


namespace NUMINAMATH_GPT_ratio_of_voters_l1142_114284

theorem ratio_of_voters (V_X V_Y : ℝ) 
  (h1 : 0.62 * V_X + 0.38 * V_Y = 0.54 * (V_X + V_Y)) : V_X / V_Y = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_voters_l1142_114284


namespace NUMINAMATH_GPT_sum_last_two_digits_of_powers_l1142_114244

theorem sum_last_two_digits_of_powers (h₁ : 9 = 10 - 1) (h₂ : 11 = 10 + 1) :
  (9^20 + 11^20) % 100 / 10 + (9^20 + 11^20) % 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_last_two_digits_of_powers_l1142_114244


namespace NUMINAMATH_GPT_no_int_coords_equilateral_l1142_114295

--- Define a structure for points with integer coordinates
structure Point :=
(x : ℤ)
(y : ℤ)

--- Definition of the distance squared between two points
def dist_squared (P Q : Point) : ℤ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

--- Statement that given three points with integer coordinates, they cannot form an equilateral triangle
theorem no_int_coords_equilateral (A B C : Point) :
  ¬ (dist_squared A B = dist_squared B C ∧ dist_squared B C = dist_squared C A ∧ dist_squared C A = dist_squared A B) :=
sorry

end NUMINAMATH_GPT_no_int_coords_equilateral_l1142_114295


namespace NUMINAMATH_GPT_prism_visibility_percentage_l1142_114252

theorem prism_visibility_percentage
  (base_edge : ℝ)
  (height : ℝ)
  (cell_side : ℝ)
  (wraps : ℕ)
  (lateral_surface_area : ℝ)
  (transparent_area : ℝ) :
  base_edge = 3.2 →
  height = 5 →
  cell_side = 1 →
  wraps = 2 →
  lateral_surface_area = base_edge * height * 3 →
  transparent_area = 13.8 →
  (transparent_area / lateral_surface_area) * 100 = 28.75 :=
by
  intros h_base_edge h_height h_cell_side h_wraps h_lateral_surface_area h_transparent_area
  sorry

end NUMINAMATH_GPT_prism_visibility_percentage_l1142_114252


namespace NUMINAMATH_GPT_snail_climbs_well_l1142_114204

theorem snail_climbs_well (h : ℕ) (c : ℕ) (s : ℕ) (d : ℕ) (h_eq : h = 12) (c_eq : c = 3) (s_eq : s = 2) : d = 10 :=
by
  sorry

end NUMINAMATH_GPT_snail_climbs_well_l1142_114204


namespace NUMINAMATH_GPT_enrique_speed_l1142_114214

theorem enrique_speed (distance : ℝ) (time : ℝ) (speed_diff : ℝ) (E : ℝ) :
  distance = 200 ∧ time = 8 ∧ speed_diff = 7 ∧ 
  (2 * E + speed_diff) * time = distance → 
  E = 9 :=
by
  sorry

end NUMINAMATH_GPT_enrique_speed_l1142_114214


namespace NUMINAMATH_GPT_abs_diff_eq_sqrt_l1142_114263

theorem abs_diff_eq_sqrt (x1 x2 a b : ℝ) (h1 : x1 + x2 = a) (h2 : x1 * x2 = b) : 
  |x1 - x2| = Real.sqrt (a^2 - 4 * b) :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_eq_sqrt_l1142_114263


namespace NUMINAMATH_GPT_fraction_of_satisfactory_grades_is_3_4_l1142_114288

def num_grades (grades : String → ℕ) : ℕ := 
  grades "A" + grades "B" + grades "C" + grades "D" + grades "F"

def satisfactory_grades (grades : String → ℕ) : ℕ := 
  grades "A" + grades "B" + grades "C" + grades "D"

def fraction_satisfactory (grades : String → ℕ) : ℚ := 
  satisfactory_grades grades / num_grades grades

theorem fraction_of_satisfactory_grades_is_3_4 
  (grades : String → ℕ)
  (hA : grades "A" = 5)
  (hB : grades "B" = 4)
  (hC : grades "C" = 3)
  (hD : grades "D" = 3)
  (hF : grades "F" = 5) : 
  fraction_satisfactory grades = (3 : ℚ) / 4 := by
{
  sorry
}

end NUMINAMATH_GPT_fraction_of_satisfactory_grades_is_3_4_l1142_114288


namespace NUMINAMATH_GPT_compound_interest_rate_l1142_114276

theorem compound_interest_rate (SI CI : ℝ) (P1 P2 : ℝ) (T1 T2 : ℝ) (R1 : ℝ) (R : ℝ) 
    (H1 : SI = (P1 * R1 * T1) / 100)
    (H2 : CI = 2 * SI)
    (H3 : CI = P2 * ((1 + R/100)^2 - 1))
    (H4 : P1 = 1272)
    (H5 : P2 = 5000)
    (H6 : T1 = 5)
    (H7 : T2 = 2)
    (H8 : R1 = 10) :
  R = 12 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l1142_114276


namespace NUMINAMATH_GPT_largest_4_digit_divisible_by_12_l1142_114217

theorem largest_4_digit_divisible_by_12 : ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ 12 ∣ n ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ 12 ∣ m → m ≤ n :=
sorry

end NUMINAMATH_GPT_largest_4_digit_divisible_by_12_l1142_114217


namespace NUMINAMATH_GPT_probability_first_two_heads_l1142_114215

-- The probability of getting heads in a single flip of a fair coin
def probability_heads_single_flip : ℚ := 1 / 2

-- Independence of coin flips
def independent_flips {α : Type} (p : α → Prop) := ∀ a b : α, a ≠ b → p a ∧ p b

-- The event of getting heads on a coin flip
def heads_event : Prop := true

-- Problem statement: The probability that the first two flips are both heads
theorem probability_first_two_heads : probability_heads_single_flip * probability_heads_single_flip = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_first_two_heads_l1142_114215


namespace NUMINAMATH_GPT_length_of_bridge_l1142_114248

theorem length_of_bridge (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_to_cross : ℕ)
  (lt : length_of_train = 140)
  (st : speed_of_train_kmh = 45)
  (tc : time_to_cross = 30) : 
  ∃ length_of_bridge, length_of_bridge = 235 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1142_114248


namespace NUMINAMATH_GPT_saree_blue_stripes_l1142_114282

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    brown_stripes = 4 →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_gold h_blue h_brown
  sorry

end NUMINAMATH_GPT_saree_blue_stripes_l1142_114282


namespace NUMINAMATH_GPT_greatest_two_digit_product_12_l1142_114220

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end NUMINAMATH_GPT_greatest_two_digit_product_12_l1142_114220


namespace NUMINAMATH_GPT_solution_set_I_range_of_m_II_l1142_114277

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem solution_set_I : {x : ℝ | 0 ≤ x ∧ x ≤ 3} = {x : ℝ | f x ≤ 3} :=
sorry

theorem range_of_m_II (x : ℝ) (hx : x > 0) : ∃ m : ℝ, ∀ (x : ℝ), f x ≤ m - x - 4 / x → m ≥ 5 :=
sorry

end NUMINAMATH_GPT_solution_set_I_range_of_m_II_l1142_114277


namespace NUMINAMATH_GPT_relationship_t_s_l1142_114236

variable {a b : ℝ}

theorem relationship_t_s (a b : ℝ) (t : ℝ) (s : ℝ) (ht : t = a + 2 * b) (hs : s = a + b^2 + 1) :
  t ≤ s := 
sorry

end NUMINAMATH_GPT_relationship_t_s_l1142_114236


namespace NUMINAMATH_GPT_range_of_k_l1142_114253

theorem range_of_k (x : ℝ) (k : ℝ) (h_pos : x > 0) (h_ne : x ≠ 2) :
  (1 / (x - 2) + 3 = (3 - k) / (2 - x)) ↔ (k > -2 ∧ k ≠ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1142_114253


namespace NUMINAMATH_GPT_sum_of_two_integers_l1142_114297

noncomputable def sum_of_integers (a b : ℕ) : ℕ :=
a + b

theorem sum_of_two_integers (a b : ℕ) (h1 : a - b = 14) (h2 : a * b = 120) : sum_of_integers a b = 26 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_two_integers_l1142_114297


namespace NUMINAMATH_GPT_lee_can_make_36_cookies_l1142_114279

-- Conditions
def initial_cups_of_flour : ℕ := 2
def initial_cookies_made : ℕ := 18
def initial_total_flour : ℕ := 5
def spilled_flour : ℕ := 1

-- Define the remaining cups of flour after spilling
def remaining_flour := initial_total_flour - spilled_flour

-- Define the proportion to solve for the number of cookies made with remaining_flour
def cookies_with_remaining_flour (c : ℕ) : Prop :=
  (initial_cookies_made / initial_cups_of_flour) = (c / remaining_flour)

-- The statement to prove
theorem lee_can_make_36_cookies : cookies_with_remaining_flour 36 :=
  sorry

end NUMINAMATH_GPT_lee_can_make_36_cookies_l1142_114279


namespace NUMINAMATH_GPT_indolent_student_probability_l1142_114262

-- Define the constants of the problem
def n : ℕ := 30  -- total number of students
def k : ℕ := 3   -- number of students selected each lesson
def m : ℕ := 10  -- number of students from the previous lesson

-- Define the probabilities
def P_asked_in_one_lesson : ℚ := 1 / k
def P_asked_twice_in_a_row : ℚ := 1 / n
def P_overall : ℚ := P_asked_in_one_lesson + P_asked_in_one_lesson - P_asked_twice_in_a_row
def P_avoid_reciting : ℚ := 1 - P_overall

theorem indolent_student_probability : P_avoid_reciting = 11 / 30 := 
  sorry

end NUMINAMATH_GPT_indolent_student_probability_l1142_114262


namespace NUMINAMATH_GPT_initial_cats_l1142_114230

theorem initial_cats (C : ℕ) (h1 : 36 + 12 - 20 + C = 57) : C = 29 :=
by
  sorry

end NUMINAMATH_GPT_initial_cats_l1142_114230


namespace NUMINAMATH_GPT_unique_zero_identity_l1142_114210

theorem unique_zero_identity (n : ℤ) : (∀ z : ℤ, z + n = z ∧ z * n = 0) → n = 0 :=
by
  intro h
  have h1 : ∀ z : ℤ, z + n = z := fun z => (h z).left
  have h2 : ∀ z : ℤ, z * n = 0 := fun z => (h z).right
  sorry

end NUMINAMATH_GPT_unique_zero_identity_l1142_114210


namespace NUMINAMATH_GPT_compute_z_pow_7_l1142_114228

namespace ComplexProof

noncomputable def z : ℂ := (Real.sqrt 3 + Complex.I) / 2

theorem compute_z_pow_7 : z ^ 7 = - (Real.sqrt 3 / 2) - (1 / 2) * Complex.I :=
by
  sorry

end ComplexProof

end NUMINAMATH_GPT_compute_z_pow_7_l1142_114228


namespace NUMINAMATH_GPT_parallelogram_theorem_l1142_114247

noncomputable def parallelogram (A B C D O : Type) (θ : ℝ) :=
  let DBA := θ
  let DBC := 3 * θ
  let CAB := 9 * θ
  let ACB := 180 - (9 * θ + 3 * θ)
  let AOB := 180 - 12 * θ
  let s := ACB / AOB
  s = 4 / 5

theorem parallelogram_theorem (A B C D O : Type) (θ : ℝ) 
  (h1: θ > 0): parallelogram A B C D O θ := by
  sorry

end NUMINAMATH_GPT_parallelogram_theorem_l1142_114247


namespace NUMINAMATH_GPT_min_value_d1_d2_l1142_114202

noncomputable def min_distance_sum : ℝ :=
  let d1 (u : ℝ) : ℝ := (1 / 5) * abs (3 * Real.cos u - 4 * Real.sin u - 10)
  let d2 (u : ℝ) : ℝ := 3 - Real.cos u
  let d_sum (u : ℝ) : ℝ := d1 u + d2 u
  ((5 - (4 * Real.sqrt 5 / 5)))

theorem min_value_d1_d2 :
  ∀ (P : ℝ × ℝ) (u : ℝ),
    P = (Real.cos u, Real.sin u) →
    (P.1 ^ 2 + P.2 ^ 2 = 1) →
    let d1 := (1 / 5) * abs (3 * P.1 - 4 * P.2 - 10)
    let d2 := 3 - P.1
    d1 + d2 ≥ (5 - (4 * Real.sqrt 5 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_d1_d2_l1142_114202


namespace NUMINAMATH_GPT_tom_needs_495_boxes_l1142_114233

-- Define the conditions
def total_chocolate_bars : ℕ := 3465
def chocolate_bars_per_box : ℕ := 7

-- Define the proof statement
theorem tom_needs_495_boxes : total_chocolate_bars / chocolate_bars_per_box = 495 :=
by
  sorry

end NUMINAMATH_GPT_tom_needs_495_boxes_l1142_114233


namespace NUMINAMATH_GPT_value_a2_plus_b2_l1142_114258

noncomputable def a_minus_b : ℝ := 8
noncomputable def ab : ℝ := 49.99999999999999

theorem value_a2_plus_b2 (a b : ℝ) (h1 : a - b = a_minus_b) (h2 : a * b = ab) :
  a^2 + b^2 = 164 := by
  sorry

end NUMINAMATH_GPT_value_a2_plus_b2_l1142_114258


namespace NUMINAMATH_GPT_fuel_relationship_l1142_114278

theorem fuel_relationship (y : ℕ → ℕ) (h₀ : y 0 = 80) (h₁ : y 1 = 70) (h₂ : y 2 = 60) (h₃ : y 3 = 50) :
  ∀ x : ℕ, y x = 80 - 10 * x :=
by
  sorry

end NUMINAMATH_GPT_fuel_relationship_l1142_114278


namespace NUMINAMATH_GPT_find_a4_a5_l1142_114238

variable {α : Type*} [LinearOrderedField α]

-- Variables representing the terms of the geometric sequence
variables (a₁ a₂ a₃ a₄ a₅ q : α)

-- Conditions given in the problem
-- Geometric sequence condition
def is_geometric_sequence (a₁ a₂ a₃ a₄ a₅ q : α) : Prop :=
  a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧ a₅ = a₄ * q

-- First condition
def condition1 : Prop := a₁ + a₂ = 3

-- Second condition
def condition2 : Prop := a₂ + a₃ = 6

-- Theorem stating that a₄ + a₅ = 24 given the conditions
theorem find_a4_a5
  (h1 : condition1 a₁ a₂)
  (h2 : condition2 a₂ a₃)
  (hg : is_geometric_sequence a₁ a₂ a₃ a₄ a₅ q) :
  a₄ + a₅ = 24 := 
sorry

end NUMINAMATH_GPT_find_a4_a5_l1142_114238


namespace NUMINAMATH_GPT_numbers_identification_l1142_114206

-- Definitions
def is_natural (n : ℤ) : Prop := n ≥ 0
def is_integer (n : ℤ) : Prop := True

-- Theorem
theorem numbers_identification :
  (is_natural 0 ∧ is_natural 2 ∧ is_natural 6 ∧ is_natural 7) ∧
  (is_integer (-15) ∧ is_integer (-3) ∧ is_integer 0 ∧ is_integer 4) :=
by
  sorry

end NUMINAMATH_GPT_numbers_identification_l1142_114206


namespace NUMINAMATH_GPT_totalNumberOfBalls_l1142_114218

def numberOfBoxes : ℕ := 3
def numberOfBallsPerBox : ℕ := 5

theorem totalNumberOfBalls : numberOfBoxes * numberOfBallsPerBox = 15 := 
by
  sorry

end NUMINAMATH_GPT_totalNumberOfBalls_l1142_114218


namespace NUMINAMATH_GPT_min_value_x4_y3_z2_l1142_114241

theorem min_value_x4_y3_z2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 1/x + 1/y + 1/z = 9) : 
  x^4 * y^3 * z^2 ≥ 1 / 9^9 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_min_value_x4_y3_z2_l1142_114241


namespace NUMINAMATH_GPT_original_population_multiple_of_3_l1142_114272

theorem original_population_multiple_of_3 (x y z : ℕ) (h1 : x^2 + 121 = y^2) (h2 : y^2 + 121 = z^2) :
  3 ∣ x^2 :=
sorry

end NUMINAMATH_GPT_original_population_multiple_of_3_l1142_114272


namespace NUMINAMATH_GPT_find_second_number_l1142_114273

-- Definitions for the conditions
def ratio_condition (x : ℕ) : Prop := 5 * x = 40

-- The theorem we need to prove, i.e., the second number is 8 given the conditions
theorem find_second_number (x : ℕ) (h : ratio_condition x) : x = 8 :=
by sorry

end NUMINAMATH_GPT_find_second_number_l1142_114273


namespace NUMINAMATH_GPT_find_a2_l1142_114245

noncomputable def geometric_sequence (n : ℕ) : ℝ := sorry -- Define the geometric sequence

variable (a1 : ℝ) (a3a5_eq : ℝ) -- Variables for given conditions

-- Main theorem statement
theorem find_a2 (h_geo : ∀ n, geometric_sequence n = a1 * (2 : ℝ) ^ (n - 1))
  (h_a1 : a1 = 1 / 4)
  (h_a3a5 : (geometric_sequence 3) * (geometric_sequence 5) = 4 * (geometric_sequence 4 - 1)) :
  geometric_sequence 2 = 1 / 2 :=
sorry  -- Proof is omitted

end NUMINAMATH_GPT_find_a2_l1142_114245


namespace NUMINAMATH_GPT_problem_l1142_114290

def f (x : ℤ) : ℤ := 3 * x - 1
def g (x : ℤ) : ℤ := 2 * x + 5

theorem problem (h : ℤ) :
  (g (f (g (3))) : ℚ) / f (g (f (3))) = 69 / 206 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1142_114290


namespace NUMINAMATH_GPT_evaluate_expression_l1142_114229

theorem evaluate_expression (x z : ℝ) (h1 : x ≠ 0) (h2 : z ≠ 0) (y : ℝ) (h3 : y = 1 / x + z) : 
    (x - 1 / x) * (y + 1 / y) = (x^2 - 1) * (1 + 2 * x * z + x^2 * z^2 + x^2) / (x^2 * (1 + x * z)) := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1142_114229


namespace NUMINAMATH_GPT_tony_water_drink_l1142_114211

theorem tony_water_drink (W : ℝ) (h : W - 0.04 * W = 48) : W = 50 :=
sorry

end NUMINAMATH_GPT_tony_water_drink_l1142_114211


namespace NUMINAMATH_GPT_minimum_value_2x_4y_l1142_114275

theorem minimum_value_2x_4y (x y : ℝ) (h : x + 2 * y = 3) : 
  ∃ (min_val : ℝ), min_val = 2 ^ (5/2) ∧ (2 ^ x + 4 ^ y = min_val) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_2x_4y_l1142_114275
