import Mathlib

namespace NUMINAMATH_GPT_fractional_difference_l1942_194264

def recurring72 : ℚ := 72 / 99
def decimal72 : ℚ := 72 / 100

theorem fractional_difference : recurring72 - decimal72 = 2 / 275 := by
  sorry

end NUMINAMATH_GPT_fractional_difference_l1942_194264


namespace NUMINAMATH_GPT_speed_of_train_A_is_90_kmph_l1942_194212

-- Definitions based on the conditions
def train_length_A := 225 -- in meters
def train_length_B := 150 -- in meters
def crossing_time := 15 -- in seconds

-- The total distance covered by train A to cross train B
def total_distance := train_length_A + train_length_B

-- The speed of train A in m/s
def speed_in_mps := total_distance / crossing_time

-- Conversion factor from m/s to km/hr
def mps_to_kmph (mps: ℕ) := mps * 36 / 10

-- The speed of train A in km/hr
def speed_in_kmph := mps_to_kmph speed_in_mps

-- The theorem to be proved
theorem speed_of_train_A_is_90_kmph : speed_in_kmph = 90 := by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_speed_of_train_A_is_90_kmph_l1942_194212


namespace NUMINAMATH_GPT_distinct_real_roots_of_quadratic_l1942_194232

/-
Given a quadratic equation x^2 + 4x = 0,
prove that the equation has two distinct real roots.
-/

theorem distinct_real_roots_of_quadratic : 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = 0 → (b^2 - 4 * a * c) > 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ (r₁^2 + 4 * r₁ = 0) ∧ (r₂^2 + 4 * r₂ = 0) := 
by
  intros a b c ha hb hc hΔ
  sorry -- Proof to be provided later

end NUMINAMATH_GPT_distinct_real_roots_of_quadratic_l1942_194232


namespace NUMINAMATH_GPT_tire_price_l1942_194222

theorem tire_price (x : ℝ) (h1 : 2 * x + 5 = 185) : x = 90 :=
by
  sorry

end NUMINAMATH_GPT_tire_price_l1942_194222


namespace NUMINAMATH_GPT_min_value_of_f_in_D_l1942_194239

noncomputable def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x * y + y^2) - 3 * (x + y) + 5

def D (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem min_value_of_f_in_D : ∃ (x y : ℝ), D x y ∧ f x y = 2 ∧ (∀ (u v : ℝ), D u v → f u v ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_in_D_l1942_194239


namespace NUMINAMATH_GPT_company_employees_count_l1942_194218

theorem company_employees_count :
  ∃ E : ℕ, E = 80 + 100 - 30 + 20 := 
sorry

end NUMINAMATH_GPT_company_employees_count_l1942_194218


namespace NUMINAMATH_GPT_simplify_trig_expression_l1942_194204

theorem simplify_trig_expression : 2 * Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_simplify_trig_expression_l1942_194204


namespace NUMINAMATH_GPT_force_for_wrenches_l1942_194266

open Real

theorem force_for_wrenches (F : ℝ) (k : ℝ) :
  (F * 12 = 3600) → 
  (k = 3600) →
  (3600 / 8 = 450) →
  (3600 / 18 = 200) →
  true :=
by
  intro hF hk h8 h18
  trivial

end NUMINAMATH_GPT_force_for_wrenches_l1942_194266


namespace NUMINAMATH_GPT_solve_N1N2_identity_l1942_194210

theorem solve_N1N2_identity :
  (∃ N1 N2 : ℚ,
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 3 →
      (42 * x - 37) / (x^2 - 4 * x + 3) =
      N1 / (x - 1) + N2 / (x - 3)) ∧ 
      N1 * N2 = -445 / 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_N1N2_identity_l1942_194210


namespace NUMINAMATH_GPT_circumcircle_eq_l1942_194224

-- Definitions and conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4
def point_P : (ℝ × ℝ) := (4, 2)
def is_tangent_point (x y : ℝ) : Prop := sorry -- You need a proper definition for tangency

theorem circumcircle_eq :
  ∃ (hA : is_tangent_point 0 2) (hB : ∃ x y, is_tangent_point x y),
  ∃ (x y : ℝ), (circle_eq 0 2 ∧ circle_eq x y) ∧ (x-2)^2 + (y-1)^2 = 5 :=
  sorry

end NUMINAMATH_GPT_circumcircle_eq_l1942_194224


namespace NUMINAMATH_GPT_intersection_M_N_l1942_194285

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {x | x ≥ 3}

theorem intersection_M_N : M ∩ N = {3, 4} := 
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1942_194285


namespace NUMINAMATH_GPT_change_calculation_l1942_194219

/-!
# Problem
Adam has $5 to buy an airplane that costs $4.28. How much change will he get after buying the airplane?

# Conditions
Adam has $5.
The airplane costs $4.28.

# Statement
Prove that the change Adam will get is $0.72.
-/

theorem change_calculation : 
  let amount := 5.00
  let cost := 4.28
  let change := 0.72
  amount - cost = change :=
by 
  sorry

end NUMINAMATH_GPT_change_calculation_l1942_194219


namespace NUMINAMATH_GPT_length_of_goods_train_l1942_194289

theorem length_of_goods_train
  (speed_man_train : ℕ) (speed_goods_train : ℕ) (passing_time : ℕ)
  (h1 : speed_man_train = 40)
  (h2 : speed_goods_train = 72)
  (h3 : passing_time = 9) :
  (112 * 1000 / 3600) * passing_time = 280 := 
by
  sorry

end NUMINAMATH_GPT_length_of_goods_train_l1942_194289


namespace NUMINAMATH_GPT_price_per_glass_second_day_l1942_194234

theorem price_per_glass_second_day (O : ℝ) (P : ℝ) 
  (V1 : ℝ := 2 * O) -- Volume on the first day
  (V2 : ℝ := 3 * O) -- Volume on the second day
  (price_first_day : ℝ := 0.30) -- Price per glass on the first day
  (revenue_equal : V1 * price_first_day = V2 * P) :
  P = 0.20 := 
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_price_per_glass_second_day_l1942_194234


namespace NUMINAMATH_GPT_sara_total_money_eq_640_l1942_194223

def days_per_week : ℕ := 5
def cakes_per_day : ℕ := 4
def price_per_cake : ℕ := 8
def weeks : ℕ := 4

theorem sara_total_money_eq_640 :
  (days_per_week * cakes_per_day * price_per_cake * weeks) = 640 := 
sorry

end NUMINAMATH_GPT_sara_total_money_eq_640_l1942_194223


namespace NUMINAMATH_GPT_part1_part2_l1942_194275

open Set

variable {m x : ℝ}

def A (m : ℝ) : Set ℝ := { x | x^2 - (m+1)*x + m = 0 }
def B (m : ℝ) : Set ℝ := { x | x * m - 1 = 0 }

theorem part1 (h : A m ⊆ B m) : m = 1 :=
by
  sorry

theorem part2 (h : B m ⊂ A m) : m = 0 ∨ m = -1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1942_194275


namespace NUMINAMATH_GPT_total_toothpicks_needed_l1942_194236

/-- The number of toothpicks needed to construct both a large and smaller equilateral triangle 
    side by side, given the large triangle has a base of 100 small triangles and the smaller triangle 
    has a base of 50 small triangles -/
theorem total_toothpicks_needed 
  (base_large : ℕ) (base_small : ℕ) (shared_boundary : ℕ) 
  (h1 : base_large = 100) (h2 : base_small = 50) (h3 : shared_boundary = base_small) :
  3 * (100 * 101 / 2) / 2 + 3 * (50 * 51 / 2) / 2 - shared_boundary = 9462 := 
sorry

end NUMINAMATH_GPT_total_toothpicks_needed_l1942_194236


namespace NUMINAMATH_GPT_evaluate_expression_l1942_194203

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1942_194203


namespace NUMINAMATH_GPT_min_value_x2y3z_l1942_194202

theorem min_value_x2y3z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 2 / x + 3 / y + 1 / z = 12) :
  x^2 * y^3 * z ≥ 1 / 64 :=
by
  sorry

end NUMINAMATH_GPT_min_value_x2y3z_l1942_194202


namespace NUMINAMATH_GPT_basketball_substitution_mod_1000_l1942_194290

def basketball_substitution_count_mod (n_playing n_substitutes max_subs : ℕ) : ℕ :=
  let no_subs := 1
  let one_sub := n_playing * n_substitutes
  let two_subs := n_playing * (n_playing - 1) * (n_substitutes * (n_substitutes - 1)) / 2
  let three_subs := n_playing * (n_playing - 1) * (n_playing - 2) *
                    (n_substitutes * (n_substitutes - 1) * (n_substitutes - 2)) / 6
  no_subs + one_sub + two_subs + three_subs 

theorem basketball_substitution_mod_1000 :
  basketball_substitution_count_mod 9 9 3 % 1000 = 10 :=
  by 
    -- Here the proof would be implemented
    sorry

end NUMINAMATH_GPT_basketball_substitution_mod_1000_l1942_194290


namespace NUMINAMATH_GPT_problem_solution_l1942_194229

variable (U : Set Real) (a b : Real) (t : Real)
variable (A B : Set Real)

-- Conditions
def condition1 : U = Set.univ := sorry

def condition2 : ∀ x, a ≠ 0 → ax^2 + 2 * x + b > 0 ↔ x ≠ -1 / a := sorry

def condition3 : a > b := sorry

def condition4 : t = (a^2 + b^2) / (a - b) := sorry

def condition5 : ∀ m, (∀ x, |x + 1| - |x - 3| ≤ m^2 - 3 * m) → m ∈ B := sorry

-- To Prove
theorem problem_solution : A ∩ (Set.univ \ B) = {m : Real | 2 * Real.sqrt 2 ≤ m ∧ m < 4} := sorry

end NUMINAMATH_GPT_problem_solution_l1942_194229


namespace NUMINAMATH_GPT_distance_between_first_and_last_tree_l1942_194213

theorem distance_between_first_and_last_tree (n : ℕ) (d : ℕ) 
  (h1 : n = 10) 
  (h2 : d = 100) 
  (h3 : d / 5 = 20) :
  (20 * 9 = 180) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_first_and_last_tree_l1942_194213


namespace NUMINAMATH_GPT_inequality_proof_l1942_194268

theorem inequality_proof (a b c A α : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (h_sum : a + b + c = A) (hA : A ≤ 1) (hα : α > 0) :
  ( (1 / a - a) ^ α + (1 / b - b) ^ α + (1 / c - c) ^ α ) ≥ 3 * ( (3 / A) - (A / 3) ) ^ α :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1942_194268


namespace NUMINAMATH_GPT_number_of_customers_before_lunch_rush_l1942_194231

-- Defining the total number of customers during the lunch rush
def total_customers_during_lunch_rush : ℕ := 49 + 2

-- Defining the number of additional customers during the lunch rush
def additional_customers : ℕ := 12

-- Target statement to prove
theorem number_of_customers_before_lunch_rush : total_customers_during_lunch_rush - additional_customers = 39 :=
  by sorry

end NUMINAMATH_GPT_number_of_customers_before_lunch_rush_l1942_194231


namespace NUMINAMATH_GPT_count_sequences_of_length_15_l1942_194201

def countingValidSequences (n : ℕ) : ℕ := sorry

theorem count_sequences_of_length_15 :
  countingValidSequences 15 = 266 :=
  sorry

end NUMINAMATH_GPT_count_sequences_of_length_15_l1942_194201


namespace NUMINAMATH_GPT_sum_of_circle_areas_l1942_194269

theorem sum_of_circle_areas 
    (r s t : ℝ)
    (h1 : r + s = 6)
    (h2 : r + t = 8)
    (h3 : s + t = 10) : 
    (π * r^2 + π * s^2 + π * t^2) = 36 * π := 
by
    sorry

end NUMINAMATH_GPT_sum_of_circle_areas_l1942_194269


namespace NUMINAMATH_GPT_factory_produces_correct_number_of_doors_l1942_194284

variable (initial_planned_production : ℕ) (metal_shortage_decrease : ℕ) (pandemic_decrease_factor : ℕ)
variable (doors_per_car : ℕ)

theorem factory_produces_correct_number_of_doors
  (h1 : initial_planned_production = 200)
  (h2 : metal_shortage_decrease = 50)
  (h3 : pandemic_decrease_factor = 50)
  (h4 : doors_per_car = 5) :
  (initial_planned_production - metal_shortage_decrease) * (100 - pandemic_decrease_factor) * doors_per_car / 100 = 375 :=
by
  sorry

end NUMINAMATH_GPT_factory_produces_correct_number_of_doors_l1942_194284


namespace NUMINAMATH_GPT_avg_price_of_pen_l1942_194278

theorem avg_price_of_pen 
  (total_pens : ℕ) (total_pencils : ℕ) (total_cost : ℕ) 
  (avg_price_pencil : ℕ) (total_pens_cost : ℕ) (total_pencils_cost : ℕ)
  (total_cost_eq : total_cost = total_pens_cost + total_pencils_cost)
  (total_pencils_cost_eq : total_pencils_cost = total_pencils * avg_price_pencil)
  (pencils_count : total_pencils = 75) (pens_count : total_pens = 30) 
  (avg_price_pencil_eq : avg_price_pencil = 2)
  (total_cost_eq' : total_cost = 450) :
  total_pens_cost / total_pens = 10 :=
by
  sorry

end NUMINAMATH_GPT_avg_price_of_pen_l1942_194278


namespace NUMINAMATH_GPT_combined_variance_is_178_l1942_194279

noncomputable def average_weight_A := 60
noncomputable def variance_A := 100
noncomputable def average_weight_B := 64
noncomputable def variance_B := 200
noncomputable def ratio_A_B := (1, 3)

theorem combined_variance_is_178 :
  let nA := ratio_A_B.1
  let nB := ratio_A_B.2
  let avg_comb := (nA * average_weight_A + nB * average_weight_B) / (nA + nB)
  let var_comb := (nA * (variance_A + (average_weight_A - avg_comb)^2) + 
                   nB * (variance_B + (average_weight_B - avg_comb)^2)) / 
                   (nA + nB)
  var_comb = 178 := 
by
  sorry

end NUMINAMATH_GPT_combined_variance_is_178_l1942_194279


namespace NUMINAMATH_GPT_find_d_l1942_194276

theorem find_d (c a m d : ℝ) (h : m = (c * a * d) / (a - d)) : d = (m * a) / (m + c * a) :=
by sorry

end NUMINAMATH_GPT_find_d_l1942_194276


namespace NUMINAMATH_GPT_boiling_point_water_standard_l1942_194283

def boiling_point_water_celsius : ℝ := 100

theorem boiling_point_water_standard (bp_f : ℝ := 212) (ice_melting_c : ℝ := 0) (ice_melting_f : ℝ := 32) (pot_temp_c : ℝ := 55) (pot_temp_f : ℝ := 131) : boiling_point_water_celsius = 100 :=
by 
  -- Assuming standard atmospheric conditions, the boiling point of water in Celsius is 100.
  sorry

end NUMINAMATH_GPT_boiling_point_water_standard_l1942_194283


namespace NUMINAMATH_GPT_expression_independent_of_alpha_l1942_194211

theorem expression_independent_of_alpha
  (α : Real) (n : ℤ) (h : α ≠ (n * (π / 2)) + (π / 12)) :
  (1 - 2 * Real.sin (α - (3 * π / 2))^2 + (Real.sqrt 3) * Real.cos (2 * α + (3 * π / 2))) /
  (Real.sin (π / 6 - 2 * α)) = -2 := 
sorry

end NUMINAMATH_GPT_expression_independent_of_alpha_l1942_194211


namespace NUMINAMATH_GPT_bills_average_speed_l1942_194259

theorem bills_average_speed :
  ∃ v t : ℝ, 
      (v + 5) * (t + 2) + v * t = 680 ∧ 
      (t + 2) + t = 18 ∧ 
      v = 35 :=
by
  sorry

end NUMINAMATH_GPT_bills_average_speed_l1942_194259


namespace NUMINAMATH_GPT_salt_solution_percentage_l1942_194215

theorem salt_solution_percentage
  (x : ℝ)
  (y : ℝ)
  (h1 : 600 + y = 1000)
  (h2 : 600 * x + y * 0.12 = 1000 * 0.084) :
  x = 0.06 :=
by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_salt_solution_percentage_l1942_194215


namespace NUMINAMATH_GPT_average_four_numbers_l1942_194298

variable {x : ℝ}

theorem average_four_numbers (h : (15 + 25 + x + 30) / 4 = 23) : x = 22 :=
by
  sorry

end NUMINAMATH_GPT_average_four_numbers_l1942_194298


namespace NUMINAMATH_GPT_find_xy_value_l1942_194241

theorem find_xy_value (x y z w : ℕ) (h1 : x = w) (h2 : y = z) (h3 : w + w = z * w) (h4 : y = w)
    (h5 : w + w = w * w) (h6 : z = 3) : x * y = 4 := by
  -- Given that w = 2 based on the conditions
  sorry

end NUMINAMATH_GPT_find_xy_value_l1942_194241


namespace NUMINAMATH_GPT_problem_l1942_194263

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * b * x + c

def f_prime (a b x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + 3 * b

theorem problem (a b c : ℝ) (h1 : f_prime a b 2 = 0) (h2 : f_prime a b 1 = -3) :
  a = -1 ∧ b = 0 ∧ (let f_min := f (-1) 0 c 2 
                   let f_max := 0 
                   f_max - f_min = 4) :=
by
  sorry

end NUMINAMATH_GPT_problem_l1942_194263


namespace NUMINAMATH_GPT_molecular_weight_correct_l1942_194272

-- Define the atomic weights
def atomic_weight_Cu : ℝ := 63.546
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

-- Define the number of atoms in the compound
def num_atoms_Cu : ℕ := 1
def num_atoms_C : ℕ := 1
def num_atoms_O : ℕ := 3

-- Define the molecular weight calculation
def molecular_weight : ℝ :=
  num_atoms_Cu * atomic_weight_Cu + 
  num_atoms_C * atomic_weight_C + 
  num_atoms_O * atomic_weight_O

-- Prove the molecular weight of the compound
theorem molecular_weight_correct : molecular_weight = 123.554 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_correct_l1942_194272


namespace NUMINAMATH_GPT_thief_speed_l1942_194225

theorem thief_speed (v : ℝ) (hv : v > 0) : 
  let head_start_duration := (1/2 : ℝ)  -- 30 minutes, converted to hours
  let owner_speed := (75 : ℝ)  -- speed of owner in kmph
  let chase_duration := (2 : ℝ)  -- duration of the chase in hours
  let distance_by_owner := owner_speed * chase_duration  -- distance covered by the owner
  let total_distance_thief := head_start_duration * v + chase_duration * v  -- total distance covered by the thief
  distance_by_owner = 150 ->  -- given that owner covers 150 km
  total_distance_thief = 150  -- and so should the thief
  -> v = 60 := sorry

end NUMINAMATH_GPT_thief_speed_l1942_194225


namespace NUMINAMATH_GPT_solve_equation_l1942_194287

theorem solve_equation :
  ∀ x : ℝ, (4 * x - 2 * x + 1 - 3 = 0) ↔ (x = 1 ∨ x = -1) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_equation_l1942_194287


namespace NUMINAMATH_GPT_investment_Q_correct_l1942_194244

-- Define the investments of P and Q
def investment_P : ℝ := 40000
def investment_Q : ℝ := 60000

-- Define the profit share ratio
def profit_ratio_PQ : ℝ × ℝ := (2, 3)

-- State the theorem to prove
theorem investment_Q_correct :
  (investment_P / investment_Q = (profit_ratio_PQ.1 / profit_ratio_PQ.2)) → 
  investment_Q = 60000 := 
by 
  sorry

end NUMINAMATH_GPT_investment_Q_correct_l1942_194244


namespace NUMINAMATH_GPT_value_of_expression_l1942_194260

theorem value_of_expression (x y : ℝ) (h : x + 2 * y = 30) : (x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3) = 16 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1942_194260


namespace NUMINAMATH_GPT_find_k_for_min_value_zero_l1942_194230

theorem find_k_for_min_value_zero :
  (∃ k : ℝ, ∀ x y : ℝ, 9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 ≥ 0 ∧
                         ∃ x y : ℝ, 9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 = 0) →
  k = 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_k_for_min_value_zero_l1942_194230


namespace NUMINAMATH_GPT_range_of_a_l1942_194238

theorem range_of_a (a : ℝ) (h1 : ∃ x : ℝ, x > 0 ∧ |x| = a * x - a) (h2 : ∀ x : ℝ, x < 0 → |x| ≠ a * x - a) : a > 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1942_194238


namespace NUMINAMATH_GPT_eduardo_needs_l1942_194294

variable (flour_per_24_cookies sugar_per_24_cookies : ℝ)
variable (num_cookies : ℝ)

axiom h_flour : flour_per_24_cookies = 1.5
axiom h_sugar : sugar_per_24_cookies = 0.5
axiom h_cookies : num_cookies = 120

theorem eduardo_needs (scaling_factor : ℝ) 
    (flour_needed : ℝ)
    (sugar_needed : ℝ)
    (h_scaling : scaling_factor = num_cookies / 24)
    (h_flour_needed : flour_needed = flour_per_24_cookies * scaling_factor)
    (h_sugar_needed : sugar_needed = sugar_per_24_cookies * scaling_factor) :
  flour_needed = 7.5 ∧ sugar_needed = 2.5 :=
sorry

end NUMINAMATH_GPT_eduardo_needs_l1942_194294


namespace NUMINAMATH_GPT_g_cross_horizontal_asymptote_at_l1942_194209

noncomputable def g (x : ℝ) : ℝ :=
  (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 6)

theorem g_cross_horizontal_asymptote_at (x : ℝ) : g x = 3 ↔ x = 13 / 4 :=
by
  sorry

end NUMINAMATH_GPT_g_cross_horizontal_asymptote_at_l1942_194209


namespace NUMINAMATH_GPT_f_g_2_eq_1_l1942_194249

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := -2 * x + 5

theorem f_g_2_eq_1 : f (g 2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_f_g_2_eq_1_l1942_194249


namespace NUMINAMATH_GPT_horner_method_poly_at_neg2_l1942_194243

-- Define the polynomial using the given conditions and Horner's method transformation
def polynomial : ℤ → ℤ := fun x => (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 3) * x + 2

-- State the theorem
theorem horner_method_poly_at_neg2 : polynomial (-2) = -40 := by
  sorry

end NUMINAMATH_GPT_horner_method_poly_at_neg2_l1942_194243


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1942_194216

-- Definitions of assumptions and conditions.
structure Problem :=
  (boys : ℕ) -- number of boys
  (girls : ℕ) -- number of girls
  (subjects : ℕ) -- number of subjects
  (boyA_not_math : Prop) -- Boy A can't be a representative of the mathematics course
  (girlB_chinese : Prop) -- Girl B must be a representative of the Chinese language course

-- Problem 1: Calculate the number of ways satisfying condition (1)
theorem problem_1 (p : Problem) (h1 : p.girls < p.boys) :
  ∃ n : ℕ, n = 5520 := sorry

-- Problem 2: Calculate the number of ways satisfying condition (2)
theorem problem_2 (p : Problem) (h1 : p.boys ≥ 1) (h2 : p.boyA_not_math) :
  ∃ n : ℕ, n = 3360 := sorry

-- Problem 3: Calculate the number of ways satisfying condition (3)
theorem problem_3 (p : Problem) (h1 : p.boys ≥ 1) (h2 : p.boyA_not_math) (h3 : p.girlB_chinese) :
  ∃ n : ℕ, n = 360 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1942_194216


namespace NUMINAMATH_GPT_slope_angle_tangent_line_at_zero_l1942_194292

noncomputable def curve (x : ℝ) : ℝ := 2 * x - Real.exp x

noncomputable def slope_at (x : ℝ) : ℝ := 
  (deriv curve) x

theorem slope_angle_tangent_line_at_zero : 
  Real.arctan (slope_at 0) = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_slope_angle_tangent_line_at_zero_l1942_194292


namespace NUMINAMATH_GPT_smallest_value_inequality_l1942_194200

variable (a b c d : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

theorem smallest_value_inequality :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
sorry

end NUMINAMATH_GPT_smallest_value_inequality_l1942_194200


namespace NUMINAMATH_GPT_ratio_of_John_to_Mary_l1942_194247

-- Definitions based on conditions
variable (J M T : ℕ)
variable (hT : T = 60)
variable (hJ : J = T / 2)
variable (hAvg : (J + M + T) / 3 = 35)

-- Statement to prove
theorem ratio_of_John_to_Mary : J / M = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_John_to_Mary_l1942_194247


namespace NUMINAMATH_GPT_gcd_8994_13326_37566_l1942_194274

-- Define the integers involved
def a := 8994
def b := 13326
def c := 37566

-- Assert the GCD relation
theorem gcd_8994_13326_37566 : Int.gcd a (Int.gcd b c) = 2 := by
  sorry

end NUMINAMATH_GPT_gcd_8994_13326_37566_l1942_194274


namespace NUMINAMATH_GPT_three_digit_perfect_squares_div_by_4_count_l1942_194246

theorem three_digit_perfect_squares_div_by_4_count : 
  (∃ count : ℕ, count = 11 ∧ (∀ n : ℕ, 10 ≤ n ∧ n ≤ 31 → n^2 ≥ 100 ∧ n^2 ≤ 999 ∧ n^2 % 4 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_perfect_squares_div_by_4_count_l1942_194246


namespace NUMINAMATH_GPT_coprime_powers_l1942_194271

theorem coprime_powers (n : ℕ) : Nat.gcd (n^5 + 4 * n^3 + 3 * n) (n^4 + 3 * n^2 + 1) = 1 :=
sorry

end NUMINAMATH_GPT_coprime_powers_l1942_194271


namespace NUMINAMATH_GPT_quadratic_no_real_roots_implies_inequality_l1942_194277

theorem quadratic_no_real_roots_implies_inequality (a b c : ℝ) :
  let A := b + c
  let B := a + c
  let C := a + b
  (B^2 - 4 * A * C < 0) → 4 * a * c - b^2 ≤ 3 * a * (a + b + c) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_implies_inequality_l1942_194277


namespace NUMINAMATH_GPT_students_per_group_l1942_194226

-- Define the conditions:
def total_students : ℕ := 120
def not_picked_students : ℕ := 22
def groups : ℕ := 14

-- Calculate the picked students:
def picked_students : ℕ := total_students - not_picked_students

-- Statement of the problem:
theorem students_per_group : picked_students / groups = 7 :=
  by sorry

end NUMINAMATH_GPT_students_per_group_l1942_194226


namespace NUMINAMATH_GPT_total_initial_passengers_l1942_194245

theorem total_initial_passengers (M W : ℕ) 
  (h1 : W = M / 3) 
  (h2 : M - 24 = W + 12) : 
  M + W = 72 :=
sorry

end NUMINAMATH_GPT_total_initial_passengers_l1942_194245


namespace NUMINAMATH_GPT_vec_parallel_l1942_194281

variable {R : Type*} [LinearOrderedField R]

def is_parallel (a b : R × R) : Prop :=
  ∃ k : R, a = (k * b.1, k * b.2)

theorem vec_parallel {x : R} : 
  is_parallel (1, x) (-3, 4) ↔ x = -4/3 := by
  sorry

end NUMINAMATH_GPT_vec_parallel_l1942_194281


namespace NUMINAMATH_GPT_average_fuel_consumption_correct_l1942_194240

def distance_to_x : ℕ := 150
def distance_to_y : ℕ := 220
def fuel_to_x : ℕ := 20
def fuel_to_y : ℕ := 15

def total_distance : ℕ := distance_to_x + distance_to_y
def total_fuel_used : ℕ := fuel_to_x + fuel_to_y
def avg_fuel_consumption : ℚ := total_fuel_used / total_distance

theorem average_fuel_consumption_correct :
  avg_fuel_consumption = 0.0946 := by
  sorry

end NUMINAMATH_GPT_average_fuel_consumption_correct_l1942_194240


namespace NUMINAMATH_GPT_rational_mul_example_l1942_194248

theorem rational_mul_example : ((19 + 15 / 16) * (-8)) = (-159 - 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_rational_mul_example_l1942_194248


namespace NUMINAMATH_GPT_calc_value_l1942_194291

noncomputable def f : ℝ → ℝ := sorry 

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom non_const_zero : ∃ x : ℝ, f x ≠ 0
axiom functional_eq : ∀ x : ℝ, x * f (x + 1) = (x + 1) * f x

theorem calc_value : f (f (5 / 2)) = 0 :=
sorry

end NUMINAMATH_GPT_calc_value_l1942_194291


namespace NUMINAMATH_GPT_inequality_proof_l1942_194227

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a < a - b :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1942_194227


namespace NUMINAMATH_GPT_consumption_increase_percentage_l1942_194253

theorem consumption_increase_percentage
  (T C : ℝ)
  (H1 : 0.90 * (1 + X/100) = 0.9999999999999858) :
  X = 11.11111111110953 :=
by
  sorry

end NUMINAMATH_GPT_consumption_increase_percentage_l1942_194253


namespace NUMINAMATH_GPT_positive_integers_of_inequality_l1942_194228

theorem positive_integers_of_inequality (x : ℕ) (h : 9 - 3 * x > 0) : x = 1 ∨ x = 2 :=
sorry

end NUMINAMATH_GPT_positive_integers_of_inequality_l1942_194228


namespace NUMINAMATH_GPT_weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900_l1942_194251

noncomputable def Wa : ℕ := 
  let volume_a := (3/5) * 4
  let volume_b := (2/5) * 4
  let weight_b := 700
  let total_weight := 3280
  (total_weight - (weight_b * volume_b)) / volume_a

theorem weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900 :
  Wa = 900 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900_l1942_194251


namespace NUMINAMATH_GPT_product_three_consecutive_not_power_l1942_194250

theorem product_three_consecutive_not_power (n k m : ℕ) (hn : n > 0) (hm : m ≥ 2) : 
  (n-1) * n * (n+1) ≠ k^m :=
by sorry

end NUMINAMATH_GPT_product_three_consecutive_not_power_l1942_194250


namespace NUMINAMATH_GPT_yellow_fraction_after_changes_l1942_194208

theorem yellow_fraction_after_changes (y : ℕ) :
  let green_initial := (4 / 7 : ℚ) * y
  let yellow_initial := (3 / 7 : ℚ) * y
  let yellow_new := 3 * yellow_initial
  let green_new := green_initial + (1 / 2) * green_initial
  let total_new := green_new + yellow_new
  yellow_new / total_new = (3 / 5 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_yellow_fraction_after_changes_l1942_194208


namespace NUMINAMATH_GPT_sum_of_fractions_l1942_194220

theorem sum_of_fractions : (1 / 6) + (2 / 9) + (1 / 3) = 13 / 18 := by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l1942_194220


namespace NUMINAMATH_GPT_nth_equation_l1942_194280

theorem nth_equation (n : ℕ) (h : 0 < n) : (10 * n + 5) ^ 2 = n * (n + 1) * 100 + 5 ^ 2 := 
sorry

end NUMINAMATH_GPT_nth_equation_l1942_194280


namespace NUMINAMATH_GPT_mimi_spent_on_clothes_l1942_194288

noncomputable def total_cost : ℤ := 8000
noncomputable def cost_adidas : ℤ := 600
noncomputable def cost_nike : ℤ := 3 * cost_adidas
noncomputable def cost_skechers : ℤ := 5 * cost_adidas
noncomputable def cost_clothes : ℤ := total_cost - (cost_adidas + cost_nike + cost_skechers)

theorem mimi_spent_on_clothes :
  cost_clothes = 2600 :=
by
  sorry

end NUMINAMATH_GPT_mimi_spent_on_clothes_l1942_194288


namespace NUMINAMATH_GPT_rectangle_y_value_l1942_194295

theorem rectangle_y_value (y : ℝ) (h₁ : (-2, y) ≠ (10, y))
  (h₂ : (-2, -1) ≠ (10, -1))
  (h₃ : 12 * (y + 1) = 108)
  (y_pos : 0 < y) :
  y = 8 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_y_value_l1942_194295


namespace NUMINAMATH_GPT_visibility_beach_to_hill_visibility_ferry_to_tree_l1942_194242

noncomputable def altitude_lake : ℝ := 104
noncomputable def altitude_hill_tree : ℝ := 154
noncomputable def map_distance_1 : ℝ := 70 / 100 -- Convert cm to meters
noncomputable def map_distance_2 : ℝ := 38.5 / 100 -- Convert cm to meters
noncomputable def map_scale : ℝ := 95000
noncomputable def earth_circumference : ℝ := 40000000 -- Convert km to meters

noncomputable def earth_radius : ℝ := earth_circumference / (2 * Real.pi)

noncomputable def visible_distance (height : ℝ) : ℝ :=
  Real.sqrt (2 * earth_radius * height)

noncomputable def actual_distance_1 : ℝ := map_distance_1 * map_scale
noncomputable def actual_distance_2 : ℝ := map_distance_2 * map_scale

theorem visibility_beach_to_hill :
  actual_distance_1 > visible_distance (altitude_hill_tree - altitude_lake) :=
by
  sorry

theorem visibility_ferry_to_tree :
  actual_distance_2 > visible_distance (altitude_hill_tree - altitude_lake) :=
by
  sorry

end NUMINAMATH_GPT_visibility_beach_to_hill_visibility_ferry_to_tree_l1942_194242


namespace NUMINAMATH_GPT_range_of_m_l1942_194286

theorem range_of_m (m : ℝ) :
  (∀ x, |x^2 - 4 * x + m| ≤ x + 4 ↔ (-4 ≤ m ∧ m ≤ 4)) ∧
  (∀ x, (x = 0 → |0^2 - 4 * 0 + m| ≤ 0 + 4) ∧ (x = 2 → ¬(|2^2 - 4 * 2 + m| ≤ 2 + 4))) →
  (-4 ≤ m ∧ m < -2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1942_194286


namespace NUMINAMATH_GPT_ways_to_start_writing_l1942_194258

def ratio_of_pens_to_notebooks (pens notebooks : ℕ) : Prop := 
    pens * 4 = notebooks * 5

theorem ways_to_start_writing 
    (pens notebooks : ℕ) 
    (h_ratio : ratio_of_pens_to_notebooks pens notebooks) 
    (h_pens : pens = 50)
    (h_notebooks : notebooks = 40) : 
    ∃ ways : ℕ, ways = 40 :=
by
  sorry

end NUMINAMATH_GPT_ways_to_start_writing_l1942_194258


namespace NUMINAMATH_GPT_sqrt_product_equals_l1942_194296

noncomputable def sqrt128 : ℝ := Real.sqrt 128
noncomputable def sqrt50 : ℝ := Real.sqrt 50
noncomputable def sqrt18 : ℝ := Real.sqrt 18

theorem sqrt_product_equals : sqrt128 * sqrt50 * sqrt18 = 240 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_product_equals_l1942_194296


namespace NUMINAMATH_GPT_binomial_expansion_coefficient_l1942_194252

theorem binomial_expansion_coefficient :
  let a_0 : ℚ := (1 + 2 * (0:ℚ))^5
  (1 + 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_3 = 80 :=
by 
  sorry

end NUMINAMATH_GPT_binomial_expansion_coefficient_l1942_194252


namespace NUMINAMATH_GPT_positive_integer_as_sum_of_distinct_factors_l1942_194217

-- Defining that all elements of a list are factors of a given number
def AllFactorsOf (factors : List ℕ) (n : ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Defining that the sum of elements in the list equals a given number
def SumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

-- Theorem statement
theorem positive_integer_as_sum_of_distinct_factors (n m : ℕ) (hn : 0 < n) (hm : 1 ≤ m ∧ m ≤ n!) :
  ∃ factors : List ℕ, factors.length ≤ n ∧ AllFactorsOf factors n! ∧ SumList factors = m := 
sorry

end NUMINAMATH_GPT_positive_integer_as_sum_of_distinct_factors_l1942_194217


namespace NUMINAMATH_GPT_perpendicular_lines_l1942_194282

theorem perpendicular_lines (a : ℝ) :
  (∃ x y : ℝ, x * a + 3 * y - 1 = 0) ∧ (∃ x y : ℝ, 2 * x + (a - 1) * y + 1 = 0) ∧
  (∀ m1 m2 : ℝ, m1 = - a / 3 → m2 = - 2 / (a - 1) → m1 * m2 = -1) →
  a = 3 / 5 :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_l1942_194282


namespace NUMINAMATH_GPT_total_food_consumed_l1942_194214

theorem total_food_consumed (n1 n2 f1 f2 : ℕ) (h1 : n1 = 4000) (h2 : n2 = n1 - 500) (h3 : f1 = 10) (h4 : f2 = f1 - 2) : 
    n1 * f1 + n2 * f2 = 68000 := by 
  sorry

end NUMINAMATH_GPT_total_food_consumed_l1942_194214


namespace NUMINAMATH_GPT_min_days_required_l1942_194265

theorem min_days_required (n : ℕ) (h1 : n ≥ 1) (h2 : 2 * (2^n - 1) ≥ 100) : n = 6 :=
sorry

end NUMINAMATH_GPT_min_days_required_l1942_194265


namespace NUMINAMATH_GPT_range_of_a_l1942_194262

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x^2 + 2 * a * x + 1) → -1 ≤ a ∧ a ≤ 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1942_194262


namespace NUMINAMATH_GPT_betty_age_l1942_194261

theorem betty_age {A M B : ℕ} (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 14) : B = 7 :=
sorry

end NUMINAMATH_GPT_betty_age_l1942_194261


namespace NUMINAMATH_GPT_f_neg_a_eq_neg_2_l1942_194233

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 1

-- Given condition: f(a) = 4
variable (a : ℝ)
axiom h_f_a : f a = 4

-- We need to prove that: f(-a) = -2
theorem f_neg_a_eq_neg_2 (a : ℝ) (h_f_a : f a = 4) : f (-a) = -2 :=
by
  sorry

end NUMINAMATH_GPT_f_neg_a_eq_neg_2_l1942_194233


namespace NUMINAMATH_GPT_no_playful_two_digit_numbers_l1942_194256

def is_playful (a b : ℕ) : Prop := 10 * a + b = a^3 + b^2

theorem no_playful_two_digit_numbers :
  (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → ¬ is_playful a b) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_playful_two_digit_numbers_l1942_194256


namespace NUMINAMATH_GPT_min_value_x_squared_plus_y_squared_l1942_194270

theorem min_value_x_squared_plus_y_squared {x y : ℝ} 
  (h : x^2 + y^2 - 4*x - 6*y + 12 = 0) : 
  ∃ m : ℝ, m = 14 - 2 * Real.sqrt 13 ∧ ∀ u v : ℝ, (u^2 + v^2 - 4*u - 6*v + 12 = 0) → (u^2 + v^2 ≥ m) :=
by
  sorry

end NUMINAMATH_GPT_min_value_x_squared_plus_y_squared_l1942_194270


namespace NUMINAMATH_GPT_sum_of_ages_l1942_194207

/--
Given:
- Beckett's age is 12.
- Olaf is 3 years older than Beckett.
- Shannen is 2 years younger than Olaf.
- Jack is 5 more than twice as old as Shannen.

Prove that the sum of the ages of Beckett, Olaf, Shannen, and Jack is 71 years.
-/
theorem sum_of_ages :
  let Beckett := 12
  let Olaf := Beckett + 3
  let Shannen := Olaf - 2
  let Jack := 2 * Shannen + 5
  Beckett + Olaf + Shannen + Jack = 71 :=
by
  let Beckett := 12
  let Olaf := Beckett + 3
  let Shannen := Olaf - 2
  let Jack := 2 * Shannen + 5
  show Beckett + Olaf + Shannen + Jack = 71
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1942_194207


namespace NUMINAMATH_GPT_cost_of_ABC_book_l1942_194206

theorem cost_of_ABC_book (x : ℕ) 
  (h₁ : 8 = 8)  -- Cost of "TOP" book is 8 dollars
  (h₂ : 13 * 8 = 104)  -- Thirteen "TOP" books sold last week
  (h₃ : 104 - 4 * x = 12)  -- Difference in earnings is $12
  : x = 23 :=
sorry

end NUMINAMATH_GPT_cost_of_ABC_book_l1942_194206


namespace NUMINAMATH_GPT_find_a_l1942_194255

open Set Real

-- Defining sets A and B, and the condition A ∩ B = {3}
def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

-- Mathematically equivalent proof statement
theorem find_a (a : ℝ) (h : A ∩ B a = {3}) : a = 1 :=
  sorry

end NUMINAMATH_GPT_find_a_l1942_194255


namespace NUMINAMATH_GPT_sphere_diameter_l1942_194235

theorem sphere_diameter 
  (shadow_sphere : ℝ)
  (height_pole : ℝ)
  (shadow_pole : ℝ)
  (parallel_rays : Prop)
  (vertical_objects : Prop)
  (tan_theta : ℝ) :
  shadow_sphere = 12 →
  height_pole = 1.5 →
  shadow_pole = 3 →
  (tan_theta = height_pole / shadow_pole) →
  parallel_rays →
  vertical_objects →
  2 * (shadow_sphere * tan_theta) = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_sphere_diameter_l1942_194235


namespace NUMINAMATH_GPT_tangent_circle_equation_l1942_194297

theorem tangent_circle_equation :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi →
    ∃ c : ℝ × ℝ, ∃ r : ℝ,
      (∀ (a b : ℝ), c = (a, b) →
        (|a * Real.cos θ + b * Real.sin θ - Real.cos θ - 2 * Real.sin θ - 2| = r) ∧
        (r = 2)) ∧
      (∃ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = r^2)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_circle_equation_l1942_194297


namespace NUMINAMATH_GPT_frustum_has_only_two_parallel_surfaces_l1942_194273

-- Definitions for the geometric bodies in terms of their properties
structure Pyramid where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 0

structure Prism where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 6

structure Frustum where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 2

structure Cuboid where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 6

-- The main theorem stating that the Frustum is the one with exactly two parallel surfaces.
theorem frustum_has_only_two_parallel_surfaces (pyramid : Pyramid) (prism : Prism) (frustum : Frustum) (cuboid : Cuboid) :
  frustum.parallel_surfaces = 2 ∧
  pyramid.parallel_surfaces ≠ 2 ∧
  prism.parallel_surfaces ≠ 2 ∧
  cuboid.parallel_surfaces ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_frustum_has_only_two_parallel_surfaces_l1942_194273


namespace NUMINAMATH_GPT_prism_volume_l1942_194257

theorem prism_volume (x y z : ℝ) (h1 : x * y = 24) (h2 : y * z = 8) (h3 : x * z = 3) : 
  x * y * z = 24 :=
sorry

end NUMINAMATH_GPT_prism_volume_l1942_194257


namespace NUMINAMATH_GPT_final_price_percentage_l1942_194237

theorem final_price_percentage (P : ℝ) (h₀ : P > 0)
  (h₁ : ∃ P₁, P₁ = 0.80 * P)
  (h₂ : ∃ P₂, P₁ = 0.80 * P ∧ P₂ = P₁ - 0.10 * P₁) :
  P₂ = 0.72 * P :=
by
  sorry

end NUMINAMATH_GPT_final_price_percentage_l1942_194237


namespace NUMINAMATH_GPT_inequality_proof_l1942_194254

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (a + c) ≥ 2 * Real.sqrt (a * b * c * (a + b + c)) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1942_194254


namespace NUMINAMATH_GPT_larger_angle_measure_l1942_194299

theorem larger_angle_measure (x : ℝ) (hx : 7 * x = 90) : 4 * x = 360 / 7 := by
sorry

end NUMINAMATH_GPT_larger_angle_measure_l1942_194299


namespace NUMINAMATH_GPT_students_circle_no_regular_exists_zero_regular_school_students_l1942_194205

noncomputable def students_circle_no_regular (n : ℕ) 
    (student : ℕ → String)
    (neighbor_right : ℕ → ℕ)
    (lies_to : ℕ → ℕ → Bool) : Prop :=
  ∀ i, student i = "Gymnasium student" →
    (if lies_to i (neighbor_right i)
     then (student (neighbor_right i) ≠ "Gymnasium student")
     else student (neighbor_right i) = "Gymnasium student") →
    (if lies_to (neighbor_right i) i
     then (student i ≠ "Gymnasium student")
     else student i = "Gymnasium student")

theorem students_circle_no_regular_exists_zero_regular_school_students
  (n : ℕ) 
  (student : ℕ → String)
  (neighbor_right : ℕ → ℕ)
  (lies_to : ℕ → ℕ → Bool)
  (h : students_circle_no_regular n student neighbor_right lies_to)
  : (∀ i, student i ≠ "Regular school student") :=
sorry

end NUMINAMATH_GPT_students_circle_no_regular_exists_zero_regular_school_students_l1942_194205


namespace NUMINAMATH_GPT_number_of_unique_outfits_l1942_194293

-- Define the given conditions
def num_shirts : ℕ := 8
def num_ties : ℕ := 6
def special_shirt_ties : ℕ := 3
def remaining_shirts := num_shirts - 1
def remaining_ties := num_ties

-- Define the proof problem
theorem number_of_unique_outfits : num_shirts * num_ties - remaining_shirts * remaining_ties + special_shirt_ties = 45 :=
by
  sorry

end NUMINAMATH_GPT_number_of_unique_outfits_l1942_194293


namespace NUMINAMATH_GPT_fraction_relationship_l1942_194267

theorem fraction_relationship (a b c : ℚ)
  (h1 : a / b = 3 / 5)
  (h2 : b / c = 2 / 7) :
  c / a = 35 / 6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_relationship_l1942_194267


namespace NUMINAMATH_GPT_fraction_a_over_b_l1942_194221

theorem fraction_a_over_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_fraction_a_over_b_l1942_194221
