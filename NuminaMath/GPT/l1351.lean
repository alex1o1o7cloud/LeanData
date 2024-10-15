import Mathlib

namespace NUMINAMATH_GPT_largest_number_obtained_l1351_135149

theorem largest_number_obtained : 
  ∃ n : ℤ, 10 ≤ n ∧ n ≤ 99 ∧ (∀ m, 10 ≤ m ∧ m ≤ 99 → (250 - 3 * m)^2 ≤ (250 - 3 * n)^2) ∧ (250 - 3 * n)^2 = 4 :=
sorry

end NUMINAMATH_GPT_largest_number_obtained_l1351_135149


namespace NUMINAMATH_GPT_find_y_l1351_135109

-- Define the known values and the proportion relation
variable (x y : ℝ)
variable (h1 : 0.75 / x = y / 7)
variable (h2 : x = 1.05)

theorem find_y : y = 5 :=
by
sorry

end NUMINAMATH_GPT_find_y_l1351_135109


namespace NUMINAMATH_GPT_length_of_platform_l1351_135186

-- Definitions based on the problem conditions
def train_length : ℝ := 300
def platform_crossing_time : ℝ := 39
def signal_pole_crossing_time : ℝ := 18

-- The main theorem statement
theorem length_of_platform : ∀ (L : ℝ), train_length + L = (train_length / signal_pole_crossing_time) * platform_crossing_time → L = 350.13 :=
by
  intro L h
  sorry

end NUMINAMATH_GPT_length_of_platform_l1351_135186


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1351_135168

theorem hyperbola_eccentricity (h : ∀ x y m : ℝ, x^2 - y^2 / m = 1 → m > 0 → (Real.sqrt (1 + m) = Real.sqrt 3)) : ∃ m : ℝ, m = 2 := sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1351_135168


namespace NUMINAMATH_GPT_mass_percentage_H_in_CaH₂_l1351_135100

def atomic_mass_Ca : ℝ := 40.08
def atomic_mass_H : ℝ := 1.008
def molar_mass_CaH₂ : ℝ := atomic_mass_Ca + 2 * atomic_mass_H

theorem mass_percentage_H_in_CaH₂ :
  (2 * atomic_mass_H / molar_mass_CaH₂) * 100 = 4.79 := 
by
  -- Skipping the detailed proof for now
  sorry

end NUMINAMATH_GPT_mass_percentage_H_in_CaH₂_l1351_135100


namespace NUMINAMATH_GPT_value_of_x_l1351_135121

theorem value_of_x (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_x_l1351_135121


namespace NUMINAMATH_GPT_find_a_sq_plus_b_sq_l1351_135104

-- Variables and conditions
variables (a b : ℝ)
-- Conditions from the problem
axiom h1 : a - b = 3
axiom h2 : a * b = 9

-- The proof statement
theorem find_a_sq_plus_b_sq (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 9) : a^2 + b^2 = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_sq_plus_b_sq_l1351_135104


namespace NUMINAMATH_GPT_shifted_parabola_relationship_l1351_135162

-- Step a) and conditions
def original_function (x : ℝ) : ℝ := -2 * x ^ 2 + 4

def shift_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := fun x => f (x + a)
def shift_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := fun x => f x + b

-- Step c) encoding the proof problem
theorem shifted_parabola_relationship :
  (shift_up (shift_left original_function 2) 3 = fun x => -2 * (x + 2) ^ 2 + 7) :=
by
  sorry

end NUMINAMATH_GPT_shifted_parabola_relationship_l1351_135162


namespace NUMINAMATH_GPT_find_p_from_conditions_l1351_135142

variable (p : ℝ) (y x : ℝ)

noncomputable def parabola_eq : Prop := y^2 = 2 * p * x

noncomputable def p_positive : Prop := p > 0

noncomputable def point_on_parabola : Prop := parabola_eq p 1 (p / 4)

theorem find_p_from_conditions (hp : p_positive p) (hpp : point_on_parabola p) : p = Real.sqrt 2 :=
by 
  -- The actual proof goes here
  sorry

end NUMINAMATH_GPT_find_p_from_conditions_l1351_135142


namespace NUMINAMATH_GPT_divides_three_and_eleven_l1351_135112

theorem divides_three_and_eleven (n : ℕ) (h : n ≥ 1) : (n ∣ 3^n + 1 ∧ n ∣ 11^n + 1) ↔ (n = 1 ∨ n = 2) := by
  sorry

end NUMINAMATH_GPT_divides_three_and_eleven_l1351_135112


namespace NUMINAMATH_GPT_find_some_value_l1351_135145

theorem find_some_value (m n : ℝ) (some_value : ℝ) (p : ℝ) 
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + p = (n + some_value) / 6 - 2 / 5)
  (h3 : p = 3)
  : some_value = -12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_some_value_l1351_135145


namespace NUMINAMATH_GPT_remainder_17_pow_77_mod_7_l1351_135197

theorem remainder_17_pow_77_mod_7 : (17^77) % 7 = 5 := 
by sorry

end NUMINAMATH_GPT_remainder_17_pow_77_mod_7_l1351_135197


namespace NUMINAMATH_GPT_functional_equation_solution_l1351_135189

theorem functional_equation_solution (f : ℝ → ℝ) (t : ℝ) (h : t ≠ -1) :
  (∀ x y : ℝ, (t + 1) * f (1 + x * y) - f (x + y) = f (x + 1) * f (y + 1)) →
  (∀ x, f x = 0) ∨ (∀ x, f x = t) ∨ (∀ x, f x = (t + 1) * x - (t + 2)) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1351_135189


namespace NUMINAMATH_GPT_inequality_solution_l1351_135127

theorem inequality_solution (a : ℝ) (h : a > 0) :
  {x : ℝ | ax ^ 2 - (a + 1) * x + 1 < 0} =
    if a = 1 then ∅
    else if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1 / a}
    else if a > 1 then {x : ℝ | 1 / a < x ∧ x < 1} 
    else ∅ := sorry

end NUMINAMATH_GPT_inequality_solution_l1351_135127


namespace NUMINAMATH_GPT_tank_capacity_l1351_135171

variable (C : ℝ)

noncomputable def leak_rate := C / 6 -- litres per hour
noncomputable def inlet_rate := 6 * 60 -- litres per hour
noncomputable def net_emptying_rate := C / 12 -- litres per hour

theorem tank_capacity : 
  (360 - leak_rate C = net_emptying_rate C) → 
  C = 1440 :=
by 
  sorry

end NUMINAMATH_GPT_tank_capacity_l1351_135171


namespace NUMINAMATH_GPT_exists_even_among_pythagorean_triplet_l1351_135117

theorem exists_even_among_pythagorean_triplet (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ x, (x = a ∨ x = b ∨ x = c) ∧ x % 2 = 0 :=
sorry

end NUMINAMATH_GPT_exists_even_among_pythagorean_triplet_l1351_135117


namespace NUMINAMATH_GPT_days_to_fill_tank_l1351_135185

-- Definitions based on the problem conditions
def tank_capacity_liters : ℕ := 50
def liters_to_milliliters : ℕ := 1000
def rain_collection_per_day : ℕ := 800
def river_collection_per_day : ℕ := 1700
def total_collection_per_day : ℕ := rain_collection_per_day + river_collection_per_day
def tank_capacity_milliliters : ℕ := tank_capacity_liters * liters_to_milliliters

-- Statement of the proof that Jacob needs 20 days to fill the tank
theorem days_to_fill_tank : tank_capacity_milliliters / total_collection_per_day = 20 := by
  sorry

end NUMINAMATH_GPT_days_to_fill_tank_l1351_135185


namespace NUMINAMATH_GPT_positive_y_percentage_l1351_135150

theorem positive_y_percentage (y : ℝ) (hy_pos : 0 < y) (h : 0.01 * y * y = 9) : y = 30 := by
  sorry

end NUMINAMATH_GPT_positive_y_percentage_l1351_135150


namespace NUMINAMATH_GPT_inequality_not_less_than_four_by_at_least_one_l1351_135140

-- Definitions based on the conditions
def not_less_than_by_at_least (y : ℝ) (a b : ℝ) : Prop := y - a ≥ b

-- Problem statement (theorem) based on the given question and correct answer
theorem inequality_not_less_than_four_by_at_least_one (y : ℝ) :
  not_less_than_by_at_least y 4 1 → y ≥ 5 :=
by
  sorry

end NUMINAMATH_GPT_inequality_not_less_than_four_by_at_least_one_l1351_135140


namespace NUMINAMATH_GPT_winning_percentage_l1351_135183

/-- In an election with two candidates, wherein the winner received 490 votes and won by 280 votes,
we aim to prove that the winner received 70% of the total votes. -/

theorem winning_percentage (votes_winner : ℕ) (votes_margin : ℕ) (total_votes : ℕ)
  (h1 : votes_winner = 490) (h2 : votes_margin = 280)
  (h3 : total_votes = votes_winner + (votes_winner - votes_margin)) :
  (votes_winner * 100 / total_votes) = 70 :=
by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_winning_percentage_l1351_135183


namespace NUMINAMATH_GPT_circle_tangent_area_l1351_135164

noncomputable def circle_tangent_area_problem 
  (radiusA radiusB radiusC : ℝ) (tangent_midpoint : Bool) : ℝ :=
  if (radiusA = 1 ∧ radiusB = 1 ∧ radiusC = 2 ∧ tangent_midpoint) then 
    (4 * Real.pi) - (2 * Real.pi) 
  else 0

theorem circle_tangent_area (radiusA radiusB radiusC : ℝ) (tangent_midpoint : Bool) :
  radiusA = 1 → radiusB = 1 → radiusC = 2 → tangent_midpoint = true → 
  circle_tangent_area_problem radiusA radiusB radiusC tangent_midpoint = 2 * Real.pi :=
by
  intros
  simp [circle_tangent_area_problem]
  split_ifs
  · sorry
  · sorry

end NUMINAMATH_GPT_circle_tangent_area_l1351_135164


namespace NUMINAMATH_GPT_total_voters_in_districts_l1351_135141

theorem total_voters_in_districts :
  let D1 := 322
  let D2 := (D1 / 2) - 19
  let D3 := 2 * D1
  let D4 := D2 + 45
  let D5 := (3 * D3) - 150
  let D6 := (D1 + D4) + (1 / 5) * (D1 + D4)
  let D7 := D2 + (D5 - D2) / 2
  D1 + D2 + D3 + D4 + D5 + D6 + D7 = 4650 := 
by
  sorry

end NUMINAMATH_GPT_total_voters_in_districts_l1351_135141


namespace NUMINAMATH_GPT_probability_green_ball_l1351_135191

theorem probability_green_ball 
  (total_balls : ℕ) 
  (green_balls : ℕ) 
  (white_balls : ℕ) 
  (h_total : total_balls = 9) 
  (h_green : green_balls = 7)
  (h_white : white_balls = 2)
  (h_total_eq : total_balls = green_balls + white_balls) : 
  (green_balls / total_balls : ℚ) = 7 / 9 := 
by
  sorry

end NUMINAMATH_GPT_probability_green_ball_l1351_135191


namespace NUMINAMATH_GPT_remainder_of_n_mod_5_l1351_135113

theorem remainder_of_n_mod_5
  (n : Nat)
  (h1 : n^2 ≡ 4 [MOD 5])
  (h2 : n^3 ≡ 2 [MOD 5]) :
  n ≡ 3 [MOD 5] :=
sorry

end NUMINAMATH_GPT_remainder_of_n_mod_5_l1351_135113


namespace NUMINAMATH_GPT_molecular_weight_CaO_l1351_135106

def atomic_weight_Ca : Float := 40.08
def atomic_weight_O : Float := 16.00

def molecular_weight (atoms : List (String × Float)) : Float :=
  atoms.foldr (fun (_, w) acc => w + acc) 0.0

theorem molecular_weight_CaO :
  molecular_weight [("Ca", atomic_weight_Ca), ("O", atomic_weight_O)] = 56.08 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_CaO_l1351_135106


namespace NUMINAMATH_GPT_arithmetic_sqrt_of_25_l1351_135154

theorem arithmetic_sqrt_of_25 : ∃ (x : ℝ), x^2 = 25 ∧ x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sqrt_of_25_l1351_135154


namespace NUMINAMATH_GPT_number_of_sides_l1351_135190

theorem number_of_sides (n : ℕ) : 
  (2 / 9) * (n - 2) * 180 = 360 → n = 11 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_sides_l1351_135190


namespace NUMINAMATH_GPT_prove_correct_options_l1351_135101

theorem prove_correct_options (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 2) :
  (min (((1 : ℝ) / x) + (1 / y)) = 2) ∧
  (max (x * y) = 1) ∧
  (min (x^2 + y^2) = 2) ∧
  (max (x * (y + 1)) = (9 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_prove_correct_options_l1351_135101


namespace NUMINAMATH_GPT_max_min_value_f_l1351_135165

theorem max_min_value_f (x m : ℝ) : ∃ m : ℝ, (∀ x : ℝ, x^2 - 2*m*x + 8*m + 4 ≥ -m^2 + 8*m + 4) ∧ (∀ n : ℝ, -n^2 + 8*n + 4 ≤ 20) :=
  sorry

end NUMINAMATH_GPT_max_min_value_f_l1351_135165


namespace NUMINAMATH_GPT_steel_strength_value_l1351_135170

theorem steel_strength_value 
  (s : ℝ) 
  (condition: s = 4.6 * 10^8) : 
  s = 460000000 := 
by sorry

end NUMINAMATH_GPT_steel_strength_value_l1351_135170


namespace NUMINAMATH_GPT_constant_subsequence_exists_l1351_135157

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem constant_subsequence_exists (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (f : ℕ → ℕ) (c : ℕ), (∀ n m, n < m → f n < f m) ∧ (∀ n, sum_of_digits (⌊a * ↑(f n) + b⌋₊) = c) :=
sorry

end NUMINAMATH_GPT_constant_subsequence_exists_l1351_135157


namespace NUMINAMATH_GPT_tan_angle_identity_l1351_135160

open Real

theorem tan_angle_identity (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : sin β / cos β = (1 + cos (2 * α)) / (2 * cos α + sin (2 * α))) :
  tan (α + 2 * β + π / 4) = -1 := 
sorry

end NUMINAMATH_GPT_tan_angle_identity_l1351_135160


namespace NUMINAMATH_GPT_xy_square_difference_l1351_135193

variable (x y : ℚ)

theorem xy_square_difference (h1 : x + y = 8/15) (h2 : x - y = 1/45) : 
  x^2 - y^2 = 8/675 := by
  sorry

end NUMINAMATH_GPT_xy_square_difference_l1351_135193


namespace NUMINAMATH_GPT_expression_for_f_l1351_135110

variable {R : Type*} [CommRing R]

def f (x : R) : R := sorry

theorem expression_for_f (x : R) :
  (f (x-1) = x^2 + 4*x - 5) → (f x = x^2 + 6*x) := by
  sorry

end NUMINAMATH_GPT_expression_for_f_l1351_135110


namespace NUMINAMATH_GPT_gwen_walked_time_l1351_135120

-- Definition of given conditions
def time_jogged : ℕ := 15
def ratio_jogged_to_walked (j w : ℕ) : Prop := j * 3 = w * 5

-- Definition to state the exact time walked with given ratio
theorem gwen_walked_time (j w : ℕ) (h1 : j = time_jogged) (h2 : ratio_jogged_to_walked j w) : w = 9 :=
by
  sorry

end NUMINAMATH_GPT_gwen_walked_time_l1351_135120


namespace NUMINAMATH_GPT_trigonometric_identity_l1351_135130

noncomputable def cos190 := Real.cos (190 * Real.pi / 180)
noncomputable def sin290 := Real.sin (290 * Real.pi / 180)
noncomputable def cos40 := Real.cos (40 * Real.pi / 180)
noncomputable def tan10 := Real.tan (10 * Real.pi / 180)

theorem trigonometric_identity :
  (cos190 * (1 + Real.sqrt 3 * tan10)) / (sin290 * Real.sqrt (1 - cos40)) = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1351_135130


namespace NUMINAMATH_GPT_cost_of_pencils_and_notebooks_l1351_135126

variable (p n : ℝ)

theorem cost_of_pencils_and_notebooks 
  (h1 : 9 * p + 10 * n = 5.06) 
  (h2 : 6 * p + 4 * n = 2.42) :
  20 * p + 14 * n = 8.31 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_pencils_and_notebooks_l1351_135126


namespace NUMINAMATH_GPT_total_spears_is_78_l1351_135188

-- Define the spear production rates for each type of wood
def spears_from_sapling := 3
def spears_from_log := 9
def spears_from_bundle := 7
def spears_from_trunk := 15

-- Define the quantity of each type of wood
def saplings := 6
def logs := 1
def bundles := 3
def trunks := 2

-- Prove that the total number of spears is 78
theorem total_spears_is_78 : (saplings * spears_from_sapling) + (logs * spears_from_log) + (bundles * spears_from_bundle) + (trunks * spears_from_trunk) = 78 :=
by 
  -- Calculation can be filled here
  sorry

end NUMINAMATH_GPT_total_spears_is_78_l1351_135188


namespace NUMINAMATH_GPT_cost_of_adult_ticket_l1351_135143

def cost_of_child_ticket : ℝ := 3.50
def total_tickets : ℕ := 21
def total_cost : ℝ := 83.50
def adult_tickets : ℕ := 5

theorem cost_of_adult_ticket
  (A : ℝ)
  (h : 5 * A + 16 * cost_of_child_ticket = total_cost) :
  A = 5.50 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_adult_ticket_l1351_135143


namespace NUMINAMATH_GPT_solve_system_equation_152_l1351_135135

theorem solve_system_equation_152 (x y z a b c : ℝ)
  (h1 : x * y - 2 * y - 3 * x = 0)
  (h2 : y * z - 3 * z - 5 * y = 0)
  (h3 : x * z - 5 * x - 2 * z = 0)
  (h4 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h5 : x = a)
  (h6 : y = b)
  (h7 : z = c) :
  a^2 + b^2 + c^2 = 152 := by
  sorry

end NUMINAMATH_GPT_solve_system_equation_152_l1351_135135


namespace NUMINAMATH_GPT_symmetric_points_x_axis_l1351_135163

theorem symmetric_points_x_axis (m n : ℤ) :
  (-4, m - 3) = (2 * n, -1) → (m = 2 ∧ n = -2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_x_axis_l1351_135163


namespace NUMINAMATH_GPT_min_value_expression_l1351_135181

theorem min_value_expression (x : ℚ) : ∃ x : ℚ, (2 * x - 5)^2 + 18 = 18 :=
by {
  use 2.5,
  sorry
}

end NUMINAMATH_GPT_min_value_expression_l1351_135181


namespace NUMINAMATH_GPT_percent_non_sugar_l1351_135118

-- Definitions based on the conditions in the problem.
def pie_weight : ℕ := 200
def sugar_weight : ℕ := 50

-- Statement of the proof problem.
theorem percent_non_sugar : ((pie_weight - sugar_weight) * 100) / pie_weight = 75 :=
by
  sorry

end NUMINAMATH_GPT_percent_non_sugar_l1351_135118


namespace NUMINAMATH_GPT_trig_identity_l1351_135198

open Real

theorem trig_identity (α : ℝ) (h : tan α = 2) :
  2 * cos (2 * α) + 3 * sin (2 * α) - sin (α) ^ 2 = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_trig_identity_l1351_135198


namespace NUMINAMATH_GPT_greatest_value_of_x_for_equation_l1351_135152

theorem greatest_value_of_x_for_equation :
  ∃ x : ℝ, (4 * x - 5) ≠ 0 ∧ ((5 * x - 20) / (4 * x - 5)) ^ 2 + ((5 * x - 20) / (4 * x - 5)) = 18 ∧ x = 50 / 29 :=
sorry

end NUMINAMATH_GPT_greatest_value_of_x_for_equation_l1351_135152


namespace NUMINAMATH_GPT_at_least_one_term_le_one_l1351_135147

theorem at_least_one_term_le_one
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) :
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
  sorry

end NUMINAMATH_GPT_at_least_one_term_le_one_l1351_135147


namespace NUMINAMATH_GPT_find_constants_l1351_135167

open Matrix 

noncomputable def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 2, 1; 2, 0, 2; 1, 2, 0]

theorem find_constants :
  let s := (-10 : ℤ)
  let t := (-8 : ℤ)
  let u := (-36 : ℤ)
  B^3 + s • (B^2) + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := sorry

end NUMINAMATH_GPT_find_constants_l1351_135167


namespace NUMINAMATH_GPT_sum_of_f_greater_than_zero_l1351_135137

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem sum_of_f_greater_than_zero 
  (a b c : ℝ) 
  (h1 : a + b > 0) 
  (h2 : b + c > 0) 
  (h3 : c + a > 0) : 
  f a + f b + f c > 0 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_f_greater_than_zero_l1351_135137


namespace NUMINAMATH_GPT_rational_solutions_zero_l1351_135108

theorem rational_solutions_zero (x y z : ℚ) (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by 
  sorry

end NUMINAMATH_GPT_rational_solutions_zero_l1351_135108


namespace NUMINAMATH_GPT_arithmetic_sum_S8_l1351_135102

theorem arithmetic_sum_S8 (S : ℕ → ℕ)
  (h_arithmetic : ∀ n, S (n + 1) - S n = S 1 - S 0)
  (h_positive : ∀ n, S n > 0)
  (h_S4 : S 4 = 10)
  (h_S12 : S 12 = 130) : 
  S 8 = 40 :=
sorry

end NUMINAMATH_GPT_arithmetic_sum_S8_l1351_135102


namespace NUMINAMATH_GPT_xyz_leq_36_l1351_135122

theorem xyz_leq_36 {x y z : ℝ} 
    (hx0 : x > 0) (hy0 : y > 0) (hz0 : z > 0) 
    (hx2 : x ≤ 2) (hy3 : y ≤ 3) 
    (hxyz_sum : x + y + z = 11) : 
    x * y * z ≤ 36 := 
by
  sorry

end NUMINAMATH_GPT_xyz_leq_36_l1351_135122


namespace NUMINAMATH_GPT_people_left_first_hour_l1351_135195

theorem people_left_first_hour 
  (X : ℕ)
  (h1 : X ≥ 0)
  (h2 : 94 - X + 18 - 9 = 76) :
  X = 27 := 
sorry

end NUMINAMATH_GPT_people_left_first_hour_l1351_135195


namespace NUMINAMATH_GPT_leah_coins_value_l1351_135125

theorem leah_coins_value :
  ∃ (p n d : ℕ), 
    p + n + d = 20 ∧
    p = n ∧
    p = d + 4 ∧
    1 * p + 5 * n + 10 * d = 88 :=
by
  sorry

end NUMINAMATH_GPT_leah_coins_value_l1351_135125


namespace NUMINAMATH_GPT_derivative_f_l1351_135159

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.exp (Real.sin x))

theorem derivative_f (x : ℝ) : deriv f x = ((Real.cos x)^2 - Real.sin x) * (Real.exp (Real.sin x)) :=
by
  sorry

end NUMINAMATH_GPT_derivative_f_l1351_135159


namespace NUMINAMATH_GPT_area_triangle_AMC_l1351_135138

noncomputable def area_of_triangle_AMC (AB AD AM : ℝ) : ℝ :=
  if AB = 10 ∧ AD = 12 ∧ AM = 9 then
    (1 / 2) * AM * AB
  else 0

theorem area_triangle_AMC :
  ∀ (AB AD AM : ℝ), AB = 10 → AD = 12 → AM = 9 → area_of_triangle_AMC AB AD AM = 45 := by
  intros AB AD AM hAB hAD hAM
  simp [area_of_triangle_AMC, hAB, hAD, hAM]
  sorry

end NUMINAMATH_GPT_area_triangle_AMC_l1351_135138


namespace NUMINAMATH_GPT_triangle_side_length_l1351_135172

theorem triangle_side_length (a b c : ℝ) (h1 : a + b + c = 20)
  (h2 : (1 / 2) * b * c * (Real.sin (Real.pi / 3)) = 10 * Real.sqrt 3) : a = 7 :=
sorry

end NUMINAMATH_GPT_triangle_side_length_l1351_135172


namespace NUMINAMATH_GPT_crt_solution_l1351_135115

/-- Congruences from the conditions -/
def congruences : Prop :=
  ∃ x : ℤ, 
    (x % 2 = 1) ∧
    (x % 3 = 2) ∧
    (x % 5 = 3) ∧
    (x % 7 = 4)

/-- The target result from the Chinese Remainder Theorem -/
def target_result : Prop :=
  ∃ x : ℤ, 
    (x % 210 = 53)

/-- The proof problem stating that the given conditions imply the target result -/
theorem crt_solution : congruences → target_result :=
by
  sorry

end NUMINAMATH_GPT_crt_solution_l1351_135115


namespace NUMINAMATH_GPT_cumulative_distribution_F1_cumulative_distribution_F2_joint_density_joint_cumulative_distribution_l1351_135153

noncomputable def p1 (x : ℝ) : ℝ :=
  if x < -1 ∨ x > 1 then 0 else 0.5

noncomputable def p2 (y : ℝ) : ℝ :=
  if y < 0 ∨ y > 2 then 0 else 0.5

noncomputable def F1 (x : ℝ) : ℝ :=
  if x ≤ -1 then 0 else if x ≤ 1 then 0.5 * (x + 1) else 1

noncomputable def F2 (y : ℝ) : ℝ :=
  if y ≤ 0 then 0 else if y ≤ 2 then 0.5 * y else 1

noncomputable def p (x : ℝ) (y : ℝ) : ℝ :=
  if (x < -1 ∨ x > 1 ∨ y < 0 ∨ y > 2) then 0 else 0.25

noncomputable def F (x : ℝ) (y : ℝ) : ℝ :=
  if x ≤ -1 ∨ y ≤ 0 then 0
  else if x ≤ 1 ∧ y ≤ 2 then 0.25 * (x + 1) * y 
  else if x ≤ 1 ∧ y > 2 then 0.5 * (x + 1)
  else if x > 1 ∧ y ≤ 2 then 0.5 * y
  else 1

theorem cumulative_distribution_F1 (x : ℝ) : 
  F1 x = if x ≤ -1 then 0 else if x ≤ 1 then 0.5 * (x + 1) else 1 := by sorry

theorem cumulative_distribution_F2 (y : ℝ) : 
  F2 y = if y ≤ 0 then 0 else if y ≤ 2 then 0.5 * y else 1 := by sorry

theorem joint_density (x : ℝ) (y : ℝ) : 
  p x y = if (x < -1 ∨ x > 1 ∨ y < 0 ∨ y > 2) then 0 else 0.25 := by sorry

theorem joint_cumulative_distribution (x : ℝ) (y : ℝ) : 
  F x y = if x ≤ -1 ∨ y ≤ 0 then 0
          else if x ≤ 1 ∧ y ≤ 2 then 0.25 * (x + 1) * y
          else if x ≤ 1 ∧ y > 2 then 0.5 * (x + 1)
          else if x > 1 ∧ y ≤ 2 then 0.5 * y
          else 1 := by sorry

end NUMINAMATH_GPT_cumulative_distribution_F1_cumulative_distribution_F2_joint_density_joint_cumulative_distribution_l1351_135153


namespace NUMINAMATH_GPT_intersection_A_B_l1351_135192

theorem intersection_A_B :
  let A := {1, 3, 5, 7}
  let B := {x | x^2 - 2 * x - 5 ≤ 0}
  A ∩ B = {1, 3} := by
sorry

end NUMINAMATH_GPT_intersection_A_B_l1351_135192


namespace NUMINAMATH_GPT_purple_ring_weight_l1351_135134

def orange_ring_weight : ℝ := 0.08
def white_ring_weight : ℝ := 0.42
def total_weight : ℝ := 0.83

theorem purple_ring_weight : 
  ∃ (purple_ring_weight : ℝ), purple_ring_weight = total_weight - (orange_ring_weight + white_ring_weight) := 
  by
  use 0.33
  sorry

end NUMINAMATH_GPT_purple_ring_weight_l1351_135134


namespace NUMINAMATH_GPT_final_state_probability_l1351_135144

-- Define the initial state and conditions of the problem
structure GameState where
  raashan : ℕ
  sylvia : ℕ
  ted : ℕ
  uma : ℕ

-- Conditions: each player starts with $2, and the game evolves over 500 rounds
def initial_state : GameState :=
  { raashan := 2, sylvia := 2, ted := 2, uma := 2 }

def valid_statements (state : GameState) : Prop :=
  state.raashan = 2 ∧ state.sylvia = 2 ∧ state.ted = 2 ∧ state.uma = 2

-- Final theorem statement
theorem final_state_probability :
  let states := 500 -- representing the number of rounds
  -- proof outline implies that after the games have properly transitioned and bank interactions, the probability is calculated
  -- state after the transitions
  ∃ (prob : ℚ), prob = 1/4 ∧ valid_statements initial_state :=
  sorry

end NUMINAMATH_GPT_final_state_probability_l1351_135144


namespace NUMINAMATH_GPT_symmetric_points_y_axis_l1351_135105

theorem symmetric_points_y_axis (a b : ℤ) 
  (h1 : a + 1 = 2) 
  (h2 : b + 2 = 3) : 
  a + b = 2 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_y_axis_l1351_135105


namespace NUMINAMATH_GPT_inequality_solution_set_l1351_135175

theorem inequality_solution_set (x : ℝ) : (x + 2) * (x - 1) > 0 ↔ x < -2 ∨ x > 1 := sorry

end NUMINAMATH_GPT_inequality_solution_set_l1351_135175


namespace NUMINAMATH_GPT_correct_multiplication_l1351_135148

theorem correct_multiplication (n : ℕ) (wrong_answer correct_answer : ℕ) 
    (h1 : wrong_answer = 559981)
    (h2 : correct_answer = 987 * n)
    (h3 : ∃ (x y : ℕ), correct_answer = 500000 + x + 901 + y ∧ x ≠ 98 ∧ y ≠ 98 ∧ (wrong_answer - correct_answer) % 10 = 0) :
    correct_answer = 559989 :=
by
  sorry

end NUMINAMATH_GPT_correct_multiplication_l1351_135148


namespace NUMINAMATH_GPT_count_paths_to_form_2005_l1351_135187

/-- Define the structure of a circle label. -/
inductive CircleLabel
| two
| zero
| five

open CircleLabel

/-- Define the number of possible moves from each circle. -/
def moves_from_two : Nat := 6
def moves_from_zero_to_zero : Nat := 2
def moves_from_zero_to_five : Nat := 3

/-- Define the total number of paths to form 2005. -/
def total_paths : Nat := moves_from_two * moves_from_zero_to_zero * moves_from_zero_to_five

/-- The proof statement: The total number of different paths to form the number 2005 is 36. -/
theorem count_paths_to_form_2005 : total_paths = 36 :=
by
  sorry

end NUMINAMATH_GPT_count_paths_to_form_2005_l1351_135187


namespace NUMINAMATH_GPT_potion_kits_needed_l1351_135155

-- Definitions
def num_spellbooks := 5
def cost_spellbook_gold := 5
def cost_potion_kit_silver := 20
def num_owls := 1
def cost_owl_gold := 28
def silver_per_gold := 9
def total_silver := 537

-- Prove that Harry needs to buy 3 potion kits.
def Harry_needs_to_buy : Prop :=
  let cost_spellbooks_silver := num_spellbooks * cost_spellbook_gold * silver_per_gold
  let cost_owl_silver := num_owls * cost_owl_gold * silver_per_gold
  let total_cost_silver := cost_spellbooks_silver + cost_owl_silver
  let remaining_silver := total_silver - total_cost_silver
  let num_potion_kits := remaining_silver / cost_potion_kit_silver
  num_potion_kits = 3

theorem potion_kits_needed : Harry_needs_to_buy :=
  sorry

end NUMINAMATH_GPT_potion_kits_needed_l1351_135155


namespace NUMINAMATH_GPT_gcd_2720_1530_l1351_135103

theorem gcd_2720_1530 : Nat.gcd 2720 1530 = 170 := by
  sorry

end NUMINAMATH_GPT_gcd_2720_1530_l1351_135103


namespace NUMINAMATH_GPT_iron_aluminum_weight_difference_l1351_135133

theorem iron_aluminum_weight_difference :
  let iron_weight := 11.17
  let aluminum_weight := 0.83
  iron_weight - aluminum_weight = 10.34 :=
by
  sorry

end NUMINAMATH_GPT_iron_aluminum_weight_difference_l1351_135133


namespace NUMINAMATH_GPT_sam_quarters_mowing_lawns_l1351_135158

-- Definitions based on the given conditions
def pennies : ℕ := 9
def total_amount_dollars : ℝ := 1.84
def penny_value_dollars : ℝ := 0.01
def quarter_value_dollars : ℝ := 0.25

-- Theorem statement that Sam got 7 quarters given the conditions
theorem sam_quarters_mowing_lawns : 
  (total_amount_dollars - pennies * penny_value_dollars) / quarter_value_dollars = 7 := by
  sorry

end NUMINAMATH_GPT_sam_quarters_mowing_lawns_l1351_135158


namespace NUMINAMATH_GPT_smallest_x_for_div_by9_l1351_135146

-- Define the digit sum of the number 761*829 with a placeholder * for x
def digit_sum_with_x (x : Nat) : Nat :=
  7 + 6 + 1 + x + 8 + 2 + 9

-- State the theorem to prove the smallest value of x makes the sum divisible by 9
theorem smallest_x_for_div_by9 : ∃ x : Nat, digit_sum_with_x x % 9 = 0 ∧ (∀ y : Nat, y < x → digit_sum_with_x y % 9 ≠ 0) :=
sorry

end NUMINAMATH_GPT_smallest_x_for_div_by9_l1351_135146


namespace NUMINAMATH_GPT_binomial_coefficient_19_13_l1351_135131

theorem binomial_coefficient_19_13 
  (h1 : Nat.choose 20 13 = 77520) 
  (h2 : Nat.choose 20 14 = 38760) 
  (h3 : Nat.choose 18 13 = 18564) :
  Nat.choose 19 13 = 37128 := 
sorry

end NUMINAMATH_GPT_binomial_coefficient_19_13_l1351_135131


namespace NUMINAMATH_GPT_george_choices_l1351_135128

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- State the theorem to prove the number of ways to choose 3 out of 9 colors is 84
theorem george_choices : binomial 9 3 = 84 := by
  sorry

end NUMINAMATH_GPT_george_choices_l1351_135128


namespace NUMINAMATH_GPT_profit_maximization_l1351_135156

-- Define the conditions 
variable (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5)

-- Expression for yield ω
noncomputable def yield (x : ℝ) : ℝ := 4 - (3 / (x + 1))

-- Expression for profit function L(x)
noncomputable def profit (x : ℝ) : ℝ := 16 * yield x - x - 2 * x

-- Theorem stating the profit function expression and its maximum
theorem profit_maximization (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5) :
  profit x = 64 - 48 / (x + 1) - 3 * x ∧ 
  (∀ x₀, 0 ≤ x₀ ∧ x₀ ≤ 5 → profit x₀ ≤ profit 3) :=
sorry

end NUMINAMATH_GPT_profit_maximization_l1351_135156


namespace NUMINAMATH_GPT_quadratic_radical_type_l1351_135116

-- Problem statement: Given that sqrt(2a + 1) is a simplest quadratic radical and the same type as sqrt(48), prove that a = 1.

theorem quadratic_radical_type (a : ℝ) (h1 : ((2 * a) + 1) = 3) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_radical_type_l1351_135116


namespace NUMINAMATH_GPT_divisible_by_condition_a_l1351_135174

theorem divisible_by_condition_a (a b c k : ℤ) 
  (h : ∃ k : ℤ, a - b * c = (10 * c + 1) * k) : 
  ∃ k : ℤ, 10 * a + b = (10 * c + 1) * k :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_condition_a_l1351_135174


namespace NUMINAMATH_GPT_paint_cost_for_flag_l1351_135119

noncomputable def flag_width : ℕ := 12
noncomputable def flag_height : ℕ := 10
noncomputable def paint_cost_per_quart : ℝ := 3.5
noncomputable def coverage_per_quart : ℕ := 4

theorem paint_cost_for_flag : (flag_width * flag_height * 2 / coverage_per_quart : ℝ) * paint_cost_per_quart = 210 := by
  sorry

end NUMINAMATH_GPT_paint_cost_for_flag_l1351_135119


namespace NUMINAMATH_GPT_range_of_m_in_third_quadrant_l1351_135151

theorem range_of_m_in_third_quadrant (m : ℝ) : (1 - (1/3) * m < 0) ∧ (m - 5 < 0) ↔ (3 < m ∧ m < 5) := 
by 
  intros
  sorry

end NUMINAMATH_GPT_range_of_m_in_third_quadrant_l1351_135151


namespace NUMINAMATH_GPT_range_of_m_l1351_135161

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 / x) + (1 / y) = 1) (h2 : x + 2 * y > m^2 + 2 * m) : -4 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1351_135161


namespace NUMINAMATH_GPT_xyz_inequality_l1351_135199

theorem xyz_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end NUMINAMATH_GPT_xyz_inequality_l1351_135199


namespace NUMINAMATH_GPT_expected_value_is_correct_l1351_135194

-- Define the monetary outcomes associated with each side
def monetaryOutcome (X : String) : ℚ :=
  if X = "A" then 2 else 
  if X = "B" then -4 else 
  if X = "C" then 6 else 
  0

-- Define the probabilities associated with each side
def probability (X : String) : ℚ :=
  if X = "A" then 1/3 else 
  if X = "B" then 1/2 else 
  if X = "C" then 1/6 else 
  0

-- Compute the expected value
def expectedMonetaryOutcome : ℚ := (probability "A" * monetaryOutcome "A") 
                                + (probability "B" * monetaryOutcome "B") 
                                + (probability "C" * monetaryOutcome "C")

theorem expected_value_is_correct : 
  expectedMonetaryOutcome = -2/3 := by
  sorry

end NUMINAMATH_GPT_expected_value_is_correct_l1351_135194


namespace NUMINAMATH_GPT_smallest_k_l1351_135184

theorem smallest_k (k : ℕ) : 
  (k > 0 ∧ (k*(k+1)*(2*k+1)/6) % 400 = 0) → k = 800 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_l1351_135184


namespace NUMINAMATH_GPT_december_revenue_times_average_l1351_135114

def revenue_in_december_is_multiple_of_average_revenue (R_N R_J R_D : ℝ) : Prop :=
  R_N = (3/5) * R_D ∧    -- Condition: November's revenue is 3/5 of December's revenue
  R_J = (1/3) * R_N ∧    -- Condition: January's revenue is 1/3 of November's revenue
  R_D = 2.5 * ((R_N + R_J) / 2)   -- Question: December's revenue is 2.5 times the average of November's and January's revenue

theorem december_revenue_times_average (R_N R_J R_D : ℝ) :
  revenue_in_december_is_multiple_of_average_revenue R_N R_J R_D :=
by
  -- adding sorry to skip the proof
  sorry

end NUMINAMATH_GPT_december_revenue_times_average_l1351_135114


namespace NUMINAMATH_GPT_positive_number_sum_square_l1351_135111

theorem positive_number_sum_square (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end NUMINAMATH_GPT_positive_number_sum_square_l1351_135111


namespace NUMINAMATH_GPT_min_expression_value_l1351_135178

theorem min_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (z : ℝ) (h3 : x^2 + y^2 = z) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) = -2040200 :=
  sorry

end NUMINAMATH_GPT_min_expression_value_l1351_135178


namespace NUMINAMATH_GPT_y_not_directly_nor_inversely_proportional_l1351_135132

theorem y_not_directly_nor_inversely_proportional (x y : ℝ) :
  (∃ k : ℝ, x + y = 0 ∧ y = k * x) ∨
  (∃ k : ℝ, 3 * x * y = 10 ∧ x * y = k) ∨
  (∃ k : ℝ, x = 5 * y ∧ x = k * y) ∨
  (∃ k : ℝ, (y = 10 - x^2 - 3 * x) ∧ y ≠ k * x ∧ y * x ≠ k) ∨
  (∃ k : ℝ, x / y = Real.sqrt 3 ∧ x = k * y)
  → (∃ k : ℝ, y = 10 - x^2 - 3 * x ∧ y ≠ k * x ∧ y * x ≠ k) :=
by
  sorry

end NUMINAMATH_GPT_y_not_directly_nor_inversely_proportional_l1351_135132


namespace NUMINAMATH_GPT_ratio_flowers_l1351_135177

theorem ratio_flowers (flowers_monday flowers_tuesday flowers_week total_flowers flowers_friday : ℕ)
    (h_monday : flowers_monday = 4)
    (h_tuesday : flowers_tuesday = 8)
    (h_total : total_flowers = 20)
    (h_week : total_flowers = flowers_monday + flowers_tuesday + flowers_friday) :
    flowers_friday / flowers_monday = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_flowers_l1351_135177


namespace NUMINAMATH_GPT_root_equation_solution_l1351_135129

theorem root_equation_solution (a : ℝ) (h : 3 * a^2 - 5 * a - 2 = 0) : 6 * a^2 - 10 * a = 4 :=
by 
  sorry

end NUMINAMATH_GPT_root_equation_solution_l1351_135129


namespace NUMINAMATH_GPT_girls_attending_ball_l1351_135179

theorem girls_attending_ball (g b : ℕ) 
    (h1 : g + b = 1500) 
    (h2 : 3 * g / 4 + 2 * b / 3 = 900) : 
    g = 1200 ∧ 3 * 1200 / 4 = 900 := 
by
  sorry

end NUMINAMATH_GPT_girls_attending_ball_l1351_135179


namespace NUMINAMATH_GPT_sean_div_julie_eq_two_l1351_135107

def sum_n (n : ℕ) := n * (n + 1) / 2

def sean_sum := 2 * sum_n 500

def julie_sum := sum_n 500

theorem sean_div_julie_eq_two : sean_sum / julie_sum = 2 := 
by sorry

end NUMINAMATH_GPT_sean_div_julie_eq_two_l1351_135107


namespace NUMINAMATH_GPT_ball_travel_distance_five_hits_l1351_135176

def total_distance_traveled (h₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  let descents := List.range (n + 1) |>.map (λ i => h₀ * r ^ i)
  let ascents := List.range n |>.map (λ i => h₀ * r ^ (i + 1))
  (descents.sum + ascents.sum)

theorem ball_travel_distance_five_hits :
  total_distance_traveled 120 (3 / 4) 5 = 612.1875 :=
by
  sorry

end NUMINAMATH_GPT_ball_travel_distance_five_hits_l1351_135176


namespace NUMINAMATH_GPT_three_segments_form_triangle_l1351_135139

theorem three_segments_form_triangle
    (lengths : Fin 10 → ℕ)
    (h1 : lengths 0 = 1)
    (h2 : lengths 1 = 1)
    (h3 : lengths 9 = 50) :
    ∃ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    lengths i + lengths j > lengths k ∧ 
    lengths i + lengths k > lengths j ∧ 
    lengths j + lengths k > lengths i := 
sorry

end NUMINAMATH_GPT_three_segments_form_triangle_l1351_135139


namespace NUMINAMATH_GPT_smaller_angle_at_3_pm_l1351_135169

-- Define the condition for minute hand position at 3:00 p.m.
def minute_hand_position_at_3_pm_deg : ℝ := 0

-- Define the condition for hour hand position at 3:00 p.m.
def hour_hand_position_at_3_pm_deg : ℝ := 90

-- Define the angle between the minute hand and hour hand
def angle_between_hands (minute_deg hour_deg : ℝ) : ℝ :=
  abs (hour_deg - minute_deg)

-- The main theorem we need to prove
theorem smaller_angle_at_3_pm :
  angle_between_hands minute_hand_position_at_3_pm_deg hour_hand_position_at_3_pm_deg = 90 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_at_3_pm_l1351_135169


namespace NUMINAMATH_GPT_min_benches_l1351_135180
-- Import the necessary library

-- Defining the problem in Lean statement
theorem min_benches (N : ℕ) :
  (∀ a c : ℕ, (8 * N = a) ∧ (12 * N = c) ∧ (a = c)) → N = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_benches_l1351_135180


namespace NUMINAMATH_GPT_capital_growth_rate_l1351_135196

theorem capital_growth_rate
  (loan_amount : ℝ) (interest_rate : ℝ) (repayment_period : ℝ) (surplus : ℝ) (growth_rate : ℝ) :
  loan_amount = 2000000 ∧ interest_rate = 0.08 ∧ repayment_period = 2 ∧ surplus = 720000 ∧
  (loan_amount * (1 + growth_rate)^repayment_period = loan_amount * (1 + interest_rate) + surplus) →
  growth_rate = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_capital_growth_rate_l1351_135196


namespace NUMINAMATH_GPT_quadratic_root_exists_l1351_135182

theorem quadratic_root_exists (a b c : ℝ) : 
  ∃ x : ℝ, (a * x^2 + 2 * b * x + c = 0) ∨ (b * x^2 + 2 * c * x + a = 0) ∨ (c * x^2 + 2 * a * x + b = 0) :=
by sorry

end NUMINAMATH_GPT_quadratic_root_exists_l1351_135182


namespace NUMINAMATH_GPT_fully_factor_expression_l1351_135123

theorem fully_factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by
  -- pending proof, represented by sorry
  sorry

end NUMINAMATH_GPT_fully_factor_expression_l1351_135123


namespace NUMINAMATH_GPT_certain_number_approx_l1351_135136

theorem certain_number_approx (x : ℝ) : 213 * 16 = 3408 → x * 2.13 = 0.3408 → x = 0.1600 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_certain_number_approx_l1351_135136


namespace NUMINAMATH_GPT_solve_for_A_l1351_135173

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x^2 - 3 * B^2
def g (B x : ℝ) : ℝ := B * x^2

-- A Lean theorem that formalizes the given math problem.
theorem solve_for_A (A B : ℝ) (h₁ : B ≠ 0) (h₂ : f A B (g B 1) = 0) : A = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_A_l1351_135173


namespace NUMINAMATH_GPT_algebraic_inequality_l1351_135124

noncomputable def problem_statement (a b c d : ℝ) : Prop :=
  |a| > 1 ∧ |b| > 1 ∧ |c| > 1 ∧ |d| > 1 ∧
  a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0 →
  (1 / (a - 1)) + (1 / (b - 1)) + (1 / (c - 1)) + (1 / (d - 1)) > 0

theorem algebraic_inequality (a b c d : ℝ) :
  problem_statement a b c d :=
by
  sorry

end NUMINAMATH_GPT_algebraic_inequality_l1351_135124


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1351_135166

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 4) (h2 : b = 6) : 
  ∃ p, (p = 14 ∨ p = 16) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1351_135166
