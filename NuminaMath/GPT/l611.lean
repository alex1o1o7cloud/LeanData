import Mathlib

namespace NUMINAMATH_GPT_find_a_perpendicular_lines_l611_61110

theorem find_a_perpendicular_lines (a : ℝ) :
    (∀ x y : ℝ, a * x - y + 2 * a = 0 → (2 * a - 1) * x + a * y + a = 0) →
    (a = 0 ∨ a = 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_perpendicular_lines_l611_61110


namespace NUMINAMATH_GPT_bill_harry_combined_l611_61135

-- Definitions based on the given conditions
def sue_nuts := 48
def harry_nuts := 2 * sue_nuts
def bill_nuts := 6 * harry_nuts

-- The theorem we want to prove
theorem bill_harry_combined : bill_nuts + harry_nuts = 672 :=
by
  sorry

end NUMINAMATH_GPT_bill_harry_combined_l611_61135


namespace NUMINAMATH_GPT_find_f_prime_2_l611_61122

theorem find_f_prime_2 (a : ℝ) (f' : ℝ → ℝ) 
    (h1 : f' 1 = -5)
    (h2 : ∀ x, f' x = 3 * a * x^2 + 2 * f' 2 * x) : f' 2 = -4 := by
    sorry

end NUMINAMATH_GPT_find_f_prime_2_l611_61122


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequences_l611_61140

theorem sum_of_arithmetic_sequences (n : ℕ) (h : n ≠ 0) :
  (2 * n * (n + 3) = n * (n + 12)) → (n = 6) :=
by
  intro h_eq
  have h_nonzero : n ≠ 0 := h
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequences_l611_61140


namespace NUMINAMATH_GPT_cube_surface_area_increase_l611_61144

theorem cube_surface_area_increase (s : ℝ) :
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  (A_new - A_original) / A_original * 100 = 224 :=
by
  -- Definitions from the conditions
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  -- Rest of the proof; replace sorry with the actual proof
  sorry

end NUMINAMATH_GPT_cube_surface_area_increase_l611_61144


namespace NUMINAMATH_GPT_smallest_integer_proof_l611_61131

theorem smallest_integer_proof :
  ∃ (x : ℤ), x^2 = 3 * x + 75 ∧ ∀ (y : ℤ), y^2 = 3 * y + 75 → x ≤ y := 
  sorry

end NUMINAMATH_GPT_smallest_integer_proof_l611_61131


namespace NUMINAMATH_GPT_find_lowest_temperature_l611_61158

noncomputable def lowest_temperature 
(T1 T2 T3 T4 T5 : ℝ) : ℝ :=
if h : T1 + T2 + T3 + T4 + T5 = 200 ∧ max (max (max T1 T2) (max T3 T4)) T5 - min (min (min T1 T2) (min T3 T4)) T5 = 50 then
   min (min (min T1 T2) (min T3 T4)) T5
else 
  0

theorem find_lowest_temperature (T1 T2 T3 T4 T5 : ℝ) 
  (h_avg : T1 + T2 + T3 + T4 + T5 = 200)
  (h_range : max (max (max T1 T2) (max T3 T4)) T5 - min (min (min T1 T2) (min T3 T4)) T5 ≤ 50) : 
  lowest_temperature T1 T2 T3 T4 T5 = 30 := 
sorry

end NUMINAMATH_GPT_find_lowest_temperature_l611_61158


namespace NUMINAMATH_GPT_Gina_tip_is_5_percent_l611_61184

noncomputable def Gina_tip_percentage : ℝ := 5

theorem Gina_tip_is_5_percent (bill_amount : ℝ) (good_tipper_percentage : ℝ)
    (good_tipper_extra_tip_cents : ℝ) (good_tipper_tip : ℝ) 
    (Gina_tip_extra_cents : ℝ):
    bill_amount = 26 ∧
    good_tipper_percentage = 20 ∧
    Gina_tip_extra_cents = 390 ∧
    good_tipper_tip = (20 / 100) * 26 ∧
    Gina_tip_extra_cents = 390 ∧
    (Gina_tip_percentage / 100) * bill_amount + (Gina_tip_extra_cents / 100) = good_tipper_tip
    → Gina_tip_percentage = 5 :=
by
  sorry

end NUMINAMATH_GPT_Gina_tip_is_5_percent_l611_61184


namespace NUMINAMATH_GPT_penny_half_dollar_same_probability_l611_61102

def probability_penny_half_dollar_same : ℚ :=
  1 / 2

theorem penny_half_dollar_same_probability :
  probability_penny_half_dollar_same = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_penny_half_dollar_same_probability_l611_61102


namespace NUMINAMATH_GPT_total_get_well_cards_l611_61127

def dozens_to_cards (d : ℕ) : ℕ := d * 12
def hundreds_to_cards (h : ℕ) : ℕ := h * 100

theorem total_get_well_cards 
  (d_hospital : ℕ) (h_hospital : ℕ)
  (d_home : ℕ) (h_home : ℕ) :
  d_hospital = 25 ∧ h_hospital = 7 ∧ d_home = 39 ∧ h_home = 3 →
  (dozens_to_cards d_hospital + hundreds_to_cards h_hospital +
   dozens_to_cards d_home + hundreds_to_cards h_home) = 1768 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_get_well_cards_l611_61127


namespace NUMINAMATH_GPT_side_ratio_triangle_square_pentagon_l611_61162

-- Define the conditions
def perimeter_triangle (t : ℝ) := 3 * t = 18
def perimeter_square (s : ℝ) := 4 * s = 16
def perimeter_pentagon (p : ℝ) := 5 * p = 20

-- Statement to be proved
theorem side_ratio_triangle_square_pentagon 
  (t s p : ℝ)
  (ht : perimeter_triangle t)
  (hs : perimeter_square s)
  (hp : perimeter_pentagon p) : 
  (t / s = 3 / 2) ∧ (t / p = 3 / 2) := 
sorry

end NUMINAMATH_GPT_side_ratio_triangle_square_pentagon_l611_61162


namespace NUMINAMATH_GPT_find_bullet_l611_61136

theorem find_bullet (x y : ℝ) (h₁ : 3 * x + y = 8) (h₂ : y = -1) : 2 * x - y = 7 :=
sorry

end NUMINAMATH_GPT_find_bullet_l611_61136


namespace NUMINAMATH_GPT_inequality_solution_set_l611_61153

theorem inequality_solution_set (x : ℝ) :
  (3 * (x + 2) - x > 4) ∧ ((1 + 2 * x) / 3 ≥ x - 1) ↔ (-1 < x ∧ x ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l611_61153


namespace NUMINAMATH_GPT_quadratic_roots_l611_61198

theorem quadratic_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*m = 0) ∧ (x2^2 + 2*x2 + 2*m = 0)) ↔ m < 1/2 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_l611_61198


namespace NUMINAMATH_GPT_base_for_195₁₀_four_digit_even_final_digit_l611_61172

theorem base_for_195₁₀_four_digit_even_final_digit :
  ∃ b : ℕ, (b^3 ≤ 195 ∧ 195 < b^4) ∧ (∃ d : ℕ, 195 % b = d ∧ d % 2 = 0) ∧ b = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_base_for_195₁₀_four_digit_even_final_digit_l611_61172


namespace NUMINAMATH_GPT_stationery_sales_calculation_l611_61155

-- Definitions
def total_sales : ℕ := 120
def fabric_percentage : ℝ := 0.30
def jewelry_percentage : ℝ := 0.20
def knitting_percentage : ℝ := 0.15
def home_decor_percentage : ℝ := 0.10
def stationery_percentage := 1 - (fabric_percentage + jewelry_percentage + knitting_percentage + home_decor_percentage)
def stationery_sales := stationery_percentage * total_sales

-- Statement to prove
theorem stationery_sales_calculation : stationery_sales = 30 := by
  -- Providing the initial values and assumptions to the context
  have h1 : total_sales = 120 := rfl
  have h2 : fabric_percentage = 0.30 := rfl
  have h3 : jewelry_percentage = 0.20 := rfl
  have h4 : knitting_percentage = 0.15 := rfl
  have h5 : home_decor_percentage = 0.10 := rfl
  
  -- Calculating the stationery percentage and sales
  have h_stationery_percentage : stationery_percentage = 1 - (fabric_percentage + jewelry_percentage + knitting_percentage + home_decor_percentage) := rfl
  have h_stationery_sales : stationery_sales = stationery_percentage * total_sales := rfl

  -- The calculated value should match the proof's requirement
  sorry

end NUMINAMATH_GPT_stationery_sales_calculation_l611_61155


namespace NUMINAMATH_GPT_roots_square_sum_l611_61196

theorem roots_square_sum (r s t p q : ℝ) 
  (h1 : r + s + t = p) 
  (h2 : r * s + r * t + s * t = q) : 
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
by 
  -- proof skipped
  sorry

end NUMINAMATH_GPT_roots_square_sum_l611_61196


namespace NUMINAMATH_GPT_probability_three_girls_chosen_l611_61138

theorem probability_three_girls_chosen :
  let total_members := 15;
  let boys := 7;
  let girls := 8;
  let total_ways := Nat.choose total_members 3;
  let girls_ways := Nat.choose girls 3;
  total_ways = Nat.choose 15 3 ∧ girls_ways = Nat.choose 8 3 →
  (girls_ways : ℚ) / (total_ways : ℚ) = 8 / 65 := 
by  
  sorry

end NUMINAMATH_GPT_probability_three_girls_chosen_l611_61138


namespace NUMINAMATH_GPT_break_even_price_l611_61175

noncomputable def initial_investment : ℝ := 1500
noncomputable def cost_per_tshirt : ℝ := 3
noncomputable def num_tshirts_break_even : ℝ := 83
noncomputable def total_cost_equipment_tshirts : ℝ := initial_investment + (cost_per_tshirt * num_tshirts_break_even)
noncomputable def price_per_tshirt := total_cost_equipment_tshirts / num_tshirts_break_even

theorem break_even_price : price_per_tshirt = 21.07 := by
  sorry

end NUMINAMATH_GPT_break_even_price_l611_61175


namespace NUMINAMATH_GPT_part1_solution_set_a_eq_1_part2_range_of_values_a_l611_61177

def f (x a : ℝ) : ℝ := |(2 * x - a)| + |(x - 3 * a)|

theorem part1_solution_set_a_eq_1 :
  ∀ x : ℝ, f x 1 ≤ 4 ↔ 0 ≤ x ∧ x ≤ 2 :=
by sorry

theorem part2_range_of_values_a :
  ∀ a : ℝ, (∀ x : ℝ, f x a ≥ |(x - a / 2)| + a^2 + 1) ↔
    ((-2 : ℝ) ≤ a ∧ a ≤ -1 / 2) ∨ (1 / 2 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_GPT_part1_solution_set_a_eq_1_part2_range_of_values_a_l611_61177


namespace NUMINAMATH_GPT_simplified_expression_l611_61120

variable {x y : ℝ}

theorem simplified_expression 
  (P : ℝ := x^2 + y^2) 
  (Q : ℝ := x^2 - y^2) : 
  ( (P + 3 * Q) / (P - Q) - (P - 3 * Q) / (P + Q) ) = (2 * x^4 - y^4) / (x^2 * y^2) := 
  by sorry

end NUMINAMATH_GPT_simplified_expression_l611_61120


namespace NUMINAMATH_GPT_subtraction_solution_l611_61187

noncomputable def x : ℝ := 47.806

theorem subtraction_solution :
  (3889 : ℝ) + 12.808 - x = 3854.002 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_solution_l611_61187


namespace NUMINAMATH_GPT_compute_expression_equals_375_l611_61119

theorem compute_expression_equals_375 : 15 * (30 / 6) ^ 2 = 375 := 
by 
  have frac_simplified : 30 / 6 = 5 := by sorry
  have power_calculated : 5 ^ 2 = 25 := by sorry
  have final_result : 15 * 25 = 375 := by sorry
  sorry

end NUMINAMATH_GPT_compute_expression_equals_375_l611_61119


namespace NUMINAMATH_GPT_value_of_a_l611_61115

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_a (a : ℝ) (f_symmetric : ∀ x y : ℝ, y = f x ↔ -y = 2^(-x + a)) (sum_f_condition : f (-2) + f (-4) = 1) :
  a = 2 :=
sorry

end NUMINAMATH_GPT_value_of_a_l611_61115


namespace NUMINAMATH_GPT_selling_prices_maximize_profit_l611_61170

-- Definitions for the conditions
def total_items : ℕ := 200
def budget : ℤ := 5000
def cost_basketball : ℤ := 30
def cost_volleyball : ℤ := 24
def selling_price_ratio : ℚ := 3 / 2
def school_purchase_basketballs_value : ℤ := 1800
def school_purchase_volleyballs_value : ℤ := 1500
def basketballs_fewer_than_volleyballs : ℤ := 10

-- Part 1: Proof of selling prices
theorem selling_prices (x : ℚ) :
  (school_purchase_volleyballs_value / x - school_purchase_basketballs_value / (x * selling_price_ratio) = basketballs_fewer_than_volleyballs)
  → ∃ (basketball_price volleyball_price : ℚ), basketball_price = 45 ∧ volleyball_price = 30 :=
by
  sorry

-- Part 2: Proof of maximizing profit
theorem maximize_profit (a : ℕ) :
  (cost_basketball * a + cost_volleyball * (total_items - a) ≤ budget)
  → ∃ optimal_a : ℕ, (optimal_a = 33 ∧ total_items - optimal_a = 167) :=
by
  sorry

end NUMINAMATH_GPT_selling_prices_maximize_profit_l611_61170


namespace NUMINAMATH_GPT_smallest_perfect_square_5336100_l611_61100

def smallestPerfectSquareDivisibleBy (a b c d : Nat) (s : Nat) : Prop :=
  ∃ k : Nat, s = k * k ∧ s % a = 0 ∧ s % b = 0 ∧ s % c = 0 ∧ s % d = 0

theorem smallest_perfect_square_5336100 :
  smallestPerfectSquareDivisibleBy 6 14 22 30 5336100 :=
sorry

end NUMINAMATH_GPT_smallest_perfect_square_5336100_l611_61100


namespace NUMINAMATH_GPT_determinant_problem_l611_61103

theorem determinant_problem (a b c d : ℝ)
  (h : Matrix.det ![![a, b], ![c, d]] = 4) :
  Matrix.det ![![a, 5*a + 3*b], ![c, 5*c + 3*d]] = 12 := by
  sorry

end NUMINAMATH_GPT_determinant_problem_l611_61103


namespace NUMINAMATH_GPT_binom_2023_2_eq_l611_61186

theorem binom_2023_2_eq : Nat.choose 2023 2 = 2045323 := by
  sorry

end NUMINAMATH_GPT_binom_2023_2_eq_l611_61186


namespace NUMINAMATH_GPT_cost_of_fencing_l611_61129

noncomputable def fencingCost :=
  let π := 3.14159
  let diameter := 32
  let costPerMeter := 1.50
  let circumference := π * diameter
  let totalCost := costPerMeter * circumference
  totalCost

theorem cost_of_fencing :
  let roundedCost := (fencingCost).round
  roundedCost = 150.80 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_l611_61129


namespace NUMINAMATH_GPT_total_travel_ways_l611_61126

-- Define the number of car departures
def car_departures : ℕ := 3

-- Define the number of train departures
def train_departures : ℕ := 4

-- Define the number of ship departures
def ship_departures : ℕ := 2

-- The total number of ways to travel from location A to location B
def total_ways : ℕ := car_departures + train_departures + ship_departures

-- The theorem stating the total number of ways to travel given the conditions
theorem total_travel_ways :
  total_ways = 9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_travel_ways_l611_61126


namespace NUMINAMATH_GPT_product_segment_doubles_l611_61116

-- Define the problem conditions and proof statement in Lean.
theorem product_segment_doubles
  (a b e : ℝ)
  (d : ℝ := (a * b) / e)
  (e' : ℝ := e / 2)
  (d' : ℝ := (a * b) / e') :
  d' = 2 * d := 
  sorry

end NUMINAMATH_GPT_product_segment_doubles_l611_61116


namespace NUMINAMATH_GPT_find_uv_non_integer_l611_61167

def p (b : Fin 14 → ℚ) (x y : ℚ) : ℚ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3 + 
  b 10 * x^4 + b 11 * y^4 + b 12 * x^3 * y^2 + b 13 * y^3 * x^2

variables (b : Fin 14 → ℚ)
variables (u v : ℚ)

def zeros_at_specific_points :=
  p b 0 0 = 0 ∧ p b 1 0 = 0 ∧ p b (-1) 0 = 0 ∧
  p b 0 1 = 0 ∧ p b 0 (-1) = 0 ∧ p b 1 1 = 0 ∧
  p b (-1) (-1) = 0 ∧ p b 2 2 = 0 ∧ 
  p b 2 (-2) = 0 ∧ p b (-2) 2 = 0

theorem find_uv_non_integer
  (h : zeros_at_specific_points b) :
  p b (5/19) (16/19) = 0 :=
sorry

end NUMINAMATH_GPT_find_uv_non_integer_l611_61167


namespace NUMINAMATH_GPT_percent_germinated_is_31_l611_61192

-- Define given conditions
def seeds_first_plot : ℕ := 300
def seeds_second_plot : ℕ := 200
def germination_rate_first_plot : ℝ := 0.25
def germination_rate_second_plot : ℝ := 0.40

-- Calculate the number of germinated seeds in each plot
def germinated_first_plot : ℝ := germination_rate_first_plot * seeds_first_plot
def germinated_second_plot : ℝ := germination_rate_second_plot * seeds_second_plot

-- Calculate total number of seeds and total number of germinated seeds
def total_seeds : ℕ := seeds_first_plot + seeds_second_plot
def total_germinated : ℝ := germinated_first_plot + germinated_second_plot

-- Prove the percentage of the total number of seeds that germinated
theorem percent_germinated_is_31 :
  ((total_germinated / total_seeds) * 100) = 31 := 
by
  sorry

end NUMINAMATH_GPT_percent_germinated_is_31_l611_61192


namespace NUMINAMATH_GPT_max_sides_13_eq_13_max_sides_1950_eq_1950_l611_61128

noncomputable def max_sides (n : ℕ) : ℕ := n

theorem max_sides_13_eq_13 : max_sides 13 = 13 :=
by {
  sorry
}

theorem max_sides_1950_eq_1950 : max_sides 1950 = 1950 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_sides_13_eq_13_max_sides_1950_eq_1950_l611_61128


namespace NUMINAMATH_GPT_zach_cookies_total_l611_61176

theorem zach_cookies_total :
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  cookies_monday + cookies_tuesday + cookies_wednesday = 92 :=
by
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  sorry

end NUMINAMATH_GPT_zach_cookies_total_l611_61176


namespace NUMINAMATH_GPT_david_money_left_l611_61113

noncomputable section
open Real

def money_left_after_week (rate_per_hour : ℝ) (hours_per_day : ℝ) (days_per_week : ℝ) : ℝ :=
  let total_hours := hours_per_day * days_per_week
  let total_money := total_hours * rate_per_hour
  let money_after_shoes := total_money / 2
  let money_after_mom := (total_money - money_after_shoes) / 2
  total_money - money_after_shoes - money_after_mom

theorem david_money_left :
  money_left_after_week 14 2 7 = 49 := by simp [money_left_after_week]; norm_num

end NUMINAMATH_GPT_david_money_left_l611_61113


namespace NUMINAMATH_GPT_nadine_spent_money_l611_61107

theorem nadine_spent_money (table_cost : ℕ) (chair_cost : ℕ) (num_chairs : ℕ) 
    (h_table_cost : table_cost = 34) 
    (h_chair_cost : chair_cost = 11) 
    (h_num_chairs : num_chairs = 2) : 
    table_cost + num_chairs * chair_cost = 56 :=
by
  sorry

end NUMINAMATH_GPT_nadine_spent_money_l611_61107


namespace NUMINAMATH_GPT_total_amount_shared_l611_61178

noncomputable def z : ℝ := 300
noncomputable def y : ℝ := 1.2 * z
noncomputable def x : ℝ := 1.25 * y

theorem total_amount_shared (z y x : ℝ) (hz : z = 300) (hy : y = 1.2 * z) (hx : x = 1.25 * y) :
  x + y + z = 1110 :=
by
  simp [hx, hy, hz]
  -- Add intermediate steps here if necessary
  sorry

end NUMINAMATH_GPT_total_amount_shared_l611_61178


namespace NUMINAMATH_GPT_three_digit_numbers_satisfying_condition_l611_61185

theorem three_digit_numbers_satisfying_condition :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000) →
    ∃ (a b c : ℕ),
      (N = 100 * a + 10 * b + c) ∧ (N = 11 * (a^2 + b^2 + c^2)) 
    ↔ (N = 550 ∨ N = 803) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_satisfying_condition_l611_61185


namespace NUMINAMATH_GPT_solve_equation_l611_61117

theorem solve_equation (x : ℝ) : 
  16 * (x - 1) ^ 2 - 9 = 0 ↔ (x = 7 / 4 ∨ x = 1 / 4) := by
  sorry

end NUMINAMATH_GPT_solve_equation_l611_61117


namespace NUMINAMATH_GPT_bus_driver_total_compensation_l611_61142

-- Define the regular rate
def regular_rate : ℝ := 16

-- Define the number of regular hours
def regular_hours : ℕ := 40

-- Define the overtime rate as 75% higher than the regular rate
def overtime_rate : ℝ := regular_rate * 1.75

-- Define the total hours worked in the week
def total_hours_worked : ℕ := 48

-- Calculate the overtime hours
def overtime_hours : ℕ := total_hours_worked - regular_hours

-- Calculate the total compensation
def total_compensation : ℝ :=
  (regular_hours * regular_rate) + (overtime_hours * overtime_rate)

-- Theorem to prove that the total compensation is $864
theorem bus_driver_total_compensation : total_compensation = 864 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_bus_driver_total_compensation_l611_61142


namespace NUMINAMATH_GPT_money_returned_l611_61165

theorem money_returned (individual group taken : ℝ)
  (h1 : individual = 12000)
  (h2 : group = 16000)
  (h3 : taken = 26400) :
  (individual + group - taken) = 1600 :=
by
  -- The proof has been omitted
  sorry

end NUMINAMATH_GPT_money_returned_l611_61165


namespace NUMINAMATH_GPT_sum_geometric_series_nine_l611_61137

noncomputable def geometric_series_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  S n = a 0 * (1 - a 1 ^ n) / (1 - a 1)

theorem sum_geometric_series_nine
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (S_3 : S 3 = 12)
  (S_6 : S 6 = 60) :
  S 9 = 252 := by
  sorry

end NUMINAMATH_GPT_sum_geometric_series_nine_l611_61137


namespace NUMINAMATH_GPT_sam_drove_200_miles_l611_61163

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end NUMINAMATH_GPT_sam_drove_200_miles_l611_61163


namespace NUMINAMATH_GPT_least_positive_integer_divisors_l611_61174

theorem least_positive_integer_divisors (n m k : ℕ) (h₁ : (∀ d : ℕ, d ∣ n ↔ d ≤ 2023))
(h₂ : n = m * 6^k) (h₃ : (∀ d : ℕ, d ∣ 6 → ¬(d ∣ m))) : m + k = 80 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_divisors_l611_61174


namespace NUMINAMATH_GPT_average_is_1380_l611_61123

def avg_of_numbers : Prop := 
  (1200 + 1300 + 1400 + 1510 + 1520 + 1530 + 1200) / 7 = 1380

theorem average_is_1380 : avg_of_numbers := by
  sorry

end NUMINAMATH_GPT_average_is_1380_l611_61123


namespace NUMINAMATH_GPT_closest_years_l611_61106

theorem closest_years (a b c d : ℕ) (h1 : 10 * a + b + 10 * c + d = 10 * b + c) :
  (a = 1 ∧ b = 8 ∧ c = 6 ∧ d = 8) ∨ (a = 2 ∧ b = 3 ∧ c = 0 ∧ d =7) ↔
  ((10 * 1 + 8 + 10 * 6 + 8 = 10 * 8 + 6) ∧ (10 * 2 + 3 + 10 * 0 + 7 = 10 * 3 + 0)) :=
sorry

end NUMINAMATH_GPT_closest_years_l611_61106


namespace NUMINAMATH_GPT_equivalent_conditions_l611_61105

theorem equivalent_conditions 
  (f : ℕ+ → ℕ+)
  (H1 : ∀ (m n : ℕ+), m ≤ n → (f m + n) ∣ (f n + m))
  (H2 : ∀ (m n : ℕ+), m ≥ n → (f m + n) ∣ (f n + m)) :
  (∀ (m n : ℕ+), m ≤ n → (f m + n) ∣ (f n + m)) ↔ 
  (∀ (m n : ℕ+), m ≥ n → (f m + n) ∣ (f n + m)) :=
sorry

end NUMINAMATH_GPT_equivalent_conditions_l611_61105


namespace NUMINAMATH_GPT_expression_evaluation_l611_61121

theorem expression_evaluation :
  (40 - (2040 - 210)) + (2040 - (210 - 40)) = 80 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l611_61121


namespace NUMINAMATH_GPT_sum_of_interior_angles_l611_61193

noncomputable def exterior_angle (n : ℕ) := 360 / n

theorem sum_of_interior_angles (n : ℕ) (h : exterior_angle n = 45) :
  180 * (n - 2) = 1080 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_l611_61193


namespace NUMINAMATH_GPT_quadratic_two_distinct_roots_l611_61161

theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k*x^2 - 6*x + 9 = 0) ∧ (k*y^2 - 6*y + 9 = 0)) ↔ (k < 1 ∧ k ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_roots_l611_61161


namespace NUMINAMATH_GPT_isosceles_triangle_largest_angle_l611_61188

theorem isosceles_triangle_largest_angle (a b c : ℝ) (h1 : a = b) (h2 : b_angle = 50) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) 
  (h6 : a + b + c = 180) : c ≥ a ∨ c ≥ b → c = 80 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_largest_angle_l611_61188


namespace NUMINAMATH_GPT_expenses_opposite_to_income_l611_61132

theorem expenses_opposite_to_income (income_5 : ℤ) (h_income : income_5 = 5) : -income_5 = -5 :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_expenses_opposite_to_income_l611_61132


namespace NUMINAMATH_GPT_find_k2_minus_b2_l611_61104

theorem find_k2_minus_b2 (k b : ℝ) (h1 : 3 = k * 1 + b) (h2 : 2 = k * (-1) + b) : k^2 - b^2 = -6 := 
by
  sorry

end NUMINAMATH_GPT_find_k2_minus_b2_l611_61104


namespace NUMINAMATH_GPT_pipe_a_fills_cistern_l611_61152

theorem pipe_a_fills_cistern :
  ∀ (x : ℝ), (1 / x + 1 / 120 - 1 / 120 = 1 / 60) → x = 60 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_pipe_a_fills_cistern_l611_61152


namespace NUMINAMATH_GPT_remainder_333_pow_333_mod_11_l611_61125

theorem remainder_333_pow_333_mod_11 : (333 ^ 333) % 11 = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_333_pow_333_mod_11_l611_61125


namespace NUMINAMATH_GPT_nine_digit_number_conditions_l611_61164

def nine_digit_number := 900900000

def remove_second_digit (n : ℕ) : ℕ := n / 100000000 * 10000000 + n % 10000000
def remove_third_digit (n : ℕ) : ℕ := n / 10000000 * 1000000 + n % 1000000
def remove_ninth_digit (n : ℕ) : ℕ := n / 10

theorem nine_digit_number_conditions :
  (remove_second_digit nine_digit_number) % 2 = 0 ∧
  (remove_third_digit nine_digit_number) % 3 = 0 ∧
  (remove_ninth_digit nine_digit_number) % 9 = 0 :=
by
  -- Proof steps would be included here.
  sorry

end NUMINAMATH_GPT_nine_digit_number_conditions_l611_61164


namespace NUMINAMATH_GPT_factorization_correct_l611_61139

def factor_expression (x : ℝ) : ℝ :=
  (12 * x^4 - 27 * x^3 + 45 * x) - (-3 * x^4 - 6 * x^3 + 9 * x)

theorem factorization_correct (x : ℝ) : 
  factor_expression x = 3 * x * (5 * x^3 - 7 * x^2 + 12) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l611_61139


namespace NUMINAMATH_GPT_linear_equation_a_is_the_only_one_l611_61195

-- Definitions for each equation
def equation_a (x y : ℝ) : Prop := x + y = 2
def equation_b (x : ℝ) : Prop := x + 1 = -10
def equation_c (x y : ℝ) : Prop := x - 1/y = 6
def equation_d (x y : ℝ) : Prop := x^2 = 2 * y

-- Proof that equation_a is the only linear equation with two variables
theorem linear_equation_a_is_the_only_one (x y : ℝ) : 
  equation_a x y ∧ ¬equation_b x ∧ ¬(∃ y, equation_c x y) ∧ ¬(∃ y, equation_d x y) :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_a_is_the_only_one_l611_61195


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l611_61108

noncomputable def A : Set ℕ := {x | x > 0 ∧ x ≤ 3}
def B : Set ℕ := {x | 0 < x ∧ x < 4}

theorem intersection_of_A_and_B : 
  A ∩ B = {1, 2, 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_A_and_B_l611_61108


namespace NUMINAMATH_GPT_simplify_expression_l611_61169

theorem simplify_expression : (0.4 * 0.5 + 0.3 * 0.2) = 0.26 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l611_61169


namespace NUMINAMATH_GPT_sum_of_first_6033_terms_l611_61160

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_6033_terms (a r : ℝ) (h1 : geometric_sum a r 2011 = 200) 
  (h2 : geometric_sum a r 4022 = 380) : 
  geometric_sum a r 6033 = 542 :=
sorry

end NUMINAMATH_GPT_sum_of_first_6033_terms_l611_61160


namespace NUMINAMATH_GPT_min_plus_max_value_of_x_l611_61149

theorem min_plus_max_value_of_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  let m := (10 - Real.sqrt 304) / 6
  let M := (10 + Real.sqrt 304) / 6
  m + M = 10 / 3 := by 
  sorry

end NUMINAMATH_GPT_min_plus_max_value_of_x_l611_61149


namespace NUMINAMATH_GPT_find_income_l611_61141

-- Definitions of percentages used in calculations
def rent_percentage : ℝ := 0.15
def education_percentage : ℝ := 0.15
def misc_percentage : ℝ := 0.10
def medical_percentage : ℝ := 0.15

-- Remaining amount after all expenses
def final_amount : ℝ := 5548

-- Income calculation function
def calc_income (X : ℝ) : ℝ :=
  let after_rent := X * (1 - rent_percentage)
  let after_education := after_rent * (1 - education_percentage)
  let after_misc := after_education * (1 - misc_percentage)
  let after_medical := after_misc * (1 - medical_percentage)
  after_medical

-- Theorem statement to prove the woman's income
theorem find_income (X : ℝ) (h : calc_income X = final_amount) : X = 10038.46 := by
  sorry

end NUMINAMATH_GPT_find_income_l611_61141


namespace NUMINAMATH_GPT_store_credit_card_discount_proof_l611_61134

def full_price : ℕ := 125
def sale_discount_percentage : ℕ := 20
def coupon_discount : ℕ := 10
def total_savings : ℕ := 44

def sale_discount := full_price * sale_discount_percentage / 100
def price_after_sale_discount := full_price - sale_discount
def price_after_coupon := price_after_sale_discount - coupon_discount
def store_credit_card_discount := total_savings - sale_discount - coupon_discount
def discount_percentage_of_store_credit := (store_credit_card_discount * 100) / price_after_coupon

theorem store_credit_card_discount_proof : discount_percentage_of_store_credit = 10 := by
  sorry

end NUMINAMATH_GPT_store_credit_card_discount_proof_l611_61134


namespace NUMINAMATH_GPT_no_integer_solutions_l611_61166

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^4 + y^2 = 6 * y - 3 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l611_61166


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_ellipse_l611_61156

theorem necessary_but_not_sufficient_condition_for_ellipse (m : ℝ) :
  (2 < m ∧ m < 6) ↔ ((∃ m, 2 < m ∧ m < 6 ∧ m ≠ 4) ∧ (∀ m, (2 < m ∧ m < 6) → ¬(m = 4))) := 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_ellipse_l611_61156


namespace NUMINAMATH_GPT_determine_k_values_parallel_lines_l611_61168

theorem determine_k_values_parallel_lines :
  ∀ k : ℝ, ((k - 3) * x + (4 - k) * y + 1 = 0 ∧ 2 * (k - 3) * x - 2 * y + 3 = 0)
  → k = 2 ∨ k = 3 ∨ k = 6 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_values_parallel_lines_l611_61168


namespace NUMINAMATH_GPT_evaluate_f_g_f_l611_61182

def f (x: ℝ) : ℝ := 5 * x + 4
def g (x: ℝ) : ℝ := 3 * x + 5

theorem evaluate_f_g_f :
  f (g (f 3)) = 314 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_g_f_l611_61182


namespace NUMINAMATH_GPT_yellow_less_than_three_times_red_l611_61118

def num_red : ℕ := 40
def less_than_three_times (Y : ℕ) : Prop := Y < 120
def blue_half_yellow (Y B : ℕ) : Prop := B = Y / 2
def remaining_after_carlos (B : ℕ) : Prop := 40 + B = 90
def difference_three_times_red (Y : ℕ) : ℕ := 3 * num_red - Y

theorem yellow_less_than_three_times_red (Y B : ℕ) 
  (h1 : less_than_three_times Y) 
  (h2 : blue_half_yellow Y B) 
  (h3 : remaining_after_carlos B) : 
  difference_three_times_red Y = 20 := by
  sorry

end NUMINAMATH_GPT_yellow_less_than_three_times_red_l611_61118


namespace NUMINAMATH_GPT_gcd_8917_4273_l611_61180

theorem gcd_8917_4273 : Int.gcd 8917 4273 = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_8917_4273_l611_61180


namespace NUMINAMATH_GPT_necessary_condition_l611_61189

theorem necessary_condition (x : ℝ) (h : (x-1) * (x-2) ≤ 0) : x^2 - 3 * x ≤ 0 :=
sorry

end NUMINAMATH_GPT_necessary_condition_l611_61189


namespace NUMINAMATH_GPT_perfect_square_divisors_of_240_l611_61183

theorem perfect_square_divisors_of_240 : 
  (∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, 0 < k ∧ k < n → ¬(k = 1 ∨ k = 4 ∨ k = 16)) := 
sorry

end NUMINAMATH_GPT_perfect_square_divisors_of_240_l611_61183


namespace NUMINAMATH_GPT_largest_possible_number_of_sweets_in_each_tray_l611_61143

-- Define the initial conditions as given in the problem statement
def tim_sweets : ℕ := 36
def peter_sweets : ℕ := 44

-- Define the statement that we want to prove
theorem largest_possible_number_of_sweets_in_each_tray :
  Nat.gcd tim_sweets peter_sweets = 4 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_number_of_sweets_in_each_tray_l611_61143


namespace NUMINAMATH_GPT_max_min_difference_l611_61199

noncomputable def difference_max_min_z (x y z : ℝ) : ℝ :=
  if h₁ : x + y + z = 3 ∧ x^2 + y^2 + z^2 = 18 then 6 else 0

theorem max_min_difference (x y z : ℝ) (h₁ : x + y + z = 3) (h₂ : x^2 + y^2 + z^2 = 18) :
  difference_max_min_z x y z = 6 :=
by sorry

end NUMINAMATH_GPT_max_min_difference_l611_61199


namespace NUMINAMATH_GPT_f_is_odd_l611_61190

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.sqrt (1 + x^2))

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_GPT_f_is_odd_l611_61190


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l611_61101

theorem cone_lateral_surface_area (r h : ℝ) (h_r : r = 3) (h_h : h = 4) : 
  (1/2) * (2 * Real.pi * r) * (Real.sqrt (r ^ 2 + h ^ 2)) = 15 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l611_61101


namespace NUMINAMATH_GPT_sum_of_first_9000_terms_of_geometric_sequence_l611_61114

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end NUMINAMATH_GPT_sum_of_first_9000_terms_of_geometric_sequence_l611_61114


namespace NUMINAMATH_GPT_victoria_gym_sessions_l611_61191

-- Define the initial conditions
def starts_on_monday := true
def sessions_per_two_week_cycle := 6
def total_sessions := 30

-- Define the sought day of the week when all gym sessions are completed
def final_day := "Thursday"

-- The theorem stating the problem
theorem victoria_gym_sessions : 
  starts_on_monday →
  sessions_per_two_week_cycle = 6 →
  total_sessions = 30 →
  final_day = "Thursday" := 
by
  intros
  exact sorry

end NUMINAMATH_GPT_victoria_gym_sessions_l611_61191


namespace NUMINAMATH_GPT_quadratic_opposite_roots_l611_61112

theorem quadratic_opposite_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 + x2 = 0 ∧ x1 * x2 = k + 1) ↔ k = -2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_opposite_roots_l611_61112


namespace NUMINAMATH_GPT_least_number_subtracted_l611_61147

theorem least_number_subtracted (x : ℤ) (N : ℤ) :
  N = 2590 - x →
  (N % 9 = 6) →
  (N % 11 = 6) →
  (N % 13 = 6) →
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_l611_61147


namespace NUMINAMATH_GPT_line_solutions_l611_61133

-- Definition for points
def point := ℝ × ℝ

-- Conditions for lines and points
def line1 (p : point) : Prop := 3 * p.1 + 4 * p.2 = 2
def line2 (p : point) : Prop := 2 * p.1 + p.2 = -2
def line3 : Prop := ∃ p : point, line1 p ∧ line2 p

def lineL (p : point) : Prop := 2 * p.1 + p.2 = -2 -- Line l we need to prove
def perp_lineL : Prop := ∃ p : point, lineL p ∧ p.1 - 2 * p.2 = 1

-- Symmetry condition for the line
def symmetric_line (p : point) : Prop := 2 * p.1 + p.2 = 2 -- Symmetric line we need to prove

-- Main theorem to prove
theorem line_solutions :
  line3 →
  perp_lineL →
  (∀ p, lineL p ↔ 2 * p.1 + p.2 = -2) ∧
  (∀ p, symmetric_line p ↔ 2 * p.1 + p.2 = 2) :=
sorry

end NUMINAMATH_GPT_line_solutions_l611_61133


namespace NUMINAMATH_GPT_candies_per_child_rounded_l611_61159

/-- There are 15 pieces of candy divided equally among 7 children. The number of candies per child, rounded to the nearest tenth, is 2.1. -/
theorem candies_per_child_rounded :
  let candies := 15
  let children := 7
  Float.round (candies / children * 10) / 10 = 2.1 :=
by
  sorry

end NUMINAMATH_GPT_candies_per_child_rounded_l611_61159


namespace NUMINAMATH_GPT_x_minus_y_eq_neg3_l611_61197

theorem x_minus_y_eq_neg3 (x y : ℝ) (i : ℂ) (h1 : x * i + 2 = y - i) (h2 : i^2 = -1) : x - y = -3 := 
  sorry

end NUMINAMATH_GPT_x_minus_y_eq_neg3_l611_61197


namespace NUMINAMATH_GPT_arithmetic_sequence_geo_ratio_l611_61111

theorem arithmetic_sequence_geo_ratio
  (a_n : ℕ → ℝ)
  (d : ℝ)
  (h_nonzero : d ≠ 0)
  (S : ℕ → ℝ)
  (h_seq : ∀ n, S n = (n * (2 * a_n 1 + (n - 1) * d)) / 2)
  (h_geo : (S 2) ^ 2 = S 1 * S 4) :
  (a_n 2 + a_n 3) / a_n 1 = 8 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_geo_ratio_l611_61111


namespace NUMINAMATH_GPT_car_second_hour_speed_l611_61109

theorem car_second_hour_speed (s1 s2 : ℕ) (h1 : s1 = 100) (avg : (s1 + s2) / 2 = 80) : s2 = 60 :=
by
  sorry

end NUMINAMATH_GPT_car_second_hour_speed_l611_61109


namespace NUMINAMATH_GPT_circle_center_l611_61157

theorem circle_center 
    (x y : ℝ)
    (h : x^2 + y^2 - 4 * x + 6 * y = 0) :
    (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = (x^2 - 4*x + 4) + (y^2 + 6*y + 9) 
    → (x, y) = (2, -3)) :=
sorry

end NUMINAMATH_GPT_circle_center_l611_61157


namespace NUMINAMATH_GPT_line_equation_l611_61154

def line_through (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let m := (y₂ - y₁) / (x₂ - x₁)
  y - y₁ = m * (x - x₁)

noncomputable def is_trisection_point (A B QR : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (qx, qy) := QR
  (qx = (2 * x₂ + x₁) / 3 ∧ qy = (2 * y₂ + y₁) / 3) ∨
  (qx = (x₂ + 2 * x₁) / 3 ∧ qy = (y₂ + 2 * y₁) / 3)

theorem line_equation (A B P Q : ℝ × ℝ)
  (hA : A = (3, 4))
  (hB : B = (-4, 5))
  (hP : is_trisection_point B A P)
  (hQ : is_trisection_point B A Q) :
  line_through A P 1 3 ∨ line_through A P 2 1 → 
  (line_through A P 3 4 → P = (1, 3)) ∧ 
  (line_through A P 2 1 → P = (2, 1)) ∧ 
  (line_through A P x y → x - 4 * y + 13 = 0) := 
by 
  sorry

end NUMINAMATH_GPT_line_equation_l611_61154


namespace NUMINAMATH_GPT_find_row_with_sum_2013_squared_l611_61130

-- Define the sum of the numbers in the nth row
def sum_of_row (n : ℕ) : ℕ := (2 * n - 1)^2

theorem find_row_with_sum_2013_squared : (∃ n : ℕ, sum_of_row n = 2013^2) ∧ (sum_of_row 1007 = 2013^2) :=
by
  sorry

end NUMINAMATH_GPT_find_row_with_sum_2013_squared_l611_61130


namespace NUMINAMATH_GPT_find_other_endpoint_l611_61194

def other_endpoint (midpoint endpoint: ℝ × ℝ) : ℝ × ℝ :=
  let (mx, my) := midpoint
  let (ex, ey) := endpoint
  (2 * mx - ex, 2 * my - ey)

theorem find_other_endpoint :
  other_endpoint (3, 1) (7, -4) = (-1, 6) :=
by
  -- Midpoint formula to find other endpoint
  sorry

end NUMINAMATH_GPT_find_other_endpoint_l611_61194


namespace NUMINAMATH_GPT_positive_y_percent_y_eq_16_l611_61171

theorem positive_y_percent_y_eq_16 (y : ℝ) (hy : 0 < y) (h : 0.01 * y * y = 16) : y = 40 :=
by
  sorry

end NUMINAMATH_GPT_positive_y_percent_y_eq_16_l611_61171


namespace NUMINAMATH_GPT_point_in_first_quadrant_l611_61179

theorem point_in_first_quadrant (x y : ℝ) (hx : x = 6) (hy : y = 2) : x > 0 ∧ y > 0 :=
by
  rw [hx, hy]
  exact ⟨by norm_num, by norm_num⟩

end NUMINAMATH_GPT_point_in_first_quadrant_l611_61179


namespace NUMINAMATH_GPT_tan_angle_sum_l611_61150

noncomputable def tan_sum (θ : ℝ) : ℝ := Real.tan (θ + (Real.pi / 4))

theorem tan_angle_sum :
  let x := 1
  let y := 2
  let θ := Real.arctan (y / x)
  tan_sum θ = -3 := by
  sorry

end NUMINAMATH_GPT_tan_angle_sum_l611_61150


namespace NUMINAMATH_GPT_probability_two_white_balls_l611_61146

-- Definitions
def totalBalls : ℕ := 5
def whiteBalls : ℕ := 3
def blackBalls : ℕ := 2
def totalWaysToDrawTwoBalls : ℕ := Nat.choose totalBalls 2
def waysToDrawTwoWhiteBalls : ℕ := Nat.choose whiteBalls 2

-- Theorem statement
theorem probability_two_white_balls :
  (waysToDrawTwoWhiteBalls : ℚ) / totalWaysToDrawTwoBalls = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_probability_two_white_balls_l611_61146


namespace NUMINAMATH_GPT_expression_value_l611_61148

theorem expression_value :
    (2.502 + 0.064)^2 - ((2.502 - 0.064)^2) / (2.502 * 0.064) = 4.002 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_expression_value_l611_61148


namespace NUMINAMATH_GPT_find_x_values_l611_61145

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem find_x_values :
  { x : ℕ | combination 10 x = combination 10 (3 * x - 2) } = {1, 3} :=
by
  sorry

end NUMINAMATH_GPT_find_x_values_l611_61145


namespace NUMINAMATH_GPT_four_cards_probability_l611_61124

theorem four_cards_probability :
  let deck_size := 52
  let suits_size := 13
  ∀ (C D H S : ℕ), 
  C = 1 ∧ D = 13 ∧ H = 13 ∧ S = 13 →
  (C / deck_size) *
  (D / (deck_size - 1)) *
  (H / (deck_size - 2)) *
  (S / (deck_size - 3)) = (2197 / 499800) :=
by
  intros deck_size suits_size C D H S h
  sorry

end NUMINAMATH_GPT_four_cards_probability_l611_61124


namespace NUMINAMATH_GPT_simplify_expression_l611_61181

theorem simplify_expression (y : ℝ) : (3 * y + 4 * y + 5 * y + 7) = (12 * y + 7) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l611_61181


namespace NUMINAMATH_GPT_solve_equation_l611_61173

theorem solve_equation (x : ℝ) :
  (x + 1)^2 = (2 * x - 1)^2 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l611_61173


namespace NUMINAMATH_GPT_area_of_each_small_concave_quadrilateral_l611_61151

noncomputable def inner_diameter : ℝ := 8
noncomputable def outer_diameter : ℝ := 10
noncomputable def total_area_covered_by_annuli : ℝ := 112.5
noncomputable def pi : ℝ := 3.14

theorem area_of_each_small_concave_quadrilateral (inner_diameter outer_diameter total_area_covered_by_annuli pi: ℝ)
    (h1 : inner_diameter = 8)
    (h2 : outer_diameter = 10)
    (h3 : total_area_covered_by_annuli = 112.5)
    (h4 : pi = 3.14) :
    (π * (outer_diameter / 2) ^ 2 - π * (inner_diameter / 2) ^ 2) * 5 - total_area_covered_by_annuli / 4 = 7.2 := 
sorry

end NUMINAMATH_GPT_area_of_each_small_concave_quadrilateral_l611_61151
