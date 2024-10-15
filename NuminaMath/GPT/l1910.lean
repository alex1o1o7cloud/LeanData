import Mathlib

namespace NUMINAMATH_GPT_output_value_is_3_l1910_191068

-- Define the variables and the program logic
def program (a b : ℕ) : ℕ :=
  if a > b then a else b

-- The theorem statement
theorem output_value_is_3 (a b : ℕ) (ha : a = 2) (hb : b = 3) : program a b = 3 :=
by
  -- Automatically assume the given conditions and conclude the proof. The actual proof is skipped.
  sorry

end NUMINAMATH_GPT_output_value_is_3_l1910_191068


namespace NUMINAMATH_GPT_unique_solution_iff_a_eq_2019_l1910_191098

theorem unique_solution_iff_a_eq_2019 (x a : ℝ) :
  (∃! x : ℝ, x - 1000 ≥ 1018 ∧ x + 1 ≤ a) ↔ a = 2019 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_iff_a_eq_2019_l1910_191098


namespace NUMINAMATH_GPT_triangle_angles_l1910_191011

noncomputable def angle_triangle (E : ℝ) :=
if E = 45 then (90, 45, 45) else if E = 36 then (72, 72, 36) else (0, 0, 0)

theorem triangle_angles (E : ℝ) :
  (∃ E, E = 45 → angle_triangle E = (90, 45, 45))
  ∨
  (∃ E, E = 36 → angle_triangle E = (72, 72, 36)) :=
by
    sorry

end NUMINAMATH_GPT_triangle_angles_l1910_191011


namespace NUMINAMATH_GPT_solution_inequality_l1910_191099

variable (a x : ℝ)

theorem solution_inequality (h : ∀ x, |x - a| + |x + 4| ≥ 1) : a ≤ -5 ∨ a ≥ -3 := by
  sorry

end NUMINAMATH_GPT_solution_inequality_l1910_191099


namespace NUMINAMATH_GPT_value_of_b_is_one_l1910_191015

open Complex

theorem value_of_b_is_one (a b : ℝ) (h : (1 + I) / (1 - I) = a + b * I) : b = 1 := 
by
  sorry

end NUMINAMATH_GPT_value_of_b_is_one_l1910_191015


namespace NUMINAMATH_GPT_base_of_524_l1910_191092

theorem base_of_524 : 
  ∀ (b : ℕ), (5 * b^2 + 2 * b + 4 = 340) → b = 8 :=
by
  intros b h
  sorry

end NUMINAMATH_GPT_base_of_524_l1910_191092


namespace NUMINAMATH_GPT_vector_magnitude_parallel_l1910_191062

/-- Given two plane vectors a = (1, 2) and b = (-2, y),
if a is parallel to b, then |2a - b| = 4 * sqrt 5. -/
theorem vector_magnitude_parallel (y : ℝ) 
  (h_parallel : (1 : ℝ) / (-2 : ℝ) = (2 : ℝ) / y) : 
  ‖2 • (1, 2) - (-2, y)‖ = 4 * Real.sqrt 5 := 
by
  sorry

end NUMINAMATH_GPT_vector_magnitude_parallel_l1910_191062


namespace NUMINAMATH_GPT_problem_statement_l1910_191088

noncomputable def min_expression_value (θ1 θ2 θ3 θ4 : ℝ) : ℝ :=
  (2 * (Real.sin θ1)^2 + 1 / (Real.sin θ1)^2) *
  (2 * (Real.sin θ2)^2 + 1 / (Real.sin θ2)^2) *
  (2 * (Real.sin θ3)^2 + 1 / (Real.sin θ3)^2) *
  (2 * (Real.sin θ4)^2 + 1 / (Real.sin θ4)^2)

theorem problem_statement (θ1 θ2 θ3 θ4 : ℝ) (h_pos: θ1 > 0 ∧ θ2 > 0 ∧ θ3 > 0 ∧ θ4 > 0) (h_sum: θ1 + θ2 + θ3 + θ4 = Real.pi) :
  min_expression_value θ1 θ2 θ3 θ4 = 81 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1910_191088


namespace NUMINAMATH_GPT_yen_per_cad_l1910_191004

theorem yen_per_cad (yen cad : ℝ) (h : yen / cad = 5000 / 60) : yen = 83 := by
  sorry

end NUMINAMATH_GPT_yen_per_cad_l1910_191004


namespace NUMINAMATH_GPT_different_digits_probability_l1910_191049

noncomputable def number_nonidentical_probability : ℚ :=
  let total_numbers := 900
  let identical_numbers := 9
  -- The probability of identical digits.
  let identical_probability := identical_numbers / total_numbers
  -- The probability of non-identical digits.
  1 - identical_probability

theorem different_digits_probability : number_nonidentical_probability = 99 / 100 := by
  sorry

end NUMINAMATH_GPT_different_digits_probability_l1910_191049


namespace NUMINAMATH_GPT_range_of_a_minus_b_l1910_191091

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 2 < b ∧ b < 4) : -4 < a - b ∧ a - b < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_minus_b_l1910_191091


namespace NUMINAMATH_GPT_rent_3600_yields_88_optimal_rent_is_4100_and_max_revenue_is_304200_l1910_191031

/-
Via conditions:
1. The rental company owns 100 cars.
2. When the monthly rent for each car is set at 3000 yuan, all cars can be rented out.
3. For every 50 yuan increase in the monthly rent per car, there will be one more car that is not rented out.
4. The maintenance cost for each rented car is 200 yuan per month.
-/

noncomputable def num_rented_cars (rent_per_car : ℕ) : ℕ :=
  if rent_per_car < 3000 then 100 else max 0 (100 - (rent_per_car - 3000) / 50)

noncomputable def monthly_revenue (rent_per_car : ℕ) : ℕ :=
  let cars_rented := num_rented_cars rent_per_car
  let maintenance_cost := 200 * cars_rented
  (rent_per_car - maintenance_cost) * cars_rented

theorem rent_3600_yields_88 : num_rented_cars 3600 = 88 :=
  sorry

theorem optimal_rent_is_4100_and_max_revenue_is_304200 :
  ∃ rent_per_car, rent_per_car = 4100 ∧ monthly_revenue rent_per_car = 304200 :=
  sorry

end NUMINAMATH_GPT_rent_3600_yields_88_optimal_rent_is_4100_and_max_revenue_is_304200_l1910_191031


namespace NUMINAMATH_GPT_part1_solution_set_part2_values_a_b_part3_range_m_l1910_191008

-- Definitions for the given functions
def y1 (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def y2 (x : ℝ) : ℝ := x^2 + x - 2

-- Proof that the solution set for y2 < 0 is (-2, 1)
theorem part1_solution_set : ∀ x : ℝ, y2 x < 0 ↔ (x > -2 ∧ x < 1) :=
sorry

-- Given |y1| ≤ |y2| for all x ∈ ℝ, prove that a = 1 and b = -2
theorem part2_values_a_b (a b : ℝ) : (∀ x : ℝ, |y1 x a b| ≤ |y2 x|) → a = 1 ∧ b = -2 :=
sorry

-- Given y1 > (m-2)x - m for all x > 1 under condition from part 2, prove the range for m is (-∞, 2√2 + 5)
theorem part3_range_m (a b : ℝ) (m : ℝ) : 
  (∀ x : ℝ, |y1 x a b| ≤ |y2 x|) → a = 1 ∧ b = -2 →
  (∀ x : ℝ, x > 1 → y1 x a b > (m-2) * x - m) → m < 2 * Real.sqrt 2 + 5 :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_values_a_b_part3_range_m_l1910_191008


namespace NUMINAMATH_GPT_find_radius_l1910_191097

noncomputable def square_radius (r : ℝ) : Prop :=
  let s := (2 * r) / Real.sqrt 2  -- side length of the square derived from the radius
  let perimeter := 4 * s         -- perimeter of the square
  let area := Real.pi * r^2      -- area of the circumscribed circle
  perimeter = area               -- given condition

theorem find_radius (r : ℝ) (h : square_radius r) : r = (4 * Real.sqrt 2) / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_find_radius_l1910_191097


namespace NUMINAMATH_GPT_exists_root_in_interval_l1910_191084

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x - 2

theorem exists_root_in_interval :
  (f 1 < 0) ∧ (f 2 > 0) ∧ (∀ x > 0, ContinuousAt f x) → (∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0) :=
by sorry

end NUMINAMATH_GPT_exists_root_in_interval_l1910_191084


namespace NUMINAMATH_GPT_f_2015_l1910_191065

noncomputable def f : ℝ → ℝ := sorry
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_periodic : ∀ x : ℝ, f (x - 2) = -f x
axiom f_initial_segment : ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 0) → f x = 2^x

theorem f_2015 : f 2015 = 1 / 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_f_2015_l1910_191065


namespace NUMINAMATH_GPT_exists_root_in_interval_l1910_191003

theorem exists_root_in_interval
    (a b c x₁ x₂ : ℝ)
    (h₁ : a * x₁^2 + b * x₁ + c = 0)
    (h₂ : -a * x₂^2 + b * x₂ + c = 0) :
    ∃ x₃ : ℝ, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧ ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) :=
sorry

end NUMINAMATH_GPT_exists_root_in_interval_l1910_191003


namespace NUMINAMATH_GPT_remainder_when_sum_divided_by_30_l1910_191028

theorem remainder_when_sum_divided_by_30 {c d : ℕ} (p q : ℕ)
  (hc : c = 60 * p + 58)
  (hd : d = 90 * q + 85) :
  (c + d) % 30 = 23 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_sum_divided_by_30_l1910_191028


namespace NUMINAMATH_GPT_right_triangle_sides_l1910_191022

theorem right_triangle_sides (m n : ℝ) (x : ℝ) (a b c : ℝ)
  (h1 : 2 * x < m + n) 
  (h2 : a = Real.sqrt (2 * m * n) - m)
  (h3 : b = Real.sqrt (2 * m * n) - n)
  (h4 : c = m + n - Real.sqrt (2 * m * n))
  (h5 : a^2 + b^2 = c^2)
  (h6 : 4 * x^2 = (m - 2 * x)^2 + (n - 2 * x)^2) :
  a = Real.sqrt (2 * m * n) - m ∧ b = Real.sqrt (2 * m * n) - n ∧ c = m + n - Real.sqrt (2 * m * n) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_sides_l1910_191022


namespace NUMINAMATH_GPT_correct_statement_A_l1910_191072

-- Declare Avogadro's constant
def Avogadro_constant : ℝ := 6.022e23

-- Given conditions
def gas_mass_ethene : ℝ := 5.6 -- grams of ethylene
def gas_mass_cyclopropane : ℝ := 5.6 -- grams of cyclopropane
def gas_combined_carbon_atoms : ℝ := 0.4 * Avogadro_constant

-- Assertion to prove
theorem correct_statement_A :
    gas_combined_carbon_atoms = 0.4 * Avogadro_constant :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_A_l1910_191072


namespace NUMINAMATH_GPT_rows_of_seats_l1910_191027

theorem rows_of_seats (students sections_per_row students_per_section : ℕ) (h1 : students_per_section = 2) (h2 : sections_per_row = 2) (h3 : students = 52) :
  (students / students_per_section / sections_per_row) = 13 :=
sorry

end NUMINAMATH_GPT_rows_of_seats_l1910_191027


namespace NUMINAMATH_GPT_banks_policies_for_seniors_justified_l1910_191034

-- Defining conditions
def better_credit_repayment_reliability : Prop := sorry
def stable_pension_income : Prop := sorry
def indirect_younger_relative_contributions : Prop := sorry
def pensioners_inclination_to_save : Prop := sorry
def regular_monthly_income : Prop := sorry
def preference_for_long_term_deposits : Prop := sorry

-- Lean theorem statement using the conditions
theorem banks_policies_for_seniors_justified :
  better_credit_repayment_reliability →
  stable_pension_income →
  indirect_younger_relative_contributions →
  pensioners_inclination_to_save →
  regular_monthly_income →
  preference_for_long_term_deposits →
  (banks_should_offer_higher_deposit_and_lower_loan_rates_to_seniors : Prop) :=
by
  -- Insert proof here that given all the conditions the conclusion follows
  sorry -- proof not required, so skipping

end NUMINAMATH_GPT_banks_policies_for_seniors_justified_l1910_191034


namespace NUMINAMATH_GPT_percent_divisible_by_six_up_to_120_l1910_191030

theorem percent_divisible_by_six_up_to_120 : 
  let total_numbers := 120
  let divisible_by_six := total_numbers / 6
  let percentage := (divisible_by_six * 100) / total_numbers
  percentage = 50 / 3 := sorry

end NUMINAMATH_GPT_percent_divisible_by_six_up_to_120_l1910_191030


namespace NUMINAMATH_GPT_no_sensor_in_option_B_l1910_191067

/-- Define the technologies and whether they involve sensors --/
def technology_involves_sensor (opt : String) : Prop :=
  opt = "A" ∨ opt = "C" ∨ opt = "D"

theorem no_sensor_in_option_B :
  ¬ technology_involves_sensor "B" :=
by
  -- We assume the proof for the sake of this example.
  sorry

end NUMINAMATH_GPT_no_sensor_in_option_B_l1910_191067


namespace NUMINAMATH_GPT_max_abs_x_minus_2y_plus_1_l1910_191053

theorem max_abs_x_minus_2y_plus_1 (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) :
  |x - 2 * y + 1| ≤ 5 :=
sorry

end NUMINAMATH_GPT_max_abs_x_minus_2y_plus_1_l1910_191053


namespace NUMINAMATH_GPT_semicircle_area_increase_l1910_191059

noncomputable def area_semicircle (r : ℝ) : ℝ :=
  (1 / 2) * Real.pi * r^2

noncomputable def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem semicircle_area_increase :
  let r_long := 6
  let r_short := 4
  let area_long := 2 * area_semicircle r_long
  let area_short := 2 * area_semicircle r_short
  percent_increase area_short area_long = 125 :=
by
  let r_long := 6
  let r_short := 4
  let area_long := 2 * area_semicircle r_long
  let area_short := 2 * area_semicircle r_short
  have : area_semicircle r_long = 18 * Real.pi := by sorry
  have : area_semicircle r_short = 8 * Real.pi := by sorry
  have : area_long = 36 * Real.pi := by sorry
  have : area_short = 16 * Real.pi := by sorry
  have : percent_increase area_short area_long = 125 := by sorry
  exact this

end NUMINAMATH_GPT_semicircle_area_increase_l1910_191059


namespace NUMINAMATH_GPT_value_of_m_l1910_191066

theorem value_of_m (m : ℤ) : (∃ (f : ℤ → ℤ), ∀ x : ℤ, x^2 + m * x + 16 = (f x)^2) ↔ (m = 8 ∨ m = -8) := 
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1910_191066


namespace NUMINAMATH_GPT_rational_expression_is_rational_l1910_191093

theorem rational_expression_is_rational (a b c : ℚ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ r : ℚ, 
    r = Real.sqrt ((1 / (a - b)^2) + (1 / (b - c)^2) + (1 / (c - a)^2)) :=
sorry

end NUMINAMATH_GPT_rational_expression_is_rational_l1910_191093


namespace NUMINAMATH_GPT_marathon_yards_l1910_191043

theorem marathon_yards (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (marathons_run : ℕ) 
  (total_miles : ℕ) (total_yards : ℕ) (h1 : miles_per_marathon = 26) (h2 : yards_per_marathon = 385)
  (h3 : yards_per_mile = 1760) (h4 : marathons_run = 15) (h5 : 
  total_miles = marathons_run * miles_per_marathon + (marathons_run * yards_per_marathon) / yards_per_mile) 
  (h6 : total_yards = (marathons_run * yards_per_marathon) % yards_per_mile) : 
  total_yards = 495 :=
by
  -- This will be our process to verify the transformation
  sorry

end NUMINAMATH_GPT_marathon_yards_l1910_191043


namespace NUMINAMATH_GPT_total_marbles_l1910_191096

variable (w o p : ℝ)

-- Conditions as hypothesis
axiom h1 : o + p = 10
axiom h2 : w + p = 12
axiom h3 : w + o = 5

theorem total_marbles : w + o + p = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_l1910_191096


namespace NUMINAMATH_GPT_robert_birth_year_l1910_191085

theorem robert_birth_year (n : ℕ) (h1 : (n + 1)^2 - n^2 = 89) : n = 44 ∧ n^2 = 1936 :=
by {
  sorry
}

end NUMINAMATH_GPT_robert_birth_year_l1910_191085


namespace NUMINAMATH_GPT_soccer_team_games_l1910_191029

theorem soccer_team_games (pizzas : ℕ) (slices_per_pizza : ℕ) (average_goals_per_game : ℕ) (total_games : ℕ) 
  (h1 : pizzas = 6) 
  (h2 : slices_per_pizza = 12) 
  (h3 : average_goals_per_game = 9) 
  (h4 : total_games = (pizzas * slices_per_pizza) / average_goals_per_game) :
  total_games = 8 :=
by
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_soccer_team_games_l1910_191029


namespace NUMINAMATH_GPT_frank_fencemaker_fence_length_l1910_191048

theorem frank_fencemaker_fence_length :
  ∃ (L W : ℕ), W = 40 ∧
               (L * W = 200) ∧
               (2 * L + W = 50) :=
by
  sorry

end NUMINAMATH_GPT_frank_fencemaker_fence_length_l1910_191048


namespace NUMINAMATH_GPT_product_of_numbers_l1910_191010

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 72) (h2 : x - y = 20) : x * y = 1196 := 
sorry

end NUMINAMATH_GPT_product_of_numbers_l1910_191010


namespace NUMINAMATH_GPT_hexagon_height_correct_l1910_191078

-- Define the dimensions of the original rectangle
def original_rectangle_width := 16
def original_rectangle_height := 9
def original_rectangle_area := original_rectangle_width * original_rectangle_height

-- Define the dimensions of the new rectangle formed by the hexagons
def new_rectangle_width := 12
def new_rectangle_height := 12
def new_rectangle_area := new_rectangle_width * new_rectangle_height

-- Define the parameter x, which is the height of the hexagons
def hexagon_height := 6

-- Theorem stating the equivalence of the areas and the specific height x
theorem hexagon_height_correct :
  original_rectangle_area = new_rectangle_area ∧
  hexagon_height * 2 = new_rectangle_height :=
by
  sorry

end NUMINAMATH_GPT_hexagon_height_correct_l1910_191078


namespace NUMINAMATH_GPT_cost_of_football_correct_l1910_191001

-- We define the variables for the costs
def total_amount_spent : ℝ := 20.52
def cost_of_marbles : ℝ := 9.05
def cost_of_baseball : ℝ := 6.52
def cost_of_football : ℝ := total_amount_spent - cost_of_marbles - cost_of_baseball

-- We now state what needs to be proven: that Mike spent $4.95 on the football.
theorem cost_of_football_correct : cost_of_football = 4.95 := by
  sorry

end NUMINAMATH_GPT_cost_of_football_correct_l1910_191001


namespace NUMINAMATH_GPT_find_a6_l1910_191095

-- Define the geometric sequence and the given terms
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

variables {a : ℕ → ℝ} (r : ℝ)

-- Given conditions
axiom a_2 : a 2 = 2
axiom a_10 : a 10 = 8
axiom geo_seq : geometric_sequence a

-- Statement to prove
theorem find_a6 : a 6 = 4 :=
sorry

end NUMINAMATH_GPT_find_a6_l1910_191095


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1910_191082

open Real

variable {a a5 a3 a4 S4 q : ℝ}

theorem geometric_sequence_sum (h1 : q < 1)
                             (h2 : a + a5 = 20)
                             (h3 : a3 * a5 = 64) :
                             S4 = 120 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1910_191082


namespace NUMINAMATH_GPT_initial_amount_l1910_191076

def pie_cost : Real := 6.75
def juice_cost : Real := 2.50
def gift : Real := 10.00
def mary_final : Real := 52.00

theorem initial_amount (M : Real) :
  M = mary_final + pie_cost + juice_cost + gift :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_l1910_191076


namespace NUMINAMATH_GPT_total_shaded_area_l1910_191020

theorem total_shaded_area (S T U : ℝ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 2)
  (h3 : T / U = 2) :
  1 * (S * S) + 4 * (T * T) + 8 * (U * U) = 22.5 := by
sorry

end NUMINAMATH_GPT_total_shaded_area_l1910_191020


namespace NUMINAMATH_GPT_nanometers_to_scientific_notation_l1910_191039

theorem nanometers_to_scientific_notation :
  (246 : ℝ) * (10 ^ (-9 : ℝ)) = (2.46 : ℝ) * (10 ^ (-7 : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_nanometers_to_scientific_notation_l1910_191039


namespace NUMINAMATH_GPT_quadratic_roots_transformation_l1910_191042

theorem quadratic_roots_transformation {a b c r s : ℝ}
  (h1 : r + s = -b / a)
  (h2 : r * s = c / a) :
  (∃ p q : ℝ, p = a * r + 2 * b ∧ q = a * s + 2 * b ∧ 
     (∀ x, x^2 - 3 * b * x + 2 * b^2 + a * c = (x - p) * (x - q))) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_transformation_l1910_191042


namespace NUMINAMATH_GPT_percentage_politics_not_local_politics_l1910_191074

variables (total_reporters : ℝ) 
variables (reporters_cover_local_politics : ℝ) 
variables (reporters_not_cover_politics : ℝ)

theorem percentage_politics_not_local_politics :
  total_reporters = 100 → 
  reporters_cover_local_politics = 5 → 
  reporters_not_cover_politics = 92.85714285714286 → 
  (total_reporters - reporters_not_cover_politics) - reporters_cover_local_politics = 2.14285714285714 := 
by 
  intros ht hr hn
  rw [ht, hr, hn]
  norm_num


end NUMINAMATH_GPT_percentage_politics_not_local_politics_l1910_191074


namespace NUMINAMATH_GPT_relationship_m_n_l1910_191007

theorem relationship_m_n (m n : ℕ) (h : 10 / (m + 10 + n) = (m + n) / (m + 10 + n)) : m + n = 10 := 
by sorry

end NUMINAMATH_GPT_relationship_m_n_l1910_191007


namespace NUMINAMATH_GPT_vector_problem_l1910_191033

open Real

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  (v.1 ^ 2 + v.2 ^ 2) ^ (1 / 2)

variables (a b : ℝ × ℝ)
variables (h1 : a ≠ (0, 0)) (h2 : b ≠ (0, 0))
variables (h3 : a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0)
variables (h4 : 2 * magnitude a = magnitude b) (h5 : magnitude b =2)

theorem vector_problem : magnitude (2 * a.1 - b.1, 2 * a.2 - b.2) = 2 :=
sorry

end NUMINAMATH_GPT_vector_problem_l1910_191033


namespace NUMINAMATH_GPT_john_needs_total_planks_l1910_191080

theorem john_needs_total_planks : 
  let large_planks := 12
  let small_planks := 17
  large_planks + small_planks = 29 :=
by
  sorry

end NUMINAMATH_GPT_john_needs_total_planks_l1910_191080


namespace NUMINAMATH_GPT_find_m_l1910_191037

theorem find_m {m : ℕ} (h1 : Even (m^2 - 2 * m - 3)) (h2 : m^2 - 2 * m - 3 < 0) : m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_l1910_191037


namespace NUMINAMATH_GPT_second_discount_correct_l1910_191055

noncomputable def second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : ℝ :=
  let first_discount_amount := first_discount / 100 * original_price
  let price_after_first_discount := original_price - first_discount_amount
  let second_discount_amount := price_after_first_discount - final_price
  (second_discount_amount / price_after_first_discount) * 100

theorem second_discount_correct :
  second_discount_percentage 510 12 381.48 = 15 :=
by
  sorry

end NUMINAMATH_GPT_second_discount_correct_l1910_191055


namespace NUMINAMATH_GPT_time_to_cross_platform_l1910_191017

-- Definitions for the length of the train, the length of the platform, and the speed of the train
def length_train : ℕ := 750
def length_platform : ℕ := 750
def speed_train_kmh : ℕ := 90

-- Conversion constants
def meters_per_kilometer : ℕ := 1000
def seconds_per_hour : ℕ := 3600

-- Convert speed from km/hr to m/s
def speed_train_ms : ℚ := speed_train_kmh * meters_per_kilometer / seconds_per_hour

-- Total distance the train covers to cross the platform
def total_distance : ℕ := length_train + length_platform

-- Proof problem: To prove that the time taken to cross the platform is 60 seconds
theorem time_to_cross_platform : total_distance / speed_train_ms = 60 := by
  sorry

end NUMINAMATH_GPT_time_to_cross_platform_l1910_191017


namespace NUMINAMATH_GPT_three_point_three_seven_five_as_fraction_l1910_191035

theorem three_point_three_seven_five_as_fraction :
  3.375 = (27 / 8 : ℚ) :=
by sorry

end NUMINAMATH_GPT_three_point_three_seven_five_as_fraction_l1910_191035


namespace NUMINAMATH_GPT_completing_square_correct_l1910_191044

theorem completing_square_correct :
  ∀ x : ℝ, (x^2 - 4 * x + 2 = 0) ↔ ((x - 2)^2 = 2) := 
by
  intros x
  sorry

end NUMINAMATH_GPT_completing_square_correct_l1910_191044


namespace NUMINAMATH_GPT_paint_weight_correct_l1910_191090

def weight_of_paint (total_weight : ℕ) (half_paint_weight : ℕ) : ℕ :=
  2 * (total_weight - half_paint_weight)

theorem paint_weight_correct :
  weight_of_paint 24 14 = 20 := by 
  sorry

end NUMINAMATH_GPT_paint_weight_correct_l1910_191090


namespace NUMINAMATH_GPT_quarterly_insurance_payment_l1910_191023

theorem quarterly_insurance_payment (annual_payment : ℕ) (quarters_in_year : ℕ) (quarterly_payment : ℕ) : 
  annual_payment = 1512 → quarters_in_year = 4 → quarterly_payment * quarters_in_year = annual_payment → quarterly_payment = 378 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  sorry

end NUMINAMATH_GPT_quarterly_insurance_payment_l1910_191023


namespace NUMINAMATH_GPT_youth_gathering_l1910_191071

theorem youth_gathering (x : ℕ) (h1 : ∃ x, 9 * (2 * x + 12) = 20 * x) : 
  2 * x + 12 = 120 :=
by sorry

end NUMINAMATH_GPT_youth_gathering_l1910_191071


namespace NUMINAMATH_GPT_second_bus_percentage_full_l1910_191063

noncomputable def bus_capacity : ℕ := 150
noncomputable def employees_in_buses : ℕ := 195
noncomputable def first_bus_percentage : ℚ := 0.60

theorem second_bus_percentage_full :
  let employees_first_bus := first_bus_percentage * bus_capacity
  let employees_second_bus := (employees_in_buses : ℚ) - employees_first_bus
  let second_bus_percentage := (employees_second_bus / bus_capacity) * 100
  second_bus_percentage = 70 :=
by
  sorry

end NUMINAMATH_GPT_second_bus_percentage_full_l1910_191063


namespace NUMINAMATH_GPT_intersection_eq_l1910_191077

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2}

-- Prove the intersection of A and B is {0, 1}
theorem intersection_eq : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1910_191077


namespace NUMINAMATH_GPT_sum_gcd_lcm_l1910_191041

theorem sum_gcd_lcm (a b c d : ℕ) (ha : a = 15) (hb : b = 45) (hc : c = 30) :
  Int.gcd a b + Nat.lcm a c = 45 := 
by
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_l1910_191041


namespace NUMINAMATH_GPT_sqrt_difference_l1910_191094

theorem sqrt_difference : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_difference_l1910_191094


namespace NUMINAMATH_GPT_other_number_eq_462_l1910_191005

theorem other_number_eq_462 (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 4620) 
  (gcd_ab : Nat.gcd a b = 21) 
  (a_eq : a = 210) : b = 462 := 
by
  sorry

end NUMINAMATH_GPT_other_number_eq_462_l1910_191005


namespace NUMINAMATH_GPT_mean_score_l1910_191052

theorem mean_score (mu sigma : ℝ) 
  (h1 : 86 = mu - 7 * sigma) 
  (h2 : 90 = mu + 3 * sigma) :
  mu = 88.8 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_mean_score_l1910_191052


namespace NUMINAMATH_GPT_Hilltown_Volleyball_Club_Members_l1910_191014

-- Definitions corresponding to the conditions
def knee_pad_cost : ℕ := 6
def uniform_cost : ℕ := 14
def total_expenditure : ℕ := 4000

-- Definition of total cost per member
def cost_per_member : ℕ := 2 * (knee_pad_cost + uniform_cost)

-- Proof statement
theorem Hilltown_Volleyball_Club_Members :
  total_expenditure % cost_per_member = 0 ∧ total_expenditure / cost_per_member = 100 := by
    sorry

end NUMINAMATH_GPT_Hilltown_Volleyball_Club_Members_l1910_191014


namespace NUMINAMATH_GPT_sum_of_areas_B_D_l1910_191016

theorem sum_of_areas_B_D (area_large_square : ℝ) (area_small_square : ℝ) (B D : ℝ) 
  (h1 : area_large_square = 9) 
  (h2 : area_small_square = 1)
  (h3 : B + D = 4) : 
  B + D = 4 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_B_D_l1910_191016


namespace NUMINAMATH_GPT_average_weight_increase_l1910_191069

theorem average_weight_increase 
  (A : ℝ) (X : ℝ)
  (h1 : 8 * (A + X) = 8 * A + 36) :
  X = 4.5 := 
sorry

end NUMINAMATH_GPT_average_weight_increase_l1910_191069


namespace NUMINAMATH_GPT_probability_no_prize_l1910_191040

theorem probability_no_prize : (1 : ℚ) - (1 : ℚ) / (50 * 50) = 2499 / 2500 :=
by
  sorry

end NUMINAMATH_GPT_probability_no_prize_l1910_191040


namespace NUMINAMATH_GPT_coin_flip_probability_l1910_191002

noncomputable def probability_successful_outcomes : ℚ :=
  let total_outcomes := 32
  let successful_outcomes := 3
  successful_outcomes / total_outcomes

theorem coin_flip_probability :
  probability_successful_outcomes = 3 / 32 :=
by
  sorry

end NUMINAMATH_GPT_coin_flip_probability_l1910_191002


namespace NUMINAMATH_GPT_probability_of_X_le_1_l1910_191073

noncomputable def C (n k : ℕ) : ℚ := Nat.choose n k

noncomputable def P_X_le_1 := 
  (C 4 3 / C 6 3) + (C 4 2 * C 2 1 / C 6 3)

theorem probability_of_X_le_1 : P_X_le_1 = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_of_X_le_1_l1910_191073


namespace NUMINAMATH_GPT_volume_of_prism_l1910_191019

theorem volume_of_prism (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 6) :
  l * w * h = 30 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l1910_191019


namespace NUMINAMATH_GPT_tom_caught_16_trout_l1910_191086

theorem tom_caught_16_trout (melanie_trout : ℕ) (tom_caught_twice : melanie_trout * 2 = 16) : 
  2 * melanie_trout = 16 :=
by 
  sorry

end NUMINAMATH_GPT_tom_caught_16_trout_l1910_191086


namespace NUMINAMATH_GPT_range_of_m_values_l1910_191075

theorem range_of_m_values {P Q : ℝ × ℝ} (hP : P = (-1, 1)) (hQ : Q = (2, 2)) (m : ℝ) :
  -3 < m ∧ m < -2 / 3 → (∃ (l : ℝ → ℝ), ∀ x y, y = l x → x + m * y + m = 0) :=
sorry

end NUMINAMATH_GPT_range_of_m_values_l1910_191075


namespace NUMINAMATH_GPT_qin_jiushao_operations_required_l1910_191009

def polynomial (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem qin_jiushao_operations_required : 
  (∃ x : ℝ, polynomial x = (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1)) →
  (∃ m a : ℕ, m = 5 ∧ a = 5) := by
  sorry

end NUMINAMATH_GPT_qin_jiushao_operations_required_l1910_191009


namespace NUMINAMATH_GPT_possible_values_of_m_l1910_191012

-- Defining sets A and B based on the given conditions
def set_A : Set ℝ := { x | x^2 - 2 * x - 3 = 0 }
def set_B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

-- The main theorem statement
theorem possible_values_of_m (m : ℝ) :
  (set_A ∪ set_B m = set_A) ↔ (m = 0 ∨ m = -1 / 3 ∨ m = 1) := by
  sorry

end NUMINAMATH_GPT_possible_values_of_m_l1910_191012


namespace NUMINAMATH_GPT_isabella_original_hair_length_l1910_191013

-- Define conditions from the problem
def isabella_current_hair_length : ℕ := 9
def hair_cut_length : ℕ := 9

-- The proof problem to show original hair length equals 18 inches
theorem isabella_original_hair_length 
  (hc : isabella_current_hair_length = 9)
  (ht : hair_cut_length = 9) : 
  isabella_current_hhair_length + hair_cut_length = 18 := 
sorry

end NUMINAMATH_GPT_isabella_original_hair_length_l1910_191013


namespace NUMINAMATH_GPT_circumference_is_720_l1910_191081

-- Given conditions
def uniform_speed (A_speed B_speed : ℕ) : Prop := A_speed > 0 ∧ B_speed > 0
def diametrically_opposite_start (A_pos B_pos : ℕ) (circumference : ℕ) : Prop := A_pos = 0 ∧ B_pos = circumference / 2
def meets_first_after_B_travel (A_distance B_distance : ℕ) : Prop := B_distance = 150
def meets_second_90_yards_before_A_lap (A_distance_lap B_distance_lap A_distance B_distance : ℕ) : Prop := 
  A_distance_lap = A_distance + 2 * (A_distance - B_distance) - 90 ∧ B_distance_lap = A_distance - B_distance_lap + (B_distance + 90)

theorem circumference_is_720 (circumference A_speed B_speed A_pos B_pos
                     A_distance B_distance
                     A_distance_lap B_distance_lap : ℕ) :
  uniform_speed A_speed B_speed →
  diametrically_opposite_start A_pos B_pos circumference →
  meets_first_after_B_travel A_distance B_distance →
  meets_second_90_yards_before_A_lap A_distance_lap B_distance_lap A_distance B_distance →
  circumference = 720 :=
sorry

end NUMINAMATH_GPT_circumference_is_720_l1910_191081


namespace NUMINAMATH_GPT_carl_garden_area_l1910_191032

theorem carl_garden_area 
  (total_posts : Nat)
  (length_post_distance : Nat)
  (corner_posts : Nat)
  (longer_side_multiplier : Nat)
  (posts_per_shorter_side : Nat)
  (posts_per_longer_side : Nat)
  (shorter_side_distance : Nat)
  (longer_side_distance : Nat) :
  total_posts = 24 →
  length_post_distance = 5 →
  corner_posts = 4 →
  longer_side_multiplier = 2 →
  posts_per_shorter_side = (24 + 4) / 6 →
  posts_per_longer_side = (24 + 4) / 6 * 2 →
  shorter_side_distance = (posts_per_shorter_side - 1) * length_post_distance →
  longer_side_distance = (posts_per_longer_side - 1) * length_post_distance →
  shorter_side_distance * longer_side_distance = 900 :=
by
  intros
  sorry

end NUMINAMATH_GPT_carl_garden_area_l1910_191032


namespace NUMINAMATH_GPT_quadratic_discriminant_one_solution_l1910_191045

theorem quadratic_discriminant_one_solution (m : ℚ) : 
  (3 * (1 : ℚ))^2 - 12 * m = 0 → m = 49 / 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_discriminant_one_solution_l1910_191045


namespace NUMINAMATH_GPT_imaginary_part_of_complex_z_l1910_191021

noncomputable def complex_z : ℂ := (1 + Complex.I) / (1 - Complex.I) + (1 - Complex.I) ^ 2

theorem imaginary_part_of_complex_z : complex_z.im = -1 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_complex_z_l1910_191021


namespace NUMINAMATH_GPT_exists_k_ge_2_l1910_191006

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def weak (a b n : ℕ) : Prop :=
  ¬ ∃ x y : ℕ, a * x + b * y = n

theorem exists_k_ge_2 (a b n : ℕ) (h_coprime : coprime a b) (h_positive : 0 < n) (h_weak : weak a b n) (h_bound : n < a * b / 6) :
  ∃ k : ℕ, 2 ≤ k ∧ weak a b (k * n) :=
sorry

end NUMINAMATH_GPT_exists_k_ge_2_l1910_191006


namespace NUMINAMATH_GPT_megan_numbers_difference_l1910_191024

theorem megan_numbers_difference 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h_mean3 : (x1 + x2 + x3) / 3 = -3)
  (h_mean4 : (x1 + x2 + x3 + x4) / 4 = 4)
  (h_mean5 : (x1 + x2 + x3 + x4 + x5) / 5 = -5) :
  x4 - x5 = 66 :=
by
  sorry

end NUMINAMATH_GPT_megan_numbers_difference_l1910_191024


namespace NUMINAMATH_GPT_solution_to_first_equation_solution_to_second_equation_l1910_191054

theorem solution_to_first_equation (x : ℝ) : 
  x^2 - 6 * x + 1 = 0 ↔ x = 3 + 2 * Real.sqrt 2 ∨ x = 3 - 2 * Real.sqrt 2 :=
by sorry

theorem solution_to_second_equation (x : ℝ) : 
  (2 * x - 3)^2 = 5 * (2 * x - 3) ↔ x = 3 / 2 ∨ x = 4 :=
by sorry

end NUMINAMATH_GPT_solution_to_first_equation_solution_to_second_equation_l1910_191054


namespace NUMINAMATH_GPT_domain_of_ln_function_l1910_191058

theorem domain_of_ln_function (x : ℝ) : 3 - 4 * x > 0 ↔ x < 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_domain_of_ln_function_l1910_191058


namespace NUMINAMATH_GPT_iPhones_sold_l1910_191070

theorem iPhones_sold (x : ℕ) (h1 : (1000 * x + 18000 + 16000) / (x + 100) = 670) : x = 100 :=
by
  sorry

end NUMINAMATH_GPT_iPhones_sold_l1910_191070


namespace NUMINAMATH_GPT_total_cups_sold_is_46_l1910_191047

-- Define the number of cups sold last week
def cups_sold_last_week : ℕ := 20

-- Define the percentage increase
def percentage_increase : ℕ := 30

-- Calculate the number of cups sold this week
def cups_sold_this_week : ℕ := cups_sold_last_week + (cups_sold_last_week * percentage_increase / 100)

-- Calculate the total number of cups sold over both weeks
def total_cups_sold : ℕ := cups_sold_last_week + cups_sold_this_week

-- State the theorem to prove the total number of cups sold
theorem total_cups_sold_is_46 : total_cups_sold = 46 := sorry

end NUMINAMATH_GPT_total_cups_sold_is_46_l1910_191047


namespace NUMINAMATH_GPT_moles_H2O_formed_l1910_191046

-- Define the conditions
def moles_HCl : ℕ := 6
def moles_CaCO3 : ℕ := 3
def moles_CaCl2 : ℕ := 3
def moles_CO2 : ℕ := 3

-- Proposition that we need to prove
theorem moles_H2O_formed : moles_CaCl2 = 3 ∧ moles_CO2 = 3 ∧ moles_CaCO3 = 3 ∧ moles_HCl = 6 → moles_CaCO3 = 3 := by
  sorry

end NUMINAMATH_GPT_moles_H2O_formed_l1910_191046


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1910_191025

theorem geometric_sequence_sum
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (r : ℝ)
  (h1 : r ≠ 1)
  (h2 : ∀ n, S n = a 0 * (1 - r^(n + 1)) / (1 - r))
  (h3 : S 5 = 3)
  (h4 : S 10 = 9) :
  S 15 = 21 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1910_191025


namespace NUMINAMATH_GPT_find_a_plus_b_l1910_191056

theorem find_a_plus_b (a b : ℚ) (y : ℚ) (x : ℚ) :
  (y = a + b / x) →
  (2 = a + b / (-2 : ℚ)) →
  (3 = a + b / (-6 : ℚ)) →
  a + b = 13 / 2 :=
by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1910_191056


namespace NUMINAMATH_GPT_correct_operation_l1910_191060

theorem correct_operation : 
  ¬(3 * x^2 + 2 * x^2 = 6 * x^4) ∧ 
  ¬((-2 * x^2)^3 = -6 * x^6) ∧ 
  ¬(x^3 * x^2 = x^6) ∧ 
  (-6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1910_191060


namespace NUMINAMATH_GPT_cos_300_eq_half_l1910_191018

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_300_eq_half_l1910_191018


namespace NUMINAMATH_GPT_largest_gold_coins_l1910_191050

theorem largest_gold_coins (k : ℤ) (h1 : 13 * k + 3 < 100) : 91 ≤ 13 * k + 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_gold_coins_l1910_191050


namespace NUMINAMATH_GPT_hyperbola_m_range_l1910_191038

theorem hyperbola_m_range (m : ℝ) (h_eq : ∀ x y, (x^2 / m) - (y^2 / (2*m - 1)) = 1) : 
  0 < m ∧ m < 1/2 :=
sorry

end NUMINAMATH_GPT_hyperbola_m_range_l1910_191038


namespace NUMINAMATH_GPT_betty_garden_total_plants_l1910_191057

theorem betty_garden_total_plants (B O : ℕ) 
  (h1 : B = 5) 
  (h2 : O = 2 + 2 * B) : 
  B + O = 17 := by
  sorry

end NUMINAMATH_GPT_betty_garden_total_plants_l1910_191057


namespace NUMINAMATH_GPT_solve_for_s_l1910_191083

-- Definition of the condition
def condition (s : ℝ) : Prop := (s - 60) / 3 = (6 - 3 * s) / 4

-- Theorem stating that if the condition holds, then s = 19.85
theorem solve_for_s (s : ℝ) : condition s → s = 19.85 := 
by {
  sorry -- Proof is skipped as per requirements
}

end NUMINAMATH_GPT_solve_for_s_l1910_191083


namespace NUMINAMATH_GPT_chucks_team_final_score_l1910_191000

variable (RedTeamScore : ℕ) (scoreDifference : ℕ)

-- Given conditions
def red_team_score := RedTeamScore = 76
def score_difference := scoreDifference = 19

-- Question: What was the final score of Chuck's team?
def chucks_team_score (RedTeamScore scoreDifference : ℕ) : ℕ := 
  RedTeamScore + scoreDifference

-- Proof statement
theorem chucks_team_final_score : red_team_score 76 ∧ score_difference 19 → chucks_team_score 76 19 = 95 :=
by
  sorry

end NUMINAMATH_GPT_chucks_team_final_score_l1910_191000


namespace NUMINAMATH_GPT_max_sum_combined_shape_l1910_191051

-- Definitions for the initial prism
def faces_prism := 6
def edges_prism := 12
def vertices_prism := 8

-- Definitions for the changes when pyramid is added to a rectangular face
def additional_faces_rect := 4
def additional_edges_rect := 4
def additional_vertices_rect := 1

-- Definition for the maximum sum calculation
def max_sum := faces_prism - 1 + additional_faces_rect + 
               edges_prism + additional_edges_rect + 
               vertices_prism + additional_vertices_rect

-- The theorem to prove the maximum sum
theorem max_sum_combined_shape : max_sum = 34 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_combined_shape_l1910_191051


namespace NUMINAMATH_GPT_average_seq_13_to_52_l1910_191036

-- Define the sequence of natural numbers from 13 to 52
def seq : List ℕ := List.range' 13 52

-- Define the average of a list of natural numbers
def average (xs : List ℕ) : ℚ := (xs.sum : ℚ) / xs.length

-- Define the specific set of numbers and their average
theorem average_seq_13_to_52 : average seq = 32.5 := 
by 
  sorry

end NUMINAMATH_GPT_average_seq_13_to_52_l1910_191036


namespace NUMINAMATH_GPT_solve_equation_l1910_191026

noncomputable def equation_to_solve (x : ℝ) : ℝ :=
  1 / (4^(3*x) - 13 * 4^(2*x) + 51 * 4^x - 60) + 1 / (4^(2*x) - 7 * 4^x + 12)

theorem solve_equation :
  (equation_to_solve (1/2) = 0) ∧ (equation_to_solve (Real.log 6 / Real.log 4) = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_equation_l1910_191026


namespace NUMINAMATH_GPT_rationalize_cube_root_sum_l1910_191089

theorem rationalize_cube_root_sum :
  let a := (5 : ℝ)^(1/3)
  let b := (3 : ℝ)^(1/3)
  let numerator := a^2 + a * b + b^2
  let denom := a - b
  let fraction := 1 / denom * numerator
  let A := 25
  let B := 15
  let C := 9
  let D := 2
  A + B + C + D = 51 :=
by
  let a := (5 : ℝ)^(1/3)
  let b := (3 : ℝ)^(1/3)
  let numerator := a^2 + a * b + b^2
  let denom := a - b
  let fraction := 1 / denom * numerator
  let A := 25
  let B := 15
  let C := 9
  let D := 2
  have step1 : (a^3 = 5) := by sorry
  have step2 : (b^3 = 3) := by sorry
  have denom_eq : denom = 2 := by sorry
  have frac_simp : fraction = (A^(1/3) + B^(1/3) + C^(1/3)) / D := by sorry
  show A + B + C + D = 51
  sorry

end NUMINAMATH_GPT_rationalize_cube_root_sum_l1910_191089


namespace NUMINAMATH_GPT_remainder_of_95_times_97_div_12_l1910_191064

theorem remainder_of_95_times_97_div_12 : 
  (95 * 97) % 12 = 11 := by
  sorry

end NUMINAMATH_GPT_remainder_of_95_times_97_div_12_l1910_191064


namespace NUMINAMATH_GPT_find_a4_l1910_191079

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1) / 2) * d

theorem find_a4 (a d : ℤ)
    (h₁ : sum_first_n_terms a d 5 = 15)
    (h₂ : sum_first_n_terms a d 9 = 63) :
  arithmetic_sequence a d 4 = 5 :=
sorry

end NUMINAMATH_GPT_find_a4_l1910_191079


namespace NUMINAMATH_GPT_cost_rose_bush_l1910_191087

-- Define the constants
def total_roses := 6
def friend_roses := 2
def total_aloes := 2
def cost_aloe := 100
def total_spent_self := 500

-- Prove the cost of each rose bush
theorem cost_rose_bush : (total_spent_self - total_aloes * cost_aloe) / (total_roses - friend_roses) = 75 :=
by
  sorry

end NUMINAMATH_GPT_cost_rose_bush_l1910_191087


namespace NUMINAMATH_GPT_rectangle_breadth_approx_1_1_l1910_191061

theorem rectangle_breadth_approx_1_1 (s b : ℝ) (h1 : 4 * s = 2 * (16 + b))
  (h2 : abs ((π * s / 2) + s - 21.99) < 0.01) : abs (b - 1.1) < 0.01 :=
sorry

end NUMINAMATH_GPT_rectangle_breadth_approx_1_1_l1910_191061
