import Mathlib

namespace NUMINAMATH_GPT_problem1_problem2_l1949_194983

noncomputable def p (x a : ℝ) : Prop := x^2 + 4 * a * x + 3 * a^2 < 0
noncomputable def q (x : ℝ) : Prop := (x^2 - 6 * x - 72 ≤ 0) ∧ (x^2 + x - 6 > 0)
noncomputable def condition1 (a : ℝ) : Prop := 
  a = -1 ∧ (∃ x, p x a ∨ q x)

noncomputable def condition2 (a : ℝ) : Prop :=
  ∀ x, ¬ p x a → ¬ q x

theorem problem1 (x : ℝ) (a : ℝ) (h₁ : condition1 a) : -6 ≤ x ∧ x < -3 ∨ 1 < x ∧ x ≤ 12 := 
sorry

theorem problem2 (a : ℝ) (h₂ : condition2 a) : -4 ≤ a ∧ a ≤ -2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1949_194983


namespace NUMINAMATH_GPT_trucks_transportation_l1949_194940

theorem trucks_transportation (k : ℕ) (H : ℝ) : 
  (∃ (A B C : ℕ), 
     A + B + C = k ∧ 
     A ≤ k / 2 ∧ B ≤ k / 2 ∧ C ≤ k / 2 ∧ 
     (0 ≤ (k - 2*A)) ∧ (0 ≤ (k - 2*B)) ∧ (0 ≤ (k - 2*C))) 
  →  (k = 7 → (2 : ℕ) = 2) :=
sorry

end NUMINAMATH_GPT_trucks_transportation_l1949_194940


namespace NUMINAMATH_GPT_ed_money_left_l1949_194996

theorem ed_money_left
  (cost_per_hour_night : ℝ := 1.5)
  (cost_per_hour_morning : ℝ := 2)
  (initial_money : ℝ := 80)
  (hours_night : ℝ := 6)
  (hours_morning : ℝ := 4) :
  initial_money - (cost_per_hour_night * hours_night + cost_per_hour_morning * hours_morning) = 63 := 
  by
  sorry

end NUMINAMATH_GPT_ed_money_left_l1949_194996


namespace NUMINAMATH_GPT_countEquilateralTriangles_l1949_194946

-- Define the problem conditions
def numSmallTriangles := 18  -- The number of small equilateral triangles
def includesMarkedTriangle: Prop := True  -- All counted triangles include the marked triangle "**"

-- Define the main question as a proposition
def totalEquilateralTriangles : Prop :=
  (numSmallTriangles = 18 ∧ includesMarkedTriangle) → (1 + 4 + 1 = 6)

-- The theorem stating the number of equilateral triangles containing the marked triangle
theorem countEquilateralTriangles : totalEquilateralTriangles :=
  by
    sorry

end NUMINAMATH_GPT_countEquilateralTriangles_l1949_194946


namespace NUMINAMATH_GPT_perimeter_of_polygon_l1949_194915

-- Conditions
variables (a b : ℝ) (polygon_is_part_of_rectangle : 0 < a ∧ 0 < b)

-- Prove that if the polygon completes a rectangle with perimeter 28,
-- then the perimeter of the polygon is 28.
theorem perimeter_of_polygon (h : 2 * (a + b) = 28) : 2 * (a + b) = 28 :=
by
  exact h

end NUMINAMATH_GPT_perimeter_of_polygon_l1949_194915


namespace NUMINAMATH_GPT_profit_share_difference_correct_l1949_194945

noncomputable def profit_share_difference (a_capital b_capital c_capital b_profit : ℕ) : ℕ :=
  let total_parts := 4 + 5 + 6
  let part_size := b_profit / 5
  let a_profit := 4 * part_size
  let c_profit := 6 * part_size
  c_profit - a_profit

theorem profit_share_difference_correct :
  profit_share_difference 8000 10000 12000 1600 = 640 :=
by
  sorry

end NUMINAMATH_GPT_profit_share_difference_correct_l1949_194945


namespace NUMINAMATH_GPT_ice_cream_maker_completion_time_l1949_194967

def start_time := 9
def time_to_half := 3
def end_time := start_time + 2 * time_to_half

theorem ice_cream_maker_completion_time :
  end_time = 15 :=
by
  -- Definitions: 9:00 AM -> 9, 12:00 PM -> 12, 3:00 PM -> 15
  -- Calculation: end_time = 9 + 2 * 3 = 15
  sorry

end NUMINAMATH_GPT_ice_cream_maker_completion_time_l1949_194967


namespace NUMINAMATH_GPT_mass_percent_O_CaOH2_is_correct_mass_percent_O_Na2CO3_is_correct_mass_percent_O_K2SO4_is_correct_l1949_194931

-- Definitions for molar masses used in calculations
def molar_mass_Ca := 40.08
def molar_mass_O := 16.00
def molar_mass_H := 1.01
def molar_mass_Na := 22.99
def molar_mass_C := 12.01
def molar_mass_K := 39.10
def molar_mass_S := 32.07

-- Molar masses of the compounds
def molar_mass_CaOH2 := molar_mass_Ca + 2 * molar_mass_O + 2 * molar_mass_H
def molar_mass_Na2CO3 := 2 * molar_mass_Na + molar_mass_C + 3 * molar_mass_O
def molar_mass_K2SO4 := 2 * molar_mass_K + molar_mass_S + 4 * molar_mass_O

-- Mass of O in each compound
def mass_O_CaOH2 := 2 * molar_mass_O
def mass_O_Na2CO3 := 3 * molar_mass_O
def mass_O_K2SO4 := 4 * molar_mass_O

-- Mass percentages of O in each compound
def mass_percent_O_CaOH2 := (mass_O_CaOH2 / molar_mass_CaOH2) * 100
def mass_percent_O_Na2CO3 := (mass_O_Na2CO3 / molar_mass_Na2CO3) * 100
def mass_percent_O_K2SO4 := (mass_O_K2SO4 / molar_mass_K2SO4) * 100

theorem mass_percent_O_CaOH2_is_correct :
  mass_percent_O_CaOH2 = 43.19 := by sorry

theorem mass_percent_O_Na2CO3_is_correct :
  mass_percent_O_Na2CO3 = 45.29 := by sorry

theorem mass_percent_O_K2SO4_is_correct :
  mass_percent_O_K2SO4 = 36.73 := by sorry

end NUMINAMATH_GPT_mass_percent_O_CaOH2_is_correct_mass_percent_O_Na2CO3_is_correct_mass_percent_O_K2SO4_is_correct_l1949_194931


namespace NUMINAMATH_GPT_find_n_for_sine_equality_l1949_194993

theorem find_n_for_sine_equality : 
  ∃ (n: ℤ), -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.sin (670 * Real.pi / 180) ∧ n = -50 := by
  sorry

end NUMINAMATH_GPT_find_n_for_sine_equality_l1949_194993


namespace NUMINAMATH_GPT_allowance_amount_l1949_194943

variable (initial_money spent_money final_money : ℕ)

theorem allowance_amount (initial_money : ℕ) (spent_money : ℕ) (final_money : ℕ) (h1: initial_money = 5) (h2: spent_money = 2) (h3: final_money = 8) : (final_money - (initial_money - spent_money)) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_allowance_amount_l1949_194943


namespace NUMINAMATH_GPT_smaller_cuboid_length_l1949_194997

theorem smaller_cuboid_length
  (width_sm : ℝ)
  (height_sm : ℝ)
  (length_lg : ℝ)
  (width_lg : ℝ)
  (height_lg : ℝ)
  (num_sm : ℝ)
  (h1 : width_sm = 2)
  (h2 : height_sm = 3)
  (h3 : length_lg = 18)
  (h4 : width_lg = 15)
  (h5 : height_lg = 2)
  (h6 : num_sm = 18) :
  ∃ (length_sm : ℝ), (108 * length_sm = 540) ∧ (length_sm = 5) :=
by
  -- proof logic will be here
  sorry

end NUMINAMATH_GPT_smaller_cuboid_length_l1949_194997


namespace NUMINAMATH_GPT_lateral_surface_area_of_cylinder_l1949_194956

theorem lateral_surface_area_of_cylinder :
  (∀ (side_length : ℕ), side_length = 10 → 
  ∃ (lateral_surface_area : ℝ), lateral_surface_area = 100 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_cylinder_l1949_194956


namespace NUMINAMATH_GPT_part1_part2_l1949_194965

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≤ 0) : a ≥ 1 / Real.exp 1 :=
  sorry

noncomputable def g (x b : ℝ) : ℝ := Real.log x + 1/2 * x^2 - (b + 1) * x

theorem part2 (b : ℝ) (x1 x2 : ℝ) (h1 : b ≥ 3/2) (h2 : x1 < x2) (hx3 : g x1 b - g x2 b ≥ k) : k ≤ 15/8 - 2 * Real.log 2 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1949_194965


namespace NUMINAMATH_GPT_total_distance_collinear_centers_l1949_194979

theorem total_distance_collinear_centers (r1 r2 r3 : ℝ) (d12 d13 d23 : ℝ) 
  (h1 : r1 = 6) 
  (h2 : r2 = 14) 
  (h3 : d12 = r1 + r2) 
  (h4 : d13 = r3 - r1) 
  (h5 : d23 = r3 - r2) :
  d13 = d12 + r1 := by
  -- proof follows here
  sorry

end NUMINAMATH_GPT_total_distance_collinear_centers_l1949_194979


namespace NUMINAMATH_GPT_find_c_l1949_194991

theorem find_c (c : ℝ) : (∀ x : ℝ, -2 < x ∧ x < 1 → x^2 + x - c < 0) → c = 2 :=
by
  intros h
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_find_c_l1949_194991


namespace NUMINAMATH_GPT_total_charge_for_3_hours_l1949_194998

namespace TherapyCharges

-- Conditions
variables (A F : ℝ)
variable (h1 : F = A + 20)
variable (h2 : F + 4 * A = 300)

-- Prove that the total charge for 3 hours of therapy is 188
theorem total_charge_for_3_hours : F + 2 * A = 188 :=
by
  sorry

end TherapyCharges

end NUMINAMATH_GPT_total_charge_for_3_hours_l1949_194998


namespace NUMINAMATH_GPT_find_k_l1949_194966

theorem find_k :
  ∀ (k : ℤ),
    (∃ a1 a2 a3 : ℤ,
        a1 = 49 + k ∧
        a2 = 225 + k ∧
        a3 = 484 + k ∧
        2 * a2 = a1 + a3) →
    k = 324 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1949_194966


namespace NUMINAMATH_GPT_minimum_value_is_12_l1949_194913

noncomputable def smallest_possible_value (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) : ℝ :=
(a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (a + d)) + (1 / (b + c)) + (1 / (b + d)) + (1 / (c + d)))

theorem minimum_value_is_12 (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) :
  smallest_possible_value a b c d h1 h2 h3 h4 h5 h6 h7 ≥ 12 :=
sorry

end NUMINAMATH_GPT_minimum_value_is_12_l1949_194913


namespace NUMINAMATH_GPT_second_polygon_sides_l1949_194906

/--
Given two regular polygons where:
- The first polygon has 42 sides.
- Each side of the first polygon is three times the length of each side of the second polygon.
- The perimeters of both polygons are equal.
Prove that the second polygon has 126 sides.
-/
theorem second_polygon_sides
  (s : ℝ) -- the side length of the second polygon
  (h1 : ∃ n : ℕ, n = 42) -- the first polygon has 42 sides
  (h2 : ∃ m : ℝ, m = 3 * s) -- the side length of the first polygon is three times the side length of the second polygon
  (h3 : ∃ k : ℕ, k * (3 * s) = n * s) -- the perimeters of both polygons are equal
  : ∃ n2 : ℕ, n2 = 126 := 
by
  sorry

end NUMINAMATH_GPT_second_polygon_sides_l1949_194906


namespace NUMINAMATH_GPT_corn_syrup_amount_l1949_194922

-- Definitions based on given conditions
def flavoring_to_corn_syrup_standard := 1 / 12
def flavoring_to_water_standard := 1 / 30

def flavoring_to_corn_syrup_sport := (3 * flavoring_to_corn_syrup_standard)
def flavoring_to_water_sport := (1 / 2) * flavoring_to_water_standard

def common_factor := (30 : ℝ)

-- Amounts in sport formulation after adjustment
def flavoring_to_corn_syrup_ratio_sport := 1 / 4
def flavoring_to_water_ratio_sport := 1 / 60

def total_flavoring_corn_syrup := 15 -- Since ratio is 15:60:60 and given water is 15 ounces

theorem corn_syrup_amount (water_ounces : ℝ) :
  water_ounces = 15 → 
  (60 / 60) * water_ounces = 15 :=
by
  sorry

end NUMINAMATH_GPT_corn_syrup_amount_l1949_194922


namespace NUMINAMATH_GPT_sqrt_9_eq_pm3_l1949_194962

theorem sqrt_9_eq_pm3 : ∃ x : ℝ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_9_eq_pm3_l1949_194962


namespace NUMINAMATH_GPT_nonzero_rational_pow_zero_l1949_194901

theorem nonzero_rational_pow_zero 
  (num : ℤ) (denom : ℤ) (hnum : num = -1241376497) (hdenom : denom = 294158749357) (h_nonzero: num ≠ 0 ∧ denom ≠ 0) :
  (num / denom : ℚ) ^ 0 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_nonzero_rational_pow_zero_l1949_194901


namespace NUMINAMATH_GPT_exists_multiple_representations_l1949_194986

def V (n : ℕ) : Set ℕ := {m : ℕ | ∃ k : ℕ, m = 1 + k * n}

def indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V n ∧ ¬∃ (p q : ℕ), p ∈ V n ∧ q ∈ V n ∧ p * q = m

theorem exists_multiple_representations (n : ℕ) (h : 2 < n) :
  ∃ r ∈ V n, ∃ s t u v : ℕ, 
    indecomposable n s ∧ indecomposable n t ∧ indecomposable n u ∧ indecomposable n v ∧ 
    r = s * t ∧ r = u * v ∧ (s ≠ u ∨ t ≠ v) :=
sorry

end NUMINAMATH_GPT_exists_multiple_representations_l1949_194986


namespace NUMINAMATH_GPT_three_digit_numbers_l1949_194976

theorem three_digit_numbers (N : ℕ) (a b c : ℕ) 
  (h1 : N = 100 * a + 10 * b + c)
  (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : b ≤ 9 ∧ c ≤ 9)
  (h4 : a - b + c % 11 = 0)
  (h5 : N % 11 = 0)
  (h6 : N = 11 * (a^2 + b^2 + c^2)) :
  N = 550 ∨ N = 803 :=
  sorry

end NUMINAMATH_GPT_three_digit_numbers_l1949_194976


namespace NUMINAMATH_GPT_sum_of_powers_eight_l1949_194937

variable {a b : ℝ}

theorem sum_of_powers_eight :
  a + b = 1 → 
  a^2 + b^2 = 3 → 
  a^3 + b^3 = 4 → 
  a^4 + b^4 = 7 → 
  a^5 + b^5 = 11 → 
  a^8 + b^8 = 47 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_sum_of_powers_eight_l1949_194937


namespace NUMINAMATH_GPT_find_n_divisible_by_highest_power_of_2_l1949_194934

def a_n (n : ℕ) : ℕ :=
  10^n * 999 + 488

theorem find_n_divisible_by_highest_power_of_2:
  ∀ n : ℕ, (n > 0) → (a_n n = 10^n * 999 + 488) → (∃ k : ℕ, 2^(k + 9) ∣ a_n 6) := sorry

end NUMINAMATH_GPT_find_n_divisible_by_highest_power_of_2_l1949_194934


namespace NUMINAMATH_GPT_points_lie_on_circle_l1949_194920

theorem points_lie_on_circle (t : ℝ) : 
  let x := Real.cos t
  let y := Real.sin t
  x^2 + y^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_points_lie_on_circle_l1949_194920


namespace NUMINAMATH_GPT_lattice_points_in_region_l1949_194970

theorem lattice_points_in_region : ∃ n : ℕ, n = 1 ∧ ∀ p : ℤ × ℤ, 
  (p.snd = abs p.fst ∨ p.snd = -(p.fst ^ 3) + 6 * (p.fst)) → n = 1 :=
by
  sorry

end NUMINAMATH_GPT_lattice_points_in_region_l1949_194970


namespace NUMINAMATH_GPT_book_costs_and_scenarios_l1949_194951

theorem book_costs_and_scenarios :
  (∃ (x y : ℕ), x + 3 * y = 180 ∧ 3 * x + y = 140 ∧ 
    (x = 30) ∧ (y = 50)) ∧ 
  (∀ (m : ℕ), (30 * m + 75 * m) ≤ 700 → (∃ (m_values : Finset ℕ), 
    m_values = {2, 4, 6} ∧ (m ∈ m_values))) :=
  sorry

end NUMINAMATH_GPT_book_costs_and_scenarios_l1949_194951


namespace NUMINAMATH_GPT_least_possible_integer_discussed_l1949_194953
open Nat

theorem least_possible_integer_discussed (N : ℕ) (H : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 → k ≠ 8 ∧ k ≠ 9 → k ∣ N) : N = 2329089562800 :=
sorry

end NUMINAMATH_GPT_least_possible_integer_discussed_l1949_194953


namespace NUMINAMATH_GPT_car_trip_eq_560_miles_l1949_194963

noncomputable def car_trip_length (v L : ℝ) :=
  -- Conditions from the problem
  -- 1. Car travels for 2 hours before the delay
  let pre_delay_time := 2
  -- 2. Delay time is 1 hour
  let delay_time := 1
  -- 3. Post-delay speed is 2/3 of the initial speed
  let post_delay_speed := (2 / 3) * v
  -- 4. Car arrives 4 hours late under initial scenario:
  let late_4_hours_time := 2 + 1 + (3 * (L - 2 * v)) / (2 * v)
  -- Expected travel time without any delays is 2 + (L / v)
  -- Difference indicates delay of 4 hours
  let without_delay_time := (L / v)
  let time_diff_late_4 := (late_4_hours_time - without_delay_time = 4)
  -- 5. Delay 120 miles farther, car arrives 3 hours late
  let delay_120_miles_farther := 120
  let late_3_hours_time := 2 + delay_120_miles_farther / v + 1 + (3 * (L - 2 * v - 120)) / (2 * v)
  let time_diff_late_3 := (late_3_hours_time - without_delay_time = 3)

  -- Combining conditions to solve for L
  -- Goal: Prove L = 560
  L = 560 -> time_diff_late_4 ∧ time_diff_late_3

theorem car_trip_eq_560_miles (v : ℝ) : ∃ (L : ℝ), car_trip_length v L := 
by 
  sorry

end NUMINAMATH_GPT_car_trip_eq_560_miles_l1949_194963


namespace NUMINAMATH_GPT_total_wood_needed_l1949_194950

theorem total_wood_needed : 
      (4 * 4 + 4 * (4 * 5)) + 
      (10 * 6 + 10 * (6 - 3)) + 
      (8 * 5.5) + 
      (6 * (5.5 * 2) + 6 * (5.5 * 1.5)) = 345.5 := 
by 
  sorry

end NUMINAMATH_GPT_total_wood_needed_l1949_194950


namespace NUMINAMATH_GPT_solve_triangle_l1949_194942

open Real

noncomputable def triangle_sides_angles (a b c A B C : ℝ) : Prop :=
  b^2 - (2 * (sqrt 3 / 3) * b * c * sin A) + c^2 = a^2

theorem solve_triangle 
  (b c : ℝ) (hb : b = 2) (hc : c = 3)
  (h : triangle_sides_angles a b c A B C) : 
  (A = π / 3) ∧ 
  (a = sqrt 7) ∧ 
  (sin (2 * B - A) = 3 * sqrt 3 / 14) := 
by
  sorry

end NUMINAMATH_GPT_solve_triangle_l1949_194942


namespace NUMINAMATH_GPT_jaden_toy_cars_problem_l1949_194961

theorem jaden_toy_cars_problem :
  let initial := 14
  let bought := 28
  let birthday := 12
  let to_vinnie := 3
  let left := 43
  let total := initial + bought + birthday
  let after_vinnie := total - to_vinnie
  (after_vinnie - left = 8) :=
by
  sorry

end NUMINAMATH_GPT_jaden_toy_cars_problem_l1949_194961


namespace NUMINAMATH_GPT_johns_avg_speed_l1949_194930

/-
John cycled 40 miles at 8 miles per hour and 20 miles at 40 miles per hour.
We want to prove that his average speed for the entire trip is 10.91 miles per hour.
-/

theorem johns_avg_speed :
  let distance1 := 40
  let speed1 := 8
  let distance2 := 20
  let speed2 := 40
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_distance / total_time
  avg_speed = 10.91 :=
by
  sorry

end NUMINAMATH_GPT_johns_avg_speed_l1949_194930


namespace NUMINAMATH_GPT_quadratic_equation_C_has_real_solutions_l1949_194927

theorem quadratic_equation_C_has_real_solutions :
  ∀ (x : ℝ), ∃ (a b c : ℝ), a = 1 ∧ b = 3 ∧ c = -2 ∧ a*x^2 + b*x + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_C_has_real_solutions_l1949_194927


namespace NUMINAMATH_GPT_total_spending_is_140_l1949_194957

-- Define definitions for each day's spending based on the conditions.
def monday_spending : ℕ := 6
def tuesday_spending : ℕ := 2 * monday_spending
def wednesday_spending : ℕ := 2 * (monday_spending + tuesday_spending)
def thursday_spending : ℕ := (monday_spending + tuesday_spending + wednesday_spending) / 3
def friday_spending : ℕ := thursday_spending - 4
def saturday_spending : ℕ := friday_spending + (friday_spending / 2)
def sunday_spending : ℕ := tuesday_spending + saturday_spending

-- The total spending for the week.
def total_spending : ℕ := 
  monday_spending + 
  tuesday_spending + 
  wednesday_spending + 
  thursday_spending + 
  friday_spending + 
  saturday_spending + 
  sunday_spending

-- The theorem to prove that the total spending is $140.
theorem total_spending_is_140 : total_spending = 140 := 
  by {
    -- Due to the problem's requirement, we skip the proof steps.
    sorry
  }

end NUMINAMATH_GPT_total_spending_is_140_l1949_194957


namespace NUMINAMATH_GPT_quadratic_factor_n_l1949_194974

theorem quadratic_factor_n (n : ℤ) (h : ∃ m : ℤ, (x + 5) * (x + m) = x^2 + 7 * x + n) : n = 10 :=
sorry

end NUMINAMATH_GPT_quadratic_factor_n_l1949_194974


namespace NUMINAMATH_GPT_divisible_by_17_l1949_194952

theorem divisible_by_17 (n : ℕ) : 17 ∣ (2 ^ (5 * n + 3) + 5 ^ n * 3 ^ (n + 2)) := 
by {
  sorry
}

end NUMINAMATH_GPT_divisible_by_17_l1949_194952


namespace NUMINAMATH_GPT_bahs_equivalent_to_1500_yahs_l1949_194994

-- Definitions from conditions
def bahs := ℕ
def rahs := ℕ
def yahs := ℕ

-- Conversion ratios given in conditions
def ratio_bah_rah : ℚ := 10 / 16
def ratio_rah_yah : ℚ := 9 / 15

-- Given the conditions
def condition1 (b r : ℚ) : Prop := b / r = ratio_bah_rah
def condition2 (r y : ℚ) : Prop := r / y = ratio_rah_yah

-- Goal: proving the question
theorem bahs_equivalent_to_1500_yahs (b : ℚ) (r : ℚ) (y : ℚ)
  (h1 : condition1 b r) (h2 : condition2 r y) : b * (1500 / y) = 562.5
:=
sorry

end NUMINAMATH_GPT_bahs_equivalent_to_1500_yahs_l1949_194994


namespace NUMINAMATH_GPT_gunny_bag_capacity_l1949_194988

def pounds_per_ton : ℝ := 2500
def ounces_per_pound : ℝ := 16
def packets : ℝ := 2000
def packet_weight_pounds : ℝ := 16
def packet_weight_ounces : ℝ := 4

theorem gunny_bag_capacity :
  (packets * (packet_weight_pounds + packet_weight_ounces / ounces_per_pound) / pounds_per_ton) = 13 := 
by
  sorry

end NUMINAMATH_GPT_gunny_bag_capacity_l1949_194988


namespace NUMINAMATH_GPT_simplify_expression_l1949_194909

variable (a b : ℤ)

theorem simplify_expression :
  (15 * a + 45 * b) + (20 * a + 35 * b) - (25 * a + 55 * b) + (30 * a - 5 * b) = 
  40 * a + 20 * b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1949_194909


namespace NUMINAMATH_GPT_pairs_divisible_by_7_l1949_194908

theorem pairs_divisible_by_7 :
  (∃ (pairs : List (ℕ × ℕ)), 
    (∀ p ∈ pairs, (1 ≤ p.fst ∧ p.fst ≤ 1000) ∧ (1 ≤ p.snd ∧ p.snd ≤ 1000) ∧ (p.fst^2 + p.snd^2) % 7 = 0) ∧ 
    pairs.length = 20164) :=
sorry

end NUMINAMATH_GPT_pairs_divisible_by_7_l1949_194908


namespace NUMINAMATH_GPT_inequality_proof_l1949_194911

theorem inequality_proof (a b : ℝ) (h : (a = 0 ∨ b = 0 ∨ (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0))) :
  a^4 + 2*a^3*b + 2*a*b^3 + b^4 ≥ 6*a^2*b^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1949_194911


namespace NUMINAMATH_GPT_piles_can_be_combined_l1949_194941

-- Define a predicate indicating that two integers x and y are similar sizes
def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

-- Define a function stating that we can combine piles while maintaining the similar sizes property
noncomputable def combine_piles (piles : List ℕ) : ℕ :=
  sorry

-- State the theorem where we prove that any initial configuration of piles can be combined into a single pile
theorem piles_can_be_combined (piles : List ℕ) :
  ∃ n : ℕ, combine_piles piles = n :=
by sorry

end NUMINAMATH_GPT_piles_can_be_combined_l1949_194941


namespace NUMINAMATH_GPT_symmetric_circle_proof_l1949_194905

-- Define the original circle equation
def original_circle_eq (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop :=
  y = x

-- Define the symmetric circle equation
def symmetric_circle_eq (x y : ℝ) : Prop :=
  x^2 + (y + 2)^2 = 5

-- The theorem to prove
theorem symmetric_circle_proof (x y : ℝ) :
  (original_circle_eq x y) ↔ (symmetric_circle_eq x y) :=
sorry

end NUMINAMATH_GPT_symmetric_circle_proof_l1949_194905


namespace NUMINAMATH_GPT_sum_f_x₁_f_x₂_lt_0_l1949_194914

variable (f : ℝ → ℝ)
variable (x₁ x₂ : ℝ)

-- Condition: y = f(x + 2) is an odd function
def odd_function_on_shifted_domain : Prop :=
  ∀ x, f (x + 2) = -f (-(x + 2))

-- Condition: f(x) is monotonically increasing for x > 2
def monotonically_increasing_for_x_gt_2 : Prop :=
  ∀ x₁ x₂, 2 < x₁ → x₁ < x₂ → f x₁ < f x₂

-- Condition: x₁ + x₂ < 4
def sum_lt_4 : Prop :=
  x₁ + x₂ < 4

-- Condition: (x₁-2)(x₂-2) < 0
def product_shift_lt_0 : Prop :=
  (x₁ - 2) * (x₂ - 2) < 0

-- Main theorem to prove f(x₁) + f(x₂) < 0
theorem sum_f_x₁_f_x₂_lt_0
  (h1 : odd_function_on_shifted_domain f)
  (h2 : monotonically_increasing_for_x_gt_2 f)
  (h3 : sum_lt_4 x₁ x₂)
  (h4 : product_shift_lt_0 x₁ x₂) :
  f x₁ + f x₂ < 0 := sorry

end NUMINAMATH_GPT_sum_f_x₁_f_x₂_lt_0_l1949_194914


namespace NUMINAMATH_GPT_bathtub_problem_l1949_194912

theorem bathtub_problem (T : ℝ) (h1 : 1 / T - 1 / 12 = 1 / 60) : T = 10 := 
by {
  -- Sorry, skip the proof as requested
  sorry
}

end NUMINAMATH_GPT_bathtub_problem_l1949_194912


namespace NUMINAMATH_GPT_annual_income_is_32000_l1949_194972

noncomputable def compute_tax (p A: ℝ) : ℝ := 
  0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000)

noncomputable def stated_tax (p A: ℝ) : ℝ := 
  0.01 * (p + 0.25) * A

theorem annual_income_is_32000 (p : ℝ) (A : ℝ) :
  compute_tax p A = stated_tax p A → A = 32000 :=
by
  intros h
  have : 0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = 0.01 * (p + 0.25) * A := h
  sorry

end NUMINAMATH_GPT_annual_income_is_32000_l1949_194972


namespace NUMINAMATH_GPT_bike_distance_from_rest_l1949_194971

variable (u : ℝ) (a : ℝ) (t : ℝ)

theorem bike_distance_from_rest (h1 : u = 0) (h2 : a = 0.5) (h3 : t = 8) : 
  (1 / 2 * a * t^2 = 16) :=
by
  sorry

end NUMINAMATH_GPT_bike_distance_from_rest_l1949_194971


namespace NUMINAMATH_GPT_statement_3_correct_l1949_194995

-- Definitions based on the conditions
def DeductiveReasoningGeneralToSpecific := True
def SyllogismForm := True
def ConclusionDependsOnPremisesAndForm := True

-- Proof problem statement
theorem statement_3_correct : SyllogismForm := by
  exact True.intro

end NUMINAMATH_GPT_statement_3_correct_l1949_194995


namespace NUMINAMATH_GPT_four_point_questions_l1949_194990

theorem four_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : y = 10 :=
sorry

end NUMINAMATH_GPT_four_point_questions_l1949_194990


namespace NUMINAMATH_GPT_brad_weighs_more_l1949_194960

theorem brad_weighs_more :
  ∀ (Billy Brad Carl : ℕ), 
    (Billy = Brad + 9) → 
    (Carl = 145) → 
    (Billy = 159) → 
    (Brad - Carl = 5) :=
by
  intros Billy Brad Carl h1 h2 h3
  sorry

end NUMINAMATH_GPT_brad_weighs_more_l1949_194960


namespace NUMINAMATH_GPT_remainder_18_pow_63_mod_5_l1949_194981

theorem remainder_18_pow_63_mod_5 :
  (18:ℤ) ^ 63 % 5 = 2 :=
by
  -- Given conditions
  have h1 : (18:ℤ) % 5 = 3 := by norm_num
  have h2 : (3:ℤ) ^ 4 % 5 = 1 := by norm_num
  sorry

end NUMINAMATH_GPT_remainder_18_pow_63_mod_5_l1949_194981


namespace NUMINAMATH_GPT_people_got_rid_of_some_snails_l1949_194999

namespace SnailProblem

def originalSnails : ℕ := 11760
def remainingSnails : ℕ := 8278
def snailsGotRidOf : ℕ := 3482

theorem people_got_rid_of_some_snails :
  originalSnails - remainingSnails = snailsGotRidOf :=
by 
  sorry

end SnailProblem

end NUMINAMATH_GPT_people_got_rid_of_some_snails_l1949_194999


namespace NUMINAMATH_GPT_remainder_sum_mod_13_l1949_194902

theorem remainder_sum_mod_13 : (1230 + 1231 + 1232 + 1233 + 1234) % 13 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_mod_13_l1949_194902


namespace NUMINAMATH_GPT_value_of_a_l1949_194907

theorem value_of_a 
  (a b c d e : ℤ)
  (h1 : a + 4 = b + 2)
  (h2 : a + 2 = b)
  (h3 : a + c = 146)
  (he : e = 79)
  (h4 : e = d + 2)
  (h5 : d = c + 2)
  (h6 : c = b + 2) :
  a = 71 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1949_194907


namespace NUMINAMATH_GPT_largest_value_l1949_194964

def expr_A : ℕ := 3 + 1 + 0 + 5
def expr_B : ℕ := 3 * 1 + 0 + 5
def expr_C : ℕ := 3 + 1 * 0 + 5
def expr_D : ℕ := 3 * 1 + 0 * 5
def expr_E : ℕ := 3 * 1 + 0 * 5 * 3

theorem largest_value :
  expr_A > expr_B ∧
  expr_A > expr_C ∧
  expr_A > expr_D ∧
  expr_A > expr_E :=
by
  sorry

end NUMINAMATH_GPT_largest_value_l1949_194964


namespace NUMINAMATH_GPT_domain_f_a_5_abs_inequality_ab_l1949_194978

-- Definition for the domain of f(x) when a=5
def domain_of_f_a_5 (x : ℝ) : Prop := |x + 1| + |x + 2| - 5 ≥ 0

-- The theorem to find the domain A of the function f(x) when a=5.
theorem domain_f_a_5 (x : ℝ) : domain_of_f_a_5 x ↔ (x ≤ -4 ∨ x ≥ 1) :=
by
  sorry

-- Theorem to prove the inequality for a, b ∈ (-1, 1)
theorem abs_inequality_ab (a b : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) :
  |a + b| / 2 < |1 + a * b / 4| :=
by
  sorry

end NUMINAMATH_GPT_domain_f_a_5_abs_inequality_ab_l1949_194978


namespace NUMINAMATH_GPT_oatmeal_cookies_divisible_by_6_l1949_194938

theorem oatmeal_cookies_divisible_by_6 (O : ℕ) (h1 : 48 % 6 = 0) (h2 : O % 6 = 0) :
    ∃ x : ℕ, O = 6 * x :=
by sorry

end NUMINAMATH_GPT_oatmeal_cookies_divisible_by_6_l1949_194938


namespace NUMINAMATH_GPT_sequence_general_formula_l1949_194968

theorem sequence_general_formula (n : ℕ) (h : n > 0) :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ (∀ n, a (n + 1) = a n / (3 * a n + 1)) ∧ a n = 1 / (3 * n - 2) :=
by sorry

end NUMINAMATH_GPT_sequence_general_formula_l1949_194968


namespace NUMINAMATH_GPT_weight_of_new_person_l1949_194933

theorem weight_of_new_person (W : ℝ) (N : ℝ) (h1 : (W + (8 * 2.5)) = (W - 20 + N)) : N = 40 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l1949_194933


namespace NUMINAMATH_GPT_largest_sphere_surface_area_in_cone_l1949_194949

theorem largest_sphere_surface_area_in_cone :
  (∀ (r : ℝ), (∃ (r : ℝ), r > 0 ∧ (1^2 + (3^2 - r^2) = 3^2)) →
    4 * π * r^2 ≤ 2 * π) :=
by
  sorry

end NUMINAMATH_GPT_largest_sphere_surface_area_in_cone_l1949_194949


namespace NUMINAMATH_GPT_emily_subtracts_99_l1949_194932

theorem emily_subtracts_99 (a b : ℕ) : (a = 50) → (b = 1) → (49^2 = 50^2 - 99) :=
by
  sorry

end NUMINAMATH_GPT_emily_subtracts_99_l1949_194932


namespace NUMINAMATH_GPT_Ravi_Prakash_finish_together_l1949_194936

-- Definitions based on conditions
def Ravi_time := 24
def Prakash_time := 40

-- Main theorem statement
theorem Ravi_Prakash_finish_together :
  (1 / Ravi_time + 1 / Prakash_time) = 1 / 15 :=
by
  sorry

end NUMINAMATH_GPT_Ravi_Prakash_finish_together_l1949_194936


namespace NUMINAMATH_GPT_perimeter_of_triangle_hyperbola_l1949_194989

theorem perimeter_of_triangle_hyperbola (x y : ℝ) (F1 F2 A B : ℝ) :
  (x^2 / 16) - (y^2 / 9) = 1 →
  |A - F2| - |A - F1| = 8 →
  |B - F2| - |B - F1| = 8 →
  |B - A| = 5 →
  |A - F2| + |B - F2| + |B - A| = 26 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_hyperbola_l1949_194989


namespace NUMINAMATH_GPT_value_of_x_l1949_194984

theorem value_of_x (x : ℝ) (h : x = 52 * (1 + 20 / 100)) : x = 62.4 :=
by sorry

end NUMINAMATH_GPT_value_of_x_l1949_194984


namespace NUMINAMATH_GPT_not_characteristic_of_algorithm_l1949_194939

def characteristic_of_algorithm (c : String) : Prop :=
  c = "Abstraction" ∨ c = "Precision" ∨ c = "Finiteness"

theorem not_characteristic_of_algorithm : 
  ¬ characteristic_of_algorithm "Uniqueness" :=
by
  sorry

end NUMINAMATH_GPT_not_characteristic_of_algorithm_l1949_194939


namespace NUMINAMATH_GPT_wilson_theorem_non_prime_divisibility_l1949_194929

theorem wilson_theorem (p : ℕ) (h : Nat.Prime p) : p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

theorem non_prime_divisibility (p : ℕ) (h : ¬ Nat.Prime p) : ¬ p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

end NUMINAMATH_GPT_wilson_theorem_non_prime_divisibility_l1949_194929


namespace NUMINAMATH_GPT_unit_vector_parallel_to_d_l1949_194919

theorem unit_vector_parallel_to_d (x y: ℝ): (4 * x - 3 * y = 0) ∧ (x^2 + y^2 = 1) → (x = 3/5 ∧ y = 4/5) ∨ (x = -3/5 ∧ y = -4/5) :=
by sorry

end NUMINAMATH_GPT_unit_vector_parallel_to_d_l1949_194919


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1949_194955

-- Define a and b with given values
def a := 1 / 2
def b := 1 / 3

-- Define the expression
def expr := 5 * (3 * a ^ 2 * b - a * b ^ 2) - (a * b ^ 2 + 3 * a ^ 2 * b)

-- State the theorem
theorem simplify_and_evaluate_expression : expr = 2 / 3 := 
by
  -- Proof can be inserted here
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1949_194955


namespace NUMINAMATH_GPT_max_square_test_plots_l1949_194987

theorem max_square_test_plots (h_field_dims : (24 : ℝ) = 24 ∧ (52 : ℝ) = 52)
    (h_total_fencing : 1994 = 1994)
    (h_partitioning : ∀ (n : ℤ), n % 6 = 0 → n ≤ 19 → 
      (104 * n - 76 ≤ 1994) → (n / 6 * 13)^2 = 702) :
    ∃ n : ℤ, (n / 6 * 13)^2 = 702 := sorry

end NUMINAMATH_GPT_max_square_test_plots_l1949_194987


namespace NUMINAMATH_GPT_find_m_value_l1949_194928

-- Definitions for the problem conditions are given below
variables (m : ℝ)

-- Conditions
def conditions := (6 < m) ∧ (m < 10) ∧ (4 = 2 * 2) ∧ (4 = (m - 2) - (10 - m))

-- Proof statement
theorem find_m_value : conditions m → m = 8 :=
sorry

end NUMINAMATH_GPT_find_m_value_l1949_194928


namespace NUMINAMATH_GPT_jerry_trays_l1949_194925

theorem jerry_trays :
  ∀ (trays_from_table1 trays_from_table2 trips trays_per_trip : ℕ),
  trays_from_table1 = 9 →
  trays_from_table2 = 7 →
  trips = 2 →
  trays_from_table1 + trays_from_table2 = 16 →
  trays_per_trip = (trays_from_table1 + trays_from_table2) / trips →
  trays_per_trip = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jerry_trays_l1949_194925


namespace NUMINAMATH_GPT_tangent_line_is_correct_l1949_194969

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, -1)

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := -3 * x + 2

-- Statement of the problem (to prove)
theorem tangent_line_is_correct :
  curve point_of_tangency.1 = point_of_tangency.2 ∧
  ∃ m b, (∀ x, (tangent_line x) = m * x + b) ∧
         tangent_line point_of_tangency.1 = point_of_tangency.2 ∧
         (∀ x, deriv (curve) x = -3 ↔ deriv (tangent_line) point_of_tangency.1 = -3) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_is_correct_l1949_194969


namespace NUMINAMATH_GPT_range_of_m_l1949_194985

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x + m) * (2 - x) < 1) ↔ (-4 < m ∧ m < 0) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1949_194985


namespace NUMINAMATH_GPT_shortest_distance_from_origin_l1949_194904

noncomputable def shortest_distance_to_circle (x y : ℝ) : ℝ :=
  x^2 + 6 * x + y^2 - 8 * y + 18

theorem shortest_distance_from_origin :
  ∃ (d : ℝ), d = 5 - Real.sqrt 7 ∧ ∀ (x y : ℝ), shortest_distance_to_circle x y = 0 →
    (Real.sqrt ((x - 0)^2 + (y - 0)^2) - Real.sqrt ((x + 3)^2 + (y - 4)^2)) = d := sorry

end NUMINAMATH_GPT_shortest_distance_from_origin_l1949_194904


namespace NUMINAMATH_GPT_max_n_base_10_l1949_194959

theorem max_n_base_10:
  ∃ (A B C n: ℕ), (A < 5 ∧ B < 5 ∧ C < 5) ∧
                 (n = 25 * A + 5 * B + C) ∧ (n = 81 * C + 9 * B + A) ∧ 
                 (∀ (A' B' C' n': ℕ), 
                 (A' < 5 ∧ B' < 5 ∧ C' < 5) ∧ (n' = 25 * A' + 5 * B' + C') ∧ 
                 (n' = 81 * C' + 9 * B' + A') → n' ≤ n) →
  n = 111 :=
by {
    sorry
}

end NUMINAMATH_GPT_max_n_base_10_l1949_194959


namespace NUMINAMATH_GPT_least_number_to_add_l1949_194948

theorem least_number_to_add (x : ℕ) : (1021 + x) % 25 = 0 ↔ x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_least_number_to_add_l1949_194948


namespace NUMINAMATH_GPT_ff_of_10_eq_2_l1949_194944

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then x^2 + 1 else Real.log x

theorem ff_of_10_eq_2 : f (f 10) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ff_of_10_eq_2_l1949_194944


namespace NUMINAMATH_GPT_K_3_15_10_eq_151_30_l1949_194947

def K (a b c : ℕ) : ℚ := (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a

theorem K_3_15_10_eq_151_30 : K 3 15 10 = 151 / 30 := 
by
  sorry

end NUMINAMATH_GPT_K_3_15_10_eq_151_30_l1949_194947


namespace NUMINAMATH_GPT_sector_area_l1949_194923

theorem sector_area (r θ : ℝ) (hr : r = 1) (hθ : θ = 2) : 
  (1 / 2) * r * r * θ = 1 := by
sorry

end NUMINAMATH_GPT_sector_area_l1949_194923


namespace NUMINAMATH_GPT_tangent_slope_at_pi_over_four_l1949_194977

theorem tangent_slope_at_pi_over_four :
  deriv (fun x => Real.tan x) (Real.pi / 4) = 2 :=
sorry

end NUMINAMATH_GPT_tangent_slope_at_pi_over_four_l1949_194977


namespace NUMINAMATH_GPT_value_of_x_l1949_194918

theorem value_of_x (x : ℝ) (h : x = 90 + (11 / 100) * 90) : x = 99.9 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_x_l1949_194918


namespace NUMINAMATH_GPT_decimal_zeros_l1949_194935

theorem decimal_zeros (h : 2520 = 2^3 * 3^2 * 5 * 7) : 
  ∃ (n : ℕ), n = 2 ∧ (∃ d : ℚ, d = 5 / 2520 ∧ ↑d = 0.004) :=
by
  -- We assume the factorization of 2520 is correct
  have h_fact := h
  -- We need to prove there are exactly 2 zeros between the decimal point and the first non-zero digit
  sorry

end NUMINAMATH_GPT_decimal_zeros_l1949_194935


namespace NUMINAMATH_GPT_return_trip_time_l1949_194992

variable {d p w : ℝ} -- Distance, plane's speed in calm air, wind speed

theorem return_trip_time (h1 : d = 75 * (p - w)) 
                         (h2 : d / (p + w) = d / p - 10) :
                         (d / (p + w) = 15 ∨ d / (p + w) = 50) :=
sorry

end NUMINAMATH_GPT_return_trip_time_l1949_194992


namespace NUMINAMATH_GPT_part1_part2_l1949_194980

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 6

-- Part (I)
theorem part1 (a : ℝ) (h : a = 5) : ∀ x : ℝ, f x 5 < 0 ↔ -3 < x ∧ x < -2 := by
  sorry

-- Part (II)
theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > 0) ↔ -2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1949_194980


namespace NUMINAMATH_GPT_team_score_is_correct_l1949_194958

-- Definitions based on given conditions
def connor_score : ℕ := 2
def amy_score : ℕ := connor_score + 4
def jason_score : ℕ := 2 * amy_score
def combined_score : ℕ := connor_score + amy_score + jason_score
def emily_score : ℕ := 3 * combined_score
def team_score : ℕ := connor_score + amy_score + jason_score + emily_score

-- Theorem stating team_score should be 80
theorem team_score_is_correct : team_score = 80 := by
  sorry

end NUMINAMATH_GPT_team_score_is_correct_l1949_194958


namespace NUMINAMATH_GPT_find_angle_l1949_194975

theorem find_angle (x : Real) : 
  (x - (1 / 2) * (180 - x) = -18 - 24/60 - 36/3600) -> 
  x = 47 + 43/60 + 36/3600 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_l1949_194975


namespace NUMINAMATH_GPT_period_of_repeating_decimal_l1949_194954

def is_100_digit_number_with_98_sevens (a : ℕ) : Prop :=
  ∃ (n : ℕ), n = 10^98 ∧ a = 1776 + 1777 * n

theorem period_of_repeating_decimal (a : ℕ) (h : is_100_digit_number_with_98_sevens a) : 
  (1:ℚ) / a == 1 / 99 := 
  sorry

end NUMINAMATH_GPT_period_of_repeating_decimal_l1949_194954


namespace NUMINAMATH_GPT_save_after_increase_l1949_194924

def monthly_saving_initial (salary : ℕ) (saving_percentage : ℕ) : ℕ :=
  salary * saving_percentage / 100

def monthly_expense_initial (salary : ℕ) (saving : ℕ) : ℕ :=
  salary - saving

def increase_by_percentage (amount : ℕ) (percentage : ℕ) : ℕ :=
  amount * percentage / 100

def new_expense (initial_expense : ℕ) (increase : ℕ) : ℕ :=
  initial_expense + increase

def new_saving (salary : ℕ) (expense : ℕ) : ℕ :=
  salary - expense

theorem save_after_increase (salary saving_percentage increase_percentage : ℕ) 
  (H_salary : salary = 5500) 
  (H_saving_percentage : saving_percentage = 20) 
  (H_increase_percentage : increase_percentage = 20) :
  new_saving salary (new_expense (monthly_expense_initial salary (monthly_saving_initial salary saving_percentage)) (increase_by_percentage (monthly_expense_initial salary (monthly_saving_initial salary saving_percentage)) increase_percentage)) = 220 := 
by
  sorry

end NUMINAMATH_GPT_save_after_increase_l1949_194924


namespace NUMINAMATH_GPT_no_solution_exists_l1949_194921

theorem no_solution_exists (p : ℝ) : (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) = (x - p) / (x - 8) → false) ↔ p = 7 :=
by sorry

end NUMINAMATH_GPT_no_solution_exists_l1949_194921


namespace NUMINAMATH_GPT_inequality_proof_l1949_194903

variables {x y a b ε m : ℝ}

theorem inequality_proof (h1 : |x - a| < ε / (2 * m))
                        (h2 : |y - b| < ε / (2 * |a|))
                        (h3 : 0 < y ∧ y < m) :
                        |x * y - a * b| < ε :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1949_194903


namespace NUMINAMATH_GPT_grunters_win_all_6_games_l1949_194982

noncomputable def prob_no_overtime_win : ℚ := 0.54
noncomputable def prob_overtime_win : ℚ := 0.05
noncomputable def prob_win_any_game : ℚ := prob_no_overtime_win + prob_overtime_win
noncomputable def prob_win_all_6_games : ℚ := prob_win_any_game ^ 6

theorem grunters_win_all_6_games :
  prob_win_all_6_games = (823543 / 10000000) :=
by sorry

end NUMINAMATH_GPT_grunters_win_all_6_games_l1949_194982


namespace NUMINAMATH_GPT_total_number_of_fish_l1949_194910

theorem total_number_of_fish (fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1: fishbowls = 261) (h2: fish_per_bowl = 23) : 
  fishbowls * fish_per_bowl = 6003 := 
by
  sorry

end NUMINAMATH_GPT_total_number_of_fish_l1949_194910


namespace NUMINAMATH_GPT_johannes_sells_48_kg_l1949_194926

-- Define Johannes' earnings
def earnings_wednesday : ℕ := 30
def earnings_friday : ℕ := 24
def earnings_today : ℕ := 42

-- Price per kilogram of cabbage
def price_per_kg : ℕ := 2

-- Prove that the total kilograms of cabbage sold is 48
theorem johannes_sells_48_kg :
  ((earnings_wednesday + earnings_friday + earnings_today) / price_per_kg) = 48 := by
  sorry

end NUMINAMATH_GPT_johannes_sells_48_kg_l1949_194926


namespace NUMINAMATH_GPT_percentage_square_area_in_rectangle_l1949_194900

variable (s : ℝ)
variable (W : ℝ) (L : ℝ)
variable (hW : W = 3 * s) -- Width is 3 times the side of the square
variable (hL : L = (3 / 2) * W) -- Length is 3/2 times the width

theorem percentage_square_area_in_rectangle :
  (s^2 / ((27 * s^2) / 2)) * 100 = 7.41 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_square_area_in_rectangle_l1949_194900


namespace NUMINAMATH_GPT_consecutive_even_legs_sum_l1949_194916

theorem consecutive_even_legs_sum (x : ℕ) (h : x % 2 = 0) (hx : x ^ 2 + (x + 2) ^ 2 = 34 ^ 2) : x + (x + 2) = 48 := by
  sorry

end NUMINAMATH_GPT_consecutive_even_legs_sum_l1949_194916


namespace NUMINAMATH_GPT_problem_proof_l1949_194917

theorem problem_proof (x y z : ℝ) 
  (h1 : 1/x + 2/y + 3/z = 0) 
  (h2 : 1/x - 6/y - 5/z = 0) : 
  (x / y + y / z + z / x) = -1 := 
by
  sorry

end NUMINAMATH_GPT_problem_proof_l1949_194917


namespace NUMINAMATH_GPT_green_disks_more_than_blue_l1949_194973

theorem green_disks_more_than_blue (total_disks : ℕ) (b y g : ℕ) (h1 : total_disks = 108)
  (h2 : b / y = 3 / 7) (h3 : b / g = 3 / 8) : g - b = 30 :=
by
  sorry

end NUMINAMATH_GPT_green_disks_more_than_blue_l1949_194973
