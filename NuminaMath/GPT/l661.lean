import Mathlib

namespace NUMINAMATH_GPT_sum_first_3k_plus_2_terms_l661_66115

variable (k : ℕ)

def first_term : ℕ := k^2 + 1

def sum_of_sequence (n : ℕ) : ℕ :=
  let a₁ := first_term k
  let aₙ := a₁ + (n - 1)
  n * (a₁ + aₙ) / 2

theorem sum_first_3k_plus_2_terms :
  sum_of_sequence k (3 * k + 2) = 3 * k^3 + 8 * k^2 + 6 * k + 3 :=
by
  -- Here we define the sequence and compute the sum
  sorry

end NUMINAMATH_GPT_sum_first_3k_plus_2_terms_l661_66115


namespace NUMINAMATH_GPT_largest_perfect_square_factor_9240_l661_66101

theorem largest_perfect_square_factor_9240 :
  ∃ n : ℕ, n * n = 36 ∧ ∃ m : ℕ, m ∣ 9240 ∧ m = n * n :=
by
  -- We will construct the proof here using the prime factorization
  sorry

end NUMINAMATH_GPT_largest_perfect_square_factor_9240_l661_66101


namespace NUMINAMATH_GPT_sum_of_roots_l661_66105

theorem sum_of_roots (x y : ℝ) (h : ∀ z, z^2 + 2023 * z - 2024 = 0 → z = x ∨ z = y) : x + y = -2023 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l661_66105


namespace NUMINAMATH_GPT_not_all_odd_l661_66127

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1
def divides (a b c d : ℕ) : Prop := a = b * c + d ∧ 0 ≤ d ∧ d < b

theorem not_all_odd (a b c d : ℕ) 
  (h_div : divides a b c d)
  (h_odd_a : is_odd a)
  (h_odd_b : is_odd b)
  (h_odd_c : is_odd c)
  (h_odd_d : is_odd d) :
  False :=
sorry

end NUMINAMATH_GPT_not_all_odd_l661_66127


namespace NUMINAMATH_GPT_quotient_of_1575_210_l661_66130

theorem quotient_of_1575_210 (a b q : ℕ) (h1 : a = 1575) (h2 : b = a - 1365) (h3 : a % b = 15) : q = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_quotient_of_1575_210_l661_66130


namespace NUMINAMATH_GPT_quadratic_factor_transformation_l661_66100

theorem quadratic_factor_transformation (x : ℝ) :
  x^2 - 6 * x + 5 = 0 → (x - 3)^2 = 14 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_factor_transformation_l661_66100


namespace NUMINAMATH_GPT_sum_of_perimeters_l661_66150

theorem sum_of_perimeters (s : ℝ) : (∀ n : ℕ, n >= 0) → 
  (∑' n : ℕ, (4 * s) / (2 ^ n)) = 8 * s :=
by
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_l661_66150


namespace NUMINAMATH_GPT_value_at_x12_l661_66167

def quadratic_function (d e f x : ℝ) : ℝ :=
  d * x^2 + e * x + f

def axis_of_symmetry (d e f : ℝ) : ℝ := 10.5

def point_on_graph (d e f : ℝ) : Prop :=
  quadratic_function d e f 3 = -5

theorem value_at_x12 (d e f : ℝ)
  (Hsymm : axis_of_symmetry d e f = 10.5)
  (Hpoint : point_on_graph d e f) :
  quadratic_function d e f 12 = -5 :=
sorry

end NUMINAMATH_GPT_value_at_x12_l661_66167


namespace NUMINAMATH_GPT_p_p_eq_twenty_l661_66143

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else if x ≥ 0 ∧ y < 0 then 4 * x + 2 * y
  else 3 * x + 2 * y

theorem p_p_eq_twenty : p (p 2 (-3)) (p (-3) (-4)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_p_p_eq_twenty_l661_66143


namespace NUMINAMATH_GPT_polynomial_coefficient_sum_l661_66132

theorem polynomial_coefficient_sum
  (a b c d : ℤ)
  (h1 : (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + 2 * x^3 - 5 * x^2 + 8 * x - 12) :
  a + b + c + d = 6 := 
sorry

end NUMINAMATH_GPT_polynomial_coefficient_sum_l661_66132


namespace NUMINAMATH_GPT_student_distribution_l661_66193

-- Definition to check the number of ways to distribute 7 students into two dormitories A and B
-- with each dormitory having at least 2 students equals 56.
theorem student_distribution (students dorms : Nat) (min_students : Nat) (dist_plans : Nat) :
  students = 7 → dorms = 2 → min_students = 2 → dist_plans = 56 → 
  true := sorry

end NUMINAMATH_GPT_student_distribution_l661_66193


namespace NUMINAMATH_GPT_roof_area_l661_66136

theorem roof_area (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : l - w = 28) : 
  l * w = 3136 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_roof_area_l661_66136


namespace NUMINAMATH_GPT_find_position_of_2017_l661_66160

theorem find_position_of_2017 :
  ∃ (row col : ℕ), row = 45 ∧ col = 81 ∧ 2017 = (row - 1)^2 + col :=
by
  sorry

end NUMINAMATH_GPT_find_position_of_2017_l661_66160


namespace NUMINAMATH_GPT_part1_part2_l661_66177

-- Part (1)
theorem part1 (a : ℝ) (P Q : Set ℝ) (hP : P = {x | 4 <= x ∧ x <= 7})
              (hQ : Q = {x | -2 <= x ∧ x <= 5}) :
  (Set.compl P ∩ Q) = {x | -2 <= x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (P Q : Set ℝ)
              (hP : P = {x | a + 1 <= x ∧ x <= 2 * a + 1})
              (hQ : Q = {x | -2 <= x ∧ x <= 5})
              (h_sufficient : ∀ x, x ∈ P → x ∈ Q) 
              (h_not_necessary : ∃ x, x ∈ Q ∧ x ∉ P) :
  (0 <= a ∧ a <= 2) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l661_66177


namespace NUMINAMATH_GPT_minimum_other_sales_met_l661_66171

-- Define the sales percentages for pens, pencils, and the condition for other items
def pens_sales : ℝ := 40
def pencils_sales : ℝ := 28
def minimum_other_sales : ℝ := 20

-- Define the total percentage and calculate the required percentage for other items
def total_sales : ℝ := 100
def required_other_sales : ℝ := total_sales - (pens_sales + pencils_sales)

-- The Lean4 statement to prove the percentage of sales for other items
theorem minimum_other_sales_met 
  (pens_sales_eq : pens_sales = 40)
  (pencils_sales_eq : pencils_sales = 28)
  (total_sales_eq : total_sales = 100)
  (minimum_other_sales_eq : minimum_other_sales = 20)
  (required_other_sales_eq : required_other_sales = total_sales - (pens_sales + pencils_sales)) 
  : required_other_sales = 32 ∧ pens_sales + pencils_sales + required_other_sales = 100 := 
by
  sorry

end NUMINAMATH_GPT_minimum_other_sales_met_l661_66171


namespace NUMINAMATH_GPT_cyclic_quadrilateral_XF_XG_l661_66191

/-- 
Given:
- A cyclic quadrilateral ABCD inscribed in a circle O,
- Side lengths: AB = 4, BC = 3, CD = 7, DA = 9,
- Points X and Y such that DX/BD = 1/3 and BY/BD = 1/4,
- E is the intersection of line AX and the line through Y parallel to BC,
- F is the intersection of line CX and the line through E parallel to AB,
- G is the other intersection of line CX with circle O,
Prove:
- XF * XG = 36.5.
-/
theorem cyclic_quadrilateral_XF_XG (AB BC CD DA DX BD BY : ℝ) 
  (h_AB : AB = 4) (h_BC : BC = 3) (h_CD : CD = 7) (h_DA : DA = 9)
  (h_ratio1 : DX / BD = 1 / 3) (h_ratio2 : BY / BD = 1 / 4)
  (BD := Real.sqrt 73) :
  ∃ (XF XG : ℝ), XF * XG = 36.5 :=
by
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_XF_XG_l661_66191


namespace NUMINAMATH_GPT_train_speed_correct_l661_66168

noncomputable def jogger_speed_km_per_hr := 9
noncomputable def jogger_speed_m_per_s := 9 * 1000 / 3600
noncomputable def train_speed_km_per_hr := 45
noncomputable def distance_ahead_m := 270
noncomputable def train_length_m := 120
noncomputable def total_distance_m := distance_ahead_m + train_length_m
noncomputable def time_seconds := 39

theorem train_speed_correct :
  let relative_speed_m_per_s := total_distance_m / time_seconds
  let train_speed_m_per_s := relative_speed_m_per_s + jogger_speed_m_per_s
  let train_speed_km_per_hr_calculated := train_speed_m_per_s * 3600 / 1000
  train_speed_km_per_hr_calculated = train_speed_km_per_hr :=
by
  sorry

end NUMINAMATH_GPT_train_speed_correct_l661_66168


namespace NUMINAMATH_GPT_sheets_bought_l661_66197

variable (x y : ℕ)

-- Conditions based on the problem statement
def A_condition (x y : ℕ) : Prop := x + 40 = y
def B_condition (x y : ℕ) : Prop := 3 * x + 40 = y

-- Proven that if these conditions are met, then the number of sheets of stationery bought by A and B is 120
theorem sheets_bought (x y : ℕ) (hA : A_condition x y) (hB : B_condition x y) : y = 120 :=
by
  sorry

end NUMINAMATH_GPT_sheets_bought_l661_66197


namespace NUMINAMATH_GPT_find_number_of_cups_l661_66142

theorem find_number_of_cups (a C B : ℝ) (h1 : a * C + 2 * B = 12.75) (h2 : 2 * C + 5 * B = 14.00) (h3 : B = 1.5) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_cups_l661_66142


namespace NUMINAMATH_GPT_regular_polygon_sides_l661_66139

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 160) : n = 18 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l661_66139


namespace NUMINAMATH_GPT_turnip_total_correct_l661_66138

def turnips_left (melanie benny sarah david m_sold d_sold : ℕ) : ℕ :=
  let melanie_left := melanie - m_sold
  let david_left := david - d_sold
  benny + sarah + melanie_left + david_left

theorem turnip_total_correct :
  turnips_left 139 113 195 87 32 15 = 487 :=
by
  sorry

end NUMINAMATH_GPT_turnip_total_correct_l661_66138


namespace NUMINAMATH_GPT_intervals_equinumerous_l661_66195

-- Definitions and statements
theorem intervals_equinumerous (a : ℝ) (h : 0 < a) : 
  ∃ (f : Set.Icc 0 1 → Set.Icc 0 a), Function.Bijective f :=
by
  sorry

end NUMINAMATH_GPT_intervals_equinumerous_l661_66195


namespace NUMINAMATH_GPT_equal_savings_l661_66149

theorem equal_savings (A B AE BE AS BS : ℕ) 
  (hA : A = 2000)
  (hA_B : 5 * B = 4 * A)
  (hAE_BE : 3 * BE = 2 * AE)
  (hSavings : AS = A - AE ∧ BS = B - BE ∧ AS = BS) :
  AS = 800 ∧ BS = 800 :=
by
  -- Placeholders for definitions and calculations
  sorry

end NUMINAMATH_GPT_equal_savings_l661_66149


namespace NUMINAMATH_GPT_decreasing_function_range_l661_66152

theorem decreasing_function_range (f : ℝ → ℝ) (a : ℝ) (h_decreasing : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < 1 → -1 < x2 ∧ x2 < 1 ∧ x1 > x2 → f x1 < f x2)
  (h_ineq: f (1 - a) < f (3 * a - 1)) : 0 < a ∧ a < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_range_l661_66152


namespace NUMINAMATH_GPT_solve_trig_eq_l661_66185

noncomputable def rad (d : ℝ) := d * (Real.pi / 180)

theorem solve_trig_eq (z : ℝ) (k : ℤ) :
  (7 * Real.cos (z) ^ 3 - 6 * Real.cos (z) = 3 * Real.cos (3 * z)) ↔
  (z = rad 90 + k * rad 180 ∨
   z = rad 39.2333 + k * rad 180 ∨
   z = rad 140.7667 + k * rad 180) :=
sorry

end NUMINAMATH_GPT_solve_trig_eq_l661_66185


namespace NUMINAMATH_GPT_find_a_given_coefficient_l661_66155

theorem find_a_given_coefficient (a : ℝ) :
  (∀ x : ℝ, a ≠ 0 → x ≠ 0 → a^4 * x^4 + 4 * a^3 * x^2 * (1/x) + 6 * a^2 * (1/x)^2 * x^4 + 4 * a * (1/x)^3 * x^6 + (1/x)^4 * x^8 = (ax + 1/x)^4) → (4 * a^3 = 32) → a = 2 :=
by
  intros H1 H2
  sorry

end NUMINAMATH_GPT_find_a_given_coefficient_l661_66155


namespace NUMINAMATH_GPT_flower_beds_fraction_l661_66124

-- Definitions based on given conditions
def yard_length := 30
def yard_width := 6
def trapezoid_parallel_side1 := 20
def trapezoid_parallel_side2 := 30
def flower_bed_leg := (trapezoid_parallel_side2 - trapezoid_parallel_side1) / 2
def flower_bed_area := (1 / 2) * flower_bed_leg ^ 2
def total_flower_bed_area := 2 * flower_bed_area
def yard_area := yard_length * yard_width
def occupied_fraction := total_flower_bed_area / yard_area

-- Statement to prove
theorem flower_beds_fraction :
  occupied_fraction = 5 / 36 :=
by
  -- sorries to skip the proofs
  sorry

end NUMINAMATH_GPT_flower_beds_fraction_l661_66124


namespace NUMINAMATH_GPT_at_least_one_lands_l661_66158

def p : Prop := sorry -- Proposition that Person A lands in the designated area
def q : Prop := sorry -- Proposition that Person B lands in the designated area

theorem at_least_one_lands : p ∨ q := sorry

end NUMINAMATH_GPT_at_least_one_lands_l661_66158


namespace NUMINAMATH_GPT_TinaTotalPens_l661_66112

variable (p g b : ℕ)
axiom H1 : p = 12
axiom H2 : g = p - 9
axiom H3 : b = g + 3

theorem TinaTotalPens : p + g + b = 21 := by
  sorry

end NUMINAMATH_GPT_TinaTotalPens_l661_66112


namespace NUMINAMATH_GPT_window_area_properties_l661_66121

theorem window_area_properties
  (AB : ℝ) (AD : ℝ) (ratio : ℝ)
  (h1 : ratio = 3 / 1)
  (h2 : AB = 40)
  (h3 : AD = 3 * AB) :
  (AD * AB / (π * (AB / 2) ^ 2) = 12 / π) ∧
  (AD * AB + π * (AB / 2) ^ 2 = 4800 + 400 * π) :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_window_area_properties_l661_66121


namespace NUMINAMATH_GPT_time_for_runnerA_to_complete_race_l661_66196

variable (speedA : ℝ) -- speed of runner A in meters per second
variable (t : ℝ) -- time taken by runner A to complete the race in seconds
variable (tB : ℝ) -- time taken by runner B to complete the race in seconds

noncomputable def distanceA : ℝ := 1000 -- distance covered by runner A in meters
noncomputable def distanceB : ℝ := 950 -- distance covered by runner B in meters when A finishes
noncomputable def speedB : ℝ := distanceB / tB -- speed of runner B in meters per second

theorem time_for_runnerA_to_complete_race
    (h1 : distanceA = speedA * t)
    (h2 : distanceB = speedA * (t + 20)) :
    t = 400 :=
by
  sorry

end NUMINAMATH_GPT_time_for_runnerA_to_complete_race_l661_66196


namespace NUMINAMATH_GPT_composite_divisible_by_six_l661_66163

theorem composite_divisible_by_six (n : ℤ) (h : ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ n = a * b) : 6 ∣ (n^4 - n) :=
sorry

end NUMINAMATH_GPT_composite_divisible_by_six_l661_66163


namespace NUMINAMATH_GPT_value_of_f_2_pow_100_l661_66173

def f : ℕ → ℕ :=
sorry

axiom f_base : f 1 = 1
axiom f_recursive : ∀ n : ℕ, f (2 * n) = n * f n

theorem value_of_f_2_pow_100 : f (2^100) = 2^4950 :=
sorry

end NUMINAMATH_GPT_value_of_f_2_pow_100_l661_66173


namespace NUMINAMATH_GPT_contrapositive_equiv_l661_66192

variable {α : Type}  -- Type of elements
variable (P : Set α) (a b : α)

theorem contrapositive_equiv (h : a ∈ P → b ∉ P) : b ∈ P → a ∉ P :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_equiv_l661_66192


namespace NUMINAMATH_GPT_EventB_is_random_l661_66170

-- Define the events A, B, C, and D as propositions
def EventA : Prop := ∀ (x : ℕ), true -- A coin thrown will fall due to gravity (certain event)
def EventB : Prop := ∃ (n : ℕ), n > 0 -- Hitting the target with a score of 10 points (random event)
def EventC : Prop := ∀ (x : ℕ), true -- The sun rises from the east (certain event)
def EventD : Prop := ∀ (x : ℕ), false -- Horse runs at 70 meters per second (impossible event)

-- Prove that EventB is random, we can use a custom predicate for random events
def is_random_event (e : Prop) : Prop := (∃ (n : ℕ), n > 1) ∧ ¬ ∀ (x : ℕ), e

-- Main statement
theorem EventB_is_random :
  is_random_event EventB :=
by sorry -- The proof will be written here

end NUMINAMATH_GPT_EventB_is_random_l661_66170


namespace NUMINAMATH_GPT_barbara_spent_on_other_goods_l661_66120

theorem barbara_spent_on_other_goods
  (cost_tuna : ℝ := 5 * 2)
  (cost_water : ℝ := 4 * 1.5)
  (total_paid : ℝ := 56) :
  total_paid - (cost_tuna + cost_water) = 40 := by
  sorry

end NUMINAMATH_GPT_barbara_spent_on_other_goods_l661_66120


namespace NUMINAMATH_GPT_alice_sold_20_pears_l661_66134

-- Definitions (Conditions)
def canned_more_than_poached (C P : ℝ) : Prop := C = P + 0.2 * P
def poached_less_than_sold (P S : ℝ) : Prop := P = 0.5 * S
def total_pears (S C P : ℝ) : Prop := S + C + P = 42

-- Theorem statement
theorem alice_sold_20_pears (S C P : ℝ) (h1 : canned_more_than_poached C P) (h2 : poached_less_than_sold P S) (h3 : total_pears S C P) : S = 20 :=
by 
  -- This is where the proof would go, but for now, we use sorry to signify it's omitted.
  sorry

end NUMINAMATH_GPT_alice_sold_20_pears_l661_66134


namespace NUMINAMATH_GPT_no_real_solution_f_of_f_f_eq_x_l661_66181

-- Defining the quadratic polynomial f(x) = ax^2 + bx + c
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Stating the main theorem
theorem no_real_solution_f_of_f_f_eq_x (a b c : ℝ) (h : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x :=
by 
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_no_real_solution_f_of_f_f_eq_x_l661_66181


namespace NUMINAMATH_GPT_original_cost_l661_66183

theorem original_cost (C : ℝ) (h : 670 = C + 0.35 * C) : C = 496.30 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_original_cost_l661_66183


namespace NUMINAMATH_GPT_part_a_solution_l661_66125

theorem part_a_solution (x y : ℤ) : xy + 3 * x - 5 * y = -3 ↔ 
  (x = 6 ∧ y = -21) ∨ 
  (x = -13 ∧ y = -2) ∨ 
  (x = 4 ∧ y = 15) ∨ 
  (x = 23 ∧ y = -4) ∨ 
  (x = 7 ∧ y = -12) ∨ 
  (x = -4 ∧ y = -1) ∨ 
  (x = 3 ∧ y = 6) ∨ 
  (x = 14 ∧ y = -5) ∨ 
  (x = 8 ∧ y = -9) ∨ 
  (x = -1 ∧ y = 0) ∨ 
  (x = 2 ∧ y = 3) ∨ 
  (x = 11 ∧ y = -6) := 
by sorry

end NUMINAMATH_GPT_part_a_solution_l661_66125


namespace NUMINAMATH_GPT_purely_imaginary_has_specific_a_l661_66182

theorem purely_imaginary_has_specific_a (a : ℝ) :
  (a^2 - 1 + (a - 1 : ℂ) * Complex.I) = (a - 1 : ℂ) * Complex.I → a = -1 := 
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_has_specific_a_l661_66182


namespace NUMINAMATH_GPT_angela_age_in_fifteen_years_l661_66111

-- Condition 1: Angela is currently 3 times as old as Beth
def angela_age_three_times_beth (A B : ℕ) := A = 3 * B

-- Condition 2: Angela is half as old as Derek
def angela_half_derek (A D : ℕ) := A = D / 2

-- Condition 3: Twenty years ago, the sum of their ages was equal to Derek's current age
def sum_ages_twenty_years_ago (A B D : ℕ) := (A - 20) + (B - 20) + (D - 20) = D

-- Condition 4: In seven years, the difference in the square root of Angela's age and one-third of Beth's age is a quarter of Derek's age
def age_diff_seven_years (A B D : ℕ) := Real.sqrt (A + 7) - (B + 7) / 3 = D / 4

-- Define the main theorem to be proven
theorem angela_age_in_fifteen_years (A B D : ℕ) 
  (h1 : angela_age_three_times_beth A B)
  (h2 : angela_half_derek A D) 
  (h3 : sum_ages_twenty_years_ago A B D) 
  (h4 : age_diff_seven_years A B D) :
  A + 15 = 60 := 
  sorry

end NUMINAMATH_GPT_angela_age_in_fifteen_years_l661_66111


namespace NUMINAMATH_GPT_calculate_expression_l661_66157

theorem calculate_expression :
  3 ^ 3 * 2 ^ 2 * 7 ^ 2 * 11 = 58212 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l661_66157


namespace NUMINAMATH_GPT_find_k_range_of_m_l661_66131

-- Given conditions and function definition
def f (x k : ℝ) : ℝ := x^2 + (2*k-3)*x + k^2 - 7

-- Prove that k = 3 when the zeros of f(x) are -1 and -2
theorem find_k (k : ℝ) (h₁ : f (-1) k = 0) (h₂ : f (-2) k = 0) : k = 3 := 
by sorry

-- Prove the range of m such that f(x) < m for x in [-2, 2]
theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + 3*x + 2 < m) ↔ 12 < m :=
by sorry

end NUMINAMATH_GPT_find_k_range_of_m_l661_66131


namespace NUMINAMATH_GPT_total_comics_in_box_l661_66148

theorem total_comics_in_box 
  (pages_per_comic : ℕ)
  (total_pages_found : ℕ)
  (untorn_comics : ℕ)
  (comics_fixed : ℕ := total_pages_found / pages_per_comic)
  (total_comics : ℕ := comics_fixed + untorn_comics)
  (h_pages_per_comic : pages_per_comic = 25)
  (h_total_pages_found : total_pages_found = 150)
  (h_untorn_comics : untorn_comics = 5) :
  total_comics = 11 :=
by
  sorry

end NUMINAMATH_GPT_total_comics_in_box_l661_66148


namespace NUMINAMATH_GPT_inscribed_rectangle_area_correct_l661_66188

noncomputable def area_of_inscribed_rectangle : Prop := 
  let AD : ℝ := 15 / (12 / (1 / 3) + 3)
  let AB : ℝ := 1 / 3 * AD
  AD * AB = 25 / 12

theorem inscribed_rectangle_area_correct :
  area_of_inscribed_rectangle
  := by
  let hf : ℝ := 12
  let eg : ℝ := 15
  let ad : ℝ := 15 / (hf / (1 / 3) + 3)
  let ab : ℝ := 1 / 3 * ad
  have area : ad * ab = 25 / 12 := by sorry
  exact area

end NUMINAMATH_GPT_inscribed_rectangle_area_correct_l661_66188


namespace NUMINAMATH_GPT_construction_cost_is_correct_l661_66146

def land_cost (cost_per_sqm : ℕ) (area : ℕ) : ℕ :=
  cost_per_sqm * area

def bricks_cost (cost_per_1000 : ℕ) (quantity : ℕ) : ℕ :=
  (cost_per_1000 * quantity) / 1000

def roof_tiles_cost (cost_per_tile : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_tile * quantity

def cement_bags_cost (cost_per_bag : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_bag * quantity

def wooden_beams_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def steel_bars_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def electrical_wiring_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def plumbing_pipes_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def total_cost : ℕ :=
  land_cost 60 2500 +
  bricks_cost 120 15000 +
  roof_tiles_cost 12 800 +
  cement_bags_cost 8 250 +
  wooden_beams_cost 25 1000 +
  steel_bars_cost 15 500 +
  electrical_wiring_cost 2 2000 +
  plumbing_pipes_cost 4 3000

theorem construction_cost_is_correct : total_cost = 212900 :=
  by
    sorry

end NUMINAMATH_GPT_construction_cost_is_correct_l661_66146


namespace NUMINAMATH_GPT_last_two_digits_of_sum_of_first_15_factorials_eq_13_l661_66114

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits_sum : ℕ :=
  let partial_sum := (factorial 1 % 100) + (factorial 2 % 100) + (factorial 3 % 100) +
                     (factorial 4 % 100) + (factorial 5 % 100) + (factorial 6 % 100) +
                     (factorial 7 % 100) + (factorial 8 % 100) + (factorial 9 % 100)
  partial_sum % 100

theorem last_two_digits_of_sum_of_first_15_factorials_eq_13 : last_two_digits_sum = 13 := by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_sum_of_first_15_factorials_eq_13_l661_66114


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_eq_213_l661_66156

theorem sum_of_squares_of_roots_eq_213
  {a b : ℝ}
  (h1 : a + b = 15)
  (h2 : a * b = 6) :
  a^2 + b^2 = 213 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_eq_213_l661_66156


namespace NUMINAMATH_GPT_algebra_expression_solution_l661_66162

theorem algebra_expression_solution
  (m : ℝ)
  (h : m^2 + m - 1 = 0) :
  m^3 + 2 * m^2 - 2001 = -2000 := by
  sorry

end NUMINAMATH_GPT_algebra_expression_solution_l661_66162


namespace NUMINAMATH_GPT_arithmetic_geometric_sequences_l661_66123

variable {S T : ℕ → ℝ}
variable {a b : ℕ → ℝ}

theorem arithmetic_geometric_sequences (h1 : a 3 = b 3)
  (h2 : a 4 = b 4)
  (h3 : (S 5 - S 3) / (T 4 - T 2) = 5) :
  (a 5 + a 3) / (b 5 + b 3) = - (3 / 5) := by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequences_l661_66123


namespace NUMINAMATH_GPT_bottle_caps_per_group_l661_66107

theorem bottle_caps_per_group (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) 
  (h1 : total_caps = 12) (h2 : num_groups = 6) : 
  total_caps / num_groups = caps_per_group := by
  sorry

end NUMINAMATH_GPT_bottle_caps_per_group_l661_66107


namespace NUMINAMATH_GPT_min_occupied_seats_l661_66169

theorem min_occupied_seats (n : ℕ) (h_n : n = 150) : 
  ∃ k : ℕ, k = 37 ∧ ∀ (occupied : Finset ℕ), 
    occupied.card < k → ∃ i : ℕ, i ∉ occupied ∧ ∀ j : ℕ, j ∈ occupied → j + 1 ≠ i ∧ j - 1 ≠ i :=
by
  sorry

end NUMINAMATH_GPT_min_occupied_seats_l661_66169


namespace NUMINAMATH_GPT_find_n_l661_66184

theorem find_n (n : ℤ) (h1 : 1 ≤ n) (h2 : n ≤ 9) (h3 : n % 10 = -245 % 10) : n = 5 := 
  sorry

end NUMINAMATH_GPT_find_n_l661_66184


namespace NUMINAMATH_GPT_ivan_speed_ratio_l661_66102

/-- 
A group of tourists started a hike from a campsite. Fifteen minutes later, Ivan returned to the campsite for a flashlight 
and started catching up with the group at a faster constant speed. He reached them 2.5 hours after initially leaving. 
Prove Ivan's speed is 1.2 times the group's speed.
-/
theorem ivan_speed_ratio (d_g d_i : ℝ) (t_g t_i : ℝ) (v_g v_i : ℝ)
    (h1 : t_g = 2.25)       -- Group's travel time (2.25 hours after initial 15 minutes)
    (h2 : t_i = 2.5)        -- Ivan's total travel time
    (h3 : d_g = t_g * v_g)  -- Distance covered by group
    (h4 : d_i = 3 * (v_g * (15 / 60))) -- Ivan's distance covered
    (h5 : d_g = d_i)        -- Ivan eventually catches up with the group
  : v_i / v_g = 1.2 := sorry

end NUMINAMATH_GPT_ivan_speed_ratio_l661_66102


namespace NUMINAMATH_GPT_XiaoMing_team_award_l661_66189

def points (x : ℕ) : ℕ := 2 * x + (8 - x)

theorem XiaoMing_team_award (x : ℕ) : 2 * x + (8 - x) ≥ 12 := 
by 
  sorry

end NUMINAMATH_GPT_XiaoMing_team_award_l661_66189


namespace NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l661_66151

theorem sum_of_other_endpoint_coordinates {x y : ℝ} :
  let P1 := (1, 2)
  let M := (5, 6)
  let P2 := (x, y)
  (M.1 = (P1.1 + P2.1) / 2 ∧ M.2 = (P1.2 + P2.2) / 2) → (x + y) = 19 :=
by
  intros P1 M P2 h
  sorry

end NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l661_66151


namespace NUMINAMATH_GPT_xy_squared_value_l661_66117

theorem xy_squared_value (x y : ℝ) (h1 : x * (x + y) = 22) (h2 : y * (x + y) = 78 - y) :
  (x + y) ^ 2 = 100 :=
  sorry

end NUMINAMATH_GPT_xy_squared_value_l661_66117


namespace NUMINAMATH_GPT_standard_deviations_below_l661_66199

variable (σ : ℝ)
variable (mean : ℝ)
variable (score98 : ℝ)
variable (score58 : ℝ)

-- Conditions translated to Lean definitions
def condition_1 : Prop := score98 = mean + 3 * σ
def condition_2 : Prop := mean = 74
def condition_3 : Prop := σ = 8

-- Target statement: Prove that the score of 58 is 2 standard deviations below the mean
theorem standard_deviations_below : condition_1 σ mean score98 → condition_2 mean → condition_3 σ → score58 = 74 - 2 * σ :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_standard_deviations_below_l661_66199


namespace NUMINAMATH_GPT_center_of_circle_is_1_2_l661_66122

theorem center_of_circle_is_1_2 :
  ∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y = 0 ↔ ∃ (r : ℝ), (x - 1)^2 + (y - 2)^2 = r^2 := by
  sorry

end NUMINAMATH_GPT_center_of_circle_is_1_2_l661_66122


namespace NUMINAMATH_GPT_members_playing_badminton_l661_66137

theorem members_playing_badminton
  (total_members : ℕ := 42)
  (tennis_players : ℕ := 23)
  (neither_players : ℕ := 6)
  (both_players : ℕ := 7) :
  ∃ (badminton_players : ℕ), badminton_players = 20 :=
by
  have union_players := total_members - neither_players
  have badminton_players := union_players - (tennis_players - both_players)
  use badminton_players
  sorry

end NUMINAMATH_GPT_members_playing_badminton_l661_66137


namespace NUMINAMATH_GPT_checkerboard_problem_l661_66128

def is_valid_square (size : ℕ) : Prop :=
  size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10

def check_10_by_10 : ℕ :=
  24 + 36 + 25 + 16 + 9 + 4 + 1

theorem checkerboard_problem :
  ∀ size : ℕ, ( size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10 ) →
  check_10_by_10 = 115 := 
sorry

end NUMINAMATH_GPT_checkerboard_problem_l661_66128


namespace NUMINAMATH_GPT_blackboard_final_number_lower_bound_l661_66147

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def L (c : ℝ) : ℝ := 1 + Real.log c / Real.log phi

theorem blackboard_final_number_lower_bound (c : ℝ) (n : ℕ) (h_pos_c : c > 1) (h_pos_n : n > 0) :
  ∃ x, x ≥ ((c^(n / (L c)) - 1) / (c^(1 / (L c)) - 1))^(L c) :=
sorry

end NUMINAMATH_GPT_blackboard_final_number_lower_bound_l661_66147


namespace NUMINAMATH_GPT_floor_e_is_two_l661_66190

noncomputable def e : ℝ := Real.exp 1

theorem floor_e_is_two : ⌊e⌋ = 2 := by
  sorry

end NUMINAMATH_GPT_floor_e_is_two_l661_66190


namespace NUMINAMATH_GPT_max_a_plus_b_l661_66154

/-- Given real numbers a and b such that 5a + 3b <= 11 and 3a + 6b <= 12,
    the largest possible value of a + b is 23/9. -/
theorem max_a_plus_b (a b : ℝ) (h1 : 5 * a + 3 * b ≤ 11) (h2 : 3 * a + 6 * b ≤ 12) :
  a + b ≤ 23 / 9 :=
sorry

end NUMINAMATH_GPT_max_a_plus_b_l661_66154


namespace NUMINAMATH_GPT_ice_cream_ratio_l661_66145

theorem ice_cream_ratio :
  ∃ (B C : ℕ), 
    C = 1 ∧
    (∃ (W D : ℕ), 
      D = 2 ∧
      W = B + 1 ∧
      B + W + C + D = 10 ∧
      B / C = 3
    ) := sorry

end NUMINAMATH_GPT_ice_cream_ratio_l661_66145


namespace NUMINAMATH_GPT_no_solution_x_l661_66110

theorem no_solution_x : ¬ ∃ x : ℝ, x * (x - 1) * (x - 2) + (100 - x) * (99 - x) * (98 - x) = 0 := 
sorry

end NUMINAMATH_GPT_no_solution_x_l661_66110


namespace NUMINAMATH_GPT_range_3a_2b_l661_66108

theorem range_3a_2b (a b : ℝ) (h : a^2 + b^2 = 4) : 
  -2 * Real.sqrt 13 ≤ 3 * a + 2 * b ∧ 3 * a + 2 * b ≤ 2 * Real.sqrt 13 := 
by 
  sorry

end NUMINAMATH_GPT_range_3a_2b_l661_66108


namespace NUMINAMATH_GPT_right_isosceles_triangle_acute_angle_45_l661_66116

theorem right_isosceles_triangle_acute_angle_45
    (a : ℝ)
    (h_leg_conditions : ∀ b : ℝ, a = b)
    (h_hypotenuse_condition : ∀ c : ℝ, c^2 = 2 * (a * a)) :
    ∃ θ : ℝ, θ = 45 :=
by
    sorry

end NUMINAMATH_GPT_right_isosceles_triangle_acute_angle_45_l661_66116


namespace NUMINAMATH_GPT_rate_of_stream_equation_l661_66194

theorem rate_of_stream_equation 
  (v : ℝ) 
  (boat_speed : ℝ) 
  (travel_time : ℝ) 
  (distance : ℝ)
  (h_boat_speed : boat_speed = 16)
  (h_travel_time : travel_time = 5)
  (h_distance : distance = 105)
  (h_equation : distance = (boat_speed + v) * travel_time) : v = 5 :=
by 
  sorry

end NUMINAMATH_GPT_rate_of_stream_equation_l661_66194


namespace NUMINAMATH_GPT_sample_size_l661_66144

theorem sample_size (f r n : ℕ) (freq_def : f = 36) (rate_def : r = 25 / 100) (relation : r = f / n) : n = 144 :=
sorry

end NUMINAMATH_GPT_sample_size_l661_66144


namespace NUMINAMATH_GPT_find_x_parallel_vectors_l661_66174

theorem find_x_parallel_vectors
   (x : ℝ)
   (ha : (x, 2) = (x, 2))
   (hb : (-2, 4) = (-2, 4))
   (hparallel : ∀ (k : ℝ), (x, 2) = (k * -2, k * 4)) :
   x = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_parallel_vectors_l661_66174


namespace NUMINAMATH_GPT_find_x_l661_66118

theorem find_x (a b x: ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0)
    (h4 : (4 * a)^(4 * b) = a^b * x^(2 * b)) : x = 16 * a^(3 / 2) := by
  sorry

end NUMINAMATH_GPT_find_x_l661_66118


namespace NUMINAMATH_GPT_total_number_of_tiles_l661_66141

theorem total_number_of_tiles {s : ℕ} 
  (h1 : ∃ s : ℕ, (s^2 - 4*s + 896 = 0))
  (h2 : 225 = 2*s - 1 + s^2 / 4 - s / 2) :
  s^2 = 1024 := by
  sorry

end NUMINAMATH_GPT_total_number_of_tiles_l661_66141


namespace NUMINAMATH_GPT_find_y_l661_66179

theorem find_y (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -5) : y = 18 := by
  -- Proof can go here
  sorry

end NUMINAMATH_GPT_find_y_l661_66179


namespace NUMINAMATH_GPT_sum_of_legs_is_43_l661_66180

theorem sum_of_legs_is_43 (x : ℕ) (h1 : x * x + (x + 1) * (x + 1) = 31 * 31) :
  x + (x + 1) = 43 :=
sorry

end NUMINAMATH_GPT_sum_of_legs_is_43_l661_66180


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l661_66159

-- Problem 1: System of Equations
theorem problem1_solution (x y : ℝ) (h_eq1 : x - y = 2) (h_eq2 : 2 * x + y = 7) : x = 3 ∧ y = 1 :=
by {
  sorry -- Proof to be filled in
}

-- Problem 2: Fractional Equation
theorem problem2_solution (y : ℝ) (h_eq : 3 / (1 - y) = y / (y - 1) - 5) : y = 2 :=
by {
  sorry -- Proof to be filled in
}

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l661_66159


namespace NUMINAMATH_GPT_radius_of_circle_l661_66133

theorem radius_of_circle
  (AC BD : ℝ) (h_perpendicular : AC * BD = 0)
  (h_intersect_center : AC / 2 = BD / 2)
  (AB : ℝ) (h_AB : AB = 3)
  (CD : ℝ) (h_CD : CD = 4) :
  (∃ R : ℝ, R = 5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l661_66133


namespace NUMINAMATH_GPT_player_A_success_l661_66166

/-- Representation of the problem conditions --/
structure GameState where
  coins : ℕ
  boxes : ℕ
  n_coins : ℕ 
  n_boxes : ℕ 
  arrangement: ℕ → ℕ 
  (h_coins : coins ≥ 2012)
  (h_boxes : boxes = 2012)
  (h_initial_distribution : (∀ b, arrangement b ≥ 1))
  
/-- The main theorem for player A to ensure at least 1 coin in each box --/
theorem player_A_success (s : GameState) : 
  s.coins ≥ 4022 → (∀ b, s.arrangement b ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_player_A_success_l661_66166


namespace NUMINAMATH_GPT_delores_initial_money_l661_66129

-- Definitions and conditions based on the given problem
def original_computer_price : ℝ := 400
def original_printer_price : ℝ := 40
def original_headphones_price : ℝ := 60

def computer_discount : ℝ := 0.10
def computer_tax : ℝ := 0.08
def printer_tax : ℝ := 0.05
def headphones_tax : ℝ := 0.06

def leftover_money : ℝ := 10

-- Final proof problem statement
theorem delores_initial_money :
  original_computer_price * (1 - computer_discount) * (1 + computer_tax) +
  original_printer_price * (1 + printer_tax) +
  original_headphones_price * (1 + headphones_tax) + leftover_money = 504.40 := by
  sorry -- Proof is not required

end NUMINAMATH_GPT_delores_initial_money_l661_66129


namespace NUMINAMATH_GPT_find_C_l661_66104

theorem find_C (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 350) : C = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_C_l661_66104


namespace NUMINAMATH_GPT_sin_minus_cos_eq_minus_1_l661_66172

theorem sin_minus_cos_eq_minus_1 (x : ℝ) 
  (h : Real.sin x ^ 3 - Real.cos x ^ 3 = -1) :
  Real.sin x - Real.cos x = -1 := by
  sorry

end NUMINAMATH_GPT_sin_minus_cos_eq_minus_1_l661_66172


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l661_66103

noncomputable def f (x a : ℝ) : ℝ := abs x * (x - a)

-- 1. Prove a = 0 if f(x) is odd
theorem problem1 (h: ∀ x : ℝ, f (-x) a = -f x a) : a = 0 :=
sorry

-- 2. Prove a ≤ 0 if f(x) is increasing on the interval [0, 2]
theorem problem2 (h: ∀ x y : ℝ, 0 ≤ x → x ≤ y → y ≤ 2 → f x a ≤ f y a) : a ≤ 0 :=
sorry

-- 3. Prove there exists an a < 0 such that the maximum value of f(x) on [-1, 1/2] is 2, and find a = -3
theorem problem3 (h: ∃ a : ℝ, a < 0 ∧ ∀ x : ℝ, -1 ≤ x → x ≤ 1/2 → f x a ≤ 2 ∧ ∃ x : ℝ, -1 ≤ x → x ≤ 1/2 → f x a = 2) : a = -3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l661_66103


namespace NUMINAMATH_GPT_women_in_room_l661_66153

theorem women_in_room (x q : ℕ) (h1 : 4 * x + 2 = 14) (h2 : q = 2 * (5 * x - 3)) : q = 24 :=
by sorry

end NUMINAMATH_GPT_women_in_room_l661_66153


namespace NUMINAMATH_GPT_sale_price_after_discounts_l661_66187

/-- The sale price of the television as a percentage of its original price after successive discounts of 25% followed by 10%. -/
theorem sale_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 350 → discount1 = 0.25 → discount2 = 0.10 →
  (original_price * (1 - discount1) * (1 - discount2) / original_price) * 100 = 67.5 :=
by
  intro h_price h_discount1 h_discount2
  sorry

end NUMINAMATH_GPT_sale_price_after_discounts_l661_66187


namespace NUMINAMATH_GPT_incorrect_option_D_l661_66109

variable (AB BC BO DO AO CO : ℝ)
variable (DAB : ℝ)
variable (ABCD_is_rectangle ABCD_is_rhombus ABCD_is_square: Prop)

def conditions_statement :=
  AB = BC ∧
  DAB = 90 ∧
  BO = DO ∧
  AO = CO ∧
  (ABCD_is_rectangle ↔ (AB = BC ∧ AB ≠ BC)) ∧
  (ABCD_is_rhombus ↔ AB = BC ∧ AB ≠ BC) ∧
  (ABCD_is_square ↔ ABCD_is_rectangle ∧ ABCD_is_rhombus)

theorem incorrect_option_D
  (h1: BO = DO)
  (h2: AO = CO)
  (h3: ABCD_is_rectangle)
  (h4: conditions_statement AB BC BO DO AO CO DAB ABCD_is_rectangle ABCD_is_rhombus ABCD_is_square):
  ¬ ABCD_is_square :=
by
  sorry
  -- Proof omitted

end NUMINAMATH_GPT_incorrect_option_D_l661_66109


namespace NUMINAMATH_GPT_factorize_l661_66175

theorem factorize (m : ℝ) : m^3 - 4 * m = m * (m + 2) * (m - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_l661_66175


namespace NUMINAMATH_GPT_main_inequality_l661_66113

noncomputable def b (c : ℝ) : ℝ := (1 + c) / (2 + c)

def f (c : ℝ) (x : ℝ) : ℝ := sorry

lemma f_continuous (c : ℝ) (h_c : 0 < c) : Continuous (f c) := sorry

lemma condition1 (c : ℝ) (h_c : 0 < c) (x : ℝ) (h_x : 0 ≤ x ∧ x ≤ 1/2) : 
  b c * f c (2 * x) = f c x := sorry

lemma condition2 (c : ℝ) (h_c : 0 < c) (x : ℝ) (h_x : 1/2 ≤ x ∧ x ≤ 1) : 
  f c x = b c + (1 - b c) * f c (2 * x - 1) := sorry

theorem main_inequality (c : ℝ) (h_c : 0 < c) : 
  ∀ x : ℝ, (0 < x ∧ x < 1) → (0 < f c x - x ∧ f c x - x < c) := sorry

end NUMINAMATH_GPT_main_inequality_l661_66113


namespace NUMINAMATH_GPT_find_interest_rate_l661_66164

noncomputable def compoundInterestRate (P A : ℝ) (t : ℕ) : ℝ := 
  ((A / P) ^ (1 / t)) - 1

theorem find_interest_rate :
  ∀ (P A : ℝ) (t : ℕ),
    P = 1200 → 
    A = 1200 + 873.60 →
    t = 3 →
    compoundInterestRate P A t = 0.2 :=
by
  intros P A t hP hA ht
  sorry

end NUMINAMATH_GPT_find_interest_rate_l661_66164


namespace NUMINAMATH_GPT_impossibility_of_quadratic_conditions_l661_66106

open Real

theorem impossibility_of_quadratic_conditions :
  ∀ (a b c t : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ t ∧ b ≠ t ∧ c ≠ t →
  (b * t) ^ 2 - 4 * a * c > 0 →
  c ^ 2 - 4 * b * a > 0 →
  (a * t) ^ 2 - 4 * b * c > 0 →
  false :=
by sorry

end NUMINAMATH_GPT_impossibility_of_quadratic_conditions_l661_66106


namespace NUMINAMATH_GPT_probability_of_exactly_one_solves_l661_66126

variable (p1 p2 : ℝ)

theorem probability_of_exactly_one_solves (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_exactly_one_solves_l661_66126


namespace NUMINAMATH_GPT_point_on_circle_l661_66176

theorem point_on_circle (a b : ℝ) 
  (h1 : (b + 2) * x + a * y + 4 = 0) 
  (h2 : a * x + (2 - b) * y - 3 = 0) 
  (parallel_lines : ∀ x y : ℝ, ∀ C1 C2 : ℝ, 
    (b + 2) * x + a * y + C1 = 0 ∧ a * x + (2 - b) * y + C2 = 0 → 
    - (b + 2) / a = - a / (2 - b)
  ) : a^2 + b^2 = 4 :=
sorry

end NUMINAMATH_GPT_point_on_circle_l661_66176


namespace NUMINAMATH_GPT_find_initial_cards_l661_66161

theorem find_initial_cards (B : ℕ) :
  let Tim_initial := 20
  let Sarah_initial := 15
  let Tim_after_give_to_Sarah := Tim_initial - 5
  let Sarah_after_give_to_Sarah := Sarah_initial + 5
  let Tim_after_receive_from_Sarah := Tim_after_give_to_Sarah + 2
  let Sarah_after_receive_from_Sarah := Sarah_after_give_to_Sarah - 2
  let Tim_after_exchange_with_Ben := Tim_after_receive_from_Sarah - 3
  let Ben_after_exchange := B + 13
  let Ben_after_all_transactions := 3 * Tim_after_exchange_with_Ben
  Ben_after_exchange = Ben_after_all_transactions -> B = 29 := by
  sorry

end NUMINAMATH_GPT_find_initial_cards_l661_66161


namespace NUMINAMATH_GPT_geometric_monotonic_condition_l661_66186

-- Definition of a geometrically increasing sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Definition of a monotonically increasing sequence
def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

-- The theorem statement
theorem geometric_monotonic_condition (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (a 1 < a 2 ∧ a 2 < a 3) ↔ monotonically_increasing a :=
sorry

end NUMINAMATH_GPT_geometric_monotonic_condition_l661_66186


namespace NUMINAMATH_GPT_original_average_weight_l661_66178

theorem original_average_weight (W : ℝ) (h : (7 * W + 110 + 60) / 9 = 113) : W = 121 :=
by
  sorry

end NUMINAMATH_GPT_original_average_weight_l661_66178


namespace NUMINAMATH_GPT_largest_integer_dividing_consecutive_product_l661_66135

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_dividing_consecutive_product_l661_66135


namespace NUMINAMATH_GPT_no_tetrahedron_with_given_heights_l661_66140

theorem no_tetrahedron_with_given_heights (h1 h2 h3 h4 : ℝ) (V : ℝ) (V_pos : V > 0)
    (S1 : ℝ := 3*V) (S2 : ℝ := (3/2)*V) (S3 : ℝ := V) (S4 : ℝ := V/2) :
    (h1 = 1) → (h2 = 2) → (h3 = 3) → (h4 = 6) → ¬ ∃ (S1 S2 S3 S4 : ℝ), S1 < S2 + S3 + S4 := by
  intros
  sorry

end NUMINAMATH_GPT_no_tetrahedron_with_given_heights_l661_66140


namespace NUMINAMATH_GPT_find_number_l661_66198

theorem find_number (x : ℝ) (h : x / 5 + 10 = 21) : x = 55 :=
sorry

end NUMINAMATH_GPT_find_number_l661_66198


namespace NUMINAMATH_GPT_W_555_2_last_three_digits_l661_66165

noncomputable def W : ℕ → ℕ → ℕ
| n, 0     => n ^ n
| n, (k+1) => W (W n k) k

theorem W_555_2_last_three_digits :
  (W 555 2) % 1000 = 875 :=
sorry

end NUMINAMATH_GPT_W_555_2_last_three_digits_l661_66165


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l661_66119

theorem molecular_weight_of_compound (total_weight_of_3_moles : ℝ) (n_moles : ℝ) 
  (h1 : total_weight_of_3_moles = 528) (h2 : n_moles = 3) : 
  (total_weight_of_3_moles / n_moles) = 176 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l661_66119
