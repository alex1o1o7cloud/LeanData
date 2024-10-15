import Mathlib

namespace NUMINAMATH_GPT_john_days_to_lose_weight_l1232_123216

noncomputable def john_calories_intake : ℕ := 1800
noncomputable def john_calories_burned : ℕ := 2300
noncomputable def calories_to_lose_1_pound : ℕ := 4000
noncomputable def pounds_to_lose : ℕ := 10

theorem john_days_to_lose_weight :
  (john_calories_burned - john_calories_intake) * (pounds_to_lose * calories_to_lose_1_pound / (john_calories_burned - john_calories_intake)) = 80 :=
by
  sorry

end NUMINAMATH_GPT_john_days_to_lose_weight_l1232_123216


namespace NUMINAMATH_GPT_rhombus_longer_diagonal_l1232_123228

theorem rhombus_longer_diagonal (d1 d2 : ℝ) (h_d1 : d1 = 11) (h_area : (d1 * d2) / 2 = 110) : d2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_longer_diagonal_l1232_123228


namespace NUMINAMATH_GPT_yuebao_scientific_notation_l1232_123239

-- Definition of converting a number to scientific notation
def scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10 ^ n

-- The specific problem statement
theorem yuebao_scientific_notation :
  scientific_notation (1853 * 10 ^ 9) 1.853 11 :=
by
  sorry

end NUMINAMATH_GPT_yuebao_scientific_notation_l1232_123239


namespace NUMINAMATH_GPT_smallest_n_div_75_has_75_divisors_l1232_123276

theorem smallest_n_div_75_has_75_divisors :
  ∃ n : ℕ, (n % 75 = 0) ∧ (n.factors.length = 75) ∧ (n / 75 = 432) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_div_75_has_75_divisors_l1232_123276


namespace NUMINAMATH_GPT_sum_fib_2019_eq_fib_2021_minus_1_l1232_123206

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

def sum_fib : ℕ → ℕ
| 0 => 0
| n + 1 => sum_fib n + fib (n + 1)

theorem sum_fib_2019_eq_fib_2021_minus_1 : sum_fib 2019 = fib 2021 - 1 := 
by sorry -- proof here

end NUMINAMATH_GPT_sum_fib_2019_eq_fib_2021_minus_1_l1232_123206


namespace NUMINAMATH_GPT_intersection_eq_zero_l1232_123278

def M := { x : ℤ | abs (x - 3) < 4 }
def N := { x : ℤ | x^2 + x - 2 < 0 }

theorem intersection_eq_zero : M ∩ N = {0} := 
  by
    sorry

end NUMINAMATH_GPT_intersection_eq_zero_l1232_123278


namespace NUMINAMATH_GPT_multiply_seven_l1232_123299

variable (x : ℕ)

theorem multiply_seven (h : 8 * x = 64) : 7 * x = 56 := by
  sorry


end NUMINAMATH_GPT_multiply_seven_l1232_123299


namespace NUMINAMATH_GPT_smallest_integer_n_l1232_123272

theorem smallest_integer_n (n : ℕ) (h : ∃ k : ℕ, 432 * n = k ^ 2) : n = 3 := 
sorry

end NUMINAMATH_GPT_smallest_integer_n_l1232_123272


namespace NUMINAMATH_GPT_subway_ways_l1232_123256

theorem subway_ways (total_ways : ℕ) (bus_ways : ℕ) (h1 : total_ways = 7) (h2 : bus_ways = 4) :
  total_ways - bus_ways = 3 :=
by
  sorry

end NUMINAMATH_GPT_subway_ways_l1232_123256


namespace NUMINAMATH_GPT_cds_total_l1232_123286

theorem cds_total (dawn_cds : ℕ) (h1 : dawn_cds = 10) (h2 : ∀ kristine_cds : ℕ, kristine_cds = dawn_cds + 7) :
  dawn_cds + (dawn_cds + 7) = 27 :=
by
  sorry

end NUMINAMATH_GPT_cds_total_l1232_123286


namespace NUMINAMATH_GPT_smallest_base10_integer_l1232_123237

theorem smallest_base10_integer : 
  ∃ (a b x : ℕ), a > 2 ∧ b > 2 ∧ x = 2 * a + 1 ∧ x = b + 2 ∧ x = 7 := by
  sorry

end NUMINAMATH_GPT_smallest_base10_integer_l1232_123237


namespace NUMINAMATH_GPT_find_x_y_l1232_123210

theorem find_x_y (x y : ℝ) (h : (2 * x - 3 * y + 5) ^ 2 + |x - y + 2| = 0) : x = -1 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_y_l1232_123210


namespace NUMINAMATH_GPT_cos_C_l1232_123219

-- Define the data and conditions of the problem
variables {A B C : ℝ}
variables (triangle_ABC : Prop)
variable (h_sinA : Real.sin A = 4 / 5)
variable (h_cosB : Real.cos B = 12 / 13)

-- Statement of the theorem
theorem cos_C (h1 : triangle_ABC)
  (h2 : Real.sin A = 4 / 5)
  (h3 : Real.cos B = 12 / 13) :
  Real.cos C = -16 / 65 :=
sorry

end NUMINAMATH_GPT_cos_C_l1232_123219


namespace NUMINAMATH_GPT_smallest_prime_divisor_of_sum_first_100_is_5_l1232_123257

-- Conditions: The sum of the first 100 natural numbers
def sum_first_n_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Prime checking function to identify the smallest prime divisor
def smallest_prime_divisor (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 else
  if n % 3 = 0 then 3 else
  if n % 5 = 0 then 5 else
  n -- Such a simplification works because we know the answer must be within the first few primes.

-- Proof statement
theorem smallest_prime_divisor_of_sum_first_100_is_5 : smallest_prime_divisor (sum_first_n_numbers 100) = 5 :=
by
  -- Proof steps would follow here.
  sorry

end NUMINAMATH_GPT_smallest_prime_divisor_of_sum_first_100_is_5_l1232_123257


namespace NUMINAMATH_GPT_part_a_part_b_l1232_123295
open Set

def fantastic (n : ℕ) : Prop :=
  ∃ a b : ℚ, a > 0 ∧ b > 0 ∧ n = a + 1 / a + b + 1 / b

theorem part_a : ∃ᶠ p in at_top, Prime p ∧ ∀ k, ¬ fantastic (k * p) := 
  sorry

theorem part_b : ∃ᶠ p in at_top, Prime p ∧ ∃ k, fantastic (k * p) :=
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1232_123295


namespace NUMINAMATH_GPT_max_PA_PB_l1232_123269

noncomputable def max_distance (PA PB : ℝ) : ℝ :=
  PA + PB

theorem max_PA_PB {A B : ℝ × ℝ} (m : ℝ) :
  A = (0, 0) ∧
  B = (1, 3) ∧
  dist A B = 10 →
  max_distance (dist A B) (dist (1, 3) B) = 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_max_PA_PB_l1232_123269


namespace NUMINAMATH_GPT_probability_of_rain_l1232_123253

theorem probability_of_rain {p : ℝ} (h : p = 0.95) :
  ∃ (q : ℝ), q = (1 - p) ∧ q < p :=
by
  sorry

end NUMINAMATH_GPT_probability_of_rain_l1232_123253


namespace NUMINAMATH_GPT_factorization_result_l1232_123245

theorem factorization_result (a b : ℤ) (h : (16:ℚ) * x^2 - 106 * x - 105 = (8 * x + a) * (2 * x + b)) : a + 2 * b = -23 := by
  sorry

end NUMINAMATH_GPT_factorization_result_l1232_123245


namespace NUMINAMATH_GPT_unique_triad_l1232_123283

theorem unique_triad (x y z : ℕ) 
  (h_distinct: x ≠ y ∧ y ≠ z ∧ z ≠ x) 
  (h_gcd: Nat.gcd (Nat.gcd x y) z = 1)
  (h_div_properties: (z ∣ x + y) ∧ (x ∣ y + z) ∧ (y ∣ z + x)) :
  (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
sorry

end NUMINAMATH_GPT_unique_triad_l1232_123283


namespace NUMINAMATH_GPT_work_done_days_l1232_123226

theorem work_done_days (a_days : ℕ) (b_days : ℕ) (together_days : ℕ) (a_work_done : ℚ) (b_work_done : ℚ) (together_work : ℚ) : 
  a_days = 12 ∧ b_days = 15 ∧ together_days = 5 ∧ 
  a_work_done = 1/12 ∧ b_work_done = 1/15 ∧ together_work = 3/4 → 
  ∃ days : ℚ, a_days > 0 ∧ b_days > 0 ∧ together_days > 0 ∧ days = 3 := 
  sorry

end NUMINAMATH_GPT_work_done_days_l1232_123226


namespace NUMINAMATH_GPT_range_of_a_l1232_123292

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem range_of_a (a : ℝ) :
  (∀ (b : ℝ), (b ≤ 0) → ∀ (x : ℝ), (x > Real.exp 1 ∧ x ≤ Real.exp 2) → f a b x ≥ x) →
  a ≥ Real.exp 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1232_123292


namespace NUMINAMATH_GPT_shifted_parabola_expression_l1232_123260

theorem shifted_parabola_expression (x y x' y' : ℝ) 
  (h_initial : y = (x + 2)^2 + 3)
  (h_shift_right : x' = x - 3)
  (h_shift_down : y' = y - 2)
  : y' = (x' - 1)^2 + 1 := 
sorry

end NUMINAMATH_GPT_shifted_parabola_expression_l1232_123260


namespace NUMINAMATH_GPT_proof_problem_l1232_123214

-- Define the conditions
def a : ℤ := -3
def b : ℤ := -4
def cond1 := a^4 = 81
def cond2 := b^3 = -64

-- Define the goal in terms of the conditions
theorem proof_problem : a^4 + b^3 = 17 :=
by
  have h1 : a^4 = 81 := sorry
  have h2 : b^3 = -64 := sorry
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_proof_problem_l1232_123214


namespace NUMINAMATH_GPT_subway_train_speed_l1232_123287

open Nat

-- Define the speed function
def speed (s : ℕ) : ℕ := s^2 + 2*s

-- Define the theorem to be proved
theorem subway_train_speed (t : ℕ) (ht : 0 ≤ t ∧ t ≤ 7) (h_speed : speed 7 - speed t = 28) : t = 5 :=
by
  sorry

end NUMINAMATH_GPT_subway_train_speed_l1232_123287


namespace NUMINAMATH_GPT_bart_trees_needed_l1232_123296

-- Define the constants and conditions given
def firewood_per_tree : Nat := 75
def logs_burned_per_day : Nat := 5
def days_in_november : Nat := 30
def days_in_december : Nat := 31
def days_in_january : Nat := 31
def days_in_february : Nat := 28

-- Calculate the total number of days from November 1 through February 28
def total_days : Nat := days_in_november + days_in_december + days_in_january + days_in_february

-- Calculate the total number of pieces of firewood needed
def total_firewood_needed : Nat := total_days * logs_burned_per_day

-- Calculate the number of trees needed
def trees_needed : Nat := total_firewood_needed / firewood_per_tree

-- The proof statement
theorem bart_trees_needed : trees_needed = 8 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_bart_trees_needed_l1232_123296


namespace NUMINAMATH_GPT_proof_problem_l1232_123234

-- Define the given condition as a constant
def condition : Prop := 213 * 16 = 3408

-- Define the statement we need to prove under the given condition
theorem proof_problem (h : condition) : 0.16 * 2.13 = 0.3408 := 
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l1232_123234


namespace NUMINAMATH_GPT_fraction_problem_l1232_123293

theorem fraction_problem (N D : ℚ) (h1 : 1.30 * N / (0.85 * D) = 25 / 21) : 
  N / D = 425 / 546 :=
sorry

end NUMINAMATH_GPT_fraction_problem_l1232_123293


namespace NUMINAMATH_GPT_maria_gave_towels_l1232_123207

def maria_towels (green_white total_left : Nat) : Nat :=
  green_white - total_left

theorem maria_gave_towels :
  ∀ (green white left given : Nat),
    green = 35 →
    white = 21 →
    left = 22 →
    given = 34 →
    maria_towels (green + white) left = given :=
by
  intros green white left given
  intros hgreen hwhite hleft hgiven
  rw [hgreen, hwhite, hleft, hgiven]
  sorry

end NUMINAMATH_GPT_maria_gave_towels_l1232_123207


namespace NUMINAMATH_GPT_sqrt_9_eq_pos_neg_3_l1232_123233

theorem sqrt_9_eq_pos_neg_3 : ∀ x : ℝ, x^2 = 9 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_9_eq_pos_neg_3_l1232_123233


namespace NUMINAMATH_GPT_solution_of_fractional_equation_l1232_123275

theorem solution_of_fractional_equation :
  (∃ x, x ≠ 3 ∧ (x / (x - 3) - 2 = (m - 1) / (x - 3))) → m = 4 := by
  sorry

end NUMINAMATH_GPT_solution_of_fractional_equation_l1232_123275


namespace NUMINAMATH_GPT_tan_20_add_4sin_20_eq_sqrt3_l1232_123274

theorem tan_20_add_4sin_20_eq_sqrt3 : Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180) = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_20_add_4sin_20_eq_sqrt3_l1232_123274


namespace NUMINAMATH_GPT_circle_transformation_l1232_123254

theorem circle_transformation (c : ℝ × ℝ) (v : ℝ × ℝ) (h_center : c = (8, -3)) (h_vector : v = (2, -5)) :
  let reflected := (c.2, c.1)
  let translated := (reflected.1 + v.1, reflected.2 + v.2)
  translated = (-1, 3) :=
by
  sorry

end NUMINAMATH_GPT_circle_transformation_l1232_123254


namespace NUMINAMATH_GPT_percent_daffodils_is_57_l1232_123289

-- Condition 1: Four-sevenths of the flowers are yellow
def fraction_yellow : ℚ := 4 / 7

-- Condition 2: Two-thirds of the red flowers are daffodils
def fraction_red_daffodils_given_red : ℚ := 2 / 3

-- Condition 3: Half of the yellow flowers are tulips
def fraction_yellow_tulips_given_yellow : ℚ := 1 / 2

-- Calculate fractions of yellow and red flowers
def fraction_red : ℚ := 1 - fraction_yellow

-- Calculate fractions of daffodils
def fraction_yellow_daffodils : ℚ := fraction_yellow * (1 - fraction_yellow_tulips_given_yellow)
def fraction_red_daffodils : ℚ := fraction_red * fraction_red_daffodils_given_red

-- Total fraction of daffodils
def fraction_daffodils : ℚ := fraction_yellow_daffodils + fraction_red_daffodils

-- Proof statement
theorem percent_daffodils_is_57 :
  fraction_daffodils * 100 = 57 := by
  sorry

end NUMINAMATH_GPT_percent_daffodils_is_57_l1232_123289


namespace NUMINAMATH_GPT_aunt_may_milk_left_l1232_123235

theorem aunt_may_milk_left
  (morning_milk : ℕ)
  (evening_milk : ℕ)
  (sold_milk : ℕ)
  (leftover_milk : ℕ)
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk + leftover_milk - sold_milk = 148 :=
by
  sorry

end NUMINAMATH_GPT_aunt_may_milk_left_l1232_123235


namespace NUMINAMATH_GPT_hypotenuse_square_l1232_123202

-- Define the right triangle property and the consecutive integer property
variables (a b c : ℤ)

-- Noncomputable definition will be used as we are proving a property related to integers
noncomputable def consecutive_integers (a b : ℤ) : Prop := b = a + 1

-- Define the statement to prove
theorem hypotenuse_square (h_consec : consecutive_integers a b) (h_right_triangle : a * a + b * b = c * c) : 
  c * c = 2 * a * a + 2 * a + 1 :=
by {
  -- We only need to state the theorem
  sorry
}

end NUMINAMATH_GPT_hypotenuse_square_l1232_123202


namespace NUMINAMATH_GPT_findYearsForTwiceAge_l1232_123266

def fatherSonAges : ℕ := 33

def fatherAge : ℕ := fatherSonAges + 35

def yearsForTwiceAge (x : ℕ) : Prop :=
  fatherAge + x = 2 * (fatherSonAges + x)

theorem findYearsForTwiceAge : ∃ x, yearsForTwiceAge x :=
  ⟨2, sorry⟩

end NUMINAMATH_GPT_findYearsForTwiceAge_l1232_123266


namespace NUMINAMATH_GPT_water_formed_on_combining_l1232_123271

theorem water_formed_on_combining (molar_mass_water : ℝ) (n_NaOH : ℝ) (n_HCl : ℝ) :
  n_NaOH = 1 ∧ n_HCl = 1 ∧ molar_mass_water = 18.01528 → 
  n_NaOH * molar_mass_water = 18.01528 :=
by sorry

end NUMINAMATH_GPT_water_formed_on_combining_l1232_123271


namespace NUMINAMATH_GPT_range_of_m_iff_l1232_123261

noncomputable def range_of_m (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (0 < x) → (0 < y) → ((2 / x) + (1 / y) = 1) → (x + 2 * y > m^2 + 2 * m)

theorem range_of_m_iff : (range_of_m m) ↔ (-4 < m ∧ m < 2) :=
  sorry

end NUMINAMATH_GPT_range_of_m_iff_l1232_123261


namespace NUMINAMATH_GPT_quadratic_inequality_solution_range_l1232_123270

theorem quadratic_inequality_solution_range (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3 / 8 < 0) ↔ (-3 / 2 < k ∧ k < 0) := sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_range_l1232_123270


namespace NUMINAMATH_GPT_complex_expression_value_l1232_123285

theorem complex_expression_value :
  ((6^2 - 4^2) + 2)^3 / 2 = 5324 :=
by
  sorry

end NUMINAMATH_GPT_complex_expression_value_l1232_123285


namespace NUMINAMATH_GPT_sin_cos_sum_l1232_123215

theorem sin_cos_sum (α x y r : ℝ) (h1 : x = 2) (h2 : y = -1) (h3 : r = Real.sqrt 5)
    (h4 : ∀ θ, x = r * Real.cos θ) (h5 : ∀ θ, y = r * Real.sin θ) : 
    Real.sin α + Real.cos α = (- 1 / Real.sqrt 5) + (2 / Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_sum_l1232_123215


namespace NUMINAMATH_GPT_michael_remaining_books_l1232_123297

theorem michael_remaining_books (total_books : ℕ) (read_percentage : ℚ) 
  (H1 : total_books = 210) (H2 : read_percentage = 0.60) : 
  (total_books - (read_percentage * total_books) : ℚ) = 84 :=
by
  sorry

end NUMINAMATH_GPT_michael_remaining_books_l1232_123297


namespace NUMINAMATH_GPT_important_emails_l1232_123264

theorem important_emails (total_emails : ℕ) (spam_frac : ℚ) (promotional_frac : ℚ) (spam_email_count : ℕ) (remaining_emails : ℕ) (promotional_email_count : ℕ) (important_email_count : ℕ) :
  total_emails = 800 ∧ spam_frac = 3 / 7 ∧ promotional_frac = 5 / 11 ∧ spam_email_count = 343 ∧ remaining_emails = 457 ∧ promotional_email_count = 208 →
sorry

end NUMINAMATH_GPT_important_emails_l1232_123264


namespace NUMINAMATH_GPT_absent_children_l1232_123288

/-- On a school's annual day, sweets were to be equally distributed amongst 112 children. 
But on that particular day, some children were absent. Thus, the remaining children got 6 extra sweets. 
Each child was originally supposed to get 15 sweets. Prove that 32 children were absent. -/
theorem absent_children (A : ℕ) 
  (total_children : ℕ := 112) 
  (sweets_per_child : ℕ := 15) 
  (extra_sweets : ℕ := 6)
  (absent_eq : (total_children - A) * (sweets_per_child + extra_sweets) = total_children * sweets_per_child) : 
  A = 32 := 
by
  sorry

end NUMINAMATH_GPT_absent_children_l1232_123288


namespace NUMINAMATH_GPT_average_speed_l1232_123224

theorem average_speed (x : ℝ) (h1 : x > 0) :
  let dist1 := x
  let speed1 := 40
  let dist2 := 4 * x
  let speed2 := 20
  let total_dist := dist1 + dist2
  let time1 := dist1 / speed1
  let time2 := dist2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_dist / total_time
  avg_speed = 200 / 9 :=
by
  -- Definitions
  let dist1 := x
  let speed1 := 40
  let dist2 := 4 * x
  let speed2 := 20
  let total_dist := dist1 + dist2
  let time1 := dist1 / speed1
  let time2 := dist2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_dist / total_time
  -- Proof structure, concluding with the correct answer.
  sorry

end NUMINAMATH_GPT_average_speed_l1232_123224


namespace NUMINAMATH_GPT_boxes_in_carton_l1232_123209

theorem boxes_in_carton (cost_per_pack : ℕ) (packs_per_box : ℕ) (cost_dozen_cartons : ℕ) 
  (h1 : cost_per_pack = 1) (h2 : packs_per_box = 10) (h3 : cost_dozen_cartons = 1440) :
  (cost_dozen_cartons / 12) / (cost_per_pack * packs_per_box) = 12 :=
by
  sorry

end NUMINAMATH_GPT_boxes_in_carton_l1232_123209


namespace NUMINAMATH_GPT_Julia_played_with_11_kids_on_Monday_l1232_123290

theorem Julia_played_with_11_kids_on_Monday
  (kids_on_Tuesday : ℕ)
  (kids_on_Monday : ℕ) 
  (h1 : kids_on_Tuesday = 12)
  (h2 : kids_on_Tuesday = kids_on_Monday + 1) : 
  kids_on_Monday = 11 := 
by
  sorry

end NUMINAMATH_GPT_Julia_played_with_11_kids_on_Monday_l1232_123290


namespace NUMINAMATH_GPT_range_of_a_l1232_123222

variable (a : ℝ)

def proposition_p (a : ℝ) : Prop := 0 < a ∧ a < 1

def proposition_q (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 - x + a > 0 ∧ 1 - 4 * a^2 < 0

theorem range_of_a : (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) →
  (0 < a ∧ a ≤ 1/2 ∨ a ≥ 1) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1232_123222


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l1232_123243

open Real

theorem trigonometric_identity_proof (x y : ℝ) (hx : sin x / sin y = 4) (hy : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 169 / 381 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l1232_123243


namespace NUMINAMATH_GPT_pq_square_identity_l1232_123236

theorem pq_square_identity (p q : ℝ) (h1 : p - q = 4) (h2 : p * q = -2) : p^2 + q^2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_pq_square_identity_l1232_123236


namespace NUMINAMATH_GPT_elsa_data_remaining_l1232_123227

variable (data_total : ℕ) (data_youtube : ℕ)

def data_remaining_after_youtube (data_total data_youtube : ℕ) : ℕ := data_total - data_youtube

def data_fraction_spent_on_facebook (data_left : ℕ) : ℕ := (2 * data_left) / 5

theorem elsa_data_remaining
  (h_data_total : data_total = 500)
  (h_data_youtube : data_youtube = 300) :
  data_remaining_after_youtube data_total data_youtube
  - data_fraction_spent_on_facebook (data_remaining_after_youtube data_total data_youtube) 
  = 120 :=
by
  sorry

end NUMINAMATH_GPT_elsa_data_remaining_l1232_123227


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1232_123240

variable (p q : Prop)

theorem sufficient_but_not_necessary (h : p ∧ q) : ¬¬p :=
  by sorry -- Proof not required

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1232_123240


namespace NUMINAMATH_GPT_solve_system_l1232_123244

def eq1 (x y : ℝ) : Prop := x^2 * y - x * y^2 - 5 * x + 5 * y + 3 = 0
def eq2 (x y : ℝ) : Prop := x^3 * y - x * y^3 - 5 * x^2 + 5 * y^2 + 15 = 0

theorem solve_system :
  ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧ x = 4 ∧ y = 1 := by
  sorry

end NUMINAMATH_GPT_solve_system_l1232_123244


namespace NUMINAMATH_GPT_number_of_companion_relation_subsets_l1232_123225

def isCompanionRelationSet (A : Set ℚ) : Prop :=
  ∀ x ∈ A, (x ≠ 0 → (1 / x) ∈ A)

def M : Set ℚ := {-1, 0, 1 / 3, 1 / 2, 1, 2, 3, 4}

theorem number_of_companion_relation_subsets :
  ∃ n, n = 15 ∧
  (∀ A ⊆ M, isCompanionRelationSet A) :=
sorry

end NUMINAMATH_GPT_number_of_companion_relation_subsets_l1232_123225


namespace NUMINAMATH_GPT_range_of_BD_l1232_123232

-- Define the types of points and triangle
variables {α : Type*} [MetricSpace α]

-- Hypothesis: AD is the median of triangle ABC
-- Definition of lengths AB, AC, and that BD = CD.
def isMedianOnBC (A B C D : α) : Prop :=
  dist A B = 5 ∧ dist A C = 7 ∧ dist B D = dist C D

-- The theorem to be proven
theorem range_of_BD {A B C D : α} (h : isMedianOnBC A B C D) : 
  1 < dist B D ∧ dist B D < 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_BD_l1232_123232


namespace NUMINAMATH_GPT_smallest_n_such_that_floor_eq_1989_l1232_123221

theorem smallest_n_such_that_floor_eq_1989 :
  ∃ (n : ℕ), (∀ k, k < n -> ¬(∃ x : ℤ, ⌊(10^k : ℚ) / x⌋ = 1989)) ∧ (∃ x : ℤ, ⌊(10^n : ℚ) / x⌋ = 1989) :=
sorry

end NUMINAMATH_GPT_smallest_n_such_that_floor_eq_1989_l1232_123221


namespace NUMINAMATH_GPT_equip_20posts_with_5new_weapons_l1232_123229

/-- 
Theorem: In a line of 20 defense posts, the number of ways to equip 5 different new weapons 
such that:
1. The first and last posts are not equipped with new weapons.
2. Each set of 5 consecutive posts has at least one post equipped with a new weapon.
3. No two adjacent posts are equipped with new weapons.
is 69600. 
-/
theorem equip_20posts_with_5new_weapons : ∃ ways : ℕ, ways = 69600 :=
by
  sorry

end NUMINAMATH_GPT_equip_20posts_with_5new_weapons_l1232_123229


namespace NUMINAMATH_GPT_jasmine_pies_l1232_123242

-- Definitions based on the given conditions
def total_pies : Nat := 30
def raspberry_part : Nat := 2
def peach_part : Nat := 5
def plum_part : Nat := 3
def total_parts : Nat := raspberry_part + peach_part + plum_part

-- Calculate pies per part
def pies_per_part : Nat := total_pies / total_parts

-- Prove the statement
theorem jasmine_pies :
  (plum_part * pies_per_part = 9) :=
by
  -- The statement and proof will go here, but we are skipping the proof part.
  sorry

end NUMINAMATH_GPT_jasmine_pies_l1232_123242


namespace NUMINAMATH_GPT_sum_of_arithmetic_series_l1232_123230

-- Define the conditions
def first_term := 1
def last_term := 12
def number_of_terms := 12

-- Prop statement that the sum of the arithmetic series equals 78
theorem sum_of_arithmetic_series : (number_of_terms / 2) * (first_term + last_term) = 78 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_series_l1232_123230


namespace NUMINAMATH_GPT_find_x_l1232_123252

theorem find_x {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0)
    (h1 : x + 1/y = 10) (h2 : y + 1/x = 5/12) : x = 4 ∨ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1232_123252


namespace NUMINAMATH_GPT_side_length_square_base_l1232_123204

theorem side_length_square_base 
  (height : ℕ) (volume : ℕ) (A : ℕ) (s : ℕ) 
  (h_height : height = 8) 
  (h_volume : volume = 288) 
  (h_base_area : A = volume / height) 
  (h_square_base : A = s ^ 2) :
  s = 6 :=
by
  sorry

end NUMINAMATH_GPT_side_length_square_base_l1232_123204


namespace NUMINAMATH_GPT_k_cannot_be_zero_l1232_123268

theorem k_cannot_be_zero (k : ℝ) (h₁ : k ≠ 0) (h₂ : 4 - 2 * k > 0) : k ≠ 0 :=
by 
  exact h₁

end NUMINAMATH_GPT_k_cannot_be_zero_l1232_123268


namespace NUMINAMATH_GPT_polynomial_root_sum_eq_48_l1232_123284

theorem polynomial_root_sum_eq_48 {r s t : ℕ} (h1 : r * s * t = 2310) 
  (h2 : r > 0) (h3 : s > 0) (h4 : t > 0) : r + s + t = 48 :=
sorry

end NUMINAMATH_GPT_polynomial_root_sum_eq_48_l1232_123284


namespace NUMINAMATH_GPT_probability_no_consecutive_tails_probability_no_consecutive_tails_in_five_tosses_l1232_123238

def countWays (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else countWays (n - 1) + countWays (n - 2)

theorem probability_no_consecutive_tails : countWays 5 = 13 :=
by
  sorry

theorem probability_no_consecutive_tails_in_five_tosses : 
  (countWays 5) / (2^5 : ℕ) = 13 / 32 :=
by
  sorry

end NUMINAMATH_GPT_probability_no_consecutive_tails_probability_no_consecutive_tails_in_five_tosses_l1232_123238


namespace NUMINAMATH_GPT_min_students_changed_l1232_123294

-- Define the initial percentage of "Yes" and "No" at the beginning of the year
def initial_yes_percentage : ℝ := 0.40
def initial_no_percentage : ℝ := 0.60

-- Define the final percentage of "Yes" and "No" at the end of the year
def final_yes_percentage : ℝ := 0.80
def final_no_percentage : ℝ := 0.20

-- Define the minimum possible percentage of students that changed their mind
def min_changed_percentage : ℝ := 0.40

-- Prove that the minimum possible percentage of students that changed their mind is 40%
theorem min_students_changed :
  (final_yes_percentage - initial_yes_percentage = min_changed_percentage) ∧
  (initial_yes_percentage = final_yes_percentage - min_changed_percentage) ∧
  (initial_no_percentage - min_changed_percentage = final_no_percentage) :=
by
  sorry

end NUMINAMATH_GPT_min_students_changed_l1232_123294


namespace NUMINAMATH_GPT_exists_ab_negated_l1232_123251

theorem exists_ab_negated :
  ¬ (∀ a b : ℝ, (a + b = 0 → a^2 + b^2 = 0)) ↔ 
  ∃ a b : ℝ, (a + b = 0 ∧ a^2 + b^2 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_exists_ab_negated_l1232_123251


namespace NUMINAMATH_GPT_maximal_value_fraction_l1232_123258

noncomputable def maximum_value_ratio (a b c : ℝ) (S : ℝ) : ℝ :=
  if S = c^2 / 4 then 2 * Real.sqrt 2 else 0

theorem maximal_value_fraction (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (area_cond : 1/2 * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = c^2 / 4) :
  maximum_value_ratio a b c (c^2/4) = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_maximal_value_fraction_l1232_123258


namespace NUMINAMATH_GPT_blocks_found_l1232_123255

def initial_blocks : ℕ := 2
def final_blocks : ℕ := 86

theorem blocks_found : (final_blocks - initial_blocks) = 84 :=
by
  sorry

end NUMINAMATH_GPT_blocks_found_l1232_123255


namespace NUMINAMATH_GPT_bricks_needed_to_build_wall_l1232_123211

def volume_of_brick (length_brick height_brick thickness_brick : ℤ) : ℤ :=
  length_brick * height_brick * thickness_brick

def volume_of_wall (length_wall height_wall thickness_wall : ℤ) : ℤ :=
  length_wall * height_wall * thickness_wall

def number_of_bricks_needed (length_wall height_wall thickness_wall length_brick height_brick thickness_brick : ℤ) : ℤ :=
  (volume_of_wall length_wall height_wall thickness_wall + volume_of_brick length_brick height_brick thickness_brick - 1) / 
  volume_of_brick length_brick height_brick thickness_brick

theorem bricks_needed_to_build_wall : number_of_bricks_needed 800 100 5 25 11 6 = 243 := 
  by 
    sorry

end NUMINAMATH_GPT_bricks_needed_to_build_wall_l1232_123211


namespace NUMINAMATH_GPT_area_of_rectangle_l1232_123281

theorem area_of_rectangle (x : ℝ) (hx : 0 < x) :
  let length := 3 * x - 1
  let width := 2 * x + 1 / 2
  let area := length * width
  area = 6 * x^2 - 1 / 2 * x - 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1232_123281


namespace NUMINAMATH_GPT_problem_l1232_123212

variable (α : ℝ)

def setA : Set ℝ := {Real.sin α, Real.cos α, 1}
def setB : Set ℝ := {Real.sin α ^ 2, Real.sin α + Real.cos α, 0}
theorem problem (h : setA α = setB α) : Real.sin α ^ 2009 + Real.cos α ^ 2009 = -1 := 
by 
  sorry

end NUMINAMATH_GPT_problem_l1232_123212


namespace NUMINAMATH_GPT_class_raised_initial_amount_l1232_123217

/-- Miss Grayson's class raised some money for their field trip.
Each student contributed $5 each.
There are 20 students in her class.
The cost of the trip is $7 for each student.
After all the field trip costs were paid, there is $10 left in Miss Grayson's class fund.
Prove that the class initially raised $150 for the field trip. -/
theorem class_raised_initial_amount
  (students : ℕ)
  (contribution_per_student : ℕ)
  (cost_per_student : ℕ)
  (remaining_fund : ℕ)
  (total_students : students = 20)
  (per_student_contribution : contribution_per_student = 5)
  (per_student_cost : cost_per_student = 7)
  (remaining_amount : remaining_fund = 10) :
  (students * contribution_per_student + remaining_fund) = 150 := 
sorry

end NUMINAMATH_GPT_class_raised_initial_amount_l1232_123217


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_l1232_123249

theorem inequality_holds_for_all_x (m : ℝ) :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_l1232_123249


namespace NUMINAMATH_GPT_new_solid_edges_l1232_123247

-- Definitions based on conditions
def original_vertices : ℕ := 8
def original_edges : ℕ := 12
def new_edges_per_vertex : ℕ := 3
def number_of_vertices : ℕ := original_vertices

-- Conclusion to prove
theorem new_solid_edges : 
  (original_edges + new_edges_per_vertex * number_of_vertices) = 36 := 
by
  sorry

end NUMINAMATH_GPT_new_solid_edges_l1232_123247


namespace NUMINAMATH_GPT_ratio_of_a_to_b_l1232_123200

theorem ratio_of_a_to_b (a y b : ℝ) (h1 : a = 0) (h2 : b = 2 * y) : a / b = 0 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_a_to_b_l1232_123200


namespace NUMINAMATH_GPT_reciprocal_is_correct_l1232_123298

-- Define the initial number
def num : ℚ := -1 / 2023

-- Define the expected reciprocal
def reciprocal : ℚ := -2023

-- Theorem stating the reciprocal of the given number is the expected reciprocal
theorem reciprocal_is_correct : 1 / num = reciprocal :=
  by
    -- The actual proof can be filled in here
    sorry

end NUMINAMATH_GPT_reciprocal_is_correct_l1232_123298


namespace NUMINAMATH_GPT_pattern_generalization_l1232_123223

theorem pattern_generalization (n : ℕ) (h : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 :=
by
  -- TODO: The proof will be filled in later
  sorry

end NUMINAMATH_GPT_pattern_generalization_l1232_123223


namespace NUMINAMATH_GPT_larger_exceeds_smaller_times_l1232_123241

theorem larger_exceeds_smaller_times {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_diff : a ≠ b)
  (h_eq : a^3 - b^3 = 3 * (2 * a^2 * b - 3 * a * b^2 + b^3)) : a = 4 * b :=
sorry

end NUMINAMATH_GPT_larger_exceeds_smaller_times_l1232_123241


namespace NUMINAMATH_GPT_height_inequality_triangle_l1232_123248

theorem height_inequality_triangle (a b c h_a h_b h_c Δ : ℝ) (n : ℝ) 
  (ha : h_a = 2 * Δ / a)
  (hb : h_b = 2 * Δ / b)
  (hc : h_c = 2 * Δ / c)
  (n_pos : n > 0) :
  (a * h_b)^n + (b * h_c)^n + (c * h_a)^n ≥ 3 * 2^n * Δ^n := 
sorry

end NUMINAMATH_GPT_height_inequality_triangle_l1232_123248


namespace NUMINAMATH_GPT_problem_statement_l1232_123201

theorem problem_statement :
  102^3 + 3 * 102^2 + 3 * 102 + 1 = 1092727 :=
  by sorry

end NUMINAMATH_GPT_problem_statement_l1232_123201


namespace NUMINAMATH_GPT_range_of_m_satisfies_inequality_l1232_123273

theorem range_of_m_satisfies_inequality (m : ℝ) :
  ((∀ x : ℝ, (1 - m^2) * x^2 - (1 + m) * x - 1 < 0) ↔ (m ≤ -1 ∨ m > 5/3)) :=
sorry

end NUMINAMATH_GPT_range_of_m_satisfies_inequality_l1232_123273


namespace NUMINAMATH_GPT_remainder_of_3_pow_99_plus_5_mod_9_l1232_123205

theorem remainder_of_3_pow_99_plus_5_mod_9 : (3 ^ 99 + 5) % 9 = 5 := by
  -- Here we state the main goal
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_remainder_of_3_pow_99_plus_5_mod_9_l1232_123205


namespace NUMINAMATH_GPT_n_decomposable_form_l1232_123213

theorem n_decomposable_form (n : ℕ) (a : ℕ) (h₁ : a > 2) (h₂ : ∃ k, 1 < k ∧ n = 2^k) :
  (∀ d : ℕ, d ∣ n ∧ d ≠ n → (a^n - 2^n) % (a^d + 2^d) = 0) → ∃ k, 1 < k ∧ n = 2^k :=
by {
  sorry
}

end NUMINAMATH_GPT_n_decomposable_form_l1232_123213


namespace NUMINAMATH_GPT_remainder_of_3a_minus_b_divided_by_5_l1232_123265

theorem remainder_of_3a_minus_b_divided_by_5 (a b : ℕ) (m n : ℤ) 
(h1 : 3 * a > b) 
(h2 : a = 5 * m + 1) 
(h3 : b = 5 * n + 4) : 
(3 * a - b) % 5 = 4 := 
sorry

end NUMINAMATH_GPT_remainder_of_3a_minus_b_divided_by_5_l1232_123265


namespace NUMINAMATH_GPT_cistern_emptying_time_l1232_123231

theorem cistern_emptying_time (R L : ℝ) (h1 : R * 8 = 1) (h2 : (R - L) * 10 = 1) : 1 / L = 40 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_cistern_emptying_time_l1232_123231


namespace NUMINAMATH_GPT_karen_age_is_10_l1232_123220

-- Definitions for the given conditions
def ages : List ℕ := [2, 4, 6, 8, 10, 12, 14]

def to_park (a b : ℕ) : Prop := a + b = 20
def to_pool (a b : ℕ) : Prop := 3 < a ∧ a < 9 ∧ 3 < b ∧ b < 9
def stayed_home (karen_age : ℕ) : Prop := karen_age = 10

-- Theorem stating Karen's age is 10 given the conditions
theorem karen_age_is_10 :
  ∃ (a b c d e f g : ℕ),
  ages = [a, b, c, d, e, f, g] ∧
  ((to_park a b ∨ to_park a c ∨ to_park a d ∨ to_park a e ∨ to_park a f ∨ to_park a g ∨
  to_park b c ∨ to_park b d ∨ to_park b e ∨ to_park b f ∨ to_park b g ∨
  to_park c d ∨ to_park c e ∨ to_park c f ∨ to_park c g ∨
  to_park d e ∨ to_park d f ∨ to_park d g ∨
  to_park e f ∨ to_park e g ∨
  to_park f g)) ∧
  ((to_pool a b ∨ to_pool a c ∨ to_pool a d ∨ to_pool a e ∨ to_pool a f ∨ to_pool a g ∨
  to_pool b c ∨ to_pool b d ∨ to_pool b e ∨ to_pool b f ∨ to_pool b g ∨
  to_pool c d ∨ to_pool c e ∨ to_pool c f ∨
  to_pool d e ∨ to_pool d f ∨
  to_pool e f ∨
  to_pool f g)) ∧
  stayed_home 4 :=
sorry

end NUMINAMATH_GPT_karen_age_is_10_l1232_123220


namespace NUMINAMATH_GPT_cos_210_eq_neg_sqrt_3_div_2_l1232_123263

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end NUMINAMATH_GPT_cos_210_eq_neg_sqrt_3_div_2_l1232_123263


namespace NUMINAMATH_GPT_circle_radius_c_value_l1232_123218

theorem circle_radius_c_value (x y c : ℝ) (h₁ : x^2 + 8 * x + y^2 + 10 * y + c = 0) (h₂ : (x+4)^2 + (y+5)^2 = 25) :
  c = -16 :=
by sorry

end NUMINAMATH_GPT_circle_radius_c_value_l1232_123218


namespace NUMINAMATH_GPT_average_of_first_5_subjects_l1232_123277

theorem average_of_first_5_subjects (avg_6_subjects : ℚ) (marks_6th_subject : ℚ) (total_subjects : ℕ) (total_marks_6_subjects : ℚ) (total_marks_5_subjects : ℚ) (avg_5_subjects : ℚ) :
  avg_6_subjects = 77 ∧ marks_6th_subject = 92 ∧ total_subjects = 6 ∧ total_marks_6_subjects = avg_6_subjects * total_subjects ∧ total_marks_5_subjects = total_marks_6_subjects - marks_6th_subject ∧ avg_5_subjects = total_marks_5_subjects / 5
  → avg_5_subjects = 74 := by
  sorry

end NUMINAMATH_GPT_average_of_first_5_subjects_l1232_123277


namespace NUMINAMATH_GPT_location_determined_l1232_123280

def determine_location(p : String) : Prop :=
  p = "Longitude 118°E, Latitude 40°N"

axiom row_2_in_cinema : ¬determine_location "Row 2 in a cinema"
axiom daqiao_south_road_nanjing : ¬determine_location "Daqiao South Road in Nanjing"
axiom thirty_degrees_northeast : ¬determine_location "30° northeast"
axiom longitude_latitude : determine_location "Longitude 118°E, Latitude 40°N"

theorem location_determined : determine_location "Longitude 118°E, Latitude 40°N" :=
longitude_latitude

end NUMINAMATH_GPT_location_determined_l1232_123280


namespace NUMINAMATH_GPT_minnie_mounts_time_period_l1232_123250

theorem minnie_mounts_time_period (M D : ℕ) 
  (mickey_daily_mounts_eq : 2 * M - 6 = 14)
  (minnie_mounts_per_day_eq : M = D + 3) : 
  D = 7 := 
by
  sorry

end NUMINAMATH_GPT_minnie_mounts_time_period_l1232_123250


namespace NUMINAMATH_GPT_min_coins_needed_l1232_123208

-- Definitions for coins
def coins (pennies nickels dimes quarters : Nat) : Nat :=
  pennies + nickels + dimes + quarters

-- Condition: minimum number of coins to pay any amount less than a dollar
def can_pay_any_amount (pennies nickels dimes quarters : Nat) : Prop :=
  ∀ (amount : Nat), 1 ≤ amount ∧ amount < 100 →
  ∃ (p n d q : Nat), p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧ q ≤ quarters ∧
  p + 5 * n + 10 * d + 25 * q = amount

-- The main Lean 4 statement
theorem min_coins_needed :
  ∃ (pennies nickels dimes quarters : Nat),
    coins pennies nickels dimes quarters = 11 ∧
    can_pay_any_amount pennies nickels dimes quarters :=
sorry

end NUMINAMATH_GPT_min_coins_needed_l1232_123208


namespace NUMINAMATH_GPT_average_productivity_l1232_123279

theorem average_productivity (T : ℕ) (total_words : ℕ) (increased_time_fraction : ℚ) (increased_productivity_fraction : ℚ) :
  T = 100 →
  total_words = 60000 →
  increased_time_fraction = 0.2 →
  increased_productivity_fraction = 1.5 →
  (total_words / T : ℚ) = 600 :=
by
  sorry

end NUMINAMATH_GPT_average_productivity_l1232_123279


namespace NUMINAMATH_GPT_find_n_l1232_123267

theorem find_n :
  ∃ n : ℕ, 50 ≤ n ∧ n ≤ 150 ∧
          n % 7 = 0 ∧
          n % 9 = 3 ∧
          n % 6 = 3 ∧
          n = 75 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1232_123267


namespace NUMINAMATH_GPT_subset_iff_l1232_123259

open Set

noncomputable def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < a}

theorem subset_iff (a : ℝ) : A ⊆ B a ↔ 2 ≤ a :=
by sorry

end NUMINAMATH_GPT_subset_iff_l1232_123259


namespace NUMINAMATH_GPT_binom_7_4_l1232_123291

theorem binom_7_4 : Nat.choose 7 4 = 35 := 
by
  sorry

end NUMINAMATH_GPT_binom_7_4_l1232_123291


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1232_123262

def transformable (w1 w2 : String) : Prop :=
∀ q : String → String → Prop,
  (q "xy" "yyx") →
  (q "xt" "ttx") →
  (q "yt" "ty") →
  (q w1 w2)

theorem part_a : ¬ transformable "xy" "xt" :=
sorry

theorem part_b : ¬ transformable "xytx" "txyt" :=
sorry

theorem part_c : transformable "xtxyy" "ttxyyyyx" :=
sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1232_123262


namespace NUMINAMATH_GPT_xiaozhang_participates_in_martial_arts_l1232_123203

theorem xiaozhang_participates_in_martial_arts
  (row : Prop) (shoot : Prop) (martial : Prop)
  (Zhang Wang Li: Prop → Prop)
  (H1 : ¬  Zhang row ∧ ¬ Wang row)
  (H2 : ∃ (n m : ℕ), Zhang (shoot ∨ martial) = (n > 0) ∧ Wang (shoot ∨ martial) = (m > 0) ∧ m = n + 1)
  (H3 : ¬ Li shoot ∧ (Li martial ∨ Li row)) :
  Zhang martial :=
by
  sorry

end NUMINAMATH_GPT_xiaozhang_participates_in_martial_arts_l1232_123203


namespace NUMINAMATH_GPT_volume_of_cube_is_correct_l1232_123282

-- Define necessary constants and conditions
def cost_in_paise : ℕ := 34398
def rate_per_sq_cm : ℕ := 13
def surface_area : ℕ := cost_in_paise / rate_per_sq_cm
def face_area : ℕ := surface_area / 6
def side_length : ℕ := Nat.sqrt face_area
def volume : ℕ := side_length ^ 3

-- Prove the volume of the cube
theorem volume_of_cube_is_correct : volume = 9261 := by
  -- Using given conditions and basic arithmetic 
  sorry

end NUMINAMATH_GPT_volume_of_cube_is_correct_l1232_123282


namespace NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l1232_123246

noncomputable def alpha : ℝ := sorry
axiom alpha_in_range : -Real.pi / 2 < alpha ∧ alpha < 0
axiom cos_alpha : Real.cos alpha = (Real.sqrt 5) / 5

theorem tan_alpha_minus_pi_over_4 : Real.tan (alpha - Real.pi / 4) = 3 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l1232_123246
