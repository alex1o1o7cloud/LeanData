import Mathlib

namespace NUMINAMATH_GPT_problem_l482_48258

-- Definitions and conditions
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ (∀ n, 2 ≤ n → 2 * a n / (a n * (Finset.sum (Finset.range n) a) - (Finset.sum (Finset.range n) a) ^ 2) = 1)

-- Sum of the first n terms
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := Finset.sum (Finset.range n) a

-- The proof statement
theorem problem (a : ℕ → ℚ) (h : seq a) : S a 2017 = 1 / 1009 := sorry

end NUMINAMATH_GPT_problem_l482_48258


namespace NUMINAMATH_GPT_younger_by_17_l482_48260

variables (A B C : ℕ)

-- Given condition
axiom age_condition : A + B = B + C + 17

-- To show
theorem younger_by_17 : A - C = 17 :=
by
  sorry

end NUMINAMATH_GPT_younger_by_17_l482_48260


namespace NUMINAMATH_GPT_circle_tangent_lines_l482_48256

theorem circle_tangent_lines (h k : ℝ) (r : ℝ) (h_gt_10 : h > 10) (k_gt_10 : k > 10)
  (tangent_y_eq_10 : k - 10 = r)
  (tangent_y_eq_x : r = (|h - k| / Real.sqrt 2)) :
  (h, k) = (10 + (1 + Real.sqrt 2) * r, 10 + r) :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_lines_l482_48256


namespace NUMINAMATH_GPT_tan_of_angle_in_fourth_quadrant_l482_48225

theorem tan_of_angle_in_fourth_quadrant (α : ℝ) (h1 : Real.sin α = -5 / 13) (h2 : α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) :
  Real.tan α = -5 / 12 :=
sorry

end NUMINAMATH_GPT_tan_of_angle_in_fourth_quadrant_l482_48225


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l482_48200

-- Formal statement of the problem in Lean 4
theorem number_of_ordered_pairs : 
  ∃ (n : ℕ), n = 128 ∧ 
  ∀ (a b : ℝ), (∃ (x y : ℤ), (a * x + b * y = 1) ∧ (x^2 + y^2 = 65)) ↔ n = 128 :=
sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l482_48200


namespace NUMINAMATH_GPT_magpies_gather_7_trees_magpies_not_gather_6_trees_l482_48229

-- Define the problem conditions.
def trees (n : ℕ) := (∀ (i : ℕ), i < n → ∃ (m : ℕ), m = i * 10)

-- Define the movement condition for magpies.
def magpie_move (n : ℕ) (d : ℕ) :=
  (∀ (i j : ℕ), i < n ∧ j < n ∧ i ≠ j → ∃ (k : ℕ), k = d ∧ ((i + d < n ∧ j - d < n) ∨ (i - d < n ∧ j + d < n)))

-- Prove that all magpies can gather on one tree for 7 trees.
theorem magpies_gather_7_trees : 
  ∃ (i : ℕ), i < 7 ∧ trees 7 ∧ magpie_move 7 (i * 10) → True :=
by
  -- proof steps here, which are not necessary for the task
  sorry

-- Prove that all magpies cannot gather on one tree for 6 trees.
theorem magpies_not_gather_6_trees : 
  ∀ (i : ℕ), i < 6 ∧ trees 6 ∧ magpie_move 6 (i * 10) → False :=
by
  -- proof steps here, which are not necessary for the task
  sorry

end NUMINAMATH_GPT_magpies_gather_7_trees_magpies_not_gather_6_trees_l482_48229


namespace NUMINAMATH_GPT_alice_weight_l482_48214

theorem alice_weight (a c : ℝ) (h1 : a + c = 200) (h2 : a - c = a / 3) : a = 120 :=
by
  sorry

end NUMINAMATH_GPT_alice_weight_l482_48214


namespace NUMINAMATH_GPT_cube_of_prism_volume_l482_48230

theorem cube_of_prism_volume (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y) * (y * z) * (z * x) = (x * y * z)^3 :=
by
  sorry

end NUMINAMATH_GPT_cube_of_prism_volume_l482_48230


namespace NUMINAMATH_GPT_cranberries_left_l482_48255

def initial_cranberries : ℕ := 60000
def harvested_by_humans : ℕ := initial_cranberries * 40 / 100
def eaten_by_elk : ℕ := 20000

theorem cranberries_left (c : ℕ) : c = initial_cranberries - harvested_by_humans - eaten_by_elk → c = 16000 := by
  sorry

end NUMINAMATH_GPT_cranberries_left_l482_48255


namespace NUMINAMATH_GPT_third_dog_average_daily_miles_l482_48285

/-- Bingo has three dogs. On average, they walk a total of 100 miles a week.

    The first dog walks an average of 2 miles a day.

    The second dog walks 1 mile if it is an odd day of the month and 3 miles if it is an even day of the month.

    Considering a 30-day month, the goal is to find the average daily miles of the third dog. -/
theorem third_dog_average_daily_miles :
  let total_dogs := 3
  let weekly_total_miles := 100
  let first_dog_daily_miles := 2
  let second_dog_odd_day_miles := 1
  let second_dog_even_day_miles := 3
  let days_in_month := 30
  let odd_days_in_month := 15
  let even_days_in_month := 15
  let weeks_in_month := days_in_month / 7
  let first_dog_monthly_miles := days_in_month * first_dog_daily_miles
  let second_dog_monthly_miles := (second_dog_odd_day_miles * odd_days_in_month) + (second_dog_even_day_miles * even_days_in_month)
  let third_dog_monthly_miles := (weekly_total_miles * weeks_in_month) - (first_dog_monthly_miles + second_dog_monthly_miles)
  let third_dog_daily_miles := third_dog_monthly_miles / days_in_month
  third_dog_daily_miles = 10.33 :=
by
  sorry

end NUMINAMATH_GPT_third_dog_average_daily_miles_l482_48285


namespace NUMINAMATH_GPT_comparison_of_powers_l482_48259

theorem comparison_of_powers : 6 ^ 0.7 > 0.7 ^ 6 ∧ 0.7 ^ 6 > 0.6 ^ 7 := by
  sorry

end NUMINAMATH_GPT_comparison_of_powers_l482_48259


namespace NUMINAMATH_GPT_sufficient_not_necessary_for_ellipse_l482_48205

-- Define the conditions
def positive_denominator_m (m : ℝ) : Prop := m > 0
def positive_denominator_2m_minus_1 (m : ℝ) : Prop := 2 * m - 1 > 0
def denominators_not_equal (m : ℝ) : Prop := m ≠ 1

-- Define the question
def is_ellipse_condition (m : ℝ) : Prop := m > 1

-- The main theorem
theorem sufficient_not_necessary_for_ellipse (m : ℝ) :
  positive_denominator_m m ∧ positive_denominator_2m_minus_1 m ∧ denominators_not_equal m → is_ellipse_condition m :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_for_ellipse_l482_48205


namespace NUMINAMATH_GPT_parallelogram_area_l482_48201

theorem parallelogram_area (b h : ℝ) (hb : b = 20) (hh : h = 4) : b * h = 80 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l482_48201


namespace NUMINAMATH_GPT_boat_travel_distance_downstream_l482_48272

-- Define the conditions given in the problem
def speed_boat_still_water := 22 -- in km/hr
def speed_stream := 5 -- in km/hr
def time_downstream := 2 -- in hours

-- Define a function to compute the effective speed downstream
def effective_speed_downstream (speed_boat: ℝ) (speed_stream: ℝ) : ℝ :=
  speed_boat + speed_stream

-- Define a function to compute the distance travelled downstream
def distance_downstream (speed: ℝ) (time: ℝ) : ℝ :=
  speed * time

-- The main theorem to prove
theorem boat_travel_distance_downstream :
  distance_downstream (effective_speed_downstream speed_boat_still_water speed_stream) time_downstream = 54 :=
by
  -- Proof is to be filled in later
  sorry

end NUMINAMATH_GPT_boat_travel_distance_downstream_l482_48272


namespace NUMINAMATH_GPT_percentage_increase_l482_48275

theorem percentage_increase
  (W R : ℝ)
  (H1 : 0.70 * R = 1.04999999999999982 * W) :
  (R - W) / W * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l482_48275


namespace NUMINAMATH_GPT_total_weight_of_rice_l482_48295

theorem total_weight_of_rice :
  (29 * 4) / 16 = 7.25 := by
sorry

end NUMINAMATH_GPT_total_weight_of_rice_l482_48295


namespace NUMINAMATH_GPT_cos_value_l482_48247

theorem cos_value (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (4 * π / 3 - α) = -3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_cos_value_l482_48247


namespace NUMINAMATH_GPT_Kiran_money_l482_48294

theorem Kiran_money (R G K : ℕ) (h1 : R / G = 6 / 7) (h2 : G / K = 6 / 15) (h3 : R = 36) : K = 105 := by
  sorry

end NUMINAMATH_GPT_Kiran_money_l482_48294


namespace NUMINAMATH_GPT_parabola_translation_l482_48240

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2 - 2

-- Define the transformation: translate one unit to the right
def translate_right (x : ℝ) : ℝ := initial_parabola (x - 1)

-- Define the transformation: move up three units
def move_up (y : ℝ) : ℝ := y + 3

-- Define the resulting equation after the transformations
def resulting_parabola (x : ℝ) : ℝ := move_up (translate_right x)

-- Define the target equation
def target_parabola (x : ℝ) : ℝ := (x - 1)^2 + 1

-- Formalize the proof problem
theorem parabola_translation :
  ∀ x : ℝ, resulting_parabola x = target_parabola x :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_parabola_translation_l482_48240


namespace NUMINAMATH_GPT_solution_set_of_inequality_l482_48212

theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = f x) (h2 : ∀ x, 0 ≤ x → f x = x - 1) :
  { x : ℝ | f (x - 1) > 1 } = { x | x < -1 ∨ x > 3 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l482_48212


namespace NUMINAMATH_GPT_numberOfValidFiveDigitNumbers_l482_48278

namespace MathProof

def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def isDivisibleBy5 (n : ℕ) : Prop := n % 5 = 0

def firstAndLastDigitsEqual (n : ℕ) : Prop := 
  let firstDigit := (n / 10000) % 10
  let lastDigit := n % 10
  firstDigit = lastDigit

def sumOfDigitsDivisibleBy5 (n : ℕ) : Prop := 
  let d1 := (n / 10000) % 10
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  (d1 + d2 + d3 + d4 + d5) % 5 = 0

theorem numberOfValidFiveDigitNumbers :
  ∃ (count : ℕ), count = 200 ∧ 
  count = Nat.card {n : ℕ // isFiveDigitNumber n ∧ 
                                isDivisibleBy5 n ∧ 
                                firstAndLastDigitsEqual n ∧ 
                                sumOfDigitsDivisibleBy5 n} :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_numberOfValidFiveDigitNumbers_l482_48278


namespace NUMINAMATH_GPT_part1_part2_l482_48208

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -1)
noncomputable def vec_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2)
noncomputable def f (x : ℝ) : ℝ := (vec_m x).1 * (vec_n x).1 + (vec_m x).2 * (vec_n x).2 + 1

-- Part 1
theorem part1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) (hf : f x = 11 / 10) : 
  x = (Real.pi / 6) + Real.arcsin (3 / 5) :=
sorry

-- Part 2
theorem part2 {A B C a b c : ℝ} 
  (hABC : A + B + C = Real.pi) 
  (habc : 2 * b * Real.cos A ≤ 2 * c - Real.sqrt 3 * a) : 
  (0 < B ∧ B ≤ Real.pi / 6) → 
  ∃ y, (0 < y ∧ y ≤ 1 / 2 ∧ f B = y) :=
sorry

end NUMINAMATH_GPT_part1_part2_l482_48208


namespace NUMINAMATH_GPT_intersection_complement_eq_l482_48261

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := { x | x < 0 }

def B : Set ℝ := { x | x ≤ -1 }

theorem intersection_complement_eq : A ∩ (U \ B) = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l482_48261


namespace NUMINAMATH_GPT_subset_eq_possible_sets_of_B_l482_48232

theorem subset_eq_possible_sets_of_B (B : Set ℕ) 
  (h1 : {1, 2} ⊆ B)
  (h2 : B ⊆ {1, 2, 3, 4}) :
  B = {1, 2} ∨ B = {1, 2, 3} ∨ B = {1, 2, 4} :=
sorry

end NUMINAMATH_GPT_subset_eq_possible_sets_of_B_l482_48232


namespace NUMINAMATH_GPT_cone_bead_path_l482_48298

theorem cone_bead_path (r h : ℝ) (h_sqrt : h / r = 3 * Real.sqrt 11) : 3 + 11 = 14 := by
  sorry

end NUMINAMATH_GPT_cone_bead_path_l482_48298


namespace NUMINAMATH_GPT_time_reading_per_week_l482_48253

-- Define the given conditions
def time_meditating_per_day : ℕ := 1
def time_reading_per_day : ℕ := 2 * time_meditating_per_day
def days_in_week : ℕ := 7

-- Define the target property to prove
theorem time_reading_per_week : time_reading_per_day * days_in_week = 14 :=
by
  sorry

end NUMINAMATH_GPT_time_reading_per_week_l482_48253


namespace NUMINAMATH_GPT_has_two_zeros_of_f_l482_48277

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.exp x - a

theorem has_two_zeros_of_f (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (-1 / Real.exp 2 < a ∧ a < 0) := by
sorry

end NUMINAMATH_GPT_has_two_zeros_of_f_l482_48277


namespace NUMINAMATH_GPT_mr_william_land_percentage_l482_48241

/--
Given:
1. Farm tax is levied on 90% of the cultivated land.
2. The tax department collected a total of $3840 through the farm tax from the village.
3. Mr. William paid $480 as farm tax.

Prove: The percentage of total land of Mr. William over the total taxable land of the village is 12.5%.
-/
theorem mr_william_land_percentage (T W : ℝ) 
  (h1 : 0.9 * W = 480) 
  (h2 : 0.9 * T = 3840) : 
  (W / T) * 100 = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_mr_william_land_percentage_l482_48241


namespace NUMINAMATH_GPT_factorization_correct_l482_48243

theorem factorization_correct :
  ∀ (x y : ℝ), 
    (¬ ( (y - 1) * (y + 1) = y^2 - 1 ) ) ∧
    (¬ ( x^2 * y + x * y^2 - 1 = x * y * (x + y) - 1 ) ) ∧
    (¬ ( (x - 2) * (x - 3) = (3 - x) * (2 - x) ) ) ∧
    ( x^2 - 4 * x + 4 = (x - 2)^2 ) :=
by
  intros x y
  repeat { constructor }
  all_goals { sorry }

end NUMINAMATH_GPT_factorization_correct_l482_48243


namespace NUMINAMATH_GPT_walnut_trees_in_park_l482_48224

def num_current_walnut_trees (num_plant : ℕ) (num_total : ℕ) : ℕ :=
  num_total - num_plant

theorem walnut_trees_in_park :
  num_current_walnut_trees 6 10 = 4 :=
by
  -- By the definition of num_current_walnut_trees
  -- We have 10 (total) - 6 (to be planted) = 4 (current)
  sorry

end NUMINAMATH_GPT_walnut_trees_in_park_l482_48224


namespace NUMINAMATH_GPT_problem_statement_l482_48288

noncomputable def decimalPartSqrtFive : ℝ := Real.sqrt 5 - 2
def integerPartSqrtThirteen : ℕ := 3

theorem problem_statement :
  decimalPartSqrtFive + integerPartSqrtThirteen - Real.sqrt 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l482_48288


namespace NUMINAMATH_GPT_richard_twice_as_old_as_scott_in_8_years_l482_48209

theorem richard_twice_as_old_as_scott_in_8_years :
  (richard_age - david_age = 6) ∧ (david_age - scott_age = 8) ∧ (david_age = 14) →
  (richard_age + 8 = 2 * (scott_age + 8)) :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end NUMINAMATH_GPT_richard_twice_as_old_as_scott_in_8_years_l482_48209


namespace NUMINAMATH_GPT_emily_annual_holidays_l482_48246

theorem emily_annual_holidays 
    (holidays_per_month : ℕ) 
    (months_in_year : ℕ) 
    (h1: holidays_per_month = 2)
    (h2: months_in_year = 12)
    : holidays_per_month * months_in_year = 24 := 
by
  sorry

end NUMINAMATH_GPT_emily_annual_holidays_l482_48246


namespace NUMINAMATH_GPT_sixty_percent_of_40_greater_than_four_fifths_of_25_l482_48270

theorem sixty_percent_of_40_greater_than_four_fifths_of_25 :
  let x := (60 / 100 : ℝ) * 40
  let y := (4 / 5 : ℝ) * 25
  x - y = 4 :=
by
  sorry

end NUMINAMATH_GPT_sixty_percent_of_40_greater_than_four_fifths_of_25_l482_48270


namespace NUMINAMATH_GPT_john_reads_days_per_week_l482_48287

-- Define the conditions
def john_reads_books_per_day := 4
def total_books_read := 48
def total_weeks := 6

-- Theorem statement
theorem john_reads_days_per_week :
  (total_books_read / john_reads_books_per_day) / total_weeks = 2 :=
by
  sorry

end NUMINAMATH_GPT_john_reads_days_per_week_l482_48287


namespace NUMINAMATH_GPT_compute_expression_l482_48227

theorem compute_expression : 2 + 8 * 3 - 4 + 6 * 5 / 2 - 3 ^ 2 = 28 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l482_48227


namespace NUMINAMATH_GPT_percentage_error_in_area_l482_48222

theorem percentage_error_in_area (s : ℝ) (x : ℝ) (h₁ : s' = 1.08 * s) 
  (h₂ : s^2 = (2 * A)) (h₃ : x^2 = (2 * A)) : 
  (abs ((1.1664 * s^2 - s^2) / s^2 * 100) - 17) ≤ 0.5 := 
sorry

end NUMINAMATH_GPT_percentage_error_in_area_l482_48222


namespace NUMINAMATH_GPT_inequality_solution_set_l482_48237

theorem inequality_solution_set (a b c : ℝ)
  (h1 : ∀ x, (ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, ¬ (bx + c > 0 ↔ x > 6)) ∧ (∀ x, (cx^2 + bx + a < 0 ↔ -1/3 < x ∧ x < 1/2)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l482_48237


namespace NUMINAMATH_GPT_coloring_integers_l482_48267

theorem coloring_integers 
  (color : ℤ → ℕ) 
  (x y : ℤ) 
  (hx : x % 2 = 1) 
  (hy : y % 2 = 1) 
  (h_neq : |x| ≠ |y|) 
  (h_color_range : ∀ n : ℤ, color n < 4) :
  ∃ a b : ℤ, color a = color b ∧ (a - b = x ∨ a - b = y ∨ a - b = x + y ∨ a - b = x - y) :=
sorry

end NUMINAMATH_GPT_coloring_integers_l482_48267


namespace NUMINAMATH_GPT_tangent_line_coordinates_l482_48220

theorem tangent_line_coordinates :
  ∃ x₀ : ℝ, ∃ y₀ : ℝ, (x₀ = 1 ∧ y₀ = Real.exp 1) ∧
  (∀ x : ℝ, ∀ y : ℝ, y = Real.exp x → ∃ m : ℝ, 
    (m = Real.exp 1 ∧ (y - y₀ = m * (x - x₀))) ∧
    (0 - y₀ = m * (0 - x₀))) := sorry

end NUMINAMATH_GPT_tangent_line_coordinates_l482_48220


namespace NUMINAMATH_GPT_arithmetic_sequence_z_value_l482_48276

theorem arithmetic_sequence_z_value :
  ∃ z : ℤ, (3 ^ 2 = 9 ∧ 3 ^ 4 = 81) ∧ z = (9 + 81) / 2 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_z_value_l482_48276


namespace NUMINAMATH_GPT_largest_whole_number_l482_48223

theorem largest_whole_number (n : ℤ) (h : (1 : ℝ) / 4 + n / 8 < 2) : n ≤ 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_whole_number_l482_48223


namespace NUMINAMATH_GPT_price_per_glass_first_day_l482_48274

theorem price_per_glass_first_day
    (O W : ℝ) (P1 P2 : ℝ)
    (h1 : O = W)
    (h2 : P2 = 0.40)
    (h3 : 2 * O * P1 = 3 * O * P2) :
    P1 = 0.60 :=
by
    sorry

end NUMINAMATH_GPT_price_per_glass_first_day_l482_48274


namespace NUMINAMATH_GPT_xyz_expression_l482_48283

theorem xyz_expression (x y z : ℝ) 
  (h1 : x^2 - y * z = 2)
  (h2 : y^2 - z * x = 2)
  (h3 : z^2 - x * y = 2) :
  x * y + y * z + z * x = -2 :=
sorry

end NUMINAMATH_GPT_xyz_expression_l482_48283


namespace NUMINAMATH_GPT_siblings_of_kevin_l482_48286

-- Define traits of each child
structure Child where
  eye_color : String
  hair_color : String

def Oliver : Child := ⟨"Green", "Red"⟩
def Kevin : Child := ⟨"Grey", "Brown"⟩
def Lily : Child := ⟨"Grey", "Red"⟩
def Emma : Child := ⟨"Green", "Brown"⟩
def Noah : Child := ⟨"Green", "Red"⟩
def Mia : Child := ⟨"Green", "Brown"⟩

-- Define the condition that siblings must share at least one trait
def share_at_least_one_trait (c1 c2 : Child) : Prop :=
  c1.eye_color = c2.eye_color ∨ c1.hair_color = c2.hair_color

-- Prove that Emma and Mia are Kevin's siblings
theorem siblings_of_kevin : share_at_least_one_trait Kevin Emma ∧ share_at_least_one_trait Kevin Mia ∧ share_at_least_one_trait Emma Mia :=
  sorry

end NUMINAMATH_GPT_siblings_of_kevin_l482_48286


namespace NUMINAMATH_GPT_rectangle_perimeter_l482_48233

theorem rectangle_perimeter (t s : ℝ) (h : t ≥ s) : 2 * (t - s) + 2 * s = 2 * t := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l482_48233


namespace NUMINAMATH_GPT_hall_length_l482_48268

theorem hall_length
  (width : ℝ)
  (stone_length : ℝ)
  (stone_width : ℝ)
  (num_stones : ℕ)
  (h₁ : width = 15)
  (h₂ : stone_length = 0.8)
  (h₃ : stone_width = 0.5)
  (h₄ : num_stones = 1350) :
  ∃ length : ℝ, length = 36 :=
by
  sorry

end NUMINAMATH_GPT_hall_length_l482_48268


namespace NUMINAMATH_GPT_number_of_true_propositions_l482_48264

-- Define the original condition
def original_proposition (a b : ℝ) : Prop := (a + b = 1) → (a * b ≤ 1 / 4)

-- Define contrapositive
def contrapositive (a b : ℝ) : Prop := (a * b > 1 / 4) → (a + b ≠ 1)

-- Define inverse
def inverse (a b : ℝ) : Prop := (a * b ≤ 1 / 4) → (a + b = 1)

-- Define converse
def converse (a b : ℝ) : Prop := (a + b ≠ 1) → (a * b > 1 / 4)

-- State the problem
theorem number_of_true_propositions (a b : ℝ) :
  (original_proposition a b ∧ contrapositive a b ∧ ¬inverse a b ∧ ¬converse a b) → 
  (∃ n : ℕ, n = 1) :=
by sorry

end NUMINAMATH_GPT_number_of_true_propositions_l482_48264


namespace NUMINAMATH_GPT_brianne_savings_ratio_l482_48284

theorem brianne_savings_ratio
  (r : ℝ)
  (H1 : 10 * r^4 = 160) :
  r = 2 :=
by 
  sorry

end NUMINAMATH_GPT_brianne_savings_ratio_l482_48284


namespace NUMINAMATH_GPT_call_processing_ratio_l482_48292

variables (A B C : ℝ)
variable (total_calls : ℝ)
variable (calls_processed_by_A_per_member calls_processed_by_B_per_member : ℝ)

-- Given conditions
def team_A_agents_ratio : Prop := A = (5 / 8) * B
def team_B_calls_ratio : Prop := calls_processed_by_B_per_member * B = (4 / 7) * total_calls
def team_A_calls_ratio : Prop := calls_processed_by_A_per_member * A = (3 / 7) * total_calls

-- Proving the ratio of calls processed by each member
theorem call_processing_ratio
    (hA : team_A_agents_ratio A B)
    (hB_calls : team_B_calls_ratio B total_calls calls_processed_by_B_per_member)
    (hA_calls : team_A_calls_ratio A total_calls calls_processed_by_A_per_member) :
  calls_processed_by_A_per_member / calls_processed_by_B_per_member = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_call_processing_ratio_l482_48292


namespace NUMINAMATH_GPT_sum_of_possible_values_l482_48244

theorem sum_of_possible_values (x : ℝ) (h : (x + 3) * (x - 5) = 20) : x = -2 ∨ x = 7 :=
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l482_48244


namespace NUMINAMATH_GPT_exists_pentagon_from_midpoints_l482_48219

noncomputable def pentagon_from_midpoints (A1 B1 C1 D1 E1 : ℝ × ℝ) : Prop :=
  ∃ (A B C D E : ℝ × ℝ), 
    (A1 = (A + B) / 2) ∧ 
    (B1 = (B + C) / 2) ∧ 
    (C1 = (C + D) / 2) ∧ 
    (D1 = (D + E) / 2) ∧ 
    (E1 = (E + A) / 2)

-- statement of the theorem
theorem exists_pentagon_from_midpoints (A1 B1 C1 D1 E1 : ℝ × ℝ) :
  pentagon_from_midpoints A1 B1 C1 D1 E1 :=
sorry

end NUMINAMATH_GPT_exists_pentagon_from_midpoints_l482_48219


namespace NUMINAMATH_GPT_maximum_regular_hours_is_40_l482_48228

-- Definitions based on conditions
def regular_pay_per_hour := 3
def overtime_pay_per_hour := 6
def total_payment_received := 168
def overtime_hours := 8
def overtime_earnings := overtime_hours * overtime_pay_per_hour
def regular_earnings := total_payment_received - overtime_earnings
def maximum_regular_hours := regular_earnings / regular_pay_per_hour

-- Lean theorem statement corresponding to the proof problem
theorem maximum_regular_hours_is_40 : maximum_regular_hours = 40 := by
  sorry

end NUMINAMATH_GPT_maximum_regular_hours_is_40_l482_48228


namespace NUMINAMATH_GPT_playground_width_l482_48293

open Nat

theorem playground_width (garden_width playground_length perimeter_garden : ℕ) (garden_area_eq_playground_area : Bool) :
  garden_width = 8 →
  playground_length = 16 →
  perimeter_garden = 64 →
  garden_area_eq_playground_area →
  ∃ (W : ℕ), W = 12 :=
by
  intros h_t1 h_t2 h_t3 h_t4
  sorry

end NUMINAMATH_GPT_playground_width_l482_48293


namespace NUMINAMATH_GPT_reckha_code_count_l482_48234

theorem reckha_code_count :
  let total_codes := 1000
  let codes_with_one_digit_different := 27
  let permutations_of_045 := 2
  let original_code := 1
  total_codes - codes_with_one_digit_different - permutations_of_045 - original_code = 970 :=
by
  let total_codes := 1000
  let codes_with_one_digit_different := 27
  let permutations_of_045 := 2
  let original_code := 1
  show total_codes - codes_with_one_digit_different - permutations_of_045 - original_code = 970
  sorry

end NUMINAMATH_GPT_reckha_code_count_l482_48234


namespace NUMINAMATH_GPT_most_convincing_method_l482_48231

-- Defining the survey data
def male_participants : Nat := 4258
def male_believe_doping : Nat := 2360
def female_participants : Nat := 3890
def female_believe_framed : Nat := 2386

-- Defining the question-to-answer equivalence related to the most convincing method
theorem most_convincing_method :
  "Independence Test" = "Independence Test" := 
by
  sorry

end NUMINAMATH_GPT_most_convincing_method_l482_48231


namespace NUMINAMATH_GPT_Jamie_needs_to_climb_40_rungs_l482_48291

-- Define the conditions
def height_of_new_tree : ℕ := 20
def rungs_climbed_previous : ℕ := 12
def height_of_previous_tree : ℕ := 6
def rungs_per_foot := rungs_climbed_previous / height_of_previous_tree

-- Define the theorem
theorem Jamie_needs_to_climb_40_rungs :
  height_of_new_tree * rungs_per_foot = 40 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_Jamie_needs_to_climb_40_rungs_l482_48291


namespace NUMINAMATH_GPT_fewer_onions_grown_l482_48299

def num_tomatoes := 2073
def num_cobs_of_corn := 4112
def num_onions := 985

theorem fewer_onions_grown : num_tomatoes + num_cobs_of_corn - num_onions = 5200 := by
  sorry

end NUMINAMATH_GPT_fewer_onions_grown_l482_48299


namespace NUMINAMATH_GPT_horizontal_length_tv_screen_l482_48257

theorem horizontal_length_tv_screen : 
  ∀ (a b : ℝ), (a / b = 4 / 3) → (a ^ 2 + b ^ 2 = 27 ^ 2) → a = 21.5 := 
by 
  sorry

end NUMINAMATH_GPT_horizontal_length_tv_screen_l482_48257


namespace NUMINAMATH_GPT_points_calculation_correct_l482_48207

-- Definitions
def points_per_enemy : ℕ := 9
def total_enemies : ℕ := 11
def enemies_undestroyed : ℕ := 3
def enemies_destroyed : ℕ := total_enemies - enemies_undestroyed

def points_earned : ℕ := enemies_destroyed * points_per_enemy

-- Theorem statement
theorem points_calculation_correct : points_earned = 72 := by
  sorry

end NUMINAMATH_GPT_points_calculation_correct_l482_48207


namespace NUMINAMATH_GPT_NOQZ_has_same_product_as_MNOQ_l482_48250

/-- Each letter of the alphabet is assigned a value (A=1, B=2, C=3, ..., Z=26). -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13
  | 'N' => 14 | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19
  | 'T' => 20 | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _   => 0  -- We'll assume only uppercase letters are inputs

/-- The product of a four-letter list is the product of the values of its four letters. -/
def list_product (lst : List Char) : ℕ :=
  lst.map letter_value |>.foldl (· * ·) 1

/-- The product of the list MNOQ is calculated. -/
def product_MNOQ : ℕ := list_product ['M', 'N', 'O', 'Q']
/-- The product of the list BEHK is calculated. -/
def product_BEHK : ℕ := list_product ['B', 'E', 'H', 'K']
/-- The product of the list NOQZ is calculated. -/
def product_NOQZ : ℕ := list_product ['N', 'O', 'Q', 'Z']

theorem NOQZ_has_same_product_as_MNOQ :
  product_NOQZ = product_MNOQ := by
  sorry

end NUMINAMATH_GPT_NOQZ_has_same_product_as_MNOQ_l482_48250


namespace NUMINAMATH_GPT_deposit_percentage_l482_48282

noncomputable def last_year_cost : ℝ := 250
noncomputable def increase_percentage : ℝ := 0.40
noncomputable def amount_paid_at_pickup : ℝ := 315
noncomputable def total_cost := last_year_cost * (1 + increase_percentage)
noncomputable def deposit := total_cost - amount_paid_at_pickup
noncomputable def percentage_deposit := deposit / total_cost * 100

theorem deposit_percentage :
  percentage_deposit = 10 := 
  by
    sorry

end NUMINAMATH_GPT_deposit_percentage_l482_48282


namespace NUMINAMATH_GPT_calculateRemainingMoney_l482_48281

def initialAmount : ℝ := 100
def actionFiguresCount : ℕ := 3
def actionFigureOriginalPrice : ℝ := 12
def actionFigureDiscount : ℝ := 0.25
def boardGamesCount : ℕ := 2
def boardGamePrice : ℝ := 11
def puzzleSetsCount : ℕ := 4
def puzzleSetPrice : ℝ := 6
def salesTax : ℝ := 0.05

theorem calculateRemainingMoney :
  initialAmount - (
    (actionFigureOriginalPrice * (1 - actionFigureDiscount) * actionFiguresCount) +
    (boardGamePrice * boardGamesCount) +
    (puzzleSetPrice * puzzleSetsCount)
  ) * (1 + salesTax) = 23.35 :=
by
  sorry

end NUMINAMATH_GPT_calculateRemainingMoney_l482_48281


namespace NUMINAMATH_GPT_largest_mersenne_prime_less_than_500_l482_48273

def mersenne_prime (n : ℕ) : ℕ := 2^n - 1

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem largest_mersenne_prime_less_than_500 :
  ∃ n, is_prime n ∧ mersenne_prime n < 500 ∧ ∀ m, is_prime m ∧ mersenne_prime m < 500 → mersenne_prime m ≤ mersenne_prime n :=
  sorry

end NUMINAMATH_GPT_largest_mersenne_prime_less_than_500_l482_48273


namespace NUMINAMATH_GPT_physics_marks_l482_48202

theorem physics_marks
  (P C M : ℕ)
  (h1 : P + C + M = 240)
  (h2 : P + M = 180)
  (h3 : P + C = 140) :
  P = 80 :=
by
  sorry

end NUMINAMATH_GPT_physics_marks_l482_48202


namespace NUMINAMATH_GPT_petrov_vasechkin_boards_l482_48263

theorem petrov_vasechkin_boards:
  ∃ n : ℕ, 
  (∃ x y : ℕ, 2 * x + 3 * y = 87 ∧ x + y = n) ∧ 
  (∃ u v : ℕ, 3 * u + 5 * v = 94 ∧ u + v = n) ∧ 
  n = 30 := 
sorry

end NUMINAMATH_GPT_petrov_vasechkin_boards_l482_48263


namespace NUMINAMATH_GPT_garden_length_is_60_l482_48242

noncomputable def garden_length (w l : ℕ) : Prop :=
  l = 2 * w ∧ 2 * w + 2 * l = 180

theorem garden_length_is_60 (w l : ℕ) (h : garden_length w l) : l = 60 :=
by
  sorry

end NUMINAMATH_GPT_garden_length_is_60_l482_48242


namespace NUMINAMATH_GPT_total_marbles_l482_48211

variable (b : ℝ)
variable (r : ℝ) (g : ℝ)
variable (h₁ : r = 1.3 * b)
variable (h₂ : g = 1.5 * b)

theorem total_marbles (b : ℝ) (r : ℝ) (g : ℝ) (h₁ : r = 1.3 * b) (h₂ : g = 1.5 * b) : r + b + g = 3.8 * b :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_l482_48211


namespace NUMINAMATH_GPT_pure_imaginary_solution_l482_48217

theorem pure_imaginary_solution (a : ℝ) (ha : a + 5 * Complex.I / (1 - 2 * Complex.I) = a + (1 : ℂ) * Complex.I) :
  a = 2 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_solution_l482_48217


namespace NUMINAMATH_GPT_sum_remainders_mod_13_l482_48290

theorem sum_remainders_mod_13 :
  ∀ (a b c d e : ℕ),
  a % 13 = 3 →
  b % 13 = 5 →
  c % 13 = 7 →
  d % 13 = 9 →
  e % 13 = 11 →
  (a + b + c + d + e) % 13 = 9 :=
by
  intros a b c d e ha hb hc hd he
  sorry

end NUMINAMATH_GPT_sum_remainders_mod_13_l482_48290


namespace NUMINAMATH_GPT_probability_of_selecting_one_is_correct_l482_48297

-- Define the number of elements in the first 20 rows of Pascal's triangle
def totalElementsInPascalFirst20Rows : ℕ := 210

-- Define the number of ones in the first 20 rows of Pascal's triangle
def totalOnesInPascalFirst20Rows : ℕ := 39

-- The probability as a rational number
def probabilityOfSelectingOne : ℚ := totalOnesInPascalFirst20Rows / totalElementsInPascalFirst20Rows

theorem probability_of_selecting_one_is_correct :
  probabilityOfSelectingOne = 13 / 70 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_probability_of_selecting_one_is_correct_l482_48297


namespace NUMINAMATH_GPT_find_two_digit_number_t_l482_48251

theorem find_two_digit_number_t (t : ℕ) (ht1 : 10 ≤ t) (ht2 : t ≤ 99) (ht3 : 13 * t % 100 = 52) : t = 12 := 
sorry

end NUMINAMATH_GPT_find_two_digit_number_t_l482_48251


namespace NUMINAMATH_GPT_prime_pairs_perfect_square_l482_48266

theorem prime_pairs_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ k : ℕ, p^(q-1) + q^(p-1) = k^2 ↔ (p = 2 ∧ q = 2) :=
by
  sorry

end NUMINAMATH_GPT_prime_pairs_perfect_square_l482_48266


namespace NUMINAMATH_GPT_correct_answer_l482_48280

variable (x : ℝ)

theorem correct_answer : {x : ℝ | x^2 + 2*x + 1 = 0} = {-1} :=
by sorry -- the actual proof is not required, just the statement

end NUMINAMATH_GPT_correct_answer_l482_48280


namespace NUMINAMATH_GPT_speed_of_A_is_3_l482_48226

theorem speed_of_A_is_3:
  (∃ x : ℝ, 3 * x + 3 * (x + 2) = 24) → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_A_is_3_l482_48226


namespace NUMINAMATH_GPT_weights_less_than_90_l482_48265

variable (a b c : ℝ)
-- conditions
axiom h1 : a + b = 100
axiom h2 : a + c = 101
axiom h3 : b + c = 102

theorem weights_less_than_90 (a b c : ℝ) (h1 : a + b = 100) (h2 : a + c = 101) (h3 : b + c = 102) : a < 90 ∧ b < 90 ∧ c < 90 := 
by sorry

end NUMINAMATH_GPT_weights_less_than_90_l482_48265


namespace NUMINAMATH_GPT_negation_of_existential_square_inequality_l482_48296

theorem negation_of_existential_square_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_square_inequality_l482_48296


namespace NUMINAMATH_GPT_quarters_value_percentage_l482_48279

theorem quarters_value_percentage (dimes_count quarters_count dimes_value quarters_value : ℕ) (h1 : dimes_count = 75)
    (h2 : quarters_count = 30) (h3 : dimes_value = 10) (h4 : quarters_value = 25) :
    (quarters_count * quarters_value * 100) / (dimes_count * dimes_value + quarters_count * quarters_value) = 50 := 
by
    sorry

end NUMINAMATH_GPT_quarters_value_percentage_l482_48279


namespace NUMINAMATH_GPT_money_last_weeks_l482_48236

-- Define the conditions
def dollars_mowing : ℕ := 68
def dollars_weed_eating : ℕ := 13
def dollars_per_week : ℕ := 9

-- Define the total money made
def total_dollars := dollars_mowing + dollars_weed_eating

-- State the theorem to prove the question
theorem money_last_weeks : (total_dollars / dollars_per_week) = 9 :=
by
  sorry

end NUMINAMATH_GPT_money_last_weeks_l482_48236


namespace NUMINAMATH_GPT_expression_evaluation_l482_48203

theorem expression_evaluation (m : ℝ) (h : m = Real.sqrt 2023 + 2) : m^2 - 4 * m + 5 = 2024 :=
by sorry

end NUMINAMATH_GPT_expression_evaluation_l482_48203


namespace NUMINAMATH_GPT_shaded_area_l482_48249

-- Definitions and conditions from the problem
def Square1Side := 4 -- in inches
def Square2Side := 12 -- in inches
def Triangle_DGF_similar_to_Triangle_AHF : Prop := (4 / 12) = (3 / 16)

theorem shaded_area
  (h1 : Square1Side = 4)
  (h2 : Square2Side = 12)
  (h3 : Triangle_DGF_similar_to_Triangle_AHF) :
  ∃ shaded_area : ℕ, shaded_area = 10 :=
by
  -- Calculation steps here
  sorry

end NUMINAMATH_GPT_shaded_area_l482_48249


namespace NUMINAMATH_GPT_T_description_l482_48238

-- Definitions of conditions
def T (x y : ℝ) : Prop :=
  (x + 3 = 4 ∧ y ≤ 9) ∨
  (y - 5 = 4 ∧ x ≤ 1) ∨
  (x + 3 = y - 5 ∧ x ≥ 1)

-- The problem statement in Lean: Prove that T describes three rays with a common point (1, 9)
theorem T_description :
  ∀ x y, T x y ↔ 
    ((x = 1 ∧ y ≤ 9) ∨
     (x ≤ 1 ∧ y = 9) ∨
     (x ≥ 1 ∧ y = x + 8)) :=
by sorry

end NUMINAMATH_GPT_T_description_l482_48238


namespace NUMINAMATH_GPT_x_plus_p_l482_48271

theorem x_plus_p (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2 * p + 3 :=
by
  sorry

end NUMINAMATH_GPT_x_plus_p_l482_48271


namespace NUMINAMATH_GPT_sum_of_geometric_sequence_l482_48213

theorem sum_of_geometric_sequence :
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 3280 / 6561 :=
by
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  sorry

end NUMINAMATH_GPT_sum_of_geometric_sequence_l482_48213


namespace NUMINAMATH_GPT_value_of_a_l482_48248

theorem value_of_a (a : ℝ) (H1 : A = a) (H2 : B = 1) (H3 : C = a - 3) (H4 : C + B = 0) : a = 2 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l482_48248


namespace NUMINAMATH_GPT_find_equidistant_point_l482_48235

theorem find_equidistant_point :
  ∃ (x z : ℝ),
    ((x - 1)^2 + 4^2 + z^2 = (x - 2)^2 + 2^2 + (z - 3)^2) ∧
    ((x - 1)^2 + 4^2 + z^2 = (x - 3)^2 + 9 + (z + 2)^2) ∧
    (x + 2 * z = 5) ∧
    (x = 15 / 8) ∧
    (z = 5 / 8) :=
by
  sorry

end NUMINAMATH_GPT_find_equidistant_point_l482_48235


namespace NUMINAMATH_GPT_flour_in_cupboard_l482_48289

theorem flour_in_cupboard :
  let flour_on_counter := 100
  let flour_in_pantry := 100
  let flour_per_loaf := 200
  let loaves := 2
  let total_flour_needed := loaves * flour_per_loaf
  let flour_outside_cupboard := flour_on_counter + flour_in_pantry
  let flour_in_cupboard := total_flour_needed - flour_outside_cupboard
  flour_in_cupboard = 200 :=
by
  sorry

end NUMINAMATH_GPT_flour_in_cupboard_l482_48289


namespace NUMINAMATH_GPT_number_of_outfits_l482_48204

-- Define the counts of each item
def redShirts : Nat := 6
def greenShirts : Nat := 4
def pants : Nat := 7
def greenHats : Nat := 10
def redHats : Nat := 9

-- Total number of outfits satisfying the conditions
theorem number_of_outfits :
  (redShirts * greenHats * pants) + (greenShirts * redHats * pants) = 672 :=
by
  sorry

end NUMINAMATH_GPT_number_of_outfits_l482_48204


namespace NUMINAMATH_GPT_y1_less_than_y2_l482_48252

noncomputable def y1 : ℝ := 2 * (-5) + 1
noncomputable def y2 : ℝ := 2 * 3 + 1

theorem y1_less_than_y2 : y1 < y2 := by
  sorry

end NUMINAMATH_GPT_y1_less_than_y2_l482_48252


namespace NUMINAMATH_GPT_complement_intersection_l482_48254

open Set

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 2, 5}

theorem complement_intersection :
  ((U \ A) ∩ B) = {1, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l482_48254


namespace NUMINAMATH_GPT_find_a_given_conditions_l482_48269

theorem find_a_given_conditions (a : ℤ)
  (hA : ∃ (x : ℤ), x = 12 ∨ x = a^2 + 4 * a ∨ x = a - 2)
  (hA_contains_minus3 : ∃ (x : ℤ), (-3 = x) ∧ (x = 12 ∨ x = a^2 + 4 * a ∨ x = a - 2)) : a = -3 := 
by
  sorry

end NUMINAMATH_GPT_find_a_given_conditions_l482_48269


namespace NUMINAMATH_GPT_probability_X_interval_l482_48218

noncomputable def fx (x c : ℝ) : ℝ :=
  if -c ≤ x ∧ x ≤ c then (1 / c) * (1 - (|x| / c))
  else 0

theorem probability_X_interval (c : ℝ) (hc : 0 < c) :
  (∫ x in (c / 2)..c, fx x c) = 1 / 8 :=
sorry

end NUMINAMATH_GPT_probability_X_interval_l482_48218


namespace NUMINAMATH_GPT_april_rainfall_correct_l482_48245

-- Define the constants for the rainfalls in March and the difference in April
def march_rainfall : ℝ := 0.81
def rain_difference : ℝ := 0.35

-- Define the expected April rainfall based on the conditions
def april_rainfall : ℝ := march_rainfall - rain_difference

-- Theorem to prove that the April rainfall is 0.46 inches
theorem april_rainfall_correct : april_rainfall = 0.46 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_april_rainfall_correct_l482_48245


namespace NUMINAMATH_GPT_num_of_triangles_with_perimeter_10_l482_48206

theorem num_of_triangles_with_perimeter_10 :
  ∃ (triangles : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → 
      a + b + c = 10 ∧ 
      a + b > c ∧ 
      a + c > b ∧ 
      b + c > a) ∧ 
    triangles.card = 4 := sorry

end NUMINAMATH_GPT_num_of_triangles_with_perimeter_10_l482_48206


namespace NUMINAMATH_GPT_average_power_heater_l482_48215

structure Conditions where
  (M : ℝ)    -- mass of the piston
  (tau : ℝ)  -- time period τ
  (a : ℝ)    -- constant acceleration
  (c : ℝ)    -- specific heat at constant volume
  (R : ℝ)    -- universal gas constant

theorem average_power_heater (cond : Conditions) : 
  let P := cond.M * cond.a^2 * cond.tau / 2 * (1 + cond.c / cond.R)
  P = (cond.M * cond.a^2 * cond.tau / 2) * (1 + cond.c / cond.R) :=
by
  sorry

end NUMINAMATH_GPT_average_power_heater_l482_48215


namespace NUMINAMATH_GPT_final_temperature_l482_48210

theorem final_temperature (initial_temp cost_per_tree spent amount temperature_drop : ℝ) 
  (h1 : initial_temp = 80) 
  (h2 : cost_per_tree = 6)
  (h3 : spent = 108) 
  (h4 : temperature_drop = 0.1) 
  (trees_planted : ℝ) 
  (h5 : trees_planted = spent / cost_per_tree) 
  (temp_reduction : ℝ) 
  (h6 : temp_reduction = trees_planted * temperature_drop) 
  (final_temp : ℝ) 
  (h7 : final_temp = initial_temp - temp_reduction) : 
  final_temp = 78.2 := 
by
  sorry

end NUMINAMATH_GPT_final_temperature_l482_48210


namespace NUMINAMATH_GPT_polynomial_divisibility_l482_48221

theorem polynomial_divisibility (a b x y : ℤ) : 
  ∃ k : ℤ, (a * x + b * y)^3 + (b * x + a * y)^3 = k * (a + b) * (x + y) := by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l482_48221


namespace NUMINAMATH_GPT_translate_line_down_l482_48216

theorem translate_line_down (k : ℝ) (b : ℝ) : 
  (∀ x : ℝ, b = 0 → (y = k * x - 3) = (y = k * x - 3)) :=
by
  sorry

end NUMINAMATH_GPT_translate_line_down_l482_48216


namespace NUMINAMATH_GPT_rectangle_length_l482_48262

theorem rectangle_length (P W : ℝ) (hP : P = 40) (hW : W = 8) : ∃ L : ℝ, 2 * (L + W) = P ∧ L = 12 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_length_l482_48262


namespace NUMINAMATH_GPT_find_square_value_l482_48239

variable (a b : ℝ)
variable (square : ℝ)

-- Conditions: Given the equation square * 3 * a = -3 * a^2 * b
axiom condition : square * 3 * a = -3 * a^2 * b

-- Theorem: Prove that square = -a * b
theorem find_square_value (a b : ℝ) (square : ℝ) (h : square * 3 * a = -3 * a^2 * b) : 
    square = -a * b :=
by
  exact sorry

end NUMINAMATH_GPT_find_square_value_l482_48239
