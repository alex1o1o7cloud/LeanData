import Mathlib

namespace NUMINAMATH_CALUDE_factory_bulb_supply_percentage_l1162_116270

theorem factory_bulb_supply_percentage 
  (prob_x : ℝ) 
  (prob_y : ℝ) 
  (prob_total : ℝ) 
  (h1 : prob_x = 0.59) 
  (h2 : prob_y = 0.65) 
  (h3 : prob_total = 0.62) : 
  ∃ (p : ℝ), p * prob_x + (1 - p) * prob_y = prob_total ∧ p = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_factory_bulb_supply_percentage_l1162_116270


namespace NUMINAMATH_CALUDE_doll_collection_increase_l1162_116230

theorem doll_collection_increase (original_count : ℕ) (increase_percentage : ℚ) (final_count : ℕ) 
  (h1 : increase_percentage = 25 / 100)
  (h2 : final_count = 10)
  (h3 : final_count = original_count + (increase_percentage * original_count).floor) :
  final_count - original_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_doll_collection_increase_l1162_116230


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1162_116292

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

-- Theorem statement
theorem min_value_of_sum (a : ℝ) :
  (∃ x, x = 2 ∧ f_derivative a x = 0) →
  (∀ m n, m ∈ Set.Icc (-1 : ℝ) 1 → n ∈ Set.Icc (-1 : ℝ) 1 →
    f a m + f_derivative a n ≥ -13) ∧
  (∃ m n, m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f_derivative a n = -13) :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_sum_l1162_116292


namespace NUMINAMATH_CALUDE_complex_number_calculation_l1162_116273

theorem complex_number_calculation : 
  (Complex.mk 1 3) * (Complex.mk 2 (-4)) + (Complex.mk 2 5) * (Complex.mk 2 (-1)) = Complex.mk 13 10 := by
sorry

end NUMINAMATH_CALUDE_complex_number_calculation_l1162_116273


namespace NUMINAMATH_CALUDE_min_occupied_seats_l1162_116241

/-- Represents the seating arrangement problem --/
def SeatingArrangement (total_seats : ℕ) (pattern : List ℕ) (occupied : ℕ) : Prop :=
  -- The total number of seats is 150
  total_seats = 150 ∧
  -- The pattern alternates between 4 and 3 empty seats
  pattern = [4, 3] ∧
  -- The occupied seats ensure the next person must sit next to someone
  occupied ≥ 
    -- Calculate the minimum number of occupied seats
    let full_units := total_seats / (pattern.sum + pattern.length)
    let remaining_seats := total_seats % (pattern.sum + pattern.length)
    let seats_in_full_units := full_units * pattern.length
    let additional_seats := if remaining_seats ≥ pattern.head! then 2 else 0
    seats_in_full_units + additional_seats

/-- The theorem stating the minimum number of occupied seats --/
theorem min_occupied_seats :
  ∃ (occupied : ℕ), SeatingArrangement 150 [4, 3] occupied ∧ occupied = 50 := by
  sorry

end NUMINAMATH_CALUDE_min_occupied_seats_l1162_116241


namespace NUMINAMATH_CALUDE_final_walnuts_count_l1162_116225

-- Define the initial conditions and actions
def initial_walnuts : ℕ := 25
def boy_gathered : ℕ := 15
def boy_dropped : ℕ := 3
def boy_hidden : ℕ := 5
def girl_brought : ℕ := 12
def girl_eaten : ℕ := 4
def girl_given : ℕ := 3
def girl_lost : ℕ := 2

-- Theorem to prove
theorem final_walnuts_count :
  initial_walnuts + 
  (boy_gathered - boy_dropped - boy_hidden) + 
  (girl_brought - girl_eaten - girl_given - girl_lost) = 35 := by
  sorry

end NUMINAMATH_CALUDE_final_walnuts_count_l1162_116225


namespace NUMINAMATH_CALUDE_juice_bar_solution_l1162_116276

/-- Represents the juice bar problem --/
def juice_bar_problem (total_spent : ℕ) (mango_price : ℕ) (pineapple_price : ℕ) (pineapple_spent : ℕ) : Prop :=
  ∃ (mango_count pineapple_count : ℕ),
    mango_count * mango_price + pineapple_count * pineapple_price = total_spent ∧
    pineapple_count * pineapple_price = pineapple_spent ∧
    mango_count + pineapple_count = 17

/-- The theorem stating the solution to the juice bar problem --/
theorem juice_bar_solution :
  juice_bar_problem 94 5 6 54 :=
sorry

end NUMINAMATH_CALUDE_juice_bar_solution_l1162_116276


namespace NUMINAMATH_CALUDE_ratio_from_equation_l1162_116260

theorem ratio_from_equation (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_from_equation_l1162_116260


namespace NUMINAMATH_CALUDE_bryce_raisins_l1162_116222

/-- Proves that Bryce received 12 raisins given the conditions of the problem -/
theorem bryce_raisins : 
  ∀ (bryce carter emma : ℕ), 
    bryce = carter + 8 →
    carter = bryce / 3 →
    emma = 2 * carter →
    bryce = 12 := by
  sorry

end NUMINAMATH_CALUDE_bryce_raisins_l1162_116222


namespace NUMINAMATH_CALUDE_price_difference_l1162_116289

/-- The original price of the toy rabbit -/
def original_price : ℝ := 25

/-- The price increase percentage for Store A -/
def increase_percentage : ℝ := 0.1

/-- The price decrease percentage for Store A -/
def decrease_percentage_A : ℝ := 0.2

/-- The price decrease percentage for Store B -/
def decrease_percentage_B : ℝ := 0.1

/-- The final price of the toy rabbit in Store A -/
def price_A : ℝ := original_price * (1 + increase_percentage) * (1 - decrease_percentage_A)

/-- The final price of the toy rabbit in Store B -/
def price_B : ℝ := original_price * (1 - decrease_percentage_B)

/-- Theorem stating that the price in Store A is 0.5 yuan less than in Store B -/
theorem price_difference : price_B - price_A = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_l1162_116289


namespace NUMINAMATH_CALUDE_existence_of_differences_l1162_116259

theorem existence_of_differences (n : ℕ) (x : Fin n → Fin n → ℚ)
  (h : ∀ (i j k : Fin n), x i j + x j k + x k i = 0) :
  ∃ (a : Fin n → ℚ), ∀ (i j : Fin n), x i j = a i - a j := by
  sorry

end NUMINAMATH_CALUDE_existence_of_differences_l1162_116259


namespace NUMINAMATH_CALUDE_yang_hui_field_theorem_l1162_116229

/-- Represents a rectangular field with given area and perimeter --/
structure RectangularField where
  area : ℕ
  perimeter : ℕ

/-- Calculates the difference between length and width of a rectangular field --/
def lengthWidthDifference (field : RectangularField) : ℕ :=
  let length := (field.perimeter + (field.perimeter^2 - 16 * field.area).sqrt) / 4
  let width := field.perimeter / 2 - length
  length - width

/-- Theorem stating the difference between length and width for the specific field --/
theorem yang_hui_field_theorem : 
  ∀ (field : RectangularField), 
  field.area = 864 ∧ field.perimeter = 120 → lengthWidthDifference field = 12 := by
  sorry

end NUMINAMATH_CALUDE_yang_hui_field_theorem_l1162_116229


namespace NUMINAMATH_CALUDE_cookie_pans_problem_l1162_116226

/-- Given a number of cookies per pan and a total number of cookies,
    calculate the number of pans needed. -/
def calculate_pans (cookies_per_pan : ℕ) (total_cookies : ℕ) : ℕ :=
  total_cookies / cookies_per_pan

theorem cookie_pans_problem :
  let cookies_per_pan : ℕ := 8
  let total_cookies : ℕ := 40
  calculate_pans cookies_per_pan total_cookies = 5 := by
  sorry

#eval calculate_pans 8 40

end NUMINAMATH_CALUDE_cookie_pans_problem_l1162_116226


namespace NUMINAMATH_CALUDE_inequality_solution_l1162_116205

theorem inequality_solution (x : ℝ) : 
  (x - 1)^2 < 12 - x ↔ (1 - 3 * Real.sqrt 5) / 2 < x ∧ x < (1 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1162_116205


namespace NUMINAMATH_CALUDE_board_crossing_area_l1162_116204

/-- The area of the parallelogram formed by two boards crossed at a 45-degree angle -/
theorem board_crossing_area (width1 width2 : ℝ) (angle : ℝ) : 
  width1 = 5 → width2 = 6 → angle = π/4 → 
  width2 * width1 = 30 := by sorry

end NUMINAMATH_CALUDE_board_crossing_area_l1162_116204


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1162_116283

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + x - 1 ≤ 0) ↔ (∃ x : ℝ, 2 * x^2 + x - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1162_116283


namespace NUMINAMATH_CALUDE_hyperbola_focus_k_value_l1162_116265

/-- Given a hyperbola with equation x^2 - ky^2 = 1 and one focus at (3,0), prove that k = 1/8 -/
theorem hyperbola_focus_k_value (k : ℝ) : 
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 - k*(y t)^2 = 1) →  -- Hyperbola equation
  (∃ (x₀ y₀ : ℝ), x₀^2 - k*y₀^2 = 1 ∧ x₀ = 3 ∧ y₀ = 0) →  -- Focus at (3,0)
  k = 1/8 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_k_value_l1162_116265


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l1162_116223

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 4*I) * (a + b*I) = y*I) : a/b = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l1162_116223


namespace NUMINAMATH_CALUDE_train_length_l1162_116244

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 12 → speed_kmh * (1000 / 3600) * time_s = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1162_116244


namespace NUMINAMATH_CALUDE_first_three_decimal_digits_l1162_116240

theorem first_three_decimal_digits (n : ℕ) (x : ℝ) : 
  n = 2003 → x = (10^n + 1)^(11/7) → 
  ∃ (y : ℝ), x = 10^2861 + y ∧ 0.571 < y/10^858 ∧ y/10^858 < 0.572 :=
by sorry

end NUMINAMATH_CALUDE_first_three_decimal_digits_l1162_116240


namespace NUMINAMATH_CALUDE_river_bank_bottom_width_l1162_116208

/-- Given a trapezium-shaped cross-section of a river bank, prove that the bottom width is 8 meters -/
theorem river_bank_bottom_width
  (top_width : ℝ)
  (depth : ℝ)
  (area : ℝ)
  (h1 : top_width = 12)
  (h2 : depth = 50)
  (h3 : area = 500)
  (h4 : area = (top_width + bottom_width) * depth / 2) :
  bottom_width = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_river_bank_bottom_width_l1162_116208


namespace NUMINAMATH_CALUDE_same_color_probability_l1162_116252

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def total_plates : ℕ := red_plates + blue_plates
def plates_selected : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_selected + Nat.choose blue_plates plates_selected) /
  Nat.choose total_plates plates_selected = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1162_116252


namespace NUMINAMATH_CALUDE_vector_subtraction_l1162_116239

/-- Given two complex numbers representing vectors OA and OB, 
    prove that the complex number representing vector AB is their difference. -/
theorem vector_subtraction (OA OB : ℂ) : 
  OA = 5 + 10*I → OB = 3 - 4*I → (OB - OA) = -2 - 14*I := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1162_116239


namespace NUMINAMATH_CALUDE_alternating_color_probability_l1162_116227

def box := {white : ℕ // white = 5} × {black : ℕ // black = 5}

def total_arrangements (b : box) : ℕ := Nat.choose (b.1 + b.2) b.1

def alternating_arrangements : ℕ := 2

theorem alternating_color_probability (b : box) :
  (alternating_arrangements : ℚ) / (total_arrangements b : ℚ) = 1 / 126 :=
sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l1162_116227


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l1162_116263

theorem cosine_sine_identity : 
  Real.cos (32 * π / 180) * Real.sin (62 * π / 180) - 
  Real.sin (32 * π / 180) * Real.sin (28 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l1162_116263


namespace NUMINAMATH_CALUDE_fraction_comparison_l1162_116274

theorem fraction_comparison : (1 / (Real.sqrt 5 - 2)) < (1 / (Real.sqrt 6 - Real.sqrt 5)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1162_116274


namespace NUMINAMATH_CALUDE_sachins_age_l1162_116297

theorem sachins_age (sachin rahul : ℕ) : 
  rahul = sachin + 18 →
  sachin * 9 = rahul * 7 →
  sachin = 63 := by sorry

end NUMINAMATH_CALUDE_sachins_age_l1162_116297


namespace NUMINAMATH_CALUDE_five_long_sides_l1162_116246

/-- A convex hexagon with specific properties -/
structure ConvexHexagon where
  -- The two distinct side lengths
  short_side : ℝ
  long_side : ℝ
  -- The number of sides with each length
  num_short_sides : ℕ
  num_long_sides : ℕ
  -- Properties
  is_convex : Bool
  distinct_lengths : short_side ≠ long_side
  total_sides : num_short_sides + num_long_sides = 6
  perimeter : num_short_sides * short_side + num_long_sides * long_side = 40
  short_side_length : short_side = 4
  long_side_length : long_side = 7

/-- Theorem: In a convex hexagon with the given properties, there are exactly 5 sides measuring 7 units -/
theorem five_long_sides (h : ConvexHexagon) : h.num_long_sides = 5 :=
  sorry

end NUMINAMATH_CALUDE_five_long_sides_l1162_116246


namespace NUMINAMATH_CALUDE_min_good_pairs_l1162_116278

/-- A circular arrangement of integers from 1 to 100 -/
def CircularArrangement := Fin 100 → ℕ

/-- Property that each number is either greater than both neighbors or less than both neighbors -/
def ValidArrangement (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 100, (arr i > arr (i - 1) ∧ arr i > arr (i + 1)) ∨ 
                 (arr i < arr (i - 1) ∧ arr i < arr (i + 1))

/-- Definition of a "good" pair -/
def GoodPair (arr : CircularArrangement) (i : Fin 100) : Prop :=
  ValidArrangement (Function.update (Function.update arr i (arr (i + 1))) (i + 1) (arr i))

/-- The main theorem stating that any valid arrangement has at least 51 good pairs -/
theorem min_good_pairs (arr : CircularArrangement) (h : ValidArrangement arr) :
  ∃ (s : Finset (Fin 100)), s.card ≥ 51 ∧ ∀ i ∈ s, GoodPair arr i :=
sorry

end NUMINAMATH_CALUDE_min_good_pairs_l1162_116278


namespace NUMINAMATH_CALUDE_jucas_marbles_l1162_116219

theorem jucas_marbles :
  ∃! B : ℕ, 0 < B ∧ B < 800 ∧
  B % 3 = 2 ∧
  B % 4 = 3 ∧
  B % 5 = 4 ∧
  B % 7 = 6 ∧
  B % 20 = 19 ∧
  B = 419 :=
by sorry

end NUMINAMATH_CALUDE_jucas_marbles_l1162_116219


namespace NUMINAMATH_CALUDE_perfect_squares_theorem_l1162_116277

theorem perfect_squares_theorem :
  -- Part 1: Infinitely many n such that 2n+1 and 3n+1 are perfect squares
  (∃ f : ℕ → ℤ, ∀ k, ∃ a b : ℤ, 2 * f k + 1 = a^2 ∧ 3 * f k + 1 = b^2) ∧
  -- Part 2: Such n are multiples of 40
  (∀ n : ℤ, (∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2) → ∃ k : ℤ, n = 40 * k) ∧
  -- Part 3: Generalization for any positive integer m
  (∀ m : ℕ, m > 0 →
    ∃ g : ℕ → ℤ, ∀ k, ∃ a b : ℤ, m * g k + 1 = a^2 ∧ (m + 1) * g k + 1 = b^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_theorem_l1162_116277


namespace NUMINAMATH_CALUDE_expression_evaluation_l1162_116243

theorem expression_evaluation : 
  let x : ℚ := 3
  let y : ℚ := -3
  (1/2 * x - 2 * (x - 1/3 * y^2) + (-3/2 * x + 1/3 * y^2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1162_116243


namespace NUMINAMATH_CALUDE_arc_length_ninety_degrees_radius_three_l1162_116266

/-- The arc length of a sector with a central angle of 90° and a radius of 3 is equal to (3/2)π. -/
theorem arc_length_ninety_degrees_radius_three :
  let central_angle : ℝ := 90
  let radius : ℝ := 3
  let arc_length : ℝ := (central_angle * π * radius) / 180
  arc_length = (3/2) * π := by sorry

end NUMINAMATH_CALUDE_arc_length_ninety_degrees_radius_three_l1162_116266


namespace NUMINAMATH_CALUDE_worker_payment_l1162_116298

/-- The daily wage in rupees -/
def daily_wage : ℚ := 20

/-- The number of days worked in a week -/
def days_worked : ℚ := 11/3 + 2/3 + 1/8 + 3/4

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem worker_payment :
  round_to_nearest (daily_wage * days_worked) = 104 := by
  sorry

end NUMINAMATH_CALUDE_worker_payment_l1162_116298


namespace NUMINAMATH_CALUDE_deepak_age_l1162_116299

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 22 →
  deepak_age = 12 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1162_116299


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l1162_116242

/-- The interval between segments in systematic sampling -/
def systematic_sampling_interval (N : ℕ) (n : ℕ) : ℕ :=
  N / n

/-- Theorem: For a population of 1500 and a sample size of 50, 
    the systematic sampling interval is 30 -/
theorem systematic_sampling_interval_example :
  systematic_sampling_interval 1500 50 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l1162_116242


namespace NUMINAMATH_CALUDE_ellipse_intercept_inequality_l1162_116295

-- Define the ellipse E
def E (m : ℝ) (x y : ℝ) : Prop := x^2 / m + y^2 / 4 = 1

-- Define the discriminant for the line y = kx + 1
def discriminant1 (m k : ℝ) : ℝ := 16 * m^2 * k^2 + 48 * m

-- Define the discriminant for the line kx + y - 2 = 0
def discriminant2 (m k : ℝ) : ℝ := 16 * m^2 * k^2

-- Theorem statement
theorem ellipse_intercept_inequality (m : ℝ) (h : m > 0) :
  ∀ k : ℝ, discriminant1 m k ≠ discriminant2 m k :=
sorry

end NUMINAMATH_CALUDE_ellipse_intercept_inequality_l1162_116295


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l1162_116236

theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 3003 → l + w + h ≥ 45 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l1162_116236


namespace NUMINAMATH_CALUDE_quadratic_value_theorem_l1162_116282

theorem quadratic_value_theorem (m : ℝ) (h : m^2 - 2*m - 1 = 0) :
  3*m^2 - 6*m + 2020 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_theorem_l1162_116282


namespace NUMINAMATH_CALUDE_cos_difference_angle_l1162_116203

theorem cos_difference_angle (α β : ℝ) : 
  Real.cos (α - β) = Real.cos α * Real.cos β + Real.sin α * Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_angle_l1162_116203


namespace NUMINAMATH_CALUDE_range_of_m_l1162_116257

def f (x : ℝ) : ℝ := sorry

theorem range_of_m (h1 : ∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≠ 0)
                   (h2 : ∀ x, f (-x) = -f x)
                   (h3 : ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ 2 → f x > f y)
                   (h4 : ∀ m, f (1 + m) + f m < 0) :
  ∀ m, (-1/2 < m ∧ m ≤ 1) ↔ (∃ x, -2 ≤ x ∧ x ≤ 2 ∧ f (1 + x) + f x < 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1162_116257


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l1162_116212

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
  a^3 + b^3 + c^3 = -36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l1162_116212


namespace NUMINAMATH_CALUDE_last_even_number_in_sequence_l1162_116268

theorem last_even_number_in_sequence (n : ℕ) : 
  (4 * (n * (n + 1) * (2 * n + 1)) / 6 = 560) → n = 7 :=
by sorry

end NUMINAMATH_CALUDE_last_even_number_in_sequence_l1162_116268


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1162_116215

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, -3)

theorem perpendicular_vectors (k : ℝ) : 
  (k • a - 2 • b) • a = 0 → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1162_116215


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1162_116249

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- Theorem: In an arithmetic sequence, if 3a₉ - a₁₅ - a₃ = 20, then 2a₈ - a₇ = 20 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) 
  (eq : 3 * a 9 - a 15 - a 3 = 20) : 
  2 * a 8 - a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1162_116249


namespace NUMINAMATH_CALUDE_egg_distribution_l1162_116202

theorem egg_distribution (total_eggs : Nat) (num_students : Nat) 
  (h1 : total_eggs = 73) (h2 : num_students = 9) :
  ∃ (eggs_per_student : Nat) (leftover : Nat),
    total_eggs = num_students * eggs_per_student + leftover ∧
    eggs_per_student = 8 ∧
    leftover = 1 := by
  sorry

end NUMINAMATH_CALUDE_egg_distribution_l1162_116202


namespace NUMINAMATH_CALUDE_value_of_m_l1162_116287

theorem value_of_m (m : ℝ) (h1 : m ≠ 0) 
  (h2 : ∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 12)) : 
  m = 12 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l1162_116287


namespace NUMINAMATH_CALUDE_james_room_area_l1162_116272

/-- Calculates the total area of rooms given initial dimensions and modifications --/
def total_area (initial_length initial_width increase : ℕ) : ℕ :=
  let new_length := initial_length + increase
  let new_width := initial_width + increase
  let single_room_area := new_length * new_width
  4 * single_room_area + 2 * single_room_area

/-- Theorem stating the total area for the given problem --/
theorem james_room_area :
  total_area 13 18 2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_james_room_area_l1162_116272


namespace NUMINAMATH_CALUDE_rhombus_longest_diagonal_l1162_116206

theorem rhombus_longest_diagonal (area : ℝ) (ratio_long : ℝ) (ratio_short : ℝ) :
  area = 108 →
  ratio_long = 3 →
  ratio_short = 2 →
  let diagonal_long := ratio_long * (2 * area / (ratio_long * ratio_short)) ^ (1/2 : ℝ)
  diagonal_long = 18 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longest_diagonal_l1162_116206


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1162_116237

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (α / 2) + Real.cos (α / 2) = Real.sqrt 6 / 2) : 
  Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1162_116237


namespace NUMINAMATH_CALUDE_leftover_sets_problem_l1162_116224

/-- Given a total number of crayons, number of friends, and crayons per set,
    calculate the number of complete sets left over after distributing one set to each friend. -/
def leftover_sets (total_crayons : ℕ) (num_friends : ℕ) (crayons_per_set : ℕ) : ℕ :=
  (total_crayons / crayons_per_set) % num_friends

theorem leftover_sets_problem :
  leftover_sets 210 30 5 = 12 := by
  sorry

#eval leftover_sets 210 30 5

end NUMINAMATH_CALUDE_leftover_sets_problem_l1162_116224


namespace NUMINAMATH_CALUDE_dice_sum_probability_l1162_116221

theorem dice_sum_probability (n : ℕ) : n = 36 →
  ∃ (d1 d2 : Finset ℕ),
    d1.card = 6 ∧ d2.card = 6 ∧
    (∀ k : ℕ, k ∈ Finset.range (n + 1) →
      (∃! (x y : ℕ), x ∈ d1 ∧ y ∈ d2 ∧ x + y = k)) :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_probability_l1162_116221


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l1162_116293

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = q * a n

theorem common_ratio_of_geometric_sequence (a : ℕ → ℚ) 
  (h_geometric : geometric_sequence a) 
  (h_a1 : a 1 = 1/8)
  (h_a4 : a 4 = -1) :
  ∃ q : ℚ, q = -2 ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
by
  sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l1162_116293


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1162_116247

theorem complex_equation_solution :
  ∀ (z : ℂ), z * (Complex.I - 1) = 2 * Complex.I → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1162_116247


namespace NUMINAMATH_CALUDE_pepper_spray_ratio_l1162_116209

theorem pepper_spray_ratio (total animals : ℕ) (raccoons : ℕ) : 
  total = 84 → raccoons = 12 → (total - raccoons) / raccoons = 6 := by
  sorry

end NUMINAMATH_CALUDE_pepper_spray_ratio_l1162_116209


namespace NUMINAMATH_CALUDE_num_chords_ten_points_l1162_116234

/-- The number of chords formed by connecting 2 points out of n points on a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- There are 10 points marked on the circumference of a circle -/
def num_points : ℕ := 10

/-- Theorem: The number of chords formed by connecting 2 points out of 10 points on a circle is 45 -/
theorem num_chords_ten_points : num_chords num_points = 45 := by
  sorry

end NUMINAMATH_CALUDE_num_chords_ten_points_l1162_116234


namespace NUMINAMATH_CALUDE_mias_christmas_gifts_l1162_116251

/-- Proves that the amount spent on each parent's gift is $30 -/
theorem mias_christmas_gifts (total_spent : ℕ) (sibling_gift : ℕ) (num_siblings : ℕ) :
  total_spent = 150 ∧ sibling_gift = 30 ∧ num_siblings = 3 →
  ∃ (parent_gift : ℕ), 
    parent_gift * 2 + sibling_gift * num_siblings = total_spent ∧
    parent_gift = 30 :=
by sorry

end NUMINAMATH_CALUDE_mias_christmas_gifts_l1162_116251


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1162_116250

theorem sqrt_equation_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ Real.sqrt x + 2 * Real.sqrt (x^2 + 7*x) + Real.sqrt (x + 7) = 35 - 2*x ∧ x = 841/144 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1162_116250


namespace NUMINAMATH_CALUDE_solution_set_equals_plus_minus_one_l1162_116210

def solution_set : Set ℝ := {x | x^2 - 1 = 0}

theorem solution_set_equals_plus_minus_one : solution_set = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equals_plus_minus_one_l1162_116210


namespace NUMINAMATH_CALUDE_tenth_configuration_stones_l1162_116218

/-- The number of stones in the n-th configuration of Anya's pentagon pattern -/
def stones (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- We define n = 0 as having no stones for completeness
  | 1 => 1
  | n + 1 => stones n + 3 * (n + 1) - 2

/-- The theorem stating that the 10th configuration has 145 stones -/
theorem tenth_configuration_stones :
  stones 10 = 145 := by
  sorry

/-- Helper lemma to show the first four configurations match the given values -/
lemma first_four_configurations :
  stones 1 = 1 ∧ stones 2 = 5 ∧ stones 3 = 12 ∧ stones 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_tenth_configuration_stones_l1162_116218


namespace NUMINAMATH_CALUDE_existence_of_composite_nx_plus_one_l1162_116281

theorem existence_of_composite_nx_plus_one (n : ℤ) : ∃ x : ℤ, ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ n * x + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_existence_of_composite_nx_plus_one_l1162_116281


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l1162_116213

theorem quadratic_function_minimum (a b c : ℝ) 
  (h1 : b > a) 
  (h2 : a > 0) 
  (h3 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) : 
  (a + b + c) / (b - a) ≥ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l1162_116213


namespace NUMINAMATH_CALUDE_total_vowels_written_l1162_116238

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 3

/-- Theorem: The total number of vowels written on the board is 15 -/
theorem total_vowels_written : num_vowels * times_written = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_vowels_written_l1162_116238


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_l1162_116269

theorem no_real_sqrt_negative : ∃ (a b c d : ℝ), 
  (a = (-3)^2 ∧ ∃ x : ℝ, x^2 = a) ∧ 
  (b = 0 ∧ ∃ x : ℝ, x^2 = b) ∧ 
  (c = 1/8 ∧ ∃ x : ℝ, x^2 = c) ∧ 
  (d = -6^3 ∧ ¬∃ x : ℝ, x^2 = d) :=
by sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_l1162_116269


namespace NUMINAMATH_CALUDE_second_number_value_l1162_116291

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 660 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a →
  b = 180 := by sorry

end NUMINAMATH_CALUDE_second_number_value_l1162_116291


namespace NUMINAMATH_CALUDE_slope_from_sin_cos_sum_l1162_116290

theorem slope_from_sin_cos_sum (θ : Real) 
  (h : Real.sin θ + Real.cos θ = Real.sqrt 5 / 5) : 
  Real.tan θ = -2 := by
  sorry

end NUMINAMATH_CALUDE_slope_from_sin_cos_sum_l1162_116290


namespace NUMINAMATH_CALUDE_factorization_proof_l1162_116261

theorem factorization_proof (a : ℝ) : 2 * a^2 - 2 * a + (1/2 : ℝ) = 2 * (a - 1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1162_116261


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l1162_116267

theorem polynomial_product_expansion (x : ℝ) :
  (3 * x^2 + 4) * (2 * x^3 + x^2 + 5) = 6 * x^5 + 3 * x^4 + 8 * x^3 + 19 * x^2 + 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l1162_116267


namespace NUMINAMATH_CALUDE_sticker_difference_l1162_116248

theorem sticker_difference (belle_stickers carolyn_stickers : ℕ) 
  (h1 : belle_stickers = 97)
  (h2 : carolyn_stickers = 79)
  (h3 : carolyn_stickers < belle_stickers) : 
  belle_stickers - carolyn_stickers = 18 := by
  sorry

end NUMINAMATH_CALUDE_sticker_difference_l1162_116248


namespace NUMINAMATH_CALUDE_candy_game_solution_l1162_116211

/-- 
Given a game where:
- 50 questions are asked
- Correct answers result in gaining 7 candies
- Incorrect answers result in losing 3 candies
- The net change in candies is zero

Prove that the number of correctly answered questions is 15.
-/
theorem candy_game_solution (total_questions : Nat) 
  (correct_reward : Nat) (incorrect_penalty : Nat) 
  (x : Nat) : 
  total_questions = 50 → 
  correct_reward = 7 → 
  incorrect_penalty = 3 → 
  x * correct_reward = (total_questions - x) * incorrect_penalty → 
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_candy_game_solution_l1162_116211


namespace NUMINAMATH_CALUDE_square_root_of_four_l1162_116232

theorem square_root_of_four :
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l1162_116232


namespace NUMINAMATH_CALUDE_marjs_wallet_after_purchase_l1162_116255

/-- The amount of money left in Marj's wallet after buying a cake -/
def money_left_in_wallet (twenty_bills : ℕ) (five_bills : ℕ) (loose_coins : ℚ) (cake_cost : ℚ) : ℚ :=
  (twenty_bills * 20 + five_bills * 5 : ℚ) + loose_coins - cake_cost

/-- Theorem stating the amount of money left in Marj's wallet -/
theorem marjs_wallet_after_purchase :
  money_left_in_wallet 2 3 4.5 17.5 = 42 := by sorry

end NUMINAMATH_CALUDE_marjs_wallet_after_purchase_l1162_116255


namespace NUMINAMATH_CALUDE_equation_solution_l1162_116264

theorem equation_solution : ∃ y : ℝ, y^4 - 20*y + 1 = 22 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1162_116264


namespace NUMINAMATH_CALUDE_exist_three_similar_non_congruent_triangles_l1162_116216

/-- A structure representing a triangle with side lengths and an angle -/
structure Triangle :=
  (a b c : ℝ)
  (angle_B : ℝ)

/-- Definition of similarity between two triangles -/
def similar (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.angle_B = t2.angle_B

/-- Definition of congruence between two triangles -/
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧ t1.angle_B = t2.angle_B

/-- Theorem stating the existence of three pairwise similar but non-congruent triangles -/
theorem exist_three_similar_non_congruent_triangles :
  ∃ (t1 t2 t3 : Triangle),
    similar t1 t2 ∧ similar t2 t3 ∧ similar t3 t1 ∧
    ¬congruent t1 t2 ∧ ¬congruent t2 t3 ∧ ¬congruent t3 t1 :=
by
  sorry

end NUMINAMATH_CALUDE_exist_three_similar_non_congruent_triangles_l1162_116216


namespace NUMINAMATH_CALUDE_product_not_divisible_by_sum_l1162_116286

theorem product_not_divisible_by_sum (a b : ℕ) (h : a + b = 201) : ¬(201 ∣ (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_product_not_divisible_by_sum_l1162_116286


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1162_116254

theorem units_digit_of_expression (k : ℕ) (h : k = 2012^2 + 2^2012) :
  (k^3 + 2^(k+1)) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1162_116254


namespace NUMINAMATH_CALUDE_exam_pass_count_l1162_116228

theorem exam_pass_count (total : ℕ) (overall_avg passed_avg failed_avg : ℚ) : 
  total = 120 →
  overall_avg = 35 →
  passed_avg = 39 →
  failed_avg = 15 →
  ∃ (passed : ℕ), 
    passed ≤ total ∧ 
    (passed : ℚ) * passed_avg + (total - passed : ℚ) * failed_avg = (total : ℚ) * overall_avg ∧
    passed = 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_count_l1162_116228


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l1162_116296

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Stratified
  | Systematic

/-- Represents a population with subgroups -/
structure Population where
  total : ℕ
  subgroups : List ℕ

/-- Represents a sampling scenario -/
structure SamplingScenario where
  population : Population
  sample_size : ℕ

/-- Determines the most appropriate sampling method for a given scenario -/
def most_appropriate_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- The student population -/
def student_population : Population :=
  { total := 10000, subgroups := [2000, 4500, 3500] }

/-- The product population -/
def product_population : Population :=
  { total := 1002, subgroups := [1002] }

/-- The student sampling scenario -/
def student_scenario : SamplingScenario :=
  { population := student_population, sample_size := 200 }

/-- The product sampling scenario -/
def product_scenario : SamplingScenario :=
  { population := product_population, sample_size := 20 }

theorem appropriate_sampling_methods :
  (most_appropriate_method student_scenario = SamplingMethod.Stratified) ∧
  (most_appropriate_method product_scenario = SamplingMethod.Systematic) :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l1162_116296


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_21_mod_30_l1162_116279

theorem smallest_five_digit_congruent_to_21_mod_30 :
  ∀ n : ℕ, 
    n ≥ 10000 ∧ n ≤ 99999 ∧ n ≡ 21 [MOD 30] → 
    n ≥ 10011 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_21_mod_30_l1162_116279


namespace NUMINAMATH_CALUDE_shooter_probabilities_l1162_116233

/-- The probability of hitting the target in a single shot -/
def hit_probability : ℝ := 0.9

/-- The number of shots -/
def num_shots : ℕ := 4

/-- The probability of hitting the target on the third shot -/
def third_shot_probability : ℝ := hit_probability

/-- The probability of hitting the target at least once in four shots -/
def at_least_one_hit_probability : ℝ := 1 - (1 - hit_probability) ^ num_shots

/-- The number of correct statements -/
def correct_statements : ℕ := 2

theorem shooter_probabilities :
  (third_shot_probability = hit_probability) ∧
  (at_least_one_hit_probability = 1 - (1 - hit_probability) ^ num_shots) ∧
  (correct_statements = 2) := by sorry

end NUMINAMATH_CALUDE_shooter_probabilities_l1162_116233


namespace NUMINAMATH_CALUDE_cards_added_l1162_116217

theorem cards_added (initial_cards : ℕ) (total_cards : ℕ) (added_cards : ℕ) : 
  initial_cards = 4 → total_cards = 7 → total_cards = initial_cards + added_cards → added_cards = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_added_l1162_116217


namespace NUMINAMATH_CALUDE_coal_supply_duration_l1162_116200

/-- 
Given a coal supply that was originally planned to last for a certain number of days 
with a specific daily consumption, and the actual daily consumption being a percentage 
less than planned, calculate the number of days the coal supply will actually last.
-/
theorem coal_supply_duration 
  (planned_daily_consumption : ℝ) 
  (planned_duration : ℝ) 
  (consumption_reduction_percentage : ℝ) : 
  planned_daily_consumption = 0.25 →
  planned_duration = 80 →
  consumption_reduction_percentage = 20 →
  (planned_daily_consumption * planned_duration) / 
  (planned_daily_consumption * (1 - consumption_reduction_percentage / 100)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_coal_supply_duration_l1162_116200


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_triangle_equality_condition_l1162_116288

/-- Represents a triangle with side lengths and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_area : 0 < S
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The inequality holds for any triangle -/
theorem triangle_inequality_theorem (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * Real.sqrt 3 * t.S :=
sorry

/-- The equality condition for the theorem -/
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- The equality holds if and only if the triangle is equilateral -/
theorem triangle_equality_condition (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 = 4 * Real.sqrt 3 * t.S ↔ is_equilateral t :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_triangle_equality_condition_l1162_116288


namespace NUMINAMATH_CALUDE_inequality_proof_l1162_116214

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1162_116214


namespace NUMINAMATH_CALUDE_total_votes_polled_l1162_116271

-- Define the total number of votes
variable (V : ℝ)

-- Define the number of votes for each candidate
variable (T S R F : ℝ)

-- Define the conditions
def condition1 : Prop := T = S + 0.15 * V
def condition2 : Prop := S = R + 0.05 * V
def condition3 : Prop := R = F + 0.07 * V
def condition4 : Prop := T + S + R + F = V
def condition5 : Prop := T - 2500 - 2000 = S + 2500
def condition6 : Prop := S + 2500 = R + 2000 + 0.05 * V

-- State the theorem
theorem total_votes_polled
  (h1 : condition1 V T S)
  (h2 : condition2 V S R)
  (h3 : condition3 V R F)
  (h4 : condition4 V T S R F)
  (h5 : condition5 T S)
  (h6 : condition6 V S R) :
  V = 30000 := by
  sorry


end NUMINAMATH_CALUDE_total_votes_polled_l1162_116271


namespace NUMINAMATH_CALUDE_simplify_fraction_expression_l1162_116220

theorem simplify_fraction_expression (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m / n - n / m) / (1 / m - 1 / n) = -m - n := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_expression_l1162_116220


namespace NUMINAMATH_CALUDE_unique_last_digit_for_multiple_of_6_l1162_116280

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def last_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem unique_last_digit_for_multiple_of_6 :
  ∃! d : ℕ, d < 10 ∧ is_multiple_of_6 (64310 + d) :=
by sorry

end NUMINAMATH_CALUDE_unique_last_digit_for_multiple_of_6_l1162_116280


namespace NUMINAMATH_CALUDE_ravi_jump_multiple_l1162_116275

def jump_heights : List ℝ := [23, 27, 28]
def ravi_jump : ℝ := 39

theorem ravi_jump_multiple :
  ravi_jump / (jump_heights.sum / jump_heights.length) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ravi_jump_multiple_l1162_116275


namespace NUMINAMATH_CALUDE_complex_cut_cube_edges_l1162_116285

/-- A cube with complex cuts at each vertex -/
structure ComplexCutCube where
  /-- The number of vertices in the original cube -/
  originalVertices : Nat
  /-- The number of edges in the original cube -/
  originalEdges : Nat
  /-- The number of cuts per vertex -/
  cutsPerVertex : Nat
  /-- The number of new edges introduced per vertex due to cuts -/
  newEdgesPerVertex : Nat

/-- Theorem stating that a cube with complex cuts results in 60 edges -/
theorem complex_cut_cube_edges (c : ComplexCutCube) 
  (h1 : c.originalVertices = 8)
  (h2 : c.originalEdges = 12)
  (h3 : c.cutsPerVertex = 2)
  (h4 : c.newEdgesPerVertex = 6) : 
  c.originalEdges + c.originalVertices * c.newEdgesPerVertex = 60 := by
  sorry

/-- The total number of edges in a complex cut cube is 60 -/
def total_edges (c : ComplexCutCube) : Nat :=
  c.originalEdges + c.originalVertices * c.newEdgesPerVertex

#check complex_cut_cube_edges

end NUMINAMATH_CALUDE_complex_cut_cube_edges_l1162_116285


namespace NUMINAMATH_CALUDE_marc_journey_fraction_l1162_116256

/-- Represents the time in minutes for a round trip journey -/
def roundTripTime (cyclingTime walkingTime : ℝ) : ℝ := cyclingTime + walkingTime

/-- Represents the time for Marc's modified journey -/
def modifiedJourneyTime (cyclingFraction : ℝ) : ℝ :=
  20 * cyclingFraction + 60 * (1 - cyclingFraction)

theorem marc_journey_fraction :
  ∃ (cyclingFraction : ℝ),
    roundTripTime 20 60 = 80 ∧
    modifiedJourneyTime cyclingFraction = 52 ∧
    cyclingFraction = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_marc_journey_fraction_l1162_116256


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1162_116253

theorem algebraic_simplification (a b : ℝ) :
  (2 * a^2 * b - 5 * a * b) - 2 * (-a * b + a^2 * b) = -3 * a * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1162_116253


namespace NUMINAMATH_CALUDE_staples_remaining_after_stapling_l1162_116231

/-- Calculates the number of staples left in a stapler after stapling reports. -/
def staples_left (initial_staples : ℕ) (reports_stapled : ℕ) : ℕ :=
  initial_staples - reports_stapled

/-- Converts dozens to individual units. -/
def dozens_to_units (dozens : ℕ) : ℕ :=
  dozens * 12

theorem staples_remaining_after_stapling :
  let initial_staples := 50
  let reports_in_dozens := 3
  let reports_stapled := dozens_to_units reports_in_dozens
  staples_left initial_staples reports_stapled = 14 := by
sorry

end NUMINAMATH_CALUDE_staples_remaining_after_stapling_l1162_116231


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_plus_2x_l1162_116258

theorem factorization_of_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_plus_2x_l1162_116258


namespace NUMINAMATH_CALUDE_perfect_square_base8_l1162_116245

/-- Represents a number in base 8 of the form ab3c -/
structure Base8Number where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_valid : b < 8
  c_valid : c < 8

/-- Converts a Base8Number to its decimal representation -/
def toDecimal (n : Base8Number) : Nat :=
  512 * n.a + 64 * n.b + 24 + n.c

theorem perfect_square_base8 (n : Base8Number) :
  (∃ m : Nat, toDecimal n = m * m) → n.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_base8_l1162_116245


namespace NUMINAMATH_CALUDE_product_properties_l1162_116207

-- Define a function to count trailing zeros
def trailingZeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + trailingZeros (n / 10)
  else 0

theorem product_properties :
  (trailingZeros (360 * 5) = 2) ∧ (250 * 4 = 1000) := by
  sorry

end NUMINAMATH_CALUDE_product_properties_l1162_116207


namespace NUMINAMATH_CALUDE_base_5_to_binary_44_l1162_116235

def base_5_to_decimal (n : ℕ) : ℕ := 
  (n / 10) * 5 + (n % 10)

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem base_5_to_binary_44 :
  decimal_to_binary (base_5_to_decimal 44) = [1, 1, 0, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_base_5_to_binary_44_l1162_116235


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1162_116201

/-- The total capacity of a water tank in gallons -/
def tank_capacity : ℝ → Prop := λ T =>
  -- When the tank is 40% full, it contains 36 gallons less than when it is 70% full
  0.7 * T - 0.4 * T = 36

theorem water_tank_capacity : ∃ T : ℝ, tank_capacity T ∧ T = 120 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1162_116201


namespace NUMINAMATH_CALUDE_triangle_inequality_for_positive_reals_l1162_116262

theorem triangle_inequality_for_positive_reals (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  Real.sqrt (a^2 + b^2) ≤ a + b := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_for_positive_reals_l1162_116262


namespace NUMINAMATH_CALUDE_f_is_even_l1162_116294

def f (x : ℝ) : ℝ := (x + 2)^2 + (2*x - 1)^2

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1162_116294


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l1162_116284

theorem gcd_of_squares_sum : Nat.gcd (130^2 + 241^2 + 352^2) (129^2 + 240^2 + 353^2 + 2^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l1162_116284
