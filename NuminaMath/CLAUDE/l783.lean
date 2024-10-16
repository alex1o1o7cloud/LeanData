import Mathlib

namespace NUMINAMATH_CALUDE_chip_notebook_usage_l783_78300

/-- Calculates the number of packs of notebook paper used by Chip over a given number of weeks. -/
def notebook_packs_used (pages_per_day_per_class : ℕ) (classes : ℕ) (days_per_week : ℕ) 
  (weeks : ℕ) (sheets_per_pack : ℕ) : ℕ :=
  let total_pages := pages_per_day_per_class * classes * days_per_week * weeks
  (total_pages + sheets_per_pack - 1) / sheets_per_pack

/-- Proves that Chip uses 3 packs of notebook paper after 6 weeks. -/
theorem chip_notebook_usage : 
  notebook_packs_used 2 5 5 6 100 = 3 := by
  sorry

end NUMINAMATH_CALUDE_chip_notebook_usage_l783_78300


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l783_78363

/-- A quadratic function that intersects the x-axis at (-1, 0) and (2, 0), and the y-axis at (0, -2) -/
def f (x : ℝ) : ℝ := x^2 - x - 2

/-- The theorem stating that f is the unique quadratic function satisfying the given conditions -/
theorem quadratic_function_unique :
  (f (-1) = 0) ∧ 
  (f 2 = 0) ∧ 
  (f 0 = -2) ∧ 
  (∀ x : ℝ, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧
  (∀ g : ℝ → ℝ, (g (-1) = 0) → (g 2 = 0) → (g 0 = -2) → 
    (∀ x : ℝ, ∃ a b c : ℝ, g x = a * x^2 + b * x + c) → 
    (∀ x : ℝ, g x = f x)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l783_78363


namespace NUMINAMATH_CALUDE_quadratic_zero_range_l783_78322

theorem quadratic_zero_range (a : ℝ) : 
  let f := fun x : ℝ => x^2 + (a^2 - 1)*x + (a - 2)
  (∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0) ∧ x₁ < 1 ∧ 1 < x₂) ↔ -2 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_zero_range_l783_78322


namespace NUMINAMATH_CALUDE_inequality_proof_l783_78353

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 2) * (y^2 + 2) * (z^2 + 2) ≥ 9 * (x*y + y*z + z*x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l783_78353


namespace NUMINAMATH_CALUDE_hexagon_angle_Q_l783_78314

/-- A hexagon with specified interior angles -/
structure Hexagon :=
  (angleS : ℝ)
  (angleT : ℝ)
  (angleU : ℝ)
  (angleV : ℝ)
  (angleW : ℝ)
  (h_angleS : angleS = 120)
  (h_angleT : angleT = 130)
  (h_angleU : angleU = 140)
  (h_angleV : angleV = 100)
  (h_angleW : angleW = 85)

/-- The measure of angle Q in the hexagon -/
def angleQ (h : Hexagon) : ℝ := 720 - (h.angleS + h.angleT + h.angleU + h.angleV + h.angleW)

theorem hexagon_angle_Q (h : Hexagon) : angleQ h = 145 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_Q_l783_78314


namespace NUMINAMATH_CALUDE_c_months_is_six_l783_78344

/-- Represents the rental scenario for a pasture -/
structure PastureRental where
  total_rent : ℕ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  b_payment : ℕ

/-- Calculates the number of months c put in the horses -/
def calculate_c_months (rental : PastureRental) : ℕ :=
  sorry

/-- Theorem stating that c put in the horses for 6 months -/
theorem c_months_is_six (rental : PastureRental)
  (h1 : rental.total_rent = 870)
  (h2 : rental.a_horses = 12)
  (h3 : rental.a_months = 8)
  (h4 : rental.b_horses = 16)
  (h5 : rental.b_months = 9)
  (h6 : rental.c_horses = 18)
  (h7 : rental.b_payment = 360) :
  calculate_c_months rental = 6 :=
sorry

end NUMINAMATH_CALUDE_c_months_is_six_l783_78344


namespace NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l783_78348

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units is 48π square units. -/
theorem circle_area_equilateral_triangle : 
  ∀ (s : ℝ) (A : ℝ),
  s = 12 →  -- Side length of the equilateral triangle
  A = π * (s / Real.sqrt 3)^2 →  -- Area formula for circumscribed circle
  A = 48 * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l783_78348


namespace NUMINAMATH_CALUDE_original_alcohol_percentage_l783_78365

/-- Proves that given a 15-liter mixture of alcohol and water, if adding 3 liters of water
    results in a new mixture with 20.833333333333336% alcohol, then the original mixture
    contained 25% alcohol. -/
theorem original_alcohol_percentage
  (original_volume : ℝ)
  (added_water : ℝ)
  (new_alcohol_percentage : ℝ)
  (h1 : original_volume = 15)
  (h2 : added_water = 3)
  (h3 : new_alcohol_percentage = 20.833333333333336)
  : ∃ (original_alcohol_percentage : ℝ),
    original_alcohol_percentage = 25 ∧
    (original_alcohol_percentage / 100) * original_volume =
    (new_alcohol_percentage / 100) * (original_volume + added_water) :=
by sorry

end NUMINAMATH_CALUDE_original_alcohol_percentage_l783_78365


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l783_78316

theorem orthogonal_vectors (x y : ℝ) : 
  (3 * x + 4 * (-2) = 0 ∧ 3 * 1 + 4 * y = 0) ↔ (x = 8/3 ∧ y = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l783_78316


namespace NUMINAMATH_CALUDE_gcd_56_63_l783_78302

theorem gcd_56_63 : Nat.gcd 56 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_56_63_l783_78302


namespace NUMINAMATH_CALUDE_joan_dimes_l783_78334

/-- The number of dimes Joan has after spending some -/
def remaining_dimes (initial : ℕ) (spent : ℕ) : ℕ := initial - spent

/-- Theorem: If Joan had 5 dimes initially and spent 2 dimes, she now has 3 dimes -/
theorem joan_dimes : remaining_dimes 5 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_joan_dimes_l783_78334


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l783_78368

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b - a * b = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y - x * y = 0 → a + 2 * b ≤ x + 2 * y :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l783_78368


namespace NUMINAMATH_CALUDE_nice_sequence_classification_l783_78338

/-- A sequence of integers -/
def IntegerSequence := ℕ → ℤ

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- A nice sequence satisfies the given condition for some function f -/
def IsNice (a : IntegerSequence) : Prop :=
  ∃ f : PositiveIntFunction, ∀ i j n : ℕ+,
    (a i.val - a j.val) % n.val = 0 ↔ (i.val - j.val) % f n = 0

/-- A sequence is periodic with period k -/
def IsPeriodic (a : IntegerSequence) (k : ℕ+) : Prop :=
  ∀ i : ℕ, a (i + k) = a i

/-- A sequence is an arithmetic sequence -/
def IsArithmetic (a : IntegerSequence) : Prop :=
  ∃ d : ℤ, ∀ i : ℕ, a (i + 1) = a i + d

/-- The main theorem: nice sequences are either constant, periodic with period 2, or arithmetic -/
theorem nice_sequence_classification (a : IntegerSequence) :
  IsNice a → (IsPeriodic a 1 ∨ IsPeriodic a 2 ∨ IsArithmetic a) :=
sorry

end NUMINAMATH_CALUDE_nice_sequence_classification_l783_78338


namespace NUMINAMATH_CALUDE_wall_length_height_ratio_l783_78374

/-- Represents the dimensions and volume of a rectangular wall. -/
structure Wall where
  breadth : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

/-- Theorem stating the ratio of length to height for a specific wall. -/
theorem wall_length_height_ratio (w : Wall) 
  (h_volume : w.volume = 12.8)
  (h_breadth : w.breadth = 0.4)
  (h_height : w.height = 5 * w.breadth)
  (h_volume_calc : w.volume = w.breadth * w.height * w.length) :
  w.length / w.height = 4 := by
  sorry

#check wall_length_height_ratio

end NUMINAMATH_CALUDE_wall_length_height_ratio_l783_78374


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l783_78327

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (2/5 : ℂ) + (1/3 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (2/5 : ℂ) - (1/3 : ℂ) * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l783_78327


namespace NUMINAMATH_CALUDE_pages_left_to_write_l783_78398

/-- Calculates the remaining pages to write given the daily page counts and total book length -/
theorem pages_left_to_write (total_pages day1 day2 day3 day4 day5 : ℝ) : 
  total_pages = 750 →
  day1 = 30 →
  day2 = 1.5 * day1 →
  day3 = 0.5 * day2 →
  day4 = 2.5 * day3 →
  day5 = 15 →
  total_pages - (day1 + day2 + day3 + day4 + day5) = 581.25 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_write_l783_78398


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l783_78320

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_count : vertices.card = 20
  edge_count : edges.card = 30
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that are endpoints of an edge in a regular dodecahedron -/
def edge_endpoint_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem: The probability of randomly selecting two vertices that are endpoints of an edge in a regular dodecahedron is 3/19 -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_endpoint_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l783_78320


namespace NUMINAMATH_CALUDE_product_of_two_positive_quantities_l783_78312

theorem product_of_two_positive_quantities (s : ℝ) (h : s > 0) :
  ¬(∀ x : ℝ, 0 < x → x < s → 
    (x * (s - x) ≤ y * (s - y) → (x = 0 ∨ x = s))) :=
sorry

end NUMINAMATH_CALUDE_product_of_two_positive_quantities_l783_78312


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l783_78380

-- Define an odd function with period 3
def is_odd_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 3) = f x)

-- Main theorem
theorem odd_periodic_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h_odd_periodic : is_odd_periodic f)
  (h_f1 : f 1 < 1)
  (h_f2 : f 2 = a) :
  a < 0 ∧ a ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l783_78380


namespace NUMINAMATH_CALUDE_smallest_linear_combination_l783_78369

theorem smallest_linear_combination (m n : ℤ) : ∃ (k : ℕ), k > 0 ∧ (∃ (a b : ℤ), k = 2017 * a + 48576 * b) ∧ 
  ∀ (l : ℕ), l > 0 → (∃ (c d : ℤ), l = 2017 * c + 48576 * d) → k ≤ l :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_l783_78369


namespace NUMINAMATH_CALUDE_third_pipe_rate_l783_78395

def pipe_A_rate : ℚ := 1 / 10
def pipe_B_rate : ℚ := 1 / 12
def combined_rate : ℚ := 7 / 60

theorem third_pipe_rate :
  ∃ (pipe_C_rate : ℚ),
    pipe_C_rate > 0 ∧
    pipe_A_rate + pipe_B_rate - pipe_C_rate = combined_rate ∧
    pipe_C_rate = 1 / 15 :=
by sorry

end NUMINAMATH_CALUDE_third_pipe_rate_l783_78395


namespace NUMINAMATH_CALUDE_display_rows_l783_78382

/-- Represents the number of cans in a row given its position from the top -/
def cans_in_row (n : ℕ) : ℕ := 3 * n - 2

/-- Calculates the total number of cans in the first n rows -/
def total_cans (n : ℕ) : ℕ := n * (3 * n - 1) / 2

theorem display_rows : ∃ n : ℕ, total_cans n = 225 ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_display_rows_l783_78382


namespace NUMINAMATH_CALUDE_percentage_increase_l783_78381

theorem percentage_increase (x y z : ℝ) : 
  y = 0.4 * z → x = 0.48 * z → (x - y) / y = 0.2 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_l783_78381


namespace NUMINAMATH_CALUDE_part_1_part_2_part_3_l783_78323

-- Part 1
theorem part_1 (p q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 3 ∧ x₁ + p / x₁ = q ∧ x₂ + p / x₂ = q) →
  p = -6 ∧ q = 1 := by sorry

-- Part 2
theorem part_2 :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ + 7 / x₁ = 8 ∧ x₂ + 7 / x₂ = 8) →
  (∃ x : ℝ, x + 7 / x = 8 ∧ ∀ y : ℝ, y + 7 / y = 8 → y ≤ x) →
  (∃ x : ℝ, x + 7 / x = 8 ∧ x = 7) := by sorry

-- Part 3
theorem part_3 (n : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    2 * x₁ + (n^2 - n) / (2 * x₁ - 1) = 2 * n ∧
    2 * x₂ + (n^2 - n) / (2 * x₂ - 1) = 2 * n) →
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    2 * x₁ + (n^2 - n) / (2 * x₁ - 1) = 2 * n ∧
    2 * x₂ + (n^2 - n) / (2 * x₂ - 1) = 2 * n ∧
    (2 * x₁ - 1) / (2 * x₂) = (n - 1) / (n + 1)) := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_part_3_l783_78323


namespace NUMINAMATH_CALUDE_curve_self_intersection_l783_78392

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t^2 - 3, t^3 - 6*t + 2)

-- Theorem statement
theorem curve_self_intersection :
  ∃! p : ℝ × ℝ, ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ curve t₁ = p ∧ curve t₂ = p ∧ p = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l783_78392


namespace NUMINAMATH_CALUDE_min_value_quadratic_l783_78356

theorem min_value_quadratic (x y : ℝ) :
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 45 / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l783_78356


namespace NUMINAMATH_CALUDE_cos_pi_half_minus_A_l783_78351

theorem cos_pi_half_minus_A (A : ℝ) (h : Real.sin (π - A) = 1/2) : 
  Real.cos (π/2 - A) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_half_minus_A_l783_78351


namespace NUMINAMATH_CALUDE_group_division_ways_l783_78350

theorem group_division_ways (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) : 
  Nat.choose n k = 20 := by
  sorry

end NUMINAMATH_CALUDE_group_division_ways_l783_78350


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l783_78347

open Real

theorem indefinite_integral_proof (x : ℝ) : 
  deriv (fun x => (1/2) * log (abs (x^2 - x + 1)) + 
                  Real.sqrt 3 * arctan ((2*x - 1) / Real.sqrt 3) + 
                  (1/2) * log (abs (x^2 + 1))) x = 
  (2*x^3 + 2*x + 1) / ((x^2 - x + 1) * (x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l783_78347


namespace NUMINAMATH_CALUDE_probability_two_non_red_marbles_l783_78352

/-- Given a bag of marbles, calculate the probability of drawing two non-red marbles in succession with replacement after the first draw. -/
theorem probability_two_non_red_marbles 
  (total_marbles : ℕ) 
  (red_marbles : ℕ) 
  (h1 : total_marbles = 84) 
  (h2 : red_marbles = 12) :
  (total_marbles - red_marbles : ℚ) / total_marbles * 
  ((total_marbles - red_marbles : ℚ) / total_marbles) = 36/49 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_non_red_marbles_l783_78352


namespace NUMINAMATH_CALUDE_inequality_proof_l783_78310

theorem inequality_proof (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l783_78310


namespace NUMINAMATH_CALUDE_system_solutions_l783_78341

def is_solution (x y z u : ℤ) : Prop :=
  x + y + z + u = 12 ∧
  x^2 + y^2 + z^2 + u^2 = 170 ∧
  x^3 + y^3 + z^3 + u^3 = 1764 ∧
  x * y = z * u

def solutions : List (ℤ × ℤ × ℤ × ℤ) :=
  [(12, -1, 4, -3), (12, -1, -3, 4), (-1, 12, 4, -3), (-1, 12, -3, 4),
   (4, -3, 12, -1), (4, -3, -1, 12), (-3, 4, 12, -1), (-3, 4, -1, 12)]

theorem system_solutions :
  (∀ x y z u : ℤ, is_solution x y z u ↔ (x, y, z, u) ∈ solutions) ∧
  solutions.length = 8 := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l783_78341


namespace NUMINAMATH_CALUDE_common_root_of_equations_l783_78389

theorem common_root_of_equations : ∃ x : ℚ, 
  2 * x^3 - 5 * x^2 + 6 * x - 2 = 0 ∧ 
  6 * x^3 - 3 * x^2 - 2 * x + 1 = 0 := by
  use 1/2
  sorry

#eval (2 * (1/2)^3 - 5 * (1/2)^2 + 6 * (1/2) - 2 : ℚ)
#eval (6 * (1/2)^3 - 3 * (1/2)^2 - 2 * (1/2) + 1 : ℚ)

end NUMINAMATH_CALUDE_common_root_of_equations_l783_78389


namespace NUMINAMATH_CALUDE_kelly_points_l783_78360

def golden_state_team (kelly : ℕ) : Prop :=
  let draymond := 12
  let curry := 2 * draymond
  let durant := 2 * kelly
  let klay := draymond / 2
  draymond + curry + kelly + durant + klay = 69

theorem kelly_points : ∃ (k : ℕ), golden_state_team k ∧ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_kelly_points_l783_78360


namespace NUMINAMATH_CALUDE_apple_boxes_weights_l783_78359

theorem apple_boxes_weights (a b c d : ℝ) 
  (h1 : a + b + c = 70)
  (h2 : a + b + d = 80)
  (h3 : a + c + d = 73)
  (h4 : b + c + d = 77)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0) :
  a = 23 ∧ b = 27 ∧ c = 20 ∧ d = 30 := by
sorry

end NUMINAMATH_CALUDE_apple_boxes_weights_l783_78359


namespace NUMINAMATH_CALUDE_rita_breaststroke_hours_l783_78355

/-- Calculates the hours of breaststroke completed by Rita --/
def breaststroke_hours (total_required : ℕ) (backstroke : ℕ) (butterfly : ℕ) (freestyle_sidestroke_per_month : ℕ) (months : ℕ) : ℕ :=
  total_required - (backstroke + butterfly + freestyle_sidestroke_per_month * months)

/-- Theorem stating that Rita completed 9 hours of breaststroke --/
theorem rita_breaststroke_hours : 
  breaststroke_hours 1500 50 121 220 6 = 9 := by
  sorry

#eval breaststroke_hours 1500 50 121 220 6

end NUMINAMATH_CALUDE_rita_breaststroke_hours_l783_78355


namespace NUMINAMATH_CALUDE_book_pages_proof_l783_78328

/-- The number of pages Jack reads per day -/
def pages_per_day : ℕ := 23

/-- The number of pages Jack reads on the last day -/
def last_day_pages : ℕ := 9

/-- The total number of pages in the book -/
def total_pages : ℕ := 32

theorem book_pages_proof :
  ∃ (full_days : ℕ), total_pages = pages_per_day * full_days + last_day_pages :=
by sorry

end NUMINAMATH_CALUDE_book_pages_proof_l783_78328


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l783_78394

/-- Given a line segment PQ with midpoint M(0,4) and P moving along x + y - 2 = 0,
    prove that the trajectory of Q is x + y - 6 = 0 -/
theorem trajectory_of_Q (P Q : ℝ × ℝ) (t : ℝ) : 
  let M := (0, 4)
  let P := (t, 2 - t)  -- parametric form of x + y - 2 = 0
  let Q := (2 * M.1 - P.1, 2 * M.2 - P.2)  -- Q is symmetric to P with respect to M
  Q.1 + Q.2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l783_78394


namespace NUMINAMATH_CALUDE_determinant_of_roots_l783_78330

theorem determinant_of_roots (p q : ℝ) (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 4*a^2 + p*a + q = 0 →
  b^3 - 4*b^2 + p*b + q = 0 →
  c^3 - 4*c^2 + p*c + q = 0 →
  Matrix.det !![a, b, c; b, c, a; c, a, b] = -64 + 12*p - 2*q := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_roots_l783_78330


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l783_78336

theorem geometric_sequence_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ ≠ 0)
  (h₂ : a₂ = 3 * a₁ + 3)
  (h₃ : a₃ = 6 * a₁ + 6)
  (h₄ : a₂^2 = a₁ * a₃)  -- Condition for geometric sequence
  : ∃ (r : ℝ), r ≠ 0 ∧ a₂ = r * a₁ ∧ a₃ = r * a₂ ∧ r * a₃ = -24 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l783_78336


namespace NUMINAMATH_CALUDE_only_striped_has_eight_legs_l783_78315

/-- Represents the color of an octopus -/
inductive OctopusColor
  | Green
  | DarkBlue
  | Purple
  | Striped

/-- Represents an octopus with its color and number of legs -/
structure Octopus where
  color : OctopusColor
  legs : ℕ

/-- Determines if an octopus tells the truth based on its number of legs -/
def tellsTruth (o : Octopus) : Prop :=
  o.legs % 2 = 0

/-- Represents the statements made by each octopus -/
def greenStatement (green darkBlue : Octopus) : Prop :=
  green.legs = 8 ∧ darkBlue.legs = 6

def darkBlueStatement (darkBlue green : Octopus) : Prop :=
  darkBlue.legs = 8 ∧ green.legs = 7

def purpleStatement (darkBlue purple : Octopus) : Prop :=
  darkBlue.legs = 8 ∧ purple.legs = 9

def stripedStatement (green darkBlue purple striped : Octopus) : Prop :=
  green.legs ≠ 8 ∧ darkBlue.legs ≠ 8 ∧ purple.legs ≠ 8 ∧ striped.legs = 8

/-- The main theorem stating that only the striped octopus has 8 legs -/
theorem only_striped_has_eight_legs
  (green darkBlue purple striped : Octopus)
  (h_green : green.color = OctopusColor.Green)
  (h_darkBlue : darkBlue.color = OctopusColor.DarkBlue)
  (h_purple : purple.color = OctopusColor.Purple)
  (h_striped : striped.color = OctopusColor.Striped)
  (h_greenStatement : tellsTruth green = greenStatement green darkBlue)
  (h_darkBlueStatement : tellsTruth darkBlue = darkBlueStatement darkBlue green)
  (h_purpleStatement : tellsTruth purple = purpleStatement darkBlue purple)
  (h_stripedStatement : tellsTruth striped = stripedStatement green darkBlue purple striped) :
  striped.legs = 8 ∧ green.legs ≠ 8 ∧ darkBlue.legs ≠ 8 ∧ purple.legs ≠ 8 :=
sorry

end NUMINAMATH_CALUDE_only_striped_has_eight_legs_l783_78315


namespace NUMINAMATH_CALUDE_stock_price_increase_l783_78325

theorem stock_price_increase (X : ℝ) : 
  (1 + X / 100) * (1 - 25 / 100) * (1 + 15 / 100) = 103.5 / 100 → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l783_78325


namespace NUMINAMATH_CALUDE_eighth_term_ratio_l783_78387

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := (n : ℚ) * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem eighth_term_ratio
  (a₁ d b₁ e : ℚ)
  (h : ∀ n : ℕ, arithmetic_sum a₁ d n / arithmetic_sum b₁ e n = (5 * n + 6 : ℚ) / (3 * n + 30 : ℚ)) :
  (arithmetic_sequence a₁ d 8) / (arithmetic_sequence b₁ e 8) = 4 / 3 :=
sorry

end NUMINAMATH_CALUDE_eighth_term_ratio_l783_78387


namespace NUMINAMATH_CALUDE_inverse_f_at_seven_l783_78335

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 3

-- State the theorem
theorem inverse_f_at_seven (x : ℝ) : f x = 7 → x = 101 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_seven_l783_78335


namespace NUMINAMATH_CALUDE_range_of_a_l783_78309

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x + 3 ≤ a^2 - 2*a - 1)) → 
  (-1 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l783_78309


namespace NUMINAMATH_CALUDE_certain_number_problem_l783_78326

theorem certain_number_problem (x : ℤ) (h : x + 5 * 8 = 340) : x = 300 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l783_78326


namespace NUMINAMATH_CALUDE_f_negative_alpha_l783_78376

noncomputable def f (x : ℝ) : ℝ := Real.tan x + 1 / Real.tan x

theorem f_negative_alpha (α : ℝ) (h : f α = 5) : f (-α) = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_alpha_l783_78376


namespace NUMINAMATH_CALUDE_apps_deleted_l783_78339

/-- Given that Dave initially had 150 apps on his phone and 65 apps remained after deletion,
    prove that the number of apps deleted is 85. -/
theorem apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) (h1 : initial_apps = 150) (h2 : remaining_apps = 65) :
  initial_apps - remaining_apps = 85 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_l783_78339


namespace NUMINAMATH_CALUDE_hexagon_side_length_l783_78385

theorem hexagon_side_length (perimeter : ℝ) (h : perimeter = 48) : 
  perimeter / 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l783_78385


namespace NUMINAMATH_CALUDE_smallest_valid_n_l783_78354

/-- Represents the graph structure with 8 vertices --/
def Graph := Fin 8 → Fin 8 → Bool

/-- The specific graph structure given in the problem --/
def problemGraph : Graph := sorry

/-- Checks if two numbers are coprime --/
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Checks if two numbers have a common divisor greater than 1 --/
def hasCommonDivisorGreaterThanOne (a b : ℕ) : Prop := ∃ (d : ℕ), d > 1 ∧ d ∣ a ∧ d ∣ b

/-- Represents a valid arrangement of numbers in the graph --/
def ValidArrangement (n : ℕ) (arr : Fin 8 → ℕ) : Prop :=
  (∀ i j, i ≠ j → arr i ≠ arr j) ∧
  (∀ i j, ¬problemGraph i j → coprime (arr i + arr j) n) ∧
  (∀ i j, problemGraph i j → hasCommonDivisorGreaterThanOne (arr i + arr j) n)

/-- The main theorem stating that 35 is the smallest valid n --/
theorem smallest_valid_n :
  (∃ (arr : Fin 8 → ℕ), ValidArrangement 35 arr) ∧
  (∀ n < 35, ¬∃ (arr : Fin 8 → ℕ), ValidArrangement n arr) := by sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l783_78354


namespace NUMINAMATH_CALUDE_negation_equivalence_l783_78321

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ ≤ 0 ∧ x₀^2 ≥ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l783_78321


namespace NUMINAMATH_CALUDE_finite_solutions_except_two_l783_78393

/-- The set of positive integer solutions x for the equation xn+1 | n^2+kn+1 -/
def S (n k : ℕ+) : Set ℕ+ :=
  {x | ∃ m : ℕ+, (x * n + 1) * m = n^2 + k * n + 1}

/-- The set of positive integers n for which S n k has at least two elements -/
def P (k : ℕ+) : Set ℕ+ :=
  {n | ∃ x y : ℕ+, x ≠ y ∧ x ∈ S n k ∧ y ∈ S n k}

theorem finite_solutions_except_two :
  ∀ k : ℕ+, k ≠ 2 → Set.Finite (P k) :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_except_two_l783_78393


namespace NUMINAMATH_CALUDE_cyclists_meet_time_l783_78384

/-- The time (in hours after 8:00 AM) when Cassie and Brian meet -/
def meeting_time : ℝ := 2.68333333

/-- The total distance of the route in miles -/
def total_distance : ℝ := 75

/-- Cassie's speed in miles per hour -/
def cassie_speed : ℝ := 15

/-- Brian's speed in miles per hour -/
def brian_speed : ℝ := 18

/-- The time difference between Cassie and Brian's departure in hours -/
def time_difference : ℝ := 0.75

theorem cyclists_meet_time :
  cassie_speed * meeting_time + brian_speed * (meeting_time - time_difference) = total_distance :=
sorry

end NUMINAMATH_CALUDE_cyclists_meet_time_l783_78384


namespace NUMINAMATH_CALUDE_division_equality_l783_78362

theorem division_equality : (999 - 99 + 9) / 9 = 101 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l783_78362


namespace NUMINAMATH_CALUDE_brothers_ages_l783_78317

theorem brothers_ages (a b : ℕ) (h1 : a > b) (h2 : a / b = 3 / 2) (h3 : a - b = 24) :
  a + b = 120 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ages_l783_78317


namespace NUMINAMATH_CALUDE_at_least_two_equal_l783_78324

theorem at_least_two_equal (x y z : ℝ) : 
  (x - y) / (2 + x * y) + (y - z) / (2 + y * z) + (z - x) / (2 + z * x) = 0 →
  (x = y ∨ y = z ∨ z = x) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_equal_l783_78324


namespace NUMINAMATH_CALUDE_machine_times_solution_l783_78377

/-- Represents the time taken by three machines to complete a task individually and together -/
structure MachineTimes where
  first : ℝ
  second : ℝ
  third : ℝ
  together : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (t : MachineTimes) : Prop :=
  t.second = t.first + 2 ∧
  t.third = 2 * t.first ∧
  t.together = 8/3

/-- The theorem statement -/
theorem machine_times_solution (t : MachineTimes) :
  satisfies_conditions t → t.first = 6 ∧ t.second = 8 ∧ t.third = 12 := by
  sorry

end NUMINAMATH_CALUDE_machine_times_solution_l783_78377


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l783_78378

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  ((1 + i)^10) / (1 - i) = -16 + 16*i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l783_78378


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l783_78399

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 10 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l783_78399


namespace NUMINAMATH_CALUDE_quadratic_has_minimum_l783_78372

/-- Given a quadratic function f(x) = ax^2 + bx + c where c = b^2 / (9a) and a > 0,
    prove that the graph of y = f(x) has a minimum. -/
theorem quadratic_has_minimum (a b : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + b^2 / (9 * a)
  ∃ x_min : ℝ, ∀ x : ℝ, f x_min ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_has_minimum_l783_78372


namespace NUMINAMATH_CALUDE_three_intersection_points_l783_78358

-- Define the three lines
def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 8 * x - 12 * y = 9

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
by
  sorry

end NUMINAMATH_CALUDE_three_intersection_points_l783_78358


namespace NUMINAMATH_CALUDE_math_club_exclusive_members_l783_78346

theorem math_club_exclusive_members :
  ∀ (total_students : ℕ) (both_clubs : ℕ) (math_club : ℕ) (science_club : ℕ),
    total_students = 30 →
    both_clubs = 2 →
    math_club = 3 * science_club →
    total_students = math_club + science_club - both_clubs →
    math_club - both_clubs = 20 :=
by sorry

end NUMINAMATH_CALUDE_math_club_exclusive_members_l783_78346


namespace NUMINAMATH_CALUDE_problem_solution_l783_78397

/-- An increasing linear function on ℝ -/
def IncreasingLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ ∀ x, f x = a * x + b

theorem problem_solution (f g : ℝ → ℝ) (m : ℝ) :
  IncreasingLinearFunction f →
  (∀ x, g x = f x * (x + m)) →
  (∀ x, f (f x) = 16 * x + 5) →
  (∃ M, M = 13 ∧ ∀ x ∈ Set.Icc 1 3, g x ≤ M) →
  (∀ x, f x = 4 * x + 1) ∧ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l783_78397


namespace NUMINAMATH_CALUDE_inequality_solution_set_l783_78396

theorem inequality_solution_set :
  {x : ℝ | 2*x - 6 < 0} = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l783_78396


namespace NUMINAMATH_CALUDE_arbitrarily_large_solution_exists_l783_78361

theorem arbitrarily_large_solution_exists (N : ℕ) : 
  ∃ (a b c d : ℤ), 
    (a * a + b * b + c * c + d * d = a * b * c + a * b * d + a * c * d + b * c * d) ∧ 
    (min a (min b (min c d)) ≥ N) := by
  sorry

end NUMINAMATH_CALUDE_arbitrarily_large_solution_exists_l783_78361


namespace NUMINAMATH_CALUDE_chord_equation_l783_78304

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

theorem chord_equation (m n s t : ℝ) 
  (h_positive : m > 0 ∧ n > 0 ∧ s > 0 ∧ t > 0)
  (h_sum : m + n = 2)
  (h_ratio : m / s + n / t = 9)
  (h_min : s + t = 4 / 9)
  (h_midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    ellipse x₁ y₁ ∧ 
    ellipse x₂ y₂ ∧ 
    m = (x₁ + x₂) / 2 ∧ 
    n = (y₁ + y₂) / 2) :
  ∃ (a b c : ℝ), a * m + b * n + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3 :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l783_78304


namespace NUMINAMATH_CALUDE_curve_is_two_intersecting_lines_l783_78345

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  2 * x^2 - y^2 - 4 * x - 4 * y - 2 = 0

/-- The first line equation derived from the curve equation -/
def line1 (x y : ℝ) : Prop :=
  y = Real.sqrt 2 * x - Real.sqrt 2 - 2

/-- The second line equation derived from the curve equation -/
def line2 (x y : ℝ) : Prop :=
  y = -Real.sqrt 2 * x + Real.sqrt 2 - 2

/-- Theorem stating that the curve equation represents two intersecting lines -/
theorem curve_is_two_intersecting_lines :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (∀ x y, curve_equation x y ↔ (line1 x y ∨ line2 x y)) ∧ 
    (line1 x₁ y₁ ∧ line2 x₁ y₁) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
  sorry

end NUMINAMATH_CALUDE_curve_is_two_intersecting_lines_l783_78345


namespace NUMINAMATH_CALUDE_three_sqrt_two_gt_sqrt_seventeen_l783_78349

theorem three_sqrt_two_gt_sqrt_seventeen : 3 * Real.sqrt 2 > Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_three_sqrt_two_gt_sqrt_seventeen_l783_78349


namespace NUMINAMATH_CALUDE_three_digit_integer_property_l783_78318

def three_digit_integer (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

def digit_sum (a b c : ℕ) : ℕ := a + b + c

def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

theorem three_digit_integer_property (a b c : ℕ) 
  (h1 : a < 10 ∧ b < 10 ∧ c < 10) 
  (h2 : three_digit_integer a b c - 7 * digit_sum a b c = 100) :
  ∃ y : ℕ, reversed_number a b c = y * digit_sum a b c ∧ y = 43 := by
sorry

end NUMINAMATH_CALUDE_three_digit_integer_property_l783_78318


namespace NUMINAMATH_CALUDE_special_function_property_l783_78303

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

theorem special_function_property (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 ≠ 0) :
  (f 5 - f 1) / f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l783_78303


namespace NUMINAMATH_CALUDE_x_value_theorem_l783_78390

theorem x_value_theorem (x : ℝ) : x * (x * (x + 1) + 2) + 3 = x^3 + x^2 + x - 6 → x = -9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_theorem_l783_78390


namespace NUMINAMATH_CALUDE_function_range_theorem_l783_78373

open Real

theorem function_range_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, 9 * f x < x * (deriv f x) ∧ x * (deriv f x) < 10 * f x)
  (h2 : ∀ x > 0, f x > 0) :
  2^9 < f 2 / f 1 ∧ f 2 / f 1 < 2^10 := by
sorry

end NUMINAMATH_CALUDE_function_range_theorem_l783_78373


namespace NUMINAMATH_CALUDE_opposite_of_a_is_smallest_positive_integer_l783_78343

theorem opposite_of_a_is_smallest_positive_integer (a : ℤ) : 
  (∃ (x : ℤ), x > 0 ∧ ∀ (y : ℤ), y > 0 → x ≤ y) ∧ (-a = x) → 3*a - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_a_is_smallest_positive_integer_l783_78343


namespace NUMINAMATH_CALUDE_max_distance_between_circles_l783_78307

/-- Circle C₁ with equation x² + (y+3)² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + (y+3)^2 = 1

/-- Circle C₂ with equation (x-4)² + y² = 4 -/
def C₂ (x y : ℝ) : Prop := (x-4)^2 + y^2 = 4

/-- The maximum distance between any point on C₁ and any point on C₂ is 8 -/
theorem max_distance_between_circles :
  ∃ (max_dist : ℝ),
    (∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ → 
      (x₁ - x₂)^2 + (y₁ - y₂)^2 ≤ max_dist^2) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = max_dist^2) ∧
    max_dist = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_circles_l783_78307


namespace NUMINAMATH_CALUDE_lara_flowers_in_vase_l783_78364

/-- The number of flowers Lara put in the vase -/
def flowers_in_vase (total : ℕ) (to_mom : ℕ) (extra_to_grandma : ℕ) : ℕ :=
  total - (to_mom + (to_mom + extra_to_grandma))

/-- Theorem stating the number of flowers Lara put in the vase -/
theorem lara_flowers_in_vase :
  flowers_in_vase 52 15 6 = 16 := by sorry

end NUMINAMATH_CALUDE_lara_flowers_in_vase_l783_78364


namespace NUMINAMATH_CALUDE_even_four_digit_count_is_336_l783_78366

/-- A function that counts the number of even integers between 4000 and 8000 with four different digits -/
def count_even_four_digit_numbers : ℕ :=
  336

/-- Theorem stating that the count of even integers between 4000 and 8000 with four different digits is 336 -/
theorem even_four_digit_count_is_336 : count_even_four_digit_numbers = 336 := by
  sorry

end NUMINAMATH_CALUDE_even_four_digit_count_is_336_l783_78366


namespace NUMINAMATH_CALUDE_average_of_solutions_l783_78301

variable (b : ℝ)

def quadratic_equation (x : ℝ) : Prop :=
  3 * x^2 - 6 * b * x + 2 * b = 0

def has_two_real_solutions : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation b x₁ ∧ quadratic_equation b x₂

theorem average_of_solutions :
  has_two_real_solutions b →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic_equation b x₁ ∧ 
    quadratic_equation b x₂ ∧
    (x₁ + x₂) / 2 = b :=
by sorry

end NUMINAMATH_CALUDE_average_of_solutions_l783_78301


namespace NUMINAMATH_CALUDE_sum_of_integers_l783_78357

theorem sum_of_integers (x y : ℤ) : 
  3 * x + 2 * y = 115 → (x = 25 ∨ y = 25) → (x = 20 ∨ y = 20) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l783_78357


namespace NUMINAMATH_CALUDE_sam_age_l783_78308

theorem sam_age (drew_age : ℕ) (sam_age : ℕ) : 
  drew_age + sam_age = 54 →
  sam_age = drew_age / 2 →
  sam_age = 18 := by
sorry

end NUMINAMATH_CALUDE_sam_age_l783_78308


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l783_78332

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x : ℝ, x < 0 → x < a) ∧ 
  (∃ x : ℝ, x ≥ 0 ∧ x < a) ↔ 
  a > 0 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l783_78332


namespace NUMINAMATH_CALUDE_inequality_condition_l783_78340

theorem inequality_condition (b : ℝ) : 
  (b > 0) → (∃ x : ℝ, |x - 2| + |x - 5| < b) ↔ b > 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l783_78340


namespace NUMINAMATH_CALUDE_pepperoni_coverage_is_four_ninths_l783_78333

/-- Represents a circular pizza with pepperoni toppings -/
structure PepperoniPizza where
  pizza_diameter : ℝ
  pepperoni_across_diameter : ℕ
  total_pepperoni : ℕ

/-- Calculates the fraction of the pizza covered by pepperoni -/
def pepperoni_coverage (p : PepperoniPizza) : ℚ :=
  sorry

/-- Theorem stating that the fraction of the pizza covered by pepperoni is 4/9 -/
theorem pepperoni_coverage_is_four_ninths (p : PepperoniPizza) 
  (h1 : p.pizza_diameter = 18)
  (h2 : p.pepperoni_across_diameter = 9)
  (h3 : p.total_pepperoni = 36) : 
  pepperoni_coverage p = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_is_four_ninths_l783_78333


namespace NUMINAMATH_CALUDE_hyperbola_center_l783_78313

/-- The center of a hyperbola with foci at (3, 6) and (11, 10) is at (7, 8) -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) (h1 : f1 = (3, 6)) (h2 : f2 = (11, 10)) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  center = (7, 8) := by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l783_78313


namespace NUMINAMATH_CALUDE_binary_of_28_l783_78319

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem binary_of_28 :
  decimal_to_binary 28 = [1, 1, 1, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_binary_of_28_l783_78319


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_2_dividing_32_factorial_l783_78388

/-- The largest power of 2 that divides n! -/
def largestPowerOf2DividingFactorial (n : ℕ) : ℕ :=
  sorry

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  onesDigit (2^(largestPowerOf2DividingFactorial 32)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_2_dividing_32_factorial_l783_78388


namespace NUMINAMATH_CALUDE_union_equality_implies_a_values_l783_78311

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, a}
def B (a : ℝ) : Set ℝ := {a^2}

-- State the theorem
theorem union_equality_implies_a_values (a : ℝ) :
  A a ∪ B a = A a → a = -1 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_values_l783_78311


namespace NUMINAMATH_CALUDE_knitting_time_theorem_l783_78383

/-- Represents the time in hours to knit each item -/
structure KnittingTime where
  hat : ℝ
  scarf : ℝ
  sweater : ℝ
  mittens : ℝ
  socks : ℝ

/-- Calculates the total time to knit multiple sets of clothes -/
def totalKnittingTime (time : KnittingTime) (sets : ℕ) : ℝ :=
  (time.hat + time.scarf + time.sweater + time.mittens + time.socks) * sets

/-- Theorem: The total time to knit 3 sets of clothes with given knitting times is 48 hours -/
theorem knitting_time_theorem (time : KnittingTime) 
  (h_hat : time.hat = 2)
  (h_scarf : time.scarf = 3)
  (h_sweater : time.sweater = 6)
  (h_mittens : time.mittens = 2)
  (h_socks : time.socks = 3) :
  totalKnittingTime time 3 = 48 := by
  sorry

#check knitting_time_theorem

end NUMINAMATH_CALUDE_knitting_time_theorem_l783_78383


namespace NUMINAMATH_CALUDE_inequality_preservation_l783_78391

theorem inequality_preservation (x y : ℝ) : x < y → 2 * x < 2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l783_78391


namespace NUMINAMATH_CALUDE_acid_mixture_theorem_l783_78379

/-- Represents an acid solution with a given concentration and volume -/
structure AcidSolution where
  concentration : ℝ
  volume : ℝ

/-- Calculates the amount of pure acid in a solution -/
def pureAcid (solution : AcidSolution) : ℝ :=
  solution.concentration * solution.volume

/-- Theorem: Mixing 4L of 60% acid with 16L of 75% acid yields 20L of 72% acid -/
theorem acid_mixture_theorem :
  let solution1 : AcidSolution := { concentration := 0.60, volume := 4 }
  let solution2 : AcidSolution := { concentration := 0.75, volume := 16 }
  let finalSolution : AcidSolution := { concentration := 0.72, volume := 20 }
  pureAcid solution1 + pureAcid solution2 = pureAcid finalSolution :=
by sorry

end NUMINAMATH_CALUDE_acid_mixture_theorem_l783_78379


namespace NUMINAMATH_CALUDE_problem_statement_l783_78337

theorem problem_statement (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 4) :
  x^2 * y^3 + y^2 * x^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l783_78337


namespace NUMINAMATH_CALUDE_pool_filling_time_l783_78386

/-- Proves that filling a 24,000-gallon pool with 5 hoses supplying 3 gallons per minute takes 27 hours (rounded) -/
theorem pool_filling_time :
  let pool_capacity : ℕ := 24000
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℕ := 3
  let minutes_per_hour : ℕ := 60
  let total_flow_rate := num_hoses * flow_rate_per_hose * minutes_per_hour
  let filling_time := (pool_capacity + total_flow_rate - 1) / total_flow_rate
  filling_time = 27 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_l783_78386


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_average_l783_78329

theorem consecutive_odd_numbers_average (a b c d : ℕ) : 
  a = 27 ∧ 
  b = a - 2 ∧ 
  c = b - 2 ∧ 
  d = c - 2 ∧ 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d → 
  (a + b + c + d) / 4 = 24 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_average_l783_78329


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l783_78367

theorem average_of_four_numbers (p q r s : ℝ) 
  (h : (5 : ℝ) / 4 * (p + q + r + s) = 15) : 
  (p + q + r + s) / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l783_78367


namespace NUMINAMATH_CALUDE_abs_equation_solution_l783_78371

theorem abs_equation_solution : ∃! x : ℝ, |x - 3| = 5 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l783_78371


namespace NUMINAMATH_CALUDE_absolute_value_expression_l783_78375

theorem absolute_value_expression (x : ℤ) (h : x = -2023) :
  |abs (abs x - x) - abs x| - x = 4046 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l783_78375


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l783_78342

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 4*x + 8) / ((x - 1)*(x - 4)*(x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) ∧
      P = 1/3 ∧ Q = -4/3 ∧ R = 2 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l783_78342


namespace NUMINAMATH_CALUDE_entree_percentage_is_80_percent_l783_78305

/-- Calculates the percentage of total cost that went to entrees -/
def entree_percentage (total_cost appetizer_cost : ℚ) (num_appetizers : ℕ) : ℚ :=
  let appetizer_total := appetizer_cost * num_appetizers
  let entree_total := total_cost - appetizer_total
  (entree_total / total_cost) * 100

/-- Theorem stating that the percentage of total cost that went to entrees is 80% -/
theorem entree_percentage_is_80_percent :
  entree_percentage 50 5 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_entree_percentage_is_80_percent_l783_78305


namespace NUMINAMATH_CALUDE_sin_monotone_decreasing_l783_78306

theorem sin_monotone_decreasing (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (π / 3 - 2 * x)
  ∀ x y, x ∈ Set.Icc (k * π - π / 12) (k * π + 5 * π / 12) →
         y ∈ Set.Icc (k * π - π / 12) (k * π + 5 * π / 12) →
         x ≤ y → f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_sin_monotone_decreasing_l783_78306


namespace NUMINAMATH_CALUDE_spadesuit_problem_l783_78370

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spadesuit_problem : spadesuit (spadesuit 2 3) (spadesuit 6 (spadesuit 9 4)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spadesuit_problem_l783_78370


namespace NUMINAMATH_CALUDE_normal_pump_rate_l783_78331

/-- Proves that the normal pump rate is 6 gallons per minute given the conditions -/
theorem normal_pump_rate (pond_capacity : ℝ) (fill_time : ℝ) (rate_fraction : ℝ) : 
  pond_capacity = 200 → 
  fill_time = 50 → 
  rate_fraction = 2/3 → 
  (rate_fraction * (pond_capacity / fill_time)) = 6 := by
  sorry

#check normal_pump_rate

end NUMINAMATH_CALUDE_normal_pump_rate_l783_78331
