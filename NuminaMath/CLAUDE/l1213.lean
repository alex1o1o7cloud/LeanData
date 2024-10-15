import Mathlib

namespace NUMINAMATH_CALUDE_not_perfect_square_l1213_121397

theorem not_perfect_square (n : ℕ) : ¬ ∃ m : ℤ, (3^n : ℤ) + 2 * (17^n : ℤ) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1213_121397


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l1213_121347

/-- Given two similar triangles PQR and STU, prove that PQ = 10.5 -/
theorem similar_triangles_side_length 
  (PQ ST PR SU : ℝ) 
  (h_similar : PQ / ST = PR / SU) 
  (h_ST : ST = 4.5) 
  (h_PR : PR = 21) 
  (h_SU : SU = 9) : 
  PQ = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l1213_121347


namespace NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l1213_121320

def alphabet : Finset Char := sorry

def mathematics : Finset Char := sorry

theorem probability_of_letter_in_mathematics :
  (mathematics.card : ℚ) / (alphabet.card : ℚ) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l1213_121320


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1213_121337

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1213_121337


namespace NUMINAMATH_CALUDE_earnings_per_lawn_l1213_121322

theorem earnings_per_lawn (total_lawns forgotten_lawns : ℕ) (total_earnings : ℚ) :
  total_lawns = 12 →
  forgotten_lawns = 8 →
  total_earnings = 36 →
  total_earnings / (total_lawns - forgotten_lawns) = 9 := by
sorry

end NUMINAMATH_CALUDE_earnings_per_lawn_l1213_121322


namespace NUMINAMATH_CALUDE_puppy_discount_percentage_l1213_121319

/-- Calculates the discount percentage given the total cost before discount and the amount spent after discount -/
def discount_percentage (total_cost : ℚ) (amount_spent : ℚ) : ℚ :=
  (total_cost - amount_spent) / total_cost * 100

/-- Proves that the new-customer discount percentage is 20% for Julia's puppy purchases -/
theorem puppy_discount_percentage :
  let adoption_fee : ℚ := 20
  let dog_food : ℚ := 20
  let treats : ℚ := 2 * 2.5
  let toys : ℚ := 15
  let crate : ℚ := 20
  let bed : ℚ := 20
  let collar_leash : ℚ := 15
  let total_cost : ℚ := dog_food + treats + toys + crate + bed + collar_leash
  let total_spent : ℚ := 96
  let store_spent : ℚ := total_spent - adoption_fee
  discount_percentage total_cost store_spent = 20 := by
sorry

#eval discount_percentage 95 76

end NUMINAMATH_CALUDE_puppy_discount_percentage_l1213_121319


namespace NUMINAMATH_CALUDE_equilateral_not_unique_from_angle_and_median_l1213_121327

/-- Represents a triangle -/
structure Triangle where
  a : ℝ  -- side length a
  b : ℝ  -- side length b
  c : ℝ  -- side length c
  α : ℝ  -- angle opposite to side a
  β : ℝ  -- angle opposite to side b
  γ : ℝ  -- angle opposite to side c

/-- Represents a median of a triangle -/
def Median (t : Triangle) (side : ℕ) : ℝ := sorry

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- Theorem stating that one angle and the median to the opposite side
    do not uniquely determine an equilateral triangle -/
theorem equilateral_not_unique_from_angle_and_median :
  ∃ (t1 t2 : Triangle) (side : ℕ),
    t1.α = t2.α ∧
    Median t1 side = Median t2 side ∧
    IsEquilateral t1 ∧
    IsEquilateral t2 ∧
    t1 ≠ t2 :=
  sorry

end NUMINAMATH_CALUDE_equilateral_not_unique_from_angle_and_median_l1213_121327


namespace NUMINAMATH_CALUDE_money_distribution_l1213_121377

def total_proportion : ℕ := 5 + 2 + 4 + 3

theorem money_distribution (S : ℚ) (A_share B_share C_share D_share : ℚ) : 
  A_share = 2500 ∧ 
  A_share = (5 : ℚ) / total_proportion * S ∧
  B_share = (2 : ℚ) / total_proportion * S ∧
  C_share = (4 : ℚ) / total_proportion * S ∧
  D_share = (3 : ℚ) / total_proportion * S →
  C_share - D_share = 500 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1213_121377


namespace NUMINAMATH_CALUDE_rectangle_tiling_no_walls_l1213_121335

/-- A domino tiling of a rectangle. -/
def DominoTiling (m n : ℕ) := Unit

/-- A wall in a domino tiling. -/
def Wall (m n : ℕ) (tiling : DominoTiling m n) := Unit

/-- Predicate indicating if a tiling has no walls. -/
def HasNoWalls (m n : ℕ) (tiling : DominoTiling m n) : Prop :=
  ∀ w : Wall m n tiling, False

theorem rectangle_tiling_no_walls 
  (m n : ℕ) 
  (h_even : Even (m * n))
  (h_m : m ≥ 5)
  (h_n : n ≥ 5)
  (h_not_six : ¬(m = 6 ∧ n = 6)) :
  ∃ (tiling : DominoTiling m n), HasNoWalls m n tiling :=
sorry

end NUMINAMATH_CALUDE_rectangle_tiling_no_walls_l1213_121335


namespace NUMINAMATH_CALUDE_fourth_sampled_number_l1213_121383

/-- Represents a random number table -/
def RandomNumberTable := List (List Nat)

/-- Represents a position in the random number table -/
structure TablePosition where
  row : Nat
  column : Nat

/-- Checks if a number is valid for sampling (between 1 and 40) -/
def isValidNumber (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 40

/-- Gets the next position in the table -/
def nextPosition (pos : TablePosition) (tableWidth : Nat) : TablePosition :=
  if pos.column < tableWidth then
    { row := pos.row, column := pos.column + 1 }
  else
    { row := pos.row + 1, column := 1 }

/-- Samples the next valid number from the table -/
def sampleNextNumber (table : RandomNumberTable) (startPos : TablePosition) : Option Nat :=
  sorry

/-- Samples n valid numbers from the table -/
def sampleNumbers (table : RandomNumberTable) (startPos : TablePosition) (n : Nat) : List Nat :=
  sorry

/-- The main theorem to prove -/
theorem fourth_sampled_number
  (table : RandomNumberTable)
  (startPos : TablePosition)
  (h_table : table = [
    [84, 42, 17, 56, 31, 07, 23, 55, 06, 82, 77, 04, 74, 43, 59, 76, 30, 63, 50, 25, 83, 92, 12, 06],
    [63, 01, 63, 78, 59, 16, 95, 56, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38]
  ])
  (h_startPos : startPos = { row := 0, column := 7 })
  : (sampleNumbers table startPos 4).get! 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_sampled_number_l1213_121383


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l1213_121394

theorem line_ellipse_intersection (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x + 7) ↔ m^2 ≥ (9/50) := by
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l1213_121394


namespace NUMINAMATH_CALUDE_angle_terminal_side_value_l1213_121382

/-- Given that the terminal side of angle α passes through the point P(-4a,3a) where a ≠ 0,
    the value of 2sin α + cos α is either 2/5 or -2/5 -/
theorem angle_terminal_side_value (a : ℝ) (α : ℝ) (h : a ≠ 0) :
  let P : ℝ × ℝ := (-4 * a, 3 * a)
  (∃ t : ℝ, t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  2 * Real.sin α + Real.cos α = 2/5 ∨ 2 * Real.sin α + Real.cos α = -2/5 :=
by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_value_l1213_121382


namespace NUMINAMATH_CALUDE_rectangle_overlap_theorem_l1213_121300

/-- A rectangle with a given area -/
structure Rectangle where
  area : ℝ

/-- A configuration of rectangles placed within a larger rectangle -/
structure Configuration where
  outer : Rectangle
  inner : List Rectangle

/-- Predicate to check if two rectangles overlap by at least a given area -/
def overlaps (r1 r2 : Rectangle) (min_overlap : ℝ) : Prop :=
  ∃ (overlap_area : ℝ), overlap_area ≥ min_overlap

theorem rectangle_overlap_theorem (config : Configuration) :
  config.outer.area = 5 →
  config.inner.length = 9 →
  ∀ r ∈ config.inner, r.area = 1 →
  ∃ (r1 r2 : Rectangle), r1 ∈ config.inner ∧ r2 ∈ config.inner ∧ r1 ≠ r2 ∧ overlaps r1 r2 (1/9) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_overlap_theorem_l1213_121300


namespace NUMINAMATH_CALUDE_circle_iff_a_eq_neg_one_l1213_121371

/-- Represents a quadratic equation in x and y with parameter a -/
def is_circle (a : ℝ) : Prop :=
  ∃ h k r, ∀ x y : ℝ, 
    a^2 * x^2 + (a + 2) * y^2 + 2*a*x + a = 0 ↔ 
    (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

/-- The equation represents a circle if and only if a = -1 -/
theorem circle_iff_a_eq_neg_one :
  ∀ a : ℝ, is_circle a ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_circle_iff_a_eq_neg_one_l1213_121371


namespace NUMINAMATH_CALUDE_average_monthly_increase_l1213_121356

/-- Represents the monthly growth rate as a real number between 0 and 1 -/
def monthly_growth_rate : ℝ := sorry

/-- The initial turnover in January in millions of yuan -/
def initial_turnover : ℝ := 2

/-- The turnover in March in millions of yuan -/
def march_turnover : ℝ := 2.88

/-- The number of months between January and March -/
def months_passed : ℕ := 2

theorem average_monthly_increase :
  initial_turnover * (1 + monthly_growth_rate) ^ months_passed = march_turnover ∧
  monthly_growth_rate = 0.2 := by sorry

end NUMINAMATH_CALUDE_average_monthly_increase_l1213_121356


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l1213_121392

/-- A function that generates all valid eight-digit numbers using the digits 4, 0, 2, 6 twice each -/
def validNumbers : List Nat := sorry

/-- The largest eight-digit number that can be formed using the digits 4, 0, 2, 6 twice each -/
def largestNumber : Nat := sorry

/-- The smallest eight-digit number that can be formed using the digits 4, 0, 2, 6 twice each -/
def smallestNumber : Nat := sorry

/-- Theorem stating that the sum of the largest and smallest valid numbers is 86,466,666 -/
theorem sum_of_largest_and_smallest :
  largestNumber + smallestNumber = 86466666 := by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l1213_121392


namespace NUMINAMATH_CALUDE_piggy_bank_dimes_l1213_121359

theorem piggy_bank_dimes (total_value : ℚ) (total_coins : ℕ) 
  (quarter_value : ℚ) (dime_value : ℚ) :
  total_value = 39.5 ∧ 
  total_coins = 200 ∧ 
  quarter_value = 0.25 ∧ 
  dime_value = 0.1 →
  ∃ (quarters dimes : ℕ), 
    quarters + dimes = total_coins ∧
    quarter_value * quarters + dime_value * dimes = total_value ∧
    dimes = 70 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_dimes_l1213_121359


namespace NUMINAMATH_CALUDE_sum_product_inequality_l1213_121328

theorem sum_product_inequality (a b c x y z k : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hk1 : a + x = k) (hk2 : b + y = k) (hk3 : c + z = k) : 
  a * y + b * z + c * x < k^2 := by
sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l1213_121328


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_plane_l1213_121380

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (p : Plane) : Prop := sorry

/-- Two geometric objects are different -/
def different (a b : α) : Prop := a ≠ b

theorem line_parallel_perpendicular_plane 
  (m n : Line) (α : Plane) 
  (h1 : different m n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n α) : 
  perpendicular m α := by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_plane_l1213_121380


namespace NUMINAMATH_CALUDE_regular_polygon_area_l1213_121369

theorem regular_polygon_area (n : ℕ) (R : ℝ) (h : R > 0) :
  (1 / 2 : ℝ) * n * R^2 * Real.sin (2 * Real.pi / n) = 3 * R^2 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_area_l1213_121369


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l1213_121314

theorem fraction_product_theorem : 
  (7 : ℚ) / 4 * 8 / 14 * 16 / 24 * 32 / 48 * 28 / 7 * 15 / 9 * 50 / 25 * 21 / 35 = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l1213_121314


namespace NUMINAMATH_CALUDE_otimes_neg_two_four_otimes_equation_l1213_121343

/-- Define the ⊗ operation for rational numbers -/
def otimes (a b : ℚ) : ℚ := a * b^2 + 2 * a * b + a

/-- Theorem 1: (-2) ⊗ 4 = -50 -/
theorem otimes_neg_two_four : otimes (-2) 4 = -50 := by sorry

/-- Theorem 2: If x ⊗ 3 = y ⊗ (-3), then 8x - 2y + 5 = 5 -/
theorem otimes_equation (x y : ℚ) (h : otimes x 3 = otimes y (-3)) : 
  8 * x - 2 * y + 5 = 5 := by sorry

end NUMINAMATH_CALUDE_otimes_neg_two_four_otimes_equation_l1213_121343


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_81_l1213_121358

theorem alpha_plus_beta_equals_81 
  (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 72*x + 945) / (x^2 + 45*x - 3240)) : 
  α + β = 81 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_81_l1213_121358


namespace NUMINAMATH_CALUDE_total_water_flow_l1213_121352

def water_flow_rate : ℚ := 2 + 2/3
def time_period : ℕ := 9

theorem total_water_flow (rate : ℚ) (time : ℕ) (h1 : rate = water_flow_rate) (h2 : time = time_period) :
  rate * time = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_water_flow_l1213_121352


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1213_121345

/-- The sum of the coefficients of the expanded expression -(2x - 5)(4x + 3(2x - 5)) is -15 -/
theorem sum_of_coefficients : ∃ a b c : ℚ,
  -(2 * X - 5) * (4 * X + 3 * (2 * X - 5)) = a * X^2 + b * X + c ∧ a + b + c = -15 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1213_121345


namespace NUMINAMATH_CALUDE_y_squared_mod_30_l1213_121355

theorem y_squared_mod_30 (y : ℤ) (h1 : 6 * y ≡ 12 [ZMOD 30]) (h2 : 5 * y ≡ 25 [ZMOD 30]) :
  y^2 ≡ 19 [ZMOD 30] := by
  sorry

end NUMINAMATH_CALUDE_y_squared_mod_30_l1213_121355


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l1213_121333

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 2) :
  2 * x^2 + 3 * y^2 + z^2 ≥ 24 / 11 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l1213_121333


namespace NUMINAMATH_CALUDE_horner_v3_equals_108_l1213_121346

def horner_v (coeffs : List ℝ) (x : ℝ) : List ℝ :=
  coeffs.scanl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 6 * x + 7

theorem horner_v3_equals_108 :
  let coeffs := [2, -5, -4, 3, -6, 7]
  let x := 5
  let v := horner_v coeffs x
  v[3] = 108 := by sorry

end NUMINAMATH_CALUDE_horner_v3_equals_108_l1213_121346


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_wire_length_is_15840_l1213_121302

/-- The length of wire required to go 15 times around a square field with area 69696 m² -/
theorem wire_length_around_square_field : ℝ :=
  let field_area : ℝ := 69696
  let side_length : ℝ := Real.sqrt field_area
  let perimeter : ℝ := 4 * side_length
  let num_rounds : ℝ := 15
  num_rounds * perimeter

/-- Proof that the wire length is 15840 m -/
theorem wire_length_is_15840 : wire_length_around_square_field = 15840 := by
  sorry


end NUMINAMATH_CALUDE_wire_length_around_square_field_wire_length_is_15840_l1213_121302


namespace NUMINAMATH_CALUDE_license_plate_count_l1213_121324

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of consonants in the alphabet -/
def consonant_count : ℕ := 21

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 5

/-- The number of odd digits -/
def odd_digit_count : ℕ := 5

/-- The number of even digits -/
def even_digit_count : ℕ := 5

/-- The total number of possible license plates -/
def total_plates : ℕ := alphabet_size * consonant_count * vowel_count * odd_digit_count * odd_digit_count * even_digit_count * even_digit_count

theorem license_plate_count : total_plates = 1706250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1213_121324


namespace NUMINAMATH_CALUDE_floor_problem_solution_exists_and_unique_floor_length_satisfies_conditions_l1213_121363

/-- Represents the dimensions and costs of a rectangular floor with a painted border. -/
structure FloorProblem where
  breadth : ℝ
  length_ratio : ℝ
  floor_paint_rate : ℝ
  floor_paint_cost : ℝ
  border_paint_rate : ℝ
  total_paint_cost : ℝ

/-- The main theorem stating the existence and uniqueness of a solution to the floor problem. -/
theorem floor_problem_solution_exists_and_unique :
  ∃! (fp : FloorProblem),
    fp.length_ratio = 3 ∧
    fp.floor_paint_rate = 3.00001 ∧
    fp.floor_paint_cost = 361 ∧
    fp.border_paint_rate = 15 ∧
    fp.total_paint_cost = 500 ∧
    fp.floor_paint_rate * (fp.length_ratio * fp.breadth * fp.breadth) = fp.floor_paint_cost ∧
    fp.border_paint_rate * (2 * (fp.length_ratio * fp.breadth + fp.breadth)) + fp.floor_paint_cost = fp.total_paint_cost :=
  sorry

/-- Function to calculate the length of the floor given a FloorProblem instance. -/
def calculate_floor_length (fp : FloorProblem) : ℝ :=
  fp.length_ratio * fp.breadth

/-- Theorem stating that the calculated floor length satisfies the problem conditions. -/
theorem floor_length_satisfies_conditions (fp : FloorProblem) :
  fp.length_ratio = 3 →
  fp.floor_paint_rate = 3.00001 →
  fp.floor_paint_cost = 361 →
  fp.border_paint_rate = 15 →
  fp.total_paint_cost = 500 →
  fp.floor_paint_rate * (fp.length_ratio * fp.breadth * fp.breadth) = fp.floor_paint_cost →
  fp.border_paint_rate * (2 * (fp.length_ratio * fp.breadth + fp.breadth)) + fp.floor_paint_cost = fp.total_paint_cost →
  ∃ (length : ℝ), length = calculate_floor_length fp :=
  sorry

end NUMINAMATH_CALUDE_floor_problem_solution_exists_and_unique_floor_length_satisfies_conditions_l1213_121363


namespace NUMINAMATH_CALUDE_instrument_probability_l1213_121331

theorem instrument_probability (total : ℕ) (at_least_one : ℕ) (two_or_more : ℕ) : 
  total = 800 →
  at_least_one = total / 5 →
  two_or_more = 32 →
  (at_least_one - two_or_more : ℚ) / total = 16 / 100 := by
  sorry

end NUMINAMATH_CALUDE_instrument_probability_l1213_121331


namespace NUMINAMATH_CALUDE_work_duration_l1213_121361

/-- Given workers A and B with their individual work rates and the time B takes to finish after A leaves,
    prove that A and B worked together for 2 days. -/
theorem work_duration (a_rate b_rate : ℚ) (b_finish_time : ℚ) : 
  a_rate = 1/4 →
  b_rate = 1/10 →
  b_finish_time = 3 →
  ∃ (x : ℚ), x = 2 ∧ (a_rate + b_rate) * x + b_rate * b_finish_time = 1 := by
  sorry

#eval (1/4 : ℚ) + (1/10 : ℚ)  -- Combined work rate
#eval ((1/4 : ℚ) + (1/10 : ℚ)) * 2 + (1/10 : ℚ) * 3  -- Total work done

end NUMINAMATH_CALUDE_work_duration_l1213_121361


namespace NUMINAMATH_CALUDE_max_value_theorem_l1213_121375

theorem max_value_theorem (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h : (a₂ - a₁)^2 + (a₃ - a₂)^2 + (a₄ - a₃)^2 + (a₅ - a₄)^2 + (a₆ - a₅)^2 = 1) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ 
  ∀ (x : ℝ), x = (a₅ + a₆) - (a₁ + a₄) → x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1213_121375


namespace NUMINAMATH_CALUDE_divisibility_of_concatenated_integers_l1213_121332

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of concatenating integers from 1 to n
  sorry

theorem divisibility_of_concatenated_integers :
  ∃ M : ℕ, M = concatenate_integers 50 ∧ M % 51 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_concatenated_integers_l1213_121332


namespace NUMINAMATH_CALUDE_optimal_circle_radii_equilateral_triangle_l1213_121344

/-- Given an equilateral triangle with side length 1, this theorem states that
    the maximum area covered by three circles centered at the vertices,
    not intersecting each other or the opposite sides, is achieved when
    the radii are R_a = √3/2 and R_b = R_c = 1 - √3/2. -/
theorem optimal_circle_radii_equilateral_triangle :
  let side_length : ℝ := 1
  let height : ℝ := Real.sqrt 3 / 2
  let R_a : ℝ := height
  let R_b : ℝ := 1 - height
  let R_c : ℝ := 1 - height
  let area_covered (r_a r_b r_c : ℝ) : ℝ := π / 6 * (r_a^2 + r_b^2 + r_c^2)
  let is_valid_radii (r_a r_b r_c : ℝ) : Prop :=
    r_a ≤ height ∧ r_a ≥ 1/2 ∧
    r_b ≤ 1 - r_a ∧ r_c ≤ 1 - r_a
  ∀ r_a r_b r_c : ℝ,
    is_valid_radii r_a r_b r_c →
    area_covered r_a r_b r_c ≤ area_covered R_a R_b R_c :=
by sorry

end NUMINAMATH_CALUDE_optimal_circle_radii_equilateral_triangle_l1213_121344


namespace NUMINAMATH_CALUDE_least_integer_in_ratio_l1213_121353

theorem least_integer_in_ratio (a b c : ℕ+) : 
  (a : ℚ) + (b : ℚ) + (c : ℚ) = 90 →
  (b : ℚ) = 2 * (a : ℚ) →
  (c : ℚ) = 5 * (a : ℚ) →
  (a : ℚ) = 45 / 4 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_in_ratio_l1213_121353


namespace NUMINAMATH_CALUDE_no_statements_imply_negation_l1213_121323

theorem no_statements_imply_negation (p q : Prop) : 
  ¬((p ∨ q) → ¬(p ∨ q)) ∧
  ¬((p ∨ ¬q) → ¬(p ∨ q)) ∧
  ¬((¬p ∨ q) → ¬(p ∨ q)) ∧
  ¬((¬p ∧ q) → ¬(p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_no_statements_imply_negation_l1213_121323


namespace NUMINAMATH_CALUDE_three_men_five_jobs_earnings_l1213_121381

/-- Calculates the total earnings for a group of workers completing multiple jobs -/
def totalEarnings (numWorkers : ℕ) (numJobs : ℕ) (hourlyRate : ℕ) (hoursPerJob : ℕ) : ℕ :=
  numWorkers * numJobs * hourlyRate * hoursPerJob

/-- Proves that 3 men working on 5 jobs at $10 per hour, with each job taking 1 hour, earn $150 in total -/
theorem three_men_five_jobs_earnings :
  totalEarnings 3 5 10 1 = 150 := by
  sorry

end NUMINAMATH_CALUDE_three_men_five_jobs_earnings_l1213_121381


namespace NUMINAMATH_CALUDE_twice_not_equal_squared_l1213_121318

theorem twice_not_equal_squared (m : ℝ) : 2 * m ≠ m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_twice_not_equal_squared_l1213_121318


namespace NUMINAMATH_CALUDE_recycling_money_calculation_l1213_121378

/-- Calculates the total money received from recycling cans and newspapers. -/
def recycling_money (can_rate : ℚ) (newspaper_rate : ℚ) (cans : ℕ) (newspapers : ℕ) : ℚ :=
  (can_rate * (cans / 12 : ℚ)) + (newspaper_rate * (newspapers / 5 : ℚ))

/-- Theorem: Given the recycling rates and the family's collection, the total money received is $12. -/
theorem recycling_money_calculation :
  recycling_money (1/2) (3/2) 144 20 = 12 := by
  sorry

end NUMINAMATH_CALUDE_recycling_money_calculation_l1213_121378


namespace NUMINAMATH_CALUDE_cat_food_finished_l1213_121365

def daily_consumption : ℚ := 1/4 + 1/6

def total_cans : ℕ := 10

def days_to_finish : ℕ := 15

theorem cat_food_finished :
  (daily_consumption * days_to_finish : ℚ) ≥ total_cans ∧
  (daily_consumption * (days_to_finish - 1) : ℚ) < total_cans := by
  sorry

end NUMINAMATH_CALUDE_cat_food_finished_l1213_121365


namespace NUMINAMATH_CALUDE_matrix_power_equality_l1213_121396

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 2, 1]

def B : Matrix (Fin 2) (Fin 2) ℕ := !![17, 12; 24, 17]

theorem matrix_power_equality :
  A^10 = B^5 := by sorry

end NUMINAMATH_CALUDE_matrix_power_equality_l1213_121396


namespace NUMINAMATH_CALUDE_absolute_difference_of_product_and_sum_l1213_121366

theorem absolute_difference_of_product_and_sum (m n : ℝ) 
  (h1 : m * n = 8) (h2 : m + n = 6) : |m - n| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_product_and_sum_l1213_121366


namespace NUMINAMATH_CALUDE_scientific_notation_of_113700_l1213_121336

theorem scientific_notation_of_113700 :
  (113700 : ℝ) = 1.137 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_113700_l1213_121336


namespace NUMINAMATH_CALUDE_bottles_per_case_is_ten_l1213_121313

/-- The number of bottles produced per day -/
def bottles_per_day : ℕ := 72000

/-- The number of cases required for daily production -/
def cases_per_day : ℕ := 7200

/-- The number of bottles that a case can hold -/
def bottles_per_case : ℕ := bottles_per_day / cases_per_day

theorem bottles_per_case_is_ten : bottles_per_case = 10 := by
  sorry

end NUMINAMATH_CALUDE_bottles_per_case_is_ten_l1213_121313


namespace NUMINAMATH_CALUDE_ab_difference_l1213_121309

theorem ab_difference (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_difference_l1213_121309


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l1213_121348

theorem coconut_grove_problem (x : ℝ) : 
  (((x + 2) * 40 + x * 120 + (x - 2) * 180) / (3 * x) = 100) → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l1213_121348


namespace NUMINAMATH_CALUDE_symmetry_implies_linear_plus_periodic_l1213_121306

/-- A function has two centers of symmetry if there exist two distinct points
    such that reflecting the graph through these points leaves it unchanged. -/
def has_two_centers_of_symmetry (f : ℝ → ℝ) : Prop :=
  ∃ (C₁ C₂ : ℝ × ℝ), C₁ ≠ C₂ ∧
  ∀ (x y : ℝ), f y = x ↔ f (2 * C₁.1 - y) = 2 * C₁.2 - x ∧
                      f (2 * C₂.1 - y) = 2 * C₂.2 - x

/-- A function is the sum of a linear function and a periodic function if
    there exist real numbers b and a ≠ 0, and a periodic function g with period a,
    such that f(x) = bx + g(x) for all x. -/
def is_sum_of_linear_and_periodic (f : ℝ → ℝ) : Prop :=
  ∃ (b : ℝ) (a : ℝ) (g : ℝ → ℝ), a ≠ 0 ∧
  (∀ x, g (x + a) = g x) ∧
  (∀ x, f x = b * x + g x)

/-- Theorem: If a function has two centers of symmetry,
    then it can be expressed as the sum of a linear function and a periodic function. -/
theorem symmetry_implies_linear_plus_periodic (f : ℝ → ℝ) :
  has_two_centers_of_symmetry f → is_sum_of_linear_and_periodic f := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_linear_plus_periodic_l1213_121306


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l1213_121334

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def tens_digit_less_than_5 (n : ℕ) : Prop := (n / 10) % 10 < 5

def divisible_by_its_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ 0 ∧ tens ≠ 0 ∧ ones ≠ 0 ∧
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones ∧
  n % hundreds = 0 ∧ n % tens = 0 ∧ n % ones = 0

theorem largest_three_digit_divisible_by_digits :
  ∀ n : ℕ, is_three_digit n →
    tens_digit_less_than_5 n →
    divisible_by_its_digits n →
    n ≤ 936 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l1213_121334


namespace NUMINAMATH_CALUDE_sum_digits_base_8_999_l1213_121338

def base_8_representation (n : ℕ) : List ℕ := sorry

theorem sum_digits_base_8_999 : 
  (base_8_representation 999).sum = 19 := by sorry

end NUMINAMATH_CALUDE_sum_digits_base_8_999_l1213_121338


namespace NUMINAMATH_CALUDE_expression_evaluation_l1213_121391

theorem expression_evaluation : 
  let f (x : ℝ) := (x + 1) / (x - 1)
  let g (x : ℝ) := (f x + 1) / (f x - 1)
  g (1/2) = -3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1213_121391


namespace NUMINAMATH_CALUDE_isosceles_triangles_remainder_l1213_121312

/-- The number of vertices in the regular polygon --/
def n : ℕ := 2019

/-- The number of isosceles triangles in a regular n-gon --/
def num_isosceles (n : ℕ) : ℕ := (n * (n - 1) / 2 : ℕ) - (2 * n / 3 : ℕ)

/-- The theorem stating that the remainder when the number of isosceles triangles
    in a regular 2019-gon is divided by 100 is equal to 25 --/
theorem isosceles_triangles_remainder :
  num_isosceles n % 100 = 25 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_remainder_l1213_121312


namespace NUMINAMATH_CALUDE_work_left_after_collaboration_l1213_121384

/-- Represents the fraction of work left after two workers collaborate for a given number of days. -/
def work_left (a_days b_days collab_days : ℕ) : ℚ :=
  1 - (collab_days : ℚ) * (1 / a_days + 1 / b_days)

/-- Theorem stating that if A can complete the work in 15 days and B in 20 days,
    then after working together for 4 days, 8/15 of the work is left. -/
theorem work_left_after_collaboration :
  work_left 15 20 4 = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_work_left_after_collaboration_l1213_121384


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l1213_121374

theorem walking_rate_ratio (usual_time new_time usual_rate new_rate : ℝ) 
  (h1 : usual_time = 49)
  (h2 : new_time = usual_time - 7)
  (h3 : usual_rate * usual_time = new_rate * new_time) :
  new_rate / usual_rate = 7 / 6 := by
sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l1213_121374


namespace NUMINAMATH_CALUDE_sin_25pi_div_6_l1213_121370

theorem sin_25pi_div_6 : Real.sin (25 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_25pi_div_6_l1213_121370


namespace NUMINAMATH_CALUDE_fraction_simplification_l1213_121316

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1213_121316


namespace NUMINAMATH_CALUDE_quilt_shaded_half_l1213_121367

/-- Represents a square quilt composed of unit squares -/
structure Quilt :=
  (size : ℕ)
  (shaded_rows : ℕ)

/-- The fraction of the quilt that is shaded -/
def shaded_fraction (q : Quilt) : ℚ :=
  q.shaded_rows / q.size

/-- Theorem: For a 4x4 quilt with 2 shaded rows, the shaded fraction is 1/2 -/
theorem quilt_shaded_half (q : Quilt) 
  (h1 : q.size = 4) 
  (h2 : q.shaded_rows = 2) : 
  shaded_fraction q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quilt_shaded_half_l1213_121367


namespace NUMINAMATH_CALUDE_perfect_linear_correlation_l1213_121360

/-- A scatter plot where all points fall on a straight line -/
structure PerfectLinearScatterPlot where
  /-- The slope of the line (non-zero real number) -/
  slope : ℝ
  /-- Assumption that the slope is non-zero -/
  slope_nonzero : slope ≠ 0

/-- The correlation coefficient R^2 for a scatter plot -/
def correlation_coefficient (plot : PerfectLinearScatterPlot) : ℝ := sorry

/-- Theorem: The correlation coefficient R^2 is 1 for a perfect linear scatter plot -/
theorem perfect_linear_correlation 
  (plot : PerfectLinearScatterPlot) : 
  correlation_coefficient plot = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_linear_correlation_l1213_121360


namespace NUMINAMATH_CALUDE_local_minimum_implies_b_range_l1213_121389

def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

theorem local_minimum_implies_b_range (b : ℝ) :
  (∃ x₀ ∈ Set.Ioo 0 1, IsLocalMin (f b) x₀) →
  0 < b ∧ b < 1 := by
sorry

end NUMINAMATH_CALUDE_local_minimum_implies_b_range_l1213_121389


namespace NUMINAMATH_CALUDE_ac_length_l1213_121387

/-- Given a line segment AB of length 4 with a point C on AB, 
    prove that if AC is the mean proportional between AB and BC, 
    then the length of AC is 2√5 - 2 -/
theorem ac_length (A B C : ℝ) (h1 : B - A = 4) (h2 : A ≤ C ∧ C ≤ B) 
  (h3 : (C - A)^2 = (B - A) * (B - C)) : C - A = 2 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_l1213_121387


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1213_121350

/-- The greatest possible distance between the centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 15)
  (h_height : rectangle_height = 10)
  (h_diameter : circle_diameter = 5)
  (h_nonneg_width : 0 ≤ rectangle_width)
  (h_nonneg_height : 0 ≤ rectangle_height)
  (h_nonneg_diameter : 0 ≤ circle_diameter)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = 5 * Real.sqrt 5 ∧
    ∀ (d' : ℝ), d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        circle_diameter / 2 ≤ x₁ ∧ x₁ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₁ ∧ y₁ ≤ rectangle_height - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ x₂ ∧ x₂ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₂ ∧ y₂ ≤ rectangle_height - circle_diameter / 2 ∧
        d' = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1213_121350


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l1213_121342

/-- Reflects a point (x, y) across the line y = -x -/
def reflect_across_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The original center of the circle -/
def original_center : ℝ × ℝ := (3, -7)

/-- The expected center after reflection -/
def expected_reflected_center : ℝ × ℝ := (7, -3)

theorem reflection_of_circle_center :
  reflect_across_y_neg_x original_center = expected_reflected_center :=
by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l1213_121342


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l1213_121307

theorem partial_fraction_sum (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l1213_121307


namespace NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l1213_121388

theorem unique_solution_ceiling_equation :
  ∃! c : ℝ, c + ⌈c⌉ = 23.2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l1213_121388


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l1213_121373

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_point_one :
  f' 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l1213_121373


namespace NUMINAMATH_CALUDE_work_completion_proof_l1213_121330

/-- The number of days Matt and Peter worked together -/
def days_worked_together : ℕ := 12

/-- The time (in days) it takes Matt and Peter to complete the work together -/
def time_together : ℕ := 20

/-- The time (in days) it takes Peter to complete the work alone -/
def time_peter_alone : ℕ := 35

/-- The time (in days) it takes Peter to complete the remaining work after Matt stops -/
def time_peter_remaining : ℕ := 14

theorem work_completion_proof :
  (days_worked_together : ℚ) / time_together + 
  time_peter_remaining / time_peter_alone = 1 := by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l1213_121330


namespace NUMINAMATH_CALUDE_function_composition_implies_sum_l1213_121315

/-- Given two functions f and g, where f(x) = ax + b and g(x) = 3x - 6,
    and the condition that g(f(x)) = 4x + 3 for all x,
    prove that a + b = 13/3 -/
theorem function_composition_implies_sum (a b : ℝ) :
  (∀ x, 3 * (a * x + b) - 6 = 4 * x + 3) →
  a + b = 13 / 3 := by
sorry

end NUMINAMATH_CALUDE_function_composition_implies_sum_l1213_121315


namespace NUMINAMATH_CALUDE_periodic_function_value_l1213_121301

theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 2008 = -1 → f 2009 = 1 := by sorry

end NUMINAMATH_CALUDE_periodic_function_value_l1213_121301


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l1213_121357

/-- Given collinear points A, B, C, D, and E with specified distances between them,
    this function calculates the sum of squared distances from these points to a point P on AE. -/
def sum_of_squared_distances (x : ℝ) : ℝ :=
  x^2 + (x - 2)^2 + (x - 4)^2 + (x - 7)^2 + (x - 11)^2

/-- The theorem states that the minimum value of the sum of squared distances
    from points A, B, C, D, and E to any point P on line segment AE is 54.8,
    given the specified distances between the points. -/
theorem min_sum_squared_distances :
  ∃ (min_value : ℝ), min_value = 54.8 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 11 → sum_of_squared_distances x ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l1213_121357


namespace NUMINAMATH_CALUDE_log_expression_simplification_l1213_121339

theorem log_expression_simplification 
  (a b c d x y z w : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) : 
  Real.log (a / b) + Real.log (b / c) + Real.log (c / d) - Real.log (a * y * z / (d * x * w)) = Real.log (x * w / (y * z)) := by
  sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l1213_121339


namespace NUMINAMATH_CALUDE_specific_bike_ride_north_distance_l1213_121310

/-- Represents a bike ride with given distances and final position -/
structure BikeRide where
  west : ℝ
  initialNorth : ℝ
  east : ℝ
  finalDistance : ℝ

/-- Calculates the final northward distance after going east for a given bike ride -/
def finalNorthDistance (ride : BikeRide) : ℝ :=
  sorry

/-- Theorem stating that for the specific bike ride described, the final northward distance after going east is 15 miles -/
theorem specific_bike_ride_north_distance :
  let ride : BikeRide := {
    west := 8,
    initialNorth := 5,
    east := 4,
    finalDistance := 20.396078054371138
  }
  finalNorthDistance ride = 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_bike_ride_north_distance_l1213_121310


namespace NUMINAMATH_CALUDE_starting_number_is_100_l1213_121305

/-- The starting number of a range ending at 400, where the average of the integers
    in this range is 100 greater than the average of the integers from 50 to 250. -/
def starting_number : ℤ :=
  let avg_50_to_250 := (50 + 250) / 2
  let avg_x_to_400 := avg_50_to_250 + 100
  2 * avg_x_to_400 - 400

theorem starting_number_is_100 : starting_number = 100 := by
  sorry

end NUMINAMATH_CALUDE_starting_number_is_100_l1213_121305


namespace NUMINAMATH_CALUDE_last_week_tv_hours_l1213_121317

/-- The number of hours of television watched last week -/
def last_week_hours : ℝ := sorry

/-- The average number of hours watched over three weeks -/
def average_hours : ℝ := 10

/-- The number of hours watched the week before last -/
def week_before_hours : ℝ := 8

/-- The number of hours to be watched next week -/
def next_week_hours : ℝ := 12

theorem last_week_tv_hours : last_week_hours = 10 :=
  by
    have h1 : (week_before_hours + last_week_hours + next_week_hours) / 3 = average_hours := by sorry
    sorry


end NUMINAMATH_CALUDE_last_week_tv_hours_l1213_121317


namespace NUMINAMATH_CALUDE_train_length_l1213_121385

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 27 → ∃ (length_m : ℝ), abs (length_m - 450.09) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1213_121385


namespace NUMINAMATH_CALUDE_can_distribution_properties_l1213_121354

/-- Represents the distribution of cans across bags -/
structure CanDistribution where
  total_cans : ℕ
  num_bags : ℕ
  first_bags_limit : ℕ
  last_bags_limit : ℕ

/-- Calculates the number of cans in each of the last bags -/
def cans_in_last_bags (d : CanDistribution) : ℕ :=
  let cans_in_first_bags := d.first_bags_limit * (d.num_bags / 2)
  let remaining_cans := d.total_cans - cans_in_first_bags
  remaining_cans / (d.num_bags / 2)

/-- Calculates the difference between cans in first and last bag -/
def cans_difference (d : CanDistribution) : ℕ :=
  d.first_bags_limit - cans_in_last_bags d

/-- Theorem stating the properties of the can distribution -/
theorem can_distribution_properties (d : CanDistribution) 
    (h1 : d.total_cans = 200)
    (h2 : d.num_bags = 6)
    (h3 : d.first_bags_limit = 40)
    (h4 : d.last_bags_limit = 30) :
    cans_in_last_bags d = 26 ∧ cans_difference d = 14 := by
  sorry

#eval cans_in_last_bags { total_cans := 200, num_bags := 6, first_bags_limit := 40, last_bags_limit := 30 }
#eval cans_difference { total_cans := 200, num_bags := 6, first_bags_limit := 40, last_bags_limit := 30 }

end NUMINAMATH_CALUDE_can_distribution_properties_l1213_121354


namespace NUMINAMATH_CALUDE_letter_cost_l1213_121390

/-- The cost to mail each letter, given the total cost, package cost, and number of letters and packages. -/
theorem letter_cost (total_cost package_cost : ℚ) (num_letters num_packages : ℕ) : 
  total_cost = 4.49 →
  package_cost = 0.88 →
  num_letters = 5 →
  num_packages = 3 →
  (num_letters : ℚ) * ((total_cost - (package_cost * (num_packages : ℚ))) / (num_letters : ℚ)) = 0.37 := by
  sorry

end NUMINAMATH_CALUDE_letter_cost_l1213_121390


namespace NUMINAMATH_CALUDE_complex_modulus_range_l1213_121386

theorem complex_modulus_range (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (x : ℝ), x = Complex.abs ((z - 2) * (z + 1)^2) ∧ 0 ≤ x ∧ x ≤ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l1213_121386


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1213_121304

theorem quadratic_function_property 
  (a c y₁ y₂ y₃ y₄ : ℝ) 
  (h_a : a < 0)
  (h_y₁ : y₁ = a * (-2)^2 - 4 * a * (-2) + c)
  (h_y₂ : y₂ = c)
  (h_y₃ : y₃ = a * 3^2 - 4 * a * 3 + c)
  (h_y₄ : y₄ = a * 5^2 - 4 * a * 5 + c)
  (h_y₂y₄ : y₂ * y₄ < 0) :
  y₁ * y₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1213_121304


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1213_121376

theorem inequality_system_solution (m : ℝ) : 
  (∀ x, (x + 5 < 4*x - 1 ∧ x > m) ↔ x > 2) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1213_121376


namespace NUMINAMATH_CALUDE_car_tank_capacity_l1213_121326

/-- Calculates the capacity of a car's gas tank given initial and final mileage, efficiency, and number of fill-ups -/
def tank_capacity (initial_mileage final_mileage : ℕ) (efficiency : ℚ) (fill_ups : ℕ) : ℚ :=
  (final_mileage - initial_mileage : ℚ) / (efficiency * fill_ups)

/-- Proves that the car's tank capacity is 20 gallons given the problem conditions -/
theorem car_tank_capacity :
  tank_capacity 1728 2928 30 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_tank_capacity_l1213_121326


namespace NUMINAMATH_CALUDE_sqrt_17_property_l1213_121308

theorem sqrt_17_property (a b : ℝ) : 
  (∀ x : ℤ, (x : ℝ) ≤ Real.sqrt 17 → (x + 1 : ℝ) > Real.sqrt 17 → a = x) →
  b = Real.sqrt 17 - a →
  b ^ 2020 * (a + Real.sqrt 17) ^ 2021 = Real.sqrt 17 + 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_17_property_l1213_121308


namespace NUMINAMATH_CALUDE_cricketer_average_score_l1213_121340

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_set_matches : ℕ) 
  (second_set_matches : ℕ) 
  (first_set_average : ℝ) 
  (second_set_average : ℝ) 
  (h1 : total_matches = first_set_matches + second_set_matches)
  (h2 : total_matches = 5)
  (h3 : first_set_matches = 2)
  (h4 : second_set_matches = 3)
  (h5 : first_set_average = 40)
  (h6 : second_set_average = 10) :
  (first_set_matches * first_set_average + second_set_matches * second_set_average) / total_matches = 22 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l1213_121340


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l1213_121372

theorem modular_inverse_of_5_mod_31 :
  ∃ x : ℕ, x < 31 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l1213_121372


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l1213_121329

/-- The volume of a sphere inscribed in a right circular cone with specific properties -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 24) : 
  let r := d / 4
  4 / 3 * π * r^3 = 288 * π := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l1213_121329


namespace NUMINAMATH_CALUDE_solution_of_system_l1213_121379

theorem solution_of_system (x y : ℚ) 
  (eq1 : 3 * y - 4 * x = 8)
  (eq2 : 2 * y + x = -1) : 
  x = -19/11 ∧ y = 4/11 := by
sorry

end NUMINAMATH_CALUDE_solution_of_system_l1213_121379


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1213_121395

theorem sqrt_inequality (a : ℝ) (h : a > 5) :
  Real.sqrt (a - 5) - Real.sqrt (a - 3) < Real.sqrt (a - 2) - Real.sqrt a :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1213_121395


namespace NUMINAMATH_CALUDE_max_value_of_f_l1213_121325

-- Define the function
def f (x : ℝ) : ℝ := x * (1 - 3 * x)

-- State the theorem
theorem max_value_of_f :
  ∃ (max_y : ℝ), max_y = 1/12 ∧
  ∀ (x : ℝ), 0 < x → x < 1/3 → f x ≤ max_y :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1213_121325


namespace NUMINAMATH_CALUDE_divisibility_by_264_l1213_121341

theorem divisibility_by_264 (n : ℕ) (h : n > 1) :
  ∃ k : ℤ, (7 : ℤ)^(2*n) - (4 : ℤ)^(2*n) - 297 = 264 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_264_l1213_121341


namespace NUMINAMATH_CALUDE_marbles_given_to_eric_l1213_121351

def marble_redistribution (tyrone_initial : ℕ) (eric_initial : ℕ) (x : ℕ) : Prop :=
  (tyrone_initial - x) = 3 * (eric_initial + x)

theorem marbles_given_to_eric 
  (tyrone_initial : ℕ) (eric_initial : ℕ) (x : ℕ)
  (h1 : tyrone_initial = 120)
  (h2 : eric_initial = 20)
  (h3 : marble_redistribution tyrone_initial eric_initial x) :
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_eric_l1213_121351


namespace NUMINAMATH_CALUDE_zero_point_existence_l1213_121364

def f (x : ℝ) := x^3 + 2*x - 5

theorem zero_point_existence :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  have h1 : Continuous f := sorry
  have h2 : f 1 < 0 := sorry
  have h3 : f 2 > 0 := sorry
  sorry

end NUMINAMATH_CALUDE_zero_point_existence_l1213_121364


namespace NUMINAMATH_CALUDE_beehives_for_candles_l1213_121399

/-- Given that 3 beehives make enough wax for 12 candles, 
    prove that 24 hives are needed to make 96 candles. -/
theorem beehives_for_candles : 
  (3 : ℚ) * 96 / 12 = 24 := by sorry

end NUMINAMATH_CALUDE_beehives_for_candles_l1213_121399


namespace NUMINAMATH_CALUDE_fish_pond_population_l1213_121303

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 50 →
  second_catch = 50 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = initial_tagged / (initial_tagged * second_catch / tagged_in_second) :=
by
  sorry

#eval (50 * 50) / 2  -- Should evaluate to 1250

end NUMINAMATH_CALUDE_fish_pond_population_l1213_121303


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1213_121368

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1/2 →
  a 2 * a 4 = 4 * (a 3 - 1) →
  a 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1213_121368


namespace NUMINAMATH_CALUDE_final_rope_length_l1213_121398

/-- Represents the weekly rope transactions in feet -/
def weekly_transactions : List ℝ :=
  [6, 18, 14, -9, 8, -1, 3, -10]

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Calculates the total rope length in inches after all transactions -/
def total_rope_length : ℝ :=
  (weekly_transactions.sum * feet_to_inches)

theorem final_rope_length :
  total_rope_length = 348 := by sorry

end NUMINAMATH_CALUDE_final_rope_length_l1213_121398


namespace NUMINAMATH_CALUDE_max_servings_is_ten_l1213_121349

/-- Represents the number of chunks per serving for each fruit type -/
structure FruitRatio where
  cantaloupe : ℕ
  honeydew : ℕ
  pineapple : ℕ
  watermelon : ℕ

/-- Represents the available chunks of each fruit type -/
structure AvailableFruit where
  cantaloupe : ℕ
  honeydew : ℕ
  pineapple : ℕ
  watermelon : ℕ

/-- Calculates the maximum number of servings that can be made -/
def maxServings (ratio : FruitRatio) (available : AvailableFruit) : ℕ :=
  min
    (available.cantaloupe / ratio.cantaloupe)
    (min
      (available.honeydew / ratio.honeydew)
      (min
        (available.pineapple / ratio.pineapple)
        (available.watermelon / ratio.watermelon)))

/-- The given fruit ratio -/
def givenRatio : FruitRatio :=
  { cantaloupe := 3
  , honeydew := 2
  , pineapple := 1
  , watermelon := 4 }

/-- The available fruit chunks -/
def givenAvailable : AvailableFruit :=
  { cantaloupe := 30
  , honeydew := 42
  , pineapple := 12
  , watermelon := 56 }

theorem max_servings_is_ten :
  maxServings givenRatio givenAvailable = 10 := by
  sorry


end NUMINAMATH_CALUDE_max_servings_is_ten_l1213_121349


namespace NUMINAMATH_CALUDE_intercept_sum_l1213_121362

/-- A line is described by the equation y + 3 = -3(x - 5) -/
def line_equation (x y : ℝ) : Prop := y + 3 = -3 * (x - 5)

/-- The x-intercept of the line -/
def x_intercept : ℝ := 4

/-- The y-intercept of the line -/
def y_intercept : ℝ := 12

/-- The sum of x-intercept and y-intercept is 16 -/
theorem intercept_sum : x_intercept + y_intercept = 16 := by sorry

end NUMINAMATH_CALUDE_intercept_sum_l1213_121362


namespace NUMINAMATH_CALUDE_oak_trees_in_park_l1213_121393

theorem oak_trees_in_park (current_trees : ℕ) 
  (h1 : current_trees + 4 = 9) : current_trees = 5 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_in_park_l1213_121393


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1213_121311

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a < 0 ∧ -1 < b ∧ b < 0) →
  (a + a * b < 0) ∧
  ∃ (x y : ℝ), x + x * y < 0 ∧ ¬(x < 0 ∧ -1 < y ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1213_121311


namespace NUMINAMATH_CALUDE_equation_positive_root_l1213_121321

theorem equation_positive_root (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2 / (x - 1) - k / (1 - x) = 1)) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l1213_121321
