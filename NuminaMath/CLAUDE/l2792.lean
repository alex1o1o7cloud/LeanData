import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l2792_279274

/-- Theorem: For a rectangular plot with breadth 14 meters and length 10 meters greater than its breadth, the ratio of its area to its breadth is 24:1. -/
theorem rectangle_area_breadth_ratio :
  ∀ (length breadth area : ℝ),
  breadth = 14 →
  length = breadth + 10 →
  area = length * breadth →
  area / breadth = 24 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l2792_279274


namespace NUMINAMATH_CALUDE_snail_final_position_l2792_279294

/-- Represents the direction the snail is facing -/
inductive Direction
  | Up
  | Right
  | Down
  | Left

/-- Represents a position on the grid -/
structure Position where
  row : Nat
  col : Nat

/-- Represents the state of the snail -/
structure SnailState where
  pos : Position
  dir : Direction
  visited : Set Position

/-- The grid dimensions -/
def gridWidth : Nat := 300
def gridHeight : Nat := 50

/-- Check if a position is within the grid -/
def isValidPosition (p : Position) : Bool :=
  p.row >= 1 && p.row <= gridHeight && p.col >= 1 && p.col <= gridWidth

/-- Move the snail according to the rules -/
def moveSnail (state : SnailState) : SnailState :=
  sorry -- Implementation of snail movement logic

/-- The main theorem stating the final position of the snail -/
theorem snail_final_position :
  ∃ (finalState : SnailState),
    (∀ (p : Position), isValidPosition p → p ∈ finalState.visited) ∧
    finalState.pos = Position.mk 25 26 := by
  sorry


end NUMINAMATH_CALUDE_snail_final_position_l2792_279294


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2792_279219

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -4 * p.1 + 6}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 5 * p.1 - 3}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2792_279219


namespace NUMINAMATH_CALUDE_circle_equation_chord_length_implies_k_min_distance_l2792_279281

-- Define the circle
def circle_center : ℝ × ℝ := (3, -2)
def circle_radius : ℝ := 5

-- Define points A and B
def point_A : ℝ × ℝ := (-1, 1)
def point_B : ℝ × ℝ := (-2, -2)

-- Define the line l that the circle center lies on
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the chord line
def chord_line (k x y : ℝ) : Prop := k * x - y + 5 = 0

-- Define the line for minimum distance
def line_min_dist (x y : ℝ) : Prop := x - y + 5 = 0

-- Theorem statements
theorem circle_equation : 
  ∀ x y : ℝ, (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 ↔
  (x - 3)^2 + (y + 2)^2 = 25 := by sorry

theorem chord_length_implies_k :
  ∃ k : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ,
    chord_line k x₁ y₁ ∧ chord_line k x₂ y₂ ∧
    (x₁ - circle_center.1)^2 + (y₁ - circle_center.2)^2 = circle_radius^2 ∧
    (x₂ - circle_center.1)^2 + (y₂ - circle_center.2)^2 = circle_radius^2 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 64) →
  k = -20/21 := by sorry

theorem min_distance :
  ∀ P Q : ℝ × ℝ,
  ((P.1 - circle_center.1)^2 + (P.2 - circle_center.2)^2 = circle_radius^2) →
  line_min_dist Q.1 Q.2 →
  ∃ d : ℝ, d ≥ 5 * Real.sqrt 2 - 5 ∧
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ≥ d^2 := by sorry

end NUMINAMATH_CALUDE_circle_equation_chord_length_implies_k_min_distance_l2792_279281


namespace NUMINAMATH_CALUDE_complement_M_N_l2792_279238

def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}

theorem complement_M_N : M \ N = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_complement_M_N_l2792_279238


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_l2792_279268

theorem zeros_before_first_nonzero (n : ℕ) (m : ℕ) : 
  let fraction := 1 / (2^n * 5^m)
  let zeros := m - n
  zeros > 0 → zeros = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_l2792_279268


namespace NUMINAMATH_CALUDE_square_of_sum_eleven_five_l2792_279263

theorem square_of_sum_eleven_five : 11^2 + 2*(11*5) + 5^2 = 256 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_eleven_five_l2792_279263


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l2792_279299

theorem greatest_three_digit_multiple_of_23 : 
  (∀ n : ℕ, n ≤ 999 ∧ n ≥ 100 ∧ 23 ∣ n → n ≤ 989) ∧ 
  989 ≤ 999 ∧ 989 ≥ 100 ∧ 23 ∣ 989 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l2792_279299


namespace NUMINAMATH_CALUDE_sweets_expenditure_l2792_279244

theorem sweets_expenditure (initial_amount : ℝ) (amount_per_friend : ℝ) (num_friends : ℕ) :
  initial_amount = 10.50 →
  amount_per_friend = 3.40 →
  num_friends = 2 →
  initial_amount - (amount_per_friend * num_friends) = 3.70 :=
by sorry

end NUMINAMATH_CALUDE_sweets_expenditure_l2792_279244


namespace NUMINAMATH_CALUDE_complex_number_sum_l2792_279266

theorem complex_number_sum (z : ℂ) 
  (h : 16 * Complex.abs z ^ 2 = 3 * Complex.abs (z + 1) ^ 2 + Complex.abs (z ^ 2 - 2) ^ 2 + 43) : 
  z + 8 / z = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_sum_l2792_279266


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2792_279213

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  a^2 + b^2 = c^2 →  -- right-angled triangle (Pythagorean theorem)
  a^2 + b^2 + c^2 = 1800 →  -- sum of squares of all sides
  c = 30 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2792_279213


namespace NUMINAMATH_CALUDE_ab_power_2022_l2792_279280

theorem ab_power_2022 (a b : ℝ) (h : |3*a + 1| + (b - 3)^2 = 0) : (a*b)^2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_power_2022_l2792_279280


namespace NUMINAMATH_CALUDE_excellent_chinese_or_math_excellent_all_subjects_l2792_279239

def excellent_chinese : Finset ℕ := sorry
def excellent_math : Finset ℕ := sorry
def excellent_english : Finset ℕ := sorry

axiom total_excellent : (excellent_chinese ∪ excellent_math ∪ excellent_english).card = 18
axiom chinese_count : excellent_chinese.card = 9
axiom math_count : excellent_math.card = 11
axiom english_count : excellent_english.card = 8
axiom chinese_math_count : (excellent_chinese ∩ excellent_math).card = 5
axiom math_english_count : (excellent_math ∩ excellent_english).card = 3
axiom chinese_english_count : (excellent_chinese ∩ excellent_english).card = 4

theorem excellent_chinese_or_math : 
  (excellent_chinese ∪ excellent_math).card = 15 := by sorry

theorem excellent_all_subjects : 
  (excellent_chinese ∩ excellent_math ∩ excellent_english).card = 2 := by sorry

end NUMINAMATH_CALUDE_excellent_chinese_or_math_excellent_all_subjects_l2792_279239


namespace NUMINAMATH_CALUDE_digit_2500_is_8_l2792_279248

/-- The number of digits in the representation of positive integers from 1 to n -/
def digitCount (n : ℕ) : ℕ := sorry

/-- The nth digit in the concatenation of integers from 1 to 1099 -/
def nthDigit (n : ℕ) : ℕ := sorry

theorem digit_2500_is_8 : nthDigit 2500 = 8 := by sorry

end NUMINAMATH_CALUDE_digit_2500_is_8_l2792_279248


namespace NUMINAMATH_CALUDE_banana_arrangements_l2792_279269

theorem banana_arrangements : 
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  (total_letters.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l2792_279269


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2792_279273

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - 3 < 0) ↔ k ∈ Set.Ioc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2792_279273


namespace NUMINAMATH_CALUDE_combined_molecular_weight_l2792_279208

/-- Atomic weight of Carbon in g/mol -/
def carbon_weight : Float := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : Float := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def oxygen_weight : Float := 16.00

/-- Molecular weight of Butanoic acid (C4H8O2) in g/mol -/
def butanoic_weight : Float :=
  4 * carbon_weight + 8 * hydrogen_weight + 2 * oxygen_weight

/-- Molecular weight of Propanoic acid (C3H6O2) in g/mol -/
def propanoic_weight : Float :=
  3 * carbon_weight + 6 * hydrogen_weight + 2 * oxygen_weight

/-- Number of moles of Butanoic acid in the mixture -/
def butanoic_moles : Float := 9

/-- Number of moles of Propanoic acid in the mixture -/
def propanoic_moles : Float := 5

/-- Theorem: The combined molecular weight of the mixture is 1163.326 grams -/
theorem combined_molecular_weight :
  butanoic_moles * butanoic_weight + propanoic_moles * propanoic_weight = 1163.326 := by
  sorry

end NUMINAMATH_CALUDE_combined_molecular_weight_l2792_279208


namespace NUMINAMATH_CALUDE_grid_sum_bottom_corners_l2792_279277

/-- Represents a 3x3 grid where each cell contains a number -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Checks if a given number appears exactly once in each row -/
def rowValid (g : Grid) (n : Nat) : Prop :=
  ∀ i : Fin 3, ∃! j : Fin 3, g i j = n

/-- Checks if a given number appears exactly once in each column -/
def colValid (g : Grid) (n : Nat) : Prop :=
  ∀ j : Fin 3, ∃! i : Fin 3, g i j = n

/-- Checks if the grid contains only the numbers 4, 5, and 6 -/
def gridContainsOnly456 (g : Grid) : Prop :=
  ∀ i j : Fin 3, g i j = 4 ∨ g i j = 5 ∨ g i j = 6

/-- The main theorem statement -/
theorem grid_sum_bottom_corners (g : Grid) :
  rowValid g 4 ∧ rowValid g 5 ∧ rowValid g 6 ∧
  colValid g 4 ∧ colValid g 5 ∧ colValid g 6 ∧
  gridContainsOnly456 g ∧
  g 0 0 = 5 ∧ g 1 1 = 4 →
  g 2 0 + g 2 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_grid_sum_bottom_corners_l2792_279277


namespace NUMINAMATH_CALUDE_perfect_square_from_sqrt_l2792_279253

theorem perfect_square_from_sqrt (n : ℤ) :
  ∃ (m : ℤ), m = 2 + 2 * Real.sqrt (28 * n^2 + 1) → ∃ (k : ℤ), m = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_from_sqrt_l2792_279253


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_225_with_ones_and_zeros_l2792_279231

def is_composed_of_ones_and_zeros (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1 ∨ d = 0

def smallest_divisible_by_225_with_ones_and_zeros : ℕ := 11111111100

theorem smallest_number_divisible_by_225_with_ones_and_zeros :
  (smallest_divisible_by_225_with_ones_and_zeros % 225 = 0) ∧
  is_composed_of_ones_and_zeros smallest_divisible_by_225_with_ones_and_zeros ∧
  ∀ n : ℕ, n < smallest_divisible_by_225_with_ones_and_zeros →
    ¬(n % 225 = 0 ∧ is_composed_of_ones_and_zeros n) :=
by sorry

#eval smallest_divisible_by_225_with_ones_and_zeros

end NUMINAMATH_CALUDE_smallest_number_divisible_by_225_with_ones_and_zeros_l2792_279231


namespace NUMINAMATH_CALUDE_positive_integers_relation_l2792_279297

theorem positive_integers_relation (a b : ℕ) : 
  a > 0 → b > 0 → (a, b) ≠ (1, 1) → (a * b - 1) ∣ (a^2 + b^2) → a^2 + b^2 = 5 * a * b - 5 := by
  sorry

end NUMINAMATH_CALUDE_positive_integers_relation_l2792_279297


namespace NUMINAMATH_CALUDE_odd_function_sum_l2792_279224

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_domain : ∀ x ∈ [-3, 3], f x = f x) (h_value : f 3 = -2) :
  f (-3) + f 0 = 2 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_sum_l2792_279224


namespace NUMINAMATH_CALUDE_exponent_division_l2792_279260

theorem exponent_division (a : ℝ) : a^7 / a^4 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2792_279260


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2792_279247

theorem quadratic_one_solution (q : ℝ) : 
  (q ≠ 0 ∧ ∃! x : ℝ, q * x^2 - 16 * x + 9 = 0) ↔ q = 64/9 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2792_279247


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2792_279292

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equality : i / (2 + i) = (1 + 2*i) / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2792_279292


namespace NUMINAMATH_CALUDE_dihedral_angle_range_l2792_279229

/-- The dihedral angle between two adjacent lateral faces of a regular n-sided pyramid -/
def dihedral_angle (n : ℕ) (h : ℝ) : ℝ :=
  sorry

/-- The internal angle of a regular n-sided polygon -/
def internal_angle (n : ℕ) : ℝ :=
  sorry

theorem dihedral_angle_range (n : ℕ) (h : ℝ) :
  0 < dihedral_angle n h ∧ dihedral_angle n h < π :=
by sorry

end NUMINAMATH_CALUDE_dihedral_angle_range_l2792_279229


namespace NUMINAMATH_CALUDE_sam_dimes_l2792_279223

/-- The number of dimes Sam has after receiving more from his dad -/
def total_dimes (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Proof that Sam has 16 dimes after receiving more from his dad -/
theorem sam_dimes : total_dimes 9 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sam_dimes_l2792_279223


namespace NUMINAMATH_CALUDE_Q_characterization_l2792_279242

def Ω : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 2008}

def superior (p q : ℝ × ℝ) : Prop := p.1 ≤ q.1 ∧ p.2 ≥ q.2

def Q : Set (ℝ × ℝ) := {q ∈ Ω | ∀ p ∈ Ω, superior p q → p = q}

theorem Q_characterization : Q = {p ∈ Ω | p.1^2 + p.2^2 = 2008 ∧ p.1 ≤ 0 ∧ p.2 ≥ 0} := by sorry

end NUMINAMATH_CALUDE_Q_characterization_l2792_279242


namespace NUMINAMATH_CALUDE_math_score_calculation_math_score_is_75_l2792_279296

theorem math_score_calculation (avg_four : ℝ) (drop : ℝ) : ℝ :=
  let total_four := 4 * avg_four
  let avg_five := avg_four - drop
  let total_five := 5 * avg_five
  total_five - total_four

theorem math_score_is_75 :
  math_score_calculation 90 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_math_score_calculation_math_score_is_75_l2792_279296


namespace NUMINAMATH_CALUDE_magnitude_BC_l2792_279278

/-- Given two points A and C in ℝ², and a vector AB, prove that the magnitude of BC is √29 -/
theorem magnitude_BC (A C B : ℝ × ℝ) (h1 : A = (2, -1)) (h2 : C = (0, 2)) 
  (h3 : B.1 - A.1 = 3 ∧ B.2 - A.2 = 5) : 
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = Real.sqrt 29 := by
  sorry

#check magnitude_BC

end NUMINAMATH_CALUDE_magnitude_BC_l2792_279278


namespace NUMINAMATH_CALUDE_profit_share_calculation_l2792_279222

/-- Calculates the share of profit for an investor given the investments and durations -/
def calculate_profit_share (a_investment : ℚ) (b_investment : ℚ) (a_duration : ℚ) (b_duration : ℚ) (total_profit : ℚ) : ℚ :=
  let a_investment_time := a_investment * a_duration
  let b_investment_time := b_investment * b_duration
  let total_investment_time := a_investment_time + b_investment_time
  let a_ratio := a_investment_time / total_investment_time
  a_ratio * total_profit

theorem profit_share_calculation (a_investment b_investment a_duration b_duration total_profit : ℚ) :
  calculate_profit_share a_investment b_investment a_duration b_duration total_profit =
  total_profit * (a_investment * a_duration) / (a_investment * a_duration + b_investment * b_duration) :=
by sorry

end NUMINAMATH_CALUDE_profit_share_calculation_l2792_279222


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l2792_279254

theorem ice_cream_sundaes (n : ℕ) (k : ℕ) : n = 8 ∧ k = 2 → Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l2792_279254


namespace NUMINAMATH_CALUDE_lcm_three_consecutive_naturals_l2792_279201

theorem lcm_three_consecutive_naturals (n : ℕ) :
  let lcm := Nat.lcm (Nat.lcm n (n + 1)) (n + 2)
  lcm = if Even (n + 1) then n * (n + 1) * (n + 2)
        else (n * (n + 1) * (n + 2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_lcm_three_consecutive_naturals_l2792_279201


namespace NUMINAMATH_CALUDE_tan_half_sum_angles_l2792_279211

theorem tan_half_sum_angles (x y : ℝ) 
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 8/17) : 
  Real.tan ((x + y)/2) = 40/51 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_angles_l2792_279211


namespace NUMINAMATH_CALUDE_class_gathering_problem_l2792_279225

theorem class_gathering_problem (male_students : ℕ) (female_students : ℕ) :
  female_students = male_students + 6 →
  (female_students : ℚ) / (male_students + female_students) = 2 / 3 →
  male_students + female_students = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_class_gathering_problem_l2792_279225


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2792_279264

-- Define the complex number z
def z : ℂ := sorry

-- State the given condition
axiom z_condition : (1 - Complex.I) * z = 2 * Complex.I

-- Define the second quadrant
def second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

-- Theorem statement
theorem z_in_second_quadrant : second_quadrant z := by sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2792_279264


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2792_279256

theorem container_volume_ratio : 
  ∀ (volume_container1 volume_container2 : ℚ),
  volume_container1 > 0 →
  volume_container2 > 0 →
  (4 / 5 : ℚ) * volume_container1 = (2 / 3 : ℚ) * volume_container2 →
  volume_container1 / volume_container2 = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2792_279256


namespace NUMINAMATH_CALUDE_two_sqrt_three_in_set_l2792_279284

theorem two_sqrt_three_in_set : 2 * Real.sqrt 3 ∈ {x : ℝ | x < 4} := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_three_in_set_l2792_279284


namespace NUMINAMATH_CALUDE_opposite_of_sin_60_degrees_l2792_279214

theorem opposite_of_sin_60_degrees : 
  -(Real.sin (π / 3)) = -(Real.sqrt 3 / 2) := by sorry

end NUMINAMATH_CALUDE_opposite_of_sin_60_degrees_l2792_279214


namespace NUMINAMATH_CALUDE_abc_remainder_mod_9_l2792_279203

theorem abc_remainder_mod_9 (a b c : ℕ) : 
  a < 9 → b < 9 → c < 9 →
  (a + 2*b + 3*c) % 9 = 1 →
  (2*a + 3*b + c) % 9 = 2 →
  (3*a + b + 2*c) % 9 = 3 →
  (a * b * c) % 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_remainder_mod_9_l2792_279203


namespace NUMINAMATH_CALUDE_count_numbers_with_property_l2792_279271

-- Define a two-digit number
def two_digit_number (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

-- Define the property we're interested in
def has_property (a b : ℕ) : Prop :=
  two_digit_number a b ∧ (10 * a + b - (a + b)) % 10 = 6

-- The theorem to prove
theorem count_numbers_with_property :
  ∃ (S : Finset ℕ), S.card = 10 ∧ 
  (∀ n, n ∈ S ↔ ∃ a b, has_property a b ∧ n = 10 * a + b) :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_property_l2792_279271


namespace NUMINAMATH_CALUDE_direct_proportion_through_point_l2792_279234

/-- A direct proportion function passing through (1, 3) has k = 3 -/
theorem direct_proportion_through_point (k : ℝ) :
  (∀ x y : ℝ, y = k * x) →  -- Direct proportion function
  3 = k * 1 →               -- Passes through (1, 3)
  k = 3 :=                  -- k equals 3
by sorry

end NUMINAMATH_CALUDE_direct_proportion_through_point_l2792_279234


namespace NUMINAMATH_CALUDE_inequality_comparison_l2792_279216

theorem inequality_comparison : 
  (-0.1 < -0.01) ∧ ¬(-1 > 0) ∧ ¬((1:ℚ)/2 < (1:ℚ)/3) ∧ ¬(-5 > 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_comparison_l2792_279216


namespace NUMINAMATH_CALUDE_notebook_purchase_problem_l2792_279202

theorem notebook_purchase_problem :
  ∀ (price_A price_B : ℝ) (quantity_A quantity_B : ℕ),
  -- Conditions
  (price_B = price_A + 1) →
  (110 / price_A = 120 / price_B) →
  (quantity_A + quantity_B = 100) →
  (quantity_B ≤ 3 * quantity_A) →
  -- Conclusions
  (price_A = 11) ∧
  (price_B = 12) ∧
  (∀ (total_cost : ℝ),
    total_cost = price_A * quantity_A + price_B * quantity_B →
    total_cost ≥ 1100) :=
by sorry

end NUMINAMATH_CALUDE_notebook_purchase_problem_l2792_279202


namespace NUMINAMATH_CALUDE_cube_vertices_l2792_279276

/-- A cube is a polyhedron with 6 faces and 12 edges -/
structure Cube where
  faces : ℕ
  edges : ℕ
  faces_eq : faces = 6
  edges_eq : edges = 12

/-- The number of vertices in a cube -/
def num_vertices (c : Cube) : ℕ := sorry

theorem cube_vertices (c : Cube) : num_vertices c = 8 := by sorry

end NUMINAMATH_CALUDE_cube_vertices_l2792_279276


namespace NUMINAMATH_CALUDE_gcd_of_198_and_308_l2792_279240

theorem gcd_of_198_and_308 : Nat.gcd 198 308 = 22 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_198_and_308_l2792_279240


namespace NUMINAMATH_CALUDE_carlos_blocks_l2792_279227

theorem carlos_blocks (initial_blocks : ℕ) (given_blocks : ℕ) : 
  initial_blocks = 58 → given_blocks = 21 → initial_blocks - given_blocks = 37 := by
  sorry

end NUMINAMATH_CALUDE_carlos_blocks_l2792_279227


namespace NUMINAMATH_CALUDE_nested_radical_inequality_l2792_279250

theorem nested_radical_inequality (x : ℝ) (hx : x > 0) :
  Real.sqrt (2 * x * Real.sqrt ((2 * x + 1) * Real.sqrt ((2 * x + 2) * Real.sqrt (2 * x + 3)))) < (15 * x + 6) / 8 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_inequality_l2792_279250


namespace NUMINAMATH_CALUDE_youngest_child_age_l2792_279283

theorem youngest_child_age (n : ℕ) 
  (h1 : ∃ x : ℕ, x + (x + 2) + (x + 4) = 48)
  (h2 : ∃ y : ℕ, y + (y + 3) + (y + 6) = 60)
  (h3 : ∃ z : ℕ, z + (z + 4) = 30)
  (h4 : n = 8) :
  ∃ w : ℕ, (w = 13 ∧ w ≤ x ∧ w ≤ y ∧ w ≤ z) :=
by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_l2792_279283


namespace NUMINAMATH_CALUDE_children_boarding_bus_l2792_279286

theorem children_boarding_bus (initial_children final_children : ℕ) 
  (h1 : initial_children = 18)
  (h2 : final_children = 25) :
  final_children - initial_children = 7 := by
  sorry

end NUMINAMATH_CALUDE_children_boarding_bus_l2792_279286


namespace NUMINAMATH_CALUDE_circle_tangent_sum_of_radii_l2792_279290

theorem circle_tangent_sum_of_radii :
  ∀ r : ℝ,
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 2)^2) →
  ∃ r' : ℝ,
  (r' > 0) ∧
  ((r' - 4)^2 + r'^2 = (r' + 2)^2) ∧
  (r + r' = 12) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_sum_of_radii_l2792_279290


namespace NUMINAMATH_CALUDE_robins_hair_length_l2792_279295

theorem robins_hair_length :
  ∀ (initial_length : ℝ),
    initial_length + 8 - 20 = 2 →
    initial_length = 14 :=
by sorry

end NUMINAMATH_CALUDE_robins_hair_length_l2792_279295


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l2792_279236

/-- Given a line segment with midpoint (3, -1) and one endpoint (7, 3), 
    prove that the other endpoint is (-1, -5). -/
theorem other_endpoint_of_line_segment 
  (midpoint : ℝ × ℝ)
  (endpoint1 : ℝ × ℝ)
  (h_midpoint : midpoint = (3, -1))
  (h_endpoint1 : endpoint1 = (7, 3)) :
  ∃ endpoint2 : ℝ × ℝ, 
    endpoint2 = (-1, -5) ∧ 
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l2792_279236


namespace NUMINAMATH_CALUDE_certain_number_equation_l2792_279232

theorem certain_number_equation (x : ℝ) : 7 * x = 4 * x + 12 + 6 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2792_279232


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2792_279289

theorem polynomial_evaluation :
  let f : ℝ → ℝ := λ x ↦ x^4 + x^3 + x^2 + x + 2
  f 2 = 32 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2792_279289


namespace NUMINAMATH_CALUDE_sum_of_common_divisors_l2792_279275

def numbers : List Int := [36, 72, -24, 120, 96]

def is_common_divisor (d : Nat) : Bool :=
  numbers.all (fun n => n % d = 0)

def common_divisors : List Nat :=
  (List.range 37).filter is_common_divisor

theorem sum_of_common_divisors :
  common_divisors.sum = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_divisors_l2792_279275


namespace NUMINAMATH_CALUDE_arg_cube_equals_pi_l2792_279220

theorem arg_cube_equals_pi (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 3)
  (h₂ : Complex.abs z₂ = 5)
  (h₃ : Complex.abs (z₁ + z₂) = 7) :
  (Complex.arg (z₁ - z₂))^3 = π := by
  sorry

end NUMINAMATH_CALUDE_arg_cube_equals_pi_l2792_279220


namespace NUMINAMATH_CALUDE_coal_cost_equilibrium_point_verify_equilibrium_point_l2792_279210

/-- Represents the cost of coal at a point on the line segment AB -/
def coal_cost (x : ℝ) (from_a : Bool) : ℝ :=
  if from_a then
    3.75 + 0.008 * x
  else
    4.25 + 0.008 * (225 - x)

/-- Theorem stating the existence and uniqueness of point C -/
theorem coal_cost_equilibrium_point :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ 225 ∧
    coal_cost x true = coal_cost x false ∧
    ∀ y : ℝ, 0 ≤ y ∧ y ≤ 225 → coal_cost y true ≤ coal_cost x true ∧ coal_cost y false ≤ coal_cost x false :=
by
  sorry

/-- The actual equilibrium point -/
def equilibrium_point : ℝ := 143.75

/-- The cost of coal at the equilibrium point -/
def equilibrium_cost : ℝ := 4.90

/-- Theorem verifying the equilibrium point and cost -/
theorem verify_equilibrium_point :
  coal_cost equilibrium_point true = equilibrium_cost ∧
  coal_cost equilibrium_point false = equilibrium_cost :=
by
  sorry

end NUMINAMATH_CALUDE_coal_cost_equilibrium_point_verify_equilibrium_point_l2792_279210


namespace NUMINAMATH_CALUDE_carolyn_shared_marbles_l2792_279230

/-- The number of marbles Carolyn shared with Diana -/
def marbles_shared (initial_marbles final_marbles : ℕ) : ℕ :=
  initial_marbles - final_marbles

/-- Theorem stating that Carolyn shared 42 marbles with Diana -/
theorem carolyn_shared_marbles :
  let initial_marbles : ℕ := 47
  let final_marbles : ℕ := 5
  marbles_shared initial_marbles final_marbles = 42 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_shared_marbles_l2792_279230


namespace NUMINAMATH_CALUDE_spell_contest_orders_l2792_279291

/-- The number of competitors in the spell-casting contest -/
def num_competitors : ℕ := 4

/-- The function to calculate the number of possible finishing orders -/
def finishing_orders (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem stating that the number of possible finishing orders for 4 competitors is 24 -/
theorem spell_contest_orders : finishing_orders num_competitors = 24 := by
  sorry

end NUMINAMATH_CALUDE_spell_contest_orders_l2792_279291


namespace NUMINAMATH_CALUDE_inequality_proof_l2792_279255

theorem inequality_proof (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c > 0) :
  a + b ≤ 2 * c ∧ 2 * c ≤ 3 * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2792_279255


namespace NUMINAMATH_CALUDE_black_dogs_count_l2792_279288

def total_dogs : Nat := 45
def brown_dogs : Nat := 20
def white_dogs : Nat := 10

theorem black_dogs_count : total_dogs - (brown_dogs + white_dogs) = 15 := by
  sorry

end NUMINAMATH_CALUDE_black_dogs_count_l2792_279288


namespace NUMINAMATH_CALUDE_complex_function_equality_l2792_279246

-- Define the complex function f
def f : ℂ → ℂ := fun z ↦ 2 * (1 - z) - Complex.I

-- State the theorem
theorem complex_function_equality :
  (1 + Complex.I) * f (1 - Complex.I) = -1 + Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_function_equality_l2792_279246


namespace NUMINAMATH_CALUDE_cars_in_ten_hours_l2792_279258

-- Define the time interval between cars (in minutes)
def time_interval : ℕ := 20

-- Define the total duration (in hours)
def total_duration : ℕ := 10

-- Define the function to calculate the number of cars
def num_cars (interval : ℕ) (duration : ℕ) : ℕ :=
  (duration * 60) / interval

-- Theorem to prove
theorem cars_in_ten_hours :
  num_cars time_interval total_duration = 30 := by
  sorry

end NUMINAMATH_CALUDE_cars_in_ten_hours_l2792_279258


namespace NUMINAMATH_CALUDE_committee_selection_l2792_279252

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l2792_279252


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l2792_279265

theorem gcd_factorial_eight_ten : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l2792_279265


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2792_279245

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2792_279245


namespace NUMINAMATH_CALUDE_sqrt_22_properties_l2792_279228

theorem sqrt_22_properties (h : 4 < Real.sqrt 22 ∧ Real.sqrt 22 < 5) :
  (∃ (i : ℤ) (d : ℝ), i = 4 ∧ d = Real.sqrt 22 - 4 ∧ Real.sqrt 22 = i + d) ∧
  (∃ (m n : ℝ), 
    m = 7 - Real.sqrt 22 - Int.floor (7 - Real.sqrt 22) ∧
    n = 7 + Real.sqrt 22 - Int.floor (7 + Real.sqrt 22) ∧
    m + n = 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_22_properties_l2792_279228


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l2792_279257

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ x = 131/11 ∧ 
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 10 → x ≤ y) ∧
  (⌊x^2⌋ - x * ⌊x⌋ = 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l2792_279257


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2792_279241

theorem sum_of_a_and_b (a b : ℝ) (h1 : a > b) (h2 : |a| = 9) (h3 : b^2 = 4) :
  a + b = 11 ∨ a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2792_279241


namespace NUMINAMATH_CALUDE_gcd_1978_2017_l2792_279293

theorem gcd_1978_2017 : Nat.gcd 1978 2017 = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_1978_2017_l2792_279293


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2792_279215

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (1 + 2*i) / (1 - i)
  Complex.im z = 3/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2792_279215


namespace NUMINAMATH_CALUDE_product_remainder_eleven_l2792_279285

theorem product_remainder_eleven : (1010 * 1011 * 1012 * 1013 * 1014) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_eleven_l2792_279285


namespace NUMINAMATH_CALUDE_positive_integer_triplets_equation_l2792_279204

theorem positive_integer_triplets_equation :
  ∀ a b c : ℕ+,
    (6 : ℕ) ^ a.val = 1 + 2 ^ b.val + 3 ^ c.val ↔
    ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 3, 3) ∨ (a, b, c) = (2, 5, 1)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_triplets_equation_l2792_279204


namespace NUMINAMATH_CALUDE_sqrt_three_plus_two_power_l2792_279270

theorem sqrt_three_plus_two_power : (Real.sqrt 3 + Real.sqrt 2) ^ 2023 * (Real.sqrt 3 - Real.sqrt 2) ^ 2022 = Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_plus_two_power_l2792_279270


namespace NUMINAMATH_CALUDE_bracelet_arrangement_l2792_279205

/-- The number of unique arrangements of beads on a bracelet -/
def uniqueArrangements (n : ℕ) : ℕ := sorry

/-- Two specific beads are always adjacent -/
def adjacentBeads : Prop := sorry

/-- Rotations and reflections of the same arrangement are considered identical -/
def symmetryEquivalence : Prop := sorry

theorem bracelet_arrangement :
  uniqueArrangements 8 = 720 ∧ adjacentBeads ∧ symmetryEquivalence :=
sorry

end NUMINAMATH_CALUDE_bracelet_arrangement_l2792_279205


namespace NUMINAMATH_CALUDE_ellipse_dimensions_l2792_279206

/-- Given an ellipse and a parabola with specific properties, prove the dimensions of the ellipse. -/
theorem ellipse_dimensions (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (∀ x y, x^2 / m^2 + y^2 / n^2 = 1) →  -- Ellipse equation
  (∃ x₀, ∀ y, y^2 = 8*x₀ ∧ x₀ = 2) →   -- Parabola focus
  (let c := Real.sqrt (m^2 - n^2);
   c / m = 1 / 2) →                    -- Eccentricity
  m^2 = 16 ∧ n^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_ellipse_dimensions_l2792_279206


namespace NUMINAMATH_CALUDE_some_number_value_l2792_279209

theorem some_number_value (x y N : ℝ) 
  (eq1 : 2 * x + y = N) 
  (eq2 : x + 2 * y = 5) 
  (eq3 : (x + y) / 3 = 1) : 
  N = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2792_279209


namespace NUMINAMATH_CALUDE_eraser_cost_tyler_eraser_cost_l2792_279287

/-- Calculates the cost of each eraser given Tyler's shopping scenario -/
theorem eraser_cost (initial_amount : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) 
  (eraser_count : ℕ) (remaining_amount : ℕ) : ℕ :=
  by
  sorry

/-- Proves that each eraser costs $4 in Tyler's specific scenario -/
theorem tyler_eraser_cost : eraser_cost 100 8 5 10 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_eraser_cost_tyler_eraser_cost_l2792_279287


namespace NUMINAMATH_CALUDE_pentagon_3010th_position_l2792_279226

/-- Represents the possible positions of the pentagon --/
inductive PentagonPosition
  | ABCDE
  | EABCD
  | DCBAE
  | EDABC

/-- Represents the operations that can be performed on the pentagon --/
inductive Operation
  | Rotate
  | Reflect

/-- Applies an operation to a pentagon position --/
def applyOperation (pos : PentagonPosition) (op : Operation) : PentagonPosition :=
  match pos, op with
  | PentagonPosition.ABCDE, Operation.Rotate => PentagonPosition.EABCD
  | PentagonPosition.EABCD, Operation.Reflect => PentagonPosition.DCBAE
  | PentagonPosition.DCBAE, Operation.Rotate => PentagonPosition.EDABC
  | PentagonPosition.EDABC, Operation.Reflect => PentagonPosition.ABCDE
  | _, _ => pos  -- Default case to satisfy exhaustiveness

/-- Applies a sequence of alternating rotate and reflect operations --/
def applySequence (n : Nat) : PentagonPosition :=
  match n % 4 with
  | 0 => PentagonPosition.ABCDE
  | 1 => PentagonPosition.EABCD
  | 2 => PentagonPosition.DCBAE
  | _ => PentagonPosition.EDABC

theorem pentagon_3010th_position :
  applySequence 3010 = PentagonPosition.ABCDE :=
sorry


end NUMINAMATH_CALUDE_pentagon_3010th_position_l2792_279226


namespace NUMINAMATH_CALUDE_large_box_125_times_small_box_l2792_279218

-- Define the dimensions of the large box
def large_width : ℝ := 30
def large_length : ℝ := 20
def large_height : ℝ := 5

-- Define the dimensions of the small box
def small_width : ℝ := 6
def small_length : ℝ := 4
def small_height : ℝ := 1

-- Define the volume calculation function for a cuboid
def cuboid_volume (width length height : ℝ) : ℝ := width * length * height

-- Theorem statement
theorem large_box_125_times_small_box :
  cuboid_volume large_width large_length large_height =
  125 * cuboid_volume small_width small_length small_height := by
  sorry

end NUMINAMATH_CALUDE_large_box_125_times_small_box_l2792_279218


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2792_279272

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0)
  (h4 : s + 2*y = 3*s) -- outer square side length
  (h5 : x + s = 3*s) -- outer square side length
  (h6 : (3*s)^2 = 9*s^2) -- area relation
  : x / y = 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2792_279272


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2792_279262

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 * a = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (2 * a - 1) * x + a * y = 0

-- Define perpendicularity
def perpendicular (a : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, line1 a x₁ y₁ ∧ line2 a x₂ y₂ →
    (x₁ - x₂) * (y₁ - y₂) = 0

-- Theorem statement
theorem perpendicular_lines_a_values :
  ∀ a : ℝ, perpendicular a → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2792_279262


namespace NUMINAMATH_CALUDE_monthly_earnings_calculation_l2792_279251

/-- Proves that a person with given savings and earnings parameters earns a specific monthly amount -/
theorem monthly_earnings_calculation (savings_per_month : ℕ) 
                                     (car_cost : ℕ) 
                                     (total_earnings : ℕ) 
                                     (h1 : savings_per_month = 500)
                                     (h2 : car_cost = 45000)
                                     (h3 : total_earnings = 360000) : 
  (total_earnings / (car_cost / savings_per_month) : ℚ) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_earnings_calculation_l2792_279251


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l2792_279261

theorem volume_of_inscribed_sphere (edge_length : ℝ) (h : edge_length = 6) :
  let radius : ℝ := edge_length / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * radius ^ 3
  sphere_volume = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l2792_279261


namespace NUMINAMATH_CALUDE_peters_pond_depth_l2792_279221

theorem peters_pond_depth (marks_depth peter_depth : ℝ) 
  (h1 : marks_depth = 3 * peter_depth + 4)
  (h2 : marks_depth = 19) : peter_depth = 5 := by
  sorry

end NUMINAMATH_CALUDE_peters_pond_depth_l2792_279221


namespace NUMINAMATH_CALUDE_odd_number_divisibility_l2792_279200

theorem odd_number_divisibility (a : ℤ) (h : ∃ n : ℤ, a = 2*n + 1) :
  ∃ k : ℤ, a^4 + 9*(9 - 2*a^2) = 16*k := by
sorry

end NUMINAMATH_CALUDE_odd_number_divisibility_l2792_279200


namespace NUMINAMATH_CALUDE_union_determines_m_l2792_279249

def A (m : ℝ) : Set ℝ := {2, m}
def B (m : ℝ) : Set ℝ := {1, m^2}

theorem union_determines_m :
  ∀ m : ℝ, A m ∪ B m = {1, 2, 3, 9} → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_determines_m_l2792_279249


namespace NUMINAMATH_CALUDE_three_intersection_points_l2792_279212

/-- The number of distinct points satisfying the given equations -/
def num_intersection_points : ℕ := 3

/-- First equation -/
def equation1 (x y : ℝ) : Prop :=
  (x + y - 7) * (2*x - 3*y + 7) = 0

/-- Second equation -/
def equation2 (x y : ℝ) : Prop :=
  (x - y + 3) * (3*x + 2*y - 18) = 0

/-- Theorem stating that there are exactly 3 distinct points satisfying both equations -/
theorem three_intersection_points :
  ∃! (points : Finset (ℝ × ℝ)), 
    points.card = num_intersection_points ∧
    ∀ p ∈ points, equation1 p.1 p.2 ∧ equation2 p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_three_intersection_points_l2792_279212


namespace NUMINAMATH_CALUDE_shopkeeper_discount_l2792_279259

theorem shopkeeper_discount (cost_price : ℝ) (h_positive : cost_price > 0) : 
  let labeled_price := cost_price * (1 + 0.4)
  let selling_price := cost_price * (1 + 0.33)
  let discount := labeled_price - selling_price
  let discount_percentage := (discount / labeled_price) * 100
  discount_percentage = 5 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_discount_l2792_279259


namespace NUMINAMATH_CALUDE_gcd_of_35_91_840_l2792_279282

theorem gcd_of_35_91_840 : Nat.gcd 35 (Nat.gcd 91 840) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_35_91_840_l2792_279282


namespace NUMINAMATH_CALUDE_line_only_count_l2792_279243

/-- Represents the alphabet with its properties -/
structure Alphabet where
  total : ℕ
  dot_and_line : ℕ
  dot_only : ℕ
  has_dot_or_line : total = dot_and_line + dot_only + (total - (dot_and_line + dot_only))

/-- The specific alphabet from the problem -/
def problem_alphabet : Alphabet := {
  total := 50
  dot_and_line := 16
  dot_only := 4
  has_dot_or_line := by sorry
}

/-- The number of letters with a straight line but no dot -/
def line_only (a : Alphabet) : ℕ := a.total - (a.dot_and_line + a.dot_only)

/-- Theorem stating the result for the problem alphabet -/
theorem line_only_count : line_only problem_alphabet = 30 := by sorry

end NUMINAMATH_CALUDE_line_only_count_l2792_279243


namespace NUMINAMATH_CALUDE_shuffleboard_games_l2792_279233

/-- The number of games won by Jerry -/
def jerry_wins : ℕ := 7

/-- The number of games won by Dave -/
def dave_wins : ℕ := jerry_wins + 3

/-- The number of games won by Ken -/
def ken_wins : ℕ := dave_wins + 5

/-- The number of games won by Larry -/
def larry_wins : ℕ := 2 * jerry_wins

/-- The total number of ties -/
def total_ties : ℕ := jerry_wins

/-- The total number of games played -/
def total_games : ℕ := ken_wins + dave_wins + jerry_wins + larry_wins + total_ties

theorem shuffleboard_games :
  (∀ player : ℕ, player ∈ [ken_wins, dave_wins, jerry_wins, larry_wins] → player ≥ 5) →
  total_games = 53 := by
  sorry

end NUMINAMATH_CALUDE_shuffleboard_games_l2792_279233


namespace NUMINAMATH_CALUDE_boat_capacity_l2792_279207

theorem boat_capacity (trips_per_day : ℕ) (total_people : ℕ) (total_days : ℕ) :
  trips_per_day = 4 →
  total_people = 96 →
  total_days = 2 →
  total_people / (trips_per_day * total_days) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_capacity_l2792_279207


namespace NUMINAMATH_CALUDE_tencent_aligns_with_dialectical_materialism_l2792_279267

/-- Represents the principles of dialectical materialism --/
structure DialecticalMaterialism where
  dialecticalNegation : Bool
  innovationAndDevelopment : Bool
  unityOfDevelopment : Bool
  unityOfProgressAndTortuosity : Bool
  unityOfQuantitativeAndQualitativeChanges : Bool

/-- Represents Tencent's development patterns --/
structure TencentDevelopment where
  technologicalInnovation : Bool
  overcomingDifficulties : Bool
  seizedOpportunities : Bool
  achievedQualitativeLeap : Bool

/-- Theorem stating that Tencent's development aligns with dialectical materialism --/
theorem tencent_aligns_with_dialectical_materialism 
  (dm : DialecticalMaterialism) 
  (td : TencentDevelopment) : 
  (dm.dialecticalNegation ∧ 
   dm.innovationAndDevelopment ∧ 
   dm.unityOfDevelopment ∧ 
   dm.unityOfProgressAndTortuosity ∧ 
   dm.unityOfQuantitativeAndQualitativeChanges) → 
  (td.technologicalInnovation ∧ 
   td.overcomingDifficulties ∧ 
   td.seizedOpportunities ∧ 
   td.achievedQualitativeLeap) := by
  sorry

end NUMINAMATH_CALUDE_tencent_aligns_with_dialectical_materialism_l2792_279267


namespace NUMINAMATH_CALUDE_lansing_new_students_average_l2792_279298

/-- The average number of new students per school in Lansing -/
def average_new_students_per_school (total_schools : Float) (total_new_students : Float) : Float :=
  total_new_students / total_schools

/-- Theorem: The average number of new students per school in Lansing is 9.88 -/
theorem lansing_new_students_average :
  let total_schools : Float := 25.0
  let total_new_students : Float := 247.0
  average_new_students_per_school total_schools total_new_students = 9.88 := by
  sorry

end NUMINAMATH_CALUDE_lansing_new_students_average_l2792_279298


namespace NUMINAMATH_CALUDE_sales_increase_percentage_l2792_279235

def saturday_sales : ℕ := 60
def total_sales : ℕ := 150

def sunday_sales : ℕ := total_sales - saturday_sales

def percentage_increase : ℚ := (sunday_sales - saturday_sales : ℚ) / saturday_sales * 100

theorem sales_increase_percentage :
  sunday_sales > saturday_sales →
  percentage_increase = 50 := by
  sorry

end NUMINAMATH_CALUDE_sales_increase_percentage_l2792_279235


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2792_279237

/-- Given a geometric sequence where a₁ = 3 and a₃ = 1/9, prove that a₆ = 1/81 -/
theorem geometric_sequence_sixth_term (a : ℕ → ℚ) (h1 : a 1 = 3) (h3 : a 3 = 1/9) :
  a 6 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2792_279237


namespace NUMINAMATH_CALUDE_elevator_weight_average_l2792_279279

theorem elevator_weight_average (initial_people : Nat) (initial_avg_weight : ℝ) (new_person_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 156 →
  new_person_weight = 121 →
  let total_weight := initial_people * initial_avg_weight + new_person_weight
  let new_people_count := initial_people + 1
  let new_avg_weight := total_weight / new_people_count
  new_avg_weight = 151 := by
sorry

end NUMINAMATH_CALUDE_elevator_weight_average_l2792_279279


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l2792_279217

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  (1 / (1 + a)) + (4 / (4 + b)) ≥ 9/8 := by sorry

theorem min_value_achievable (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 3 ∧
  (1 / (1 + a₀)) + (4 / (4 + b₀)) = 9/8 := by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l2792_279217
