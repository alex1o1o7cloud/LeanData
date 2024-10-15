import Mathlib

namespace NUMINAMATH_CALUDE_base_number_theorem_l1833_183309

theorem base_number_theorem (x w : ℝ) (h1 : x^(2*w) = 8^(w-4)) (h2 : w = 12) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_theorem_l1833_183309


namespace NUMINAMATH_CALUDE_box_cube_volume_l1833_183311

/-- Given a box with dimensions 12 cm × 16 cm × 6 cm built using 384 identical cubic cm cubes,
    prove that the volume of each cube is 3 cm³. -/
theorem box_cube_volume (length width height : ℝ) (num_cubes : ℕ) (cube_volume : ℝ) :
  length = 12 →
  width = 16 →
  height = 6 →
  num_cubes = 384 →
  length * width * height = num_cubes * cube_volume →
  cube_volume = 3 := by
  sorry

end NUMINAMATH_CALUDE_box_cube_volume_l1833_183311


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l1833_183365

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_nonzero (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def product_divisible_by_1000 (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (n * (10 * a + b) * a) % 1000 = 0

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_three_digit n ∧
             digits_nonzero n ∧
             product_divisible_by_1000 n ∧
             n = 875 := by sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l1833_183365


namespace NUMINAMATH_CALUDE_smallest_class_size_l1833_183325

theorem smallest_class_size (n : ℕ) : 
  (4*n + 2 > 40) ∧ 
  (∀ m : ℕ, m < n → 4*m + 2 ≤ 40) → 
  4*n + 2 = 42 :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_l1833_183325


namespace NUMINAMATH_CALUDE_elizabeth_stickers_l1833_183301

/-- The number of stickers Elizabeth uses on her water bottles -/
def total_stickers (initial_bottles : ℕ) (lost_bottles : ℕ) (stolen_bottles : ℕ) (stickers_per_bottle : ℕ) : ℕ :=
  (initial_bottles - lost_bottles - stolen_bottles) * stickers_per_bottle

/-- Theorem: Elizabeth uses 21 stickers in total -/
theorem elizabeth_stickers :
  total_stickers 10 2 1 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_stickers_l1833_183301


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1833_183390

/-- Two circles are externally tangent when the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂

/-- The problem statement -/
theorem circles_externally_tangent :
  let r₁ : ℝ := 1
  let r₂ : ℝ := 3
  let d : ℝ := 4
  externally_tangent r₁ r₂ d :=
by
  sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1833_183390


namespace NUMINAMATH_CALUDE_black_tiles_to_total_l1833_183310

/-- Represents a square hall tiled with square tiles -/
structure SquareHall where
  side : ℕ

/-- Calculates the number of black tiles in the hall -/
def black_tiles (hall : SquareHall) : ℕ :=
  2 * hall.side

/-- Calculates the total number of tiles in the hall -/
def total_tiles (hall : SquareHall) : ℕ :=
  hall.side * hall.side

/-- Theorem stating the relationship between black tiles and total tiles -/
theorem black_tiles_to_total (hall : SquareHall) :
  black_tiles hall - 3 = 153 → total_tiles hall = 6084 := by
  sorry

end NUMINAMATH_CALUDE_black_tiles_to_total_l1833_183310


namespace NUMINAMATH_CALUDE_blossom_room_area_l1833_183347

/-- Represents the length of a side of a square room in feet -/
def room_side_length : ℕ := 10

/-- Represents the number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Calculates the area of a square room in square inches -/
def room_area_sq_inches (side_length : ℕ) (inches_per_foot : ℕ) : ℕ :=
  (side_length * inches_per_foot) ^ 2

/-- Theorem stating that the area of Blossom's room is 14400 square inches -/
theorem blossom_room_area :
  room_area_sq_inches room_side_length inches_per_foot = 14400 := by
  sorry

end NUMINAMATH_CALUDE_blossom_room_area_l1833_183347


namespace NUMINAMATH_CALUDE_real_solutions_iff_a_in_range_l1833_183386

/-- Given a system of equations with real parameter a, 
    prove that real solutions exist if and only if 1 ≤ a ≤ 2 -/
theorem real_solutions_iff_a_in_range (a : ℝ) :
  (∃ x y : ℝ, x + y = a * (Real.sqrt x - Real.sqrt y) ∧
               x^2 + y^2 = a * (Real.sqrt x - Real.sqrt y)^2) ↔
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_iff_a_in_range_l1833_183386


namespace NUMINAMATH_CALUDE_sandra_beignet_consumption_l1833_183342

/-- The number of beignets Sandra eats per day -/
def daily_beignets : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks -/
def num_weeks : ℕ := 16

/-- The total number of beignets Sandra eats in the given period -/
def total_beignets : ℕ := daily_beignets * days_per_week * num_weeks

theorem sandra_beignet_consumption :
  total_beignets = 336 := by sorry

end NUMINAMATH_CALUDE_sandra_beignet_consumption_l1833_183342


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1833_183331

theorem polynomial_divisibility : ∀ x : ℂ,
  (x^100 + x^75 + x^50 + x^25 + 1) % (x^9 + x^6 + x^3 + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1833_183331


namespace NUMINAMATH_CALUDE_max_value_sqrt_product_max_value_achievable_l1833_183394

theorem max_value_sqrt_product (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1/2) : 
  Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1 / Real.sqrt 2 + 1 / 2 :=
by sorry

theorem max_value_achievable : 
  ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1/2 ∧
  Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) = 1 / Real.sqrt 2 + 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_product_max_value_achievable_l1833_183394


namespace NUMINAMATH_CALUDE_pyramid_height_l1833_183356

/-- Given a square pyramid whose lateral faces unfold into a square with side length 18,
    prove that the height of the pyramid is 6. -/
theorem pyramid_height (s : ℝ) (h : s > 0) : 
  s * s = 18 * 18 / 2 → (6 : ℝ) * s = 18 * 18 / 2 := by
  sorry

#check pyramid_height

end NUMINAMATH_CALUDE_pyramid_height_l1833_183356


namespace NUMINAMATH_CALUDE_triangle_abc_area_l1833_183380

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  Real.sqrt (1/4 * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))

theorem triangle_abc_area :
  ∀ (A B C : ℝ) (a b c : ℝ),
  (Real.sin A - Real.sin B) * (Real.sin A + Real.sin B) = Real.sin A * Real.sin C - Real.sin C^2 →
  c = 2*a ∧ c = 2 * Real.sqrt 2 →
  triangle_area a b c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l1833_183380


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1833_183399

-- Define the universal set U
def U : Set ℕ := {x | 1 < x ∧ x < 5}

-- Define set A
def A : Set ℕ := {2, 3}

-- State the theorem
theorem complement_of_A_in_U :
  (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1833_183399


namespace NUMINAMATH_CALUDE_number_theory_statements_l1833_183354

theorem number_theory_statements :
  (∃ n : ℕ, 20 = 4 * n) ∧
  (∃ n : ℕ, 209 = 19 * n) ∧ ¬(∃ n : ℕ, 63 = 19 * n) ∧
  ¬(∃ n : ℕ, 75 = 12 * n) ∧ ¬(∃ n : ℕ, 29 = 12 * n) ∧
  (∃ n : ℕ, 33 = 11 * n) ∧ ¬(∃ n : ℕ, 64 = 11 * n) ∧
  (∃ n : ℕ, 180 = 9 * n) := by
sorry

end NUMINAMATH_CALUDE_number_theory_statements_l1833_183354


namespace NUMINAMATH_CALUDE_parallel_planes_from_intersecting_parallel_lines_l1833_183350

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)

-- Define the property of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the property of two lines intersecting
variable (lines_intersect : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_from_intersecting_parallel_lines
  (α β : Plane) (l₁ l₂ : Line)
  (h1 : line_in_plane l₁ α)
  (h2 : line_in_plane l₂ α)
  (h3 : lines_intersect l₁ l₂)
  (h4 : line_parallel_to_plane l₁ β)
  (h5 : line_parallel_to_plane l₂ β) :
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_intersecting_parallel_lines_l1833_183350


namespace NUMINAMATH_CALUDE_range_of_m_l1833_183384

theorem range_of_m (m : ℝ) : 
  let p := (1^2 + 1^2 - 2*m*1 + 2*m*1 + 2*m^2 - 4 < 0)
  let q := ∀ (x y : ℝ), m*x - y + 1 + 2*m = 0 → (x > 0 → y ≥ 0)
  (p ∨ q) ∧ ¬(p ∧ q) → ((-1 < m ∧ m < 0) ∨ m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1833_183384


namespace NUMINAMATH_CALUDE_largest_power_of_six_divisor_l1833_183330

theorem largest_power_of_six_divisor : 
  (∃ k : ℕ, 6^k ∣ (8 * 48 * 81) ∧ 
   ∀ m : ℕ, m > k → ¬(6^m ∣ (8 * 48 * 81))) → 
  (∃ k : ℕ, k = 5 ∧ 6^k ∣ (8 * 48 * 81) ∧ 
   ∀ m : ℕ, m > k → ¬(6^m ∣ (8 * 48 * 81))) := by
sorry

end NUMINAMATH_CALUDE_largest_power_of_six_divisor_l1833_183330


namespace NUMINAMATH_CALUDE_inequality_proof_l1833_183314

theorem inequality_proof (x : ℝ) (hx : x > 0) :
  Real.sqrt (x^2 - x + 1/2) ≥ 1 / (x + 1/x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1833_183314


namespace NUMINAMATH_CALUDE_triangle_rectangle_side_ratio_l1833_183303

/-- Given an equilateral triangle and a rectangle with the same perimeter and a specific length-width ratio for the rectangle, this theorem proves that the ratio of the triangle's side length to the rectangle's length is 1. -/
theorem triangle_rectangle_side_ratio (perimeter : ℝ) (length_width_ratio : ℝ) :
  perimeter > 0 →
  length_width_ratio = 2 →
  let triangle_side := perimeter / 3
  let rectangle_width := perimeter / (2 * (length_width_ratio + 1))
  let rectangle_length := length_width_ratio * rectangle_width
  (triangle_side / rectangle_length) = 1 := by
  sorry

#check triangle_rectangle_side_ratio

end NUMINAMATH_CALUDE_triangle_rectangle_side_ratio_l1833_183303


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l1833_183355

open Real

theorem arctan_tan_difference (θ₁ θ₂ : ℝ) (h₁ : θ₁ = 70 * π / 180) (h₂ : θ₂ = 20 * π / 180) :
  arctan (tan θ₁ - 3 * tan θ₂) = 50 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l1833_183355


namespace NUMINAMATH_CALUDE_equation_true_when_x_is_three_l1833_183358

theorem equation_true_when_x_is_three : ∀ x : ℝ, x = 3 → 3 * x - 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_true_when_x_is_three_l1833_183358


namespace NUMINAMATH_CALUDE_line_slope_problem_l1833_183372

/-- Given m > 0 and points (m, 4) and (2, m) lie on a line with slope m^2, prove m = 2 -/
theorem line_slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = m^2) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l1833_183372


namespace NUMINAMATH_CALUDE_new_person_weight_is_109_5_l1833_183329

/-- Calculates the weight of a new person in a group when the average weight changes --/
def newPersonWeight (numPersons : ℕ) (avgWeightIncrease : ℝ) (oldPersonWeight : ℝ) : ℝ :=
  oldPersonWeight + numPersons * avgWeightIncrease

/-- Theorem: The weight of the new person is 109.5 kg --/
theorem new_person_weight_is_109_5 :
  newPersonWeight 15 2.3 75 = 109.5 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_109_5_l1833_183329


namespace NUMINAMATH_CALUDE_monroe_family_children_l1833_183345

/-- Given the total number of granola bars, the number eaten by parents, and the number given to each child,
    calculate the number of children in the family. -/
def number_of_children (total_bars : ℕ) (eaten_by_parents : ℕ) (bars_per_child : ℕ) : ℕ :=
  (total_bars - eaten_by_parents) / bars_per_child

/-- Theorem stating that the number of children in Monroe's family is 6. -/
theorem monroe_family_children :
  number_of_children 200 80 20 = 6 := by
  sorry

end NUMINAMATH_CALUDE_monroe_family_children_l1833_183345


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1833_183326

theorem algebraic_expression_value (a : ℤ) (h : a = -2) : a + 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1833_183326


namespace NUMINAMATH_CALUDE_sphere_volume_from_cube_surface_l1833_183352

theorem sphere_volume_from_cube_surface (cube_side : ℝ) (sphere_radius : ℝ) : 
  cube_side = 3 → 
  (6 * cube_side^2 : ℝ) = 4 * π * sphere_radius^2 → 
  (4 / 3 : ℝ) * π * sphere_radius^3 = 54 * Real.sqrt 6 / Real.sqrt π := by
  sorry

#check sphere_volume_from_cube_surface

end NUMINAMATH_CALUDE_sphere_volume_from_cube_surface_l1833_183352


namespace NUMINAMATH_CALUDE_no_equilateral_from_splice_l1833_183336

/-- Represents a triangle with a 45° angle -/
structure Triangle45 where
  -- We only need to define two sides, as the third is determined by the right angle
  side1 : ℝ
  side2 : ℝ
  positive_sides : 0 < side1 ∧ 0 < side2

/-- Represents the result of splicing two Triangle45 objects -/
inductive SplicedShape
  | Equilateral
  | Other

/-- Function to splice two Triangle45 objects -/
def splice (t1 t2 : Triangle45) : SplicedShape :=
  sorry

/-- Theorem stating that splicing two Triangle45 objects cannot result in an equilateral triangle -/
theorem no_equilateral_from_splice (t1 t2 : Triangle45) :
  splice t1 t2 ≠ SplicedShape.Equilateral :=
sorry

end NUMINAMATH_CALUDE_no_equilateral_from_splice_l1833_183336


namespace NUMINAMATH_CALUDE_correct_replacement_l1833_183392

/-- Represents a digit in the addition problem -/
inductive Digit
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Represents whether a digit is correct or potentially incorrect -/
inductive DigitStatus
| correct
| potentiallyIncorrect

/-- Function to get the status of a digit -/
def digitStatus (d : Digit) : DigitStatus :=
  match d with
  | Digit.zero | Digit.one | Digit.three | Digit.four | Digit.five | Digit.six | Digit.eight => DigitStatus.correct
  | Digit.two | Digit.seven | Digit.nine => DigitStatus.potentiallyIncorrect

/-- Function to check if replacing a digit corrects the addition -/
def replacementCorrects (d : Digit) (replacement : Digit) : Prop :=
  match d, replacement with
  | Digit.two, Digit.six => True
  | _, _ => False

/-- Theorem stating that replacing 2 with 6 corrects the addition -/
theorem correct_replacement :
  ∃ (d : Digit) (replacement : Digit),
    digitStatus d = DigitStatus.potentiallyIncorrect ∧
    digitStatus replacement = DigitStatus.correct ∧
    replacementCorrects d replacement :=
by sorry

end NUMINAMATH_CALUDE_correct_replacement_l1833_183392


namespace NUMINAMATH_CALUDE_min_length_for_prob_threshold_l1833_183332

/-- The probability that a random sequence of length n using digits 0, 1, and 2 does not contain all three digits -/
def prob_not_all_digits (n : ℕ) : ℚ :=
  (2^n - 1) / 3^(n-1)

/-- The probability that a random sequence of length n using digits 0, 1, and 2 contains all three digits -/
def prob_all_digits (n : ℕ) : ℚ :=
  1 - prob_not_all_digits n

theorem min_length_for_prob_threshold :
  prob_all_digits 5 ≥ 61/100 ∧
  ∀ k < 5, prob_all_digits k < 61/100 :=
sorry

end NUMINAMATH_CALUDE_min_length_for_prob_threshold_l1833_183332


namespace NUMINAMATH_CALUDE_no_integer_cubes_l1833_183319

theorem no_integer_cubes (a b : ℤ) : 
  a ≥ 1 → b ≥ 1 → 
  (∃ x : ℤ, a^5 * b + 3 = x^3) → 
  (∃ y : ℤ, a * b^5 + 3 = y^3) → 
  False :=
sorry

end NUMINAMATH_CALUDE_no_integer_cubes_l1833_183319


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_diagonal_squared_l1833_183373

/-- An isosceles trapezoid with bases a and b, lateral side c, and diagonal d -/
structure IsoscelesTrapezoid (a b c d : ℝ) : Prop where
  bases_positive : 0 < a ∧ 0 < b
  lateral_positive : 0 < c
  diagonal_positive : 0 < d
  is_isosceles : true  -- This is a placeholder for the isosceles property

/-- The diagonal of an isosceles trapezoid satisfies d^2 = ab + c^2 -/
theorem isosceles_trapezoid_diagonal_squared 
  (a b c d : ℝ) (trap : IsoscelesTrapezoid a b c d) : 
  d^2 = a * b + c^2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_diagonal_squared_l1833_183373


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l1833_183383

/-- Given a mixture of water and salt solution, calculate the volume of salt solution needed --/
theorem salt_solution_mixture (x : ℝ) : 
  (1 : ℝ) + x > 0 →  -- Total volume is positive
  0.6 * x = 0.2 * (1 + x) → -- Salt conservation equation
  x = 0.5 := by
sorry


end NUMINAMATH_CALUDE_salt_solution_mixture_l1833_183383


namespace NUMINAMATH_CALUDE_next_base3_number_l1833_183389

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to its base 3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The base 3 representation of M -/
def M : List Nat := [0, 2, 0, 1]

theorem next_base3_number (h : base3ToDecimal M = base3ToDecimal [0, 2, 0, 1]) :
  decimalToBase3 (base3ToDecimal M + 1) = [1, 2, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_next_base3_number_l1833_183389


namespace NUMINAMATH_CALUDE_bus_passengers_l1833_183302

theorem bus_passengers (men women : ℕ) : 
  women = men / 2 → 
  men - 16 = women + 8 → 
  men + women = 72 :=
by sorry

end NUMINAMATH_CALUDE_bus_passengers_l1833_183302


namespace NUMINAMATH_CALUDE_tan_2x_value_l1833_183346

theorem tan_2x_value (f : ℝ → ℝ) (x : ℝ) 
  (h1 : ∀ x, f x = Real.sin x + Real.cos x)
  (h2 : ∀ x, deriv f x = 3 * f x) : 
  Real.tan (2 * x) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_2x_value_l1833_183346


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l1833_183398

theorem pentagon_angle_measure :
  ∀ (a b c d e : ℝ),
  a + b + c + d + e = 540 →
  a = 111 →
  b = 113 →
  c = 92 →
  d = 128 →
  e = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l1833_183398


namespace NUMINAMATH_CALUDE_moores_law_1985_to_1995_l1833_183322

/-- Moore's law doubling period in years -/
def moore_period : ℕ := 2

/-- Initial year for transistor count -/
def initial_year : ℕ := 1985

/-- Final year for transistor count -/
def final_year : ℕ := 1995

/-- Initial transistor count in 1985 -/
def initial_transistors : ℕ := 500000

/-- Calculate the number of transistors according to Moore's law -/
def transistor_count (start_year end_year start_count : ℕ) : ℕ :=
  start_count * 2 ^ ((end_year - start_year) / moore_period)

/-- Theorem stating that the transistor count in 1995 is 16,000,000 -/
theorem moores_law_1985_to_1995 :
  transistor_count initial_year final_year initial_transistors = 16000000 := by
  sorry

end NUMINAMATH_CALUDE_moores_law_1985_to_1995_l1833_183322


namespace NUMINAMATH_CALUDE_difference_exists_l1833_183335

def sequence_property (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧ ∀ n, x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n

theorem difference_exists (x : ℕ → ℕ) (h : sequence_property x) :
  ∀ k : ℕ, k > 0 → ∃ r s, x r - x s = k := by
  sorry

end NUMINAMATH_CALUDE_difference_exists_l1833_183335


namespace NUMINAMATH_CALUDE_eggs_leftover_l1833_183320

theorem eggs_leftover (daniel : Nat) (eliza : Nat) (fiona : Nat) (george : Nat)
  (h1 : daniel = 53)
  (h2 : eliza = 68)
  (h3 : fiona = 26)
  (h4 : george = 47) :
  (daniel + eliza + fiona + george) % 15 = 14 := by
  sorry

end NUMINAMATH_CALUDE_eggs_leftover_l1833_183320


namespace NUMINAMATH_CALUDE_moon_speed_in_km_per_hour_l1833_183353

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_sec : ℚ := 9/10

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_sec_to_km_per_hour (speed : ℚ) : ℚ :=
  speed * seconds_per_hour

theorem moon_speed_in_km_per_hour :
  km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3240 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_in_km_per_hour_l1833_183353


namespace NUMINAMATH_CALUDE_arrangements_with_non_adjacent_students_l1833_183348

def number_of_students : ℕ := 5

-- Total number of permutations for n students
def total_permutations (n : ℕ) : ℕ := n.factorial

-- Number of permutations where A and B are adjacent
def adjacent_permutations (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem arrangements_with_non_adjacent_students :
  total_permutations number_of_students - adjacent_permutations number_of_students = 72 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_non_adjacent_students_l1833_183348


namespace NUMINAMATH_CALUDE_homecoming_ticket_sales_l1833_183351

theorem homecoming_ticket_sales
  (single_price : ℕ)
  (couple_price : ℕ)
  (total_attendance : ℕ)
  (couple_tickets_sold : ℕ)
  (h1 : single_price = 20)
  (h2 : couple_price = 35)
  (h3 : total_attendance = 128)
  (h4 : couple_tickets_sold = 16) :
  single_price * (total_attendance - 2 * couple_tickets_sold) +
  couple_price * couple_tickets_sold = 2480 := by
sorry


end NUMINAMATH_CALUDE_homecoming_ticket_sales_l1833_183351


namespace NUMINAMATH_CALUDE_xiao_hua_first_place_l1833_183361

def fish_counts : List Nat := [23, 20, 15, 18, 13]

def xiao_hua_count : Nat := 20

def min_additional_fish (counts : List Nat) (xiao_hua : Nat) : Nat :=
  match counts.maximum? with
  | none => 0
  | some max_count => max_count - xiao_hua + 1

theorem xiao_hua_first_place (counts : List Nat) (xiao_hua : Nat) :
  counts = fish_counts ∧ xiao_hua = xiao_hua_count →
  min_additional_fish counts xiao_hua = 4 :=
by sorry

end NUMINAMATH_CALUDE_xiao_hua_first_place_l1833_183361


namespace NUMINAMATH_CALUDE_allen_reading_time_l1833_183375

/-- Calculates the number of days required to finish reading a book -/
def days_to_finish (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- Proves that Allen took 12 days to finish reading the book -/
theorem allen_reading_time : days_to_finish 120 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_allen_reading_time_l1833_183375


namespace NUMINAMATH_CALUDE_friends_receiving_pens_l1833_183396

/-- The number of friends Kendra and Tony will give pens to -/
def num_friends (kendra_packs tony_packs kendra_pens_per_pack tony_pens_per_pack kept_pens : ℕ) : ℕ :=
  kendra_packs * kendra_pens_per_pack + tony_packs * tony_pens_per_pack - 2 * kept_pens

/-- Theorem stating the number of friends Kendra and Tony will give pens to -/
theorem friends_receiving_pens :
  num_friends 7 5 4 6 3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_friends_receiving_pens_l1833_183396


namespace NUMINAMATH_CALUDE_complement_of_alpha_l1833_183333

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle
def alpha : Angle := ⟨75, 12⟩

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_alpha :
  complement alpha = ⟨14, 48⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_alpha_l1833_183333


namespace NUMINAMATH_CALUDE_abc_def_ratio_l1833_183305

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 2) :
  a * b * c / (d * e * f) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l1833_183305


namespace NUMINAMATH_CALUDE_root_product_zero_l1833_183395

theorem root_product_zero (x₁ x₂ x₃ : ℝ) :
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (Real.sqrt 2025) * x₁^3 - 4050 * x₁^2 - 4 = 0 ∧
  (Real.sqrt 2025) * x₂^3 - 4050 * x₂^2 - 4 = 0 ∧
  (Real.sqrt 2025) * x₃^3 - 4050 * x₃^2 - 4 = 0 →
  x₂ * (x₁ + x₃) = 0 := by
sorry

end NUMINAMATH_CALUDE_root_product_zero_l1833_183395


namespace NUMINAMATH_CALUDE_set_A_properties_l1833_183371

-- Define the set A
def A : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

-- Theorem statement
theorem set_A_properties :
  (∀ n : ℤ, (2*n + 1) ∈ A) ∧
  (∀ k : ℤ, (4*k - 2) ∉ A) ∧
  (∀ a b : ℤ, a ∈ A → b ∈ A → (a * b) ∈ A) := by
  sorry

end NUMINAMATH_CALUDE_set_A_properties_l1833_183371


namespace NUMINAMATH_CALUDE_other_number_proof_l1833_183317

/-- Given two positive integers with specific HCF, LCM, and one known value, prove the other value -/
theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 396) (h3 : a = 36) : b = 132 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l1833_183317


namespace NUMINAMATH_CALUDE_exponent_problem_l1833_183344

theorem exponent_problem (x m n : ℝ) (hm : x^m = 5) (hn : x^n = 10) :
  x^(2*m - n) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l1833_183344


namespace NUMINAMATH_CALUDE_quadratic_vertex_coordinates_l1833_183376

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 5

-- Define the vertex coordinates
def vertex : ℝ × ℝ := (-1, 8)

-- Theorem statement
theorem quadratic_vertex_coordinates :
  (∀ x : ℝ, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_coordinates_l1833_183376


namespace NUMINAMATH_CALUDE_donut_selection_problem_l1833_183391

theorem donut_selection_problem :
  let n : ℕ := 5  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 56 :=
by sorry

end NUMINAMATH_CALUDE_donut_selection_problem_l1833_183391


namespace NUMINAMATH_CALUDE_polynomial_has_negative_root_l1833_183377

theorem polynomial_has_negative_root : ∃ x : ℝ, x < 0 ∧ x^7 + 2*x^5 + 5*x^3 - x + 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_has_negative_root_l1833_183377


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l1833_183387

theorem collinear_points_b_value :
  ∀ b : ℚ,
  let p1 : ℚ × ℚ := (4, -6)
  let p2 : ℚ × ℚ := (b + 3, 4)
  let p3 : ℚ × ℚ := (3*b + 4, 3)
  (p1.2 - p2.2) * (p2.1 - p3.1) = (p2.2 - p3.2) * (p1.1 - p2.1) →
  b = -3/7 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l1833_183387


namespace NUMINAMATH_CALUDE_distance_to_line_l1833_183388

/-- The point from which we're measuring the distance -/
def P : ℝ × ℝ × ℝ := (0, 1, 5)

/-- The point on the line -/
def Q : ℝ → ℝ × ℝ × ℝ := λ t => (4 + 3*t, 5 - t, 6 + 2*t)

/-- The direction vector of the line -/
def v : ℝ × ℝ × ℝ := (3, -1, 2)

/-- The distance from a point to a line -/
def distanceToLine (P : ℝ × ℝ × ℝ) (Q : ℝ → ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ := 
  sorry

theorem distance_to_line : 
  distanceToLine P Q v = Real.sqrt 1262 / 7 := by sorry

end NUMINAMATH_CALUDE_distance_to_line_l1833_183388


namespace NUMINAMATH_CALUDE_power_of_fraction_cube_l1833_183368

theorem power_of_fraction_cube (x : ℝ) : ((1/2) * x^3)^2 = (1/4) * x^6 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_cube_l1833_183368


namespace NUMINAMATH_CALUDE_janet_action_figures_l1833_183340

/-- Calculates the final number of action figures Janet has -/
def final_action_figures (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  let after_selling := initial - sold
  let after_buying := after_selling + bought
  let brothers_collection := 2 * after_buying
  after_buying + brothers_collection

/-- Proves that Janet ends up with 24 action figures given the initial conditions -/
theorem janet_action_figures :
  final_action_figures 10 6 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_janet_action_figures_l1833_183340


namespace NUMINAMATH_CALUDE_tylers_eggs_l1833_183360

/-- Given a recipe for 4 people requiring 2 eggs, prove that if Tyler needs to buy 1 more egg
    to make a cake for 8 people, then Tyler has 3 eggs in the fridge. -/
theorem tylers_eggs (recipe_eggs : ℕ) (people : ℕ) (scale_factor : ℕ) (eggs_to_buy : ℕ) : 
  recipe_eggs = 2 →
  people = 8 →
  scale_factor = people / 4 →
  eggs_to_buy = 1 →
  recipe_eggs * scale_factor - eggs_to_buy = 3 := by
  sorry

end NUMINAMATH_CALUDE_tylers_eggs_l1833_183360


namespace NUMINAMATH_CALUDE_degenerate_ellipse_c_l1833_183374

/-- The equation of a potentially degenerate ellipse -/
def ellipse_equation (x y c : ℝ) : Prop :=
  2 * x^2 + y^2 + 8 * x - 10 * y + c = 0

/-- A degenerate ellipse is represented by a single point -/
def is_degenerate_ellipse (c : ℝ) : Prop :=
  ∃! (x y : ℝ), ellipse_equation x y c

/-- The value of c for which the ellipse is degenerate -/
theorem degenerate_ellipse_c : 
  ∃! c : ℝ, is_degenerate_ellipse c ∧ c = 33 :=
sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_c_l1833_183374


namespace NUMINAMATH_CALUDE_tangent_slope_angle_range_l1833_183304

theorem tangent_slope_angle_range :
  ∀ (x : ℝ),
  let y := x^3 - x + 2/3
  let slope := (3 * x^2 - 1 : ℝ)
  let α := Real.arctan slope
  α ∈ Set.union (Set.Ico 0 (π/2)) (Set.Icc (3*π/4) π) := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_range_l1833_183304


namespace NUMINAMATH_CALUDE_max_visible_cube_l1833_183343

/-- The size of the cube's edge -/
def n : ℕ := 13

/-- The number of unit cubes visible on one face -/
def face_visible : ℕ := n^2

/-- The number of unit cubes visible along one edge (excluding the corner) -/
def edge_visible : ℕ := n - 1

/-- The maximum number of unit cubes visible from a single point -/
def max_visible : ℕ := 3 * face_visible - 3 * edge_visible + 1

theorem max_visible_cube :
  max_visible = 472 :=
sorry

end NUMINAMATH_CALUDE_max_visible_cube_l1833_183343


namespace NUMINAMATH_CALUDE_course_selection_schemes_l1833_183370

theorem course_selection_schemes (pe art : ℕ) (total_courses : ℕ) : 
  pe = 4 → art = 4 → total_courses = pe + art →
  (Nat.choose pe 1 * Nat.choose art 1) + 
  (Nat.choose pe 2 * Nat.choose art 1 + Nat.choose pe 1 * Nat.choose art 2) = 64 :=
by sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l1833_183370


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1833_183327

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1833_183327


namespace NUMINAMATH_CALUDE_jose_play_time_l1833_183364

/-- Calculates the total hours played given the minutes spent on football and basketball -/
def total_hours_played (football_minutes : ℕ) (basketball_minutes : ℕ) : ℚ :=
  (football_minutes + basketball_minutes : ℚ) / 60

/-- Proves that given Jose played football for 30 minutes and basketball for 60 minutes, 
    the total time he played is equal to 1.5 hours -/
theorem jose_play_time : total_hours_played 30 60 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_jose_play_time_l1833_183364


namespace NUMINAMATH_CALUDE_fraction_relation_l1833_183362

theorem fraction_relation (p r t u : ℚ) 
  (h1 : p / r = 8)
  (h2 : t / r = 5)
  (h3 : t / u = 2 / 3) :
  u / p = 15 / 16 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l1833_183362


namespace NUMINAMATH_CALUDE_last_digit_3_count_l1833_183315

/-- The last digit of 7^n -/
def last_digit (n : ℕ) : ℕ := (7^n) % 10

/-- Whether the last digit of 7^n is 3 -/
def is_last_digit_3 (n : ℕ) : Prop := last_digit n = 3

/-- The number of terms in the sequence 7^1, 7^2, ..., 7^n whose last digit is 3 -/
def count_last_digit_3 (n : ℕ) : ℕ := (n + 3) / 4

theorem last_digit_3_count :
  count_last_digit_3 2009 = 502 :=
sorry

end NUMINAMATH_CALUDE_last_digit_3_count_l1833_183315


namespace NUMINAMATH_CALUDE_four_number_equation_solutions_l1833_183312

def is_solution (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  x₁ + x₂*x₃*x₄ = 2 ∧
  x₂ + x₁*x₃*x₄ = 2 ∧
  x₃ + x₁*x₂*x₄ = 2 ∧
  x₄ + x₁*x₂*x₃ = 2

theorem four_number_equation_solutions :
  ∀ x₁ x₂ x₃ x₄ : ℝ, is_solution x₁ x₂ x₃ x₄ ↔
    ((x₁, x₂, x₃, x₄) = (1, 1, 1, 1) ∨
     (x₁, x₂, x₃, x₄) = (-1, -1, -1, 3) ∨
     (x₁, x₂, x₃, x₄) = (-1, -1, 3, -1) ∨
     (x₁, x₂, x₃, x₄) = (-1, 3, -1, -1) ∨
     (x₁, x₂, x₃, x₄) = (3, -1, -1, -1)) :=
by sorry


end NUMINAMATH_CALUDE_four_number_equation_solutions_l1833_183312


namespace NUMINAMATH_CALUDE_repair_cost_is_2400_l1833_183300

/-- The total cost of car repairs given labor rate, labor hours, and part cost. -/
def total_repair_cost (labor_rate : ℕ) (labor_hours : ℕ) (part_cost : ℕ) : ℕ :=
  labor_rate * labor_hours + part_cost

/-- Theorem stating that the total repair cost is $2400 given the specified conditions. -/
theorem repair_cost_is_2400 :
  total_repair_cost 75 16 1200 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_is_2400_l1833_183300


namespace NUMINAMATH_CALUDE_rectangle_area_is_eight_l1833_183382

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a rectangle on a 2D grid -/
structure GridRectangle where
  bottomLeft : GridPoint
  topRight : GridPoint

/-- Calculates the area of a grid rectangle -/
def gridRectangleArea (rect : GridRectangle) : ℤ :=
  (rect.topRight.x - rect.bottomLeft.x) * (rect.topRight.y - rect.bottomLeft.y)

theorem rectangle_area_is_eight :
  let rect : GridRectangle := {
    bottomLeft := { x := 0, y := 0 },
    topRight := { x := 4, y := 2 }
  }
  gridRectangleArea rect = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_eight_l1833_183382


namespace NUMINAMATH_CALUDE_inner_triangle_area_l1833_183324

/-- Given a triangle with area T, prove that the area of the triangle formed by
    joining the points that divide each side into three equal segments is T/9. -/
theorem inner_triangle_area (T : ℝ) (h : T > 0) :
  ∃ (M : ℝ), M = T / 9 ∧ M / T = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_area_l1833_183324


namespace NUMINAMATH_CALUDE_inverse_exp_range_l1833_183349

noncomputable def f : ℝ → ℝ := Real.log

theorem inverse_exp_range (a b : ℝ) :
  (∀ x, f x = Real.log x) →
  (|f a| = |f b|) →
  (a ≠ b) →
  (∀ x > 2, ∃ a b : ℝ, a + b = x ∧ |f a| = |f b| ∧ a ≠ b) ∧
  (|f a| = |f b| ∧ a ≠ b → a + b > 2) :=
sorry

end NUMINAMATH_CALUDE_inverse_exp_range_l1833_183349


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l1833_183318

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + x

theorem f_derivative_at_one :
  deriv f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l1833_183318


namespace NUMINAMATH_CALUDE_aeroplane_distance_l1833_183307

theorem aeroplane_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) 
  (h1 : speed1 = 590)
  (h2 : time1 = 8)
  (h3 : speed2 = 1716.3636363636363)
  (h4 : time2 = 2.75)
  (h5 : speed1 * time1 = speed2 * time2) : 
  speed1 * time1 = 4720 := by
  sorry

end NUMINAMATH_CALUDE_aeroplane_distance_l1833_183307


namespace NUMINAMATH_CALUDE_sin_has_property_T_l1833_183306

-- Define property T
def has_property_T (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (deriv f x1) * (deriv f x2) = -1

-- State the theorem
theorem sin_has_property_T : has_property_T Real.sin := by
  sorry


end NUMINAMATH_CALUDE_sin_has_property_T_l1833_183306


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1833_183338

theorem quadratic_equation_roots (p : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + p * x - 6 = 0 ∧ x = -2) → 
  (∃ y : ℝ, 3 * y^2 + p * y - 6 = 0 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1833_183338


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l1833_183328

theorem quadratic_inequality_minimum (a b : ℝ) (h1 : a > b) 
  (h2 : ∀ x, (a*x^2 + 2*x + b > 0) ↔ (x ≠ -1/a)) :
  ∃ m : ℝ, m = 6 ∧ ∀ x, x = (a^2 + b^2 + 7)/(a - b) → x ≥ m := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l1833_183328


namespace NUMINAMATH_CALUDE_student_number_exists_l1833_183385

theorem student_number_exists : ∃ x : ℝ, Real.sqrt (2 * x^2 - 138) = 9 := by
  sorry

end NUMINAMATH_CALUDE_student_number_exists_l1833_183385


namespace NUMINAMATH_CALUDE_percentage_loss_l1833_183367

theorem percentage_loss (cost_price selling_price : ℝ) 
  (h1 : cost_price = 1400)
  (h2 : selling_price = 1050) : 
  (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_loss_l1833_183367


namespace NUMINAMATH_CALUDE_evaluate_expression_l1833_183357

theorem evaluate_expression : 5^2 + 2*(5 - 2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1833_183357


namespace NUMINAMATH_CALUDE_base_conversion_512_l1833_183323

/-- Converts a base-10 number to its base-6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

theorem base_conversion_512 :
  toBase6 512 = [2, 2, 1, 2] :=
sorry

end NUMINAMATH_CALUDE_base_conversion_512_l1833_183323


namespace NUMINAMATH_CALUDE_dart_probability_l1833_183378

/-- Represents an equilateral triangle divided into regions -/
structure DividedTriangle where
  total_regions : ℕ
  shaded_regions : ℕ
  h_positive : 0 < total_regions
  h_shaded_le_total : shaded_regions ≤ total_regions

/-- The probability of a dart landing in a shaded region -/
def shaded_probability (triangle : DividedTriangle) : ℚ :=
  triangle.shaded_regions / triangle.total_regions

/-- The specific triangle described in the problem -/
def problem_triangle : DividedTriangle where
  total_regions := 6
  shaded_regions := 3
  h_positive := by norm_num
  h_shaded_le_total := by norm_num

theorem dart_probability :
  shaded_probability problem_triangle = 1/2 := by sorry

end NUMINAMATH_CALUDE_dart_probability_l1833_183378


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1833_183341

/-- The perimeter of a rhombus with diagonals of 8 inches and 30 inches is 4√241 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 4 * Real.sqrt 241 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1833_183341


namespace NUMINAMATH_CALUDE_correct_sum_after_card_swap_l1833_183321

theorem correct_sum_after_card_swap : 
  ∃ (a b : ℕ), 
    (a + b = 81380) ∧ 
    (a ≠ 37541 ∨ b ≠ 43839) ∧
    (∃ (x y : ℕ), (x = 37541 ∧ y = 43839) ∧ (x + y = 80280)) :=
by sorry

end NUMINAMATH_CALUDE_correct_sum_after_card_swap_l1833_183321


namespace NUMINAMATH_CALUDE_wire_shapes_l1833_183369

/-- Given a wire of length 28 cm, prove properties about shapes formed from it -/
theorem wire_shapes (wire_length : ℝ) (h_wire : wire_length = 28) :
  let square_side : ℝ := wire_length / 4
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := wire_length / 2 - rectangle_length
  (square_side = 7 ∧ rectangle_width = 2) := by
  sorry

#check wire_shapes

end NUMINAMATH_CALUDE_wire_shapes_l1833_183369


namespace NUMINAMATH_CALUDE_alligator_journey_time_l1833_183393

theorem alligator_journey_time (initial_time : ℝ) (return_time : ℝ) : 
  initial_time = 4 →
  return_time = initial_time + 2 * Real.sqrt initial_time →
  initial_time + return_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_alligator_journey_time_l1833_183393


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l1833_183381

-- Define the lines
def line1 (a x y : ℝ) : Prop := (a - 1) * x + 2 * y + 1 = 0
def line2 (a x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define parallelism
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂, f x₁ y₁ → f x₂ y₂ → g x₁ y₁ → g x₂ y₂ → 
    (y₂ - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x₂ - x₁)

-- Theorem statement
theorem parallel_lines_condition (a : ℝ) :
  parallel (line1 a) (line2 a) → a = -1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l1833_183381


namespace NUMINAMATH_CALUDE_no_solution_for_sock_problem_l1833_183366

theorem no_solution_for_sock_problem :
  ¬∃ (n m : ℕ), n + m = 2009 ∧ 
  (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1) : ℚ) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_sock_problem_l1833_183366


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1833_183363

theorem min_value_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 4) : 
  (1 / x + 4 / y + 9 / z) ≥ 9 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 4 ∧ 1 / x + 4 / y + 9 / z = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1833_183363


namespace NUMINAMATH_CALUDE_new_mean_after_removal_l1833_183379

def original_mean : ℝ := 42
def original_count : ℕ := 65
def removed_score : ℝ := 50
def removed_count : ℕ := 6

theorem new_mean_after_removal :
  let original_sum := original_mean * original_count
  let removed_sum := removed_score * removed_count
  let new_sum := original_sum - removed_sum
  let new_count := original_count - removed_count
  let new_mean := new_sum / new_count
  ∃ ε > 0, |new_mean - 41.2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_new_mean_after_removal_l1833_183379


namespace NUMINAMATH_CALUDE_cindys_calculation_l1833_183308

theorem cindys_calculation (x : ℚ) : 4 * (x / 2 - 6) = 24 → (2 * x - 4) / 6 = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l1833_183308


namespace NUMINAMATH_CALUDE_cistern_width_is_six_l1833_183316

/-- Represents the dimensions and properties of a rectangular cistern --/
structure Cistern where
  length : ℝ
  width : ℝ
  waterDepth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the total wet surface area of a cistern --/
def totalWetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.waterDepth + 2 * c.width * c.waterDepth

/-- Theorem stating that a cistern with given dimensions has a width of 6 meters --/
theorem cistern_width_is_six (c : Cistern) 
  (h1 : c.length = 9)
  (h2 : c.waterDepth = 2.25)
  (h3 : c.wetSurfaceArea = 121.5)
  (h4 : totalWetSurfaceArea c = c.wetSurfaceArea) : 
  c.width = 6 := by
  sorry

end NUMINAMATH_CALUDE_cistern_width_is_six_l1833_183316


namespace NUMINAMATH_CALUDE_not_parabola_l1833_183359

/-- The equation x² + ky² = 1 cannot represent a parabola for any real k -/
theorem not_parabola (k : ℝ) : 
  ¬ ∃ (a b c d e : ℝ), ∀ (x y : ℝ), 
    (x^2 + k*y^2 = 1) ↔ (a*x^2 + b*x*y + c*y^2 + d*x + e*y = 0 ∧ b^2 = 4*a*c) :=
sorry

end NUMINAMATH_CALUDE_not_parabola_l1833_183359


namespace NUMINAMATH_CALUDE_largest_points_with_empty_square_fifteen_points_optimal_l1833_183337

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in 2D space -/
structure Square where
  center : Point
  side_length : ℝ

/-- Checks if a point is inside a square -/
def is_point_inside_square (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side_length / 2 ∧
  abs (p.y - s.center.y) ≤ s.side_length / 2

/-- The main theorem -/
theorem largest_points_with_empty_square :
  ∀ (points : List Point),
    (∀ p ∈ points, 0 < p.x ∧ p.x < 4 ∧ 0 < p.y ∧ p.y < 4) →
    points.length ≤ 15 →
    ∃ (s : Square),
      s.side_length = 1 ∧
      0 ≤ s.center.x ∧ s.center.x ≤ 3 ∧
      0 ≤ s.center.y ∧ s.center.y ≤ 3 ∧
      ∀ p ∈ points, ¬is_point_inside_square p s :=
by sorry

/-- The optimality of 15 -/
theorem fifteen_points_optimal :
  ∃ (points : List Point),
    points.length = 16 ∧
    (∀ p ∈ points, 0 < p.x ∧ p.x < 4 ∧ 0 < p.y ∧ p.y < 4) ∧
    ∀ (s : Square),
      s.side_length = 1 →
      0 ≤ s.center.x ∧ s.center.x ≤ 3 →
      0 ≤ s.center.y ∧ s.center.y ≤ 3 →
      ∃ p ∈ points, is_point_inside_square p s :=
by sorry

end NUMINAMATH_CALUDE_largest_points_with_empty_square_fifteen_points_optimal_l1833_183337


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1833_183397

theorem absolute_value_equation : ∀ x : ℝ, 
  (abs x) * (abs (-25) - abs 5) = 40 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1833_183397


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l1833_183339

theorem least_positive_integer_multiple_of_53 :
  ∀ x : ℕ+, x < 21 → ¬(∃ k : ℤ, (3*x)^2 + 2*43*3*x + 43^2 = 53*k) ∧
  ∃ k : ℤ, (3*21)^2 + 2*43*3*21 + 43^2 = 53*k :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l1833_183339


namespace NUMINAMATH_CALUDE_quadratic_form_with_factor_l1833_183334

/-- A quadratic expression with (x + 3) as a factor and m = 2 -/
def quadratic_expression (c : ℝ) (x : ℝ) : ℝ :=
  2 * (x + 3) * (x + c)

/-- Theorem stating the form of the quadratic expression -/
theorem quadratic_form_with_factor (f : ℝ → ℝ) :
  (∃ (g : ℝ → ℝ), ∀ x, f x = (x + 3) * g x) →  -- (x + 3) is a factor
  (∃ c, ∀ x, f x = quadratic_expression c x) :=
by
  sorry

#check quadratic_form_with_factor

end NUMINAMATH_CALUDE_quadratic_form_with_factor_l1833_183334


namespace NUMINAMATH_CALUDE_f_max_value_l1833_183313

/-- The function f(x) = 9x - 4x^2 -/
def f (x : ℝ) : ℝ := 9*x - 4*x^2

/-- The maximum value of f(x) is 81/16 -/
theorem f_max_value : ∀ x : ℝ, f x ≤ 81/16 := by sorry

end NUMINAMATH_CALUDE_f_max_value_l1833_183313
