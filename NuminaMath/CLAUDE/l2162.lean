import Mathlib

namespace NUMINAMATH_CALUDE_smallest_h_divisible_l2162_216295

theorem smallest_h_divisible : ∃! h : ℕ, 
  (∀ k : ℕ, k < h → ¬((k + 5) % 8 = 0 ∧ (k + 5) % 11 = 0 ∧ (k + 5) % 24 = 0)) ∧
  (h + 5) % 8 = 0 ∧ (h + 5) % 11 = 0 ∧ (h + 5) % 24 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_h_divisible_l2162_216295


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l2162_216207

/-- Given points A, B, C, and F as the midpoint of AC, prove that the sum of the slope
    and y-intercept of the line passing through F and B is 3/4 -/
theorem slope_intercept_sum (A B C F : ℝ × ℝ) : 
  A = (0, 6) →
  B = (0, 0) →
  C = (8, 0) →
  F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  let m := (F.2 - B.2) / (F.1 - B.1)
  let b := B.2
  m + b = 3/4 := by sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l2162_216207


namespace NUMINAMATH_CALUDE_joan_has_eight_kittens_l2162_216211

/-- The number of kittens Joan has at the end, given the initial conditions and actions. -/
def joans_final_kittens (joan_initial : ℕ) (neighbor_initial : ℕ) 
  (joan_gave_away : ℕ) (neighbor_gave_away : ℕ) (joan_wants_to_adopt : ℕ) : ℕ :=
  let joan_after_giving := joan_initial - joan_gave_away
  let neighbor_after_giving := neighbor_initial - neighbor_gave_away
  let joan_can_adopt := min joan_wants_to_adopt neighbor_after_giving
  joan_after_giving + joan_can_adopt

/-- Theorem stating that Joan ends up with 8 kittens given the specific conditions. -/
theorem joan_has_eight_kittens : 
  joans_final_kittens 8 6 2 4 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_eight_kittens_l2162_216211


namespace NUMINAMATH_CALUDE_banana_bunches_l2162_216213

theorem banana_bunches (total_bananas : ℕ) (eight_bunch_count : ℕ) (bananas_per_eight_bunch : ℕ) : 
  total_bananas = 83 →
  eight_bunch_count = 6 →
  bananas_per_eight_bunch = 8 →
  ∃ (seven_bunch_count : ℕ),
    seven_bunch_count * 7 + eight_bunch_count * bananas_per_eight_bunch = total_bananas ∧
    seven_bunch_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_banana_bunches_l2162_216213


namespace NUMINAMATH_CALUDE_equation_solution_l2162_216268

theorem equation_solution (x : ℝ) : 
  (1 / x + 1 / (x + 2) - 1 / (x + 4) - 1 / (x + 6) - 1 / (x + 8) - 1 / (x + 10) + 1 / (x + 12) + 1 / (x + 14) = 0) ↔ 
  (x = -7 ∨ x = -7 + Real.sqrt (19 + 6 * Real.sqrt 5) ∨ 
   x = -7 - Real.sqrt (19 + 6 * Real.sqrt 5) ∨ 
   x = -7 + Real.sqrt (19 - 6 * Real.sqrt 5) ∨ 
   x = -7 - Real.sqrt (19 - 6 * Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2162_216268


namespace NUMINAMATH_CALUDE_hcf_of_three_numbers_l2162_216275

theorem hcf_of_three_numbers (a b c : ℕ+) :
  (Nat.lcm (Nat.lcm a.val b.val) c.val = 1200) →
  (a.val * b.val * c.val = 108000) →
  (Nat.gcd (Nat.gcd a.val b.val) c.val = 90) := by
sorry

end NUMINAMATH_CALUDE_hcf_of_three_numbers_l2162_216275


namespace NUMINAMATH_CALUDE_lcm_12_18_l2162_216294

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l2162_216294


namespace NUMINAMATH_CALUDE_a_range_l2162_216247

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem a_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 a, f x ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 a, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 a, f x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc 0 a, f x = 2) →
  a ∈ Set.Icc 1 2 :=
by sorry


end NUMINAMATH_CALUDE_a_range_l2162_216247


namespace NUMINAMATH_CALUDE_binary_arithmetic_proof_l2162_216293

theorem binary_arithmetic_proof : 
  let a : ℕ := 0b1100101
  let b : ℕ := 0b1101
  let c : ℕ := 0b101
  let result : ℕ := 0b11111010
  (a * b) / c = result := by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_proof_l2162_216293


namespace NUMINAMATH_CALUDE_tangent_segment_area_l2162_216222

theorem tangent_segment_area (r : ℝ) (l : ℝ) (h_r : r = 3) (h_l : l = 6) :
  let outer_radius := (r^2 + (l/2)^2).sqrt
  (π * outer_radius^2 - π * r^2) = 9 * π := by sorry

end NUMINAMATH_CALUDE_tangent_segment_area_l2162_216222


namespace NUMINAMATH_CALUDE_unit_vectors_equal_magnitude_l2162_216223

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors a and b
variable (a b : V)

-- State the theorem
theorem unit_vectors_equal_magnitude
  (ha : ‖a‖ = 1) -- a is a unit vector
  (hb : ‖b‖ = 1) -- b is a unit vector
  : ‖a‖ = ‖b‖ := by
  sorry

end NUMINAMATH_CALUDE_unit_vectors_equal_magnitude_l2162_216223


namespace NUMINAMATH_CALUDE_mean_temperature_l2162_216256

def temperatures : List ℤ := [-6, -3, -3, -4, 2, 4, 1]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -9/7 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l2162_216256


namespace NUMINAMATH_CALUDE_condition_equivalence_l2162_216267

theorem condition_equivalence :
  (∀ x y : ℝ, x > y ↔ x^3 > y^3) ∧
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧
  (∃ x y : ℝ, x^2 > y^2 ∧ x ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_condition_equivalence_l2162_216267


namespace NUMINAMATH_CALUDE_birth_probability_l2162_216215

theorem birth_probability (n : ℕ) (p : ℝ) (h1 : n = 5) (h2 : p = 1/2) :
  let prob_all_same := p^n
  let prob_three_two := (n.choose 3) * p^n
  prob_three_two > prob_all_same :=
by sorry

end NUMINAMATH_CALUDE_birth_probability_l2162_216215


namespace NUMINAMATH_CALUDE_train_length_l2162_216282

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 122 →  -- speed in km/hr
  time = 4.425875438161669 →  -- time in seconds
  speed * (5 / 18) * time = 150 :=  -- length in meters
by sorry

end NUMINAMATH_CALUDE_train_length_l2162_216282


namespace NUMINAMATH_CALUDE_female_managers_count_l2162_216243

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  male_employees : ℕ
  female_employees : ℕ
  total_managers : ℕ
  male_managers : ℕ
  female_managers : ℕ

/-- Conditions for the company -/
def ValidCompany (c : Company) : Prop :=
  c.female_employees = 1000 ∧
  c.total_employees = c.male_employees + c.female_employees ∧
  c.total_managers = c.male_managers + c.female_managers ∧
  5 * c.total_managers = 2 * c.total_employees ∧
  5 * c.male_managers = 2 * c.male_employees

/-- Theorem stating that in a valid company, the number of female managers is 400 -/
theorem female_managers_count (c : Company) (h : ValidCompany c) :
  c.female_managers = 400 := by
  sorry


end NUMINAMATH_CALUDE_female_managers_count_l2162_216243


namespace NUMINAMATH_CALUDE_complex_equation_proof_l2162_216289

theorem complex_equation_proof (a b : ℝ) : (a - 2 * I) / I = b + I → a - b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l2162_216289


namespace NUMINAMATH_CALUDE_one_fifth_of_eight_point_five_l2162_216241

theorem one_fifth_of_eight_point_five : (8.5 : ℚ) / 5 = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_eight_point_five_l2162_216241


namespace NUMINAMATH_CALUDE_fraction_addition_l2162_216296

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2162_216296


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2162_216226

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

theorem solution_set_of_inequality 
  (f : ℝ → ℝ) 
  (h_decreasing : is_decreasing f) 
  (h_point1 : f 0 = 1) 
  (h_point2 : f 3 = -1) :
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2162_216226


namespace NUMINAMATH_CALUDE_integer_ratio_problem_l2162_216261

theorem integer_ratio_problem (a b : ℤ) :
  1996 * a + b / 96 = a + b →
  b / a = 2016 ∨ a / b = 1 / 2016 := by
sorry

end NUMINAMATH_CALUDE_integer_ratio_problem_l2162_216261


namespace NUMINAMATH_CALUDE_quadratic_polynomial_condition_l2162_216209

/-- A second degree polynomial -/
structure QuadraticPolynomial where
  u : ℝ
  v : ℝ
  w : ℝ

/-- Evaluation of a quadratic polynomial at a point x -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.u * x^2 + p.v * x + p.w

theorem quadratic_polynomial_condition (p : QuadraticPolynomial) :
  (∀ a : ℝ, a ≥ 1 → p.eval (a^2 + a) ≥ a * p.eval (a + 1)) ↔
  (p.u > 0 ∧ p.w ≤ 4 * p.u) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_condition_l2162_216209


namespace NUMINAMATH_CALUDE_sequence_sum_divisible_by_five_l2162_216201

/-- Represents a four-digit integer -/
structure FourDigitInt where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : thousands ≥ 1 ∧ thousands ≤ 9 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Represents a sequence of four FourDigitInts with the given property -/
structure SpecialSequence where
  term1 : FourDigitInt
  term2 : FourDigitInt
  term3 : FourDigitInt
  term4 : FourDigitInt
  property : term2.hundreds = term1.tens ∧ term2.tens = term1.units ∧
             term3.hundreds = term2.tens ∧ term3.tens = term2.units ∧
             term4.hundreds = term3.tens ∧ term4.tens = term3.units ∧
             term1.hundreds = term4.tens ∧ term1.tens = term4.units

/-- Calculates the sum of all terms in the sequence -/
def sequenceSum (seq : SpecialSequence) : Nat :=
  let toNum (t : FourDigitInt) := t.thousands * 1000 + t.hundreds * 100 + t.tens * 10 + t.units
  toNum seq.term1 + toNum seq.term2 + toNum seq.term3 + toNum seq.term4

theorem sequence_sum_divisible_by_five (seq : SpecialSequence) :
  ∃ k : Nat, sequenceSum seq = 5 * k := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_divisible_by_five_l2162_216201


namespace NUMINAMATH_CALUDE_function_identity_l2162_216252

theorem function_identity (f : ℝ → ℝ) (h : ∀ x, f (2*x + 1) = 4*x^2 + 4*x) :
  ∀ x, f x = x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2162_216252


namespace NUMINAMATH_CALUDE_t_shirt_cost_l2162_216266

/-- Calculates the cost of each t-shirt given the total cost, number of t-shirts and pants, and cost of each pair of pants. -/
theorem t_shirt_cost (total_cost : ℕ) (num_tshirts num_pants pants_cost : ℕ) :
  total_cost = 1500 ∧ 
  num_tshirts = 5 ∧ 
  num_pants = 4 ∧ 
  pants_cost = 250 →
  (total_cost - num_pants * pants_cost) / num_tshirts = 100 := by
  sorry

end NUMINAMATH_CALUDE_t_shirt_cost_l2162_216266


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2162_216260

def arithmetic_sequence (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ := a1 + (n - 1) * d

def sum_arithmetic_sequence (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_geometric_sequence (d : ℚ) (h : d ≠ 0) :
  let a := arithmetic_sequence 1 d
  (a 2) * (a 9) = (a 4)^2 →
  (∃ q : ℚ, q = 5/2 ∧
    (∀ n : ℕ, arithmetic_sequence 1 d n = 3*n - 2) ∧
    (∀ n : ℕ, sum_arithmetic_sequence 1 d n = (3*n^2 - n) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2162_216260


namespace NUMINAMATH_CALUDE_vertical_tangents_condition_l2162_216220

/-- The function f(x) = x(a - 1/e^x) has two distinct points with vertical tangents
    if and only if a is in the open interval (0, 2/e) -/
theorem vertical_tangents_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, (x * (a - Real.exp (-x))) = 0 → x = x₁ ∨ x = x₂) ∧
    (∀ x : ℝ, (a - (1 + x) * Real.exp (-x)) = 0 → x = x₁ ∨ x = x₂)) ↔
  (0 < a ∧ a < 2 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_vertical_tangents_condition_l2162_216220


namespace NUMINAMATH_CALUDE_area_triangle_DEF_l2162_216274

/-- Triangle DEF with hypotenuse DE, angle between DF and DE is 45°, and length of DF is 4 units -/
structure Triangle_DEF where
  DE : ℝ  -- Length of hypotenuse DE
  DF : ℝ  -- Length of side DF
  EF : ℝ  -- Length of side EF
  angle_DF_DE : ℝ  -- Angle between DF and DE in radians
  hypotenuse_DE : DE = DF * Real.sqrt 2  -- DE is hypotenuse
  angle_45_deg : angle_DF_DE = π / 4  -- Angle is 45°
  DF_length : DF = 4  -- Length of DF is 4 units

/-- The area of triangle DEF is 8 square units -/
theorem area_triangle_DEF (t : Triangle_DEF) : (1 / 2) * t.DF * t.EF = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_DEF_l2162_216274


namespace NUMINAMATH_CALUDE_line_equation_proof_l2162_216278

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_equation_proof (l : Line) :
  l.contains (0, 3) ∧
  l.perpendicular ⟨1, 1, 1⟩ →
  l = ⟨1, -1, 3⟩ := by
  sorry

#check line_equation_proof

end NUMINAMATH_CALUDE_line_equation_proof_l2162_216278


namespace NUMINAMATH_CALUDE_farm_feet_count_l2162_216270

/-- Represents a farm with hens and cows -/
structure Farm where
  total_heads : ℕ
  num_hens : ℕ

/-- Calculates the total number of feet in the farm -/
def total_feet (f : Farm) : ℕ :=
  2 * f.num_hens + 4 * (f.total_heads - f.num_hens)

/-- Theorem stating that a farm with 48 total heads and 26 hens has 140 feet -/
theorem farm_feet_count : 
  ∀ (f : Farm), f.total_heads = 48 → f.num_hens = 26 → total_feet f = 140 := by
  sorry


end NUMINAMATH_CALUDE_farm_feet_count_l2162_216270


namespace NUMINAMATH_CALUDE_tom_rides_11860_miles_l2162_216264

/-- Tom's daily bike riding distance for the first part of the year -/
def first_part_distance : ℕ := 30

/-- Number of days in the first part of the year -/
def first_part_days : ℕ := 183

/-- Tom's daily bike riding distance for the second part of the year -/
def second_part_distance : ℕ := 35

/-- Total number of days in a year -/
def total_days : ℕ := 365

/-- Calculate the total miles Tom rides in a year -/
def total_miles : ℕ := first_part_distance * first_part_days + 
                        second_part_distance * (total_days - first_part_days)

theorem tom_rides_11860_miles : total_miles = 11860 := by
  sorry

end NUMINAMATH_CALUDE_tom_rides_11860_miles_l2162_216264


namespace NUMINAMATH_CALUDE_square_diagonal_length_l2162_216284

theorem square_diagonal_length (side_length : ℝ) (h : side_length = 30 * Real.sqrt 3) :
  Real.sqrt (2 * side_length ^ 2) = 30 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_length_l2162_216284


namespace NUMINAMATH_CALUDE_subtraction_problem_l2162_216297

theorem subtraction_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2162_216297


namespace NUMINAMATH_CALUDE_sphere_radii_formula_l2162_216254

/-- Given three mutually tangent spheres touched by a plane at points A, B, and C,
    where the sides of triangle ABC are a, b, and c, prove that the radii of the
    spheres (x, y, z) are given by the formulas stated. -/
theorem sphere_radii_formula (a b c x y z : ℝ) 
  (h1 : a = 2 * Real.sqrt (x * y))
  (h2 : b = 2 * Real.sqrt (y * z))
  (h3 : c = 2 * Real.sqrt (x * z))
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  x = a * c / (2 * b) ∧ 
  y = a * b / (2 * c) ∧ 
  z = b * c / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radii_formula_l2162_216254


namespace NUMINAMATH_CALUDE_martha_blocks_theorem_l2162_216262

/-- The number of blocks Martha starts with -/
def starting_blocks : ℕ := 11

/-- The number of blocks Martha finds -/
def found_blocks : ℕ := 129

/-- The total number of blocks Martha ends up with -/
def total_blocks : ℕ := starting_blocks + found_blocks

theorem martha_blocks_theorem : total_blocks = 140 := by
  sorry

end NUMINAMATH_CALUDE_martha_blocks_theorem_l2162_216262


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2162_216291

theorem fraction_multiplication :
  (2 : ℚ) / 3 * 5 / 7 * 9 / 13 * 4 / 11 = 120 / 1001 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2162_216291


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2162_216272

theorem polynomial_coefficient_sum (A B C D : ℚ) : 
  (∀ x : ℚ, (x + 3) * (4 * x^2 - 2 * x + 6) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 32 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2162_216272


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2162_216269

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the angles and side lengths
def angle (t : Triangle) (v1 v2 v3 : ℝ × ℝ) : ℝ := sorry

def side_length (a b : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter
def perimeter (t : Triangle) : ℝ :=
  side_length t.X t.Y + side_length t.Y t.Z + side_length t.Z t.X

-- State the theorem
theorem triangle_perimeter (t : Triangle) :
  angle t t.X t.Y t.Z = angle t t.X t.Z t.Y →
  side_length t.Y t.Z = 8 →
  side_length t.X t.Z = 10 →
  perimeter t = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2162_216269


namespace NUMINAMATH_CALUDE_stratified_sampling_l2162_216255

theorem stratified_sampling (first_grade : ℕ) (second_grade : ℕ) (sample_size : ℕ) (third_grade_sampled : ℕ) :
  first_grade = 24 →
  second_grade = 36 →
  sample_size = 20 →
  third_grade_sampled = 10 →
  ∃ (total_parts : ℕ) (third_grade : ℕ) (second_grade_sampled : ℕ),
    total_parts = first_grade + second_grade + third_grade ∧
    third_grade = 60 ∧
    second_grade_sampled = 6 ∧
    (third_grade : ℚ) / total_parts = (third_grade_sampled : ℚ) / sample_size ∧
    (second_grade : ℚ) / total_parts = (second_grade_sampled : ℚ) / sample_size :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2162_216255


namespace NUMINAMATH_CALUDE_sum_coordinates_endpoint_l2162_216214

/-- Given a line segment CD with midpoint M(4,7) and endpoint C(6,2),
    the sum of coordinates of the other endpoint D is 14. -/
theorem sum_coordinates_endpoint (C D M : ℝ × ℝ) : 
  C = (6, 2) → M = (4, 7) → M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_endpoint_l2162_216214


namespace NUMINAMATH_CALUDE_root_value_implies_m_l2162_216217

theorem root_value_implies_m (m : ℝ) : (∃ x : ℝ, x^2 + m*x - 3 = 0 ∧ x = 1) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_value_implies_m_l2162_216217


namespace NUMINAMATH_CALUDE_red_balls_count_l2162_216240

theorem red_balls_count (total : ℕ) (white green yellow purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  white = 22 ∧
  green = 18 ∧
  yellow = 5 ∧
  purple = 9 ∧
  prob = 3/4 ∧
  (white + green + yellow : ℚ) / total = prob →
  total - (white + green + yellow + purple) = 6 :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l2162_216240


namespace NUMINAMATH_CALUDE_trapezoid_base_lengths_l2162_216228

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  b : ℝ  -- smaller base
  h : ℝ  -- altitude
  B : ℝ  -- larger base
  d : ℝ  -- common difference
  arithmetic_progression : b = h - 2 * d ∧ B = h + 2 * d
  area : (b + B) * h / 2 = 48

/-- Theorem stating the base lengths of the trapezoid -/
theorem trapezoid_base_lengths (t : Trapezoid) : 
  t.b = Real.sqrt 48 - 2 * Real.sqrt 3 ∧ 
  t.B = Real.sqrt 48 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_base_lengths_l2162_216228


namespace NUMINAMATH_CALUDE_max_pairs_with_distinct_sums_l2162_216265

/-- Given a set of integers from 1 to 2010, we can choose at most 803 pairs
    such that the elements of each pair are distinct, no two pairs share an element,
    and the sum of each pair is unique and not greater than 2010. -/
theorem max_pairs_with_distinct_sums :
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 803 ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ Finset.range 2010 ∧ p.2 ∈ Finset.range 2010) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 2010) ∧
    pairs.card = k ∧
    (∀ (m : ℕ) (other_pairs : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ other_pairs → p.1 ∈ Finset.range 2010 ∧ p.2 ∈ Finset.range 2010) →
      (∀ (p : ℕ × ℕ), p ∈ other_pairs → p.1 < p.2) →
      (∀ (p q : ℕ × ℕ), p ∈ other_pairs → q ∈ other_pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p q : ℕ × ℕ), p ∈ other_pairs → q ∈ other_pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      (∀ (p : ℕ × ℕ), p ∈ other_pairs → p.1 + p.2 ≤ 2010) →
      other_pairs.card = m →
      m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_with_distinct_sums_l2162_216265


namespace NUMINAMATH_CALUDE_proposition_truth_count_l2162_216204

theorem proposition_truth_count : 
  let P1 := ∀ x : ℝ, x > -3 → x > -6
  let P2 := ∀ x : ℝ, x > -6 → x > -3
  let P3 := ∀ x : ℝ, x ≤ -3 → x ≤ -6
  let P4 := ∀ x : ℝ, x ≤ -6 → x ≤ -3
  (P1 ∧ ¬P2 ∧ ¬P3 ∧ P4) ∨
  (P1 ∧ ¬P2 ∧ P3 ∧ ¬P4) ∨
  (P1 ∧ P2 ∧ ¬P3 ∧ ¬P4) ∨
  (¬P1 ∧ P2 ∧ ¬P3 ∧ P4) ∨
  (¬P1 ∧ P2 ∧ P3 ∧ ¬P4) ∨
  (¬P1 ∧ ¬P2 ∧ P3 ∧ P4) :=
by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_count_l2162_216204


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l2162_216299

/-- Given two people moving in opposite directions, with one person's speed and their final separation known, prove the other person's speed. -/
theorem opposite_direction_speed 
  (riya_speed : ℝ) 
  (total_separation : ℝ) 
  (time : ℝ) 
  (h1 : riya_speed = 21)
  (h2 : total_separation = 43)
  (h3 : time = 1) :
  let priya_speed := total_separation / time - riya_speed
  priya_speed = 22 := by sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l2162_216299


namespace NUMINAMATH_CALUDE_unique_g_property_l2162_216216

theorem unique_g_property : ∃! (g : ℕ+), 
  (∀ (p : ℕ) (hp : Nat.Prime p) (ho : Odd p), 
    ∃ (n : ℕ+), 
      (p ∣ g.val^n.val - n.val) ∧ 
      (p ∣ g.val^(n.val + 1) - (n.val + 1))) ∧ 
  g.val = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_g_property_l2162_216216


namespace NUMINAMATH_CALUDE_ticket_cost_theorem_l2162_216290

def adult_price : ℝ := 11
def child_price : ℝ := 8
def senior_price : ℝ := 9

def husband_discount : ℝ := 0.25
def parents_discount : ℝ := 0.15
def nephew_discount : ℝ := 0.10
def sister_discount : ℝ := 0.30

def num_adults : ℕ := 5
def num_children : ℕ := 4
def num_seniors : ℕ := 3

def total_cost : ℝ :=
  (adult_price * (1 - husband_discount) + adult_price) +  -- Mrs. Lopez and husband
  (senior_price * 2 * (1 - parents_discount)) +           -- Parents
  (child_price * 3 + child_price + adult_price * (1 - nephew_discount)) + -- Children and nephews
  senior_price +                                          -- Aunt (buy-one-get-one-free)
  (adult_price * 2) +                                     -- Two friends
  (adult_price * (1 - sister_discount))                   -- Sister

theorem ticket_cost_theorem : total_cost = 115.15 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_theorem_l2162_216290


namespace NUMINAMATH_CALUDE_marble_distribution_l2162_216258

theorem marble_distribution (n : ℕ) (hn : n = 720) :
  (Finset.filter (fun m => m > 1 ∧ m < n ∧ n % m = 0) (Finset.range (n + 1))).card = 28 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l2162_216258


namespace NUMINAMATH_CALUDE_system_solution_l2162_216263

theorem system_solution (x y z u : ℝ) : 
  (x^3 * y^2 * z = 2 ∧ 
   z^3 * u^2 * x = 32 ∧ 
   y^3 * z^2 * u = 8 ∧ 
   u^3 * x^2 * y = 8) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 2 ∧ u = 2) ∨
   (x = 1 ∧ y = -1 ∧ z = 2 ∧ u = -2) ∨
   (x = -1 ∧ y = 1 ∧ z = -2 ∧ u = 2) ∨
   (x = -1 ∧ y = -1 ∧ z = -2 ∧ u = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2162_216263


namespace NUMINAMATH_CALUDE_exactly_two_identical_pairs_l2162_216245

/-- Two lines in the xy-plane -/
structure TwoLines where
  a : ℝ
  d : ℝ

/-- The condition for two lines to be identical -/
def are_identical (l : TwoLines) : Prop :=
  (4 / l.a = -l.d / 3) ∧ (l.d / l.a = 6)

/-- The theorem stating that there are exactly two pairs (a, d) that make the lines identical -/
theorem exactly_two_identical_pairs :
  ∃! (s : Finset TwoLines), (∀ l ∈ s, are_identical l) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_identical_pairs_l2162_216245


namespace NUMINAMATH_CALUDE_total_pies_is_119_l2162_216200

/-- The number of pies Eddie can bake in a day -/
def eddie_daily_pies : ℕ := 3

/-- The number of pies Eddie's sister can bake in a day -/
def sister_daily_pies : ℕ := 6

/-- The number of pies Eddie's mother can bake in a day -/
def mother_daily_pies : ℕ := 8

/-- The number of days they will bake pies -/
def baking_days : ℕ := 7

/-- The total number of pies baked by Eddie, his sister, and his mother in 7 days -/
def total_pies : ℕ := (eddie_daily_pies + sister_daily_pies + mother_daily_pies) * baking_days

theorem total_pies_is_119 : total_pies = 119 := by
  sorry

end NUMINAMATH_CALUDE_total_pies_is_119_l2162_216200


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l2162_216206

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (4 * x = x^2 - 8) ↔ (x^2 - 4*x - 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l2162_216206


namespace NUMINAMATH_CALUDE_skier_race_l2162_216279

/-- Two skiers race with given conditions -/
theorem skier_race (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (9 / y + 9 = 9 / x) ∧ (29 / y + 9 = 25 / x) → y = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_skier_race_l2162_216279


namespace NUMINAMATH_CALUDE_intersection_line_circle_l2162_216232

/-- Given a line intersecting a circle, prove the value of parameter a -/
theorem intersection_line_circle (a : ℝ) : 
  ∃ (A B : ℝ × ℝ),
    (∀ (x y : ℝ), x - y + 2*a = 0 → x^2 + y^2 - 2*a*y - 2 = 0 → (x, y) = A ∨ (x, y) = B) ∧
    ‖A - B‖ = 4 * Real.sqrt 3 →
    a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l2162_216232


namespace NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l2162_216230

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 300! is 74 -/
theorem factorial_300_trailing_zeros :
  trailingZeros 300 = 74 := by
  sorry

end NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l2162_216230


namespace NUMINAMATH_CALUDE_factory_problem_l2162_216227

/-- Represents a factory with workers and production methods -/
structure Factory where
  total_workers : ℕ
  production_increase : ℝ → ℝ
  new_method_factor : ℝ

/-- The conditions and proof goals for the factory problem -/
theorem factory_problem (f : Factory) : 
  (f.production_increase (40 / f.total_workers) = 1.2) →
  (f.production_increase 0.6 = 2.5) →
  (f.total_workers = 500 ∧ f.new_method_factor = 3.5) := by
  sorry


end NUMINAMATH_CALUDE_factory_problem_l2162_216227


namespace NUMINAMATH_CALUDE_base8_arithmetic_result_l2162_216237

/-- Convert a base 8 number to base 10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Convert a base 10 number to base 8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Perform base 8 arithmetic: multiply by 2 and subtract --/
def base8Arithmetic (a b : ℕ) : ℕ :=
  base10ToBase8 ((base8ToBase10 a) * 2 - (base8ToBase10 b))

theorem base8_arithmetic_result :
  base8Arithmetic 45 76 = 14 := by sorry

end NUMINAMATH_CALUDE_base8_arithmetic_result_l2162_216237


namespace NUMINAMATH_CALUDE_min_time_to_finish_tasks_l2162_216298

def wash_rice_time : ℕ := 2
def cook_porridge_time : ℕ := 10
def wash_vegetables_time : ℕ := 3
def chop_vegetables_time : ℕ := 5

def total_vegetable_time : ℕ := wash_vegetables_time + chop_vegetables_time

theorem min_time_to_finish_tasks : ℕ := by
  have h1 : wash_rice_time + cook_porridge_time = 12 := by sorry
  have h2 : total_vegetable_time ≤ cook_porridge_time := by sorry
  exact 12

end NUMINAMATH_CALUDE_min_time_to_finish_tasks_l2162_216298


namespace NUMINAMATH_CALUDE_polygon_sides_l2162_216218

theorem polygon_sides (n : ℕ) (h : n > 2) : 
  (2 : ℚ) / 9 * ((n - 2) * 180) = 360 → n = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2162_216218


namespace NUMINAMATH_CALUDE_street_tree_count_l2162_216257

theorem street_tree_count (road_length : ℕ) (interval : ℕ) (h1 : road_length = 2575) (h2 : interval = 25) : 
  2 * (road_length / interval + 1) = 208 := by
  sorry

end NUMINAMATH_CALUDE_street_tree_count_l2162_216257


namespace NUMINAMATH_CALUDE_jaymee_is_22_l2162_216233

def shara_age : ℕ := 10

def jaymee_age : ℕ := 2 * shara_age + 2

theorem jaymee_is_22 : jaymee_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_jaymee_is_22_l2162_216233


namespace NUMINAMATH_CALUDE_two_numbers_squares_sum_cube_cubes_sum_square_l2162_216210

theorem two_numbers_squares_sum_cube_cubes_sum_square :
  ∃ (a b : ℕ), a ≠ b ∧ a > 0 ∧ b > 0 ∧
  (∃ (c : ℕ), a^2 + b^2 = c^3) ∧
  (∃ (d : ℕ), a^3 + b^3 = d^2) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_squares_sum_cube_cubes_sum_square_l2162_216210


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l2162_216287

theorem smallest_x_absolute_value_equation :
  ∃ (x : ℝ), x = -5.5 ∧ |4*x + 7| = 15 ∧ ∀ (y : ℝ), |4*y + 7| = 15 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l2162_216287


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l2162_216248

theorem least_k_for_inequality (n : ℕ) : 
  (0.0010101 : ℝ) * (10 : ℝ) ^ ((1586 : ℝ) / 500) > (n^2 - 3*n + 5 : ℝ) / (n^3 + 1) ∧ 
  ∀ k : ℚ, k < 1586/500 → (0.0010101 : ℝ) * (10 : ℝ) ^ (k : ℝ) ≤ (n^2 - 3*n + 5 : ℝ) / (n^3 + 1) :=
sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l2162_216248


namespace NUMINAMATH_CALUDE_sum_possible_constants_eq_1232_l2162_216236

/-- 
Given a quadratic equation ax² + bx + c = 0 with two distinct negative integer roots,
where b = 24, this function computes the sum of all possible values for c.
-/
def sum_possible_constants : ℤ := by
  sorry

/-- The main theorem stating that the sum of all possible constant terms is 1232 -/
theorem sum_possible_constants_eq_1232 : sum_possible_constants = 1232 := by
  sorry

end NUMINAMATH_CALUDE_sum_possible_constants_eq_1232_l2162_216236


namespace NUMINAMATH_CALUDE_true_discount_for_given_values_l2162_216286

/-- Given a banker's discount and a sum due, calculate the true discount -/
def true_discount (bankers_discount : ℚ) (sum_due : ℚ) : ℚ :=
  bankers_discount / (1 + bankers_discount / sum_due)

/-- Theorem stating that for the given values, the true discount is 120 -/
theorem true_discount_for_given_values :
  true_discount 144 720 = 120 := by
  sorry

end NUMINAMATH_CALUDE_true_discount_for_given_values_l2162_216286


namespace NUMINAMATH_CALUDE_inverse_proportion_values_l2162_216280

/-- α is inversely proportional to β with α = 5 when β = -4 -/
def inverse_proportion (α β : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ α * β = k ∧ 5 * (-4) = k

theorem inverse_proportion_values (α β : ℝ) (h : inverse_proportion α β) :
  (β = -10 → α = 2) ∧ (β = 2 → α = -10) := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_values_l2162_216280


namespace NUMINAMATH_CALUDE_star_property_l2162_216235

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element
  | five : Element

-- Define the * operation
def star : Element → Element → Element
  | Element.one, x => x
  | Element.two, Element.one => Element.two
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.five
  | Element.two, Element.five => Element.one
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.five
  | Element.three, Element.four => Element.one
  | Element.three, Element.five => Element.two
  | Element.four, Element.one => Element.four
  | Element.four, Element.two => Element.five
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.two
  | Element.four, Element.five => Element.three
  | Element.five, Element.one => Element.five
  | Element.five, Element.two => Element.one
  | Element.five, Element.three => Element.two
  | Element.five, Element.four => Element.three
  | Element.five, Element.five => Element.four

theorem star_property : 
  star (star Element.three Element.five) (star Element.two Element.four) = Element.one := by
  sorry

end NUMINAMATH_CALUDE_star_property_l2162_216235


namespace NUMINAMATH_CALUDE_compound_interest_proof_l2162_216229

/-- Calculate compound interest and prove the total interest earned --/
theorem compound_interest_proof (P : ℝ) (r : ℝ) (n : ℕ) (h1 : P = 1000) (h2 : r = 0.1) (h3 : n = 3) :
  (P * (1 + r)^n - P) = 331 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_proof_l2162_216229


namespace NUMINAMATH_CALUDE_earnings_difference_l2162_216277

/-- Given Paul's and Vinnie's earnings, prove the difference between them. -/
theorem earnings_difference (paul_earnings vinnie_earnings : ℕ) 
  (h1 : paul_earnings = 14)
  (h2 : vinnie_earnings = 30) : 
  vinnie_earnings - paul_earnings = 16 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_l2162_216277


namespace NUMINAMATH_CALUDE_evans_county_population_l2162_216271

theorem evans_county_population (num_cities : ℕ) (lower_bound upper_bound : ℝ) :
  num_cities = 25 →
  lower_bound = 3200 →
  upper_bound = 3600 →
  (num_cities : ℝ) * ((lower_bound + upper_bound) / 2) = 85000 := by
  sorry

end NUMINAMATH_CALUDE_evans_county_population_l2162_216271


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l2162_216242

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 16)
  (h2 : c + a = 18)
  (h3 : a + b = 20) :
  Real.sqrt (a * b * c * (a + b + c)) = 231 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l2162_216242


namespace NUMINAMATH_CALUDE_modular_congruence_l2162_216244

theorem modular_congruence (n : ℕ) : 37^29 ≡ 7 [ZMOD 65] :=
by sorry

end NUMINAMATH_CALUDE_modular_congruence_l2162_216244


namespace NUMINAMATH_CALUDE_subset_implies_a_greater_than_half_l2162_216285

-- Define the sets M and N
def M : Set ℝ := {x | -2 * x + 1 ≥ 0}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem subset_implies_a_greater_than_half (a : ℝ) :
  M ⊆ N a → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_greater_than_half_l2162_216285


namespace NUMINAMATH_CALUDE_b_value_l2162_216283

theorem b_value (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 15 * b) : b = 147 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l2162_216283


namespace NUMINAMATH_CALUDE_correct_selection_count_l2162_216205

/-- The number of ways to select course representatives from a class -/
def select_representatives (num_boys num_girls num_subjects : ℕ) : ℕ × ℕ × ℕ :=
  let scenario1 := sorry
  let scenario2 := sorry
  let scenario3 := sorry
  (scenario1, scenario2, scenario3)

/-- Theorem stating the correct number of ways to select representatives under different conditions -/
theorem correct_selection_count :
  select_representatives 6 4 5 = (22320, 12096, 1008) := by
  sorry

end NUMINAMATH_CALUDE_correct_selection_count_l2162_216205


namespace NUMINAMATH_CALUDE_h_inverse_correct_l2162_216219

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := 4 * x + 5
def g (x : ℝ) : ℝ := 3 * x^2 - 2
def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := Real.sqrt ((x + 3) / 12)

-- Theorem statement
theorem h_inverse_correct (x : ℝ) : h (h_inv x) = x ∧ h_inv (h x) = x :=
  sorry

end NUMINAMATH_CALUDE_h_inverse_correct_l2162_216219


namespace NUMINAMATH_CALUDE_smallest_multiple_l2162_216212

theorem smallest_multiple (y : ℕ) : y = 32 ↔ 
  (y > 0 ∧ 
   900 * y % 1152 = 0 ∧ 
   ∀ z : ℕ, z > 0 → z < y → 900 * z % 1152 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2162_216212


namespace NUMINAMATH_CALUDE_continuous_at_3_l2162_216231

def f (x : ℝ) : ℝ := -3 * x^2 - 9

theorem continuous_at_3 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuous_at_3_l2162_216231


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2162_216253

theorem complex_arithmetic_equality : (5 - Complex.I) - (3 - Complex.I) - 5 * Complex.I = 2 - 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2162_216253


namespace NUMINAMATH_CALUDE_forty_three_base7_equals_thirty_four_base9_l2162_216259

/-- Converts a number from base-7 to base-10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base-10 to a given base -/
def base10ToBase (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

theorem forty_three_base7_equals_thirty_four_base9 :
  let n := 43
  let base7Value := base7ToBase10 n
  let reversedDigits := reverseDigits n
  base10ToBase base7Value 9 = reversedDigits := by sorry

end NUMINAMATH_CALUDE_forty_three_base7_equals_thirty_four_base9_l2162_216259


namespace NUMINAMATH_CALUDE_technician_round_trip_l2162_216203

theorem technician_round_trip (D : ℝ) (h : D > 0) :
  let total_distance := 2 * D
  let distance_traveled := 0.55 * total_distance
  let return_distance := distance_traveled - D
  return_distance / D = 0.1 := by
sorry

end NUMINAMATH_CALUDE_technician_round_trip_l2162_216203


namespace NUMINAMATH_CALUDE_flu_probability_l2162_216250

/-- The probability of a randomly selected person having the flu given the flu rates and population ratios for three areas -/
theorem flu_probability (flu_rate_A flu_rate_B flu_rate_C : ℝ) 
  (pop_ratio_A pop_ratio_B pop_ratio_C : ℕ) : 
  flu_rate_A = 0.06 →
  flu_rate_B = 0.05 →
  flu_rate_C = 0.04 →
  pop_ratio_A = 6 →
  pop_ratio_B = 5 →
  pop_ratio_C = 4 →
  (flu_rate_A * pop_ratio_A + flu_rate_B * pop_ratio_B + flu_rate_C * pop_ratio_C) / 
  (pop_ratio_A + pop_ratio_B + pop_ratio_C) = 77 / 1500 := by
sorry


end NUMINAMATH_CALUDE_flu_probability_l2162_216250


namespace NUMINAMATH_CALUDE_g_of_neg_three_eq_one_l2162_216202

/-- Given a function g(x) = (3x + 4) / (x - 2), prove that g(-3) = 1 -/
theorem g_of_neg_three_eq_one (g : ℝ → ℝ) (h : ∀ x, x ≠ 2 → g x = (3 * x + 4) / (x - 2)) : 
  g (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_of_neg_three_eq_one_l2162_216202


namespace NUMINAMATH_CALUDE_x1_x2_range_l2162_216225

noncomputable section

def f (x : ℝ) : ℝ := if x ≥ 1 then Real.log x else 1 - x / 2

def F (m : ℝ) (x : ℝ) : ℝ := f (f x + 1) + m

theorem x1_x2_range (m : ℝ) (x₁ x₂ : ℝ) (h₁ : F m x₁ = 0) (h₂ : F m x₂ = 0) (h₃ : x₁ ≠ x₂) :
  x₁ * x₂ < Real.sqrt (Real.exp 1) ∧ ∀ y : ℝ, ∃ m : ℝ, ∃ x₁ x₂ : ℝ, 
    F m x₁ = 0 ∧ F m x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ * x₂ < y :=
by sorry

end

end NUMINAMATH_CALUDE_x1_x2_range_l2162_216225


namespace NUMINAMATH_CALUDE_probability_is_two_thirty_thirds_l2162_216239

/-- A square with side length 3 and 12 equally spaced points on its perimeter -/
structure SquareWithPoints where
  side_length : ℝ
  num_points : ℕ
  points_per_side : ℕ

/-- The probability of selecting two points that are one unit apart -/
def probability_one_unit_apart (s : SquareWithPoints) : ℚ :=
  4 / (s.num_points.choose 2)

/-- The main theorem stating the probability is 2/33 -/
theorem probability_is_two_thirty_thirds :
  let s : SquareWithPoints := {
    side_length := 3,
    num_points := 12,
    points_per_side := 3
  }
  probability_one_unit_apart s = 2 / 33 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_two_thirty_thirds_l2162_216239


namespace NUMINAMATH_CALUDE_cubic_function_monotonicity_l2162_216251

/-- A cubic function with a parameter b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + x

/-- The derivative of f with respect to x -/
def f' (b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + 1

/-- The number of monotonic intervals of f -/
def monotonic_intervals (b : ℝ) : ℕ := sorry

theorem cubic_function_monotonicity (b : ℝ) :
  monotonic_intervals b = 3 → b ∈ Set.Iic (-Real.sqrt 3) ∪ Set.Ioi (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_monotonicity_l2162_216251


namespace NUMINAMATH_CALUDE_decimal_85_equals_base7_151_l2162_216292

/-- Converts a number from decimal to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to decimal --/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 7 * acc) 0

theorem decimal_85_equals_base7_151 : fromBase7 [1, 5, 1] = 85 := by
  sorry

#eval toBase7 85  -- Should output [1, 5, 1]
#eval fromBase7 [1, 5, 1]  -- Should output 85

end NUMINAMATH_CALUDE_decimal_85_equals_base7_151_l2162_216292


namespace NUMINAMATH_CALUDE_bisecting_plane_intersects_24_cubes_l2162_216238

/-- Represents a cube composed of unit cubes -/
structure LargeCube where
  side_length : ℕ
  total_cubes : ℕ

/-- Represents a plane that bisects an internal diagonal of a cube -/
structure BisectingPlane where
  cube : LargeCube
  
/-- The number of unit cubes intersected by the bisecting plane -/
def intersected_cubes (plane : BisectingPlane) : ℕ := sorry

/-- Main theorem: A plane bisecting an internal diagonal of a 4x4x4 cube intersects 24 unit cubes -/
theorem bisecting_plane_intersects_24_cubes 
  (cube : LargeCube) 
  (plane : BisectingPlane) 
  (h1 : cube.side_length = 4) 
  (h2 : cube.total_cubes = 64) 
  (h3 : plane.cube = cube) :
  intersected_cubes plane = 24 := by sorry

end NUMINAMATH_CALUDE_bisecting_plane_intersects_24_cubes_l2162_216238


namespace NUMINAMATH_CALUDE_parabola_c_value_l2162_216234

/-- A parabola in the xy-plane with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord (-3) = 2 →   -- Vertex condition
  p.x_coord (-5) = 0 →   -- Point condition
  p.c = -5/2 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2162_216234


namespace NUMINAMATH_CALUDE_election_percentage_l2162_216249

theorem election_percentage (total_members votes_cast : ℕ) 
  (percentage_of_total : ℚ) (h1 : total_members = 1600) 
  (h2 : votes_cast = 525) (h3 : percentage_of_total = 19.6875 / 100) : 
  (percentage_of_total * total_members) / votes_cast = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_election_percentage_l2162_216249


namespace NUMINAMATH_CALUDE_factorization_of_16x_squared_minus_1_l2162_216246

theorem factorization_of_16x_squared_minus_1 (x : ℝ) : 16 * x^2 - 1 = (4*x + 1) * (4*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_16x_squared_minus_1_l2162_216246


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2162_216273

theorem equilateral_triangle_side_length 
  (total_wire_length : ℝ) 
  (h1 : total_wire_length = 63) : 
  total_wire_length / 3 = 21 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2162_216273


namespace NUMINAMATH_CALUDE_entire_line_purple_exactly_integers_purple_not_exactly_rationals_purple_l2162_216221

-- Define the coloring function
def Coloring := ℝ → Bool

-- Define the property of being purple
def isPurple (c : Coloring) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ y z : ℝ, |x - y| < ε ∧ |x - z| < ε ∧ c y ≠ c z

-- Theorem for part a
theorem entire_line_purple :
  ∃ c : Coloring, ∀ x : ℝ, isPurple c x :=
sorry

-- Theorem for part b
theorem exactly_integers_purple :
  ∃ c : Coloring, ∀ x : ℝ, isPurple c x ↔ ∃ n : ℤ, x = n :=
sorry

-- Theorem for part c
theorem not_exactly_rationals_purple :
  ¬ ∃ c : Coloring, ∀ x : ℝ, isPurple c x ↔ ∃ q : ℚ, x = q :=
sorry

end NUMINAMATH_CALUDE_entire_line_purple_exactly_integers_purple_not_exactly_rationals_purple_l2162_216221


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l2162_216276

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * cos(C) = b + (2/3) * c, then ABC is an obtuse triangle -/
theorem triangle_is_obtuse (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a * Real.cos C = b + (2/3) * c →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  π / 2 < A := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_obtuse_l2162_216276


namespace NUMINAMATH_CALUDE_piggy_bank_coins_l2162_216208

/-- The number of coins remaining after each day of withdrawal --/
def coins_remaining (initial : ℕ) : Fin 9 → ℕ
| 0 => initial  -- Initial number of coins
| 1 => initial * 8 / 9  -- After day 1
| 2 => initial * 8 * 7 / (9 * 8)  -- After day 2
| 3 => initial * 8 * 7 * 6 / (9 * 8 * 7)  -- After day 3
| 4 => initial * 8 * 7 * 6 * 5 / (9 * 8 * 7 * 6)  -- After day 4
| 5 => initial * 8 * 7 * 6 * 5 * 4 / (9 * 8 * 7 * 6 * 5)  -- After day 5
| 6 => initial * 8 * 7 * 6 * 5 * 4 * 3 / (9 * 8 * 7 * 6 * 5 * 4)  -- After day 6
| 7 => initial * 8 * 7 * 6 * 5 * 4 * 3 * 2 / (9 * 8 * 7 * 6 * 5 * 4 * 3)  -- After day 7
| 8 => initial * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 / (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2)  -- After day 8

theorem piggy_bank_coins (initial : ℕ) : 
  (coins_remaining initial 8 = 5) → (initial = 45) := by
  sorry

#eval coins_remaining 45 8  -- Should output 5

end NUMINAMATH_CALUDE_piggy_bank_coins_l2162_216208


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l2162_216281

theorem function_satisfies_equation (x y : ℚ) (hx : 0 < x) (hy : 0 < y) :
  let f : ℚ → ℚ := λ t => 1 / (t^2)
  f x + f y + 2 * x * y * f (x * y) = f (x * y) / f (x + y) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l2162_216281


namespace NUMINAMATH_CALUDE_matrix_power_calculation_l2162_216224

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 2, 1]

theorem matrix_power_calculation :
  (2 • A)^10 = !![1024, 0; 20480, 1024] := by sorry

end NUMINAMATH_CALUDE_matrix_power_calculation_l2162_216224


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2162_216288

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 4 * (a - 2) * x - 4 < 0) ↔ (1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2162_216288
