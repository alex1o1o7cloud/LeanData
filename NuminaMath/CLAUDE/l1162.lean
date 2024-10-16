import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_sum_squared_l1162_116298

theorem sqrt_sum_squared (x y z : ℝ) : 
  (Real.sqrt 80 + 3 * Real.sqrt 5 + Real.sqrt 450 / 3)^2 = 295 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squared_l1162_116298


namespace NUMINAMATH_CALUDE_trailing_zeros_625_factorial_l1162_116224

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The factorial of n -/
def factorial (n : ℕ) : ℕ := sorry

theorem trailing_zeros_625_factorial :
  trailingZeros (factorial 625) = 156 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_625_factorial_l1162_116224


namespace NUMINAMATH_CALUDE_min_sum_five_numbers_exist_min_sum_five_numbers_l1162_116201

/-- The sum of five nonnegative real numbers whose pairwise absolute differences sum to 1 is at least 1/4. -/
theorem min_sum_five_numbers (a b c d e : ℝ) (nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e) 
  (sum_diff : |a - b| + |a - c| + |a - d| + |a - e| + |b - c| + |b - d| + |b - e| + |c - d| + |c - e| + |d - e| = 1) :
  a + b + c + d + e ≥ 1/4 := by
  sorry

/-- There exist five nonnegative real numbers whose pairwise absolute differences sum to 1 and whose sum is exactly 1/4. -/
theorem exist_min_sum_five_numbers : 
  ∃ a b c d e : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧
  |a - b| + |a - c| + |a - d| + |a - e| + |b - c| + |b - d| + |b - e| + |c - d| + |c - e| + |d - e| = 1 ∧
  a + b + c + d + e = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_five_numbers_exist_min_sum_five_numbers_l1162_116201


namespace NUMINAMATH_CALUDE_floor_area_approx_l1162_116291

/-- The length of the floor in feet -/
def floor_length_ft : ℝ := 15

/-- The width of the floor in feet -/
def floor_width_ft : ℝ := 10

/-- The conversion factor from feet to meters -/
def ft_to_m : ℝ := 0.3048

/-- The area of the floor in square meters -/
def floor_area_m2 : ℝ := floor_length_ft * ft_to_m * floor_width_ft * ft_to_m

theorem floor_area_approx :
  ∃ ε > 0, abs (floor_area_m2 - 13.93) < ε :=
sorry

end NUMINAMATH_CALUDE_floor_area_approx_l1162_116291


namespace NUMINAMATH_CALUDE_cafe_menu_problem_l1162_116249

theorem cafe_menu_problem (total_dishes : ℕ) 
  (vegan_ratio : ℚ) (gluten_ratio : ℚ) (nut_ratio : ℚ) :
  total_dishes = 30 →
  vegan_ratio = 1 / 3 →
  gluten_ratio = 2 / 5 →
  nut_ratio = 1 / 4 →
  (total_dishes : ℚ) * vegan_ratio * (1 - gluten_ratio - nut_ratio) / total_dishes = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cafe_menu_problem_l1162_116249


namespace NUMINAMATH_CALUDE_zigzag_angle_theorem_l1162_116230

/-- In a zigzag inside a rectangle, if certain angles are given, prove that angle CDE (θ) equals 11 degrees. -/
theorem zigzag_angle_theorem (ACB FEG DCE DEC : ℝ) (h1 : ACB = 80) (h2 : FEG = 64) (h3 : DCE = 86) (h4 : DEC = 83) :
  180 - DCE - DEC = 11 :=
sorry

end NUMINAMATH_CALUDE_zigzag_angle_theorem_l1162_116230


namespace NUMINAMATH_CALUDE_shuttlecock_weight_probability_l1162_116213

/-- The probability that a shuttlecock's weight is less than 4.8 g -/
def prob_less_4_8 : ℝ := 0.3

/-- The probability that a shuttlecock's weight is not greater than 4.85 g -/
def prob_not_greater_4_85 : ℝ := 0.32

/-- The probability that a shuttlecock's weight is within the range [4.8, 4.85] g -/
def prob_range_4_8_to_4_85 : ℝ := prob_not_greater_4_85 - prob_less_4_8

theorem shuttlecock_weight_probability :
  prob_range_4_8_to_4_85 = 0.02 := by sorry

end NUMINAMATH_CALUDE_shuttlecock_weight_probability_l1162_116213


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1162_116254

theorem roots_of_polynomial (x : ℝ) :
  (x^2 - 5*x + 6)*(x - 3)*(2*x - 8) = 0 ↔ x = 2 ∨ x = 3 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1162_116254


namespace NUMINAMATH_CALUDE_correlated_relationships_l1162_116203

-- Define the set of all relationships
inductive Relationship
| A  -- A person's height and weight
| B  -- The distance traveled by a vehicle moving at a constant speed and the time of travel
| C  -- A person's height and eyesight
| D  -- The volume of a cube and its edge length

-- Define a function to check if a relationship is correlated
def is_correlated (r : Relationship) : Prop :=
  match r with
  | Relationship.A => true  -- Height and weight are correlated
  | Relationship.B => true  -- Distance and time at constant speed are correlated (functional)
  | Relationship.C => false -- Height and eyesight are not correlated
  | Relationship.D => true  -- Volume and edge length of a cube are correlated (functional)

-- Theorem stating that the set of correlated relationships is {A, B, D}
theorem correlated_relationships :
  {r : Relationship | is_correlated r} = {Relationship.A, Relationship.B, Relationship.D} :=
by sorry

end NUMINAMATH_CALUDE_correlated_relationships_l1162_116203


namespace NUMINAMATH_CALUDE_tailoring_cost_james_suits_tailoring_cost_l1162_116250

theorem tailoring_cost (cost_first_suit : ℕ) (total_cost : ℕ) : ℕ :=
  let cost_second_suit := 3 * cost_first_suit
  let tailoring_cost := total_cost - cost_first_suit - cost_second_suit
  tailoring_cost

theorem james_suits_tailoring_cost : tailoring_cost 300 1400 = 200 := by
  sorry

end NUMINAMATH_CALUDE_tailoring_cost_james_suits_tailoring_cost_l1162_116250


namespace NUMINAMATH_CALUDE_expression_evaluation_l1162_116219

theorem expression_evaluation : (3^2 - 1) - (4^2 - 2) + (5^2 - 3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1162_116219


namespace NUMINAMATH_CALUDE_probability_in_word_l1162_116212

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The word we're analyzing -/
def word : String := "MATHEMATICS"

/-- The set of unique letters in the word -/
def unique_letters : Finset Char := word.toList.toFinset

/-- The probability of selecting a letter from the alphabet that appears in the word -/
theorem probability_in_word : 
  (unique_letters.card : ℚ) / alphabet_size = 4 / 13 := by sorry

end NUMINAMATH_CALUDE_probability_in_word_l1162_116212


namespace NUMINAMATH_CALUDE_circle_ratio_proof_l1162_116271

theorem circle_ratio_proof (b a c : ℝ) (h1 : b > 0) (h2 : a > 0) (h3 : c > 0)
  (h4 : b^2 - c^2 = 2 * a^2) (h5 : c = 1.5 * a) :
  a / b = 2 / Real.sqrt 17 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_proof_l1162_116271


namespace NUMINAMATH_CALUDE_one_cow_drinking_time_l1162_116264

/-- Represents the drinking rate of cows and the spring inflow rate -/
structure PondSystem where
  /-- Amount of water one cow drinks per day -/
  cow_drink_rate : ℝ
  /-- Amount of water springs add to the pond per day -/
  spring_rate : ℝ
  /-- Total volume of the pond -/
  pond_volume : ℝ

/-- Given the conditions, proves that one cow will take 75 days to drink the pond -/
theorem one_cow_drinking_time (sys : PondSystem)
  (h1 : sys.pond_volume + 3 * sys.spring_rate = 3 * 17 * sys.cow_drink_rate)
  (h2 : sys.pond_volume + 30 * sys.spring_rate = 30 * 2 * sys.cow_drink_rate) :
  sys.pond_volume + 75 * sys.spring_rate = 75 * sys.cow_drink_rate :=
by sorry


end NUMINAMATH_CALUDE_one_cow_drinking_time_l1162_116264


namespace NUMINAMATH_CALUDE_sugar_amount_l1162_116200

/-- The number of cups of flour Mary still needs to add -/
def flour_needed : ℕ := 21

/-- The difference between the total cups of flour and sugar in the recipe -/
def flour_sugar_difference : ℕ := 8

/-- The number of cups of sugar the recipe calls for -/
def sugar_in_recipe : ℕ := flour_needed - flour_sugar_difference

theorem sugar_amount : sugar_in_recipe = 13 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_l1162_116200


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_l1162_116221

theorem imaginary_part_of_complex (z : ℂ) (h : z = 3 - 4 * I) : z.im = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_l1162_116221


namespace NUMINAMATH_CALUDE_sequence_elements_l1162_116292

theorem sequence_elements : ∃ (n₁ n₂ : ℕ+), 
  (n₁.val^2 + n₁.val = 12) ∧ 
  (n₂.val^2 + n₂.val = 30) ∧ 
  (∀ n : ℕ+, n.val^2 + n.val ≠ 18) ∧ 
  (∀ n : ℕ+, n.val^2 + n.val ≠ 25) := by
  sorry

end NUMINAMATH_CALUDE_sequence_elements_l1162_116292


namespace NUMINAMATH_CALUDE_blue_chip_value_l1162_116293

theorem blue_chip_value (yellow_value : ℕ) (green_value : ℕ) (yellow_count : ℕ) (blue_count : ℕ) (total_product : ℕ) :
  yellow_value = 2 →
  green_value = 5 →
  yellow_count = 4 →
  blue_count = blue_count →  -- This represents that blue and green chip counts are equal
  total_product = 16000 →
  total_product = yellow_value ^ yellow_count * blue_value ^ blue_count * green_value ^ blue_count →
  blue_value = 8 := by
  sorry

#check blue_chip_value

end NUMINAMATH_CALUDE_blue_chip_value_l1162_116293


namespace NUMINAMATH_CALUDE_orchid_seed_weight_scientific_notation_l1162_116225

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

theorem orchid_seed_weight_scientific_notation :
  toScientificNotation 0.0000005 = ScientificNotation.mk 5 (-7) (by norm_num) := by sorry

end NUMINAMATH_CALUDE_orchid_seed_weight_scientific_notation_l1162_116225


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1162_116226

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- The probability that a normal random variable is less than or equal to a given value -/
noncomputable def normalCdf (X : NormalRandomVariable) (x : ℝ) : ℝ := sorry

theorem normal_distribution_probability 
  (X : NormalRandomVariable)
  (h1 : X.μ = 3)
  (h2 : normalCdf X 6 = 0.9) :
  normalCdf X 3 - normalCdf X 0 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1162_116226


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l1162_116207

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 8*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -2*x + 8

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 6*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l1162_116207


namespace NUMINAMATH_CALUDE_consecutive_integers_perfect_square_product_specific_consecutive_integers_l1162_116267

theorem consecutive_integers_perfect_square_product :
  ∃ (n : ℤ), (n - 1) * n * (n + 1) * (n + 2) = (n^2 + n - 1)^2 - 1 ∧
  ∃ (k : ℤ), (n^2 + n - 1)^2 = k^2 + 1 ∧
  (n = 0 ∨ n = -1 ∨ n = 1 ∨ n = -2) :=
by sorry

theorem specific_consecutive_integers :
  (-1 : ℤ) * 0 * 1 * 2 = 0^2 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_perfect_square_product_specific_consecutive_integers_l1162_116267


namespace NUMINAMATH_CALUDE_laurie_has_37_marbles_l1162_116227

/-- The number of marbles each person has. -/
structure Marbles where
  dennis : ℕ
  kurt : ℕ
  laurie : ℕ

/-- The conditions of the marble problem. -/
def marble_problem (m : Marbles) : Prop :=
  m.dennis = 70 ∧
  m.kurt = m.dennis - 45 ∧
  m.laurie = m.kurt + 12

/-- Theorem stating that Laurie has 37 marbles under the given conditions. -/
theorem laurie_has_37_marbles (m : Marbles) (h : marble_problem m) : m.laurie = 37 := by
  sorry


end NUMINAMATH_CALUDE_laurie_has_37_marbles_l1162_116227


namespace NUMINAMATH_CALUDE_min_value_product_l1162_116234

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a / b + b / c + c / a + b / a + c / b + a / c = 7) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ 35 / 2 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    a₀ / b₀ + b₀ / c₀ + c₀ / a₀ + b₀ / a₀ + c₀ / b₀ + a₀ / c₀ = 7 ∧
    (a₀ / b₀ + b₀ / c₀ + c₀ / a₀) * (b₀ / a₀ + c₀ / b₀ + a₀ / c₀) = 35 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l1162_116234


namespace NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l1162_116274

theorem angle_measure_in_special_quadrilateral :
  ∀ (E F G H : ℝ),
  E = 3 * F ∧ E = 4 * G ∧ E = 6 * H →
  E + F + G + H = 360 →
  round E = 206 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l1162_116274


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1162_116296

theorem cone_lateral_surface_area 
  (base_radius : ℝ) 
  (height : ℝ) 
  (lateral_surface_area : ℝ) 
  (h1 : base_radius = 3) 
  (h2 : height = 4) : 
  lateral_surface_area = 15 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1162_116296


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1162_116272

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1162_116272


namespace NUMINAMATH_CALUDE_appropriate_word_count_l1162_116286

def speech_duration_min : ℝ := 40
def speech_duration_max : ℝ := 50
def speech_rate : ℝ := 160
def word_count : ℕ := 7600

theorem appropriate_word_count : 
  speech_duration_min * speech_rate ≤ word_count ∧ 
  word_count ≤ speech_duration_max * speech_rate := by
  sorry

end NUMINAMATH_CALUDE_appropriate_word_count_l1162_116286


namespace NUMINAMATH_CALUDE_exponent_sum_equality_l1162_116285

theorem exponent_sum_equality : (-3)^(4^2) + 2^(3^2) = 43047233 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_equality_l1162_116285


namespace NUMINAMATH_CALUDE_dinner_bill_friends_l1162_116236

theorem dinner_bill_friends (total_bill : ℝ) (silas_payment : ℝ) (one_friend_payment : ℝ) : 
  total_bill = 150 →
  silas_payment = total_bill / 2 →
  one_friend_payment = 18 →
  ∃ (num_friends : ℕ),
    num_friends = 6 ∧
    (num_friends - 1) * one_friend_payment = (total_bill - silas_payment) * 1.1 :=
by sorry

end NUMINAMATH_CALUDE_dinner_bill_friends_l1162_116236


namespace NUMINAMATH_CALUDE_equation_solution_l1162_116295

theorem equation_solution : ∃ x : ℚ, (1 / 5 + 5 / x = 12 / x + 1 / 12) ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1162_116295


namespace NUMINAMATH_CALUDE_parallelepiped_inequality_l1162_116204

theorem parallelepiped_inequality (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_diagonal : d^2 = a^2 + b^2 + c^2) : 
  a^2 + b^2 + c^2 ≥ d^2 / 3 := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_inequality_l1162_116204


namespace NUMINAMATH_CALUDE_a_6_equals_25_l1162_116237

/-- An increasing geometric sequence -/
def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The roots of x^2 - 6x + 5 = 0 -/
def is_root_of_equation (x : ℝ) : Prop :=
  x^2 - 6*x + 5 = 0

/-- Theorem: For an increasing geometric sequence where a_2 and a_4 are roots of x^2 - 6x + 5 = 0, a_6 = 25 -/
theorem a_6_equals_25 (a : ℕ → ℝ) 
  (h1 : is_increasing_geometric_sequence a)
  (h2 : is_root_of_equation (a 2))
  (h3 : is_root_of_equation (a 4)) :
  a 6 = 25 :=
sorry

end NUMINAMATH_CALUDE_a_6_equals_25_l1162_116237


namespace NUMINAMATH_CALUDE_sum_first_three_terms_l1162_116245

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem sum_first_three_terms (a : ℤ) (d : ℤ) :
  arithmetic_sequence a d 4 = 8 ∧
  arithmetic_sequence a d 5 = 12 ∧
  arithmetic_sequence a d 6 = 16 →
  arithmetic_sequence a d 1 + arithmetic_sequence a d 2 + arithmetic_sequence a d 3 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_first_three_terms_l1162_116245


namespace NUMINAMATH_CALUDE_larger_root_of_quadratic_l1162_116289

theorem larger_root_of_quadratic (x : ℝ) : 
  (x - 5/8) * (x - 5/8) + (x - 5/8) * (x - 1/3) = 0 →
  x = 5/8 ∨ x = 23/48 →
  (5/8 : ℝ) > (23/48 : ℝ) →
  x = 5/8 := by sorry

end NUMINAMATH_CALUDE_larger_root_of_quadratic_l1162_116289


namespace NUMINAMATH_CALUDE_ternary_35_implies_k_2_l1162_116297

def ternary_to_decimal (k : ℕ+) : ℕ := 1 * 3^3 + k * 3^2 + 2

theorem ternary_35_implies_k_2 : 
  ∀ k : ℕ+, ternary_to_decimal k = 35 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ternary_35_implies_k_2_l1162_116297


namespace NUMINAMATH_CALUDE_function_properties_l1162_116299

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + 7 * Real.pi / 6) + a

-- State the theorem
theorem function_properties :
  ∃ (a : ℝ),
    (∀ x, f x a ≤ 2) ∧  -- Maximum value is 2
    (∃ x, f x a = 2) ∧  -- Maximum value is attained
    (a = 1) ∧  -- Value of a
    (∀ x, f x a = f (x + Real.pi) a) ∧  -- Smallest positive period is π
    (∀ k : ℤ, ∀ x ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi),
      ∀ y ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi),
      x ≤ y → f y a ≤ f x a)  -- Monotonically decreasing intervals
    :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1162_116299


namespace NUMINAMATH_CALUDE_total_spider_legs_is_33_l1162_116266

/-- The total number of spider legs in a room with 5 spiders -/
def total_spider_legs : ℕ :=
  let spider1 := 6
  let spider2 := 7
  let spider3 := 8
  let spider4 := 5
  let spider5 := 7
  spider1 + spider2 + spider3 + spider4 + spider5

/-- Theorem stating that the total number of spider legs is 33 -/
theorem total_spider_legs_is_33 : total_spider_legs = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_is_33_l1162_116266


namespace NUMINAMATH_CALUDE_square_equation_solutions_l1162_116258

/-- p-arithmetic field -/
structure PArithmetic (p : ℕ) where
  carrier : Type
  zero : carrier
  one : carrier
  add : carrier → carrier → carrier
  mul : carrier → carrier → carrier
  neg : carrier → carrier
  inv : carrier → carrier
  -- Add necessary field axioms here

/-- Definition of squaring in p-arithmetic -/
def square {p : ℕ} (F : PArithmetic p) (x : F.carrier) : F.carrier :=
  F.mul x x

/-- Main theorem: In p-arithmetic (p ≠ 2), x² = a has two distinct solutions for non-zero a -/
theorem square_equation_solutions {p : ℕ} (hp : p ≠ 2) (F : PArithmetic p) :
  ∀ a : F.carrier, a ≠ F.zero →
    ∃ x y : F.carrier, x ≠ y ∧ square F x = a ∧ square F y = a ∧
      ∀ z : F.carrier, square F z = a → (z = x ∨ z = y) :=
sorry

end NUMINAMATH_CALUDE_square_equation_solutions_l1162_116258


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1162_116210

theorem complex_expression_simplification :
  (7 - 3 * Complex.I) - (2 - 5 * Complex.I) - (3 + 2 * Complex.I) = (2 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1162_116210


namespace NUMINAMATH_CALUDE_additional_correct_answers_needed_l1162_116257

def total_problems : ℕ := 80
def arithmetic_problems : ℕ := 15
def algebra_problems : ℕ := 25
def geometry_problems : ℕ := 40
def arithmetic_correct_ratio : ℚ := 4/5
def algebra_correct_ratio : ℚ := 1/2
def geometry_correct_ratio : ℚ := 11/20
def passing_grade_ratio : ℚ := 13/20

def correct_answers : ℕ := 
  (arithmetic_problems * arithmetic_correct_ratio).ceil.toNat +
  (algebra_problems * algebra_correct_ratio).ceil.toNat +
  (geometry_problems * geometry_correct_ratio).ceil.toNat

def passing_threshold : ℕ := (total_problems * passing_grade_ratio).ceil.toNat

theorem additional_correct_answers_needed : 
  passing_threshold - correct_answers = 5 := by sorry

end NUMINAMATH_CALUDE_additional_correct_answers_needed_l1162_116257


namespace NUMINAMATH_CALUDE_unique_similar_triangles_l1162_116214

theorem unique_similar_triangles :
  ∀ (a b c a' b' c' : ℕ),
    a = 8 →
    a < a' →
    a < b →
    b < c →
    (b = b' ∧ c = c') ∨ (a = b' ∧ b = c') →
    (a' * b = a * b') ∧ (a' * c = a * c') →
    (a = 8 ∧ b = 12 ∧ c = 18 ∧ a' = 12 ∧ b' = 12 ∧ c' = 18) ∨
    (a = 8 ∧ b = 12 ∧ c = 18 ∧ a' = 12 ∧ b' = 18 ∧ c' = 27) :=
by sorry

end NUMINAMATH_CALUDE_unique_similar_triangles_l1162_116214


namespace NUMINAMATH_CALUDE_x_axis_intercept_l1162_116206

/-- The x-axis intercept of the line x + 2y + 1 = 0 is -1. -/
theorem x_axis_intercept :
  ∃ (x : ℝ), x + 2 * 0 + 1 = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_axis_intercept_l1162_116206


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1162_116253

theorem trigonometric_identity (α β : Real) :
  (Real.cos α)^2 + (Real.cos β)^2 - 2 * (Real.cos α) * (Real.cos β) * Real.cos (α + β) =
  (Real.sin α)^2 + (Real.sin β)^2 + 2 * (Real.sin α) * (Real.sin β) * Real.sin (α + β) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1162_116253


namespace NUMINAMATH_CALUDE_length_XX₁_l1162_116241

/-- Configuration of two right triangles with angle bisectors -/
structure TriangleConfig where
  -- Triangle DEF
  DE : ℝ
  DF : ℝ
  hDE : DE = 13
  hDF : DF = 5
  hDEF_right : DE^2 = DF^2 + EF^2
  
  -- D₁ is on EF such that ∠FDD₁ = ∠EDD₁
  D₁F : ℝ
  D₁E : ℝ
  hD₁_on_EF : D₁F + D₁E = EF
  hD₁_bisector : D₁F / D₁E = DF / EF
  
  -- Triangle XYZ
  XY : ℝ
  XZ : ℝ
  hXY : XY = D₁E
  hXZ : XZ = D₁F
  hXYZ_right : XY^2 = XZ^2 + YZ^2
  
  -- X₁ is on YZ such that ∠ZXX₁ = ∠YXX₁
  X₁Z : ℝ
  X₁Y : ℝ
  hX₁_on_YZ : X₁Z + X₁Y = YZ
  hX₁_bisector : X₁Z / X₁Y = XZ / XY

/-- The length of XX₁ in the given configuration is 20/17 -/
theorem length_XX₁ (config : TriangleConfig) : X₁Z = 20/17 := by
  sorry

end NUMINAMATH_CALUDE_length_XX₁_l1162_116241


namespace NUMINAMATH_CALUDE_smallest_multiple_l1162_116251

theorem smallest_multiple (n : ℕ) : n = 2015 ↔ 
  n > 0 ∧ 
  31 ∣ n ∧ 
  n % 97 = 6 ∧ 
  ∀ m : ℕ, m > 0 → 31 ∣ m → m % 97 = 6 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1162_116251


namespace NUMINAMATH_CALUDE_Q_trajectory_equation_l1162_116282

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line on which point P moves -/
def line_P (p : Point) : Prop :=
  2 * p.x - p.y + 3 = 0

/-- The fixed point M -/
def M : Point :=
  ⟨-1, 2⟩

/-- Q is on the extension of PM and PM = MQ -/
def Q_position (p q : Point) : Prop :=
  q.x - M.x = M.x - p.x ∧ q.y - M.y = M.y - p.y

/-- The trajectory of point Q -/
def Q_trajectory (q : Point) : Prop :=
  2 * q.x - q.y + 5 = 0

/-- Theorem: The trajectory of Q satisfies the equation 2x - y + 5 = 0 -/
theorem Q_trajectory_equation :
  ∀ p q : Point, line_P p → Q_position p q → Q_trajectory q :=
by sorry

end NUMINAMATH_CALUDE_Q_trajectory_equation_l1162_116282


namespace NUMINAMATH_CALUDE_composition_may_have_no_fixed_point_l1162_116229

-- Define a type for our functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to have a fixed point
def has_fixed_point (f : RealFunction) : Prop :=
  ∃ x : ℝ, f x = x

-- State the theorem
theorem composition_may_have_no_fixed_point :
  ∃ (f g : RealFunction),
    has_fixed_point f ∧ 
    has_fixed_point g ∧ 
    ¬(has_fixed_point (f ∘ g)) :=
sorry

end NUMINAMATH_CALUDE_composition_may_have_no_fixed_point_l1162_116229


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1162_116284

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x + 1) * (x - a) < 0}
  if a > -1 then
    S = {x : ℝ | -1 < x ∧ x < a}
  else if a = -1 then
    S = ∅
  else
    S = {x : ℝ | a < x ∧ x < -1} :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1162_116284


namespace NUMINAMATH_CALUDE_min_sum_squares_roots_l1162_116261

theorem min_sum_squares_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 4 * m * x₁ + 2 * m^2 + 3 * m - 2 = 0) →
  (2 * x₂^2 - 4 * m * x₂ + 2 * m^2 + 3 * m - 2 = 0) →
  (∀ m' : ℝ, ∃ x₁' x₂' : ℝ, 2 * x₁'^2 - 4 * m' * x₁' + 2 * m'^2 + 3 * m' - 2 = 0 ∧
                             2 * x₂'^2 - 4 * m' * x₂' + 2 * m'^2 + 3 * m' - 2 = 0) →
  x₁^2 + x₂^2 ≥ 8/9 ∧ (x₁^2 + x₂^2 = 8/9 ↔ m = 2/3) := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_roots_l1162_116261


namespace NUMINAMATH_CALUDE_interest_rate_for_doubling_in_eight_years_l1162_116218

/-- 
Given a sum of money that doubles itself in 8 years at simple interest,
prove that the rate percent per annum is 12.5%.
-/
theorem interest_rate_for_doubling_in_eight_years 
  (P : ℝ) -- Principal amount
  (h_positive : P > 0) -- Assumption that the principal is positive
  (h_double : P + (P * R * 8) / 100 = 2 * P) -- Condition that the sum doubles in 8 years
  : R = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_for_doubling_in_eight_years_l1162_116218


namespace NUMINAMATH_CALUDE_count_elements_with_leftmost_seven_l1162_116248

/-- The set of powers of 5 up to 5000 -/
def S : Set ℕ := {n : ℕ | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 5000 ∧ n = 5^k}

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The leftmost digit of a natural number -/
def leftmost_digit (n : ℕ) : ℕ := sorry

/-- The count of numbers in S with 7 as the leftmost digit -/
def count_leftmost_seven (S : Set ℕ) : ℕ := sorry

theorem count_elements_with_leftmost_seven :
  num_digits (5^5000) = 3501 →
  leftmost_digit (5^5000) = 7 →
  count_leftmost_seven S = 1501 := by sorry

end NUMINAMATH_CALUDE_count_elements_with_leftmost_seven_l1162_116248


namespace NUMINAMATH_CALUDE_glitched_clock_correct_time_fraction_l1162_116252

/-- Represents a 12-hour digital clock with a glitch where '2' is displayed as '7' -/
structure GlitchedClock where
  /-- The number of hours in the clock cycle -/
  hours : Nat
  /-- The number of minutes in an hour -/
  minutes_per_hour : Nat
  /-- The digit that is erroneously displayed -/
  glitched_digit : Nat
  /-- The digit that replaces the glitched digit -/
  replacement_digit : Nat

/-- The fraction of the day that the glitched clock shows the correct time -/
def correct_time_fraction (clock : GlitchedClock) : ℚ :=
  sorry

/-- Theorem stating that the fraction of correct time for the given clock is 55/72 -/
theorem glitched_clock_correct_time_fraction :
  let clock : GlitchedClock := {
    hours := 12,
    minutes_per_hour := 60,
    glitched_digit := 2,
    replacement_digit := 7
  }
  correct_time_fraction clock = 55 / 72 := by
  sorry

end NUMINAMATH_CALUDE_glitched_clock_correct_time_fraction_l1162_116252


namespace NUMINAMATH_CALUDE_probability_between_C_and_D_l1162_116220

/-- Given a line segment AB with points A, B, C, D, and E, prove that the probability
    of a randomly selected point on AB being between C and D is 1/2. -/
theorem probability_between_C_and_D (A B C D E : ℝ) : 
  A < B ∧ 
  B - A = 4 * (E - A) ∧
  B - A = 8 * (B - D) ∧
  D - A = 3 * (E - A) ∧
  B - D = 5 * (B - E) ∧
  C = D + (1/8) * (B - A) →
  (C - D) / (B - A) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_probability_between_C_and_D_l1162_116220


namespace NUMINAMATH_CALUDE_largest_negative_smallest_positive_smallest_abs_l1162_116279

theorem largest_negative_smallest_positive_smallest_abs (a b c : ℤ) : 
  (∀ x : ℤ, x < 0 → x ≤ a) →  -- a is the largest negative integer
  (∀ x : ℤ, x > 0 → b ≤ x) →  -- b is the smallest positive integer
  (∀ x : ℤ, |c| ≤ |x|) →      -- c has the smallest absolute value
  a + c - b = -2 := by
sorry

end NUMINAMATH_CALUDE_largest_negative_smallest_positive_smallest_abs_l1162_116279


namespace NUMINAMATH_CALUDE_gray_area_calculation_l1162_116276

theorem gray_area_calculation (r_small : ℝ) (r_large : ℝ) : 
  r_small = 2 →
  r_large = 3 * r_small →
  (π * r_large^2 - π * r_small^2) = 32 * π := by
  sorry

end NUMINAMATH_CALUDE_gray_area_calculation_l1162_116276


namespace NUMINAMATH_CALUDE_lemonade_sales_ratio_l1162_116232

theorem lemonade_sales_ratio :
  ∀ (katya_sales ricky_sales tina_sales : ℕ),
    katya_sales = 8 →
    ricky_sales = 9 →
    tina_sales = katya_sales + 26 →
    ∃ (m : ℕ), tina_sales = m * (katya_sales + ricky_sales) →
    (tina_sales : ℚ) / (katya_sales + ricky_sales : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_sales_ratio_l1162_116232


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1162_116280

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b = 5) :
  (1/a + 1/b) ≥ 5 + 2*Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1162_116280


namespace NUMINAMATH_CALUDE_marble_arrangement_l1162_116239

/-- The number of blue marbles --/
def blue_marbles : ℕ := 6

/-- The maximum number of yellow marbles --/
def yellow_marbles : ℕ := 17

/-- The total number of marbles --/
def total_marbles : ℕ := blue_marbles + yellow_marbles

/-- The number of ways to arrange the marbles --/
def arrangement_count : ℕ := Nat.choose (total_marbles + blue_marbles - 1) blue_marbles

/-- The theorem to be proved --/
theorem marble_arrangement :
  arrangement_count = 100947 ∧ arrangement_count % 1000 = 947 := by
  sorry


end NUMINAMATH_CALUDE_marble_arrangement_l1162_116239


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_S_l1162_116222

def K : ℚ := (1 : ℚ) + (1/2 : ℚ) + (1/3 : ℚ) + (1/4 : ℚ)

def S (n : ℕ) : ℚ := n * (5^(n-1) : ℚ) * K

def is_integer (q : ℚ) : Prop := ∃ (z : ℤ), q = z

theorem smallest_n_for_integer_S :
  ∀ n : ℕ, (n > 0 ∧ is_integer (S n)) → n ≥ 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_S_l1162_116222


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l1162_116228

theorem missing_fraction_sum (x : ℚ) : 
  (1/3 : ℚ) + (-5/6 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-5/6 : ℚ) + x = 5/6 → x = 13/6 := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l1162_116228


namespace NUMINAMATH_CALUDE_sphere_cylinder_surface_area_difference_l1162_116260

/-- The difference between the surface area of a sphere and the lateral surface area of its inscribed cylinder is zero. -/
theorem sphere_cylinder_surface_area_difference (R : ℝ) (R_pos : R > 0) : 
  4 * Real.pi * R^2 - (2 * Real.pi * R * (2 * R)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_surface_area_difference_l1162_116260


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l1162_116240

theorem greatest_integer_solution (x : ℝ) : 
  x^3 = 7 - 2*x → 
  (∀ n : ℤ, n > (x - 2 : ℝ) → n ≤ 3) ∧ 
  (3 : ℝ) > (x - 2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l1162_116240


namespace NUMINAMATH_CALUDE_solution_set_inequality1_solution_set_inequality2_l1162_116287

-- First inequality
theorem solution_set_inequality1 (x : ℝ) :
  x ≠ 2 →
  ((x + 1) / (x - 2) ≥ 3) ↔ (2 < x ∧ x ≤ 7/2) :=
sorry

-- Second inequality
theorem solution_set_inequality2 (x a : ℝ) :
  x^2 - a*x - 2*a^2 ≤ 0 ↔
    (a = 0 ∧ x = 0) ∨
    (a > 0 ∧ -a ≤ x ∧ x ≤ 2*a) ∨
    (a < 0 ∧ 2*a ≤ x ∧ x ≤ -a) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality1_solution_set_inequality2_l1162_116287


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1162_116290

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ (∀ x : ℝ, x ≥ 0 → x ≥ Real.sin x)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x < Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1162_116290


namespace NUMINAMATH_CALUDE_fred_earnings_l1162_116217

/-- The amount of money Fred made washing cars -/
def money_made (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Theorem: Fred's earnings from washing cars -/
theorem fred_earnings :
  let initial_amount := 23
  let final_amount := 86
  money_made initial_amount final_amount = 63 := by
sorry

end NUMINAMATH_CALUDE_fred_earnings_l1162_116217


namespace NUMINAMATH_CALUDE_diamond_symmetry_lines_l1162_116294

-- Define the binary operation
def diamond (a b : ℝ) : ℝ := a^2 + a*b - b^2

-- Theorem statement
theorem diamond_symmetry_lines :
  ∀ x y : ℝ, diamond x y = diamond y x ↔ y = x ∨ y = -x :=
sorry

end NUMINAMATH_CALUDE_diamond_symmetry_lines_l1162_116294


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1162_116209

/-- Represents a sampling method -/
inductive SamplingMethod
| Lottery
| RandomNumber
| Systematic
| Stratified

/-- Represents a population with two equal-sized subgroups -/
structure Population :=
  (size : ℕ)
  (subgroup1_size : ℕ)
  (subgroup2_size : ℕ)
  (h_equal_size : subgroup1_size = subgroup2_size)
  (h_total_size : subgroup1_size + subgroup2_size = size)

/-- Represents the goal of understanding differences between subgroups -/
def UnderstandDifferences : Prop := True

/-- The most appropriate sampling method for a given population and goal -/
def MostAppropriateSamplingMethod (p : Population) (goal : UnderstandDifferences) : SamplingMethod :=
  SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most appropriate method 
    for a population with two equal-sized subgroups when the goal is to 
    understand differences between these subgroups -/
theorem stratified_sampling_most_appropriate 
  (p : Population) (goal : UnderstandDifferences) :
  MostAppropriateSamplingMethod p goal = SamplingMethod.Stratified :=
by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l1162_116209


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1162_116268

theorem unique_solution_exponential_equation :
  ∃! (n : ℕ+), Real.exp (1 / n.val) + Real.exp (-1 / n.val) = Real.sqrt n.val :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1162_116268


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1162_116244

def M : Set ℝ := {x | |x - 1| < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

theorem necessary_but_not_sufficient : 
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧ (∃ b : ℝ, b ∈ M ∧ b ∉ N) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1162_116244


namespace NUMINAMATH_CALUDE_son_age_proof_l1162_116270

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l1162_116270


namespace NUMINAMATH_CALUDE_perseverance_arrangement_count_l1162_116259

/-- The number of letters in the word "PERSEVERANCE" -/
def total_letters : ℕ := 12

/-- The number of times the letter 'E' appears in "PERSEVERANCE" -/
def e_count : ℕ := 3

/-- The number of times the letter 'R' appears in "PERSEVERANCE" -/
def r_count : ℕ := 2

/-- The number of unique arrangements of the letters in "PERSEVERANCE" -/
def perseverance_arrangements : ℕ := Nat.factorial total_letters / (Nat.factorial e_count * Nat.factorial r_count)

theorem perseverance_arrangement_count : perseverance_arrangements = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_perseverance_arrangement_count_l1162_116259


namespace NUMINAMATH_CALUDE_binomial_probability_two_successes_l1162_116231

/-- A random variable following a binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The probability mass function for a binomial distribution -/
def binomialPMF (X : BinomialDistribution) (k : ℕ) : ℝ :=
  (Nat.choose X.n k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

theorem binomial_probability_two_successes :
  let X : BinomialDistribution := { n := 6, p := 1/3, h_p := by norm_num }
  binomialPMF X 2 = 80/243 := by sorry

end NUMINAMATH_CALUDE_binomial_probability_two_successes_l1162_116231


namespace NUMINAMATH_CALUDE_same_grade_percentage_l1162_116255

theorem same_grade_percentage (total_students : ℕ) (same_grade_students : ℕ) : 
  total_students = 40 →
  same_grade_students = 17 →
  (same_grade_students : ℚ) / (total_students : ℚ) * 100 = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l1162_116255


namespace NUMINAMATH_CALUDE_expected_sedans_is_48_l1162_116233

/-- Represents the car dealership's sales plan -/
structure SalesPlan where
  sportsCarRatio : ℕ
  sedanRatio : ℕ
  totalTarget : ℕ
  plannedSportsCars : ℕ

/-- Calculates the number of sedans to be sold based on the sales plan -/
def expectedSedans (plan : SalesPlan) : ℕ :=
  plan.sedanRatio * plan.plannedSportsCars / plan.sportsCarRatio

/-- Theorem stating that the expected number of sedans is 48 given the specified conditions -/
theorem expected_sedans_is_48 (plan : SalesPlan)
  (h1 : plan.sportsCarRatio = 5)
  (h2 : plan.sedanRatio = 8)
  (h3 : plan.totalTarget = 78)
  (h4 : plan.plannedSportsCars = 30)
  (h5 : plan.plannedSportsCars + expectedSedans plan = plan.totalTarget) :
  expectedSedans plan = 48 := by
  sorry

#eval expectedSedans {sportsCarRatio := 5, sedanRatio := 8, totalTarget := 78, plannedSportsCars := 30}

end NUMINAMATH_CALUDE_expected_sedans_is_48_l1162_116233


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_x_axis_l1162_116265

/-- A line parallel to the x-axis has a constant y-coordinate --/
def parallelToXAxis (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f x₁ = f x₂

/-- The equation of a line passing through (4, 2) and parallel to the x-axis --/
def lineEquation : ℝ → ℝ := λ x => 2

theorem line_through_point_parallel_to_x_axis :
  parallelToXAxis lineEquation ∧ lineEquation 4 = 2 := by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_x_axis_l1162_116265


namespace NUMINAMATH_CALUDE_sequence_difference_sum_l1162_116273

theorem sequence_difference_sum : 
  (Finset.sum (Finset.range 100) (fun i => 3001 + i)) - 
  (Finset.sum (Finset.range 100) (fun i => 201 + i)) = 280000 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_sum_l1162_116273


namespace NUMINAMATH_CALUDE_arithmetic_progression_unique_solution_l1162_116202

theorem arithmetic_progression_unique_solution (n₁ n₂ : ℕ) (hn : n₁ ≠ n₂) :
  ∃! (a₁ d : ℚ),
    (∀ (n : ℕ), n * (2 * a₁ + (n - 1) * d) / 2 = n^2) ∧
    (n₁ * (2 * a₁ + (n₁ - 1) * d) / 2 = n₁^2) ∧
    (n₂ * (2 * a₁ + (n₂ - 1) * d) / 2 = n₂^2) ∧
    a₁ = 1 ∧ d = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_unique_solution_l1162_116202


namespace NUMINAMATH_CALUDE_perimeter_division_ratio_l1162_116238

/-- A square with a point M on its diagonal and a line passing through M -/
structure SquareWithDiagonalPoint where
  /-- Side length of the square -/
  s : ℝ
  /-- Point M divides the diagonal AC in the ratio AM : MC = 3 : 2 -/
  m_divides_diagonal : ℝ
  /-- Ratio of areas divided by the line passing through M -/
  area_ratio : ℝ × ℝ
  /-- Assumption that s > 0 -/
  s_pos : s > 0
  /-- Assumption that m_divides_diagonal is the ratio 3 : 2 -/
  m_divides_diagonal_eq : m_divides_diagonal = 3 / 5
  /-- Assumption that area_ratio is 9 : 11 -/
  area_ratio_eq : area_ratio = (9, 11)

/-- Theorem: The line divides the perimeter in the ratio 19 : 21 -/
theorem perimeter_division_ratio (sq : SquareWithDiagonalPoint) : 
  (19 : ℝ) / 21 = 19 / (19 + 21) := by sorry

end NUMINAMATH_CALUDE_perimeter_division_ratio_l1162_116238


namespace NUMINAMATH_CALUDE_city_population_multiple_l1162_116263

/- Define the populations of the cities and the multiple -/
def willowdale_population : ℕ := 2000
def sun_city_population : ℕ := 12000

/- Define the relationship between the cities' populations -/
def roseville_population (m : ℕ) : ℤ := m * willowdale_population - 500
def sun_city_relation (m : ℕ) : Prop := 
  sun_city_population = 2 * (roseville_population m) + 1000

/- State the theorem -/
theorem city_population_multiple : ∃ m : ℕ, sun_city_relation m ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_city_population_multiple_l1162_116263


namespace NUMINAMATH_CALUDE_fraction_equality_l1162_116205

theorem fraction_equality (a b : ℝ) (h : a + b ≠ 0) : (-a - b) / (a + b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1162_116205


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1162_116256

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : (1 - 2*I)*z = 5*I) : 
  z.im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1162_116256


namespace NUMINAMATH_CALUDE_D_largest_l1162_116269

def D : ℚ := 3006 / 3005 + 3006 / 3007
def E : ℚ := 3006 / 3007 + 3008 / 3007
def F : ℚ := 3007 / 3006 + 3007 / 3008

theorem D_largest : D > E ∧ D > F := by
  sorry

end NUMINAMATH_CALUDE_D_largest_l1162_116269


namespace NUMINAMATH_CALUDE_train_speed_l1162_116215

/-- The speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) (h1 : train_length = 360) (h2 : bridge_length = 140) (h3 : time = 40) :
  (train_length + bridge_length) / time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1162_116215


namespace NUMINAMATH_CALUDE_sharmila_hourly_wage_l1162_116288

/-- Sharmila's work schedule and earnings -/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate the total hours worked in a week -/
def total_hours (schedule : WorkSchedule) : ℕ :=
  schedule.monday_hours + schedule.tuesday_hours + schedule.wednesday_hours +
  schedule.thursday_hours + schedule.friday_hours

/-- Calculate the hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sharmila's actual work schedule -/
def sharmila_schedule : WorkSchedule :=
  { monday_hours := 10
  , tuesday_hours := 8
  , wednesday_hours := 10
  , thursday_hours := 8
  , friday_hours := 10
  , weekly_earnings := 460 }

theorem sharmila_hourly_wage :
  hourly_wage sharmila_schedule = 10 := by
  sorry

end NUMINAMATH_CALUDE_sharmila_hourly_wage_l1162_116288


namespace NUMINAMATH_CALUDE_third_side_length_l1162_116278

/-- A triangle with sides a, b, and c is valid if it satisfies the triangle inequality theorem --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given two sides of a triangle with lengths 1 and 3, the third side must be 3 --/
theorem third_side_length :
  ∀ x : ℝ, is_valid_triangle 1 3 x → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_l1162_116278


namespace NUMINAMATH_CALUDE_bigBoxesPackedIs120_l1162_116275

/-- Given the total number of items, items per small box, and small boxes per big box,
    calculate the number of big boxes packed. -/
def bigBoxesPacked (totalItems smallBoxItems smallBoxesPerBigBox : ℕ) : ℕ :=
  (totalItems / smallBoxItems) / smallBoxesPerBigBox

/-- Theorem stating that given the specific values from the problem,
    the number of big boxes packed is 120. -/
theorem bigBoxesPackedIs120 :
  bigBoxesPacked 8640 12 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_bigBoxesPackedIs120_l1162_116275


namespace NUMINAMATH_CALUDE_tessa_final_debt_l1162_116246

/-- Calculates the final debt given an initial debt, a fractional repayment, and an additional loan --/
def finalDebt (initialDebt : ℚ) (repaymentFraction : ℚ) (additionalLoan : ℚ) : ℚ :=
  initialDebt - (repaymentFraction * initialDebt) + additionalLoan

/-- Proves that Tessa's final debt is $30 --/
theorem tessa_final_debt :
  finalDebt 40 (1/2) 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_tessa_final_debt_l1162_116246


namespace NUMINAMATH_CALUDE_scientific_notation_of_43000000_l1162_116223

theorem scientific_notation_of_43000000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 43000000 = a * (10 : ℝ) ^ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_43000000_l1162_116223


namespace NUMINAMATH_CALUDE_angle_D_value_l1162_116283

-- Define the angles as real numbers
variable (A B C D : ℝ)

-- State the given conditions
axiom angle_sum : A + B = 180
axiom angle_relation : C = D + 10
axiom angle_A : A = 50
axiom triangle_sum : B + C + D = 180

-- State the theorem to be proved
theorem angle_D_value : D = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_value_l1162_116283


namespace NUMINAMATH_CALUDE_tensor_properties_l1162_116262

/-- Define a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Define the ⊗ operation -/
def tensor (a b : Vector2D) : ℝ :=
  a.x * b.y - b.x * a.y

/-- Define the dot product -/
def dot (a b : Vector2D) : ℝ :=
  a.x * b.x + a.y * b.y

theorem tensor_properties (m n p q : ℝ) :
  let a : Vector2D := ⟨m, n⟩
  let b : Vector2D := ⟨p, q⟩
  (tensor a a = 0) ∧
  ((tensor a b)^2 + (dot a b)^2 = (m^2 + q^2) * (n^2 + p^2)) := by
  sorry

end NUMINAMATH_CALUDE_tensor_properties_l1162_116262


namespace NUMINAMATH_CALUDE_triangle_proof_l1162_116242

theorem triangle_proof (A B C : ℝ) (a b c : ℝ) (P : ℝ × ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a = Real.sqrt 5 ∧
  c * Real.sin A = Real.sqrt 2 * Real.sin ((A + B) / 2) ∧
  Real.sqrt 5 = Real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2) ∧
  Real.sqrt 5 = Real.sqrt ((1 - P.1)^2 + (0 - P.2)^2) ∧
  1 = Real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2) ∧
  3 * π / 4 = Real.arccos ((P.1 * 1 + P.2 * 0) / (Real.sqrt (P.1^2 + P.2^2) * Real.sqrt 5)) →
  C = π / 2 ∧
  Real.sqrt ((1 - P.1)^2 + (0 - P.2)^2) = Real.sqrt ((P.1 - 1)^2 + P.2^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l1162_116242


namespace NUMINAMATH_CALUDE_sqrt_product_equals_200_l1162_116243

theorem sqrt_product_equals_200 : Real.sqrt 100 * Real.sqrt 50 * Real.sqrt 8 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_200_l1162_116243


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l1162_116277

-- Define the grid type
def Grid := Matrix (Fin 5) (Fin 5) ℕ

-- Define the sum of a list of numbers
def list_sum (l : List ℕ) : ℕ := l.foldl (·+·) 0

-- Define the property that a grid contains numbers 1 to 12
def contains_one_to_twelve (g : Grid) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → ∃ i j, g i j = n

-- Define the sum of central columns
def central_columns_sum (g : Grid) : Prop :=
  list_sum [g 0 2, g 1 2, g 2 2, g 3 2] = 26 ∧
  list_sum [g 0 3, g 1 3, g 2 3, g 3 3] = 26

-- Define the sum of central rows
def central_rows_sum (g : Grid) : Prop :=
  list_sum [g 2 0, g 2 1, g 2 2, g 2 3] = 26 ∧
  list_sum [g 3 0, g 3 1, g 3 2, g 3 3] = 26

-- Define the sum of roses pattern
def roses_sum (g : Grid) : Prop :=
  list_sum [g 0 2, g 1 2, g 2 2, g 2 3] = 26

-- Define the sum of shamrocks pattern
def shamrocks_sum (g : Grid) : Prop :=
  list_sum [g 2 0, g 3 1, g 4 2, g 1 2] = 26

-- Define the sum of thistle pattern
def thistle_sum (g : Grid) : Prop :=
  list_sum [g 2 2, g 3 2, g 3 3] = 26

-- The main theorem
theorem exists_valid_arrangement :
  ∃ g : Grid,
    contains_one_to_twelve g ∧
    central_columns_sum g ∧
    central_rows_sum g ∧
    roses_sum g ∧
    shamrocks_sum g ∧
    thistle_sum g := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l1162_116277


namespace NUMINAMATH_CALUDE_paint_combinations_count_l1162_116235

/-- The number of available paint colors -/
def num_colors : ℕ := 6

/-- The number of available painting tools -/
def num_tools : ℕ := 4

/-- The number of combinations of color and different tools for two objects -/
def num_combinations : ℕ := num_colors * num_tools * (num_tools - 1)

theorem paint_combinations_count :
  num_combinations = 72 := by
  sorry

end NUMINAMATH_CALUDE_paint_combinations_count_l1162_116235


namespace NUMINAMATH_CALUDE_protein_percentage_of_first_meal_l1162_116208

-- Define the constants
def total_weight : ℝ := 280
def mixture_protein_percentage : ℝ := 13
def cornmeal_protein_percentage : ℝ := 7
def first_meal_weight : ℝ := 240
def cornmeal_weight : ℝ := total_weight - first_meal_weight

-- Define the theorem
theorem protein_percentage_of_first_meal :
  let total_protein := total_weight * mixture_protein_percentage / 100
  let cornmeal_protein := cornmeal_weight * cornmeal_protein_percentage / 100
  let first_meal_protein := total_protein - cornmeal_protein
  first_meal_protein / first_meal_weight * 100 = 14 := by
sorry

end NUMINAMATH_CALUDE_protein_percentage_of_first_meal_l1162_116208


namespace NUMINAMATH_CALUDE_function_property_l1162_116281

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic_neg (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def half_x_on_unit (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 1/2 * x

theorem function_property (f : ℝ → ℝ) 
  (h1 : is_odd f) 
  (h2 : periodic_neg f) 
  (h3 : half_x_on_unit f) : 
  {x : ℝ | f x = -1/2} = {x : ℝ | ∃ k : ℤ, x = 4 * k - 1} := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1162_116281


namespace NUMINAMATH_CALUDE_planes_distance_l1162_116211

/-- Represents a plane in 3D space defined by the equation ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the distance between two parallel planes -/
def distance_between_planes (p1 p2 : Plane) : ℝ :=
  sorry

/-- The two planes in the problem -/
def plane1 : Plane := ⟨3, -1, 2, -4⟩
def plane2 : Plane := ⟨6, -2, 4, 3⟩

theorem planes_distance :
  distance_between_planes plane1 plane2 = 11 * Real.sqrt 14 / 28 := by
  sorry

end NUMINAMATH_CALUDE_planes_distance_l1162_116211


namespace NUMINAMATH_CALUDE_intersection_implies_z_value_l1162_116216

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sets M and N
def M (z : ℂ) : Set ℂ := {1, 2, z * i}
def N : Set ℂ := {3, 4}

-- State the theorem
theorem intersection_implies_z_value (z : ℂ) : 
  M z ∩ N = {4} → z = -4 * i :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_z_value_l1162_116216


namespace NUMINAMATH_CALUDE_football_hits_ground_time_l1162_116247

def football_height (t : ℝ) : ℝ := -16 * t^2 + 18 * t + 60

theorem football_hits_ground_time :
  ∃ t : ℝ, t > 0 ∧ football_height t = 0 ∧ t = 41 / 16 := by
  sorry

end NUMINAMATH_CALUDE_football_hits_ground_time_l1162_116247
