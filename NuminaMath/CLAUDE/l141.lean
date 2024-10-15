import Mathlib

namespace NUMINAMATH_CALUDE_total_out_of_pocket_is_190_50_l141_14137

def consultation_cost : ℝ := 300
def consultation_coverage : ℝ := 0.8
def xray_cost : ℝ := 150
def xray_coverage : ℝ := 0.7
def medication_cost : ℝ := 75
def medication_coverage : ℝ := 0.5
def therapy_cost : ℝ := 120
def therapy_coverage : ℝ := 0.6

def total_out_of_pocket_cost : ℝ :=
  (1 - consultation_coverage) * consultation_cost +
  (1 - xray_coverage) * xray_cost +
  (1 - medication_coverage) * medication_cost +
  (1 - therapy_coverage) * therapy_cost

theorem total_out_of_pocket_is_190_50 :
  total_out_of_pocket_cost = 190.50 := by
  sorry

end NUMINAMATH_CALUDE_total_out_of_pocket_is_190_50_l141_14137


namespace NUMINAMATH_CALUDE_jake_balloons_count_l141_14168

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 5

/-- The additional number of balloons Jake brought compared to Allan -/
def jake_extra_balloons : ℕ := 6

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := allan_balloons + jake_extra_balloons

theorem jake_balloons_count : jake_balloons = 11 := by sorry

end NUMINAMATH_CALUDE_jake_balloons_count_l141_14168


namespace NUMINAMATH_CALUDE_infinite_sum_floor_floor_2x_l141_14161

/-- For any real number x, the sum of floor((x + 2^k) / 2^(k+1)) from k=0 to infinity is equal to floor(x). -/
theorem infinite_sum_floor (x : ℝ) : 
  (∑' k, ⌊(x + 2^k) / 2^(k+1)⌋) = ⌊x⌋ :=
by sorry

/-- For any real number x, floor(2x) = floor(x) + floor(x + 1/2). -/
theorem floor_2x (x : ℝ) : 
  ⌊2*x⌋ = ⌊x⌋ + ⌊x + 1/2⌋ :=
by sorry

end NUMINAMATH_CALUDE_infinite_sum_floor_floor_2x_l141_14161


namespace NUMINAMATH_CALUDE_equation_solution_l141_14152

theorem equation_solution : 
  let t : ℚ := -8
  (1 : ℚ) / (t + 2) + (2 * t) / (t + 2) - (3 : ℚ) / (t + 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l141_14152


namespace NUMINAMATH_CALUDE_floor_plus_x_equals_seventeen_fourths_l141_14101

theorem floor_plus_x_equals_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by sorry

end NUMINAMATH_CALUDE_floor_plus_x_equals_seventeen_fourths_l141_14101


namespace NUMINAMATH_CALUDE_rectangle_formation_ways_l141_14127

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 5

theorem rectangle_formation_ways : 
  (Nat.choose horizontal_lines 2) * (Nat.choose vertical_lines 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_ways_l141_14127


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l141_14121

/-- An arithmetic sequence with a positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_positive : d > 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence where a_2, a_6, and a_12 form a geometric sequence,
    the ratio of a_12 to a_2 is 9/4 -/
theorem arithmetic_geometric_ratio
  (seq : ArithmeticSequence)
  (h_geometric : (seq.a 6) ^ 2 = (seq.a 2) * (seq.a 12)) :
  (seq.a 12) / (seq.a 2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l141_14121


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l141_14183

theorem cubic_equation_solutions :
  {x : ℝ | x^3 + (2 - x)^3 = 8} = {0, 2} := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l141_14183


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l141_14149

theorem smallest_angle_in_special_triangle : 
  ∀ (a b c : ℝ), 
    a + b + c = 180 →  -- Sum of angles is 180 degrees
    c = 5 * a →        -- Largest angle is 5 times the smallest
    b = 3 * a →        -- Middle angle is 3 times the smallest
    a = 20 :=          -- Smallest angle is 20 degrees
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l141_14149


namespace NUMINAMATH_CALUDE_santanas_brothers_birthdays_l141_14154

/-- The number of Santana's brothers -/
def total_brothers : ℕ := 7

/-- The number of brothers with birthdays in March -/
def march_birthdays : ℕ := 3

/-- The number of brothers with birthdays in November -/
def november_birthdays : ℕ := 1

/-- The number of brothers with birthdays in December -/
def december_birthdays : ℕ := 2

/-- The difference in presents between the second and first half of the year -/
def present_difference : ℕ := 8

/-- The number of brothers with birthdays in October -/
def october_birthdays : ℕ := total_brothers - (march_birthdays + november_birthdays + december_birthdays)

theorem santanas_brothers_birthdays :
  october_birthdays = 1 :=
by sorry

end NUMINAMATH_CALUDE_santanas_brothers_birthdays_l141_14154


namespace NUMINAMATH_CALUDE_equal_roots_k_value_l141_14193

/-- The cubic equation with parameter k -/
def cubic_equation (x k : ℝ) : ℝ :=
  3 * x^3 + 9 * x^2 - 162 * x + k

/-- Theorem stating that if the cubic equation has two equal roots and k is positive, then k = 7983/125 -/
theorem equal_roots_k_value (k : ℝ) :
  (∃ a b : ℝ, a ≠ b ∧
    cubic_equation a k = 0 ∧
    cubic_equation b k = 0 ∧
    (∃ x : ℝ, x ≠ a ∧ x ≠ b ∧ cubic_equation x k = 0)) →
  k > 0 →
  k = 7983 / 125 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_k_value_l141_14193


namespace NUMINAMATH_CALUDE_hyperbola_dot_product_nonnegative_l141_14150

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- The left vertex of the hyperbola -/
def A : ℝ × ℝ := (-2, 0)

/-- The right vertex of the hyperbola -/
def B : ℝ × ℝ := (2, 0)

/-- The dot product of vectors PA and PB -/
def dot_product (P : ℝ × ℝ) : ℝ :=
  let (m, n) := P
  ((-2 - m) * (2 - m)) + (n * n)

theorem hyperbola_dot_product_nonnegative :
  ∀ P : ℝ × ℝ, hyperbola P.1 P.2 → dot_product P ≥ 0 := by sorry

end NUMINAMATH_CALUDE_hyperbola_dot_product_nonnegative_l141_14150


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l141_14103

/-- The coordinates of a point with respect to the origin are the same as its given coordinates. -/
theorem point_coordinates_wrt_origin (x y : ℝ) :
  let M : ℝ × ℝ := (x, y)
  M = (x, y) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l141_14103


namespace NUMINAMATH_CALUDE_conclusion_l141_14159

-- Define the variables
variable (p q r s u v : ℝ)

-- State the given conditions
axiom cond1 : p > q → r > s
axiom cond2 : r = s → u < v
axiom cond3 : p = q → s > r

-- State the theorem to be proved
theorem conclusion : p ≠ q → s ≠ r := by
  sorry

end NUMINAMATH_CALUDE_conclusion_l141_14159


namespace NUMINAMATH_CALUDE_largest_divisible_by_thirtyseven_with_decreasing_digits_l141_14122

/-- 
A function that checks if a natural number's digits are in strictly decreasing order.
-/
def isStrictlyDecreasing (n : ℕ) : Prop :=
  sorry

/-- 
A function that finds the largest natural number less than or equal to n 
that is divisible by 37 and has strictly decreasing digits.
-/
def largestDivisibleByThirtySevenWithDecreasingDigits (n : ℕ) : ℕ :=
  sorry

theorem largest_divisible_by_thirtyseven_with_decreasing_digits :
  largestDivisibleByThirtySevenWithDecreasingDigits 9876543210 = 987654 :=
sorry

end NUMINAMATH_CALUDE_largest_divisible_by_thirtyseven_with_decreasing_digits_l141_14122


namespace NUMINAMATH_CALUDE_quartic_inequality_l141_14110

theorem quartic_inequality (a b : ℝ) : 
  (∃ x : ℝ, x^4 - a*x^3 + 2*x^2 - b*x + 1 = 0) → a^2 + b^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_quartic_inequality_l141_14110


namespace NUMINAMATH_CALUDE_vector_dot_product_symmetry_and_value_l141_14136

/-- Given vectors a and b, and function f as defined, prove the axis of symmetry and a specific function value. -/
theorem vector_dot_product_symmetry_and_value 
  (x θ : ℝ) 
  (a : ℝ → ℝ × ℝ)
  (b : ℝ → ℝ × ℝ)
  (f : ℝ → ℝ)
  (h1 : a = λ x => (Real.sin x, 1))
  (h2 : b = λ x => (1, Real.cos x))
  (h3 : f = λ x => (a x).1 * (b x).1 + (a x).2 * (b x).2)
  (h4 : f (θ + π/4) = Real.sqrt 2 / 3)
  (h5 : 0 < θ)
  (h6 : θ < π/2) :
  (∃ k : ℤ, ∀ x, f x = f (2 * (k * π + π/4) - x)) ∧ 
  f (θ - π/4) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_symmetry_and_value_l141_14136


namespace NUMINAMATH_CALUDE_total_amount_calculation_total_amount_is_3693_2_l141_14171

/-- Calculate the total amount received after selling three items with given prices, losses, and VAT -/
theorem total_amount_calculation (price_A price_B price_C : ℝ)
                                 (loss_A loss_B loss_C : ℝ)
                                 (vat : ℝ) : ℝ :=
  let selling_price_A := price_A * (1 - loss_A)
  let selling_price_B := price_B * (1 - loss_B)
  let selling_price_C := price_C * (1 - loss_C)
  let total_selling_price := selling_price_A + selling_price_B + selling_price_C
  let total_with_vat := total_selling_price * (1 + vat)
  total_with_vat

/-- The total amount received after selling all three items, including VAT, is Rs. 3693.2 -/
theorem total_amount_is_3693_2 :
  total_amount_calculation 1300 750 1800 0.20 0.15 0.10 0.12 = 3693.2 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_calculation_total_amount_is_3693_2_l141_14171


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l141_14176

/-- A color type representing red, green, or blue -/
inductive Color
| Red
| Green
| Blue

/-- A type representing a 4 × 82 grid where each cell is colored -/
def Grid := Fin 4 → Fin 82 → Color

/-- A function to check if four points form a rectangle with the same color -/
def isMonochromaticRectangle (g : Grid) (x1 y1 x2 y2 : ℕ) : Prop :=
  x1 < x2 ∧ y1 < y2 ∧
  g ⟨x1, by sorry⟩ ⟨y1, by sorry⟩ = g ⟨x1, by sorry⟩ ⟨y2, by sorry⟩ ∧
  g ⟨x1, by sorry⟩ ⟨y1, by sorry⟩ = g ⟨x2, by sorry⟩ ⟨y1, by sorry⟩ ∧
  g ⟨x1, by sorry⟩ ⟨y1, by sorry⟩ = g ⟨x2, by sorry⟩ ⟨y2, by sorry⟩

/-- Theorem: In any 3-coloring of a 4 × 82 grid, there exists a rectangle whose vertices are all the same color -/
theorem monochromatic_rectangle_exists (g : Grid) :
  ∃ (x1 y1 x2 y2 : ℕ), isMonochromaticRectangle g x1 y1 x2 y2 :=
sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l141_14176


namespace NUMINAMATH_CALUDE_cookie_cost_difference_l141_14178

theorem cookie_cost_difference (cookie_cost diane_money : ℕ) 
  (h1 : cookie_cost = 65)
  (h2 : diane_money = 27) :
  cookie_cost - diane_money = 38 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_difference_l141_14178


namespace NUMINAMATH_CALUDE_cos_36_degrees_l141_14174

theorem cos_36_degrees (x y : ℝ) : 
  x = Real.cos (36 * π / 180) →
  y = Real.cos (72 * π / 180) →
  y = 2 * x^2 - 1 →
  x = 2 * y^2 - 1 →
  x = (1 + Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_cos_36_degrees_l141_14174


namespace NUMINAMATH_CALUDE_pet_store_birds_pet_store_birds_after_changes_l141_14109

/-- The number of birds in a pet store after sales and additions --/
theorem pet_store_birds (num_cages : ℕ) (initial_parrots : ℕ) (initial_parakeets : ℕ) (initial_canaries : ℕ)
  (sold_parrots : ℕ) (sold_canaries : ℕ) (added_parakeets : ℕ) : ℕ :=
  let total_initial_parrots := num_cages * initial_parrots
  let total_initial_parakeets := num_cages * initial_parakeets
  let total_initial_canaries := num_cages * initial_canaries
  let final_parrots := total_initial_parrots - sold_parrots
  let final_parakeets := total_initial_parakeets + added_parakeets
  let final_canaries := total_initial_canaries - sold_canaries
  final_parrots + final_parakeets + final_canaries

/-- The number of birds in the pet store after changes is 235 --/
theorem pet_store_birds_after_changes : pet_store_birds 15 3 8 5 5 2 2 = 235 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_pet_store_birds_after_changes_l141_14109


namespace NUMINAMATH_CALUDE_total_water_consumed_l141_14188

-- Define the conversion rate from quarts to ounces
def quart_to_ounce : ℚ := 32

-- Define the amount of water in the bottle (in quarts)
def bottle_water : ℚ := 3/2

-- Define the amount of water in the can (in ounces)
def can_water : ℚ := 12

-- Theorem statement
theorem total_water_consumed :
  bottle_water * quart_to_ounce + can_water = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_water_consumed_l141_14188


namespace NUMINAMATH_CALUDE_room_tiles_l141_14114

/-- Calculates the number of tiles needed for a rectangular room with a border --/
def total_tiles (length width : ℕ) (border_width : ℕ) : ℕ :=
  let border_tiles := 2 * (length + width - 2 * border_width) * border_width
  let inner_length := length - 2 * border_width
  let inner_width := width - 2 * border_width
  let inner_tiles := (inner_length * inner_width) / 4
  border_tiles + inner_tiles

/-- Theorem stating that a 15x20 room with a 2-foot border requires 168 tiles --/
theorem room_tiles : total_tiles 20 15 2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_room_tiles_l141_14114


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l141_14199

theorem algebraic_expression_value (x y : ℝ) :
  x^4 + 6*x^2*y + 9*y^2 + 2*x^2 + 6*y + 4 = 7 →
  (x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = -2) ∨
  (x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = 14) := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l141_14199


namespace NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_one_l141_14147

theorem negation_of_forall_x_squared_gt_one :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_one_l141_14147


namespace NUMINAMATH_CALUDE_min_value_circle_l141_14132

theorem min_value_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (min : ℝ), (∀ (a b : ℝ), a^2 + b^2 - 4*a + 1 = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ min = 7 - 4*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_circle_l141_14132


namespace NUMINAMATH_CALUDE_binary_101111011_equals_379_l141_14164

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def binary_101111011 : List Bool := [true, true, false, true, true, true, true, false, true]

theorem binary_101111011_equals_379 :
  binary_to_decimal binary_101111011 = 379 := by
  sorry

end NUMINAMATH_CALUDE_binary_101111011_equals_379_l141_14164


namespace NUMINAMATH_CALUDE_final_digit_independent_of_sequence_l141_14155

/-- Represents the count of each digit on the blackboard -/
structure DigitCount where
  zeros : Nat
  ones : Nat
  twos : Nat

/-- Represents a single step of the digit replacement operation -/
def replaceDigits (count : DigitCount) : DigitCount :=
  sorry

/-- Determines if the operation can continue (more than one digit type remains) -/
def canContinue (count : DigitCount) : Bool :=
  sorry

/-- Performs the digit replacement operations until only one digit type remains -/
def performOperations (initial : DigitCount) : Nat :=
  sorry

theorem final_digit_independent_of_sequence (initial : DigitCount) :
  ∀ (seq1 seq2 : List (DigitCount → DigitCount)),
    (seq1.foldl (fun acc f => f acc) initial).zeros +
    (seq1.foldl (fun acc f => f acc) initial).ones +
    (seq1.foldl (fun acc f => f acc) initial).twos = 1 →
    (seq2.foldl (fun acc f => f acc) initial).zeros +
    (seq2.foldl (fun acc f => f acc) initial).ones +
    (seq2.foldl (fun acc f => f acc) initial).twos = 1 →
    (seq1.foldl (fun acc f => f acc) initial) = (seq2.foldl (fun acc f => f acc) initial) :=
  sorry

end NUMINAMATH_CALUDE_final_digit_independent_of_sequence_l141_14155


namespace NUMINAMATH_CALUDE_multiply_by_seven_l141_14131

theorem multiply_by_seven (x : ℝ) (h : 8 * x = 64) : 7 * x = 56 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_seven_l141_14131


namespace NUMINAMATH_CALUDE_abc_triangle_properties_l141_14196

/-- Given positive real numbers x, y, and z, we define a, b, and c as follows:
    a = x + 1/y
    b = y + 1/z
    c = z + 1/x
    Assuming a, b, and c form the sides of a triangle, we prove two statements about them. -/
theorem abc_triangle_properties (x y z : ℝ) 
    (hx : x > 0) (hy : y > 0) (hz : z > 0)
    (ha : a = x + 1/y) (hb : b = y + 1/z) (hc : c = z + 1/x)
    (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) :
    (max a b ≥ 2 ∨ c ≥ 2) ∧ (a + b) / (1 + a + b) > c / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_abc_triangle_properties_l141_14196


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l141_14190

theorem imaginary_part_of_z : 
  let z : ℂ := (1 - I) / (1 + 3*I)
  Complex.im z = -2/5 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l141_14190


namespace NUMINAMATH_CALUDE_english_only_enrollment_l141_14181

/-- The number of students enrolled only in English -/
def students_only_english (total : ℕ) (both_eng_ger : ℕ) (german : ℕ) (spanish : ℕ) : ℕ :=
  total - (german + spanish - both_eng_ger)

theorem english_only_enrollment :
  let total := 75
  let both_eng_ger := 18
  let german := 32
  let spanish := 25
  students_only_english total both_eng_ger german spanish = 18 := by
  sorry

#eval students_only_english 75 18 32 25

end NUMINAMATH_CALUDE_english_only_enrollment_l141_14181


namespace NUMINAMATH_CALUDE_not_enough_money_l141_14108

/-- The cost of a new smartwatch in rubles -/
def smartwatch_cost : ℕ := 2019

/-- The amount of money Namzhil has in rubles -/
def namzhil_money : ℕ := (500^2 + 4 * 500 + 3) * 498^2 - 500^2 * 503 * 497

/-- Theorem stating that Namzhil does not have enough money to buy the smartwatch -/
theorem not_enough_money : namzhil_money < smartwatch_cost := by
  sorry

end NUMINAMATH_CALUDE_not_enough_money_l141_14108


namespace NUMINAMATH_CALUDE_study_group_lawyers_l141_14167

theorem study_group_lawyers (total_members : ℝ) (h1 : total_members > 0) : 
  let women_ratio : ℝ := 0.4
  let women_lawyer_prob : ℝ := 0.08
  let women_lawyer_ratio : ℝ := women_lawyer_prob / women_ratio
  women_lawyer_ratio = 0.2 := by sorry

end NUMINAMATH_CALUDE_study_group_lawyers_l141_14167


namespace NUMINAMATH_CALUDE_exist_nonzero_superintegers_with_zero_product_l141_14173

-- Define a super-integer as a function from ℕ to ℕ
def SuperInteger := ℕ → ℕ

-- Define a zero super-integer
def isZeroSuperInteger (x : SuperInteger) : Prop :=
  ∀ n, x n = 0

-- Define non-zero super-integer
def isNonZeroSuperInteger (x : SuperInteger) : Prop :=
  ∃ n, x n ≠ 0

-- Define the product of two super-integers
def superIntegerProduct (x y : SuperInteger) : SuperInteger :=
  fun n => (x n * y n) % (10^n)

-- Theorem statement
theorem exist_nonzero_superintegers_with_zero_product :
  ∃ (x y : SuperInteger),
    isNonZeroSuperInteger x ∧
    isNonZeroSuperInteger y ∧
    isZeroSuperInteger (superIntegerProduct x y) := by
  sorry


end NUMINAMATH_CALUDE_exist_nonzero_superintegers_with_zero_product_l141_14173


namespace NUMINAMATH_CALUDE_train_speed_problem_l141_14158

theorem train_speed_problem (distance : ℝ) (speed_ab : ℝ) (time_difference : ℝ) :
  distance = 480 →
  speed_ab = 160 →
  time_difference = 1 →
  let time_ab := distance / speed_ab
  let time_ba := time_ab + time_difference
  let speed_ba := distance / time_ba
  speed_ba = 120 := by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l141_14158


namespace NUMINAMATH_CALUDE_cos_10_coeff_sum_l141_14102

/-- Represents the coefficients in the expansion of cos(10α) -/
structure Cos10Coeffs where
  m : ℤ
  n : ℤ
  p : ℤ

/-- 
Given the equation for cos(10α) in the form:
cos(10α) = m*cos^10(α) - 1280*cos^8(α) + 1120*cos^6(α) + n*cos^4(α) + p*cos^2(α) - 1,
prove that m - n + p = 962
-/
theorem cos_10_coeff_sum (coeffs : Cos10Coeffs) : coeffs.m - coeffs.n + coeffs.p = 962 := by
  sorry

end NUMINAMATH_CALUDE_cos_10_coeff_sum_l141_14102


namespace NUMINAMATH_CALUDE_cubic_polynomial_solution_l141_14144

noncomputable section

variable (a b c : ℝ)
variable (P : ℝ → ℝ)

def cubic_equation (x : ℝ) : Prop := x^3 + 2*x^2 + 4*x + 6 = 0

theorem cubic_polynomial_solution :
  cubic_equation a ∧ cubic_equation b ∧ cubic_equation c ∧
  (∀ x, ∃ p q r s, P x = p*x^3 + q*x^2 + r*x + s) ∧
  P a = b + c ∧
  P b = a + c ∧
  P c = a + b ∧
  P (a + b + c) = -18 →
  ∀ x, P x = 9/4*x^3 + 5/2*x^2 + 7*x + 15/2 :=
by sorry

end

end NUMINAMATH_CALUDE_cubic_polynomial_solution_l141_14144


namespace NUMINAMATH_CALUDE_pairing_probabilities_correct_probabilities_sum_to_one_l141_14113

-- Define the total number of teams and English teams
def total_teams : ℕ := 8
def english_teams : ℕ := 4

-- Define the total number of possible pairings
def total_pairings : ℕ := 105

-- Define the probabilities for each scenario
def prob_no_english_pairs : ℚ := 24 / 105
def prob_two_english_pairs : ℚ := 18 / 105
def prob_one_english_pair : ℚ := 72 / 105

-- Theorem to prove the correctness of the probabilities
theorem pairing_probabilities_correct :
  (prob_no_english_pairs + prob_two_english_pairs + prob_one_english_pair = 1) ∧
  (prob_no_english_pairs = 24 / 105) ∧
  (prob_two_english_pairs = 18 / 105) ∧
  (prob_one_english_pair = 72 / 105) := by
  sorry

-- Theorem to prove that the probabilities sum to 1
theorem probabilities_sum_to_one :
  prob_no_english_pairs + prob_two_english_pairs + prob_one_english_pair = 1 := by
  sorry

end NUMINAMATH_CALUDE_pairing_probabilities_correct_probabilities_sum_to_one_l141_14113


namespace NUMINAMATH_CALUDE_digit_five_minus_nine_in_book_pages_l141_14169

/-- Counts the occurrences of a digit in a number -/
def countDigit (d : Nat) (n : Nat) : Nat :=
  sorry

/-- Counts the occurrences of a digit in a range of numbers -/
def countDigitInRange (d : Nat) (start finish : Nat) : Nat :=
  sorry

theorem digit_five_minus_nine_in_book_pages : 
  ∀ (n : Nat), n = 599 →
  (countDigitInRange 5 1 n) - (countDigitInRange 9 1 n) = 100 := by
  sorry

end NUMINAMATH_CALUDE_digit_five_minus_nine_in_book_pages_l141_14169


namespace NUMINAMATH_CALUDE_prime_sum_problem_l141_14192

theorem prime_sum_problem (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r →
  p + q = r + 2 →
  1 < p →
  p < q →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l141_14192


namespace NUMINAMATH_CALUDE_expression_value_l141_14182

theorem expression_value (x y : ℝ) (h : x^2 - 2*y = -1) : 
  3*x^2 - 6*y + 2023 = 2020 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l141_14182


namespace NUMINAMATH_CALUDE_division_problem_l141_14116

theorem division_problem (x y : ℕ+) 
  (h1 : (x : ℝ) / (y : ℝ) = 96.12)
  (h2 : (x : ℝ) % (y : ℝ) = 11.52) : 
  y = 96 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l141_14116


namespace NUMINAMATH_CALUDE_parabola_point_distance_to_origin_l141_14112

theorem parabola_point_distance_to_origin :
  ∀ (x y : ℝ),
  y^2 = 2*x →  -- Point A is on the parabola y^2 = 2x
  (x + 1/2) / |y| = 5/4 →  -- Ratio condition
  ((x - 1/2)^2 + y^2)^(1/2) > 2 →  -- |AF| > 2
  (x^2 + y^2)^(1/2) = 2 * (2^(1/2)) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_point_distance_to_origin_l141_14112


namespace NUMINAMATH_CALUDE_average_of_xyz_l141_14197

theorem average_of_xyz (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 15) : 
  (x + y + z) / 3 = 2 := by
sorry

end NUMINAMATH_CALUDE_average_of_xyz_l141_14197


namespace NUMINAMATH_CALUDE_binary_sum_equality_l141_14166

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

theorem binary_sum_equality : 
  let b1 := [true, true, false, true]  -- 1101₂
  let b2 := [true, false, true]        -- 101₂
  let b3 := [true, true, true, false]  -- 1110₂
  let b4 := [true, false, true, true, true]  -- 10111₂
  let b5 := [true, true, false, false, false]  -- 11000₂
  let sum := [true, true, true, false, false, false, true, false]  -- 11100010₂
  binary_to_nat b1 + binary_to_nat b2 + binary_to_nat b3 + 
  binary_to_nat b4 + binary_to_nat b5 = binary_to_nat sum := by
  sorry

#eval binary_to_nat [true, true, true, false, false, false, true, false]  -- Should output 226

end NUMINAMATH_CALUDE_binary_sum_equality_l141_14166


namespace NUMINAMATH_CALUDE_first_concert_attendance_l141_14117

theorem first_concert_attendance (second_concert : ℕ) (difference : ℕ) : 
  second_concert = 66018 → difference = 119 → second_concert - difference = 65899 := by
  sorry

end NUMINAMATH_CALUDE_first_concert_attendance_l141_14117


namespace NUMINAMATH_CALUDE_double_inequality_solution_l141_14140

theorem double_inequality_solution (x : ℝ) : 
  (3 ≤ |x - 3| ∧ |x - 3| ≤ 6 ∧ x ≤ 8) ↔ ((-3 ≤ x ∧ x ≤ 3) ∨ (6 ≤ x ∧ x ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l141_14140


namespace NUMINAMATH_CALUDE_quadratic_to_alternative_form_integer_values_iff_integer_coefficients_l141_14138

/-- Represents a quadratic expression Ax² + Bx + C -/
structure QuadraticExpression (α : Type) [Ring α] where
  A : α
  B : α
  C : α

/-- Represents the alternative form k(x(x-1)/2) + lx + m -/
structure AlternativeForm (α : Type) [Ring α] where
  k : α
  l : α
  m : α

/-- States that a quadratic expression can be written in the alternative form -/
theorem quadratic_to_alternative_form {α : Type} [Ring α] (q : QuadraticExpression α) :
  ∃ (a : AlternativeForm α), a.k = 2 * q.A ∧ a.l = q.A + q.B ∧ a.m = q.C :=
sorry

/-- States that the quadratic expression takes integer values for all integer x
    if and only if k, l, m in the alternative form are integers -/
theorem integer_values_iff_integer_coefficients (q : QuadraticExpression ℤ) :
  (∀ x : ℤ, ∃ y : ℤ, q.A * x^2 + q.B * x + q.C = y) ↔
  (∃ (a : AlternativeForm ℤ), a.k = 2 * q.A ∧ a.l = q.A + q.B ∧ a.m = q.C) :=
sorry

end NUMINAMATH_CALUDE_quadratic_to_alternative_form_integer_values_iff_integer_coefficients_l141_14138


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l141_14187

/-- The vertex of the parabola y = 1/2 * (x + 2)^2 + 1 has coordinates (-2, 1) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x ↦ (1/2) * (x + 2)^2 + 1
  ∃! (h k : ℝ), (∀ x, f x = (1/2) * (x - h)^2 + k) ∧ h = -2 ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l141_14187


namespace NUMINAMATH_CALUDE_inequality_theorem_l141_14139

theorem inequality_theorem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x * (x - z)^2 + y * (y - z)^2 ≥ (x - z) * (y - z) * (x + y - z) ∧
  (x * (x - z)^2 + y * (y - z)^2 = (x - z) * (y - z) * (x + y - z) ↔ 
    (x = y ∧ y = z) ∨ (x = y ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l141_14139


namespace NUMINAMATH_CALUDE_line_parameterization_l141_14141

/-- Given a line y = 5x - 7 parameterized by (x, y) = (s, 2) + t(3, m),
    prove that s = 9/5 and m = 3 -/
theorem line_parameterization (s m : ℝ) :
  (∀ t : ℝ, ∀ x y : ℝ, 
    x = s + 3*t ∧ y = 2 + m*t → y = 5*x - 7) →
  s = 9/5 ∧ m = 3 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l141_14141


namespace NUMINAMATH_CALUDE_translation_proof_l141_14172

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a line vertically by a given distance -/
def translateVertically (l : Line) (distance : ℝ) : Line :=
  { slope := l.slope, yIntercept := l.yIntercept + distance }

theorem translation_proof (l₁ l₂ : Line) :
  l₁.slope = 2 ∧ l₁.yIntercept = -2 ∧ l₂.slope = 2 ∧ l₂.yIntercept = 0 →
  translateVertically l₁ 2 = l₂ := by
  sorry

end NUMINAMATH_CALUDE_translation_proof_l141_14172


namespace NUMINAMATH_CALUDE_percentage_problem_l141_14145

theorem percentage_problem (p : ℝ) : 
  (0.65 * 40 = p / 100 * 60 + 23) → p = 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l141_14145


namespace NUMINAMATH_CALUDE_fraction_sum_equals_half_l141_14148

theorem fraction_sum_equals_half : (2 / 12 : ℚ) + (4 / 24 : ℚ) + (6 / 36 : ℚ) = (1 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_half_l141_14148


namespace NUMINAMATH_CALUDE_hallies_net_earnings_l141_14128

/-- Represents a day's work information -/
structure WorkDay where
  hours : ℕ
  hourlyRate : ℚ
  tips : ℚ

/-- Calculates the net earnings for the week -/
def calculateNetEarnings (week : List WorkDay) (taxRate : ℚ) (thursdayDiscountRate : ℚ) : ℚ :=
  sorry

/-- The main theorem stating Hallie's net earnings for the week -/
theorem hallies_net_earnings :
  let week : List WorkDay := [
    ⟨7, 10, 18⟩,  -- Monday
    ⟨5, 12, 12⟩,  -- Tuesday
    ⟨7, 10, 20⟩,  -- Wednesday
    ⟨8, 11, 25⟩,  -- Thursday
    ⟨6, 9, 15⟩    -- Friday
  ]
  let taxRate : ℚ := 5 / 100
  let thursdayDiscountRate : ℚ := 10 / 100
  calculateNetEarnings week taxRate thursdayDiscountRate = 406.1 :=
by sorry

end NUMINAMATH_CALUDE_hallies_net_earnings_l141_14128


namespace NUMINAMATH_CALUDE_age_ratio_proof_l141_14194

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  ∃ k : ℕ, b = k * c →  -- b is some multiple of c's age
  a + b + c = 32 →  -- The total of the ages of a, b, and c is 32
  b = 12 →  -- b is 12 years old
  b = 2 * c  -- The ratio of b's age to c's age is 2:1
:= by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l141_14194


namespace NUMINAMATH_CALUDE_sum_of_equations_l141_14123

theorem sum_of_equations (a b c d : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 4)
  (eq4 : d - a + b = 1) :
  2*a + 2*b + 2*c + 2*d = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_equations_l141_14123


namespace NUMINAMATH_CALUDE_probability_adjacent_knights_l141_14115

/-- The number of knights at the round table -/
def total_knights : ℕ := 30

/-- The number of knights chosen for the quest -/
def chosen_knights : ℕ := 4

/-- The probability that at least two of the four chosen knights were sitting next to each other -/
def Q : ℚ := 389 / 437

/-- Theorem stating that Q is the correct probability -/
theorem probability_adjacent_knights : 
  Q = 1 - (total_knights - chosen_knights) * (total_knights - chosen_knights - 2) * 
        (total_knights - chosen_knights - 4) * (total_knights - chosen_knights - 6) / 
        ((total_knights - 1) * total_knights * (total_knights + 1) * (total_knights - chosen_knights + 3)) :=
by sorry

end NUMINAMATH_CALUDE_probability_adjacent_knights_l141_14115


namespace NUMINAMATH_CALUDE_tetrahedron_volume_formula_l141_14146

/-- A tetrahedron with its properties -/
structure Tetrahedron where
  S : ℝ  -- Surface area
  R : ℝ  -- Radius of inscribed sphere
  V : ℝ  -- Volume

/-- Theorem: The volume of a tetrahedron is one-third the product of its surface area and the radius of its inscribed sphere -/
theorem tetrahedron_volume_formula (t : Tetrahedron) : t.V = (1/3) * t.S * t.R := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_formula_l141_14146


namespace NUMINAMATH_CALUDE_infinite_series_sum_l141_14100

theorem infinite_series_sum : 
  let series := fun n : ℕ => (n + 1 : ℝ) / 5^n
  ∑' n, series n = 9/16 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l141_14100


namespace NUMINAMATH_CALUDE_dennis_initial_money_l141_14124

-- Define the sale discount
def sale_discount : ℚ := 25 / 100

-- Define the original price of the shirts
def original_price : ℚ := 125

-- Define the amount Dennis paid
def amount_paid : ℚ := 100 + 50 + 4 * 5

-- Define the change Dennis received
def change_received : ℚ := 3 * 20 + 10 + 2 * 5 + 4

-- Theorem statement
theorem dennis_initial_money :
  let discounted_price := original_price * (1 - sale_discount)
  let initial_money := discounted_price + change_received
  initial_money = 177.75 := by
  sorry

end NUMINAMATH_CALUDE_dennis_initial_money_l141_14124


namespace NUMINAMATH_CALUDE_train_crossing_time_train_crossing_time_specific_l141_14191

/-- The time taken for a train to cross a post, given its speed and length -/
theorem train_crossing_time (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  let speed_ms : ℝ := speed_kmh * 1000 / 3600
  length_m / speed_ms

/-- Proof that a train with speed 40 km/h and length 220.0176 m takes approximately 19.80176 seconds to cross a post -/
theorem train_crossing_time_specific :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |train_crossing_time 40 220.0176 - 19.80176| < ε :=
sorry

end NUMINAMATH_CALUDE_train_crossing_time_train_crossing_time_specific_l141_14191


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l141_14180

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n_terms a n / sum_n_terms b n = (2 * n + 2 : ℚ) / (n + 3)) →
  a.a 10 / b.a 10 = 20 / 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l141_14180


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l141_14111

/-- The area of a square sheet of wrapping paper for a box with base side length s -/
def wrapping_paper_area (s : ℝ) : ℝ := 4 * s^2

/-- Theorem: The area of the square sheet of wrapping paper is 4s² -/
theorem wrapping_paper_area_theorem (s : ℝ) (h : s > 0) :
  wrapping_paper_area s = 4 * s^2 := by
  sorry

#check wrapping_paper_area_theorem

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l141_14111


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l141_14130

/-- An isosceles triangle with two sides of lengths 3 and 4 has a perimeter of either 10 or 11. -/
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = b ∧ (a = 3 ∧ c = 4 ∨ a = 4 ∧ c = 3)) ∨ (a = c ∧ (a = 3 ∧ b = 4 ∨ a = 4 ∧ b = 3)) ∨ (b = c ∧ (b = 3 ∧ a = 4 ∨ b = 4 ∧ a = 3)) →
  a + b + c = 10 ∨ a + b + c = 11 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l141_14130


namespace NUMINAMATH_CALUDE_rationalize_denominator_l141_14134

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (B < D) ∧
    (3 : ℚ) / (2 * Real.sqrt 18 + 5 * Real.sqrt 20) =
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    A = -18 ∧ B = 2 ∧ C = 30 ∧ D = 5 ∧ E = 428 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l141_14134


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l141_14156

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ)
  (ha : arithmetic_sequence a)
  (hb : arithmetic_sequence b)
  (h : ∀ n : ℕ+, sum_of_arithmetic_sequence a n / sum_of_arithmetic_sequence b n = (n + 1) / (2 * n - 1)) :
  (a 3 + a 7) / (b 1 + b 9) = 10 / 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l141_14156


namespace NUMINAMATH_CALUDE_quadratic_root_form_l141_14125

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 - 7*x + d = 0 ↔ x = (7 + Real.sqrt d) / 2 ∨ x = (7 - Real.sqrt d) / 2) → 
  d = 49 / 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_form_l141_14125


namespace NUMINAMATH_CALUDE_inner_shape_area_ratio_l141_14170

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- Points that trisect the sides of a hexagon -/
def trisection_points (h : RegularHexagon) : Fin 6 → ℝ × ℝ :=
  sorry

/-- The shape formed by joining the trisection points -/
def inner_shape (h : RegularHexagon) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating that the area of the inner shape is 2/3 of the original hexagon -/
theorem inner_shape_area_ratio (h : RegularHexagon) :
    area (inner_shape h) = (2 / 3) * area (Set.range h.vertices) := by
  sorry

end NUMINAMATH_CALUDE_inner_shape_area_ratio_l141_14170


namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l141_14157

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l141_14157


namespace NUMINAMATH_CALUDE_skateboard_ramp_speed_increase_l141_14118

/-- Calculates the additional speed required to reach the top of a skateboard ramp -/
theorem skateboard_ramp_speed_increase 
  (ramp_height : ℝ) 
  (ramp_incline : ℝ) 
  (speed_without_wind : ℝ) 
  (trial_speeds : List ℝ) 
  (wind_resistance_min : ℝ) 
  (wind_resistance_max : ℝ) : 
  ramp_height = 50 → 
  ramp_incline = 30 → 
  speed_without_wind = 40 → 
  trial_speeds = [36, 34, 38] → 
  wind_resistance_min = 3 → 
  wind_resistance_max = 5 → 
  (List.sum trial_speeds / trial_speeds.length + 
   (wind_resistance_min + wind_resistance_max) / 2 + 
   speed_without_wind) - 
  (List.sum trial_speeds / trial_speeds.length) = 8 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_ramp_speed_increase_l141_14118


namespace NUMINAMATH_CALUDE_kelly_games_to_give_away_l141_14107

/-- The number of games Kelly needs to give away to reach her desired number of games -/
def games_to_give_away (initial_games : ℕ) (desired_games : ℕ) : ℕ :=
  initial_games - desired_games

/-- Proof that Kelly needs to give away 15 games -/
theorem kelly_games_to_give_away :
  games_to_give_away 50 35 = 15 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_to_give_away_l141_14107


namespace NUMINAMATH_CALUDE_tape_pieces_for_cube_l141_14184

/-- Represents a cube with side length n -/
structure Cube where
  sideLength : ℕ

/-- Represents a piece of tape with width 1 cm -/
structure Tape where
  length : ℕ

/-- Function to calculate the number of tape pieces needed to cover a cube -/
def tapePiecesNeeded (c : Cube) : ℕ :=
  2 * c.sideLength

/-- Theorem stating that the number of tape pieces needed is 2n -/
theorem tape_pieces_for_cube (c : Cube) :
  tapePiecesNeeded c = 2 * c.sideLength := by
  sorry

#check tape_pieces_for_cube

end NUMINAMATH_CALUDE_tape_pieces_for_cube_l141_14184


namespace NUMINAMATH_CALUDE_gcd_105_88_l141_14195

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_88_l141_14195


namespace NUMINAMATH_CALUDE_orchestra_females_count_l141_14163

/-- The number of females in the orchestra -/
def females_in_orchestra : ℕ := 12

theorem orchestra_females_count :
  let males_in_orchestra : ℕ := 11
  let choir_size : ℕ := 12 + 17
  let total_musicians : ℕ := 98
  females_in_orchestra = 
    (total_musicians - choir_size - males_in_orchestra - 2 * males_in_orchestra) / 3 :=
by sorry

end NUMINAMATH_CALUDE_orchestra_females_count_l141_14163


namespace NUMINAMATH_CALUDE_number_problem_l141_14151

theorem number_problem (x : ℝ) : (0.40 * x = 0.80 * 5 + 2) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l141_14151


namespace NUMINAMATH_CALUDE_roots_have_unit_modulus_l141_14162

theorem roots_have_unit_modulus (z : ℂ) :
  (11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_have_unit_modulus_l141_14162


namespace NUMINAMATH_CALUDE_min_value_of_expression_l141_14133

theorem min_value_of_expression (x y : ℝ) : 
  (x * y - 2)^2 + (x + y + 1)^2 ≥ 5 ∧ 
  ∃ (a b : ℝ), (a * b - 2)^2 + (a + b + 1)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l141_14133


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l141_14160

theorem yellow_marbles_count (yellow blue : ℕ) 
  (h1 : blue = yellow - 2)
  (h2 : yellow + blue = 240) : 
  yellow = 121 := by
sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l141_14160


namespace NUMINAMATH_CALUDE_cubic_coefficient_sum_l141_14177

theorem cubic_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) : 
  (∀ x : ℝ, (5*x + 4)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) → 
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_coefficient_sum_l141_14177


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l141_14198

theorem quadratic_equation_root (m : ℝ) (α : ℝ) :
  (∃ x : ℂ, x^2 + (1 - 2*I)*x + 3*m - I = 0) →
  (α^2 + (1 - 2*I)*α + 3*m - I = 0) →
  (∃ β : ℂ, β^2 + (1 - 2*I)*β + 3*m - I = 0 ∧ β ≠ α) →
  (β = -1/2 + 2*I) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l141_14198


namespace NUMINAMATH_CALUDE_textbook_reading_time_l141_14153

/-- Calculates the total reading time in hours for a textbook with given parameters. -/
def totalReadingTime (totalChapters : ℕ) (readingTimePerChapter : ℕ) : ℚ :=
  let chaptersRead := totalChapters - (totalChapters / 3)
  (chaptersRead * readingTimePerChapter : ℚ) / 60

/-- Proves that the total reading time for the given textbook is 7 hours. -/
theorem textbook_reading_time :
  totalReadingTime 31 20 = 7 := by
  sorry

#eval totalReadingTime 31 20

end NUMINAMATH_CALUDE_textbook_reading_time_l141_14153


namespace NUMINAMATH_CALUDE_sqrt_76_between_8_and_9_l141_14126

theorem sqrt_76_between_8_and_9 : 8 < Real.sqrt 76 ∧ Real.sqrt 76 < 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_76_between_8_and_9_l141_14126


namespace NUMINAMATH_CALUDE_equation_solution_l141_14142

theorem equation_solution :
  ∀ x : ℚ, (6 * x / (x + 4) - 2 / (x + 4) = 3 / (x + 4)) ↔ (x = 5 / 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l141_14142


namespace NUMINAMATH_CALUDE_problem_solution_l141_14185

theorem problem_solution : 
  ((-5) * (-7) + 20 / (-4) = 30) ∧ 
  ((1 / 9 + 1 / 6 - 1 / 4) * (-36) = -1) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l141_14185


namespace NUMINAMATH_CALUDE_matrix_transformation_l141_14120

theorem matrix_transformation (N : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : N.mulVec ![1, 2] = ![4, 1])
  (h2 : N.mulVec ![2, -3] = ![1, 4]) :
  N.mulVec ![7, -2] = ![(84:ℚ)/7, (81:ℚ)/7] := by
  sorry

end NUMINAMATH_CALUDE_matrix_transformation_l141_14120


namespace NUMINAMATH_CALUDE_max_sum_absolute_values_l141_14135

theorem max_sum_absolute_values (x y : ℝ) (h : 4 * x^2 + y^2 = 4) :
  ∃ (M : ℝ), M = 2 ∧ ∀ (a b : ℝ), 4 * a^2 + b^2 = 4 → |a| + |b| ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_absolute_values_l141_14135


namespace NUMINAMATH_CALUDE_gcd_2_powers_l141_14165

theorem gcd_2_powers : 
  Nat.gcd (2^2025 - 1) (2^2016 - 1) = 2^9 - 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_2_powers_l141_14165


namespace NUMINAMATH_CALUDE_odot_problem_l141_14106

-- Define the custom operation
def odot (a b : ℚ) : ℚ := a + (5 * a) / (3 * b)

-- State the theorem
theorem odot_problem : (odot 12 9) + 3 = 155 / 9 := by
  sorry

end NUMINAMATH_CALUDE_odot_problem_l141_14106


namespace NUMINAMATH_CALUDE_calculate_Y_l141_14104

theorem calculate_Y : ∀ A B Y : ℚ,
  A = 3081 / 4 →
  B = A * 2 →
  Y = A - B →
  Y = -770.25 := by
sorry

end NUMINAMATH_CALUDE_calculate_Y_l141_14104


namespace NUMINAMATH_CALUDE_sqrt_representation_condition_l141_14119

theorem sqrt_representation_condition (A B : ℚ) :
  (∃ x y : ℚ, ∀ (sign : Bool), 
    Real.sqrt (A + (-1)^(sign.toNat : ℕ) * Real.sqrt B) = 
    Real.sqrt x + (-1)^(sign.toNat : ℕ) * Real.sqrt y) 
  ↔ 
  ∃ k : ℚ, A^2 - B = k^2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_representation_condition_l141_14119


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l141_14129

theorem x_fourth_plus_inverse_fourth (x : ℝ) (h : x^2 - 15*x + 1 = 0) : 
  x^4 + 1/x^4 = 49727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l141_14129


namespace NUMINAMATH_CALUDE_johnny_earnings_l141_14189

/-- Calculates the total earnings from two jobs with overtime --/
def total_earnings (
  job1_rate : ℚ)
  (job1_hours : ℕ)
  (job1_regular_hours : ℕ)
  (job1_overtime_multiplier : ℚ)
  (job2_rate : ℚ)
  (job2_hours : ℕ) : ℚ :=
  let job1_regular_pay := job1_rate * (job1_regular_hours : ℚ)
  let job1_overtime_hours := job1_hours - job1_regular_hours
  let job1_overtime_rate := job1_rate * job1_overtime_multiplier
  let job1_overtime_pay := job1_overtime_rate * (job1_overtime_hours : ℚ)
  let job1_total_pay := job1_regular_pay + job1_overtime_pay
  let job2_pay := job2_rate * (job2_hours : ℚ)
  job1_total_pay + job2_pay

/-- Johnny's total earnings from two jobs with overtime --/
theorem johnny_earnings : 
  total_earnings 3.25 8 6 1.5 4.5 5 = 58.25 := by
  sorry

end NUMINAMATH_CALUDE_johnny_earnings_l141_14189


namespace NUMINAMATH_CALUDE_eds_cats_l141_14186

/-- Proves that Ed has 3 cats given the conditions of the problem -/
theorem eds_cats (dogs : ℕ) (cats : ℕ) (fish : ℕ) : 
  dogs = 2 → 
  fish = 2 * (cats + dogs) → 
  dogs + cats + fish = 15 → 
  cats = 3 := by
sorry

end NUMINAMATH_CALUDE_eds_cats_l141_14186


namespace NUMINAMATH_CALUDE_factorial_ratio_simplification_l141_14179

theorem factorial_ratio_simplification (N : ℕ) :
  (Nat.factorial N * (N + 2)) / Nat.factorial (N + 3) = 1 / ((N + 3) * (N + 1)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_ratio_simplification_l141_14179


namespace NUMINAMATH_CALUDE_cricketer_average_score_l141_14175

theorem cricketer_average_score 
  (initial_innings : ℕ) 
  (last_inning_score : ℕ) 
  (average_increase : ℕ) 
  (h1 : initial_innings = 18) 
  (h2 : last_inning_score = 95) 
  (h3 : average_increase = 4) :
  (initial_innings * (average_increase + (last_inning_score / (initial_innings + 1))) + last_inning_score) / (initial_innings + 1) = 23 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l141_14175


namespace NUMINAMATH_CALUDE_roots_equal_opposite_signs_l141_14143

theorem roots_equal_opposite_signs (c d e : ℝ) :
  (∃ y₁ y₂ : ℝ, y₁ = -y₂ ∧ y₁ ≠ 0 ∧
    (y₁^2 - d*y₁) / (c*y₁ - e) = (n - 2) / (n + 2) ∧
    (y₂^2 - d*y₂) / (c*y₂ - e) = (n - 2) / (n + 2)) →
  n = -2 :=
by sorry

end NUMINAMATH_CALUDE_roots_equal_opposite_signs_l141_14143


namespace NUMINAMATH_CALUDE_dataset_reduction_fraction_l141_14105

theorem dataset_reduction_fraction (initial : ℕ) (increase_percent : ℚ) (final : ℕ) : 
  initial = 200 →
  increase_percent = 1/5 →
  final = 180 →
  (initial + initial * increase_percent - final) / (initial + initial * increase_percent) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_dataset_reduction_fraction_l141_14105
