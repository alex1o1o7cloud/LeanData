import Mathlib

namespace NUMINAMATH_CALUDE_bottom_is_red_l36_3633

/-- Represents the colors of the squares -/
inductive Color
  | R | B | O | Y | G | W | P

/-- Represents a face of the cube -/
structure Face where
  color : Color

/-- Represents the cube configuration -/
structure Cube where
  top : Face
  bottom : Face
  sides : List Face
  outward : Face

/-- Theorem: Given the cube configuration, the bottom face is Red -/
theorem bottom_is_red (cube : Cube)
  (h1 : cube.top.color = Color.W)
  (h2 : cube.outward.color = Color.P)
  (h3 : cube.sides.length = 4)
  (h4 : ∀ c : Color, c ≠ Color.P → c ∈ (cube.top :: cube.bottom :: cube.sides).map Face.color) :
  cube.bottom.color = Color.R :=
sorry

end NUMINAMATH_CALUDE_bottom_is_red_l36_3633


namespace NUMINAMATH_CALUDE_total_jellybeans_l36_3650

/-- The number of jellybeans in a bag with black, green, and orange beans. -/
def jellybean_count (black green orange : ℕ) : ℕ :=
  black + green + orange

/-- Theorem stating the total number of jellybeans in the bag -/
theorem total_jellybeans :
  ∃ (black green orange : ℕ),
    black = 8 ∧
    green = black + 2 ∧
    orange = green - 1 ∧
    jellybean_count black green orange = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_jellybeans_l36_3650


namespace NUMINAMATH_CALUDE_cube_difference_equality_l36_3642

theorem cube_difference_equality : 
  - (666 : ℤ)^3 + (555 : ℤ)^3 = ((666 : ℤ)^2 - 666 * 555 + (555 : ℤ)^2) * (-124072470) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_equality_l36_3642


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l36_3641

theorem quadratic_equation_solutions (b c x₁ x₂ : ℝ) : 
  (x₁^2 + b*x₁ + c = 0) →
  (x₂^2 + b*x₂ + c = 0) →
  (x₁ ≠ x₂) →
  (|x₁ - x₂| = 1) →
  (|b - c| = 1) →
  ((b = -1 ∧ c = 0) ∨ (b = 5 ∧ c = 6) ∨ (b = 1 ∧ c = 0) ∨ (b = 3 ∧ c = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l36_3641


namespace NUMINAMATH_CALUDE_max_consecutive_interesting_l36_3674

/-- A positive integer is interesting if it is a product of two prime numbers -/
def IsInteresting (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p * q

/-- The maximum number of consecutive interesting positive integers -/
theorem max_consecutive_interesting : 
  (∃ k : ℕ, k > 0 ∧ ∀ i : ℕ, i < k → IsInteresting (i + 1)) ∧ 
  (∀ k : ℕ, k > 3 → ∃ i : ℕ, i < k ∧ ¬IsInteresting (i + 1)) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_interesting_l36_3674


namespace NUMINAMATH_CALUDE_expression_equals_8_175_l36_3623

-- Define the expression
def expression : ℝ := (4.5 - 1.23) * 2.5

-- State the theorem
theorem expression_equals_8_175 : expression = 8.175 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_8_175_l36_3623


namespace NUMINAMATH_CALUDE_speed_ratio_l36_3648

/-- The speed of runner A in meters per hour -/
def speed_A : ℝ := sorry

/-- The speed of runner B in meters per hour -/
def speed_B : ℝ := sorry

/-- The length of the track in meters -/
def track_length : ℝ := sorry

/-- When running in the same direction, A catches up with B after 3 hours -/
axiom same_direction : 3 * (speed_A - speed_B) = track_length

/-- When running in opposite directions, A and B meet after 2 hours -/
axiom opposite_direction : 2 * (speed_A + speed_B) = track_length

/-- The ratio of A's speed to B's speed is 5:1 -/
theorem speed_ratio : speed_A / speed_B = 5 := by sorry

end NUMINAMATH_CALUDE_speed_ratio_l36_3648


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l36_3695

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The specific function f(x) = x^2 + (a-1)x + a -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  x^2 + (a - 1) * x + a

theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, IsEven (f a) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l36_3695


namespace NUMINAMATH_CALUDE_increasing_quadratic_iff_l36_3659

/-- A function f is increasing on an interval [x0, +∞) if for all x, y in the interval with x < y, f(x) < f(y) -/
def IncreasingOn (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  ∀ x y, x0 ≤ x → x < y → f x < f y

/-- The quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem increasing_quadratic_iff (a : ℝ) :
  IncreasingOn (f a) 4 ↔ a ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_increasing_quadratic_iff_l36_3659


namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l36_3669

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The radius of the Sun in meters -/
def sun_radius : ℝ := 696000000

/-- Converts a real number to scientific notation -/
noncomputable def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem sun_radius_scientific_notation :
  to_scientific_notation sun_radius = ScientificNotation.mk 6.96 8 sorry := by
  sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l36_3669


namespace NUMINAMATH_CALUDE_sqrt_sum_zero_implies_y_minus_x_l36_3666

theorem sqrt_sum_zero_implies_y_minus_x (x y : ℝ) :
  Real.sqrt (2 * x + y) + Real.sqrt (x^2 - 9) = 0 →
  (y - x = -9 ∨ y - x = 9) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_zero_implies_y_minus_x_l36_3666


namespace NUMINAMATH_CALUDE_determinant_inequality_l36_3691

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_inequality (x : ℝ) :
  det2x2 7 (x^2) 2 1 > det2x2 3 (-2) 1 x ↔ -5/2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_determinant_inequality_l36_3691


namespace NUMINAMATH_CALUDE_fly_probabilities_l36_3634

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def fly_probability (n m : ℕ) : ℚ :=
  (binomial (n + m) n : ℚ) / (2 ^ (n + m))

def fly_probability_through_segment (n1 m1 n2 m2 : ℕ) : ℚ :=
  ((binomial (n1 + m1) n1 : ℚ) * (binomial (n2 + m2) n2)) / (2 ^ (n1 + m1 + n2 + m2 + 1))

def fly_probability_through_circle (n m r : ℕ) : ℚ :=
  let total_steps := n + m
  let mid_steps := total_steps / 2
  (2 * (binomial mid_steps 2 : ℚ) * (binomial mid_steps (mid_steps - 2)) +
   2 * (binomial mid_steps 3 : ℚ) * (binomial mid_steps (mid_steps - 3)) +
   (binomial mid_steps 4 : ℚ) * (binomial mid_steps (mid_steps - 4))) /
  (2 ^ total_steps)

theorem fly_probabilities :
  fly_probability 8 10 = (binomial 18 8 : ℚ) / (2^18) ∧
  fly_probability_through_segment 5 6 2 4 = ((binomial 11 5 : ℚ) * (binomial 6 2)) / (2^18) ∧
  fly_probability_through_circle 8 10 3 = 
    (2 * (binomial 9 2 : ℚ) * (binomial 9 6) + 
     2 * (binomial 9 3 : ℚ) * (binomial 9 5) + 
     (binomial 9 4 : ℚ) * (binomial 9 4)) / (2^18) := by
  sorry

end NUMINAMATH_CALUDE_fly_probabilities_l36_3634


namespace NUMINAMATH_CALUDE_parallel_vectors_characterization_l36_3685

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  x₁ * y₂ - x₂ * y₁ = 0

/-- The proposed condition for parallel vectors -/
def proposed_condition (a b : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  x₁ * y₂ = x₂ * y₁

theorem parallel_vectors_characterization (a b : ℝ × ℝ) :
  (are_parallel a b ↔ proposed_condition a b) ∧
  (∃ a b : ℝ × ℝ, are_parallel a b ≠ proposed_condition a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_characterization_l36_3685


namespace NUMINAMATH_CALUDE_reciprocal_equal_reciprocal_equal_opposite_sign_l36_3605

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem for numbers equal to their own reciprocal
theorem reciprocal_equal (x : ℝ) : x = 1 / x ↔ x = 1 ∨ x = -1 := by sorry

-- Theorem for numbers equal to their own reciprocal with opposite sign
theorem reciprocal_equal_opposite_sign (y : ℂ) : y = -1 / y ↔ y = i ∨ y = -i := by sorry

end NUMINAMATH_CALUDE_reciprocal_equal_reciprocal_equal_opposite_sign_l36_3605


namespace NUMINAMATH_CALUDE_cube_root_of_eight_l36_3693

theorem cube_root_of_eight (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_eight_l36_3693


namespace NUMINAMATH_CALUDE_trig_simplification_l36_3621

/-- Proves that 1/cos(80°) - √3/sin(80°) = 4 --/
theorem trig_simplification : 
  1 / Real.cos (80 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l36_3621


namespace NUMINAMATH_CALUDE_predicted_weight_approx_l36_3668

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 0.849 * x - 85.712

-- Define the height of the student
def student_height : ℝ := 172

-- Define the tolerance for "approximately" (e.g., within 0.001)
def tolerance : ℝ := 0.001

-- Theorem statement
theorem predicted_weight_approx :
  ∃ (predicted_weight : ℝ), 
    regression_equation student_height = predicted_weight ∧ 
    abs (predicted_weight - 60.316) < tolerance := by
  sorry

end NUMINAMATH_CALUDE_predicted_weight_approx_l36_3668


namespace NUMINAMATH_CALUDE_right_triangle_area_l36_3660

/-- The area of a right triangle with hypotenuse 5√2 and one leg 5 is 12.5 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) 
  (h2 : c = 5 * Real.sqrt 2) (h3 : a = 5) : (1/2) * a * b = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l36_3660


namespace NUMINAMATH_CALUDE_total_cds_l36_3687

theorem total_cds (a b : ℕ) : 
  (b + 6 = 2 * (a - 6)) →
  (a + 6 = b - 6) →
  a + b = 72 := by
sorry

end NUMINAMATH_CALUDE_total_cds_l36_3687


namespace NUMINAMATH_CALUDE_geometric_sequence_product_bound_l36_3617

theorem geometric_sequence_product_bound (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_seq : (4 * a^2 + b^2)^2 = a * b) : a * b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_bound_l36_3617


namespace NUMINAMATH_CALUDE_max_salad_servings_is_56_l36_3614

/-- Represents the ingredients required for one serving of salad -/
structure SaladServing where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Represents the available ingredients in the restaurant's warehouse -/
structure WarehouseStock where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Calculates the maximum number of salad servings that can be made -/
def maxSaladServings (serving : SaladServing) (stock : WarehouseStock) : ℕ :=
  min
    (stock.cucumbers / serving.cucumbers)
    (min
      (stock.tomatoes / serving.tomatoes)
      (min
        (stock.brynza / serving.brynza)
        (stock.peppers / serving.peppers)))

/-- Theorem stating that the maximum number of salad servings is 56 -/
theorem max_salad_servings_is_56 :
  let serving := SaladServing.mk 2 2 75 1
  let stock := WarehouseStock.mk 117 116 4200 60
  maxSaladServings serving stock = 56 := by
  sorry

#eval maxSaladServings (SaladServing.mk 2 2 75 1) (WarehouseStock.mk 117 116 4200 60)

end NUMINAMATH_CALUDE_max_salad_servings_is_56_l36_3614


namespace NUMINAMATH_CALUDE_magnitude_relationship_l36_3619

noncomputable def a : ℝ := Real.sqrt 5 + 2
noncomputable def b : ℝ := 2 - Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 5 - 2

theorem magnitude_relationship : a > c ∧ c > b :=
by sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l36_3619


namespace NUMINAMATH_CALUDE_max_value_quadratic_l36_3652

theorem max_value_quadratic (p q : ℝ) : 
  q = p - 2 → 
  ∃ (max : ℝ), max = 26 + 2/3 ∧ 
  ∀ (p : ℝ), -3 * p^2 + 24 * p - 50 + 10 * q ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l36_3652


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l36_3694

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l36_3694


namespace NUMINAMATH_CALUDE_johns_weekly_sleep_l36_3610

/-- Calculates the total sleep John got in a week given the specified conditions --/
def johnsTotalSleep (daysInWeek : ℕ) (shortSleepDays : ℕ) (shortSleepHours : ℝ) 
  (recommendedSleep : ℝ) (percentOfRecommended : ℝ) : ℝ :=
  let normalSleepDays := daysInWeek - shortSleepDays
  let normalSleepHours := recommendedSleep * percentOfRecommended
  shortSleepDays * shortSleepHours + normalSleepDays * normalSleepHours

/-- Theorem stating that John's total sleep for the week equals 30 hours --/
theorem johns_weekly_sleep :
  johnsTotalSleep 7 2 3 8 0.6 = 30 := by
  sorry

#eval johnsTotalSleep 7 2 3 8 0.6

end NUMINAMATH_CALUDE_johns_weekly_sleep_l36_3610


namespace NUMINAMATH_CALUDE_min_value_theorem_l36_3628

theorem min_value_theorem (x y z w : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w))) + 
  (1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w))) ≥ 2 ∧
  (1 / ((1 - 0) * (1 - 0) * (1 - 0) * (1 - 0))) + 
  (1 / ((1 + 0) * (1 + 0) * (1 + 0) * (1 + 0))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l36_3628


namespace NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l36_3606

theorem max_stamps_with_50_dollars (stamp_price : ℚ) (total_money : ℚ) :
  stamp_price = 25 / 100 →
  total_money = 50 →
  ⌊total_money / stamp_price⌋ = 200 := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l36_3606


namespace NUMINAMATH_CALUDE_oranges_in_box_l36_3643

/-- Given an initial number of oranges in a box and a number of oranges added,
    the final number of oranges in the box is equal to the sum of the initial number and the added number. -/
theorem oranges_in_box (initial : ℝ) (added : ℝ) :
  initial + added = 90 :=
by sorry

end NUMINAMATH_CALUDE_oranges_in_box_l36_3643


namespace NUMINAMATH_CALUDE_range_of_a_l36_3618

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ 2 * x * (3 * x + a) < 1) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l36_3618


namespace NUMINAMATH_CALUDE_village_population_percentage_l36_3611

theorem village_population_percentage :
  let total_population : ℕ := 24000
  let part_population : ℕ := 23040
  let percentage : ℚ := (part_population : ℚ) / total_population * 100
  percentage = 96 := by
  sorry

end NUMINAMATH_CALUDE_village_population_percentage_l36_3611


namespace NUMINAMATH_CALUDE_min_value_shifted_function_l36_3692

def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + 5 - c

theorem min_value_shifted_function (c : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f c x ≥ m ∧ ∃ (x₀ : ℝ), f c x₀ = m) ∧
  (∀ (x : ℝ), f c x ≥ 2) ∧
  (∃ (x₁ : ℝ), f c x₁ = 2) →
  (∃ (m : ℝ), ∀ (x : ℝ), f c (x - 3) ≥ m ∧ ∃ (x₀ : ℝ), f c (x₀ - 3) = m) ∧
  (∀ (x : ℝ), f c (x - 3) ≥ 2) ∧
  (∃ (x₁ : ℝ), f c (x₁ - 3) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_shifted_function_l36_3692


namespace NUMINAMATH_CALUDE_tan_product_equals_15_l36_3654

theorem tan_product_equals_15 : 
  15 * Real.tan (44 * π / 180) * Real.tan (45 * π / 180) * Real.tan (46 * π / 180) = 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_equals_15_l36_3654


namespace NUMINAMATH_CALUDE_inequality_proof_l36_3697

theorem inequality_proof (x a : ℝ) (hx : x > 0) (ha : a > 0) :
  (1 / Real.sqrt (x + 1)) + (1 / Real.sqrt (a + 1)) + Real.sqrt (a * x / (a * x + 8)) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l36_3697


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l36_3609

/-- The trajectory of point P given vertices A and B and slope product condition -/
theorem trajectory_of_point_P (x y : ℝ) :
  let A := (0, -Real.sqrt 2)
  let B := (0, Real.sqrt 2)
  let slope_PA := (y - A.2) / (x - A.1)
  let slope_PB := (y - B.2) / (x - B.1)
  x ≠ 0 →
  slope_PA * slope_PB = -2 →
  y^2 / 2 + x^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l36_3609


namespace NUMINAMATH_CALUDE_total_fruits_is_213_l36_3680

/-- Represents a fruit grower -/
structure FruitGrower where
  watermelons : ℕ
  pineapples : ℕ

/-- Calculates the total fruits grown by a single grower -/
def totalFruits (grower : FruitGrower) : ℕ :=
  grower.watermelons + grower.pineapples

/-- Represents the group of fruit growers -/
def fruitGrowers : List FruitGrower :=
  [{ watermelons := 37, pineapples := 56 },  -- Jason
   { watermelons := 68, pineapples := 27 },  -- Mark
   { watermelons := 11, pineapples := 14 }]  -- Sandy

/-- Theorem: The total number of fruits grown by the group is 213 -/
theorem total_fruits_is_213 : 
  (fruitGrowers.map totalFruits).sum = 213 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_is_213_l36_3680


namespace NUMINAMATH_CALUDE_dumplings_eaten_l36_3670

theorem dumplings_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 14 → remaining = 7 → eaten = initial - remaining :=
by sorry

end NUMINAMATH_CALUDE_dumplings_eaten_l36_3670


namespace NUMINAMATH_CALUDE_smallest_divisor_with_remainder_fifteen_satisfies_condition_fifteen_is_smallest_l36_3657

theorem smallest_divisor_with_remainder (d : ℕ) : d > 0 ∧ 2021 % d = 11 → d ≥ 15 := by
  sorry

theorem fifteen_satisfies_condition : 2021 % 15 = 11 := by
  sorry

theorem fifteen_is_smallest : ∀ d : ℕ, d > 0 ∧ 2021 % d = 11 → d ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_with_remainder_fifteen_satisfies_condition_fifteen_is_smallest_l36_3657


namespace NUMINAMATH_CALUDE_flower_pot_price_difference_l36_3699

theorem flower_pot_price_difference 
  (n : ℕ) 
  (total_cost : ℚ) 
  (largest_pot_cost : ℚ) 
  (h1 : n = 6) 
  (h2 : total_cost = 39/5) 
  (h3 : largest_pot_cost = 77/40) : 
  ∃ (d : ℚ), d = 1/4 ∧ 
  ∃ (x : ℚ), 
    (x + (n - 1) * d = largest_pot_cost) ∧ 
    (n * x + (n * (n - 1) / 2) * d = total_cost) :=
by sorry

end NUMINAMATH_CALUDE_flower_pot_price_difference_l36_3699


namespace NUMINAMATH_CALUDE_vector_problem_l36_3679

-- Define the vectors a and b
def a (m : ℝ) : Fin 2 → ℝ := ![m, 2]
def b : Fin 2 → ℝ := ![2, -3]

-- Define the parallel condition
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v 0 * w 1 = k * v 1 * w 0

-- State the theorem
theorem vector_problem (m : ℝ) :
  are_parallel (a m + b) (a m - b) → m = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l36_3679


namespace NUMINAMATH_CALUDE_work_completion_time_proportional_aartis_triple_work_time_l36_3645

/-- If a person can complete a piece of work in a certain number of days,
    then the time to complete a multiple of that work is proportional. -/
theorem work_completion_time_proportional
  (days_for_single_work : ℕ) (work_multiple : ℕ) :
  let days_for_multiple_work := days_for_single_work * work_multiple
  days_for_multiple_work = days_for_single_work * work_multiple :=
by sorry

/-- Aarti's work completion time for triple work -/
theorem aartis_triple_work_time :
  let days_for_single_work := 9
  let work_multiple := 3
  let days_for_triple_work := days_for_single_work * work_multiple
  days_for_triple_work = 27 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_proportional_aartis_triple_work_time_l36_3645


namespace NUMINAMATH_CALUDE_min_distance_ellipse_line_l36_3646

def ellipse_C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

def point_on_ellipse (x y : ℝ) : Prop := ellipse_C x y

def point_A (y : ℝ) : ℝ × ℝ := (4, y)

def point_B (m n : ℝ) : ℝ × ℝ := (m, n)

def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

theorem min_distance_ellipse_line :
  ∃ (y m n : ℝ),
    point_on_ellipse 0 (Real.sqrt 5) ∧
    point_on_ellipse m n ∧
    perpendicular (point_A y) (point_B m n) ∧
    ∀ (y' m' n' : ℝ),
      point_on_ellipse m' n' →
      perpendicular (point_A y') (point_B m' n') →
      (m - 4)^2 + (n - y)^2 ≤ (m' - 4)^2 + (n' - y')^2 ∧
      (m - 4)^2 + (n - y)^2 = 21 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_line_l36_3646


namespace NUMINAMATH_CALUDE_vasya_no_purchase_days_l36_3683

theorem vasya_no_purchase_days :
  ∀ (x y z w : ℕ),
    x + y + z + w = 15 →  -- Total school days
    9 * x + 4 * z = 30 →  -- Total marshmallows bought
    2 * y + z = 9 →       -- Total meat pies bought
    w = 7 :=              -- Days with no purchase
by
  sorry

end NUMINAMATH_CALUDE_vasya_no_purchase_days_l36_3683


namespace NUMINAMATH_CALUDE_real_y_condition_l36_3662

theorem real_y_condition (x y : ℝ) : 
  (4 * y^2 + 6 * x * y + x + 8 = 0) → 
  (∃ y : ℝ, 4 * y^2 + 6 * x * y + x + 8 = 0) ↔ (x ≤ -8/9 ∨ x ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l36_3662


namespace NUMINAMATH_CALUDE_jacob_insects_compared_to_dean_l36_3639

theorem jacob_insects_compared_to_dean :
  ∀ (angela_insects jacob_insects dean_insects : ℕ),
    angela_insects = 75 →
    dean_insects = 30 →
    angela_insects * 2 = jacob_insects →
    jacob_insects / dean_insects = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_jacob_insects_compared_to_dean_l36_3639


namespace NUMINAMATH_CALUDE_C_share_approx_l36_3649

-- Define the total rent
def total_rent : ℚ := 225

-- Define the number of oxen and months for each person
def oxen_A : ℕ := 10
def months_A : ℕ := 7
def oxen_B : ℕ := 12
def months_B : ℕ := 5
def oxen_C : ℕ := 15
def months_C : ℕ := 3
def oxen_D : ℕ := 20
def months_D : ℕ := 6

-- Calculate oxen-months for each person
def oxen_months_A : ℕ := oxen_A * months_A
def oxen_months_B : ℕ := oxen_B * months_B
def oxen_months_C : ℕ := oxen_C * months_C
def oxen_months_D : ℕ := oxen_D * months_D

-- Calculate total oxen-months
def total_oxen_months : ℕ := oxen_months_A + oxen_months_B + oxen_months_C + oxen_months_D

-- Calculate C's share of the rent
def C_share : ℚ := total_rent * (oxen_months_C : ℚ) / (total_oxen_months : ℚ)

-- Theorem to prove
theorem C_share_approx : ∃ ε > 0, abs (C_share - 34.32) < ε :=
sorry

end NUMINAMATH_CALUDE_C_share_approx_l36_3649


namespace NUMINAMATH_CALUDE_harvard_attendance_percentage_l36_3653

theorem harvard_attendance_percentage 
  (total_applicants : ℕ) 
  (acceptance_rate : ℚ)
  (other_schools_rate : ℚ)
  (attending_students : ℕ) :
  total_applicants = 20000 →
  acceptance_rate = 5 / 100 →
  other_schools_rate = 1 / 10 →
  attending_students = 900 →
  (attending_students : ℚ) / (total_applicants * acceptance_rate) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_harvard_attendance_percentage_l36_3653


namespace NUMINAMATH_CALUDE_reporters_covering_local_politics_l36_3626

theorem reporters_covering_local_politics
  (total_reporters : ℕ)
  (h1 : total_reporters > 0)
  (percent_not_covering_politics : ℚ)
  (h2 : percent_not_covering_politics = 1/2)
  (percent_not_covering_local_politics : ℚ)
  (h3 : percent_not_covering_local_politics = 3/10)
  : (↑total_reporters - (percent_not_covering_politics * ↑total_reporters) -
     (percent_not_covering_local_politics * (↑total_reporters - (percent_not_covering_politics * ↑total_reporters))))
    / ↑total_reporters = 7/20 :=
by sorry

end NUMINAMATH_CALUDE_reporters_covering_local_politics_l36_3626


namespace NUMINAMATH_CALUDE_special_gp_ratio_equation_ratio_approx_value_l36_3622

/-- A geometric progression with positive terms where any term is equal to the sum of the next three following terms -/
structure SpecialGeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

/-- The common ratio of a special geometric progression satisfies a specific equation -/
theorem special_gp_ratio_equation (gp : SpecialGeometricProgression) :
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 :=
sorry

/-- The positive real solution to the equation x^3 + x^2 + x - 1 = 0 is approximately 0.543689 -/
theorem ratio_approx_value :
  ∃ x : ℝ, x > 0 ∧ x^3 + x^2 + x - 1 = 0 ∧ abs (x - 0.543689) < 0.000001 :=
sorry

end NUMINAMATH_CALUDE_special_gp_ratio_equation_ratio_approx_value_l36_3622


namespace NUMINAMATH_CALUDE_division_and_subtraction_l36_3601

theorem division_and_subtraction : (12 / (1/6)) - (1/3) = 215/3 := by
  sorry

end NUMINAMATH_CALUDE_division_and_subtraction_l36_3601


namespace NUMINAMATH_CALUDE_residue_modulo_17_l36_3637

theorem residue_modulo_17 : (101 * 15 - 7 * 9 + 5) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_residue_modulo_17_l36_3637


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_one_l36_3689

theorem simplify_and_evaluate (m : ℝ) (h1 : m ≠ -3) (h2 : m ≠ 3) (h3 : m ≠ 0) :
  (m / (m + 3) - 2 * m / (m - 3)) / (m / (m^2 - 9)) = -m - 9 :=
by sorry

-- Evaluation at m = 1
theorem evaluate_at_one :
  (1 / (1 + 3) - 2 * 1 / (1 - 3)) / (1 / (1^2 - 9)) = -10 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_one_l36_3689


namespace NUMINAMATH_CALUDE_quadratic_exponent_implies_m_eq_two_l36_3684

/-- A function is quadratic if it can be expressed as ax² + bx + c, where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The main theorem: If y = (m+2)x^(m²-2) is quadratic, then m = 2 -/
theorem quadratic_exponent_implies_m_eq_two (m : ℝ) :
  IsQuadratic (fun x ↦ (m + 2) * x^(m^2 - 2)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_exponent_implies_m_eq_two_l36_3684


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l36_3690

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (b > a ∧ a > 0 → (1 : ℝ) / a^2 > (1 : ℝ) / b^2) ∧
  ∃ a b : ℝ, (1 : ℝ) / a^2 > (1 : ℝ) / b^2 ∧ ¬(b > a ∧ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l36_3690


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l36_3607

theorem stratified_sampling_proportion (total_male : ℕ) (total_female : ℕ) (selected_male : ℕ) :
  total_male = 56 →
  total_female = 42 →
  selected_male = 8 →
  ∃ (selected_female : ℕ),
    selected_female = 6 ∧
    (selected_male : ℚ) / total_male = (selected_female : ℚ) / total_female :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l36_3607


namespace NUMINAMATH_CALUDE_negation_existential_proposition_l36_3667

theorem negation_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_proposition_l36_3667


namespace NUMINAMATH_CALUDE_stratified_sample_size_l36_3665

/-- Represents the proportion of a population group -/
structure PopulationProportion where
  value : ℚ
  nonneg : 0 ≤ value

/-- Represents a stratified sample -/
structure StratifiedSample where
  total_size : ℕ
  middle_aged_size : ℕ
  middle_aged_size_le_total : middle_aged_size ≤ total_size

/-- Given population proportions and a stratified sample, proves the total sample size -/
theorem stratified_sample_size 
  (elderly : PopulationProportion)
  (middle_aged : PopulationProportion)
  (young : PopulationProportion)
  (sample : StratifiedSample)
  (h1 : elderly.value + middle_aged.value + young.value = 1)
  (h2 : elderly.value = 2 / 10)
  (h3 : middle_aged.value = 3 / 10)
  (h4 : young.value = 5 / 10)
  (h5 : sample.middle_aged_size = 12) :
  sample.total_size = 40 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l36_3665


namespace NUMINAMATH_CALUDE_painting_time_l36_3696

theorem painting_time (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : 
  total_rooms = 10 → time_per_room = 8 → painted_rooms = 8 → 
  (total_rooms - painted_rooms) * time_per_room = 16 := by
sorry

end NUMINAMATH_CALUDE_painting_time_l36_3696


namespace NUMINAMATH_CALUDE_equation_has_real_root_l36_3677

theorem equation_has_real_root (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 2) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l36_3677


namespace NUMINAMATH_CALUDE_ball_count_l36_3600

theorem ball_count (blue_count : ℕ) (prob_blue : ℚ) (green_count : ℕ) : 
  blue_count = 8 → 
  prob_blue = 1 / 5 → 
  prob_blue = blue_count / (blue_count + green_count) →
  green_count = 32 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_l36_3600


namespace NUMINAMATH_CALUDE_greatest_common_length_l36_3663

theorem greatest_common_length (a b c : ℕ) (ha : a = 48) (hb : b = 64) (hc : c = 80) :
  Nat.gcd a (Nat.gcd b c) = 16 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_length_l36_3663


namespace NUMINAMATH_CALUDE_cosine_of_point_on_terminal_side_l36_3615

def point_on_terminal_side (α : Real) (x y : Real) : Prop :=
  ∃ t : Real, t > 0 ∧ x = t * Real.cos α ∧ y = t * Real.sin α

theorem cosine_of_point_on_terminal_side (α : Real) :
  point_on_terminal_side α (-3) 4 → Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_point_on_terminal_side_l36_3615


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l36_3613

theorem hot_dogs_remainder (total : Nat) (package_size : Nat) : 
  total = 25197624 → package_size = 4 → total % package_size = 0 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l36_3613


namespace NUMINAMATH_CALUDE_dhoni_leftover_earnings_l36_3671

theorem dhoni_leftover_earnings (total_earnings rent_percentage dishwasher_discount : ℝ) :
  rent_percentage = 40 →
  dishwasher_discount = 20 →
  let dishwasher_percentage := rent_percentage - (dishwasher_discount / 100) * rent_percentage
  let total_spent_percentage := rent_percentage + dishwasher_percentage
  let leftover_percentage := 100 - total_spent_percentage
  leftover_percentage = 28 :=
by sorry

end NUMINAMATH_CALUDE_dhoni_leftover_earnings_l36_3671


namespace NUMINAMATH_CALUDE_fraction_equality_l36_3632

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 1 / 3) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l36_3632


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l36_3681

/-- Proves that for a rectangular plot with given conditions, the length is 20 meters more than the breadth. -/
theorem rectangular_plot_length_difference (length breadth : ℝ) : 
  length = 60 ∧ 
  length > breadth ∧ 
  2 * (length + breadth) * 26.5 = 5300 → 
  length - breadth = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l36_3681


namespace NUMINAMATH_CALUDE_parabola_symmetry_l36_3612

-- Define the parabolas
def C₁ (x : ℝ) : ℝ := x^2 - 2*x + 3
def C₂ (x : ℝ) : ℝ := C₁ (x + 1)
def C₃ (x : ℝ) : ℝ := C₂ (-x)

-- State the theorem
theorem parabola_symmetry :
  ∀ x : ℝ, C₃ x = x^2 + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l36_3612


namespace NUMINAMATH_CALUDE_jellybean_ratio_l36_3661

/-- Prove that the ratio of Matilda's jellybeans to Matt's jellybeans is 1:2 -/
theorem jellybean_ratio :
  let steve_jellybeans : ℕ := 84
  let matt_jellybeans : ℕ := 10 * steve_jellybeans
  let matilda_jellybeans : ℕ := 420
  (matilda_jellybeans : ℚ) / (matt_jellybeans : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_ratio_l36_3661


namespace NUMINAMATH_CALUDE_min_value_of_f_l36_3672

-- Define the function f(x)
def f (x : ℝ) : ℝ := 27 * x - x^3

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) 2 ∧
  (∀ y ∈ Set.Icc (-4 : ℝ) 2, f y ≥ f x) ∧
  f x = -54 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l36_3672


namespace NUMINAMATH_CALUDE_min_value_of_a_l36_3673

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

-- Define the theorem
theorem min_value_of_a (h_a : ℝ) (h_a_pos : h_a > 0) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ Set.Icc (-1 : ℝ) 2, f x₁ = g h_a x₂) → 
  h_a ≥ 3 := by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_min_value_of_a_l36_3673


namespace NUMINAMATH_CALUDE_floral_shop_sales_theorem_l36_3678

/-- Represents the sales and prices of bouquets for a floral shop over three days -/
structure FloralShopSales where
  /-- Number of rose bouquets sold on Monday -/
  rose_monday : ℕ
  /-- Number of lily bouquets sold on Monday -/
  lily_monday : ℕ
  /-- Number of orchid bouquets sold on Monday -/
  orchid_monday : ℕ
  /-- Price of rose bouquets on Monday -/
  rose_price_monday : ℕ
  /-- Price of lily bouquets on Monday -/
  lily_price_monday : ℕ
  /-- Price of orchid bouquets on Monday -/
  orchid_price_monday : ℕ
  /-- Price of rose bouquets on Tuesday -/
  rose_price_tuesday : ℕ
  /-- Price of lily bouquets on Tuesday -/
  lily_price_tuesday : ℕ
  /-- Price of orchid bouquets on Tuesday -/
  orchid_price_tuesday : ℕ
  /-- Price of rose bouquets on Wednesday -/
  rose_price_wednesday : ℕ
  /-- Price of lily bouquets on Wednesday -/
  lily_price_wednesday : ℕ
  /-- Price of orchid bouquets on Wednesday -/
  orchid_price_wednesday : ℕ

/-- Calculates the total number and value of bouquets sold over three days -/
def calculate_total_sales (sales : FloralShopSales) : ℕ × ℕ :=
  let rose_tuesday := 3 * sales.rose_monday
  let lily_tuesday := 2 * sales.lily_monday
  let orchid_tuesday := sales.orchid_monday / 2
  let rose_wednesday := rose_tuesday / 3
  let lily_wednesday := lily_tuesday / 4
  let orchid_wednesday := (2 * orchid_tuesday) / 3
  
  let total_roses := sales.rose_monday + rose_tuesday + rose_wednesday
  let total_lilies := sales.lily_monday + lily_tuesday + lily_wednesday
  let total_orchids := sales.orchid_monday + orchid_tuesday + orchid_wednesday
  
  let total_bouquets := total_roses + total_lilies + total_orchids
  
  let rose_value := sales.rose_monday * sales.rose_price_monday + 
                    rose_tuesday * sales.rose_price_tuesday + 
                    rose_wednesday * sales.rose_price_wednesday
  let lily_value := sales.lily_monday * sales.lily_price_monday + 
                    lily_tuesday * sales.lily_price_tuesday + 
                    lily_wednesday * sales.lily_price_wednesday
  let orchid_value := sales.orchid_monday * sales.orchid_price_monday + 
                      orchid_tuesday * sales.orchid_price_tuesday + 
                      orchid_wednesday * sales.orchid_price_wednesday
  
  let total_value := rose_value + lily_value + orchid_value
  
  (total_bouquets, total_value)

theorem floral_shop_sales_theorem (sales : FloralShopSales) 
  (h1 : sales.rose_monday = 12)
  (h2 : sales.lily_monday = 8)
  (h3 : sales.orchid_monday = 6)
  (h4 : sales.rose_price_monday = 10)
  (h5 : sales.lily_price_monday = 15)
  (h6 : sales.orchid_price_monday = 20)
  (h7 : sales.rose_price_tuesday = 12)
  (h8 : sales.lily_price_tuesday = 18)
  (h9 : sales.orchid_price_tuesday = 22)
  (h10 : sales.rose_price_wednesday = 8)
  (h11 : sales.lily_price_wednesday = 12)
  (h12 : sales.orchid_price_wednesday = 16) :
  calculate_total_sales sales = (99, 1322) := by
  sorry


end NUMINAMATH_CALUDE_floral_shop_sales_theorem_l36_3678


namespace NUMINAMATH_CALUDE_neg_two_star_neg_one_l36_3635

/-- Custom binary operation ※ -/
def star (a b : ℤ) : ℤ := b^2 - a*b

/-- Theorem stating that (-2) ※ (-1) = -1 -/
theorem neg_two_star_neg_one : star (-2) (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_neg_two_star_neg_one_l36_3635


namespace NUMINAMATH_CALUDE_land_plot_side_length_l36_3647

theorem land_plot_side_length (area : ℝ) (side : ℝ) : 
  area = 1600 → side * side = area → side = 40 := by
  sorry

end NUMINAMATH_CALUDE_land_plot_side_length_l36_3647


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_line_not_perpendicular_no_perpendicular_line_perpendicular_intersection_perpendicular_l36_3676

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (on_plane : Point → Plane → Prop)
variable (on_line : Point → Line → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line → Prop)

-- Define the given conditions
variable (l : Line) (α β γ : Plane)
variable (h_diff : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Statement 1
theorem perpendicular_planes_parallel_line :
  perpendicular α β → ∃ m : Line, line_in_plane m α ∧ parallel m β := by sorry

-- Statement 2
theorem not_perpendicular_no_perpendicular_line :
  ¬perpendicular α β → ¬∃ m : Line, line_in_plane m α ∧ line_perpendicular_plane m β := by sorry

-- Statement 3
theorem perpendicular_intersection_perpendicular :
  perpendicular α γ → perpendicular β γ → intersection α β l → line_perpendicular_plane l γ := by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_line_not_perpendicular_no_perpendicular_line_perpendicular_intersection_perpendicular_l36_3676


namespace NUMINAMATH_CALUDE_range_of_m_l36_3640

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a^2 + b^2 = 1) (h2 : a^3 + b^3 + 1 = m * (a + b + 1)^3) :
  (3 * Real.sqrt 2 - 4) / 2 ≤ m ∧ m < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l36_3640


namespace NUMINAMATH_CALUDE_m_less_than_five_l36_3658

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem m_less_than_five
  (h_increasing : Increasing f)
  (h_inequality : ∀ m : ℝ, f (2 * m + 1) > f (3 * m - 4)) :
  ∀ m : ℝ, m < 5 := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_five_l36_3658


namespace NUMINAMATH_CALUDE_gold_coins_puzzle_l36_3631

theorem gold_coins_puzzle (n c : ℕ) 
  (h1 : n = 9 * (c - 2))  -- Condition 1: 9 coins per chest, 2 empty chests
  (h2 : n = 6 * c + 3)    -- Condition 2: 6 coins per chest, 3 coins leftover
  : n = 45 := by
  sorry

end NUMINAMATH_CALUDE_gold_coins_puzzle_l36_3631


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l36_3627

theorem simplify_sqrt_expression (y : ℝ) (h : y ≥ 5/2) :
  Real.sqrt (y + 2 + 3 * Real.sqrt (2 * y - 5)) - Real.sqrt (y - 2 + Real.sqrt (2 * y - 5)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l36_3627


namespace NUMINAMATH_CALUDE_quadratic_term_coefficient_and_constant_term_l36_3616

/-- Represents a quadratic equation in the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation -3x² - 2x = 0 -/
def givenEquation : QuadraticEquation :=
  { a := -3, b := -2, c := 0 }

theorem quadratic_term_coefficient_and_constant_term :
  (givenEquation.a = -3) ∧ (givenEquation.c = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_term_coefficient_and_constant_term_l36_3616


namespace NUMINAMATH_CALUDE_spiral_grid_sum_third_row_l36_3629

/-- Represents a square grid with side length n -/
def Grid (n : ℕ) := Fin n → Fin n → ℕ

/-- Fills the grid in a clockwise spiral starting from the center -/
def fillSpiral (n : ℕ) : Grid n :=
  sorry

/-- Returns the largest number in a given row of the grid -/
def largestInRow (g : Grid 17) (row : Fin 17) : ℕ :=
  sorry

/-- Returns the smallest number in a given row of the grid -/
def smallestInRow (g : Grid 17) (row : Fin 17) : ℕ :=
  sorry

theorem spiral_grid_sum_third_row :
  let g := fillSpiral 17
  let thirdRow : Fin 17 := 2
  (largestInRow g thirdRow) + (smallestInRow g thirdRow) = 526 :=
by sorry

end NUMINAMATH_CALUDE_spiral_grid_sum_third_row_l36_3629


namespace NUMINAMATH_CALUDE_power_product_squared_l36_3682

theorem power_product_squared (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l36_3682


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l36_3620

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 3 = 16 →
  a 7 = 2 →
  a 5 = 8 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l36_3620


namespace NUMINAMATH_CALUDE_parentheses_removal_equality_l36_3602

theorem parentheses_removal_equality (x : ℝ) : -(x - 2) - 2 * (x^2 + 2) = -x + 2 - 2*x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_equality_l36_3602


namespace NUMINAMATH_CALUDE_exactly_two_base_pairs_l36_3651

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4 else (a * x^2 + a * x - 1) / x

-- Define what it means for two points to be symmetric about the origin
def symmetricAboutOrigin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

-- Define a base pair
def basePair (a : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  symmetricAboutOrigin p1 p2 ∧ p1.2 = f a p1.1 ∧ p2.2 = f a p2.1

-- The main theorem
theorem exactly_two_base_pairs (a : ℝ) : 
  (∃ p1 p2 p3 p4 : ℝ × ℝ, 
    basePair a p1 p2 ∧ basePair a p3 p4 ∧ 
    p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧
    (∀ p5 p6 : ℝ × ℝ, basePair a p5 p6 → (p5 = p1 ∧ p6 = p2) ∨ (p5 = p3 ∧ p6 = p4) ∨ 
                                         (p5 = p2 ∧ p6 = p1) ∨ (p5 = p4 ∧ p6 = p3))) ↔ 
  a > -6 + 2 * Real.sqrt 6 ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_base_pairs_l36_3651


namespace NUMINAMATH_CALUDE_tshirt_price_l36_3603

/-- The original price of a t-shirt satisfies the given conditions -/
theorem tshirt_price (discount : ℚ) (quantity : ℕ) (revenue : ℚ) 
  (h1 : discount = 8)
  (h2 : quantity = 130)
  (h3 : revenue = 5590) :
  ∃ (original_price : ℚ), 
    quantity * (original_price - discount) = revenue ∧ 
    original_price = 51 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_price_l36_3603


namespace NUMINAMATH_CALUDE_line_intersects_unit_circle_l36_3656

theorem line_intersects_unit_circle 
  (a b : ℝ) (θ : ℝ) (h_neq : a ≠ b) 
  (h_a : a^2 * Real.sin θ + a * Real.cos θ - Real.pi/4 = 0)
  (h_b : b^2 * Real.sin θ + b * Real.cos θ - Real.pi/4 = 0) : 
  ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (b + a) * x - y - a * b = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_unit_circle_l36_3656


namespace NUMINAMATH_CALUDE_seeds_in_bag_l36_3698

-- Define the problem parameters
def seeds_per_ear : ℕ := 4
def price_per_ear : ℚ := 1 / 10
def cost_per_bag : ℚ := 1 / 2
def profit : ℚ := 40
def ears_sold : ℕ := 500

-- Define the theorem
theorem seeds_in_bag : 
  ∃ (seeds_per_bag : ℕ), 
    (ears_sold : ℚ) * price_per_ear - profit = 
    (ears_sold * seeds_per_ear : ℚ) / seeds_per_bag * cost_per_bag ∧ 
    seeds_per_bag = 100 :=
sorry

end NUMINAMATH_CALUDE_seeds_in_bag_l36_3698


namespace NUMINAMATH_CALUDE_root_of_polynomial_l36_3638

theorem root_of_polynomial : ∃ (x : ℝ), x^3 = 5 ∧ x^6 - 6*x^4 - 10*x^3 - 60*x + 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l36_3638


namespace NUMINAMATH_CALUDE_carpenter_needs_eight_more_logs_l36_3630

/-- Represents the carpenter's log and woodblock problem -/
def CarpenterProblem (total_woodblocks : ℕ) (initial_logs : ℕ) (woodblocks_per_log : ℕ) : Prop :=
  let initial_woodblocks := initial_logs * woodblocks_per_log
  let remaining_woodblocks := total_woodblocks - initial_woodblocks
  remaining_woodblocks % woodblocks_per_log = 0 ∧
  remaining_woodblocks / woodblocks_per_log = 8

/-- The carpenter needs 8 more logs to reach the required 80 woodblocks -/
theorem carpenter_needs_eight_more_logs :
  CarpenterProblem 80 8 5 := by
  sorry

#check carpenter_needs_eight_more_logs

end NUMINAMATH_CALUDE_carpenter_needs_eight_more_logs_l36_3630


namespace NUMINAMATH_CALUDE_right_triangle_is_stable_l36_3624

-- Define the concept of a shape
structure Shape :=
  (name : String)

-- Define the property of stability
def is_stable (s : Shape) : Prop := sorry

-- Define a right triangle
def right_triangle : Shape :=
  { name := "Right Triangle" }

-- Define structural rigidity
def has_structural_rigidity (s : Shape) : Prop := sorry

-- Define resistance to deformation
def resists_deformation (s : Shape) : Prop := sorry

-- Theorem: A right triangle is stable
theorem right_triangle_is_stable :
  has_structural_rigidity right_triangle →
  resists_deformation right_triangle →
  is_stable right_triangle :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_is_stable_l36_3624


namespace NUMINAMATH_CALUDE_expression_evaluation_l36_3688

/-- Given a = -2 and b = -1/2, prove that 2(3a^2 - 4ab) - [a^2 - 3(2a + 3ab)] evaluates to 9 -/
theorem expression_evaluation :
  let a : ℚ := -2
  let b : ℚ := -1/2
  2 * (3 * a^2 - 4 * a * b) - (a^2 - 3 * (2 * a + 3 * a * b)) = 9 := by
sorry


end NUMINAMATH_CALUDE_expression_evaluation_l36_3688


namespace NUMINAMATH_CALUDE_fraction_sum_equals_seven_l36_3604

theorem fraction_sum_equals_seven : 
  let U := (1 / (4 - Real.sqrt 15)) - (1 / (Real.sqrt 15 - Real.sqrt 14)) + 
           (1 / (Real.sqrt 14 - Real.sqrt 13)) - (1 / (Real.sqrt 13 - Real.sqrt 12)) + 
           (1 / (Real.sqrt 12 - 3))
  U = 7 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_seven_l36_3604


namespace NUMINAMATH_CALUDE_snackles_leftover_candies_l36_3625

theorem snackles_leftover_candies (m : ℕ) (h : m % 9 = 8) : (2 * m) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_snackles_leftover_candies_l36_3625


namespace NUMINAMATH_CALUDE_ceiling_of_e_l36_3608

theorem ceiling_of_e : ⌈Real.exp 1⌉ = 3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_e_l36_3608


namespace NUMINAMATH_CALUDE_second_boy_speed_l36_3686

/-- Given two boys walking in the same direction for 16 hours, with one boy walking at 5.5 kmph
    and ending up 32 km apart, prove that the speed of the second boy is 7.5 kmph. -/
theorem second_boy_speed (first_speed : ℝ) (time : ℝ) (distance : ℝ) (second_speed : ℝ) :
  first_speed = 5.5 →
  time = 16 →
  distance = 32 →
  distance = (second_speed - first_speed) * time →
  second_speed = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_second_boy_speed_l36_3686


namespace NUMINAMATH_CALUDE_expression_simplification_l36_3644

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (((-2 * x + y)^2 - (2 * x - y) * (y + 2 * x) - 6 * y) / (2 * y)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l36_3644


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l36_3675

theorem cube_sum_divisibility (a b c : ℤ) : 
  (∃ k : ℤ, a + b + c = 6 * k) → (∃ m : ℤ, a^3 + b^3 + c^3 = 6 * m) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l36_3675


namespace NUMINAMATH_CALUDE_colored_cells_count_l36_3664

theorem colored_cells_count (k l : ℕ) : 
  k * l = 74 → 
  (∃ (rows cols : ℕ), 
    rows = 2 * k + 1 ∧ 
    cols = 2 * l + 1 ∧ 
    (rows * cols - 74 = 301 ∨ rows * cols - 74 = 373)) := by
  sorry

end NUMINAMATH_CALUDE_colored_cells_count_l36_3664


namespace NUMINAMATH_CALUDE_no_allowable_formations_l36_3655

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem no_allowable_formations :
  ∀ s t : ℕ,
    s * t = 240 →
    is_prime s →
    8 ≤ t →
    t ≤ 30 →
    ¬∃ (s t : ℕ), s * t = 240 ∧ is_prime s ∧ 8 ≤ t ∧ t ≤ 30 :=
by
  sorry

#check no_allowable_formations

end NUMINAMATH_CALUDE_no_allowable_formations_l36_3655


namespace NUMINAMATH_CALUDE_function_inequality_range_l36_3636

theorem function_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → x^2 - 2*a*x + 2 ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_range_l36_3636
