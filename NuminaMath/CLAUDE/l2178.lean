import Mathlib

namespace NUMINAMATH_CALUDE_equivalent_operation_l2178_217862

theorem equivalent_operation (x : ℝ) : (x / (5/4)) * (4/3) = x * (16/15) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l2178_217862


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2178_217842

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x ≥ 3 → x > 2) ∧
  ¬(∀ x : ℝ, x > 2 → x ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2178_217842


namespace NUMINAMATH_CALUDE_sets_theorem_l2178_217813

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ 3 - 2*a}

theorem sets_theorem :
  (∀ a : ℝ, (Aᶜ ∪ B a = U) ↔ a ≤ 0) ∧
  (∀ a : ℝ, (A ∩ B a ≠ B a) ↔ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_sets_theorem_l2178_217813


namespace NUMINAMATH_CALUDE_circle_circumference_area_equal_diameter_l2178_217840

/-- When the circumference and area of a circle are numerically equal, the diameter is 4. -/
theorem circle_circumference_area_equal_diameter (r : ℝ) :
  2 * Real.pi * r = Real.pi * r^2 → 2 * r = 4 := by sorry

end NUMINAMATH_CALUDE_circle_circumference_area_equal_diameter_l2178_217840


namespace NUMINAMATH_CALUDE_firefighters_total_fires_l2178_217858

/-- The number of fires put out by three firefighters -/
def total_fires (doug_fires : ℕ) (kai_multiplier : ℕ) (eli_divisor : ℕ) : ℕ :=
  doug_fires + (doug_fires * kai_multiplier) + (doug_fires * kai_multiplier / eli_divisor)

/-- Theorem stating the total number of fires put out by Doug, Kai, and Eli -/
theorem firefighters_total_fires :
  total_fires 20 3 2 = 110 := by
  sorry

#eval total_fires 20 3 2

end NUMINAMATH_CALUDE_firefighters_total_fires_l2178_217858


namespace NUMINAMATH_CALUDE_polygon_problem_l2178_217872

theorem polygon_problem :
  ∀ (a b n d : ℤ),
    a > 0 →
    a^2 - 1 = 123 * 125 →
    (x^3 - 16*x^2 - 9*x + a) % (x - 2) = b →
    (n * (n - 3)) / 2 = b + 4 →
    (1 - n) / 2 = (d - 1) / 2 →
    a = 124 ∧ b = 50 ∧ n = 12 ∧ d = -10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_problem_l2178_217872


namespace NUMINAMATH_CALUDE_function_relationship_l2178_217892

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = f (-x + 2))

-- State the theorem
theorem function_relationship :
  f (7/2) < f 1 ∧ f 1 < f (5/2) := by sorry

end NUMINAMATH_CALUDE_function_relationship_l2178_217892


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2178_217841

-- Define set A
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2178_217841


namespace NUMINAMATH_CALUDE_nonzero_sum_zero_power_equality_l2178_217814

theorem nonzero_sum_zero_power_equality (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0)
  (power_equality : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  a^2 + b^2 + c^2 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_nonzero_sum_zero_power_equality_l2178_217814


namespace NUMINAMATH_CALUDE_exponential_function_properties_l2178_217897

theorem exponential_function_properties (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, (x = 0 → a^x = 1) ∧
            (x = 1 → a^x = a) ∧
            (x = -1 → a^x = 1/a) ∧
            (x < 0 → a^x > 0 ∧ ∀ ε > 0, ∃ N : ℝ, ∀ y < N, 0 < a^y ∧ a^y < ε)) :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_properties_l2178_217897


namespace NUMINAMATH_CALUDE_truncated_pyramid_ratio_l2178_217846

/-- Given a right prism with a square base of side length L₁ and height H, 
    and a truncated pyramid extracted from it with square bases of side lengths 
    L₁ (bottom) and L₂ (top) and height H, if the volume of the truncated pyramid 
    is 2/3 of the total volume of the prism, then L₁/L₂ = (1 + √5) / 2. -/
theorem truncated_pyramid_ratio (L₁ L₂ H : ℝ) (h₁ : L₁ > 0) (h₂ : L₂ > 0) (h₃ : H > 0) :
  (H / 3 * (L₁^2 + L₁*L₂ + L₂^2) = 2/3 * H * L₁^2) → L₁ / L₂ = (1 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_ratio_l2178_217846


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2178_217898

open Set

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (Bᶜ) = {x | 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l2178_217898


namespace NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l2178_217818

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallestDivisibleByOneToTen : ℕ := 2520

/-- Proposition: smallestDivisibleByOneToTen is the smallest positive integer 
    divisible by all integers from 1 to 10 -/
theorem smallest_divisible_by_one_to_ten :
  ∀ n : ℕ, n > 0 → (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) → smallestDivisibleByOneToTen ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l2178_217818


namespace NUMINAMATH_CALUDE_average_of_w_x_z_l2178_217879

theorem average_of_w_x_z (w x y z a : ℝ) 
  (h1 : 2/w + 2/x + 2/z = 2/y)
  (h2 : w*x*z = y)
  (h3 : w + x + z = a) :
  (w + x + z) / 3 = a / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_w_x_z_l2178_217879


namespace NUMINAMATH_CALUDE_girls_examined_l2178_217847

theorem girls_examined (boys : ℕ) (girls : ℕ) 
  (h1 : boys = 50)
  (h2 : (25 : ℝ) + 0.6 * girls = 0.5667 * (boys + girls)) :
  girls = 100 := by
  sorry

end NUMINAMATH_CALUDE_girls_examined_l2178_217847


namespace NUMINAMATH_CALUDE_student_transportation_l2178_217878

theorem student_transportation (total : ℚ) 
  (bus car scooter skateboard : ℚ) 
  (h1 : total = 1)
  (h2 : bus = 1/3)
  (h3 : car = 1/5)
  (h4 : scooter = 1/6)
  (h5 : skateboard = 1/8) :
  total - (bus + car + scooter + skateboard) = 7/40 := by
  sorry

end NUMINAMATH_CALUDE_student_transportation_l2178_217878


namespace NUMINAMATH_CALUDE_piggy_bank_pennies_l2178_217861

theorem piggy_bank_pennies (compartments initial_per_compartment final_total : ℕ) 
  (h1 : compartments = 12)
  (h2 : initial_per_compartment = 2)
  (h3 : final_total = 96)
  : (final_total - compartments * initial_per_compartment) / compartments = 6 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_pennies_l2178_217861


namespace NUMINAMATH_CALUDE_joan_has_77_balloons_l2178_217899

/-- The number of balloons Joan has after giving some away and receiving more -/
def joans_balloons (initial_blue initial_red mark_blue mark_red sarah_blue additional_red : ℕ) : ℕ :=
  (initial_blue - mark_blue - sarah_blue) + (initial_red - mark_red + additional_red)

/-- Theorem stating that Joan has 77 balloons given the problem conditions -/
theorem joan_has_77_balloons :
  joans_balloons 72 48 15 10 24 6 = 77 := by
  sorry

#eval joans_balloons 72 48 15 10 24 6

end NUMINAMATH_CALUDE_joan_has_77_balloons_l2178_217899


namespace NUMINAMATH_CALUDE_max_triangle_area_l2178_217893

/-- The maximum area of a triangle with constrained side lengths -/
theorem max_triangle_area (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 1 ≤ b ∧ b ≤ 2) (hc : 2 ≤ c ∧ c ≤ 3)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (S : ℝ), S ≤ 1 ∧ ∀ (S' : ℝ), (∃ (a' b' c' : ℝ),
    0 ≤ a' ∧ a' ≤ 1 ∧
    1 ≤ b' ∧ b' ≤ 2 ∧
    2 ≤ c' ∧ c' ≤ 3 ∧
    a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    S' = (a' * b' * Real.sqrt (1 - (a'*a' + b'*b' - c'*c')^2 / (4*a'*a'*b'*b'))) / 2) →
    S' ≤ S :=
by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_l2178_217893


namespace NUMINAMATH_CALUDE_davids_math_marks_l2178_217833

def englishMarks : ℝ := 70
def physicsMarks : ℝ := 78
def chemistryMarks : ℝ := 60
def biologyMarks : ℝ := 65
def averageMarks : ℝ := 66.6
def totalSubjects : ℕ := 5

theorem davids_math_marks :
  let totalMarks := averageMarks * totalSubjects
  let knownSubjectsMarks := englishMarks + physicsMarks + chemistryMarks + biologyMarks
  let mathMarks := totalMarks - knownSubjectsMarks
  mathMarks = 60 := by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l2178_217833


namespace NUMINAMATH_CALUDE_smallest_measurement_count_l2178_217852

theorem smallest_measurement_count : ∃ N : ℕ+, 
  (∀ m : ℕ+, m < N → 
    (¬(20 * m.val % 100 = 0) ∨ 
     ¬(375 * m.val % 1000 = 0) ∨ 
     ¬(25 * m.val % 100 = 0) ∨ 
     ¬(125 * m.val % 1000 = 0) ∨ 
     ¬(5 * m.val % 100 = 0))) ∧
  (20 * N.val % 100 = 0) ∧ 
  (375 * N.val % 1000 = 0) ∧ 
  (25 * N.val % 100 = 0) ∧ 
  (125 * N.val % 1000 = 0) ∧ 
  (5 * N.val % 100 = 0) ∧
  N.val = 40 := by
sorry

end NUMINAMATH_CALUDE_smallest_measurement_count_l2178_217852


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_l2178_217825

theorem sum_of_squares_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_l2178_217825


namespace NUMINAMATH_CALUDE_parabola_shift_l2178_217823

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := -2 * (x - 1)^2 + 3

-- Theorem stating that the shifted parabola is the result of the described transformations
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l2178_217823


namespace NUMINAMATH_CALUDE_triangle_angle_bounds_l2178_217812

/-- Given a positive real number R, if R, 1, and R+1/2 form a triangle with θ as the angle between R and R+1/2, then 1 < 2Rθ < π. -/
theorem triangle_angle_bounds (R : ℝ) (θ : ℝ) (h_pos : R > 0) 
  (h_triangle : R + 1 > R + 1/2 ∧ R + (R + 1/2) > 1 ∧ 1 + (R + 1/2) > R) 
  (h_angle : θ = Real.arccos ((R^2 + (R + 1/2)^2 - 1) / (2 * R * (R + 1/2)))) :
  1 < 2 * R * θ ∧ 2 * R * θ < π :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_bounds_l2178_217812


namespace NUMINAMATH_CALUDE_diagonals_perpendicular_l2178_217896

-- Define a cube
structure Cube where
  -- Add necessary properties of a cube
  is_cube : Bool

-- Define the angle between diagonals of adjacent faces
def angle_between_diagonals (c : Cube) : ℝ :=
  sorry

-- Theorem statement
theorem diagonals_perpendicular (c : Cube) :
  angle_between_diagonals c = 90 :=
sorry

end NUMINAMATH_CALUDE_diagonals_perpendicular_l2178_217896


namespace NUMINAMATH_CALUDE_red_flesh_probability_l2178_217869

/-- Represents the probability of a tomato having yellow skin -/
def yellow_skin_prob : ℚ := 3/8

/-- Represents the probability of a tomato having red flesh given it has yellow skin -/
def red_flesh_given_yellow_skin_prob : ℚ := 8/15

/-- Represents the probability of a tomato having yellow skin given it doesn't have red flesh -/
def yellow_skin_given_not_red_flesh_prob : ℚ := 7/30

/-- Theorem stating that the probability of red flesh is 1/4 given the conditions -/
theorem red_flesh_probability :
  let yellow_and_not_red : ℚ := yellow_skin_prob * (1 - red_flesh_given_yellow_skin_prob)
  let not_red_flesh_prob : ℚ := yellow_and_not_red / yellow_skin_given_not_red_flesh_prob
  let red_flesh_prob : ℚ := 1 - not_red_flesh_prob
  red_flesh_prob = 1/4 := by sorry

end NUMINAMATH_CALUDE_red_flesh_probability_l2178_217869


namespace NUMINAMATH_CALUDE_marching_band_weight_theorem_l2178_217882

/-- Represents the weight carried by each instrument player in the marching band --/
structure BandWeights where
  trumpet_clarinet : ℕ
  trombone : ℕ
  tuba : ℕ
  drum : ℕ

/-- Represents the number of players for each instrument in the marching band --/
structure BandComposition where
  trumpets : ℕ
  clarinets : ℕ
  trombones : ℕ
  tubas : ℕ
  drummers : ℕ

/-- Calculates the total weight carried by the marching band --/
def total_weight (weights : BandWeights) (composition : BandComposition) : ℕ :=
  (weights.trumpet_clarinet * (composition.trumpets + composition.clarinets)) +
  (weights.trombone * composition.trombones) +
  (weights.tuba * composition.tubas) +
  (weights.drum * composition.drummers)

theorem marching_band_weight_theorem (weights : BandWeights) (composition : BandComposition) :
  weights.trombone = 10 →
  weights.tuba = 20 →
  weights.drum = 15 →
  composition.trumpets = 6 →
  composition.clarinets = 9 →
  composition.trombones = 8 →
  composition.tubas = 3 →
  composition.drummers = 2 →
  total_weight weights composition = 245 →
  weights.trumpet_clarinet = 5 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_weight_theorem_l2178_217882


namespace NUMINAMATH_CALUDE_elephant_to_big_cat_ratio_l2178_217857

/-- Represents the population of animals in a park -/
structure ParkPopulation where
  lions : ℕ
  leopards : ℕ
  elephants : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem about the ratio of elephants to lions and leopards in a park -/
theorem elephant_to_big_cat_ratio 
  (park : ParkPopulation) 
  (h1 : park.lions = 2 * park.leopards) 
  (h2 : park.lions = 200) 
  (h3 : park.lions + park.leopards + park.elephants = 450) : 
  Ratio.mk park.elephants (park.lions + park.leopards) = Ratio.mk 1 2 := by
  sorry

end NUMINAMATH_CALUDE_elephant_to_big_cat_ratio_l2178_217857


namespace NUMINAMATH_CALUDE_inequality_always_true_range_l2178_217866

theorem inequality_always_true_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_always_true_range_l2178_217866


namespace NUMINAMATH_CALUDE_zoo_count_difference_l2178_217848

/-- Proves that the difference between the number of monkeys and giraffes is 22 -/
theorem zoo_count_difference : 
  let zebras : ℕ := 12
  let camels : ℕ := zebras / 2
  let monkeys : ℕ := 4 * camels
  let giraffes : ℕ := 2
  monkeys - giraffes = 22 := by sorry

end NUMINAMATH_CALUDE_zoo_count_difference_l2178_217848


namespace NUMINAMATH_CALUDE_variance_linear_transform_l2178_217816

-- Define a random variable X
variable (X : ℝ → ℝ)

-- Define the variance function D
noncomputable def D (Y : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem variance_linear_transform (h : D X = 2) : D (fun ω => 3 * X ω + 2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_variance_linear_transform_l2178_217816


namespace NUMINAMATH_CALUDE_average_students_count_l2178_217851

theorem average_students_count (total : ℕ) (top_yes : ℕ) (avg_yes : ℕ) (under_yes : ℕ) :
  total = 30 →
  top_yes = 19 →
  avg_yes = 12 →
  under_yes = 9 →
  ∃ (top avg under : ℕ),
    top + avg + under = total ∧
    top = top_yes ∧
    avg = avg_yes ∧
    under = under_yes :=
by
  sorry

end NUMINAMATH_CALUDE_average_students_count_l2178_217851


namespace NUMINAMATH_CALUDE_solve_equation_l2178_217824

theorem solve_equation (x : ℝ) (h : x + 1 = 3) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2178_217824


namespace NUMINAMATH_CALUDE_circus_dog_paws_l2178_217837

theorem circus_dog_paws (total_dogs : ℕ) (back_leg_fraction : ℚ) : total_dogs = 24 → back_leg_fraction = 2/3 → (total_dogs : ℚ) * back_leg_fraction * 2 + (total_dogs : ℚ) * (1 - back_leg_fraction) * 4 = 64 := by
  sorry

end NUMINAMATH_CALUDE_circus_dog_paws_l2178_217837


namespace NUMINAMATH_CALUDE_square_root_of_nine_l2178_217883

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l2178_217883


namespace NUMINAMATH_CALUDE_train_length_calculation_l2178_217865

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed : Real) (crossing_time : Real) (bridge_length : Real) :
  let speed_ms : Real := train_speed * (1000 / 3600)
  let total_distance : Real := speed_ms * crossing_time
  let train_length : Real := total_distance - bridge_length
  train_speed = 45 ∧ crossing_time = 30 ∧ bridge_length = 230 →
  train_length = 145 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2178_217865


namespace NUMINAMATH_CALUDE_angle_measure_problem_l2178_217860

theorem angle_measure_problem (C D E F G : Real) : 
  C = 120 →
  C + D = 180 →
  E = 50 →
  F = D →
  E + F + G = 180 →
  G = 70 := by sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l2178_217860


namespace NUMINAMATH_CALUDE_binary_eight_ones_decimal_l2178_217827

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Nat

/-- Converts a binary number to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + bit * 2^i) 0

/-- The binary number (11111111)₂ -/
def binaryEightOnes : BinaryNumber := [1,1,1,1,1,1,1,1]

theorem binary_eight_ones_decimal :
  binaryToDecimal binaryEightOnes = 2^8 - 1 := by
  sorry

end NUMINAMATH_CALUDE_binary_eight_ones_decimal_l2178_217827


namespace NUMINAMATH_CALUDE_line_through_point_l2178_217801

theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, b*x + (b-1)*y = b^2 - 1 → (x = 2 ∧ y = -5)) → 
  (b = (-3 + Real.sqrt 33) / 2 ∨ b = (-3 - Real.sqrt 33) / 2) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_l2178_217801


namespace NUMINAMATH_CALUDE_adam_nuts_purchase_l2178_217829

theorem adam_nuts_purchase (nuts_price dried_fruits_price dried_fruits_weight total_cost : ℝ) 
  (h1 : nuts_price = 12)
  (h2 : dried_fruits_price = 8)
  (h3 : dried_fruits_weight = 2.5)
  (h4 : total_cost = 56) :
  ∃ (nuts_weight : ℝ), 
    nuts_weight * nuts_price + dried_fruits_weight * dried_fruits_price = total_cost ∧ 
    nuts_weight = 3 := by
sorry


end NUMINAMATH_CALUDE_adam_nuts_purchase_l2178_217829


namespace NUMINAMATH_CALUDE_cos_sum_thirteen_l2178_217890

theorem cos_sum_thirteen : 
  Real.cos (2 * Real.pi / 13) + Real.cos (6 * Real.pi / 13) + Real.cos (8 * Real.pi / 13) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_thirteen_l2178_217890


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l2178_217894

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base_7_num := to_base_10 [0, 3, 4, 2, 5] 7
  let base_8_num := to_base_10 [0, 2, 3, 4] 8
  base_7_num - base_8_num = 10652 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l2178_217894


namespace NUMINAMATH_CALUDE_inequality_proof_l2178_217808

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a * b + b * c + c * a = 1) : 
  (a + b + c ≥ Real.sqrt 3) ∧ 
  (Real.sqrt (a / (b * c)) + Real.sqrt (b / (c * a)) + Real.sqrt (c / (a * b)) ≥ 
   Real.sqrt 3 * (Real.sqrt a + Real.sqrt b + Real.sqrt c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2178_217808


namespace NUMINAMATH_CALUDE_final_price_theorem_l2178_217838

def mothers_day_discount : ℝ := 0.10
def additional_children_discount : ℝ := 0.04
def vip_discount : ℝ := 0.05
def shoes_cost : ℝ := 125
def handbag_cost : ℝ := 75
def min_purchase : ℝ := 150

def total_cost : ℝ := shoes_cost + handbag_cost

def discounted_price (price : ℝ) : ℝ :=
  let price_after_mothers_day := price * (1 - mothers_day_discount)
  let price_after_children := price_after_mothers_day * (1 - additional_children_discount)
  price_after_children * (1 - vip_discount)

theorem final_price_theorem :
  total_cost ≥ min_purchase →
  discounted_price total_cost = 164.16 :=
by sorry

end NUMINAMATH_CALUDE_final_price_theorem_l2178_217838


namespace NUMINAMATH_CALUDE_power_of_square_l2178_217802

theorem power_of_square (a : ℝ) : (a^2)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l2178_217802


namespace NUMINAMATH_CALUDE_general_term_formula_l2178_217804

def S (n : ℕ) : ℤ := 1 - 2^n

def a (n : ℕ) : ℤ := -2^(n-1)

theorem general_term_formula (n : ℕ) (h : n ≥ 2) : 
  a n = S n - S (n-1) := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_l2178_217804


namespace NUMINAMATH_CALUDE_problem_solution_l2178_217887

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 2|

-- Theorem statement
theorem problem_solution :
  (∃ (M : ℝ), (∀ x, f x ≥ M) ∧ (∃ x, f x = M) ∧ M = 3) ∧
  ({x : ℝ | f x < 3 + |2*x + 2|} = Set.Ioo (-1) 2) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 2*b^2 = 3 → 2*a + b ≤ 3*Real.sqrt 6 / 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 2*b^2 = 3 ∧ 2*a + b = 3*Real.sqrt 6 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l2178_217887


namespace NUMINAMATH_CALUDE_typing_service_problem_l2178_217855

/-- Typing service problem -/
theorem typing_service_problem
  (total_pages : ℕ)
  (first_time_cost : ℕ)
  (revision_cost : ℕ)
  (pages_revised_once : ℕ)
  (total_cost : ℕ)
  (h1 : total_pages = 100)
  (h2 : first_time_cost = 5)
  (h3 : revision_cost = 3)
  (h4 : pages_revised_once = 30)
  (h5 : total_cost = 710)
  : ∃ (pages_revised_twice : ℕ),
    pages_revised_twice = 20 ∧
    total_cost = total_pages * first_time_cost +
                 pages_revised_once * revision_cost +
                 pages_revised_twice * revision_cost * 2 :=
by sorry

end NUMINAMATH_CALUDE_typing_service_problem_l2178_217855


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l2178_217888

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l2178_217888


namespace NUMINAMATH_CALUDE_birdhouse_flew_1200_feet_l2178_217844

/-- The distance the car was transported, in feet -/
def car_distance : ℕ := 200

/-- The distance the lawn chair was blown, in feet -/
def lawn_chair_distance : ℕ := 2 * car_distance

/-- The distance the birdhouse flew, in feet -/
def birdhouse_distance : ℕ := 3 * lawn_chair_distance

/-- Theorem stating that the birdhouse flew 1200 feet -/
theorem birdhouse_flew_1200_feet : birdhouse_distance = 1200 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_flew_1200_feet_l2178_217844


namespace NUMINAMATH_CALUDE_voter_percentage_for_candidate_A_l2178_217850

theorem voter_percentage_for_candidate_A
  (total_voters : ℝ)
  (democrat_percentage : ℝ)
  (democrat_support_A : ℝ)
  (republican_support_A : ℝ)
  (h1 : democrat_percentage = 0.6)
  (h2 : democrat_support_A = 0.7)
  (h3 : republican_support_A = 0.2)
  (h4 : total_voters > 0) :
  let republican_percentage := 1 - democrat_percentage
  let voters_for_A := total_voters * (democrat_percentage * democrat_support_A + republican_percentage * republican_support_A)
  voters_for_A / total_voters = 0.5 := by
sorry

end NUMINAMATH_CALUDE_voter_percentage_for_candidate_A_l2178_217850


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l2178_217864

def f (x a : ℝ) : ℝ := |2 * x - a| + a

theorem solution_set_implies_a_value :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x 1 ≤ 6) →
  (∀ x : ℝ, f x 1 ≤ 6 → -2 ≤ x ∧ x ≤ 3) →
  (∃ a : ℝ, ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x a ≤ 6) →
  (∃! a : ℝ, ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x a ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l2178_217864


namespace NUMINAMATH_CALUDE_neg_one_pow_2022_eq_one_neg_one_pow_2022_and_one_are_opposite_l2178_217809

/-- Two real numbers are opposite if their sum is zero -/
def are_opposite (a b : ℝ) : Prop := a + b = 0

/-- -1^2022 equals 1 -/
theorem neg_one_pow_2022_eq_one : (-1 : ℝ)^2022 = 1 := by sorry

theorem neg_one_pow_2022_and_one_are_opposite :
  are_opposite ((-1 : ℝ)^2022) 1 := by sorry

end NUMINAMATH_CALUDE_neg_one_pow_2022_eq_one_neg_one_pow_2022_and_one_are_opposite_l2178_217809


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2178_217832

theorem arithmetic_equality : 12.05 * 5.4 + 0.6 = 65.67 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2178_217832


namespace NUMINAMATH_CALUDE_max_perfect_squares_pairwise_products_l2178_217845

/-- Given two distinct natural numbers, the maximum number of perfect squares
    among the pairwise products of these numbers and their +2 counterparts is 2. -/
theorem max_perfect_squares_pairwise_products (a b : ℕ) (h : a ≠ b) :
  let products := {a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)}
  (∃ (s : Finset ℕ), s ⊆ products ∧ (∀ x ∈ s, ∃ y, x = y^2) ∧ s.card = 2) ∧
  (∀ (s : Finset ℕ), s ⊆ products → (∀ x ∈ s, ∃ y, x = y^2) → s.card ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_max_perfect_squares_pairwise_products_l2178_217845


namespace NUMINAMATH_CALUDE_least_trees_for_rows_trees_168_divisible_least_trees_is_168_l2178_217820

theorem least_trees_for_rows (n : ℕ) : n > 0 ∧ 6 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n → n ≥ 168 := by
  sorry

theorem trees_168_divisible : 6 ∣ 168 ∧ 7 ∣ 168 ∧ 8 ∣ 168 := by
  sorry

theorem least_trees_is_168 : ∃ (n : ℕ), n > 0 ∧ 6 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 6 ∣ m ∧ 7 ∣ m ∧ 8 ∣ m) → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_least_trees_for_rows_trees_168_divisible_least_trees_is_168_l2178_217820


namespace NUMINAMATH_CALUDE_certain_number_problem_l2178_217854

theorem certain_number_problem (h : 2994 / 14.5 = 179) : 
  ∃ x : ℝ, x / 1.45 = 17.9 ∧ x = 25.955 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2178_217854


namespace NUMINAMATH_CALUDE_number_equals_scientific_notation_l2178_217859

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 858000000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation := {
  coefficient := 8.58
  exponent := 8
  is_valid := by sorry
}

/-- Theorem stating that the number is equal to its scientific notation representation -/
theorem number_equals_scientific_notation : 
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end NUMINAMATH_CALUDE_number_equals_scientific_notation_l2178_217859


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2178_217805

theorem inequality_solution_set : 
  {x : ℝ | 2*x + 1 > x + 2} = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2178_217805


namespace NUMINAMATH_CALUDE_num_algebraic_expressions_is_five_l2178_217856

/-- An expression is algebraic if it consists of numbers, variables, and arithmetic operations, without equality or inequality symbols. -/
def is_algebraic_expression (e : String) : Bool :=
  match e with
  | "2x^2" => true
  | "1-2x=0" => false
  | "ab" => true
  | "a>0" => false
  | "0" => true
  | "1/a" => true
  | "π" => true
  | _ => false

/-- The list of expressions to be checked -/
def expressions : List String :=
  ["2x^2", "1-2x=0", "ab", "a>0", "0", "1/a", "π"]

/-- The number of algebraic expressions in the list -/
def num_algebraic_expressions : Nat :=
  (expressions.filter is_algebraic_expression).length

theorem num_algebraic_expressions_is_five :
  num_algebraic_expressions = 5 := by
  sorry

end NUMINAMATH_CALUDE_num_algebraic_expressions_is_five_l2178_217856


namespace NUMINAMATH_CALUDE_area_of_encompassing_rectangle_l2178_217800

/-- Given two identical rectangles with intersecting extended sides, prove the area of the encompassing rectangle. -/
theorem area_of_encompassing_rectangle 
  (area_BNHM : ℝ) 
  (area_MBCK : ℝ) 
  (area_MLGH : ℝ) 
  (h1 : area_BNHM = 12)
  (h2 : area_MBCK = 63)
  (h3 : area_MLGH = 28) : 
  ∃ (area_IFJD : ℝ), area_IFJD = 418 := by
sorry

end NUMINAMATH_CALUDE_area_of_encompassing_rectangle_l2178_217800


namespace NUMINAMATH_CALUDE_quarterback_sacks_l2178_217836

theorem quarterback_sacks (total_attempts : ℕ) (no_throw_percentage : ℚ) (sack_ratio : ℚ) : 
  total_attempts = 80 → 
  no_throw_percentage = 30 / 100 → 
  sack_ratio = 1 / 2 → 
  ⌊(total_attempts : ℚ) * no_throw_percentage * sack_ratio⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_quarterback_sacks_l2178_217836


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2178_217821

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ (a b c : ℝ),
  (∀ x, f a b c (x + 1) - f a b c x = 2 * x) →
  f a b c 0 = 1 →
  (∃ m : ℝ, m = -1 ∧
    (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a b c x ≥ 2 * x + m) ∧
    (∀ m' : ℝ, m' > m →
      ∃ x, x ∈ Set.Icc (-1 : ℝ) 1 ∧ f a b c x < 2 * x + m')) →
  (∀ x, f a b c x = x^2 - x + 1) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l2178_217821


namespace NUMINAMATH_CALUDE_special_divisors_count_l2178_217870

/-- The base number -/
def base : ℕ := 540

/-- The exponent of the base number -/
def exponent : ℕ := 540

/-- The number of divisors we're looking for -/
def target_divisors : ℕ := 108

/-- A function that counts the number of positive integer divisors of base^exponent 
    that are divisible by exactly target_divisors positive integers -/
def count_special_divisors (base exponent target_divisors : ℕ) : ℕ := sorry

/-- The main theorem stating that the count of special divisors is 6 -/
theorem special_divisors_count : 
  count_special_divisors base exponent target_divisors = 6 := by sorry

end NUMINAMATH_CALUDE_special_divisors_count_l2178_217870


namespace NUMINAMATH_CALUDE_divisor_problem_l2178_217868

theorem divisor_problem (n d k q : ℤ) : 
  n = 25 * k + 4 →
  n + 15 = d * q + 4 →
  d > 0 →
  d = 19 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2178_217868


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2178_217881

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0 ∧ x₁ > 2 ∧ x₂ < 2) ↔ 
  m > 3/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2178_217881


namespace NUMINAMATH_CALUDE_binary_11_equals_3_l2178_217880

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 3 -/
def binary_three : List Bool := [true, true]

/-- Theorem stating that the binary number 11 (base 2) is equal to 3 (base 10) -/
theorem binary_11_equals_3 : binary_to_decimal binary_three = 3 := by
  sorry

end NUMINAMATH_CALUDE_binary_11_equals_3_l2178_217880


namespace NUMINAMATH_CALUDE_multiple_solutions_exist_l2178_217891

-- Define the system of equations
def system (x y z w : ℝ) : Prop :=
  x = z + w - z*w ∧
  y = w + x - w*x ∧
  z = x + y - x*y ∧
  w = y + z - y*z

-- Theorem statement
theorem multiple_solutions_exist :
  ∃ (x₁ y₁ z₁ w₁ x₂ y₂ z₂ w₂ : ℝ),
    system x₁ y₁ z₁ w₁ ∧
    system x₂ y₂ z₂ w₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂ ∨ z₁ ≠ z₂ ∨ w₁ ≠ w₂) :=
by
  sorry


end NUMINAMATH_CALUDE_multiple_solutions_exist_l2178_217891


namespace NUMINAMATH_CALUDE_reena_loan_interest_l2178_217863

/-- Calculate simple interest for a loan where the loan period in years equals the interest rate -/
def simple_interest (principal : ℚ) (rate : ℚ) : ℚ :=
  principal * rate * rate / 100

theorem reena_loan_interest :
  let principal : ℚ := 1200
  let rate : ℚ := 4
  simple_interest principal rate = 192 := by
sorry

end NUMINAMATH_CALUDE_reena_loan_interest_l2178_217863


namespace NUMINAMATH_CALUDE_jellybean_count_l2178_217849

theorem jellybean_count (initial_count : ℕ) : 
  (initial_count : ℝ) * (0.7 ^ 3) = 28 → initial_count = 82 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l2178_217849


namespace NUMINAMATH_CALUDE_positive_intervals_l2178_217843

def f (x : ℝ) := (x + 2) * (x - 2) * (x + 1)

theorem positive_intervals (x : ℝ) : 
  f x > 0 ↔ (x > -2 ∧ x < -1) ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_positive_intervals_l2178_217843


namespace NUMINAMATH_CALUDE_prop_3_prop_4_l2178_217886

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Define the lines and planes
variable (m n l : Line)
variable (α : Plane)

-- State the theorems
theorem prop_3 (h1 : parallel m n) (h2 : perpendicular_plane m α) :
  perpendicular_plane n α := by sorry

theorem prop_4 (h1 : skew m n) (h2 : parallel_plane m α) (h3 : parallel_plane n α)
  (h4 : perpendicular m l) (h5 : perpendicular n l) :
  perpendicular_plane l α := by sorry

end NUMINAMATH_CALUDE_prop_3_prop_4_l2178_217886


namespace NUMINAMATH_CALUDE_test_probabilities_l2178_217839

/-- Probability of an event occurring -/
def Prob (event : Prop) : ℝ := sorry

/-- The probability that individual A passes the test -/
def probA : ℝ := 0.8

/-- The probability that individual B passes the test -/
def probB : ℝ := 0.6

/-- The probability that individual C passes the test -/
def probC : ℝ := 0.5

/-- A passes the test -/
def A : Prop := sorry

/-- B passes the test -/
def B : Prop := sorry

/-- C passes the test -/
def C : Prop := sorry

theorem test_probabilities :
  (Prob A = probA) ∧
  (Prob B = probB) ∧
  (Prob C = probC) ∧
  (Prob (A ∧ B ∧ C) = 0.24) ∧
  (Prob (A ∨ B ∨ C) = 0.96) := by sorry

end NUMINAMATH_CALUDE_test_probabilities_l2178_217839


namespace NUMINAMATH_CALUDE_mixture_replacement_l2178_217876

/-- Given a mixture of liquids A and B with an initial ratio of 4:1 and a final ratio of 2:3 after
    replacing some mixture with pure B, prove that 60 liters of mixture were replaced when the
    initial amount of liquid A was 48 liters. -/
theorem mixture_replacement (initial_A : ℝ) (initial_B : ℝ) (replaced : ℝ) :
  initial_A = 48 →
  initial_A / initial_B = 4 / 1 →
  initial_A / (initial_B + replaced) = 2 / 3 →
  replaced = 60 :=
by sorry

end NUMINAMATH_CALUDE_mixture_replacement_l2178_217876


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2178_217873

def box_values : List ℝ := [0.01, 1, 5, 10, 25, 50, 100, 200, 300, 400, 500, 750, 1000, 5000, 10000, 25000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000]

def total_boxes : ℕ := 30

def threshold : ℝ := 200000

theorem deal_or_no_deal_probability (boxes_to_eliminate : ℕ) :
  boxes_to_eliminate = 16 ↔
    (((box_values.filter (λ x => x ≥ threshold)).length : ℝ) / (total_boxes - boxes_to_eliminate : ℝ) = 1/2 ∧
     boxes_to_eliminate < total_boxes ∧
     ∀ n : ℕ, n < boxes_to_eliminate →
       ((box_values.filter (λ x => x ≥ threshold)).length : ℝ) / (total_boxes - n : ℝ) < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2178_217873


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l2178_217826

theorem constant_ratio_problem (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) (k : ℝ) :
  (3 * x₁ - 4) / (y₁ + 15) = k →
  (3 * x₂ - 4) / (y₂ + 15) = k →
  x₁ = 2 →
  y₁ = 3 →
  y₂ = 12 →
  x₂ = 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l2178_217826


namespace NUMINAMATH_CALUDE_product_absolute_value_one_l2178_217875

theorem product_absolute_value_one 
  (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d)
  (h1 : a + 1/b = b + 1/c)
  (h2 : b + 1/c = c + 1/d)
  (h3 : c + 1/d = d + 1/a) :
  |a * b * c * d| = 1 := by
sorry

end NUMINAMATH_CALUDE_product_absolute_value_one_l2178_217875


namespace NUMINAMATH_CALUDE_street_lights_configuration_l2178_217885

theorem street_lights_configuration (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 12 →
  k = 4 →
  m = n - k - 1 →
  Nat.choose m k = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_street_lights_configuration_l2178_217885


namespace NUMINAMATH_CALUDE_sequence_of_primes_l2178_217835

theorem sequence_of_primes (a p : ℕ → ℕ) 
  (h_increasing : ∀ n m, n < m → a n < a m)
  (h_positive : ∀ n, 0 < a n)
  (h_prime : ∀ n, Nat.Prime (p n))
  (h_distinct : ∀ n m, n ≠ m → p n ≠ p m)
  (h_divides : ∀ n, p n ∣ a n)
  (h_difference : ∀ n k, a n - a k = p n - p k) :
  ∀ n, a n = p n :=
sorry

end NUMINAMATH_CALUDE_sequence_of_primes_l2178_217835


namespace NUMINAMATH_CALUDE_third_quadrant_condition_l2178_217877

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m + 3) (m - 1)

-- Define what it means for a complex number to be in the third quadrant
def in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

-- The theorem statement
theorem third_quadrant_condition (m : ℝ) :
  in_third_quadrant (z m) ↔ m < -3 := by
  sorry

end NUMINAMATH_CALUDE_third_quadrant_condition_l2178_217877


namespace NUMINAMATH_CALUDE_factors_of_36_l2178_217819

def number : ℕ := 36

-- Sum of positive factors
def sum_of_factors (n : ℕ) : ℕ := sorry

-- Product of prime factors
def product_of_prime_factors (n : ℕ) : ℕ := sorry

theorem factors_of_36 :
  sum_of_factors number = 91 ∧ product_of_prime_factors number = 6 := by sorry

end NUMINAMATH_CALUDE_factors_of_36_l2178_217819


namespace NUMINAMATH_CALUDE_divisible_by_eight_last_digits_l2178_217803

theorem divisible_by_eight_last_digits : 
  ∃! (S : Finset Nat), 
    (∀ n ∈ S, n < 10) ∧ 
    (∀ m : Nat, m % 8 = 0 → m % 10 ∈ S) ∧
    Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_last_digits_l2178_217803


namespace NUMINAMATH_CALUDE_count_satisfying_numbers_l2178_217822

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧
  n + reverse_digits n = 110 ∧
  sum_of_digits n % 3 = 0

theorem count_satisfying_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_conditions n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_satisfying_numbers_l2178_217822


namespace NUMINAMATH_CALUDE_special_number_theorem_l2178_217811

def is_smallest_nontrivial_divisor (a n : ℕ) : Prop :=
  a ≠ 1 ∧ a ∣ n ∧ ∀ d, 1 < d → d < a → ¬(d ∣ n)

theorem special_number_theorem :
  ∀ n : ℕ, n ≥ 2 →
  (∃ a b : ℕ, is_smallest_nontrivial_divisor a n ∧ b ∣ n ∧ n = a^2 + b^2) →
  (n = 8 ∨ n = 20) :=
sorry

end NUMINAMATH_CALUDE_special_number_theorem_l2178_217811


namespace NUMINAMATH_CALUDE_harkamal_payment_l2178_217884

def grapes_qty : ℝ := 8
def grapes_price : ℝ := 80
def mangoes_qty : ℝ := 9
def mangoes_price : ℝ := 55
def apples_qty : ℝ := 6
def apples_price : ℝ := 120
def oranges_qty : ℝ := 4
def oranges_price : ℝ := 75
def apple_discount : ℝ := 0.1
def sales_tax : ℝ := 0.05

def total_cost : ℝ :=
  grapes_qty * grapes_price +
  mangoes_qty * mangoes_price +
  apples_qty * apples_price * (1 - apple_discount) +
  oranges_qty * oranges_price

def final_cost : ℝ := total_cost * (1 + sales_tax)

theorem harkamal_payment : final_cost = 2187.15 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l2178_217884


namespace NUMINAMATH_CALUDE_earth_livable_fraction_l2178_217853

/-- The fraction of the earth's surface not covered by water -/
def land_fraction : ℚ := 1/3

/-- The fraction of exposed land that is inhabitable -/
def inhabitable_fraction : ℚ := 1/3

/-- The fraction of the earth's surface that humans can live on -/
def livable_fraction : ℚ := land_fraction * inhabitable_fraction

theorem earth_livable_fraction :
  livable_fraction = 1/9 := by sorry

end NUMINAMATH_CALUDE_earth_livable_fraction_l2178_217853


namespace NUMINAMATH_CALUDE_divisor_property_characterization_l2178_217806

/-- A positive integer n > 1 satisfies the divisor property if all its positive divisors
    greater than 1 are of the form a^r + 1, where a and r are positive integers and r > 1 -/
def satisfies_divisor_property (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d ∣ n → ∃ a r : ℕ, a > 0 ∧ r > 1 ∧ d = a^r + 1

/-- The main theorem stating that if n satisfies the divisor property,
    then n = 10 or n is a prime of the form a^2 + 1 -/
theorem divisor_property_characterization (n : ℕ) :
  satisfies_divisor_property n →
  (n = 10 ∨ (Nat.Prime n ∧ ∃ a : ℕ, n = a^2 + 1)) :=
by sorry


end NUMINAMATH_CALUDE_divisor_property_characterization_l2178_217806


namespace NUMINAMATH_CALUDE_sum_m_n_equals_three_l2178_217830

theorem sum_m_n_equals_three (m n : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : m / (1 + i) = 1 - n * i) : m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_three_l2178_217830


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2178_217889

theorem decimal_to_fraction :
  (3.75 : ℚ) = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2178_217889


namespace NUMINAMATH_CALUDE_parallelogram_base_proof_l2178_217834

/-- The area of a parallelogram -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_base_proof (area height : ℝ) (h1 : area = 96) (h2 : height = 8) :
  parallelogram_area (area / height) height = area → area / height = 12 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_proof_l2178_217834


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2178_217874

def M : ℕ := 36 * 36 * 65 * 272

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M) * 510 = sum_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2178_217874


namespace NUMINAMATH_CALUDE_max_c_value_l2178_217871

theorem max_c_value (a b c x y z : ℝ) : 
  a ≥ 1 → b ≥ 1 → c ≥ 1 → x > 0 → y > 0 → z > 0 →
  a^x + b^y + c^z = 4 →
  x * a^x + y * b^y + z * c^z = 6 →
  x^2 * a^x + y^2 * b^y + z^2 * c^z = 9 →
  c ≤ Real.rpow 4 (1/3) :=
sorry

end NUMINAMATH_CALUDE_max_c_value_l2178_217871


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2178_217895

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 20
  let k : ℕ := 3
  let a : ℤ := 2
  let b : ℤ := -3
  (n.choose k) * a^k * b^(n-k) = -1174898049840 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2178_217895


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2178_217815

theorem sum_of_decimals : (7.46 : ℝ) + 4.29 = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2178_217815


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_seven_to_500_l2178_217810

theorem x_one_minus_f_equals_seven_to_500 :
  let x : ℝ := (3 + Real.sqrt 2) ^ 500
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 7 ^ 500 := by
sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_seven_to_500_l2178_217810


namespace NUMINAMATH_CALUDE_dessert_preference_l2178_217867

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ) :
  total = 40 →
  apple = 18 →
  chocolate = 15 →
  neither = 12 →
  ∃ (both : ℕ), both = 5 ∧ total = apple + chocolate - both + neither :=
by
  sorry

end NUMINAMATH_CALUDE_dessert_preference_l2178_217867


namespace NUMINAMATH_CALUDE_cookie_problem_l2178_217807

/-- Calculates the number of chocolate chips per cookie -/
def chocolate_chips_per_cookie (cookies_per_batch : ℕ) (family_members : ℕ) (batches : ℕ) (chips_per_person : ℕ) : ℕ :=
  let total_cookies := cookies_per_batch * batches
  let cookies_per_person := total_cookies / family_members
  chips_per_person / cookies_per_person

/-- Proves that the number of chocolate chips per cookie is 2 under given conditions -/
theorem cookie_problem : 
  chocolate_chips_per_cookie 12 4 3 18 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l2178_217807


namespace NUMINAMATH_CALUDE_count_grid_paths_l2178_217831

/-- The number of paths from (0,0) to (m, n) on a grid, moving only right or up by one unit at a time -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem stating that the number of paths from (0,0) to (m, n) on a grid,
    moving only right or up by one unit at a time, is equal to (m+n choose m) -/
theorem count_grid_paths (m n : ℕ) : 
  gridPaths m n = Nat.choose (m + n) m := by
  sorry

end NUMINAMATH_CALUDE_count_grid_paths_l2178_217831


namespace NUMINAMATH_CALUDE_range_of_a_proposition_holds_l2178_217828

/-- The proposition that the inequality ax^2 - 2ax - 3 ≥ 0 does not hold for all real x -/
def proposition (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬(a * x^2 - 2 * a * x - 3 ≥ 0)

/-- The theorem stating that if the proposition holds, then a is in the range (-3, 0] -/
theorem range_of_a (a : ℝ) (h : proposition a) : -3 < a ∧ a ≤ 0 := by
  sorry

/-- The theorem stating that if a is in the range (-3, 0], then the proposition holds -/
theorem proposition_holds (a : ℝ) (h : -3 < a ∧ a ≤ 0) : proposition a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_proposition_holds_l2178_217828


namespace NUMINAMATH_CALUDE_balls_in_boxes_l2178_217817

/-- The number of ways to place n different balls into m different boxes, with at most one ball per box -/
def place_balls (n m : ℕ) : ℕ :=
  Nat.descFactorial m n

theorem balls_in_boxes : place_balls 3 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l2178_217817
