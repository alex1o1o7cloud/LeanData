import Mathlib

namespace NUMINAMATH_CALUDE_increase_by_percentage_l717_71797

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 3680 ∧ percentage = 84.3 ∧ final = initial * (1 + percentage / 100) →
  final = 6782.64 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l717_71797


namespace NUMINAMATH_CALUDE_emily_spent_28_dollars_l717_71777

/-- Calculates the total cost of Emily's flower purchase --/
def emily_flower_cost (rose_price daisy_price tulip_price lily_price : ℕ)
  (rose_qty daisy_qty tulip_qty lily_qty : ℕ) : ℕ :=
  rose_price * rose_qty + daisy_price * daisy_qty + tulip_price * tulip_qty + lily_price * lily_qty

/-- Proves that Emily spent 28 dollars on flowers --/
theorem emily_spent_28_dollars :
  emily_flower_cost 4 3 5 6 2 3 1 1 = 28 := by
  sorry

end NUMINAMATH_CALUDE_emily_spent_28_dollars_l717_71777


namespace NUMINAMATH_CALUDE_large_box_height_is_four_l717_71798

-- Define the dimensions of the larger box
def large_box_length : ℝ := 6
def large_box_width : ℝ := 5

-- Define the dimensions of the smaller box in meters
def small_box_length : ℝ := 0.6
def small_box_width : ℝ := 0.5
def small_box_height : ℝ := 0.4

-- Define the maximum number of small boxes
def max_small_boxes : ℕ := 1000

-- Theorem statement
theorem large_box_height_is_four :
  ∃ (h : ℝ), 
    h = 4 ∧ 
    large_box_length * large_box_width * h = 
      (max_small_boxes : ℝ) * small_box_length * small_box_width * small_box_height :=
by sorry

end NUMINAMATH_CALUDE_large_box_height_is_four_l717_71798


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l717_71743

theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, a * x^2 = (x + 1) * (x - 1)) → a ≠ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l717_71743


namespace NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l717_71789

/-- A police emergency number is a positive integer that ends with 133 in decimal representation. -/
def is_police_emergency_number (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7. -/
theorem police_emergency_number_has_large_prime_divisor (n : ℕ) :
  is_police_emergency_number n → ∃ p : ℕ, p > 7 ∧ Nat.Prime p ∧ p ∣ n :=
sorry

end NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l717_71789


namespace NUMINAMATH_CALUDE_point_trajectory_l717_71736

/-- The trajectory of a point with constant ratio between distances to axes -/
theorem point_trajectory (k : ℝ) (h : k ≠ 0) :
  ∀ x y : ℝ, x ≠ 0 →
  (|y| / |x| = k) ↔ (y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_point_trajectory_l717_71736


namespace NUMINAMATH_CALUDE_smallest_n_for_125_l717_71712

/-- The sequence term defined as a function of n -/
def a (n : ℕ) : ℤ := 2 * n^2 - 3

/-- The proposition that 8 is the smallest positive integer n for which a(n) = 125 -/
theorem smallest_n_for_125 : ∀ n : ℕ, n > 0 → a n = 125 → n ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_125_l717_71712


namespace NUMINAMATH_CALUDE_average_weight_increase_l717_71755

theorem average_weight_increase (group_size : ℕ) (original_weight new_weight : ℝ) :
  group_size = 6 →
  original_weight = 65 →
  new_weight = 74 →
  (new_weight - original_weight) / group_size = 1.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l717_71755


namespace NUMINAMATH_CALUDE_shadows_parallel_l717_71775

-- Define a structure for a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a structure for a point on a projection plane
structure ProjectedPoint where
  x : ℝ
  y : ℝ

-- Define a structure for a light source (parallel lighting)
structure ParallelLight where
  direction : Point3D

-- Define a function to project a 3D point onto a plane
def project (p : Point3D) (plane : ℝ) (light : ParallelLight) : ProjectedPoint :=
  sorry

-- Define a function to check if two line segments are parallel
def areParallel (p1 p2 q1 q2 : ProjectedPoint) : Prop :=
  sorry

-- Theorem statement
theorem shadows_parallel 
  (A B C : Point3D) 
  (plane1 plane2 : ℝ) 
  (light : ParallelLight) :
  let A1 := project A plane1 light
  let A2 := project A plane2 light
  let B1 := project B plane1 light
  let B2 := project B plane2 light
  let C1 := project C plane1 light
  let C2 := project C plane2 light
  areParallel A1 A2 B1 B2 ∧ areParallel B1 B2 C1 C2 :=
sorry

end NUMINAMATH_CALUDE_shadows_parallel_l717_71775


namespace NUMINAMATH_CALUDE_lara_flowers_in_vase_l717_71728

/-- Calculates the number of flowers Lara put in the vase --/
def flowersInVase (totalFlowers : ℕ) (toMom : ℕ) (extraToGrandma : ℕ) : ℕ :=
  let toGrandma := toMom + extraToGrandma
  let remainingAfterMomAndGrandma := totalFlowers - toMom - toGrandma
  let toSister := remainingAfterMomAndGrandma / 3
  let toBestFriend := toSister + toSister / 4
  remainingAfterMomAndGrandma - toSister - toBestFriend

theorem lara_flowers_in_vase :
  flowersInVase 52 15 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_lara_flowers_in_vase_l717_71728


namespace NUMINAMATH_CALUDE_special_line_equation_l717_71767

/-- A line passing through point A(-3, 4) with x-intercept twice the y-intercept -/
structure SpecialLine where
  -- The slope-intercept form of the line: y = mx + b
  slope : ℝ
  y_intercept : ℝ
  -- The line passes through (-3, 4)
  point_condition : 4 = slope * (-3) + y_intercept
  -- The x-intercept is twice the y-intercept
  intercept_condition : -2 * y_intercept = y_intercept / slope

/-- The equation of the special line is either 3y + 4x = 0 or 2x - y - 5 = 0 -/
theorem special_line_equation (L : SpecialLine) :
  (3 * L.slope + 4 = 0 ∧ 3 * L.y_intercept = 0) ∨
  (2 = L.slope ∧ -5 = L.y_intercept) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l717_71767


namespace NUMINAMATH_CALUDE_max_draws_until_white_l717_71744

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : Nat
  white : Nat

/-- Represents the process of drawing balls from the bag -/
def drawUntilWhite (bag : BagContents) : Nat :=
  sorry

/-- Theorem stating the maximum number of draws needed -/
theorem max_draws_until_white (bag : BagContents) 
  (h1 : bag.red = 6) 
  (h2 : bag.white = 5) : 
  drawUntilWhite bag ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_draws_until_white_l717_71744


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l717_71776

/-- The trajectory of the midpoint of chords passing through the origin of a circle --/
theorem midpoint_trajectory (x y : ℝ) :
  (0 < x) → (x ≤ 1) →
  (∃ (a b : ℝ), (a - 1)^2 + b^2 = 1 ∧ (x = a/2) ∧ (y = b/2)) →
  (x - 1/2)^2 + y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l717_71776


namespace NUMINAMATH_CALUDE_smallest_integer_solution_two_is_smallest_l717_71723

theorem smallest_integer_solution (y : ℤ) : (10 - 5 * y < 5) ↔ y ≥ 2 := by sorry

theorem two_is_smallest : ∃ (y : ℤ), (10 - 5 * y < 5) ∧ (∀ (z : ℤ), 10 - 5 * z < 5 → z ≥ y) ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_two_is_smallest_l717_71723


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l717_71792

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  average : Rat

/-- Calculates the new average after an additional inning -/
def newAverage (bp : BatsmanPerformance) (newRuns : Nat) : Rat :=
  (bp.totalRuns + newRuns) / (bp.innings + 1)

/-- Theorem: Given the conditions, prove that the batsman's average after the 17th inning is 39 -/
theorem batsman_average_after_17th_inning 
  (bp : BatsmanPerformance)
  (h1 : bp.innings = 16)
  (h2 : newAverage bp 87 = bp.average + 3)
  : newAverage bp 87 = 39 := by
  sorry

#check batsman_average_after_17th_inning

end NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l717_71792


namespace NUMINAMATH_CALUDE_inequality_range_l717_71746

theorem inequality_range (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_mn : m + n + 3 = m * n) :
  (∀ x : ℝ, (m + n) * x^2 + 2 * x + m * n - 13 ≥ 0) ↔ 
  (∀ x : ℝ, x ≤ -1 ∨ x ≥ 2/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l717_71746


namespace NUMINAMATH_CALUDE_weight_replacement_l717_71725

theorem weight_replacement (W : ℝ) (original_weight replaced_weight : ℝ) : 
  W / 10 + 2.5 = (W - replaced_weight + 75) / 10 → replaced_weight = 50 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l717_71725


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l717_71707

open Real

/-- The function f(x) = x ln x is monotonically decreasing on the interval (0, 1/e) -/
theorem f_decreasing_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < (1 : ℝ)/Real.exp 1 →
  x₁ * log x₁ > x₂ * log x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l717_71707


namespace NUMINAMATH_CALUDE_candy_mixture_price_l717_71702

/-- Given two types of candy mixed to produce a mixture with known total weight and value,
    prove that the price of the second candy is $4.30 per pound. -/
theorem candy_mixture_price (x : ℝ) :
  x > 0 ∧
  x + 6.25 = 10 ∧
  3.5 * x + 6.25 * 4.3 = 4 * 10 →
  4.3 = (4 * 10 - 3.5 * x) / 6.25 :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_price_l717_71702


namespace NUMINAMATH_CALUDE_product_equals_result_l717_71791

theorem product_equals_result : ∃ x : ℝ, 469158 * x = 4691110842 ∧ x = 10000.2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_result_l717_71791


namespace NUMINAMATH_CALUDE_initial_lives_count_l717_71756

/-- Proves that if a person loses 6 lives, then gains 37 lives, and ends up with 41 lives, they must have started with 10 lives. -/
theorem initial_lives_count (initial_lives : ℕ) : 
  initial_lives - 6 + 37 = 41 → initial_lives = 10 := by
  sorry

#check initial_lives_count

end NUMINAMATH_CALUDE_initial_lives_count_l717_71756


namespace NUMINAMATH_CALUDE_fraction_problem_l717_71749

theorem fraction_problem (x : ℚ) : x = 3/5 ↔ (2/5 * 300 : ℚ) - (x * 125) = 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l717_71749


namespace NUMINAMATH_CALUDE_solve_for_a_l717_71796

theorem solve_for_a : ∃ a : ℝ, (1/2 * 2 + a = -1) ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l717_71796


namespace NUMINAMATH_CALUDE_jersey_tshirt_cost_difference_l717_71772

/-- Calculates the final cost difference between a jersey and a t-shirt --/
theorem jersey_tshirt_cost_difference :
  let jersey_price : ℚ := 115
  let tshirt_price : ℚ := 25
  let jersey_discount : ℚ := 10 / 100
  let tshirt_discount : ℚ := 15 / 100
  let sales_tax : ℚ := 8 / 100
  let jersey_shipping : ℚ := 5
  let tshirt_shipping : ℚ := 3

  let jersey_discounted := jersey_price * (1 - jersey_discount)
  let tshirt_discounted := tshirt_price * (1 - tshirt_discount)

  let jersey_with_tax := jersey_discounted * (1 + sales_tax)
  let tshirt_with_tax := tshirt_discounted * (1 + sales_tax)

  let jersey_final := jersey_with_tax + jersey_shipping
  let tshirt_final := tshirt_with_tax + tshirt_shipping

  jersey_final - tshirt_final = 90.83 := by sorry

end NUMINAMATH_CALUDE_jersey_tshirt_cost_difference_l717_71772


namespace NUMINAMATH_CALUDE_smaller_circles_radius_l717_71740

/-- Configuration of circles -/
structure CircleConfiguration where
  centralRadius : ℝ
  smallerRadius : ℝ
  numSmallerCircles : ℕ

/-- Defines a valid configuration of circles -/
def isValidConfiguration (config : CircleConfiguration) : Prop :=
  config.centralRadius = 1 ∧
  config.numSmallerCircles = 6 ∧
  -- Each smaller circle touches two others and the central circle
  -- (This condition is implicit in the geometry of the problem)
  True

/-- Theorem stating the radius of smaller circles in the given configuration -/
theorem smaller_circles_radius (config : CircleConfiguration)
  (h : isValidConfiguration config) :
  config.smallerRadius = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circles_radius_l717_71740


namespace NUMINAMATH_CALUDE_max_unique_sundaes_l717_71784

/-- The number of ice cream flavors --/
def num_flavors : ℕ := 8

/-- The number of flavors that must be served together --/
def num_paired_flavors : ℕ := 2

/-- The number of distinct choices after pairing --/
def num_choices : ℕ := num_flavors - num_paired_flavors + 1

/-- The number of scoops in a sundae --/
def scoops_per_sundae : ℕ := 2

theorem max_unique_sundaes :
  (Nat.choose (num_choices - 1) (scoops_per_sundae - 1)) + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_unique_sundaes_l717_71784


namespace NUMINAMATH_CALUDE_no_integer_coefficients_l717_71778

theorem no_integer_coefficients : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_coefficients_l717_71778


namespace NUMINAMATH_CALUDE_fraction_inequality_l717_71759

theorem fraction_inequality (a b c d : ℕ+) 
  (h1 : a + c ≤ 1982)
  (h2 : (a : ℚ) / b + (c : ℚ) / d < 1) :
  1 - (a : ℚ) / b - (c : ℚ) / d > 1 / (1983 ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l717_71759


namespace NUMINAMATH_CALUDE_middle_number_calculation_l717_71757

theorem middle_number_calculation (n : ℕ) (total_avg first_avg last_avg : ℚ) : 
  n = 11 →
  total_avg = 9.9 →
  first_avg = 10.5 →
  last_avg = 11.4 →
  ∃ (middle : ℚ), 
    middle = 22.5 ∧
    n * total_avg = (n / 2 : ℚ) * first_avg + (n / 2 : ℚ) * last_avg - middle :=
by
  sorry

end NUMINAMATH_CALUDE_middle_number_calculation_l717_71757


namespace NUMINAMATH_CALUDE_younger_person_age_l717_71748

theorem younger_person_age (y e : ℕ) : 
  e = y + 20 →                  -- The ages differ by 20 years
  e - 8 = 5 * (y - 8) →         -- 8 years ago, elder was 5 times younger's age
  y = 13                        -- The younger person's age is 13
  := by sorry

end NUMINAMATH_CALUDE_younger_person_age_l717_71748


namespace NUMINAMATH_CALUDE_fraction_of_cats_l717_71781

theorem fraction_of_cats (total_animals : ℕ) (total_dog_legs : ℕ) : 
  total_animals = 300 →
  total_dog_legs = 400 →
  (2 : ℚ) / 3 = (total_animals - (total_dog_legs / 4)) / total_animals :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_cats_l717_71781


namespace NUMINAMATH_CALUDE_complex_multiplication_l717_71787

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l717_71787


namespace NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l717_71758

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := -x + 2*y - 4*x*y
def B (x y : ℝ) : ℝ := -3*x - y + x*y

-- Theorem 1: Simplification of 2A - 3B
theorem simplify_2A_minus_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7*x + 7*y - 11*x*y := by sorry

-- Theorem 2: Value of 2A - 3B under specific conditions
theorem value_2A_minus_3B_specific (x y : ℝ) 
  (h1 : x + y = 6/7) (h2 : x * y = -2) :
  2 * A x y - 3 * B x y = 28 := by sorry

-- Theorem 3: Value of 2A - 3B when independent of y
theorem value_2A_minus_3B_independent (x : ℝ) 
  (h : ∀ y : ℝ, 2 * A x y - 3 * B x y = 2 * A x 0 - 3 * B x 0) :
  2 * A x 0 - 3 * B x 0 = 49/11 := by sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l717_71758


namespace NUMINAMATH_CALUDE_line_up_count_distribution_count_l717_71774

/-- Represents a student --/
inductive Student : Type
| A
| B
| C
| D
| E

/-- Represents a line-up of students --/
def LineUp := List Student

/-- Represents a distribution of students into classes --/
def Distribution := List (List Student)

/-- Checks if two students are adjacent in a line-up --/
def areAdjacent (s1 s2 : Student) (lineup : LineUp) : Prop := sorry

/-- Checks if a distribution is valid (three non-empty classes) --/
def isValidDistribution (d : Distribution) : Prop := sorry

/-- Counts the number of valid line-ups --/
def countValidLineUps : Nat := sorry

/-- Counts the number of valid distributions --/
def countValidDistributions : Nat := sorry

theorem line_up_count :
  countValidLineUps = 12 := by sorry

theorem distribution_count :
  countValidDistributions = 150 := by sorry

end NUMINAMATH_CALUDE_line_up_count_distribution_count_l717_71774


namespace NUMINAMATH_CALUDE_max_k_value_l717_71765

theorem max_k_value (k : ℤ) : 
  (∀ x : ℝ, x > 1 → x * Real.log x - k * x > 3) → k ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l717_71765


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l717_71762

/-- The number of dots in each row and column of the square array -/
def gridSize : ℕ := 5

/-- The number of different rectangles that can be formed in the grid -/
def numberOfRectangles : ℕ := (gridSize.choose 2) * (gridSize.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_5x5_grid : numberOfRectangles = 100 := by sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l717_71762


namespace NUMINAMATH_CALUDE_class_size_proof_l717_71752

theorem class_size_proof (x : ℕ) 
  (h1 : x > 3)
  (h2 : (85 - 78) / (x - 3 : ℝ) = 0.75) : 
  x = 13 := by
sorry

end NUMINAMATH_CALUDE_class_size_proof_l717_71752


namespace NUMINAMATH_CALUDE_coffee_package_size_l717_71718

theorem coffee_package_size (total_coffee : ℝ) (known_size : ℝ) (extra_known : ℕ) (unknown_count : ℕ) :
  total_coffee = 85 ∧ 
  known_size = 5 ∧ 
  extra_known = 2 ∧ 
  unknown_count = 5 → 
  ∃ (unknown_size : ℝ), 
    unknown_size * unknown_count + known_size * (unknown_count + extra_known) = total_coffee ∧ 
    unknown_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_coffee_package_size_l717_71718


namespace NUMINAMATH_CALUDE_not_prime_n_l717_71750

theorem not_prime_n (p a b c n : ℕ) : 
  Nat.Prime p → 
  0 < a → 0 < b → 0 < c → 0 < n →
  a < p → b < p → c < p →
  p^2 ∣ (a + (n-1) * b) →
  p^2 ∣ (b + (n-1) * c) →
  p^2 ∣ (c + (n-1) * a) →
  ¬(Nat.Prime n) :=
by sorry


end NUMINAMATH_CALUDE_not_prime_n_l717_71750


namespace NUMINAMATH_CALUDE_house_rooms_count_l717_71794

/-- The number of rooms with 4 walls -/
def rooms_with_four_walls : ℕ := 5

/-- The number of rooms with 5 walls -/
def rooms_with_five_walls : ℕ := 4

/-- The number of walls each person should paint -/
def walls_per_person : ℕ := 8

/-- The number of people in Amanda's family -/
def family_members : ℕ := 5

/-- The total number of rooms in the house -/
def total_rooms : ℕ := rooms_with_four_walls + rooms_with_five_walls

theorem house_rooms_count : total_rooms = 9 := by
  sorry

end NUMINAMATH_CALUDE_house_rooms_count_l717_71794


namespace NUMINAMATH_CALUDE_similarity_coefficients_are_valid_l717_71729

/-- A triangle with sides 2, 3, and 3 -/
structure OriginalTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  h1 : side1 = 2
  h2 : side2 = 3
  h3 : side3 = 3

/-- Similarity coefficients for the four triangles -/
structure SimilarityCoefficients where
  k1 : ℝ
  k2 : ℝ
  k3 : ℝ
  k4 : ℝ

/-- Predicate to check if the similarity coefficients are valid -/
def valid_coefficients (sc : SimilarityCoefficients) : Prop :=
  (sc.k1 = 1/2 ∧ sc.k2 = 1/2 ∧ sc.k3 = 1/2 ∧ sc.k4 = 1/2) ∨
  (sc.k1 = 6/13 ∧ sc.k2 = 4/13 ∧ sc.k3 = 9/13 ∧ sc.k4 = 6/13)

/-- Theorem stating that the similarity coefficients for the divided triangles are valid -/
theorem similarity_coefficients_are_valid (t : OriginalTriangle) (sc : SimilarityCoefficients) :
  valid_coefficients sc := by sorry

end NUMINAMATH_CALUDE_similarity_coefficients_are_valid_l717_71729


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l717_71720

/-- A geometric sequence with first four terms a, x, b, 2x -/
structure GeometricSequence (α : Type*) [Field α] where
  a : α
  x : α
  b : α

/-- The ratio between consecutive terms in a geometric sequence is constant -/
def is_geometric_sequence {α : Type*} [Field α] (seq : GeometricSequence α) : Prop :=
  seq.x / seq.a = seq.b / seq.x ∧ seq.b / seq.x = 2

theorem ratio_a_to_b {α : Type*} [Field α] (seq : GeometricSequence α) 
  (h : is_geometric_sequence seq) : seq.a / seq.b = 1 / 4 := by
  sorry

#check ratio_a_to_b

end NUMINAMATH_CALUDE_ratio_a_to_b_l717_71720


namespace NUMINAMATH_CALUDE_function_maximum_l717_71741

/-- The function f(x) = x + 4/x for x < 0 has a maximum value of -4 -/
theorem function_maximum (x : ℝ) (h : x < 0) : 
  x + 4 / x ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_function_maximum_l717_71741


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l717_71705

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define the theorem
theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (l m n : ℕ) (a' b c : ℝ) 
  (h_arithmetic : is_arithmetic_sequence a)
  (h_l : a l = 1 / a')
  (h_m : a m = 1 / b)
  (h_n : a n = 1 / c) :
  (l - m : ℝ) * a' * b + (m - n : ℝ) * b * c + (n - l : ℝ) * c * a' = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l717_71705


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l717_71730

/-- An isosceles trapezoid with perpendicular diagonals -/
structure IsoscelesTrapezoid where
  /-- The length of the midsegment of the trapezoid -/
  midsegment : ℝ
  /-- The diagonals of the trapezoid are perpendicular -/
  diagonals_perpendicular : Bool
  /-- The trapezoid is isosceles -/
  isosceles : Bool

/-- The area of an isosceles trapezoid with perpendicular diagonals -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem: The area of an isosceles trapezoid with perpendicular diagonals 
    and midsegment of length 5 is 25 -/
theorem isosceles_trapezoid_area 
  (t : IsoscelesTrapezoid) 
  (h1 : t.midsegment = 5) 
  (h2 : t.diagonals_perpendicular = true) 
  (h3 : t.isosceles = true) : 
  area t = 25 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l717_71730


namespace NUMINAMATH_CALUDE_dad_jayson_age_ratio_l717_71734

/-- Represents the ages and relationships in Jayson's family -/
structure Family where
  jayson_age : ℕ
  mom_age : ℕ
  dad_age : ℕ
  mom_age_at_birth : ℕ

/-- The conditions given in the problem -/
def problem_conditions (f : Family) : Prop :=
  f.jayson_age = 10 ∧
  f.mom_age = f.mom_age_at_birth + f.jayson_age ∧
  f.dad_age = f.mom_age + 2 ∧
  f.mom_age_at_birth = 28

/-- The theorem stating the ratio of Jayson's dad's age to Jayson's age -/
theorem dad_jayson_age_ratio (f : Family) :
  problem_conditions f → (f.dad_age : ℚ) / f.jayson_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_dad_jayson_age_ratio_l717_71734


namespace NUMINAMATH_CALUDE_tmall_transaction_scientific_notation_l717_71733

theorem tmall_transaction_scientific_notation :
  let transaction_volume : ℝ := 2135 * 10^9
  transaction_volume = 2.135 * 10^11 := by
  sorry

end NUMINAMATH_CALUDE_tmall_transaction_scientific_notation_l717_71733


namespace NUMINAMATH_CALUDE_min_roots_symmetric_function_l717_71771

/-- A function with specific symmetry properties -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 - x) = f (2 + x)) ∧
  (∀ x, f (7 - x) = f (7 + x)) ∧
  f 0 = 0

/-- The set of roots of f in the interval [-1000, 1000] -/
def RootSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x ∈ Set.Icc (-1000) 1000 ∧ f x = 0}

/-- The theorem stating the minimum number of roots -/
theorem min_roots_symmetric_function (f : ℝ → ℝ) (h : SymmetricFunction f) :
    401 ≤ (RootSet f).ncard := by
  sorry

end NUMINAMATH_CALUDE_min_roots_symmetric_function_l717_71771


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l717_71773

theorem simplify_and_evaluate :
  let a : ℝ := (1/2 : ℝ) + Real.sqrt (1/2)
  (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - 6) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l717_71773


namespace NUMINAMATH_CALUDE_jason_attended_twelve_games_l717_71703

def games_attended (planned_this_month : ℕ) (planned_last_month : ℕ) (missed : ℕ) : ℕ :=
  planned_this_month + planned_last_month - missed

theorem jason_attended_twelve_games :
  games_attended 11 17 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_jason_attended_twelve_games_l717_71703


namespace NUMINAMATH_CALUDE_kids_to_adult_ticket_ratio_l717_71742

def admission_price : ℝ := 30
def group_size : ℕ := 10
def num_children : ℕ := 4
def num_adults : ℕ := group_size - num_children
def discount_rate : ℝ := 0.2
def soda_price : ℝ := 5
def total_paid : ℝ := 197

def adult_ticket_price : ℝ := admission_price

theorem kids_to_adult_ticket_ratio :
  ∃ (kids_ticket_price : ℝ),
    kids_ticket_price > 0 ∧
    adult_ticket_price > 0 ∧
    (1 - discount_rate) * (num_adults * adult_ticket_price + num_children * kids_ticket_price) + soda_price = total_paid ∧
    kids_ticket_price / adult_ticket_price = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_kids_to_adult_ticket_ratio_l717_71742


namespace NUMINAMATH_CALUDE_hyperbola_equation_l717_71766

/-- Given a hyperbola with asymptotes y = ± 2(x-1) and one focus at (1+2√5, 0),
    prove that its equation is (x - 1)²/5 - y²/20 = 1 -/
theorem hyperbola_equation 
  (asymptotes : ℝ → ℝ → Prop)
  (focus : ℝ × ℝ)
  (h_asymptotes : ∀ x y, asymptotes x y ↔ y = 2*(x-1) ∨ y = -2*(x-1))
  (h_focus : focus = (1 + 2*Real.sqrt 5, 0)) :
  ∀ x y, ((x - 1)^2 / 5 - y^2 / 20 = 1) ↔ 
    (∃ a b c : ℝ, a*(x-1)^2 + b*y^2 + c = 0 ∧ 
    (∀ x' y', asymptotes x' y' → a*(x'-1)^2 + b*y'^2 + c = 0) ∧
    a*(focus.1-1)^2 + b*focus.2^2 + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l717_71766


namespace NUMINAMATH_CALUDE_eccentricity_decreases_as_a_increases_ellipse_approaches_circle_l717_71719

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  h_a_pos : 1 < a
  h_a_bound : a < 2 + Real.sqrt 5

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.a^2 - 1) / (4 * e.a))

/-- Theorem: As 'a' increases, the eccentricity decreases -/
theorem eccentricity_decreases_as_a_increases (e1 e2 : Ellipse) 
    (h : e1.a < e2.a) : eccentricity e2 < eccentricity e1 := by
  sorry

/-- Corollary: As 'a' increases, the ellipse becomes closer to a circle -/
theorem ellipse_approaches_circle (e1 e2 : Ellipse) (h : e1.a < e2.a) :
    ∃ (c : ℝ), 0 < c ∧ c < 1 ∧ eccentricity e2 < c * eccentricity e1 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_decreases_as_a_increases_ellipse_approaches_circle_l717_71719


namespace NUMINAMATH_CALUDE_red_rose_value_l717_71768

def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def selling_price : ℚ := 75

def roses : ℕ := total_flowers - tulips
def red_roses : ℕ := roses - white_roses
def roses_to_sell : ℕ := red_roses / 2

theorem red_rose_value (total_flowers tulips white_roses selling_price : ℕ) 
  (h1 : total_flowers = 400)
  (h2 : tulips = 120)
  (h3 : white_roses = 80)
  (h4 : selling_price = 75) :
  (selling_price : ℚ) / roses_to_sell = 3/4 := by
  sorry

#eval (75 : ℚ) / 100  -- To verify the result is indeed 0.75

end NUMINAMATH_CALUDE_red_rose_value_l717_71768


namespace NUMINAMATH_CALUDE_impossible_30_gon_numbering_l717_71717

theorem impossible_30_gon_numbering : ¬ ∃ (f : Fin 30 → Nat),
  (∀ i, f i ∈ Finset.range 30) ∧
  (∀ i, f i ≠ 0) ∧
  (∀ i j, i ≠ j → f i ≠ f j) ∧
  (∀ i : Fin 30, ∃ k : Nat, (f i + f ((i + 1) % 30) : Nat) = k^2) := by
  sorry

end NUMINAMATH_CALUDE_impossible_30_gon_numbering_l717_71717


namespace NUMINAMATH_CALUDE_inequality_equivalence_l717_71722

theorem inequality_equivalence (a : ℝ) : 
  (∀ x, (4*x + a)/3 > 1 ↔ -((2*x + 1)/2) < 0) → a ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l717_71722


namespace NUMINAMATH_CALUDE_digit_sum_divisibility_27_l717_71761

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem digit_sum_divisibility_27 : 
  ∃ n : ℕ, (sum_of_digits n % 27 = 0) ∧ (n % 27 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_digit_sum_divisibility_27_l717_71761


namespace NUMINAMATH_CALUDE_donny_gas_change_l717_71799

/-- Calculates the change Donny receives after filling up his truck's gas tank. -/
theorem donny_gas_change (tank_capacity : ℕ) (initial_fuel : ℕ) (fuel_cost : ℕ) (payment : ℕ) : 
  tank_capacity = 150 →
  initial_fuel = 38 →
  fuel_cost = 3 →
  payment = 350 →
  payment - (tank_capacity - initial_fuel) * fuel_cost = 14 := by
  sorry

#check donny_gas_change

end NUMINAMATH_CALUDE_donny_gas_change_l717_71799


namespace NUMINAMATH_CALUDE_remainder_theorem_application_l717_71739

theorem remainder_theorem_application (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D * x^6 + E * x^4 + F * x^2 + 6
  (q 2 = 16) → (q (-2) = 16) := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_application_l717_71739


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l717_71737

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l717_71737


namespace NUMINAMATH_CALUDE_matilda_jellybeans_l717_71726

/-- Given that:
    1. Matilda has half as many jellybeans as Matt
    2. Matt has ten times as many jellybeans as Steve
    3. Steve has 84 jellybeans
    Prove that Matilda has 420 jellybeans. -/
theorem matilda_jellybeans (steve_jellybeans : ℕ) (matt_jellybeans : ℕ) (matilda_jellybeans : ℕ)
  (h1 : steve_jellybeans = 84)
  (h2 : matt_jellybeans = 10 * steve_jellybeans)
  (h3 : matilda_jellybeans = matt_jellybeans / 2) :
  matilda_jellybeans = 420 := by
  sorry

end NUMINAMATH_CALUDE_matilda_jellybeans_l717_71726


namespace NUMINAMATH_CALUDE_star_2_5_star_neg2_neg5_l717_71710

-- Define the ★ operation
def star (a b : Int) : Int := a * b - a - b + 1

-- Theorem statements
theorem star_2_5 : star 2 5 = 4 := by sorry

theorem star_neg2_neg5 : star (-2) (-5) = 18 := by sorry

end NUMINAMATH_CALUDE_star_2_5_star_neg2_neg5_l717_71710


namespace NUMINAMATH_CALUDE_f_inequality_l717_71779

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

theorem f_inequality (a : ℝ) : 
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → ((x + 1) * Real.log x + 2 * a) / ((x + 1)^2) < Real.log x / (x - 1)) ↔ 
  a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_f_inequality_l717_71779


namespace NUMINAMATH_CALUDE_original_price_calculation_l717_71780

/-- Given an item sold at a 20% loss with a selling price of 480, prove that the original price was 600. -/
theorem original_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 480 → 
  loss_percentage = 20 → 
  ∃ original_price : ℝ, 
    selling_price = original_price * (1 - loss_percentage / 100) ∧ 
    original_price = 600 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l717_71780


namespace NUMINAMATH_CALUDE_chord_existence_l717_71795

/-- A continuous curve in a 2D plane -/
def ContinuousCurve := Set (ℝ × ℝ)

/-- Defines if a curve connects two points -/
def connects (curve : ContinuousCurve) (A B : ℝ × ℝ) : Prop := sorry

/-- Defines if a curve has a chord of a given length parallel to a line segment -/
def has_parallel_chord (curve : ContinuousCurve) (A B : ℝ × ℝ) (length : ℝ) : Prop := sorry

/-- The distance between two points in 2D space -/
def distance (A B : ℝ × ℝ) : ℝ := sorry

theorem chord_existence (n : ℕ) (hn : n > 0) (A B : ℝ × ℝ) (curve : ContinuousCurve) :
  distance A B = 1 →
  connects curve A B →
  has_parallel_chord curve A B (1 / n) := by sorry

end NUMINAMATH_CALUDE_chord_existence_l717_71795


namespace NUMINAMATH_CALUDE_combined_average_marks_average_marks_two_classes_l717_71721

/-- Given two classes with specified number of students and average marks,
    calculate the average mark of all students combined. -/
theorem combined_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  total_marks / total_students = (n1 * avg1 + n2 * avg2) / (n1 + n2) :=
by sorry

/-- The average marks of all students from two classes. -/
theorem average_marks_two_classes :
  let class1_students : ℕ := 30
  let class2_students : ℕ := 50
  let class1_avg : ℚ := 40
  let class2_avg : ℚ := 70
  let total_students := class1_students + class2_students
  let total_marks := class1_students * class1_avg + class2_students * class2_avg
  total_marks / total_students = 58.75 :=
by sorry

end NUMINAMATH_CALUDE_combined_average_marks_average_marks_two_classes_l717_71721


namespace NUMINAMATH_CALUDE_ones_digit_of_power_l717_71783

-- Define a function to get the ones digit of a natural number
def onesDigit (n : ℕ) : ℕ := n % 10

-- Define the exponent
def exponent : ℕ := 22 * (11^11)

-- Theorem statement
theorem ones_digit_of_power : onesDigit (22^exponent) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_power_l717_71783


namespace NUMINAMATH_CALUDE_f_402_equals_zero_l717_71716

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom period_condition : ∀ x, f (x + 4) - f x = 2 * f 2
axiom symmetry_condition : ∀ x, f (2 - x) = f x

-- Theorem to prove
theorem f_402_equals_zero : f 402 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_402_equals_zero_l717_71716


namespace NUMINAMATH_CALUDE_power_multiplication_l717_71735

theorem power_multiplication (n : ℕ) :
  3000 * (3000 ^ 3000) = 3000 ^ (3000 + 1) :=
by sorry

end NUMINAMATH_CALUDE_power_multiplication_l717_71735


namespace NUMINAMATH_CALUDE_sum_and_convert_l717_71715

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Adds two numbers in base 8 -/
def add_base8 (a b : ℕ) : ℕ := sorry

theorem sum_and_convert :
  let a := 1453
  let b := 567
  base8_to_base10 (add_base8 a b) = 1124 := by sorry

end NUMINAMATH_CALUDE_sum_and_convert_l717_71715


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l717_71709

/-- The perimeter of a rhombus with diagonals of 10 inches and 24 inches is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l717_71709


namespace NUMINAMATH_CALUDE_eldora_paper_clips_count_l717_71754

/-- The cost of one box of paper clips in dollars -/
def paper_clip_cost : ℚ := 185 / 100

/-- The total cost of Eldora's purchase in dollars -/
def eldora_total : ℚ := 5540 / 100

/-- The number of packages of index cards Eldora bought -/
def eldora_index_cards : ℕ := 7

/-- The total cost of Finn's purchase in dollars -/
def finn_total : ℚ := 6170 / 100

/-- The number of boxes of paper clips Finn bought -/
def finn_paper_clips : ℕ := 12

/-- The number of packages of index cards Finn bought -/
def finn_index_cards : ℕ := 10

/-- The number of boxes of paper clips Eldora bought -/
def eldora_paper_clips : ℕ := 15

theorem eldora_paper_clips_count :
  ∃ (index_card_cost : ℚ),
    index_card_cost * finn_index_cards + paper_clip_cost * finn_paper_clips = finn_total ∧
    index_card_cost * eldora_index_cards + paper_clip_cost * eldora_paper_clips = eldora_total :=
by sorry

end NUMINAMATH_CALUDE_eldora_paper_clips_count_l717_71754


namespace NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l717_71782

theorem tan_beta_minus_2alpha (α β : ℝ) 
  (h1 : Real.tan α = 1 / 2) 
  (h2 : Real.tan (α - β) = -1 / 3) : 
  Real.tan (β - 2 * α) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l717_71782


namespace NUMINAMATH_CALUDE_largest_n_for_product_l717_71706

/-- Arithmetic sequence (a_n) with initial value 1 and common difference x -/
def a (n : ℕ) (x : ℤ) : ℤ := 1 + (n - 1 : ℤ) * x

/-- Arithmetic sequence (b_n) with initial value 1 and common difference y -/
def b (n : ℕ) (y : ℤ) : ℤ := 1 + (n - 1 : ℤ) * y

theorem largest_n_for_product (x y : ℤ) (hx : x > 0) (hy : y > 0) 
  (h_a2_b2 : 1 < a 2 x ∧ a 2 x ≤ b 2 y) :
  (∃ n : ℕ, a n x * b n y = 1764) →
  (∀ m : ℕ, a m x * b m y = 1764 → m ≤ 44) ∧
  (∃ n : ℕ, a n x * b n y = 1764 ∧ n = 44) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_product_l717_71706


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l717_71713

-- Define a line in slope-intercept form (y = mx + b)
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.b

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.m = l2.m

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l.equation p.x p.y

-- The given line
def given_line : Line :=
  { m := -2, b := 3 }

-- The point (0, 1)
def point : Point :=
  { x := 0, y := 1 }

-- The theorem to prove
theorem parallel_line_through_point :
  ∃! l : Line, parallel l given_line ∧ passes_through l point ∧ l.equation 0 1 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l717_71713


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_two_l717_71790

theorem cube_root_sum_equals_two (x : ℝ) (h1 : x > 0) 
  (h2 : (2 - x^3)^(1/3) + (2 + x^3)^(1/3) = 2) : x^6 = 100/27 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_two_l717_71790


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l717_71753

theorem least_positive_integer_to_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (625 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (625 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l717_71753


namespace NUMINAMATH_CALUDE_probability_selecting_A_and_B_l717_71751

theorem probability_selecting_A_and_B : 
  let total_students : ℕ := 5
  let selected_students : ℕ := 3
  let total_combinations := Nat.choose total_students selected_students
  let favorable_combinations := Nat.choose (total_students - 2) (selected_students - 2)
  (favorable_combinations : ℚ) / total_combinations = 3 / 10 :=
sorry

end NUMINAMATH_CALUDE_probability_selecting_A_and_B_l717_71751


namespace NUMINAMATH_CALUDE_steve_salary_calculation_l717_71701

def steve_take_home_pay (salary : ℝ) (tax_rate : ℝ) (healthcare_rate : ℝ) (union_dues : ℝ) : ℝ :=
  salary - (salary * tax_rate) - (salary * healthcare_rate) - union_dues

theorem steve_salary_calculation :
  steve_take_home_pay 40000 0.20 0.10 800 = 27200 := by
  sorry

end NUMINAMATH_CALUDE_steve_salary_calculation_l717_71701


namespace NUMINAMATH_CALUDE_min_value_theorem_l717_71763

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y - 2 = 0) :
  2/x + 9/y ≥ 25/2 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l717_71763


namespace NUMINAMATH_CALUDE_coefficient_relation_l717_71764

/-- A polynomial function with specific properties -/
def g (a b c d e : ℝ) (x : ℝ) : ℝ := a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- Theorem stating the relationship between coefficients a and b -/
theorem coefficient_relation (a b c d e : ℝ) :
  (g a b c d e (-1) = 0) →
  (g a b c d e 0 = 0) →
  (g a b c d e 1 = 0) →
  (g a b c d e 2 = 0) →
  (g a b c d e 0 = 3) →
  b = -2*a := by sorry

end NUMINAMATH_CALUDE_coefficient_relation_l717_71764


namespace NUMINAMATH_CALUDE_hannah_stocking_stuffers_l717_71747

/-- The number of candy canes per stocking -/
def candy_canes : Nat := 4

/-- The number of beanie babies per stocking -/
def beanie_babies : Nat := 2

/-- The number of books per stocking -/
def books : Nat := 1

/-- The number of kids Hannah has -/
def num_kids : Nat := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stuffers : Nat := (candy_canes + beanie_babies + books) * num_kids

theorem hannah_stocking_stuffers :
  total_stuffers = 21 := by sorry

end NUMINAMATH_CALUDE_hannah_stocking_stuffers_l717_71747


namespace NUMINAMATH_CALUDE_peters_mothers_age_l717_71788

/-- Proves that Peter's mother's age is 60 given the problem conditions -/
theorem peters_mothers_age :
  ∀ (harriet_age peter_age mother_age : ℕ),
    harriet_age = 13 →
    peter_age + 4 = 2 * (harriet_age + 4) →
    peter_age = mother_age / 2 →
    mother_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_peters_mothers_age_l717_71788


namespace NUMINAMATH_CALUDE_polar_rectangular_equivalence_l717_71724

theorem polar_rectangular_equivalence (ρ θ x y : ℝ) :
  y = ρ * Real.sin θ →
  x = ρ * Real.cos θ →
  (y^2 = 12 * x) ↔ (ρ * Real.sin θ^2 = 12 * Real.cos θ) :=
sorry

end NUMINAMATH_CALUDE_polar_rectangular_equivalence_l717_71724


namespace NUMINAMATH_CALUDE_negation_equivalence_l717_71793

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Adult : U → Prop)
variable (GoodCook : U → Prop)

-- Define the statements
def AllAdultsAreGoodCooks : Prop := ∀ x, Adult x → GoodCook x
def AtLeastOneAdultIsBadCook : Prop := ∃ x, Adult x ∧ ¬GoodCook x

-- Theorem statement
theorem negation_equivalence : 
  AtLeastOneAdultIsBadCook U Adult GoodCook ↔ ¬(AllAdultsAreGoodCooks U Adult GoodCook) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l717_71793


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l717_71785

theorem max_value_trig_expression (x : ℝ) : 11 - 8 * Real.cos x - 2 * (Real.sin x)^2 ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l717_71785


namespace NUMINAMATH_CALUDE_quadratic_inequality_boundary_l717_71711

theorem quadratic_inequality_boundary (c : ℝ) : 
  (∀ x : ℝ, x * (3 * x + 1) < c ↔ -5/2 < x ∧ x < 3) ↔ c = 30 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_boundary_l717_71711


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l717_71727

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 12) : 
  a^3 + 1/a^3 = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l717_71727


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l717_71760

theorem cryptarithmetic_puzzle (A B C : ℕ) : 
  A + B + C = 10 →
  B + A + 1 = 10 →
  A + 1 = 3 →
  (A ≠ B ∧ A ≠ C ∧ B ≠ C) →
  C = 1 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l717_71760


namespace NUMINAMATH_CALUDE_z_sixth_power_l717_71769

theorem z_sixth_power (z : ℂ) : 
  z = (Real.sqrt 3 + Complex.I) / 2 → 
  z^6 = (1 + Real.sqrt 3) / 4 - ((Real.sqrt 3 + 1) / 8) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_z_sixth_power_l717_71769


namespace NUMINAMATH_CALUDE_subtraction_multiplication_problem_l717_71731

theorem subtraction_multiplication_problem (x : ℝ) : 
  8.9 - x = 3.1 → (x * 3.1) * 2.5 = 44.95 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_problem_l717_71731


namespace NUMINAMATH_CALUDE_expression_simplification_l717_71700

theorem expression_simplification : 
  ((0.3 * 0.8) / 0.2) + (0.1 * 0.5) ^ 2 - 1 / (0.5 * 0.8)^2 = -5.0475 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l717_71700


namespace NUMINAMATH_CALUDE_x_value_proof_l717_71704

theorem x_value_proof (x : ℕ) : 
  (Nat.lcm x 18 - Nat.gcd x 18 = 120) → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l717_71704


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l717_71770

/-- Proves that the salary of the new supervisor is $510 given the conditions of the problem -/
theorem new_supervisor_salary
  (initial_people : ℕ)
  (initial_average_salary : ℚ)
  (old_supervisor_salary : ℚ)
  (new_average_salary : ℚ)
  (h_initial_people : initial_people = 9)
  (h_initial_average : initial_average_salary = 430)
  (h_old_supervisor : old_supervisor_salary = 870)
  (h_new_average : new_average_salary = 390)
  : ∃ (new_supervisor_salary : ℚ),
    new_supervisor_salary = 510 ∧
    (initial_people - 1) * (initial_average_salary * initial_people - old_supervisor_salary) / (initial_people - 1) +
    new_supervisor_salary = new_average_salary * initial_people :=
sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_l717_71770


namespace NUMINAMATH_CALUDE_basketball_five_bounces_l717_71745

/-- Calculates the total distance traveled by a basketball dropped from a given height,
    rebounding to a fraction of its previous height, for a given number of bounces. -/
def basketballDistance (initialHeight : ℝ) (reboundFraction : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- Theorem stating that a basketball dropped from 80 feet, rebounding to three-quarters
    of its previous height each time, will have traveled 408.125 feet when it hits the
    ground for the fifth time. -/
theorem basketball_five_bounces :
  basketballDistance 80 0.75 5 = 408.125 := by sorry

end NUMINAMATH_CALUDE_basketball_five_bounces_l717_71745


namespace NUMINAMATH_CALUDE_probability_two_math_teachers_l717_71708

def english_teachers : ℕ := 3
def math_teachers : ℕ := 4
def social_teachers : ℕ := 2
def total_teachers : ℕ := english_teachers + math_teachers + social_teachers
def selected_members : ℕ := 2

theorem probability_two_math_teachers :
  (Nat.choose math_teachers selected_members : ℚ) / (Nat.choose total_teachers selected_members) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_math_teachers_l717_71708


namespace NUMINAMATH_CALUDE_jane_mean_score_l717_71732

def jane_scores : List ℕ := [95, 88, 94, 86, 92, 91]

theorem jane_mean_score :
  (jane_scores.sum / jane_scores.length : ℚ) = 91 := by sorry

end NUMINAMATH_CALUDE_jane_mean_score_l717_71732


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l717_71786

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |m - (9^3 + 7^3)^(1/3)| ≥ |n - (9^3 + 7^3)^(1/3)| :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l717_71786


namespace NUMINAMATH_CALUDE_stock_yield_calculation_l717_71714

theorem stock_yield_calculation (a_price b_price b_yield : ℝ) 
  (h1 : a_price = 96)
  (h2 : b_price = 115.2)
  (h3 : b_yield = 0.12)
  (h4 : a_price * b_yield = b_price * (a_yield : ℝ)) :
  a_yield = 0.10 :=
by
  sorry

end NUMINAMATH_CALUDE_stock_yield_calculation_l717_71714


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l717_71738

/-- A hyperbola with given parameters a and b -/
structure Hyperbola (a b : ℝ) :=
  (ha : a > 0)
  (hb : b > 0)

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola a b) : (ℝ × ℝ → Prop) × (ℝ × ℝ → Prop) := sorry

/-- A point is in the first quadrant -/
def is_in_first_quadrant (p : ℝ × ℝ) : Prop := sorry

/-- A point lies on a line -/
def lies_on (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop := sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : ℝ × ℝ → Prop) : Prop := sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : ℝ × ℝ → Prop) : Prop := sorry

/-- The line through two points -/
def line_through (p1 p2 : ℝ × ℝ) : ℝ × ℝ → Prop := sorry

theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (p : ℝ × ℝ) (hp1 : is_in_first_quadrant p) 
  (hp2 : lies_on p (asymptotes h).1) 
  (hp3 : perpendicular (line_through p (left_focus h)) (asymptotes h).2)
  (hp4 : parallel (line_through p (right_focus h)) (asymptotes h).2) :
  eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l717_71738
