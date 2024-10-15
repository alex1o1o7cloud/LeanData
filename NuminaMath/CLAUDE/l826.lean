import Mathlib

namespace NUMINAMATH_CALUDE_volunteer_distribution_theorem_l826_82696

/-- The number of ways to distribute volunteers among exits -/
def distribute_volunteers (num_volunteers : ℕ) (num_exits : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements -/
theorem volunteer_distribution_theorem :
  distribute_volunteers 5 4 = 240 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_theorem_l826_82696


namespace NUMINAMATH_CALUDE_rectangle_area_l826_82639

/-- Given a rectangle with diagonal length 2a + b, its area is 2ab -/
theorem rectangle_area (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = (2*a + b)^2 ∧ x * y = 2*a*b :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l826_82639


namespace NUMINAMATH_CALUDE_max_consecutive_semiprimes_l826_82679

def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def IsSemiPrime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, IsPrime p ∧ IsPrime q ∧ p ≠ q ∧ n = p + q

def ConsecutiveSemiPrimes (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → IsSemiPrime (k + 1)

theorem max_consecutive_semiprimes :
  ∀ n : ℕ, ConsecutiveSemiPrimes n → n ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_semiprimes_l826_82679


namespace NUMINAMATH_CALUDE_bee_hatch_count_l826_82649

/-- The number of bees hatching from the queen's eggs every day -/
def daily_hatch : ℕ := 3001

/-- The number of bees the queen loses every day -/
def daily_loss : ℕ := 900

/-- The number of days -/
def days : ℕ := 7

/-- The total number of bees in the hive after 7 days -/
def final_bees : ℕ := 27201

/-- The initial number of bees -/
def initial_bees : ℕ := 12500

/-- Theorem stating that the number of bees hatching daily is correct -/
theorem bee_hatch_count :
  initial_bees + days * (daily_hatch - daily_loss) = final_bees :=
by sorry

end NUMINAMATH_CALUDE_bee_hatch_count_l826_82649


namespace NUMINAMATH_CALUDE_arcsin_one_half_l826_82632

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_l826_82632


namespace NUMINAMATH_CALUDE_floor_of_e_l826_82681

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_e_l826_82681


namespace NUMINAMATH_CALUDE_store_products_l826_82695

theorem store_products (big_box_capacity small_box_capacity total_products : ℕ) 
  (h1 : big_box_capacity = 50)
  (h2 : small_box_capacity = 40)
  (h3 : total_products = 212) :
  ∃ (big_boxes small_boxes : ℕ), 
    big_boxes * big_box_capacity + small_boxes * small_box_capacity = total_products :=
by sorry

end NUMINAMATH_CALUDE_store_products_l826_82695


namespace NUMINAMATH_CALUDE_optimal_production_plan_l826_82674

/-- Represents the production plan for the factory -/
structure ProductionPlan where
  hoursA : ℝ  -- Hours to produce Product A
  hoursB : ℝ  -- Hours to produce Product B

/-- Calculates the total profit for a given production plan -/
def totalProfit (plan : ProductionPlan) : ℝ :=
  30 * plan.hoursA + 40 * plan.hoursB

/-- Checks if a production plan is feasible given the material constraints -/
def isFeasible (plan : ProductionPlan) : Prop :=
  3 * plan.hoursA + 2 * plan.hoursB ≤ 1200 ∧
  plan.hoursA + 2 * plan.hoursB ≤ 800 ∧
  plan.hoursA ≥ 0 ∧ plan.hoursB ≥ 0

/-- The optimal production plan -/
def optimalPlan : ProductionPlan :=
  { hoursA := 200, hoursB := 300 }

theorem optimal_production_plan :
  isFeasible optimalPlan ∧
  ∀ plan : ProductionPlan, isFeasible plan →
    totalProfit plan ≤ totalProfit optimalPlan ∧
  totalProfit optimalPlan = 18000 := by
  sorry


end NUMINAMATH_CALUDE_optimal_production_plan_l826_82674


namespace NUMINAMATH_CALUDE_regular_washes_count_l826_82677

/-- Represents the number of gallons of water used for different types of washes --/
structure WaterUsage where
  heavy : ℕ
  regular : ℕ
  light : ℕ

/-- Represents the number of different types of washes --/
structure Washes where
  heavy : ℕ
  regular : ℕ
  light : ℕ
  bleached : ℕ

/-- Calculates the total water usage for a given set of washes --/
def calculateWaterUsage (usage : WaterUsage) (washes : Washes) : ℕ :=
  usage.heavy * washes.heavy +
  usage.regular * washes.regular +
  usage.light * washes.light +
  usage.light * washes.bleached

/-- Theorem stating that there are 3 regular washes given the problem conditions --/
theorem regular_washes_count (usage : WaterUsage) (washes : Washes) :
  usage.heavy = 20 →
  usage.regular = 10 →
  usage.light = 2 →
  washes.heavy = 2 →
  washes.light = 1 →
  washes.bleached = 2 →
  calculateWaterUsage usage washes = 76 →
  washes.regular = 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_washes_count_l826_82677


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l826_82673

theorem two_numbers_with_given_means (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  Real.sqrt (a * b) = Real.sqrt 5 ∧ 
  (a + b) / 2 = 4 → 
  (a = 4 + Real.sqrt 11 ∧ b = 4 - Real.sqrt 11) ∨ 
  (a = 4 - Real.sqrt 11 ∧ b = 4 + Real.sqrt 11) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l826_82673


namespace NUMINAMATH_CALUDE_complex_equation_solution_l826_82671

theorem complex_equation_solution (z : ℂ) :
  Complex.abs z + z = 2 + 4 * Complex.I → z = -3 + 4 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l826_82671


namespace NUMINAMATH_CALUDE_approximation_equality_l826_82623

/-- For any function f, f(69.28 × 0.004) / 0.03 = f(9.237333...) -/
theorem approximation_equality (f : ℝ → ℝ) : f (69.28 * 0.004) / 0.03 = f 9.237333333333333 := by
  sorry

end NUMINAMATH_CALUDE_approximation_equality_l826_82623


namespace NUMINAMATH_CALUDE_perimeter_of_externally_touching_circles_l826_82680

/-- Given two externally touching circles with radii in the ratio 3:1 and a common external tangent
    of length 6√3, the perimeter of the figure formed by the external tangents and the external
    parts of the circles is 14π + 12√3. -/
theorem perimeter_of_externally_touching_circles (r R : ℝ) (h1 : R = 3 * r) 
    (h2 : r > 0) (h3 : 6 * Real.sqrt 3 = 2 * r * Real.sqrt 3) : 
    2 * (6 * Real.sqrt 3) + 2 * π * r * (1/3) + 2 * π * R * (2/3) = 14 * π + 12 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_externally_touching_circles_l826_82680


namespace NUMINAMATH_CALUDE_stone_145_is_2_l826_82692

/-- The number of stones in the arrangement -/
def num_stones : ℕ := 14

/-- The period of the counting sequence -/
def period : ℕ := 26

/-- The target count we're looking for -/
def target_count : ℕ := 145

/-- Function to convert the new count to the original stone number -/
def count_to_original (n : ℕ) : ℕ :=
  if n % period ≤ num_stones then n % period
  else period - (n % period) + 1

theorem stone_145_is_2 :
  count_to_original target_count = 2 := by sorry

end NUMINAMATH_CALUDE_stone_145_is_2_l826_82692


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_lt_neg_one_l826_82637

/-- Define set A as {x | -1 ≤ x < 2} -/
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}

/-- Define set B as {x | x ≤ a} -/
def B (a : ℝ) : Set ℝ := {x | x ≤ a}

/-- Theorem: The intersection of A and B is empty if and only if a < -1 -/
theorem intersection_empty_iff_a_lt_neg_one (a : ℝ) :
  A ∩ B a = ∅ ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_lt_neg_one_l826_82637


namespace NUMINAMATH_CALUDE_base_9_to_10_3562_l826_82621

def base_9_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

theorem base_9_to_10_3562 :
  base_9_to_10 [2, 6, 5, 3] = 2648 := by
  sorry

end NUMINAMATH_CALUDE_base_9_to_10_3562_l826_82621


namespace NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_when_disjoint_l826_82628

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_when_a_half :
  A (1/2) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

theorem range_of_a_when_disjoint (h1 : A a ≠ ∅) (h2 : A a ∩ B = ∅) :
  (-2 < a ∧ a ≤ 1/2) ∨ (a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_when_disjoint_l826_82628


namespace NUMINAMATH_CALUDE_farm_chicken_count_l826_82607

theorem farm_chicken_count :
  ∀ (num_hens num_roosters : ℕ),
    num_hens = 52 →
    num_roosters = num_hens + 16 →
    num_hens + num_roosters = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_chicken_count_l826_82607


namespace NUMINAMATH_CALUDE_shaded_area_constant_l826_82691

/-- The total area of two triangles formed by joining the ends of two 1 cm segments 
    on opposite sides of an 8 cm square is always 4 cm², regardless of the segments' positions. -/
theorem shaded_area_constant (h : ℝ) (h_range : 0 ≤ h ∧ h ≤ 8) : 
  (1/2 * 1 * h) + (1/2 * 1 * (8 - h)) = 4 := by sorry

end NUMINAMATH_CALUDE_shaded_area_constant_l826_82691


namespace NUMINAMATH_CALUDE_polynomial_value_l826_82610

theorem polynomial_value (a : ℝ) (h : a = Real.sqrt 17 - 1) : 
  a^5 + 2*a^4 - 17*a^3 - a^2 + 18*a - 17 = -1 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_l826_82610


namespace NUMINAMATH_CALUDE_expression_value_at_three_l826_82627

theorem expression_value_at_three : 
  let x : ℝ := 3
  x + x * (x^3 - x) = 75 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l826_82627


namespace NUMINAMATH_CALUDE_function_domain_implies_m_range_l826_82661

/-- The function f(x) = √(mx² - (1-m)x + m) has domain R if and only if m ≥ 1/3 -/
theorem function_domain_implies_m_range (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (m * x^2 - (1 - m) * x + m)) ↔ m ≥ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_implies_m_range_l826_82661


namespace NUMINAMATH_CALUDE_count_special_four_digit_numbers_l826_82653

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (ℕ × ℕ × ℕ × ℕ)

/-- Checks if a FourDigitNumber is valid (between 1000 and 9999) -/
def is_valid (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

/-- Converts a pair of digits to a two-digit number -/
def to_two_digit (a b : ℕ) : ℕ := 10 * a + b

/-- Checks if three two-digit numbers form an increasing arithmetic sequence -/
def is_increasing_arithmetic_seq (ab bc cd : ℕ) : Prop :=
  ab < bc ∧ bc < cd ∧ bc - ab = cd - bc

/-- The main theorem to be proved -/
theorem count_special_four_digit_numbers :
  (∃ (S : Finset FourDigitNumber),
    (∀ n ∈ S, is_valid n ∧ 
      let (a, b, c, d) := n
      is_increasing_arithmetic_seq (to_two_digit a b) (to_two_digit b c) (to_two_digit c d)) ∧
    S.card = 17 ∧
    (∀ n : FourDigitNumber, 
      is_valid n ∧ 
      let (a, b, c, d) := n
      is_increasing_arithmetic_seq (to_two_digit a b) (to_two_digit b c) (to_two_digit c d) 
      → n ∈ S)) := by
  sorry

end NUMINAMATH_CALUDE_count_special_four_digit_numbers_l826_82653


namespace NUMINAMATH_CALUDE_line_not_intersecting_segment_l826_82694

/-- Given points P and Q, and a line l that does not intersect line segment PQ,
    prove that the parameter m in the line equation satisfies m < -2/3 or m > 1/2 -/
theorem line_not_intersecting_segment (m : ℝ) :
  let P : ℝ × ℝ := (-1, 1)
  let Q : ℝ × ℝ := (2, 2)
  let l := {(x, y) : ℝ × ℝ | x + m * y + m = 0}
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) • P + t • Q ∉ l) →
  m < -2/3 ∨ m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_line_not_intersecting_segment_l826_82694


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l826_82625

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and an asymptote x/3 + y = 0 is √10/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b/a = 1/3) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 10 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l826_82625


namespace NUMINAMATH_CALUDE_average_book_cost_l826_82658

/-- Given that Fred had $236 initially, bought 6 books, and had $14 left after the purchase,
    prove that the average cost of each book is $37. -/
theorem average_book_cost (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) :
  initial_amount = 236 →
  num_books = 6 →
  remaining_amount = 14 →
  (initial_amount - remaining_amount) / num_books = 37 :=
by sorry

end NUMINAMATH_CALUDE_average_book_cost_l826_82658


namespace NUMINAMATH_CALUDE_cannot_form_flipped_shape_asymmetrical_shape_requires_flipping_asymmetrical_shape_cannot_be_formed_l826_82635

/-- Represents a rhombus with two colored triangles -/
structure ColoredRhombus :=
  (orientation : ℕ)  -- Represents the rotation (0, 90, 180, 270 degrees)

/-- Represents a larger shape composed of multiple rhombuses -/
structure LargerShape :=
  (rhombuses : List ColoredRhombus)

/-- Represents whether a shape requires flipping to be formed -/
def requiresFlipping (shape : LargerShape) : Prop :=
  sorry  -- Definition of when a shape requires flipping

/-- Represents whether a shape can be formed by rotation only -/
def canFormByRotationOnly (shape : LargerShape) : Prop :=
  sorry  -- Definition of when a shape can be formed by rotation only

/-- Theorem: A shape that requires flipping cannot be formed by rotation only -/
theorem cannot_form_flipped_shape
  (shape : LargerShape) :
  requiresFlipping shape → ¬(canFormByRotationOnly shape) :=
by sorry

/-- The asymmetrical shape that cannot be formed -/
def asymmetricalShape : LargerShape :=
  sorry  -- Definition of the specific asymmetrical shape

/-- Theorem: The asymmetrical shape requires flipping -/
theorem asymmetrical_shape_requires_flipping :
  requiresFlipping asymmetricalShape :=
by sorry

/-- Main theorem: The asymmetrical shape cannot be formed by rotation only -/
theorem asymmetrical_shape_cannot_be_formed :
  ¬(canFormByRotationOnly asymmetricalShape) :=
by sorry

end NUMINAMATH_CALUDE_cannot_form_flipped_shape_asymmetrical_shape_requires_flipping_asymmetrical_shape_cannot_be_formed_l826_82635


namespace NUMINAMATH_CALUDE_exists_m_for_all_x_m_range_when_exists_x_l826_82656

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Theorem 1: Existence of m such that m + f(x) > 0 for all x
theorem exists_m_for_all_x (m : ℝ) : 
  (∀ x, m + f x > 0) ↔ m > -2 := by sorry

-- Theorem 2: Range of m when there exists x such that m - f(x) > 0
theorem m_range_when_exists_x (m : ℝ) :
  (∃ x, m - f x > 0) → m > 2 := by sorry

end NUMINAMATH_CALUDE_exists_m_for_all_x_m_range_when_exists_x_l826_82656


namespace NUMINAMATH_CALUDE_f_negative_t_zero_l826_82606

theorem f_negative_t_zero (f : ℝ → ℝ) (t : ℝ) :
  (∀ x, f x = 3 * x + Real.sin x + 1) →
  f t = 2 →
  f (-t) = 0 := by
sorry

end NUMINAMATH_CALUDE_f_negative_t_zero_l826_82606


namespace NUMINAMATH_CALUDE_fifteen_by_fifteen_grid_toothpicks_l826_82654

/-- Calculates the number of toothpicks in a square grid with a missing corner --/
def toothpicks_in_grid (height : ℕ) (width : ℕ) : ℕ :=
  (height + 1) * width + (width + 1) * height - 1

/-- Theorem: A 15x15 square grid with a missing corner uses 479 toothpicks --/
theorem fifteen_by_fifteen_grid_toothpicks :
  toothpicks_in_grid 15 15 = 479 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_by_fifteen_grid_toothpicks_l826_82654


namespace NUMINAMATH_CALUDE_pencil_count_l826_82616

theorem pencil_count (num_pens : ℕ) (max_students : ℕ) (num_pencils : ℕ) :
  num_pens = 1001 →
  max_students = 91 →
  (∃ (s : ℕ), s ≤ max_students ∧ num_pens % s = 0 ∧ num_pencils % s = 0) →
  ∃ (k : ℕ), num_pencils = 91 * k :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l826_82616


namespace NUMINAMATH_CALUDE_ellipse_k_value_l826_82650

/-- The equation of an ellipse with parameter k -/
def ellipse_equation (k x y : ℝ) : Prop :=
  2 * k * x^2 + k * y^2 = 1

/-- The focus of the ellipse -/
def focus : ℝ × ℝ := (0, -4)

/-- Theorem stating that for an ellipse with the given equation and focus, k = 1/32 -/
theorem ellipse_k_value :
  ∃ (k : ℝ), k ≠ 0 ∧
  (∀ x y : ℝ, ellipse_equation k x y ↔ 2 * k * x^2 + k * y^2 = 1) ∧
  (∃ x y : ℝ, ellipse_equation k x y ∧ (x, y) = focus) ∧
  k = 1/32 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l826_82650


namespace NUMINAMATH_CALUDE_projection_implies_y_coordinate_l826_82676

/-- Given vectors a and b, if the projection of b in the direction of a is -√2, then the y-coordinate of b is 4. -/
theorem projection_implies_y_coordinate (a b : ℝ × ℝ) :
  a = (1, -1) →
  b.1 = 2 →
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt ((a.1 ^ 2 + a.2 ^ 2) : ℝ) = -Real.sqrt 2 →
  b.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_projection_implies_y_coordinate_l826_82676


namespace NUMINAMATH_CALUDE_expression_equals_seven_l826_82689

theorem expression_equals_seven :
  (-2023)^0 + Real.sqrt 4 - 2 * Real.sin (30 * π / 180) + abs (-5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_seven_l826_82689


namespace NUMINAMATH_CALUDE_new_year_firework_boxes_l826_82675

/-- Calculates the number of firework boxes used in a New Year's Eve display. -/
def firework_boxes_used (total_fireworks : ℕ) (fireworks_per_digit : ℕ) (fireworks_per_letter : ℕ) (fireworks_per_box : ℕ) (year_digits : ℕ) (phrase_letters : ℕ) : ℕ :=
  let year_fireworks := fireworks_per_digit * year_digits
  let phrase_fireworks := fireworks_per_letter * phrase_letters
  let remaining_fireworks := total_fireworks - (year_fireworks + phrase_fireworks)
  remaining_fireworks / fireworks_per_box

/-- The number of firework boxes used in the New Year's Eve display is 50. -/
theorem new_year_firework_boxes :
  firework_boxes_used 484 6 5 8 4 12 = 50 := by
  sorry

end NUMINAMATH_CALUDE_new_year_firework_boxes_l826_82675


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l826_82668

/-- The radius of an inscribed circle in a right triangle -/
theorem inscribed_circle_radius_right_triangle (a b c r : ℝ) 
  (h_right : a^2 + c^2 = b^2) -- Pythagorean theorem for right triangle
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) -- Positive side lengths
  : r = (a + c - b) / 2 ↔ 
    -- Definition of inscribed circle: 
    -- The circle touches all three sides of the triangle
    ∃ (x y : ℝ), 
      x > 0 ∧ y > 0 ∧
      x + y = b ∧
      x + r = c ∧
      y + r = a :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l826_82668


namespace NUMINAMATH_CALUDE_hare_wolf_distance_l826_82614

def track_length : ℝ := 200
def hare_speed : ℝ := 5
def wolf_speed : ℝ := 3
def time_elapsed : ℝ := 40

def distance_traveled (speed : ℝ) : ℝ := speed * time_elapsed

theorem hare_wolf_distance : 
  ∃ (initial_distance : ℝ), 
    (initial_distance = 40 ∨ initial_distance = 60) ∧
    (
      (distance_traveled hare_speed - distance_traveled wolf_speed) % track_length = 0 ∨
      (distance_traveled hare_speed - distance_traveled wolf_speed + initial_distance) % track_length = initial_distance
    ) :=
by sorry

end NUMINAMATH_CALUDE_hare_wolf_distance_l826_82614


namespace NUMINAMATH_CALUDE_village_population_problem_l826_82642

theorem village_population_problem (P : ℝ) : 
  (P * 1.3 * 0.7 = 13650) → P = 15000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l826_82642


namespace NUMINAMATH_CALUDE_spider_web_paths_spider_web_problem_l826_82615

theorem spider_web_paths : Nat → Nat → Nat
  | m, n => Nat.choose (m + n) m

theorem spider_web_problem : spider_web_paths 5 6 = 462 := by
  sorry

end NUMINAMATH_CALUDE_spider_web_paths_spider_web_problem_l826_82615


namespace NUMINAMATH_CALUDE_inequality_proof_l826_82665

theorem inequality_proof (a b c : ℝ) (h : a^6 + b^6 + c^6 = 3) :
  a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l826_82665


namespace NUMINAMATH_CALUDE_subset_condition_implies_a_values_l826_82646

theorem subset_condition_implies_a_values (a : ℝ) : 
  let A : Set ℝ := {x | x^2 = 1}
  let B : Set ℝ := {x | a * x = 1}
  B ⊆ A → a ∈ ({-1, 0, 1} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_subset_condition_implies_a_values_l826_82646


namespace NUMINAMATH_CALUDE_problem_solution_l826_82647

theorem problem_solution (x : ℝ) (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 12) :
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 200/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l826_82647


namespace NUMINAMATH_CALUDE_order_of_values_l826_82687

theorem order_of_values : 
  let a := Real.sin (80 * π / 180)
  let b := (1/2)⁻¹
  let c := Real.log 3 / Real.log (1/2)
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_values_l826_82687


namespace NUMINAMATH_CALUDE_derivative_at_one_l826_82667

def f (x : ℝ) (k : ℝ) : ℝ := x^3 - 2*k*x + 1

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, deriv f x = f' x) →
  (∃ k, ∀ x, f x = x^3 - 2*k*x + 1) →
  f' 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l826_82667


namespace NUMINAMATH_CALUDE_work_completion_time_l826_82682

/-- Given workers A and B, where A can complete a job in 15 days and B in 9 days,
    if A works for 5 days and then leaves, B will complete the remaining work in 6 days. -/
theorem work_completion_time (a_total_days b_total_days a_worked_days : ℕ) 
    (ha : a_total_days = 15)
    (hb : b_total_days = 9)
    (hw : a_worked_days = 5) : 
    (b_total_days : ℚ) * (1 - (a_worked_days : ℚ) / (a_total_days : ℚ)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l826_82682


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l826_82699

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 100) : 
  a * b = -3 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l826_82699


namespace NUMINAMATH_CALUDE_arithmetic_sum_example_l826_82684

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sum_example : arithmetic_sum 2 20 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_example_l826_82684


namespace NUMINAMATH_CALUDE_paiges_science_problems_l826_82690

theorem paiges_science_problems 
  (math_problems : ℕ) 
  (total_problems : ℕ → ℕ → ℕ) 
  (finished_problems : ℕ) 
  (remaining_problems : ℕ) 
  (h1 : math_problems = 43)
  (h2 : ∀ m s, total_problems m s = m + s)
  (h3 : finished_problems = 44)
  (h4 : remaining_problems = 11)
  (h5 : ∀ s, remaining_problems = total_problems math_problems s - finished_problems) :
  ∃ s : ℕ, s = 12 ∧ total_problems math_problems s = finished_problems + remaining_problems :=
sorry

end NUMINAMATH_CALUDE_paiges_science_problems_l826_82690


namespace NUMINAMATH_CALUDE_grandma_olga_grandchildren_l826_82604

-- Define the number of daughters and sons
def num_daughters : ℕ := 3
def num_sons : ℕ := 3

-- Define the number of children for each daughter and son
def sons_per_daughter : ℕ := 6
def daughters_per_son : ℕ := 5

-- Define the total number of grandchildren
def total_grandchildren : ℕ := num_daughters * sons_per_daughter + num_sons * daughters_per_son

-- Theorem statement
theorem grandma_olga_grandchildren : total_grandchildren = 33 := by
  sorry

end NUMINAMATH_CALUDE_grandma_olga_grandchildren_l826_82604


namespace NUMINAMATH_CALUDE_parallelogram_base_l826_82685

/-- Given a parallelogram with area 416 cm² and height 16 cm, its base is 26 cm. -/
theorem parallelogram_base (area height : ℝ) (h_area : area = 416) (h_height : height = 16) :
  area / height = 26 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l826_82685


namespace NUMINAMATH_CALUDE_max_value_of_a_l826_82620

theorem max_value_of_a (a b c d : ℤ) 
  (h1 : a < 2*b) 
  (h2 : b < 3*c) 
  (h3 : c < 4*d) 
  (h4 : d < 100) : 
  a ≤ 2367 ∧ ∃ (a₀ b₀ c₀ d₀ : ℤ), 
    a₀ < 2*b₀ ∧ 
    b₀ < 3*c₀ ∧ 
    c₀ < 4*d₀ ∧ 
    d₀ < 100 ∧ 
    a₀ = 2367 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l826_82620


namespace NUMINAMATH_CALUDE_manuscript_cost_calculation_l826_82640

/-- The total cost of typing a manuscript with given revision requirements -/
def manuscript_typing_cost (initial_cost : ℕ) (revision_cost : ℕ) (total_pages : ℕ) 
  (once_revised : ℕ) (twice_revised : ℕ) : ℕ :=
  (initial_cost * total_pages) + 
  (revision_cost * once_revised) + 
  (2 * revision_cost * twice_revised)

/-- Theorem stating the total cost of typing the manuscript -/
theorem manuscript_cost_calculation : 
  manuscript_typing_cost 6 4 100 35 15 = 860 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_calculation_l826_82640


namespace NUMINAMATH_CALUDE_traci_flour_amount_l826_82611

/-- The amount of flour Harris has in grams -/
def harris_flour : ℕ := 400

/-- The amount of flour needed for one cake in grams -/
def flour_per_cake : ℕ := 100

/-- The number of cakes Traci created -/
def traci_cakes : ℕ := 9

/-- The number of cakes Harris created -/
def harris_cakes : ℕ := 9

/-- The amount of flour Traci brought from her own house in grams -/
def traci_flour : ℕ := 1400

theorem traci_flour_amount :
  traci_flour = (flour_per_cake * (traci_cakes + harris_cakes)) - harris_flour :=
by sorry

end NUMINAMATH_CALUDE_traci_flour_amount_l826_82611


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l826_82618

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

theorem cubic_function_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0) →
  a ∈ Set.Iic (-3) ∪ Set.Ioi 6 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l826_82618


namespace NUMINAMATH_CALUDE_money_sharing_problem_l826_82608

theorem money_sharing_problem (total : ℕ) (amanda ben carlos : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 2 * (total / 13) →
  ben = 3 * (total / 13) →
  carlos = 8 * (total / 13) →
  ben = 30 →
  total = 130 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_problem_l826_82608


namespace NUMINAMATH_CALUDE_set_operations_and_range_l826_82645

def U := Set ℝ

def A : Set ℝ := {x | x ≥ 3}

def B : Set ℝ := {x | x^2 - 8*x + 7 ≤ 0}

def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem set_operations_and_range :
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x : ℝ | x ≥ 1}) ∧
  (∀ a : ℝ, C a ∪ A = A → a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l826_82645


namespace NUMINAMATH_CALUDE_circle_radius_sqrt_61_l826_82622

/-- Given a circle with center on the x-axis passing through points (2,5) and (3,2),
    its radius is √61. -/
theorem circle_radius_sqrt_61 :
  ∀ x : ℝ,
  (∃ r : ℝ, r > 0 ∧
    r^2 = (x - 2)^2 + 5^2 ∧
    r^2 = (x - 3)^2 + 2^2) →
  ∃ r : ℝ, r > 0 ∧ r^2 = 61 :=
by sorry


end NUMINAMATH_CALUDE_circle_radius_sqrt_61_l826_82622


namespace NUMINAMATH_CALUDE_max_marks_calculation_l826_82613

/-- Proves that if a student scores 80% and receives 240 marks, the maximum possible marks in the examination is 300. -/
theorem max_marks_calculation (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) 
  (h1 : percentage = 0.80) 
  (h2 : scored_marks = 240) 
  (h3 : percentage * max_marks = scored_marks) : 
  max_marks = 300 := by
  sorry

end NUMINAMATH_CALUDE_max_marks_calculation_l826_82613


namespace NUMINAMATH_CALUDE_sqrt_11_simplest_l826_82636

def is_simplest_sqrt (n : ℕ) (others : List ℕ) : Prop :=
  ∀ m ∈ others, ¬ (∃ k : ℕ, k > 1 ∧ k * k ∣ n) ∧ (∃ k : ℕ, k > 1 ∧ k * k ∣ m)

theorem sqrt_11_simplest : is_simplest_sqrt 11 [8, 12, 36] := by
  sorry

end NUMINAMATH_CALUDE_sqrt_11_simplest_l826_82636


namespace NUMINAMATH_CALUDE_petya_cannot_equalize_coins_l826_82651

/-- Represents the state of Petya's coins -/
structure CoinState where
  two_kopeck : ℕ
  ten_kopeck : ℕ

/-- Represents a single transaction with the machine -/
inductive Transaction
  | insert_two
  | insert_ten

/-- Applies a single transaction to the current coin state -/
def apply_transaction (state : CoinState) (t : Transaction) : CoinState :=
  match t with
  | Transaction.insert_two => CoinState.mk (state.two_kopeck - 1) (state.ten_kopeck + 5)
  | Transaction.insert_ten => CoinState.mk (state.two_kopeck + 5) (state.ten_kopeck - 1)

/-- Applies a sequence of transactions to the initial state -/
def apply_transactions (initial : CoinState) (ts : List Transaction) : CoinState :=
  ts.foldl apply_transaction initial

/-- The theorem stating that Petya cannot end up with equal coins -/
theorem petya_cannot_equalize_coins :
  ∀ (ts : List Transaction),
    let final_state := apply_transactions (CoinState.mk 1 0) ts
    final_state.two_kopeck ≠ final_state.ten_kopeck :=
by sorry


end NUMINAMATH_CALUDE_petya_cannot_equalize_coins_l826_82651


namespace NUMINAMATH_CALUDE_monotone_increasing_implies_k_bound_l826_82643

/-- A function f is monotonically increasing on an interval [a, b] if for any x, y in [a, b] with x ≤ y, we have f(x) ≤ f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

/-- The main theorem stating that if f(x) = kx - ln(x) is monotonically increasing on [2, 5], then k ≥ 1/2 -/
theorem monotone_increasing_implies_k_bound (k : ℝ) :
  MonotonicallyIncreasing (fun x => k * x - Real.log x) 2 5 → k ≥ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_monotone_increasing_implies_k_bound_l826_82643


namespace NUMINAMATH_CALUDE_cube_plane_intersection_l826_82664

-- Define a cube
def Cube : Type := Unit

-- Define a plane
def Plane : Type := Unit

-- Define the set of faces of a cube
def faces (Q : Cube) : Set Unit := sorry

-- Define the union of faces
def S (Q : Cube) : Set Unit := faces Q

-- Define the set of planes intersecting the cube
def intersecting_planes (Q : Cube) (k : ℕ) : Set Plane := sorry

-- Define the union of intersecting planes
def P (Q : Cube) (k : ℕ) : Set Unit := sorry

-- Define the set of one-third points on the edges of a cube face
def one_third_points (face : Unit) : Set Unit := sorry

-- Define the set of segments joining one-third points on the same face
def one_third_segments (Q : Cube) : Set Unit := sorry

-- State the theorem
theorem cube_plane_intersection (Q : Cube) :
  ∃ k : ℕ, 
    (∀ k' : ℕ, k' ≥ k → 
      (P Q k' ∩ S Q = one_third_segments Q) → 
      k' = k) ∧
    (∀ k' : ℕ, k' ≤ k → 
      (P Q k' ∩ S Q = one_third_segments Q) → 
      k' = k) :=
sorry

end NUMINAMATH_CALUDE_cube_plane_intersection_l826_82664


namespace NUMINAMATH_CALUDE_total_savings_l826_82633

def holiday_savings (sam victory alex : ℕ) : Prop :=
  victory = sam - 200 ∧ alex = 2 * victory ∧ sam = 1200

theorem total_savings (sam victory alex : ℕ) 
  (h : holiday_savings sam victory alex) : 
  sam + victory + alex = 4200 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_l826_82633


namespace NUMINAMATH_CALUDE_key_west_turtle_race_time_l826_82678

/-- Represents the race times of turtles in the Key West Turtle Race -/
structure TurtleRaceTimes where
  greta : Float
  george : Float
  gloria : Float
  gary : Float
  gwen : Float

/-- Calculates the total race time for all turtles -/
def total_race_time (times : TurtleRaceTimes) : Float :=
  times.greta + times.george + times.gloria + times.gary + times.gwen

/-- Theorem stating the total race time for the given conditions -/
theorem key_west_turtle_race_time : ∃ (times : TurtleRaceTimes),
  times.greta = 6.5 ∧
  times.george = times.greta - 1.5 ∧
  times.gloria = 2 * times.george ∧
  times.gary = times.george + times.gloria + 1.75 ∧
  times.gwen = (times.greta + times.george) * 0.6 ∧
  total_race_time times = 45.15 := by
  sorry

end NUMINAMATH_CALUDE_key_west_turtle_race_time_l826_82678


namespace NUMINAMATH_CALUDE_michael_goals_multiplier_l826_82670

theorem michael_goals_multiplier (bruce_goals : ℕ) (total_goals : ℕ) : 
  bruce_goals = 4 → total_goals = 16 → 
  ∃ x : ℕ, x * bruce_goals = total_goals - bruce_goals ∧ x = 3 := by
sorry

end NUMINAMATH_CALUDE_michael_goals_multiplier_l826_82670


namespace NUMINAMATH_CALUDE_jumble_words_count_l826_82669

/-- The number of letters in the Jumble alphabet -/
def alphabet_size : ℕ := 21

/-- The maximum word length in the Jumble language -/
def max_word_length : ℕ := 5

/-- The number of words of length n in the Jumble language that contain at least one 'A' -/
def words_with_a (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else alphabet_size^n - (alphabet_size - 1)^n

/-- The total number of words in the Jumble language -/
def total_words : ℕ :=
  (List.range max_word_length).map (λ i => words_with_a (i + 1)) |>.sum

theorem jumble_words_count :
  total_words = 920885 := by sorry

end NUMINAMATH_CALUDE_jumble_words_count_l826_82669


namespace NUMINAMATH_CALUDE_average_and_difference_l826_82609

theorem average_and_difference (y : ℝ) : 
  (35 + y) / 2 = 42 → |35 - y| = 14 := by sorry

end NUMINAMATH_CALUDE_average_and_difference_l826_82609


namespace NUMINAMATH_CALUDE_counterexample_exists_l826_82641

theorem counterexample_exists : ∃ n : ℕ, 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b) ∧ 
  (∃ k : ℕ, n = 3 * k) ∧ 
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ n - 2 = x * y) ∧ 
  n - 2 ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l826_82641


namespace NUMINAMATH_CALUDE_range_of_a_l826_82644

-- Define the propositions p and q
def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q (x a : ℝ) : Prop := x^2 - (2*a + 1) * x + a * (a + 1) ≤ 0

-- Define the set A for proposition p
def A : Set ℝ := {x | p x}

-- Define the set B for proposition q
def B (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ∧ (∃ x, x ∈ B a ∧ x ∉ A) → 0 ≤ a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l826_82644


namespace NUMINAMATH_CALUDE_revenue_maximization_l826_82655

/-- Revenue function for a scenic area with three ticket options -/
def revenue (x : ℝ) : ℝ := -0.1 * x^2 + 1.8 * x + 180

/-- Initial number of people choosing option A -/
def initial_A : ℝ := 20000

/-- Initial number of people choosing option B -/
def initial_B : ℝ := 10000

/-- Initial number of people choosing combined option -/
def initial_combined : ℝ := 10000

/-- Number of people switching from A to combined per 1 yuan decrease -/
def switch_rate_A : ℝ := 400

/-- Number of people switching from B to combined per 1 yuan decrease -/
def switch_rate_B : ℝ := 600

/-- Price of ticket A -/
def price_A : ℝ := 30

/-- Price of ticket B -/
def price_B : ℝ := 50

/-- Initial price of combined ticket -/
def initial_price_combined : ℝ := 70

theorem revenue_maximization :
  ∃ (x : ℝ), x = 9 ∧ 
  revenue x = 188.1 ∧ 
  ∀ y, revenue y ≤ revenue x :=
sorry

#check revenue_maximization

end NUMINAMATH_CALUDE_revenue_maximization_l826_82655


namespace NUMINAMATH_CALUDE_sara_remaining_money_l826_82698

/-- Calculates the remaining money after a two-week pay period and a purchase -/
def remaining_money (hours_per_week : ℕ) (hourly_rate : ℚ) (purchase_cost : ℚ) : ℚ :=
  2 * (hours_per_week : ℚ) * hourly_rate - purchase_cost

/-- Proves that given the specified work conditions and purchase, the remaining money is $510 -/
theorem sara_remaining_money :
  remaining_money 40 (11.5) 410 = 510 := by
  sorry

end NUMINAMATH_CALUDE_sara_remaining_money_l826_82698


namespace NUMINAMATH_CALUDE_batsman_average_increase_l826_82697

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average for a batsman -/
def averageIncrease (prevInnings : ℕ) (prevTotalRuns : ℕ) (newScore : ℕ) : ℚ :=
  let newAverage := (prevTotalRuns + newScore) / (prevInnings + 1)
  let prevAverage := prevTotalRuns / prevInnings
  newAverage - prevAverage

/-- Theorem: The batsman's average increased by 3 runs -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 12 →
    b.average = 47 →
    averageIncrease 11 (11 * (b.totalRuns / 11)) 80 = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l826_82697


namespace NUMINAMATH_CALUDE_library_reorganization_l826_82672

theorem library_reorganization (total_books : Nat) (books_per_new_stack : Nat) 
    (h1 : total_books = 1450)
    (h2 : books_per_new_stack = 45) : 
  total_books % books_per_new_stack = 10 := by
  sorry

end NUMINAMATH_CALUDE_library_reorganization_l826_82672


namespace NUMINAMATH_CALUDE_divide_algebraic_expression_l826_82666

theorem divide_algebraic_expression (a b c : ℝ) (h : b ≠ 0) :
  4 * a^2 * b^2 * c / (-2 * a * b^2) = -2 * a * c := by
  sorry

end NUMINAMATH_CALUDE_divide_algebraic_expression_l826_82666


namespace NUMINAMATH_CALUDE_doubled_cost_percentage_new_cost_percentage_l826_82657

-- Define the cost function
def cost (t b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_cost_percentage (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  cost t (2 * b) = 16 * cost t b :=
by sorry

-- Main theorem
theorem new_cost_percentage (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  (cost t (2 * b) / cost t b) * 100 = 1600 :=
by sorry

end NUMINAMATH_CALUDE_doubled_cost_percentage_new_cost_percentage_l826_82657


namespace NUMINAMATH_CALUDE_no_real_arithmetic_progression_l826_82617

theorem no_real_arithmetic_progression : ¬ ∃ (a b : ℝ), 
  (b - a = a - 15) ∧ (a * b - b = b - a) := by
  sorry

end NUMINAMATH_CALUDE_no_real_arithmetic_progression_l826_82617


namespace NUMINAMATH_CALUDE_function_range_condition_l826_82660

theorem function_range_condition (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, |2 * x₁ - a| + |2 * x₁ + 3| = |x₂ - 1| + 2) →
  (a ≥ -1 ∨ a ≤ -5) := by
sorry

end NUMINAMATH_CALUDE_function_range_condition_l826_82660


namespace NUMINAMATH_CALUDE_students_liking_both_sports_l826_82663

/-- The number of students who like both basketball and cricket -/
def students_liking_both (b c t : ℕ) : ℕ := b + c - t

/-- Theorem: Given the conditions, prove that 3 students like both basketball and cricket -/
theorem students_liking_both_sports :
  let b := 7  -- number of students who like basketball
  let c := 5  -- number of students who like cricket
  let t := 9  -- total number of students who like basketball or cricket or both
  students_liking_both b c t = 3 := by
sorry

end NUMINAMATH_CALUDE_students_liking_both_sports_l826_82663


namespace NUMINAMATH_CALUDE_equation_solutions_l826_82659

theorem equation_solutions : 
  (∃ (s₁ : Set ℝ), s₁ = {x : ℝ | x^2 - 4*x = 0} ∧ s₁ = {0, 4}) ∧
  (∃ (s₂ : Set ℝ), s₂ = {x : ℝ | x^2 = -2*x + 3} ∧ s₂ = {-3, 1}) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l826_82659


namespace NUMINAMATH_CALUDE_rational_equation_solution_l826_82648

theorem rational_equation_solution :
  ∃ (x : ℝ), (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21) ∧ x = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l826_82648


namespace NUMINAMATH_CALUDE_f_properties_l826_82624

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 1 / x

-- Theorem statement
theorem f_properties (a : ℝ) :
  (∀ x > 0, (deriv (f a)) x = 0 → x = 1) →
  (a = 0) ∧
  (∀ x > 0, f 0 x ≤ x * Real.exp x - x + 1 / x - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l826_82624


namespace NUMINAMATH_CALUDE_matrix_power_in_M_l826_82619

/-- The set M of 2x2 complex matrices where ab = cd -/
def M : Set (Matrix (Fin 2) (Fin 2) ℂ) :=
  {A | A 0 0 * A 0 1 = A 1 0 * A 1 1}

/-- Theorem statement -/
theorem matrix_power_in_M
  (A : Matrix (Fin 2) (Fin 2) ℂ)
  (k : ℕ)
  (hk : k ≥ 1)
  (hA : A ∈ M)
  (hAk : A ^ k ∈ M)
  (hAk1 : A ^ (k + 1) ∈ M)
  (hAk2 : A ^ (k + 2) ∈ M) :
  ∀ n : ℕ, n ≥ 1 → A ^ n ∈ M :=
sorry

end NUMINAMATH_CALUDE_matrix_power_in_M_l826_82619


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l826_82634

theorem repeating_decimal_sum (a b c : Nat) : 
  a < 10 ∧ b < 10 ∧ c < 10 →
  (10 * a + b) / 99 + (100 * a + 10 * b) / 9900 + (10 * b + c) / 99 = 25 / 99 →
  100 * a + 10 * b + c = 23 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l826_82634


namespace NUMINAMATH_CALUDE_part_I_part_II_l826_82631

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 2 - m

-- Define the set A
def A (m : ℝ) : Set ℝ := {y | ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ y = f m x}

-- Theorem for part (I)
theorem part_I (m : ℝ) : 
  (∀ x, f m x ≥ x - m*x) → m ∈ Set.Icc (-7 : ℝ) 1 := by sorry

-- Theorem for part (II)
theorem part_II : 
  (∃ m : ℝ, A m ⊆ Set.Ici 0 ∧ ∀ m' : ℝ, A m' ⊆ Set.Ici 0 → m' ≤ m) → 
  (∃ m : ℝ, m = 1 ∧ A m ⊆ Set.Ici 0 ∧ ∀ m' : ℝ, A m' ⊆ Set.Ici 0 → m' ≤ m) := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l826_82631


namespace NUMINAMATH_CALUDE_max_probability_at_twenty_l826_82602

-- Define the total number of bulbs
def total_bulbs : ℕ := 100

-- Define the number of bulbs picked
def bulbs_picked : ℕ := 10

-- Define the number of defective bulbs in the picked sample
def defective_in_sample : ℕ := 2

-- Define the probability function f(n)
def f (n : ℕ) : ℚ :=
  (Nat.choose n defective_in_sample * Nat.choose (total_bulbs - n) (bulbs_picked - defective_in_sample)) /
  Nat.choose total_bulbs bulbs_picked

-- State the theorem
theorem max_probability_at_twenty {n : ℕ} (h1 : 2 ≤ n) (h2 : n ≤ 92) :
  ∀ m : ℕ, 2 ≤ m ∧ m ≤ 92 → f n ≤ f 20 :=
sorry

end NUMINAMATH_CALUDE_max_probability_at_twenty_l826_82602


namespace NUMINAMATH_CALUDE_min_sums_theorem_l826_82683

def min_sums_for_unique_determination (n : ℕ) : ℕ :=
  Nat.choose (n - 1) 2 + 1

theorem min_sums_theorem (n : ℕ) (h : n ≥ 3) :
  ∀ (a : Fin n → ℝ),
  ∀ (k : ℕ),
  (k < min_sums_for_unique_determination n →
    ∃ (b₁ b₂ : Fin n → ℝ),
      b₁ ≠ b₂ ∧
      (∀ (i j : Fin n), i.val > j.val →
        (∃ (S : Finset (Fin n × Fin n)),
          S.card = k ∧
          (∀ (p : Fin n × Fin n), p ∈ S → p.1.val > p.2.val) ∧
          (∀ (p : Fin n × Fin n), p ∈ S → a (p.1) + a (p.2) = b₁ (p.1) + b₁ (p.2)) ∧
          (∀ (p : Fin n × Fin n), p ∈ S → a (p.1) + a (p.2) = b₂ (p.1) + b₂ (p.2))))) ∧
  (k ≥ min_sums_for_unique_determination n →
    ∀ (b : Fin n → ℝ),
    (∀ (S : Finset (Fin n × Fin n)),
      S.card = k →
      (∀ (p : Fin n × Fin n), p ∈ S → p.1.val > p.2.val) →
      (∃! (c : Fin n → ℝ), ∀ (p : Fin n × Fin n), p ∈ S → c (p.1) + c (p.2) = b (p.1) + b (p.2))))
  := by sorry

end NUMINAMATH_CALUDE_min_sums_theorem_l826_82683


namespace NUMINAMATH_CALUDE_railway_graph_theorem_l826_82693

/-- A graph representing the railway network --/
structure RailwayGraph where
  V : Finset Nat
  E : Finset (Nat × Nat)
  edge_in_V : ∀ (e : Nat × Nat), e ∈ E → e.1 ∈ V ∧ e.2 ∈ V

/-- The degree of a vertex in the graph --/
def degree (G : RailwayGraph) (v : Nat) : Nat :=
  (G.E.filter (fun e => e.1 = v ∨ e.2 = v)).card

/-- The theorem statement --/
theorem railway_graph_theorem (G : RailwayGraph) 
  (hV : G.V.card = 9)
  (hM : degree G 1 = 7)
  (hSP : degree G 2 = 5)
  (hT : degree G 3 = 4)
  (hY : degree G 4 = 2)
  (hB : degree G 5 = 2)
  (hS : degree G 6 = 2)
  (hZ : degree G 7 = 1)
  (hEven : Even (G.E.card * 2))
  (hVV : G.V.card = 9 → ∃ v ∈ G.V, v ≠ 1 ∧ v ≠ 2 ∧ v ≠ 3 ∧ v ≠ 4 ∧ v ≠ 5 ∧ v ≠ 6 ∧ v ≠ 7 ∧ v ≠ 8) :
  ∃ v ∈ G.V, v ≠ 1 ∧ v ≠ 2 ∧ v ≠ 3 ∧ v ≠ 4 ∧ v ≠ 5 ∧ v ≠ 6 ∧ v ≠ 7 ∧ v ≠ 8 ∧ 
    (degree G v = 2 ∨ degree G v = 3 ∨ degree G v = 4 ∨ degree G v = 5) :=
by sorry

end NUMINAMATH_CALUDE_railway_graph_theorem_l826_82693


namespace NUMINAMATH_CALUDE_min_value_of_F_l826_82605

/-- The feasible region defined by the given constraints -/
def FeasibleRegion (x₁ x₂ : ℝ) : Prop :=
  2 - 2*x₁ - x₂ ≥ 0 ∧
  2 - x₁ + x₂ ≥ 0 ∧
  5 - x₁ - x₂ ≥ 0 ∧
  x₁ ≥ 0 ∧
  x₂ ≥ 0

/-- The objective function to be minimized -/
def F (x₁ x₂ : ℝ) : ℝ := x₂ - x₁

/-- Theorem stating that the minimum value of F in the feasible region is -2 -/
theorem min_value_of_F :
  ∀ x₁ x₂ : ℝ, FeasibleRegion x₁ x₂ → F x₁ x₂ ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_F_l826_82605


namespace NUMINAMATH_CALUDE_delivery_pay_calculation_l826_82612

/-- The amount paid per delivery for Oula and Tona --/
def amount_per_delivery : ℝ := sorry

/-- The number of deliveries made by Oula --/
def oula_deliveries : ℕ := 96

/-- The number of deliveries made by Tona --/
def tona_deliveries : ℕ := (3 * oula_deliveries) / 4

/-- The difference in pay between Oula and Tona --/
def pay_difference : ℝ := 2400

theorem delivery_pay_calculation :
  amount_per_delivery * (oula_deliveries - tona_deliveries : ℝ) = pay_difference ∧
  amount_per_delivery = 100 := by sorry

end NUMINAMATH_CALUDE_delivery_pay_calculation_l826_82612


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l826_82600

theorem infinitely_many_solutions (k : ℝ) : 
  (∀ x : ℝ, 5 * (3 * x - k) = 3 * (5 * x + 15)) ↔ k = -9 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l826_82600


namespace NUMINAMATH_CALUDE_school_boys_count_l826_82626

/-- Represents the number of boys in the school -/
def num_boys : ℕ := 410

/-- Represents the initial number of girls in the school -/
def initial_girls : ℕ := 632

/-- Represents the number of additional girls that joined the school -/
def additional_girls : ℕ := 465

/-- Represents the difference between girls and boys after the addition -/
def girl_boy_difference : ℕ := 687

/-- Proves that the number of boys in the school is 410 -/
theorem school_boys_count :
  initial_girls + additional_girls = num_boys + girl_boy_difference := by
  sorry

#check school_boys_count

end NUMINAMATH_CALUDE_school_boys_count_l826_82626


namespace NUMINAMATH_CALUDE_ten_person_meeting_handshakes_l826_82629

/-- The number of handshakes in a meeting of n people where each person
    shakes hands exactly once with every other person. -/
def handshakes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem stating that in a meeting of 10 people, where each person
    shakes hands exactly once with every other person, the total number
    of handshakes is 45. -/
theorem ten_person_meeting_handshakes :
  handshakes 10 = 45 := by sorry

end NUMINAMATH_CALUDE_ten_person_meeting_handshakes_l826_82629


namespace NUMINAMATH_CALUDE_usual_time_calculation_l826_82686

/-- Given a man who takes 24 minutes more to cover a distance when walking at 75% of his usual speed, 
    his usual time to cover this distance is 72 minutes. -/
theorem usual_time_calculation (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_time > 0) 
  (h2 : usual_speed > 0)
  (h3 : usual_speed * usual_time = 0.75 * usual_speed * (usual_time + 24)) : 
  usual_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_calculation_l826_82686


namespace NUMINAMATH_CALUDE_orthocenter_preservation_l826_82601

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : Point := sorry

-- Define the function to check if a point is inside a triangle
def is_inside (p : Point) (t : Triangle) : Prop := sorry

-- Define the function to check if a point is on a line segment
def on_segment (p : Point) (a b : Point) : Prop := sorry

-- Define the function to find the intersection of two line segments
def intersection (a b c d : Point) : Point := sorry

-- Main theorem
theorem orthocenter_preservation 
  (A B C H A₁ B₁ C₁ A₂ B₂ C₂ : Point) 
  (ABC : Triangle) :
  -- Given conditions
  (orthocenter ABC = H) →
  (is_inside A₁ (B, C, H)) →
  (is_inside B₁ (C, A, H)) →
  (is_inside C₁ (A, B, H)) →
  (orthocenter (A₁, B₁, C₁) = H) →
  (A₂ = intersection A H B₁ C₁) →
  (B₂ = intersection B H C₁ A₁) →
  (C₂ = intersection C H A₁ B₁) →
  -- Conclusion
  (orthocenter (A₂, B₂, C₂) = H) := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_preservation_l826_82601


namespace NUMINAMATH_CALUDE_clock_adjustment_l826_82638

/-- Represents the number of minutes lost per day by the clock -/
def minutes_lost_per_day : ℕ := 3

/-- Represents the number of days between March 15 1 P.M. and March 22 9 A.M. -/
def days_elapsed : ℕ := 7

/-- Represents the total number of minutes lost by the clock -/
def total_minutes_lost : ℕ := minutes_lost_per_day * days_elapsed

theorem clock_adjustment :
  total_minutes_lost = 21 := by sorry

end NUMINAMATH_CALUDE_clock_adjustment_l826_82638


namespace NUMINAMATH_CALUDE_locus_of_R_l826_82688

-- Define the square ABCD
structure Square :=
  (A B C D : ℝ × ℝ)

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the perimeter of a square
def perimeter (s : Square) : Set Point := sorry

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (P Q R : Point)

-- Define a rotation around a point
def rotate (center : Point) (angle : ℝ) (p : Point) : Point := sorry

-- Define the theorem
theorem locus_of_R (ABCD : Square) (Q : Point) :
  ∀ P ∈ perimeter ABCD,
  Q ∉ perimeter ABCD →
  ∃ (PQR : EquilateralTriangle),
  PQR.P = P ∧ PQR.Q = Q →
  ∃ (A₁B₁C₁D₁ A₂B₂C₂D₂ : Square),
  A₁B₁C₁D₁ = Square.mk (rotate Q (π/3) ABCD.A) (rotate Q (π/3) ABCD.B) (rotate Q (π/3) ABCD.C) (rotate Q (π/3) ABCD.D) ∧
  A₂B₂C₂D₂ = Square.mk (rotate Q (-π/3) ABCD.A) (rotate Q (-π/3) ABCD.B) (rotate Q (-π/3) ABCD.C) (rotate Q (-π/3) ABCD.D) ∧
  PQR.R ∈ perimeter A₁B₁C₁D₁ ∪ perimeter A₂B₂C₂D₂ :=
by sorry

end NUMINAMATH_CALUDE_locus_of_R_l826_82688


namespace NUMINAMATH_CALUDE_shopkeeper_theft_loss_l826_82630

theorem shopkeeper_theft_loss (profit_percent : ℝ) (loss_percent : ℝ) : 
  profit_percent = 10 → loss_percent = 45 → 
  (loss_percent / 100) * (1 + profit_percent / 100) * 100 = 49.5 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_theft_loss_l826_82630


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l826_82603

theorem wire_ratio_proof (total_length shorter_length : ℕ) 
  (h1 : total_length = 49)
  (h2 : shorter_length = 14)
  (h3 : shorter_length < total_length) :
  let longer_length := total_length - shorter_length
  (shorter_length : ℚ) / longer_length = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l826_82603


namespace NUMINAMATH_CALUDE_two_digit_number_existence_l826_82652

/-- Two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- First digit of a two-digit number -/
def firstDigit (n : ℕ) : ℕ := n / 10

/-- Second digit of a two-digit number -/
def secondDigit (n : ℕ) : ℕ := n % 10

/-- Sum of digits of a two-digit number -/
def digitSum (n : ℕ) : ℕ := firstDigit n + secondDigit n

/-- Absolute difference of digits of a two-digit number -/
def digitDiff (n : ℕ) : ℕ := Int.natAbs (firstDigit n - secondDigit n)

theorem two_digit_number_existence :
  ∃ (X Y : ℕ), 
    TwoDigitNumber X ∧ 
    TwoDigitNumber Y ∧ 
    X = 2 * Y ∧
    (firstDigit Y = digitSum X ∨ secondDigit Y = digitSum X) ∧
    (firstDigit Y = digitDiff X ∨ secondDigit Y = digitDiff X) ∧
    X = 34 ∧ 
    Y = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_existence_l826_82652


namespace NUMINAMATH_CALUDE_isosceles_right_pyramid_leg_length_l826_82662

/-- Represents a pyramid with an isosceles right triangle base -/
structure IsoscelesRightPyramid where
  height : ℝ
  volume : ℝ
  leg : ℝ

/-- The volume of a pyramid is one-third the product of its base area and height -/
axiom pyramid_volume (p : IsoscelesRightPyramid) : p.volume = (1/3) * (1/2 * p.leg^2) * p.height

/-- Theorem: If a pyramid with an isosceles right triangle base has height 4 and volume 6,
    then the length of the leg of the base triangle is 3 -/
theorem isosceles_right_pyramid_leg_length :
  ∀ (p : IsoscelesRightPyramid), p.height = 4 → p.volume = 6 → p.leg = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_right_pyramid_leg_length_l826_82662
