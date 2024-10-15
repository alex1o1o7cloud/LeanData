import Mathlib

namespace NUMINAMATH_CALUDE_unique_tangent_implies_radius_l975_97500

/-- A circle in the x-y plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the x-y plane -/
def Point := ℝ × ℝ

/-- The number of tangent lines from a point to a circle -/
def numTangentLines (c : Circle) (p : Point) : ℕ := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem unique_tangent_implies_radius (c : Circle) (p : Point) :
  c.center = (3, -1) →
  p = (-2, 1) →
  numTangentLines c p = 1 →
  c.radius = Real.sqrt 29 := by sorry

end NUMINAMATH_CALUDE_unique_tangent_implies_radius_l975_97500


namespace NUMINAMATH_CALUDE_min_selling_price_l975_97550

/-- The minimum selling price for a product line given specific conditions --/
theorem min_selling_price (n : ℕ) (avg_price : ℝ) (low_price_count : ℕ) (max_price : ℝ) :
  n = 20 →
  avg_price = 1200 →
  low_price_count = 10 →
  max_price = 11000 →
  ∃ (min_price : ℝ),
    min_price = 400 ∧
    min_price * low_price_count + 1000 * (n - low_price_count - 1) + max_price = n * avg_price :=
by sorry

end NUMINAMATH_CALUDE_min_selling_price_l975_97550


namespace NUMINAMATH_CALUDE_altitude_intersection_property_l975_97511

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a line is perpendicular to another line -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Finds the intersection point of two lines -/
def intersectionPoint (p1 p2 p3 p4 : Point) : Point := sorry

/-- Main theorem -/
theorem altitude_intersection_property (t : Triangle) (P Q H : Point) :
  isAcute t →
  isPerpendicular t.B t.C t.A P →
  isPerpendicular t.A t.C t.B Q →
  H = intersectionPoint t.A P t.B Q →
  distance H P = 4 →
  distance H Q = 3 →
  let B' := intersectionPoint t.A t.C t.B P
  let C' := intersectionPoint t.A t.B t.C P
  let A' := intersectionPoint t.B t.C t.A Q
  let C'' := intersectionPoint t.A t.B t.C Q
  (distance t.B P * distance P C') - (distance t.A Q * distance Q C'') = 7 := by
  sorry

end NUMINAMATH_CALUDE_altitude_intersection_property_l975_97511


namespace NUMINAMATH_CALUDE_find_m_l975_97539

def U : Set Nat := {0, 1, 2, 3}

def A (m : ℝ) : Set Nat := {x ∈ U | x^2 + m * x = 0}

def complement_A : Set Nat := {1, 2}

theorem find_m :
  ∃ m : ℝ, (A m = U \ complement_A) ∧ (m = -3) :=
sorry

end NUMINAMATH_CALUDE_find_m_l975_97539


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_relation_l975_97563

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
structure ArithmeticSequence where
  -- The sequence itself
  a : ℕ → ℝ
  -- The constant difference between consecutive terms
  d : ℝ
  -- The property that defines an arithmetic sequence
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S seq n + seq.a (n + 1)

/-- The main theorem about the relationship between S_3n, S_2n, and S_n -/
theorem arithmetic_sequence_sum_relation (seq : ArithmeticSequence) :
  ∀ n, S seq (3 * n) = 3 * (S seq (2 * n) - S seq n) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_relation_l975_97563


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l975_97538

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = x + y + 1 → a + 2 * b ≤ x + 2 * y ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ + 1 ∧ a₀ + 2 * b₀ = 7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l975_97538


namespace NUMINAMATH_CALUDE_dogs_running_l975_97586

theorem dogs_running (total : ℕ) (playing : ℕ) (barking : ℕ) (idle : ℕ) : 
  total = 88 →
  playing = total / 2 →
  barking = total / 4 →
  idle = 10 →
  total - playing - barking - idle = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_dogs_running_l975_97586


namespace NUMINAMATH_CALUDE_bus_interval_l975_97528

/-- Given a circular bus route where two buses have an interval of 21 minutes between them,
    prove that three buses on the same route will have an interval of 14 minutes between them. -/
theorem bus_interval (total_time : ℕ) (two_bus_interval : ℕ) (three_bus_interval : ℕ) : 
  two_bus_interval = 21 → 
  total_time = 2 * two_bus_interval → 
  three_bus_interval = total_time / 3 → 
  three_bus_interval = 14 := by
  sorry

#eval 42 / 3  -- This should output 14

end NUMINAMATH_CALUDE_bus_interval_l975_97528


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l975_97574

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottomRadius : ℝ
  topRadius : ℝ
  sphereRadius : ℝ
  isTangent : Bool

/-- The theorem stating the radius of the sphere tangent to a specific truncated cone -/
theorem sphere_radius_in_truncated_cone
  (cone : TruncatedConeWithSphere)
  (h1 : cone.bottomRadius = 18)
  (h2 : cone.topRadius = 2)
  (h3 : cone.isTangent = true) :
  cone.sphereRadius = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l975_97574


namespace NUMINAMATH_CALUDE_same_color_marble_probability_l975_97516

/-- The probability of drawing three marbles of the same color from a bag containing
    6 red marbles, 4 white marbles, and 8 blue marbles, without replacement. -/
theorem same_color_marble_probability :
  let red : ℕ := 6
  let white : ℕ := 4
  let blue : ℕ := 8
  let total : ℕ := red + white + blue
  let prob_same_color : ℚ := (Nat.choose red 3 + Nat.choose white 3 + Nat.choose blue 3 : ℚ) / Nat.choose total 3
  prob_same_color = 5 / 51 :=
by sorry

end NUMINAMATH_CALUDE_same_color_marble_probability_l975_97516


namespace NUMINAMATH_CALUDE_bowling_ball_weighs_18_pounds_l975_97592

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := sorry

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := sorry

/-- Theorem stating the weight of one bowling ball is 18 pounds -/
theorem bowling_ball_weighs_18_pounds :
  (10 * bowling_ball_weight = 6 * canoe_weight) →
  (4 * canoe_weight = 120) →
  bowling_ball_weight = 18 := by sorry

end NUMINAMATH_CALUDE_bowling_ball_weighs_18_pounds_l975_97592


namespace NUMINAMATH_CALUDE_bakery_boxes_l975_97564

theorem bakery_boxes (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) : 
  total_muffins = 95 → 
  muffins_per_box = 5 → 
  available_boxes = 10 → 
  (total_muffins - available_boxes * muffins_per_box + muffins_per_box - 1) / muffins_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_bakery_boxes_l975_97564


namespace NUMINAMATH_CALUDE_max_value_expression_l975_97599

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (⨆ x, 2 * (a - x) * (x + c * Real.sqrt (x^2 + b^2))) = a^2 + c^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l975_97599


namespace NUMINAMATH_CALUDE_unit_circle_problem_l975_97560

theorem unit_circle_problem (y₀ : ℝ) (B : ℝ × ℝ) :
  (-3/5)^2 + y₀^2 = 1 →  -- A is on the unit circle
  y₀ > 0 →  -- A is in the second quadrant
  ((-3/5) * B.1 + y₀ * B.2) / ((-3/5)^2 + y₀^2) = 1/2 →  -- Angle between OA and OB is 60°
  B.1^2 + B.2^2 = 4 →  -- |OB| = 2
  (2 * y₀^2 + 2 * (-3/5) * y₀ = 8/25) ∧  -- Part 1: 2sin²α + sin2α = 8/25
  ((B.2 - y₀) / (B.1 + 3/5) = 3/4)  -- Part 2: Slope of AB = 3/4
  := by sorry

end NUMINAMATH_CALUDE_unit_circle_problem_l975_97560


namespace NUMINAMATH_CALUDE_strawberry_harvest_l975_97542

/-- Calculates the expected strawberry harvest from a rectangular garden. -/
theorem strawberry_harvest (length width plants_per_sqft berries_per_plant : ℕ) :
  length = 10 →
  width = 12 →
  plants_per_sqft = 5 →
  berries_per_plant = 8 →
  length * width * plants_per_sqft * berries_per_plant = 4800 := by
  sorry

#check strawberry_harvest

end NUMINAMATH_CALUDE_strawberry_harvest_l975_97542


namespace NUMINAMATH_CALUDE_lillys_fish_l975_97515

theorem lillys_fish (rosys_fish : ℕ) (total_fish : ℕ) (h1 : rosys_fish = 14) (h2 : total_fish = 24) :
  total_fish - rosys_fish = 10 := by
sorry

end NUMINAMATH_CALUDE_lillys_fish_l975_97515


namespace NUMINAMATH_CALUDE_johnsons_class_size_l975_97567

theorem johnsons_class_size (finley_class : ℕ) (johnson_class : ℕ) 
  (h1 : finley_class = 24) 
  (h2 : johnson_class = finley_class / 2 + 10) : 
  johnson_class = 22 := by
  sorry

end NUMINAMATH_CALUDE_johnsons_class_size_l975_97567


namespace NUMINAMATH_CALUDE_doubled_to_original_ratio_l975_97519

theorem doubled_to_original_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 51) : 
  (2 * x) / x = 2 :=
by sorry

end NUMINAMATH_CALUDE_doubled_to_original_ratio_l975_97519


namespace NUMINAMATH_CALUDE_class_representatives_count_l975_97503

/-- The number of ways to select and arrange class representatives -/
def class_representatives (num_boys num_girls num_subjects : ℕ) : ℕ :=
  (num_boys.choose 1) * (num_girls.choose 2) * (num_subjects.factorial)

/-- Theorem: The number of ways to select 2 girls from 3 girls, 1 boy from 3 boys,
    and arrange them as representatives for 3 subjects is 54 -/
theorem class_representatives_count :
  class_representatives 3 3 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_class_representatives_count_l975_97503


namespace NUMINAMATH_CALUDE_specific_gathering_interactions_l975_97526

/-- The number of interactions in a gathering of witches and zombies -/
def interactions (num_witches num_zombies : ℕ) : ℕ :=
  (num_zombies * (num_zombies - 1)) / 2 + num_witches * num_zombies

/-- Theorem stating the number of interactions in a specific gathering -/
theorem specific_gathering_interactions :
  interactions 25 18 = 603 := by
  sorry

end NUMINAMATH_CALUDE_specific_gathering_interactions_l975_97526


namespace NUMINAMATH_CALUDE_odd_integer_quadratic_function_property_l975_97595

theorem odd_integer_quadratic_function_property (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  (Nat.gcd a n = 1) ∧ (Nat.gcd b n = 1) ∧
  (∃ (k : ℕ), n * k = (a^2 + b)) ∧
  (∀ (x : ℕ), x ≥ 1 → ∃ (p : ℕ), Prime p ∧ p ∣ ((x + a)^2 + b) ∧ ¬(p ∣ n)) := by
  sorry

end NUMINAMATH_CALUDE_odd_integer_quadratic_function_property_l975_97595


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l975_97527

/-- The line y = mx + (2m+1) always passes through the point (-2, 1) for any real m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ((-2 : ℝ) * m + (2 * m + 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l975_97527


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l975_97561

theorem quadratic_root_transformation (a b c r s : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = r ∨ x = s) →
  (∀ y, y^2 - b * y + 4 * a * c = 0 ↔ y = 2 * a * r + b ∨ y = 2 * a * s + b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l975_97561


namespace NUMINAMATH_CALUDE_base9_521_equals_base10_424_l975_97553

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 9^2 + tens * 9^1 + ones * 9^0

/-- The theorem stating that 521 in base 9 is equal to 424 in base 10 -/
theorem base9_521_equals_base10_424 :
  base9ToBase10 5 2 1 = 424 := by
  sorry

end NUMINAMATH_CALUDE_base9_521_equals_base10_424_l975_97553


namespace NUMINAMATH_CALUDE_expression_evaluation_l975_97549

theorem expression_evaluation : 3 - (5 : ℝ)^(3-3) = 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l975_97549


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l975_97556

theorem floor_expression_equals_eight :
  ⌊(3005^3 : ℝ) / (3003 * 3004) - (3003^3 : ℝ) / (3004 * 3005)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l975_97556


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l975_97518

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are externally tangent --/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 = (c1.radius + c2.radius) ^ 2

theorem circles_externally_tangent : 
  let c1 : Circle := { center := (4, 2), radius := 3 }
  let c2 : Circle := { center := (0, -1), radius := 2 }
  are_externally_tangent c1 c2 := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l975_97518


namespace NUMINAMATH_CALUDE_find_k_l975_97529

theorem find_k (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h1 : a^2 + b^2 + c^2 = 49)
    (h2 : x^2 + y^2 + z^2 = 64)
    (h3 : a*x + b*y + c*z = 56)
    (h4 : ∃ k, a = k*x ∧ b = k*y ∧ c = k*z) :
  ∃ k, a = k*x ∧ b = k*y ∧ c = k*z ∧ k = 7/8 := by
sorry

end NUMINAMATH_CALUDE_find_k_l975_97529


namespace NUMINAMATH_CALUDE_sum_of_squares_is_integer_l975_97582

theorem sum_of_squares_is_integer 
  (a b c : ℚ) 
  (h1 : ∃ k : ℤ, (a + b + c : ℚ) = k)
  (h2 : ∃ m : ℤ, (a * b + b * c + c * a) / (a + b + c) = m) :
  ∃ n : ℤ, (a^2 + b^2 + c^2) / (a + b + c) = n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_is_integer_l975_97582


namespace NUMINAMATH_CALUDE_triangle_angle_identity_l975_97530

theorem triangle_angle_identity (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_less_than_pi : α < Real.pi ∧ β < Real.pi ∧ γ < Real.pi) : 
  (Real.cos α) / (Real.sin β * Real.sin γ) + 
  (Real.cos β) / (Real.sin α * Real.sin γ) + 
  (Real.cos γ) / (Real.sin α * Real.sin β) = 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_angle_identity_l975_97530


namespace NUMINAMATH_CALUDE_polygon_14_diagonals_interior_angles_sum_l975_97531

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

theorem polygon_14_diagonals_interior_angles_sum :
  ∃ n : ℕ, num_diagonals n = 14 ∧ sum_interior_angles n = 900 :=
sorry

end NUMINAMATH_CALUDE_polygon_14_diagonals_interior_angles_sum_l975_97531


namespace NUMINAMATH_CALUDE_polynomial_factorization_l975_97535

theorem polynomial_factorization (a b c : ℝ) : 
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  (a - b) * (b - c) * (c - a) * (a*b^2 + b*c^2 + c*a^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l975_97535


namespace NUMINAMATH_CALUDE_problem_statement_l975_97594

theorem problem_statement (a b : ℝ) 
  (h1 : |a| = 4) 
  (h2 : |b| = 6) : 
  (ab > 0 → (a - b = 2 ∨ a - b = -2)) ∧ 
  (|a + b| = -(a + b) → (a + b = -10 ∨ a + b = -2)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l975_97594


namespace NUMINAMATH_CALUDE_double_root_values_l975_97547

def polynomial (b₃ b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 50

def is_double_root (p : ℤ → ℤ) (s : ℤ) : Prop :=
  p s = 0 ∧ (∃ q : ℤ → ℤ, ∀ x, p x = (x - s)^2 * q x)

theorem double_root_values (b₃ b₂ b₁ : ℤ) (s : ℤ) :
  is_double_root (polynomial b₃ b₂ b₁) s → s ∈ ({-5, -2, -1, 1, 2, 5} : Set ℤ) :=
by sorry

end NUMINAMATH_CALUDE_double_root_values_l975_97547


namespace NUMINAMATH_CALUDE_complement_union_theorem_l975_97570

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | |x| < 2}
def N : Set ℝ := {y | ∃ x, y = 2^x - 1}

-- State the theorem
theorem complement_union_theorem :
  (Set.compl M ∪ Set.compl N) = {x | x ≤ -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l975_97570


namespace NUMINAMATH_CALUDE_largest_integer_l975_97557

theorem largest_integer (a b c d : ℤ) 
  (sum1 : a + b + c = 163)
  (sum2 : a + b + d = 178)
  (sum3 : a + c + d = 184)
  (sum4 : b + c + d = 194) :
  max a (max b (max c d)) = 77 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_l975_97557


namespace NUMINAMATH_CALUDE_function_inequality_l975_97596

theorem function_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 3 * x + 2) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 2| < b → |f x + 4| < a) ↔
  b ≤ a / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l975_97596


namespace NUMINAMATH_CALUDE_a_plus_b_equals_five_l975_97572

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem a_plus_b_equals_five (a b : ℝ) :
  (A ∪ B a b = Set.univ) →
  (A ∩ B a b = {x | 3 < x ∧ x ≤ 4}) →
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_five_l975_97572


namespace NUMINAMATH_CALUDE_max_value_of_function_l975_97507

theorem max_value_of_function (x : ℝ) : 
  1 / (x^2 + x + 1) ≤ 4/3 ∧ ∃ y : ℝ, 1 / (y^2 + y + 1) = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l975_97507


namespace NUMINAMATH_CALUDE_stating_thirty_cents_combinations_l975_97523

/-- The value of a penny in cents -/
def pennyValue : ℕ := 1

/-- The value of a nickel in cents -/
def nickelValue : ℕ := 5

/-- The value of a dime in cents -/
def dimeValue : ℕ := 10

/-- The total value we want to achieve in cents -/
def totalValue : ℕ := 30

/-- 
A function that calculates the number of ways to make a given amount of cents
using pennies, nickels, and dimes.
-/
def countCombinations (cents : ℕ) : ℕ := sorry

/-- 
Theorem stating that the number of combinations to make 30 cents
using pennies, nickels, and dimes is 20.
-/
theorem thirty_cents_combinations : 
  countCombinations totalValue = 20 := by sorry

end NUMINAMATH_CALUDE_stating_thirty_cents_combinations_l975_97523


namespace NUMINAMATH_CALUDE_last_digit_of_multiple_of_six_l975_97590

theorem last_digit_of_multiple_of_six (x : ℕ) :
  x < 10 →
  (43560 + x) % 6 = 0 →
  x = 0 ∨ x = 6 := by
sorry

end NUMINAMATH_CALUDE_last_digit_of_multiple_of_six_l975_97590


namespace NUMINAMATH_CALUDE_defective_bulb_probability_l975_97510

/-- The probability of randomly picking a defective bulb from a box with a given pass rate -/
theorem defective_bulb_probability (pass_rate : ℝ) (h : pass_rate = 0.875) :
  1 - pass_rate = 0.125 := by
  sorry

#check defective_bulb_probability

end NUMINAMATH_CALUDE_defective_bulb_probability_l975_97510


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l975_97559

theorem greatest_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ y : ℤ, 4 * |y| - 6 < 34 → y ≤ x) ∧ (4 * |x| - 6 < 34) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l975_97559


namespace NUMINAMATH_CALUDE_percentage_and_absolute_difference_l975_97548

/-- Given two initial values and an annual percentage increase, 
    calculate the percentage difference and the absolute difference after 5 years. -/
theorem percentage_and_absolute_difference 
  (initial_value1 : ℝ) 
  (initial_value2 : ℝ) 
  (annual_increase : ℝ) 
  (h1 : initial_value1 = 0.60 * 5000) 
  (h2 : initial_value2 = 0.42 * 3000) :
  let difference := initial_value1 - initial_value2
  let percentage_difference := (difference / initial_value1) * 100
  let new_difference := difference * (1 + annual_increase / 100) ^ 5
  percentage_difference = 58 ∧ 
  new_difference = 1740 * (1 + annual_increase / 100) ^ 5 := by
sorry

end NUMINAMATH_CALUDE_percentage_and_absolute_difference_l975_97548


namespace NUMINAMATH_CALUDE_total_games_is_105_l975_97546

/-- The number of teams in the league -/
def num_teams : ℕ := 15

/-- The total number of games played in the league -/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the total number of games played is 105 -/
theorem total_games_is_105 : total_games num_teams = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_games_is_105_l975_97546


namespace NUMINAMATH_CALUDE_zeros_of_log_linear_function_l975_97591

open Real

theorem zeros_of_log_linear_function (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m > 0) 
  (hx : x₁ < x₂) 
  (hz₁ : m * log x₁ = x₁) 
  (hz₂ : m * log x₂ = x₂) : 
  x₁ < exp 1 ∧ exp 1 < x₂ := by
sorry

end NUMINAMATH_CALUDE_zeros_of_log_linear_function_l975_97591


namespace NUMINAMATH_CALUDE_recreation_spending_comparison_l975_97522

theorem recreation_spending_comparison (last_week_wages : ℝ) : 
  let last_week_recreation := 0.20 * last_week_wages
  let this_week_wages := 0.80 * last_week_wages
  let this_week_recreation := 0.40 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 160 := by
sorry

end NUMINAMATH_CALUDE_recreation_spending_comparison_l975_97522


namespace NUMINAMATH_CALUDE_height_comparison_l975_97585

theorem height_comparison (a b : ℝ) (h : a = 0.8 * b) : b = 1.25 * a := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l975_97585


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l975_97543

theorem cube_root_of_negative_eight : 
  ∃ x : ℝ, x^3 = -8 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l975_97543


namespace NUMINAMATH_CALUDE_increase_by_percentage_l975_97502

theorem increase_by_percentage (initial : ℕ) (percentage : ℕ) (result : ℕ) : 
  initial = 150 → percentage = 40 → result = initial + (initial * percentage) / 100 → result = 210 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l975_97502


namespace NUMINAMATH_CALUDE_outfit_count_l975_97540

/-- The number of different outfits that can be made with the given clothing items. -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (belts : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (belts + 1)

/-- Theorem stating the number of outfits for the given clothing items. -/
theorem outfit_count :
  number_of_outfits 7 4 5 2 = 504 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l975_97540


namespace NUMINAMATH_CALUDE_complex_simplification_l975_97532

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that the simplification of the given complex expression equals 30 -/
theorem complex_simplification : 6 * (4 - 2 * i) + 2 * i * (6 - 3 * i) = 30 := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l975_97532


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l975_97506

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l975_97506


namespace NUMINAMATH_CALUDE_probability_three_tails_one_head_probability_three_tails_one_head_proof_l975_97554

/-- The probability of getting exactly three tails and one head when tossing four coins simultaneously -/
theorem probability_three_tails_one_head : ℚ :=
  1 / 4

/-- Proof that the probability of getting exactly three tails and one head when tossing four coins simultaneously is 1/4 -/
theorem probability_three_tails_one_head_proof :
  probability_three_tails_one_head = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_tails_one_head_probability_three_tails_one_head_proof_l975_97554


namespace NUMINAMATH_CALUDE_smallest_n_with_shared_digit_arrangement_l975_97520

/-- A function that checks if two natural numbers share a digit in their decimal representation -/
def share_digit (a b : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ (d ∈ a.digits 10) ∧ (d ∈ b.digits 10)

/-- A function that checks if a list of natural numbers satisfies the neighboring digit condition -/
def valid_arrangement (lst : List ℕ) : Prop :=
  ∀ i : ℕ, i < lst.length → share_digit (lst.get! i) (lst.get! ((i + 1) % lst.length))

/-- The main theorem stating that 29 is the smallest N satisfying the conditions -/
theorem smallest_n_with_shared_digit_arrangement :
  ∀ N : ℕ, N ≥ 2 →
  (∃ lst : List ℕ, lst.length = N ∧ lst.toFinset = Finset.range N.succ ∧ valid_arrangement lst) →
  N ≥ 29 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_shared_digit_arrangement_l975_97520


namespace NUMINAMATH_CALUDE_problem_solution_l975_97544

theorem problem_solution :
  ∀ a b c d : ℝ,
  (100 * a = 35^2 - 15^2) →
  ((a - 1)^2 = 3^(4 * b)) →
  (b^2 + c * b - 5 = 0) →
  (∃ k : ℝ, 2 * (x^2) + 3 * x + 4 * d = (x + c) * k) →
  (a = 10 ∧ b = 1 ∧ c = 4 ∧ d = -5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l975_97544


namespace NUMINAMATH_CALUDE_musician_earnings_per_song_l975_97509

/-- Represents the earnings of a musician over a period of time --/
structure MusicianEarnings where
  songs_per_month : ℕ
  total_earnings : ℕ
  years : ℕ

/-- Calculates the earnings per song for a musician --/
def earnings_per_song (m : MusicianEarnings) : ℚ :=
  m.total_earnings / (m.songs_per_month * 12 * m.years)

/-- Theorem: A musician releasing 3 songs per month and earning $216,000 in 3 years makes $2,000 per song --/
theorem musician_earnings_per_song :
  let m : MusicianEarnings := { songs_per_month := 3, total_earnings := 216000, years := 3 }
  earnings_per_song m = 2000 := by
  sorry


end NUMINAMATH_CALUDE_musician_earnings_per_song_l975_97509


namespace NUMINAMATH_CALUDE_average_of_three_quantities_l975_97568

theorem average_of_three_quantities 
  (total_count : Nat) 
  (total_average : ℚ) 
  (subset_count : Nat) 
  (subset_average : ℚ) 
  (h1 : total_count = 5) 
  (h2 : total_average = 11) 
  (h3 : subset_count = 2) 
  (h4 : subset_average = 21.5) : 
  (total_count * total_average - subset_count * subset_average) / (total_count - subset_count) = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_quantities_l975_97568


namespace NUMINAMATH_CALUDE_stamp_price_l975_97534

theorem stamp_price (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 6 → purchase_price = (1 / 5) * original_price → original_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_stamp_price_l975_97534


namespace NUMINAMATH_CALUDE_cost_for_23_days_l975_97558

/-- Calculates the total cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 13
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- Proves that the cost of staying for 23 days in the student youth hostel is $334.00. -/
theorem cost_for_23_days : hostelCost 23 = 334 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_23_days_l975_97558


namespace NUMINAMATH_CALUDE_inequality_proof_l975_97552

theorem inequality_proof (a m : ℝ) (ha : a > 0) :
  abs (m + a) + abs (m + 1 / a) + abs (-1 / m + a) + abs (-1 / m + 1 / a) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l975_97552


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l975_97569

theorem arctan_sum_special_case : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l975_97569


namespace NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l975_97501

theorem systematic_sampling_smallest_number 
  (total_classes : Nat) 
  (selected_classes : Nat) 
  (sum_of_selected : Nat) : 
  total_classes = 30 → 
  selected_classes = 6 → 
  sum_of_selected = 87 → 
  (total_classes / selected_classes : Nat) = 5 → 
  ∃ x : Nat, 
    x + (x + 5) + (x + 10) + (x + 15) + (x + 20) + (x + 25) = sum_of_selected ∧ 
    x = 2 ∧ 
    (∀ y : Nat, y + (y + 5) + (y + 10) + (y + 15) + (y + 20) + (y + 25) = sum_of_selected → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l975_97501


namespace NUMINAMATH_CALUDE_smallest_sum_of_c_and_d_l975_97533

theorem smallest_sum_of_c_and_d (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h1 : ∃ x : ℝ, x^2 + c*x + 3*d = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*d*x + c = 0) :
  c + d ≥ (4*Real.sqrt 3 + 4/3) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_c_and_d_l975_97533


namespace NUMINAMATH_CALUDE_largest_equal_digit_sums_l975_97545

/-- Calculates the sum of digits of a natural number in a given base. -/
def digit_sum (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Checks if a number has equal digit sums in base 10 and base 3. -/
def equal_digit_sums (n : ℕ) : Prop :=
  digit_sum n 10 = digit_sum n 3

theorem largest_equal_digit_sums :
  ∀ m : ℕ, m < 1000 → m > 310 → ¬(equal_digit_sums m) ∧ equal_digit_sums 310 := by sorry

end NUMINAMATH_CALUDE_largest_equal_digit_sums_l975_97545


namespace NUMINAMATH_CALUDE_triangle_exists_iff_altitudes_condition_l975_97537

/-- A triangle with altitudes m_a, m_b, and m_c exists if and only if
    1/m_a + 1/m_b > 1/m_c and 1/m_b + 1/m_c > 1/m_a and 1/m_c + 1/m_a > 1/m_b -/
theorem triangle_exists_iff_altitudes_condition
  (m_a m_b m_c : ℝ) (h_pos_a : m_a > 0) (h_pos_b : m_b > 0) (h_pos_c : m_c > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * m_a = b * m_b) ∧ (b * m_b = c * m_c) ∧ (c * m_c = a * m_a) ↔
  (1 / m_a + 1 / m_b > 1 / m_c) ∧
  (1 / m_b + 1 / m_c > 1 / m_a) ∧
  (1 / m_c + 1 / m_a > 1 / m_b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_exists_iff_altitudes_condition_l975_97537


namespace NUMINAMATH_CALUDE_baseball_games_played_l975_97566

theorem baseball_games_played (wins losses played : ℕ) : 
  wins = 5 → 
  played = wins + losses → 
  played = 2 * losses → 
  played = 10 := by
sorry

end NUMINAMATH_CALUDE_baseball_games_played_l975_97566


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l975_97581

theorem complex_power_one_minus_i_six :
  (1 - Complex.I) ^ 6 = 8 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l975_97581


namespace NUMINAMATH_CALUDE_units_digit_G_3_l975_97505

-- Define G_n
def G (n : ℕ) : ℕ := 2^(2^(2^n)) + 1

-- Theorem statement
theorem units_digit_G_3 : G 3 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_3_l975_97505


namespace NUMINAMATH_CALUDE_red_spools_count_l975_97575

-- Define the variables
def spools_per_beret : ℕ := 3
def black_spools : ℕ := 15
def blue_spools : ℕ := 6
def total_berets : ℕ := 11

-- Define the theorem
theorem red_spools_count : 
  ∃ (red_spools : ℕ), 
    red_spools + black_spools + blue_spools = spools_per_beret * total_berets ∧ 
    red_spools = 12 := by
  sorry

end NUMINAMATH_CALUDE_red_spools_count_l975_97575


namespace NUMINAMATH_CALUDE_geometric_progression_relation_l975_97571

/-- Given two geometric progressions, prove that their first terms are related as stated. -/
theorem geometric_progression_relation (a b q : ℝ) (n : ℕ) (h : q ≠ 1) :
  (a * (q^(2*n) - 1)) / (q - 1) = (b * (q^(2*n) - 1)) / (q^2 - 1) →
  b = a + a * q :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_relation_l975_97571


namespace NUMINAMATH_CALUDE_company_blocks_l975_97525

/-- Calculates the number of blocks in a company based on gift budget and workers per block -/
theorem company_blocks (total_amount : ℝ) (gift_worth : ℝ) (workers_per_block : ℝ) :
  total_amount = 4000 ∧ gift_worth = 4 ∧ workers_per_block = 100 →
  (total_amount / gift_worth) / workers_per_block = 10 := by
  sorry

end NUMINAMATH_CALUDE_company_blocks_l975_97525


namespace NUMINAMATH_CALUDE_circle_symmetry_l975_97593

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 2016

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 2016

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ),
  original_circle x y ∧ symmetry_line x y →
  ∃ (x' y' : ℝ), symmetric_circle x' y' ∧ symmetry_line ((x + x') / 2) ((y + y') / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l975_97593


namespace NUMINAMATH_CALUDE_binary_digit_difference_l975_97583

theorem binary_digit_difference (n m : ℕ) (hn : n = 1280) (hm : m = 320) :
  (Nat.log 2 n + 1) - (Nat.log 2 m + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l975_97583


namespace NUMINAMATH_CALUDE_estimate_white_balls_l975_97577

/-- The number of red balls in the bag -/
def red_balls : ℕ := 6

/-- The probability of drawing a red ball -/
def prob_red : ℚ := 1/5

/-- The number of white balls in the bag -/
def white_balls : ℕ := 24

theorem estimate_white_balls :
  (red_balls : ℚ) / (red_balls + white_balls) = prob_red := by
  sorry

end NUMINAMATH_CALUDE_estimate_white_balls_l975_97577


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l975_97555

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 3 * (x^3 - 2*x^2 + 3) - 5 * (x^4 - 4*x^2 + 2)

/-- The coefficients of the fully simplified expression -/
def coefficients : List ℝ := [-5, 3, 14, -1]

/-- Theorem: The sum of the squares of the coefficients of the fully simplified expression is 231 -/
theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 231 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l975_97555


namespace NUMINAMATH_CALUDE_banana_arrangements_l975_97573

-- Define the word and its properties
def banana_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

-- Theorem statement
theorem banana_arrangements : 
  (banana_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l975_97573


namespace NUMINAMATH_CALUDE_line_through_two_points_l975_97578

/-- Given a line passing through points (-3, 5) and (0, -4), prove that m + b = -7 
    where y = mx + b is the equation of the line. -/
theorem line_through_two_points (m b : ℝ) : 
  (5 = -3 * m + b) ∧ (-4 = 0 * m + b) → m + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_line_through_two_points_l975_97578


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l975_97584

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4 * x + a > 0) ↔ a > 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l975_97584


namespace NUMINAMATH_CALUDE_farmland_and_spray_theorem_l975_97562

/-- Represents the farmland areas and drone spraying capacities in two zones -/
structure FarmlandData where
  zone_a : ℝ  -- Farmland area in Zone A
  zone_b : ℝ  -- Farmland area in Zone B
  spray_a : ℝ  -- Average area sprayed per sortie in Zone A

/-- The conditions given in the problem -/
def problem_conditions (data : FarmlandData) : Prop :=
  data.zone_a = data.zone_b + 10000 ∧  -- Zone A has 10,000 mu more farmland
  0.8 * data.zone_a = data.zone_b ∧  -- 80% of Zone A equals all of Zone B (suitable area)
  (data.zone_b / data.spray_a) * 1.2 = data.zone_b / (data.spray_a - 50/3)  -- Drone sortie relationship

/-- The theorem to be proved -/
theorem farmland_and_spray_theorem (data : FarmlandData) :
  problem_conditions data →
  data.zone_a = 50000 ∧ data.zone_b = 40000 ∧ data.spray_a = 100 := by
  sorry

end NUMINAMATH_CALUDE_farmland_and_spray_theorem_l975_97562


namespace NUMINAMATH_CALUDE_balloon_count_l975_97580

/-- The number of filled water balloons Max and Zach have in total -/
def total_balloons (max_rate : ℕ) (max_time : ℕ) (zach_rate : ℕ) (zach_time : ℕ) (popped : ℕ) : ℕ :=
  max_rate * max_time + zach_rate * zach_time - popped

/-- Theorem stating the total number of filled water balloons Max and Zach have -/
theorem balloon_count : total_balloons 2 30 3 40 10 = 170 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l975_97580


namespace NUMINAMATH_CALUDE_pizza_count_l975_97587

def num_toppings : ℕ := 8

def zero_topping_pizzas : ℕ := 1

def one_topping_pizzas (n : ℕ) : ℕ := n

def two_topping_pizzas (n : ℕ) : ℕ := n.choose 2

def total_pizzas (n : ℕ) : ℕ :=
  zero_topping_pizzas + one_topping_pizzas n + two_topping_pizzas n

theorem pizza_count : total_pizzas num_toppings = 37 := by
  sorry

end NUMINAMATH_CALUDE_pizza_count_l975_97587


namespace NUMINAMATH_CALUDE_percent_relation_l975_97512

theorem percent_relation (x y z P : ℝ) 
  (hy : y = 0.75 * x) 
  (hz : z = 0.65 * x) 
  (hP : P / 100 * z = 0.39 * y) : 
  P = 45 := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l975_97512


namespace NUMINAMATH_CALUDE_total_vehicles_calculation_l975_97504

theorem total_vehicles_calculation (lanes : ℕ) (trucks_per_lane : ℕ) : 
  lanes = 4 →
  trucks_per_lane = 60 →
  (lanes * trucks_per_lane + lanes * (2 * lanes * trucks_per_lane)) = 2160 := by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_calculation_l975_97504


namespace NUMINAMATH_CALUDE_sqrt_equation_condition_l975_97521

theorem sqrt_equation_condition (a b : ℝ) (k : ℕ+) :
  (Real.sqrt (a^2 + (k.val * b)^2) = a + k.val * b) ↔ (a * k.val * b = 0 ∧ a + k.val * b ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_sqrt_equation_condition_l975_97521


namespace NUMINAMATH_CALUDE_fraction_equals_d_minus_one_l975_97508

theorem fraction_equals_d_minus_one (n d : ℕ) (h : d ∣ n) :
  ∃ k : ℕ, k < n ∧ (k : ℚ) / (n - k : ℚ) = d - 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_d_minus_one_l975_97508


namespace NUMINAMATH_CALUDE_mrs_dunbar_roses_l975_97551

/-- Calculates the total number of white roses needed for a wedding arrangement -/
def total_roses (num_bouquets : ℕ) (num_table_decorations : ℕ) (roses_per_bouquet : ℕ) (roses_per_table_decoration : ℕ) : ℕ :=
  num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration

/-- Proves that the total number of white roses needed for Mrs. Dunbar's wedding arrangement is 109 -/
theorem mrs_dunbar_roses : total_roses 5 7 5 12 = 109 := by
  sorry

end NUMINAMATH_CALUDE_mrs_dunbar_roses_l975_97551


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l975_97579

theorem integer_solutions_of_equation :
  ∀ m n : ℤ, m^5 - n^5 = 16*m*n ↔ (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = -2) ∨ (m = -2 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l975_97579


namespace NUMINAMATH_CALUDE_function_inequality_range_l975_97588

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem function_inequality_range 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f (2*x - 1) < f (1/3)} = Set.Ioo (1/3) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_range_l975_97588


namespace NUMINAMATH_CALUDE_average_age_decrease_l975_97576

/-- Given a group of 10 persons with an unknown average age A,
    prove that replacing a person aged 40 with a person aged 10
    decreases the average age by 3 years. -/
theorem average_age_decrease (A : ℝ) : 
  A - ((10 * A - 30) / 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_age_decrease_l975_97576


namespace NUMINAMATH_CALUDE_washer_cost_l975_97565

/-- Given a washer-dryer combination costing 1200 dollars, where the washer costs 220 dollars more than the dryer, prove that the washer costs 710 dollars. -/
theorem washer_cost (total : ℝ) (difference : ℝ) (washer : ℝ) (dryer : ℝ) : 
  total = 1200 →
  difference = 220 →
  washer = dryer + difference →
  total = washer + dryer →
  washer = 710 := by
sorry

end NUMINAMATH_CALUDE_washer_cost_l975_97565


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l975_97517

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Calculates the total population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Calculates the number of people to be sampled from each stratum -/
def stratifiedSample (p : Population) (sampleSize : ℕ) : Population :=
  { elderly := (p.elderly * sampleSize) / totalPopulation p,
    middleAged := (p.middleAged * sampleSize) / totalPopulation p,
    young := (p.young * sampleSize) / totalPopulation p }

/-- The theorem to be proved -/
theorem correct_stratified_sample :
  let p : Population := { elderly := 27, middleAged := 54, young := 81 }
  let sample := stratifiedSample p 36
  sample.elderly = 6 ∧ sample.middleAged = 12 ∧ sample.young = 18 := by
  sorry


end NUMINAMATH_CALUDE_correct_stratified_sample_l975_97517


namespace NUMINAMATH_CALUDE_fourth_term_is_seven_l975_97524

-- Define the sequence sum function
def S (n : ℕ) : ℕ := n^2

-- Define the sequence term function
def a (n : ℕ) : ℤ := S n - S (n-1)

-- Theorem statement
theorem fourth_term_is_seven : a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_seven_l975_97524


namespace NUMINAMATH_CALUDE_coffee_stock_percentage_l975_97514

theorem coffee_stock_percentage (initial_stock : ℝ) (initial_decaf_percent : ℝ)
  (additional_stock : ℝ) (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 30)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 60) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := (initial_stock * initial_decaf_percent / 100) +
                     (additional_stock * additional_decaf_percent / 100)
  total_decaf / total_stock * 100 = 36 := by
sorry

end NUMINAMATH_CALUDE_coffee_stock_percentage_l975_97514


namespace NUMINAMATH_CALUDE_tic_tac_toe_4x4_carl_wins_l975_97597

/-- Represents a 4x4 tic-tac-toe board --/
def Board := Fin 4 → Fin 4 → Option Bool

/-- Represents a winning line on the board --/
structure WinningLine :=
  (positions : List (Fin 4 × Fin 4))
  (is_valid : positions.length = 4)

/-- All possible winning lines on a 4x4 board --/
def winningLines : List WinningLine := sorry

/-- Checks if a given board configuration is valid --/
def isValidBoard (b : Board) : Prop := sorry

/-- Checks if Carl wins with exactly 4 O's --/
def carlWinsWithFourO (b : Board) : Prop := sorry

/-- The number of ways Carl can win with exactly 4 O's --/
def numWaysToWin : ℕ := sorry

theorem tic_tac_toe_4x4_carl_wins :
  numWaysToWin = 4950 := by sorry

end NUMINAMATH_CALUDE_tic_tac_toe_4x4_carl_wins_l975_97597


namespace NUMINAMATH_CALUDE_janet_fertilizer_spread_rate_l975_97541

theorem janet_fertilizer_spread_rate 
  (horses : ℕ) 
  (fertilizer_per_horse : ℚ) 
  (acres : ℕ) 
  (fertilizer_per_acre : ℚ) 
  (days : ℕ) 
  (h1 : horses = 80)
  (h2 : fertilizer_per_horse = 5)
  (h3 : acres = 20)
  (h4 : fertilizer_per_acre = 400)
  (h5 : days = 25)
  : (acres : ℚ) / days = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_janet_fertilizer_spread_rate_l975_97541


namespace NUMINAMATH_CALUDE_initial_money_proof_l975_97598

/-- The amount of money Mrs. Hilt had initially -/
def initial_money : ℕ := 15

/-- The cost of the pencil in cents -/
def pencil_cost : ℕ := 11

/-- The amount of money left after buying the pencil -/
def money_left : ℕ := 4

/-- Theorem stating that the initial money equals the sum of the pencil cost and money left -/
theorem initial_money_proof : initial_money = pencil_cost + money_left := by
  sorry

end NUMINAMATH_CALUDE_initial_money_proof_l975_97598


namespace NUMINAMATH_CALUDE_simplify_expression_l975_97589

theorem simplify_expression (a b : ℝ) : 
  -2 * (a^3 - 3*b^2) + 4 * (-b^2 + a^3) = 2*a^3 + 2*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l975_97589


namespace NUMINAMATH_CALUDE_intersection_sum_l975_97536

/-- Given two functions f and g where
    f(x) = -|x-a| + b
    g(x) = |x-c| + d
    that intersect at points (2,5) and (8,3),
    prove that a + c = 10 -/
theorem intersection_sum (a b c d : ℝ) :
  (∀ x, -|x - a| + b = |x - c| + d ↔ (x = 2 ∧ -|x - a| + b = 5) ∨ (x = 8 ∧ -|x - a| + b = 3)) →
  a + c = 10 :=
sorry

end NUMINAMATH_CALUDE_intersection_sum_l975_97536


namespace NUMINAMATH_CALUDE_amusement_park_spending_l975_97513

theorem amusement_park_spending (admission_cost food_cost total_cost : ℕ) : 
  food_cost = admission_cost - 13 →
  total_cost = admission_cost + food_cost →
  total_cost = 77 →
  admission_cost = 45 := by
sorry

end NUMINAMATH_CALUDE_amusement_park_spending_l975_97513
