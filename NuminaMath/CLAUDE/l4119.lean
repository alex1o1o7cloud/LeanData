import Mathlib

namespace NUMINAMATH_CALUDE_meal_cost_is_27_l4119_411924

/-- Represents the cost of a meal with tax and tip. -/
structure MealCost where
  pretax : ℝ
  tax_rate : ℝ
  tip_rate : ℝ
  total : ℝ

/-- Calculates the total cost of a meal including tax and tip. -/
def total_cost (m : MealCost) : ℝ :=
  m.pretax * (1 + m.tax_rate + m.tip_rate)

/-- Theorem stating that given the conditions, the pre-tax meal cost is $27. -/
theorem meal_cost_is_27 :
  ∃ (m : MealCost),
    m.tax_rate = 0.08 ∧
    m.tip_rate = 0.18 ∧
    m.total = 33.60 ∧
    total_cost m = m.total ∧
    m.pretax = 27 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_is_27_l4119_411924


namespace NUMINAMATH_CALUDE_steves_cookies_l4119_411981

theorem steves_cookies (total_spent milk_cost cereal_cost banana_cost apple_cost : ℚ)
  (cereal_boxes banana_count apple_count : ℕ)
  (h_total : total_spent = 25)
  (h_milk : milk_cost = 3)
  (h_cereal : cereal_cost = 7/2)
  (h_cereal_boxes : cereal_boxes = 2)
  (h_banana : banana_cost = 1/4)
  (h_banana_count : banana_count = 4)
  (h_apple : apple_cost = 1/2)
  (h_apple_count : apple_count = 4)
  (h_cookie_cost : ∀ x, x = 2 * milk_cost) :
  ∃ (cookie_boxes : ℕ), cookie_boxes = 2 ∧
    total_spent = milk_cost + cereal_cost * cereal_boxes + 
      banana_cost * banana_count + apple_cost * apple_count + 
      (2 * milk_cost) * cookie_boxes :=
by sorry

end NUMINAMATH_CALUDE_steves_cookies_l4119_411981


namespace NUMINAMATH_CALUDE_no_grades_four_or_five_l4119_411946

theorem no_grades_four_or_five (n : ℕ) (x : ℕ) : 
  (5 : ℕ) ≠ 0 → -- There are 5 problems
  n * x + (x + 1) = 25 → -- Total problems solved
  9 ≤ n + 1 → -- At least 9 students (including Peter)
  x + 1 ≤ 5 → -- Maximum grade is 5
  ¬(x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_no_grades_four_or_five_l4119_411946


namespace NUMINAMATH_CALUDE_initial_scissors_l4119_411991

theorem initial_scissors (added : ℕ) (total : ℕ) (h1 : added = 22) (h2 : total = 76) :
  total - added = 54 := by
  sorry

end NUMINAMATH_CALUDE_initial_scissors_l4119_411991


namespace NUMINAMATH_CALUDE_element_in_set_given_complement_l4119_411916

def U : Finset Nat := {1, 2, 3, 4, 5}

theorem element_in_set_given_complement (M : Finset Nat) 
  (h : (U \ M) = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_given_complement_l4119_411916


namespace NUMINAMATH_CALUDE_alice_bob_meet_l4119_411905

/-- The number of points on the circular track -/
def n : ℕ := 15

/-- Alice's movement in clockwise direction per turn -/
def a : ℕ := 7

/-- Bob's movement in counterclockwise direction per turn -/
def b : ℕ := 10

/-- The function that calculates the position after k turns -/
def position (movement : ℕ) (k : ℕ) : ℕ :=
  (movement * k) % n

/-- The theorem stating that Alice and Bob meet after 8 turns -/
theorem alice_bob_meet :
  (∀ k : ℕ, k < 8 → position a k ≠ position (n - b) k) ∧
  position a 8 = position (n - b) 8 :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l4119_411905


namespace NUMINAMATH_CALUDE_carnations_ordered_l4119_411952

/-- Proves that given the specified conditions, the number of carnations ordered is 375 -/
theorem carnations_ordered (tulips : ℕ) (roses : ℕ) (price_per_flower : ℕ) (total_expenses : ℕ) : 
  tulips = 250 → roses = 320 → price_per_flower = 2 → total_expenses = 1890 →
  ∃ carnations : ℕ, carnations = 375 ∧ 
    price_per_flower * (tulips + roses + carnations) = total_expenses := by
  sorry

#check carnations_ordered

end NUMINAMATH_CALUDE_carnations_ordered_l4119_411952


namespace NUMINAMATH_CALUDE_quadratic_maximum_l4119_411973

theorem quadratic_maximum : 
  (∃ (s : ℝ), -3 * s^2 + 24 * s + 5 = 53) ∧ 
  (∀ (s : ℝ), -3 * s^2 + 24 * s + 5 ≤ 53) := by
sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l4119_411973


namespace NUMINAMATH_CALUDE_min_square_size_and_unused_area_l4119_411909

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- The shapes contained within the larger square -/
def contained_shapes : List Rectangle := [
  { width := 2, height := 2 },  -- 2x2 square
  { width := 1, height := 3 },  -- 1x3 rectangle
  { width := 2, height := 1 }   -- 2x1 rectangle
]

/-- Theorem: The minimum side length of the containing square is 5,
    and the minimum unused area is 16 -/
theorem min_square_size_and_unused_area :
  let min_side := 5
  let total_area := min_side * min_side
  let shapes_area := (contained_shapes.map Rectangle.area).sum
  let unused_area := total_area - shapes_area
  (∀ side : ℕ, side ≥ min_side → 
    side * side - shapes_area ≥ unused_area) ∧
  unused_area = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_square_size_and_unused_area_l4119_411909


namespace NUMINAMATH_CALUDE_fish_westward_l4119_411974

/-- The number of fish that swam westward -/
def W : ℕ := sorry

/-- The number of fish that swam eastward -/
def E : ℕ := 3200

/-- The number of fish that swam north -/
def N : ℕ := 500

/-- The fraction of eastward-swimming fish caught by fishers -/
def east_catch_ratio : ℚ := 2 / 5

/-- The fraction of westward-swimming fish caught by fishers -/
def west_catch_ratio : ℚ := 3 / 4

/-- The number of fish left in the sea after catching -/
def fish_left : ℕ := 2870

theorem fish_westward :
  W = 1800 ∧
  (W : ℚ) + E + N - (east_catch_ratio * E + west_catch_ratio * W) = fish_left :=
sorry

end NUMINAMATH_CALUDE_fish_westward_l4119_411974


namespace NUMINAMATH_CALUDE_not_always_true_a_gt_b_implies_ac_sq_gt_bc_sq_l4119_411985

theorem not_always_true_a_gt_b_implies_ac_sq_gt_bc_sq :
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 :=
sorry

end NUMINAMATH_CALUDE_not_always_true_a_gt_b_implies_ac_sq_gt_bc_sq_l4119_411985


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l4119_411932

-- Problem 1
theorem problem_1 : -1^2009 + Real.rpow 27 (1/3) - |1 - Real.sqrt 2| + Real.sqrt 8 = 3 + Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = Real.sqrt 2 + 1) (hy : y = Real.sqrt 2 - 1) :
  y / x + x / y + 2 = 8 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l4119_411932


namespace NUMINAMATH_CALUDE_convex_polyhedron_properties_l4119_411982

/-- A convex polyhedron with congruent isosceles triangular faces -/
structure ConvexPolyhedron where
  vertices : ℕ
  faces : ℕ
  edges : ℕ
  isConvex : Bool
  hasCongruentIsoscelesFaces : Bool
  formGeometricSequence : Bool

/-- Euler's formula for polyhedra -/
axiom euler_formula {p : ConvexPolyhedron} : p.vertices + p.faces = p.edges + 2

/-- Relation between faces and edges in a polyhedron with triangular faces -/
axiom triangular_faces_relation {p : ConvexPolyhedron} : 2 * p.edges = 3 * p.faces

/-- Geometric sequence property -/
axiom geometric_sequence {p : ConvexPolyhedron} (h : p.formGeometricSequence) :
  p.faces / p.vertices = p.edges / p.faces

/-- Main theorem: A convex polyhedron with the given properties has 8 vertices, 12 faces, and 18 edges -/
theorem convex_polyhedron_properties (p : ConvexPolyhedron)
  (h1 : p.isConvex)
  (h2 : p.hasCongruentIsoscelesFaces)
  (h3 : p.formGeometricSequence) :
  p.vertices = 8 ∧ p.faces = 12 ∧ p.edges = 18 := by
  sorry

end NUMINAMATH_CALUDE_convex_polyhedron_properties_l4119_411982


namespace NUMINAMATH_CALUDE_cubic_integer_roots_l4119_411988

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ

/-- The number of distinct integer roots of a cubic polynomial -/
def num_distinct_integer_roots (p : CubicPolynomial) : ℕ :=
  sorry

/-- Theorem stating the possible values for the number of distinct integer roots -/
theorem cubic_integer_roots (p : CubicPolynomial) :
  num_distinct_integer_roots p ∈ ({0, 1, 2, 3} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_cubic_integer_roots_l4119_411988


namespace NUMINAMATH_CALUDE_ring_stack_height_l4119_411979

/-- Represents a stack of linked rings -/
structure RingStack where
  top_diameter : ℝ
  bottom_diameter : ℝ
  ring_thickness : ℝ

/-- Calculates the total height of the ring stack -/
def stack_height (stack : RingStack) : ℝ :=
  sorry

/-- Theorem: The height of the given ring stack is 72 cm -/
theorem ring_stack_height :
  let stack := RingStack.mk 20 4 2
  stack_height stack = 72 := by
  sorry

end NUMINAMATH_CALUDE_ring_stack_height_l4119_411979


namespace NUMINAMATH_CALUDE_triangle_area_l4119_411901

/-- The area of a triangle with base 2t and height 2t - 6, where t = 9 -/
theorem triangle_area (t : ℝ) (h : t = 9) : 
  (1/2 : ℝ) * (2*t) * (2*t - 6) = 108 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l4119_411901


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l4119_411964

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ+) : 
  Nat.gcd A B = 23 →
  A = 391 →
  Nat.lcm A B = 23 * 16 * X →
  X = 17 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l4119_411964


namespace NUMINAMATH_CALUDE_hotel_room_charges_percentage_increase_l4119_411919

/-- Proves that if the charge for a single room at hotel P is 70% less than hotel R
    and 10% less than hotel G, then the charge for a single room at hotel R is 170%
    greater than hotel G. -/
theorem hotel_room_charges (P R G : ℝ) 
    (h1 : P = R * 0.3)  -- P is 70% less than R
    (h2 : P = G * 0.9)  -- P is 10% less than G
    : R = G * 2.7 := by
  sorry

/-- Proves that if R = G * 2.7, then R is 170% greater than G. -/
theorem percentage_increase (R G : ℝ) (h : R = G * 2.7) 
    : (R - G) / G * 100 = 170 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charges_percentage_increase_l4119_411919


namespace NUMINAMATH_CALUDE_complement_of_M_in_N_l4119_411917

def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {0, 2, 3, 4, 5}

theorem complement_of_M_in_N :
  N \ M = {0, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_N_l4119_411917


namespace NUMINAMATH_CALUDE_bridge_length_l4119_411900

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 245 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l4119_411900


namespace NUMINAMATH_CALUDE_biloca_path_theorem_l4119_411902

/-- Represents the dimensions and paths of ants on a tiled floor -/
structure AntPaths where
  diagonal_length : ℝ
  tile_width : ℝ
  tile_length : ℝ
  pipoca_path : ℝ
  tonica_path : ℝ
  cotinha_path : ℝ

/-- Calculates the length of Biloca's path -/
def biloca_path_length (ap : AntPaths) : ℝ :=
  3 * ap.diagonal_length + 4 * ap.tile_width + 2 * ap.tile_length

/-- Theorem stating the length of Biloca's path -/
theorem biloca_path_theorem (ap : AntPaths) 
  (h1 : ap.pipoca_path = 5 * ap.diagonal_length)
  (h2 : ap.pipoca_path = 25)
  (h3 : ap.tonica_path = 5 * ap.diagonal_length + 4 * ap.tile_width)
  (h4 : ap.tonica_path = 37)
  (h5 : ap.cotinha_path = 5 * ap.tile_length + 4 * ap.tile_width)
  (h6 : ap.cotinha_path = 32) :
  biloca_path_length ap = 35 := by
  sorry


end NUMINAMATH_CALUDE_biloca_path_theorem_l4119_411902


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l4119_411945

/-- A normal distribution with mean 54 and 3 standard deviations below the mean greater than 47 has a standard deviation less than 2.33 -/
theorem normal_distribution_std_dev (σ : ℝ) 
  (h1 : 54 - 3 * σ > 47) : 
  σ < 2.33 := by
sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l4119_411945


namespace NUMINAMATH_CALUDE_product_of_integers_l4119_411912

theorem product_of_integers (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (sum_eq : p + q + r = 24)
  (frac_eq : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 240 / (p * q * r) = 1) :
  p * q * r = 384 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l4119_411912


namespace NUMINAMATH_CALUDE_shane_chewed_eleven_pieces_l4119_411927

def elyse_initial_gum : ℕ := 100
def shane_remaining_gum : ℕ := 14

def rick_gum : ℕ := elyse_initial_gum / 2
def shane_initial_gum : ℕ := rick_gum / 2

def shane_chewed_gum : ℕ := shane_initial_gum - shane_remaining_gum

theorem shane_chewed_eleven_pieces : shane_chewed_gum = 11 := by
  sorry

end NUMINAMATH_CALUDE_shane_chewed_eleven_pieces_l4119_411927


namespace NUMINAMATH_CALUDE_sandras_puppies_l4119_411987

theorem sandras_puppies (total_portions : ℕ) (num_days : ℕ) (feedings_per_day : ℕ) :
  total_portions = 105 →
  num_days = 5 →
  feedings_per_day = 3 →
  (total_portions / num_days) / feedings_per_day = 7 :=
by sorry

end NUMINAMATH_CALUDE_sandras_puppies_l4119_411987


namespace NUMINAMATH_CALUDE_system_solution_l4119_411996

theorem system_solution (x y : ℝ) : 
  (x = 1 ∧ y = 4) → 
  (Real.sqrt (y / x) - 2 * Real.sqrt (x / y) = 1 ∧ 
   Real.sqrt (5 * x + y) + Real.sqrt (5 * x - y) = 4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4119_411996


namespace NUMINAMATH_CALUDE_min_max_sum_l4119_411961

theorem min_max_sum (p q r s t u : ℕ+) 
  (sum_eq : p + q + r + s + t + u = 2023) : 
  810 ≤ max (p + q) (max (q + r) (max (r + s) (max (s + t) (t + u)))) := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_l4119_411961


namespace NUMINAMATH_CALUDE_outfit_combinations_l4119_411968

/-- The number of available shirts, pants, and hats -/
def num_items : ℕ := 7

/-- The number of available colors -/
def num_colors : ℕ := 7

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_items * num_items * num_items

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- The number of valid outfit combinations -/
def valid_outfits : ℕ := total_combinations - same_color_outfits

theorem outfit_combinations : valid_outfits = 336 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l4119_411968


namespace NUMINAMATH_CALUDE_kim_total_points_l4119_411950

/-- Represents the points structure of a math contest --/
structure ContestPoints where
  easy : Nat
  average : Nat
  hard : Nat
  expert : Nat
  bonusPerComplex : Nat

/-- Represents a contestant's performance in the math contest --/
structure ContestPerformance where
  points : ContestPoints
  easyCorrect : Nat
  averageCorrect : Nat
  hardCorrect : Nat
  expertCorrect : Nat
  complexSolved : Nat

/-- Calculates the total points for a contestant --/
def calculateTotalPoints (performance : ContestPerformance) : Nat :=
  performance.easyCorrect * performance.points.easy +
  performance.averageCorrect * performance.points.average +
  performance.hardCorrect * performance.points.hard +
  performance.expertCorrect * performance.points.expert +
  performance.complexSolved * performance.points.bonusPerComplex

/-- Theorem stating that Kim's total points in the contest equal 61 --/
theorem kim_total_points :
  let contestPoints : ContestPoints := {
    easy := 2,
    average := 3,
    hard := 5,
    expert := 7,
    bonusPerComplex := 1
  }
  let kimPerformance : ContestPerformance := {
    points := contestPoints,
    easyCorrect := 6,
    averageCorrect := 2,
    hardCorrect := 4,
    expertCorrect := 3,
    complexSolved := 2
  }
  calculateTotalPoints kimPerformance = 61 := by
  sorry


end NUMINAMATH_CALUDE_kim_total_points_l4119_411950


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l4119_411966

theorem circle_equation_k_value (x y k : ℝ) : 
  (∃ h c : ℝ, ∀ x y : ℝ, x^2 + 12*x + y^2 + 14*y - k = 0 ↔ (x - h)^2 + (y - c)^2 = 8^2) ↔ 
  k = 85 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l4119_411966


namespace NUMINAMATH_CALUDE_ordering_of_exponentials_l4119_411995

theorem ordering_of_exponentials (x a b : ℝ) :
  x > 0 → 1 < b^x → b^x < a^x → 1 < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ordering_of_exponentials_l4119_411995


namespace NUMINAMATH_CALUDE_remainder_123456789012_mod_252_l4119_411938

theorem remainder_123456789012_mod_252 : 123456789012 % 252 = 108 := by
  sorry

end NUMINAMATH_CALUDE_remainder_123456789012_mod_252_l4119_411938


namespace NUMINAMATH_CALUDE_fence_cost_l4119_411947

-- Define the side lengths of the pentagon
def side1 : ℕ := 10
def side2 : ℕ := 14
def side3 : ℕ := 12
def side4 : ℕ := 8
def side5 : ℕ := 6

-- Define the prices per foot for each group of sides
def price1 : ℕ := 45  -- Price for first two sides
def price2 : ℕ := 55  -- Price for third and fourth sides
def price3 : ℕ := 60  -- Price for last side

-- Define the total cost function
def totalCost : ℕ := 
  side1 * price1 + side2 * price1 + 
  side3 * price2 + side4 * price2 + 
  side5 * price3

-- Theorem stating that the total cost is 2540
theorem fence_cost : totalCost = 2540 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_l4119_411947


namespace NUMINAMATH_CALUDE_probability_of_selecting_letter_l4119_411986

theorem probability_of_selecting_letter (total_letters : ℕ) (unique_letters : ℕ) 
  (h1 : total_letters = 26) (h2 : unique_letters = 8) : 
  (unique_letters : ℚ) / total_letters = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_letter_l4119_411986


namespace NUMINAMATH_CALUDE_good_iff_mod_three_l4119_411959

/-- A number n > 3 is "good" if the set of weights {1, 2, 3, ..., n} can be divided into three piles of equal mass. -/
def IsGood (n : ℕ) : Prop :=
  n > 3 ∧ ∃ (a b c : Finset ℕ), a ∪ b ∪ c = Finset.range n ∧ a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
    (a.sum id = b.sum id) ∧ (b.sum id = c.sum id)

theorem good_iff_mod_three (n : ℕ) : IsGood n ↔ n > 3 ∧ (n % 3 = 0 ∨ n % 3 = 2) :=
sorry

end NUMINAMATH_CALUDE_good_iff_mod_three_l4119_411959


namespace NUMINAMATH_CALUDE_students_in_all_classes_l4119_411972

/-- Represents the number of students in each class combination --/
structure ClassCombinations where
  code_only : ℕ
  chess_only : ℕ
  photo_only : ℕ
  code_chess : ℕ
  code_photo : ℕ
  chess_photo : ℕ
  all_three : ℕ

/-- The problem statement and conditions --/
theorem students_in_all_classes 
  (total_students : ℕ)
  (code_students : ℕ)
  (chess_students : ℕ)
  (photo_students : ℕ)
  (multi_class_students : ℕ)
  (h1 : total_students = 25)
  (h2 : code_students = 12)
  (h3 : chess_students = 15)
  (h4 : photo_students = 10)
  (h5 : multi_class_students = 10)
  (combinations : ClassCombinations)
  (h6 : total_students = 
    combinations.code_only + combinations.chess_only + combinations.photo_only + 
    combinations.code_chess + combinations.code_photo + combinations.chess_photo + 
    combinations.all_three)
  (h7 : code_students = 
    combinations.code_only + combinations.code_chess + combinations.code_photo + 
    combinations.all_three)
  (h8 : chess_students = 
    combinations.chess_only + combinations.code_chess + combinations.chess_photo + 
    combinations.all_three)
  (h9 : photo_students = 
    combinations.photo_only + combinations.code_photo + combinations.chess_photo + 
    combinations.all_three)
  (h10 : multi_class_students = 
    combinations.code_chess + combinations.code_photo + combinations.chess_photo + 
    combinations.all_three) :
  combinations.all_three = 4 := by
  sorry


end NUMINAMATH_CALUDE_students_in_all_classes_l4119_411972


namespace NUMINAMATH_CALUDE_largest_root_of_quadratic_l4119_411907

theorem largest_root_of_quadratic (y : ℝ) :
  (6 * y ^ 2 - 31 * y + 35 = 0) → y ≤ (5 / 2 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_root_of_quadratic_l4119_411907


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l4119_411963

/-- Given two vectors a and b in ℝ², prove that when a = (1,3) and b = (x,1) are perpendicular, x = -3 -/
theorem perpendicular_vectors (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 3]
  let b : Fin 2 → ℝ := ![x, 1]
  (∀ i, i < 2 → a i * b i = 0) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l4119_411963


namespace NUMINAMATH_CALUDE_inverse_variation_sqrt_l4119_411967

/-- Given that z varies inversely as √w, prove that w = 64 when z = 2, 
    given that z = 8 when w = 4. -/
theorem inverse_variation_sqrt (z w : ℝ) (h : ∃ k : ℝ, ∀ w : ℝ, w > 0 → z * Real.sqrt w = k) 
    (h1 : 8 * Real.sqrt 4 = z * Real.sqrt w) : 
    z = 2 → w = 64 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_sqrt_l4119_411967


namespace NUMINAMATH_CALUDE_orange_bin_theorem_l4119_411930

/-- Calculates the final number of oranges in a bin after changes. -/
def final_oranges (initial : ℕ) (thrown_away : ℕ) (added : ℕ) : ℕ :=
  initial - thrown_away + added

/-- Proves that the final number of oranges is correct given the initial conditions. -/
theorem orange_bin_theorem (initial : ℕ) (thrown_away : ℕ) (added : ℕ) :
  final_oranges initial thrown_away added = initial - thrown_away + added :=
by sorry

end NUMINAMATH_CALUDE_orange_bin_theorem_l4119_411930


namespace NUMINAMATH_CALUDE_ratio_problem_l4119_411992

theorem ratio_problem (a b c : ℝ) 
  (hab : a / b = 11 / 3) 
  (hac : a / c = 11 / 15) : 
  b / c = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l4119_411992


namespace NUMINAMATH_CALUDE_union_A_complement_B_equals_interval_l4119_411990

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- State the theorem
theorem union_A_complement_B_equals_interval :
  A ∪ (U \ B) = Set.Iio 2 := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_equals_interval_l4119_411990


namespace NUMINAMATH_CALUDE_sentence_is_true_l4119_411921

def sentence := "In this phrase, 1/2 of all digits are '1', fractions of digits '2' and '5' are equal and equal to 1/5, and the proportion of all other digits is 1/10."

def digit_1 := 1
def digit_2 := 2
def digit_3 := 5

def fraction_1 := (1 : ℚ) / 2
def fraction_2 := (1 : ℚ) / 5
def fraction_3 := (1 : ℚ) / 10

theorem sentence_is_true :
  (fraction_1 + 2 * fraction_2 + fraction_3 = 1) ∧
  (digit_1 ≠ digit_2 ∧ digit_1 ≠ digit_3 ∧ digit_2 ≠ digit_3) ∧
  (fraction_1 ≠ fraction_2 ∧ fraction_1 ≠ fraction_3 ∧ fraction_2 ≠ fraction_3) →
  ∃ (total_digits : ℕ),
    (fraction_1 * total_digits).num = (total_digits.digits 10).count digit_1 ∧
    (fraction_2 * total_digits).num = (total_digits.digits 10).count digit_2 ∧
    (fraction_2 * total_digits).num = (total_digits.digits 10).count digit_3 ∧
    (fraction_3 * total_digits).num = total_digits - (fraction_1 * total_digits).num - 2 * (fraction_2 * total_digits).num :=
by sorry

end NUMINAMATH_CALUDE_sentence_is_true_l4119_411921


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l4119_411999

/-- Given two right triangles with sides 5, 12, and 13, where a square of side length x
    is inscribed in the first triangle with a vertex coinciding with the right angle,
    and a square of side length y is inscribed in the second triangle with a side lying
    on the hypotenuse, the ratio x/y equals 12/13. -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  (x > 0 ∧ y > 0) →
  (5^2 + 12^2 = 13^2) →
  (x / 12 = x / 5) →
  ((12 - y) / y = (5 - y) / y) →
  x / y = 12 / 13 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l4119_411999


namespace NUMINAMATH_CALUDE_units_digit_of_product_division_l4119_411944

theorem units_digit_of_product_division : 
  (12 * 13 * 14 * 15 * 16 * 17) / 2000 % 10 = 6 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_division_l4119_411944


namespace NUMINAMATH_CALUDE_prime_even_intersection_l4119_411941

-- Define the set of prime numbers
def P : Set ℕ := {n : ℕ | Nat.Prime n}

-- Define the set of even numbers
def Q : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2 * k}

-- Theorem statement
theorem prime_even_intersection :
  P ∩ Q = {2} :=
sorry

end NUMINAMATH_CALUDE_prime_even_intersection_l4119_411941


namespace NUMINAMATH_CALUDE_max_product_sum_2004_l4119_411960

theorem max_product_sum_2004 :
  (∃ (a b : ℤ), a + b = 2004 ∧ a * b = 1004004) ∧
  (∀ (x y : ℤ), x + y = 2004 → x * y ≤ 1004004) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2004_l4119_411960


namespace NUMINAMATH_CALUDE_function_derivative_implies_coefficients_l4119_411923

/-- Given a function f(x) = x^m + ax with derivative f'(x) = 2x + 1, prove that m = 3 and a = 1 -/
theorem function_derivative_implies_coefficients 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^m + a*x) 
  (h2 : ∀ x, deriv f x = 2*x + 1) : 
  m = 3 ∧ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_implies_coefficients_l4119_411923


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l4119_411926

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : ∀ m : ℝ, 2 ≤ m ∧ m ≤ 3 → a + b ≤ m^2 - 2*m + 6) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    (∀ m : ℝ, 2 ≤ m ∧ m ≤ 3 → x + y ≤ m^2 - 2*m + 6) ∧
    Real.sqrt (x + 1) + Real.sqrt (y + 1) = 4 ∧
    (∀ c d : ℝ, c > 0 → d > 0 →
      (∀ m : ℝ, 2 ≤ m ∧ m ≤ 3 → c + d ≤ m^2 - 2*m + 6) →
      Real.sqrt (c + 1) + Real.sqrt (d + 1) ≤ 4) := by
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l4119_411926


namespace NUMINAMATH_CALUDE_parallel_perpendicular_transitivity_l4119_411954

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Line : Type
  Plane : Type
  parallel_lines : Line → Line → Prop
  perpendicular_line_plane : Line → Plane → Prop

/-- The theorem stating that if a line is parallel to another line that is perpendicular to a plane, 
    then the first line is also perpendicular to that plane -/
theorem parallel_perpendicular_transitivity 
  {S : Space3D} {m n : S.Line} {α : S.Plane} :
  S.parallel_lines m n → S.perpendicular_line_plane m α → S.perpendicular_line_plane n α :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_transitivity_l4119_411954


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l4119_411955

theorem quadratic_root_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 2023*x₁ + 1 = 0 → 
  x₂^2 - 2023*x₂ + 1 = 0 → 
  x₁ ≠ x₂ →
  (1/x₁) + (1/x₂) = 2023 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l4119_411955


namespace NUMINAMATH_CALUDE_total_spent_proof_l4119_411929

/-- The price of each flower in dollars -/
def flower_price : ℕ := 3

/-- The number of roses Zoe bought -/
def roses_bought : ℕ := 8

/-- The number of daisies Zoe bought -/
def daisies_bought : ℕ := 2

/-- Theorem: Given the price of each flower and the number of roses and daisies bought,
    prove that the total amount spent is 30 dollars -/
theorem total_spent_proof :
  flower_price * (roses_bought + daisies_bought) = 30 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_proof_l4119_411929


namespace NUMINAMATH_CALUDE_sine_cosine_equality_l4119_411989

theorem sine_cosine_equality (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) :
  Real.sin (n * Real.pi / 180) = Real.cos (510 * Real.pi / 180) → n = -60 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_equality_l4119_411989


namespace NUMINAMATH_CALUDE_shorter_base_length_l4119_411983

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  midpoint_segment : ℝ

/-- Calculates the length of the shorter base of a trapezoid -/
def shorter_base (t : Trapezoid) : ℝ :=
  t.long_base - 2 * t.midpoint_segment

/-- Theorem stating the length of the shorter base for a specific trapezoid -/
theorem shorter_base_length (t : Trapezoid) 
  (h1 : t.long_base = 102) 
  (h2 : t.midpoint_segment = 5) : 
  shorter_base t = 92 := by
  sorry

end NUMINAMATH_CALUDE_shorter_base_length_l4119_411983


namespace NUMINAMATH_CALUDE_multiply_32519_9999_l4119_411936

theorem multiply_32519_9999 : 32519 * 9999 = 324857481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_32519_9999_l4119_411936


namespace NUMINAMATH_CALUDE_conors_potato_chopping_l4119_411925

/-- The number of potatoes Conor can chop in a day -/
def potatoes_per_day : ℕ := sorry

/-- The number of eggplants Conor can chop in a day -/
def eggplants_per_day : ℕ := 12

/-- The number of carrots Conor can chop in a day -/
def carrots_per_day : ℕ := 9

/-- The number of days Conor works per week -/
def work_days_per_week : ℕ := 4

/-- The total number of vegetables Conor chops in a week -/
def total_vegetables_per_week : ℕ := 116

theorem conors_potato_chopping :
  potatoes_per_day = 8 ∧
  work_days_per_week * (eggplants_per_day + carrots_per_day + potatoes_per_day) = total_vegetables_per_week :=
by sorry

end NUMINAMATH_CALUDE_conors_potato_chopping_l4119_411925


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l4119_411928

/-- Given a point A with coordinates (-3, 2), its symmetric point
    with respect to the y-axis has coordinates (3, 2). -/
theorem symmetric_point_y_axis :
  let A : ℝ × ℝ := (-3, 2)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
  symmetric_point A = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l4119_411928


namespace NUMINAMATH_CALUDE_negative_half_power_twenty_times_negative_two_power_twentysix_l4119_411904

theorem negative_half_power_twenty_times_negative_two_power_twentysix :
  -0.5^20 * (-2)^26 = -64 := by
  sorry

end NUMINAMATH_CALUDE_negative_half_power_twenty_times_negative_two_power_twentysix_l4119_411904


namespace NUMINAMATH_CALUDE_product_decreasing_inequality_l4119_411953

theorem product_decreasing_inequality 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h_deriv : ∀ x, (deriv f x) * g x + f x * (deriv g x) < 0) 
  {a b x : ℝ} 
  (h_interval : a < x ∧ x < b) : 
  f x * g x > f b * g b :=
sorry

end NUMINAMATH_CALUDE_product_decreasing_inequality_l4119_411953


namespace NUMINAMATH_CALUDE_average_running_distance_l4119_411943

def monday_distance : ℝ := 4.2
def tuesday_distance : ℝ := 3.8
def wednesday_distance : ℝ := 3.6
def thursday_distance : ℝ := 4.4
def number_of_days : ℕ := 4

theorem average_running_distance :
  (monday_distance + tuesday_distance + wednesday_distance + thursday_distance) / number_of_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_running_distance_l4119_411943


namespace NUMINAMATH_CALUDE_intersection_point_lines_parallel_line_equation_y_intercept_4_equation_l4119_411934

-- Define the lines and point M
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0
def l (x y : ℝ) : Prop := 2 * x + 4 * y - 5 = 0

-- M is the intersection point of l₁ and l₂
def M : ℝ × ℝ := (-1, 2)

-- Define the equations of the lines we want to prove
def line_parallel (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def line_y_intercept_4 (x y : ℝ) : Prop := 2 * x - y + 4 = 0

theorem intersection_point_lines (x y : ℝ) :
  l₁ x y ∧ l₂ x y ↔ (x, y) = M :=
sorry

theorem parallel_line_equation :
  ∀ x y : ℝ, (x, y) = M → line_parallel x y ∧ ∃ k : ℝ, ∀ x y : ℝ, line_parallel x y ↔ l (x + k) (y + k) :=
sorry

theorem y_intercept_4_equation :
  ∀ x y : ℝ, (x, y) = M → line_y_intercept_4 x y ∧ line_y_intercept_4 0 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_lines_parallel_line_equation_y_intercept_4_equation_l4119_411934


namespace NUMINAMATH_CALUDE_curve_intersection_distance_l4119_411903

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - x + y = 0

def C₂ (t x y : ℝ) : Prop := x = 1/2 - (Real.sqrt 2 / 2) * t ∧ y = (Real.sqrt 2 / 2) * t

-- Theorem statement
theorem curve_intersection_distance : 
  -- The polar curve ρ = cos θ - sin θ is equivalent to C₁
  (∀ (ρ θ : ℝ), ρ = Real.cos θ - Real.sin θ ↔ C₁ (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  -- The distance between intersection points is √6/2
  (∃ (t₁ t₂ : ℝ), 
    (C₁ (1/2 - (Real.sqrt 2 / 2) * t₁) ((Real.sqrt 2 / 2) * t₁)) ∧
    (C₁ (1/2 - (Real.sqrt 2 / 2) * t₂) ((Real.sqrt 2 / 2) * t₂)) ∧
    (C₂ t₁ (1/2 - (Real.sqrt 2 / 2) * t₁) ((Real.sqrt 2 / 2) * t₁)) ∧
    (C₂ t₂ (1/2 - (Real.sqrt 2 / 2) * t₂) ((Real.sqrt 2 / 2) * t₂)) ∧
    (t₁ ≠ t₂) ∧
    ((1/2 - (Real.sqrt 2 / 2) * t₁ - (1/2 - (Real.sqrt 2 / 2) * t₂))^2 + 
     ((Real.sqrt 2 / 2) * t₁ - (Real.sqrt 2 / 2) * t₂)^2 = 3/2)) := by
  sorry

end NUMINAMATH_CALUDE_curve_intersection_distance_l4119_411903


namespace NUMINAMATH_CALUDE_probability_prime_or_square_l4119_411993

/-- A function that returns true if a number is prime --/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns true if a number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop := sorry

/-- The number of sides on each die --/
def numSides : ℕ := 8

/-- The set of possible outcomes when rolling two dice --/
def outcomes : Finset (ℕ × ℕ) := sorry

/-- The set of favorable outcomes (sum is prime or perfect square) --/
def favorableOutcomes : Finset (ℕ × ℕ) := sorry

/-- Theorem stating the probability of getting a sum that is either prime or a perfect square --/
theorem probability_prime_or_square :
  (Finset.card favorableOutcomes : ℚ) / (Finset.card outcomes : ℚ) = 35 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_prime_or_square_l4119_411993


namespace NUMINAMATH_CALUDE_croissant_making_time_l4119_411962

/-- Proves that the total time for making croissants is 6 hours -/
theorem croissant_making_time : 
  let fold_time : ℕ := 4 * 5
  let rest_time : ℕ := 4 * 75
  let mix_time : ℕ := 10
  let bake_time : ℕ := 30
  let minutes_per_hour : ℕ := 60
  (fold_time + rest_time + mix_time + bake_time) / minutes_per_hour = 6 := by
  sorry

end NUMINAMATH_CALUDE_croissant_making_time_l4119_411962


namespace NUMINAMATH_CALUDE_prob_800_to_1000_l4119_411956

/-- Probability that a light bulb works after 800 hours -/
def prob_800 : ℝ := 0.8

/-- Probability that a light bulb works after 1000 hours -/
def prob_1000 : ℝ := 0.5

/-- Theorem stating the probability of a light bulb continuing to work from 800 to 1000 hours -/
theorem prob_800_to_1000 : (prob_1000 / prob_800 : ℝ) = 5/8 := by sorry

end NUMINAMATH_CALUDE_prob_800_to_1000_l4119_411956


namespace NUMINAMATH_CALUDE_soccer_camp_afternoon_l4119_411911

/-- Given 2000 kids in total, with half going to soccer camp and 1/4 of those going in the morning,
    the number of kids going to soccer camp in the afternoon is 750. -/
theorem soccer_camp_afternoon (total : ℕ) (soccer : ℕ) (morning : ℕ) : 
  total = 2000 →
  soccer = total / 2 →
  morning = soccer / 4 →
  soccer - morning = 750 := by
  sorry

end NUMINAMATH_CALUDE_soccer_camp_afternoon_l4119_411911


namespace NUMINAMATH_CALUDE_sum_g_equals_negative_one_l4119_411997

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the conditions
axiom functional_equation : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y
axiom f_equality : f (-2) = f 1
axiom f_nonzero : f 1 ≠ 0

-- State the theorem to be proved
theorem sum_g_equals_negative_one : g 1 + g (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_sum_g_equals_negative_one_l4119_411997


namespace NUMINAMATH_CALUDE_sequence_general_term_l4119_411976

theorem sequence_general_term (n : ℕ) :
  let S : ℕ → ℤ := λ k => 3 * k^2 - 2 * k
  let a : ℕ → ℤ := λ k => S k - S (k - 1)
  a n = 6 * n - 5 :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l4119_411976


namespace NUMINAMATH_CALUDE_ship_river_flow_equation_l4119_411975

theorem ship_river_flow_equation (v : ℝ) : 
  (144 / (30 + v) = 96 / (30 - v)) ↔ 
  (144 / (30 + v) = 96 / (30 - v) ∧ 
   v > 0 ∧ v < 30 ∧
   144 / (30 + v) = 96 / (30 - v)) :=
by sorry

end NUMINAMATH_CALUDE_ship_river_flow_equation_l4119_411975


namespace NUMINAMATH_CALUDE_congruence_problem_l4119_411977

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 19 = 3 → (3 * x + 18) % 19 = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l4119_411977


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l4119_411914

theorem fraction_equals_zero (x : ℝ) (h : 5 * x ≠ 0) :
  (x - 6) / (5 * x) = 0 ↔ x = 6 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l4119_411914


namespace NUMINAMATH_CALUDE_same_heads_probability_l4119_411937

/-- The number of possible outcomes when tossing two pennies -/
def keiko_outcomes : ℕ := 4

/-- The number of possible outcomes when tossing three pennies -/
def ephraim_outcomes : ℕ := 8

/-- The number of ways Keiko and Ephraim can get the same number of heads -/
def matching_outcomes : ℕ := 7

/-- The total number of possible outcomes when Keiko tosses two pennies and Ephraim tosses three pennies -/
def total_outcomes : ℕ := keiko_outcomes * ephraim_outcomes

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def probability : ℚ := matching_outcomes / total_outcomes

theorem same_heads_probability : probability = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_same_heads_probability_l4119_411937


namespace NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l4119_411908

/-- The probability of drawing a green or yellow marble from a bag -/
theorem green_or_yellow_marble_probability
  (green : ℕ) (yellow : ℕ) (white : ℕ)
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_white : white = 8) :
  (green + yellow) / (green + yellow + white) = 7 / 15 := by
sorry

end NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l4119_411908


namespace NUMINAMATH_CALUDE_flu_infection_model_l4119_411970

/-- 
Given two rounds of flu infection where:
- In each round, on average, one person infects x people
- After two rounds, a total of 144 people had the flu

This theorem states that the equation (1+x)^2 = 144 correctly models 
the total number of people infected after two rounds.
-/
theorem flu_infection_model (x : ℝ) : 
  (∃ (infected_first_round infected_second_round : ℕ),
    infected_first_round = x ∧ 
    infected_second_round = x * infected_first_round ∧
    1 + infected_first_round + infected_second_round = 144) ↔ 
  (1 + x)^2 = 144 :=
sorry

end NUMINAMATH_CALUDE_flu_infection_model_l4119_411970


namespace NUMINAMATH_CALUDE_even_function_implies_even_g_l4119_411940

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_implies_even_g
  (f g : ℝ → ℝ)
  (h1 : ∀ x, f x - x^2 = g x)
  (h2 : IsEven f) :
  IsEven g := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_even_g_l4119_411940


namespace NUMINAMATH_CALUDE_least_valid_n_l4119_411939

def is_valid (n : ℕ) : Prop :=
  ∃ k₁ k₂ : ℕ, 1 ≤ k₁ ∧ k₁ ≤ n + 1 ∧ 1 ≤ k₂ ∧ k₂ ≤ n + 1 ∧
  (n^2 - n) % k₁ = 0 ∧ (n^2 - n) % k₂ ≠ 0

theorem least_valid_n :
  is_valid 5 ∧ ∀ m : ℕ, m < 5 → ¬is_valid m :=
sorry

end NUMINAMATH_CALUDE_least_valid_n_l4119_411939


namespace NUMINAMATH_CALUDE_value_of_3x2_minus_3y2_l4119_411922

theorem value_of_3x2_minus_3y2 (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 4) : 
  3 * (x^2 - y^2) = 240 := by
sorry

end NUMINAMATH_CALUDE_value_of_3x2_minus_3y2_l4119_411922


namespace NUMINAMATH_CALUDE_intersection_value_l4119_411994

theorem intersection_value (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  (A ∩ B = {3}) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_value_l4119_411994


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4119_411980

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 1000 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 92 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4119_411980


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4119_411906

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = -3 + 2 * Complex.I) :
  z.im = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4119_411906


namespace NUMINAMATH_CALUDE_perpendicular_line_exists_l4119_411949

-- Define a line in 2D space
def Line2D := Set (ℝ × ℝ)

-- Define a point in 2D space
def Point2D := ℝ × ℝ

-- Function to check if a line is perpendicular to another line
def isPerpendicular (l1 l2 : Line2D) : Prop := sorry

-- Function to check if a point is on a line
def isPointOnLine (p : Point2D) (l : Line2D) : Prop := sorry

-- Theorem: For any line and any point, there exists a perpendicular line through that point
theorem perpendicular_line_exists (AB : Line2D) (P : Point2D) : 
  ∃ (l : Line2D), isPerpendicular l AB ∧ isPointOnLine P l := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_exists_l4119_411949


namespace NUMINAMATH_CALUDE_ellipse_equation_from_line_l4119_411915

/-- The standard equation of an ellipse -/
structure EllipseEquation where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- A line passing through a focus and vertex of an ellipse -/
structure EllipseLine where
  slope : ℝ
  intercept : ℝ

/-- The theorem statement -/
theorem ellipse_equation_from_line (l : EllipseLine) 
  (h1 : l.slope = 1/2 ∧ l.intercept = -1) 
  (h2 : ∃ (f v : ℝ × ℝ), l.slope * f.1 + l.intercept = f.2 ∧ 
                          l.slope * v.1 + l.intercept = v.2) 
  (h3 : l.slope * 0 + l.intercept = 1) :
  ∃ (e1 e2 : EllipseEquation), 
    (e1.a^2 = 5 ∧ e1.b^2 = 1) ∨ 
    (e2.a^2 = 5 ∧ e2.b^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_line_l4119_411915


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l4119_411957

theorem quadratic_form_minimum : ∀ x y : ℝ, 
  2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x - 6 * y + 1 ≥ -3 ∧ 
  (2 * (3/2)^2 + 4 * (3/2) * (1/2) + 5 * (1/2)^2 - 4 * (3/2) - 6 * (1/2) + 1 = -3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l4119_411957


namespace NUMINAMATH_CALUDE_money_distribution_l4119_411948

/-- Represents the share of money for each person -/
structure Shares :=
  (w : ℚ) (x : ℚ) (y : ℚ) (z : ℚ)

/-- The theorem statement -/
theorem money_distribution (s : Shares) :
  s.w + s.x + s.y + s.z > 0 ∧  -- Ensure total sum is positive
  s.x = 6 * s.w ∧              -- Proportion for x
  s.y = 2 * s.w ∧              -- Proportion for y
  s.z = 4 * s.w ∧              -- Proportion for z
  s.x = s.y + 1500             -- x gets $1500 more than y
  →
  s.w = 375 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l4119_411948


namespace NUMINAMATH_CALUDE_problem_solution_l4119_411910

theorem problem_solution (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) : 
  (1/3) * x^8 * y^9 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4119_411910


namespace NUMINAMATH_CALUDE_point_slope_equation_l4119_411969

/-- Given a line with slope 3 passing through the point (-1, -2),
    prove that its point-slope form equation is y + 2 = 3(x + 1) -/
theorem point_slope_equation (x y : ℝ) :
  let slope : ℝ := 3
  let point : ℝ × ℝ := (-1, -2)
  (y - point.2 = slope * (x - point.1)) ↔ (y + 2 = 3 * (x + 1)) := by
sorry

end NUMINAMATH_CALUDE_point_slope_equation_l4119_411969


namespace NUMINAMATH_CALUDE_snacks_expenditure_l4119_411942

theorem snacks_expenditure (total : ℝ) (movies books music ice_cream snacks : ℝ) :
  total = 50 ∧
  movies = (1/4) * total ∧
  books = (1/8) * total ∧
  music = (1/4) * total ∧
  ice_cream = (1/5) * total ∧
  snacks = total - (movies + books + music + ice_cream) →
  snacks = 8.75 := by sorry

end NUMINAMATH_CALUDE_snacks_expenditure_l4119_411942


namespace NUMINAMATH_CALUDE_harkamal_mangoes_l4119_411951

/-- Calculates the amount of mangoes purchased given the total cost, grape quantity, grape price, and mango price -/
def mangoes_purchased (total_cost : ℕ) (grape_quantity : ℕ) (grape_price : ℕ) (mango_price : ℕ) : ℕ :=
  (total_cost - grape_quantity * grape_price) / mango_price

theorem harkamal_mangoes :
  mangoes_purchased 1145 8 70 65 = 9 := by
sorry

end NUMINAMATH_CALUDE_harkamal_mangoes_l4119_411951


namespace NUMINAMATH_CALUDE_three_squared_sum_equals_three_cubed_l4119_411918

theorem three_squared_sum_equals_three_cubed (a : ℕ) :
  3^2 + 3^2 + 3^2 = 3^a → a = 3 := by
sorry

end NUMINAMATH_CALUDE_three_squared_sum_equals_three_cubed_l4119_411918


namespace NUMINAMATH_CALUDE_four_points_not_coplanar_iff_any_three_not_collinear_lines_no_common_point_iff_skew_l4119_411971

-- Define the types for points and lines in space
variable (Point Line : Type)

-- Define the properties
variable (coplanar : Point → Point → Point → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (have_common_point : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Theorem 1
theorem four_points_not_coplanar_iff_any_three_not_collinear 
  (p1 p2 p3 p4 : Point) : 
  ¬(coplanar p1 p2 p3 p4) ↔ 
  (¬(collinear p1 p2 p3) ∧ ¬(collinear p1 p2 p4) ∧ 
   ¬(collinear p1 p3 p4) ∧ ¬(collinear p2 p3 p4)) :=
sorry

-- Theorem 2
theorem lines_no_common_point_iff_skew (l1 l2 : Line) :
  ¬(have_common_point l1 l2) ↔ skew l1 l2 :=
sorry

end NUMINAMATH_CALUDE_four_points_not_coplanar_iff_any_three_not_collinear_lines_no_common_point_iff_skew_l4119_411971


namespace NUMINAMATH_CALUDE_example_bridge_properties_l4119_411933

/-- Represents a railway bridge with its length and load capacity. -/
structure RailwayBridge where
  length : ℝ  -- Length in kilometers
  loadCapacity : ℝ  -- Load capacity in tons

/-- Defines a specific railway bridge with given properties. -/
def exampleBridge : RailwayBridge :=
  { length := 2,
    loadCapacity := 80 }

/-- Theorem stating the properties of the example bridge. -/
theorem example_bridge_properties :
  exampleBridge.length = 2 ∧ exampleBridge.loadCapacity = 80 := by
  sorry

#check example_bridge_properties

end NUMINAMATH_CALUDE_example_bridge_properties_l4119_411933


namespace NUMINAMATH_CALUDE_three_cubes_volume_l4119_411965

-- Define the volume of a cube
def cubeVolume (edge : ℝ) : ℝ := edge ^ 3

-- Define the total volume of three cubes
def totalVolume (edge1 edge2 edge3 : ℝ) : ℝ :=
  cubeVolume edge1 + cubeVolume edge2 + cubeVolume edge3

-- Theorem statement
theorem three_cubes_volume :
  totalVolume 3 5 6 = 368 := by
  sorry

end NUMINAMATH_CALUDE_three_cubes_volume_l4119_411965


namespace NUMINAMATH_CALUDE_size_relationship_l4119_411998

theorem size_relationship (a b c : ℝ) 
  (ha : a = (0.2 : ℝ) ^ (1.5 : ℝ))
  (hb : b = (2 : ℝ) ^ (0.1 : ℝ))
  (hc : c = (0.2 : ℝ) ^ (1.3 : ℝ)) :
  a < c ∧ c < b :=
by sorry

end NUMINAMATH_CALUDE_size_relationship_l4119_411998


namespace NUMINAMATH_CALUDE_min_values_theorem_l4119_411984

theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 1) :
  (∀ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 → 1 / x + 2 / y ≥ 9) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 ∧ 1 / x + 2 / y = 9) ∧
  (∀ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 → 2^x + 4^y ≥ 2 * Real.sqrt 2) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 ∧ 2^x + 4^y = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_values_theorem_l4119_411984


namespace NUMINAMATH_CALUDE_sample_size_proof_l4119_411978

theorem sample_size_proof (n : ℕ) (f₁ f₂ f₃ f₄ f₅ f₆ : ℕ) : 
  f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = n →
  f₁ + f₂ + f₃ = 27 →
  ∃ (k : ℕ), f₁ = 2*k ∧ f₂ = 3*k ∧ f₃ = 4*k ∧ f₄ = 6*k ∧ f₅ = 4*k ∧ f₆ = k →
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_sample_size_proof_l4119_411978


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l4119_411935

theorem cubic_equation_solution :
  ∃! x : ℝ, (x^3 - 5*x^2 + 5*x - 1) + (x - 1) = 0 :=
by
  -- The unique solution is x = 2
  use 2
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l4119_411935


namespace NUMINAMATH_CALUDE_peach_count_l4119_411920

/-- The number of peaches Sally had initially -/
def initial_peaches : ℕ := 13

/-- The number of peaches Sally picked -/
def picked_peaches : ℕ := 55

/-- The total number of peaches Sally has now -/
def total_peaches : ℕ := initial_peaches + picked_peaches

theorem peach_count : total_peaches = 68 := by
  sorry

end NUMINAMATH_CALUDE_peach_count_l4119_411920


namespace NUMINAMATH_CALUDE_checkers_draw_fraction_l4119_411931

theorem checkers_draw_fraction (dan_wins eve_wins : ℚ) (h1 : dan_wins = 4/9) (h2 : eve_wins = 1/3) :
  1 - (dan_wins + eve_wins) = 2/9 := by
sorry

end NUMINAMATH_CALUDE_checkers_draw_fraction_l4119_411931


namespace NUMINAMATH_CALUDE_incorrect_expression_l4119_411913

theorem incorrect_expression (a b : ℝ) : 
  2 * ((a^2 + b^2) - a*b) ≠ (a + b)^2 - 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l4119_411913


namespace NUMINAMATH_CALUDE_equation_describes_spiral_l4119_411958

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The equation r * θ = c -/
def spiralEquation (p : CylindricalPoint) (c : ℝ) : Prop :=
  p.r * p.θ = c

/-- A spiral in cylindrical coordinates -/
def isSpiral (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, ∀ p ∈ S, spiralEquation p c

/-- The shape described by r * θ = c is a spiral -/
theorem equation_describes_spiral (c : ℝ) :
  isSpiral {p : CylindricalPoint | spiralEquation p c} :=
sorry

end NUMINAMATH_CALUDE_equation_describes_spiral_l4119_411958
