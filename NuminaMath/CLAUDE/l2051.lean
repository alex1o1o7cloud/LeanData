import Mathlib

namespace NUMINAMATH_CALUDE_flour_cups_needed_l2051_205164

-- Define the total number of 1/4 cup scoops
def total_scoops : ℚ := 15

-- Define the amounts of other ingredients in cups
def white_sugar : ℚ := 1
def brown_sugar : ℚ := 1/4
def oil : ℚ := 1/2

-- Define the conversion factor from scoops to cups
def scoops_to_cups : ℚ := 1/4

-- Theorem to prove
theorem flour_cups_needed :
  let other_ingredients_scoops := white_sugar / scoops_to_cups + brown_sugar / scoops_to_cups + oil / scoops_to_cups
  let flour_scoops := total_scoops - other_ingredients_scoops
  let flour_cups := flour_scoops * scoops_to_cups
  flour_cups = 2 := by sorry

end NUMINAMATH_CALUDE_flour_cups_needed_l2051_205164


namespace NUMINAMATH_CALUDE_tom_dance_duration_l2051_205153

/-- Given that Tom dances 4 times a week for 10 years and danced for a total of 4160 hours,
    prove that he dances for 2 hours at a time. -/
theorem tom_dance_duration (
  dances_per_week : ℕ)
  (years : ℕ)
  (total_hours : ℕ)
  (h1 : dances_per_week = 4)
  (h2 : years = 10)
  (h3 : total_hours = 4160) :
  total_hours / (dances_per_week * years * 52) = 2 := by
sorry

end NUMINAMATH_CALUDE_tom_dance_duration_l2051_205153


namespace NUMINAMATH_CALUDE_sum_equals_two_thirds_l2051_205113

theorem sum_equals_two_thirds :
  let original_sum := (1:ℚ)/3 + 1/6 + 1/9 + 1/12 + 1/15 + 1/18
  let removed_terms := 1/12 + 1/15
  let remaining_sum := original_sum - removed_terms
  remaining_sum = 2/3 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_two_thirds_l2051_205113


namespace NUMINAMATH_CALUDE_expression_change_l2051_205121

theorem expression_change (x : ℝ) (b : ℝ) (h : b > 0) :
  (b*x)^2 - 5 - (x^2 - 5) = (b^2 - 1) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_change_l2051_205121


namespace NUMINAMATH_CALUDE_num_planes_determined_by_skew_lines_l2051_205198

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line in 3D space
  -- (We don't need to fully implement this for the statement)

/-- A point in 3D space -/
structure Point3D where
  -- Define properties of a point in 3D space
  -- (We don't need to fully implement this for the statement)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane in 3D space
  -- (We don't need to fully implement this for the statement)

/-- Two lines are skew if they are not parallel and do not intersect -/
def areSkewLines (l1 l2 : Line3D) : Prop :=
  sorry

/-- A point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- A plane is determined by a line and a point not on that line -/
def planeDeterminedByLineAndPoint (l : Line3D) (p : Point3D) : Plane3D :=
  sorry

/-- The number of unique planes determined by two skew lines and points on them -/
def numUniquePlanes (a b : Line3D) (pointsOnA pointsOnB : Finset Point3D) : ℕ :=
  sorry

theorem num_planes_determined_by_skew_lines 
  (a b : Line3D) 
  (pointsOnA pointsOnB : Finset Point3D) 
  (h_skew : areSkewLines a b)
  (h_pointsA : ∀ p ∈ pointsOnA, pointOnLine p a)
  (h_pointsB : ∀ p ∈ pointsOnB, pointOnLine p b)
  (h_countA : pointsOnA.card = 5)
  (h_countB : pointsOnB.card = 4) :
  numUniquePlanes a b pointsOnA pointsOnB = 5 := by
  sorry

end NUMINAMATH_CALUDE_num_planes_determined_by_skew_lines_l2051_205198


namespace NUMINAMATH_CALUDE_jacobs_age_l2051_205186

/-- Proves Jacob's age given the conditions of the problem -/
theorem jacobs_age :
  ∀ (rehana_age phoebe_age jacob_age : ℕ),
  rehana_age = 25 →
  rehana_age + 5 = 3 * (phoebe_age + 5) →
  jacob_age = (3 * phoebe_age) / 5 →
  jacob_age = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jacobs_age_l2051_205186


namespace NUMINAMATH_CALUDE_distinct_values_of_binomial_sum_l2051_205134

theorem distinct_values_of_binomial_sum : ∃ (S : Finset ℕ),
  (∀ r : ℕ, r > 0 ∧ r + 1 ≤ 10 ∧ 17 - r ≤ 10 →
    (Nat.choose 10 (r + 1) + Nat.choose 10 (17 - r)) ∈ S) ∧
  Finset.card S = 2 := by
  sorry

end NUMINAMATH_CALUDE_distinct_values_of_binomial_sum_l2051_205134


namespace NUMINAMATH_CALUDE_min_value_product_l2051_205156

theorem min_value_product (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_sum : x/y + y/z + z/x + y/x + z/y + x/z = 6) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l2051_205156


namespace NUMINAMATH_CALUDE_paint_usage_l2051_205166

/-- Calculates the total amount of paint used by an artist for large and small canvases -/
theorem paint_usage (large_paint : ℕ) (small_paint : ℕ) (large_count : ℕ) (small_count : ℕ) :
  large_paint = 3 →
  small_paint = 2 →
  large_count = 3 →
  small_count = 4 →
  large_paint * large_count + small_paint * small_count = 17 := by
  sorry


end NUMINAMATH_CALUDE_paint_usage_l2051_205166


namespace NUMINAMATH_CALUDE_root_difference_quadratic_equation_l2051_205147

theorem root_difference_quadratic_equation :
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := -12
  let larger_root : ℝ := (-b + (b^2 - 4*a*c).sqrt) / (2*a)
  let smaller_root : ℝ := (-b - (b^2 - 4*a*c).sqrt) / (2*a)
  larger_root - smaller_root = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_equation_l2051_205147


namespace NUMINAMATH_CALUDE_some_number_value_l2051_205180

theorem some_number_value (some_number : ℝ) : 
  (40 / some_number) * (40 / 80) = 1 → some_number = 80 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l2051_205180


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2051_205188

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

-- Statement to prove
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2051_205188


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2051_205107

theorem equal_roots_quadratic (m : ℝ) :
  (∃ x : ℝ, 4 * x^2 - 6 * x + m = 0 ∧
   ∀ y : ℝ, 4 * y^2 - 6 * y + m = 0 → y = x) →
  m = 9/4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2051_205107


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l2051_205104

theorem triangle_angle_theorem (a b c : ℝ) : 
  (a = 2 * b) →                 -- One angle is twice the second angle
  (c = b + 30) →                -- The third angle is 30° more than the second angle
  (a + b + c = 180) →           -- Sum of angles in a triangle is 180°
  (a = 75 ∧ b = 37.5 ∧ c = 67.5) -- The measures of the angles are 75°, 37.5°, and 67.5°
  := by sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l2051_205104


namespace NUMINAMATH_CALUDE_paulas_walking_distance_l2051_205111

/-- Represents a pedometer with a maximum step count before reset --/
structure Pedometer where
  max_steps : ℕ
  steps_per_km : ℕ

/-- Represents the yearly walking data --/
structure YearlyWalkingData where
  pedometer : Pedometer
  resets : ℕ
  final_reading : ℕ

def calculate_total_steps (data : YearlyWalkingData) : ℕ :=
  data.resets * (data.pedometer.max_steps + 1) + data.final_reading

def calculate_kilometers (data : YearlyWalkingData) : ℚ :=
  (calculate_total_steps data : ℚ) / data.pedometer.steps_per_km

theorem paulas_walking_distance (data : YearlyWalkingData) 
  (h1 : data.pedometer.max_steps = 49999)
  (h2 : data.pedometer.steps_per_km = 1200)
  (h3 : data.resets = 76)
  (h4 : data.final_reading = 25000) :
  ∃ (k : ℕ), k ≥ 3187 ∧ k ≤ 3200 ∧ calculate_kilometers data = k := by
  sorry

#eval calculate_kilometers {
  pedometer := { max_steps := 49999, steps_per_km := 1200 },
  resets := 76,
  final_reading := 25000
}

end NUMINAMATH_CALUDE_paulas_walking_distance_l2051_205111


namespace NUMINAMATH_CALUDE_perpendicular_and_tangent_l2051_205165

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

/-- The given line -/
def L1 : Set (ℝ × ℝ) := {(x, y) | 2*x - 6*y + 1 = 0}

/-- The line to be proven -/
def L2 : Set (ℝ × ℝ) := {(x, y) | 3*x + y + 6 = 0}

theorem perpendicular_and_tangent :
  (∃ (a b : ℝ), (a, b) ∈ L2 ∧ f a = b) ∧  -- L2 is tangent to the curve
  (∀ (x1 y1 x2 y2 : ℝ), (x1, y1) ∈ L1 ∧ (x2, y2) ∈ L1 ∧ x1 ≠ x2 →
    (x1 - x2) * (3 * (y1 - y2)) = -1) -- L1 and L2 are perpendicular
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_and_tangent_l2051_205165


namespace NUMINAMATH_CALUDE_function_constraint_l2051_205127

theorem function_constraint (a : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → a * x + 6 ≤ 10) ↔ (a = -4 ∨ a = 2 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_constraint_l2051_205127


namespace NUMINAMATH_CALUDE_solution_set_l2051_205139

noncomputable def Solutions (a : ℝ) : Set (ℝ × ℝ × ℝ) :=
  { (1, Real.sqrt (-a), -Real.sqrt (-a)),
    (1, -Real.sqrt (-a), Real.sqrt (-a)),
    (Real.sqrt (-a), -Real.sqrt (-a), 1),
    (-Real.sqrt (-a), 1, Real.sqrt (-a)),
    (Real.sqrt (-a), 1, -Real.sqrt (-a)),
    (-Real.sqrt (-a), Real.sqrt (-a), 1) }

theorem solution_set (a : ℝ) :
  ∀ (x y z : ℝ),
    (x + y + z = 1 ∧
     1/x + 1/y + 1/z = 1 ∧
     x*y*z = a) ↔
    (x, y, z) ∈ Solutions a := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l2051_205139


namespace NUMINAMATH_CALUDE_inverse_fraction_minus_abs_diff_l2051_205144

theorem inverse_fraction_minus_abs_diff : (1/3)⁻¹ - |Real.sqrt 3 - 3| = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_fraction_minus_abs_diff_l2051_205144


namespace NUMINAMATH_CALUDE_unique_triplet_l2051_205174

theorem unique_triplet :
  ∃! (a b c : ℕ), 1 < a ∧ a < b ∧ b < c ∧
  (c ∣ a * b + 1) ∧
  (b ∣ a * c + 1) ∧
  (a ∣ b * c + 1) ∧
  a = 2 ∧ b = 3 ∧ c = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_triplet_l2051_205174


namespace NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l2051_205168

theorem roots_of_cubic_polynomial :
  let f : ℝ → ℝ := λ x => x^3 - 2*x^2 - 5*x + 6
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3) := by sorry

end NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l2051_205168


namespace NUMINAMATH_CALUDE_expression_one_equality_expression_two_equality_l2051_205170

-- Expression 1
theorem expression_one_equality : 
  0.25 * (-1/2)^(-4) - 4 / 2^0 - (1/16)^(-1/2) = -4 := by sorry

-- Expression 2
theorem expression_two_equality :
  2 * (Real.log 2 / Real.log 3) - 
  (Real.log (32/9) / Real.log 3) + 
  (Real.log 8 / Real.log 3) - 
  ((Real.log 3 / Real.log 4) + (Real.log 3 / Real.log 8)) * 
  ((Real.log 2 / Real.log 3) + (Real.log 2 / Real.log 9)) = 3/4 := by sorry

end NUMINAMATH_CALUDE_expression_one_equality_expression_two_equality_l2051_205170


namespace NUMINAMATH_CALUDE_value_subtracted_after_multiplication_l2051_205197

theorem value_subtracted_after_multiplication (N : ℝ) (V : ℝ) : 
  N = 12 → 4 * N - V = 9 * (N - 7) → V = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_subtracted_after_multiplication_l2051_205197


namespace NUMINAMATH_CALUDE_coopers_fence_depth_l2051_205123

/-- Proves that the depth of each wall in Cooper's fence is 2 bricks -/
theorem coopers_fence_depth (num_walls : ℕ) (wall_length : ℕ) (wall_height : ℕ) (total_bricks : ℕ) :
  num_walls = 4 →
  wall_length = 20 →
  wall_height = 5 →
  total_bricks = 800 →
  (total_bricks / (num_walls * wall_length * wall_height) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_coopers_fence_depth_l2051_205123


namespace NUMINAMATH_CALUDE_linear_function_through_point_l2051_205106

theorem linear_function_through_point (k : ℝ) : 
  (∀ x : ℝ, (k * x = k * 3) → (k * x = 1)) → k = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_point_l2051_205106


namespace NUMINAMATH_CALUDE_cost_per_item_jings_purchase_l2051_205135

/-- Given a total cost and number of identical items, prove that the cost per item
    is equal to the total cost divided by the number of items. -/
theorem cost_per_item (total_cost : ℝ) (num_items : ℕ) (h : num_items > 0) :
  let cost_per_item := total_cost / num_items
  cost_per_item = total_cost / num_items :=
by
  sorry

/-- For Jing's purchase of 8 identical items with a total cost of $26,
    prove that the cost per item is $26 divided by 8. -/
theorem jings_purchase :
  let total_cost : ℝ := 26
  let num_items : ℕ := 8
  let cost_per_item := total_cost / num_items
  cost_per_item = 26 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_per_item_jings_purchase_l2051_205135


namespace NUMINAMATH_CALUDE_tomato_harvest_ratio_l2051_205114

/-- Proves that the ratio of tomatoes harvested on Wednesday to Thursday is 2:1 --/
theorem tomato_harvest_ratio :
  ∀ (thursday_harvest : ℕ),
  400 + thursday_harvest + (700 + 700) = 2000 →
  (400 : ℚ) / thursday_harvest = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_harvest_ratio_l2051_205114


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2051_205131

def complex_number : ℂ := 2 - Complex.I

theorem complex_number_in_fourth_quadrant :
  Real.sign (complex_number.re) = 1 ∧ Real.sign (complex_number.im) = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2051_205131


namespace NUMINAMATH_CALUDE_rose_count_prediction_l2051_205133

/-- Given a sequence of rose counts for four consecutive months, where the differences
    between consecutive counts form an arithmetic sequence with a common difference of 12,
    prove that the next term in the sequence will be 224. -/
theorem rose_count_prediction (a b c d : ℕ) (hab : b - a = 18) (hbc : c - b = 30) (hcd : d - c = 42) :
  d + (d - c + 12) = 224 :=
sorry

end NUMINAMATH_CALUDE_rose_count_prediction_l2051_205133


namespace NUMINAMATH_CALUDE_factors_of_12_and_18_l2051_205130

def factors (n : ℕ) : Set ℕ := {x | x ∣ n}

theorem factors_of_12_and_18 : 
  factors 12 = {1, 2, 3, 4, 6, 12} ∧ factors 18 = {1, 2, 3, 6, 9, 18} := by
  sorry

end NUMINAMATH_CALUDE_factors_of_12_and_18_l2051_205130


namespace NUMINAMATH_CALUDE_chili_composition_l2051_205151

/-- Represents the number of cans of each ingredient in a normal batch of chili -/
structure ChiliBatch where
  chilis : ℕ
  beans : ℕ
  tomatoes : ℕ

/-- Calculates the total number of cans in a batch of chili -/
def totalCans (batch : ChiliBatch) : ℕ :=
  batch.chilis + batch.beans + batch.tomatoes

/-- Calculates the percentage of more tomatoes than beans -/
def percentageMoreTomatoes (batch : ChiliBatch) : ℚ :=
  ((batch.tomatoes : ℚ) - (batch.beans : ℚ)) / (batch.beans : ℚ) * 100

theorem chili_composition (batch : ChiliBatch) 
    (h1 : batch.chilis = 1)
    (h2 : batch.beans = 2)
    (h3 : 4 * totalCans batch = 24) :
  percentageMoreTomatoes batch = 50 := by
  sorry

end NUMINAMATH_CALUDE_chili_composition_l2051_205151


namespace NUMINAMATH_CALUDE_insect_meeting_point_l2051_205189

/-- Triangle PQR with given side lengths -/
structure Triangle (PQ QR PR : ℝ) where
  positive : 0 < PQ ∧ 0 < QR ∧ 0 < PR
  triangle_inequality : PQ + QR > PR ∧ QR + PR > PQ ∧ PR + PQ > QR

/-- Point S where insects meet -/
def MeetingPoint (t : Triangle PQ QR PR) := 
  {S : ℝ // 0 ≤ S ∧ S ≤ QR}

/-- Theorem stating that QS = 5 under given conditions -/
theorem insect_meeting_point 
  (t : Triangle 7 8 9) 
  (S : MeetingPoint t) : 
  S.val = 5 := by sorry

end NUMINAMATH_CALUDE_insect_meeting_point_l2051_205189


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_9_l2051_205110

theorem smallest_digit_for_divisibility_by_9 :
  ∃ (d : Nat), d < 10 ∧ (562000 + d * 100 + 48) % 9 = 0 ∧
  ∀ (k : Nat), k < d → k < 10 → (562000 + k * 100 + 48) % 9 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_9_l2051_205110


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_1_to_18_not_19_20_l2051_205146

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def divisible_up_to (n m : ℕ) : Prop := ∀ i : ℕ, 1 ≤ i → i ≤ m → is_divisible n i

theorem smallest_number_divisible_by_1_to_18_not_19_20 :
  ∃ n : ℕ, 
    n > 0 ∧
    divisible_up_to n 18 ∧
    ¬(is_divisible n 19) ∧
    ¬(is_divisible n 20) ∧
    ∀ m : ℕ, m > 0 → divisible_up_to m 18 → ¬(is_divisible m 19) → ¬(is_divisible m 20) → n ≤ m :=
by
  sorry

#eval 12252240

end NUMINAMATH_CALUDE_smallest_number_divisible_by_1_to_18_not_19_20_l2051_205146


namespace NUMINAMATH_CALUDE_reciprocal_sum_equality_l2051_205140

theorem reciprocal_sum_equality (a b c : ℝ) (n : ℕ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : 1 / a + 1 / b + 1 / c = 1 / (a + b + c)) : 
  1 / a^(2*n+1) + 1 / b^(2*n+1) + 1 / c^(2*n+1) = 
  1 / (a^(2*n+1) + b^(2*n+1) + c^(2*n+1)) := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equality_l2051_205140


namespace NUMINAMATH_CALUDE_xyz_sum_product_l2051_205199

theorem xyz_sum_product (x y z : ℝ) 
  (h1 : 3 * (x + y + z) = x^2 + y^2 + z^2) 
  (h2 : x + y + z = 3) : 
  x * y + x * z + y * z = 0 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_product_l2051_205199


namespace NUMINAMATH_CALUDE_max_a_for_inequality_solution_R_l2051_205102

theorem max_a_for_inequality_solution_R : 
  ∃ (a_max : ℝ), 
    (∀ (a : ℝ), (∀ (x : ℝ), |x - a| + |x - 3| ≥ 2*a) → a ≤ a_max) ∧
    (∀ (x : ℝ), |x - a_max| + |x - 3| ≥ 2*a_max) ∧
    (∀ (a : ℝ), a > a_max → ∃ (x : ℝ), |x - a| + |x - 3| < 2*a) ∧
    a_max = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_inequality_solution_R_l2051_205102


namespace NUMINAMATH_CALUDE_dice_sides_l2051_205125

theorem dice_sides (total_dice : ℕ) (total_sides : ℕ) (sides_per_die : ℕ) 
  (h1 : total_dice = 8) 
  (h2 : total_sides = 48) 
  (h3 : total_sides = total_dice * sides_per_die) : 
  sides_per_die = 6 := by
  sorry

end NUMINAMATH_CALUDE_dice_sides_l2051_205125


namespace NUMINAMATH_CALUDE_solve_candy_problem_l2051_205157

def candy_problem (kit_kat : ℕ) (nerds : ℕ) (lollipops : ℕ) (baby_ruth : ℕ) (remaining : ℕ) : Prop :=
  let hershey := 3 * kit_kat
  let reese := baby_ruth / 2
  let total := kit_kat + hershey + nerds + lollipops + baby_ruth + reese
  let given_away := total - remaining
  given_away = 5

theorem solve_candy_problem :
  candy_problem 5 8 11 10 49 := by sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l2051_205157


namespace NUMINAMATH_CALUDE_sum_of_ages_five_children_l2051_205184

/-- Calculates the sum of ages for a group of children born at regular intervals -/
def sumOfAges (numChildren : ℕ) (ageInterval : ℕ) (youngestAge : ℕ) : ℕ :=
  let ages := List.range numChildren |>.map (fun i => youngestAge + i * ageInterval)
  ages.sum

/-- Proves that the sum of ages for 5 children born at 2-year intervals, with the youngest being 6, is 50 -/
theorem sum_of_ages_five_children :
  sumOfAges 5 2 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_five_children_l2051_205184


namespace NUMINAMATH_CALUDE_cindy_calculation_l2051_205190

theorem cindy_calculation (h : 50^2 = 2500) : 50^2 - 49^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l2051_205190


namespace NUMINAMATH_CALUDE_cars_meeting_time_l2051_205176

/-- Given two cars driving toward each other, prove that they meet in 4 hours -/
theorem cars_meeting_time (speed1 : ℝ) (speed2 : ℝ) (distance : ℝ) : 
  speed1 = 100 →
  speed1 = 1.25 * speed2 →
  distance = 720 →
  distance / (speed1 + speed2) = 4 := by
sorry


end NUMINAMATH_CALUDE_cars_meeting_time_l2051_205176


namespace NUMINAMATH_CALUDE_count_D3_le_200_eq_9_l2051_205192

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- Count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D3_le_200 : ℕ := sorry

theorem count_D3_le_200_eq_9 : count_D3_le_200 = 9 := by sorry

end NUMINAMATH_CALUDE_count_D3_le_200_eq_9_l2051_205192


namespace NUMINAMATH_CALUDE_problem_solution_l2051_205160

theorem problem_solution : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2051_205160


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2051_205181

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x^2 - (a + 2) * x + 2 < 0}
  (a = 0 → S = {x : ℝ | x > 1}) ∧
  (0 < a ∧ a < 2 → S = {x : ℝ | 1 < x ∧ x < 2/a}) ∧
  (a = 2 → S = ∅) ∧
  (a > 2 → S = {x : ℝ | 2/a < x ∧ x < 1}) ∧
  (a < 0 → S = {x : ℝ | x < 2/a ∨ x > 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2051_205181


namespace NUMINAMATH_CALUDE_min_a_value_l2051_205177

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem min_a_value (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, g (-x) = g x) →   -- g is even
  (∀ x, f x + g x = 2^x) →  -- f(x) + g(x) = 2^x
  (∀ x ∈ Set.Icc 1 2, a * f x + g (2*x) ≥ 0) →  -- inequality holds for x ∈ [1, 2]
  a ≥ -17/6 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l2051_205177


namespace NUMINAMATH_CALUDE_concert_ticket_discount_l2051_205185

theorem concert_ticket_discount (normal_price : ℝ) (scalper_markup : ℝ) (scalper_discount : ℝ) (total_paid : ℝ) :
  normal_price = 50 →
  scalper_markup = 2.4 →
  scalper_discount = 10 →
  total_paid = 360 →
  ∃ (discounted_price : ℝ),
    2 * normal_price + 2 * (scalper_markup * normal_price - scalper_discount / 2) + discounted_price = total_paid ∧
    discounted_price / normal_price = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_discount_l2051_205185


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l2051_205117

theorem employee_pay_percentage (total_pay B_pay : ℝ) (h1 : total_pay = 550) (h2 : B_pay = 249.99999999999997) :
  let A_pay := total_pay - B_pay
  (A_pay / B_pay) * 100 = 120 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l2051_205117


namespace NUMINAMATH_CALUDE_hyperbola_circle_max_radius_l2051_205124

/-- Given a hyperbola and a circle with specific properties, prove that the maximum radius of the circle is √3 -/
theorem hyperbola_circle_max_radius (a b r : ℝ) (e : ℝ) :
  a > 0 →
  b > 0 →
  r > 0 →
  e ≤ 2 →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, (x - 2)^2 + y^2 = r^2) →
  (∃ x y : ℝ, b * x + a * y = 0 ∨ b * x - a * y = 0) →
  (∀ x y : ℝ, (b * x + a * y = 0 ∨ b * x - a * y = 0) → 
    ((x - 2)^2 + y^2 = r^2 → (x - 2)^2 + y^2 ≥ r^2)) →
  r ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_max_radius_l2051_205124


namespace NUMINAMATH_CALUDE_complex_real_condition_l2051_205173

theorem complex_real_condition (m : ℝ) : 
  (Complex.I : ℂ) * (1 + m * Complex.I) + (m^2 : ℂ) * (1 + m * Complex.I) ∈ Set.range (Complex.ofReal) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_real_condition_l2051_205173


namespace NUMINAMATH_CALUDE_exist_consecutive_lucky_tickets_l2051_205103

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a number is a six-digit number -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- A predicate that checks if a number is lucky (sum of digits divisible by 7) -/
def is_lucky (n : ℕ) : Prop := sum_of_digits n % 7 = 0

/-- Theorem stating that there exist two consecutive six-digit numbers that are both lucky -/
theorem exist_consecutive_lucky_tickets : 
  ∃ n : ℕ, is_six_digit n ∧ is_six_digit (n + 1) ∧ is_lucky n ∧ is_lucky (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_exist_consecutive_lucky_tickets_l2051_205103


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2051_205105

theorem division_remainder_proof (dividend : ℕ) (divisor : ℚ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 16698 →
  divisor = 187.46067415730337 →
  quotient = 89 →
  dividend = (divisor * quotient).floor + remainder →
  remainder = 14 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2051_205105


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_l2051_205150

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 37

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 60

theorem fgh_supermarkets_count : 
  us_supermarkets = 37 ∧ 
  us_supermarkets + canada_supermarkets = total_supermarkets ∧
  us_supermarkets = canada_supermarkets + 14 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_l2051_205150


namespace NUMINAMATH_CALUDE_negative_modulus_of_complex_l2051_205183

theorem negative_modulus_of_complex (z : ℂ) (h : z = 6 + 8*I) : -Complex.abs z = -10 := by
  sorry

end NUMINAMATH_CALUDE_negative_modulus_of_complex_l2051_205183


namespace NUMINAMATH_CALUDE_simplify_scientific_notation_l2051_205115

theorem simplify_scientific_notation :
  (12 * 10^10) / (6 * 10^2) = 2 * 10^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_scientific_notation_l2051_205115


namespace NUMINAMATH_CALUDE_ellipse_equation_l2051_205195

/-- The standard equation of an ellipse with given foci and a point on the ellipse -/
theorem ellipse_equation (P A B : ℝ × ℝ) (h_P : P = (5/2, -3/2)) (h_A : A = (-2, 0)) (h_B : B = (2, 0)) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 ↔
    (x - A.1)^2 + (y - A.2)^2 + (x - B.1)^2 + (y - B.2)^2 = 4 * a^2 ∧
    (x - P.1)^2 + (y - P.2)^2 = ((x - A.1)^2 + (y - A.2)^2)^(1/2) * ((x - B.1)^2 + (y - B.2)^2)^(1/2)) ∧
  a^2 = 10 ∧ b^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2051_205195


namespace NUMINAMATH_CALUDE_negation_equivalence_l2051_205116

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2051_205116


namespace NUMINAMATH_CALUDE_elle_weekly_practice_hours_l2051_205172

/-- The number of minutes Elle practices piano on a weekday -/
def weekday_practice : ℕ := 30

/-- The number of weekdays Elle practices piano -/
def weekday_count : ℕ := 5

/-- The factor by which Elle's Saturday practice is longer than a weekday practice -/
def saturday_factor : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating that Elle spends 4 hours practicing piano each week -/
theorem elle_weekly_practice_hours : 
  (weekday_practice * weekday_count + weekday_practice * saturday_factor) / minutes_per_hour = 4 := by
  sorry

end NUMINAMATH_CALUDE_elle_weekly_practice_hours_l2051_205172


namespace NUMINAMATH_CALUDE_semicircle_area_ratio_l2051_205143

theorem semicircle_area_ratio (r : ℝ) (h : r > 0) :
  let semicircle_area := π * (r / Real.sqrt 2)^2 / 2
  let circle_area := π * r^2
  2 * semicircle_area / circle_area = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_semicircle_area_ratio_l2051_205143


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_l2051_205193

/-- Represents the set of cards with their numbers -/
def cards : Multiset ℕ := {1, 1, 1, 2, 2, 3, 4}

/-- The number of cards drawn -/
def draw_count : ℕ := 4

/-- Calculates the number of different four-digit numbers that can be formed -/
noncomputable def four_digit_numbers (c : Multiset ℕ) (d : ℕ) : ℕ := sorry

/-- Theorem stating that the number of different four-digit numbers is 114 -/
theorem four_digit_numbers_count : four_digit_numbers cards draw_count = 114 := by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_l2051_205193


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_15_l2051_205169

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  (List.range n).map factorial |> List.sum

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_sum_factorials_15 :
  units_digit (sum_factorials 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_15_l2051_205169


namespace NUMINAMATH_CALUDE_expand_expression_l2051_205101

theorem expand_expression (x : ℝ) : (2*x - 3) * (2*x + 3) * (4*x^2 + 9) = 4*x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2051_205101


namespace NUMINAMATH_CALUDE_p_amount_l2051_205163

theorem p_amount : ∃ (p : ℚ), p = 49 ∧ p = (2 * (1/7) * p + 35) := by
  sorry

end NUMINAMATH_CALUDE_p_amount_l2051_205163


namespace NUMINAMATH_CALUDE_pauls_crayons_l2051_205141

/-- Given Paul's crayon situation, prove the difference between given and lost crayons. -/
theorem pauls_crayons (initial : ℕ) (given : ℕ) (lost : ℕ) 
  (h1 : initial = 589) 
  (h2 : given = 571) 
  (h3 : lost = 161) : 
  given - lost = 410 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_l2051_205141


namespace NUMINAMATH_CALUDE_cubic_polynomial_relation_l2051_205175

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 7

theorem cubic_polynomial_relation (h : ℝ → ℝ) 
  (h_cubic : ∃ a b c d : ℝ, ∀ x, h x = a*x^3 + b*x^2 + c*x + d)
  (h_zero : h 0 = 7)
  (h_roots : ∀ r : ℝ, f r = 0 → ∃ s : ℝ, h s = 0 ∧ s = r^3) :
  h (-8) = -1813 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_relation_l2051_205175


namespace NUMINAMATH_CALUDE_fraction_equality_l2051_205109

theorem fraction_equality (w x y : ℝ) (hw : w / x = 1 / 3) (hxy : (x + y) / y = 3) :
  w / y = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2051_205109


namespace NUMINAMATH_CALUDE_unknown_road_length_l2051_205118

/-- Represents a road network with four cities and five roads. -/
structure RoadNetwork where
  /-- The length of the first known road -/
  road1 : ℕ
  /-- The length of the second known road -/
  road2 : ℕ
  /-- The length of the third known road -/
  road3 : ℕ
  /-- The length of the fourth known road -/
  road4 : ℕ
  /-- The length of the unknown road -/
  x : ℕ

/-- The theorem stating that the only possible value for the unknown road length is 17 km. -/
theorem unknown_road_length (network : RoadNetwork) 
  (h1 : network.road1 = 10)
  (h2 : network.road2 = 5)
  (h3 : network.road3 = 8)
  (h4 : network.road4 = 21) :
  network.x = 17 := by
  sorry


end NUMINAMATH_CALUDE_unknown_road_length_l2051_205118


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2051_205119

-- Define the sets M and N
def M : Set ℝ := {x | 2 * x - 3 < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2051_205119


namespace NUMINAMATH_CALUDE_power_five_hundred_mod_eighteen_l2051_205129

theorem power_five_hundred_mod_eighteen : 
  (5 : ℤ) ^ 100 % 18 = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_five_hundred_mod_eighteen_l2051_205129


namespace NUMINAMATH_CALUDE_value_of_x_l2051_205171

theorem value_of_x (w y z x : ℕ) 
  (hw : w = 95)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 7) :
  x = 142 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2051_205171


namespace NUMINAMATH_CALUDE_speed_ratio_l2051_205120

/-- The ratio of average speeds of two travelers -/
theorem speed_ratio (distance_AB distance_AC : ℝ) (time_Eddy time_Freddy : ℝ) 
  (h1 : distance_AB = 540)
  (h2 : distance_AC = 300)
  (h3 : time_Eddy = 3)
  (h4 : time_Freddy = 4)
  (h5 : distance_AB > 0)
  (h6 : distance_AC > 0)
  (h7 : time_Eddy > 0)
  (h8 : time_Freddy > 0) :
  (distance_AB / time_Eddy) / (distance_AC / time_Freddy) = 12 / 5 := by
sorry


end NUMINAMATH_CALUDE_speed_ratio_l2051_205120


namespace NUMINAMATH_CALUDE_roots_sum_expression_l2051_205187

def quadratic_equation (x : ℝ) : Prop := 5 * x^2 - 3 * x - 4 = 0

theorem roots_sum_expression (x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation x₁) 
  (h₂ : quadratic_equation x₂) 
  (h₃ : x₁ ≠ x₂) : 
  2 * x₁^2 + 3 * x₂^2 = 178 / 25 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_expression_l2051_205187


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_b9_l2051_205194

def fibonacci_like_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) = b (n + 1) + b n

theorem fibonacci_like_sequence_b9 (b : ℕ → ℕ) :
  fibonacci_like_sequence b →
  (∀ n m : ℕ, n < m → b n < b m) →
  b 8 = 100 →
  b 9 = 194 := by sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_b9_l2051_205194


namespace NUMINAMATH_CALUDE_triangle_inequality_third_side_range_l2051_205137

theorem triangle_inequality (a b c : ℝ) : 
  (0 < a ∧ 0 < b ∧ 0 < c) → (a < b + c ∧ b < a + c ∧ c < a + b) := by sorry

theorem third_side_range : 
  ∀ a : ℝ, (∃ (s1 s2 : ℝ), s1 = 3 ∧ s2 = 5 ∧ 0 < a ∧ 
    (a < s1 + s2 ∧ s1 < a + s2 ∧ s2 < a + s1)) → 
  (2 < a ∧ a < 8) := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_third_side_range_l2051_205137


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l2051_205191

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x-1)

theorem function_satisfies_equation :
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 →
    f ((x - 1) / x) + f (1 / (1 - x)) = 2 - 2 * x :=
by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l2051_205191


namespace NUMINAMATH_CALUDE_platinum_to_gold_ratio_is_two_to_one_l2051_205154

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard

theorem platinum_to_gold_ratio_is_two_to_one 
  (cards : SallysCards)
  (h1 : cards.gold.balance = cards.gold.limit / 3)
  (h2 : cards.platinum.balance = cards.platinum.limit / 6)
  (h3 : cards.platinum.balance + cards.gold.balance = cards.platinum.limit / 3) :
  cards.platinum.limit / cards.gold.limit = 2 := by
  sorry

end NUMINAMATH_CALUDE_platinum_to_gold_ratio_is_two_to_one_l2051_205154


namespace NUMINAMATH_CALUDE_pizza_cost_is_twelve_l2051_205155

/-- Calculates the cost of each pizza given the number of people, people per pizza, 
    earnings per night, and number of nights worked. -/
def pizza_cost (total_people : ℕ) (people_per_pizza : ℕ) (earnings_per_night : ℕ) (nights_worked : ℕ) : ℚ :=
  let total_pizzas := (total_people + people_per_pizza - 1) / people_per_pizza
  let total_earnings := earnings_per_night * nights_worked
  total_earnings / total_pizzas

/-- Proves that the cost of each pizza is $12 under the given conditions. -/
theorem pizza_cost_is_twelve :
  pizza_cost 15 3 4 15 = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cost_is_twelve_l2051_205155


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_hyperbola_standard_equation_l2051_205136

-- Ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

theorem ellipse_standard_equation
  (major_axis : ℝ)
  (focal_distance : ℝ)
  (h_major_axis : major_axis = 4)
  (h_focal_distance : focal_distance = 2)
  (h_foci_on_x_axis : True) :
  ∀ x y : ℝ, ellipse_equation x y ↔ 
    x^2 / (major_axis^2 / 4) + y^2 / ((major_axis^2 / 4) - focal_distance^2) = 1 :=
sorry

-- Hyperbola
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

theorem hyperbola_standard_equation
  (k : ℝ)
  (d : ℝ)
  (h_asymptote : k = 3/4)
  (h_directrix : d = 16/5) :
  ∀ x y : ℝ, hyperbola_equation x y ↔ 
    x^2 / (d^2 / (1 + k^2)) - y^2 / ((d^2 * k^2) / (1 + k^2)) = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_hyperbola_standard_equation_l2051_205136


namespace NUMINAMATH_CALUDE_least_value_theorem_l2051_205112

theorem least_value_theorem (p q : ℕ) (x : ℚ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : 17 * (p + 1) = x * (q + 1))
  (h4 : ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = x * (q' + 1) → p' + q' ≥ 40)
  (h5 : p + q = 40) : 
  x = 14 := by
sorry

end NUMINAMATH_CALUDE_least_value_theorem_l2051_205112


namespace NUMINAMATH_CALUDE_steve_book_earnings_l2051_205128

/-- Calculates an author's net earnings from book sales -/
def authorNetEarnings (copies : ℕ) (earningsPerCopy : ℚ) (agentPercentage : ℚ) : ℚ :=
  let totalEarnings := copies * earningsPerCopy
  let agentCommission := totalEarnings * agentPercentage
  totalEarnings - agentCommission

/-- Proves that given the specified conditions, the author's net earnings are $1,800,000 -/
theorem steve_book_earnings :
  authorNetEarnings 1000000 2 (1/10) = 1800000 := by
  sorry

#eval authorNetEarnings 1000000 2 (1/10)

end NUMINAMATH_CALUDE_steve_book_earnings_l2051_205128


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2051_205158

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) :
  (a + 2 + 4 / (a - 2)) / (a^3 / (a^2 - 4*a + 4)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2051_205158


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l2051_205145

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_x_1 :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m*x + b ↔ x - y - 1 = 0) ∧
    m = f' 1 ∧
    f 1 = m*1 + b :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l2051_205145


namespace NUMINAMATH_CALUDE_function_properties_l2051_205138

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define g' as the derivative of g
variable (g' : ℝ → ℝ)
variable (h : ∀ x, HasDerivAt g (g' x) x)

-- Define the conditions
variable (cond1 : ∀ x, f x + g' x - 10 = 0)
variable (cond2 : ∀ x, f x - g' (4 - x) - 10 = 0)
variable (cond3 : ∀ x, g x = g (-x))  -- g is an even function

-- Theorem statement
theorem function_properties :
  (f 1 + f 3 = 20) ∧ (f 4 = 10) ∧ (f 2022 = 10) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2051_205138


namespace NUMINAMATH_CALUDE_f_2017_negative_two_equals_three_fifths_l2051_205100

def f (x : ℚ) : ℚ := (x - 1) / (3 * x + 1)

def iterate_f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem f_2017_negative_two_equals_three_fifths :
  iterate_f 2017 (-2 : ℚ) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_f_2017_negative_two_equals_three_fifths_l2051_205100


namespace NUMINAMATH_CALUDE_carrot_weight_theorem_l2051_205167

/-- Given 30 carrots, where 27 of them have an average weight of 200 grams
    and 3 of them have an average weight of 180 grams,
    the total weight of all 30 carrots is 5.94 kg. -/
theorem carrot_weight_theorem :
  let total_carrots : ℕ := 30
  let remaining_carrots : ℕ := 27
  let removed_carrots : ℕ := 3
  let avg_weight_remaining : ℝ := 200 -- in grams
  let avg_weight_removed : ℝ := 180 -- in grams
  let total_weight_grams : ℝ := remaining_carrots * avg_weight_remaining + removed_carrots * avg_weight_removed
  let total_weight_kg : ℝ := total_weight_grams / 1000
  total_weight_kg = 5.94 := by
  sorry

end NUMINAMATH_CALUDE_carrot_weight_theorem_l2051_205167


namespace NUMINAMATH_CALUDE_box_ratio_proof_l2051_205149

def box_problem (total_balls white_balls : ℕ) (blue_white_diff : ℕ) : Prop :=
  let blue_balls : ℕ := white_balls + blue_white_diff
  let red_balls : ℕ := total_balls - (white_balls + blue_balls)
  (red_balls : ℚ) / blue_balls = 2 / 1

theorem box_ratio_proof :
  box_problem 100 16 12 := by
  sorry

end NUMINAMATH_CALUDE_box_ratio_proof_l2051_205149


namespace NUMINAMATH_CALUDE_three_digit_prime_integers_count_l2051_205159

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := sorry

-- Define a function to get all single-digit prime numbers
def singleDigitPrimes : List ℕ := sorry

-- Define a function to count the number of three-digit positive integers
-- where the digits are three different prime numbers
def countThreeDigitPrimeIntegers : ℕ := sorry

-- Theorem statement
theorem three_digit_prime_integers_count :
  countThreeDigitPrimeIntegers = 24 := by sorry

end NUMINAMATH_CALUDE_three_digit_prime_integers_count_l2051_205159


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2051_205142

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 7*x + 6
  (f 1 = 0) ∧ (f 6 = 0) ∧
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2051_205142


namespace NUMINAMATH_CALUDE_negative_x_fourth_cubed_l2051_205152

theorem negative_x_fourth_cubed (x : ℝ) : (-x^4)^3 = -x^12 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_fourth_cubed_l2051_205152


namespace NUMINAMATH_CALUDE_raisin_mixture_problem_l2051_205182

/-- The number of scoops of natural seedless raisins needed to create a mixture with
    a specific cost per scoop, given the costs and quantities of two types of raisins. -/
theorem raisin_mixture_problem (cost_natural : ℚ) (cost_golden : ℚ) (scoops_golden : ℕ) (cost_mixture : ℚ) :
  cost_natural = 345/100 →
  cost_golden = 255/100 →
  scoops_golden = 20 →
  cost_mixture = 3 →
  ∃ scoops_natural : ℕ,
    scoops_natural = 20 ∧
    (cost_natural * scoops_natural + cost_golden * scoops_golden) / (scoops_natural + scoops_golden) = cost_mixture :=
by sorry

end NUMINAMATH_CALUDE_raisin_mixture_problem_l2051_205182


namespace NUMINAMATH_CALUDE_remainder_mod_nine_l2051_205179

theorem remainder_mod_nine (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_nine_l2051_205179


namespace NUMINAMATH_CALUDE_blue_spotted_fish_count_l2051_205122

theorem blue_spotted_fish_count (total_fish : ℕ) (blue_percentage : ℚ) (spotted_fraction : ℚ) : 
  total_fish = 150 →
  blue_percentage = 2/5 →
  spotted_fraction = 3/5 →
  (total_fish : ℚ) * blue_percentage * spotted_fraction = 36 := by
sorry

end NUMINAMATH_CALUDE_blue_spotted_fish_count_l2051_205122


namespace NUMINAMATH_CALUDE_xy_squared_equals_one_l2051_205148

theorem xy_squared_equals_one (x y : ℝ) (h : |x - 2| + (3 + y)^2 = 0) : (x + y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_equals_one_l2051_205148


namespace NUMINAMATH_CALUDE_problem_proof_l2051_205108

theorem problem_proof : 2^0 - |(-3)| + (-1/2) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2051_205108


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_train_platform_crossing_time_specific_l2051_205178

/-- Calculates the time taken for a train to cross a platform -/
theorem train_platform_crossing_time 
  (train_length platform_length : ℝ) 
  (time_to_cross_pole : ℝ) : ℝ :=
  let train_speed := train_length / time_to_cross_pole
  let total_distance := train_length + platform_length
  total_distance / train_speed

/-- Proves that the time taken for a 300m train to cross a 250m platform 
    is approximately 33 seconds, given that it takes 18 seconds to cross a signal pole -/
theorem train_platform_crossing_time_specific : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_platform_crossing_time 300 250 18 - 33| < ε :=
sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_train_platform_crossing_time_specific_l2051_205178


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2051_205126

theorem sum_of_three_numbers (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 989)
  (h_sum_pairs_squared : (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013) :
  a + b + c = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2051_205126


namespace NUMINAMATH_CALUDE_product_sum_and_reciprocals_bound_l2051_205161

theorem product_sum_and_reciprocals_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 ∧
  ∀ ε > 0, ∃ a' b' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    (a' + b' + c') * (1/a' + 1/b' + 1/c') < 9 + ε :=
by sorry

end NUMINAMATH_CALUDE_product_sum_and_reciprocals_bound_l2051_205161


namespace NUMINAMATH_CALUDE_complement_of_intersection_l2051_205132

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {1,3,5}
def B : Set Nat := {1,2}

theorem complement_of_intersection (U A B : Set Nat) :
  U = {1,2,3,4,5,6} →
  A = {1,3,5} →
  B = {1,2} →
  (U \ (A ∩ B)) = {2,3,4,5,6} :=
by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l2051_205132


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2051_205196

-- Define the inverse variation relationship
def inverse_variation (y z : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ y^4 * z^(1/4) = k

-- State the theorem
theorem inverse_variation_problem (y z : ℝ) :
  inverse_variation y z →
  (3 : ℝ)^4 * 16^(1/4) = 6^4 * z^(1/4) →
  z = 1 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2051_205196


namespace NUMINAMATH_CALUDE_power_calculation_l2051_205162

theorem power_calculation : 27^3 * 9^2 / 3^15 = (1 : ℚ) / 9 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l2051_205162
