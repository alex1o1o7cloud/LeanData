import Mathlib

namespace NUMINAMATH_CALUDE_bakery_combinations_l1171_117181

/-- The number of ways to distribute n items among k categories, 
    with at least m items in each of the first two categories -/
def distribute (n k m : ℕ) : ℕ :=
  -- We don't provide the implementation, just the type signature
  sorry

/-- The specific case for the bakery problem -/
theorem bakery_combinations : distribute 8 5 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_bakery_combinations_l1171_117181


namespace NUMINAMATH_CALUDE_base5_divisibility_by_29_l1171_117109

def base5ToDecimal (a b c d : ℕ) : ℕ := a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

def isDivisibleBy29 (n : ℕ) : Prop := ∃ k : ℕ, n = 29 * k

theorem base5_divisibility_by_29 (y : ℕ) :
  isDivisibleBy29 (base5ToDecimal 4 2 y 3) ↔ y = 4 := by sorry

end NUMINAMATH_CALUDE_base5_divisibility_by_29_l1171_117109


namespace NUMINAMATH_CALUDE_star_equation_solution_l1171_117158

-- Define the custom operation ※
def star (a b : ℝ) : ℝ := a^2 - 3*a + b

-- State the theorem
theorem star_equation_solution :
  ∃ x₁ x₂ : ℝ, (x₁ = -1 ∨ x₁ = 4) ∧ (x₂ = -1 ∨ x₂ = 4) ∧
  (∀ x : ℝ, star x 2 = 6 ↔ (x = x₁ ∨ x = x₂)) :=
sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1171_117158


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_3465_is_9_l1171_117148

/-- The largest perfect square factor of 3465 -/
def largest_perfect_square_factor_of_3465 : ℕ := 9

/-- Theorem stating that the largest perfect square factor of 3465 is 9 -/
theorem largest_perfect_square_factor_of_3465_is_9 :
  ∀ n : ℕ, n^2 ∣ 3465 → n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_3465_is_9_l1171_117148


namespace NUMINAMATH_CALUDE_parallelogram_angle_difference_parallelogram_angle_difference_proof_l1171_117167

/-- 
In a parallelogram with a smaller angle of 55 degrees, 
the difference between the larger and smaller angles is 70 degrees.
-/
theorem parallelogram_angle_difference : ℝ → Prop :=
  fun smaller_angle : ℝ =>
    smaller_angle = 55 →
    ∃ larger_angle : ℝ,
      smaller_angle + larger_angle = 180 ∧
      larger_angle - smaller_angle = 70

-- The proof is omitted
theorem parallelogram_angle_difference_proof : 
  parallelogram_angle_difference 55 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_difference_parallelogram_angle_difference_proof_l1171_117167


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1171_117196

/-- Given a function f and a triangle ABC, prove the lengths of sides b and c. -/
theorem triangle_side_lengths 
  (f : ℝ → ℝ) 
  (vec_a vec_b : ℝ → ℝ × ℝ)
  (A B C : ℝ) 
  (a b c : ℝ) :
  (∀ x, f x = (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2) →
  (∀ x, vec_a x = (2 * Real.cos x, -Real.sqrt 3 * Real.sin (2 * x))) →
  (∀ x, vec_b x = (Real.cos x, 1)) →
  f A = -1 →
  a = Real.sqrt 7 / 2 →
  ∃ (k : ℝ), 3 * Real.sin C = 2 * Real.sin B →
  b = 3/2 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l1171_117196


namespace NUMINAMATH_CALUDE_max_product_constrained_l1171_117114

theorem max_product_constrained (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_constraint : 3*x + 2*y = 12) :
  x * y ≤ 6 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3*x₀ + 2*y₀ = 12 ∧ x₀ * y₀ = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_l1171_117114


namespace NUMINAMATH_CALUDE_max_spheres_in_frustum_l1171_117103

/-- Represents a frustum with given height and spheres inside it -/
structure Frustum :=
  (height : ℝ)
  (O₁_radius : ℝ)
  (O₂_radius : ℝ)

/-- Calculates the maximum number of additional spheres that can fit in the frustum -/
def max_additional_spheres (f : Frustum) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- The main theorem stating the maximum number of additional spheres -/
theorem max_spheres_in_frustum (f : Frustum) 
  (h₁ : f.height = 8)
  (h₂ : f.O₁_radius = 2)
  (h₃ : f.O₂_radius = 3) :
  max_additional_spheres f = 2 :=
sorry

end NUMINAMATH_CALUDE_max_spheres_in_frustum_l1171_117103


namespace NUMINAMATH_CALUDE_K2Cr2O7_molecular_weight_l1171_117153

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (K_atoms Cr_atoms O_atoms : ℕ) (K_weight Cr_weight O_weight : ℝ) : ℝ :=
  K_atoms * K_weight + Cr_atoms * Cr_weight + O_atoms * O_weight

/-- The molecular weight of the compound K₂Cr₂O₇ is 294.192 g/mol -/
theorem K2Cr2O7_molecular_weight : 
  molecular_weight 2 2 7 39.10 51.996 16.00 = 294.192 := by
  sorry

#eval molecular_weight 2 2 7 39.10 51.996 16.00

end NUMINAMATH_CALUDE_K2Cr2O7_molecular_weight_l1171_117153


namespace NUMINAMATH_CALUDE_texts_sent_per_month_l1171_117171

/-- Represents the number of texts sent per month -/
def T : ℕ := sorry

/-- Represents the cost of the current plan in dollars -/
def current_plan_cost : ℕ := 12

/-- Represents the cost per 30 texts in dollars -/
def text_cost_per_30 : ℕ := 1

/-- Represents the cost per 20 minutes of calls in dollars -/
def call_cost_per_20_min : ℕ := 3

/-- Represents the number of minutes spent on calls per month -/
def call_minutes : ℕ := 60

/-- Represents the cost difference between current and alternative plans in dollars -/
def cost_difference : ℕ := 1

theorem texts_sent_per_month :
  T = 60 ∧
  (T / 30 : ℚ) * text_cost_per_30 + 
  (call_minutes / 20 : ℚ) * call_cost_per_20_min = 
  current_plan_cost - cost_difference :=
by sorry

end NUMINAMATH_CALUDE_texts_sent_per_month_l1171_117171


namespace NUMINAMATH_CALUDE_distance_swam_against_current_l1171_117173

/-- Proves that the distance swam against the current is 10 km -/
theorem distance_swam_against_current
  (still_water_speed : ℝ)
  (water_speed : ℝ)
  (time_taken : ℝ)
  (h1 : still_water_speed = 12)
  (h2 : water_speed = 2)
  (h3 : time_taken = 1) :
  still_water_speed - water_speed * time_taken = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_swam_against_current_l1171_117173


namespace NUMINAMATH_CALUDE_complex_power_sum_l1171_117180

/-- If z is a complex number satisfying z + 1/z = 2 cos 5°, then z^1500 + 1/z^1500 = 1 -/
theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1500 + 1/z^1500 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1171_117180


namespace NUMINAMATH_CALUDE_inequality_for_elements_in_M_l1171_117122

-- Define the set M
def M : Set ℝ := {x : ℝ | -1/2 < x ∧ x < 1}

-- State the theorem
theorem inequality_for_elements_in_M (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  |a - b| < |1 - a * b| := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_elements_in_M_l1171_117122


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_one_second_l1171_117170

-- Define the height function
def h (t : ℝ) : ℝ := -4.9 * t^2 + 4.8 * t + 11

-- Define the velocity function as the derivative of the height function
def v (t : ℝ) : ℝ := -9.8 * t + 4.8

-- Theorem statement
theorem instantaneous_velocity_at_one_second :
  v 1 = -5 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_one_second_l1171_117170


namespace NUMINAMATH_CALUDE_line_segment_point_sum_l1171_117115

/-- Given a line y = -2/3x + 6 that crosses the x-axis at P and y-axis at Q,
    and a point T(r, s) on line segment PQ, prove that if the area of triangle POQ
    is four times the area of triangle TOP, then r + s = 8.25. -/
theorem line_segment_point_sum (r s : ℝ) : 
  let line := fun (x : ℝ) ↦ -2/3 * x + 6
  let P := (9, 0)
  let Q := (0, 6)
  let T := (r, s)
  (T.1 ≥ 0 ∧ T.1 ≤ 9) →  -- T is on line segment PQ
  (T.2 = line T.1) →  -- T is on the line
  (1/2 * 9 * 6 = 4 * (1/2 * 9 * s)) →  -- Area condition
  r + s = 8.25 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_point_sum_l1171_117115


namespace NUMINAMATH_CALUDE_loop_structure_and_body_l1171_117159

/-- Represents an algorithmic structure -/
structure AlgorithmicStructure where
  repeatedExecution : Bool
  conditionalExecution : Bool

/-- Represents a processing step in an algorithm -/
structure ProcessingStep where
  isRepeated : Bool

/-- Definition of a loop structure -/
def isLoopStructure (s : AlgorithmicStructure) : Prop :=
  s.repeatedExecution ∧ s.conditionalExecution

/-- Definition of a loop body -/
def isLoopBody (p : ProcessingStep) : Prop :=
  p.isRepeated

/-- Theorem stating the relationship between loop structures and loop bodies -/
theorem loop_structure_and_body 
    (s : AlgorithmicStructure) 
    (p : ProcessingStep) 
    (h1 : s.repeatedExecution) 
    (h2 : s.conditionalExecution) 
    (h3 : p.isRepeated) : 
  isLoopStructure s ∧ isLoopBody p := by
  sorry


end NUMINAMATH_CALUDE_loop_structure_and_body_l1171_117159


namespace NUMINAMATH_CALUDE_other_student_correct_answers_l1171_117124

/-- 
Given:
- Martin answered 40 questions correctly
- Martin answered three fewer questions correctly than Kelsey
- Kelsey answered eight more questions correctly than another student

Prove: The other student answered 35 questions correctly
-/
theorem other_student_correct_answers 
  (martin_correct : ℕ) 
  (kelsey_martin_diff : ℕ) 
  (kelsey_other_diff : ℕ) 
  (h1 : martin_correct = 40)
  (h2 : kelsey_martin_diff = 3)
  (h3 : kelsey_other_diff = 8) :
  martin_correct + kelsey_martin_diff - kelsey_other_diff = 35 := by
sorry

end NUMINAMATH_CALUDE_other_student_correct_answers_l1171_117124


namespace NUMINAMATH_CALUDE_square_ratio_proof_l1171_117110

theorem square_ratio_proof (area_ratio : ℚ) :
  area_ratio = 300 / 75 →
  ∃ (a b c : ℕ), 
    (a * Real.sqrt b : ℝ) / c = Real.sqrt area_ratio ∧
    a = 2 ∧ b = 1 ∧ c = 1 ∧
    a + b + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_proof_l1171_117110


namespace NUMINAMATH_CALUDE_can_find_genuine_coin_l1171_117174

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a group of coins -/
structure CoinGroup where
  size : Nat
  counterfeit : Nat

/-- Represents the state of coins -/
structure CoinState where
  total : Nat
  counterfeit : Nat

/-- Represents a weighing operation -/
def weigh (left right : CoinGroup) : WeighResult :=
  sorry

/-- Represents the process of finding a genuine coin -/
def findGenuineCoin (state : CoinState) : Prop :=
  ∃ (g1 g2 g3 : CoinGroup),
    g1.size + g2.size + g3.size = state.total ∧
    g1.counterfeit + g2.counterfeit + g3.counterfeit = state.counterfeit ∧
    (∃ (result : WeighResult),
      result = weigh g1 g2 ∧
      (result = WeighResult.Equal →
        ∃ (c1 c2 : CoinGroup),
          c1.size = 1 ∧ c2.size = 1 ∧
          c1.size + c2.size ≤ g3.size ∧
          (weigh c1 c2 = WeighResult.Equal ∨
           weigh c1 c2 = WeighResult.LeftHeavier ∨
           weigh c1 c2 = WeighResult.RightHeavier)) ∧
      ((result = WeighResult.LeftHeavier ∨ result = WeighResult.RightHeavier) →
        ∃ (c1 c2 : CoinGroup),
          c1.size = 1 ∧ c2.size = 1 ∧
          (c1.size ≤ g1.size ∧ c2.size ≤ g2.size) ∧
          (weigh c1 c2 = WeighResult.Equal ∨
           weigh c1 c2 = WeighResult.LeftHeavier ∨
           weigh c1 c2 = WeighResult.RightHeavier)))

theorem can_find_genuine_coin (state : CoinState)
  (h1 : state.total = 100)
  (h2 : state.counterfeit = 4)
  (h3 : state.counterfeit < state.total) :
  findGenuineCoin state :=
  sorry

end NUMINAMATH_CALUDE_can_find_genuine_coin_l1171_117174


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_inequality_l1171_117183

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |4*x - a| + a^2 - 4*a

-- Define the function g
def g (x : ℝ) : ℝ := |x - 1|

-- Theorem for part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | -2 ≤ f 1 x ∧ f 1 x ≤ 4} = 
  {x : ℝ | -3/2 ≤ x ∧ x ≤ 0} ∪ {x : ℝ | 1/2 ≤ x ∧ x ≤ 2} :=
by sorry

-- Theorem for part 2
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x : ℝ, f a x - 4 * g x ≤ 6} = 
  {a : ℝ | (5 - Real.sqrt 33) / 2 ≤ a ∧ a ≤ 5} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_inequality_l1171_117183


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1171_117176

theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 120 ∧ crossing_time = 8 → 
  ∃ (speed : ℝ), speed = 54 ∧ 
  (2 * train_length) / crossing_time * 3.6 = 2 * speed := by
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1171_117176


namespace NUMINAMATH_CALUDE_work_completion_equality_first_group_size_l1171_117190

/-- The number of days it takes the first group to complete the work -/
def days_first_group : ℕ := 20

/-- The number of men in the second group -/
def men_second_group : ℕ := 12

/-- The number of days it takes the second group to complete the work -/
def days_second_group : ℕ := 30

/-- The number of men in the first group -/
def men_first_group : ℕ := 18

theorem work_completion_equality :
  men_first_group * days_first_group = men_second_group * days_second_group :=
by sorry

theorem first_group_size :
  men_first_group = (men_second_group * days_second_group) / days_first_group :=
by sorry

end NUMINAMATH_CALUDE_work_completion_equality_first_group_size_l1171_117190


namespace NUMINAMATH_CALUDE_spider_web_paths_spider_web_problem_l1171_117129

theorem spider_web_paths : Nat → Nat → Nat
  | m, n => Nat.choose (m + n) m

theorem spider_web_problem : spider_web_paths 5 6 = 462 := by
  sorry

end NUMINAMATH_CALUDE_spider_web_paths_spider_web_problem_l1171_117129


namespace NUMINAMATH_CALUDE_ponderosa_pine_price_l1171_117137

/-- The price of each ponderosa pine tree, given the total number of trees,
    number of Douglas fir trees, price of each Douglas fir, and total amount paid. -/
theorem ponderosa_pine_price
  (total_trees : ℕ)
  (douglas_fir_trees : ℕ)
  (douglas_fir_price : ℕ)
  (total_amount : ℕ)
  (h1 : total_trees = 850)
  (h2 : douglas_fir_trees = 350)
  (h3 : douglas_fir_price = 300)
  (h4 : total_amount = 217500) :
  (total_amount - douglas_fir_trees * douglas_fir_price) / (total_trees - douglas_fir_trees) = 225 := by
sorry


end NUMINAMATH_CALUDE_ponderosa_pine_price_l1171_117137


namespace NUMINAMATH_CALUDE_journey_time_proof_l1171_117197

theorem journey_time_proof (highway_distance : ℝ) (mountain_distance : ℝ) 
  (speed_ratio : ℝ) (mountain_time : ℝ) :
  highway_distance = 60 →
  mountain_distance = 20 →
  speed_ratio = 4 →
  mountain_time = 40 →
  highway_distance / (speed_ratio * (mountain_distance / mountain_time)) + mountain_time = 70 :=
by sorry

end NUMINAMATH_CALUDE_journey_time_proof_l1171_117197


namespace NUMINAMATH_CALUDE_id_number_permutations_l1171_117118

/-- The number of permutations of n distinct elements -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The problem statement -/
theorem id_number_permutations :
  permutations 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_id_number_permutations_l1171_117118


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l1171_117182

theorem degree_to_radian_conversion (x : Real) : 
  x * (π / 180) = -5 * π / 3 → x = -300 :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l1171_117182


namespace NUMINAMATH_CALUDE_students_in_both_groups_l1171_117101

theorem students_in_both_groups 
  (total : ℕ) 
  (math : ℕ) 
  (english : ℕ) 
  (h1 : total = 52) 
  (h2 : math = 32) 
  (h3 : english = 40) : 
  total = math + english - 20 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_groups_l1171_117101


namespace NUMINAMATH_CALUDE_joyce_bananas_l1171_117135

/-- Given a number of boxes and bananas per box, calculates the total number of bananas -/
def total_bananas (num_boxes : ℕ) (bananas_per_box : ℕ) : ℕ :=
  num_boxes * bananas_per_box

/-- Proves that 10 boxes with 4 bananas each results in 40 bananas total -/
theorem joyce_bananas : total_bananas 10 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_joyce_bananas_l1171_117135


namespace NUMINAMATH_CALUDE_train_length_l1171_117193

/-- The length of a train given its speed and time to cross a platform -/
theorem train_length (v : ℝ) (t : ℝ) (platform_length : ℝ) : 
  v = 72 * (5/18) → 
  t = 36 → 
  platform_length = 250 → 
  v * t - platform_length = 470 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1171_117193


namespace NUMINAMATH_CALUDE_triangle_altitude_on_rectangle_diagonal_l1171_117191

/-- Given a rectangle with side lengths a and b, and a triangle constructed on its diagonal
    as base with area equal to the rectangle's area, the altitude of the triangle is
    (2 * a * b) / sqrt(a^2 + b^2). -/
theorem triangle_altitude_on_rectangle_diagonal 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ (h : ℝ), h = (2 * a * b) / Real.sqrt (a^2 + b^2) ∧ 
  h * Real.sqrt (a^2 + b^2) / 2 = a * b := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_on_rectangle_diagonal_l1171_117191


namespace NUMINAMATH_CALUDE_xyz_maximum_l1171_117138

theorem xyz_maximum (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_eq : x * y - z = (x - z) * (y - z)) (h_sum : x + y + z = 1) :
  x * y * z ≤ 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_xyz_maximum_l1171_117138


namespace NUMINAMATH_CALUDE_petyas_class_girls_count_l1171_117121

theorem petyas_class_girls_count :
  ∀ (x y : ℕ),
  x + y ≤ 40 →
  (2 : ℚ) / 3 * x + (1 : ℚ) / 7 * y = (1 : ℚ) / 3 * (x + y) →
  x = 12 :=
λ x y h1 h2 => by
  sorry

end NUMINAMATH_CALUDE_petyas_class_girls_count_l1171_117121


namespace NUMINAMATH_CALUDE_correct_num_pants_purchased_l1171_117131

/-- Represents the purchase and refund scenario at a clothing retailer -/
structure ClothingPurchase where
  shirtPrice : ℝ
  pantsPrice : ℝ
  totalCost : ℝ
  refundRate : ℝ
  numShirts : ℕ

/-- The number of pairs of pants purchased given the conditions -/
def numPantsPurchased (purchase : ClothingPurchase) : ℕ :=
  1

theorem correct_num_pants_purchased (purchase : ClothingPurchase) 
  (h1 : purchase.shirtPrice ≠ purchase.pantsPrice)
  (h2 : purchase.shirtPrice = 45)
  (h3 : purchase.numShirts = 2)
  (h4 : purchase.totalCost = 120)
  (h5 : purchase.refundRate = 0.25)
  : numPantsPurchased purchase = 1 := by
  sorry

#check correct_num_pants_purchased

end NUMINAMATH_CALUDE_correct_num_pants_purchased_l1171_117131


namespace NUMINAMATH_CALUDE_periodic_sequence_quadratic_root_l1171_117100

def is_periodic (x : ℕ → ℝ) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, x (n + p) = x n

def sequence_property (x : ℕ → ℝ) : Prop :=
  x 0 > 1 ∧ ∀ n : ℕ, x (n + 1) = 1 / (x n - ⌊x n⌋)

def is_quadratic_root (r : ℝ) : Prop :=
  ∃ a b c : ℤ, a ≠ 0 ∧ a * r^2 + b * r + c = 0

theorem periodic_sequence_quadratic_root (x : ℕ → ℝ) :
  is_periodic x → sequence_property x → is_quadratic_root (x 0) := by
  sorry

end NUMINAMATH_CALUDE_periodic_sequence_quadratic_root_l1171_117100


namespace NUMINAMATH_CALUDE_characterization_of_k_set_l1171_117104

-- Define h as 2^r where r is a non-negative integer
def h (r : ℕ) : ℕ := 2^r

-- Define the set of k that satisfy the conditions
def k_set (h : ℕ) : Set ℕ := {k : ℕ | ∃ (m n : ℕ), m > n ∧ k ∣ (m^h - 1) ∧ n^((m^h - 1) / k) ≡ -1 [ZMOD m]}

-- The theorem to prove
theorem characterization_of_k_set (r : ℕ) : 
  k_set (h r) = {k : ℕ | ∃ (s t : ℕ), k = 2^(r+s) * t ∧ Odd t} :=
sorry

end NUMINAMATH_CALUDE_characterization_of_k_set_l1171_117104


namespace NUMINAMATH_CALUDE_symmetric_line_x_axis_l1171_117107

/-- The equation of a line symmetric to another line with respect to the x-axis -/
theorem symmetric_line_x_axis (a b c : ℝ) :
  (∀ x y, a * x + b * y + c = 0 ↔ a * x - b * y - c = 0) →
  (∀ x y, 3 * x + 4 * y - 5 = 0 ↔ 3 * x - 4 * y + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_x_axis_l1171_117107


namespace NUMINAMATH_CALUDE_area_ratio_equals_side_ratio_l1171_117155

/-- Triangle PQR with angle bisector PS -/
structure AngleBisectorTriangle where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- PS is an angle bisector -/
  PS_is_angle_bisector : Bool

/-- The ratio of areas of triangles formed by an angle bisector -/
def area_ratio (t : AngleBisectorTriangle) : ℝ :=
  sorry

/-- Theorem: The ratio of areas of triangles formed by an angle bisector
    is equal to the ratio of the lengths of the sides adjacent to the bisected angle -/
theorem area_ratio_equals_side_ratio (t : AngleBisectorTriangle) 
  (h : t.PS_is_angle_bisector = true) (h1 : t.PQ = 45) (h2 : t.PR = 75) (h3 : t.QR = 64) : 
  area_ratio t = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_equals_side_ratio_l1171_117155


namespace NUMINAMATH_CALUDE_shaded_area_is_one_third_l1171_117178

/-- Represents a 3x3 square quilt block -/
structure QuiltBlock :=
  (size : Nat)
  (shaded_area : ℚ)

/-- The size of the quilt block is 3 -/
def quilt_size : Nat := 3

/-- The quilt block with the given shaded pattern -/
def patterned_quilt : QuiltBlock :=
  { size := quilt_size,
    shaded_area := 1 }

/-- Theorem stating that the shaded area of the patterned quilt is 1/3 of the total area -/
theorem shaded_area_is_one_third (q : QuiltBlock) (h : q = patterned_quilt) :
  q.shaded_area / (q.size * q.size : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_one_third_l1171_117178


namespace NUMINAMATH_CALUDE_ab_range_l1171_117156

theorem ab_range (a b : ℝ) (h : a * b = a + b + 3) :
  (a * b ≤ 1) ∨ (a * b ≥ 9) := by sorry

end NUMINAMATH_CALUDE_ab_range_l1171_117156


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l1171_117141

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l1171_117141


namespace NUMINAMATH_CALUDE_max_m_value_l1171_117168

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x < m → x^2 - 2*x - 8 > 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m) ∧
  (∀ m' : ℝ, m' > m → 
    ¬((∀ x : ℝ, x < m' → x^2 - 2*x - 8 > 0) ∧ 
      (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m'))) →
  m = 4 := by sorry

end NUMINAMATH_CALUDE_max_m_value_l1171_117168


namespace NUMINAMATH_CALUDE_math_test_results_l1171_117172

/-- Represents the score distribution for a math test -/
structure ScoreDistribution where
  prob_45 : ℚ
  prob_50 : ℚ
  prob_55 : ℚ
  prob_60 : ℚ

/-- Represents the conditions of the math test -/
structure MathTest where
  total_questions : ℕ
  options_per_question : ℕ
  points_per_correct : ℕ
  certain_correct : ℕ
  uncertain_two_eliminated : ℕ
  uncertain_one_eliminated : ℕ

/-- Calculates the probability of scoring 55 points given the test conditions -/
def prob_55 (test : MathTest) : ℚ :=
  sorry

/-- Calculates the score distribution given the test conditions -/
def score_distribution (test : MathTest) : ScoreDistribution :=
  sorry

/-- Calculates the expected value of the score given the score distribution -/
def expected_value (dist : ScoreDistribution) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem math_test_results (test : MathTest) 
  (h1 : test.total_questions = 12)
  (h2 : test.options_per_question = 4)
  (h3 : test.points_per_correct = 5)
  (h4 : test.certain_correct = 9)
  (h5 : test.uncertain_two_eliminated = 2)
  (h6 : test.uncertain_one_eliminated = 1) :
  prob_55 test = 1/3 ∧ 
  expected_value (score_distribution test) = 165/3 :=
sorry

end NUMINAMATH_CALUDE_math_test_results_l1171_117172


namespace NUMINAMATH_CALUDE_smallest_number_with_given_properties_l1171_117142

theorem smallest_number_with_given_properties : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → ¬(8 ∣ m ∧ m % 2 = 1 ∧ m % 3 = 1 ∧ m % 4 = 1 ∧ m % 5 = 1 ∧ m % 7 = 1)) ∧ 
  (8 ∣ n) ∧ 
  (n % 2 = 1) ∧ 
  (n % 3 = 1) ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 7 = 1) ∧ 
  n = 7141 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_properties_l1171_117142


namespace NUMINAMATH_CALUDE_pencil_count_l1171_117130

theorem pencil_count (num_pens : ℕ) (max_students : ℕ) (num_pencils : ℕ) :
  num_pens = 1001 →
  max_students = 91 →
  (∃ (s : ℕ), s ≤ max_students ∧ num_pens % s = 0 ∧ num_pencils % s = 0) →
  ∃ (k : ℕ), num_pencils = 91 * k :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l1171_117130


namespace NUMINAMATH_CALUDE_gloria_ticket_boxes_l1171_117144

/-- Given that Gloria has 45 tickets and each box holds 5 tickets,
    prove that the number of boxes Gloria has is 9. -/
theorem gloria_ticket_boxes : ∀ (total_tickets boxes_count tickets_per_box : ℕ),
  total_tickets = 45 →
  tickets_per_box = 5 →
  total_tickets = boxes_count * tickets_per_box →
  boxes_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_gloria_ticket_boxes_l1171_117144


namespace NUMINAMATH_CALUDE_reading_days_l1171_117166

-- Define the reading speed in words per hour
def reading_speed : ℕ := 100

-- Define the number of words in each book
def book1_words : ℕ := 200
def book2_words : ℕ := 400
def book3_words : ℕ := 300

-- Define the average reading time per day in minutes
def avg_reading_time : ℕ := 54

-- Define the total number of words
def total_words : ℕ := book1_words + book2_words + book3_words

-- Theorem to prove
theorem reading_days : 
  (total_words / reading_speed : ℚ) / (avg_reading_time / 60 : ℚ) = 10 := by
  sorry


end NUMINAMATH_CALUDE_reading_days_l1171_117166


namespace NUMINAMATH_CALUDE_money_split_l1171_117136

theorem money_split (total : ℝ) (share : ℝ) (n : ℕ) :
  n = 2 →
  share = 32.5 →
  n * share = total →
  total = 65 := by
sorry

end NUMINAMATH_CALUDE_money_split_l1171_117136


namespace NUMINAMATH_CALUDE_turns_result_in_opposite_direction_l1171_117149

/-- Two turns result in opposite direction if they are in the same direction and sum to 180 degrees -/
def opposite_direction (turn1 : ℝ) (turn2 : ℝ) : Prop :=
  (turn1 > 0 ∧ turn2 > 0) ∧ turn1 + turn2 = 180

/-- The specific turns given in the problem -/
def first_turn : ℝ := 53
def second_turn : ℝ := 127

/-- Theorem stating that the given turns result in opposite direction -/
theorem turns_result_in_opposite_direction :
  opposite_direction first_turn second_turn := by
  sorry

#check turns_result_in_opposite_direction

end NUMINAMATH_CALUDE_turns_result_in_opposite_direction_l1171_117149


namespace NUMINAMATH_CALUDE_number_calculation_l1171_117105

theorem number_calculation (x : ℝ) : 0.25 * x = 0.20 * 650 + 190 → x = 1280 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l1171_117105


namespace NUMINAMATH_CALUDE_common_chord_length_l1171_117187

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 2*x - 4*y + 3 = 0

-- Theorem statement
theorem common_chord_length :
  ∃ (a b c d : ℝ),
    C₁ a b ∧ C₁ c d ∧ C₂ a b ∧ C₂ c d ∧
    common_chord a b ∧ common_chord c d ∧
    ((a - c)^2 + (b - d)^2) = 11 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l1171_117187


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1171_117184

/-- The quadratic function f(x) = ax² + mx + m - 1 -/
def f (a m x : ℝ) : ℝ := a * x^2 + m * x + m - 1

theorem quadratic_function_properties (a m : ℝ) (h_a : a ≠ 0) :
  /- Part 1: Number of zeros when f(-1) = 0 -/
  (f a m (-1) = 0 → (∃ x, f a m x = 0) ∧ (∃ x y, x ≠ y ∧ f a m x = 0 ∧ f a m y = 0)) ∧
  /- Part 2: Condition for always having two distinct zeros -/
  ((∀ m : ℝ, ∃ x y : ℝ, x ≠ y ∧ f a m x = 0 ∧ f a m y = 0) ↔ 0 < a ∧ a < 1) ∧
  /- Part 3: Existence of root between x₁ and x₂ -/
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a m x₁ ≠ f a m x₂ →
    ∃ x : ℝ, x₁ < x ∧ x < x₂ ∧ f a m x = (f a m x₁ + f a m x₂) / 2) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l1171_117184


namespace NUMINAMATH_CALUDE_rational_sum_l1171_117102

theorem rational_sum (a b : ℚ) (h : |a + 6| + (b - 4)^2 = 0) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_l1171_117102


namespace NUMINAMATH_CALUDE_parabola_focus_l1171_117192

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = 4 * x^2 - 3

-- Define the focus of a parabola
def is_focus (x y : ℝ) (p : ℝ → ℝ → Prop) : Prop :=
  ∀ (px py : ℝ), p px py →
    (px - x)^2 + (py - y)^2 = (py - (y - 1/4))^2

-- Theorem statement
theorem parabola_focus :
  is_focus 0 (-47/16) parabola := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1171_117192


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l1171_117179

theorem quadratic_inequality_always_positive (c : ℝ) :
  (∀ x : ℝ, x^2 + x + c > 0) ↔ c > 1/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l1171_117179


namespace NUMINAMATH_CALUDE_power_inequality_l1171_117123

theorem power_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a^6 + b^6 ≥ a*b*(a^4 + b^4) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1171_117123


namespace NUMINAMATH_CALUDE_function_properties_l1171_117143

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - 9*x + 11

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x - 9

theorem function_properties :
  ∃ (a : ℝ),
    (f_derivative a 1 = -12) ∧
    (a = 3) ∧
    (∀ x, f a x ≤ 16) ∧
    (∃ x, f a x = 16) ∧
    (∀ x, f a x ≥ -16) ∧
    (∃ x, f a x = -16) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1171_117143


namespace NUMINAMATH_CALUDE_square_perimeter_from_p_l1171_117145

/-- Given a square cut into four equal rectangles that form a letter P with perimeter 56,
    the perimeter of the original square is 32. -/
theorem square_perimeter_from_p (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  2 * (14 * width) = 56 →
  4 * (4 * width) = 32 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_p_l1171_117145


namespace NUMINAMATH_CALUDE_men_in_room_l1171_117119

theorem men_in_room (x : ℕ) 
  (h1 : 2 * (5 * x - 3) = 24) -- Women doubled and final count is 24
  : 4 * x + 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_men_in_room_l1171_117119


namespace NUMINAMATH_CALUDE_four_digit_number_relation_l1171_117140

theorem four_digit_number_relation : 
  let n : ℕ := 1197
  let thousands : ℕ := n / 1000
  let hundreds : ℕ := (n / 100) % 10
  let tens : ℕ := (n / 10) % 10
  let units : ℕ := n % 10
  units = hundreds - 2 →
  thousands + hundreds + tens + units = 18 →
  thousands = hundreds - 2 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_relation_l1171_117140


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_10_l1171_117177

theorem x_plus_2y_equals_10 (x y : ℝ) (hx : x = 4) (hy : y = 3) : x + 2*y = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_10_l1171_117177


namespace NUMINAMATH_CALUDE_inequality_proof_l1171_117126

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  Real.sqrt ((1 + a^2 * b) / (1 + a * b)) + 
  Real.sqrt ((1 + b^2 * c) / (1 + b * c)) + 
  Real.sqrt ((1 + c^2 * a) / (1 + c * a)) ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1171_117126


namespace NUMINAMATH_CALUDE_initial_water_is_11_l1171_117106

/-- Represents the hiking scenario with given conditions -/
structure HikeScenario where
  hikeLength : ℝ
  hikeDuration : ℝ
  leakRate : ℝ
  lastMileConsumption : ℝ
  regularConsumption : ℝ
  remainingWater : ℝ

/-- Calculates the initial amount of water in the canteen -/
def initialWater (scenario : HikeScenario) : ℝ :=
  scenario.regularConsumption * (scenario.hikeLength - 1) +
  scenario.lastMileConsumption +
  scenario.leakRate * scenario.hikeDuration +
  scenario.remainingWater

/-- Theorem stating that the initial amount of water is 11 cups -/
theorem initial_water_is_11 (scenario : HikeScenario) 
  (hLength : scenario.hikeLength = 7)
  (hDuration : scenario.hikeDuration = 3)
  (hLeak : scenario.leakRate = 1)
  (hLastMile : scenario.lastMileConsumption = 3)
  (hRegular : scenario.regularConsumption = 0.5)
  (hRemaining : scenario.remainingWater = 2) :
  initialWater scenario = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_is_11_l1171_117106


namespace NUMINAMATH_CALUDE_translated_circle_equation_l1171_117113

-- Define the points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (5, 8)

-- Define the translation vector u
def u : ℝ × ℝ := (2, -1)

-- Define the theorem
theorem translated_circle_equation :
  let diameter := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let radius := diameter / 2
  let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let new_center := (center.1 + u.1, center.2 + u.2)
  ∀ x y : ℝ, (x - new_center.1)^2 + (y - new_center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_translated_circle_equation_l1171_117113


namespace NUMINAMATH_CALUDE_fifteen_exponent_division_l1171_117134

theorem fifteen_exponent_division : (15 : ℕ)^11 / (15 : ℕ)^8 = 3375 := by sorry

end NUMINAMATH_CALUDE_fifteen_exponent_division_l1171_117134


namespace NUMINAMATH_CALUDE_prime_sequence_finite_l1171_117132

/-- A sequence of primes satisfying the given conditions -/
def PrimeSequence (p : ℕ → ℕ) : Prop :=
  (∀ n, Nat.Prime (p n)) ∧ 
  (∀ i ≥ 2, p i = 2 * p (i-1) - 1 ∨ p i = 2 * p (i-1) + 1)

/-- The theorem stating that any such sequence is finite -/
theorem prime_sequence_finite (p : ℕ → ℕ) (h : PrimeSequence p) : 
  ∃ N, ∀ n > N, ¬ Nat.Prime (p n) :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_finite_l1171_117132


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l1171_117133

theorem correct_average_after_error_correction 
  (n : Nat) 
  (initial_average : ℚ) 
  (wrong_number correct_number : ℚ) :
  n = 10 →
  initial_average = 5 →
  wrong_number = 26 →
  correct_number = 36 →
  (n : ℚ) * initial_average + (correct_number - wrong_number) = n * 6 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l1171_117133


namespace NUMINAMATH_CALUDE_aunt_zhang_laundry_additions_l1171_117151

/-- Represents the washing machine setup and calculates optimal additions -/
def optimal_laundry_additions (total_capacity : ℝ) (clothes_weight : ℝ) 
  (initial_detergent_scoops : ℕ) (scoop_weight : ℝ) (optimal_ratio : ℝ) : 
  (ℝ × ℝ) :=
  let initial_detergent := initial_detergent_scoops * scoop_weight
  let total_weight := total_capacity
  let water_weight := total_weight - clothes_weight - initial_detergent
  let optimal_detergent := water_weight * optimal_ratio
  let additional_detergent := optimal_detergent - initial_detergent
  let additional_water := water_weight
  (additional_detergent, additional_water)

/-- Theorem stating the optimal additions for Aunt Zhang's laundry -/
theorem aunt_zhang_laundry_additions : 
  let (add_detergent, add_water) := 
    optimal_laundry_additions 20 5 2 0.02 0.004
  add_detergent = 0.02 ∧ add_water = 14.94 := by
  sorry

end NUMINAMATH_CALUDE_aunt_zhang_laundry_additions_l1171_117151


namespace NUMINAMATH_CALUDE_group_size_calculation_l1171_117154

theorem group_size_calculation (initial_avg : ℝ) (final_avg : ℝ) (new_member1 : ℝ) (new_member2 : ℝ) :
  initial_avg = 48 →
  final_avg = 51 →
  new_member1 = 78 →
  new_member2 = 93 →
  ∃ n : ℕ, n * initial_avg + new_member1 + new_member2 = (n + 2) * final_avg ∧ n = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l1171_117154


namespace NUMINAMATH_CALUDE_base4_to_decimal_equality_l1171_117161

/-- Converts a base 4 number represented as a list of digits to its decimal (base 10) equivalent. -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number we want to convert. -/
def base4Number : List Nat := [3, 0, 1, 2, 1]

/-- Theorem stating that the base 4 number 30121₄ is equal to 793 in base 10. -/
theorem base4_to_decimal_equality :
  base4ToDecimal base4Number = 793 := by
  sorry

end NUMINAMATH_CALUDE_base4_to_decimal_equality_l1171_117161


namespace NUMINAMATH_CALUDE_fill_cistern_time_cistern_filling_problem_l1171_117120

/-- The time taken for two pipes to fill a cistern together -/
theorem fill_cistern_time (time_A time_B : ℝ) (h1 : time_A > 0) (h2 : time_B > 0) :
  let combined_rate := 1 / time_A + 1 / time_B
  combined_rate⁻¹ = (time_A * time_B) / (time_A + time_B) := by sorry

/-- Proof of the cistern filling problem -/
theorem cistern_filling_problem :
  let time_A : ℝ := 36  -- Time for Pipe A to fill the entire cistern
  let time_B : ℝ := 24  -- Time for Pipe B to fill the entire cistern
  let combined_time := (time_A * time_B) / (time_A + time_B)
  combined_time = 14.4 := by sorry

end NUMINAMATH_CALUDE_fill_cistern_time_cistern_filling_problem_l1171_117120


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1171_117128

def U : Set Int := {x | x^2 ≤ 2*x + 3}
def A : Set Int := {0, 1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1171_117128


namespace NUMINAMATH_CALUDE_age_problem_l1171_117189

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = a - 3 →
  a + b + c + d = 44 →
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1171_117189


namespace NUMINAMATH_CALUDE_negation_of_P_l1171_117163

-- Define the original proposition P
def P : Prop := ∃ n : ℕ, n^2 > 2^n

-- State the theorem that the negation of P is equivalent to the given statement
theorem negation_of_P : (¬ P) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_P_l1171_117163


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1171_117147

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  condition1 : a 3 + a 5 = 8
  condition2 : a 1 * a 5 = 4

/-- The ratio of the 13th term to the 9th term is 9 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : seq.a 13 / seq.a 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1171_117147


namespace NUMINAMATH_CALUDE_odd_function_value_and_range_and_inequality_l1171_117160

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 4 / (2 * a^x + a)

theorem odd_function_value_and_range_and_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, f a x = -f a (-x)) ∧
  (∀ x, -1 < (2^x - 1) / (2^x + 1) ∧ (2^x - 1) / (2^x + 1) < 1) ∧
  (∀ x ∈ Set.Ioo 0 1, ∃ t ≥ 0, ∀ s ≥ t, s * ((2^x - 1) / (2^x + 1)) ≥ 2^x - 2) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_value_and_range_and_inequality_l1171_117160


namespace NUMINAMATH_CALUDE_emily_walks_farther_l1171_117165

def troy_base_distance : ℕ := 75
def emily_base_distance : ℕ := 98

def troy_daily_distances : List ℕ := [90, 95, 85, 85, 80]
def emily_daily_distances : List ℕ := [108, 123, 108, 123, 108]

def calculate_total_distance (daily_distances : List ℕ) : ℕ :=
  2 * (daily_distances.sum)

theorem emily_walks_farther :
  calculate_total_distance emily_daily_distances - calculate_total_distance troy_daily_distances = 270 := by
  sorry

end NUMINAMATH_CALUDE_emily_walks_farther_l1171_117165


namespace NUMINAMATH_CALUDE_estimated_y_value_at_28_l1171_117162

/-- Linear regression equation -/
def linear_regression (x : ℝ) : ℝ := 4.75 * x + 257

/-- Theorem: The estimated y value is 390 when x is 28 -/
theorem estimated_y_value_at_28 : linear_regression 28 = 390 := by
  sorry

end NUMINAMATH_CALUDE_estimated_y_value_at_28_l1171_117162


namespace NUMINAMATH_CALUDE_new_vasyuki_max_area_l1171_117127

/-- Represents a city with square blocks -/
structure City where
  side_length : Real
  block_size : Real

/-- Calculates the maximum area that can be covered in a city given a walking distance -/
def max_covered_area (c : City) (walking_distance : Real) : Real :=
  sorry

/-- Theorem: The maximum area covered by walking 10 km in New-Vasyuki is 4 km² -/
theorem new_vasyuki_max_area :
  let new_vasyuki : City := { side_length := 5, block_size := 0.2 }
  max_covered_area new_vasyuki 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_new_vasyuki_max_area_l1171_117127


namespace NUMINAMATH_CALUDE_abs_ratio_greater_than_one_l1171_117112

theorem abs_ratio_greater_than_one (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  |a| / |b| > 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_greater_than_one_l1171_117112


namespace NUMINAMATH_CALUDE_incoming_students_l1171_117111

theorem incoming_students (n : ℕ) : n < 600 ∧ n % 26 = 25 ∧ n % 24 = 15 → n = 519 :=
by sorry

end NUMINAMATH_CALUDE_incoming_students_l1171_117111


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_20_mod_5_l1171_117152

theorem remainder_of_3_pow_20_mod_5 : 3^20 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_20_mod_5_l1171_117152


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1171_117157

theorem sum_with_radical_conjugate :
  let x : ℝ := 5 - Real.sqrt 500
  let y : ℝ := 5 + Real.sqrt 500
  x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1171_117157


namespace NUMINAMATH_CALUDE_board_numbers_transformation_impossibility_of_returning_to_original_numbers_l1171_117194

theorem board_numbers_transformation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a - b / 2) ^ 2 + (b + a / 2) ^ 2 > a ^ 2 + b ^ 2 := by
  sorry

theorem impossibility_of_returning_to_original_numbers :
  ∀ (numbers : List ℝ), 
  (∀ n ∈ numbers, n ≠ 0) →
  ∃ (new_numbers : List ℝ),
  (new_numbers.length = numbers.length) ∧
  (List.sum (List.map (λ x => x^2) new_numbers) > List.sum (List.map (λ x => x^2) numbers)) := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_transformation_impossibility_of_returning_to_original_numbers_l1171_117194


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1171_117186

theorem remainder_divisibility (N : ℤ) : 
  (N % 779 = 47) → (N % 19 = 9) := by
sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1171_117186


namespace NUMINAMATH_CALUDE_second_smallest_five_digit_in_pascal_l1171_117169

/-- Pascal's triangle function -/
def pascal (n k : ℕ) : ℕ := sorry

/-- Predicate to check if a number is in Pascal's triangle -/
def inPascalTriangle (x : ℕ) : Prop :=
  ∃ n k : ℕ, pascal n k = x

/-- Predicate to check if a number is a five-digit number -/
def isFiveDigit (x : ℕ) : Prop :=
  10000 ≤ x ∧ x ≤ 99999

/-- The second smallest five-digit number in Pascal's triangle is 10001 -/
theorem second_smallest_five_digit_in_pascal :
  ∃! x : ℕ, inPascalTriangle x ∧ isFiveDigit x ∧
  (∃! y : ℕ, y < x ∧ inPascalTriangle y ∧ isFiveDigit y) ∧
  x = 10001 := by sorry

end NUMINAMATH_CALUDE_second_smallest_five_digit_in_pascal_l1171_117169


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_odd_l1171_117125

def is_sum_of_five_consecutive_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, 2 * k + 1 + (2 * k + 3) + (2 * k + 5) + (2 * k + 7) + (2 * k + 9) = n

theorem sum_of_five_consecutive_odd :
  ¬ (is_sum_of_five_consecutive_odd 16) ∧
  (is_sum_of_five_consecutive_odd 40) ∧
  (is_sum_of_five_consecutive_odd 72) ∧
  (is_sum_of_five_consecutive_odd 100) ∧
  (is_sum_of_five_consecutive_odd 200) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_odd_l1171_117125


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_lawrence_county_kids_count_proof_l1171_117198

theorem lawrence_county_kids_count : ℕ → ℕ → ℕ → Prop :=
  fun kids_home kids_camp total_kids =>
    kids_home = 274865 ∧ 
    kids_camp = 38608 ∧ 
    total_kids = kids_home + kids_camp → 
    total_kids = 313473

-- The proof is omitted
theorem lawrence_county_kids_count_proof : 
  ∃ (total_kids : ℕ), lawrence_county_kids_count 274865 38608 total_kids :=
sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_count_lawrence_county_kids_count_proof_l1171_117198


namespace NUMINAMATH_CALUDE_largest_x_floor_div_l1171_117199

theorem largest_x_floor_div (x : ℝ) : 
  (∀ y : ℝ, (↑⌊y⌋ : ℝ) / y = 6 / 7 → y ≤ x) ↔ x = 35 / 6 :=
sorry

end NUMINAMATH_CALUDE_largest_x_floor_div_l1171_117199


namespace NUMINAMATH_CALUDE_product_of_numbers_l1171_117146

theorem product_of_numbers (x y : ℝ) 
  (h1 : x - y = 15) 
  (h2 : x^2 + y^2 = 578) : 
  x * y = (931 - 15 * Real.sqrt 931) / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1171_117146


namespace NUMINAMATH_CALUDE_polynomial_root_nature_l1171_117164

def P (x : ℝ) : ℝ := x^6 - 5*x^5 - 7*x^3 - 2*x + 9

theorem polynomial_root_nature :
  (∀ x < 0, P x ≠ 0) ∧ (∃ x > 0, P x = 0) :=
sorry

end NUMINAMATH_CALUDE_polynomial_root_nature_l1171_117164


namespace NUMINAMATH_CALUDE_chocolates_in_boxes_l1171_117150

theorem chocolates_in_boxes (total_chocolates : ℕ) (filled_boxes : ℕ) (loose_chocolates : ℕ) (friend_chocolates : ℕ) (box_capacity : ℕ) : 
  total_chocolates = 50 →
  filled_boxes = 3 →
  loose_chocolates = 5 →
  friend_chocolates = 25 →
  box_capacity = 15 →
  (total_chocolates - loose_chocolates) / filled_boxes = box_capacity →
  (loose_chocolates + friend_chocolates) / box_capacity = 2 := by
sorry

end NUMINAMATH_CALUDE_chocolates_in_boxes_l1171_117150


namespace NUMINAMATH_CALUDE_nickels_to_dimes_ratio_l1171_117116

/-- Represents the number of coins of each type in Tommy's collection -/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Defines Tommy's coin collection based on the given conditions -/
def tommys_collection : CoinCollection where
  pennies := 40
  nickels := 100
  dimes := 50
  quarters := 4

/-- Theorem stating the ratio of nickels to dimes in Tommy's collection -/
theorem nickels_to_dimes_ratio (c : CoinCollection) 
  (h1 : c.dimes = c.pennies + 10)
  (h2 : c.quarters = 4)
  (h3 : c.pennies = 10 * c.quarters)
  (h4 : c.nickels = 100) :
  c.nickels / c.dimes = 2 := by
  sorry

#check nickels_to_dimes_ratio tommys_collection

end NUMINAMATH_CALUDE_nickels_to_dimes_ratio_l1171_117116


namespace NUMINAMATH_CALUDE_largest_possible_median_l1171_117195

def number_set (x : ℤ) : Finset ℤ := {x, 2*x, 6, 4, 7}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (λ i => i ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ i => i ≥ m)).card ≥ s.card

theorem largest_possible_median :
  ∃ (x : ℤ), is_median 7 (number_set x) ∧
  ∀ (y : ℤ) (m : ℤ), is_median m (number_set y) → m ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_median_l1171_117195


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1171_117188

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (2 - Complex.I) ∧ Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1171_117188


namespace NUMINAMATH_CALUDE_exactly_one_correct_proposition_l1171_117117

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the perpendicular relation between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the subset relation for a line in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the "not subset" relation for a line not in a plane
variable (line_not_in_plane : Line → Plane → Prop)

theorem exactly_one_correct_proposition (a b : Line) (M : Plane) : 
  (∃! i : Fin 4, 
    (i = 0 → (parallel_line_plane a M ∧ parallel_line_plane b M → parallel_line_line a b)) ∧
    (i = 1 → (line_in_plane b M ∧ line_not_in_plane a M ∧ parallel_line_line a b → parallel_line_plane a M)) ∧
    (i = 2 → (perpendicular_line_line a b ∧ line_in_plane b M → perpendicular_line_plane a M)) ∧
    (i = 3 → (perpendicular_line_plane a M ∧ perpendicular_line_line a b → parallel_line_plane b M))) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_correct_proposition_l1171_117117


namespace NUMINAMATH_CALUDE_elon_has_13_teslas_l1171_117185

/-- The number of Teslas Chris has -/
def chris_teslas : ℕ := 6

/-- The number of Teslas Sam has -/
def sam_teslas : ℕ := chris_teslas / 2

/-- The number of Teslas Elon has -/
def elon_teslas : ℕ := sam_teslas + 10

theorem elon_has_13_teslas : elon_teslas = 13 := by
  sorry

end NUMINAMATH_CALUDE_elon_has_13_teslas_l1171_117185


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1171_117108

theorem quadratic_equation_solution : ∃ (x c d : ℕ), 
  (x^2 + 14*x = 84) ∧ 
  (x = Real.sqrt c - d) ∧ 
  (c > 0) ∧ 
  (d > 0) ∧ 
  (c + d = 140) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1171_117108


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_third_term_l1171_117175

/-- An arithmetic sequence with a positive first term and a_1 * a_2 = -2 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 > 0 ∧ a 1 * a 2 = -2 ∧ ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def CommonDifference (a : ℕ → ℝ) : ℝ :=
  (a 2) - (a 1)

/-- The third term of an arithmetic sequence -/
def ThirdTerm (a : ℕ → ℝ) : ℝ :=
  a 3

theorem arithmetic_sequence_max_third_term
  (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  (∀ d : ℝ, CommonDifference a ≤ d → ThirdTerm a ≤ ThirdTerm (fun n ↦ a 1 + (n - 1) * d)) →
  CommonDifference a = -3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_third_term_l1171_117175


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l1171_117139

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l1171_117139
