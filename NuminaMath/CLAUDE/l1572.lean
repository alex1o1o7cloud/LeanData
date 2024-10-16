import Mathlib

namespace NUMINAMATH_CALUDE_f_greater_than_one_range_l1572_157222

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

theorem f_greater_than_one_range :
  {x₀ : ℝ | f x₀ > 1} = Set.Ioi 1 ∪ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_f_greater_than_one_range_l1572_157222


namespace NUMINAMATH_CALUDE_circle_ratio_l1572_157201

theorem circle_ratio (R r a b : ℝ) (h1 : 0 < r) (h2 : r < R) (h3 : 0 < a) (h4 : 0 < b) 
  (h5 : π * R^2 = (b/a) * (π * R^2 - π * r^2)) : 
  R / r = (b / (a - b))^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l1572_157201


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l1572_157277

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (boys_soccer_percentage : ℚ) :
  total_students = 420 →
  boys = 312 →
  soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  (total_students - boys - (soccer_players - (boys_soccer_percentage * soccer_players).floor)) = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l1572_157277


namespace NUMINAMATH_CALUDE_angle_A_is_60_l1572_157215

-- Define the triangle ABC
variable (A B C : ℝ)
variable (a b c : ℝ)

-- Define the conditions
axiom acute_triangle : 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90
axiom side_a : a = 2 * Real.sqrt 3
axiom side_b : b = 2 * Real.sqrt 2
axiom angle_B : B = 45

-- Theorem to prove
theorem angle_A_is_60 : A = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_60_l1572_157215


namespace NUMINAMATH_CALUDE_max_plus_min_of_f_l1572_157279

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + 1)^2 / (Real.sin x^2 + 1)

theorem max_plus_min_of_f : 
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
                (∀ x, m ≤ f x) ∧ (∃ x, f x = m) ∧ 
                (M + m = 2) :=
sorry

end NUMINAMATH_CALUDE_max_plus_min_of_f_l1572_157279


namespace NUMINAMATH_CALUDE_atMostOneHead_atLeastTwoHeads_mutually_exclusive_atMostOneHead_atLeastTwoHeads_cover_all_l1572_157240

-- Define the sample space for tossing two coins
inductive CoinToss
  | HH -- Two heads
  | HT -- Head then tail
  | TH -- Tail then head
  | TT -- Two tails

-- Define the events
def atMostOneHead (outcome : CoinToss) : Prop :=
  outcome = CoinToss.HT ∨ outcome = CoinToss.TH ∨ outcome = CoinToss.TT

def atLeastTwoHeads (outcome : CoinToss) : Prop :=
  outcome = CoinToss.HH

-- Theorem: The events are mutually exclusive
theorem atMostOneHead_atLeastTwoHeads_mutually_exclusive :
  ∀ (outcome : CoinToss), ¬(atMostOneHead outcome ∧ atLeastTwoHeads outcome) :=
by
  sorry

-- Theorem: The events cover all possible outcomes
theorem atMostOneHead_atLeastTwoHeads_cover_all :
  ∀ (outcome : CoinToss), atMostOneHead outcome ∨ atLeastTwoHeads outcome :=
by
  sorry

end NUMINAMATH_CALUDE_atMostOneHead_atLeastTwoHeads_mutually_exclusive_atMostOneHead_atLeastTwoHeads_cover_all_l1572_157240


namespace NUMINAMATH_CALUDE_linear_function_inequality_l1572_157275

/-- A linear function passing through first, second, and fourth quadrants -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  first_quadrant : ∃ x y, x > 0 ∧ y > 0 ∧ y = a * x + b
  second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = a * x + b
  fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = a * x + b
  x_intercept : a * 2 + b = 0

/-- The solution set of a(x-1)-b > 0 for a LinearFunction is x < -1 -/
theorem linear_function_inequality (f : LinearFunction) :
  {x : ℝ | f.a * (x - 1) - f.b > 0} = {x : ℝ | x < -1} := by
  sorry

end NUMINAMATH_CALUDE_linear_function_inequality_l1572_157275


namespace NUMINAMATH_CALUDE_smallest_base_sum_l1572_157272

theorem smallest_base_sum : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a > 1 ∧ 
  b > 1 ∧
  5 * a + 2 = 2 * b + 5 ∧ 
  (∀ (a' b' : ℕ), a' ≠ b' → a' > 1 → b' > 1 → 5 * a' + 2 = 2 * b' + 5 → a + b ≤ a' + b') ∧
  a + b = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_sum_l1572_157272


namespace NUMINAMATH_CALUDE_lisa_heavier_than_sam_l1572_157296

/-- Proves that Lisa is 7.8 pounds heavier than Sam given the specified conditions -/
theorem lisa_heavier_than_sam (jack sam lisa : ℝ) 
  (total_weight : jack + sam + lisa = 210)
  (jack_weight : jack = 52)
  (sam_jack_relation : jack = sam * 0.8)
  (lisa_jack_relation : lisa = jack * 1.4) :
  lisa - sam = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_lisa_heavier_than_sam_l1572_157296


namespace NUMINAMATH_CALUDE_baking_soda_cost_is_one_l1572_157247

/-- Represents the cost of supplies for a science project. -/
structure SupplyCost where
  students : ℕ
  bowCost : ℕ
  vinegarCost : ℕ
  totalCost : ℕ

/-- Calculates the cost of each box of baking soda. -/
def bakingSodaCost (s : SupplyCost) : ℕ :=
  (s.totalCost - (s.students * (s.bowCost + s.vinegarCost))) / s.students

/-- Theorem stating that the cost of each box of baking soda is $1. -/
theorem baking_soda_cost_is_one (s : SupplyCost)
  (h1 : s.students = 23)
  (h2 : s.bowCost = 5)
  (h3 : s.vinegarCost = 2)
  (h4 : s.totalCost = 184) :
  bakingSodaCost s = 1 := by
  sorry

end NUMINAMATH_CALUDE_baking_soda_cost_is_one_l1572_157247


namespace NUMINAMATH_CALUDE_baseball_cards_l1572_157211

theorem baseball_cards (n : ℕ) : ∃ (total : ℕ), 
  (total = 3 * n + 1) ∧ (∃ (k : ℕ), total = 3 * k + 1) := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_l1572_157211


namespace NUMINAMATH_CALUDE_sum_twenty_ways_l1572_157260

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the target sum
def target_sum : ℕ := 20

-- Define the minimum value on a die
def min_value : ℕ := 1

-- Define the maximum value on a die
def max_value : ℕ := 6

-- Function to calculate the number of ways to achieve the target sum
def ways_to_achieve_sum (n d s min max : ℕ) : ℕ :=
  sorry

-- Theorem stating that the number of ways to achieve a sum of 20 with 5 dice is 721
theorem sum_twenty_ways : ways_to_achieve_sum num_dice target_sum min_value max_value = 721 := by
  sorry

end NUMINAMATH_CALUDE_sum_twenty_ways_l1572_157260


namespace NUMINAMATH_CALUDE_fred_weekend_earnings_l1572_157257

/-- Fred's earnings from delivering newspapers -/
def newspaper_earnings : ℕ := 16

/-- Fred's earnings from washing cars -/
def car_wash_earnings : ℕ := 74

/-- Fred's total earnings over the weekend -/
def total_earnings : ℕ := newspaper_earnings + car_wash_earnings

/-- Theorem stating that Fred's total earnings over the weekend equal $90 -/
theorem fred_weekend_earnings : total_earnings = 90 := by
  sorry

end NUMINAMATH_CALUDE_fred_weekend_earnings_l1572_157257


namespace NUMINAMATH_CALUDE_second_platform_speed_l1572_157207

/-- The speed of Alex's platform in ft/s -/
def alex_speed : ℝ := 1

/-- The distance Alex's platform travels before falling, in ft -/
def fall_distance : ℝ := 100

/-- The time Edward arrives after Alex's platform starts, in seconds -/
def edward_arrival_time : ℝ := 60

/-- Edward's calculation time before launching the second platform, in seconds -/
def edward_calc_time : ℝ := 5

/-- The length of both platforms, in ft -/
def platform_length : ℝ := 5

/-- The optimal speed of the second platform that maximizes Alex's transfer time -/
def optimal_speed : ℝ := 1.125

theorem second_platform_speed (v : ℝ) :
  v = optimal_speed ↔
    (v > 0) ∧
    (v * (fall_distance / alex_speed - edward_arrival_time - edward_calc_time) = 
      fall_distance - alex_speed * edward_arrival_time + platform_length) ∧
    (∀ u : ℝ, u > 0 →
      (u * (fall_distance / alex_speed - edward_arrival_time - edward_calc_time) = 
        fall_distance - alex_speed * edward_arrival_time + platform_length) →
      v ≥ u) :=
by sorry

end NUMINAMATH_CALUDE_second_platform_speed_l1572_157207


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_72_l1572_157208

theorem six_digit_divisible_by_72 (A B : ℕ) : 
  A < 10 →
  B < 10 →
  (A * 100000 + 44610 + B) % 72 = 0 →
  A + B = 12 := by
sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_72_l1572_157208


namespace NUMINAMATH_CALUDE_sidney_cat_food_l1572_157213

/-- Represents the amount of food each adult cat eats per day -/
def adult_cat_food : ℝ := 1

theorem sidney_cat_food :
  let num_kittens : ℕ := 4
  let num_adult_cats : ℕ := 3
  let initial_food : ℕ := 7
  let kitten_food_per_day : ℚ := 3/4
  let additional_food : ℕ := 35
  let days : ℕ := 7
  
  (num_kittens : ℝ) * kitten_food_per_day * days +
  (num_adult_cats : ℝ) * adult_cat_food * days =
  (initial_food : ℝ) + additional_food :=
by sorry

#check sidney_cat_food

end NUMINAMATH_CALUDE_sidney_cat_food_l1572_157213


namespace NUMINAMATH_CALUDE_no_integer_solution_l1572_157268

theorem no_integer_solution : ∀ x y : ℤ, x^2 + 3*x*y - 2*y^2 ≠ 122 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1572_157268


namespace NUMINAMATH_CALUDE_milk_jug_problem_l1572_157223

theorem milk_jug_problem (x y : ℝ) : 
  x + y = 70 ∧ 
  0.875 * x = y + 0.125 * x → 
  x = 40 ∧ y = 30 := by
sorry

end NUMINAMATH_CALUDE_milk_jug_problem_l1572_157223


namespace NUMINAMATH_CALUDE_proportionality_problem_l1572_157270

/-- Given that x is directly proportional to y², y is inversely proportional to z²,
    and x = 5 when z = 8, prove that x = 5/256 when z = 32 -/
theorem proportionality_problem (x y z : ℝ) (k₁ k₂ : ℝ) 
    (h₁ : x = k₁ * y^2)
    (h₂ : y * z^2 = k₂)
    (h₃ : x = 5 ∧ z = 8) :
  x = 5/256 ∧ z = 32 := by
  sorry

end NUMINAMATH_CALUDE_proportionality_problem_l1572_157270


namespace NUMINAMATH_CALUDE_expand_product_l1572_157284

theorem expand_product (x : ℝ) : (x - 4) * (x^2 + 2*x + 1) = x^3 - 2*x^2 - 7*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1572_157284


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l1572_157255

theorem geometric_progression_ratio (a b c : ℝ) (x : ℝ) (r : ℝ) : 
  a = 30 → b = 80 → c = 160 →
  (b + x)^2 = (a + x) * (c + x) →
  x = 160 / 3 →
  r = (b + x) / (a + x) →
  r = (c + x) / (b + x) →
  r = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l1572_157255


namespace NUMINAMATH_CALUDE_smallest_obtuse_consecutive_triangle_perimeter_l1572_157282

/-- A triangle with consecutive integer side lengths -/
structure ConsecutiveTriangle where
  a : ℕ
  sides : Fin 3 → ℕ
  consecutive : ∀ i : Fin 2, sides i.succ = sides i + 1
  valid : a > 0 ∧ sides 0 = a

/-- Checks if a triangle is obtuse -/
def isObtuse (t : ConsecutiveTriangle) : Prop :=
  let a := t.sides 0
  let b := t.sides 1
  let c := t.sides 2
  a^2 + b^2 < c^2 ∨ a^2 + c^2 < b^2 ∨ b^2 + c^2 < a^2

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveTriangle) : ℕ :=
  t.sides 0 + t.sides 1 + t.sides 2

/-- The main theorem -/
theorem smallest_obtuse_consecutive_triangle_perimeter :
  ∃ (t : ConsecutiveTriangle), isObtuse t ∧
    (∀ (t' : ConsecutiveTriangle), isObtuse t' → perimeter t ≤ perimeter t') ∧
    perimeter t = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_obtuse_consecutive_triangle_perimeter_l1572_157282


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1572_157210

theorem expression_simplification_and_evaluation (a : ℝ) 
  (h1 : a ≠ 1) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (1 - 3 / (a + 2)) / ((a^2 - 2*a + 1) / (a^2 - 4)) = (a - 2) / (a - 1) ∧
  (0 - 2) / (0 - 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1572_157210


namespace NUMINAMATH_CALUDE_robyn_packs_l1572_157233

def lucy_packs : ℕ := 19
def total_packs : ℕ := 35

theorem robyn_packs : total_packs - lucy_packs = 16 := by sorry

end NUMINAMATH_CALUDE_robyn_packs_l1572_157233


namespace NUMINAMATH_CALUDE_expression_equivalence_l1572_157261

theorem expression_equivalence (a b : ℝ) : (a) - (b) - 3 * (a + b) - b = a - 8 * b := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l1572_157261


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1572_157264

/-- The function f(x) = -x^2 + 2ax - a - a^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x - a - a^2

theorem max_value_implies_a_equals_one (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ -2) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = -2) →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1572_157264


namespace NUMINAMATH_CALUDE_grade_average_condition_l1572_157287

theorem grade_average_condition (grades : List ℤ) (n : ℕ) :
  n > 0 →
  n = grades.length →
  (grades.sum : ℚ) / n = 46 / 10 →
  ∃ k : ℕ, n = 5 * k :=
by sorry

end NUMINAMATH_CALUDE_grade_average_condition_l1572_157287


namespace NUMINAMATH_CALUDE_fifteen_guests_four_rooms_l1572_157218

/-- The number of ways to distribute n guests into k rooms such that no room is empty. -/
def distributeGuests (n k : ℕ) : ℕ :=
  (k^n : ℕ) - k * ((k-1)^n : ℕ) + (k.choose 2) * ((k-2)^n : ℕ) - (k.choose 3) * ((k-3)^n : ℕ)

/-- Theorem stating that the number of ways to distribute 15 guests into 4 rooms
    such that no room is empty is equal to 4^15 - 4 * 3^15 + 6 * 2^15 - 4. -/
theorem fifteen_guests_four_rooms :
  distributeGuests 15 4 = 4^15 - 4 * 3^15 + 6 * 2^15 - 4 := by
  sorry

#eval distributeGuests 15 4

end NUMINAMATH_CALUDE_fifteen_guests_four_rooms_l1572_157218


namespace NUMINAMATH_CALUDE_job_completion_time_l1572_157273

/-- Given workers A, B, and C who can complete a job individually in 18, 30, and 45 days respectively,
    prove that they can complete the job together in 9 days. -/
theorem job_completion_time (a b c : ℝ) (ha : a = 18) (hb : b = 30) (hc : c = 45) :
  (1 / a + 1 / b + 1 / c)⁻¹ = 9 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1572_157273


namespace NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l1572_157265

theorem margin_in_terms_of_selling_price 
  (n : ℝ) (C S M : ℝ) 
  (h1 : M = (2/n) * C) 
  (h2 : S = C + M) : 
  M = (2/(n+2)) * S := by
sorry

end NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l1572_157265


namespace NUMINAMATH_CALUDE_tire_usage_calculation_l1572_157220

/- Define the problem parameters -/
def total_miles : ℕ := 42000
def total_tires : ℕ := 7
def tires_used_simultaneously : ℕ := 6

/- Theorem statement -/
theorem tire_usage_calculation :
  let total_tire_miles : ℕ := total_miles * tires_used_simultaneously
  let miles_per_tire : ℕ := total_tire_miles / total_tires
  miles_per_tire = 36000 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_calculation_l1572_157220


namespace NUMINAMATH_CALUDE_collinearity_condition_acute_angle_condition_l1572_157245

-- Define the vectors
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -3 - m)

-- Define collinearity
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C.1 - A.1 = t * (B.1 - A.1) ∧ C.2 - A.2 = t * (B.2 - A.2)

-- Define acute angle
def acute_angle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) > 0

-- Theorem for collinearity
theorem collinearity_condition (m : ℝ) :
  collinear OA OB (OC m) ↔ m = 1/2 := by sorry

-- Theorem for acute angle
theorem acute_angle_condition (m : ℝ) :
  acute_angle OA OB (OC m) ↔ m ∈ Set.Ioo (-3/4) (1/2) ∪ Set.Ioi (1/2) := by sorry

end NUMINAMATH_CALUDE_collinearity_condition_acute_angle_condition_l1572_157245


namespace NUMINAMATH_CALUDE_image_of_negative_four_two_l1572_157225

/-- The mapping f from R² to R² defined by f(x, y) = (xy, x + y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 * p.2, p.1 + p.2)

/-- Theorem stating that f(-4, 2) = (-8, -2) -/
theorem image_of_negative_four_two :
  f (-4, 2) = (-8, -2) := by
  sorry

end NUMINAMATH_CALUDE_image_of_negative_four_two_l1572_157225


namespace NUMINAMATH_CALUDE_cube_split_59_l1572_157224

/-- The number of odd terms in the split of m³ -/
def split_terms (m : ℕ) : ℕ := (m + 2) * (m - 1) / 2

/-- The nth odd number starting from 3 -/
def nth_odd (n : ℕ) : ℕ := 2 * n + 1

theorem cube_split_59 (m : ℕ) (h1 : m > 1) :
  (∃ k, k ≤ split_terms m ∧ nth_odd k = 59) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_59_l1572_157224


namespace NUMINAMATH_CALUDE_smallest_cut_length_l1572_157293

theorem smallest_cut_length (z : ℕ) : z ≥ 9 →
  (∃ x y : ℕ, x = z / 2 ∧ y = z - 2) →
  (13 - z / 2 + 22 - z ≤ 25 - z) →
  (13 - z / 2 + 25 - z ≤ 22 - z) →
  (22 - z + 25 - z ≤ 13 - z / 2) →
  ∀ w : ℕ, w ≥ 9 → w < z →
    ¬((13 - w / 2 + 22 - w ≤ 25 - w) ∧
      (13 - w / 2 + 25 - w ≤ 22 - w) ∧
      (22 - w + 25 - w ≤ 13 - w / 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_cut_length_l1572_157293


namespace NUMINAMATH_CALUDE_only_parallel_assertion_correct_l1572_157214

/-- Represents a line in 3D space -/
structure Line3D where
  -- This is just a placeholder definition
  dummy : Unit

/-- Perpendicular relation between two lines -/
def perpendicular (a b : Line3D) : Prop :=
  sorry

/-- Skew relation between two lines -/
def skew (a b : Line3D) : Prop :=
  sorry

/-- Intersection relation between two lines -/
def intersects (a b : Line3D) : Prop :=
  sorry

/-- Coplanar relation between two lines -/
def coplanar (a b : Line3D) : Prop :=
  sorry

/-- Parallel relation between two lines -/
def parallel (a b : Line3D) : Prop :=
  sorry

/-- Theorem stating that only the parallel assertion is correct -/
theorem only_parallel_assertion_correct (a b c : Line3D) :
  (¬ (∀ a b c, perpendicular a b → perpendicular b c → perpendicular a c)) ∧
  (¬ (∀ a b c, skew a b → skew b c → skew a c)) ∧
  (¬ (∀ a b c, intersects a b → intersects b c → intersects a c)) ∧
  (¬ (∀ a b c, coplanar a b → coplanar b c → coplanar a c)) ∧
  (∀ a b c, parallel a b → parallel b c → parallel a c) :=
by sorry

end NUMINAMATH_CALUDE_only_parallel_assertion_correct_l1572_157214


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1572_157263

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 2) :
  let set := List.replicate (n - 2) 2 ++ [1 - 2 / n, 1 - 2 / n]
  (List.sum set) / n = 2 - 2 / n - 4 / n^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1572_157263


namespace NUMINAMATH_CALUDE_opposite_of_pi_l1572_157290

theorem opposite_of_pi : -(Real.pi) = -Real.pi := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_pi_l1572_157290


namespace NUMINAMATH_CALUDE_suzy_book_count_l1572_157291

/-- Calculates the final number of books Suzy has after three days of transactions -/
def final_book_count (initial : ℕ) (wed_out : ℕ) (thu_in : ℕ) (thu_out : ℕ) (fri_in : ℕ) : ℕ :=
  initial - wed_out + thu_in - thu_out + fri_in

/-- Theorem stating that given the specific transactions, Suzy ends up with 80 books -/
theorem suzy_book_count : 
  final_book_count 98 43 23 5 7 = 80 := by
  sorry

#eval final_book_count 98 43 23 5 7

end NUMINAMATH_CALUDE_suzy_book_count_l1572_157291


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l1572_157242

def income : ℕ := 21000
def savings : ℕ := 3000
def expenditure : ℕ := income - savings

def ratio_income_expenditure : ℚ := income / expenditure

theorem income_expenditure_ratio :
  ratio_income_expenditure = 7 / 6 := by sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l1572_157242


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l1572_157230

theorem ratio_a_to_b (a b : ℚ) (h : 2 * a = 3 * b) : 
  ∃ (k : ℚ), k > 0 ∧ a = (3 * k) ∧ b = (2 * k) := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l1572_157230


namespace NUMINAMATH_CALUDE_abs_neg_three_not_pm_three_l1572_157276

theorem abs_neg_three_not_pm_three : ¬(|(-3 : ℤ)| = 3 ∧ |(-3 : ℤ)| = -3) := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_not_pm_three_l1572_157276


namespace NUMINAMATH_CALUDE_battle_station_staffing_l1572_157232

theorem battle_station_staffing (n m : ℕ) (h1 : n = 12) (h2 : m = 4) :
  (n.factorial / ((n - m).factorial * m.factorial)) = 11880 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l1572_157232


namespace NUMINAMATH_CALUDE_existence_of_n_l1572_157241

theorem existence_of_n (k : ℕ+) : ∃ n : ℤ, 
  Real.sqrt (n + 1981^k.val : ℝ) + Real.sqrt (n : ℝ) = (Real.sqrt 1982 + 1)^k.val := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l1572_157241


namespace NUMINAMATH_CALUDE_sum_of_four_digit_numbers_eq_93324_l1572_157267

def digits : List Nat := [2, 4, 5, 3]

/-- The sum of all four-digit numbers formed by using the digits 2, 4, 5, and 3 once each -/
def sum_of_four_digit_numbers : Nat :=
  let sum_of_digits := digits.sum
  let count_per_place := Nat.factorial 4 / 4
  sum_of_digits * count_per_place * (1000 + 100 + 10 + 1)

theorem sum_of_four_digit_numbers_eq_93324 :
  sum_of_four_digit_numbers = 93324 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_numbers_eq_93324_l1572_157267


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l1572_157289

/-- The perimeter of a semi-circle with radius 31.50774690151576 cm is 162.12300409103152 cm. -/
theorem semicircle_perimeter : 
  let r : ℝ := 31.50774690151576
  let π : ℝ := Real.pi
  let semicircle_perimeter : ℝ := π * r + 2 * r
  semicircle_perimeter = 162.12300409103152 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l1572_157289


namespace NUMINAMATH_CALUDE_no_primes_in_perm_numbers_l1572_157231

/-- A permutation of the digits 1, 2, 3, 4, 5 -/
def Perm5 : Type := Fin 5 → Fin 5

/-- Converts a permutation to a 5-digit number -/
def toNumber (p : Perm5) : ℕ :=
  10000 * (p 0).val + 1000 * (p 1).val + 100 * (p 2).val + 10 * (p 3).val + (p 4).val + 11111

/-- The set of all 5-digit numbers formed by permutations of 1, 2, 3, 4, 5 -/
def PermNumbers : Set ℕ :=
  {n | ∃ p : Perm5, toNumber p = n}

theorem no_primes_in_perm_numbers : ∀ n ∈ PermNumbers, ¬ Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_perm_numbers_l1572_157231


namespace NUMINAMATH_CALUDE_unique_positive_integers_sum_l1572_157226

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 37) / 3 + 5 / 3)

theorem unique_positive_integers_sum (d e f : ℕ+) : 
  y^50 = 3*y^48 + 10*y^45 + 9*y^43 - y^25 + (d:ℝ)*y^21 + (e:ℝ)*y^19 + (f:ℝ)*y^15 →
  d + e + f = 119 := by sorry

end NUMINAMATH_CALUDE_unique_positive_integers_sum_l1572_157226


namespace NUMINAMATH_CALUDE_max_N_value_l1572_157286

def N (a b c : ℕ) : ℕ := a * b * c + a * b + b * c + a - b - c

theorem max_N_value :
  ∃ (a b c : ℕ),
    a ∈ ({2, 3, 4, 5, 6} : Set ℕ) ∧
    b ∈ ({2, 3, 4, 5, 6} : Set ℕ) ∧
    c ∈ ({2, 3, 4, 5, 6} : Set ℕ) ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∀ (x y z : ℕ),
      x ∈ ({2, 3, 4, 5, 6} : Set ℕ) →
      y ∈ ({2, 3, 4, 5, 6} : Set ℕ) →
      z ∈ ({2, 3, 4, 5, 6} : Set ℕ) →
      x ≠ y → y ≠ z → x ≠ z →
      N a b c ≥ N x y z ∧
    N a b c = 167 :=
  sorry

end NUMINAMATH_CALUDE_max_N_value_l1572_157286


namespace NUMINAMATH_CALUDE_mushroom_collection_proof_l1572_157209

theorem mushroom_collection_proof :
  ∃ (x₁ x₂ x₃ x₄ : ℕ),
    x₁ + x₂ = 6 ∧
    x₁ + x₃ = 7 ∧
    x₁ + x₄ = 9 ∧
    x₂ + x₃ = 9 ∧
    x₂ + x₄ = 11 ∧
    x₃ + x₄ = 12 ∧
    x₁ = 2 ∧
    x₂ = 4 ∧
    x₃ = 5 ∧
    x₄ = 7 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_proof_l1572_157209


namespace NUMINAMATH_CALUDE_unique_number_guess_l1572_157278

/-- Represents the color feedback for a digit guess -/
inductive Color
  | Green
  | Yellow
  | Gray

/-- Represents a single round of guessing -/
structure GuessRound where
  digits : Fin 5 → Nat
  colors : Fin 5 → Color

/-- The set of all possible digits (0-9) -/
def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The correct five-digit number we're trying to prove -/
def CorrectNumber : Fin 5 → Nat := ![7, 1, 2, 8, 4]

theorem unique_number_guess (round1 round2 round3 : GuessRound) : 
  (round1.digits = ![2, 6, 1, 3, 8] ∧ 
   round1.colors = ![Color.Yellow, Color.Gray, Color.Yellow, Color.Gray, Color.Yellow]) →
  (round2.digits = ![4, 1, 9, 6, 2] ∧
   round2.colors = ![Color.Yellow, Color.Green, Color.Gray, Color.Gray, Color.Yellow]) →
  (round3.digits = ![8, 1, 0, 2, 5] ∧
   round3.colors = ![Color.Yellow, Color.Green, Color.Gray, Color.Yellow, Color.Gray]) →
  (∀ n : Fin 5, CorrectNumber n ∈ Digits) →
  (∀ i j : Fin 5, i ≠ j → CorrectNumber i ≠ CorrectNumber j) →
  CorrectNumber = ![7, 1, 2, 8, 4] := by
  sorry


end NUMINAMATH_CALUDE_unique_number_guess_l1572_157278


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1572_157216

-- System (1)
theorem system_one_solution (x y : ℚ) :
  (3 * x + 4 * y = 16) ∧ (5 * x - 8 * y = 34) → x = 6 ∧ y = -1/2 := by sorry

-- System (2)
theorem system_two_solution (x y : ℚ) :
  ((x - 1) / 2 + (y + 1) / 3 = 1) ∧ (x + y = 4) → x = -1 ∧ y = 5 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1572_157216


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l1572_157262

theorem least_positive_angle_theorem (θ : Real) : 
  (θ > 0 ∧ θ ≤ π / 2) → 
  (Real.cos (10 * π / 180) = Real.sin (20 * π / 180) + Real.sin θ) → 
  θ = 40 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l1572_157262


namespace NUMINAMATH_CALUDE_lunch_expense_calculation_l1572_157250

theorem lunch_expense_calculation (initial_money : ℝ) (gasoline_expense : ℝ) (gift_expense_per_person : ℝ) (grandma_gift_per_person : ℝ) (return_trip_money : ℝ) :
  initial_money = 50 →
  gasoline_expense = 8 →
  gift_expense_per_person = 5 →
  grandma_gift_per_person = 10 →
  return_trip_money = 36.35 →
  let total_money := initial_money + 2 * grandma_gift_per_person
  let total_expense := gasoline_expense + 2 * gift_expense_per_person
  let lunch_expense := total_money - total_expense - return_trip_money
  lunch_expense = 15.65 := by
sorry

end NUMINAMATH_CALUDE_lunch_expense_calculation_l1572_157250


namespace NUMINAMATH_CALUDE_sum_of_integers_l1572_157283

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 8) 
  (h2 : x.val * y.val = 135) : 
  x.val + y.val = 26 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1572_157283


namespace NUMINAMATH_CALUDE_total_hotdogs_sold_l1572_157253

/-- Represents the number of hotdogs sold in each size category -/
structure HotdogSales where
  small : Float
  medium : Float
  large : Float
  extra_large : Float

/-- Calculates the total number of hotdogs sold -/
def total_hotdogs (sales : HotdogSales) : Float :=
  sales.small + sales.medium + sales.large + sales.extra_large

/-- Theorem: The total number of hotdogs sold is 131.3 -/
theorem total_hotdogs_sold (sales : HotdogSales)
  (h1 : sales.small = 58.3)
  (h2 : sales.medium = 21.7)
  (h3 : sales.large = 35.9)
  (h4 : sales.extra_large = 15.4) :
  total_hotdogs sales = 131.3 := by
  sorry

#eval total_hotdogs { small := 58.3, medium := 21.7, large := 35.9, extra_large := 15.4 }

end NUMINAMATH_CALUDE_total_hotdogs_sold_l1572_157253


namespace NUMINAMATH_CALUDE_parallelograms_in_divided_triangle_l1572_157274

/-- The number of parallelograms formed in a triangle with sides divided into n equal parts -/
def num_parallelograms (n : ℕ) : ℕ :=
  3 * (Nat.choose (n + 2) 4)

/-- Theorem stating the number of parallelograms in a divided triangle -/
theorem parallelograms_in_divided_triangle (n : ℕ) :
  num_parallelograms n = 3 * (Nat.choose (n + 2) 4) :=
by sorry

end NUMINAMATH_CALUDE_parallelograms_in_divided_triangle_l1572_157274


namespace NUMINAMATH_CALUDE_fun_run_ratio_l1572_157266

def runners_last_year : ℕ := 200 - 40
def runners_this_year : ℕ := 320

theorem fun_run_ratio : 
  (runners_this_year : ℚ) / (runners_last_year : ℚ) = 2 := by sorry

end NUMINAMATH_CALUDE_fun_run_ratio_l1572_157266


namespace NUMINAMATH_CALUDE_markus_final_candies_l1572_157254

theorem markus_final_candies 
  (markus_initial : ℕ) 
  (katharina_initial : ℕ) 
  (sanjiv_distribution : ℕ) 
  (h1 : markus_initial = 9)
  (h2 : katharina_initial = 5)
  (h3 : sanjiv_distribution = 10)
  (h4 : ∃ (x : ℕ), x + markus_initial + x + katharina_initial = markus_initial + katharina_initial + sanjiv_distribution) :
  ∃ (markus_final : ℕ), markus_final = 12 ∧ 2 * markus_final = markus_initial + katharina_initial + sanjiv_distribution :=
sorry

end NUMINAMATH_CALUDE_markus_final_candies_l1572_157254


namespace NUMINAMATH_CALUDE_final_student_score_problem_solution_l1572_157285

theorem final_student_score (total_students : ℕ) (graded_students : ℕ) 
  (initial_average : ℚ) (final_average : ℚ) : ℚ :=
  let remaining_students := total_students - graded_students
  let initial_total := initial_average * graded_students
  let final_total := final_average * total_students
  (final_total - initial_total) / remaining_students

theorem problem_solution :
  final_student_score 20 19 75 78 = 135 := by sorry

end NUMINAMATH_CALUDE_final_student_score_problem_solution_l1572_157285


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l1572_157205

theorem polynomial_product_sum (p q : ℚ) : 
  (∀ x, (4 * x^2 - 5 * x + p) * (6 * x^2 + q * x - 12) = 
   24 * x^4 - 62 * x^3 - 69 * x^2 + 94 * x - 36) → 
  p + q = 43 / 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l1572_157205


namespace NUMINAMATH_CALUDE_trains_meet_at_11am_l1572_157246

/-- The distance between stations A and B in kilometers -/
def distance_between_stations : ℝ := 155

/-- The speed of the first train in km/h -/
def speed_train1 : ℝ := 20

/-- The speed of the second train in km/h -/
def speed_train2 : ℝ := 25

/-- The time difference between the trains' departures in hours -/
def time_difference : ℝ := 1

/-- The meeting time of the trains after the second train's departure -/
def meeting_time : ℝ := 3

theorem trains_meet_at_11am :
  speed_train1 * (time_difference + meeting_time) +
  speed_train2 * meeting_time = distance_between_stations :=
sorry

end NUMINAMATH_CALUDE_trains_meet_at_11am_l1572_157246


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1572_157256

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1572_157256


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l1572_157258

/-- Given that 34 cows eat 34 bags of husk in 34 days, prove that one cow will eat one bag of husk in 34 days. -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 34 ∧ bags = 34 ∧ days = 34) :
  (1 : ℕ) * days = 34 := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l1572_157258


namespace NUMINAMATH_CALUDE_total_work_hours_l1572_157295

theorem total_work_hours (hours_per_day : ℕ) (days_worked : ℕ) : 
  hours_per_day = 3 → days_worked = 5 → hours_per_day * days_worked = 15 :=
by sorry

end NUMINAMATH_CALUDE_total_work_hours_l1572_157295


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1572_157297

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 3 ∧ x₂ = -5 ∧ 
  x₁^2 + 2*x₁ - 15 = 0 ∧ 
  x₂^2 + 2*x₂ - 15 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1572_157297


namespace NUMINAMATH_CALUDE_convex_quadrilaterals_count_l1572_157204

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of distinct points on the circle -/
def num_points : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

theorem convex_quadrilaterals_count :
  binomial num_points vertices_per_quadrilateral = 495 := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilaterals_count_l1572_157204


namespace NUMINAMATH_CALUDE_hidden_integers_product_l1572_157244

theorem hidden_integers_product (w x y z : ℕ+) 
  (h1 : x * y * z = 280)
  (h2 : w * y * z = 168)
  (h3 : w * x * z = 105)
  (h4 : w * x * y = 120) :
  w * x * y * z = 840 := by
  sorry

end NUMINAMATH_CALUDE_hidden_integers_product_l1572_157244


namespace NUMINAMATH_CALUDE_video_game_points_l1572_157248

/-- 
Given a video game where:
- Each enemy defeated gives 9 points
- There are 11 enemies total in a level
- You destroy all but 3 enemies

Prove that the number of points earned is 72.
-/
theorem video_game_points : 
  (∀ (points_per_enemy : ℕ) (total_enemies : ℕ) (enemies_left : ℕ),
    points_per_enemy = 9 → 
    total_enemies = 11 → 
    enemies_left = 3 → 
    (total_enemies - enemies_left) * points_per_enemy = 72) :=
by sorry

end NUMINAMATH_CALUDE_video_game_points_l1572_157248


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1572_157259

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1572_157259


namespace NUMINAMATH_CALUDE_vector_addition_problem_l1572_157252

theorem vector_addition_problem (a b : ℝ × ℝ) :
  a = (2, -1) → b = (-3, 4) → 2 • a + b = (1, 2) := by sorry

end NUMINAMATH_CALUDE_vector_addition_problem_l1572_157252


namespace NUMINAMATH_CALUDE_emilys_number_proof_l1572_157227

theorem emilys_number_proof :
  ∃! n : ℕ, 
    (216 ∣ n) ∧ 
    (45 ∣ n) ∧ 
    (1000 < n) ∧ 
    (n < 3000) ∧ 
    (n = 2160) := by
  sorry

end NUMINAMATH_CALUDE_emilys_number_proof_l1572_157227


namespace NUMINAMATH_CALUDE_square_roots_problem_l1572_157229

theorem square_roots_problem (x : ℝ) (n : ℝ) (h1 : n > 0) 
  (h2 : x + 1 = Real.sqrt n) (h3 : x - 5 = Real.sqrt n) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1572_157229


namespace NUMINAMATH_CALUDE_initial_water_percentage_l1572_157237

/-- Given a container with capacity 100 liters, prove that the initial percentage
    of water is 30% if adding 45 liters makes it 3/4 full. -/
theorem initial_water_percentage
  (capacity : ℝ)
  (added_water : ℝ)
  (final_fraction : ℝ)
  (h1 : capacity = 100)
  (h2 : added_water = 45)
  (h3 : final_fraction = 3/4)
  (h4 : (initial_percentage / 100) * capacity + added_water = final_fraction * capacity) :
  initial_percentage = 30 :=
sorry

#check initial_water_percentage

end NUMINAMATH_CALUDE_initial_water_percentage_l1572_157237


namespace NUMINAMATH_CALUDE_largest_square_four_digits_base7_l1572_157238

/-- Converts a decimal number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ := sorry

/-- Checks if a number has exactly 4 digits when written in base 7 -/
def hasFourDigitsBase7 (n : ℕ) : Prop :=
  (toBase7 n).length = 4

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := sorry

theorem largest_square_four_digits_base7 :
  M = (toBase7 66).foldl (fun acc d => acc * 7 + d) 0 ∧
  hasFourDigitsBase7 (M ^ 2) ∧
  ∀ n : ℕ, n > M → ¬hasFourDigitsBase7 (n ^ 2) :=
sorry

end NUMINAMATH_CALUDE_largest_square_four_digits_base7_l1572_157238


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l1572_157281

/-- Given a square and a circle that intersect such that each side of the square
    contains a chord of the circle equal in length to twice the radius of the circle,
    the ratio of the area of the square to the area of the circle is 2/π. -/
theorem square_circle_area_ratio (s : Real) (r : Real) (π : Real) :
  s > 0 ∧ r > 0 ∧ π > 0 ∧ 
  (2 * r = s) ∧  -- chord length equals side length
  (π * r^2 = π * r * r) →  -- definition of circle area
  (s^2 / (π * r^2) = 2 / π) :=
sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l1572_157281


namespace NUMINAMATH_CALUDE_power_equation_l1572_157288

theorem power_equation (x y : ℝ) (m n : ℝ) 
  (hm : 10^x = m) (hn : 10^y = n) : 
  10^(2*x + 3*y) = m^2 * n^3 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l1572_157288


namespace NUMINAMATH_CALUDE_trajectory_of_M_l1572_157269

/-- The trajectory of point M satisfying the given conditions -/
theorem trajectory_of_M (x y : ℝ) (h : x ≥ 3/2) :
  (∀ (t : ℝ), t^2 + y^2 = 1 → 
    Real.sqrt ((x - t)^2 + y^2) = Real.sqrt ((x - 2)^2 + y^2) + 1) →
  3 * x^2 - y^2 - 8 * x + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_M_l1572_157269


namespace NUMINAMATH_CALUDE_dans_car_mpg_l1572_157294

/-- Calculates the miles per gallon of Dan's car given the cost of gas and distance traveled on a certain amount of money. -/
theorem dans_car_mpg (gas_cost : ℝ) (miles : ℝ) (spent : ℝ) : 
  gas_cost = 4 → miles = 432 → spent = 54 → (miles / (spent / gas_cost)) = 32 :=
by sorry

end NUMINAMATH_CALUDE_dans_car_mpg_l1572_157294


namespace NUMINAMATH_CALUDE_ticket_price_increase_l1572_157298

theorem ticket_price_increase (P V : ℝ) (h1 : P > 0) (h2 : V > 0) : 
  (P + 0.5 * P) * (0.8 * V) = 1.2 * (P * V) := by sorry

#check ticket_price_increase

end NUMINAMATH_CALUDE_ticket_price_increase_l1572_157298


namespace NUMINAMATH_CALUDE_smallest_largest_multiples_l1572_157202

theorem smallest_largest_multiples :
  ∃ (smallest largest : ℕ),
    (smallest ≥ 10 ∧ smallest < 100) ∧
    (largest ≥ 100 ∧ largest < 1000) ∧
    (∀ n : ℕ, n ≥ 10 ∧ n < 100 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n → n ≥ smallest) ∧
    (∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n → n ≤ largest) ∧
    2 ∣ smallest ∧ 3 ∣ smallest ∧ 5 ∣ smallest ∧
    2 ∣ largest ∧ 3 ∣ largest ∧ 5 ∣ largest ∧
    smallest = 30 ∧ largest = 990 := by
  sorry

end NUMINAMATH_CALUDE_smallest_largest_multiples_l1572_157202


namespace NUMINAMATH_CALUDE_koi_fish_after_six_weeks_l1572_157206

/-- Represents the number of fish in the tank -/
structure FishTank where
  koi : ℕ
  goldfish : ℕ
  angelfish : ℕ

/-- Calculates the total number of fish in the tank -/
def FishTank.total (ft : FishTank) : ℕ := ft.koi + ft.goldfish + ft.angelfish

/-- Represents the daily and weekly changes in fish numbers -/
structure FishChanges where
  koi_per_day : ℕ
  goldfish_per_day : ℕ
  angelfish_per_week : ℕ

/-- Calculates the new fish numbers after a given number of weeks -/
def apply_changes (initial : FishTank) (changes : FishChanges) (weeks : ℕ) : FishTank :=
  { koi := initial.koi + changes.koi_per_day * 7 * weeks,
    goldfish := initial.goldfish + changes.goldfish_per_day * 7 * weeks,
    angelfish := initial.angelfish + changes.angelfish_per_week * weeks }

theorem koi_fish_after_six_weeks
  (initial : FishTank)
  (changes : FishChanges)
  (h_initial_total : initial.total = 450)
  (h_changes : changes = { koi_per_day := 4, goldfish_per_day := 7, angelfish_per_week := 9 })
  (h_final_goldfish : (apply_changes initial changes 6).goldfish = 300)
  (h_final_angelfish : (apply_changes initial changes 6).angelfish = 180) :
  (apply_changes initial changes 6).koi = 486 :=
sorry

end NUMINAMATH_CALUDE_koi_fish_after_six_weeks_l1572_157206


namespace NUMINAMATH_CALUDE_hyperbola_focus_l1572_157234

/-- Definition of the hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  -x^2 + 2*y^2 - 10*x - 16*y + 1 = 0

/-- Theorem stating that one of the foci of the hyperbola is at (-5, 7) or (-5, 1) -/
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ ((x = -5 ∧ y = 7) ∨ (x = -5 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l1572_157234


namespace NUMINAMATH_CALUDE_smallest_factor_of_4814_l1572_157271

theorem smallest_factor_of_4814 (a b : ℕ) : 
  10 ≤ a ∧ a ≤ 99 ∧
  10 ≤ b ∧ b ≤ 99 ∧
  a * b = 4814 ∧
  a ≤ b →
  a = 53 := by sorry

end NUMINAMATH_CALUDE_smallest_factor_of_4814_l1572_157271


namespace NUMINAMATH_CALUDE_lasagna_ratio_is_two_to_one_l1572_157251

/-- Represents the ratio of noodles to beef in Tom's lasagna recipe -/
def lasagna_ratio (beef_amount : ℕ) (initial_noodles : ℕ) (package_weight : ℕ) (packages_needed : ℕ) : ℚ :=
  let total_noodles := initial_noodles + package_weight * packages_needed
  (total_noodles : ℚ) / beef_amount

/-- The ratio of noodles to beef in Tom's lasagna recipe is 2:1 -/
theorem lasagna_ratio_is_two_to_one :
  lasagna_ratio 10 4 2 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lasagna_ratio_is_two_to_one_l1572_157251


namespace NUMINAMATH_CALUDE_min_value_constraint_l1572_157221

theorem min_value_constraint (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + 2*y = 1) :
  2*x + 3*y^2 ≥ 0.75 := by
  sorry

end NUMINAMATH_CALUDE_min_value_constraint_l1572_157221


namespace NUMINAMATH_CALUDE_inequality_solution_l1572_157228

open Set

theorem inequality_solution (x : ℝ) : 
  3 * x - 2 < (x + 2)^2 ∧ (x + 2)^2 < 9 * x - 8 ↔ x ∈ Ioo 3 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1572_157228


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l1572_157217

theorem unique_two_digit_integer (t : ℕ) : 
  (10 ≤ t ∧ t < 100) ∧ (13 * t) % 100 = 52 ↔ t = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l1572_157217


namespace NUMINAMATH_CALUDE_points_one_unit_from_negative_two_l1572_157280

theorem points_one_unit_from_negative_two : 
  ∀ x : ℝ, abs (x - (-2)) = 1 ↔ x = -3 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_points_one_unit_from_negative_two_l1572_157280


namespace NUMINAMATH_CALUDE_circle_graph_proportion_l1572_157235

theorem circle_graph_proportion (angle : ℝ) (percentage : ℝ) :
  angle = 180 →
  angle / 360 = percentage / 100 →
  percentage = 50 := by
sorry

end NUMINAMATH_CALUDE_circle_graph_proportion_l1572_157235


namespace NUMINAMATH_CALUDE_triangle_properties_l1572_157203

-- Define the points in the complex plane
def A : ℂ := 1
def B : ℂ := -Complex.I
def C : ℂ := -1 + 2 * Complex.I

-- Define the vectors
def AB : ℂ := B - A
def AC : ℂ := C - A
def BC : ℂ := C - B

-- Theorem statement
theorem triangle_properties :
  (AB.re = -1 ∧ AB.im = -1) ∧
  (AC.re = -2 ∧ AC.im = 2) ∧
  (BC.re = -1 ∧ BC.im = 3) ∧
  (AB.re * AC.re + AB.im * AC.im = 0) := by
  sorry

-- The last condition (AB.re * AC.re + AB.im * AC.im = 0) checks if AB and AC are perpendicular,
-- which implies that the triangle is right-angled.

end NUMINAMATH_CALUDE_triangle_properties_l1572_157203


namespace NUMINAMATH_CALUDE_inequality_proof_l1572_157292

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1572_157292


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1572_157249

theorem polynomial_remainder (x : ℝ) : 
  (x^15 + 3) % (x + 2) = -32765 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1572_157249


namespace NUMINAMATH_CALUDE_prob_two_red_balls_l1572_157212

/-- The probability of picking two red balls from a bag containing 3 red balls, 2 blue balls,
    and 3 green balls, when 2 balls are picked at random without replacement. -/
theorem prob_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ)
    (h1 : total_balls = red_balls + blue_balls + green_balls)
    (h2 : red_balls = 3)
    (h3 : blue_balls = 2)
    (h4 : green_balls = 3) :
    (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_balls_l1572_157212


namespace NUMINAMATH_CALUDE_bicycle_price_increase_l1572_157219

theorem bicycle_price_increase (P : ℝ) : 
  (P * 1.15 = 253) → P = 220 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_increase_l1572_157219


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1572_157200

theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℤ) -- a is the arithmetic sequence
  (h1 : a 0 = 3^2) -- first term is 3^2
  (h2 : a 2 = 3^4) -- third term is 3^4
  (h3 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- arithmetic sequence
  : a 1 = 45 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1572_157200


namespace NUMINAMATH_CALUDE_total_shirts_count_l1572_157243

/-- The number of packages of white t-shirts bought -/
def num_packages : ℕ := 28

/-- The number of white t-shirts in each package -/
def shirts_per_package : ℕ := 2

/-- The total number of white t-shirts -/
def total_shirts : ℕ := num_packages * shirts_per_package

theorem total_shirts_count : total_shirts = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_shirts_count_l1572_157243


namespace NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l1572_157236

theorem arccos_gt_arctan_iff (x : ℝ) : Real.arccos x > Real.arctan x ↔ x ∈ Set.Ici (-1) ∩ Set.Iio 1 := by
  sorry

end NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l1572_157236


namespace NUMINAMATH_CALUDE_highest_of_seven_consecutive_with_average_33_l1572_157239

theorem highest_of_seven_consecutive_with_average_33 :
  ∀ (a : ℤ), 
  (∃ (x : ℤ), a = x - 3 ∧ 
    (x - 3 + (x - 2) + (x - 1) + x + (x + 1) + (x + 2) + (x + 3)) / 7 = 33) →
  (∃ (y : ℤ), a + 6 = y ∧ y = 36) :=
by sorry

end NUMINAMATH_CALUDE_highest_of_seven_consecutive_with_average_33_l1572_157239


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1572_157299

theorem fraction_evaluation (a b : ℝ) (h1 : a = 7) (h2 : b = 4) :
  5 / (a - b)^2 = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1572_157299
