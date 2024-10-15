import Mathlib

namespace NUMINAMATH_CALUDE_sequence_existence_condition_l2277_227781

def is_valid_sequence (x : ℕ → Fin 2) (n m : ℕ) : Prop :=
  (∀ i, x i = 0 → x (i + m) = 1) ∧ (∀ i, x i = 1 → x (i + n) = 0)

theorem sequence_existence_condition (n m : ℕ) :
  (∃ x : ℕ → Fin 2, is_valid_sequence x n m) ↔
  (∃ (d p q : ℕ), n = 2^d * p ∧ m = 2^d * q ∧ Odd p ∧ Odd q) :=
sorry

end NUMINAMATH_CALUDE_sequence_existence_condition_l2277_227781


namespace NUMINAMATH_CALUDE_greek_cross_dissection_l2277_227733

/-- Represents a symmetric Greek cross -/
structure SymmetricGreekCross where
  -- Add necessary properties to define a symmetric Greek cross

/-- Represents a square -/
structure Square where
  -- Add necessary properties to define a square

/-- Represents a part of the dissected cross -/
inductive CrossPart
  | SmallCross : SymmetricGreekCross → CrossPart
  | OtherPart : CrossPart

/-- Theorem stating that a symmetric Greek cross can be dissected as described -/
theorem greek_cross_dissection (cross : SymmetricGreekCross) :
  ∃ (parts : Finset CrossPart) (square : Square),
    parts.card = 5 ∧
    (∃ small_cross : SymmetricGreekCross, CrossPart.SmallCross small_cross ∈ parts) ∧
    (∃ other_parts : Finset CrossPart,
      other_parts.card = 4 ∧
      (∀ p ∈ other_parts, p ∈ parts ∧ p ≠ CrossPart.SmallCross small_cross) ∧
      -- Here we would need to define how the other parts form the square
      True) := by
  sorry

end NUMINAMATH_CALUDE_greek_cross_dissection_l2277_227733


namespace NUMINAMATH_CALUDE_derrick_has_34_pictures_l2277_227746

/-- The number of wild animal pictures Ralph has -/
def ralph_pictures : ℕ := 26

/-- The additional number of pictures Derrick has compared to Ralph -/
def additional_pictures : ℕ := 8

/-- The number of wild animal pictures Derrick has -/
def derrick_pictures : ℕ := ralph_pictures + additional_pictures

/-- Theorem stating that Derrick has 34 wild animal pictures -/
theorem derrick_has_34_pictures : derrick_pictures = 34 := by sorry

end NUMINAMATH_CALUDE_derrick_has_34_pictures_l2277_227746


namespace NUMINAMATH_CALUDE_restaurant_period_days_l2277_227756

def pies_per_day : ℕ := 8
def total_pies : ℕ := 56

theorem restaurant_period_days : 
  total_pies / pies_per_day = 7 := by sorry

end NUMINAMATH_CALUDE_restaurant_period_days_l2277_227756


namespace NUMINAMATH_CALUDE_oliver_seashell_collection_l2277_227752

-- Define the number of seashells collected on each day
def monday_shells : ℕ := 2
def tuesday_shells : ℕ := 2

-- Define the total number of seashells
def total_shells : ℕ := monday_shells + tuesday_shells

-- Theorem statement
theorem oliver_seashell_collection : total_shells = 4 := by
  sorry

end NUMINAMATH_CALUDE_oliver_seashell_collection_l2277_227752


namespace NUMINAMATH_CALUDE_true_discount_calculation_l2277_227798

/-- Calculates the true discount given the banker's discount and present value -/
def true_discount (bankers_discount : ℚ) (present_value : ℚ) : ℚ :=
  bankers_discount / (1 + bankers_discount / present_value)

/-- Theorem stating that given a banker's discount of 36 and a present value of 180, 
    the true discount is 30 -/
theorem true_discount_calculation :
  true_discount 36 180 = 30 := by
  sorry

#eval true_discount 36 180

end NUMINAMATH_CALUDE_true_discount_calculation_l2277_227798


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2277_227717

theorem quadratic_equation_solution (x : ℝ) : x^2 - 5 = 0 ↔ x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2277_227717


namespace NUMINAMATH_CALUDE_plain_croissant_price_l2277_227788

/-- The price of Sean's Sunday pastry purchase --/
def sean_pastry_purchase : ℝ → Prop :=
  fun plain_croissant_price =>
    let almond_croissant_price : ℝ := 4.50
    let salami_cheese_croissant_price : ℝ := 4.50
    let focaccia_price : ℝ := 4.00
    let latte_price : ℝ := 2.50
    let total_spent : ℝ := 21.00
    
    almond_croissant_price +
    salami_cheese_croissant_price +
    plain_croissant_price +
    focaccia_price +
    2 * latte_price = total_spent

theorem plain_croissant_price : ∃ (price : ℝ), sean_pastry_purchase price ∧ price = 3.00 := by
  sorry

end NUMINAMATH_CALUDE_plain_croissant_price_l2277_227788


namespace NUMINAMATH_CALUDE_max_surface_area_30_cubes_l2277_227768

/-- Represents a configuration of connected unit cubes -/
structure CubeConfiguration where
  num_cubes : ℕ
  surface_area : ℕ

/-- The number of cubes in our problem -/
def total_cubes : ℕ := 30

/-- Function to calculate the surface area of a linear arrangement of cubes -/
def linear_arrangement_surface_area (n : ℕ) : ℕ :=
  if n ≤ 1 then 6 * n else 2 + 4 * n

/-- Theorem stating that the maximum surface area for 30 connected unit cubes is 122 -/
theorem max_surface_area_30_cubes :
  (∀ c : CubeConfiguration, c.num_cubes = total_cubes → c.surface_area ≤ 122) ∧
  (∃ c : CubeConfiguration, c.num_cubes = total_cubes ∧ c.surface_area = 122) := by
  sorry

#eval linear_arrangement_surface_area total_cubes

end NUMINAMATH_CALUDE_max_surface_area_30_cubes_l2277_227768


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_l2277_227754

theorem min_max_abs_quadratic :
  ∃ y : ℝ, ∀ z : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x^2 - x*y + x| ≤ (⨆ x ∈ {x : ℝ | 0 ≤ x ∧ x ≤ 2}, |x^2 - x*z + x|)) ∧
  (⨆ x ∈ {x : ℝ | 0 ≤ x ∧ x ≤ 2}, |x^2 - x*y + x|) = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_l2277_227754


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2277_227789

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 3 ≥ C*(x + y + z)) ↔ C ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2277_227789


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l2277_227706

-- Define the vectors a and b
def a (x : ℝ) : Fin 3 → ℝ := λ i => match i with
  | 0 => 2
  | 1 => 4
  | 2 => x

def b (y : ℝ) : Fin 3 → ℝ := λ i => match i with
  | 0 => 2
  | 1 => y
  | 2 => 2

-- Theorem statement
theorem parallel_vectors_sum (x y : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i : Fin 3, a x i = k * b y i)) →
  x + y = 6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l2277_227706


namespace NUMINAMATH_CALUDE_quadratic_completion_l2277_227792

theorem quadratic_completion (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 36 = (x + m)^2 + 20) → 
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_l2277_227792


namespace NUMINAMATH_CALUDE_algebra_test_female_students_l2277_227728

theorem algebra_test_female_students 
  (total_average : ℝ) 
  (num_male : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 88) 
  (h2 : num_male = 15) 
  (h3 : male_average = 80) 
  (h4 : female_average = 94) : 
  ∃ (num_female : ℕ), 
    (num_male * male_average + num_female * female_average) / (num_male + num_female) = total_average ∧ 
    num_female = 20 := by
sorry


end NUMINAMATH_CALUDE_algebra_test_female_students_l2277_227728


namespace NUMINAMATH_CALUDE_sum_zero_iff_squared_sum_equal_l2277_227735

theorem sum_zero_iff_squared_sum_equal {a b c : ℝ} (h : ¬(a = b ∧ b = c)) :
  a + b + c = 0 ↔ a^2 + a*b + b^2 = b^2 + b*c + c^2 ∧ b^2 + b*c + c^2 = c^2 + c*a + a^2 :=
sorry

end NUMINAMATH_CALUDE_sum_zero_iff_squared_sum_equal_l2277_227735


namespace NUMINAMATH_CALUDE_orange_juice_concentrate_size_l2277_227716

/-- The size of a can of orange juice concentrate in ounces -/
def concentrate_size : ℝ := 420

/-- The number of servings to be prepared -/
def num_servings : ℕ := 280

/-- The size of each serving in ounces -/
def serving_size : ℝ := 6

/-- The ratio of water cans to concentrate cans -/
def water_to_concentrate_ratio : ℝ := 3

theorem orange_juice_concentrate_size :
  concentrate_size * (1 + water_to_concentrate_ratio) * num_servings = serving_size * num_servings :=
sorry

end NUMINAMATH_CALUDE_orange_juice_concentrate_size_l2277_227716


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l2277_227707

theorem negation_of_existential_proposition :
  (¬ ∃ a : ℝ, a ≥ -1 ∧ Real.log (Real.exp n + 1) > 1/2) ↔
  (∀ a : ℝ, a ≥ -1 → Real.log (Real.exp n + 1) ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l2277_227707


namespace NUMINAMATH_CALUDE_expected_worth_of_coin_flip_l2277_227718

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Probability of getting heads on a single flip -/
def probHeads : ℚ := 2/3

/-- Probability of getting tails on a single flip -/
def probTails : ℚ := 1/3

/-- Reward for getting heads -/
def rewardHeads : ℚ := 5

/-- Penalty for getting tails -/
def penaltyTails : ℚ := -9

/-- Additional penalty for three consecutive tails -/
def penaltyThreeTails : ℚ := -10

/-- Expected value of a single coin flip -/
def expectedValueSingleFlip : ℚ := probHeads * rewardHeads + probTails * penaltyTails

/-- Probability of getting three consecutive tails -/
def probThreeTails : ℚ := probTails^3

/-- Additional expected loss from three consecutive tails -/
def expectedAdditionalLoss : ℚ := probThreeTails * penaltyThreeTails

/-- Total expected value of a coin flip -/
def totalExpectedValue : ℚ := expectedValueSingleFlip + expectedAdditionalLoss

theorem expected_worth_of_coin_flip :
  totalExpectedValue = -1/27 := by sorry

end NUMINAMATH_CALUDE_expected_worth_of_coin_flip_l2277_227718


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l2277_227711

theorem smallest_dual_base_representation : ∃ (a b : ℕ), 
  a > 3 ∧ b > 3 ∧ 
  13 = 1 * a + 3 ∧
  13 = 3 * b + 1 ∧
  (∀ (x y : ℕ), x > 3 → y > 3 → 1 * x + 3 = 3 * y + 1 → 1 * x + 3 ≥ 13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l2277_227711


namespace NUMINAMATH_CALUDE_box_length_l2277_227767

/-- The length of a rectangular box given specific conditions --/
theorem box_length (width : ℝ) (volume_gallons : ℝ) (height_inches : ℝ) (conversion_factor : ℝ) : 
  width = 25 →
  volume_gallons = 4687.5 →
  height_inches = 6 →
  conversion_factor = 7.5 →
  ∃ (length : ℝ), length = 50 := by
sorry

end NUMINAMATH_CALUDE_box_length_l2277_227767


namespace NUMINAMATH_CALUDE_always_not_three_l2277_227701

def is_single_digit (n : ℕ) : Prop := n < 10

def statement_I (n : ℕ) : Prop := n = 2
def statement_II (n : ℕ) : Prop := n ≠ 3
def statement_III (n : ℕ) : Prop := n = 5
def statement_IV (n : ℕ) : Prop := Even n

theorem always_not_three (n : ℕ) (h_single_digit : is_single_digit n) 
  (h_three_true : ∃ (a b c : Prop) (ha : a) (hb : b) (hc : c), 
    (a = statement_I n ∨ a = statement_II n ∨ a = statement_III n ∨ a = statement_IV n) ∧
    (b = statement_I n ∨ b = statement_II n ∨ b = statement_III n ∨ b = statement_IV n) ∧
    (c = statement_I n ∨ c = statement_II n ∨ c = statement_III n ∨ c = statement_IV n) ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  statement_II n := by
  sorry

end NUMINAMATH_CALUDE_always_not_three_l2277_227701


namespace NUMINAMATH_CALUDE_money_sharing_l2277_227710

theorem money_sharing (amanda ben carlos diana total : ℕ) : 
  amanda = 45 →
  amanda + ben + carlos + diana = total →
  3 * ben = 5 * amanda →
  3 * carlos = 6 * amanda →
  3 * diana = 8 * amanda →
  total = 330 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l2277_227710


namespace NUMINAMATH_CALUDE_bathroom_size_is_150_l2277_227771

/-- Represents the size of a bathroom module in square feet -/
def bathroom_size : ℝ := sorry

/-- Represents the total size of the home in square feet -/
def total_home_size : ℝ := 2000

/-- Represents the size of the kitchen module in square feet -/
def kitchen_size : ℝ := 400

/-- Represents the cost of the kitchen module in dollars -/
def kitchen_cost : ℝ := 20000

/-- Represents the cost of a bathroom module in dollars -/
def bathroom_cost : ℝ := 12000

/-- Represents the cost per square foot of other modules in dollars -/
def other_module_cost_per_sqft : ℝ := 100

/-- Represents the total cost of the home in dollars -/
def total_home_cost : ℝ := 174000

/-- Represents the number of bathrooms in the home -/
def num_bathrooms : ℕ := 2

/-- Theorem stating that the bathroom size is 150 square feet -/
theorem bathroom_size_is_150 : bathroom_size = 150 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_size_is_150_l2277_227771


namespace NUMINAMATH_CALUDE_first_platform_length_l2277_227738

/-- Given a train and two platforms, calculate the length of the first platform. -/
theorem first_platform_length
  (train_length : ℝ)
  (first_crossing_time : ℝ)
  (second_platform_length : ℝ)
  (second_crossing_time : ℝ)
  (h1 : train_length = 310)
  (h2 : first_crossing_time = 15)
  (h3 : second_platform_length = 250)
  (h4 : second_crossing_time = 20) :
  ∃ (first_platform_length : ℝ),
    first_platform_length = 110 ∧
    (train_length + first_platform_length) / first_crossing_time =
    (train_length + second_platform_length) / second_crossing_time :=
by sorry

end NUMINAMATH_CALUDE_first_platform_length_l2277_227738


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2277_227783

/-- 
Given x = γ sin((θ - α)/2) and y = γ sin((θ + α)/2), 
prove that x^2 - 2xy cos α + y^2 = γ^2 sin^2 α
-/
theorem trigonometric_identity 
  (γ θ α x y : ℝ) 
  (hx : x = γ * Real.sin ((θ - α) / 2))
  (hy : y = γ * Real.sin ((θ + α) / 2)) :
  x^2 - 2*x*y*Real.cos α + y^2 = γ^2 * Real.sin α^2 := by
  sorry


end NUMINAMATH_CALUDE_trigonometric_identity_l2277_227783


namespace NUMINAMATH_CALUDE_hostel_provisions_l2277_227725

/-- Given a hostel with provisions for a certain number of men, 
    calculate the initial number of days the provisions were planned for. -/
theorem hostel_provisions 
  (initial_men : ℕ) 
  (men_left : ℕ) 
  (days_after_leaving : ℕ) 
  (h1 : initial_men = 250)
  (h2 : men_left = 50)
  (h3 : days_after_leaving = 60) :
  (initial_men * (initial_men - men_left) * days_after_leaving) / 
  ((initial_men - men_left) * initial_men) = 48 := by
sorry

end NUMINAMATH_CALUDE_hostel_provisions_l2277_227725


namespace NUMINAMATH_CALUDE_function_f_at_zero_l2277_227744

/-- A function f: ℝ → ℝ satisfying f(x+y) = f(x) + f(y) + 1/2 for all real x and y -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + (1/2 : ℝ)

/-- Theorem: For a function f satisfying the given property, f(0) = -1/2 -/
theorem function_f_at_zero (f : ℝ → ℝ) (h : FunctionF f) : f 0 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_f_at_zero_l2277_227744


namespace NUMINAMATH_CALUDE_f_range_l2277_227775

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x - 5

-- Define the domain
def domain : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem f_range : 
  {y : ℝ | ∃ x ∈ domain, f x = y} = {y : ℝ | -9 ≤ y ∧ y ≤ 7} := by
  sorry

end NUMINAMATH_CALUDE_f_range_l2277_227775


namespace NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l2277_227751

theorem coefficient_x6_in_expansion : ∃ c : ℤ, c = -10 ∧ 
  (Polynomial.coeff ((1 + X + X^2) * (1 - X)^6) 6 = c) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l2277_227751


namespace NUMINAMATH_CALUDE_circle_radius_l2277_227796

/-- A circle with center (0, k) where k > 10, tangent to y = x, y = -x, y = 10, and x-axis has radius 20 -/
theorem circle_radius (k : ℝ) (h1 : k > 10) : 
  let circle := { (x, y) | x^2 + (y - k)^2 = (k - 10)^2 }
  (∀ (x y : ℝ), (x = y ∨ x = -y ∨ y = 10 ∨ y = 0) → 
    (x, y) ∈ circle → x^2 + (y - k)^2 = (k - 10)^2) →
  k - 10 = 20 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l2277_227796


namespace NUMINAMATH_CALUDE_rectangle_opposite_vertex_l2277_227794

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Predicate to check if four points form a rectangle --/
def is_rectangle (r : Rectangle) : Prop :=
  let midpoint1 := ((r.v1.1 + r.v3.1) / 2, (r.v1.2 + r.v3.2) / 2)
  let midpoint2 := ((r.v2.1 + r.v4.1) / 2, (r.v2.2 + r.v4.2) / 2)
  midpoint1 = midpoint2

/-- The theorem to be proved --/
theorem rectangle_opposite_vertex 
  (r : Rectangle)
  (h1 : r.v1 = (5, 10))
  (h2 : r.v3 = (15, -6))
  (h3 : r.v2 = (11, 2))
  (h4 : is_rectangle r) :
  r.v4 = (9, 2) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_opposite_vertex_l2277_227794


namespace NUMINAMATH_CALUDE_platform_length_l2277_227705

/-- The length of a platform given a train's speed, crossing time, and length -/
theorem platform_length 
  (train_speed : Real) 
  (crossing_time : Real) 
  (train_length : Real) : 
  train_speed = 72 * (5/18) → 
  crossing_time = 36 → 
  train_length = 470.06 → 
  (train_speed * crossing_time) - train_length = 249.94 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2277_227705


namespace NUMINAMATH_CALUDE_sarah_trucks_left_l2277_227715

/-- The number of trucks Sarah has left after giving some away -/
def trucks_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Sarah has 38 trucks left after starting with 51 and giving away 13 -/
theorem sarah_trucks_left :
  trucks_left 51 13 = 38 := by
  sorry

end NUMINAMATH_CALUDE_sarah_trucks_left_l2277_227715


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2277_227795

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x - 3 = 0 ∧ y^2 + m*y - 3 = 0) ∧
  (3^2 + m*3 - 3 = 0 → (-1)^2 + m*(-1) - 3 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2277_227795


namespace NUMINAMATH_CALUDE_prime_square_mod_60_l2277_227763

-- Define the set of primes greater than 3
def PrimesGreaterThan3 : Set ℕ := {p : ℕ | Nat.Prime p ∧ p > 3}

-- Theorem statement
theorem prime_square_mod_60 (p : ℕ) (h : p ∈ PrimesGreaterThan3) : 
  p ^ 2 % 60 = 1 ∨ p ^ 2 % 60 = 49 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_60_l2277_227763


namespace NUMINAMATH_CALUDE_largest_A_when_quotient_equals_remainder_l2277_227727

theorem largest_A_when_quotient_equals_remainder (A B C : ℕ) : 
  A = 7 * B + C → B = C → A ≤ 48 ∧ ∃ (A₀ B₀ C₀ : ℕ), A₀ = 7 * B₀ + C₀ ∧ B₀ = C₀ ∧ A₀ = 48 := by
  sorry

end NUMINAMATH_CALUDE_largest_A_when_quotient_equals_remainder_l2277_227727


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2277_227782

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 3 = 4) →
  (a 4 + a 5 = 16) →
  (a 8 + a 9 = 256) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2277_227782


namespace NUMINAMATH_CALUDE_focal_length_of_specific_conic_l2277_227750

/-- A conic section centered at the origin with coordinate axes as its axes of symmetry -/
structure ConicSection where
  /-- The eccentricity of the conic section -/
  eccentricity : ℝ
  /-- A point that the conic section passes through -/
  point : ℝ × ℝ

/-- The focal length of a conic section -/
def focalLength (c : ConicSection) : ℝ := sorry

/-- Theorem: The focal length of the specified conic section is 6√2 -/
theorem focal_length_of_specific_conic :
  let c : ConicSection := { eccentricity := Real.sqrt 2, point := (5, 4) }
  focalLength c = 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_focal_length_of_specific_conic_l2277_227750


namespace NUMINAMATH_CALUDE_rectangular_equation_of_C_chord_length_l2277_227703

-- Define the polar curve C
def polar_curve (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 8 * Real.cos θ

-- Define the line l
def line_l (t x y : ℝ) : Prop := x = 2 + t ∧ y = Real.sqrt 3 * t

-- Theorem for the rectangular equation of curve C
theorem rectangular_equation_of_C (x y : ℝ) :
  (∃ ρ θ, polar_curve ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  y^2 = 8*x :=
sorry

-- Theorem for the chord length |AB|
theorem chord_length (A B : ℝ × ℝ) :
  (∃ t₁, line_l t₁ A.1 A.2 ∧ A.2^2 = 8*A.1) →
  (∃ t₂, line_l t₂ B.1 B.2 ∧ B.2^2 = 8*B.1) →
  A ≠ B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32/3 :=
sorry

end NUMINAMATH_CALUDE_rectangular_equation_of_C_chord_length_l2277_227703


namespace NUMINAMATH_CALUDE_negation_of_all_cats_not_pets_l2277_227760

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for "is a cat" and "is a pet"
variable (Cat : U → Prop)
variable (Pet : U → Prop)

-- Define the original statement "All cats are not pets"
def all_cats_not_pets : Prop := ∀ x, Cat x → ¬(Pet x)

-- Define the negation "Some cats are pets"
def some_cats_are_pets : Prop := ∃ x, Cat x ∧ Pet x

-- Theorem statement
theorem negation_of_all_cats_not_pets :
  ¬(all_cats_not_pets U Cat Pet) ↔ some_cats_are_pets U Cat Pet :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_cats_not_pets_l2277_227760


namespace NUMINAMATH_CALUDE_other_piece_price_is_96_l2277_227736

/-- The price of one of the other pieces of clothing --/
def other_piece_price (total_spent : ℕ) (num_pieces : ℕ) (price1 : ℕ) (price2 : ℕ) : ℕ :=
  (total_spent - price1 - price2) / (num_pieces - 2)

/-- Theorem stating that the price of one of the other pieces is 96 --/
theorem other_piece_price_is_96 :
  other_piece_price 610 7 49 81 = 96 := by
  sorry

end NUMINAMATH_CALUDE_other_piece_price_is_96_l2277_227736


namespace NUMINAMATH_CALUDE_cubic_difference_l2277_227778

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 65) : 
  a^3 - b^3 = 511 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l2277_227778


namespace NUMINAMATH_CALUDE_divisibility_problem_l2277_227745

theorem divisibility_problem (x y : ℕ) : 
  (∀ z : ℕ, z < x → ¬((1056 + z) % 28 = 0 ∧ (1056 + z) % 42 = 0)) ∧
  ((1056 + x) % 28 = 0 ∧ (1056 + x) % 42 = 0) ∧
  (∀ w : ℕ, w > y → ¬((1056 - w) % 28 = 0 ∧ (1056 - w) % 42 = 0)) ∧
  ((1056 - y) % 28 = 0 ∧ (1056 - y) % 42 = 0) →
  x = 36 ∧ y = 48 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2277_227745


namespace NUMINAMATH_CALUDE_joey_age_l2277_227758

def brothers_ages : List ℕ := [4, 6, 8, 10, 12]

def movies_condition (a b : ℕ) : Prop := a + b = 18

def park_condition (a b : ℕ) : Prop := a < 9 ∧ b < 9

theorem joey_age : 
  ∃ (a b c d : ℕ),
    a ∈ brothers_ages ∧
    b ∈ brothers_ages ∧
    c ∈ brothers_ages ∧
    d ∈ brothers_ages ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    movies_condition a b ∧
    park_condition c d ∧
    6 ∉ [a, b, c, d] →
    10 ∈ brothers_ages \ [a, b, c, d, 6] :=
by
  sorry

end NUMINAMATH_CALUDE_joey_age_l2277_227758


namespace NUMINAMATH_CALUDE_find_y_l2277_227747

theorem find_y : ∃ y : ℚ, (12 : ℚ)^2 * (6 : ℚ)^3 / y = 72 → y = 432 := by sorry

end NUMINAMATH_CALUDE_find_y_l2277_227747


namespace NUMINAMATH_CALUDE_dollar_operation_theorem_l2277_227723

/-- Define the dollar operation -/
def dollar (k : ℝ) (a b : ℝ) : ℝ := k * (a - b)^2

/-- Theorem stating that (2x - 3y)² $₃ (3y - 2x)² = 0 for any real x and y -/
theorem dollar_operation_theorem (x y : ℝ) : 
  dollar 3 ((2*x - 3*y)^2) ((3*y - 2*x)^2) = 0 := by
  sorry

#check dollar_operation_theorem

end NUMINAMATH_CALUDE_dollar_operation_theorem_l2277_227723


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l2277_227702

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_roots : (a 2)^2 + 12*(a 2) - 8 = 0 ∧ (a 10)^2 + 12*(a 10) - 8 = 0) :
  a 6 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l2277_227702


namespace NUMINAMATH_CALUDE_prime_triplet_existence_l2277_227786

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_triplet_existence :
  (∃ n : ℕ, isPrime (n - 96) ∧ isPrime n ∧ isPrime (n + 96)) ∧
  (¬∃ n : ℕ, isPrime (n - 1996) ∧ isPrime n ∧ isPrime (n + 1996)) :=
sorry

end NUMINAMATH_CALUDE_prime_triplet_existence_l2277_227786


namespace NUMINAMATH_CALUDE_greatest_number_less_than_200_with_odd_factors_l2277_227721

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_odd_number_of_factors (n : ℕ) : Prop := is_perfect_square n

theorem greatest_number_less_than_200_with_odd_factors : 
  (∀ n : ℕ, n < 200 → has_odd_number_of_factors n → n ≤ 196) ∧ 
  has_odd_number_of_factors 196 ∧ 
  196 < 200 :=
sorry

end NUMINAMATH_CALUDE_greatest_number_less_than_200_with_odd_factors_l2277_227721


namespace NUMINAMATH_CALUDE_sophists_count_l2277_227730

/-- Represents the types of inhabitants on the Isle of Logic -/
inductive Inhabitant
  | Knight
  | Liar
  | Sophist

/-- The Isle of Logic and its inhabitants -/
structure IsleOfLogic where
  knights : Nat
  liars : Nat
  sophists : Nat

/-- Predicate to check if a statement is valid for a sophist -/
def isSophistStatement (isle : IsleOfLogic) (statementLiars : Nat) : Prop :=
  statementLiars ≠ isle.liars ∧ 
  ¬(statementLiars = isle.liars + 1 ∧ isle.sophists > isle.liars)

/-- Theorem: The number of sophists on the Isle of Logic -/
theorem sophists_count (isle : IsleOfLogic) : 
  isle.knights = 40 →
  isle.liars = 25 →
  isSophistStatement isle 26 →
  isle.sophists ≤ 26 →
  isle.sophists = 27 := by
  sorry

end NUMINAMATH_CALUDE_sophists_count_l2277_227730


namespace NUMINAMATH_CALUDE_zeros_sum_greater_than_four_l2277_227769

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp x - k * x + k

theorem zeros_sum_greater_than_four (k : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : k > Real.exp 2)
  (h₂ : f k x₁ = 0)
  (h₃ : f k x₂ = 0)
  (h₄ : x₁ ≠ x₂) :
  x₁ + x₂ > 4 := by
  sorry

end NUMINAMATH_CALUDE_zeros_sum_greater_than_four_l2277_227769


namespace NUMINAMATH_CALUDE_stamp_exchange_theorem_l2277_227739

/-- Represents the number of stamp collectors and countries -/
def n : ℕ := 26

/-- The minimum number of letters needed to exchange stamps -/
def min_letters (n : ℕ) : ℕ := 2 * (n - 1)

/-- Theorem stating the minimum number of letters needed for stamp exchange -/
theorem stamp_exchange_theorem :
  min_letters n = 50 :=
by sorry

end NUMINAMATH_CALUDE_stamp_exchange_theorem_l2277_227739


namespace NUMINAMATH_CALUDE_no_solution_for_four_l2277_227772

theorem no_solution_for_four : 
  ∀ X : ℕ, X < 10 →
  (∀ Y : ℕ, Y < 10 → ¬(100 * X + 30 + Y) % 11 = 0) ↔ X = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_four_l2277_227772


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2277_227785

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (6, 3)
  let b : ℝ → ℝ × ℝ := fun m ↦ (m, 2)
  ∀ m : ℝ, are_parallel a (b m) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2277_227785


namespace NUMINAMATH_CALUDE_arrange_five_photos_l2277_227708

theorem arrange_five_photos (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_photos_l2277_227708


namespace NUMINAMATH_CALUDE_cut_prism_edge_count_l2277_227741

/-- A rectangular prism with cut corners -/
structure CutPrism where
  /-- The number of vertices in the original rectangular prism -/
  original_vertices : Nat
  /-- The number of edges in the original rectangular prism -/
  original_edges : Nat
  /-- The number of new edges created by each cut -/
  new_edges_per_cut : Nat
  /-- The planes cutting the prism do not intersect within the prism -/
  non_intersecting_cuts : Prop

/-- The number of edges in the new figure after cutting the corners -/
def new_edge_count (p : CutPrism) : Nat :=
  p.original_edges + p.original_vertices * p.new_edges_per_cut

/-- Theorem stating that a rectangular prism with cut corners has 36 edges -/
theorem cut_prism_edge_count :
  ∀ (p : CutPrism),
  p.original_vertices = 8 →
  p.original_edges = 12 →
  p.new_edges_per_cut = 3 →
  p.non_intersecting_cuts →
  new_edge_count p = 36 := by
  sorry

end NUMINAMATH_CALUDE_cut_prism_edge_count_l2277_227741


namespace NUMINAMATH_CALUDE_no_double_application_function_l2277_227732

theorem no_double_application_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 2017 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l2277_227732


namespace NUMINAMATH_CALUDE_five_solutions_l2277_227787

/-- The number of distinct ordered pairs of positive integers satisfying the equation -/
def count_solutions : ℕ := 5

/-- The equation that the ordered pairs must satisfy -/
def satisfies_equation (x y : ℕ+) : Prop :=
  (x.val ^ 4 * y.val ^ 4) - (20 * x.val ^ 2 * y.val ^ 2) + 64 = 0

/-- The theorem stating that there are exactly 5 distinct ordered pairs satisfying the equation -/
theorem five_solutions :
  (∃! (s : Finset (ℕ+ × ℕ+)), s.card = count_solutions ∧
    ∀ p ∈ s, satisfies_equation p.1 p.2 ∧
    ∀ p : ℕ+ × ℕ+, satisfies_equation p.1 p.2 → p ∈ s) :=
  sorry

end NUMINAMATH_CALUDE_five_solutions_l2277_227787


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l2277_227764

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x * (x - a) < 0}
def B : Set ℝ := {x | x^2 - 7*x - 18 < 0}

-- State the theorem
theorem set_inclusion_implies_a_range (a : ℝ) : A a ⊆ B → a ∈ Set.Icc (-2) 9 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l2277_227764


namespace NUMINAMATH_CALUDE_f_expression_and_g_monotonicity_l2277_227755

/-- A linear function f that is increasing on ℝ and satisfies f(f(x)) = 16x + 5 -/
def f : ℝ → ℝ :=
  sorry

/-- g is defined as g(x) = f(x)(x+m) -/
def g (m : ℝ) : ℝ → ℝ :=
  λ x ↦ f x * (x + m)

theorem f_expression_and_g_monotonicity :
  (∀ x y, x < y → f x < f y) ∧  -- f is increasing
  (∀ x, f (f x) = 16 * x + 5) →  -- f(f(x)) = 16x + 5
  (f = λ x ↦ 4 * x + 1) ∧  -- f(x) = 4x + 1
  (∀ m, (∀ x y, 1 < x ∧ x < y → g m x < g m y) → -9/4 ≤ m)  -- If g is increasing on (1,+∞), then m ≥ -9/4
  := by sorry

end NUMINAMATH_CALUDE_f_expression_and_g_monotonicity_l2277_227755


namespace NUMINAMATH_CALUDE_solve_inequality_one_solve_inequality_two_l2277_227766

namespace InequalitySolver

-- Part 1
theorem solve_inequality_one (x : ℝ) :
  (3 * x - 2) / (x - 1) > 1 ↔ x > 1 :=
sorry

-- Part 2
theorem solve_inequality_two (x a : ℝ) :
  (a = 0 → ¬∃x, x^2 - a*x - 2*a^2 < 0) ∧
  (a > 0 → (x^2 - a*x - 2*a^2 < 0 ↔ -a < x ∧ x < 2*a)) ∧
  (a < 0 → (x^2 - a*x - 2*a^2 < 0 ↔ 2*a < x ∧ x < -a)) :=
sorry

end InequalitySolver

end NUMINAMATH_CALUDE_solve_inequality_one_solve_inequality_two_l2277_227766


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l2277_227731

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m ^ 3 = n

def prime_factors (n : ℕ) : List ℕ := sorry

def satisfies_conditions (n : ℕ) : Prop :=
  n > 0 ∧
  ¬ is_prime n ∧
  ¬ is_cube n ∧
  (prime_factors n).length % 2 = 0 ∧
  ∀ p ∈ prime_factors n, p > 60

theorem smallest_satisfying_number : 
  satisfies_conditions 3721 ∧ 
  ∀ m : ℕ, m < 3721 → ¬ satisfies_conditions m :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l2277_227731


namespace NUMINAMATH_CALUDE_altitude_length_l2277_227737

/-- Given a rectangle with length l and width w, and a triangle constructed on its diagonal
    with an area equal to the rectangle's area, the length of the altitude drawn from the
    opposite vertex of the triangle to the diagonal is (2lw) / √(l^2 + w^2). -/
theorem altitude_length (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := (1/2) * diagonal * altitude
  altitude = (2 * l * w) / diagonal →
  triangle_area = rectangle_area :=
by
  sorry


end NUMINAMATH_CALUDE_altitude_length_l2277_227737


namespace NUMINAMATH_CALUDE_complete_square_sum_l2277_227729

theorem complete_square_sum (x : ℝ) : 
  (x^2 - 10*x + 15 = 0) → 
  ∃ (a b : ℤ), ((x + a : ℝ)^2 = b) ∧ (a + b = 5) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_sum_l2277_227729


namespace NUMINAMATH_CALUDE_cos_squared_derivative_l2277_227784

theorem cos_squared_derivative (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = (Real.cos (2 * x))^2) →
  (deriv f) x = -2 * Real.sin (4 * x) := by
sorry

end NUMINAMATH_CALUDE_cos_squared_derivative_l2277_227784


namespace NUMINAMATH_CALUDE_test_total_points_l2277_227722

theorem test_total_points (total_questions : ℕ) (two_point_questions : ℕ) : 
  total_questions = 40 → 
  two_point_questions = 30 → 
  (total_questions - two_point_questions) * 4 + two_point_questions * 2 = 100 := by
sorry

end NUMINAMATH_CALUDE_test_total_points_l2277_227722


namespace NUMINAMATH_CALUDE_negation_equivalence_l2277_227797

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 2 * x^2 < Real.cos x) ↔ (∀ x : ℝ, 2 * x^2 ≥ Real.cos x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2277_227797


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2277_227720

theorem system_of_equations_solution (a b : ℝ) 
  (eq1 : 2 * a + b = 7) 
  (eq2 : a - b = 2) : 
  3 * a = 9 := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2277_227720


namespace NUMINAMATH_CALUDE_expression_evaluation_l2277_227704

theorem expression_evaluation : (4 * 6) / (12 * 16) * (8 * 12 * 16) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2277_227704


namespace NUMINAMATH_CALUDE_total_paid_is_705_l2277_227770

/-- Calculates the total amount paid for fruits given their quantities and rates -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem: The total amount paid for the given quantities and rates of grapes and mangoes is 705 -/
theorem total_paid_is_705 :
  total_amount_paid 3 70 9 55 = 705 := by
  sorry

end NUMINAMATH_CALUDE_total_paid_is_705_l2277_227770


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l2277_227748

/-- A line in 2D space, represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space, represented by the equation (x-h)^2 + (y-k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  (c.h * l.a + c.k * l.b + l.c)^2 = (l.a^2 + l.b^2) * c.r^2

/-- The main theorem -/
theorem tangent_lines_to_circle (l : Line) (c : Circle) :
  (l.a = 2 ∧ l.b = -1 ∧ c.h = 0 ∧ c.k = 0 ∧ c.r^2 = 5) →
  (∃ l1 l2 : Line,
    (are_parallel l l1 ∧ is_tangent l1 c) ∧
    (are_parallel l l2 ∧ is_tangent l2 c) ∧
    (l1.c = 5 ∨ l1.c = -5) ∧
    (l2.c = 5 ∨ l2.c = -5) ∧
    (l1.c + l2.c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l2277_227748


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_30_l2277_227712

/-- The maximum area of a rectangle with perimeter 30 meters is 225/4 square meters. -/
theorem max_area_rectangle_with_perimeter_30 :
  let perimeter : ℝ := 30
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 * (x + y) = perimeter ∧
    ∀ (a b : ℝ), a > 0 → b > 0 → 2 * (a + b) = perimeter →
      x * y ≥ a * b ∧ x * y = 225 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_30_l2277_227712


namespace NUMINAMATH_CALUDE_enrollment_difference_l2277_227774

/-- Represents the enrollment of a school --/
structure School where
  name : String
  enrollment : Nat

/-- Theorem: The positive difference between the maximum and minimum enrollments is 700 --/
theorem enrollment_difference (schools : List School) 
    (h1 : schools = [
      ⟨"Varsity", 1150⟩, 
      ⟨"Northwest", 1530⟩, 
      ⟨"Central", 1850⟩, 
      ⟨"Greenbriar", 1680⟩, 
      ⟨"Riverside", 1320⟩
    ]) : 
    (List.maximum (schools.map School.enrollment)).getD 0 - 
    (List.minimum (schools.map School.enrollment)).getD 0 = 700 := by
  sorry


end NUMINAMATH_CALUDE_enrollment_difference_l2277_227774


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2277_227799

/-- Given a shopkeeper selling cloth with the following conditions:
  * The shopkeeper sells 200 metres of cloth
  * The selling price is Rs. 12000
  * The shopkeeper incurs a loss of Rs. 6 per metre
  Prove that the cost price for one metre of cloth is Rs. 66 -/
theorem cloth_cost_price 
  (total_metres : ℕ) 
  (selling_price : ℕ) 
  (loss_per_metre : ℕ) 
  (h1 : total_metres = 200)
  (h2 : selling_price = 12000)
  (h3 : loss_per_metre = 6) :
  (selling_price + total_metres * loss_per_metre) / total_metres = 66 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2277_227799


namespace NUMINAMATH_CALUDE_ice_cream_cones_l2277_227734

theorem ice_cream_cones (cost_per_cone total_spent : ℕ) (h1 : cost_per_cone = 99) (h2 : total_spent = 198) :
  total_spent / cost_per_cone = 2 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_cones_l2277_227734


namespace NUMINAMATH_CALUDE_sum_of_roots_l2277_227777

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 15*a^2 + 20*a - 50 = 0)
  (hb : 8*b^3 - 60*b^2 - 290*b + 2575 = 0) : 
  a + b = 15/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2277_227777


namespace NUMINAMATH_CALUDE_triangle_side_length_l2277_227714

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Given conditions
  a = Real.sqrt 3 →
  2 * (Real.cos ((A + C) / 2))^2 = (Real.sqrt 2 - 1) * Real.cos B →
  A = π / 3 →
  -- Conclusion
  c = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2277_227714


namespace NUMINAMATH_CALUDE_kolya_can_break_rods_to_form_triangles_l2277_227742

/-- Represents a rod broken into three parts -/
structure BrokenRod :=
  (part1 : ℝ)
  (part2 : ℝ)
  (part3 : ℝ)
  (sum_to_one : part1 + part2 + part3 = 1)
  (all_positive : part1 > 0 ∧ part2 > 0 ∧ part3 > 0)

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Checks if it's possible to form three triangles from three broken rods -/
def can_form_three_triangles (rod1 rod2 rod3 : BrokenRod) : Prop :=
  ∃ (perm1 perm2 perm3 : Fin 3 → Fin 3),
    can_form_triangle (rod1.part1) (rod2.part1) (rod3.part1) ∧
    can_form_triangle (rod1.part2) (rod2.part2) (rod3.part2) ∧
    can_form_triangle (rod1.part3) (rod2.part3) (rod3.part3)

/-- The main theorem stating that Kolya can break the rods to always form three triangles -/
theorem kolya_can_break_rods_to_form_triangles :
  ∃ (kolya_rod1 kolya_rod2 : BrokenRod),
    ∀ (vasya_rod : BrokenRod),
      can_form_three_triangles kolya_rod1 vasya_rod kolya_rod2 :=
sorry

end NUMINAMATH_CALUDE_kolya_can_break_rods_to_form_triangles_l2277_227742


namespace NUMINAMATH_CALUDE_divisible_by_six_l2277_227709

theorem divisible_by_six (x : ℤ) : 
  (∃ k : ℤ, x^2 + 5*x - 12 = 6*k) ↔ (∃ t : ℤ, x = 3*t ∨ x = 3*t + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_six_l2277_227709


namespace NUMINAMATH_CALUDE_solve_equation_l2277_227743

theorem solve_equation (x y : ℝ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2277_227743


namespace NUMINAMATH_CALUDE_existence_of_m_n_l2277_227776

theorem existence_of_m_n (h k : ℕ+) (ε : ℝ) (hε : ε > 0) :
  ∃ (m n : ℕ+), ε < |h * Real.sqrt m - k * Real.sqrt n| ∧ |h * Real.sqrt m - k * Real.sqrt n| < 2 * ε :=
sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l2277_227776


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_three_l2277_227740

theorem sum_of_fractions_geq_three (a b c : ℝ) (h : a * b * c = 1) :
  (1 + a + a * b) / (1 + b + a * b) +
  (1 + b + b * c) / (1 + c + b * c) +
  (1 + c + a * c) / (1 + a + a * c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_three_l2277_227740


namespace NUMINAMATH_CALUDE_quilt_shaded_area_is_40_percent_l2277_227759

/-- Represents a square quilt with shaded areas -/
structure Quilt where
  total_squares : ℕ
  fully_shaded : ℕ
  half_shaded : ℕ
  quarter_shaded : ℕ

/-- Calculates the percentage of shaded area in the quilt -/
def shaded_percentage (q : Quilt) : ℚ :=
  let shaded_area := q.fully_shaded + q.half_shaded / 2 + q.quarter_shaded / 2
  (shaded_area / q.total_squares) * 100

/-- Theorem stating that the given quilt has 40% shaded area -/
theorem quilt_shaded_area_is_40_percent :
  let q := Quilt.mk 25 4 8 4
  shaded_percentage q = 40 := by sorry

end NUMINAMATH_CALUDE_quilt_shaded_area_is_40_percent_l2277_227759


namespace NUMINAMATH_CALUDE_inequality_solution_l2277_227713

theorem inequality_solution (a b : ℝ) :
  (∀ x, b - a * x > 0 ↔ 
    ((a > 0 ∧ x < b / a) ∨ 
     (a < 0 ∧ x > b / a) ∨ 
     (a = 0 ∧ False))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2277_227713


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2277_227793

theorem cubic_equation_roots : ∃ (z : ℂ), z^3 + z^2 - z = 7 + 7*I :=
by
  -- Prove that 4 + i and -3 - i are roots of the equation
  have h1 : (4 + I)^3 + (4 + I)^2 - (4 + I) = 7 + 7*I := by sorry
  have h2 : (-3 - I)^3 + (-3 - I)^2 - (-3 - I) = 7 + 7*I := by sorry
  
  -- Show that at least one of these roots satisfies the equation
  exact ⟨4 + I, h1⟩

-- Note: This theorem only proves the existence of one root,
-- but we know there are at least two roots satisfying the equation.

end NUMINAMATH_CALUDE_cubic_equation_roots_l2277_227793


namespace NUMINAMATH_CALUDE_sheridan_fish_problem_l2277_227773

/-- The number of fish Mrs. Sheridan's sister gave her -/
def fish_given (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem sheridan_fish_problem : fish_given 22 69 = 47 := by sorry

end NUMINAMATH_CALUDE_sheridan_fish_problem_l2277_227773


namespace NUMINAMATH_CALUDE_hexagon_walk_distance_l2277_227762

def regular_hexagon_side_length : ℝ := 3
def walk_distance : ℝ := 10

theorem hexagon_walk_distance (start_point end_point : ℝ × ℝ) : 
  start_point = (0, 0) →
  end_point = (0.5, -Real.sqrt 3 / 2) →
  Real.sqrt ((end_point.1 - start_point.1)^2 + (end_point.2 - start_point.2)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_walk_distance_l2277_227762


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l2277_227749

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A pentagon is a polygon with 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The sum of the interior angles of a pentagon is 540° -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon_sides = 540 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l2277_227749


namespace NUMINAMATH_CALUDE_lawn_mowing_time_l2277_227765

/-- Time required to mow a rectangular lawn -/
theorem lawn_mowing_time : 
  ∀ (length width swath_width overlap speed : ℝ),
  length = 90 →
  width = 150 →
  swath_width = 28 / 12 →
  overlap = 4 / 12 →
  speed = 5000 →
  (width / (swath_width - overlap) * length) / speed = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_time_l2277_227765


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l2277_227724

theorem function_not_in_first_quadrant
  (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : b < -1) :
  ∀ x y : ℝ, y = a^x + b → ¬(x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l2277_227724


namespace NUMINAMATH_CALUDE_cauchy_not_dense_implies_linear_l2277_227780

/-- A function satisfying the Cauchy functional equation -/
def CauchyFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- The graph of a function is not dense in the plane -/
def NotDenseGraph (f : ℝ → ℝ) : Prop :=
  ∃ U : Set (ℝ × ℝ), IsOpen U ∧ U.Nonempty ∧ ∀ x : ℝ, (x, f x) ∉ U

/-- A function is linear -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

theorem cauchy_not_dense_implies_linear (f : ℝ → ℝ) 
  (h_cauchy : CauchyFunction f) (h_not_dense : NotDenseGraph f) : 
  LinearFunction f := by
  sorry

end NUMINAMATH_CALUDE_cauchy_not_dense_implies_linear_l2277_227780


namespace NUMINAMATH_CALUDE_sandy_nickels_theorem_sandy_specific_case_l2277_227761

/-- The number of nickels Sandy has after her dad borrows some -/
def nickels_remaining (initial_nickels borrowed_nickels : ℕ) : ℕ :=
  initial_nickels - borrowed_nickels

/-- Theorem stating that Sandy's remaining nickels is the difference between initial and borrowed -/
theorem sandy_nickels_theorem (initial_nickels borrowed_nickels : ℕ) 
  (h : borrowed_nickels ≤ initial_nickels) :
  nickels_remaining initial_nickels borrowed_nickels = initial_nickels - borrowed_nickels :=
by
  sorry

/-- Sandy's specific case -/
theorem sandy_specific_case :
  nickels_remaining 31 20 = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_nickels_theorem_sandy_specific_case_l2277_227761


namespace NUMINAMATH_CALUDE_manager_percentage_reduction_l2277_227719

/-- Calculates the percentage of managers after some leave the room. -/
def target_percentage (total_employees : ℕ) (initial_percentage : ℚ) (managers_leaving : ℚ) : ℚ :=
  let initial_managers : ℚ := (initial_percentage / 100) * total_employees
  let remaining_managers : ℚ := initial_managers - managers_leaving
  (remaining_managers / total_employees) * 100

/-- The target percentage of managers is approximately 49% -/
theorem manager_percentage_reduction :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (target_percentage 100 99 49.99999999999996 - 49) < ε :=
sorry

end NUMINAMATH_CALUDE_manager_percentage_reduction_l2277_227719


namespace NUMINAMATH_CALUDE_trapezoid_median_length_l2277_227757

/-- Given a trapezoid and an equilateral triangle with specific properties,
    prove that the median of the trapezoid has length 24. -/
theorem trapezoid_median_length
  (trapezoid_area : ℝ) 
  (triangle_area : ℝ) 
  (trapezoid_height : ℝ) 
  (triangle_height : ℝ) 
  (h1 : trapezoid_area = 3 * triangle_area)
  (h2 : trapezoid_height = 8 * Real.sqrt 3)
  (h3 : triangle_height = 8 * Real.sqrt 3) :
  trapezoid_area / trapezoid_height = 24 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_median_length_l2277_227757


namespace NUMINAMATH_CALUDE_gcf_of_32_and_12_l2277_227700

theorem gcf_of_32_and_12 (n : ℕ) (h1 : n = 32) (h2 : Nat.lcm n 12 = 48) :
  Nat.gcd n 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_32_and_12_l2277_227700


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l2277_227726

/-- The difference in visitors between the current day and the sum of the previous two days -/
def visitor_difference (current_day : ℕ) (previous_day : ℕ) (two_days_ago : ℕ) : ℤ :=
  (current_day : ℤ) - (previous_day + two_days_ago : ℤ)

/-- Theorem stating the visitor difference for the given numbers -/
theorem buckingham_palace_visitors :
  visitor_difference 1321 890 765 = -334 := by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l2277_227726


namespace NUMINAMATH_CALUDE_smallest_whole_number_gt_100_odd_factors_l2277_227779

theorem smallest_whole_number_gt_100_odd_factors : ∀ n : ℕ, n > 100 → (∃ k : ℕ, n = k^2) → ∀ m : ℕ, m > 100 → (∃ j : ℕ, m = j^2) → n ≤ m → n = 121 := by
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_gt_100_odd_factors_l2277_227779


namespace NUMINAMATH_CALUDE_smallest_number_of_nuts_l2277_227753

theorem smallest_number_of_nuts (N : ℕ) : N = 320 ↔ 
  N > 0 ∧
  N % 11 = 1 ∧
  N % 13 = 8 ∧
  N % 17 = 3 ∧
  N > 41 ∧
  (∀ M : ℕ, M > 0 ∧ M % 11 = 1 ∧ M % 13 = 8 ∧ M % 17 = 3 ∧ M > 41 → N ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_nuts_l2277_227753


namespace NUMINAMATH_CALUDE_statement_D_no_related_factor_l2277_227790

-- Define a type for statements
inductive Statement
| A : Statement  -- A timely snow promises a good harvest
| B : Statement  -- If the upper beam is not straight, the lower beam will be crooked
| C : Statement  -- Smoking is harmful to health
| D : Statement  -- Magpies signify joy, crows signify mourning

-- Define what it means for a statement to have a related factor
def has_related_factor (s : Statement) : Prop :=
  ∃ (x y : Prop), (x → y) ∧ (s = Statement.A ∨ s = Statement.B ∨ s = Statement.C)

-- Theorem: Statement D does not have a related factor
theorem statement_D_no_related_factor :
  ¬ has_related_factor Statement.D :=
by
  sorry


end NUMINAMATH_CALUDE_statement_D_no_related_factor_l2277_227790


namespace NUMINAMATH_CALUDE_inverse_proportionality_l2277_227791

/-- Two real numbers are inversely proportional if their product is constant. -/
theorem inverse_proportionality (x y k : ℝ) (h : x * y = k) :
  ∃ (c : ℝ), ∀ (x' y' : ℝ), x' * y' = k → y' = c / x' :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l2277_227791
