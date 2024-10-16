import Mathlib

namespace NUMINAMATH_CALUDE_square_inequality_l114_11460

theorem square_inequality (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l114_11460


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l114_11448

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l114_11448


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l114_11423

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 + a 7 = 2 * Real.pi →
  a 6 * (a 4 + 2 * a 6 + a 8) = 4 * Real.pi^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l114_11423


namespace NUMINAMATH_CALUDE_water_drinkers_l114_11419

theorem water_drinkers (total : ℕ) (fruit_juice : ℕ) (h1 : fruit_juice = 140) 
  (h2 : (fruit_juice : ℚ) / total = 7 / 10) : 
  (total - fruit_juice : ℚ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_drinkers_l114_11419


namespace NUMINAMATH_CALUDE_cosine_amplitude_l114_11494

theorem cosine_amplitude (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∀ x, ∃ y, y = a * Real.cos (b * x + c) + d) →
  (∃ x1, a * Real.cos (b * x1 + c) + d = 5) →
  (∃ x2, a * Real.cos (b * x2 + c) + d = -3) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l114_11494


namespace NUMINAMATH_CALUDE_divide_angle_19_degrees_l114_11401

theorem divide_angle_19_degrees (angle : ℝ) (n : ℕ) : 
  angle = 19 ∧ n = 19 → (angle / n : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_divide_angle_19_degrees_l114_11401


namespace NUMINAMATH_CALUDE_triangle_area_l114_11459

theorem triangle_area (A B C : ℝ × ℝ) : 
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let angle_B := Real.arccos ((AC^2 + BC^2 - (A.1 - B.1)^2 - (A.2 - B.2)^2) / (2 * AC * BC))
  AC = Real.sqrt 7 ∧ BC = 2 ∧ angle_B = π/3 →
  (1/2) * AC * BC * Real.sin angle_B = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l114_11459


namespace NUMINAMATH_CALUDE_smallest_with_ten_factors_l114_11417

/-- The number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly ten distinct positive factors -/
def has_ten_factors (n : ℕ+) : Prop := num_factors n = 10

theorem smallest_with_ten_factors :
  ∃ (m : ℕ+), has_ten_factors m ∧ ∀ (k : ℕ+), has_ten_factors k → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_smallest_with_ten_factors_l114_11417


namespace NUMINAMATH_CALUDE_triangle_existence_l114_11496

/-- Given three angles and an area q^2, prove the existence of a triangle with these properties -/
theorem triangle_existence (α β γ : Real) (q : Real) 
  (angle_sum : α + β + γ = Real.pi)
  (positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ)
  (positive_area : 0 < q) :
  ∃ (a b c : Real),
    0 < a ∧ 0 < b ∧ 0 < c ∧
    (a * b * Real.sin γ) / 2 = q^2 ∧
    Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) = α ∧
    Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) = β ∧
    Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) = γ :=
by sorry


end NUMINAMATH_CALUDE_triangle_existence_l114_11496


namespace NUMINAMATH_CALUDE_adults_trekking_l114_11412

/-- Represents the trekking group and meal information -/
structure TrekkingGroup where
  total_children : ℕ
  meal_adults : ℕ
  meal_children : ℕ
  adults_eaten : ℕ
  remaining_children : ℕ

/-- Theorem stating the number of adults who went trekking -/
theorem adults_trekking (group : TrekkingGroup) 
  (h1 : group.total_children = 70)
  (h2 : group.meal_adults = 70)
  (h3 : group.meal_children = 90)
  (h4 : group.adults_eaten = 42)
  (h5 : group.remaining_children = 36) :
  ∃ (adults_trekking : ℕ), adults_trekking = 70 := by
  sorry


end NUMINAMATH_CALUDE_adults_trekking_l114_11412


namespace NUMINAMATH_CALUDE_stratum_c_sample_size_l114_11404

/-- Calculates the sample size for a stratum in stratified sampling -/
def stratumSampleSize (stratumSize : ℕ) (totalPopulation : ℕ) (totalSampleSize : ℕ) : ℕ :=
  (stratumSize * totalSampleSize) / totalPopulation

theorem stratum_c_sample_size :
  let stratum_a_size : ℕ := 400
  let stratum_b_size : ℕ := 800
  let stratum_c_size : ℕ := 600
  let total_population : ℕ := stratum_a_size + stratum_b_size + stratum_c_size
  let total_sample_size : ℕ := 90
  stratumSampleSize stratum_c_size total_population total_sample_size = 30 := by
  sorry

#eval stratumSampleSize 600 1800 90

end NUMINAMATH_CALUDE_stratum_c_sample_size_l114_11404


namespace NUMINAMATH_CALUDE_scientific_notation_of_12000_l114_11456

theorem scientific_notation_of_12000 :
  (12000 : ℝ) = 1.2 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_12000_l114_11456


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l114_11443

/-- The height of a tree that triples its height every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height every year and reaches 81 feet after 4 years
    will be 9 feet tall after 2 years -/
theorem tree_height_after_two_years
  (h : ∃ (initial_height : ℝ), tree_height initial_height 4 = 81) :
  ∃ (initial_height : ℝ), tree_height initial_height 2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l114_11443


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l114_11476

theorem arithmetic_calculation : 72 / (6 / 2) * 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l114_11476


namespace NUMINAMATH_CALUDE_student_calculation_difference_l114_11426

/-- Proves that dividing a number by 4/5 instead of multiplying it by 4/5 results in a specific difference -/
theorem student_calculation_difference (number : ℝ) (h : number = 40.000000000000014) :
  (number / (4/5)) - (number * (4/5)) = 18.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_difference_l114_11426


namespace NUMINAMATH_CALUDE_smallest_number_l114_11492

def yoongi_number : ℕ := 4
def jungkook_number : ℕ := 6 + 3
def yuna_number : ℕ := 5

theorem smallest_number : 
  yoongi_number ≤ jungkook_number ∧ yoongi_number ≤ yuna_number :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l114_11492


namespace NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l114_11432

theorem range_of_a_for_false_proposition :
  (∀ x ∈ Set.Icc 0 1, 2 * x + a ≥ 0) ↔ a > 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l114_11432


namespace NUMINAMATH_CALUDE_solution_set_is_open_ray_l114_11444

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f x > 2 * x + 4}

theorem solution_set_is_open_ray
  (f : ℝ → ℝ)
  (h1 : Differentiable ℝ f)
  (h2 : ∀ x, deriv f x > 2)
  (h3 : f (-1) = 2) :
  solution_set f = Set.Ioi (-1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_ray_l114_11444


namespace NUMINAMATH_CALUDE_negation_equivalence_l114_11438

theorem negation_equivalence :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l114_11438


namespace NUMINAMATH_CALUDE_product_of_real_and_imaginary_parts_l114_11482

theorem product_of_real_and_imaginary_parts : ∃ (z : ℂ), z = (2 + Complex.I) * Complex.I ∧ (z.re * z.im = -2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_and_imaginary_parts_l114_11482


namespace NUMINAMATH_CALUDE_cost_per_box_l114_11497

/-- The cost per box for packaging a fine arts collection --/
theorem cost_per_box (box_length box_width box_height : ℝ)
  (total_volume min_total_cost : ℝ) :
  box_length = 20 ∧ box_width = 20 ∧ box_height = 12 →
  total_volume = 2160000 →
  min_total_cost = 225 →
  (min_total_cost / (total_volume / (box_length * box_width * box_height))) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_box_l114_11497


namespace NUMINAMATH_CALUDE_problem_1_l114_11441

theorem problem_1 (x y : ℝ) : (2*x + y)^2 - 8*(2*x + y) - 9 = 0 → 2*x + y = 9 ∨ 2*x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l114_11441


namespace NUMINAMATH_CALUDE_tin_in_mixed_alloy_tin_amount_in_new_alloy_l114_11495

/-- Amount of tin in a mixture of two alloys -/
theorem tin_in_mixed_alloy (mass_A mass_B : ℝ) 
  (lead_tin_ratio_A : ℝ) (tin_copper_ratio_B : ℝ) : ℝ :=
  let tin_fraction_A := lead_tin_ratio_A / (1 + lead_tin_ratio_A)
  let tin_fraction_B := tin_copper_ratio_B / (1 + tin_copper_ratio_B)
  tin_fraction_A * mass_A + tin_fraction_B * mass_B

/-- The amount of tin in the new alloy is 221.25 kg -/
theorem tin_amount_in_new_alloy : 
  tin_in_mixed_alloy 170 250 (1/3) (3/5) = 221.25 := by
  sorry

end NUMINAMATH_CALUDE_tin_in_mixed_alloy_tin_amount_in_new_alloy_l114_11495


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l114_11449

theorem cylinder_volume_relation (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let volume_C := π * h^2 * r
  let volume_D := π * r^2 * h
  (volume_D = 3 * volume_C) → (volume_D = 9 * π * h^3) := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l114_11449


namespace NUMINAMATH_CALUDE_tangent_sum_product_l114_11405

theorem tangent_sum_product (α β : ℝ) : 
  let γ := Real.arctan (-Real.tan (α + β))
  Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_product_l114_11405


namespace NUMINAMATH_CALUDE_traffic_light_probability_l114_11472

theorem traffic_light_probability : 
  let p_A : ℚ := 25 / 60
  let p_B : ℚ := 35 / 60
  let p_C : ℚ := 45 / 60
  p_A * p_B * p_C = 35 / 192 := by
sorry

end NUMINAMATH_CALUDE_traffic_light_probability_l114_11472


namespace NUMINAMATH_CALUDE_angle_triple_complement_l114_11471

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l114_11471


namespace NUMINAMATH_CALUDE_inequality_solution_set_l114_11477

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, (a * x) / (x - 1) < 1 ↔ (x < b ∨ x > 3)) → 
  ((3 * a) / (3 - 1) = 1) → 
  a - b = -1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l114_11477


namespace NUMINAMATH_CALUDE_gemma_tip_calculation_l114_11429

/-- Calculates the tip given to a delivery person based on the number of pizzas ordered,
    the cost per pizza, the amount paid, and the change received. -/
def calculate_tip (num_pizzas : ℕ) (cost_per_pizza : ℕ) (amount_paid : ℕ) (change : ℕ) : ℕ :=
  amount_paid - change - (num_pizzas * cost_per_pizza)

/-- Proves that given the specified conditions, the tip Gemma gave to the delivery person was $5. -/
theorem gemma_tip_calculation :
  let num_pizzas : ℕ := 4
  let cost_per_pizza : ℕ := 10
  let amount_paid : ℕ := 50
  let change : ℕ := 5
  calculate_tip num_pizzas cost_per_pizza amount_paid change = 5 := by
  sorry

#eval calculate_tip 4 10 50 5

end NUMINAMATH_CALUDE_gemma_tip_calculation_l114_11429


namespace NUMINAMATH_CALUDE_sum_of_integers_l114_11489

theorem sum_of_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4 →
  m + n + p + q = 24 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l114_11489


namespace NUMINAMATH_CALUDE_special_triangle_area_l114_11428

-- Define a right triangle with a 30° angle and hypotenuse of 20 inches
def special_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧  -- Pythagorean theorem for right triangle
  c = 20 ∧  -- Hypotenuse length
  a / c = 1 / 2  -- Sine of 30° angle (opposite / hypotenuse)

-- Theorem statement
theorem special_triangle_area (a b c : ℝ) 
  (h : special_triangle a b c) : a * b / 2 = 50 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_area_l114_11428


namespace NUMINAMATH_CALUDE_quadratic_minimum_l114_11484

theorem quadratic_minimum (x : ℝ) : 
  3 * x^2 - 18 * x + 2023 ≥ 1996 ∧ ∃ y : ℝ, 3 * y^2 - 18 * y + 2023 = 1996 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l114_11484


namespace NUMINAMATH_CALUDE_jogger_train_distance_jogger_train_problem_l114_11488

theorem jogger_train_distance (jogger_speed : Real) (train_speed : Real) 
  (train_length : Real) (passing_time : Real) : Real :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let distance_covered := relative_speed * passing_time
  distance_covered - train_length

theorem jogger_train_problem :
  jogger_train_distance 9 45 120 40 = 280 := by
  sorry

end NUMINAMATH_CALUDE_jogger_train_distance_jogger_train_problem_l114_11488


namespace NUMINAMATH_CALUDE_unique_integer_with_16_divisors_l114_11421

def hasSixteenDivisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 16

def divisorsOrdered (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 16 → (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList.nthLe i sorry <
    (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList.nthLe j sorry

def divisorProperty (n : ℕ) : Prop :=
  let divisors := (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList
  let d₂ := divisors.nthLe 1 sorry
  let d₄ := divisors.nthLe 3 sorry
  let d₅ := divisors.nthLe 4 sorry
  let d₆ := divisors.nthLe 5 sorry
  divisors.nthLe (d₅ - 1) sorry = (d₂ + d₄) * d₆

theorem unique_integer_with_16_divisors :
  ∃! n : ℕ, n > 0 ∧ hasSixteenDivisors n ∧ divisorsOrdered n ∧ divisorProperty n ∧ n = 2002 :=
sorry

end NUMINAMATH_CALUDE_unique_integer_with_16_divisors_l114_11421


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l114_11485

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9/16) = Real.sqrt 137 / 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l114_11485


namespace NUMINAMATH_CALUDE_stone_volume_l114_11466

/-- Represents a rectangular cuboid bowl -/
structure Bowl where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the volume of water in the bowl given its height -/
def water_volume (b : Bowl) (water_height : ℝ) : ℝ :=
  b.width * b.length * water_height

theorem stone_volume (b : Bowl) (initial_water_height final_water_height : ℝ) :
  b.width = 16 →
  b.length = 14 →
  b.height = 9 →
  initial_water_height = 4 →
  final_water_height = 9 →
  water_volume b final_water_height - water_volume b initial_water_height = 1120 :=
by sorry

end NUMINAMATH_CALUDE_stone_volume_l114_11466


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l114_11475

/-- Time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120) 
  (h2 : train_speed_kmh = 60) 
  (h3 : bridge_length = 170) : 
  ∃ (time : ℝ), abs (time - 17.40) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l114_11475


namespace NUMINAMATH_CALUDE_m_equals_two_iff_parallel_l114_11445

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := 2 * x - m * y - 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m - 1) * x - y + 1 = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), l₁ m x y ↔ l₂ m x y

-- Theorem statement
theorem m_equals_two_iff_parallel :
  ∀ m : ℝ, m = 2 ↔ parallel m := by sorry

end NUMINAMATH_CALUDE_m_equals_two_iff_parallel_l114_11445


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l114_11468

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 4 + a 5 + a 6 + a 7 = 450 →
  a 2 + a 8 = 180 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l114_11468


namespace NUMINAMATH_CALUDE_land_conversion_rates_l114_11457

/-- Represents the daily conversion rates and conditions for land conversion --/
structure LandConversion where
  total_area : ℝ
  rate_ratio : ℝ
  time_difference : ℝ
  team_b_rate : ℝ

/-- Theorem stating the correct daily conversion rates given the conditions --/
theorem land_conversion_rates (lc : LandConversion)
  (h1 : lc.total_area = 1500)
  (h2 : lc.rate_ratio = 1.2)
  (h3 : lc.time_difference = 5)
  (h4 : lc.total_area / lc.team_b_rate - lc.time_difference = lc.total_area / (lc.rate_ratio * lc.team_b_rate)) :
  lc.team_b_rate = 50 ∧ lc.rate_ratio * lc.team_b_rate = 60 := by
  sorry

end NUMINAMATH_CALUDE_land_conversion_rates_l114_11457


namespace NUMINAMATH_CALUDE_book_price_adjustment_l114_11474

theorem book_price_adjustment (x : ℝ) :
  (1 + x / 100) * (1 - x / 100) = 0.75 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_price_adjustment_l114_11474


namespace NUMINAMATH_CALUDE_total_roses_l114_11442

/-- The total number of roses in a special n-gon garden -/
def roseCount (n : ℕ) : ℕ :=
  Nat.choose n 4 + Nat.choose (n - 1) 2

/-- Properties of the rose garden -/
structure RoseGarden (n : ℕ) where
  convex : n ≥ 4
  redRoses : Fin n → Unit  -- One red rose at each vertex
  paths : Fin n → Fin n → Unit  -- Path between each pair of vertices
  noTripleIntersection : Unit  -- No three paths intersect at a single point
  whiteRoses : Unit  -- One white/black rose in each region

/-- Theorem: The total number of roses in the garden is given by roseCount -/
theorem total_roses (n : ℕ) (garden : RoseGarden n) : 
  (Fin n → Unit) × Unit → ℕ :=
by sorry

end NUMINAMATH_CALUDE_total_roses_l114_11442


namespace NUMINAMATH_CALUDE_painting_cost_is_474_l114_11487

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a wall -/
def wallArea (d : Dimensions) : ℝ := d.length * d.height

/-- Calculates the area of a rectangular opening -/
def openingArea (d : Dimensions) : ℝ := d.length * d.width

/-- Represents a room with its dimensions and openings -/
structure Room where
  dimensions : Dimensions
  doorDimensions : Dimensions
  largeDoorCount : ℕ
  largeWindowDimensions : Dimensions
  largeWindowCount : ℕ
  smallWindowDimensions : Dimensions
  smallWindowCount : ℕ

/-- Calculates the total wall area of a room -/
def totalWallArea (r : Room) : ℝ :=
  2 * (wallArea r.dimensions + wallArea { length := r.dimensions.width, width := r.dimensions.width, height := r.dimensions.height })

/-- Calculates the total area of openings in a room -/
def totalOpeningArea (r : Room) : ℝ :=
  (r.largeDoorCount : ℝ) * openingArea r.doorDimensions +
  (r.largeWindowCount : ℝ) * openingArea r.largeWindowDimensions +
  (r.smallWindowCount : ℝ) * openingArea r.smallWindowDimensions

/-- Calculates the paintable area of a room -/
def paintableArea (r : Room) : ℝ :=
  totalWallArea r - totalOpeningArea r

/-- Theorem: The cost of painting the given room is Rs. 474 -/
theorem painting_cost_is_474 (r : Room)
  (h1 : r.dimensions = { length := 10, width := 7, height := 5 })
  (h2 : r.doorDimensions = { length := 1, width := 3, height := 3 })
  (h3 : r.largeDoorCount = 2)
  (h4 : r.largeWindowDimensions = { length := 2, width := 1.5, height := 1.5 })
  (h5 : r.largeWindowCount = 1)
  (h6 : r.smallWindowDimensions = { length := 1, width := 1.5, height := 1.5 })
  (h7 : r.smallWindowCount = 2)
  : paintableArea r * 3 = 474 := by
  sorry

end NUMINAMATH_CALUDE_painting_cost_is_474_l114_11487


namespace NUMINAMATH_CALUDE_jessica_cut_two_roses_l114_11440

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that Jessica cut 2 roses -/
theorem jessica_cut_two_roses : roses_cut 15 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_two_roses_l114_11440


namespace NUMINAMATH_CALUDE_refrigerator_savings_l114_11498

/- Define the parameters of the problem -/
def cash_price : ℕ := 8000
def deposit : ℕ := 3000
def num_installments : ℕ := 30
def installment_amount : ℕ := 300

/- Define the total amount paid in installments -/
def total_installments : ℕ := num_installments * installment_amount + deposit

/- Theorem statement -/
theorem refrigerator_savings : total_installments - cash_price = 4000 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_savings_l114_11498


namespace NUMINAMATH_CALUDE_marys_blue_crayons_l114_11424

/-- The number of crayons Mary has initially, gives away, and has left -/
structure Crayons where
  initial_green : ℕ
  initial_blue : ℕ
  given_green : ℕ
  given_blue : ℕ
  remaining : ℕ

/-- Theorem stating the initial number of blue crayons Mary had -/
theorem marys_blue_crayons (c : Crayons)
  (h1 : c.initial_green = 5)
  (h2 : c.given_green = 3)
  (h3 : c.given_blue = 1)
  (h4 : c.remaining = 9)
  (h5 : c.initial_green + c.initial_blue = c.remaining + c.given_green + c.given_blue) :
  c.initial_blue = 8 := by
  sorry

#check marys_blue_crayons

end NUMINAMATH_CALUDE_marys_blue_crayons_l114_11424


namespace NUMINAMATH_CALUDE_divisibility_by_24_l114_11450

theorem divisibility_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  24 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l114_11450


namespace NUMINAMATH_CALUDE_intersecting_line_parameter_range_l114_11447

/-- Given a line segment PQ and a line l, this theorem proves the range of m for which l intersects the extension of PQ. -/
theorem intersecting_line_parameter_range 
  (P : ℝ × ℝ) 
  (Q : ℝ × ℝ) 
  (l : ℝ → ℝ → Prop) 
  (h_P : P = (-1, 1)) 
  (h_Q : Q = (2, 2)) 
  (h_l : ∀ x y, l x y ↔ x + m * y + m = 0) 
  (h_intersect : ∃ x y, l x y ∧ (∃ t : ℝ, (x, y) = (1 - t) • P + t • Q ∧ t ∉ [0, 1])) :
  m ∈ Set.Ioo (-3 : ℝ) (-2/3) :=
sorry

end NUMINAMATH_CALUDE_intersecting_line_parameter_range_l114_11447


namespace NUMINAMATH_CALUDE_base_conversion_2025_to_octal_l114_11455

theorem base_conversion_2025_to_octal :
  (2025 : ℕ) = (3 * 8^3 + 7 * 8^2 + 5 * 8^1 + 1 * 8^0 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_2025_to_octal_l114_11455


namespace NUMINAMATH_CALUDE_evelyns_marbles_l114_11473

/-- The number of marbles Evelyn has in total -/
def total_marbles (initial : ℕ) (from_henry : ℕ) (from_grace : ℕ) (cards : ℕ) (marbles_per_card : ℕ) : ℕ :=
  initial + from_henry + from_grace + cards * marbles_per_card

/-- Theorem stating that Evelyn's total number of marbles is 140 -/
theorem evelyns_marbles :
  total_marbles 95 9 12 6 4 = 140 := by
  sorry

#eval total_marbles 95 9 12 6 4

end NUMINAMATH_CALUDE_evelyns_marbles_l114_11473


namespace NUMINAMATH_CALUDE_bookshop_inventory_bookshop_current_inventory_l114_11400

/-- Calculates the current number of books in a bookshop after a weekend of sales and a new shipment. -/
theorem bookshop_inventory (
  initial_inventory : ℕ
  ) (saturday_in_store : ℕ) (saturday_online : ℕ)
  (sunday_in_store_multiplier : ℕ) (sunday_online_increase : ℕ)
  (new_shipment : ℕ) : ℕ :=
  let sunday_in_store := sunday_in_store_multiplier * saturday_in_store
  let sunday_online := saturday_online + sunday_online_increase
  let total_sold := saturday_in_store + saturday_online + sunday_in_store + sunday_online
  let net_change := new_shipment - total_sold
  initial_inventory + net_change

/-- The bookshop currently has 502 books. -/
theorem bookshop_current_inventory :
  bookshop_inventory 743 37 128 2 34 160 = 502 := by
  sorry

end NUMINAMATH_CALUDE_bookshop_inventory_bookshop_current_inventory_l114_11400


namespace NUMINAMATH_CALUDE_spinner_probability_l114_11467

theorem spinner_probability : ∀ (p_C : ℚ),
  (1 : ℚ) / 5 + (1 : ℚ) / 3 + p_C + p_C + 2 * p_C = 1 →
  p_C = (7 : ℚ) / 60 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l114_11467


namespace NUMINAMATH_CALUDE_same_terminal_side_as_610_degrees_l114_11403

theorem same_terminal_side_as_610_degrees :
  ∀ θ : ℝ, (∃ k : ℤ, θ = k * 360 + 250) ↔ (∃ n : ℤ, θ = n * 360 + 610) :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_as_610_degrees_l114_11403


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_to_15000_l114_11420

theorem last_four_digits_of_5_to_15000 (h : 5^500 ≡ 1 [ZMOD 1250]) :
  5^15000 ≡ 1 [ZMOD 1250] := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_to_15000_l114_11420


namespace NUMINAMATH_CALUDE_percent_y_of_x_l114_11463

theorem percent_y_of_x (x y : ℝ) (h : (1/2) * (x - y) = (1/5) * (x + y)) :
  y = (3/7) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_y_of_x_l114_11463


namespace NUMINAMATH_CALUDE_total_worth_calculation_l114_11499

/-- Calculates the total worth of purchases given tax information and cost of tax-free items -/
def total_worth (tax_rate : ℚ) (sales_tax : ℚ) (tax_free_cost : ℚ) : ℚ :=
  let taxable_cost := sales_tax / tax_rate
  taxable_cost + tax_free_cost

/-- Theorem stating that given the specific tax information and tax-free item cost, 
    the total worth of purchases is 24.7 rupees -/
theorem total_worth_calculation :
  total_worth (1/10) (3/10) (217/10) = 247/10 := by
  sorry

end NUMINAMATH_CALUDE_total_worth_calculation_l114_11499


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l114_11439

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  ⟨p.x, -p.y⟩

theorem reflection_across_x_axis :
  let P : Point := ⟨-3, 2⟩
  reflect_x P = ⟨-3, -2⟩ := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l114_11439


namespace NUMINAMATH_CALUDE_find_A_in_terms_of_B_and_C_l114_11481

/-- Given two functions f and g, and constants A, B, and C, prove that A can be expressed in terms of B and C. -/
theorem find_A_in_terms_of_B_and_C
  (f g : ℝ → ℝ)
  (A B C : ℝ)
  (h₁ : ∀ x, f x = A * x^2 - 3 * B * C)
  (h₂ : ∀ x, g x = C * x^2)
  (h₃ : B ≠ 0)
  (h₄ : C ≠ 0)
  (h₅ : f (g 2) = A - 3 * C) :
  A = (3 * C * (B - 1)) / (16 * C^2 - 1) := by
sorry


end NUMINAMATH_CALUDE_find_A_in_terms_of_B_and_C_l114_11481


namespace NUMINAMATH_CALUDE_eighteen_team_tournament_games_l114_11493

/-- Calculates the number of games in a knockout tournament with byes -/
def knockout_tournament_games (total_teams : ℕ) (bye_teams : ℕ) : ℕ :=
  total_teams - 1

/-- Theorem: A knockout tournament with 18 teams and 2 byes has 17 games -/
theorem eighteen_team_tournament_games :
  knockout_tournament_games 18 2 = 17 := by
  sorry

#eval knockout_tournament_games 18 2

end NUMINAMATH_CALUDE_eighteen_team_tournament_games_l114_11493


namespace NUMINAMATH_CALUDE_max_cubic_sum_under_constraint_l114_11483

theorem max_cubic_sum_under_constraint (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + a + b + c + d = 8) :
  a^3 + b^3 + c^3 + d^3 ≤ 15.625 := by
  sorry

end NUMINAMATH_CALUDE_max_cubic_sum_under_constraint_l114_11483


namespace NUMINAMATH_CALUDE_length_breadth_difference_l114_11427

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Theorem stating the difference between length and breadth of the plot. -/
theorem length_breadth_difference (plot : RectangularPlot)
  (h1 : plot.length = 75)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : plot.total_fencing_cost = (2 * plot.length + 2 * plot.breadth) * plot.fencing_cost_per_meter) :
  plot.length - plot.breadth = 50 := by
  sorry

#check length_breadth_difference

end NUMINAMATH_CALUDE_length_breadth_difference_l114_11427


namespace NUMINAMATH_CALUDE_prime_squares_end_in_nine_l114_11409

theorem prime_squares_end_in_nine :
  ∀ p q : ℕ,
  Prime p → Prime q →
  (p * p + q * q) % 10 = 9 →
  ((p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2)) := by
sorry

end NUMINAMATH_CALUDE_prime_squares_end_in_nine_l114_11409


namespace NUMINAMATH_CALUDE_safe_menu_fraction_l114_11479

theorem safe_menu_fraction (total_dishes : ℕ) (vegetarian_dishes : ℕ) (gluten_free_vegetarian : ℕ) :
  vegetarian_dishes = total_dishes / 3 →
  gluten_free_vegetarian = vegetarian_dishes - 5 →
  (gluten_free_vegetarian : ℚ) / total_dishes = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_safe_menu_fraction_l114_11479


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l114_11406

theorem inverse_as_linear_combination (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : N = ![![3, 0], ![2, -4]]) : 
  ∃ (c d : ℝ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ 
  c = (1 : ℝ) / 12 ∧ d = (1 : ℝ) / 12 := by
sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l114_11406


namespace NUMINAMATH_CALUDE_second_group_size_l114_11452

/-- Represents the number of man-days required to complete the work -/
def totalManDays : ℕ := 18 * 20

/-- Proves that 12 men can complete the work in 30 days, given that 18 men can complete it in 20 days -/
theorem second_group_size (days : ℕ) (h : days = 30) : 
  (totalManDays / days : ℕ) = 12 := by
  sorry

#check second_group_size

end NUMINAMATH_CALUDE_second_group_size_l114_11452


namespace NUMINAMATH_CALUDE_rectangular_to_cylindrical_l114_11416

theorem rectangular_to_cylindrical :
  let x : ℝ := 3
  let y : ℝ := -3 * Real.sqrt 3
  let z : ℝ := 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 5 * π / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 6 ∧
  θ = 5 * π / 3 ∧
  z = 2 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ := by
sorry

end NUMINAMATH_CALUDE_rectangular_to_cylindrical_l114_11416


namespace NUMINAMATH_CALUDE_apron_sewing_ratio_l114_11414

/-- Prove that the ratio of aprons sewn today to aprons sewn before today is 3:1 -/
theorem apron_sewing_ratio :
  let total_aprons : ℕ := 150
  let aprons_before : ℕ := 13
  let aprons_tomorrow : ℕ := 49
  let aprons_remaining : ℕ := 2 * aprons_tomorrow
  let aprons_sewn_before_tomorrow : ℕ := total_aprons - aprons_remaining
  let aprons_today : ℕ := aprons_sewn_before_tomorrow - aprons_before
  ∃ (n : ℕ), n > 0 ∧ aprons_today = 3 * n ∧ aprons_before = n :=
by
  sorry

end NUMINAMATH_CALUDE_apron_sewing_ratio_l114_11414


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_union_equal_B_iff_m_in_range_l114_11491

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

-- Theorem 1
theorem intersection_when_m_is_one :
  A 1 ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2
theorem union_equal_B_iff_m_in_range (m : ℝ) :
  A m ∪ B = B ↔ m ≥ 3 ∨ m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_union_equal_B_iff_m_in_range_l114_11491


namespace NUMINAMATH_CALUDE_smallest_7digit_binary_proof_l114_11402

/-- The smallest positive integer with a 7-digit binary representation -/
def smallest_7digit_binary : ℕ := 64

/-- The binary representation of a natural number -/
def binary_representation (n : ℕ) : List Bool :=
  sorry

/-- The length of the binary representation of a natural number -/
def binary_length (n : ℕ) : ℕ :=
  (binary_representation n).length

theorem smallest_7digit_binary_proof :
  (∀ m : ℕ, m < smallest_7digit_binary → binary_length m < 7) ∧
  binary_length smallest_7digit_binary = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_7digit_binary_proof_l114_11402


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l114_11410

theorem quadratic_roots_expression (m n : ℝ) : 
  m^2 + m - 2023 = 0 → n^2 + n - 2023 = 0 → m^2 + 2*m + n = 2022 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l114_11410


namespace NUMINAMATH_CALUDE_apple_juice_distribution_l114_11454

/-- Given a total amount of apple juice and the difference between two people's consumption,
    calculate the amount consumed by the person who drinks more. -/
theorem apple_juice_distribution (total : ℝ) (difference : ℝ) (kyu_yeon_amount : ℝ) : 
  total = 12.4 ∧ difference = 2.6 → kyu_yeon_amount = 7.5 := by
  sorry

#check apple_juice_distribution

end NUMINAMATH_CALUDE_apple_juice_distribution_l114_11454


namespace NUMINAMATH_CALUDE_f_plus_g_is_non_horizontal_line_l114_11470

/-- Represents a parabola in vertex form -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ
  a_nonzero : a ≠ 0

/-- The function resulting from translating the original parabola 7 units right -/
def f (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h + 7)^2 + p.k

/-- The function resulting from reflecting the parabola and translating 7 units left -/
def g (p : Parabola) (x : ℝ) : ℝ :=
  -p.a * (x - p.h - 7)^2 - p.k

/-- The sum of f and g -/
def f_plus_g (p : Parabola) (x : ℝ) : ℝ :=
  f p x + g p x

/-- Theorem stating that f_plus_g is a non-horizontal line -/
theorem f_plus_g_is_non_horizontal_line (p : Parabola) :
  ∃ m b, m ≠ 0 ∧ ∀ x, f_plus_g p x = m * x + b := by
  sorry

end NUMINAMATH_CALUDE_f_plus_g_is_non_horizontal_line_l114_11470


namespace NUMINAMATH_CALUDE_most_frequent_is_mode_l114_11422

/-- The mode of a dataset is the value that appears most frequently. -/
def mode (dataset : Multiset α) [DecidableEq α] : Set α :=
  {x | ∀ y, dataset.count x ≥ dataset.count y}

/-- The most frequent data in a dataset is the mode. -/
theorem most_frequent_is_mode (dataset : Multiset α) [DecidableEq α] :
  ∀ x ∈ mode dataset, ∀ y, dataset.count x ≥ dataset.count y :=
sorry

end NUMINAMATH_CALUDE_most_frequent_is_mode_l114_11422


namespace NUMINAMATH_CALUDE_work_completion_time_l114_11478

theorem work_completion_time (x_total_days y_completion_days : ℕ) 
  (x_work_days : ℕ) (h1 : x_total_days = 20) (h2 : x_work_days = 10) 
  (h3 : y_completion_days = 12) : 
  (x_total_days * y_completion_days) / (y_completion_days - x_work_days) = 24 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l114_11478


namespace NUMINAMATH_CALUDE_four_row_arrangement_has_27_triangles_l114_11435

/-- Represents a triangular arrangement of smaller triangles -/
structure TriangularArrangement where
  rows : ℕ

/-- Counts the number of small triangles in the arrangement -/
def count_small_triangles (arr : TriangularArrangement) : ℕ :=
  (arr.rows * (arr.rows + 1)) / 2

/-- Counts the number of medium triangles (made of 4 small triangles) -/
def count_medium_triangles (arr : TriangularArrangement) : ℕ :=
  if arr.rows ≥ 3 then
    ((arr.rows - 2) * (arr.rows - 1)) / 2
  else
    0

/-- Counts the number of large triangles (made of 9 small triangles) -/
def count_large_triangles (arr : TriangularArrangement) : ℕ :=
  if arr.rows ≥ 4 then
    (arr.rows - 3)
  else
    0

/-- Counts the total number of triangles in the arrangement -/
def total_triangles (arr : TriangularArrangement) : ℕ :=
  count_small_triangles arr + count_medium_triangles arr + count_large_triangles arr

/-- Theorem: In a triangular arrangement with 4 rows, there are 27 triangles in total -/
theorem four_row_arrangement_has_27_triangles :
  ∀ (arr : TriangularArrangement), arr.rows = 4 → total_triangles arr = 27 := by
  sorry

end NUMINAMATH_CALUDE_four_row_arrangement_has_27_triangles_l114_11435


namespace NUMINAMATH_CALUDE_f_properties_l114_11415

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * abs (x + a) - (1/2) * Real.log x

theorem f_properties :
  (∀ x > 0, ∀ a : ℝ,
    (a = 0 → (∀ y > (1/2), f a y > f a x) ∧ (∀ z ∈ Set.Ioo 0 (1/2), f a z < f a x)) ∧
    (a < 0 →
      (a < -2 → ∃ x₁ x₂, x₁ = (-a - Real.sqrt (a^2 - 4)) / 4 ∧
                         x₂ = (-a + Real.sqrt (a^2 - 4)) / 4 ∧
                         (∀ y ≠ x₁, f a y ≥ f a x₁) ∧
                         (∀ y ≠ x₂, f a y ≤ f a x₂)) ∧
      (-2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2 → ∀ y > 0, f a y ≠ f a x) ∧
      (-Real.sqrt 2 / 2 < a ∧ a < 0 →
        ∃ x₃, x₃ = (-a + Real.sqrt (a^2 + 4)) / 4 ∧
               (∀ y ≠ x₃, f a y ≥ f a x₃)))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l114_11415


namespace NUMINAMATH_CALUDE_curve_intersection_property_m_range_l114_11437

/-- The curve C defined by y² = 4x for x > 0 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4*p.1 ∧ p.1 > 0}

/-- The line passing through (m, 0) with slope 1/t -/
def line (m t : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = t*p.2 + m}

/-- The dot product of vectors FA and FB where F is (1, 0) -/
def dot_product (A B : ℝ × ℝ) : ℝ := (A.1 - 1)*(B.1 - 1) + A.2*B.2

theorem curve_intersection_property :
  ∃ (m : ℝ), m > 0 ∧
  ∀ (t : ℝ), ∀ (A B : ℝ × ℝ),
    A ∈ C → B ∈ C → A ∈ line m t → B ∈ line m t → A ≠ B →
    dot_product A B < 0 :=
sorry

theorem m_range (m : ℝ) :
  (∀ (t : ℝ), ∀ (A B : ℝ × ℝ),
    A ∈ C → B ∈ C → A ∈ line m t → B ∈ line m t → A ≠ B →
    dot_product A B < 0) ↔
  3 - 2*Real.sqrt 2 < m ∧ m < 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_curve_intersection_property_m_range_l114_11437


namespace NUMINAMATH_CALUDE_polynomial_factorization_l114_11465

theorem polynomial_factorization (a b x : ℝ) : 
  a + (a+b)*x + (a+2*b)*x^2 + (a+3*b)*x^3 + 3*b*x^4 + 2*b*x^5 + b*x^6 = 
  (1 + x)*(1 + x^2)*(a + b*x + b*x^2 + b*x^3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l114_11465


namespace NUMINAMATH_CALUDE_probability_point_between_B_and_E_l114_11407

/-- Given a line segment AB with points A, B, C, D, and E such that AB = 4AD = 8BE = 2BC,
    the probability that a randomly chosen point on AB lies between B and E is 1/8. -/
theorem probability_point_between_B_and_E (A B C D E : ℝ) : 
  A < D ∧ D < E ∧ E < B ∧ B < C →  -- Points are ordered on the line
  (B - A) = 4 * (D - A) →          -- AB = 4AD
  (B - A) = 8 * (B - E) →          -- AB = 8BE
  (B - A) = 2 * (C - B) →          -- AB = 2BC
  (B - E) / (B - A) = 1 / 8 :=     -- Probability is 1/8
by sorry

end NUMINAMATH_CALUDE_probability_point_between_B_and_E_l114_11407


namespace NUMINAMATH_CALUDE_zephyria_license_plates_l114_11446

/-- The number of letters in the English alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letters in a Zephyrian license plate -/
def num_plate_letters : ℕ := 3

/-- The number of digits in a Zephyrian license plate -/
def num_plate_digits : ℕ := 4

/-- The total number of possible valid license plates in Zephyria -/
def total_license_plates : ℕ := num_letters ^ num_plate_letters * num_digits ^ num_plate_digits

theorem zephyria_license_plates :
  total_license_plates = 175760000 :=
by sorry

end NUMINAMATH_CALUDE_zephyria_license_plates_l114_11446


namespace NUMINAMATH_CALUDE_not_p_false_sufficient_not_necessary_for_p_or_q_true_l114_11469

theorem not_p_false_sufficient_not_necessary_for_p_or_q_true (p q : Prop) :
  (¬¬p → p ∨ q) ∧ ∃ (p q : Prop), (p ∨ q) ∧ ¬(¬¬p) :=
sorry

end NUMINAMATH_CALUDE_not_p_false_sufficient_not_necessary_for_p_or_q_true_l114_11469


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l114_11425

theorem perfect_square_quadratic (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, x^2 + (m + 2) * x + 36 = y^2) →
  m = 10 ∨ m = -14 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l114_11425


namespace NUMINAMATH_CALUDE_slope_angle_45_implies_a_equals_1_l114_11411

theorem slope_angle_45_implies_a_equals_1 (a : ℝ) : 
  (∃ (x y : ℝ), a * x + (2 * a - 3) * y = 0 ∧ 
   Real.tan (45 * π / 180) = -(a / (2 * a - 3))) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_45_implies_a_equals_1_l114_11411


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l114_11431

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 where a, b > 0,
    focal length 10, and point P(3, 4) on one of its asymptotes,
    prove that the standard equation of C is x²/9 - y²/16 = 1 -/
theorem hyperbola_standard_equation 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hf : (2 : ℝ) * Real.sqrt (a^2 + b^2) = 10) 
  (hp : (3 : ℝ)^2 / a^2 - (4 : ℝ)^2 / b^2 = 0) :
  ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 9 - y^2 / 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l114_11431


namespace NUMINAMATH_CALUDE_modulus_of_complex_l114_11458

/-- Given that i is the imaginary unit and z is defined as z = (2+i)/i, prove that |z| = √5 -/
theorem modulus_of_complex (i : ℂ) (z : ℂ) :
  i * i = -1 →
  z = (2 + i) / i →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l114_11458


namespace NUMINAMATH_CALUDE_min_zeros_odd_period_two_l114_11413

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has period 2 if f(x) = f(x+2) for all x ∈ ℝ -/
def HasPeriodTwo (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (x + 2)

/-- The number of zeros of a function f: ℝ → ℝ in an interval [a, b] -/
def NumberOfZeros (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry  -- Definition omitted for brevity

/-- The theorem stating the minimum number of zeros for an odd function with period 2 -/
theorem min_zeros_odd_period_two (f : ℝ → ℝ) (h_odd : IsOdd f) (h_period : HasPeriodTwo f) :
  NumberOfZeros f 0 2009 ≥ 2010 :=
sorry

end NUMINAMATH_CALUDE_min_zeros_odd_period_two_l114_11413


namespace NUMINAMATH_CALUDE_field_dimension_l114_11451

/-- The value of m for a rectangular field with given dimensions and area -/
theorem field_dimension (m : ℝ) : (3*m + 11) * (m - 3) = 80 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_field_dimension_l114_11451


namespace NUMINAMATH_CALUDE_zero_of_f_necessary_not_sufficient_for_decreasing_g_l114_11490

noncomputable def f (m : ℝ) (x : ℝ) := 2^x + m - 1
noncomputable def g (m : ℝ) (x : ℝ) := Real.log x / Real.log m

theorem zero_of_f_necessary_not_sufficient_for_decreasing_g :
  (∀ m : ℝ, (∃ x : ℝ, f m x = 0) → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → g m x₁ > g m x₂)) ∧
  (∃ m : ℝ, (∃ x : ℝ, f m x = 0) ∧ 
    ¬(∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → g m x₁ > g m x₂)) :=
by sorry

end NUMINAMATH_CALUDE_zero_of_f_necessary_not_sufficient_for_decreasing_g_l114_11490


namespace NUMINAMATH_CALUDE_weekend_to_weekday_ratio_is_three_to_one_l114_11461

/-- The number of episodes watched on a weekday -/
def weekday_episodes : ℕ := 8

/-- The total number of episodes watched in a week -/
def total_episodes : ℕ := 88

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The ratio of episodes watched on a weekend day to episodes watched on a weekday -/
def weekend_to_weekday_ratio : ℚ :=
  (total_episodes - weekday_episodes * weekdays) / (weekend_days * weekday_episodes)

theorem weekend_to_weekday_ratio_is_three_to_one :
  weekend_to_weekday_ratio = 3 := by sorry

end NUMINAMATH_CALUDE_weekend_to_weekday_ratio_is_three_to_one_l114_11461


namespace NUMINAMATH_CALUDE_problem_statement_l114_11480

theorem problem_statement (a b : ℝ) (h : a + b - 1 = 0) : 3 * a^2 + 6 * a * b + 3 * b^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l114_11480


namespace NUMINAMATH_CALUDE_child_patients_per_hour_l114_11433

/-- Represents the number of adult patients seen per hour -/
def adults_per_hour : ℕ := 4

/-- Represents the cost of an adult office visit in dollars -/
def adult_visit_cost : ℕ := 50

/-- Represents the cost of a child office visit in dollars -/
def child_visit_cost : ℕ := 25

/-- Represents the total revenue for a typical 8-hour day in dollars -/
def total_daily_revenue : ℕ := 2200

/-- Represents the number of hours in a typical workday -/
def hours_per_day : ℕ := 8

/-- 
Proves that the number of child patients seen per hour is 3, 
given the conditions specified in the problem.
-/
theorem child_patients_per_hour : 
  ∃ (c : ℕ), 
    hours_per_day * (adults_per_hour * adult_visit_cost + c * child_visit_cost) = total_daily_revenue ∧
    c = 3 := by
  sorry

end NUMINAMATH_CALUDE_child_patients_per_hour_l114_11433


namespace NUMINAMATH_CALUDE_cubic_three_monotonic_intervals_l114_11462

/-- A cubic function with a linear term -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x

/-- The derivative of f -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem cubic_three_monotonic_intervals (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f_deriv a x = 0 ∧ f_deriv a y = 0) ↔ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_three_monotonic_intervals_l114_11462


namespace NUMINAMATH_CALUDE_solve_average_weight_l114_11464

def average_weight_problem (weight_16 : ℝ) (weight_all : ℝ) (num_16 : ℕ) (num_8 : ℕ) : Prop :=
  let num_total : ℕ := num_16 + num_8
  let weight_8 : ℝ := (num_total * weight_all - num_16 * weight_16) / num_8
  weight_16 = 50.25 ∧ 
  weight_all = 48.55 ∧ 
  num_16 = 16 ∧ 
  num_8 = 8 ∧ 
  weight_8 = 45.15

theorem solve_average_weight : 
  ∃ (weight_16 weight_all : ℝ) (num_16 num_8 : ℕ), 
    average_weight_problem weight_16 weight_all num_16 num_8 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_average_weight_l114_11464


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l114_11408

theorem complex_magnitude_equation (a : ℝ) : 
  Complex.abs ((1 + Complex.I) / (a * Complex.I)) = Real.sqrt 2 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l114_11408


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l114_11418

theorem factor_implies_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 40 = (x + 5) * k) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l114_11418


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l114_11453

theorem sqrt_sum_equality : Real.sqrt 50 + Real.sqrt 72 = 11 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l114_11453


namespace NUMINAMATH_CALUDE_symmetric_trapezoid_feasibility_l114_11436

/-- Represents a symmetric trapezoid with one parallel side equal to the legs -/
structure SymmetricTrapezoid where
  /-- Length of the legs -/
  a : ℝ
  /-- Distance from the intersection point of the diagonals to one endpoint of the other parallel side -/
  b : ℝ
  /-- Assumption that a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0

/-- Theorem stating the feasibility condition for constructing the symmetric trapezoid -/
theorem symmetric_trapezoid_feasibility (t : SymmetricTrapezoid) :
  (∃ (trapezoid : SymmetricTrapezoid), trapezoid.a = t.a ∧ trapezoid.b = t.b) ↔ 3 * t.b > 2 * t.a := by
  sorry

end NUMINAMATH_CALUDE_symmetric_trapezoid_feasibility_l114_11436


namespace NUMINAMATH_CALUDE_mod_product_equals_one_l114_11434

theorem mod_product_equals_one (m : ℕ) : 
  187 * 973 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_product_equals_one_l114_11434


namespace NUMINAMATH_CALUDE_triangle_perimeter_l114_11486

/-- Given a triangle with inradius 2.5 cm and area 50 cm², its perimeter is 40 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 50 → A = r * (p / 2) → p = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l114_11486


namespace NUMINAMATH_CALUDE_seven_classes_tournament_l114_11430

/-- Calculate the number of matches in a round-robin tournament -/
def numberOfMatches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: For 7 classes in a round-robin tournament, the total number of matches is 21 -/
theorem seven_classes_tournament : numberOfMatches 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_seven_classes_tournament_l114_11430
