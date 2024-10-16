import Mathlib

namespace NUMINAMATH_CALUDE_card_sum_problem_l3374_337468

theorem card_sum_problem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
  sorry

end NUMINAMATH_CALUDE_card_sum_problem_l3374_337468


namespace NUMINAMATH_CALUDE_wheel_distance_l3374_337466

/-- Proves that a wheel rotating 10 times per minute and moving 20 cm per rotation will move 12000 cm in 1 hour -/
theorem wheel_distance (rotations_per_minute : ℕ) (cm_per_rotation : ℕ) (minutes_per_hour : ℕ) :
  rotations_per_minute = 10 →
  cm_per_rotation = 20 →
  minutes_per_hour = 60 →
  rotations_per_minute * minutes_per_hour * cm_per_rotation = 12000 := by
  sorry

#check wheel_distance

end NUMINAMATH_CALUDE_wheel_distance_l3374_337466


namespace NUMINAMATH_CALUDE_tims_lunch_cost_l3374_337433

theorem tims_lunch_cost (tip_percentage : ℝ) (total_spent : ℝ) (lunch_cost : ℝ) : 
  tip_percentage = 0.20 → 
  total_spent = 72.6 → 
  lunch_cost * (1 + tip_percentage) = total_spent → 
  lunch_cost = 60.5 := by
sorry

end NUMINAMATH_CALUDE_tims_lunch_cost_l3374_337433


namespace NUMINAMATH_CALUDE_inverse_variation_proof_l3374_337498

/-- Given that x varies inversely as the square of y, prove that x = 1/9 when y = 6, given that y = 2 when x = 1. -/
theorem inverse_variation_proof (x y : ℝ) (h : ∃ k : ℝ, ∀ x y, x = k / (y ^ 2)) 
  (h1 : ∃ x₀, x₀ = 1 ∧ y = 2) : 
  (y = 6) → (x = 1 / 9) := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_proof_l3374_337498


namespace NUMINAMATH_CALUDE_expression_value_l3374_337437

theorem expression_value (a b c : ℤ) 
  (eq1 : (25 : ℝ) ^ a * 5 ^ (2 * b) = 5 ^ 6)
  (eq2 : (4 : ℝ) ^ b / 4 ^ c = 4) : 
  a ^ 2 + a * b + 3 * c = 6 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3374_337437


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l3374_337447

/-- Given a bag of popping corn with white and blue kernels, calculate the probability
    that a randomly selected kernel that popped was white. -/
theorem popped_kernel_probability (total_kernels : ℝ) (h_total_pos : 0 < total_kernels) : 
  let white_ratio : ℝ := 3/4
  let blue_ratio : ℝ := 1/4
  let white_pop_prob : ℝ := 3/5
  let blue_pop_prob : ℝ := 3/4
  let white_kernels := white_ratio * total_kernels
  let blue_kernels := blue_ratio * total_kernels
  let popped_white := white_pop_prob * white_kernels
  let popped_blue := blue_pop_prob * blue_kernels
  let total_popped := popped_white + popped_blue
  (popped_white / total_popped) = 12/13 :=
by sorry

end NUMINAMATH_CALUDE_popped_kernel_probability_l3374_337447


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3374_337425

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| ≤ 2}
def B : Set ℝ := {x : ℝ | 3*x - 2 ≥ 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3374_337425


namespace NUMINAMATH_CALUDE_min_value_theorem_l3374_337470

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 7) :
  (1 / (1 + a)) + (4 / (2 + b)) ≥ (13 + 4 * Real.sqrt 3) / 14 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 7 ∧
    (1 / (1 + a₀)) + (4 / (2 + b₀)) = (13 + 4 * Real.sqrt 3) / 14 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3374_337470


namespace NUMINAMATH_CALUDE_complex_absolute_value_squared_l3374_337439

theorem complex_absolute_value_squared : 
  (Complex.abs (-3 - (8/5)*Complex.I))^2 = 289/25 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_squared_l3374_337439


namespace NUMINAMATH_CALUDE_value_of_3a_plus_6b_l3374_337458

theorem value_of_3a_plus_6b (a b : ℝ) (h : a + 2*b - 1 = 0) : 3*a + 6*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_3a_plus_6b_l3374_337458


namespace NUMINAMATH_CALUDE_system_solution_l3374_337450

theorem system_solution :
  ∃! (x y : ℝ), (3 * x = 2 * y) ∧ (x - 2 * y = -4) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3374_337450


namespace NUMINAMATH_CALUDE_sufficient_condition_l3374_337475

theorem sufficient_condition (θ P₁ P₂ : Prop) 
  (h1 : P₁ → θ) 
  (h2 : P₂ → P₁) : 
  P₂ → θ := by
sorry

end NUMINAMATH_CALUDE_sufficient_condition_l3374_337475


namespace NUMINAMATH_CALUDE_circumcircle_tangency_l3374_337434

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary functions and relations
variable (circumcircle : Point → Point → Point → Circle)
variable (on_arc : Point → Point → Point → Point → Prop)
variable (incircle_center : Point → Point → Point → Point)
variable (touches : Circle → Circle → Prop)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem circumcircle_tangency
  (A B C D I_A I_B : Point) (k : Circle) :
  k = circumcircle A B C →
  on_arc A B C D →
  I_A = incircle_center A D C →
  I_B = incircle_center B D C →
  touches (circumcircle I_A I_B C) k ↔
    distance A D / distance B D =
    (distance A C + distance C D) / (distance B C + distance C D) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_tangency_l3374_337434


namespace NUMINAMATH_CALUDE_clothing_distribution_l3374_337419

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 47)
  (h2 : first_load = 17)
  (h3 : num_small_loads = 5) :
  (total - first_load) / num_small_loads = 6 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l3374_337419


namespace NUMINAMATH_CALUDE_landscape_length_is_240_l3374_337405

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playgroundArea : ℝ
  totalArea : ℝ

/-- The length of the landscape is 8 times its breadth -/
def lengthIsTotalRule (l : Landscape) : Prop :=
  l.length = 8 * l.breadth

/-- The playground occupies 1/6 of the total landscape area -/
def playgroundRule (l : Landscape) : Prop :=
  l.playgroundArea = l.totalArea / 6

/-- The playground has an area of 1200 square meters -/
def playgroundAreaRule (l : Landscape) : Prop :=
  l.playgroundArea = 1200

/-- The total area of the landscape is the product of its length and breadth -/
def totalAreaRule (l : Landscape) : Prop :=
  l.totalArea = l.length * l.breadth

/-- Theorem: Given the conditions, the length of the landscape is 240 meters -/
theorem landscape_length_is_240 (l : Landscape) 
  (h1 : lengthIsTotalRule l) 
  (h2 : playgroundRule l) 
  (h3 : playgroundAreaRule l) 
  (h4 : totalAreaRule l) : 
  l.length = 240 := by
  sorry

end NUMINAMATH_CALUDE_landscape_length_is_240_l3374_337405


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3374_337442

/-- The coordinates of a point (-1, 2) with respect to the origin in a Cartesian coordinate system are (-1, 2) -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-1, 2)
  P = P :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3374_337442


namespace NUMINAMATH_CALUDE_linear_function_properties_l3374_337467

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem linear_function_properties (k b : ℝ) (hk : k < 0) (hb : b > 0) :
  -- 1. The graph passes through the first, second, and fourth quadrants
  (∃ x y, x > 0 ∧ y > 0 ∧ y = linear_function k b x) ∧
  (∃ x y, x < 0 ∧ y > 0 ∧ y = linear_function k b x) ∧
  (∃ x y, x > 0 ∧ y < 0 ∧ y = linear_function k b x) ∧
  -- 2. y decreases as x increases
  (∀ x₁ x₂, x₁ < x₂ → linear_function k b x₁ > linear_function k b x₂) ∧
  -- 3. The graph intersects the y-axis at the point (0, b)
  (linear_function k b 0 = b) ∧
  -- 4. When x > -b/k, y < 0
  (∀ x, x > -b/k → linear_function k b x < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l3374_337467


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_main_theorem_l3374_337462

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∃ q : ℝ, ∀ n, a (n + 1) = a 1 * q^n :=
sorry

theorem main_theorem (a : ℕ → ℝ) (h1 : geometric_sequence a) 
  (h2 : ∀ n, a n > 0)
  (h3 : 2 * (1/2 * a 3) = 3 * a 1 + 2 * a 2) :
  (a 10 + a 12 + a 15 + a 19 + a 20 + a 23) / 
  (a 8 + a 10 + a 13 + a 17 + a 18 + a 21) = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_main_theorem_l3374_337462


namespace NUMINAMATH_CALUDE_jack_jill_equal_payment_l3374_337423

/-- Represents the pizza order and consumption details --/
structure PizzaOrder where
  totalSlices : ℕ
  baseCost : ℚ
  pepperoniCost : ℚ
  jackPepperoniSlices : ℕ
  jackCheeseSlices : ℕ

/-- Calculates the total cost of the pizza --/
def totalCost (order : PizzaOrder) : ℚ :=
  order.baseCost + order.pepperoniCost

/-- Calculates the cost per slice --/
def costPerSlice (order : PizzaOrder) : ℚ :=
  totalCost order / order.totalSlices

/-- Calculates Jack's payment --/
def jackPayment (order : PizzaOrder) : ℚ :=
  costPerSlice order * (order.jackPepperoniSlices + order.jackCheeseSlices)

/-- Calculates Jill's payment --/
def jillPayment (order : PizzaOrder) : ℚ :=
  costPerSlice order * (order.totalSlices - order.jackPepperoniSlices - order.jackCheeseSlices)

/-- Theorem: Jack and Jill pay the same amount for their share of the pizza --/
theorem jack_jill_equal_payment (order : PizzaOrder)
  (h1 : order.totalSlices = 12)
  (h2 : order.baseCost = 12)
  (h3 : order.pepperoniCost = 3)
  (h4 : order.jackPepperoniSlices = 4)
  (h5 : order.jackCheeseSlices = 2) :
  jackPayment order = jillPayment order := by
  sorry

end NUMINAMATH_CALUDE_jack_jill_equal_payment_l3374_337423


namespace NUMINAMATH_CALUDE_max_value_of_a_l3374_337403

theorem max_value_of_a : 
  (∀ x : ℝ, (x + a)^2 - 16 > 0 ↔ x ≤ -3 ∨ x ≥ 2) → 
  (∀ b : ℝ, (∀ x : ℝ, (x + b)^2 - 16 > 0 ↔ x ≤ -3 ∨ x ≥ 2) → b ≤ 2) ∧
  (∀ x : ℝ, (x + 2)^2 - 16 > 0 ↔ x ≤ -3 ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3374_337403


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_32_l3374_337411

theorem smallest_positive_multiple_of_32 :
  ∀ n : ℕ, n > 0 → 32 * n ≥ 32 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_32_l3374_337411


namespace NUMINAMATH_CALUDE_quadrilaterals_from_circle_points_l3374_337443

/-- The number of points on the circumference of the circle -/
def n : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def k : ℕ := 4

/-- The number of different convex quadrilaterals that can be formed -/
def num_quadrilaterals : ℕ := Nat.choose n k

theorem quadrilaterals_from_circle_points : num_quadrilaterals = 495 := by
  sorry

end NUMINAMATH_CALUDE_quadrilaterals_from_circle_points_l3374_337443


namespace NUMINAMATH_CALUDE_investment_growth_l3374_337415

/-- Calculates the final amount after simple interest --/
def final_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the conditions, the final amount after 6 years is $380 --/
theorem investment_growth (principal : ℝ) (amount_after_2_years : ℝ) :
  principal = 200 →
  amount_after_2_years = 260 →
  final_amount principal ((amount_after_2_years - principal) / (principal * 2)) 6 = 380 :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l3374_337415


namespace NUMINAMATH_CALUDE_tan_product_l3374_337428

theorem tan_product (α β : Real) 
  (h1 : Real.cos (α + β) = 1/5)
  (h2 : Real.cos (α - β) = 3/5) : 
  Real.tan α * Real.tan β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_l3374_337428


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l3374_337407

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l3374_337407


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_for_f_plus_g_geq_3_l3374_337456

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Theorem for the first part of the problem
theorem solution_set_f_leq_6 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_f_plus_g_geq_3 :
  {a : ℝ | ∀ x, f a x + g x ≥ 3} = {a : ℝ | a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_for_f_plus_g_geq_3_l3374_337456


namespace NUMINAMATH_CALUDE_system_solution_existence_l3374_337436

theorem system_solution_existence (a : ℝ) : 
  (∃ x y : ℝ, y = (x + |x|) / x ∧ (x - a)^2 = y + a) ↔ 
  (a > -1 ∧ a ≤ 0) ∨ (a > 0 ∧ a < 1) ∨ (a ≥ 1 ∧ a ≤ 2) ∨ (a > 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l3374_337436


namespace NUMINAMATH_CALUDE_crank_slider_motion_l3374_337430

/-- Crank-slider mechanism parameters -/
structure CrankSlider where
  OA : ℝ
  AB : ℝ
  ω : ℝ
  AM : ℝ

/-- Position and velocity of point M -/
structure PointM where
  x : ℝ → ℝ
  y : ℝ → ℝ
  vx : ℝ → ℝ
  vy : ℝ → ℝ

/-- Theorem stating the equations of motion for point M -/
theorem crank_slider_motion (cs : CrankSlider) (t : ℝ) : 
  cs.OA = 90 ∧ cs.AB = 90 ∧ cs.ω = 10 ∧ cs.AM = 60 →
  ∃ (pm : PointM),
    pm.x t = 90 * Real.cos (10 * t) - 60 * Real.sin (10 * t) ∧
    pm.y t = 90 * Real.sin (10 * t) - 60 * Real.cos (10 * t) ∧
    pm.vx t = -900 * Real.sin (10 * t) - 600 * Real.cos (10 * t) ∧
    pm.vy t = 900 * Real.cos (10 * t) + 600 * Real.sin (10 * t) := by
  sorry


end NUMINAMATH_CALUDE_crank_slider_motion_l3374_337430


namespace NUMINAMATH_CALUDE_belinda_age_l3374_337454

theorem belinda_age (tony_age belinda_age : ℕ) : 
  tony_age + belinda_age = 56 →
  belinda_age = 2 * tony_age + 8 →
  tony_age = 16 →
  belinda_age = 40 := by
sorry

end NUMINAMATH_CALUDE_belinda_age_l3374_337454


namespace NUMINAMATH_CALUDE_distance_between_trees_l3374_337413

theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 180)
  (h2 : num_trees = 11)
  (h3 : num_trees ≥ 2) :
  let distance := yard_length / (num_trees - 1)
  distance = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3374_337413


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3374_337492

/-- The polynomial being divided -/
def f (x : ℝ) : ℝ := x^5 + 3*x^3 + x^2 + 4

/-- The divisor -/
def g (x : ℝ) : ℝ := (x - 2)^2

/-- The remainder -/
def r (x : ℝ) : ℝ := 35*x + 48

/-- The quotient -/
def q (x : ℝ) : ℝ := sorry

theorem polynomial_division_theorem :
  ∀ x : ℝ, f x = g x * q x + r x := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3374_337492


namespace NUMINAMATH_CALUDE_reciprocal_of_i_l3374_337473

theorem reciprocal_of_i : Complex.I⁻¹ = -Complex.I := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_i_l3374_337473


namespace NUMINAMATH_CALUDE_smallest_opposite_l3374_337446

theorem smallest_opposite (a b c d : ℝ) (ha : a = -1) (hb : b = 0) (hc : c = Real.sqrt 5) (hd : d = -1/3) :
  min (-a) (min (-b) (min (-c) (-d))) = -c :=
by sorry

end NUMINAMATH_CALUDE_smallest_opposite_l3374_337446


namespace NUMINAMATH_CALUDE_rover_spots_l3374_337412

theorem rover_spots (granger cisco rover : ℕ) : 
  granger = 5 * cisco →
  cisco = rover / 2 - 5 →
  granger + cisco = 108 →
  rover = 46 := by
sorry

end NUMINAMATH_CALUDE_rover_spots_l3374_337412


namespace NUMINAMATH_CALUDE_hartley_puppies_count_l3374_337444

/-- The number of puppies Hartley has -/
def num_puppies : ℕ := 4

/-- The weight of each puppy in kilograms -/
def puppy_weight : ℝ := 7.5

/-- The number of cats -/
def num_cats : ℕ := 14

/-- The weight of each cat in kilograms -/
def cat_weight : ℝ := 2.5

/-- The difference in total weight between cats and puppies in kilograms -/
def weight_difference : ℝ := 5

theorem hartley_puppies_count :
  num_puppies * puppy_weight = num_cats * cat_weight - weight_difference :=
by sorry

end NUMINAMATH_CALUDE_hartley_puppies_count_l3374_337444


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_3_union_equality_iff_m_range_l3374_337483

-- Define the sets A and B
def A : Set ℝ := {x | |x| ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 2*m + 1}

-- Define the complement of A
def C_U_A : Set ℝ := {x | |x| > 3}

-- Theorem 1
theorem intersection_complement_when_m_3 :
  (C_U_A ∩ B 3) = {x | 3 < x ∧ x < 7} := by sorry

-- Theorem 2
theorem union_equality_iff_m_range :
  ∀ m : ℝ, (A ∪ B m = A) ↔ (-2 ≤ m ∧ m ≤ 1) := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_3_union_equality_iff_m_range_l3374_337483


namespace NUMINAMATH_CALUDE_weed_pulling_l3374_337452

theorem weed_pulling (day1 : ℕ) : 
  let day2 := 3 * day1
  let day3 := day2 / 5
  let day4 := day3 - 10
  day1 + day2 + day3 + day4 = 120 →
  day1 = 25 := by
sorry

end NUMINAMATH_CALUDE_weed_pulling_l3374_337452


namespace NUMINAMATH_CALUDE_stock_exchange_problem_l3374_337451

/-- The number of stocks that closed higher today -/
def higher_stocks : ℕ := 1080

/-- The number of stocks that closed lower today -/
def lower_stocks : ℕ := 900

/-- The total number of stocks on the stock exchange -/
def total_stocks : ℕ := higher_stocks + lower_stocks

theorem stock_exchange_problem :
  (higher_stocks = lower_stocks * 120 / 100) →
  (total_stocks = 1980) := by
  sorry

end NUMINAMATH_CALUDE_stock_exchange_problem_l3374_337451


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3374_337421

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) :
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3374_337421


namespace NUMINAMATH_CALUDE_tenth_house_gnomes_l3374_337445

/-- Represents the number of gnomes in each house on the street. -/
structure GnomeCounts where
  house1 : Nat
  house2 : Nat
  house3 : Nat
  house4 : Nat
  house5 : Nat
  house6 : Nat
  house7 : Nat
  house8 : Nat
  house9 : Nat

/-- The theorem stating that the tenth house must have 3 gnomes. -/
theorem tenth_house_gnomes (g : GnomeCounts) : 
  g.house1 = 4 ∧
  g.house2 = 2 * g.house1 ∧
  g.house3 = g.house2 - 3 ∧
  g.house4 = g.house1 + g.house3 ∧
  g.house5 = 5 ∧
  g.house6 = 2 ∧
  g.house7 = 7 ∧
  g.house8 = g.house4 + 3 ∧
  g.house9 = 10 →
  65 - (g.house1 + g.house2 + g.house3 + g.house4 + g.house5 + g.house6 + g.house7 + g.house8 + g.house9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tenth_house_gnomes_l3374_337445


namespace NUMINAMATH_CALUDE_area_of_triangle_l3374_337490

/-- Two externally tangent circles with centers O and O' and radii 1 and 2 -/
structure TangentCircles where
  O : ℝ × ℝ
  O' : ℝ × ℝ
  radius_C : ℝ
  radius_C' : ℝ
  tangent_externally : (O.1 - O'.1)^2 + (O.2 - O'.2)^2 = (radius_C + radius_C')^2
  radius_C_eq_1 : radius_C = 1
  radius_C'_eq_2 : radius_C' = 2

/-- Point P is on circle C, and P' is on circle C' -/
def TangentPoints (tc : TangentCircles) :=
  {P : ℝ × ℝ | (P.1 - tc.O.1)^2 + (P.2 - tc.O.2)^2 = tc.radius_C^2} ×
  {P' : ℝ × ℝ | (P'.1 - tc.O'.1)^2 + (P'.2 - tc.O'.2)^2 = tc.radius_C'^2}

/-- X is the intersection point of O'P and OP' -/
def IntersectionPoint (tc : TangentCircles) (tp : TangentPoints tc) : ℝ × ℝ :=
  sorry -- Definition of X as the intersection point

/-- The area of triangle OXO' -/
def TriangleArea (tc : TangentCircles) (tp : TangentPoints tc) : ℝ :=
  let X := IntersectionPoint tc tp
  sorry -- Definition of the area of triangle OXO'

/-- Main theorem: The area of triangle OXO' is (4√2 - √5) / 3 -/
theorem area_of_triangle (tc : TangentCircles) (tp : TangentPoints tc) :
  TriangleArea tc tp = (4 * Real.sqrt 2 - Real.sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_l3374_337490


namespace NUMINAMATH_CALUDE_four_possible_d_values_l3374_337455

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition of two 5-digit numbers resulting in another 5-digit number -/
def ValidAddition (a b c d : Digit) : Prop :=
  ∃ (n : ℕ), n < 100000 ∧
  10000 * a.val + 1000 * b.val + 100 * c.val + 10 * d.val + a.val +
  10000 * c.val + 1000 * b.val + 100 * a.val + 10 * d.val + d.val =
  10000 * d.val + 1000 * d.val + 100 * d.val + 10 * c.val + b.val

/-- The main theorem stating that there are exactly 4 possible values for D -/
theorem four_possible_d_values :
  ∃! (s : Finset Digit), s.card = 4 ∧
  ∀ d : Digit, d ∈ s ↔
    ∃ (a b c : Digit), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d ∧ a ≠ c ∧ b ≠ d ∧
    ValidAddition a b c d :=
sorry

end NUMINAMATH_CALUDE_four_possible_d_values_l3374_337455


namespace NUMINAMATH_CALUDE_triangle_properties_l3374_337489

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circumradius
def circumradius (t : Triangle) : ℝ := sorry

-- Define the length of a side
def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define an angle in a triangle
def angle (t : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem triangle_properties (t : Triangle) :
  side_length t.A t.B = Real.sqrt 10 →
  side_length t.A t.C = Real.sqrt 2 →
  circumradius t = Real.sqrt 5 →
  angle t t.C < Real.pi / 2 →
  side_length t.B t.C = 4 ∧ angle t t.C = Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3374_337489


namespace NUMINAMATH_CALUDE_largest_divisible_by_seven_l3374_337482

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    n = A * 10000 + B * 1000 + B * 100 + C * 10 + A

theorem largest_divisible_by_seven :
  ∀ n : ℕ, is_valid_number n → n % 7 = 0 → n ≤ 98879 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_by_seven_l3374_337482


namespace NUMINAMATH_CALUDE_total_silk_dyed_l3374_337449

theorem total_silk_dyed (green_silk : ℕ) (pink_silk : ℕ) 
  (h1 : green_silk = 61921) (h2 : pink_silk = 49500) : 
  green_silk + pink_silk = 111421 := by
  sorry

end NUMINAMATH_CALUDE_total_silk_dyed_l3374_337449


namespace NUMINAMATH_CALUDE_playground_ball_cost_l3374_337457

theorem playground_ball_cost (jump_rope_cost board_game_cost savings_from_allowance savings_from_uncle additional_needed : ℕ) :
  jump_rope_cost = 7 →
  board_game_cost = 12 →
  savings_from_allowance = 6 →
  savings_from_uncle = 13 →
  additional_needed = 4 →
  ∃ (playground_ball_cost : ℕ),
    playground_ball_cost = 4 ∧
    jump_rope_cost + board_game_cost + playground_ball_cost = savings_from_allowance + savings_from_uncle + additional_needed :=
by sorry

end NUMINAMATH_CALUDE_playground_ball_cost_l3374_337457


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3374_337469

theorem matrix_equation_solution : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -4; 3, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-16, -6; 7, 2]
  let M : Matrix (Fin 2) (Fin 2) ℤ := !![5, -7; -2, 3]
  M * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3374_337469


namespace NUMINAMATH_CALUDE_first_question_percentage_l3374_337440

theorem first_question_percentage
  (second_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : second_correct = 49)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 32)
  : ∃ first_correct : ℝ, first_correct = 63 := by
  sorry

end NUMINAMATH_CALUDE_first_question_percentage_l3374_337440


namespace NUMINAMATH_CALUDE_linear_function_composition_l3374_337417

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  is_linear f → (∀ x, f (f x) = 9 * x + 4) → 
  (∀ x, f x = 3 * x + 1) ∨ (∀ x, f x = -3 * x - 2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_composition_l3374_337417


namespace NUMINAMATH_CALUDE_shelter_dogs_count_l3374_337409

theorem shelter_dogs_count (dogs cats : ℕ) : 
  (dogs : ℚ) / cats = 15 / 7 →
  dogs / (cats + 8) = 15 / 11 →
  dogs = 30 := by
sorry

end NUMINAMATH_CALUDE_shelter_dogs_count_l3374_337409


namespace NUMINAMATH_CALUDE_star_two_ten_star_not_distributive_l3374_337427

/-- Define the ※ operation for rational numbers -/
def star (m n : ℚ) : ℚ := 3 * m - n

/-- Theorem stating that 2※10 = -4 -/
theorem star_two_ten : star 2 10 = -4 := by sorry

/-- Theorem stating that the ※ operation does not satisfy the distributive law -/
theorem star_not_distributive : ∃ a b c : ℚ, star a (b + c) ≠ star a b + star a c := by sorry

end NUMINAMATH_CALUDE_star_two_ten_star_not_distributive_l3374_337427


namespace NUMINAMATH_CALUDE_absolute_value_of_two_is_not_negative_two_l3374_337465

theorem absolute_value_of_two_is_not_negative_two : ¬(|2| = -2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_two_is_not_negative_two_l3374_337465


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3374_337429

theorem complex_magnitude_product : Complex.abs (5 - 3 * Complex.I) * Complex.abs (5 + 3 * Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3374_337429


namespace NUMINAMATH_CALUDE_equation_solutions_l3374_337491

-- Define the equation
def equation (m n : ℕ+) : Prop := 3^(m.val) - 2^(n.val) = 1

-- State the theorem
theorem equation_solutions :
  ∀ m n : ℕ+, equation m n ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3374_337491


namespace NUMINAMATH_CALUDE_fraction_subtraction_and_division_l3374_337477

theorem fraction_subtraction_and_division :
  (5/6 - 1/3) / (2/9) = 9/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_subtraction_and_division_l3374_337477


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_sum_4_l3374_337408

/-- Sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Check if a year is after 2020 and has sum of digits equal to 4 -/
def isValidYear (year : ℕ) : Prop :=
  year > 2020 ∧ sumOfDigits year = 4

/-- 2022 is the first year after 2020 with sum of digits equal to 4 -/
theorem first_year_after_2020_with_sum_4 :
  (∀ y : ℕ, y < 2022 → ¬(isValidYear y)) ∧ isValidYear 2022 := by
  sorry

#eval sumOfDigits 2020  -- Should output 4
#eval sumOfDigits 2022  -- Should output 4

end NUMINAMATH_CALUDE_first_year_after_2020_with_sum_4_l3374_337408


namespace NUMINAMATH_CALUDE_fifth_term_is_thirteen_l3374_337478

/-- An arithmetic sequence with the first term 1 and common difference 3 -/
def arithmeticSeq : ℕ → ℤ
  | 0 => 1
  | n+1 => arithmeticSeq n + 3

/-- The theorem stating that the 5th term of the sequence is 13 -/
theorem fifth_term_is_thirteen : arithmeticSeq 4 = 13 := by
  sorry

#eval arithmeticSeq 4  -- This will evaluate to 13

end NUMINAMATH_CALUDE_fifth_term_is_thirteen_l3374_337478


namespace NUMINAMATH_CALUDE_range_of_d_l3374_337495

noncomputable def circleC (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

def pointA : ℝ × ℝ := (0, -1)
def pointB : ℝ × ℝ := (0, 1)

def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def d (p : ℝ × ℝ) : ℝ :=
  distanceSquared p pointA + distanceSquared p pointB

theorem range_of_d :
  ∀ p : ℝ × ℝ, circleC p.1 p.2 → 32 ≤ d p ∧ d p ≤ 72 :=
sorry

end NUMINAMATH_CALUDE_range_of_d_l3374_337495


namespace NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l3374_337410

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: If (S_3/3) - (S_2/2) = 1 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_seq_common_diff
  (seq : ArithmeticSequence)
  (h : seq.S 3 / 3 - seq.S 2 / 2 = 1) :
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l3374_337410


namespace NUMINAMATH_CALUDE_lydia_porch_flowers_l3374_337435

/-- The number of flowers on Lydia's porch --/
def flowers_on_porch (total_plants : ℕ) (flowering_percent : ℚ) 
  (seven_flower_percent : ℚ) (seven_flower_plants : ℕ) (four_flower_plants : ℕ) : ℕ :=
  seven_flower_plants * 7 + four_flower_plants * 4

/-- Theorem stating the number of flowers on Lydia's porch --/
theorem lydia_porch_flowers :
  flowers_on_porch 120 (35/100) (60/100) 8 6 = 80 := by
  sorry

end NUMINAMATH_CALUDE_lydia_porch_flowers_l3374_337435


namespace NUMINAMATH_CALUDE_intersection_empty_iff_k_greater_than_six_l3374_337484

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 7}
def N (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2*k - 1}

theorem intersection_empty_iff_k_greater_than_six (k : ℝ) : 
  M ∩ N k = ∅ ↔ k > 6 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_k_greater_than_six_l3374_337484


namespace NUMINAMATH_CALUDE_minimum_cookies_cookies_exist_l3374_337479

theorem minimum_cookies (b : ℕ) : b ≡ 5 [ZMOD 6] ∧ b ≡ 7 [ZMOD 8] ∧ b ≡ 8 [ZMOD 9] → b ≥ 239 := by
  sorry

theorem cookies_exist : ∃ b : ℕ, b ≡ 5 [ZMOD 6] ∧ b ≡ 7 [ZMOD 8] ∧ b ≡ 8 [ZMOD 9] ∧ b = 239 := by
  sorry

end NUMINAMATH_CALUDE_minimum_cookies_cookies_exist_l3374_337479


namespace NUMINAMATH_CALUDE_hen_count_l3374_337400

theorem hen_count (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 48)
  (h2 : total_feet = 136)
  (h3 : hen_feet = 2)
  (h4 : cow_feet = 4) : 
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧
    hens = 28 := by
  sorry

end NUMINAMATH_CALUDE_hen_count_l3374_337400


namespace NUMINAMATH_CALUDE_last_four_digits_theorem_l3374_337463

theorem last_four_digits_theorem :
  ∃ (N : ℕ+),
    (∃ (a b c d : ℕ),
      a ≠ 0 ∧
      a ≠ 6 ∧ b ≠ 6 ∧ c ≠ 6 ∧
      N % 10000 = a * 1000 + b * 100 + c * 10 + d ∧
      (N * N) % 10000 = a * 1000 + b * 100 + c * 10 + d ∧
      a * 100 + b * 10 + c = 106) :=
by sorry

end NUMINAMATH_CALUDE_last_four_digits_theorem_l3374_337463


namespace NUMINAMATH_CALUDE_class_average_approximation_l3374_337441

/-- Represents the class data for a test --/
structure ClassData where
  total_students : ℕ
  section1_percent : ℝ
  section1_average : ℝ
  section2_percent : ℝ
  section2_average : ℝ
  section3_percent : ℝ
  section3_average : ℝ
  section4_average : ℝ
  weight1 : ℝ
  weight2 : ℝ
  weight3 : ℝ
  weight4 : ℝ

/-- Calculates the weighted overall class average --/
def weightedAverage (data : ClassData) : ℝ :=
  data.section1_average * data.weight1 +
  data.section2_average * data.weight2 +
  data.section3_average * data.weight3 +
  data.section4_average * data.weight4

/-- Theorem stating that the weighted overall class average is approximately 86% --/
theorem class_average_approximation (data : ClassData) 
  (h1 : data.total_students = 120)
  (h2 : data.section1_percent = 0.187)
  (h3 : data.section1_average = 0.965)
  (h4 : data.section2_percent = 0.355)
  (h5 : data.section2_average = 0.784)
  (h6 : data.section3_percent = 0.258)
  (h7 : data.section3_average = 0.882)
  (h8 : data.section4_average = 0.647)
  (h9 : data.weight1 = 0.35)
  (h10 : data.weight2 = 0.25)
  (h11 : data.weight3 = 0.30)
  (h12 : data.weight4 = 0.10)
  (h13 : data.section1_percent + data.section2_percent + data.section3_percent + 
         (1 - data.section1_percent - data.section2_percent - data.section3_percent) = 1) :
  abs (weightedAverage data - 0.86) < 0.005 := by
  sorry


end NUMINAMATH_CALUDE_class_average_approximation_l3374_337441


namespace NUMINAMATH_CALUDE_cake_price_l3374_337418

theorem cake_price (num_cakes : ℕ) (num_pies : ℕ) (pie_price : ℚ) (total_revenue : ℚ) : 
  num_cakes = 453 → 
  num_pies = 126 → 
  pie_price = 7 → 
  total_revenue = 6318 → 
  ∃ (cake_price : ℚ), cake_price = 12 ∧ num_cakes * cake_price + num_pies * pie_price = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_cake_price_l3374_337418


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l3374_337472

theorem ratio_of_sum_to_difference (a b : ℝ) : 
  0 < b ∧ b < a ∧ a + b = 7 * (a - b) → a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l3374_337472


namespace NUMINAMATH_CALUDE_p_only_root_zero_l3374_337464

/-- Recursive definition of polynomial p_n(x) -/
def p : ℕ → ℝ → ℝ
| 0, x => 0
| 1, x => x
| (n+2), x => x * p (n+1) x + (1 - x) * p n x

/-- Theorem stating that 0 is the only real root of p_n(x) for n ≥ 1 -/
theorem p_only_root_zero (n : ℕ) (h : n ≥ 1) :
  ∀ x : ℝ, p n x = 0 ↔ x = 0 := by
  sorry

#check p_only_root_zero

end NUMINAMATH_CALUDE_p_only_root_zero_l3374_337464


namespace NUMINAMATH_CALUDE_sqrt_sin_cos_identity_l3374_337422

theorem sqrt_sin_cos_identity : 
  Real.sqrt (1 - 2 * Real.sin (π + 2) * Real.cos (π - 2)) = Real.sin 2 - Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sin_cos_identity_l3374_337422


namespace NUMINAMATH_CALUDE_manuscript_completion_time_l3374_337494

/-- The planned completion time for inputting the manuscript -/
def completion_time : ℚ := 5/3

/-- The original number of computers -/
def original_computers : ℕ := 9

theorem manuscript_completion_time :
  (completion_time = 5/3) ∧
  (original_computers : ℚ) / ((original_computers : ℚ) + 3) = 3/4 ∧
  (original_computers : ℚ) / ((original_computers : ℚ) - 3) = completion_time / (completion_time + 5/6) :=
by sorry

end NUMINAMATH_CALUDE_manuscript_completion_time_l3374_337494


namespace NUMINAMATH_CALUDE_line_proof_l3374_337404

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - 3*y + 10 = 0
def line2 (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def line3 (x y : ℝ) : Prop := 3*x - 2*y + 4 = 0
def result_line (x y : ℝ) : Prop := 2*x + 3*y - 2 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_proof :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    result_line x y ∧
    perpendicular (3/2) (-2/3) :=
by sorry

end NUMINAMATH_CALUDE_line_proof_l3374_337404


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l3374_337488

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l3374_337488


namespace NUMINAMATH_CALUDE_petr_speed_l3374_337460

theorem petr_speed (total_distance : ℝ) (ivan_speed : ℝ) (remaining_distance : ℝ) (time : ℝ) :
  total_distance = 153 →
  ivan_speed = 46 →
  remaining_distance = 24 →
  time = 1.5 →
  ∃ petr_speed : ℝ,
    petr_speed = 40 ∧
    total_distance - remaining_distance = (ivan_speed + petr_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_petr_speed_l3374_337460


namespace NUMINAMATH_CALUDE_rectangle_area_l3374_337485

-- Define the rectangle
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

-- Define the area function
def area (r : Rectangle) : ℝ :=
  (r.x2 - r.x1) * (r.y2 - r.y1)

-- Theorem statement
theorem rectangle_area :
  let r : Rectangle := { x1 := 0, y1 := 0, x2 := 3, y2 := 3 }
  area r = 9 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3374_337485


namespace NUMINAMATH_CALUDE_unique_solution_l3374_337480

-- Define the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := lg (x + 1) = (1 / 2) * (Real.log x / Real.log 3)

-- Theorem statement
theorem unique_solution : ∃! x : ℝ, x > 0 ∧ equation x ∧ x = 9 :=
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3374_337480


namespace NUMINAMATH_CALUDE_x_value_proof_l3374_337406

theorem x_value_proof (x y z : ℤ) 
  (eq1 : x + y = 20) 
  (eq2 : x - y = 10) 
  (eq3 : x + y + z = 30) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3374_337406


namespace NUMINAMATH_CALUDE_population_growth_proof_l3374_337499

/-- The percentage increase in population during the first year -/
def first_year_increase : ℝ := 25

/-- The initial population two years ago -/
def initial_population : ℝ := 1200

/-- The population after two years of growth -/
def final_population : ℝ := 1950

/-- The percentage increase in population during the second year -/
def second_year_increase : ℝ := 30

theorem population_growth_proof :
  initial_population * (1 + first_year_increase / 100) * (1 + second_year_increase / 100) = final_population :=
sorry

end NUMINAMATH_CALUDE_population_growth_proof_l3374_337499


namespace NUMINAMATH_CALUDE_shifted_checkerboard_half_shaded_l3374_337496

/-- Represents a square grid -/
structure SquareGrid :=
  (size : Nat)

/-- Represents a shading pattern on a square grid -/
structure ShadingPattern :=
  (grid : SquareGrid)
  (shaded_squares : Nat)

/-- A 6x6 grid with a shifted checkerboard shading pattern -/
def shifted_checkerboard : ShadingPattern :=
  { grid := { size := 6 },
    shaded_squares := 18 }

/-- Calculate the percentage of shaded squares -/
def shaded_percentage (pattern : ShadingPattern) : Rat :=
  pattern.shaded_squares / (pattern.grid.size * pattern.grid.size) * 100

/-- Theorem: The shaded percentage of a 6x6 shifted checkerboard is 50% -/
theorem shifted_checkerboard_half_shaded :
  shaded_percentage shifted_checkerboard = 50 := by
  sorry

end NUMINAMATH_CALUDE_shifted_checkerboard_half_shaded_l3374_337496


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l3374_337474

theorem reciprocal_of_negative_one_third :
  let x : ℚ := -1/3
  let y : ℚ := -3
  (x * y = 1) → (y = 1 / x) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l3374_337474


namespace NUMINAMATH_CALUDE_exercise_distribution_properties_l3374_337476

/-- Represents the frequency distribution of daily exercise time --/
structure ExerciseDistribution :=
  (less_than_70 : ℕ)
  (between_70_and_80 : ℕ)
  (between_80_and_90 : ℕ)
  (greater_than_90 : ℕ)

/-- Theorem stating the properties of the exercise distribution --/
theorem exercise_distribution_properties
  (dist : ExerciseDistribution)
  (total_surveyed : ℕ)
  (h1 : dist.less_than_70 = 14)
  (h2 : dist.between_70_and_80 = 40)
  (h3 : dist.between_80_and_90 = 35)
  (h4 : total_surveyed = 100) :
  let m := (dist.between_70_and_80 : ℚ) / total_surveyed * 100
  let n := dist.greater_than_90
  let estimated_80_plus := ((dist.between_80_and_90 + dist.greater_than_90 : ℚ) / total_surveyed * 1000).floor
  let p := 86
  (m = 40 ∧ n = 11) ∧
  estimated_80_plus = 460 ∧
  (((11 : ℚ) / total_surveyed * 100 ≤ 25) ∧ 
   ((11 + 35 : ℚ) / total_surveyed * 100 ≥ 25)) := by
  sorry

#check exercise_distribution_properties

end NUMINAMATH_CALUDE_exercise_distribution_properties_l3374_337476


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l3374_337424

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (max : ℝ), max = Real.sqrt 13 ∧ x + 1/x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l3374_337424


namespace NUMINAMATH_CALUDE_company_fund_problem_l3374_337481

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (initial_fund = 60 * n - 10) →    -- Fund was $10 short for $60 bonuses
  (50 * n + 120 = initial_fund) →   -- $50 bonuses given, $120 remained
  (initial_fund = 770) :=           -- Prove initial fund was $770
by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l3374_337481


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l3374_337471

theorem max_reciprocal_sum (t q u₁ u₂ : ℝ) : 
  (u₁ * u₂ = q) →
  (u₁ + u₂ = t) →
  (u₁ + u₂ = u₁^2 + u₂^2) →
  (u₁ + u₂ = u₁^4 + u₂^4) →
  (∃ (x : ℝ), x^2 - t*x + q = 0) →
  (∀ (v₁ v₂ : ℝ), v₁ * v₂ = q ∧ v₁ + v₂ = t ∧ v₁ + v₂ = v₁^2 + v₂^2 ∧ v₁ + v₂ = v₁^4 + v₂^4 →
    1/u₁^2009 + 1/u₂^2009 ≥ 1/v₁^2009 + 1/v₂^2009) →
  1/u₁^2009 + 1/u₂^2009 = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l3374_337471


namespace NUMINAMATH_CALUDE_line_in_quadrants_l3374_337453

-- Define a line y = kx + b
structure Line where
  k : ℝ
  b : ℝ

-- Define quadrants
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to check if a line passes through a quadrant
def passesThrough (l : Line) (q : Quadrant) : Prop := sorry

-- Theorem statement
theorem line_in_quadrants (l : Line) :
  passesThrough l Quadrant.first ∧ 
  passesThrough l Quadrant.third ∧ 
  passesThrough l Quadrant.fourth →
  l.k > 0 := by sorry

end NUMINAMATH_CALUDE_line_in_quadrants_l3374_337453


namespace NUMINAMATH_CALUDE_complex_number_range_l3374_337426

theorem complex_number_range (x : ℝ) :
  let z : ℂ := (x + Complex.I) / (3 - Complex.I)
  (z.re < 0 ∧ z.im > 0) → (-3 < x ∧ x < 1/3) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_range_l3374_337426


namespace NUMINAMATH_CALUDE_max_edges_no_triangle_max_edges_no_K4_l3374_337401

/-- The Turán number T(n, r) is the maximum number of edges in a graph with n vertices that does not contain a complete subgraph of r+1 vertices. -/
def turan_number (n : ℕ) (r : ℕ) : ℕ := sorry

/-- A graph with n vertices -/
structure Graph (n : ℕ) where
  edges : Finset (Fin n × Fin n)

/-- The number of edges in a graph -/
def num_edges {n : ℕ} (G : Graph n) : ℕ := G.edges.card

/-- A graph contains a triangle if it has a complete subgraph of 3 vertices -/
def has_triangle {n : ℕ} (G : Graph n) : Prop := sorry

/-- A graph contains a K4 if it has a complete subgraph of 4 vertices -/
def has_K4 {n : ℕ} (G : Graph n) : Prop := sorry

theorem max_edges_no_triangle (G : Graph 30) :
  ¬has_triangle G → num_edges G ≤ 225 :=
sorry

theorem max_edges_no_K4 (G : Graph 30) :
  ¬has_K4 G → num_edges G ≤ 200 :=
sorry

end NUMINAMATH_CALUDE_max_edges_no_triangle_max_edges_no_K4_l3374_337401


namespace NUMINAMATH_CALUDE_y_percent_of_x_in_terms_of_z_l3374_337431

theorem y_percent_of_x_in_terms_of_z (x y z : ℝ) 
  (h1 : 0.7 * (x - y) = 0.3 * (x + y))
  (h2 : 0.6 * (x + z) = 0.4 * (y - z)) :
  y = 0.4 * x := by
  sorry

end NUMINAMATH_CALUDE_y_percent_of_x_in_terms_of_z_l3374_337431


namespace NUMINAMATH_CALUDE_prize_interval_l3374_337448

/-- Proves that the interval between prizes is 400 given the conditions of the tournament. -/
theorem prize_interval (total_prize : ℕ) (first_prize : ℕ) (interval : ℕ) : 
  total_prize = 4800 → 
  first_prize = 2000 → 
  total_prize = first_prize + (first_prize - interval) + (first_prize - 2 * interval) →
  interval = 400 := by
  sorry

#check prize_interval

end NUMINAMATH_CALUDE_prize_interval_l3374_337448


namespace NUMINAMATH_CALUDE_equation_solutions_l3374_337497

theorem equation_solutions : 
  {x : ℝ | (x + 1) * (x - 2) = x + 1} = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3374_337497


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3374_337487

theorem contrapositive_equivalence : 
  (∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) ↔ 
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3374_337487


namespace NUMINAMATH_CALUDE_system_inequalities_solution_l3374_337461

theorem system_inequalities_solution (a b : ℝ) : 
  (∀ x : ℝ, (x + a - 2 > 0 ∧ 2*x - b - 1 < 0) ↔ (0 < x ∧ x < 1)) →
  (a = 2 ∧ b = 1) := by
sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_l3374_337461


namespace NUMINAMATH_CALUDE_remainder_1234567890_mod_99_l3374_337402

theorem remainder_1234567890_mod_99 : 1234567890 % 99 = 72 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567890_mod_99_l3374_337402


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_l3374_337432

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_l3374_337432


namespace NUMINAMATH_CALUDE_planes_perp_to_line_are_parallel_lines_perp_to_plane_are_parallel_lines_perp_to_line_not_always_parallel_planes_perp_to_plane_not_always_parallel_l3374_337416

-- Define basic geometric objects
variable (Point Line Plane : Type)

-- Define perpendicular and parallel relations
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Theorem for proposition ②
theorem planes_perp_to_line_are_parallel 
  (l : Line) (p1 p2 : Plane) 
  (h1 : perpendicular_plane_line p1 l) 
  (h2 : perpendicular_plane_line p2 l) : 
  parallel_plane p1 p2 :=
sorry

-- Theorem for proposition ③
theorem lines_perp_to_plane_are_parallel 
  (p : Plane) (l1 l2 : Line) 
  (h1 : perpendicular_line_plane l1 p) 
  (h2 : perpendicular_line_plane l2 p) : 
  parallel_line l1 l2 :=
sorry

-- Theorem for proposition ① (to be proven false)
theorem lines_perp_to_line_not_always_parallel 
  (l : Line) (l1 l2 : Line) 
  (h1 : perpendicular_line_line l1 l) 
  (h2 : perpendicular_line_line l2 l) : 
  ¬(parallel_line l1 l2) :=
sorry

-- Theorem for proposition ④ (to be proven false)
theorem planes_perp_to_plane_not_always_parallel 
  (p : Plane) (p1 p2 : Plane) 
  (h1 : perpendicular_plane_plane p1 p) 
  (h2 : perpendicular_plane_plane p2 p) : 
  ¬(parallel_plane p1 p2) :=
sorry

end NUMINAMATH_CALUDE_planes_perp_to_line_are_parallel_lines_perp_to_plane_are_parallel_lines_perp_to_line_not_always_parallel_planes_perp_to_plane_not_always_parallel_l3374_337416


namespace NUMINAMATH_CALUDE_smallest_n_value_l3374_337438

theorem smallest_n_value (N : ℕ) (c₁ c₂ c₃ c₄ : ℕ) : 
  (c₁ ≤ N) ∧ (c₂ ≤ N) ∧ (c₃ ≤ N) ∧ (c₄ ≤ N) ∧ 
  (c₁ = 4 * c₂ - 3) ∧
  (N + c₂ = 4 * c₄) ∧
  (2 * N + c₃ = 4 * c₃ - 1) ∧
  (3 * N + c₄ = 4 * c₁ - 3) →
  N = 1 ∧ c₁ = 1 ∧ c₂ = 1 ∧ c₃ = 1 ∧ c₄ = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l3374_337438


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l3374_337486

def initial_people : ℕ := 7
def initial_average_age : ℚ := 28
def leaving_person_age : ℕ := 20

theorem average_age_after_leaving :
  let total_age : ℚ := initial_people * initial_average_age
  let remaining_total_age : ℚ := total_age - leaving_person_age
  let remaining_people : ℕ := initial_people - 1
  remaining_total_age / remaining_people = 29.33 := by sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l3374_337486


namespace NUMINAMATH_CALUDE_investment_net_change_l3374_337493

def initial_investment : ℝ := 200
def first_year_loss_rate : ℝ := 0.1
def second_year_gain_rate : ℝ := 0.3

theorem investment_net_change :
  let first_year_amount := initial_investment * (1 - first_year_loss_rate)
  let second_year_amount := first_year_amount * (1 + second_year_gain_rate)
  let net_change_rate := (second_year_amount - initial_investment) / initial_investment
  net_change_rate = 0.17 := by
sorry

end NUMINAMATH_CALUDE_investment_net_change_l3374_337493


namespace NUMINAMATH_CALUDE_permutation_ratio_l3374_337420

/-- The number of permutations of m elements chosen from n elements -/
def A (n m : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - m))

/-- Theorem stating that the ratio of A(n,m) to A(n-1,m-1) equals n -/
theorem permutation_ratio (n m : ℕ) (h : n ≥ m) :
  A n m / A (n - 1) (m - 1) = n := by sorry

end NUMINAMATH_CALUDE_permutation_ratio_l3374_337420


namespace NUMINAMATH_CALUDE_statue_of_liberty_model_height_l3374_337414

/-- The scale ratio of the model to the actual size -/
def scaleRatio : ℚ := 1 / 25

/-- The actual height of the Statue of Liberty in feet -/
def actualHeight : ℕ := 305

/-- Rounds a rational number to the nearest integer -/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- The height of the scale model in feet -/
def modelHeight : ℚ := actualHeight * scaleRatio

theorem statue_of_liberty_model_height :
  roundToNearest modelHeight = 12 := by
  sorry

end NUMINAMATH_CALUDE_statue_of_liberty_model_height_l3374_337414


namespace NUMINAMATH_CALUDE_intersection_implies_a_nonpositive_l3374_337459

def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {1, 3, a}

theorem intersection_implies_a_nonpositive (a : ℝ) :
  (A ∩ B a).Nonempty → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_nonpositive_l3374_337459
