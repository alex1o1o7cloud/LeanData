import Mathlib

namespace NUMINAMATH_CALUDE_three_digit_number_puzzle_l2163_216362

theorem three_digit_number_puzzle (A B : ℕ) : 
  (100 ≤ A * 100 + 30 + B) ∧ 
  (A * 100 + 30 + B < 1000) ∧ 
  (A * 100 + 30 + B - 41 = 591) → 
  B = 2 := by sorry

end NUMINAMATH_CALUDE_three_digit_number_puzzle_l2163_216362


namespace NUMINAMATH_CALUDE_tree_height_difference_l2163_216398

/-- Given three trees with specific height relationships, prove the difference between half the height of the tallest tree and the height of the middle-sized tree. -/
theorem tree_height_difference (tallest middle smallest : ℝ) : 
  tallest = 108 →
  smallest = 12 →
  smallest = (1/4) * middle →
  (tallest / 2) - middle = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_height_difference_l2163_216398


namespace NUMINAMATH_CALUDE_dog_paws_on_ground_l2163_216314

theorem dog_paws_on_ground : ∀ (total_dogs : ℕ) (dogs_on_back_legs : ℕ) (dogs_on_all_fours : ℕ),
  total_dogs = 12 →
  dogs_on_back_legs = total_dogs / 2 →
  dogs_on_all_fours = total_dogs / 2 →
  dogs_on_back_legs + dogs_on_all_fours = total_dogs →
  dogs_on_all_fours * 4 + dogs_on_back_legs * 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_dog_paws_on_ground_l2163_216314


namespace NUMINAMATH_CALUDE_divisibility_by_1946_l2163_216366

theorem divisibility_by_1946 (n : ℕ) (hn : n ≤ 1945) :
  ∃ k : ℤ, 1492^n - 1770^n - 1863^n + 2141^n = 1946 * k := by
  sorry


end NUMINAMATH_CALUDE_divisibility_by_1946_l2163_216366


namespace NUMINAMATH_CALUDE_square_ratio_theorem_l2163_216301

theorem square_ratio_theorem (area_ratio : ℚ) (side_ratio : ℚ) 
  (a b c : ℕ) (h1 : area_ratio = 50 / 98) :
  side_ratio = Real.sqrt (area_ratio) ∧
  side_ratio = 5 / 7 ∧
  (a : ℚ) * Real.sqrt b / (c : ℚ) = side_ratio ∧
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_theorem_l2163_216301


namespace NUMINAMATH_CALUDE_simplify_radical_product_l2163_216350

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  Real.sqrt (28 * x) * Real.sqrt (15 * x) * Real.sqrt (21 * x) = 42 * x * Real.sqrt (5 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l2163_216350


namespace NUMINAMATH_CALUDE_total_crayons_l2163_216340

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) (h1 : crayons_per_child = 12) (h2 : num_children = 18) :
  crayons_per_child * num_children = 216 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l2163_216340


namespace NUMINAMATH_CALUDE_distribute_balls_count_l2163_216349

/-- The number of ways to put 4 different balls into 4 different boxes, leaving exactly two boxes empty -/
def ways_to_distribute_balls : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating that the number of ways to distribute the balls is 84 -/
theorem distribute_balls_count : ways_to_distribute_balls = 84 := by
  sorry

end NUMINAMATH_CALUDE_distribute_balls_count_l2163_216349


namespace NUMINAMATH_CALUDE_triangle_vertices_from_midpoints_l2163_216345

/-- Given a triangle with midpoints, prove its vertices -/
theorem triangle_vertices_from_midpoints :
  let m1 : ℚ × ℚ := (1/4, 13/4)
  let m2 : ℚ × ℚ := (-1/2, 1)
  let m3 : ℚ × ℚ := (-5/4, 5/4)
  let v1 : ℚ × ℚ := (-2, -1)
  let v2 : ℚ × ℚ := (-1/2, 13/4)
  let v3 : ℚ × ℚ := (1, 7/2)
  (m1.1 = (v2.1 + v3.1) / 2 ∧ m1.2 = (v2.2 + v3.2) / 2) ∧
  (m2.1 = (v1.1 + v3.1) / 2 ∧ m2.2 = (v1.2 + v3.2) / 2) ∧
  (m3.1 = (v1.1 + v2.1) / 2 ∧ m3.2 = (v1.2 + v2.2) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_vertices_from_midpoints_l2163_216345


namespace NUMINAMATH_CALUDE_no_maximum_on_interval_l2163_216365

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

-- Define the property of being an even function
def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Theorem statement
theorem no_maximum_on_interval (m : ℝ) :
  is_even_function (f m) →
  ¬∃ (y : ℝ), ∀ x ∈ Set.Ioo (-2 : ℝ) (-1), f m x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_no_maximum_on_interval_l2163_216365


namespace NUMINAMATH_CALUDE_may_savings_l2163_216311

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (0-indexed)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_may_savings_l2163_216311


namespace NUMINAMATH_CALUDE_garage_wheels_count_l2163_216303

/-- The number of bikes that can be assembled -/
def num_bikes : ℕ := 9

/-- The number of wheels required for each bike -/
def wheels_per_bike : ℕ := 2

/-- The total number of wheels in the garage -/
def total_wheels : ℕ := num_bikes * wheels_per_bike

theorem garage_wheels_count : total_wheels = 18 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_count_l2163_216303


namespace NUMINAMATH_CALUDE_two_integer_k_values_for_nontrivial_solution_l2163_216312

/-- The system of equations has a non-trivial solution for exactly two integer values of k. -/
theorem two_integer_k_values_for_nontrivial_solution :
  ∃! (s : Finset ℤ), (∀ k ∈ s, ∃ a b c : ℝ, (a, b, c) ≠ (0, 0, 0) ∧
    a^2 + b^2 = k * c * (a + b) ∧
    b^2 + c^2 = k * a * (b + c) ∧
    c^2 + a^2 = k * b * (c + a)) ∧
  s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_integer_k_values_for_nontrivial_solution_l2163_216312


namespace NUMINAMATH_CALUDE_equal_cost_at_60_messages_l2163_216320

/-- Cost of Plan A for x text messages -/
def planACost (x : ℕ) : ℚ := 0.25 * x + 9

/-- Cost of Plan B for x text messages -/
def planBCost (x : ℕ) : ℚ := 0.40 * x

/-- The number of text messages where both plans cost the same -/
def equalCostMessages : ℕ := 60

theorem equal_cost_at_60_messages :
  planACost equalCostMessages = planBCost equalCostMessages :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_60_messages_l2163_216320


namespace NUMINAMATH_CALUDE_mobile_phone_sales_growth_l2163_216352

/-- Represents the sales growth of mobile phones over two months -/
theorem mobile_phone_sales_growth 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (monthly_growth_rate : ℝ) 
  (h1 : initial_sales = 400) 
  (h2 : final_sales = 900) :
  initial_sales * (1 + monthly_growth_rate)^2 = final_sales := by
  sorry

end NUMINAMATH_CALUDE_mobile_phone_sales_growth_l2163_216352


namespace NUMINAMATH_CALUDE_hyperbola_x_axis_l2163_216341

/-- Given k > 1, the equation (1-k)x^2 + y^2 = k^2 - 1 represents a hyperbola with its real axis along the x-axis -/
theorem hyperbola_x_axis (k : ℝ) (h : k > 1) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), ((1-k)*x^2 + y^2 = k^2 - 1) ↔ (x^2/a^2 - y^2/b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_x_axis_l2163_216341


namespace NUMINAMATH_CALUDE_h_min_neg_l2163_216378

-- Define the functions f, g, and h
variable (f g : ℝ → ℝ)
variable (a b : ℝ)

def h (x : ℝ) := a * f x + b * g x + 2

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_odd : ∀ x, g (-x) = -g x

-- Define the maximum value of h on (0, +∞)
axiom h_max : ∀ x > 0, h x ≤ 5

-- State the theorem to be proved
theorem h_min_neg : (∀ x < 0, h x ≥ -1) := by sorry

end NUMINAMATH_CALUDE_h_min_neg_l2163_216378


namespace NUMINAMATH_CALUDE_triangle_max_area_l2163_216317

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  b = 2 →
  (1 - Real.sqrt 3 * Real.cos B) / (Real.sqrt 3 * Real.sin B) = 1 / Real.tan C →
  ∃ (S : ℝ), S = Real.sqrt ((1 / 4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2)) ∧
  ∀ (S' : ℝ), S' = Real.sqrt ((1 / 4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2)) → S' ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2163_216317


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2163_216339

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2/x + 8/y = 1) :
  x + y ≥ 18 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2/x + 8/y = 1 ∧ x + y = 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2163_216339


namespace NUMINAMATH_CALUDE_friends_ratio_l2163_216330

theorem friends_ratio (james_friends : ℕ) (shared_friends : ℕ) (combined_list : ℕ) :
  james_friends = 75 →
  shared_friends = 25 →
  combined_list = 275 →
  ∃ (john_friends : ℕ),
    john_friends = combined_list - james_friends →
    (john_friends : ℚ) / james_friends = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_friends_ratio_l2163_216330


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l2163_216316

theorem tripled_base_and_exponent (c d : ℤ) (y : ℚ) (h1 : d ≠ 0) :
  (3 * c : ℚ) ^ (3 * d) = c ^ d * y ^ d → y = 27 * c ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l2163_216316


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l2163_216302

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem stating that "q > 1" is neither necessary nor sufficient for a geometric sequence to be monotonically increasing -/
theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  ¬(GeometricSequence a q ∧ (q > 1 ↔ MonotonicallyIncreasing a)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l2163_216302


namespace NUMINAMATH_CALUDE_tan_product_theorem_l2163_216359

theorem tan_product_theorem (α β : Real) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) (h5 : α + β = π / 4) :
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_theorem_l2163_216359


namespace NUMINAMATH_CALUDE_genevieve_money_proof_l2163_216309

/-- The amount of money Genevieve had initially -/
def genevieve_initial_amount (cost_per_kg : ℚ) (bought_kg : ℚ) (short_amount : ℚ) : ℚ :=
  cost_per_kg * bought_kg - short_amount

/-- Proof that Genevieve's initial amount was $1600 -/
theorem genevieve_money_proof (cost_per_kg : ℚ) (bought_kg : ℚ) (short_amount : ℚ)
  (h1 : cost_per_kg = 8)
  (h2 : bought_kg = 250)
  (h3 : short_amount = 400) :
  genevieve_initial_amount cost_per_kg bought_kg short_amount = 1600 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_money_proof_l2163_216309


namespace NUMINAMATH_CALUDE_yogurt_combinations_yogurt_shop_combinations_l2163_216328

theorem yogurt_combinations (n : ℕ) (k : ℕ) : n ≥ k → (n.choose k) = n.factorial / (k.factorial * (n - k).factorial) := by sorry

theorem yogurt_shop_combinations : 
  (5 : ℕ) * ((7 : ℕ).choose 3) = 175 := by sorry

end NUMINAMATH_CALUDE_yogurt_combinations_yogurt_shop_combinations_l2163_216328


namespace NUMINAMATH_CALUDE_roots_inside_unit_circle_iff_triangle_interior_l2163_216360

/-- The region in the (a,b) plane where both roots of z^2 + az + b = 0 satisfy |z| < 1 -/
def roots_inside_unit_circle (a b : ℝ) : Prop :=
  ∀ z : ℂ, z^2 + a*z + b = 0 → Complex.abs z < 1

/-- The interior of the triangle with vertices (2, 1), (-2, 1), and (0, -1) -/
def triangle_interior (a b : ℝ) : Prop :=
  b < 1 ∧ b > a - 1 ∧ b > -a - 1 ∧ b > -1

theorem roots_inside_unit_circle_iff_triangle_interior (a b : ℝ) :
  roots_inside_unit_circle a b ↔ triangle_interior a b :=
sorry

end NUMINAMATH_CALUDE_roots_inside_unit_circle_iff_triangle_interior_l2163_216360


namespace NUMINAMATH_CALUDE_gcd_90_450_l2163_216315

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by sorry

end NUMINAMATH_CALUDE_gcd_90_450_l2163_216315


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2163_216388

/-- A shape made of unit cubes -/
structure CubeShape where
  /-- The number of cubes in the base -/
  base_cubes : ℕ
  /-- The number of layers -/
  layers : ℕ
  /-- The total number of cubes -/
  total_cubes : ℕ
  /-- Condition: The base is a square -/
  base_is_square : base_cubes = 4
  /-- Condition: There are two layers -/
  two_layers : layers = 2
  /-- Condition: Total cubes is the product of base cubes and layers -/
  total_cubes_eq : total_cubes = base_cubes * layers

/-- The volume of the shape in cubic units -/
def volume (shape : CubeShape) : ℕ := shape.total_cubes

/-- The surface area of the shape in square units -/
def surface_area (shape : CubeShape) : ℕ :=
  6 * shape.total_cubes - 2 * shape.base_cubes

/-- The theorem stating the ratio of volume to surface area -/
theorem volume_to_surface_area_ratio (shape : CubeShape) :
  2 * (volume shape) = surface_area shape := by
  sorry

#check volume_to_surface_area_ratio

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2163_216388


namespace NUMINAMATH_CALUDE_pauls_rate_l2163_216396

/-- The number of cars Paul and Jack can service in a day -/
def total_cars (paul_rate : ℝ) : ℝ := 8 * (paul_rate + 3)

/-- Theorem stating Paul's rate of changing oil in cars per hour -/
theorem pauls_rate : ∃ (paul_rate : ℝ), total_cars paul_rate = 40 ∧ paul_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_pauls_rate_l2163_216396


namespace NUMINAMATH_CALUDE_range_of_m_value_of_m_l2163_216364

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + m - 2

-- Define the condition for real roots
def has_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0

-- Define the additional condition
def additional_condition (x₁ x₂ : ℝ) : Prop :=
  (x₁ + 2) * (x₂ + 2) - 2 * x₁ * x₂ = 17

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  has_real_roots m → (m ≥ 2/3 ∧ m ≠ 1) :=
sorry

-- Theorem for the value of m given the additional condition
theorem value_of_m (m : ℝ) (x₁ x₂ : ℝ) :
  has_real_roots m →
  quadratic_equation m x₁ = 0 →
  quadratic_equation m x₂ = 0 →
  additional_condition x₁ x₂ →
  m = 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_value_of_m_l2163_216364


namespace NUMINAMATH_CALUDE_nabla_problem_l2163_216300

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^(a - 1)

-- State the theorem
theorem nabla_problem : nabla (nabla 2 3) 4 = 1027 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l2163_216300


namespace NUMINAMATH_CALUDE_least_value_quadratic_equation_l2163_216332

theorem least_value_quadratic_equation :
  let f : ℝ → ℝ := λ y => 3 * y^2 + 5 * y + 2
  ∃ y_min : ℝ, (f y_min = 4) ∧ (∀ y : ℝ, f y = 4 → y ≥ y_min) ∧ y_min = -2 := by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_equation_l2163_216332


namespace NUMINAMATH_CALUDE_retail_price_calculation_l2163_216397

/-- Proves that the retail price of a machine is $120 given the specified conditions -/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  wholesale_price = 90 →
  discount_rate = 0.1 →
  profit_rate = 0.2 →
  ∃ (retail_price : ℝ),
    retail_price = 120 ∧
    wholesale_price * (1 + profit_rate) = retail_price * (1 - discount_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l2163_216397


namespace NUMINAMATH_CALUDE_jake_newspaper_count_l2163_216318

/-- The number of newspapers Jake delivers in a week -/
def jake_newspapers : ℕ := 234

/-- The number of newspapers Miranda delivers in a week -/
def miranda_newspapers : ℕ := 2 * jake_newspapers

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

theorem jake_newspaper_count : jake_newspapers = 234 :=
  by
    have h1 : miranda_newspapers = 2 * jake_newspapers := by rfl
    have h2 : weeks_in_month * miranda_newspapers - weeks_in_month * jake_newspapers = 936 :=
      by sorry
    sorry

end NUMINAMATH_CALUDE_jake_newspaper_count_l2163_216318


namespace NUMINAMATH_CALUDE_choose_three_from_eight_l2163_216375

theorem choose_three_from_eight :
  Nat.choose 8 3 = 56 := by sorry

end NUMINAMATH_CALUDE_choose_three_from_eight_l2163_216375


namespace NUMINAMATH_CALUDE_baker_pastries_l2163_216353

theorem baker_pastries (cakes_made : ℕ) (pastries_sold : ℕ) (total_cakes_sold : ℕ) (difference : ℕ) :
  cakes_made = 14 →
  pastries_sold = 8 →
  total_cakes_sold = 97 →
  total_cakes_sold - pastries_sold = difference →
  difference = 89 →
  pastries_sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_baker_pastries_l2163_216353


namespace NUMINAMATH_CALUDE_double_after_increase_decrease_l2163_216368

theorem double_after_increase_decrease (r s N : ℝ) 
  (hr : r > 0) (hs : s > 0) (hN : N > 0) (hs_bound : s < 50) :
  N * (1 + r / 100) * (1 - s / 100) = 2 * N ↔ 
  r = (10000 + 100 * s) / (100 - s) :=
by sorry

end NUMINAMATH_CALUDE_double_after_increase_decrease_l2163_216368


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l2163_216380

theorem rational_solutions_quadratic (k : ℕ+) :
  (∃ x : ℚ, 2 * k * x^2 + 36 * x + 3 * k = 0) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l2163_216380


namespace NUMINAMATH_CALUDE_probability_different_groups_l2163_216344

/-- The number of study groups -/
def num_groups : ℕ := 6

/-- The number of members in each study group -/
def members_per_group : ℕ := 3

/-- The total number of people -/
def total_people : ℕ := num_groups * members_per_group

/-- The number of people to be selected -/
def selection_size : ℕ := 3

/-- The probability of selecting 3 people from different study groups -/
theorem probability_different_groups : 
  (Nat.choose num_groups selection_size : ℚ) / (Nat.choose total_people selection_size : ℚ) = 5 / 204 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_groups_l2163_216344


namespace NUMINAMATH_CALUDE_problem_solution_l2163_216382

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x : ℝ | a + 1 ≤ x ∧ x ≤ 2 * a - 1}

theorem problem_solution :
  (∀ x : ℝ, x ∈ (M ∪ N (7/2)) ↔ -2 ≤ x ∧ x ≤ 6) ∧
  (∀ x : ℝ, x ∈ ((Set.univ \ M) ∩ N (7/2)) ↔ 5 < x ∧ x ≤ 6) ∧
  (∀ a : ℝ, M ⊇ N a ↔ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2163_216382


namespace NUMINAMATH_CALUDE_consecutive_binomial_ratio_l2163_216377

theorem consecutive_binomial_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k+1) : ℚ) = 1/3 ∧
  (n.choose (k+1) : ℚ) / (n.choose (k+2) : ℚ) = 3/5 →
  n + k = 8 := by
sorry

end NUMINAMATH_CALUDE_consecutive_binomial_ratio_l2163_216377


namespace NUMINAMATH_CALUDE_wand_cost_is_60_l2163_216351

/-- The cost of a magic wand at Wizards Park -/
def wand_cost : ℕ → Prop := λ x =>
  -- Kate buys 3 wands and sells 2 of them
  -- She sells each wand for $5 more than she paid
  -- She collected $130 after the sale
  2 * (x + 5) = 130

/-- The cost of each wand is $60 -/
theorem wand_cost_is_60 : wand_cost 60 := by sorry

end NUMINAMATH_CALUDE_wand_cost_is_60_l2163_216351


namespace NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l2163_216392

theorem one_and_two_thirds_of_x_is_45 : ∃ x : ℝ, (5/3) * x = 45 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l2163_216392


namespace NUMINAMATH_CALUDE_sequence_nth_term_l2163_216329

/-- Given a sequence {a_n} where the differences between successive terms form
    a geometric sequence with first term 1 and common ratio r, 
    prove that the nth term of the sequence is (1-r^(n-1))/(1-r) -/
theorem sequence_nth_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n : ℕ, a (n+1) - a n = r^(n-1)) :
  ∀ n : ℕ, a n = (1 - r^(n-1)) / (1 - r) :=
sorry

end NUMINAMATH_CALUDE_sequence_nth_term_l2163_216329


namespace NUMINAMATH_CALUDE_triangle_problem_l2163_216372

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a > b →
  a = 5 →
  c = 6 →
  Real.sin B = 3/5 →
  (b = Real.sqrt 13 ∧ 
   Real.sin A = (3 * Real.sqrt 13) / 13) ∧
  Real.sin (2*A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2163_216372


namespace NUMINAMATH_CALUDE_cafe_tables_theorem_l2163_216385

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- Calculates the number of tables needed given the number of people and people per table --/
def tablesNeeded (people : Nat) (peoplePerTable : Nat) : Nat :=
  people / peoplePerTable

theorem cafe_tables_theorem (seatingCapacity : Nat) (peoplePerTable : Nat) :
  seatingCapacity = 312 ∧ peoplePerTable = 3 →
  tablesNeeded (base7ToBase10 seatingCapacity) peoplePerTable = 52 := by
  sorry

#eval base7ToBase10 312  -- Should output 156
#eval tablesNeeded 156 3  -- Should output 52

end NUMINAMATH_CALUDE_cafe_tables_theorem_l2163_216385


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2163_216324

/-- The original hyperbola equation -/
def original_hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

/-- The new hyperbola equation -/
def new_hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 12 = 1

/-- Definition of asymptotes for a hyperbola -/
def has_same_asymptotes (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    ∀ (x y : ℝ), (f x y ↔ g (k * x) (k * y))

/-- The main theorem to prove -/
theorem hyperbola_properties :
  has_same_asymptotes original_hyperbola new_hyperbola ∧
  new_hyperbola 2 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2163_216324


namespace NUMINAMATH_CALUDE_coordinates_wrt_symmetric_point_l2163_216391

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry about y-axis
def symmetricAboutYAxis (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = p.y

-- Theorem statement
theorem coordinates_wrt_symmetric_point (A B : Point2D) :
  A.x = -5 ∧ A.y = 2 ∧ symmetricAboutYAxis A B →
  (A.x - B.x = 5 ∧ A.y - B.y = 0) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_symmetric_point_l2163_216391


namespace NUMINAMATH_CALUDE_max_sector_area_l2163_216306

/-- The maximum area of a sector with circumference 4 -/
theorem max_sector_area (r l : ℝ) (h1 : r > 0) (h2 : l > 0) (h3 : 2*r + l = 4) :
  (1/2) * l * r ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_sector_area_l2163_216306


namespace NUMINAMATH_CALUDE_anya_andrea_erasers_l2163_216393

theorem anya_andrea_erasers : 
  ∀ (andrea_erasers : ℕ) (anya_multiplier : ℕ),
    andrea_erasers = 4 →
    anya_multiplier = 4 →
    anya_multiplier * andrea_erasers - andrea_erasers = 12 := by
  sorry

end NUMINAMATH_CALUDE_anya_andrea_erasers_l2163_216393


namespace NUMINAMATH_CALUDE_quadratic_sum_cubes_twice_product_l2163_216307

theorem quadratic_sum_cubes_twice_product (m : ℝ) : 
  (∃ a b : ℝ, 3 * a^2 + 6 * a + m = 0 ∧ 
              3 * b^2 + 6 * b + m = 0 ∧ 
              a ≠ b ∧ 
              a^3 + b^3 = 2 * a * b) ↔ 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_cubes_twice_product_l2163_216307


namespace NUMINAMATH_CALUDE_geometric_relations_l2163_216399

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (perpendicular : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)
variable (contains : Plane → Point → Prop)
variable (on_line : Point → Line → Prop)
variable (plane_through_point_perp_to_line : Point → Line → Plane)
variable (line_perp_to_plane : Point → Plane → Line)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem geometric_relations 
  (α β : Plane) (l : Line) (P : Point) 
  (h1 : perpendicular α β)
  (h2 : intersection α β = l)
  (h3 : contains α P)
  (h4 : ¬ on_line P l) :
  (perpendicular (plane_through_point_perp_to_line P l) β) ∧ 
  (parallel (line_perp_to_plane P α) β) ∧
  (line_in_plane (line_perp_to_plane P β) α) :=
sorry

end NUMINAMATH_CALUDE_geometric_relations_l2163_216399


namespace NUMINAMATH_CALUDE_union_M_N_l2163_216390

def M : Set ℝ := {x | x ≥ -1}
def N : Set ℝ := {x | 2 - x^2 ≥ 0}

theorem union_M_N : M ∪ N = {x : ℝ | x ≥ -Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_union_M_N_l2163_216390


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2163_216371

theorem complex_equation_solution (z : ℂ) : 
  (Complex.I * z = 4 + 3 * Complex.I) → (z = 3 - 4 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2163_216371


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l2163_216394

theorem sum_of_special_numbers :
  ∀ A B : ℤ,
  (A = -3 - (-5)) →
  (B = 2 + (-2)) →
  A + B = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l2163_216394


namespace NUMINAMATH_CALUDE_power_of_power_l2163_216347

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2163_216347


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2163_216358

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 30| + |x - 20| = |3*x - 90| :=
by
  -- The unique solution is x = 40
  use 40
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2163_216358


namespace NUMINAMATH_CALUDE_round_trip_fuel_efficiency_l2163_216361

/-- Calculates the average fuel efficiency for a round trip given the conditions. -/
theorem round_trip_fuel_efficiency 
  (distance : ℝ) 
  (efficiency1 : ℝ) 
  (efficiency2 : ℝ) 
  (h1 : distance = 120) 
  (h2 : efficiency1 = 30) 
  (h3 : efficiency2 = 20) : 
  (2 * distance) / (distance / efficiency1 + distance / efficiency2) = 24 :=
by
  sorry

#check round_trip_fuel_efficiency

end NUMINAMATH_CALUDE_round_trip_fuel_efficiency_l2163_216361


namespace NUMINAMATH_CALUDE_cat_resisting_time_l2163_216355

/-- Proves that given a total time of 28 minutes, a walking distance of 64 feet,
    and a walking rate of 8 feet/minute, the time spent resisting is 20 minutes. -/
theorem cat_resisting_time
  (total_time : ℕ)
  (walking_distance : ℕ)
  (walking_rate : ℕ)
  (h1 : total_time = 28)
  (h2 : walking_distance = 64)
  (h3 : walking_rate = 8)
  : total_time - walking_distance / walking_rate = 20 := by
  sorry

#check cat_resisting_time

end NUMINAMATH_CALUDE_cat_resisting_time_l2163_216355


namespace NUMINAMATH_CALUDE_alphabetic_sequences_count_l2163_216346

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the sequence -/
def sequence_length : ℕ := 2013

/-- The number of alphabetic sequences of given length with letters in alphabetic order -/
def alphabetic_sequences (n : ℕ) : ℕ := Nat.choose (n + alphabet_size - 1) (alphabet_size - 1)

theorem alphabetic_sequences_count : 
  alphabetic_sequences sequence_length = Nat.choose 2038 25 := by sorry

end NUMINAMATH_CALUDE_alphabetic_sequences_count_l2163_216346


namespace NUMINAMATH_CALUDE_total_fish_count_l2163_216384

theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1 : num_fishbowls = 261) 
  (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l2163_216384


namespace NUMINAMATH_CALUDE_pirate_treasure_l2163_216335

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_l2163_216335


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2163_216322

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) * (x + 1) < 0 ↔ -1 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2163_216322


namespace NUMINAMATH_CALUDE_two_different_pitchers_l2163_216370

-- Define the type for pitchers
structure Pitcher :=
  (shape : ℕ)
  (color : ℕ)

-- Define the theorem
theorem two_different_pitchers 
  (pitchers : Set Pitcher) 
  (h1 : ∃ (a b : Pitcher), a ∈ pitchers ∧ b ∈ pitchers ∧ a.shape ≠ b.shape)
  (h2 : ∃ (c d : Pitcher), c ∈ pitchers ∧ d ∈ pitchers ∧ c.color ≠ d.color) :
  ∃ (x y : Pitcher), x ∈ pitchers ∧ y ∈ pitchers ∧ x.shape ≠ y.shape ∧ x.color ≠ y.color :=
sorry

end NUMINAMATH_CALUDE_two_different_pitchers_l2163_216370


namespace NUMINAMATH_CALUDE_equation_solutions_l2163_216387

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 - (x - 2) = 0 ↔ x = 2 ∨ x = 3) ∧
  (∀ x : ℝ, x^2 - x = x + 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2163_216387


namespace NUMINAMATH_CALUDE_bug_path_tiles_l2163_216348

/-- The number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tiles_visited (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- Theorem stating that a bug walking diagonally across an 18x24 rectangular floor visits 36 tiles -/
theorem bug_path_tiles : tiles_visited 18 24 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l2163_216348


namespace NUMINAMATH_CALUDE_parabola_point_distance_to_focus_l2163_216381

/-- Given a parabola y = 4x² and a point M(x, y) on the parabola,
    if the distance from M to the focus (0, 1/16) is 1,
    then the y-coordinate of M is 15/16 -/
theorem parabola_point_distance_to_focus (x y : ℝ) :
  y = 4 * x^2 →
  (x - 0)^2 + (y - 1/16)^2 = 1 →
  y = 15/16 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_to_focus_l2163_216381


namespace NUMINAMATH_CALUDE_rotation_of_A_about_B_l2163_216337

-- Define the points
def A : ℝ × ℝ := (-4, 1)
def B : ℝ × ℝ := (-1, 4)

-- Define the rotation function
def rotate180AboutPoint (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  let (cx, cy) := center
  (2 * cx - px, 2 * cy - py)

-- Theorem statement
theorem rotation_of_A_about_B :
  rotate180AboutPoint A B = (2, 7) := by sorry

end NUMINAMATH_CALUDE_rotation_of_A_about_B_l2163_216337


namespace NUMINAMATH_CALUDE_river_current_speed_l2163_216342

/-- Theorem: Given a swimmer's speed in still water and the ratio of upstream to downstream swimming time, we can determine the speed of the river's current. -/
theorem river_current_speed 
  (swimmer_speed : ℝ) 
  (upstream_downstream_ratio : ℝ) 
  (h1 : swimmer_speed = 10) 
  (h2 : upstream_downstream_ratio = 3) : 
  ∃ (current_speed : ℝ), current_speed = 5 ∧ 
  (swimmer_speed + current_speed) * upstream_downstream_ratio = 
  (swimmer_speed - current_speed) * (upstream_downstream_ratio + 1) := by
  sorry

end NUMINAMATH_CALUDE_river_current_speed_l2163_216342


namespace NUMINAMATH_CALUDE_min_value_theorem_l2163_216379

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 8) :
  (2/x + 3/y) ≥ 25/8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 8 ∧ 2/x₀ + 3/y₀ = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2163_216379


namespace NUMINAMATH_CALUDE_f_of_one_equals_twentyone_l2163_216304

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 3)^2 - 3 * (x + 3) + 1

-- State the theorem
theorem f_of_one_equals_twentyone : f 1 = 21 := by sorry

end NUMINAMATH_CALUDE_f_of_one_equals_twentyone_l2163_216304


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l2163_216323

/-- f(n) is the exponent of 2 in the prime factorization of n! -/
def f (n : ℕ+) : ℕ :=
  sorry

/-- For any positive integer a, there exist infinitely many positive integers n
    such that n - f(n) = a -/
theorem infinitely_many_solutions (a : ℕ+) :
  ∃ (S : Set ℕ+), Infinite S ∧ ∀ n ∈ S, n.val - f n = a.val := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l2163_216323


namespace NUMINAMATH_CALUDE_loss_equals_cost_of_five_balls_l2163_216313

def number_of_balls : ℕ := 13
def selling_price : ℕ := 720
def cost_per_ball : ℕ := 90

theorem loss_equals_cost_of_five_balls :
  (number_of_balls * cost_per_ball - selling_price) / cost_per_ball = 5 := by
  sorry

end NUMINAMATH_CALUDE_loss_equals_cost_of_five_balls_l2163_216313


namespace NUMINAMATH_CALUDE_roots_reciprocal_sum_l2163_216395

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 5 = 0 → x₂^2 - 2*x₂ - 5 = 0 → 1/x₁ + 1/x₂ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_roots_reciprocal_sum_l2163_216395


namespace NUMINAMATH_CALUDE_probability_sum_6_is_5_36_l2163_216334

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := faces * faces

/-- The number of combinations that result in a sum of 6 -/
def favorable_outcomes : ℕ := 5

/-- The probability of rolling a sum of 6 with two dice -/
def probability_sum_6 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_6_is_5_36 : probability_sum_6 = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_6_is_5_36_l2163_216334


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2163_216308

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x - 4

-- State the theorem
theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ f x > a) → a < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2163_216308


namespace NUMINAMATH_CALUDE_general_admission_tickets_l2163_216357

theorem general_admission_tickets (student_price general_price : ℕ) 
  (total_tickets total_revenue : ℕ) : 
  student_price = 4 →
  general_price = 6 →
  total_tickets = 525 →
  total_revenue = 2876 →
  ∃ (student_tickets general_tickets : ℕ),
    student_tickets + general_tickets = total_tickets ∧
    student_tickets * student_price + general_tickets * general_price = total_revenue ∧
    general_tickets = 388 :=
by sorry

end NUMINAMATH_CALUDE_general_admission_tickets_l2163_216357


namespace NUMINAMATH_CALUDE_fraction_zero_iff_x_plus_minus_five_l2163_216356

theorem fraction_zero_iff_x_plus_minus_five (x : ℝ) :
  (x^2 - 25) / (4 * x^2 - 2 * x) = 0 ↔ x = 5 ∨ x = -5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_iff_x_plus_minus_five_l2163_216356


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l2163_216369

theorem gold_coin_distribution (x y : ℕ) (h1 : x + y = 25) :
  ∃ k : ℕ, x^2 - y^2 = k * (x - y) → k = 25 := by
sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l2163_216369


namespace NUMINAMATH_CALUDE_midpoint_x_sum_l2163_216383

/-- Given a triangle in the Cartesian plane where the sum of x-coordinates of its vertices is 15,
    the sum of x-coordinates of the midpoints of its sides is also 15. -/
theorem midpoint_x_sum (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_x_sum_l2163_216383


namespace NUMINAMATH_CALUDE_equilateral_triangle_not_unique_l2163_216319

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  angle : ℝ

/-- Given one angle and the side opposite to it, an equilateral triangle is not uniquely determined -/
theorem equilateral_triangle_not_unique (α : ℝ) (s : ℝ) : 
  ∃ (t1 t2 : EquilateralTriangle), t1.angle = α ∧ t1.side = s ∧ t2.angle = α ∧ t2.side = s ∧ t1 ≠ t2 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_not_unique_l2163_216319


namespace NUMINAMATH_CALUDE_variance_of_transformed_data_l2163_216386

-- Define a type for our dataset
def Dataset := List ℝ

-- Define the variance of a dataset
noncomputable def variance (X : Dataset) : ℝ := sorry

-- Define the transformation function
def transform (X : Dataset) : Dataset := X.map (λ x => 2 * x - 5)

-- Theorem statement
theorem variance_of_transformed_data (X : Dataset) :
  variance X = 1/2 → variance (transform X) = 2 := by sorry

end NUMINAMATH_CALUDE_variance_of_transformed_data_l2163_216386


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2163_216374

/-- The proposition p -/
def p (m a : ℝ) : Prop := m^2 - 4*a*m + 3*a^2 < 0 ∧ a < 0

/-- The proposition q -/
def q (m : ℝ) : Prop := ∀ x > 0, x + 4/x ≥ 1 - m

theorem p_necessary_not_sufficient_for_q :
  (∃ a m : ℝ, q m → p m a) ∧
  (∃ a m : ℝ, p m a ∧ ¬(q m)) ∧
  (∀ a : ℝ, (∃ m : ℝ, p m a ∧ q m) ↔ a ∈ Set.Icc (-1) 0) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2163_216374


namespace NUMINAMATH_CALUDE_triangle_inequalities_l2163_216363

theorem triangle_inequalities (A B C : ℝ) (h_triangle : A + B + C = π) (h_obtuse : A > π/2) :
  (1 + Real.sin (A/2) + Real.sin (B/2) + Real.sin (C/2) < Real.cos (A/2) + Real.cos (B/2) + Real.cos (C/2)) ∧
  (1 - Real.cos A + Real.sin B + Real.sin C < Real.sin A + Real.cos B + Real.cos C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l2163_216363


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2163_216376

def i : ℂ := Complex.I

theorem modulus_of_complex_fraction : 
  Complex.abs ((1 + 3 * i) / (1 - 2 * i)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2163_216376


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l2163_216338

theorem arithmetic_mean_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 70) : 
  c - a = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l2163_216338


namespace NUMINAMATH_CALUDE_employed_males_percentage_l2163_216327

theorem employed_males_percentage (total_employed_percent : Real) 
  (employed_females_percent : Real) (h1 : total_employed_percent = 64) 
  (h2 : employed_females_percent = 28.125) : 
  (total_employed_percent / 100) * (100 - employed_females_percent) = 45.96 :=
sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l2163_216327


namespace NUMINAMATH_CALUDE_reduce_piles_to_zero_reduce_table_to_zero_l2163_216373

/-- Represents the state of three piles of stones -/
structure ThreePiles :=
  (pile1 pile2 pile3 : Nat)

/-- Represents the state of an 8x5 table of natural numbers -/
def Table := Fin 8 → Fin 5 → Nat

/-- Allowed operations on three piles of stones -/
inductive PileOperation
  | removeOne : PileOperation
  | doubleOne : Fin 3 → PileOperation

/-- Allowed operations on the table -/
inductive TableOperation
  | doubleColumn : Fin 5 → TableOperation
  | subtractRow : Fin 8 → TableOperation

/-- Applies a pile operation to a ThreePiles state -/
def applyPileOp (s : ThreePiles) (op : PileOperation) : ThreePiles :=
  match op with
  | PileOperation.removeOne => ⟨s.pile1 - 1, s.pile2 - 1, s.pile3 - 1⟩
  | PileOperation.doubleOne i =>
      match i with
      | 0 => ⟨s.pile1 * 2, s.pile2, s.pile3⟩
      | 1 => ⟨s.pile1, s.pile2 * 2, s.pile3⟩
      | 2 => ⟨s.pile1, s.pile2, s.pile3 * 2⟩

/-- Applies a table operation to a Table state -/
def applyTableOp (t : Table) (op : TableOperation) : Table :=
  match op with
  | TableOperation.doubleColumn j => fun i k => if k = j then t i k * 2 else t i k
  | TableOperation.subtractRow i => fun j k => if j = i then t j k - 1 else t j k

/-- Theorem stating that any ThreePiles state can be reduced to zero -/
theorem reduce_piles_to_zero (s : ThreePiles) :
  ∃ (ops : List PileOperation), (ops.foldl applyPileOp s).pile1 = 0 ∧
                                (ops.foldl applyPileOp s).pile2 = 0 ∧
                                (ops.foldl applyPileOp s).pile3 = 0 :=
  sorry

/-- Theorem stating that any Table state can be reduced to zero -/
theorem reduce_table_to_zero (t : Table) :
  ∃ (ops : List TableOperation), ∀ i j, (ops.foldl applyTableOp t) i j = 0 :=
  sorry

end NUMINAMATH_CALUDE_reduce_piles_to_zero_reduce_table_to_zero_l2163_216373


namespace NUMINAMATH_CALUDE_melted_mixture_weight_l2163_216354

def zinc_weight : ℝ := 31.5
def zinc_ratio : ℝ := 9
def copper_ratio : ℝ := 11

theorem melted_mixture_weight :
  let copper_weight := (copper_ratio / zinc_ratio) * zinc_weight
  let total_weight := zinc_weight + copper_weight
  total_weight = 70 := by sorry

end NUMINAMATH_CALUDE_melted_mixture_weight_l2163_216354


namespace NUMINAMATH_CALUDE_complex_trajectory_l2163_216343

theorem complex_trajectory (x y : ℝ) (h1 : x ≥ (1/2 : ℝ)) (z : ℂ) 
  (h2 : z = Complex.mk x y) (h3 : Complex.abs (z - 1) = x) : 
  y^2 = 2*x - 1 := by
sorry

end NUMINAMATH_CALUDE_complex_trajectory_l2163_216343


namespace NUMINAMATH_CALUDE_five_dice_not_same_probability_l2163_216331

theorem five_dice_not_same_probability :
  let n : ℕ := 6  -- number of sides on each die
  let k : ℕ := 5  -- number of dice rolled
  let total_outcomes : ℕ := n^k
  let same_number_outcomes : ℕ := n
  let not_same_number_probability : ℚ := 1 - (same_number_outcomes : ℚ) / total_outcomes
  not_same_number_probability = 1295 / 1296 :=
by sorry

end NUMINAMATH_CALUDE_five_dice_not_same_probability_l2163_216331


namespace NUMINAMATH_CALUDE_cost_of_16_pencils_10_notebooks_l2163_216367

/-- The cost of pencils and notebooks given specific quantities -/
def cost_of_items (pencil_price notebook_price : ℚ) (num_pencils num_notebooks : ℕ) : ℚ :=
  pencil_price * num_pencils + notebook_price * num_notebooks

/-- The theorem stating the cost of 16 pencils and 10 notebooks -/
theorem cost_of_16_pencils_10_notebooks :
  ∀ (pencil_price notebook_price : ℚ),
    cost_of_items pencil_price notebook_price 7 8 = 415/100 →
    cost_of_items pencil_price notebook_price 5 3 = 177/100 →
    cost_of_items pencil_price notebook_price 16 10 = 584/100 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_16_pencils_10_notebooks_l2163_216367


namespace NUMINAMATH_CALUDE_french_toast_loaves_l2163_216310

/-- Calculates the number of loaves of bread needed for french toast over a given number of weeks -/
def loaves_needed (slices_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (slices_per_loaf : ℕ) : ℕ :=
  (slices_per_day * days_per_week * weeks + slices_per_loaf - 1) / slices_per_loaf

theorem french_toast_loaves :
  let slices_per_day : ℕ := 3  -- Suzanne (1) + husband (1) + daughters (0.5 + 0.5)
  let days_per_week : ℕ := 2   -- Saturday and Sunday
  let weeks : ℕ := 52
  let slices_per_loaf : ℕ := 12
  loaves_needed slices_per_day days_per_week weeks slices_per_loaf = 26 := by
  sorry

#eval loaves_needed 3 2 52 12

end NUMINAMATH_CALUDE_french_toast_loaves_l2163_216310


namespace NUMINAMATH_CALUDE_sasha_muffins_count_l2163_216326

/-- The number of muffins Sasha made -/
def sasha_muffins : ℕ := 50

/-- The number of muffins Melissa made -/
def melissa_muffins : ℕ := 4 * sasha_muffins

/-- The number of muffins Tiffany made -/
def tiffany_muffins : ℕ := (sasha_muffins + melissa_muffins) / 2

/-- The total number of muffins made -/
def total_muffins : ℕ := sasha_muffins + melissa_muffins + tiffany_muffins

/-- The price of each muffin in cents -/
def muffin_price : ℕ := 400

/-- The total amount raised in cents -/
def total_raised : ℕ := 90000

theorem sasha_muffins_count : 
  sasha_muffins = 50 ∧ 
  melissa_muffins = 4 * sasha_muffins ∧
  tiffany_muffins = (sasha_muffins + melissa_muffins) / 2 ∧
  total_muffins * muffin_price = total_raised := by
  sorry

end NUMINAMATH_CALUDE_sasha_muffins_count_l2163_216326


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2163_216336

theorem cube_volume_problem (a : ℕ) : 
  (a + 1) * (a + 1) * (a - 2) = a^3 - 27 → a^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2163_216336


namespace NUMINAMATH_CALUDE_cookies_per_pack_l2163_216325

/-- Given information about Candy's cookie distribution --/
structure CookieDistribution where
  trays : ℕ
  cookies_per_tray : ℕ
  packs : ℕ
  trays_eq : trays = 4
  cookies_per_tray_eq : cookies_per_tray = 24
  packs_eq : packs = 8

/-- Theorem: The number of cookies in each pack is 12 --/
theorem cookies_per_pack (cd : CookieDistribution) : 
  (cd.trays * cd.cookies_per_tray) / cd.packs = 12 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_pack_l2163_216325


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l2163_216389

theorem complex_sum_of_powers (i : ℂ) : i^2 = -1 → i + i^2 + i^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l2163_216389


namespace NUMINAMATH_CALUDE_rain_duration_problem_l2163_216305

theorem rain_duration_problem (x : ℝ) : 
  let first_day := 10
  let second_day := first_day + x
  let third_day := 2 * second_day
  first_day + second_day + third_day = 46 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_rain_duration_problem_l2163_216305


namespace NUMINAMATH_CALUDE_correct_option_is_B_l2163_216321

-- Define the statements
def statement1 : Prop := False
def statement2 : Prop := True
def statement3 : Prop := True
def statement4 : Prop := False

-- Define the options
def optionA : Prop := statement1 ∧ statement2 ∧ statement3
def optionB : Prop := statement2 ∧ statement3
def optionC : Prop := statement2 ∧ statement4
def optionD : Prop := statement1 ∧ statement3 ∧ statement4

-- Theorem: The correct option is B
theorem correct_option_is_B : 
  (¬statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4) → 
  (optionB ∧ ¬optionA ∧ ¬optionC ∧ ¬optionD) :=
by sorry

end NUMINAMATH_CALUDE_correct_option_is_B_l2163_216321


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l2163_216333

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- State the theorem
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : t.B = (t.A + t.C) / 2)  -- B is arithmetic mean of A and C
  (h2 : t.b^2 = t.a * t.c)      -- b is geometric mean of a and c
  : t.A = t.B ∧ t.B = t.C ∧ t.a = t.b ∧ t.b = t.c :=
by sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l2163_216333
