import Mathlib

namespace NUMINAMATH_CALUDE_tree_distance_l1872_187233

/-- Given 8 equally spaced trees along a road, where the distance between
    the first and fifth tree is 100 feet, the distance between the first
    and last tree is 175 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  let distance_between (i j : ℕ) := d * (j - i : ℝ) / 4
  distance_between 1 n = 175 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l1872_187233


namespace NUMINAMATH_CALUDE_ball_probability_l1872_187231

theorem ball_probability (n : ℕ) : 
  (2 : ℝ) / ((n : ℝ) + 2) = 0.4 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l1872_187231


namespace NUMINAMATH_CALUDE_fish_population_calculation_l1872_187284

/-- Calculates the number of fish in a lake on May 1 based on sampling data --/
theorem fish_population_calculation (tagged_may : ℕ) (caught_sept : ℕ) (tagged_sept : ℕ) 
  (death_rate : ℚ) (new_fish_rate : ℚ) :
  tagged_may = 60 →
  caught_sept = 70 →
  tagged_sept = 3 →
  death_rate = 1/4 →
  new_fish_rate = 2/5 →
  (1 - death_rate) * tagged_may * caught_sept / tagged_sept * (1 - new_fish_rate) = 630 := by
  sorry

end NUMINAMATH_CALUDE_fish_population_calculation_l1872_187284


namespace NUMINAMATH_CALUDE_distance_difference_l1872_187246

theorem distance_difference (john_distance jill_distance jim_distance : ℝ) : 
  john_distance = 15 →
  jim_distance = 0.2 * jill_distance →
  jim_distance = 2 →
  john_distance - jill_distance = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l1872_187246


namespace NUMINAMATH_CALUDE_b_33_mod_35_l1872_187230

-- Definition of b_n
def b (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem b_33_mod_35 : b 33 % 35 = 21 := by sorry

end NUMINAMATH_CALUDE_b_33_mod_35_l1872_187230


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_achievable_l1872_187258

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (1 / x + 1 / y) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_reciprocal_sum_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_achievable_l1872_187258


namespace NUMINAMATH_CALUDE_houses_in_block_is_five_l1872_187203

/-- The number of houses in a block -/
def houses_in_block : ℕ := 5

/-- The number of candies received from each house -/
def candies_per_house : ℕ := 7

/-- The total number of candies received from each block -/
def candies_per_block : ℕ := 35

/-- Theorem: The number of houses in a block is 5 -/
theorem houses_in_block_is_five :
  houses_in_block = candies_per_block / candies_per_house :=
by sorry

end NUMINAMATH_CALUDE_houses_in_block_is_five_l1872_187203


namespace NUMINAMATH_CALUDE_polygon_sides_l1872_187287

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1260 → ∃ n : ℕ, n = 9 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1872_187287


namespace NUMINAMATH_CALUDE_inequality_proof_l1872_187237

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a + b)^n - a^n - b^n ≥ (2^n - 2) / 2^(n - 2) * a * b * (a + b)^(n - 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1872_187237


namespace NUMINAMATH_CALUDE_inequality_proof_l1872_187239

theorem inequality_proof (x y : ℝ) (a : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin a)^2 * y^(Real.cos a)^2 < x + y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1872_187239


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1872_187244

/-- The number of dots on each side of the square grid -/
def grid_size : ℕ := 5

/-- The number of different rectangles that can be formed in the grid -/
def num_rectangles : ℕ := (grid_size.choose 2) * (grid_size.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_5x5_grid : num_rectangles = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1872_187244


namespace NUMINAMATH_CALUDE_potato_distribution_l1872_187214

theorem potato_distribution (total : ℕ) (gina : ℕ) (left : ℕ) :
  total = 300 →
  gina = 69 →
  left = 47 →
  ∃ (tom : ℕ) (k : ℕ),
    tom = k * gina ∧
    total = gina + tom + (tom / 3) + left →
    tom / gina = 2 :=
by sorry

end NUMINAMATH_CALUDE_potato_distribution_l1872_187214


namespace NUMINAMATH_CALUDE_range_of_a_l1872_187281

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | x < 2*a ∨ x > -a}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- State the theorem
theorem range_of_a (a : ℝ) : 
  a < 0 → 
  (∀ x, ¬(p x) → ¬(q a x)) ∧ 
  (∃ x, ¬(p x) ∧ q a x) → 
  a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1872_187281


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1872_187201

/-- A geometric sequence with a_3 = 2 and a_6 = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n) ∧ 
  a 3 = 2 ∧ 
  a 6 = 16

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 2^(n - 2)

theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  ∀ n : ℕ, a n = general_term n :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1872_187201


namespace NUMINAMATH_CALUDE_number_equation_proof_l1872_187265

theorem number_equation_proof (n : ℤ) : n - 8 = 5 * 7 + 12 ↔ n = 55 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l1872_187265


namespace NUMINAMATH_CALUDE_right_triangle_existence_l1872_187251

theorem right_triangle_existence (α β : ℝ) :
  (∃ (x y z h : ℝ),
    x > 0 ∧ y > 0 ∧ z > 0 ∧ h > 0 ∧
    x^2 + y^2 = z^2 ∧
    x * y = z * h ∧
    x - y = α ∧
    z - h = β) ↔
  β > α :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l1872_187251


namespace NUMINAMATH_CALUDE_part1_part2_l1872_187219

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - 2*a^2

/-- Part 1: The range of a when f(x) > -9 always holds -/
theorem part1 (a : ℝ) : (∀ x, f a x > -9) → a ∈ Set.Ioo (-2) 2 := by sorry

/-- Part 2: Solving the inequality f(x) > 0 with respect to x -/
theorem part2 (a : ℝ) (x : ℝ) :
  (a > 0 → (f a x > 0 ↔ x ∈ Set.Iio (-a) ∪ Set.Ioi (2*a))) ∧
  (a = 0 → (f a x > 0 ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 0)) ∧
  (a < 0 → (f a x > 0 ↔ x ∈ Set.Iio (2*a) ∪ Set.Ioi (-a))) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1872_187219


namespace NUMINAMATH_CALUDE_b_to_c_interest_rate_b_to_c_interest_rate_is_12_percent_l1872_187298

/-- The interest rate at which B lent money to C, given the following conditions:
  * A lends Rs. 3500 to B at 10% per annum
  * B lends the same sum to C
  * B's gain over 3 years is Rs. 210
-/
theorem b_to_c_interest_rate : ℝ :=
  let principal : ℝ := 3500
  let a_to_b_rate : ℝ := 0.1
  let time : ℝ := 3
  let b_gain : ℝ := 210
  let a_to_b_interest : ℝ := principal * a_to_b_rate * time
  let total_interest_from_c : ℝ := a_to_b_interest + b_gain
  total_interest_from_c / (principal * time)

/-- Proof that the interest rate at which B lent money to C is 12% per annum -/
theorem b_to_c_interest_rate_is_12_percent : b_to_c_interest_rate = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_b_to_c_interest_rate_b_to_c_interest_rate_is_12_percent_l1872_187298


namespace NUMINAMATH_CALUDE_product_equals_specific_number_l1872_187229

theorem product_equals_specific_number (A B : ℕ) :
  990 * 991 * 992 * 993 = 966428000000 + A * 10000000 + 910000 + B * 100 + 40 →
  A * 10 + B = 50 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_specific_number_l1872_187229


namespace NUMINAMATH_CALUDE_number_problem_l1872_187283

theorem number_problem (x : ℝ) : 0.75 * x = 0.45 * 1500 + 495 → x = 1560 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1872_187283


namespace NUMINAMATH_CALUDE_no_integer_root_quadratic_pair_l1872_187235

theorem no_integer_root_quadratic_pair :
  ¬ ∃ (a b c : ℤ),
    (∃ (x₁ x₂ : ℤ), a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂) ∧
    (∃ (y₁ y₂ : ℤ), (a + 1) * y₁^2 + (b + 1) * y₁ + (c + 1) = 0 ∧ (a + 1) * y₂^2 + (b + 1) * y₂ + (c + 1) = 0 ∧ y₁ ≠ y₂) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_root_quadratic_pair_l1872_187235


namespace NUMINAMATH_CALUDE_linear_equation_negative_root_m_range_l1872_187202

theorem linear_equation_negative_root_m_range 
  (m : ℝ) 
  (h : ∃ x : ℝ, (3 * x - m + 1 = 2 * x - 1) ∧ (x < 0)) : 
  m < 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_negative_root_m_range_l1872_187202


namespace NUMINAMATH_CALUDE_intersection_of_odd_integers_and_open_interval_l1872_187253

def A : Set ℝ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem intersection_of_odd_integers_and_open_interval :
  A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_odd_integers_and_open_interval_l1872_187253


namespace NUMINAMATH_CALUDE_price_difference_l1872_187211

/-- The original price of Liz's old car -/
def original_price : ℝ := 32500

/-- The selling price of Liz's old car as a percentage of the original price -/
def selling_percentage : ℝ := 0.80

/-- The additional amount Liz needs to buy the new car -/
def additional_amount : ℝ := 4000

/-- The price of the new car -/
def new_car_price : ℝ := 30000

/-- The theorem stating the difference between the original price of the old car and the price of the new car -/
theorem price_difference : original_price - new_car_price = 2500 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_l1872_187211


namespace NUMINAMATH_CALUDE_quadratic_inequality_constant_value_theorem_constant_function_inequality_l1872_187208

-- 1. Prove that for all real x, x^2 + 2x + 2 ≥ 1
theorem quadratic_inequality (x : ℝ) : x^2 + 2*x + 2 ≥ 1 := by sorry

-- 2. Prove that for a > 0 and c < 0, min(3|ax^2 + bx + c| + 2) = 2
theorem constant_value_theorem (a b c : ℝ) (ha : a > 0) (hc : c < 0) :
  ∀ x, 3 * |a * x^2 + b * x + c| + 2 ≥ 2 := by sorry

-- 3. Prove that for y = ax^2 + bx + c where b > a > 0 and y ≥ 0 for all real x,
--    if (a+b+c)/(a+b) > m for all a, b, c satisfying the conditions, then m ≤ 9/8
theorem constant_function_inequality (a b c : ℝ) (hab : b > a) (ha : a > 0) 
  (h_nonneg : ∀ x, a * x^2 + b * x + c ≥ 0) 
  (h_inequality : (a + b + c) / (a + b) > 0) :
  (a + b + c) / (a + b) ≤ 9/8 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_constant_value_theorem_constant_function_inequality_l1872_187208


namespace NUMINAMATH_CALUDE_cone_cylinder_theorem_l1872_187299

/-- Given a cone with base radius 2 and slant height 4, and a cylinder with height √3 inside the cone -/
def cone_cylinder_problem :=
  ∃ (cone_base_radius cone_slant_height cylinder_height : ℝ),
    cone_base_radius = 2 ∧
    cone_slant_height = 4 ∧
    cylinder_height = Real.sqrt 3

theorem cone_cylinder_theorem (h : cone_cylinder_problem) :
  ∃ (max_cylinder_area sphere_surface_area sphere_volume : ℝ),
    max_cylinder_area = 2 * (1 + Real.sqrt 3) * Real.pi ∧
    sphere_surface_area = 7 * Real.pi ∧
    sphere_volume = (7 * Real.sqrt 7 * Real.pi) / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_cone_cylinder_theorem_l1872_187299


namespace NUMINAMATH_CALUDE_profit_increase_l1872_187224

theorem profit_increase (march_profit : ℝ) (march_profit_pos : march_profit > 0) :
  let april_profit := march_profit * 1.35
  let may_profit := april_profit * 0.8
  let june_profit := may_profit * 1.5
  (june_profit - march_profit) / march_profit = 0.62 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_l1872_187224


namespace NUMINAMATH_CALUDE_impossible_time_reduction_l1872_187225

theorem impossible_time_reduction (initial_speed : ℝ) (time_reduction : ℝ) : 
  initial_speed = 60 → time_reduction = 1 → ¬ ∃ (new_speed : ℝ), 
    new_speed > 0 ∧ (1 / new_speed) * 60 = (1 / initial_speed) * 60 - time_reduction :=
by
  sorry

end NUMINAMATH_CALUDE_impossible_time_reduction_l1872_187225


namespace NUMINAMATH_CALUDE_james_payment_l1872_187285

/-- Calculates James's payment at a restaurant given meal prices and tip percentage. -/
theorem james_payment (james_meal : ℝ) (friend_meal : ℝ) (tip_percentage : ℝ) : 
  james_meal = 16 →
  friend_meal = 14 →
  tip_percentage = 0.2 →
  james_meal + 0.5 * friend_meal + 0.5 * tip_percentage * (james_meal + friend_meal) = 19 :=
by sorry

end NUMINAMATH_CALUDE_james_payment_l1872_187285


namespace NUMINAMATH_CALUDE_six_circles_l1872_187292

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (a : Point) (b : Point) (c : Point)

/-- Two identical equilateral triangles sharing one vertex -/
structure TwoTriangles :=
  (t1 : EquilateralTriangle)
  (t2 : EquilateralTriangle)
  (shared_vertex : Point)
  (h1 : t1.c = shared_vertex)
  (h2 : t2.a = shared_vertex)

/-- A function that returns all circles satisfying the conditions -/
def circles_through_vertices (triangles : TwoTriangles) : Finset Circle := sorry

/-- The main theorem -/
theorem six_circles (triangles : TwoTriangles) :
  (circles_through_vertices triangles).card = 6 := by sorry

end NUMINAMATH_CALUDE_six_circles_l1872_187292


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1872_187207

theorem inequality_equivalence (x : ℝ) : 
  -1 < (x^2 - 10*x + 9) / (x^2 - 4*x + 5) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 5) < 1 ↔ x > 5.3 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1872_187207


namespace NUMINAMATH_CALUDE_only_one_correct_probability_l1872_187241

theorem only_one_correct_probability (p_a p_b : ℝ) : 
  p_a = 1/5 → p_b = 1/4 → 
  p_a * (1 - p_b) + (1 - p_a) * p_b = 7/20 := by
  sorry

end NUMINAMATH_CALUDE_only_one_correct_probability_l1872_187241


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l1872_187245

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The main theorem stating that if tan(A+B)(1-tan A tan B) = (√3 sin C) / (sin A cos B), then A = π/3. -/
theorem triangle_angle_theorem (t : Triangle) :
  Real.tan (t.A + t.B) * (1 - Real.tan t.A * Real.tan t.B) = (Real.sqrt 3 * Real.sin t.C) / (Real.sin t.A * Real.cos t.B) →
  t.A = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l1872_187245


namespace NUMINAMATH_CALUDE_f_is_even_l1872_187249

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem that f is an even function
theorem f_is_even : ∀ x : ℝ, f x = f (-x) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1872_187249


namespace NUMINAMATH_CALUDE_composition_ratio_l1872_187295

def f (x : ℝ) : ℝ := 3 * x + 1

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : f (g (f 3)) / g (f (g 3)) = 112 / 109 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l1872_187295


namespace NUMINAMATH_CALUDE_initial_acidic_percentage_l1872_187222

/-- Proves that the initial percentage of acidic liquid is 40% given the conditions -/
theorem initial_acidic_percentage (initial_volume : ℝ) (final_concentration : ℝ) (water_removed : ℝ) : 
  initial_volume = 18 →
  final_concentration = 60 →
  water_removed = 6 →
  (initial_volume * (40 / 100) = (initial_volume - water_removed) * (final_concentration / 100)) :=
by sorry

end NUMINAMATH_CALUDE_initial_acidic_percentage_l1872_187222


namespace NUMINAMATH_CALUDE_possible_values_of_a_minus_b_l1872_187278

theorem possible_values_of_a_minus_b (a b : ℝ) (ha : |a| = 7) (hb : |b| = 5) :
  {x | ∃ (a' b' : ℝ), |a'| = 7 ∧ |b'| = 5 ∧ x = a' - b'} = {2, 12, -12, -2} := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_minus_b_l1872_187278


namespace NUMINAMATH_CALUDE_baseball_team_grouping_l1872_187234

theorem baseball_team_grouping (new_players returning_players num_groups : ℕ) 
  (h1 : new_players = 4)
  (h2 : returning_players = 6)
  (h3 : num_groups = 2) :
  (new_players + returning_players) / num_groups = 5 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_grouping_l1872_187234


namespace NUMINAMATH_CALUDE_leila_toy_donation_l1872_187210

theorem leila_toy_donation (leila_bags : ℕ) (mohamed_bags : ℕ) (mohamed_toys_per_bag : ℕ) (extra_toys : ℕ) :
  leila_bags = 2 →
  mohamed_bags = 3 →
  mohamed_toys_per_bag = 19 →
  extra_toys = 7 →
  mohamed_bags * mohamed_toys_per_bag = leila_bags * (mohamed_bags * mohamed_toys_per_bag - extra_toys) / leila_bags →
  (mohamed_bags * mohamed_toys_per_bag - extra_toys) / leila_bags = 25 :=
by sorry

end NUMINAMATH_CALUDE_leila_toy_donation_l1872_187210


namespace NUMINAMATH_CALUDE_total_highlighters_count_l1872_187220

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 4

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 2

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := 5

/-- The total number of highlighters in the teacher's desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters

/-- Theorem stating that the total number of highlighters is 11 -/
theorem total_highlighters_count : total_highlighters = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_count_l1872_187220


namespace NUMINAMATH_CALUDE_compare_negative_decimals_l1872_187256

theorem compare_negative_decimals : -0.5 > -0.75 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_decimals_l1872_187256


namespace NUMINAMATH_CALUDE_correct_calculation_l1872_187271

theorem correct_calculation : ∃ (x : ℝ), x * 5 = 40 ∧ x * 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1872_187271


namespace NUMINAMATH_CALUDE_max_value_of_f_l1872_187289

/-- The quadratic function f(z) = -9z^2 + 27z + 3 -/
def f (z : ℝ) : ℝ := -9 * z^2 + 27 * z + 3

theorem max_value_of_f :
  ∃ (max : ℝ), max = 117/4 ∧ ∀ (z : ℝ), f z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1872_187289


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l1872_187240

theorem difference_of_squares_special_case : (831 : ℤ) * 831 - 830 * 832 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l1872_187240


namespace NUMINAMATH_CALUDE_larger_cube_volume_l1872_187293

/-- Proves that a cube containing 64 smaller cubes of 1 cubic inch each, with a surface area
    difference of 288 square inches between the sum of the smaller cubes' surface areas and
    the larger cube's surface area, has a volume of 64 cubic inches. -/
theorem larger_cube_volume (s : ℝ) (h1 : s > 0) :
  (s^3 : ℝ) = 64 ∧
  64 * (6 : ℝ) - 6 * s^2 = 288 →
  (s^3 : ℝ) = 64 := by sorry

end NUMINAMATH_CALUDE_larger_cube_volume_l1872_187293


namespace NUMINAMATH_CALUDE_distance_one_fourth_way_l1872_187266

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ
  apogee : ℝ

/-- Calculates the distance from the focus to a point on the orbit -/
def distanceFromFocus (orbit : EllipticalOrbit) (fraction : ℝ) : ℝ :=
  orbit.perigee + fraction * (orbit.apogee - orbit.perigee)

/-- Theorem: For the given elliptical orbit, the distance from the focus to a point
    one-fourth way from perigee to apogee is 6.75 AU -/
theorem distance_one_fourth_way (orbit : EllipticalOrbit)
    (h1 : orbit.perigee = 3)
    (h2 : orbit.apogee = 15) :
    distanceFromFocus orbit (1/4) = 6.75 := by
  sorry

#check distance_one_fourth_way

end NUMINAMATH_CALUDE_distance_one_fourth_way_l1872_187266


namespace NUMINAMATH_CALUDE_hardly_arrangements_l1872_187209

/-- The number of letters in the word "hardly" -/
def word_length : Nat := 6

/-- The number of letters to be arranged (excluding the fixed 'd') -/
def letters_to_arrange : Nat := 5

/-- Factorial function -/
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem hardly_arrangements :
  factorial letters_to_arrange = 120 :=
by sorry

end NUMINAMATH_CALUDE_hardly_arrangements_l1872_187209


namespace NUMINAMATH_CALUDE_kabadi_kho_kho_players_l1872_187216

theorem kabadi_kho_kho_players (total : ℕ) (kabadi : ℕ) (kho_kho_only : ℕ) 
  (h_total : total = 50)
  (h_kabadi : kabadi = 10)
  (h_kho_kho_only : kho_kho_only = 40) :
  total = kabadi + kho_kho_only - 0 :=
by sorry

end NUMINAMATH_CALUDE_kabadi_kho_kho_players_l1872_187216


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1872_187205

theorem min_value_of_expression (x y : ℝ) : (x^2*y - 2)^2 + (x^2 + y)^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1872_187205


namespace NUMINAMATH_CALUDE_digits_of_2_15_times_5_6_l1872_187260

/-- The number of digits in 2^15 * 5^6 is 9 -/
theorem digits_of_2_15_times_5_6 : (Nat.digits 10 (2^15 * 5^6)).length = 9 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_2_15_times_5_6_l1872_187260


namespace NUMINAMATH_CALUDE_system_solution_l1872_187290

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℚ),
    (x₁^2 - 9*y₁^2 = 36 ∧ 3*x₁ + y₁ = 6) ∧
    (x₂^2 - 9*y₂^2 = 36 ∧ 3*x₂ + y₂ = 6) ∧
    x₁ = 12/5 ∧ y₁ = -6/5 ∧ x₂ = 3 ∧ y₂ = -3 ∧
    ∀ (x y : ℚ), (x^2 - 9*y^2 = 36 ∧ 3*x + y = 6) → ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l1872_187290


namespace NUMINAMATH_CALUDE_new_person_weight_l1872_187269

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 89 kg -/
theorem new_person_weight :
  weight_of_new_person 8 3 65 = 89 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1872_187269


namespace NUMINAMATH_CALUDE_inequality_range_l1872_187243

theorem inequality_range (a : ℚ) : 
  a^7 < a^5 ∧ a^5 < a^3 ∧ a^3 < a ∧ a < a^2 ∧ a^2 < a^4 ∧ a^4 < a^6 → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1872_187243


namespace NUMINAMATH_CALUDE_smallest_prime_sum_of_three_primes_l1872_187212

-- Define a function to check if a number is prime
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is the sum of three different primes
def isSumOfThreeDifferentPrimes (n : Nat) : Prop :=
  ∃ (p q r : Nat), isPrime p ∧ isPrime q ∧ isPrime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ p + q + r = n

-- State the theorem
theorem smallest_prime_sum_of_three_primes :
  isPrime 19 ∧ 
  isSumOfThreeDifferentPrimes 19 ∧ 
  ∀ n : Nat, n < 19 → ¬(isPrime n ∧ isSumOfThreeDifferentPrimes n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_sum_of_three_primes_l1872_187212


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l1872_187270

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y + 6 = x * y) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' + 6 = x' * y' → 2 * x + y ≤ 2 * x' + y') ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ + 6 = x₀ * y₀ ∧ 2 * x₀ + y₀ = 12) :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l1872_187270


namespace NUMINAMATH_CALUDE_hypotenuse_length_l1872_187200

def right_triangle_hypotenuse (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = 36 ∧  -- perimeter condition
  (1/2) * a * b = 24 ∧  -- area condition
  a^2 + b^2 = c^2  -- Pythagorean theorem

theorem hypotenuse_length :
  ∃ a b c : ℝ, right_triangle_hypotenuse a b c ∧ c = 50/3 :=
by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l1872_187200


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l1872_187294

theorem square_difference_divided_by_nine : (121^2 - 112^2) / 9 = 233 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l1872_187294


namespace NUMINAMATH_CALUDE_hyperbola_foci_and_incenter_l1872_187252

/-- Definition of the hyperbola C -/
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := (-5, 0)

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := (5, 0)

/-- Definition of a point being on the left branch of the hyperbola -/
def on_left_branch (x y : ℝ) : Prop :=
  hyperbola x y ∧ x < 0

/-- The center of the incircle of a triangle -/
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- Definition of incenter calculation

theorem hyperbola_foci_and_incenter :
  (∀ x y : ℝ, hyperbola x y → 
    (F₁ = (-5, 0) ∧ F₂ = (5, 0))) ∧
  (∀ x y : ℝ, on_left_branch x y →
    (incenter F₁ (x, y) F₂).1 = -3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_and_incenter_l1872_187252


namespace NUMINAMATH_CALUDE_greater_number_problem_l1872_187232

theorem greater_number_problem (A B : ℕ+) : 
  (Nat.gcd A B = 11) → 
  (A * B = 363) → 
  (max A B = 33) := by
sorry

end NUMINAMATH_CALUDE_greater_number_problem_l1872_187232


namespace NUMINAMATH_CALUDE_carlos_won_one_game_l1872_187296

/-- Represents a chess player in the tournament -/
structure Player where
  wins : ℕ
  losses : ℕ

/-- Represents the chess tournament -/
structure Tournament where
  laura : Player
  mike : Player
  carlos : Player
  total_games : ℕ

/-- The number of games Carlos won in the tournament -/
def carlos_wins (t : Tournament) : ℕ :=
  t.total_games - (t.laura.wins + t.laura.losses + t.mike.wins + t.mike.losses + t.carlos.losses)

theorem carlos_won_one_game (t : Tournament) 
  (h1 : t.laura.wins = 5)
  (h2 : t.laura.losses = 4)
  (h3 : t.mike.wins = 7)
  (h4 : t.mike.losses = 2)
  (h5 : t.carlos.losses = 5)
  (h6 : t.total_games = (t.laura.wins + t.laura.losses + t.mike.wins + t.mike.losses + t.carlos.losses + carlos_wins t) / 2) :
  carlos_wins t = 1 := by
  sorry

end NUMINAMATH_CALUDE_carlos_won_one_game_l1872_187296


namespace NUMINAMATH_CALUDE_average_points_is_27_l1872_187262

/-- Represents a hockey team's record --/
structure TeamRecord where
  wins : ℕ
  ties : ℕ

/-- Calculates the points for a team given their record --/
def calculatePoints (record : TeamRecord) : ℕ :=
  2 * record.wins + record.ties

/-- The number of teams in the playoffs --/
def numTeams : ℕ := 3

/-- The records of the three playoff teams --/
def team1 : TeamRecord := ⟨12, 4⟩
def team2 : TeamRecord := ⟨13, 1⟩
def team3 : TeamRecord := ⟨8, 10⟩

/-- Theorem: The average number of points for the playoff teams is 27 --/
theorem average_points_is_27 : 
  (calculatePoints team1 + calculatePoints team2 + calculatePoints team3) / numTeams = 27 := by
  sorry


end NUMINAMATH_CALUDE_average_points_is_27_l1872_187262


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1872_187238

theorem diophantine_equation_solution :
  ∀ (x y : ℤ), 3 * x + 5 * y = 7 ↔ ∃ k : ℤ, x = 4 + 5 * k ∧ y = -1 - 3 * k :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1872_187238


namespace NUMINAMATH_CALUDE_sector_max_area_l1872_187272

/-- Given a sector with perimeter 20 cm, its area is maximized when the central angle is 2 radians, 
    and the maximum area is 25 cm². -/
theorem sector_max_area (r : ℝ) (α : ℝ) (l : ℝ) (S : ℝ) :
  0 < r → r < 10 →
  l + 2 * r = 20 →
  l = r * α →
  S = 1/2 * r * l →
  (∀ r' α' l' S', 
    0 < r' → r' < 10 →
    l' + 2 * r' = 20 →
    l' = r' * α' →
    S' = 1/2 * r' * l' →
    S' ≤ S) →
  α = 2 ∧ S = 25 := by
sorry


end NUMINAMATH_CALUDE_sector_max_area_l1872_187272


namespace NUMINAMATH_CALUDE_problem_solution_l1872_187257

theorem problem_solution (a b : ℝ) (h1 : a + b = 4) (h2 : a^2 - 2*a*b + b^2 + 2*a + 2*b = 17) : 
  ((a + 1) * (b + 1) - a * b = 5) ∧ ((a - b)^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1872_187257


namespace NUMINAMATH_CALUDE_sets_theorem_l1872_187242

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 5}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

-- State the theorem
theorem sets_theorem :
  (∀ x : ℝ, x ∈ A ∩ B (-4) ↔ 1/2 ≤ x ∧ x < 2) ∧
  (∀ x : ℝ, x ∈ A ∪ B (-4) ↔ -2 < x ∧ x ≤ 3) ∧
  (∀ a : ℝ, (B a ∩ (Aᶜ : Set ℝ) = B a) ↔ a ≥ -1/4) :=
sorry

end NUMINAMATH_CALUDE_sets_theorem_l1872_187242


namespace NUMINAMATH_CALUDE_exists_non_adjacent_divisible_l1872_187226

/-- A circular arrangement of seven natural numbers -/
def CircularArrangement := Fin 7 → ℕ+

/-- Predicate to check if one number divides another -/
def divides (a b : ℕ+) : Prop := ∃ k : ℕ+, b = a * k

/-- Two positions in the circular arrangement are adjacent -/
def adjacent (i j : Fin 7) : Prop := i = j + 1 ∨ j = i + 1 ∨ (i = 0 ∧ j = 6) ∨ (j = 0 ∧ i = 6)

/-- Two positions in the circular arrangement are non-adjacent -/
def non_adjacent (i j : Fin 7) : Prop := ¬(adjacent i j) ∧ i ≠ j

/-- The main theorem -/
theorem exists_non_adjacent_divisible (arr : CircularArrangement) 
  (h : ∀ i j : Fin 7, adjacent i j → (divides (arr i) (arr j) ∨ divides (arr j) (arr i))) :
  ∃ i j : Fin 7, non_adjacent i j ∧ (divides (arr i) (arr j) ∨ divides (arr j) (arr i)) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_adjacent_divisible_l1872_187226


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1872_187206

theorem fraction_evaluation (a b : ℝ) (h1 : a = 5) (h2 : b = 3) :
  2 / (a - b) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1872_187206


namespace NUMINAMATH_CALUDE_different_gender_choices_eq_450_l1872_187276

/-- The number of boys in the club -/
def num_boys : ℕ := 15

/-- The number of girls in the club -/
def num_girls : ℕ := 15

/-- The total number of members in the club -/
def total_members : ℕ := num_boys + num_girls

/-- The number of ways to choose a president and a vice-president of different genders -/
def different_gender_choices : ℕ := num_boys * num_girls * 2

theorem different_gender_choices_eq_450 : different_gender_choices = 450 := by
  sorry

end NUMINAMATH_CALUDE_different_gender_choices_eq_450_l1872_187276


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1872_187282

theorem trigonometric_problem (α : Real) 
  (h1 : α > π / 2 ∧ α < π) 
  (h2 : Real.sin (α / 2) + Real.cos (α / 2) = 3 * Real.sqrt 5 / 5) : 
  Real.sin α = 4 / 5 ∧ 
  Real.cos (2 * α + π / 3) = (24 * Real.sqrt 3 - 7) / 50 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1872_187282


namespace NUMINAMATH_CALUDE_min_value_xy_l1872_187288

theorem min_value_xy (x y : ℝ) (h1 : x * y + 9 = 6 * x + 2 * y) (h2 : x > 2) :
  ∃ (min_xy : ℝ), min_xy = 27 ∧ ∀ (x' y' : ℝ), x' * y' + 9 = 6 * x' + 2 * y' → x' > 2 → x' * y' ≥ min_xy := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_l1872_187288


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l1872_187291

/-- The speed of the stream in mph -/
def stream_speed : ℝ := 3.5

/-- The speed of the boat in still water in mph -/
def boat_speed : ℝ := 15

/-- The distance traveled in miles -/
def distance : ℝ := 60

/-- The time difference between upstream and downstream trips in hours -/
def time_difference : ℝ := 2

theorem stream_speed_calculation :
  (distance / (boat_speed - stream_speed)) - (distance / (boat_speed + stream_speed)) = time_difference :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l1872_187291


namespace NUMINAMATH_CALUDE_investment_amount_l1872_187286

/-- Represents the investment scenario with simple and compound interest --/
structure Investment where
  P : ℝ  -- Principal amount invested
  y : ℝ  -- Interest rate (in percentage)
  simpleInterest : ℝ  -- Simple interest earned
  compoundInterest : ℝ  -- Compound interest earned

/-- The investment satisfies the given conditions --/
def validInvestment (inv : Investment) : Prop :=
  inv.simpleInterest = inv.P * inv.y * 2 / 100 ∧
  inv.compoundInterest = inv.P * ((1 + inv.y / 100)^2 - 1) ∧
  inv.simpleInterest = 500 ∧
  inv.compoundInterest = 512.50

/-- The theorem stating that the investment amount is 5000 --/
theorem investment_amount (inv : Investment) 
  (h : validInvestment inv) : inv.P = 5000 := by
  sorry

end NUMINAMATH_CALUDE_investment_amount_l1872_187286


namespace NUMINAMATH_CALUDE_sqrt_two_four_three_two_five_two_l1872_187279

theorem sqrt_two_four_three_two_five_two : Real.sqrt (2^4 * 3^2 * 5^2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_four_three_two_five_two_l1872_187279


namespace NUMINAMATH_CALUDE_common_tangent_implies_t_value_l1872_187227

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := t * Real.log x
def g (x : ℝ) : ℝ := x^2 - 1

theorem common_tangent_implies_t_value :
  ∀ t : ℝ,
  (f t 1 = g 1) →
  (deriv (f t) 1 = deriv g 1) →
  t = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_common_tangent_implies_t_value_l1872_187227


namespace NUMINAMATH_CALUDE_find_constant_a_l1872_187228

theorem find_constant_a (t k a : ℝ) :
  (∀ x : ℝ, x^2 + 10*x + t = (x + a)^2 + k) →
  a = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_constant_a_l1872_187228


namespace NUMINAMATH_CALUDE_congruence_problem_l1872_187261

theorem congruence_problem (x : ℤ) : 
  (4 * x + 9) % 25 = 3 → (3 * x + 14) % 25 = 22 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1872_187261


namespace NUMINAMATH_CALUDE_quadratic_roots_inversely_proportional_l1872_187277

/-- 
Given a quadratic equation x^2 + px + q = 0 where q is constant and p is variable,
prove that the roots x₁ and x₂ are inversely proportional to each other.
-/
theorem quadratic_roots_inversely_proportional 
  (p q : ℝ) (x₁ x₂ : ℝ) (h_const : q ≠ 0) :
  (x₁^2 + p*x₁ + q = 0) → (x₂^2 + p*x₂ + q = 0) → 
  ∃ (k : ℝ), k ≠ 0 ∧ x₁ * x₂ = k :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_inversely_proportional_l1872_187277


namespace NUMINAMATH_CALUDE_old_supervisor_salary_l1872_187247

/-- Proves that the old supervisor's salary was $870 given the problem conditions -/
theorem old_supervisor_salary
  (num_workers : ℕ)
  (initial_average : ℚ)
  (new_average : ℚ)
  (new_supervisor_salary : ℚ)
  (h_num_workers : num_workers = 8)
  (h_initial_average : initial_average = 430)
  (h_new_average : new_average = 390)
  (h_new_supervisor_salary : new_supervisor_salary = 510)
  : ∃ (old_supervisor_salary : ℚ),
    (num_workers + 1) * initial_average = num_workers * new_average + old_supervisor_salary
    ∧ old_supervisor_salary = 870 :=
by sorry

end NUMINAMATH_CALUDE_old_supervisor_salary_l1872_187247


namespace NUMINAMATH_CALUDE_sheep_distribution_l1872_187263

theorem sheep_distribution (A B C D : ℕ) : 
  C = D + 10 ∧ 
  (3 * C) / 4 + A = B + C / 4 + D ∧
  (∃ (x : ℕ), x > 0 ∧ 
    (2 * A) / 3 + (B + A / 3 - (B + A / 3) / 4) + 
    (C + (B + A / 3) / 4 - (C + (B + A / 3) / 4) / 5) + 
    (D + (C + (B + A / 3) / 4) / 5 + x) = 
    4 * ((2 * A) / 3 + (B + A / 3 - (B + A / 3) / 4) + x)) →
  A = 60 ∧ B = 50 ∧ C = 40 ∧ D = 30 := by
sorry

end NUMINAMATH_CALUDE_sheep_distribution_l1872_187263


namespace NUMINAMATH_CALUDE_part_one_part_two_l1872_187259

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

-- Part 1
theorem part_one : 
  (Set.univ \ B (1/2)) ∩ A (1/2) = {x : ℝ | 9/4 ≤ x ∧ x < 5/2} := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B a) ↔ a ∈ Set.Icc (-1/2) ((3 - Real.sqrt 5) / 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1872_187259


namespace NUMINAMATH_CALUDE_chord_equation_l1872_187273

/-- The equation of a line containing a chord of the ellipse x^2/2 + y^2 = 1,
    passing through and bisected by the point (1/2, 1/2) -/
theorem chord_equation (x y : ℝ) : 
  (∃ (x1 y1 x2 y2 : ℝ),
    -- Ellipse equation
    x1^2 / 2 + y1^2 = 1 ∧ 
    x2^2 / 2 + y2^2 = 1 ∧
    -- Point P is on the ellipse
    (1/2)^2 / 2 + (1/2)^2 = 1 ∧
    -- P is the midpoint of the chord
    (x1 + x2) / 2 = 1/2 ∧
    (y1 + y2) / 2 = 1/2 ∧
    -- The line passes through P
    y - 1/2 = (y - 1/2) / (x - 1/2) * (x - 1/2)) →
  2*x + 4*y - 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l1872_187273


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1872_187248

theorem rectangle_diagonal (a b : ℝ) (h_perimeter : 2 * (a + b) = 178) (h_area : a * b = 1848) :
  Real.sqrt (a^2 + b^2) = 65 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1872_187248


namespace NUMINAMATH_CALUDE_closest_integer_to_expression_l1872_187217

theorem closest_integer_to_expression : ∃ n : ℤ, 
  n = round ((3/2 : ℚ) * (4/9 : ℚ) + (7/2 : ℚ)) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_expression_l1872_187217


namespace NUMINAMATH_CALUDE_total_handshakes_l1872_187213

-- Define the number of people in each group
def group_a : Nat := 25  -- people who all know each other
def group_b : Nat := 10  -- people who know no one
def group_c : Nat := 5   -- people who only know each other

-- Define the total number of people
def total_people : Nat := group_a + group_b + group_c

-- Define the function to calculate handshakes between two groups
def handshakes_between (group1 : Nat) (group2 : Nat) : Nat := group1 * group2

-- Define the function to calculate handshakes within a group
def handshakes_within (group : Nat) : Nat := group * (group - 1) / 2

-- Theorem statement
theorem total_handshakes : 
  handshakes_between group_a group_b + 
  handshakes_between group_a group_c + 
  handshakes_between group_b group_c + 
  handshakes_within group_b = 470 := by
  sorry

end NUMINAMATH_CALUDE_total_handshakes_l1872_187213


namespace NUMINAMATH_CALUDE_notebooks_given_to_paula_notebooks_given_to_paula_is_five_l1872_187215

theorem notebooks_given_to_paula (gerald_notebooks : ℕ) (jack_initial_extra : ℕ) 
  (given_to_mike : ℕ) (jack_remaining : ℕ) : ℕ :=
  let jack_initial := gerald_notebooks + jack_initial_extra
  let jack_after_paula := jack_remaining + given_to_mike
  let given_to_paula := jack_initial - jack_after_paula
  given_to_paula

theorem notebooks_given_to_paula_is_five :
  notebooks_given_to_paula 8 13 6 10 = 5 := by sorry

end NUMINAMATH_CALUDE_notebooks_given_to_paula_notebooks_given_to_paula_is_five_l1872_187215


namespace NUMINAMATH_CALUDE_Q_formula_l1872_187274

def T (n : ℕ) : ℕ := (n * (n + 1)) / 2

def Q (n : ℕ) : ℚ :=
  if n < 2 then 0
  else Finset.prod (Finset.range (n - 1)) (fun k => (T (k + 2) : ℚ) / ((T (k + 3) : ℚ) - 1))

theorem Q_formula (n : ℕ) (h : n ≥ 2) : Q n = 2 / (n + 3) := by
  sorry

end NUMINAMATH_CALUDE_Q_formula_l1872_187274


namespace NUMINAMATH_CALUDE_number_interval_l1872_187264

theorem number_interval (x : ℝ) (h : x = (1/x) * (-x) + 4) : 2 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_number_interval_l1872_187264


namespace NUMINAMATH_CALUDE_photos_per_album_l1872_187280

theorem photos_per_album (total_photos : ℕ) (num_albums : ℕ) (h1 : total_photos = 180) (h2 : num_albums = 9) :
  total_photos / num_albums = 20 := by
sorry

end NUMINAMATH_CALUDE_photos_per_album_l1872_187280


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1872_187221

-- Part 1
theorem problem_1 : 8 - (-4) / (2^2) * 3 = 11 := by sorry

-- Part 2
theorem problem_2 (x : ℝ) : 2 * x^2 + 3 * (2*x - x^2) = -x^2 + 6*x := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1872_187221


namespace NUMINAMATH_CALUDE_tyler_scissors_purchase_l1872_187268

theorem tyler_scissors_purchase
  (initial_amount : ℕ)
  (scissors_cost : ℕ)
  (eraser_count : ℕ)
  (eraser_cost : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 100)
  (h2 : scissors_cost = 5)
  (h3 : eraser_count = 10)
  (h4 : eraser_cost = 4)
  (h5 : remaining_amount = 20) :
  ∃ (scissors_count : ℕ), 
    scissors_count * scissors_cost + eraser_count * eraser_cost = initial_amount - remaining_amount ∧
    scissors_count = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_tyler_scissors_purchase_l1872_187268


namespace NUMINAMATH_CALUDE_log_expression_equality_l1872_187250

theorem log_expression_equality : 
  Real.log 3 / Real.log 2 * (Real.log 4 / Real.log 3) + 
  (Real.log 24 / Real.log 2 - Real.log 6 / Real.log 2 + 6) ^ (2/3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l1872_187250


namespace NUMINAMATH_CALUDE_exists_strictly_increasing_set_function_l1872_187297

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define the power set of positive integers
def PowerSetOfPositiveIntegers : Set (Set ℕ) :=
  {X : Set ℕ | X ⊆ PositiveIntegers}

-- State the theorem
theorem exists_strictly_increasing_set_function :
  ∃ (f : ℝ → Set ℕ),
    (∀ x, f x ∈ PowerSetOfPositiveIntegers) ∧
    (∀ a b, a < b → f a ⊂ f b ∧ f a ≠ f b) :=
sorry

end NUMINAMATH_CALUDE_exists_strictly_increasing_set_function_l1872_187297


namespace NUMINAMATH_CALUDE_jill_watching_time_l1872_187254

/-- The total time Jill spent watching shows -/
def total_time (first_show_duration : ℕ) (multiplier : ℕ) : ℕ :=
  first_show_duration + first_show_duration * multiplier

/-- Proof that Jill spent 150 minutes watching shows -/
theorem jill_watching_time : total_time 30 4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_jill_watching_time_l1872_187254


namespace NUMINAMATH_CALUDE_greatest_4digit_base7_divisible_by_7_l1872_187204

/-- Converts a base 7 number to decimal --/
def toDecimal (n : List Nat) : Nat :=
  n.reverse.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Checks if a number is divisible by 7 --/
def isDivisibleBy7 (n : Nat) : Bool :=
  n % 7 = 0

/-- Converts a decimal number to base 7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Checks if a list represents a 4-digit base 7 number --/
def is4DigitBase7 (n : List Nat) : Bool :=
  n.length = 4 && n.all (· < 7) && n.head! ≠ 0

theorem greatest_4digit_base7_divisible_by_7 :
  let n := [6, 6, 6, 0]
  is4DigitBase7 n ∧
  isDivisibleBy7 (toDecimal n) ∧
  ∀ m, is4DigitBase7 m → isDivisibleBy7 (toDecimal m) → toDecimal m ≤ toDecimal n :=
by sorry

end NUMINAMATH_CALUDE_greatest_4digit_base7_divisible_by_7_l1872_187204


namespace NUMINAMATH_CALUDE_trigonometric_calculation_quadratic_equation_solution_l1872_187218

-- Problem 1
theorem trigonometric_calculation :
  3 * Real.tan (30 * π / 180) - Real.tan (45 * π / 180)^2 + 2 * Real.sin (60 * π / 180) = 2 * Real.sqrt 3 - 1 := by
  sorry

-- Problem 2
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ (3*x - 1)*(x + 2) - (11*x - 4)
  (∃ x : ℝ, f x = 0) ↔ (f ((3 + Real.sqrt 3) / 3) = 0 ∧ f ((3 - Real.sqrt 3) / 3) = 0) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_calculation_quadratic_equation_solution_l1872_187218


namespace NUMINAMATH_CALUDE_no_consecutive_even_fibonacci_l1872_187275

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem no_consecutive_even_fibonacci :
  ∀ n : ℕ, ¬(Even (fibonacci n) ∧ Even (fibonacci (n + 1))) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_even_fibonacci_l1872_187275


namespace NUMINAMATH_CALUDE_blue_pill_cost_l1872_187255

def treatment_duration : ℕ := 21 -- 3 weeks * 7 days

def daily_blue_pills : ℕ := 2
def daily_orange_pills : ℕ := 1

def total_cost : ℕ := 966

theorem blue_pill_cost (orange_pill_cost : ℕ) 
  (h1 : orange_pill_cost + 2 = 16) 
  (h2 : (daily_blue_pills * (orange_pill_cost + 2) + daily_orange_pills * orange_pill_cost) * treatment_duration = total_cost) : 
  orange_pill_cost + 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_blue_pill_cost_l1872_187255


namespace NUMINAMATH_CALUDE_magic_square_d_plus_e_l1872_187223

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  sum_eq : sum = 30 + e + 15
         ∧ sum = 10 + c + d
         ∧ sum = a + 25 + b
         ∧ sum = 30 + 10 + a
         ∧ sum = e + c + 25
         ∧ sum = 15 + d + b
         ∧ sum = 30 + c + b
         ∧ sum = a + c + e
         ∧ sum = 15 + 25 + a

theorem magic_square_d_plus_e (sq : MagicSquare) : sq.d + sq.e = 25 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_d_plus_e_l1872_187223


namespace NUMINAMATH_CALUDE_interest_years_satisfies_equation_l1872_187236

/-- The number of years that satisfies the compound and simple interest difference equation -/
def interest_years : ℕ := 2

/-- The principal amount in rupees -/
def principal : ℚ := 3600

/-- The annual interest rate as a decimal -/
def rate : ℚ := 1/10

/-- The difference between compound and simple interest in rupees -/
def interest_difference : ℚ := 36

/-- The equation that relates the number of years to the interest difference -/
def interest_equation (n : ℕ) : Prop :=
  (1 + rate) ^ n - 1 - rate * n = interest_difference / principal

theorem interest_years_satisfies_equation : 
  interest_equation interest_years :=
sorry

end NUMINAMATH_CALUDE_interest_years_satisfies_equation_l1872_187236


namespace NUMINAMATH_CALUDE_intersection_sum_l1872_187267

/-- Given two lines y = mx + 3 and y = 4x + b intersecting at (8, 14),
    where m and b are constants, prove that b + m = -133/8 -/
theorem intersection_sum (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + 3 ↔ y = 4 * x + b) → 
  (14 : ℚ) = m * 8 + 3 → 
  (14 : ℚ) = 4 * 8 + b → 
  b + m = -133/8 := by sorry

end NUMINAMATH_CALUDE_intersection_sum_l1872_187267
