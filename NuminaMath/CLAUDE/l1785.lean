import Mathlib

namespace NUMINAMATH_CALUDE_inequalities_proof_l1785_178535

theorem inequalities_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (a * b > b * c) ∧ (2022^(a - c) + a > 2022^(b - c) + b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1785_178535


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l1785_178588

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 3*m - 1}

-- Theorem for part (1)
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x | -2 ≤ x ∧ x ≤ 2}) ∧
  (A ∪ B 3 = {x | -3 ≤ x ∧ x ≤ 8}) := by sorry

-- Theorem for part (2)
theorem intersection_equals_B_iff_m_leq_1 :
  ∀ m : ℝ, (A ∩ B m = B m) ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l1785_178588


namespace NUMINAMATH_CALUDE_unique_solution_l1785_178591

-- Define the colors
inductive Color
| Red
| Blue

-- Define a structure for clothing
structure Clothing :=
  (tshirt : Color)
  (shorts : Color)

-- Define the children
structure Children :=
  (alyna : Clothing)
  (bohdan : Clothing)
  (vika : Clothing)
  (grysha : Clothing)

-- Define the conditions
def satisfiesConditions (c : Children) : Prop :=
  (c.alyna.tshirt = Color.Red) ∧
  (c.bohdan.tshirt = Color.Red) ∧
  (c.alyna.shorts ≠ c.bohdan.shorts) ∧
  (c.vika.tshirt ≠ c.grysha.tshirt) ∧
  (c.vika.shorts = Color.Blue) ∧
  (c.grysha.shorts = Color.Blue) ∧
  (c.alyna.tshirt ≠ c.vika.tshirt) ∧
  (c.alyna.shorts ≠ c.vika.shorts)

-- Define the correct answer
def correctAnswer : Children :=
  { alyna := { tshirt := Color.Red, shorts := Color.Red },
    bohdan := { tshirt := Color.Red, shorts := Color.Blue },
    vika := { tshirt := Color.Blue, shorts := Color.Blue },
    grysha := { tshirt := Color.Red, shorts := Color.Blue } }

-- Theorem statement
theorem unique_solution :
  ∀ c : Children, satisfiesConditions c → c = correctAnswer :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1785_178591


namespace NUMINAMATH_CALUDE_fraction_inequivalence_l1785_178528

theorem fraction_inequivalence :
  ∃ k : ℝ, k ≠ 0 ∧ k ≠ -1 ∧ (3 * k + 9) / (4 * k + 4) ≠ 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequivalence_l1785_178528


namespace NUMINAMATH_CALUDE_trapezoid_properties_l1785_178510

/-- Represents a trapezoid with side lengths and angles -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ

/-- The main theorem about trapezoid properties -/
theorem trapezoid_properties (t : Trapezoid) :
  (t.a = t.d * Real.cos t.α + t.b * Real.cos t.β - t.c * Real.cos (t.β + t.γ)) ∧
  (t.a = t.d * Real.cos t.α + t.b * Real.cos t.β - t.c * Real.cos (t.α + t.δ)) ∧
  (t.a * Real.sin t.α = t.c * Real.sin t.δ + t.b * Real.sin (t.α + t.β)) ∧
  (t.a * Real.sin t.β = t.c * Real.sin t.γ + t.d * Real.sin (t.α + t.β)) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_properties_l1785_178510


namespace NUMINAMATH_CALUDE_largest_fraction_l1785_178502

theorem largest_fraction (a b c d e : ℚ) : 
  a = 2/5 → b = 3/6 → c = 5/10 → d = 7/15 → e = 8/20 →
  (b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e) ∧
  (c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e) ∧
  b = c := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l1785_178502


namespace NUMINAMATH_CALUDE_factorial_sum_representations_l1785_178566

/-- For any natural number n ≥ 4, there exist at least n! ways to write n! as a sum of elements
    from the set {1!, 2!, ..., (n-1)!}, where each element can be used multiple times. -/
theorem factorial_sum_representations (n : ℕ) (h : n ≥ 4) :
  ∃ (ways : ℕ), ways ≥ n! ∧
    ∀ (representation : List ℕ),
      (∀ k ∈ representation, k ∈ Finset.range n ∧ k > 0) →
      representation.sum = n! →
      (ways : ℕ) ≥ (representation.map Nat.factorial).sum :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_representations_l1785_178566


namespace NUMINAMATH_CALUDE_a_value_l1785_178544

def A : Set ℝ := {0, 2}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem a_value (a : ℝ) : A ∪ B a = {0, 1, 2, 4} → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l1785_178544


namespace NUMINAMATH_CALUDE_raspberry_green_grape_difference_l1785_178505

def fruit_salad (green_grapes raspberries red_grapes : ℕ) : Prop :=
  green_grapes + raspberries + red_grapes = 102 ∧
  red_grapes = 67 ∧
  red_grapes = 3 * green_grapes + 7 ∧
  raspberries < green_grapes

theorem raspberry_green_grape_difference 
  (green_grapes raspberries red_grapes : ℕ) :
  fruit_salad green_grapes raspberries red_grapes →
  green_grapes - raspberries = 5 := by
sorry

end NUMINAMATH_CALUDE_raspberry_green_grape_difference_l1785_178505


namespace NUMINAMATH_CALUDE_granger_cisco_spots_l1785_178557

/-- The number of spots Rover has -/
def rover_spots : ℕ := 46

/-- The number of spots Cisco has -/
def cisco_spots : ℕ := rover_spots / 2 - 5

/-- The number of spots Granger has -/
def granger_spots : ℕ := 5 * cisco_spots

/-- The total number of spots Granger and Cisco have combined -/
def total_spots : ℕ := granger_spots + cisco_spots

theorem granger_cisco_spots : total_spots = 108 := by
  sorry

end NUMINAMATH_CALUDE_granger_cisco_spots_l1785_178557


namespace NUMINAMATH_CALUDE_sandwich_cost_is_three_l1785_178531

/-- The cost of a sandwich given the total cost and number of items. -/
def sandwich_cost (total_cost : ℚ) (water_cost : ℚ) (num_sandwiches : ℕ) : ℚ :=
  (total_cost - water_cost) / num_sandwiches

/-- Theorem stating that the cost of each sandwich is 3 given the problem conditions. -/
theorem sandwich_cost_is_three :
  sandwich_cost 11 2 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_is_three_l1785_178531


namespace NUMINAMATH_CALUDE_ball_probability_l1785_178551

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 60)
  (h2 : white = 22)
  (h3 : green = 18)
  (h4 : yellow = 2)
  (h5 : red = 15)
  (h6 : purple = 3)
  (h7 : total = white + green + yellow + red + purple) :
  (white + green + yellow : ℚ) / total = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l1785_178551


namespace NUMINAMATH_CALUDE_count_missed_toddlers_l1785_178580

/-- The number of toddlers Bill missed -/
def toddlers_missed (total_count : ℕ) (double_counted : ℕ) (actual_toddlers : ℕ) : ℕ :=
  actual_toddlers - (total_count - double_counted)

/-- Theorem stating that the number of toddlers Bill missed is equal to
    the actual number of toddlers minus the number he actually counted -/
theorem count_missed_toddlers 
  (total_count : ℕ) (double_counted : ℕ) (actual_toddlers : ℕ) :
  toddlers_missed total_count double_counted actual_toddlers = 
  actual_toddlers - (total_count - double_counted) :=
by
  sorry

end NUMINAMATH_CALUDE_count_missed_toddlers_l1785_178580


namespace NUMINAMATH_CALUDE_min_odd_counties_big_island_l1785_178503

/-- Represents a rectangular county with a diagonal road -/
structure County where
  has_diagonal_road : Bool

/-- Represents an island configuration -/
structure Island where
  counties : List County
  is_valid : Bool

/-- Checks if a given number of counties can form a valid Big Island configuration -/
def is_valid_big_island (n : Nat) : Prop :=
  ∃ (island : Island),
    island.counties.length = n ∧
    n % 2 = 1 ∧
    island.is_valid = true

/-- Theorem stating that 9 is the minimum odd number of counties for a valid Big Island -/
theorem min_odd_counties_big_island :
  (∀ k, k < 9 → k % 2 = 1 → ¬ is_valid_big_island k) ∧
  is_valid_big_island 9 := by
  sorry

end NUMINAMATH_CALUDE_min_odd_counties_big_island_l1785_178503


namespace NUMINAMATH_CALUDE_perimeter_region_with_270_degree_arc_l1785_178533

/-- The perimeter of a region formed by two radii and a 270° arc in a circle -/
theorem perimeter_region_with_270_degree_arc (r : ℝ) (h : r = 7) :
  2 * r + (3/4) * (2 * Real.pi * r) = 14 + (21 * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_perimeter_region_with_270_degree_arc_l1785_178533


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1785_178574

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 24 * x - 4 * y^2 + 16 * y + 44 = 0

/-- The distance between vertices of the hyperbola -/
def vertex_distance : ℝ := 2

/-- Theorem stating that the distance between vertices of the given hyperbola is 2 -/
theorem hyperbola_vertex_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola_equation x₁ y₁ ∧
    hyperbola_equation x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    ∀ (x y : ℝ), hyperbola_equation x y → 
      (x - x₁)^2 + (y - y₁)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 ∧
      (x - x₂)^2 + (y - y₂)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = vertex_distance^2 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1785_178574


namespace NUMINAMATH_CALUDE_egyptian_fraction_decomposition_l1785_178527

theorem egyptian_fraction_decomposition (n : ℕ) (h : n ≥ 2) :
  (2 : ℚ) / (2 * n + 1) = 1 / (n + 1) + 1 / ((n + 1) * (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_decomposition_l1785_178527


namespace NUMINAMATH_CALUDE_ellipse_properties_l1785_178552

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_eccentricity : (a^2 - b^2) / a^2 = 3/4
  h_point_on_ellipse : 2/a^2 + 1/(2*b^2) = 1

/-- The theorem statement -/
theorem ellipse_properties (C : Ellipse) :
  C.a^2 = 4 ∧ C.b^2 = 2 ∧
  (∀ (P Q : ℝ × ℝ) (l : Set (ℝ × ℝ)),
    P ∈ l ∧ Q ∈ l ∧
    P.1^2/4 + P.2^2 = 1 ∧
    Q.1^2/4 + Q.2^2 = 1 ∧
    P.1 * Q.1 + P.2 * Q.2 = 0 →
    1/2 * abs (P.1 * Q.2 - P.2 * Q.1) ≥ 4/5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1785_178552


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l1785_178513

/-- The market value of a machine after two years, given its initial price and annual depreciation rate. -/
def market_value_after_two_years (initial_price : ℝ) (depreciation_rate : ℝ) : ℝ :=
  initial_price * (1 - depreciation_rate)^2

/-- Theorem stating that a machine purchased for $8000 with a 10% annual depreciation rate
    will have a market value of $6480 after two years. -/
theorem machine_value_after_two_years :
  market_value_after_two_years 8000 0.1 = 6480 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_after_two_years_l1785_178513


namespace NUMINAMATH_CALUDE_distance_between_points_l1785_178563

theorem distance_between_points (d : ℝ) : 
  (∃ (x : ℝ), d / 2 + x = d - 5) ∧ 
  (d / 2 + d / 2 - 45 / 8 = 45 / 8) → 
  d = 90 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l1785_178563


namespace NUMINAMATH_CALUDE_least_subtrahend_l1785_178593

theorem least_subtrahend (n : ℕ) (h : n = 427398) : 
  ∃! x : ℕ, x ≤ n ∧ 
  (∀ y : ℕ, y < x → ¬((n - y) % 17 = 0 ∧ (n - y) % 19 = 0 ∧ (n - y) % 31 = 0)) ∧
  (n - x) % 17 = 0 ∧ (n - x) % 19 = 0 ∧ (n - x) % 31 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_l1785_178593


namespace NUMINAMATH_CALUDE_remainder_1493825_div_6_l1785_178569

theorem remainder_1493825_div_6 : (1493825 % 6 = 5) := by
  sorry

end NUMINAMATH_CALUDE_remainder_1493825_div_6_l1785_178569


namespace NUMINAMATH_CALUDE_school_population_l1785_178577

theorem school_population (total : ℚ) 
  (h1 : (2 : ℚ) / 3 * total = total - (1 : ℚ) / 3 * total) 
  (h2 : (1 : ℚ) / 10 * ((1 : ℚ) / 3 * total) = (1 : ℚ) / 3 * total - 90) 
  (h3 : (9 : ℚ) / 10 * ((1 : ℚ) / 3 * total) = 90) : 
  total = 300 := by sorry

end NUMINAMATH_CALUDE_school_population_l1785_178577


namespace NUMINAMATH_CALUDE_tan_585_degrees_l1785_178515

theorem tan_585_degrees : Real.tan (585 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_585_degrees_l1785_178515


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1785_178523

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I * (1 - a) = -a - 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1785_178523


namespace NUMINAMATH_CALUDE_checkerboard_diagonal_squares_l1785_178511

theorem checkerboard_diagonal_squares (m n : ℕ) (hm : m = 91) (hn : n = 28) :
  m + n - Nat.gcd m n = 112 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_diagonal_squares_l1785_178511


namespace NUMINAMATH_CALUDE_smaller_part_area_l1785_178516

/-- The area of the smaller part of a field satisfying given conditions -/
theorem smaller_part_area (total_area : ℝ) (smaller_area larger_area : ℝ) : 
  total_area = 1800 →
  smaller_area + larger_area = total_area →
  larger_area - smaller_area = (smaller_area + larger_area) / 6 →
  smaller_area = 750 := by
  sorry

end NUMINAMATH_CALUDE_smaller_part_area_l1785_178516


namespace NUMINAMATH_CALUDE_letter_150_is_z_l1785_178567

def repeating_sequence : ℕ → Char
  | n => if n % 3 = 1 then 'X' else if n % 3 = 2 then 'Y' else 'Z'

theorem letter_150_is_z : repeating_sequence 150 = 'Z' := by
  sorry

end NUMINAMATH_CALUDE_letter_150_is_z_l1785_178567


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1785_178582

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  2 = Nat.minFac (3^13 + 9^11) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1785_178582


namespace NUMINAMATH_CALUDE_train_passing_platform_l1785_178537

/-- Calculates the time for a train to pass a platform given its length, speed, and the platform length -/
theorem train_passing_platform 
  (train_length : Real) 
  (time_to_cross_tree : Real) 
  (platform_length : Real) : 
  train_length = 1200 ∧ 
  time_to_cross_tree = 120 ∧ 
  platform_length = 1000 → 
  (train_length + platform_length) / (train_length / time_to_cross_tree) = 220 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_platform_l1785_178537


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l1785_178571

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l1785_178571


namespace NUMINAMATH_CALUDE_angle_around_point_l1785_178532

/-- 
Given three angles around a point in a plane, where one angle is 130°, 
and one of the other angles (y) is 30° more than the third angle (x), 
prove that x = 100° and y = 130°.
-/
theorem angle_around_point (x y : ℝ) : 
  x + y + 130 = 360 →   -- Sum of angles around a point is 360°
  y = x + 30 →          -- y is 30° more than x
  x = 100 ∧ y = 130 :=  -- Conclusion: x = 100° and y = 130°
by sorry

end NUMINAMATH_CALUDE_angle_around_point_l1785_178532


namespace NUMINAMATH_CALUDE_max_product_constraint_l1785_178550

theorem max_product_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = 1) :
  x * y ≤ 1/16 := by
sorry

end NUMINAMATH_CALUDE_max_product_constraint_l1785_178550


namespace NUMINAMATH_CALUDE_triangle_max_area_triangle_area_eight_exists_l1785_178508

/-- Triangle with sides a, b, c and area S -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  h1 : 4 * S = a^2 - (b - c)^2
  h2 : b + c = 8
  h3 : a > 0 ∧ b > 0 ∧ c > 0 -- Ensuring positive side lengths

/-- The maximum area of a triangle satisfying the given conditions is 8 -/
theorem triangle_max_area (t : Triangle) : t.S ≤ 8 := by
  sorry

/-- There exists a triangle satisfying the conditions with area equal to 8 -/
theorem triangle_area_eight_exists : ∃ t : Triangle, t.S = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_triangle_area_eight_exists_l1785_178508


namespace NUMINAMATH_CALUDE_angle_x_is_72_degrees_l1785_178518

-- Define a regular pentagon
structure RegularPentagon where
  -- All sides are equal (implied by regularity)
  -- All angles are equal (implied by regularity)

-- Define the enclosing structure
structure EnclosingStructure where
  pentagon : RegularPentagon
  -- Squares and triangles enclose the pentagon (implied by the structure)

-- Define the angle x formed by two squares and the pentagon
def angle_x (e : EnclosingStructure) : ℝ := sorry

-- Theorem statement
theorem angle_x_is_72_degrees (e : EnclosingStructure) : 
  angle_x e = 72 := by sorry

end NUMINAMATH_CALUDE_angle_x_is_72_degrees_l1785_178518


namespace NUMINAMATH_CALUDE_pythagorean_triple_with_ratio_exists_l1785_178573

theorem pythagorean_triple_with_ratio_exists (k : ℚ) (hk : k > 1) :
  ∃ (a b c : ℕ+), (a.val^2 + b.val^2 = c.val^2) ∧ ((a.val + c.val) / b.val : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_with_ratio_exists_l1785_178573


namespace NUMINAMATH_CALUDE_right_triangle_min_perimeter_l1785_178509

theorem right_triangle_min_perimeter (a b c : ℝ) (h_area : a * b / 2 = 1) (h_right : a^2 + b^2 = c^2) :
  a + b + c ≥ 2 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_min_perimeter_l1785_178509


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_l1785_178583

theorem integral_sqrt_one_minus_x_squared : ∫ x in (-1)..(1), Real.sqrt (1 - x^2) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_l1785_178583


namespace NUMINAMATH_CALUDE_expression_evaluation_l1785_178596

theorem expression_evaluation :
  let f (x : ℤ) := 8 * x^2 - (x - 2) * (3 * x + 1) - 2 * (x + 1) * (x - 1)
  f (-2) = 6 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1785_178596


namespace NUMINAMATH_CALUDE_f_properties_l1785_178598

def f (x : ℝ) : ℝ := -(x - 2)^2 + 4

theorem f_properties :
  (∀ x : ℝ, ∃ y : ℝ, f x = y) ∧
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧
  (∀ x y : ℝ, x < y ∧ y < 0 → f x < f y) ∧
  (∀ x : ℝ, f x ≤ 4) ∧
  (∃ x : ℝ, f x = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1785_178598


namespace NUMINAMATH_CALUDE_hash_four_two_l1785_178553

-- Define the # operation
def hash (a b : ℝ) : ℝ := (a^2 + b^2) * (a - b)

-- Theorem statement
theorem hash_four_two : hash 4 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_hash_four_two_l1785_178553


namespace NUMINAMATH_CALUDE_farm_animals_feet_count_l1785_178564

theorem farm_animals_feet_count (total_heads : ℕ) (hen_count : ℕ) : 
  total_heads = 48 → hen_count = 28 → 
  (hen_count * 2 + (total_heads - hen_count) * 4 : ℕ) = 136 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_feet_count_l1785_178564


namespace NUMINAMATH_CALUDE_contradiction_assumption_l1785_178572

theorem contradiction_assumption (a b c : ℝ) : 
  (¬(a > 0 ∧ b > 0 ∧ c > 0)) ↔ (¬(a > 0) ∨ ¬(b > 0) ∨ ¬(c > 0)) :=
by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l1785_178572


namespace NUMINAMATH_CALUDE_stock_loss_percentage_l1785_178594

theorem stock_loss_percentage (total_stock : ℝ) (profit_percentage : ℝ) (profit_portion : ℝ) (overall_loss : ℝ) :
  total_stock = 12500 →
  profit_percentage = 10 →
  profit_portion = 20 →
  overall_loss = 250 →
  ∃ (L : ℝ),
    overall_loss = (1 - profit_portion / 100) * total_stock * (L / 100) - (profit_portion / 100) * total_stock * (profit_percentage / 100) ∧
    L = 5 := by
  sorry

end NUMINAMATH_CALUDE_stock_loss_percentage_l1785_178594


namespace NUMINAMATH_CALUDE_solution_system_equations_l1785_178512

theorem solution_system_equations :
  ∃ (a b : ℝ), 
    (a * 2 + b * 1 = 7 ∧ a * 2 - b * 1 = 1) → 
    (a - b = -1) :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l1785_178512


namespace NUMINAMATH_CALUDE_missing_month_sale_correct_grocer_sale_problem_l1785_178500

/-- Calculates the missing month's sale given sales data for 5 months and the average sale --/
def calculate_missing_month_sale (sale1 sale2 sale4 sale5 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Proves that the calculated missing month's sale satisfies the average sale condition --/
theorem missing_month_sale_correct 
  (sale1 sale2 sale4 sale5 sale6 average_sale : ℕ) :
  let sale3 := calculate_missing_month_sale sale1 sale2 sale4 sale5 sale6 average_sale
  (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

/-- Applies the theorem to the specific problem values --/
theorem grocer_sale_problem :
  let sale1 : ℕ := 5921
  let sale2 : ℕ := 5468
  let sale4 : ℕ := 6088
  let sale5 : ℕ := 6433
  let sale6 : ℕ := 5922
  let average_sale : ℕ := 5900
  let sale3 := calculate_missing_month_sale sale1 sale2 sale4 sale5 sale6 average_sale
  sale3 = 5568 ∧ (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

end NUMINAMATH_CALUDE_missing_month_sale_correct_grocer_sale_problem_l1785_178500


namespace NUMINAMATH_CALUDE_equation_solution_l1785_178534

theorem equation_solution : ∃ x : ℝ, 14*x + 15*x + 18*x + 11 = 152 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1785_178534


namespace NUMINAMATH_CALUDE_rectangle_arrangement_exists_l1785_178579

theorem rectangle_arrangement_exists : ∃ (a b c d : ℕ+), 
  (a * b + c * d = 81) ∧ 
  ((2 * (a + b) = 4 * (c + d)) ∨ (4 * (a + b) = 2 * (c + d))) :=
sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_exists_l1785_178579


namespace NUMINAMATH_CALUDE_percent_relation_l1785_178504

theorem percent_relation (x y z : ℝ) (h1 : x = 1.3 * y) (h2 : y = 0.6 * z) : 
  x = 0.78 * z := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l1785_178504


namespace NUMINAMATH_CALUDE_triangle_problem_l1785_178554

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (a * Real.sin (2 * B) = Real.sqrt 3 * b * Real.sin A) →
  (Real.cos A = 1 / 3) →
  (B = π / 6) ∧
  (Real.sin C = (2 * Real.sqrt 6 + 1) / 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1785_178554


namespace NUMINAMATH_CALUDE_probability_two_females_l1785_178521

theorem probability_two_females (total : Nat) (females : Nat) (chosen : Nat) :
  total = 8 →
  females = 5 →
  chosen = 2 →
  (Nat.choose females chosen : ℚ) / (Nat.choose total chosen : ℚ) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_l1785_178521


namespace NUMINAMATH_CALUDE_surface_area_of_problem_structure_l1785_178549

/-- Represents a structure made of unit cubes -/
structure CubeStructure where
  base : Nat × Nat × Nat  -- dimensions of the base cube
  stacked : Nat  -- number of cubes stacked on top
  total : Nat  -- total number of cubes

/-- Calculates the surface area of a cube structure -/
def surfaceArea (cs : CubeStructure) : Nat :=
  sorry

/-- The specific cube structure in the problem -/
def problemStructure : CubeStructure :=
  { base := (2, 2, 2),
    stacked := 4,
    total := 12 }

theorem surface_area_of_problem_structure :
  surfaceArea problemStructure = 32 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_problem_structure_l1785_178549


namespace NUMINAMATH_CALUDE_expression_simplification_l1785_178570

theorem expression_simplification (a : ℚ) (h : a = -2) : 
  ((a + 7) / (a - 1) - 2 / (a + 1)) / ((a^2 + 3*a) / (a^2 - 1)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1785_178570


namespace NUMINAMATH_CALUDE_group_meal_cost_l1785_178542

/-- Calculates the total cost for a group meal including tax and tip -/
def calculate_total_cost (vegetarian_price chicken_price steak_price kids_price : ℚ)
                         (tax_rate tip_rate : ℚ)
                         (vegetarian_count chicken_count steak_count kids_count : ℕ) : ℚ :=
  let subtotal := vegetarian_price * vegetarian_count +
                  chicken_price * chicken_count +
                  steak_price * steak_count +
                  kids_price * kids_count
  let tax := subtotal * tax_rate
  let tip := subtotal * tip_rate
  subtotal + tax + tip

/-- Theorem stating that the total cost for the given group is $90 -/
theorem group_meal_cost :
  calculate_total_cost 5 7 10 3 (1/10) (15/100) 3 4 2 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_group_meal_cost_l1785_178542


namespace NUMINAMATH_CALUDE_impossible_grid_2005_l1785_178597

theorem impossible_grid_2005 : ¬ ∃ (a b c d e f g h i : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (a * b * c = 2005) ∧ (d * e * f = 2005) ∧ (g * h * i = 2005) ∧
  (a * d * g = 2005) ∧ (b * e * h = 2005) ∧ (c * f * i = 2005) ∧
  (a * e * i = 2005) ∧ (c * e * g = 2005) :=
by sorry


end NUMINAMATH_CALUDE_impossible_grid_2005_l1785_178597


namespace NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_l1785_178556

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_factorial_sum :
  sum_factorials 2003 % 100 = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_l1785_178556


namespace NUMINAMATH_CALUDE_circle_and_max_distance_l1785_178592

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the ray 3x - y = 0 (x ≥ 0)
def Ray := {p : ℝ × ℝ | 3 * p.1 - p.2 = 0 ∧ p.1 ≥ 0}

-- Define the line x = 4
def TangentLine := {p : ℝ × ℝ | p.1 = 4}

-- Define the line 3x + 4y + 10 = 0
def ChordLine := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 + 10 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 0)

-- Define the theorem
theorem circle_and_max_distance :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- Circle C's center is on the ray
    center ∈ Ray ∧
    -- Circle C is tangent to the line x = 4
    (∃ (p : ℝ × ℝ), p ∈ Circle center radius ∧ p ∈ TangentLine) ∧
    -- The chord intercepted by the line has length 4√3
    (∃ (p q : ℝ × ℝ), p ∈ Circle center radius ∧ q ∈ Circle center radius ∧
      p ∈ ChordLine ∧ q ∈ ChordLine ∧
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = 48) ∧
    -- The equation of circle C is x^2 + y^2 = 16
    Circle center radius = {p : ℝ × ℝ | p.1^2 + p.2^2 = 16} ∧
    -- The maximum value of |PA|^2 + |PB|^2 is 38 + 8√2
    (∀ (p : ℝ × ℝ), p ∈ Circle center radius →
      (p.1 - A.1)^2 + (p.2 - A.2)^2 + (p.1 - B.1)^2 + (p.2 - B.2)^2 ≤ 38 + 8 * Real.sqrt 2) ∧
    (∃ (p : ℝ × ℝ), p ∈ Circle center radius ∧
      (p.1 - A.1)^2 + (p.2 - A.2)^2 + (p.1 - B.1)^2 + (p.2 - B.2)^2 = 38 + 8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_max_distance_l1785_178592


namespace NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l1785_178587

/-- The sum of the infinite series ∑(n=1 to ∞) (3n^2 - 2n + 1) / (n^4 - n^3 + n^2 - n + 1) is equal to 1. -/
theorem infinite_series_sum_equals_one :
  let a : ℕ → ℚ := λ n => (3*n^2 - 2*n + 1) / (n^4 - n^3 + n^2 - n + 1)
  ∑' n, a n = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l1785_178587


namespace NUMINAMATH_CALUDE_equation_solution_l1785_178599

theorem equation_solution :
  let f (x : ℝ) := (6*x + 3) / (3*x^2 + 6*x - 9) - 3*x / (3*x - 3)
  ∀ x : ℝ, x ≠ 1 → (f x = 0 ↔ x = (3 + Real.sqrt 21) / 2 ∨ x = (3 - Real.sqrt 21) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1785_178599


namespace NUMINAMATH_CALUDE_bigger_part_of_60_l1785_178565

theorem bigger_part_of_60 (x y : ℝ) (h1 : x + y = 60) (h2 : 10 * x + 22 * y = 780) 
  (h3 : x > 0) (h4 : y > 0) : max x y = 45 := by
  sorry

end NUMINAMATH_CALUDE_bigger_part_of_60_l1785_178565


namespace NUMINAMATH_CALUDE_square_area_side_3_l1785_178581

/-- The area of a square with side length 3 is 9 square units. -/
theorem square_area_side_3 : 
  ∀ (s : ℝ), s = 3 → s * s = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_side_3_l1785_178581


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1785_178585

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) :
  Complex.abs (z + 2 * Complex.I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1785_178585


namespace NUMINAMATH_CALUDE_larger_city_size_proof_l1785_178560

/-- The number of cubic yards in the larger city -/
def larger_city_size : ℕ := 9000

/-- The population density in people per cubic yard -/
def population_density : ℕ := 80

/-- The size of the smaller city in cubic yards -/
def smaller_city_size : ℕ := 6400

/-- The population difference between the larger and smaller city -/
def population_difference : ℕ := 208000

theorem larger_city_size_proof :
  population_density * larger_city_size = 
  population_density * smaller_city_size + population_difference := by
  sorry

end NUMINAMATH_CALUDE_larger_city_size_proof_l1785_178560


namespace NUMINAMATH_CALUDE_log_function_fixed_point_l1785_178595

theorem log_function_fixed_point (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log a + 1
  f 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_log_function_fixed_point_l1785_178595


namespace NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l1785_178507

/-- Given a cone whose lateral surface is a semicircle with radius a,
    prove that the height of the cone is (√3/2)a. -/
theorem cone_height_from_lateral_surface (a : ℝ) (h : a > 0) :
  let l := a  -- slant height
  let r := a / 2  -- radius of the base
  let h := Real.sqrt ((l ^ 2) - (r ^ 2))  -- height of the cone
  h = (Real.sqrt 3 / 2) * a :=
by sorry

end NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l1785_178507


namespace NUMINAMATH_CALUDE_max_sum_abc_l1785_178514

theorem max_sum_abc (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : a + b = 719) 
  (h3 : c - a = 915) : 
  (∀ x y z : ℕ, x < y → x + y = 719 → z - x = 915 → x + y + z ≤ 1993) ∧ 
  (∃ x y z : ℕ, x < y ∧ x + y = 719 ∧ z - x = 915 ∧ x + y + z = 1993) :=
sorry

end NUMINAMATH_CALUDE_max_sum_abc_l1785_178514


namespace NUMINAMATH_CALUDE_tangent_circles_a_values_l1785_178568

/-- Two circles are tangent if the distance between their centers is equal to
    the sum or difference of their radii -/
def are_tangent (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = (r1 + r2)^2) ∨
  (((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = (r1 - r2)^2)

theorem tangent_circles_a_values :
  ∀ a : ℝ,
  are_tangent (0, 0) (-4, a) 1 5 →
  (a = 0 ∨ a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_a_values_l1785_178568


namespace NUMINAMATH_CALUDE_sum_of_seven_terms_l1785_178519

/-- An arithmetic sequence with a_4 = 7 -/
def arithmetic_seq (n : ℕ) : ℝ :=
  sorry

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ :=
  sorry

theorem sum_of_seven_terms :
  arithmetic_seq 4 = 7 → S 7 = 49 :=
sorry

end NUMINAMATH_CALUDE_sum_of_seven_terms_l1785_178519


namespace NUMINAMATH_CALUDE_acid_mixture_theorem_l1785_178501

/-- Represents an acid solution with a given volume and concentration. -/
structure AcidSolution where
  volume : ℝ
  concentration : ℝ

/-- Calculates the amount of pure acid in a solution. -/
def pureAcid (solution : AcidSolution) : ℝ :=
  solution.volume * solution.concentration

/-- Theorem: Mixing 4 liters of 60% acid solution with 16 liters of 75% acid solution
    results in a 72% acid solution with a total volume of 20 liters. -/
theorem acid_mixture_theorem : 
  let solution1 : AcidSolution := ⟨4, 0.6⟩
  let solution2 : AcidSolution := ⟨16, 0.75⟩
  let totalVolume := solution1.volume + solution2.volume
  let totalPureAcid := pureAcid solution1 + pureAcid solution2
  totalVolume = 20 ∧ 
  totalPureAcid / totalVolume = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_acid_mixture_theorem_l1785_178501


namespace NUMINAMATH_CALUDE_chord_length_implies_center_l1785_178530

/-- Given a circle and a line cutting a chord from it, prove the possible values of the circle's center. -/
theorem chord_length_implies_center (a : ℝ) : 
  (∃ (x y : ℝ), (x - a)^2 + y^2 = 4 ∧ x - y - 2 = 0) → 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - a)^2 + y₁^2 = 4 ∧ 
    (x₂ - a)^2 + y₂^2 = 4 ∧ 
    x₁ - y₁ - 2 = 0 ∧ 
    x₂ - y₂ - 2 = 0 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) → 
  a = 0 ∨ a = 4 := by
sorry

end NUMINAMATH_CALUDE_chord_length_implies_center_l1785_178530


namespace NUMINAMATH_CALUDE_altitude_difference_example_l1785_178517

/-- The difference between the highest and lowest altitudes among three given altitudes -/
def altitude_difference (a b c : Int) : Int :=
  max a (max b c) - min a (min b c)

/-- Theorem stating that the altitude difference for the given values is 77 meters -/
theorem altitude_difference_example : altitude_difference (-102) (-80) (-25) = 77 := by
  sorry

end NUMINAMATH_CALUDE_altitude_difference_example_l1785_178517


namespace NUMINAMATH_CALUDE_power_of_power_l1785_178589

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1785_178589


namespace NUMINAMATH_CALUDE_total_cakes_per_week_l1785_178548

/-- Represents the quantities of cakes served during lunch on a weekday -/
structure LunchCakes :=
  (chocolate : ℕ)
  (vanilla : ℕ)
  (cheesecake : ℕ)

/-- Represents the quantities of cakes served during dinner on a weekday -/
structure DinnerCakes :=
  (chocolate : ℕ)
  (vanilla : ℕ)
  (cheesecake : ℕ)
  (carrot : ℕ)

/-- Calculates the total number of cakes served on a weekday -/
def weekdayTotal (lunch : LunchCakes) (dinner : DinnerCakes) : ℕ :=
  lunch.chocolate + lunch.vanilla + lunch.cheesecake +
  dinner.chocolate + dinner.vanilla + dinner.cheesecake + dinner.carrot

/-- Calculates the total number of cakes served on a weekend day -/
def weekendTotal (lunch : LunchCakes) (dinner : DinnerCakes) : ℕ :=
  2 * (lunch.chocolate + lunch.vanilla + lunch.cheesecake +
       dinner.chocolate + dinner.vanilla + dinner.cheesecake + dinner.carrot)

/-- Theorem: The total number of cakes served during an entire week is 522 -/
theorem total_cakes_per_week
  (lunch : LunchCakes)
  (dinner : DinnerCakes)
  (h1 : lunch.chocolate = 6)
  (h2 : lunch.vanilla = 8)
  (h3 : lunch.cheesecake = 10)
  (h4 : dinner.chocolate = 9)
  (h5 : dinner.vanilla = 7)
  (h6 : dinner.cheesecake = 5)
  (h7 : dinner.carrot = 13) :
  5 * weekdayTotal lunch dinner + 2 * weekendTotal lunch dinner = 522 := by
  sorry

end NUMINAMATH_CALUDE_total_cakes_per_week_l1785_178548


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l1785_178506

theorem father_son_age_ratio : 
  ∀ (father_age son_age : ℕ),
  father_age * son_age = 756 →
  (father_age + 6) / (son_age + 6) = 2 →
  father_age / son_age = 7 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l1785_178506


namespace NUMINAMATH_CALUDE_distance_from_negative_one_l1785_178576

theorem distance_from_negative_one : ∀ x : ℝ, |x - (-1)| = 6 ↔ x = 5 ∨ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_negative_one_l1785_178576


namespace NUMINAMATH_CALUDE_car_speed_theorem_l1785_178529

/-- Represents a car with specific driving characteristics -/
structure Car where
  cooldownTime : ℕ  -- Time required for cooldown in hours
  drivingCycleTime : ℕ  -- Time of continuous driving before cooldown in hours
  totalTime : ℕ  -- Total time of the journey in hours
  totalDistance : ℕ  -- Total distance covered in miles

/-- Calculates the speed of the car in miles per hour -/
def calculateSpeed (c : Car) : ℚ :=
  let totalCycles : ℕ := c.totalTime / (c.drivingCycleTime + c.cooldownTime)
  let remainingTime : ℕ := c.totalTime % (c.drivingCycleTime + c.cooldownTime)
  let actualDrivingTime : ℕ := (totalCycles * c.drivingCycleTime) + min remainingTime c.drivingCycleTime
  c.totalDistance / actualDrivingTime

theorem car_speed_theorem (c : Car) 
    (h1 : c.cooldownTime = 1)
    (h2 : c.drivingCycleTime = 5)
    (h3 : c.totalTime = 13)
    (h4 : c.totalDistance = 88) :
  calculateSpeed c = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_theorem_l1785_178529


namespace NUMINAMATH_CALUDE_equation_solutions_l1785_178543

theorem equation_solutions :
  (∃ x₁ x₂, 3 * (x₁ - 2)^2 = 27 ∧ 3 * (x₂ - 2)^2 = 27 ∧ x₁ = 5 ∧ x₂ = -1) ∧
  (∃ x, (x + 5)^3 + 27 = 0 ∧ x = -8) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1785_178543


namespace NUMINAMATH_CALUDE_regular_polygon_30_degree_central_angle_l1785_178562

/-- A regular polygon with a central angle of 30° has 12 sides. -/
theorem regular_polygon_30_degree_central_angle (n : ℕ) : 
  (360 : ℝ) / n = 30 → n = 12 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_30_degree_central_angle_l1785_178562


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1785_178584

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 30 → b = 40 → c^2 = a^2 + b^2 → c = 50 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1785_178584


namespace NUMINAMATH_CALUDE_smallest_sum_of_two_three_digit_numbers_l1785_178590

-- Define a type for 3-digit numbers
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n < 1000 }

-- Define a function to check if a number uses given digits
def usesGivenDigits (n : ℕ) (digits : List ℕ) : Prop := sorry

-- Define a function to check if two numbers use all given digits exactly once
def useAllDigitsOnce (a b : ℕ) (digits : List ℕ) : Prop := sorry

-- Theorem statement
theorem smallest_sum_of_two_three_digit_numbers :
  ∃ (a b : ThreeDigitNumber),
    useAllDigitsOnce a.val b.val [1, 2, 3, 7, 8, 9] ∧
    (∀ (x y : ThreeDigitNumber),
      useAllDigitsOnce x.val y.val [1, 2, 3, 7, 8, 9] →
      a.val + b.val ≤ x.val + y.val) ∧
    a.val + b.val = 912 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_two_three_digit_numbers_l1785_178590


namespace NUMINAMATH_CALUDE_corn_purchase_amount_l1785_178536

/-- Represents the purchase of corn, beans, and rice -/
structure Purchase where
  corn : ℝ
  beans : ℝ
  rice : ℝ

/-- Checks if a purchase satisfies the given conditions -/
def isValidPurchase (p : Purchase) : Prop :=
  p.corn + p.beans + p.rice = 30 ∧
  1.1 * p.corn + 0.6 * p.beans + 0.9 * p.rice = 24 ∧
  p.rice = 0.5 * p.beans

theorem corn_purchase_amount :
  ∃ (p : Purchase), isValidPurchase p ∧ p.corn = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_corn_purchase_amount_l1785_178536


namespace NUMINAMATH_CALUDE_largest_n_with_conditions_l1785_178541

theorem largest_n_with_conditions : ∃ n : ℕ, n = 289 ∧ 
  (∃ m : ℤ, n^2 = (m+1)^3 - m^3) ∧
  (∃ k : ℕ, 2*n + 99 = k^2) ∧
  (∀ n' : ℕ, n' > n → 
    (¬∃ m : ℤ, n'^2 = (m+1)^3 - m^3) ∨
    (¬∃ k : ℕ, 2*n' + 99 = k^2)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_conditions_l1785_178541


namespace NUMINAMATH_CALUDE_existence_of_integers_l1785_178561

theorem existence_of_integers (a b : ℝ) (h : a ≠ b) : 
  ∃ (m n : ℤ), a * (m : ℝ) + b * (n : ℝ) < 0 ∧ b * (m : ℝ) + a * (n : ℝ) > 0 := by
sorry

end NUMINAMATH_CALUDE_existence_of_integers_l1785_178561


namespace NUMINAMATH_CALUDE_janinas_pancakes_l1785_178575

/-- Calculates the minimum number of pancakes Janina must sell to cover her expenses -/
theorem janinas_pancakes (rent : ℝ) (supplies : ℝ) (taxes_wages : ℝ) (price_per_pancake : ℝ) :
  rent = 75.50 →
  supplies = 28.40 →
  taxes_wages = 32.10 →
  price_per_pancake = 1.75 →
  ∃ n : ℕ, n ≥ 78 ∧ n * price_per_pancake ≥ rent + supplies + taxes_wages :=
by sorry

end NUMINAMATH_CALUDE_janinas_pancakes_l1785_178575


namespace NUMINAMATH_CALUDE_expression_evaluation_l1785_178578

theorem expression_evaluation (a b c : ℝ) (ha : a = 13) (hb : b = 17) (hc : c = 19) :
  (b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b) + a^2 * (1/b - 1/c)) /
  (b * (1/c - 1/a) + c * (1/a - 1/b) + a * (1/b - 1/c)) = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1785_178578


namespace NUMINAMATH_CALUDE_tangent_circle_radius_double_inscribed_l1785_178547

/-- Given a right triangle ABC with legs a and b, hypotenuse c, inscribed circle radius r,
    circumscribed circle radius R, and a circle with radius ρ touching both legs and the
    circumscribed circle, prove that ρ = 2r. -/
theorem tangent_circle_radius_double_inscribed (a b c r R ρ : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 → R > 0 → ρ > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem for right triangle
  R = c / 2 →  -- Radius of circumscribed circle is half the hypotenuse
  r = (a + b - c) / 2 →  -- Formula for inscribed circle radius
  ρ^2 - (a + b - c) * ρ = 0 →  -- Equation derived from tangency conditions
  ρ = 2 * r := by
sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_double_inscribed_l1785_178547


namespace NUMINAMATH_CALUDE_inequality_theorem_stronger_inequality_best_constant_l1785_178555

theorem inequality_theorem (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) : 
  |(a + b) / (a - b)| + |(b + c) / (b - c)| + |(c + a) / (c - a)| ≥ 2 := by
  sorry

theorem stronger_inequality (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) 
  (pa : a ≥ 0) (pb : b ≥ 0) (pc : c ≥ 0) : 
  |(a + b) / (a - b)| + |(b + c) / (b - c)| + |(c + a) / (c - a)| > 3 := by
  sorry

theorem best_constant (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
  |(a + b) / (a - b)| + |(b + c) / (b - c)| + |(c + a) / (c - a)| < 3 + ε := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_stronger_inequality_best_constant_l1785_178555


namespace NUMINAMATH_CALUDE_product_of_solutions_l1785_178538

theorem product_of_solutions (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016) :
  (2 - x₁/y₁) * (2 - x₂/y₂) * (2 - x₃/y₃) = 26219/2016 := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l1785_178538


namespace NUMINAMATH_CALUDE_orange_balls_count_l1785_178546

theorem orange_balls_count (total : Nat) (red : Nat) (blue : Nat) (pink : Nat) (orange : Nat) : 
  total = 50 →
  red = 20 →
  blue = 10 →
  total = red + blue + pink + orange →
  pink = 3 * orange →
  orange = 5 := by
sorry

end NUMINAMATH_CALUDE_orange_balls_count_l1785_178546


namespace NUMINAMATH_CALUDE_range_of_b_over_a_l1785_178558

theorem range_of_b_over_a (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 5 - 3 * a ≤ b) (h2 : b ≤ 4 - a) (h3 : Real.log b ≥ a) :
  ∃ (x : ℝ), x = b / a ∧ e ≤ x ∧ x ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_over_a_l1785_178558


namespace NUMINAMATH_CALUDE_fraction_simplification_l1785_178540

theorem fraction_simplification : (10 : ℝ) / (10 * 11 - 10^2) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1785_178540


namespace NUMINAMATH_CALUDE_tangent_circles_triangle_area_l1785_178522

/-- The area of the triangle formed by the points of tangency of three
    mutually externally tangent circles with radii 2, 3, and 4 -/
theorem tangent_circles_triangle_area :
  ∃ (A B C : ℝ × ℝ),
    let r₁ : ℝ := 2
    let r₂ : ℝ := 3
    let r₃ : ℝ := 4
    let O₁ : ℝ × ℝ := (0, 0)
    let O₂ : ℝ × ℝ := (r₁ + r₂, 0)
    let O₃ : ℝ × ℝ := (0, r₁ + r₃)
    -- A, B, C are points of tangency
    A.1^2 + A.2^2 = r₁^2 ∧
    (A.1 - (r₁ + r₂))^2 + A.2^2 = r₂^2 ∧
    B.1^2 + B.2^2 = r₁^2 ∧
    B.1^2 + (B.2 - (r₁ + r₃))^2 = r₃^2 ∧
    (C.1 - (r₁ + r₂))^2 + C.2^2 = r₂^2 ∧
    C.1^2 + (C.2 - (r₁ + r₃))^2 = r₃^2 →
    -- Area of triangle ABC
    abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1) / 2 = 25 / 14 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_triangle_area_l1785_178522


namespace NUMINAMATH_CALUDE_bruce_initial_amount_l1785_178526

def crayons_cost : ℕ := 5 * 5
def books_cost : ℕ := 10 * 5
def calculators_cost : ℕ := 3 * 5
def total_spent : ℕ := crayons_cost + books_cost + calculators_cost
def bags_cost : ℕ := 11 * 10
def initial_amount : ℕ := total_spent + bags_cost

theorem bruce_initial_amount : initial_amount = 200 := by
  sorry

end NUMINAMATH_CALUDE_bruce_initial_amount_l1785_178526


namespace NUMINAMATH_CALUDE_essays_total_pages_l1785_178525

def words_per_page : ℕ := 235

def johnny_words : ℕ := 195
def madeline_words : ℕ := 2 * johnny_words
def timothy_words : ℕ := madeline_words + 50
def samantha_words : ℕ := 3 * madeline_words
def ryan_words : ℕ := johnny_words + 100

def pages_needed (words : ℕ) : ℕ :=
  (words + words_per_page - 1) / words_per_page

def total_pages : ℕ :=
  pages_needed johnny_words +
  pages_needed madeline_words +
  pages_needed timothy_words +
  pages_needed samantha_words +
  pages_needed ryan_words

theorem essays_total_pages : total_pages = 12 := by
  sorry

end NUMINAMATH_CALUDE_essays_total_pages_l1785_178525


namespace NUMINAMATH_CALUDE_spherical_segment_max_volume_l1785_178586

/-- Given a spherical segment with surface area S, its maximum volume V is S √(S / (18π)) -/
theorem spherical_segment_max_volume (S : ℝ) (h : S > 0) :
  ∃ V : ℝ, V = S * Real.sqrt (S / (18 * Real.pi)) ∧
  ∀ (V' : ℝ), (∃ (R h : ℝ), R > 0 ∧ h > 0 ∧ h ≤ 2*R ∧ S = 2 * Real.pi * R * h ∧
                V' = Real.pi * h^2 * (3*R - h) / 3) →
  V' ≤ V :=
sorry

end NUMINAMATH_CALUDE_spherical_segment_max_volume_l1785_178586


namespace NUMINAMATH_CALUDE_roots_sum_square_l1785_178520

theorem roots_sum_square (α β : ℝ) : 
  (α^2 - α - 2006 = 0) → 
  (β^2 - β - 2006 = 0) → 
  (α + β = 1) →
  α + β^2 = 2007 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_square_l1785_178520


namespace NUMINAMATH_CALUDE_last_digits_of_powers_l1785_178559

theorem last_digits_of_powers : 
  (∃ n : ℕ, 1989^1989 ≡ 9 [MOD 10]) ∧
  (∃ n : ℕ, 1989^1992 ≡ 1 [MOD 10]) ∧
  (∃ n : ℕ, 1992^1989 ≡ 2 [MOD 10]) ∧
  (∃ n : ℕ, 1992^1992 ≡ 6 [MOD 10]) :=
by sorry

end NUMINAMATH_CALUDE_last_digits_of_powers_l1785_178559


namespace NUMINAMATH_CALUDE_inequality_proof_l1785_178524

theorem inequality_proof (n : ℕ) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1785_178524


namespace NUMINAMATH_CALUDE_product_of_roots_zero_l1785_178545

theorem product_of_roots_zero (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 4*a = 0 →
  b^3 - 4*b = 0 →
  c^3 - 4*c = 0 →
  a * b * c = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_zero_l1785_178545


namespace NUMINAMATH_CALUDE_square_fraction_count_l1785_178539

theorem square_fraction_count : ∃! (n : ℤ), 0 < n ∧ n < 25 ∧ ∃ (k : ℤ), (n : ℚ) / (25 - n) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_count_l1785_178539
