import Mathlib

namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_plus_four_l4028_402844

theorem square_plus_reciprocal_square_plus_four (m : ℝ) (h : m + 1/m = 10) :
  m^2 + 1/m^2 + 4 = 102 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_plus_four_l4028_402844


namespace NUMINAMATH_CALUDE_compare_fractions_l4028_402874

theorem compare_fractions : 2/3 < Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_compare_fractions_l4028_402874


namespace NUMINAMATH_CALUDE_exactly_four_points_C_l4028_402889

/-- Given two points A and B in a plane that are 12 units apart, this function
    returns the number of points C such that the perimeter of triangle ABC is 60 units
    and the area of triangle ABC is 72 square units. -/
def count_points_C (A B : ℝ × ℝ) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 4 points C satisfying the conditions. -/
theorem exactly_four_points_C (A B : ℝ × ℝ) (h : dist A B = 12) :
  count_points_C A B = 4 :=
sorry

end NUMINAMATH_CALUDE_exactly_four_points_C_l4028_402889


namespace NUMINAMATH_CALUDE_selling_price_difference_difference_is_approximately_56_l4028_402895

/-- The difference in selling prices given the original selling price and profit percentages -/
theorem selling_price_difference (original_selling_price : ℝ) : ℝ :=
  let original_profit_rate := 0.1
  let new_purchase_discount := 0.1
  let new_profit_rate := 0.3
  
  let original_purchase_price := original_selling_price / (1 + original_profit_rate)
  let new_purchase_price := original_purchase_price * (1 - new_purchase_discount)
  let new_selling_price := new_purchase_price * (1 + new_profit_rate)
  
  new_selling_price - original_selling_price

/-- The difference in selling prices is approximately $56 -/
theorem difference_is_approximately_56 :
  ∃ ε > 0, abs (selling_price_difference 879.9999999999993 - 56) < ε :=
sorry

end NUMINAMATH_CALUDE_selling_price_difference_difference_is_approximately_56_l4028_402895


namespace NUMINAMATH_CALUDE_shaded_square_area_ratio_l4028_402815

/-- The ratio of the area of a square formed by connecting the centers of four adjacent unit squares
    in a 5x5 grid to the area of the entire 5x5 grid is 2/25. -/
theorem shaded_square_area_ratio : 
  let grid_side : ℕ := 5
  let unit_square_side : ℝ := 1
  let grid_area : ℝ := (grid_side ^ 2 : ℝ) * unit_square_side ^ 2
  let shaded_square_side : ℝ := Real.sqrt 2 * unit_square_side
  let shaded_square_area : ℝ := shaded_square_side ^ 2
  shaded_square_area / grid_area = 2 / 25 := by
sorry


end NUMINAMATH_CALUDE_shaded_square_area_ratio_l4028_402815


namespace NUMINAMATH_CALUDE_max_sum_of_factors_48_l4028_402866

theorem max_sum_of_factors_48 : 
  ∃ (a b : ℕ), a * b = 48 ∧ a + b = 49 ∧ ∀ (x y : ℕ), x * y = 48 → x + y ≤ 49 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_48_l4028_402866


namespace NUMINAMATH_CALUDE_overlap_area_is_1_2_l4028_402854

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Calculates the area of overlap between two triangles -/
def areaOfOverlap (t1 t2 : Triangle) : ℝ :=
  sorry

/-- The 3x3 grid of points -/
def grid : List Point :=
  [ {x := 0, y := 2}, {x := 1.5, y := 2}, {x := 3, y := 2},
    {x := 0, y := 1}, {x := 1.5, y := 1}, {x := 3, y := 1},
    {x := 0, y := 0}, {x := 1.5, y := 0}, {x := 3, y := 0} ]

/-- Triangle 1: top-left corner, middle of right edge, bottom-center point -/
def triangle1 : Triangle :=
  { p1 := {x := 0, y := 2},
    p2 := {x := 3, y := 1},
    p3 := {x := 1.5, y := 0} }

/-- Triangle 2: bottom-left corner, middle of top edge, right-center point -/
def triangle2 : Triangle :=
  { p1 := {x := 0, y := 0},
    p2 := {x := 1.5, y := 2},
    p3 := {x := 3, y := 1} }

/-- Theorem stating that the area of overlap between triangle1 and triangle2 is 1.2 square units -/
theorem overlap_area_is_1_2 : areaOfOverlap triangle1 triangle2 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_is_1_2_l4028_402854


namespace NUMINAMATH_CALUDE_least_x_for_1894x_divisible_by_3_l4028_402809

theorem least_x_for_1894x_divisible_by_3 : 
  ∃ x : ℕ, (∀ y : ℕ, y < x → ¬(3 ∣ 1894 * y)) ∧ (3 ∣ 1894 * x) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_x_for_1894x_divisible_by_3_l4028_402809


namespace NUMINAMATH_CALUDE_difference_of_values_l4028_402829

theorem difference_of_values (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (diff_squares_eq : x^2 - y^2 = 16) : 
  x - y = 2 := by
sorry

end NUMINAMATH_CALUDE_difference_of_values_l4028_402829


namespace NUMINAMATH_CALUDE_pebble_splash_width_proof_l4028_402812

/-- The width of a splash made by a pebble -/
def pebble_splash_width : ℝ := 0.25

theorem pebble_splash_width_proof 
  (total_splash_width : ℝ) 
  (rock_splash_width : ℝ) 
  (boulder_splash_width : ℝ) 
  (pebble_count : ℕ) 
  (rock_count : ℕ) 
  (boulder_count : ℕ) 
  (h1 : total_splash_width = 7) 
  (h2 : rock_splash_width = 1/2) 
  (h3 : boulder_splash_width = 2) 
  (h4 : pebble_count = 6) 
  (h5 : rock_count = 3) 
  (h6 : boulder_count = 2) : 
  pebble_splash_width = (total_splash_width - rock_count * rock_splash_width - boulder_count * boulder_splash_width) / pebble_count :=
by sorry

end NUMINAMATH_CALUDE_pebble_splash_width_proof_l4028_402812


namespace NUMINAMATH_CALUDE_circle_circumference_l4028_402876

-- Define the circles x and y
def circle_x : Real → Prop := sorry
def circle_y : Real → Prop := sorry

-- Define the area of a circle
def area (circle : Real → Prop) : Real := sorry

-- Define the radius of a circle
def radius (circle : Real → Prop) : Real := sorry

-- Define the circumference of a circle
def circumference (circle : Real → Prop) : Real := sorry

-- State the theorem
theorem circle_circumference :
  (area circle_x = area circle_y) →  -- Circles x and y have the same area
  (radius circle_y / 2 = 3.5) →      -- Half of the radius of circle y is 3.5
  (circumference circle_x = 14 * Real.pi) := -- The circumference of circle x is 14π
by sorry

end NUMINAMATH_CALUDE_circle_circumference_l4028_402876


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4028_402819

theorem inequality_equivalence (x : ℝ) (h : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) ≥ 15 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4028_402819


namespace NUMINAMATH_CALUDE_birds_after_week_l4028_402843

def initial_chickens : ℕ := 300
def initial_turkeys : ℕ := 200
def initial_guinea_fowls : ℕ := 80

def daily_loss_chickens : ℕ := 20
def daily_loss_turkeys : ℕ := 8
def daily_loss_guinea_fowls : ℕ := 5

def days_in_week : ℕ := 7

def remaining_birds : ℕ := 
  (initial_chickens - daily_loss_chickens * days_in_week) +
  (initial_turkeys - daily_loss_turkeys * days_in_week) +
  (initial_guinea_fowls - daily_loss_guinea_fowls * days_in_week)

theorem birds_after_week : remaining_birds = 349 := by
  sorry

end NUMINAMATH_CALUDE_birds_after_week_l4028_402843


namespace NUMINAMATH_CALUDE_largest_square_area_in_isosceles_triangle_l4028_402873

/-- The area of the largest square that can be cut from an isosceles triangle -/
theorem largest_square_area_in_isosceles_triangle 
  (base height : ℝ) 
  (h_base : base = 2) 
  (h_height : height = 3) :
  let side_length := (2 * base * height) / (base + height)
  (side_length^2 : ℝ) = 36 / 25 := by sorry

end NUMINAMATH_CALUDE_largest_square_area_in_isosceles_triangle_l4028_402873


namespace NUMINAMATH_CALUDE_count_convex_polygons_l4028_402828

/-- The number of ways to select 33 non-adjacent vertices from a 100-vertex polygon -/
def select_nonadjacent_vertices (n m : ℕ) : ℕ :=
  Nat.choose (n - 33 + 1) 33 + Nat.choose (n - 34 + 1) 32

/-- Theorem: The number of ways to select 33 non-adjacent vertices from a 100-vertex polygon,
    ensuring no shared sides, is equal to ⁽⁶⁷₃₃⁾ + ⁽⁶⁶₃₂⁾ -/
theorem count_convex_polygons :
  select_nonadjacent_vertices 100 33 = Nat.choose 67 33 + Nat.choose 66 32 := by
  sorry

end NUMINAMATH_CALUDE_count_convex_polygons_l4028_402828


namespace NUMINAMATH_CALUDE_quadratic_has_minimum_l4028_402857

theorem quadratic_has_minimum (a b : ℝ) (ha : a > 0) :
  let g (x : ℝ) := a * x^2 + 2 * b * x + ((4 * b^2) / a - 3)
  ∃ (m : ℝ), ∀ (x : ℝ), g x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_minimum_l4028_402857


namespace NUMINAMATH_CALUDE_equation_solutions_l4028_402894

theorem equation_solutions :
  (∃ x : ℚ, 8 * (x + 2)^3 = 27 ↔ x = -1/2) ∧
  (∃ x₁ x₂ : ℚ, 25 * (x₁ - 1)^2 = 4 ∧ 25 * (x₂ - 1)^2 = 4 ↔ x₁ = 7/5 ∧ x₂ = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4028_402894


namespace NUMINAMATH_CALUDE_percent_to_decimal_four_percent_to_decimal_l4028_402836

theorem percent_to_decimal (p : ℚ) : p / 100 = p * (1 / 100) := by sorry

theorem four_percent_to_decimal : (4 : ℚ) / 100 = 0.04 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_four_percent_to_decimal_l4028_402836


namespace NUMINAMATH_CALUDE_binary_to_base4_conversion_l4028_402891

/-- Converts a binary number represented as a list of bits to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation. -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary representation of 1101101001₂ -/
def binary_num : List Bool :=
  [true, true, false, true, true, false, true, false, false, true]

/-- The base 4 representation of 13201₄ -/
def base4_num : List ℕ := [1, 3, 2, 0, 1]

theorem binary_to_base4_conversion :
  decimal_to_base4 (binary_to_decimal binary_num) = base4_num := by
  sorry

end NUMINAMATH_CALUDE_binary_to_base4_conversion_l4028_402891


namespace NUMINAMATH_CALUDE_tshirt_cost_l4028_402853

def total_spent : ℝ := 199
def num_tshirts : ℕ := 20

theorem tshirt_cost : total_spent / num_tshirts = 9.95 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_l4028_402853


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l4028_402826

theorem triangle_angle_measure (A B C : ℝ) : 
  A = 40 → B = 2 * C → A + B + C = 180 → C = 140 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l4028_402826


namespace NUMINAMATH_CALUDE_percentage_difference_l4028_402849

theorem percentage_difference (A B y : ℝ) : 
  A > B ∧ B > 0 → B = A * (1 - y / 100) → y = 100 * (A - B) / A := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l4028_402849


namespace NUMINAMATH_CALUDE_lcm_of_105_and_360_l4028_402834

theorem lcm_of_105_and_360 : Nat.lcm 105 360 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_105_and_360_l4028_402834


namespace NUMINAMATH_CALUDE_indefinite_stick_shortening_process_l4028_402831

theorem indefinite_stick_shortening_process : ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a > b ∧ a > c ∧
  a > b + c ∧
  ∃ (a' b' c' : ℝ), 
    a' = b + c ∧
    b' = b ∧
    c' = c ∧
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' > b' ∧ a' > c' ∧
    a' > b' + c' :=
by sorry


end NUMINAMATH_CALUDE_indefinite_stick_shortening_process_l4028_402831


namespace NUMINAMATH_CALUDE_total_pencils_l4028_402897

/-- Given that Reeta has 20 pencils and Anika has 4 more than twice the number of pencils as Reeta,
    prove that they have 64 pencils in total. -/
theorem total_pencils (reeta_pencils : ℕ) (anika_pencils : ℕ) : 
  reeta_pencils = 20 →
  anika_pencils = 2 * reeta_pencils + 4 →
  anika_pencils + reeta_pencils = 64 := by
sorry

end NUMINAMATH_CALUDE_total_pencils_l4028_402897


namespace NUMINAMATH_CALUDE_solve_for_y_l4028_402807

theorem solve_for_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 4) : y = -6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l4028_402807


namespace NUMINAMATH_CALUDE_expected_sales_at_2_degrees_l4028_402872

/-- Represents the linear regression model for hot drink sales based on temperature -/
def hot_drink_sales (x : ℝ) : ℝ := -2.35 * x + 147.7

/-- Theorem stating that when the temperature is 2°C, the expected number of hot drinks sold is 143 -/
theorem expected_sales_at_2_degrees :
  Int.floor (hot_drink_sales 2) = 143 := by
  sorry

end NUMINAMATH_CALUDE_expected_sales_at_2_degrees_l4028_402872


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l4028_402847

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (2, -1)
  parallel a b → m = -1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l4028_402847


namespace NUMINAMATH_CALUDE_linear_equation_solution_l4028_402837

theorem linear_equation_solution :
  ∃ x : ℝ, 2 * x - 1 = 1 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l4028_402837


namespace NUMINAMATH_CALUDE_square_minus_circle_area_l4028_402888

theorem square_minus_circle_area : 
  let square_side : ℝ := 4
  let circle_diameter : ℝ := 2
  let square_area := square_side * square_side
  let circle_area := Real.pi * (circle_diameter / 2) ^ 2
  square_area - circle_area = 16 - Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_minus_circle_area_l4028_402888


namespace NUMINAMATH_CALUDE_vector_projection_l4028_402892

/-- Given two vectors a and b in ℝ², and a vector c such that a + c = 0,
    prove that the projection of c onto b is -√65/5 -/
theorem vector_projection (a b c : ℝ × ℝ) : 
  a = (2, 3) → 
  b = (-4, 7) → 
  a + c = (0, 0) → 
  (c.1 * b.1 + c.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l4028_402892


namespace NUMINAMATH_CALUDE_hypotenuse_length_of_area_49_l4028_402880

/-- An isosceles right triangle with given area and hypotenuse length -/
structure IsoscelesRightTriangle where
  -- The length of a leg
  leg : ℝ
  -- The area of the triangle
  area : ℝ
  -- The hypotenuse length
  hypotenuse : ℝ
  -- Condition: The area is (1/2) * leg^2
  area_eq : area = (1/2) * leg^2
  -- Condition: The hypotenuse is √2 times the leg
  hypotenuse_eq : hypotenuse = Real.sqrt 2 * leg

/-- The main theorem: If the area is 49, then the hypotenuse length is 14 -/
theorem hypotenuse_length_of_area_49 (t : IsoscelesRightTriangle) (h : t.area = 49) :
  t.hypotenuse = 14 := by
  sorry


end NUMINAMATH_CALUDE_hypotenuse_length_of_area_49_l4028_402880


namespace NUMINAMATH_CALUDE_binomial_coefficient_max_l4028_402881

theorem binomial_coefficient_max (n : ℕ) (h : 2^n = 256) : 
  Finset.sup (Finset.range (n + 1)) (fun k => Nat.choose n k) = 70 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_max_l4028_402881


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l4028_402816

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_seven_balls_three_boxes : 
  distribute_balls 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l4028_402816


namespace NUMINAMATH_CALUDE_equation_solution_product_l4028_402811

theorem equation_solution_product : ∃ (r s : ℝ), 
  r ≠ s ∧ 
  (r - 3) * (3 * r + 6) = r^2 - 16 * r + 63 ∧
  (s - 3) * (3 * s + 6) = s^2 - 16 * s + 63 ∧
  (r + 2) * (s + 2) = -19.14 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_product_l4028_402811


namespace NUMINAMATH_CALUDE_tom_weekly_earnings_l4028_402802

/-- Calculates the weekly earnings from crab fishing given the number of buckets, crabs per bucket, price per crab, and days in a week. -/
def weekly_crab_earnings (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_in_week : ℕ) : ℕ :=
  buckets * crabs_per_bucket * price_per_crab * days_in_week

/-- Proves that Tom's weekly earnings from crab fishing is $3360. -/
theorem tom_weekly_earnings : 
  weekly_crab_earnings 8 12 5 7 = 3360 := by
  sorry

end NUMINAMATH_CALUDE_tom_weekly_earnings_l4028_402802


namespace NUMINAMATH_CALUDE_unique_complex_root_l4028_402861

/-- The equation has exactly one complex root if and only if k = 2i or k = -2i -/
theorem unique_complex_root (k : ℂ) : 
  (∃! z : ℂ, (z^2 / (z+1)) + (z^2 / (z+2)) = k*z^2) ↔ k = 2*I ∨ k = -2*I :=
by sorry

end NUMINAMATH_CALUDE_unique_complex_root_l4028_402861


namespace NUMINAMATH_CALUDE_count_special_integers_l4028_402824

theorem count_special_integers : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 2 ≤ n ∧ n ≤ 2016 ∧ 
      (2 ∣ n^n - 1) ∧ (3 ∣ n^n - 1) ∧ (5 ∣ n^n - 1) ∧ (7 ∣ n^n - 1)) ∧
    S.card = 9 ∧
    (∀ n : ℕ, 2 ≤ n ∧ n ≤ 2016 ∧ 
      (2 ∣ n^n - 1) ∧ (3 ∣ n^n - 1) ∧ (5 ∣ n^n - 1) ∧ (7 ∣ n^n - 1) → n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_count_special_integers_l4028_402824


namespace NUMINAMATH_CALUDE_equation_solution_l4028_402830

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ x ≠ -3 ∧ (2 / x + x / (x + 3) = 1) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4028_402830


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l4028_402896

theorem triangle_area_ratio (a b c : ℝ) (A : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  let S := a^2 - (b-c)^2
  S = (1/2) * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  (Real.sin A) / (1 - Real.cos A) = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l4028_402896


namespace NUMINAMATH_CALUDE_min_value_sin_product_l4028_402867

theorem min_value_sin_product (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h_pos : θ₁ > 0 ∧ θ₂ > 0 ∧ θ₃ > 0 ∧ θ₄ > 0) 
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = π) : 
  (2 * Real.sin θ₁ ^ 2 + 1 / Real.sin θ₁ ^ 2) * 
  (2 * Real.sin θ₂ ^ 2 + 1 / Real.sin θ₂ ^ 2) * 
  (2 * Real.sin θ₃ ^ 2 + 1 / Real.sin θ₃ ^ 2) * 
  (2 * Real.sin θ₄ ^ 2 + 1 / Real.sin θ₄ ^ 2) ≥ 81 := by
  sorry

#check min_value_sin_product

end NUMINAMATH_CALUDE_min_value_sin_product_l4028_402867


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l4028_402868

theorem number_puzzle_solution : ∃ x : ℝ, x^2 + 85 = (x - 17)^2 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l4028_402868


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l4028_402885

/-- Represents a repeating decimal with a single digit repeating. -/
def RepeatingDecimal (n : ℕ) : ℚ :=
  n / 9

/-- The sum of 0.666... + 0.222... - 0.444... equals 4/9 -/
theorem repeating_decimal_sum : 
  RepeatingDecimal 6 + RepeatingDecimal 2 - RepeatingDecimal 4 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l4028_402885


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l4028_402825

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 4

def prob_treasure_only : ℚ := 1/5
def prob_traps_only : ℚ := 1/10
def prob_both : ℚ := 1/10
def prob_neither : ℚ := 3/5

def prob_treasure : ℚ := prob_treasure_only + prob_both
def prob_no_treasure_no_traps : ℚ := prob_neither

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  (prob_treasure ^ num_treasure_islands) *
  (prob_no_treasure_no_traps ^ (num_islands - num_treasure_islands)) =
  91854/1250000 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l4028_402825


namespace NUMINAMATH_CALUDE_min_platforms_proof_l4028_402869

/-- The minimum number of platforms required to transport all granite slabs -/
def min_platforms : ℕ := 40

/-- The number of 7-ton granite slabs -/
def slabs_7ton : ℕ := 120

/-- The number of 9-ton granite slabs -/
def slabs_9ton : ℕ := 80

/-- The maximum weight a platform can carry (in tons) -/
def max_platform_weight : ℕ := 40

/-- The weight of a 7-ton slab -/
def weight_7ton : ℕ := 7

/-- The weight of a 9-ton slab -/
def weight_9ton : ℕ := 9

theorem min_platforms_proof :
  min_platforms * 3 ≥ slabs_7ton ∧
  min_platforms * 2 ≥ slabs_9ton ∧
  3 * weight_7ton + 2 * weight_9ton ≤ max_platform_weight ∧
  ∀ n : ℕ, n < min_platforms →
    n * 3 < slabs_7ton ∨ n * 2 < slabs_9ton :=
by sorry

end NUMINAMATH_CALUDE_min_platforms_proof_l4028_402869


namespace NUMINAMATH_CALUDE_detergent_for_clothes_l4028_402865

-- Define the detergent usage rate
def detergent_per_pound : ℝ := 2

-- Define the amount of clothes to be washed
def clothes_weight : ℝ := 9

-- Theorem to prove
theorem detergent_for_clothes : detergent_per_pound * clothes_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_detergent_for_clothes_l4028_402865


namespace NUMINAMATH_CALUDE_existence_of_prime_and_integers_l4028_402882

theorem existence_of_prime_and_integers (k : ℕ+) : 
  ∃ (p : ℕ) (a : Fin (k+3) → ℕ), 
    Prime p ∧ 
    (∀ i : Fin (k+3), 1 ≤ a i ∧ a i < p) ∧
    (∀ i j : Fin (k+3), i ≠ j → a i ≠ a j) ∧
    (∀ i : Fin k, p ∣ (a i * a (i+1) * a (i+2) * a (i+3) - i)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_and_integers_l4028_402882


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4028_402863

theorem trigonometric_identity (α : ℝ) : 
  (Real.cos (2 * α - π / 2) + Real.sin (3 * π - 4 * α) - Real.cos (5 * π / 2 + 6 * α)) / 
  (4 * Real.sin (5 * π - 3 * α) * Real.cos (α - 2 * π)) = Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4028_402863


namespace NUMINAMATH_CALUDE_necessary_condition_inequality_l4028_402800

theorem necessary_condition_inequality (a b c : ℝ) (hc : c ≠ 0) :
  (∀ c, c ≠ 0 → a * c^2 > b * c^2) → a > b :=
sorry

end NUMINAMATH_CALUDE_necessary_condition_inequality_l4028_402800


namespace NUMINAMATH_CALUDE_library_visitors_l4028_402817

theorem library_visitors (sunday_avg : ℕ) (total_days : ℕ) (month_avg : ℕ) :
  sunday_avg = 540 →
  total_days = 30 →
  month_avg = 290 →
  let sundays : ℕ := 5
  let other_days : ℕ := total_days - sundays
  let other_days_avg : ℕ := (total_days * month_avg - sundays * sunday_avg) / other_days
  other_days_avg = 240 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_l4028_402817


namespace NUMINAMATH_CALUDE_indefinite_stick_shortening_l4028_402898

theorem indefinite_stick_shortening :
  ∃ (f : ℕ → ℝ × ℝ × ℝ),
    (∀ n : ℕ, (f n).1 > 0 ∧ (f n).2.1 > 0 ∧ (f n).2.2 > 0) ∧
    (∀ n : ℕ, max (f n).1 (max (f n).2.1 (f n).2.2) > (f n).1 + (f n).2.1 + (f n).2.2 - max (f n).1 (max (f n).2.1 (f n).2.2)) ∧
    (∀ n : ℕ, 
      let (a, b, c) := f n
      let m := max a (max b c)
      f (n + 1) = (if m = a then (b + c, b, c) else if m = b then (a, a + c, c) else (a, b, a + b))) :=
sorry

end NUMINAMATH_CALUDE_indefinite_stick_shortening_l4028_402898


namespace NUMINAMATH_CALUDE_lcm_36_48_75_l4028_402846

theorem lcm_36_48_75 : Nat.lcm (Nat.lcm 36 48) 75 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_48_75_l4028_402846


namespace NUMINAMATH_CALUDE_solution_unique_l4028_402835

theorem solution_unique : 
  ∃! (x y : ℚ), (15 * x + 24 * y = 18) ∧ (24 * x + 15 * y = 63) ∧ (x = 46/13) ∧ (y = -19/13) := by
sorry

end NUMINAMATH_CALUDE_solution_unique_l4028_402835


namespace NUMINAMATH_CALUDE_min_sum_squares_l4028_402856

theorem min_sum_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 11) :
  x^2 + y^2 + z^2 ≥ 121/29 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l4028_402856


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_two_sqrt_three_equals_zero_l4028_402820

theorem sqrt_twelve_minus_two_sqrt_three_equals_zero :
  Real.sqrt 12 - 2 * Real.sqrt 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_two_sqrt_three_equals_zero_l4028_402820


namespace NUMINAMATH_CALUDE_networking_event_handshakes_l4028_402883

/-- Represents a group of people at a networking event -/
structure NetworkingEvent where
  total_people : Nat
  partner_pairs : Nat
  handshakes_per_person : Nat

/-- Calculates the total number of handshakes at a networking event -/
def total_handshakes (event : NetworkingEvent) : Nat :=
  event.total_people * event.handshakes_per_person / 2

/-- Theorem: The number of handshakes at the specific networking event is 60 -/
theorem networking_event_handshakes :
  ∃ (event : NetworkingEvent),
    event.total_people = 12 ∧
    event.partner_pairs = 6 ∧
    event.handshakes_per_person = 10 ∧
    total_handshakes event = 60 := by
  sorry

end NUMINAMATH_CALUDE_networking_event_handshakes_l4028_402883


namespace NUMINAMATH_CALUDE_lizzies_group_area_l4028_402864

/-- The area covered by Lizzie's group given the total area, area covered by another group, and remaining area to be cleaned. -/
theorem lizzies_group_area (total_area other_group_area remaining_area : ℕ) 
  (h1 : total_area = 900)
  (h2 : other_group_area = 265)
  (h3 : remaining_area = 385) :
  total_area - other_group_area - remaining_area = 250 := by
  sorry

end NUMINAMATH_CALUDE_lizzies_group_area_l4028_402864


namespace NUMINAMATH_CALUDE_custom_op_inequality_l4028_402840

/-- Custom operation ⊗ on ℝ -/
def custom_op (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem statement -/
theorem custom_op_inequality (a : ℝ) :
  (∀ x > 2, custom_op (x - a) x ≤ a + 2) →
  a ∈ Set.Iic 7 :=
by sorry

end NUMINAMATH_CALUDE_custom_op_inequality_l4028_402840


namespace NUMINAMATH_CALUDE_tiffany_bags_total_l4028_402822

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 4

/-- The number of bags Tiffany found the next day -/
def next_day_bags : ℕ := 8

/-- The total number of bags Tiffany had -/
def total_bags : ℕ := monday_bags + next_day_bags

theorem tiffany_bags_total : total_bags = 12 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_total_l4028_402822


namespace NUMINAMATH_CALUDE_students_absent_correct_absent_count_l4028_402859

theorem students_absent (students_yesterday : ℕ) (students_registered : ℕ) : ℕ :=
  let students_today := (2 * students_yesterday) - (2 * students_yesterday / 10)
  students_registered - students_today

theorem correct_absent_count : students_absent 70 156 = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_absent_correct_absent_count_l4028_402859


namespace NUMINAMATH_CALUDE_fraction_addition_l4028_402851

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l4028_402851


namespace NUMINAMATH_CALUDE_inequality_proof_l4028_402845

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a + 1/b > b + 1/a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4028_402845


namespace NUMINAMATH_CALUDE_or_implies_at_least_one_true_l4028_402870

theorem or_implies_at_least_one_true (p q : Prop) : 
  (p ∨ q) → (p ∨ q) :=
by
  sorry

end NUMINAMATH_CALUDE_or_implies_at_least_one_true_l4028_402870


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l4028_402860

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ (z₁ : ℂ), z₁ = m * (m - 1) + (m - 1) * Complex.I ∧ z₁.re = 0 ∧ z₁.im ≠ 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l4028_402860


namespace NUMINAMATH_CALUDE_three_digit_special_property_l4028_402855

def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_three_digit (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem three_digit_special_property : 
  {n : Nat | is_three_digit n ∧ 
             is_three_digit (6 * n) ∧ 
             digit_sum n = digit_sum (6 * n)} = {117, 135} := by
  sorry

end NUMINAMATH_CALUDE_three_digit_special_property_l4028_402855


namespace NUMINAMATH_CALUDE_go_game_probabilities_l4028_402858

/-- Represents the probability of a player winning a single game -/
structure GameProbability where
  player_a : ℝ
  player_b : ℝ
  sum_to_one : player_a + player_b = 1

/-- Represents the state of the game after the first two games -/
structure InitialState where
  a_wins : ℕ
  b_wins : ℕ
  total_games : a_wins + b_wins = 2

/-- Calculates the probability of player A winning the competition -/
def probability_a_wins (p : GameProbability) (init : InitialState) : ℝ :=
  p.player_a * p.player_a +
  p.player_b * p.player_a * p.player_a +
  p.player_a * p.player_b * p.player_a

/-- Calculates the probability of the competition ending after 5 games -/
def probability_end_after_five (p : GameProbability) (init : InitialState) : ℝ :=
  p.player_b * p.player_a * p.player_a +
  p.player_a * p.player_b * p.player_a +
  p.player_a * p.player_b * p.player_b +
  p.player_b * p.player_a * p.player_b

/-- The main theorem stating the probabilities for the Go game competition -/
theorem go_game_probabilities 
  (p : GameProbability) 
  (init : InitialState) 
  (h_p : p.player_a = 0.6 ∧ p.player_b = 0.4) 
  (h_init : init.a_wins = 1 ∧ init.b_wins = 1) : 
  probability_a_wins p init = 0.648 ∧ 
  probability_end_after_five p init = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_go_game_probabilities_l4028_402858


namespace NUMINAMATH_CALUDE_percentage_calculation_l4028_402804

theorem percentage_calculation : 
  (789524.37 : ℝ) * (7.5 / 100) = 59214.32825 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l4028_402804


namespace NUMINAMATH_CALUDE_common_root_of_polynomials_l4028_402890

/-- Given three polynomials P, Q, and R, prove that 7 is their common root. -/
theorem common_root_of_polynomials :
  let P : ℝ → ℝ := λ x => x^3 + 41*x^2 - 49*x - 2009
  let Q : ℝ → ℝ := λ x => x^3 + 5*x^2 - 49*x - 245
  let R : ℝ → ℝ := λ x => x^3 + 39*x^2 - 117*x - 1435
  P 7 = 0 ∧ Q 7 = 0 ∧ R 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_root_of_polynomials_l4028_402890


namespace NUMINAMATH_CALUDE_rainfall_second_week_l4028_402893

theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) :
  total_rainfall = 40 →
  ratio = 1.5 →
  ∃ (first_week : ℝ) (second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = ratio * first_week ∧
    second_week = 24 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_second_week_l4028_402893


namespace NUMINAMATH_CALUDE_inequality_and_arithmetic_geometric_mean_l4028_402862

theorem inequality_and_arithmetic_geometric_mean :
  (∀ x y : ℝ, x > 0 → y > 0 → x^3 + y^3 ≥ x^2*y + x*y^2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → (x^3 + y^3 = x^2*y + x*y^2 ↔ x = y)) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → (a + b + c) / 3 ≥ (a*b*c)^(1/3)) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → ((a + b + c) / 3 = (a*b*c)^(1/3) ↔ a = b ∧ b = c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_arithmetic_geometric_mean_l4028_402862


namespace NUMINAMATH_CALUDE_two_thousand_nineteen_in_group_63_l4028_402884

/-- The last number in the nth group -/
def last_in_group (n : ℕ) : ℕ := n * (n + 1) / 2 + n

/-- The first number in the nth group -/
def first_in_group (n : ℕ) : ℕ := last_in_group (n - 1) + 1

/-- Predicate to check if a number is in the nth group -/
def in_group (x n : ℕ) : Prop :=
  first_in_group n ≤ x ∧ x ≤ last_in_group n

theorem two_thousand_nineteen_in_group_63 :
  in_group 2019 63 := by sorry

end NUMINAMATH_CALUDE_two_thousand_nineteen_in_group_63_l4028_402884


namespace NUMINAMATH_CALUDE_line_b_production_l4028_402833

/-- Represents the production of cement bags by three production lines -/
structure CementProduction where
  total : ℕ
  lineA : ℕ
  lineB : ℕ
  lineC : ℕ
  sum_eq_total : lineA + lineB + lineC = total
  arithmetic_sequence : lineA - lineB = lineB - lineC

/-- Theorem stating that under given conditions, production line B produces 6500 bags -/
theorem line_b_production (prod : CementProduction) (h : prod.total = 19500) : 
  prod.lineB = 6500 := by
  sorry

end NUMINAMATH_CALUDE_line_b_production_l4028_402833


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4028_402839

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the nth terms of two arithmetic sequences -/
def SumOfTerms (a b : ℕ → ℝ) (n : ℕ) : ℝ := a n + b n

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  ArithmeticSequence a → ArithmeticSequence b →
  SumOfTerms a b 1 = 7 → SumOfTerms a b 3 = 21 →
  SumOfTerms a b 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4028_402839


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l4028_402886

/-- Represents the speed and distance of a journey between three towns -/
structure JourneyData where
  speed_qb : ℝ  -- Speed from Q to B
  speed_bc : ℝ  -- Speed from B to C
  dist_bc : ℝ   -- Distance from B to C
  avg_speed : ℝ -- Average speed of the whole journey

/-- Theorem stating the conditions and the result to be proved -/
theorem journey_speed_calculation (j : JourneyData) 
  (h1 : j.speed_qb = 60)
  (h2 : j.avg_speed = 36)
  (h3 : j.dist_bc > 0) :
  j.speed_qb = 60 ∧ 
  j.avg_speed = 36 ∧ 
  (3 * j.dist_bc) / (2 * j.dist_bc / j.speed_qb + j.dist_bc / j.speed_bc) = j.avg_speed →
  j.speed_bc = 20 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l4028_402886


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4028_402827

theorem partial_fraction_decomposition (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 2 → (47 * x - 35) / (x^2 - 3*x + 2) = N₁ / (x - 1) + N₂ / (x - 2)) →
  N₁ * N₂ = -708 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4028_402827


namespace NUMINAMATH_CALUDE_discount_percentage_is_five_percent_l4028_402810

def cameras_cost : ℝ := 2 * 110
def frames_cost : ℝ := 3 * 120
def total_cost : ℝ := cameras_cost + frames_cost
def discounted_price : ℝ := 551

theorem discount_percentage_is_five_percent :
  (total_cost - discounted_price) / total_cost * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_is_five_percent_l4028_402810


namespace NUMINAMATH_CALUDE_parking_lot_spaces_l4028_402878

theorem parking_lot_spaces (total_spaces motorcycle_spaces ev_spaces : ℕ)
  (full_size_ratio compact_ratio : ℕ) :
  total_spaces = 750 →
  motorcycle_spaces = 50 →
  ev_spaces = 30 →
  full_size_ratio = 11 →
  compact_ratio = 4 →
  ∃ (full_size_spaces : ℕ),
    full_size_spaces = 489 ∧
    full_size_spaces * compact_ratio = (total_spaces - motorcycle_spaces - ev_spaces - full_size_spaces) * full_size_ratio :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_spaces_l4028_402878


namespace NUMINAMATH_CALUDE_ladder_distance_l4028_402852

theorem ladder_distance (ladder_length height : ℝ) (h1 : ladder_length = 15) (h2 : height = 12) :
  ∃ (distance : ℝ), distance^2 + height^2 = ladder_length^2 ∧ distance = 9 := by
  sorry

end NUMINAMATH_CALUDE_ladder_distance_l4028_402852


namespace NUMINAMATH_CALUDE_largest_prime_1005_digits_squared_minus_one_div_24_l4028_402838

-- Define q as the largest prime with 1005 digits
def q : ℕ := sorry

-- Axiom: q is prime
axiom q_prime : Nat.Prime q

-- Axiom: q has 1005 digits
axiom q_digits : 10^1004 ≤ q ∧ q < 10^1005

-- Theorem to prove
theorem largest_prime_1005_digits_squared_minus_one_div_24 :
  24 ∣ (q^2 - 1) := by sorry

end NUMINAMATH_CALUDE_largest_prime_1005_digits_squared_minus_one_div_24_l4028_402838


namespace NUMINAMATH_CALUDE_imaginary_power_sum_product_l4028_402801

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the periodicity of i
axiom i_period (n : ℕ) : i^(n + 4) = i^n

-- State the theorem
theorem imaginary_power_sum_product : (i^22 + i^222) * i = -2 * i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_product_l4028_402801


namespace NUMINAMATH_CALUDE_ratio_equality_l4028_402808

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x - z) = (x + 2*y) / z ∧ (x + 2*y) / z = x / (y + z)) :
  x / (y + z) = (2*y - z) / (y + z) := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l4028_402808


namespace NUMINAMATH_CALUDE_sprite_liters_sprite_liters_value_l4028_402879

def maaza_liters : ℕ := 60
def pepsi_liters : ℕ := 144
def total_cans : ℕ := 143

theorem sprite_liters : ℕ :=
  let can_size := Nat.gcd maaza_liters pepsi_liters
  let maaza_cans := maaza_liters / can_size
  let pepsi_cans := pepsi_liters / can_size
  let sprite_cans := total_cans - (maaza_cans + pepsi_cans)
  sprite_cans * can_size

theorem sprite_liters_value : sprite_liters = 1512 := by sorry

end NUMINAMATH_CALUDE_sprite_liters_sprite_liters_value_l4028_402879


namespace NUMINAMATH_CALUDE_paul_work_time_l4028_402806

-- Define the work rates and time
def george_work_rate : ℚ := 3 / (5 * 9)
def total_work : ℚ := 1
def george_paul_time : ℚ := 4
def george_initial_work : ℚ := 3 / 5

-- Theorem statement
theorem paul_work_time (paul_work_rate : ℚ) : 
  george_work_rate + paul_work_rate = (total_work - george_initial_work) / george_paul_time →
  total_work / paul_work_rate = 90 / 13 := by
  sorry

end NUMINAMATH_CALUDE_paul_work_time_l4028_402806


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l4028_402803

theorem cubic_sum_theorem (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x*y + x*z + y*z = -5) 
  (h3 : x*y*z = -6) : 
  x^3 + y^3 + z^3 = 18 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l4028_402803


namespace NUMINAMATH_CALUDE_theresas_sons_l4028_402841

theorem theresas_sons (meatballs_per_plate : ℕ) (fraction_eaten : ℚ) (meatballs_left : ℕ) :
  meatballs_per_plate = 3 →
  fraction_eaten = 2/3 →
  meatballs_left = 3 →
  (meatballs_left : ℚ) / (1 - fraction_eaten) = 9 :=
by sorry

end NUMINAMATH_CALUDE_theresas_sons_l4028_402841


namespace NUMINAMATH_CALUDE_min_stamps_47_cents_l4028_402875

/-- Represents the number of ways to make 47 cents using 5 and 7 cent stamps -/
def stamp_combinations : Set (ℕ × ℕ) :=
  {(c, f) | c * 5 + f * 7 = 47 ∧ c ≥ 0 ∧ f ≥ 0}

/-- The total number of stamps used in a combination -/
def total_stamps (combo : ℕ × ℕ) : ℕ :=
  combo.1 + combo.2

/-- The theorem stating the minimum number of stamps needed is 7 -/
theorem min_stamps_47_cents :
  ∃ (min_combo : ℕ × ℕ),
    min_combo ∈ stamp_combinations ∧
    ∀ combo ∈ stamp_combinations, total_stamps min_combo ≤ total_stamps combo ∧
    total_stamps min_combo = 7 :=
  sorry

end NUMINAMATH_CALUDE_min_stamps_47_cents_l4028_402875


namespace NUMINAMATH_CALUDE_tetrahedron_to_polyhedron_ratios_l4028_402899

/-- Regular tetrahedron -/
structure RegularTetrahedron where
  surface_area : ℝ
  volume : ℝ

/-- Polyhedron G formed by removing four smaller tetrahedrons from a regular tetrahedron -/
structure PolyhedronG where
  surface_area : ℝ
  volume : ℝ

/-- Given a regular tetrahedron and the polyhedron G formed from it, 
    prove that the surface area ratio is 9/7 and the volume ratio is 27/23 -/
theorem tetrahedron_to_polyhedron_ratios 
  (p : RegularTetrahedron) 
  (g : PolyhedronG) 
  (h : g = PolyhedronG.mk ((28/36) * p.surface_area) ((23/27) * p.volume)) : 
  p.surface_area / g.surface_area = 9/7 ∧ p.volume / g.volume = 27/23 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_to_polyhedron_ratios_l4028_402899


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l4028_402805

theorem inequality_and_equality_condition (n : ℕ+) :
  (1/3 : ℝ) * n.val^2 + (1/2 : ℝ) * n.val + (1/6 : ℝ) ≥ (n.val.factorial : ℝ)^((2 : ℝ) / n.val) ∧
  ((1/3 : ℝ) * n.val^2 + (1/2 : ℝ) * n.val + (1/6 : ℝ) = (n.val.factorial : ℝ)^((2 : ℝ) / n.val) ↔ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l4028_402805


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l4028_402848

theorem missing_fraction_sum (x : ℚ) : 
  1/3 + 1/2 + (-5/6) + 1/5 + 1/4 + (-5/6) + x = 5/6 → x = 73/60 := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l4028_402848


namespace NUMINAMATH_CALUDE_david_cell_phone_cost_l4028_402813

/-- Calculates the total cost of a cell phone plan. -/
def cell_phone_plan_cost (base_fee : ℝ) (text_cost : ℝ) (extra_minute_cost : ℝ) 
                         (standard_hours : ℝ) (texts_sent : ℕ) (hours_used : ℝ) : ℝ :=
  base_fee + 
  (text_cost * texts_sent) + 
  (extra_minute_cost * (hours_used - standard_hours) * 60)

/-- Theorem stating that David's cell phone plan cost is $54. -/
theorem david_cell_phone_cost : 
  cell_phone_plan_cost 30 0.1 0.15 20 150 21 = 54 := by
  sorry


end NUMINAMATH_CALUDE_david_cell_phone_cost_l4028_402813


namespace NUMINAMATH_CALUDE_apartment_floors_proof_l4028_402821

/-- The number of apartment buildings -/
def num_buildings : ℕ := 2

/-- The number of apartments per floor -/
def apartments_per_floor : ℕ := 6

/-- The number of doors needed per apartment -/
def doors_per_apartment : ℕ := 7

/-- The total number of doors needed -/
def total_doors : ℕ := 1008

/-- The number of floors in each apartment building -/
def floors_per_building : ℕ := 12

theorem apartment_floors_proof :
  floors_per_building * num_buildings * apartments_per_floor * doors_per_apartment = total_doors :=
by sorry

end NUMINAMATH_CALUDE_apartment_floors_proof_l4028_402821


namespace NUMINAMATH_CALUDE_lesser_number_proof_l4028_402871

theorem lesser_number_proof (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : min a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_lesser_number_proof_l4028_402871


namespace NUMINAMATH_CALUDE_min_dimension_sum_for_2541_volume_l4028_402850

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Calculates the sum of dimensions of a box -/
def dimensionSum (d : BoxDimensions) : ℕ := d.length + d.width + d.height

/-- Theorem stating the minimum sum of dimensions for a box with volume 2541 -/
theorem min_dimension_sum_for_2541_volume :
  (∃ (d : BoxDimensions), boxVolume d = 2541) →
  (∃ (d : BoxDimensions), boxVolume d = 2541 ∧ 
    (∀ (d' : BoxDimensions), boxVolume d' = 2541 → dimensionSum d ≤ dimensionSum d') ∧
    dimensionSum d = 191) := by
  sorry

end NUMINAMATH_CALUDE_min_dimension_sum_for_2541_volume_l4028_402850


namespace NUMINAMATH_CALUDE_second_year_associates_percentage_l4028_402877

/-- Represents the percentage of associates in each category -/
structure AssociatePercentages where
  notFirstYear : ℝ
  moreThanTwoYears : ℝ

/-- Calculates the percentage of second-year associates -/
def secondYearPercentage (p : AssociatePercentages) : ℝ :=
  p.notFirstYear - p.moreThanTwoYears

/-- Theorem stating the percentage of second-year associates -/
theorem second_year_associates_percentage 
  (p : AssociatePercentages)
  (h1 : p.notFirstYear = 75)
  (h2 : p.moreThanTwoYears = 50) :
  secondYearPercentage p = 25 := by
  sorry

#check second_year_associates_percentage

end NUMINAMATH_CALUDE_second_year_associates_percentage_l4028_402877


namespace NUMINAMATH_CALUDE_f_plus_f_neg_l4028_402832

def f (x : ℝ) : ℝ := 5 * x^3

theorem f_plus_f_neg (x : ℝ) : f x + f (-x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_plus_f_neg_l4028_402832


namespace NUMINAMATH_CALUDE_cos_seventeen_pi_sixths_l4028_402842

theorem cos_seventeen_pi_sixths : 
  Real.cos (17 * π / 6) = - Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seventeen_pi_sixths_l4028_402842


namespace NUMINAMATH_CALUDE_probability_at_least_two_white_l4028_402823

def total_balls : ℕ := 17
def white_balls : ℕ := 8
def black_balls : ℕ := 9
def drawn_balls : ℕ := 3

theorem probability_at_least_two_white :
  (Nat.choose white_balls 2 * black_balls +
   Nat.choose white_balls 3) /
  Nat.choose total_balls drawn_balls =
  154 / 340 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_two_white_l4028_402823


namespace NUMINAMATH_CALUDE_simple_interest_years_l4028_402887

theorem simple_interest_years (principal interest_amount rate : ℝ) (h1 : principal = 1600)
  (h2 : interest_amount = 200) (h3 : rate = 3.125) :
  (interest_amount * 100) / (principal * rate) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_years_l4028_402887


namespace NUMINAMATH_CALUDE_square_difference_equals_810_l4028_402814

theorem square_difference_equals_810 : (27 + 15)^2 - (27^2 + 15^2) = 810 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_810_l4028_402814


namespace NUMINAMATH_CALUDE_decagon_diagonals_l4028_402818

/-- The number of distinct diagonals in a convex decagon -/
def num_diagonals_decagon : ℕ := 35

/-- A convex decagon is a 10-sided polygon -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals_decagon = (decagon_sides * (decagon_sides - 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l4028_402818
