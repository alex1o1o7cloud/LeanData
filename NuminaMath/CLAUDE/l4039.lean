import Mathlib

namespace z_plus_two_over_z_traces_ellipse_l4039_403994

/-- Given a complex number z with |z| = 3, prove that z + 2/z traces an ellipse -/
theorem z_plus_two_over_z_traces_ellipse (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  ∀ (w : ℂ), w = z + 2 / z → (w.re / a)^2 + (w.im / b)^2 = 1 :=
sorry

end z_plus_two_over_z_traces_ellipse_l4039_403994


namespace total_gross_profit_after_discounts_l4039_403900

/-- Calculate the total gross profit for three items after discounts --/
theorem total_gross_profit_after_discounts
  (price_A price_B price_C : ℝ)
  (gross_profit_percentage : ℝ)
  (discount_A discount_B discount_C : ℝ)
  (h1 : price_A = 91)
  (h2 : price_B = 110)
  (h3 : price_C = 240)
  (h4 : gross_profit_percentage = 1.60)
  (h5 : discount_A = 0.10)
  (h6 : discount_B = 0.05)
  (h7 : discount_C = 0.12) :
  let cost_A := price_A / (1 + gross_profit_percentage)
  let cost_B := price_B / (1 + gross_profit_percentage)
  let cost_C := price_C / (1 + gross_profit_percentage)
  let discounted_price_A := price_A * (1 - discount_A)
  let discounted_price_B := price_B * (1 - discount_B)
  let discounted_price_C := price_C * (1 - discount_C)
  let gross_profit_A := discounted_price_A - cost_A
  let gross_profit_B := discounted_price_B - cost_B
  let gross_profit_C := discounted_price_C - cost_C
  let total_gross_profit := gross_profit_A + gross_profit_B + gross_profit_C
  ∃ ε > 0, |total_gross_profit - 227.98| < ε :=
by
  sorry

end total_gross_profit_after_discounts_l4039_403900


namespace parallel_transitivity_perpendicular_to_parallel_not_always_intersects_l4039_403910

-- Define a 3D space
structure Space3D where
  -- Add necessary fields for 3D space

-- Define a line in 3D space
structure Line3D where
  -- Add necessary fields for a line in 3D space

-- Define parallel lines in 3D space
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

-- Define perpendicular lines in 3D space
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

-- Define line intersection in 3D space
def intersects (l1 l2 : Line3D) : Prop :=
  sorry

theorem parallel_transitivity (l1 l2 l3 : Line3D) :
  parallel l1 l2 → parallel l2 l3 → parallel l1 l3 :=
  sorry

theorem perpendicular_to_parallel (l1 l2 l3 : Line3D) :
  parallel l1 l2 → perpendicular l3 l1 → perpendicular l3 l2 :=
  sorry

theorem not_always_intersects (l1 l2 l3 : Line3D) :
  ¬(parallel l1 l2 → intersects l3 l1 → intersects l3 l2) :=
  sorry

end parallel_transitivity_perpendicular_to_parallel_not_always_intersects_l4039_403910


namespace haji_mother_sales_l4039_403984

/-- Calculate the total sales for Haji's mother given the following conditions:
  - Tough week sales: $800
  - Tough week sales are half of good week sales
  - Number of good weeks: 5
  - Number of tough weeks: 3
-/
theorem haji_mother_sales (tough_week_sales : ℕ) (good_weeks : ℕ) (tough_weeks : ℕ)
  (h1 : tough_week_sales = 800)
  (h2 : good_weeks = 5)
  (h3 : tough_weeks = 3) :
  tough_week_sales * 2 * good_weeks + tough_week_sales * tough_weeks = 10400 :=
by sorry

end haji_mother_sales_l4039_403984


namespace bookstore_earnings_difference_l4039_403922

/-- Represents the earnings difference between two books --/
def earnings_difference (price_top : ℕ) (price_abc : ℕ) (quantity_top : ℕ) (quantity_abc : ℕ) : ℕ :=
  (price_top * quantity_top) - (price_abc * quantity_abc)

/-- Theorem: The earnings difference between "TOP" and "ABC" books is $12 --/
theorem bookstore_earnings_difference :
  earnings_difference 8 23 13 4 = 12 := by
  sorry

end bookstore_earnings_difference_l4039_403922


namespace linear_function_property_l4039_403914

-- Define a linear function g
def g (x : ℝ) : ℝ := sorry

-- State the theorem
theorem linear_function_property :
  (∀ x y a b : ℝ, g (a * x + b * y) = a * g x + b * g y) →  -- g is linear
  (∀ x : ℝ, g x = 3 * g⁻¹ x + 5) →  -- g(x) = 3g^(-1)(x) + 5
  g 0 = 3 →  -- g(0) = 3
  g (-1) = 3 - Real.sqrt 3 :=  -- g(-1) = 3 - √3
by sorry

end linear_function_property_l4039_403914


namespace outfit_combinations_l4039_403997

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (shoes : ℕ) :
  shirts = 4 → pants = 5 → shoes = 3 →
  shirts * pants * shoes = 60 :=
by
  sorry

end outfit_combinations_l4039_403997


namespace negative_two_in_M_l4039_403905

def M : Set ℝ := {x | x^2 - 4 = 0}

theorem negative_two_in_M : -2 ∈ M := by
  sorry

end negative_two_in_M_l4039_403905


namespace kennel_cat_dog_ratio_l4039_403938

theorem kennel_cat_dog_ratio :
  ∀ (num_dogs num_cats : ℕ),
    num_dogs = 32 →
    num_cats = num_dogs - 8 →
    (num_cats : ℚ) / (num_dogs : ℚ) = 3 / 4 := by
  sorry

end kennel_cat_dog_ratio_l4039_403938


namespace intersection_of_sets_l4039_403916

theorem intersection_of_sets : 
  let P : Set ℕ := {3, 5, 6, 8}
  let Q : Set ℕ := {4, 5, 7, 8}
  P ∩ Q = {5, 8} := by
  sorry

end intersection_of_sets_l4039_403916


namespace initial_water_percentage_l4039_403940

theorem initial_water_percentage 
  (initial_volume : ℝ) 
  (added_water : ℝ) 
  (final_water_percentage : ℝ) :
  initial_volume = 120 →
  added_water = 8 →
  final_water_percentage = 25 →
  (initial_volume * (20 / 100) + added_water) / (initial_volume + added_water) = final_water_percentage / 100 :=
by sorry

end initial_water_percentage_l4039_403940


namespace line_intersects_circle_l4039_403915

/-- The line l defined by 2mx - y - 8m - 3 = 0 -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  2 * m * x - y - 8 * m - 3 = 0

/-- The circle C defined by (x - 3)² + (y + 6)² = 25 -/
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 6)^2 = 25

/-- The theorem stating that the line l intersects the circle C for any real m -/
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ x y : ℝ, line_l m x y ∧ circle_C x y :=
sorry

end line_intersects_circle_l4039_403915


namespace trig_identity_l4039_403921

theorem trig_identity (α : Real) (h : (1 + Real.tan α) / (1 - Real.tan α) = 2012) :
  1 / Real.cos (2 * α) + Real.tan (2 * α) = 2012 := by
  sorry

end trig_identity_l4039_403921


namespace four_digit_difference_l4039_403980

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ n / 1000 = 7

def reverse_last_three_digits (n : ℕ) : ℕ :=
  let a := (n / 100) % 10
  let b := (n / 10) % 10
  let c := n % 10
  1000 * c + 100 * b + 10 * a + 7

theorem four_digit_difference (n : ℕ) : 
  is_valid_number n → n = reverse_last_three_digits n + 3546 → 
  n = 7053 ∨ n = 7163 ∨ n = 7273 ∨ n = 7383 ∨ n = 7493 :=
sorry

end four_digit_difference_l4039_403980


namespace parallel_vectors_x_value_l4039_403946

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![2, x]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 8]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, v i = k * w i)

-- Theorem statement
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (a x) (b x) → x = 4 ∨ x = -4 := by
  sorry

end parallel_vectors_x_value_l4039_403946


namespace road_trip_ratio_l4039_403937

theorem road_trip_ratio : 
  ∀ (x : ℝ),
  x > 0 →
  x + 2*x + 40 + 2*(x + 2*x + 40) = 560 →
  40 / x = 9 / 11 := by
  sorry

end road_trip_ratio_l4039_403937


namespace floor_equation_solution_l4039_403990

theorem floor_equation_solution (x : ℝ) : 
  (⌊(5 + 6*x) / 8⌋ : ℝ) = (15*x - 7) / 5 ↔ x = 7/15 ∨ x = 4/5 := by
  sorry

end floor_equation_solution_l4039_403990


namespace no_xy_term_iff_k_eq_four_l4039_403943

/-- The polynomial multiplication (x+2y)(2x-ky-1) does not contain the term xy if and only if k = 4 -/
theorem no_xy_term_iff_k_eq_four (k : ℝ) : 
  (∀ x y : ℝ, (x + 2*y) * (2*x - k*y - 1) = 2*x^2 - x - 2*k*y^2 - 2*y) ↔ k = 4 := by
  sorry

end no_xy_term_iff_k_eq_four_l4039_403943


namespace filter_kit_cost_difference_l4039_403993

/-- Proves that buying the camera lens filter kit costs more than buying filters individually -/
theorem filter_kit_cost_difference : 
  let kit_price : ℚ := 87.5
  let filter_price_1 : ℚ := 16.45
  let filter_price_2 : ℚ := 14.05
  let filter_price_3 : ℚ := 19.5
  let individual_total : ℚ := 2 * filter_price_1 + 2 * filter_price_2 + filter_price_3
  kit_price - individual_total = 7 :=
by sorry

end filter_kit_cost_difference_l4039_403993


namespace root_sum_squares_l4039_403908

theorem root_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 2*h*x = 3 ∧ y^2 + 2*h*y = 3 ∧ x^2 + y^2 = 10) → 
  |h| = 1 :=
by sorry

end root_sum_squares_l4039_403908


namespace system_solution_l4039_403918

theorem system_solution : ∃! (x y : ℚ), 
  2 * x - 3 * y = 5 ∧ 
  4 * x - 6 * y = 10 ∧ 
  x + y = 7 ∧ 
  x = 26 / 5 ∧ 
  y = 9 / 5 := by
  sorry

end system_solution_l4039_403918


namespace square_root_calculations_l4039_403981

theorem square_root_calculations : 
  (Real.sqrt 3)^2 = 3 ∧ Real.sqrt 8 * Real.sqrt 2 = 4 := by
  sorry

end square_root_calculations_l4039_403981


namespace expression_equals_zero_l4039_403974

theorem expression_equals_zero (x y : ℝ) : 
  (5 * x^2 - 3 * x + 2) * (107 - 107) + (7 * y^2 + 4 * y - 1) * (93 - 93) = 0 := by
  sorry

end expression_equals_zero_l4039_403974


namespace same_terminal_side_l4039_403933

theorem same_terminal_side : ∀ (k : ℤ), 95 = -265 + k * 360 → 95 ≡ -265 [ZMOD 360] := by
  sorry

end same_terminal_side_l4039_403933


namespace tan_double_angle_gt_double_tan_l4039_403907

theorem tan_double_angle_gt_double_tan (α : Real) (h1 : 0 < α) (h2 : α < π/4) :
  Real.tan (2 * α) > 2 * Real.tan α := by
  sorry

end tan_double_angle_gt_double_tan_l4039_403907


namespace max_distance_to_c_l4039_403948

/-- The maximum distance from the origin to point C in an equilateral triangle ABC, 
    where A is on the unit circle and B is at (3,0) -/
theorem max_distance_to_c (A B C : ℝ × ℝ) : 
  (A.1^2 + A.2^2 = 1) →  -- A is on the unit circle
  (B = (3, 0)) →         -- B is at (3,0)
  (dist A B = dist B C ∧ dist B C = dist C A) →  -- ABC is equilateral
  (∃ (D : ℝ × ℝ), (D.1^2 + D.2^2 = 1) ∧  -- D is another point on the unit circle
    (dist D B = dist B C ∧ dist B C = dist C D) →  -- DBC is also equilateral
    dist (0, 0) C ≤ 4) :=  -- The distance from O to C is at most 4
by sorry

end max_distance_to_c_l4039_403948


namespace smallest_candy_count_l4039_403979

theorem smallest_candy_count : ∃ (n : ℕ), 
  100 ≤ n ∧ n < 1000 ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 9) % 6 = 0 → False) ∧
  n = 101 := by
sorry

end smallest_candy_count_l4039_403979


namespace function_satisfying_equation_l4039_403927

theorem function_satisfying_equation (r s : ℚ) :
  ∀ f : ℚ → ℚ, (∀ x y : ℚ, f (x + f y) = f (x + r) + y + s) →
  (∀ x : ℚ, f x = x + r + s) ∨ (∀ x : ℚ, f x = -x + r - s) :=
by sorry

end function_satisfying_equation_l4039_403927


namespace cube_ending_with_ones_l4039_403956

theorem cube_ending_with_ones (k : ℕ) : ∃ n : ℤ, ∃ m : ℕ, n^3 = m * 10^k + (10^k - 1) := by
  sorry

end cube_ending_with_ones_l4039_403956


namespace proportion_solve_l4039_403989

theorem proportion_solve (x : ℚ) : (3 : ℚ) / 12 = x / 16 → x = 4 := by
  sorry

end proportion_solve_l4039_403989


namespace remainder_3_87_plus_5_mod_9_l4039_403988

theorem remainder_3_87_plus_5_mod_9 : (3^87 + 5) % 9 = 5 := by
  sorry

end remainder_3_87_plus_5_mod_9_l4039_403988


namespace geometric_sequence_ratio_l4039_403970

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- Positive terms
  (q ≠ 1) →  -- Common ratio not equal to 1
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence definition
  (a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1) →  -- Arithmetic sequence condition
  ((a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2) := by
  sorry

end geometric_sequence_ratio_l4039_403970


namespace cyclist_round_trip_l4039_403941

/-- A cyclist's round trip with given conditions -/
theorem cyclist_round_trip (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (distance1 : ℝ) (distance2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = distance1 + distance2)
  (h2 : distance1 = 12)
  (h3 : distance2 = 24)
  (h4 : speed1 = 8)
  (h5 : speed2 = 12)
  (h6 : total_time = 7.5) :
  (2 * total_distance) / (total_time - (distance1 / speed1 + distance2 / speed2)) = 9 := by
sorry

end cyclist_round_trip_l4039_403941


namespace f_domain_and_range_l4039_403911

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt (1 - Real.cos (2 * x) + 2 * Real.sin x) + 1 / Real.sqrt (Real.sin x ^ 2 + Real.sin x)

def domain (x : ℝ) : Prop := ∃ k : ℤ, 2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi

theorem f_domain_and_range :
  (∀ x : ℝ, f x ≠ 0 → domain x) ∧
  (∀ y : ℝ, y ≥ 2 * (2 : ℝ) ^ (1/4) → ∃ x : ℝ, f x = y) :=
sorry

end f_domain_and_range_l4039_403911


namespace sum_remainder_modulo_9_l4039_403951

theorem sum_remainder_modulo_9 : 
  (8 + 77 + 666 + 5555 + 44444 + 333333 + 2222222 + 11111111) % 9 = 3 := by
  sorry

end sum_remainder_modulo_9_l4039_403951


namespace line_parameterization_l4039_403962

/-- Given a line y = 3x - 11 parameterized by (x, y) = (r, 1) + t(4, k),
    prove that r = 4 and k = 12 -/
theorem line_parameterization (r k : ℝ) : 
  (∀ t : ℝ, (r + 4*t, 1 + k*t) ∈ {p : ℝ × ℝ | p.2 = 3*p.1 - 11}) ↔ 
  (r = 4 ∧ k = 12) :=
by sorry

end line_parameterization_l4039_403962


namespace binary_110101101_equals_429_l4039_403919

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110101101_equals_429 :
  binary_to_decimal [true, false, true, true, false, true, false, true, true] = 429 := by
  sorry

end binary_110101101_equals_429_l4039_403919


namespace incorrect_negation_l4039_403957

theorem incorrect_negation : 
  ¬(¬(∀ x : ℝ, x^2 - x = 0 → x = 0 ∨ x = 1) ↔ 
    (∀ x : ℝ, x^2 - x = 0 → x ≠ 0 ∧ x ≠ 1)) := by sorry

end incorrect_negation_l4039_403957


namespace product_list_price_l4039_403902

/-- Given a product with the following properties:
  - Sold at 90% of its list price
  - Earns a profit of 20%
  - Has a cost price of 21 yuan
  Prove that its list price is 28 yuan. -/
theorem product_list_price (list_price : ℝ) : 
  (0.9 * list_price - 21 = 21 * 0.2) → list_price = 28 := by
  sorry

end product_list_price_l4039_403902


namespace strawberry_picking_problem_l4039_403983

/-- The strawberry picking problem -/
theorem strawberry_picking_problem 
  (betty_strawberries : ℕ)
  (matthew_strawberries : ℕ)
  (natalie_strawberries : ℕ)
  (strawberries_per_jar : ℕ)
  (price_per_jar : ℕ)
  (total_money_made : ℕ)
  (h1 : betty_strawberries = 16)
  (h2 : matthew_strawberries > betty_strawberries)
  (h3 : matthew_strawberries = 2 * natalie_strawberries)
  (h4 : strawberries_per_jar = 7)
  (h5 : price_per_jar = 4)
  (h6 : total_money_made = 40)
  (h7 : betty_strawberries + matthew_strawberries + natalie_strawberries = 
        (total_money_made / price_per_jar) * strawberries_per_jar) :
  matthew_strawberries - betty_strawberries = 20 := by
sorry

end strawberry_picking_problem_l4039_403983


namespace pecans_weight_l4039_403912

def total_nuts : ℝ := 0.52
def almonds : ℝ := 0.14

theorem pecans_weight : total_nuts - almonds = 0.38 := by
  sorry

end pecans_weight_l4039_403912


namespace equation_solution_l4039_403996

theorem equation_solution :
  ∃ x : ℝ, (x^2 - x + 2) / (x - 1) = x + 3 ∧ x = 5/3 :=
by sorry

end equation_solution_l4039_403996


namespace pascal_triangle_30_rows_sum_l4039_403934

/-- The number of entries in the nth row of Pascal's Triangle -/
def pascalRowEntries (n : ℕ) : ℕ := n + 1

/-- The sum of entries in the first n rows of Pascal's Triangle -/
def pascalTriangleSum (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem pascal_triangle_30_rows_sum :
  pascalTriangleSum 30 = 465 := by
  sorry

end pascal_triangle_30_rows_sum_l4039_403934


namespace polar_to_cartesian_circle_l4039_403966

/-- Given a polar coordinate equation r = 3, prove it represents a circle with radius 3 centered at the origin in Cartesian coordinates. -/
theorem polar_to_cartesian_circle (x y : ℝ) : 
  (∃ θ : ℝ, x = 3 * Real.cos θ ∧ y = 3 * Real.sin θ) ↔ x^2 + y^2 = 9 := by sorry

end polar_to_cartesian_circle_l4039_403966


namespace unique_determination_from_sums_and_products_l4039_403917

theorem unique_determination_from_sums_and_products 
  (x y z : ℝ) 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ y ≠ z) 
  (sum_xy sum_xz sum_yz : ℝ) 
  (prod_xy prod_xz prod_yz : ℝ) 
  (h_sums : sum_xy = x + y ∧ sum_xz = x + z ∧ sum_yz = y + z) 
  (h_prods : prod_xy = x * y ∧ prod_xz = x * z ∧ prod_yz = y * z) :
  ∃! (a b c : ℝ), (a = x ∧ b = y ∧ c = z) ∨ (a = x ∧ b = z ∧ c = y) ∨ 
                   (a = y ∧ b = x ∧ c = z) ∨ (a = y ∧ b = z ∧ c = x) ∨ 
                   (a = z ∧ b = x ∧ c = y) ∨ (a = z ∧ b = y ∧ c = x) :=
by sorry

end unique_determination_from_sums_and_products_l4039_403917


namespace conic_is_ellipse_l4039_403932

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 + 3*y^2 - 6*x - 12*y + 9 = 0

/-- The standard form of an ellipse equation -/
def is_ellipse (h k a b : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- Theorem stating that the given equation represents an ellipse -/
theorem conic_is_ellipse :
  ∃ (h k a b : ℝ), ∀ (x y : ℝ),
    conic_equation x y ↔ is_ellipse h k a b x y :=
sorry

end conic_is_ellipse_l4039_403932


namespace expression_bounds_bounds_are_tight_l4039_403945

theorem expression_bounds (p q r s : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) (hr : 0 ≤ r ∧ r ≤ 1) (hs : 0 ≤ s ∧ s ≤ 1) : 
  2 * Real.sqrt 2 ≤ Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) ∧
  Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) ≤ 4 :=
by sorry

theorem bounds_are_tight : 
  ∃ (p q r s : ℝ), (0 ≤ p ∧ p ≤ 1) ∧ (0 ≤ q ∧ q ≤ 1) ∧ (0 ≤ r ∧ r ≤ 1) ∧ (0 ≤ s ∧ s ≤ 1) ∧
    Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) = 2 * Real.sqrt 2 ∧
  ∃ (p q r s : ℝ), (0 ≤ p ∧ p ≤ 1) ∧ (0 ≤ q ∧ q ≤ 1) ∧ (0 ≤ r ∧ r ≤ 1) ∧ (0 ≤ s ∧ s ≤ 1) ∧
    Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) = 4 :=
by sorry

end expression_bounds_bounds_are_tight_l4039_403945


namespace system_solutions_l4039_403969

def is_solution (x y z : ℝ) : Prop :=
  x + y + z = 3 ∧
  x + 2*y - z = 2 ∧
  x + y*z + z*x = 3

theorem system_solutions :
  (∃ (x y z : ℝ), is_solution x y z) ∧
  (∀ (x y z : ℝ), is_solution x y z →
    ((x = 6 + Real.sqrt 29 ∧
      y = (-7 - 2 * Real.sqrt 29) / 3 ∧
      z = (-2 - Real.sqrt 29) / 3) ∨
     (x = 6 - Real.sqrt 29 ∧
      y = (-7 + 2 * Real.sqrt 29) / 3 ∧
      z = (-2 + Real.sqrt 29) / 3))) :=
by sorry

end system_solutions_l4039_403969


namespace conference_tables_needed_l4039_403952

-- Define the base 7 number
def base7_number : ℕ := 312

-- Define the base conversion function
def base7_to_decimal (n : ℕ) : ℕ :=
  3 * 7^2 + 1 * 7^1 + 2 * 7^0

-- Define the number of attendees per table
def attendees_per_table : ℕ := 3

-- Theorem statement
theorem conference_tables_needed :
  (base7_to_decimal base7_number) / attendees_per_table = 52 := by
  sorry

end conference_tables_needed_l4039_403952


namespace rivertown_puzzle_l4039_403939

theorem rivertown_puzzle (p h s c d : ℕ) : 
  p = 4 * h →
  s = 5 * c →
  d = 4 * p →
  ¬ ∃ (h c : ℕ), 99 = 21 * h + 6 * c :=
by sorry

end rivertown_puzzle_l4039_403939


namespace not_divisible_five_power_l4039_403935

theorem not_divisible_five_power (n k : ℕ) : ¬ ((5^k - 1) ∣ (5^n + 1)) := by
  sorry

end not_divisible_five_power_l4039_403935


namespace distance_between_complex_points_l4039_403913

theorem distance_between_complex_points :
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 17 :=
by sorry

end distance_between_complex_points_l4039_403913


namespace sum_of_reciprocals_l4039_403973

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 50) (h2 : x * y = 25) :
  1 / x + 1 / y = 2 := by
sorry

end sum_of_reciprocals_l4039_403973


namespace perpendicular_tangent_line_l4039_403965

/-- The curve y = x^3 -/
def f (x : ℝ) : ℝ := x^3

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem perpendicular_tangent_line (a b : ℝ) :
  (∃ (x y : ℝ), a * x - b * y - 2 = 0) →  -- Given line exists
  f 1 = 1 →  -- Point (1,1) is on the curve
  (a / b) * (f' 1) = -1 →  -- Perpendicular condition
  b / a = -3 := by
  sorry

end perpendicular_tangent_line_l4039_403965


namespace common_days_off_l4039_403958

/-- Earl's work cycle in days -/
def earl_cycle : ℕ := 4

/-- Bob's work cycle in days -/
def bob_cycle : ℕ := 10

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Number of common rest days in one LCM period -/
def common_rest_days_per_lcm : ℕ := 2

/-- Theorem stating the number of common days off for Earl and Bob -/
theorem common_days_off : ℕ := by
  sorry

end common_days_off_l4039_403958


namespace hyperbola_equation_for_given_conditions_l4039_403925

/-- A hyperbola with given eccentricity and foci -/
structure Hyperbola where
  eccentricity : ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

/-- Theorem: A hyperbola with eccentricity 2 and foci at (-4,0) and (4,0) has the equation x^2/4 - y^2/12 = 1 -/
theorem hyperbola_equation_for_given_conditions (h : Hyperbola) 
    (h_ecc : h.eccentricity = 2)
    (h_foci : h.focus1 = (-4, 0) ∧ h.focus2 = (4, 0)) :
    ∀ x y : ℝ, hyperbola_equation h x y :=
  sorry

end hyperbola_equation_for_given_conditions_l4039_403925


namespace triangle_third_vertex_l4039_403961

/-- Given a triangle with vertices at (8,5), (0,0), and (x,0) where x < 0,
    if the area of the triangle is 40 square units, then x = -16. -/
theorem triangle_third_vertex (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * abs (8 * 0 - x * 5) = 40 → x = -16 := by
  sorry

end triangle_third_vertex_l4039_403961


namespace smallest_n_congruent_to_neg_2023_mod_9_l4039_403991

theorem smallest_n_congruent_to_neg_2023_mod_9 : 
  ∃ n : ℕ, 
    (4 ≤ n ∧ n ≤ 12) ∧ 
    n ≡ -2023 [ZMOD 9] ∧
    (∀ m : ℕ, (4 ≤ m ∧ m ≤ 12) ∧ m ≡ -2023 [ZMOD 9] → n ≤ m) ∧
    n = 11 :=
by sorry

end smallest_n_congruent_to_neg_2023_mod_9_l4039_403991


namespace vector_problem_l4039_403920

theorem vector_problem (a b : Fin 2 → ℝ) (x : ℝ) 
    (h1 : a + b = ![2, x])
    (h2 : a - b = ![-2, 1])
    (h3 : ‖a‖^2 - ‖b‖^2 = -1) : 
  x = 3 := by sorry

end vector_problem_l4039_403920


namespace child_age_proof_l4039_403987

/-- Represents a family with its members and their ages -/
structure Family where
  members : ℕ
  total_age : ℕ

/-- Calculates the average age of a family -/
def average_age (f : Family) : ℚ :=
  f.total_age / f.members

theorem child_age_proof (initial_family : Family)
  (h1 : initial_family.members = 5)
  (h2 : average_age initial_family = 17)
  (h3 : ∃ (new_family : Family),
    new_family.members = initial_family.members + 1 ∧
    new_family.total_age = initial_family.total_age + 3 * initial_family.members + 2 ∧
    average_age new_family = average_age initial_family) :
  2 = 2 := by
  sorry

#check child_age_proof

end child_age_proof_l4039_403987


namespace prob_first_ace_equal_sum_prob_is_one_l4039_403923

/-- Represents a player in the card game -/
inductive Player : Type
| one : Player
| two : Player
| three : Player
| four : Player

/-- The total number of cards in the deck -/
def totalCards : ℕ := 32

/-- The number of aces in the deck -/
def numAces : ℕ := 4

/-- The number of players in the game -/
def numPlayers : ℕ := 4

/-- Calculates the probability of a player getting the first ace -/
def probFirstAce (p : Player) : ℚ :=
  1 / 8

/-- Theorem: The probability of each player getting the first ace is 1/8 -/
theorem prob_first_ace_equal (p : Player) : 
  probFirstAce p = 1 / 8 := by
  sorry

/-- Theorem: The sum of probabilities for all players is 1 -/
theorem sum_prob_is_one : 
  (probFirstAce Player.one) + (probFirstAce Player.two) + 
  (probFirstAce Player.three) + (probFirstAce Player.four) = 1 := by
  sorry

end prob_first_ace_equal_sum_prob_is_one_l4039_403923


namespace quadratic_inequality_range_l4039_403992

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a ≤ 0) ↔ a ≤ 1 := by sorry

end quadratic_inequality_range_l4039_403992


namespace line_inclination_angle_l4039_403982

theorem line_inclination_angle (x y : ℝ) :
  x + Real.sqrt 3 * y - 1 = 0 →
  ∃ θ : ℝ, θ = 5 * Real.pi / 6 ∧ Real.tan θ = -1 / Real.sqrt 3 :=
by sorry

end line_inclination_angle_l4039_403982


namespace inequality_proof_l4039_403995

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by sorry

end inequality_proof_l4039_403995


namespace prime_triple_divisibility_l4039_403909

theorem prime_triple_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  p ∣ (q + r) ∧ q ∣ (r + 2*p) ∧ r ∣ (p + 3*q) →
  ((p = 5 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = 11 ∧ r = 7) ∨ 
   (p = 2 ∧ q = 3 ∧ r = 11)) :=
by sorry

#check prime_triple_divisibility

end prime_triple_divisibility_l4039_403909


namespace complex_power_simplification_l4039_403926

theorem complex_power_simplification :
  ((2 + Complex.I) / (2 - Complex.I)) ^ 150 = 1 := by
  sorry

end complex_power_simplification_l4039_403926


namespace geometric_sequence_ratio_sum_l4039_403947

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) 
  (h1 : k ≠ 0)
  (h2 : p ≠ 1)
  (h3 : r ≠ 1)
  (h4 : p ≠ r)
  (h5 : a₂ = k * p)
  (h6 : a₃ = k * p^2)
  (h7 : b₂ = k * r)
  (h8 : b₃ = k * r^2)
  (h9 : a₃ - b₃ = 4 * (a₂ - b₂)) :
  p + r = 4 := by
sorry

end geometric_sequence_ratio_sum_l4039_403947


namespace division_simplification_l4039_403975

theorem division_simplification : (180 : ℚ) / (12 + 15 * 3) = 180 / 57 := by sorry

end division_simplification_l4039_403975


namespace sebastian_took_no_arabs_l4039_403928

theorem sebastian_took_no_arabs (x : ℕ) (y : ℕ) (z : ℕ) : x > 0 →
  -- x is the initial number of each type of soldier
  -- y is the number of cowboys taken (equal to remaining Eskimos)
  -- z is the number of Arab soldiers taken
  y ≤ x →  -- Number of cowboys taken cannot exceed initial number
  4 * x / 3 = y + (x - y) + x / 3 + z →  -- Total soldiers taken
  z = 0 := by
sorry

end sebastian_took_no_arabs_l4039_403928


namespace factorization_of_difference_of_squares_not_factorization_expansion_not_complete_factorization_not_factorization_expansion_2_l4039_403931

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 4 = (x - 2) * (x + 2) :=
by sorry

theorem not_factorization_expansion (a : ℝ) :
  (a - 1)^2 = a^2 - 2*a + 1 :=
by sorry

theorem not_complete_factorization (x : ℝ) :
  x^2 - 2*x - 6 = x*(x - 2) - 6 :=
by sorry

theorem not_factorization_expansion_2 (x : ℝ) :
  x*(x - 1) = x^2 - x :=
by sorry

end factorization_of_difference_of_squares_not_factorization_expansion_not_complete_factorization_not_factorization_expansion_2_l4039_403931


namespace max_sum_of_integers_l4039_403985

theorem max_sum_of_integers (a c d : ℤ) (b : ℕ+) 
  (eq1 : a + b = c) 
  (eq2 : b + c = d) 
  (eq3 : c + d = a) : 
  a + b + c + d ≤ -5 := by
sorry

end max_sum_of_integers_l4039_403985


namespace sum_equals_zero_l4039_403929

theorem sum_equals_zero (a b c : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a * b + c^2 + 4 = 0) : 
  a + b = 0 := by
  sorry

end sum_equals_zero_l4039_403929


namespace sandys_phone_bill_l4039_403977

theorem sandys_phone_bill (kim_age : ℕ) (sandy_age : ℕ) (sandy_bill : ℕ) : 
  kim_age = 10 →
  sandy_age + 2 = 3 * (kim_age + 2) →
  sandy_bill = 10 * sandy_age →
  sandy_bill = 340 :=
by
  sorry

end sandys_phone_bill_l4039_403977


namespace angle_C_in_right_triangle_l4039_403930

-- Define a right triangle ABC
structure RightTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  right_angle : A = 90
  angle_sum : A + B + C = 180

-- Theorem statement
theorem angle_C_in_right_triangle (t : RightTriangle) (h : t.B = 50) : t.C = 40 := by
  sorry

end angle_C_in_right_triangle_l4039_403930


namespace pencil_distribution_l4039_403955

theorem pencil_distribution (P : ℕ) (h : P % 9 = 8) :
  ∃ k : ℕ, P = 9 * k + 8 := by
sorry

end pencil_distribution_l4039_403955


namespace johns_weekly_sleep_l4039_403901

/-- Calculates the total sleep for a week given specific sleep patterns -/
def totalSleepInWeek (daysInWeek : ℕ) (lowSleepDays : ℕ) (lowSleepHours : ℝ) 
                     (recommendedSleep : ℝ) (percentNormalSleep : ℝ) : ℝ :=
  let normalSleepDays := daysInWeek - lowSleepDays
  let normalSleepHours := recommendedSleep * percentNormalSleep
  lowSleepDays * lowSleepHours + normalSleepDays * normalSleepHours

/-- Proves that John's total sleep for the week is 30 hours -/
theorem johns_weekly_sleep : 
  totalSleepInWeek 7 2 3 8 0.6 = 30 := by
  sorry


end johns_weekly_sleep_l4039_403901


namespace regular_polygon_sides_l4039_403950

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (∀ θ : ℝ, θ = 150 ∧ θ = (n - 2 : ℝ) * 180 / n) → n = 12 := by
  sorry

end regular_polygon_sides_l4039_403950


namespace systematic_sampling_interval_l4039_403906

theorem systematic_sampling_interval
  (population_size : ℕ)
  (sample_size : ℕ)
  (h1 : population_size = 800)
  (h2 : sample_size = 40)
  : population_size / sample_size = 20 := by
  sorry

end systematic_sampling_interval_l4039_403906


namespace sqrt_expression_equality_l4039_403964

theorem sqrt_expression_equality (x : ℝ) :
  (Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt 0.81 / Real.sqrt x = 2.507936507936508) →
  x = 0.49 := by
sorry

end sqrt_expression_equality_l4039_403964


namespace largest_four_digit_with_product_72_l4039_403953

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_product (n : ℕ) : ℕ :=
  (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_with_product_72 :
  ∃ M : ℕ, is_four_digit M ∧ 
    digit_product M = 72 ∧
    (∀ n : ℕ, is_four_digit n → digit_product n = 72 → n ≤ M) ∧
    digit_sum M = 17 := by
  sorry

end largest_four_digit_with_product_72_l4039_403953


namespace sweater_selling_price_l4039_403949

/-- The selling price of a sweater given the cost of materials and total gain -/
theorem sweater_selling_price 
  (balls_per_sweater : ℕ) 
  (cost_per_ball : ℕ) 
  (total_gain : ℕ) 
  (num_sweaters : ℕ) : 
  balls_per_sweater = 4 → 
  cost_per_ball = 6 → 
  total_gain = 308 → 
  num_sweaters = 28 → 
  (balls_per_sweater * cost_per_ball * num_sweaters + total_gain) / num_sweaters = 35 := by
  sorry

#check sweater_selling_price

end sweater_selling_price_l4039_403949


namespace isosceles_triangle_area_l4039_403963

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  /-- The length of the altitude to the base -/
  altitude : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : True

/-- The area of an isosceles triangle -/
def triangle_area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area : 
  ∀ (t : IsoscelesTriangle), t.altitude = 10 ∧ t.perimeter = 40 → triangle_area t = 75 :=
by sorry

end isosceles_triangle_area_l4039_403963


namespace initial_markup_percentage_l4039_403978

theorem initial_markup_percentage 
  (C : ℝ) 
  (M : ℝ) 
  (h1 : C > 0) 
  (h2 : (C * (1 + M) * 1.25 * 0.75) = (C * 1.125)) : 
  M = 0.2 := by
sorry

end initial_markup_percentage_l4039_403978


namespace distance_between_foci_l4039_403967

def ellipse (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y + 9)^2) = 22

def focus1 : ℝ × ℝ := (4, -5)
def focus2 : ℝ × ℝ := (-6, 9)

theorem distance_between_foci :
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 74 := by
  sorry

end distance_between_foci_l4039_403967


namespace intersection_M_N_l4039_403936

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complement_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := {x | x ∈ U ∧ x ∉ complement_N}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 0} := by
  sorry

end intersection_M_N_l4039_403936


namespace unique_modular_residue_l4039_403960

theorem unique_modular_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -3736 [ZMOD 6] := by
  sorry

end unique_modular_residue_l4039_403960


namespace percent_relationship_l4039_403976

theorem percent_relationship (x y z : ℝ) (h1 : x = 1.20 * y) (h2 : y = 0.70 * z) :
  x = 0.84 * z := by sorry

end percent_relationship_l4039_403976


namespace alternating_arrangements_adjacent_ab_arrangements_l4039_403986

/-- Represents the number of male students -/
def num_male : Nat := 2

/-- Represents the number of female students -/
def num_female : Nat := 3

/-- Represents the total number of students -/
def total_students : Nat := num_male + num_female

/-- Calculates the number of ways to arrange n distinct objects -/
def arrangements (n : Nat) : Nat := Nat.factorial n

/-- Theorem stating the number of alternating arrangements -/
theorem alternating_arrangements : 
  arrangements num_male * arrangements num_female = 12 := by sorry

/-- Theorem stating the number of arrangements with A and B adjacent -/
theorem adjacent_ab_arrangements : 
  arrangements (total_students - 1) * 2 = 48 := by sorry

end alternating_arrangements_adjacent_ab_arrangements_l4039_403986


namespace at_least_two_equal_l4039_403944

theorem at_least_two_equal (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2/y + y^2/z + z^2/x = x^2/z + y^2/x + z^2/y) :
  x = y ∨ y = z ∨ z = x := by
  sorry

end at_least_two_equal_l4039_403944


namespace lisas_lasagna_consumption_l4039_403954

/-- The number of pieces Lisa eats from a lasagna, given the eating habits of her friends. -/
def lisas_lasagna_pieces (total_pieces manny_pieces aaron_pieces : ℚ) : ℚ :=
  let kai_pieces := 2 * manny_pieces
  let raphael_pieces := manny_pieces / 2
  total_pieces - (manny_pieces + kai_pieces + raphael_pieces + aaron_pieces)

/-- Theorem stating that Lisa will eat 2.5 pieces of lasagna given the specific conditions. -/
theorem lisas_lasagna_consumption :
  lisas_lasagna_pieces 6 1 0 = 5/2 := by
  sorry

end lisas_lasagna_consumption_l4039_403954


namespace line_equation_slope_5_through_0_2_l4039_403999

/-- The equation of a line with slope 5 passing through (0, 2) -/
theorem line_equation_slope_5_through_0_2 :
  ∀ (x y : ℝ), (5 * x - y + 2 = 0) ↔ 
  (∃ (t : ℝ), x = t ∧ y = 5 * t + 2) := by sorry

end line_equation_slope_5_through_0_2_l4039_403999


namespace f_properties_l4039_403959

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) * Real.exp x - (Real.exp 1 / 3) * x^3

noncomputable def g (x : ℝ) : ℝ := f x - 2

theorem f_properties :
  (∀ M : ℝ, ∃ x : ℝ, f x > M) ∧
  (∃ x₀ : ℝ, x₀ = 1 ∧ f x₀ = (2/3) * Real.exp 1 ∧ ∀ x : ℝ, f x ≥ f x₀) ∧
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧ ∀ x ∈ Set.Ioo x₁ x₂, g x < 0) :=
by sorry

end f_properties_l4039_403959


namespace summer_program_sophomores_l4039_403968

theorem summer_program_sophomores :
  ∀ (total_students : ℕ) 
    (non_soph_jun : ℕ)
    (soph_debate_ratio : ℚ)
    (jun_debate_ratio : ℚ),
  total_students = 40 →
  non_soph_jun = 5 →
  soph_debate_ratio = 1/5 →
  jun_debate_ratio = 1/4 →
  ∃ (sophomores juniors : ℚ),
    sophomores + juniors = total_students - non_soph_jun ∧
    sophomores * soph_debate_ratio = juniors * jun_debate_ratio ∧
    sophomores = 175/9 :=
by sorry

end summer_program_sophomores_l4039_403968


namespace chris_birthday_money_l4039_403924

theorem chris_birthday_money (x : ℕ) : 
  x + 25 + 20 + 75 = 279 → x = 159 := by
sorry

end chris_birthday_money_l4039_403924


namespace bakery_pie_division_l4039_403903

theorem bakery_pie_division (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 5/6 ∧ num_people = 4 → total_pie / num_people = 5/24 := by
  sorry

end bakery_pie_division_l4039_403903


namespace a_equals_one_sufficient_not_necessary_l4039_403904

theorem a_equals_one_sufficient_not_necessary :
  (∃ a : ℝ, a = 1 → a^2 = 1) ∧ 
  (∃ a : ℝ, a^2 = 1 ∧ a ≠ 1) :=
by sorry

end a_equals_one_sufficient_not_necessary_l4039_403904


namespace perfect_linear_correlation_l4039_403998

/-- A scatter plot where all points lie on a straight line with non-zero slope -/
structure PerfectLinearScatterPlot where
  points : Set (ℝ × ℝ)
  non_zero_slope : ℝ
  line_equation : ℝ → ℝ
  all_points_on_line : ∀ (x y : ℝ), (x, y) ∈ points → y = line_equation x
  slope_non_zero : non_zero_slope ≠ 0

/-- The correlation coefficient of a scatter plot -/
def correlation_coefficient (plot : PerfectLinearScatterPlot) : ℝ :=
  sorry

/-- Theorem: The correlation coefficient of a perfect linear scatter plot is 1 -/
theorem perfect_linear_correlation (plot : PerfectLinearScatterPlot) :
  correlation_coefficient plot = 1 :=
sorry

end perfect_linear_correlation_l4039_403998


namespace age_difference_proof_l4039_403971

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Calculates the age of a person after a given number of years -/
def age_after (p : Person) (years : ℕ) : ℕ := p.age + years

/-- Calculates the age of a person before a given number of years -/
def age_before (p : Person) (years : ℕ) : ℕ := p.age - years

/-- The problem statement -/
theorem age_difference_proof (john james james_brother : Person) 
    (h1 : john.age = 39)
    (h2 : age_before john 3 = 2 * age_after james 6)
    (h3 : james_brother.age = 16) :
  james_brother.age - james.age = 4 := by
  sorry


end age_difference_proof_l4039_403971


namespace square_ratio_side_length_l4039_403942

theorem square_ratio_side_length (area_ratio : ℚ) : 
  area_ratio = 75 / 128 →
  ∃ (a b c : ℕ), 
    (a = 5 ∧ b = 6 ∧ c = 16) ∧
    (Real.sqrt area_ratio * c = a * Real.sqrt b) :=
by sorry

end square_ratio_side_length_l4039_403942


namespace sin_alpha_value_l4039_403972

theorem sin_alpha_value (α : Real) 
  (h1 : 2 * Real.tan α * Real.sin α = 3)
  (h2 : -π/2 < α ∧ α < 0) : 
  Real.sin α = -Real.sqrt 3 / 2 := by
  sorry

end sin_alpha_value_l4039_403972
