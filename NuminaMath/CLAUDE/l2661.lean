import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l2661_266166

theorem simplify_expression (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ 0) (h3 : a ≠ -1) :
  (a^2 - 2*a + 1) / (a^2 - 1) / (a - 2*a / (a + 1)) = 1 / a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2661_266166


namespace NUMINAMATH_CALUDE_min_balls_guarantee_l2661_266143

def red_balls : ℕ := 35
def blue_balls : ℕ := 25
def green_balls : ℕ := 22
def yellow_balls : ℕ := 18
def white_balls : ℕ := 14
def black_balls : ℕ := 12

def total_balls : ℕ := red_balls + blue_balls + green_balls + yellow_balls + white_balls + black_balls

def min_balls_for_guarantee : ℕ := 95

theorem min_balls_guarantee :
  ∀ (drawn : ℕ), drawn ≥ min_balls_for_guarantee →
    ∃ (color : ℕ), color ≥ 18 ∧
      (color ≤ red_balls ∨ color ≤ blue_balls ∨ color ≤ green_balls ∨
       color ≤ yellow_balls ∨ color ≤ white_balls ∨ color ≤ black_balls) :=
by sorry

end NUMINAMATH_CALUDE_min_balls_guarantee_l2661_266143


namespace NUMINAMATH_CALUDE_chicken_difference_l2661_266102

theorem chicken_difference (mary john ray : ℕ) 
  (h1 : john = mary + 5)
  (h2 : ray + 6 = mary)
  (h3 : ray = 10) : 
  john - ray = 11 :=
by sorry

end NUMINAMATH_CALUDE_chicken_difference_l2661_266102


namespace NUMINAMATH_CALUDE_correct_difference_is_1552_l2661_266119

/-- Calculates the correct difference given the erroneous calculation and mistakes made --/
def correct_difference (erroneous_difference : ℕ) 
  (units_mistake : ℕ) (tens_mistake : ℕ) (hundreds_mistake : ℕ) : ℕ :=
  erroneous_difference - hundreds_mistake + tens_mistake - units_mistake

/-- Proves that the correct difference is 1552 given the specific mistakes in the problem --/
theorem correct_difference_is_1552 : 
  correct_difference 1994 2 60 500 = 1552 := by sorry

end NUMINAMATH_CALUDE_correct_difference_is_1552_l2661_266119


namespace NUMINAMATH_CALUDE_no_equal_sums_for_given_sequences_l2661_266146

theorem no_equal_sums_for_given_sequences : ¬ ∃ (n : ℕ), n > 0 ∧
  (let a₁ := 9
   let d₁ := 6
   let t₁ := n * (2 * a₁ + (n - 1) * d₁) / 2
   let a₂ := 11
   let d₂ := 3
   let t₂ := n * (2 * a₂ + (n - 1) * d₂) / 2
   t₁ = t₂) :=
sorry

end NUMINAMATH_CALUDE_no_equal_sums_for_given_sequences_l2661_266146


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2661_266121

theorem cubic_root_sum (a b m n p : ℝ) 
  (hm : m^3 + a*m + b = 0)
  (hn : n^3 + a*n + b = 0)
  (hp : p^3 + a*p + b = 0)
  (hmn : m ≠ n)
  (hnp : n ≠ p)
  (hmp : m ≠ p) :
  m + n + p = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2661_266121


namespace NUMINAMATH_CALUDE_strawberry_plants_l2661_266161

theorem strawberry_plants (initial : ℕ) : 
  (((initial * 2) * 2) * 2) - 4 = 20 → initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_plants_l2661_266161


namespace NUMINAMATH_CALUDE_luka_aubrey_age_difference_l2661_266140

structure Person where
  name : String
  age : ℕ

structure Dog where
  name : String
  age : ℕ

def age_difference (p1 p2 : Person) : ℤ :=
  p1.age - p2.age

theorem luka_aubrey_age_difference 
  (luka aubrey : Person) 
  (max : Dog) 
  (h1 : max.age = luka.age - 4)
  (h2 : aubrey.age = 8)
  (h3 : max.age = 6) : 
  age_difference luka aubrey = 2 := by
sorry

end NUMINAMATH_CALUDE_luka_aubrey_age_difference_l2661_266140


namespace NUMINAMATH_CALUDE_inequality_condition_l2661_266104

theorem inequality_condition (a b : ℝ) : 
  (a * |a + b| < |a| * (a + b)) ↔ (a < 0 ∧ b > -a) := by sorry

end NUMINAMATH_CALUDE_inequality_condition_l2661_266104


namespace NUMINAMATH_CALUDE_max_q_plus_r_for_1057_l2661_266180

theorem max_q_plus_r_for_1057 :
  ∃ (q r : ℕ+), 1057 = 23 * q + r ∧ ∀ (q' r' : ℕ+), 1057 = 23 * q' + r' → q + r ≥ q' + r' :=
by sorry

end NUMINAMATH_CALUDE_max_q_plus_r_for_1057_l2661_266180


namespace NUMINAMATH_CALUDE_simplify_fraction_l2661_266117

theorem simplify_fraction (a : ℝ) (h : a = 5) : 15 * a^4 / (75 * a^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2661_266117


namespace NUMINAMATH_CALUDE_sin_390_degrees_l2661_266170

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_390_degrees_l2661_266170


namespace NUMINAMATH_CALUDE_cubic_inequality_l2661_266192

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 + 30*x > 0 ↔ (0 < x ∧ x < 5) ∨ (x > 6) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2661_266192


namespace NUMINAMATH_CALUDE_reflection_sum_theorem_l2661_266112

def point (x y : ℝ) := (x, y)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def sum_coordinates (p1 p2 : ℝ × ℝ) : ℝ :=
  p1.1 + p1.2 + p2.1 + p2.2

theorem reflection_sum_theorem :
  let C : ℝ × ℝ := point 5 (-3)
  let D : ℝ × ℝ := reflect_over_x_axis C
  sum_coordinates C D = 10 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_theorem_l2661_266112


namespace NUMINAMATH_CALUDE_function_properties_l2661_266150

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_properties (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_neg : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧ (f 2 = f 0) := by sorry

end NUMINAMATH_CALUDE_function_properties_l2661_266150


namespace NUMINAMATH_CALUDE_sum_bound_l2661_266124

theorem sum_bound (w x y z : ℝ) 
  (sum_zero : w + x + y + z = 0) 
  (sum_squares_one : w^2 + x^2 + y^2 + z^2 = 1) : 
  -1 ≤ w*x + x*y + y*z + z*w ∧ w*x + x*y + y*z + z*w ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_bound_l2661_266124


namespace NUMINAMATH_CALUDE_base_conversion_1987_to_base5_l2661_266169

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Theorem: 1987 in base 10 is equal to 30422 in base 5 -/
theorem base_conversion_1987_to_base5 :
  1987 = fromBase5 [2, 2, 4, 0, 3] := by sorry

end NUMINAMATH_CALUDE_base_conversion_1987_to_base5_l2661_266169


namespace NUMINAMATH_CALUDE_a_squared_gt_a_necessary_not_sufficient_l2661_266148

theorem a_squared_gt_a_necessary_not_sufficient :
  (∀ a : ℝ, a > 1 → a^2 > a) ∧
  (∃ a : ℝ, a^2 > a ∧ ¬(a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_a_squared_gt_a_necessary_not_sufficient_l2661_266148


namespace NUMINAMATH_CALUDE_reciprocals_product_l2661_266173

theorem reciprocals_product (a b : ℝ) (h : a * b = 1) : 4 * a * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocals_product_l2661_266173


namespace NUMINAMATH_CALUDE_random_walk_properties_l2661_266113

/-- Represents a random walk on a line. -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of a random walk. -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of a random walk. -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences that achieve the maximum range. -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Theorem stating the properties of the random walk. -/
theorem random_walk_properties (w : RandomWalk) :
  (max_range w = w.a) ∧
  (min_range w = w.a - w.b) ∧
  (max_range_sequences w = w.b + 1) := by
  sorry


end NUMINAMATH_CALUDE_random_walk_properties_l2661_266113


namespace NUMINAMATH_CALUDE_salt_mixing_theorem_l2661_266193

def salt_mixing_problem (x : ℚ) : Prop :=
  let known_salt_weight : ℚ := 40
  let known_salt_price : ℚ := 25 / 100
  let unknown_salt_weight : ℚ := 60
  let total_weight : ℚ := known_salt_weight + unknown_salt_weight
  let selling_price : ℚ := 48 / 100
  let profit_percentage : ℚ := 20 / 100
  let total_cost : ℚ := known_salt_weight * known_salt_price + unknown_salt_weight * x
  let selling_revenue : ℚ := total_weight * selling_price
  selling_revenue = total_cost * (1 + profit_percentage) ∧ x = 50 / 100

theorem salt_mixing_theorem : ∃ x : ℚ, salt_mixing_problem x :=
  sorry

end NUMINAMATH_CALUDE_salt_mixing_theorem_l2661_266193


namespace NUMINAMATH_CALUDE_factorial_divisibility_l2661_266199

theorem factorial_divisibility (m n : ℕ) : 
  ∃ k : ℕ, (Nat.factorial (2*m) * Nat.factorial (2*n)) = 
    k * (Nat.factorial m * Nat.factorial n * Nat.factorial (m+n)) :=
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l2661_266199


namespace NUMINAMATH_CALUDE_min_value_of_f_l2661_266145

def f (x : ℝ) := -2 * x + 5

theorem min_value_of_f :
  ∀ x ∈ Set.Icc 2 4, f x ≥ f 4 ∧ f 4 = -3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2661_266145


namespace NUMINAMATH_CALUDE_flour_weight_acceptable_l2661_266176

/-- A weight is acceptable if it falls within the labeled range -/
def is_acceptable (labeled_weight : ℝ) (tolerance : ℝ) (actual_weight : ℝ) : Prop :=
  actual_weight ≥ labeled_weight - tolerance ∧ actual_weight ≤ labeled_weight + tolerance

/-- Theorem stating that 99.80 kg is acceptable for a bag labeled as 100 ± 0.25 kg -/
theorem flour_weight_acceptable :
  is_acceptable 100 0.25 99.80 := by
  sorry

end NUMINAMATH_CALUDE_flour_weight_acceptable_l2661_266176


namespace NUMINAMATH_CALUDE_system_solution_l2661_266138

theorem system_solution (x y : ℝ) (eq1 : x + 2*y = 1) (eq2 : 2*x + y = 2) : x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2661_266138


namespace NUMINAMATH_CALUDE_difference_of_squares_l2661_266134

theorem difference_of_squares (a b : ℝ) : (a - b) * (-a - b) = b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2661_266134


namespace NUMINAMATH_CALUDE_smallest_equal_hotdogs_and_buns_l2661_266197

theorem smallest_equal_hotdogs_and_buns :
  (∃ n : ℕ+, ∀ k : ℕ+, (∃ m : ℕ+, 6 * k = 8 * m) → n ≤ k) ∧
  (∃ m : ℕ+, 6 * 4 = 8 * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_equal_hotdogs_and_buns_l2661_266197


namespace NUMINAMATH_CALUDE_smallest_number_is_negative_sqrt_5_l2661_266120

theorem smallest_number_is_negative_sqrt_5 :
  let a := (-5 : ℝ)^0
  let b := -Real.sqrt 5
  let c := -(1 / 5 : ℝ)
  let d := |(-5 : ℝ)|
  b < a ∧ b < c ∧ b < d := by sorry

end NUMINAMATH_CALUDE_smallest_number_is_negative_sqrt_5_l2661_266120


namespace NUMINAMATH_CALUDE_intersection_right_angle_coordinates_l2661_266163

-- Define the line and parabola
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define points A and B as intersections
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ parabola A.1 A.2 ∧
  line B.1 B.2 ∧ parabola B.1 B.2 ∧
  A ≠ B

-- Define point C on the parabola
def point_on_parabola (C : ℝ × ℝ) : Prop :=
  parabola C.1 C.2

-- Define the right angle condition
def right_angle (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem intersection_right_angle_coordinates :
  ∀ A B C : ℝ × ℝ,
  intersection_points A B →
  point_on_parabola C →
  right_angle A B C →
  (C = (1, -2) ∨ C = (9, -6)) :=
sorry

end NUMINAMATH_CALUDE_intersection_right_angle_coordinates_l2661_266163


namespace NUMINAMATH_CALUDE_election_result_l2661_266159

theorem election_result (total_votes : ℕ) (majority : ℕ) (winning_percentage : ℚ) : 
  total_votes = 800 →
  majority = 320 →
  winning_percentage = 70 →
  (winning_percentage / 100) * total_votes - ((100 - winning_percentage) / 100) * total_votes = majority :=
by sorry

end NUMINAMATH_CALUDE_election_result_l2661_266159


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l2661_266182

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 16

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 10

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 2

/-- The number of times Terrell must lift the new weights to achieve the same total weight -/
def new_lifts : ℕ := (num_weights * original_lifts * original_weight) / (num_weights * new_weight)

theorem terrell_weight_lifting :
  new_lifts = 40 :=
sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l2661_266182


namespace NUMINAMATH_CALUDE_greatest_common_factor_90_135_180_l2661_266141

theorem greatest_common_factor_90_135_180 : Nat.gcd 90 (Nat.gcd 135 180) = 45 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_90_135_180_l2661_266141


namespace NUMINAMATH_CALUDE_expression_simplification_l2661_266122

theorem expression_simplification : (((3 + 6 + 9 + 12) / 3) + ((3 * 4 - 6) / 2)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2661_266122


namespace NUMINAMATH_CALUDE_stamp_difference_l2661_266123

theorem stamp_difference (k a : ℕ) (h1 : k * 3 = a * 5) 
  (h2 : (k - 12) * 6 = (a + 12) * 8) : k - 12 - (a + 12) = 32 := by
  sorry

end NUMINAMATH_CALUDE_stamp_difference_l2661_266123


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2661_266109

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x - k = 0 ∧ 
   ∀ y : ℝ, y^2 + 3*y - k = 0 → y = x) → 
  k = -9/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2661_266109


namespace NUMINAMATH_CALUDE_stating_regular_polygon_triangle_counts_l2661_266101

variable (n : ℕ)

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- The number of right-angled triangles in a regular polygon with 2n sides -/
def num_right_triangles (n : ℕ) : ℕ := 2*n*(n-1)

/-- The number of acute-angled triangles in a regular polygon with 2n sides -/
def num_acute_triangles (n : ℕ) : ℕ := n*(n-1)*(n-2)/3

/-- 
Theorem stating the number of right-angled and acute-angled triangles 
in a regular polygon with 2n sides
-/
theorem regular_polygon_triangle_counts (n : ℕ) (p : RegularPolygon n) :
  (num_right_triangles n = 2*n*(n-1)) ∧ 
  (num_acute_triangles n = n*(n-1)*(n-2)/3) := by
  sorry


end NUMINAMATH_CALUDE_stating_regular_polygon_triangle_counts_l2661_266101


namespace NUMINAMATH_CALUDE_compare_roots_l2661_266187

theorem compare_roots : 
  let a := (2 : ℝ) ^ (1/2)
  let b := (3 : ℝ) ^ (1/3)
  let c := (8 : ℝ) ^ (1/8)
  let d := (9 : ℝ) ^ (1/9)
  (b > a ∧ b > c ∧ b > d) ∧ (a > c ∧ a > d) := by sorry

end NUMINAMATH_CALUDE_compare_roots_l2661_266187


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l2661_266175

theorem quadratic_equation_k_value (k : ℝ) : 
  (∃ (r₁ r₂ : ℝ), r₁ > r₂ ∧ 
    2 * r₁^2 + 5 * r₁ = k ∧
    2 * r₂^2 + 5 * r₂ = k ∧
    r₁ - r₂ = 5.5) →
  k = -28.875 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l2661_266175


namespace NUMINAMATH_CALUDE_bill_experience_l2661_266189

/-- Represents the work experience of a person -/
structure Experience where
  current : ℕ
  fiveYearsAgo : ℕ

/-- The problem setup -/
def libraryProblem : Prop := ∃ (bill joan : Experience),
  -- Bill's current age
  40 = bill.current + bill.fiveYearsAgo
  -- Joan's current age
  ∧ 50 = joan.current + joan.fiveYearsAgo
  -- 5 years ago, Joan had 3 times as much experience as Bill
  ∧ joan.fiveYearsAgo = 3 * bill.fiveYearsAgo
  -- Now, Joan has twice as much experience as Bill
  ∧ joan.current = 2 * bill.current
  -- Bill's current experience is 10 years
  ∧ bill.current = 10

/-- The theorem to prove -/
theorem bill_experience : libraryProblem := by sorry

end NUMINAMATH_CALUDE_bill_experience_l2661_266189


namespace NUMINAMATH_CALUDE_equation_solution_l2661_266132

theorem equation_solution :
  ∀ x : ℚ, (25 - 7 : ℚ) = 5/2 + x → x = 31/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2661_266132


namespace NUMINAMATH_CALUDE_factorization_equality_l2661_266184

theorem factorization_equality (x : ℝ) :
  (4 * x^3 + 100 * x^2 - 28) - (-9 * x^3 + 2 * x^2 - 28) = 13 * x^2 * (x + 7) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2661_266184


namespace NUMINAMATH_CALUDE_point_M_coordinates_l2661_266194

-- Define point M
def M (a : ℝ) : ℝ × ℝ := (a + 3, a + 1)

-- Define the condition for a point to be on the x-axis
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Theorem statement
theorem point_M_coordinates :
  ∀ a : ℝ, on_x_axis (M a) → M a = (2, 0) :=
by sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l2661_266194


namespace NUMINAMATH_CALUDE_michael_digging_time_l2661_266129

/-- The time it takes Michael to dig his hole given the conditions -/
theorem michael_digging_time 
  (father_rate : ℝ) 
  (father_time : ℝ) 
  (michael_rate : ℝ) 
  (michael_depth_diff : ℝ) :
  father_rate = 4 →
  father_time = 400 →
  michael_rate = father_rate →
  michael_depth_diff = 400 →
  (2 * (father_rate * father_time) - michael_depth_diff) / michael_rate = 700 :=
by sorry

end NUMINAMATH_CALUDE_michael_digging_time_l2661_266129


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l2661_266118

/-- Calculates the discount percentage on half of the bricks given the total number of bricks,
    full price per brick, and total amount spent. -/
theorem discount_percentage_calculation
  (total_bricks : ℕ)
  (full_price_per_brick : ℚ)
  (total_spent : ℚ)
  (h1 : total_bricks = 1000)
  (h2 : full_price_per_brick = 1/2)
  (h3 : total_spent = 375) :
  let half_bricks := total_bricks / 2
  let full_price_half := half_bricks * full_price_per_brick
  let discounted_price := total_spent - full_price_half
  let discount_amount := full_price_half - discounted_price
  let discount_percentage := (discount_amount / full_price_half) * 100
  discount_percentage = 50 := by sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l2661_266118


namespace NUMINAMATH_CALUDE_valid_numerical_pyramid_exists_l2661_266155

/-- Represents a row in the numerical pyramid --/
structure PyramidRow where
  digits : List ℕ
  result : ℕ

/-- Represents the entire numerical pyramid --/
structure NumericalPyramid where
  row1 : PyramidRow
  row2 : PyramidRow
  row3 : PyramidRow
  row4 : PyramidRow
  row5 : PyramidRow
  row6 : PyramidRow
  row7 : PyramidRow

/-- Function to check if a pyramid satisfies all conditions --/
def is_valid_pyramid (p : NumericalPyramid) : Prop :=
  p.row1.digits = [1, 2] ∧ p.row1.result = 3 ∧
  p.row2.digits = [1, 2, 3] ∧ p.row2.result = 4 ∧
  p.row3.digits = [1, 2, 3, 4] ∧ p.row3.result = 5 ∧
  p.row4.digits = [1, 2, 3, 4, 5] ∧ p.row4.result = 6 ∧
  p.row5.digits = [1, 2, 3, 4, 5, 6] ∧ p.row5.result = 7 ∧
  p.row6.digits = [1, 2, 3, 4, 5, 6, 7] ∧ p.row6.result = 8 ∧
  p.row7.digits = [1, 2, 3, 4, 5, 6, 7, 8] ∧ p.row7.result = 9

/-- Theorem stating that a valid numerical pyramid exists --/
theorem valid_numerical_pyramid_exists : ∃ (p : NumericalPyramid), is_valid_pyramid p := by
  sorry

end NUMINAMATH_CALUDE_valid_numerical_pyramid_exists_l2661_266155


namespace NUMINAMATH_CALUDE_product_of_solutions_l2661_266179

theorem product_of_solutions (x : ℝ) : 
  (|18 / x - 4| = 3) → (∃ y : ℝ, (|18 / y - 4| = 3) ∧ (x * y = 324 / 7)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2661_266179


namespace NUMINAMATH_CALUDE_prime_roots_sum_fraction_l2661_266154

theorem prime_roots_sum_fraction (p q m : ℕ) : 
  Prime p → Prime q → 
  p^2 - 99*p + m = 0 → 
  q^2 - 99*q + m = 0 → 
  (p : ℚ) / q + (q : ℚ) / p = 9413 / 194 := by
  sorry

end NUMINAMATH_CALUDE_prime_roots_sum_fraction_l2661_266154


namespace NUMINAMATH_CALUDE_min_side_difference_in_triangle_l2661_266116

theorem min_side_difference_in_triangle (xy xz yz : ℕ) : 
  xy + xz + yz = 3021 →
  xy < xz →
  xz < yz →
  2 ≤ yz - xy :=
by sorry

end NUMINAMATH_CALUDE_min_side_difference_in_triangle_l2661_266116


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2661_266128

theorem arithmetic_calculation : 5 * 12 + 6 * 11 - 2 * 15 + 7 * 9 = 159 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2661_266128


namespace NUMINAMATH_CALUDE_episode_length_proof_l2661_266172

/-- Represents the length of a single episode in minutes -/
def episode_length : ℕ := 33

/-- Represents the total number of episodes watched in a week -/
def total_episodes : ℕ := 8

/-- Represents the minutes watched on Monday -/
def monday_minutes : ℕ := 138

/-- Represents the minutes watched on Thursday -/
def thursday_minutes : ℕ := 21

/-- Represents the number of episodes watched on Friday -/
def friday_episodes : ℕ := 2

/-- Represents the minutes watched over the weekend -/
def weekend_minutes : ℕ := 105

/-- Proves that the given episode length satisfies the conditions of the problem -/
theorem episode_length_proof : 
  monday_minutes + thursday_minutes + weekend_minutes = total_episodes * episode_length := by
  sorry

end NUMINAMATH_CALUDE_episode_length_proof_l2661_266172


namespace NUMINAMATH_CALUDE_light_bulb_investigation_l2661_266157

/-- Represents the method of investigation -/
inductive InvestigationMethod
  | SamplingSurvey
  | Census

/-- Represents the characteristics of the investigation -/
structure InvestigationCharacteristics where
  largeQuantity : Bool
  destructiveTesting : Bool

/-- Determines the appropriate investigation method based on the characteristics -/
def appropriateMethod (chars : InvestigationCharacteristics) : InvestigationMethod :=
  if chars.largeQuantity && chars.destructiveTesting then
    InvestigationMethod.SamplingSurvey
  else
    InvestigationMethod.Census

/-- Theorem stating that for light bulb service life investigation with given characteristics, 
    sampling survey is the appropriate method -/
theorem light_bulb_investigation 
  (chars : InvestigationCharacteristics) 
  (h1 : chars.largeQuantity = true) 
  (h2 : chars.destructiveTesting = true) : 
  appropriateMethod chars = InvestigationMethod.SamplingSurvey := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_investigation_l2661_266157


namespace NUMINAMATH_CALUDE_princess_count_proof_l2661_266105

/-- Represents the number of princesses at the ball -/
def num_princesses : ℕ := 8

/-- Represents the number of knights at the ball -/
def num_knights : ℕ := 22 - num_princesses

/-- Represents the total number of people at the ball -/
def total_people : ℕ := 22

/-- Function to calculate the number of knights a princess dances with -/
def knights_danced_with (princess_index : ℕ) : ℕ := 6 + princess_index

theorem princess_count_proof :
  (num_princesses + num_knights = total_people) ∧ 
  (knights_danced_with num_princesses = num_knights) ∧
  (∀ i, i ≥ 1 → i ≤ num_princesses → knights_danced_with i ≤ num_knights) :=
sorry

end NUMINAMATH_CALUDE_princess_count_proof_l2661_266105


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_3_l2661_266177

theorem cos_2alpha_plus_4pi_3 (α : ℝ) (h : Real.sqrt 3 * Real.sin α * Real.cos α = 1 / 2) :
  Real.cos (2 * α + 4 * π / 3) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_3_l2661_266177


namespace NUMINAMATH_CALUDE_x_values_when_two_in_set_l2661_266186

theorem x_values_when_two_in_set (x : ℝ) : 2 ∈ ({1, x^2 + x} : Set ℝ) → x = 1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_values_when_two_in_set_l2661_266186


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2661_266181

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 8 = 0) → (x₂^2 - 2*x₂ - 8 = 0) → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2661_266181


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2661_266167

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^2 / (x^2 + y^2) ≤ 2 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a + b)^2 / (a^2 + b^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2661_266167


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l2661_266103

-- Equation 1
theorem solve_equation_one : 
  ∃ x : ℚ, (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 ∧ x = -20 := by sorry

-- Equation 2
theorem solve_equation_two : 
  ∃ x : ℚ, (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 3 ∧ x = 67 / 23 := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l2661_266103


namespace NUMINAMATH_CALUDE_prob_three_red_large_deck_l2661_266115

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (hsum : total_cards = red_cards + black_cards)

/-- Probability of drawing three red cards in a row -/
def prob_three_red (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards *
  ((d.red_cards - 1) : ℚ) / (d.total_cards - 1) *
  ((d.red_cards - 2) : ℚ) / (d.total_cards - 2)

/-- The main theorem -/
theorem prob_three_red_large_deck :
  let d : Deck := ⟨104, 52, 52, rfl⟩
  prob_three_red d = 425 / 3502 := by sorry

end NUMINAMATH_CALUDE_prob_three_red_large_deck_l2661_266115


namespace NUMINAMATH_CALUDE_sum_sqrt_squared_pairs_geq_sqrt2_l2661_266195

theorem sum_sqrt_squared_pairs_geq_sqrt2 (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) : 
  Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + a^2) ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_squared_pairs_geq_sqrt2_l2661_266195


namespace NUMINAMATH_CALUDE_customers_left_first_l2661_266127

/-- Proves the number of customers who left in the first group -/
theorem customers_left_first (initial : ℝ) (second_group : ℝ) (final : ℝ) :
  initial = 36.0 →
  second_group = 14.0 →
  final = 3 →
  initial - (initial - second_group - final) = 19.0 := by
  sorry

end NUMINAMATH_CALUDE_customers_left_first_l2661_266127


namespace NUMINAMATH_CALUDE_final_sparrow_count_l2661_266151

/-- Given the initial number of sparrows, the number of sparrows that joined,
    and the number of sparrows that flew away, prove that the final number
    of sparrows on the fence is 3. -/
theorem final_sparrow_count
  (initial_sparrows : ℕ)
  (joined_sparrows : ℕ)
  (flew_away_sparrows : ℕ)
  (h1 : initial_sparrows = 2)
  (h2 : joined_sparrows = 4)
  (h3 : flew_away_sparrows = 3) :
  initial_sparrows + joined_sparrows - flew_away_sparrows = 3 :=
by sorry

end NUMINAMATH_CALUDE_final_sparrow_count_l2661_266151


namespace NUMINAMATH_CALUDE_unique_initial_pair_l2661_266164

def arithmetic_mean_operation (a b : ℕ) : ℕ × ℕ :=
  if (a + b) % 2 = 0 then
    let mean := (a + b) / 2
    if mean < a then (mean, a) else (b, mean)
  else
    (a, b)

def perform_operations (n : ℕ) (pair : ℕ × ℕ) : ℕ × ℕ :=
  match n with
  | 0 => pair
  | n + 1 => perform_operations n (arithmetic_mean_operation pair.1 pair.2)

theorem unique_initial_pair :
  ∀ x : ℕ,
    x < 2015 →
    x ≠ 991 →
    ∃ i : ℕ,
      i ≤ 10 ∧
      (perform_operations i (x, 2015)).1 = (perform_operations i (x, 2015)).2 :=
sorry

end NUMINAMATH_CALUDE_unique_initial_pair_l2661_266164


namespace NUMINAMATH_CALUDE_probability_cousins_names_l2661_266153

/-- Represents the number of letters in each cousin's name -/
structure NameLengths where
  amelia : ℕ
  bethany : ℕ
  claire : ℕ

/-- The probability of selecting two cards from different cousins' names -/
def probability_different_names (nl : NameLengths) : ℚ :=
  let total := nl.amelia + nl.bethany + nl.claire
  2 * (nl.amelia * nl.bethany + nl.amelia * nl.claire + nl.bethany * nl.claire) / (total * (total - 1))

/-- Theorem stating the probability of selecting two cards from different cousins' names -/
theorem probability_cousins_names :
  let nl : NameLengths := { amelia := 6, bethany := 7, claire := 6 }
  probability_different_names nl = 40 / 57 := by
  sorry


end NUMINAMATH_CALUDE_probability_cousins_names_l2661_266153


namespace NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l2661_266133

theorem units_digit_of_n_squared_plus_two_to_n (n : ℕ) : 
  n = 1234^2 + 2^1234 → (n^2 + 2^n) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l2661_266133


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2661_266185

theorem arithmetic_sequence_sum (n : ℕ) : 
  (Finset.range (n + 3)).sum (fun i => 2 * i + 3) = n^2 + 8*n + 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2661_266185


namespace NUMINAMATH_CALUDE_system_of_equations_solution_range_l2661_266178

theorem system_of_equations_solution_range (a x y : ℝ) : 
  x + 3*y = 3 - a →
  2*x + y = 1 + 3*a →
  x + y > 3*a + 4 →
  a < -3/2 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_range_l2661_266178


namespace NUMINAMATH_CALUDE_walkway_area_is_296_l2661_266152

-- Define the garden layout
def num_rows : ℕ := 4
def num_cols : ℕ := 3
def bed_width : ℕ := 4
def bed_height : ℕ := 3
def walkway_width : ℕ := 2

-- Define the total garden dimensions
def garden_width : ℕ := num_cols * bed_width + (num_cols + 1) * walkway_width
def garden_height : ℕ := num_rows * bed_height + (num_rows + 1) * walkway_width

-- Define the total garden area
def total_garden_area : ℕ := garden_width * garden_height

-- Define the total flower bed area
def total_bed_area : ℕ := num_rows * num_cols * bed_width * bed_height

-- Theorem to prove
theorem walkway_area_is_296 : total_garden_area - total_bed_area = 296 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_296_l2661_266152


namespace NUMINAMATH_CALUDE_expected_lotus_seed_zongzi_l2661_266125

theorem expected_lotus_seed_zongzi 
  (total_zongzi : ℕ) 
  (lotus_seed_zongzi : ℕ) 
  (selected_zongzi : ℕ) 
  (h1 : total_zongzi = 180) 
  (h2 : lotus_seed_zongzi = 54) 
  (h3 : selected_zongzi = 10) :
  (selected_zongzi : ℚ) * (lotus_seed_zongzi : ℚ) / (total_zongzi : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_lotus_seed_zongzi_l2661_266125


namespace NUMINAMATH_CALUDE_biker_passes_l2661_266147

/-- Represents a biker's total travels along the road -/
structure BikerTravel where
  travels : ℕ

/-- Represents the scenario of two bikers on a road -/
structure BikerScenario where
  biker1 : BikerTravel
  biker2 : BikerTravel

/-- Calculates the number of passes between two bikers -/
def calculatePasses (scenario : BikerScenario) : ℕ :=
  sorry

theorem biker_passes (scenario : BikerScenario) :
  scenario.biker1.travels = 11 →
  scenario.biker2.travels = 7 →
  calculatePasses scenario = 8 :=
sorry

end NUMINAMATH_CALUDE_biker_passes_l2661_266147


namespace NUMINAMATH_CALUDE_bus_journey_speed_l2661_266136

theorem bus_journey_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_part_distance : ℝ) 
  (second_part_speed : ℝ) 
  (h1 : total_distance = 250) 
  (h2 : total_time = 5.2) 
  (h3 : first_part_distance = 124) 
  (h4 : second_part_speed = 60) :
  ∃ (first_part_speed : ℝ), 
    first_part_speed = 40 ∧ 
    first_part_distance / first_part_speed + 
    (total_distance - first_part_distance) / second_part_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_bus_journey_speed_l2661_266136


namespace NUMINAMATH_CALUDE_fir_tree_needles_l2661_266191

/-- Represents the number of fir trees in the forest -/
def num_trees : ℕ := 710000

/-- Represents the maximum number of needles a tree can have -/
def max_needles : ℕ := 100000

/-- Represents the minimum number of trees we want to prove have the same number of needles -/
def min_same_needles : ℕ := 7

theorem fir_tree_needles :
  ∃ (n : ℕ) (trees : Finset (Fin num_trees)),
    n ≤ max_needles ∧
    trees.card ≥ min_same_needles ∧
    ∀ t ∈ trees, (fun i => i.val) t = n :=
by sorry

end NUMINAMATH_CALUDE_fir_tree_needles_l2661_266191


namespace NUMINAMATH_CALUDE_six_digit_number_concatenation_divisibility_l2661_266142

theorem six_digit_number_concatenation_divisibility : 
  let a : ℕ := 166667
  let b : ℕ := 333334
  -- a and b are six-digit numbers
  (100000 ≤ a ∧ a < 1000000) ∧
  (100000 ≤ b ∧ b < 1000000) ∧
  -- The concatenated number is divisible by the product
  (1000000 * a + b) % (a * b) = 0 := by
sorry

end NUMINAMATH_CALUDE_six_digit_number_concatenation_divisibility_l2661_266142


namespace NUMINAMATH_CALUDE_second_question_probability_l2661_266100

theorem second_question_probability 
  (p_first : ℝ) 
  (p_neither : ℝ) 
  (p_both : ℝ) 
  (h1 : p_first = 0.65)
  (h2 : p_neither = 0.20)
  (h3 : p_both = 0.40)
  : ∃ p_second : ℝ, p_second = 0.75 ∧ 
    p_first + p_second - p_both + p_neither = 1 :=
sorry

end NUMINAMATH_CALUDE_second_question_probability_l2661_266100


namespace NUMINAMATH_CALUDE_average_tv_watching_l2661_266165

def tv_hours : List ℝ := [10, 8, 12]

theorem average_tv_watching :
  (tv_hours.sum / tv_hours.length : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_tv_watching_l2661_266165


namespace NUMINAMATH_CALUDE_f_13_equals_223_l2661_266183

/-- Define the function f for natural numbers -/
def f (n : ℕ) : ℕ := n^2 + n + 41

/-- Theorem stating that f(13) equals 223 -/
theorem f_13_equals_223 : f 13 = 223 := by
  sorry

end NUMINAMATH_CALUDE_f_13_equals_223_l2661_266183


namespace NUMINAMATH_CALUDE_distance_opposite_points_l2661_266171

-- Define a point in polar coordinates
structure PolarPoint where
  r : ℝ
  θ : ℝ

-- Define the distance function between two polar points
def polarDistance (A B : PolarPoint) : ℝ :=
  sorry

-- Theorem statement
theorem distance_opposite_points (A B : PolarPoint) 
    (h : abs (B.θ - A.θ) = Real.pi) : 
  polarDistance A B = A.r + B.r := by
  sorry

end NUMINAMATH_CALUDE_distance_opposite_points_l2661_266171


namespace NUMINAMATH_CALUDE_henry_jeans_cost_l2661_266106

/-- The cost of Henry's clothing items -/
structure ClothingCosts where
  socks : ℝ
  tshirt : ℝ
  jeans : ℝ

/-- The conditions for Henry's clothing costs -/
def henry_clothing_conditions (c : ClothingCosts) : Prop :=
  c.jeans = 2 * c.tshirt ∧
  c.tshirt = c.socks + 10 ∧
  c.socks = 5

/-- Theorem: Given the conditions, the cost of Henry's jeans is $30 -/
theorem henry_jeans_cost (c : ClothingCosts) 
  (h : henry_clothing_conditions c) : c.jeans = 30 := by
  sorry

end NUMINAMATH_CALUDE_henry_jeans_cost_l2661_266106


namespace NUMINAMATH_CALUDE_sara_birds_count_l2661_266126

-- Define a dozen as 12
def dozen : ℕ := 12

-- Define the number of dozens Sara saw
def sara_dozens : ℕ := 8

-- Theorem: If Sara saw 8 dozen birds, then she saw 96 birds in total
theorem sara_birds_count : sara_dozens * dozen = 96 := by
  sorry

end NUMINAMATH_CALUDE_sara_birds_count_l2661_266126


namespace NUMINAMATH_CALUDE_adrianna_gum_l2661_266110

def gum_problem (initial_gum : ℕ) (bought_gum : ℕ) (friends : ℕ) : Prop :=
  let total_gum := initial_gum + bought_gum
  let remaining_gum := total_gum - friends
  remaining_gum = 2

theorem adrianna_gum : gum_problem 10 3 11 := by
  sorry

end NUMINAMATH_CALUDE_adrianna_gum_l2661_266110


namespace NUMINAMATH_CALUDE_apple_cost_price_l2661_266158

/-- The cost price of an apple given its selling price and loss ratio. -/
def cost_price (selling_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  selling_price / (1 - loss_ratio)

/-- Theorem stating the cost price of an apple given specific conditions. -/
theorem apple_cost_price :
  let selling_price : ℚ := 17
  let loss_ratio : ℚ := 1/6
  cost_price selling_price loss_ratio = 20.4 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_price_l2661_266158


namespace NUMINAMATH_CALUDE_smallest_positive_integer_modulo_l2661_266162

theorem smallest_positive_integer_modulo (y : ℕ) : y = 14 ↔ 
  (y > 0 ∧ (y + 3050) % 15 = 1234 % 15 ∧ ∀ z : ℕ, z > 0 → (z + 3050) % 15 = 1234 % 15 → y ≤ z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_modulo_l2661_266162


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2661_266114

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, -2) and (2, 10) is 9. -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 8
  let y1 : ℝ := -2
  let x2 : ℝ := 2
  let y2 : ℝ := 10
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 9 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2661_266114


namespace NUMINAMATH_CALUDE_product_positive_l2661_266149

theorem product_positive (a b : ℝ) (h1 : a > b) (h2 : b > 0) : b * (a - b) > 0 := by
  sorry

end NUMINAMATH_CALUDE_product_positive_l2661_266149


namespace NUMINAMATH_CALUDE_xy_value_l2661_266196

theorem xy_value (x y : ℝ) (h : |x - 1| + (x + y)^2 = 0) : x * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2661_266196


namespace NUMINAMATH_CALUDE_red_peaches_count_l2661_266168

theorem red_peaches_count (total : ℕ) (yellow : ℕ) (green : ℕ) (red : ℕ) 
  (h1 : total = 30)
  (h2 : yellow = 15)
  (h3 : green = 8)
  (h4 : total = red + yellow + green) : 
  red = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l2661_266168


namespace NUMINAMATH_CALUDE_simplify_fourth_root_l2661_266139

theorem simplify_fourth_root : 
  (2^5 * 5^3 : ℝ)^(1/4) = 2 * (250 : ℝ)^(1/4) := by sorry

end NUMINAMATH_CALUDE_simplify_fourth_root_l2661_266139


namespace NUMINAMATH_CALUDE_airplane_seats_l2661_266174

theorem airplane_seats : ∃ (total : ℝ), 
  (30 : ℝ) + 0.2 * total + 0.75 * total = total ∧ total = 600 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l2661_266174


namespace NUMINAMATH_CALUDE_tyson_jenna_meeting_l2661_266160

/-- The distance between points A and B in miles -/
def total_distance : ℝ := 80

/-- Jenna's head start in hours -/
def jenna_head_start : ℝ := 1.5

/-- Jenna's walking speed in miles per hour -/
def jenna_speed : ℝ := 3.5

/-- Tyson's walking speed in miles per hour -/
def tyson_speed : ℝ := 2.8

/-- The distance Tyson walked when he met Jenna -/
def tyson_distance : ℝ := 33.25

theorem tyson_jenna_meeting :
  ∃ t : ℝ, t > 0 ∧
  jenna_speed * (t + jenna_head_start) + tyson_speed * t = total_distance ∧
  tyson_speed * t = tyson_distance :=
sorry

end NUMINAMATH_CALUDE_tyson_jenna_meeting_l2661_266160


namespace NUMINAMATH_CALUDE_green_ball_probability_l2661_266198

structure Container where
  red : ℕ
  green : ℕ

def Set1 : List Container := [
  ⟨2, 8⟩,  -- Container A
  ⟨8, 2⟩,  -- Container B
  ⟨8, 2⟩   -- Container C
]

def Set2 : List Container := [
  ⟨8, 2⟩,  -- Container A
  ⟨2, 8⟩,  -- Container B
  ⟨2, 8⟩   -- Container C
]

def probability_green (set : List Container) : ℚ :=
  let total_balls (c : Container) := c.red + c.green
  let green_prob (c : Container) := c.green / (total_balls c)
  (set.map green_prob).sum / set.length

theorem green_ball_probability :
  (1 / 2 : ℚ) * probability_green Set1 + (1 / 2 : ℚ) * probability_green Set2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2661_266198


namespace NUMINAMATH_CALUDE_apple_probabilities_l2661_266135

structure ApplePlot where
  name : String
  first_grade_ratio : ℚ
  production_ratio : ℕ

def plot_a : ApplePlot := ⟨"A", 3/4, 2⟩
def plot_b : ApplePlot := ⟨"B", 3/5, 5⟩
def plot_c : ApplePlot := ⟨"C", 4/5, 3⟩

def total_production : ℕ := plot_a.production_ratio + plot_b.production_ratio + plot_c.production_ratio

theorem apple_probabilities :
  (plot_a.production_ratio : ℚ) / total_production = 1/5 ∧
  (plot_a.production_ratio * plot_a.first_grade_ratio +
   plot_b.production_ratio * plot_b.first_grade_ratio +
   plot_c.production_ratio * plot_c.first_grade_ratio) / total_production = 69/100 ∧
  (plot_a.production_ratio * plot_a.first_grade_ratio) /
  (plot_a.production_ratio * plot_a.first_grade_ratio +
   plot_b.production_ratio * plot_b.first_grade_ratio +
   plot_c.production_ratio * plot_c.first_grade_ratio) = 5/23 := by
  sorry


end NUMINAMATH_CALUDE_apple_probabilities_l2661_266135


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2661_266137

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  x : ℚ
  first_term : ℚ := 3 * x - 4
  second_term : ℚ := 6 * x - 14
  third_term : ℚ := 4 * x + 3
  is_arithmetic : second_term - first_term = third_term - second_term

/-- The nth term of the sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.first_term + (n - 1) * (seq.second_term - seq.first_term)

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  ∃ n : ℕ, nth_term seq n = 3012 ∧ n = 247 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2661_266137


namespace NUMINAMATH_CALUDE_gcd_square_le_sum_l2661_266156

theorem gcd_square_le_sum (a b : ℕ) (h1 : (a + 1) % b = 0) (h2 : (b + 1) % a = 0) : 
  (Nat.gcd a b)^2 ≤ a + b := by
  sorry

end NUMINAMATH_CALUDE_gcd_square_le_sum_l2661_266156


namespace NUMINAMATH_CALUDE_floor_sqrt_equality_l2661_266131

theorem floor_sqrt_equality (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 1)⌋ ∧
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_equality_l2661_266131


namespace NUMINAMATH_CALUDE_neighborhood_b_cookie_boxes_l2661_266107

/-- 
Proves that each home in Neighborhood B buys 5 boxes of cookies given the conditions of the problem.
-/
theorem neighborhood_b_cookie_boxes : 
  let neighborhood_a_homes : ℕ := 10
  let neighborhood_a_boxes_per_home : ℕ := 2
  let neighborhood_b_homes : ℕ := 5
  let price_per_box : ℕ := 2
  let better_neighborhood_revenue : ℕ := 50
  
  neighborhood_b_homes > 0 →
  (neighborhood_a_homes * neighborhood_a_boxes_per_home * price_per_box < better_neighborhood_revenue) →
  
  ∃ (boxes_per_home_b : ℕ),
    boxes_per_home_b * neighborhood_b_homes * price_per_box = better_neighborhood_revenue ∧
    boxes_per_home_b = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_neighborhood_b_cookie_boxes_l2661_266107


namespace NUMINAMATH_CALUDE_kenny_book_purchase_l2661_266188

/-- Calculates the number of books Kenny can buy after mowing lawns and purchasing video games -/
def books_kenny_can_buy (lawn_price : ℕ) (video_game_price : ℕ) (book_price : ℕ) 
                        (lawns_mowed : ℕ) (video_games_to_buy : ℕ) : ℕ :=
  let total_earnings := lawn_price * lawns_mowed
  let video_games_cost := video_game_price * video_games_to_buy
  let remaining_money := total_earnings - video_games_cost
  remaining_money / book_price

/-- Theorem stating that Kenny can buy 60 books given the problem conditions -/
theorem kenny_book_purchase :
  books_kenny_can_buy 15 45 5 35 5 = 60 := by
  sorry

#eval books_kenny_can_buy 15 45 5 35 5

end NUMINAMATH_CALUDE_kenny_book_purchase_l2661_266188


namespace NUMINAMATH_CALUDE_probability_same_color_l2661_266130

/-- The number of marbles of each color in the box -/
def marbles_per_color : ℕ := 3

/-- The total number of colors -/
def num_colors : ℕ := 3

/-- The total number of marbles in the box -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- The number of marbles drawn -/
def drawn_marbles : ℕ := 3

/-- The probability of drawing 3 marbles of the same color -/
theorem probability_same_color :
  (num_colors * (Nat.choose marbles_per_color drawn_marbles)) /
  (Nat.choose total_marbles drawn_marbles) = 1 / 28 :=
sorry

end NUMINAMATH_CALUDE_probability_same_color_l2661_266130


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_one_l2661_266108

theorem sum_of_roots_equals_one :
  ∀ x₁ x₂ : ℝ, (x₁ + 3) * (x₁ - 4) = 22 ∧ (x₂ + 3) * (x₂ - 4) = 22 → x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_one_l2661_266108


namespace NUMINAMATH_CALUDE_max_hubs_is_six_l2661_266111

/-- A structure representing a state with cities and roads --/
structure State where
  num_cities : ℕ
  num_roads : ℕ
  num_hubs : ℕ

/-- Definition of a valid state configuration --/
def is_valid_state (s : State) : Prop :=
  s.num_cities = 10 ∧
  s.num_roads = 40 ∧
  s.num_hubs ≤ s.num_cities ∧
  s.num_hubs * (s.num_hubs - 1) / 2 + s.num_hubs * (s.num_cities - s.num_hubs) ≤ s.num_roads

/-- Theorem stating that the maximum number of hubs in a valid state is 6 --/
theorem max_hubs_is_six :
  ∀ s : State, is_valid_state s → s.num_hubs ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_hubs_is_six_l2661_266111


namespace NUMINAMATH_CALUDE_unique_solution_x_y_l2661_266190

theorem unique_solution_x_y (x y : ℕ) (h1 : y ∣ (x^2 + 1)) (h2 : x^2 ∣ (y^3 + 1)) : x = 1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_x_y_l2661_266190


namespace NUMINAMATH_CALUDE_min_value_of_f_l2661_266144

def f (x : ℕ) : ℤ := 3 * x^2 - 12 * x + 800

theorem min_value_of_f :
  ∀ x : ℕ, f x ≥ 788 ∧ ∃ x₀ : ℕ, f x₀ = 788 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2661_266144
