import Mathlib

namespace NUMINAMATH_CALUDE_sara_balloons_count_l1258_125834

/-- The number of yellow balloons that Tom has -/
def tom_balloons : ℕ := 9

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 17

/-- The number of yellow balloons that Sara has -/
def sara_balloons : ℕ := total_balloons - tom_balloons

theorem sara_balloons_count : sara_balloons = 8 := by
  sorry

end NUMINAMATH_CALUDE_sara_balloons_count_l1258_125834


namespace NUMINAMATH_CALUDE_girls_average_age_l1258_125802

/-- Proves that the average age of girls is 11 years given the school statistics --/
theorem girls_average_age (total_students : ℕ) (boys_avg_age : ℚ) (school_avg_age : ℚ) (num_girls : ℕ) :
  total_students = 600 →
  boys_avg_age = 12 →
  school_avg_age = 47 / 4 →  -- 11.75 years
  num_girls = 150 →
  let num_boys : ℕ := total_students - num_girls
  let total_age : ℚ := total_students * school_avg_age
  let boys_total_age : ℚ := num_boys * boys_avg_age
  let girls_total_age : ℚ := total_age - boys_total_age
  girls_total_age / num_girls = 11 := by
sorry


end NUMINAMATH_CALUDE_girls_average_age_l1258_125802


namespace NUMINAMATH_CALUDE_close_interval_for_m_and_n_l1258_125815

-- Define the functions m and n
def m (x : ℝ) := x^2 - 3*x + 4
def n (x : ℝ) := 2*x - 3

-- Define what it means for two functions to be close on an interval
def are_close (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, |f x - g x| ≤ 1

-- Theorem statement
theorem close_interval_for_m_and_n :
  are_close m n 2 3 :=
sorry

end NUMINAMATH_CALUDE_close_interval_for_m_and_n_l1258_125815


namespace NUMINAMATH_CALUDE_unique_integral_solution_l1258_125870

theorem unique_integral_solution :
  ∃! (x y z : ℕ), 
    (z^x = y^(3*x)) ∧ 
    (2^z = 4 * 8^x) ∧ 
    (x + y + z = 20) ∧
    x = 2 ∧ y = 2 ∧ z = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_integral_solution_l1258_125870


namespace NUMINAMATH_CALUDE_lines_intersection_l1258_125894

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℚ) : Prop :=
  l.a * x + l.b * y = l.c

/-- The three lines given in the problem -/
def line1 : Line := ⟨-3, 2, 4⟩
def line2 : Line := ⟨1, 3, 3⟩
def line3 : Line := ⟨5, -3, 6⟩

/-- Theorem stating that the given lines intersect at the specified points -/
theorem lines_intersection :
  (line1.contains (10/11) (13/11) ∧
   line2.contains (10/11) (13/11) ∧
   line3.contains 24 38) ∧
  (line1.contains 24 38 ∧
   line2.contains 24 38 ∧
   line3.contains 24 38) := by
  sorry

end NUMINAMATH_CALUDE_lines_intersection_l1258_125894


namespace NUMINAMATH_CALUDE_no_prime_sum_56_l1258_125825

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the property we want to prove
theorem no_prime_sum_56 : ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 56 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_56_l1258_125825


namespace NUMINAMATH_CALUDE_auction_bid_ratio_l1258_125810

/-- Auction bidding problem --/
theorem auction_bid_ratio :
  let start_price : ℕ := 300
  let harry_first_bid : ℕ := start_price + 200
  let second_bid : ℕ := 2 * harry_first_bid
  let harry_final_bid : ℕ := 4000
  let third_bid : ℕ := harry_final_bid - 1500
  (third_bid : ℚ) / harry_first_bid = 5 := by sorry

end NUMINAMATH_CALUDE_auction_bid_ratio_l1258_125810


namespace NUMINAMATH_CALUDE_efgh_is_parallelogram_l1258_125855

-- Define the types for points and quadrilaterals
variable (Point : Type) (Quadrilateral : Type)

-- Define the property of being a convex quadrilateral
variable (is_convex_quadrilateral : Quadrilateral → Prop)

-- Define the property of forming an equilateral triangle
variable (forms_equilateral_triangle : Point → Point → Point → Prop)

-- Define the property of a triangle being directed outward or inward
variable (is_outward : Point → Point → Point → Quadrilateral → Prop)
variable (is_inward : Point → Point → Point → Quadrilateral → Prop)

-- Define the property of being a parallelogram
variable (is_parallelogram : Point → Point → Point → Point → Prop)

-- Theorem statement
theorem efgh_is_parallelogram 
  (A B C D E F G H : Point) (Q : Quadrilateral) :
  is_convex_quadrilateral Q →
  forms_equilateral_triangle A B E →
  forms_equilateral_triangle B C F →
  forms_equilateral_triangle C D G →
  forms_equilateral_triangle D A H →
  is_outward A B E Q →
  is_outward C D G Q →
  is_inward B C F Q →
  is_inward D A H Q →
  is_parallelogram E F G H :=
by sorry

end NUMINAMATH_CALUDE_efgh_is_parallelogram_l1258_125855


namespace NUMINAMATH_CALUDE_linear_function_point_range_l1258_125878

theorem linear_function_point_range (x y : ℝ) : 
  y = 4 - 3 * x → y > -5 → x < 3 := by sorry

end NUMINAMATH_CALUDE_linear_function_point_range_l1258_125878


namespace NUMINAMATH_CALUDE_max_value_d_l1258_125829

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 17) :
  d ≤ (5 + Real.sqrt 123) / 2 ∧ 
  ∃ (a' b' c' : ℝ), a' + b' + c' + (5 + Real.sqrt 123) / 2 = 10 ∧ 
    a'*b' + a'*c' + a'*((5 + Real.sqrt 123) / 2) + b'*c' + 
    b'*((5 + Real.sqrt 123) / 2) + c'*((5 + Real.sqrt 123) / 2) = 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_d_l1258_125829


namespace NUMINAMATH_CALUDE_cost_per_tire_to_produce_l1258_125879

/-- Proves that the cost per tire to produce is $8 given the specified conditions --/
theorem cost_per_tire_to_produce
  (fixed_cost : ℝ)
  (selling_price : ℝ)
  (batch_size : ℝ)
  (profit_per_tire : ℝ)
  (h1 : fixed_cost = 22500)
  (h2 : selling_price = 20)
  (h3 : batch_size = 15000)
  (h4 : profit_per_tire = 10.5) :
  ∃ (cost_per_tire : ℝ),
    cost_per_tire = 8 ∧
    batch_size * (selling_price - cost_per_tire) - fixed_cost = batch_size * profit_per_tire :=
by sorry

end NUMINAMATH_CALUDE_cost_per_tire_to_produce_l1258_125879


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1258_125841

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : isArithmeticSequence a)
  (h_sum : a 6 + a 9 = 16)
  (h_a4 : a 4 = 1) :
  a 11 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1258_125841


namespace NUMINAMATH_CALUDE_ellipse_chord_properties_l1258_125892

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse mx² + ny² = 1 -/
structure Ellipse where
  m : ℝ
  n : ℝ
  h_positive : m > 0 ∧ n > 0
  h_distinct : m ≠ n

/-- Theorem about properties of chords in an ellipse -/
theorem ellipse_chord_properties (e : Ellipse) (a b c d : Point) (e_mid f_mid : Point) : 
  -- AB is a chord with slope 1
  (b.y - a.y) / (b.x - a.x) = 1 →
  -- CD is perpendicular to AB
  (d.y - c.y) / (d.x - c.x) = -1 →
  -- E is midpoint of AB
  e_mid.x = (a.x + b.x) / 2 ∧ e_mid.y = (a.y + b.y) / 2 →
  -- F is midpoint of CD
  f_mid.x = (c.x + d.x) / 2 ∧ f_mid.y = (c.y + d.y) / 2 →
  -- A, B, C, D are on the ellipse
  e.m * a.x^2 + e.n * a.y^2 = 1 ∧
  e.m * b.x^2 + e.n * b.y^2 = 1 ∧
  e.m * c.x^2 + e.n * c.y^2 = 1 ∧
  e.m * d.x^2 + e.n * d.y^2 = 1 →
  -- Conclusion 1: |CD|² - |AB|² = 4|EF|²
  ((c.x - d.x)^2 + (c.y - d.y)^2) - ((a.x - b.x)^2 + (a.y - b.y)^2) = 
    4 * ((e_mid.x - f_mid.x)^2 + (e_mid.y - f_mid.y)^2) ∧
  -- Conclusion 2: A, B, C, D are concyclic
  ∃ (center : Point) (r : ℝ),
    (a.x - center.x)^2 + (a.y - center.y)^2 = r^2 ∧
    (b.x - center.x)^2 + (b.y - center.y)^2 = r^2 ∧
    (c.x - center.x)^2 + (c.y - center.y)^2 = r^2 ∧
    (d.x - center.x)^2 + (d.y - center.y)^2 = r^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_properties_l1258_125892


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l1258_125869

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, prove that when 6S = a²sin A + b²sin B and (a+b)/c is maximized,
    cos C = 7/9 -/
theorem triangle_cosine_theorem (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h3 : A + B + C = π)
  (h4 : S = (1/2) * a * b * Real.sin C)
  (h5 : 6 * S = a^2 * Real.sin A + b^2 * Real.sin B)
  (h6 : ∀ (x y z : ℝ), (x + y) / z ≤ (a + b) / c) :
  Real.cos C = 7/9 :=
sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l1258_125869


namespace NUMINAMATH_CALUDE_bob_water_percentage_l1258_125876

def corn_water_usage : ℝ := 20
def cotton_water_usage : ℝ := 80
def bean_water_usage : ℝ := 2 * corn_water_usage

def bob_corn_acres : ℝ := 3
def bob_cotton_acres : ℝ := 9
def bob_bean_acres : ℝ := 12

def brenda_corn_acres : ℝ := 6
def brenda_cotton_acres : ℝ := 7
def brenda_bean_acres : ℝ := 14

def bernie_corn_acres : ℝ := 2
def bernie_cotton_acres : ℝ := 12

def bob_water_usage : ℝ := 
  bob_corn_acres * corn_water_usage + 
  bob_cotton_acres * cotton_water_usage + 
  bob_bean_acres * bean_water_usage

def total_water_usage : ℝ := 
  (bob_corn_acres + brenda_corn_acres + bernie_corn_acres) * corn_water_usage +
  (bob_cotton_acres + brenda_cotton_acres + bernie_cotton_acres) * cotton_water_usage +
  (bob_bean_acres + brenda_bean_acres) * bean_water_usage

theorem bob_water_percentage : 
  bob_water_usage / total_water_usage * 100 = 36 := by sorry

end NUMINAMATH_CALUDE_bob_water_percentage_l1258_125876


namespace NUMINAMATH_CALUDE_base_16_to_binary_bits_l1258_125865

/-- The base-16 number represented as a natural number -/
def base_16_number : ℕ := 8 * 16^4 + 8 * 16^3 + 8 * 16^2 + 8 * 16^1 + 8 * 16^0

/-- The number of bits required to represent a natural number -/
def num_bits (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

/-- Theorem stating that the number of bits required to represent base_16_number in base 2 is 20 -/
theorem base_16_to_binary_bits :
  num_bits base_16_number = 20 := by
  sorry

end NUMINAMATH_CALUDE_base_16_to_binary_bits_l1258_125865


namespace NUMINAMATH_CALUDE_fifteen_switch_network_connections_l1258_125842

/-- Represents a network of switches -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- Calculates the total number of connections in the network -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- Theorem: In a network of 15 switches, where each switch is connected to 4 others,
    the total number of connections is 30 -/
theorem fifteen_switch_network_connections :
  let network : SwitchNetwork := ⟨15, 4⟩
  total_connections network = 30 := by
  sorry


end NUMINAMATH_CALUDE_fifteen_switch_network_connections_l1258_125842


namespace NUMINAMATH_CALUDE_binomial_square_condition_l1258_125844

/-- If ax^2 + 24x + 9 is the square of a binomial, then a = 16 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 24 * x + 9 = (r * x + s)^2) → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l1258_125844


namespace NUMINAMATH_CALUDE_remainder_problem_l1258_125848

theorem remainder_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) (hmn : m > n) 
  (hm_mod : m % 6 = 2) (hdiff_mod : (m - n) % 6 = 5) : n % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1258_125848


namespace NUMINAMATH_CALUDE_increasing_magnitude_l1258_125843

-- Define the variables and conditions
theorem increasing_magnitude (a : ℝ) 
  (h1 : 0.8 < a) (h2 : a < 0.9)
  (y : ℝ) (hy : y = a^a)
  (z : ℝ) (hz : z = a^(a^a))
  (w : ℝ) (hw : w = a^(Real.log a)) :
  a < z ∧ z < y ∧ y < w := by sorry

end NUMINAMATH_CALUDE_increasing_magnitude_l1258_125843


namespace NUMINAMATH_CALUDE_condition_2_is_sufficient_for_condition_1_l1258_125850

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationship between conditions
def condition_relationship (A B C D : Prop) : Prop :=
  (C < D → A > B)

-- Define sufficient condition
def is_sufficient_condition (P Q : Prop) : Prop :=
  P → Q

-- Theorem statement
theorem condition_2_is_sufficient_for_condition_1 
  (h : condition_relationship A B C D) :
  is_sufficient_condition (C < D) (A > B) :=
sorry

end NUMINAMATH_CALUDE_condition_2_is_sufficient_for_condition_1_l1258_125850


namespace NUMINAMATH_CALUDE_min_value_cos_squared_plus_sin_l1258_125868

theorem min_value_cos_squared_plus_sin (f : ℝ → ℝ) :
  (∀ x, -π/4 ≤ x ∧ x ≤ π/4 → f x = Real.cos x ^ 2 + Real.sin x) →
  ∃ x₀, -π/4 ≤ x₀ ∧ x₀ ≤ π/4 ∧ f x₀ = (1 - Real.sqrt 2) / 2 ∧
  ∀ x, -π/4 ≤ x ∧ x ≤ π/4 → f x₀ ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_squared_plus_sin_l1258_125868


namespace NUMINAMATH_CALUDE_mean_temperature_is_84_l1258_125837

def temperatures : List ℚ := [80, 79, 81, 85, 87, 89, 87]

theorem mean_temperature_is_84 :
  (temperatures.sum / temperatures.length : ℚ) = 84 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_84_l1258_125837


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1258_125800

/-- A quadratic function with the given properties -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The symmetry property of f -/
def symmetry_property (b c : ℝ) : Prop :=
  ∀ x, f b c (2 + x) = f b c (2 - x)

theorem quadratic_function_properties (b c : ℝ) 
  (h : symmetry_property b c) : 
  b = 4 ∧ 
  (∀ a : ℝ, f b c (5/4) ≥ f b c (-a^2 - a + 1)) ∧
  (∀ a : ℝ, f b c (5/4) = f b c (-a^2 - a + 1) ↔ a = -1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1258_125800


namespace NUMINAMATH_CALUDE_total_bills_calculation_l1258_125836

def withdrawal1 : ℕ := 450
def withdrawal2 : ℕ := 750
def bill_value : ℕ := 20

theorem total_bills_calculation : 
  (withdrawal1 + withdrawal2) / bill_value = 60 := by sorry

end NUMINAMATH_CALUDE_total_bills_calculation_l1258_125836


namespace NUMINAMATH_CALUDE_smallest_with_18_divisors_l1258_125821

/-- The number of positive divisors of a positive integer -/
def numDivisors (n : ℕ+) : ℕ := sorry

/-- Returns true if n is the smallest positive integer with exactly k positive divisors -/
def isSmallestWithDivisors (n k : ℕ+) : Prop :=
  numDivisors n = k ∧ ∀ m : ℕ+, m < n → numDivisors m ≠ k

theorem smallest_with_18_divisors :
  isSmallestWithDivisors 288 18 := by sorry

end NUMINAMATH_CALUDE_smallest_with_18_divisors_l1258_125821


namespace NUMINAMATH_CALUDE_exists_four_digit_divisible_by_23_digit_sum_23_l1258_125880

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Proposition: There exists a four-digit number divisible by 23 with digit sum 23 -/
theorem exists_four_digit_divisible_by_23_digit_sum_23 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ digit_sum n = 23 ∧ n % 23 = 0 := by
sorry

end NUMINAMATH_CALUDE_exists_four_digit_divisible_by_23_digit_sum_23_l1258_125880


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_perpendicular_to_two_planes_are_parallel_l1258_125875

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)

-- Statement ②
theorem line_parallel_to_plane 
  (l : Line) (α : Plane) :
  parallel_plane l α → 
  ∃ (S : Set Line), (∀ m ∈ S, in_plane m α ∧ parallel l m) ∧ Set.Infinite S :=
sorry

-- Statement ④
theorem perpendicular_to_two_planes_are_parallel 
  (m : Line) (α β : Plane) :
  perpendicular_plane m α → perpendicular_plane m β → parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_perpendicular_to_two_planes_are_parallel_l1258_125875


namespace NUMINAMATH_CALUDE_complex_cube_root_problem_l1258_125835

theorem complex_cube_root_problem : ∃ (c : ℤ), (1 + 3*I : ℂ)^3 = -26 + c*I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_problem_l1258_125835


namespace NUMINAMATH_CALUDE_simplify_expression_l1258_125840

theorem simplify_expression (y : ℝ) : 3 * y + 4.5 * y + 7 * y = 14.5 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1258_125840


namespace NUMINAMATH_CALUDE_new_person_weight_weight_calculation_l1258_125827

theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) : ℝ :=
  let total_increase := n * avg_increase
  let new_weight := old_weight + total_increase
  new_weight

theorem weight_calculation :
  new_person_weight 8 3 65 = 89 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_weight_calculation_l1258_125827


namespace NUMINAMATH_CALUDE_ticket_sales_l1258_125838

theorem ticket_sales (total : ℕ) (reduced_first_week : ℕ) (full_price : ℕ) :
  total = 25200 →
  reduced_first_week = 5400 →
  full_price = 5 * reduced_first_week →
  total = reduced_first_week + full_price →
  full_price = 27000 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_l1258_125838


namespace NUMINAMATH_CALUDE_laptop_savings_l1258_125807

/-- The in-store price of the laptop in dollars -/
def in_store_price : ℚ := 299.99

/-- The cost of one payment in the radio offer in dollars -/
def radio_payment : ℚ := 55.98

/-- The number of payments in the radio offer -/
def num_payments : ℕ := 5

/-- The shipping and handling charge in dollars -/
def shipping_charge : ℚ := 12.99

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

theorem laptop_savings : 
  (in_store_price - (radio_payment * num_payments + shipping_charge)) * cents_per_dollar = 710 := by
  sorry

end NUMINAMATH_CALUDE_laptop_savings_l1258_125807


namespace NUMINAMATH_CALUDE_three_planes_seven_parts_intersection_lines_l1258_125899

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Represents the division of space by planes -/
structure SpaceDivision where
  planes : List Plane3D
  num_parts : Nat

/-- Counts the number of intersection lines between planes -/
def count_intersection_lines (division : SpaceDivision) : Nat :=
  sorry

theorem three_planes_seven_parts_intersection_lines 
  (division : SpaceDivision) 
  (h_planes : division.planes.length = 3)
  (h_parts : division.num_parts = 7) :
  count_intersection_lines division = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_planes_seven_parts_intersection_lines_l1258_125899


namespace NUMINAMATH_CALUDE_x_over_y_value_l1258_125897

theorem x_over_y_value (x y a b : ℝ) 
  (h1 : (2 * a - x) / (3 * b - y) = 3)
  (h2 : a / b = 4.5) :
  x / y = 3 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_value_l1258_125897


namespace NUMINAMATH_CALUDE_max_brownies_is_100_l1258_125849

/-- Represents the dimensions of a rectangular pan of brownies -/
structure BrowniePan where
  m : ℕ+  -- length
  n : ℕ+  -- width

/-- The total number of brownies in the pan -/
def totalBrownies (pan : BrowniePan) : ℕ := pan.m.val * pan.n.val

/-- The number of brownies along the perimeter of the pan -/
def perimeterBrownies (pan : BrowniePan) : ℕ := 2 * (pan.m.val + pan.n.val) - 4

/-- The condition that the total number of brownies is twice the perimeter brownies -/
def validCut (pan : BrowniePan) : Prop :=
  totalBrownies pan = 2 * perimeterBrownies pan

theorem max_brownies_is_100 :
  ∃ (pan : BrowniePan), validCut pan ∧
    (∀ (other : BrowniePan), validCut other → totalBrownies other ≤ totalBrownies pan) ∧
    totalBrownies pan = 100 :=
sorry

end NUMINAMATH_CALUDE_max_brownies_is_100_l1258_125849


namespace NUMINAMATH_CALUDE_tetrahedron_height_formula_l1258_125822

/-- Configuration of four mutually tangent spheres -/
structure SpheresConfiguration where
  small_radius : ℝ
  large_radius : ℝ
  small_spheres_count : ℕ
  on_flat_floor : Prop

/-- Tetrahedron circumscribing the spheres configuration -/
def circumscribing_tetrahedron (config : SpheresConfiguration) : Prop :=
  sorry

/-- Height of the tetrahedron from the floor to the opposite vertex -/
noncomputable def tetrahedron_height (config : SpheresConfiguration) : ℝ :=
  sorry

/-- Theorem stating the height of the tetrahedron -/
theorem tetrahedron_height_formula (config : SpheresConfiguration) 
  (h1 : config.small_radius = 2)
  (h2 : config.large_radius = 3)
  (h3 : config.small_spheres_count = 3)
  (h4 : config.on_flat_floor)
  (h5 : circumscribing_tetrahedron config) :
  tetrahedron_height config = (Real.sqrt 177 + 9 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_height_formula_l1258_125822


namespace NUMINAMATH_CALUDE_expression_simplification_l1258_125833

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (x - 2) / (x - 1) / (x + 1 - 3 / (x - 1)) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1258_125833


namespace NUMINAMATH_CALUDE_area_between_parabola_and_line_l1258_125826

theorem area_between_parabola_and_line : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := 1
  let lower_bound := -1
  let upper_bound := 1
  (∫ (x : ℝ) in lower_bound..upper_bound, g x - f x) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_area_between_parabola_and_line_l1258_125826


namespace NUMINAMATH_CALUDE_sequential_discount_equivalence_l1258_125824

/-- The equivalent single discount percentage for two sequential discounts -/
def equivalent_discount (first_discount second_discount : ℝ) : ℝ :=
  1 - (1 - first_discount) * (1 - second_discount)

/-- Theorem stating that a 15% discount followed by a 25% discount 
    is equivalent to a single 36.25% discount -/
theorem sequential_discount_equivalence : 
  equivalent_discount 0.15 0.25 = 0.3625 := by
  sorry

#eval equivalent_discount 0.15 0.25

end NUMINAMATH_CALUDE_sequential_discount_equivalence_l1258_125824


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1258_125816

theorem inequality_solution_set : 
  {x : ℝ | (2 / x + Real.sqrt (1 - x) ≥ 1 + Real.sqrt (1 - x)) ∧ (x > 0) ∧ (x ≤ 1)} = 
  {x : ℝ | x > 0 ∧ x ≤ 1} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1258_125816


namespace NUMINAMATH_CALUDE_store_purchase_count_l1258_125832

def num_cookie_flavors : ℕ := 6
def num_milk_flavors : ℕ := 4

def gamma_purchase_options (n : ℕ) : ℕ :=
  Nat.choose (num_cookie_flavors + num_milk_flavors) n

def delta_cookie_options (n : ℕ) : ℕ :=
  if n = 1 then
    num_cookie_flavors
  else if n = 2 then
    Nat.choose num_cookie_flavors 2 + num_cookie_flavors
  else if n = 3 then
    Nat.choose num_cookie_flavors 3 + num_cookie_flavors * (num_cookie_flavors - 1) + num_cookie_flavors
  else
    0

def total_purchase_options : ℕ :=
  gamma_purchase_options 3 +
  gamma_purchase_options 2 * delta_cookie_options 1 +
  gamma_purchase_options 1 * delta_cookie_options 2 +
  delta_cookie_options 3

theorem store_purchase_count : total_purchase_options = 656 := by
  sorry

end NUMINAMATH_CALUDE_store_purchase_count_l1258_125832


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1258_125809

theorem age_ratio_problem (a b : ℕ) 
  (h1 : a = 2 * b)  -- Present age ratio 6:3 simplifies to 2:1
  (h2 : a - 4 = b + 4)  -- A's age 4 years ago equals B's age 4 years hence
  : (a + 4) / (b - 4) = 5 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1258_125809


namespace NUMINAMATH_CALUDE_solve_average_height_l1258_125808

def average_height_problem (n : ℕ) (initial_average : ℝ) (incorrect_height : ℝ) (correct_height : ℝ) : Prop :=
  let total_incorrect := n * initial_average
  let height_difference := incorrect_height - correct_height
  let total_correct := total_incorrect - height_difference
  let actual_average := total_correct / n
  actual_average = 174.25

theorem solve_average_height :
  average_height_problem 20 175 151 136 := by sorry

end NUMINAMATH_CALUDE_solve_average_height_l1258_125808


namespace NUMINAMATH_CALUDE_fib_pisano_period_l1258_125884

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Pisano period for modulus 10 -/
def pisano_period : ℕ := 60

theorem fib_pisano_period :
  (∀ n : ℕ, n > 0 → fib n % 10 = fib (n + pisano_period) % 10) ∧
  (∀ t : ℕ, t > 0 → t < pisano_period →
    ∃ n : ℕ, n > 0 ∧ fib n % 10 ≠ fib (n + t) % 10) := by
  sorry

end NUMINAMATH_CALUDE_fib_pisano_period_l1258_125884


namespace NUMINAMATH_CALUDE_abigail_cookies_l1258_125887

theorem abigail_cookies (grayson_boxes : ℚ) (olivia_boxes : ℕ) (cookies_per_box : ℕ) (total_cookies : ℕ) :
  grayson_boxes = 3/4 →
  olivia_boxes = 3 →
  cookies_per_box = 48 →
  total_cookies = 276 →
  (total_cookies - (grayson_boxes * cookies_per_box + olivia_boxes * cookies_per_box)) / cookies_per_box = 2 := by
  sorry

end NUMINAMATH_CALUDE_abigail_cookies_l1258_125887


namespace NUMINAMATH_CALUDE_a_10_equals_133_l1258_125851

/-- The number of subsets of {1,2,...,n} with at least two elements and 
    the absolute difference between any two elements greater than 1 -/
def a (n : ℕ) : ℕ :=
  if n ≤ 2 then 0
  else if n = 3 then 1
  else if n = 4 then 3
  else a (n-1) + a (n-2) + (n-2)

/-- The main theorem to prove -/
theorem a_10_equals_133 : a 10 = 133 := by
  sorry

end NUMINAMATH_CALUDE_a_10_equals_133_l1258_125851


namespace NUMINAMATH_CALUDE_inequality_systems_solution_l1258_125859

theorem inequality_systems_solution :
  (∀ x : ℝ, (2 * x ≥ x - 1 ∧ 4 * x + 10 > x + 1) ↔ x ≥ -1) ∧
  (∀ x : ℝ, (2 * x - 7 < 5 - 2 * x ∧ x / 4 - 1 ≤ (x - 1) / 2) ↔ -2 ≤ x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_systems_solution_l1258_125859


namespace NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l1258_125871

/-- Proves that mixing solutions A and B results in a solution with approximately 1.44% liquid X -/
theorem liquid_x_percentage_in_mixed_solution :
  let solution_a_weight : ℝ := 400
  let solution_b_weight : ℝ := 700
  let liquid_x_percent_a : ℝ := 0.8
  let liquid_x_percent_b : ℝ := 1.8
  let total_weight := solution_a_weight + solution_b_weight
  let liquid_x_weight_a := solution_a_weight * (liquid_x_percent_a / 100)
  let liquid_x_weight_b := solution_b_weight * (liquid_x_percent_b / 100)
  let total_liquid_x_weight := liquid_x_weight_a + liquid_x_weight_b
  let result_percent := (total_liquid_x_weight / total_weight) * 100
  ∃ ε > 0, |result_percent - 1.44| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l1258_125871


namespace NUMINAMATH_CALUDE_cleaning_staff_lcm_l1258_125877

theorem cleaning_staff_lcm : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_staff_lcm_l1258_125877


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1258_125857

/-- Proves that (7 + 14i) / (3 - 4i) = 77/25 + 70/25 * i -/
theorem complex_fraction_simplification :
  (7 + 14 * Complex.I) / (3 - 4 * Complex.I) = 77/25 + 70/25 * Complex.I :=
by sorry


end NUMINAMATH_CALUDE_complex_fraction_simplification_l1258_125857


namespace NUMINAMATH_CALUDE_area_covered_five_strips_l1258_125805

/-- The area covered by overlapping rectangular strips -/
def area_covered (n : ℕ) (length width : ℝ) (intersection_width : ℝ) : ℝ :=
  n * length * width - (n.choose 2) * 2 * intersection_width^2

/-- Theorem stating the area covered by the specific configuration of strips -/
theorem area_covered_five_strips :
  area_covered 5 15 2 2 = 70 := by sorry

end NUMINAMATH_CALUDE_area_covered_five_strips_l1258_125805


namespace NUMINAMATH_CALUDE_add_1678_minutes_to_3_15_pm_l1258_125845

-- Define a custom time type
structure Time where
  hours : ℕ
  minutes : ℕ
  is_pm : Bool

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : ℕ) : Time :=
  sorry

-- Define the starting time (3:15 p.m.)
def startTime : Time :=
  { hours := 3, minutes := 15, is_pm := true }

-- Define the ending time (7:13 p.m. the next day)
def endTime : Time :=
  { hours := 7, minutes := 13, is_pm := true }

-- Theorem statement
theorem add_1678_minutes_to_3_15_pm (t : Time) :
  t = startTime → addMinutes t 1678 = endTime :=
  sorry

end NUMINAMATH_CALUDE_add_1678_minutes_to_3_15_pm_l1258_125845


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l1258_125811

/-- The trajectory of point Q given the conditions of the problem -/
theorem trajectory_of_Q (A P Q : ℝ × ℝ) : 
  A = (4, 0) →
  (P.1^2 + P.2^2 = 4) →
  (Q.1 - A.1, Q.2 - A.2) = (2*(P.1 - Q.1), 2*(P.2 - Q.2)) →
  (Q.1 - 4/3)^2 + Q.2^2 = 16/9 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l1258_125811


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1258_125853

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 1944 →
  margin = 288 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) * 100 = 54 := by
sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1258_125853


namespace NUMINAMATH_CALUDE_fashion_show_runway_time_l1258_125896

/-- Fashion Show Runway Time Calculation -/
theorem fashion_show_runway_time :
  let num_models : ℕ := 6
  let bathing_suits_per_model : ℕ := 2
  let evening_wear_per_model : ℕ := 3
  let time_per_trip : ℕ := 2

  let total_trips_per_model : ℕ := bathing_suits_per_model + evening_wear_per_model
  let total_trips : ℕ := num_models * total_trips_per_model
  let total_time : ℕ := total_trips * time_per_trip

  total_time = 60 := by sorry

end NUMINAMATH_CALUDE_fashion_show_runway_time_l1258_125896


namespace NUMINAMATH_CALUDE_greatest_product_l1258_125813

def digits : List Nat := [3, 5, 8, 6, 1]

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) * (10 * d + e)

theorem greatest_product :
  ∀ a b c d e,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 8 3 1 6 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_product_l1258_125813


namespace NUMINAMATH_CALUDE_divisibility_property_l1258_125886

theorem divisibility_property (A B : ℤ) 
  (h : ∀ k : ℤ, 1 ≤ k ∧ k ≤ 65 → (A + B) % k = 0) : 
  ((A + B) % 66 = 0) ∧ ¬(∀ C D : ℤ, (∀ k : ℤ, 1 ≤ k ∧ k ≤ 65 → (C + D) % k = 0) → (C + D) % 67 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l1258_125886


namespace NUMINAMATH_CALUDE_distance_between_points_l1258_125804

/-- The distance between the points (3, -2) and (10, 8) is √149 units. -/
theorem distance_between_points : Real.sqrt 149 = Real.sqrt ((10 - 3)^2 + (8 - (-2))^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1258_125804


namespace NUMINAMATH_CALUDE_remainder_problem_l1258_125862

theorem remainder_problem (k : ℕ+) (h : ∃ q : ℕ, 120 = k^2 * q + 12) :
  ∃ r : ℕ, 160 = k * (160 / k) + r ∧ r < k ∧ r = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1258_125862


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1258_125856

theorem arithmetic_sequence_sum (d : ℝ) (h : d ≠ 0) :
  let a : ℕ → ℝ := fun n => (n - 1 : ℝ) * d
  ∃ m : ℕ, a m = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) ∧ m = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1258_125856


namespace NUMINAMATH_CALUDE_wyatt_envelopes_l1258_125861

theorem wyatt_envelopes (blue : ℕ) (yellow : ℕ) : 
  yellow = blue - 4 →
  blue + yellow = 16 →
  blue = 10 := by
sorry

end NUMINAMATH_CALUDE_wyatt_envelopes_l1258_125861


namespace NUMINAMATH_CALUDE_discount_calculation_l1258_125898

/-- Proves that given a list price of 70, a final price of 59.85, and two successive discounts
    where one is 10%, the other discount percentage is 5%. -/
theorem discount_calculation (list_price : ℝ) (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  list_price = 70 →
  final_price = 59.85 →
  discount1 = 10 →
  final_price = list_price * (1 - discount1 / 100) * (1 - discount2 / 100) →
  discount2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l1258_125898


namespace NUMINAMATH_CALUDE_diagonals_30_gon_skipping_2_l1258_125873

/-- A convex polygon with n sides --/
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of diagonals in a convex n-gon that skip exactly k adjacent vertices at each end --/
def diagonals_skipping (n k : ℕ) : ℕ :=
  (n * (n - 2*k - 1)) / 2

/-- Theorem: In a 30-sided convex polygon, there are 375 diagonals that skip exactly 2 adjacent vertices at each end --/
theorem diagonals_30_gon_skipping_2 :
  diagonals_skipping 30 2 = 375 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_30_gon_skipping_2_l1258_125873


namespace NUMINAMATH_CALUDE_line_equation_sum_l1258_125867

/-- Given a line with slope 5 passing through the point (2,4), prove that m + b = -1 --/
theorem line_equation_sum (m b : ℝ) : 
  m = 5 →                   -- The slope is 5
  4 = 5 * 2 + b →           -- The line passes through (2,4)
  m + b = -1 :=             -- Prove that m + b = -1
by sorry

end NUMINAMATH_CALUDE_line_equation_sum_l1258_125867


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l1258_125854

theorem base_conversion_theorem : 
  ∃! n : ℕ, ∃ S : Finset ℕ, 
    (∀ c ∈ S, c ≥ 2 ∧ c^3 ≤ 250 ∧ 250 < c^4) ∧ 
    Finset.card S = n ∧ 
    n = 3 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l1258_125854


namespace NUMINAMATH_CALUDE_integral_equals_sqrt3_over_2_minus_ln2_l1258_125872

noncomputable def integral_function (x : ℝ) : ℝ := (Real.cos x)^2 / (1 + Real.cos x - Real.sin x)^2

theorem integral_equals_sqrt3_over_2_minus_ln2 :
  ∫ x in -((2 * Real.pi) / 3)..0, integral_function x = Real.sqrt 3 / 2 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_sqrt3_over_2_minus_ln2_l1258_125872


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1258_125819

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b = 1) :
  (1/a + 1/b) ≥ 455/36 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1258_125819


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l1258_125891

theorem roots_sum_of_squares (m n a b : ℝ) : 
  (∀ x, x^2 - m*x + n = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = m^2 - 2*n := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l1258_125891


namespace NUMINAMATH_CALUDE_rectangle_combinations_l1258_125860

-- Define the number of horizontal and vertical lines
def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 4

-- Define the number of lines needed to form a rectangle
def lines_for_rectangle : ℕ := 2

-- Theorem statement
theorem rectangle_combinations :
  (Nat.choose horizontal_lines lines_for_rectangle) *
  (Nat.choose vertical_lines lines_for_rectangle) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_combinations_l1258_125860


namespace NUMINAMATH_CALUDE_bounds_of_P_l1258_125812

/-- A convex n-gon divided into triangles by non-intersecting diagonals -/
structure ConvexNGon (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- The minimum number of transformations required to convert any subdivision into any other -/
def P (n : ℕ) : ℕ := sorry

/-- Main theorem about the bounds of P(n) -/
theorem bounds_of_P (n : ℕ) (polygon : ConvexNGon n) :
  (P n ≥ n - 3) ∧
  (P n ≤ 2*n - 7) ∧
  (n ≥ 13 → P n ≤ 2*n - 10) := by sorry

end NUMINAMATH_CALUDE_bounds_of_P_l1258_125812


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1258_125839

-- Define the slopes of the two lines
def slope1 : ℚ := 3 / 4
def slope2 (b : ℚ) : ℚ := -b / 2

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem perpendicular_lines_b_value :
  ∃ b : ℚ, perpendicular b ∧ b = 8 / 3 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1258_125839


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1258_125823

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (5 * x + 9) = 11 → x = 112 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1258_125823


namespace NUMINAMATH_CALUDE_weighted_mean_car_sales_approx_l1258_125828

/-- Represents the car sales data for a week -/
structure CarSalesWeek where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  tuesday_discount : ℚ
  wednesday_commission : ℚ
  friday_discount : ℚ
  saturday_commission : ℚ

/-- Calculates the weighted mean of car sales for a week -/
def weightedMeanCarSales (sales : CarSalesWeek) : ℚ :=
  let monday_weighted := sales.monday
  let tuesday_weighted := sales.tuesday * (1 - sales.tuesday_discount)
  let wednesday_weighted := sales.wednesday * (1 + sales.wednesday_commission)
  let thursday_weighted := sales.thursday
  let friday_weighted := sales.friday * (1 - sales.friday_discount)
  let saturday_weighted := sales.saturday * (1 + sales.saturday_commission)
  let total_weighted := monday_weighted + tuesday_weighted + wednesday_weighted + 
                        thursday_weighted + friday_weighted + saturday_weighted
  total_weighted / 6

/-- Theorem: The weighted mean of car sales for the given week is approximately 5.48 -/
theorem weighted_mean_car_sales_approx (sales : CarSalesWeek) 
  (h1 : sales.monday = 8)
  (h2 : sales.tuesday = 3)
  (h3 : sales.wednesday = 10)
  (h4 : sales.thursday = 4)
  (h5 : sales.friday = 4)
  (h6 : sales.saturday = 4)
  (h7 : sales.tuesday_discount = 1/10)
  (h8 : sales.wednesday_commission = 1/20)
  (h9 : sales.friday_discount = 3/20)
  (h10 : sales.saturday_commission = 7/100) :
  ∃ ε > 0, |weightedMeanCarSales sales - 548/100| < ε :=
sorry


end NUMINAMATH_CALUDE_weighted_mean_car_sales_approx_l1258_125828


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l1258_125814

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) :
  a > 1 ∨ b > 1 := by sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l1258_125814


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1258_125863

theorem functional_equation_solution 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) : 
  ∃! f : ℝ → ℝ, 
    (∀ x, x > 0 → f x > 0) ∧ 
    (∀ x, x > 0 → f (f x) + a * f x = b * (a + b) * x) ∧
    (∀ x, x > 0 → f x = b * x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1258_125863


namespace NUMINAMATH_CALUDE_algae_growth_l1258_125881

/-- Calculates the population of algae after a given time period. -/
def algaePopulation (initialPopulation : ℕ) (minutes : ℕ) : ℕ :=
  initialPopulation * 2^(minutes / 5)

/-- Theorem stating that the algae population grows from 50 to 6400 in 35 minutes. -/
theorem algae_growth :
  algaePopulation 50 35 = 6400 :=
by
  sorry

#eval algaePopulation 50 35

end NUMINAMATH_CALUDE_algae_growth_l1258_125881


namespace NUMINAMATH_CALUDE_minimize_area_between_curves_l1258_125883

/-- The cubic function C(x) = x^3 - 3x^2 + 2x -/
def C (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- The linear function L(x, a) = ax -/
def L (x a : ℝ) : ℝ := a * x

/-- The area S(a) bounded by C and L -/
def S (a : ℝ) : ℝ := sorry

/-- The theorem stating that the value of a minimizing S(a) is 38 - 27√2 -/
theorem minimize_area_between_curves :
  ∃ (a : ℝ), a > -1/4 ∧ ∀ (b : ℝ), b > -1/4 → S a ≤ S b ∧ a = 38 - 27 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_minimize_area_between_curves_l1258_125883


namespace NUMINAMATH_CALUDE_smallest_d_value_l1258_125831

theorem smallest_d_value (d : ℝ) : 
  (5 * Real.sqrt 5)^2 + (d + 4)^2 = (5 * d)^2 → d ≥ (1 + Real.sqrt 212.5) / 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_value_l1258_125831


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1258_125803

/-- Given that the solution set of ax² + bx + c > 0 is (-1/3, 2),
    prove that the solution set of cx² + bx + a < 0 is (-3, 1/2) -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1258_125803


namespace NUMINAMATH_CALUDE_consecutive_numbers_with_lcm_660_l1258_125801

theorem consecutive_numbers_with_lcm_660 (a b c : ℕ) : 
  b = a + 1 ∧ c = b + 1 ∧ Nat.lcm (Nat.lcm a b) c = 660 → 
  a = 10 ∧ b = 11 ∧ c = 12 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_with_lcm_660_l1258_125801


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l1258_125888

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ↔ C ≤ Real.sqrt (4/3) := by
  sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l1258_125888


namespace NUMINAMATH_CALUDE_ellipse_condition_ellipse_condition_converse_l1258_125893

/-- Represents a point in a 2D rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation m(x^2 + y^2 + 2y + 1) = (x - 2y + 3)^2 -/
def ellipseEquation (m : ℝ) (p : Point) : Prop :=
  m * (p.x^2 + p.y^2 + 2*p.y + 1) = (p.x - 2*p.y + 3)^2

/-- Defines what it means for the equation to represent an ellipse -/
def isEllipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    ∀ (p : Point), ellipseEquation m p ↔ 
      (p.x^2 / a^2) + (p.y^2 / b^2) = 1

/-- The main theorem: if the equation represents an ellipse, then m > 5 -/
theorem ellipse_condition (m : ℝ) :
  isEllipse m → m > 5 := by
  sorry

/-- The converse: if m > 5, then the equation represents an ellipse -/
theorem ellipse_condition_converse (m : ℝ) :
  m > 5 → isEllipse m := by
  sorry

end NUMINAMATH_CALUDE_ellipse_condition_ellipse_condition_converse_l1258_125893


namespace NUMINAMATH_CALUDE_min_cos_B_angle_A_values_l1258_125889

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a + t.c = 3 * Real.sqrt 3 ∧ t.b = 3

/-- The minimum value of cos B -/
theorem min_cos_B (t : Triangle) (h : TriangleConditions t) :
    (∀ t' : Triangle, TriangleConditions t' → Real.cos t'.B ≥ Real.cos t.B) →
    Real.cos t.B = 1/3 := by sorry

/-- The possible values of angle A when BA · BC = 3 -/
theorem angle_A_values (t : Triangle) (h : TriangleConditions t) :
    t.a * t.c * Real.cos t.B = 3 →
    t.A = Real.pi / 2 ∨ t.A = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_min_cos_B_angle_A_values_l1258_125889


namespace NUMINAMATH_CALUDE_average_score_of_group_specific_group_average_l1258_125846

theorem average_score_of_group (total_people : ℕ) (group1_size : ℕ) (group2_size : ℕ) 
  (group1_avg : ℝ) (group2_avg : ℝ) :
  total_people = group1_size + group2_size →
  (group1_size : ℝ) * group1_avg + (group2_size : ℝ) * group2_avg = 
    (total_people : ℝ) * ((group1_size : ℝ) * group1_avg + (group2_size : ℝ) * group2_avg) / (total_people : ℝ) :=
by
  sorry

-- The specific problem instance
theorem specific_group_average :
  let total_people : ℕ := 10
  let group1_size : ℕ := 6
  let group2_size : ℕ := 4
  let group1_avg : ℝ := 90
  let group2_avg : ℝ := 80
  ((group1_size : ℝ) * group1_avg + (group2_size : ℝ) * group2_avg) / (total_people : ℝ) = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_average_score_of_group_specific_group_average_l1258_125846


namespace NUMINAMATH_CALUDE_circle_radius_from_inscribed_rectangle_l1258_125847

theorem circle_radius_from_inscribed_rectangle (r : ℝ) : 
  (∃ (s : ℝ), s^2 = 72 ∧ s^2 = 2 * r^2) → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_inscribed_rectangle_l1258_125847


namespace NUMINAMATH_CALUDE_classrooms_needed_l1258_125890

def total_students : ℕ := 1675
def students_per_classroom : ℕ := 37

theorem classrooms_needed : 
  ∃ (n : ℕ), n * students_per_classroom ≥ total_students ∧ 
  ∀ (m : ℕ), m * students_per_classroom ≥ total_students → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_classrooms_needed_l1258_125890


namespace NUMINAMATH_CALUDE_manu_win_probability_l1258_125858

def coin_flip_game (num_players : ℕ) (manu_position : ℕ) (manu_heads_needed : ℕ) : ℚ :=
  sorry

theorem manu_win_probability :
  coin_flip_game 4 4 2 = 1 / 30 := by sorry

end NUMINAMATH_CALUDE_manu_win_probability_l1258_125858


namespace NUMINAMATH_CALUDE_max_popsicles_with_budget_l1258_125874

def single_price : ℚ := 3/2
def box3_price : ℚ := 3
def box7_price : ℚ := 5
def budget : ℚ := 12

def max_popsicles (s p3 p7 : ℕ) : ℕ := s + 3 * p3 + 7 * p7

def valid_purchase (s p3 p7 : ℕ) : Prop :=
  single_price * s + box3_price * p3 + box7_price * p7 ≤ budget

theorem max_popsicles_with_budget :
  ∃ (s p3 p7 : ℕ), valid_purchase s p3 p7 ∧
    max_popsicles s p3 p7 = 15 ∧
    ∀ (s' p3' p7' : ℕ), valid_purchase s' p3' p7' →
      max_popsicles s' p3' p7' ≤ 15 := by sorry

end NUMINAMATH_CALUDE_max_popsicles_with_budget_l1258_125874


namespace NUMINAMATH_CALUDE_product_restoration_l1258_125885

theorem product_restoration (P : ℕ) : 
  P = (List.range 11).foldl (· * ·) 1 →
  ∃ (a b c : ℕ), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    P = 399 * 100000 + a * 10000 + 68 * 100 + b * 10 + c →
  P = 39916800 := by sorry

end NUMINAMATH_CALUDE_product_restoration_l1258_125885


namespace NUMINAMATH_CALUDE_complex_set_forms_line_l1258_125882

/-- The set of complex numbers z such that (2-3i)z is real forms a line in the complex plane -/
theorem complex_set_forms_line : 
  ∃ (m : ℝ) (b : ℝ), 
    {z : ℂ | ∃ (r : ℝ), (2 - 3*I) * z = r} = 
    {z : ℂ | z.im = m * z.re + b} :=
by sorry

end NUMINAMATH_CALUDE_complex_set_forms_line_l1258_125882


namespace NUMINAMATH_CALUDE_adelkas_numbers_l1258_125852

theorem adelkas_numbers : ∃ (a b : ℕ), 
  0 < a ∧ a < b ∧ b < 100 ∧
  (Nat.gcd a b) < a ∧ a < b ∧ b < (Nat.lcm a b) ∧ (Nat.lcm a b) < 100 ∧
  (Nat.lcm a b) / (Nat.gcd a b) = Nat.gcd (Nat.gcd a b) (Nat.gcd a (Nat.gcd b (Nat.lcm a b))) ∧
  a = 12 ∧ b = 18 := by
sorry

end NUMINAMATH_CALUDE_adelkas_numbers_l1258_125852


namespace NUMINAMATH_CALUDE_f_3_eq_9_l1258_125895

/-- A function f that is monotonic on R and satisfies f(f(x) - 2^x) = 3 for all x ∈ R -/
def f : ℝ → ℝ :=
  sorry

/-- f is monotonic on R -/
axiom f_monotonic : Monotone f

/-- f satisfies f(f(x) - 2^x) = 3 for all x ∈ R -/
axiom f_property (x : ℝ) : f (f x - 2^x) = 3

/-- The main theorem: f(3) = 9 -/
theorem f_3_eq_9 : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_3_eq_9_l1258_125895


namespace NUMINAMATH_CALUDE_mary_money_l1258_125830

def quarters : ℕ := 21
def dimes : ℕ := (quarters - 7) / 2

def quarter_value : ℚ := 25 / 100
def dime_value : ℚ := 10 / 100

def total_money : ℚ := quarters * quarter_value + dimes * dime_value

theorem mary_money : total_money = 595 / 100 := by
  sorry

end NUMINAMATH_CALUDE_mary_money_l1258_125830


namespace NUMINAMATH_CALUDE_ascetics_equal_distance_l1258_125866

theorem ascetics_equal_distance (h m : ℝ) (h_pos : h > 0) (m_pos : m > 0) :
  ∃ x : ℝ, x > 0 ∧ 
  (x + (((x + h)^2 + (m * h)^2).sqrt) = h + m * h) ∧
  x = (h * m) / (m + 2) := by
sorry

end NUMINAMATH_CALUDE_ascetics_equal_distance_l1258_125866


namespace NUMINAMATH_CALUDE_trust_meteorologist_l1258_125820

-- Define the probability of a clear day
def prob_clear_day : ℝ := 0.74

-- Define the accuracy of a senator's forecast (as a variable)
variable (p : ℝ)

-- Define the accuracy of the meteorologist's forecast
def meteorologist_accuracy (p : ℝ) : ℝ := 1.5 * p

-- Define the event of both senators predicting a clear day and the meteorologist predicting rain
def forecast_event (p : ℝ) : ℝ := 
  (1 - meteorologist_accuracy p) * p * p * prob_clear_day + 
  meteorologist_accuracy p * (1 - p) * (1 - p) * (1 - prob_clear_day)

-- Theorem statement
theorem trust_meteorologist (p : ℝ) (h1 : 0 < p) (h2 : p < 1) : 
  meteorologist_accuracy p * (1 - p) * (1 - p) * (1 - prob_clear_day) / forecast_event p > 
  (1 - meteorologist_accuracy p) * p * p * prob_clear_day / forecast_event p :=
sorry

end NUMINAMATH_CALUDE_trust_meteorologist_l1258_125820


namespace NUMINAMATH_CALUDE_jean_calories_l1258_125806

/-- Calculates the total calories consumed based on pages written and calories per donut -/
def total_calories (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

/-- Proves that Jean eats 900 calories given the conditions -/
theorem jean_calories : total_calories 12 2 150 = 900 := by
  sorry

end NUMINAMATH_CALUDE_jean_calories_l1258_125806


namespace NUMINAMATH_CALUDE_job_choice_diploma_percentage_l1258_125818

theorem job_choice_diploma_percentage :
  let total_population : ℝ := 100
  let no_diploma_with_job : ℝ := 12
  let with_job_choice : ℝ := 40
  let with_diploma : ℝ := 43
  let without_job_choice : ℝ := total_population - with_job_choice
  let with_diploma_and_job : ℝ := with_job_choice - no_diploma_with_job
  let with_diploma_without_job : ℝ := with_diploma - with_diploma_and_job
  (with_diploma_without_job / without_job_choice) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_job_choice_diploma_percentage_l1258_125818


namespace NUMINAMATH_CALUDE_log_x2y2_value_l1258_125817

theorem log_x2y2_value (x y : ℝ) (hxy4 : Real.log (x * y^4) = 1) (hx3y : Real.log (x^3 * y) = 1) :
  Real.log (x^2 * y^2) = 10/11 := by
sorry

end NUMINAMATH_CALUDE_log_x2y2_value_l1258_125817


namespace NUMINAMATH_CALUDE_domino_coloring_properties_l1258_125864

/-- Definition of the number of possible colorings for a domino of length n -/
def A (n : ℕ) : ℕ := 2^n

/-- Definition of the number of valid colorings (no adjacent painted squares) for a domino of length n -/
def F : ℕ → ℕ
  | 0 => 1  -- Base case for convenience
  | 1 => 2
  | 2 => 3
  | (n+3) => F (n+2) + F (n+1)

theorem domino_coloring_properties :
  (∀ n : ℕ, A n = 2^n) ∧
  F 1 = 2 ∧ F 2 = 3 ∧ F 3 = 5 ∧ F 4 = 8 ∧
  (∀ n : ℕ, n ≥ 3 → F n = F (n-1) + F (n-2)) ∧
  (∀ n p : ℕ+, F (n + p + 1) = F n * F p + F (n-1) * F (p-1)) := by
  sorry

end NUMINAMATH_CALUDE_domino_coloring_properties_l1258_125864
