import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l4116_411602

/-- Given that 4x^5 + 3x^3 - 2x + 1 + g(x) = 7x^3 - 5x^2 + 4x - 3,
    prove that g(x) = -4x^5 + 4x^3 - 5x^2 + 6x - 4 -/
theorem problem_solution (x : ℝ) (g : ℝ → ℝ) 
    (h : ∀ x, 4*x^5 + 3*x^3 - 2*x + 1 + g x = 7*x^3 - 5*x^2 + 4*x - 3) : 
  g x = -4*x^5 + 4*x^3 - 5*x^2 + 6*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4116_411602


namespace NUMINAMATH_CALUDE_tan_alpha_value_l4116_411639

theorem tan_alpha_value (α β : Real) 
  (h1 : Real.tan (α - β) = 2) 
  (h2 : Real.tan β = 3) : 
  Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l4116_411639


namespace NUMINAMATH_CALUDE_customer_coin_count_l4116_411659

/-- Represents the quantity of each type of coin turned in by the customer --/
structure CoinQuantities where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat
  halfDollars : Nat
  oneDollarCoins : Nat
  twoDollarCoins : Nat
  australianFiftyCentCoins : Nat
  mexicanOnePesoCoins : Nat

/-- Calculates the total number of coins turned in --/
def totalCoins (coins : CoinQuantities) : Nat :=
  coins.pennies +
  coins.nickels +
  coins.dimes +
  coins.quarters +
  coins.halfDollars +
  coins.oneDollarCoins +
  coins.twoDollarCoins +
  coins.australianFiftyCentCoins +
  coins.mexicanOnePesoCoins

/-- Theorem: The total number of coins turned in by the customer is 159 --/
theorem customer_coin_count :
  ∃ (coins : CoinQuantities),
    coins.pennies = 38 ∧
    coins.nickels = 27 ∧
    coins.dimes = 19 ∧
    coins.quarters = 24 ∧
    coins.halfDollars = 13 ∧
    coins.oneDollarCoins = 17 ∧
    coins.twoDollarCoins = 5 ∧
    coins.australianFiftyCentCoins = 4 ∧
    coins.mexicanOnePesoCoins = 12 ∧
    totalCoins coins = 159 := by
  sorry

end NUMINAMATH_CALUDE_customer_coin_count_l4116_411659


namespace NUMINAMATH_CALUDE_exponent_properties_l4116_411668

theorem exponent_properties (a m n : ℝ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) (h4 : n > 1) :
  (a^(m/n) = (a^m)^(1/n)) ∧ 
  (a^0 = 1) ∧ 
  (a^(-m/n) = 1 / (a^m)^(1/n)) := by
sorry

end NUMINAMATH_CALUDE_exponent_properties_l4116_411668


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l4116_411643

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (h1 : n = 81) (h2 : sum = 9^5) :
  let median := sum / n
  median = 729 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l4116_411643


namespace NUMINAMATH_CALUDE_break_even_point_l4116_411656

/-- The break-even point for a company producing exam preparation manuals -/
theorem break_even_point (Q : ℝ) :
  (Q > 0) →  -- Ensure Q is positive for division
  (300 : ℝ) = 100 + 100000 / Q →  -- Price equals average cost
  Q = 500 := by
sorry

end NUMINAMATH_CALUDE_break_even_point_l4116_411656


namespace NUMINAMATH_CALUDE_existence_uniqueness_midpoint_l4116_411695

/-- Polygonal distance between two points -/
def polygonal_distance (A B : ℝ × ℝ) : ℝ :=
  |A.1 - B.1| + |A.2 - B.2|

/-- Theorem: Existence and uniqueness of point C satisfying given conditions -/
theorem existence_uniqueness_midpoint (A B : ℝ × ℝ) (h : A ≠ B) :
  ∃! C : ℝ × ℝ, 
    polygonal_distance A C + polygonal_distance C B = polygonal_distance A B ∧
    polygonal_distance A C = polygonal_distance C B ∧
    C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) := by
  sorry

#check existence_uniqueness_midpoint

end NUMINAMATH_CALUDE_existence_uniqueness_midpoint_l4116_411695


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l4116_411630

-- Define the original proposition
def original_prop (a c : ℝ) : Prop := a > 0 → a * c^2 ≥ 0

-- Define the inverse proposition
def inverse_prop (a c : ℝ) : Prop := a * c^2 ≥ 0 → a > 0

-- Theorem stating that inverse_prop is the inverse of original_prop
theorem inverse_of_proposition :
  ∀ a c : ℝ, inverse_prop a c ↔ ¬(∃ a c : ℝ, original_prop a c ∧ ¬(inverse_prop a c)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l4116_411630


namespace NUMINAMATH_CALUDE_max_triples_1955_l4116_411620

/-- The maximum number of triples that can be chosen from a set of points,
    such that each pair of triples has one point in common. -/
def max_triples (n : ℕ) : ℕ :=
  ((n - 1) * (n - 2)) / 4

/-- Theorem stating that for 1955 points, the maximum number of triples
    that can be chosen such that each pair of triples has one point in
    common is 977. -/
theorem max_triples_1955 :
  max_triples 1955 = 977 := by
  sorry


end NUMINAMATH_CALUDE_max_triples_1955_l4116_411620


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_APF_l4116_411649

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the right focus F
def F : ℝ × ℝ := (2, 0)

-- Define point A
def A : ℝ × ℝ := (-1, 1)

-- Define a point P on the left branch of the hyperbola
def P : ℝ × ℝ := sorry

-- Define the perimeter of triangle APF
def perimeter (P : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem min_perimeter_triangle_APF :
  ∀ P, hyperbola P.1 P.2 → P.1 < 0 →
  perimeter P ≥ 3 * Real.sqrt 2 + Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_APF_l4116_411649


namespace NUMINAMATH_CALUDE_yearly_pet_feeding_cost_l4116_411699

/-- Calculate the yearly cost to feed Harry's pets -/
theorem yearly_pet_feeding_cost :
  let num_geckos : ℕ := 3
  let num_iguanas : ℕ := 2
  let num_snakes : ℕ := 4
  let gecko_cost_per_month : ℕ := 15
  let iguana_cost_per_month : ℕ := 5
  let snake_cost_per_month : ℕ := 10
  let months_per_year : ℕ := 12
  
  (num_geckos * gecko_cost_per_month + 
   num_iguanas * iguana_cost_per_month + 
   num_snakes * snake_cost_per_month) * months_per_year = 1140 := by
  sorry

end NUMINAMATH_CALUDE_yearly_pet_feeding_cost_l4116_411699


namespace NUMINAMATH_CALUDE_polygon_with_120_degree_interior_angles_has_6_sides_l4116_411698

theorem polygon_with_120_degree_interior_angles_has_6_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 120 →
    (n - 2) * 180 = n * interior_angle →
    n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_120_degree_interior_angles_has_6_sides_l4116_411698


namespace NUMINAMATH_CALUDE_bus_capacity_fraction_l4116_411640

/-- The capacity of the train in number of people -/
def train_capacity : ℕ := 120

/-- The combined capacity of the two buses in number of people -/
def combined_bus_capacity : ℕ := 40

/-- The fraction of the train's capacity that each bus can hold -/
def bus_fraction : ℚ := 1 / 6

theorem bus_capacity_fraction :
  bus_fraction = combined_bus_capacity / (2 * train_capacity) :=
sorry

end NUMINAMATH_CALUDE_bus_capacity_fraction_l4116_411640


namespace NUMINAMATH_CALUDE_max_students_with_different_options_l4116_411647

/-- Represents an answer sheet for a test with 6 questions, each with 3 options -/
def AnswerSheet := Fin 6 → Fin 3

/-- Checks if three answer sheets have at least one question where all options are different -/
def hasDifferentOptions (s1 s2 s3 : AnswerSheet) : Prop :=
  ∃ q : Fin 6, s1 q ≠ s2 q ∧ s1 q ≠ s3 q ∧ s2 q ≠ s3 q

/-- The main theorem stating the maximum number of students -/
theorem max_students_with_different_options :
  ∀ n : ℕ,
  (∀ sheets : Fin n → AnswerSheet,
    ∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
      hasDifferentOptions (sheets i) (sheets j) (sheets k)) →
  n ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_students_with_different_options_l4116_411647


namespace NUMINAMATH_CALUDE_vector_parallelism_l4116_411624

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, 3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 = t * w.1 ∧ v.2 = t * w.2

theorem vector_parallelism :
  ∃! k : ℝ, parallel ((k * a.1 + b.1, k * a.2 + b.2) : ℝ × ℝ) ((a.1 - 3 * b.1, a.2 - 3 * b.2) : ℝ × ℝ) ∧
  k = -1/3 := by sorry

end NUMINAMATH_CALUDE_vector_parallelism_l4116_411624


namespace NUMINAMATH_CALUDE_least_positive_integer_with_property_l4116_411600

/-- Represents a three-digit number as 100a + 10b + c -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_digit : b < 10
  c_digit : c < 10

/-- The value of a ThreeDigitNumber -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The value of a ThreeDigitNumber with the leftmost digit removed -/
def ThreeDigitNumber.valueWithoutLeftmost (n : ThreeDigitNumber) : Nat :=
  10 * n.b + n.c

theorem least_positive_integer_with_property :
  ∃ (n : ThreeDigitNumber),
    n.value = 725 ∧
    n.valueWithoutLeftmost = n.value / 29 ∧
    ∀ (m : ThreeDigitNumber), m.valueWithoutLeftmost = m.value / 29 → n.value ≤ m.value :=
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_property_l4116_411600


namespace NUMINAMATH_CALUDE_tan_sum_quarter_pi_l4116_411606

theorem tan_sum_quarter_pi (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) : 
  Real.tan (α + π/4) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_quarter_pi_l4116_411606


namespace NUMINAMATH_CALUDE_trapezoid_area_l4116_411676

/-- Given an outer equilateral triangle with area 36, an inner equilateral triangle
    with area 4, and three congruent trapezoids between them, the area of one
    trapezoid is 32/3. -/
theorem trapezoid_area (outer_triangle : Real) (inner_triangle : Real) (num_trapezoids : Nat) :
  outer_triangle = 36 →
  inner_triangle = 4 →
  num_trapezoids = 3 →
  (outer_triangle - inner_triangle) / num_trapezoids = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l4116_411676


namespace NUMINAMATH_CALUDE_sisters_sandcastle_height_l4116_411661

theorem sisters_sandcastle_height 
  (janet_height : Float) 
  (height_difference : Float) 
  (h1 : janet_height = 3.6666666666666665) 
  (h2 : height_difference = 1.3333333333333333) : 
  janet_height - height_difference = 2.333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_sisters_sandcastle_height_l4116_411661


namespace NUMINAMATH_CALUDE_choose_from_four_and_three_l4116_411655

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (set1_size set2_size : ℕ) : ℕ :=
  set1_size * set2_size

/-- Theorem: Choosing one item from a set of 4 and one from a set of 3 results in 12 possibilities -/
theorem choose_from_four_and_three :
  choose_one_from_each 4 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_choose_from_four_and_three_l4116_411655


namespace NUMINAMATH_CALUDE_f_upper_bound_l4116_411664

def f_property (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 ≤ x ∧ 0 ≤ y → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 2016)

theorem f_upper_bound (f : ℝ → ℝ) (h : f_property f) :
  ∀ x, 0 ≤ x → f x ≤ x^2 := by sorry

end NUMINAMATH_CALUDE_f_upper_bound_l4116_411664


namespace NUMINAMATH_CALUDE_password_length_l4116_411613

/-- Represents the structure of Pat's password --/
structure PasswordStructure where
  lowercase_letters : Nat
  alternating_chars : Nat
  digits : Nat
  symbols : Nat

/-- Theorem stating that Pat's password contains 22 characters --/
theorem password_length (pw : PasswordStructure) 
  (h1 : pw.lowercase_letters = 10)
  (h2 : pw.alternating_chars = 6)
  (h3 : pw.digits = 4)
  (h4 : pw.symbols = 2) : 
  pw.lowercase_letters + pw.alternating_chars + pw.digits + pw.symbols = 22 := by
  sorry

#check password_length

end NUMINAMATH_CALUDE_password_length_l4116_411613


namespace NUMINAMATH_CALUDE_symmetry_about_59_l4116_411696

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the two functions y₁ and y₂
def y₁ (x : ℝ) : ℝ := f (x - 19)
def y₂ (x : ℝ) : ℝ := f (99 - x)

-- Theorem stating that y₁ and y₂ are symmetric about x = 59
theorem symmetry_about_59 :
  ∀ (x : ℝ), y₁ f (118 - x) = y₂ f x :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_59_l4116_411696


namespace NUMINAMATH_CALUDE_sum_of_two_primes_52_l4116_411635

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem sum_of_two_primes_52 :
  ∃! (count : ℕ), ∃ (pairs : List (ℕ × ℕ)),
    (∀ (p q : ℕ), (p, q) ∈ pairs → is_prime p ∧ is_prime q ∧ p + q = 52) ∧
    (∀ (p q : ℕ), is_prime p → is_prime q → p + q = 52 → (p, q) ∈ pairs ∨ (q, p) ∈ pairs) ∧
    count = pairs.length ∧
    count = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_two_primes_52_l4116_411635


namespace NUMINAMATH_CALUDE_unique_alpha_l4116_411621

def A (α : ℝ) : Set ℕ := {n : ℕ | ∃ k : ℕ, n = ⌊k * α⌋}

theorem unique_alpha : ∃! α : ℝ, 
  α ≥ 1 ∧ 
  (∃ r : ℕ, r < 2021 ∧ 
    (∀ n : ℕ, n > 0 → (n ∉ A α ↔ n % 2021 = r))) ∧
  α = 2021 / 2020 := by
sorry

end NUMINAMATH_CALUDE_unique_alpha_l4116_411621


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l4116_411679

theorem isosceles_triangle_condition (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- Ensure positive angles
  A + B + C = π →  -- Triangle angle sum
  2 * Real.cos B * Real.sin A = Real.sin C →  -- Given condition
  A = B  -- Conclusion: isosceles triangle
:= by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l4116_411679


namespace NUMINAMATH_CALUDE_tan_half_product_squared_l4116_411638

theorem tan_half_product_squared (a b : ℝ) :
  7 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_product_squared_l4116_411638


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_five_l4116_411652

/-- Given two nonconstant geometric sequences with different common ratios,
    if a specific condition is met, prove that the sum of their common ratios is 5. -/
theorem sum_of_common_ratios_is_five
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r)
  (h : k * p^2 - k * r^2 = 5 * (k * p - k * r)) (hk : k ≠ 0) :
  p + r = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_five_l4116_411652


namespace NUMINAMATH_CALUDE_corey_sunday_vs_saturday_l4116_411618

/-- Corey's goal for finding golf balls -/
def goal : ℕ := 48

/-- Number of golf balls Corey found on Saturday -/
def saturdayBalls : ℕ := 16

/-- Number of golf balls Corey still needs to reach his goal -/
def stillNeeded : ℕ := 14

/-- Number of golf balls Corey found on Sunday -/
def sundayBalls : ℕ := goal - saturdayBalls - stillNeeded

theorem corey_sunday_vs_saturday : sundayBalls - saturdayBalls = 2 := by
  sorry

end NUMINAMATH_CALUDE_corey_sunday_vs_saturday_l4116_411618


namespace NUMINAMATH_CALUDE_egg_count_l4116_411614

theorem egg_count (initial_eggs used_eggs chickens eggs_per_chicken : ℕ) :
  initial_eggs ≥ used_eggs →
  (initial_eggs - used_eggs) + chickens * eggs_per_chicken =
  initial_eggs - used_eggs + chickens * eggs_per_chicken :=
by sorry

end NUMINAMATH_CALUDE_egg_count_l4116_411614


namespace NUMINAMATH_CALUDE_flight_time_around_earth_l4116_411651

def earth_radius : ℝ := 6000
def jet_speed : ℝ := 600

theorem flight_time_around_earth :
  let circumference := 2 * Real.pi * earth_radius
  let flight_time := circumference / jet_speed
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ abs (flight_time - 63) < ε := by
sorry

end NUMINAMATH_CALUDE_flight_time_around_earth_l4116_411651


namespace NUMINAMATH_CALUDE_book_selling_price_l4116_411681

-- Define the cost price and profit rate
def cost_price : ℝ := 50
def profit_rate : ℝ := 0.20

-- Define the selling price function
def selling_price (cost : ℝ) (rate : ℝ) : ℝ :=
  cost * (1 + rate)

-- Theorem statement
theorem book_selling_price :
  selling_price cost_price profit_rate = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_selling_price_l4116_411681


namespace NUMINAMATH_CALUDE_factor_expression_l4116_411625

theorem factor_expression (x : ℝ) : 3*x*(x-5) + 7*(x-5) - 2*(x-5) = (3*x+5)*(x-5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4116_411625


namespace NUMINAMATH_CALUDE_impossibleRectangle_l4116_411667

/-- Represents the counts of sticks of each length -/
structure StickCounts where
  one_cm : Nat
  two_cm : Nat
  three_cm : Nat
  four_cm : Nat

/-- Calculates the total length of all sticks -/
def totalLength (counts : StickCounts) : Nat :=
  counts.one_cm * 1 + counts.two_cm * 2 + counts.three_cm * 3 + counts.four_cm * 4

/-- Theorem stating that it's impossible to form a rectangle with the given sticks -/
theorem impossibleRectangle (counts : StickCounts) 
  (h1 : counts.one_cm = 4)
  (h2 : counts.two_cm = 4)
  (h3 : counts.three_cm = 7)
  (h4 : counts.four_cm = 5)
  (h5 : totalLength counts = 53) :
  ¬∃ (a b : Nat), a + b = (totalLength counts) / 2 := by
  sorry

#eval totalLength { one_cm := 4, two_cm := 4, three_cm := 7, four_cm := 5 }

end NUMINAMATH_CALUDE_impossibleRectangle_l4116_411667


namespace NUMINAMATH_CALUDE_tables_needed_for_children_twenty_tables_needed_l4116_411603

theorem tables_needed_for_children (num_children : ℕ) (table_capacity : ℕ) (num_tables : ℕ) : Prop :=
  num_children > 0 ∧ 
  table_capacity > 0 ∧ 
  num_tables * table_capacity ≥ num_children ∧ 
  (num_tables - 1) * table_capacity < num_children

theorem twenty_tables_needed : tables_needed_for_children 156 8 20 := by
  sorry

end NUMINAMATH_CALUDE_tables_needed_for_children_twenty_tables_needed_l4116_411603


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l4116_411685

/-- Given a line L1 with equation x - 2y + 1 = 0, 
    and a line of symmetry y = x,
    the line L2 symmetric to L1 with respect to y = x
    has the equation x + 2y - 1 = 0 -/
theorem symmetric_line_equation :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ x - 2*y + 1 = 0
  let symmetry_line : ℝ → ℝ → Prop := λ x y ↦ y = x
  let L2 : ℝ → ℝ → Prop := λ x y ↦ x + 2*y - 1 = 0
  ∀ x y : ℝ, L2 x y ↔ (∃ x' y' : ℝ, L1 x' y' ∧ 
    (x = (x' + y')/2 ∧ y = (x' + y')/2))
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l4116_411685


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_h_l4116_411662

/-- Given a quadratic expression 3x^2 + 9x + 20, when expressed in the form a(x - h)^2 + k, h = -3/2 -/
theorem quadratic_vertex_form_h (x : ℝ) : 
  ∃ (a k : ℝ), 3*x^2 + 9*x + 20 = a*(x - (-3/2))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_h_l4116_411662


namespace NUMINAMATH_CALUDE_multiply_63_37_l4116_411642

theorem multiply_63_37 : 63 * 37 = 2331 := by
  sorry

end NUMINAMATH_CALUDE_multiply_63_37_l4116_411642


namespace NUMINAMATH_CALUDE_percentage_of_muslim_boys_l4116_411650

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) : 
  total_boys = 850 →
  hindu_percentage = 32/100 →
  sikh_percentage = 10/100 →
  other_boys = 119 →
  (total_boys - (hindu_percentage * total_boys).num - (sikh_percentage * total_boys).num - other_boys) / total_boys = 44/100 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_muslim_boys_l4116_411650


namespace NUMINAMATH_CALUDE_zoo_visitors_l4116_411604

theorem zoo_visitors (num_cars : ℝ) (people_per_car : ℝ) 
  (h1 : num_cars = 3.0) 
  (h2 : people_per_car = 63.0) : 
  num_cars * people_per_car = 189.0 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l4116_411604


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4116_411609

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) is 2,
    given that one of its asymptotes is tangent to the circle (x - √3)² + (y - 1)² = 1. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 ∧
    ((x - Real.sqrt 3)^2 + (y - 1)^2 = 1 ∨
     (x + Real.sqrt 3)^2 + (y - 1)^2 = 1)) →
  Real.sqrt ((a^2 + b^2) / a^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4116_411609


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4116_411687

theorem quadratic_inequality (x : ℝ) : (x - 2) * (x + 2) > 0 ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4116_411687


namespace NUMINAMATH_CALUDE_cake_fraction_eaten_l4116_411692

theorem cake_fraction_eaten (total_slices : ℕ) (kept_slices : ℕ) : 
  total_slices = 12 → kept_slices = 9 → (total_slices - kept_slices : ℚ) / total_slices = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cake_fraction_eaten_l4116_411692


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l4116_411658

theorem geometric_arithmetic_sequence_problem (x y z : ℝ) 
  (h1 : (12 * y)^2 = 9 * x * 15 * z)  -- 9x, 12y, 15z form a geometric sequence
  (h2 : 2 / y = 1 / x + 1 / z)        -- 1/x, 1/y, 1/z form an arithmetic sequence
  : x / z + z / x = 34 / 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l4116_411658


namespace NUMINAMATH_CALUDE_mango_rate_is_65_l4116_411623

/-- The rate per kg for mangoes given the following conditions:
    - Tom purchased 8 kg of apples at 70 per kg
    - Tom purchased 9 kg of mangoes
    - Tom paid a total of 1145 to the shopkeeper -/
def mango_rate (apple_weight : ℕ) (apple_rate : ℕ) (mango_weight : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - apple_weight * apple_rate) / mango_weight

/-- Theorem stating that the rate per kg for mangoes is 65 -/
theorem mango_rate_is_65 : mango_rate 8 70 9 1145 = 65 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_is_65_l4116_411623


namespace NUMINAMATH_CALUDE_two_numbers_problem_l4116_411612

theorem two_numbers_problem (x y : ℕ) (h1 : x + y = 60) (h2 : Nat.gcd x y + Nat.lcm x y = 84) :
  (x = 24 ∧ y = 36) ∨ (x = 36 ∧ y = 24) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l4116_411612


namespace NUMINAMATH_CALUDE_rightmost_book_price_l4116_411671

/-- Represents the price of a book at a given position. -/
def book_price (first_price : ℕ) (position : ℕ) : ℕ :=
  first_price + 3 * (position - 1)

/-- The theorem states that for a sequence of 41 books with the given conditions,
    the price of the rightmost book is $150. -/
theorem rightmost_book_price (first_price : ℕ) :
  (book_price first_price 41 = 
   book_price first_price 20 + 
   book_price first_price 21 + 
   book_price first_price 22) →
  book_price first_price 41 = 150 := by
sorry

end NUMINAMATH_CALUDE_rightmost_book_price_l4116_411671


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_three_one_forty_one_satisfies_conditions_main_result_l4116_411677

theorem greatest_integer_with_gcf_three (n : ℕ) : n < 150 ∧ Nat.gcd n 24 = 3 → n ≤ 141 :=
by
  sorry

theorem one_forty_one_satisfies_conditions : 141 < 150 ∧ Nat.gcd 141 24 = 3 :=
by
  sorry

theorem main_result : ∃ (n : ℕ), n < 150 ∧ Nat.gcd n 24 = 3 ∧ 
  ∀ (m : ℕ), m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_three_one_forty_one_satisfies_conditions_main_result_l4116_411677


namespace NUMINAMATH_CALUDE_equation_positive_root_m_value_l4116_411660

theorem equation_positive_root_m_value (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ m / (x - 3) - 1 / (3 - x) = 2) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_equation_positive_root_m_value_l4116_411660


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l4116_411626

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  (a ≥ 3/2 ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l4116_411626


namespace NUMINAMATH_CALUDE_difference_of_squares_equals_cube_l4116_411673

theorem difference_of_squares_equals_cube (r : ℕ+) :
  ∃ m n : ℤ, m^2 - n^2 = (r : ℤ)^3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_equals_cube_l4116_411673


namespace NUMINAMATH_CALUDE_faster_speed_calculation_l4116_411636

/-- Proves that given a 1200-mile trip, if driving at a certain speed saves 4 hours
    compared to driving at 50 miles per hour, then that certain speed is 60 miles per hour. -/
theorem faster_speed_calculation (trip_distance : ℝ) (original_speed : ℝ) (time_saved : ℝ) 
    (faster_speed : ℝ) : 
    trip_distance = 1200 → 
    original_speed = 50 → 
    time_saved = 4 → 
    trip_distance / original_speed - trip_distance / faster_speed = time_saved → 
    faster_speed = 60 := by
  sorry

#check faster_speed_calculation

end NUMINAMATH_CALUDE_faster_speed_calculation_l4116_411636


namespace NUMINAMATH_CALUDE_bottles_from_625_l4116_411672

/-- The number of new bottles that can be made from a given number of initial bottles -/
def new_bottles (initial : ℕ) : ℕ :=
  if initial < 5 then 0
  else (initial / 5) + new_bottles (initial / 5)

/-- The theorem stating that 625 initial bottles will result in 195 new bottles -/
theorem bottles_from_625 :
  new_bottles 625 = 195 := by
sorry

end NUMINAMATH_CALUDE_bottles_from_625_l4116_411672


namespace NUMINAMATH_CALUDE_orangeade_price_day_two_l4116_411694

/-- Represents the price of orangeade per glass on a given day. -/
structure OrangeadePrice where
  day : Nat
  price : ℚ

/-- Represents the amount of ingredients used to make orangeade on a given day. -/
structure OrangeadeIngredients where
  day : Nat
  orange_juice : ℚ
  water : ℚ

/-- Calculates the total volume of orangeade made on a given day. -/
def totalVolume (ingredients : OrangeadeIngredients) : ℚ :=
  ingredients.orange_juice + ingredients.water

/-- Calculates the revenue from selling orangeade on a given day. -/
def revenue (price : OrangeadePrice) (ingredients : OrangeadeIngredients) : ℚ :=
  price.price * totalVolume ingredients

/-- Theorem stating that the price of orangeade on the second day is $0.40 given the conditions. -/
theorem orangeade_price_day_two
  (day1_price : OrangeadePrice)
  (day1_ingredients : OrangeadeIngredients)
  (day2_ingredients : OrangeadeIngredients)
  (h1 : day1_price.day = 1)
  (h2 : day1_price.price = 6/10)
  (h3 : day1_ingredients.day = 1)
  (h4 : day1_ingredients.orange_juice = day1_ingredients.water)
  (h5 : day2_ingredients.day = 2)
  (h6 : day2_ingredients.orange_juice = day1_ingredients.orange_juice)
  (h7 : day2_ingredients.water = 2 * day1_ingredients.water)
  (h8 : revenue day1_price day1_ingredients = revenue { day := 2, price := 4/10 } day2_ingredients) :
  ∃ (day2_price : OrangeadePrice), day2_price.day = 2 ∧ day2_price.price = 4/10 :=
by sorry


end NUMINAMATH_CALUDE_orangeade_price_day_two_l4116_411694


namespace NUMINAMATH_CALUDE_A_B_symmetrical_wrt_origin_l4116_411693

/-- Two points are symmetrical with respect to the origin if their coordinates are negatives of each other -/
def symmetrical_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

/-- Given points A and B in the Cartesian coordinate system -/
def A : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (-2, 1)

/-- Theorem: Points A and B are symmetrical with respect to the origin -/
theorem A_B_symmetrical_wrt_origin : symmetrical_wrt_origin A B := by
  sorry

end NUMINAMATH_CALUDE_A_B_symmetrical_wrt_origin_l4116_411693


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_sum_zero_implies_extremes_sum_zero_l4116_411674

theorem sum_and_reciprocal_sum_zero_implies_extremes_sum_zero
  (a b c d : ℝ)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (h_sum : a + b + c + d = 0)
  (h_reciprocal_sum : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_sum_zero_implies_extremes_sum_zero_l4116_411674


namespace NUMINAMATH_CALUDE_subtract_preserves_inequality_l4116_411653

theorem subtract_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtract_preserves_inequality_l4116_411653


namespace NUMINAMATH_CALUDE_f_eval_approx_l4116_411631

/-- The polynomial function f(x) -/
def f (x : ℝ) : ℝ := 1 + x + 0.5*x^2 + 0.16667*x^3 + 0.04167*x^4 + 0.00833*x^5

/-- The evaluation point -/
def x₀ : ℝ := -0.2

/-- The theorem stating that f(x₀) is approximately equal to 0.81873 -/
theorem f_eval_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ |f x₀ - 0.81873| < ε := by
  sorry

end NUMINAMATH_CALUDE_f_eval_approx_l4116_411631


namespace NUMINAMATH_CALUDE_boris_candy_problem_l4116_411684

/-- Given the initial number of candy pieces, the number eaten by the daughter,
    the number of bowls, and the number of pieces taken from each bowl,
    calculate the final number of pieces in one bowl. -/
def candyInBowl (initial : ℕ) (eaten : ℕ) (bowls : ℕ) (taken : ℕ) : ℕ :=
  (initial - eaten) / bowls - taken

theorem boris_candy_problem :
  candyInBowl 100 8 4 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_boris_candy_problem_l4116_411684


namespace NUMINAMATH_CALUDE_zero_in_interval_implies_m_leq_neg_one_l4116_411632

/-- A function f(x) = x² + (m-1)x + 1 has a zero point in the interval [0, 2] -/
def has_zero_in_interval (m : ℝ) : Prop :=
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^2 + (m-1)*x + 1 = 0

/-- If f(x) = x² + (m-1)x + 1 has a zero point in the interval [0, 2], then m ≤ -1 -/
theorem zero_in_interval_implies_m_leq_neg_one (m : ℝ) :
  has_zero_in_interval m → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_implies_m_leq_neg_one_l4116_411632


namespace NUMINAMATH_CALUDE_evaluate_g_l4116_411680

/-- The function g(x) = 3x^2 - 5x + 7 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

/-- Theorem: 3g(5) + 4g(-2) = 287 -/
theorem evaluate_g : 3 * g 5 + 4 * g (-2) = 287 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l4116_411680


namespace NUMINAMATH_CALUDE_fourth_side_is_six_l4116_411657

/-- Represents a quadrilateral pyramid with a base ABCD and apex S -/
structure QuadrilateralPyramid where
  /-- Length of side AB of the base -/
  ab : ℝ
  /-- Length of side BC of the base -/
  bc : ℝ
  /-- Length of side CD of the base -/
  cd : ℝ
  /-- Length of side DA of the base -/
  da : ℝ
  /-- Predicate indicating that all dihedral angles at the base are equal -/
  equal_dihedral_angles : Prop

/-- Theorem stating that for a quadrilateral pyramid with given side lengths and equal dihedral angles,
    the fourth side of the base is 6 -/
theorem fourth_side_is_six (p : QuadrilateralPyramid)
  (h1 : p.ab = 5)
  (h2 : p.bc = 7)
  (h3 : p.cd = 8)
  (h4 : p.equal_dihedral_angles) :
  p.da = 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_is_six_l4116_411657


namespace NUMINAMATH_CALUDE_alligator_growth_rate_l4116_411688

def alligator_population (initial_population : ℕ) (rate : ℕ) (periods : ℕ) : ℕ :=
  initial_population + rate * periods

theorem alligator_growth_rate :
  ∀ (rate : ℕ),
    alligator_population 4 rate 2 = 16 →
    rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_alligator_growth_rate_l4116_411688


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l4116_411619

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 - 2*m - 2025 = 0) → 
  (n^2 - 2*n - 2025 = 0) → 
  (m^2 - 3*m - n = 2023) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l4116_411619


namespace NUMINAMATH_CALUDE_incenter_centroid_parallel_implies_arithmetic_sequence_l4116_411633

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle. -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle. -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Checks if two points form a line parallel to a side of the triangle. -/
def is_parallel_to_side (p q : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- The lengths of the sides of a triangle. -/
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- Checks if three numbers form an arithmetic sequence. -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := sorry

theorem incenter_centroid_parallel_implies_arithmetic_sequence (t : Triangle) :
  is_parallel_to_side (incenter t) (centroid t) t →
  let (a, b, c) := side_lengths t
  is_arithmetic_sequence a b c := by sorry

end NUMINAMATH_CALUDE_incenter_centroid_parallel_implies_arithmetic_sequence_l4116_411633


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_ratio_l4116_411691

/-- For an ellipse with equation mx^2 + ny^2 = 1, foci on the x-axis, and eccentricity 1/2,
    the ratio m/n is equal to 3/4. -/
theorem ellipse_eccentricity_ratio (m n : ℝ) (m_pos : 0 < m) (n_pos : 0 < n) :
  (∀ x y : ℝ, m * x^2 + n * y^2 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c^2 = 1/m - 1/n) →  -- Foci on x-axis condition
  ((1 - 1/n) / (1/m))^(1/2) = 1/2 →  -- Eccentricity condition
  m / n = 3/4 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_ratio_l4116_411691


namespace NUMINAMATH_CALUDE_inequality_proof_l4116_411610

theorem inequality_proof (x : ℝ) : 3 * (2 * x - 1) - 2 * (x + 1) ≤ 1 → x ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4116_411610


namespace NUMINAMATH_CALUDE_initial_friends_correct_l4116_411644

/-- The number of friends James had initially -/
def initial_friends : ℕ := 20

/-- The number of friends James lost due to an argument -/
def friends_lost : ℕ := 2

/-- The number of new friends James made -/
def new_friends : ℕ := 1

/-- The number of friends James has now -/
def current_friends : ℕ := 19

/-- Theorem stating that the initial number of friends is correct given the conditions -/
theorem initial_friends_correct :
  initial_friends = current_friends + friends_lost - new_friends :=
by sorry

end NUMINAMATH_CALUDE_initial_friends_correct_l4116_411644


namespace NUMINAMATH_CALUDE_optimal_garden_max_area_l4116_411605

/-- Represents a rectangular garden with given constraints --/
structure Garden where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 200
  length_min : length ≥ 100
  width_min : width ≥ 50
  length_width_diff : length ≥ width + 20

/-- The area of a garden --/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- The optimal garden dimensions and area --/
def optimal_garden : Garden := {
  length := 120,
  width := 80,
  perimeter_constraint := by sorry,
  length_min := by sorry,
  width_min := by sorry,
  length_width_diff := by sorry
}

/-- Theorem stating that the optimal garden has the maximum area --/
theorem optimal_garden_max_area :
  ∀ g : Garden, garden_area g ≤ garden_area optimal_garden := by sorry

end NUMINAMATH_CALUDE_optimal_garden_max_area_l4116_411605


namespace NUMINAMATH_CALUDE_route_choice_and_expected_value_l4116_411607

-- Define the data types
structure RouteData where
  good : ℕ
  average : ℕ

structure GenderRouteData where
  male : ℕ
  female : ℕ

-- Define the constants
def total_tourists : ℕ := 300
def route_a : RouteData := { good := 50, average := 75 }
def route_b : RouteData := { good := 75, average := 100 }
def gender_data : GenderRouteData := { male := 120, female := 180 }

-- Define the K^2 formula
def k_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for K^2 at 0.001 significance level
def k_critical : ℚ := 10.828

-- Define the expected value calculation
def expected_value (good_prob : ℚ) : ℚ :=
  let good_score := 5
  let avg_score := 2
  (1 - good_prob)^3 * (3 * avg_score) +
  3 * good_prob * (1 - good_prob)^2 * (2 * avg_score + good_score) +
  3 * good_prob^2 * (1 - good_prob) * (avg_score + 2 * good_score) +
  good_prob^3 * (3 * good_score)

-- Theorem statement
theorem route_choice_and_expected_value :
  let k_value := k_squared gender_data.male (gender_data.female - gender_data.male)
                            (total_tourists - gender_data.male - gender_data.female) gender_data.female
  let prob_a := (route_a.good : ℚ) / (route_a.good + route_a.average)
  let prob_b := (route_b.good : ℚ) / (route_b.good + route_b.average)
  k_value > k_critical ∧ expected_value prob_a > expected_value prob_b := by
  sorry

end NUMINAMATH_CALUDE_route_choice_and_expected_value_l4116_411607


namespace NUMINAMATH_CALUDE_darrel_coin_counting_machine_result_l4116_411616

/-- Calculates the amount received after fees for a given coin type -/
def amountAfterFee (coinValue : ℚ) (count : ℕ) (feePercentage : ℚ) : ℚ :=
  let totalValue := coinValue * count
  totalValue - (totalValue * feePercentage / 100)

/-- Theorem stating the total amount Darrel receives after fees -/
theorem darrel_coin_counting_machine_result : 
  let quarterCount : ℕ := 127
  let dimeCount : ℕ := 183
  let nickelCount : ℕ := 47
  let pennyCount : ℕ := 237
  let halfDollarCount : ℕ := 64
  
  let quarterValue : ℚ := 25 / 100
  let dimeValue : ℚ := 10 / 100
  let nickelValue : ℚ := 5 / 100
  let pennyValue : ℚ := 1 / 100
  let halfDollarValue : ℚ := 50 / 100
  
  let quarterFee : ℚ := 12
  let dimeFee : ℚ := 7
  let nickelFee : ℚ := 15
  let pennyFee : ℚ := 10
  let halfDollarFee : ℚ := 5
  
  let totalAfterFees := 
    amountAfterFee quarterValue quarterCount quarterFee +
    amountAfterFee dimeValue dimeCount dimeFee +
    amountAfterFee nickelValue nickelCount nickelFee +
    amountAfterFee pennyValue pennyCount pennyFee +
    amountAfterFee halfDollarValue halfDollarCount halfDollarFee
  
  totalAfterFees = 7949 / 100 := by
  sorry


end NUMINAMATH_CALUDE_darrel_coin_counting_machine_result_l4116_411616


namespace NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_400_l4116_411611

theorem multiplicative_inverse_123_mod_400 : ∃ a : ℕ, a < 400 ∧ (123 * a) % 400 = 1 :=
by
  use 387
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_400_l4116_411611


namespace NUMINAMATH_CALUDE_second_number_value_l4116_411608

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 330 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a → 
  b = 90 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l4116_411608


namespace NUMINAMATH_CALUDE_prob_same_heads_value_l4116_411663

-- Define the probability of heads for a fair coin
def fair_coin_prob : ℚ := 1/2

-- Define the probability of heads for the biased coin
def biased_coin_prob : ℚ := 5/8

-- Define the number of fair coins
def num_fair_coins : ℕ := 3

-- Define the number of biased coins
def num_biased_coins : ℕ := 1

-- Define the total number of coins
def total_coins : ℕ := num_fair_coins + num_biased_coins

-- Define the function to calculate the probability of getting the same number of heads
def prob_same_heads : ℚ := sorry

-- Theorem statement
theorem prob_same_heads_value : prob_same_heads = 77/225 := by sorry

end NUMINAMATH_CALUDE_prob_same_heads_value_l4116_411663


namespace NUMINAMATH_CALUDE_fraction_value_l4116_411678

theorem fraction_value (x y : ℝ) (hx : x = 4) (hy : y = -3) :
  (x - 2*y) / (x + y) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l4116_411678


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l4116_411670

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote
def asymptote (a b x y : ℝ) : Prop := y = (b / a) * x ∨ y = -(b / a) * x

-- Define the perpendicularity condition
def perpendicular (F₁ F₂ A : ℝ × ℝ) : Prop :=
  (A.2 - F₂.2) * (F₂.1 - F₁.1) = (F₂.2 - F₁.2) * (A.1 - F₂.1)

-- Define the distance condition
def distance_condition (O F₁ A : ℝ × ℝ) : Prop :=
  let d := abs ((A.2 - F₁.2) * O.1 - (A.1 - F₁.1) * O.2 + A.1 * F₁.2 - A.2 * F₁.1) /
            Real.sqrt ((A.2 - F₁.2)^2 + (A.1 - F₁.1)^2)
  d = (1/3) * Real.sqrt (F₁.1^2 + F₁.2^2)

-- Main theorem
theorem hyperbola_asymptote_slope (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (A F₁ F₂ O : ℝ × ℝ) :
  hyperbola a b A.1 A.2 →
  asymptote a b A.1 A.2 →
  perpendicular F₁ F₂ A →
  distance_condition O F₁ A →
  (b / a = Real.sqrt 2 / 2) ∨ (b / a = -Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l4116_411670


namespace NUMINAMATH_CALUDE_students_before_yoongi_l4116_411669

theorem students_before_yoongi (total_students : ℕ) (finished_after_yoongi : ℕ) : 
  total_students = 20 → finished_after_yoongi = 11 → 
  total_students - finished_after_yoongi - 1 = 8 := by
sorry

end NUMINAMATH_CALUDE_students_before_yoongi_l4116_411669


namespace NUMINAMATH_CALUDE_range_of_m_l4116_411634

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x + x^2 - a*x + 2

theorem range_of_m (a : ℝ) (x₀ : ℝ) :
  (∀ a ∈ Set.Icc (-2) 0, ∃ x₀ ∈ Set.Ioc 0 1, 
    f x₀ a > a^2 + 3*a + 2 - 2*m*(Real.exp a)*(a+1)) →
  m ∈ Set.Icc (-1/2) (5*(Real.exp 2)/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l4116_411634


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_l4116_411697

theorem sqrt_six_times_sqrt_three : Real.sqrt 6 * Real.sqrt 3 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_l4116_411697


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x3_prism_l4116_411682

/-- Represents a rectangular prism -/
structure RectangularPrism :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Counts the number of unpainted cubes in a painted rectangular prism -/
def count_unpainted_cubes (prism : RectangularPrism) : ℕ :=
  if prism.height ≤ 2 then 0
  else (prism.length - 2) * (prism.width - 2)

/-- Theorem stating that a 6 × 6 × 3 painted prism has 16 unpainted cubes -/
theorem unpainted_cubes_in_6x6x3_prism :
  let prism : RectangularPrism := ⟨6, 6, 3⟩
  count_unpainted_cubes prism = 16 := by
  sorry

#eval count_unpainted_cubes ⟨6, 6, 3⟩

end NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x3_prism_l4116_411682


namespace NUMINAMATH_CALUDE_max_sum_abs_on_unit_sphere_l4116_411683

theorem max_sum_abs_on_unit_sphere :
  ∃ (M : ℝ), M = Real.sqrt 3 ∧
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |x| + |y| + |z| ≤ M) ∧
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ |x| + |y| + |z| = M) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_abs_on_unit_sphere_l4116_411683


namespace NUMINAMATH_CALUDE_circle_equation_diameter_circle_equation_points_line_l4116_411601

-- Define points and line
def P₁ : ℝ × ℝ := (4, 9)
def P₂ : ℝ × ℝ := (6, 3)
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (-2, -5)
def l (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Theorem for the first circle
theorem circle_equation_diameter (x y : ℝ) : 
  (x - 5)^2 + (y - 6)^2 = 10 ↔ 
  ∃ (t : ℝ), (x, y) = (1 - t) • P₁ + t • P₂ ∧ 0 ≤ t ∧ t ≤ 1 :=
sorry

-- Theorem for the second circle
theorem circle_equation_points_line (x y : ℝ) :
  x^2 + y^2 + 2*x + 4*y - 5 = 0 ↔
  (∃ (cx cy : ℝ), (x - cx)^2 + (y - cy)^2 = ((x - 2)^2 + (y + 3)^2) ∧
                  (x - (-2))^2 + (y - (-5))^2 = ((x - 2)^2 + (y + 3)^2) ∧
                  l cx cy) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_diameter_circle_equation_points_line_l4116_411601


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l4116_411641

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 1 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 3 / 5 →
  n + k = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l4116_411641


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l4116_411648

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a (k + 1) = 3 * a k) →  -- Common ratio is 3
  (∀ k, a k > 0) →  -- Positive terms
  a m * a n = 9 * a 2 ^ 2 →  -- Given condition
  (∀ p q : ℕ, a p * a q = 9 * a 2 ^ 2 → 2 / m + 1 / (2 * n) ≤ 2 / p + 1 / (2 * q)) →
  2 / m + 1 / (2 * n) = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l4116_411648


namespace NUMINAMATH_CALUDE_vector_addition_l4116_411628

theorem vector_addition : 
  let v1 : Fin 2 → ℝ := ![- 5, 3]
  let v2 : Fin 2 → ℝ := ![7, -6]
  v1 + v2 = ![2, -3] :=
by sorry

end NUMINAMATH_CALUDE_vector_addition_l4116_411628


namespace NUMINAMATH_CALUDE_road_vehicles_l4116_411637

/-- Given a road with the specified conditions, prove the total number of vehicles -/
theorem road_vehicles (lanes : Nat) (trucks_per_lane : Nat) (cars_multiplier : Nat) : 
  lanes = 4 → 
  trucks_per_lane = 60 → 
  cars_multiplier = 2 →
  (lanes * trucks_per_lane + lanes * cars_multiplier * lanes * trucks_per_lane) = 2160 := by
  sorry

#check road_vehicles

end NUMINAMATH_CALUDE_road_vehicles_l4116_411637


namespace NUMINAMATH_CALUDE_total_amount_shared_l4116_411622

/-- The total amount shared by A, B, and C given specific conditions -/
theorem total_amount_shared (a b c : ℝ) : 
  a = (1/3) * (b + c) →  -- A gets one-third of what B and C together get
  b = (2/7) * (a + c) →  -- B gets two-sevenths of what A and C together get
  a = b + 35 →           -- A's amount is $35 more than B's amount
  a + b + c = 1260 :=    -- The total amount shared
by sorry

end NUMINAMATH_CALUDE_total_amount_shared_l4116_411622


namespace NUMINAMATH_CALUDE_special_function_value_l4116_411615

/-- A binary function on positive integers satisfying certain properties -/
def special_function (f : ℕ+ → ℕ+ → ℕ+) : Prop :=
  (∀ x, f x x = x) ∧
  (∀ x y, f x y = f y x) ∧
  (∀ x y, (x + y) * (f x y) = y * (f x (x + y)))

/-- Theorem stating that f(12, 16) = 48 for any function satisfying the special properties -/
theorem special_function_value (f : ℕ+ → ℕ+ → ℕ+) (h : special_function f) : 
  f 12 16 = 48 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l4116_411615


namespace NUMINAMATH_CALUDE_pen_sale_profit_percentage_l4116_411629

/-- Calculate the profit percentage for a store owner's pen sale --/
theorem pen_sale_profit_percentage 
  (purchase_quantity : ℕ) 
  (marked_price_quantity : ℕ) 
  (discount_percentage : ℝ) : ℝ :=
by
  -- Assume purchase_quantity = 200
  -- Assume marked_price_quantity = 180
  -- Assume discount_percentage = 2
  
  -- Define cost price
  let cost_price := marked_price_quantity

  -- Define selling price per item
  let selling_price_per_item := 1 - (1 * discount_percentage / 100)

  -- Calculate total revenue
  let total_revenue := purchase_quantity * selling_price_per_item

  -- Calculate profit
  let profit := total_revenue - cost_price

  -- Calculate profit percentage
  let profit_percentage := (profit / cost_price) * 100

  -- Prove that profit_percentage ≈ 8.89
  sorry

-- The statement of the theorem
#check pen_sale_profit_percentage

end NUMINAMATH_CALUDE_pen_sale_profit_percentage_l4116_411629


namespace NUMINAMATH_CALUDE_train_crossing_time_l4116_411690

/-- Proves that a train crossing a platform of its own length in 60 seconds
    will take 30 seconds to cross a signal pole. -/
theorem train_crossing_time (train_length : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ)
  (h1 : train_length = 420)
  (h2 : platform_length = train_length)
  (h3 : platform_crossing_time = 60) :
  train_length / ((train_length + platform_length) / platform_crossing_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l4116_411690


namespace NUMINAMATH_CALUDE_sum_of_fractions_l4116_411627

theorem sum_of_fractions : (1 : ℚ) / 3 + 2 / 9 + 1 / 6 = 13 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l4116_411627


namespace NUMINAMATH_CALUDE_digit_sum_characterization_l4116_411645

/-- Digit sum in base 4038 -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Sequence of distinct positive integers -/
def validSequence (s : ℕ → ℕ) : Prop :=
  (∀ i j, i ≠ j → s i ≠ s j) ∧ (∀ n, s n > 0)

/-- Condition on the sequence growth -/
def boundedByA (s : ℕ → ℕ) (a : ℝ) : Prop :=
  ∀ n, (s n : ℝ) ≤ a * n

/-- Infinitely many terms with digit sum not divisible by 2019 -/
def infinitelyManyNotDivisible (s : ℕ → ℕ) : Prop :=
  ∀ N, ∃ n > N, ¬ 2019 ∣ digitSum (s n)

theorem digit_sum_characterization (a : ℝ) (h : a ≥ 1) :
  (∀ s : ℕ → ℕ, validSequence s → boundedByA s a → infinitelyManyNotDivisible s) ↔
  a < 2019 := by sorry

end NUMINAMATH_CALUDE_digit_sum_characterization_l4116_411645


namespace NUMINAMATH_CALUDE_f_derivative_zero_l4116_411617

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_differentiable : Differentiable ℝ f
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y

-- State the theorem
theorem f_derivative_zero : deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_zero_l4116_411617


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l4116_411686

theorem consecutive_integers_cube_sum (n : ℕ) :
  (n > 0) →
  ((n - 1)^2 + n^2 + (n + 1)^2 = 7805) →
  ((n - 1)^3 + n^3 + (n + 1)^3 = 398259) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l4116_411686


namespace NUMINAMATH_CALUDE_odd_number_proposition_l4116_411646

theorem odd_number_proposition (P : ℕ → Prop) 
  (base : P 1) 
  (step : ∀ k : ℕ, P k → P (k + 2)) : 
  ∀ n : ℕ, Odd n → P n :=
sorry

end NUMINAMATH_CALUDE_odd_number_proposition_l4116_411646


namespace NUMINAMATH_CALUDE_division_properties_l4116_411666

theorem division_properties (a b : ℤ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (¬ (a ∣ b^2 ↔ a ∣ b)) ∧ (a^2 ∣ b^2 ↔ a ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_division_properties_l4116_411666


namespace NUMINAMATH_CALUDE_newspaper_reading_time_l4116_411665

/-- Represents Hank's daily reading habits and total weekly reading time -/
structure ReadingHabits where
  newspaper_time : ℕ  -- Time spent reading newspaper each weekday morning
  novel_time : ℕ      -- Time spent reading novel each weekday evening (60 minutes)
  weekday_count : ℕ   -- Number of weekdays (5)
  weekend_count : ℕ   -- Number of weekend days (2)
  total_time : ℕ      -- Total reading time in a week (810 minutes)

/-- Theorem stating that given Hank's reading habits, he spends 30 minutes reading the newspaper each morning -/
theorem newspaper_reading_time (h : ReadingHabits) 
  (h_novel : h.novel_time = 60)
  (h_weekday : h.weekday_count = 5)
  (h_weekend : h.weekend_count = 2)
  (h_total : h.total_time = 810) :
  h.newspaper_time = 30 := by
  sorry


end NUMINAMATH_CALUDE_newspaper_reading_time_l4116_411665


namespace NUMINAMATH_CALUDE_animal_count_l4116_411675

theorem animal_count (frogs : ℕ) (h1 : frogs = 160) : ∃ (dogs cats : ℕ),
  frogs = 2 * dogs ∧
  cats = dogs - dogs / 5 ∧
  frogs + dogs + cats = 304 := by
sorry

end NUMINAMATH_CALUDE_animal_count_l4116_411675


namespace NUMINAMATH_CALUDE_same_solution_for_k_17_l4116_411689

theorem same_solution_for_k_17 :
  ∃ x : ℝ, (2 * x + 4 = 4 * (x - 2)) ∧ (17 * x - 91 = 2 * x - 1) := by
  sorry

#check same_solution_for_k_17

end NUMINAMATH_CALUDE_same_solution_for_k_17_l4116_411689


namespace NUMINAMATH_CALUDE_dolphin_population_estimate_l4116_411654

/-- Estimate the number of dolphins in a coastal area on January 1st -/
theorem dolphin_population_estimate (tagged_initial : ℕ) (captured_june : ℕ) (tagged_june : ℕ)
  (migration_rate : ℚ) (new_arrival_rate : ℚ) :
  tagged_initial = 100 →
  captured_june = 90 →
  tagged_june = 4 →
  migration_rate = 1/5 →
  new_arrival_rate = 1/2 →
  ∃ (initial_population : ℕ), initial_population = 1125 :=
by sorry

end NUMINAMATH_CALUDE_dolphin_population_estimate_l4116_411654
