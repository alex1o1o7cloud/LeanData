import Mathlib

namespace NUMINAMATH_CALUDE_sine_sum_gt_cosine_sum_in_acute_triangle_l2058_205838

/-- In any acute-angled triangle ABC, the sum of the sines of its angles is greater than the sum of the cosines of its angles. -/
theorem sine_sum_gt_cosine_sum_in_acute_triangle (A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_triangle : A + B + C = π) : 
  Real.sin A + Real.sin B + Real.sin C > Real.cos A + Real.cos B + Real.cos C := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_gt_cosine_sum_in_acute_triangle_l2058_205838


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l2058_205890

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_fraction :
  (3 - 4 * i) / (5 - 2 * i) = 7 / 29 - (14 / 29) * i :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l2058_205890


namespace NUMINAMATH_CALUDE_order_of_products_l2058_205896

theorem order_of_products (m n : ℝ) (hm : m < 0) (hn : -1 < n ∧ n < 0) :
  m < m * n^2 ∧ m * n^2 < m * n := by sorry

end NUMINAMATH_CALUDE_order_of_products_l2058_205896


namespace NUMINAMATH_CALUDE_typists_productivity_l2058_205807

/-- Given that 25 typists can type 60 letters in 20 minutes, prove that 75 typists 
    working at the same rate can complete 540 letters in 1 hour. -/
theorem typists_productivity (typists_base : ℕ) (letters_base : ℕ) (minutes_base : ℕ) 
  (typists_new : ℕ) (minutes_new : ℕ) :
  typists_base = 25 →
  letters_base = 60 →
  minutes_base = 20 →
  typists_new = 75 →
  minutes_new = 60 →
  (typists_new * letters_base * minutes_new) / (typists_base * minutes_base) = 540 :=
by sorry

end NUMINAMATH_CALUDE_typists_productivity_l2058_205807


namespace NUMINAMATH_CALUDE_intersection_count_theorem_l2058_205872

/-- Given two perpendicular lines L1 and L2, with n points on L1 and m points on L2,
    calculates the maximum number of intersection points when every point on L1
    is connected to every point on L2 by a line segment. -/
def max_intersections (n m : ℕ) : ℕ :=
  (n.choose 2) * (m.choose 2)

/-- Theorem stating that with 8 points on one line and 6 points on a perpendicular line,
    the maximum number of intersection points is 420 when all points are connected. -/
theorem intersection_count_theorem :
  max_intersections 8 6 = 420 := by sorry

end NUMINAMATH_CALUDE_intersection_count_theorem_l2058_205872


namespace NUMINAMATH_CALUDE_f_of_3_equals_4_l2058_205883

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2

-- Theorem statement
theorem f_of_3_equals_4 : f 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_4_l2058_205883


namespace NUMINAMATH_CALUDE_calculator_square_presses_l2058_205899

def square (x : ℕ) : ℕ := x * x

def exceed_1000 (n : ℕ) : Prop := n > 1000

theorem calculator_square_presses :
  (∃ k : ℕ, exceed_1000 (square (square (square 3)))) ∧
  (∀ m : ℕ, m < 3 → ¬exceed_1000 (Nat.iterate square 3 m)) :=
by sorry

end NUMINAMATH_CALUDE_calculator_square_presses_l2058_205899


namespace NUMINAMATH_CALUDE_investment_result_l2058_205867

/-- Given a total investment split between two interest rates, calculates the total investment with interest after one year. -/
def total_investment_with_interest (total_investment : ℝ) (amount_at_low_rate : ℝ) (low_rate : ℝ) (high_rate : ℝ) : ℝ :=
  let amount_at_high_rate := total_investment - amount_at_low_rate
  let interest_low := amount_at_low_rate * low_rate
  let interest_high := amount_at_high_rate * high_rate
  total_investment + interest_low + interest_high

/-- Theorem stating that given the specific investment conditions, the total investment with interest is $1,046.00 -/
theorem investment_result : 
  let total_investment := 1000
  let amount_at_low_rate := 699.99
  let low_rate := 0.04
  let high_rate := 0.06
  (total_investment_with_interest total_investment amount_at_low_rate low_rate high_rate) = 1046 := by
sorry

end NUMINAMATH_CALUDE_investment_result_l2058_205867


namespace NUMINAMATH_CALUDE_symmetric_circle_correct_l2058_205865

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 11 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-2, 1)

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 5)^2 + y^2 = 16

-- Theorem statement
theorem symmetric_circle_correct :
  ∀ (x y : ℝ),
  symmetric_circle x y ↔
  ∃ (x₀ y₀ : ℝ),
    original_circle x₀ y₀ ∧
    x = 2 * point_P.1 - x₀ ∧
    y = 2 * point_P.2 - y₀ :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_correct_l2058_205865


namespace NUMINAMATH_CALUDE_product_sum_inequality_l2058_205803

theorem product_sum_inequality (a b c x y z : ℝ) 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) : 
  a * x + b * y + c * z ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l2058_205803


namespace NUMINAMATH_CALUDE_solve_for_d_l2058_205878

theorem solve_for_d (n c b d : ℝ) (h : n = (d * c * b) / (c - d)) :
  d = (n * c) / (c * b + n) :=
by sorry

end NUMINAMATH_CALUDE_solve_for_d_l2058_205878


namespace NUMINAMATH_CALUDE_total_earnings_proof_l2058_205812

/-- Calculates the total earnings for a three-day fundraiser car wash activity. -/
def total_earnings (friday_earnings : ℕ) : ℕ :=
  let saturday_earnings := 2 * friday_earnings + 7
  let sunday_earnings := friday_earnings + 78
  friday_earnings + saturday_earnings + sunday_earnings

/-- Proves that the total earnings over three days is 673, given the specified conditions. -/
theorem total_earnings_proof :
  total_earnings 147 = 673 := by
  sorry

#eval total_earnings 147

end NUMINAMATH_CALUDE_total_earnings_proof_l2058_205812


namespace NUMINAMATH_CALUDE_restaurant_order_combinations_l2058_205804

theorem restaurant_order_combinations :
  let main_dish_options : ℕ := 12
  let side_dish_options : ℕ := 5
  let person_count : ℕ := 2
  main_dish_options ^ person_count * side_dish_options = 720 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_order_combinations_l2058_205804


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2058_205879

-- Define the equations
def equation1 (x : ℝ) : Prop := x - 2 * Real.sqrt x + 1 = 0
def equation2 (x : ℝ) : Prop := x + 2 + Real.sqrt (x + 2) = 0

-- Theorem for the first equation
theorem solution_equation1 : ∃ (x : ℝ), equation1 x ∧ x = 1 :=
  sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ (x : ℝ), equation2 x ∧ x = -2 :=
  sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2058_205879


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2058_205898

theorem fractional_equation_solution :
  ∃! x : ℝ, x ≠ 3 ∧ (1 - x) / (x - 3) = 1 / (3 - x) - 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2058_205898


namespace NUMINAMATH_CALUDE_min_horizontal_distance_l2058_205823

def f (x : ℝ) := x^3 - x^2 - x - 6

theorem min_horizontal_distance :
  ∃ (x1 x2 : ℝ),
    f x1 = 8 ∧
    f x2 = -8 ∧
    ∀ (y1 y2 : ℝ),
      f y1 = 8 → f y2 = -8 →
      |x1 - x2| ≤ |y1 - y2| ∧
      |x1 - x2| = 1 :=
sorry

end NUMINAMATH_CALUDE_min_horizontal_distance_l2058_205823


namespace NUMINAMATH_CALUDE_min_people_for_tests_l2058_205845

/-- The minimum number of people required to achieve the given score ranges -/
def min_people (ranges : List ℕ) (min_range : ℕ) : ℕ :=
  if ranges.maximum = some min_range then 2 else 1

/-- Theorem: Given the conditions, at least 2 people took the tests -/
theorem min_people_for_tests : min_people [17, 28, 35, 45] 45 = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_people_for_tests_l2058_205845


namespace NUMINAMATH_CALUDE_x_minus_y_value_l2058_205893

theorem x_minus_y_value (x y : ℝ) (h : |x + 2| + (y - 3)^2 = 0) : x - y = -5 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l2058_205893


namespace NUMINAMATH_CALUDE_sequence_properties_l2058_205830

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom a_1 : sequence_a 1 = 1

axiom a_relation (n : ℕ) : n > 0 → 2 * sequence_a (n + 1) + S n - 2 = 0

def sequence_b (n : ℕ) : ℝ := n * sequence_a n

def T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sequence_a n = (1/2)^(n-1)) ∧
  (∀ n : ℕ, n > 0 → T n = 4 - (n + 2) * (1/2)^(n-1)) := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2058_205830


namespace NUMINAMATH_CALUDE_special_sequence_a11_l2058_205849

/-- A sequence satisfying the given conditions -/
def SpecialSequence (a : ℕ+ → ℤ) : Prop :=
  (∀ p q : ℕ+, a (p + q) = a p + a q) ∧ (a 2 = -6)

/-- The theorem statement -/
theorem special_sequence_a11 (a : ℕ+ → ℤ) (h : SpecialSequence a) : a 11 = -33 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_a11_l2058_205849


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2058_205869

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 2 * x / (x - 2) + (2 * x^2 - 24) / x - 11
  ∃ (y : ℝ), y = (1 - Real.sqrt 65) / 4 ∧ f y = 0 ∧ ∀ (z : ℝ), f z = 0 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2058_205869


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_plus_i_l2058_205829

theorem imaginary_part_of_one_over_one_plus_i :
  let z : ℂ := 1 / (1 + Complex.I)
  Complex.im z = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_plus_i_l2058_205829


namespace NUMINAMATH_CALUDE_absolute_value_problem_l2058_205801

theorem absolute_value_problem (a b : ℝ) (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a + b| = 4) :
  ∃ (x : ℝ), |a - b| = x :=
sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l2058_205801


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2058_205806

/-- An ellipse with foci at (-3, 0) and (3, 0), passing through (0, 3) -/
structure Ellipse where
  /-- The equation of the ellipse in the form (x²/a² + y²/b² = 1) -/
  equation : ℝ → ℝ → Prop
  /-- The foci are at (-3, 0) and (3, 0) -/
  foci : equation (-3) 0 ∧ equation 3 0
  /-- The point (0, 3) is on the ellipse -/
  point : equation 0 3

/-- The standard form of the ellipse equation -/
def standard_equation (x y : ℝ) : Prop :=
  x^2 / 18 + y^2 / 9 = 1

/-- Theorem: The standard equation of the ellipse is x²/18 + y²/9 = 1 -/
theorem ellipse_standard_equation (e : Ellipse) : e.equation = standard_equation := by
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l2058_205806


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2058_205805

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    with at least one box remaining empty -/
def distributeWithEmptyBox (n k : ℕ) : ℕ :=
  if n < k then distribute n k else distribute n k

theorem distribute_five_balls_four_boxes :
  distributeWithEmptyBox 5 4 = 1024 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2058_205805


namespace NUMINAMATH_CALUDE_complex_multiplication_l2058_205860

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (3 + 4*i) = -4 + 3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2058_205860


namespace NUMINAMATH_CALUDE_product_evaluation_l2058_205813

theorem product_evaluation :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) = 7^128 - 5^128 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2058_205813


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2058_205824

theorem sum_of_roots_quadratic : ∀ (a b c : ℝ), a ≠ 0 → 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  (∃ s : ℝ, s = -(b / a) ∧ s = 7) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2058_205824


namespace NUMINAMATH_CALUDE_binary_101011_equals_43_l2058_205864

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101011_equals_43 :
  binary_to_decimal [true, true, false, true, false, true] = 43 := by
  sorry

end NUMINAMATH_CALUDE_binary_101011_equals_43_l2058_205864


namespace NUMINAMATH_CALUDE_shelter_new_pets_l2058_205880

theorem shelter_new_pets (initial_dogs : ℕ) (initial_cats : ℕ) (initial_lizards : ℕ)
  (dog_adoption_rate : ℚ) (cat_adoption_rate : ℚ) (lizard_adoption_rate : ℚ)
  (pets_after_month : ℕ) :
  initial_dogs = 30 →
  initial_cats = 28 →
  initial_lizards = 20 →
  dog_adoption_rate = 1/2 →
  cat_adoption_rate = 1/4 →
  lizard_adoption_rate = 1/5 →
  pets_after_month = 65 →
  ∃ new_pets : ℕ,
    new_pets = 13 ∧
    pets_after_month = 
      (initial_dogs - initial_dogs * dog_adoption_rate).floor +
      (initial_cats - initial_cats * cat_adoption_rate).floor +
      (initial_lizards - initial_lizards * lizard_adoption_rate).floor +
      new_pets :=
by
  sorry

end NUMINAMATH_CALUDE_shelter_new_pets_l2058_205880


namespace NUMINAMATH_CALUDE_function_value_at_pi_over_12_l2058_205818

theorem function_value_at_pi_over_12 (x : Real) (h : x = π / 12) :
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2) - 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_pi_over_12_l2058_205818


namespace NUMINAMATH_CALUDE_jessicas_class_farm_trip_cost_l2058_205894

/-- Calculate the total cost for a field trip to a farm -/
def farm_trip_cost (num_students : ℕ) (num_adults : ℕ) (student_fee : ℕ) (adult_fee : ℕ) : ℕ :=
  num_students * student_fee + num_adults * adult_fee

/-- Theorem: The total cost for Jessica's class field trip to the farm is $199 -/
theorem jessicas_class_farm_trip_cost : farm_trip_cost 35 4 5 6 = 199 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_class_farm_trip_cost_l2058_205894


namespace NUMINAMATH_CALUDE_little_john_initial_money_l2058_205851

theorem little_john_initial_money :
  let sweets_cost : ℚ := 1.25
  let friends_count : ℕ := 2
  let money_per_friend : ℚ := 1.20
  let money_left : ℚ := 4.85
  let initial_money : ℚ := sweets_cost + friends_count * money_per_friend + money_left
  initial_money = 8.50 := by sorry

end NUMINAMATH_CALUDE_little_john_initial_money_l2058_205851


namespace NUMINAMATH_CALUDE_john_spent_15_dollars_l2058_205877

def price_per_dozen : ℕ := 5
def rolls_bought : ℕ := 36

theorem john_spent_15_dollars : 
  (rolls_bought / 12) * price_per_dozen = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_spent_15_dollars_l2058_205877


namespace NUMINAMATH_CALUDE_find_d_l2058_205889

theorem find_d (a b c d : ℕ+) 
  (eq1 : a ^ 2 = c * (d + 20))
  (eq2 : b ^ 2 = c * (d - 18)) :
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_d_l2058_205889


namespace NUMINAMATH_CALUDE_two_non_congruent_triangles_l2058_205802

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if two triangles are congruent -/
def is_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all triangles with perimeter 7 -/
def triangles_with_perimeter_7 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 7}

/-- The theorem to be proved -/
theorem two_non_congruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ triangles_with_perimeter_7 ∧
    t2 ∈ triangles_with_perimeter_7 ∧
    ¬ is_congruent t1 t2 ∧
    ∀ (t : IntTriangle),
      t ∈ triangles_with_perimeter_7 →
      is_congruent t t1 ∨ is_congruent t t2 :=
sorry

end NUMINAMATH_CALUDE_two_non_congruent_triangles_l2058_205802


namespace NUMINAMATH_CALUDE_orchestra_seat_price_l2058_205815

/-- Represents the theater ticket sales scenario --/
structure TheaterSales where
  orchestra_price : ℕ
  balcony_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ
  balcony_orchestra_diff : ℕ

/-- Theorem stating the orchestra seat price given the conditions --/
theorem orchestra_seat_price (ts : TheaterSales)
  (h1 : ts.balcony_price = 8)
  (h2 : ts.total_tickets = 340)
  (h3 : ts.total_revenue = 3320)
  (h4 : ts.balcony_orchestra_diff = 40) :
  ts.orchestra_price = 12 := by
  sorry


end NUMINAMATH_CALUDE_orchestra_seat_price_l2058_205815


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2058_205819

-- Define the universal set I
def I : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 3, 5}

-- Define set B
def B : Set Nat := {2, 3, 6}

-- Theorem statement
theorem complement_intersection_theorem :
  (I \ A) ∩ B = {2, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2058_205819


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l2058_205853

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 10 meters and height 7 meters is 70 square meters -/
theorem parallelogram_area_example : parallelogram_area 10 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l2058_205853


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2058_205800

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  c = 2 →
  Real.sin C * (Real.cos B - Real.sqrt 3 * Real.sin B) = Real.sin A →
  Real.cos A = 2 * Real.sqrt 2 / 3 →
  -- Conclusions
  C = 5 * π / 6 ∧
  b = (4 * Real.sqrt 2 - 2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2058_205800


namespace NUMINAMATH_CALUDE_johnsRemainingMoneyTheorem_l2058_205808

/-- The amount of money John has left after purchasing pizzas and drinks -/
def johnsRemainingMoney (d : ℝ) : ℝ :=
  let drinkCost := d
  let mediumPizzaCost := 3 * d
  let largePizzaCost := 4 * d
  let totalCost := 5 * drinkCost + mediumPizzaCost + 2 * largePizzaCost
  50 - totalCost

/-- Theorem stating that John's remaining money is 50 - 16d -/
theorem johnsRemainingMoneyTheorem (d : ℝ) :
  johnsRemainingMoney d = 50 - 16 * d :=
by sorry

end NUMINAMATH_CALUDE_johnsRemainingMoneyTheorem_l2058_205808


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2058_205861

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x > -a) ↔ a > -3/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2058_205861


namespace NUMINAMATH_CALUDE_wire_length_ratio_l2058_205837

theorem wire_length_ratio :
  let edge_length : ℕ := 8
  let large_cube_wire_length : ℕ := 12 * edge_length
  let large_cube_volume : ℕ := edge_length ^ 3
  let unit_cube_wire_length : ℕ := 12
  let total_unit_cubes : ℕ := large_cube_volume
  let total_unit_cube_wire_length : ℕ := total_unit_cubes * unit_cube_wire_length
  (large_cube_wire_length : ℚ) / total_unit_cube_wire_length = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l2058_205837


namespace NUMINAMATH_CALUDE_custard_pie_problem_l2058_205873

theorem custard_pie_problem (price_per_slice : ℚ) (slices_per_pie : ℕ) (total_revenue : ℚ) :
  price_per_slice = 3 →
  slices_per_pie = 10 →
  total_revenue = 180 →
  (total_revenue / (price_per_slice * slices_per_pie : ℚ)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_custard_pie_problem_l2058_205873


namespace NUMINAMATH_CALUDE_max_prime_factor_of_arithmetic_sequence_number_l2058_205875

/-- A 3-digit decimal number with digits forming an arithmetic sequence -/
def ArithmeticSequenceNumber (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = 100 * a + 10 * b + c ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    b = a + d ∧
    c = a + 2 * d

theorem max_prime_factor_of_arithmetic_sequence_number :
  ∀ n : ℕ, ArithmeticSequenceNumber n →
    (∀ p : ℕ, Nat.Prime p → p ∣ n → p ≤ 317) ∧
    (∃ m : ℕ, ArithmeticSequenceNumber m ∧ ∃ p : ℕ, Nat.Prime p ∧ p ∣ m ∧ p = 317) :=
by sorry

end NUMINAMATH_CALUDE_max_prime_factor_of_arithmetic_sequence_number_l2058_205875


namespace NUMINAMATH_CALUDE_problem_solution_l2058_205891

theorem problem_solution (x y z : ℚ) : 
  x = 2/3 → y = 3/2 → z = 1/3 → (1/3) * x^7 * y^5 * z^4 = 11/600 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2058_205891


namespace NUMINAMATH_CALUDE_total_earnings_is_4350_l2058_205825

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  invest_ratio_A : ℕ
  invest_ratio_B : ℕ
  invest_ratio_C : ℕ
  return_ratio_A : ℕ
  return_ratio_B : ℕ
  return_ratio_C : ℕ

/-- Calculates the total earnings given investment data and the earnings difference between B and A -/
def calculate_total_earnings (data : InvestmentData) (earnings_diff_B_A : ℕ) : ℕ :=
  let earnings_A := data.invest_ratio_A * data.return_ratio_A
  let earnings_B := data.invest_ratio_B * data.return_ratio_B
  let earnings_C := data.invest_ratio_C * data.return_ratio_C
  let total_ratio := earnings_A + earnings_B + earnings_C
  (total_ratio * earnings_diff_B_A) / (earnings_B - earnings_A)

/-- Theorem stating that given the specific investment ratios and conditions, the total earnings is 4350 -/
theorem total_earnings_is_4350 : 
  let data : InvestmentData := {
    invest_ratio_A := 3,
    invest_ratio_B := 4,
    invest_ratio_C := 5,
    return_ratio_A := 6,
    return_ratio_B := 5,
    return_ratio_C := 4
  }
  calculate_total_earnings data 150 = 4350 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_is_4350_l2058_205825


namespace NUMINAMATH_CALUDE_no_covalent_bond_IA_VIIA_l2058_205892

/-- Represents an element in the periodic table -/
structure Element where
  group : ℕ
  isHydrogen : Bool

/-- Represents the bonding behavior of an element -/
inductive BondingBehavior
  | LoseElectrons
  | GainElectrons

/-- Determines the bonding behavior of an element based on its group -/
def bondingBehavior (e : Element) : BondingBehavior :=
  if e.group = 1 ∧ ¬e.isHydrogen then BondingBehavior.LoseElectrons
  else if e.group = 17 then BondingBehavior.GainElectrons
  else BondingBehavior.LoseElectrons  -- Default case, not relevant for this problem

/-- Determines if two elements can form a covalent bond -/
def canFormCovalentBond (e1 e2 : Element) : Prop :=
  bondingBehavior e1 = bondingBehavior e2

/-- Theorem stating that elements in Group IA (except H) and Group VIIA cannot form covalent bonds -/
theorem no_covalent_bond_IA_VIIA :
  ∀ (e1 e2 : Element),
    ((e1.group = 1 ∧ ¬e1.isHydrogen) ∨ e1.group = 17) →
    ((e2.group = 1 ∧ ¬e2.isHydrogen) ∨ e2.group = 17) →
    ¬(canFormCovalentBond e1 e2) :=
by
  sorry

end NUMINAMATH_CALUDE_no_covalent_bond_IA_VIIA_l2058_205892


namespace NUMINAMATH_CALUDE_power_product_squared_l2058_205831

theorem power_product_squared (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l2058_205831


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l2058_205811

theorem reciprocal_sum_of_roots (γ δ : ℝ) : 
  (∃ r s : ℝ, 7 * r^2 + 4 * r + 9 = 0 ∧ 
              7 * s^2 + 4 * s + 9 = 0 ∧ 
              γ = 1 / r ∧ 
              δ = 1 / s) → 
  γ + δ = -4/9 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l2058_205811


namespace NUMINAMATH_CALUDE_navigation_time_is_21_days_l2058_205854

/-- Represents the timeline of a cargo shipment from Shanghai to Vancouver --/
structure CargoShipment where
  /-- Number of days for the ship to navigate from Shanghai to Vancouver --/
  navigationDays : ℕ
  /-- Number of days for customs and regulatory processes in Vancouver --/
  customsDays : ℕ
  /-- Number of days from port to warehouse --/
  portToWarehouseDays : ℕ
  /-- Number of days since the ship departed --/
  daysSinceDeparture : ℕ
  /-- Number of days until expected arrival at the warehouse --/
  daysUntilArrival : ℕ

/-- The theorem stating that the navigation time is 21 days --/
theorem navigation_time_is_21_days (shipment : CargoShipment)
  (h1 : shipment.customsDays = 4)
  (h2 : shipment.portToWarehouseDays = 7)
  (h3 : shipment.daysSinceDeparture = 30)
  (h4 : shipment.daysUntilArrival = 2)
  (h5 : shipment.navigationDays + shipment.customsDays + shipment.portToWarehouseDays =
        shipment.daysSinceDeparture + shipment.daysUntilArrival) :
  shipment.navigationDays = 21 := by
  sorry

end NUMINAMATH_CALUDE_navigation_time_is_21_days_l2058_205854


namespace NUMINAMATH_CALUDE_root_of_f_equals_two_max_m_for_inequality_ab_equals_one_l2058_205840

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := a^x + b^x

-- Define the conditions on a and b
class PositiveNotOne (r : ℝ) : Prop where
  pos : r > 0
  not_one : r ≠ 1

-- Theorem 1a
theorem root_of_f_equals_two 
  (h₁ : PositiveNotOne 2) 
  (h₂ : PositiveNotOne (1/2)) :
  ∃ x : ℝ, f 2 (1/2) x = 2 ∧ x = 0 := by sorry

-- Theorem 1b
theorem max_m_for_inequality 
  (h₁ : PositiveNotOne 2) 
  (h₂ : PositiveNotOne (1/2)) :
  ∃ m : ℝ, (∀ x : ℝ, f 2 (1/2) (2*x) ≥ m * f 2 (1/2) x - 6) ∧ 
  (∀ m' : ℝ, (∀ x : ℝ, f 2 (1/2) (2*x) ≥ m' * f 2 (1/2) x - 6) → m' ≤ m) ∧
  m = 4 := by sorry

-- Define function g
def g (a b x : ℝ) : ℝ := f a b x - 2

-- Theorem 2
theorem ab_equals_one 
  (ha : 0 < a ∧ a < 1) 
  (hb : b > 1) 
  (h : PositiveNotOne a) 
  (h' : PositiveNotOne b) 
  (hg : ∃! x : ℝ, g a b x = 0) :
  a * b = 1 := by sorry

end NUMINAMATH_CALUDE_root_of_f_equals_two_max_m_for_inequality_ab_equals_one_l2058_205840


namespace NUMINAMATH_CALUDE_multiply_b_equals_four_l2058_205843

theorem multiply_b_equals_four (a b x : ℝ) 
  (h1 : 3 * a = x * b) 
  (h2 : a * b ≠ 0) 
  (h3 : a / 4 = b / 3) : 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_multiply_b_equals_four_l2058_205843


namespace NUMINAMATH_CALUDE_cycle_original_price_l2058_205856

/-- Given a cycle sold for Rs. 1080 with a gain of 8%, prove that the original price was Rs. 1000 -/
theorem cycle_original_price (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 1080)
  (h2 : gain_percentage = 8) :
  let original_price := selling_price / (1 + gain_percentage / 100)
  original_price = 1000 := by sorry

end NUMINAMATH_CALUDE_cycle_original_price_l2058_205856


namespace NUMINAMATH_CALUDE_chemistry_marks_proof_l2058_205834

def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 82
def biology_marks : ℕ := 85
def average_marks : ℕ := 74
def total_subjects : ℕ := 5

def chemistry_marks : ℕ := 67

theorem chemistry_marks_proof :
  chemistry_marks = total_subjects * average_marks - (english_marks + math_marks + physics_marks + biology_marks) :=
by sorry

end NUMINAMATH_CALUDE_chemistry_marks_proof_l2058_205834


namespace NUMINAMATH_CALUDE_unique_power_of_two_product_l2058_205827

theorem unique_power_of_two_product (a b : ℕ) :
  (∃ k : ℕ, (a + b^2) * (b + a^2) = 2^k) ↔ (a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_power_of_two_product_l2058_205827


namespace NUMINAMATH_CALUDE_base_3_division_theorem_l2058_205887

def base_3_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, d) acc => acc + d * (3 ^ i)) 0

def decimal_to_base_3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
  aux n []

theorem base_3_division_theorem :
  let dividend := [1, 0, 2, 1]  -- 1021₃ in reverse order
  let divisor := [1, 1]         -- 11₃ in reverse order
  let quotient := [2, 2]        -- 22₃ in reverse order
  (base_3_to_decimal dividend) / (base_3_to_decimal divisor) = base_3_to_decimal quotient :=
by sorry

end NUMINAMATH_CALUDE_base_3_division_theorem_l2058_205887


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2058_205809

/-- Given a principal amount P and a time period of 10 years,
    prove that the rate of simple interest is 6% per annum
    when the simple interest is 3/5 of the principal amount. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) :
  let SI := (3/5) * P  -- Simple interest is 3/5 of principal
  let T := 10  -- Time period in years
  let r := 6  -- Rate percent per annum
  SI = (P * r * T) / 100  -- Simple interest formula
  := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2058_205809


namespace NUMINAMATH_CALUDE_surjective_sum_iff_constant_l2058_205847

/-- A function is surjective if every element in the codomain is mapped to by at least one element in the domain. -/
def Surjective (f : ℤ → ℤ) : Prop :=
  ∀ y : ℤ, ∃ x : ℤ, f x = y

/-- The sum of two functions -/
def FunctionSum (f g : ℤ → ℤ) : ℤ → ℤ := λ x => f x + g x

/-- A function is constant if it maps all inputs to the same output -/
def ConstantFunction (f : ℤ → ℤ) : Prop :=
  ∃ c : ℤ, ∀ x : ℤ, f x = c

/-- The main theorem: a function f preserves surjectivity of g when added to it
    if and only if f is constant -/
theorem surjective_sum_iff_constant (f : ℤ → ℤ) :
  (∀ g : ℤ → ℤ, Surjective g → Surjective (FunctionSum f g)) ↔ ConstantFunction f :=
sorry

end NUMINAMATH_CALUDE_surjective_sum_iff_constant_l2058_205847


namespace NUMINAMATH_CALUDE_arithmetic_sequence_5_to_119_l2058_205836

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Proof that the arithmetic sequence from 5 to 119 with common difference 3 has 39 terms -/
theorem arithmetic_sequence_5_to_119 :
  arithmeticSequenceLength 5 119 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_5_to_119_l2058_205836


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2058_205846

theorem complex_power_magnitude (z : ℂ) : z = (1 / 2 : ℂ) + (Complex.I * (Real.sqrt 3 / 2)) → Complex.abs (z ^ 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2058_205846


namespace NUMINAMATH_CALUDE_equal_ratios_imply_p_equals_13_l2058_205814

theorem equal_ratios_imply_p_equals_13 
  (a b c p : ℝ) 
  (h1 : (5 : ℝ) / (a + b) = p / (a + c)) 
  (h2 : p / (a + c) = (8 : ℝ) / (c - b)) : 
  p = 13 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_imply_p_equals_13_l2058_205814


namespace NUMINAMATH_CALUDE_min_value_of_f_l2058_205857

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 5) / (2*x - 4)

theorem min_value_of_f (x : ℝ) (h : x ≥ 5/2) : f x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2058_205857


namespace NUMINAMATH_CALUDE_range_of_difference_l2058_205833

theorem range_of_difference (a b : ℝ) (ha : 12 < a ∧ a < 60) (hb : 15 < b ∧ b < 36) :
  -24 < a - b ∧ a - b < 45 := by
  sorry

end NUMINAMATH_CALUDE_range_of_difference_l2058_205833


namespace NUMINAMATH_CALUDE_correct_factorization_l2058_205848

theorem correct_factorization (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

#check correct_factorization

end NUMINAMATH_CALUDE_correct_factorization_l2058_205848


namespace NUMINAMATH_CALUDE_lottery_probabilities_l2058_205859

/-- Represents the outcome of a customer's lottery participation -/
inductive LotteryResult
  | Gold
  | Silver
  | NoWin

/-- Models the lottery promotion scenario -/
structure LotteryPromotion where
  totalTickets : Nat
  surveySize : Nat
  noWinRatio : Rat
  silverRatioAmongWinners : Rat

/-- Calculates the probability of at least one gold prize winner among 3 randomly selected customers -/
def probAtLeastOneGold (lp : LotteryPromotion) : Rat :=
  sorry

/-- Calculates the probability that the number of gold prize winners is not more than 
    the number of silver prize winners among 3 randomly selected customers -/
def probGoldNotMoreThanSilver (lp : LotteryPromotion) : Rat :=
  sorry

/-- The main theorem stating the probabilities for the given lottery promotion scenario -/
theorem lottery_probabilities (lp : LotteryPromotion) 
  (h1 : lp.totalTickets = 2000)
  (h2 : lp.surveySize = 30)
  (h3 : lp.noWinRatio = 2/3)
  (h4 : lp.silverRatioAmongWinners = 3/5) :
  probAtLeastOneGold lp = 73/203 ∧ 
  probGoldNotMoreThanSilver lp = 157/203 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probabilities_l2058_205859


namespace NUMINAMATH_CALUDE_projection_matrix_values_l2058_205826

/-- A 2x2 matrix is a projection matrix if and only if P² = P -/
def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P ^ 2 = P

/-- The specific 2x2 matrix we're working with -/
def P (b d : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![b, 12/25; d, 13/25]

/-- The theorem stating the values of b and d for the projection matrix -/
theorem projection_matrix_values :
  ∀ b d : ℚ, is_projection_matrix (P b d) → b = 37/50 ∧ d = 19/50 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l2058_205826


namespace NUMINAMATH_CALUDE_symmetric_sine_extreme_value_l2058_205876

/-- Given a function f(x) = 2sin(ωx + φ) that satisfies f(π/4 + x) = f(π/4 - x) for all x,
    prove that f(π/4) equals either 2 or -2. -/
theorem symmetric_sine_extreme_value 
  (ω φ : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * Real.sin (ω * x + φ)) 
  (h2 : ∀ x, f (π/4 + x) = f (π/4 - x)) : 
  f (π/4) = 2 ∨ f (π/4) = -2 :=
sorry

end NUMINAMATH_CALUDE_symmetric_sine_extreme_value_l2058_205876


namespace NUMINAMATH_CALUDE_onions_count_prove_onions_count_l2058_205850

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onion_difference : ℕ := 5200

theorem onions_count : ℕ :=
  (tomatoes + corn) - onion_difference

theorem prove_onions_count : onions_count = 985 := by
  sorry

end NUMINAMATH_CALUDE_onions_count_prove_onions_count_l2058_205850


namespace NUMINAMATH_CALUDE_common_difference_is_five_l2058_205895

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

/-- Theorem: The common difference is 5 given the conditions -/
theorem common_difference_is_five (seq : ArithmeticSequence)
  (h1 : seq.S 17 = 255)
  (h2 : seq.a 10 = 20) :
  commonDifference seq = 5 := by
  sorry

#check common_difference_is_five

end NUMINAMATH_CALUDE_common_difference_is_five_l2058_205895


namespace NUMINAMATH_CALUDE_potatoes_left_l2058_205866

theorem potatoes_left (initial : ℕ) (salad : ℕ) (mashed : ℕ) (h1 : initial = 52) (h2 : salad = 15) (h3 : mashed = 24) : initial - (salad + mashed) = 13 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_left_l2058_205866


namespace NUMINAMATH_CALUDE_max_value_of_f_l2058_205817

def f (x : ℝ) := x^2 + 2*x

theorem max_value_of_f :
  ∃ (M : ℝ), M = 8 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2058_205817


namespace NUMINAMATH_CALUDE_no_geometric_progression_with_1_2_5_l2058_205882

theorem no_geometric_progression_with_1_2_5 :
  ¬ ∃ (a q : ℝ) (m n p : ℕ), 
    m ≠ n ∧ n ≠ p ∧ m ≠ p ∧
    a * q^m = 1 ∧ a * q^n = 2 ∧ a * q^p = 5 :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_progression_with_1_2_5_l2058_205882


namespace NUMINAMATH_CALUDE_minimize_distance_sum_l2058_205844

/-- The point that minimizes the sum of distances to two fixed points in a plane lies on the line segment connecting those points. -/
axiom distance_minimizing_point_on_segment {A B C : ℝ × ℝ} :
  (∀ D : ℝ × ℝ, dist A C + dist B C ≤ dist A D + dist B D) →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (t • A.1 + (1 - t) • B.1, t • A.2 + (1 - t) • B.2)

/-- The theorem stating that the point C(k, 0) minimizing the sum of distances to A(7, 4) and B(3, -2) has k = 5. -/
theorem minimize_distance_sum : 
  let A : ℝ × ℝ := (7, 4)
  let B : ℝ × ℝ := (3, -2)
  let C : ℝ → ℝ × ℝ := λ k ↦ (k, 0)
  ∃ k : ℝ, k = 5 ∧ ∀ x : ℝ, dist A (C k) + dist B (C k) ≤ dist A (C x) + dist B (C x) :=
sorry

end NUMINAMATH_CALUDE_minimize_distance_sum_l2058_205844


namespace NUMINAMATH_CALUDE_population_reaches_limit_l2058_205888

/-- The number of years it takes for the population to reach or exceed the sustainable limit -/
def years_to_sustainable_limit : ℕ :=
  -- We'll define this later in the proof
  210

/-- The amount of acres required per person for sustainable living -/
def acres_per_person : ℕ := 2

/-- The total available acres for human habitation -/
def total_acres : ℕ := 35000

/-- The initial population in 2005 -/
def initial_population : ℕ := 150

/-- The number of years it takes for the population to double -/
def years_to_double : ℕ := 30

/-- The maximum sustainable population -/
def max_sustainable_population : ℕ := total_acres / acres_per_person

/-- The population after a given number of years -/
def population_after_years (years : ℕ) : ℕ :=
  initial_population * (2 ^ (years / years_to_double))

/-- Theorem stating that the population reaches or exceeds the sustainable limit in the specified number of years -/
theorem population_reaches_limit :
  population_after_years years_to_sustainable_limit ≥ max_sustainable_population ∧
  population_after_years (years_to_sustainable_limit - years_to_double) < max_sustainable_population :=
by sorry

end NUMINAMATH_CALUDE_population_reaches_limit_l2058_205888


namespace NUMINAMATH_CALUDE_project_budget_equality_l2058_205852

/-- Represents the annual budget change for a project -/
structure BudgetChange where
  initial : ℕ  -- Initial budget in dollars
  annual : ℤ   -- Annual change in dollars (positive for increase, negative for decrease)

/-- Calculates the budget after a given number of years -/
def budget_after_years (bc : BudgetChange) (years : ℕ) : ℤ :=
  bc.initial + years * bc.annual

/-- The problem statement -/
theorem project_budget_equality (q v : BudgetChange) 
  (hq_initial : q.initial = 540000)
  (hv_initial : v.initial = 780000)
  (hq_annual : q.annual = 30000)
  (h_equal_after_4 : budget_after_years q 4 = budget_after_years v 4) :
  v.annual = -30000 := by
  sorry

end NUMINAMATH_CALUDE_project_budget_equality_l2058_205852


namespace NUMINAMATH_CALUDE_certain_number_proof_l2058_205816

theorem certain_number_proof (x : ℝ) : x / 14.5 = 179 → x = 2595.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2058_205816


namespace NUMINAMATH_CALUDE_initial_blue_marbles_l2058_205863

theorem initial_blue_marbles (blue red : ℕ) : 
  (blue : ℚ) / red = 5 / 3 →
  ((blue - 10 : ℚ) / (red + 25) = 1 / 4) →
  blue = 19 := by
sorry

end NUMINAMATH_CALUDE_initial_blue_marbles_l2058_205863


namespace NUMINAMATH_CALUDE_product_of_diff_squares_l2058_205835

theorem product_of_diff_squares (a b c d : ℕ+) 
  (ha : ∃ (x y : ℕ+), a = x^2 - y^2)
  (hb : ∃ (z w : ℕ+), b = z^2 - w^2)
  (hc : ∃ (p q : ℕ+), c = p^2 - q^2)
  (hd : ∃ (r s : ℕ+), d = r^2 - s^2) :
  ∃ (u v : ℕ+), (a * b * c * d : ℕ) = u^2 - v^2 :=
sorry

end NUMINAMATH_CALUDE_product_of_diff_squares_l2058_205835


namespace NUMINAMATH_CALUDE_valid_course_combinations_l2058_205868

def total_courses : ℕ := 7
def required_courses : ℕ := 4
def math_courses : ℕ := 3
def other_courses : ℕ := 4

def valid_combinations : ℕ := (total_courses - 1).choose (required_courses - 1) - other_courses.choose (required_courses - 1)

theorem valid_course_combinations :
  valid_combinations = 16 :=
sorry

end NUMINAMATH_CALUDE_valid_course_combinations_l2058_205868


namespace NUMINAMATH_CALUDE_square_sum_equals_25_l2058_205839

theorem square_sum_equals_25 (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) :
  x^2 + y^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_25_l2058_205839


namespace NUMINAMATH_CALUDE_stating_magical_stack_size_magical_stack_n_l2058_205841

/-- Represents a stack of cards with the described properties. -/
structure CardStack :=
  (n : ℕ)  -- Half the total number of cards
  (is_magical : Bool)  -- Whether the stack is magical after restacking

/-- 
  Theorem stating that a magical stack where card 101 retains its position
  must have 302 cards in total.
-/
theorem magical_stack_size 
  (stack : CardStack) 
  (h_magical : stack.is_magical = true) 
  (h_101_position : ∃ (pos : ℕ), pos ≤ 2 * stack.n ∧ pos = 101) :
  2 * stack.n = 302 := by
  sorry

/-- 
  Corollary: The value of n in a magical stack where card 101 
  retains its position is 151.
-/
theorem magical_stack_n 
  (stack : CardStack) 
  (h_magical : stack.is_magical = true) 
  (h_101_position : ∃ (pos : ℕ), pos ≤ 2 * stack.n ∧ pos = 101) :
  stack.n = 151 := by
  sorry

end NUMINAMATH_CALUDE_stating_magical_stack_size_magical_stack_n_l2058_205841


namespace NUMINAMATH_CALUDE_min_jam_prob_route_l2058_205870

structure Route where
  segments : List (Char × Char)

def no_jam_prob (r : Route) (probs : List ℚ) : ℚ :=
  probs.prod

def jam_prob (r : Route) (probs : List ℚ) : ℚ :=
  1 - no_jam_prob r probs

theorem min_jam_prob_route (route1 route2 route3 : Route)
  (probs1 probs2 probs3 : List ℚ) :
  route1.segments = [('A', 'C'), ('C', 'D'), ('D', 'B')] →
  route2.segments = [('A', 'C'), ('C', 'F'), ('F', 'B')] →
  route3.segments = [('A', 'E'), ('E', 'F'), ('F', 'B')] →
  probs1 = [9/10, 14/15, 5/6] →
  probs2 = [9/10, 9/10, 15/16] →
  probs3 = [9/10, 9/10, 19/20] →
  jam_prob route1 probs1 < jam_prob route2 probs2 ∧
  jam_prob route1 probs1 < jam_prob route3 probs3 :=
by sorry

end NUMINAMATH_CALUDE_min_jam_prob_route_l2058_205870


namespace NUMINAMATH_CALUDE_height_of_specific_prism_l2058_205858

/-- A right triangular prism with base PQR -/
structure RightTriangularPrism where
  /-- Length of side PQ of the base triangle -/
  pq : ℝ
  /-- Length of side PR of the base triangle -/
  pr : ℝ
  /-- Volume of the prism -/
  volume : ℝ

/-- Theorem: The height of a specific right triangular prism is 10 -/
theorem height_of_specific_prism (prism : RightTriangularPrism)
  (h_pq : prism.pq = Real.sqrt 5)
  (h_pr : prism.pr = Real.sqrt 5)
  (h_vol : prism.volume = 25.000000000000004) :
  (2 * prism.volume) / (prism.pq * prism.pr) = 10 := by
  sorry


end NUMINAMATH_CALUDE_height_of_specific_prism_l2058_205858


namespace NUMINAMATH_CALUDE_roots_relation_l2058_205897

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 4

-- Define the polynomial j(x)
def j (b c d x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- Theorem statement
theorem roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → ∃ y : ℝ, j b c d y = 0 ∧ y = x^3) →
  b = -8 ∧ c = 36 ∧ d = -64 := by
sorry

end NUMINAMATH_CALUDE_roots_relation_l2058_205897


namespace NUMINAMATH_CALUDE_least_bananas_l2058_205842

theorem least_bananas (b₁ b₂ b₃ : ℕ) : 
  (∃ (A B C : ℕ), 
    A = b₁ / 2 + b₂ / 3 + 5 * b₃ / 12 ∧
    B = b₁ / 4 + 2 * b₂ / 3 + 5 * b₃ / 12 ∧
    C = b₁ / 4 + b₂ / 3 + b₃ / 6 ∧
    A = 4 * k ∧ B = 3 * k ∧ C = 2 * k ∧
    (∀ m, m < b₁ + b₂ + b₃ → 
      ¬(∃ (A' B' C' : ℕ), 
        A' = m / 2 + (b₁ + b₂ + b₃ - m) / 3 + 5 * (b₁ + b₂ + b₃ - m) / 12 ∧
        B' = m / 4 + 2 * (b₁ + b₂ + b₃ - m) / 3 + 5 * (b₁ + b₂ + b₃ - m) / 12 ∧
        C' = m / 4 + (b₁ + b₂ + b₃ - m) / 3 + (b₁ + b₂ + b₃ - m) / 6 ∧
        A' = 4 * k' ∧ B' = 3 * k' ∧ C' = 2 * k'))) →
  b₁ + b₂ + b₃ = 276 :=
by sorry

end NUMINAMATH_CALUDE_least_bananas_l2058_205842


namespace NUMINAMATH_CALUDE_frank_whack_a_mole_tickets_frank_whack_a_mole_tickets_proof_l2058_205832

theorem frank_whack_a_mole_tickets : ℕ → Prop :=
  fun whack_tickets : ℕ =>
    let skee_ball_tickets : ℕ := 9
    let candy_cost : ℕ := 6
    let candies_bought : ℕ := 7
    let total_tickets : ℕ := whack_tickets + skee_ball_tickets
    total_tickets = candy_cost * candies_bought →
    whack_tickets = 33

-- The proof would go here
theorem frank_whack_a_mole_tickets_proof : frank_whack_a_mole_tickets 33 := by
  sorry

end NUMINAMATH_CALUDE_frank_whack_a_mole_tickets_frank_whack_a_mole_tickets_proof_l2058_205832


namespace NUMINAMATH_CALUDE_line_l_equation_l2058_205884

/-- A line l passes through point P(-1,2) and has equal distances from points A(2,3) and B(-4,6) -/
def line_l (x y : ℝ) : Prop :=
  (x = -1 ∧ y = 2) ∨ 
  (abs ((2 * x - y + 2) / Real.sqrt (x^2 + 1)) = abs ((-4 * x - y + 2) / Real.sqrt (x^2 + 1)))

/-- The equation of line l is either x+2y-3=0 or x=-1 -/
theorem line_l_equation : 
  ∀ x y : ℝ, line_l x y ↔ (x + 2*y - 3 = 0 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_l_equation_l2058_205884


namespace NUMINAMATH_CALUDE_sine_equality_implies_equal_coefficients_l2058_205874

theorem sine_equality_implies_equal_coefficients 
  (α β γ δ : ℝ) 
  (h_positive : α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0) 
  (h_equality : ∀ x : ℝ, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (δ * x)) : 
  α = γ ∨ α = δ := by
sorry

end NUMINAMATH_CALUDE_sine_equality_implies_equal_coefficients_l2058_205874


namespace NUMINAMATH_CALUDE_solve_for_t_l2058_205885

theorem solve_for_t (s t : ℚ) 
  (eq1 : 15 * s + 7 * t = 236)
  (eq2 : t = 2 * s + 1) : 
  t = 487 / 29 := by
sorry

end NUMINAMATH_CALUDE_solve_for_t_l2058_205885


namespace NUMINAMATH_CALUDE_speed_ratio_problem_l2058_205810

/-- 
Given two people traveling in opposite directions for one hour, 
if one person takes 35 minutes longer to reach the other's destination when they swap,
then the ratio of their speeds is 3:4.
-/
theorem speed_ratio_problem (v₁ v₂ : ℝ) : 
  v₁ > 0 → v₂ > 0 → 
  (60 * v₂ = 60 * v₁ / v₂ + 35 * v₁) → 
  v₁ / v₂ = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_speed_ratio_problem_l2058_205810


namespace NUMINAMATH_CALUDE_factor_expression_l2058_205871

theorem factor_expression (y : ℝ) : 16 * y^3 + 8 * y^2 = 8 * y^2 * (2 * y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2058_205871


namespace NUMINAMATH_CALUDE_f_inequality_implies_b_geq_one_l2058_205881

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp (-x)

-- State the theorem
theorem f_inequality_implies_b_geq_one :
  ∀ b : ℝ,
  (∀ a : ℝ, a ≤ 0 → ∀ x : ℝ, x ≥ 0 → f a x ≤ b * Real.log (x + 1)) →
  b ≥ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_f_inequality_implies_b_geq_one_l2058_205881


namespace NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l2058_205855

-- Define a triangle
structure Triangle where
  a : ℝ  -- angle a
  b : ℝ  -- angle b
  c : ℝ  -- angle c
  sum_180 : a + b + c = 180  -- sum of angles in a triangle is 180 degrees

-- Define what it means for two angles to be complementary
def complementary (x y : ℝ) : Prop := x + y = 90

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop := t.a = 90 ∨ t.b = 90 ∨ t.c = 90

-- Theorem statement
theorem complementary_angles_imply_right_triangle (t : Triangle) :
  (complementary t.a t.b ∨ complementary t.b t.c ∨ complementary t.a t.c) →
  is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l2058_205855


namespace NUMINAMATH_CALUDE_remainder_sum_l2058_205820

theorem remainder_sum (c d : ℤ) 
  (hc : c % 90 = 84) 
  (hd : d % 120 = 117) : 
  (c + d) % 30 = 21 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l2058_205820


namespace NUMINAMATH_CALUDE_no_solution_iff_m_special_l2058_205821

/-- The equation has no solution if and only if m is -4, 6, or 1 -/
theorem no_solution_iff_m_special (m : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → 2 / (x - 2) + m * x / (x^2 - 4) ≠ 3 / (x + 2)) ↔ 
  (m = -4 ∨ m = 6 ∨ m = 1) :=
sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_special_l2058_205821


namespace NUMINAMATH_CALUDE_power_equality_l2058_205862

theorem power_equality (q : ℕ) (h : (81 : ℕ)^6 = 3^q) : q = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2058_205862


namespace NUMINAMATH_CALUDE_largest_integer_less_than_sqrt5_plus_sqrt3_to_6th_l2058_205886

theorem largest_integer_less_than_sqrt5_plus_sqrt3_to_6th (n : ℕ) : 
  n = 3322 ↔ n = ⌊(Real.sqrt 5 + Real.sqrt 3)^6⌋ :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_sqrt5_plus_sqrt3_to_6th_l2058_205886


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2058_205828

open Function Real

theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, x + y + z = 0 → f (x^3) + f y^3 + f z^3 = 3*x*y*z) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2058_205828


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2058_205822

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value (k : ℝ) :
  let a : ℝ × ℝ := (2, k)
  let b : ℝ × ℝ := (1, 2)
  parallel a b → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2058_205822
