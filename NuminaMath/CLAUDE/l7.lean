import Mathlib

namespace NUMINAMATH_CALUDE_full_price_revenue_l7_750

/-- Represents the fundraiser scenario -/
structure Fundraiser where
  total_tickets : ℕ
  total_revenue : ℚ
  full_price : ℚ
  full_price_tickets : ℕ

/-- The fundraiser satisfies the given conditions -/
def valid_fundraiser (f : Fundraiser) : Prop :=
  f.total_tickets = 180 ∧
  f.total_revenue = 2600 ∧
  f.full_price > 0 ∧
  f.full_price_tickets ≤ f.total_tickets ∧
  f.full_price_tickets * f.full_price + (f.total_tickets - f.full_price_tickets) * (f.full_price / 3) = f.total_revenue

/-- The theorem stating that the revenue from full-price tickets is $975 -/
theorem full_price_revenue (f : Fundraiser) (h : valid_fundraiser f) : 
  f.full_price_tickets * f.full_price = 975 := by
  sorry

end NUMINAMATH_CALUDE_full_price_revenue_l7_750


namespace NUMINAMATH_CALUDE_apple_price_theorem_l7_774

/-- The price of apples with a two-tier pricing system -/
theorem apple_price_theorem 
  (l q : ℝ) 
  (h1 : 30 * l + 3 * q = 360) 
  (h2 : 30 * l + 6 * q = 420) : 
  25 * l = 250 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_theorem_l7_774


namespace NUMINAMATH_CALUDE_max_value_theorem_l7_735

theorem max_value_theorem (x y z : ℝ) (h : x + y + z = 3) :
  Real.sqrt (2 * x + 13) + (3 * y + 5) ^ (1/3) + (8 * z + 12) ^ (1/4) ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l7_735


namespace NUMINAMATH_CALUDE_two_p_plus_q_l7_748

theorem two_p_plus_q (p q : ℚ) (h : p / q = 6 / 7) : 2 * p + q = (19 / 7) * q := by
  sorry

end NUMINAMATH_CALUDE_two_p_plus_q_l7_748


namespace NUMINAMATH_CALUDE_angle_triple_supplement_measure_l7_784

theorem angle_triple_supplement_measure : 
  ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ x = 3 * (180 - x) ∧ x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_measure_l7_784


namespace NUMINAMATH_CALUDE_inverse_of_matrix_A_l7_732

theorem inverse_of_matrix_A (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A * !![1, 2; 0, 6] = !![(-1), (-2); 0, 3] →
  A⁻¹ = !![(-1), 0; 0, 2] := by sorry

end NUMINAMATH_CALUDE_inverse_of_matrix_A_l7_732


namespace NUMINAMATH_CALUDE_absolute_value_equation_range_l7_769

theorem absolute_value_equation_range (x : ℝ) : 
  |x - 1| + x - 1 = 0 → x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_range_l7_769


namespace NUMINAMATH_CALUDE_triangle_radii_inequality_l7_727

/-- For any triangle with circumradius R, inradius r, and exradii r_a, r_b, r_c,
    the inequality (r * r_a * r_b * r_c) / R^4 ≤ 27/16 holds. -/
theorem triangle_radii_inequality (R r r_a r_b r_c : ℝ) 
    (h_R : R > 0) 
    (h_r : r > 0) 
    (h_ra : r_a > 0) 
    (h_rb : r_b > 0) 
    (h_rc : r_c > 0) 
    (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      R = (a * b * c) / (4 * (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b))^(1/2) ∧
      r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * (a + b + c)) ∧
      r_a = (b + c - a) / 2 ∧
      r_b = (c + a - b) / 2 ∧
      r_c = (a + b - c) / 2) :
  (r * r_a * r_b * r_c) / R^4 ≤ 27/16 := by
sorry

end NUMINAMATH_CALUDE_triangle_radii_inequality_l7_727


namespace NUMINAMATH_CALUDE_line_circle_no_intersection_l7_789

/-- The range of m for which a line and circle have no intersection -/
theorem line_circle_no_intersection (m : ℝ) : 
  (∀ x y : ℝ, 3*x + 4*y + m ≠ 0 ∨ (x+1)^2 + (y-2)^2 ≠ 1) →
  m < -10 ∨ m > 0 :=
sorry

end NUMINAMATH_CALUDE_line_circle_no_intersection_l7_789


namespace NUMINAMATH_CALUDE_sand_pile_volume_l7_791

/-- The volume of a conical sand pile -/
theorem sand_pile_volume (d h r : ℝ) : 
  d = 10 →  -- diameter is 10 feet
  h = 0.6 * d →  -- height is 60% of diameter
  r = d / 2 →  -- radius is half of diameter
  (1 / 3) * π * r^2 * h = 50 * π := by
  sorry

end NUMINAMATH_CALUDE_sand_pile_volume_l7_791


namespace NUMINAMATH_CALUDE_softball_team_size_l7_768

theorem softball_team_size :
  ∀ (men women : ℕ),
  women = men + 2 →
  (men : ℚ) / (women : ℚ) = 7777777777777778 / 10000000000000000 →
  men + women = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_size_l7_768


namespace NUMINAMATH_CALUDE_cooking_probability_l7_722

-- Define a finite set of courses
def Courses : Type := Fin 4

-- Define a probability measure on the set of courses
def prob : Courses → ℚ := λ _ => 1 / 4

-- Theorem statement
theorem cooking_probability :
  ∀ (c : Courses), prob c = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_cooking_probability_l7_722


namespace NUMINAMATH_CALUDE_b_2017_eq_1_l7_798

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Sequence of Fibonacci numbers modulo 3 -/
def b (n : ℕ) : ℕ := fib n % 3

/-- The sequence b has period 8 -/
axiom b_period (n : ℕ) : b (n + 8) = b n

theorem b_2017_eq_1 : b 2017 = 1 := by sorry

end NUMINAMATH_CALUDE_b_2017_eq_1_l7_798


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l7_770

/-- The original price of the shirt -/
def shirt_price : ℝ := 156.52

/-- The original price of the coat -/
def coat_price : ℝ := 3 * shirt_price

/-- The original price of the pants -/
def pants_price : ℝ := 2 * shirt_price

/-- The total cost after discounts -/
def total_cost : ℝ := 900

theorem shirt_price_calculation :
  (shirt_price * 0.9 + coat_price * 0.95 + pants_price) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l7_770


namespace NUMINAMATH_CALUDE_triangle_side_length_l7_778

theorem triangle_side_length 
  (AB : ℝ) 
  (angle_ADB : ℝ) 
  (sin_A : ℝ) 
  (sin_C : ℝ) 
  (h1 : AB = 30)
  (h2 : angle_ADB = Real.pi / 2)
  (h3 : sin_A = 2/3)
  (h4 : sin_C = 1/4) :
  ∃ (DC : ℝ), DC = 20 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l7_778


namespace NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_fraction_in_lowest_terms_l7_740

def repeating_decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a : ℚ) + (b * 10 + c : ℚ) / 990

theorem repeating_decimal_equiv_fraction :
  repeating_decimal_to_fraction 4 1 7 = 413 / 990 :=
sorry

theorem fraction_in_lowest_terms : ∀ n : ℕ, n > 1 → n ∣ 413 → n ∣ 990 → False :=
sorry

#eval repeating_decimal_to_fraction 4 1 7

end NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_fraction_in_lowest_terms_l7_740


namespace NUMINAMATH_CALUDE_set_operations_and_inclusion_l7_755

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a ≥ 0}

-- Theorem statement
theorem set_operations_and_inclusion :
  (A ∩ B = {x | 2 ≤ x ∧ x < 3}) ∧
  (A ∪ B = {x | x ≥ -1}) ∧
  (∀ a : ℝ, B ⊆ C a → a > -4) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_inclusion_l7_755


namespace NUMINAMATH_CALUDE_fruit_cost_difference_l7_782

/-- Represents the cost and quantity of a fruit carton -/
structure FruitCarton where
  cost : ℚ  -- Cost in dollars
  quantity : ℚ  -- Quantity in ounces
  inv_mk : cost > 0 ∧ quantity > 0

/-- Calculates the number of cartons needed for a given amount of fruit -/
def cartonsNeeded (fruit : FruitCarton) (amount : ℚ) : ℚ :=
  amount / fruit.quantity

/-- Calculates the total cost for a given number of cartons -/
def totalCost (fruit : FruitCarton) (cartons : ℚ) : ℚ :=
  fruit.cost * cartons

/-- The main theorem to prove -/
theorem fruit_cost_difference 
  (blueberries : FruitCarton)
  (raspberries : FruitCarton)
  (batches : ℕ)
  (fruitPerBatch : ℚ)
  (h1 : blueberries.cost = 5)
  (h2 : blueberries.quantity = 6)
  (h3 : raspberries.cost = 3)
  (h4 : raspberries.quantity = 8)
  (h5 : batches = 4)
  (h6 : fruitPerBatch = 12) :
  totalCost blueberries (cartonsNeeded blueberries (batches * fruitPerBatch)) -
  totalCost raspberries (cartonsNeeded raspberries (batches * fruitPerBatch)) = 22 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_difference_l7_782


namespace NUMINAMATH_CALUDE_arithmetic_equality_l7_779

theorem arithmetic_equality : 45 * 52 + 28 * 45 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l7_779


namespace NUMINAMATH_CALUDE_u_equivalence_l7_799

theorem u_equivalence (u : ℝ) : 
  u = 1 / (2 - Real.rpow 3 (1/3)) → 
  u = ((2 + Real.rpow 3 (1/3)) * (4 + Real.rpow 9 (1/3))) / 7 := by
sorry

end NUMINAMATH_CALUDE_u_equivalence_l7_799


namespace NUMINAMATH_CALUDE_min_face_sum_l7_754

-- Define a cube as a set of 8 integers
def Cube := Fin 8 → ℕ

-- Define a face as a set of 4 vertices
def Face := Fin 4 → Fin 8

-- Condition: numbers are from 1 to 8
def valid_cube (c : Cube) : Prop :=
  (∀ i, c i ≥ 1 ∧ c i ≤ 8) ∧ (∀ i j, i ≠ j → c i ≠ c j)

-- Condition: sum of any three vertices on a face is at least 10
def valid_face_sums (c : Cube) (f : Face) : Prop :=
  ∀ i j k, i < j → j < k → c (f i) + c (f j) + c (f k) ≥ 10

-- The sum of numbers on a face
def face_sum (c : Cube) (f : Face) : ℕ :=
  (c (f 0)) + (c (f 1)) + (c (f 2)) + (c (f 3))

-- The theorem to prove
theorem min_face_sum (c : Cube) :
  valid_cube c → (∀ f : Face, valid_face_sums c f) →
  ∃ f : Face, face_sum c f = 16 ∧ ∀ g : Face, face_sum c g ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_face_sum_l7_754


namespace NUMINAMATH_CALUDE_proportion_problem_l7_785

theorem proportion_problem : ∃ X : ℝ, (8 / 4 = X / 240) ∧ X = 480 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l7_785


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l7_707

-- Define the sets P and Q
def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem union_of_P_and_Q : P ∪ Q = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l7_707


namespace NUMINAMATH_CALUDE_betty_garden_total_l7_718

/-- Represents Betty's herb garden -/
structure HerbGarden where
  basil : ℕ
  oregano : ℕ

/-- The number of oregano plants is 2 more than twice the number of basil plants -/
def oregano_rule (garden : HerbGarden) : Prop :=
  garden.oregano = 2 + 2 * garden.basil

/-- Betty's garden has 5 basil plants -/
def betty_garden : HerbGarden :=
  { basil := 5, oregano := 2 + 2 * 5 }

/-- The total number of plants in the garden -/
def total_plants (garden : HerbGarden) : ℕ :=
  garden.basil + garden.oregano

theorem betty_garden_total : total_plants betty_garden = 17 := by
  sorry

end NUMINAMATH_CALUDE_betty_garden_total_l7_718


namespace NUMINAMATH_CALUDE_parallelepiped_ring_sum_exists_l7_729

/-- Represents a rectangular parallelepiped with dimensions a × b × c -/
structure Parallelepiped (a b c : ℕ) where
  dim_a : a > 0
  dim_b : b > 0
  dim_c : c > 0

/-- Represents an assignment of numbers to the faces of a parallelepiped -/
def FaceAssignment (a b c : ℕ) := Fin 6 → ℕ

/-- Calculates the sum of numbers in a 1-unit-wide ring around the parallelepiped -/
def ringSum (p : Parallelepiped 3 4 5) (assignment : FaceAssignment 3 4 5) : ℕ :=
  2 * (4 * assignment 0 + 5 * assignment 2 +
       3 * assignment 0 + 5 * assignment 4 +
       3 * assignment 2 + 4 * assignment 4)

/-- The main theorem stating that there exists an assignment satisfying the condition -/
theorem parallelepiped_ring_sum_exists :
  ∃ (assignment : FaceAssignment 3 4 5),
    ∀ (p : Parallelepiped 3 4 5), ringSum p assignment = 120 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_ring_sum_exists_l7_729


namespace NUMINAMATH_CALUDE_eight_b_value_l7_702

theorem eight_b_value (a b : ℚ) 
  (eq1 : 6 * a + 3 * b = 3) 
  (eq2 : b = 2 * a - 3) : 
  8 * b = -8 := by
sorry

end NUMINAMATH_CALUDE_eight_b_value_l7_702


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l7_772

theorem smallest_x_absolute_value_equation : 
  (∀ x : ℝ, |5*x + 15| = 40 → x ≥ -11) ∧ 
  (|5*(-11) + 15| = 40) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l7_772


namespace NUMINAMATH_CALUDE_function_zero_in_interval_l7_710

/-- The function f(x) = 2ax^2 + 2x - 3 - a has a zero in the interval [-1, 1] 
    if and only if a ≤ (-3 - √7)/2 or a ≥ 1 -/
theorem function_zero_in_interval (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * a * x^2 + 2 * x - 3 - a = 0) ↔ 
  (a ≤ (-3 - Real.sqrt 7) / 2 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_function_zero_in_interval_l7_710


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l7_764

theorem geometric_arithmetic_sequence_ratio (x y z : ℝ) 
  (h1 : (4 * y) / (3 * x) = (5 * z) / (4 * y))  -- geometric sequence condition
  (h2 : 1 / y - 1 / x = 1 / z - 1 / y)         -- arithmetic sequence condition
  (h3 : x ≠ 0)
  (h4 : y ≠ 0)
  (h5 : z ≠ 0) :
  x / z + z / x = 34 / 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l7_764


namespace NUMINAMATH_CALUDE_asian_games_mascot_sales_l7_737

/-- Asian Games Mascot Sales Problem -/
theorem asian_games_mascot_sales 
  (initial_price : ℝ) 
  (cost_price : ℝ) 
  (initial_sales : ℝ) 
  (price_reduction_factor : ℝ) :
  initial_price = 80 ∧ 
  cost_price = 50 ∧ 
  initial_sales = 200 ∧ 
  price_reduction_factor = 20 →
  ∃ (sales_function : ℝ → ℝ) 
    (profit_function : ℝ → ℝ) 
    (optimal_price : ℝ),
    (∀ x, sales_function x = -20 * x + 1800) ∧
    (profit_function 65 = 7500 ∧ profit_function 75 = 7500) ∧
    (optimal_price = 70 ∧ 
     ∀ x, profit_function x ≤ profit_function optimal_price) :=
by sorry

end NUMINAMATH_CALUDE_asian_games_mascot_sales_l7_737


namespace NUMINAMATH_CALUDE_exists_n_sum_digits_decreases_l7_761

-- Define the sum of digits function
def S (a : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_n_sum_digits_decreases :
  ∃ n : ℕ, n > 1000 ∧ S (2^n) > S (2^(n+1)) := by sorry

end NUMINAMATH_CALUDE_exists_n_sum_digits_decreases_l7_761


namespace NUMINAMATH_CALUDE_binomial_divides_lcm_l7_738

theorem binomial_divides_lcm (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, k * Nat.choose (2 * n) n = Finset.lcm (Finset.range (2 * n + 1)) id :=
by sorry

end NUMINAMATH_CALUDE_binomial_divides_lcm_l7_738


namespace NUMINAMATH_CALUDE_stickers_distribution_l7_760

/-- Calculates the number of stickers each of the other students received -/
def stickers_per_other_student (total_stickers : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) 
  (leftover_stickers : ℕ) (total_students : ℕ) : ℕ :=
  let stickers_given_to_friends := friends * stickers_per_friend
  let total_stickers_given := total_stickers - leftover_stickers
  let stickers_for_others := total_stickers_given - stickers_given_to_friends
  let other_students := total_students - 1 - friends
  stickers_for_others / other_students

theorem stickers_distribution (total_stickers : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) 
  (leftover_stickers : ℕ) (total_students : ℕ)
  (h1 : total_stickers = 50)
  (h2 : friends = 5)
  (h3 : stickers_per_friend = 4)
  (h4 : leftover_stickers = 8)
  (h5 : total_students = 17) :
  stickers_per_other_student total_stickers friends stickers_per_friend leftover_stickers total_students = 2 := by
  sorry

end NUMINAMATH_CALUDE_stickers_distribution_l7_760


namespace NUMINAMATH_CALUDE_average_age_decrease_l7_708

theorem average_age_decrease (initial_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (total_students : ℕ) : 
  initial_avg = 48 →
  new_students = 120 →
  new_avg = 32 →
  total_students = 160 →
  let original_students := total_students - new_students
  let total_age := initial_avg * original_students + new_avg * new_students
  let new_avg_age := total_age / total_students
  initial_avg - new_avg_age = 12 := by
sorry

end NUMINAMATH_CALUDE_average_age_decrease_l7_708


namespace NUMINAMATH_CALUDE_common_solution_iff_y_eq_one_l7_703

/-- The first equation: x^2 + y^2 - 4 = 0 -/
def equation1 (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0

/-- The second equation: x^2 - 4y + y^2 = 0 -/
def equation2 (x y : ℝ) : Prop := x^2 - 4*y + y^2 = 0

/-- The theorem stating that the equations have common real solutions iff y = 1 -/
theorem common_solution_iff_y_eq_one :
  (∃ x : ℝ, equation1 x 1 ∧ equation2 x 1) ∧
  (∀ y : ℝ, y ≠ 1 → ¬∃ x : ℝ, equation1 x y ∧ equation2 x y) :=
sorry

end NUMINAMATH_CALUDE_common_solution_iff_y_eq_one_l7_703


namespace NUMINAMATH_CALUDE_no_prime_sum_53_less_than_30_l7_766

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def primeSum53LessThan30 : Prop :=
  ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 53 ∧ (p < 30 ∨ q < 30)

theorem no_prime_sum_53_less_than_30 : primeSum53LessThan30 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_53_less_than_30_l7_766


namespace NUMINAMATH_CALUDE_parabola_p_value_l7_751

/-- The latus rectum of a parabola y^2 = 2px --/
def latus_rectum (p : ℝ) : ℝ := 4 * p

/-- Theorem: For a parabola y^2 = 2px with latus rectum equal to 4, p equals 2 --/
theorem parabola_p_value : ∀ p : ℝ, latus_rectum p = 4 → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_p_value_l7_751


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l7_758

-- First expression
theorem simplify_expression_1 (x y : ℝ) :
  4 * y^2 + 3 * x - 5 + 6 - 4 * x - 2 * y^2 = 2 * y^2 - x + 1 := by sorry

-- Second expression
theorem simplify_expression_2 (m n : ℝ) :
  3/2 * (m^2 - m*n) - 2 * (m*n + m^2) = -1/2 * m^2 - 7/2 * m*n := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l7_758


namespace NUMINAMATH_CALUDE_initial_roses_count_l7_797

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses added to the vase -/
def added_roses : ℕ := 13

/-- The total number of roses in the vase after adding -/
def total_roses : ℕ := 20

/-- Theorem stating that the initial number of roses is 7 -/
theorem initial_roses_count : initial_roses = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_roses_count_l7_797


namespace NUMINAMATH_CALUDE_min_sum_m_n_l7_753

theorem min_sum_m_n (m n : ℕ+) (h : 45 * m = n^3) : 
  (∀ m' n' : ℕ+, 45 * m' = n'^3 → m' + n' ≥ m + n) → m + n = 90 := by
sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l7_753


namespace NUMINAMATH_CALUDE_distinct_roots_equal_integer_roots_l7_717

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (m + 3) * x + 2 * m

-- Part 1: Prove the equation always has two distinct real roots
theorem distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

-- Part 2: Prove the specific case has two equal integer roots
theorem equal_integer_roots : 
  ∃ x : ℤ, quadratic 2 (x : ℝ) = 0 ∧ x = -2 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_equal_integer_roots_l7_717


namespace NUMINAMATH_CALUDE_regular_octagon_diagonal_ratio_l7_780

/-- The ratio of the shortest diagonal to the longest diagonal in a regular octagon -/
theorem regular_octagon_diagonal_ratio : 
  ∃ (shortest_diagonal longest_diagonal : ℝ), 
    shortest_diagonal > 0 ∧ 
    longest_diagonal > 0 ∧
    shortest_diagonal / longest_diagonal = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_diagonal_ratio_l7_780


namespace NUMINAMATH_CALUDE_reflection_sum_coordinates_l7_763

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflect a point over the y-axis -/
def reflectOverYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- Sum of coordinates of two points -/
def sumCoordinates (p1 p2 : Point2D) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

theorem reflection_sum_coordinates (a : ℝ) :
  let C : Point2D := { x := a, y := 8 }
  let D : Point2D := reflectOverYAxis C
  sumCoordinates C D = 16 := by
  sorry

end NUMINAMATH_CALUDE_reflection_sum_coordinates_l7_763


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l7_730

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, -1/2 < x ∧ x < 1/3 ↔ a * x^2 + b * x + 2 > 0) →
  a - b = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l7_730


namespace NUMINAMATH_CALUDE_parabola_chord_perpendicular_bisector_l7_715

/-- The parabola y^2 = 8(x+2) with focus at (0, 0) -/
def parabola (x y : ℝ) : Prop := y^2 = 8*(x+2)

/-- The line y = x passing through (0, 0) -/
def line (x y : ℝ) : Prop := y = x

/-- The perpendicular bisector of a chord on the line y = x -/
def perp_bisector (x y : ℝ) : Prop := y = -x + 2*x

theorem parabola_chord_perpendicular_bisector :
  ∀ (x : ℝ),
  (∃ (y : ℝ), parabola x y ∧ line x y) →
  (∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = 0 ∧ perp_bisector P.1 P.2) →
  x = x := by sorry

end NUMINAMATH_CALUDE_parabola_chord_perpendicular_bisector_l7_715


namespace NUMINAMATH_CALUDE_arc_length_quarter_circle_l7_700

/-- Given a circle with circumference 120 feet and a central angle of 90°, 
    the length of the corresponding arc is 30 feet. -/
theorem arc_length_quarter_circle (D : Real) (EF : Real) (EOF : Real) : 
  D = 120 → EOF = 90 → EF = 30 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_quarter_circle_l7_700


namespace NUMINAMATH_CALUDE_contractor_absent_days_l7_793

/-- Represents the problem of calculating a contractor's absent days. -/
def ContractorProblem (total_days : ℕ) (daily_wage : ℚ) (daily_fine : ℚ) (total_amount : ℚ) : Prop :=
  ∃ (worked_days absent_days : ℕ),
    worked_days + absent_days = total_days ∧
    daily_wage * worked_days - daily_fine * absent_days = total_amount

/-- Theorem stating that given the problem conditions, the number of absent days is 10. -/
theorem contractor_absent_days :
  ContractorProblem 30 25 (15/2) 425 →
  ∃ (worked_days absent_days : ℕ),
    worked_days + absent_days = 30 ∧
    absent_days = 10 := by
  sorry

#check contractor_absent_days

end NUMINAMATH_CALUDE_contractor_absent_days_l7_793


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l7_706

/-- For a quadratic equation ax^2 + 2x + 1 = 0 to have real roots, 
    a must satisfy: a ≤ 1 and a ≠ 0 -/
theorem quadratic_real_roots_condition (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) ↔ (a ≤ 1 ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l7_706


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l7_767

theorem divisibility_by_seven (k : ℕ) : 
  (∀ n : ℕ, n > 0 → ∃ q : ℤ, (3^(6*n - 1) - k * 2^(3*n - 2) + 1 : ℤ) = 7 * q) ↔ 
  (∃ m : ℤ, k = 7 * m + 3) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l7_767


namespace NUMINAMATH_CALUDE_integral_exp_sin_l7_790

open Real

theorem integral_exp_sin (α β : ℝ) :
  deriv (fun x => (exp (α * x) * (α * sin (β * x) - β * cos (β * x))) / (α^2 + β^2)) =
  fun x => exp (α * x) * sin (β * x) := by
sorry

end NUMINAMATH_CALUDE_integral_exp_sin_l7_790


namespace NUMINAMATH_CALUDE_geometric_sequence_decreasing_l7_786

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a₁ * q^(n-1)

theorem geometric_sequence_decreasing (a₁ q : ℝ) :
  (∀ n : ℕ, geometric_sequence a₁ q (n+1) < geometric_sequence a₁ q n) ↔
  ((a₁ > 0 ∧ 0 < q ∧ q < 1) ∨ (a₁ < 0 ∧ q > 1)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_decreasing_l7_786


namespace NUMINAMATH_CALUDE_cloth_sale_meters_l7_787

/-- Proves that the number of meters of cloth sold is 75 given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 4950)
    (h2 : profit_per_meter = 15)
    (h3 : cost_price_per_meter = 51) :
    (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_meters_l7_787


namespace NUMINAMATH_CALUDE_sara_balloons_l7_716

/-- The number of red balloons Sara has left after giving some away -/
def balloons_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Sara is left with 7 red balloons -/
theorem sara_balloons : balloons_left 31 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sara_balloons_l7_716


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l7_714

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ), ∀ (x : ℚ), x ≠ 3 → x ≠ 5 →
    (4 * x) / ((x - 5) * (x - 3)^2) = A / (x - 5) + B / (x - 3) + C / (x - 3)^2 ∧
    A = 5 ∧ B = -5 ∧ C = -6 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l7_714


namespace NUMINAMATH_CALUDE_teresas_colored_pencils_l7_745

/-- Given information about Teresa's pencils and her siblings, prove the number of colored pencils she has. -/
theorem teresas_colored_pencils 
  (black_pencils : ℕ) 
  (num_siblings : ℕ) 
  (pencils_per_sibling : ℕ) 
  (pencils_kept : ℕ) 
  (h1 : black_pencils = 35)
  (h2 : num_siblings = 3)
  (h3 : pencils_per_sibling = 13)
  (h4 : pencils_kept = 10) :
  black_pencils + (num_siblings * pencils_per_sibling + pencils_kept) - black_pencils = 14 :=
by sorry

end NUMINAMATH_CALUDE_teresas_colored_pencils_l7_745


namespace NUMINAMATH_CALUDE_soccer_field_kids_l7_739

/-- Given an initial number of kids on a soccer field and the number of friends each kid invites,
    calculate the total number of kids on the field after invitations. -/
def total_kids_after_invitations (initial_kids : ℕ) (friends_per_kid : ℕ) : ℕ :=
  initial_kids + initial_kids * friends_per_kid

/-- Theorem: If there are initially 14 kids on a soccer field and each kid invites 3 friends,
    then the total number of kids on the field after invitations is 56. -/
theorem soccer_field_kids : total_kids_after_invitations 14 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_soccer_field_kids_l7_739


namespace NUMINAMATH_CALUDE_workshop_sample_size_l7_743

/-- Calculates the sample size for a stratum in stratified sampling -/
def stratumSampleSize (totalPopulation : ℕ) (totalSampleSize : ℕ) (stratumSize : ℕ) : ℕ :=
  (totalSampleSize * stratumSize) / totalPopulation

theorem workshop_sample_size :
  let totalProducts : ℕ := 1024
  let sampleSize : ℕ := 64
  let workshopProduction : ℕ := 128
  stratumSampleSize totalProducts sampleSize workshopProduction = 8 := by
  sorry

end NUMINAMATH_CALUDE_workshop_sample_size_l7_743


namespace NUMINAMATH_CALUDE_f_difference_l7_775

/-- k(n) is the largest odd divisor of n -/
def k (n : ℕ+) : ℕ+ := sorry

/-- f(n) is the sum of k(i) from i=1 to n -/
def f (n : ℕ+) : ℕ := sorry

/-- Theorem: f(2n) - f(n) = n^2 for any positive integer n -/
theorem f_difference (n : ℕ+) : f (2 * n) - f n = n^2 := by sorry

end NUMINAMATH_CALUDE_f_difference_l7_775


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l7_725

theorem simplify_fraction_product : 8 * (15 / 14) * (-49 / 45) = -28 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l7_725


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l7_771

/-- The value of m for which an ellipse and a hyperbola with given equations have the same foci -/
theorem ellipse_hyperbola_same_foci (m : ℝ) : 
  (∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1 → ∃ c : ℝ, c^2 = 4 - m^2 ∧ (x = c ∨ x = -c) ∧ y = 0) →
  (∀ x y : ℝ, x^2 / m - y^2 / 2 = 1 → ∃ c : ℝ, c^2 = m + 2 ∧ (x = c ∨ x = -c) ∧ y = 0) →
  (∃ c : ℝ, c^2 = 4 - m^2 ∧ c^2 = m + 2) →
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l7_771


namespace NUMINAMATH_CALUDE_four_point_partition_l7_781

/-- A point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A straight line in a plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a line --/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- A set of four points in a plane --/
def FourPoints := Fin 4 → Point

/-- A partition of four points into two non-empty subsets --/
structure Partition (pts : FourPoints) where
  set1 : Set (Fin 4)
  set2 : Set (Fin 4)
  partition : set1 ∪ set2 = Set.univ
  nonempty1 : set1.Nonempty
  nonempty2 : set2.Nonempty

/-- Check if a line separates two sets of points --/
def separates (l : Line) (pts : FourPoints) (p : Partition pts) : Prop :=
  (∀ i ∈ p.set1, (pts i).onLine l) ∧ (∀ i ∈ p.set2, ¬(pts i).onLine l) ∨
  (∀ i ∈ p.set1, ¬(pts i).onLine l) ∧ (∀ i ∈ p.set2, (pts i).onLine l)

/-- The main theorem --/
theorem four_point_partition (pts : FourPoints) :
  ∃ p : Partition pts, ∀ l : Line, ¬separates l pts p := by
  sorry

end NUMINAMATH_CALUDE_four_point_partition_l7_781


namespace NUMINAMATH_CALUDE_geometric_series_sum_l7_736

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a := 3 / 4
  let r := 3 / 4
  let n := 15
  geometric_sum a r n = 3177884751 / 1073741824 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l7_736


namespace NUMINAMATH_CALUDE_mean_of_xyz_l7_734

theorem mean_of_xyz (original_mean : ℝ) (new_mean : ℝ) (x y z : ℝ) : 
  original_mean = 40 →
  new_mean = 50 →
  z = x + 10 →
  (12 * original_mean + x + y + z) / 15 = new_mean →
  (x + y + z) / 3 = 90 := by
sorry

end NUMINAMATH_CALUDE_mean_of_xyz_l7_734


namespace NUMINAMATH_CALUDE_election_theorem_l7_783

def total_candidates : ℕ := 20
def past_officers : ℕ := 8
def positions_available : ℕ := 4

def elections_with_at_least_two_past_officers : ℕ :=
  Nat.choose past_officers 2 * Nat.choose (total_candidates - past_officers) 2 +
  Nat.choose past_officers 3 * Nat.choose (total_candidates - past_officers) 1 +
  Nat.choose past_officers 4 * Nat.choose (total_candidates - past_officers) 0

theorem election_theorem :
  elections_with_at_least_two_past_officers = 2590 :=
by sorry

end NUMINAMATH_CALUDE_election_theorem_l7_783


namespace NUMINAMATH_CALUDE_chord_tangent_angle_l7_792

-- Define the circle and chord
def Circle : Type := Unit
def Chord (c : Circle) : Type := Unit

-- Define the ratio of arc division
def arc_ratio (c : Circle) (ch : Chord c) : ℚ × ℚ := (11, 16)

-- Define the angle between tangents
def angle_between_tangents (c : Circle) (ch : Chord c) : ℚ := 100 / 3

-- Theorem statement
theorem chord_tangent_angle (c : Circle) (ch : Chord c) :
  arc_ratio c ch = (11, 16) →
  angle_between_tangents c ch = 100 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_chord_tangent_angle_l7_792


namespace NUMINAMATH_CALUDE_smallest_shift_for_scaled_periodic_function_l7_726

-- Define a periodic function with period 20
def isPeriodic20 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 20) = f x

-- Define the property we want to prove
def smallestShift (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f ((x - a) / 5) = f (x / 5)) ∧
  (∀ b, 0 < b → b < a → ∃ x, f ((x - b) / 5) ≠ f (x / 5))

-- Theorem statement
theorem smallest_shift_for_scaled_periodic_function (f : ℝ → ℝ) (h : isPeriodic20 f) :
  smallestShift f 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_shift_for_scaled_periodic_function_l7_726


namespace NUMINAMATH_CALUDE_hiker_distance_l7_765

theorem hiker_distance (hours_day1 : ℝ) : 
  hours_day1 > 0 →
  3 * hours_day1 + 4 * (hours_day1 - 1) + 4 * hours_day1 = 62 →
  3 * hours_day1 = 18 :=
by sorry

end NUMINAMATH_CALUDE_hiker_distance_l7_765


namespace NUMINAMATH_CALUDE_defective_product_probability_l7_701

/-- The probability of drawing a defective product on the second draw,
    given that the first draw was a defective product, when there are
    10 total products, 4 of which are defective, and 2 products are
    drawn successively without replacement. -/
theorem defective_product_probability :
  let total_products : ℕ := 10
  let defective_products : ℕ := 4
  let qualified_products : ℕ := total_products - defective_products
  let first_draw_defective_prob : ℚ := defective_products / total_products
  let second_draw_defective_prob : ℚ :=
    (defective_products - 1) / (total_products - 1)
  let conditional_prob : ℚ :=
    (first_draw_defective_prob * second_draw_defective_prob) / first_draw_defective_prob
  conditional_prob = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_defective_product_probability_l7_701


namespace NUMINAMATH_CALUDE_f_properties_l7_746

/-- The function f(x) = -x³ + ax² + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

/-- The function g(x) = f(x) - ax² + 3 -/
def g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x - a*x^2 + 3

/-- The derivative of f(x) -/
def f_derivative (a b : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x + b

theorem f_properties (a b c : ℝ) :
  (f_derivative a b 1 = -3) ∧  -- Tangent line condition
  (f a b c 1 = -2) ∧          -- Point P(1, f(1)) condition
  (∀ x, g a b c x = -g a b c (-x)) →  -- g(x) is an odd function
  (∃ a' b' c', 
    (∀ x, f a' b' c' x = -x^3 - 2*x^2 + 4*x - 3) ∧
    (∀ x, f a' b' c' x ≥ -11) ∧
    (f a' b' c' (-2) = -11) ∧
    (∀ x, f a' b' c' x ≤ -41/27) ∧
    (f a' b' c' (2/3) = -41/27)) := by sorry

end NUMINAMATH_CALUDE_f_properties_l7_746


namespace NUMINAMATH_CALUDE_pupils_in_singing_only_l7_731

/-- Given a class with pupils in debate and singing activities, calculate the number of pupils in singing only. -/
theorem pupils_in_singing_only
  (total : ℕ)
  (debate_only : ℕ)
  (both : ℕ)
  (h_total : total = 55)
  (h_debate_only : debate_only = 10)
  (h_both : both = 17) :
  total - debate_only - both = 45 :=
by sorry

end NUMINAMATH_CALUDE_pupils_in_singing_only_l7_731


namespace NUMINAMATH_CALUDE_largest_subarray_sum_l7_795

/-- A type representing a 5x5 array of natural numbers -/
def Array5x5 := Fin 5 → Fin 5 → ℕ

/-- Predicate to check if an array contains distinct numbers from 1 to 25 -/
def isValidArray (a : Array5x5) : Prop :=
  ∀ i j, 1 ≤ a i j ∧ a i j ≤ 25 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → a i j ≠ a i' j'

/-- Sum of a 2x2 subarray starting at position (i, j) -/
def subarraySum (a : Array5x5) (i j : Fin 4) : ℕ :=
  a i j + a i (j + 1) + a (i + 1) j + a (i + 1) (j + 1)

/-- Theorem stating that 45 is the largest N satisfying the given property -/
theorem largest_subarray_sum : 
  (∀ a : Array5x5, isValidArray a → ∀ i j : Fin 4, subarraySum a i j ≥ 45) ∧
  ¬(∀ a : Array5x5, isValidArray a → ∀ i j : Fin 4, subarraySum a i j ≥ 46) :=
sorry

end NUMINAMATH_CALUDE_largest_subarray_sum_l7_795


namespace NUMINAMATH_CALUDE_compound_not_uniquely_determined_l7_742

/-- Represents a chemical compound -/
structure Compound where
  elements : List String
  mass_percentages : List Float
  mass_percentage_sum_eq_100 : mass_percentages.sum = 100

/-- A compound contains Cl with a mass percentage of 47.3% -/
def chlorine_compound : Compound := {
  elements := ["Cl", "Unknown"],
  mass_percentages := [47.3, 52.7],
  mass_percentage_sum_eq_100 := by sorry
}

/-- Predicate to check if a compound matches the given chlorine compound -/
def matches_chlorine_compound (c : Compound) : Prop :=
  "Cl" ∈ c.elements ∧ 47.3 ∈ c.mass_percentages

/-- Theorem stating that the compound cannot be uniquely determined -/
theorem compound_not_uniquely_determined :
  ∃ (c1 c2 : Compound), c1 ≠ c2 ∧ matches_chlorine_compound c1 ∧ matches_chlorine_compound c2 :=
by sorry

end NUMINAMATH_CALUDE_compound_not_uniquely_determined_l7_742


namespace NUMINAMATH_CALUDE_perfect_square_solutions_l7_796

theorem perfect_square_solutions : 
  {n : ℤ | ∃ m : ℤ, n^2 + 6*n + 24 = m^2} = {4, -2, -4, -10} := by sorry

end NUMINAMATH_CALUDE_perfect_square_solutions_l7_796


namespace NUMINAMATH_CALUDE_orange_juice_fraction_is_three_tenths_l7_794

/-- Represents the capacity and fill level of a pitcher -/
structure Pitcher where
  capacity : ℚ
  fillLevel : ℚ

/-- Calculates the fraction of orange juice in the mixture -/
def orangeJuiceFraction (pitchers : List Pitcher) : ℚ :=
  let totalJuice := pitchers.foldl (fun acc p => acc + p.capacity * p.fillLevel) 0
  let totalVolume := pitchers.foldl (fun acc p => acc + p.capacity) 0
  totalJuice / totalVolume

/-- Theorem stating that the fraction of orange juice in the mixture is 3/10 -/
theorem orange_juice_fraction_is_three_tenths :
  let pitchers := [
    Pitcher.mk 500 (1/5),
    Pitcher.mk 700 (3/7),
    Pitcher.mk 800 (1/4)
  ]
  orangeJuiceFraction pitchers = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_is_three_tenths_l7_794


namespace NUMINAMATH_CALUDE_circle_constant_l7_744

/-- Theorem: For a circle with equation x^2 + 10x + y^2 + 8y + c = 0 and radius 5, the value of c is 16. -/
theorem circle_constant (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 10*x + y^2 + 8*y + c = 0 ↔ (x+5)^2 + (y+4)^2 = 25) → 
  c = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_constant_l7_744


namespace NUMINAMATH_CALUDE_partition_sum_condition_l7_719

def sum_set (s : Finset Nat) : Nat := s.sum id

theorem partition_sum_condition (k : Nat) :
  (∃ (A B : Finset Nat), A ∩ B = ∅ ∧ A ∪ B = Finset.range k ∧ sum_set A = 2 * sum_set B) ↔
  (∃ m : Nat, m > 0 ∧ (k = 3 * m ∨ k = 3 * m - 1)) :=
by sorry

end NUMINAMATH_CALUDE_partition_sum_condition_l7_719


namespace NUMINAMATH_CALUDE_chips_sales_problem_l7_777

theorem chips_sales_problem (total_sales : ℕ) (first_week : ℕ) (second_week : ℕ) :
  total_sales = 100 →
  first_week = 15 →
  second_week = 3 * first_week →
  ∃ (third_fourth_week : ℕ),
    third_fourth_week * 2 = total_sales - (first_week + second_week) ∧
    third_fourth_week = 20 := by
  sorry

end NUMINAMATH_CALUDE_chips_sales_problem_l7_777


namespace NUMINAMATH_CALUDE_dry_grapes_weight_l7_741

/-- Calculates the weight of dry grapes obtained from fresh grapes -/
theorem dry_grapes_weight
  (fresh_water_content : Real)
  (dry_water_content : Real)
  (fresh_weight : Real)
  (h1 : fresh_water_content = 0.90)
  (h2 : dry_water_content = 0.20)
  (h3 : fresh_weight = 20)
  : Real :=
by
  -- The weight of dry grapes obtained from fresh_weight of fresh grapes
  -- is equal to 2.5
  sorry

#check dry_grapes_weight

end NUMINAMATH_CALUDE_dry_grapes_weight_l7_741


namespace NUMINAMATH_CALUDE_max_min_values_l7_728

theorem max_min_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 2 * y = 10) :
  (Real.sqrt (3 * x) + Real.sqrt (2 * y) ≤ 2 * Real.sqrt 5) ∧
  (3 / x + 2 / y ≥ 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_l7_728


namespace NUMINAMATH_CALUDE_probability_of_q_section_l7_776

/- Define the spinner -/
def spinner_sections : ℕ := 6
def q_sections : ℕ := 2

/- Define the probability function -/
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

/- Theorem statement -/
theorem probability_of_q_section :
  probability q_sections spinner_sections = 2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_q_section_l7_776


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l7_705

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equality :
  lg (4 * Real.sqrt 2 / 7) - lg (2 / 3) + lg (7 * Real.sqrt 5) = lg 6 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l7_705


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l7_720

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 - x + 4

-- State the theorem
theorem decreasing_interval_of_f :
  ∀ x y : ℝ, x ≥ -1/2 → y > x → f y < f x :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l7_720


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l7_723

def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l7_723


namespace NUMINAMATH_CALUDE_population_scientific_notation_l7_752

def population : ℝ := 1411750000

theorem population_scientific_notation : 
  population = 1.41175 * (10 : ℝ) ^ 9 :=
sorry

end NUMINAMATH_CALUDE_population_scientific_notation_l7_752


namespace NUMINAMATH_CALUDE_rectangle_hexagon_apothem_comparison_l7_709

theorem rectangle_hexagon_apothem_comparison :
  ∀ (w l : ℝ) (s : ℝ),
    w > 0 ∧ l > 0 ∧ s > 0 →
    l = 3 * w →
    w * l = 2 * (w + l) →
    3 * Real.sqrt 3 / 2 * s^2 = 6 * s →
    w / 2 = 2/3 * (s * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_hexagon_apothem_comparison_l7_709


namespace NUMINAMATH_CALUDE_ten_percent_of_n_l7_762

theorem ten_percent_of_n (n f : ℝ) (h : n - (1/4 * 2) - (1/3 * 3) - f * n = 27) : 
  (0.1 : ℝ) * n = (0.1 : ℝ) * (28.5 / (1 - f)) := by
sorry

end NUMINAMATH_CALUDE_ten_percent_of_n_l7_762


namespace NUMINAMATH_CALUDE_possible_values_of_a_l7_724

-- Define the sets A and B
def A : Set ℝ := {0, 1}
def B (a : ℝ) : Set ℝ := {x | a * x^2 + x - 1 = 0}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : A ⊇ B a → a = 0 ∨ a < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l7_724


namespace NUMINAMATH_CALUDE_odd_function_implies_a_equals_one_l7_733

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = ax^3 + (a-1)x^2 + x -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + (a - 1) * x^2 + x

theorem odd_function_implies_a_equals_one :
  ∀ a : ℝ, IsOdd (f a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_equals_one_l7_733


namespace NUMINAMATH_CALUDE_only_proposition4_is_correct_l7_711

-- Define the propositions
def proposition1 : Prop := ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) - a n = 0) → (∃ r, ∀ n, a (n + 1) = r * a n)
def proposition2 : Prop := ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) = (1/2) * a n) → (∀ n, a (n + 1) < a n)
def proposition3 : Prop := ∀ a b c : ℝ, (b^2 = a * c) ↔ (∃ r, b = a * r ∧ c = b * r)
def proposition4 : Prop := ∀ a b c : ℝ, (2 * b = a + c) ↔ (∃ d, b = a + d ∧ c = b + d)

-- Theorem statement
theorem only_proposition4_is_correct :
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 :=
sorry

end NUMINAMATH_CALUDE_only_proposition4_is_correct_l7_711


namespace NUMINAMATH_CALUDE_sum_reciprocal_squared_bound_l7_788

theorem sum_reciprocal_squared_bound (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) 
  (h_sum : x₁ + x₂ + x₃ = 1) : 
  (1 / (1 + x₁^2)) + (1 / (1 + x₂^2)) + (1 / (1 + x₃^2)) ≤ 27/10 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squared_bound_l7_788


namespace NUMINAMATH_CALUDE_dice_roll_probability_l7_773

def probability_first_die : ℚ := 3 / 8
def probability_second_die : ℚ := 3 / 4

theorem dice_roll_probability :
  probability_first_die * probability_second_die = 9 / 32 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l7_773


namespace NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_three_quarters_l7_713

theorem mean_of_five_numbers_with_sum_three_quarters
  (a b c d e : ℝ) (h : a + b + c + d + e = 3/4) :
  (a + b + c + d + e) / 5 = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_three_quarters_l7_713


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l7_721

theorem perpendicular_lines_slope (a : ℝ) : 
  (∃ x y : ℝ, y = a * x - 2) ∧ 
  (∃ x y : ℝ, y = 2 * x + 1) ∧ 
  (a * 2 = -1) → 
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l7_721


namespace NUMINAMATH_CALUDE_log_equation_equivalence_l7_756

-- Define the logarithm function with base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_equivalence (x : ℝ) (h : x > 0) :
  lg x ^ 2 + lg (x ^ 2) = 0 ↔ lg x ^ 2 + 2 * lg x = 0 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_equivalence_l7_756


namespace NUMINAMATH_CALUDE_pauls_cousin_score_l7_704

/-- Given Paul's score and the total score of Paul and his cousin, 
    calculate Paul's cousin's score. -/
theorem pauls_cousin_score (paul_score total_score : ℕ) 
  (h1 : paul_score = 3103)
  (h2 : total_score = 5816) :
  total_score - paul_score = 2713 := by
  sorry

end NUMINAMATH_CALUDE_pauls_cousin_score_l7_704


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_5_m_range_when_solution_set_is_real_l7_749

def f (x m : ℝ) : ℝ := |x + 1| + |x - 2| - m

theorem solution_set_when_m_is_5 :
  {x : ℝ | f x 5 > 0} = {x : ℝ | x < -2 ∨ x > 3} :=
sorry

theorem m_range_when_solution_set_is_real :
  (∀ x : ℝ, f x m ≥ 2) → m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_5_m_range_when_solution_set_is_real_l7_749


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l7_759

theorem negative_fractions_comparison : -3/4 < -2/3 := by sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l7_759


namespace NUMINAMATH_CALUDE_machine_output_l7_747

/-- The number of shirts an industrial machine can make in a minute. -/
def shirts_per_minute : ℕ := sorry

/-- The number of minutes the machine worked today. -/
def minutes_worked_today : ℕ := 12

/-- The total number of shirts made today. -/
def total_shirts_today : ℕ := 72

/-- Theorem stating that the machine can make 6 shirts per minute. -/
theorem machine_output : shirts_per_minute = 6 := by
  sorry

end NUMINAMATH_CALUDE_machine_output_l7_747


namespace NUMINAMATH_CALUDE_nba_scheduling_impossibility_l7_712

theorem nba_scheduling_impossibility :
  ∀ (k : ℕ) (x y z : ℕ),
    k ≤ 30 ∧
    x + y + z = 1230 ∧
    82 * k = 2 * x + z →
    z ≠ (x + y + z) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_nba_scheduling_impossibility_l7_712


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l7_757

/-- A function f: ℝ → ℝ satisfying f(x+y) ≤ f(x²+y) for all x, y ∈ ℝ is constant. -/
theorem function_satisfying_inequality_is_constant (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) ≤ f (x^2 + y)) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l7_757
