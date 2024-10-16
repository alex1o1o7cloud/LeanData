import Mathlib

namespace NUMINAMATH_CALUDE_adjacent_probability_l3688_368888

/-- The number of seats in the arrangement -/
def total_seats : ℕ := 9

/-- The number of students to be seated -/
def num_students : ℕ := 8

/-- The number of rows in the seating arrangement -/
def num_rows : ℕ := 3

/-- The number of columns in the seating arrangement -/
def num_columns : ℕ := 3

/-- Calculate the total number of possible seating arrangements -/
def total_arrangements : ℕ := Nat.factorial total_seats

/-- Calculate the number of favorable arrangements where Abby and Bridget are adjacent -/
def favorable_arrangements : ℕ :=
  (num_rows * (num_columns - 1) + num_columns * (num_rows - 1)) * 2 * Nat.factorial (num_students - 1)

/-- The probability that Abby and Bridget are adjacent in the same row or column -/
theorem adjacent_probability :
  (favorable_arrangements : ℚ) / total_arrangements = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_probability_l3688_368888


namespace NUMINAMATH_CALUDE_bus_tour_sales_l3688_368878

theorem bus_tour_sales (total_tickets : ℕ) (senior_price regular_price : ℕ) (regular_tickets : ℕ)
  (h1 : total_tickets = 65)
  (h2 : senior_price = 10)
  (h3 : regular_price = 15)
  (h4 : regular_tickets = 41) :
  (total_tickets - regular_tickets) * senior_price + regular_tickets * regular_price = 855 := by
  sorry

end NUMINAMATH_CALUDE_bus_tour_sales_l3688_368878


namespace NUMINAMATH_CALUDE_uniform_random_transformation_l3688_368805

/-- A uniform random variable on an interval -/
def UniformRandom (a b : ℝ) (X : ℝ → Prop) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → X x

theorem uniform_random_transformation (b₁ : ℝ → Prop) (b : ℝ → Prop) :
  UniformRandom 0 1 b₁ →
  (∀ x, b x ↔ ∃ y, b₁ y ∧ x = 3 * (y - 2)) →
  UniformRandom (-6) (-3) b :=
sorry

end NUMINAMATH_CALUDE_uniform_random_transformation_l3688_368805


namespace NUMINAMATH_CALUDE_purchase_cost_l3688_368824

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda in dollars -/
def soda_cost : ℕ := 1

/-- The number of sandwiches to purchase -/
def num_sandwiches : ℕ := 6

/-- The number of sodas to purchase -/
def num_sodas : ℕ := 10

/-- The total cost of the purchase in dollars -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem purchase_cost : total_cost = 34 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l3688_368824


namespace NUMINAMATH_CALUDE_existence_of_special_multiple_l3688_368876

theorem existence_of_special_multiple : ∃ n : ℕ,
  (n % 2020 = 0) ∧
  (∀ d : Fin 10, ∃! pos : ℕ, 
    (n / 10^pos % 10 : Fin 10) = d) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_multiple_l3688_368876


namespace NUMINAMATH_CALUDE_equation_implication_l3688_368811

theorem equation_implication (x y : ℝ) :
  x^2 - 3*x*y + 2*y^2 + x - y = 0 →
  x^2 - 2*x*y + y^2 - 5*x + 7*y = 0 →
  x*y - 12*x + 15*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_implication_l3688_368811


namespace NUMINAMATH_CALUDE_min_product_of_three_l3688_368829

def S : Finset Int := {-9, -5, -1, 1, 3, 5, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → a * b * c ≤ x * y * z) ∧ 
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = -360) :=
by sorry

end NUMINAMATH_CALUDE_min_product_of_three_l3688_368829


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l3688_368870

theorem simplify_product_of_square_roots (x : ℝ) (hx : x > 0) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (28 * x) * Real.sqrt (5 * x) = 60 * x^2 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l3688_368870


namespace NUMINAMATH_CALUDE_max_sum_squared_sum_l3688_368836

theorem max_sum_squared_sum (a b c : ℝ) (h : a + b + c = a^2 + b^2 + c^2) :
  a + b + c ≤ 3 ∧ ∃ x y z : ℝ, x + y + z = x^2 + y^2 + z^2 ∧ x + y + z = 3 :=
sorry

end NUMINAMATH_CALUDE_max_sum_squared_sum_l3688_368836


namespace NUMINAMATH_CALUDE_max_value_abc_inverse_sum_cubed_l3688_368864

theorem max_value_abc_inverse_sum_cubed (a b c : ℝ) (h : a + b + c = 0) :
  abc * (1/a + 1/b + 1/c)^3 ≤ 27/8 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_inverse_sum_cubed_l3688_368864


namespace NUMINAMATH_CALUDE_circle_areas_theorem_l3688_368875

theorem circle_areas_theorem (r : ℝ) (h1 : r > 0) (h2 : π * r^2 = 81 * π) :
  let small_r := r / 2
  (1/3) * (π * r^2) + (1/3) * (π * small_r^2) = 33.75 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_areas_theorem_l3688_368875


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_198_l3688_368816

theorem first_nonzero_digit_after_decimal_1_198 : ∃ (n : ℕ) (d : ℕ), 
  1 ≤ d ∧ d ≤ 9 ∧ 
  (∃ (m : ℕ), 1/198 = (n : ℚ)/10^m + d/(10^(m+1) : ℚ) + (1/198 - (n : ℚ)/10^m - d/(10^(m+1) : ℚ)) ∧ 
   0 ≤ 1/198 - (n : ℚ)/10^m - d/(10^(m+1) : ℚ) ∧ 
   1/198 - (n : ℚ)/10^m - d/(10^(m+1) : ℚ) < 1/(10^(m+1) : ℚ)) ∧
  d = 5 :=
by sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_198_l3688_368816


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3688_368803

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) * (n + 2) = 1680 → (n - 1) + n + (n + 1) + (n + 2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3688_368803


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3688_368830

theorem remainder_divisibility (x : ℤ) : x % 8 = 3 → x % 72 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3688_368830


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l3688_368818

theorem unique_positive_integer_solution : 
  ∃! (z : ℕ), z > 0 ∧ (4 * z)^2 - z = 2345 :=
by
  use 7
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l3688_368818


namespace NUMINAMATH_CALUDE_gcd_problem_l3688_368899

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 528 * k) : 
  Nat.gcd (Int.natAbs (2*b^4 + b^3 + 5*b^2 + 6*b + 132)) (Int.natAbs b) = 132 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3688_368899


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l3688_368809

/-- The number of real values of b for which the line y = 2x + b passes through the vertex of the parabola y = x^2 + b^2 + 1 is zero. -/
theorem line_through_parabola_vertex (b : ℝ) : ¬∃ b, 2 * 0 + b = 0^2 + b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l3688_368809


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l3688_368882

/-- Represents a trapezoid ABCD with sides AB and CD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ

/-- The theorem stating the relationship between the sides of the trapezoid
    given the area ratio of triangles ABC and ADC -/
theorem trapezoid_side_length (ABCD : Trapezoid)
    (h1 : (ABCD.AB / ABCD.CD) = (7 : ℝ) / 3)
    (h2 : ABCD.AB + ABCD.CD = 210) :
    ABCD.AB = 147 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l3688_368882


namespace NUMINAMATH_CALUDE_school_classes_count_l3688_368895

/-- Represents a school with classes -/
structure School where
  total_students : ℕ
  largest_class : ℕ
  class_difference : ℕ

/-- Calculates the number of classes in a school -/
def number_of_classes (s : School) : ℕ :=
  sorry

/-- Theorem stating that for a school with 120 students, largest class of 28, 
    and class difference of 2, the number of classes is 5 -/
theorem school_classes_count (s : School) 
  (h1 : s.total_students = 120) 
  (h2 : s.largest_class = 28) 
  (h3 : s.class_difference = 2) : 
  number_of_classes s = 5 := by
  sorry

end NUMINAMATH_CALUDE_school_classes_count_l3688_368895


namespace NUMINAMATH_CALUDE_f_max_value_l3688_368889

/-- The quadratic function f(x) = -3x^2 + 6x + 4 --/
def f (x : ℝ) : ℝ := -3 * x^2 + 6 * x + 4

/-- The maximum value of f(x) over all real numbers x --/
def max_value : ℝ := 7

/-- Theorem stating that the maximum value of f(x) is 7 --/
theorem f_max_value : ∀ x : ℝ, f x ≤ max_value := by sorry

end NUMINAMATH_CALUDE_f_max_value_l3688_368889


namespace NUMINAMATH_CALUDE_matching_color_probability_l3688_368810

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ

/-- Calculates the total number of jelly beans a person has -/
def total_jelly_beans (jb : JellyBeans) : ℕ := jb.green + jb.red + jb.blue

/-- Abe's jelly beans -/
def abe_jb : JellyBeans := { green := 2, red := 3, blue := 0 }

/-- Bob's jelly beans -/
def bob_jb : JellyBeans := { green := 2, red := 3, blue := 2 }

/-- Calculates the probability of picking a specific color -/
def prob_color (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / (total_jelly_beans jb)

/-- Theorem: The probability of Abe and Bob showing the same color is 13/35 -/
theorem matching_color_probability : 
  (prob_color abe_jb abe_jb.green * prob_color bob_jb bob_jb.green) +
  (prob_color abe_jb abe_jb.red * prob_color bob_jb bob_jb.red) = 13/35 := by
  sorry

end NUMINAMATH_CALUDE_matching_color_probability_l3688_368810


namespace NUMINAMATH_CALUDE_exponent_equality_l3688_368835

theorem exponent_equality 
  (a b c d : ℝ) 
  (x y q z : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b) 
  (h3 : c^(2*y) = a^(3*z)) 
  (h4 : c^(2*y) = d) 
  (h5 : a ≠ 0) 
  (h6 : c ≠ 0) : 
  2*x * 3*z = 3*q * 2*y := by
sorry

end NUMINAMATH_CALUDE_exponent_equality_l3688_368835


namespace NUMINAMATH_CALUDE_f_1_eq_0_f_increasing_f_inequality_solution_l3688_368877

noncomputable section

variable (f : ℝ → ℝ)

axiom domain : ∀ x, x > 0 → f x ≠ 0 → True

axiom f_4 : f 4 = 1

axiom f_product : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

axiom f_neg_on_unit : ∀ x, 0 < x → x < 1 → f x < 0

theorem f_1_eq_0 : f 1 = 0 := by sorry

theorem f_increasing : ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by sorry

theorem f_inequality_solution : 
  ∀ x, x > 0 → (f (3 * x + 1) + f (2 * x - 6) ≤ 3 ↔ 3 < x ∧ x ≤ 5) := by sorry

end

end NUMINAMATH_CALUDE_f_1_eq_0_f_increasing_f_inequality_solution_l3688_368877


namespace NUMINAMATH_CALUDE_rectangular_field_width_l3688_368845

/-- Given a rectangular field with perimeter 240 meters and perimeter equal to 3 times its length, prove that its width is 40 meters. -/
theorem rectangular_field_width (length width : ℝ) : 
  (2 * length + 2 * width = 240) →  -- Perimeter formula
  (240 = 3 * length) →              -- Perimeter is 3 times length
  width = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l3688_368845


namespace NUMINAMATH_CALUDE_division_in_ratio_l3688_368817

theorem division_in_ratio (total : ℕ) (x_ratio y_ratio : ℕ) (x_amount : ℕ) : 
  total = 5000 → 
  x_ratio = 2 → 
  y_ratio = 8 → 
  x_amount = total * x_ratio / (x_ratio + y_ratio) → 
  x_amount = 1000 := by
sorry

end NUMINAMATH_CALUDE_division_in_ratio_l3688_368817


namespace NUMINAMATH_CALUDE_regular_octahedron_faces_regular_octahedron_has_eight_faces_l3688_368890

/-- A regular octahedron is a Platonic solid with equilateral triangular faces. -/
structure RegularOctahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of faces of a regular octahedron is 8. -/
theorem regular_octahedron_faces (o : RegularOctahedron) : Nat :=
  8

/-- Prove that a regular octahedron has 8 faces. -/
theorem regular_octahedron_has_eight_faces (o : RegularOctahedron) :
  regular_octahedron_faces o = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_octahedron_faces_regular_octahedron_has_eight_faces_l3688_368890


namespace NUMINAMATH_CALUDE_forum_posts_per_day_l3688_368872

/-- A forum with questions and answers -/
structure Forum where
  members : ℕ
  questionsPerHour : ℕ
  answerRatio : ℕ

/-- Calculate the total posts (questions and answers) per day -/
def totalPostsPerDay (f : Forum) : ℕ :=
  let questionsPerDay := f.members * (f.questionsPerHour * 24)
  let answersPerDay := questionsPerDay * f.answerRatio
  questionsPerDay + answersPerDay

/-- Theorem: Given the conditions, the forum has 57600 posts per day -/
theorem forum_posts_per_day :
  ∀ (f : Forum),
    f.members = 200 →
    f.questionsPerHour = 3 →
    f.answerRatio = 3 →
    totalPostsPerDay f = 57600 := by
  sorry

end NUMINAMATH_CALUDE_forum_posts_per_day_l3688_368872


namespace NUMINAMATH_CALUDE_ratio_sum_equality_l3688_368860

theorem ratio_sum_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 16)
  (sum_xyz : x^2 + y^2 + z^2 = 49)
  (sum_prod : a*x + b*y + c*z = 28) :
  (a + b + c) / (x + y + z) = 4/7 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_equality_l3688_368860


namespace NUMINAMATH_CALUDE_binomial_16_12_l3688_368843

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_binomial_16_12_l3688_368843


namespace NUMINAMATH_CALUDE_intersection_condition_l3688_368856

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem intersection_condition (a : ℝ) : A ∩ B a = B a → a = 0 ∨ a = 1/2 ∨ a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3688_368856


namespace NUMINAMATH_CALUDE_equation_one_solutions_l3688_368827

theorem equation_one_solutions (x : ℝ) : (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l3688_368827


namespace NUMINAMATH_CALUDE_existence_of_m_n_l3688_368880

theorem existence_of_m_n (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m + n ≤ (p + 1) / 2 ∧ p ∣ 2^n * 3^m - 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l3688_368880


namespace NUMINAMATH_CALUDE_amusement_park_admission_l3688_368898

/-- The number of children admitted to an amusement park -/
def num_children : ℕ := 180

/-- The number of adults admitted to an amusement park -/
def num_adults : ℕ := 315 - num_children

/-- The admission fee for children in dollars -/
def child_fee : ℚ := 3/2

/-- The admission fee for adults in dollars -/
def adult_fee : ℚ := 4

/-- The total number of people admitted to the park -/
def total_people : ℕ := 315

/-- The total admission fees collected in dollars -/
def total_fees : ℚ := 810

theorem amusement_park_admission :
  (num_children : ℚ) * child_fee + (num_adults : ℚ) * adult_fee = total_fees ∧
  num_children + num_adults = total_people :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_admission_l3688_368898


namespace NUMINAMATH_CALUDE_triangle_side_length_l3688_368861

/-- In a triangle ABC, given that angle C is four times angle A, 
    side a is 35, and side c is 64, prove that side b equals 140cos²A -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  C = 4 * A ∧              -- Angle C is four times angle A
  a = 35 ∧                 -- Side a is 35
  c = 64 →                 -- Side c is 64
  b = 140 * (Real.cos A)^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3688_368861


namespace NUMINAMATH_CALUDE_fourth_term_is_eight_l3688_368885

/-- An arithmetic progression with the given property -/
def ArithmeticProgression (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = S n + 1

theorem fourth_term_is_eight
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : ArithmeticProgression a S) :
  a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_eight_l3688_368885


namespace NUMINAMATH_CALUDE_factor_value_theorem_l3688_368883

theorem factor_value_theorem (m n : ℚ) : 
  (∀ x : ℚ, (x - 3) * (x + 1) ∣ (3 * x^4 - m * x^2 + n * x - 5)) → 
  |3 * m - 2 * n| = 302 / 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_value_theorem_l3688_368883


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3688_368857

theorem polynomial_factorization :
  ∀ (x y a b : ℝ),
    (12 * x^3 * y - 3 * x * y^2 = 3 * x * y * (4 * x^2 - y)) ∧
    (x - 9 * x^3 = x * (1 + 3 * x) * (1 - 3 * x)) ∧
    (3 * a^2 - 12 * a * b * (a - b) = 3 * (a - 2 * b)^2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3688_368857


namespace NUMINAMATH_CALUDE_leahs_coins_value_l3688_368826

/-- Represents the number and value of coins Leah has -/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total number of coins -/
def CoinCollection.total (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes

/-- Calculates the total value of coins in cents -/
def CoinCollection.value (c : CoinCollection) : ℕ :=
  c.pennies + 5 * c.nickels + 10 * c.dimes

/-- Theorem stating that Leah's coins are worth 66 cents -/
theorem leahs_coins_value (c : CoinCollection) : c.value = 66 :=
  by
  have h1 : c.total = 15 := sorry
  have h2 : c.pennies = c.nickels + 3 := sorry
  sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l3688_368826


namespace NUMINAMATH_CALUDE_adam_apples_l3688_368844

theorem adam_apples (x : ℕ) : 
  x + 3 * x + 12 * x = 240 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_adam_apples_l3688_368844


namespace NUMINAMATH_CALUDE_pqr_product_l3688_368896

theorem pqr_product (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h_eq : p^2 + 2/q = q^2 + 2/r ∧ q^2 + 2/r = r^2 + 2/p) :
  |p * q * r| = 2 := by
  sorry

end NUMINAMATH_CALUDE_pqr_product_l3688_368896


namespace NUMINAMATH_CALUDE_simplify_fraction_l3688_368847

theorem simplify_fraction : (5^4 + 5^2) / (5^3 - 5) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3688_368847


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l3688_368852

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l3688_368852


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3688_368866

theorem complex_equation_solution (z : ℂ) : (z - Complex.I) * Complex.I = 2 + Complex.I → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3688_368866


namespace NUMINAMATH_CALUDE_sum_and_product_implications_l3688_368800

theorem sum_and_product_implications (a b : ℝ) 
  (h1 : a + b = 2) (h2 : a * b = -1) : 
  a^2 + b^2 = 6 ∧ (a - b)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_implications_l3688_368800


namespace NUMINAMATH_CALUDE_number_of_cars_l3688_368869

theorem number_of_cars (total_distance : ℝ) (car_spacing : ℝ) (h1 : total_distance = 242) (h2 : car_spacing = 5.5) :
  ⌊total_distance / car_spacing⌋ + 1 = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cars_l3688_368869


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l3688_368891

theorem quadratic_equation_result (m : ℝ) (h : 2 * m^2 + m = -1) : 4 * m^2 + 2 * m + 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l3688_368891


namespace NUMINAMATH_CALUDE_number_puzzle_l3688_368859

theorem number_puzzle (N : ℝ) : 6 + (1/2) * (1/3) * (1/5) * N = (1/15) * N → N = 180 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3688_368859


namespace NUMINAMATH_CALUDE_remainder_problem_l3688_368894

theorem remainder_problem (x : ℤ) (h : (x + 13) % 41 = 18) : x % 82 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3688_368894


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l3688_368807

theorem real_part_of_complex_number (z : ℂ) (a : ℝ) :
  z = (1 : ℂ) / (1 + I) + a * I → z.im = 0 → a = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l3688_368807


namespace NUMINAMATH_CALUDE_triathlon_speeds_correct_l3688_368840

/-- Represents the minimum speeds required for Maria to complete the triathlon within the given time limit. -/
def triathlon_speeds (swim_dist : ℝ) (cycle_dist : ℝ) (run_dist : ℝ) (time_limit : ℝ) : ℝ × ℝ × ℝ :=
  let swim_speed : ℝ := 60
  let run_speed : ℝ := 3 * swim_speed
  let cycle_speed : ℝ := 2.5 * run_speed
  (swim_speed, cycle_speed, run_speed)

/-- Theorem stating that the calculated speeds are correct for the given triathlon conditions. -/
theorem triathlon_speeds_correct 
  (swim_dist : ℝ) (cycle_dist : ℝ) (run_dist : ℝ) (time_limit : ℝ)
  (h_swim : swim_dist = 800)
  (h_cycle : cycle_dist = 20000)
  (h_run : run_dist = 4000)
  (h_time : time_limit = 80) :
  let (swim_speed, cycle_speed, run_speed) := triathlon_speeds swim_dist cycle_dist run_dist time_limit
  swim_speed = 60 ∧ cycle_speed = 450 ∧ run_speed = 180 ∧
  swim_dist / swim_speed + cycle_dist / cycle_speed + run_dist / run_speed ≤ time_limit :=
by sorry

#check triathlon_speeds_correct

end NUMINAMATH_CALUDE_triathlon_speeds_correct_l3688_368840


namespace NUMINAMATH_CALUDE_inverse_sum_reciprocal_l3688_368887

theorem inverse_sum_reciprocal (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (x * z + y * z + x * y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocal_l3688_368887


namespace NUMINAMATH_CALUDE_second_to_tallest_ratio_l3688_368865

/-- The heights of four buildings satisfying certain conditions -/
structure BuildingHeights where
  t : ℝ  -- height of the tallest building
  s : ℝ  -- height of the second tallest building
  u : ℝ  -- height of the third tallest building
  v : ℝ  -- height of the fourth tallest building
  h1 : t = 100  -- the tallest building is 100 feet tall
  h2 : u = s / 2  -- the third tallest is half as tall as the second
  h3 : v = u / 5  -- the fourth is one-fifth as tall as the third
  h4 : t + s + u + v = 180  -- all 4 buildings together are 180 feet tall

/-- The ratio of the second tallest building to the tallest is 1:2 -/
theorem second_to_tallest_ratio (b : BuildingHeights) : b.s / b.t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_second_to_tallest_ratio_l3688_368865


namespace NUMINAMATH_CALUDE_gold_copper_alloy_density_l3688_368801

/-- The density of gold relative to water -/
def gold_density : ℝ := 10

/-- The density of copper relative to water -/
def copper_density : ℝ := 5

/-- The desired density of the alloy relative to water -/
def alloy_density : ℝ := 9

/-- The ratio of gold to copper in the alloy -/
def gold_copper_ratio : ℝ := 4

theorem gold_copper_alloy_density :
  ∀ (g c : ℝ),
  g > 0 → c > 0 →
  g / c = gold_copper_ratio →
  (gold_density * g + copper_density * c) / (g + c) = alloy_density :=
by sorry

end NUMINAMATH_CALUDE_gold_copper_alloy_density_l3688_368801


namespace NUMINAMATH_CALUDE_morning_afternoon_email_difference_l3688_368892

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 5

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 16

/-- The theorem states that Jack received 2 more emails in the morning than in the afternoon -/
theorem morning_afternoon_email_difference : morning_emails - afternoon_emails = 2 := by
  sorry

end NUMINAMATH_CALUDE_morning_afternoon_email_difference_l3688_368892


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3688_368834

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | x > -1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 4} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3688_368834


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l3688_368821

theorem restaurant_bill_theorem (num_people : ℕ) (total_bill : ℚ) (gratuity_rate : ℚ) :
  num_people = 7 →
  total_bill = 840 →
  gratuity_rate = 1/5 →
  (total_bill / (1 + gratuity_rate)) / num_people = 100 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l3688_368821


namespace NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l3688_368819

theorem fourth_root_over_sixth_root_of_seven (x : ℝ) :
  (7 : ℝ) ^ (1 / 4) / (7 : ℝ) ^ (1 / 6) = (7 : ℝ) ^ (1 / 12) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l3688_368819


namespace NUMINAMATH_CALUDE_arithmetic_comparisons_l3688_368832

theorem arithmetic_comparisons :
  (80 / 4 > 80 / 5) ∧
  (16 * 21 > 14 * 22) ∧
  (32 * 25 = 16 * 50) ∧
  (320 / 8 < 320 / 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_comparisons_l3688_368832


namespace NUMINAMATH_CALUDE_max_individual_score_l3688_368822

theorem max_individual_score (n : ℕ) (total_score : ℕ) (min_score : ℕ) 
  (h1 : n = 12)
  (h2 : total_score = 100)
  (h3 : min_score = 7)
  (h4 : ∀ player, player ∈ Finset.range n → player ≥ min_score) :
  (total_score - (n - 1) * min_score) = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_individual_score_l3688_368822


namespace NUMINAMATH_CALUDE_lcm_sum_implies_product_div_3_or_5_l3688_368825

theorem lcm_sum_implies_product_div_3_or_5 (a b c d : ℕ) :
  Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = a + b + c + d →
  3 ∣ (a * b * c * d) ∨ 5 ∣ (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_lcm_sum_implies_product_div_3_or_5_l3688_368825


namespace NUMINAMATH_CALUDE_john_nada_money_multiple_l3688_368804

/-- Given the money distribution among Ali, Nada, and John, prove that John has 4 times Nada's amount. -/
theorem john_nada_money_multiple (total : ℕ) (john_money : ℕ) (nada_money : ℕ) :
  total = 67 →
  john_money = 48 →
  nada_money + (nada_money - 5) + john_money = total →
  john_money = 4 * nada_money :=
by sorry

end NUMINAMATH_CALUDE_john_nada_money_multiple_l3688_368804


namespace NUMINAMATH_CALUDE_sqrt_equality_l3688_368893

theorem sqrt_equality : ∃ (a b : ℕ+), a < b ∧ Real.sqrt (1 + Real.sqrt (45 + 18 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_l3688_368893


namespace NUMINAMATH_CALUDE_expression_simplification_l3688_368808

theorem expression_simplification (y : ℝ) : 
  3 * y + 4 * y^2 - 2 - (7 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3688_368808


namespace NUMINAMATH_CALUDE_triangle_side_length_l3688_368897

theorem triangle_side_length (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BD := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  AB = 30 →
  (B.1 - D.1) * (A.1 - D.1) + (B.2 - D.2) * (A.2 - D.2) = 0 →
  (A.2 - D.2) / AB = 4/5 →
  BD / BC = 1/5 →
  C.2 = D.2 →
  C.1 > D.1 →
  CD = 24 * Real.sqrt 23 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3688_368897


namespace NUMINAMATH_CALUDE_inequality_proof_l3688_368871

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b + 3/4) * (b^2 + c + 3/4) * (c^2 + a + 3/4) ≥ (2*a + 1/2) * (2*b + 1/2) * (2*c + 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3688_368871


namespace NUMINAMATH_CALUDE_max_value_product_l3688_368862

open Real

-- Define the function f(x) = ln(x+2) - x
noncomputable def f (x : ℝ) : ℝ := log (x + 2) - x

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := 1 / (x + 2) - 1

theorem max_value_product (a b : ℝ) : f' a = 0 → f a = b → a * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_product_l3688_368862


namespace NUMINAMATH_CALUDE_solution_set_inequality_holds_max_m_value_l3688_368820

-- Define the function f
def f (x : ℝ) := |2*x + 1| + |3*x - 2|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -4/5 ≤ x ∧ x ≤ 6/5} :=
sorry

-- Theorem for the inequality |x-1| + |x+2| ≥ 3
theorem inequality_holds (x : ℝ) :
  |x - 1| + |x + 2| ≥ 3 :=
sorry

-- Theorem for the maximum value of m
theorem max_m_value :
  ∃ (m : ℝ), m = 2 ∧ 
  (∀ (x : ℝ), m^2 - 3*m + 5 ≤ |x - 1| + |x + 2|) ∧
  (∀ (m' : ℝ), (∀ (x : ℝ), m'^2 - 3*m' + 5 ≤ |x - 1| + |x + 2|) → m' ≤ m) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_holds_max_m_value_l3688_368820


namespace NUMINAMATH_CALUDE_right_pyramid_base_side_l3688_368854

/-- Represents a right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- Theorem: If the area of one lateral face of a right pyramid with a square base is 200 square meters
    and the slant height is 40 meters, then the length of the side of its base is 10 meters. -/
theorem right_pyramid_base_side (p : RightPyramid) 
  (h1 : p.lateral_face_area = 200)
  (h2 : p.slant_height = 40) : 
  p.base_side = 10 := by
  sorry

#check right_pyramid_base_side

end NUMINAMATH_CALUDE_right_pyramid_base_side_l3688_368854


namespace NUMINAMATH_CALUDE_impossible_visit_all_squares_l3688_368858

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Fin 6)
  (y : Fin 6)

/-- Represents a jump on the chessboard -/
inductive Jump
  | One
  | Two

/-- Represents a sequence of jumps -/
def JumpSequence := List Jump

/-- Checks if a jump sequence is valid (alternating One and Two) -/
def isValidJumpSequence : JumpSequence → Bool
  | [] => true
  | [_] => true
  | Jump.One :: Jump.Two :: rest => isValidJumpSequence rest
  | Jump.Two :: Jump.One :: rest => isValidJumpSequence rest
  | _ => false

/-- Applies a jump to a position -/
def applyJump (pos : Position) (jump : Jump) (direction : Bool) : Position :=
  match jump, direction with
  | Jump.One, true => ⟨pos.x + 1, pos.y⟩
  | Jump.One, false => ⟨pos.x, pos.y + 1⟩
  | Jump.Two, true => ⟨pos.x + 2, pos.y⟩
  | Jump.Two, false => ⟨pos.x, pos.y + 2⟩

/-- Applies a sequence of jumps to a position -/
def applyJumpSequence (pos : Position) (jumps : JumpSequence) (directions : List Bool) : List Position :=
  match jumps, directions with
  | [], _ => [pos]
  | j :: js, d :: ds => pos :: applyJumpSequence (applyJump pos j d) js ds
  | _, _ => [pos]

/-- Theorem: It's impossible to visit all squares on a 6x6 chessboard
    with 35 jumps alternating between 1 and 2 squares -/
theorem impossible_visit_all_squares :
  ∀ (start : Position) (jumps : JumpSequence) (directions : List Bool),
    isValidJumpSequence jumps →
    jumps.length = 35 →
    directions.length = 35 →
    ¬(∀ (p : Position), p ∈ applyJumpSequence start jumps directions) :=
by
  sorry

end NUMINAMATH_CALUDE_impossible_visit_all_squares_l3688_368858


namespace NUMINAMATH_CALUDE_midpoint_of_intersection_l3688_368849

-- Define the line
def line (x y : ℝ) : Prop := x - y = 2

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ parabola A.1 A.2 ∧
  line B.1 B.2 ∧ parabola B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem midpoint_of_intersection :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  (A.1 + B.1) / 2 = 4 ∧ (A.2 + B.2) / 2 = 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_of_intersection_l3688_368849


namespace NUMINAMATH_CALUDE_complex_equal_parts_l3688_368833

theorem complex_equal_parts (a : ℝ) :
  let z : ℂ := a - 2 * Complex.I
  z.re = z.im → a = -2 := by sorry

end NUMINAMATH_CALUDE_complex_equal_parts_l3688_368833


namespace NUMINAMATH_CALUDE_stratified_sampling_grade10_l3688_368867

theorem stratified_sampling_grade10 (total_students : ℕ) (sample_size : ℕ) (grade10_in_sample : ℕ) :
  total_students = 1800 →
  sample_size = 90 →
  grade10_in_sample = 42 →
  (grade10_in_sample : ℚ) / (sample_size : ℚ) = (840 : ℚ) / (total_students : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_grade10_l3688_368867


namespace NUMINAMATH_CALUDE_romeo_chocolate_bars_l3688_368838

theorem romeo_chocolate_bars : 
  ∀ (buy_cost sell_total packaging_cost profit num_bars : ℕ),
    buy_cost = 5 →
    sell_total = 90 →
    packaging_cost = 2 →
    profit = 55 →
    num_bars * (buy_cost + packaging_cost) + profit = sell_total →
    num_bars = 5 := by
  sorry

end NUMINAMATH_CALUDE_romeo_chocolate_bars_l3688_368838


namespace NUMINAMATH_CALUDE_A_initial_investment_l3688_368851

/-- Represents the initial investment of A in dollars -/
def A_investment : ℝ := sorry

/-- Represents B's investment in dollars -/
def B_investment : ℝ := 9000

/-- Represents the number of months A invested -/
def A_months : ℕ := 12

/-- Represents the number of months B invested -/
def B_months : ℕ := 7

/-- Represents A's share in the profit ratio -/
def A_ratio : ℕ := 2

/-- Represents B's share in the profit ratio -/
def B_ratio : ℕ := 3

theorem A_initial_investment :
  A_investment * A_months * B_ratio = B_investment * B_months * A_ratio :=
sorry

end NUMINAMATH_CALUDE_A_initial_investment_l3688_368851


namespace NUMINAMATH_CALUDE_product_selection_probability_l3688_368802

/-- Given a set of products with some being first-class, this function calculates
    the probability that one of two randomly selected products is not first-class,
    given that one of them is first-class. -/
def conditional_probability (total : ℕ) (first_class : ℕ) : ℚ :=
  let not_first_class := total - first_class
  let total_combinations := (total.choose 2 : ℚ)
  let one_not_first_class := (first_class * not_first_class : ℚ)
  one_not_first_class / total_combinations

/-- The theorem states that for 8 total products with 6 being first-class,
    the conditional probability of selecting one non-first-class product
    given that one first-class product is selected is 12/13. -/
theorem product_selection_probability :
  conditional_probability 8 6 = 12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_product_selection_probability_l3688_368802


namespace NUMINAMATH_CALUDE_linear_function_constraint_l3688_368812

/-- A linear function y = kx + b -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

/-- Predicate to check if a point (x, y) is in the third quadrant -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Predicate to check if a point (x, y) is the origin -/
def is_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0

/-- Theorem stating that if a linear function doesn't pass through the third quadrant or origin, 
    then k < 0 and b > 0 -/
theorem linear_function_constraint (k b : ℝ) :
  (∀ x : ℝ, ¬(in_third_quadrant x (linear_function k b x))) ∧
  (∀ x : ℝ, ¬(is_origin x (linear_function k b x))) →
  k < 0 ∧ b > 0 :=
sorry

end NUMINAMATH_CALUDE_linear_function_constraint_l3688_368812


namespace NUMINAMATH_CALUDE_largest_ball_on_torus_l3688_368814

/-- The radius of the largest spherical ball that can be placed on top of a torus -/
def largest_ball_radius (inner_radius outer_radius : ℝ) (torus_center : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating that the largest ball radius for the given torus is 4 -/
theorem largest_ball_on_torus :
  let inner_radius : ℝ := 3
  let outer_radius : ℝ := 5
  let torus_center : ℝ × ℝ × ℝ := (4, 0, 1)
  largest_ball_radius inner_radius outer_radius torus_center = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_ball_on_torus_l3688_368814


namespace NUMINAMATH_CALUDE_arrangement_count_l3688_368853

/-- Represents the number of different books of each subject -/
structure BookCounts where
  math : Nat
  physics : Nat
  chemistry : Nat

/-- Calculates the number of arrangements given the book counts and constraints -/
def countArrangements (books : BookCounts) : Nat :=
  let totalBooks := books.math + books.physics + books.chemistry
  let mathUnit := 1  -- Treat math books as a single unit
  let nonMathBooks := books.physics + books.chemistry
  let totalUnits := mathUnit + nonMathBooks
  sorry

/-- The theorem to be proven -/
theorem arrangement_count :
  let books : BookCounts := { math := 3, physics := 2, chemistry := 1 }
  countArrangements books = 2592 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l3688_368853


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l3688_368879

theorem quadratic_polynomial_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h_distinct_a_b : a ≠ b) (h_distinct_b_c : b ≠ c) (h_distinct_a_c : a ≠ c)
  (h_quadratic : ∃ p q r : ℝ, ∀ x, f x = p * x^2 + q * x + r)
  (h_f_a : f a = b * c) (h_f_b : f b = c * a) (h_f_c : f c = a * b) :
  f (a + b + c) = a * b + b * c + a * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l3688_368879


namespace NUMINAMATH_CALUDE_two_integers_problem_l3688_368874

theorem two_integers_problem :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x - y = 8 ∧ x * y = 180 ∧ x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_problem_l3688_368874


namespace NUMINAMATH_CALUDE_triangle_problem_l3688_368813

/-- Given a triangle ABC with area 3√15, b - c = 2, and cos A = -1/4, prove the following: -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (1/2 * b * c * Real.sqrt (1 - (-1/4)^2) = 3 * Real.sqrt 15) →
  (b - c = 2) →
  (Real.cos A = -1/4) →
  (a^2 = b^2 + c^2 - 2*b*c*(-1/4)) →
  (a / Real.sqrt (1 - (-1/4)^2) = c / Real.sin C) →
  (a = 8 ∧ 
   Real.sin C = Real.sqrt 15 / 8 ∧ 
   Real.cos (2*A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3688_368813


namespace NUMINAMATH_CALUDE_nuts_ratio_l3688_368842

/-- Given the following conditions:
  - Sue has 48 nuts
  - Bill has 6 times as many nuts as Harry
  - Bill and Harry have combined 672 nuts
  Prove that the ratio of Harry's nuts to Sue's nuts is 2:1 -/
theorem nuts_ratio (sue_nuts : ℕ) (bill_harry_total : ℕ) :
  sue_nuts = 48 →
  bill_harry_total = 672 →
  ∃ (harry_nuts : ℕ),
    harry_nuts + 6 * harry_nuts = bill_harry_total ∧
    harry_nuts / sue_nuts = 2 :=
by sorry

end NUMINAMATH_CALUDE_nuts_ratio_l3688_368842


namespace NUMINAMATH_CALUDE_inequality_solution_l3688_368884

def solution_set : Set ℤ := {-3, 2}

def inequality (x : ℤ) : Prop :=
  (x^2 + 6*x + 8) * (x^2 - 4*x + 3) < 0

theorem inequality_solution :
  ∀ x : ℤ, inequality x ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3688_368884


namespace NUMINAMATH_CALUDE_goods_train_speed_l3688_368886

/-- The speed of a goods train passing a man in another train -/
theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (passing_time : ℝ) 
  (goods_train_length : ℝ) 
  (h1 : man_train_speed = 50) 
  (h2 : passing_time = 9 / 3600) 
  (h3 : goods_train_length = 0.28) : 
  ∃ (goods_train_speed : ℝ), goods_train_speed = 62 := by
sorry

end NUMINAMATH_CALUDE_goods_train_speed_l3688_368886


namespace NUMINAMATH_CALUDE_chinese_dream_speech_competition_l3688_368839

theorem chinese_dream_speech_competition :
  let num_contestants : ℕ := 4
  let num_topics : ℕ := 4
  let num_topics_used : ℕ := 3
  
  (num_topics.choose 1) * (num_topics_used ^ num_contestants) = 324 :=
by sorry

end NUMINAMATH_CALUDE_chinese_dream_speech_competition_l3688_368839


namespace NUMINAMATH_CALUDE_oranges_from_ann_l3688_368863

theorem oranges_from_ann (initial_oranges final_oranges : ℕ) 
  (h1 : initial_oranges = 9)
  (h2 : final_oranges = 38) :
  final_oranges - initial_oranges = 29 := by
  sorry

end NUMINAMATH_CALUDE_oranges_from_ann_l3688_368863


namespace NUMINAMATH_CALUDE_swamp_ecosystem_l3688_368846

/-- In a swamp ecosystem, prove that each gharial needs to eat 15 fish per day given the following conditions:
  * Each frog eats 30 flies per day
  * Each fish eats 8 frogs per day
  * There are 9 gharials in the swamp
  * 32,400 flies are eaten every day
-/
theorem swamp_ecosystem (flies_per_frog : ℕ) (frogs_per_fish : ℕ) (num_gharials : ℕ) (total_flies : ℕ)
  (h1 : flies_per_frog = 30)
  (h2 : frogs_per_fish = 8)
  (h3 : num_gharials = 9)
  (h4 : total_flies = 32400) :
  total_flies / (flies_per_frog * frogs_per_fish * num_gharials) = 15 := by
  sorry

end NUMINAMATH_CALUDE_swamp_ecosystem_l3688_368846


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l3688_368868

/-- Given an ellipse with equation 16(x+2)^2 + 4y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance :
  ∀ (x y : ℝ), 16 * (x + 2)^2 + 4 * y^2 = 64 →
  ∃ (C D : ℝ × ℝ),
    (C.1 + 2)^2 / 4 + C.2^2 / 16 = 1 ∧
    (D.1 + 2)^2 / 4 + D.2^2 / 16 = 1 ∧
    (C.2 = 0 ∨ C.2 = 0) ∧
    (D.1 = -2 ∨ D.1 = -2) ∧
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l3688_368868


namespace NUMINAMATH_CALUDE_johns_height_l3688_368837

theorem johns_height (building_height : ℝ) (building_shadow : ℝ) (johns_shadow_inches : ℝ) :
  building_height = 60 →
  building_shadow = 20 →
  johns_shadow_inches = 18 →
  ∃ (johns_height : ℝ), johns_height = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_johns_height_l3688_368837


namespace NUMINAMATH_CALUDE_elena_frog_count_l3688_368806

/-- Given a total number of frog eyes and the number of eyes per frog,
    calculate the number of frogs. -/
def count_frogs (total_eyes : ℕ) (eyes_per_frog : ℕ) : ℕ :=
  total_eyes / eyes_per_frog

/-- The problem statement -/
theorem elena_frog_count :
  let total_eyes : ℕ := 20
  let eyes_per_frog : ℕ := 2
  count_frogs total_eyes eyes_per_frog = 10 := by
  sorry

end NUMINAMATH_CALUDE_elena_frog_count_l3688_368806


namespace NUMINAMATH_CALUDE_exam_time_proof_l3688_368848

/-- Proves that the examination time is 3 hours given the specified conditions -/
theorem exam_time_proof (total_questions : ℕ) (type_a_problems : ℕ) (type_a_time : ℚ) :
  total_questions = 200 →
  type_a_problems = 15 →
  type_a_time = 25116279069767444 / 1000000000000000 →
  ∃ (type_b_time : ℚ),
    type_b_time > 0 ∧
    type_a_time = 2 * type_b_time * type_a_problems ∧
    (type_b_time * (total_questions - type_a_problems) + type_a_time) / 60 = 3 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_proof_l3688_368848


namespace NUMINAMATH_CALUDE_smallest_circle_equation_l3688_368855

/-- The equation of the circle with the smallest area that is tangent to the line 3x + 4y + 3 = 0
    and has its center on the curve y = 3/x (x > 0) -/
theorem smallest_circle_equation (x y : ℝ) :
  (∀ a : ℝ, a > 0 → ∃ r : ℝ, r > 0 ∧
    (∀ x₀ y₀ : ℝ, (x₀ - a)^2 + (y₀ - 3/a)^2 = r^2 →
      (3*x₀ + 4*y₀ + 3 = 0 → False) ∧
      (3*x₀ + 4*y₀ + 3 ≠ 0 → (3*x₀ + 4*y₀ + 3)^2 > 25*r^2))) →
  (x - 2)^2 + (y - 3/2)^2 = 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_equation_l3688_368855


namespace NUMINAMATH_CALUDE_bathtub_capacity_l3688_368873

/-- The capacity of a bathtub given tap flow rate, filling time, and drain leak rate -/
theorem bathtub_capacity 
  (tap_flow : ℝ)  -- Tap flow rate in liters per minute
  (fill_time : ℝ)  -- Filling time in minutes
  (leak_rate : ℝ)  -- Drain leak rate in liters per minute
  (h1 : tap_flow = 21 / 6)  -- Tap flow rate condition
  (h2 : fill_time = 22.5)  -- Filling time condition
  (h3 : leak_rate = 0.3)  -- Drain leak rate condition
  : tap_flow * fill_time - leak_rate * fill_time = 72 := by
  sorry


end NUMINAMATH_CALUDE_bathtub_capacity_l3688_368873


namespace NUMINAMATH_CALUDE_system_solutions_l3688_368815

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x^2 + y^2 = z^2 ∧ x*z = y^2 ∧ x*y = 10

-- State the theorem
theorem system_solutions :
  ∃ (x y z : ℝ), system x y z ∧
  ((x = Real.sqrt 10 ∧ y = Real.sqrt 10 ∧ z = Real.sqrt 10) ∨
   (x = -Real.sqrt 10 ∧ y = -Real.sqrt 10 ∧ z = -Real.sqrt 10)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3688_368815


namespace NUMINAMATH_CALUDE_digit_sum_proof_l3688_368823

theorem digit_sum_proof (A B : ℕ) :
  A ≤ 9 ∧ B ≤ 9 ∧ 
  111 * A + 110 * A + B + 100 * A + 11 * B + 111 * B = 1503 →
  A = 2 ∧ B = 7 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_proof_l3688_368823


namespace NUMINAMATH_CALUDE_parabola_rotation_180_l3688_368881

/-- Represents a parabola in the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Rotates a parabola 180° around its vertex -/
def rotate_180 (p : Parabola) : Parabola :=
  { a := -p.a, b := p.b }

theorem parabola_rotation_180 (p : Parabola) (h : p = { a := 1/2, b := 1 }) :
  rotate_180 p = { a := -1/2, b := 1 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_rotation_180_l3688_368881


namespace NUMINAMATH_CALUDE_largest_perimeter_is_24_l3688_368831

/-- Represents a configuration of two regular polygons and a circle meeting at a point -/
structure ShapeConfiguration where
  n : ℕ  -- number of sides in each polygon
  polygonSideLength : ℝ
  circleRadius : ℝ
  polygonAngleSum : ℝ  -- sum of interior angles of polygons at meeting point
  circleAngle : ℝ  -- angle subtended by circle at meeting point

/-- The perimeter of the configuration, excluding the circle's circumference -/
def perimeter (config : ShapeConfiguration) : ℝ :=
  2 * config.n * config.polygonSideLength

/-- Theorem stating the largest possible perimeter for the given configuration -/
theorem largest_perimeter_is_24 (config : ShapeConfiguration) : 
  config.polygonSideLength = 2 ∧ 
  config.circleRadius = 1 ∧ 
  config.polygonAngleSum + config.circleAngle = 360 →
  ∃ (maxConfig : ShapeConfiguration), 
    perimeter maxConfig = 24 ∧ 
    ∀ (c : ShapeConfiguration), perimeter c ≤ perimeter maxConfig :=
by
  sorry

end NUMINAMATH_CALUDE_largest_perimeter_is_24_l3688_368831


namespace NUMINAMATH_CALUDE_investment_ratio_is_three_l3688_368841

/-- Represents the investment scenario of three partners A, B, and C --/
structure Investment where
  x : ℝ  -- A's initial investment
  m : ℝ  -- Ratio of C's investment to A's investment
  total_gain : ℝ  -- Total annual gain
  a_share : ℝ  -- A's share of the gain

/-- The ratio of C's investment to A's investment in the given scenario --/
def investment_ratio (inv : Investment) : ℝ :=
  let a_investment := inv.x * 12  -- A's investment for 12 months
  let b_investment := 2 * inv.x * 6  -- B's investment for 6 months
  let c_investment := inv.m * inv.x * 4  -- C's investment for 4 months
  let total_investment := a_investment + b_investment + c_investment
  inv.m

/-- Theorem stating that the investment ratio is 3 given the conditions --/
theorem investment_ratio_is_three (inv : Investment)
  (h1 : inv.total_gain = 15000)
  (h2 : inv.a_share = 5000)
  (h3 : inv.x > 0)
  : investment_ratio inv = 3 := by
  sorry

#check investment_ratio_is_three

end NUMINAMATH_CALUDE_investment_ratio_is_three_l3688_368841


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3688_368828

theorem inequality_system_solution (x : ℝ) : 
  (2 + x > 7 - 4 * x ∧ x < (4 + x) / 2) ↔ (1 < x ∧ x < 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3688_368828


namespace NUMINAMATH_CALUDE_compare_expressions_l3688_368850

theorem compare_expressions (x y : ℝ) (h1 : x * y > 0) (h2 : x ≠ y) :
  x^4 + 6*x^2*y^2 + y^4 > 4*x*y*(x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l3688_368850
