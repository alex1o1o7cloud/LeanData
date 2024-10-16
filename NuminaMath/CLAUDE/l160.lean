import Mathlib

namespace NUMINAMATH_CALUDE_pizza_promotion_savings_l160_16007

/-- Represents the pizza promotion savings calculation --/
theorem pizza_promotion_savings : 
  let regular_medium_price : ℚ := 18
  let discounted_medium_price : ℚ := 5
  let num_medium_pizzas : ℕ := 3
  let large_pizza_toppings_cost : ℚ := 2 + 1.5 + 1 + 2.5
  let medium_pizza_toppings_cost : ℚ := large_pizza_toppings_cost

  let medium_pizza_savings : ℚ := (regular_medium_price - discounted_medium_price) * num_medium_pizzas
  let toppings_savings : ℚ := medium_pizza_toppings_cost * num_medium_pizzas

  let total_savings : ℚ := medium_pizza_savings + toppings_savings

  total_savings = 60 := by
  sorry

end NUMINAMATH_CALUDE_pizza_promotion_savings_l160_16007


namespace NUMINAMATH_CALUDE_recycling_efficiency_l160_16065

/-- The number of pounds Vanessa recycled -/
def vanessa_pounds : ℕ := 20

/-- The number of pounds Vanessa's friends recycled -/
def friends_pounds : ℕ := 16

/-- The total number of points earned -/
def total_points : ℕ := 4

/-- The number of pounds needed to earn one point -/
def pounds_per_point : ℚ := (vanessa_pounds + friends_pounds) / total_points

theorem recycling_efficiency : pounds_per_point = 9 := by sorry

end NUMINAMATH_CALUDE_recycling_efficiency_l160_16065


namespace NUMINAMATH_CALUDE_range_of_a_l160_16000

theorem range_of_a (p q : Prop) (h_p : p ↔ ∀ x ∈ Set.Icc (1/2) 1, 1/x - a ≥ 0)
  (h_q : q ↔ ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) (h_pq : p ∧ q) :
  a ∈ Set.Iic (-2) ∪ {1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l160_16000


namespace NUMINAMATH_CALUDE_square_difference_153_147_l160_16022

theorem square_difference_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_153_147_l160_16022


namespace NUMINAMATH_CALUDE_rectangle_cover_theorem_l160_16091

/-- An increasing function from [0, 1] to [0, 1] -/
def IncreasingFunction := {f : ℝ → ℝ | Monotone f ∧ Set.range f ⊆ Set.Icc 0 1}

/-- A rectangle with sides parallel to the coordinate axes -/
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- A set of rectangles covers the graph of a function -/
def covers (rs : Set Rectangle) (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, ∃ r ∈ rs, x ∈ Set.Icc r.x (r.x + r.width) ∧ f x ∈ Set.Icc r.y (r.y + r.height)

/-- Main theorem -/
theorem rectangle_cover_theorem (f : IncreasingFunction) (n : ℕ) :
  ∃ (rs : Set Rectangle), (∀ r ∈ rs, r.area = 1 / (2 * n)) ∧ covers rs f := by sorry

end NUMINAMATH_CALUDE_rectangle_cover_theorem_l160_16091


namespace NUMINAMATH_CALUDE_solve_equation_l160_16023

theorem solve_equation (x : ℝ) : 2 - 2 / (1 - x) = 2 / (1 - x) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l160_16023


namespace NUMINAMATH_CALUDE_expression_evaluation_l160_16030

theorem expression_evaluation : (980^2 : ℚ) / (210^2 - 206^2) = 577.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l160_16030


namespace NUMINAMATH_CALUDE_distance_to_yz_plane_l160_16079

/-- Given a point P(x, -6, z) where the distance from P to the x-axis is half
    the distance from P to the yz-plane, prove that the distance from P
    to the yz-plane is 12 units. -/
theorem distance_to_yz_plane (x z : ℝ) :
  let P : ℝ × ℝ × ℝ := (x, -6, z)
  abs (-6) = (1/2) * abs x →
  abs x = 12 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_yz_plane_l160_16079


namespace NUMINAMATH_CALUDE_angus_token_count_l160_16012

/-- The number of tokens Elsa has -/
def elsa_tokens : ℕ := 60

/-- The value of each token in dollars -/
def token_value : ℕ := 4

/-- The difference in token value between Elsa and Angus in dollars -/
def token_value_difference : ℕ := 20

/-- The number of tokens Angus has -/
def angus_tokens : ℕ := elsa_tokens - (token_value_difference / token_value)

theorem angus_token_count : angus_tokens = 55 := by
  sorry

end NUMINAMATH_CALUDE_angus_token_count_l160_16012


namespace NUMINAMATH_CALUDE_tables_needed_for_children_twenty_tables_needed_l160_16095

theorem tables_needed_for_children (num_children : ℕ) (table_capacity : ℕ) (num_tables : ℕ) : Prop :=
  num_children > 0 ∧ 
  table_capacity > 0 ∧ 
  num_tables * table_capacity ≥ num_children ∧ 
  (num_tables - 1) * table_capacity < num_children

theorem twenty_tables_needed : tables_needed_for_children 156 8 20 := by
  sorry

end NUMINAMATH_CALUDE_tables_needed_for_children_twenty_tables_needed_l160_16095


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_with_two_identical_l160_16048

/-- The count of four-digit numbers starting with 9 and having exactly two identical digits -/
def four_digit_numbers_with_two_identical : ℕ :=
  let first_case := 9 * 8 * 3  -- when 9 is repeated
  let second_case := 9 * 8 * 3 -- when a digit other than 9 is repeated
  first_case + second_case

/-- Theorem stating that the count of such numbers is 432 -/
theorem count_four_digit_numbers_with_two_identical :
  four_digit_numbers_with_two_identical = 432 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_with_two_identical_l160_16048


namespace NUMINAMATH_CALUDE_unpainted_area_l160_16097

/-- The area of the unpainted region on a 5-inch wide board when crossed with a 7-inch wide board at a 45-degree angle -/
theorem unpainted_area (board1_width board2_width crossing_angle : ℝ) : 
  board1_width = 5 →
  board2_width = 7 →
  crossing_angle = 45 →
  ∃ (area : ℝ), area = 35 * Real.sqrt 2 ∧ 
    area = board1_width * Real.sqrt 2 * board2_width := by
  sorry

end NUMINAMATH_CALUDE_unpainted_area_l160_16097


namespace NUMINAMATH_CALUDE_quadratic_factorization_l160_16017

theorem quadratic_factorization (x : ℂ) : 
  2 * x^2 + 8 * x + 26 = 2 * (x + 2 - 3 * I) * (x + 2 + 3 * I) := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l160_16017


namespace NUMINAMATH_CALUDE_mike_toys_total_cost_l160_16081

def marbles_cost : ℝ := 9.05
def football_cost : ℝ := 4.95
def baseball_cost : ℝ := 6.52
def toy_car_original_cost : ℝ := 5.50
def toy_car_discount_rate : ℝ := 0.10
def puzzle_cost : ℝ := 2.90
def action_figure_cost : ℝ := 8.80

def total_cost : ℝ :=
  marbles_cost +
  football_cost +
  baseball_cost +
  (toy_car_original_cost * (1 - toy_car_discount_rate)) +
  puzzle_cost +
  action_figure_cost

theorem mike_toys_total_cost :
  total_cost = 36.17 := by
  sorry

end NUMINAMATH_CALUDE_mike_toys_total_cost_l160_16081


namespace NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l160_16020

/-- Given a circle with an inscribed rectangle of dimensions 9 cm by 12 cm,
    the circumference of the circle is 15π cm. -/
theorem circle_circumference_with_inscribed_rectangle :
  ∀ (C : ℝ → ℝ → Prop) (r : ℝ),
    (∃ (x y : ℝ), C x y ∧ x^2 + y^2 = r^2 ∧ x = 9 ∧ y = 12) →
    2 * π * r = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l160_16020


namespace NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_for_ac2_gt_bc2_l160_16066

theorem a_gt_b_necessary_not_sufficient_for_ac2_gt_bc2 :
  (∃ (a b c : ℝ), a > b ∧ a * c^2 ≤ b * c^2) ∧
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) :=
sorry

end NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_for_ac2_gt_bc2_l160_16066


namespace NUMINAMATH_CALUDE_subset_implies_a_zero_l160_16067

theorem subset_implies_a_zero (a : ℝ) :
  let P : Set ℝ := {x | x^2 ≠ 1}
  let Q : Set ℝ := {x | a * x = 1}
  Q ⊆ P → a = 0 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_zero_l160_16067


namespace NUMINAMATH_CALUDE_exists_function_with_properties_l160_16099

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the properties of the function
def PassesThroughPoint (f : RealFunction) : Prop :=
  f (-2) = 1

def IncreasingInSecondQuadrant (f : RealFunction) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₁ < 0 → x₂ < 0 → f x₁ > 0 → f x₂ > 0 → f x₁ < f x₂

-- Theorem statement
theorem exists_function_with_properties :
  ∃ f : RealFunction, PassesThroughPoint f ∧ IncreasingInSecondQuadrant f :=
sorry

end NUMINAMATH_CALUDE_exists_function_with_properties_l160_16099


namespace NUMINAMATH_CALUDE_product_and_multiple_l160_16029

theorem product_and_multiple : ∃ x : ℕ, x = 320 * 6 ∧ x * 7 = 420 → x = 1920 := by
  sorry

end NUMINAMATH_CALUDE_product_and_multiple_l160_16029


namespace NUMINAMATH_CALUDE_one_fourth_of_6_8_l160_16049

theorem one_fourth_of_6_8 : (6.8 : ℚ) / 4 = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_6_8_l160_16049


namespace NUMINAMATH_CALUDE_sector_central_angle_l160_16027

/-- Given a circular sector with radius 10 cm and perimeter 45 cm, 
    its central angle is 2.5 radians. -/
theorem sector_central_angle : 
  ∀ (r p l α : ℝ), 
    r = 10 → 
    p = 45 → 
    l = p - 2 * r → 
    α = l / r → 
    α = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l160_16027


namespace NUMINAMATH_CALUDE_equation_solution_approximation_l160_16053

theorem equation_solution_approximation : ∃ x : ℝ, 
  (2.5 * ((x * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2000.0000000000002) ∧ 
  (abs (x - 3.6) < 0.0000000000000005) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_approximation_l160_16053


namespace NUMINAMATH_CALUDE_bank_account_transfer_l160_16005

/-- Represents a bank account transfer operation that doubles the amount in one account. -/
inductive Transfer : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → Prop
| double12 : ∀ a b c, Transfer (a, b, c) (a + b, 0, c)
| double13 : ∀ a b c, Transfer (a, b, c) (a + c, b, 0)
| double21 : ∀ a b c, Transfer (a, b, c) (0, a + b, c)
| double23 : ∀ a b c, Transfer (a, b, c) (a, b + c, 0)
| double31 : ∀ a b c, Transfer (a, b, c) (0, b, a + c)
| double32 : ∀ a b c, Transfer (a, b, c) (a, 0, b + c)

/-- Represents a sequence of transfers. -/
def TransferSeq : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → Prop :=
  Relation.ReflTransGen Transfer

theorem bank_account_transfer :
  (∀ a b c : ℕ, ∃ a' b' c', TransferSeq (a, b, c) (a', b', c') ∧ (a' = 0 ∨ b' = 0 ∨ c' = 0)) ∧
  (∃ a b c : ℕ, ∀ a' b' c', TransferSeq (a, b, c) (a', b', c') → ¬(a' = 0 ∧ b' = 0) ∧ ¬(a' = 0 ∧ c' = 0) ∧ ¬(b' = 0 ∧ c' = 0)) :=
by sorry

end NUMINAMATH_CALUDE_bank_account_transfer_l160_16005


namespace NUMINAMATH_CALUDE_prism_pyramid_volume_ratio_l160_16016

/-- Given a triangular prism with height m, we extend a side edge by x to form a pyramid.
    The volume ratio k of the remaining part of the prism (outside the pyramid) to the original prism
    must be less than or equal to 3/4. -/
theorem prism_pyramid_volume_ratio (m : ℝ) (x : ℝ) (k : ℝ) 
  (h1 : m > 0) (h2 : x > 0) (h3 : k > 0) : k ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prism_pyramid_volume_ratio_l160_16016


namespace NUMINAMATH_CALUDE_squares_sum_equality_l160_16050

/-- Represents a 3-4-5 right triangle with squares on each side -/
structure Triangle345WithSquares where
  /-- Area of the square on the side of length 3 -/
  A : ℝ
  /-- Area of the square on the side of length 4 -/
  B : ℝ
  /-- Area of the square on the hypotenuse (side of length 5) -/
  C : ℝ
  /-- The area of the square on side 3 is 9 -/
  h_A : A = 9
  /-- The area of the square on side 4 is 16 -/
  h_B : B = 16
  /-- The area of the square on the hypotenuse is 25 -/
  h_C : C = 25

/-- 
For a 3-4-5 right triangle with squares constructed on each side, 
the sum of the areas of the squares on the two shorter sides 
equals the area of the square on the hypotenuse.
-/
theorem squares_sum_equality (t : Triangle345WithSquares) : t.A + t.B = t.C := by
  sorry

end NUMINAMATH_CALUDE_squares_sum_equality_l160_16050


namespace NUMINAMATH_CALUDE_interchanged_digits_theorem_l160_16032

theorem interchanged_digits_theorem (n m a b : ℕ) : 
  n = 10 * a + b → 
  n = m * (a + b + a) → 
  10 * b + a = (9 - m) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_interchanged_digits_theorem_l160_16032


namespace NUMINAMATH_CALUDE_happy_properties_l160_16035

/-- A positive integer is happy if it can be expressed as the sum of two squares. -/
def IsHappy (n : ℕ+) : Prop :=
  ∃ a b : ℤ, n.val = a^2 + b^2

theorem happy_properties (t : ℕ+) (ht : IsHappy t) :
  (IsHappy (2 * t)) ∧ (¬IsHappy (3 * t)) := by
  sorry

end NUMINAMATH_CALUDE_happy_properties_l160_16035


namespace NUMINAMATH_CALUDE_count_congruent_integers_l160_16044

theorem count_congruent_integers (n : ℕ) : 
  (Finset.filter (fun x => x > 0 ∧ x < 2000 ∧ x % 13 = 3) (Finset.range 2000)).card = 154 := by
  sorry

end NUMINAMATH_CALUDE_count_congruent_integers_l160_16044


namespace NUMINAMATH_CALUDE_max_n_inequality_l160_16075

theorem max_n_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (∀ n : ℝ, 1 / (a - b) + 1 / (b - c) ≥ n / (a - c)) →
  (∃ n : ℝ, 1 / (a - b) + 1 / (b - c) = n / (a - c) ∧
            ∀ m : ℝ, 1 / (a - b) + 1 / (b - c) ≥ m / (a - c) → m ≤ n) →
  (∃ n : ℝ, n = 4 ∧
            1 / (a - b) + 1 / (b - c) = n / (a - c) ∧
            ∀ m : ℝ, 1 / (a - b) + 1 / (b - c) ≥ m / (a - c) → m ≤ n) :=
by sorry


end NUMINAMATH_CALUDE_max_n_inequality_l160_16075


namespace NUMINAMATH_CALUDE_smallest_m_divisible_by_15_l160_16025

-- Define q as the largest prime with 2011 digits
def q : ℕ := sorry

-- Axiom: q is prime
axiom q_prime : Nat.Prime q

-- Axiom: q has 2011 digits
axiom q_digits : 10^2010 ≤ q ∧ q < 10^2011

-- Define the property we want to prove
def is_divisible_by_15 (m : ℕ) : Prop :=
  ∃ k : ℤ, (q^2 - m : ℤ) = 15 * k

-- Theorem statement
theorem smallest_m_divisible_by_15 :
  ∃ m : ℕ, m > 0 ∧ is_divisible_by_15 m ∧
  ∀ n : ℕ, 0 < n ∧ n < m → ¬is_divisible_by_15 n :=
sorry

end NUMINAMATH_CALUDE_smallest_m_divisible_by_15_l160_16025


namespace NUMINAMATH_CALUDE_expression_value_l160_16071

/-- Given x = 7.5 and y = 2.5, prove that (x^y + √x + y^x) - (x^2 + y^y + √y) ≈ 679.2044 -/
theorem expression_value (x y : ℝ) (hx : x = 7.5) (hy : y = 2.5) :
  ∃ ε > 0, abs ((x^y + Real.sqrt x + y^x) - (x^2 + y^y + Real.sqrt y) - 679.2044) < ε :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l160_16071


namespace NUMINAMATH_CALUDE_area_of_triangle_APO_l160_16019

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Area of a triangle given three points -/
def triangleArea (P Q R : Point) : ℝ := sorry

/-- Area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := sorry

/-- Check if a point is on a line segment between two other points -/
def onSegment (P Q R : Point) : Prop := sorry

/-- Check if a line bisects another line segment -/
def bisectsSegment (P Q R S : Point) : Prop := sorry

/-- Main theorem -/
theorem area_of_triangle_APO (ABCD : Parallelogram) (P Q O : Point) (k : ℝ) :
  parallelogramArea ABCD = k →
  bisectsSegment ABCD.D P ABCD.C O →
  bisectsSegment ABCD.A Q ABCD.B O →
  onSegment ABCD.A P ABCD.B →
  onSegment ABCD.C Q ABCD.D →
  triangleArea ABCD.A P O = k / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_APO_l160_16019


namespace NUMINAMATH_CALUDE_larger_number_proof_l160_16037

theorem larger_number_proof (a b : ℝ) : 
  a + b = 104 → 
  a^2 - b^2 = 208 → 
  max a b = 53 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l160_16037


namespace NUMINAMATH_CALUDE_quadratic_rational_solution_l160_16088

theorem quadratic_rational_solution (a b : ℕ+) :
  (∃ x : ℚ, x^2 + (a + b : ℚ)^2 * x + 4 * (a : ℚ) * (b : ℚ) = 1) ↔ a = b :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rational_solution_l160_16088


namespace NUMINAMATH_CALUDE_smallest_number_l160_16003

def number_set : Set ℤ := {-1, 0, 1, 2}

theorem smallest_number : ∀ x ∈ number_set, -1 ≤ x := by sorry

end NUMINAMATH_CALUDE_smallest_number_l160_16003


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l160_16070

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let expr := (a-b)*(a-c)/(a+b+c) + (b-c)*(b-d)/(b+c+d) + 
               (c-d)*(c-a)/(c+d+a) + (d-a)*(d-b)/(d+a+b)
  (expr ≥ 0) ∧ 
  (expr = 0 ↔ a = c ∧ b = d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l160_16070


namespace NUMINAMATH_CALUDE_special_number_pair_l160_16041

/-- Given two distinct positive integers a and b, such that b is a multiple of a,
    both a and b consist of 2n digits in decimal form with no leading zeros,
    and the first n digits of a are the same as the last n digits of b (and vice versa),
    prove that a = (10^(2n) - 1) / 7 and b = 6 * (10^(2n) - 1) / 7 -/
theorem special_number_pair (n : ℕ) (a b : ℕ) :
  (a ≠ b) →
  (a > 0) →
  (b > 0) →
  (∃ (k : ℕ), b = k * a) →
  (10^n ≤ a) →
  (a < 10^(2*n)) →
  (10^n ≤ b) →
  (b < 10^(2*n)) →
  (∃ (x y : ℕ), a = 10^n * x + y ∧ b = 10^n * y + x ∧ x < 10^n ∧ y < 10^n) →
  (a = (10^(2*n) - 1) / 7 ∧ b = 6 * (10^(2*n) - 1) / 7) := by
sorry

end NUMINAMATH_CALUDE_special_number_pair_l160_16041


namespace NUMINAMATH_CALUDE_count_valid_sequences_l160_16086

/-- Represents the number of advertisements -/
def total_ads : ℕ := 5

/-- Represents the number of commercial advertisements -/
def commercial_ads : ℕ := 3

/-- Represents the number of National Games promotional advertisements -/
def national_games_ads : ℕ := 2

/-- Represents the constraint that the last advertisement must be a National Games promotional advertisement -/
def last_ad_is_national_games : Prop := True

/-- Represents the constraint that the two National Games adverts cannot be played consecutively -/
def national_games_not_consecutive : Prop := True

/-- Calculates the number of valid broadcasting sequences -/
def valid_sequences : ℕ := 36

/-- Theorem stating that the number of valid broadcasting sequences is 36 -/
theorem count_valid_sequences :
  total_ads = 5 ∧
  commercial_ads = 3 ∧
  national_games_ads = 2 ∧
  last_ad_is_national_games ∧
  national_games_not_consecutive →
  valid_sequences = 36 :=
by sorry

end NUMINAMATH_CALUDE_count_valid_sequences_l160_16086


namespace NUMINAMATH_CALUDE_rectangular_field_length_l160_16004

theorem rectangular_field_length : 
  ∀ (w : ℝ), 
    w > 0 → 
    w^2 + (w + 10)^2 = 22^2 → 
    w + 10 = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l160_16004


namespace NUMINAMATH_CALUDE_linda_remaining_candies_l160_16018

/-- The number of candies Linda has left after giving some away -/
def candies_left (initial : ℝ) (given_away : ℝ) : ℝ := initial - given_away

/-- Theorem stating that Linda's remaining candies is the difference between initial and given away -/
theorem linda_remaining_candies (initial : ℝ) (given_away : ℝ) :
  candies_left initial given_away = initial - given_away :=
by sorry

end NUMINAMATH_CALUDE_linda_remaining_candies_l160_16018


namespace NUMINAMATH_CALUDE_problem_statement_l160_16028

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : 1/a + 1/b + 1/c = 1) : 
  (∃ (min : ℝ), min = 36 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 1/x + 1/y + 1/z = 1 → x + 4*y + 9*z ≥ min) ∧ 
  ((b+c)/Real.sqrt a + (a+c)/Real.sqrt b + (a+b)/Real.sqrt c ≥ 2 * Real.sqrt (a*b*c)) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l160_16028


namespace NUMINAMATH_CALUDE_queen_mary_heads_l160_16026

/-- The number of heads on the luxury liner Queen Mary II -/
def total_heads : ℕ := by sorry

/-- The number of legs on the luxury liner Queen Mary II -/
def total_legs : ℕ := 41

/-- The number of cats on the ship -/
def num_cats : ℕ := 5

/-- The number of legs each cat has -/
def cat_legs : ℕ := 4

/-- The number of legs each sailor or cook has -/
def crew_legs : ℕ := 2

/-- The number of legs the captain has -/
def captain_legs : ℕ := 1

/-- The number of sailors and cooks combined -/
def num_crew : ℕ := by sorry

theorem queen_mary_heads :
  total_heads = num_cats + num_crew + 1 ∧
  total_legs = num_cats * cat_legs + num_crew * crew_legs + captain_legs ∧
  total_heads = 16 := by sorry

end NUMINAMATH_CALUDE_queen_mary_heads_l160_16026


namespace NUMINAMATH_CALUDE_unique_digit_multiplication_l160_16047

theorem unique_digit_multiplication :
  ∃! (A B C D E : Nat),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    A ≠ 0 ∧
    (A * 10000 + B * 1000 + C * 100 + D * 10 + E) * 4 =
    E * 10000 + D * 1000 + C * 100 + B * 10 + A ∧
    A = 2 ∧ B = 1 ∧ C = 9 ∧ D = 7 ∧ E = 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_multiplication_l160_16047


namespace NUMINAMATH_CALUDE_least_positive_integer_with_property_l160_16046

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

end NUMINAMATH_CALUDE_least_positive_integer_with_property_l160_16046


namespace NUMINAMATH_CALUDE_nonzero_y_solution_l160_16058

theorem nonzero_y_solution (y : ℝ) (hy : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_y_solution_l160_16058


namespace NUMINAMATH_CALUDE_zoo_visitors_l160_16096

theorem zoo_visitors (num_cars : ℝ) (people_per_car : ℝ) 
  (h1 : num_cars = 3.0) 
  (h2 : people_per_car = 63.0) : 
  num_cars * people_per_car = 189.0 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l160_16096


namespace NUMINAMATH_CALUDE_contractor_wage_l160_16057

def contractor_problem (total_days : ℕ) (absent_days : ℕ) (fine_per_day : ℚ) (total_amount : ℚ) : Prop :=
  ∃ (daily_wage : ℚ),
    (total_days - absent_days) * daily_wage - absent_days * fine_per_day = total_amount ∧
    daily_wage = 25

theorem contractor_wage :
  contractor_problem 30 8 (25/2) 490 :=
sorry

end NUMINAMATH_CALUDE_contractor_wage_l160_16057


namespace NUMINAMATH_CALUDE_transform_to_successor_l160_16083

/-- Represents the allowed operations on natural numbers -/
inductive Operation
  | AddNine
  | EraseOne

/-- Applies a single operation to a natural number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddNine => n + 9
  | Operation.EraseOne => sorry  -- Implementation of erasing 1 is complex and not provided

/-- Applies a sequence of operations to a natural number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- 
Theorem: For any natural number A, there exists a sequence of operations 
that transforms A into A+1
-/
theorem transform_to_successor (A : ℕ) : 
  ∃ (ops : List Operation), applyOperations A ops = A + 1 :=
sorry

end NUMINAMATH_CALUDE_transform_to_successor_l160_16083


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l160_16010

theorem quadratic_complete_square (x : ℝ) : 
  x^2 + 10*x + 7 = 0 → ∃ c d : ℝ, (x + c)^2 = d ∧ d = 18 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l160_16010


namespace NUMINAMATH_CALUDE_factorization_cube_minus_linear_l160_16063

theorem factorization_cube_minus_linear (a b : ℝ) : a^3 * b - a * b = a * b * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cube_minus_linear_l160_16063


namespace NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l160_16094

/-- Represents a round-robin tournament --/
structure Tournament :=
  (num_teams : ℕ)
  (wins : Fin num_teams → ℕ)

/-- The total number of games in a round-robin tournament --/
def total_games (t : Tournament) : ℕ :=
  t.num_teams * (t.num_teams - 1) / 2

/-- The maximum number of wins for any team in the tournament --/
def max_wins (t : Tournament) : ℕ :=
  Finset.sup Finset.univ t.wins

/-- The number of teams tied for the maximum number of wins --/
def num_teams_with_max_wins (t : Tournament) : ℕ :=
  Finset.card (Finset.filter (λ i => t.wins i = max_wins t) Finset.univ)

/-- The main theorem --/
theorem max_teams_tied_for_most_wins :
  ∃ (t : Tournament), t.num_teams = 8 ∧
  (∀ (t' : Tournament), t'.num_teams = 8 →
    num_teams_with_max_wins t' ≤ num_teams_with_max_wins t) ∧
  num_teams_with_max_wins t = 7 :=
sorry

end NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l160_16094


namespace NUMINAMATH_CALUDE_cube_root_of_nested_roots_l160_16021

theorem cube_root_of_nested_roots (x : ℝ) (h : x ≥ 0) :
  (x * (x * x^(1/3))^(1/2))^(1/3) = x^(5/9) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_nested_roots_l160_16021


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l160_16089

theorem complex_magnitude_equation : 
  ∃ (t : ℝ), t > 0 ∧ Complex.abs (9 + t * Complex.I) = 15 ↔ t = 12 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l160_16089


namespace NUMINAMATH_CALUDE_tuesday_rain_less_than_monday_l160_16015

/-- The rainfall difference between two days -/
def rainfall_difference (day1_rain : Real) (day2_rain : Real) : Real :=
  day1_rain - day2_rain

/-- Theorem stating the rainfall difference between Monday and Tuesday -/
theorem tuesday_rain_less_than_monday :
  let monday_rain : Real := 0.9
  let tuesday_rain : Real := 0.2
  rainfall_difference monday_rain tuesday_rain = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_rain_less_than_monday_l160_16015


namespace NUMINAMATH_CALUDE_quadratic_equation_equal_coefficients_l160_16059

/-- A quadratic equation with coefficients forming an arithmetic sequence and reciprocal roots -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  roots_reciprocal : ∃ (r s : ℝ), r * s = 1 ∧ a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0
  coeff_arithmetic : b - a = c - b

/-- The coefficients of a quadratic equation with reciprocal roots and coefficients in arithmetic sequence are equal -/
theorem quadratic_equation_equal_coefficients (eq : QuadraticEquation) : eq.a = eq.b ∧ eq.b = eq.c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equal_coefficients_l160_16059


namespace NUMINAMATH_CALUDE_spinner_product_even_probability_l160_16098

def spinner1 : Finset Nat := {2, 5, 7, 11}
def spinner2 : Finset Nat := {3, 4, 6, 8, 10}

def isEven (n : Nat) : Bool := n % 2 = 0

theorem spinner_product_even_probability :
  let totalOutcomes := spinner1.card * spinner2.card
  let evenProductOutcomes := (spinner1.card * spinner2.card) - 
    (spinner1.filter (λ x => ¬isEven x)).card * (spinner2.filter (λ x => ¬isEven x)).card
  (evenProductOutcomes : ℚ) / totalOutcomes = 17 / 20 := by
  sorry

end NUMINAMATH_CALUDE_spinner_product_even_probability_l160_16098


namespace NUMINAMATH_CALUDE_strictly_increasing_f_range_of_k_l160_16073

-- Define the properties of the function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ a b, a + b ≠ 0 → (f a + f b) / (a + b) > 0)

-- Theorem 1
theorem strictly_increasing_f (f : ℝ → ℝ) (h : is_valid_f f) :
  ∀ a b, a > b → f a > f b :=
sorry

-- Theorem 2
theorem range_of_k (f : ℝ → ℝ) (h : is_valid_f f) :
  (∀ x : ℝ, x ≥ 0 → f (9^x - 2*3^x) + f (2*9^x - k) > 0) → k < 1 :=
sorry

end NUMINAMATH_CALUDE_strictly_increasing_f_range_of_k_l160_16073


namespace NUMINAMATH_CALUDE_min_sum_with_constraints_l160_16006

theorem min_sum_with_constraints (x y z w : ℕ+) (h : (4 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val ∧ (6 : ℕ) * z.val = (7 : ℕ) * w.val) :
  x.val + y.val + z.val + w.val ≥ 319 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_constraints_l160_16006


namespace NUMINAMATH_CALUDE_max_price_is_100_l160_16087

-- Define the maximum market price
def max_price : ℝ := 100

-- Define the selling prices for Store A and Store B
def selling_price_A : ℝ := max_price - 10
def selling_price_B : ℝ := max_price - 20

-- Define the profit percentages for Store A and Store B
def profit_percentage_A : ℝ := 0.1
def profit_percentage_B : ℝ := 0.2

-- Define the profit amounts for Store A and Store B
def profit_A : ℝ := selling_price_A * profit_percentage_A
def profit_B : ℝ := selling_price_B * profit_percentage_B

-- Theorem statement
theorem max_price_is_100 : 
  profit_A = profit_B ∧ 
  selling_price_A = max_price - 10 ∧ 
  selling_price_B = max_price - 20 ∧ 
  profit_A = selling_price_A * profit_percentage_A ∧ 
  profit_B = selling_price_B * profit_percentage_B →
  max_price = 100 := by sorry

end NUMINAMATH_CALUDE_max_price_is_100_l160_16087


namespace NUMINAMATH_CALUDE_harvester_equations_l160_16042

theorem harvester_equations (x y : ℝ) : True → ∃ (eq1 eq2 : ℝ → ℝ → Prop),
  (∀ a b, eq1 a b ↔ 2 * (2 * a + 5 * b) = 3.6) ∧
  (∀ a b, eq2 a b ↔ 5 * (3 * a + 2 * b) = 8) ∧
  (eq1 x y ∧ eq2 x y) :=
by
  sorry

end NUMINAMATH_CALUDE_harvester_equations_l160_16042


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l160_16060

theorem quadratic_roots_relation (m n p : ℝ) : 
  m ≠ 0 → n ≠ 0 → p ≠ 0 →
  (∃ s₁ s₂ : ℝ, (s₁ * s₂ = m) ∧ 
               (s₁ + s₂ = -p) ∧
               ((3 * s₁) * (3 * s₂) = n)) →
  n / p = -27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l160_16060


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l160_16008

-- Problem 1
theorem problem_1 : |Real.sqrt 3 - 2| + (3 - Real.pi)^0 - Real.sqrt 12 + 6 * Real.cos (30 * π / 180) = 3 := by sorry

-- Problem 2
theorem problem_2 : (1 / ((-5)^2 - 3*(-5))) / (2 / ((-5)^2 - 9)) = 1/5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l160_16008


namespace NUMINAMATH_CALUDE_kishore_savings_l160_16074

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 3940
def savings_percentage : ℚ := 1 / 10

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous

theorem kishore_savings :
  let monthly_salary := total_expenses / (1 - savings_percentage)
  (monthly_salary * savings_percentage).floor = 2160 := by
  sorry

end NUMINAMATH_CALUDE_kishore_savings_l160_16074


namespace NUMINAMATH_CALUDE_football_team_members_l160_16055

/-- The total number of members in a football team after new members join -/
def total_members (initial : ℕ) (new : ℕ) : ℕ :=
  initial + new

/-- Theorem stating that the total number of members in the football team is 59 -/
theorem football_team_members :
  total_members 42 17 = 59 := by sorry

end NUMINAMATH_CALUDE_football_team_members_l160_16055


namespace NUMINAMATH_CALUDE_taylor_series_expansion_of_f_l160_16036

def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 2

def taylor_expansion (x : ℝ) : ℝ := -12 + 16*(x + 1) - 7*(x + 1)^2 + (x + 1)^3

theorem taylor_series_expansion_of_f :
  ∀ x : ℝ, f x = taylor_expansion x := by
  sorry

end NUMINAMATH_CALUDE_taylor_series_expansion_of_f_l160_16036


namespace NUMINAMATH_CALUDE_least_divisor_for_perfect_square_l160_16001

theorem least_divisor_for_perfect_square (n : ℕ) (h : n = 16800) :
  ∃ (d : ℕ), d = 21 ∧ 
  (∀ (k : ℕ), k < d → ¬∃ (m : ℕ), n / k = m ^ 2) ∧
  ∃ (m : ℕ), n / d = m ^ 2 :=
sorry

end NUMINAMATH_CALUDE_least_divisor_for_perfect_square_l160_16001


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l160_16009

theorem infinitely_many_solutions (c : ℝ) : 
  (∀ x : ℝ, 3 * (5 + c * x) = 18 * x + 15) ↔ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l160_16009


namespace NUMINAMATH_CALUDE_womens_average_age_l160_16052

/-- The average age of two women given the following conditions:
    - There are initially 6 men
    - Two men aged 10 and 12 are replaced by two women
    - The average age increases by 2 years after the replacement
-/
theorem womens_average_age (initial_men : ℕ) (age_increase : ℝ) 
  (replaced_man1_age replaced_man2_age : ℕ) :
  initial_men = 6 →
  age_increase = 2 →
  replaced_man1_age = 10 →
  replaced_man2_age = 12 →
  ∃ (initial_avg : ℝ),
    ((initial_men : ℝ) * initial_avg - (replaced_man1_age + replaced_man2_age : ℝ) + 
     2 * ((initial_avg + age_increase) : ℝ)) / 2 = 17 :=
by sorry

end NUMINAMATH_CALUDE_womens_average_age_l160_16052


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_l160_16040

theorem polynomial_nonnegative (x : ℝ) : x^4 - x^3 + 3*x^2 - 2*x + 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_l160_16040


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l160_16092

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ 1 / 3 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 12 ∧ 1 / a₀ + 1 / b₀ = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l160_16092


namespace NUMINAMATH_CALUDE_weight_replacement_l160_16062

theorem weight_replacement (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) : 
  initial_count = 5 → 
  replaced_weight = 65 → 
  avg_increase = 1.5 → 
  (initial_count : ℝ) * avg_increase + replaced_weight = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l160_16062


namespace NUMINAMATH_CALUDE_residue_of_5_1234_mod_19_l160_16078

theorem residue_of_5_1234_mod_19 : 
  (5 : ℤ)^1234 ≡ 7 [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_residue_of_5_1234_mod_19_l160_16078


namespace NUMINAMATH_CALUDE_num_valid_configs_eq_30_l160_16093

/-- A valid grid configuration -/
structure GridConfig where
  numbers : Fin 9 → Bool
  positions : Fin 6 → Fin 9
  left_greater_right : ∀ i : Fin 2, positions (2*i) > positions (2*i + 1)
  top_smaller_bottom : ∀ i : Fin 3, positions i < positions (i + 3)
  all_different : ∀ i j : Fin 6, i ≠ j → positions i ≠ positions j
  used_numbers : ∀ i : Fin 9, numbers i = (∃ j : Fin 6, positions j = i)

/-- The number of valid grid configurations -/
def num_valid_configs : ℕ := sorry

/-- The main theorem: there are exactly 30 valid grid configurations -/
theorem num_valid_configs_eq_30 : num_valid_configs = 30 := by sorry

end NUMINAMATH_CALUDE_num_valid_configs_eq_30_l160_16093


namespace NUMINAMATH_CALUDE_equality_proof_l160_16080

theorem equality_proof (a b c p : ℝ) (h : a + b + c = 2 * p) :
  (2 * a * p + b * c) * (2 * b * p + a * c) * (2 * c * p + a * b) =
  (a + b)^2 * (a + c)^2 * (b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_equality_proof_l160_16080


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l160_16038

theorem arithmetic_expression_equality : 5 * 7 - 6 + 2 * 12 + 2 * 6 + 7 * 3 = 86 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l160_16038


namespace NUMINAMATH_CALUDE_power_inequality_l160_16077

theorem power_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a^6 + b^6 ≥ a*b*(a^4 + b^4) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l160_16077


namespace NUMINAMATH_CALUDE_consecutive_pair_sum_divisible_by_five_l160_16024

theorem consecutive_pair_sum_divisible_by_five (n : ℕ) : 
  n < 1500 → 
  (n + (n + 1)) % 5 = 0 → 
  (57 + 58) % 5 = 0 → 
  57 = n := by
sorry

end NUMINAMATH_CALUDE_consecutive_pair_sum_divisible_by_five_l160_16024


namespace NUMINAMATH_CALUDE_bus_departure_interval_l160_16013

/-- Represents the number of minutes between 6:00 AM and 7:00 AM -/
def total_minutes : ℕ := 60

/-- Represents the number of bus departures between 6:00 AM and 7:00 AM -/
def num_departures : ℕ := 11

/-- Calculates the interval between consecutive bus departures -/
def interval (total : ℕ) (departures : ℕ) : ℚ :=
  (total : ℚ) / ((departures - 1) : ℚ)

/-- Proves that the interval between consecutive bus departures is 6 minutes -/
theorem bus_departure_interval :
  interval total_minutes num_departures = 6 := by
  sorry

end NUMINAMATH_CALUDE_bus_departure_interval_l160_16013


namespace NUMINAMATH_CALUDE_pears_picked_total_l160_16054

/-- The number of pears Sara picked -/
def sara_pears : ℕ := 45

/-- The number of pears Sally picked -/
def sally_pears : ℕ := 11

/-- The total number of pears picked -/
def total_pears : ℕ := sara_pears + sally_pears

theorem pears_picked_total : total_pears = 56 := by
  sorry

end NUMINAMATH_CALUDE_pears_picked_total_l160_16054


namespace NUMINAMATH_CALUDE_billys_restaurant_bill_l160_16043

/-- The total bill for a group at Billy's Restaurant -/
def total_bill (num_adults : ℕ) (num_children : ℕ) (cost_per_meal : ℕ) : ℕ :=
  (num_adults + num_children) * cost_per_meal

/-- Theorem: The bill for 2 adults and 5 children with meals costing $3 each is $21 -/
theorem billys_restaurant_bill :
  total_bill 2 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_billys_restaurant_bill_l160_16043


namespace NUMINAMATH_CALUDE_magnitude_n_equals_five_l160_16069

/-- Given two vectors m and n in ℝ², prove that |n| = 5 -/
theorem magnitude_n_equals_five (m n : ℝ × ℝ) 
  (h1 : m.1 * n.1 + m.2 * n.2 = 0)  -- m is perpendicular to n
  (h2 : (m.1 - 2 * n.1, m.2 - 2 * n.2) = (11, -2))  -- m - 2n = (11, -2)
  (h3 : Real.sqrt (m.1^2 + m.2^2) = 5)  -- |m| = 5
  : Real.sqrt (n.1^2 + n.2^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_n_equals_five_l160_16069


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l160_16034

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Define M as the sum of divisors of 450
def M : ℕ := sum_of_divisors 450

-- Define a function to get the largest prime factor
def largest_prime_factor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_sum_of_divisors_450 :
  largest_prime_factor M = 13 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l160_16034


namespace NUMINAMATH_CALUDE_misread_addition_l160_16002

theorem misread_addition (X : Nat) : 
  X < 10 → 57 + (10 * X + 6) = 123 → (10 * X + 9) = 69 := by
  sorry

end NUMINAMATH_CALUDE_misread_addition_l160_16002


namespace NUMINAMATH_CALUDE_poles_count_l160_16076

/-- The number of telephone poles given the interval distance and total distance -/
def num_poles (interval : ℕ) (total_distance : ℕ) : ℕ :=
  (total_distance / interval) + 1

/-- Theorem stating that the number of poles is 61 given the specific conditions -/
theorem poles_count : num_poles 25 1500 = 61 := by
  sorry

end NUMINAMATH_CALUDE_poles_count_l160_16076


namespace NUMINAMATH_CALUDE_positive_real_inequality_l160_16045

theorem positive_real_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  a * (a - 1) + b * (b - 1) + c * (c - 1) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l160_16045


namespace NUMINAMATH_CALUDE_sinusoidal_function_translation_l160_16082

/-- Given a function f(x) = sin(ωx + φ) with the following properties:
    - ω > 0
    - |φ| < π/2
    - Smallest positive period is π
    - Graph is translated left by π/6 units
    - Resulting function is odd
    Then f(x) = sin(2x - π/3) -/
theorem sinusoidal_function_translation (f : ℝ → ℝ) (ω φ : ℝ) 
    (h_omega : ω > 0)
    (h_phi : |φ| < π/2)
    (h_period : ∀ x, f (x + π) = f x)
    (h_smallest_period : ∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ π)
    (h_translation : ∀ x, f (x + π/6) = -f (-x + π/6)) :
  ∀ x, f x = Real.sin (2*x - π/3) := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_translation_l160_16082


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l160_16014

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem digit_sum_theorem (a b c d : ℕ) (square : ℕ) :
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit square →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * 100 + 60 + b - (400 + c * 10 + d) = 2 →
  a + b + c + d = 10 ∨ a + b + c + d = 18 ∨ a + b + c + d = 19 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l160_16014


namespace NUMINAMATH_CALUDE_last_digit_of_3_power_2012_l160_16031

/-- The last digit of 3^n for any natural number n -/
def lastDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

theorem last_digit_of_3_power_2012 :
  lastDigitOf3Power 2012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_3_power_2012_l160_16031


namespace NUMINAMATH_CALUDE_min_value_theorem_l160_16061

theorem min_value_theorem (a b : ℝ) (h1 : 2*a + 3*b = 6) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2*x + 3*y = 6 → 2/x + 3/y ≥ 25/6) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + 3*y = 6 ∧ 2/x + 3/y = 25/6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l160_16061


namespace NUMINAMATH_CALUDE_sequence_sum_equals_110_over_7_l160_16085

def arithmetic_sequence_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

def sequence_sum : ℚ :=
  arithmetic_sequence_sum (2/7) (2/7) 10

theorem sequence_sum_equals_110_over_7 : sequence_sum = 110/7 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_110_over_7_l160_16085


namespace NUMINAMATH_CALUDE_workshop_workers_count_l160_16033

theorem workshop_workers_count :
  let average_salary : ℕ := 9000
  let technician_count : ℕ := 7
  let technician_salary : ℕ := 12000
  let non_technician_salary : ℕ := 6000
  ∃ (total_workers : ℕ),
    total_workers * average_salary = 
      technician_count * technician_salary + 
      (total_workers - technician_count) * non_technician_salary ∧
    total_workers = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l160_16033


namespace NUMINAMATH_CALUDE_children_age_sum_l160_16090

/-- Given 5 children with an age difference of 2 years between each, 
    and the eldest being 12 years old, the sum of their ages is 40 years. -/
theorem children_age_sum : 
  let num_children : ℕ := 5
  let age_diff : ℕ := 2
  let eldest_age : ℕ := 12
  let ages : List ℕ := List.range num_children |>.map (λ i => eldest_age - i * age_diff)
  ages.sum = 40 := by sorry

end NUMINAMATH_CALUDE_children_age_sum_l160_16090


namespace NUMINAMATH_CALUDE_prob_qualified_A_value_l160_16072

/-- The probability of purchasing a qualified light bulb produced by Factory A -/
def prob_qualified_A (prop_A prop_B pass_rate_A pass_rate_B : ℝ) : ℝ :=
  prop_A * pass_rate_A

/-- Theorem: The probability of purchasing a qualified light bulb produced by Factory A is 0.665 -/
theorem prob_qualified_A_value :
  prob_qualified_A 0.7 0.3 0.95 0.8 = 0.665 := by
  sorry

#eval prob_qualified_A 0.7 0.3 0.95 0.8

end NUMINAMATH_CALUDE_prob_qualified_A_value_l160_16072


namespace NUMINAMATH_CALUDE_round_trip_car_time_is_eight_l160_16056

/-- Represents the time in minutes for various trip configurations -/
structure TripTime where
  carAndWalk : ℕ  -- Time for car there and walk back
  walkBoth : ℕ    -- Time for walking both ways

/-- Calculates the time for a round trip by car given the TripTime -/
def roundTripCarTime (t : TripTime) : ℕ :=
  2 * (t.carAndWalk - t.walkBoth / 2)

/-- Theorem: Given the specific trip times, the round trip car time is 8 minutes -/
theorem round_trip_car_time_is_eight (t : TripTime) 
  (h1 : t.carAndWalk = 20) 
  (h2 : t.walkBoth = 32) : 
  roundTripCarTime t = 8 := by
  sorry

#eval roundTripCarTime { carAndWalk := 20, walkBoth := 32 }

end NUMINAMATH_CALUDE_round_trip_car_time_is_eight_l160_16056


namespace NUMINAMATH_CALUDE_g_sum_symmetric_l160_16084

-- Define the function g
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^8 + e * x^6 - f * x^4 + 5

-- Theorem statement
theorem g_sum_symmetric (d e f : ℝ) :
  (∃ x, g d e f x = 7) → g d e f 2 + g d e f (-2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_symmetric_l160_16084


namespace NUMINAMATH_CALUDE_quadratic_coefficient_count_l160_16064

theorem quadratic_coefficient_count : ∀ n : ℤ, 
  (∃ p q : ℤ, p * q = 30 ∧ p + q = n) → 
  (∃ S : Finset ℤ, S.card = 8 ∧ n ∈ S ∧ ∀ m : ℤ, m ∈ S ↔ ∃ p q : ℤ, p * q = 30 ∧ p + q = m) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_count_l160_16064


namespace NUMINAMATH_CALUDE_chord_length_theorem_l160_16039

theorem chord_length_theorem (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x + y = 2*k - 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 = 1 ∧ 
    x₂^2 + y₂^2 = 1 ∧ 
    x₁ + y₁ = 2*k - 1 ∧ 
    x₂ + y₂ = 2*k - 1 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 2) →
  k = 0 ∨ k = 1 := by
sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l160_16039


namespace NUMINAMATH_CALUDE_factorization_equality_l160_16068

theorem factorization_equality (a b : ℝ) : a^2 * b - b^3 = b * (a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l160_16068


namespace NUMINAMATH_CALUDE_problem_solution_l160_16051

def is_product_of_three_primes_less_than_10 (n : ℕ) : Prop :=
  ∃ p q r, p < 10 ∧ q < 10 ∧ r < 10 ∧ Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ n = p * q * r

def all_primes_less_than_10_present (a b : ℕ) : Prop :=
  ∀ p, p < 10 → Nat.Prime p → (p ∣ a ∨ p ∣ b)

theorem problem_solution (a b : ℕ) :
  is_product_of_three_primes_less_than_10 a ∧
  is_product_of_three_primes_less_than_10 b ∧
  all_primes_less_than_10_present a b ∧
  Nat.gcd a b = Nat.gcd (a / 15) b ∧
  Nat.gcd a b = 2 * Nat.gcd a (b / 4) →
  a = 30 ∧ b = 28 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l160_16051


namespace NUMINAMATH_CALUDE_garden_area_l160_16011

theorem garden_area (length_distance width_distance : ℝ) 
  (h1 : length_distance * 30 = 1500)
  (h2 : (2 * length_distance + 2 * width_distance) * 12 = 1500) :
  length_distance * width_distance = 625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l160_16011
