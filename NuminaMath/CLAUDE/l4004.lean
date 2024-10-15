import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l4004_400408

theorem units_digit_of_fraction : (30 * 31 * 32 * 33 * 34 * 35) / 5000 % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l4004_400408


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l4004_400493

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Check if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (n : ℝ) : Prop :=
  sn.coefficient * (10 : ℝ) ^ sn.exponent = n

/-- The number we want to represent in scientific notation -/
def target_number : ℝ := 2034000

/-- The proposed scientific notation representation -/
def proposed_representation : ScientificNotation := {
  coefficient := 2.034
  exponent := 6
  coeff_range := by sorry
}

/-- Theorem stating that the proposed representation is correct -/
theorem correct_scientific_notation :
  represents proposed_representation target_number :=
by sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l4004_400493


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l4004_400486

theorem unique_n_satisfying_conditions : ∃! (n : ℕ), n ≥ 1 ∧
  ∃ (a b : ℕ+), 
    (∀ (p : ℕ), Prime p → ¬(p^3 ∣ (a.val^2 + b.val + 3))) ∧
    ((a.val * b.val + 3 * b.val + 8) : ℚ) / (a.val^2 + b.val + 3) = n ∧
    n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l4004_400486


namespace NUMINAMATH_CALUDE_combinatorial_identities_l4004_400492

-- Define combinatorial choice function
def C (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define permutation function
def A (n m : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - m))

theorem combinatorial_identities :
  (3 * C 8 3 - 2 * C 5 2 = 148) ∧
  (∀ n m : ℕ, n ≥ m → m ≥ 2 → A n m = n * A (n-1) (m-1)) :=
by sorry

end NUMINAMATH_CALUDE_combinatorial_identities_l4004_400492


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4004_400423

theorem inequality_solution_set :
  {x : ℝ | (x + 1) * (2 - x) < 0} = {x : ℝ | x > 2 ∨ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4004_400423


namespace NUMINAMATH_CALUDE_factory_profit_l4004_400411

noncomputable section

-- Define the daily cost function
def C (x : ℝ) : ℝ := 3 + x

-- Define the daily sales revenue function
def S (x k : ℝ) : ℝ := 
  if 0 < x ∧ x < 6 then 3*x + k/(x-8) + 5
  else if x ≥ 6 then 14
  else 0  -- undefined for x ≤ 0

-- Define the daily profit function
def L (x k : ℝ) : ℝ := S x k - C x

-- State the theorem
theorem factory_profit (k : ℝ) :
  (L 2 k = 3) →  -- Condition: when x = 2, L = 3
  (k = 18 ∧ 
   ∀ x, 0 < x → L x k ≤ 6 ∧
   L 5 k = 6) := by
  sorry

end

end NUMINAMATH_CALUDE_factory_profit_l4004_400411


namespace NUMINAMATH_CALUDE_negation_is_false_l4004_400410

theorem negation_is_false : 
  ¬(∀ x y : ℝ, (x > 2 ∧ y > 3) → x + y > 5) = False := by sorry

end NUMINAMATH_CALUDE_negation_is_false_l4004_400410


namespace NUMINAMATH_CALUDE_extrema_and_tangent_line_l4004_400464

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem extrema_and_tangent_line :
  -- Local extrema conditions
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≤ f (-1)) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  -- Tangent line condition
  (∃ x₀ : ℝ, 9*x₀ - f x₀ + 16 = 0 ∧
    ∀ x : ℝ, 9*x - f x + 16 = 0 → x = x₀) :=
by sorry

end NUMINAMATH_CALUDE_extrema_and_tangent_line_l4004_400464


namespace NUMINAMATH_CALUDE_second_cat_blue_eyes_l4004_400465

/-- The number of blue-eyed kittens the first cat has -/
def first_cat_blue : ℕ := 3

/-- The number of brown-eyed kittens the first cat has -/
def first_cat_brown : ℕ := 7

/-- The number of brown-eyed kittens the second cat has -/
def second_cat_brown : ℕ := 6

/-- The percentage of kittens with blue eyes -/
def blue_eye_percentage : ℚ := 35 / 100

/-- The number of blue-eyed kittens the second cat has -/
def second_cat_blue : ℕ := 4

theorem second_cat_blue_eyes :
  (first_cat_blue + second_cat_blue : ℚ) / 
  (first_cat_blue + first_cat_brown + second_cat_blue + second_cat_brown) = 
  blue_eye_percentage := by
  sorry

#check second_cat_blue_eyes

end NUMINAMATH_CALUDE_second_cat_blue_eyes_l4004_400465


namespace NUMINAMATH_CALUDE_trees_in_yard_l4004_400445

/-- The number of trees planted along a yard. -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem: Given a yard 441 metres long with trees planted at equal distances,
    one tree at each end, and 21 metres between consecutive trees,
    there are 22 trees planted along the yard. -/
theorem trees_in_yard :
  let yard_length : ℕ := 441
  let tree_distance : ℕ := 21
  number_of_trees yard_length tree_distance = 22 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l4004_400445


namespace NUMINAMATH_CALUDE_simplify_fraction_l4004_400434

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (45 * b^3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4004_400434


namespace NUMINAMATH_CALUDE_johns_journey_distance_l4004_400429

/-- Calculates the total distance traveled by John given his journey conditions -/
def total_distance (
  initial_driving_speed : ℝ)
  (initial_driving_time : ℝ)
  (second_driving_speed : ℝ)
  (second_driving_time : ℝ)
  (biking_speed : ℝ)
  (biking_time : ℝ)
  (walking_speed : ℝ)
  (walking_time : ℝ) : ℝ :=
  initial_driving_speed * initial_driving_time +
  second_driving_speed * second_driving_time +
  biking_speed * biking_time +
  walking_speed * walking_time

/-- Theorem stating that John's total travel distance is 179 miles -/
theorem johns_journey_distance : 
  total_distance 55 2 45 1 15 1.5 3 0.5 = 179 := by
  sorry

end NUMINAMATH_CALUDE_johns_journey_distance_l4004_400429


namespace NUMINAMATH_CALUDE_ceiling_examples_ceiling_equals_two_m_range_equation_solutions_l4004_400425

-- Definition of the ceiling function for rational numbers
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Theorem 1: Calculating specific ceiling values
theorem ceiling_examples : ceiling (4.7) = 5 ∧ ceiling (-5.3) = -5 := by sorry

-- Theorem 2: Relationship when ceiling equals 2
theorem ceiling_equals_two (a : ℚ) : ceiling a = 2 ↔ 1 < a ∧ a ≤ 2 := by sorry

-- Theorem 3: Range of m satisfying the given condition
theorem m_range (m : ℚ) : ceiling (-2*m + 7) = -3 ↔ 5 ≤ m ∧ m < 5.5 := by sorry

-- Theorem 4: Solutions to the equation
theorem equation_solutions (n : ℚ) : ceiling (4.5*n - 2.5) = 3*n + 1 ↔ n = 2 ∨ n = 7/3 := by sorry

end NUMINAMATH_CALUDE_ceiling_examples_ceiling_equals_two_m_range_equation_solutions_l4004_400425


namespace NUMINAMATH_CALUDE_order_of_expressions_l4004_400487

-- Define the base of the logarithm
def b : Real := 0.2

-- State the theorem
theorem order_of_expressions (a : Real) (h : a > 1) :
  Real.log a / Real.log b < b * a ∧ b * a < a ^ b :=
sorry

end NUMINAMATH_CALUDE_order_of_expressions_l4004_400487


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4004_400472

/-- An arithmetic sequence with 2036 terms -/
def ArithmeticSequence (a d : ℝ) : ℕ → ℝ :=
  fun n => a + (n - 1) * d

theorem arithmetic_sequence_sum (a d : ℝ) :
  let t := ArithmeticSequence a d
  t 2018 = 100 →
  t 2000 + 5 * t 2015 + 5 * t 2021 + t 2036 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4004_400472


namespace NUMINAMATH_CALUDE_square_to_rectangle_ratio_l4004_400413

theorem square_to_rectangle_ratio : 
  ∀ (square_side : ℝ) (rectangle_base rectangle_height : ℝ),
  square_side = 3 →
  rectangle_base * rectangle_height = square_side^2 →
  rectangle_base = (square_side^2 + (square_side/2)^2).sqrt →
  (rectangle_height / rectangle_base) = 4/5 :=
by
  sorry

end NUMINAMATH_CALUDE_square_to_rectangle_ratio_l4004_400413


namespace NUMINAMATH_CALUDE_common_intersection_point_l4004_400403

-- Define a type for points in a plane
variable {Point : Type}

-- Define a type for half-planes
variable {HalfPlane : Type}

-- Define a function to check if a point is in a half-plane
variable (in_half_plane : Point → HalfPlane → Prop)

-- Define a set of half-planes
variable {S : Set HalfPlane}

-- Theorem statement
theorem common_intersection_point 
  (h : ∀ (a b c : HalfPlane), a ∈ S → b ∈ S → c ∈ S → 
    ∃ (p : Point), in_half_plane p a ∧ in_half_plane p b ∧ in_half_plane p c) :
  ∃ (p : Point), ∀ (h : HalfPlane), h ∈ S → in_half_plane p h :=
sorry

end NUMINAMATH_CALUDE_common_intersection_point_l4004_400403


namespace NUMINAMATH_CALUDE_quadratic_solution_average_l4004_400449

/-- Given a quadratic equation 2x^2 - 6x + c = 0 with two real solutions and discriminant 12,
    prove that the average of the solutions is 1.5 -/
theorem quadratic_solution_average (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 2 * x₁^2 - 6 * x₁ + c = 0 ∧ 2 * x₂^2 - 6 * x₂ + c = 0) →
  ((-6)^2 - 4 * 2 * c = 12) →
  (∃ x₁ x₂ : ℝ, 2 * x₁^2 - 6 * x₁ + c = 0 ∧ 2 * x₂^2 - 6 * x₂ + c = 0 ∧ (x₁ + x₂) / 2 = 1.5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_average_l4004_400449


namespace NUMINAMATH_CALUDE_rhombus_side_length_l4004_400428

/-- A rhombus with area K and one diagonal three times the length of the other has side length √(5K/3) -/
theorem rhombus_side_length (K : ℝ) (K_pos : K > 0) : 
  ∃ (d₁ d₂ s : ℝ), d₁ > 0 ∧ d₂ > 0 ∧ s > 0 ∧ 
  d₂ = 3 * d₁ ∧ 
  K = (1/2) * d₁ * d₂ ∧
  s^2 = (d₁/2)^2 + (d₂/2)^2 ∧
  s = Real.sqrt ((5 * K) / 3) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l4004_400428


namespace NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_factorial_ends_with_zeros_l4004_400453

def factorial_sum_last_two_digits : ℕ := 46

theorem last_two_digits_of_factorial_sum :
  let sum := List.sum (List.map Nat.factorial (List.range 25 |>.map (fun i => 4 * i + 3)))
  (sum % 100 = factorial_sum_last_two_digits) :=
by
  sorry

theorem factorial_ends_with_zeros (n : ℕ) (h : n ≥ 10) :
  ∃ k : ℕ, Nat.factorial n = 100 * k :=
by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_factorial_ends_with_zeros_l4004_400453


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l4004_400476

/-- The time taken by two cyclists meeting on the road -/
theorem cyclists_meeting_time (x y : ℝ) 
  (h1 : x - 4 = y - 9)  -- Time before meeting is equal for both cyclists
  (h2 : 4 / (y - 9) = (x - 4) / 9)  -- Proportion of speeds based on distances
  : x = 10 ∧ y = 15 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l4004_400476


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l4004_400442

theorem angle_inequality_equivalence (θ : Real) : 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → 
    x^3 * Real.sin θ + x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔ 
  (π / 12 < θ ∧ θ < 5 * π / 12) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l4004_400442


namespace NUMINAMATH_CALUDE_pond_water_after_50_days_l4004_400499

/-- Calculates the remaining water in a pond after a given number of days, considering evaporation. -/
def remaining_water (initial_water : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_water - evaporation_rate * days

/-- Theorem stating that a pond with 500 gallons of water, losing 1 gallon per day, will have 450 gallons after 50 days. -/
theorem pond_water_after_50_days :
  remaining_water 500 1 50 = 450 := by
  sorry

#eval remaining_water 500 1 50

end NUMINAMATH_CALUDE_pond_water_after_50_days_l4004_400499


namespace NUMINAMATH_CALUDE_subset_intersection_iff_bounds_l4004_400400

theorem subset_intersection_iff_bounds (a : ℝ) : 
  let A := {x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 22}
  (∃ x, x ∈ A) → (A ⊆ A ∩ B ↔ 6 ≤ a ∧ a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_subset_intersection_iff_bounds_l4004_400400


namespace NUMINAMATH_CALUDE_inverse_trig_sum_zero_l4004_400427

theorem inverse_trig_sum_zero : 
  Real.arctan (Real.sqrt 3 / 3) + Real.arcsin (-1/2) + Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_trig_sum_zero_l4004_400427


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4004_400481

/-- An isosceles triangle with sides a, b, and c, where a and b satisfy |a-2|+(b-5)^2=0 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)
  satisfiesEquation : |a - 2| + (b - 5)^2 = 0

/-- The perimeter of an isosceles triangle is 12 if it satisfies the given condition -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : t.a + t.b + t.c = 12 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4004_400481


namespace NUMINAMATH_CALUDE_second_reduction_percentage_l4004_400494

theorem second_reduction_percentage (P : ℝ) (R : ℝ) (h1 : P > 0) :
  (1 - R / 100) * (0.75 * P) = 0.375 * P →
  R = 50 := by
sorry

end NUMINAMATH_CALUDE_second_reduction_percentage_l4004_400494


namespace NUMINAMATH_CALUDE_expression_nonpositive_l4004_400466

theorem expression_nonpositive (x : ℝ) : (6 * x - 1) / 4 - 2 * x ≤ 0 ↔ x ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonpositive_l4004_400466


namespace NUMINAMATH_CALUDE_cookie_pie_ratio_is_seven_fourths_l4004_400498

/-- The ratio of students preferring cookies to those preferring pie -/
def cookie_pie_ratio (total_students : ℕ) (cookie_preference : ℕ) (pie_preference : ℕ) : ℚ :=
  cookie_preference / pie_preference

theorem cookie_pie_ratio_is_seven_fourths :
  cookie_pie_ratio 800 280 160 = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_pie_ratio_is_seven_fourths_l4004_400498


namespace NUMINAMATH_CALUDE_triangle_side_length_l4004_400406

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a = 3 ∧ c = 5 ∧ B = 2 * A ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  b = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4004_400406


namespace NUMINAMATH_CALUDE_right_triangle_area_l4004_400452

theorem right_triangle_area (a b : ℝ) (h1 : a^2 - 7*a + 12 = 0) (h2 : b^2 - 7*b + 12 = 0) (h3 : a ≠ b) :
  ∃ (area : ℝ), (area = 6 ∨ area = (3 * Real.sqrt 7) / 2) ∧
  ((area = a * b / 2) ∨ (area = a * Real.sqrt (b^2 - a^2) / 2) ∨ (area = b * Real.sqrt (a^2 - b^2) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l4004_400452


namespace NUMINAMATH_CALUDE_head_start_calculation_l4004_400412

/-- Represents a runner in the race -/
inductive Runner : Type
  | A | B | C | D

/-- The head start (in meters) that Runner A can give to another runner -/
def headStart (r : Runner) : ℕ :=
  match r with
  | Runner.A => 0
  | Runner.B => 150
  | Runner.C => 310
  | Runner.D => 400

/-- The head start one runner can give to another -/
def headStartBetween (r1 r2 : Runner) : ℤ :=
  (headStart r2 : ℤ) - (headStart r1 : ℤ)

theorem head_start_calculation :
  (headStartBetween Runner.B Runner.C = 160) ∧
  (headStartBetween Runner.C Runner.D = 90) ∧
  (headStartBetween Runner.B Runner.D = 250) := by
  sorry

#check head_start_calculation

end NUMINAMATH_CALUDE_head_start_calculation_l4004_400412


namespace NUMINAMATH_CALUDE_book_sale_earnings_l4004_400474

/-- Calculates the total earnings from a book sale --/
theorem book_sale_earnings (total_books : ℕ) (price_high : ℚ) (price_low : ℚ) : 
  total_books = 10 ∧ 
  price_high = 5/2 ∧ 
  price_low = 2 → 
  (2/5 * total_books : ℚ) * price_high + (3/5 * total_books : ℚ) * price_low = 22 := by
  sorry

#check book_sale_earnings

end NUMINAMATH_CALUDE_book_sale_earnings_l4004_400474


namespace NUMINAMATH_CALUDE_card_arrangement_probability_l4004_400490

theorem card_arrangement_probability : 
  let total_arrangements : ℕ := 24
  let favorable_arrangements : ℕ := 2
  let probability : ℚ := favorable_arrangements / total_arrangements
  probability = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_card_arrangement_probability_l4004_400490


namespace NUMINAMATH_CALUDE_lines_perpendicular_iff_product_slopes_neg_one_l4004_400448

/-- Two lines y = k₁x + l₁ and y = k₂x + l₂, where k₁ ≠ 0 and k₂ ≠ 0, are perpendicular if and only if k₁k₂ = -1 -/
theorem lines_perpendicular_iff_product_slopes_neg_one
  (k₁ k₂ l₁ l₂ : ℝ) (hk₁ : k₁ ≠ 0) (hk₂ : k₂ ≠ 0) :
  (∃ (x y : ℝ), y = k₁ * x + l₁ ∧ y = k₂ * x + l₂) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    y₁ = k₁ * x₁ + l₁ →
    y₂ = k₂ * x₂ + l₂ →
    (x₂ - x₁) * (y₂ - y₁) = 0) ↔
  k₁ * k₂ = -1 := by
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_iff_product_slopes_neg_one_l4004_400448


namespace NUMINAMATH_CALUDE_tangency_points_form_cyclic_quadrilateral_l4004_400469

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being externally tangent
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define a point of tangency between two circles
def tangency_point (c1 c2 : Circle) : ℝ × ℝ :=
  sorry

-- Define the property of a quadrilateral being cyclic
def is_cyclic_quadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  sorry

-- Theorem statement
theorem tangency_points_form_cyclic_quadrilateral 
  (S1 S2 S3 S4 : Circle)
  (h12 : externally_tangent S1 S2)
  (h23 : externally_tangent S2 S3)
  (h34 : externally_tangent S3 S4)
  (h41 : externally_tangent S4 S1) :
  let p1 := tangency_point S1 S2
  let p2 := tangency_point S2 S3
  let p3 := tangency_point S3 S4
  let p4 := tangency_point S4 S1
  is_cyclic_quadrilateral p1 p2 p3 p4 :=
by
  sorry

end NUMINAMATH_CALUDE_tangency_points_form_cyclic_quadrilateral_l4004_400469


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l4004_400404

/-- Given a hyperbola with equation x^2 - y^2/m^2 = 1 where m > 0,
    if one of its asymptotes is x + √3 * y = 0, then m = √3/3 -/
theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) 
  (h2 : ∃ (x y : ℝ), x^2 - y^2/m^2 = 1 ∧ x + Real.sqrt 3 * y = 0) : 
  m = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l4004_400404


namespace NUMINAMATH_CALUDE_complex_subtraction_l4004_400414

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + I) :
  a - 3*b = -1 - 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l4004_400414


namespace NUMINAMATH_CALUDE_jellybean_count_l4004_400479

theorem jellybean_count (black green orange : ℕ) 
  (green_count : green = black + 2)
  (orange_count : orange = green - 1)
  (total_count : black + green + orange = 27) :
  black = 8 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l4004_400479


namespace NUMINAMATH_CALUDE_system_equations_solutions_l4004_400401

theorem system_equations_solutions (a x y : ℝ) : 
  (x - y = 2*a + 1 ∧ 2*x + 3*y = 9*a - 8) →
  ((x = y → a = -1/2) ∧
   (x > 0 ∧ y < 0 ∧ x + y = 0 → a = 3/4)) := by
  sorry

end NUMINAMATH_CALUDE_system_equations_solutions_l4004_400401


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l4004_400419

-- Define the universal set U
def U : Set ℕ := {0, 1, 2}

-- Define set A
def A : Set ℕ := {x ∈ U | x^2 - x = 0}

-- State the theorem
theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l4004_400419


namespace NUMINAMATH_CALUDE_distance_ratio_l4004_400462

-- Define the speeds and time for both cars
def speed_A : ℝ := 70
def speed_B : ℝ := 35
def time : ℝ := 10

-- Define the distances traveled by each car
def distance_A : ℝ := speed_A * time
def distance_B : ℝ := speed_B * time

-- Theorem to prove the ratio of distances
theorem distance_ratio :
  distance_A / distance_B = 2 := by sorry

end NUMINAMATH_CALUDE_distance_ratio_l4004_400462


namespace NUMINAMATH_CALUDE_count_four_digit_with_seven_l4004_400455

/-- A four-digit positive integer with 7 as the thousands digit -/
def FourDigitWithSeven : Type := { n : ℕ // 7000 ≤ n ∧ n ≤ 7999 }

/-- The count of four-digit positive integers with 7 as the thousands digit -/
def CountFourDigitWithSeven : ℕ := Finset.card (Finset.filter (λ n => 7000 ≤ n ∧ n ≤ 7999) (Finset.range 10000))

theorem count_four_digit_with_seven :
  CountFourDigitWithSeven = 1000 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_with_seven_l4004_400455


namespace NUMINAMATH_CALUDE_range_of_g_l4004_400440

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_g :
  Set.range g = {y | -π/2 ≤ y ∧ y ≤ Real.arctan 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l4004_400440


namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l4004_400454

/-- Given two angles x and y in a quadrilateral satisfying certain conditions,
    prove that x equals (1 + √13) / 6 degrees. -/
theorem quadrilateral_angle_measure (x y : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x = a ∧ y = b) →  -- x and y are positive real numbers (representing angles)
  (3 * x^2 - x + 4 = 5) →                        -- First condition
  (x^2 + y^2 = 9) →                              -- Second condition
  x = (1 + Real.sqrt 13) / 6 :=                  -- Conclusion
by sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l4004_400454


namespace NUMINAMATH_CALUDE_intersection_integer_coordinates_l4004_400407

theorem intersection_integer_coordinates (n : ℕ+) 
  (h : ∃ (x y : ℤ), 17 * x + 7 * y = 833 ∧ y = n * x - 3) : n = 15 := by
  sorry

end NUMINAMATH_CALUDE_intersection_integer_coordinates_l4004_400407


namespace NUMINAMATH_CALUDE_max_mineral_worth_l4004_400426

-- Define the mineral types
inductive Mineral
| Sapphire
| Ruby
| Emerald

-- Define the properties of each mineral
def weight (m : Mineral) : Nat :=
  match m with
  | Mineral.Sapphire => 6
  | Mineral.Ruby => 3
  | Mineral.Emerald => 2

def value (m : Mineral) : Nat :=
  match m with
  | Mineral.Sapphire => 18
  | Mineral.Ruby => 9
  | Mineral.Emerald => 4

-- Define the maximum carrying capacity
def maxWeight : Nat := 20

-- Define the minimum available quantity of each mineral
def minQuantity : Nat := 30

-- Define a function to calculate the total weight of a combination of minerals
def totalWeight (s r e : Nat) : Nat :=
  s * weight Mineral.Sapphire + r * weight Mineral.Ruby + e * weight Mineral.Emerald

-- Define a function to calculate the total value of a combination of minerals
def totalValue (s r e : Nat) : Nat :=
  s * value Mineral.Sapphire + r * value Mineral.Ruby + e * value Mineral.Emerald

-- Theorem: The maximum worth of minerals Joe can carry is $58
theorem max_mineral_worth :
  ∃ s r e : Nat,
    s ≤ minQuantity ∧ r ≤ minQuantity ∧ e ≤ minQuantity ∧
    totalWeight s r e ≤ maxWeight ∧
    totalValue s r e = 58 ∧
    ∀ s' r' e' : Nat,
      s' ≤ minQuantity → r' ≤ minQuantity → e' ≤ minQuantity →
      totalWeight s' r' e' ≤ maxWeight →
      totalValue s' r' e' ≤ 58 :=
by sorry

end NUMINAMATH_CALUDE_max_mineral_worth_l4004_400426


namespace NUMINAMATH_CALUDE_jose_profit_share_l4004_400460

/-- Calculates the share of profit for an investor in a partnership --/
def calculate_profit_share (investment1 : ℕ) (months1 : ℕ) (investment2 : ℕ) (months2 : ℕ) (total_profit : ℕ) : ℕ :=
  let total_investment := investment1 * months1 + investment2 * months2
  let share_ratio := investment2 * months2 * total_profit / total_investment
  share_ratio

/-- Proves that Jose's share of the profit is 3500 --/
theorem jose_profit_share :
  calculate_profit_share 3000 12 4500 10 6300 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_jose_profit_share_l4004_400460


namespace NUMINAMATH_CALUDE_prob_open_door_third_attempt_l4004_400421

/-- Probability of opening a door on the third attempt given 5 keys with only one correct key -/
theorem prob_open_door_third_attempt (total_keys : ℕ) (correct_keys : ℕ) (attempt : ℕ) :
  total_keys = 5 →
  correct_keys = 1 →
  attempt = 3 →
  (1 : ℚ) / total_keys = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_open_door_third_attempt_l4004_400421


namespace NUMINAMATH_CALUDE_unique_four_digit_square_with_repeated_digits_l4004_400405

/-- A four-digit number with repeated first two digits and last two digits -/
def FourDigitRepeated (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = 1100 * a + 11 * b

/-- The property of being a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem unique_four_digit_square_with_repeated_digits : 
  ∃! n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ FourDigitRepeated n ∧ IsPerfectSquare n ∧ n = 7744 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_with_repeated_digits_l4004_400405


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l4004_400496

/-- Given vectors a and b, if ka + b is perpendicular to a, then k = 2/5 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : b = (-2, 0)) 
  (h3 : (k • a.1 + b.1) * a.1 + (k • a.2 + b.2) * a.2 = 0) : 
  k = 2/5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l4004_400496


namespace NUMINAMATH_CALUDE_cereal_box_servings_l4004_400422

theorem cereal_box_servings (total_cups : ℕ) (serving_size : ℕ) (h1 : total_cups = 18) (h2 : serving_size = 2) :
  total_cups / serving_size = 9 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_servings_l4004_400422


namespace NUMINAMATH_CALUDE_kaylaScoreEighthLevel_l4004_400473

/-- Fibonacci sequence starting with 1 and 1 -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Kayla's score at a given level -/
def kaylaScore : ℕ → ℤ
  | 0 => 2
  | n + 1 => if n % 2 = 0 then kaylaScore n - fib n else kaylaScore n + fib n

theorem kaylaScoreEighthLevel : kaylaScore 7 = -7 := by
  sorry

end NUMINAMATH_CALUDE_kaylaScoreEighthLevel_l4004_400473


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l4004_400495

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 5 ∧ Real.sqrt (x + 1) + Real.sqrt (y + 3) ≥ Real.sqrt (a + 1) + Real.sqrt (b + 3)) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 5 → Real.sqrt (x + 1) + Real.sqrt (y + 3) ≤ 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l4004_400495


namespace NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l4004_400444

/-- A sequence of zeros and ones -/
def BinarySequence := List Bool

/-- Count pairs (1,0) with even number of digits between them -/
def countEvenPairs (seq : BinarySequence) : ℕ := sorry

/-- Count pairs (1,0) with odd number of digits between them -/
def countOddPairs (seq : BinarySequence) : ℕ := sorry

/-- Theorem: In any binary sequence, the number of (1,0) pairs with even digits between
    is greater than or equal to the number of (1,0) pairs with odd digits between -/
theorem even_pairs_ge_odd_pairs (seq : BinarySequence) :
  countEvenPairs seq ≥ countOddPairs seq := by
  sorry

end NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l4004_400444


namespace NUMINAMATH_CALUDE_no_solution_exists_l4004_400409

theorem no_solution_exists : ¬ ∃ n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 10467 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l4004_400409


namespace NUMINAMATH_CALUDE_tommy_calculation_l4004_400432

theorem tommy_calculation (x : ℚ) : (x - 7) / 5 = 23 → (x - 5) / 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_tommy_calculation_l4004_400432


namespace NUMINAMATH_CALUDE_x_minus_y_values_l4004_400418

theorem x_minus_y_values (x y : ℝ) 
  (hx : |x| = 4) 
  (hy : |y| = 2) 
  (hxy : |x + y| = x + y) : 
  x - y = 2 ∨ x - y = 6 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l4004_400418


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_7_8_9_l4004_400433

theorem six_digit_divisible_by_7_8_9 :
  ∃ (n₁ n₂ : ℕ),
    523000 ≤ n₁ ∧ n₁ < 524000 ∧
    523000 ≤ n₂ ∧ n₂ < 524000 ∧
    n₁ ≠ n₂ ∧
    n₁ % 7 = 0 ∧ n₁ % 8 = 0 ∧ n₁ % 9 = 0 ∧
    n₂ % 7 = 0 ∧ n₂ % 8 = 0 ∧ n₂ % 9 = 0 ∧
    n₁ = 523152 ∧ n₂ = 523656 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_7_8_9_l4004_400433


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solution_l4004_400447

theorem cubic_equation_integer_solution :
  ∃! (x : ℤ), 2 * x^3 + 5 * x^2 - 9 * x - 18 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solution_l4004_400447


namespace NUMINAMATH_CALUDE_loom_weaving_time_l4004_400441

/-- The rate at which the loom weaves cloth in meters per second -/
def weaving_rate : ℝ := 0.128

/-- The time it takes to weave 15 meters of cloth in seconds -/
def time_for_15_meters : ℝ := 117.1875

/-- The amount of cloth woven in 15 meters -/
def cloth_amount : ℝ := 15

theorem loom_weaving_time (C : ℝ) :
  C ≥ 0 →
  weaving_rate > 0 →
  time_for_15_meters * weaving_rate = cloth_amount →
  C / weaving_rate = (C : ℝ) / 0.128 := by
  sorry

end NUMINAMATH_CALUDE_loom_weaving_time_l4004_400441


namespace NUMINAMATH_CALUDE_correct_dispersion_measure_l4004_400450

-- Define a type for measures of data dispersion
structure DisperesionMeasure where
  makeFullUseOfData : Bool
  useMultipleNumericalValues : Bool
  smallerValueForLargerDispersion : Bool

-- Define a function to check if a dispersion measure is correct
def isCorrectMeasure (m : DisperesionMeasure) : Prop :=
  m.makeFullUseOfData ∧ m.useMultipleNumericalValues ∧ ¬m.smallerValueForLargerDispersion

-- Theorem: The correct dispersion measure makes full use of data and uses multiple numerical values
theorem correct_dispersion_measure :
  ∃ (m : DisperesionMeasure), isCorrectMeasure m ∧
    m.makeFullUseOfData = true ∧
    m.useMultipleNumericalValues = true :=
  sorry


end NUMINAMATH_CALUDE_correct_dispersion_measure_l4004_400450


namespace NUMINAMATH_CALUDE_spend_representation_l4004_400459

-- Define a type for monetary transactions
inductive Transaction
| receive (amount : ℤ)
| spend (amount : ℤ)

-- Define a function to represent transactions
def represent : Transaction → ℤ
| Transaction.receive amount => amount
| Transaction.spend amount => -amount

-- Theorem statement
theorem spend_representation (amount : ℤ) :
  represent (Transaction.receive amount) = amount →
  represent (Transaction.spend amount) = -amount :=
by
  sorry

end NUMINAMATH_CALUDE_spend_representation_l4004_400459


namespace NUMINAMATH_CALUDE_inequalities_for_positive_sum_two_l4004_400402

theorem inequalities_for_positive_sum_two (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  a^2 + b^2 ≥ 2 ∧ 1/a + 1/b ≥ 2 := by sorry

end NUMINAMATH_CALUDE_inequalities_for_positive_sum_two_l4004_400402


namespace NUMINAMATH_CALUDE_local_max_implies_c_eq_6_l4004_400438

/-- The function f(x) = x(x-c)^2 -/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- f has a local maximum at x = 2 -/
def has_local_max_at_2 (c : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → f c x ≤ f c 2

theorem local_max_implies_c_eq_6 :
  ∀ c : ℝ, has_local_max_at_2 c → c = 6 := by sorry

end NUMINAMATH_CALUDE_local_max_implies_c_eq_6_l4004_400438


namespace NUMINAMATH_CALUDE_circle_op_five_three_l4004_400420

-- Define the operation ∘
def circle_op (a b : ℕ) : ℕ := 4*a + 6*b + 1

-- State the theorem
theorem circle_op_five_three : circle_op 5 3 = 39 := by sorry

end NUMINAMATH_CALUDE_circle_op_five_three_l4004_400420


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l4004_400470

def q (x : ℚ) : ℚ := -20/93 * x^3 - 110/93 * x^2 - 372/93 * x - 525/93

theorem cubic_polynomial_satisfies_conditions :
  q 1 = -11 ∧ q 2 = -15 ∧ q 3 = -25 ∧ q 5 = -65 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l4004_400470


namespace NUMINAMATH_CALUDE_minimal_difference_factors_l4004_400485

theorem minimal_difference_factors : ∃ (a b : ℤ),
  a * b = 1234567890 ∧
  ∀ (x y : ℤ), x * y = 1234567890 → |x - y| ≥ |a - b| ∧
  a = 36070 ∧ b = 34227 := by sorry

end NUMINAMATH_CALUDE_minimal_difference_factors_l4004_400485


namespace NUMINAMATH_CALUDE_jack_letters_difference_l4004_400458

theorem jack_letters_difference (morning_emails morning_letters afternoon_emails afternoon_letters : ℕ) :
  morning_emails = 6 →
  morning_letters = 8 →
  afternoon_emails = 2 →
  afternoon_letters = 7 →
  morning_letters - afternoon_letters = 1 := by
  sorry

end NUMINAMATH_CALUDE_jack_letters_difference_l4004_400458


namespace NUMINAMATH_CALUDE_purchase_payment_possible_l4004_400457

theorem purchase_payment_possible :
  ∃ (x y : ℕ), x ≤ 15 ∧ y ≤ 15 ∧ 3 * x - 5 * y = 19 :=
sorry

end NUMINAMATH_CALUDE_purchase_payment_possible_l4004_400457


namespace NUMINAMATH_CALUDE_range_of_H_l4004_400456

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

-- State the theorem about the range of H
theorem range_of_H :
  Set.range H = Set.Icc 1 5 := by sorry

end NUMINAMATH_CALUDE_range_of_H_l4004_400456


namespace NUMINAMATH_CALUDE_area_increase_6_to_7_l4004_400435

/-- Calculates the increase in area of a square when its side length is increased by 1 unit -/
def area_increase (side_length : ℝ) : ℝ :=
  (side_length + 1)^2 - side_length^2

/-- Theorem: The increase in area of a square with side length 6 units, 
    when increased by 1 unit, is 13 square units -/
theorem area_increase_6_to_7 : area_increase 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_area_increase_6_to_7_l4004_400435


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l4004_400461

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : 
  train_length = 410 ∧ 
  train_speed_kmh = 45 ∧ 
  time_to_pass = 44 → 
  (train_speed_kmh * 1000 / 3600) * time_to_pass - train_length = 140 := by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l4004_400461


namespace NUMINAMATH_CALUDE_range_of_p_l4004_400463

open Set

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 10*x

def A : Set ℝ := {x | (deriv f) x ≤ 0}

def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem range_of_p (p : ℝ) (h : A ∪ B p = A) : p ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l4004_400463


namespace NUMINAMATH_CALUDE_y₁_gt_y₂_l4004_400437

/-- The quadratic function f(x) = x² - 4x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 3

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (-1, f (-1))

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (1, f 1)

/-- y₁ coordinate of point A -/
def y₁ : ℝ := A.2

/-- y₂ coordinate of point B -/
def y₂ : ℝ := B.2

theorem y₁_gt_y₂ : y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_gt_y₂_l4004_400437


namespace NUMINAMATH_CALUDE_factor_expression_l4004_400483

theorem factor_expression (b : ℝ) : 49 * b^2 + 98 * b = 49 * b * (b + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4004_400483


namespace NUMINAMATH_CALUDE_power_equation_solution_l4004_400484

theorem power_equation_solution (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^42 → n = 42 := by
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l4004_400484


namespace NUMINAMATH_CALUDE_streetlight_problem_l4004_400467

/-- The number of ways to select k non-adjacent items from a sequence of n items,
    excluding the first and last items. -/
def non_adjacent_selections (n k : ℕ) : ℕ :=
  Nat.choose (n - k - 1) k

/-- The problem statement -/
theorem streetlight_problem :
  non_adjacent_selections 12 3 = Nat.choose 7 3 := by
  sorry

end NUMINAMATH_CALUDE_streetlight_problem_l4004_400467


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l4004_400415

theorem rectangular_plot_length 
  (metallic_cost : ℝ) 
  (wooden_cost : ℝ) 
  (gate_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : metallic_cost = 26.5)
  (h2 : wooden_cost = 22)
  (h3 : gate_cost = 240)
  (h4 : total_cost = 5600) :
  ∃ (breadth length : ℝ),
    length = breadth + 14 ∧ 
    (2 * length + breadth) * metallic_cost + breadth * wooden_cost + gate_cost = total_cost ∧
    length = 59.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l4004_400415


namespace NUMINAMATH_CALUDE_sphere_radius_in_cone_l4004_400488

/-- A right circular cone with a base radius of 7 and height of 15 -/
structure Cone where
  baseRadius : ℝ := 7
  height : ℝ := 15

/-- A sphere with radius r -/
structure Sphere (r : ℝ) where

/-- Configuration of four spheres in the cone -/
structure SphereConfiguration (r : ℝ) where
  cone : Cone
  spheres : Fin 4 → Sphere r
  bottomThreeTangent : Bool
  bottomThreeTouchBase : Bool
  bottomThreeTouchSide : Bool
  topSphereTouchesOthers : Bool
  topSphereTouchesSide : Bool
  topSphereNotTouchBase : Bool

/-- The theorem stating the radius of the spheres in the given configuration -/
theorem sphere_radius_in_cone (config : SphereConfiguration r) :
  r = (162 - 108 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_in_cone_l4004_400488


namespace NUMINAMATH_CALUDE_park_breadth_l4004_400417

/-- The breadth of a rectangular park given its perimeter and length -/
theorem park_breadth (perimeter length breadth : ℝ) : 
  perimeter = 1000 →
  length = 300 →
  perimeter = 2 * (length + breadth) →
  breadth = 200 := by
sorry

end NUMINAMATH_CALUDE_park_breadth_l4004_400417


namespace NUMINAMATH_CALUDE_Al2O3_weight_and_H2_volume_l4004_400468

/-- Molar mass of Aluminum in g/mol -/
def molar_mass_Al : ℝ := 26.98

/-- Molar mass of Oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Volume occupied by 1 mole of gas at STP in liters -/
def molar_volume_STP : ℝ := 22.4

/-- Molar mass of Al2O3 in g/mol -/
def molar_mass_Al2O3 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_O

/-- Number of moles of Al2O3 -/
def moles_Al2O3 : ℝ := 6

/-- Theorem stating the weight of Al2O3 and volume of H2 produced -/
theorem Al2O3_weight_and_H2_volume :
  (moles_Al2O3 * molar_mass_Al2O3 = 611.76) ∧
  (moles_Al2O3 * 3 * molar_volume_STP = 403.2) := by
  sorry

end NUMINAMATH_CALUDE_Al2O3_weight_and_H2_volume_l4004_400468


namespace NUMINAMATH_CALUDE_slope_of_line_l4004_400443

/-- The slope of a line given by the equation 4y = 5x - 8 is 5/4 -/
theorem slope_of_line (x y : ℝ) : 4 * y = 5 * x - 8 → (∃ m b : ℝ, y = m * x + b ∧ m = 5 / 4) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l4004_400443


namespace NUMINAMATH_CALUDE_modulus_of_z_l4004_400430

/-- The modulus of the complex number z = 1 / (i - 1) is equal to √2/2 -/
theorem modulus_of_z (z : ℂ) : z = 1 / (Complex.I - 1) → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l4004_400430


namespace NUMINAMATH_CALUDE_os_value_l4004_400424

/-- Square with center and points on its diagonals -/
structure SquareWithPoints where
  /-- Side length of the square -/
  a : ℝ
  /-- Center of the square -/
  O : ℝ × ℝ
  /-- Point P on OA -/
  P : ℝ × ℝ
  /-- Point Q on OB -/
  Q : ℝ × ℝ
  /-- Point R on OC -/
  R : ℝ × ℝ
  /-- Point S on OD -/
  S : ℝ × ℝ
  /-- A is a vertex of the square -/
  A : ℝ × ℝ
  /-- B is a vertex of the square -/
  B : ℝ × ℝ
  /-- C is a vertex of the square -/
  C : ℝ × ℝ
  /-- D is a vertex of the square -/
  D : ℝ × ℝ
  /-- O is the center of the square ABCD -/
  h_center : O = (0, 0)
  /-- ABCD is a square with side length 2a -/
  h_square : A = (-a, a) ∧ B = (a, a) ∧ C = (a, -a) ∧ D = (-a, -a)
  /-- P is on OA with OP = 3 -/
  h_P : P = (-3*a/Real.sqrt 2, 3*a/Real.sqrt 2) ∧ Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = 3
  /-- Q is on OB with OQ = 5 -/
  h_Q : Q = (5*a/Real.sqrt 2, 5*a/Real.sqrt 2) ∧ Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) = 5
  /-- R is on OC with OR = 4 -/
  h_R : R = (4*a/Real.sqrt 2, -4*a/Real.sqrt 2) ∧ Real.sqrt ((R.1 - O.1)^2 + (R.2 - O.2)^2) = 4
  /-- S is on OD -/
  h_S : ∃ x : ℝ, S = (-x*a/Real.sqrt 2, -x*a/Real.sqrt 2)
  /-- X is the intersection of AB and PQ -/
  h_X : ∃ X : ℝ × ℝ, X.2 = a ∧ X.2 = (1/4)*X.1 + 15*a/(4*Real.sqrt 2)
  /-- Y is the intersection of BC and QR -/
  h_Y : ∃ Y : ℝ × ℝ, Y.1 = a ∧ Y.2 = 9*Y.1 - 40*a/Real.sqrt 2
  /-- Z is the intersection of CD and RS -/
  h_Z : ∃ Z : ℝ × ℝ, Z.2 = -a ∧ Z.2 + 4*a/Real.sqrt 2 = (4*a - S.1)/(-(4*a + S.1)) * (Z.1 - 4*a/Real.sqrt 2)
  /-- X, Y, and Z are collinear -/
  h_collinear : ∀ X Y Z : ℝ × ℝ, 
    (X.2 = a ∧ X.2 = (1/4)*X.1 + 15*a/(4*Real.sqrt 2)) →
    (Y.1 = a ∧ Y.2 = 9*Y.1 - 40*a/Real.sqrt 2) →
    (Z.2 = -a ∧ Z.2 + 4*a/Real.sqrt 2 = (4*a - S.1)/(-(4*a + S.1)) * (Z.1 - 4*a/Real.sqrt 2)) →
    (Y.2 - X.2)*(Z.1 - X.1) = (Z.2 - X.2)*(Y.1 - X.1)

/-- The main theorem: OS = 60/23 -/
theorem os_value (sq : SquareWithPoints) : 
  Real.sqrt ((sq.S.1 - sq.O.1)^2 + (sq.S.2 - sq.O.2)^2) = 60/23 := by
  sorry

end NUMINAMATH_CALUDE_os_value_l4004_400424


namespace NUMINAMATH_CALUDE_inequality_proof_l4004_400497

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  1 / (x^2 + y + z) + 1 / (x + y^2 + z) + 1 / (x + y + z^2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4004_400497


namespace NUMINAMATH_CALUDE_fourth_month_sale_l4004_400446

theorem fourth_month_sale
  (sale1 sale2 sale3 sale5 sale6 : ℕ)
  (average_sale : ℕ)
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 6191)
  (h_avg : average_sale = 6700)
  (h_total : average_sale * 6 = sale1 + sale2 + sale3 + sale5 + sale6 + sale4) :
  sale4 = 7230 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l4004_400446


namespace NUMINAMATH_CALUDE_opposite_reciprocal_equation_l4004_400439

theorem opposite_reciprocal_equation (a b c d : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposites
  (h2 : c * d = 1)  -- c and d are reciprocals
  : (a + b)^2 - 3*(c*d)^4 = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_equation_l4004_400439


namespace NUMINAMATH_CALUDE_fraction_powers_equality_l4004_400436

theorem fraction_powers_equality : (0.4 ^ 4) / (0.04 ^ 3) = 400 := by
  sorry

end NUMINAMATH_CALUDE_fraction_powers_equality_l4004_400436


namespace NUMINAMATH_CALUDE_disrespectful_quadratic_max_root_sum_l4004_400491

/-- A quadratic polynomial with real coefficients and leading coefficient 1 -/
def QuadraticPolynomial (b c : ℝ) := fun (x : ℝ) ↦ x^2 + b*x + c

/-- The condition for a polynomial to be "disrespectful" -/
def isDisrespectful (p : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x : ℝ, p (p x) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)

/-- The sum of roots of a quadratic polynomial -/
def sumOfRoots (b c : ℝ) : ℝ := -b

theorem disrespectful_quadratic_max_root_sum (b c : ℝ) :
  let p := QuadraticPolynomial b c
  isDisrespectful p ∧ 
  (∀ b' c' : ℝ, isDisrespectful (QuadraticPolynomial b' c') → sumOfRoots b' c' ≤ sumOfRoots b c) →
  p 1 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_disrespectful_quadratic_max_root_sum_l4004_400491


namespace NUMINAMATH_CALUDE_total_power_cost_l4004_400480

/-- Represents the cost of power for each appliance in Joseph's house --/
structure ApplianceCosts where
  waterHeater : ℝ
  refrigerator : ℝ
  electricOven : ℝ
  airConditioner : ℝ
  washingMachine : ℝ

/-- Calculates the total cost of power for all appliances --/
def totalCost (costs : ApplianceCosts) : ℝ :=
  costs.waterHeater + costs.refrigerator + costs.electricOven + costs.airConditioner + costs.washingMachine

/-- Theorem stating the total cost of power for all appliances --/
theorem total_power_cost (costs : ApplianceCosts) 
  (h1 : costs.refrigerator = 3 * costs.waterHeater)
  (h2 : costs.electricOven = 500)
  (h3 : costs.electricOven = 2.5 * costs.waterHeater)
  (h4 : costs.airConditioner = 300)
  (h5 : costs.washingMachine = 100) :
  totalCost costs = 1700 := by
  sorry


end NUMINAMATH_CALUDE_total_power_cost_l4004_400480


namespace NUMINAMATH_CALUDE_slope_product_theorem_l4004_400416

theorem slope_product_theorem (m n : ℝ) (θ₁ θ₂ : ℝ) : 
  θ₁ = 3 * θ₂ →
  m = 9 * n →
  m ≠ 0 →
  m = Real.tan θ₁ →
  n = Real.tan θ₂ →
  m * n = 27 / 13 :=
by sorry

end NUMINAMATH_CALUDE_slope_product_theorem_l4004_400416


namespace NUMINAMATH_CALUDE_workshop_efficiency_l4004_400451

theorem workshop_efficiency (x : ℝ) : 
  (1500 / x - 1500 / (2.5 * x) = 18) → x = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_efficiency_l4004_400451


namespace NUMINAMATH_CALUDE_total_winter_clothing_l4004_400475

/-- Represents the contents of a box of winter clothing -/
structure BoxContents where
  scarves : ℕ
  mittens : ℕ
  hats : ℕ

/-- Calculates the total number of items in a box -/
def totalItemsInBox (box : BoxContents) : ℕ :=
  box.scarves + box.mittens + box.hats

/-- The contents of the four boxes -/
def box1 : BoxContents := { scarves := 3, mittens := 5, hats := 2 }
def box2 : BoxContents := { scarves := 4, mittens := 3, hats := 1 }
def box3 : BoxContents := { scarves := 2, mittens := 6, hats := 3 }
def box4 : BoxContents := { scarves := 1, mittens := 7, hats := 2 }

/-- Theorem stating that the total number of winter clothing items is 39 -/
theorem total_winter_clothing : 
  totalItemsInBox box1 + totalItemsInBox box2 + totalItemsInBox box3 + totalItemsInBox box4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l4004_400475


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l4004_400471

theorem cubic_equation_solution :
  ∃ y : ℝ, y > 0 ∧ 6 * y^(1/3) - 3 * (y / y^(2/3)) = 12 + 2 * y^(1/3) ∧ y = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l4004_400471


namespace NUMINAMATH_CALUDE_first_player_always_wins_l4004_400477

/-- Represents a position on the rectangular table -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents the state of the game -/
structure GameState :=
  (table : Set Position)
  (occupied : Set Position)
  (currentPlayer : Bool)

/-- The winning strategy for the first player -/
def firstPlayerStrategy (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Position),
    ∀ (state : GameState),
      state.currentPlayer = true →
      strategy state ∉ state.occupied →
      strategy state ∈ state.table

/-- The main theorem stating that the first player always has a winning strategy -/
theorem first_player_always_wins :
  ∀ (initialState : GameState),
    initialState.occupied = ∅ →
    initialState.table.Nonempty →
    ∃ (center : Position), center ∈ initialState.table →
      firstPlayerStrategy initialState :=
sorry

end NUMINAMATH_CALUDE_first_player_always_wins_l4004_400477


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_even_integers_l4004_400431

theorem largest_divisor_of_five_consecutive_even_integers (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (d : ℕ), d = 96 ∧
  (∀ (k : ℕ), k > 96 → ¬(k ∣ n * (n + 2) * (n + 4) * (n + 6) * (n + 8))) ∧
  (96 ∣ n * (n + 2) * (n + 4) * (n + 6) * (n + 8)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_even_integers_l4004_400431


namespace NUMINAMATH_CALUDE_octahedron_sum_theorem_l4004_400489

/-- Represents an octahedron with numbered faces -/
structure NumberedOctahedron where
  lowest_number : ℕ
  face_count : ℕ
  is_consecutive : Bool
  opposite_faces_diff : ℕ

/-- The sum of numbers on an octahedron with the given properties -/
def octahedron_sum (o : NumberedOctahedron) : ℕ :=
  8 * o.lowest_number + 28

/-- Theorem stating the sum of numbers on the octahedron -/
theorem octahedron_sum_theorem (o : NumberedOctahedron) :
  o.face_count = 8 ∧ 
  o.is_consecutive = true ∧ 
  o.opposite_faces_diff = 2 →
  octahedron_sum o = 8 * o.lowest_number + 28 :=
by
  sorry

#check octahedron_sum_theorem

end NUMINAMATH_CALUDE_octahedron_sum_theorem_l4004_400489


namespace NUMINAMATH_CALUDE_sequence_values_bound_l4004_400482

theorem sequence_values_bound (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 2) = -f x) :
  let a : ℕ → ℝ := λ n => f n
  ∃ S : Finset ℝ, S.card ≤ 4 ∧ ∀ n : ℕ, a n ∈ S :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_values_bound_l4004_400482


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4004_400478

/-- Given a > 0 and a ≠ 1, if f(x) = ax is decreasing on ℝ, then g(x) = (2-a)x³ is increasing on ℝ, 
    but the converse is not always true. -/
theorem sufficient_but_not_necessary (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a * x < a * y) →
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  ¬(∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3 → a * x < a * y) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4004_400478
