import Mathlib

namespace NUMINAMATH_CALUDE_x_minus_y_equals_three_l1435_143569

theorem x_minus_y_equals_three 
  (h1 : 3 * x - 5 * y = 5) 
  (h2 : x / (x + y) = 5 / 7) : 
  x - y = 3 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_three_l1435_143569


namespace NUMINAMATH_CALUDE_cubic_function_property_l1435_143552

/-- Given a cubic function f(x) = ax³ + bx + 2 where f(-12) = 3, prove that f(12) = 1 -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x + 2)
  (h2 : f (-12) = 3) : 
  f 12 = 1 := by sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1435_143552


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l1435_143551

/-- The amount of flour Mary has already put in the recipe -/
def flour_already_added : ℕ := 3

/-- The amount of flour Mary still needs to add to the recipe -/
def flour_to_be_added : ℕ := 6

/-- The total amount of flour required for the recipe -/
def total_flour_required : ℕ := flour_already_added + flour_to_be_added

theorem recipe_flour_amount : total_flour_required = 9 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l1435_143551


namespace NUMINAMATH_CALUDE_max_entropy_theorem_l1435_143546

/-- Represents the gender of a child -/
inductive Gender
| Boy
| Girl

/-- Represents a configuration of children around a circular table -/
def Configuration (n : ℕ) := Fin (2*n) → Gender

/-- Calculates the entropy of a given configuration -/
def entropy (n : ℕ) (config : Configuration n) : ℕ := sorry

/-- Theorem stating the maximal possible entropy -/
theorem max_entropy_theorem (n : ℕ) (h : n > 3) :
  (∃ (config : Configuration n), ∀ (other_config : Configuration n),
    entropy n config ≥ entropy n other_config) ∧
  (∀ (config : Configuration n), entropy n config ≤ n - 2) :=
sorry

end NUMINAMATH_CALUDE_max_entropy_theorem_l1435_143546


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1435_143515

/-- Given a geometric sequence {a_n} where the first three terms are x, x-1, and 2x-2 respectively,
    prove that the general term is a_n = -2^(n-1) -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) (x : ℝ) (h1 : a 1 = x) (h2 : a 2 = x - 1) (h3 : a 3 = 2*x - 2) :
  ∀ n : ℕ, a n = -2^(n-1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1435_143515


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1435_143592

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1435_143592


namespace NUMINAMATH_CALUDE_gold_coins_per_hour_l1435_143563

def scuba_diving_hours : ℕ := 8
def treasure_chest_coins : ℕ := 100
def smaller_bags_count : ℕ := 2

def smaller_bag_coins : ℕ := treasure_chest_coins / 2

def total_coins : ℕ := treasure_chest_coins + smaller_bags_count * smaller_bag_coins

theorem gold_coins_per_hour :
  total_coins / scuba_diving_hours = 25 := by sorry

end NUMINAMATH_CALUDE_gold_coins_per_hour_l1435_143563


namespace NUMINAMATH_CALUDE_ginos_brown_bears_l1435_143568

theorem ginos_brown_bears :
  ∀ (total white black brown : ℕ),
    total = 66 →
    white = 24 →
    black = 27 →
    total = white + black + brown →
    brown = 15 := by
  sorry

end NUMINAMATH_CALUDE_ginos_brown_bears_l1435_143568


namespace NUMINAMATH_CALUDE_floor_product_theorem_l1435_143538

theorem floor_product_theorem :
  ∃ (x : ℝ), x > 0 ∧ (↑⌊x⌋ : ℝ) * x = 90 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_product_theorem_l1435_143538


namespace NUMINAMATH_CALUDE_train_length_l1435_143518

/-- Proves that a train crossing a 350-meter platform in 39 seconds and a signal pole in 18 seconds has a length of 300 meters -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 350)
  (h2 : platform_time = 39)
  (h3 : pole_time = 18) :
  let train_length := (platform_length * pole_time) / (platform_time - pole_time)
  train_length = 300 := by
sorry


end NUMINAMATH_CALUDE_train_length_l1435_143518


namespace NUMINAMATH_CALUDE_min_value_theorem_l1435_143534

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b^2 * c^3 = 256) : 
  a^2 + 8*a*b + 16*b^2 + 2*c^5 ≥ 768 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 
    a₀ * b₀^2 * c₀^3 = 256 ∧ 
    a₀^2 + 8*a₀*b₀ + 16*b₀^2 + 2*c₀^5 = 768 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1435_143534


namespace NUMINAMATH_CALUDE_only_setB_forms_triangle_l1435_143507

/-- Represents a set of three line segments --/
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a set of line segments can form a triangle --/
def canFormTriangle (s : LineSegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

/-- The given sets of line segments --/
def setA : LineSegmentSet := ⟨1, 2, 4⟩
def setB : LineSegmentSet := ⟨4, 6, 8⟩
def setC : LineSegmentSet := ⟨5, 6, 12⟩
def setD : LineSegmentSet := ⟨2, 3, 5⟩

/-- Theorem: Only set B can form a triangle --/
theorem only_setB_forms_triangle :
  ¬(canFormTriangle setA) ∧
  canFormTriangle setB ∧
  ¬(canFormTriangle setC) ∧
  ¬(canFormTriangle setD) :=
by sorry

end NUMINAMATH_CALUDE_only_setB_forms_triangle_l1435_143507


namespace NUMINAMATH_CALUDE_staircase_shape_perimeter_l1435_143595

/-- Represents the shape described in the problem -/
structure StaircaseShape where
  width : ℝ
  height : ℝ
  staircase_sides : ℕ
  area : ℝ

/-- Calculates the perimeter of the StaircaseShape -/
def perimeter (shape : StaircaseShape) : ℝ :=
  shape.width + shape.height + 4 + 5 + (shape.staircase_sides : ℝ)

/-- Theorem stating the perimeter of the specific shape described in the problem -/
theorem staircase_shape_perimeter : 
  ∀ (shape : StaircaseShape), 
    shape.width = 12 ∧ 
    shape.staircase_sides = 10 ∧ 
    shape.area = 72 → 
    perimeter shape = 42.25 := by
  sorry


end NUMINAMATH_CALUDE_staircase_shape_perimeter_l1435_143595


namespace NUMINAMATH_CALUDE_car_discount_proof_l1435_143564

/-- Given a car's original price and trading conditions, prove the discount percentage. -/
theorem car_discount_proof (P : ℝ) (P_b P_s : ℝ) (h1 : P > 0) (h2 : P_s = 1.60 * P_b) (h3 : P_s = 1.52 * P) : 
  ∃ D : ℝ, D = 0.05 ∧ P_b = P * (1 - D) := by
sorry

end NUMINAMATH_CALUDE_car_discount_proof_l1435_143564


namespace NUMINAMATH_CALUDE_radical_simplification_l1435_143548

theorem radical_simplification (q : ℝ) : 
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (8 * q^3) = 6 * q^3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l1435_143548


namespace NUMINAMATH_CALUDE_perfect_square_preserver_iff_square_multiple_l1435_143558

/-- A function is a perfect square preserver if it preserves the property of
    the sum of three distinct positive integers being a perfect square. -/
def IsPerfectSquarePreserver (f : ℕ → ℕ) : Prop :=
  ∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z →
    (∃ n : ℕ, x + y + z = n^2) ↔ (∃ m : ℕ, f x + f y + f z = m^2)

/-- A function is a square multiple if it's of the form f(x) = k²x for some k ∈ ℕ. -/
def IsSquareMultiple (f : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, ∀ x : ℕ, f x = k^2 * x

theorem perfect_square_preserver_iff_square_multiple (f : ℕ → ℕ) :
  IsPerfectSquarePreserver f ↔ IsSquareMultiple f := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_preserver_iff_square_multiple_l1435_143558


namespace NUMINAMATH_CALUDE_range_of_a_l1435_143512

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * |x - 1| + |x - a| ≥ 2) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1435_143512


namespace NUMINAMATH_CALUDE_max_book_price_l1435_143596

theorem max_book_price (total_money : ℕ) (num_books : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) :
  total_money = 200 →
  num_books = 20 →
  entrance_fee = 5 →
  tax_rate = 7 / 100 →
  ∃ (max_price : ℕ),
    (max_price ≤ (total_money - entrance_fee) / (num_books * (1 + tax_rate))) ∧
    (∀ (price : ℕ), price > max_price →
      price * num_books * (1 + tax_rate) > (total_money - entrance_fee)) ∧
    max_price = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_book_price_l1435_143596


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1435_143544

theorem compound_interest_rate (P : ℝ) (r : ℝ) : 
  P * (1 + r/100)^2 = 3650 ∧ 
  P * (1 + r/100)^3 = 4015 → 
  r = 10 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1435_143544


namespace NUMINAMATH_CALUDE_G_equals_4F_l1435_143574

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((4 * x - x^3) / (1 + 4 * x^2))

theorem G_equals_4F (x : ℝ) : G x = 4 * F x :=
  sorry

end NUMINAMATH_CALUDE_G_equals_4F_l1435_143574


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_neg_four_complement_intersection_condition_l1435_143593

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem intersection_and_union_when_a_is_neg_four :
  (A ∩ B (-4)) = {x | 3/2 ≤ x ∧ x < 2} ∧
  (A ∪ B (-4)) = {x | -2 < x ∧ x ≤ 3} := by sorry

theorem complement_intersection_condition (a : ℝ) :
  (Aᶜ ∩ B a) = B a ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_neg_four_complement_intersection_condition_l1435_143593


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_mixture_l1435_143520

/-- Represents a solution with a specific ratio of alcohol to water -/
structure Solution :=
  (alcohol : ℚ)
  (water : ℚ)

/-- Calculates the percentage of alcohol in a solution -/
def alcoholPercentage (s : Solution) : ℚ :=
  s.alcohol / (s.alcohol + s.water)

/-- Represents the mixing of two solutions in a specific ratio -/
structure Mixture :=
  (s1 : Solution)
  (s2 : Solution)
  (ratio1 : ℚ)
  (ratio2 : ℚ)

/-- Calculates the percentage of alcohol in a mixture -/
def mixtureAlcoholPercentage (m : Mixture) : ℚ :=
  (alcoholPercentage m.s1 * m.ratio1 + alcoholPercentage m.s2 * m.ratio2) / (m.ratio1 + m.ratio2)

theorem alcohol_percentage_in_mixture :
  let solutionA : Solution := ⟨21, 4⟩
  let solutionB : Solution := ⟨2, 3⟩
  let mixture : Mixture := ⟨solutionA, solutionB, 5, 6⟩
  mixtureAlcoholPercentage mixture = 3/5 := by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_mixture_l1435_143520


namespace NUMINAMATH_CALUDE_alyssa_picked_25_limes_l1435_143547

/-- The number of limes picked by Alyssa -/
def alyssas_limes : ℕ := 57 - 32

/-- The total number of limes picked -/
def total_limes : ℕ := 57

/-- The number of limes picked by Mike -/
def mikes_limes : ℕ := 32

theorem alyssa_picked_25_limes : alyssas_limes = 25 := by sorry

end NUMINAMATH_CALUDE_alyssa_picked_25_limes_l1435_143547


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1435_143532

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (x^2 + 5*x - 6) / (x^3 - x) = 6 / x + (-5*x + 5) / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1435_143532


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1435_143508

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x| ≤ 1} = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1435_143508


namespace NUMINAMATH_CALUDE_reciprocal_roots_identity_l1435_143562

theorem reciprocal_roots_identity (p q r s : ℝ) : 
  (∃ a : ℝ, a^2 + p*a + q = 0 ∧ (1/a)^2 + r*(1/a) + s = 0) →
  (p*s - r)*(q*r - p) = (q*s - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_roots_identity_l1435_143562


namespace NUMINAMATH_CALUDE_walts_age_inconsistency_l1435_143516

theorem walts_age_inconsistency :
  ¬ ∃ (w : ℕ), 
    (3 * w + 12 = 2 * (w + 12)) ∧ 
    (4 * w + 15 = 3 * (w + 15)) := by
  sorry

end NUMINAMATH_CALUDE_walts_age_inconsistency_l1435_143516


namespace NUMINAMATH_CALUDE_sqrt_x_minus_y_equals_plus_minus_two_l1435_143509

theorem sqrt_x_minus_y_equals_plus_minus_two
  (x y : ℝ) 
  (h : Real.sqrt (x - 3) + 2 * abs (y + 1) = 0) :
  Real.sqrt (x - y) = 2 ∨ Real.sqrt (x - y) = -2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_y_equals_plus_minus_two_l1435_143509


namespace NUMINAMATH_CALUDE_total_votes_proof_l1435_143524

theorem total_votes_proof (total_votes : ℕ) (votes_against : ℕ) : 
  (votes_against = total_votes * 40 / 100) →
  (total_votes - votes_against = votes_against + 70) →
  total_votes = 350 := by
sorry

end NUMINAMATH_CALUDE_total_votes_proof_l1435_143524


namespace NUMINAMATH_CALUDE_line_and_circle_problem_l1435_143553

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 = 0

-- Define the line m
def line_m (k : ℝ) (x y : ℝ) : Prop := x - k * y + 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define parallel lines
def parallel (k : ℝ) : Prop := ∃ (c : ℝ), ∀ x y, line_l k x y ↔ line_m k (x + c) (y + c * k)

-- Define tangent line to circle
def tangent (k : ℝ) : Prop := ∃! (x y : ℝ), line_l k x y ∧ circle_C x y

theorem line_and_circle_problem :
  (∀ k, parallel k → (k = 1 ∨ k = -1)) ∧
  (∀ k, tangent k → k = 1) :=
sorry

end NUMINAMATH_CALUDE_line_and_circle_problem_l1435_143553


namespace NUMINAMATH_CALUDE_annas_cupcakes_l1435_143543

theorem annas_cupcakes (C : ℕ) : 
  (C : ℚ) * (1 / 5) - 3 = 9 → C = 60 := by
  sorry

end NUMINAMATH_CALUDE_annas_cupcakes_l1435_143543


namespace NUMINAMATH_CALUDE_m_3_sufficient_m_3_not_necessary_l1435_143578

/-- Represents an ellipse with equation x²/4 + y²/m = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ (x y : ℝ), x^2/4 + y^2/m = 1

/-- The focal length of an ellipse -/
def focal_length (e : Ellipse m) : ℝ := 
  sorry

/-- Theorem stating that m = 3 is sufficient for focal length 2 -/
theorem m_3_sufficient (e : Ellipse 3) : focal_length e = 2 :=
  sorry

/-- Theorem stating that m = 3 is not necessary for focal length 2 -/
theorem m_3_not_necessary : ∃ (m : ℝ), m ≠ 3 ∧ ∃ (e : Ellipse m), focal_length e = 2 :=
  sorry

end NUMINAMATH_CALUDE_m_3_sufficient_m_3_not_necessary_l1435_143578


namespace NUMINAMATH_CALUDE_factorization_equality_l1435_143570

theorem factorization_equality (a b c d : ℝ) :
  a * (b - c)^3 + b * (c - d)^3 + c * (d - a)^3 + d * (a - b)^3 = 
  (a - b) * (b - c) * (c - d) * (d - a) * (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1435_143570


namespace NUMINAMATH_CALUDE_line_intersection_bound_l1435_143571

/-- Given points A(2,7) and B(9,6) in the Cartesian plane, and a line y = kx (k ≠ 0) that
    intersects the line segment AB, prove that k is bounded by 2/3 ≤ k ≤ 7/2. -/
theorem line_intersection_bound (k : ℝ) : k ≠ 0 → 
  (∃ x y : ℝ, x ∈ Set.Icc 2 9 ∧ y ∈ Set.Icc 6 7 ∧ y = k * x ∧ y - 7 = (6 - 7) / (9 - 2) * (x - 2)) →
  2/3 ≤ k ∧ k ≤ 7/2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_bound_l1435_143571


namespace NUMINAMATH_CALUDE_quadratic_roots_l1435_143529

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 2 ∧ 
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1435_143529


namespace NUMINAMATH_CALUDE_shoes_cost_eleven_l1435_143526

/-- The cost of shoes given initial amount, sweater cost, T-shirt cost, and remaining amount -/
def cost_of_shoes (initial_amount sweater_cost tshirt_cost remaining_amount : ℕ) : ℕ :=
  initial_amount - sweater_cost - tshirt_cost - remaining_amount

/-- Theorem stating that the cost of shoes is 11 given the problem conditions -/
theorem shoes_cost_eleven :
  cost_of_shoes 91 24 6 50 = 11 := by
  sorry

end NUMINAMATH_CALUDE_shoes_cost_eleven_l1435_143526


namespace NUMINAMATH_CALUDE_smallest_three_digit_square_ends_identical_l1435_143580

/-- A function that returns true if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

/-- A function that returns true if the square of a number ends with three identical non-zero digits -/
def squareEndsWithThreeIdenticalNonZeroDigits (n : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧ n^2 % 1000 = 111 * d

/-- Theorem stating that 462 is the smallest three-digit number whose square ends with three identical non-zero digits -/
theorem smallest_three_digit_square_ends_identical : 
  (isThreeDigit 462 ∧ 
   squareEndsWithThreeIdenticalNonZeroDigits 462 ∧ 
   ∀ n : ℕ, isThreeDigit n → squareEndsWithThreeIdenticalNonZeroDigits n → 462 ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_square_ends_identical_l1435_143580


namespace NUMINAMATH_CALUDE_sum_of_perpendiculars_l1435_143590

/-- An equilateral triangle with side length 6 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 6

/-- The centroid of a triangle -/
structure Centroid (T : EquilateralTriangle) where

/-- The perpendicular from the centroid to a side of the triangle -/
def perpendicular (T : EquilateralTriangle) (C : Centroid T) : ℝ := sorry

/-- The theorem stating the sum of perpendiculars from the centroid equals 3√3 -/
theorem sum_of_perpendiculars (T : EquilateralTriangle) (C : Centroid T) :
  3 * (perpendicular T C) = 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_perpendiculars_l1435_143590


namespace NUMINAMATH_CALUDE_positive_integer_problem_l1435_143530

theorem positive_integer_problem (n p : ℕ) (h_p_prime : Nat.Prime p) 
  (h_division : n / (12 * p) = 2) (h_n_ge_48 : n ≥ 48) : n = 48 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_problem_l1435_143530


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l1435_143501

theorem inequality_holds_iff_m_in_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1 / (x + 1) + 4 / y = 1) :
  (∀ m : ℝ, x + y / 4 > m^2 - 5*m - 3) ↔ ∀ m : ℝ, -1 < m ∧ m < 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l1435_143501


namespace NUMINAMATH_CALUDE_armands_guessing_game_l1435_143565

theorem armands_guessing_game (x : ℕ) : x = 33 ↔ 3 * x = 2 * 51 - 3 := by
  sorry

end NUMINAMATH_CALUDE_armands_guessing_game_l1435_143565


namespace NUMINAMATH_CALUDE_remainder_after_adding_2030_l1435_143550

theorem remainder_after_adding_2030 (m : ℤ) (h : m % 7 = 2) : (m + 2030) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2030_l1435_143550


namespace NUMINAMATH_CALUDE_min_rotation_regular_pentagon_l1435_143599

/-- The angle of rotation for a regular pentagon to overlap with itself -/
def pentagon_rotation_angle : ℝ := 72

/-- A regular pentagon has 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The minimum angle of rotation for a regular pentagon to overlap with itself is 72 degrees -/
theorem min_rotation_regular_pentagon :
  pentagon_rotation_angle = 360 / pentagon_sides :=
sorry

end NUMINAMATH_CALUDE_min_rotation_regular_pentagon_l1435_143599


namespace NUMINAMATH_CALUDE_combinations_equal_twenty_l1435_143556

/-- The number of available paint colors. -/
def num_colors : ℕ := 5

/-- The number of available painting methods. -/
def num_methods : ℕ := 4

/-- The total number of combinations of paint colors and painting methods. -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 20. -/
theorem combinations_equal_twenty : total_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_twenty_l1435_143556


namespace NUMINAMATH_CALUDE_log_sum_cubes_l1435_143554

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_sum_cubes (h : lg 2 + lg 5 = 1) :
  (lg 2)^3 + 3*(lg 2)*(lg 5) + (lg 5)^3 = 1 := by sorry

end NUMINAMATH_CALUDE_log_sum_cubes_l1435_143554


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1435_143514

-- Define the proposition
theorem necessary_not_sufficient_condition (a b c : ℝ) :
  (∃ x y z : ℝ, x > y ∧ x * z^2 ≤ y * z^2) ∧
  (∀ x y z : ℝ, x * z^2 > y * z^2 → x > y) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1435_143514


namespace NUMINAMATH_CALUDE_ellipse_dot_product_bounds_l1435_143506

/-- Given an ellipse with specific properties, prove that the dot product of vectors AP and FP is bounded. -/
theorem ellipse_dot_product_bounds (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0)
  (h_top_focus : 2 = Real.sqrt (a^2 - b^2))
  (h_eccentricity : (Real.sqrt (a^2 - b^2)) / a = 1/2) :
  ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 →
  0 ≤ (x + 2) * (x + 1) + y^2 ∧ (x + 2) * (x + 1) + y^2 ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_bounds_l1435_143506


namespace NUMINAMATH_CALUDE_cone_height_l1435_143527

/-- Proves that a cone with lateral area 15π cm² and base radius 3 cm has a height of 4 cm -/
theorem cone_height (lateral_area : ℝ) (base_radius : ℝ) (height : ℝ) : 
  lateral_area = 15 * Real.pi ∧ 
  base_radius = 3 ∧ 
  lateral_area = Real.pi * base_radius * (Real.sqrt (height^2 + base_radius^2)) →
  height = 4 := by
  sorry


end NUMINAMATH_CALUDE_cone_height_l1435_143527


namespace NUMINAMATH_CALUDE_xiao_yu_better_l1435_143573

/-- The number of optional questions -/
def total_questions : ℕ := 8

/-- The number of questions randomly selected -/
def selected_questions : ℕ := 4

/-- The probability of Xiao Ming correctly answering a single question -/
def xiao_ming_prob : ℚ := 3/4

/-- The number of questions Xiao Yu can correctly complete -/
def xiao_yu_correct : ℕ := 6

/-- The number of questions Xiao Yu cannot complete -/
def xiao_yu_incorrect : ℕ := 2

/-- The probability of Xiao Ming correctly completing at least 3 questions -/
def xiao_ming_at_least_three : ℚ :=
  Nat.choose selected_questions 3 * xiao_ming_prob^3 * (1 - xiao_ming_prob) +
  Nat.choose selected_questions 4 * xiao_ming_prob^4

/-- The probability of Xiao Yu correctly completing at least 3 questions -/
def xiao_yu_at_least_three : ℚ :=
  (Nat.choose xiao_yu_correct 3 * Nat.choose xiao_yu_incorrect 1 +
   Nat.choose xiao_yu_correct 4 * Nat.choose xiao_yu_incorrect 0) /
  Nat.choose total_questions selected_questions

/-- Theorem stating that Xiao Yu has a higher probability of correctly completing at least 3 questions -/
theorem xiao_yu_better : xiao_yu_at_least_three > xiao_ming_at_least_three := by
  sorry

end NUMINAMATH_CALUDE_xiao_yu_better_l1435_143573


namespace NUMINAMATH_CALUDE_inequality_proof_l1435_143535

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1435_143535


namespace NUMINAMATH_CALUDE_triangle_with_tan_A_2_tan_B_3_is_acute_l1435_143528

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define what it means for a triangle to be acute
def is_acute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

-- State the theorem
theorem triangle_with_tan_A_2_tan_B_3_is_acute :
  ∀ t : Triangle,
  Real.tan t.A = 2 →
  Real.tan t.B = 3 →
  is_acute t :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_tan_A_2_tan_B_3_is_acute_l1435_143528


namespace NUMINAMATH_CALUDE_cars_meeting_time_l1435_143575

/-- Two cars traveling towards each other on a highway meet after a certain time -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (meeting_time : ℝ) :
  highway_length = 500 →
  speed1 = 40 →
  speed2 = 60 →
  meeting_time * (speed1 + speed2) = highway_length →
  meeting_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l1435_143575


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1435_143566

theorem smallest_sum_of_reciprocals (a b : ℕ+) : 
  a ≠ b → 
  (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 → 
  (∀ c d : ℕ+, c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → (a : ℕ) + (b : ℕ) ≤ (c : ℕ) + (d : ℕ)) →
  (a : ℕ) + (b : ℕ) = 49 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1435_143566


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l1435_143577

theorem cubic_root_ratio (a b c d : ℝ) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 2 ∨ x = 3) → 
  c / d = -1 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l1435_143577


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1435_143598

theorem inequality_system_solution :
  ∀ x : ℝ, (x + 2 < 3 * x ∧ (5 - x) / 2 + 1 < 0) ↔ x > 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1435_143598


namespace NUMINAMATH_CALUDE_smallest_bound_inequality_l1435_143540

theorem smallest_bound_inequality (a b c : ℝ) : 
  ∃ (M : ℝ), M = (9 * Real.sqrt 2) / 32 ∧ 
  |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2 ∧
  ∀ (N : ℝ), (∀ (x y z : ℝ), 
    |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N * (x^2 + y^2 + z^2)^2) → 
  M ≤ N :=
sorry

end NUMINAMATH_CALUDE_smallest_bound_inequality_l1435_143540


namespace NUMINAMATH_CALUDE_children_education_expense_l1435_143589

def monthly_salary (saved_amount : ℚ) (savings_rate : ℚ) : ℚ :=
  saved_amount / savings_rate

def total_expenses (rent milk groceries petrol misc education : ℚ) : ℚ :=
  rent + milk + groceries + petrol + misc + education

theorem children_education_expense 
  (rent milk groceries petrol misc : ℚ)
  (savings_rate saved_amount : ℚ)
  (h1 : rent = 5000)
  (h2 : milk = 1500)
  (h3 : groceries = 4500)
  (h4 : petrol = 2000)
  (h5 : misc = 5200)
  (h6 : savings_rate = 1/10)
  (h7 : saved_amount = 2300)
  : ∃ (education : ℚ), 
    education = 2500 ∧ 
    total_expenses rent milk groceries petrol misc education = 
      monthly_salary saved_amount savings_rate := by
  sorry

end NUMINAMATH_CALUDE_children_education_expense_l1435_143589


namespace NUMINAMATH_CALUDE_at_most_one_acute_forming_point_l1435_143522

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A function to check if a triangle is acute-angled -/
def isAcuteTriangle (p q r : Point) : Prop :=
  sorry -- Definition of acute triangle

/-- The theorem stating that at most one point can form acute triangles with any other two points -/
theorem at_most_one_acute_forming_point (points : Finset Point) (h : points.card = 2006) :
  ∃ (p : Point), p ∈ points ∧
    (∀ (q r : Point), q ∈ points → r ∈ points → q ≠ r → q ≠ p → r ≠ p → isAcuteTriangle p q r) →
    ∀ (p' : Point), p' ∈ points → p' ≠ p →
      ∃ (q r : Point), q ∈ points ∧ r ∈ points ∧ q ≠ r ∧ q ≠ p' ∧ r ≠ p' ∧ ¬isAcuteTriangle p' q r :=
by
  sorry

end NUMINAMATH_CALUDE_at_most_one_acute_forming_point_l1435_143522


namespace NUMINAMATH_CALUDE_john_shirts_total_l1435_143500

theorem john_shirts_total (initial_shirts : ℕ) (bought_shirts : ℕ) : 
  initial_shirts = 12 → bought_shirts = 4 → initial_shirts + bought_shirts = 16 :=
by sorry

end NUMINAMATH_CALUDE_john_shirts_total_l1435_143500


namespace NUMINAMATH_CALUDE_perfect_square_existence_l1435_143505

theorem perfect_square_existence : ∃ n : ℕ, 
  (10^199 - 10^100 : ℕ) < n^2 ∧ n^2 < 10^199 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_existence_l1435_143505


namespace NUMINAMATH_CALUDE_pages_left_to_read_l1435_143523

theorem pages_left_to_read (total_pages : ℕ) (saturday_morning : ℕ) (saturday_night : ℕ) : 
  total_pages = 360 →
  saturday_morning = 40 →
  saturday_night = 10 →
  total_pages - (saturday_morning + saturday_night + 2 * (saturday_morning + saturday_night)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l1435_143523


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1435_143519

/-- Given a regular triangular prism with an inscribed sphere of radius r
    and a circumscribed sphere of radius R, prove that the ratio of their
    surface areas is 5:1 -/
theorem sphere_surface_area_ratio (r R : ℝ) :
  r > 0 →
  R = r * Real.sqrt 5 →
  (4 * Real.pi * R^2) / (4 * Real.pi * r^2) = 5 :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1435_143519


namespace NUMINAMATH_CALUDE_divisors_of_27n_cubed_l1435_143579

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_27n_cubed (n : ℕ) (h_odd : Odd n) (h_divisors : num_divisors n = 12) :
  num_divisors (27 * n^3) = 256 := by sorry

end NUMINAMATH_CALUDE_divisors_of_27n_cubed_l1435_143579


namespace NUMINAMATH_CALUDE_sam_nickels_count_l1435_143560

/-- Calculates the final number of nickels Sam has -/
def final_nickels (initial : ℕ) (added : ℕ) (taken : ℕ) : ℕ :=
  initial + added - taken

theorem sam_nickels_count : final_nickels 29 24 13 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sam_nickels_count_l1435_143560


namespace NUMINAMATH_CALUDE_expression_evaluation_l1435_143561

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(3*x)) / (y^(2*y) * x^(3*x)) = x^(2*y - 3*x) * y^(3*x - 2*y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1435_143561


namespace NUMINAMATH_CALUDE_cara_catches_47_l1435_143545

/-- The number of animals Martha's cat catches -/
def martha_animals : ℕ := 10

/-- The number of animals Cara's cat catches -/
def cara_animals : ℕ := 5 * martha_animals - 3

/-- Theorem stating that Cara's cat catches 47 animals -/
theorem cara_catches_47 : cara_animals = 47 := by
  sorry

end NUMINAMATH_CALUDE_cara_catches_47_l1435_143545


namespace NUMINAMATH_CALUDE_ravi_selection_probability_l1435_143582

theorem ravi_selection_probability 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 4/7)
  (h2 : p_both = 0.11428571428571428) :
  p_both / p_ram = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ravi_selection_probability_l1435_143582


namespace NUMINAMATH_CALUDE_root_sum_fraction_l1435_143597

theorem root_sum_fraction (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 →
  q^3 - 8*q^2 + 10*q - 3 = 0 →
  r^3 - 8*r^2 + 10*r - 3 = 0 →
  p/(q*r + 1) + q/(p*r + 1) + r/(p*q + 1) = 59/22 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l1435_143597


namespace NUMINAMATH_CALUDE_bus_stop_time_l1435_143584

/-- Proves that a bus with given speeds stops for 10 minutes per hour -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 54) 
  (h2 : speed_with_stops = 45) : ℝ :=
by
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l1435_143584


namespace NUMINAMATH_CALUDE_equation_solutions_l1435_143559

def equation (x y : ℝ) : Prop :=
  x^2 + x*y + y^2 + 2*x - 3*y - 3 = 0

def solution_set : Set (ℝ × ℝ) :=
  {(1, 2), (1, 0), (-5, 2), (-5, 6), (-3, 0)}

theorem equation_solutions :
  (∀ (x y : ℝ), (x, y) ∈ solution_set ↔ equation x y) ∧
  equation 1 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1435_143559


namespace NUMINAMATH_CALUDE_cheburashka_count_l1435_143511

/-- Represents the number of characters in a row -/
def n : ℕ := 16

/-- Represents the total number of Krakozyabras -/
def total_krakozyabras : ℕ := 29

/-- Represents the number of Cheburashkas -/
def num_cheburashkas : ℕ := 11

theorem cheburashka_count :
  (2 * (n - 1) = total_krakozyabras) ∧
  (num_cheburashkas * 2 + (num_cheburashkas - 1) * 2 + num_cheburashkas = n) :=
by sorry

#check cheburashka_count

end NUMINAMATH_CALUDE_cheburashka_count_l1435_143511


namespace NUMINAMATH_CALUDE_bruce_bought_five_crayons_l1435_143502

/-- Calculates the number of packs of crayons Bruce bought given the conditions of the problem. -/
def bruces_crayons (crayonPrice bookPrice calculatorPrice bagPrice totalMoney : ℕ) 
  (numBooks numCalculators numBags : ℕ) : ℕ :=
  let bookCost := numBooks * bookPrice
  let calculatorCost := numCalculators * calculatorPrice
  let bagCost := numBags * bagPrice
  let remainingMoney := totalMoney - bookCost - calculatorCost - bagCost
  remainingMoney / crayonPrice

/-- Theorem stating that Bruce bought 5 packs of crayons given the conditions of the problem. -/
theorem bruce_bought_five_crayons : 
  bruces_crayons 5 5 5 10 200 10 3 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bruce_bought_five_crayons_l1435_143502


namespace NUMINAMATH_CALUDE_min_difference_when_sum_maximized_l1435_143531

theorem min_difference_when_sum_maximized :
  ∀ x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℕ+,
    x₁ < x₂ → x₂ < x₃ → x₃ < x₄ → x₄ < x₅ → x₅ < x₆ → x₆ < x₇ → x₇ < x₈ → x₈ < x₉ →
    x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ + x₈ + x₉ = 220 →
    (∀ y₁ y₂ y₃ y₄ y₅ : ℕ+,
      y₁ < y₂ → y₂ < y₃ → y₃ < y₄ → y₄ < y₅ →
      y₁ + y₂ + y₃ + y₄ + y₅ ≤ x₁ + x₂ + x₃ + x₄ + x₅) →
    x₉ - x₁ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_difference_when_sum_maximized_l1435_143531


namespace NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l1435_143588

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x > f y

theorem even_periodic_function_monotonicity (f : ℝ → ℝ)
  (h_even : is_even f) (h_period : has_period f 2) :
  increasing_on f 0 1 ↔ decreasing_on f 3 4 := by sorry

end NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l1435_143588


namespace NUMINAMATH_CALUDE_gadget_sales_sum_l1435_143594

/-- The sum of an arithmetic sequence with first term 2, common difference 4, and 15 terms -/
def arithmetic_sum : ℕ := sorry

/-- The first term of the sequence -/
def a₁ : ℕ := 2

/-- The common difference of the sequence -/
def d : ℕ := 4

/-- The number of terms in the sequence -/
def n : ℕ := 15

/-- The last term of the sequence -/
def aₙ : ℕ := a₁ + (n - 1) * d

theorem gadget_sales_sum : arithmetic_sum = 450 := by sorry

end NUMINAMATH_CALUDE_gadget_sales_sum_l1435_143594


namespace NUMINAMATH_CALUDE_anderson_trousers_count_l1435_143591

theorem anderson_trousers_count :
  let total_clothing : ℕ := 934
  let shirts : ℕ := 589
  let trousers : ℕ := total_clothing - shirts
  trousers = 345 := by sorry

end NUMINAMATH_CALUDE_anderson_trousers_count_l1435_143591


namespace NUMINAMATH_CALUDE_arevalo_dinner_change_l1435_143536

/-- The change calculation for the Arevalo family dinner --/
theorem arevalo_dinner_change (salmon_cost black_burger_cost chicken_katsu_cost : ℝ)
  (service_charge_rate tip_rate : ℝ) (amount_paid : ℝ)
  (h1 : salmon_cost = 40)
  (h2 : black_burger_cost = 15)
  (h3 : chicken_katsu_cost = 25)
  (h4 : service_charge_rate = 0.1)
  (h5 : tip_rate = 0.05)
  (h6 : amount_paid = 100) :
  amount_paid - (salmon_cost + black_burger_cost + chicken_katsu_cost +
    (salmon_cost + black_burger_cost + chicken_katsu_cost) * service_charge_rate +
    (salmon_cost + black_burger_cost + chicken_katsu_cost) * tip_rate) = 8 := by
  sorry

end NUMINAMATH_CALUDE_arevalo_dinner_change_l1435_143536


namespace NUMINAMATH_CALUDE_triangle_areas_l1435_143525

/-- Given points Q, A, C, and D on the x-y coordinate plane, prove the areas of triangles QCA and ACD. -/
theorem triangle_areas (p : ℝ) : 
  let Q : ℝ × ℝ := (0, 15)
  let A : ℝ × ℝ := (3, 15)
  let C : ℝ × ℝ := (0, p)
  let D : ℝ × ℝ := (3, 0)
  
  let area_QCA := (45 - 3 * p) / 2
  let area_ACD := 22.5

  (∃ (area_function : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ),
    area_function Q C A = area_QCA ∧
    area_function A C D = area_ACD) := by
  sorry

end NUMINAMATH_CALUDE_triangle_areas_l1435_143525


namespace NUMINAMATH_CALUDE_emma_sister_age_relationship_l1435_143572

/-- Emma's current age -/
def emma_age : ℕ := 7

/-- Age difference between Emma and her sister -/
def age_difference : ℕ := 9

/-- Emma's age when her sister is 56 -/
def emma_future_age : ℕ := 47

/-- Emma's sister's age when Emma is 47 -/
def sister_future_age : ℕ := 56

/-- Theorem stating the relationship between Emma's age and her sister's age -/
theorem emma_sister_age_relationship (x : ℕ) :
  x ≥ age_difference →
  emma_future_age = sister_future_age - age_difference →
  x - age_difference = emma_age + (x - sister_future_age) :=
by
  sorry

end NUMINAMATH_CALUDE_emma_sister_age_relationship_l1435_143572


namespace NUMINAMATH_CALUDE_square_root_and_cube_l1435_143549

theorem square_root_and_cube (x : ℝ) (x_nonzero : x ≠ 0) :
  (Real.sqrt 144 + 3^3) / x = 39 / x :=
sorry

end NUMINAMATH_CALUDE_square_root_and_cube_l1435_143549


namespace NUMINAMATH_CALUDE_square_equation_solution_cubic_equation_solution_l1435_143539

-- Problem 1
theorem square_equation_solution (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

-- Problem 2
theorem cubic_equation_solution (x : ℝ) : 64 * x^3 + 27 = 0 ↔ x = -3/4 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_cubic_equation_solution_l1435_143539


namespace NUMINAMATH_CALUDE_rectangle_other_vertices_y_sum_l1435_143542

/-- A rectangle in a 2D plane --/
structure Rectangle where
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  vertex3 : ℝ × ℝ
  vertex4 : ℝ × ℝ

/-- The property that two points are opposite vertices of a rectangle --/
def areOppositeVertices (p1 p2 : ℝ × ℝ) (r : Rectangle) : Prop :=
  (r.vertex1 = p1 ∧ r.vertex3 = p2) ∨ (r.vertex1 = p2 ∧ r.vertex3 = p1) ∨
  (r.vertex2 = p1 ∧ r.vertex4 = p2) ∨ (r.vertex2 = p2 ∧ r.vertex4 = p1)

/-- The sum of y-coordinates of two points --/
def sumYCoordinates (p1 p2 : ℝ × ℝ) : ℝ :=
  p1.2 + p2.2

theorem rectangle_other_vertices_y_sum 
  (r : Rectangle) 
  (h : areOppositeVertices (3, 17) (9, -4) r) : 
  ∃ (v1 v2 : ℝ × ℝ), 
    ((v1 = r.vertex2 ∧ v2 = r.vertex4) ∨ (v1 = r.vertex1 ∧ v2 = r.vertex3)) ∧
    sumYCoordinates v1 v2 = 13 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_other_vertices_y_sum_l1435_143542


namespace NUMINAMATH_CALUDE_expression_simplification_l1435_143583

theorem expression_simplification (p : ℝ) :
  (2 * (3 * p + 4) - 5 * p * 2)^2 + (6 - 2 / 2) * (9 * p - 12) = 16 * p^2 - 19 * p + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1435_143583


namespace NUMINAMATH_CALUDE_parallel_vectors_l1435_143503

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
axiom parallel_iff_cross_zero {u v : ℝ × ℝ} : 
  (∃ (k : ℝ), u = k • v ∨ v = k • u) ↔ u.1 * v.2 - u.2 * v.1 = 0

/-- Given vectors a and b, prove that a is parallel to b if and only if y = -6 -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (-1, 3)) (h2 : b = (2, y)) :
  (∃ (k : ℝ), a = k • b ∨ b = k • a) ↔ y = -6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l1435_143503


namespace NUMINAMATH_CALUDE_probability_one_each_color_l1435_143513

def total_marbles : ℕ := 7
def red_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def green_marbles : ℕ := 2
def yellow_marbles : ℕ := 1
def marbles_drawn : ℕ := 3

theorem probability_one_each_color (total : ℕ) (red : ℕ) (blue : ℕ) (green : ℕ) (yellow : ℕ) (drawn : ℕ)
  (h1 : total = red + blue + green + yellow)
  (h2 : drawn = 3)
  (h3 : red = 2)
  (h4 : blue = 2)
  (h5 : green = 2)
  (h6 : yellow = 1) :
  (red * blue * green : ℚ) / Nat.choose total drawn = 8 / 35 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_each_color_l1435_143513


namespace NUMINAMATH_CALUDE_decagon_sign_change_impossible_l1435_143541

/-- Represents a point in the decagon where a number is placed -/
structure Point where
  value : Int
  is_vertex : Bool
  is_intersection : Bool

/-- Represents the decagon configuration -/
structure Decagon where
  points : List Point
  
/-- Represents an operation that can be performed on the decagon -/
inductive Operation
  | FlipSide : Nat → Operation  -- Flip signs along the nth side
  | FlipDiagonal : Nat → Operation  -- Flip signs along the nth diagonal

/-- Applies an operation to the decagon -/
def apply_operation (d : Decagon) (op : Operation) : Decagon :=
  sorry

/-- Checks if all points in the decagon have negative values -/
def all_negative (d : Decagon) : Bool :=
  sorry

/-- Initial setup of the decagon with all +1 values -/
def initial_decagon : Decagon :=
  sorry

theorem decagon_sign_change_impossible :
  ∀ (ops : List Operation),
    ¬(all_negative (ops.foldl apply_operation initial_decagon)) :=
  sorry

end NUMINAMATH_CALUDE_decagon_sign_change_impossible_l1435_143541


namespace NUMINAMATH_CALUDE_river_width_l1435_143521

/-- A configuration of points for measuring river width -/
structure RiverMeasurement where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  AC_eq_40 : dist A C = 40
  CD_eq_12 : dist C D = 12
  AE_eq_24 : dist A E = 24
  EC_eq_16 : dist E C = 16
  AB_perp_CD : ((B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) : ℝ) = 0
  E_on_AB : ∃ t : ℝ, E = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

/-- The width of the river is 18 meters -/
theorem river_width (m : RiverMeasurement) : dist m.A m.B = 18 := by
  sorry

end NUMINAMATH_CALUDE_river_width_l1435_143521


namespace NUMINAMATH_CALUDE_costs_equal_at_60_guests_l1435_143576

/-- The number of guests for which the costs of Caesar's and Venus Hall are equal -/
def equal_cost_guests : ℕ := 60

/-- Caesar's room rental cost -/
def caesars_rental : ℕ := 800

/-- Caesar's per-meal cost -/
def caesars_meal : ℕ := 30

/-- Venus Hall's room rental cost -/
def venus_rental : ℕ := 500

/-- Venus Hall's per-meal cost -/
def venus_meal : ℕ := 35

/-- Theorem stating that the costs are equal for the given number of guests -/
theorem costs_equal_at_60_guests : 
  caesars_rental + caesars_meal * equal_cost_guests = 
  venus_rental + venus_meal * equal_cost_guests :=
by sorry

end NUMINAMATH_CALUDE_costs_equal_at_60_guests_l1435_143576


namespace NUMINAMATH_CALUDE_cone_vertex_angle_l1435_143587

noncomputable def vertex_angle_third_cone : ℝ := 2 * Real.arcsin (1/4)

theorem cone_vertex_angle 
  (first_two_cones_identical : Bool)
  (fourth_cone_internal : Bool)
  (first_two_cones_half_fourth : Bool) :
  ∃ (α : ℝ), 
    α = π/6 + Real.arcsin (1/4) ∧ 
    α > 0 ∧ 
    α < π/2 ∧
    2 * α = vertex_angle_third_cone ∧
    first_two_cones_identical = true ∧
    fourth_cone_internal = true ∧
    first_two_cones_half_fourth = true :=
by sorry

end NUMINAMATH_CALUDE_cone_vertex_angle_l1435_143587


namespace NUMINAMATH_CALUDE_parallel_lines_unique_m_l1435_143504

/-- Given two lines l₁ and l₂, prove that m = -4 is the only value that makes them parallel -/
theorem parallel_lines_unique_m : ∃! m : ℝ, 
  (∀ x y : ℝ, (m - 2) * x - 3 * y - 1 = 0 ↔ ((m - 2) / 3) * x - 1 / 3 = y) ∧ 
  (∀ x y : ℝ, m * x + (m + 2) * y + 1 = 0 ↔ (-m / (m + 2)) * x - 1 / (m + 2) = y) ∧
  ((m - 2) / 3 = -m / (m + 2)) ∧
  (m - 2 ≠ -m) ∧
  m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_unique_m_l1435_143504


namespace NUMINAMATH_CALUDE_choir_meeting_interval_l1435_143510

/-- The number of days between drama club meetings -/
def drama_interval : ℕ := 3

/-- The number of days until the next joint meeting -/
def next_joint_meeting : ℕ := 15

/-- The number of days between choir meetings -/
def choir_interval : ℕ := 5

theorem choir_meeting_interval :
  (next_joint_meeting % drama_interval = 0) ∧
  (next_joint_meeting % choir_interval = 0) ∧
  (∀ x : ℕ, x > 1 ∧ x < choir_interval →
    ¬(next_joint_meeting % x = 0 ∧ next_joint_meeting % drama_interval = 0)) :=
by sorry

end NUMINAMATH_CALUDE_choir_meeting_interval_l1435_143510


namespace NUMINAMATH_CALUDE_oliver_games_l1435_143537

def number_of_games (initial_money : ℕ) (money_spent : ℕ) (game_cost : ℕ) : ℕ :=
  (initial_money - money_spent) / game_cost

theorem oliver_games : number_of_games 35 7 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_oliver_games_l1435_143537


namespace NUMINAMATH_CALUDE_mothers_age_l1435_143557

-- Define variables for current ages
variable (A : ℕ) -- Allen's current age
variable (M : ℕ) -- Mother's current age
variable (S : ℕ) -- Sister's current age

-- Define the conditions
axiom allen_younger : A = M - 30
axiom sister_older : S = A + 5
axiom future_sum : (A + 7) + (M + 7) + (S + 7) = 110
axiom mother_sister_diff : M - S = 25

-- Theorem to prove
theorem mothers_age : M = 48 := by
  sorry

end NUMINAMATH_CALUDE_mothers_age_l1435_143557


namespace NUMINAMATH_CALUDE_min_value_cubic_function_l1435_143586

/-- A cubic function f(x) = (a/3)x^3 + bx^2 + cx + d is monotonically increasing on ℝ 
    if and only if its derivative is non-negative for all x ∈ ℝ -/
def monotonically_increasing (a b c : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 2 * b * x + c ≥ 0

/-- The theorem stating the minimum value of (a + 2b + 3c)/(b - a) 
    for a monotonically increasing cubic function with a < b -/
theorem min_value_cubic_function (a b c : ℝ) 
    (h1 : a < b) 
    (h2 : monotonically_increasing a b c) : 
  (a + 2*b + 3*c) / (b - a) ≥ 8 + 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_cubic_function_l1435_143586


namespace NUMINAMATH_CALUDE_max_xy_value_l1435_143585

theorem max_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 6 * x + 8 * y = 72) (h4 : x = 2 * y) :
  ∃ (max_xy : ℝ), max_xy = 25.92 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 6 * x' + 8 * y' = 72 → x' = 2 * y' → x' * y' ≤ max_xy :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l1435_143585


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1435_143555

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (-2 + Complex.I) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1435_143555


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1435_143533

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x + 1| > a) → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1435_143533


namespace NUMINAMATH_CALUDE_m_range_for_three_roots_l1435_143517

/-- The function f(x) defined in the problem -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 3

/-- The function g(x) defined in the problem -/
def g (x m : ℝ) : ℝ := f x - m

/-- Theorem stating the range of m for which g(x) has exactly 3 real roots -/
theorem m_range_for_three_roots :
  ∀ m : ℝ, (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, g x m = 0) → m ∈ Set.Ioo (-24) 8 :=
sorry

end NUMINAMATH_CALUDE_m_range_for_three_roots_l1435_143517


namespace NUMINAMATH_CALUDE_expression_lower_bound_l1435_143581

theorem expression_lower_bound (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b + c) + 1 / (b + c) + 1 / (c + a)) ≥ 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_lower_bound_l1435_143581


namespace NUMINAMATH_CALUDE_magical_stack_with_89_fixed_has_266_cards_l1435_143567

/-- Represents a stack of cards -/
structure CardStack :=
  (n : ℕ)
  (is_magical : Bool)
  (card_89_position : ℕ)

/-- Checks if a card stack is magical and card 89 retains its position -/
def is_magical_with_89_fixed (stack : CardStack) : Prop :=
  stack.is_magical ∧ stack.card_89_position = 89

/-- Theorem: A magical stack where card 89 retains its position has 266 cards -/
theorem magical_stack_with_89_fixed_has_266_cards (stack : CardStack) :
  is_magical_with_89_fixed stack → 2 * stack.n = 266 := by
  sorry

#check magical_stack_with_89_fixed_has_266_cards

end NUMINAMATH_CALUDE_magical_stack_with_89_fixed_has_266_cards_l1435_143567
