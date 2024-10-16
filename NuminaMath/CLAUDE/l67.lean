import Mathlib

namespace NUMINAMATH_CALUDE_prob_odd_product_six_rolls_main_theorem_l67_6708

/-- A standard die has six faces numbered 1 through 6 -/
def standardDie : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- The probability of rolling an odd number on a standard die -/
def probOddRoll : Rat := 1/2

/-- The number of times the die is rolled -/
def numRolls : Nat := 6

/-- Theorem: The probability of rolling a standard die six times and obtaining an odd product is 1/64 -/
theorem prob_odd_product_six_rolls :
  (probOddRoll ^ numRolls : Rat) = 1/64 := by
  sorry

/-- Main theorem: The probability of rolling a standard die six times and obtaining an odd product is 1/64 -/
theorem main_theorem :
  ∃ (p : Rat), p = (probOddRoll ^ numRolls) ∧ p = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_product_six_rolls_main_theorem_l67_6708


namespace NUMINAMATH_CALUDE_binomial_12_9_l67_6728

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l67_6728


namespace NUMINAMATH_CALUDE_three_digit_number_operation_l67_6743

theorem three_digit_number_operation : ∀ a b c : ℕ,
  a ≥ 1 → a ≤ 9 →
  b ≥ 0 → b ≤ 9 →
  c ≥ 0 → c ≤ 9 →
  a = c + 3 →
  (100 * a + 10 * b + c) - ((100 * c + 10 * b + a) + 50) ≡ 7 [MOD 10] := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_operation_l67_6743


namespace NUMINAMATH_CALUDE_num_machines_is_five_l67_6784

/-- The number of machines in the first scenario -/
def num_machines : ℕ := 5

/-- The production rate of the machines in the first scenario -/
def production_rate_1 : ℚ := 20 / (10 * num_machines)

/-- The production rate of the machines in the second scenario -/
def production_rate_2 : ℚ := 200 / (25 * 20)

/-- Theorem stating that the number of machines in the first scenario is 5 -/
theorem num_machines_is_five :
  num_machines = 5 ∧ production_rate_1 = production_rate_2 :=
sorry

end NUMINAMATH_CALUDE_num_machines_is_five_l67_6784


namespace NUMINAMATH_CALUDE_gift_fund_equations_correct_l67_6745

/-- Represents the crowdfunding scenario for teachers' New Year gift package. -/
structure GiftFundScenario where
  x : ℕ  -- number of teachers
  y : ℕ  -- price of the gift package

/-- The correct system of equations for the gift fund scenario. -/
def correct_equations (s : GiftFundScenario) : Prop :=
  18 * s.x = s.y + 3 ∧ 17 * s.x = s.y - 4

/-- Theorem stating that the given system of equations correctly describes the gift fund scenario. -/
theorem gift_fund_equations_correct (s : GiftFundScenario) : correct_equations s :=
sorry

end NUMINAMATH_CALUDE_gift_fund_equations_correct_l67_6745


namespace NUMINAMATH_CALUDE_intersection_A_B_l67_6727

-- Define set A
def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}

-- Define set B
def B : Set ℝ := {y | 0 ≤ y ∧ y < 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l67_6727


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_reciprocal_l67_6732

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 10*x - 6

-- State the theorem
theorem roots_sum_of_squares_reciprocal :
  ∃ (a b c : ℝ), (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (1 / a^2 + 1 / b^2 + 1 / c^2 = 46 / 9) := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_reciprocal_l67_6732


namespace NUMINAMATH_CALUDE_prob_non_defective_pencils_l67_6725

/-- The probability of selecting 5 non-defective pencils from a box of 12 pencils
    where 4 are defective is 7/99. -/
theorem prob_non_defective_pencils :
  let total_pencils : ℕ := 12
  let defective_pencils : ℕ := 4
  let selected_pencils : ℕ := 5
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  (Nat.choose non_defective_pencils selected_pencils : ℚ) /
  (Nat.choose total_pencils selected_pencils : ℚ) = 7 / 99 := by
sorry

end NUMINAMATH_CALUDE_prob_non_defective_pencils_l67_6725


namespace NUMINAMATH_CALUDE_star_properties_l67_6738

/-- Custom binary operation ※ -/
def star (x y : ℚ) : ℚ := x * y + 1

/-- Theorem stating the properties of the ※ operation -/
theorem star_properties :
  (star 2 4 = 9) ∧
  (star (star 1 4) (-2) = -9) ∧
  (∀ a b c : ℚ, star a (b + c) + 1 = star a b + star a c) :=
by sorry

end NUMINAMATH_CALUDE_star_properties_l67_6738


namespace NUMINAMATH_CALUDE_divisibility_condition_l67_6720

/-- s_n is the sum of all integers in [1,n] that are mutually prime to n -/
def s_n (n : ℕ) : ℕ := sorry

/-- t_n is the sum of the remaining integers in [1,n] -/
def t_n (n : ℕ) : ℕ := sorry

/-- Theorem: For all integers n ≥ 2, n divides (s_n - t_n) if and only if n is odd -/
theorem divisibility_condition (n : ℕ) (h : n ≥ 2) :
  n ∣ (s_n n - t_n n) ↔ Odd n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l67_6720


namespace NUMINAMATH_CALUDE_price_decrease_sales_increase_l67_6724

/-- Given a price decrease and revenue increase, calculate the increase in number of items sold -/
theorem price_decrease_sales_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_decrease_percentage : ℝ)
  (revenue_increase_percentage : ℝ)
  (h_price_decrease : price_decrease_percentage = 20)
  (h_revenue_increase : revenue_increase_percentage = 28.000000000000025)
  (h_positive_price : original_price > 0)
  (h_positive_quantity : original_quantity > 0) :
  let new_price := original_price * (1 - price_decrease_percentage / 100)
  let new_quantity := original_quantity * (1 + revenue_increase_percentage / 100) / (1 - price_decrease_percentage / 100)
  let quantity_increase_percentage := (new_quantity / original_quantity - 1) * 100
  ∃ ε > 0, |quantity_increase_percentage - 60| < ε :=
sorry

end NUMINAMATH_CALUDE_price_decrease_sales_increase_l67_6724


namespace NUMINAMATH_CALUDE_domain_of_g_l67_6764

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-2) 4

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (-x)

-- Define the domain of g
def domain_g : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem domain_of_g :
  ∀ x, x ∈ domain_g ↔ (x ∈ domain_f ∧ (-x) ∈ domain_f) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l67_6764


namespace NUMINAMATH_CALUDE_union_cardinality_of_subset_count_l67_6726

/-- Given two finite sets A and B, if the number of sets which are subsets of A or subsets of B is 144, then the cardinality of their union is 8. -/
theorem union_cardinality_of_subset_count (A B : Finset ℕ) : 
  (Finset.powerset A).card + (Finset.powerset B).card - (Finset.powerset (A ∩ B)).card = 144 →
  (A ∪ B).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_union_cardinality_of_subset_count_l67_6726


namespace NUMINAMATH_CALUDE_rootsOfTwo_is_well_defined_set_rootsOfTwo_has_two_elements_l67_6731

-- Define the set of real number roots of x^2 = 2
def rootsOfTwo : Set ℝ := {x : ℝ | x^2 = 2}

-- Theorem stating that rootsOfTwo is a well-defined set
theorem rootsOfTwo_is_well_defined_set : 
  ∃ (S : Set ℝ), S = rootsOfTwo ∧ (∀ x : ℝ, x ∈ S ↔ x^2 = 2) :=
by
  sorry

-- Theorem stating that rootsOfTwo contains exactly two elements
theorem rootsOfTwo_has_two_elements :
  ∃ (a b : ℝ), a ≠ b ∧ rootsOfTwo = {a, b} :=
by
  sorry

end NUMINAMATH_CALUDE_rootsOfTwo_is_well_defined_set_rootsOfTwo_has_two_elements_l67_6731


namespace NUMINAMATH_CALUDE_prob_hit_twice_in_three_shots_l67_6735

/-- The probability of hitting a target exactly twice in three independent shots -/
theorem prob_hit_twice_in_three_shots 
  (p1 : Real) (p2 : Real) (p3 : Real)
  (h1 : p1 = 0.4) (h2 : p2 = 0.5) (h3 : p3 = 0.7) :
  p1 * p2 * (1 - p3) + (1 - p1) * p2 * p3 + p1 * (1 - p2) * p3 = 0.41 := by
sorry

end NUMINAMATH_CALUDE_prob_hit_twice_in_three_shots_l67_6735


namespace NUMINAMATH_CALUDE_determinant_equation_solution_l67_6795

/-- Definition of a second-order determinant -/
def secondOrderDet (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating that if |x+3 x-3; x-3 x+3| = 12, then x = 1 -/
theorem determinant_equation_solution :
  ∀ x : ℝ, secondOrderDet (x + 3) (x - 3) (x - 3) (x + 3) = 12 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equation_solution_l67_6795


namespace NUMINAMATH_CALUDE_expression_evaluation_l67_6766

theorem expression_evaluation (m : ℝ) (h : m = 2) : 
  (m^2 - 9) / (m^2 - 6*m + 9) / (1 - 2/(m-3)) = -5/3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l67_6766


namespace NUMINAMATH_CALUDE_cistern_filling_time_l67_6751

theorem cistern_filling_time (t : ℝ) : t > 0 → 
  (4 * (1 / t + 1 / 18) + 8 / 18 = 1) → t = 12 := by sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l67_6751


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l67_6792

/-- The perimeter of an isosceles triangle given specific conditions -/
theorem isosceles_triangle_perimeter : ∀ (equilateral_perimeter isosceles_base : ℝ),
  equilateral_perimeter = 60 →
  isosceles_base = 30 →
  ∃ (isosceles_perimeter : ℝ),
    isosceles_perimeter = equilateral_perimeter / 3 + equilateral_perimeter / 3 + isosceles_base ∧
    isosceles_perimeter = 70 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l67_6792


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruent_to_one_mod_seventeen_l67_6722

theorem smallest_four_digit_congruent_to_one_mod_seventeen :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 17 = 1 → n ≥ 1003 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruent_to_one_mod_seventeen_l67_6722


namespace NUMINAMATH_CALUDE_ice_cream_bars_per_friend_l67_6759

/-- Proves that given the conditions of the ice cream problem, each friend wants to eat 2 bars. -/
theorem ice_cream_bars_per_friend 
  (box_cost : ℚ) 
  (bars_per_box : ℕ) 
  (num_friends : ℕ) 
  (cost_per_person : ℚ) 
  (h1 : box_cost = 15/2)
  (h2 : bars_per_box = 3)
  (h3 : num_friends = 6)
  (h4 : cost_per_person = 5) : 
  (num_friends * cost_per_person / box_cost * bars_per_box) / num_friends = 2 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_bars_per_friend_l67_6759


namespace NUMINAMATH_CALUDE_integral_2x_minus_1_l67_6723

theorem integral_2x_minus_1 : ∫ x in (0 : ℝ)..3, (2*x - 1) = 6 := by sorry

end NUMINAMATH_CALUDE_integral_2x_minus_1_l67_6723


namespace NUMINAMATH_CALUDE_person_speed_l67_6783

theorem person_speed (street_length : Real) (crossing_time : Real) (speed : Real) :
  street_length = 300 →
  crossing_time = 4 →
  speed = (street_length / 1000) / (crossing_time / 60) →
  speed = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_person_speed_l67_6783


namespace NUMINAMATH_CALUDE_total_spent_proof_l67_6773

def jayda_stall1 : ℝ := 400
def jayda_stall2 : ℝ := 120
def jayda_stall3 : ℝ := 250
def aitana_multiplier : ℝ := 1.4 -- 1 + 2/5
def jayda_discount1 : ℝ := 0.05
def aitana_discount2 : ℝ := 0.10
def sales_tax : ℝ := 0.10
def exchange_rate : ℝ := 1.25

def total_spent_cad : ℝ :=
  ((jayda_stall1 * (1 - jayda_discount1) + jayda_stall2 + jayda_stall3) * (1 + sales_tax) +
   (jayda_stall1 * aitana_multiplier + 
    jayda_stall2 * aitana_multiplier * (1 - aitana_discount2) + 
    jayda_stall3 * aitana_multiplier) * (1 + sales_tax)) * exchange_rate

theorem total_spent_proof : total_spent_cad = 2490.40 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_proof_l67_6773


namespace NUMINAMATH_CALUDE_negative_three_hash_six_l67_6770

/-- The '#' operation for rational numbers -/
def hash (a b : ℚ) : ℚ := a^2 + a*b - 5

/-- Theorem: (-3)#6 = -14 -/
theorem negative_three_hash_six : hash (-3) 6 = -14 := by sorry

end NUMINAMATH_CALUDE_negative_three_hash_six_l67_6770


namespace NUMINAMATH_CALUDE_sum_equals_369_l67_6742

theorem sum_equals_369 : 333 + 33 + 3 = 369 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_369_l67_6742


namespace NUMINAMATH_CALUDE_equal_intercept_line_properties_l67_6733

/-- A line passing through (1, 2) with equal intercepts on both axes -/
def equal_intercept_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 3}

theorem equal_intercept_line_properties :
  (1, 2) ∈ equal_intercept_line ∧
  ∃ a : ℝ, a ≠ 0 ∧ (a, 0) ∈ equal_intercept_line ∧ (0, a) ∈ equal_intercept_line :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_properties_l67_6733


namespace NUMINAMATH_CALUDE_bullet_problem_l67_6744

theorem bullet_problem :
  ∀ (initial_bullets : ℕ),
    (5 * (initial_bullets - 4) = initial_bullets) →
    initial_bullets = 5 := by
  sorry

end NUMINAMATH_CALUDE_bullet_problem_l67_6744


namespace NUMINAMATH_CALUDE_three_number_sum_l67_6786

theorem three_number_sum (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 10) 
  (h4 : (a + b + c) / 3 = a + 20) (h5 : (a + b + c) / 3 = c - 30) : 
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_three_number_sum_l67_6786


namespace NUMINAMATH_CALUDE_min_box_value_l67_6703

theorem min_box_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 32 * x^2 + box * x + 32) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  (∃ a' b' box', (∀ x, (a' * x + b') * (b' * x + a') = 32 * x^2 + box' * x + 32) ∧
                 a' ≠ b' ∧ a' ≠ box' ∧ b' ≠ box' ∧
                 box' ≥ 80) →
  box ≥ 80 :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l67_6703


namespace NUMINAMATH_CALUDE_five_objects_three_containers_l67_6712

/-- The number of ways to put n distinguishable objects into k distinguishable containers -/
def num_ways (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to put 5 distinguishable objects into 3 distinguishable containers is 3^5 -/
theorem five_objects_three_containers : num_ways 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_five_objects_three_containers_l67_6712


namespace NUMINAMATH_CALUDE_inverse_expression_equals_one_sixth_l67_6760

theorem inverse_expression_equals_one_sixth :
  (2 + 4 * (4 - 3)⁻¹)⁻¹ = (1 : ℚ) / 6 := by sorry

end NUMINAMATH_CALUDE_inverse_expression_equals_one_sixth_l67_6760


namespace NUMINAMATH_CALUDE_circular_cross_section_shapes_l67_6761

-- Define the shapes
inductive Shape
  | Cone
  | Cylinder
  | Sphere
  | PentagonalPrism

-- Define a function to check if a shape can have a circular cross-section
def canHaveCircularCrossSection (s : Shape) : Prop :=
  match s with
  | Shape.Cone => true
  | Shape.Cylinder => true
  | Shape.Sphere => true
  | Shape.PentagonalPrism => false

-- Theorem statement
theorem circular_cross_section_shapes :
  ∀ s : Shape, canHaveCircularCrossSection s ↔ (s = Shape.Cone ∨ s = Shape.Cylinder ∨ s = Shape.Sphere) :=
by sorry

end NUMINAMATH_CALUDE_circular_cross_section_shapes_l67_6761


namespace NUMINAMATH_CALUDE_jacket_price_correct_l67_6755

/-- The original price of the jacket -/
def original_price : ℝ := 250

/-- The regular discount percentage -/
def regular_discount : ℝ := 0.4

/-- The special sale discount percentage -/
def special_discount : ℝ := 0.1

/-- The final price after both discounts -/
def final_price : ℝ := original_price * (1 - regular_discount) * (1 - special_discount)

theorem jacket_price_correct : final_price = 135 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_correct_l67_6755


namespace NUMINAMATH_CALUDE_min_radius_of_circle_l67_6719

theorem min_radius_of_circle (r a b : ℝ) : 
  ((a - (r + 1))^2 + b^2 = r^2) →  -- Point (a, b) is on the circle
  (b^2 ≥ 4*a) →                    -- Condition b^2 ≥ 4a
  (r ≥ 0) →                        -- Radius is non-negative
  (r ≥ 4) :=                       -- Minimum value of r is 4
by sorry

end NUMINAMATH_CALUDE_min_radius_of_circle_l67_6719


namespace NUMINAMATH_CALUDE_star_property_l67_6774

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := 
  fun (a, b) (c, d) => (a + d, b - c)

theorem star_property : 
  ∃ (y : ℤ), star (3, y) (4, 2) = star (4, 5) (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_star_property_l67_6774


namespace NUMINAMATH_CALUDE_product_of_roots_l67_6768

theorem product_of_roots (a b c d : ℂ) : 
  (3 * a^4 - 8 * a^3 + a^2 + 4 * a - 10 = 0) ∧ 
  (3 * b^4 - 8 * b^3 + b^2 + 4 * b - 10 = 0) ∧ 
  (3 * c^4 - 8 * c^3 + c^2 + 4 * c - 10 = 0) ∧ 
  (3 * d^4 - 8 * d^3 + d^2 + 4 * d - 10 = 0) →
  a * b * c * d = -10/3 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l67_6768


namespace NUMINAMATH_CALUDE_largest_root_is_four_l67_6785

/-- The polynomial function representing the difference between the curve and the line -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^6 - 10*x^5 + 29*x^4 - 4*x^3 + a*x^2 - b*x - c

/-- The statement that the polynomial has exactly three distinct roots, each with multiplicity 2 -/
def has_three_double_roots (a b c : ℝ) : Prop :=
  ∃ (p q r : ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    ∀ x, f a b c x = 0 ↔ (x = p ∨ x = q ∨ x = r)

/-- The theorem stating that under the given conditions, 4 is the largest root -/
theorem largest_root_is_four (a b c : ℝ) (h : has_three_double_roots a b c) :
  ∃ (p q r : ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (∀ x, f a b c x = 0 ↔ (x = p ∨ x = q ∨ x = r)) ∧
    4 = max p (max q r) :=
  sorry

end NUMINAMATH_CALUDE_largest_root_is_four_l67_6785


namespace NUMINAMATH_CALUDE_rationalize_sqrt_sum_l67_6754

def rationalize_denominator (x y z : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  sorry

theorem rationalize_sqrt_sum : 
  let (A, B, C, D, E, F) := rationalize_denominator (Real.sqrt 3) (Real.sqrt 5) (Real.sqrt 11)
  A + B + C + D + E + F = 97 := by sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_sum_l67_6754


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l67_6787

theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (right_angle : a^2 + b^2 = c^2) : 
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l67_6787


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_three_l67_6771

def is_divisible_by_three (n : ℕ) : Prop :=
  ∃ k : ℕ, 8*(n+2)^5 - n^2 + 14*n - 30 = 3*k

theorem largest_n_divisible_by_three :
  (∀ m : ℕ, m < 100000 → is_divisible_by_three m) ∧
  (∀ m : ℕ, m > 99999 → m < 100000 → ¬is_divisible_by_three m) :=
sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_three_l67_6771


namespace NUMINAMATH_CALUDE_solve_linear_equation_l67_6797

theorem solve_linear_equation (x : ℝ) (h : 3*x - 5*x + 7*x = 150) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l67_6797


namespace NUMINAMATH_CALUDE_smallest_change_l67_6788

def original : ℚ := 0.123456

def change_digit (n : ℕ) (d : ℕ) : ℚ :=
  if n = 1 then 0.823456
  else if n = 2 then 0.183456
  else if n = 3 then 0.128456
  else if n = 4 then 0.123856
  else if n = 6 then 0.123458
  else original

theorem smallest_change :
  ∀ n : ℕ, n ≠ 6 → change_digit 6 8 < change_digit n 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_change_l67_6788


namespace NUMINAMATH_CALUDE_point_on_line_l67_6793

/-- Given six points on a line and a point P satisfying certain conditions, prove OP -/
theorem point_on_line (a b c d e : ℝ) : 
  ∀ (O A B C D E P : ℝ), 
    O < A ∧ A < B ∧ B < C ∧ C < D ∧ D < E ∧   -- Points in order
    A - O = a ∧                               -- Distance OA
    B - O = b ∧                               -- Distance OB
    C - O = c ∧                               -- Distance OC
    D - O = d ∧                               -- Distance OE
    E - O = e ∧                               -- Distance OE
    C ≤ P ∧ P ≤ D ∧                           -- P between C and D
    (A - P) * (P - D) = (C - P) * (P - E) →   -- AP:PE = CP:PD
    P - O = (c * e - a * d) / (a - c + e - d) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l67_6793


namespace NUMINAMATH_CALUDE_committee_with_female_count_l67_6782

def total_members : ℕ := 30
def female_members : ℕ := 12
def male_members : ℕ := 18
def committee_size : ℕ := 5

theorem committee_with_female_count :
  (Nat.choose total_members committee_size) - (Nat.choose male_members committee_size) = 133938 :=
by sorry

end NUMINAMATH_CALUDE_committee_with_female_count_l67_6782


namespace NUMINAMATH_CALUDE_janelle_initial_green_marbles_l67_6772

/-- The number of bags of blue marbles Janelle bought -/
def blue_bags : ℕ := 6

/-- The number of marbles in each bag -/
def marbles_per_bag : ℕ := 10

/-- The number of green marbles in the gift -/
def green_marbles_gift : ℕ := 6

/-- The number of blue marbles in the gift -/
def blue_marbles_gift : ℕ := 8

/-- The number of marbles Janelle has left after giving the gift -/
def marbles_left : ℕ := 72

/-- The initial number of green marbles Janelle had -/
def initial_green_marbles : ℕ := 26

theorem janelle_initial_green_marbles :
  initial_green_marbles = green_marbles_gift + (marbles_left - (blue_bags * marbles_per_bag - blue_marbles_gift)) :=
by sorry

end NUMINAMATH_CALUDE_janelle_initial_green_marbles_l67_6772


namespace NUMINAMATH_CALUDE_white_shirts_count_l67_6762

/-- The number of white T-shirts in each pack -/
def white_shirts_per_pack : ℕ := sorry

/-- The number of packs of white T-shirts bought -/
def white_packs : ℕ := 3

/-- The number of packs of blue T-shirts bought -/
def blue_packs : ℕ := 2

/-- The number of blue T-shirts in each pack -/
def blue_shirts_per_pack : ℕ := 4

/-- The total number of T-shirts bought -/
def total_shirts : ℕ := 26

theorem white_shirts_count : white_shirts_per_pack = 6 := by
  sorry

end NUMINAMATH_CALUDE_white_shirts_count_l67_6762


namespace NUMINAMATH_CALUDE_four_circle_intersection_l67_6750

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A collection of circles in a square --/
structure CirclesInSquare where
  circles : List Circle
  squareSideLength : ℝ
  totalCircumference : ℝ

/-- A theorem stating that given a square of side length 1 containing multiple circles
    with a total circumference of 10, there exists a line that intersects at least
    four of these circles --/
theorem four_circle_intersection
  (cis : CirclesInSquare)
  (h1 : cis.squareSideLength = 1)
  (h2 : cis.totalCircumference = 10)
  : ∃ (line : ℝ × ℝ → Prop), ∃ (intersectedCircles : Finset Circle),
    intersectedCircles.card ≥ 4 ∧
    ∀ c ∈ intersectedCircles, c ∈ cis.circles ∧ ∃ p, line p ∧ (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 :=
sorry

end NUMINAMATH_CALUDE_four_circle_intersection_l67_6750


namespace NUMINAMATH_CALUDE_vertex_locus_is_hyperbola_l67_6753

/-- The locus of the vertex of a parabola is a hyperbola -/
theorem vertex_locus_is_hyperbola 
  (a b : ℝ) 
  (h : 8 * a^2 + 4 * a * b = b^3) : 
  ∃ (x y : ℝ), x * y = 1 ∧ 
  x = -b / (2 * a) ∧ 
  y = (4 * a - b^2) / (4 * a) := by
  sorry

end NUMINAMATH_CALUDE_vertex_locus_is_hyperbola_l67_6753


namespace NUMINAMATH_CALUDE_inequality_preservation_l67_6709

theorem inequality_preservation (a b : ℝ) (h : a < b) : a - 3 < b - 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l67_6709


namespace NUMINAMATH_CALUDE_blanche_eggs_l67_6714

theorem blanche_eggs (gertrude nancy martha blanche total_eggs : ℕ) : 
  gertrude = 4 →
  nancy = 2 →
  martha = 2 →
  total_eggs = gertrude + nancy + martha + blanche →
  total_eggs - 2 = 9 →
  blanche = 3 := by
sorry

end NUMINAMATH_CALUDE_blanche_eggs_l67_6714


namespace NUMINAMATH_CALUDE_corn_increase_factor_l67_6776

theorem corn_increase_factor (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 1) 
  (h3 : 1 - x + x = 1/2) 
  (h4 : 1 - x + x/2 = 1/2) : 
  (3/2 * x) / (1/2 * x) = 3 := by sorry

end NUMINAMATH_CALUDE_corn_increase_factor_l67_6776


namespace NUMINAMATH_CALUDE_multiplication_formula_examples_l67_6729

theorem multiplication_formula_examples : 
  (102 * 98 = 9996) ∧ (99^2 = 9801) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_formula_examples_l67_6729


namespace NUMINAMATH_CALUDE_investment_rate_problem_l67_6730

/-- Proves that given the conditions of the investment problem, the rate of the first investment is 10% -/
theorem investment_rate_problem (total_investment : ℝ) (second_investment : ℝ) (second_rate : ℝ) (income_difference : ℝ) :
  total_investment = 2000 →
  second_investment = 750 →
  second_rate = 0.08 →
  income_difference = 65 →
  let first_investment := total_investment - second_investment
  let first_rate := (income_difference + second_investment * second_rate) / first_investment
  first_rate = 0.1 := by
  sorry

#check investment_rate_problem

end NUMINAMATH_CALUDE_investment_rate_problem_l67_6730


namespace NUMINAMATH_CALUDE_sequences_properties_l67_6737

def a (n : ℕ) : ℤ := (-3) ^ n
def b (n : ℕ) : ℤ := (-3) ^ n - 3
def c (n : ℕ) : ℤ := -(-3) ^ n - 1

def m (n : ℕ) : ℤ := a n + b n + c n

theorem sequences_properties :
  (a 5 = -243 ∧ b 5 = -246 ∧ c 5 = 242) ∧
  (∃ k : ℕ, a k + a (k + 1) + a (k + 2) = -1701) ∧
  (∀ n : ℕ,
    (n % 2 = 1 → max (a n) (max (b n) (c n)) - min (a n) (min (b n) (c n)) = -2 * m n - 6) ∧
    (n % 2 = 0 → max (a n) (max (b n) (c n)) - min (a n) (min (b n) (c n)) = 2 * m n + 9)) :=
by sorry

end NUMINAMATH_CALUDE_sequences_properties_l67_6737


namespace NUMINAMATH_CALUDE_operation_result_is_four_digit_l67_6794

/-- A nonzero digit is a natural number between 1 and 9, inclusive. -/
def NonzeroDigit : Type :=
  { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

/-- The result of the operation 543C + 721 - DE4 for any nonzero digits C, D, and E. -/
def OperationResult (C D E : NonzeroDigit) : ℕ :=
  5430 + C.val + 721 - (100 * D.val + 10 * E.val + 4)

/-- The theorem stating that the result of the operation is always a 4-digit number. -/
theorem operation_result_is_four_digit (C D E : NonzeroDigit) :
  1000 ≤ OperationResult C D E ∧ OperationResult C D E < 10000 :=
by sorry

end NUMINAMATH_CALUDE_operation_result_is_four_digit_l67_6794


namespace NUMINAMATH_CALUDE_alloy_gold_percentage_l67_6779

-- Define the weights and percentages
def total_weight : ℝ := 12.4
def metal_weight : ℝ := 6.2
def gold_percent_1 : ℝ := 0.60
def gold_percent_2 : ℝ := 0.40

-- Theorem statement
theorem alloy_gold_percentage :
  (metal_weight * gold_percent_1 + metal_weight * gold_percent_2) / total_weight = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_alloy_gold_percentage_l67_6779


namespace NUMINAMATH_CALUDE_shared_divisors_count_l67_6796

theorem shared_divisors_count (a b : ℕ) (ha : a = 9240) (hb : b = 8820) :
  (Finset.filter (fun d : ℕ => d ∣ a ∧ d ∣ b) (Finset.range (min a b + 1))).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_shared_divisors_count_l67_6796


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l67_6781

theorem smallest_angle_in_special_triangle : 
  ∀ (a b c : ℝ),
  a + b + c = 180 →
  c = 5 * a →
  b = 3 * a →
  a = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l67_6781


namespace NUMINAMATH_CALUDE_smallest_b_for_composite_l67_6765

theorem smallest_b_for_composite (b : ℕ+) (h : b = 9) :
  (∀ x : ℤ, ∃ a c : ℤ, a > 1 ∧ c > 1 ∧ x^4 + b^2 = a * c) ∧
  (∀ b' : ℕ+, b' < b → ∃ x : ℤ, ∀ a c : ℤ, (a > 1 ∧ c > 1 → x^4 + b'^2 ≠ a * c)) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_composite_l67_6765


namespace NUMINAMATH_CALUDE_nine_powers_equal_three_power_l67_6734

theorem nine_powers_equal_three_power (n : ℕ) : 
  9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n = 3^2012 → n = 1005 := by
  sorry

end NUMINAMATH_CALUDE_nine_powers_equal_three_power_l67_6734


namespace NUMINAMATH_CALUDE_subset_implies_x_equals_one_l67_6741

def A : Set ℝ := {0, 1, 2}
def B (x : ℝ) : Set ℝ := {1, 2/x}

theorem subset_implies_x_equals_one (x : ℝ) (h : B x ⊆ A) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_x_equals_one_l67_6741


namespace NUMINAMATH_CALUDE_inverse_proportion_solution_l67_6705

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_solution (x y : ℝ) :
  InverselyProportional x y →
  x + y = 30 →
  x - y = 10 →
  (∃ y' : ℝ, InverselyProportional 4 y' ∧ y' = 50) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_solution_l67_6705


namespace NUMINAMATH_CALUDE_max_value_z_plus_x_l67_6702

theorem max_value_z_plus_x :
  ∀ x y z t : ℝ,
  x^2 + y^2 = 4 →
  z^2 + t^2 = 9 →
  x*t + y*z ≥ 6 →
  z + x ≤ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_z_plus_x_l67_6702


namespace NUMINAMATH_CALUDE_rachel_made_18_dollars_l67_6769

/-- The amount of money Rachel made selling chocolate bars -/
def rachel_money (total_bars : ℕ) (unsold_bars : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Theorem stating that Rachel made $18 -/
theorem rachel_made_18_dollars :
  rachel_money 13 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_rachel_made_18_dollars_l67_6769


namespace NUMINAMATH_CALUDE_inequality_pattern_l67_6763

theorem inequality_pattern (x a : ℝ) : 
  x > 0 →
  x + 1/x ≥ 2 →
  x + 4/x^2 ≥ 3 →
  x + 27/x^3 ≥ 4 →
  x + a/x^4 ≥ 5 →
  a = 4^4 := by
sorry

end NUMINAMATH_CALUDE_inequality_pattern_l67_6763


namespace NUMINAMATH_CALUDE_certain_number_problem_l67_6736

theorem certain_number_problem (x : ℝ) : 
  (0.20 * x) - (1/3) * (0.20 * x) = 24 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l67_6736


namespace NUMINAMATH_CALUDE_manufacturing_department_percentage_l67_6704

theorem manufacturing_department_percentage (total_degrees : ℝ) (manufacturing_degrees : ℝ) :
  total_degrees = 360 →
  manufacturing_degrees = 216 →
  (manufacturing_degrees / total_degrees) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_department_percentage_l67_6704


namespace NUMINAMATH_CALUDE_sophie_donuts_left_l67_6707

/-- Calculates the number of donuts left for Sophie after giving some away --/
def donuts_left (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_given_to_mom : ℕ) (donuts_given_to_sister : ℕ) : ℕ :=
  total_boxes * donuts_per_box - boxes_given_to_mom * donuts_per_box - donuts_given_to_sister

/-- Proves that Sophie has 30 donuts left --/
theorem sophie_donuts_left : donuts_left 4 12 1 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sophie_donuts_left_l67_6707


namespace NUMINAMATH_CALUDE_product_equals_243_l67_6767

theorem product_equals_243 : 
  (1 / 3 : ℚ) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l67_6767


namespace NUMINAMATH_CALUDE_square_division_perimeter_paradox_l67_6700

theorem square_division_perimeter_paradox :
  ∃ (a : ℚ) (x : ℚ), 0 < x ∧ x < a ∧ 
    (2 * (a + x)).isInt ∧ 
    (2 * (2 * a - x)).isInt ∧ 
    ¬(4 * a).isInt := by
  sorry

end NUMINAMATH_CALUDE_square_division_perimeter_paradox_l67_6700


namespace NUMINAMATH_CALUDE_inequality_equivalent_to_interval_l67_6775

-- Define the inequality
def inequality (x : ℝ) : Prop := |8 - x| / 4 < 3

-- Define the interval
def interval (x : ℝ) : Prop := -4 < x ∧ x < 20

-- Theorem statement
theorem inequality_equivalent_to_interval :
  ∀ x : ℝ, inequality x ↔ interval x :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalent_to_interval_l67_6775


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l67_6701

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  b / a = 2 / 7 →  -- ratio of angles is 2:7
  a = 110 :=  -- complement of larger angle is 110°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l67_6701


namespace NUMINAMATH_CALUDE_min_teachers_is_six_l67_6715

/-- Represents the number of subjects for each discipline -/
structure SubjectCounts where
  maths : Nat
  physics : Nat
  chemistry : Nat

/-- Represents the constraints of the teaching system -/
structure TeachingSystem where
  subjects : SubjectCounts
  max_subjects_per_teacher : Nat
  specialized : Bool

/-- Calculates the minimum number of teachers required -/
def min_teachers_required (system : TeachingSystem) : Nat :=
  if system.specialized then
    let maths_teachers := (system.subjects.maths + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher
    let physics_teachers := (system.subjects.physics + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher
    let chemistry_teachers := (system.subjects.chemistry + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher
    maths_teachers + physics_teachers + chemistry_teachers
  else
    let total_subjects := system.subjects.maths + system.subjects.physics + system.subjects.chemistry
    (total_subjects + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher

/-- The main theorem stating that the minimum number of teachers required is 6 -/
theorem min_teachers_is_six (system : TeachingSystem) 
  (h1 : system.subjects = { maths := 6, physics := 5, chemistry := 5 })
  (h2 : system.max_subjects_per_teacher = 4)
  (h3 : system.specialized = true) : 
  min_teachers_required system = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_teachers_is_six_l67_6715


namespace NUMINAMATH_CALUDE_elijah_score_l67_6798

/-- Proves that Elijah's score is 43 points given the team's total score,
    number of players, and average score of other players. -/
theorem elijah_score (total_score : ℕ) (num_players : ℕ) (other_avg : ℕ) 
  (h1 : total_score = 85)
  (h2 : num_players = 8)
  (h3 : other_avg = 6) :
  total_score - (num_players - 1) * other_avg = 43 := by
  sorry

#check elijah_score

end NUMINAMATH_CALUDE_elijah_score_l67_6798


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l67_6746

/-- Given a quadratic inequality x² + bx + c < 0 with solution set (-1, 2),
    prove that bx² + x + c < 0 has solution set ℝ -/
theorem quadratic_inequality_solution_set
  (b c : ℝ)
  (h : Set.Ioo (-1 : ℝ) 2 = {x | x^2 + b*x + c < 0}) :
  Set.univ = {x | b*x^2 + x + c < 0} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l67_6746


namespace NUMINAMATH_CALUDE_sector_cone_theorem_l67_6716

/-- Represents a cone formed from a circular sector -/
structure SectorCone where
  sector_angle : ℝ  -- Central angle of the sector in degrees
  sector_radius : ℝ  -- Radius of the sector
  base_radius : ℝ    -- Radius of the cone's base
  slant_height : ℝ   -- Slant height of the cone

/-- Checks if the cone's dimensions are consistent with the sector -/
def is_valid_sector_cone (cone : SectorCone) : Prop :=
  cone.sector_angle = 270 ∧
  cone.sector_radius = 12 ∧
  cone.base_radius = 9 ∧
  cone.slant_height = 12 ∧
  cone.sector_angle / 360 * (2 * Real.pi * cone.sector_radius) = 2 * Real.pi * cone.base_radius ∧
  cone.slant_height = cone.sector_radius

theorem sector_cone_theorem (cone : SectorCone) :
  is_valid_sector_cone cone :=
sorry

end NUMINAMATH_CALUDE_sector_cone_theorem_l67_6716


namespace NUMINAMATH_CALUDE_BD_expression_A_B_D_collinear_l67_6748

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the non-collinear vectors a and b
variable (a b : V)
variable (h_non_collinear : a ≠ 0 ∧ b ≠ 0 ∧ ¬∃ (r : ℝ), a = r • b)

-- Define the vectors AB, OB, and OD
def AB (a b : V) : V := 2 • a - 8 • b
def OB (a b : V) : V := a + 3 • b
def OD (a b : V) : V := 2 • a - b

-- Statement 1: Express BD in terms of a and b
theorem BD_expression (a b : V) : OD a b - OB a b = a - 4 • b := by sorry

-- Statement 2: Prove that A, B, and D are collinear
theorem A_B_D_collinear (a b : V) : 
  ∃ (r : ℝ), AB a b = r • (OD a b - OB a b) := by sorry

end NUMINAMATH_CALUDE_BD_expression_A_B_D_collinear_l67_6748


namespace NUMINAMATH_CALUDE_initial_caps_count_l67_6789

-- Define the variables
def lost_caps : ℕ := 66
def current_caps : ℕ := 25

-- Define the theorem
theorem initial_caps_count : ∃ initial_caps : ℕ, initial_caps = lost_caps + current_caps :=
  sorry

end NUMINAMATH_CALUDE_initial_caps_count_l67_6789


namespace NUMINAMATH_CALUDE_flower_bed_area_ratio_l67_6757

theorem flower_bed_area_ratio :
  ∀ (l w : ℝ), l > 0 → w > 0 →
  (l * w) / ((2 * l) * (3 * w)) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_area_ratio_l67_6757


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l67_6791

/-- Given an arithmetic sequence where the first term is 10/11 and the fifteenth term is 8/9,
    the eighth term is 89/99. -/
theorem arithmetic_sequence_eighth_term 
  (a : ℕ → ℚ)  -- a is the sequence
  (h1 : a 1 = 10 / 11)  -- first term is 10/11
  (h15 : a 15 = 8 / 9)  -- fifteenth term is 8/9
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- arithmetic sequence condition
  : a 8 = 89 / 99 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l67_6791


namespace NUMINAMATH_CALUDE_expression_evaluation_l67_6790

theorem expression_evaluation :
  let a : ℤ := 1
  let b : ℤ := -1
  a + 2*b + 2*(a + 2*b) + 1 = -2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l67_6790


namespace NUMINAMATH_CALUDE_minimal_edge_count_l67_6706

/-- A graph with 7 vertices satisfying the given conditions -/
structure MinimalGraph where
  -- The set of vertices
  V : Finset ℕ
  -- The set of edges
  E : Finset (Finset ℕ)
  -- There are exactly 7 vertices
  vertex_count : V.card = 7
  -- Each edge connects exactly two vertices
  edge_valid : ∀ e ∈ E, e.card = 2 ∧ e ⊆ V
  -- Among any three vertices, at least two are connected
  connected_condition : ∀ {a b c}, a ∈ V → b ∈ V → c ∈ V → a ≠ b → b ≠ c → a ≠ c →
    {a, b} ∈ E ∨ {b, c} ∈ E ∨ {a, c} ∈ E

/-- The theorem stating that the minimal number of edges is 9 -/
theorem minimal_edge_count (G : MinimalGraph) : G.E.card = 9 := by
  sorry

end NUMINAMATH_CALUDE_minimal_edge_count_l67_6706


namespace NUMINAMATH_CALUDE_fraction_sum_l67_6752

theorem fraction_sum (a b : ℚ) (h : a / b = 1 / 2) : (a + b) / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l67_6752


namespace NUMINAMATH_CALUDE_team_a_wins_l67_6718

/-- Represents the outcome of a match for a team -/
inductive MatchResult
  | Win
  | Draw
  | Loss

/-- Calculates points for a given match result -/
def pointsForResult (result : MatchResult) : Nat :=
  match result with
  | MatchResult.Win => 3
  | MatchResult.Draw => 1
  | MatchResult.Loss => 0

/-- Represents the results of a series of matches for a team -/
structure TeamResults where
  wins : Nat
  draws : Nat
  losses : Nat

/-- Calculates total points for a team's results -/
def totalPoints (results : TeamResults) : Nat :=
  results.wins * (pointsForResult MatchResult.Win) +
  results.draws * (pointsForResult MatchResult.Draw) +
  results.losses * (pointsForResult MatchResult.Loss)

theorem team_a_wins (total_matches : Nat) (team_a_points : Nat)
    (h1 : total_matches = 10)
    (h2 : team_a_points = 22)
    (h3 : ∀ (r : TeamResults), 
      r.wins + r.draws = total_matches → 
      r.losses = 0 → 
      totalPoints r = team_a_points → 
      r.wins = 6) :
  ∃ (r : TeamResults), r.wins + r.draws = total_matches ∧ 
                       r.losses = 0 ∧ 
                       totalPoints r = team_a_points ∧ 
                       r.wins = 6 := by
  sorry

#check team_a_wins

end NUMINAMATH_CALUDE_team_a_wins_l67_6718


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l67_6740

theorem arithmetic_simplification : 2000 - 80 + 200 - 120 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l67_6740


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l67_6799

/-- An isosceles triangle PQR with given side lengths and altitude properties -/
structure IsoscelesTriangle where
  /-- Length of equal sides PQ and PR -/
  side : ℝ
  /-- Length of base QR -/
  base : ℝ
  /-- Altitude PS bisects base QR -/
  altitude_bisects_base : True

/-- The area of the isosceles triangle PQR is 360 square units -/
theorem isosceles_triangle_area
  (t : IsoscelesTriangle)
  (h1 : t.side = 41)
  (h2 : t.base = 18) :
  t.side * t.base / 2 = 360 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l67_6799


namespace NUMINAMATH_CALUDE_profit_is_085_l67_6710

/-- Calculates the total profit for Niko's sock reselling business --/
def calculate_profit : ℝ :=
  let initial_cost : ℝ := 9 * 2
  let discount_rate : ℝ := 0.1
  let discount : ℝ := initial_cost * discount_rate
  let cost_after_discount : ℝ := initial_cost - discount
  let shipping_storage : ℝ := 5
  let total_cost : ℝ := cost_after_discount + shipping_storage
  let resell_price_4_pairs : ℝ := 4 * (2 + 2 * 0.25)
  let resell_price_5_pairs : ℝ := 5 * (2 + 0.2)
  let total_resell_price : ℝ := resell_price_4_pairs + resell_price_5_pairs
  let sales_tax_rate : ℝ := 0.05
  let sales_tax : ℝ := total_resell_price * sales_tax_rate
  let total_revenue : ℝ := total_resell_price + sales_tax
  total_revenue - total_cost

theorem profit_is_085 : calculate_profit = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_085_l67_6710


namespace NUMINAMATH_CALUDE_three_rulers_left_l67_6756

/-- The number of rulers left in a drawer after some are removed -/
def rulers_left (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem stating that 3 rulers are left in the drawer -/
theorem three_rulers_left : rulers_left 14 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_rulers_left_l67_6756


namespace NUMINAMATH_CALUDE_valid_mixture_weight_l67_6777

/-- A cement mixture composed of sand, water, and gravel -/
structure CementMixture where
  total_weight : ℝ
  sand_fraction : ℝ
  water_fraction : ℝ
  gravel_weight : ℝ

/-- The cement mixture satisfies the given conditions -/
def is_valid_mixture (m : CementMixture) : Prop :=
  m.sand_fraction = 1/3 ∧
  m.water_fraction = 1/4 ∧
  m.gravel_weight = 10 ∧
  m.sand_fraction * m.total_weight + m.water_fraction * m.total_weight + m.gravel_weight = m.total_weight

/-- The theorem stating that a valid mixture has a total weight of 24 pounds -/
theorem valid_mixture_weight (m : CementMixture) (h : is_valid_mixture m) : m.total_weight = 24 := by
  sorry

end NUMINAMATH_CALUDE_valid_mixture_weight_l67_6777


namespace NUMINAMATH_CALUDE_valera_car_position_l67_6713

/-- Represents a train with a fixed number of cars -/
structure Train :=
  (num_cars : ℕ)

/-- Represents the meeting of two trains -/
structure TrainMeeting :=
  (train1 : Train)
  (train2 : Train)
  (total_passing_time : ℕ)
  (sasha_passing_time : ℕ)
  (sasha_car : ℕ)

/-- Theorem stating the position of Valera's car -/
theorem valera_car_position
  (meeting : TrainMeeting)
  (h1 : meeting.train1.num_cars = 15)
  (h2 : meeting.train2.num_cars = 15)
  (h3 : meeting.total_passing_time = 60)
  (h4 : meeting.sasha_passing_time = 28)
  (h5 : meeting.sasha_car = 3) :
  ∃ (valera_car : ℕ), valera_car = 12 :=
by sorry

end NUMINAMATH_CALUDE_valera_car_position_l67_6713


namespace NUMINAMATH_CALUDE_peter_reading_time_l67_6749

/-- The time it takes Peter to read one book -/
def peter_time : ℝ := 18

/-- The number of books Peter and Kristin each need to read -/
def total_books : ℕ := 20

/-- The ratio of Peter's reading speed to Kristin's -/
def speed_ratio : ℝ := 3

/-- The time it takes Kristin to read half of her books -/
def kristin_half_time : ℝ := 540

theorem peter_reading_time :
  peter_time = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_peter_reading_time_l67_6749


namespace NUMINAMATH_CALUDE_distance_inequality_l67_6778

-- Define the types for planes, lines, and points
variable (Plane Line Point : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the "in" relation for lines and planes
variable (in_plane : Line → Plane → Prop)

-- Define the "on" relation for points and lines
variable (on_line : Point → Line → Prop)

-- Define the distance function
variable (distance : Point → Point → ℝ)
variable (distance_point_to_line : Point → Line → ℝ)
variable (distance_line_to_line : Line → Line → ℝ)

-- Define the specific objects in our problem
variable (α β : Plane) (m n : Line) (A B : Point)

-- Define the theorem
theorem distance_inequality 
  (h_parallel : parallel α β)
  (h_m_in_α : in_plane m α)
  (h_n_in_β : in_plane n β)
  (h_A_on_m : on_line A m)
  (h_B_on_n : on_line B n)
  (h_a : distance A B = a)
  (h_b : distance_point_to_line A n = b)
  (h_c : distance_line_to_line m n = c)
  : c ≤ a ∧ a ≤ b :=
by sorry

end NUMINAMATH_CALUDE_distance_inequality_l67_6778


namespace NUMINAMATH_CALUDE_lawn_mowing_l67_6711

theorem lawn_mowing (total_time : ℝ) (worked_time : ℝ) :
  total_time = 6 →
  worked_time = 3 →
  1 - (worked_time / total_time) = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_l67_6711


namespace NUMINAMATH_CALUDE_sum_of_bases_equals_999_l67_6780

/-- Converts a number from base 11 to base 10 -/
def base11To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 12 to base 10 -/
def base12To10 (n : ℕ) : ℕ := sorry

/-- Represents the digit A in base 12 -/
def A : ℕ := 10

theorem sum_of_bases_equals_999 :
  base11To10 379 + base12To10 (3 * 12^2 + A * 12 + 9) = 999 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equals_999_l67_6780


namespace NUMINAMATH_CALUDE_absolute_value_inequality_implies_range_l67_6717

theorem absolute_value_inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, |2*x - 1| + |x + 2| ≥ a^2 + (1/2)*a + 2) →
  -1 ≤ a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_implies_range_l67_6717


namespace NUMINAMATH_CALUDE_a_n_formula_l67_6747

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem a_n_formula (a : ℕ → ℝ) (h1 : arithmetic_sequence (λ n => a (n + 1) - a n))
  (h2 : a 1 - a 0 = 1) (h3 : ∀ n : ℕ, (a (n + 2) - a (n + 1)) - (a (n + 1) - a n) = 2) :
  ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end NUMINAMATH_CALUDE_a_n_formula_l67_6747


namespace NUMINAMATH_CALUDE_focus_of_parabola_l67_6739

/-- The focus of the parabola x = -1/4 * y^2 -/
def parabola_focus : ℝ × ℝ := (-1, 0)

/-- The equation of the parabola -/
def is_on_parabola (x y : ℝ) : Prop := x = -1/4 * y^2

/-- Theorem stating that the focus of the parabola x = -1/4 * y^2 is at (-1, 0) -/
theorem focus_of_parabola :
  let (f, g) := parabola_focus
  ∀ (x y : ℝ), is_on_parabola x y →
    (x - f)^2 + y^2 = (x - (-f))^2 :=
by sorry

end NUMINAMATH_CALUDE_focus_of_parabola_l67_6739


namespace NUMINAMATH_CALUDE_min_value_function_extremum_function_l67_6721

-- Part 1
theorem min_value_function (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) + 6 ≥ 9 ∧
  (x + 4 / (x + 1) + 6 = 9 ↔ x = 1) :=
sorry

-- Part 2
theorem extremum_function (x : ℝ) (h : x > 1) :
  (x^2 + 8) / (x - 1) ≥ 8 ∧
  ((x^2 + 8) / (x - 1) = 8 ↔ x = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_function_extremum_function_l67_6721


namespace NUMINAMATH_CALUDE_compound_interest_rate_exists_unique_l67_6758

theorem compound_interest_rate_exists_unique (P : ℝ) (h1 : P > 0) :
  ∃! r : ℝ, r > 0 ∧ r < 1 ∧ 
    800 = P * (1 + r)^3 ∧
    820 = P * (1 + r)^4 :=
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_exists_unique_l67_6758
