import Mathlib

namespace NUMINAMATH_CALUDE_opposite_unit_vector_l1130_113000

/-- Given a vector a = (-3, 4), prove that the unit vector a₀ in the opposite direction of a has coordinates (3/5, -4/5). -/
theorem opposite_unit_vector (a : ℝ × ℝ) (h : a = (-3, 4)) :
  let a₀ := (-(a.1) / Real.sqrt ((a.1)^2 + (a.2)^2), -(a.2) / Real.sqrt ((a.1)^2 + (a.2)^2))
  a₀ = (3/5, -4/5) := by
sorry


end NUMINAMATH_CALUDE_opposite_unit_vector_l1130_113000


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1130_113008

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : Nat, p.Prime ∧ p ∣ (36^2 + 49^2) ∧ ∀ q : Nat, q.Prime → q ∣ (36^2 + 49^2) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1130_113008


namespace NUMINAMATH_CALUDE_travis_payment_l1130_113007

def payment_calculation (total_bowls glass_bowls ceramic_bowls base_fee safe_delivery_fee
                         broken_glass_charge broken_ceramic_charge lost_glass_charge lost_ceramic_charge
                         additional_glass_fee additional_ceramic_fee lost_glass lost_ceramic
                         broken_glass broken_ceramic : ℕ) : ℚ :=
  let safe_glass := glass_bowls - lost_glass - broken_glass
  let safe_ceramic := ceramic_bowls - lost_ceramic - broken_ceramic
  let safe_delivery_payment := (safe_glass + safe_ceramic) * safe_delivery_fee
  let broken_lost_charges := broken_glass * broken_glass_charge + broken_ceramic * broken_ceramic_charge +
                             lost_glass * lost_glass_charge + lost_ceramic * lost_ceramic_charge
  let additional_moving_fee := glass_bowls * additional_glass_fee + ceramic_bowls * additional_ceramic_fee
  (base_fee + safe_delivery_payment - broken_lost_charges + additional_moving_fee : ℚ)

theorem travis_payment :
  payment_calculation 638 375 263 100 3 5 4 6 3 (1/2) (1/4) 9 3 10 5 = 2053.25 := by
  sorry

end NUMINAMATH_CALUDE_travis_payment_l1130_113007


namespace NUMINAMATH_CALUDE_deposit_calculation_l1130_113034

theorem deposit_calculation (total_price : ℝ) (deposit_percentage : ℝ) (remaining_amount : ℝ) 
  (h1 : deposit_percentage = 0.1)
  (h2 : remaining_amount = 720)
  (h3 : total_price * (1 - deposit_percentage) = remaining_amount) :
  total_price * deposit_percentage = 80 := by
  sorry

end NUMINAMATH_CALUDE_deposit_calculation_l1130_113034


namespace NUMINAMATH_CALUDE_pencil_eraser_notebook_cost_l1130_113087

theorem pencil_eraser_notebook_cost 
  (h1 : 20 * x + 3 * y + 2 * z = 32) 
  (h2 : 39 * x + 5 * y + 3 * z = 58) : 
  5 * x + 5 * y + 5 * z = 30 := by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_notebook_cost_l1130_113087


namespace NUMINAMATH_CALUDE_carnival_tickets_l1130_113005

theorem carnival_tickets (num_friends : ℕ) (total_tickets : ℕ) (h1 : num_friends = 6) (h2 : total_tickets = 234) :
  (total_tickets / num_friends : ℕ) = 39 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_l1130_113005


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1130_113083

def A : Set ℝ := {1, 2, 3, 4}

def B : Set ℝ := {x : ℝ | ∃ y ∈ A, y = 2 * x}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1130_113083


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l1130_113084

theorem regular_polygon_interior_angle_sum :
  ∀ n : ℕ,
  n > 2 →
  (360 : ℝ) / n = 20 →
  (n - 2) * 180 = 2880 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l1130_113084


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l1130_113032

theorem quadratic_root_zero (a : ℝ) : 
  (∃ x, (a - 1) * x^2 + x + a^2 - 1 = 0) ∧ 
  ((a - 1) * 0^2 + 0 + a^2 - 1 = 0) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l1130_113032


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1130_113018

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 3 - Complex.I → z = -1 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1130_113018


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1130_113044

theorem simplify_polynomial (x : ℝ) : (3*x)^4 + (3*x)*(x^3) + 2*x^5 = 84*x^4 + 2*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1130_113044


namespace NUMINAMATH_CALUDE_last_colored_square_l1130_113095

/-- Represents a position in the rectangle --/
structure Position where
  row : Nat
  col : Nat

/-- Represents the dimensions of the rectangle --/
structure Dimensions where
  width : Nat
  height : Nat

/-- Represents the spiral coloring process --/
def spiralColor (dims : Dimensions) : Position :=
  sorry

/-- Theorem stating the last colored square in a 200x100 rectangle --/
theorem last_colored_square :
  spiralColor ⟨200, 100⟩ = ⟨51, 50⟩ := by
  sorry

end NUMINAMATH_CALUDE_last_colored_square_l1130_113095


namespace NUMINAMATH_CALUDE_equation_solutions_no_solutions_l1130_113004

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem equation_solutions (n k : ℕ) :
  (∃ A : ℕ, A = 7 ∧ factorial n + A * n = n^k) ↔ (n = 2 ∧ k = 4) ∨ (n = 3 ∧ k = 3) :=
sorry

theorem no_solutions (n k : ℕ) :
  ¬(∃ A : ℕ, A = 2012 ∧ factorial n + A * n = n^k) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_no_solutions_l1130_113004


namespace NUMINAMATH_CALUDE_platform_length_l1130_113058

/-- Given a train of length 300 meters that crosses a signal pole in 20 seconds
    and a platform in 39 seconds, the length of the platform is 285 meters. -/
theorem platform_length (train_length : ℝ) (pole_time : ℝ) (platform_time : ℝ) :
  train_length = 300 →
  pole_time = 20 →
  platform_time = 39 →
  ∃ platform_length : ℝ,
    platform_length = 285 ∧
    train_length / pole_time * platform_time = train_length + platform_length :=
by
  sorry


end NUMINAMATH_CALUDE_platform_length_l1130_113058


namespace NUMINAMATH_CALUDE_max_area_central_angle_l1130_113080

/-- The circumference of the sector -/
def circumference : ℝ := 40

/-- The radius of the sector -/
noncomputable def radius : ℝ := sorry

/-- The arc length of the sector -/
noncomputable def arc_length : ℝ := sorry

/-- The area of the sector -/
noncomputable def area (r : ℝ) : ℝ := 20 * r - r^2

/-- The central angle of the sector -/
noncomputable def central_angle : ℝ := sorry

/-- Theorem: The central angle that maximizes the area of a sector with circumference 40 is 2 radians -/
theorem max_area_central_angle :
  circumference = 2 * radius + arc_length →
  arc_length = central_angle * radius →
  central_angle = 2 ∧ IsLocalMax area radius :=
sorry

end NUMINAMATH_CALUDE_max_area_central_angle_l1130_113080


namespace NUMINAMATH_CALUDE_bianca_extra_flowers_l1130_113051

/-- The number of extra flowers Bianca picked -/
def extra_flowers (tulips roses daffodils sunflowers used : ℕ) : ℕ :=
  tulips + roses + daffodils + sunflowers - used

/-- Proof that Bianca picked 29 extra flowers -/
theorem bianca_extra_flowers :
  extra_flowers 57 73 45 35 181 = 29 := by
  sorry

end NUMINAMATH_CALUDE_bianca_extra_flowers_l1130_113051


namespace NUMINAMATH_CALUDE_count_true_propositions_l1130_113023

/-- The number of true propositions among the original, converse, inverse, and contrapositive
    of the statement "For real numbers a, b, c, and d, if a=b and c=d, then a+c=b+d" -/
def num_true_propositions : ℕ := 2

/-- The original proposition -/
def original_prop (a b c d : ℝ) : Prop :=
  (a = b ∧ c = d) → (a + c = b + d)

theorem count_true_propositions :
  (∀ a b c d : ℝ, original_prop a b c d) ∧
  (∃ a b c d : ℝ, ¬(a + c = b + d → a = b ∧ c = d)) ∧
  num_true_propositions = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_true_propositions_l1130_113023


namespace NUMINAMATH_CALUDE_discount_order_difference_l1130_113096

theorem discount_order_difference : 
  let original_price : ℚ := 25
  let flat_discount : ℚ := 4
  let percentage_discount : ℚ := 0.2
  let price_flat_then_percent : ℚ := (original_price - flat_discount) * (1 - percentage_discount)
  let price_percent_then_flat : ℚ := (original_price * (1 - percentage_discount)) - flat_discount
  price_flat_then_percent - price_percent_then_flat = 0.8 := by sorry

end NUMINAMATH_CALUDE_discount_order_difference_l1130_113096


namespace NUMINAMATH_CALUDE_expand_and_simplify_expression_l1130_113049

theorem expand_and_simplify_expression (x : ℝ) :
  2*x*(3*x^2 - 4*x + 5) - (x^2 - 3*x)*(4*x + 5) = 2*x^3 - x^2 + 25*x := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_expression_l1130_113049


namespace NUMINAMATH_CALUDE_cookie_sales_proof_l1130_113021

/-- Represents the total number of boxes of cookies sold -/
def total_boxes (chocolate_chip_boxes : ℝ) (plain_boxes : ℝ) : ℝ :=
  chocolate_chip_boxes + plain_boxes

/-- Represents the total sales value -/
def total_sales (chocolate_chip_boxes : ℝ) (plain_boxes : ℝ) : ℝ :=
  1.25 * chocolate_chip_boxes + 0.75 * plain_boxes

theorem cookie_sales_proof :
  ∀ (chocolate_chip_boxes : ℝ) (plain_boxes : ℝ),
    plain_boxes = 793.375 →
    total_sales chocolate_chip_boxes plain_boxes = 1586.75 →
    total_boxes chocolate_chip_boxes plain_boxes = 1586.75 :=
by
  sorry

#check cookie_sales_proof

end NUMINAMATH_CALUDE_cookie_sales_proof_l1130_113021


namespace NUMINAMATH_CALUDE_min_value_theorem_l1130_113054

theorem min_value_theorem (m : ℝ) (hm : m > 0)
  (h : ∀ x : ℝ, |x + 1| + |2*x - 1| ≥ m)
  (a b c : ℝ) (heq : a^2 + 2*b^2 + 3*c^2 = m) :
  ∀ a' b' c' : ℝ, a'^2 + 2*b'^2 + 3*c'^2 = m → a + 2*b + 3*c ≥ a' + 2*b' + 3*c' → a + 2*b + 3*c ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1130_113054


namespace NUMINAMATH_CALUDE_permutation_square_sum_bounds_l1130_113036

def is_permutation (a : Fin 10 → ℕ) : Prop :=
  ∀ i : Fin 10, ∃ j : Fin 10, a j = i.val + 1

theorem permutation_square_sum_bounds 
  (a b : Fin 10 → ℕ) 
  (ha : is_permutation a) 
  (hb : is_permutation b) :
  (∃ k : Fin 10, a k ^ 2 + b k ^ 2 ≥ 101) ∧
  (∃ k : Fin 10, a k ^ 2 + b k ^ 2 ≤ 61) :=
sorry

end NUMINAMATH_CALUDE_permutation_square_sum_bounds_l1130_113036


namespace NUMINAMATH_CALUDE_donut_distribution_ways_l1130_113037

/-- The number of types of donuts available --/
def num_types : ℕ := 5

/-- The total number of donuts to be purchased --/
def total_donuts : ℕ := 8

/-- The number of donuts that must be purchased of the first type --/
def first_type_min : ℕ := 2

/-- The number of donuts that must be purchased of each other type --/
def other_types_min : ℕ := 1

/-- The number of remaining donuts to be distributed after mandatory purchases --/
def remaining_donuts : ℕ := total_donuts - (first_type_min + (num_types - 1) * other_types_min)

theorem donut_distribution_ways : 
  (Nat.choose (remaining_donuts + num_types - 1) (num_types - 1)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_donut_distribution_ways_l1130_113037


namespace NUMINAMATH_CALUDE_wood_sawed_off_l1130_113015

theorem wood_sawed_off (original_length final_length : ℝ) 
  (h1 : original_length = 0.41)
  (h2 : final_length = 0.08) :
  original_length - final_length = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_wood_sawed_off_l1130_113015


namespace NUMINAMATH_CALUDE_ratio_of_segments_l1130_113042

/-- Given four points A, B, C, and D on a line in that order, with AB = 2, BC = 5, and AD = 14,
    prove that the ratio of AC to BD is 7/12. -/
theorem ratio_of_segments (A B C D : ℝ) : 
  (A < B) → (B < C) → (C < D) → 
  (B - A = 2) → (C - B = 5) → (D - A = 14) →
  (C - A) / (D - B) = 7 / 12 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l1130_113042


namespace NUMINAMATH_CALUDE_sequence_gcd_theorem_l1130_113013

theorem sequence_gcd_theorem (d m : ℕ) (hd : d > 1) :
  ∃ k l : ℕ, k ≠ l ∧ Nat.gcd (2^(2^k) + d) (2^(2^l) + d) > m := by
  sorry

end NUMINAMATH_CALUDE_sequence_gcd_theorem_l1130_113013


namespace NUMINAMATH_CALUDE_divisibility_problem_l1130_113002

theorem divisibility_problem (x y : ℤ) (h : 5 ∣ (x + 9*y)) : 5 ∣ (8*x + 7*y) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1130_113002


namespace NUMINAMATH_CALUDE_distinct_convex_polygons_l1130_113025

/-- The number of points marked on the circle -/
def num_points : ℕ := 12

/-- The total number of subsets of the points -/
def total_subsets : ℕ := 2^num_points

/-- The number of subsets with 0 members -/
def subsets_0 : ℕ := (num_points.choose 0)

/-- The number of subsets with 1 member -/
def subsets_1 : ℕ := (num_points.choose 1)

/-- The number of subsets with 2 members -/
def subsets_2 : ℕ := (num_points.choose 2)

/-- The number of distinct convex polygons with three or more sides -/
def num_polygons : ℕ := total_subsets - subsets_0 - subsets_1 - subsets_2

theorem distinct_convex_polygons :
  num_polygons = 4017 :=
by sorry

end NUMINAMATH_CALUDE_distinct_convex_polygons_l1130_113025


namespace NUMINAMATH_CALUDE_always_real_roots_discriminant_one_implies_m_two_l1130_113040

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ :=
  2 * m * x^2 - (5 * m - 1) * x + 3 * m - 1

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ :=
  (5 * m - 1)^2 - 4 * 2 * m * (3 * m - 1)

-- Theorem stating that the equation always has real roots
theorem always_real_roots (m : ℝ) :
  ∃ x : ℝ, quadratic_equation m x = 0 :=
sorry

-- Theorem stating that when the discriminant is 1, m = 2
theorem discriminant_one_implies_m_two :
  ∀ m : ℝ, discriminant m = 1 → m = 2 :=
sorry

end NUMINAMATH_CALUDE_always_real_roots_discriminant_one_implies_m_two_l1130_113040


namespace NUMINAMATH_CALUDE_derivative_at_one_l1130_113063

-- Define the function
def f (x : ℝ) : ℝ := (2*x + 1)^2

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 12 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1130_113063


namespace NUMINAMATH_CALUDE_unique_solution_l1130_113022

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- Checks if a number has the pattern 1*1 -/
def hasPattern1x1 (n : ThreeDigitNumber) : Prop :=
  n.val / 100 = 1 ∧ n.val % 10 = 1

theorem unique_solution :
  ∀ (ab cd : TwoDigitNumber) (n : ThreeDigitNumber),
    ab.val * cd.val = n.val ∧ hasPattern1x1 n →
    ab.val = 11 ∧ cd.val = 11 ∧ n.val = 121 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1130_113022


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_existence_condition_l1130_113071

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |2*x + a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 2} :=
sorry

-- Part II
theorem range_of_a_for_existence_condition :
  (∃ x₀ : ℝ, f a x₀ + |x₀ - 2| < 3) ↔ (-7 < a ∧ a < -1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_existence_condition_l1130_113071


namespace NUMINAMATH_CALUDE_initial_number_count_l1130_113081

theorem initial_number_count (n : ℕ) (S : ℝ) : 
  S / n = 62 →
  (S - 45 - 55) / (n - 2) = 62.5 →
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_initial_number_count_l1130_113081


namespace NUMINAMATH_CALUDE_prob_A_not_in_A_is_two_thirds_l1130_113062

-- Define the number of volunteers and communities
def num_volunteers : ℕ := 4
def num_communities : ℕ := 3

-- Define a type for volunteers and communities
inductive Volunteer : Type
| A | B | C | D

inductive Community : Type
| A | B | C

-- Define an assignment as a function from Volunteer to Community
def Assignment := Volunteer → Community

-- Define a valid assignment
def valid_assignment (a : Assignment) : Prop :=
  ∀ c : Community, ∃ v : Volunteer, a v = c

-- Define the probability that volunteer A is not in community A
def prob_A_not_in_A (total_assignments : ℕ) (valid_assignments : ℕ) : ℚ :=
  (valid_assignments - (total_assignments / num_communities)) / valid_assignments

-- State the theorem
theorem prob_A_not_in_A_is_two_thirds :
  ∃ (total_assignments valid_assignments : ℕ),
    total_assignments > 0 ∧
    valid_assignments > 0 ∧
    valid_assignments ≤ total_assignments ∧
    prob_A_not_in_A total_assignments valid_assignments = 2/3 :=
sorry

end NUMINAMATH_CALUDE_prob_A_not_in_A_is_two_thirds_l1130_113062


namespace NUMINAMATH_CALUDE_ice_cube_distribution_l1130_113031

theorem ice_cube_distribution (total_cubes : ℕ) (num_chests : ℕ) (cubes_per_chest : ℕ) 
  (h1 : total_cubes = 294)
  (h2 : num_chests = 7)
  (h3 : total_cubes = num_chests * cubes_per_chest) :
  cubes_per_chest = 42 := by
  sorry

end NUMINAMATH_CALUDE_ice_cube_distribution_l1130_113031


namespace NUMINAMATH_CALUDE_least_possible_value_of_x_l1130_113069

theorem least_possible_value_of_x : 
  ∃ (x y z : ℤ), 
    (∃ k : ℤ, x = 2 * k) ∧ 
    (∃ m n : ℤ, y = 2 * m + 1 ∧ z = 2 * n + 1) ∧ 
    y - x > 5 ∧ 
    (∀ w : ℤ, w - x ≥ 9 → w ≥ z) ∧ 
    (∀ v : ℤ, (∃ j : ℤ, v = 2 * j) → v ≥ x) → 
    x = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_value_of_x_l1130_113069


namespace NUMINAMATH_CALUDE_multiple_of_six_between_twelve_and_thirty_l1130_113076

theorem multiple_of_six_between_twelve_and_thirty (x : ℕ) :
  (∃ k : ℕ, x = 6 * k) →
  x^2 > 144 →
  x < 30 →
  x = 18 ∨ x = 24 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_six_between_twelve_and_thirty_l1130_113076


namespace NUMINAMATH_CALUDE_ernie_circles_l1130_113067

theorem ernie_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) 
  (ali_circles : ℕ) (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) 
  (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) : 
  (total_boxes - ali_circles * ali_boxes_per_circle) / ernie_boxes_per_circle = 4 := by
  sorry

end NUMINAMATH_CALUDE_ernie_circles_l1130_113067


namespace NUMINAMATH_CALUDE_problem_statement_l1130_113057

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 1/y = 1 ∧ 1/x + y < 1/a + b) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 1/y = 1 → 1/x + y ≥ 4) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 1/y = 1 → x/y ≤ 1/4) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 1/y = 1 ∧ x/y = 1/4) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 1/y = 1 → 1/2 * y - x ≥ Real.sqrt 2 - 1) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 1/y = 1 ∧ 1/2 * y - x = Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1130_113057


namespace NUMINAMATH_CALUDE_mirror_wall_height_l1130_113038

def hall_of_mirrors (wall1_width wall2_width wall3_width total_area : ℝ) : Prop :=
  ∃ (height : ℝ),
    wall1_width * height + wall2_width * height + wall3_width * height = total_area

theorem mirror_wall_height :
  hall_of_mirrors 30 30 20 960 →
  ∃ (height : ℝ), height = 12 := by
sorry

end NUMINAMATH_CALUDE_mirror_wall_height_l1130_113038


namespace NUMINAMATH_CALUDE_bert_shopping_trip_l1130_113016

theorem bert_shopping_trip (initial_amount : ℝ) : 
  initial_amount = 52 →
  let hardware_spend := initial_amount / 4
  let after_hardware := initial_amount - hardware_spend
  let dryclean_spend := 9
  let after_dryclean := after_hardware - dryclean_spend
  let grocery_spend := after_dryclean / 2
  let final_amount := after_dryclean - grocery_spend
  final_amount = 15 := by
sorry

end NUMINAMATH_CALUDE_bert_shopping_trip_l1130_113016


namespace NUMINAMATH_CALUDE_expression_range_l1130_113094

def expression_value (parenthesization : List (List Nat)) : ℚ :=
  sorry

theorem expression_range :
  ∀ p : List (List Nat),
    (∀ n, n ∈ p.join → n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) →
    (∀ n, n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → n ∈ p.join) →
    1 / 362880 ≤ expression_value p ∧ expression_value p ≤ 181440 :=
  sorry

end NUMINAMATH_CALUDE_expression_range_l1130_113094


namespace NUMINAMATH_CALUDE_whole_number_between_fractions_l1130_113074

theorem whole_number_between_fractions (M : ℤ) : 
  (5 < (M : ℚ) / 4) ∧ ((M : ℚ) / 4 < 5.5) → M = 21 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_fractions_l1130_113074


namespace NUMINAMATH_CALUDE_right_triangle_polyhedron_faces_even_l1130_113006

/-- A convex polyhedron with right-angled triangular faces -/
structure RightTrianglePolyhedron where
  faces : ℕ
  isConvex : Bool
  allFacesRightTriangle : Bool
  facesAtLeastFour : faces ≥ 4

/-- Theorem stating that the number of faces in a right-angled triangle polyhedron is even -/
theorem right_triangle_polyhedron_faces_even (p : RightTrianglePolyhedron) : 
  Even p.faces := by sorry

end NUMINAMATH_CALUDE_right_triangle_polyhedron_faces_even_l1130_113006


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l1130_113086

theorem tan_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (40 * π / 180) = (Real.sqrt 3 + 1) / (Real.sqrt 3 * Real.cos (40 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l1130_113086


namespace NUMINAMATH_CALUDE_abes_age_l1130_113082

theorem abes_age (present_age : ℕ) : 
  present_age + (present_age - 7) = 29 → present_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_abes_age_l1130_113082


namespace NUMINAMATH_CALUDE_triangle_trig_max_l1130_113047

open Real

theorem triangle_trig_max (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  π/4 < B ∧ B < π/2 ∧
  a * cos B - b * cos A = (3/5) * c →
  ∃ (max_val : ℝ), max_val = -512 ∧ 
    ∀ x, x = tan (2*B) * (tan A)^3 → x ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_triangle_trig_max_l1130_113047


namespace NUMINAMATH_CALUDE_prime_counterexample_l1130_113088

theorem prime_counterexample : ∃ n : ℕ, 
  (Nat.Prime n ∧ ¬Nat.Prime (n + 2)) ∨ (¬Nat.Prime n ∧ Nat.Prime (n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_counterexample_l1130_113088


namespace NUMINAMATH_CALUDE_hexagon_division_existence_l1130_113017

/-- A hexagon is a polygon with six sides -/
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A line is represented by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A triangle is represented by three points -/
structure Triangle where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Predicate to check if two triangles are congruent -/
def areCongruentTriangles (t1 t2 : Triangle) : Prop := sorry

/-- Predicate to check if a line divides a hexagon into four congruent triangles -/
def dividesIntoFourCongruentTriangles (h : Hexagon) (l : Line) : Prop :=
  ∃ t1 t2 t3 t4 : Triangle,
    areCongruentTriangles t1 t2 ∧
    areCongruentTriangles t1 t3 ∧
    areCongruentTriangles t1 t4

/-- Theorem stating that there exists a hexagon that can be divided by a single line into four congruent triangles -/
theorem hexagon_division_existence :
  ∃ (h : Hexagon) (l : Line), dividesIntoFourCongruentTriangles h l := by sorry

end NUMINAMATH_CALUDE_hexagon_division_existence_l1130_113017


namespace NUMINAMATH_CALUDE_largest_product_sum_1976_l1130_113065

theorem largest_product_sum_1976 (n : ℕ) (factors : List ℕ) : 
  (factors.sum = 1976) →
  (factors.prod ≤ 2 * 3^658) :=
sorry

end NUMINAMATH_CALUDE_largest_product_sum_1976_l1130_113065


namespace NUMINAMATH_CALUDE_total_painting_cost_l1130_113064

/-- Calculate the last term of an arithmetic sequence -/
def lastTerm (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

/-- Count the number of digits in a natural number -/
def digitCount (n : ℕ) : ℕ :=
  if n < 10 then 1 else if n < 100 then 2 else 3

/-- Calculate the cost of painting numbers for one side of the street -/
def sideCost (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  let lastNum := lastTerm a₁ d n
  let twoDigitCount := (min 99 lastNum - a₁) / d + 1
  let threeDigitCount := n - twoDigitCount
  2 * (2 * twoDigitCount + 3 * threeDigitCount)

/-- The main theorem stating the total cost for painting all house numbers -/
theorem total_painting_cost : 
  sideCost 5 7 30 + sideCost 6 8 30 = 312 := by sorry

end NUMINAMATH_CALUDE_total_painting_cost_l1130_113064


namespace NUMINAMATH_CALUDE_distinct_remainders_l1130_113090

def sequence_a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2^(sequence_a n) + sequence_a n

theorem distinct_remainders (n m : ℕ) (hn : n < 243) (hm : m < 243) (hnm : n ≠ m) :
  sequence_a n % 243 ≠ sequence_a m % 243 := by
  sorry

end NUMINAMATH_CALUDE_distinct_remainders_l1130_113090


namespace NUMINAMATH_CALUDE_ed_remaining_money_l1130_113024

-- Define the hotel rates
def night_rate : ℝ := 1.50
def morning_rate : ℝ := 2

-- Define Ed's initial money
def initial_money : ℝ := 80

-- Define the duration of stay
def night_hours : ℝ := 6
def morning_hours : ℝ := 4

-- Theorem to prove
theorem ed_remaining_money :
  let night_cost := night_rate * night_hours
  let morning_cost := morning_rate * morning_hours
  let total_cost := night_cost + morning_cost
  let remaining_money := initial_money - total_cost
  remaining_money = 63 := by sorry

end NUMINAMATH_CALUDE_ed_remaining_money_l1130_113024


namespace NUMINAMATH_CALUDE_expression_equals_one_l1130_113003

theorem expression_equals_one : 
  (121^2 - 13^2) / (91^2 - 17^2) * ((91-17)*(91+17)) / ((121-13)*(121+13)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1130_113003


namespace NUMINAMATH_CALUDE_complex_calculation_l1130_113089

theorem complex_calculation (z : ℂ) (h : z = 1 + I) : z - 2 / z^2 = 1 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_l1130_113089


namespace NUMINAMATH_CALUDE_at_least_three_to_six_colorings_l1130_113020

/-- Represents the colors that can be used to color the hexagons -/
inductive Color
| Red
| Yellow
| Green
| Blue

/-- Represents a hexagon in the figure -/
structure Hexagon where
  color : Color

/-- Represents the central hexagon and its six adjacent hexagons -/
structure CentralHexagonWithAdjacent where
  center : Hexagon
  adjacent : Fin 6 → Hexagon

/-- Two hexagons are considered adjacent if they share a side -/
def areAdjacent (h1 h2 : Hexagon) : Prop := sorry

/-- A coloring is valid if no two adjacent hexagons have the same color -/
def isValidColoring (config : CentralHexagonWithAdjacent) : Prop :=
  config.center.color = Color.Red ∧
  ∀ i j : Fin 6, i ≠ j →
    config.adjacent i ≠ config.adjacent j ∧
    config.adjacent i ≠ config.center ∧
    config.adjacent j ≠ config.center

/-- The number of valid colorings for the central hexagon and its adjacent hexagons -/
def numValidColorings : ℕ := sorry

theorem at_least_three_to_six_colorings :
  numValidColorings ≥ 3^6 := by sorry

end NUMINAMATH_CALUDE_at_least_three_to_six_colorings_l1130_113020


namespace NUMINAMATH_CALUDE_two_cubic_feet_to_cubic_inches_l1130_113073

/-- Converts cubic feet to cubic inches -/
def cubic_feet_to_cubic_inches (cf : ℝ) : ℝ := cf * (12^3)

/-- Theorem stating that 2 cubic feet equals 3456 cubic inches -/
theorem two_cubic_feet_to_cubic_inches : 
  cubic_feet_to_cubic_inches 2 = 3456 := by
  sorry

end NUMINAMATH_CALUDE_two_cubic_feet_to_cubic_inches_l1130_113073


namespace NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_sin_24_l1130_113030

theorem cos_96_cos_24_minus_sin_96_sin_24 :
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.sin (24 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_sin_24_l1130_113030


namespace NUMINAMATH_CALUDE_max_bowls_proof_l1130_113077

/-- Represents the number of clusters in a spoonful for the nth bowl -/
def clusters_per_spoon (n : ℕ) : ℕ := 3 + n

/-- Represents the number of spoonfuls in the nth bowl -/
def spoonfuls_per_bowl (n : ℕ) : ℕ := 27 - 2 * n

/-- Calculates the total clusters used up to and including the nth bowl -/
def total_clusters (n : ℕ) : ℕ := 
  (List.range n).foldl (λ acc i => acc + clusters_per_spoon (i + 1) * spoonfuls_per_bowl (i + 1)) 0

/-- The maximum number of bowls that can be made from 500 clusters -/
def max_bowls : ℕ := 4

theorem max_bowls_proof : 
  total_clusters max_bowls ≤ 500 ∧ 
  total_clusters (max_bowls + 1) > 500 := by
  sorry

#eval max_bowls

end NUMINAMATH_CALUDE_max_bowls_proof_l1130_113077


namespace NUMINAMATH_CALUDE_ring_toss_total_l1130_113079

/-- Calculates the total number of rings used in a ring toss game -/
def total_rings (rings_per_game : ℕ) (games_played : ℕ) : ℕ :=
  rings_per_game * games_played

/-- Theorem: Given 6 rings per game and 8 games played, the total rings used is 48 -/
theorem ring_toss_total :
  total_rings 6 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_total_l1130_113079


namespace NUMINAMATH_CALUDE_right_triangle_PR_length_l1130_113092

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  PR : ℝ
  sinR : ℝ
  angle_Q_is_right : True  -- Represents ∠Q = 90°

-- State the theorem
theorem right_triangle_PR_length 
  (triangle : RightTriangle) 
  (h1 : triangle.PQ = 9) 
  (h2 : triangle.sinR = 3/5) : 
  triangle.PR = 15 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_PR_length_l1130_113092


namespace NUMINAMATH_CALUDE_inverse_g_at_113_l1130_113045

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_g_at_113 : g⁻¹ 113 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_at_113_l1130_113045


namespace NUMINAMATH_CALUDE_cost_of_pencils_l1130_113041

/-- Given that 100 pencils cost $30, prove that 1500 pencils cost $450. -/
theorem cost_of_pencils :
  (∃ (cost_per_100 : ℝ), cost_per_100 = 30 ∧ 
   (1500 / 100) * cost_per_100 = 450) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_pencils_l1130_113041


namespace NUMINAMATH_CALUDE_rotation_maps_points_l1130_113028

-- Define points in R²
def C : ℝ × ℝ := (3, -2)
def C' : ℝ × ℝ := (-3, 2)
def D : ℝ × ℝ := (4, -5)
def D' : ℝ × ℝ := (-4, 5)

-- Define rotation by 180°
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement
theorem rotation_maps_points :
  rotate180 C = C' ∧ rotate180 D = D' :=
sorry

end NUMINAMATH_CALUDE_rotation_maps_points_l1130_113028


namespace NUMINAMATH_CALUDE_power_difference_equality_l1130_113066

theorem power_difference_equality : (3^4)^4 - (4^3)^3 = 42792577 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l1130_113066


namespace NUMINAMATH_CALUDE_wall_width_is_0_05_meters_l1130_113019

-- Define the brick dimensions in meters
def brick_length : Real := 0.21
def brick_width : Real := 0.10
def brick_height : Real := 0.08

-- Define the wall dimensions
def wall_length : Real := 9
def wall_height : Real := 18.5

-- Define the number of bricks
def num_bricks : Real := 4955.357142857142

-- Theorem to prove
theorem wall_width_is_0_05_meters :
  let brick_volume := brick_length * brick_width * brick_height
  let total_brick_volume := brick_volume * num_bricks
  let wall_width := total_brick_volume / (wall_length * wall_height)
  wall_width = 0.05 := by sorry

end NUMINAMATH_CALUDE_wall_width_is_0_05_meters_l1130_113019


namespace NUMINAMATH_CALUDE_log_sqrt8_512sqrt8_equals_7_l1130_113068

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sqrt8_512sqrt8_equals_7 :
  log (Real.sqrt 8) (512 * Real.sqrt 8) = 7 := by sorry

end NUMINAMATH_CALUDE_log_sqrt8_512sqrt8_equals_7_l1130_113068


namespace NUMINAMATH_CALUDE_max_k_value_l1130_113046

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 4 = k^2 * (x^2/y^2 + 2 + y^2/x^2) + k^3 * (x/y + y/x)) :
  k ≤ 4 * Real.sqrt 2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l1130_113046


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1130_113010

theorem sqrt_product_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (8 * p) * Real.sqrt (12 * p^5) = 60 * p^4 * Real.sqrt (2 * p) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1130_113010


namespace NUMINAMATH_CALUDE_triangle_properties_l1130_113048

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a + t.c = t.b * (Real.cos t.C + Real.sqrt 3 * Real.sin t.C))
  (h2 : t.b = 2) : 
  t.B = π / 3 ∧ 
  ∀ (s : Triangle), s.b = 2 → 
    Real.sqrt 3 / 4 * s.a * s.c * Real.sin s.B ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1130_113048


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1130_113039

/-- The perimeter of a rhombus with diagonals of lengths 72 and 30 is 156 -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 72) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 156 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1130_113039


namespace NUMINAMATH_CALUDE_average_height_theorem_l1130_113097

def tree_heights (h₁ h₂ h₃ h₄ h₅ : ℝ) : Prop :=
  h₂ = 15 ∧
  (h₂ = h₁ + 5 ∨ h₂ = h₁ - 3) ∧
  (h₃ = h₂ + 5 ∨ h₃ = h₂ - 3) ∧
  (h₄ = h₃ + 5 ∨ h₄ = h₃ - 3) ∧
  (h₅ = h₄ + 5 ∨ h₅ = h₄ - 3)

theorem average_height_theorem (h₁ h₂ h₃ h₄ h₅ : ℝ) :
  tree_heights h₁ h₂ h₃ h₄ h₅ →
  ∃ (k : ℤ), (h₁ + h₂ + h₃ + h₄ + h₅) / 5 = k + 0.4 →
  (h₁ + h₂ + h₃ + h₄ + h₅) / 5 = 20.4 :=
by sorry

end NUMINAMATH_CALUDE_average_height_theorem_l1130_113097


namespace NUMINAMATH_CALUDE_roller_plate_acceleration_l1130_113043

noncomputable def plate_acceleration (R r : ℝ) (m : ℝ) (α : ℝ) (g : ℝ) : ℝ :=
  g * Real.sqrt ((1 - Real.cos α) / 2)

noncomputable def plate_direction (α : ℝ) : ℝ :=
  Real.arcsin (Real.sqrt ((1 - Real.cos α) / 2))

theorem roller_plate_acceleration 
  (R : ℝ) 
  (r : ℝ) 
  (m : ℝ) 
  (α : ℝ) 
  (g : ℝ) 
  (h_R : R = 1) 
  (h_r : r = 0.4) 
  (h_m : m = 150) 
  (h_α : α = Real.arccos 0.68) 
  (h_g : g = 10) :
  plate_acceleration R r m α g = 4 ∧ 
  plate_direction α = Real.arcsin 0.4 ∧
  plate_acceleration R r m α g = g * Real.sin (α / 2) :=
by
  sorry

#check roller_plate_acceleration

end NUMINAMATH_CALUDE_roller_plate_acceleration_l1130_113043


namespace NUMINAMATH_CALUDE_student_count_problem_l1130_113072

theorem student_count_problem : 
  ∃ n : ℕ, n > 1 ∧ 
  (n - 1) % 2 = 1 ∧ 
  (n - 1) % 7 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m < n → (m - 1) % 2 ≠ 1 ∨ (m - 1) % 7 ≠ 1) ∧
  n = 44 := by
sorry

end NUMINAMATH_CALUDE_student_count_problem_l1130_113072


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l1130_113091

/-- Given a rectangular box with dimensions a, b, and c, if the sum of the lengths of its twelve edges
    is 156 and the distance from one corner to the farthest corner is 25, then its total surface area is 896. -/
theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : 4 * a + 4 * b + 4 * c = 156)
  (diagonal : Real.sqrt (a^2 + b^2 + c^2) = 25) :
  2 * (a * b + b * c + c * a) = 896 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l1130_113091


namespace NUMINAMATH_CALUDE_expression_value_l1130_113035

theorem expression_value : -20 + 8 * (5^2 - 3) = 156 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1130_113035


namespace NUMINAMATH_CALUDE_right_triangle_roots_l1130_113061

-- Define the equation
def equation (m x : ℝ) : Prop := x^2 - (2*m + 1)*x + m^2 + m = 0

-- Define the roots
def roots (m : ℝ) : Set ℝ := {x | equation m x}

theorem right_triangle_roots (m : ℝ) :
  let a := (2*m + 1 + 1) / 2
  let b := (2*m + 1 - 1) / 2
  (∀ x ∈ roots m, x = a ∨ x = b) →
  a^2 + b^2 = 5^2 →
  m = 3 := by sorry

end NUMINAMATH_CALUDE_right_triangle_roots_l1130_113061


namespace NUMINAMATH_CALUDE_cos_A_right_triangle_l1130_113060

theorem cos_A_right_triangle (adjacent hypotenuse : ℝ) 
  (h1 : adjacent = 5)
  (h2 : hypotenuse = 13)
  (h3 : adjacent > 0)
  (h4 : hypotenuse > 0)
  (h5 : adjacent < hypotenuse) : 
  Real.cos (Real.arccos (adjacent / hypotenuse)) = 5 / 13 := by
sorry

end NUMINAMATH_CALUDE_cos_A_right_triangle_l1130_113060


namespace NUMINAMATH_CALUDE_parts_from_64_blanks_l1130_113001

/-- Calculates the total number of parts that can be produced from a given number of initial blanks,
    where shavings from a certain number of parts can be remelted into one new blank. -/
def total_parts (initial_blanks : ℕ) (parts_per_remelted_blank : ℕ) : ℕ :=
  let first_batch := initial_blanks
  let second_batch := initial_blanks / parts_per_remelted_blank
  let third_batch := second_batch / parts_per_remelted_blank
  first_batch + second_batch + third_batch

/-- Theorem stating that given 64 initial blanks and the ability to remelt shavings 
    from 8 parts into one new blank, the total number of parts that can be produced is 73. -/
theorem parts_from_64_blanks : total_parts 64 8 = 73 := by
  sorry

end NUMINAMATH_CALUDE_parts_from_64_blanks_l1130_113001


namespace NUMINAMATH_CALUDE_inverse_not_in_M_exponential_in_M_logarithmic_in_M_l1130_113070

-- Define set M
def M (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

-- Problem 1
theorem inverse_not_in_M :
  ¬ M (fun x => 1 / x) := by sorry

-- Problem 2
theorem exponential_in_M (k b : ℝ) :
  M (fun x => k * 2^x + b) ↔ (k = 0 ∧ b = 0) ∨ (k ≠ 0 ∧ (2 * k + b) / k > 0) := by sorry

-- Problem 3
theorem logarithmic_in_M :
  ∀ a : ℝ, M (fun x => Real.log (a / (x^2 + 2))) ↔ 
    (a ≥ 3/2 ∧ a ≤ 6 ∧ a ≠ 3) := by sorry

end NUMINAMATH_CALUDE_inverse_not_in_M_exponential_in_M_logarithmic_in_M_l1130_113070


namespace NUMINAMATH_CALUDE_complex_equality_l1130_113009

/-- Given a real number b, if the real part is equal to the imaginary part
    for the complex number (1+i)/(1-i) + (1/2)b, then b = 2 -/
theorem complex_equality (b : ℝ) : 
  (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I) + (1 / 2 : ℂ) * b).re = 
  (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I) + (1 / 2 : ℂ) * b).im → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l1130_113009


namespace NUMINAMATH_CALUDE_pool_filling_buckets_l1130_113029

theorem pool_filling_buckets 
  (george_buckets : ℕ) 
  (harry_buckets : ℕ) 
  (total_rounds : ℕ) :
  george_buckets = 2 →
  harry_buckets = 3 →
  total_rounds = 22 →
  (george_buckets + harry_buckets) * total_rounds = 110 := by
sorry

end NUMINAMATH_CALUDE_pool_filling_buckets_l1130_113029


namespace NUMINAMATH_CALUDE_max_min_values_on_interval_l1130_113078

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def additive (f : ℝ → ℝ) : Prop := ∀ x y, f (x + y) = f x + f y

theorem max_min_values_on_interval
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_additive : additive f)
  (h_neg : ∀ x > 0, f x < 0)
  (h_f1 : f 1 = -2) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ 6) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ -6) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 6) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = -6) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_on_interval_l1130_113078


namespace NUMINAMATH_CALUDE_circles_externally_tangent_m_value_l1130_113075

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

def circle_C2 (x y m : ℝ) : Prop := x^2 + y^2 - 8*x - 10*y + m + 6 = 0

-- Define external tangency
def externally_tangent (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧
  ∀ (x' y' : ℝ), (C1 x' y' ∧ C2 x' y') → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_externally_tangent_m_value :
  externally_tangent circle_C1 (circle_C2 · · 26) :=
sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_m_value_l1130_113075


namespace NUMINAMATH_CALUDE_males_count_l1130_113056

/-- Represents the population of a village -/
structure Village where
  total_population : ℕ
  num_groups : ℕ
  males_in_one_group : Bool

/-- Theorem: In a village with 520 people divided into 4 equal groups,
    if one group represents all males, then the number of males is 130 -/
theorem males_count (v : Village)
  (h1 : v.total_population = 520)
  (h2 : v.num_groups = 4)
  (h3 : v.males_in_one_group = true) :
  v.total_population / v.num_groups = 130 := by
  sorry

#check males_count

end NUMINAMATH_CALUDE_males_count_l1130_113056


namespace NUMINAMATH_CALUDE_office_supplies_cost_l1130_113033

def pencil_cost : ℝ := 0.5
def folder_cost : ℝ := 0.9
def pencil_quantity : ℕ := 24  -- two dozen
def folder_quantity : ℕ := 20

def total_cost : ℝ := pencil_cost * pencil_quantity + folder_cost * folder_quantity

theorem office_supplies_cost : total_cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_office_supplies_cost_l1130_113033


namespace NUMINAMATH_CALUDE_homework_time_decrease_l1130_113011

theorem homework_time_decrease (x : ℝ) : 
  (∀ t : ℝ, t > 0 → (t * (1 - x))^2 = t * (1 - x)^2) →
  100 * (1 - x)^2 = 70 :=
by sorry

end NUMINAMATH_CALUDE_homework_time_decrease_l1130_113011


namespace NUMINAMATH_CALUDE_triangle_area_condition_l1130_113059

/-- The area of the triangle formed by the line x - 2y + 2m = 0 and the coordinate axes is not less than 1 if and only if m ∈ (-∞, -1] ∪ [1, +∞) -/
theorem triangle_area_condition (m : ℝ) : 
  (∃ (x y : ℝ), x - 2*y + 2*m = 0 ∧ 
   (1/2) * |x| * |y| ≥ 1) ↔ 
  (m ≤ -1 ∨ m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_condition_l1130_113059


namespace NUMINAMATH_CALUDE_smallest_square_area_l1130_113026

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum side length of a square that can contain two rectangles -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  min (max r1.width r2.width + min r1.height r2.height)
      (max r1.height r2.height + min r1.width r2.width)

/-- The theorem stating the smallest possible area of the square -/
theorem smallest_square_area (r1 r2 : Rectangle)
  (h1 : r1 = ⟨3, 5⟩)
  (h2 : r2 = ⟨4, 6⟩) :
  (minSquareSide r1 r2) ^ 2 = 81 := by
  sorry

#eval (minSquareSide ⟨3, 5⟩ ⟨4, 6⟩) ^ 2

end NUMINAMATH_CALUDE_smallest_square_area_l1130_113026


namespace NUMINAMATH_CALUDE_monica_reading_plan_l1130_113099

def books_last_year : ℕ := 16

def books_this_year : ℕ := 2 * books_last_year

def books_next_year : ℕ := 2 * books_this_year + 5

theorem monica_reading_plan : books_next_year = 69 := by
  sorry

end NUMINAMATH_CALUDE_monica_reading_plan_l1130_113099


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l1130_113027

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication 
  (m n : Line) (a : Plane) : 
  parallel m n → perpendicular m a → perpendicular n a :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l1130_113027


namespace NUMINAMATH_CALUDE_simplify_expression_l1130_113093

theorem simplify_expression (r : ℝ) : 120 * r - 68 * r + 15 * r = 67 * r := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1130_113093


namespace NUMINAMATH_CALUDE_factorization_sum_l1130_113012

theorem factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 9*x + 14 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 4*x - 21 = (x + b)*(x - c)) →
  a + b + c = 12 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l1130_113012


namespace NUMINAMATH_CALUDE_total_tickets_sold_l1130_113053

theorem total_tickets_sold (adult_tickets student_tickets : ℕ) 
  (h1 : adult_tickets = 410)
  (h2 : student_tickets = 436) :
  adult_tickets + student_tickets = 846 := by
  sorry

#check total_tickets_sold

end NUMINAMATH_CALUDE_total_tickets_sold_l1130_113053


namespace NUMINAMATH_CALUDE_inequality_proof_l1130_113055

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (a + 1/b)^2 + (b + 1/a)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1130_113055


namespace NUMINAMATH_CALUDE_sum_cos_fractions_24_pi_zero_l1130_113085

def simplest_proper_fractions_24 : List ℚ := [
  1/24, 5/24, 7/24, 11/24, 13/24, 17/24, 19/24, 23/24
]

theorem sum_cos_fractions_24_pi_zero : 
  (simplest_proper_fractions_24.map (fun x => Real.cos (x * Real.pi))).sum = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_cos_fractions_24_pi_zero_l1130_113085


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l1130_113052

theorem normal_distribution_std_dev (μ σ x : ℝ) (hμ : μ = 17.5) (hσ : σ = 2.5) (hx : x = 12.5) :
  (x - μ) / σ = -2 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l1130_113052


namespace NUMINAMATH_CALUDE_sara_earnings_l1130_113014

/-- Sara's cake-making and selling scenario --/
def sara_cake_scenario (weekdays_per_week : ℕ) (cakes_per_day : ℕ) (price_per_cake : ℕ) (num_weeks : ℕ) : ℕ :=
  weekdays_per_week * cakes_per_day * price_per_cake * num_weeks

/-- Theorem: Sara's earnings over 4 weeks --/
theorem sara_earnings : sara_cake_scenario 5 4 8 4 = 640 := by
  sorry

end NUMINAMATH_CALUDE_sara_earnings_l1130_113014


namespace NUMINAMATH_CALUDE_continuous_function_satisfying_integral_equation_is_constant_l1130_113050

/-- A continuous function satisfying the given integral equation is constant -/
theorem continuous_function_satisfying_integral_equation_is_constant 
  (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ a b : ℝ, (a^2 + a*b + b^2) * ∫ x in a..b, f x = 3 * ∫ x in a..b, x^2 * f x) : 
  ∃ C : ℝ, ∀ x : ℝ, f x = C := by
sorry

end NUMINAMATH_CALUDE_continuous_function_satisfying_integral_equation_is_constant_l1130_113050


namespace NUMINAMATH_CALUDE_yellow_bows_count_l1130_113098

theorem yellow_bows_count (total : ℚ) :
  (1 / 6 : ℚ) * total +  -- yellow bows
  (1 / 3 : ℚ) * total +  -- purple bows
  (1 / 8 : ℚ) * total +  -- orange bows
  40 = total →           -- black bows
  (1 / 6 : ℚ) * total = 160 / 9 := by
sorry

end NUMINAMATH_CALUDE_yellow_bows_count_l1130_113098
