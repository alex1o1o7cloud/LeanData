import Mathlib

namespace NUMINAMATH_CALUDE_coffee_shop_spending_prove_coffee_shop_spending_l3916_391672

theorem coffee_shop_spending : ℝ → ℝ → Prop :=
  fun (ben_spent david_spent : ℝ) =>
    (david_spent = ben_spent / 2) →
    (ben_spent = david_spent + 15) →
    (ben_spent + david_spent = 45)

/-- Proof of the coffee shop spending theorem -/
theorem prove_coffee_shop_spending :
  ∃ (ben_spent david_spent : ℝ),
    coffee_shop_spending ben_spent david_spent :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_spending_prove_coffee_shop_spending_l3916_391672


namespace NUMINAMATH_CALUDE_upstream_travel_time_l3916_391695

theorem upstream_travel_time
  (distance : ℝ)
  (downstream_time : ℝ)
  (current_speed : ℝ)
  (h1 : distance = 126)
  (h2 : downstream_time = 7)
  (h3 : current_speed = 2)
  : (distance / (distance / downstream_time - 2 * current_speed)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_upstream_travel_time_l3916_391695


namespace NUMINAMATH_CALUDE_fraction_simplification_l3916_391687

theorem fraction_simplification (a b d : ℝ) (h : a^2 + d^2 - b^2 + 2*a*d ≠ 0) :
  (a^2 + b^2 + d^2 + 2*b*d) / (a^2 + d^2 - b^2 + 2*a*d) = 
  (a^2 + (b+d)^2) / ((a+d)^2 + a^2 - b^2) := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3916_391687


namespace NUMINAMATH_CALUDE_food_fraction_proof_l3916_391678

def initial_amount : ℝ := 499.9999999999999

theorem food_fraction_proof (clothes_fraction : ℝ) (travel_fraction : ℝ) (food_fraction : ℝ) 
  (h1 : clothes_fraction = 1/3)
  (h2 : travel_fraction = 1/4)
  (h3 : initial_amount * (1 - clothes_fraction) * (1 - food_fraction) * (1 - travel_fraction) = 200) :
  food_fraction = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_food_fraction_proof_l3916_391678


namespace NUMINAMATH_CALUDE_athletes_with_four_points_after_seven_rounds_l3916_391681

/-- The number of athletes with k points after m rounds in a tournament of 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n-m) * (m.choose k)

/-- The total number of athletes with 4 points after 7 rounds in a tournament of 2^n + 6 participants -/
def athletes_with_four_points (n : ℕ) : ℕ := 35 * 2^(n-7) + 2

theorem athletes_with_four_points_after_seven_rounds (n : ℕ) (h : n > 7) :
  athletes_with_four_points n = f n 7 4 + 2 :=
sorry

#check athletes_with_four_points_after_seven_rounds

end NUMINAMATH_CALUDE_athletes_with_four_points_after_seven_rounds_l3916_391681


namespace NUMINAMATH_CALUDE_another_root_of_p_l3916_391643

-- Define the polynomials p and q
def p (a b : ℤ) (x : ℂ) : ℂ := x^3 + a*x^2 + b*x - 1
def q (c d : ℤ) (x : ℂ) : ℂ := x^3 + c*x^2 + d*x + 1

-- State the theorem
theorem another_root_of_p (a b c d : ℤ) (α : ℂ) :
  (∃ (a b : ℤ), p a b α = 0) →  -- α is a root of p(x) = 0
  (∀ (r : ℚ), p a b r ≠ 0) →  -- p(x) is irreducible over the rationals
  (∃ (c d : ℤ), q c d (α + 1) = 0) →  -- α + 1 is a root of q(x) = 0
  (∃ β : ℂ, p a b β = 0 ∧ (β = -1/(α+1) ∨ β = -(α+1)/α)) :=
by sorry

end NUMINAMATH_CALUDE_another_root_of_p_l3916_391643


namespace NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l3916_391648

theorem tripled_base_doubled_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (3 * a) ^ (2 * b) = a ^ (2 * b) * y ^ b → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l3916_391648


namespace NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l3916_391673

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to check if points are coplanar
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define a function to count the number of unique planes determined by four points
def countPlanesFromPoints (p1 p2 p3 p4 : Point3D) : ℕ := sorry

-- Theorem statement
theorem four_noncoplanar_points_determine_four_planes 
  (p1 p2 p3 p4 : Point3D) 
  (h : ¬ areCoplanar p1 p2 p3 p4) : 
  countPlanesFromPoints p1 p2 p3 p4 = 4 := by sorry

end NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l3916_391673


namespace NUMINAMATH_CALUDE_fairy_tale_book_weighs_1_1_kg_l3916_391639

/-- The weight of the fairy tale book in kilograms -/
def fairy_tale_book_weight : ℝ := sorry

/-- The total weight on the other side of the scale in kilograms -/
def other_side_weight : ℝ := 0.5 + 0.3 + 0.3

/-- The scale is level, so the weights on both sides are equal -/
axiom scale_balance : fairy_tale_book_weight = other_side_weight

/-- Theorem: The fairy tale book weighs 1.1 kg -/
theorem fairy_tale_book_weighs_1_1_kg : fairy_tale_book_weight = 1.1 := by sorry

end NUMINAMATH_CALUDE_fairy_tale_book_weighs_1_1_kg_l3916_391639


namespace NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l3916_391629

theorem evaluate_sqrt_fraction (y : ℝ) (h : y < -2) :
  Real.sqrt (y / (1 - (y + 1) / (y + 2))) = -y := by
  sorry

end NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l3916_391629


namespace NUMINAMATH_CALUDE_value_of_X_l3916_391646

theorem value_of_X : ∃ X : ℚ, (1/4 : ℚ) * (1/8 : ℚ) * X = (1/2 : ℚ) * (1/6 : ℚ) * 120 ∧ X = 320 := by
  sorry

end NUMINAMATH_CALUDE_value_of_X_l3916_391646


namespace NUMINAMATH_CALUDE_moles_NaHCO3_equals_moles_HCl_l3916_391602

/-- Represents a chemical species in the reaction -/
inductive Species
| NaHCO3
| HCl
| NaCl
| H2O
| CO2

/-- Represents the balanced chemical equation -/
def balanced_equation (reactants products : Species → ℕ) : Prop :=
  reactants Species.NaHCO3 = 1 ∧
  reactants Species.HCl = 1 ∧
  products Species.NaCl = 1 ∧
  products Species.H2O = 1 ∧
  products Species.CO2 = 1

/-- The number of moles of HCl given -/
def moles_HCl : ℕ := 3

/-- The number of moles of products formed -/
def moles_products : Species → ℕ
| Species.NaCl => 3
| Species.H2O => 3
| Species.CO2 => 3
| _ => 0

/-- Theorem stating that the number of moles of NaHCO3 required equals the number of moles of HCl -/
theorem moles_NaHCO3_equals_moles_HCl 
  (eq : balanced_equation (λ _ => 1) (λ _ => 1))
  (prod : ∀ s, moles_products s = moles_HCl ∨ moles_products s = 0) :
  moles_HCl = moles_HCl := by sorry

end NUMINAMATH_CALUDE_moles_NaHCO3_equals_moles_HCl_l3916_391602


namespace NUMINAMATH_CALUDE_triangle_side_difference_minimum_l3916_391651

theorem triangle_side_difference_minimum (x : ℝ) : 
  (5/3 < x) →
  (x < 11/3) →
  (x + 6 + (4*x - 1) > x + 10) →
  (x + 6 + (x + 10) > 4*x - 1) →
  ((4*x - 1) + (x + 10) > x + 6) →
  (x + 10 > x + 6) →
  (x + 10 > 4*x - 1) →
  (x + 10) - (x + 6) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_minimum_l3916_391651


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l3916_391688

/-- Represents the number of female students in a stratified sample -/
def female_in_sample (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ) : ℚ :=
  (female_students : ℚ) * (sample_size : ℚ) / (total_students : ℚ)

/-- Theorem: In a school with 2100 total students (900 female),
    a stratified sample of 70 students will contain 30 female students -/
theorem stratified_sample_theorem :
  female_in_sample 2100 900 70 = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l3916_391688


namespace NUMINAMATH_CALUDE_ellipse_condition_equiv_k_range_ellipse_standard_equation_l3916_391675

-- Define the curve C
def curve_C (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (4 - k) - y^2 / (1 - k) = 1

-- Define the condition for an ellipse with foci on the x-axis
def is_ellipse_x_axis (k : ℝ) : Prop :=
  4 - k > 0 ∧ k - 1 > 0 ∧ 4 - k > k - 1

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  1 < k ∧ k < 5/2

-- Define the ellipse passing through (√6, 2) with foci at (-2,0) and (2,0)
def ellipse_through_point (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 8 = 1

-- Theorem 1: Equivalence of ellipse condition and k range
theorem ellipse_condition_equiv_k_range (k : ℝ) :
  is_ellipse_x_axis k ↔ k_range k :=
sorry

-- Theorem 2: Standard equation of the ellipse
theorem ellipse_standard_equation :
  ellipse_through_point (Real.sqrt 6) 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_equiv_k_range_ellipse_standard_equation_l3916_391675


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l3916_391654

theorem min_value_expression (x y : ℝ) : 
  (x * y - 1/2)^2 + (x - y)^2 ≥ 1/4 :=
sorry

theorem min_value_attainable : 
  ∃ x y : ℝ, (x * y - 1/2)^2 + (x - y)^2 = 1/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l3916_391654


namespace NUMINAMATH_CALUDE_flooring_boxes_needed_l3916_391605

def room_length : ℝ := 16
def room_width : ℝ := 20
def flooring_per_box : ℝ := 10
def flooring_laid : ℝ := 250

theorem flooring_boxes_needed : 
  ⌈(room_length * room_width - flooring_laid) / flooring_per_box⌉ = 7 := by
  sorry

end NUMINAMATH_CALUDE_flooring_boxes_needed_l3916_391605


namespace NUMINAMATH_CALUDE_largest_number_with_equal_costs_l3916_391627

/-- Calculates the sum of squares of decimal digits for a given number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ := sorry

/-- Calculates the number of 1's in the binary representation of a given number -/
def count_ones_in_binary (n : ℕ) : ℕ := sorry

/-- Theorem stating that 503 is the largest number less than 2000 where 
    sum of squares of digits equals the number of 1's in binary representation -/
theorem largest_number_with_equal_costs : 
  ∀ n : ℕ, n < 2000 → n > 503 → 
    sum_of_squares_of_digits n ≠ count_ones_in_binary n := by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_equal_costs_l3916_391627


namespace NUMINAMATH_CALUDE_M_equals_N_l3916_391611

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l3916_391611


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l3916_391609

/-- 
Given a convex pentagon with interior angles measuring y, 2y+2, 3y-3, 4y+4, and 5y-5 degrees,
where the sum of these angles is 540 degrees, prove that the largest angle measures 176 degrees
when rounded to the nearest integer.
-/
theorem pentagon_largest_angle : 
  ∀ y : ℝ, 
  y + (2*y+2) + (3*y-3) + (4*y+4) + (5*y-5) = 540 → 
  round (5*y - 5) = 176 := by
sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l3916_391609


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3916_391667

theorem arithmetic_expression_equality : (24 / (8 + 2 - 6)) * 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3916_391667


namespace NUMINAMATH_CALUDE_geometric_progression_solutions_l3916_391619

theorem geometric_progression_solutions : 
  ∃ (x₁ x₂ a₁ a₂ : ℝ), 
    (x₁ = 2 ∧ a₁ = 3 ∧ 3 * |x₁| * Real.sqrt (x₁ + 2) = 5 * x₁ + 2) ∧
    (x₂ = -2/9 ∧ a₂ = 1/2 ∧ 3 * |x₂| * Real.sqrt (x₂ + 2) = 5 * x₂ + 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solutions_l3916_391619


namespace NUMINAMATH_CALUDE_abs_5x_minus_2_zero_l3916_391606

theorem abs_5x_minus_2_zero (x : ℚ) : |5*x - 2| = 0 ↔ x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_abs_5x_minus_2_zero_l3916_391606


namespace NUMINAMATH_CALUDE_f_derivative_at_2_when_a_0_f_minimum_at_0_iff_a_lt_2_g_not_tangent_to_line_with_slope_3_2_l3916_391657

noncomputable section

open Real

/-- The base of the natural logarithm -/
def e : ℝ := exp 1

/-- The function f(x) = (x^2 + ax + a)e^(-x) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x + a) * (e^(-x))

/-- The function g(x) = (4 - x)e^(x - 2) for x < 2 -/
def g (x : ℝ) : ℝ := (4 - x) * (e^(x - 2))

theorem f_derivative_at_2_when_a_0 :
  (deriv (f 0)) 2 = 0 := by sorry

theorem f_minimum_at_0_iff_a_lt_2 (a : ℝ) :
  (∀ x, f a 0 ≤ f a x) ↔ a < 2 := by sorry

theorem g_not_tangent_to_line_with_slope_3_2 :
  ¬ ∃ (c : ℝ), ∃ (x : ℝ), x < 2 ∧ g x = (3/2) * x + c ∧ (deriv g) x = 3/2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_when_a_0_f_minimum_at_0_iff_a_lt_2_g_not_tangent_to_line_with_slope_3_2_l3916_391657


namespace NUMINAMATH_CALUDE_park_shape_l3916_391680

theorem park_shape (total_cost : ℕ) (cost_per_side : ℕ) (h1 : total_cost = 224) (h2 : cost_per_side = 56) :
  total_cost / cost_per_side = 4 :=
by sorry

end NUMINAMATH_CALUDE_park_shape_l3916_391680


namespace NUMINAMATH_CALUDE_random_walk_properties_l3916_391618

/-- Represents a random walk on a line. -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of the random walk. -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk. -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences that achieve the maximum range. -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Theorem stating the properties of the random walk. -/
theorem random_walk_properties (w : RandomWalk) :
  (max_range w = w.a) ∧
  (min_range w = w.a - w.b) ∧
  (max_range_sequences w = w.b + 1) := by sorry

end NUMINAMATH_CALUDE_random_walk_properties_l3916_391618


namespace NUMINAMATH_CALUDE_M_remainder_l3916_391665

/-- The number of positive integers less than or equal to 2010 whose base-2 representation has more 1's than 0's -/
def M : ℕ := 1162

/-- 2010 is less than 2^11 - 1 -/
axiom h1 : 2010 < 2^11 - 1

/-- The sum of binary numbers where the number of 1's is more than 0's up to 2^11 - 1 -/
def total_sum : ℕ := 2^11 - 1

/-- The number of binary numbers more than 2010 and ≤ 2047 -/
def excess : ℕ := 37

/-- The sum of center elements in Pascal's Triangle rows 0 to 5 -/
def center_sum : ℕ := 351

theorem M_remainder (h2 : M = (total_sum + center_sum) / 2 - excess) :
  M % 1000 = 162 := by sorry

end NUMINAMATH_CALUDE_M_remainder_l3916_391665


namespace NUMINAMATH_CALUDE_caitlin_age_l3916_391656

theorem caitlin_age (anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ)
  (h1 : anna_age = 42)
  (h2 : brianna_age = anna_age / 2)
  (h3 : caitlin_age = brianna_age - 5) :
  caitlin_age = 16 := by
sorry

end NUMINAMATH_CALUDE_caitlin_age_l3916_391656


namespace NUMINAMATH_CALUDE_bob_picked_450_apples_l3916_391612

/-- The number of apples Bob picked for his family -/
def apples_picked (num_children : ℕ) (apples_per_child : ℕ) (num_adults : ℕ) (apples_per_adult : ℕ) : ℕ :=
  num_children * apples_per_child + num_adults * apples_per_adult

/-- Theorem stating that Bob picked 450 apples for his family -/
theorem bob_picked_450_apples : 
  apples_picked 33 10 40 3 = 450 := by
  sorry

end NUMINAMATH_CALUDE_bob_picked_450_apples_l3916_391612


namespace NUMINAMATH_CALUDE_doubling_points_theorem_l3916_391642

/-- Definition of a "doubling point" -/
def is_doubling_point (P Q : ℝ × ℝ) : Prop :=
  2 * (P.1 + Q.1) = P.2 + Q.2

/-- The point P₁ -/
def P₁ : ℝ × ℝ := (1, 0)

/-- Q₁ and Q₂ are specified points -/
def Q₁ : ℝ × ℝ := (3, 8)
def Q₂ : ℝ × ℝ := (-2, -2)

/-- The parabola y = x² - 2x - 3 -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem doubling_points_theorem :
  (is_doubling_point P₁ Q₁) ∧
  (is_doubling_point P₁ Q₂) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    is_doubling_point P₁ (x₁, parabola x₁) ∧
    is_doubling_point P₁ (x₂, parabola x₂)) ∧
  (∀ (Q : ℝ × ℝ), is_doubling_point P₁ Q → 
    Real.sqrt ((Q.1 - P₁.1)^2 + (Q.2 - P₁.2)^2) ≥ 4 * Real.sqrt 5 / 5) ∧
  (∃ (Q : ℝ × ℝ), is_doubling_point P₁ Q ∧
    Real.sqrt ((Q.1 - P₁.1)^2 + (Q.2 - P₁.2)^2) = 4 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_doubling_points_theorem_l3916_391642


namespace NUMINAMATH_CALUDE_similar_triangle_shorter_sides_sum_l3916_391601

theorem similar_triangle_shorter_sides_sum (a b c : ℝ) (k : ℝ) :
  a = 8 ∧ b = 10 ∧ c = 12 →
  k * (a + b + c) = 180 →
  k * a + k * b = 108 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_shorter_sides_sum_l3916_391601


namespace NUMINAMATH_CALUDE_coloring_books_total_l3916_391696

theorem coloring_books_total (initial : ℕ) (given_away : ℕ) (bought : ℕ) : 
  initial = 45 → given_away = 6 → bought = 20 → 
  initial - given_away + bought = 59 := by
sorry

end NUMINAMATH_CALUDE_coloring_books_total_l3916_391696


namespace NUMINAMATH_CALUDE_smallest_n_for_logarithm_sum_l3916_391636

theorem smallest_n_for_logarithm_sum : ∃ (n : ℕ), n = 3 ∧ 
  (∀ m : ℕ, m < n → 2^(2^(m+1)) < 512) ∧ 
  2^(2^(n+1)) ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_logarithm_sum_l3916_391636


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3916_391664

theorem simplify_square_roots : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3916_391664


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3916_391679

/-- Represents a T-shaped structure made of unit cubes -/
structure TCube where
  base : Nat
  stack : Nat

/-- Calculates the volume of the T-shaped structure -/
def volume (t : TCube) : Nat :=
  t.base + t.stack

/-- Calculates the surface area of the T-shaped structure -/
def surfaceArea (t : TCube) : Nat :=
  2 * (5 + 3) + 1 + 3 * 5

/-- The specific T-shaped structure described in the problem -/
def specificT : TCube :=
  { base := 4, stack := 4 }

theorem volume_to_surface_area_ratio :
  (volume specificT : ℚ) / (surfaceArea specificT : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3916_391679


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l3916_391638

theorem quadratic_inequality_always_true (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - a * x + 1 > 0) ↔ (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l3916_391638


namespace NUMINAMATH_CALUDE_floor_times_self_162_l3916_391674

theorem floor_times_self_162 (x : ℝ) : ⌊x⌋ * x = 162 → x = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_162_l3916_391674


namespace NUMINAMATH_CALUDE_prob_both_red_given_one_red_l3916_391653

/-- Represents a card with two sides -/
structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

/-- Represents the box of cards -/
def box : List Card := [
  {side1 := false, side2 := false},
  {side1 := false, side2 := false},
  {side1 := false, side2 := false},
  {side1 := false, side2 := false},
  {side1 := true,  side2 := false},
  {side1 := true,  side2 := false},
  {side1 := true,  side2 := true},
  {side1 := true,  side2 := true},
  {side1 := true,  side2 := true}
]

/-- The probability of drawing a card with a red side -/
def probRedSide : Rat := 8 / 18

/-- The probability that both sides are red, given that one side is red -/
theorem prob_both_red_given_one_red :
  (3 : Rat) / 4 = (List.filter (fun c => c.side1 ∧ c.side2) box).length / 
                  (List.filter (fun c => c.side1 ∨ c.side2) box).length :=
by sorry

end NUMINAMATH_CALUDE_prob_both_red_given_one_red_l3916_391653


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l3916_391666

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, geometric_sequence a q)
  (h_a3 : a 3 = 2)
  (h_a6 : a 6 = 1/4) :
  ∃ q : ℝ, geometric_sequence a q ∧ q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l3916_391666


namespace NUMINAMATH_CALUDE_total_crayons_lost_l3916_391630

/-- Represents a box of crayons with initial and final counts -/
structure CrayonBox where
  initial : Nat
  final : Nat

/-- Calculates the number of crayons lost or given away from a box -/
def crayonsLost (box : CrayonBox) : Nat :=
  box.initial - box.final

theorem total_crayons_lost (box1 box2 box3 : CrayonBox)
  (h1 : box1.initial = 479 ∧ box1.final = 134)
  (h2 : box2.initial = 352 ∧ box2.final = 221)
  (h3 : box3.initial = 621 ∧ box3.final = 487) :
  crayonsLost box1 + crayonsLost box2 + crayonsLost box3 = 610 := by
  sorry

#eval crayonsLost ⟨479, 134⟩ + crayonsLost ⟨352, 221⟩ + crayonsLost ⟨621, 487⟩

end NUMINAMATH_CALUDE_total_crayons_lost_l3916_391630


namespace NUMINAMATH_CALUDE_line_symmetry_l3916_391624

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are symmetric with respect to a third line -/
def symmetric (l1 l2 ls : Line) : Prop :=
  ∀ x y : ℝ, l1.contains x y → 
    ∃ x' y' : ℝ, l2.contains x' y' ∧
      (x + x') / 2 = (y + y') / 2 ∧ ls.contains ((x + x') / 2) ((y + y') / 2)

theorem line_symmetry :
  let l1 : Line := ⟨-2, 1, 1⟩  -- y = 2x + 1
  let l2 : Line := ⟨1, -2, 0⟩  -- x - 2y = 0
  let ls : Line := ⟨1, 1, 1⟩  -- x + y + 1 = 0
  symmetric l1 l2 ls := by sorry

end NUMINAMATH_CALUDE_line_symmetry_l3916_391624


namespace NUMINAMATH_CALUDE_hot_sauce_duration_l3916_391663

-- Define the volume of a quart in ounces
def quart_volume : ℝ := 32

-- Define the size of the hot sauce container
def container_size : ℝ := quart_volume - 2

-- Define the size of one serving in ounces
def serving_size : ℝ := 0.5

-- Define the number of servings James uses per day
def servings_per_day : ℝ := 3

-- Define the amount of hot sauce James uses per day
def daily_usage : ℝ := serving_size * servings_per_day

-- Theorem: The hot sauce will last 20 days
theorem hot_sauce_duration :
  container_size / daily_usage = 20 := by sorry

end NUMINAMATH_CALUDE_hot_sauce_duration_l3916_391663


namespace NUMINAMATH_CALUDE_book_pages_count_book_pages_count_proof_l3916_391621

theorem book_pages_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (days : ℕ) (avg_first_three : ℕ) (avg_next_three : ℕ) (last_day : ℕ) =>
    days = 7 →
    avg_first_three = 42 →
    avg_next_three = 39 →
    last_day = 28 →
    3 * avg_first_three + 3 * avg_next_three + last_day = 271

-- The proof is omitted
theorem book_pages_count_proof : book_pages_count 7 42 39 28 := by sorry

end NUMINAMATH_CALUDE_book_pages_count_book_pages_count_proof_l3916_391621


namespace NUMINAMATH_CALUDE_angle_measures_l3916_391635

/-- Given supplementary angles A and B, where A is 6 times B, and B forms a complementary angle C,
    prove the measures of angles A, B, and C. -/
theorem angle_measures (A B C : ℝ) : 
  A + B = 180 →  -- A and B are supplementary
  A = 6 * B →    -- A is 6 times B
  B + C = 90 →   -- B and C are complementary
  A = 180 * 6 / 7 ∧ B = 180 / 7 ∧ C = 90 - 180 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_measures_l3916_391635


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l3916_391640

/-- Represents the speed of the man in still water -/
def man_speed : ℝ := 9

/-- Represents the speed of the stream -/
def stream_speed : ℝ := 3

/-- The distance traveled downstream -/
def downstream_distance : ℝ := 36

/-- The distance traveled upstream -/
def upstream_distance : ℝ := 18

/-- The time taken for both downstream and upstream journeys -/
def journey_time : ℝ := 3

theorem man_speed_in_still_water :
  (man_speed + stream_speed) * journey_time = downstream_distance ∧
  (man_speed - stream_speed) * journey_time = upstream_distance →
  man_speed = 9 := by
sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l3916_391640


namespace NUMINAMATH_CALUDE_room_width_proof_l3916_391634

theorem room_width_proof (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) : 
  length = 5.5 → 
  cost_per_sqm = 750 → 
  total_cost = 16500 → 
  width * length * cost_per_sqm = total_cost → 
  width = 4 := by
sorry

end NUMINAMATH_CALUDE_room_width_proof_l3916_391634


namespace NUMINAMATH_CALUDE_fifth_term_sum_l3916_391644

def sequence_a (n : ℕ) : ℕ := 2 * n - 1

def sequence_b (n : ℕ) : ℕ := 2^(n-1)

def sequence_c (n : ℕ) : ℕ := sequence_a n * sequence_b n

theorem fifth_term_sum :
  sequence_a 5 + sequence_b 5 + sequence_c 5 = 169 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_sum_l3916_391644


namespace NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l3916_391626

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)

-- Define the "within" relation for a line being in a plane
variable (within : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_sufficient_not_necessary 
  (l m n : Line) (α : Plane)
  (m_in_α : within m α) (n_in_α : within n α) :
  (∀ l m n α, perp_plane l α → perp_line l m ∧ perp_line l n) ∧
  ¬(∀ l m n α, perp_line l m ∧ perp_line l n → perp_plane l α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l3916_391626


namespace NUMINAMATH_CALUDE_man_money_calculation_l3916_391604

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def total_amount (n_50 n_500 : ℕ) : ℕ :=
  50 * n_50 + 500 * n_500

/-- Proves that a man with 36 notes, 17 of which are 50 rupee notes and the rest are 500 rupee notes, has 10350 rupees in total -/
theorem man_money_calculation :
  let total_notes : ℕ := 36
  let n_50 : ℕ := 17
  let n_500 : ℕ := total_notes - n_50
  total_amount n_50 n_500 = 10350 := by
  sorry

#eval total_amount 17 19

end NUMINAMATH_CALUDE_man_money_calculation_l3916_391604


namespace NUMINAMATH_CALUDE_greatest_integer_x_cubed_le_27_l3916_391649

theorem greatest_integer_x_cubed_le_27 :
  ∃ (x : ℕ), x > 0 ∧ (x^6 / x^3 : ℚ) ≤ 27 ∧ ∀ (y : ℕ), y > x → (y^6 / y^3 : ℚ) > 27 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_x_cubed_le_27_l3916_391649


namespace NUMINAMATH_CALUDE_brainiacs_liking_neither_count_l3916_391668

/-- The number of brainiacs who like neither rebus teasers nor math teasers -/
def brainiacs_liking_neither (total : ℕ) (rebus : ℕ) (math : ℕ) (both : ℕ) : ℕ :=
  total - (rebus + math - both)

/-- Theorem stating the number of brainiacs liking neither type of teaser -/
theorem brainiacs_liking_neither_count :
  let total := 100
  let rebus := 2 * math
  let both := 18
  let math_not_rebus := 20
  let math := both + math_not_rebus
  brainiacs_liking_neither total rebus math both = 4 := by
  sorry

#eval brainiacs_liking_neither 100 76 38 18

end NUMINAMATH_CALUDE_brainiacs_liking_neither_count_l3916_391668


namespace NUMINAMATH_CALUDE_all_cloaks_still_too_short_l3916_391620

/-- Represents a knight with a height and a cloak length -/
structure Knight where
  height : ℝ
  cloakLength : ℝ

/-- Predicate to check if a cloak is too short for a knight -/
def isCloakTooShort (k : Knight) : Prop := k.cloakLength < k.height

/-- Function to redistribute cloaks -/
def redistributeCloaks (knights : List Knight) : List Knight :=
  sorry

theorem all_cloaks_still_too_short (knights : List Knight) 
  (h1 : knights.length = 20)
  (h2 : ∀ k ∈ knights, isCloakTooShort k)
  (h3 : List.Pairwise (λ k1 k2 => k1.height ≤ k2.height) knights)
  : ∀ k ∈ redistributeCloaks knights, isCloakTooShort k :=
by sorry

end NUMINAMATH_CALUDE_all_cloaks_still_too_short_l3916_391620


namespace NUMINAMATH_CALUDE_two_face_painted_count_l3916_391677

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  is_painted : Bool

/-- Represents a cube that has been cut into unit cubes -/
structure CutCube (n : ℕ) extends PaintedCube n where
  unit_cubes : Fin n → Fin n → Fin n → PaintedCube 1

/-- Returns the number of unit cubes with exactly two painted faces -/
def count_two_face_painted (c : CutCube 4) : ℕ := sorry

/-- Theorem stating that a 4-inch painted cube cut into 1-inch cubes has 24 cubes with exactly two painted faces -/
theorem two_face_painted_count (c : CutCube 4) : count_two_face_painted c = 24 := by sorry

end NUMINAMATH_CALUDE_two_face_painted_count_l3916_391677


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3916_391686

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_of_angle_between_vectors
  (c d : V)
  (h1 : ‖c‖ = 5)
  (h2 : ‖d‖ = 7)
  (h3 : ‖c + d‖ = 10) :
  inner c d / (‖c‖ * ‖d‖) = 13 / 35 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3916_391686


namespace NUMINAMATH_CALUDE_no_positive_roots_l3916_391693

theorem no_positive_roots :
  ∀ x : ℝ, x^3 + 6*x^2 + 11*x + 6 = 0 → x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_roots_l3916_391693


namespace NUMINAMATH_CALUDE_original_number_is_429_l3916_391645

/-- Given a three-digit number abc, this function returns the sum of all its permutations -/
def sum_of_permutations (a b c : Nat) : Nat :=
  100 * a + 10 * b + c +
  100 * a + 10 * c + b +
  100 * b + 10 * a + c +
  100 * b + 10 * c + a +
  100 * c + 10 * a + b +
  100 * c + 10 * b + a

/-- The sum of all permutations of the three-digit number we're looking for -/
def S : Nat := 4239

/-- Theorem stating that the original three-digit number is 429 -/
theorem original_number_is_429 :
  ∃ (a b c : Nat), a < 10 ∧ b < 10 ∧ c < 10 ∧ a ≠ 0 ∧ sum_of_permutations a b c = S ∧ a = 4 ∧ b = 2 ∧ c = 9 := by
  sorry


end NUMINAMATH_CALUDE_original_number_is_429_l3916_391645


namespace NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l3916_391633

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : Nat :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The theorem stating the minimum number of voters required for the Tall giraffe to win -/
theorem min_voters_for_tall_giraffe_win (vs : VotingStructure) 
  (h1 : vs.total_voters = 135)
  (h2 : vs.num_districts = 5)
  (h3 : vs.precincts_per_district = 9)
  (h4 : vs.voters_per_precinct = 3)
  (h5 : vs.total_voters = vs.num_districts * vs.precincts_per_district * vs.voters_per_precinct) :
  min_voters_to_win vs = 30 := by
  sorry

#eval min_voters_to_win ⟨135, 5, 9, 3⟩

end NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l3916_391633


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3916_391658

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmeticSequence a) 
  (h_a1 : a 1 = 3) 
  (h_a3 : a 3 = 7) : 
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3916_391658


namespace NUMINAMATH_CALUDE_solution_set_solves_inequality_l3916_391623

/-- The solution set of the inequality 12x^2 - ax > a^2 for a given real number a -/
def solution_set (a : ℝ) : Set ℝ :=
  if a > 0 then {x | x < -a/4 ∨ x > a/3}
  else if a = 0 then {x | x ≠ 0}
  else {x | x < a/3 ∨ x > -a/4}

/-- Theorem stating that the solution_set function correctly solves the inequality -/
theorem solution_set_solves_inequality (a : ℝ) :
  ∀ x, x ∈ solution_set a ↔ 12 * x^2 - a * x > a^2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_solves_inequality_l3916_391623


namespace NUMINAMATH_CALUDE_n_times_n_plus_one_divisible_by_two_l3916_391613

theorem n_times_n_plus_one_divisible_by_two (n : ℤ) (h : 1 ≤ n ∧ n ≤ 99) : 
  2 ∣ (n * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_n_times_n_plus_one_divisible_by_two_l3916_391613


namespace NUMINAMATH_CALUDE_fifteen_percent_of_number_l3916_391632

theorem fifteen_percent_of_number (x : ℝ) : 12 = 0.15 * x → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_number_l3916_391632


namespace NUMINAMATH_CALUDE_f_monotone_iff_m_range_l3916_391684

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - m) * x - m else Real.log x / Real.log m

-- State the theorem
theorem f_monotone_iff_m_range (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) ↔ (3/2 ≤ m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_iff_m_range_l3916_391684


namespace NUMINAMATH_CALUDE_f_properties_l3916_391628

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a| + |2*x - 1/a|

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x, f 1 x ≤ 6 ↔ x ∈ Set.Icc (-7/3) (5/3)) ∧
  (∀ x, f a x ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3916_391628


namespace NUMINAMATH_CALUDE_comparison_theorem_l3916_391637

theorem comparison_theorem :
  let a : ℝ := (5/3)^(1/5)
  let b : ℝ := (2/3)^10
  let c : ℝ := Real.log 6 / Real.log 0.3
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_comparison_theorem_l3916_391637


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3916_391694

theorem polynomial_factorization (x : ℝ) :
  x^12 - 3*x^9 + 3*x^3 + 1 = (x+1)^4 * (x^2-x+1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3916_391694


namespace NUMINAMATH_CALUDE_quadratic_max_iff_a_neg_l3916_391652

/-- A quadratic function -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Definition of having a maximum value for a quadratic function -/
def has_maximum (f : QuadraticFunction) : Prop :=
  ∃ x₀ : ℝ, ∀ x : ℝ, f.a * x^2 + f.b * x + f.c ≤ f.a * x₀^2 + f.b * x₀ + f.c

/-- Theorem: A quadratic function has a maximum value if and only if a < 0 -/
theorem quadratic_max_iff_a_neg (f : QuadraticFunction) :
  has_maximum f ↔ f.a < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_iff_a_neg_l3916_391652


namespace NUMINAMATH_CALUDE_movie_tickets_bought_l3916_391697

def computer_game_cost : ℕ := 66
def movie_ticket_cost : ℕ := 12
def total_spent : ℕ := 102

theorem movie_tickets_bought : 
  ∃ (x : ℕ), x * movie_ticket_cost + computer_game_cost = total_spent ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_movie_tickets_bought_l3916_391697


namespace NUMINAMATH_CALUDE_student_number_problem_l3916_391671

theorem student_number_problem (x : ℝ) : 4 * x - 142 = 110 → x = 63 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3916_391671


namespace NUMINAMATH_CALUDE_sum_equation_implies_n_value_l3916_391670

theorem sum_equation_implies_n_value : 
  990 + 992 + 994 + 996 + 998 = 5000 - N → N = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_implies_n_value_l3916_391670


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l3916_391650

/-- A parabola defined by the equation y = -x² + 2x + m --/
def parabola (x y m : ℝ) : Prop := y = -x^2 + 2*x + m

/-- Point A on the parabola --/
def point_A (y₁ m : ℝ) : Prop := parabola (-1) y₁ m

/-- Point B on the parabola --/
def point_B (y₂ m : ℝ) : Prop := parabola 1 y₂ m

/-- Point C on the parabola --/
def point_C (y₃ m : ℝ) : Prop := parabola 2 y₃ m

theorem parabola_point_relationship (y₁ y₂ y₃ m : ℝ) 
  (hA : point_A y₁ m) (hB : point_B y₂ m) (hC : point_C y₃ m) : 
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l3916_391650


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l3916_391661

/-- The area of a rhombus formed by the intersection of two equilateral triangles in a square -/
theorem rhombus_area_in_square (square_side : ℝ) (h_square_side : square_side = 4) :
  let triangle_side : ℝ := square_side
  let triangle_height : ℝ := (Real.sqrt 3 / 2) * triangle_side
  let rhombus_diagonal1 : ℝ := square_side
  let rhombus_diagonal2 : ℝ := triangle_height
  let rhombus_area : ℝ := (1 / 2) * rhombus_diagonal1 * rhombus_diagonal2
  rhombus_area = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_in_square_l3916_391661


namespace NUMINAMATH_CALUDE_convex_pentagon_probability_l3916_391616

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 5

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from total_chords -/
def ways_to_select_chords : ℕ := total_chords.choose k

/-- The number of convex pentagons that can be formed -/
def convex_pentagons : ℕ := n.choose k

/-- The probability of forming a convex pentagon -/
def probability : ℚ := convex_pentagons / ways_to_select_chords

theorem convex_pentagon_probability : probability = 1 / 1755 := by
  sorry

end NUMINAMATH_CALUDE_convex_pentagon_probability_l3916_391616


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3916_391692

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric a →
  (∀ n : ℕ+, a n > 0) →
  a 1 * a 3 + 2 * a 2 * a 5 + a 4 * a 6 = 36 →
  a 2 + a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3916_391692


namespace NUMINAMATH_CALUDE_determinant_scaling_l3916_391660

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = -3 →
  Matrix.det ![![3*x, 3*y], ![5*z, 5*w]] = -45 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l3916_391660


namespace NUMINAMATH_CALUDE_fraction_equalities_l3916_391676

theorem fraction_equalities (x y : ℚ) (h : x / y = 5 / 6) : 
  ((3 * x + 2 * y) / y = 9 / 2) ∧ 
  (y / (2 * x - y) = 3 / 2) ∧ 
  ((x - 3 * y) / y = -13 / 6) ∧ 
  ((2 * x) / (3 * y) = 5 / 9) ∧ 
  ((x + y) / (2 * y) = 11 / 12) := by
  sorry


end NUMINAMATH_CALUDE_fraction_equalities_l3916_391676


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3916_391659

theorem inequality_solution_set (x : ℝ) : 
  1 / (x^2 + 1) < 5 / x + 21 / 10 ↔ x ∈ Set.Ioi (-1/2) ∪ Set.Ioi 0 \ {-1/2} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3916_391659


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l3916_391600

theorem danny_bottle_caps (thrown_away old_caps new_caps_initial new_caps_additional : ℕ) 
  (h1 : thrown_away = 6)
  (h2 : new_caps_initial = 50)
  (h3 : new_caps_additional = thrown_away + 44) :
  new_caps_initial + new_caps_additional - thrown_away = 94 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l3916_391600


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3916_391691

/-- Represents the speed of a swimmer in still water and the speed of the stream -/
structure SwimmerSpeeds where
  man : ℝ  -- Speed of the man in still water
  stream : ℝ  -- Speed of the stream

/-- Calculates the effective speed when swimming downstream -/
def downstream_speed (s : SwimmerSpeeds) : ℝ := s.man + s.stream

/-- Calculates the effective speed when swimming upstream -/
def upstream_speed (s : SwimmerSpeeds) : ℝ := s.man - s.stream

/-- Theorem: Given the conditions of the swimming problem, the man's speed in still water is 8 km/h -/
theorem swimmer_speed_in_still_water :
  ∃ (s : SwimmerSpeeds),
    (downstream_speed s * 4 = 48) ∧
    (upstream_speed s * 6 = 24) ∧
    (s.man = 8) := by
  sorry

#check swimmer_speed_in_still_water

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3916_391691


namespace NUMINAMATH_CALUDE_function_bound_l3916_391622

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 13

-- State the theorem
theorem function_bound (x m : ℝ) (h : |x - m| < 1) : |f x - f m| < 2 * (|m| + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l3916_391622


namespace NUMINAMATH_CALUDE_mirror_area_is_2016_l3916_391662

/-- Calculates the area of a rectangular mirror inside a frame with rounded corners. -/
def mirror_area (frame_width : ℝ) (frame_height : ℝ) (frame_side_width : ℝ) : ℝ :=
  (frame_width - 2 * frame_side_width) * (frame_height - 2 * frame_side_width)

/-- Proves that the area of the mirror is 2016 cm² given the frame dimensions. -/
theorem mirror_area_is_2016 :
  mirror_area 50 70 7 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_mirror_area_is_2016_l3916_391662


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3916_391603

theorem sum_of_three_numbers (second : ℕ) (h1 : second = 30) : ∃ (first third : ℕ),
  first = 2 * second ∧ 
  third = first / 3 ∧ 
  first + second + third = 110 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3916_391603


namespace NUMINAMATH_CALUDE_point_distance_from_origin_l3916_391699

theorem point_distance_from_origin (A : ℝ) : 
  (|A - 0| = 4) → (A = 4 ∨ A = -4) := by
  sorry

end NUMINAMATH_CALUDE_point_distance_from_origin_l3916_391699


namespace NUMINAMATH_CALUDE_connie_blue_markers_l3916_391641

/-- Given that Connie has 41 red markers and a total of 105 markers,
    prove that she has 64 blue markers. -/
theorem connie_blue_markers :
  let red_markers : ℕ := 41
  let total_markers : ℕ := 105
  let blue_markers := total_markers - red_markers
  blue_markers = 64 := by
  sorry

end NUMINAMATH_CALUDE_connie_blue_markers_l3916_391641


namespace NUMINAMATH_CALUDE_impossible_tiling_l3916_391669

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_square : size * size = size^2)

/-- Represents a tile -/
structure Tile :=
  (length : Nat)
  (width : Nat)

/-- Represents a tiling configuration -/
structure TilingConfiguration :=
  (board : Chessboard)
  (tile : Tile)
  (num_tiles : Nat)
  (central_square_uncovered : Bool)

/-- Main theorem: Impossibility of specific tiling -/
theorem impossible_tiling (config : TilingConfiguration) : 
  config.board.size = 13 ∧ 
  config.tile.length = 4 ∧ 
  config.tile.width = 1 ∧
  config.num_tiles = 42 ∧
  config.central_square_uncovered = true
  → False :=
sorry

end NUMINAMATH_CALUDE_impossible_tiling_l3916_391669


namespace NUMINAMATH_CALUDE_right_triangle_cone_volumes_l3916_391682

/-- Given a right triangle with legs a and b, if the volume of the cone formed by
    rotating about leg a is 675π cm³ and the volume of the cone formed by rotating
    about leg b is 1215π cm³, then the length of the hypotenuse is 3√106 cm. -/
theorem right_triangle_cone_volumes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / 3 : ℝ) * π * b^2 * a = 675 * π →
  (1 / 3 : ℝ) * π * a^2 * b = 1215 * π →
  Real.sqrt (a^2 + b^2) = 3 * Real.sqrt 106 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_cone_volumes_l3916_391682


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3916_391615

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem complement_intersection_theorem : 
  (U \ A) ∩ (U \ B) = {4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3916_391615


namespace NUMINAMATH_CALUDE_sum_of_fractions_minus_eight_l3916_391698

theorem sum_of_fractions_minus_eight (a b c d e f : ℚ) : 
  a = 4 / 2 →
  b = 7 / 4 →
  c = 11 / 8 →
  d = 21 / 16 →
  e = 41 / 32 →
  f = 81 / 64 →
  a + b + c + d + e + f - 8 = 63 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_minus_eight_l3916_391698


namespace NUMINAMATH_CALUDE_unique_zero_implies_m_equals_one_l3916_391655

/-- A quadratic function with coefficient 1 for x^2, 2 for x, and m as the constant term -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

/-- The discriminant of the quadratic function -/
def discriminant (m : ℝ) : ℝ := 4 - 4*m

theorem unique_zero_implies_m_equals_one (m : ℝ) :
  (∃! x, quadratic m x = 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_implies_m_equals_one_l3916_391655


namespace NUMINAMATH_CALUDE_ranch_minimum_animals_l3916_391690

theorem ranch_minimum_animals (ponies horses : ℕ) : 
  ponies > 0 →
  horses = ponies + 3 →
  ∃ (ponies_with_horseshoes icelandic_ponies : ℕ),
    ponies_with_horseshoes = (3 * ponies) / 10 ∧
    icelandic_ponies = (5 * ponies_with_horseshoes) / 8 →
  ponies + horses ≥ 35 :=
by
  sorry

end NUMINAMATH_CALUDE_ranch_minimum_animals_l3916_391690


namespace NUMINAMATH_CALUDE_sum_of_solutions_equals_zero_l3916_391683

theorem sum_of_solutions_equals_zero : 
  ∃ (S : Finset Int), 
    (∀ x ∈ S, x^4 - 13*x^2 + 36 = 0) ∧ 
    (∀ x : Int, x^4 - 13*x^2 + 36 = 0 → x ∈ S) ∧ 
    (S.sum id = 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equals_zero_l3916_391683


namespace NUMINAMATH_CALUDE_Q_subset_P_l3916_391689

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def Q : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem Q_subset_P : Q ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_Q_subset_P_l3916_391689


namespace NUMINAMATH_CALUDE_triangle_sum_l3916_391685

theorem triangle_sum (AC BC : ℝ) (HE HD : ℝ) (a b : ℝ) :
  AC = 16.25 →
  BC = 13.75 →
  HE = 6 →
  HD = 3 →
  b - a = 5 →
  BC * (HD + b) = AC * (HE + a) →
  a + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_l3916_391685


namespace NUMINAMATH_CALUDE_gcd_98_63_l3916_391607

theorem gcd_98_63 : Int.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l3916_391607


namespace NUMINAMATH_CALUDE_dice_edge_length_l3916_391647

/-- The volume of the dice in cubic centimeters -/
def dice_volume : ℝ := 8

/-- The conversion factor from centimeters to millimeters -/
def cm_to_mm : ℝ := 10

/-- The length of one edge of the dice in millimeters -/
def edge_length_mm : ℝ := 20

theorem dice_edge_length :
  edge_length_mm = (dice_volume ^ (1/3 : ℝ)) * cm_to_mm :=
by sorry

end NUMINAMATH_CALUDE_dice_edge_length_l3916_391647


namespace NUMINAMATH_CALUDE_floor_plus_double_eq_sixteen_l3916_391625

theorem floor_plus_double_eq_sixteen (r : ℝ) : (⌊r⌋ : ℝ) + 2 * r = 16 ↔ r = (5.5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_double_eq_sixteen_l3916_391625


namespace NUMINAMATH_CALUDE_no_perfect_square_natural_l3916_391608

theorem no_perfect_square_natural (n : ℕ) : ¬∃ (m : ℕ), n^5 - 5*n^3 + 4*n + 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_natural_l3916_391608


namespace NUMINAMATH_CALUDE_food_budget_fraction_l3916_391610

theorem food_budget_fraction (grocery_fraction eating_out_fraction : ℚ) 
  (h1 : grocery_fraction = 0.6)
  (h2 : eating_out_fraction = 0.2) : 
  grocery_fraction + eating_out_fraction = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_food_budget_fraction_l3916_391610


namespace NUMINAMATH_CALUDE_school_year_weekly_hours_l3916_391631

def summer_weekly_hours : ℕ := 60
def summer_weeks : ℕ := 8
def summer_earnings : ℕ := 6000
def school_year_weeks : ℕ := 40
def school_year_earnings : ℕ := 7500

theorem school_year_weekly_hours : ℕ := by
  sorry

end NUMINAMATH_CALUDE_school_year_weekly_hours_l3916_391631


namespace NUMINAMATH_CALUDE_cuboid_first_edge_length_l3916_391617

/-- The length of the first edge of a cuboid with volume 30 cm³ and other edges 5 cm and 3 cm -/
def first_edge_length : ℝ := 2

/-- The volume of the cuboid -/
def cuboid_volume : ℝ := 30

/-- The width of the cuboid -/
def cuboid_width : ℝ := 5

/-- The height of the cuboid -/
def cuboid_height : ℝ := 3

theorem cuboid_first_edge_length :
  first_edge_length * cuboid_width * cuboid_height = cuboid_volume :=
by sorry

end NUMINAMATH_CALUDE_cuboid_first_edge_length_l3916_391617


namespace NUMINAMATH_CALUDE_square_of_difference_l3916_391614

theorem square_of_difference (a b : ℝ) : (a - b)^2 = a^2 - 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l3916_391614
