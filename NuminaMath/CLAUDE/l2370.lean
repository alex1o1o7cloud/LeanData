import Mathlib

namespace NUMINAMATH_CALUDE_mitch_macarons_count_l2370_237023

/-- The number of macarons Mitch made -/
def mitch_macarons : ℕ := 20

/-- The number of macarons Joshua made -/
def joshua_macarons : ℕ := mitch_macarons + 6

/-- The number of macarons Miles made -/
def miles_macarons : ℕ := 2 * joshua_macarons

/-- The number of macarons Renz made -/
def renz_macarons : ℕ := (3 * miles_macarons) / 4 - 1

/-- The total number of macarons given to kids -/
def total_macarons : ℕ := 68 * 2

theorem mitch_macarons_count : 
  mitch_macarons + joshua_macarons + miles_macarons + renz_macarons = total_macarons :=
by sorry

end NUMINAMATH_CALUDE_mitch_macarons_count_l2370_237023


namespace NUMINAMATH_CALUDE_cat_meows_l2370_237030

theorem cat_meows (cat1 : ℝ) (cat2 : ℝ) (cat3 : ℝ) : 
  cat2 = 2 * cat1 →
  cat3 = (2 * cat1) / 3 →
  5 * cat1 + 5 * cat2 + 5 * cat3 = 55 →
  cat1 = 3 := by
sorry

end NUMINAMATH_CALUDE_cat_meows_l2370_237030


namespace NUMINAMATH_CALUDE_intersection_when_a_is_neg_two_intersection_equals_A_iff_l2370_237070

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x + a < 0}

-- Theorem for part (1)
theorem intersection_when_a_is_neg_two :
  A ∩ B (-2) = {x : ℝ | 1/2 ≤ x ∧ x < 2} := by sorry

-- Theorem for part (2)
theorem intersection_equals_A_iff (a : ℝ) :
  A ∩ B a = A ↔ a < -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_neg_two_intersection_equals_A_iff_l2370_237070


namespace NUMINAMATH_CALUDE_max_score_is_94_l2370_237086

/-- Represents an operation that can be applied to a number -/
inductive Operation
  | Add : Operation
  | Square : Operation

/-- Applies an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Add => n + 1
  | Operation.Square => n * n

/-- Applies a sequence of operations to a starting number -/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Calculates the minimum distance from a number to any perfect square -/
def minDistanceToPerfectSquare (n : ℕ) : ℕ :=
  let sqrtFloor := (n.sqrt : ℕ)
  let sqrtCeil := sqrtFloor + 1
  min (n - sqrtFloor * sqrtFloor) (sqrtCeil * sqrtCeil - n)

/-- The main theorem -/
theorem max_score_is_94 :
  (∃ (ops : List Operation),
    ops.length = 100 ∧
    minDistanceToPerfectSquare (applyOperations 0 ops) = 94) ∧
  (∀ (ops : List Operation),
    ops.length = 100 →
    minDistanceToPerfectSquare (applyOperations 0 ops) ≤ 94) :=
  sorry


end NUMINAMATH_CALUDE_max_score_is_94_l2370_237086


namespace NUMINAMATH_CALUDE_quadratic_zeros_l2370_237097

theorem quadratic_zeros (x : ℝ) : 
  (x^2 - 2*x - 3 = 0) ↔ (x = -1 ∨ x = 3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_zeros_l2370_237097


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l2370_237016

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + a * b = 3) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y + x * y = 3 → 2 * a + b ≤ 2 * x + y :=
by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + a * b = 3) :
  2 * a + b = 4 * Real.sqrt 2 - 3 ↔ a = Real.sqrt 2 - 1 ∧ b = 2 * Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l2370_237016


namespace NUMINAMATH_CALUDE_vector_subtraction_l2370_237020

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := fun m ↦ (4, m)

theorem vector_subtraction (m : ℝ) (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  (a.1 - (b m).1, a.2 - (b m).2) = (-3, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2370_237020


namespace NUMINAMATH_CALUDE_symmetric_points_of_M_l2370_237044

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Original point M -/
def M : Point3D := ⟨1, -2, 3⟩

/-- Symmetric point with respect to xy-plane -/
def symmetricXY (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

/-- Symmetric point with respect to z-axis -/
def symmetricZ (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, p.z⟩

theorem symmetric_points_of_M :
  (symmetricXY M = ⟨1, -2, -3⟩) ∧ (symmetricZ M = ⟨-1, 2, 3⟩) := by sorry

end NUMINAMATH_CALUDE_symmetric_points_of_M_l2370_237044


namespace NUMINAMATH_CALUDE_m_range_theorem_l2370_237068

def f (x : ℝ) : ℝ := x^2 - 2*x

def g (m : ℝ) (x : ℝ) : ℝ := m*x + 2

theorem m_range_theorem (m : ℝ) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, g m x₁ = f x₀) →
  m ∈ Set.Icc (-1 : ℝ) (1/2) :=
by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l2370_237068


namespace NUMINAMATH_CALUDE_evaluate_expression_l2370_237026

theorem evaluate_expression : 3000 * (3000 ^ 3001) = 3000 ^ 3002 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2370_237026


namespace NUMINAMATH_CALUDE_rectangle_circle_tangency_l2370_237045

theorem rectangle_circle_tangency (r : ℝ) (a b : ℝ) : 
  r = 6 →                             -- Circle radius is 6 cm
  a ≥ b →                             -- a is the longer side, b is the shorter side
  b = 2 * r →                         -- Circle is tangent to shorter side
  a * b = 3 * (π * r^2) →             -- Rectangle area is triple the circle area
  b = 12 :=                           -- Shorter side length is 12 cm
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_tangency_l2370_237045


namespace NUMINAMATH_CALUDE_papi_calot_plants_to_buy_l2370_237040

/-- Calculates the total number of plants needed for a given crop -/
def totalPlants (rows : ℕ) (plantsPerRow : ℕ) (additional : ℕ) : ℕ :=
  rows * plantsPerRow + additional

/-- Represents Papi Calot's garden planning -/
structure GardenPlan where
  potatoRows : ℕ
  potatoPlantsPerRow : ℕ
  additionalPotatoes : ℕ
  carrotRows : ℕ
  carrotPlantsPerRow : ℕ
  additionalCarrots : ℕ
  onionRows : ℕ
  onionPlantsPerRow : ℕ
  additionalOnions : ℕ

/-- Theorem stating the correct number of plants Papi Calot needs to buy -/
theorem papi_calot_plants_to_buy (plan : GardenPlan)
  (h_potato : plan.potatoRows = 10 ∧ plan.potatoPlantsPerRow = 25 ∧ plan.additionalPotatoes = 20)
  (h_carrot : plan.carrotRows = 15 ∧ plan.carrotPlantsPerRow = 30 ∧ plan.additionalCarrots = 30)
  (h_onion : plan.onionRows = 12 ∧ plan.onionPlantsPerRow = 20 ∧ plan.additionalOnions = 10) :
  totalPlants plan.potatoRows plan.potatoPlantsPerRow plan.additionalPotatoes = 270 ∧
  totalPlants plan.carrotRows plan.carrotPlantsPerRow plan.additionalCarrots = 480 ∧
  totalPlants plan.onionRows plan.onionPlantsPerRow plan.additionalOnions = 250 := by
  sorry

end NUMINAMATH_CALUDE_papi_calot_plants_to_buy_l2370_237040


namespace NUMINAMATH_CALUDE_anusha_share_is_84_l2370_237095

/-- Represents the share distribution problem among three children -/
structure ShareDistribution where
  total : ℝ
  anusha : ℝ
  babu : ℝ
  esha : ℝ
  sum_equals_total : anusha + babu + esha = total
  shares_relation : 12 * anusha = 8 * babu ∧ 8 * babu = 6 * esha

/-- Theorem stating that Anusha's share is 84 rupees given the conditions -/
theorem anusha_share_is_84 (sd : ShareDistribution) (h : sd.total = 378) : sd.anusha = 84 := by
  sorry

end NUMINAMATH_CALUDE_anusha_share_is_84_l2370_237095


namespace NUMINAMATH_CALUDE_equation_solution_l2370_237091

theorem equation_solution : 
  ∃ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ↔ x = -48 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2370_237091


namespace NUMINAMATH_CALUDE_lcm_primes_sum_l2370_237031

theorem lcm_primes_sum (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x > y → Nat.lcm x y = 10 → 2 * x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_primes_sum_l2370_237031


namespace NUMINAMATH_CALUDE_lg_sum_five_two_l2370_237003

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_five_two : lg 5 + lg 2 = 1 := by sorry

end NUMINAMATH_CALUDE_lg_sum_five_two_l2370_237003


namespace NUMINAMATH_CALUDE_system_solution_l2370_237079

theorem system_solution :
  let solutions : List (ℤ × ℤ × ℤ) := [(0, 12, 0), (2, 7, 3), (4, 2, 6)]
  ∀ x y z : ℤ,
    (x + y + z = 12 ∧ 8*x + 5*y + 3*z = 60) ↔ (x, y, z) ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2370_237079


namespace NUMINAMATH_CALUDE_complex_determinant_equation_l2370_237001

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the theorem
theorem complex_determinant_equation :
  ∃ (z : ℂ), det z Complex.I 1 Complex.I = 1 + Complex.I ∧ z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_determinant_equation_l2370_237001


namespace NUMINAMATH_CALUDE_intersection_complement_when_a_2_a_value_when_union_equals_A_l2370_237019

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - (a+1)*x + a = 0}

-- Part 1
theorem intersection_complement_when_a_2 :
  A ∩ (Set.univ \ B 2) = {-3} := by sorry

-- Part 2
theorem a_value_when_union_equals_A :
  ∀ a : ℝ, A ∪ B a = A → a = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_a_2_a_value_when_union_equals_A_l2370_237019


namespace NUMINAMATH_CALUDE_complex_coordinate_l2370_237029

theorem complex_coordinate (z : ℂ) (h : Complex.I * z = 2 + 4 * Complex.I) : 
  z = 4 - 2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_coordinate_l2370_237029


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l2370_237049

theorem restaurant_bill_theorem (total_people : ℕ) (total_bill : ℚ) (gratuity_rate : ℚ) :
  total_people = 6 →
  total_bill = 720 →
  gratuity_rate = 1/5 →
  (total_bill / (1 + gratuity_rate)) / total_people = 100 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l2370_237049


namespace NUMINAMATH_CALUDE_divisibility_property_l2370_237028

theorem divisibility_property (n : ℕ) 
  (h1 : 3 ∣ (n + 2)) 
  (h2 : 4 ∣ (n + 3)) : 
  6 ∣ (n + 5) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2370_237028


namespace NUMINAMATH_CALUDE_lloyd_decks_required_l2370_237059

/-- Represents the number of cards in a standard deck --/
def cards_per_deck : ℕ := 52

/-- Represents the number of layers in Lloyd's house of cards --/
def num_layers : ℕ := 32

/-- Represents the number of cards per layer in Lloyd's house of cards --/
def cards_per_layer : ℕ := 26

/-- Calculates the total number of cards in the house of cards --/
def total_cards : ℕ := num_layers * cards_per_layer

/-- Theorem: The number of complete decks required for Lloyd's house of cards is 16 --/
theorem lloyd_decks_required : (total_cards / cards_per_deck) = 16 := by
  sorry

end NUMINAMATH_CALUDE_lloyd_decks_required_l2370_237059


namespace NUMINAMATH_CALUDE_triangle_problem_l2370_237078

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A) ∧
  (a = 2) ∧
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →
  (A = π/3) ∧ (a + b + c = 6) := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l2370_237078


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2370_237012

theorem fraction_multiplication : (1 : ℚ) / 3 * (3 : ℚ) / 4 * (4 : ℚ) / 5 = (1 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2370_237012


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l2370_237050

/-- 
Given an infinite geometric series with common ratio 1/4 and sum 40,
prove that the second term of the sequence is 7.5.
-/
theorem second_term_of_geometric_series (a : ℝ) : 
  (∑' n, a * (1/4)^n = 40) → a * (1/4) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l2370_237050


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_450_l2370_237024

theorem largest_multiple_of_15_less_than_450 :
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 450 → n ≤ 435 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_450_l2370_237024


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l2370_237061

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) : 
  x = Real.sqrt (8 + 9 / 16) → x = Real.sqrt 137 / 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l2370_237061


namespace NUMINAMATH_CALUDE_largest_product_sum_of_digits_l2370_237090

def is_prime (p : ℕ) : Prop := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem largest_product_sum_of_digits :
  ∃ (n d e : ℕ),
    is_prime d ∧ is_prime e ∧ is_prime (10 * e + d) ∧
    d ∈ ({5, 7} : Set ℕ) ∧ e ∈ ({3, 7} : Set ℕ) ∧
    n = d * e * (10 * e + d) ∧
    (∀ (m d' e' : ℕ),
      is_prime d' ∧ is_prime e' ∧ is_prime (10 * e' + d') ∧
      d' ∈ ({5, 7} : Set ℕ) ∧ e' ∈ ({3, 7} : Set ℕ) ∧
      m = d' * e' * (10 * e' + d') →
      m ≤ n) ∧
    sum_of_digits n = 21 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_sum_of_digits_l2370_237090


namespace NUMINAMATH_CALUDE_single_digit_sum_l2370_237046

/-- Given two different single-digit numbers A and B where AB × 6 = BBB, 
    prove that A + B = 11 -/
theorem single_digit_sum (A B : ℕ) : 
  A ≠ B ∧ 
  A < 10 ∧ 
  B < 10 ∧ 
  (10 * A + B) * 6 = 100 * B + 10 * B + B → 
  A + B = 11 := by
sorry

end NUMINAMATH_CALUDE_single_digit_sum_l2370_237046


namespace NUMINAMATH_CALUDE_apple_cost_price_l2370_237013

theorem apple_cost_price (SP : ℝ) (CP : ℝ) : SP = 16 ∧ SP = (5/6) * CP → CP = 19.2 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_price_l2370_237013


namespace NUMINAMATH_CALUDE_extremum_implies_a_b_values_l2370_237055

/-- The function f(x) = x^3 - ax^2 - bx + a^2 has an extremum value of 10 at x = 1 -/
def has_extremum (a b : ℝ) : Prop :=
  let f := fun x : ℝ => x^3 - a*x^2 - b*x + a^2
  (∃ ε > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < ε → f x ≤ f 1) ∧
  (∃ ε > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < ε → f x ≥ f 1) ∧
  f 1 = 10

/-- If f(x) = x^3 - ax^2 - bx + a^2 has an extremum value of 10 at x = 1, then a = -4 and b = 11 -/
theorem extremum_implies_a_b_values :
  ∀ a b : ℝ, has_extremum a b → a = -4 ∧ b = 11 := by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_a_b_values_l2370_237055


namespace NUMINAMATH_CALUDE_smallest_multiple_of_one_to_five_l2370_237048

theorem smallest_multiple_of_one_to_five : ∃ n : ℕ+, 
  (∀ m : ℕ, 1 ≤ m ∧ m ≤ 5 → m ∣ n) ∧
  (∀ k : ℕ+, (∀ m : ℕ, 1 ≤ m ∧ m ≤ 5 → m ∣ k) → n ≤ k) ∧
  n = 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_one_to_five_l2370_237048


namespace NUMINAMATH_CALUDE_right_triangle_set_l2370_237067

theorem right_triangle_set (a b c : ℝ) : 
  (a = 1.5 ∧ b = 2 ∧ c = 2.5) → 
  a^2 + b^2 = c^2 ∧
  ¬(4^2 + 5^2 = 6^2) ∧
  ¬(1^2 + (Real.sqrt 2)^2 = 2.5^2) ∧
  ¬(2^2 + 3^2 = 4^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_set_l2370_237067


namespace NUMINAMATH_CALUDE_least_common_period_is_36_l2370_237088

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 6) + f (x - 6) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- The least common positive period for all functions satisfying the functional equation -/
def LeastCommonPeriod (p : ℝ) : Prop :=
  p > 0 ∧
  (∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f → IsPeriod f p) ∧
  ∀ q, q > 0 → (∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f → IsPeriod f q) → p ≤ q

theorem least_common_period_is_36 : LeastCommonPeriod 36 := by
  sorry

end NUMINAMATH_CALUDE_least_common_period_is_36_l2370_237088


namespace NUMINAMATH_CALUDE_magic_trick_result_l2370_237052

theorem magic_trick_result (x : ℚ) : ((2 * x + 8) / 4) - (x / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_magic_trick_result_l2370_237052


namespace NUMINAMATH_CALUDE_local_minimum_implies_b_bound_l2370_237008

/-- A function f with a parameter b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

/-- The first derivative of f with respect to x -/
def f_derivative (b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*b

theorem local_minimum_implies_b_bound :
  ∀ b : ℝ, (∃ x : ℝ, 0 < x ∧ x < 1 ∧
    (∀ y : ℝ, 0 < y ∧ y < 1 → f b x ≤ f b y) ∧
    (∃ δ : ℝ, δ > 0 ∧ ∀ y : ℝ, 0 < |y - x| ∧ |y - x| < δ → f b x < f b y)) →
  (0 < b ∧ b < 1) :=
by sorry

end NUMINAMATH_CALUDE_local_minimum_implies_b_bound_l2370_237008


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2370_237066

theorem contrapositive_equivalence (p q : Prop) :
  (q → p) → (¬p → ¬q) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2370_237066


namespace NUMINAMATH_CALUDE_other_divisor_of_h_l2370_237053

def is_divisor (d n : ℕ) : Prop := n % d = 0

theorem other_divisor_of_h (h a b c : ℕ) : 
  h > 0 →
  is_divisor 225 h →
  h = 2^a * 3^b * 5^c →
  a > 0 →
  b > 0 →
  c > 0 →
  a + b + c ≥ 8 →
  (∀ a' b' c' : ℕ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' < a + b + c → ¬(h = 2^a' * 3^b' * 5^c')) →
  ∃ d : ℕ, d ≠ 225 ∧ is_divisor d h ∧ d = 16 :=
by sorry

end NUMINAMATH_CALUDE_other_divisor_of_h_l2370_237053


namespace NUMINAMATH_CALUDE_car_distance_theorem_l2370_237099

/-- Calculates the total distance traveled by a car over a given number of hours,
    where the car's speed increases by a fixed amount each hour. -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc i => acc + (initialSpeed + i * speedIncrease)) 0

/-- Proves that a car traveling for 12 hours, starting at 45 km/h and increasing
    speed by 2 km/h each hour, travels a total of 672 km. -/
theorem car_distance_theorem :
  totalDistance 45 2 12 = 672 := by
  sorry

#eval totalDistance 45 2 12

end NUMINAMATH_CALUDE_car_distance_theorem_l2370_237099


namespace NUMINAMATH_CALUDE_max_pieces_is_72_l2370_237051

/-- Represents a rectangular cake with dimensions m and n -/
structure Cake where
  m : ℕ+
  n : ℕ+

/-- Calculates the number of pieces in the two central rows -/
def central_pieces (c : Cake) : ℕ := (c.m - 4) * (c.n - 4)

/-- Calculates the number of pieces on the perimeter -/
def perimeter_pieces (c : Cake) : ℕ := 2 * c.m + 2 * c.n - 4

/-- Checks if the cake satisfies the chef's condition -/
def satisfies_condition (c : Cake) : Prop :=
  central_pieces c = perimeter_pieces c

/-- Calculates the total number of pieces -/
def total_pieces (c : Cake) : ℕ := c.m * c.n

/-- States that the maximum number of pieces satisfying the condition is 72 -/
theorem max_pieces_is_72 :
  ∃ (c : Cake), satisfies_condition c ∧
    total_pieces c = 72 ∧
    ∀ (c' : Cake), satisfies_condition c' → total_pieces c' ≤ 72 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_is_72_l2370_237051


namespace NUMINAMATH_CALUDE_number_difference_l2370_237094

theorem number_difference (a b : ℕ) : 
  b = 10 * a + 5 →
  a + b = 22500 →
  b - a = 18410 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2370_237094


namespace NUMINAMATH_CALUDE_apple_basket_problem_l2370_237083

/-- The number of baskets in the apple-picking problem -/
def number_of_baskets : ℕ := 11

/-- The total number of apples initially -/
def total_apples : ℕ := 1000

/-- The number of apples left after picking -/
def apples_left : ℕ := 340

/-- The number of children picking apples -/
def number_of_children : ℕ := 10

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem apple_basket_problem :
  (number_of_children * sum_of_first_n number_of_baskets = total_apples - apples_left) ∧
  (number_of_baskets > 0) :=
sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l2370_237083


namespace NUMINAMATH_CALUDE_consecutive_points_segment_length_l2370_237093

/-- Given five consecutive points on a straight line, prove the length of a specific segment -/
theorem consecutive_points_segment_length 
  (a b c d e : ℝ) -- Define points as real numbers representing their positions on the line
  (consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Ensure points are consecutive
  (bc_cd : c - b = 2 * (d - c)) -- bc = 2 cd
  (de_length : e - d = 4) -- de = 4
  (ab_length : b - a = 5) -- ab = 5
  (ae_length : e - a = 18) -- ae = 18
  : c - a = 11 := by -- Prove that ac = 11
  sorry

end NUMINAMATH_CALUDE_consecutive_points_segment_length_l2370_237093


namespace NUMINAMATH_CALUDE_concert_ticket_price_l2370_237062

/-- Given concert ticket information, prove the cost of a section B ticket -/
theorem concert_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (section_a_tickets : ℕ) 
  (section_b_tickets : ℕ) 
  (section_a_price : ℚ)
  (h1 : total_tickets = section_a_tickets + section_b_tickets)
  (h2 : total_tickets = 4500)
  (h3 : total_revenue = 30000)
  (h4 : section_a_tickets = 2900)
  (h5 : section_b_tickets = 1600)
  (h6 : section_a_price = 8) :
  ∃ (section_b_price : ℚ), 
    section_b_price = 4.25 ∧ 
    total_revenue = section_a_tickets * section_a_price + section_b_tickets * section_b_price :=
by sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l2370_237062


namespace NUMINAMATH_CALUDE_vector_dot_product_l2370_237036

/-- Given vectors a and b in ℝ², prove that (2a + b) · a = 6 -/
theorem vector_dot_product (a b : ℝ × ℝ) (h1 : a = (2, -1)) (h2 : b = (-1, 2)) :
  (2 • a + b) • a = 6 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l2370_237036


namespace NUMINAMATH_CALUDE_min_q_value_l2370_237058

/-- The number of cards in the deck -/
def num_cards : ℕ := 52

/-- The probability that Alex and Dylan are on the same team given that Alex picks one of the cards a or a+11, and Dylan picks the other -/
def q (a : ℕ) : ℚ :=
  let lower_team := Nat.choose (a - 1) 2
  let higher_team := Nat.choose (num_cards - a - 11) 2
  let total_combinations := Nat.choose (num_cards - 2) 2
  (lower_team + higher_team : ℚ) / total_combinations

/-- The minimum value of a for which q(a) ≥ 1/2 -/
def min_a : ℕ := 4

theorem min_q_value :
  q min_a = 91 / 175 ∧ ∀ a : ℕ, 1 ≤ a ∧ a ≤ num_cards - 11 → q a ≥ 91 / 175 := by sorry

end NUMINAMATH_CALUDE_min_q_value_l2370_237058


namespace NUMINAMATH_CALUDE_birds_in_tree_l2370_237073

theorem birds_in_tree (initial_birds final_birds : ℕ) (h1 : initial_birds = 179) (h2 : final_birds = 217) :
  final_birds - initial_birds = 38 := by
sorry

end NUMINAMATH_CALUDE_birds_in_tree_l2370_237073


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2370_237038

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 1 ∧ 
  (∀ (m : ℕ), (13294 - m) % 97 = 0 → m ≥ n) ∧
  (13294 - n) % 97 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2370_237038


namespace NUMINAMATH_CALUDE_gcd_20586_58768_l2370_237082

theorem gcd_20586_58768 : Nat.gcd 20586 58768 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_20586_58768_l2370_237082


namespace NUMINAMATH_CALUDE_f_simplification_f_specific_value_l2370_237004

noncomputable def f (α : Real) : Real :=
  (Real.sin (4 * Real.pi - α) * Real.cos (Real.pi - α) * Real.cos ((3 * Real.pi) / 2 + α) * Real.cos ((7 * Real.pi) / 2 - α)) /
  (Real.cos (Real.pi + α) * Real.sin (2 * Real.pi - α) * Real.sin (Real.pi + α) * Real.sin ((9 * Real.pi) / 2 - α))

theorem f_simplification (α : Real) : f α = Real.tan α := by sorry

theorem f_specific_value : f (-(31 / 6) * Real.pi) = -(Real.sqrt 3 / 3) := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_specific_value_l2370_237004


namespace NUMINAMATH_CALUDE_polynomial_expansion_coefficient_l2370_237084

theorem polynomial_expansion_coefficient (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^10 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
            a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9 + a₁₀*(x-1)^10) →
  a₈ = 45 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_coefficient_l2370_237084


namespace NUMINAMATH_CALUDE_box_volume_l2370_237011

/-- A rectangular box with specific proportions -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  front_half_top : length * height = 0.5 * (length * width)
  top_1_5_side : length * width = 1.5 * (width * height)
  side_area : width * height = 72

/-- The volume of a box is equal to 648 cubic units -/
theorem box_volume (b : Box) : b.length * b.width * b.height = 648 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l2370_237011


namespace NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l2370_237077

-- Define the first inequality
def inequality1 (x : ℝ) : Prop := (5 * (x - 1)) / 6 - 1 < (x + 2) / 3

-- Define the second system of inequalities
def inequality2 (x : ℝ) : Prop := 3 * x - 2 ≤ x + 6 ∧ (5 * x + 3) / 2 > x

-- Theorem for the first inequality
theorem solution_inequality1 : 
  {x : ℕ | inequality1 x} = {1, 2, 3, 4} :=
sorry

-- Theorem for the second system of inequalities
theorem solution_inequality2 : 
  {x : ℝ | inequality2 x} = {x : ℝ | -1 < x ∧ x ≤ 4} :=
sorry

end NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l2370_237077


namespace NUMINAMATH_CALUDE_plane_speed_l2370_237037

theorem plane_speed (distance_with_wind : ℝ) (distance_against_wind : ℝ) (wind_speed : ℝ) :
  distance_with_wind = 400 →
  distance_against_wind = 320 →
  wind_speed = 20 →
  ∃ (still_air_speed : ℝ) (time : ℝ),
    time > 0 ∧
    distance_with_wind = (still_air_speed + wind_speed) * time ∧
    distance_against_wind = (still_air_speed - wind_speed) * time ∧
    still_air_speed = 180 :=
by sorry

end NUMINAMATH_CALUDE_plane_speed_l2370_237037


namespace NUMINAMATH_CALUDE_percentage_of_12_to_80_l2370_237039

theorem percentage_of_12_to_80 : ∀ (x : ℝ), x = 12 ∧ (x / 80) * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_percentage_of_12_to_80_l2370_237039


namespace NUMINAMATH_CALUDE_function_equal_to_parabola_l2370_237042

-- Define a property for functions that have the same intersection behavior as x^2
def HasSameIntersections (f : ℝ → ℝ) : Prop :=
  ∀ (a b : ℝ), (∀ x : ℝ, (a * x + b = f x) ↔ (a * x + b = x^2))

-- State the theorem
theorem function_equal_to_parabola (f : ℝ → ℝ) :
  HasSameIntersections f → (∀ x : ℝ, f x = x^2) :=
by sorry

end NUMINAMATH_CALUDE_function_equal_to_parabola_l2370_237042


namespace NUMINAMATH_CALUDE_kitchen_length_l2370_237027

theorem kitchen_length (tile_area : ℝ) (kitchen_width : ℝ) (num_tiles : ℕ) :
  tile_area = 6 →
  kitchen_width = 48 →
  num_tiles = 96 →
  (kitchen_width * (num_tiles * tile_area / kitchen_width) : ℝ) = 12 * kitchen_width :=
by sorry

end NUMINAMATH_CALUDE_kitchen_length_l2370_237027


namespace NUMINAMATH_CALUDE_product_of_sums_is_even_l2370_237080

/-- A card with two numbers -/
structure Card where
  front : Nat
  back : Nat

/-- The set of all cards -/
def deck : Finset Card := sorry

/-- The theorem to prove -/
theorem product_of_sums_is_even :
  (∀ c ∈ deck, c.front ∈ Finset.range 100 ∧ c.back ∈ Finset.range 100) →
  deck.card = 99 →
  (Finset.range 100).card = deck.card + 1 →
  (∀ n ∈ Finset.range 100, (deck.filter (λ c => c.front = n)).card +
    (deck.filter (λ c => c.back = n)).card = 1) →
  Even ((deck.prod (λ c => c.front + c.back))) :=
by sorry

end NUMINAMATH_CALUDE_product_of_sums_is_even_l2370_237080


namespace NUMINAMATH_CALUDE_sin_B_value_max_perimeter_l2370_237075

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition (2a-c)cosB = bcosC -/
def triangle_condition (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C

/-- Theorem 1: If (2a-c)cosB = bcosC, then sinB = √3/2 -/
theorem sin_B_value (t : Triangle) (h : triangle_condition t) : 
  Real.sin t.B = Real.sqrt 3 / 2 := by sorry

/-- Theorem 2: If b = √7, then the maximum perimeter is 3√7 -/
theorem max_perimeter (t : Triangle) (h : t.b = Real.sqrt 7) :
  t.a + t.b + t.c ≤ 3 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_sin_B_value_max_perimeter_l2370_237075


namespace NUMINAMATH_CALUDE_line_transformation_theorem_l2370_237033

/-- Given a line with equation y = mx + b, returns a new line with half the slope and twice the y-intercept -/
def transform_line (m b : ℚ) : ℚ × ℚ := (m / 2, 2 * b)

theorem line_transformation_theorem :
  let original_line := ((2 : ℚ) / 3, 4)
  let transformed_line := transform_line original_line.1 original_line.2
  transformed_line = ((1 : ℚ) / 3, 8) := by sorry

end NUMINAMATH_CALUDE_line_transformation_theorem_l2370_237033


namespace NUMINAMATH_CALUDE_not_perfect_square_floor_theorem_l2370_237071

theorem not_perfect_square_floor_theorem (A : ℕ) (h : ¬ ∃ k : ℕ, A = k ^ 2) :
  ∃ n : ℕ, A = ⌊(n : ℝ) + Real.sqrt n + 1/2⌋ := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_floor_theorem_l2370_237071


namespace NUMINAMATH_CALUDE_sibling_age_difference_l2370_237096

theorem sibling_age_difference (youngest_age : ℕ) (average_age : ℕ) : 
  youngest_age = 17 → 
  average_age = 21 → 
  ∃ (x : ℕ), 
    (youngest_age + (youngest_age + x) + (youngest_age + 4) + (youngest_age + 7)) / 4 = average_age ∧ 
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_sibling_age_difference_l2370_237096


namespace NUMINAMATH_CALUDE_good_goods_sufficient_condition_l2370_237005

-- Define propositions
variable (G : Prop) -- G represents "goods are good"
variable (C : Prop) -- C represents "goods are cheap"

-- Define the statement "Good goods are not cheap"
def good_goods_not_cheap : Prop := G → ¬C

-- Theorem to prove
theorem good_goods_sufficient_condition (h : good_goods_not_cheap G C) : 
  G → ¬C :=
by
  sorry


end NUMINAMATH_CALUDE_good_goods_sufficient_condition_l2370_237005


namespace NUMINAMATH_CALUDE_percentage_comparison_l2370_237065

theorem percentage_comparison (base : ℝ) (first second : ℝ) 
  (h1 : first = base * 1.71)
  (h2 : second = base * 1.80) :
  first / second * 100 = 95 := by
  sorry

end NUMINAMATH_CALUDE_percentage_comparison_l2370_237065


namespace NUMINAMATH_CALUDE_triangle_side_length_l2370_237000

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 2 → b = 1 → C = Real.pi / 3 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2370_237000


namespace NUMINAMATH_CALUDE_find_M_l2370_237085

theorem find_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1800) ∧ (M = 2520) := by
  sorry

end NUMINAMATH_CALUDE_find_M_l2370_237085


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2370_237081

theorem polynomial_simplification (x : ℝ) :
  (x + 1)^4 - 4*(x + 1)^3 + 6*(x + 1)^2 - 4*(x + 1) + 1 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2370_237081


namespace NUMINAMATH_CALUDE_fewer_girls_than_boys_l2370_237064

theorem fewer_girls_than_boys (total_students : ℕ) (girls : ℕ) (boys : ℕ) : 
  total_students = 27 →
  girls = 11 →
  total_students = girls + boys →
  boys - girls = 5 := by
sorry

end NUMINAMATH_CALUDE_fewer_girls_than_boys_l2370_237064


namespace NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l2370_237032

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f(x) = 1 + log_a(x-1)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + log a (x - 1)

-- State the theorem
theorem fixed_point_of_logarithmic_function (a : ℝ) (h : a > 0 ∧ a ≠ 1) : f a 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l2370_237032


namespace NUMINAMATH_CALUDE_expression_equals_five_l2370_237098

theorem expression_equals_five :
  (1 - Real.sqrt 5) ^ 0 + |-Real.sqrt 2| - 2 * Real.cos (π / 4) + (1 / 4)⁻¹ = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_five_l2370_237098


namespace NUMINAMATH_CALUDE_bottle_caps_problem_l2370_237076

theorem bottle_caps_problem (katherine_initial : ℕ) (hippopotamus_eaten : ℕ) : 
  katherine_initial = 34 →
  hippopotamus_eaten = 8 →
  (katherine_initial / 2 : ℕ) - hippopotamus_eaten = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_problem_l2370_237076


namespace NUMINAMATH_CALUDE_anna_savings_account_l2370_237009

def geometricSeriesSum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem anna_savings_account (a : ℕ) (r : ℕ) (target : ℕ) :
  a = 2 → r = 2 → target = 500 →
  (∀ k < 8, geometricSeriesSum a r k < target) ∧
  geometricSeriesSum a r 8 ≥ target :=
by sorry

end NUMINAMATH_CALUDE_anna_savings_account_l2370_237009


namespace NUMINAMATH_CALUDE_banana_arrangements_l2370_237087

/-- The number of ways to arrange letters in a word -/
def arrange_letters (total : ℕ) (freq1 freq2 freq3 : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial freq1 * Nat.factorial freq2 * Nat.factorial freq3)

/-- Theorem: The number of arrangements of BANANA is 60 -/
theorem banana_arrangements :
  arrange_letters 6 1 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l2370_237087


namespace NUMINAMATH_CALUDE_fourth_root_squared_cubed_l2370_237060

theorem fourth_root_squared_cubed (x : ℝ) : ((x^(1/4))^2)^3 = 1296 → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_squared_cubed_l2370_237060


namespace NUMINAMATH_CALUDE_money_division_l2370_237047

theorem money_division (a b c : ℚ) : 
  (4 * a = 5 * b) → 
  (5 * b = 10 * c) → 
  (c = 160) → 
  (a + b + c = 880) := by
sorry

end NUMINAMATH_CALUDE_money_division_l2370_237047


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l2370_237063

theorem two_digit_number_problem : ∃ (x : ℕ), 
  (x ≥ 10 ∧ x ≤ 99) ∧ 
  (500 + x = 9 * x - 12) ∧ 
  x = 64 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l2370_237063


namespace NUMINAMATH_CALUDE_basketball_team_average_weight_l2370_237057

/-- Given a basketball team with boys and girls, calculate the average weight of all players. -/
theorem basketball_team_average_weight 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (avg_weight_boys : ℚ) 
  (avg_weight_girls : ℚ) 
  (h_num_boys : num_boys = 8) 
  (h_num_girls : num_girls = 5) 
  (h_avg_weight_boys : avg_weight_boys = 160) 
  (h_avg_weight_girls : avg_weight_girls = 130) : 
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 148 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_average_weight_l2370_237057


namespace NUMINAMATH_CALUDE_line_through_quadrants_line_through_fixed_point_point_slope_form_slope_intercept_form_l2370_237022

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- 1. Line passing through first, second, and fourth quadrants
theorem line_through_quadrants (l : Line) :
  (∃ x y, x > 0 ∧ y > 0 ∧ y = l.slope * x + l.intercept) ∧
  (∃ x y, x < 0 ∧ y > 0 ∧ y = l.slope * x + l.intercept) ∧
  (∃ x y, x > 0 ∧ y < 0 ∧ y = l.slope * x + l.intercept) →
  l.slope < 0 ∧ l.intercept > 0 :=
sorry

-- 2. Line passing through a fixed point
theorem line_through_fixed_point (k : ℝ) :
  ∃ x y, k * x - y - 2 * k + 3 = 0 ∧ x = 2 ∧ y = 3 :=
sorry

-- 3. Point-slope form equation
theorem point_slope_form (p : Point) (m : ℝ) :
  p.x = 2 ∧ p.y = -1 ∧ m = -Real.sqrt 3 →
  ∀ x y, y + 1 = -Real.sqrt 3 * (x - 2) ↔ y - p.y = m * (x - p.x) :=
sorry

-- 4. Slope-intercept form equation
theorem slope_intercept_form (l : Line) :
  l.slope = -2 ∧ l.intercept = 3 →
  ∀ x y, y = l.slope * x + l.intercept ↔ y = -2 * x + 3 :=
sorry

end NUMINAMATH_CALUDE_line_through_quadrants_line_through_fixed_point_point_slope_form_slope_intercept_form_l2370_237022


namespace NUMINAMATH_CALUDE_sqrt_25000_simplified_l2370_237043

theorem sqrt_25000_simplified : Real.sqrt 25000 = 50 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_25000_simplified_l2370_237043


namespace NUMINAMATH_CALUDE_system_unique_solution_l2370_237014

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  Real.arccos ((4 - y) / 4) = Real.arccos ((a + x) / 2) ∧
  x^2 + y^2 + 2*x - 8*y = b

-- Define the condition for a
def a_condition (a : ℝ) : Prop :=
  a ≤ -9 ∨ a ≥ 11

-- Theorem statement
theorem system_unique_solution (a : ℝ) :
  (∀ b : ℝ, (∃! p : ℝ × ℝ, system a b p.1 p.2) ∨ (¬ ∃ p : ℝ × ℝ, system a b p.1 p.2)) ↔
  a_condition a :=
sorry

end NUMINAMATH_CALUDE_system_unique_solution_l2370_237014


namespace NUMINAMATH_CALUDE_employees_abroad_l2370_237002

theorem employees_abroad (total : ℕ) (fraction : ℚ) (abroad : ℕ) : 
  total = 450 → fraction = 0.06 → abroad = (total : ℚ) * fraction → abroad = 27 := by
sorry

end NUMINAMATH_CALUDE_employees_abroad_l2370_237002


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l2370_237054

theorem acute_triangle_properties (A B C a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧  -- Sum of angles in a triangle
  Real.sqrt ((1 - Real.cos (2 * C)) / 2) + Real.sin (B - A) = 2 * Real.sin (2 * A) ∧  -- Given equation
  c ≥ max a b ∧  -- AB is the longest side
  a = Real.sin A ∧ b = Real.sin B ∧ c = Real.sin C  -- Law of sines
  →
  a / b = 1 / 2 ∧ 0 < Real.cos C ∧ Real.cos C ≤ 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l2370_237054


namespace NUMINAMATH_CALUDE_fishbowl_count_l2370_237041

theorem fishbowl_count (total_fish : ℕ) (fish_per_bowl : ℕ) (h1 : total_fish = 6003) (h2 : fish_per_bowl = 23) :
  total_fish / fish_per_bowl = 261 := by
sorry

end NUMINAMATH_CALUDE_fishbowl_count_l2370_237041


namespace NUMINAMATH_CALUDE_fraction_addition_l2370_237007

theorem fraction_addition : (1 : ℚ) / 4 + (3 : ℚ) / 8 = (5 : ℚ) / 8 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_l2370_237007


namespace NUMINAMATH_CALUDE_inequality_proof_l2370_237089

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2370_237089


namespace NUMINAMATH_CALUDE_three_men_three_women_arrangements_l2370_237018

/-- The number of ways to arrange n men and n women in a row, such that no two men or two women are adjacent -/
def alternating_arrangements (n : ℕ) : ℕ :=
  2 * (n.factorial * n.factorial)

theorem three_men_three_women_arrangements :
  alternating_arrangements 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_three_men_three_women_arrangements_l2370_237018


namespace NUMINAMATH_CALUDE_competition_problem_l2370_237069

/-- Represents the number of students who solved each combination of problems -/
structure ProblemSolvers where
  onlyA : ℕ
  onlyB : ℕ
  onlyC : ℕ
  AB : ℕ
  AC : ℕ
  BC : ℕ
  ABC : ℕ

/-- The theorem statement -/
theorem competition_problem (s : ProblemSolvers) : s.onlyB = 6 :=
  by
  have total : s.onlyA + s.onlyB + s.onlyC + s.AB + s.AC + s.BC + s.ABC = 25 := by sorry
  have solved_A : s.onlyA = s.AB + s.AC + s.ABC + 1 := by sorry
  have not_A_BC : s.onlyB + s.BC = 2 * (s.onlyC + s.BC) := by sorry
  have only_one_not_A : s.onlyB + s.onlyC = s.onlyA := by sorry
  sorry

end NUMINAMATH_CALUDE_competition_problem_l2370_237069


namespace NUMINAMATH_CALUDE_annual_growth_rate_l2370_237006

theorem annual_growth_rate (initial_value final_value : ℝ) (h1 : initial_value = 70400) 
  (h2 : final_value = 89100) : ∃ r : ℝ, initial_value * (1 + r)^2 = final_value ∧ r = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_annual_growth_rate_l2370_237006


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l2370_237072

theorem pasta_preference_ratio :
  let total_students : ℕ := 1000
  let lasagna_pref : ℕ := 300
  let manicotti_pref : ℕ := 200
  let ravioli_pref : ℕ := 150
  let spaghetti_pref : ℕ := 270
  let fettuccine_pref : ℕ := 80
  (spaghetti_pref : ℚ) / (manicotti_pref : ℚ) = 27 / 20 :=
by sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l2370_237072


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l2370_237010

def P : Set ℕ := {1, 2, 3, 4}

def Q : Set ℕ := {x : ℕ | 3 ≤ x ∧ x < 7}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l2370_237010


namespace NUMINAMATH_CALUDE_expand_expression_l2370_237035

theorem expand_expression (x : ℝ) : 24 * (3 * x - 4) = 72 * x - 96 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2370_237035


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l2370_237056

theorem gcd_of_squares_sum : Nat.gcd (123^2 + 235^2 + 347^2) (122^2 + 234^2 + 348^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l2370_237056


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l2370_237021

theorem abs_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ + 3| = 15 ∧ |x₂ + 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l2370_237021


namespace NUMINAMATH_CALUDE_NQ_passes_through_fixed_point_l2370_237092

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line l
def line (k p : ℝ) (x y : ℝ) : Prop := y = k*(x + p/2)

-- Define the intersection points M and N
def intersection_points (p k : ℝ) (M N : ℝ × ℝ) : Prop :=
  parabola p M.1 M.2 ∧ parabola p N.1 N.2 ∧
  line k p M.1 M.2 ∧ line k p N.1 N.2 ∧
  M ≠ N

-- Define the chord length condition
def chord_length_condition (p : ℝ) (M N : ℝ × ℝ) : Prop :=
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 16*15

-- Define the third intersection point Q
def third_intersection (p : ℝ) (M N Q : ℝ × ℝ) : Prop :=
  parabola p Q.1 Q.2 ∧ Q ≠ M ∧ Q ≠ N

-- Define point B
def point_B : ℝ × ℝ := (1, -1)

-- Define the condition that MQ passes through B
def MQ_through_B (M Q : ℝ × ℝ) : Prop :=
  (point_B.2 - M.2) * (Q.1 - M.1) = (Q.2 - M.2) * (point_B.1 - M.1)

-- Theorem statement
theorem NQ_passes_through_fixed_point (p k : ℝ) (M N Q : ℝ × ℝ) :
  p > 0 →
  k = 1/2 →
  intersection_points p k M N →
  chord_length_condition p M N →
  third_intersection p M N Q →
  MQ_through_B M Q →
  ∃ (fixed_point : ℝ × ℝ), fixed_point = (1, -4) ∧
    (fixed_point.2 - N.2) * (Q.1 - N.1) = (Q.2 - N.2) * (fixed_point.1 - N.1) :=
sorry

end NUMINAMATH_CALUDE_NQ_passes_through_fixed_point_l2370_237092


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2370_237025

theorem complex_number_quadrant (z : ℂ) (h : z * (1 - Complex.I) = 4 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2370_237025


namespace NUMINAMATH_CALUDE_fraction_integer_condition_l2370_237034

theorem fraction_integer_condition (p : ℕ+) :
  (↑p : ℚ) ∈ ({3, 5, 9, 35} : Set ℚ) ↔ ∃ (k : ℤ), k > 0 ∧ (3 * p + 25 : ℚ) / (2 * p - 5 : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_condition_l2370_237034


namespace NUMINAMATH_CALUDE_minimum_selling_price_for_profit_margin_l2370_237015

/-- The minimum selling price for a small refrigerator to maintain a 20% profit margin --/
theorem minimum_selling_price_for_profit_margin
  (average_sales : ℕ)
  (refrigerator_cost : ℝ)
  (shipping_fee : ℝ)
  (storefront_fee : ℝ)
  (repair_cost : ℝ)
  (profit_margin : ℝ)
  (h_average_sales : average_sales = 50)
  (h_refrigerator_cost : refrigerator_cost = 1200)
  (h_shipping_fee : shipping_fee = 20)
  (h_storefront_fee : storefront_fee = 10000)
  (h_repair_cost : repair_cost = 5000)
  (h_profit_margin : profit_margin = 0.2)
  : ∃ (x : ℝ), x ≥ 1824 ∧
    (average_sales : ℝ) * x - (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) ≥
    (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) * profit_margin ∧
    ∀ (y : ℝ), y < x →
      (average_sales : ℝ) * y - (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) <
      (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) * profit_margin :=
by sorry

end NUMINAMATH_CALUDE_minimum_selling_price_for_profit_margin_l2370_237015


namespace NUMINAMATH_CALUDE_blood_cell_count_l2370_237074

/-- Given two blood samples with a total of 7341 blood cells, where the first sample
    contains 4221 blood cells, prove that the second sample contains 3120 blood cells. -/
theorem blood_cell_count (total : ℕ) (first_sample : ℕ) (second_sample : ℕ) 
    (h1 : total = 7341)
    (h2 : first_sample = 4221)
    (h3 : total = first_sample + second_sample) : 
  second_sample = 3120 := by
  sorry

end NUMINAMATH_CALUDE_blood_cell_count_l2370_237074


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expressions_l2370_237017

theorem simplify_trigonometric_expressions :
  (∀ α : ℝ, (1 + Real.tan α ^ 2) * Real.cos α ^ 2 = 1) ∧
  (Real.sin (7 * π / 6) + Real.tan (5 * π / 4) = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expressions_l2370_237017
