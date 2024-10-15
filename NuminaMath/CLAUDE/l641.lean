import Mathlib

namespace NUMINAMATH_CALUDE_thirteenth_root_unity_product_l641_64198

theorem thirteenth_root_unity_product (w : ℂ) : w^13 = 1 → (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 2657205 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_root_unity_product_l641_64198


namespace NUMINAMATH_CALUDE_least_positive_integer_divisibility_l641_64159

theorem least_positive_integer_divisibility (n : ℕ) : 
  (n % 2 = 1) → (∃ (a : ℕ), a > 0 ∧ (55^n + a * 32^n) % 2001 = 0) → 
  (∃ (a : ℕ), a > 0 ∧ a ≤ 436 ∧ (55^n + a * 32^n) % 2001 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisibility_l641_64159


namespace NUMINAMATH_CALUDE_count_prime_digit_even_sum_integers_l641_64182

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is a three-digit integer
def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

-- Define a function to get the digits of a three-digit number
def getDigits (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  (hundreds, tens, ones)

-- Define the main theorem
theorem count_prime_digit_even_sum_integers :
  (∃ S : Finset ℕ, 
    (∀ n ∈ S, isThreeDigit n ∧ 
              let (d1, d2, d3) := getDigits n
              isPrime d1 ∧ isPrime d2 ∧ isPrime d3 ∧
              (d1 + d2 + d3) % 2 = 0) ∧
    S.card = 18) := by sorry

end NUMINAMATH_CALUDE_count_prime_digit_even_sum_integers_l641_64182


namespace NUMINAMATH_CALUDE_ratio_bounds_l641_64120

theorem ratio_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 5 - 3*a ≤ b) (h2 : b ≤ 4 - a) (h3 : Real.log b ≥ a) :
  e ≤ b/a ∧ b/a ≤ 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_bounds_l641_64120


namespace NUMINAMATH_CALUDE_integer_solutions_for_equation_l641_64196

theorem integer_solutions_for_equation : 
  {(x, y) : ℤ × ℤ | x^2 - y^4 = 2009} = {(45, 2), (45, -2), (-45, 2), (-45, -2)} :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_for_equation_l641_64196


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l641_64119

theorem max_value_expression (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (|7*a + 8*b - a*b| + |2*a + 8*b - 6*a*b|) / (a * Real.sqrt (1 + b^2)) ≤ 9 * Real.sqrt 2 :=
by sorry

theorem max_value_achievable :
  ∃ (a b : ℝ), a ≥ 1 ∧ b ≥ 1 ∧
  (|7*a + 8*b - a*b| + |2*a + 8*b - 6*a*b|) / (a * Real.sqrt (1 + b^2)) = 9 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l641_64119


namespace NUMINAMATH_CALUDE_curve_scaling_transformation_l641_64172

/-- Given a curve C that undergoes a scaling transformation,
    prove that the equation of the original curve is x^2/4 + 9y^2 = 1 -/
theorem curve_scaling_transformation (x y x' y' : ℝ) :
  (x' = 1/2 * x) →
  (y' = 3 * y) →
  (x'^2 + y'^2 = 1) →
  (x^2/4 + 9*y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_curve_scaling_transformation_l641_64172


namespace NUMINAMATH_CALUDE_initial_eggs_count_l641_64177

theorem initial_eggs_count (eggs_used : ℕ) (chickens : ℕ) (eggs_per_chicken : ℕ) (final_eggs : ℕ) : 
  eggs_used = 5 → chickens = 2 → eggs_per_chicken = 3 → final_eggs = 11 →
  ∃ initial_eggs : ℕ, initial_eggs = 10 ∧ initial_eggs - eggs_used + chickens * eggs_per_chicken = final_eggs :=
by
  sorry


end NUMINAMATH_CALUDE_initial_eggs_count_l641_64177


namespace NUMINAMATH_CALUDE_solve_equation_l641_64150

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((2 / x) + 3) = 5 / 3 → x = -9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l641_64150


namespace NUMINAMATH_CALUDE_angle_C_measure_l641_64168

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem angle_C_measure (t : Triangle) : 
  t.A = 39 * π / 180 ∧ 
  (t.a^2 - t.b^2) * (t.a^2 + t.a * t.c - t.b^2) = t.b^2 * t.c^2 ∧
  t.A + t.B + t.C = π →
  t.C = 115 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l641_64168


namespace NUMINAMATH_CALUDE_smallest_sports_team_size_l641_64141

theorem smallest_sports_team_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 1 ∧ 
  n % 4 = 2 ∧ 
  n % 6 = 4 ∧ 
  ∃ m : ℕ, n = m ^ 2 ∧
  ∀ k : ℕ, k > 0 → k % 3 = 1 → k % 4 = 2 → k % 6 = 4 → (∃ l : ℕ, k = l ^ 2) → k ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_sports_team_size_l641_64141


namespace NUMINAMATH_CALUDE_sheetrock_area_is_30_l641_64155

/-- Represents the area of a rectangular sheetrock given its length and width. -/
def sheetrockArea (length width : ℝ) : ℝ := length * width

/-- Theorem stating that the area of a rectangular sheetrock with length 6 feet and width 5 feet is 30 square feet. -/
theorem sheetrock_area_is_30 : sheetrockArea 6 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sheetrock_area_is_30_l641_64155


namespace NUMINAMATH_CALUDE_first_day_visitors_l641_64105

/-- Given the initial stock and restock amount, calculate the number of people who showed up on the first day -/
theorem first_day_visitors (initial_stock : ℕ) (first_restock : ℕ) (cans_per_person : ℕ) : 
  initial_stock = 2000 →
  first_restock = 1500 →
  cans_per_person = 1 →
  (initial_stock - first_restock) / cans_per_person = 500 := by
  sorry

#check first_day_visitors

end NUMINAMATH_CALUDE_first_day_visitors_l641_64105


namespace NUMINAMATH_CALUDE_dessert_preference_l641_64145

structure Classroom where
  total : ℕ
  apple : ℕ
  chocolate : ℕ
  pumpkin : ℕ
  none : ℕ

def likes_apple_and_chocolate_not_pumpkin (c : Classroom) : ℕ :=
  c.apple + c.chocolate - (c.total - c.none) - 2

theorem dessert_preference (c : Classroom) 
  (h_total : c.total = 50)
  (h_apple : c.apple = 25)
  (h_chocolate : c.chocolate = 20)
  (h_pumpkin : c.pumpkin = 10)
  (h_none : c.none = 16) :
  likes_apple_and_chocolate_not_pumpkin c = 9 := by
  sorry

end NUMINAMATH_CALUDE_dessert_preference_l641_64145


namespace NUMINAMATH_CALUDE_f_symmetry_l641_64127

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^7 + a*x^5 + b*x - 5

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b (-3) = 5 → f a b 3 = -15 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l641_64127


namespace NUMINAMATH_CALUDE_novelty_shop_costs_l641_64174

/-- Represents the cost of items in dollars -/
structure ItemCost where
  magazine : ℝ
  chocolate : ℝ
  candy : ℝ
  toy : ℝ

/-- The conditions given in the problem -/
def shopConditions (cost : ItemCost) : Prop :=
  cost.magazine = 1 ∧
  4 * cost.chocolate = 8 * cost.magazine ∧
  2 * cost.candy + 3 * cost.toy = 5 * cost.magazine

/-- The theorem stating the cost of a dozen chocolate bars and the indeterminacy of candy and toy costs -/
theorem novelty_shop_costs (cost : ItemCost) (h : shopConditions cost) :
  12 * cost.chocolate = 24 ∧
  ∃ (c t : ℝ), c ≠ cost.candy ∧ t ≠ cost.toy ∧ shopConditions { magazine := cost.magazine, chocolate := cost.chocolate, candy := c, toy := t } :=
by sorry

end NUMINAMATH_CALUDE_novelty_shop_costs_l641_64174


namespace NUMINAMATH_CALUDE_solution_set_theorem_l641_64188

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

-- State the theorem
theorem solution_set_theorem (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, f a b (-2*x) < 0 ↔ x < -3/2 ∨ 1/2 < x) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l641_64188


namespace NUMINAMATH_CALUDE_sin_half_and_third_max_solutions_l641_64186

open Real

theorem sin_half_and_third_max_solutions (α : ℝ) : 
  (∃ (s : Finset ℝ), (∀ x ∈ s, ∃ k : ℤ, (x = α/2 + k*π ∨ x = (π - α)/2 + k*π) ∧ sin x = sin α) ∧ s.card ≤ 4) ∧
  (∃ (t : Finset ℝ), (∀ x ∈ t, ∃ k : ℤ, x = α/3 + 2*k*π/3 ∧ sin x = sin α) ∧ t.card ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_sin_half_and_third_max_solutions_l641_64186


namespace NUMINAMATH_CALUDE_min_value_theorem_l641_64148

theorem min_value_theorem (x y a : ℝ) 
  (h1 : (x - 3)^3 + 2016 * (x - 3) = a) 
  (h2 : (2 * y - 3)^3 + 2016 * (2 * y - 3) = -a) : 
  ∃ (m : ℝ), m = 28 ∧ ∀ (x y : ℝ), x^2 + 4 * y^2 + 4 * x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l641_64148


namespace NUMINAMATH_CALUDE_area_equality_l641_64128

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the shapes
def is_convex_quadrilateral (A C D F : ℝ × ℝ) : Prop := sorry
def is_equilateral_triangle (A B E : ℝ × ℝ) : Prop := sorry
def is_square (A C D F : ℝ × ℝ) : Prop := sorry
def is_rectangle (A C D F : ℝ × ℝ) : Prop := sorry

-- Define the point on side condition
def point_on_side (P Q R : ℝ × ℝ) : Prop := sorry

-- Define the area calculation function
def area_triangle (P Q R : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem area_equality 
  (h_quad : is_convex_quadrilateral A C D F)
  (h_tri : is_equilateral_triangle A B E)
  (h_common : A = A)  -- Common vertex
  (h_B_on_CF : point_on_side B C F)
  (h_E_on_FD : point_on_side E F D)
  (h_shape : is_square A C D F ∨ is_rectangle A C D F) :
  area_triangle A D E + area_triangle A B C = area_triangle B E F := by
  sorry

end NUMINAMATH_CALUDE_area_equality_l641_64128


namespace NUMINAMATH_CALUDE_prob_third_draw_exactly_l641_64100

/-- Simple random sampling without replacement from a finite population -/
structure SimpleRandomSampling where
  population_size : ℕ
  sample_size : ℕ
  h_sample_size : sample_size ≤ population_size

/-- The probability of drawing a specific individual on the nth draw,
    given they were not drawn in the previous n-1 draws -/
def prob_draw_on_nth (srs : SimpleRandomSampling) (n : ℕ) : ℚ :=
  if n ≤ srs.sample_size
  then 1 / (srs.population_size - n + 1)
  else 0

/-- The probability of not drawing a specific individual on the nth draw,
    given they were not drawn in the previous n-1 draws -/
def prob_not_draw_on_nth (srs : SimpleRandomSampling) (n : ℕ) : ℚ :=
  if n ≤ srs.sample_size
  then (srs.population_size - n) / (srs.population_size - n + 1)
  else 1

theorem prob_third_draw_exactly
  (srs : SimpleRandomSampling)
  (h : srs.population_size = 6 ∧ srs.sample_size = 3) :
  prob_not_draw_on_nth srs 1 * prob_not_draw_on_nth srs 2 * prob_draw_on_nth srs 3 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_third_draw_exactly_l641_64100


namespace NUMINAMATH_CALUDE_photocopy_savings_theorem_l641_64167

/-- Represents the cost structure for photocopies -/
structure CostStructure where
  base_cost : Real
  color_cost : Real
  double_sided_cost : Real
  discount_tier1 : Real
  discount_tier2 : Real
  discount_tier3 : Real

/-- Represents an order of photocopies -/
structure Order where
  bw_one_sided : Nat
  bw_double_sided : Nat
  color_one_sided : Nat
  color_double_sided : Nat

/-- Calculates the cost of an order without discount -/
def orderCost (cs : CostStructure) (o : Order) : Real := sorry

/-- Calculates the discount percentage based on the total number of copies -/
def discountPercentage (cs : CostStructure) (total_copies : Nat) : Real := sorry

/-- Calculates the total cost of combined orders with discount -/
def combinedOrderCost (cs : CostStructure) (o1 o2 : Order) : Real := sorry

/-- Calculates the savings when combining two orders -/
def savings (cs : CostStructure) (o1 o2 : Order) : Real := sorry

theorem photocopy_savings_theorem (cs : CostStructure) (steve_order dennison_order : Order) :
  cs.base_cost = 0.02 ∧
  cs.color_cost = 0.08 ∧
  cs.double_sided_cost = 0.03 ∧
  cs.discount_tier1 = 0.1 ∧
  cs.discount_tier2 = 0.2 ∧
  cs.discount_tier3 = 0.3 ∧
  steve_order.bw_one_sided = 35 ∧
  steve_order.bw_double_sided = 25 ∧
  steve_order.color_one_sided = 0 ∧
  steve_order.color_double_sided = 15 ∧
  dennison_order.bw_one_sided = 20 ∧
  dennison_order.bw_double_sided = 40 ∧
  dennison_order.color_one_sided = 12 ∧
  dennison_order.color_double_sided = 0 →
  savings cs steve_order dennison_order = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_photocopy_savings_theorem_l641_64167


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l641_64140

theorem algebraic_expression_equality : 
  Real.sqrt (5 - 2 * Real.sqrt 6) + Real.sqrt (7 - 4 * Real.sqrt 3) = 2 - Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l641_64140


namespace NUMINAMATH_CALUDE_total_desks_is_1776_total_desks_within_capacity_l641_64104

/-- Represents the total number of classrooms in the school. -/
def total_classrooms : ℕ := 50

/-- Represents the number of desks in classrooms of type 1. -/
def desks_type1 : ℕ := 45

/-- Represents the number of desks in classrooms of type 2. -/
def desks_type2 : ℕ := 38

/-- Represents the number of desks in classrooms of type 3. -/
def desks_type3 : ℕ := 32

/-- Represents the number of desks in classrooms of type 4. -/
def desks_type4 : ℕ := 25

/-- Represents the fraction of classrooms of type 1. -/
def fraction_type1 : ℚ := 3 / 10

/-- Represents the fraction of classrooms of type 2. -/
def fraction_type2 : ℚ := 1 / 4

/-- Represents the fraction of classrooms of type 3. -/
def fraction_type3 : ℚ := 1 / 5

/-- Represents the maximum student capacity allowed by regulations. -/
def max_capacity : ℕ := 1800

/-- Theorem stating that the total number of desks in the school is 1776. -/
theorem total_desks_is_1776 : 
  (↑total_classrooms * fraction_type1).floor * desks_type1 +
  (↑total_classrooms * fraction_type2).floor * desks_type2 +
  (↑total_classrooms * fraction_type3).floor * desks_type3 +
  (total_classrooms - 
    (↑total_classrooms * fraction_type1).floor - 
    (↑total_classrooms * fraction_type2).floor - 
    (↑total_classrooms * fraction_type3).floor) * desks_type4 = 1776 :=
by sorry

/-- Theorem stating that the total number of desks does not exceed the maximum capacity. -/
theorem total_desks_within_capacity : 
  (↑total_classrooms * fraction_type1).floor * desks_type1 +
  (↑total_classrooms * fraction_type2).floor * desks_type2 +
  (↑total_classrooms * fraction_type3).floor * desks_type3 +
  (total_classrooms - 
    (↑total_classrooms * fraction_type1).floor - 
    (↑total_classrooms * fraction_type2).floor - 
    (↑total_classrooms * fraction_type3).floor) * desks_type4 ≤ max_capacity :=
by sorry

end NUMINAMATH_CALUDE_total_desks_is_1776_total_desks_within_capacity_l641_64104


namespace NUMINAMATH_CALUDE_dance_lesson_cost_l641_64136

/-- Calculates the total cost of dance lessons given the number of lessons,
    cost per lesson, and number of free lessons. -/
def total_cost (total_lessons : ℕ) (cost_per_lesson : ℕ) (free_lessons : ℕ) : ℕ :=
  (total_lessons - free_lessons) * cost_per_lesson

/-- Theorem stating that given 10 dance lessons costing $10 each,
    with 2 lessons for free, the total cost is $80. -/
theorem dance_lesson_cost :
  total_cost 10 10 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_dance_lesson_cost_l641_64136


namespace NUMINAMATH_CALUDE_five_eighths_of_twelve_fifths_l641_64113

theorem five_eighths_of_twelve_fifths : (5 / 8 : ℚ) * (12 / 5 : ℚ) = (3 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_five_eighths_of_twelve_fifths_l641_64113


namespace NUMINAMATH_CALUDE_gcf_lcm_40_120_100_l641_64197

theorem gcf_lcm_40_120_100 :
  (let a := 40
   let b := 120
   let c := 100
   (Nat.gcd a (Nat.gcd b c) = 20) ∧
   (Nat.lcm a (Nat.lcm b c) = 600)) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_40_120_100_l641_64197


namespace NUMINAMATH_CALUDE_existence_of_solution_l641_64162

theorem existence_of_solution : ∃ (x y : ℕ), x^99 = 2013 * y^100 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l641_64162


namespace NUMINAMATH_CALUDE_circles_product_radii_equals_sum_squares_l641_64134

/-- Given two circles passing through a point M(x₁, y₁) and tangent to both the x-axis and y-axis
    with radii r₁ and r₂, the product of their radii equals the sum of squares of the coordinates of M. -/
theorem circles_product_radii_equals_sum_squares (x₁ y₁ r₁ r₂ : ℝ) 
    (h1 : ∃ (a₁ b₁ : ℝ), (x₁ - a₁)^2 + (y₁ - b₁)^2 = r₁^2 ∧ |a₁| = r₁ ∧ |b₁| = r₁)
    (h2 : ∃ (a₂ b₂ : ℝ), (x₁ - a₂)^2 + (y₁ - b₂)^2 = r₂^2 ∧ |a₂| = r₂ ∧ |b₂| = r₂) :
  r₁ * r₂ = x₁^2 + y₁^2 := by
  sorry

end NUMINAMATH_CALUDE_circles_product_radii_equals_sum_squares_l641_64134


namespace NUMINAMATH_CALUDE_coordinate_translation_l641_64101

/-- Given a translation of the coordinate system where point A moves from (-1, 3) to (-3, -1),
    prove that the new origin O' has coordinates (2, 4). -/
theorem coordinate_translation (A_old A_new O'_new : ℝ × ℝ) : 
  A_old = (-1, 3) → A_new = (-3, -1) → O'_new = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_coordinate_translation_l641_64101


namespace NUMINAMATH_CALUDE_quadratic_shift_and_roots_l641_64116

/-- A quadratic function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_shift_and_roots (a b c : ℝ) (h : a > 0) :
  (∀ k > 0, ∀ x, quadratic a b (c - k) x < quadratic a b c x) ∧
  (∀ x, quadratic a b c x ≠ 0 →
    ∃ k > 0, ∃ x, quadratic a b (c - k) x = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_and_roots_l641_64116


namespace NUMINAMATH_CALUDE_sophie_bought_four_boxes_l641_64123

/-- The number of boxes of donuts Sophie bought -/
def boxes_bought : ℕ := sorry

/-- The number of donuts in each box -/
def donuts_per_box : ℕ := 12

/-- The number of boxes Sophie gave to her mom -/
def boxes_to_mom : ℕ := 1

/-- The number of donuts Sophie gave to her sister -/
def donuts_to_sister : ℕ := 6

/-- The number of donuts Sophie had left for herself -/
def donuts_left : ℕ := 30

theorem sophie_bought_four_boxes : boxes_bought = 4 := by sorry

end NUMINAMATH_CALUDE_sophie_bought_four_boxes_l641_64123


namespace NUMINAMATH_CALUDE_a_range_when_A_union_B_is_R_A_union_B_is_R_when_a_in_range_l641_64171

/-- The set A defined by the inequality (x - 1)(x - a) ≥ 0 -/
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}

/-- The set B defined by the inequality x ≥ a - 1 -/
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

/-- Theorem stating that if A ∪ B = ℝ, then a ∈ (-∞, 2] -/
theorem a_range_when_A_union_B_is_R (a : ℝ) 
  (h : A a ∪ B a = Set.univ) : a ≤ 2 := by
  sorry

/-- Theorem stating that if a ∈ (-∞, 2], then A ∪ B = ℝ -/
theorem A_union_B_is_R_when_a_in_range (a : ℝ) 
  (h : a ≤ 2) : A a ∪ B a = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_a_range_when_A_union_B_is_R_A_union_B_is_R_when_a_in_range_l641_64171


namespace NUMINAMATH_CALUDE_triangle_side_sum_bound_l641_64137

/-- Given a triangle ABC with side lengths a, b, and c, where c = 2 and the dot product 
    of vectors AC and AB is equal to b² - (1/2)ab, prove that 2 < a + b ≤ 4 -/
theorem triangle_side_sum_bound (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let c : ℝ := 2
  let dot_product : ℝ := b^2 - (1/2) * a * b
  2 < a + b ∧ a + b ≤ 4 := by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_bound_l641_64137


namespace NUMINAMATH_CALUDE_height_prediction_age_10_l641_64130

/-- Regression model for height prediction -/
def height_model (age : ℝ) : ℝ := 7.19 * age + 73.93

/-- The predicted height at age 10 is approximately 145.83 cm -/
theorem height_prediction_age_10 :
  ∃ ε > 0, abs (height_model 10 - 145.83) < ε :=
sorry

end NUMINAMATH_CALUDE_height_prediction_age_10_l641_64130


namespace NUMINAMATH_CALUDE_negation_of_implication_l641_64146

theorem negation_of_implication (P Q : Prop) :
  ¬(P → Q) ↔ (¬P → ¬Q) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l641_64146


namespace NUMINAMATH_CALUDE_margin_expression_l641_64165

/-- Given a selling price S, a ratio m, and a cost C, prove that the margin M
    can be expressed as (1/m)S. -/
theorem margin_expression (S m : ℝ) (h_m : m ≠ 0) :
  let M := (1 / m) * S
  let C := S - M
  M = (1 / m) * S := by sorry

end NUMINAMATH_CALUDE_margin_expression_l641_64165


namespace NUMINAMATH_CALUDE_percent_y_of_x_l641_64132

theorem percent_y_of_x (x y : ℝ) (h : 0.2 * (x - y) = 0.15 * (x + y)) : 
  y = (100 / 7) * x / 100 := by
  sorry

end NUMINAMATH_CALUDE_percent_y_of_x_l641_64132


namespace NUMINAMATH_CALUDE_min_speed_to_arrive_first_l641_64178

/-- Proves the minimum speed required for the second person to arrive first -/
theorem min_speed_to_arrive_first (distance : ℝ) (speed_A : ℝ) (head_start : ℝ) 
  (h1 : distance = 180)
  (h2 : speed_A = 40)
  (h3 : head_start = 0.5)
  (h4 : speed_A > 0) : 
  ∃ (min_speed : ℝ), min_speed > 45 ∧ 
    ∀ (speed_B : ℝ), speed_B > min_speed → 
      distance / speed_B < distance / speed_A - head_start := by
sorry

end NUMINAMATH_CALUDE_min_speed_to_arrive_first_l641_64178


namespace NUMINAMATH_CALUDE_percentage_of_long_term_employees_l641_64135

/-- Represents the number of employees in each year range at the Pythagoras company -/
structure EmployeeDistribution where
  less_than_1_year : ℕ
  one_to_2_years : ℕ
  two_to_3_years : ℕ
  three_to_4_years : ℕ
  four_to_5_years : ℕ
  five_to_6_years : ℕ
  six_to_7_years : ℕ
  seven_to_8_years : ℕ
  eight_to_9_years : ℕ
  nine_to_10_years : ℕ
  ten_to_11_years : ℕ
  eleven_to_12_years : ℕ
  twelve_to_13_years : ℕ
  thirteen_to_14_years : ℕ
  fourteen_to_15_years : ℕ

/-- Calculates the total number of employees -/
def totalEmployees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1_year + d.one_to_2_years + d.two_to_3_years + d.three_to_4_years +
  d.four_to_5_years + d.five_to_6_years + d.six_to_7_years + d.seven_to_8_years +
  d.eight_to_9_years + d.nine_to_10_years + d.ten_to_11_years + d.eleven_to_12_years +
  d.twelve_to_13_years + d.thirteen_to_14_years + d.fourteen_to_15_years

/-- Calculates the number of employees who have worked for 10 years or more -/
def employeesWithTenYearsOrMore (d : EmployeeDistribution) : ℕ :=
  d.ten_to_11_years + d.eleven_to_12_years + d.twelve_to_13_years +
  d.thirteen_to_14_years + d.fourteen_to_15_years

/-- Theorem: The percentage of employees who have worked at the Pythagoras company for 10 years or more is 15% -/
theorem percentage_of_long_term_employees (d : EmployeeDistribution)
  (h : d = { less_than_1_year := 4, one_to_2_years := 6, two_to_3_years := 7,
             three_to_4_years := 4, four_to_5_years := 3, five_to_6_years := 3,
             six_to_7_years := 2, seven_to_8_years := 2, eight_to_9_years := 1,
             nine_to_10_years := 1, ten_to_11_years := 2, eleven_to_12_years := 1,
             twelve_to_13_years := 1, thirteen_to_14_years := 1, fourteen_to_15_years := 1 }) :
  (employeesWithTenYearsOrMore d : ℚ) / (totalEmployees d : ℚ) = 15 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_long_term_employees_l641_64135


namespace NUMINAMATH_CALUDE_r_and_s_earnings_l641_64156

/-- The daily earnings of individuals p, q, r, and s --/
structure DailyEarnings where
  p : ℚ
  q : ℚ
  r : ℚ
  s : ℚ

/-- The conditions given in the problem --/
def problem_conditions (e : DailyEarnings) : Prop :=
  e.p + e.q + e.r + e.s = 2380 / 9 ∧
  e.p + e.r = 600 / 5 ∧
  e.q + e.s = 800 / 6 ∧
  e.q + e.r = 910 / 7 ∧
  e.p = 150 / 3

/-- The theorem stating that r and s together earn 430/3 Rs per day --/
theorem r_and_s_earnings (e : DailyEarnings) :
  problem_conditions e → e.r + e.s = 430 / 3 := by
  sorry

#check r_and_s_earnings

end NUMINAMATH_CALUDE_r_and_s_earnings_l641_64156


namespace NUMINAMATH_CALUDE_emily_days_off_l641_64157

/-- The number of holidays Emily took in a year -/
def total_holidays : ℕ := 24

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of times Emily took a day off each month -/
def days_off_per_month : ℚ := total_holidays / months_in_year

theorem emily_days_off : days_off_per_month = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_days_off_l641_64157


namespace NUMINAMATH_CALUDE_rectangle_area_l641_64166

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) :
  square_area = 1225 →
  rectangle_breadth = 10 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := circle_radius / 4
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l641_64166


namespace NUMINAMATH_CALUDE_min_value_a_l641_64169

theorem min_value_a (x y z : ℝ) (h : x^2 + 4*y^2 + z^2 = 6) :
  (∃ (a : ℝ), ∀ (x y z : ℝ), x^2 + 4*y^2 + z^2 = 6 → x + 2*y + 3*z ≤ a) ∧
  (∀ (b : ℝ), (∀ (x y z : ℝ), x^2 + 4*y^2 + z^2 = 6 → x + 2*y + 3*z ≤ b) → Real.sqrt 66 ≤ b) :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l641_64169


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l641_64152

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + 5 * x - 2 > 0) ↔ (1/2 < x ∧ x < b)) → 
  (a = -2 ∧ b = 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l641_64152


namespace NUMINAMATH_CALUDE_tomato_plants_per_row_is_eight_l641_64195

/-- Represents the garden planting scenario -/
structure GardenPlanting where
  cucumber_to_tomato_ratio : ℚ
  total_rows : ℕ
  tomatoes_per_plant : ℕ
  total_tomatoes : ℕ

/-- Calculates the number of tomato plants per row -/
def tomato_plants_per_row (g : GardenPlanting) : ℚ :=
  g.total_tomatoes / (g.tomatoes_per_plant * (g.total_rows / (1 + g.cucumber_to_tomato_ratio)))

/-- Theorem stating that the number of tomato plants per row is 8 -/
theorem tomato_plants_per_row_is_eight (g : GardenPlanting) 
  (h1 : g.cucumber_to_tomato_ratio = 2)
  (h2 : g.total_rows = 15)
  (h3 : g.tomatoes_per_plant = 3)
  (h4 : g.total_tomatoes = 120) : 
  tomato_plants_per_row g = 8 := by
  sorry

end NUMINAMATH_CALUDE_tomato_plants_per_row_is_eight_l641_64195


namespace NUMINAMATH_CALUDE_three_color_plane_coloring_l641_64193

-- Define a type for colors
inductive Color
| Red
| Green
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for lines in the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define a predicate to check if a point is on a line
def IsOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a predicate to check if a line contains at most two colors
def LineContainsAtMostTwoColors (coloring : Coloring) (l : Line) : Prop :=
  ∃ (c1 c2 : Color), ∀ (p : Point), IsOnLine p l → coloring p = c1 ∨ coloring p = c2

-- Define a predicate to check if all three colors are used
def AllColorsUsed (coloring : Coloring) : Prop :=
  (∃ (p : Point), coloring p = Color.Red) ∧
  (∃ (p : Point), coloring p = Color.Green) ∧
  (∃ (p : Point), coloring p = Color.Blue)

-- Theorem statement
theorem three_color_plane_coloring :
  ∃ (coloring : Coloring),
    (∀ (l : Line), LineContainsAtMostTwoColors coloring l) ∧
    AllColorsUsed coloring :=
by
  sorry

end NUMINAMATH_CALUDE_three_color_plane_coloring_l641_64193


namespace NUMINAMATH_CALUDE_sin_cos_sum_equal_shifted_cos_l641_64191

theorem sin_cos_sum_equal_shifted_cos (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.cos (3 * x - π / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equal_shifted_cos_l641_64191


namespace NUMINAMATH_CALUDE_square_of_binomial_form_l641_64164

theorem square_of_binomial_form (x y : ℝ) :
  ∃ (a b : ℝ), (1/3 * x + y) * (y - 1/3 * x) = (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_form_l641_64164


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l641_64185

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 3rd term is 5 and the 7th term is 29, the 10th term is 47. -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h_arith : ArithmeticSequence a) 
  (h_3rd : a 3 = 5)
  (h_7th : a 7 = 29) : 
  a 10 = 47 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l641_64185


namespace NUMINAMATH_CALUDE_estimate_fish_population_l641_64122

/-- Estimates the number of fish in a lake using the mark-recapture method. -/
theorem estimate_fish_population (n m k : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) (h4 : k ≤ m) (h5 : k ≤ n) :
  (n * m : ℚ) / k = (m : ℚ) / (k : ℚ) * n :=
by sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l641_64122


namespace NUMINAMATH_CALUDE_jennys_bottle_cap_bounce_fraction_l641_64190

theorem jennys_bottle_cap_bounce_fraction :
  ∀ (jenny_initial : ℝ) (mark_initial : ℝ) (jenny_fraction : ℝ),
    jenny_initial = 18 →
    mark_initial = 15 →
    (mark_initial + 2 * mark_initial) - (jenny_initial + jenny_initial * jenny_fraction) = 21 →
    jenny_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_jennys_bottle_cap_bounce_fraction_l641_64190


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l641_64111

/-- Function to replace 2s with 5s and 5s with 2s in a number -/
def replaceDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is a 5-digit odd number -/
def isFiveDigitOdd (n : ℕ) : Prop := sorry

theorem unique_number_satisfying_conditions :
  ∀ x y : ℕ,
    isFiveDigitOdd x →
    y = replaceDigits x →
    y = 2 * (x + 1) →
    x = 29995 := by sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l641_64111


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l641_64106

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l641_64106


namespace NUMINAMATH_CALUDE_arccos_less_than_arctan_in_interval_l641_64170

theorem arccos_less_than_arctan_in_interval :
  ∀ x : ℝ, 0.5 < x ∧ x ≤ 1 → Real.arccos x < Real.arctan x := by
  sorry

end NUMINAMATH_CALUDE_arccos_less_than_arctan_in_interval_l641_64170


namespace NUMINAMATH_CALUDE_total_money_earned_l641_64183

def clementine_cookies : ℕ := 72
def jake_cookies : ℕ := 2 * clementine_cookies
def combined_cookies : ℕ := jake_cookies + clementine_cookies
def tory_cookies : ℕ := combined_cookies / 2
def total_cookies : ℕ := clementine_cookies + jake_cookies + tory_cookies
def price_per_cookie : ℕ := 2

theorem total_money_earned : total_cookies * price_per_cookie = 648 := by
  sorry

end NUMINAMATH_CALUDE_total_money_earned_l641_64183


namespace NUMINAMATH_CALUDE_overtime_hours_is_eight_l641_64139

/-- Calculates overtime hours given regular pay rate, regular hours, overtime rate multiplier, and total pay -/
def calculate_overtime_hours (regular_rate : ℚ) (regular_hours : ℚ) (overtime_multiplier : ℚ) (total_pay : ℚ) : ℚ :=
  let regular_pay := regular_rate * regular_hours
  let overtime_rate := regular_rate * overtime_multiplier
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate

/-- Proves that given the problem conditions, the number of overtime hours is 8 -/
theorem overtime_hours_is_eight :
  let regular_rate : ℚ := 3
  let regular_hours : ℚ := 40
  let overtime_multiplier : ℚ := 2
  let total_pay : ℚ := 168
  calculate_overtime_hours regular_rate regular_hours overtime_multiplier total_pay = 8 := by
  sorry

#eval calculate_overtime_hours 3 40 2 168

end NUMINAMATH_CALUDE_overtime_hours_is_eight_l641_64139


namespace NUMINAMATH_CALUDE_division_problem_l641_64108

theorem division_problem (N : ℕ) (n : ℕ) (h1 : N > 0) :
  (∀ k : ℕ, k ≤ n → ∃ part : ℚ, part = N / (k * (k + 1))) →
  (N / (n * (n + 1)) = N / 400) →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l641_64108


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l641_64107

/-- A point on a parabola with a specific distance to the focus -/
structure ParabolaPoint where
  y : ℝ
  x : ℝ
  parabola_eq : x = 4 * y^2
  focus_distance : Real.sqrt ((x - 1/4)^2 + y^2) = 1/2

/-- The x-coordinate of a point on a parabola with a specific distance to the focus -/
theorem parabola_point_x_coordinate (M : ParabolaPoint) : M.x = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l641_64107


namespace NUMINAMATH_CALUDE_seventh_roots_of_unity_product_l641_64179

theorem seventh_roots_of_unity_product (z : ℂ) (h : z = Complex.exp (2 * Real.pi * Complex.I / 7)) :
  (3 - z) * (3 - z^2) * (3 - z^3) * (3 - z^4) * (3 - z^5) * (3 - z^6) = 1093 := by
  sorry

end NUMINAMATH_CALUDE_seventh_roots_of_unity_product_l641_64179


namespace NUMINAMATH_CALUDE_fraction_subtraction_addition_l641_64180

theorem fraction_subtraction_addition : 
  (1 : ℚ) / 12 - 5 / 6 + 1 / 3 = -5 / 12 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_addition_l641_64180


namespace NUMINAMATH_CALUDE_rationalize_and_sum_l641_64176

theorem rationalize_and_sum (a b c d s f : ℚ) (p q r : ℕ) :
  let x := (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 6 + Real.sqrt 8)
  let y := (a * Real.sqrt p + b * Real.sqrt q + c * Real.sqrt r + d * Real.sqrt s) / f
  (∃ (a b c d s : ℚ) (p q r : ℕ) (f : ℚ), 
    f > 0 ∧ 
    x = y ∧
    (p = 5 ∧ q = 6 ∧ r = 2 ∧ s = 1) ∧
    (a = 9 ∧ b = 7 ∧ c = -18 ∧ d = 0)) →
  a + b + c + d + s + f = 111 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_and_sum_l641_64176


namespace NUMINAMATH_CALUDE_min_value_abc_l641_64115

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 → (a + b) / (a * b * c) ≤ (x + y) / (x * y * z)) →
  (a + b) / (a * b * c) = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_abc_l641_64115


namespace NUMINAMATH_CALUDE_doodads_for_thingamabobs_l641_64144

/-- The number of doodads required to make one widget -/
def doodads_per_widget : ℚ := 18 / 5

/-- The number of widgets required to make one thingamabob -/
def widgets_per_thingamabob : ℚ := 11 / 4

/-- The number of thingamabobs we want to make -/
def target_thingamabobs : ℕ := 80

/-- Theorem stating that 792 doodads are required to make 80 thingamabobs -/
theorem doodads_for_thingamabobs : 
  ⌈(target_thingamabobs : ℚ) * widgets_per_thingamabob * doodads_per_widget⌉ = 792 := by
  sorry

end NUMINAMATH_CALUDE_doodads_for_thingamabobs_l641_64144


namespace NUMINAMATH_CALUDE_extra_flowers_l641_64158

theorem extra_flowers (tulips roses used : ℕ) : 
  tulips = 4 → roses = 11 → used = 11 → tulips + roses - used = 4 := by
  sorry

end NUMINAMATH_CALUDE_extra_flowers_l641_64158


namespace NUMINAMATH_CALUDE_inequality_proof_l641_64160

theorem inequality_proof (a b c d : ℝ) (h1 : d ≥ 0) (h2 : a + b = 2) (h3 : c + d = 2) :
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l641_64160


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l641_64194

theorem complex_modulus_problem (i : ℂ) (a : ℝ) :
  i^2 = -1 →
  (∃ (b : ℝ), (2 - i) / (a + i) = b * i) →
  Complex.abs ((2 * a + 1) + Real.sqrt 2 * i) = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l641_64194


namespace NUMINAMATH_CALUDE_hostel_accommodation_l641_64154

/-- Proves that 20 additional students were accommodated in the hostel --/
theorem hostel_accommodation :
  ∀ (initial_students : ℕ) 
    (initial_avg_expenditure : ℚ)
    (avg_decrease : ℚ)
    (total_increase : ℕ)
    (new_total_expenditure : ℕ),
  initial_students = 100 →
  avg_decrease = 5 →
  total_increase = 400 →
  new_total_expenditure = 5400 →
  ∃ (additional_students : ℕ),
    additional_students = 20 ∧
    (initial_avg_expenditure - avg_decrease) * (initial_students + additional_students) = new_total_expenditure :=
by sorry

end NUMINAMATH_CALUDE_hostel_accommodation_l641_64154


namespace NUMINAMATH_CALUDE_marble_color_convergence_l641_64117

/-- Represents the number of marbles of each color -/
structure MarbleState :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)

/-- The total number of marbles -/
def totalMarbles : ℕ := 2015

/-- Possible operations on the marble state -/
inductive MarbleOperation
  | RedGreenToBlue
  | RedBlueToGreen
  | GreenBlueToRed

/-- Apply a marble operation to a state -/
def applyOperation (state : MarbleState) (op : MarbleOperation) : MarbleState :=
  match op with
  | MarbleOperation.RedGreenToBlue => 
      { red := state.red - 1, green := state.green - 1, blue := state.blue + 2 }
  | MarbleOperation.RedBlueToGreen => 
      { red := state.red - 1, green := state.green + 2, blue := state.blue - 1 }
  | MarbleOperation.GreenBlueToRed => 
      { red := state.red + 2, green := state.green - 1, blue := state.blue - 1 }

/-- Check if all marbles are the same color -/
def allSameColor (state : MarbleState) : Prop :=
  (state.red = totalMarbles ∧ state.green = 0 ∧ state.blue = 0) ∨
  (state.red = 0 ∧ state.green = totalMarbles ∧ state.blue = 0) ∨
  (state.red = 0 ∧ state.green = 0 ∧ state.blue = totalMarbles)

/-- The main theorem to prove -/
theorem marble_color_convergence 
  (initial : MarbleState) 
  (h_total : initial.red + initial.green + initial.blue = totalMarbles) :
  ∃ (operations : List MarbleOperation), 
    allSameColor (operations.foldl applyOperation initial) :=
sorry

end NUMINAMATH_CALUDE_marble_color_convergence_l641_64117


namespace NUMINAMATH_CALUDE_expression_simplification_l641_64161

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (x + 1) / (x^2 + 2*x + 1) / (1 - 2 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l641_64161


namespace NUMINAMATH_CALUDE_water_tower_height_l641_64124

/-- Given a bamboo pole and a water tower under the same lighting conditions,
    this theorem proves the height of the water tower based on the similar triangles concept. -/
theorem water_tower_height
  (bamboo_height : ℝ)
  (bamboo_shadow : ℝ)
  (tower_shadow : ℝ)
  (h_bamboo_height : bamboo_height = 2)
  (h_bamboo_shadow : bamboo_shadow = 1.5)
  (h_tower_shadow : tower_shadow = 24) :
  bamboo_height / bamboo_shadow * tower_shadow = 32 :=
by sorry

end NUMINAMATH_CALUDE_water_tower_height_l641_64124


namespace NUMINAMATH_CALUDE_monotone_increasing_interval_minimum_m_for_inequality_l641_64138

noncomputable section

def f (m : ℝ) (x : ℝ) := Real.log x - m * x^2
def g (m : ℝ) (x : ℝ) := (1/2) * m * x^2 + x

theorem monotone_increasing_interval (x : ℝ) :
  StrictMonoOn (f (1/2)) (Set.Ioo 0 1) := by sorry

theorem minimum_m_for_inequality :
  ∀ m : ℕ, (∀ x : ℝ, x > 0 → f m x + g m x ≤ m * x - 1) →
  m ≥ 2 := by sorry

end

end NUMINAMATH_CALUDE_monotone_increasing_interval_minimum_m_for_inequality_l641_64138


namespace NUMINAMATH_CALUDE_vector_subtraction_l641_64173

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (5, 3) → b = (1, -2) → a - 2 • b = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l641_64173


namespace NUMINAMATH_CALUDE_cube_root_simplification_l641_64199

theorem cube_root_simplification :
  (20^3 + 30^3 + 40^3 + 8000)^(1/3) = 8 * (1500)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l641_64199


namespace NUMINAMATH_CALUDE_student_number_problem_l641_64175

theorem student_number_problem (x : ℤ) : (8 * x - 138 = 102) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l641_64175


namespace NUMINAMATH_CALUDE_attendants_with_both_tools_l641_64126

theorem attendants_with_both_tools (pencil_users : ℕ) (pen_users : ℕ) (single_tool_users : ℕ) : 
  pencil_users = 25 →
  pen_users = 15 →
  single_tool_users = 20 →
  pencil_users + pen_users - single_tool_users = 10 := by
sorry

end NUMINAMATH_CALUDE_attendants_with_both_tools_l641_64126


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l641_64192

theorem function_value_at_negative_one (f : ℝ → ℝ) 
  (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x = x^2 + 2 * (deriv f 2) * x + 3) : 
  f (-1) = 12 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l641_64192


namespace NUMINAMATH_CALUDE_units_digit_pow2_2010_l641_64129

/-- The units digit of 2^n for n ≥ 1 -/
def units_digit_pow2 (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 0 => 6
  | _ => 0  -- This case should never occur

/-- The units digit of 2^2010 is 4 -/
theorem units_digit_pow2_2010 : units_digit_pow2 2010 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_pow2_2010_l641_64129


namespace NUMINAMATH_CALUDE_domain_of_shifted_sum_l641_64110

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc 0 4

-- State the theorem
theorem domain_of_shifted_sum (hf : Set.range f = dom_f) :
  {x : ℝ | ∃ y, y ∈ dom_f ∧ x + 1 = y} ∩ {x : ℝ | ∃ y, y ∈ dom_f ∧ x - 1 = y} = Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_domain_of_shifted_sum_l641_64110


namespace NUMINAMATH_CALUDE_right_triangular_prism_dimension_l641_64151

/-- 
Given a right triangular prism with:
- base edges a = 5 and b = 12
- height c = 13
- body diagonal d = 15
Prove that the third dimension of a rectangular face (h) is equal to 2√14
-/
theorem right_triangular_prism_dimension (a b c d h : ℝ) : 
  a = 5 → b = 12 → c = 13 → d = 15 →
  a^2 + b^2 = c^2 →
  d^2 = a^2 + b^2 + h^2 →
  h = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_prism_dimension_l641_64151


namespace NUMINAMATH_CALUDE_roots_equation_l641_64133

theorem roots_equation (A B a b c d : ℝ) : 
  (a^2 + A*a + 1 = 0) → 
  (b^2 + A*b + 1 = 0) → 
  (c^2 + B*c + 1 = 0) → 
  (d^2 + B*d + 1 = 0) → 
  (a - c)*(b - c)*(a + d)*(b + d) = B^2 - A^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_l641_64133


namespace NUMINAMATH_CALUDE_license_plate_count_l641_64109

/-- The number of consonants in the English alphabet (excluding Y) -/
def num_consonants : ℕ := 20

/-- The number of vowels in the English alphabet (including Y) -/
def num_vowels : ℕ := 6

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of special symbols allowed (!, #, $) -/
def num_special_symbols : ℕ := 3

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits * num_special_symbols

theorem license_plate_count : total_license_plates = 72000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l641_64109


namespace NUMINAMATH_CALUDE_tangent_lines_imply_a_range_l641_64114

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.sqrt x

def has_two_tangent_lines (f g : ℝ → ℝ) : Prop :=
  ∃ (l₁ l₂ : ℝ → ℝ), l₁ ≠ l₂ ∧
    (∃ (x₁ : ℝ), l₁ x₁ = f x₁ ∧ (∀ x, l₁ x ≤ f x)) ∧
    (∃ (x₂ : ℝ), l₂ x₂ = f x₂ ∧ (∀ x, l₂ x ≤ f x)) ∧
    (∃ (y₁ : ℝ), l₁ y₁ = g y₁ ∧ (∀ y, l₁ y ≤ g y)) ∧
    (∃ (y₂ : ℝ), l₂ y₂ = g y₂ ∧ (∀ y, l₂ y ≤ g y))

theorem tangent_lines_imply_a_range (a : ℝ) :
  has_two_tangent_lines f (g a) → 0 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_imply_a_range_l641_64114


namespace NUMINAMATH_CALUDE_sin_squared_simplification_l641_64103

theorem sin_squared_simplification (x y : ℝ) : 
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_simplification_l641_64103


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l641_64102

/-- The number of ways to distribute students into dormitories -/
def distribute_students (total_students : ℕ) (num_dorms : ℕ) (min_per_dorm : ℕ) (max_per_dorm : ℕ) : ℕ := sorry

/-- The number of ways to distribute students with one student excluded from one dorm -/
def distribute_students_with_exclusion (total_students : ℕ) (num_dorms : ℕ) (min_per_dorm : ℕ) (max_per_dorm : ℕ) : ℕ := sorry

/-- Theorem stating the number of ways to distribute 5 students into 3 dormitories with constraints -/
theorem student_distribution_theorem :
  distribute_students_with_exclusion 5 3 1 2 = 60 := by sorry

end NUMINAMATH_CALUDE_student_distribution_theorem_l641_64102


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l641_64112

/-- Reflects a point over the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- Calculates the area of a triangle given three points -/
def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  let (x1, y1) := a
  let (x2, y2) := b
  let (x3, y3) := c
  0.5 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_of_triangle_ABC' :
  let A : ℝ × ℝ := (3, 4)
  let B' := reflect_y_axis A
  let C' := reflect_y_eq_x B'
  triangle_area A B' C' = 21 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l641_64112


namespace NUMINAMATH_CALUDE_nancy_pearl_beads_difference_l641_64163

/-- Prove that Nancy has 60 more pearl beads than metal beads -/
theorem nancy_pearl_beads_difference (beads_per_bracelet : ℕ) 
  (total_bracelets : ℕ) (nancy_metal_beads : ℕ) (rose_crystal_beads : ℕ) :
  beads_per_bracelet = 8 →
  total_bracelets = 20 →
  nancy_metal_beads = 40 →
  rose_crystal_beads = 20 →
  ∃ (nancy_pearl_beads : ℕ),
    nancy_pearl_beads = beads_per_bracelet * total_bracelets - 
      (nancy_metal_beads + rose_crystal_beads + 2 * rose_crystal_beads) ∧
    nancy_pearl_beads - nancy_metal_beads = 60 :=
by sorry

end NUMINAMATH_CALUDE_nancy_pearl_beads_difference_l641_64163


namespace NUMINAMATH_CALUDE_maximize_expression_l641_64189

def a : Set Int := {-3, -2, -1, 0, 1, 2, 3}

theorem maximize_expression (v : ℝ) :
  ∃ (x y z : Int),
    x ∈ a ∧ y ∈ a ∧ z ∈ a ∧
    (∀ (x' y' z' : Int), x' ∈ a → y' ∈ a → z' ∈ a → v * x' - y' * z' ≤ v * x - y * z) ∧
    v * x - y * z = 15 ∧
    y = -3 ∧ z = 3 :=
sorry

end NUMINAMATH_CALUDE_maximize_expression_l641_64189


namespace NUMINAMATH_CALUDE_elinas_garden_area_l641_64142

/-- The area of Elina's rectangular garden --/
def garden_area (length width : ℝ) : ℝ := length * width

/-- The perimeter of Elina's rectangular garden --/
def garden_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

theorem elinas_garden_area :
  ∃ (length width : ℝ),
    length > 0 ∧
    width > 0 ∧
    length * 30 = 1500 ∧
    garden_perimeter length width * 12 = 1500 ∧
    garden_area length width = 625 := by
  sorry

end NUMINAMATH_CALUDE_elinas_garden_area_l641_64142


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_l641_64143

/-- The angle (in degrees) that has the same terminal side as 1303° -/
def equivalent_angle : ℝ := -137

/-- Theorem stating that the angle with the same terminal side as 1303° is -137° -/
theorem angle_with_same_terminal_side :
  ∃ (k : ℤ), 1303 = 360 * k + equivalent_angle := by
  sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_l641_64143


namespace NUMINAMATH_CALUDE_quarters_to_dollars_l641_64184

/-- The number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- The total number of quarters -/
def total_quarters : ℕ := 8

/-- The dollar amount equivalent to the total number of quarters -/
def dollar_amount : ℚ := total_quarters / quarters_per_dollar

theorem quarters_to_dollars : dollar_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_quarters_to_dollars_l641_64184


namespace NUMINAMATH_CALUDE_angle_bisector_shorter_than_median_l641_64147

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the angle bisector
def angle_bisector (t : Triangle) : ℝ × ℝ := sorry

-- Define the median
def median (t : Triangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem angle_bisector_shorter_than_median (t : Triangle) :
  length t.A t.B ≤ length t.A t.C →
  length t.A (angle_bisector t) ≤ length t.A (median t) ∧
  (length t.A (angle_bisector t) = length t.A (median t) ↔ length t.A t.B = length t.A t.C) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_shorter_than_median_l641_64147


namespace NUMINAMATH_CALUDE_smaller_angle_measure_l641_64125

-- Define a parallelogram
structure Parallelogram where
  -- Smaller angle
  angle1 : ℝ
  -- Larger angle
  angle2 : ℝ
  -- Condition: angle2 exceeds angle1 by 70 degrees
  angle_diff : angle2 = angle1 + 70
  -- Condition: adjacent angles are supplementary
  supplementary : angle1 + angle2 = 180

-- Theorem statement
theorem smaller_angle_measure (p : Parallelogram) : p.angle1 = 55 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_measure_l641_64125


namespace NUMINAMATH_CALUDE_probability_of_four_given_different_numbers_l641_64187

/-- Two fair dice are rolled once each -/
def roll_two_dice : Type := Unit

/-- Event A: The two dice show different numbers -/
def event_A (roll : roll_two_dice) : Prop := sorry

/-- Event B: A 4 is rolled -/
def event_B (roll : roll_two_dice) : Prop := sorry

/-- P(B|A) is the conditional probability of event B given event A -/
def conditional_probability (A B : roll_two_dice → Prop) : ℝ := sorry

theorem probability_of_four_given_different_numbers :
  conditional_probability event_A event_B = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_of_four_given_different_numbers_l641_64187


namespace NUMINAMATH_CALUDE_david_widget_production_l641_64149

theorem david_widget_production (w t : ℕ) (h : w = 3 * t) :
  w * t - (w + 5) * (t - 3) = 4 * t + 15 := by
  sorry

end NUMINAMATH_CALUDE_david_widget_production_l641_64149


namespace NUMINAMATH_CALUDE_rakesh_salary_l641_64118

/-- Rakesh's salary calculation -/
theorem rakesh_salary (salary : ℝ) : 
  (salary * (1 - 0.15) * (1 - 0.30) = 2380) → salary = 4000 := by
  sorry

end NUMINAMATH_CALUDE_rakesh_salary_l641_64118


namespace NUMINAMATH_CALUDE_non_intersecting_lines_parallel_or_skew_l641_64181

/-- A line in 3D space represented by a point and a direction vector. -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Determines if two lines intersect in 3D space. -/
def intersect (l1 l2 : Line3D) : Prop :=
  ∃ t s : ℝ, l1.point + t • l1.direction = l2.point + s • l2.direction

/-- Two lines are parallel if their direction vectors are scalar multiples of each other. -/
def parallel (l1 l2 : Line3D) : Prop :=
  ∃ k : ℝ, l1.direction = k • l2.direction

/-- Two lines are skew if they are neither intersecting nor parallel. -/
def skew (l1 l2 : Line3D) : Prop :=
  ¬(intersect l1 l2) ∧ ¬(parallel l1 l2)

/-- Theorem: If two lines in 3D space do not intersect, then they are either parallel or skew. -/
theorem non_intersecting_lines_parallel_or_skew (l1 l2 : Line3D) :
  ¬(intersect l1 l2) → parallel l1 l2 ∨ skew l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_non_intersecting_lines_parallel_or_skew_l641_64181


namespace NUMINAMATH_CALUDE_square_with_seven_in_tens_place_l641_64121

theorem square_with_seven_in_tens_place (a : ℕ) (b : Fin 10) :
  ∃ k : ℕ, ((10 * a + b) ^ 2) % 100 = 70 + k ∧ k < 10 →
  (b = 4 ∨ b = 6) := by
sorry

end NUMINAMATH_CALUDE_square_with_seven_in_tens_place_l641_64121


namespace NUMINAMATH_CALUDE_inclined_line_properties_l641_64153

/-- A line passing through a point with a given inclination angle -/
structure InclinedLine where
  point : ℝ × ℝ
  angle : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about the equation and triangle area of an inclined line -/
theorem inclined_line_properties (l : InclinedLine) 
  (h1 : l.point = (Real.sqrt 3, -2))
  (h2 : l.angle = π / 3) : 
  ∃ (eq : LineEquation) (area : ℝ),
    eq.a = Real.sqrt 3 ∧ 
    eq.b = -1 ∧ 
    eq.c = -5 ∧
    area = (25 * Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_inclined_line_properties_l641_64153


namespace NUMINAMATH_CALUDE_ST_SQ_ratio_is_930_2197_l641_64131

-- Define the points
variable (P Q R S T : ℝ × ℝ)

-- Define the triangles and their properties
def triangle_PQR_right_at_R : Prop := sorry
def PR_length : ℝ := 5
def RQ_length : ℝ := 12

def triangle_PQS_right_at_P : Prop := sorry
def PS_length : ℝ := 15

-- R and S are on opposite sides of PQ
def R_S_opposite_sides : Prop := sorry

-- Line through S parallel to PR meets RQ extended at T
def S_parallel_PR_meets_RQ_at_T : Prop := sorry

-- Define the ratio ST/SQ
def ST_SQ_ratio : ℝ := sorry

-- Theorem statement
theorem ST_SQ_ratio_is_930_2197 
  (h1 : triangle_PQR_right_at_R)
  (h2 : triangle_PQS_right_at_P)
  (h3 : R_S_opposite_sides)
  (h4 : S_parallel_PR_meets_RQ_at_T) :
  ST_SQ_ratio = 930 / 2197 := by sorry

end NUMINAMATH_CALUDE_ST_SQ_ratio_is_930_2197_l641_64131
