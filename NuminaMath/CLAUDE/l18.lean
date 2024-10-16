import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l18_1882

theorem equation_solution (x : ℂ) : 
  (x^2 + 4*x + 8) / (x - 3) = 2 ↔ x = -1 + (7*Real.sqrt 2/2)*I ∨ x = -1 - (7*Real.sqrt 2/2)*I :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l18_1882


namespace NUMINAMATH_CALUDE_circle_center_l18_1890

/-- The center of the circle given by the equation x^2 + 10x + y^2 - 14y + 25 = 0 is (-5, 7) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + 10*x + y^2 - 14*y + 25 = 0) → 
  (∃ r : ℝ, (x + 5)^2 + (y - 7)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l18_1890


namespace NUMINAMATH_CALUDE_remainder_difference_l18_1853

theorem remainder_difference (m n : ℕ) (hm : m % 6 = 2) (hn : n % 6 = 3) (h_gt : m > n) :
  (m - n) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_difference_l18_1853


namespace NUMINAMATH_CALUDE_largest_result_is_630_l18_1878

-- Define the set of available digits
def Digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the allowed operations
inductive Operation
| Add : Operation
| Sub : Operation
| Mul : Operation
| Div : Operation

-- Define a sequence of operations
def OperationSequence := List (Operation × Nat)

-- Function to apply a sequence of operations
def applyOperations (seq : OperationSequence) : Nat :=
  sorry

-- Theorem stating that 630 is the largest possible result
theorem largest_result_is_630 :
  ∀ (seq : OperationSequence),
    (∀ n ∈ Digits, (seq.map Prod.snd).count n = 1) →
    applyOperations seq ≤ 630 :=
  sorry

end NUMINAMATH_CALUDE_largest_result_is_630_l18_1878


namespace NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l18_1813

theorem modular_inverse_28_mod_29 : ∃ x : ℕ, x ≤ 28 ∧ (28 * x) % 29 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l18_1813


namespace NUMINAMATH_CALUDE_defective_pens_count_l18_1812

/-- The number of pens in the box -/
def total_pens : ℕ := 16

/-- The probability of selecting two non-defective pens -/
def prob_two_non_defective : ℚ := 65/100

/-- The number of defective pens in the box -/
def defective_pens : ℕ := 3

/-- Theorem stating that given the total number of pens and the probability of
    selecting two non-defective pens, the number of defective pens is 3 -/
theorem defective_pens_count (n : ℕ) (h1 : n = total_pens) 
  (h2 : (n - defective_pens : ℚ) / n * ((n - defective_pens - 1) : ℚ) / (n - 1) = prob_two_non_defective) :
  defective_pens = 3 := by
  sorry

#eval defective_pens

end NUMINAMATH_CALUDE_defective_pens_count_l18_1812


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_when_B_equals_A_l18_1815

-- Define sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 5) < 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x | -2 < x ∧ x ≤ 1 ∨ 3 ≤ x ∧ x < 5} := by sorry

-- Part 2
theorem range_of_a_when_B_equals_A :
  (∀ x, x ∈ B a ↔ x ∈ A) → -1 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_when_B_equals_A_l18_1815


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l18_1835

theorem chocolate_gain_percent (C S : ℝ) (h : 24 * C = 16 * S) : 
  (S - C) / C * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l18_1835


namespace NUMINAMATH_CALUDE_shaded_area_semicircles_l18_1800

/-- The area of shaded region formed by semicircles in a pattern -/
theorem shaded_area_semicircles (diameter : ℝ) (pattern_length : ℝ) : 
  diameter = 3 →
  pattern_length = 12 →
  (pattern_length / diameter) * (π * (diameter / 2)^2 / 2) = 9 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircles_l18_1800


namespace NUMINAMATH_CALUDE_seeds_in_fourth_pot_is_one_l18_1814

/-- Given a total number of seeds, number of pots, and number of seeds per pot for the first three pots,
    calculate the number of seeds that will be planted in the fourth pot. -/
def seeds_in_fourth_pot (total_seeds : ℕ) (num_pots : ℕ) (seeds_per_pot : ℕ) : ℕ :=
  total_seeds - (seeds_per_pot * (num_pots - 1))

/-- Theorem stating that for the given problem, the number of seeds in the fourth pot is 1. -/
theorem seeds_in_fourth_pot_is_one :
  seeds_in_fourth_pot 10 4 3 = 1 := by
  sorry

#eval seeds_in_fourth_pot 10 4 3

end NUMINAMATH_CALUDE_seeds_in_fourth_pot_is_one_l18_1814


namespace NUMINAMATH_CALUDE_fraction_problem_l18_1818

/-- The fraction of p's amount that q and r each have -/
def fraction_of_p (p q r : ℚ) : ℚ :=
  q / p

/-- The problem statement -/
theorem fraction_problem (p q r : ℚ) : 
  p = 56 → 
  p = 2 * (fraction_of_p p q r) * p + 42 → 
  q = r → 
  fraction_of_p p q r = 1/8 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l18_1818


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l18_1862

theorem smallest_undefined_value (y : ℝ) : 
  (∀ z : ℝ, z < y → (z - 3) / (6 * z^2 - 37 * z + 6) ≠ 0) ∧ 
  ((y - 3) / (6 * y^2 - 37 * y + 6) = 0) → 
  y = 1/6 := by sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l18_1862


namespace NUMINAMATH_CALUDE_laptop_price_l18_1867

theorem laptop_price (upfront_percentage : ℚ) (upfront_payment : ℚ) :
  upfront_percentage = 20 / 100 →
  upfront_payment = 240 →
  upfront_percentage * 1200 = upfront_payment :=
by sorry

end NUMINAMATH_CALUDE_laptop_price_l18_1867


namespace NUMINAMATH_CALUDE_sum_of_pairwise_products_of_cubic_roots_l18_1830

theorem sum_of_pairwise_products_of_cubic_roots (p q r : ℂ) : 
  (6 * p^3 - 9 * p^2 + 17 * p - 12 = 0) →
  (6 * q^3 - 9 * q^2 + 17 * q - 12 = 0) →
  (6 * r^3 - 9 * r^2 + 17 * r - 12 = 0) →
  p * q + q * r + r * p = 17 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_pairwise_products_of_cubic_roots_l18_1830


namespace NUMINAMATH_CALUDE_sixteen_right_triangles_l18_1836

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a right-angled triangle
structure RightTriangle where
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  vertex3 : ℝ × ℝ

-- Function to check if two circles do not intersect
def nonIntersecting (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 > (c1.radius + c2.radius)^2

-- Function to check if a line is tangent to a circle
def isTangent (line : ℝ × ℝ → ℝ × ℝ → Prop) (circle : Circle) : Prop :=
  ∃ p : ℝ × ℝ, line p p ∧ 
    let (x, y) := p
    let (cx, cy) := circle.center
    (x - cx)^2 + (y - cy)^2 = circle.radius^2

-- Function to check if a line is a common external tangent
def isCommonExternalTangent (line : ℝ × ℝ → ℝ × ℝ → Prop) (c1 c2 : Circle) : Prop :=
  isTangent line c1 ∧ isTangent line c2

-- Function to check if a line is a common internal tangent
def isCommonInternalTangent (line : ℝ × ℝ → ℝ × ℝ → Prop) (c1 c2 : Circle) : Prop :=
  isTangent line c1 ∧ isTangent line c2

-- Main theorem
theorem sixteen_right_triangles (c1 c2 : Circle) :
  nonIntersecting c1 c2 →
  ∃! (triangles : Finset RightTriangle),
    triangles.card = 16 ∧
    ∀ t ∈ triangles,
      ∃ (hypotenuse leg1 leg2 internalTangent : ℝ × ℝ → ℝ × ℝ → Prop),
        isCommonExternalTangent hypotenuse c1 c2 ∧
        isTangent leg1 c1 ∧
        isTangent leg2 c2 ∧
        isCommonInternalTangent internalTangent c1 c2 ∧
        (∃ p : ℝ × ℝ, internalTangent p p ∧ leg1 p p ∧ leg2 p p) :=
by
  sorry

end NUMINAMATH_CALUDE_sixteen_right_triangles_l18_1836


namespace NUMINAMATH_CALUDE_hyperbola_center_l18_1828

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 81 * x - 16 * y^2 + 64 * y + 144 = 0

-- Define the center of a hyperbola
def is_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), eq x y ↔ eq (x - c.1) (y - c.2)

-- Theorem statement
theorem hyperbola_center :
  is_center (9/2, 2) hyperbola_eq :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l18_1828


namespace NUMINAMATH_CALUDE_faye_coloring_books_l18_1874

/-- Calculates the number of coloring books Faye bought -/
def coloring_books_bought (initial : ℕ) (given_away : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial - given_away)

theorem faye_coloring_books :
  coloring_books_bought 34 3 79 = 48 := by
  sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l18_1874


namespace NUMINAMATH_CALUDE_sock_pair_difference_sock_conditions_l18_1869

/-- Represents the number of socks of a specific color -/
structure SockCount where
  red : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- Represents the sock collection of Joseph -/
def josephSocks : SockCount where
  red := 6
  blue := 12
  white := 8
  black := 2

theorem sock_pair_difference : 
  let blue_pairs := josephSocks.blue / 2
  let black_pairs := josephSocks.black / 2
  blue_pairs - black_pairs = 5 := by
  sorry

theorem sock_conditions : 
  -- Joseph has more pairs of blue socks than black socks
  josephSocks.blue > josephSocks.black ∧
  -- He has one less pair of red socks than white socks
  josephSocks.red / 2 + 1 = josephSocks.white / 2 ∧
  -- He has twice as many blue socks as red socks
  josephSocks.blue = 2 * josephSocks.red ∧
  -- He has 28 socks in total
  josephSocks.red + josephSocks.blue + josephSocks.white + josephSocks.black = 28 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_difference_sock_conditions_l18_1869


namespace NUMINAMATH_CALUDE_kate_keyboard_cost_l18_1817

/-- The amount Kate spent on the keyboard -/
def keyboard_cost (march_savings april_savings may_savings mouse_cost remaining : ℕ) : ℕ :=
  (march_savings + april_savings + may_savings) - (mouse_cost + remaining)

theorem kate_keyboard_cost :
  keyboard_cost 27 13 28 5 14 = 49 := by
  sorry

end NUMINAMATH_CALUDE_kate_keyboard_cost_l18_1817


namespace NUMINAMATH_CALUDE_total_driving_hours_l18_1848

/-- Carl's driving schedule --/
structure DrivingSchedule :=
  (mon : ℕ) (tue : ℕ) (wed : ℕ) (thu : ℕ) (fri : ℕ)

/-- Calculate total hours for a week --/
def weeklyHours (s : DrivingSchedule) : ℕ :=
  s.mon + s.tue + s.wed + s.thu + s.fri

/-- Carl's normal schedule --/
def normalSchedule : DrivingSchedule :=
  ⟨2, 3, 4, 2, 5⟩

/-- Carl's schedule after promotion --/
def promotedSchedule : DrivingSchedule :=
  ⟨3, 5, 7, 6, 5⟩

/-- Carl's schedule for the second week with two days off --/
def secondWeekSchedule : DrivingSchedule :=
  ⟨3, 5, 0, 0, 5⟩

theorem total_driving_hours :
  weeklyHours promotedSchedule + weeklyHours secondWeekSchedule = 39 := by
  sorry

#eval weeklyHours promotedSchedule + weeklyHours secondWeekSchedule

end NUMINAMATH_CALUDE_total_driving_hours_l18_1848


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_exist_quadratic_with_two_roots_l18_1870

/-- A quadratic equation x^2 + bx + c = 0 has two distinct real roots if and only if its discriminant is positive -/
theorem quadratic_two_distinct_roots (b c : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + b*x₁ + c = 0 ∧ x₂^2 + b*x₂ + c = 0 ↔ b^2 - 4*c > 0 := by
  sorry

/-- There exist real values b and c such that the quadratic equation x^2 + bx + c = 0 has two distinct real roots -/
theorem exist_quadratic_with_two_roots : ∃ (b c : ℝ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + b*x₁ + c = 0 ∧ x₂^2 + b*x₂ + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_exist_quadratic_with_two_roots_l18_1870


namespace NUMINAMATH_CALUDE_max_non_managers_l18_1887

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 →
  (managers : ℚ) / non_managers > 5 / 24 →
  non_managers ≤ 38 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l18_1887


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l18_1822

-- Define the hyperbola equation
def hyperbola_equation (m : ℝ) (x y : ℝ) : Prop := x^2 - m*y^2 = 1

-- Define the condition for axis lengths
def axis_length_condition (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a^2 = 1 ∧ b^2 = 1/m ∧ 2*a = 2*(2*b)

-- Theorem statement
theorem hyperbola_m_value (m : ℝ) :
  (∀ x y : ℝ, hyperbola_equation m x y) →
  axis_length_condition m →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l18_1822


namespace NUMINAMATH_CALUDE_equation_solutions_l18_1851

/-- The set of solutions to the equation (x^3 + 3x^2√3 + 9x + 3√3) + (x + √3) = 0 -/
def solution_set : Set ℂ :=
  {z : ℂ | z = -Real.sqrt 3 ∨ z = -Real.sqrt 3 + Complex.I ∨ z = -Real.sqrt 3 - Complex.I}

/-- The equation (x^3 + 3x^2√3 + 9x + 3√3) + (x + √3) = 0 -/
def equation (x : ℂ) : Prop :=
  (x^3 + 3*x^2*Real.sqrt 3 + 9*x + 3*Real.sqrt 3) + (x + Real.sqrt 3) = 0

theorem equation_solutions :
  ∀ x : ℂ, equation x ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l18_1851


namespace NUMINAMATH_CALUDE_tan_2theta_l18_1829

theorem tan_2theta (θ : Real) (h1 : θ ∈ (Set.Ioo 0 Real.pi)) 
  (h2 : Real.sin (Real.pi / 4 - θ) = Real.sqrt 2 / 10) : 
  Real.tan (2 * θ) = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_2theta_l18_1829


namespace NUMINAMATH_CALUDE_hyperbola_focus_on_y_axis_range_l18_1820

/-- Represents the equation (m+1)x^2 + (2-m)y^2 = 1 -/
def hyperbola_equation (m x y : ℝ) : Prop :=
  (m + 1) * x^2 + (2 - m) * y^2 = 1

/-- Condition for the equation to represent a hyperbola with focus on y-axis -/
def is_hyperbola_on_y_axis (m : ℝ) : Prop :=
  m + 1 < 0 ∧ 2 - m > 0

/-- The theorem stating the range of m for which the equation represents
    a hyperbola with focus on the y-axis -/
theorem hyperbola_focus_on_y_axis_range :
  ∀ m : ℝ, is_hyperbola_on_y_axis m ↔ m < -1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_on_y_axis_range_l18_1820


namespace NUMINAMATH_CALUDE_greatest_x_value_l18_1877

theorem greatest_x_value (x : ℤ) (h : 2.134 * (10 : ℝ) ^ (x : ℝ) < 240000) :
  x ≤ 5 ∧ ∃ y : ℤ, y > 5 → 2.134 * (10 : ℝ) ^ (y : ℝ) ≥ 240000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l18_1877


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l18_1899

theorem sufficient_but_not_necessary_condition (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m = n → m^2 = n^2) ∧ ¬(m^2 = n^2 → m = n) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l18_1899


namespace NUMINAMATH_CALUDE_goods_train_length_l18_1845

/-- The length of a goods train given its speed, platform length, and time to cross the platform. -/
theorem goods_train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  speed = 96 →
  platform_length = 360 →
  crossing_time = 32 →
  let speed_mps := speed * (5 / 18)
  let total_distance := speed_mps * crossing_time
  let train_length := total_distance - platform_length
  train_length = 493.44 := by
  sorry

end NUMINAMATH_CALUDE_goods_train_length_l18_1845


namespace NUMINAMATH_CALUDE_vector_representation_l18_1806

def a : Fin 2 → ℝ := ![3, -1]
def e1B : Fin 2 → ℝ := ![-1, 2]
def e2B : Fin 2 → ℝ := ![3, 2]
def e1A : Fin 2 → ℝ := ![0, 0]
def e2A : Fin 2 → ℝ := ![3, 2]
def e1C : Fin 2 → ℝ := ![3, 5]
def e2C : Fin 2 → ℝ := ![6, 10]
def e1D : Fin 2 → ℝ := ![-3, 5]
def e2D : Fin 2 → ℝ := ![3, -5]

theorem vector_representation :
  (∃ α β : ℝ, a = α • e1B + β • e2B) ∧
  (∀ α β : ℝ, a ≠ α • e1A + β • e2A) ∧
  (∀ α β : ℝ, a ≠ α • e1C + β • e2C) ∧
  (∀ α β : ℝ, a ≠ α • e1D + β • e2D) :=
by sorry

end NUMINAMATH_CALUDE_vector_representation_l18_1806


namespace NUMINAMATH_CALUDE_extreme_value_at_one_l18_1823

-- Define the function f(x) = x³ - ax
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem extreme_value_at_one (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 3 := by
  sorry


end NUMINAMATH_CALUDE_extreme_value_at_one_l18_1823


namespace NUMINAMATH_CALUDE_quadratic_inequality_l18_1859

theorem quadratic_inequality (x : ℝ) : x^2 + 3*x - 18 > 0 ↔ x < -6 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l18_1859


namespace NUMINAMATH_CALUDE_florist_chrysanthemums_l18_1808

theorem florist_chrysanthemums (narcissus : ℕ) (bouquets : ℕ) (flowers_per_bouquet : ℕ) 
  (h1 : narcissus = 75)
  (h2 : bouquets = 33)
  (h3 : flowers_per_bouquet = 5)
  (h4 : narcissus + chrysanthemums = bouquets * flowers_per_bouquet) :
  chrysanthemums = 90 :=
by sorry

end NUMINAMATH_CALUDE_florist_chrysanthemums_l18_1808


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l18_1804

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ), x^2 + a*x + b = 0 ↔ x = 2 - 3*I ∨ x = 2 + 3*I :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l18_1804


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l18_1879

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 3) - 4
  f (-3) = -3 := by sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l18_1879


namespace NUMINAMATH_CALUDE_platform_length_l18_1880

/-- Calculates the length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_tree : ℝ)
  (time_platform : ℝ)
  (h1 : train_length = 600)
  (h2 : time_tree = 60)
  (h3 : time_platform = 105) :
  let train_speed := train_length / time_tree
  let platform_length := train_speed * time_platform - train_length
  platform_length = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_l18_1880


namespace NUMINAMATH_CALUDE_committee_probability_l18_1852

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

def probability_at_least_two_of_each : ℚ :=
  1 - (Nat.choose num_girls committee_size +
       num_boys * Nat.choose num_girls (committee_size - 1) +
       Nat.choose num_boys committee_size +
       num_girls * Nat.choose num_boys (committee_size - 1)) /
      Nat.choose total_members committee_size

theorem committee_probability :
  probability_at_least_two_of_each = 457215 / 593775 :=
by sorry

end NUMINAMATH_CALUDE_committee_probability_l18_1852


namespace NUMINAMATH_CALUDE_f_neg_two_eq_twelve_l18_1843

/-- The polynomial function f(x) = x^5 + 4x^4 + x^2 + 20x + 16 -/
def f (x : ℝ) : ℝ := x^5 + 4*x^4 + x^2 + 20*x + 16

/-- Theorem: The value of f(-2) is 12 -/
theorem f_neg_two_eq_twelve : f (-2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_twelve_l18_1843


namespace NUMINAMATH_CALUDE_janet_additional_money_needed_l18_1811

def janet_savings : ℕ := 2225
def monthly_rent : ℕ := 1250
def months_advance : ℕ := 2
def deposit : ℕ := 500
def utility_deposit : ℕ := 300
def moving_costs : ℕ := 150

theorem janet_additional_money_needed :
  janet_savings + (monthly_rent * months_advance + deposit + utility_deposit + moving_costs - janet_savings) = 3450 :=
by sorry

end NUMINAMATH_CALUDE_janet_additional_money_needed_l18_1811


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l18_1885

theorem floor_equation_solutions : 
  (∃ (S : Finset ℤ), S.card = 30 ∧ 
    (∀ x ∈ S, 0 ≤ x ∧ x < 30 ∧ x = ⌊x/2⌋ + ⌊x/3⌋ + ⌊x/5⌋) ∧
    (∀ x : ℤ, 0 ≤ x ∧ x < 30 ∧ x = ⌊x/2⌋ + ⌊x/3⌋ + ⌊x/5⌋ → x ∈ S)) :=
by sorry


end NUMINAMATH_CALUDE_floor_equation_solutions_l18_1885


namespace NUMINAMATH_CALUDE_marble_distribution_l18_1864

theorem marble_distribution (total_marbles : ℕ) (num_friends : ℕ) (marbles_per_friend : ℕ) :
  total_marbles = 30 →
  num_friends = 5 →
  total_marbles = num_friends * marbles_per_friend →
  marbles_per_friend = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l18_1864


namespace NUMINAMATH_CALUDE_ellipse_equation_l18_1876

/-- An ellipse with foci on the x-axis, focal distance 2√6, passing through (√3, √2) -/
structure Ellipse where
  /-- Half the distance between the foci -/
  c : ℝ
  /-- Semi-major axis -/
  a : ℝ
  /-- Semi-minor axis -/
  b : ℝ
  /-- Focal distance is 2√6 -/
  h_focal_distance : c = Real.sqrt 6
  /-- a > b > 0 -/
  h_a_gt_b : a > b ∧ b > 0
  /-- c² = a² - b² -/
  h_c_squared : c^2 = a^2 - b^2
  /-- The ellipse passes through (√3, √2) -/
  h_point : 3 / a^2 + 2 / b^2 = 1

/-- The standard equation of the ellipse is x²/9 + y²/3 = 1 -/
theorem ellipse_equation (e : Ellipse) : e.a^2 = 9 ∧ e.b^2 = 3 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ellipse_equation_l18_1876


namespace NUMINAMATH_CALUDE_f_properties_l18_1855

noncomputable def f (x : ℝ) := Real.exp x - x + (1/2) * x^2

theorem f_properties :
  (∃ (x₀ : ℝ), f x₀ = 1 ∧ ∀ (x : ℝ), f x ≥ f x₀) ∧  -- Minimum value is 1
  (∀ (M : ℝ), ∃ (x : ℝ), f x > M) ∧                -- No maximum value
  (∀ (a b : ℝ), (∀ (x : ℝ), (1/2) * x^2 - f x ≤ a * x + b) →
    (1 - a) * b ≥ -Real.exp 1 / 2) ∧               -- Minimum value of (1-a)b
  (∃ (a b : ℝ), (∀ (x : ℝ), (1/2) * x^2 - f x ≤ a * x + b) ∧
    (1 - a) * b = -Real.exp 1 / 2) :=               -- Minimum is attained
by sorry

end NUMINAMATH_CALUDE_f_properties_l18_1855


namespace NUMINAMATH_CALUDE_average_of_7_12_and_M_l18_1897

theorem average_of_7_12_and_M :
  ∃ M : ℝ, 10 < M ∧ M < 20 ∧ ((7 + 12 + M) / 3 = 11 ∨ (7 + 12 + M) / 3 = 13) := by
  sorry

end NUMINAMATH_CALUDE_average_of_7_12_and_M_l18_1897


namespace NUMINAMATH_CALUDE_hamburger_count_l18_1841

theorem hamburger_count (total_spent single_cost double_cost double_count : ℚ) 
  (h1 : total_spent = 64.5)
  (h2 : single_cost = 1)
  (h3 : double_cost = 1.5)
  (h4 : double_count = 29) :
  ∃ (single_count : ℚ), 
    single_count * single_cost + double_count * double_cost = total_spent ∧ 
    single_count + double_count = 50 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_count_l18_1841


namespace NUMINAMATH_CALUDE_rooster_earnings_l18_1810

/-- Calculates the total earnings from selling roosters -/
def total_earnings (price_per_kg : ℝ) (weight1 : ℝ) (weight2 : ℝ) : ℝ :=
  price_per_kg * (weight1 + weight2)

/-- Theorem: The total earnings from selling two roosters weighing 30 kg and 40 kg at $0.50 per kg is $35 -/
theorem rooster_earnings : total_earnings 0.5 30 40 = 35 := by
  sorry

end NUMINAMATH_CALUDE_rooster_earnings_l18_1810


namespace NUMINAMATH_CALUDE_vacation_tents_l18_1896

/-- Calculates the number of tents needed given the total number of people,
    the number of people the house can accommodate, and the capacity of each tent. -/
def tents_needed (total_people : ℕ) (house_capacity : ℕ) (tent_capacity : ℕ) : ℕ :=
  ((total_people - house_capacity) + tent_capacity - 1) / tent_capacity

theorem vacation_tents :
  tents_needed 14 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vacation_tents_l18_1896


namespace NUMINAMATH_CALUDE_log_8_4096_sum_bounds_l18_1801

theorem log_8_4096_sum_bounds : ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) ≤ Real.log 4096 / Real.log 8 ∧ Real.log 4096 / Real.log 8 < (b : ℝ) ∧ a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_8_4096_sum_bounds_l18_1801


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l18_1883

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume 
  (l : ℝ) (α : ℝ) (h_l : l > 0) (h_α : 0 < α ∧ α < π / 2) :
  let volume := (l^3 * Real.sqrt 3 * Real.sin (2 * α) * Real.cos α) / 8
  ∃ (V : ℝ), V = volume ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l18_1883


namespace NUMINAMATH_CALUDE_inches_to_represent_distance_l18_1860

/-- Represents the scale of a map in miles per inch -/
def map_scale : ℝ := 28

/-- Represents the relationship between inches and miles on the map -/
theorem inches_to_represent_distance (D : ℝ) :
  ∃ I : ℝ, I * map_scale = D ∧ I = D / map_scale :=
sorry

end NUMINAMATH_CALUDE_inches_to_represent_distance_l18_1860


namespace NUMINAMATH_CALUDE_probability_of_matching_pair_l18_1826

def num_blue_socks : ℕ := 12
def num_green_socks : ℕ := 10

def total_socks : ℕ := num_blue_socks + num_green_socks

def ways_to_pick_two (n : ℕ) : ℕ := n * (n - 1) / 2

def matching_pairs : ℕ := ways_to_pick_two num_blue_socks + ways_to_pick_two num_green_socks

def total_ways : ℕ := ways_to_pick_two total_socks

theorem probability_of_matching_pair :
  (matching_pairs : ℚ) / total_ways = 111 / 231 := by sorry

end NUMINAMATH_CALUDE_probability_of_matching_pair_l18_1826


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l18_1803

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l18_1803


namespace NUMINAMATH_CALUDE_certain_number_problem_l18_1819

theorem certain_number_problem : ∃ x : ℝ, (0.45 * x = 0.35 * 40 + 13) ∧ (x = 60) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l18_1819


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l18_1893

theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (d : ℝ) : 
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  d ≠ 0 →  -- non-zero common difference
  ∃ r, r = (a 3) / (a 2) ∧ r = (a 6) / (a 3) →  -- geometric sequence condition
  (a 3) / (a 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l18_1893


namespace NUMINAMATH_CALUDE_expression_simplification_l18_1872

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (1 + 1/x) * (1 - 2/(x+1)) * (1 + 2/(x-1)) = (x + 1) / x :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l18_1872


namespace NUMINAMATH_CALUDE_product_adjacent_faces_is_144_l18_1842

/-- Represents a face of the cube --/
structure Face :=
  (number : Nat)

/-- Represents the cube formed from the numbered net --/
structure Cube :=
  (faces : List Face)
  (adjacent_to_one : List Face)
  (h_adjacent : adjacent_to_one.length = 4)

/-- The product of the numbers on the faces adjacent to face 1 --/
def product_adjacent_faces (c : Cube) : Nat :=
  c.adjacent_to_one.map Face.number |>.foldl (· * ·) 1

/-- Theorem stating that the product of numbers on faces adjacent to face 1 is 144 --/
theorem product_adjacent_faces_is_144 (c : Cube) 
  (h_adjacent_numbers : c.adjacent_to_one.map Face.number = [2, 3, 4, 6]) :
  product_adjacent_faces c = 144 := by
  sorry

end NUMINAMATH_CALUDE_product_adjacent_faces_is_144_l18_1842


namespace NUMINAMATH_CALUDE_cubic_equation_result_l18_1857

theorem cubic_equation_result (a : ℝ) (h : a^3 + 2*a = -2) :
  3*a^6 + 12*a^4 - a^3 + 12*a^2 - 2*a - 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_result_l18_1857


namespace NUMINAMATH_CALUDE_probability_is_three_sixty_fourth_l18_1865

/-- Represents a person with their blocks -/
structure Person :=
  (blocks : Fin 4 → Color)

/-- Represents the possible colors of blocks -/
inductive Color
  | Red
  | Blue
  | Yellow
  | Green
  | White

/-- Represents a placement of blocks in boxes -/
def Placement := Fin 4 → Fin 3 → Color

/-- The set of all possible placements -/
def allPlacements : Set Placement := sorry

/-- Predicate for a placement having at least one box with 3 blocks of the same color -/
def hasThreeSameColor (p : Placement) : Prop := sorry

/-- The probability of a placement having at least one box with 3 blocks of the same color -/
def probability : ℚ := sorry

/-- The main theorem stating the probability -/
theorem probability_is_three_sixty_fourth : probability = 3 / 64 := sorry

end NUMINAMATH_CALUDE_probability_is_three_sixty_fourth_l18_1865


namespace NUMINAMATH_CALUDE_michelle_savings_denomination_l18_1863

/-- Given a total savings amount and a number of bills, calculate the denomination of each bill. -/
def billDenomination (totalSavings : ℕ) (numBills : ℕ) : ℕ :=
  totalSavings / numBills

/-- Theorem: Given Michelle's total savings of $800 and 8 bills, the denomination of each bill is $100. -/
theorem michelle_savings_denomination :
  billDenomination 800 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_michelle_savings_denomination_l18_1863


namespace NUMINAMATH_CALUDE_sequence_transformations_l18_1892

def Sequence (α : Type) := ℕ → α

def is_obtainable (s t : Sequence ℝ) : Prop :=
  ∃ (operations : List (Sequence ℝ → Sequence ℝ)),
    (operations.foldl (λ acc op => op acc) s) = t

theorem sequence_transformations (a b c : Sequence ℝ) :
  (∀ n, a n = n^2) ∧
  (∀ n, b n = n + Real.sqrt 2) ∧
  (∀ n, c n = (n^2000 + 1) / n) →
  (is_obtainable a (λ n => n)) ∧
  (¬ is_obtainable b (λ n => n)) ∧
  (is_obtainable c (λ n => n)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_transformations_l18_1892


namespace NUMINAMATH_CALUDE_quadratic_max_value_l18_1807

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

-- Theorem statement
theorem quadratic_max_value :
  ∃ (max : ℝ), max = -3 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l18_1807


namespace NUMINAMATH_CALUDE_builder_boards_count_l18_1871

/-- The number of boards in each package -/
def boards_per_package : ℕ := 3

/-- The number of packages the builder needs to buy -/
def packages_needed : ℕ := 52

/-- The total number of boards needed -/
def total_boards : ℕ := boards_per_package * packages_needed

theorem builder_boards_count : total_boards = 156 := by
  sorry

end NUMINAMATH_CALUDE_builder_boards_count_l18_1871


namespace NUMINAMATH_CALUDE_absolute_value_plus_power_l18_1847

theorem absolute_value_plus_power : |-5| + 2^0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_power_l18_1847


namespace NUMINAMATH_CALUDE_smallest_append_digits_for_2014_l18_1824

def is_divisible_by_all_less_than_10 (n : ℕ) : Prop :=
  ∀ k : ℕ, k < 10 → k > 0 → n % k = 0

def append_digits (base n digits : ℕ) : ℕ :=
  base * (10 ^ digits) + n

theorem smallest_append_digits_for_2014 :
  (∃ n : ℕ, n < 10000 ∧ is_divisible_by_all_less_than_10 (append_digits 2014 4 n)) ∧
  (∀ d : ℕ, d < 4 → ∀ n : ℕ, n < 10^d → ¬is_divisible_by_all_less_than_10 (append_digits 2014 d n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_append_digits_for_2014_l18_1824


namespace NUMINAMATH_CALUDE_open_box_volume_l18_1875

/-- The volume of an open box formed by cutting squares from corners of a rectangular sheet -/
theorem open_box_volume (sheet_length sheet_width cut_length : ℝ) :
  sheet_length = 48 ∧ 
  sheet_width = 36 ∧ 
  cut_length = 8 →
  let box_length := sheet_length - 2 * cut_length
  let box_width := sheet_width - 2 * cut_length
  let box_height := cut_length
  box_length * box_width * box_height = 5120 := by
sorry

end NUMINAMATH_CALUDE_open_box_volume_l18_1875


namespace NUMINAMATH_CALUDE_percentage_problem_l18_1809

theorem percentage_problem (x : ℝ) (P : ℝ) : 
  x = 680 →
  (P / 100) * x = 0.20 * 1000 - 30 →
  P = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l18_1809


namespace NUMINAMATH_CALUDE_min_value_theorem_l18_1837

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l18_1837


namespace NUMINAMATH_CALUDE_hand_74_falls_off_after_20_minutes_l18_1881

/-- Represents a clock hand with its rotation speed and fall-off time. -/
structure ClockHand where
  speed : ℕ
  fallOffTime : ℚ

/-- Represents a clock with multiple hands. -/
def Clock := List ClockHand

/-- Creates a clock with the specified number of hands. -/
def createClock (n : ℕ) : Clock :=
  List.range n |>.map (fun i => { speed := i + 1, fallOffTime := 0 })

/-- Calculates the fall-off time for a specific hand in the clock. -/
def calculateFallOffTime (clock : Clock) (handSpeed : ℕ) : ℚ :=
  sorry

/-- Theorem: The 74th hand in a 150-hand clock falls off after 20 minutes. -/
theorem hand_74_falls_off_after_20_minutes :
  let clock := createClock 150
  calculateFallOffTime clock 74 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_hand_74_falls_off_after_20_minutes_l18_1881


namespace NUMINAMATH_CALUDE_smallest_possible_value_l18_1827

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 13 * a (n - 1) - 2 * n

/-- The sequence is positive -/
def PositiveSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

theorem smallest_possible_value (a : ℕ → ℝ) 
    (h_recurrence : RecurrenceSequence a) 
    (h_positive : PositiveSequence a) :
    (∀ a₁ : ℝ, a 1 ≥ a₁ → a₁ ≥ 13/36) :=
  sorry

end NUMINAMATH_CALUDE_smallest_possible_value_l18_1827


namespace NUMINAMATH_CALUDE_white_balls_count_l18_1886

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  green = 30 →
  yellow = 10 →
  red = 7 →
  purple = 3 →
  prob_not_red_purple = 9/10 →
  total - (green + yellow + red + purple) = 50 := by
  sorry

#check white_balls_count

end NUMINAMATH_CALUDE_white_balls_count_l18_1886


namespace NUMINAMATH_CALUDE_last_person_coins_l18_1832

/-- Represents the amount of coins each person receives in an arithmetic sequence. -/
structure CoinDistribution where
  a : ℚ
  d : ℚ

/-- Calculates the total number of coins distributed. -/
def totalCoins (dist : CoinDistribution) : ℚ :=
  5 * dist.a

/-- Checks if the sum of the first two equals the sum of the last three. -/
def sumCondition (dist : CoinDistribution) : Prop :=
  (dist.a - 2*dist.d) + (dist.a - dist.d) = dist.a + (dist.a + dist.d) + (dist.a + 2*dist.d)

/-- The main theorem stating the amount the last person receives. -/
theorem last_person_coins (dist : CoinDistribution) 
  (h1 : totalCoins dist = 5)
  (h2 : sumCondition dist) :
  dist.a + 2*dist.d = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_last_person_coins_l18_1832


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l18_1840

theorem cubic_roots_sum (a b c : ℝ) : 
  (3 * a^3 - 9 * a^2 + 54 * a - 12 = 0) →
  (3 * b^3 - 9 * b^2 + 54 * b - 12 = 0) →
  (3 * c^3 - 9 * c^2 + 54 * c - 12 = 0) →
  (a + 2*b - 2)^3 + (b + 2*c - 2)^3 + (c + 2*a - 2)^3 = 162 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l18_1840


namespace NUMINAMATH_CALUDE_yaras_ship_speed_l18_1868

/-- Prove that Yara's ship speed is 30 nautical miles per hour -/
theorem yaras_ship_speed (theons_speed : ℝ) (distance : ℝ) (time_difference : ℝ) :
  theons_speed = 15 →
  distance = 90 →
  time_difference = 3 →
  distance / (distance / theons_speed - time_difference) = 30 :=
by sorry

end NUMINAMATH_CALUDE_yaras_ship_speed_l18_1868


namespace NUMINAMATH_CALUDE_increase_per_page_correct_l18_1816

/-- The increase in drawings per page -/
def increase_per_page : ℕ := 5

/-- The number of drawings on the first page -/
def first_page_drawings : ℕ := 5

/-- The number of pages we're considering -/
def num_pages : ℕ := 5

/-- The total number of drawings on the first five pages -/
def total_drawings : ℕ := 75

/-- Theorem stating that the increase per page is correct -/
theorem increase_per_page_correct : 
  first_page_drawings + 
  (first_page_drawings + increase_per_page) + 
  (first_page_drawings + 2 * increase_per_page) + 
  (first_page_drawings + 3 * increase_per_page) + 
  (first_page_drawings + 4 * increase_per_page) = total_drawings :=
by sorry

end NUMINAMATH_CALUDE_increase_per_page_correct_l18_1816


namespace NUMINAMATH_CALUDE_millet_majority_on_fourth_day_l18_1831

/-- Represents the proportion of millet remaining after birds consume 40% --/
def milletRemainingRatio : ℝ := 0.6

/-- Represents the proportion of millet in the daily seed addition --/
def dailyMilletAddition : ℝ := 0.4

/-- Calculates the total proportion of millet in the feeder after n days --/
def milletProportion (n : ℕ) : ℝ :=
  1 - milletRemainingRatio ^ n

/-- Theorem stating that on the fourth day, the proportion of millet exceeds 50% for the first time --/
theorem millet_majority_on_fourth_day :
  (milletProportion 4 > 1/2) ∧ 
  (∀ k : ℕ, k < 4 → milletProportion k ≤ 1/2) := by
  sorry


end NUMINAMATH_CALUDE_millet_majority_on_fourth_day_l18_1831


namespace NUMINAMATH_CALUDE_sum_of_integers_l18_1873

theorem sum_of_integers (w x y z : ℤ) 
  (eq1 : w - x + y = 7)
  (eq2 : x - y + z = 8)
  (eq3 : y - z + w = 4)
  (eq4 : z - w + x = 3) :
  w + x + y + z = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l18_1873


namespace NUMINAMATH_CALUDE_ratio_change_l18_1854

theorem ratio_change (x y : ℚ) : 
  x / y = 3 / 4 → 
  y = 40 → 
  (x + 10) / (y + 10) = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_change_l18_1854


namespace NUMINAMATH_CALUDE_problem_solution_l18_1856

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 10

theorem problem_solution (m : ℝ) (h_m : m > 1) :
  (∀ x, f m x = x^2 - 2*m*x + 10) →
  (f m m = 1 → ∀ x, f m x = x^2 - 6*x + 10) ∧
  (((∀ x ≤ 2, ∀ y ≤ 2, x < y → f m x > f m y) ∧
    (∀ x ∈ Set.Icc 1 (m + 1), ∀ y ∈ Set.Icc 1 (m + 1), |f m x - f m y| ≤ 9)) →
   m ∈ Set.Icc 2 4) ∧
  ((∃ x ∈ Set.Icc 3 5, f m x = 0) →
   m ∈ Set.Icc (Real.sqrt 10) (7/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l18_1856


namespace NUMINAMATH_CALUDE_expression_evaluation_l18_1805

theorem expression_evaluation (x y : ℤ) (hx : x = -1) (hy : y = 2) :
  x^2 - 2*(3*y^2 - x*y) + (y^2 - 2*x*y) = -19 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l18_1805


namespace NUMINAMATH_CALUDE_bryan_bookshelves_l18_1861

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 38 / 2

/-- The number of books per bookshelf -/
def books_per_shelf : ℕ := 2

/-- The total number of books -/
def total_books : ℕ := 38

theorem bryan_bookshelves : 
  (num_bookshelves * books_per_shelf = total_books) ∧ (num_bookshelves = 19) :=
by sorry

end NUMINAMATH_CALUDE_bryan_bookshelves_l18_1861


namespace NUMINAMATH_CALUDE_age_difference_l18_1898

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 18) : a = c + 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l18_1898


namespace NUMINAMATH_CALUDE_prob_limit_theorem_l18_1866

/-- The probability that every boy chooses a different number than every girl
    when n boys and n girls choose numbers uniformly from {1, 2, 3, 4, 5} -/
def p (n : ℕ) : ℝ := sorry

/-- The limit of the nth root of p_n as n approaches infinity -/
def limit_p : ℝ := sorry

theorem prob_limit_theorem : 
  limit_p = 6 / 25 := by sorry

end NUMINAMATH_CALUDE_prob_limit_theorem_l18_1866


namespace NUMINAMATH_CALUDE_expand_cubic_sum_simplify_complex_fraction_l18_1888

-- Problem 1
theorem expand_cubic_sum (x y : ℝ) : (x + y) * (x^2 - x*y + y^2) = x^3 + y^3 := by
  sorry

-- Problem 2
theorem simplify_complex_fraction (a b c d : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (a^2 * b / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = -a^3 * b^3 / (8 * c * d^6) := by
  sorry

end NUMINAMATH_CALUDE_expand_cubic_sum_simplify_complex_fraction_l18_1888


namespace NUMINAMATH_CALUDE_complex_magnitude_l18_1833

theorem complex_magnitude (z : ℂ) (h : z + Complex.I = z * Complex.I) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l18_1833


namespace NUMINAMATH_CALUDE_point_on_line_l18_1858

theorem point_on_line (k : ℝ) : 
  (1 + 3 * k * (-1/3) = -4 * 4) → k = 17 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l18_1858


namespace NUMINAMATH_CALUDE_average_visitors_is_276_l18_1849

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays := 4
  let totalOtherDays := 26
  let totalVisitors := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  totalVisitors / 30

/-- Theorem stating that the average number of visitors is 276 given the specified conditions -/
theorem average_visitors_is_276 :
  averageVisitors 510 240 = 276 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_is_276_l18_1849


namespace NUMINAMATH_CALUDE_dave_won_fifteen_tickets_l18_1821

/-- Calculates the number of tickets Dave won later at the arcade -/
def tickets_won_later (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Theorem stating that Dave won 15 tickets later -/
theorem dave_won_fifteen_tickets :
  tickets_won_later 25 22 18 = 15 := by
  sorry

end NUMINAMATH_CALUDE_dave_won_fifteen_tickets_l18_1821


namespace NUMINAMATH_CALUDE_speed_increase_reduces_time_l18_1839

/-- Given a 600-mile trip at 50 mph, prove that increasing speed by 25 mph reduces travel time by 4 hours -/
theorem speed_increase_reduces_time : ∀ (distance : ℝ) (initial_speed : ℝ) (speed_increase : ℝ),
  distance = 600 →
  initial_speed = 50 →
  speed_increase = 25 →
  distance / initial_speed - distance / (initial_speed + speed_increase) = 4 :=
by
  sorry

#check speed_increase_reduces_time

end NUMINAMATH_CALUDE_speed_increase_reduces_time_l18_1839


namespace NUMINAMATH_CALUDE_modulus_of_z_l18_1844

theorem modulus_of_z (z : ℂ) (h : z^2 = 48 - 14*I) : Complex.abs z = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l18_1844


namespace NUMINAMATH_CALUDE_range_of_k_l18_1838

theorem range_of_k (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ < 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ < 2*n + 1 ∧
   |x₁ - 2*n| = k * Real.sqrt x₁ ∧
   |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l18_1838


namespace NUMINAMATH_CALUDE_triangle_inequality_ac_not_fourteen_l18_1850

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  c < a + b ∧ b < a + c ∧ a < b + c :=
sorry

theorem ac_not_fourteen (ab bc : ℝ) (hab : ab = 5) (hbc : bc = 8) :
  ¬ (∃ (ac : ℝ), ac = 14 ∧ 
    (ac < ab + bc ∧ bc < ab + ac ∧ ab < bc + ac) ∧
    (0 < ab ∧ 0 < bc ∧ 0 < ac)) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_ac_not_fourteen_l18_1850


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l18_1891

theorem smallest_satisfying_number : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 10 → n % k = k - 1) ∧
  (∀ (m : ℕ), m > 0 ∧ 
    (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 10 → m % k = k - 1) → m ≥ 2519) ∧
  (2519 % 10 = 9 ∧ 
   2519 % 9 = 8 ∧ 
   2519 % 8 = 7 ∧ 
   2519 % 7 = 6 ∧ 
   2519 % 6 = 5 ∧ 
   2519 % 5 = 4 ∧ 
   2519 % 4 = 3 ∧ 
   2519 % 3 = 2 ∧ 
   2519 % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l18_1891


namespace NUMINAMATH_CALUDE_least_multiple_of_first_four_primes_two_ten_divisible_by_first_four_primes_least_multiple_is_two_ten_l18_1802

theorem least_multiple_of_first_four_primes : 
  ∀ n : ℕ, n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n → n ≥ 210 :=
by sorry

theorem two_ten_divisible_by_first_four_primes : 
  2 ∣ 210 ∧ 3 ∣ 210 ∧ 5 ∣ 210 ∧ 7 ∣ 210 :=
by sorry

theorem least_multiple_is_two_ten : 
  ∃! n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → n ≤ m) ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_of_first_four_primes_two_ten_divisible_by_first_four_primes_least_multiple_is_two_ten_l18_1802


namespace NUMINAMATH_CALUDE_infinite_fraction_reciprocal_l18_1834

theorem infinite_fraction_reciprocal (y : ℝ) : 
  y = 1 + (Real.sqrt 3) / (1 + (Real.sqrt 3) / (1 + y)) → 
  1 / ((y + 1) * (y - 2)) = -(Real.sqrt 3) - 2 :=
by sorry

end NUMINAMATH_CALUDE_infinite_fraction_reciprocal_l18_1834


namespace NUMINAMATH_CALUDE_gcd_108_45_is_9_l18_1846

theorem gcd_108_45_is_9 : Nat.gcd 108 45 = 9 := by
  -- Euclidean algorithm
  have h1 : 108 = 2 * 45 + 18 := by sorry
  have h2 : 45 = 2 * 18 + 9 := by sorry
  have h3 : 18 = 2 * 9 := by sorry

  -- Method of successive subtraction
  have s1 : 108 - 45 = 63 := by sorry
  have s2 : 63 - 45 = 18 := by sorry
  have s3 : 45 - 18 = 27 := by sorry
  have s4 : 27 - 18 = 9 := by sorry
  have s5 : 18 - 9 = 9 := by sorry

  sorry -- Proof to be completed

end NUMINAMATH_CALUDE_gcd_108_45_is_9_l18_1846


namespace NUMINAMATH_CALUDE_right_triangle_integer_area_l18_1895

theorem right_triangle_integer_area (a b : ℕ) :
  (∃ A : ℕ, A = a * b / 2) ↔ (Even a ∨ Even b) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_integer_area_l18_1895


namespace NUMINAMATH_CALUDE_a_ln_a_gt_b_ln_b_l18_1884

theorem a_ln_a_gt_b_ln_b (a b : ℝ) (h1 : a > b) (h2 : b > 1) : a * Real.log a > b * Real.log b := by
  sorry

end NUMINAMATH_CALUDE_a_ln_a_gt_b_ln_b_l18_1884


namespace NUMINAMATH_CALUDE_log_problem_l18_1889

theorem log_problem (y : ℝ) : y = (Real.log 4 / Real.log 16) ^ (Real.log 16 / Real.log 4) → Real.log y / Real.log 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l18_1889


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l18_1894

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : ∀ x, quadratic_function a b c (-x + 1) = quadratic_function a b c (x + 1))
  (h2 : quadratic_function a b c 2 = 0)
  (h3 : ∃! x, quadratic_function a b c x = x) :
  a = -1/2 ∧ b = 1 ∧ c = 0 ∧
  ∃ m n : ℝ, m = -4 ∧ n = 0 ∧
    (∀ x, m ≤ x ∧ x ≤ n → 3*m ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 3*n) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l18_1894


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l18_1825

theorem greatest_integer_inequality : ∃ (y : ℤ), (5 : ℚ) / 8 > (y : ℚ) / 17 ∧ 
  ∀ (z : ℤ), (5 : ℚ) / 8 > (z : ℚ) / 17 → z ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l18_1825
