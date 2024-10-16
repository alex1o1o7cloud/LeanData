import Mathlib

namespace NUMINAMATH_CALUDE_total_covered_area_l619_61954

/-- Represents a rectangular strip with length and width -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular strip -/
def Strip.area (s : Strip) : ℝ := s.length * s.width

/-- Calculates the area of overlap between two strips -/
def overlap_area (width : ℝ) (overlap_length : ℝ) : ℝ := width * overlap_length

/-- Theorem: The total area covered by three intersecting strips -/
theorem total_covered_area (s : Strip) (overlap_length : ℝ) : 
  s.length = 12 → s.width = 2 → overlap_length = 2 →
  3 * s.area - 3 * overlap_area s.width overlap_length = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_covered_area_l619_61954


namespace NUMINAMATH_CALUDE_expression_evaluation_l619_61922

/-- Proves that the given expression evaluates to 58.51045 -/
theorem expression_evaluation :
  (3.415 * 2.67) + (8.641 - 1.23) / (0.125 * 4.31) + 5.97^2 = 58.51045 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l619_61922


namespace NUMINAMATH_CALUDE_intersection_length_and_product_l619_61905

noncomputable section

-- Define the line L
def line (α : Real) (t : Real) : Real × Real :=
  (2 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

-- Define the curve C
def curve (θ : Real) : Real × Real :=
  (2 * Real.cos θ, Real.sin θ)

-- Define point P
def P : Real × Real := (2, Real.sqrt 3)

-- Define the origin O
def O : Real × Real := (0, 0)

-- Define the distance between two points
def distance (p1 p2 : Real × Real) : Real :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_length_and_product (α : Real) :
  (α = Real.pi/3 → ∃ A B : Real × Real,
    A ≠ B ∧
    (∃ t : Real, line α t = A) ∧
    (∃ θ : Real, curve θ = A) ∧
    (∃ t : Real, line α t = B) ∧
    (∃ θ : Real, curve θ = B) ∧
    distance A B = 8 * Real.sqrt 10 / 13) ∧
  (Real.tan α = Real.sqrt 5 / 4 →
    ∃ A B : Real × Real,
    A ≠ B ∧
    (∃ t : Real, line α t = A) ∧
    (∃ θ : Real, curve θ = A) ∧
    (∃ t : Real, line α t = B) ∧
    (∃ θ : Real, curve θ = B) ∧
    distance P A * distance P B = distance O P ^ 2) :=
sorry

end

end NUMINAMATH_CALUDE_intersection_length_and_product_l619_61905


namespace NUMINAMATH_CALUDE_base_conversion_sum_l619_61911

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The value of C in base 13 -/
def C : ℕ := 12

/-- The first number in base 9 -/
def num1 : List ℕ := [7, 5, 2]

/-- The second number in base 13 -/
def num2 : List ℕ := [6, C, 3]

theorem base_conversion_sum :
  to_base_10 num1 9 + to_base_10 num2 13 = 1787 := by
  sorry


end NUMINAMATH_CALUDE_base_conversion_sum_l619_61911


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_l619_61941

theorem max_sum_under_constraints (a b : ℝ) :
  4 * a + 3 * b ≤ 10 →
  3 * a + 5 * b ≤ 11 →
  a + b ≤ 156 / 55 := by
sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_l619_61941


namespace NUMINAMATH_CALUDE_same_weaving_rate_first_group_weavers_count_l619_61988

/-- The number of weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of weavers in the second group -/
def second_group_weavers : ℕ := 14

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 49

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 14

/-- The rate of weaving is the same for both groups -/
theorem same_weaving_rate :
  (first_group_mats : ℚ) / (first_group_days * first_group_weavers) =
  (second_group_mats : ℚ) / (second_group_days * second_group_weavers) :=
sorry

/-- The number of weavers in the first group is 4 -/
theorem first_group_weavers_count :
  first_group_weavers = 4 :=
sorry

end NUMINAMATH_CALUDE_same_weaving_rate_first_group_weavers_count_l619_61988


namespace NUMINAMATH_CALUDE_cost_of_item_d_l619_61931

/-- Represents the prices and taxes for items in a shopping scenario -/
structure ShoppingScenario where
  total_spent : ℝ
  total_abc : ℝ
  total_tax : ℝ
  tax_rate_a : ℝ
  tax_rate_b : ℝ
  tax_rate_c : ℝ
  discount_a : ℝ
  discount_b : ℝ

/-- Theorem stating that the cost of item D is 25 given the shopping scenario -/
theorem cost_of_item_d (s : ShoppingScenario)
  (h1 : s.total_spent = 250)
  (h2 : s.total_abc = 225)
  (h3 : s.total_tax = 30)
  (h4 : s.tax_rate_a = 0.05)
  (h5 : s.tax_rate_b = 0.12)
  (h6 : s.tax_rate_c = 0.18)
  (h7 : s.discount_a = 0.1)
  (h8 : s.discount_b = 0.05) :
  s.total_spent - s.total_abc = 25 := by
  sorry

#check cost_of_item_d

end NUMINAMATH_CALUDE_cost_of_item_d_l619_61931


namespace NUMINAMATH_CALUDE_evaluate_y_l619_61989

theorem evaluate_y (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 6*x + 9) - 2 = |x - 2| + |x + 3| - 2 :=
by sorry

end NUMINAMATH_CALUDE_evaluate_y_l619_61989


namespace NUMINAMATH_CALUDE_max_x_value_l619_61919

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 7) 
  (prod_sum_eq : x * y + x * z + y * z = 12) : 
  x ≤ (14 + 2 * Real.sqrt 46) / 6 := by
sorry

end NUMINAMATH_CALUDE_max_x_value_l619_61919


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l619_61973

theorem fraction_product_simplification : 
  (36 : ℚ) / 34 * 26 / 48 * 136 / 78 * 9 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l619_61973


namespace NUMINAMATH_CALUDE_max_gold_coins_l619_61958

theorem max_gold_coins (n : ℕ) (h1 : n < 150) 
  (h2 : ∃ (k : ℕ), n = 13 * k + 3) : n ≤ 143 :=
by
  sorry

end NUMINAMATH_CALUDE_max_gold_coins_l619_61958


namespace NUMINAMATH_CALUDE_min_pressure_cyclic_process_l619_61993

-- Define the constants and variables
variable (V₀ T₀ a b c R : ℝ)
variable (V T P : ℝ → ℝ)

-- Define the cyclic process equation
def cyclic_process (t : ℝ) : Prop :=
  ((V t) / V₀ - a)^2 + ((T t) / T₀ - b)^2 = c^2

-- Define the ideal gas law
def ideal_gas_law (t : ℝ) : Prop :=
  (P t) * (V t) = R * (T t)

-- State the theorem
theorem min_pressure_cyclic_process
  (h1 : ∀ t, cyclic_process V₀ T₀ a b c V T t)
  (h2 : ∀ t, ideal_gas_law R V T P t)
  (h3 : c^2 < a^2 + b^2) :
  ∃ P_min : ℝ, ∀ t, P t ≥ P_min ∧ 
    P_min = (R * T₀ / V₀) * (a * Real.sqrt (a^2 + b^2 - c^2) - b * c) / 
      (b * Real.sqrt (a^2 + b^2 - c^2) + a * c) :=
sorry

end NUMINAMATH_CALUDE_min_pressure_cyclic_process_l619_61993


namespace NUMINAMATH_CALUDE_transform_range_transform_uniform_l619_61961

/-- A uniform random variable in the interval [0,1] -/
def uniform_01 : Type := {x : ℝ // 0 ≤ x ∧ x ≤ 1}

/-- The transformation function -/
def transform (a₁ : uniform_01) : ℝ := a₁.val * 5 - 2

/-- Theorem stating that the transformation maps [0,1] to [-2,3] -/
theorem transform_range :
  ∀ (a₁ : uniform_01), -2 ≤ transform a₁ ∧ transform a₁ ≤ 3 := by
  sorry

/-- Theorem stating that the transformation preserves uniformity -/
theorem transform_uniform :
  uniform_01 → {x : ℝ // -2 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_transform_range_transform_uniform_l619_61961


namespace NUMINAMATH_CALUDE_optimal_price_for_target_profit_l619_61906

-- Define the problem parameters
def cost : ℝ := 30
def initialPrice : ℝ := 40
def initialSales : ℝ := 600
def priceIncreaseSalesDrop : ℝ := 20
def priceDecreaseSalesIncrease : ℝ := 200
def stock : ℝ := 1210
def targetProfit : ℝ := 8400

-- Define the sales function based on price change
def sales (priceChange : ℝ) : ℝ :=
  initialSales + priceDecreaseSalesIncrease * priceChange

-- Define the profit function
def profit (priceChange : ℝ) : ℝ :=
  (initialPrice - priceChange - cost) * (sales priceChange)

-- Theorem statement
theorem optimal_price_for_target_profit :
  ∃ (priceChange : ℝ), profit priceChange = targetProfit ∧ 
  initialPrice - priceChange = 37 ∧
  sales priceChange ≤ stock :=
sorry

end NUMINAMATH_CALUDE_optimal_price_for_target_profit_l619_61906


namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l619_61947

/-- The area of a stripe wrapped around a cylindrical object -/
theorem stripe_area_on_cylinder (diameter : ℝ) (stripe_width : ℝ) (revolutions : ℕ) :
  diameter = 30 →
  stripe_width = 4 →
  revolutions = 3 →
  stripe_width * revolutions * (π * diameter) = 360 * π := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l619_61947


namespace NUMINAMATH_CALUDE_fraction_transformation_l619_61959

theorem fraction_transformation (x : ℚ) : 
  x = 437 → (537 - x) / (463 + x) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l619_61959


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l619_61985

theorem quadratic_equation_properties (m : ℝ) (hm : m ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 - (m^2 + 2)*x + m^2 + 1
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - 2*x₁ - 1 = m^2 - 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l619_61985


namespace NUMINAMATH_CALUDE_mod_equivalence_l619_61944

theorem mod_equivalence (m : ℕ) : 
  176 * 929 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_l619_61944


namespace NUMINAMATH_CALUDE_cyclist_stump_problem_l619_61996

/-- Represents the problem of cyclists on a road with stumps -/
theorem cyclist_stump_problem 
  (road_length : ℝ)
  (speed_1 speed_2 : ℝ)
  (rest_time : ℕ)
  (num_stumps : ℕ) :
  road_length = 37 →
  speed_1 = 15 →
  speed_2 = 20 →
  rest_time > 0 →
  num_stumps > 1 →
  (road_length / speed_1 + num_stumps * rest_time / 60) =
  (road_length / speed_2 + num_stumps * (2 * rest_time) / 60) →
  num_stumps = 37 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_stump_problem_l619_61996


namespace NUMINAMATH_CALUDE_wall_building_theorem_l619_61934

/-- The number of men in the first group that can build a 112-metre wall in 6 days,
    given that 40 men can build a similar wall in 3 days. -/
def number_of_men : ℕ := 80

/-- The length of the wall in metres. -/
def wall_length : ℕ := 112

/-- The number of days it takes the first group to build the wall. -/
def days_first_group : ℕ := 6

/-- The number of men in the second group. -/
def men_second_group : ℕ := 40

/-- The number of days it takes the second group to build the wall. -/
def days_second_group : ℕ := 3

theorem wall_building_theorem :
  number_of_men * days_second_group = men_second_group * days_first_group :=
sorry

end NUMINAMATH_CALUDE_wall_building_theorem_l619_61934


namespace NUMINAMATH_CALUDE_y_equals_152_when_x_is_50_l619_61945

/-- A line passing through three given points -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  x₃ : ℝ
  y₃ : ℝ
  h₁ : x₁ = 0 ∧ y₁ = 2
  h₂ : x₂ = 5 ∧ y₂ = 17
  h₃ : x₃ = 10 ∧ y₃ = 32

/-- The y-coordinate of a point on the line when x = 50 -/
def y_at_50 (l : Line) : ℝ := 152

/-- Theorem stating that for the given line, y = 152 when x = 50 -/
theorem y_equals_152_when_x_is_50 (l : Line) : y_at_50 l = 152 := by
  sorry

end NUMINAMATH_CALUDE_y_equals_152_when_x_is_50_l619_61945


namespace NUMINAMATH_CALUDE_intersection_of_sets_l619_61964

theorem intersection_of_sets : 
  let A : Set ℤ := {1, 2, 3}
  let B : Set ℤ := {-2, 2}
  A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l619_61964


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l619_61990

/-- The circle C is defined by the equation x^2 + y^2 + ax - 2y + b = 0 -/
def circle_equation (x y a b : ℝ) : Prop :=
  x^2 + y^2 + a*x - 2*y + b = 0

/-- The line of symmetry is defined by the equation x + y - 1 = 0 -/
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- Point P has coordinates (2,1) -/
def point_P : ℝ × ℝ := (2, 1)

/-- The symmetric point of P with respect to the line x + y - 1 = 0 -/
def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.2 - 1, P.1 - 1)

theorem circle_center_coordinates (a b : ℝ) :
  (circle_equation 2 1 a b) ∧ 
  (circle_equation (symmetric_point point_P).1 (symmetric_point point_P).2 a b) →
  ∃ (h k : ℝ), h = 0 ∧ k = 1 ∧ 
    ∀ (x y : ℝ), circle_equation x y a b ↔ (x - h)^2 + (y - k)^2 = h^2 + k^2 - b :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l619_61990


namespace NUMINAMATH_CALUDE_solution_set_l619_61936

open Set
open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, 2 * (deriv f x) - f x < 0)
variable (h2 : f (log 2) = 2)

-- Define the set we want to prove is equal to (0, 2)
def S (f : ℝ → ℝ) : Set ℝ := {x | x > 0 ∧ f (log x) - Real.sqrt (2 * x) > 0}

-- State the theorem
theorem solution_set (f : ℝ → ℝ) (h1 : ∀ x, 2 * (deriv f x) - f x < 0) (h2 : f (log 2) = 2) :
  S f = Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l619_61936


namespace NUMINAMATH_CALUDE_range_of_m_l619_61970

theorem range_of_m (x m : ℝ) : 
  (∀ x, (1 < x ∧ x < m - 2) → (1 < x ∧ x < 4)) ∧ 
  (∃ x, (1 < x ∧ x < 4) ∧ ¬(1 < x ∧ x < m - 2)) → 
  m ∈ Set.Ioi 6 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l619_61970


namespace NUMINAMATH_CALUDE_hyperbola_branch_from_condition_l619_61924

/-- The set of points forming one branch of a hyperbola -/
def HyperbolaBranch : Set (ℝ × ℝ) :=
  {P | ∃ (x y : ℝ), P = (x, y) ∧ 
    Real.sqrt ((x + 3)^2 + y^2) - Real.sqrt ((x - 3)^2 + y^2) = 4}

/-- Theorem stating that the given condition forms one branch of a hyperbola -/
theorem hyperbola_branch_from_condition :
  ∃ (F₁ F₂ : ℝ × ℝ), F₁ = (-3, 0) ∧ F₂ = (3, 0) ∧
  HyperbolaBranch = {P | |P.1 - F₁.1| - |P.1 - F₂.1| = 4} :=
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_branch_from_condition_l619_61924


namespace NUMINAMATH_CALUDE_last_number_is_2802_l619_61966

/-- Represents a piece of paper with a given width and height in characters. -/
structure Paper where
  width : Nat
  height : Nat

/-- Represents the space required to write a number, including the following space. -/
def spaceRequired (n : Nat) : Nat :=
  if n < 10 then 2
  else if n < 100 then 3
  else if n < 1000 then 4
  else 5

/-- The last number that can be fully written on the paper. -/
def lastNumberWritten (p : Paper) : Nat :=
  2802

/-- Theorem stating that 2802 is the last number that can be fully written on a 100x100 character paper. -/
theorem last_number_is_2802 (p : Paper) (h1 : p.width = 100) (h2 : p.height = 100) :
  lastNumberWritten p = 2802 := by
  sorry

end NUMINAMATH_CALUDE_last_number_is_2802_l619_61966


namespace NUMINAMATH_CALUDE_closest_to_N_div_M_l619_61956

/-- Mersenne prime M -/
def M : ℕ := 2^127 - 1

/-- Mersenne prime N -/
def N : ℕ := 2^607 - 1

/-- Approximation of log_2 -/
def log2_approx : ℝ := 0.3010

/-- Theorem stating that 10^144 is closest to N/M among given options -/
theorem closest_to_N_div_M :
  let options : List ℝ := [10^140, 10^142, 10^144, 10^146]
  ∀ x ∈ options, |((N : ℝ) / M) - 10^144| ≤ |((N : ℝ) / M) - x| :=
sorry

end NUMINAMATH_CALUDE_closest_to_N_div_M_l619_61956


namespace NUMINAMATH_CALUDE_pizza_distribution_l619_61928

/-- Given 12 coworkers sharing 3 pizzas equally, where each pizza is cut into 8 slices,
    prove that each coworker will receive 2 slices. -/
theorem pizza_distribution (coworkers : ℕ) (pizzas : ℕ) (slices_per_pizza : ℕ) 
    (h1 : coworkers = 12)
    (h2 : pizzas = 3)
    (h3 : slices_per_pizza = 8) :
    (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_distribution_l619_61928


namespace NUMINAMATH_CALUDE_train_length_proof_l619_61971

/-- Given a train that crosses a 500-meter platform in 48 seconds and a signal pole in 18 seconds,
    prove that its length is 300 meters. -/
theorem train_length_proof (L : ℝ) : (L + 500) / 48 = L / 18 ↔ L = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l619_61971


namespace NUMINAMATH_CALUDE_water_remaining_l619_61994

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → 
  used = 11/8 → 
  remaining = initial - used → 
  remaining = 13/8 :=
by
  sorry

#eval (13/8 : ℚ) -- To show that 13/8 is equivalent to 1 5/8

end NUMINAMATH_CALUDE_water_remaining_l619_61994


namespace NUMINAMATH_CALUDE_olympiad_survey_l619_61976

theorem olympiad_survey (P : ℝ) (a b c d : ℝ) 
  (h1 : (a + b + d) / P = 0.9)
  (h2 : (a + c + d) / P = 0.6)
  (h3 : (b + c + d) / P = 0.9)
  (h4 : a + b + c + d = P)
  (h5 : P > 0) :
  d / P = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_olympiad_survey_l619_61976


namespace NUMINAMATH_CALUDE_polynomial_properties_l619_61935

theorem polynomial_properties (p q : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k ↔ Even q ∧ Odd p) ∧ 
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k + 1 ↔ Odd q ∧ Odd p) ∧
  (∀ x : ℤ, ∃ k : ℤ, x^3 + p*x + q = 3*k ↔ q % 3 = 0 ∧ p % 3 = 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_properties_l619_61935


namespace NUMINAMATH_CALUDE_smallest_d_for_10000_l619_61999

theorem smallest_d_for_10000 : 
  ∃ (p q r : Nat), 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    (∀ d : Nat, d > 0 → 
      (∃ (p' q' r' : Nat), 
        Prime p' ∧ Prime q' ∧ Prime r' ∧ 
        p' ≠ q' ∧ p' ≠ r' ∧ q' ≠ r' ∧
        10000 * d = (p' * q' * r')^2) → 
      d ≥ 53361) ∧
    10000 * 53361 = (p * q * r)^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_10000_l619_61999


namespace NUMINAMATH_CALUDE_f_of_3_equals_neg_9_l619_61962

/-- Given a function f(x) = 2x^7 - 3x^3 + 4x - 6 where f(-3) = -3, prove that f(3) = -9 -/
theorem f_of_3_equals_neg_9 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2*x^7 - 3*x^3 + 4*x - 6)
  (h2 : f (-3) = -3) : 
  f 3 = -9 := by
sorry


end NUMINAMATH_CALUDE_f_of_3_equals_neg_9_l619_61962


namespace NUMINAMATH_CALUDE_exists_six_digit_number_without_identical_endings_l619_61903

theorem exists_six_digit_number_without_identical_endings : ∃ A : ℕ, 
  (100000 ≤ A ∧ A < 1000000) ∧ 
  ∀ k : ℕ, k ≤ 500000 → ∀ d : ℕ, d < 10 → 
    (k * A) % 1000000 ≠ d * 111111 := by
  sorry

end NUMINAMATH_CALUDE_exists_six_digit_number_without_identical_endings_l619_61903


namespace NUMINAMATH_CALUDE_sqrt_2_2801_eq_1_51_square_diff_16_2_16_1_square_diff_less_than_3_01_l619_61968

-- Define the square function
def square (x : ℝ) : ℝ := x * x

-- Statement 1: √2.2801 = 1.51
theorem sqrt_2_2801_eq_1_51 : Real.sqrt 2.2801 = 1.51 := by sorry

-- Statement 2: 16.2² - 16.1² = 3.23
theorem square_diff_16_2_16_1 : square 16.2 - square 16.1 = 3.23 := by sorry

-- Statement 3: For any x where 0 < x < 15, (x + 0.1)² - x² < 3.01
theorem square_diff_less_than_3_01 (x : ℝ) (h1 : 0 < x) (h2 : x < 15) :
  square (x + 0.1) - square x < 3.01 := by sorry

end NUMINAMATH_CALUDE_sqrt_2_2801_eq_1_51_square_diff_16_2_16_1_square_diff_less_than_3_01_l619_61968


namespace NUMINAMATH_CALUDE_largest_y_floor_div_l619_61900

theorem largest_y_floor_div : 
  ∀ y : ℝ, (↑(Int.floor y) / y = 8 / 9) → y ≤ 63 / 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_y_floor_div_l619_61900


namespace NUMINAMATH_CALUDE_book_width_average_l619_61915

theorem book_width_average : 
  let book_widths : List ℝ := [3, 3/4, 1.2, 4, 9, 0.5, 8]
  let total_width : ℝ := book_widths.sum
  let num_books : ℕ := book_widths.length
  let average_width : ℝ := total_width / num_books
  ∃ ε > 0, |average_width - 3.8| < ε := by
sorry

end NUMINAMATH_CALUDE_book_width_average_l619_61915


namespace NUMINAMATH_CALUDE_fence_painting_time_l619_61913

theorem fence_painting_time (taimour_time : ℝ) (h1 : taimour_time = 21) :
  let jamshid_time := taimour_time / 2
  let combined_rate := 1 / taimour_time + 1 / jamshid_time
  1 / combined_rate = 7 := by sorry

end NUMINAMATH_CALUDE_fence_painting_time_l619_61913


namespace NUMINAMATH_CALUDE_sum_of_distinct_roots_is_zero_l619_61933

/-- Given real numbers a and b, if x^2 + ax + b and x^2 + bx + a each have two distinct real roots,
    and the product of their roots results in exactly three distinct real roots,
    then the sum of these three distinct roots is 0. -/
theorem sum_of_distinct_roots_is_zero (a b : ℝ) 
    (h1 : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ ∀ x : ℝ, x^2 + a*x + b = 0 ↔ x = r1 ∨ x = r2)
    (h2 : ∃ s1 s2 : ℝ, s1 ≠ s2 ∧ ∀ x : ℝ, x^2 + b*x + a = 0 ↔ x = s1 ∨ x = s2)
    (h3 : ∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ 
          ∀ x : ℝ, (x = t1 ∨ x = t2 ∨ x = t3) ↔ (x^2 + a*x + b = 0 ∨ x^2 + b*x + a = 0)) :
    t1 + t2 + t3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_distinct_roots_is_zero_l619_61933


namespace NUMINAMATH_CALUDE_carrie_tshirt_purchase_l619_61937

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℝ := 9.15

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 22

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := tshirt_cost * num_tshirts

theorem carrie_tshirt_purchase : total_cost = 201.30 := by
  sorry

end NUMINAMATH_CALUDE_carrie_tshirt_purchase_l619_61937


namespace NUMINAMATH_CALUDE_bike_riders_proportion_l619_61929

theorem bike_riders_proportion (total_students bus_riders walkers : ℕ) 
  (h1 : total_students = 92)
  (h2 : bus_riders = 20)
  (h3 : walkers = 27) :
  (total_students - bus_riders - walkers : ℚ) / (total_students - bus_riders : ℚ) = 45 / 72 :=
by sorry

end NUMINAMATH_CALUDE_bike_riders_proportion_l619_61929


namespace NUMINAMATH_CALUDE_g_neg_three_eq_four_l619_61983

/-- The function g is defined as g(x) = x^2 + 2x + 1 for all real x. -/
def g (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- Theorem: The value of g(-3) is equal to 4. -/
theorem g_neg_three_eq_four : g (-3) = 4 := by sorry

end NUMINAMATH_CALUDE_g_neg_three_eq_four_l619_61983


namespace NUMINAMATH_CALUDE_square_root_problem_l619_61921

theorem square_root_problem (a b : ℝ) 
  (h1 : 3^2 = a + 7)
  (h2 : 2^3 = 2*b + 2) :
  ∃ (x : ℝ), x^2 = 3*a + b ∧ (x = 3 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l619_61921


namespace NUMINAMATH_CALUDE_five_b_value_l619_61908

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 0) (h2 : a = b - 3) : 5 * b = 45 / 7 := by
  sorry

end NUMINAMATH_CALUDE_five_b_value_l619_61908


namespace NUMINAMATH_CALUDE_john_total_calories_l619_61939

/-- Calculates the total calories consumed by John given the following conditions:
  * John eats 15 potato chips with a total of 90 calories
  * He eats 10 cheezits, each with 2/5 more calories than a chip
  * He eats 8 pretzels, each with 25% fewer calories than a cheezit
-/
theorem john_total_calories : ℝ := by
  -- Define the number of each item eaten
  let num_chips : ℕ := 15
  let num_cheezits : ℕ := 10
  let num_pretzels : ℕ := 8

  -- Define the total calories from chips
  let total_chip_calories : ℝ := 90

  -- Define the calorie increase ratio for cheezits compared to chips
  let cheezit_increase_ratio : ℝ := 2 / 5

  -- Define the calorie decrease ratio for pretzels compared to cheezits
  let pretzel_decrease_ratio : ℝ := 1 / 4

  -- Calculate the total calories
  have h : ∃ (total_calories : ℝ), total_calories = 224.4 := by sorry

  exact h.choose

end NUMINAMATH_CALUDE_john_total_calories_l619_61939


namespace NUMINAMATH_CALUDE_inequality_solution_set_l619_61981

theorem inequality_solution_set (x : ℝ) : 
  1 / (x + 2) + 8 / (x + 6) ≥ 1 ↔ 
  x ∈ Set.Ici 5 ∪ Set.Iic (-6) ∪ Set.Icc (-2) 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l619_61981


namespace NUMINAMATH_CALUDE_football_victory_points_l619_61991

/-- Represents the points system in a football competition -/
structure FootballPoints where
  victory : ℕ
  draw : ℕ := 1
  defeat : ℕ := 0

/-- Represents the state of a team in the competition -/
structure TeamState where
  totalMatches : ℕ := 20
  playedMatches : ℕ := 5
  currentPoints : ℕ := 8
  targetPoints : ℕ := 40
  minRemainingWins : ℕ := 9

/-- The minimum number of points for a victory that satisfies the given conditions -/
def minVictoryPoints (points : FootballPoints) (state : TeamState) : Prop :=
  points.victory = 3 ∧
  points.victory * state.minRemainingWins + 
    (state.totalMatches - state.playedMatches - state.minRemainingWins) * points.draw ≥ 
    state.targetPoints - state.currentPoints ∧
  ∀ v : ℕ, v < points.victory → 
    v * state.minRemainingWins + 
      (state.totalMatches - state.playedMatches - state.minRemainingWins) * points.draw < 
      state.targetPoints - state.currentPoints

theorem football_victory_points :
  ∃ (points : FootballPoints) (state : TeamState), minVictoryPoints points state := by
  sorry

end NUMINAMATH_CALUDE_football_victory_points_l619_61991


namespace NUMINAMATH_CALUDE_max_sections_five_l619_61910

/-- The maximum number of sections created by n line segments in a rectangle -/
def max_sections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => max_sections m + m + 1

/-- Theorem: The maximum number of sections created by 5 line segments in a rectangle is 16 -/
theorem max_sections_five : max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sections_five_l619_61910


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l619_61963

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x, (a * x^2 + b * x + c ≤ 0) ↔ (x ≤ -2 ∨ x ≥ 3)) →
  (a < 0 ∧
   (∀ x, (a * x + c > 0) ↔ x < 6) ∧
   (∀ x, (c * x^2 + b * x + a < 0) ↔ (-1/2 < x ∧ x < 1/3))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l619_61963


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_over_one_plus_i_l619_61940

theorem imaginary_part_of_z_over_one_plus_i :
  ∀ (z : ℂ), z = 1 - 2 * I →
  (z / (1 + I)).im = -3/2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_over_one_plus_i_l619_61940


namespace NUMINAMATH_CALUDE_sin_400_lt_cos_40_l619_61978

theorem sin_400_lt_cos_40 : 
  Real.sin (400 * Real.pi / 180) < Real.cos (40 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_400_lt_cos_40_l619_61978


namespace NUMINAMATH_CALUDE_max_friendly_groups_19_20_l619_61950

/-- A friendly group in a tournament --/
structure FriendlyGroup (α : Type*) :=
  (a b c : α)
  (a_beats_b : a ≠ b)
  (b_beats_c : b ≠ c)
  (c_beats_a : c ≠ a)

/-- Round-robin tournament results --/
def RoundRobinTournament (α : Type*) := α → α → Prop

/-- Maximum number of friendly groups in a tournament --/
def MaxFriendlyGroups (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    n * (n - 1) * (n + 1) / 24
  else
    n * (n - 2) * (n + 2) / 24

/-- Theorem about maximum friendly groups in tournaments with 19 and 20 teams --/
theorem max_friendly_groups_19_20 :
  (MaxFriendlyGroups 19 = 285) ∧ (MaxFriendlyGroups 20 = 330) :=
by sorry

end NUMINAMATH_CALUDE_max_friendly_groups_19_20_l619_61950


namespace NUMINAMATH_CALUDE_constant_function_derivative_l619_61995

theorem constant_function_derivative (f : ℝ → ℝ) (h : ∀ x, f x = 7) :
  ∀ x, deriv f x = 0 := by sorry

end NUMINAMATH_CALUDE_constant_function_derivative_l619_61995


namespace NUMINAMATH_CALUDE_rectangle_D_max_sum_l619_61918

-- Define the rectangle structure
structure Rectangle where
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the rectangles
def rectangleA : Rectangle := ⟨9, 3, 5, 7⟩
def rectangleB : Rectangle := ⟨8, 2, 4, 6⟩
def rectangleC : Rectangle := ⟨7, 1, 3, 5⟩
def rectangleD : Rectangle := ⟨10, 0, 6, 8⟩
def rectangleE : Rectangle := ⟨6, 4, 2, 0⟩

-- Define the list of all rectangles
def rectangles : List Rectangle := [rectangleA, rectangleB, rectangleC, rectangleD, rectangleE]

-- Function to check if a value is unique in a list
def isUnique (n : ℕ) (l : List ℕ) : Bool :=
  (l.filter (· = n)).length = 1

-- Theorem: Rectangle D has the maximum sum of w + z where z is unique
theorem rectangle_D_max_sum : 
  ∀ r ∈ rectangles, 
    isUnique r.z (rectangles.map Rectangle.z) → 
      r.w + r.z ≤ rectangleD.w + rectangleD.z :=
sorry

end NUMINAMATH_CALUDE_rectangle_D_max_sum_l619_61918


namespace NUMINAMATH_CALUDE_fundraising_shortfall_l619_61957

def goal : ℕ := 10000

def ken_raised : ℕ := 800

theorem fundraising_shortfall (mary_raised scott_raised amy_raised : ℕ) 
  (h1 : mary_raised = 5 * ken_raised)
  (h2 : mary_raised = 3 * scott_raised)
  (h3 : amy_raised = 2 * ken_raised)
  (h4 : amy_raised = scott_raised / 2)
  : ken_raised + mary_raised + scott_raised + amy_raised = goal - 400 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_shortfall_l619_61957


namespace NUMINAMATH_CALUDE_distance_to_optimal_shooting_point_l619_61960

/-- Given a field with width 2b, a goal with width 2a, and a distance c to the sideline,
    prove that the distance x satisfying the conditions is √((b-c)^2 - a^2). -/
theorem distance_to_optimal_shooting_point (b a c x : ℝ) 
  (h1 : b > 0)
  (h2 : a > 0)
  (h3 : c ≥ 0)
  (h4 : c < b)
  (h5 : (b - c)^2 = a^2 + x^2) :
  x = Real.sqrt ((b - c)^2 - a^2) := by
sorry

end NUMINAMATH_CALUDE_distance_to_optimal_shooting_point_l619_61960


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l619_61926

def C : Finset Nat := {34, 35, 37, 41, 43}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ 
    (∀ (m : Nat), m ∈ C → (∃ (p : Nat), Nat.Prime p ∧ p ∣ n) → 
      (∃ (q : Nat), Nat.Prime q ∧ q ∣ m → p ≤ q)) ∧
    n = 34 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l619_61926


namespace NUMINAMATH_CALUDE_five_students_arrangement_l619_61901

def total_arrangements (n : ℕ) : ℕ := n.factorial

def restricted_arrangements (n m : ℕ) : ℕ :=
  m.factorial * (n - m + 1).factorial

theorem five_students_arrangement :
  total_arrangements 5 - restricted_arrangements 5 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_five_students_arrangement_l619_61901


namespace NUMINAMATH_CALUDE_tan_22_5_decomposition_l619_61909

theorem tan_22_5_decomposition :
  ∃ (a b c d : ℕ+), 
    (Real.tan (22.5 * π / 180) = (a : ℝ).sqrt - (b : ℝ).sqrt + (c : ℝ).sqrt - (d : ℝ)) ∧
    a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
    a + b + c + d = 3 := by sorry

end NUMINAMATH_CALUDE_tan_22_5_decomposition_l619_61909


namespace NUMINAMATH_CALUDE_square_area_equals_perimeter_implies_perimeter_16_l619_61992

theorem square_area_equals_perimeter_implies_perimeter_16 :
  ∀ s : ℝ, s > 0 → s^2 = 4*s → 4*s = 16 := by sorry

end NUMINAMATH_CALUDE_square_area_equals_perimeter_implies_perimeter_16_l619_61992


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l619_61974

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l619_61974


namespace NUMINAMATH_CALUDE_hyperbola_t_squared_l619_61904

-- Define a hyperbola
structure Hyperbola where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ
  vertical : Bool

-- Define a function to check if a point is on the hyperbola
def on_hyperbola (h : Hyperbola) (p : ℝ × ℝ) : Prop :=
  if h.vertical then
    (p.2 - h.center.2)^2 / h.b^2 - (p.1 - h.center.1)^2 / h.a^2 = 1
  else
    (p.1 - h.center.1)^2 / h.a^2 - (p.2 - h.center.2)^2 / h.b^2 = 1

-- Theorem statement
theorem hyperbola_t_squared (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_point1 : on_hyperbola h (4, -3))
  (h_point2 : on_hyperbola h (0, -2))
  (h_point3 : ∃ t : ℝ, on_hyperbola h (2, t)) :
  ∃ t : ℝ, on_hyperbola h (2, t) ∧ t^2 = 21/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_t_squared_l619_61904


namespace NUMINAMATH_CALUDE_negation_of_p_l619_61930

theorem negation_of_p : ∀ x : ℝ, -2 < x ∧ x < 2 → |x - 1| + |x + 2| < 6 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l619_61930


namespace NUMINAMATH_CALUDE_trig_fraction_value_l619_61932

theorem trig_fraction_value (θ : Real) (h : Real.tan θ = -2) :
  (7 * Real.sin θ - 3 * Real.cos θ) / (4 * Real.sin θ + 5 * Real.cos θ) = 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_value_l619_61932


namespace NUMINAMATH_CALUDE_sine_function_period_l619_61938

/-- Given a function f(x) = √3 * sin(ωx + φ) where ω > 0, 
    if the distance between adjacent symmetry axes of the graph is 2π, 
    then ω = 1/2 -/
theorem sine_function_period (ω φ : ℝ) (h_ω_pos : ω > 0) :
  (∀ x : ℝ, ∃ k : ℤ, (x + 2 * π) = x + 2 * k * π / ω) →
  ω = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_function_period_l619_61938


namespace NUMINAMATH_CALUDE_breanna_books_count_l619_61943

theorem breanna_books_count (tony_total : ℕ) (dean_total : ℕ) (tony_dean_shared : ℕ) (all_shared : ℕ) (total_different : ℕ) :
  tony_total = 23 →
  dean_total = 12 →
  tony_dean_shared = 3 →
  all_shared = 1 →
  total_different = 47 →
  ∃ breanna_total : ℕ,
    tony_total - tony_dean_shared - all_shared +
    dean_total - tony_dean_shared - all_shared +
    breanna_total - all_shared = total_different ∧
    breanna_total = 20 :=
by sorry

end NUMINAMATH_CALUDE_breanna_books_count_l619_61943


namespace NUMINAMATH_CALUDE_abigail_typing_speed_l619_61965

/-- The number of words Abigail can type in half an hour -/
def words_per_half_hour : ℕ := sorry

/-- The total length of the report in words -/
def total_report_length : ℕ := 1000

/-- The number of words Abigail has already written -/
def words_already_written : ℕ := 200

/-- The number of minutes Abigail needs to finish the report -/
def minutes_to_finish : ℕ := 80

theorem abigail_typing_speed :
  words_per_half_hour = 300 := by sorry

end NUMINAMATH_CALUDE_abigail_typing_speed_l619_61965


namespace NUMINAMATH_CALUDE_base_eight_representation_l619_61925

-- Define the representation function
def represent (base : ℕ) (n : ℕ) : ℕ := 
  3 * base^4 + 0 * base^3 + 4 * base^2 + 0 * base + 7

-- Define the theorem
theorem base_eight_representation : 
  ∃ (base : ℕ), base > 1 ∧ represent base 12551 = 30407 ∧ base = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_representation_l619_61925


namespace NUMINAMATH_CALUDE_unique_solution_condition_l619_61927

/-- The equation has exactly one real solution if and only if a < 7/4 -/
theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, x^3 - a*x^2 - 3*a*x + a^2 - 2 = 0) ↔ a < 7/4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l619_61927


namespace NUMINAMATH_CALUDE_min_value_of_f_l619_61953

def f (x : ℝ) := |3 - x| + |x - 2|

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l619_61953


namespace NUMINAMATH_CALUDE_min_diff_y_x_l619_61920

theorem min_diff_y_x (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  Even x ∧ Odd y ∧ Odd z ∧
  (∀ w, w - x ≥ 9 → z ≤ w) →
  ∃ (d : ℤ), d = y - x ∧ (∀ d' : ℤ, d' = y - x → d ≤ d') ∧ d = 1 := by
sorry

end NUMINAMATH_CALUDE_min_diff_y_x_l619_61920


namespace NUMINAMATH_CALUDE_number_comparison_l619_61916

theorem number_comparison (A B : ℝ) (h : (3/4) * A = (2/3) * B) : A < B := by
  sorry

end NUMINAMATH_CALUDE_number_comparison_l619_61916


namespace NUMINAMATH_CALUDE_sturgeon_books_problem_l619_61914

theorem sturgeon_books_problem (total_volumes : ℕ) (paperback_cost hardcover_cost total_cost : ℕ) 
  (h : total_volumes = 10)
  (hp : paperback_cost = 15)
  (hh : hardcover_cost = 25)
  (ht : total_cost = 220) :
  ∃ (hardcover_count : ℕ), 
    hardcover_count * hardcover_cost + (total_volumes - hardcover_count) * paperback_cost = total_cost ∧
    hardcover_count = 7 := by
  sorry

end NUMINAMATH_CALUDE_sturgeon_books_problem_l619_61914


namespace NUMINAMATH_CALUDE_division_remainder_problem_l619_61979

theorem division_remainder_problem (dividend quotient divisor remainder : ℕ) : 
  dividend = 95 →
  quotient = 6 →
  divisor = 15 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l619_61979


namespace NUMINAMATH_CALUDE_melanie_dimes_and_choiceland_coins_l619_61987

/-- Proves the number of dimes Melanie has and their value in ChoiceLand coins -/
theorem melanie_dimes_and_choiceland_coins 
  (initial_dimes : ℕ) 
  (dad_dimes : ℕ) 
  (mom_dimes : ℕ) 
  (exchange_rate : ℚ) 
  (h1 : initial_dimes = 7)
  (h2 : dad_dimes = 8)
  (h3 : mom_dimes = 4)
  (h4 : exchange_rate = 5/2) : 
  (initial_dimes + dad_dimes + mom_dimes = 19) ∧ 
  ((initial_dimes + dad_dimes + mom_dimes : ℚ) * exchange_rate = 95/2) := by
sorry

end NUMINAMATH_CALUDE_melanie_dimes_and_choiceland_coins_l619_61987


namespace NUMINAMATH_CALUDE_vector_on_line_l619_61917

/-- Given two complex numbers z₁ and z₂, representing points A and B in the complex plane,
    we define z as the vector from A to B, and prove that when z lies on the line y = 1/2 x,
    we can determine the value of parameter a. -/
theorem vector_on_line (a : ℝ) :
  let z₁ : ℂ := 2 * a + 6 * Complex.I
  let z₂ : ℂ := -1 + Complex.I
  let z : ℂ := z₂ - z₁
  z.im = (1/2 : ℝ) * z.re →
  z = -1 - 2 * a - 5 * Complex.I ∧ a = (9/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_l619_61917


namespace NUMINAMATH_CALUDE_light_bulb_packs_theorem_l619_61969

/-- Calculates the number of light bulb packs needed given the number of bulbs required in each room -/
def light_bulb_packs_needed (bedroom bathroom kitchen basement : ℕ) : ℕ :=
  let total_without_garage := bedroom + bathroom + kitchen + basement
  let garage := total_without_garage / 2
  let total := total_without_garage + garage
  (total + 1) / 2

/-- Theorem stating that given the specific number of light bulbs needed in each room,
    the number of packs needed is 6 -/
theorem light_bulb_packs_theorem :
  light_bulb_packs_needed 2 1 1 4 = 6 := by
  sorry

#eval light_bulb_packs_needed 2 1 1 4

end NUMINAMATH_CALUDE_light_bulb_packs_theorem_l619_61969


namespace NUMINAMATH_CALUDE_can_capacity_l619_61948

/-- The capacity of a can given specific milk-water ratios --/
theorem can_capacity (initial_milk : ℝ) (initial_water : ℝ) (added_milk : ℝ) : 
  initial_water = 5 * initial_milk →
  added_milk = 2 →
  (initial_milk + added_milk) / initial_water = 2.00001 / 5.00001 →
  initial_milk + initial_water + added_milk = 14 := by
  sorry

end NUMINAMATH_CALUDE_can_capacity_l619_61948


namespace NUMINAMATH_CALUDE_hezekiahs_age_l619_61975

theorem hezekiahs_age (hezekiah_age : ℕ) (ryanne_age : ℕ) : 
  (ryanne_age = hezekiah_age + 7) → 
  (hezekiah_age + ryanne_age = 15) → 
  (hezekiah_age = 4) := by
sorry

end NUMINAMATH_CALUDE_hezekiahs_age_l619_61975


namespace NUMINAMATH_CALUDE_toy_distribution_ratio_l619_61951

theorem toy_distribution_ratio (total_toys : ℕ) (num_friends : ℕ) 
  (h1 : total_toys = 118) (h2 : num_friends = 4) :
  (total_toys / num_friends) / total_toys = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_toy_distribution_ratio_l619_61951


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l619_61942

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l619_61942


namespace NUMINAMATH_CALUDE_flowchart_transformation_l619_61986

def transform (a b c : ℕ) : ℕ × ℕ × ℕ :=
  (c, a, b)

theorem flowchart_transformation :
  transform 21 32 75 = (75, 21, 32) := by
  sorry

end NUMINAMATH_CALUDE_flowchart_transformation_l619_61986


namespace NUMINAMATH_CALUDE_train_length_l619_61997

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 7 → ∃ length : ℝ, 
  (length ≥ 116.68 ∧ length ≤ 116.70) ∧ 
  length = speed * 1000 / 3600 * time :=
sorry

end NUMINAMATH_CALUDE_train_length_l619_61997


namespace NUMINAMATH_CALUDE_tenth_place_is_unnamed_l619_61980

/-- Represents a racer in the race --/
inductive Racer
| Eda
| Simon
| Jacob
| Naomi
| Cal
| Iris
| Unnamed

/-- Represents the finishing position of a racer --/
def Position := Fin 15

/-- The race results, mapping each racer to their position --/
def RaceResult := Racer → Position

def valid_race_result (result : RaceResult) : Prop :=
  (result Racer.Jacob).val + 4 = (result Racer.Eda).val
  ∧ (result Racer.Naomi).val = (result Racer.Simon).val + 1
  ∧ (result Racer.Jacob).val = (result Racer.Cal).val + 3
  ∧ (result Racer.Simon).val = (result Racer.Iris).val + 2
  ∧ (result Racer.Cal).val + 2 = (result Racer.Iris).val
  ∧ (result Racer.Naomi).val = 7

theorem tenth_place_is_unnamed (result : RaceResult) 
  (h : valid_race_result result) : 
  ∀ r : Racer, r ≠ Racer.Unnamed → (result r).val ≠ 10 := by
  sorry

end NUMINAMATH_CALUDE_tenth_place_is_unnamed_l619_61980


namespace NUMINAMATH_CALUDE_day_284_is_saturday_l619_61967

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the day of the week for a given day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_284_is_saturday (h : dayOfWeek 25 = DayOfWeek.Saturday) :
  dayOfWeek 284 = DayOfWeek.Saturday := by
  sorry

end NUMINAMATH_CALUDE_day_284_is_saturday_l619_61967


namespace NUMINAMATH_CALUDE_midpoint_of_fractions_l619_61972

theorem midpoint_of_fractions : 
  let f1 : ℚ := 3/4
  let f2 : ℚ := 5/6
  let midpoint : ℚ := (f1 + f2) / 2
  midpoint = 19/24 := by sorry

end NUMINAMATH_CALUDE_midpoint_of_fractions_l619_61972


namespace NUMINAMATH_CALUDE_sport_formulation_corn_syrup_amount_l619_61949

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_ratio : DrinkRatio :=
  { flavoring := standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup / 3,
    water := standard_ratio.water * 2 }

/-- Amount of water in the large bottle of sport formulation -/
def water_amount : ℚ := 45

theorem sport_formulation_corn_syrup_amount :
  (water_amount * sport_ratio.corn_syrup) / sport_ratio.water = water_amount :=
sorry

end NUMINAMATH_CALUDE_sport_formulation_corn_syrup_amount_l619_61949


namespace NUMINAMATH_CALUDE_intersection_slope_l619_61902

/-- Given two lines p and q that intersect at (-4, -7), 
    prove that the slope of line q is 2.5 -/
theorem intersection_slope (k : ℝ) : 
  (∀ x y, y = 3 * x + 5 → y = k * x + 3 → x = -4 ∧ y = -7) → 
  k = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l619_61902


namespace NUMINAMATH_CALUDE_remainder_531531_mod_6_l619_61998

theorem remainder_531531_mod_6 : 531531 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_531531_mod_6_l619_61998


namespace NUMINAMATH_CALUDE_problem_proof_l619_61955

def problem (aunt_gift : ℝ) : Prop :=
  let jade_initial : ℝ := 38
  let julia_initial : ℝ := jade_initial / 2
  let jade_final : ℝ := jade_initial + aunt_gift
  let julia_final : ℝ := julia_initial + aunt_gift
  let total : ℝ := jade_final + julia_final
  total = 57 + 2 * aunt_gift

theorem problem_proof (aunt_gift : ℝ) : problem aunt_gift :=
  sorry

end NUMINAMATH_CALUDE_problem_proof_l619_61955


namespace NUMINAMATH_CALUDE_goldfish_cost_graph_l619_61984

/-- Represents the cost function for buying goldfish -/
def cost (n : ℕ) : ℚ :=
  18 * n + 3

/-- Represents the set of points on the graph -/
def graph : Set (ℕ × ℚ) :=
  {p | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 20 ∧ p = (n, cost n)}

theorem goldfish_cost_graph :
  (∃ (S : Set (ℕ × ℚ)), S.Finite ∧ (∀ p ∈ S, ∃ q ∈ S, p ≠ q) ∧ S = graph) :=
sorry

end NUMINAMATH_CALUDE_goldfish_cost_graph_l619_61984


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l619_61912

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_and_line_in_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contained_in n α) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l619_61912


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l619_61982

theorem smallest_number_divisible (n : ℕ) : n = 6297 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 18 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 70 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 100 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 21 * k)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 3) = 18 * k₁ ∧ (n + 3) = 70 * k₂ ∧ (n + 3) = 100 * k₃ ∧ (n + 3) = 21 * k₄) := by
  sorry

#check smallest_number_divisible

end NUMINAMATH_CALUDE_smallest_number_divisible_l619_61982


namespace NUMINAMATH_CALUDE_exists_natural_sqrt_nested_root_l619_61952

theorem exists_natural_sqrt_nested_root : ∃ n : ℕ, n > 1 ∧ ∃ m : ℕ, (n : ℝ)^(7/8) = m := by
  sorry

end NUMINAMATH_CALUDE_exists_natural_sqrt_nested_root_l619_61952


namespace NUMINAMATH_CALUDE_function_proof_l619_61923

theorem function_proof (f : ℤ → ℤ) (h1 : f 0 = 1) (h2 : f 2012 = 2013) :
  ∀ n : ℤ, f n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_proof_l619_61923


namespace NUMINAMATH_CALUDE_expression_simplification_l619_61946

theorem expression_simplification (x y : ℤ) : 
  (x = 1) → (y = -2) → 
  2 * x^2 - (3 * (-5/3 * x^2 + 2/3 * x * y) - (x * y - 3 * x^2)) + 2 * x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l619_61946


namespace NUMINAMATH_CALUDE_square_diff_sum_l619_61907

theorem square_diff_sum : 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_sum_l619_61907


namespace NUMINAMATH_CALUDE_one_root_in_first_quadrant_l619_61977

def complex_equation (z : ℂ) : Prop := z^7 = -1 + Complex.I * Real.sqrt 3

def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem one_root_in_first_quadrant :
  ∃! z, complex_equation z ∧ is_in_first_quadrant z :=
sorry

end NUMINAMATH_CALUDE_one_root_in_first_quadrant_l619_61977
