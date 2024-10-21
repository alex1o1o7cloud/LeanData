import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_mixture_l1169_116946

/-- Calculates the alcohol concentration of a mixture given two vessels with different alcohol concentrations and volumes, combined and diluted to a specific volume. -/
theorem alcohol_concentration_mixture 
  (volume1 volume2 final_volume : ℝ) 
  (concentration1 concentration2 : ℝ) 
  (h1 : volume1 > 0)
  (h2 : volume2 > 0)
  (h3 : final_volume > 0)
  (h4 : 0 ≤ concentration1 ∧ concentration1 ≤ 1)
  (h5 : 0 ≤ concentration2 ∧ concentration2 ≤ 1)
  (h6 : volume1 + volume2 ≤ final_volume) :
  let total_alcohol := volume1 * concentration1 + volume2 * concentration2
  let new_concentration := total_alcohol / final_volume
  new_concentration = total_alcohol / final_volume :=
by
  -- Introduce the local definitions
  intro total_alcohol new_concentration
  -- The proof is trivial as the conclusion is just restating the definition
  rfl

#check alcohol_concentration_mixture

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_mixture_l1169_116946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_theorem_l1169_116915

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2 * k * Real.sin (2 * x + Real.pi / 6) + 1

noncomputable def h (x : ℝ) : ℝ := Real.sin (2 * x) - (13 * Real.sqrt 2 / 5) * Real.sin (x + Real.pi / 4) + 369 / 100

def is_perfect_triangle_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y z, x ∈ Set.Icc a b → y ∈ Set.Icc a b → z ∈ Set.Icc a b →
    f x + f y > f z ∧ f y + f z > f x ∧ f z + f x > f y

theorem k_range_theorem (k : ℝ) :
  k > 0 →
  is_perfect_triangle_function (g k) 0 (Real.pi / 2) →
  (∀ x₁ ∈ Set.Icc 0 (Real.pi / 2), ∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), g k x₂ ≥ h x₁) →
  k ∈ Set.Icc (9 / 200) (1 / 4) := by
  sorry

#check k_range_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_theorem_l1169_116915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solutions_l1169_116968

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => (x^3 + 3*x^2*(Real.sqrt 3 : ℂ) + 9*x + 3*(Real.sqrt 3 : ℂ)) + (x + (Real.sqrt 3 : ℂ))
  ∀ x : ℂ, f x = 0 ↔ x = -(Real.sqrt 3 : ℂ) ∨ x = -(Real.sqrt 3 : ℂ) + Complex.I ∨ x = -(Real.sqrt 3 : ℂ) - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solutions_l1169_116968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l1169_116947

-- Definition of "friendly fraction group"
def friendly_fraction_group (f1 f2 : ℝ) : Prop :=
  |f1 - f2| = 2

-- Part 1
theorem part1 (a : ℝ) :
  friendly_fraction_group (3 * a / (a - 1)) ((a + 2) / (a - 1)) ∧
  friendly_fraction_group (a / (2 * a + 1)) ((5 * a + 2) / (2 * a + 1)) :=
sorry

-- Part 2
theorem part2 (a b : ℝ) (h : a * b = 1) :
  friendly_fraction_group (3 * a^2 / (a^2 + b)) ((a - 2 * b^2) / (a + b^2)) :=
sorry

-- Part 3
theorem part3 (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  friendly_fraction_group (3 * a^2 / (a^2 - 4 * b^2)) (a / (a + 2 * b)) →
  (a^2 - 2 * b^2) / (a * b) = -7/2 ∨ (a^2 - 2 * b^2) / (a * b) = -1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l1169_116947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_coordinates_l1169_116964

open Real

-- Define the triangle ABC and points D, E, P
variable (A B C D E P : ℝ × ℝ × ℝ)

-- Define the ratios
noncomputable def bd_dc_ratio : ℝ := 2
noncomputable def dc_bc_ratio : ℝ := 1/3
noncomputable def ae_ac_ratio : ℝ := 3/5

-- State the conditions
axiom D_on_BC_extended : ∃ (t : ℝ), t > 1 ∧ D = B + t • (C - B)
axiom D_ratio : D = (2 • C + B) / 3
axiom E_on_AC : ∃ (s : ℝ), 0 < s ∧ s < 1 ∧ E = A + s • (C - A)
axiom E_ratio : E = (3 • A + 2 • C) / 5
axiom P_intersection : ∃ (t s : ℝ), 
  P = B + t • (E - B) ∧ P = A + s • (D - A)

-- State the theorem
theorem intersection_coordinates :
  ∃ (x y z : ℝ), P = x • A + y • B + z • C ∧ x + y + z = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_coordinates_l1169_116964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1169_116979

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed_kmph : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) :
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_speed_mps * time_to_cross
  let train_length := total_distance - bridge_length
  train_speed_kmph = 36 ∧ time_to_cross = 27.997760179185665 ∧ bridge_length = 170 →
  ∃ ε > 0, |train_length - 109.98| < ε := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1169_116979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_values_when_a_is_2_a_value_when_diff_is_2_l1169_116933

-- Define the set A
def A : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Theorem 1
theorem min_max_values_when_a_is_2 :
  ∀ x ∈ A, (∃ (min_val max_val : ℝ),
    (∀ y ∈ A, f 2 y ≥ min_val) ∧
    (∃ z ∈ A, f 2 z = min_val) ∧
    (∀ y ∈ A, f 2 y ≤ max_val) ∧
    (∃ z ∈ A, f 2 z = max_val) ∧
    min_val = 4 ∧ max_val = 16) :=
by sorry

-- Theorem 2
theorem a_value_when_diff_is_2 :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧
  (∃ (min_val max_val : ℝ),
    (∀ x ∈ A, f a x ≥ min_val) ∧
    (∃ y ∈ A, f a y = min_val) ∧
    (∀ x ∈ A, f a x ≤ max_val) ∧
    (∃ y ∈ A, f a y = max_val) ∧
    max_val - min_val = 2 ∧ a = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_values_when_a_is_2_a_value_when_diff_is_2_l1169_116933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sum_9_l1169_116981

/-- The sum of the first n terms of an arithmetic-geometric sequence -/
noncomputable def S (n : ℕ) (a₁ r : ℝ) : ℝ := a₁ * (1 - r^n) / (1 - r)

/-- An arithmetic-geometric sequence satisfying the given conditions -/
def arithmetic_geometric_sequence (a₁ r : ℝ) : Prop :=
  S 3 a₁ r = 8 ∧ S 6 a₁ r = 10

theorem arithmetic_geometric_sum_9 :
  ∃ a₁ r : ℝ, arithmetic_geometric_sequence a₁ r → S 9 a₁ r = 21/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sum_9_l1169_116981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_magnitude_of_sum_l1169_116983

variable (a b : ℝ × ℝ)

def isUnitVector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vectorAdd (v w : ℝ × ℝ) (m : ℝ) : ℝ × ℝ := (v.1 + m * w.1, v.2 + m * w.2)

noncomputable def vectorMagnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem min_magnitude_of_sum (a b : ℝ × ℝ) 
  (ha : isUnitVector a) (hb : isUnitVector b) (hab : dotProduct a b = 3/5) :
  ∃ (min : ℝ), ∀ (m : ℝ), vectorMagnitude (vectorAdd a b m) ≥ min ∧ min = 4/5 := by
  sorry

#check min_magnitude_of_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_magnitude_of_sum_l1169_116983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1169_116987

/-- The equation of the tangent line to y = x³ - x + 3 at (1, 3) is 2x - y + 1 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => t^3 - t + 3
  let tangent_point := (1 : ℝ)
  let tangent_line := λ t => 2*t - y + 1
  (∀ t, f t = t^3 - t + 3) →
  (f 1 = 3) →
  (tangent_line 1 = 0) →
  (∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ → 
    |(f (1 + h) - f 1) / h - 2| < ε) →
  tangent_line x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1169_116987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_arrangement_probability_l1169_116930

/-- Represents the total number of ways to arrange 6 courses in 6 periods -/
def total_arrangements : ℕ := 720  -- 6!

/-- Represents the number of arrangements where cultural courses are separated by exactly one arts course -/
def case1_arrangements : ℕ := 72  -- 3 * 2 * 1 * 3 * 2 * 1

/-- Represents the number of arrangements where only one pair of cultural courses is separated by one arts course -/
def case2_arrangements : ℕ := 216  -- 3 * 2 * 1 * 3 * 2 * 3

/-- Represents the number of arrangements where all cultural courses are adjacent -/
def case3_arrangements : ℕ := 144  -- 3 * 2 * 1 * 4 * 3 * 2 * 1

/-- The probability that no two adjacent cultural courses are separated by more than one arts course -/
def probability : ℚ := (case1_arrangements + case2_arrangements + case3_arrangements : ℚ) / total_arrangements

theorem course_arrangement_probability :
  probability = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_arrangement_probability_l1169_116930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l1169_116916

noncomputable def slope₁ : ℝ := Real.sqrt 3 / 4

noncomputable def slope₂ : ℝ := -4 / Real.sqrt 3

def x_intersect : ℝ := 2

noncomputable def y_intersect₁ : ℝ := slope₁ * x_intersect

noncomputable def y_intersect₂ : ℝ := slope₂ * x_intersect

noncomputable def perimeter : ℝ := 51 * Real.sqrt 3 / 6

theorem equilateral_triangle_perimeter :
  let side_length := |y_intersect₁ - y_intersect₂|
  (3 * side_length = perimeter) ∧
  (slope₁ * slope₂ = -1) ∧
  (x_intersect > 0) := by
  sorry

#eval x_intersect -- This will work
-- #eval slope₁ -- This won't work due to being noncomputable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l1169_116916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l1169_116904

noncomputable def y (x : ℝ) : ℝ := 
  (Real.sinh x) / (4 * (Real.cosh x)^4) + 
  (3 * Real.sinh x) / (8 * (Real.cosh x)^2) + 
  (3/8) * Real.arctan (Real.sinh x)

theorem y_derivative (x : ℝ) : 
  deriv y x = (1 - 3 * (Real.cosh x)^5) / (4 * (Real.cosh x)^5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l1169_116904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_achieves_target_profit_l1169_116940

/-- Represents the price reduction and profit scenario for a clothing store --/
structure ClothingStore where
  cost_price : ℝ
  original_price : ℝ
  initial_sales : ℝ
  price_reduction : ℝ
  sales_increase_rate : ℝ
  target_profit : ℝ

/-- Calculates the daily profit based on the price reduction --/
def daily_profit (store : ClothingStore) (x : ℝ) : ℝ :=
  (store.original_price - x - store.cost_price) * (store.initial_sales + store.sales_increase_rate * x)

/-- Theorem stating that a price reduction of 20 yuan achieves the target profit --/
theorem price_reduction_achieves_target_profit (store : ClothingStore) 
  (h1 : store.cost_price = 50)
  (h2 : store.original_price = 90)
  (h3 : store.initial_sales = 20)
  (h4 : store.sales_increase_rate = 2)
  (h5 : store.target_profit = 1200) :
  daily_profit store 20 = store.target_profit :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_achieves_target_profit_l1169_116940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_micro_model_length_l1169_116945

/-- The length of a full-size mustang in inches -/
noncomputable def full_size : ℝ := 240

/-- The scale factor for the mid-size model -/
noncomputable def mid_size_scale : ℝ := 1 / 10

/-- The scale factor for the small model relative to the mid-size model -/
noncomputable def small_scale : ℝ := 1 / 4

/-- The scale factor for the mini model relative to the small model -/
noncomputable def mini_scale : ℝ := 1 / 3

/-- The scale factor for the micro model relative to the mini model -/
noncomputable def micro_scale : ℝ := 1 / 5

/-- The length of the micro model in inches -/
noncomputable def micro_length : ℝ := full_size * mid_size_scale * small_scale * mini_scale * micro_scale

/-- Theorem stating that the micro model length is 0.4 inches -/
theorem micro_model_length : micro_length = 0.4 := by
  -- Unfold definitions
  unfold micro_length full_size mid_size_scale small_scale mini_scale micro_scale
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_micro_model_length_l1169_116945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_grass_coverage_l1169_116907

/-- Represents a rectangular park with intersecting paths -/
structure Park where
  total_area : ℝ
  path1_area : ℝ
  path2_area : ℝ

/-- Calculates the percentage of grass coverage in the park -/
noncomputable def grass_coverage_percentage (p : Park) : ℝ :=
  let intersection_area := p.path1_area * p.path2_area / p.total_area
  let total_path_area := p.path1_area + p.path2_area - intersection_area
  let grass_area := p.total_area - total_path_area
  (grass_area / p.total_area) * 100

/-- Theorem stating the grass coverage percentage for the given park -/
theorem park_grass_coverage :
  let p := Park.mk 4000 400 250
  grass_coverage_percentage p = 84.375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_grass_coverage_l1169_116907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1169_116974

/-- Definition of the circle D -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 14*y + 72 = -y^2 - 12*x

/-- The center of the circle -/
def center : ℝ × ℝ := (-6, -7)

/-- The radius of the circle -/
noncomputable def radius : ℝ := Real.sqrt 13

/-- Theorem stating the properties of the circle D -/
theorem circle_properties :
  (∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
  center.1 + center.2 + radius = -13 + Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1169_116974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_99_l1169_116911

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 0, -2; 8, 5, -3; 3, 3, 6]

theorem det_A_eq_99 : Matrix.det A = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_99_l1169_116911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1169_116996

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length : ℝ) (time : ℝ) : ℝ :=
  (train_length + bridge_length) / time

/-- Theorem stating the speed of the train -/
theorem train_speed_calculation :
  let train_length : ℝ := 100
  let bridge_length : ℝ := 160
  let time : ℝ := 25.997920166386688
  let calculated_speed := train_speed train_length bridge_length time
  ∀ ε > 0, |calculated_speed - 10.0003099233| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1169_116996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_function_a_range_l1169_116982

/-- Given that f(x) = log_a[(3-a)x - a] is an increasing function on its domain,
    prove that a is in the interval (1,3). -/
theorem increasing_log_function_a_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = Real.log ((3 - a) * x - a) / Real.log a) 
  (h2 : StrictMono f) : 
  1 < a ∧ a < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_function_a_range_l1169_116982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_derivative_expansion_l1169_116937

def f (x : ℝ) : ℝ := (1 - 2*x)^10

theorem coefficient_x_squared_in_derivative_expansion :
  ∃ (c : ℝ), c = -2880 ∧ 
  ∃ (g : ℝ → ℝ), 
    (∀ x, HasDerivAt f (g x) x) ∧
    (∃ (p : ℕ → ℝ), (∀ x, g x = ∑' i, p i * x^i) ∧ p 2 = c) := by
  sorry

#check coefficient_x_squared_in_derivative_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_derivative_expansion_l1169_116937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_diamond_equation_l1169_116949

-- Define the ⋄ operation
noncomputable def diamond (a b : ℝ) : ℝ := (a^2 - b^2) / (2*b - 2*a)

-- Theorem statement
theorem solve_diamond_equation :
  ∃ x : ℝ, (diamond 3 x = -10) ∧ (x = 17) :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_diamond_equation_l1169_116949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_21_exists_l1169_116926

theorem sum_21_exists (S : Finset ℕ) (h1 : S.card = 11) (h2 : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 20) : 
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = 21 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_21_exists_l1169_116926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l1169_116944

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1 / (Real.exp x)

-- State the theorem
theorem f_increasing_on_positive_reals :
  ∀ x : ℝ, x > 0 → (deriv f) x > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l1169_116944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_1_prop_2_false_prop_3_prop_4_false_correct_propositions_l1169_116942

-- Definition of an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Proposition ①
theorem prop_1 {f : ℝ → ℝ} (h : f (-2) ≠ f 2) : ¬ is_even_function f :=
  sorry

-- Proposition ②
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

def power_function (n : ℕ) : ℝ → ℝ :=
  λ x => x ^ n

theorem prop_2_false : ¬ (∀ n : Fin 2, is_straight_line (power_function n)) :=
  sorry

-- Proposition ③
theorem prop_3 : (∀ a b : ℝ, a ≠ 0 ∧ b ≠ 0 → a * b ≠ 0) ↔ (∀ a b : ℝ, a * b = 0 → a = 0 ∨ b = 0) :=
  sorry

-- Proposition ④
def has_extremum (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x

def cubic_function (a b c d : ℝ) : ℝ → ℝ :=
  λ x => a * x^3 + b * x^2 + c * x + d

theorem prop_4_false : ¬ (∀ a b c d : ℝ, has_extremum (cubic_function a b c d) ↔ b^2 - 3*a*c ≥ 0) :=
  sorry

-- Main theorem
theorem correct_propositions : 
  (∀ f : ℝ → ℝ, f (-2) ≠ f 2 → ¬ is_even_function f) ∧
  ((∀ a b : ℝ, a ≠ 0 ∧ b ≠ 0 → a * b ≠ 0) ↔ (∀ a b : ℝ, a * b = 0 → a = 0 ∨ b = 0)) ∧
  ¬ (∀ n : Fin 2, is_straight_line (power_function n)) ∧
  ¬ (∀ a b c d : ℝ, has_extremum (cubic_function a b c d) ↔ b^2 - 3*a*c ≥ 0) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_1_prop_2_false_prop_3_prop_4_false_correct_propositions_l1169_116942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_sqrt_25_l1169_116994

theorem sqrt_of_sqrt_25 : ∃ (x : ℝ), x^2 = Real.sqrt 25 ∧ (x = Real.sqrt 5 ∨ x = -Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_sqrt_25_l1169_116994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_square_lateral_surface_area_l1169_116972

/-- The lateral surface area of a solid formed by rotating a square with side length 1 around one of its sides -/
noncomputable def lateral_surface_area : ℝ := 2 * Real.pi

/-- Theorem stating that the lateral surface area of the solid is 2π -/
theorem rotate_square_lateral_surface_area : lateral_surface_area = 2 * Real.pi := by
  -- Unfold the definition of lateral_surface_area
  unfold lateral_surface_area
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_square_lateral_surface_area_l1169_116972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_flavors_count_l1169_116906

def num_red_candies : ℕ := 6
def num_green_candies : ℕ := 5

def is_valid_ratio (r g : ℕ) : Bool :=
  r ≤ num_red_candies && g ≤ num_green_candies && (r ≠ 0 || g ≠ 0)

def same_flavor (r1 g1 r2 g2 : ℕ) : Bool :=
  r1 * g2 = r2 * g1

def count_distinct_flavors : ℕ :=
  let all_ratios := List.product (List.range (num_red_candies + 1)) (List.range (num_green_candies + 1))
  let valid_ratios := all_ratios.filter (fun (r, g) => is_valid_ratio r g)
  let distinct_flavors := valid_ratios.foldl
    (fun acc (r, g) =>
      if acc.any (fun (r', g') => same_flavor r g r' g')
      then acc
      else (r, g) :: acc)
    []
  distinct_flavors.length

theorem distinct_flavors_count :
  count_distinct_flavors = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_flavors_count_l1169_116906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equilateral_triangle_l1169_116934

/-- Given non-zero complex numbers z₁, z₂, z₃ satisfying z₁ + z₂ + z₃ = 0 and |z₁| = |z₂| = |z₃|,
    the points Z₁, Z₂, Z₃ corresponding to these complex numbers form an equilateral triangle. -/
theorem complex_equilateral_triangle
  (z₁ z₂ z₃ : ℂ)
  (hz_nonzero : z₁ ≠ 0 ∧ z₂ ≠ 0 ∧ z₃ ≠ 0)
  (hz_sum : z₁ + z₂ + z₃ = 0)
  (hz_mag : Complex.abs z₁ = Complex.abs z₂ ∧ Complex.abs z₂ = Complex.abs z₃) :
  ∃ (r : ℝ), r > 0 ∧ Complex.abs (z₂ - z₁) = r ∧ Complex.abs (z₃ - z₂) = r ∧ Complex.abs (z₁ - z₃) = r :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equilateral_triangle_l1169_116934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l1169_116900

-- Define the vectors a and b
def a (x : ℝ) : Fin 3 → ℝ := ![1, x, -3]
def b (y : ℝ) : Fin 3 → ℝ := ![2, 4, y]

-- Define the parallel condition
def parallel (x y : ℝ) : Prop :=
  ∃ (lambda : ℝ), a x = lambda • (b y)

-- Theorem statement
theorem parallel_vectors_sum (x y : ℝ) (h : parallel x y) : x + y = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l1169_116900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_J_2_12_9_eq_6_l1169_116927

noncomputable def J (a b c : ℝ) : ℝ := a / b + b / c + c / a

theorem J_2_12_9_eq_6 : J 2 12 9 = 6 := by
  -- Unfold the definition of J
  unfold J
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform numerical calculations
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_J_2_12_9_eq_6_l1169_116927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_is_90_degrees_l1169_116929

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  y_axis A.1 ∧ y_axis B.1 ∧ circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ A.2 ≠ B.2

-- Define the central angle
def central_angle (O A B : ℝ × ℝ) (θ : ℝ) : Prop :=
  circle_C O.1 O.2 ∧ 
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = 8 ∧
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = 8 ∧
  θ = Real.arccos ((A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2)) / 8

-- Theorem statement
theorem central_angle_is_90_degrees
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  ∃ O : ℝ × ℝ, central_angle O A B (π / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_is_90_degrees_l1169_116929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_construction_l1169_116986

/-- Given a triangle ABC with side lengths a, b, and c, prove that an inscribed circle
    can be constructed without using angle bisectors. -/
theorem inscribed_circle_construction (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (r : ℝ), r > 0 ∧ ∃ (x y : ℝ), 
    let s := (a + b + c) / 2
    x > 0 ∧ y > 0 ∧
    x + y ≤ a ∧
    y + (s - c) ≤ b ∧
    (s - c) + x ≤ c ∧
    x^2 + y^2 = r^2 ∧
    (a - x)^2 + y^2 = r^2 ∧
    x^2 + (b - y)^2 = r^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_construction_l1169_116986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_radical4_has_sqrt3_factor_l1169_116902

-- Define the radicals
noncomputable def radical1 := Real.sqrt 8
noncomputable def radical2 := Real.sqrt 18
noncomputable def radical3 := Real.sqrt (3/2)
noncomputable def radical4 := Real.sqrt 12

-- Define a predicate to check if a number can be simplified to include √3 as a factor
def hasSqrt3Factor (x : ℝ) : Prop :=
  ∃ (k : ℝ), x = k * Real.sqrt 3

-- State the theorem
theorem only_radical4_has_sqrt3_factor :
  ¬(hasSqrt3Factor radical1) ∧
  ¬(hasSqrt3Factor radical2) ∧
  ¬(hasSqrt3Factor radical3) ∧
  (hasSqrt3Factor radical4) := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma radical4_eq_2_sqrt3 : radical4 = 2 * Real.sqrt 3 := by
  sorry

lemma radical1_eq_2_sqrt2 : radical1 = 2 * Real.sqrt 2 := by
  sorry

lemma radical2_eq_3_sqrt2 : radical2 = 3 * Real.sqrt 2 := by
  sorry

lemma radical3_eq_sqrt6_div_2 : radical3 = (Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_radical4_has_sqrt3_factor_l1169_116902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_street_length_l1169_116991

-- Define the speed in km/h
noncomputable def speed : ℝ := 4.5

-- Define the time taken to cross the street in minutes
noncomputable def time : ℝ := 4

-- Define the conversion factor from km/h to m/min
noncomputable def km_h_to_m_min : ℝ := 1000 / 60

-- Theorem to prove the length of the street
theorem street_length :
  speed * km_h_to_m_min * time = 300 := by
  -- Convert speed from km/h to m/min
  have h1 : speed * km_h_to_m_min = 75 := by
    sorry
  -- Calculate the distance
  have h2 : 75 * time = 300 := by
    sorry
  -- Combine the steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_street_length_l1169_116991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_sort_theorem_l1169_116959

/-- Represents a permutation of integers from 1 to n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The sorting process as described in the problem -/
def sort_process (n : ℕ) (p : Permutation n) : Permutation n := sorry

/-- Predicate to check if a permutation is sorted -/
def is_sorted (n : ℕ) (p : Permutation n) : Prop := 
  ∀ i j : Fin n, i < j → p i < p j

/-- The number of valid permutations for a given n -/
def num_valid_permutations (n : ℕ) : ℕ := sorry

theorem tourist_sort_theorem : 
  num_valid_permutations 2018 = (Nat.factorial 1009) * (Nat.factorial 1010) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_sort_theorem_l1169_116959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l1169_116948

-- Define the new operation *
noncomputable def star (a b : ℝ) : ℝ := a * b + a + b

-- Define the properties of the * operation
axiom star_comm (a b : ℝ) : star a b = star b a
axiom star_zero_right (a : ℝ) : star a 0 = a
axiom star_assoc (a b c : ℝ) : star (star a b) c = c * (a * b) + star a c + star b c - 2 * c

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := star x (x / 2)

-- State the theorem
theorem monotone_decreasing_interval :
  ∀ x y, x < y ∧ y ≤ -3/2 → f y ≤ f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l1169_116948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recipe_scoops_l1169_116955

-- Define the recipe requirements
def total_flour : ℚ := 3 + 3/4
def scoop_capacity : ℚ := 1/3
def flour_consumed : ℚ := 1 + 1/2

-- Define the function to calculate the number of scoops
def scoops_needed (total : ℚ) (capacity : ℚ) (consumed : ℚ) : ℕ :=
  (((total + consumed) / capacity).ceil).toNat

-- Theorem statement
theorem recipe_scoops :
  scoops_needed total_flour scoop_capacity flour_consumed = 16 := by
  -- Unfold the definition of scoops_needed
  unfold scoops_needed
  -- Simplify the arithmetic expressions
  simp [total_flour, scoop_capacity, flour_consumed]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recipe_scoops_l1169_116955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1169_116984

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 / (x + 1)

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 5 }

-- Theorem stating the maximum and minimum values of f on the domain
theorem f_max_min :
  ∃ (max min : ℝ),
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    (∀ x ∈ domain, min ≤ f x) ∧
    (∃ x ∈ domain, f x = min) ∧
    max = 3 ∧
    min = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1169_116984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_functional_equation_l1169_116932

noncomputable def f (x : ℝ) : ℝ := 3^x

theorem exponential_functional_equation (f : ℝ → ℝ) :
  (∀ x y, f (x + y) = f x * f y) →
  (∀ x y, x < y → f x < f y) →
  ∃ c > 1, ∀ x, f x = c^x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_functional_equation_l1169_116932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_polar_form_l1169_116910

/-- The polar form of the parabola y^2 = 5x -/
theorem parabola_polar_form (r φ : ℝ) (h : φ ≠ 0 ∧ φ ≠ π) : 
  (r * Real.sin φ)^2 = 5 * (r * Real.cos φ) ↔ r = (5 * Real.cos φ) / (Real.sin φ)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_polar_form_l1169_116910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_age_at_birth_l1169_116992

def harry_age : ℕ := 50
def father_age_difference : ℕ := 24
def mother_father_age_difference : ℚ := 1 / 25

theorem mother_age_at_birth (harry_age : ℕ) (father_age_difference : ℕ) (mother_father_age_difference : ℚ) :
  harry_age - (harry_age + father_age_difference - (mother_father_age_difference * harry_age).floor) = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_age_at_birth_l1169_116992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_centers_eq_sum_radii_l1169_116993

/-- Two circles are externally tangent if they touch at exactly one point and do not overlap. -/
structure ExternallyTangentCircles (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  center1 : α
  center2 : α
  radius1 : ℝ
  radius2 : ℝ
  radius1_pos : 0 < radius1
  radius2_pos : 0 < radius2
  externally_tangent : ‖center1 - center2‖ = radius1 + radius2

/-- The distance between the centers of two externally tangent circles
    is equal to the sum of their radii. -/
theorem distance_centers_eq_sum_radii
  {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α]
  (circles : ExternallyTangentCircles α) :
  ‖circles.center1 - circles.center2‖ = circles.radius1 + circles.radius2 := by
  exact circles.externally_tangent

#check distance_centers_eq_sum_radii

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_centers_eq_sum_radii_l1169_116993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_15_minus_alpha_cos_alpha_minus_beta_l1169_116971

theorem sin_15_minus_alpha (α : ℝ) :
  (0 < α) ∧ (α < Real.pi / 2) →
  Real.cos (15 * Real.pi / 180 + α) = 15 / 17 →
  Real.sin (15 * Real.pi / 180 - α) = (15 - 8 * Real.sqrt 3) / 34 := by
sorry

theorem cos_alpha_minus_beta (α β : ℝ) :
  (0 < β) ∧ (β < α) ∧ (α < Real.pi / 2) →
  Real.cos α = 1 / 7 →
  Real.cos (α - β) = 13 / 14 →
  β = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_15_minus_alpha_cos_alpha_minus_beta_l1169_116971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_1024_2_l1169_116975

/-- Custom binary operation on natural numbers -/
def custom_op : ℕ → ℕ → ℕ := sorry

/-- First property of the custom operation -/
axiom prop_1 : custom_op 2 2 = 1

/-- Second property of the custom operation -/
axiom prop_2 (n : ℕ) : custom_op (2 * n + 2) 2 = custom_op (2 * n) 2 + 3

/-- Theorem stating the value of 1024※2 -/
theorem custom_op_1024_2 : custom_op 1024 2 = 1534 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_1024_2_l1169_116975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l1169_116905

/-- The distance between two parallel lines in 2D space -/
noncomputable def distance_parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a₁^2 + b₁^2)

/-- Theorem: The distance between the parallel lines x + 2y - 4 = 0 and 2x + 4y + 7 = 0 is 3√5/2 -/
theorem distance_specific_parallel_lines :
  distance_parallel_lines 1 2 (-4) 2 4 7 = (3 * Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l1169_116905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_trajectory_area_l1169_116998

/-- Represents the trajectory of a projectile --/
structure Trajectory where
  u : ℝ  -- initial velocity
  g : ℝ  -- acceleration due to gravity
  φ : ℝ  -- angle of projection

/-- The x-coordinate of the highest point of a trajectory --/
noncomputable def highest_point_x (traj : Trajectory) : ℝ :=
  (traj.u ^ 2 / (2 * traj.g)) * Real.sin (2 * traj.φ)

/-- The y-coordinate of the highest point of a trajectory --/
noncomputable def highest_point_y (traj : Trajectory) : ℝ :=
  (traj.u ^ 2 / (4 * traj.g)) * (1 - Real.cos (2 * traj.φ))

/-- The area of the curve traced by the highest points of trajectories --/
noncomputable def curve_area (u g : ℝ) : ℝ :=
  (Real.pi / 8) * (u ^ 4 / g ^ 2)

theorem projectile_trajectory_area (u g : ℝ) (h_u : u > 0) (h_g : g > 0) :
  ∃ (traj : Trajectory), traj.u = u ∧ traj.g = g ∧
  0 ≤ traj.φ ∧ traj.φ ≤ Real.pi/2 ∧
  (∀ φ, 0 ≤ φ ∧ φ ≤ Real.pi/2 →
    ∃ (x y : ℝ),
      x = highest_point_x { u := u, g := g, φ := φ } ∧
      y = highest_point_y { u := u, g := g, φ := φ } ∧
      (x ^ 2 / (u ^ 2 / (2 * g)) ^ 2) + ((y - u ^ 2 / (4 * g)) ^ 2 / (u ^ 2 / (4 * g)) ^ 2) = 1) →
  curve_area u g = (Real.pi / 8) * (u ^ 4 / g ^ 2) := by
  sorry

#check projectile_trajectory_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_trajectory_area_l1169_116998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l1169_116952

/-- Given two fixed points F₁ and F₂ in a metric space, if the distance between them is 6
    and a point M satisfies that the sum of its distances to F₁ and F₂ is also 6,
    then M lies on the line segment connecting F₁ and F₂. -/
theorem point_on_line_segment 
  {X : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X] (F₁ F₂ M : X) :
  dist F₁ F₂ = 6 →
  dist M F₁ + dist M F₂ = 6 →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • F₁ + t • F₂ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l1169_116952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_triangle_abc_l1169_116967

theorem cosine_triangle_abc (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  (b^2 + c^2 - a^2) / (2*b*c) = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_triangle_abc_l1169_116967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_variance_l1169_116965

noncomputable def data : List ℝ := [1, 3, -1, 2, 1]

noncomputable def average (l : List ℝ) : ℝ := (l.sum) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let avg := average l
  (l.map (fun x => (x - avg)^2)).sum / l.length

theorem data_variance :
  average data = 1 → variance data = 9/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_variance_l1169_116965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_5_in_sum_l1169_116985

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def highestPowerOf5 (n : ℕ) : ℕ :=
  let rec count_powers (m : ℕ) (acc : ℕ) : ℕ :=
    if m = 0 then acc
    else if m % 5 = 0 then count_powers (m / 5) (acc + 1)
    else acc
  (List.range n).foldl (λ acc x => acc + count_powers (x + 1) 0) 0

theorem highest_power_of_5_in_sum :
  highestPowerOf5 (factorial 120 + factorial 121 + factorial 122) = 28 := by
  sorry

#eval highestPowerOf5 (factorial 120 + factorial 121 + factorial 122)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_5_in_sum_l1169_116985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_lines_with_30_degree_angle_l1169_116941

-- Define a line in a 2D plane
def Line : Type := ℝ → ℝ → Prop

-- Define a point in a 2D plane
def Point : Type := ℝ × ℝ

-- Define the concept of a point being outside a line
def outside (P : Point) (a : Line) : Prop := sorry

-- Define the angle between two lines
noncomputable def angle_between (l1 l2 : Line) : ℝ := sorry

-- Main theorem
theorem infinite_lines_with_30_degree_angle 
  (P : Point) (a : Line) (h : outside P a) : 
  ∃ (S : Set Line), (∀ l ∈ S, angle_between l a = 30) ∧ Infinite S :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_lines_with_30_degree_angle_l1169_116941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l1169_116935

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from an imaginary axis endpoint to an asymptote -/
noncomputable def distance_to_asymptote (h : Hyperbola) : ℝ := h.a * h.b / Real.sqrt (h.a^2 + h.b^2)

theorem hyperbola_eccentricity_is_two (h : Hyperbola) 
  (h_distance : distance_to_asymptote h = h.b / 2) : 
  eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l1169_116935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_equals_one_f_max_at_zero_f_zero_equals_one_l1169_116943

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 - x - Real.exp (-x)

-- Define the piecewise function f_M
noncomputable def f_M (M : ℝ) (x : ℝ) : ℝ :=
  if f x ≤ M then f x else M

-- State the theorem
theorem min_M_equals_one :
  ∃ (M : ℝ), M > 0 ∧ (∀ (x : ℝ), f_M M x = f x) ∧
  (∀ (M' : ℝ), (∀ (x : ℝ), f_M M' x = f x) → M' ≥ M) ∧
  M = 1 := by
  sorry

-- Prove that the maximum of f occurs at x = 0
theorem f_max_at_zero :
  ∀ (x : ℝ), f 0 ≥ f x := by
  sorry

-- Prove that f(0) = 1
theorem f_zero_equals_one :
  f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_equals_one_f_max_at_zero_f_zero_equals_one_l1169_116943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_and_function_l1169_116978

noncomputable def angle_theta (θ : Real) : Prop :=
  ∃ (x y : Real), x = -12/13 ∧ y = 5/13 ∧ x^2 + y^2 = 1 ∧ 
  (Real.pi/2 < θ ∧ θ < Real.pi) ∧ 
  (Real.cos θ = x ∧ Real.sin θ = y)

noncomputable def f (θ : Real) : Real :=
  (Real.cos (3*Real.pi/2 + θ) + Real.cos (Real.pi - θ) * Real.tan (3*Real.pi + θ)) / 
  (Real.sin (3*Real.pi/2 - θ) * Real.sin (-θ))

theorem trigonometric_values_and_function (θ : Real) 
  (h : angle_theta θ) : 
  Real.sin θ = 5/13 ∧ 
  Real.cos θ = -12/13 ∧ 
  Real.tan θ = -5/12 ∧ 
  f θ = 5/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_and_function_l1169_116978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_accumulate_steps_necessary_not_sufficient_l1169_116917

-- Define the propositions
variable (accumulate_small_steps : Prop)
variable (reach_thousand_miles : Prop)

-- Define the given condition
axiom given_condition : ¬accumulate_small_steps → ¬reach_thousand_miles

-- Define what it means to be a necessary condition
def is_necessary (p q : Prop) : Prop := q → p

-- Define what it means to be a sufficient condition
def is_sufficient (p q : Prop) : Prop := p → q

-- Theorem to prove
theorem accumulate_steps_necessary_not_sufficient :
  (is_necessary accumulate_small_steps reach_thousand_miles) ∧
  ¬(is_sufficient accumulate_small_steps reach_thousand_miles) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_accumulate_steps_necessary_not_sufficient_l1169_116917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_5_12_between_consecutive_integers_l1169_116997

theorem log_5_12_between_consecutive_integers (m n : ℤ) : 
  (m + 1 = n) →  -- m and n are consecutive integers
  (m : ℝ) < Real.log 12 / Real.log 5 →  -- log_5(12) is greater than m
  Real.log 12 / Real.log 5 < (n : ℝ) →  -- log_5(12) is less than n
  m + n = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_5_12_between_consecutive_integers_l1169_116997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_theorem_l1169_116977

theorem log_sum_theorem (m a b : ℝ) (h1 : m^2 = a) (h2 : m^3 = b) (h3 : m > 0) (h4 : m ≠ 1) :
  2 * (Real.log a / Real.log m) + (Real.log b / Real.log m) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_theorem_l1169_116977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l1169_116923

-- Define the line l
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, 1 + t * Real.sin α)

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 = 4*y

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_angle (α : ℝ) :
  (0 ≤ α) → (α < Real.pi) →
  (∃ t1 t2 : ℝ, t1 ≠ t2 ∧
    curve_C (line_l α t1).1 (line_l α t1).2 ∧
    curve_C (line_l α t2).1 (line_l α t2).2 ∧
    distance (line_l α t1) (line_l α t2) = 8) →
  α = Real.pi/4 ∨ α = 3*Real.pi/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l1169_116923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1169_116957

-- Define the universal set U and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {4, 7, 8}

-- State the theorem
theorem set_operations :
  (U \ A = {1, 2, 6, 7, 8}) ∧
  (U \ B = {1, 2, 3, 5, 6}) ∧
  ((U \ A) ∩ (U \ B) = {1, 2, 6}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1169_116957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangle_in_circle_l1169_116925

/-- The area of the largest rectangle that can be inscribed in a circle. -/
noncomputable def area_largest_rectangle_in_circle (r : ℝ) : ℝ :=
  2 * r^2

theorem max_rectangle_in_circle (r : ℝ) (h : r = 5) : 
  ∃ (a : ℝ), a = 50 ∧ a = area_largest_rectangle_in_circle r :=
by
  -- We'll use the existence of a real number satisfying our conditions
  use 50
  constructor
  · -- First part: a = 50
    rfl
  · -- Second part: 50 = area_largest_rectangle_in_circle r
    rw [area_largest_rectangle_in_circle]
    rw [h]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangle_in_circle_l1169_116925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1169_116990

theorem solve_exponential_equation (x : ℝ) : (3 : ℝ)^(x - 2) = (9 : ℝ)^3 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1169_116990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_logs_l1169_116913

theorem max_sum_of_logs (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (Real.log x + Real.log y) ≤ 2 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_logs_l1169_116913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_300_l1169_116995

/-- Represents a rectangle with given length and perimeter -/
structure Rectangle where
  length : ℝ
  perimeter : ℝ

/-- Calculates the width of a rectangle given its length and perimeter -/
noncomputable def Rectangle.width (r : Rectangle) : ℝ :=
  (r.perimeter - 2 * r.length) / 2

/-- Calculates the area of a rectangle -/
noncomputable def Rectangle.area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Theorem: A rectangle with perimeter 70 and length 15 has an area of 300 -/
theorem rectangle_area_300 :
  ∀ (r : Rectangle), r.perimeter = 70 ∧ r.length = 15 → r.area = 300 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_300_l1169_116995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_work_rate_proof_l1169_116922

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem q_work_rate_proof 
  (p q r : ℝ)
  (h1 : work_rate p = work_rate q + work_rate r)
  (h2 : work_rate p + work_rate q = work_rate 10)
  (h3 : work_rate r = work_rate 30) :
  work_rate q = work_rate 30 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_work_rate_proof_l1169_116922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l1169_116951

/-- Given a line l and a curve C in the plane, prove properties about their equations and the minimum distance between them. -/
theorem line_and_curve_properties :
  -- Define the line l parametrically
  let l : ℝ → ℝ × ℝ := λ t ↦ (2*t + 1, 2*t)
  -- Define the curve C in polar coordinates
  let C : ℝ × ℝ → Prop := λ p ↦ let (ρ, θ) := p; ρ^2 - 4*ρ*(Real.sin θ) + 3 = 0
  -- State the properties to be proved
  ∃ (rect_eq_l : ℝ × ℝ → Prop)
    (gen_eq_C : ℝ × ℝ → Prop)
    (min_dist : ℝ),
    -- 1. The rectangular equation of line l
    (∀ x y, rect_eq_l (x, y) ↔ x - y - 1 = 0) ∧
    -- 2. The general equation of curve C
    (∀ x y, gen_eq_C (x, y) ↔ x^2 + (y - 2)^2 = 1) ∧
    -- 3. The minimum distance between a point on l and a point on C
    min_dist = Real.sqrt 14 / 2 ∧
    (∀ t, ∃ (p : ℝ × ℝ),
      gen_eq_C p ∧ 
      (Real.sqrt ((l t).1 - p.1)^2 + ((l t).2 - p.2)^2 ≥ min_dist)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l1169_116951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_two_squares_min_area_ten_squares_l1169_116970

/-- The sum of areas of squares formed from pieces of a wire -/
noncomputable def sumOfSquareAreas (pieces : List ℝ) : ℝ :=
  (pieces.map (fun x => (x / 4)^2)).sum

/-- Theorem: The sum of areas of two squares formed from a 10-meter wire
    is minimized when the wire is cut into two equal pieces of 5 meters each -/
theorem min_area_two_squares :
  let pieces := [5, 5]
  ∀ x : ℝ, 0 < x → x < 10 →
    sumOfSquareAreas [x, 10 - x] ≥ sumOfSquareAreas pieces := by
  sorry

/-- Theorem: The sum of areas of ten squares formed from a 10-meter wire
    is minimized when the wire is cut into ten equal pieces of 1 meter each -/
theorem min_area_ten_squares :
  let pieces := List.replicate 10 1
  ∀ ls : List ℝ, ls.length = 10 → ls.sum = 10 →
    sumOfSquareAreas ls ≥ sumOfSquareAreas pieces := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_two_squares_min_area_ten_squares_l1169_116970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unbounded_l1169_116999

/-- 
f(n) denotes the number of ways a positive integer n can be expressed as a sum of numbers of the form 2^a * 5^b,
where a and b are non-negative integers, the order of summands does not matter, and none of the summands is a divisor of another.
-/
def f (n : ℕ+) : ℕ := sorry

/-- f(n) is unbounded -/
theorem f_unbounded : ∀ K : ℕ, ∃ n : ℕ+, f n > K := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unbounded_l1169_116999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l1169_116914

theorem sin_cos_difference (a b : Real) (ha : a = 7 * Real.pi / 180) (hb : b = 37 * Real.pi / 180) :
  Real.sin a * Real.cos b - Real.sin (Real.pi / 2 - a) * Real.sin b = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l1169_116914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_white_area_ratio_l1169_116912

/-- The ratio of black to white areas in concentric circles -/
theorem black_white_area_ratio : 
  let r₁ : ℝ := 2
  let r₂ : ℝ := 4
  let r₃ : ℝ := 6
  let r₄ : ℝ := 8
  let black_area := π * r₁^2 + π * (r₃^2 - r₂^2)
  let white_area := π * (r₂^2 - r₁^2) + π * (r₄^2 - r₃^2)
  black_area / white_area = 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_white_area_ratio_l1169_116912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l1169_116989

theorem divisibility_in_subset (n : ℕ+) (S : Finset ℕ) : 
  S ⊆ Finset.range (2 * n) → S.card = n + 1 →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l1169_116989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l1169_116976

/-- The distance from the focus of the parabola y² = 16x to the asymptote y = x of a hyperbola with eccentricity √2 is 2√2 -/
theorem distance_focus_to_asymptote :
  let parabola : ℝ → ℝ → Prop := λ x y ↦ y^2 = 16 * x
  let focus : ℝ × ℝ := (4, 0)
  let asymptote : ℝ → ℝ := λ x ↦ x
  let eccentricity : ℝ := Real.sqrt 2
  let distance (p : ℝ × ℝ) (f : ℝ → ℝ) : ℝ :=
    abs (f p.1 - p.2) / Real.sqrt (1 + (deriv f p.1)^2)
  distance focus asymptote = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l1169_116976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_l1169_116936

/-- A quadratic function -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_root (a b c : ℝ) :
  quadratic_function a b c (-3) = 16 →
  quadratic_function a b c 0 = -5 →
  quadratic_function a b c 3 = -8 →
  quadratic_function a b c (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_l1169_116936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_four_345_l1169_116954

/-- Represents a number in base 4 as a list of digits (least significant first) -/
def BaseFour := List Nat

/-- Converts a natural number to its base 4 representation -/
def toBaseFour (n : Nat) : BaseFour := sorry

/-- Counts the number of odd digits in a base 4 number -/
def countOddDigits (n : BaseFour) : Nat := sorry

/-- Sums the odd digits in a base 4 number -/
def sumOddDigits (n : BaseFour) : Nat := sorry

theorem base_four_345 :
  let base4 : BaseFour := toBaseFour 345
  base4 = [1, 2, 1, 5] ∧
  countOddDigits base4 = 3 ∧
  sumOddDigits base4 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_four_345_l1169_116954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_twelve_l1169_116980

def next_term (n : ℕ) : ℕ :=
  if n < 20 then n * 3
  else if n % 2 = 0 ∧ n ≥ 20 ∧ n ≤ 60 then n + 10
  else if n % 2 = 0 ∧ n > 60 then n / 5
  else if n % 2 = 1 ∧ n > 20 then n - 7
  else n

def sequence_term (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => next_term (sequence_term start n)

theorem fiftieth_term_is_twelve : sequence_term 120 49 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_twelve_l1169_116980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1169_116969

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line
def my_line (k x y : ℝ) : Prop := k*x - y - k + 1 = 0

-- Define the arc ratio condition
def arc_ratio (k : ℝ) : Prop := ∃ (x₁ y₁ x₂ y₂ : ℝ),
  my_circle x₁ y₁ ∧ my_circle x₂ y₂ ∧
  my_line k x₁ y₁ ∧ my_line k x₂ y₂ ∧
  (∃ (l₁ l₂ : ℝ), l₁ / l₂ = 3 ∧ l₁ + l₂ = 2*Real.pi*2)

theorem circle_line_intersection (k : ℝ) :
  arc_ratio k → k = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1169_116969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_ahead_distance_l1169_116931

/-- Calculates the distance a jogger is ahead of a train given their speeds and the time it takes for the train to pass the jogger. -/
noncomputable def distance_ahead (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) : ℝ :=
  (train_speed - jogger_speed) * passing_time / 3.6 - train_length

/-- Theorem stating that under the given conditions, the jogger is 260 meters ahead of the train. -/
theorem jogger_ahead_distance :
  let jogger_speed : ℝ := 9
  let train_speed : ℝ := 45
  let train_length : ℝ := 120
  let passing_time : ℝ := 38
  distance_ahead jogger_speed train_speed train_length passing_time = 260 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_ahead_distance_l1169_116931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equation_solution_l1169_116988

noncomputable def h (x : ℝ) : ℝ := ((2 * x + 6) / 5) ^ (1/4)

theorem h_equation_solution :
  ∃ x : ℝ, h (3 * x) = 3 * h x ∧ x = -40/13 := by
  use (-40/13)
  constructor
  · -- Proof that h(3x) = 3h(x) when x = -40/13
    sorry
  · -- Proof that x = -40/13
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equation_solution_l1169_116988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_C_is_simplest_l1169_116958

noncomputable def fraction_A (a : ℝ) := 2 / (2 * a - 4)
noncomputable def fraction_B (a : ℝ) := a^2 / (a^2 - 2*a)
noncomputable def fraction_C (a : ℝ) := (a + 1) / (a^2 + 1)
noncomputable def fraction_D (a : ℝ) := (a - 1) / (a^2 - 1)

def is_simplest_form (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, ∀ n d : ℤ, n ≠ 0 ∧ d ≠ 0 → (n : ℝ) / (d : ℝ) ≠ f a

theorem fraction_C_is_simplest :
  is_simplest_form fraction_C :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_C_is_simplest_l1169_116958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1169_116973

/-- The number of intersection points between y = |3x + 4| and y = 5 - |2x - 1| -/
theorem intersection_points_count :
  ∃! p : ℝ × ℝ, 
    (fun x y ↦ y = |3 * x + 4|) p.1 p.2 ∧ 
    (fun x y ↦ y = 5 - |2 * x - 1|) p.1 p.2 ∧
    p = (0, 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1169_116973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_l1169_116908

/-- Given a point P(-4, 3) on the terminal side of angle a, prove that cos a = -4/5 -/
theorem cosine_of_angle (a : ℝ) : 
  let P : ℝ × ℝ := (-4, 3)
  (Real.cos a = -4/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_l1169_116908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l1169_116939

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7*θ) = -4732/8192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l1169_116939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sister_pairs_l1169_116928

-- Define the function φ(x) = 2e^x + x^2 + 2x
noncomputable def φ (x : ℝ) : ℝ := 2 * Real.exp x + x^2 + 2*x

-- State the theorem
theorem two_sister_pairs :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (-2) 0 ∧ x₂ ∈ Set.Icc (-2) 0 ∧
  φ x₁ = 0 ∧ φ x₂ = 0 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-2) 0 ∧ φ x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sister_pairs_l1169_116928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1169_116963

/-- Represents a train with constant speed -/
structure Train where
  length : ℝ
  speed : ℝ

/-- The time taken by a train to cross a platform -/
noncomputable def crossingTime (t : Train) (platformLength : ℝ) : ℝ :=
  (t.length + platformLength) / t.speed

theorem train_length_calculation (t : Train) 
  (h1 : crossingTime t 110 = 15)
  (h2 : crossingTime t 250 = 20) : 
  t.length = 310 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1169_116963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_correct_statements_two_statements_correct_l1169_116901

open Real

-- Define the statements
noncomputable def statement_A : Prop := ∀ (A : ℝ), sin A = 1/2 → A = π/6
def statement_B : Prop := ∀ (A : ℝ), cos (2*π - A) = cos A
def statement_C : Prop := ∀ (A : ℝ), sin A > 0 ∧ cos A > 0
noncomputable def statement_D : Prop := (sin (130 * π/180))^2 + (sin (140 * π/180))^2 = 1

-- Define a function to count the number of true statements
def count_true_statements (a b c d : Bool) : ℕ :=
  (if a then 1 else 0) + (if b then 1 else 0) + (if c then 1 else 0) + (if d then 1 else 0)

-- The theorem to prove
theorem number_of_correct_statements :
  count_true_statements false true false true = 2 := by
  rfl

-- Proof that exactly two statements are correct
theorem two_statements_correct :
  ¬statement_A ∧ statement_B ∧ ¬statement_C ∧ statement_D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_correct_statements_two_statements_correct_l1169_116901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_function_unique_l1169_116918

def NatPos := { n : ℕ // n > 0 }

theorem identity_function_unique 
  (f : NatPos → NatPos) 
  (h_increasing : ∀ x y : NatPos, (x.1 < y.1) → (f x).1 < (f y).1)
  (h_f2 : f ⟨2, by norm_num⟩ = ⟨2, by norm_num⟩)
  (h_mult : ∀ m n : NatPos, Nat.Coprime m.1 n.1 → 
    f ⟨m.1 * n.1, by {
      have h1 : 0 < m.1 := m.2
      have h2 : 0 < n.1 := n.2
      exact Nat.mul_pos h1 h2
    }⟩ = ⟨(f m).1 * (f n).1, by {
      have h1 : 0 < (f m).1 := (f m).2
      have h2 : 0 < (f n).1 := (f n).2
      exact Nat.mul_pos h1 h2
    }⟩) :
  ∀ n : NatPos, f n = n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_function_unique_l1169_116918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_model_l1169_116953

/-- Represents a car's fuel consumption model -/
structure CarFuel where
  initial_fuel : ℝ
  consumption_rate : ℝ

/-- Remaining fuel as a function of distance traveled -/
noncomputable def remaining_fuel (car : CarFuel) (distance : ℝ) : ℝ :=
  car.initial_fuel - car.consumption_rate * distance

/-- Maximum distance a car can travel without refueling -/
noncomputable def max_distance (car : CarFuel) : ℝ :=
  car.initial_fuel / car.consumption_rate

theorem car_fuel_model (car : CarFuel) 
    (h1 : car.initial_fuel = 48)
    (h2 : car.consumption_rate = 0.6) :
  (∀ x, remaining_fuel car x = -0.6 * x + 48) ∧
  (remaining_fuel car 35 = 27) ∧
  (max_distance car = 80) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_model_l1169_116953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_two_identities_l1169_116920

theorem tangent_two_identities (α : Real) (h : Real.tan α = 2) :
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1/6 ∧
  Real.sin α ^ 2 + Real.sin (2 * α) = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_two_identities_l1169_116920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l1169_116961

noncomputable def f (α : Real) : Real :=
  (Real.cos (Real.pi/2 + α) * Real.cos (2*Real.pi - α) * Real.sin (-α + 3*Real.pi/2)) /
  (Real.sin (-Real.pi - α) * Real.sin (3*Real.pi/2 + α))

theorem f_value_in_third_quadrant (α : Real) 
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2) -- α is in the third quadrant
  (h2 : Real.cos (α - 3*Real.pi/2) = 1/5) : -- given condition
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l1169_116961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compost_loading_time_l1169_116938

/-- The rate at which Steven's tractor can scoop up compost in pounds per minute. -/
noncomputable def stevens_rate : ℝ := 75

/-- The rate at which Darrel can scoop up compost in pounds per minute. -/
noncomputable def darrels_rate : ℝ := 10

/-- The total amount of compost to be loaded in pounds. -/
noncomputable def total_compost : ℝ := 2550

/-- The time required for Steven and Darrel to load the compost together. -/
noncomputable def loading_time : ℝ := total_compost / (stevens_rate + darrels_rate)

theorem compost_loading_time :
  loading_time = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compost_loading_time_l1169_116938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sorting_possible_l1169_116962

/-- A function that represents the cost of sorting n cards -/
def sortingCost (n : ℕ) : ℕ := sorry

/-- The maximum number of cards that can be sorted with k operations -/
def maxSortableCards (k : ℕ) : ℕ := sorry

/-- The total number of cards -/
def totalCards : ℕ := 365

/-- The maximum number of allowed operations -/
def maxOperations : ℕ := 2000

/-- Theorem stating that the sorting cost for all cards is within the allowed operations -/
theorem sorting_possible : sortingCost totalCards ≤ maxOperations := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sorting_possible_l1169_116962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_on_elliptical_table_l1169_116903

/-- The length of a place mat on an elliptical table -/
noncomputable def place_mat_length (a b : ℝ) (n : ℕ) : ℝ :=
  2 * a * Real.sin (Real.pi / (2 * n))

/-- Approximate equality with a small tolerance -/
def approx_equal (x y : ℝ) (ε : ℝ) : Prop :=
  |x - y| < ε

theorem place_mat_length_on_elliptical_table :
  ∀ (ε : ℝ), ε > 0 →
  approx_equal (place_mat_length 5 3 8) (5 * Real.sqrt (2 - Real.sqrt 2)) ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_on_elliptical_table_l1169_116903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_partition_l1169_116909

/-- A partition of n into k parts is a list of k positive integers that sum to n. -/
def IsPartition (n k : ℕ) (partition : List ℕ) : Prop :=
  partition.length = k ∧ partition.sum = n ∧ ∀ x, x ∈ partition → x > 0

/-- All elements in the list have ratio less than 2 with each other. -/
def AllRatiosLessThanTwo (list : List ℕ) : Prop :=
  ∀ x y, x ∈ list → y ∈ list → (x : ℚ) / y < 2 ∧ (y : ℚ) / x < 2

theorem stones_partition :
  ∃ partition : List ℕ, IsPartition 660 30 partition ∧ AllRatiosLessThanTwo partition := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stones_partition_l1169_116909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_with_equal_intercepts_l1169_116919

/-- Given a line with equation (a+1)x + y + 2 - a = 0 where a is a real number,
    if the line has equal intercepts on both coordinate axes,
    then its equation is either x + y + 2 = 0 or 3x + y = 0 -/
theorem line_with_equal_intercepts (a : ℝ) :
  let line := {p : ℝ × ℝ | (a + 1) * p.1 + p.2 + 2 - a = 0}
  let x_intercept := (a - 2) / (a + 1)
  let y_intercept := a - 2
  (x_intercept = y_intercept) →
  (∀ x y : ℝ, (x, y) ∈ line ↔ (x + y + 2 = 0 ∨ 3 * x + y = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_with_equal_intercepts_l1169_116919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_angle_bisector_segment_l1169_116960

/-- Theorem: The segment t of the angle bisector in a trapezoid can be expressed through its sides. -/
theorem trapezoid_angle_bisector_segment 
  (a c b d t p : ℝ) 
  (h_positive : 0 < a ∧ 0 < c ∧ 0 < b ∧ 0 < d)
  (h_semiperimeter : p = (a + b + c + d) / 2) :
  t^2 = (4 * b * d) / (b + d)^2 * (p - a) * (p - c) :=
by
  sorry -- Proof to be completed

#check trapezoid_angle_bisector_segment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_angle_bisector_segment_l1169_116960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_g_minimum_value_l1169_116966

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |a * x - x^2| + Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x / x

theorem f_monotone_intervals :
  MonotoneOn (f 2) (Set.Ioo 0 ((1 + Real.sqrt 3) / 2)) ∧
  MonotoneOn (f 2) (Set.Ioi 2) :=
sorry

theorem g_minimum_value (h : 0 < a) :
  (∃ (m : ℝ), ∀ x ∈ Set.Icc 1 (Real.exp 1), m ≤ g a x) ∧
  (∀ x ∈ Set.Icc 1 (Real.exp 1),
    (0 < a ∧ a ≤ 1 → 1 - a ≤ g a x) ∧
    (1 < a ∧ a < Real.exp 1 → Real.log a / a ≤ g a x) ∧
    (Real.exp 1 ≤ a → a - Real.exp 1 + 1 / Real.exp 1 ≤ g a x)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_g_minimum_value_l1169_116966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l1169_116924

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a function to check if a point is on a line defined by two other points
def isOnLine (p a b : Point) : Prop :=
  (p.y - a.y) * (b.x - a.x) = (b.y - a.y) * (p.x - a.x)

-- Define the theorem
theorem point_coordinates : 
  let a : Point := ⟨3, -4⟩
  let b : Point := ⟨-1, 2⟩
  ∀ p : Point, 
    isOnLine p a b → 
    distance a p = 2 * distance p b → 
    (p.x = 1/3 ∧ p.y = 0) ∨ (p.x = -5 ∧ p.y = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l1169_116924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l1169_116950

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (8 + 2*x - x^2) / Real.log 0.5

-- Define the domain of f
def domain : Set ℝ := Set.Ioo (-2) 4

-- State the theorem
theorem interval_of_increase :
  ∃ (a b : ℝ), a = 1 ∧ b = 4 ∧
  (StrictMonoOn f (Set.Icc a b ∩ domain)) ∧
  (∀ x ∈ domain, x < a → ¬StrictMonoOn f (Set.Icc x a ∩ domain)) ∧
  (∀ x ∈ domain, b < x → ¬StrictMonoOn f (Set.Icc b x ∩ domain)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l1169_116950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1169_116921

/-- The terminal side of an angle in the coordinate plane -/
def terminal_side (α : Real) : Set (Real × Real) :=
  {P | ∃ r : Real, r > 0 ∧ P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α}

theorem tan_alpha_value (α : Real) (m : Real) :
  (∃ P : Real × Real, P.1 = -Real.sqrt 3 ∧ P.2 = m ∧ P ∈ terminal_side α) →
  (Real.sin α = (Real.sqrt 2 / 4) * m) →
  Real.tan α = 1 ∨ Real.tan α = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1169_116921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_bn_bnplus1_eq_one_l1169_116956

def b (n : ℕ) : ℤ := (8^n - 1) / 7

theorem gcd_bn_bnplus1_eq_one (n : ℕ) : Int.gcd (b n) (b (n + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_bn_bnplus1_eq_one_l1169_116956
