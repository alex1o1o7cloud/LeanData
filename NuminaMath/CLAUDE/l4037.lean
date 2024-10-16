import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l4037_403735

theorem geometric_sequence_first_term
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r^2 = 3) -- third term is 3
  (h2 : a * r^4 = 27) -- fifth term is 27
  : a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l4037_403735


namespace NUMINAMATH_CALUDE_unique_prime_triple_l4037_403749

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 0 → d ∣ n → d = 1 ∨ d = n

theorem unique_prime_triple :
  ∃! (p q r : ℕ), isPrime p ∧ isPrime q ∧ isPrime r ∧ p = q + 2 ∧ q = r + 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l4037_403749


namespace NUMINAMATH_CALUDE_village_population_l4037_403751

/-- Represents the vampire population dynamics in a village --/
structure VampireVillage where
  initialVampires : ℕ
  initialPopulation : ℕ
  vampiresPerNight : ℕ
  nightsPassed : ℕ
  finalVampires : ℕ

/-- Theorem stating the initial population of the village --/
theorem village_population (v : VampireVillage) 
  (h1 : v.initialVampires = 2)
  (h2 : v.vampiresPerNight = 5)
  (h3 : v.nightsPassed = 2)
  (h4 : v.finalVampires = 72) :
  v.initialPopulation = 72 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l4037_403751


namespace NUMINAMATH_CALUDE_factor_theorem_l4037_403750

def Q (d : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + d*x + 20

theorem factor_theorem (d : ℝ) : (∀ x, (x - 4) ∣ Q d x) → d = -33 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_l4037_403750


namespace NUMINAMATH_CALUDE_function_evaluation_l4037_403708

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x, f x = 4*x - 2) : f (-3) = -14 := by
  sorry

end NUMINAMATH_CALUDE_function_evaluation_l4037_403708


namespace NUMINAMATH_CALUDE_pears_amount_correct_l4037_403748

/-- The amount of peaches received by the store in kilograms. -/
def peaches : ℕ := 250

/-- The amount of pears received by the store in kilograms. -/
def pears : ℕ := 100

/-- Theorem stating that the amount of pears is correct given the conditions. -/
theorem pears_amount_correct : peaches = 2 * pears + 50 := by sorry

end NUMINAMATH_CALUDE_pears_amount_correct_l4037_403748


namespace NUMINAMATH_CALUDE_derivative_at_zero_l4037_403740

/-- Given a function f(x) = e^x + sin x - cos x, prove that its derivative at x = 0 is 2 -/
theorem derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp x + Real.sin x - Real.cos x) :
  deriv f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l4037_403740


namespace NUMINAMATH_CALUDE_brianas_yield_percentage_l4037_403762

theorem brianas_yield_percentage (emma_investment briana_investment : ℝ)
                                 (emma_yield : ℝ)
                                 (investment_period : ℕ)
                                 (roi_difference : ℝ) :
  emma_investment = 300 →
  briana_investment = 500 →
  emma_yield = 0.15 →
  investment_period = 2 →
  roi_difference = 10 →
  briana_investment * (investment_period : ℝ) * (briana_yield / 100) -
  emma_investment * (investment_period : ℝ) * emma_yield = roi_difference →
  briana_yield = 10 :=
by
  sorry

#check brianas_yield_percentage

end NUMINAMATH_CALUDE_brianas_yield_percentage_l4037_403762


namespace NUMINAMATH_CALUDE_max_sum_of_goods_l4037_403700

theorem max_sum_of_goods (m n : ℕ) : m > 0 ∧ n > 0 ∧ 5 * m + 17 * n = 203 → m + n ≤ 31 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_goods_l4037_403700


namespace NUMINAMATH_CALUDE_second_workshop_production_l4037_403746

/-- Given three workshops producing boots with samples forming an arithmetic sequence,
    prove that the second workshop's production is 1200 pairs. -/
theorem second_workshop_production
  (total_production : ℕ)
  (a b c : ℕ)
  (h1 : total_production = 3600)
  (h2 : a + c = 2 * b)  -- arithmetic sequence property
  (h3 : a + b + c > 0)  -- ensure division is valid
  : (b : ℚ) / (a + b + c : ℚ) * total_production = 1200 :=
by sorry

end NUMINAMATH_CALUDE_second_workshop_production_l4037_403746


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_18_20_l4037_403731

theorem smallest_divisible_by_15_18_20 : ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 18 ∣ n ∧ 20 ∣ n ∧ ∀ (m : ℕ), m > 0 → 15 ∣ m → 18 ∣ m → 20 ∣ m → n ≤ m :=
by
  use 180
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_18_20_l4037_403731


namespace NUMINAMATH_CALUDE_probability_blue_between_red_and_triple_red_l4037_403792

-- Define the probability space
def Ω : Type := ℝ × ℝ

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the event where the blue point is greater than the red point but less than three times the red point
def E : Set Ω := {ω : Ω | let (x, y) := ω; 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y ∧ y < 3*x}

-- State the theorem
theorem probability_blue_between_red_and_triple_red : P E = 5/6 := sorry

end NUMINAMATH_CALUDE_probability_blue_between_red_and_triple_red_l4037_403792


namespace NUMINAMATH_CALUDE_function_equation_solution_l4037_403744

theorem function_equation_solution (a b : ℚ) :
  ∃ (f : ℚ → ℚ), (∀ x y : ℚ, f (x + a + f y) = f (x + b) + y) →
  ∃ A : ℚ, ∀ x : ℚ, f x = A * x + (a - b) / 2 := by
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l4037_403744


namespace NUMINAMATH_CALUDE_probability_of_shaded_triangle_l4037_403706

/-- Given a set of triangles where some are shaded, this theorem proves
    the probability of selecting a shaded triangle when each triangle
    has an equal probability of being selected. -/
theorem probability_of_shaded_triangle 
  (total_triangles : ℕ) 
  (shaded_triangles : ℕ) 
  (h1 : total_triangles = 8) 
  (h2 : shaded_triangles = 4) 
  (h3 : shaded_triangles ≤ total_triangles) :
  (shaded_triangles : ℚ) / total_triangles = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_shaded_triangle_l4037_403706


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4037_403734

/-- Given a real number a such that (2+ai)/(1+i) = 3+i, prove that a = 4 -/
theorem complex_equation_solution (a : ℝ) : (2 + a * Complex.I) / (1 + Complex.I) = 3 + Complex.I → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4037_403734


namespace NUMINAMATH_CALUDE_pipe_fill_time_l4037_403747

/-- Given a pipe and a tank with a leak, this theorem proves the time taken for the pipe
    to fill the tank alone, based on the time taken to fill with both pipe and leak,
    and the time taken for the leak to empty the tank. -/
theorem pipe_fill_time (fill_time_with_leak : ℝ) (leak_empty_time : ℝ) 
    (h1 : fill_time_with_leak = 18) 
    (h2 : leak_empty_time = 36) : 
    (1 : ℝ) / ((1 : ℝ) / fill_time_with_leak + (1 : ℝ) / leak_empty_time) = 12 := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l4037_403747


namespace NUMINAMATH_CALUDE_set_intersection_conditions_l4037_403789

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = 2*x - 1 ∧ 0 < x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) < 0}

-- State the theorem
theorem set_intersection_conditions (a : ℝ) :
  (A ∩ B a = A ↔ a ∈ Set.Ioc (-2) (-1)) ∧
  (A ∩ B a ≠ ∅ ↔ a ∈ Set.Ioo (-4) 1) :=
sorry

end NUMINAMATH_CALUDE_set_intersection_conditions_l4037_403789


namespace NUMINAMATH_CALUDE_sqrt_6_between_2_and_3_l4037_403739

theorem sqrt_6_between_2_and_3 : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_6_between_2_and_3_l4037_403739


namespace NUMINAMATH_CALUDE_george_sticker_count_l4037_403732

/-- Given the following sticker counts:
  * Dan has 2 times as many stickers as Tom
  * Tom has 3 times as many stickers as Bob
  * George has 5 times as many stickers as Dan
  * Bob has 12 stickers
  Prove that George has 360 stickers -/
theorem george_sticker_count :
  ∀ (bob tom dan george : ℕ),
    dan = 2 * tom →
    tom = 3 * bob →
    george = 5 * dan →
    bob = 12 →
    george = 360 := by
  sorry

end NUMINAMATH_CALUDE_george_sticker_count_l4037_403732


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_two_l4037_403796

theorem sqrt_expression_equals_two :
  Real.sqrt 4 + Real.sqrt 2 * Real.sqrt 6 - 6 * Real.sqrt (1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_two_l4037_403796


namespace NUMINAMATH_CALUDE_base4_calculation_l4037_403775

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Multiplies two base 4 numbers --/
def multiplyBase4 (a b : ℕ) : ℕ := sorry

/-- Divides two base 4 numbers --/
def divideBase4 (a b : ℕ) : ℕ := sorry

/-- Subtracts two base 4 numbers --/
def subtractBase4 (a b : ℕ) : ℕ := sorry

theorem base4_calculation : 
  let a := 230
  let b := 21
  let c := 2
  let d := 12
  let e := 3
  subtractBase4 (divideBase4 (multiplyBase4 a b) c) (multiplyBase4 d e) = 3222 := by
  sorry

end NUMINAMATH_CALUDE_base4_calculation_l4037_403775


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l4037_403723

/-- Proves that a train with given length and speed takes a specific time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) : 
  train_length = 180 ∧ 
  train_speed_kmh = 72 ∧ 
  bridge_length = 270 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l4037_403723


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l4037_403754

theorem reciprocal_of_negative_fraction (n : ℤ) (h : n ≠ 0) :
  (-(1 : ℚ) / n)⁻¹ = -n := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l4037_403754


namespace NUMINAMATH_CALUDE_sum_even_10_mod_6_l4037_403763

/-- The sum of the first n even numbers starting from 2 -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- The theorem stating that the remainder of the sum of the first 10 even numbers divided by 6 is 2 -/
theorem sum_even_10_mod_6 : sum_even 10 % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_10_mod_6_l4037_403763


namespace NUMINAMATH_CALUDE_division_remainder_problem_l4037_403705

theorem division_remainder_problem : ∃ r, 0 ≤ r ∧ r < 9 ∧ 83 = 9 * 9 + r := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l4037_403705


namespace NUMINAMATH_CALUDE_simplify_fraction_l4037_403773

theorem simplify_fraction (a : ℝ) (ha : a > 0) :
  a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4037_403773


namespace NUMINAMATH_CALUDE_candy_mixture_problem_l4037_403761

theorem candy_mixture_problem (X Y : ℝ) : 
  X + Y = 10 →
  3.50 * X + 4.30 * Y = 40 →
  Y = 6.25 := by
sorry

end NUMINAMATH_CALUDE_candy_mixture_problem_l4037_403761


namespace NUMINAMATH_CALUDE_game_cost_l4037_403777

/-- The cost of a new game given initial money, birthday gift, and remaining money -/
theorem game_cost (initial : ℕ) (gift : ℕ) (remaining : ℕ) : 
  initial = 16 → gift = 28 → remaining = 19 → initial + gift - remaining = 25 := by
  sorry

end NUMINAMATH_CALUDE_game_cost_l4037_403777


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4037_403757

theorem min_value_quadratic (x y : ℝ) : x^2 + 4*x*y + 5*y^2 - 8*x - 6*y ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4037_403757


namespace NUMINAMATH_CALUDE_original_number_calculation_l4037_403793

theorem original_number_calculation (r : ℝ) : 
  (r + 0.15 * r) - (r - 0.30 * r) = 40 → r = 40 / 0.45 := by
sorry

end NUMINAMATH_CALUDE_original_number_calculation_l4037_403793


namespace NUMINAMATH_CALUDE_total_calculators_l4037_403720

/-- Represents the number of calculators assembled by a person in a unit of time -/
structure AssemblyRate where
  calculators : ℕ
  time_units : ℕ

/-- The problem setup -/
def calculator_problem (erika nick sam : AssemblyRate) : Prop :=
  -- Erika assembles 3 calculators in the same time Nick assembles 2
  erika.calculators * nick.time_units = 3 * nick.calculators * erika.time_units ∧
  -- Nick assembles 1 calculator in the same time Sam assembles 3
  nick.calculators * sam.time_units = sam.calculators * nick.time_units ∧
  -- Erika's rate is 3 calculators per time unit
  erika.calculators = 3 ∧ erika.time_units = 1

/-- The theorem to prove -/
theorem total_calculators (erika nick sam : AssemblyRate) 
  (h : calculator_problem erika nick sam) : 
  9 * erika.time_units / erika.calculators * 
  (erika.calculators + nick.calculators * erika.time_units / nick.time_units + 
   sam.calculators * erika.time_units / sam.time_units) = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_calculators_l4037_403720


namespace NUMINAMATH_CALUDE_bisecting_chord_equation_l4037_403783

/-- The equation of a line bisecting a chord of a parabola -/
theorem bisecting_chord_equation (x y : ℝ → ℝ) :
  (∀ t, (y t)^2 = 16 * (x t)) →  -- Parabola equation
  (∃ t₁ t₂, t₁ ≠ t₂ ∧ 
    (x t₁ + x t₂) / 2 = 2 ∧ 
    (y t₁ + y t₂) / 2 = 1) →  -- Midpoint condition
  (∃ a b c : ℝ, ∀ t, a * (x t) + b * (y t) + c = 0 ∧ 
    a = 8 ∧ b = -1 ∧ c = -15) := by
sorry

end NUMINAMATH_CALUDE_bisecting_chord_equation_l4037_403783


namespace NUMINAMATH_CALUDE_no_divisibility_l4037_403733

theorem no_divisibility (d a n : ℕ) (h1 : 3 ≤ d) (h2 : d ≤ 2^(n+1)) :
  ¬(d ∣ a^(2^n) + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_divisibility_l4037_403733


namespace NUMINAMATH_CALUDE_zero_smallest_natural_l4037_403703

theorem zero_smallest_natural : ∀ n : ℕ, 0 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_zero_smallest_natural_l4037_403703


namespace NUMINAMATH_CALUDE_embankment_height_bounds_l4037_403760

/-- Represents the properties of a trapezoidal embankment -/
structure Embankment where
  length : ℝ
  lower_base : ℝ
  slope_angle : ℝ
  volume_min : ℝ
  volume_max : ℝ

/-- Theorem stating the height bounds for the embankment -/
theorem embankment_height_bounds (e : Embankment)
  (h_length : e.length = 100)
  (h_lower_base : e.lower_base = 5)
  (h_slope_angle : e.slope_angle = π/4)
  (h_volume : e.volume_min = 400 ∧ e.volume_max = 500)
  (h_upper_base_min : ∀ b, b ≥ 2 → 
    400 ≤ 25 * (5^2 - b^2) ∧ 25 * (5^2 - b^2) ≤ 500) :
  ∃ (h : ℝ), 1 ≤ h ∧ h ≤ (5 - Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_embankment_height_bounds_l4037_403760


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l4037_403730

/-- Given a line L1: 4x - 3y = 12, prove that a line L2 perpendicular to L1 
    with y-intercept 3 has x-intercept 4 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y => 4 * x - 3 * y = 12
  let m1 : ℝ := 4 / 3  -- slope of L1
  let m2 : ℝ := -3 / 4  -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y => y = m2 * x + 3  -- L2 with y-intercept 3
  ∃ x : ℝ, x = 4 ∧ L2 x 0 :=
by
  sorry

#check perpendicular_line_x_intercept

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l4037_403730


namespace NUMINAMATH_CALUDE_jordan_danielle_roses_l4037_403791

def roses_remaining (initial : ℕ) (additional : ℕ) : ℕ :=
  let total := initial + additional
  let after_first_day := total / 2
  let after_second_day := after_first_day / 2
  after_second_day

theorem jordan_danielle_roses : roses_remaining 24 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jordan_danielle_roses_l4037_403791


namespace NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l4037_403736

/-- The equation x^2 - 9y^2 = 0 represents a pair of straight lines -/
theorem equation_represents_pair_of_lines :
  ∃ (m₁ m₂ : ℝ), ∀ (x y : ℝ), x^2 - 9*y^2 = 0 ↔ (y = m₁*x ∨ y = m₂*x) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l4037_403736


namespace NUMINAMATH_CALUDE_loom_weaving_time_l4037_403743

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

end NUMINAMATH_CALUDE_loom_weaving_time_l4037_403743


namespace NUMINAMATH_CALUDE_revenue_decrease_l4037_403721

/-- Proves that a 43.529411764705884% decrease to $48.0 billion results in an original revenue of $85.0 billion -/
theorem revenue_decrease (current_revenue : ℝ) (decrease_percentage : ℝ) (original_revenue : ℝ) :
  current_revenue = 48.0 ∧
  decrease_percentage = 43.529411764705884 ∧
  current_revenue = original_revenue * (1 - decrease_percentage / 100) →
  original_revenue = 85.0 := by
sorry

end NUMINAMATH_CALUDE_revenue_decrease_l4037_403721


namespace NUMINAMATH_CALUDE_fraction_equality_l4037_403725

theorem fraction_equality (a b : ℝ) (h : a ≠ 0) : b / a = (a * b) / (a^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4037_403725


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l4037_403712

theorem imaginary_part_of_complex_product : Complex.im ((5 + Complex.I) * (1 - Complex.I)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l4037_403712


namespace NUMINAMATH_CALUDE_prism_volume_l4037_403727

/-- The volume of a prism with an isosceles right triangular base and given dimensions -/
theorem prism_volume (leg : ℝ) (height : ℝ) (h_leg : leg = Real.sqrt 5) (h_height : height = 10) :
  (1 / 2) * leg * leg * height = 25 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l4037_403727


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4037_403722

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  b * (Real.sin B - Real.sin C) + (c - a) * (Real.sin A + Real.sin C) = 0 →
  a = Real.sqrt 3 →
  Real.sin C = (1 + Real.sqrt 3) / 2 * Real.sin B →
  -- Conclusions
  A = π / 3 ∧
  (1 / 2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 4 := by
sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l4037_403722


namespace NUMINAMATH_CALUDE_circle_trajectory_l4037_403765

/-- A circle with equation x^2 + y^2 - ax + 2y + 1 = 0 -/
def circle1 (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 1 = 0

/-- The unit circle with equation x^2 + y^2 = 1 -/
def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- The line y = x - l -/
def symmetry_line (l : ℝ) (x y : ℝ) : Prop :=
  y = x - l

/-- Circle P passes through the point C(-a, a) -/
def circle_p_passes_through (a : ℝ) (x y : ℝ) : Prop :=
  (x + a)^2 + (y - a)^2 = x^2 + y^2

/-- Circle P is tangent to the y-axis -/
def circle_p_tangent_y_axis (x y : ℝ) : Prop :=
  x^2 + y^2 = x^2

/-- The trajectory equation of the center P -/
def trajectory_equation (x y : ℝ) : Prop :=
  y^2 + 4*x - 4*y + 8 = 0

theorem circle_trajectory :
  ∀ (a l : ℝ) (x y : ℝ),
  (∃ (x₁ y₁ : ℝ), circle1 a x₁ y₁) →
  (∃ (x₂ y₂ : ℝ), circle2 x₂ y₂) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), circle1 a x₁ y₁ → circle2 x₂ y₂ → 
    ∃ (x₃ y₃ : ℝ), symmetry_line l x₃ y₃ ∧ 
    (x₃ = (x₁ + x₂) / 2 ∧ y₃ = (y₁ + y₂) / 2)) →
  circle_p_passes_through a x y →
  circle_p_tangent_y_axis x y →
  trajectory_equation x y :=
by sorry

end NUMINAMATH_CALUDE_circle_trajectory_l4037_403765


namespace NUMINAMATH_CALUDE_stella_toilet_paper_l4037_403711

/-- The number of packs of toilet paper Stella needs to buy for 4 weeks -/
def toilet_paper_packs (bathrooms : ℕ) (rolls_per_bathroom_per_day : ℕ) 
  (days_per_week : ℕ) (rolls_per_pack : ℕ) (weeks : ℕ) : ℕ :=
  (bathrooms * rolls_per_bathroom_per_day * days_per_week * weeks) / rolls_per_pack

/-- Stella's toilet paper restocking problem -/
theorem stella_toilet_paper : 
  toilet_paper_packs 6 1 7 12 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_stella_toilet_paper_l4037_403711


namespace NUMINAMATH_CALUDE_books_per_shelf_l4037_403724

theorem books_per_shelf (mystery_shelves : ℕ) (picture_shelves : ℕ) (total_books : ℕ) :
  mystery_shelves = 6 →
  picture_shelves = 2 →
  total_books = 72 →
  total_books / (mystery_shelves + picture_shelves) = 9 :=
by sorry

end NUMINAMATH_CALUDE_books_per_shelf_l4037_403724


namespace NUMINAMATH_CALUDE_range_of_a_l4037_403713

open Real

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the theorem
theorem range_of_a (a : ℝ) (h : (¬p a) ∧ q a) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4037_403713


namespace NUMINAMATH_CALUDE_slope_of_line_l4037_403782

/-- The slope of a line given by the equation 4y = 5x - 8 is 5/4 -/
theorem slope_of_line (x y : ℝ) : 4 * y = 5 * x - 8 → (∃ m b : ℝ, y = m * x + b ∧ m = 5 / 4) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l4037_403782


namespace NUMINAMATH_CALUDE_chemistry_class_average_l4037_403709

theorem chemistry_class_average (n₁ n₂ n₃ n₄ : ℕ) (m₁ m₂ m₃ m₄ : ℚ) :
  let total_students := n₁ + n₂ + n₃ + n₄
  let total_marks := n₁ * m₁ + n₂ * m₂ + n₃ * m₃ + n₄ * m₄
  total_marks / total_students = (n₁ * m₁ + n₂ * m₂ + n₃ * m₃ + n₄ * m₄) / (n₁ + n₂ + n₃ + n₄) :=
by
  sorry

#eval (60 * 50 + 35 * 60 + 45 * 55 + 42 * 45) / (60 + 35 + 45 + 42)

end NUMINAMATH_CALUDE_chemistry_class_average_l4037_403709


namespace NUMINAMATH_CALUDE_combined_average_age_l4037_403786

theorem combined_average_age (x_count y_count : ℕ) (x_avg y_avg : ℝ) 
  (hx : x_count = 8) (hy : y_count = 5) 
  (hxa : x_avg = 30) (hya : y_avg = 45) : 
  (x_count * x_avg + y_count * y_avg) / (x_count + y_count) = 36 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_age_l4037_403786


namespace NUMINAMATH_CALUDE_brick_length_calculation_l4037_403737

/-- Proves that given a wall of dimensions 8 m x 6 m x 22.5 m and 6400 bricks each measuring x cm x 11.25 cm x 6 cm, the length x of each brick is 2500 cm. -/
theorem brick_length_calculation (wall_length : Real) (wall_width : Real) (wall_height : Real)
  (brick_count : Nat) (brick_width : Real) (brick_height : Real) :
  wall_length = 8 →
  wall_width = 6 →
  wall_height = 22.5 →
  brick_count = 6400 →
  brick_width = 11.25 →
  brick_height = 6 →
  ∃ (brick_length : Real),
    brick_length = 2500 ∧
    (wall_length * 100) * (wall_width * 100) * (wall_height * 100) =
    brick_count * brick_length * brick_width * brick_height :=
by sorry


end NUMINAMATH_CALUDE_brick_length_calculation_l4037_403737


namespace NUMINAMATH_CALUDE_quadratic_function_signs_l4037_403778

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  opens_downward : a < 0
  positive_y_intercept : c > 0
  has_two_roots : ∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ > 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0

/-- Theorem stating the signs of a, b, and c for a quadratic function with specific properties -/
theorem quadratic_function_signs (f : QuadraticFunction) : f.a < 0 ∧ f.b > 0 ∧ f.c > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_signs_l4037_403778


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l4037_403798

/-- The sampling interval for systematic sampling -/
def sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The sampling interval for a population of 1200 and sample size of 40 is 30 -/
theorem systematic_sampling_interval :
  sampling_interval 1200 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l4037_403798


namespace NUMINAMATH_CALUDE_range_of_g_l4037_403742

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_g :
  Set.range g = {y | -π/2 ≤ y ∧ y ≤ Real.arctan 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l4037_403742


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l4037_403770

theorem parallelogram_perimeter (n : ℕ) (h : n = 92) :
  ∃ (a b : ℕ), a * b = n ∧ (2 * a + 2 * b = 94 ∨ 2 * a + 2 * b = 50) :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l4037_403770


namespace NUMINAMATH_CALUDE_not_square_product_l4037_403774

theorem not_square_product (a : ℕ) : 
  (∀ n : ℕ, ¬∃ m : ℕ, n * (n + a) = m ^ 2) ↔ a = 1 ∨ a = 2 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_not_square_product_l4037_403774


namespace NUMINAMATH_CALUDE_quadratic_even_iff_b_eq_zero_l4037_403758

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b*x + c

/-- Theorem: f(x) = x^2 + bx + c is an even function if and only if b = 0 -/
theorem quadratic_even_iff_b_eq_zero (b c : ℝ) :
  IsEven (f b c) ↔ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_even_iff_b_eq_zero_l4037_403758


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l4037_403776

theorem polynomial_divisibility (W : ℕ → ℤ) :
  (∀ n : ℕ, (2^n - 1) % W n = 0) →
  (∀ n : ℕ, W n = 1 ∨ W n = -1 ∨ W n = 2*n - 1 ∨ W n = -2*n + 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l4037_403776


namespace NUMINAMATH_CALUDE_ralphSockPurchase_l4037_403772

/-- Represents the number of socks bought at each price point -/
structure SockPurchase where
  oneDollar : Nat
  twoDollar : Nat
  fourDollar : Nat

/-- Checks if the SockPurchase satisfies the problem conditions -/
def isValidPurchase (p : SockPurchase) : Prop :=
  p.oneDollar + p.twoDollar + p.fourDollar = 10 ∧
  p.oneDollar + 2 * p.twoDollar + 4 * p.fourDollar = 30 ∧
  p.oneDollar ≥ 1 ∧ p.twoDollar ≥ 1 ∧ p.fourDollar ≥ 1

/-- Theorem stating that the only valid purchase has 2 pairs of $1 socks -/
theorem ralphSockPurchase :
  ∀ p : SockPurchase, isValidPurchase p → p.oneDollar = 2 :=
by sorry

end NUMINAMATH_CALUDE_ralphSockPurchase_l4037_403772


namespace NUMINAMATH_CALUDE_quadratic_fixed_point_l4037_403794

-- Define the quadratic function
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the theorem
theorem quadratic_fixed_point 
  (p q : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 3 5, |f p q x| ≤ 1/2)
  (h2 : f p q ((7 + Real.sqrt 15) / 2) = 0) :
  (f p q)^[2017] ((7 + Real.sqrt 15) / 2) = (7 - Real.sqrt 15) / 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_fixed_point_l4037_403794


namespace NUMINAMATH_CALUDE_exterior_angle_square_octagon_exterior_angle_is_135_l4037_403729

/-- The exterior angle of a square and a regular octagon sharing a common side is 135°. -/
theorem exterior_angle_square_octagon : ℝ → Prop :=
  fun angle =>
    let square_angle := 90
    let octagon_interior_angle := 135
    let exterior_angle := 360 - square_angle - octagon_interior_angle
    exterior_angle = angle

/-- The theorem statement -/
theorem exterior_angle_is_135 : exterior_angle_square_octagon 135 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_square_octagon_exterior_angle_is_135_l4037_403729


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l4037_403710

theorem root_sum_reciprocal (a b c : ℝ) : 
  (a^3 - a - 2 = 0) → 
  (b^3 - b - 2 = 0) → 
  (c^3 - c - 2 = 0) → 
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = -3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l4037_403710


namespace NUMINAMATH_CALUDE_scooter_repair_cost_l4037_403717

/-- Calculates the repair cost of a scooter given the conditions of the problem -/
def repair_cost (cost : ℝ) : ℝ :=
  0.1 * cost

/-- Calculates the selling price of a scooter given the conditions of the problem -/
def selling_price (cost : ℝ) : ℝ :=
  1.2 * cost

/-- Theorem stating the repair cost under the given conditions -/
theorem scooter_repair_cost :
  ∀ (cost : ℝ),
  cost > 0 →
  selling_price cost - cost = 1100 →
  repair_cost cost = 550 := by
sorry

end NUMINAMATH_CALUDE_scooter_repair_cost_l4037_403717


namespace NUMINAMATH_CALUDE_negative_double_inequality_l4037_403787

theorem negative_double_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_double_inequality_l4037_403787


namespace NUMINAMATH_CALUDE_secant_length_l4037_403745

/-- Given a circle with center O and radius r, and a point A outside the circle,
    this theorem proves the length of a secant line from A with internal segment length d. -/
theorem secant_length (O A : Point) (r d a : ℝ) (h1 : r > 0) (h2 : d > 0) (h3 : a > r) :
  ∃ x : ℝ, x = d / 2 + Real.sqrt ((d / 2) ^ 2 + a * (a + 2 * r)) ∨
           x = d / 2 - Real.sqrt ((d / 2) ^ 2 + a * (a + 2 * r)) :=
by sorry

/-- Point type (placeholder) -/
def Point : Type := sorry

end NUMINAMATH_CALUDE_secant_length_l4037_403745


namespace NUMINAMATH_CALUDE_circle_sum_l4037_403759

theorem circle_sum (x y : ℝ) (h : x^2 + y^2 = 8*x - 10*y + 5) : x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_l4037_403759


namespace NUMINAMATH_CALUDE_power_division_l4037_403741

theorem power_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l4037_403741


namespace NUMINAMATH_CALUDE_odd_integer_sum_theorem_l4037_403764

/-- The sum of 60 non-consecutive, odd integers starting from -29 in increasing order -/
def oddIntegerSum : ℤ := 5340

/-- The first term of the sequence -/
def firstTerm : ℤ := -29

/-- The number of terms in the sequence -/
def numTerms : ℕ := 60

/-- The common difference between consecutive terms -/
def commonDiff : ℤ := 4

/-- The last term of the sequence -/
def lastTerm : ℤ := firstTerm + (numTerms - 1) * commonDiff

theorem odd_integer_sum_theorem :
  oddIntegerSum = (numTerms : ℤ) * (firstTerm + lastTerm) / 2 :=
sorry

end NUMINAMATH_CALUDE_odd_integer_sum_theorem_l4037_403764


namespace NUMINAMATH_CALUDE_smallest_possible_b_l4037_403702

theorem smallest_possible_b (a b : ℝ) : 
  (1 < a ∧ a < b) →
  (1 + a ≤ b) →
  (1/a + 1/b ≤ 1) →
  b ≥ (3 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l4037_403702


namespace NUMINAMATH_CALUDE_club_membership_l4037_403785

theorem club_membership (total_members attendance : ℕ) 
  (h1 : total_members = 30)
  (h2 : attendance = 20)
  (h3 : ∃ (men women : ℕ), men + women = total_members ∧ men + women / 3 = attendance) :
  ∃ (men : ℕ), men = 15 ∧ 
    ∃ (women : ℕ), men + women = total_members ∧ men + women / 3 = attendance :=
by sorry

end NUMINAMATH_CALUDE_club_membership_l4037_403785


namespace NUMINAMATH_CALUDE_max_value_expression_l4037_403769

theorem max_value_expression (x y : ℝ) : 
  ∃ (M : ℝ), M = 24 - 2 * Real.sqrt 7 ∧ 
  ∀ (a b : ℝ), a ≤ M ∧ 
  (∃ (x y : ℝ), a = (Real.sqrt (9 - Real.sqrt 7) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) * 
                   (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y - Real.cos (2 * y))) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l4037_403769


namespace NUMINAMATH_CALUDE_unique_arrangement_l4037_403779

def is_valid_arrangement (A B C D E F : ℕ) : Prop :=
  A ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  B ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  C ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  D ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  F ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F ∧
  A + D + E = 15 ∧
  7 + C + E = 15 ∧
  9 + C + A = 15 ∧
  A + 8 + F = 15 ∧
  7 + D + F = 15 ∧
  9 + D + B = 15

theorem unique_arrangement :
  ∀ A B C D E F : ℕ,
  is_valid_arrangement A B C D E F →
  A = 4 ∧ B = 1 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_arrangement_l4037_403779


namespace NUMINAMATH_CALUDE_ice_cube_water_cost_l4037_403771

/-- The cost of 1 ounce of water in Pauly's ice cube production --/
theorem ice_cube_water_cost : 
  let pounds_needed : ℝ := 10
  let ounces_per_cube : ℝ := 2
  let pound_per_cube : ℝ := 1/16
  let cubes_per_hour : ℝ := 10
  let cost_per_hour : ℝ := 1.5
  let total_cost : ℝ := 56
  
  let num_cubes : ℝ := pounds_needed / pound_per_cube
  let hours_needed : ℝ := num_cubes / cubes_per_hour
  let ice_maker_cost : ℝ := hours_needed * cost_per_hour
  let water_cost : ℝ := total_cost - ice_maker_cost
  let total_ounces : ℝ := num_cubes * ounces_per_cube
  let cost_per_ounce : ℝ := water_cost / total_ounces
  
  cost_per_ounce = 0.1 := by sorry

end NUMINAMATH_CALUDE_ice_cube_water_cost_l4037_403771


namespace NUMINAMATH_CALUDE_stock_investment_income_l4037_403728

theorem stock_investment_income 
  (investment : ℝ) 
  (stock_percentage : ℝ) 
  (stock_price : ℝ) 
  (face_value : ℝ) 
  (h1 : investment = 6800) 
  (h2 : stock_percentage = 0.20) 
  (h3 : stock_price = 136) 
  (h4 : face_value = 100) : 
  ∃ (annual_income : ℝ), 
    annual_income = 1000 ∧ 
    annual_income = (investment / stock_price) * (stock_percentage * face_value) :=
by
  sorry

end NUMINAMATH_CALUDE_stock_investment_income_l4037_403728


namespace NUMINAMATH_CALUDE_mini_van_tank_capacity_l4037_403799

/-- Proves that the capacity of a mini-van's tank is 65 liters given the specified conditions -/
theorem mini_van_tank_capacity :
  let service_cost : ℝ := 2.20
  let fuel_cost_per_liter : ℝ := 0.70
  let num_mini_vans : ℕ := 4
  let num_trucks : ℕ := 2
  let total_cost : ℝ := 395.4
  let truck_tank_ratio : ℝ := 2.2  -- 120% bigger means 2.2 times the size

  ∃ (mini_van_capacity : ℝ),
    mini_van_capacity > 0 ∧
    (service_cost * (num_mini_vans + num_trucks) +
     fuel_cost_per_liter * (num_mini_vans * mini_van_capacity + num_trucks * (truck_tank_ratio * mini_van_capacity)) = total_cost) ∧
    mini_van_capacity = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_mini_van_tank_capacity_l4037_403799


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4037_403738

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4037_403738


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l4037_403768

theorem circles_internally_tangent : ∃ (C₁ C₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
  (∀ (x y : ℝ), x^2 + y^2 - 6*x + 4*y + 12 = 0 ↔ (x - C₁.1)^2 + (y - C₁.2)^2 = r₁^2) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 14*x - 2*y + 14 = 0 ↔ (x - C₂.1)^2 + (y - C₂.2)^2 = r₂^2) ∧
  (C₂.1 - C₁.1)^2 + (C₂.2 - C₁.2)^2 = (r₂ - r₁)^2 ∧
  r₂ > r₁ := by
  sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l4037_403768


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l4037_403784

theorem toy_store_revenue_ratio :
  ∀ (N D J : ℝ),
  N > 0 →
  N = (2/5) * D →
  D = 3.75 * ((N + J) / 2) →
  J / N = 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l4037_403784


namespace NUMINAMATH_CALUDE_qadi_advice_leads_to_winner_l4037_403716

/-- Represents a son in the problem -/
structure Son where
  camel : Nat  -- Each son has a camel, represented by a natural number

/-- Represents the state of the race -/
structure RaceState where
  son1 : Son
  son2 : Son
  winner : Option Son

/-- The function that determines the winner based on arrival times -/
def determineWinner (arrivalTime1 arrivalTime2 : Nat) : Option Son :=
  if arrivalTime1 > arrivalTime2 then some { camel := 1 }
  else if arrivalTime2 > arrivalTime1 then some { camel := 2 }
  else none

/-- The function that simulates the race -/
def race (initialState : RaceState) : RaceState :=
  let arrivalTime1 := initialState.son1.camel
  let arrivalTime2 := initialState.son2.camel
  { initialState with winner := determineWinner arrivalTime1 arrivalTime2 }

/-- The function that swaps the camels -/
def swapCamels (state : RaceState) : RaceState :=
  { state with
    son1 := { camel := state.son2.camel }
    son2 := { camel := state.son1.camel } }

/-- The main theorem to prove -/
theorem qadi_advice_leads_to_winner (initialState : RaceState) :
  (race (swapCamels initialState)).winner.isSome :=
sorry


end NUMINAMATH_CALUDE_qadi_advice_leads_to_winner_l4037_403716


namespace NUMINAMATH_CALUDE_complex_division_result_l4037_403781

theorem complex_division_result : (3 - Complex.I) / Complex.I = -1 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l4037_403781


namespace NUMINAMATH_CALUDE_number_problem_l4037_403707

theorem number_problem : ∃ n : ℤ, n - 44 = 15 ∧ n = 59 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4037_403707


namespace NUMINAMATH_CALUDE_tan_alpha_value_l4037_403718

theorem tan_alpha_value (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the third quadrant
  (Real.tan (π/4 - α) = (2/3) * Real.tan (α + π)) → 
  Real.tan α = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l4037_403718


namespace NUMINAMATH_CALUDE_line_of_sight_not_blocked_l4037_403780

/-- The curve C: y = 2x^2 -/
def C : ℝ → ℝ := λ x ↦ 2 * x^2

/-- Point A: (0, -2) -/
def A : ℝ × ℝ := (0, -2)

/-- Point B: (3, a), where a is a parameter -/
def B (a : ℝ) : ℝ × ℝ := (3, a)

/-- The line of sight from A to B(a) is not blocked by C if and only if a < 10 -/
theorem line_of_sight_not_blocked (a : ℝ) : 
  (∀ x ∈ Set.Icc A.1 (B a).1, (B a).2 - A.2 > (C x - A.2) * ((B a).1 - A.1) / (x - A.1)) ↔ 
  a < 10 :=
sorry

end NUMINAMATH_CALUDE_line_of_sight_not_blocked_l4037_403780


namespace NUMINAMATH_CALUDE_function_properties_l4037_403755

-- Define the function f(x) = x^3 + ax^2 + b
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 3*x + y - 3 = 0

-- Theorem statement
theorem function_properties (a b : ℝ) :
  (tangent_line 1 (f a b 1)) →
  (∃ (f' : ℝ → ℝ), 
    (∀ x, f' x = 3*x^2 - 6*x) ∧
    (∀ x, x < 0 → (f' x > 0)) ∧
    (∀ x, 0 < x ∧ x < 2 → (f' x < 0)) ∧
    (∀ x, x > 2 → (f' x > 0))) ∧
  (∀ t, t > 0 →
    (t ≤ 2 → 
      (∀ x, x ∈ Set.Icc 0 t → f (-3) 2 x ≤ 2 ∧ f (-3) 2 t ≤ f (-3) 2 x) ∧
      f (-3) 2 t = t^3 - 3*t^2 + 2) ∧
    (2 < t ∧ t ≤ 3 →
      (∀ x, x ∈ Set.Icc 0 t → -2 ≤ f (-3) 2 x ∧ f (-3) 2 x ≤ 2)) ∧
    (t > 3 →
      (∀ x, x ∈ Set.Icc 0 t → -2 ≤ f (-3) 2 x ∧ f (-3) 2 x ≤ f (-3) 2 t) ∧
      f (-3) 2 t = t^3 - 3*t^2 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l4037_403755


namespace NUMINAMATH_CALUDE_deer_count_l4037_403767

theorem deer_count (total : ℕ) 
  (h1 : (total : ℚ) * (1/10) = (total : ℚ) * (1/10))  -- 10% of deer have 8 antlers
  (h2 : (total : ℚ) * (1/10) * (1/4) = (total : ℚ) * (1/10) * (1/4))  -- 25% of 8-antlered deer have albino fur
  (h3 : (total : ℚ) * (1/10) * (1/4) = 23)  -- There are 23 albino 8-antlered deer
  : total = 920 :=
by sorry

end NUMINAMATH_CALUDE_deer_count_l4037_403767


namespace NUMINAMATH_CALUDE_lucas_purchase_problem_l4037_403704

theorem lucas_purchase_problem :
  ∀ (a b c : ℕ),
    a + b + c = 50 →
    50 * a + 400 * b + 500 * c = 10000 →
    a = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_lucas_purchase_problem_l4037_403704


namespace NUMINAMATH_CALUDE_range_of_m_l4037_403790

-- Define propositions p and q
def p (m : ℝ) : Prop := (m - 2) / (m - 3) ≤ 2 / 3

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + m^2 > 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m < -2 ∨ (0 ≤ m ∧ m ≤ 2) ∨ m ≥ 3

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l4037_403790


namespace NUMINAMATH_CALUDE_min_overlap_coffee_tea_l4037_403797

theorem min_overlap_coffee_tea (coffee_drinkers tea_drinkers : ℝ) 
  (h1 : coffee_drinkers = 0.85)
  (h2 : tea_drinkers = 0.80) :
  0.65 ≤ coffee_drinkers + tea_drinkers - 1 :=
sorry

end NUMINAMATH_CALUDE_min_overlap_coffee_tea_l4037_403797


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l4037_403719

theorem arithmetic_sequence_ratio (a : ℕ+ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ+, a (n + 2) + a (n + 1) = 2 * a n) →
  (∀ n : ℕ+, a (n + 1) = a n * q) →
  q = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l4037_403719


namespace NUMINAMATH_CALUDE_triple_sum_squares_and_fourth_powers_l4037_403756

theorem triple_sum_squares_and_fourth_powers (t : ℤ) : 
  (4*t)^2 + (3 - 2*t - t^2)^2 + (3 + 2*t - t^2)^2 = 2*(3 + t^2)^2 ∧
  (4*t)^4 + (3 - 2*t - t^2)^4 + (3 + 2*t - t^2)^4 = 2*(3 + t^2)^4 := by
  sorry

end NUMINAMATH_CALUDE_triple_sum_squares_and_fourth_powers_l4037_403756


namespace NUMINAMATH_CALUDE_tangent_line_constant_l4037_403752

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2

/-- The tangent line function -/
def tangent_line (x b : ℝ) : ℝ := -3*x + b

/-- Theorem stating that if the line y = -3x + b is tangent to the curve y = x^3 - 3x^2, then b = 1 -/
theorem tangent_line_constant (b : ℝ) : 
  (∃ x : ℝ, f x = tangent_line x b ∧ 
    (∀ y : ℝ, y ≠ x → f y ≠ tangent_line y b)) → 
  b = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_constant_l4037_403752


namespace NUMINAMATH_CALUDE_least_lcm_value_l4037_403753

def lcm_problem (a b c : ℕ) : Prop :=
  (Nat.lcm a b = 40) ∧ 
  (Nat.lcm b c = 21) ∧
  (Nat.lcm a c ≥ 24) ∧
  ∀ x y, (Nat.lcm x y = 40) → (Nat.lcm y c = 21) → (Nat.lcm x c ≥ 24)

theorem least_lcm_value : ∃ a b c, lcm_problem a b c ∧ Nat.lcm a c = 24 :=
sorry

end NUMINAMATH_CALUDE_least_lcm_value_l4037_403753


namespace NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l4037_403795

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 2*x^2 + 18*x + 36

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! p : ℝ × ℝ, 
    (∀ x : ℝ, (x, g x) ≠ p → (g x, x) ≠ p) ∧ 
    p.1 = g p.2 ∧ 
    p.2 = g p.1 ∧
    p = (-3, -3) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_of_g_and_inverse_l4037_403795


namespace NUMINAMATH_CALUDE_complex_fraction_fourth_quadrant_l4037_403726

/-- Given that (1+i)/(2-i) = a + (b+1)i where a and b are real numbers and i is the imaginary unit,
    prove that the point corresponding to z = a + bi lies in the fourth quadrant of the complex plane. -/
theorem complex_fraction_fourth_quadrant (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (1 + i) / (2 - i) = a + (b + 1) * i →
  0 < a ∧ b < 0 :=
sorry

end NUMINAMATH_CALUDE_complex_fraction_fourth_quadrant_l4037_403726


namespace NUMINAMATH_CALUDE_sum_even_odd_is_odd_l4037_403766

def P : Set Int := {x | ∃ k, x = 2 * k}
def Q : Set Int := {x | ∃ k, x = 2 * k + 1}
def R : Set Int := {x | ∃ k, x = 4 * k + 1}

theorem sum_even_odd_is_odd (a b : Int) (ha : a ∈ P) (hb : b ∈ Q) : a + b ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_sum_even_odd_is_odd_l4037_403766


namespace NUMINAMATH_CALUDE_product_scaling_l4037_403714

theorem product_scaling (a b c : ℝ) (h : 14.97 * 46 = 688.62) :
  1.497 * 4.6 = 6.8862 := by
  sorry

end NUMINAMATH_CALUDE_product_scaling_l4037_403714


namespace NUMINAMATH_CALUDE_only_prime_three_satisfies_l4037_403715

def set_A (p : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, 1 ≤ k ∧ k ≤ (p - 1) / 2 ∧ x = (k^2 + 1) % p}

def set_B (p g : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, 1 ≤ k ∧ k ≤ (p - 1) / 2 ∧ x = (g^k) % p}

theorem only_prime_three_satisfies (p : ℕ) :
  (Nat.Prime p ∧ Odd p ∧ (∃ g : ℕ, set_A p = set_B p g)) ↔ p = 3 :=
sorry

end NUMINAMATH_CALUDE_only_prime_three_satisfies_l4037_403715


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l4037_403788

theorem smallest_integer_with_remainders : ∃! M : ℕ,
  (M > 0) ∧
  (M % 3 = 2) ∧
  (M % 4 = 3) ∧
  (M % 5 = 4) ∧
  (M % 6 = 5) ∧
  (M % 7 = 6) ∧
  (M % 11 = 10) ∧
  (∀ n : ℕ, n > 0 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 ∧ n % 11 = 10 → n ≥ M) :=
by
  sorry

#eval Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 11)))) - 1

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l4037_403788


namespace NUMINAMATH_CALUDE_spade_operation_result_l4037_403701

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_operation_result : spade 3 (spade 5 (spade 8 12)) = 2 := by sorry

end NUMINAMATH_CALUDE_spade_operation_result_l4037_403701
