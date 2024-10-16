import Mathlib

namespace NUMINAMATH_CALUDE_study_method_is_algorithm_statements_are_not_algorithms_l2639_263959

/-- Represents a series of steps or instructions -/
structure Procedure where
  steps : List String

/-- Represents a statement or fact -/
structure Statement where
  content : String

/-- Definition of an algorithm -/
def is_algorithm (p : Procedure) : Prop :=
  p.steps.length > 0 ∧ ∀ s ∈ p.steps, s ≠ ""

theorem study_method_is_algorithm (study_method : Procedure)
  (h1 : study_method.steps = ["Preview before class", 
                              "Listen carefully and take good notes during class", 
                              "Review first and then do homework after class", 
                              "Do appropriate exercises"]) : 
  is_algorithm study_method := by sorry

theorem statements_are_not_algorithms (s : Statement) : 
  ¬ is_algorithm ⟨[s.content]⟩ := by sorry

#check study_method_is_algorithm
#check statements_are_not_algorithms

end NUMINAMATH_CALUDE_study_method_is_algorithm_statements_are_not_algorithms_l2639_263959


namespace NUMINAMATH_CALUDE_fraction_simplification_l2639_263907

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -3) :
  12 / (x^2 - 9) - 2 / (x - 3) = -2 / (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2639_263907


namespace NUMINAMATH_CALUDE_incorrect_negation_even_multiple_of_seven_l2639_263912

theorem incorrect_negation_even_multiple_of_seven :
  ¬(∀ n : ℕ, ¬(2 * n % 7 = 0)) ↔ ∃ n : ℕ, 2 * n % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_negation_even_multiple_of_seven_l2639_263912


namespace NUMINAMATH_CALUDE_erased_digit_is_four_l2639_263927

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the property that a number is divisible by 9
def divisibleBy9 (n : ℕ) : Prop := n % 9 = 0

-- Main theorem
theorem erased_digit_is_four (N : ℕ) (D : ℕ) (x : ℕ) :
  D = N - sumOfDigits N →
  divisibleBy9 D →
  sumOfDigits D = 131 + x →
  x = 4 := by sorry

end NUMINAMATH_CALUDE_erased_digit_is_four_l2639_263927


namespace NUMINAMATH_CALUDE_quartic_polynomial_satisfies_conditions_l2639_263963

def q (x : ℚ) : ℚ := -1/6 * x^4 + 4/3 * x^3 - 4/3 * x^2 - 8/3 * x

theorem quartic_polynomial_satisfies_conditions :
  q 1 = -3 ∧ q 2 = -5 ∧ q 3 = -9 ∧ q 4 = -17 ∧ q 5 = -35 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_satisfies_conditions_l2639_263963


namespace NUMINAMATH_CALUDE_probability_allison_wins_l2639_263905

def allison_cube : Fin 6 → ℕ := λ _ => 6

def brian_cube : Fin 6 → ℕ := λ i => i.val + 1

def noah_cube : Fin 6 → ℕ
| 0 => 3
| 1 => 3
| _ => 5

def prob_brian_less_or_equal_5 : ℚ := 5 / 6

def prob_noah_less_or_equal_5 : ℚ := 1

theorem probability_allison_wins : ℚ := by
  sorry

end NUMINAMATH_CALUDE_probability_allison_wins_l2639_263905


namespace NUMINAMATH_CALUDE_shells_given_to_brother_l2639_263997

def shells_per_day : ℕ := 10
def days_collecting : ℕ := 6
def shells_remaining : ℕ := 58

theorem shells_given_to_brother :
  shells_per_day * days_collecting - shells_remaining = 2 := by
  sorry

end NUMINAMATH_CALUDE_shells_given_to_brother_l2639_263997


namespace NUMINAMATH_CALUDE_complex_number_equality_l2639_263921

theorem complex_number_equality : ∃ (z : ℂ), z = (2 * Complex.I) / (1 - Complex.I) ∧ z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2639_263921


namespace NUMINAMATH_CALUDE_stating_max_pairs_remaining_l2639_263943

/-- Represents the number of shoe types -/
def num_types : ℕ := 5

/-- Represents the number of shoe colors -/
def num_colors : ℕ := 5

/-- Represents the initial number of shoe pairs -/
def initial_pairs : ℕ := 25

/-- Represents the number of individual shoes lost -/
def shoes_lost : ℕ := 9

/-- 
Theorem stating that given the initial conditions, the maximum number of 
complete pairs remaining after losing shoes is 22
-/
theorem max_pairs_remaining : 
  ∀ (remaining_pairs : ℕ),
  remaining_pairs ≤ initial_pairs ∧
  remaining_pairs ≥ initial_pairs - shoes_lost / 2 →
  remaining_pairs ≤ 22 :=
by sorry

end NUMINAMATH_CALUDE_stating_max_pairs_remaining_l2639_263943


namespace NUMINAMATH_CALUDE_orange_count_l2639_263920

/-- Given the ratio of mangoes : oranges : apples and the number of mangoes and apples,
    calculate the number of oranges -/
theorem orange_count (mango_ratio orange_ratio apple_ratio mango_count apple_count : ℕ) :
  mango_ratio ≠ 0 →
  orange_ratio ≠ 0 →
  apple_ratio ≠ 0 →
  mango_ratio = 10 →
  orange_ratio = 2 →
  apple_ratio = 3 →
  mango_count = 120 →
  apple_count = 36 →
  mango_count / mango_ratio = apple_count / apple_ratio →
  (mango_count / mango_ratio) * orange_ratio = 24 := by
sorry

end NUMINAMATH_CALUDE_orange_count_l2639_263920


namespace NUMINAMATH_CALUDE_building_shadow_length_l2639_263904

/-- Given a flagpole and a building under similar shadow-casting conditions,
    calculate the length of the building's shadow. -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_height_pos : 0 < building_height)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building : building_height = 20) :
  (building_height * flagpole_shadow) / flagpole_height = 50 := by
  sorry


end NUMINAMATH_CALUDE_building_shadow_length_l2639_263904


namespace NUMINAMATH_CALUDE_function_triple_is_linear_l2639_263981

/-- A triple of injective functions from ℝ to ℝ satisfying specific conditions -/
structure FunctionTriple where
  f : ℝ → ℝ
  g : ℝ → ℝ
  h : ℝ → ℝ
  f_injective : Function.Injective f
  g_injective : Function.Injective g
  h_injective : Function.Injective h
  eq1 : ∀ x y, f (x + f y) = g x + h y
  eq2 : ∀ x y, g (x + g y) = h x + f y
  eq3 : ∀ x y, h (x + h y) = f x + g y

/-- The main theorem stating that any FunctionTriple consists of linear functions with the same constant term -/
theorem function_triple_is_linear (t : FunctionTriple) : 
  ∃ C : ℝ, ∀ x : ℝ, t.f x = x + C ∧ t.g x = x + C ∧ t.h x = x + C := by
  sorry


end NUMINAMATH_CALUDE_function_triple_is_linear_l2639_263981


namespace NUMINAMATH_CALUDE_sqrt_inequality_and_floor_l2639_263944

theorem sqrt_inequality_and_floor (n : ℕ) : 
  (Real.sqrt (n + 1) + 2 * Real.sqrt n < Real.sqrt (9 * n + 3)) ∧
  ¬∃ n : ℕ, ⌊Real.sqrt (n + 1) + 2 * Real.sqrt n⌋ < ⌊Real.sqrt (9 * n + 3)⌋ := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_and_floor_l2639_263944


namespace NUMINAMATH_CALUDE_will_summer_earnings_l2639_263908

/-- The amount of money Will spent on mower blades -/
def mower_blades_cost : ℕ := 41

/-- The number of games Will could buy with the remaining money -/
def number_of_games : ℕ := 7

/-- The cost of each game -/
def game_cost : ℕ := 9

/-- The total money Will made mowing lawns -/
def total_money : ℕ := mower_blades_cost + number_of_games * game_cost

theorem will_summer_earnings : total_money = 104 := by
  sorry

end NUMINAMATH_CALUDE_will_summer_earnings_l2639_263908


namespace NUMINAMATH_CALUDE_power_mod_eleven_l2639_263910

theorem power_mod_eleven : (5 : ℤ) ^ 1233 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l2639_263910


namespace NUMINAMATH_CALUDE_item_price_ratio_l2639_263989

theorem item_price_ratio (x y c : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_item_price_ratio_l2639_263989


namespace NUMINAMATH_CALUDE_shaded_to_white_ratio_is_five_thirds_l2639_263992

/-- A square subdivided into smaller squares where vertices of inner squares 
    are at midpoints of sides of the next larger square -/
structure SubdividedSquare :=
  (side : ℝ)
  (is_positive : side > 0)

/-- The ratio of shaded to white area in a subdivided square -/
def shaded_to_white_ratio (s : SubdividedSquare) : ℚ :=
  5 / 3

/-- Theorem stating that the ratio of shaded to white area is 5:3 -/
theorem shaded_to_white_ratio_is_five_thirds (s : SubdividedSquare) :
  shaded_to_white_ratio s = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_white_ratio_is_five_thirds_l2639_263992


namespace NUMINAMATH_CALUDE_fraction_comparison_l2639_263900

theorem fraction_comparison : (1 : ℚ) / 4 = 24999999 / (10^8 : ℚ) + 1 / (4 * 10^8 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2639_263900


namespace NUMINAMATH_CALUDE_sinusoid_amplitude_l2639_263985

theorem sinusoid_amplitude (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_sinusoid_amplitude_l2639_263985


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2639_263994

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 3) / Real.log 2}
def B : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > 0} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2639_263994


namespace NUMINAMATH_CALUDE_population_growth_determinants_l2639_263924

-- Define the factors that can potentially influence population growth
structure PopulationFactors where
  birthRate : ℝ
  deathRate : ℝ
  totalPopulation : ℝ
  socialProductionRate : ℝ
  naturalGrowthRate : ℝ

-- Define population growth pattern as a function of factors
def populationGrowthPattern (factors : PopulationFactors) : ℝ := sorry

-- Theorem stating that population growth pattern is determined by birth rate, death rate, and natural growth rate
theorem population_growth_determinants (factors : PopulationFactors) :
  populationGrowthPattern factors =
    populationGrowthPattern ⟨factors.birthRate, factors.deathRate, 0, 0, factors.naturalGrowthRate⟩ :=
by sorry

end NUMINAMATH_CALUDE_population_growth_determinants_l2639_263924


namespace NUMINAMATH_CALUDE_g_range_l2639_263923

noncomputable def g (x : ℝ) : ℝ := 3 * (x - 4)

theorem g_range :
  {y : ℝ | ∃ x : ℝ, x ≠ -5 ∧ g x = y} = {y : ℝ | y ≠ -27} := by
  sorry

end NUMINAMATH_CALUDE_g_range_l2639_263923


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l2639_263949

theorem cylinder_volume_ratio : 
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 10
  let cylinder_a_height : ℝ := rectangle_height
  let cylinder_a_circumference : ℝ := rectangle_width
  let cylinder_b_height : ℝ := rectangle_width
  let cylinder_b_circumference : ℝ := rectangle_height
  let cylinder_volume (h : ℝ) (c : ℝ) : ℝ := h * (c / (2 * π))^2 * π
  let volume_a := cylinder_volume cylinder_a_height cylinder_a_circumference
  let volume_b := cylinder_volume cylinder_b_height cylinder_b_circumference
  max volume_a volume_b / min volume_a volume_b = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l2639_263949


namespace NUMINAMATH_CALUDE_intersection_complement_sets_l2639_263936

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_complement_sets : M ∩ (U \ N) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_sets_l2639_263936


namespace NUMINAMATH_CALUDE_inequality_interval_length_l2639_263915

theorem inequality_interval_length (c d : ℝ) : 
  (∃ (x : ℝ), c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d) ∧ 
  ((d - 5) / 3 - (c - 5) / 3 = 12) → 
  d - c = 36 := by
sorry

end NUMINAMATH_CALUDE_inequality_interval_length_l2639_263915


namespace NUMINAMATH_CALUDE_quadratic_expression_a_range_l2639_263967

-- Define the quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for the inequality solution
def inequality_solution (a b c : ℝ) : Prop :=
  ∀ x, quadratic_function a b c x > -2 * x ↔ 1 < x ∧ x < 3

-- Theorem 1
theorem quadratic_expression
  (a b c : ℝ)
  (h1 : inequality_solution a b c)
  (h2 : ∃ x, quadratic_function a b c x + 6 * a = 0 ∧
              ∀ y, quadratic_function a b c y + 6 * a = 0 → y = x) :
  ∃ x, quadratic_function (-1/5) (-6/5) (-3/5) x = quadratic_function a b c x :=
sorry

-- Theorem 2
theorem a_range
  (a b c : ℝ)
  (h1 : inequality_solution a b c)
  (h2 : ∃ m, ∀ x, quadratic_function a b c x ≤ m ∧ m > 0) :
  a > -2 + Real.sqrt 3 ∨ a < -2 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_expression_a_range_l2639_263967


namespace NUMINAMATH_CALUDE_base_conversion_l2639_263931

/-- Given a base r where 175 in base r equals 125 in base 10, 
    prove that 76 in base r equals 62 in base 10 -/
theorem base_conversion (r : ℕ) (hr : r > 1) : 
  (1 * r^2 + 7 * r + 5 = 125) → (7 * r + 6 = 62) :=
by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l2639_263931


namespace NUMINAMATH_CALUDE_workers_total_earning_l2639_263946

/-- Represents the daily wages and work days of three workers -/
structure Workers where
  a_days : ℕ
  b_days : ℕ
  c_days : ℕ
  c_wage : ℕ
  wage_ratio : Fin 3 → ℕ

/-- Calculates the total earnings of the workers -/
def total_earning (w : Workers) : ℕ :=
  let unit := w.c_wage / w.wage_ratio 2
  let a_wage := unit * w.wage_ratio 0
  let b_wage := unit * w.wage_ratio 1
  a_wage * w.a_days + b_wage * w.b_days + w.c_wage * w.c_days

/-- The main theorem stating the total earning of the workers -/
theorem workers_total_earning : ∃ (w : Workers), 
  w.a_days = 6 ∧ 
  w.b_days = 9 ∧ 
  w.c_days = 4 ∧ 
  w.c_wage = 105 ∧ 
  w.wage_ratio = ![3, 4, 5] ∧
  total_earning w = 1554 := by
  sorry

end NUMINAMATH_CALUDE_workers_total_earning_l2639_263946


namespace NUMINAMATH_CALUDE_cake_sharing_l2639_263930

theorem cake_sharing (n : ℕ) : 
  (∃ (shares : Fin n → ℚ), 
    (∀ i, 0 < shares i) ∧ 
    (∃ j, shares j = 1/11) ∧
    (∃ k, shares k = 1/14) ∧
    (∀ i, 1/14 ≤ shares i ∧ shares i ≤ 1/11) ∧
    (Finset.sum Finset.univ shares = 1)) ↔ 
  (n = 12 ∨ n = 13) :=
by sorry


end NUMINAMATH_CALUDE_cake_sharing_l2639_263930


namespace NUMINAMATH_CALUDE_determinant_zero_l2639_263979

def matrix1 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 2, 3],
    ![4, 5, 6],
    ![7, 8, 9]]

def matrix2 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 4, 9],
    ![16, 25, 36],
    ![49, 64, 81]]

theorem determinant_zero (h : Matrix.det matrix1 = 0) :
  Matrix.det matrix2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_zero_l2639_263979


namespace NUMINAMATH_CALUDE_rats_meet_on_day_10_l2639_263971

/-- The thickness of the wall in feet -/
def wall_thickness : ℕ := 1000

/-- The initial drilling speed of both rats in feet per day -/
def initial_speed : ℕ := 1

/-- The function representing the total distance drilled by both rats after n days -/
def total_distance (n : ℕ) : ℚ :=
  (2^n - 1) + 2 * (1 - (1/2)^n)

/-- The theorem stating that the rats meet on the 10th day -/
theorem rats_meet_on_day_10 :
  total_distance 9 < wall_thickness ∧ total_distance 10 ≥ wall_thickness :=
sorry

end NUMINAMATH_CALUDE_rats_meet_on_day_10_l2639_263971


namespace NUMINAMATH_CALUDE_abs_value_of_z_l2639_263934

/-- Given a complex number z = ((1+i)/(1-i))^2, prove that its absolute value |z| is equal to 1. -/
theorem abs_value_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.abs ((((1:ℂ) + i) / ((1:ℂ) - i))^2) = 1 := by sorry

end NUMINAMATH_CALUDE_abs_value_of_z_l2639_263934


namespace NUMINAMATH_CALUDE_function_highest_points_omega_range_l2639_263903

/-- Given a function f(x) = 2sin(ωx + π/4) with ω > 0, if the graph of f(x) has exactly 3 highest points
    in the interval [0,1], then ω is in the range [17π/4, 25π/4). -/
theorem function_highest_points_omega_range (ω : ℝ) (h1 : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x + π / 4)
  (∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x ∈ s, x ∈ Set.Icc 0 1) ∧
    (∀ y ∈ Set.Icc 0 1, ∃ x ∈ s, f y ≤ f x) ∧
    (∀ z ∉ s, z ∈ Set.Icc 0 1 → ∃ x ∈ s, f z < f x)) →
  17 * π / 4 ≤ ω ∧ ω < 25 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_function_highest_points_omega_range_l2639_263903


namespace NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l2639_263953

/-- Represents the number of fans for each team -/
structure FanCount where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of Mets to Red Sox fans -/
theorem mets_to_red_sox_ratio (fans : FanCount) 
  (total_fans : fans.yankees + fans.mets + fans.red_sox = 330)
  (yankees_to_mets : Ratio)
  (yankees_mets_ratio : yankees_to_mets.numerator * fans.mets = yankees_to_mets.denominator * fans.yankees)
  (yankees_mets_values : yankees_to_mets.numerator = 3 ∧ yankees_to_mets.denominator = 2)
  (mets_count : fans.mets = 88) :
  ∃ (r : Ratio), r.numerator = 4 ∧ r.denominator = 5 ∧ 
    r.numerator * fans.red_sox = r.denominator * fans.mets :=
sorry

end NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l2639_263953


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2639_263962

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    if S_m = 2 and S_2m = 10, then S_3m = 24. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) :
  (∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) →  -- Definition of S_n for arithmetic sequence
  (S m = 2) →
  (S (2 * m) = 10) →
  (S (3 * m) = 24) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2639_263962


namespace NUMINAMATH_CALUDE_cos_225_degrees_l2639_263995

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l2639_263995


namespace NUMINAMATH_CALUDE_projection_vector_proof_l2639_263993

def line_direction : ℝ × ℝ := (3, 2)

theorem projection_vector_proof :
  ∃ (w : ℝ × ℝ), 
    w.1 + w.2 = 3 ∧ 
    w.1 * line_direction.1 + w.2 * line_direction.2 = 0 ∧
    w = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_projection_vector_proof_l2639_263993


namespace NUMINAMATH_CALUDE_f_36_l2639_263928

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, f (x * y) = f x + f y)
variable (h2 : f 2 = p)
variable (h3 : f 3 = q)

-- State the theorem
theorem f_36 (p q : ℝ) : f 36 = 2 * (p + q) := by sorry

end NUMINAMATH_CALUDE_f_36_l2639_263928


namespace NUMINAMATH_CALUDE_max_triangles_three_families_l2639_263976

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : ℕ)

/-- Calculates the maximum number of triangles formed by three families of parallel lines -/
def max_triangles (f1 f2 f3 : LineFamily) : ℕ :=
  150

/-- Theorem stating that three families of 10 parallel lines form at most 150 triangles -/
theorem max_triangles_three_families :
  ∀ (f1 f2 f3 : LineFamily),
    f1.count = 10 → f2.count = 10 → f3.count = 10 →
    max_triangles f1 f2 f3 = 150 :=
by
  sorry

#check max_triangles_three_families

end NUMINAMATH_CALUDE_max_triangles_three_families_l2639_263976


namespace NUMINAMATH_CALUDE_inequality_proof_l2639_263984

theorem inequality_proof (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  abs a + abs b + abs c ≥ 3 * Real.sqrt 3 * (b^2 * c^2 + c^2 * a^2 + a^2 * b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2639_263984


namespace NUMINAMATH_CALUDE_binomial_15_choose_3_l2639_263983

theorem binomial_15_choose_3 : Nat.choose 15 3 = 455 := by sorry

end NUMINAMATH_CALUDE_binomial_15_choose_3_l2639_263983


namespace NUMINAMATH_CALUDE_tangent_to_ln_curve_l2639_263942

theorem tangent_to_ln_curve (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ a * x = Real.log x ∧ (∀ y : ℝ, y > 0 → a * y ≥ Real.log y)) → 
  a = 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_to_ln_curve_l2639_263942


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2639_263964

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 - 2*x}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x + 1)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1 : ℝ) (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2639_263964


namespace NUMINAMATH_CALUDE_triangle_inequality_l2639_263933

theorem triangle_inequality (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_a_geq_b : a ≥ b)
  (h_a_geq_c : a ≥ c)
  (h_sum1 : a + b - c > 0)
  (h_sum2 : b + c - a > 0) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2639_263933


namespace NUMINAMATH_CALUDE_circle_equation_holds_l2639_263961

/-- A circle in the Cartesian plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in general form --/
def CircleEquation (x y : ℝ) := x^2 + y^2 - 6*x = 0

/-- The circle represented by the equation x^2 + y^2 - 6x = 0 --/
def specificCircle : Circle := { center := (3, 0), radius := 3 }

/-- Theorem stating that the specificCircle satisfies the given equation --/
theorem circle_equation_holds (x y : ℝ) :
  CircleEquation x y ↔ (x - specificCircle.center.1)^2 + (y - specificCircle.center.2)^2 = specificCircle.radius^2 := by
  sorry

#check circle_equation_holds

end NUMINAMATH_CALUDE_circle_equation_holds_l2639_263961


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2639_263901

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 4) : 150 * (1 / x) = 300 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2639_263901


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l2639_263998

theorem complex_equation_solutions :
  {z : ℂ | z^6 - 6*z^4 + 9*z^2 = 0} = {0, Complex.I * Real.sqrt 3, -Complex.I * Real.sqrt 3} := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l2639_263998


namespace NUMINAMATH_CALUDE_max_gcd_sum_1025_l2639_263929

theorem max_gcd_sum_1025 : 
  ∃ (max : ℕ), max > 0 ∧ 
  (∀ a b : ℕ, a > 0 → b > 0 → a + b = 1025 → Nat.gcd a b ≤ max) ∧
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a + b = 1025 ∧ Nat.gcd a b = max) ∧
  max = 205 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1025_l2639_263929


namespace NUMINAMATH_CALUDE_patrick_has_25_dollars_l2639_263935

/-- Calculates the amount of money Patrick has after saving for a bicycle and lending money to a friend. -/
def patricks_money (bicycle_price : ℕ) (amount_lent : ℕ) : ℕ :=
  bicycle_price / 2 - amount_lent

/-- Proves that Patrick has $25 after saving for a $150 bicycle and lending $50 to a friend. -/
theorem patrick_has_25_dollars :
  patricks_money 150 50 = 25 := by
  sorry

end NUMINAMATH_CALUDE_patrick_has_25_dollars_l2639_263935


namespace NUMINAMATH_CALUDE_min_planks_for_color_condition_l2639_263958

/-- Represents a fence with colored planks. -/
structure Fence where
  n : ℕ                            -- number of planks
  colors : Fin n → Fin 100         -- color of each plank

/-- Checks if the fence satisfies the color condition. -/
def satisfiesColorCondition (f : Fence) : Prop :=
  ∀ (i j : Fin 100), i ≠ j →
    ∃ (p q : Fin f.n), p < q ∧ f.colors p = i ∧ f.colors q = j

/-- The theorem stating the minimum number of planks required. -/
theorem min_planks_for_color_condition :
  (∃ (f : Fence), satisfiesColorCondition f) →
  (∀ (f : Fence), satisfiesColorCondition f → f.n ≥ 199) ∧
  (∃ (f : Fence), f.n = 199 ∧ satisfiesColorCondition f) :=
sorry

end NUMINAMATH_CALUDE_min_planks_for_color_condition_l2639_263958


namespace NUMINAMATH_CALUDE_jason_car_count_l2639_263902

/-- The number of red cars counted by Jason -/
def red_cars : ℕ := sorry

/-- The number of green cars counted by Jason -/
def green_cars : ℕ := sorry

/-- The number of purple cars counted by Jason -/
def purple_cars : ℕ := 47

theorem jason_car_count :
  (green_cars = 4 * red_cars) ∧
  (red_cars > purple_cars) ∧
  (green_cars + red_cars + purple_cars = 312) ∧
  (red_cars - purple_cars = 6) :=
by sorry

end NUMINAMATH_CALUDE_jason_car_count_l2639_263902


namespace NUMINAMATH_CALUDE_problem_solution_l2639_263969

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (x : ℝ) : ℝ := f (x + 1) - x

theorem problem_solution :
  (∃ (x_max : ℝ), ∀ (x : ℝ), g x ≤ g x_max ∧ g x_max = 0) ∧
  (∀ (n : ℕ), n > 0 → (1 + 1 / n : ℝ) ^ n < Real.exp 1) ∧
  (∀ (a b : ℝ), 0 < a → a < b → f b - f a > 2 * a * (b - a) / (a^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2639_263969


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l2639_263999

-- Define a function to convert a number from base b to base 10
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

-- Define the number in base 7
def num_base_7 : List Nat := [1, 4, 3, 2, 5]

-- Define the number in base 8
def num_base_8 : List Nat := [1, 2, 3, 4]

-- Theorem statement
theorem base_conversion_subtraction :
  to_base_10 num_base_7 7 - to_base_10 num_base_8 8 = 10610 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l2639_263999


namespace NUMINAMATH_CALUDE_rectangle_arrangement_l2639_263932

/-- Given 110 identical rectangular sheets where each sheet's length is 10 cm longer than its width,
    and when arranged as in Figure 1 they form a rectangle of length 2750 cm,
    prove that the length of the rectangle formed when arranged as in Figure 2 is 1650 cm. -/
theorem rectangle_arrangement (n : ℕ) (sheet_length sheet_width : ℝ) 
  (h1 : n = 110)
  (h2 : sheet_length = sheet_width + 10)
  (h3 : n * sheet_length = 2750) :
  n * sheet_width = 1650 := by
  sorry

#check rectangle_arrangement

end NUMINAMATH_CALUDE_rectangle_arrangement_l2639_263932


namespace NUMINAMATH_CALUDE_sequence_general_term_l2639_263916

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 1 = 2 →
  (∀ n, a (n + 1)^2 = (a n)^2 + 2) →
  ∀ n, a n = Real.sqrt (2 * n + 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2639_263916


namespace NUMINAMATH_CALUDE_toy_count_is_134_l2639_263991

/-- The number of toys initially in the box, given the conditions of the problem -/
def initial_toy_count : ℕ := by sorry

theorem toy_count_is_134 :
  -- Define variables for red and white toys
  ∀ (red white : ℕ),
  -- After removing 2 red toys, red is twice white
  (red - 2 = 2 * white) →
  -- After removing 2 red toys, there are 88 red toys
  (red - 2 = 88) →
  -- The initial toy count is the sum of red and white toys
  initial_toy_count = red + white →
  -- Prove that the initial toy count is 134
  initial_toy_count = 134 := by sorry

end NUMINAMATH_CALUDE_toy_count_is_134_l2639_263991


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2639_263917

def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioo (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2639_263917


namespace NUMINAMATH_CALUDE_y_divisibility_l2639_263938

def y : ℕ := 64 + 96 + 192 + 256 + 352 + 480 + 4096 + 8192

theorem y_divisibility : 
  (∃ k : ℕ, y = 32 * k) ∧ ¬(∃ m : ℕ, y = 64 * m) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l2639_263938


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2639_263957

theorem inequality_solution_set (x : ℝ) : 
  2 / (x + 2) + 5 / (x + 4) ≥ 3 / 2 ↔ x ∈ Set.Icc (-4 : ℝ) (2/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2639_263957


namespace NUMINAMATH_CALUDE_jim_caught_two_fish_l2639_263975

def fish_problem (ben judy billy susie jim : ℕ) : Prop :=
  ben = 4 ∧
  judy = 1 ∧
  billy = 3 ∧
  susie = 5 ∧
  ∃ (thrown_back : ℕ), thrown_back = 3 ∧
  ∃ (total_filets : ℕ), total_filets = 24 ∧
  (ben + judy + billy + susie + jim - thrown_back) * 2 = total_filets

theorem jim_caught_two_fish :
  ∀ ben judy billy susie jim : ℕ,
  fish_problem ben judy billy susie jim →
  jim = 2 :=
by sorry

end NUMINAMATH_CALUDE_jim_caught_two_fish_l2639_263975


namespace NUMINAMATH_CALUDE_function_ordering_l2639_263987

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_ordering (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_ordering_l2639_263987


namespace NUMINAMATH_CALUDE_area_of_region_R_l2639_263972

/-- Represents a rhombus ABCD -/
structure Rhombus where
  sideLength : ℝ
  angleB : ℝ

/-- Represents the region R inside the rhombus -/
def RegionR (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of region R in the rhombus -/
noncomputable def areaR (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of region R in the specific rhombus -/
theorem area_of_region_R : 
  let r : Rhombus := { sideLength := 3, angleB := 150 * π / 180 }
  ∃ ε > 0, |areaR r - 0.873| < ε :=
sorry

end NUMINAMATH_CALUDE_area_of_region_R_l2639_263972


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l2639_263911

/-- A positive geometric sequence with common ratio 2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n ∧ a n > 0

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  GeometricSequence a →
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →
  (∀ k l : ℕ, Real.sqrt (a k * a l) = 4 * a 1 → 1 / k + 4 / l ≥ 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l2639_263911


namespace NUMINAMATH_CALUDE_root_value_l2639_263925

theorem root_value (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 2*x + a = 0 ↔ x = x₁ ∨ x = x₂) →  -- x₁ and x₂ are roots of x^2 - 2x + a = 0
  (x₁ + 2*x₂ = 3 - Real.sqrt 2) →               -- given condition
  x₂ = 1 - Real.sqrt 2 :=                       -- conclusion to prove
by
  sorry  -- Proof omitted as per instructions


end NUMINAMATH_CALUDE_root_value_l2639_263925


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2639_263945

theorem imaginary_part_of_z : Complex.im ((1 + 2 * Complex.I) / (3 - Complex.I)) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2639_263945


namespace NUMINAMATH_CALUDE_base_eight_47_equals_39_l2639_263968

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (tens : Nat) (ones : Nat) : Nat :=
  tens * 8 + ones

/-- The base-eight number 47 is equal to the base-ten number 39 -/
theorem base_eight_47_equals_39 : base_eight_to_ten 4 7 = 39 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_47_equals_39_l2639_263968


namespace NUMINAMATH_CALUDE_h_is_even_l2639_263909

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define the function h
def h (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ g x * |f x|

-- State the theorem
theorem h_is_even (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) : 
  IsEven (h f g) := by
  sorry

end NUMINAMATH_CALUDE_h_is_even_l2639_263909


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l2639_263980

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l2639_263980


namespace NUMINAMATH_CALUDE_problem_solution_l2639_263906

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem problem_solution (m n : ℝ) 
  (h1 : 0 < m) (h2 : m < n) 
  (h3 : f m = f n) 
  (h4 : ∀ x ∈ Set.Icc (m^2) n, f x ≤ 2) 
  (h5 : ∃ x ∈ Set.Icc (m^2) n, f x = 2) : 
  n / m = Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2639_263906


namespace NUMINAMATH_CALUDE_b_is_eighteen_l2639_263939

/-- Represents the ages of three people a, b, and c. -/
structure Ages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.a = ages.b + 2 ∧
  ages.b = 2 * ages.c ∧
  ages.a + ages.b + ages.c = 47

/-- The theorem statement -/
theorem b_is_eighteen (ages : Ages) (h : satisfiesConditions ages) : ages.b = 18 := by
  sorry

end NUMINAMATH_CALUDE_b_is_eighteen_l2639_263939


namespace NUMINAMATH_CALUDE_bread_price_is_four_l2639_263951

/-- The price of a loaf of bread -/
def bread_price : ℝ := 4

/-- The price of a pastry -/
def pastry_price : ℝ := 2

/-- The usual number of pastries sold daily -/
def usual_pastries : ℕ := 20

/-- The usual number of loaves of bread sold daily -/
def usual_bread : ℕ := 10

/-- The number of pastries sold today -/
def today_pastries : ℕ := 14

/-- The number of loaves of bread sold today -/
def today_bread : ℕ := 25

/-- The difference between today's sales and the usual daily average -/
def sales_difference : ℝ := 48

theorem bread_price_is_four :
  (today_pastries : ℝ) * pastry_price + today_bread * bread_price -
  (usual_pastries : ℝ) * pastry_price - usual_bread * bread_price = sales_difference ∧
  bread_price = 4 := by sorry

end NUMINAMATH_CALUDE_bread_price_is_four_l2639_263951


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2639_263922

/-- A geometric sequence with first term 1 and sum of first 3 terms equal to 3/4 has common ratio -1/2 -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
    a 1 = 1 →                     -- first term is 1
    a 1 + a 2 + a 3 = 3/4 →       -- sum of first 3 terms is 3/4
    q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2639_263922


namespace NUMINAMATH_CALUDE_min_value_theorem_l2639_263966

theorem min_value_theorem (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ 
  ((n : ℝ) / 3 + 27 / (n : ℝ) = 6 ↔ n = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2639_263966


namespace NUMINAMATH_CALUDE_male_to_total_ratio_l2639_263978

/-- Represents the population of alligators on Lagoon island -/
structure AlligatorPopulation where
  maleCount : ℕ
  adultFemaleCount : ℕ
  juvenileFemaleRatio : ℚ

/-- The ratio of male alligators to total alligators is 1:2 -/
theorem male_to_total_ratio (pop : AlligatorPopulation)
    (h1 : pop.maleCount = 25)
    (h2 : pop.adultFemaleCount = 15)
    (h3 : pop.juvenileFemaleRatio = 2/5) :
    pop.maleCount / (pop.maleCount + pop.adultFemaleCount / (1 - pop.juvenileFemaleRatio)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_male_to_total_ratio_l2639_263978


namespace NUMINAMATH_CALUDE_maximize_product_l2639_263940

theorem maximize_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 60) :
  x^3 * y^2 * z^4 ≤ 20^3 * (40/3)^2 * (80/3)^4 ∧
  (x^3 * y^2 * z^4 = 20^3 * (40/3)^2 * (80/3)^4 ↔ x = 20 ∧ y = 40/3 ∧ z = 80/3) :=
by sorry

end NUMINAMATH_CALUDE_maximize_product_l2639_263940


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l2639_263918

theorem cylinder_volume_increase : ∀ (r h : ℝ),
  r > 0 → h > 0 →
  let new_r := r * 2.5
  let new_h := h * 3
  (π * new_r^2 * new_h) / (π * r^2 * h) = 18.75 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l2639_263918


namespace NUMINAMATH_CALUDE_two_hundred_twenty_fifth_number_with_digit_sum_2018_l2639_263919

def digit_sum (n : ℕ) : ℕ := sorry

def nth_number_with_digit_sum (n : ℕ) (sum : ℕ) : ℕ := sorry

theorem two_hundred_twenty_fifth_number_with_digit_sum_2018 :
  nth_number_with_digit_sum 225 2018 = 39 * 10^224 + (10^224 - 10) * 9 + 8 :=
sorry

end NUMINAMATH_CALUDE_two_hundred_twenty_fifth_number_with_digit_sum_2018_l2639_263919


namespace NUMINAMATH_CALUDE_equal_rectangle_count_l2639_263965

def count_rectangles (perimeter : ℕ) : ℕ := 
  (perimeter / 2 - 1) / 2

theorem equal_rectangle_count : 
  count_rectangles 1996 = count_rectangles 1998 :=
by sorry

end NUMINAMATH_CALUDE_equal_rectangle_count_l2639_263965


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l2639_263956

/-- The number of games played in a single-elimination tournament. -/
def games_played (num_teams : ℕ) : ℕ :=
  num_teams - 1

/-- Theorem: In a single-elimination tournament with 32 teams, 31 games are played. -/
theorem single_elimination_tournament_games :
  games_played 32 = 31 := by
  sorry

#eval games_played 32  -- Should output 31

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l2639_263956


namespace NUMINAMATH_CALUDE_gold_bars_problem_l2639_263996

theorem gold_bars_problem (initial : ℕ) : 
  (initial : ℚ) * (1 - 0.1) * 0.5 = 27 → initial = 60 := by
  sorry

end NUMINAMATH_CALUDE_gold_bars_problem_l2639_263996


namespace NUMINAMATH_CALUDE_fifth_term_is_sixteen_l2639_263947

/-- A geometric sequence with first term 1 and a_2 * a_4 = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧
  (∃ q : ℝ, ∀ n : ℕ, a n = q ^ (n - 1)) ∧
  a 2 * a 4 = 16

theorem fifth_term_is_sixteen 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  a 5 = 16 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_is_sixteen_l2639_263947


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2639_263970

theorem min_distance_to_line (x y : ℝ) :
  (3 * x + y = 10) → (x^2 + y^2 ≥ 10) := by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2639_263970


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2639_263990

def f (a x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2639_263990


namespace NUMINAMATH_CALUDE_complement_of_A_l2639_263960

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x | x ∈ U ∧ x^2 + x - 2 < 0}

theorem complement_of_A : (U \ A) = {-2, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2639_263960


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l2639_263954

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 - 18 * x + 27

-- State the theorem
theorem f_extrema_on_interval :
  (∀ x ∈ Set.Icc 0 3, f x ≤ 54) ∧
  (∃ x ∈ Set.Icc 0 3, f x = 54) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≥ 27/4) ∧
  (∃ x ∈ Set.Icc 0 3, f x = 27/4) :=
sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l2639_263954


namespace NUMINAMATH_CALUDE_f_increasing_and_odd_l2639_263974

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - 5^(-x) else 5^x - 1

theorem f_increasing_and_odd :
  (∀ x y, x < y → f x < f y) ∧
  (∀ x, f (-x) = -f x) :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_and_odd_l2639_263974


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2639_263982

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 45 ∧ 
  a = 2 * (b + c) ∧ 
  c = 4 * b → 
  a * b * c = 1080 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2639_263982


namespace NUMINAMATH_CALUDE_power_sum_sequence_l2639_263973

theorem power_sum_sequence (a b x y : ℝ) 
  (eq1 : a*x + b*y = 3)
  (eq2 : a*x^2 + b*y^2 = 7)
  (eq3 : a*x^3 + b*y^3 = 16)
  (eq4 : a*x^4 + b*y^4 = 42) :
  a*x^5 + b*y^5 = 20 := by
sorry

end NUMINAMATH_CALUDE_power_sum_sequence_l2639_263973


namespace NUMINAMATH_CALUDE_correct_calculation_l2639_263926

theorem correct_calculation : 
  (2 * Real.sqrt 5 + 3 * Real.sqrt 5 = 5 * Real.sqrt 5) ∧ 
  (Real.sqrt 8 ≠ 2) ∧ 
  (Real.sqrt ((-3)^2) ≠ -3) ∧ 
  ((Real.sqrt 2 + 1)^2 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l2639_263926


namespace NUMINAMATH_CALUDE_sum_of_three_integers_with_product_125_l2639_263941

theorem sum_of_three_integers_with_product_125 :
  ∃ (a b c : ℕ+), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a * b * c : ℕ) = 125 ∧
    (a + b + c : ℕ) = 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_integers_with_product_125_l2639_263941


namespace NUMINAMATH_CALUDE_terry_age_proof_l2639_263937

/-- Nora's current age -/
def nora_age : ℕ := 10

/-- Terry's age in 10 years -/
def terry_future_age : ℕ := 4 * nora_age

/-- Terry's current age -/
def terry_current_age : ℕ := terry_future_age - 10

theorem terry_age_proof : terry_current_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_terry_age_proof_l2639_263937


namespace NUMINAMATH_CALUDE_quadruple_prime_equation_l2639_263913

theorem quadruple_prime_equation :
  ∀ p q r : ℕ, ∀ n : ℕ+,
    Prime p ∧ Prime q ∧ Prime r →
    p^2 = q^2 + r^(n : ℕ) →
    (p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadruple_prime_equation_l2639_263913


namespace NUMINAMATH_CALUDE_additional_sleep_january_l2639_263977

def sleep_december : ℝ := 6.5
def sleep_january : ℝ := 8.5
def days_in_month : ℕ := 31

theorem additional_sleep_january : 
  (sleep_january - sleep_december) * days_in_month = 62 := by
  sorry

end NUMINAMATH_CALUDE_additional_sleep_january_l2639_263977


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2639_263988

/-- The decimal representation of 0.6̄03 as a rational number -/
def repeating_decimal : ℚ := 0.6 + (3 : ℚ) / 100 / (1 - 1/100)

/-- Theorem stating that 0.6̄03 is equal to 104/165 -/
theorem repeating_decimal_as_fraction : repeating_decimal = 104 / 165 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2639_263988


namespace NUMINAMATH_CALUDE_disjunction_truth_implication_false_l2639_263955

theorem disjunction_truth_implication_false : ¬ (∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_truth_implication_false_l2639_263955


namespace NUMINAMATH_CALUDE_oldest_turner_child_age_l2639_263948

theorem oldest_turner_child_age 
  (num_children : ℕ) 
  (average_age : ℕ) 
  (younger_children_ages : List ℕ) :
  num_children = 4 →
  average_age = 9 →
  younger_children_ages = [6, 8, 11] →
  (List.sum younger_children_ages + 11) / num_children = average_age :=
by sorry

end NUMINAMATH_CALUDE_oldest_turner_child_age_l2639_263948


namespace NUMINAMATH_CALUDE_division_result_l2639_263950

theorem division_result : ∃ (result : ℚ), 
  (40 / 2 = result) ∧ 
  (40 + result + 2 = 62) ∧ 
  (result = 20) := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2639_263950


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l2639_263952

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 500! has 124 trailing zeroes -/
theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l2639_263952


namespace NUMINAMATH_CALUDE_sticker_distribution_l2639_263914

/-- The number of ways to distribute n identical objects into k distinct containers --/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

theorem sticker_distribution : distribute 10 4 = 251 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2639_263914


namespace NUMINAMATH_CALUDE_polyhedron_property_l2639_263986

/-- A convex polyhedron with the given properties -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  t : ℕ  -- number of triangular faces
  h : ℕ  -- number of hexagonal faces
  T : ℕ  -- number of triangular faces meeting at each vertex
  H : ℕ  -- number of hexagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 40
  face_composition : F = t + h
  vertex_property : 2 * T + H = 7
  edge_count : E = (3 * t + 6 * h) / 2

theorem polyhedron_property (P : ConvexPolyhedron) : 100 * P.H + 10 * P.T + P.V = 367 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_property_l2639_263986
