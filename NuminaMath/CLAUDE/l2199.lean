import Mathlib

namespace NUMINAMATH_CALUDE_second_number_is_72_l2199_219968

theorem second_number_is_72 (a b c : ℚ) : 
  a + b + c = 264 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a → 
  b = 72 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_72_l2199_219968


namespace NUMINAMATH_CALUDE_parabola_equation_l2199_219966

/-- A parabola is defined by its vertex and a point it passes through. -/
structure Parabola where
  vertex : ℝ × ℝ
  point : ℝ × ℝ

/-- The analytical expression of a parabola. -/
def parabola_expression (p : Parabola) : ℝ → ℝ :=
  fun x => -(x + 2)^2 + 3

theorem parabola_equation (p : Parabola) 
  (h1 : p.vertex = (-2, 3)) 
  (h2 : p.point = (1, -6)) : 
  ∀ x, parabola_expression p x = -(x + 2)^2 + 3 := by
  sorry

#check parabola_equation

end NUMINAMATH_CALUDE_parabola_equation_l2199_219966


namespace NUMINAMATH_CALUDE_pyramid_volume_l2199_219905

theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) :
  base_length = 5 →
  base_width = 10 →
  edge_length = 15 →
  let base_area := base_length * base_width
  let diagonal := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal / 2)^2)
  let volume := (1 / 3) * base_area * height
  volume = 232 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2199_219905


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2199_219982

theorem min_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  (1 / m + 1 / n) ≥ 2 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ m₀ + n₀ = 2 ∧ 1 / m₀ + 1 / n₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2199_219982


namespace NUMINAMATH_CALUDE_tangent_point_abscissa_l2199_219996

noncomputable section

-- Define the function f(x) = x^2 + x - ln x
def f (x : ℝ) : ℝ := x^2 + x - Real.log x

-- Define the derivative of f(x)
def f_deriv (x : ℝ) : ℝ := 2*x + 1 - 1/x

-- Theorem statement
theorem tangent_point_abscissa (t : ℝ) (h : t > 0) :
  (f t / t = f_deriv t) → t = 1 :=
sorry


end NUMINAMATH_CALUDE_tangent_point_abscissa_l2199_219996


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2199_219948

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

/-- Prove that the opposite of -2 is 2. -/
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2199_219948


namespace NUMINAMATH_CALUDE_equation_one_solutions_l2199_219969

theorem equation_one_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l2199_219969


namespace NUMINAMATH_CALUDE_distance_between_points_l2199_219945

def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (13, 4)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 170 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2199_219945


namespace NUMINAMATH_CALUDE_tv_screen_length_tv_screen_length_approx_l2199_219975

theorem tv_screen_length (diagonal : ℝ) (ratio_length_height : ℚ) : ℝ :=
  let length := Real.sqrt ((ratio_length_height ^ 2 * diagonal ^ 2) / (1 + ratio_length_height ^ 2))
  length

theorem tv_screen_length_approx :
  ∃ ε > 0, abs (tv_screen_length 27 (4/3) - 21.6) < ε :=
sorry

end NUMINAMATH_CALUDE_tv_screen_length_tv_screen_length_approx_l2199_219975


namespace NUMINAMATH_CALUDE_total_matches_proof_l2199_219984

def grade1_classes : ℕ := 5
def grade2_classes : ℕ := 7
def grade3_classes : ℕ := 4

def matches_in_tournament (n : ℕ) : ℕ := n * (n - 1) / 2

theorem total_matches_proof :
  matches_in_tournament grade1_classes +
  matches_in_tournament grade2_classes +
  matches_in_tournament grade3_classes = 37 := by
sorry

end NUMINAMATH_CALUDE_total_matches_proof_l2199_219984


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2199_219973

theorem complex_fraction_sum (x y : ℝ) :
  (1 - Complex.I) / (2 + Complex.I) = Complex.mk x y →
  x + y = -2/5 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2199_219973


namespace NUMINAMATH_CALUDE_fraction_zero_l2199_219923

theorem fraction_zero (a : ℝ) : (a^2 - 1) / (a + 1) = 0 ↔ a = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_l2199_219923


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l2199_219958

theorem binomial_expansion_theorem (a b c k n : ℝ) :
  (n ≥ 2) →
  (a ≠ b) →
  (a * b ≠ 0) →
  (a = k * b + c) →
  (k > 0) →
  (c ≠ 0) →
  (c ≠ b * (k - 1)) →
  (∃ (x y : ℝ), (x + y)^n = (a - b)^n ∧ x + y = 0) →
  (n = -b * (k - 1) / c) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l2199_219958


namespace NUMINAMATH_CALUDE_green_curlers_count_l2199_219911

def total_curlers : ℕ := 16

def pink_curlers : ℕ := total_curlers / 4

def blue_curlers : ℕ := 2 * pink_curlers

def green_curlers : ℕ := total_curlers - (pink_curlers + blue_curlers)

theorem green_curlers_count : green_curlers = 4 := by
  sorry

end NUMINAMATH_CALUDE_green_curlers_count_l2199_219911


namespace NUMINAMATH_CALUDE_grid_coverage_iff_divisible_by_four_l2199_219983

/-- A T-tetromino is a set of four cells in the shape of a "T" -/
def TTetromino : Type := Unit

/-- Represents the property of an n × n grid being completely covered by T-tetrominoes without overlapping -/
def is_completely_covered (n : ℕ) : Prop := sorry

theorem grid_coverage_iff_divisible_by_four (n : ℕ) : 
  is_completely_covered n ↔ 4 ∣ n :=
sorry

end NUMINAMATH_CALUDE_grid_coverage_iff_divisible_by_four_l2199_219983


namespace NUMINAMATH_CALUDE_abc_sum_product_bounds_l2199_219951

theorem abc_sum_product_bounds (a b c : ℝ) (h : a + b + c = 1) :
  ∀ ε > 0, ∃ x : ℝ, x = a * b + a * c + b * c ∧ x ≤ 1/3 ∧ ∃ y : ℝ, y = a * b + a * c + b * c ∧ y < -ε :=
by sorry

end NUMINAMATH_CALUDE_abc_sum_product_bounds_l2199_219951


namespace NUMINAMATH_CALUDE_oil_redistribution_l2199_219907

theorem oil_redistribution (trucks_a : Nat) (boxes_a : Nat) (trucks_b : Nat) (boxes_b : Nat) 
  (containers_per_box : Nat) (new_trucks : Nat) :
  trucks_a = 7 →
  boxes_a = 20 →
  trucks_b = 5 →
  boxes_b = 12 →
  containers_per_box = 8 →
  new_trucks = 10 →
  (trucks_a * boxes_a + trucks_b * boxes_b) * containers_per_box / new_trucks = 160 := by
  sorry

end NUMINAMATH_CALUDE_oil_redistribution_l2199_219907


namespace NUMINAMATH_CALUDE_pond_water_volume_l2199_219993

/-- Calculates the water volume in a pond after a given number of days -/
def water_volume (initial_volume : ℕ) (evaporation_rate : ℕ) (water_added : ℕ) (add_interval : ℕ) (days : ℕ) : ℕ :=
  initial_volume - evaporation_rate * days + (days / add_interval) * water_added

theorem pond_water_volume :
  water_volume 500 1 10 7 35 = 515 := by
  sorry

end NUMINAMATH_CALUDE_pond_water_volume_l2199_219993


namespace NUMINAMATH_CALUDE_point_relationship_l2199_219987

/-- Prove that for points A(-1/2, m) and B(2, n) lying on the line y = 3x + b, m < n. -/
theorem point_relationship (m n b : ℝ) : 
  ((-1/2 : ℝ), m) ∈ {(x, y) | y = 3*x + b} →
  ((2 : ℝ), n) ∈ {(x, y) | y = 3*x + b} →
  m < n := by sorry

end NUMINAMATH_CALUDE_point_relationship_l2199_219987


namespace NUMINAMATH_CALUDE_fraction_value_l2199_219924

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  (a * b) / (c * d) = 180 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l2199_219924


namespace NUMINAMATH_CALUDE_meteorological_forecast_probability_l2199_219994

theorem meteorological_forecast_probability 
  (p q : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) 
  (hq : 0 ≤ q ∧ q ≤ 1) : 
  (p * (1 - q) : ℝ) = 
  (p : ℝ) * (1 - (q : ℝ)) := by
sorry

end NUMINAMATH_CALUDE_meteorological_forecast_probability_l2199_219994


namespace NUMINAMATH_CALUDE_m_3_sufficient_not_necessary_l2199_219992

def A (m : ℝ) : Set ℝ := {-1, m^2}
def B : Set ℝ := {2, 9}

theorem m_3_sufficient_not_necessary :
  (∀ m : ℝ, m = 3 → A m ∩ B = {9}) ∧
  ¬(∀ m : ℝ, A m ∩ B = {9} → m = 3) := by
  sorry

end NUMINAMATH_CALUDE_m_3_sufficient_not_necessary_l2199_219992


namespace NUMINAMATH_CALUDE_preimage_of_neg_one_two_l2199_219997

/-- A mapping f from ℝ² to ℝ² defined as f(x, y) = (2x, x - y) -/
def f : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (2 * x, x - y)

/-- Theorem stating that f(-1/2, -5/2) = (-1, 2) -/
theorem preimage_of_neg_one_two :
  f (-1/2, -5/2) = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_neg_one_two_l2199_219997


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2199_219936

theorem arithmetic_expression_evaluation : 15 - 2 + 4 / 1 / 2 * 8 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2199_219936


namespace NUMINAMATH_CALUDE_even_binomial_coefficients_iff_power_of_two_l2199_219931

def is_power_of_two (n : ℕ+) : Prop :=
  ∃ k : ℕ, n = 2^k

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem even_binomial_coefficients_iff_power_of_two (n : ℕ+) :
  (∀ k : ℕ, 1 ≤ k ∧ k < n → Even (binomial_coefficient n k)) ↔ is_power_of_two n :=
sorry

end NUMINAMATH_CALUDE_even_binomial_coefficients_iff_power_of_two_l2199_219931


namespace NUMINAMATH_CALUDE_average_weight_l2199_219934

theorem average_weight (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 70)
  (avg_bc : (b + c) / 2 = 50)
  (weight_b : b = 60) :
  (a + b + c) / 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_l2199_219934


namespace NUMINAMATH_CALUDE_halloween_candy_problem_l2199_219942

/-- The number of candy pieces Robin's sister gave her -/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

theorem halloween_candy_problem :
  let initial := 23
  let eaten := 7
  let final := 37
  candy_from_sister initial eaten final = 21 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_problem_l2199_219942


namespace NUMINAMATH_CALUDE_chord_length_when_k_2_single_intersection_point_l2199_219915

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 12 * x
def line (k x y : ℝ) : Prop := y = k * x - 1

-- Part 1: Chord length when k = 2
theorem chord_length_when_k_2 :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  parabola x₁ y₁ → parabola x₂ y₂ →
  line 2 x₁ y₁ → line 2 x₂ y₂ →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 75 :=
sorry

-- Part 2: Conditions for single intersection point
theorem single_intersection_point :
  ∀ k : ℝ,
  (∃! x y : ℝ, parabola x y ∧ line k x y) ↔ (k = 0 ∨ k = -3) :=
sorry

end NUMINAMATH_CALUDE_chord_length_when_k_2_single_intersection_point_l2199_219915


namespace NUMINAMATH_CALUDE_function_sum_l2199_219941

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 2)
  (h_def : ∀ x ∈ Set.Ioo 0 1, f x = Real.sin (Real.pi * x)) :
  f (-5/2) + f 1 + f 2 = -1 := by
sorry

end NUMINAMATH_CALUDE_function_sum_l2199_219941


namespace NUMINAMATH_CALUDE_dino_hourly_rate_l2199_219949

/-- Dino's monthly income calculation -/
theorem dino_hourly_rate (hours1 hours2 hours3 : ℕ) (rate2 rate3 : ℚ) 
  (expenses leftover : ℚ) (total_income : ℚ) :
  hours1 = 20 →
  hours2 = 30 →
  hours3 = 5 →
  rate2 = 20 →
  rate3 = 40 →
  expenses = 500 →
  leftover = 500 →
  total_income = expenses + leftover →
  total_income = hours1 * (total_income - hours2 * rate2 - hours3 * rate3) / hours1 + hours2 * rate2 + hours3 * rate3 →
  (total_income - hours2 * rate2 - hours3 * rate3) / hours1 = 10 :=
by sorry

end NUMINAMATH_CALUDE_dino_hourly_rate_l2199_219949


namespace NUMINAMATH_CALUDE_average_difference_l2199_219909

/-- The number of students in the school -/
def num_students : ℕ := 120

/-- The number of teachers in the school -/
def num_teachers : ℕ := 4

/-- The list of class sizes -/
def class_sizes : List ℕ := [40, 30, 30, 20]

/-- Average number of students per class from a teacher's perspective -/
def t : ℚ := (num_students : ℚ) / num_teachers

/-- Average number of students per class from a student's perspective -/
def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes) : ℚ) / num_students

theorem average_difference : t - s = -167/100 := by sorry

end NUMINAMATH_CALUDE_average_difference_l2199_219909


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2199_219999

theorem quadratic_minimum (x : ℝ) :
  ∃ (min : ℝ), ∀ y : ℝ, y = 4 * x^2 + 8 * x + 16 → y ≥ min ∧ ∃ x₀ : ℝ, 4 * x₀^2 + 8 * x₀ + 16 = min :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2199_219999


namespace NUMINAMATH_CALUDE_escalator_speed_l2199_219978

/-- Given an escalator and a person walking on it, calculate the escalator's speed. -/
theorem escalator_speed (escalator_length : ℝ) (person_speed : ℝ) (time_taken : ℝ) : 
  escalator_length = 112 →
  person_speed = 4 →
  time_taken = 8 →
  (person_speed + (escalator_length / time_taken - person_speed)) * time_taken = escalator_length →
  escalator_length / time_taken - person_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_escalator_speed_l2199_219978


namespace NUMINAMATH_CALUDE_cubic_tangent_max_l2199_219980

/-- A cubic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_tangent_max (a b m : ℝ) (hm : m ≠ 0) :
  f' a b m = 0 ∧                   -- Tangent condition (derivative = 0 at x = m)
  f a b m = 0 ∧                    -- Tangent condition (f(m) = 0)
  (∃ x, f a b x = (1/2 : ℝ)) ∧     -- Maximum value condition
  (∀ x, f a b x ≤ (1/2 : ℝ)) →     -- Maximum value condition
  m = (3/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_cubic_tangent_max_l2199_219980


namespace NUMINAMATH_CALUDE_max_balls_in_cube_l2199_219952

theorem max_balls_in_cube (cube_volume ball_volume : ℝ) (h1 : cube_volume = 1000) (h2 : ball_volume = 36 * Real.pi) :
  ⌊cube_volume / ball_volume⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_balls_in_cube_l2199_219952


namespace NUMINAMATH_CALUDE_total_solar_systems_and_planets_l2199_219930

/-- The number of planets in the galaxy -/
def num_planets : ℕ := 20

/-- The number of additional solar systems for each planet -/
def additional_solar_systems : ℕ := 8

/-- The total number of solar systems and planets in the galaxy -/
def total_count : ℕ := num_planets * (additional_solar_systems + 1) + num_planets

theorem total_solar_systems_and_planets :
  total_count = 200 :=
by sorry

end NUMINAMATH_CALUDE_total_solar_systems_and_planets_l2199_219930


namespace NUMINAMATH_CALUDE_equation_solution_l2199_219914

theorem equation_solution : 
  ∀ x : ℝ, x * (x - 1) = x ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2199_219914


namespace NUMINAMATH_CALUDE_divisibility_properties_l2199_219916

theorem divisibility_properties (a m n : ℕ) (ha : a ≥ 2) (hm : m > 0) (hn : n > 0) (h_div : m ∣ n) :
  (∃ k, a^n - 1 = k * (a^m - 1)) ∧
  ((∃ k, a^n + 1 = k * (a^m + 1)) ↔ Odd (n / m)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l2199_219916


namespace NUMINAMATH_CALUDE_remainder_theorem_l2199_219985

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : 
  (x + 2 * u * y) % y = v := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2199_219985


namespace NUMINAMATH_CALUDE_original_ghee_quantity_l2199_219947

/-- Proves that the original quantity of ghee is 30 kg given the conditions of the problem. -/
theorem original_ghee_quantity (x : ℝ) : 
  (0.5 * x = 0.3 * (x + 20)) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_original_ghee_quantity_l2199_219947


namespace NUMINAMATH_CALUDE_exists_quadratic_without_cyclic_solution_l2199_219989

/-- A quadratic polynomial function -/
def QuadraticPolynomial := ℝ → ℝ

/-- Property that checks if a function satisfies the cyclic condition for given a, b, c, d -/
def SatisfiesCyclicCondition (f : QuadraticPolynomial) (a b c d : ℝ) : Prop :=
  f a = b ∧ f b = c ∧ f c = d ∧ f d = a

/-- Theorem stating that there exists a quadratic polynomial for which no distinct a, b, c, d satisfy the cyclic condition -/
theorem exists_quadratic_without_cyclic_solution :
  ∃ f : QuadraticPolynomial, ∀ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a →
    ¬(SatisfiesCyclicCondition f a b c d) :=
sorry

end NUMINAMATH_CALUDE_exists_quadratic_without_cyclic_solution_l2199_219989


namespace NUMINAMATH_CALUDE_jean_card_money_l2199_219913

/-- The amount of money Jean puts in each card for her grandchildren --/
def money_per_card (num_grandchildren : ℕ) (cards_per_grandchild : ℕ) (total_money : ℕ) : ℚ :=
  total_money / (num_grandchildren * cards_per_grandchild)

/-- Theorem: Jean puts $80 in each card for her grandchildren --/
theorem jean_card_money :
  money_per_card 3 2 480 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jean_card_money_l2199_219913


namespace NUMINAMATH_CALUDE_brick_volume_l2199_219995

/-- The volume of a rectangular prism -/
def volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a brick with dimensions 9 cm × 4 cm × 7 cm is 252 cubic centimeters -/
theorem brick_volume :
  volume 4 9 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_brick_volume_l2199_219995


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_l2199_219950

def A : Set ℝ := {-1, 2, 3}
def B (a : ℝ) : Set ℝ := {a + 1, a^2 + 3}

theorem intersection_implies_a_values :
  ∀ a : ℝ, (A ∩ B a = {3}) → (a = 0 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_l2199_219950


namespace NUMINAMATH_CALUDE_train_length_l2199_219919

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 9 → ∃ length : ℝ, 
  (length ≥ 150 ∧ length < 151) ∧ 
  length = speed * (1000 / 3600) * time := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2199_219919


namespace NUMINAMATH_CALUDE_remainder_theorem_l2199_219933

def polynomial (x : ℝ) : ℝ := 8*x^4 - 6*x^3 + 17*x^2 - 27*x + 35

def divisor (x : ℝ) : ℝ := 2*x - 8

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (divisor x) * q x + 1863 :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2199_219933


namespace NUMINAMATH_CALUDE_polynomial_root_implies_k_l2199_219981

theorem polynomial_root_implies_k (k : ℝ) : 
  (3 : ℝ)^3 + k * 3 - 18 = 0 → k = -3 := by sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_k_l2199_219981


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2199_219939

/-- Given a rectangle with perimeter 60 meters and one side three times longer than the other,
    the maximum area is 168.75 square meters. -/
theorem rectangle_max_area (perimeter : ℝ) (ratio : ℝ) (area : ℝ) :
  perimeter = 60 →
  ratio = 3 →
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x = ratio * y ∧ 2 * (x + y) = perimeter ∧ x * y = area) →
  area = 168.75 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2199_219939


namespace NUMINAMATH_CALUDE_tan_beta_value_l2199_219927

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l2199_219927


namespace NUMINAMATH_CALUDE_ten_point_six_trillion_scientific_notation_l2199_219976

-- Define a trillion
def trillion : ℝ := 10^12

-- State the theorem
theorem ten_point_six_trillion_scientific_notation :
  (10.6 * trillion) = 1.06 * 10^13 := by sorry

end NUMINAMATH_CALUDE_ten_point_six_trillion_scientific_notation_l2199_219976


namespace NUMINAMATH_CALUDE_student_tickets_sold_l2199_219972

theorem student_tickets_sold (adult_price student_price total_tickets total_amount : ℚ)
  (h1 : adult_price = 4)
  (h2 : student_price = (5/2))
  (h3 : total_tickets = 59)
  (h4 : total_amount = (445/2))
  (h5 : ∃ (adult_tickets student_tickets : ℚ),
    adult_tickets + student_tickets = total_tickets ∧
    adult_price * adult_tickets + student_price * student_tickets = total_amount) :
  ∃ (student_tickets : ℚ), student_tickets = 9 := by
sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l2199_219972


namespace NUMINAMATH_CALUDE_slices_per_pizza_l2199_219940

theorem slices_per_pizza (total_pizzas : ℕ) (total_slices : ℕ) 
  (h1 : total_pizzas = 21) 
  (h2 : total_slices = 168) : 
  total_slices / total_pizzas = 8 := by
  sorry

end NUMINAMATH_CALUDE_slices_per_pizza_l2199_219940


namespace NUMINAMATH_CALUDE_square_sum_equality_l2199_219957

theorem square_sum_equality : 12^2 + 2*(12*5) + 5^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l2199_219957


namespace NUMINAMATH_CALUDE_sum_of_digits_of_expression_l2199_219946

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The expression (10^(4n^2 + 8) + 1)^2 -/
def expression (n : ℕ) : ℕ := (10^(4*n^2 + 8) + 1)^2

theorem sum_of_digits_of_expression (n : ℕ) (h : n > 0) : 
  sumOfDigits (expression n) = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_expression_l2199_219946


namespace NUMINAMATH_CALUDE_equation_solution_l2199_219918

theorem equation_solution (x : ℝ) :
  (x / 5) / 3 = 9 / (x / 3) → x = 15 * Real.sqrt 1.8 ∨ x = -15 * Real.sqrt 1.8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2199_219918


namespace NUMINAMATH_CALUDE_min_profit_is_266_l2199_219920

/-- Represents the production plan for the clothing factory -/
structure ProductionPlan where
  typeA : ℕ
  typeB : ℕ

/-- Calculates the total cost for a given production plan -/
def totalCost (plan : ProductionPlan) : ℕ :=
  34 * plan.typeA + 42 * plan.typeB

/-- Calculates the total revenue for a given production plan -/
def totalRevenue (plan : ProductionPlan) : ℕ :=
  39 * plan.typeA + 50 * plan.typeB

/-- Calculates the profit for a given production plan -/
def profit (plan : ProductionPlan) : ℤ :=
  totalRevenue plan - totalCost plan

/-- Theorem: The minimum profit is 266 yuan -/
theorem min_profit_is_266 :
  ∃ (minProfit : ℕ), minProfit = 266 ∧
  ∀ (plan : ProductionPlan),
    plan.typeA + plan.typeB = 40 →
    1536 ≤ totalCost plan →
    totalCost plan ≤ 1552 →
    minProfit ≤ profit plan := by
  sorry

#check min_profit_is_266

end NUMINAMATH_CALUDE_min_profit_is_266_l2199_219920


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l2199_219962

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 84 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l2199_219962


namespace NUMINAMATH_CALUDE_multiply_polynomial_equals_difference_of_powers_l2199_219963

theorem multiply_polynomial_equals_difference_of_powers (x : ℝ) :
  (x^4 + 25*x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomial_equals_difference_of_powers_l2199_219963


namespace NUMINAMATH_CALUDE_second_number_is_72_l2199_219932

theorem second_number_is_72 (a b c : ℚ) : 
  a + b + c = 264 ∧ 
  a = 2 * b ∧ 
  c = (1 / 3) * a → 
  b = 72 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_72_l2199_219932


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2199_219912

/-- For a geometric sequence with common ratio 2, the ratio of the 4th term to the 2nd term is 4. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n) :
  a 4 / a 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2199_219912


namespace NUMINAMATH_CALUDE_unique_solution_for_all_y_l2199_219998

theorem unique_solution_for_all_y :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0 :=
by
  -- The unique solution is x = 3/2
  use 3 / 2
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_all_y_l2199_219998


namespace NUMINAMATH_CALUDE_pony_price_calculation_l2199_219925

/-- The regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The discount rate for Pony jeans as a decimal -/
def pony_discount : ℝ := 0.10999999999999996

/-- The sum of discount rates for Fox and Pony jeans as a decimal -/
def total_discount : ℝ := 0.22

/-- The number of Fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings from the purchase in dollars -/
def total_savings : ℝ := 8.91

/-- The regular price of Pony jeans in dollars -/
def pony_price : ℝ := 18

theorem pony_price_calculation :
  fox_price * fox_quantity * (total_discount - pony_discount) +
  pony_price * pony_quantity * pony_discount = total_savings :=
sorry

end NUMINAMATH_CALUDE_pony_price_calculation_l2199_219925


namespace NUMINAMATH_CALUDE_middle_school_count_l2199_219965

structure School where
  total_students : ℕ
  sample_size : ℕ
  middle_school_in_sample : ℕ

def middle_school_students (s : School) : ℕ :=
  s.total_students * s.middle_school_in_sample / s.sample_size

theorem middle_school_count (s : School) 
  (h1 : s.total_students = 2000)
  (h2 : s.sample_size = 400)
  (h3 : s.middle_school_in_sample = 180) :
  middle_school_students s = 900 := by
  sorry

end NUMINAMATH_CALUDE_middle_school_count_l2199_219965


namespace NUMINAMATH_CALUDE_potato_yield_increase_l2199_219906

theorem potato_yield_increase (initial_area initial_yield final_area : ℝ) 
  (h1 : initial_area = 27)
  (h2 : final_area = 24)
  (h3 : initial_area * initial_yield = final_area * (initial_yield * (1 + yield_increase_percentage / 100))) :
  yield_increase_percentage = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_potato_yield_increase_l2199_219906


namespace NUMINAMATH_CALUDE_min_sum_parallel_vectors_l2199_219964

theorem min_sum_parallel_vectors (x y : ℝ) : 
  x > 0 → y > 0 → 
  (∃ (k : ℝ), k ≠ 0 ∧ k • (1 - x, x) = (1, -y)) →
  (∀ a b : ℝ, a > 0 → b > 0 → (∃ (k : ℝ), k ≠ 0 ∧ k • (1 - a, a) = (1, -b)) → a + b ≥ 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∃ (k : ℝ), k ≠ 0 ∧ k • (1 - a, a) = (1, -b)) ∧ a + b = 4) :=
by sorry


end NUMINAMATH_CALUDE_min_sum_parallel_vectors_l2199_219964


namespace NUMINAMATH_CALUDE_distance_from_blast_site_l2199_219974

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 330

/-- The time between the first blast and when the man heard the second blast, in seconds -/
def time_between_blasts : ℝ := 30 * 60 + 24

/-- The time between the first and second blasts, in seconds -/
def time_between_actual_blasts : ℝ := 30 * 60

/-- The distance the man traveled when he heard the second blast -/
def distance_traveled : ℝ := speed_of_sound * (time_between_blasts - time_between_actual_blasts)

theorem distance_from_blast_site :
  distance_traveled = 7920 := by sorry

end NUMINAMATH_CALUDE_distance_from_blast_site_l2199_219974


namespace NUMINAMATH_CALUDE_last_digit_389_base4_l2199_219929

def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem last_digit_389_base4 :
  (decimal_to_base4 389).getLast? = some 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_389_base4_l2199_219929


namespace NUMINAMATH_CALUDE_exists_two_sum_of_squares_representations_l2199_219901

theorem exists_two_sum_of_squares_representations : 
  ∃ (n : ℕ) (a b c d : ℕ), 
    n < 100 ∧ 
    a ≠ b ∧ 
    c ≠ d ∧ 
    (a, b) ≠ (c, d) ∧
    (a, b) ≠ (d, c) ∧
    n = a^2 + b^2 ∧ 
    n = c^2 + d^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_two_sum_of_squares_representations_l2199_219901


namespace NUMINAMATH_CALUDE_asha_money_problem_l2199_219988

/-- Asha's money problem -/
theorem asha_money_problem (brother_loan : ℕ) (father_loan : ℕ) (mother_loan : ℕ) (granny_gift : ℕ) (savings : ℕ) (spent_fraction : ℚ) :
  brother_loan = 20 →
  father_loan = 40 →
  mother_loan = 30 →
  granny_gift = 70 →
  savings = 100 →
  spent_fraction = 3 / 4 →
  ∃ (remaining : ℕ), remaining = 65 ∧ 
    remaining = (brother_loan + father_loan + mother_loan + granny_gift + savings) - 
                (spent_fraction * (brother_loan + father_loan + mother_loan + granny_gift + savings)).floor :=
by sorry

end NUMINAMATH_CALUDE_asha_money_problem_l2199_219988


namespace NUMINAMATH_CALUDE_parabola_focus_equation_l2199_219986

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in standard form -/
inductive Parabola
  | VertexAtOrigin (p : ℝ) : Parabola
  | FocusOnXAxis (p : ℝ) : Parabola

/-- Function to check if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Function to check if a point is on the x-axis -/
def isPointOnXAxis (p : Point) : Prop :=
  p.y = 0

/-- Function to check if a point is on the y-axis -/
def isPointOnYAxis (p : Point) : Prop :=
  p.x = 0

/-- Theorem stating the relationship between the focus of a parabola and its equation -/
theorem parabola_focus_equation (l : Line) (f : Point) :
  (l.a = 3 ∧ l.b = -4 ∧ l.c = -12) →
  isPointOnLine f l →
  (isPointOnXAxis f ∨ isPointOnYAxis f) →
  (∃ p : Parabola, p = Parabola.VertexAtOrigin (-12) ∨ p = Parabola.FocusOnXAxis 8) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_equation_l2199_219986


namespace NUMINAMATH_CALUDE_even_cubic_implies_odd_factor_l2199_219970

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x ∈ ℝ -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

/-- Given f(x) = x³ * g(x) is an even function, prove that g(x) is odd -/
theorem even_cubic_implies_odd_factor
    (g : ℝ → ℝ) (f : ℝ → ℝ)
    (h1 : ∀ x, f x = x^3 * g x)
    (h2 : IsEven f) :
  IsOdd g :=
by sorry

end NUMINAMATH_CALUDE_even_cubic_implies_odd_factor_l2199_219970


namespace NUMINAMATH_CALUDE_fermat_last_digit_l2199_219908

/-- Fermat number -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- The last digit of Fermat numbers for n ≥ 2 is always 7 -/
theorem fermat_last_digit (n : ℕ) (h : n ≥ 2) : F n % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fermat_last_digit_l2199_219908


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2199_219910

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2199_219910


namespace NUMINAMATH_CALUDE_christinas_speed_limit_l2199_219991

def total_distance : ℝ := 210
def friend_driving_time : ℝ := 3
def friend_speed_limit : ℝ := 40
def christina_driving_time : ℝ := 3  -- 180 minutes converted to hours

theorem christinas_speed_limit :
  ∃ (christina_speed : ℝ),
    christina_speed * christina_driving_time + 
    friend_speed_limit * friend_driving_time = total_distance ∧
    christina_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_christinas_speed_limit_l2199_219991


namespace NUMINAMATH_CALUDE_subtraction_puzzle_l2199_219903

theorem subtraction_puzzle (X Y : ℕ) : 
  X ≤ 9 → Y ≤ 9 → 45 + 8 * Y = 100 + 10 * X + 2 → X + Y = 10 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_puzzle_l2199_219903


namespace NUMINAMATH_CALUDE_action_figures_per_shelf_l2199_219938

theorem action_figures_per_shelf 
  (total_figures : ℕ) 
  (num_shelves : ℕ) 
  (h1 : total_figures = 80) 
  (h2 : num_shelves = 8) : 
  total_figures / num_shelves = 10 := by
  sorry

end NUMINAMATH_CALUDE_action_figures_per_shelf_l2199_219938


namespace NUMINAMATH_CALUDE_min_value_expression_equality_achieved_l2199_219959

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 1)^2) ≥ Real.sqrt 13 := by
  sorry

theorem equality_achieved : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 1)^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_achieved_l2199_219959


namespace NUMINAMATH_CALUDE_original_number_proof_l2199_219955

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 / x) :
  x = Real.sqrt 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2199_219955


namespace NUMINAMATH_CALUDE_powerjet_pump_l2199_219953

/-- The amount of water pumped in a given time -/
def water_pumped (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Theorem: A pump operating at 500 gallons per hour will pump 250 gallons in 30 minutes -/
theorem powerjet_pump (rate : ℝ) (time : ℝ) (h1 : rate = 500) (h2 : time = 1/2) : 
  water_pumped rate time = 250 := by
  sorry

end NUMINAMATH_CALUDE_powerjet_pump_l2199_219953


namespace NUMINAMATH_CALUDE_no_formula_fits_all_data_l2199_219902

def data : List (ℕ × ℕ) := [(1, 2), (2, 6), (3, 12), (4, 20), (5, 30)]

def formula_a (x : ℕ) : ℕ := 4 * x - 2
def formula_b (x : ℕ) : ℕ := x^3 - x^2 + 2*x
def formula_c (x : ℕ) : ℕ := 2 * x^2
def formula_d (x : ℕ) : ℕ := x^2 + 2*x + 1

theorem no_formula_fits_all_data :
  ¬(∀ (x y : ℕ), (x, y) ∈ data → 
    (y = formula_a x ∨ y = formula_b x ∨ y = formula_c x ∨ y = formula_d x)) :=
by sorry

end NUMINAMATH_CALUDE_no_formula_fits_all_data_l2199_219902


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l2199_219944

theorem greatest_integer_inequality (x : ℤ) :
  (∀ y : ℤ, 3 * y^2 - 5 * y - 2 < 4 - 2 * y → y ≤ 1) ∧
  (3 * 1^2 - 5 * 1 - 2 < 4 - 2 * 1) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l2199_219944


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l2199_219921

theorem geometric_progression_problem (b₃ b₆ : ℚ) 
  (h₁ : b₃ = -1)
  (h₂ : b₆ = 27/8) :
  ∃ (b₁ q : ℚ), 
    b₁ = -4/9 ∧ 
    q = -3/2 ∧ 
    b₃ = b₁ * q^2 ∧ 
    b₆ = b₁ * q^5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l2199_219921


namespace NUMINAMATH_CALUDE_pigs_in_blanket_calculation_l2199_219935

/-- The number of appetizers per guest -/
def appetizers_per_guest : ℕ := 6

/-- The number of guests -/
def number_of_guests : ℕ := 30

/-- The number of dozen deviled eggs -/
def dozen_deviled_eggs : ℕ := 3

/-- The number of dozen kebabs -/
def dozen_kebabs : ℕ := 2

/-- The additional number of dozen appetizers to make -/
def additional_dozen_appetizers : ℕ := 8

/-- The number of items in a dozen -/
def items_per_dozen : ℕ := 12

theorem pigs_in_blanket_calculation : 
  let total_appetizers := appetizers_per_guest * number_of_guests
  let made_appetizers := dozen_deviled_eggs * items_per_dozen + dozen_kebabs * items_per_dozen
  let remaining_appetizers := total_appetizers - made_appetizers
  let planned_additional_appetizers := additional_dozen_appetizers * items_per_dozen
  let pigs_in_blanket := remaining_appetizers - planned_additional_appetizers
  (pigs_in_blanket / items_per_dozen : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_pigs_in_blanket_calculation_l2199_219935


namespace NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l2199_219979

theorem largest_number_with_equal_quotient_and_remainder :
  ∀ (A B C : ℕ),
    A = 8 * B + C →
    B = C →
    0 ≤ C ∧ C < 8 →
    A ≤ 63 ∧ (∃ (A' : ℕ), A' = 63 ∧ ∃ (B' C' : ℕ), A' = 8 * B' + C' ∧ B' = C' ∧ 0 ≤ C' ∧ C' < 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l2199_219979


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2199_219904

/-- A sequence of 8 positive real numbers -/
structure Sequence :=
  (terms : Fin 8 → ℝ)
  (positive : ∀ i, terms i > 0)

/-- Predicate for a geometric sequence -/
def is_geometric (s : Sequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ i : Fin 7, s.terms i.succ = q * s.terms i

theorem sufficient_not_necessary_condition (s : Sequence) :
  (s.terms 0 + s.terms 7 < s.terms 3 + s.terms 4 → ¬is_geometric s) ∧
  ∃ s' : Sequence, ¬is_geometric s' ∧ s'.terms 0 + s'.terms 7 ≥ s'.terms 3 + s'.terms 4 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2199_219904


namespace NUMINAMATH_CALUDE_sequence_existence_iff_N_bound_l2199_219922

theorem sequence_existence_iff_N_bound (N : ℕ+) :
  (∃ s : ℕ → ℕ+, 
    (∀ n, s n < s (n + 1)) ∧ 
    (∃ p : ℕ+, ∀ n, s (n + 1) - s n = s (n + 1 + p) - s (n + p)) ∧
    (∀ n : ℕ+, s (s n) - s (s (n - 1)) ≤ N ∧ N < s (1 + s n) - s (s (n - 1))))
  ↔
  (∃ t : ℕ+, t^2 ≤ N ∧ N < t^2 + t) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_iff_N_bound_l2199_219922


namespace NUMINAMATH_CALUDE_power_of_power_l2199_219943

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l2199_219943


namespace NUMINAMATH_CALUDE_max_expected_expenditure_l2199_219900

/-- Linear regression model for fiscal revenue and expenditure -/
def fiscal_model (x y a b ε : ℝ) : Prop :=
  y = a + b * x + ε

/-- Theorem: Maximum expected expenditure given fiscal revenue -/
theorem max_expected_expenditure
  (a b x y ε : ℝ)
  (model : fiscal_model x y a b ε)
  (h_a : a = 2)
  (h_b : b = 0.8)
  (h_ε : |ε| ≤ 0.5)
  (h_x : x = 10) :
  y ≤ 10.5 := by
  sorry

#check max_expected_expenditure

end NUMINAMATH_CALUDE_max_expected_expenditure_l2199_219900


namespace NUMINAMATH_CALUDE_number_of_routes_l2199_219990

-- Define the cities
inductive City : Type
| A | B | C | D | F

-- Define the roads
inductive Road : Type
| AB | AD | AF | BC | BD | CD | DF

-- Define a route as a list of roads
def Route := List Road

-- Function to check if a route is valid (uses each road exactly once and starts at A and ends at B)
def isValidRoute (r : Route) : Prop := sorry

-- Function to count the number of valid routes
def countValidRoutes : Nat := sorry

-- Theorem to prove
theorem number_of_routes : countValidRoutes = 16 := by sorry

end NUMINAMATH_CALUDE_number_of_routes_l2199_219990


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2199_219967

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, x > 1 → x^2 + 2*x > 0) ∧
  (∃ x : ℝ, x^2 + 2*x > 0 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2199_219967


namespace NUMINAMATH_CALUDE_range_of_fraction_l2199_219917

theorem range_of_fraction (x y : ℝ) 
  (h1 : x - 2*y + 4 ≥ 0) 
  (h2 : x ≤ 2) 
  (h3 : x + y - 2 ≥ 0) : 
  1/4 ≤ (y + 1) / (x + 2) ∧ (y + 1) / (x + 2) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_fraction_l2199_219917


namespace NUMINAMATH_CALUDE_sin_c_special_triangle_l2199_219961

/-- Given a right triangle ABC where A is the right angle, if the logarithms of 
    the side lengths form an arithmetic sequence with a negative common difference, 
    then sin C equals (√5 - 1)/2 -/
theorem sin_c_special_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_right_angle : a^2 = b^2 + c^2)
  (h_arithmetic_seq : ∃ d : ℝ, d < 0 ∧ Real.log a - Real.log b = d ∧ Real.log b - Real.log c = d) :
  Real.sin (Real.arccos (c / a)) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_c_special_triangle_l2199_219961


namespace NUMINAMATH_CALUDE_total_puzzle_time_l2199_219926

def puzzle_time (warm_up_time : ℕ) (additional_puzzles : ℕ) (time_factor : ℕ) : ℕ :=
  warm_up_time + additional_puzzles * (warm_up_time * time_factor)

theorem total_puzzle_time :
  puzzle_time 10 2 3 = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_total_puzzle_time_l2199_219926


namespace NUMINAMATH_CALUDE_min_marked_cells_13x13_board_l2199_219956

/-- Represents a rectangular board -/
structure Board :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a rectangle that can be placed on the board -/
structure Rectangle :=
  (length : Nat)
  (width : Nat)

/-- Function to calculate the minimum number of cells to mark -/
def minMarkedCells (b : Board) (r : Rectangle) : Nat :=
  sorry

/-- Theorem stating that 84 is the minimum number of cells to mark -/
theorem min_marked_cells_13x13_board (b : Board) (r : Rectangle) :
  b.rows = 13 ∧ b.cols = 13 ∧ r.length = 6 ∧ r.width = 1 →
  minMarkedCells b r = 84 :=
by sorry

end NUMINAMATH_CALUDE_min_marked_cells_13x13_board_l2199_219956


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l2199_219928

theorem triangle_angle_relation (A B C : Real) : 
  A + B + C = Real.pi →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  Real.sin A = Real.cos B →
  Real.sin A = Real.tan C →
  Real.cos A ^ 3 + Real.cos A ^ 2 - Real.cos A = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l2199_219928


namespace NUMINAMATH_CALUDE_absent_students_count_l2199_219977

/-- The number of classes at Webster Middle School -/
def num_classes : ℕ := 18

/-- The number of students in each class -/
def students_per_class : ℕ := 28

/-- The number of students present on Monday -/
def students_present : ℕ := 496

/-- The number of absent students -/
def absent_students : ℕ := num_classes * students_per_class - students_present

theorem absent_students_count : absent_students = 8 := by
  sorry

end NUMINAMATH_CALUDE_absent_students_count_l2199_219977


namespace NUMINAMATH_CALUDE_proper_divisor_of_two_square_representations_l2199_219937

theorem proper_divisor_of_two_square_representations (n s t u v : ℕ) 
  (h1 : n = s^2 + t^2)
  (h2 : n = u^2 + v^2)
  (h3 : s ≥ t)
  (h4 : t ≥ 0)
  (h5 : u ≥ v)
  (h6 : v ≥ 0)
  (h7 : s > u) :
  1 < Nat.gcd (s * u - t * v) n ∧ Nat.gcd (s * u - t * v) n < n :=
by sorry

end NUMINAMATH_CALUDE_proper_divisor_of_two_square_representations_l2199_219937


namespace NUMINAMATH_CALUDE_linear_function_property_l2199_219971

/-- A linear function is a function f such that f(x) = mx + b for some constants m and b. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- Given a linear function g such that g(10) - g(4) = 24, prove that g(16) - g(4) = 48. -/
theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g) 
  (h_condition : g 10 - g 4 = 24) : 
  g 16 - g 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l2199_219971


namespace NUMINAMATH_CALUDE_no_solution_lcm_equation_l2199_219960

theorem no_solution_lcm_equation :
  ¬ ∃ (a b : ℕ), 2 * a + 3 * b = Nat.lcm a b := by
  sorry

end NUMINAMATH_CALUDE_no_solution_lcm_equation_l2199_219960


namespace NUMINAMATH_CALUDE_medal_ratio_is_two_to_one_l2199_219954

/-- The ratio of swimming medals to track medals -/
def medal_ratio (total_medals track_medals badminton_medals : ℕ) : ℚ :=
  let swimming_medals := total_medals - track_medals - badminton_medals
  (swimming_medals : ℚ) / track_medals

/-- Theorem stating that the ratio of swimming medals to track medals is 2:1 -/
theorem medal_ratio_is_two_to_one :
  medal_ratio 20 5 5 = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_medal_ratio_is_two_to_one_l2199_219954
