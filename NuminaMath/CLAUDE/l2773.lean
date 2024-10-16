import Mathlib

namespace NUMINAMATH_CALUDE_calculation_proof_l2773_277376

theorem calculation_proof : (-1) * (-4) + 3^2 / (7 - 4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2773_277376


namespace NUMINAMATH_CALUDE_equation_solution_l2773_277380

theorem equation_solution : ∃ x : ℝ, 35 * 2 - 10 = 5 * x + 20 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2773_277380


namespace NUMINAMATH_CALUDE_cosine_sine_equation_solutions_l2773_277363

open Real

theorem cosine_sine_equation_solutions (a α : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   cos (x₁ - a) - sin (x₁ + 2*α) = 0 ∧
   cos (x₂ - a) - sin (x₂ + 2*α) = 0 ∧
   ¬ ∃ k : ℤ, x₁ - x₂ = k * π) ↔ 
  ∃ t : ℤ, a = π * (4*t + 1) / 6 :=
by sorry

end NUMINAMATH_CALUDE_cosine_sine_equation_solutions_l2773_277363


namespace NUMINAMATH_CALUDE_cool_drink_solution_l2773_277309

/-- Proves that 12 liters of water were added to achieve the given conditions -/
theorem cool_drink_solution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (jasmine_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 80 →
  initial_concentration = 0.1 →
  jasmine_added = 8 →
  final_concentration = 0.16 →
  ∃ (water_added : ℝ),
    water_added = 12 ∧
    (initial_volume * initial_concentration + jasmine_added) / 
    (initial_volume + jasmine_added + water_added) = final_concentration :=
by
  sorry


end NUMINAMATH_CALUDE_cool_drink_solution_l2773_277309


namespace NUMINAMATH_CALUDE_division_problem_l2773_277318

theorem division_problem (a b c : ℝ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2773_277318


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2773_277359

theorem complex_arithmetic_equality : 
  ((-4 : ℝ) ^ 5) ^ (1/5) - (-5 : ℝ) ^ 2 - 5 + ((-43 : ℝ) ^ 4) ^ (1/4) - (-(3 : ℝ) ^ 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2773_277359


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2773_277360

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 - x| ≥ 1} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2773_277360


namespace NUMINAMATH_CALUDE_cone_division_ratio_l2773_277381

/-- Given a right circular cone with height 6 inches and base radius 4 inches,
    if a plane parallel to the base divides the cone into two solids C and F
    such that the ratio of their surface areas and volumes is k = 3/7,
    then the radius of the smaller cone C is (4 * (3/10)^(1/3)) / 3 times the original radius. -/
theorem cone_division_ratio (h : ℝ) (r : ℝ) (k : ℝ) :
  h = 6 →
  r = 4 →
  k = 3 / 7 →
  ∃ x : ℝ,
    x = (4 * (3 / 10) ^ (1 / 3)) / 3 * r ∧
    (π * x^2 + π * x * (Real.sqrt (h^2 + r^2) * x / r)) / 
    (π * r^2 + π * r * Real.sqrt (h^2 + r^2) - 
     (π * x^2 + π * x * (Real.sqrt (h^2 + r^2) * x / r))) = k ∧
    ((1 / 3) * π * x^2 * (h * x / r)) / 
    ((1 / 3) * π * r^2 * h - (1 / 3) * π * x^2 * (h * x / r)) = k :=
by
  sorry


end NUMINAMATH_CALUDE_cone_division_ratio_l2773_277381


namespace NUMINAMATH_CALUDE_chessboard_rearrangement_impossibility_l2773_277386

theorem chessboard_rearrangement_impossibility :
  ∀ (initial_placement final_placement : Fin 8 → Fin 8 → Bool),
  (∀ i j : Fin 8, (∃! k : Fin 8, initial_placement i k = true) ∧ 
                  (∃! k : Fin 8, initial_placement k j = true)) →
  (∀ i j : Fin 8, (∃! k : Fin 8, final_placement i k = true) ∧ 
                  (∃! k : Fin 8, final_placement k j = true)) →
  (∀ i j : Fin 8, initial_placement i j = true → 
    ∃ i' j' : Fin 8, final_placement i' j' = true ∧ 
    (i'.val + j'.val : ℕ) > (i.val + j.val)) →
  False :=
sorry

end NUMINAMATH_CALUDE_chessboard_rearrangement_impossibility_l2773_277386


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2773_277327

/-- Given an arithmetic sequence {a_n} where a_2 = 1 and a_3 + a_5 = 4,
    the common difference of the sequence is 1/2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ) -- The sequence as a function from natural numbers to rationals
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- Arithmetic sequence condition
  (h_a2 : a 2 = 1) -- Given: a_2 = 1
  (h_sum : a 3 + a 5 = 4) -- Given: a_3 + a_5 = 4
  : ∃ d : ℚ, d = 1/2 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2773_277327


namespace NUMINAMATH_CALUDE_proportional_function_quadrants_l2773_277345

theorem proportional_function_quadrants (k : ℝ) :
  let f : ℝ → ℝ := λ x => (-k^2 - 2) * x
  (∀ x y, f x = y → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by sorry

end NUMINAMATH_CALUDE_proportional_function_quadrants_l2773_277345


namespace NUMINAMATH_CALUDE_ratio_equality_product_l2773_277305

theorem ratio_equality_product (x : ℝ) : 
  (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4) → 
  ∃ y : ℝ, (x = 0 ∨ x = 5) ∧ x * y = 0 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_product_l2773_277305


namespace NUMINAMATH_CALUDE_derek_rides_more_than_carla_l2773_277303

-- Define the speeds and times
def carla_speed : ℝ := 12
def derek_speed : ℝ := 15
def derek_time : ℝ := 3
def time_difference : ℝ := 0.5

-- Theorem statement
theorem derek_rides_more_than_carla :
  derek_speed * derek_time - carla_speed * (derek_time + time_difference) = 3 := by
  sorry

end NUMINAMATH_CALUDE_derek_rides_more_than_carla_l2773_277303


namespace NUMINAMATH_CALUDE_monotone_increasing_quadratic_l2773_277371

/-- A function f(x) = 4x^2 - kx - 8 is monotonically increasing on [5, +∞) if and only if k ≤ 40 -/
theorem monotone_increasing_quadratic (k : ℝ) :
  (∀ x ≥ 5, Monotone (fun x => 4 * x^2 - k * x - 8)) ↔ k ≤ 40 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_quadratic_l2773_277371


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2773_277324

def tangent_line (f : ℝ → ℝ) (a : ℝ) (m : ℝ) (b : ℝ) :=
  ∀ x, f a + m * (x - a) = m * x + b

theorem tangent_line_sum (f : ℝ → ℝ) :
  tangent_line f 5 (-1) 8 → f 5 + (deriv f) 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2773_277324


namespace NUMINAMATH_CALUDE_peter_drew_age_difference_l2773_277391

/-- Proves that Peter is 4 years older than Drew given the conditions in the problem --/
theorem peter_drew_age_difference : 
  ∀ (maya drew peter john jacob : ℕ),
  drew = maya + 5 →
  peter > drew →
  john = 30 →
  john = 2 * maya →
  jacob + 2 = (peter + 2) / 2 →
  jacob = 11 →
  peter - drew = 4 := by
  sorry

end NUMINAMATH_CALUDE_peter_drew_age_difference_l2773_277391


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l2773_277370

/-- Given a parabola y² = 4x and a point M on the parabola whose distance to the focus is 3,
    prove that the x-coordinate of M is 2. -/
theorem parabola_point_x_coordinate (x y : ℝ) : 
  y^2 = 4*x →  -- M is on the parabola y² = 4x
  (x - 1)^2 + y^2 = 3^2 →  -- Distance from M to focus (1, 0) is 3
  x = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l2773_277370


namespace NUMINAMATH_CALUDE_square_number_divisible_by_nine_between_40_and_90_l2773_277316

theorem square_number_divisible_by_nine_between_40_and_90 :
  ∃ x : ℕ, x^2 = x ∧ x % 9 = 0 ∧ 40 < x ∧ x < 90 → x = 81 :=
by sorry

end NUMINAMATH_CALUDE_square_number_divisible_by_nine_between_40_and_90_l2773_277316


namespace NUMINAMATH_CALUDE_painted_cells_theorem_l2773_277368

/-- Represents a rectangular grid with painted cells -/
structure PaintedGrid where
  rows : ℕ
  cols : ℕ
  painted_cells : ℕ

/-- Calculates the number of painted cells in a grid with the given painting pattern -/
def calculate_painted_cells (k l : ℕ) : ℕ :=
  (2 * k + 1) * (2 * l + 1) - k * l

/-- Theorem stating the possible numbers of painted cells given the conditions -/
theorem painted_cells_theorem :
  ∀ (grid : PaintedGrid),
  (∃ (k l : ℕ), 
    grid.rows = 2 * k + 1 ∧ 
    grid.cols = 2 * l + 1 ∧ 
    k * l = 74) →
  grid.painted_cells = 373 ∨ grid.painted_cells = 301 :=
by sorry

end NUMINAMATH_CALUDE_painted_cells_theorem_l2773_277368


namespace NUMINAMATH_CALUDE_sum_of_valid_a_values_l2773_277374

theorem sum_of_valid_a_values : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, 
    (∃ x : ℝ, x + 1 > (x - 1) / 3 ∧ x + a < 3) ∧ 
    (∃ y : ℤ, y > 0 ∧ y ≠ 2 ∧ (y - a) / (y - 2) + 1 = 1 / (y - 2))) ∧
  (∀ a : ℤ, 
    ((∃ x : ℝ, x + 1 > (x - 1) / 3 ∧ x + a < 3) ∧ 
     (∃ y : ℤ, y > 0 ∧ y ≠ 2 ∧ (y - a) / (y - 2) + 1 = 1 / (y - 2))) → 
    a ∈ S) ∧
  S.sum id = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_valid_a_values_l2773_277374


namespace NUMINAMATH_CALUDE_lab_coat_uniform_ratio_l2773_277353

theorem lab_coat_uniform_ratio :
  ∀ (num_uniforms num_lab_coats num_total : ℕ),
    num_uniforms = 12 →
    num_lab_coats = 6 * num_uniforms →
    num_total = num_lab_coats + num_uniforms →
    num_total % 14 = 0 →
    num_lab_coats / num_uniforms = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_lab_coat_uniform_ratio_l2773_277353


namespace NUMINAMATH_CALUDE_angle_sum_in_triangle_l2773_277330

theorem angle_sum_in_triangle (A B C : ℝ) : 
  -- Triangle ABC exists
  -- Sum of angles A and B is 80°
  (A + B = 80) →
  -- Sum of all angles in a triangle is 180°
  (A + B + C = 180) →
  -- Angle C measures 100°
  C = 100 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_in_triangle_l2773_277330


namespace NUMINAMATH_CALUDE_max_gcd_value_l2773_277351

theorem max_gcd_value (m : ℕ+) : 
  (Nat.gcd (15 * m.val + 4) (14 * m.val + 3) ≤ 11) ∧ 
  (∃ m : ℕ+, Nat.gcd (15 * m.val + 4) (14 * m.val + 3) = 11) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_value_l2773_277351


namespace NUMINAMATH_CALUDE_thread_length_calculation_l2773_277393

theorem thread_length_calculation (original_length : ℝ) (additional_fraction : ℝ) : 
  original_length = 12 →
  additional_fraction = 3/4 →
  original_length + (additional_fraction * original_length) = 21 := by
  sorry

end NUMINAMATH_CALUDE_thread_length_calculation_l2773_277393


namespace NUMINAMATH_CALUDE_decimal_93_to_binary_l2773_277329

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryRepresentation := List Nat

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : Nat) : BinaryRepresentation :=
  sorry

/-- Checks if a given BinaryRepresentation is valid (contains only 0s and 1s) -/
def isValidBinary (b : BinaryRepresentation) : Prop :=
  sorry

/-- Converts a binary representation back to decimal -/
def binaryToDecimal (b : BinaryRepresentation) : Nat :=
  sorry

theorem decimal_93_to_binary :
  let binary : BinaryRepresentation := [1, 0, 1, 1, 1, 0, 1]
  isValidBinary binary ∧
  binaryToDecimal binary = 93 ∧
  decimalToBinary 93 = binary :=
by sorry

end NUMINAMATH_CALUDE_decimal_93_to_binary_l2773_277329


namespace NUMINAMATH_CALUDE_sequence_limit_is_two_l2773_277366

/-- The limit of the sequence √(n(n+2)) - √(n^2 - 2n + 3) as n approaches infinity is 2 -/
theorem sequence_limit_is_two :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |Real.sqrt (n * (n + 2)) - Real.sqrt (n^2 - 2*n + 3) - 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_sequence_limit_is_two_l2773_277366


namespace NUMINAMATH_CALUDE_sqrt_two_simplification_l2773_277377

theorem sqrt_two_simplification : 4 * Real.sqrt 2 - Real.sqrt 2 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_simplification_l2773_277377


namespace NUMINAMATH_CALUDE_solution_set_l2773_277323

/-- A function that checks if three positive real numbers can form a non-degenerate triangle -/
def is_triangle (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y > z ∧ x + z > y ∧ y + z > x

/-- The property that n must satisfy -/
def satisfies_condition (n : ℕ) : Prop :=
  n > 0 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    ∃ (l j k : ℕ), is_triangle (a * n ^ k) (b * n ^ j) (c * n ^ l)

/-- The main theorem stating that only 2, 3, and 4 satisfy the condition -/
theorem solution_set : {n : ℕ | satisfies_condition n} = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_l2773_277323


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l2773_277321

theorem positive_real_inequalities (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^2 - b^2 = 1 → a - b < 1) ∧
  (|a^2 - b^2| = 1 → |a - b| < 1) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l2773_277321


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l2773_277350

/-- Given a line and a circle with no common points, prove that a line through a point
    on that line intersects a specific ellipse at exactly two points. -/
theorem line_ellipse_intersection (m n : ℝ) : 
  (∀ x y : ℝ, m*x + n*y - 3 = 0 → x^2 + y^2 ≠ 3) →
  0 < m^2 + n^2 →
  m^2 + n^2 < 3 →
  ∃! (p : ℕ), p = 2 ∧ 
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
      (∃ (k : ℝ), x₁ = m*k ∧ y₁ = n*k) ∧
      (∃ (k : ℝ), x₂ = m*k ∧ y₂ = n*k) ∧
      x₁^2/7 + y₁^2/3 = 1 ∧
      x₂^2/7 + y₂^2/3 = 1 ∧
      (∀ x y : ℝ, (∃ k : ℝ, x = m*k ∧ y = n*k) → 
        x^2/7 + y^2/3 = 1 → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l2773_277350


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2773_277365

-- Define the variables
variable (a b c p q r : ℝ)

-- Define the conditions
axiom eq1 : 17 * p + b * q + c * r = 0
axiom eq2 : a * p + 29 * q + c * r = 0
axiom eq3 : a * p + b * q + 56 * r = 0
axiom a_ne_17 : a ≠ 17
axiom p_ne_0 : p ≠ 0

-- State the theorem
theorem sum_of_fractions_equals_one :
  a / (a - 17) + b / (b - 29) + c / (c - 56) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2773_277365


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2773_277390

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i ^ 2 = -1 →
  Complex.im (i / (2 + i)) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2773_277390


namespace NUMINAMATH_CALUDE_euler_conjecture_counterexample_l2773_277358

theorem euler_conjecture_counterexample : ∃! (n : ℕ), n > 0 ∧ n^5 = 133^5 + 110^5 + 84^5 + 27^5 := by
  sorry

end NUMINAMATH_CALUDE_euler_conjecture_counterexample_l2773_277358


namespace NUMINAMATH_CALUDE_sum_squares_plus_product_lower_bound_l2773_277310

theorem sum_squares_plus_product_lower_bound 
  (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a + b + c = 3) : 
  a^2 + b^2 + c^2 + a*b*c ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_plus_product_lower_bound_l2773_277310


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l2773_277352

/-- The number of jelly beans remaining in the container after distribution --/
def remaining_jelly_beans (initial : ℕ) (people : ℕ) (first_group : ℕ) (last_group : ℕ) (last_group_takes : ℕ) : ℕ :=
  initial - (first_group * (2 * last_group_takes) + last_group * last_group_takes)

/-- Theorem stating the number of remaining jelly beans --/
theorem jelly_bean_problem :
  remaining_jelly_beans 8000 10 6 4 400 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_problem_l2773_277352


namespace NUMINAMATH_CALUDE_roses_per_day_l2773_277398

theorem roses_per_day (total_roses : ℕ) (days : ℕ) (dozens_per_day : ℕ) 
  (h1 : total_roses = 168) 
  (h2 : days = 7) 
  (h3 : dozens_per_day * 12 * days = total_roses) : 
  dozens_per_day = 2 := by
  sorry

end NUMINAMATH_CALUDE_roses_per_day_l2773_277398


namespace NUMINAMATH_CALUDE_broken_line_enclosing_circle_l2773_277312

/-- A closed broken line in a metric space -/
structure ClosedBrokenLine (α : Type*) [MetricSpace α] where
  points : Set α
  is_closed : IsClosed points
  is_connected : IsConnected points
  perimeter : ℝ

/-- Theorem: Any closed broken line can be enclosed in a circle with radius not exceeding its perimeter divided by 4 -/
theorem broken_line_enclosing_circle 
  {α : Type*} [MetricSpace α] (L : ClosedBrokenLine α) :
  ∃ (center : α), ∀ (p : α), p ∈ L.points → dist center p ≤ L.perimeter / 4 := by
  sorry

end NUMINAMATH_CALUDE_broken_line_enclosing_circle_l2773_277312


namespace NUMINAMATH_CALUDE_least_k_inequality_l2773_277328

theorem least_k_inequality (a b c : ℝ) : 
  ∃ (k : ℝ), k = 8 ∧ (∀ (x : ℝ), x ≥ k → 
    (2*a/(a-b))^2 + (2*b/(b-c))^2 + (2*c/(c-a))^2 + x ≥ 
    4*((2*a/(a-b)) + (2*b/(b-c)) + (2*c/(c-a)))) ∧
  (∀ (y : ℝ), y < k → 
    ∃ (a' b' c' : ℝ), (2*a'/(a'-b'))^2 + (2*b'/(b'-c'))^2 + (2*c'/(c'-a'))^2 + y < 
    4*((2*a'/(a'-b')) + (2*b'/(b'-c')) + (2*c'/(c'-a')))) :=
sorry

end NUMINAMATH_CALUDE_least_k_inequality_l2773_277328


namespace NUMINAMATH_CALUDE_paper_strip_length_l2773_277379

theorem paper_strip_length (strip_length : ℝ) : 
  strip_length > 0 →
  strip_length + strip_length - 6 = 30 →
  strip_length = 18 := by
sorry

end NUMINAMATH_CALUDE_paper_strip_length_l2773_277379


namespace NUMINAMATH_CALUDE_sum_of_squares_l2773_277331

theorem sum_of_squares (a b c : ℝ) 
  (sum_condition : a + b + c = 5)
  (product_sum_condition : a * b + b * c + a * c = 5) :
  a^2 + b^2 + c^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2773_277331


namespace NUMINAMATH_CALUDE_total_sugar_third_layer_is_correct_l2773_277392

/-- The amount of sugar needed for the smallest layer of the cake -/
def smallest_layer_sugar : ℝ := 2

/-- The size multiplier for the second layer compared to the first -/
def second_layer_multiplier : ℝ := 1.5

/-- The size multiplier for the third layer compared to the second -/
def third_layer_multiplier : ℝ := 2.5

/-- The percentage of sugar loss while baking each layer -/
def sugar_loss_percentage : ℝ := 0.15

/-- Calculates the total cups of sugar needed for the third layer -/
def total_sugar_third_layer : ℝ :=
  smallest_layer_sugar * second_layer_multiplier * third_layer_multiplier * (1 + sugar_loss_percentage)

/-- Theorem stating that the total sugar needed for the third layer is 8.625 cups -/
theorem total_sugar_third_layer_is_correct :
  total_sugar_third_layer = 8.625 := by
  sorry

end NUMINAMATH_CALUDE_total_sugar_third_layer_is_correct_l2773_277392


namespace NUMINAMATH_CALUDE_cost_reduction_proof_l2773_277336

theorem cost_reduction_proof (x : ℝ) : 
  (x ≥ 0) →  -- Ensure x is non-negative
  (x ≤ 1) →  -- Ensure x is at most 100%
  ((1 - x)^2 = 1 - 0.36) →
  x = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_cost_reduction_proof_l2773_277336


namespace NUMINAMATH_CALUDE_percentage_females_with_glasses_l2773_277355

def total_population : ℕ := 5000
def male_population : ℕ := 2000
def females_with_glasses : ℕ := 900

theorem percentage_females_with_glasses :
  (females_with_glasses : ℚ) / ((total_population - male_population) : ℚ) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_females_with_glasses_l2773_277355


namespace NUMINAMATH_CALUDE_real_root_of_cubic_l2773_277378

def cubic_polynomial (c d x : ℝ) : ℝ := c * x^3 + 4 * x^2 + d * x - 78

theorem real_root_of_cubic (c d : ℝ) :
  (∃ (z : ℂ), z = -3 - 4*I ∧ cubic_polynomial c d z.re = 0) →
  ∃ (x : ℝ), cubic_polynomial c d x = 0 ∧ x = -3 :=
sorry

end NUMINAMATH_CALUDE_real_root_of_cubic_l2773_277378


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2773_277349

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt (4 * x + 6) / Real.sqrt (8 * x + 2) = 2 / Real.sqrt 3) → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2773_277349


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l2773_277313

theorem termite_ridden_not_collapsing (total_homes : ℕ) (termite_ridden : ℕ) (collapsing : ℕ) :
  termite_ridden = total_homes / 3 →
  collapsing = termite_ridden / 4 →
  (termite_ridden - collapsing) = total_homes / 4 :=
by sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l2773_277313


namespace NUMINAMATH_CALUDE_odd_cube_plus_three_square_minus_linear_minus_three_divisible_by_48_l2773_277384

theorem odd_cube_plus_three_square_minus_linear_minus_three_divisible_by_48 (x : ℤ) (h : ∃ k : ℤ, x = 2*k + 1) :
  ∃ m : ℤ, x^3 + 3*x^2 - x - 3 = 48*m := by
sorry

end NUMINAMATH_CALUDE_odd_cube_plus_three_square_minus_linear_minus_three_divisible_by_48_l2773_277384


namespace NUMINAMATH_CALUDE_k_squared_upper_bound_l2773_277304

theorem k_squared_upper_bound (k n : ℕ) (h1 : 121 < k^2) (h2 : k^2 < n) 
  (h3 : ∀ m : ℕ, 121 < m^2 → m^2 < n → m ≤ k + 5) : n ≤ 324 :=
sorry

end NUMINAMATH_CALUDE_k_squared_upper_bound_l2773_277304


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_six_l2773_277300

theorem largest_three_digit_divisible_by_six :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 6 = 0 → n ≤ 996 ∧ 996 % 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_six_l2773_277300


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2773_277369

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (B < D) ∧
    (5 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) = (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    A = 4 ∧ B = 7 ∧ C = -3 ∧ D = 13 ∧ E = 1 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2773_277369


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l2773_277306

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The number of terms in the sequence -/
def sequence_length (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

theorem arithmetic_sequence_remainder (a₁ d aₙ : ℕ) (h₁ : a₁ = 2) (h₂ : d = 6) (h₃ : aₙ = 278) :
  (arithmetic_sum a₁ d (sequence_length a₁ d aₙ)) % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l2773_277306


namespace NUMINAMATH_CALUDE_sequence_minimum_term_minimum_term_value_minimum_term_occurs_at_24_l2773_277346

theorem sequence_minimum_term (n : ℕ) (h1 : 7 ≤ n) (h2 : n ≤ 95) :
  (Real.sqrt (n / 6) + Real.sqrt (96 / n : ℝ)) ≥ 4 :=
by sorry

theorem minimum_term_value (n : ℕ) (h1 : 7 ≤ n) (h2 : n ≤ 95) :
  (Real.sqrt (24 / 6) + Real.sqrt (96 / 24 : ℝ)) = 4 :=
by sorry

theorem minimum_term_occurs_at_24 :
  ∃ (n : ℕ), 7 ≤ n ∧ n ≤ 95 ∧
  (Real.sqrt (n / 6) + Real.sqrt (96 / n : ℝ)) = 4 ∧
  n = 24 :=
by sorry

end NUMINAMATH_CALUDE_sequence_minimum_term_minimum_term_value_minimum_term_occurs_at_24_l2773_277346


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l2773_277339

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the relation for a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (a b : Line) (α : Plane) 
  (h1 : parallel_line_plane a α) 
  (h2 : parallel_lines a b) 
  (h3 : ¬ contained_in b α) : 
  parallel_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l2773_277339


namespace NUMINAMATH_CALUDE_circle_in_square_l2773_277317

theorem circle_in_square (r : ℝ) (h : r = 6) :
  let square_side := 2 * r
  let square_area := square_side ^ 2
  let smaller_square_side := square_side - 2
  let smaller_square_area := smaller_square_side ^ 2
  (square_area = 144 ∧ square_area - smaller_square_area = 44) := by
  sorry

end NUMINAMATH_CALUDE_circle_in_square_l2773_277317


namespace NUMINAMATH_CALUDE_product_decreasing_implies_inequality_l2773_277333

theorem product_decreasing_implies_inequality
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, (deriv f x) * g x + f x * (deriv g x) < 0)
  (a b x : ℝ)
  (h_x : a < x ∧ x < b) :
  f x * g x > f b * g b :=
sorry

end NUMINAMATH_CALUDE_product_decreasing_implies_inequality_l2773_277333


namespace NUMINAMATH_CALUDE_building_houses_200_people_l2773_277340

/-- Calculates the number of people housed in a building given the number of stories,
    apartments per floor, and people per apartment. -/
def people_in_building (stories : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  stories * apartments_per_floor * people_per_apartment

/-- Theorem stating that a 25-story building with 4 apartments per floor and 2 people
    per apartment houses 200 people. -/
theorem building_houses_200_people :
  people_in_building 25 4 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_building_houses_200_people_l2773_277340


namespace NUMINAMATH_CALUDE_optimal_decomposition_2008_l2773_277332

theorem optimal_decomposition_2008 (decomp : List Nat) :
  (decomp.sum = 2008) →
  (decomp.prod ≤ (List.replicate 668 3 ++ List.replicate 2 2).prod) :=
by sorry

end NUMINAMATH_CALUDE_optimal_decomposition_2008_l2773_277332


namespace NUMINAMATH_CALUDE_shaded_angle_is_fifteen_degrees_l2773_277311

/-- A configuration of three identical isosceles triangles in a square -/
structure TrianglesInSquare where
  /-- The measure of the angle where three triangles meet at a corner of the square -/
  corner_angle : ℝ
  /-- The measure of each of the two equal angles in each isosceles triangle -/
  isosceles_angle : ℝ
  /-- Axiom: The corner angle is formed by three equal parts -/
  corner_angle_eq : corner_angle = 90 / 3
  /-- Axiom: The sum of angles in each isosceles triangle is 180° -/
  triangle_sum : corner_angle + 2 * isosceles_angle = 180

/-- The theorem to be proved -/
theorem shaded_angle_is_fifteen_degrees (t : TrianglesInSquare) :
  90 - t.isosceles_angle = 15 := by
  sorry

end NUMINAMATH_CALUDE_shaded_angle_is_fifteen_degrees_l2773_277311


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2773_277314

theorem trigonometric_identity (α : ℝ) (h : Real.sin α + Real.cos α = 2/3) :
  (2 * Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α) / (1 + Real.tan α) = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2773_277314


namespace NUMINAMATH_CALUDE_specialSquaresTheorem_l2773_277341

/-- Function to check if a number contains the digits 0 or 5 -/
def containsZeroOrFive (n : ℕ) : Bool :=
  sorry

/-- Function to delete the second digit of a number -/
def deleteSecondDigit (n : ℕ) : ℕ :=
  sorry

/-- The set of perfect squares satisfying the given conditions -/
def specialSquares : Finset ℕ :=
  sorry

theorem specialSquaresTheorem : specialSquares = {16, 36, 121, 484} := by
  sorry

end NUMINAMATH_CALUDE_specialSquaresTheorem_l2773_277341


namespace NUMINAMATH_CALUDE_min_boxes_for_treat_bags_l2773_277308

/-- Represents the number of items in each box -/
structure BoxSizes where
  chocolate : Nat
  mint : Nat
  caramel : Nat

/-- Represents the number of boxes of each item -/
structure Boxes where
  chocolate : Nat
  mint : Nat
  caramel : Nat

/-- Calculates the total number of boxes -/
def totalBoxes (b : Boxes) : Nat :=
  b.chocolate + b.mint + b.caramel

/-- Checks if the given number of boxes results in complete treat bags with no leftovers -/
def isValidDistribution (sizes : BoxSizes) (boxes : Boxes) : Prop :=
  sizes.chocolate * boxes.chocolate = sizes.mint * boxes.mint ∧
  sizes.chocolate * boxes.chocolate = sizes.caramel * boxes.caramel

/-- The main theorem stating the minimum number of boxes needed -/
theorem min_boxes_for_treat_bags : ∃ (boxes : Boxes),
  let sizes : BoxSizes := ⟨50, 40, 25⟩
  isValidDistribution sizes boxes ∧ 
  totalBoxes boxes = 17 ∧
  (∀ (other : Boxes), isValidDistribution sizes other → totalBoxes other ≥ totalBoxes boxes) := by
  sorry

end NUMINAMATH_CALUDE_min_boxes_for_treat_bags_l2773_277308


namespace NUMINAMATH_CALUDE_range_of_a_l2773_277364

theorem range_of_a (x : ℝ) (h : x > 1) :
  ∃ (S : Set ℝ), S = {a : ℝ | a ≤ x + 1 / (x - 1)} ∧ 
  ∀ (ε : ℝ), ε > 0 → ∃ (a : ℝ), a ∈ S ∧ a > 3 - ε :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2773_277364


namespace NUMINAMATH_CALUDE_total_time_is_twelve_years_l2773_277338

def years_to_get_in_shape : ℕ := 2
def years_to_learn_climbing (y : ℕ) : ℕ := 2 * y
def number_of_mountains : ℕ := 7
def months_per_mountain : ℕ := 5
def months_to_learn_diving : ℕ := 13
def years_of_diving : ℕ := 2

def total_time : ℕ :=
  years_to_get_in_shape +
  years_to_learn_climbing years_to_get_in_shape +
  (number_of_mountains * months_per_mountain + months_to_learn_diving) / 12 +
  years_of_diving

theorem total_time_is_twelve_years :
  total_time = 12 := by sorry

end NUMINAMATH_CALUDE_total_time_is_twelve_years_l2773_277338


namespace NUMINAMATH_CALUDE_leo_weight_proof_l2773_277389

/-- Leo's current weight -/
def leo_weight : ℝ := 103.6

/-- Kendra's weight -/
def kendra_weight : ℝ := 68

/-- Jake's weight -/
def jake_weight : ℝ := kendra_weight + 30

theorem leo_weight_proof :
  -- Condition 1: If Leo gains 12 pounds, he will weigh 70% more than Kendra
  (leo_weight + 12 = 1.7 * kendra_weight) ∧
  -- Condition 2: The combined weight of Leo, Kendra, and Jake is 270 pounds
  (leo_weight + kendra_weight + jake_weight = 270) ∧
  -- Condition 3: Jake weighs 30 pounds more than Kendra
  (jake_weight = kendra_weight + 30) →
  -- Conclusion: Leo's current weight is 103.6 pounds
  leo_weight = 103.6 := by
sorry

end NUMINAMATH_CALUDE_leo_weight_proof_l2773_277389


namespace NUMINAMATH_CALUDE_quarter_difference_l2773_277397

/-- Represents the number and value of coins in Sally's savings jar. -/
structure CoinJar where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  total_coins : ℕ
  total_value : ℕ

/-- Checks if a CoinJar configuration is valid according to the problem constraints. -/
def is_valid_jar (jar : CoinJar) : Prop :=
  jar.total_coins = 150 ∧
  jar.total_value = 2000 ∧
  jar.total_coins = jar.nickels + jar.dimes + jar.quarters ∧
  jar.total_value = 5 * jar.nickels + 10 * jar.dimes + 25 * jar.quarters

/-- Finds the maximum number of quarters possible in a valid CoinJar. -/
def max_quarters (jar : CoinJar) : ℕ := sorry

/-- Finds the minimum number of quarters possible in a valid CoinJar. -/
def min_quarters (jar : CoinJar) : ℕ := sorry

/-- Theorem stating the difference between max and min quarters is 62. -/
theorem quarter_difference (jar : CoinJar) (h : is_valid_jar jar) :
  max_quarters jar - min_quarters jar = 62 := by sorry

end NUMINAMATH_CALUDE_quarter_difference_l2773_277397


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_l2773_277319

theorem smallest_sum_of_factors (a b : ℕ+) (h : a * b = 240) :
  a + b ≥ 31 ∧ ∃ (x y : ℕ+), x * y = 240 ∧ x + y = 31 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_l2773_277319


namespace NUMINAMATH_CALUDE_five_cubic_yards_equals_135_cubic_feet_l2773_277315

/-- Converts cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet (yards : ℝ) : ℝ :=
  yards * 27

theorem five_cubic_yards_equals_135_cubic_feet :
  cubic_yards_to_cubic_feet 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_five_cubic_yards_equals_135_cubic_feet_l2773_277315


namespace NUMINAMATH_CALUDE_sum_of_even_integers_ranges_l2773_277334

def S1 : ℕ := (100 / 2) * (2 + 200)

def S2 : ℕ := (150 / 2) * (102 + 400)

theorem sum_of_even_integers_ranges (R : ℕ) : R = S1 + S2 → R = 47750 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_ranges_l2773_277334


namespace NUMINAMATH_CALUDE_product_of_x_and_y_l2773_277394

theorem product_of_x_and_y (x y : ℝ) : 
  (-3 * x + 4 * y = 28) → (3 * x - 2 * y = 8) → x * y = 264 :=
by
  sorry


end NUMINAMATH_CALUDE_product_of_x_and_y_l2773_277394


namespace NUMINAMATH_CALUDE_kyle_lifting_improvement_l2773_277395

theorem kyle_lifting_improvement (current_capacity : ℕ) (ratio : ℕ) : 
  current_capacity = 80 ∧ ratio = 3 → 
  current_capacity - (current_capacity / ratio) = 53 := by
sorry

end NUMINAMATH_CALUDE_kyle_lifting_improvement_l2773_277395


namespace NUMINAMATH_CALUDE_max_remainder_two_digit_div_sum_digits_l2773_277302

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The theorem stating that the maximum remainder when dividing a two-digit number
    by the sum of its digits is 15 -/
theorem max_remainder_two_digit_div_sum_digits :
  ∃ (n : ℕ), TwoDigitNumber n ∧
    ∀ (m : ℕ), TwoDigitNumber m →
      n % (sumOfDigits n) ≥ m % (sumOfDigits m) ∧
      n % (sumOfDigits n) = 15 :=
sorry

end NUMINAMATH_CALUDE_max_remainder_two_digit_div_sum_digits_l2773_277302


namespace NUMINAMATH_CALUDE_complex_multiplication_l2773_277373

theorem complex_multiplication (i : ℂ) : i * i = -1 → (3 + i) * (1 - 2*i) = 5 - 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2773_277373


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sixths_l2773_277385

theorem smallest_fraction_greater_than_five_sixths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (a : ℚ) / b > 5 / 6 →
    81 / 97 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sixths_l2773_277385


namespace NUMINAMATH_CALUDE_emily_small_gardens_l2773_277387

def number_of_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

theorem emily_small_gardens :
  number_of_small_gardens 42 36 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_emily_small_gardens_l2773_277387


namespace NUMINAMATH_CALUDE_rectangle_area_l2773_277348

theorem rectangle_area (short_side : ℝ) (perimeter : ℝ) 
  (h1 : short_side = 11) 
  (h2 : perimeter = 52) : 
  short_side * (perimeter / 2 - short_side) = 165 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2773_277348


namespace NUMINAMATH_CALUDE_last_digit_389_quaternary_l2773_277382

def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem last_digit_389_quaternary :
  (decimal_to_quaternary 389).getLast? = some 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_389_quaternary_l2773_277382


namespace NUMINAMATH_CALUDE_min_value_expression_l2773_277372

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 27 * b^3 + 64 * c^3 + 27 / (8 * a * b * c) ≥ 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2773_277372


namespace NUMINAMATH_CALUDE_solve_chicken_problem_l2773_277354

def chicken_problem (chicken_cost total_spent potato_cost : ℕ) : Prop :=
  chicken_cost > 0 ∧
  total_spent > potato_cost ∧
  (total_spent - potato_cost) % chicken_cost = 0 ∧
  (total_spent - potato_cost) / chicken_cost = 3

theorem solve_chicken_problem :
  chicken_problem 3 15 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_chicken_problem_l2773_277354


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2773_277335

theorem inequality_and_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a^2 + 1 / b^2 + 8 * a * b ≥ 8 ∧
  (1 / a^2 + 1 / b^2 + 8 * a * b = 8 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2773_277335


namespace NUMINAMATH_CALUDE_difference_of_unit_vectors_with_sum_magnitude_one_l2773_277301

/-- Given two unit vectors a and b in a real inner product space such that
    the magnitude of their sum is 1, prove that the magnitude of their
    difference is √3. -/
theorem difference_of_unit_vectors_with_sum_magnitude_one
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hab : ‖a + b‖ = 1) :
  ‖a - b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_unit_vectors_with_sum_magnitude_one_l2773_277301


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2773_277307

theorem trigonometric_identity (x : ℝ) :
  Real.cos (4 * x) * Real.cos (π + 2 * x) - Real.sin (2 * x) * Real.cos (π / 2 - 4 * x) = 
  Real.sqrt 2 / 2 * Real.sin (4 * x) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2773_277307


namespace NUMINAMATH_CALUDE_like_terms_exponent_difference_l2773_277343

theorem like_terms_exponent_difference (m n : ℤ) : 
  (∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ 4 * x^(2*m+2) * y^(n-1) = -3 * x^(3*m+1) * y^(3*n-5)) → 
  m - n = -1 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_difference_l2773_277343


namespace NUMINAMATH_CALUDE_smallest_multiple_l2773_277357

theorem smallest_multiple (x : ℕ) : x = 32 ↔ 
  (x > 0 ∧ 
   1152 ∣ (900 * x) ∧ 
   ∀ y : ℕ, (y > 0 ∧ y < x) → ¬(1152 ∣ (900 * y))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2773_277357


namespace NUMINAMATH_CALUDE_three_prime_divisors_of_nine_power_minus_one_l2773_277367

theorem three_prime_divisors_of_nine_power_minus_one (n : ℕ) (x : ℕ) 
  (h1 : x = 9^n - 1)
  (h2 : (Nat.factors x).toFinset.card = 3)
  (h3 : 7 ∈ Nat.factors x) :
  x = 728 := by sorry

end NUMINAMATH_CALUDE_three_prime_divisors_of_nine_power_minus_one_l2773_277367


namespace NUMINAMATH_CALUDE_card_count_proof_l2773_277383

/-- The ratio of Xiao Ming's counting speed to Xiao Hua's -/
def speed_ratio : ℚ := 6 / 4

/-- The number of cards Xiao Hua counted before forgetting -/
def forgot_count : ℕ := 48

/-- The number of cards Xiao Hua counted after starting over -/
def final_count : ℕ := 112

/-- The number of cards left in the box after Xiao Hua's final count -/
def remaining_cards : ℕ := 1

/-- The original number of cards in the box -/
def original_cards : ℕ := 353

theorem card_count_proof :
  (speed_ratio * forgot_count).num.toNat + final_count + remaining_cards = original_cards :=
sorry

end NUMINAMATH_CALUDE_card_count_proof_l2773_277383


namespace NUMINAMATH_CALUDE_characterize_superinvariant_sets_l2773_277344

/-- A set S is superinvariant if for any stretching A, there exists a translation B
    such that the images of S under A and B agree -/
def IsSuperinvariant (S : Set ℝ) : Prop :=
  ∀ (x₀ a : ℝ) (ha : a > 0),
    ∃ (b : ℝ),
      (∀ x ∈ S, ∃ y ∈ S, x₀ + a * (x - x₀) = y + b) ∧
      (∀ t ∈ S, ∃ u ∈ S, t + b = x₀ + a * (u - x₀))

/-- The set of all superinvariant subsets of ℝ -/
def SuperinvariantSets : Set (Set ℝ) :=
  {S | IsSuperinvariant S}

theorem characterize_superinvariant_sets :
  SuperinvariantSets =
    {∅} ∪ {Set.univ} ∪ {{p} | p : ℝ} ∪ {Set.univ \ {p} | p : ℝ} ∪
    {Set.Ioi p | p : ℝ} ∪ {Set.Ici p | p : ℝ} ∪
    {Set.Iio p | p : ℝ} ∪ {Set.Iic p | p : ℝ} :=
  sorry

#check characterize_superinvariant_sets

end NUMINAMATH_CALUDE_characterize_superinvariant_sets_l2773_277344


namespace NUMINAMATH_CALUDE_stream_speed_is_three_l2773_277325

/-- Represents the scenario of a rower traveling upstream and downstream -/
structure RiverJourney where
  distance : ℝ
  normalSpeedDiff : ℝ
  tripleSpeedDiff : ℝ

/-- Calculates the stream speed given a RiverJourney -/
def calculateStreamSpeed (journey : RiverJourney) : ℝ :=
  sorry

/-- Theorem stating that the stream speed is 3 for the given conditions -/
theorem stream_speed_is_three (journey : RiverJourney)
  (h1 : journey.distance = 21)
  (h2 : journey.normalSpeedDiff = 4)
  (h3 : journey.tripleSpeedDiff = 0.5) :
  calculateStreamSpeed journey = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_is_three_l2773_277325


namespace NUMINAMATH_CALUDE_lattice_point_in_diagonal_pentagon_l2773_277356

/-- A point in the 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A pentagon defined by five points -/
structure Pentagon where
  a : Point
  b : Point
  c : Point
  d : Point
  e : Point

/-- Check if a pentagon is convex -/
def is_convex (p : Pentagon) : Prop := sorry

/-- Check if a point is inside or on the boundary of a polygon defined by a list of points -/
def is_inside_or_on_boundary (point : Point) (polygon : List Point) : Prop := sorry

/-- The pentagon formed by the diagonals of the given pentagon -/
def diagonal_pentagon (p : Pentagon) : List Point := sorry

theorem lattice_point_in_diagonal_pentagon (p : Pentagon) 
  (h_convex : is_convex p) : 
  ∃ (point : Point), is_inside_or_on_boundary point (diagonal_pentagon p) := by
  sorry

end NUMINAMATH_CALUDE_lattice_point_in_diagonal_pentagon_l2773_277356


namespace NUMINAMATH_CALUDE_expression_value_l2773_277342

theorem expression_value : (-1/2)^2023 * 2^2024 = -2 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2773_277342


namespace NUMINAMATH_CALUDE_dans_to_barrys_dimes_ratio_l2773_277399

/-- The ratio of Dan's initial dimes to Barry's dimes -/
theorem dans_to_barrys_dimes_ratio :
  let barry_dimes : ℕ := 1000 / 10
  let dan_final_dimes : ℕ := 52
  let dan_initial_dimes : ℕ := dan_final_dimes - 2
  (dan_initial_dimes : ℚ) / barry_dimes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_dans_to_barrys_dimes_ratio_l2773_277399


namespace NUMINAMATH_CALUDE_sum_digits_12_4_less_than_32_l2773_277320

/-- The sum of digits of a number n in base b -/
def sum_of_digits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Theorem stating that for all bases greater than 10, the sum of digits of 12^4 is less than 2^5 -/
theorem sum_digits_12_4_less_than_32 (b : ℕ) (h : b > 10) : 
  sum_of_digits (12^4) b < 2^5 := by sorry

end NUMINAMATH_CALUDE_sum_digits_12_4_less_than_32_l2773_277320


namespace NUMINAMATH_CALUDE_sentences_started_today_l2773_277388

/-- Calculates the number of sentences Janice started with today given her typing speed and work schedule. -/
theorem sentences_started_today (
  typing_speed : ℕ)  -- Sentences typed per minute
  (initial_typing_time : ℕ)  -- Minutes typed before break
  (extra_typing_time : ℕ)  -- Additional minutes typed after break
  (erased_sentences : ℕ)  -- Number of sentences erased due to errors
  (final_typing_time : ℕ)  -- Minutes typed after meeting
  (total_sentences : ℕ)  -- Total sentences in the paper by end of day
  (h1 : typing_speed = 6)
  (h2 : initial_typing_time = 20)
  (h3 : extra_typing_time = 15)
  (h4 : erased_sentences = 40)
  (h5 : final_typing_time = 18)
  (h6 : total_sentences = 536)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_sentences_started_today_l2773_277388


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2773_277362

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(14 ∣ (427398 - y))) ∧ 
  (14 ∣ (427398 - x)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2773_277362


namespace NUMINAMATH_CALUDE_walkway_area_is_416_l2773_277361

/-- Represents the garden layout and calculates the walkway area -/
def garden_walkway_area (rows : Nat) (cols : Nat) (bed_length : Nat) (bed_width : Nat) (walkway_width : Nat) : Nat :=
  let total_width := cols * bed_length + (cols + 1) * walkway_width
  let total_length := rows * bed_width + (rows + 1) * walkway_width
  let total_area := total_width * total_length
  let bed_area := rows * cols * bed_length * bed_width
  total_area - bed_area

/-- Theorem stating that the walkway area for the given garden configuration is 416 square feet -/
theorem walkway_area_is_416 :
  garden_walkway_area 4 3 8 3 2 = 416 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_416_l2773_277361


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l2773_277375

theorem quadratic_inequality_no_solution : 
  {x : ℝ | x^2 - 2*x + 3 < 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l2773_277375


namespace NUMINAMATH_CALUDE_irrational_sum_two_l2773_277396

theorem irrational_sum_two : ∃ (a b : ℝ), Irrational a ∧ Irrational b ∧ a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_irrational_sum_two_l2773_277396


namespace NUMINAMATH_CALUDE_cube_surface_area_from_prism_l2773_277326

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_from_prism (l w h : ℝ) (h1 : l = 8) (h2 : w = 2) (h3 : h = 32) :
  let prism_volume := l * w * h
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = 384 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_prism_l2773_277326


namespace NUMINAMATH_CALUDE_probability_sum_10_three_dice_l2773_277322

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The sum we're looking for -/
def targetSum : ℕ := 10

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (sum of 10) -/
def favorableOutcomes : ℕ := 27

/-- The probability of rolling a sum of 10 with three standard six-sided dice -/
theorem probability_sum_10_three_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_10_three_dice_l2773_277322


namespace NUMINAMATH_CALUDE_jade_tower_solution_l2773_277347

/-- The number of Lego pieces in Jade's tower problem -/
def jade_tower_problem (width_per_level : ℕ) (num_levels : ℕ) (pieces_left : ℕ) : Prop :=
  width_per_level * num_levels + pieces_left = 100

/-- Theorem stating the solution to Jade's Lego tower problem -/
theorem jade_tower_solution : jade_tower_problem 7 11 23 := by
  sorry

end NUMINAMATH_CALUDE_jade_tower_solution_l2773_277347


namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l2773_277337

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Payment per task in dollars -/
def payment_per_task : ℚ := 6/5

/-- Days worked per week -/
def days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * payment_per_task * days_per_week

theorem tim_weekly_earnings :
  weekly_earnings = 720 := by sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l2773_277337
