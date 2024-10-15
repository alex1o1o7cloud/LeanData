import Mathlib

namespace NUMINAMATH_CALUDE_max_area_right_triangle_l3994_399412

/-- The maximum area of a right-angled triangle with perimeter 2 is 3 - 2√2 -/
theorem max_area_right_triangle (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_perimeter : a + b + c = 2) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  (1/2) * a * b ≤ 3 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_l3994_399412


namespace NUMINAMATH_CALUDE_additional_flowers_grown_l3994_399430

theorem additional_flowers_grown 
  (initial_flowers : ℕ) 
  (dead_flowers : ℕ) 
  (final_flowers : ℕ) : 
  final_flowers > initial_flowers → 
  final_flowers - initial_flowers = 
    final_flowers - initial_flowers + dead_flowers - dead_flowers :=
by
  sorry

#check additional_flowers_grown

end NUMINAMATH_CALUDE_additional_flowers_grown_l3994_399430


namespace NUMINAMATH_CALUDE_exactly_two_rigid_motions_l3994_399437

/-- Represents a point on a plane --/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line on a plane --/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents the pattern on the line --/
inductive Pattern
  | Triangle
  | Square

/-- Represents a rigid motion transformation --/
inductive RigidMotion
  | Rotation (center : Point) (angle : ℝ)
  | Translation (dx : ℝ) (dy : ℝ)
  | ReflectionLine (l : Line)
  | ReflectionPerp (p : Point)

/-- The line with the pattern --/
def patternLine : Line := sorry

/-- The sequence of shapes along the line --/
def patternSequence : ℕ → Pattern := sorry

/-- Checks if a rigid motion preserves the pattern --/
def preservesPattern (rm : RigidMotion) : Prop := sorry

/-- The theorem to be proved --/
theorem exactly_two_rigid_motions :
  ∃! (s : Finset RigidMotion),
    s.card = 2 ∧ 
    (∀ rm ∈ s, preservesPattern rm) ∧
    (∀ rm, preservesPattern rm → rm ∈ s ∨ rm = RigidMotion.Translation 0 0) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_rigid_motions_l3994_399437


namespace NUMINAMATH_CALUDE_sum_of_roots_l3994_399447

theorem sum_of_roots (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 72 = 0)
  (hq : 10*q^3 - 75*q^2 + 50*q - 625 = 0) : 
  p + q = 2*(180^(1/3 : ℝ)) + 43/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3994_399447


namespace NUMINAMATH_CALUDE_ball_attendance_l3994_399463

theorem ball_attendance :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendance_l3994_399463


namespace NUMINAMATH_CALUDE_S_infinite_l3994_399425

/-- Sum of positive integer divisors of n -/
def d (n : ℕ) : ℕ := sorry

/-- Euler's totient function: count of integers in [0,n] coprime with n -/
def φ (n : ℕ) : ℕ := sorry

/-- The set of integers n for which d(n) * φ(n) is a perfect square -/
def S : Set ℕ := {n : ℕ | ∃ k : ℕ, d n * φ n = k^2}

/-- The main theorem: S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_S_infinite_l3994_399425


namespace NUMINAMATH_CALUDE_original_paint_intensity_l3994_399457

theorem original_paint_intensity 
  (original_fraction : Real) 
  (replacement_intensity : Real) 
  (new_intensity : Real) 
  (replaced_fraction : Real) :
  original_fraction = 0.5 →
  replacement_intensity = 0.2 →
  new_intensity = 0.15 →
  replaced_fraction = 0.5 →
  (1 - replaced_fraction) * original_fraction + replaced_fraction * replacement_intensity = new_intensity →
  original_fraction = 0.1 := by
sorry

end NUMINAMATH_CALUDE_original_paint_intensity_l3994_399457


namespace NUMINAMATH_CALUDE_quilt_cost_calculation_l3994_399499

/-- The cost of a rectangular quilt -/
def quilt_cost (length width price_per_sq_ft : ℝ) : ℝ :=
  length * width * price_per_sq_ft

/-- Theorem: The cost of a 12 ft by 15 ft quilt at $70 per square foot is $12,600 -/
theorem quilt_cost_calculation :
  quilt_cost 12 15 70 = 12600 := by
  sorry

end NUMINAMATH_CALUDE_quilt_cost_calculation_l3994_399499


namespace NUMINAMATH_CALUDE_general_equation_l3994_399498

theorem general_equation (n : ℕ+) :
  (n + 1 : ℚ) / ((n + 1)^2 - 1) - 1 / (n * (n + 1) * (n + 2)) = 1 / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_general_equation_l3994_399498


namespace NUMINAMATH_CALUDE_equality_in_different_bases_l3994_399472

theorem equality_in_different_bases : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 
  (3 * a^2 + 4 * a + 2 : ℕ) = (9 * b + 7 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_equality_in_different_bases_l3994_399472


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l3994_399439

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : Prop := k * x^2 + x - 3 = 0

-- Define the condition for distinct real roots
def has_distinct_real_roots (k : ℝ) : Prop := k > -1/12 ∧ k ≠ 0

-- Define the condition for the roots
def roots_condition (x₁ x₂ : ℝ) : Prop := (x₁ + x₂)^2 + x₁ * x₂ = 4

-- Theorem statement
theorem quadratic_equation_k_value :
  ∀ k : ℝ, 
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    quadratic_equation k x₁ ∧ 
    quadratic_equation k x₂ ∧
    has_distinct_real_roots k ∧
    roots_condition x₁ x₂) →
  k = 1/4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l3994_399439


namespace NUMINAMATH_CALUDE_marble_draw_probability_l3994_399459

/-- The probability of drawing a white marble first and a red marble second from a bag 
    containing 5 red marbles and 7 white marbles, without replacement. -/
theorem marble_draw_probability :
  let total_marbles : ℕ := 5 + 7
  let red_marbles : ℕ := 5
  let white_marbles : ℕ := 7
  let prob_white_first : ℚ := white_marbles / total_marbles
  let prob_red_second : ℚ := red_marbles / (total_marbles - 1)
  prob_white_first * prob_red_second = 35 / 132 :=
by sorry

end NUMINAMATH_CALUDE_marble_draw_probability_l3994_399459


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3994_399448

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: A rectangle with length three times its width and perimeter 160 has an area of 1200 -/
theorem rectangle_area_theorem (r : Rectangle) 
  (h1 : r.length = 3 * r.width) 
  (h2 : perimeter r = 160) : 
  area r = 1200 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3994_399448


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l3994_399451

theorem greatest_integer_radius (A : ℝ) (h : A < 90 * Real.pi) :
  ∃ (r : ℕ), r^2 * Real.pi = A ∧ ∀ (n : ℕ), n^2 * Real.pi < 90 * Real.pi → n ≤ r ∧ r ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l3994_399451


namespace NUMINAMATH_CALUDE_probability_three_defective_shipment_l3994_399413

/-- The probability of selecting three defective smartphones from a shipment --/
def probability_three_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total *
  ((defective - 1) : ℚ) / (total - 1) *
  ((defective - 2) : ℚ) / (total - 2)

/-- Theorem stating the approximate probability of selecting three defective smartphones --/
theorem probability_three_defective_shipment :
  let total := 500
  let defective := 85
  abs (probability_three_defective total defective - 0.0047) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_defective_shipment_l3994_399413


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l3994_399407

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sum_of_divisors 180 ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ sum_of_divisors 180 → q ≤ p ∧ p = 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l3994_399407


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l3994_399450

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_equals_set : 
  (A ∩ B)ᶜ = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l3994_399450


namespace NUMINAMATH_CALUDE_cricket_player_innings_l3994_399417

theorem cricket_player_innings 
  (average : ℝ) 
  (next_innings_runs : ℝ) 
  (average_increase : ℝ) 
  (h1 : average = 33) 
  (h2 : next_innings_runs = 77) 
  (h3 : average_increase = 4) :
  ∃ n : ℕ, 
    (n : ℝ) * average + next_innings_runs = (n + 1) * (average + average_increase) ∧ 
    n = 10 :=
by sorry

end NUMINAMATH_CALUDE_cricket_player_innings_l3994_399417


namespace NUMINAMATH_CALUDE_absolute_value_trigonometry_and_reciprocal_quadratic_equation_solution_l3994_399420

-- Problem 1
theorem absolute_value_trigonometry_and_reciprocal :
  |(-3)| - 4 * Real.sin (π / 6) + (1 / 3)⁻¹ = 4 := by sorry

-- Problem 2
theorem quadratic_equation_solution :
  ∀ x : ℝ, 2 * x - 6 = x^2 - 9 ↔ x = -1 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_trigonometry_and_reciprocal_quadratic_equation_solution_l3994_399420


namespace NUMINAMATH_CALUDE_barium_oxide_weight_l3994_399433

/-- The atomic weight of Barium in g/mol -/
def barium_weight : ℝ := 137.33

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (ba_count o_count : ℕ) : ℝ :=
  ba_count * barium_weight + o_count * oxygen_weight

/-- Theorem: The molecular weight of a compound with 1 Barium and 1 Oxygen atom is 153.33 g/mol -/
theorem barium_oxide_weight : molecular_weight 1 1 = 153.33 := by
  sorry

end NUMINAMATH_CALUDE_barium_oxide_weight_l3994_399433


namespace NUMINAMATH_CALUDE_ratio_bounds_l3994_399465

theorem ratio_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  2 / 3 ≤ b / a ∧ b / a ≤ 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_bounds_l3994_399465


namespace NUMINAMATH_CALUDE_one_hundred_twenty_fifth_number_with_digit_sum_5_l3994_399471

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth natural number whose digits sum to 5 -/
def nth_number_with_digit_sum_5 (n : ℕ) : ℕ := sorry

/-- The main theorem stating that the 125th number with digit sum 5 is 41000 -/
theorem one_hundred_twenty_fifth_number_with_digit_sum_5 :
  nth_number_with_digit_sum_5 125 = 41000 := by sorry

end NUMINAMATH_CALUDE_one_hundred_twenty_fifth_number_with_digit_sum_5_l3994_399471


namespace NUMINAMATH_CALUDE_difference_of_squares_l3994_399442

theorem difference_of_squares (x y : ℝ) : (x + 2*y) * (x - 2*y) = x^2 - 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3994_399442


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3994_399410

/-- The curve function f(x) = x³ - 3x² + 2x --/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, (k * x₀ = f x₀) ∧ (∀ x : ℝ, k * x ≤ f x) ∧ (k = f' x₀)) →
  (k = 2 ∨ k = -1/4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3994_399410


namespace NUMINAMATH_CALUDE_triangle_inequality_condition_unique_k_value_l3994_399452

theorem triangle_inequality_condition (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  (6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) →
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

theorem unique_k_value :
  ∀ k : ℕ, k > 0 →
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
    a + b > c ∧ b + c > a ∧ c + a > b) →
  k = 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_condition_unique_k_value_l3994_399452


namespace NUMINAMATH_CALUDE_number_of_different_products_l3994_399477

def set_a : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}
def set_b : Finset ℕ := {2, 4, 6, 19, 21, 24, 27, 31, 35}

theorem number_of_different_products : 
  (Finset.card (set_a.powersetCard 2) * Finset.card set_b) = 405 := by
  sorry

end NUMINAMATH_CALUDE_number_of_different_products_l3994_399477


namespace NUMINAMATH_CALUDE_square_difference_equality_l3994_399402

theorem square_difference_equality : 1004^2 - 998^2 - 1002^2 + 1000^2 = 8008 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3994_399402


namespace NUMINAMATH_CALUDE_average_speed_inequality_l3994_399435

theorem average_speed_inequality (a b v : ℝ) (hab : a < b) (hv : v = (2 * a * b) / (a + b)) : 
  a < v ∧ v < Real.sqrt (a * b) := by sorry

end NUMINAMATH_CALUDE_average_speed_inequality_l3994_399435


namespace NUMINAMATH_CALUDE_fifth_over_eight_fourth_power_l3994_399415

theorem fifth_over_eight_fourth_power : (5 / 8 : ℚ) ^ 4 = 625 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_fifth_over_eight_fourth_power_l3994_399415


namespace NUMINAMATH_CALUDE_package_weight_l3994_399481

theorem package_weight (total_weight : ℝ) (first_butcher_packages : ℕ) (second_butcher_packages : ℕ) (third_butcher_packages : ℕ) 
  (h1 : total_weight = 100)
  (h2 : first_butcher_packages = 10)
  (h3 : second_butcher_packages = 7)
  (h4 : third_butcher_packages = 8) :
  ∃ (package_weight : ℝ), 
    package_weight * (first_butcher_packages + second_butcher_packages + third_butcher_packages) = total_weight ∧ 
    package_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_package_weight_l3994_399481


namespace NUMINAMATH_CALUDE_product_of_roots_l3994_399404

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 26 → ∃ y : ℝ, (x + 3) * (x - 4) = 26 ∧ (y + 3) * (y - 4) = 26 ∧ x * y = -38 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3994_399404


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l3994_399408

/-- The probability of picking two red balls from a bag with 4 red, 3 blue, and 2 green balls is 1/6 -/
theorem probability_two_red_balls (total_balls : Nat) (red_balls : Nat) (blue_balls : Nat) (green_balls : Nat)
  (h1 : total_balls = red_balls + blue_balls + green_balls)
  (h2 : red_balls = 4)
  (h3 : blue_balls = 3)
  (h4 : green_balls = 2) :
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l3994_399408


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3994_399431

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  let l1 : Line := { a := 2, b := -1, c := -1 }  -- 2x - y - 1 = 0
  let l2 : Line := { a := 2, b := -1, c := 0 }   -- 2x - y = 0
  parallel l1 l2 ∧ point_on_line 1 2 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3994_399431


namespace NUMINAMATH_CALUDE_geometric_progression_in_floor_sqrt2003_l3994_399455

/-- For any positive integers k and m greater than 1, there exists a subsequence
of {⌊n√2003⌋} (n ≥ 1) that forms a geometric progression with m terms and ratio k. -/
theorem geometric_progression_in_floor_sqrt2003 (k m : ℕ) (hk : k > 1) (hm : m > 1) :
  ∃ (n : ℕ), ∀ (i : ℕ), i < m →
    (⌊(k^i * n : ℝ) * Real.sqrt 2003⌋ : ℤ) = k^i * ⌊(n : ℝ) * Real.sqrt 2003⌋ :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_in_floor_sqrt2003_l3994_399455


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3994_399416

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ),
  (5 * x^2 - 45 * x - 500 = a * (x + b)^2 + c) ∧ (a + b + c = -605.75) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3994_399416


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l3994_399418

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 5 / Real.log 30 + 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l3994_399418


namespace NUMINAMATH_CALUDE_cube_diagonal_l3994_399494

theorem cube_diagonal (s : ℝ) (h1 : 6 * s^2 = 54) (h2 : 12 * s = 36) :
  ∃ d : ℝ, d = 3 * Real.sqrt 3 ∧ d^2 = 3 * s^2 := by
  sorry

#check cube_diagonal

end NUMINAMATH_CALUDE_cube_diagonal_l3994_399494


namespace NUMINAMATH_CALUDE_handshake_theorem_l3994_399474

def corporate_event (n : ℕ) (completed_handshakes : ℕ) : Prop :=
  let total_handshakes := n * (n - 1) / 2
  total_handshakes - completed_handshakes = 42

theorem handshake_theorem :
  corporate_event 10 3 := by
  sorry

end NUMINAMATH_CALUDE_handshake_theorem_l3994_399474


namespace NUMINAMATH_CALUDE_N_is_perfect_square_l3994_399496

/-- Constructs the number N with n ones and n+1 twos, ending with 5 -/
def constructN (n : ℕ) : ℕ :=
  (10^(2*n+2) + 10^(n+2) + 25) / 9

/-- Theorem stating that N is a perfect square for any natural number n -/
theorem N_is_perfect_square (n : ℕ) : ∃ m : ℕ, (constructN n) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_N_is_perfect_square_l3994_399496


namespace NUMINAMATH_CALUDE_sunlight_rice_yield_is_correlation_l3994_399478

-- Define the concept of a relationship
structure Relationship (X Y : Type) where
  relates : X → Y → Prop

-- Define what it means for a relationship to be functional
def IsFunctional {X Y : Type} (r : Relationship X Y) : Prop :=
  ∀ x : X, ∃! y : Y, r.relates x y

-- Define what it means for a relationship to be a correlation
def IsCorrelation {X Y : Type} (r : Relationship X Y) : Prop :=
  (¬ IsFunctional r) ∧ 
  (∃ pattern : X → Y → Prop, ∀ x : X, ∃ y : Y, pattern x y ∧ r.relates x y) ∧
  (∃ x₁ x₂ : X, ∃ y₁ y₂ : Y, r.relates x₁ y₁ ∧ r.relates x₂ y₂ ∧ x₁ ≠ x₂ ∧ y₁ ≠ y₂)

-- Define the relationship between sunlight and rice yield
def SunlightRiceYield : Relationship ℝ ℝ :=
  { relates := λ sunlight yield => yield > 0 ∧ ∃ k > 0, yield ≤ k * sunlight }

-- State the theorem
theorem sunlight_rice_yield_is_correlation :
  IsCorrelation SunlightRiceYield :=
sorry

end NUMINAMATH_CALUDE_sunlight_rice_yield_is_correlation_l3994_399478


namespace NUMINAMATH_CALUDE_player_b_always_wins_l3994_399400

/-- Represents a player's move in the game -/
structure Move where
  value : ℕ

/-- Represents the state of the game after each round -/
structure GameState where
  round : ℕ
  player_a_move : Move
  player_b_move : Move
  player_a_score : ℕ
  player_b_score : ℕ

/-- The game setup with n rounds and increment d -/
structure GameSetup where
  n : ℕ
  d : ℕ
  h1 : n > 1
  h2 : d ≥ 1

/-- A strategy for player B -/
def PlayerBStrategy (setup : GameSetup) : GameState → Move := sorry

/-- Checks if a move is valid according to the game rules -/
def isValidMove (setup : GameSetup) (prev : GameState) (curr : Move) : Prop := sorry

/-- Calculates the score for a round -/
def calculateScore (a_move : Move) (b_move : Move) : ℕ × ℕ := sorry

/-- Simulates the game for n rounds -/
def playGame (setup : GameSetup) (strategy : GameState → Move) : GameState := sorry

/-- Theorem: Player B always has a winning strategy -/
theorem player_b_always_wins (setup : GameSetup) :
  ∃ (strategy : GameState → Move),
    (playGame setup strategy).player_b_score ≥ (playGame setup strategy).player_a_score := by
  sorry

end NUMINAMATH_CALUDE_player_b_always_wins_l3994_399400


namespace NUMINAMATH_CALUDE_set_intersection_equality_l3994_399419

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2^x - 1)}

-- Define set B
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

-- Define the intersection set
def intersection_set : Set ℝ := {x | 0 ≤ x ∧ x < 2}

-- Theorem statement
theorem set_intersection_equality : A ∩ B = intersection_set := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l3994_399419


namespace NUMINAMATH_CALUDE_max_value_and_k_range_l3994_399461

def f (x : ℝ) : ℝ := -3 * x^2 - 3 * x + 18

theorem max_value_and_k_range :
  (∀ x > -1, (f x - 21) / (x + 1) ≤ -3) ∧
  (∀ k : ℝ, (∀ x ∈ Set.Ioo 1 4, -3 * x^2 + k * x - 5 > 0) → k < 2 * Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_max_value_and_k_range_l3994_399461


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3994_399460

/-- The area of the region within a rectangle of dimensions 5 by 6 units,
    but outside three semicircles with radii 2, 3, and 2.5 units, 
    is equal to 30 - 14.625π square units. -/
theorem shaded_area_calculation : 
  let rectangle_area : ℝ := 5 * 6
  let semicircle_area (r : ℝ) : ℝ := (1/2) * Real.pi * r^2
  let total_semicircle_area : ℝ := semicircle_area 2 + semicircle_area 3 + semicircle_area 2.5
  rectangle_area - total_semicircle_area = 30 - 14.625 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3994_399460


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l3994_399426

-- Part 1
theorem factorization_1 (x y : ℝ) : -x^5 * y^3 + x^3 * y^5 = -x^3 * y^3 * (x + y) * (x - y) := by sorry

-- Part 2
theorem factorization_2 (a : ℝ) : (a^2 + 1)^2 - 4 * a^2 = (a + 1)^2 * (a - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l3994_399426


namespace NUMINAMATH_CALUDE_farm_horse_food_calculation_l3994_399403

/-- Calculates the total amount of horse food needed daily on a farm -/
theorem farm_horse_food_calculation (sheep_count : ℕ) (sheep_to_horse_ratio : ℚ) (food_per_horse : ℕ) : 
  sheep_count = 48 →
  sheep_to_horse_ratio = 6 / 7 →
  food_per_horse = 230 →
  (sheep_count / sheep_to_horse_ratio : ℚ).num * food_per_horse = 12880 := by
  sorry

end NUMINAMATH_CALUDE_farm_horse_food_calculation_l3994_399403


namespace NUMINAMATH_CALUDE_fixed_point_coordinates_l3994_399406

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation y - 2 = k(x + 1) -/
def lineEquation (k : ℝ) (p : Point) : Prop :=
  p.y - 2 = k * (p.x + 1)

/-- The fixed point M satisfies the line equation for all k -/
def isFixedPoint (M : Point) : Prop :=
  ∀ k : ℝ, lineEquation k M

theorem fixed_point_coordinates :
  ∀ M : Point, isFixedPoint M → M = Point.mk (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_coordinates_l3994_399406


namespace NUMINAMATH_CALUDE_addition_and_multiplication_of_integers_l3994_399464

theorem addition_and_multiplication_of_integers : 
  (-3 + 2 = -1) ∧ ((-3) * 2 = -6) := by sorry

end NUMINAMATH_CALUDE_addition_and_multiplication_of_integers_l3994_399464


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l3994_399421

/-- Proves that the ratio of Somu's age to his father's age is 1:3 given the conditions -/
theorem somu_father_age_ratio :
  ∀ (somu_age father_age : ℕ),
  somu_age = 18 →
  somu_age - 9 = (father_age - 9) / 5 →
  ∃ (k : ℕ), k > 0 ∧ somu_age * 3 = father_age * k ∧ k = 1 :=
by sorry

end NUMINAMATH_CALUDE_somu_father_age_ratio_l3994_399421


namespace NUMINAMATH_CALUDE_three_independent_events_probability_l3994_399411

/-- Given three independent events with equal probability, 
    prove that the probability of all three events occurring simultaneously 
    is the cube of the individual probability -/
theorem three_independent_events_probability 
  (p : ℝ) 
  (h1 : 0 ≤ p) 
  (h2 : p ≤ 1) 
  (h3 : p = 1/3) : 
  p * p * p = 1/27 := by
  sorry

end NUMINAMATH_CALUDE_three_independent_events_probability_l3994_399411


namespace NUMINAMATH_CALUDE_estate_value_l3994_399462

/-- Represents the estate distribution problem --/
def EstateDistribution (total : ℝ) : Prop :=
  ∃ (elder_niece younger_niece brother caretaker : ℝ),
    -- The two nieces together received half of the estate
    elder_niece + younger_niece = total / 2 ∧
    -- The nieces' shares are in the ratio of 3 to 2
    elder_niece = (3/5) * (total / 2) ∧
    younger_niece = (2/5) * (total / 2) ∧
    -- The brother got three times as much as the elder niece
    brother = 3 * elder_niece ∧
    -- The caretaker received $800
    caretaker = 800 ∧
    -- The sum of all shares equals the total estate
    elder_niece + younger_niece + brother + caretaker = total

/-- Theorem stating that the estate value is $2000 --/
theorem estate_value : EstateDistribution 2000 :=
sorry

end NUMINAMATH_CALUDE_estate_value_l3994_399462


namespace NUMINAMATH_CALUDE_salary_approximation_l3994_399438

/-- The salary of a man who spends specific fractions on expenses and has a remainder --/
def salary (food_fraction : ℚ) (rent_fraction : ℚ) (clothes_fraction : ℚ) (remainder : ℚ) : ℚ :=
  remainder / (1 - food_fraction - rent_fraction - clothes_fraction)

/-- Theorem stating the approximate salary of a man with given expenses and remainder --/
theorem salary_approximation :
  let s := salary (1/3) (1/4) (1/5) 1760
  ⌊s⌋ = 8123 := by sorry

end NUMINAMATH_CALUDE_salary_approximation_l3994_399438


namespace NUMINAMATH_CALUDE_number_equality_l3994_399432

theorem number_equality (x : ℚ) (h : (30 / 100) * x = (40 / 100) * 50) : x = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l3994_399432


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3994_399482

theorem quadratic_expression_value : 
  let x : ℝ := 2
  2 * x^2 - 3 * x + 4 = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3994_399482


namespace NUMINAMATH_CALUDE_square_digit_sum_99999_l3994_399446

/-- Given a natural number n, returns the sum of its digits -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a number consists of all nines -/
def is_all_nines (n : ℕ) : Prop := sorry

theorem square_digit_sum_99999 (n : ℕ) :
  n = 99999 → is_all_nines n → sum_of_digits (n^2) = 45 := by sorry

end NUMINAMATH_CALUDE_square_digit_sum_99999_l3994_399446


namespace NUMINAMATH_CALUDE_fencing_cost_is_5300_l3994_399484

/-- A rectangular plot with given dimensions and fencing cost -/
structure Plot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ
  length_breadth_difference : length = breadth + 60
  length_value : length = 80

/-- Calculate the total cost of fencing for a given plot -/
def total_fencing_cost (p : Plot) : ℝ :=
  2 * (p.length + p.breadth) * p.fencing_cost_per_meter

/-- Theorem: The total fencing cost for the given plot is 5300 currency units -/
theorem fencing_cost_is_5300 (p : Plot) (h : p.fencing_cost_per_meter = 26.50) : 
  total_fencing_cost p = 5300 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_5300_l3994_399484


namespace NUMINAMATH_CALUDE_equation_solution_l3994_399444

theorem equation_solution (x : ℚ) : 
  x ≠ 2/3 →
  ((7*x + 3) / (3*x^2 + 7*x - 6) = 3*x / (3*x - 2)) ↔ (x = 1/3 ∨ x = -3) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3994_399444


namespace NUMINAMATH_CALUDE_exists_minimum_top_number_l3994_399454

/-- Represents a square pyramid of blocks -/
structure SquarePyramid where
  base : Matrix (Fin 4) (Fin 4) ℕ
  layer2 : Matrix (Fin 3) (Fin 3) ℕ
  layer3 : Matrix (Fin 2) (Fin 2) ℕ
  top : ℕ

/-- Checks if the pyramid is valid according to the given conditions -/
def isValidPyramid (p : SquarePyramid) : Prop :=
  (∀ i j, p.base i j ∈ Finset.range 17) ∧
  (∀ i j, p.layer2 i j = p.base (i+1) (j+1) + p.base (i+1) j + p.base i (j+1)) ∧
  (∀ i j, p.layer3 i j = p.layer2 (i+1) (j+1) + p.layer2 (i+1) j + p.layer2 i (j+1)) ∧
  (p.top = p.layer3 1 1 + p.layer3 1 0 + p.layer3 0 1)

/-- The main theorem statement -/
theorem exists_minimum_top_number :
  ∃ (min : ℕ), ∀ (p : SquarePyramid), isValidPyramid p → p.top ≥ min :=
sorry


end NUMINAMATH_CALUDE_exists_minimum_top_number_l3994_399454


namespace NUMINAMATH_CALUDE_waiter_tables_l3994_399414

theorem waiter_tables (total_customers : ℕ) (people_per_table : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) :
  total_customers = 90 →
  people_per_table = women_per_table + men_per_table →
  women_per_table = 7 →
  men_per_table = 3 →
  total_customers / people_per_table = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l3994_399414


namespace NUMINAMATH_CALUDE_modulus_sum_complex_numbers_l3994_399495

theorem modulus_sum_complex_numbers : 
  Complex.abs ((3 : ℂ) - 8*I + (4 : ℂ) + 6*I) = Real.sqrt 53 := by sorry

end NUMINAMATH_CALUDE_modulus_sum_complex_numbers_l3994_399495


namespace NUMINAMATH_CALUDE_fraction_between_l3994_399443

theorem fraction_between (p q : ℕ+) (h1 : (6 : ℚ) / 11 < p / q) (h2 : p / q < (5 : ℚ) / 9) 
  (h3 : ∀ (r s : ℕ+), (6 : ℚ) / 11 < r / s → r / s < (5 : ℚ) / 9 → s ≥ q) : 
  p + q = 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_between_l3994_399443


namespace NUMINAMATH_CALUDE_perpendicular_lines_to_plane_are_parallel_l3994_399436

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_to_plane_are_parallel
  (α β γ : Plane) (m n : Line)
  (h_distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h_distinct_lines : m ≠ n)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_α : perpendicular n α) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_to_plane_are_parallel_l3994_399436


namespace NUMINAMATH_CALUDE_tree_distance_l3994_399497

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  let distance_between (i j : ℕ) := d * (j - i : ℝ) / 4
  distance_between 1 n = 175 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l3994_399497


namespace NUMINAMATH_CALUDE_factorization_left_to_right_l3994_399468

/-- Factorization from left to right for x^2 - 1 -/
theorem factorization_left_to_right :
  ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_left_to_right_l3994_399468


namespace NUMINAMATH_CALUDE_shooter_score_problem_l3994_399423

/-- A shooter's competition score problem -/
theorem shooter_score_problem 
  (first_six_shots : ℕ) 
  (record : ℕ) 
  (h1 : first_six_shots = 52) 
  (h2 : record = 89) 
  (h3 : ∀ shot, shot ∈ Set.Icc 1 10) :
  /- (1) Minimum score on 7th shot to break record -/
  (∃ x : ℕ, x ≥ 8 ∧ first_six_shots + x + 30 > record) ∧
  /- (2) Number of 10s needed in last 3 shots if 7th shot is 8 -/
  (first_six_shots + 8 + 30 > record) ∧
  /- (3) Necessity of at least one 10 in last 3 shots if 7th shot is 10 -/
  (∃ x y z : ℕ, x ∈ Set.Icc 1 10 ∧ y ∈ Set.Icc 1 10 ∧ z ∈ Set.Icc 1 10 ∧
    first_six_shots + 10 + x + y + z > record ∧ (x = 10 ∨ y = 10 ∨ z = 10)) := by
  sorry


end NUMINAMATH_CALUDE_shooter_score_problem_l3994_399423


namespace NUMINAMATH_CALUDE_max_min_product_l3994_399466

theorem max_min_product (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (sum_eq : p + q + r = 13) (sum_prod_eq : p * q + q * r + r * p = 30) :
  ∃ (n : ℝ), n = min (p * q) (min (q * r) (r * p)) ∧ n ≤ 10 ∧
  ∀ (m : ℝ), m = min (p * q) (min (q * r) (r * p)) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l3994_399466


namespace NUMINAMATH_CALUDE_eight_mile_taxi_ride_cost_l3994_399427

/-- Calculates the cost of a taxi ride given the base fare, cost per mile, and total miles traveled. -/
def taxiRideCost (baseFare : ℝ) (costPerMile : ℝ) (miles : ℝ) : ℝ :=
  baseFare + costPerMile * miles

/-- Theorem stating that an 8-mile taxi ride with a $2.00 base fare and $0.30 per mile costs $4.40. -/
theorem eight_mile_taxi_ride_cost :
  taxiRideCost 2.00 0.30 8 = 4.40 := by
  sorry

end NUMINAMATH_CALUDE_eight_mile_taxi_ride_cost_l3994_399427


namespace NUMINAMATH_CALUDE_miltons_zoology_books_l3994_399422

theorem miltons_zoology_books :
  ∀ (z b : ℕ), b = 4 * z → z + b = 80 → z = 16 := by sorry

end NUMINAMATH_CALUDE_miltons_zoology_books_l3994_399422


namespace NUMINAMATH_CALUDE_logical_consequences_l3994_399453

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (passed : Student → Prop)
variable (scored_above_90_percent : Student → Prop)

-- Define the given condition
variable (h : ∀ s : Student, scored_above_90_percent s → passed s)

-- Theorem to prove
theorem logical_consequences :
  (∀ s : Student, ¬(passed s) → ¬(scored_above_90_percent s)) ∧
  (∀ s : Student, ¬(scored_above_90_percent s) → ¬(passed s)) ∧
  (∀ s : Student, passed s → scored_above_90_percent s) :=
by sorry

end NUMINAMATH_CALUDE_logical_consequences_l3994_399453


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3994_399487

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) ≥ x^2)) ↔ (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3994_399487


namespace NUMINAMATH_CALUDE_q_of_one_equals_zero_l3994_399486

/-- Given a function q: ℝ → ℝ, prove that q(1) = 0 -/
theorem q_of_one_equals_zero (q : ℝ → ℝ) 
  (h1 : (1, 0) ∈ Set.range (λ x => (x, q x))) 
  (h2 : ∃ n : ℤ, q 1 = n) : 
  q 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_q_of_one_equals_zero_l3994_399486


namespace NUMINAMATH_CALUDE_factors_720_l3994_399485

/-- The number of distinct positive factors of 720 -/
def num_factors_720 : ℕ := sorry

/-- 720 has exactly 30 distinct positive factors -/
theorem factors_720 : num_factors_720 = 30 := by sorry

end NUMINAMATH_CALUDE_factors_720_l3994_399485


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l3994_399441

/-- Given that the line x + y = c is a perpendicular bisector of the line segment from (2,5) to (8,11), prove that c = 13 -/
theorem perpendicular_bisector_c_value :
  ∀ c : ℝ,
  (∀ x y : ℝ, x + y = c ↔ (x - 5)^2 + (y - 8)^2 = (5 - 2)^2 + (8 - 5)^2) →
  c = 13 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l3994_399441


namespace NUMINAMATH_CALUDE_function_transformation_l3994_399469

theorem function_transformation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2) :
  ∀ x : ℝ, f x = (x + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_function_transformation_l3994_399469


namespace NUMINAMATH_CALUDE_balloon_count_sum_l3994_399434

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := 6

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- Theorem stating that the sum of individual balloon counts equals the total -/
theorem balloon_count_sum :
  fred_balloons + sam_balloons + mary_balloons = total_balloons := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_sum_l3994_399434


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_with_incircle_l3994_399405

/-- A triangle with an incircle -/
structure TriangleWithIncircle where
  /-- The radius of the incircle -/
  r : ℝ
  /-- The length of the segment of the side divided by the tangent point -/
  a : ℝ
  /-- The length of the other segment of the side divided by the tangent point -/
  b : ℝ
  /-- The length of the longest side of the triangle -/
  longest_side : ℝ

/-- Theorem: In a triangle with an incircle of radius 5 units, where the incircle is tangent
    to one side at a point dividing it into segments of 9 and 5 units, the length of the
    longest side is 18 units. -/
theorem longest_side_of_triangle_with_incircle
  (t : TriangleWithIncircle)
  (h1 : t.r = 5)
  (h2 : t.a = 9)
  (h3 : t.b = 5) :
  t.longest_side = 18 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_with_incircle_l3994_399405


namespace NUMINAMATH_CALUDE_jake_reading_theorem_l3994_399401

def read_pattern (first_day : ℕ) : ℕ → ℕ
  | 1 => first_day
  | 2 => first_day - 20
  | 3 => 2 * (first_day - 20)
  | 4 => first_day / 2
  | _ => 0

def total_pages_read (first_day : ℕ) : ℕ :=
  (read_pattern first_day 1) + (read_pattern first_day 2) + 
  (read_pattern first_day 3) + (read_pattern first_day 4)

theorem jake_reading_theorem (book_chapters book_pages : ℕ) 
  (h1 : book_chapters = 8) (h2 : book_pages = 130) (h3 : read_pattern 37 1 = 37) :
  total_pages_read 37 = 106 := by sorry

end NUMINAMATH_CALUDE_jake_reading_theorem_l3994_399401


namespace NUMINAMATH_CALUDE_xy_negative_sufficient_not_necessary_l3994_399490

theorem xy_negative_sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x * y < 0 → |x - y| = |x| + |y|) ∧
  (∃ x y : ℝ, |x - y| = |x| + |y| ∧ x * y ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_xy_negative_sufficient_not_necessary_l3994_399490


namespace NUMINAMATH_CALUDE_square_of_two_plus_i_l3994_399424

theorem square_of_two_plus_i : (2 + Complex.I) ^ 2 = 3 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_plus_i_l3994_399424


namespace NUMINAMATH_CALUDE_unique_divisibility_l3994_399445

def is_divisible_by_only_one_small_prime (n : ℕ) : Prop :=
  ∃! p, p < 10 ∧ Nat.Prime p ∧ n % p = 0

def number_form (B : ℕ) : ℕ := 404300 + B

theorem unique_divisibility :
  ∃! B, B < 10 ∧ is_divisible_by_only_one_small_prime (number_form B) ∧ number_form B = 404304 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisibility_l3994_399445


namespace NUMINAMATH_CALUDE_fourth_number_proof_l3994_399456

theorem fourth_number_proof (numbers : Fin 6 → ℝ) 
  (avg_all : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 30)
  (avg_first_four : (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 25)
  (avg_last_three : (numbers 3 + numbers 4 + numbers 5) / 3 = 35) :
  numbers 3 = 25 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l3994_399456


namespace NUMINAMATH_CALUDE_system_solution_l3994_399409

theorem system_solution :
  ∃ (a b c d : ℝ),
    (a + c = -1) ∧
    (a * c + b + d = -1) ∧
    (a * d + b * c = -5) ∧
    (b * d = 6) ∧
    ((a = -3 ∧ b = 2 ∧ c = 2 ∧ d = 3) ∨
     (a = 2 ∧ b = 3 ∧ c = -3 ∧ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3994_399409


namespace NUMINAMATH_CALUDE_expected_value_is_eight_l3994_399467

/-- The number of marbles in the bag -/
def n : ℕ := 7

/-- The set of marble numbers -/
def marbles : Finset ℕ := Finset.range n

/-- The sum of all possible pairs of marbles -/
def sum_of_pairs : ℕ := (marbles.powerset.filter (fun s => s.card = 2)).sum (fun s => s.sum id)

/-- The number of ways to choose 2 marbles out of n -/
def num_combinations : ℕ := n.choose 2

/-- The expected value of the sum of two randomly drawn marbles -/
def expected_value : ℚ := (sum_of_pairs : ℚ) / num_combinations

theorem expected_value_is_eight : expected_value = 8 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_eight_l3994_399467


namespace NUMINAMATH_CALUDE_height_pillar_E_l3994_399473

/-- Regular octagon with pillars -/
structure OctagonWithPillars where
  /-- Side length of the octagon -/
  side_length : ℝ
  /-- Height of pillar at vertex A -/
  height_A : ℝ
  /-- Height of pillar at vertex B -/
  height_B : ℝ
  /-- Height of pillar at vertex C -/
  height_C : ℝ

/-- Theorem: Height of pillar at E in a regular octagon with given pillar heights -/
theorem height_pillar_E (octagon : OctagonWithPillars) 
  (h_A : octagon.height_A = 15)
  (h_B : octagon.height_B = 12)
  (h_C : octagon.height_C = 13) :
  ∃ (height_E : ℝ), height_E = 5 := by
  sorry

end NUMINAMATH_CALUDE_height_pillar_E_l3994_399473


namespace NUMINAMATH_CALUDE_largest_prime_mersenne_under_500_l3994_399429

def mersenne_number (n : ℕ) : ℕ := 2^n - 1

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem largest_prime_mersenne_under_500 :
  ∀ n : ℕ, is_power_of_two n → 
    mersenne_number n < 500 → 
    Nat.Prime (mersenne_number n) → 
    mersenne_number n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_mersenne_under_500_l3994_399429


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_difference_l3994_399480

theorem binomial_expansion_sum_difference (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + Real.sqrt 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_difference_l3994_399480


namespace NUMINAMATH_CALUDE_negative_and_absolute_value_l3994_399458

theorem negative_and_absolute_value : 
  (-(-4) = 4) ∧ (-|(-4)| = -4) := by sorry

end NUMINAMATH_CALUDE_negative_and_absolute_value_l3994_399458


namespace NUMINAMATH_CALUDE_notebook_problem_l3994_399483

def satisfies_notebook_conditions (n : ℕ) : Prop :=
  ∃ x y : ℕ,
    (y + 2 = n * (x - 2)) ∧
    (x + n = 2 * (y - n)) ∧
    x > 2 ∧ y > n

theorem notebook_problem :
  {n : ℕ | satisfies_notebook_conditions n} = {1, 2, 3, 8} :=
by sorry

end NUMINAMATH_CALUDE_notebook_problem_l3994_399483


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3994_399428

theorem express_y_in_terms_of_x (x y : ℝ) : 2 * x - y = 4 → y = 2 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3994_399428


namespace NUMINAMATH_CALUDE_rotten_apples_smell_percentage_l3994_399440

theorem rotten_apples_smell_percentage 
  (total_apples : ℕ) 
  (rotten_percentage : ℚ) 
  (non_smelling_rotten : ℕ) 
  (h1 : total_apples = 200)
  (h2 : rotten_percentage = 40 / 100)
  (h3 : non_smelling_rotten = 24) : 
  (total_apples * rotten_percentage - non_smelling_rotten : ℚ) / (total_apples * rotten_percentage) * 100 = 70 := by
sorry

end NUMINAMATH_CALUDE_rotten_apples_smell_percentage_l3994_399440


namespace NUMINAMATH_CALUDE_special_pair_characterization_l3994_399493

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Property of natural numbers a and b -/
def special_pair (a b : ℕ) : Prop :=
  (a^2 + 1) % b = 0 ∧ (b^2 + 1) % a = 0

/-- Main theorem -/
theorem special_pair_characterization (a b : ℕ) :
  special_pair a b → (a = 1 ∧ b = 1) ∨ (∃ n : ℕ, n ≥ 1 ∧ a = fib (2*n - 1) ∧ b = fib (2*n + 1)) :=
sorry

end NUMINAMATH_CALUDE_special_pair_characterization_l3994_399493


namespace NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l3994_399488

theorem log_sqrt10_1000sqrt10 :
  Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l3994_399488


namespace NUMINAMATH_CALUDE_largest_R_under_condition_l3994_399491

theorem largest_R_under_condition : ∃ (R : ℕ), R > 0 ∧ R^2000 < 5^3000 ∧ ∀ (S : ℕ), S > R → S^2000 ≥ 5^3000 :=
by sorry

end NUMINAMATH_CALUDE_largest_R_under_condition_l3994_399491


namespace NUMINAMATH_CALUDE_factors_of_34848_l3994_399489

/-- The number of positive factors of 34848 -/
def num_factors : ℕ := 54

/-- 34848 as a natural number -/
def n : ℕ := 34848

theorem factors_of_34848 : Nat.card (Nat.divisors n) = num_factors := by
  sorry

end NUMINAMATH_CALUDE_factors_of_34848_l3994_399489


namespace NUMINAMATH_CALUDE_rectangle_area_error_percent_l3994_399479

theorem rectangle_area_error_percent (L W : ℝ) (L' W' : ℝ) : 
  L' = L * (1 + 0.07) → 
  W' = W * (1 - 0.06) → 
  let A := L * W
  let A' := L' * W'
  (A' - A) / A * 100 = 0.58 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percent_l3994_399479


namespace NUMINAMATH_CALUDE_ellipse_problem_l3994_399476

-- Define the circles and curve C
def F₁ (r : ℝ) (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = r^2
def F₂ (r : ℝ) (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = (4 - r)^2
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point M
def M : ℝ × ℝ := (0, 1)

-- Define the orthogonality condition for points A and B
def orthogonal (A B : ℝ × ℝ) : Prop :=
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0

-- Theorem statement
theorem ellipse_problem (r : ℝ) (h : 0 < r ∧ r < 4) :
  -- 1. Equation of curve C
  (∀ x y : ℝ, (∃ r', F₁ r' x y ∧ F₂ r' x y) ↔ C x y) ∧
  -- 2. Line AB passes through fixed point
  (∀ A B : ℝ × ℝ, C A.1 A.2 → C B.1 B.2 → A ≠ B → orthogonal A B →
    ∃ t : ℝ, A.1 + t * (B.1 - A.1) = 0 ∧ A.2 + t * (B.2 - A.2) = -3/5) ∧
  -- 3. Maximum area of triangle ABM
  (∀ A B : ℝ × ℝ, C A.1 A.2 → C B.1 B.2 → A ≠ B → orthogonal A B →
    abs ((A.1 - M.1) * (B.2 - M.2) - (A.2 - M.2) * (B.1 - M.1)) / 2 ≤ 64/25) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_problem_l3994_399476


namespace NUMINAMATH_CALUDE_bills_tv_height_l3994_399492

-- Define the dimensions of the TVs
def bill_width : ℕ := 48
def bob_width : ℕ := 70
def bob_height : ℕ := 60

-- Define the weight per square inch
def weight_per_sq_inch : ℕ := 4

-- Define the weight difference in ounces
def weight_diff_oz : ℕ := 150 * 16

-- Theorem statement
theorem bills_tv_height :
  ∃ (h : ℕ),
    h * bill_width * weight_per_sq_inch =
    bob_width * bob_height * weight_per_sq_inch - weight_diff_oz ∧
    h = 75 := by
  sorry

end NUMINAMATH_CALUDE_bills_tv_height_l3994_399492


namespace NUMINAMATH_CALUDE_quadratic_shift_l3994_399449

/-- The original quadratic function -/
def g (x : ℝ) : ℝ := (x + 1)^2 + 3

/-- The transformed quadratic function -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating that f is the result of shifting g 2 units right and 1 unit down -/
theorem quadratic_shift (x : ℝ) : f x = g (x - 2) - 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_l3994_399449


namespace NUMINAMATH_CALUDE_intersection_and_subset_l3994_399475

def set_A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def set_B (a : ℝ) : Set ℝ := {x | 1 - a < x ∧ x ≤ 3*a + 1}

theorem intersection_and_subset :
  (∀ x : ℝ, x ∈ (set_A ∩ set_B 1) ↔ (0 < x ∧ x ≤ 3)) ∧
  (∀ a : ℝ, set_B a ⊆ set_A ↔ a ≤ 2/3) := by sorry

end NUMINAMATH_CALUDE_intersection_and_subset_l3994_399475


namespace NUMINAMATH_CALUDE_fraction_equality_l3994_399470

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (2 * a + b) / (a - 2 * b) = (2 * x + 1) / (x - 2) := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3994_399470
