import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_main_theorem_l2294_229447

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∃ q : ℝ, q > 0 ∧ (a 2 = a 1 * q ∧ a 3 = a 2 * q ∧ a 4 = a 3 * q ∧ a 5 = a 4 * q) :=
sorry

/-- The second, half of the third, and twice the first term form an arithmetic sequence -/
def ArithmeticSubsequence (a : ℕ → ℝ) : Prop :=
  a 2 - (1/2 * a 3) = (1/2 * a 3) - (2 * a 1)

theorem main_theorem (a : ℕ → ℝ) (h1 : GeometricSequence a) (h2 : ArithmeticSubsequence a) :
  (a 3 + a 4) / (a 4 + a 5) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_main_theorem_l2294_229447


namespace NUMINAMATH_CALUDE_percentage_problem_l2294_229490

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 40 = (5 / 100) * 60 + 23 → P = 65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2294_229490


namespace NUMINAMATH_CALUDE_solution_system_equations_l2294_229454

theorem solution_system_equations (x y : ℝ) :
  x ≠ 0 ∧
  |y - x| - |x| / x + 1 = 0 ∧
  |2 * x - y| + |x + y - 1| + |x - y| + y - 1 = 0 →
  y = x ∧ 0 < x ∧ x ≤ 0.5 :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l2294_229454


namespace NUMINAMATH_CALUDE_sine_rule_application_l2294_229434

theorem sine_rule_application (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 →
  a = 3 * b * Real.sin A →
  Real.sin B = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sine_rule_application_l2294_229434


namespace NUMINAMATH_CALUDE_prism_volume_approximation_l2294_229499

/-- Represents a right rectangular prism -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℝ := p.a * p.b * p.c

/-- The main theorem to prove -/
theorem prism_volume_approximation (p : RectangularPrism) 
  (h1 : p.a * p.b = 54)
  (h2 : p.b * p.c = 56)
  (h3 : p.a * p.c = 60) :
  round (volume p) = 426 := by
  sorry


end NUMINAMATH_CALUDE_prism_volume_approximation_l2294_229499


namespace NUMINAMATH_CALUDE_race_outcomes_l2294_229475

/-- The number of participants in the race -/
def num_participants : ℕ := 6

/-- The number of top positions we're considering -/
def top_positions : ℕ := 3

/-- Calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculate the number of permutations of n items -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of different 1st-2nd-3rd place outcomes in a race with 6 participants,
    where one specific participant is guaranteed to be in the top 3 and there are no ties -/
theorem race_outcomes : 
  top_positions * choose (num_participants - 1) (top_positions - 1) * permutations (top_positions - 1) = 60 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l2294_229475


namespace NUMINAMATH_CALUDE_remainder_problem_l2294_229477

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 7 = 1) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2294_229477


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l2294_229487

/-- A point on a parabola with a specific distance to its directrix -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2*x
  distance_to_directrix : x + 1/2 = 2

/-- The coordinates of the point are (3/2, ±√3) -/
theorem parabola_point_coordinates (p : ParabolaPoint) : 
  p.x = 3/2 ∧ (p.y = Real.sqrt 3 ∨ p.y = -Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l2294_229487


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_expansion_l2294_229404

theorem perfect_square_trinomial_expansion (x : ℝ) : 
  let a : ℝ := x
  let b : ℝ := (1 : ℝ) / 2
  2 * a * b = x := by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_expansion_l2294_229404


namespace NUMINAMATH_CALUDE_dime_count_l2294_229489

/-- Given a collection of coins consisting of dimes and nickels, 
    this theorem proves the number of dimes given the total number 
    of coins and their total value. -/
theorem dime_count 
  (total_coins : ℕ) 
  (total_value : ℚ) 
  (h_total : total_coins = 36) 
  (h_value : total_value = 31/10) : 
  ∃ (dimes nickels : ℕ),
    dimes + nickels = total_coins ∧ 
    (dimes : ℚ) / 10 + (nickels : ℚ) / 20 = total_value ∧
    dimes = 26 := by
  sorry

end NUMINAMATH_CALUDE_dime_count_l2294_229489


namespace NUMINAMATH_CALUDE_square_with_removed_triangles_l2294_229426

/-- Given a square with side length s, from which two pairs of identical isosceles right triangles
    are removed to form a rectangle, if the total area removed is 180 m², then the diagonal of the
    remaining rectangle is 18 m. -/
theorem square_with_removed_triangles (s : ℝ) (x y : ℝ) : 
  x ≥ 0 → y ≥ 0 → x + y = s → x^2 + y^2 = 180 → 
  Real.sqrt (2 * (x^2 + y^2)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_with_removed_triangles_l2294_229426


namespace NUMINAMATH_CALUDE_geometric_series_equality_l2294_229451

def C (n : ℕ) : ℚ := (1024 / 3) * (1 - 1 / (4 ^ n))

def D (n : ℕ) : ℚ := (2048 / 3) * (1 - 1 / ((-2) ^ n))

theorem geometric_series_equality (n : ℕ) (h : n ≥ 1) : 
  (C n = D n) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l2294_229451


namespace NUMINAMATH_CALUDE_exists_element_with_mass_percentage_l2294_229486

/-- Molar mass of Hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- Molar mass of Bromine in g/mol -/
def molar_mass_Br : ℝ := 79.90

/-- Molar mass of Oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Molar mass of HBrO3 in g/mol -/
def molar_mass_HBrO3 : ℝ := molar_mass_H + molar_mass_Br + 3 * molar_mass_O

/-- Mass percentage of a certain element in HBrO3 -/
def target_mass_percentage : ℝ := 0.78

theorem exists_element_with_mass_percentage :
  ∃ (element_mass : ℝ), 
    0 < element_mass ∧ 
    element_mass ≤ molar_mass_HBrO3 ∧
    (element_mass / molar_mass_HBrO3) * 100 = target_mass_percentage :=
by sorry

end NUMINAMATH_CALUDE_exists_element_with_mass_percentage_l2294_229486


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2294_229456

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 300 ∧ 
  5 * a = c - 14 ∧ 
  5 * a = b + 14 → 
  a * b * c = 664500 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2294_229456


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l2294_229498

/-- The area of the circle represented by the polar equation r = 4cosθ - 3sinθ -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := λ θ => 4 * Real.cos θ - 3 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, (r θ * Real.cos θ - center.1)^2 + (r θ * Real.sin θ - center.2)^2 = radius^2) ∧
    π * radius^2 = 25 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l2294_229498


namespace NUMINAMATH_CALUDE_x_y_relation_existence_of_k_l2294_229460

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 4
  | (n + 2) => 3 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 3 * y (n + 1) - y n

theorem x_y_relation (n : ℕ) : (x n)^2 - 5*(y n)^2 + 4 = 0 := by
  sorry

theorem existence_of_k (a b : ℕ) (h : a^2 - 5*b^2 + 4 = 0) :
  ∃ k : ℕ, x k = a ∧ y k = b := by
  sorry

end NUMINAMATH_CALUDE_x_y_relation_existence_of_k_l2294_229460


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2294_229494

theorem vector_sum_magnitude : 
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (0, 1)
  ‖a + 2 • b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2294_229494


namespace NUMINAMATH_CALUDE_fuel_cost_savings_l2294_229411

theorem fuel_cost_savings (old_efficiency : ℝ) (old_fuel_cost : ℝ) 
  (h1 : old_efficiency > 0) (h2 : old_fuel_cost > 0) : 
  let new_efficiency := 1.5 * old_efficiency
  let new_fuel_cost := 1.2 * old_fuel_cost
  let old_trip_cost := (1 / old_efficiency) * old_fuel_cost
  let new_trip_cost := (1 / new_efficiency) * new_fuel_cost
  (old_trip_cost - new_trip_cost) / old_trip_cost = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_savings_l2294_229411


namespace NUMINAMATH_CALUDE_a_1_value_c_is_arithmetic_l2294_229457

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

def sequence_c (n : ℕ) : ℝ := sorry

axiom sum_relation (n : ℕ) : sum_S n / 2 = sequence_a n - 2^n

axiom a_relation (n : ℕ) : sequence_a n = 2^n * sequence_c n

theorem a_1_value : sequence_a 1 = 4 := sorry

theorem c_is_arithmetic : ∃ (d : ℝ), ∀ (n : ℕ), n > 0 → sequence_c (n + 1) - sequence_c n = d := sorry

end NUMINAMATH_CALUDE_a_1_value_c_is_arithmetic_l2294_229457


namespace NUMINAMATH_CALUDE_triangle_vector_property_l2294_229410

theorem triangle_vector_property (A B C : ℝ) (hAcute : 0 < A ∧ A < π/2) 
    (hBcute : 0 < B ∧ B < π/2) (hCcute : 0 < C ∧ C < π/2) 
    (hSum : A + B + C = π) :
  let a : ℝ × ℝ := (Real.sin C + Real.cos C, 2 - 2 * Real.sin C)
  let b : ℝ × ℝ := (1 + Real.sin C, Real.sin C - Real.cos C)
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  2 * Real.sin A ^ 2 + Real.cos B = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_vector_property_l2294_229410


namespace NUMINAMATH_CALUDE_probability_three_primes_six_dice_l2294_229476

-- Define a 12-sided die
def die := Finset.range 12

-- Define prime numbers on a 12-sided die
def primes : Finset ℕ := {2, 3, 5, 7, 11}

-- Define the probability of rolling a prime number on one die
def prob_prime : ℚ := (primes.card : ℚ) / (die.card : ℚ)

-- Define the probability of not rolling a prime number on one die
def prob_not_prime : ℚ := 1 - prob_prime

-- Define the number of ways to choose 3 dice from 6
def choose_3_from_6 : ℕ := Nat.choose 6 3

-- Statement of the theorem
theorem probability_three_primes_six_dice :
  (choose_3_from_6 : ℚ) * prob_prime^3 * prob_not_prime^3 = 857500 / 2985984 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_primes_six_dice_l2294_229476


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2294_229413

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180

-- Define condition p
def condition_p (t : Triangle) : Prop :=
  t.A = 60 ∨ t.B = 60 ∨ t.C = 60

-- Define condition q
def condition_q (t : Triangle) : Prop :=
  t.A - t.B = t.B - t.C

-- Theorem stating p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ t : Triangle, condition_q t → condition_p t) ∧
  ¬(∀ t : Triangle, condition_p t → condition_q t) := by
  sorry


end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2294_229413


namespace NUMINAMATH_CALUDE_square_of_product_divided_by_square_l2294_229402

theorem square_of_product_divided_by_square (m n : ℝ) :
  (2 * m * n)^2 / n^2 = 4 * m^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_divided_by_square_l2294_229402


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2294_229415

universe u

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Finset ℕ := {2, 4, 5}

theorem complement_of_A_in_U :
  (U \ A : Finset ℕ) = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2294_229415


namespace NUMINAMATH_CALUDE_team_selection_theorem_l2294_229437

def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8
def team_size : ℕ := 5

def select_team_with_restrictions (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem team_selection_theorem :
  (select_team_with_restrictions (internal_medicine_doctors + surgeons - 2) (team_size - 1) = 3060) ∧
  (Nat.choose (internal_medicine_doctors + surgeons) team_size - 
   Nat.choose internal_medicine_doctors team_size - 
   Nat.choose surgeons team_size = 14656) :=
by sorry

end NUMINAMATH_CALUDE_team_selection_theorem_l2294_229437


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l2294_229443

/-- Given a rectangle formed from a rope of length 100 cm with shorter sides of 22 cm each,
    prove that the length of each longer side is 28 cm. -/
theorem rectangle_longer_side_length (total_length : ℝ) (short_side : ℝ) (long_side : ℝ) :
  total_length = 100 ∧ short_side = 22 →
  2 * short_side + 2 * long_side = total_length →
  long_side = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l2294_229443


namespace NUMINAMATH_CALUDE_x_squared_eq_5_is_quadratic_l2294_229448

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 5 -/
def f (x : ℝ) : ℝ := x^2 - 5

theorem x_squared_eq_5_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_x_squared_eq_5_is_quadratic_l2294_229448


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l2294_229464

/-- A line parallel to y = -3x + 6 passing through (3, -2) has y-intercept 7 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b y = -3 * x + b 0) →  -- b is a linear function with slope -3
  b (-2) = 3 →                   -- b passes through (3, -2)
  b 0 = 7 :=                     -- y-intercept of b is 7
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l2294_229464


namespace NUMINAMATH_CALUDE_intersecting_line_slope_angle_l2294_229428

/-- A line passing through (2,0) and intersecting y = √(2-x^2) -/
structure IntersectingLine where
  k : ℝ
  intersects_curve : ∃ (x y : ℝ), y = k * (x - 2) ∧ y = Real.sqrt (2 - x^2)

/-- The area of triangle AOB formed by the intersecting line -/
def triangleArea (l : IntersectingLine) : ℝ := sorry

/-- The slope angle of the line -/
def slopeAngle (l : IntersectingLine) : ℝ := sorry

theorem intersecting_line_slope_angle 
  (l : IntersectingLine) 
  (h : triangleArea l = 1) : 
  slopeAngle l = 150 * π / 180 := by sorry

end NUMINAMATH_CALUDE_intersecting_line_slope_angle_l2294_229428


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l2294_229452

/-- The minimum distance from a point on the circle ρ = 2 to the line ρ(cos(θ) + √3 sin(θ)) = 6 is 1 -/
theorem min_distance_circle_to_line : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let line := {p : ℝ × ℝ | p.1 + Real.sqrt 3 * p.2 = 6}
  ∃ (d : ℝ), d = 1 ∧ ∀ (p : ℝ × ℝ), p ∈ circle → 
    (∀ (q : ℝ × ℝ), q ∈ line → d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l2294_229452


namespace NUMINAMATH_CALUDE_four_point_ratio_l2294_229436

/-- Given four distinct points on a plane with segment lengths a, a, a, 2a, 2a, and b,
    prove that the ratio of b to a is 2√2 -/
theorem four_point_ratio (a b : ℝ) (h : a > 0) :
  ∃ (A B C D : ℝ × ℝ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    ({dist A B, dist A C, dist A D, dist B C, dist B D, dist C D} : Finset ℝ) =
      {a, a, a, 2*a, 2*a, b} →
    b / a = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_four_point_ratio_l2294_229436


namespace NUMINAMATH_CALUDE_negative_one_odd_power_l2294_229462

theorem negative_one_odd_power (n : ℕ) (h : Odd n) : (-1 : ℤ) ^ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_odd_power_l2294_229462


namespace NUMINAMATH_CALUDE_mean_equality_problem_l2294_229453

theorem mean_equality_problem (x : ℚ) : 
  (8 + 10 + 22) / 3 = (15 + x) / 2 → x = 35 / 3 :=
by sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l2294_229453


namespace NUMINAMATH_CALUDE_special_sequence_properties_l2294_229433

/-- A sequence satisfying certain conditions -/
structure SpecialSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  p : ℝ
  h1 : a 1 = 2
  h2 : ∀ n, a n ≠ 0
  h3 : ∀ n, a n * a (n + 1) = p * S n + 2
  h4 : ∀ n, S (n + 1) = S n + a (n + 1)

/-- The main theorem about the special sequence -/
theorem special_sequence_properties (seq : SpecialSequence) :
  (∀ n, seq.a (n + 2) - seq.a n = seq.p) ∧
  (∃ p : ℝ, p = 2 ∧ 
    (∃ d : ℝ, ∀ n, |seq.a (n + 1)| - |seq.a n| = d)) :=
sorry

end NUMINAMATH_CALUDE_special_sequence_properties_l2294_229433


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l2294_229493

/-- Proves that given a 204-gallon tank filled partially with fuel A (12% ethanol)
    and then to capacity with fuel B (16% ethanol), if the full tank contains
    30 gallons of ethanol, then the volume of fuel A added is 66 gallons. -/
theorem fuel_mixture_problem (x : ℝ) : 
  (0.12 * x + 0.16 * (204 - x) = 30) → x = 66 := by
  sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l2294_229493


namespace NUMINAMATH_CALUDE_mark_young_fish_count_l2294_229432

/-- The number of tanks Mark has for pregnant fish -/
def num_tanks : ℕ := 3

/-- The number of pregnant fish in each tank -/
def fish_per_tank : ℕ := 4

/-- The number of young fish each pregnant fish gives birth to -/
def young_per_fish : ℕ := 20

/-- The total number of young fish Mark has at the end -/
def total_young_fish : ℕ := num_tanks * fish_per_tank * young_per_fish

theorem mark_young_fish_count : total_young_fish = 240 := by
  sorry

end NUMINAMATH_CALUDE_mark_young_fish_count_l2294_229432


namespace NUMINAMATH_CALUDE_fraction_calculation_l2294_229416

theorem fraction_calculation : 
  (2 * 4.6 * 9 + 4 * 9.2 * 18) / (1 * 2.3 * 4.5 + 3 * 6.9 * 13.5) = 18/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2294_229416


namespace NUMINAMATH_CALUDE_mode_is_nine_l2294_229495

def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def frequency : Nat → Nat
| 0 => 8
| 1 => 8
| 2 => 12
| 3 => 11
| 4 => 10
| 5 => 8
| 6 => 9
| 7 => 8
| 8 => 12
| 9 => 14
| _ => 0

def is_mode (x : Nat) : Prop :=
  x ∈ digits ∧ ∀ y ∈ digits, frequency x ≥ frequency y

theorem mode_is_nine : is_mode 9 := by
  sorry

end NUMINAMATH_CALUDE_mode_is_nine_l2294_229495


namespace NUMINAMATH_CALUDE_length_of_AB_l2294_229430

-- Define the points
variable (A B C D E F G : ℝ)

-- Define the conditions
variable (h1 : C = (A + B) / 2)
variable (h2 : D = (A + C) / 2)
variable (h3 : E = (A + D) / 2)
variable (h4 : F = (A + E) / 2)
variable (h5 : G = (A + F) / 2)
variable (h6 : G - A = 1)

-- State the theorem
theorem length_of_AB : B - A = 32 := by sorry

end NUMINAMATH_CALUDE_length_of_AB_l2294_229430


namespace NUMINAMATH_CALUDE_student_circle_circumference_l2294_229461

/-- The circumference of a circle formed by people standing with overlapping arms -/
def circle_circumference (n : ℕ) (arm_span : ℝ) (overlap : ℝ) : ℝ :=
  n * (arm_span - overlap)

/-- Proof that the circumference of the circle formed by 16 students is 110.4 cm -/
theorem student_circle_circumference :
  circle_circumference 16 10.4 3.5 = 110.4 := by
  sorry

end NUMINAMATH_CALUDE_student_circle_circumference_l2294_229461


namespace NUMINAMATH_CALUDE_inequality_proof_l2294_229408

theorem inequality_proof (n : ℕ) (hn : n > 0) :
  (2 * n^2 + 3 * n + 1)^n ≥ 6^n * (n!)^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2294_229408


namespace NUMINAMATH_CALUDE_alpha_plus_beta_value_l2294_229441

theorem alpha_plus_beta_value (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/4))
  (h2 : β ∈ Set.Ioo 0 (π/4))
  (h3 : α.sin * (3*π/2 + α).cos - (π/2 + α).sin * α.cos = -3/5)
  (h4 : 3 * β.sin = (2*α + β).sin) :
  α + β = π/4 := by sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_value_l2294_229441


namespace NUMINAMATH_CALUDE_complement_union_M_N_l2294_229484

def U : Finset ℕ := {1,2,3,4,5,6,7,8}
def M : Finset ℕ := {1,3,5,7}
def N : Finset ℕ := {5,6,7}

theorem complement_union_M_N :
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l2294_229484


namespace NUMINAMATH_CALUDE_prime_cube_difference_equation_l2294_229435

theorem prime_cube_difference_equation :
  ∃! (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧
    p^3 - q^3 = 11*r ∧
    p = 13 ∧ q = 2 ∧ r = 199 := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_difference_equation_l2294_229435


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l2294_229488

theorem rope_cutting_problem : Nat.gcd 42 (Nat.gcd 56 (Nat.gcd 63 77)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l2294_229488


namespace NUMINAMATH_CALUDE_bridge_dealing_is_systematic_sampling_l2294_229497

/-- Represents the sampling method used in card dealing --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a deck of cards --/
structure Deck :=
  (size : Nat)
  (shuffled : Bool)

/-- Represents the card dealing process in bridge --/
structure BridgeDealing :=
  (deck : Deck)
  (startingCardRandom : Bool)
  (dealInOrder : Bool)
  (playerHandSize : Nat)

/-- Determines the sampling method used in bridge card dealing --/
def determineSamplingMethod (dealing : BridgeDealing) : SamplingMethod :=
  sorry

/-- Theorem stating that bridge card dealing uses Systematic Sampling --/
theorem bridge_dealing_is_systematic_sampling 
  (dealing : BridgeDealing) 
  (h1 : dealing.deck.size = 52)
  (h2 : dealing.deck.shuffled = true)
  (h3 : dealing.startingCardRandom = true)
  (h4 : dealing.dealInOrder = true)
  (h5 : dealing.playerHandSize = 13) :
  determineSamplingMethod dealing = SamplingMethod.Systematic :=
  sorry

end NUMINAMATH_CALUDE_bridge_dealing_is_systematic_sampling_l2294_229497


namespace NUMINAMATH_CALUDE_twelve_point_polygons_l2294_229403

/-- The number of distinct convex polygons with three or more sides
    that can be formed from 12 points on a circle's circumference. -/
def num_polygons (n : ℕ) : ℕ :=
  2^n - (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2)

/-- Theorem stating that the number of distinct convex polygons
    with three or more sides formed from 12 points on a circle
    is equal to 4017. -/
theorem twelve_point_polygons :
  num_polygons 12 = 4017 := by
  sorry

end NUMINAMATH_CALUDE_twelve_point_polygons_l2294_229403


namespace NUMINAMATH_CALUDE_track_circumference_l2294_229409

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (v1 v2 : ℝ) (t : ℝ) (h1 : v1 = 20) (h2 : v2 = 13) (h3 : t = 33 / 60) :
  v1 * t + v2 * t = 18.15 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_l2294_229409


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2294_229481

theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + y^2 = 36) →
  (∃ (c : ℝ), c > 0 ∧ c^2 = 32 ∧ 2 * c = 8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2294_229481


namespace NUMINAMATH_CALUDE_same_color_probability_l2294_229419

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  blue : ℕ
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total number of jelly beans a person has -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.blue + jb.green + jb.yellow + jb.red

/-- Represents the jelly bean distribution for each person -/
def abe : JellyBeans := { blue := 2, green := 1, yellow := 0, red := 0 }
def bob : JellyBeans := { blue := 1, green := 2, yellow := 1, red := 0 }
def cara : JellyBeans := { blue := 3, green := 2, yellow := 0, red := 1 }

/-- Calculates the probability of picking a specific color for a person -/
def prob_pick_color (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / jb.total

/-- Theorem: The probability of all three people picking jelly beans of the same color is 5/36 -/
theorem same_color_probability :
  (prob_pick_color abe abe.blue * prob_pick_color bob bob.blue * prob_pick_color cara cara.blue) +
  (prob_pick_color abe abe.green * prob_pick_color bob bob.green * prob_pick_color cara cara.green) =
  5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2294_229419


namespace NUMINAMATH_CALUDE_emu_egg_production_l2294_229482

/-- The number of eggs laid by each female emu per day -/
def eggs_per_female_emu_per_day (num_pens : ℕ) (emus_per_pen : ℕ) (total_eggs_per_week : ℕ) : ℚ :=
  let total_emus := num_pens * emus_per_pen
  let female_emus := total_emus / 2
  (total_eggs_per_week : ℚ) / (female_emus : ℚ) / 7

theorem emu_egg_production :
  eggs_per_female_emu_per_day 4 6 84 = 1 := by
  sorry

#eval eggs_per_female_emu_per_day 4 6 84

end NUMINAMATH_CALUDE_emu_egg_production_l2294_229482


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l2294_229473

/-- Represents the number of students in each grade -/
structure GradePopulation where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the number of students to be sampled from each grade -/
structure SampleSize where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the stratified sample size for each grade -/
def stratifiedSample (pop : GradePopulation) (totalSample : ℕ) : SampleSize :=
  let totalPop := pop.first + pop.second + pop.third
  { first := (totalSample * pop.first + totalPop - 1) / totalPop,
    second := (totalSample * pop.second + totalPop - 1) / totalPop,
    third := (totalSample * pop.third + totalPop - 1) / totalPop }

theorem correct_stratified_sample :
  let pop := GradePopulation.mk 600 680 720
  let sample := stratifiedSample pop 50
  sample.first = 15 ∧ sample.second = 17 ∧ sample.third = 18 := by
  sorry


end NUMINAMATH_CALUDE_correct_stratified_sample_l2294_229473


namespace NUMINAMATH_CALUDE_f_is_even_l2294_229469

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the function f
def f (h : ℝ → ℝ) (x : ℝ) : ℝ := |h (x^5)|

-- Theorem statement
theorem f_is_even (h : ℝ → ℝ) (h_even : IsEven h) : IsEven (f h) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l2294_229469


namespace NUMINAMATH_CALUDE_train_capacity_ratio_l2294_229455

def train_problem (red_boxcars blue_boxcars black_boxcars : ℕ)
  (black_capacity : ℕ) (red_multiplier : ℕ) (total_capacity : ℕ) : Prop :=
  red_boxcars = 3 ∧
  blue_boxcars = 4 ∧
  black_boxcars = 7 ∧
  black_capacity = 4000 ∧
  red_multiplier = 3 ∧
  total_capacity = 132000 ∧
  ∃ (blue_capacity : ℕ),
    red_boxcars * (red_multiplier * blue_capacity) +
    blue_boxcars * blue_capacity +
    black_boxcars * black_capacity = total_capacity ∧
    2 * black_capacity = blue_capacity

theorem train_capacity_ratio 
  (red_boxcars blue_boxcars black_boxcars : ℕ)
  (black_capacity : ℕ) (red_multiplier : ℕ) (total_capacity : ℕ) :
  train_problem red_boxcars blue_boxcars black_boxcars black_capacity red_multiplier total_capacity →
  ∃ (blue_capacity : ℕ), 2 * black_capacity = blue_capacity :=
by sorry

end NUMINAMATH_CALUDE_train_capacity_ratio_l2294_229455


namespace NUMINAMATH_CALUDE_function_max_min_implies_m_range_l2294_229445

/-- The function f(x) = x^2 - 2x + 3 on [0, m] with max 3 and min 2 implies m ∈ [1, 2] -/
theorem function_max_min_implies_m_range 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 - 2*x + 3) 
  (m : ℝ) 
  (h_max : ∃ x ∈ Set.Icc 0 m, ∀ y ∈ Set.Icc 0 m, f y ≤ f x)
  (h_min : ∃ x ∈ Set.Icc 0 m, ∀ y ∈ Set.Icc 0 m, f x ≤ f y)
  (h_max_val : ∃ x ∈ Set.Icc 0 m, f x = 3)
  (h_min_val : ∃ x ∈ Set.Icc 0 m, f x = 2) :
  m ∈ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_function_max_min_implies_m_range_l2294_229445


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2294_229417

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2294_229417


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_l2294_229496

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Theorem 1
theorem range_of_x_when_a_is_one (x : ℝ) (h1 : p x 1) (h2 : q x) :
  2 < x ∧ x < 3 := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) (h : a > 0) 
  (h_suff : ∀ x, ¬(p x a) → ¬(q x))
  (h_not_nec : ∃ x, ¬(q x) ∧ p x a) :
  1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_l2294_229496


namespace NUMINAMATH_CALUDE_prob_at_least_two_pass_written_is_0_6_expected_students_with_advantage_is_0_96_l2294_229458

-- Define the probabilities for each student passing the written test
def prob_written_A : ℝ := 0.4
def prob_written_B : ℝ := 0.8
def prob_written_C : ℝ := 0.5

-- Define the probabilities for each student passing the interview
def prob_interview_A : ℝ := 0.8
def prob_interview_B : ℝ := 0.4
def prob_interview_C : ℝ := 0.64

-- Function to calculate the probability of at least two students passing the written test
def prob_at_least_two_pass_written : ℝ :=
  prob_written_A * prob_written_B * (1 - prob_written_C) +
  prob_written_A * (1 - prob_written_B) * prob_written_C +
  (1 - prob_written_A) * prob_written_B * prob_written_C +
  prob_written_A * prob_written_B * prob_written_C

-- Function to calculate the probability of a student receiving admission advantage
def prob_admission_advantage (written_prob interview_prob : ℝ) : ℝ :=
  written_prob * interview_prob

-- Function to calculate the mathematical expectation of students receiving admission advantage
def expected_students_with_advantage : ℝ :=
  3 * (prob_admission_advantage prob_written_A prob_interview_A)

-- Theorem statements
theorem prob_at_least_two_pass_written_is_0_6 :
  prob_at_least_two_pass_written = 0.6 := by sorry

theorem expected_students_with_advantage_is_0_96 :
  expected_students_with_advantage = 0.96 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_pass_written_is_0_6_expected_students_with_advantage_is_0_96_l2294_229458


namespace NUMINAMATH_CALUDE_total_study_time_l2294_229478

def study_time (wednesday thursday friday weekend : ℕ) : Prop :=
  (wednesday = 2) ∧
  (thursday = 3 * wednesday) ∧
  (friday = thursday / 2) ∧
  (weekend = wednesday + thursday + friday) ∧
  (wednesday + thursday + friday + weekend = 22)

theorem total_study_time :
  ∃ (wednesday thursday friday weekend : ℕ),
    study_time wednesday thursday friday weekend :=
by sorry

end NUMINAMATH_CALUDE_total_study_time_l2294_229478


namespace NUMINAMATH_CALUDE_cube_of_prime_condition_l2294_229440

theorem cube_of_prime_condition (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ 2^n + n^2 + 25 = p^3) ↔ n = 6 :=
sorry

end NUMINAMATH_CALUDE_cube_of_prime_condition_l2294_229440


namespace NUMINAMATH_CALUDE_celia_video_streaming_budget_l2294_229438

/-- Represents Celia's monthly budget --/
structure Budget where
  food_per_week : ℕ
  rent : ℕ
  cell_phone : ℕ
  savings : ℕ
  weeks : ℕ
  savings_rate : ℚ

/-- Calculates the total known expenses --/
def total_known_expenses (b : Budget) : ℕ :=
  b.food_per_week * b.weeks + b.rent + b.cell_phone

/-- Calculates the total spending including savings --/
def total_spending (b : Budget) : ℚ :=
  b.savings / b.savings_rate

/-- Calculates the amount set aside for video streaming services --/
def video_streaming_budget (b : Budget) : ℚ :=
  total_spending b - total_known_expenses b

/-- Theorem stating that Celia's video streaming budget is $30 --/
theorem celia_video_streaming_budget :
  ∃ (b : Budget),
    b.food_per_week ≤ 100 ∧
    b.rent = 1500 ∧
    b.cell_phone = 50 ∧
    b.savings = 198 ∧
    b.weeks = 4 ∧
    b.savings_rate = 1/10 ∧
    video_streaming_budget b = 30 :=
  sorry

end NUMINAMATH_CALUDE_celia_video_streaming_budget_l2294_229438


namespace NUMINAMATH_CALUDE_closest_integer_to_two_plus_sqrt_fifteen_l2294_229491

theorem closest_integer_to_two_plus_sqrt_fifteen :
  ∀ n : ℤ, n ≠ 6 → |6 - (2 + Real.sqrt 15)| < |n - (2 + Real.sqrt 15)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_two_plus_sqrt_fifteen_l2294_229491


namespace NUMINAMATH_CALUDE_marble_ratio_l2294_229463

theorem marble_ratio (total : ℕ) (red : ℕ) (dark_blue : ℕ) 
  (h1 : total = 63) 
  (h2 : red = 38) 
  (h3 : dark_blue = 6) :
  (total - red - dark_blue) / red = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l2294_229463


namespace NUMINAMATH_CALUDE_sum_of_permutations_unique_l2294_229466

/-- Represents a positive integer with at least two digits as a list of its digits. -/
def PositiveInteger := {l : List Nat // l.length ≥ 2 ∧ l.head! ≠ 0}

/-- Calculates the sum of all permutations of a number's digits, excluding the original number. -/
def sumOfPermutations (n : PositiveInteger) : Nat :=
  sorry

/-- Theorem stating that the sum of permutations is unique for each number. -/
theorem sum_of_permutations_unique (x y : PositiveInteger) :
  x ≠ y → sumOfPermutations x ≠ sumOfPermutations y := by
  sorry

end NUMINAMATH_CALUDE_sum_of_permutations_unique_l2294_229466


namespace NUMINAMATH_CALUDE_four_heads_before_three_tails_l2294_229405

/-- The probability of getting heads or tails in a fair coin flip -/
def p_head : ℚ := 1/2
def p_tail : ℚ := 1/2

/-- The probability of encountering 4 heads before 3 tails in repeated fair coin flips -/
noncomputable def q : ℚ := sorry

/-- Theorem stating that q is equal to 28/47 -/
theorem four_heads_before_three_tails : q = 28/47 := by sorry

end NUMINAMATH_CALUDE_four_heads_before_three_tails_l2294_229405


namespace NUMINAMATH_CALUDE_base_8_first_digit_of_395_l2294_229465

def base_8_first_digit (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let p := Nat.log 8 n
    (n / 8^p) % 8

theorem base_8_first_digit_of_395 :
  base_8_first_digit 395 = 6 := by
sorry

end NUMINAMATH_CALUDE_base_8_first_digit_of_395_l2294_229465


namespace NUMINAMATH_CALUDE_exists_angle_sum_with_adjacent_le_180_l2294_229429

/-- A convex quadrilateral is a quadrilateral where each interior angle is less than 180 degrees. -/
structure ConvexQuadrilateral where
  angles : Fin 4 → ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 360
  all_angles_less_than_180 : ∀ i, angles i < 180

/-- 
In any convex quadrilateral, there exists an angle such that the sum of 
this angle with each of its adjacent angles does not exceed 180°.
-/
theorem exists_angle_sum_with_adjacent_le_180 (q : ConvexQuadrilateral) : 
  ∃ i : Fin 4, (q.angles i + q.angles ((i + 1) % 4) ≤ 180) ∧ 
                (q.angles i + q.angles ((i + 3) % 4) ≤ 180) := by
  sorry


end NUMINAMATH_CALUDE_exists_angle_sum_with_adjacent_le_180_l2294_229429


namespace NUMINAMATH_CALUDE_remaining_distance_to_hotel_l2294_229400

/-- Calculates the remaining distance to the hotel given the initial conditions of Samuel's journey --/
theorem remaining_distance_to_hotel (total_distance : ℝ) (initial_speed : ℝ) (initial_time : ℝ) (second_speed : ℝ) (second_time : ℝ) :
  total_distance = 600 ∧
  initial_speed = 50 ∧
  initial_time = 3 ∧
  second_speed = 80 ∧
  second_time = 4 →
  total_distance - (initial_speed * initial_time + second_speed * second_time) = 130 := by
  sorry

#check remaining_distance_to_hotel

end NUMINAMATH_CALUDE_remaining_distance_to_hotel_l2294_229400


namespace NUMINAMATH_CALUDE_random_triangle_probability_l2294_229439

/-- The number of ways to choose 3 different numbers from 1 to 179 -/
def total_combinations : ℕ := 939929

/-- The number of valid angle triples that form a triangle -/
def valid_triples : ℕ := 2611

/-- A function that determines if three numbers form valid angles of a triangle -/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0

/-- The probability of randomly selecting three different numbers from 1 to 179
    that form valid angles of a triangle -/
def triangle_probability : ℚ := valid_triples / total_combinations

/-- Theorem stating the probability of randomly selecting three different numbers
    from 1 to 179 that form valid angles of a triangle -/
theorem random_triangle_probability :
  triangle_probability = 2611 / 939929 := by sorry

end NUMINAMATH_CALUDE_random_triangle_probability_l2294_229439


namespace NUMINAMATH_CALUDE_divisibility_by_17_l2294_229450

theorem divisibility_by_17 (a b : ℤ) : 
  let x : ℤ := 3 * b - 5 * a
  let y : ℤ := 9 * a - 2 * b
  (17 ∣ (2 * x + 3 * y)) ∧ (17 ∣ (9 * x + 5 * y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_17_l2294_229450


namespace NUMINAMATH_CALUDE_max_truck_load_is_2000_l2294_229492

/-- Represents the maximum load a truck can carry given the following conditions:
    - There are three trucks for delivery
    - Boxes come in two weights: 10 pounds and 40 pounds
    - Customer ordered equal quantities of both lighter and heavier products
    - Total number of boxes shipped is 240
-/
def max_truck_load : ℕ :=
  let total_boxes : ℕ := 240
  let num_trucks : ℕ := 3
  let light_box_weight : ℕ := 10
  let heavy_box_weight : ℕ := 40
  let boxes_per_type : ℕ := total_boxes / 2
  let total_weight : ℕ := boxes_per_type * light_box_weight + boxes_per_type * heavy_box_weight
  total_weight / num_trucks

theorem max_truck_load_is_2000 : max_truck_load = 2000 := by
  sorry

end NUMINAMATH_CALUDE_max_truck_load_is_2000_l2294_229492


namespace NUMINAMATH_CALUDE_circle_radius_equation_l2294_229427

/-- The value of d that makes the circle with equation x^2 - 8x + y^2 + 10y + d = 0 have a radius of 5 -/
theorem circle_radius_equation (x y : ℝ) (d : ℝ) : 
  (∀ x y, x^2 - 8*x + y^2 + 10*y + d = 0 ↔ (x - 4)^2 + (y + 5)^2 = 25) → 
  d = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_equation_l2294_229427


namespace NUMINAMATH_CALUDE_equation_solution_l2294_229401

theorem equation_solution : ∃ X : ℝ, 
  (0.125 * X) / ((19/24 - 21/40) * 8*(7/16)) = 
  ((1 + 28/63 - 17/21) * 0.7) / (0.675 * 2.4 - 0.02) ∧ X = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2294_229401


namespace NUMINAMATH_CALUDE_AF_AT_ratio_l2294_229423

-- Define the triangle ABC and points D, E, F, T
variable (A B C D E F T : ℝ × ℝ)

-- Define the conditions
axiom on_AB : ∃ t : ℝ, D = (1 - t) • A + t • B ∧ 0 ≤ t ∧ t ≤ 1
axiom on_AC : ∃ s : ℝ, E = (1 - s) • A + s • C ∧ 0 ≤ s ∧ s ≤ 1
axiom on_DE : ∃ r : ℝ, F = (1 - r) • D + r • E ∧ 0 ≤ r ∧ r ≤ 1
axiom on_AT : ∃ q : ℝ, F = (1 - q) • A + q • T ∧ 0 ≤ q ∧ q ≤ 1

axiom AD_length : dist A D = 1
axiom DB_length : dist D B = 4
axiom AE_length : dist A E = 3
axiom EC_length : dist E C = 3

axiom angle_bisector : 
  dist B T / dist T C = dist A B / dist A C

-- Define the theorem to be proved
theorem AF_AT_ratio : 
  dist A F / dist A T = 11 / 40 :=
sorry

end NUMINAMATH_CALUDE_AF_AT_ratio_l2294_229423


namespace NUMINAMATH_CALUDE_school_transfer_percentage_l2294_229418

theorem school_transfer_percentage : 
  ∀ (total_students : ℕ) (school_A_percent school_C_percent : ℚ),
    school_A_percent = 60 / 100 →
    (30 / 100 * school_A_percent + 
     (school_C_percent - 30 / 100 * school_A_percent) / (1 - school_A_percent)) * total_students = 
    school_C_percent * total_students →
    school_C_percent = 34 / 100 →
    (school_C_percent - 30 / 100 * school_A_percent) / (1 - school_A_percent) = 40 / 100 :=
by sorry

end NUMINAMATH_CALUDE_school_transfer_percentage_l2294_229418


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_quartic_equation_solutions_l2294_229480

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ 2*x^2 + 4*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = -1 - Real.sqrt 6 / 2 ∧ 
             x₂ = -1 + Real.sqrt 6 / 2 ∧ 
             f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

theorem quartic_equation_solutions :
  let g : ℝ → ℝ := λ x ↦ 4*(2*x - 1)^2 - 9*(x + 4)^2
  ∃ x₁ x₂ : ℝ, x₁ = -8/11 ∧ 
             x₂ = 16/5 ∧ 
             g x₁ = 0 ∧ g x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_quartic_equation_solutions_l2294_229480


namespace NUMINAMATH_CALUDE_total_viewing_time_is_900_hours_l2294_229412

/-- Calculates the total viewing time for two people watching multiple videos at different speeds -/
def totalViewingTime (videoLength : ℕ) (numVideos : ℕ) (lilaSpeed : ℕ) (rogerSpeed : ℕ) : ℕ :=
  (videoLength * numVideos / lilaSpeed) + (videoLength * numVideos / rogerSpeed)

/-- Theorem stating that the total viewing time for Lila and Roger is 900 hours -/
theorem total_viewing_time_is_900_hours :
  totalViewingTime 100 6 2 1 = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_viewing_time_is_900_hours_l2294_229412


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l2294_229446

/-- Given that ax^2 + 5x - 2 > 0 has solution set {x | 1/2 < x < 2}, prove:
    1. a = -2
    2. The solution set of ax^2 - 5x + a^2 - 1 > 0 is {x | -3 < x < 1/2} -/
theorem quadratic_inequality_problem 
  (h : ∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) :
  (a = -2) ∧ 
  (∀ x, a*x^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l2294_229446


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l2294_229444

theorem binomial_coefficient_n_minus_two (n : ℕ+) : 
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l2294_229444


namespace NUMINAMATH_CALUDE_squares_below_line_eq_660_l2294_229414

/-- The number of squares below the line 7x + 221y = 1547 in the first quadrant -/
def squares_below_line : ℕ :=
  let x_intercept : ℕ := 221
  let y_intercept : ℕ := 7
  let total_squares : ℕ := x_intercept * y_intercept
  let diagonal_squares : ℕ := x_intercept + y_intercept - 1
  let non_diagonal_squares : ℕ := total_squares - diagonal_squares
  non_diagonal_squares / 2

/-- The number of squares below the line 7x + 221y = 1547 in the first quadrant is 660 -/
theorem squares_below_line_eq_660 : squares_below_line = 660 := by
  sorry

end NUMINAMATH_CALUDE_squares_below_line_eq_660_l2294_229414


namespace NUMINAMATH_CALUDE_candy_packing_problem_l2294_229470

theorem candy_packing_problem (a : ℕ) : 
  (a % 10 = 6) ∧ 
  (a % 15 = 11) ∧ 
  (200 ≤ a) ∧ 
  (a ≤ 250) ↔ 
  (a = 206 ∨ a = 236) :=
sorry

end NUMINAMATH_CALUDE_candy_packing_problem_l2294_229470


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l2294_229406

def A : Set ℝ := {-3, -1, 1, 3}
def B : Set ℝ := {x : ℝ | x^2 + 2*x - 3 = 0}

theorem intersection_complement_equal : A ∩ (Set.univ \ B) = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l2294_229406


namespace NUMINAMATH_CALUDE_profit_without_discount_l2294_229425

/-- Represents the profit percentage and discount percentage as rational numbers -/
def ProfitWithDiscount : ℚ := 44 / 100
def DiscountPercentage : ℚ := 4 / 100

/-- Theorem: If a shopkeeper earns a 44% profit after offering a 4% discount, 
    they would earn a 50% profit without the discount -/
theorem profit_without_discount 
  (cost_price : ℚ) 
  (selling_price : ℚ) 
  (marked_price : ℚ) 
  (h1 : selling_price = cost_price * (1 + ProfitWithDiscount))
  (h2 : selling_price = marked_price * (1 - DiscountPercentage))
  : (marked_price - cost_price) / cost_price = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_profit_without_discount_l2294_229425


namespace NUMINAMATH_CALUDE_nth_prime_47_l2294_229431

def is_nth_prime (n : ℕ) (p : ℕ) : Prop :=
  p.Prime ∧ (Finset.filter Nat.Prime (Finset.range p)).card = n

theorem nth_prime_47 (n : ℕ) :
  is_nth_prime n 47 → n = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_nth_prime_47_l2294_229431


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l2294_229479

theorem probability_at_least_one_correct (p : ℝ) (h1 : p = 1/2) :
  1 - (1 - p)^3 = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l2294_229479


namespace NUMINAMATH_CALUDE_fraction_numerator_l2294_229459

theorem fraction_numerator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : y / 20 + x = 0.35 * y) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_l2294_229459


namespace NUMINAMATH_CALUDE_courier_journey_l2294_229474

/-- The specified time for the courier's journey in minutes -/
def specified_time : ℝ := 40

/-- The total distance the courier traveled in kilometers -/
def total_distance : ℝ := 36

/-- The speed at which the courier arrives early in km/min -/
def early_speed : ℝ := 1.2

/-- The speed at which the courier arrives late in km/min -/
def late_speed : ℝ := 0.8

/-- The time by which the courier arrives early in minutes -/
def early_time : ℝ := 10

/-- The time by which the courier arrives late in minutes -/
def late_time : ℝ := 5

theorem courier_journey :
  early_speed * (specified_time - early_time) = late_speed * (specified_time + late_time) ∧
  total_distance = early_speed * (specified_time - early_time) :=
by sorry

end NUMINAMATH_CALUDE_courier_journey_l2294_229474


namespace NUMINAMATH_CALUDE_matrix_determinant_l2294_229424

def matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, -3, 3],
    ![0,  5, -1],
    ![4, -2, 1]]

theorem matrix_determinant :
  Matrix.det matrix = -45 := by sorry

end NUMINAMATH_CALUDE_matrix_determinant_l2294_229424


namespace NUMINAMATH_CALUDE_area_of_XYZW_l2294_229472

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the larger rectangle XYZW -/
def XYZW : Rectangle := { width := 14, height := 28 }

/-- Represents one of the smaller identical rectangles -/
def smallRect : Rectangle := { width := 7, height := 14 }

theorem area_of_XYZW :
  XYZW.width = smallRect.height ∧
  XYZW.height = 3 * smallRect.width + smallRect.width ∧
  XYZW.area = 392 := by
  sorry

#check area_of_XYZW

end NUMINAMATH_CALUDE_area_of_XYZW_l2294_229472


namespace NUMINAMATH_CALUDE_interchangeable_statements_l2294_229467

-- Define the concept of geometric objects
inductive GeometricObject
| Line
| Plane

-- Define the relationships between geometric objects
inductive Relationship
| Perpendicular
| Parallel

-- Define a geometric statement
structure GeometricStatement where
  obj1 : GeometricObject
  obj2 : GeometricObject
  rel1 : Relationship
  obj3 : GeometricObject
  rel2 : Relationship

-- Define the concept of an interchangeable statement
def isInterchangeable (s : GeometricStatement) : Prop :=
  (s.obj1 = GeometricObject.Line ∧ s.obj2 = GeometricObject.Plane) ∨
  (s.obj1 = GeometricObject.Plane ∧ s.obj2 = GeometricObject.Line)

-- Define the four statements
def statement1 : GeometricStatement :=
  { obj1 := GeometricObject.Line
  , obj2 := GeometricObject.Line
  , rel1 := Relationship.Perpendicular
  , obj3 := GeometricObject.Plane
  , rel2 := Relationship.Parallel }

def statement2 : GeometricStatement :=
  { obj1 := GeometricObject.Plane
  , obj2 := GeometricObject.Plane
  , rel1 := Relationship.Perpendicular
  , obj3 := GeometricObject.Plane
  , rel2 := Relationship.Parallel }

def statement3 : GeometricStatement :=
  { obj1 := GeometricObject.Line
  , obj2 := GeometricObject.Line
  , rel1 := Relationship.Parallel
  , obj3 := GeometricObject.Line
  , rel2 := Relationship.Parallel }

def statement4 : GeometricStatement :=
  { obj1 := GeometricObject.Line
  , obj2 := GeometricObject.Line
  , rel1 := Relationship.Parallel
  , obj3 := GeometricObject.Plane
  , rel2 := Relationship.Parallel }

-- Theorem to prove
theorem interchangeable_statements :
  isInterchangeable statement1 ∧ isInterchangeable statement3 ∧
  ¬isInterchangeable statement2 ∧ ¬isInterchangeable statement4 :=
sorry

end NUMINAMATH_CALUDE_interchangeable_statements_l2294_229467


namespace NUMINAMATH_CALUDE_kanul_total_amount_l2294_229483

/-- The total amount Kanul had -/
def total_amount : ℝ := 137500

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 80000

/-- The amount spent on machinery -/
def machinery : ℝ := 30000

/-- The percentage of total amount spent as cash -/
def cash_percentage : ℝ := 0.20

theorem kanul_total_amount : 
  total_amount = raw_materials + machinery + cash_percentage * total_amount := by
  sorry

end NUMINAMATH_CALUDE_kanul_total_amount_l2294_229483


namespace NUMINAMATH_CALUDE_books_for_girls_l2294_229442

theorem books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ) : 
  num_girls = 15 → 
  num_boys = 10 → 
  total_books = 375 → 
  (num_girls * (total_books / (num_girls + num_boys))) = 225 := by
sorry

end NUMINAMATH_CALUDE_books_for_girls_l2294_229442


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2294_229422

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_product : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + c*d + b*c + a*d + a*c + b*d ≥ 10 ∧ 
  (a^2 + b^2 + c^2 + d^2 + a*b + c*d + b*c + a*d + a*c + b*d = 10 ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2294_229422


namespace NUMINAMATH_CALUDE_bike_distance_l2294_229420

/-- The distance traveled by a bike given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A bike traveling at 50 m/s for 7 seconds covers a distance of 350 meters -/
theorem bike_distance : distance_traveled 50 7 = 350 := by
  sorry

end NUMINAMATH_CALUDE_bike_distance_l2294_229420


namespace NUMINAMATH_CALUDE_max_rounds_four_teams_one_match_l2294_229471

/-- Represents a round-robin tournament with 18 teams -/
structure Tournament :=
  (teams : Finset (Fin 18))
  (rounds : Fin 17 → Finset (Fin 18 × Fin 18))
  (round_valid : ∀ r, (rounds r).card = 9)
  (round_pairs : ∀ r t, (t ∈ teams) → (∃! u, (t, u) ∈ rounds r ∨ (u, t) ∈ rounds r))
  (all_play_all : ∀ t u, t ≠ u → (∃! r, (t, u) ∈ rounds r ∨ (u, t) ∈ rounds r))

/-- The property that there exist 4 teams with exactly 1 match played among them -/
def has_four_teams_one_match (T : Tournament) (n : ℕ) : Prop :=
  ∃ (a b c d : Fin 18), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∃! (i j : Fin 18) (r : Fin n), 
      ((i = a ∧ j = b) ∨ (i = a ∧ j = c) ∨ (i = a ∧ j = d) ∨
       (i = b ∧ j = c) ∨ (i = b ∧ j = d) ∨ (i = c ∧ j = d)) ∧
      ((i, j) ∈ T.rounds r ∨ (j, i) ∈ T.rounds r))

/-- The main theorem statement -/
theorem max_rounds_four_teams_one_match (T : Tournament) :
  (∀ n ≤ 7, has_four_teams_one_match T n) ∧
  (∃ n > 7, ¬has_four_teams_one_match T n) :=
sorry

end NUMINAMATH_CALUDE_max_rounds_four_teams_one_match_l2294_229471


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achievable_l2294_229485

theorem min_sum_squares (a b c : ℕ) (h : a + 2*b + 3*c = 73) : 
  a^2 + b^2 + c^2 ≥ 381 := by
sorry

theorem min_sum_squares_achievable : 
  ∃ (a b c : ℕ), a + 2*b + 3*c = 73 ∧ a^2 + b^2 + c^2 = 381 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achievable_l2294_229485


namespace NUMINAMATH_CALUDE_sum_range_l2294_229468

theorem sum_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b + 1/a + 9/b = 10) : 2 ≤ a + b ∧ a + b ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_range_l2294_229468


namespace NUMINAMATH_CALUDE_number_problem_l2294_229449

theorem number_problem : ∃ x : ℝ, (x / 5 - 5 = 5) ∧ (x = 50) := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2294_229449


namespace NUMINAMATH_CALUDE_nikolai_wins_l2294_229421

/-- Represents a mountain goat with its jump distance and number of jumps per unit time -/
structure Goat where
  name : String
  jumpDistance : ℕ
  jumpsPerUnitTime : ℕ

/-- Calculates the distance covered by a goat in one unit of time -/
def distancePerUnitTime (g : Goat) : ℕ :=
  g.jumpDistance * g.jumpsPerUnitTime

/-- Calculates the number of jumps needed to cover a given distance -/
def jumpsNeeded (g : Goat) (distance : ℕ) : ℕ :=
  (distance + g.jumpDistance - 1) / g.jumpDistance

/-- The theorem stating that Nikolai completes the journey faster -/
theorem nikolai_wins (gennady nikolai : Goat) (totalDistance : ℕ) : 
  gennady.name = "Gennady" →
  nikolai.name = "Nikolai" →
  gennady.jumpDistance = 6 →
  gennady.jumpsPerUnitTime = 2 →
  nikolai.jumpDistance = 4 →
  nikolai.jumpsPerUnitTime = 3 →
  totalDistance = 2000 →
  distancePerUnitTime gennady = distancePerUnitTime nikolai →
  jumpsNeeded nikolai totalDistance < jumpsNeeded gennady totalDistance :=
by sorry

#check nikolai_wins

end NUMINAMATH_CALUDE_nikolai_wins_l2294_229421


namespace NUMINAMATH_CALUDE_stock_sale_percentage_l2294_229407

/-- Proves that the percentage of stock sold is 100% given the provided conditions -/
theorem stock_sale_percentage
  (cash_realized : ℝ)
  (brokerage_rate : ℝ)
  (cash_after_brokerage : ℝ)
  (h1 : cash_realized = 109.25)
  (h2 : brokerage_rate = 1 / 400)
  (h3 : cash_after_brokerage = 109)
  (h4 : cash_after_brokerage = cash_realized * (1 - brokerage_rate)) :
  cash_realized / (cash_after_brokerage / (1 - brokerage_rate)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_stock_sale_percentage_l2294_229407
