import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_coordinates_P_l65_6592

/-- Given three points P, Q, and R in a plane such that PR/PQ = RQ/PQ = 1/2,
    Q = (2, 5), and R = (0, -10), prove that the sum of coordinates of P is -27. -/
theorem sum_of_coordinates_P (P Q R : ℝ × ℝ) : 
  (dist P R / dist P Q = 1/2) →
  (dist R Q / dist P Q = 1/2) →
  Q = (2, 5) →
  R = (0, -10) →
  P.1 + P.2 = -27 := by
  sorry

#check sum_of_coordinates_P

end NUMINAMATH_CALUDE_sum_of_coordinates_P_l65_6592


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l65_6593

theorem absolute_value_inequality (x : ℝ) : 
  x ≠ 2 → (|(3 * x - 2) / (x - 2)| > 3 ↔ (4/3 < x ∧ x < 2) ∨ x > 2) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l65_6593


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l65_6586

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
def arithmeticSum (a₁ d : ℤ) (n : ℕ) : ℤ := n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum_10 (a₁ a₂ a₆ : ℤ) (d : ℤ) :
  a₁ = -2 →
  a₂ + a₆ = 2 →
  (∀ n : ℕ, arithmeticSequence a₁ d n = a₁ + (n - 1) * d) →
  arithmeticSum a₁ d 10 = 25 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l65_6586


namespace NUMINAMATH_CALUDE_expected_moves_is_six_l65_6539

/-- Represents the state of the glasses --/
inductive GlassState
| Full
| Empty

/-- Represents the configuration of the 4 glasses --/
structure GlassConfig :=
(glass1 : GlassState)
(glass2 : GlassState)
(glass3 : GlassState)
(glass4 : GlassState)

/-- The initial configuration --/
def initialConfig : GlassConfig :=
{ glass1 := GlassState.Full,
  glass2 := GlassState.Empty,
  glass3 := GlassState.Full,
  glass4 := GlassState.Empty }

/-- The target configuration --/
def targetConfig : GlassConfig :=
{ glass1 := GlassState.Empty,
  glass2 := GlassState.Full,
  glass3 := GlassState.Empty,
  glass4 := GlassState.Full }

/-- Represents a valid move (pouring from a full glass to an empty one) --/
inductive ValidMove : GlassConfig → GlassConfig → Prop

/-- The expected number of moves to reach the target configuration --/
noncomputable def expectedMoves : ℝ := 6

/-- Main theorem: The expected number of moves from initial to target config is 6 --/
theorem expected_moves_is_six :
  expectedMoves = 6 :=
sorry

end NUMINAMATH_CALUDE_expected_moves_is_six_l65_6539


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l65_6597

/-- Given a quadratic polynomial P(x) = x^2 + px + q where P(q) < 0, 
    exactly one root of P(x) lies in the interval (0, 1) -/
theorem quadratic_root_in_unit_interval (p q : ℝ) :
  let P : ℝ → ℝ := λ x => x^2 + p*x + q
  (P q < 0) →
  ∃! x : ℝ, P x = 0 ∧ 0 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l65_6597


namespace NUMINAMATH_CALUDE_min_value_xyz_l65_6563

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 27) :
  x + 3 * y + 6 * z ≥ 27 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧ x₀ + 3 * y₀ + 6 * z₀ = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_l65_6563


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l65_6505

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 + 5 * x^2 + d * x + 7
  (g 2 = 11) ∧ (g (-3) = 134) → c = -35/13 ∧ d = 16/13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l65_6505


namespace NUMINAMATH_CALUDE_complex_norm_squared_l65_6559

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.normSq z = 5 - 7*I) : Complex.normSq z = (74:ℝ)/10 := by
  sorry

end NUMINAMATH_CALUDE_complex_norm_squared_l65_6559


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l65_6599

/-- The number of unique two-scoop sundaes that can be made from n types of ice cream -/
def two_scoop_sundaes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem: Given 6 types of ice cream, the number of unique two-scoop sundaes is 15 -/
theorem ice_cream_sundaes :
  two_scoop_sundaes 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l65_6599


namespace NUMINAMATH_CALUDE_min_value_theorem_l65_6547

theorem min_value_theorem (m : ℝ) (a b : ℝ) :
  0 < m → m < 1 →
  ({x : ℝ | x^2 - 2*x + 1 - m^2 < 0} = {x : ℝ | a < x ∧ x < b}) →
  (∀ x : ℝ, x^2 - 2*x + 1 - m^2 < 0 ↔ a < x ∧ x < b) →
  (∀ x : ℝ, 1/(8*a + 2*b) - 1/(3*a - 3*b) ≥ 2/5) ∧
  (∃ x : ℝ, 1/(8*a + 2*b) - 1/(3*a - 3*b) = 2/5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l65_6547


namespace NUMINAMATH_CALUDE_fathers_with_full_time_jobs_l65_6507

theorem fathers_with_full_time_jobs 
  (total_parents : ℝ) 
  (mothers_ratio : ℝ) 
  (mothers_full_time_ratio : ℝ) 
  (no_full_time_ratio : ℝ) 
  (h1 : mothers_ratio = 0.6) 
  (h2 : mothers_full_time_ratio = 5/6) 
  (h3 : no_full_time_ratio = 0.2) : 
  (total_parents * (1 - mothers_ratio) * 3/4) = 
  (total_parents * (1 - no_full_time_ratio) - total_parents * mothers_ratio * mothers_full_time_ratio) := by
sorry

end NUMINAMATH_CALUDE_fathers_with_full_time_jobs_l65_6507


namespace NUMINAMATH_CALUDE_three_integers_divisibility_l65_6503

theorem three_integers_divisibility (x y z : ℕ+) :
  (x ∣ y + z) ∧ (y ∣ x + z) ∧ (z ∣ x + y) →
  (∃ a : ℕ+, (x = a ∧ y = a ∧ z = a) ∨
             (x = a ∧ y = a ∧ z = 2 * a) ∨
             (x = a ∧ y = 2 * a ∧ z = 3 * a)) :=
by sorry

end NUMINAMATH_CALUDE_three_integers_divisibility_l65_6503


namespace NUMINAMATH_CALUDE_product_fixed_sum_squares_not_always_minimized_when_equal_l65_6541

theorem product_fixed_sum_squares_not_always_minimized_when_equal :
  ¬ (∀ (k : ℝ), k > 0 →
    ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = k →
      ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = k →
        x^2 + y^2 ≤ a^2 + b^2 → x = y) :=
by sorry

end NUMINAMATH_CALUDE_product_fixed_sum_squares_not_always_minimized_when_equal_l65_6541


namespace NUMINAMATH_CALUDE_find_divisor_l65_6544

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 301 → quotient = 14 → remainder = 7 → 
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 21 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l65_6544


namespace NUMINAMATH_CALUDE_car_average_speed_l65_6550

/-- The average speed of a car given its distance traveled in two hours -/
theorem car_average_speed (d1 d2 : ℝ) (h1 : d1 = 80) (h2 : d2 = 40) :
  (d1 + d2) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l65_6550


namespace NUMINAMATH_CALUDE_min_sum_for_product_1386_l65_6525

theorem min_sum_for_product_1386 (a b c : ℕ+) : 
  a * b * c = 1386 → 
  ∀ x y z : ℕ+, x * y * z = 1386 → a + b + c ≤ x + y + z ∧ ∃ a b c : ℕ+, a * b * c = 1386 ∧ a + b + c = 34 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_for_product_1386_l65_6525


namespace NUMINAMATH_CALUDE_continuous_triple_composition_identity_implies_identity_l65_6587

def triple_composition_identity (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f (f x)) = x

theorem continuous_triple_composition_identity_implies_identity 
  (f : ℝ → ℝ) (hf : Continuous f) (h_triple : triple_composition_identity f) : 
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_continuous_triple_composition_identity_implies_identity_l65_6587


namespace NUMINAMATH_CALUDE_P_no_real_roots_l65_6528

/-- Recursive definition of the polynomial sequence P_n(x) -/
def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | n + 1, x => x^(11 * (n + 1)) - P n x

/-- Theorem stating that P_n(x) has no real roots for all n ≥ 0 -/
theorem P_no_real_roots : ∀ (n : ℕ) (x : ℝ), P n x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_P_no_real_roots_l65_6528


namespace NUMINAMATH_CALUDE_closest_ratio_is_27_26_l65_6523

/-- The admission fee for adults -/
def adult_fee : ℕ := 30

/-- The admission fee for children -/
def child_fee : ℕ := 15

/-- The total amount collected -/
def total_collected : ℕ := 2400

/-- Represents the number of adults and children at the exhibition -/
structure Attendance where
  adults : ℕ
  children : ℕ
  adults_nonzero : adults > 0
  children_nonzero : children > 0
  total_correct : adult_fee * adults + child_fee * children = total_collected

/-- The ratio of adults to children -/
def attendance_ratio (a : Attendance) : ℚ :=
  a.adults / a.children

/-- Checks if a given ratio is closest to 1 among all possible attendances -/
def is_closest_to_one (r : ℚ) : Prop :=
  ∀ a : Attendance, |attendance_ratio a - 1| ≥ |r - 1|

/-- The main theorem stating that 27/26 is the ratio closest to 1 -/
theorem closest_ratio_is_27_26 :
  is_closest_to_one (27 / 26) :=
sorry

end NUMINAMATH_CALUDE_closest_ratio_is_27_26_l65_6523


namespace NUMINAMATH_CALUDE_trader_weight_manipulation_l65_6531

theorem trader_weight_manipulation :
  ∀ (supplier_weight : ℝ) (cost_price : ℝ),
  supplier_weight > 0 → cost_price > 0 →
  let actual_bought_weight := supplier_weight * 1.1
  let claimed_sell_weight := actual_bought_weight
  let actual_sell_weight := claimed_sell_weight / 1.65
  let weight_difference := claimed_sell_weight - actual_sell_weight
  (cost_price * actual_sell_weight) * 1.65 = cost_price * claimed_sell_weight →
  weight_difference / actual_sell_weight = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_trader_weight_manipulation_l65_6531


namespace NUMINAMATH_CALUDE_pet_store_hamsters_l65_6594

/-- Given a pet store with rabbits and hamsters, prove the number of hamsters
    when the ratio of rabbits to hamsters is 3:4 and there are 18 rabbits. -/
theorem pet_store_hamsters (rabbit_count : ℕ) (hamster_count : ℕ) : 
  (rabbit_count : ℚ) / hamster_count = 3 / 4 →
  rabbit_count = 18 →
  hamster_count = 24 := by
sorry


end NUMINAMATH_CALUDE_pet_store_hamsters_l65_6594


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_l65_6552

theorem min_value_theorem (x : ℝ) (h : x ≥ 2) :
  (∀ y : ℝ, y ≥ 2 → x + 4/x ≤ y + 4/y) ↔ x = 2 :=
by sorry

theorem min_value_is_four (x : ℝ) (h : x ≥ 2) :
  x + 4/x ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_l65_6552


namespace NUMINAMATH_CALUDE_egypt_traditional_growth_l65_6556

-- Define the set of countries
inductive Country
| UnitedStates
| Japan
| France
| Egypt

-- Define the development status of a country
inductive DevelopmentStatus
| Developed
| Developing

-- Define the population growth pattern
inductive PopulationGrowthPattern
| Modern
| Traditional

-- Function to determine the development status of a country
def developmentStatus (c : Country) : DevelopmentStatus :=
  match c with
  | Country.Egypt => DevelopmentStatus.Developing
  | _ => DevelopmentStatus.Developed

-- Function to determine the population growth pattern based on development status
def growthPattern (s : DevelopmentStatus) : PopulationGrowthPattern :=
  match s with
  | DevelopmentStatus.Developed => PopulationGrowthPattern.Modern
  | DevelopmentStatus.Developing => PopulationGrowthPattern.Traditional

-- Theorem: Egypt is the only country with a traditional population growth pattern
theorem egypt_traditional_growth : 
  ∀ c : Country, 
    growthPattern (developmentStatus c) = PopulationGrowthPattern.Traditional ↔ 
    c = Country.Egypt :=
  sorry


end NUMINAMATH_CALUDE_egypt_traditional_growth_l65_6556


namespace NUMINAMATH_CALUDE_final_painting_width_l65_6533

theorem final_painting_width (total_paintings : ℕ) (total_area : ℕ) 
  (small_paintings : ℕ) (small_painting_side : ℕ) 
  (large_painting_length : ℕ) (large_painting_width : ℕ)
  (final_painting_height : ℕ) :
  total_paintings = 5 →
  total_area = 200 →
  small_paintings = 3 →
  small_painting_side = 5 →
  large_painting_length = 10 →
  large_painting_width = 8 →
  final_painting_height = 5 →
  (total_area - 
    (small_paintings * small_painting_side * small_painting_side + 
     large_painting_length * large_painting_width)) / final_painting_height = 9 := by
  sorry

#check final_painting_width

end NUMINAMATH_CALUDE_final_painting_width_l65_6533


namespace NUMINAMATH_CALUDE_pentagon_sum_l65_6530

/-- Definition of a pentagon -/
structure Pentagon where
  sides : ℕ
  vertices : ℕ
  is_pentagon : sides = 5 ∧ vertices = 5

/-- Theorem: The sum of sides and vertices of a pentagon is 10 -/
theorem pentagon_sum (p : Pentagon) : p.sides + p.vertices = 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_sum_l65_6530


namespace NUMINAMATH_CALUDE_parabola_range_theorem_l65_6514

/-- Represents a quadratic function of the form f(x) = x^2 + bx + c -/
def QuadraticFunction (b c : ℝ) := λ x : ℝ => x^2 + b*x + c

theorem parabola_range_theorem (b c : ℝ) :
  (QuadraticFunction b c (-1) = 0) →
  (QuadraticFunction b c 3 = 0) →
  (∀ x : ℝ, QuadraticFunction b c x > -3 ↔ (x < 0 ∨ x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_range_theorem_l65_6514


namespace NUMINAMATH_CALUDE_arithmetic_sequence_64th_term_l65_6549

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence with specific properties, the 64th term is 129. -/
theorem arithmetic_sequence_64th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_3rd : a 3 = 7)
  (h_18th : a 18 = 37) :
  a 64 = 129 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_64th_term_l65_6549


namespace NUMINAMATH_CALUDE_tangent_points_x_coordinate_sum_l65_6564

/-- Parabola struct representing x^2 = 2py -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem stating the relationship between x-coordinates of tangent points and the point on y = -2p -/
theorem tangent_points_x_coordinate_sum (para : Parabola) (M A B : Point) :
  A.y = A.x^2 / (2 * para.p) →  -- A is on the parabola
  B.y = B.x^2 / (2 * para.p) →  -- B is on the parabola
  M.y = -2 * para.p →  -- M is on the line y = -2p
  (A.y - M.y) / (A.x - M.x) = A.x / para.p →  -- MA is tangent to the parabola
  (B.y - M.y) / (B.x - M.x) = B.x / para.p →  -- MB is tangent to the parabola
  A.x + B.x = 2 * M.x := by
  sorry

end NUMINAMATH_CALUDE_tangent_points_x_coordinate_sum_l65_6564


namespace NUMINAMATH_CALUDE_vertex_in_second_quadrant_l65_6511

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 1)^2 + 2

-- Define the vertex of the quadratic function
def vertex : ℝ × ℝ := (-1, 2)

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem vertex_in_second_quadrant :
  in_second_quadrant vertex := by sorry

end NUMINAMATH_CALUDE_vertex_in_second_quadrant_l65_6511


namespace NUMINAMATH_CALUDE_max_k_value_l65_6534

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 13) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l65_6534


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l65_6502

def A : Set ℤ := {1, 2}
def B : Set ℤ := {-1, 1, 4}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l65_6502


namespace NUMINAMATH_CALUDE_remainder_of_12345678901_mod_101_l65_6576

theorem remainder_of_12345678901_mod_101 : 12345678901 % 101 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_12345678901_mod_101_l65_6576


namespace NUMINAMATH_CALUDE_max_d_value_l65_6568

def a (n : ℕ+) : ℕ := n.val ^ 2 + 1000

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (N : ℕ+), d N = 4001 ∧ ∀ (n : ℕ+), d n ≤ 4001 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l65_6568


namespace NUMINAMATH_CALUDE_dropped_student_score_l65_6527

theorem dropped_student_score 
  (initial_students : ℕ) 
  (remaining_students : ℕ) 
  (initial_average : ℚ) 
  (new_average : ℚ) 
  (h1 : initial_students = 16) 
  (h2 : remaining_students = 15) 
  (h3 : initial_average = 61.5) 
  (h4 : new_average = 64) :
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * new_average = 24 := by
  sorry

#check dropped_student_score

end NUMINAMATH_CALUDE_dropped_student_score_l65_6527


namespace NUMINAMATH_CALUDE_wednesday_saturday_earnings_difference_l65_6516

def total_earnings : ℝ := 5182.50
def saturday_earnings : ℝ := 2662.50

theorem wednesday_saturday_earnings_difference :
  saturday_earnings - (total_earnings - saturday_earnings) = 142.50 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_saturday_earnings_difference_l65_6516


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l65_6512

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 16⌉ + ⌈Real.sqrt 200⌉ = 21 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l65_6512


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l65_6596

/-- The solution set of a quadratic inequality -/
def SolutionSet (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 2 > 0}

/-- The open interval (-1/2, 1/3) -/
def OpenInterval : Set ℝ := {x | -1/2 < x ∧ x < 1/3}

/-- 
If the solution set of ax^2 + bx + 2 > 0 is (-1/2, 1/3), then a + b = -14
-/
theorem quadratic_inequality_solution (a b : ℝ) :
  SolutionSet a b = OpenInterval → a + b = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l65_6596


namespace NUMINAMATH_CALUDE_circle_ratio_l65_6566

theorem circle_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : c^2 - a^2 = 4 * a^2) 
  (h2 : b^2 = (a^2 + c^2) / 2) : 
  a / c = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l65_6566


namespace NUMINAMATH_CALUDE_min_distance_MN_l65_6537

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define a line tangent to the unit circle
def tangent_line (P A B : ℝ × ℝ) : Prop :=
  unit_circle A.1 A.2 ∧ unit_circle B.1 B.2 ∧
  ∃ (t : ℝ), (1 - t) • A + t • B = P

-- Define the intersection points M and N
def intersection_points (A B : ℝ × ℝ) (M N : ℝ × ℝ) : Prop :=
  M.2 = 0 ∧ N.1 = 0 ∧ ∃ (t s : ℝ), (1 - t) • A + t • B = M ∧ (1 - s) • A + s • B = N

-- State the theorem
theorem min_distance_MN (P A B M N : ℝ × ℝ) :
  point_on_ellipse P →
  tangent_line P A B →
  intersection_points A B M N →
  ∃ (min_dist : ℝ), min_dist = 3/4 ∧ 
    ∀ (P' A' B' M' N' : ℝ × ℝ), 
      point_on_ellipse P' →
      tangent_line P' A' B' →
      intersection_points A' B' M' N' →
      Real.sqrt ((M'.1 - N'.1)^2 + (M'.2 - N'.2)^2) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_MN_l65_6537


namespace NUMINAMATH_CALUDE_inclination_angle_of_line_l65_6529

/-- The inclination angle of a line with equation y = x - 3 is 45 degrees. -/
theorem inclination_angle_of_line (x y : ℝ) :
  y = x - 3 → Real.arctan 1 = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_inclination_angle_of_line_l65_6529


namespace NUMINAMATH_CALUDE_inequality_max_a_l65_6522

theorem inequality_max_a : 
  (∀ x : ℝ, x ∈ Set.Icc 1 12 → x^2 + 25 + |x^3 - 5*x^2| ≥ (5/2)*x) ∧ 
  (∀ ε > 0, ∃ x : ℝ, x ∈ Set.Icc 1 12 ∧ x^2 + 25 + |x^3 - 5*x^2| < (5/2 + ε)*x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_max_a_l65_6522


namespace NUMINAMATH_CALUDE_ice_cream_box_cost_l65_6569

/-- Represents the cost of a box of ice cream bars -/
def box_cost : ℚ := sorry

/-- Number of ice cream bars in a box -/
def bars_per_box : ℕ := 3

/-- Number of friends -/
def num_friends : ℕ := 6

/-- Number of bars each friend wants to eat -/
def bars_per_friend : ℕ := 2

/-- Cost per person -/
def cost_per_person : ℚ := 5

theorem ice_cream_box_cost :
  box_cost = 7.5 := by sorry

end NUMINAMATH_CALUDE_ice_cream_box_cost_l65_6569


namespace NUMINAMATH_CALUDE_sequence_properties_l65_6504

def a (n : ℕ) : ℚ := 3 - 2^n

theorem sequence_properties :
  (∀ n : ℕ, a (2*n) = 3 - 4^n) ∧ (a 2 / a 3 = 1/5) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l65_6504


namespace NUMINAMATH_CALUDE_simplify_and_compare_l65_6562

theorem simplify_and_compare : 
  1.82 * (2 * Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_compare_l65_6562


namespace NUMINAMATH_CALUDE_two_digit_reverse_divisible_by_11_l65_6581

theorem two_digit_reverse_divisible_by_11 (a b : ℕ) 
  (ha : a ≤ 9) (hb : b ≤ 9) : 
  (1000 * a + 100 * b + 10 * b + a) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_reverse_divisible_by_11_l65_6581


namespace NUMINAMATH_CALUDE_probability_standard_weight_l65_6554

def total_students : ℕ := 500
def standard_weight_students : ℕ := 350

theorem probability_standard_weight :
  (standard_weight_students : ℚ) / total_students = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_standard_weight_l65_6554


namespace NUMINAMATH_CALUDE_sin_2x_minus_pi_6_l65_6536

theorem sin_2x_minus_pi_6 (x : ℝ) (h : Real.cos (x + π / 6) + Real.sin (2 * π / 3 + x) = 1 / 2) :
  Real.sin (2 * x - π / 6) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_minus_pi_6_l65_6536


namespace NUMINAMATH_CALUDE_book_reading_days_l65_6573

theorem book_reading_days : ∀ (total_pages : ℕ) (pages_per_day : ℕ) (fraction : ℚ),
  total_pages = 144 →
  pages_per_day = 8 →
  fraction = 2/3 →
  (fraction * total_pages : ℚ) / pages_per_day = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_book_reading_days_l65_6573


namespace NUMINAMATH_CALUDE_library_books_l65_6510

theorem library_books (borrowed : ℕ) (left : ℕ) (initial : ℕ) : 
  borrowed = 18 → left = 57 → initial = borrowed + left → initial = 75 := by
  sorry

end NUMINAMATH_CALUDE_library_books_l65_6510


namespace NUMINAMATH_CALUDE_expression_value_l65_6598

theorem expression_value (a b c : ℝ) (h1 : a * b * c > 0) (h2 : a * b < 0) :
  (|a| / a + 2 * b / |b| - b * c / |4 * b * c|) = -5/4 ∨
  (|a| / a + 2 * b / |b| - b * c / |4 * b * c|) = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l65_6598


namespace NUMINAMATH_CALUDE_topology_classification_l65_6595

def X : Set Char := {'a', 'b', 'c'}

def τ₁ : Set (Set Char) := {∅, {'a'}, {'c'}, {'a', 'b', 'c'}}
def τ₂ : Set (Set Char) := {∅, {'b'}, {'c'}, {'b', 'c'}, {'a', 'b', 'c'}}
def τ₃ : Set (Set Char) := {∅, {'a'}, {'a', 'b'}, {'a', 'c'}}
def τ₄ : Set (Set Char) := {∅, {'a', 'c'}, {'b', 'c'}, {'c'}, {'a', 'b', 'c'}}

def IsTopology (τ : Set (Set Char)) : Prop :=
  X ∈ τ ∧ ∅ ∈ τ ∧
  (∀ S : Set (Set Char), S ⊆ τ → ⋃₀ S ∈ τ) ∧
  (∀ S : Set (Set Char), S ⊆ τ → ⋂₀ S ∈ τ)

theorem topology_classification :
  IsTopology τ₂ ∧ IsTopology τ₄ ∧ ¬IsTopology τ₁ ∧ ¬IsTopology τ₃ :=
sorry

end NUMINAMATH_CALUDE_topology_classification_l65_6595


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l65_6501

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 8) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6)) ∧
    P = 7/15 ∧ Q = -4/3 ∧ R = 14/5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l65_6501


namespace NUMINAMATH_CALUDE_sphere_radius_l65_6582

/-- Given a sphere and a pole under parallel sun rays, where the sphere's shadow extends
    12 meters from the point of contact with the ground, and a 3-meter tall pole casts
    a 4-meter shadow, the radius of the sphere is 9 meters. -/
theorem sphere_radius (shadow_length : ℝ) (pole_height : ℝ) (pole_shadow : ℝ) :
  shadow_length = 12 →
  pole_height = 3 →
  pole_shadow = 4 →
  ∃ (sphere_radius : ℝ), sphere_radius = 9 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_l65_6582


namespace NUMINAMATH_CALUDE_chucks_team_lead_l65_6532

/-- The lead of Chuck's team over the Yellow Team -/
def lead (chuck_score yellow_score : ℕ) : ℕ := chuck_score - yellow_score

/-- Theorem stating that Chuck's team's lead over the Yellow Team is 17 points -/
theorem chucks_team_lead : lead 72 55 = 17 := by
  sorry

end NUMINAMATH_CALUDE_chucks_team_lead_l65_6532


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l65_6590

theorem common_internal_tangent_length
  (center_distance : ℝ)
  (radius1 : ℝ)
  (radius2 : ℝ)
  (h1 : center_distance = 50)
  (h2 : radius1 = 7)
  (h3 : radius2 = 10) :
  Real.sqrt (center_distance^2 - (radius1 + radius2)^2) = Real.sqrt 2211 :=
by sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l65_6590


namespace NUMINAMATH_CALUDE_x_24_value_l65_6542

theorem x_24_value (x : ℝ) (h : x + 1/x = -Real.sqrt 3) : x^24 = 390625 := by
  sorry

end NUMINAMATH_CALUDE_x_24_value_l65_6542


namespace NUMINAMATH_CALUDE_tank_water_level_l65_6574

theorem tank_water_level (tank_capacity : ℚ) (initial_fraction : ℚ) (added_water : ℚ) : 
  tank_capacity = 40 →
  initial_fraction = 3/4 →
  added_water = 5 →
  (initial_fraction * tank_capacity + added_water) / tank_capacity = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_tank_water_level_l65_6574


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_l65_6519

def initial_trees : ℕ := 237
def planting_factor : ℕ := 5

theorem oak_trees_after_planting :
  initial_trees + planting_factor * initial_trees = 1422 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_l65_6519


namespace NUMINAMATH_CALUDE_max_x0_value_l65_6546

def max_x0 (x : Fin 1997 → ℝ) : Prop :=
  (∀ i, x i > 0) ∧
  x 0 = x 1995 ∧
  (∀ i ∈ Finset.range 1995, x i + 2 / x i = 2 * x (i + 1) + 1 / x (i + 1))

theorem max_x0_value (x : Fin 1997 → ℝ) (h : max_x0 x) :
  x 0 ≤ 2^997 ∧ ∃ y : Fin 1997 → ℝ, max_x0 y ∧ y 0 = 2^997 :=
sorry

end NUMINAMATH_CALUDE_max_x0_value_l65_6546


namespace NUMINAMATH_CALUDE_cubic_polynomial_third_root_l65_6591

theorem cubic_polynomial_third_root 
  (a b : ℚ) 
  (h1 : a * 1^3 + (a + 3*b) * 1^2 + (b - 4*a) * 1 + (6 - a) = 0)
  (h2 : a * (-3)^3 + (a + 3*b) * (-3)^2 + (b - 4*a) * (-3) + (6 - a) = 0) :
  ∃ (x : ℚ), x = 7/13 ∧ a * x^3 + (a + 3*b) * x^2 + (b - 4*a) * x + (6 - a) = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_third_root_l65_6591


namespace NUMINAMATH_CALUDE_smallest_mustang_length_l65_6572

/-- Proves that the smallest model Mustang is 12 inches long given the specified conditions -/
theorem smallest_mustang_length :
  let full_size : ℝ := 240
  let mid_size_ratio : ℝ := 1 / 10
  let smallest_ratio : ℝ := 1 / 2
  let mid_size : ℝ := full_size * mid_size_ratio
  let smallest_size : ℝ := mid_size * smallest_ratio
  smallest_size = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_mustang_length_l65_6572


namespace NUMINAMATH_CALUDE_days_for_one_piece_correct_l65_6583

/-- The number of days Aarti needs to complete one piece of work -/
def days_for_one_piece : ℝ := 6

/-- The number of days Aarti needs to complete three pieces of work -/
def days_for_three_pieces : ℝ := 18

/-- Theorem stating that the number of days for one piece of work is correct -/
theorem days_for_one_piece_correct : 
  days_for_one_piece * 3 = days_for_three_pieces :=
sorry

end NUMINAMATH_CALUDE_days_for_one_piece_correct_l65_6583


namespace NUMINAMATH_CALUDE_condition_a_equals_one_sufficient_not_necessary_l65_6557

-- Define the quadratic equation
def has_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a = 2*x

-- Theorem statement
theorem condition_a_equals_one_sufficient_not_necessary :
  (has_real_roots 1) ∧ (∃ a : ℝ, a ≠ 1 ∧ has_real_roots a) :=
sorry

end NUMINAMATH_CALUDE_condition_a_equals_one_sufficient_not_necessary_l65_6557


namespace NUMINAMATH_CALUDE_root_problems_l65_6578

theorem root_problems :
  (∃ x : ℝ, x^2 = 16 ∧ (x = 4 ∨ x = -4)) ∧
  (∃ y : ℝ, y^3 = -27 ∧ y = -3) ∧
  (Real.sqrt ((-4)^2) = 4) ∧
  (∃ z : ℝ, z^2 = 9 ∧ z = 3) := by
  sorry

end NUMINAMATH_CALUDE_root_problems_l65_6578


namespace NUMINAMATH_CALUDE_agent_encryption_possible_l65_6543

theorem agent_encryption_possible : ∃ (m n p q : ℕ), 
  (m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0) ∧ 
  (7 / 100 : ℚ) = 1 / m + 1 / n ∧
  (13 / 100 : ℚ) = 1 / p + 1 / q :=
sorry

end NUMINAMATH_CALUDE_agent_encryption_possible_l65_6543


namespace NUMINAMATH_CALUDE_license_plate_count_l65_6518

/-- The number of letters in the alphabet --/
def num_letters : ℕ := 26

/-- The number of digits (0-9) --/
def num_digits : ℕ := 10

/-- The number of odd digits --/
def num_odd_digits : ℕ := 5

/-- The number of even digits --/
def num_even_digits : ℕ := 5

/-- The number of positions for digits --/
def num_digit_positions : ℕ := 3

/-- The number of valid license plates --/
def num_valid_plates : ℕ := 6591000

theorem license_plate_count :
  num_letters ^ 3 * (num_digit_positions * num_odd_digits * num_even_digits ^ 2) = num_valid_plates :=
sorry

end NUMINAMATH_CALUDE_license_plate_count_l65_6518


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l65_6584

theorem roots_sum_of_squares (p q : ℝ) : 
  (p^2 - 5*p + 6 = 0) → (q^2 - 5*q + 6 = 0) → p^2 + q^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l65_6584


namespace NUMINAMATH_CALUDE_two_numbers_with_special_properties_l65_6565

theorem two_numbers_with_special_properties : ∃ (a b : ℕ), 
  a ≠ b ∧
  a > 9 ∧ b > 9 ∧
  (a + b) / 2 ≥ 10 ∧ (a + b) / 2 ≤ 99 ∧
  Nat.sqrt (a * b) ≥ 10 ∧ Nat.sqrt (a * b) ≤ 99 ∧
  (a = 98 ∧ b = 32 ∨ a = 32 ∧ b = 98) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_with_special_properties_l65_6565


namespace NUMINAMATH_CALUDE_car_sale_profit_percentage_l65_6570

/-- Calculate the net profit percentage for a car sale --/
theorem car_sale_profit_percentage 
  (purchase_price : ℝ) 
  (repair_cost_percentage : ℝ) 
  (sales_tax_percentage : ℝ) 
  (registration_fee_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 42000)
  (h2 : repair_cost_percentage = 0.35)
  (h3 : sales_tax_percentage = 0.08)
  (h4 : registration_fee_percentage = 0.06)
  (h5 : selling_price = 64900) :
  let total_cost := purchase_price * (1 + repair_cost_percentage + sales_tax_percentage + registration_fee_percentage)
  let net_profit := selling_price - total_cost
  let net_profit_percentage := (net_profit / total_cost) * 100
  ∃ ε > 0, |net_profit_percentage - 3.71| < ε :=
by sorry

end NUMINAMATH_CALUDE_car_sale_profit_percentage_l65_6570


namespace NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l65_6561

/-- The maximum number of stamps that can be purchased with a given amount of money and stamp price. -/
def maxStamps (totalMoney stampPrice : ℕ) : ℕ :=
  totalMoney / stampPrice

/-- Theorem stating that with $50 and stamps costing 25 cents each, the maximum number of stamps that can be purchased is 200. -/
theorem max_stamps_with_50_dollars : 
  let dollarAmount : ℕ := 50
  let stampPriceCents : ℕ := 25
  let totalCents : ℕ := dollarAmount * 100
  maxStamps totalCents stampPriceCents = 200 := by
  sorry

#eval maxStamps (50 * 100) 25

end NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l65_6561


namespace NUMINAMATH_CALUDE_gina_college_cost_l65_6540

/-- Calculates the total cost of Gina's college expenses -/
def total_college_cost (num_credits : ℕ) (cost_per_credit : ℕ) (num_textbooks : ℕ) (cost_per_textbook : ℕ) (facilities_fee : ℕ) : ℕ :=
  num_credits * cost_per_credit + num_textbooks * cost_per_textbook + facilities_fee

/-- Proves that Gina's total college expenses are $7100 -/
theorem gina_college_cost :
  total_college_cost 14 450 5 120 200 = 7100 := by
  sorry

end NUMINAMATH_CALUDE_gina_college_cost_l65_6540


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l65_6579

theorem sum_of_a_and_b (a b : ℝ) (h1 : a * b > 0) (h2 : |a| = 2) (h3 : |b| = 7) :
  a + b = 9 ∨ a + b = -9 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l65_6579


namespace NUMINAMATH_CALUDE_integral_ratio_theorem_l65_6575

theorem integral_ratio_theorem (a b : ℝ) (h : a < b) :
  let f (x : ℝ) := (1 / 20 + 3 / 10) * x^2
  let g (x : ℝ) := x^2
  (∫ x in a..b, f x) / (∫ x in a..b, g x) = 35 / 100 := by
  sorry

end NUMINAMATH_CALUDE_integral_ratio_theorem_l65_6575


namespace NUMINAMATH_CALUDE_fraction_of_male_fish_l65_6558

theorem fraction_of_male_fish (total : ℕ) (female : ℕ) (h1 : total = 45) (h2 : female = 15) :
  (total - female : ℚ) / total = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_male_fish_l65_6558


namespace NUMINAMATH_CALUDE_loop_condition_correct_l65_6515

/-- A program for calculating the average of 20 numbers -/
structure AverageProgram where
  numbers : Fin 20 → ℝ
  loop_var : ℕ
  sum : ℝ

/-- The loop condition for the average calculation program -/
def loop_condition (p : AverageProgram) : Prop :=
  p.loop_var ≤ 20

/-- The correctness of the loop condition -/
theorem loop_condition_correct (p : AverageProgram) : 
  loop_condition p ↔ p.loop_var ≤ 20 := by sorry

end NUMINAMATH_CALUDE_loop_condition_correct_l65_6515


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l65_6509

theorem consecutive_integers_sum_of_squares : 
  ∃ (b : ℕ), 
    (b > 0) ∧ 
    ((b - 1) * b * (b + 1) = 12 * (3 * b)) → 
    ((b - 1)^2 + b^2 + (b + 1)^2 = 110) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l65_6509


namespace NUMINAMATH_CALUDE_car_interval_duration_l65_6585

/-- Proves that the duration of each interval is 1/7.5 hours given the conditions of the car problem -/
theorem car_interval_duration 
  (initial_speed : ℝ) 
  (speed_decrease : ℝ) 
  (fifth_interval_distance : ℝ) 
  (h1 : initial_speed = 45)
  (h2 : speed_decrease = 3)
  (h3 : fifth_interval_distance = 4.4)
  : ∃ (t : ℝ), t = 1 / 7.5 ∧ fifth_interval_distance = (initial_speed - 4 * speed_decrease) * t :=
sorry

end NUMINAMATH_CALUDE_car_interval_duration_l65_6585


namespace NUMINAMATH_CALUDE_partner_profit_percentage_l65_6567

theorem partner_profit_percentage (total_profit : ℝ) (majority_owner_percentage : ℝ) 
  (combined_amount : ℝ) (num_partners : ℕ) :
  total_profit = 80000 →
  majority_owner_percentage = 0.25 →
  combined_amount = 50000 →
  num_partners = 4 →
  let remaining_profit := total_profit * (1 - majority_owner_percentage)
  let partner_share := (combined_amount - total_profit * majority_owner_percentage) / 2
  (partner_share / remaining_profit) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_partner_profit_percentage_l65_6567


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l65_6506

theorem geometric_sequence_problem (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 210 * r = b ∧ b * r = 140 / 60) → b = 7 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l65_6506


namespace NUMINAMATH_CALUDE_perpendicular_and_parallel_conditions_l65_6500

def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, -1)

def n (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

theorem perpendicular_and_parallel_conditions :
  (∃ k : ℝ, (n k).1 * (2 * a.1 - b.1) + (n k).2 * (2 * a.2 - b.2) = 0 ∧ k = 5/3) ∧
  (∃ k : ℝ, ∃ t : ℝ, n k = t • (c.1 + k * b.1, c.2 + k * b.2) ∧ k = -1/3) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_and_parallel_conditions_l65_6500


namespace NUMINAMATH_CALUDE_soccer_camp_afternoon_attendance_l65_6548

theorem soccer_camp_afternoon_attendance (total_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : ∃ soccer_kids : ℕ, soccer_kids = total_kids / 2)
  (h3 : ∃ morning_kids : ℕ, morning_kids = soccer_kids / 4) :
  ∃ afternoon_kids : ℕ, afternoon_kids = 750 :=
by sorry

end NUMINAMATH_CALUDE_soccer_camp_afternoon_attendance_l65_6548


namespace NUMINAMATH_CALUDE_benny_birthday_money_l65_6560

/-- The amount of money Benny spent on baseball gear -/
def money_spent : ℕ := 34

/-- The amount of money Benny had left over -/
def money_left : ℕ := 33

/-- The total amount of money Benny received for his birthday -/
def total_money : ℕ := money_spent + money_left

theorem benny_birthday_money :
  total_money = 67 := by sorry

end NUMINAMATH_CALUDE_benny_birthday_money_l65_6560


namespace NUMINAMATH_CALUDE_polynomial_real_root_l65_6513

theorem polynomial_real_root (b : ℝ) : 
  (∃ x : ℝ, x^4 + b*x^3 - 2*x^2 + b*x + 4 = 0) ↔ 
  (b ≤ -3/2 ∨ b ≥ 3/2) := by
sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l65_6513


namespace NUMINAMATH_CALUDE_solve_system_l65_6571

theorem solve_system (w u y z x : ℤ) 
  (hw : w = 100)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hu : u = y + 5)
  (hx : x = u + 7) : x = 149 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l65_6571


namespace NUMINAMATH_CALUDE_square_less_than_four_implies_less_than_two_l65_6553

theorem square_less_than_four_implies_less_than_two (x : ℝ) : x^2 < 4 → x < 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_four_implies_less_than_two_l65_6553


namespace NUMINAMATH_CALUDE_complex_expression_equals_two_l65_6526

theorem complex_expression_equals_two :
  (2023 - Real.pi) ^ 0 + (1/2)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (π/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_two_l65_6526


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l65_6588

theorem largest_angle_in_triangle (x : ℝ) : 
  x + 50 + 55 = 180 → 
  max x (max 50 55) = 75 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l65_6588


namespace NUMINAMATH_CALUDE_full_time_more_than_three_years_l65_6517

/-- Represents the percentage of associates in each category -/
structure AssociatePercentages where
  secondYear : ℝ
  thirdYear : ℝ
  notFirstYear : ℝ
  partTime : ℝ
  partTimeMoreThanTwoYears : ℝ

/-- Theorem stating the percentage of full-time associates at the firm for more than three years -/
theorem full_time_more_than_three_years 
  (percentages : AssociatePercentages)
  (h1 : percentages.secondYear = 30)
  (h2 : percentages.thirdYear = 20)
  (h3 : percentages.notFirstYear = 60)
  (h4 : percentages.partTime = 10)
  (h5 : percentages.partTimeMoreThanTwoYears = percentages.partTime / 2)
  : ℝ := by
  sorry

#check full_time_more_than_three_years

end NUMINAMATH_CALUDE_full_time_more_than_three_years_l65_6517


namespace NUMINAMATH_CALUDE_polyhedron_volume_l65_6521

/-- Represents a polygon in the figure -/
inductive Polygon
| ScaleneRightTriangle : Polygon
| Rectangle : Polygon
| EquilateralTriangle : Polygon

/-- The figure consisting of multiple polygons -/
structure Figure where
  scaleneTriangles : Fin 3 → Polygon
  rectangles : Fin 3 → Polygon
  equilateralTriangle : Polygon
  scaleneTriangleLegs : ℝ × ℝ
  rectangleDimensions : ℝ × ℝ

/-- The polyhedron formed by folding the figure -/
def Polyhedron (f : Figure) : Type := Unit

/-- The volume of the polyhedron -/
noncomputable def volume (p : Polyhedron f) : ℝ := sorry

/-- The main theorem stating the volume of the polyhedron is 4 -/
theorem polyhedron_volume (f : Figure)
  (h1 : ∀ i, f.scaleneTriangles i = Polygon.ScaleneRightTriangle)
  (h2 : ∀ i, f.rectangles i = Polygon.Rectangle)
  (h3 : f.equilateralTriangle = Polygon.EquilateralTriangle)
  (h4 : f.scaleneTriangleLegs = (1, 2))
  (h5 : f.rectangleDimensions = (1, 2))
  (p : Polyhedron f) :
  volume p = 4 := by sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l65_6521


namespace NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l65_6538

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a + 3}

-- Theorem for part (1)
theorem union_and_intersection :
  (A ∪ B = {x | 1 ≤ x ∧ x < 8}) ∧
  ((Set.univ \ A) ∩ B = {x | 5 ≤ x ∧ x < 8}) := by sorry

-- Theorem for part (2)
theorem range_of_a :
  {a : ℝ | C a ∩ A = C a} = {a : ℝ | 1 ≤ a ∧ a < 2} := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l65_6538


namespace NUMINAMATH_CALUDE_odd_prime_equality_l65_6589

theorem odd_prime_equality (p m : ℕ) (x y : ℕ) 
  (h_prime : Nat.Prime p)
  (h_odd : Odd p)
  (h_x : x > 1)
  (h_y : y > 1)
  (h_eq : (x^p + y^p) / 2 = ((x + y) / 2)^m) :
  m = p := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_equality_l65_6589


namespace NUMINAMATH_CALUDE_cos_four_arccos_one_fourth_l65_6524

theorem cos_four_arccos_one_fourth : 
  Real.cos (4 * Real.arccos (1/4)) = 17/32 := by sorry

end NUMINAMATH_CALUDE_cos_four_arccos_one_fourth_l65_6524


namespace NUMINAMATH_CALUDE_greatest_ACCBA_divisible_by_11_and_3_l65_6508

/-- Represents a five-digit number in the form AC,CBA -/
def ACCBA (A B C : Nat) : Nat := A * 10000 + C * 1000 + C * 100 + B * 10 + A

/-- Checks if the digits A, B, and C are distinct -/
def distinct_digits (A B C : Nat) : Prop := A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- Checks if a number is divisible by both 11 and 3 -/
def divisible_by_11_and_3 (n : Nat) : Prop := n % 11 = 0 ∧ n % 3 = 0

/-- The main theorem statement -/
theorem greatest_ACCBA_divisible_by_11_and_3 :
  ∀ A B C : Nat,
  A < 10 ∧ B < 10 ∧ C < 10 →
  distinct_digits A B C →
  divisible_by_11_and_3 (ACCBA A B C) →
  ACCBA A B C ≤ 95695 :=
sorry

end NUMINAMATH_CALUDE_greatest_ACCBA_divisible_by_11_and_3_l65_6508


namespace NUMINAMATH_CALUDE_abs_x_minus_3_eq_1_implies_5_minus_2x_eq_neg_3_or_1_l65_6545

theorem abs_x_minus_3_eq_1_implies_5_minus_2x_eq_neg_3_or_1 (x : ℝ) :
  |x - 3| = 1 → (5 - 2*x = -3 ∨ 5 - 2*x = 1) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_minus_3_eq_1_implies_5_minus_2x_eq_neg_3_or_1_l65_6545


namespace NUMINAMATH_CALUDE_smallest_m_has_n_14_l65_6535

def is_valid_m (m : ℕ) : Prop :=
  ∃ (n : ℕ) (r : ℝ), 
    n > 0 ∧ 
    r > 0 ∧ 
    r < 1/10000 ∧ 
    m^(1/4 : ℝ) = n + r

theorem smallest_m_has_n_14 : 
  ∃ (m : ℕ), is_valid_m m ∧ 
  (∀ (k : ℕ), k < m → ¬is_valid_m k) ∧
  (∃ (r : ℝ), m^(1/4 : ℝ) = 14 + r ∧ r > 0 ∧ r < 1/10000) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_has_n_14_l65_6535


namespace NUMINAMATH_CALUDE_max_sum_of_two_max_sum_is_zero_l65_6551

def number_set : Finset Int := {1, -1, -2}

theorem max_sum_of_two (a b : Int) (ha : a ∈ number_set) (hb : b ∈ number_set) (hab : a ≠ b) :
  ∃ (x y : Int), x ∈ number_set ∧ y ∈ number_set ∧ x ≠ y ∧ x + y ≥ a + b :=
sorry

theorem max_sum_is_zero :
  ∃ (a b : Int), a ∈ number_set ∧ b ∈ number_set ∧ a ≠ b ∧
  (∀ (x y : Int), x ∈ number_set → y ∈ number_set → x ≠ y → a + b ≥ x + y) ∧
  a + b = 0 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_two_max_sum_is_zero_l65_6551


namespace NUMINAMATH_CALUDE_digit_1983_is_7_l65_6520

/-- Represents the decimal number x as described in the problem -/
def x : ℝ :=
  sorry

/-- Returns the nth digit after the decimal point in x -/
def nthDigit (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the 1983rd digit of x is 7 -/
theorem digit_1983_is_7 : nthDigit 1983 = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_1983_is_7_l65_6520


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l65_6555

theorem absolute_value_inequality (m : ℝ) : 
  (∀ x : ℝ, x + |x - 1| > m) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l65_6555


namespace NUMINAMATH_CALUDE_initial_production_was_200_l65_6580

/-- The number of doors per car -/
def doors_per_car : ℕ := 5

/-- The number of cars cut from production due to metal shortages -/
def cars_cut : ℕ := 50

/-- The fraction of remaining production after pandemic cuts -/
def production_fraction : ℚ := 1/2

/-- The final number of doors produced -/
def final_doors : ℕ := 375

/-- Theorem stating that the initial planned production was 200 cars -/
theorem initial_production_was_200 : 
  ∃ (initial_cars : ℕ), 
    (doors_per_car : ℚ) * production_fraction * (initial_cars - cars_cut) = final_doors ∧ 
    initial_cars = 200 := by
  sorry

end NUMINAMATH_CALUDE_initial_production_was_200_l65_6580


namespace NUMINAMATH_CALUDE_equation_solutions_l65_6577

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 = 4 ↔ x = -1 ∨ x = 3) ∧
  (∀ x : ℝ, x^2 + 3*x - 4 = 0 ↔ x = -4 ∨ x = 1) ∧
  (∀ x : ℝ, 4*x*(2*x + 1) = 3*(2*x + 1) ↔ x = -1/2 ∨ x = 3/4) ∧
  (∀ x : ℝ, 2*x^2 + 5*x - 3 = 0 ↔ x = 1/2 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l65_6577
