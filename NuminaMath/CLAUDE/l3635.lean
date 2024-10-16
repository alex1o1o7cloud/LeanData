import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_parallel_line_intersection_l3635_363564

-- Define a hyperbola
structure Hyperbola where
  A : ℝ
  B : ℝ
  hAB : A ≠ 0 ∧ B ≠ 0

-- Define a line parallel to the asymptote
structure ParallelLine where
  m : ℝ
  hm : m ≠ 0

-- Theorem statement
theorem hyperbola_parallel_line_intersection (h : Hyperbola) (l : ParallelLine) :
  ∃! p : ℝ × ℝ, 
    (h.A * p.1)^2 - (h.B * p.2)^2 = 1 ∧ 
    h.A * p.1 - h.B * p.2 = l.m :=
sorry

end NUMINAMATH_CALUDE_hyperbola_parallel_line_intersection_l3635_363564


namespace NUMINAMATH_CALUDE_complex_fraction_l3635_363521

def z : ℂ := Complex.mk 1 (-2)

theorem complex_fraction :
  (z + 2) / (z - 1) = Complex.mk 1 (3/2) := by sorry

end NUMINAMATH_CALUDE_complex_fraction_l3635_363521


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3635_363542

/-- Given a geometric sequence of positive integers where the first term is 5
    and the fourth term is 405, the fifth term is 405. -/
theorem geometric_sequence_fifth_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 5 →                            -- First term is 5
  a 4 = 405 →                          -- Fourth term is 405
  a 5 = 405 :=                         -- Fifth term is 405
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3635_363542


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3635_363515

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x < -3 ∧ x < 2) ↔ x < -3 := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3635_363515


namespace NUMINAMATH_CALUDE_counterexample_exists_l3635_363577

theorem counterexample_exists : ∃ n : ℕ, 
  2 ∣ n ∧ ¬ Nat.Prime n ∧ Nat.Prime (n - 3) := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3635_363577


namespace NUMINAMATH_CALUDE_gcd_of_75_and_100_l3635_363599

theorem gcd_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_75_and_100_l3635_363599


namespace NUMINAMATH_CALUDE_expected_intersection_value_l3635_363540

/-- A subset of consecutive integers from {1,2,3,4,5,6,7,8} -/
def ConsecutiveSubset := List ℕ

/-- The set of all possible consecutive subsets -/
def allSubsets : Finset ConsecutiveSubset :=
  sorry

/-- The probability of an element x being in a randomly chosen subset -/
def P (x : ℕ) : ℚ :=
  sorry

/-- The expected number of elements in the intersection of three independently chosen subsets -/
def expectedIntersection : ℚ :=
  sorry

theorem expected_intersection_value :
  expectedIntersection = 178 / 243 := by
  sorry

end NUMINAMATH_CALUDE_expected_intersection_value_l3635_363540


namespace NUMINAMATH_CALUDE_cross_section_area_specific_pyramid_l3635_363520

structure RegularTriangularPyramid where
  height : ℝ
  baseSideLength : ℝ

def crossSectionArea (pyramid : RegularTriangularPyramid) (ratio : ℝ) : ℝ :=
  sorry

theorem cross_section_area_specific_pyramid :
  let pyramid := RegularTriangularPyramid.mk 3 6
  let ratio := 8
  crossSectionArea pyramid ratio = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_specific_pyramid_l3635_363520


namespace NUMINAMATH_CALUDE_swim_club_members_l3635_363514

theorem swim_club_members : 
  ∀ (total_members passed_members not_passed_members : ℕ),
  passed_members = (30 * total_members) / 100 →
  not_passed_members = 70 →
  not_passed_members = total_members - passed_members →
  total_members = 100 := by
sorry

end NUMINAMATH_CALUDE_swim_club_members_l3635_363514


namespace NUMINAMATH_CALUDE_solution_of_equation_l3635_363544

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := 3 * a - 4 * b

-- State the theorem
theorem solution_of_equation :
  ∃ x : ℝ, customOp 2 (customOp 2 x) = customOp 1 x ∧ x = 21 / 20 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l3635_363544


namespace NUMINAMATH_CALUDE_elmo_sandwich_jam_cost_l3635_363508

/-- The cost of jam used in Elmo's sandwiches --/
theorem elmo_sandwich_jam_cost :
  ∀ (N B J H : ℕ),
    N > 1 →
    B > 0 →
    J > 0 →
    H > 0 →
    N * (3 * B + 6 * J + 2 * H) = 342 →
    N * J * 6 = 270 := by
  sorry

end NUMINAMATH_CALUDE_elmo_sandwich_jam_cost_l3635_363508


namespace NUMINAMATH_CALUDE_proper_subset_of_singleton_l3635_363541

-- Define the set P
def P : Set ℕ := {0}

-- State the theorem
theorem proper_subset_of_singleton :
  ∀ (S : Set ℕ), S ⊂ P → S = ∅ :=
sorry

end NUMINAMATH_CALUDE_proper_subset_of_singleton_l3635_363541


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3635_363587

-- Define the sets P and M
def P : Set ℝ := {x | x^2 = 1}
def M (a : ℝ) : Set ℝ := {x | a * x = 1}

-- Define the set of possible values for a
def A : Set ℝ := {1, -1, 0}

-- Statement to prove
theorem possible_values_of_a (a : ℝ) : M a ⊆ P → a ∈ A := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3635_363587


namespace NUMINAMATH_CALUDE_complex_power_sum_l3635_363519

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^600 + 1/z^600 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3635_363519


namespace NUMINAMATH_CALUDE_exists_unsolvable_configuration_l3635_363560

/-- Represents a chessboard with integers -/
def Chessboard := Matrix (Fin 2018) (Fin 2019) ℤ

/-- Represents a set of selected cells on the chessboard -/
def SelectedCells := Set (Fin 2018 × Fin 2019)

/-- Performs one step of the operation on the chessboard -/
def perform_operation (board : Chessboard) (selected : SelectedCells) : Chessboard :=
  sorry

/-- Checks if all numbers on the board are equal -/
def all_equal (board : Chessboard) : Prop :=
  sorry

/-- Theorem stating that there exists a chessboard configuration where it's impossible to make all numbers equal -/
theorem exists_unsolvable_configuration :
  ∃ (initial_board : Chessboard),
    ∀ (operations : List SelectedCells),
      ¬(all_equal (operations.foldl perform_operation initial_board)) :=
sorry

end NUMINAMATH_CALUDE_exists_unsolvable_configuration_l3635_363560


namespace NUMINAMATH_CALUDE_binomial_30_3_l3635_363586

theorem binomial_30_3 : Nat.choose 30 3 = 12180 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l3635_363586


namespace NUMINAMATH_CALUDE_sum_a1_a5_l3635_363556

/-- For a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℕ := n^2 + 1

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sum_a1_a5 : a 1 + a 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_a1_a5_l3635_363556


namespace NUMINAMATH_CALUDE_min_distance_sum_to_line_l3635_363535

/-- The minimum distance sum from two fixed points to a line --/
theorem min_distance_sum_to_line :
  let P : ℝ × ℝ := (1, 3)
  let Q : ℝ × ℝ := (-1, 2)
  let line := {M : ℝ × ℝ | M.1 - M.2 + 1 = 0}
  ∃ (min_dist : ℝ), min_dist = 3 ∧
    ∀ M ∈ line, Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) +
                 Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ≥ min_dist :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_sum_to_line_l3635_363535


namespace NUMINAMATH_CALUDE_single_digit_integer_equation_l3635_363539

theorem single_digit_integer_equation : ∃ (x a y z b : ℕ),
  (0 < x ∧ x < 10) ∧
  (0 < a ∧ a < 10) ∧
  (0 < y ∧ y < 10) ∧
  (0 < z ∧ z < 10) ∧
  (0 < b ∧ b < 10) ∧
  (x = a / 6) ∧
  (z = b / 6) ∧
  (y = (a + b) % 5) ∧
  (100 * x + 10 * y + z = 121) :=
by
  sorry

end NUMINAMATH_CALUDE_single_digit_integer_equation_l3635_363539


namespace NUMINAMATH_CALUDE_inequality_proof_l3635_363532

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 = x + y + z) :
  x + y + z + 3 ≥ 6 * ((xy + yz + zx) / 3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3635_363532


namespace NUMINAMATH_CALUDE_train_length_l3635_363574

/-- Given a train with a speed of 125.99999999999999 km/h that can cross an electric pole in 20 seconds,
    prove that the length of the train is 700 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 125.99999999999999 → 
  time = 20 → 
  length = speed * (1000 / 3600) * time → 
  length = 700 := by sorry

end NUMINAMATH_CALUDE_train_length_l3635_363574


namespace NUMINAMATH_CALUDE_candle_burning_theorem_l3635_363522

theorem candle_burning_theorem (ℓ : ℝ) (h : ℓ > 0) : 
  let t := 180 -- 3 hours in minutes
  let f := λ x : ℝ => ℓ * (1 - x / 240) -- stub length of 4-hour candle
  let g := λ x : ℝ => ℓ * (1 - x / 360) -- stub length of 6-hour candle
  g t = 3 * f t := by sorry

end NUMINAMATH_CALUDE_candle_burning_theorem_l3635_363522


namespace NUMINAMATH_CALUDE_fourth_root_difference_l3635_363547

theorem fourth_root_difference : (81 : ℝ) ^ (1/4) - (1296 : ℝ) ^ (1/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_difference_l3635_363547


namespace NUMINAMATH_CALUDE_black_white_difference_l3635_363592

/-- Represents the number of pieces in a box -/
structure PieceCount where
  black : ℕ
  white : ℕ

/-- The condition of the problem -/
def satisfiesCondition (p : PieceCount) : Prop :=
  (p.black - 1) / p.white = 9 / 7 ∧
  p.black / (p.white - 1) = 7 / 5

/-- The theorem to be proved -/
theorem black_white_difference (p : PieceCount) :
  satisfiesCondition p → p.black - p.white = 7 := by
  sorry

end NUMINAMATH_CALUDE_black_white_difference_l3635_363592


namespace NUMINAMATH_CALUDE_rectangle_area_l3635_363593

theorem rectangle_area (w : ℝ) (l : ℝ) (A : ℝ) (P : ℝ) : 
  l = w + 6 →
  A = w * l →
  P = 2 * (w + l) →
  A = 2 * P →
  w = 3 →
  A = 27 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3635_363593


namespace NUMINAMATH_CALUDE_constant_radius_is_cylinder_l3635_363597

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) (c : ℝ) : Prop :=
  ∀ p : CylindricalPoint, p ∈ S ↔ p.r = c

/-- The set of points satisfying r = c -/
def ConstantRadiusSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Theorem: The set of points satisfying r = c forms a cylinder -/
theorem constant_radius_is_cylinder (c : ℝ) :
    IsCylinder (ConstantRadiusSet c) c := by
  sorry


end NUMINAMATH_CALUDE_constant_radius_is_cylinder_l3635_363597


namespace NUMINAMATH_CALUDE_uncool_parents_count_l3635_363563

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ)
  (h1 : total = 50)
  (h2 : cool_dads = 25)
  (h3 : cool_moms = 30)
  (h4 : both_cool = 15) :
  total - (cool_dads - both_cool + cool_moms - both_cool + both_cool) = 10 := by
  sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l3635_363563


namespace NUMINAMATH_CALUDE_area_ratio_concentric_circles_l3635_363562

/-- Given two concentric circles where a 60-degree arc on the smaller circle
    has the same length as a 30-degree arc on the larger circle,
    the ratio of the area of the smaller circle to the area of the larger circle is 1/4. -/
theorem area_ratio_concentric_circles (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * Real.pi * r₁) = 30 / 360 * (2 * Real.pi * r₂)) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_concentric_circles_l3635_363562


namespace NUMINAMATH_CALUDE_two_distinct_roots_l3635_363523

/-- A quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (2*m + 1)*x + m^2 - 1

/-- The discriminant of the quadratic function f -/
def discriminant (m : ℝ) : ℝ := (2*m + 1)^2 - 4*(m^2 - 1)

/-- Theorem: The quadratic function f has two distinct real roots if and only if m > -5/4 -/
theorem two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ m > -5/4 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l3635_363523


namespace NUMINAMATH_CALUDE_even_painted_faces_count_l3635_363571

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Represents a cube cut from the block -/
structure Cube where
  x : Nat
  y : Nat
  z : Nat

/-- Returns the number of painted faces for a cube in the given position -/
def numPaintedFaces (b : Block) (c : Cube) : Nat :=
  sorry

/-- Returns true if the number is even -/
def isEven (n : Nat) : Bool :=
  sorry

/-- Counts the number of cubes with an even number of painted faces -/
def countEvenPaintedFaces (b : Block) : Nat :=
  sorry

/-- Theorem: In a 6x3x2 inch block painted on all sides and cut into 1 inch cubes,
    the number of cubes with an even number of painted faces is 20 -/
theorem even_painted_faces_count (b : Block) :
  b.length = 6 → b.width = 3 → b.height = 2 →
  countEvenPaintedFaces b = 20 :=
by sorry

end NUMINAMATH_CALUDE_even_painted_faces_count_l3635_363571


namespace NUMINAMATH_CALUDE_power_multiplication_l3635_363507

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3635_363507


namespace NUMINAMATH_CALUDE_expression_evaluation_l3635_363591

theorem expression_evaluation (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 + 1 / a) / ((a^2 - 1) / a) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3635_363591


namespace NUMINAMATH_CALUDE_g_nested_3_l3635_363573

def g (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else x^2 + 2

theorem g_nested_3 : g (g (g (g 3))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_nested_3_l3635_363573


namespace NUMINAMATH_CALUDE_angle_inequality_l3635_363513

theorem angle_inequality (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 2 → x^2 * Real.cos θ - 2*x*(1 - x) + (2 - x)^2 * Real.sin θ > 0) →
  Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_l3635_363513


namespace NUMINAMATH_CALUDE_race_speed_ratio_l3635_363505

theorem race_speed_ratio (total_distance : ℝ) (head_start : ℝ) (speed_A : ℝ) (speed_B : ℝ) 
  (h1 : total_distance = 128)
  (h2 : head_start = 64)
  (h3 : total_distance / speed_A = (total_distance - head_start) / speed_B) :
  speed_A / speed_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l3635_363505


namespace NUMINAMATH_CALUDE_fraction_simplification_l3635_363590

theorem fraction_simplification :
  (20 : ℚ) / 21 * 35 / 54 * 63 / 50 = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3635_363590


namespace NUMINAMATH_CALUDE_stratified_sample_imported_count_l3635_363596

/-- Represents the number of marker lights in a population -/
structure MarkerLightPopulation where
  total : ℕ
  coDeveloped : ℕ
  domestic : ℕ
  h_sum : total = imported + coDeveloped + domestic

/-- Represents a stratified sample of marker lights -/
structure StratifiedSample where
  populationSize : ℕ
  sampleSize : ℕ

/-- Theorem stating that the number of imported marker lights in a stratified sample
    is proportional to their representation in the population -/
theorem stratified_sample_imported_count 
  (population : MarkerLightPopulation)
  (sample : StratifiedSample)
  (h_pop_size : sample.populationSize = population.total)
  (h_imported : sample.importedInPopulation = population.imported)
  (h_sample_size : sample.sampleSize = 20)
  (h_stratified : sample.importedInSample * population.total = 
                  population.imported * sample.sampleSize) :
  sample.importedInSample = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_imported_count_l3635_363596


namespace NUMINAMATH_CALUDE_probability_qualified_bulb_factory_A_l3635_363510

/-- The probability of buying a qualified light bulb produced by Factory A from the market -/
theorem probability_qualified_bulb_factory_A 
  (factory_A_production_rate : ℝ) 
  (factory_A_pass_rate : ℝ) 
  (h1 : factory_A_production_rate = 0.7)
  (h2 : factory_A_pass_rate = 0.95) : 
  factory_A_production_rate * factory_A_pass_rate = 0.665 := by
sorry

end NUMINAMATH_CALUDE_probability_qualified_bulb_factory_A_l3635_363510


namespace NUMINAMATH_CALUDE_average_weight_increase_l3635_363558

/-- Proves that replacing a person weighing 70 kg with a person weighing 90 kg
    in a group of 8 people increases the average weight by 2.5 kg. -/
theorem average_weight_increase
  (n : ℕ)
  (old_weight new_weight : ℝ)
  (h_n : n = 8)
  (h_old : old_weight = 70)
  (h_new : new_weight = 90) :
  (new_weight - old_weight) / n = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3635_363558


namespace NUMINAMATH_CALUDE_fortieth_number_is_twelve_l3635_363567

/-- Represents the value in a specific position of the arrangement --/
def arrangementValue (position : ℕ) : ℕ :=
  let rowNum : ℕ := (position - 1).sqrt + 1
  2 * rowNum

/-- The theorem stating that the 40th number in the arrangement is 12 --/
theorem fortieth_number_is_twelve : arrangementValue 40 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fortieth_number_is_twelve_l3635_363567


namespace NUMINAMATH_CALUDE_more_numbers_with_one_l3635_363554

/-- The set of digits from 1 to 9 -/
def Digits : Finset ℕ := Finset.range 9

/-- The number of six-digit numbers with digits in ascending order -/
def TotalNumbers : ℕ := Finset.card (Digits.powersetCard 6)

/-- The number of six-digit numbers with digits in ascending order that contain 1 -/
def NumbersWithOne : ℕ := Finset.card ((Digits.erase 1).powersetCard 5)

/-- The number of six-digit numbers with digits in ascending order that do not contain 1 -/
def NumbersWithoutOne : ℕ := TotalNumbers - NumbersWithOne

/-- Theorem: There are 28 more six-digit numbers with digits in ascending order
    that contain the digit 1 than those that do not -/
theorem more_numbers_with_one : NumbersWithOne - NumbersWithoutOne = 28 := by
  sorry

end NUMINAMATH_CALUDE_more_numbers_with_one_l3635_363554


namespace NUMINAMATH_CALUDE_first_runner_time_l3635_363578

/-- Represents a 600-meter relay race with three runners -/
structure RelayRace where
  runner1_time : ℝ
  runner2_time : ℝ
  runner3_time : ℝ

/-- The conditions of the specific relay race -/
def race_conditions (race : RelayRace) : Prop :=
  race.runner2_time = race.runner1_time + 2 ∧
  race.runner3_time = race.runner1_time - 3 ∧
  race.runner1_time + race.runner2_time + race.runner3_time = 71

/-- Theorem stating that given the race conditions, the first runner's time is 24 seconds -/
theorem first_runner_time (race : RelayRace) :
  race_conditions race → race.runner1_time = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_first_runner_time_l3635_363578


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l3635_363524

theorem sqrt_equation_solutions (x : ℝ) :
  (Real.sqrt (5 * x - 4) + 12 / Real.sqrt (5 * x - 4) = 8) ↔ (x = 8 ∨ x = 8/5) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l3635_363524


namespace NUMINAMATH_CALUDE_problem_solution_l3635_363585

theorem problem_solution (a b : ℝ) (h : |a - 1| + |b + 2| = 0) : 
  (a + b)^2013 + |b| = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3635_363585


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3635_363576

theorem cone_lateral_surface_area 
  (r h : ℝ) 
  (hr : r = 4) 
  (hh : h = 3) : 
  r * (Real.sqrt (r^2 + h^2)) * π = 20 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3635_363576


namespace NUMINAMATH_CALUDE_x_minus_y_value_l3635_363552

theorem x_minus_y_value (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 4) (h3 : x < y) :
  x - y = -7 ∨ x - y = -1 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l3635_363552


namespace NUMINAMATH_CALUDE_smallest_k_for_divisible_difference_l3635_363543

theorem smallest_k_for_divisible_difference : ∃ (k : ℕ), k > 0 ∧
  (∀ (M : Finset ℕ), M ⊆ Finset.range 20 → M.card ≥ k →
    ∃ (a b c d : ℕ), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      20 ∣ (a - b + c - d)) ∧
  (∀ (k' : ℕ), k' < k →
    ∃ (M : Finset ℕ), M ⊆ Finset.range 20 ∧ M.card = k' ∧
      ∀ (a b c d : ℕ), a ∈ M → b ∈ M → c ∈ M → d ∈ M →
        a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
        ¬(20 ∣ (a - b + c - d))) ∧
  k = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_divisible_difference_l3635_363543


namespace NUMINAMATH_CALUDE_upper_limit_of_set_A_l3635_363537

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def SetA : Set ℕ := {n : ℕ | isPrime n ∧ n > 15}

theorem upper_limit_of_set_A (lower_bound : ℕ) (h1 : lower_bound ∈ SetA) 
  (h2 : ∀ x ∈ SetA, x ≥ lower_bound) 
  (h3 : ∃ upper_bound : ℕ, upper_bound ∈ SetA ∧ upper_bound - lower_bound = 14) :
  ∃ max_element : ℕ, max_element ∈ SetA ∧ max_element = 31 :=
sorry

end NUMINAMATH_CALUDE_upper_limit_of_set_A_l3635_363537


namespace NUMINAMATH_CALUDE_change5_is_census_change5_most_suitable_for_census_l3635_363550

/-- Represents a survey method -/
inductive SurveyMethod
  | Sample
  | Census

/-- Represents a survey target -/
structure SurveyTarget where
  name : String
  method : SurveyMethod

/-- Definition of a census -/
def isCensus (target : SurveyTarget) : Prop :=
  target.method = SurveyMethod.Census

/-- The "Chang'e 5" probe components survey -/
def change5Survey : SurveyTarget :=
  { name := "All components of the Chang'e 5 probe"
    method := SurveyMethod.Census }

/-- Theorem: The "Chang'e 5" probe components survey is a census -/
theorem change5_is_census : isCensus change5Survey := by
  sorry

/-- Theorem: The "Chang'e 5" probe components survey is the most suitable for a census -/
theorem change5_most_suitable_for_census (other : SurveyTarget) :
    isCensus other → other = change5Survey := by
  sorry

end NUMINAMATH_CALUDE_change5_is_census_change5_most_suitable_for_census_l3635_363550


namespace NUMINAMATH_CALUDE_monica_reading_plan_l3635_363583

/-- The number of books Monica read last year -/
def books_last_year : ℕ := 16

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 2 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 2 * books_this_year + 5

theorem monica_reading_plan : books_next_year = 69 := by
  sorry

end NUMINAMATH_CALUDE_monica_reading_plan_l3635_363583


namespace NUMINAMATH_CALUDE_gcd_2505_7350_l3635_363569

theorem gcd_2505_7350 : Nat.gcd 2505 7350 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2505_7350_l3635_363569


namespace NUMINAMATH_CALUDE_hall_length_is_six_l3635_363517

/-- A rectangular hall with given properties --/
structure Hall where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ
  area_floor_ceiling : ℝ
  area_walls : ℝ

/-- The theorem stating the conditions and the result to be proved --/
theorem hall_length_is_six (h : Hall) 
  (h_width : h.width = 6)
  (h_volume : h.volume = 108)
  (h_areas : h.area_floor_ceiling = h.area_walls)
  (h_floor_ceiling : h.area_floor_ceiling = 2 * h.length * h.width)
  (h_walls : h.area_walls = 2 * h.length * h.height + 2 * h.width * h.height)
  (h_volume_calc : h.volume = h.length * h.width * h.height) :
  h.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_hall_length_is_six_l3635_363517


namespace NUMINAMATH_CALUDE_cottage_pie_mince_usage_l3635_363527

/-- Given information about a school cafeteria's use of ground mince for lasagnas and cottage pies,
    prove that each cottage pie uses 3 pounds of ground mince. -/
theorem cottage_pie_mince_usage
  (total_dishes : Nat)
  (lasagna_count : Nat)
  (cottage_pie_count : Nat)
  (total_mince : Nat)
  (mince_per_lasagna : Nat)
  (h1 : total_dishes = lasagna_count + cottage_pie_count)
  (h2 : total_dishes = 100)
  (h3 : lasagna_count = 100)
  (h4 : cottage_pie_count = 100)
  (h5 : total_mince = 500)
  (h6 : mince_per_lasagna = 2) :
  (total_mince - lasagna_count * mince_per_lasagna) / cottage_pie_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_cottage_pie_mince_usage_l3635_363527


namespace NUMINAMATH_CALUDE_jack_and_jill_speed_l3635_363594

theorem jack_and_jill_speed : 
  ∀ x : ℝ, 
    x ≠ -2 →
    (x^2 - 7*x - 12 = (x^2 - 3*x - 10) / (x + 2)) →
    (x^2 - 7*x - 12 = 2) := by
  sorry

end NUMINAMATH_CALUDE_jack_and_jill_speed_l3635_363594


namespace NUMINAMATH_CALUDE_systematic_sampling_correct_l3635_363589

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  populationSize : ℕ
  sampleSize : ℕ
  firstItem : ℕ
  samplingInterval : ℕ

/-- Generates the sample based on the systematic sampling scheme -/
def generateSample (s : SystematicSampling) : List ℕ :=
  List.range s.sampleSize |>.map (fun i => s.firstItem + i * s.samplingInterval)

/-- Theorem: The systematic sampling for the given problem yields the correct sample -/
theorem systematic_sampling_correct :
  let s : SystematicSampling := {
    populationSize := 50,
    sampleSize := 5,
    firstItem := 7,
    samplingInterval := 10
  }
  generateSample s = [7, 17, 27, 37, 47] := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_correct_l3635_363589


namespace NUMINAMATH_CALUDE_triangle_area_l3635_363516

theorem triangle_area (a b c : ℝ) (A : ℝ) :
  A = π / 3 →  -- 60° in radians
  a = Real.sqrt 3 →
  b + c = 3 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3635_363516


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3635_363528

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ → ℝ) 
  (h1 : InverselyProportional (x 40) (y 5))
  (h2 : x 40 = 40)
  (h3 : y 5 = 5) :
  y 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3635_363528


namespace NUMINAMATH_CALUDE_ticket_sales_total_l3635_363500

/-- Calculates the total amount collected from ticket sales given the following conditions:
  * Adult ticket cost is $12
  * Child ticket cost is $4
  * Total number of tickets sold is 130
  * Number of adult tickets sold is 40
-/
theorem ticket_sales_total (adult_cost child_cost total_tickets adult_tickets : ℕ) : 
  adult_cost = 12 →
  child_cost = 4 →
  total_tickets = 130 →
  adult_tickets = 40 →
  adult_cost * adult_tickets + child_cost * (total_tickets - adult_tickets) = 840 :=
by
  sorry

#check ticket_sales_total

end NUMINAMATH_CALUDE_ticket_sales_total_l3635_363500


namespace NUMINAMATH_CALUDE_midpoint_property_l3635_363509

/-- Given two points D and E, if F is their midpoint, then 3x - 5y = 9 --/
theorem midpoint_property (D E F : ℝ × ℝ) : 
  D = (30, 10) → 
  E = (6, 8) → 
  F.1 = (D.1 + E.1) / 2 → 
  F.2 = (D.2 + E.2) / 2 → 
  3 * F.1 - 5 * F.2 = 9 := by
sorry

end NUMINAMATH_CALUDE_midpoint_property_l3635_363509


namespace NUMINAMATH_CALUDE_existence_implies_upper_bound_l3635_363588

theorem existence_implies_upper_bound (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 3 ∧ x^2 - 3*x - a > 0) → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_existence_implies_upper_bound_l3635_363588


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3635_363502

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m) ∧
  n = 59 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3635_363502


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3635_363581

theorem polynomial_expansion (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + 3*x + 1) = x^4 - 5*x^2 + 6*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3635_363581


namespace NUMINAMATH_CALUDE_quadratic_equation_constant_l3635_363549

theorem quadratic_equation_constant (C : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    x₁ > x₂ ∧ 
    x₁ - x₂ = 5.5 ∧ 
    2 * x₁^2 + 5 * x₁ - C = 0 ∧ 
    2 * x₂^2 + 5 * x₂ - C = 0) → 
  C = -12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_constant_l3635_363549


namespace NUMINAMATH_CALUDE_roses_in_garden_l3635_363501

theorem roses_in_garden (rows : ℕ) (roses_per_row : ℕ) 
  (red_fraction : ℚ) (white_fraction : ℚ) :
  rows = 10 →
  roses_per_row = 20 →
  red_fraction = 1/2 →
  white_fraction = 3/5 →
  (rows * roses_per_row * (1 - red_fraction) * (1 - white_fraction) : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_garden_l3635_363501


namespace NUMINAMATH_CALUDE_real_roots_iff_k_geq_quarter_l3635_363555

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ :=
  (k - 1)^2 * x^2 + (2*k + 1) * x + 1

-- Theorem statement
theorem real_roots_iff_k_geq_quarter :
  ∀ k : ℝ, (∃ x : ℝ, quadratic_equation k x = 0) ↔ k ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_iff_k_geq_quarter_l3635_363555


namespace NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3635_363546

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the slope of the tangent at a point
def tangent_slope (x : ℝ) : ℝ := 8 * x

-- Define the condition for perpendicular tangents
def perpendicular_tangents (a b : ℝ) : Prop :=
  tangent_slope a * tangent_slope b = -1

-- Define the y-coordinate of the intersection point
def intersection_y (a b : ℝ) : ℝ := 4 * a * b

-- Theorem statement
theorem intersection_point_y_coordinate 
  (a b : ℝ) 
  (ha : parabola a = 4 * a^2) 
  (hb : parabola b = 4 * b^2) 
  (hperp : perpendicular_tangents a b) :
  intersection_y a b = -1/4 := by sorry

end NUMINAMATH_CALUDE_intersection_point_y_coordinate_l3635_363546


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problems_l3635_363503

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_problems
  (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0) (h_arith : arithmetic_sequence a d)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_m : ∃ m : ℕ, a m = 8)
  (S : ℕ → ℝ)
  (h_S : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2)
  (h_S3 : S 3 = 9)
  (h_S6 : S 6 = 36) :
  (∃ m : ℕ, a m = 8 ∧ m = 8) ∧
  (a 7 + a 8 + a 9 = 45) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problems_l3635_363503


namespace NUMINAMATH_CALUDE_min_value_function_l3635_363580

theorem min_value_function (a b : ℝ) (h : a + b = 1) :
  ∃ (min : ℝ), min = 5 * Real.sqrt 11 ∧
  ∀ (x y : ℝ), x + y = 1 →
  3 * Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt (40 + 9 * y^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_function_l3635_363580


namespace NUMINAMATH_CALUDE_roses_picked_l3635_363533

theorem roses_picked (tulips flowers_used extra_flowers : ℕ) : 
  tulips = 4 →
  flowers_used = 11 →
  extra_flowers = 4 →
  flowers_used + extra_flowers - tulips = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_roses_picked_l3635_363533


namespace NUMINAMATH_CALUDE_rectangle_diagonal_corners_l3635_363506

/-- Represents a domino on a rectangular grid -/
structure Domino where
  x : ℕ
  y : ℕ
  horizontal : Bool

/-- Represents a diagonal in a domino -/
structure Diagonal where
  domino : Domino
  startCorner : Bool  -- true if the diagonal starts at the top-left or bottom-right corner

/-- Represents a rectangular grid filled with dominoes -/
structure RectangularGrid where
  width : ℕ
  height : ℕ
  dominoes : List Domino
  diagonals : List Diagonal

/-- Check if two diagonals have common endpoints -/
def diagonalsShareEndpoint (d1 d2 : Diagonal) : Bool := sorry

/-- Check if a point is a corner of the rectangle -/
def isRectangleCorner (x y : ℕ) (grid : RectangularGrid) : Bool := sorry

/-- Check if a point is an endpoint of a diagonal -/
def isDiagonalEndpoint (x y : ℕ) (diagonal : Diagonal) : Bool := sorry

/-- The main theorem -/
theorem rectangle_diagonal_corners (grid : RectangularGrid) :
  (∀ d1 d2 : Diagonal, d1 ∈ grid.diagonals → d2 ∈ grid.diagonals → d1 ≠ d2 → ¬(diagonalsShareEndpoint d1 d2)) →
  (∃! (n : ℕ), n = 2 ∧ 
    ∃ (corners : List (ℕ × ℕ)), corners.length = n ∧
      (∀ (x y : ℕ), (x, y) ∈ corners ↔ 
        (isRectangleCorner x y grid ∧ 
         ∃ d : Diagonal, d ∈ grid.diagonals ∧ isDiagonalEndpoint x y d))) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_corners_l3635_363506


namespace NUMINAMATH_CALUDE_avocado_count_is_two_l3635_363598

/-- Represents the contents and cost of a fruit basket -/
structure FruitBasket where
  banana_count : ℕ
  apple_count : ℕ
  strawberry_count : ℕ
  avocado_count : ℕ
  grapes_count : ℕ
  banana_price : ℚ
  apple_price : ℚ
  strawberry_price : ℚ
  avocado_price : ℚ
  grapes_price : ℚ
  total_cost : ℚ

/-- The fruit basket problem -/
def fruit_basket_problem : FruitBasket :=
  { banana_count := 4
  , apple_count := 3
  , strawberry_count := 24
  , avocado_count := 0  -- This is what we need to prove
  , grapes_count := 1
  , banana_price := 1
  , apple_price := 2
  , strawberry_price := 1/3  -- $4 for 12 strawberries
  , avocado_price := 3
  , grapes_price := 4  -- $2 for half a bunch, so $4 for a full bunch
  , total_cost := 28 }

/-- Theorem stating that the number of avocados in the fruit basket is 2 -/
theorem avocado_count_is_two (fb : FruitBasket) 
  (h1 : fb = fruit_basket_problem) :
  fb.avocado_count = 2 := by
  sorry


end NUMINAMATH_CALUDE_avocado_count_is_two_l3635_363598


namespace NUMINAMATH_CALUDE_pizza_equivalents_theorem_l3635_363572

/-- Calculates the total quantity of pizza equivalents served -/
def total_pizza_equivalents (lunch_pizzas : ℕ) (dinner_pizzas : ℕ) (lunch_calzones : ℕ) : ℕ :=
  lunch_pizzas + dinner_pizzas + (lunch_calzones / 2)

/-- Proves that the total quantity of pizza equivalents served is 17 -/
theorem pizza_equivalents_theorem :
  total_pizza_equivalents 9 6 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_pizza_equivalents_theorem_l3635_363572


namespace NUMINAMATH_CALUDE_helium_pressure_change_l3635_363525

/-- Boyle's law for ideal gases at constant temperature -/
axiom boyles_law {p1 p2 v1 v2 : ℝ} (h : p1 * v1 = p2 * v2) : 
  p1 * v1 = p2 * v2

theorem helium_pressure_change (p1 v1 p2 v2 : ℝ) 
  (h1 : p1 = 4) 
  (h2 : v1 = 3) 
  (h3 : v2 = 6) 
  (h4 : p1 * v1 = p2 * v2) : 
  p2 = 2 := by
  sorry

#check helium_pressure_change

end NUMINAMATH_CALUDE_helium_pressure_change_l3635_363525


namespace NUMINAMATH_CALUDE_planes_parallel_to_line_not_necessarily_parallel_l3635_363582

-- Define a 3D space
variable (V : Type) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Fact (finrank ℝ V = 3)]

-- Define planes and lines in the space
variable (Plane : Type) (Line : Type)

-- Define a relation for a plane being parallel to a line
variable (plane_parallel_to_line : Plane → Line → Prop)

-- Define a relation for two planes being parallel
variable (planes_parallel : Plane → Plane → Prop)

-- Theorem: Two planes parallel to the same line are not necessarily parallel
theorem planes_parallel_to_line_not_necessarily_parallel 
  (P1 P2 : Plane) (L : Line) 
  (h1 : plane_parallel_to_line P1 L) 
  (h2 : plane_parallel_to_line P2 L) :
  ¬ (∀ P1 P2 : Plane, ∀ L : Line, 
    plane_parallel_to_line P1 L → plane_parallel_to_line P2 L → planes_parallel P1 P2) :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_line_not_necessarily_parallel_l3635_363582


namespace NUMINAMATH_CALUDE_solution_set_product_l3635_363548

/-- An odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable (a b : ℝ)
variable (f g : ℝ → ℝ)

/-- The solution set for f(x) > 0 -/
def SolutionSetF : Set ℝ := Set.Ioo (a^2) b

/-- The solution set for g(x) > 0 -/
def SolutionSetG : Set ℝ := Set.Ioo (a^2/2) (b/2)

/-- The conditions given in the problem -/
structure ProblemConditions where
  f_odd : OddFunction f
  g_odd : OddFunction g
  f_solution : SolutionSetF a b = {x | f x > 0}
  g_solution : SolutionSetG a b = {x | g x > 0}
  b_gt_2a_squared : b > 2 * (a^2)

/-- The theorem to be proved -/
theorem solution_set_product (h : ProblemConditions a b f g) :
  {x | f x * g x > 0} = Set.Ioo (-b/2) (-a^2) ∪ Set.Ioo (a^2) (b/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_product_l3635_363548


namespace NUMINAMATH_CALUDE_edward_pipe_per_bolt_l3635_363518

/-- The number of feet of pipe per bolt in Edward's plumbing job -/
def feet_per_bolt (total_pipe_length : ℕ) (washers_used : ℕ) (washers_per_bolt : ℕ) : ℚ :=
  total_pipe_length / (washers_used / washers_per_bolt)

/-- Theorem stating that Edward uses 5 feet of pipe per bolt -/
theorem edward_pipe_per_bolt :
  feet_per_bolt 40 16 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_edward_pipe_per_bolt_l3635_363518


namespace NUMINAMATH_CALUDE_square_root_of_two_squared_equals_two_l3635_363584

theorem square_root_of_two_squared_equals_two :
  (Real.sqrt 2) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_two_squared_equals_two_l3635_363584


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_seven_l3635_363545

theorem smallest_n_divisible_by_seven (n : ℕ) : 
  (n > 50000 ∧ 
   (9 * (n - 2)^6 - n^3 + 20*n - 48) % 7 = 0 ∧
   ∀ m, 50000 < m ∧ m < n → (9 * (m - 2)^6 - m^3 + 20*m - 48) % 7 ≠ 0) →
  n = 50001 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_seven_l3635_363545


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l3635_363561

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 132 → y ∈ Set.Ioo (-12) (-11) := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l3635_363561


namespace NUMINAMATH_CALUDE_complex_division_l3635_363553

theorem complex_division (i : ℂ) (h : i^2 = -1) : 
  (1 - 2*i) / (1 + i) = -1/2 - 3/2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l3635_363553


namespace NUMINAMATH_CALUDE_three_digit_divisibility_l3635_363570

theorem three_digit_divisibility (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : (a + b) % 7 = 0) : (101 * a + 10 * b) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_l3635_363570


namespace NUMINAMATH_CALUDE_profit_and_marginal_profit_maxima_l3635_363534

/-- Revenue function -/
def R (x : ℕ) : ℚ := 3000 * x - 20 * x^2

/-- Cost function -/
def C (x : ℕ) : ℚ := 500 * x + 4000

/-- Profit function -/
def p (x : ℕ) : ℚ := R x - C x

/-- Marginal function -/
def M (f : ℕ → ℚ) (x : ℕ) : ℚ := f (x + 1) - f x

/-- Marginal profit function -/
def Mp (x : ℕ) : ℚ := M p x

theorem profit_and_marginal_profit_maxima :
  (∃ x : ℕ, x ≤ 100 ∧ ∀ y : ℕ, y ≤ 100 → p y ≤ p x) ∧
  (∃ x : ℕ, x ≤ 100 ∧ ∀ y : ℕ, y ≤ 100 → Mp y ≤ Mp x) ∧
  (∀ x : ℕ, x ≤ 100 → p x ≤ 74120) ∧
  (∀ x : ℕ, x ≤ 100 → Mp x ≤ 2440) ∧
  (∃ x : ℕ, x ≤ 100 ∧ p x = 74120) ∧
  (∃ x : ℕ, x ≤ 100 ∧ Mp x = 2440) :=
by sorry

end NUMINAMATH_CALUDE_profit_and_marginal_profit_maxima_l3635_363534


namespace NUMINAMATH_CALUDE_pyarelal_loss_is_900_l3635_363566

/-- Calculates Pyarelal's share of the loss given the investment ratio and total loss -/
def pyarelal_loss (pyarelal_capital : ℚ) (total_loss : ℚ) : ℚ :=
  let ashok_capital := (1 : ℚ) / 9 * pyarelal_capital
  let total_capital := ashok_capital + pyarelal_capital
  let pyarelal_ratio := pyarelal_capital / total_capital
  pyarelal_ratio * total_loss

/-- Theorem stating that Pyarelal's loss is 900 given the conditions of the problem -/
theorem pyarelal_loss_is_900 (pyarelal_capital : ℚ) (h : pyarelal_capital > 0) :
  pyarelal_loss pyarelal_capital 1000 = 900 := by
  sorry

end NUMINAMATH_CALUDE_pyarelal_loss_is_900_l3635_363566


namespace NUMINAMATH_CALUDE_rug_coverage_area_l3635_363529

/-- Given three rugs with specified overlapping areas, calculate the total floor area covered -/
theorem rug_coverage_area (total_rug_area : ℝ) (two_layer_overlap : ℝ) (three_layer_overlap : ℝ) 
  (h1 : total_rug_area = 204)
  (h2 : two_layer_overlap = 24)
  (h3 : three_layer_overlap = 20) :
  total_rug_area - two_layer_overlap - 2 * three_layer_overlap = 140 := by
  sorry

end NUMINAMATH_CALUDE_rug_coverage_area_l3635_363529


namespace NUMINAMATH_CALUDE_contradictory_statements_l3635_363512

theorem contradictory_statements (a b c : ℝ) : 
  (a * b * c ≠ 0) ∧ (a * b * c = 0) ∧ (a * b ≤ 0) → False :=
sorry

end NUMINAMATH_CALUDE_contradictory_statements_l3635_363512


namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l3635_363536

theorem sin_thirty_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l3635_363536


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l3635_363538

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l3635_363538


namespace NUMINAMATH_CALUDE_sugar_mixture_theorem_l3635_363504

/-- Given a bowl of sugar with the following properties:
  * Initially contains 320 grams of pure white sugar
  * Mixture Y is formed by removing x grams of white sugar and adding x grams of brown sugar
  * In Mixture Y, the ratio of white sugar to brown sugar is w:b in lowest terms
  * Mixture Z is formed by removing x grams of Mixture Y and adding x grams of brown sugar
  * In Mixture Z, the ratio of white sugar to brown sugar is 49:15
  Prove that x + w + b = 48 -/
theorem sugar_mixture_theorem (x w b : ℕ) : 
  x > 0 ∧ x < 320 ∧ 
  (320 - x : ℚ) / x = w / b ∧ 
  (320 - x) * (320 - x) / (320 : ℚ) / ((2 * x - x^2 / 320 : ℚ)) = 49 / 15 →
  x + w + b = 48 :=
by sorry

end NUMINAMATH_CALUDE_sugar_mixture_theorem_l3635_363504


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_binomial_expansion_l3635_363595

theorem coefficient_x3y5_in_binomial_expansion :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k : ℕ) * (1 : ℕ)^(8 - k) * (1 : ℕ)^k) = 256 ∧
  (Nat.choose 8 5 : ℕ) = 56 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_binomial_expansion_l3635_363595


namespace NUMINAMATH_CALUDE_tom_siblings_count_l3635_363511

/-- The number of siblings Tom invited -/
def num_siblings : ℕ :=
  let total_plates : ℕ := 144
  let days : ℕ := 4
  let meals_per_day : ℕ := 3
  let plates_per_meal : ℕ := 2
  let tom_and_parents : ℕ := 3
  let plates_per_person : ℕ := days * meals_per_day * plates_per_meal
  let total_people : ℕ := total_plates / plates_per_person
  total_people - tom_and_parents

theorem tom_siblings_count : num_siblings = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_siblings_count_l3635_363511


namespace NUMINAMATH_CALUDE_circle_passes_through_M_and_has_same_center_l3635_363530

-- Define the center of the given circle
def center : ℝ × ℝ := (2, -3)

-- Define the point M
def point_M : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle we want to prove
def circle_equation (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = 25

-- Theorem statement
theorem circle_passes_through_M_and_has_same_center :
  -- The circle passes through point M
  circle_equation point_M.1 point_M.2 ∧
  -- The circle has the same center as (x-2)^2 + (y+3)^2 = 16
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = 25 ↔ circle_equation x y :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_M_and_has_same_center_l3635_363530


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3635_363526

/-- Given two parallel vectors a and b in R², prove that m = -3 --/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (1 + m, 1 - m)
  (∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2) →
  m = -3 := by
sorry


end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3635_363526


namespace NUMINAMATH_CALUDE_sticker_distribution_theorem_l3635_363531

/-- The number of ways to distribute n indistinguishable items into k distinguishable boxes --/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute stickers among sheets --/
def distribute_stickers (total_stickers sheets : ℕ) : ℕ :=
  distribute (total_stickers - sheets) sheets

theorem sticker_distribution_theorem :
  distribute_stickers 12 5 = 330 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_theorem_l3635_363531


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3635_363568

theorem max_value_of_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) :
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 4 ∧ (2*a - b)^2 + (2*b - c)^2 + (2*c - a)^2 > (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2) →
  (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2 ≤ 28 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3635_363568


namespace NUMINAMATH_CALUDE_monday_children_count_l3635_363575

/-- The number of children who went to the zoo on Monday -/
def monday_children : ℕ := sorry

/-- The number of adults who went to the zoo on Monday -/
def monday_adults : ℕ := 5

/-- The number of children who went to the zoo on Tuesday -/
def tuesday_children : ℕ := 4

/-- The number of adults who went to the zoo on Tuesday -/
def tuesday_adults : ℕ := 2

/-- The cost of a child ticket -/
def child_ticket_cost : ℕ := 3

/-- The cost of an adult ticket -/
def adult_ticket_cost : ℕ := 4

/-- The total revenue for both days -/
def total_revenue : ℕ := 61

theorem monday_children_count : 
  monday_children = 7 ∧
  monday_children * child_ticket_cost + 
  monday_adults * adult_ticket_cost +
  tuesday_children * child_ticket_cost +
  tuesday_adults * adult_ticket_cost = total_revenue :=
sorry

end NUMINAMATH_CALUDE_monday_children_count_l3635_363575


namespace NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_l3635_363579

/-- Represents the weight of a packet in pounds -/
def packet_weight : ℚ := 16 + 4 / 16

/-- Represents the number of packets -/
def num_packets : ℕ := 2000

/-- Represents the weight of one ton in pounds -/
def pounds_per_ton : ℕ := 2500

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℚ := (num_packets * packet_weight) / pounds_per_ton

theorem gunny_bag_capacity_is_13 : gunny_bag_capacity = 13 := by
  sorry

end NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_l3635_363579


namespace NUMINAMATH_CALUDE_rahims_book_purchase_l3635_363565

/-- Given Rahim's book purchases, prove the amount paid for books from the first shop -/
theorem rahims_book_purchase (first_shop_books : ℕ) (second_shop_books : ℕ) 
  (second_shop_cost : ℕ) (average_price : ℕ) (h1 : first_shop_books = 42) 
  (h2 : second_shop_books = 22) (h3 : second_shop_cost = 248) (h4 : average_price = 12) :
  (first_shop_books + second_shop_books) * average_price - second_shop_cost = 520 := by
  sorry

#check rahims_book_purchase

end NUMINAMATH_CALUDE_rahims_book_purchase_l3635_363565


namespace NUMINAMATH_CALUDE_project_completion_time_l3635_363557

/-- The number of days A and B take working together -/
def AB_days : ℝ := 2

/-- The number of days B and C take working together -/
def BC_days : ℝ := 4

/-- The number of days C and A take working together -/
def CA_days : ℝ := 2.4

/-- The number of days A takes to complete the project alone -/
def A_days : ℝ := 3

theorem project_completion_time :
  (1 / A_days) * CA_days + (1 / BC_days - (1 / AB_days - 1 / A_days)) * CA_days = 1 :=
sorry

end NUMINAMATH_CALUDE_project_completion_time_l3635_363557


namespace NUMINAMATH_CALUDE_arccos_one_half_eq_pi_third_l3635_363551

theorem arccos_one_half_eq_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_eq_pi_third_l3635_363551


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_seven_l3635_363559

def digits : List Nat := [3, 5, 6, 7]

def isTwoDigitNumber (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def formedFromList (n : Nat) : Prop :=
  ∃ (d1 d2 : Nat), d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = 10 * d1 + d2

theorem smallest_two_digit_multiple_of_seven :
  ∃ (n : Nat), isTwoDigitNumber n ∧ formedFromList n ∧ n % 7 = 0 ∧
  (∀ (m : Nat), isTwoDigitNumber m → formedFromList m → m % 7 = 0 → n ≤ m) ∧
  n = 35 := by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_seven_l3635_363559
