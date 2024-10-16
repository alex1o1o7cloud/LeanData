import Mathlib

namespace NUMINAMATH_CALUDE_power_minus_one_rational_l3595_359579

/-- A complex number with rational real and imaginary parts and unit modulus -/
structure UnitRationalComplex where
  re : ℚ
  im : ℚ
  unit_modulus : re^2 + im^2 = 1

/-- The result of z^(2n) - 1 is rational for any integer n -/
theorem power_minus_one_rational (z : UnitRationalComplex) (n : ℤ) :
  ∃ (q : ℚ), (z.re + z.im * Complex.I)^(2*n) - 1 = q := by
  sorry

end NUMINAMATH_CALUDE_power_minus_one_rational_l3595_359579


namespace NUMINAMATH_CALUDE_power_congruence_l3595_359573

theorem power_congruence (h : 5^500 ≡ 1 [ZMOD 2000]) : 5^15000 ≡ 1 [ZMOD 2000] := by
  sorry

end NUMINAMATH_CALUDE_power_congruence_l3595_359573


namespace NUMINAMATH_CALUDE_final_product_is_twelve_l3595_359547

/-- Represents the count of each number on the blackboard -/
structure BoardState :=
  (ones : ℕ)
  (twos : ℕ)
  (threes : ℕ)
  (fours : ℕ)

/-- The operation performed on the board -/
def performOperation (state : BoardState) : BoardState :=
  { ones := state.ones - 1,
    twos := state.twos - 1,
    threes := state.threes - 1,
    fours := state.fours + 2 }

/-- Predicate to check if an operation can be performed -/
def canPerformOperation (state : BoardState) : Prop :=
  state.ones > 0 ∧ state.twos > 0 ∧ state.threes > 0

/-- Predicate to check if the board is in its final state -/
def isFinalState (state : BoardState) : Prop :=
  ¬(canPerformOperation state) ∧ 
  (state.ones + state.twos + state.threes + state.fours = 3)

/-- The initial state of the board -/
def initialState : BoardState :=
  { ones := 11, twos := 22, threes := 33, fours := 44 }

/-- The main theorem to prove -/
theorem final_product_is_twelve :
  ∃ (finalState : BoardState),
    (isFinalState finalState) ∧
    (finalState.ones * finalState.twos * finalState.threes * finalState.fours = 12) := by
  sorry

end NUMINAMATH_CALUDE_final_product_is_twelve_l3595_359547


namespace NUMINAMATH_CALUDE_derivative_positive_solution_set_l3595_359584

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

def solution_set : Set ℝ := Set.Ioi 2

theorem derivative_positive_solution_set :
  ∀ x > 0, x ∈ solution_set ↔ deriv f x > 0 :=
sorry

end NUMINAMATH_CALUDE_derivative_positive_solution_set_l3595_359584


namespace NUMINAMATH_CALUDE_solution_set_l3595_359570

def f (x : ℝ) := abs x + x^2 + 2

theorem solution_set (x : ℝ) :
  f (2*x - 1) > f (3 - x) ↔ x < -2 ∨ x > 4/3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l3595_359570


namespace NUMINAMATH_CALUDE_smallest_modulus_of_z_l3595_359583

theorem smallest_modulus_of_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z + 3*I) = 15) :
  Complex.abs z ≥ 8/5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_modulus_of_z_l3595_359583


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l3595_359577

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 12)^2 + (y - 3)^2 = 25
def C2 (x y : ℝ) : Prop := (x + 9)^2 + (y + 4)^2 = 49

-- Define the centers and radii
def center1 : ℝ × ℝ := (12, 3)
def center2 : ℝ × ℝ := (-9, -4)
def radius1 : ℝ := 5
def radius2 : ℝ := 7

-- Theorem statement
theorem shortest_tangent_length :
  ∃ (R S : ℝ × ℝ),
    C1 R.1 R.2 ∧ C2 S.1 S.2 ∧
    (∀ (P Q : ℝ × ℝ), C1 P.1 P.2 → C2 Q.1 Q.2 →
      Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 70 :=
by sorry


end NUMINAMATH_CALUDE_shortest_tangent_length_l3595_359577


namespace NUMINAMATH_CALUDE_gumball_probability_l3595_359550

theorem gumball_probability (orange green yellow : ℕ) 
  (h_orange : orange = 10)
  (h_green : green = 6)
  (h_yellow : yellow = 9) :
  let total := orange + green + yellow
  let p_first_orange := orange / total
  let p_second_not_orange := (green + yellow) / (total - 1)
  let p_third_orange := (orange - 1) / (total - 2)
  p_first_orange * p_second_not_orange * p_third_orange = 9 / 92 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l3595_359550


namespace NUMINAMATH_CALUDE_parallelogram_vertex_product_l3595_359578

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parallelogram defined by its four vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if two points are diagonally opposite in a parallelogram -/
def diagonallyOpposite (p : Parallelogram) (p1 p2 : Point) : Prop :=
  (p1 = p.A ∧ p2 = p.C) ∨ (p1 = p.B ∧ p2 = p.D) ∨ (p1 = p.C ∧ p2 = p.A) ∨ (p1 = p.D ∧ p2 = p.B)

/-- The main theorem -/
theorem parallelogram_vertex_product (p : Parallelogram) :
  p.A = Point.mk (-1) 3 →
  p.B = Point.mk 2 (-1) →
  p.D = Point.mk 7 6 →
  diagonallyOpposite p p.A p.D →
  p.C.x * p.C.y = 40 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_product_l3595_359578


namespace NUMINAMATH_CALUDE_eggs_per_group_l3595_359567

/-- Given 9 eggs split into 3 groups, prove that there are 3 eggs in each group. -/
theorem eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) (h1 : total_eggs = 9) (h2 : num_groups = 3) :
  total_eggs / num_groups = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_group_l3595_359567


namespace NUMINAMATH_CALUDE_coin_triangle_proof_l3595_359576

def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem coin_triangle_proof (N : ℕ) (h : triangle_sum N = 2016) :
  sum_of_digits N = 9 := by
  sorry

end NUMINAMATH_CALUDE_coin_triangle_proof_l3595_359576


namespace NUMINAMATH_CALUDE_h_equation_l3595_359589

theorem h_equation (x : ℝ) (h : ℝ → ℝ) :
  (4 * x^4 + 11 * x^3 + h x = 10 * x^3 - x^2 + 4 * x - 7) →
  h x = -4 * x^4 - x^3 - x^2 + 4 * x - 7 := by
sorry

end NUMINAMATH_CALUDE_h_equation_l3595_359589


namespace NUMINAMATH_CALUDE_minimal_poster_area_l3595_359558

theorem minimal_poster_area (m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) (h_m_ge_n : m ≥ n) : 
  ∃ (posters : Finset (ℕ × ℕ)), 
    (Finset.card posters = m * n) ∧ 
    (∀ (k l : ℕ), (k, l) ∈ posters → 1 ≤ k ∧ k ≤ m ∧ 1 ≤ l ∧ l ≤ n) →
    (minimal_area : ℕ) = m * (n * (n + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_minimal_poster_area_l3595_359558


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l3595_359504

theorem shaded_area_between_circles (R : ℝ) (r : ℝ) : 
  R = 10 → r = 4 → π * R^2 - 2 * π * r^2 = 68 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l3595_359504


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l3595_359545

theorem quadratic_root_condition (a : ℝ) :
  (∃ r₁ r₂ : ℝ, r₁ > 0 ∧ r₂ < 0 ∧ r₁^2 - r₁ + (a - 4) = 0 ∧ r₂^2 - r₂ + (a - 4) = 0) →
  a < 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l3595_359545


namespace NUMINAMATH_CALUDE_platform_length_l3595_359566

/-- Given a train with speed 54 km/hr passing a platform in 32 seconds
    and passing a man standing on the platform in 20 seconds,
    prove that the length of the platform is 180 meters. -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) :
  train_speed = 54 →
  platform_time = 32 →
  man_time = 20 →
  (train_speed * 1000 / 3600) * platform_time - (train_speed * 1000 / 3600) * man_time = 180 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3595_359566


namespace NUMINAMATH_CALUDE_min_corners_8x8_grid_l3595_359537

/-- Represents a square grid --/
structure Grid :=
  (size : Nat)

/-- Represents a seven-cell corner --/
structure SevenCellCorner

/-- The number of cells in a seven-cell corner --/
def SevenCellCorner.cells : Nat := 7

/-- Checks if a given number of seven-cell corners can fit in the grid --/
def can_fit (g : Grid) (n : Nat) : Prop :=
  g.size * g.size ≥ n * SevenCellCorner.cells

/-- Checks if after clipping n seven-cell corners, no more can be clipped --/
def no_more_corners (g : Grid) (n : Nat) : Prop :=
  can_fit g n ∧ ¬can_fit g (n + 1)

/-- The main theorem: The minimum number of seven-cell corners that can be clipped from an 8x8 grid such that no more can be clipped is 3 --/
theorem min_corners_8x8_grid :
  ∃ (n : Nat), n = 3 ∧ no_more_corners (Grid.mk 8) n ∧ ∀ m < n, ¬no_more_corners (Grid.mk 8) m :=
sorry

end NUMINAMATH_CALUDE_min_corners_8x8_grid_l3595_359537


namespace NUMINAMATH_CALUDE_estate_value_l3595_359508

def estate_problem (E : ℝ) : Prop :=
  let younger_son := E / 5
  let elder_son := 2 * younger_son
  let husband := 3 * younger_son
  let charity := 4000
  (younger_son + elder_son = 3 * E / 5) ∧
  (elder_son = 2 * younger_son) ∧
  (husband = 3 * younger_son) ∧
  (E = younger_son + elder_son + husband + charity)

theorem estate_value : ∃ E : ℝ, estate_problem E ∧ E = 20000 := by
  sorry

end NUMINAMATH_CALUDE_estate_value_l3595_359508


namespace NUMINAMATH_CALUDE_crate_width_is_sixteen_l3595_359552

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ
  height : ℝ

/-- Checks if a gas tank can fit upright in a crate -/
def fitsInCrate (tank : GasTank) (crate : CrateDimensions) : Prop :=
  (tank.radius * 2 ≤ crate.length ∧ tank.radius * 2 ≤ crate.width) ∨
  (tank.radius * 2 ≤ crate.length ∧ tank.radius * 2 ≤ crate.height) ∨
  (tank.radius * 2 ≤ crate.width ∧ tank.radius * 2 ≤ crate.height)

/-- Theorem: The width of the crate must be 16 feet -/
theorem crate_width_is_sixteen
  (crate : CrateDimensions)
  (tank : GasTank)
  (h1 : crate.length = 12)
  (h2 : crate.height = 18)
  (h3 : tank.radius = 8)
  (h4 : fitsInCrate tank crate)
  (h5 : ∀ t : GasTank, fitsInCrate t crate → t.radius ≤ tank.radius) :
  crate.width = 16 := by
  sorry

end NUMINAMATH_CALUDE_crate_width_is_sixteen_l3595_359552


namespace NUMINAMATH_CALUDE_equation_proof_l3595_359520

theorem equation_proof : 225 + 2 * 15 * 4 + 16 = 361 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3595_359520


namespace NUMINAMATH_CALUDE_average_weight_problem_l3595_359538

/-- The average weight problem -/
theorem average_weight_problem 
  (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84)
  (h2 : A = 78)
  (h3 : E = D + 6)
  (h4 : (B + C + D + E) / 4 = 79) :
  (A + B + C + D) / 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l3595_359538


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l3595_359585

theorem shirt_price_reduction (original_price : ℝ) (original_price_positive : original_price > 0) : 
  let first_sale_price := 0.9 * original_price
  let final_price := 0.9 * first_sale_price
  final_price / original_price = 0.81 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l3595_359585


namespace NUMINAMATH_CALUDE_combined_alloy_force_problem_solution_l3595_359565

/-- Represents an alloy of two metals -/
structure Alloy where
  mass : ℝ
  ratio : ℝ
  force : ℝ

/-- Theorem stating that the force exerted by a combination of two alloys
    is equal to the sum of their individual forces -/
theorem combined_alloy_force (A B : Alloy) :
  let C : Alloy := ⟨A.mass + B.mass, (A.mass * A.ratio + B.mass * B.ratio) / (A.mass + B.mass), A.force + B.force⟩
  C.force = A.force + B.force := by
  sorry

/-- Given alloys A and B with specified properties, prove that their combination
    exerts a force of 40 N -/
theorem problem_solution (A B : Alloy)
  (hA_mass : A.mass = 6)
  (hA_ratio : A.ratio = 2)
  (hA_force : A.force = 30)
  (hB_mass : B.mass = 3)
  (hB_ratio : B.ratio = 1/5)
  (hB_force : B.force = 10) :
  let C : Alloy := ⟨A.mass + B.mass, (A.mass * A.ratio + B.mass * B.ratio) / (A.mass + B.mass), A.force + B.force⟩
  C.force = 40 := by
  sorry

end NUMINAMATH_CALUDE_combined_alloy_force_problem_solution_l3595_359565


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_A_complement_A_union_B_l3595_359530

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- State the theorems to be proved
theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 7} := by sorry

theorem complement_A : (Set.univ \ A) = {x | x < 3 ∨ 7 ≤ x} := by sorry

theorem complement_A_union_B : (Set.univ \ (A ∪ B)) = {x | x ≤ 2 ∨ 10 ≤ x} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_A_complement_A_union_B_l3595_359530


namespace NUMINAMATH_CALUDE_city_population_problem_l3595_359549

theorem city_population_problem (population_b : ℕ) : 
  let population_a := (3 * population_b) / 5
  let population_c := 27500
  let total_population := population_a + population_b + population_c
  (population_c = (5 * population_b) / 4 + 4000) →
  (total_population % 250 = 0) →
  total_population = 57500 :=
by sorry

end NUMINAMATH_CALUDE_city_population_problem_l3595_359549


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3595_359541

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6) ^ 2) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3595_359541


namespace NUMINAMATH_CALUDE_distribute_7_balls_3_boxes_l3595_359572

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_7_balls_3_boxes : distribute 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_7_balls_3_boxes_l3595_359572


namespace NUMINAMATH_CALUDE_linear_program_coefficient_l3595_359503

/-- Given a set of linear constraints and a linear objective function,
    prove that the value of the coefficient m in the objective function
    is -2/3 when the minimum value of the function is -3. -/
theorem linear_program_coefficient (x y : ℝ) (m : ℝ) : 
  (x + y - 2 ≥ 0) →
  (x - y + 1 ≥ 0) →
  (x ≤ 3) →
  (∀ x y, x + y - 2 ≥ 0 → x - y + 1 ≥ 0 → x ≤ 3 → m * x + y ≥ -3) →
  (∃ x y, x + y - 2 ≥ 0 ∧ x - y + 1 ≥ 0 ∧ x ≤ 3 ∧ m * x + y = -3) →
  m = -2/3 := by
sorry

end NUMINAMATH_CALUDE_linear_program_coefficient_l3595_359503


namespace NUMINAMATH_CALUDE_fifteen_people_in_house_l3595_359535

/-- The number of people in a house --/
def num_people_in_house (initial_bedroom : ℕ) (entered_bedroom : ℕ) (living_room : ℕ) : ℕ :=
  initial_bedroom + entered_bedroom + living_room

/-- Theorem: Given the initial conditions, there are 15 people in the house --/
theorem fifteen_people_in_house :
  num_people_in_house 2 5 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_people_in_house_l3595_359535


namespace NUMINAMATH_CALUDE_subtraction_of_integers_l3595_359502

theorem subtraction_of_integers : -1 - 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_integers_l3595_359502


namespace NUMINAMATH_CALUDE_sqrt_two_between_integers_l3595_359581

theorem sqrt_two_between_integers (n : ℕ+) : 
  (n : ℝ) < Real.sqrt 2 ∧ Real.sqrt 2 < (n : ℝ) + 1 → n = 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_two_between_integers_l3595_359581


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3595_359575

theorem inequality_solution_range :
  ∀ (a : ℝ), (∃ x ∈ Set.Icc 0 3, x^2 - a*x - a + 1 ≥ 0) ↔ a ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3595_359575


namespace NUMINAMATH_CALUDE_power_zero_l3595_359588

theorem power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_l3595_359588


namespace NUMINAMATH_CALUDE_f_composition_value_l3595_359521

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else (1/2) ^ x

theorem f_composition_value : f (f (1/27)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3595_359521


namespace NUMINAMATH_CALUDE_work_completion_time_l3595_359533

/-- The number of days it takes 'a' to complete the work -/
def days_a : ℕ := 27

/-- The number of days it takes 'b' to complete the work -/
def days_b : ℕ := 2 * days_a

theorem work_completion_time : days_a = 27 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3595_359533


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l3595_359559

theorem max_value_sum_of_roots (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + 9 * c^2 = 1) :
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + 9 * z^2 = 1 ∧
    Real.sqrt x + Real.sqrt y + Real.sqrt 3 * z > Real.sqrt a + Real.sqrt b + Real.sqrt 3 * c) ∨
  Real.sqrt a + Real.sqrt b + Real.sqrt 3 * c = Real.sqrt (21 / 3) := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l3595_359559


namespace NUMINAMATH_CALUDE_starting_number_sequence_l3595_359523

theorem starting_number_sequence (n : ℕ) : 
  (n ≤ 79 ∧ 
   (∃ (a b c d : ℕ), n < a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 79 ∧
    n % 11 = 0 ∧ a % 11 = 0 ∧ b % 11 = 0 ∧ c % 11 = 0 ∧ d % 11 = 0) ∧
   (∀ m : ℕ, m < n → ¬(∃ (a b c d : ℕ), m < a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 79 ∧
    m % 11 = 0 ∧ a % 11 = 0 ∧ b % 11 = 0 ∧ c % 11 = 0 ∧ d % 11 = 0))) →
  n = 33 := by
sorry

end NUMINAMATH_CALUDE_starting_number_sequence_l3595_359523


namespace NUMINAMATH_CALUDE_fathers_age_twice_marikas_l3595_359554

/-- Marika's age in 2006 -/
def marika_age_2006 : ℕ := 10

/-- The year of Marika's 10th birthday -/
def birth_year : ℕ := 2006

/-- The ratio of father's age to Marika's age in 2006 -/
def age_ratio_2006 : ℕ := 5

/-- The year when father's age will be twice Marika's age -/
def target_year : ℕ := 2036

theorem fathers_age_twice_marikas : 
  target_year = birth_year + (age_ratio_2006 - 2) * marika_age_2006 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_twice_marikas_l3595_359554


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l3595_359599

theorem parallel_lines_slope (a : ℝ) : 
  (∃ (x y : ℝ), a * x - 5 * y - 9 = 0 ∧ 2 * x - 3 * y - 10 = 0) →
  a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l3595_359599


namespace NUMINAMATH_CALUDE_f_increasing_implies_a_range_a_range_implies_f_increasing_f_increasing_iff_a_range_l3595_359542

/-- The function f(x) defined as 2x^2 - 4(1-a)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * (1 - a) * x + 1

/-- The theorem stating that if f(x) is increasing on [3,+∞), then a ≥ -2 -/
theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y, 3 ≤ x → x < y → f a x < f a y) → a ≥ -2 :=
sorry

/-- The theorem stating that if a ≥ -2, then f(x) is increasing on [3,+∞) -/
theorem a_range_implies_f_increasing (a : ℝ) :
  a ≥ -2 → (∀ x y, 3 ≤ x → x < y → f a x < f a y) :=
sorry

/-- The main theorem stating the equivalence between f(x) being increasing on [3,+∞) and a ≥ -2 -/
theorem f_increasing_iff_a_range (a : ℝ) :
  (∀ x y, 3 ≤ x → x < y → f a x < f a y) ↔ a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_implies_a_range_a_range_implies_f_increasing_f_increasing_iff_a_range_l3595_359542


namespace NUMINAMATH_CALUDE_cos_squared_fifteen_degrees_l3595_359548

theorem cos_squared_fifteen_degrees :
  2 * (Real.cos (15 * π / 180))^2 - 1 = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_squared_fifteen_degrees_l3595_359548


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3595_359593

theorem complex_number_in_first_quadrant : 
  let z : ℂ := 1 / (2 - Complex.I)
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3595_359593


namespace NUMINAMATH_CALUDE_even_function_derivative_is_odd_l3595_359598

theorem even_function_derivative_is_odd 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x) 
  (h_deriv : ∀ x, HasDerivAt f (g x) x) : 
  ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_even_function_derivative_is_odd_l3595_359598


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3595_359543

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, 10) and (-4, -10) is 2. -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := 10
  let x₂ : ℝ := -4
  let y₂ : ℝ := -10
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 2 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3595_359543


namespace NUMINAMATH_CALUDE_unique_solution_l3595_359553

/-- Define the function f(x, y) = (x + y)(x^2 + y^2) -/
def f (x y : ℝ) : ℝ := (x + y) * (x^2 + y^2)

/-- Theorem stating that the only solution to the system of equations is (0, 0, 0, 0) -/
theorem unique_solution (a b c d : ℝ) :
  f a b = f c d ∧ f a c = f b d ∧ f a d = f b c →
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry


end NUMINAMATH_CALUDE_unique_solution_l3595_359553


namespace NUMINAMATH_CALUDE_factors_of_x4_plus_81_l3595_359534

theorem factors_of_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 3*x + 9) * (x^2 - 3*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factors_of_x4_plus_81_l3595_359534


namespace NUMINAMATH_CALUDE_player_positions_satisfy_distances_l3595_359511

/-- Represents the positions of four players on a number line -/
def PlayerPositions : Fin 4 → ℝ
| 0 => 0
| 1 => 1
| 2 => 4
| 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances := by
  sorry

#check player_positions_satisfy_distances

end NUMINAMATH_CALUDE_player_positions_satisfy_distances_l3595_359511


namespace NUMINAMATH_CALUDE_sons_age_l3595_359501

theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3595_359501


namespace NUMINAMATH_CALUDE_natashas_average_speed_l3595_359536

/-- Natasha's hill climbing problem -/
theorem natashas_average_speed
  (climb_time : ℝ)
  (descent_time : ℝ)
  (climb_speed : ℝ)
  (h_climb_time : climb_time = 4)
  (h_descent_time : descent_time = 2)
  (h_climb_speed : climb_speed = 3)
  : (2 * climb_speed * climb_time) / (climb_time + descent_time) = 4 :=
by sorry

end NUMINAMATH_CALUDE_natashas_average_speed_l3595_359536


namespace NUMINAMATH_CALUDE_pyramid_volume_l3595_359532

theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (height : ℝ) :
  base_length = 1/3 →
  base_width = 1/4 →
  height = 1 →
  (1/3) * (base_length * base_width) * height = 1/36 := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3595_359532


namespace NUMINAMATH_CALUDE_range_of_A_l3595_359517

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_A : ∀ a : ℝ, a ∈ A ↔ a ∈ Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_range_of_A_l3595_359517


namespace NUMINAMATH_CALUDE_circles_are_externally_tangent_l3595_359555

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii. -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The first circle: x^2 + y^2 = 4 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The second circle: x^2 + y^2 - 10x + 16 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 16 = 0

theorem circles_are_externally_tangent :
  externally_tangent (0, 0) (5, 0) 2 3 :=
sorry

end NUMINAMATH_CALUDE_circles_are_externally_tangent_l3595_359555


namespace NUMINAMATH_CALUDE_probability_of_event_D_is_one_l3595_359522

theorem probability_of_event_D_is_one :
  ∀ x : ℝ,
  (∃ (P_N P_D_given_N P_D : ℝ),
    P_N = 3/8 ∧
    P_D_given_N = x^2 ∧
    P_D = 5/8 + (3/8) * x^2 ∧
    0 ≤ P_N ∧ P_N ≤ 1 ∧
    0 ≤ P_D_given_N ∧ P_D_given_N ≤ 1 ∧
    0 ≤ P_D ∧ P_D ≤ 1) →
  P_D = 1 :=
sorry

end NUMINAMATH_CALUDE_probability_of_event_D_is_one_l3595_359522


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l3595_359519

theorem defective_shipped_percentage 
  (total_units : ℝ) 
  (defective_rate : ℝ) 
  (shipped_rate : ℝ) 
  (h1 : defective_rate = 0.1) 
  (h2 : shipped_rate = 0.05) : 
  defective_rate * shipped_rate * 100 = 0.5 := by
sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l3595_359519


namespace NUMINAMATH_CALUDE_hexagon_area_l3595_359544

/-- The area of a hexagon inscribed in a rectangle with corner triangles removed -/
theorem hexagon_area (rectangle_length rectangle_width triangle_base triangle_height : ℝ) 
  (h1 : rectangle_length = 6)
  (h2 : rectangle_width = 8)
  (h3 : triangle_base = 1)
  (h4 : triangle_height = 4) : 
  rectangle_length * rectangle_width - 4 * (1/2 * triangle_base * triangle_height) = 40 := by
  sorry

#check hexagon_area

end NUMINAMATH_CALUDE_hexagon_area_l3595_359544


namespace NUMINAMATH_CALUDE_circles_contained_l3595_359557

theorem circles_contained (r R d : ℝ) (hr : r = 1) (hR : R = 5) (hd : d = 3) :
  d < R - r ∧ d + r < R :=
sorry

end NUMINAMATH_CALUDE_circles_contained_l3595_359557


namespace NUMINAMATH_CALUDE_trajectory_of_vertex_C_trajectory_of_vertex_C_proof_l3595_359518

/-- The trajectory of vertex C in triangle ABC, where A(0, 2) and B(0, -2), 
    and the perimeter is 10, forms an ellipse. -/
theorem trajectory_of_vertex_C (C : ℝ × ℝ) : Prop :=
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (0, -2)
  let perimeter : ℝ := 10
  let dist (P Q : ℝ × ℝ) := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  dist A B + dist B C + dist C A = perimeter →
  C.1 ≠ 0 →
  C.1^2 / 5 + C.2^2 / 9 = 1

/-- The proof of the theorem. -/
theorem trajectory_of_vertex_C_proof : ∀ C, trajectory_of_vertex_C C := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_vertex_C_trajectory_of_vertex_C_proof_l3595_359518


namespace NUMINAMATH_CALUDE_average_speed_tony_l3595_359571

def rollercoaster_speeds : List ℝ := [50, 62, 73, 70, 40]

theorem average_speed_tony (speeds := rollercoaster_speeds) : 
  (speeds.sum / speeds.length : ℝ) = 59 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_tony_l3595_359571


namespace NUMINAMATH_CALUDE_largest_domain_l3595_359586

-- Define the property of the function f
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (f x + f (1/x) = x^2)

-- Define the domain of f
def domain (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | x ≠ 0 ∧ ∃ y : ℝ, f x = y}

-- Theorem statement
theorem largest_domain (f : ℝ → ℝ) (h : has_property f) :
  domain f = {x : ℝ | x ≠ 0} :=
sorry

end NUMINAMATH_CALUDE_largest_domain_l3595_359586


namespace NUMINAMATH_CALUDE_simplify_expression_l3595_359568

theorem simplify_expression (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  ((x + a)^4) / ((a - b)*(a - c)) + ((x + b)^4) / ((b - a)*(b - c)) + ((x + c)^4) / ((c - a)*(c - b)) = a + b + c + 4*x :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3595_359568


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l3595_359528

-- Define the point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (m + 3, m - 1)

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant_m_range :
  ∀ m : ℝ, in_fourth_quadrant (P m) ↔ -3 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l3595_359528


namespace NUMINAMATH_CALUDE_x_bound_y_bound_l3595_359574

/-- Represents the position of a particle -/
structure Position where
  x : ℕ
  y : ℕ

/-- Calculates the position of a particle after n minutes -/
def particlePosition (n : ℕ) : Position :=
  sorry

/-- The initial rightward movement is 2 units -/
axiom initial_rightward : (particlePosition 1).x = 2

/-- The y-coordinate doesn't change in the first minute -/
axiom initial_upward : (particlePosition 1).y = 0

/-- The x-coordinate never decreases -/
axiom x_nondecreasing (n : ℕ) : (particlePosition n).x ≤ (particlePosition (n + 1)).x

/-- The y-coordinate never decreases -/
axiom y_nondecreasing (n : ℕ) : (particlePosition n).y ≤ (particlePosition (n + 1)).y

/-- The x-coordinate is bounded by the initial movement plus subsequent rightward movements -/
theorem x_bound (n : ℕ) : 
  (particlePosition n).x ≤ 2 + 2 * (n / 4) * ((n / 4) + 1) :=
  sorry

/-- The y-coordinate is bounded by the sum of upward movements -/
theorem y_bound (n : ℕ) : 
  (particlePosition n).y ≤ (n - 1) * (n / 4) :=
  sorry

end NUMINAMATH_CALUDE_x_bound_y_bound_l3595_359574


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3595_359563

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem simple_interest_problem :
  let principal : ℝ := 10000
  let rate : ℝ := 0.08
  let time : ℝ := 1
  simple_interest principal rate time = 800 := by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3595_359563


namespace NUMINAMATH_CALUDE_counterfeit_coin_identification_l3595_359546

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | Left : WeighingResult  -- Left side is heavier
  | Right : WeighingResult -- Right side is heavier
  | Equal : WeighingResult -- Both sides are equal

/-- Represents a weighing operation -/
def Weighing := Nat → Nat → WeighingResult

/-- Represents a strategy to find the counterfeit coin -/
def Strategy := List (Nat × Nat) → Nat

/-- Checks if a strategy correctly identifies the counterfeit coin -/
def isValidStrategy (n : Nat) (strategy : Strategy) : Prop :=
  ∀ (counterfeit : Nat), counterfeit < n →
    ∃ (weighings : List (Nat × Nat)),
      (∀ w ∈ weighings, w.1 < n ∧ w.2 < n) ∧
      (weighings.length ≤ 3) ∧
      (strategy weighings = counterfeit)

theorem counterfeit_coin_identification (n : Nat) (h : n = 10 ∨ n = 27) :
  ∃ (strategy : Strategy), isValidStrategy n strategy :=
sorry

end NUMINAMATH_CALUDE_counterfeit_coin_identification_l3595_359546


namespace NUMINAMATH_CALUDE_coconut_jelly_beans_count_l3595_359590

def total_jelly_beans : ℕ := 4000
def red_fraction : ℚ := 3/4
def coconut_fraction : ℚ := 1/4

theorem coconut_jelly_beans_count : 
  (red_fraction * total_jelly_beans : ℚ) * coconut_fraction = 750 := by
  sorry

end NUMINAMATH_CALUDE_coconut_jelly_beans_count_l3595_359590


namespace NUMINAMATH_CALUDE_inequality_proof_l3595_359595

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3595_359595


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3595_359505

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_perpendicular_to_plane 
  (l n : Line) (α : Plane) :
  parallel l n → perpendicular_line_plane n α → perpendicular_line_plane l α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3595_359505


namespace NUMINAMATH_CALUDE_function_composition_equality_l3595_359524

/-- Given two functions f and g, where f(x) = Ax³ - B and g(x) = Bx², 
    with B ≠ 0 and f(g(2)) = 0, prove that A = 1 / (64B²) -/
theorem function_composition_equality (A B : ℝ) 
  (hB : B ≠ 0)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x^3 - B)
  (hg : ∀ x, g x = B * x^2)
  (h_comp : f (g 2) = 0) :
  A = 1 / (64 * B^2) := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3595_359524


namespace NUMINAMATH_CALUDE_star_composition_l3595_359560

-- Define the star operations
def star_right (y : ℝ) : ℝ := 9 - y
def star_left (y : ℝ) : ℝ := y - 9

-- State the theorem
theorem star_composition : star_left (star_right 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_star_composition_l3595_359560


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3595_359512

theorem tangent_line_to_circle (a : ℝ) : 
  a > 0 → 
  (∃ x : ℝ, x^2 + a^2 + 2*x - 2*a - 2 = 0 ∧ 
   ∀ y : ℝ, y ≠ a → x^2 + y^2 + 2*x - 2*y - 2 > 0) → 
  a = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3595_359512


namespace NUMINAMATH_CALUDE_yahs_to_bahs_l3595_359509

-- Define the units
variable (bah rah yah : ℕ → ℚ)

-- Define the conversion rates
axiom bah_to_rah : ∀ x, bah x = rah (2 * x)
axiom rah_to_yah : ∀ x, rah x = yah (2 * x)

-- State the theorem
theorem yahs_to_bahs : yah 1200 = bah 300 := by
  sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_l3595_359509


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3595_359540

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 6*x + 8 = 0 ∧ x ≠ 0 → x = 2 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3595_359540


namespace NUMINAMATH_CALUDE_author_earnings_l3595_359500

theorem author_earnings (paper_cover_percentage : ℝ) (hardcover_percentage : ℝ)
  (paper_cover_copies : ℕ) (hardcover_copies : ℕ)
  (paper_cover_price : ℝ) (hardcover_price : ℝ) :
  paper_cover_percentage = 0.06 →
  hardcover_percentage = 0.12 →
  paper_cover_copies = 32000 →
  hardcover_copies = 15000 →
  paper_cover_price = 0.20 →
  hardcover_price = 0.40 →
  (paper_cover_percentage * (paper_cover_copies : ℝ) * paper_cover_price) +
  (hardcover_percentage * (hardcover_copies : ℝ) * hardcover_price) = 1104 :=
by sorry

end NUMINAMATH_CALUDE_author_earnings_l3595_359500


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3595_359562

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(23,24,30), (12,30,31), (9,30,32), (4,30,33), (15,22,36), (9,18,40), (4,15,42)}

theorem diophantine_equation_solution :
  {t : ℕ × ℕ × ℕ | is_valid_triple t.1 t.2.1 t.2.2} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3595_359562


namespace NUMINAMATH_CALUDE_jimin_class_size_l3595_359556

/-- The number of students in Jimin's class -/
def total_students : ℕ := 45

/-- The number of students who like Korean -/
def korean_fans : ℕ := 38

/-- The number of students who like math -/
def math_fans : ℕ := 39

/-- The number of students who like both Korean and math -/
def both_fans : ℕ := 32

/-- There is no student who does not like both Korean and math -/
axiom no_other_students : total_students = korean_fans + math_fans - both_fans

theorem jimin_class_size :
  total_students = 45 :=
sorry

end NUMINAMATH_CALUDE_jimin_class_size_l3595_359556


namespace NUMINAMATH_CALUDE_expenditure_problem_l3595_359506

/-- Proves that given the conditions of the expenditure problem, the number of days in the next part of the week is 4. -/
theorem expenditure_problem (first_part_days : ℕ) (second_part_days : ℕ) 
  (first_part_avg : ℚ) (second_part_avg : ℚ) (total_avg : ℚ) :
  first_part_days = 3 →
  first_part_avg = 350 →
  second_part_avg = 420 →
  total_avg = 390 →
  first_part_days + second_part_days = 7 →
  (first_part_days * first_part_avg + second_part_days * second_part_avg) / 7 = total_avg →
  second_part_days = 4 := by
sorry

end NUMINAMATH_CALUDE_expenditure_problem_l3595_359506


namespace NUMINAMATH_CALUDE_bike_purchase_weeks_l3595_359564

def bike_cost : ℕ := 600
def gift_money : ℕ := 150
def weekly_earnings : ℕ := 20

def weeks_needed : ℕ := 23

theorem bike_purchase_weeks : 
  ∀ (w : ℕ), w ≥ weeks_needed ↔ gift_money + w * weekly_earnings ≥ bike_cost :=
by sorry

end NUMINAMATH_CALUDE_bike_purchase_weeks_l3595_359564


namespace NUMINAMATH_CALUDE_johns_money_to_mother_l3595_359516

theorem johns_money_to_mother (initial_amount : ℝ) (father_fraction : ℝ) (amount_left : ℝ) :
  initial_amount = 200 →
  father_fraction = 3 / 10 →
  amount_left = 65 →
  ∃ (mother_fraction : ℝ), 
    mother_fraction = 3 / 8 ∧
    amount_left = initial_amount * (1 - (mother_fraction + father_fraction)) :=
by sorry

end NUMINAMATH_CALUDE_johns_money_to_mother_l3595_359516


namespace NUMINAMATH_CALUDE_school_students_l3595_359597

def total_students (n : ℕ) (largest_class : ℕ) (diff : ℕ) : ℕ :=
  (n * (2 * largest_class - (n - 1) * diff)) / 2

theorem school_students :
  total_students 5 24 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_school_students_l3595_359597


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3595_359587

/-- Given a cube with surface area 294 square centimeters, its volume is 343 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  s > 0 →
  6 * s^2 = 294 →
  s^3 = 343 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3595_359587


namespace NUMINAMATH_CALUDE_max_weekly_profit_l3595_359582

-- Define the price reduction x
def x : ℝ := 5

-- Define the original cost per unit
def original_cost : ℝ := 5

-- Define the original selling price per unit
def original_price : ℝ := 14

-- Define the initial weekly sales volume
def initial_volume : ℝ := 75

-- Define the proportionality constant k
def k : ℝ := 5

-- Define the increase in sales volume as a function of price reduction
def m (x : ℝ) : ℝ := k * x^2

-- Define the weekly sales profit as a function of price reduction
def y (x : ℝ) : ℝ := (original_price - x - original_cost) * (initial_volume + m x)

-- State the theorem
theorem max_weekly_profit :
  y x = 800 ∧ ∀ z, 0 ≤ z ∧ z < 9 → y z ≤ y x :=
sorry

end NUMINAMATH_CALUDE_max_weekly_profit_l3595_359582


namespace NUMINAMATH_CALUDE_refusing_managers_pair_l3595_359580

/-- The number of managers to choose from -/
def total_managers : ℕ := 8

/-- The number of managers needed for the meeting -/
def meeting_size : ℕ := 4

/-- The number of ways to select managers for the meeting -/
def selection_ways : ℕ := 55

/-- Calculates the number of combinations -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The theorem to prove -/
theorem refusing_managers_pair : 
  ∃! (refusing_pairs : ℕ), 
    combinations total_managers meeting_size - 
    refusing_pairs * combinations (total_managers - 2) (meeting_size - 2) = 
    selection_ways :=
sorry

end NUMINAMATH_CALUDE_refusing_managers_pair_l3595_359580


namespace NUMINAMATH_CALUDE_circle_properties_l3595_359591

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 4

-- Define the line that contains the center of C
def center_line (x y : ℝ) : Prop :=
  2 * x - y + 2 = 0

-- Define the intersecting line
def intersecting_line (a x y : ℝ) : Prop :=
  x = a * y + 3

-- Theorem statement
theorem circle_properties :
  -- Given conditions
  (circle_C 1 0) ∧ 
  (circle_C (-1) 2) ∧
  (∃ (x₀ y₀ : ℝ), circle_C x₀ y₀ ∧ center_line x₀ y₀) →
  -- Conclusions
  (∀ (x y : ℝ), circle_C x y ↔ (x + 1)^2 + y^2 = 4) ∧
  (∃ (a : ℝ), (a = Real.sqrt 15 ∨ a = -Real.sqrt 15) ∧
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
      intersecting_line a x₁ y₁ ∧ intersecting_line a x₂ y₂ ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) :=
by sorry


end NUMINAMATH_CALUDE_circle_properties_l3595_359591


namespace NUMINAMATH_CALUDE_william_car_wash_time_l3595_359561

/-- Represents the time in minutes for each car washing task -/
structure CarWashTime where
  windows : ℕ
  body : ℕ
  tires : ℕ
  waxing : ℕ

/-- Calculates the total time for washing a normal car -/
def normalCarTime (t : CarWashTime) : ℕ :=
  t.windows + t.body + t.tires + t.waxing

/-- Theorem: William's total car washing time is 96 minutes -/
theorem william_car_wash_time :
  ∀ (t : CarWashTime),
  t.windows = 4 →
  t.body = 7 →
  t.tires = 4 →
  t.waxing = 9 →
  2 * normalCarTime t + 2 * normalCarTime t = 96 := by
  sorry

#check william_car_wash_time

end NUMINAMATH_CALUDE_william_car_wash_time_l3595_359561


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l3595_359594

theorem subset_implies_a_equals_one :
  ∀ (a : ℝ),
  let A : Set ℝ := {0, -a}
  let B : Set ℝ := {1, a - 2, 2 * a - 2}
  A ⊆ B → a = 1 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l3595_359594


namespace NUMINAMATH_CALUDE_triangle_inequality_l3595_359527

/-- Given a triangle with side lengths a, b, c, semiperimeter p, inradius r, 
    and distances from incenter to sides l_a, l_b, l_c, prove that 
    l_a * l_b * l_c ≤ r * p^2 -/
theorem triangle_inequality (a b c p r l_a l_b l_c : ℝ) 
  (h1 : l_a * l_b * l_c ≤ Real.sqrt (p^3 * (p - a) * (p - b) * (p - c)))
  (h2 : ∃ S, S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h3 : ∃ S, S = r * p) :
  l_a * l_b * l_c ≤ r * p^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3595_359527


namespace NUMINAMATH_CALUDE_sharp_composition_72_l3595_359539

def sharp (N : ℝ) : ℝ := 0.5 * N + 2

theorem sharp_composition_72 : sharp (sharp (sharp 72)) = 12.5 := by sorry

end NUMINAMATH_CALUDE_sharp_composition_72_l3595_359539


namespace NUMINAMATH_CALUDE_salary_change_percentage_loss_l3595_359551

theorem salary_change (original : ℝ) (h : original > 0) :
  let decreased := original * (1 - 0.5)
  let final := decreased * (1 + 0.5)
  final = original * 0.75 :=
by
  sorry

theorem percentage_loss : 
  1 - 0.75 = 0.25 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_loss_l3595_359551


namespace NUMINAMATH_CALUDE_patty_avoids_chores_for_ten_weeks_l3595_359510

/-- Represents the cookie exchange system set up by Patty --/
structure CookieExchange where
  cookie_per_chore : ℕ
  chores_per_week : ℕ
  money_available : ℕ
  cookies_per_pack : ℕ
  cost_per_pack : ℕ

/-- Calculates the number of weeks Patty can avoid chores --/
def weeks_without_chores (ce : CookieExchange) : ℕ :=
  let packs_bought := ce.money_available / ce.cost_per_pack
  let total_cookies := packs_bought * ce.cookies_per_pack
  let cookies_per_week := ce.cookie_per_chore * ce.chores_per_week
  total_cookies / cookies_per_week

/-- Theorem stating that Patty can avoid chores for 10 weeks --/
theorem patty_avoids_chores_for_ten_weeks :
  let ce : CookieExchange := {
    cookie_per_chore := 3,
    chores_per_week := 4,
    money_available := 15,
    cookies_per_pack := 24,
    cost_per_pack := 3
  }
  weeks_without_chores ce = 10 := by
  sorry

end NUMINAMATH_CALUDE_patty_avoids_chores_for_ten_weeks_l3595_359510


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l3595_359514

theorem infinite_geometric_series_sum : 
  let a : ℝ := 1/4  -- first term
  let r : ℝ := 1/2  -- common ratio
  let S : ℝ := ∑' n, a * r^n  -- infinite sum
  S = 1/2 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l3595_359514


namespace NUMINAMATH_CALUDE_rectangle_horizontal_length_l3595_359529

theorem rectangle_horizontal_length 
  (perimeter : ℝ) 
  (h v : ℝ) 
  (perimeter_eq : perimeter = 2 * h + 2 * v) 
  (vertical_shorter : v = h - 3) 
  (perimeter_value : perimeter = 54) : h = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_horizontal_length_l3595_359529


namespace NUMINAMATH_CALUDE_sqrt_ab_is_integer_l3595_359513

theorem sqrt_ab_is_integer (a b n : ℕ+) 
  (h : (a : ℚ) / b = ((a : ℚ)^2 + (n : ℚ)^2) / ((b : ℚ)^2 + (n : ℚ)^2)) : 
  ∃ k : ℕ, k^2 = a * b := by
sorry

end NUMINAMATH_CALUDE_sqrt_ab_is_integer_l3595_359513


namespace NUMINAMATH_CALUDE_division_simplification_l3595_359515

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y :=
by sorry

end NUMINAMATH_CALUDE_division_simplification_l3595_359515


namespace NUMINAMATH_CALUDE_inequality_proof_l3595_359592

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) : 
  0 ≤ x*y + y*z + z*x - 2*x*y*z ∧ x*y + y*z + z*x - 2*x*y*z ≤ 7/27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3595_359592


namespace NUMINAMATH_CALUDE_multiply_nine_negative_three_l3595_359525

theorem multiply_nine_negative_three : 9 * (-3) = -27 := by
  sorry

end NUMINAMATH_CALUDE_multiply_nine_negative_three_l3595_359525


namespace NUMINAMATH_CALUDE_smallest_cookie_boxes_l3595_359531

theorem smallest_cookie_boxes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (15 * m - 3) % 11 = 0 → n ≤ m) ∧ 
  (15 * n - 3) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_boxes_l3595_359531


namespace NUMINAMATH_CALUDE_M_union_N_eq_N_l3595_359507

def M : Set ℝ := {x | x^2 - 2*x ≤ 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

theorem M_union_N_eq_N : M ∪ N = N := by sorry

end NUMINAMATH_CALUDE_M_union_N_eq_N_l3595_359507


namespace NUMINAMATH_CALUDE_set_equation_solution_l3595_359569

theorem set_equation_solution (p q : ℝ) : 
  let M := {x : ℝ | x^2 + p*x - 2 = 0}
  let N := {x : ℝ | x^2 - 2*x + q = 0}
  (M ∪ N = {-1, 0, 2}) → (p = -1 ∧ q = 0) := by
sorry

end NUMINAMATH_CALUDE_set_equation_solution_l3595_359569


namespace NUMINAMATH_CALUDE_subset_iff_range_l3595_359526

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - a + 1) * (x - a - 1) ≤ 0}
def B : Set ℝ := {x | |x - 1/2| ≤ 3/2}

-- State the theorem
theorem subset_iff_range (a : ℝ) : A a ⊆ B ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_subset_iff_range_l3595_359526


namespace NUMINAMATH_CALUDE_orange_segments_total_l3595_359596

/-- Represents the number of orange segments each animal received -/
structure OrangeDistribution where
  siskin : ℕ
  hedgehog : ℕ
  beaver : ℕ

/-- Defines the conditions of the orange distribution problem -/
def validDistribution (d : OrangeDistribution) : Prop :=
  d.hedgehog = 2 * d.siskin ∧
  d.beaver = 5 * d.siskin ∧
  d.beaver = d.siskin + 8

/-- The theorem stating that the total number of orange segments is 16 -/
theorem orange_segments_total (d : OrangeDistribution) 
  (h : validDistribution d) : d.siskin + d.hedgehog + d.beaver = 16 := by
  sorry

#check orange_segments_total

end NUMINAMATH_CALUDE_orange_segments_total_l3595_359596
