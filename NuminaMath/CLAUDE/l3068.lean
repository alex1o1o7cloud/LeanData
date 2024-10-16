import Mathlib

namespace NUMINAMATH_CALUDE_rotation_transform_triangles_l3068_306830

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Clockwise rotation around a point -/
def rotateClockwise (center : Point) (angle : ℝ) (p : Point) : Point :=
  sorry

/-- Check if two triangles are congruent -/
def areCongruent (t1 t2 : Triangle) : Prop :=
  sorry

theorem rotation_transform_triangles (m x y : ℝ) : 
  let ABC := Triangle.mk (Point.mk 0 0) (Point.mk 0 12) (Point.mk 16 0)
  let A'B'C' := Triangle.mk (Point.mk 24 18) (Point.mk 36 18) (Point.mk 24 2)
  let center := Point.mk x y
  0 < m → m < 180 →
  (areCongruent (Triangle.mk 
    (rotateClockwise center m ABC.A)
    (rotateClockwise center m ABC.B)
    (rotateClockwise center m ABC.C)) A'B'C') →
  m + x + y = 108 :=
sorry

end NUMINAMATH_CALUDE_rotation_transform_triangles_l3068_306830


namespace NUMINAMATH_CALUDE_lowest_n_for_polynomial_property_l3068_306811

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Property that a polynomial takes value 2 for n distinct integers -/
def TakesValueTwoForNIntegers (P : IntPolynomial) (n : ℕ) : Prop :=
  ∃ (S : Finset ℤ), S.card = n ∧ ∀ x ∈ S, P x = 2

/-- Property that a polynomial never takes value 4 for any integer -/
def NeverTakesValueFour (P : IntPolynomial) : Prop :=
  ∀ x : ℤ, P x ≠ 4

/-- The main theorem statement -/
theorem lowest_n_for_polynomial_property : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m ≥ n → 
    ∀ (P : IntPolynomial), 
      TakesValueTwoForNIntegers P m → NeverTakesValueFour P) ∧
  (∀ (k : ℕ), 0 < k ∧ k < n → 
    ∃ (Q : IntPolynomial), 
      TakesValueTwoForNIntegers Q k ∧ ¬NeverTakesValueFour Q) ∧
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_lowest_n_for_polynomial_property_l3068_306811


namespace NUMINAMATH_CALUDE_negation_of_existence_irrational_square_l3068_306821

theorem negation_of_existence_irrational_square :
  (¬ ∃ x : ℝ, Irrational (x^2)) ↔ (∀ x : ℝ, ¬ Irrational (x^2)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_irrational_square_l3068_306821


namespace NUMINAMATH_CALUDE_carbonic_acid_weight_is_62_024_l3068_306841

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.011

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 15.999

/-- The molecular formula of carbonic acid -/
structure CarbenicAcid where
  hydrogen : ℕ := 2
  carbon : ℕ := 1
  oxygen : ℕ := 3

/-- The molecular weight of carbonic acid in atomic mass units (amu) -/
def carbonic_acid_weight (acid : CarbenicAcid) : ℝ :=
  acid.hydrogen * hydrogen_weight + 
  acid.carbon * carbon_weight + 
  acid.oxygen * oxygen_weight

/-- Theorem stating that the molecular weight of carbonic acid is 62.024 amu -/
theorem carbonic_acid_weight_is_62_024 :
  carbonic_acid_weight { } = 62.024 := by
  sorry

end NUMINAMATH_CALUDE_carbonic_acid_weight_is_62_024_l3068_306841


namespace NUMINAMATH_CALUDE_train_speed_l3068_306802

/-- Given a train of length 800 meters that crosses an electric pole in 20 seconds,
    prove that its speed is 144 km/h. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (speed_ms : ℝ) (speed_kmh : ℝ)
    (h1 : train_length = 800)
    (h2 : crossing_time = 20)
    (h3 : speed_ms = train_length / crossing_time)
    (h4 : speed_kmh = speed_ms * 3.6) :
    speed_kmh = 144 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3068_306802


namespace NUMINAMATH_CALUDE_intersection_not_in_first_quadrant_l3068_306835

theorem intersection_not_in_first_quadrant (m : ℝ) : 
  let x := -(m + 4) / 2
  let y := m / 2 - 2
  (x > 0 ∧ y > 0) → False := by
  sorry

end NUMINAMATH_CALUDE_intersection_not_in_first_quadrant_l3068_306835


namespace NUMINAMATH_CALUDE_root_quadratic_equation_l3068_306865

theorem root_quadratic_equation (m : ℝ) : 
  (2 * m^2 - 3 * m - 1 = 0) → (4 * m^2 - 6 * m = 2) := by
  sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_l3068_306865


namespace NUMINAMATH_CALUDE_golden_ratio_system_solution_l3068_306860

theorem golden_ratio_system_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : 2 * x * Real.sqrt (x + 1) - y * (y + 1) = 1)
  (eq2 : 2 * y * Real.sqrt (y + 1) - z * (z + 1) = 1)
  (eq3 : 2 * z * Real.sqrt (z + 1) - x * (x + 1) = 1) :
  x = (1 + Real.sqrt 5) / 2 ∧ y = (1 + Real.sqrt 5) / 2 ∧ z = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_golden_ratio_system_solution_l3068_306860


namespace NUMINAMATH_CALUDE_max_primes_in_table_l3068_306895

/-- A number in the table is either prime or the product of two primes -/
inductive TableNumber
  | prime : Nat → TableNumber
  | product : Nat → Nat → TableNumber

/-- Definition of the table -/
def Table := Fin 80 → Fin 80 → TableNumber

/-- Predicate to check if two TableNumbers are not coprime -/
def not_coprime : TableNumber → TableNumber → Prop :=
  sorry

/-- Predicate to check if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  sorry

/-- Predicate to check if for any number, there's another number in the same row or column that's not coprime -/
def has_not_coprime_neighbor (t : Table) : Prop :=
  sorry

/-- Count the number of prime numbers in the table -/
def count_primes (t : Table) : Nat :=
  sorry

/-- The main theorem -/
theorem max_primes_in_table :
  ∀ t : Table,
    all_distinct t →
    has_not_coprime_neighbor t →
    count_primes t ≤ 4266 :=
  sorry

end NUMINAMATH_CALUDE_max_primes_in_table_l3068_306895


namespace NUMINAMATH_CALUDE_intersection_equality_union_characterization_l3068_306875

-- Define set A
def A (a : ℝ) : Set ℝ := {x | x^2 + 3*a = (a+3)*x}

-- Define set B
def B : Set ℝ := {x | x^2 + 3 = 4*x}

-- Theorem 1: If A ∩ B = A, then a = 1 or a = 3
theorem intersection_equality (a : ℝ) : A a ∩ B = A a → a = 1 ∨ a = 3 := by
  sorry

-- Theorem 2: A ∪ B = {1, 3} when a = 1 or a = 3, and A ∪ B = {a, 1, 3} otherwise
theorem union_characterization (a : ℝ) :
  (a = 1 ∨ a = 3 → A a ∪ B = {1, 3}) ∧
  (a ≠ 1 ∧ a ≠ 3 → A a ∪ B = {a, 1, 3}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_union_characterization_l3068_306875


namespace NUMINAMATH_CALUDE_bisection_solves_x_squared_minus_two_program_flowchart_l3068_306852

/-- Represents different types of flowcharts -/
inductive FlowchartType
  | Process
  | Program
  | KnowledgeStructure
  | OrganizationalStructure

/-- Represents a method for solving equations -/
inductive SolvingMethod
  | Bisection
  | Newton
  | Secant

/-- Represents an equation to be solved -/
structure Equation where
  f : ℝ → ℝ

/-- Determines the type of flowchart used to solve an equation using a specific method -/
def flowchartTypeForSolving (eq : Equation) (method : SolvingMethod) : FlowchartType :=
  sorry

/-- The theorem stating that solving x^2 - 2 = 0 using the bisection method results in a program flowchart -/
theorem bisection_solves_x_squared_minus_two_program_flowchart :
  flowchartTypeForSolving { f := fun x => x^2 - 2 } SolvingMethod.Bisection = FlowchartType.Program :=
  sorry

end NUMINAMATH_CALUDE_bisection_solves_x_squared_minus_two_program_flowchart_l3068_306852


namespace NUMINAMATH_CALUDE_only_possible_amount_l3068_306849

/-- Represents the possible coin types in the machine -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- The value of each coin in cents -/
def coin_value : Coin → ℕ
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- The result of using a coin in the machine -/
def machine_output : Coin → List Coin
  | Coin.Penny => List.replicate 7 Coin.Dime
  | Coin.Nickel => List.replicate 5 Coin.Quarter
  | Coin.Dime => List.replicate 5 Coin.Penny
  | Coin.Quarter => []  -- Not specified in the problem, so we'll leave it empty

/-- The total value in cents after using the machine k times, starting with one penny -/
def total_value (k : ℕ) : ℕ :=
  1 + 69 * k

/-- The given options in cents -/
def options : List ℕ := [315, 483, 552, 760, 897]

/-- Theorem stating that 760 cents ($7.60) is the only possible amount from the given options -/
theorem only_possible_amount :
  ∃ k : ℕ, total_value k = 760 ∧ ∀ n ∈ options, n ≠ 760 → ¬∃ k : ℕ, total_value k = n :=
sorry


end NUMINAMATH_CALUDE_only_possible_amount_l3068_306849


namespace NUMINAMATH_CALUDE_mexican_olympiad_1988_l3068_306894

theorem mexican_olympiad_1988 (f : ℕ+ → ℕ+) 
  (h : ∀ m n : ℕ+, f (f m + f n) = m + n) : 
  f 1988 = 1988 := by sorry

end NUMINAMATH_CALUDE_mexican_olympiad_1988_l3068_306894


namespace NUMINAMATH_CALUDE_third_number_proof_l3068_306871

theorem third_number_proof (a b c : ℕ+) 
  (h_lcm : Nat.lcm a (Nat.lcm b c) = 50400)
  (h_gcd : Nat.gcd a (Nat.gcd b c) = 60)
  (h_a : a = 600)
  (h_b : b = 840) :
  c = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l3068_306871


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l3068_306870

theorem pure_imaginary_product (m : ℝ) : 
  (Complex.I : ℂ).im * ((1 + m * Complex.I) * (1 - Complex.I)).re = 0 → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l3068_306870


namespace NUMINAMATH_CALUDE_coin_division_problem_l3068_306881

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (n % 8 = 6) → 
  (n % 7 = 5) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 9 = 0) := by
sorry

end NUMINAMATH_CALUDE_coin_division_problem_l3068_306881


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_3_minus_4i_l3068_306880

theorem sum_real_imag_parts_3_minus_4i :
  let z : ℂ := 3 - 4*I
  (z.re + z.im : ℝ) = -1 := by sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_3_minus_4i_l3068_306880


namespace NUMINAMATH_CALUDE_sin_sin_2x_max_value_l3068_306890

theorem sin_sin_2x_max_value (x : ℝ) (h : 0 < x ∧ x < π/2) :
  ∃ (max : ℝ), max = (4 * Real.sqrt 3) / 9 ∧
  ∀ y : ℝ, y = Real.sin x * Real.sin (2 * x) → y ≤ max :=
sorry

end NUMINAMATH_CALUDE_sin_sin_2x_max_value_l3068_306890


namespace NUMINAMATH_CALUDE_book_discount_percentage_l3068_306858

def original_price : ℝ := 60
def discounted_price : ℝ := 45
def discount : ℝ := 15
def tax_rate : ℝ := 0.1

theorem book_discount_percentage :
  (discount / original_price) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_book_discount_percentage_l3068_306858


namespace NUMINAMATH_CALUDE_calculator_battery_life_l3068_306873

/-- Calculates the remaining battery life of a calculator after partial use and an exam -/
theorem calculator_battery_life 
  (full_battery : ℝ) 
  (used_fraction : ℝ) 
  (exam_duration : ℝ) 
  (h1 : full_battery = 60) 
  (h2 : used_fraction = 3/4) 
  (h3 : exam_duration = 2) :
  full_battery * (1 - used_fraction) - exam_duration = 13 := by
  sorry

end NUMINAMATH_CALUDE_calculator_battery_life_l3068_306873


namespace NUMINAMATH_CALUDE_marble_game_theorem_l3068_306862

/-- Represents the state of marbles for each player --/
structure MarbleState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Simulates one round of the game where the loser doubles the other players' marbles --/
def playRound (state : MarbleState) (loser : ℕ) : MarbleState :=
  match loser with
  | 1 => MarbleState.mk state.a (state.b * 3) (state.c * 3)
  | 2 => MarbleState.mk (state.a * 3) state.b (state.c * 3)
  | 3 => MarbleState.mk (state.a * 3) (state.b * 3) state.c
  | _ => state

/-- The main theorem statement --/
theorem marble_game_theorem :
  let initial_state := MarbleState.mk 165 57 21
  let after_round1 := playRound initial_state 1
  let after_round2 := playRound after_round1 2
  let final_state := playRound after_round2 3
  (after_round1.c = after_round1.a + 54) ∧
  (final_state.a = final_state.b) ∧
  (final_state.b = final_state.c) := by sorry

end NUMINAMATH_CALUDE_marble_game_theorem_l3068_306862


namespace NUMINAMATH_CALUDE_equations_not_equivalent_l3068_306825

theorem equations_not_equivalent : 
  ¬(∀ x : ℝ, (2 * (x - 10)) / (x^2 - 13*x + 30) = 1 ↔ x^2 - 15*x + 50 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equations_not_equivalent_l3068_306825


namespace NUMINAMATH_CALUDE_fourth_term_value_l3068_306839

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem fourth_term_value : a 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_value_l3068_306839


namespace NUMINAMATH_CALUDE_no_same_color_neighbors_probability_l3068_306842

-- Define the number of beads for each color
def num_red : Nat := 5
def num_white : Nat := 3
def num_blue : Nat := 2

-- Define the total number of beads
def total_beads : Nat := num_red + num_white + num_blue

-- Define a function to calculate the number of valid arrangements
def valid_arrangements : Nat := 0

-- Define a function to calculate the total number of possible arrangements
def total_arrangements : Nat := Nat.factorial total_beads / (Nat.factorial num_red * Nat.factorial num_white * Nat.factorial num_blue)

-- Theorem: The probability of no two neighboring beads being the same color is 0
theorem no_same_color_neighbors_probability :
  (valid_arrangements : ℚ) / total_arrangements = 0 := by sorry

end NUMINAMATH_CALUDE_no_same_color_neighbors_probability_l3068_306842


namespace NUMINAMATH_CALUDE_complex_location_l3068_306856

theorem complex_location (z : ℂ) (h : (z - 3) * (2 - Complex.I) = 5) : 
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_location_l3068_306856


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l3068_306826

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 2 * y^2 - 2 * x * y + x + 9 * y - 2 = 0 ↔
    ((x = 9 ∧ y = 1) ∨ (x = 2 ∧ y = 0) ∨ (x = 8 ∧ y = 2) ∨ (x = 3 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l3068_306826


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_sequence_l3068_306847

theorem arithmetic_mean_of_sequence : 
  let start : ℕ := 3
  let count : ℕ := 60
  let sequence := fun (n : ℕ) => start + n - 1
  let sum := (count * (sequence 1 + sequence count)) / 2
  (sum : ℚ) / count = 32.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_sequence_l3068_306847


namespace NUMINAMATH_CALUDE_F_negative_sufficient_not_necessary_l3068_306813

/-- Represents a general equation of the form x^2 + y^2 + Dx + Ey + F = 0 -/
structure GeneralEquation where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Predicate to check if a GeneralEquation represents a circle -/
def is_circle (eq : GeneralEquation) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ 
    eq.D = -2 * h ∧ 
    eq.E = -2 * k ∧ 
    eq.F = h^2 + k^2 - r^2

/-- Theorem stating that F < 0 is a sufficient but not necessary condition for a circle -/
theorem F_negative_sufficient_not_necessary (eq : GeneralEquation) :
  (eq.F < 0 → is_circle eq) ∧ ¬(is_circle eq → eq.F < 0) :=
sorry

end NUMINAMATH_CALUDE_F_negative_sufficient_not_necessary_l3068_306813


namespace NUMINAMATH_CALUDE_special_function_properties_l3068_306878

/-- A function satisfying certain properties -/
structure SpecialFunction where
  g : ℝ → ℝ
  pos : ∀ x, g x > 0
  mult : ∀ a b, g a * g b = g (a * b)

/-- Properties of the special function -/
theorem special_function_properties (f : SpecialFunction) :
  (f.g 1 = 1) ∧
  (∀ a ≠ 0, f.g (a⁻¹) = (f.g a)⁻¹) ∧
  (∀ a, f.g (a^2) = f.g a * f.g a) := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l3068_306878


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l3068_306810

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem first_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_a3 : a 3 = 2)
  (h_a4 : a 4 = 4) :
  a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l3068_306810


namespace NUMINAMATH_CALUDE_sad_children_count_l3068_306828

theorem sad_children_count (total_children : ℕ) (happy_children : ℕ) (neither_happy_nor_sad : ℕ)
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neither_happy_nor_sad_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : neither_happy_nor_sad = 20)
  (h4 : boys = 17)
  (h5 : girls = 43)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_happy_nor_sad_boys = 5)
  (h9 : total_children = happy_children + neither_happy_nor_sad + (total_children - happy_children - neither_happy_nor_sad))
  (h10 : boys + girls = total_children) :
  total_children - happy_children - neither_happy_nor_sad = 10 := by
  sorry

end NUMINAMATH_CALUDE_sad_children_count_l3068_306828


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3068_306834

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α - Real.cos α = 4/3) : 
  Real.sin (2 * α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3068_306834


namespace NUMINAMATH_CALUDE_ball_cost_l3068_306818

/-- Given that 3 balls cost $4.62, prove that each ball costs $1.54. -/
theorem ball_cost (total_cost : ℝ) (num_balls : ℕ) (cost_per_ball : ℝ) 
  (h1 : total_cost = 4.62)
  (h2 : num_balls = 3)
  (h3 : cost_per_ball = total_cost / num_balls) : 
  cost_per_ball = 1.54 := by
  sorry

end NUMINAMATH_CALUDE_ball_cost_l3068_306818


namespace NUMINAMATH_CALUDE_eight_power_91_greater_than_seven_power_92_l3068_306854

theorem eight_power_91_greater_than_seven_power_92 : 8^91 > 7^92 := by
  sorry

end NUMINAMATH_CALUDE_eight_power_91_greater_than_seven_power_92_l3068_306854


namespace NUMINAMATH_CALUDE_fraction_simplification_l3068_306800

theorem fraction_simplification :
  (154 : ℚ) / 10780 = 1 / 70 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3068_306800


namespace NUMINAMATH_CALUDE_area_traced_on_concentric_spheres_l3068_306806

/-- Theorem: Area traced by a sphere on concentric spheres
  Given:
  - Two concentric spheres with radii R₁ and R₂
  - A smaller sphere that traces areas on both spheres
  - The area traced on the inner sphere is A₁
  Prove:
  The area A₂ traced on the outer sphere is equal to A₁ * (R₂/R₁)²
-/
theorem area_traced_on_concentric_spheres
  (R₁ R₂ A₁ : ℝ)
  (h₁ : 0 < R₁)
  (h₂ : 0 < R₂)
  (h₃ : 0 < A₁)
  (h₄ : R₁ < R₂) :
  ∃ A₂ : ℝ, A₂ = A₁ * (R₂/R₁)^2 := by
  sorry

end NUMINAMATH_CALUDE_area_traced_on_concentric_spheres_l3068_306806


namespace NUMINAMATH_CALUDE_biscuit_price_is_two_l3068_306812

/-- Represents the bakery order problem --/
def bakery_order (quiche_price croissant_price biscuit_price : ℚ) : Prop :=
  let quiche_count : ℕ := 2
  let croissant_count : ℕ := 6
  let biscuit_count : ℕ := 6
  let discount_rate : ℚ := 1 / 10
  let discounted_total : ℚ := 54

  let original_total : ℚ := quiche_count * quiche_price + 
                            croissant_count * croissant_price + 
                            biscuit_count * biscuit_price

  let discounted_amount : ℚ := original_total * discount_rate
  
  (original_total > 50) ∧ 
  (original_total - discounted_amount = discounted_total) ∧
  (quiche_price = 15) ∧
  (croissant_price = 3) ∧
  (biscuit_price = 2)

/-- Theorem stating that the biscuit price is $2.00 --/
theorem biscuit_price_is_two :
  ∃ (quiche_price croissant_price biscuit_price : ℚ),
    bakery_order quiche_price croissant_price biscuit_price ∧
    biscuit_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_biscuit_price_is_two_l3068_306812


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3068_306807

def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℤ)
  (h_arith : arithmeticSequence a)
  (h_a9 : a 9 = -2012)
  (h_a17 : a 17 = -2012) :
  a 1 + a 25 < 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3068_306807


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l3068_306801

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, (f x = y ∧ (3 * x^2 + 1 = 4)) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l3068_306801


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l3068_306851

theorem number_with_specific_remainders (x : ℤ) :
  x % 7 = 3 →
  x^2 % 49 = 44 →
  x^3 % 343 = 111 →
  x % 343 = 17 := by
sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l3068_306851


namespace NUMINAMATH_CALUDE_liza_final_balance_l3068_306831

/-- Calculates the final balance in Liza's account after a series of transactions --/
def calculate_final_balance (initial_balance rent paycheck electricity internet phone additional_deposit : ℚ) : ℚ :=
  let balance_after_rent := initial_balance - rent
  let balance_after_paycheck := balance_after_rent + paycheck
  let balance_after_bills := balance_after_paycheck - electricity - internet
  let grocery_spending := balance_after_bills * (20 / 100)
  let balance_after_groceries := balance_after_bills - grocery_spending
  let interest := balance_after_groceries * (2 / 100)
  let balance_after_interest := balance_after_groceries + interest
  let final_balance := balance_after_interest - phone + additional_deposit
  final_balance

/-- Theorem stating that Liza's final account balance is $1562.528 --/
theorem liza_final_balance :
  calculate_final_balance 800 450 1500 117 100 70 300 = 1562.528 := by
  sorry

end NUMINAMATH_CALUDE_liza_final_balance_l3068_306831


namespace NUMINAMATH_CALUDE_car_profit_percentage_l3068_306832

/-- Calculates the profit percentage on the original price of a car
    given the discount percentage and the selling price increase percentage. -/
theorem car_profit_percentage
  (original_price : ℝ)
  (discount_percent : ℝ)
  (selling_increase_percent : ℝ)
  (h1 : discount_percent = 20)
  (h2 : selling_increase_percent = 70)
  : (((1 - discount_percent / 100) * (1 + selling_increase_percent / 100) - 1) * 100 = 36) := by
  sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l3068_306832


namespace NUMINAMATH_CALUDE_brenda_spay_count_l3068_306869

/-- The number of cats Brenda needs to spay -/
def num_cats : ℕ := 7

/-- The number of dogs Brenda needs to spay -/
def num_dogs : ℕ := 2 * num_cats

/-- The total number of animals Brenda needs to spay -/
def total_animals : ℕ := num_cats + num_dogs

theorem brenda_spay_count : total_animals = 21 := by
  sorry

end NUMINAMATH_CALUDE_brenda_spay_count_l3068_306869


namespace NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l3068_306822

/-- A rectangle in 3D space -/
structure Rectangle3D where
  ab : ℝ
  bc : ℝ

/-- A line segment in 3D space -/
structure Segment3D where
  length : ℝ
  distance_from_plane : ℝ

/-- The volume of a polyhedron formed by a rectangle and a parallel segment -/
def polyhedron_volume (rect : Rectangle3D) (seg : Segment3D) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific polyhedron ABCDKM -/
theorem volume_of_specific_polyhedron :
  let rect := Rectangle3D.mk 2 3
  let seg := Segment3D.mk 5 1
  polyhedron_volume rect seg = 9/2 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l3068_306822


namespace NUMINAMATH_CALUDE_remainder_problem_l3068_306850

theorem remainder_problem (n : ℤ) (h : n % 20 = 11) : (2 * n) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3068_306850


namespace NUMINAMATH_CALUDE_maintenance_interval_after_additive_l3068_306896

/-- Calculates the new maintenance interval after applying an additive -/
def new_maintenance_interval (original_interval : ℕ) (increase_percentage : ℕ) : ℕ :=
  original_interval * (100 + increase_percentage) / 100

/-- Theorem: Given an original maintenance interval of 50 days and a 20% increase,
    the new maintenance interval is 60 days -/
theorem maintenance_interval_after_additive :
  new_maintenance_interval 50 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_interval_after_additive_l3068_306896


namespace NUMINAMATH_CALUDE_circle_radius_with_min_distance_to_line_l3068_306808

/-- The radius of a circle with center (3, -5) that has a minimum distance of 1 to the line 4x - 3y - 2 = 0 -/
theorem circle_radius_with_min_distance_to_line : ∃ (r : ℝ), 
  r > 0 ∧ 
  (∀ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 → 
    ∃ (d : ℝ), d ≥ 1 ∧ d = |4*x - 3*y - 2| / (5 : ℝ)) ∧
  r = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_with_min_distance_to_line_l3068_306808


namespace NUMINAMATH_CALUDE_friendship_class_theorem_l3068_306891

/-- Represents the number of students in a class with specific friendship conditions. -/
structure FriendshipClass where
  boys : ℕ
  girls : ℕ

/-- Checks if the friendship conditions are satisfied for a given class. -/
def satisfiesFriendshipConditions (c : FriendshipClass) : Prop :=
  3 * c.boys = 2 * c.girls

/-- Checks if a class with the given total number of students can satisfy the friendship conditions. -/
def canHaveStudents (n : ℕ) : Prop :=
  ∃ c : FriendshipClass, c.boys + c.girls = n ∧ satisfiesFriendshipConditions c

theorem friendship_class_theorem :
  ¬(canHaveStudents 32) ∧ (canHaveStudents 30) := by sorry

end NUMINAMATH_CALUDE_friendship_class_theorem_l3068_306891


namespace NUMINAMATH_CALUDE_sequence_problem_l3068_306857

def geometric_sequence (a : ℕ → ℝ) := ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1

def arithmetic_sequence (b : ℕ → ℝ) := ∀ n : ℕ, b (n + 1) - b n = b 2 - b 1

theorem sequence_problem (a b : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : arithmetic_sequence b) 
  (h3 : a 1 * a 6 * a 11 = -3 * Real.sqrt 3) 
  (h4 : b 1 + b 6 + b 11 = 7 * Real.pi) : 
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3068_306857


namespace NUMINAMATH_CALUDE_six_thirteen_not_square_nor_cube_l3068_306836

def is_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem six_thirteen_not_square_nor_cube :
  ¬(is_square (6^13)) ∧ ¬(is_cube (6^13)) :=
sorry

end NUMINAMATH_CALUDE_six_thirteen_not_square_nor_cube_l3068_306836


namespace NUMINAMATH_CALUDE_square_tiles_count_l3068_306827

theorem square_tiles_count (total_tiles : ℕ) (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 30)
  (h_total_edges : total_edges = 100) :
  ∃ (triangles squares pentagons : ℕ),
    triangles + squares + pentagons = total_tiles ∧
    3 * triangles + 4 * squares + 5 * pentagons = total_edges ∧
    squares = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_tiles_count_l3068_306827


namespace NUMINAMATH_CALUDE_percentage_not_covering_politics_l3068_306840

-- Define the percentages as real numbers
def country_x : ℝ := 15
def country_y : ℝ := 10
def country_z : ℝ := 8
def x_elections : ℝ := 6
def y_foreign : ℝ := 5
def z_social : ℝ := 3
def not_local : ℝ := 50
def international : ℝ := 5
def economics : ℝ := 2

-- Theorem statement
theorem percentage_not_covering_politics :
  100 - (country_x + country_y + country_z + international + economics + not_local) = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_covering_politics_l3068_306840


namespace NUMINAMATH_CALUDE_arithmetic_sum_specific_l3068_306819

def arithmetic_sum (a₁ l d : ℤ) : ℤ :=
  let n := (l - a₁) / d + 1
  n * (a₁ + l) / 2

theorem arithmetic_sum_specific : arithmetic_sum (-45) 1 2 = -528 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_specific_l3068_306819


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3068_306837

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3068_306837


namespace NUMINAMATH_CALUDE_coefficient_d_nonzero_l3068_306844

/-- A polynomial of degree 5 with five distinct roots including 0 and 1 -/
def Q (a b c d f : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + f

/-- The theorem stating that the coefficient d must be nonzero -/
theorem coefficient_d_nonzero (a b c d f : ℝ) :
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ p ≠ 1 ∧ q ≠ 0 ∧ q ≠ 1 ∧ r ≠ 0 ∧ r ≠ 1 ∧
    ∀ x : ℝ, Q a b c d f x = 0 ↔ x = 0 ∨ x = 1 ∨ x = p ∨ x = q ∨ x = r) →
  d ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_coefficient_d_nonzero_l3068_306844


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3068_306884

theorem smallest_solution_of_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ 
  (∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x) →
  x = 4 - Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3068_306884


namespace NUMINAMATH_CALUDE_product_negative_implies_zero_l3068_306898

theorem product_negative_implies_zero (a b : ℝ) (h : a * b < 0) :
  a^2 * abs b - b^2 * abs a + a * b * (abs a - abs b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_negative_implies_zero_l3068_306898


namespace NUMINAMATH_CALUDE_triangles_from_parallel_lines_l3068_306817

/-- The number of triangles formed by points on two parallel lines -/
theorem triangles_from_parallel_lines (n m : ℕ) (hn : n = 6) (hm : m = 8) :
  n.choose 2 * m + n * m.choose 2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_triangles_from_parallel_lines_l3068_306817


namespace NUMINAMATH_CALUDE_time_to_meet_prove_time_to_meet_l3068_306885

/-- The time it takes for Michael to reach Eric given the specified conditions --/
theorem time_to_meet (initial_distance : ℝ) (speed_ratio : ℝ) (closing_rate : ℝ) 
  (initial_time : ℝ) (delay_time : ℝ) : ℝ :=
  65

/-- Proof of the time it takes for Michael to reach Eric --/
theorem prove_time_to_meet :
  time_to_meet 30 4 2 4 6 = 65 := by
  sorry

end NUMINAMATH_CALUDE_time_to_meet_prove_time_to_meet_l3068_306885


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3068_306823

theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_R : R > 0) :
  (P * (R + 2) * 5) / 100 - (P * R * 5) / 100 = 250 →
  P = 2500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3068_306823


namespace NUMINAMATH_CALUDE_mark_has_six_parking_tickets_l3068_306882

/-- Represents the number of tickets for each person -/
structure Tickets where
  mark_parking : ℕ
  mark_speeding : ℕ
  sarah_parking : ℕ
  sarah_speeding : ℕ
  john_parking : ℕ
  john_speeding : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (t : Tickets) : Prop :=
  t.mark_parking + t.mark_speeding + t.sarah_parking + t.sarah_speeding + t.john_parking + t.john_speeding = 36 ∧
  t.mark_parking = 2 * t.sarah_parking ∧
  t.mark_speeding = t.sarah_speeding ∧
  t.john_parking * 3 = t.mark_parking ∧
  t.john_speeding = 2 * t.sarah_speeding ∧
  t.sarah_speeding = 6

/-- The theorem stating that Mark has 6 parking tickets -/
theorem mark_has_six_parking_tickets (t : Tickets) (h : satisfies_conditions t) : t.mark_parking = 6 := by
  sorry

end NUMINAMATH_CALUDE_mark_has_six_parking_tickets_l3068_306882


namespace NUMINAMATH_CALUDE_final_turtle_count_l3068_306879

def turtle_statues : ℕ → ℕ
| 0 => 4  -- Initial number of statues
| 1 => turtle_statues 0 * 4  -- Second year: quadrupled
| 2 => turtle_statues 1 + 12 - 3  -- Third year: added 12, removed 3
| 3 => turtle_statues 2 + 2 * 3  -- Fourth year: added twice the number broken in year 3
| _ => 0  -- We only care about the first 4 years

theorem final_turtle_count : turtle_statues 3 = 31 := by
  sorry

#eval turtle_statues 3

end NUMINAMATH_CALUDE_final_turtle_count_l3068_306879


namespace NUMINAMATH_CALUDE_wall_length_proof_l3068_306897

/-- The length of a wall given brick dimensions and wall specifications -/
theorem wall_length_proof (brick_length brick_width brick_height : ℝ)
                          (wall_height wall_width : ℝ)
                          (num_bricks : ℕ) :
  brick_length = 125 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_height = 22.5 →
  wall_width = 600 →
  num_bricks = 1280 →
  (brick_length * brick_width * brick_height * num_bricks : ℝ) = 
    wall_height * wall_width * 800 :=
by sorry

end NUMINAMATH_CALUDE_wall_length_proof_l3068_306897


namespace NUMINAMATH_CALUDE_parallel_line_plane_false_l3068_306809

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

/-- Defines when a line is parallel to another line -/
def parallel_line_line (l1 l2 : Line) : Prop := sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem parallel_line_plane_false :
  ∃ (a b : Line) (p : Plane),
    ¬(line_in_plane b p) ∧
    (line_in_plane a p) ∧
    (parallel_line_plane b p) ∧
    ¬(∀ (l : Line), line_in_plane l p → parallel_line_line b l) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_plane_false_l3068_306809


namespace NUMINAMATH_CALUDE_custom_operations_fraction_l3068_306877

-- Define the custom operations
def oplus (a b : ℝ) : ℝ := a * b + b^2
def otimes (a b : ℝ) : ℝ := a - b + a * b^2

-- State the theorem
theorem custom_operations_fraction :
  (oplus 8 3) / (otimes 8 3) = 33 / 77 := by sorry

end NUMINAMATH_CALUDE_custom_operations_fraction_l3068_306877


namespace NUMINAMATH_CALUDE_train_passing_time_l3068_306843

/-- Proves that a train with given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 285 →
  train_speed_kmh = 54 →
  passing_time = 19 →
  train_length / (train_speed_kmh * 1000 / 3600) = passing_time :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l3068_306843


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3068_306874

def A : Set ℝ := {x | x + 2 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3068_306874


namespace NUMINAMATH_CALUDE_second_part_speed_l3068_306816

/-- Represents a bicycle trip with three parts -/
structure BicycleTrip where
  total_distance : ℝ
  time_per_part : ℝ
  speed_first_part : ℝ
  speed_last_part : ℝ

/-- Theorem stating the speed of the second part of the trip -/
theorem second_part_speed (trip : BicycleTrip)
  (h_distance : trip.total_distance = 12)
  (h_time : trip.time_per_part = 0.25)
  (h_speed1 : trip.speed_first_part = 16)
  (h_speed3 : trip.speed_last_part = 20) :
  let distance1 := trip.speed_first_part * trip.time_per_part
  let distance3 := trip.speed_last_part * trip.time_per_part
  let distance2 := trip.total_distance - (distance1 + distance3)
  distance2 / trip.time_per_part = 12 := by
  sorry

#check second_part_speed

end NUMINAMATH_CALUDE_second_part_speed_l3068_306816


namespace NUMINAMATH_CALUDE_A_equiv_B_l3068_306864

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ 2 * p.1 - p.2 = 2}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ p.2 = 0}

-- Theorem stating that A and B are equivalent
theorem A_equiv_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equiv_B_l3068_306864


namespace NUMINAMATH_CALUDE_lady_bird_biscuits_l3068_306829

/-- The number of biscuits Lady Bird can make with a given amount of flour -/
def biscuits_from_flour (flour : ℚ) : ℚ :=
  (flour * 9) / (5/4)

/-- The number of biscuits per guest Lady Bird can allow -/
def biscuits_per_guest (total_biscuits : ℚ) (guests : ℕ) : ℚ :=
  total_biscuits / guests

theorem lady_bird_biscuits :
  let flour_used : ℚ := 5
  let num_guests : ℕ := 18
  let total_biscuits := biscuits_from_flour flour_used
  biscuits_per_guest total_biscuits num_guests = 2 := by
sorry

end NUMINAMATH_CALUDE_lady_bird_biscuits_l3068_306829


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3068_306853

theorem inequality_equivalence (x : ℝ) : (3 + x) * (2 - x) < 0 ↔ x > 2 ∨ x < -3 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3068_306853


namespace NUMINAMATH_CALUDE_palindrome_expansion_existence_l3068_306868

theorem palindrome_expansion_existence (x y k : ℕ+) : 
  ∃ (N : ℕ+) (b : Fin (k + 1) → ℕ+),
    (∀ i : Fin (k + 1), ∃ (a c : ℕ), 
      N = a * (b i)^2 + c * (b i) + a ∧ 
      a < b i ∧ 
      c < b i) ∧
    (∃ (B : ℕ+), 
      N = x * (B^2 + 1) + y * B ∧ 
      b 0 = B) :=
sorry

end NUMINAMATH_CALUDE_palindrome_expansion_existence_l3068_306868


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3068_306861

theorem rectangular_field_area (L W : ℝ) : 
  L = 10 →                 -- One side is 10 feet
  2 * W + L = 130 →        -- Total fencing is 130 feet
  L * W = 600 :=           -- Area of the field is 600 square feet
by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3068_306861


namespace NUMINAMATH_CALUDE_profit_percentage_invariant_l3068_306872

/-- Proves that the profit percentage remains the same with or without discount -/
theorem profit_percentage_invariant (discount_rate : ℝ) (profit_with_discount : ℝ) :
  discount_rate = 0.05 →
  profit_with_discount = 0.2255 →
  profit_with_discount = (profit_with_discount * (1 + discount_rate)) / (1 + discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_invariant_l3068_306872


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3068_306848

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) :
  3*x^2 - 6*x + 9 = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3068_306848


namespace NUMINAMATH_CALUDE_total_price_is_23_l3068_306899

-- Define the price of cucumbers per kilogram
def cucumber_price : ℝ := 5

-- Define the price of tomatoes as 20% cheaper than cucumbers
def tomato_price : ℝ := cucumber_price * (1 - 0.2)

-- Define the quantity of tomatoes and cucumbers
def tomato_quantity : ℝ := 2
def cucumber_quantity : ℝ := 3

-- Theorem statement
theorem total_price_is_23 :
  tomato_quantity * tomato_price + cucumber_quantity * cucumber_price = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_price_is_23_l3068_306899


namespace NUMINAMATH_CALUDE_cow_count_l3068_306887

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (g : AnimalGroup) : ℕ := 2 * g.ducks + 4 * g.cows

/-- The total number of heads in the group -/
def totalHeads (g : AnimalGroup) : ℕ := g.ducks + g.cows

/-- Theorem: If the total number of legs is 30 more than twice the number of heads,
    then there are 15 cows in the group -/
theorem cow_count (g : AnimalGroup) :
  totalLegs g = 2 * totalHeads g + 30 → g.cows = 15 := by
  sorry


end NUMINAMATH_CALUDE_cow_count_l3068_306887


namespace NUMINAMATH_CALUDE_monotonic_decreasing_odd_function_property_l3068_306892

-- Define a monotonically decreasing function on ℝ
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Define an odd function on ℝ
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem monotonic_decreasing_odd_function_property
  (f : ℝ → ℝ) (h1 : MonoDecreasing f) (h2 : OddFunction f) :
  -f (-3) < f (-4) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_odd_function_property_l3068_306892


namespace NUMINAMATH_CALUDE_wednesday_temperature_l3068_306893

/-- The temperature on Wednesday given the temperatures for the other days of the week and the average temperature --/
theorem wednesday_temperature
  (sunday : ℝ) (monday : ℝ) (tuesday : ℝ) (thursday : ℝ) (friday : ℝ) (saturday : ℝ) 
  (average : ℝ)
  (h_sunday : sunday = 40)
  (h_monday : monday = 50)
  (h_tuesday : tuesday = 65)
  (h_thursday : thursday = 82)
  (h_friday : friday = 72)
  (h_saturday : saturday = 26)
  (h_average : average = 53)
  : ∃ (wednesday : ℝ), 
    (sunday + monday + tuesday + wednesday + thursday + friday + saturday) / 7 = average ∧ 
    wednesday = 36 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_temperature_l3068_306893


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l3068_306804

theorem similar_triangle_longest_side 
  (a b c : ℝ) 
  (ha : a = 8) 
  (hb : b = 10) 
  (hc : c = 12) 
  (perimeter : ℝ) 
  (h_perimeter : perimeter = 150) 
  (h_similar : ∃ k : ℝ, k > 0 ∧ perimeter = k * (a + b + c)) :
  ∃ longest_side : ℝ, longest_side = 60 ∧ 
    longest_side = max (k * a) (max (k * b) (k * c)) :=
by sorry


end NUMINAMATH_CALUDE_similar_triangle_longest_side_l3068_306804


namespace NUMINAMATH_CALUDE_local_max_value_is_four_l3068_306846

/-- The function f(x) = x^3 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem local_max_value_is_four (a : ℝ) :
  (∃ x : ℝ, IsLocalMin (f a) 1) →
  (∃ x : ℝ, IsLocalMax (f a) x ∧ f a x = 4) :=
by sorry

end NUMINAMATH_CALUDE_local_max_value_is_four_l3068_306846


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_50_l3068_306820

theorem factorization_of_2x_squared_minus_50 (x : ℝ) : 2 * x^2 - 50 = 2 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_50_l3068_306820


namespace NUMINAMATH_CALUDE_smallest_common_factor_l3068_306867

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (6*m + 4))) ∧ 
  (∃ k : ℕ, k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 4)) → 
  n = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l3068_306867


namespace NUMINAMATH_CALUDE_inequality_always_true_implies_a_less_than_seven_l3068_306886

theorem inequality_always_true_implies_a_less_than_seven (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 4| > a) → a < 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_implies_a_less_than_seven_l3068_306886


namespace NUMINAMATH_CALUDE_vanaspati_percentage_after_addition_l3068_306859

/-- Calculates the percentage of vanaspati in a ghee mixture after adding pure ghee -/
theorem vanaspati_percentage_after_addition
  (original_quantity : ℝ)
  (original_pure_ghee_percentage : ℝ)
  (original_vanaspati_percentage : ℝ)
  (added_pure_ghee : ℝ)
  (h1 : original_quantity = 30)
  (h2 : original_pure_ghee_percentage = 50)
  (h3 : original_vanaspati_percentage = 50)
  (h4 : added_pure_ghee = 20)
  (h5 : original_pure_ghee_percentage + original_vanaspati_percentage = 100) :
  let original_vanaspati := original_quantity * (original_vanaspati_percentage / 100)
  let new_total_quantity := original_quantity + added_pure_ghee
  (original_vanaspati / new_total_quantity) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_vanaspati_percentage_after_addition_l3068_306859


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3068_306803

/-- An arithmetic sequence -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def is_geometric (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The sequence a_n + 2^n * b_n forms an arithmetic sequence for n = 1, 3, 5 -/
def special_sequence_arithmetic (a b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, (a 3 + 4 * b 3) - (a 1 + 2 * b 1) = d ∧
            (a 5 + 8 * b 5) - (a 3 + 4 * b 3) = d

theorem geometric_sequence_ratio (a b : ℕ → ℝ) :
  is_arithmetic a →
  is_geometric b →
  special_sequence_arithmetic a b →
  b 3 * b 7 / (b 4 ^ 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3068_306803


namespace NUMINAMATH_CALUDE_lcm_of_150_and_490_l3068_306833

theorem lcm_of_150_and_490 : Nat.lcm 150 490 = 7350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_150_and_490_l3068_306833


namespace NUMINAMATH_CALUDE_min_value_theorem_l3068_306855

theorem min_value_theorem (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m > 0) (h3 : n > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 2 → 1 / m + 2 / n ≤ 1 / x + 2 / y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 ∧ 1 / x + 2 / y = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3068_306855


namespace NUMINAMATH_CALUDE_quadratic_roots_implies_d_l3068_306876

theorem quadratic_roots_implies_d (d : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 8 * x + d = 0 ↔ x = (-8 + Real.sqrt 12) / 4 ∨ x = (-8 - Real.sqrt 12) / 4) → 
  d = 6.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_implies_d_l3068_306876


namespace NUMINAMATH_CALUDE_fraction_equality_l3068_306814

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 4) 
  (h2 : r / t = 8 / 15) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3068_306814


namespace NUMINAMATH_CALUDE_jerry_tickets_l3068_306863

def ticket_calculation (initial_tickets spent_tickets additional_tickets : ℕ) : ℕ :=
  initial_tickets - spent_tickets + additional_tickets

theorem jerry_tickets :
  ticket_calculation 4 2 47 = 49 := by
  sorry

end NUMINAMATH_CALUDE_jerry_tickets_l3068_306863


namespace NUMINAMATH_CALUDE_min_stone_product_l3068_306889

theorem min_stone_product (total_stones : ℕ) (black_stones : ℕ) : 
  total_stones = 40 → 
  black_stones ≥ 20 → 
  black_stones ≤ 32 → 
  (black_stones * (total_stones - black_stones)) ≥ 256 := by
sorry

end NUMINAMATH_CALUDE_min_stone_product_l3068_306889


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3068_306815

theorem triangle_angle_measure
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle opposite to A, B, C respectively
  (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = (1/2) * b)
  (h2 : a > b)
  (h3 : 0 < A ∧ A < π) -- Ensuring A is a valid angle measure
  (h4 : 0 < B ∧ B < π) -- Ensuring B is a valid angle measure
  (h5 : 0 < C ∧ C < π) -- Ensuring C is a valid angle measure
  (h6 : A + B + C = π) -- Sum of angles in a triangle
  : B = π/6 := by
  sorry

#check triangle_angle_measure

end NUMINAMATH_CALUDE_triangle_angle_measure_l3068_306815


namespace NUMINAMATH_CALUDE_problems_per_page_l3068_306883

/-- Given a homework assignment with the following conditions:
  * There are 72 total problems
  * 32 problems have been completed
  * The remaining problems are spread equally across 5 pages
  This theorem proves that there are 8 problems on each remaining page. -/
theorem problems_per_page (total : ℕ) (completed : ℕ) (pages : ℕ) : 
  total = 72 → completed = 32 → pages = 5 → (total - completed) / pages = 8 := by
  sorry

end NUMINAMATH_CALUDE_problems_per_page_l3068_306883


namespace NUMINAMATH_CALUDE_fertilizer_pesticide_cost_l3068_306888

/-- Proves the amount spent on fertilizers and pesticides for a small farm operation --/
theorem fertilizer_pesticide_cost
  (seed_cost : ℝ)
  (labor_cost : ℝ)
  (num_bags : ℕ)
  (price_per_bag : ℝ)
  (profit_percentage : ℝ)
  (h1 : seed_cost = 50)
  (h2 : labor_cost = 15)
  (h3 : num_bags = 10)
  (h4 : price_per_bag = 11)
  (h5 : profit_percentage = 0.1)
  : ∃ (fertilizer_pesticide_cost : ℝ),
    fertilizer_pesticide_cost = 35 ∧
    price_per_bag * num_bags = (1 + profit_percentage) * (seed_cost + labor_cost + fertilizer_pesticide_cost) :=
by sorry

end NUMINAMATH_CALUDE_fertilizer_pesticide_cost_l3068_306888


namespace NUMINAMATH_CALUDE_ada_original_seat_l3068_306845

-- Define the seat numbers
inductive Seat
| one
| two
| three
| four
| five

-- Define the friends
inductive Friend
| Ada
| Bea
| Ceci
| Dee
| Edie

-- Define the seating arrangement as a function from Friend to Seat
def Seating := Friend → Seat

-- Define the movement function
def move (s : Seat) (n : Int) : Seat :=
  match s, n with
  | Seat.one, 1 => Seat.two
  | Seat.one, 2 => Seat.three
  | Seat.two, -1 => Seat.one
  | Seat.two, 1 => Seat.three
  | Seat.two, 2 => Seat.four
  | Seat.three, -1 => Seat.two
  | Seat.three, 1 => Seat.four
  | Seat.three, 2 => Seat.five
  | Seat.four, -1 => Seat.three
  | Seat.four, 1 => Seat.five
  | Seat.five, -1 => Seat.four
  | _, _ => s  -- Default case: no movement

-- Define the theorem
theorem ada_original_seat (initial_seating final_seating : Seating) :
  (∀ f : Friend, f ≠ Friend.Ada → 
    (f = Friend.Bea → move (initial_seating f) 2 = final_seating f) ∧
    (f = Friend.Ceci → move (initial_seating f) (-1) = final_seating f) ∧
    ((f = Friend.Dee ∨ f = Friend.Edie) → 
      (initial_seating Friend.Dee = final_seating Friend.Edie ∧
       initial_seating Friend.Edie = final_seating Friend.Dee))) →
  (final_seating Friend.Ada = Seat.one ∨ final_seating Friend.Ada = Seat.five) →
  initial_seating Friend.Ada = Seat.two :=
sorry

end NUMINAMATH_CALUDE_ada_original_seat_l3068_306845


namespace NUMINAMATH_CALUDE_min_value_sine_l3068_306824

/-- Given that f(x) = 3sin(x) - cos(x) attains its minimum value when x = θ, prove that sin(θ) = -3√10/10 -/
theorem min_value_sine (θ : ℝ) (h : ∀ x, 3 * Real.sin x - Real.cos x ≥ 3 * Real.sin θ - Real.cos θ) : 
  Real.sin θ = -3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sine_l3068_306824


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3068_306838

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_first : a 1 = 1)
  (h_fifth : a 5 = 5) :
  a 3 = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3068_306838


namespace NUMINAMATH_CALUDE_binomial_30_3_l3068_306805

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l3068_306805


namespace NUMINAMATH_CALUDE_fraction_simplification_l3068_306866

theorem fraction_simplification : (2468 * 2468) / (2468 + 2468) = 1234 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3068_306866
