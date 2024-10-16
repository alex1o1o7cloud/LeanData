import Mathlib

namespace NUMINAMATH_CALUDE_S_is_line_l1113_111395

-- Define the set of points satisfying the equation
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 5 * Real.sqrt ((p.1 - 1)^2 + (p.2 - 2)^2) = |3 * p.1 + 4 * p.2 - 11|}

-- Theorem stating that S is a line
theorem S_is_line : ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ S = {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0} :=
sorry

end NUMINAMATH_CALUDE_S_is_line_l1113_111395


namespace NUMINAMATH_CALUDE_computer_price_proof_l1113_111383

theorem computer_price_proof (P : ℝ) : 
  1.20 * P = 351 → 2 * P = 585 → P = 292.50 := by sorry

end NUMINAMATH_CALUDE_computer_price_proof_l1113_111383


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_third_l1113_111356

open Real

theorem derivative_f_at_pi_third (f : ℝ → ℝ) (h : ∀ x, f x = x + Real.sin x) :
  deriv f (π / 3) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_third_l1113_111356


namespace NUMINAMATH_CALUDE_escalator_length_l1113_111353

/-- The length of an escalator given two people walking in opposite directions -/
theorem escalator_length 
  (time_A : ℝ) 
  (time_B : ℝ) 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (h1 : time_A = 100) 
  (h2 : time_B = 300) 
  (h3 : speed_A = 3) 
  (h4 : speed_B = 2) : 
  (speed_A - speed_B) / (1 / time_A - 1 / time_B) = 150 := by
  sorry

end NUMINAMATH_CALUDE_escalator_length_l1113_111353


namespace NUMINAMATH_CALUDE_bisection_method_representation_l1113_111307

/-- Represents different types of diagrams --/
inductive DiagramType
  | OrganizationalStructure
  | ProcessFlowchart
  | KnowledgeStructure
  | ProgramFlowchart

/-- Represents the bisection method algorithm --/
structure BisectionMethod where
  hasLoopStructure : Bool
  hasConditionalStructure : Bool

/-- Theorem stating that the bisection method for solving x^2 - 2 = 0 is best represented by a program flowchart --/
theorem bisection_method_representation (bm : BisectionMethod) 
  (h1 : bm.hasLoopStructure = true) 
  (h2 : bm.hasConditionalStructure = true) : 
  DiagramType.ProgramFlowchart = 
    (fun (d : DiagramType) => 
      if bm.hasLoopStructure ∧ bm.hasConditionalStructure 
      then DiagramType.ProgramFlowchart 
      else d) DiagramType.ProgramFlowchart :=
by
  sorry

#check bisection_method_representation

end NUMINAMATH_CALUDE_bisection_method_representation_l1113_111307


namespace NUMINAMATH_CALUDE_length_of_24_l1113_111305

def length_of_integer (k : ℕ) : ℕ := sorry

theorem length_of_24 :
  let k : ℕ := 24
  length_of_integer k = 4 := by sorry

end NUMINAMATH_CALUDE_length_of_24_l1113_111305


namespace NUMINAMATH_CALUDE_xy_equal_three_l1113_111381

theorem xy_equal_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_equal_three_l1113_111381


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l1113_111339

/-- The y-coordinate of the vertex of the parabola y = -2x^2 + 16x + 72 is 104 -/
theorem parabola_vertex_y_coordinate :
  let f (x : ℝ) := -2 * x^2 + 16 * x + 72
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≤ f x₀ ∧ f x₀ = 104 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l1113_111339


namespace NUMINAMATH_CALUDE_exists_k_greater_than_two_l1113_111377

/-- Given a linear function y = (k-2)x + 3 that is increasing,
    prove that there exists a value of k greater than 2. -/
theorem exists_k_greater_than_two (k : ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ < x₂ → (k - 2) * x₁ + 3 < (k - 2) * x₂ + 3) : 
  ∃ k' : ℝ, k' > 2 := by
sorry

end NUMINAMATH_CALUDE_exists_k_greater_than_two_l1113_111377


namespace NUMINAMATH_CALUDE_ethyne_bond_count_l1113_111337

/-- Represents a chemical bond in a molecule -/
inductive Bond
  | Sigma
  | Pi

/-- Represents the ethyne (acetylene) molecule -/
structure Ethyne where
  /-- The number of carbon atoms in ethyne -/
  carbon_count : Nat
  /-- The number of hydrogen atoms in ethyne -/
  hydrogen_count : Nat
  /-- The structure of ethyne is linear -/
  is_linear : Bool
  /-- Each carbon atom forms a triple bond with the other carbon atom -/
  has_carbon_triple_bond : Bool
  /-- Each carbon atom forms a single bond with a hydrogen atom -/
  has_carbon_hydrogen_single_bond : Bool

/-- Counts the number of sigma bonds in ethyne -/
def count_sigma_bonds (e : Ethyne) : Nat :=
  sorry

/-- Counts the number of pi bonds in ethyne -/
def count_pi_bonds (e : Ethyne) : Nat :=
  sorry

/-- Theorem stating the number of sigma and pi bonds in ethyne -/
theorem ethyne_bond_count (e : Ethyne) :
  e.carbon_count = 2 ∧
  e.hydrogen_count = 2 ∧
  e.is_linear ∧
  e.has_carbon_triple_bond ∧
  e.has_carbon_hydrogen_single_bond →
  count_sigma_bonds e = 3 ∧ count_pi_bonds e = 2 :=
by sorry

end NUMINAMATH_CALUDE_ethyne_bond_count_l1113_111337


namespace NUMINAMATH_CALUDE_product_of_sums_powers_l1113_111378

theorem product_of_sums_powers : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^6 + 1^6) = 2394400 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_powers_l1113_111378


namespace NUMINAMATH_CALUDE_probability_real_roots_l1113_111391

-- Define the interval [0,5]
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 5}

-- Define the condition for real roots
def has_real_roots (p : ℝ) : Prop := p^2 ≥ 4

-- Define the measure of the interval where the equation has real roots
def measure_real_roots : ℝ := 3

-- Define the total measure of the interval
def total_measure : ℝ := 5

-- State the theorem
theorem probability_real_roots : 
  (measure_real_roots / total_measure : ℝ) = 0.6 := by sorry

end NUMINAMATH_CALUDE_probability_real_roots_l1113_111391


namespace NUMINAMATH_CALUDE_cost_of_3000_pencils_l1113_111382

def pencil_cost (quantity : ℕ) : ℚ :=
  let base_price := 36 / 120
  let discount_threshold := 2000
  let discount_factor := 0.9
  if quantity > discount_threshold
  then (quantity : ℚ) * base_price * discount_factor
  else (quantity : ℚ) * base_price

theorem cost_of_3000_pencils :
  pencil_cost 3000 = 810 := by sorry

end NUMINAMATH_CALUDE_cost_of_3000_pencils_l1113_111382


namespace NUMINAMATH_CALUDE_only_integer_solution_l1113_111324

theorem only_integer_solution (x y z : ℝ) (n : ℤ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  2 * x^2 + 3 * y^2 + 6 * z^2 = n →
  3 * x + 4 * y + 5 * z = 23 →
  n = 127 :=
by sorry

end NUMINAMATH_CALUDE_only_integer_solution_l1113_111324


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l1113_111369

theorem quadratic_one_solution_sum (a : ℝ) : 
  (∃! x, 3 * x^2 + a * x + 6 * x + 7 = 0) ↔ 
  (a = -6 + 2 * Real.sqrt 21 ∨ a = -6 - 2 * Real.sqrt 21) ∧
  (-6 + 2 * Real.sqrt 21) + (-6 - 2 * Real.sqrt 21) = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l1113_111369


namespace NUMINAMATH_CALUDE_investment_solution_l1113_111358

def investment_problem (x : ℝ) : Prop :=
  let total_investment : ℝ := 1500
  let rate1 : ℝ := 1.04  -- 4% annual compound interest
  let rate2 : ℝ := 1.06  -- 6% annual compound interest
  let total_after_year : ℝ := 1590
  (x * rate1 + (total_investment - x) * rate2 = total_after_year) ∧
  (0 ≤ x) ∧ (x ≤ total_investment)

theorem investment_solution :
  ∃! x : ℝ, investment_problem x ∧ x = 0 :=
sorry

end NUMINAMATH_CALUDE_investment_solution_l1113_111358


namespace NUMINAMATH_CALUDE_four_Z_three_equals_negative_eleven_l1113_111393

-- Define the Z operation
def Z (c d : ℤ) : ℤ := c^2 - 3*c*d + d^2

-- Theorem to prove
theorem four_Z_three_equals_negative_eleven : Z 4 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_three_equals_negative_eleven_l1113_111393


namespace NUMINAMATH_CALUDE_complement_of_union_is_singleton_one_l1113_111370

-- Define the universal set I
def I : Set Nat := {0, 1, 2, 3}

-- Define set M
def M : Set Nat := {0, 2}

-- Define set N
def N : Set Nat := {0, 2, 3}

-- Theorem statement
theorem complement_of_union_is_singleton_one :
  (M ∪ N)ᶜ = {1} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_singleton_one_l1113_111370


namespace NUMINAMATH_CALUDE_set_membership_implies_m_value_l1113_111327

theorem set_membership_implies_m_value (m : ℚ) : 
  let A : Set ℚ := {m + 2, 2 * m^2 + m}
  3 ∈ A → m = -3/2 := by sorry

end NUMINAMATH_CALUDE_set_membership_implies_m_value_l1113_111327


namespace NUMINAMATH_CALUDE_non_intersecting_paths_l1113_111396

/-- The number of non-intersecting pairs of paths on a grid -/
theorem non_intersecting_paths 
  (m n p q : ℕ+) 
  (h1 : p < m) 
  (h2 : q < n) : 
  ∃ S : ℕ, S = Nat.choose (m + n) m * Nat.choose (m + q - p) q - 
              Nat.choose (m + q) m * Nat.choose (m + n - p) n :=
by sorry

end NUMINAMATH_CALUDE_non_intersecting_paths_l1113_111396


namespace NUMINAMATH_CALUDE_triangle_inequality_l1113_111320

theorem triangle_inequality (a b c p : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : p = (a + b + c) / 2) :
  Real.sqrt (p - a) + Real.sqrt (p - b) + Real.sqrt (p - c) ≤ Real.sqrt (3 * p) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1113_111320


namespace NUMINAMATH_CALUDE_prime_sum_ways_8_l1113_111322

/-- A function that returns the number of unique ways to sum prime numbers to form a given natural number,
    where the prime numbers in the sum are in non-decreasing order. -/
def prime_sum_ways (n : ℕ) : ℕ := sorry

/-- A function that checks if a list of natural numbers is a valid prime sum for a given number,
    where the numbers in the list are prime and in non-decreasing order. -/
def is_valid_prime_sum (n : ℕ) (sum : List ℕ) : Prop := sorry

theorem prime_sum_ways_8 : prime_sum_ways 8 = 2 := by sorry

end NUMINAMATH_CALUDE_prime_sum_ways_8_l1113_111322


namespace NUMINAMATH_CALUDE_unique_base_solution_l1113_111316

/-- Convert a number from base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Check if the equation holds for a given base --/
def equationHolds (b : Nat) : Prop :=
  toDecimal [1, 7, 2] b + toDecimal [1, 5, 6] b = toDecimal [3, 4, 0] b

/-- The main theorem stating that 10 is the unique solution --/
theorem unique_base_solution :
  ∃! b : Nat, b > 1 ∧ equationHolds b :=
  sorry

end NUMINAMATH_CALUDE_unique_base_solution_l1113_111316


namespace NUMINAMATH_CALUDE_fraction_simplification_l1113_111333

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  (x^2 - x*y) / ((x - y)^2) = x / (x - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1113_111333


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1113_111331

theorem greatest_divisor_with_remainders :
  ∃ (d : ℕ), d > 0 ∧
    (150 % d = 50) ∧
    (230 % d = 5) ∧
    (175 % d = 25) ∧
    (∀ (k : ℕ), k > 0 →
      (150 % k = 50) →
      (230 % k = 5) →
      (175 % k = 25) →
      k ≤ d) ∧
    d = 25 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1113_111331


namespace NUMINAMATH_CALUDE_smallest_top_cube_sum_divisible_by_four_l1113_111386

/-- Represents the configuration of the bottom layer of the pyramid -/
structure BottomLayer :=
  (a b c d e f g h i : ℕ)
  (all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
                   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
                   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
                   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
                   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
                   g ≠ h ∧ g ≠ i ∧
                   h ≠ i)

/-- Calculates the sum of the top cube given the bottom layer configuration -/
def topCubeSum (bl : BottomLayer) : ℕ :=
  bl.a + bl.c + bl.g + bl.i + 2 * (bl.b + bl.d + bl.f + bl.h) + 4 * bl.e

/-- Theorem stating that the smallest possible sum for the top cube divisible by 4 is 64 -/
theorem smallest_top_cube_sum_divisible_by_four :
  ∀ bl : BottomLayer, ∃ n : ℕ, n ≥ topCubeSum bl ∧ n % 4 = 0 ∧ n ≥ 64 :=
sorry

end NUMINAMATH_CALUDE_smallest_top_cube_sum_divisible_by_four_l1113_111386


namespace NUMINAMATH_CALUDE_gear_speed_proportion_l1113_111343

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  angular_speed : ℝ

/-- Checks if two gears are meshed -/
def are_meshed (g1 g2 : Gear) : Prop :=
  g1.teeth * g1.angular_speed = g2.teeth * g2.angular_speed

theorem gear_speed_proportion (A B C D : Gear)
  (hA : A.teeth = 30)
  (hB : B.teeth = 45)
  (hC : C.teeth = 50)
  (hD : D.teeth = 60)
  (hAB : are_meshed A B)
  (hBC : are_meshed B C)
  (hCD : are_meshed C D) :
  ∃ (k : ℝ), k > 0 ∧
    A.angular_speed = 10 * k ∧
    B.angular_speed = 10 * k ∧
    C.angular_speed = 9 * k ∧
    D.angular_speed = 7.5 * k :=
sorry

end NUMINAMATH_CALUDE_gear_speed_proportion_l1113_111343


namespace NUMINAMATH_CALUDE_brownies_in_pan_l1113_111364

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℝ := d.length * d.width

/-- Represents the pan and brownie dimensions -/
def pan : Dimensions := ⟨24, 20⟩
def brownie : Dimensions := ⟨3, 2⟩

/-- Theorem stating that the pan can contain exactly 80 brownies -/
theorem brownies_in_pan : 
  (area pan) / (area brownie) = 80 ∧ 
  80 * (area brownie) = area pan := by sorry

end NUMINAMATH_CALUDE_brownies_in_pan_l1113_111364


namespace NUMINAMATH_CALUDE_local_minimum_of_f_l1113_111375

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem local_minimum_of_f :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x ≥ f x₀) ∧ x₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_of_f_l1113_111375


namespace NUMINAMATH_CALUDE_expression_evaluation_l1113_111302

theorem expression_evaluation :
  11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1113_111302


namespace NUMINAMATH_CALUDE_license_plate_combinations_eq_960_l1113_111367

/-- Represents the set of possible characters for each position in the license plate --/
def LicensePlateChoices : Fin 5 → Finset Char :=
  fun i => match i with
    | 0 => {'3', '5', '6', '8', '9'}
    | 1 => {'B', 'C', 'D'}
    | _ => {'1', '3', '6', '9'}

/-- The number of possible license plate combinations --/
def LicensePlateCombinations : ℕ :=
  (LicensePlateChoices 0).card *
  (LicensePlateChoices 1).card *
  (LicensePlateChoices 2).card *
  (LicensePlateChoices 3).card *
  (LicensePlateChoices 4).card

/-- Theorem stating that the number of possible license plate combinations is 960 --/
theorem license_plate_combinations_eq_960 :
  LicensePlateCombinations = 960 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_eq_960_l1113_111367


namespace NUMINAMATH_CALUDE_log_base_values_l1113_111350

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_base_values (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 2 4, f a x ∈ Set.Icc (f a 2) (f a 4)) ∧
  (f a 4 - f a 2 = 2 ∨ f a 2 - f a 4 = 2) →
  a = Real.sqrt 2 ∨ a = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_log_base_values_l1113_111350


namespace NUMINAMATH_CALUDE_equation_solution_unique_l1113_111345

theorem equation_solution_unique :
  ∃! x : ℝ, Real.sqrt x + 3 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 3*x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_unique_l1113_111345


namespace NUMINAMATH_CALUDE_cyclist_distance_l1113_111330

theorem cyclist_distance (x t : ℝ) 
  (h1 : (x + 1/3) * (3*t/4) = x * t)
  (h2 : (x - 1/3) * (t + 3) = x * t) :
  x * t = 132 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_distance_l1113_111330


namespace NUMINAMATH_CALUDE_row_swap_matrix_l1113_111328

theorem row_swap_matrix (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]
  N * A = !![c, d; a, b] := by sorry

end NUMINAMATH_CALUDE_row_swap_matrix_l1113_111328


namespace NUMINAMATH_CALUDE_apple_distribution_l1113_111374

/-- The number of ways to distribute n apples among k people, with each person receiving at least m apples -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- Theorem: There are 190 ways to distribute 30 apples among 3 people, with each person receiving at least 4 apples -/
theorem apple_distribution : distribution_ways 30 3 4 = 190 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1113_111374


namespace NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l1113_111354

theorem correct_calculation : 2 * Real.sqrt 5 * Real.sqrt 5 = 10 :=
by sorry

theorem incorrect_calculation_A : ¬(Real.sqrt 2 + Real.sqrt 5 = Real.sqrt 7) :=
by sorry

theorem incorrect_calculation_B : ¬(2 * Real.sqrt 3 - Real.sqrt 3 = 2) :=
by sorry

theorem incorrect_calculation_C : ¬(Real.sqrt (3^2 - 2^2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l1113_111354


namespace NUMINAMATH_CALUDE_mike_car_expenses_l1113_111348

def speakers : ℚ := 118.54
def tires : ℚ := 106.33
def windowTints : ℚ := 85.27
def seatCovers : ℚ := 79.99
def maintenance : ℚ := 199.75
def steeringWheelCover : ℚ := 15.63
def airFresheners : ℚ := 6.48 * 2  -- Assuming one set of two
def carWash : ℚ := 25

def totalExpenses : ℚ := speakers + tires + windowTints + seatCovers + maintenance + steeringWheelCover + airFresheners + carWash

theorem mike_car_expenses :
  totalExpenses = 643.47 := by sorry

end NUMINAMATH_CALUDE_mike_car_expenses_l1113_111348


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l1113_111303

/-- The set of integers from which we select numbers -/
def S : Set ℕ := {1, 2, 3, 4, 5, 6}

/-- A pair of numbers selected from S -/
def Selection := (ℕ × ℕ)

/-- Predicate for a number being even -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Predicate for a number being odd -/
def is_odd (n : ℕ) : Prop := n % 2 ≠ 0

/-- Event: Exactly one is even and exactly one is odd -/
def event1 (s : Selection) : Prop :=
  (is_even s.1 ∧ is_odd s.2) ∨ (is_odd s.1 ∧ is_even s.2)

/-- Event: At least one is odd and both are odd -/
def event2 (s : Selection) : Prop :=
  is_odd s.1 ∧ is_odd s.2

/-- Event: At least one is odd and both are even -/
def event3 (s : Selection) : Prop :=
  (is_odd s.1 ∨ is_odd s.2) ∧ (is_even s.1 ∧ is_even s.2)

/-- Event: At least one is odd and at least one is even -/
def event4 (s : Selection) : Prop :=
  (is_odd s.1 ∨ is_odd s.2) ∧ (is_even s.1 ∨ is_even s.2)

theorem mutually_exclusive_events :
  ∀ (s : Selection), s.1 ∈ S ∧ s.2 ∈ S →
    (¬(event1 s ∧ event2 s) ∧
     ¬(event1 s ∧ event3 s) ∧
     ¬(event1 s ∧ event4 s) ∧
     ¬(event2 s ∧ event3 s) ∧
     ¬(event2 s ∧ event4 s) ∧
     ¬(event3 s ∧ event4 s)) ∧
    (event3 s → ¬event1 s ∧ ¬event2 s ∧ ¬event4 s) :=
by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l1113_111303


namespace NUMINAMATH_CALUDE_shirts_total_cost_l1113_111325

/-- Calculates the total cost of shirts with given prices, quantities, discounts, and taxes -/
def totalCost (price1 price2 : ℝ) (quantity1 quantity2 : ℕ) (discount tax : ℝ) : ℝ :=
  quantity1 * (price1 * (1 - discount)) + quantity2 * (price2 * (1 + tax))

/-- Theorem stating that the total cost of the shirts is $82.50 -/
theorem shirts_total_cost :
  totalCost 15 20 3 2 0.1 0.05 = 82.5 := by
  sorry

#eval totalCost 15 20 3 2 0.1 0.05

end NUMINAMATH_CALUDE_shirts_total_cost_l1113_111325


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1113_111340

theorem negation_of_existential_proposition :
  ¬(∃ x : ℝ, x < 0 ∧ x^2 - 2*x > 0) ↔ (∀ x : ℝ, x < 0 → x^2 - 2*x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1113_111340


namespace NUMINAMATH_CALUDE_root_range_theorem_l1113_111366

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-1)*x + m^2 - 2

theorem root_range_theorem (m : ℝ) : 
  (∃ x y : ℝ, x < -1 ∧ y > 1 ∧ 
   f m x = 0 ∧ f m y = 0 ∧ 
   ∀ z : ℝ, f m z = 0 → z = x ∨ z = y) ↔ 
  m > 0 ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_root_range_theorem_l1113_111366


namespace NUMINAMATH_CALUDE_local_tax_deduction_l1113_111355

-- Define Carl's hourly wage in dollars
def carlHourlyWage : ℝ := 25

-- Define the local tax rate as a percentage
def localTaxRate : ℝ := 2.0

-- Define the conversion rate from dollars to cents
def dollarsToCents : ℝ := 100

-- Theorem to prove
theorem local_tax_deduction :
  (carlHourlyWage * dollarsToCents * (localTaxRate / 100)) = 50 := by
  sorry


end NUMINAMATH_CALUDE_local_tax_deduction_l1113_111355


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1113_111357

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64 / 9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1113_111357


namespace NUMINAMATH_CALUDE_bushes_for_zucchinis_l1113_111379

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries needed to trade for one zucchini -/
def containers_per_zucchini : ℕ := 3

/-- The number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 60

/-- The number of bushes needed to obtain the target number of zucchinis -/
def bushes_needed : ℕ := 18

theorem bushes_for_zucchinis :
  bushes_needed * containers_per_bush = target_zucchinis * containers_per_zucchini :=
by sorry

end NUMINAMATH_CALUDE_bushes_for_zucchinis_l1113_111379


namespace NUMINAMATH_CALUDE_x_varies_as_z_to_four_thirds_l1113_111349

/-- Given that x varies directly as the fourth power of y and y varies as the cube root of z,
    prove that x varies as the (4/3)th power of z. -/
theorem x_varies_as_z_to_four_thirds
  (h1 : ∃ (k : ℝ), ∀ (x y : ℝ), x = k * y^4)
  (h2 : ∃ (j : ℝ), ∀ (y z : ℝ), y = j * z^(1/3))
  : ∃ (m : ℝ), ∀ (x z : ℝ), x = m * z^(4/3) := by
  sorry

end NUMINAMATH_CALUDE_x_varies_as_z_to_four_thirds_l1113_111349


namespace NUMINAMATH_CALUDE_shrub_height_after_two_years_l1113_111312

def shrub_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

theorem shrub_height_after_two_years 
  (h : shrub_height (shrub_height 9 2) 3 = 243) : 
  shrub_height 9 2 = 9 :=
by
  sorry

#check shrub_height_after_two_years

end NUMINAMATH_CALUDE_shrub_height_after_two_years_l1113_111312


namespace NUMINAMATH_CALUDE_original_price_of_meat_pack_original_price_is_40_l1113_111380

/-- The original price of a 4 pack of fancy, sliced meat, given rush delivery conditions -/
theorem original_price_of_meat_pack : ℝ :=
  let rush_delivery_factor : ℝ := 1.3
  let price_with_rush : ℝ := 13
  let pack_size : ℕ := 4
  let single_meat_price : ℝ := price_with_rush / rush_delivery_factor
  pack_size * single_meat_price

/-- Proof that the original price of the 4 pack is $40 -/
theorem original_price_is_40 : original_price_of_meat_pack = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_price_of_meat_pack_original_price_is_40_l1113_111380


namespace NUMINAMATH_CALUDE_bread_slices_proof_l1113_111323

theorem bread_slices_proof (S : ℕ) : S ≥ 20 → (∃ T : ℕ, S = 2 * T + 10 ∧ S - 7 = 2 * T + 3) → S ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_proof_l1113_111323


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1113_111373

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1113_111373


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l1113_111304

theorem square_sum_geq_product_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a ∧
  (a^2 + b^2 + c^2 = a*b + b*c + c*a ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l1113_111304


namespace NUMINAMATH_CALUDE_paint_remaining_l1113_111346

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint = 1 → 
  (initial_paint - initial_paint / 4) / 2 = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_paint_remaining_l1113_111346


namespace NUMINAMATH_CALUDE_glass_capacity_l1113_111365

/-- Proves that if 10 glasses are 4/5 full and require 12 ounces of water to fill them completely,
    then each glass has a capacity of 6 ounces. -/
theorem glass_capacity (num_glasses : ℕ) (fullness_ratio : ℚ) (water_needed : ℚ) :
  num_glasses = 10 →
  fullness_ratio = 4/5 →
  water_needed = 12 →
  (1 - fullness_ratio) * (water_needed / num_glasses) * (1 / (1 - fullness_ratio)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_glass_capacity_l1113_111365


namespace NUMINAMATH_CALUDE_rhombus_area_l1113_111319

/-- The area of a rhombus with side length 4 and an angle of 45 degrees between adjacent sides is 8√2 -/
theorem rhombus_area (side : ℝ) (angle : ℝ) : 
  side = 4 → 
  angle = Real.pi / 4 → 
  (side * side * Real.sin angle : ℝ) = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1113_111319


namespace NUMINAMATH_CALUDE_cow_count_l1113_111308

theorem cow_count (total_legs : ℕ) (legs_per_cow : ℕ) (h1 : total_legs = 460) (h2 : legs_per_cow = 4) : 
  total_legs / legs_per_cow = 115 := by
sorry

end NUMINAMATH_CALUDE_cow_count_l1113_111308


namespace NUMINAMATH_CALUDE_calculate_fraction_l1113_111376

theorem calculate_fraction : (2015^2) / (2014^2 + 2016^2 - 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_fraction_l1113_111376


namespace NUMINAMATH_CALUDE_geometric_progression_min_sum_l1113_111301

/-- A geometric progression with positive terms -/
def GeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_progression_min_sum (a : ℕ → ℝ) (h : GeometricProgression a) 
    (h_prod : a 2 * a 10 = 9) : 
  a 5 + a 7 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_min_sum_l1113_111301


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l1113_111392

theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -8)
  let distance_to_x_axis := |P.2|
  let distance_to_y_axis := |P.1|
  distance_to_x_axis = (1/2 : ℝ) * distance_to_y_axis →
  distance_to_y_axis = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l1113_111392


namespace NUMINAMATH_CALUDE_decompose_375_l1113_111300

theorem decompose_375 : 
  375 = 3 * 100 + 7 * 10 + 5 * 1 := by sorry

end NUMINAMATH_CALUDE_decompose_375_l1113_111300


namespace NUMINAMATH_CALUDE_prob_four_green_marbles_l1113_111347

def total_marbles : ℕ := 15
def green_marbles : ℕ := 10
def purple_marbles : ℕ := 5
def total_draws : ℕ := 8
def green_draws : ℕ := 4

theorem prob_four_green_marbles :
  (Nat.choose total_draws green_draws : ℚ) *
  (green_marbles / total_marbles : ℚ) ^ green_draws *
  (purple_marbles / total_marbles : ℚ) ^ (total_draws - green_draws) =
  1120 / 6561 := by sorry

end NUMINAMATH_CALUDE_prob_four_green_marbles_l1113_111347


namespace NUMINAMATH_CALUDE_division_remainder_l1113_111311

theorem division_remainder (j : ℕ) (h1 : j > 0) (h2 : 132 % (j^2) = 12) : 250 % j = 0 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1113_111311


namespace NUMINAMATH_CALUDE_number_of_cows_l1113_111398

-- Define the types for animals
inductive Animal : Type
| Cow : Animal
| Chicken : Animal
| Pig : Animal

-- Define the farm
def Farm : Type := Animal → ℕ

-- Define the number of legs for each animal
def legs : Animal → ℕ
| Animal.Cow => 4
| Animal.Chicken => 2
| Animal.Pig => 4

-- Define the total number of animals
def total_animals (farm : Farm) : ℕ :=
  farm Animal.Cow + farm Animal.Chicken + farm Animal.Pig

-- Define the total number of legs
def total_legs (farm : Farm) : ℕ :=
  farm Animal.Cow * legs Animal.Cow +
  farm Animal.Chicken * legs Animal.Chicken +
  farm Animal.Pig * legs Animal.Pig

-- State the theorem
theorem number_of_cows (farm : Farm) : 
  farm Animal.Chicken = 6 ∧ 
  total_legs farm = 20 + 2 * total_animals farm → 
  farm Animal.Cow = 6 :=
sorry

end NUMINAMATH_CALUDE_number_of_cows_l1113_111398


namespace NUMINAMATH_CALUDE_gcd_problem_l1113_111306

theorem gcd_problem (b : ℤ) (h : 1632 ∣ b) :
  Int.gcd (b^2 + 11*b + 30) (b + 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1113_111306


namespace NUMINAMATH_CALUDE_travel_ways_theorem_l1113_111387

/-- The number of different ways to travel between two places given the number of bus, train, and ferry routes -/
def total_travel_ways (buses trains ferries : ℕ) : ℕ :=
  buses + trains + ferries

/-- Theorem stating that with 5 buses, 6 trains, and 2 ferries, there are 13 ways to travel -/
theorem travel_ways_theorem :
  total_travel_ways 5 6 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_theorem_l1113_111387


namespace NUMINAMATH_CALUDE_problem_solution_l1113_111338

theorem problem_solution (n k : ℕ) : 
  (1/2)^n * (1/81)^k = 1/18^22 → k = 11 → n = 22 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1113_111338


namespace NUMINAMATH_CALUDE_perimeter_is_twenty_l1113_111317

/-- The perimeter of a six-sided figure with specified side lengths -/
def perimeter_of_figure (h1 h2 v1 v2 v3 v4 : ℕ) : ℕ :=
  h1 + h2 + v1 + v2 + v3 + v4

/-- Theorem: The perimeter of the given figure is 20 units -/
theorem perimeter_is_twenty :
  ∃ (h1 h2 v1 v2 v3 v4 : ℕ),
    h1 + h2 = 5 ∧
    v1 = 2 ∧ v2 = 3 ∧ v3 = 3 ∧ v4 = 2 ∧
    perimeter_of_figure h1 h2 v1 v2 v3 v4 = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_perimeter_is_twenty_l1113_111317


namespace NUMINAMATH_CALUDE_inequality_proof_l1113_111336

theorem inequality_proof (x : ℝ) (h1 : 0 < x) (h2 : x < 20) :
  Real.sqrt x + Real.sqrt (20 - x) ≤ 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1113_111336


namespace NUMINAMATH_CALUDE_bryans_precious_stones_l1113_111315

theorem bryans_precious_stones (price_per_stone : ℕ) (total_amount : ℕ) (h1 : price_per_stone = 1785) (h2 : total_amount = 14280) :
  total_amount / price_per_stone = 8 := by
  sorry

end NUMINAMATH_CALUDE_bryans_precious_stones_l1113_111315


namespace NUMINAMATH_CALUDE_three_valid_configurations_l1113_111385

/-- Represents a square in the figure -/
structure Square :=
  (id : Nat)

/-- Represents the cross-shaped figure -/
def CrossFigure := List Square

/-- Represents the additional squares -/
def AdditionalSquares := List Square

/-- Represents a configuration after adding a square to the cross figure -/
def Configuration := CrossFigure × Square

/-- Checks if a configuration can be folded into a topless cubical box -/
def canFoldIntoCube (config : Configuration) : Bool :=
  sorry

/-- The main theorem stating that exactly three configurations can be folded into a topless cubical box -/
theorem three_valid_configurations 
  (cross : CrossFigure) 
  (additional : AdditionalSquares) : 
  (cross.length = 5) → 
  (additional.length = 8) → 
  (∃! (n : Nat), n = (List.filter canFoldIntoCube (List.map (λ s => (cross, s)) additional)).length ∧ n = 3) :=
sorry

end NUMINAMATH_CALUDE_three_valid_configurations_l1113_111385


namespace NUMINAMATH_CALUDE_circle_area_l1113_111351

theorem circle_area (r : ℝ) (h : 6 / (2 * Real.pi * r) = 2 * r) : 
  Real.pi * r^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l1113_111351


namespace NUMINAMATH_CALUDE_hula_hoop_radius_l1113_111372

theorem hula_hoop_radius (diameter : ℝ) (h : diameter = 14) : diameter / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_hula_hoop_radius_l1113_111372


namespace NUMINAMATH_CALUDE_three_lines_intersection_l1113_111371

/-- Three lines intersect at a single point if and only if k = -2/7 --/
theorem three_lines_intersection (x y k : ℚ) : 
  (y = 3*x + 2 ∧ y = -4*x - 14 ∧ y = 2*x + k) ↔ k = -2/7 := by
  sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l1113_111371


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1113_111368

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a parabola y^2 = 8x and a line passing through P(1, -1) intersecting 
    the parabola at points A and B, where P is the midpoint of AB, 
    prove that the equation of line AB is 4x + y - 3 = 0 -/
theorem parabola_line_intersection 
  (para : Parabola) 
  (P : Point) 
  (A B : Point) 
  (line : Line) : 
  para.p = 4 → 
  P.x = 1 → 
  P.y = -1 → 
  (A.x + B.x) / 2 = P.x → 
  (A.y + B.y) / 2 = P.y → 
  A.y^2 = 8 * A.x → 
  B.y^2 = 8 * B.x → 
  line.a * A.x + line.b * A.y + line.c = 0 → 
  line.a * B.x + line.b * B.y + line.c = 0 → 
  line.a = 4 ∧ line.b = 1 ∧ line.c = -3 := by 
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1113_111368


namespace NUMINAMATH_CALUDE_car_round_trip_speed_l1113_111341

theorem car_round_trip_speed 
  (distance : ℝ) 
  (speed_there : ℝ) 
  (avg_speed : ℝ) 
  (speed_back : ℝ) : 
  distance = 150 → 
  speed_there = 75 → 
  avg_speed = 50 → 
  (2 * distance) / (distance / speed_there + distance / speed_back) = avg_speed →
  speed_back = 37.5 := by
sorry

end NUMINAMATH_CALUDE_car_round_trip_speed_l1113_111341


namespace NUMINAMATH_CALUDE_incorrect_simplification_l1113_111335

theorem incorrect_simplification : 
  -(1 + 1/2) ≠ 1 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_simplification_l1113_111335


namespace NUMINAMATH_CALUDE_C_formula_l1113_111309

/-- 
C(n, p) represents the number of decompositions of n into sums of powers of p, 
where each power p^k appears at most p^2 - 1 times
-/
def C (n p : ℕ) : ℕ := sorry

/-- Theorem stating the formula for C(n, p) -/
theorem C_formula (n p : ℕ) (hp : p > 1) : C n p = n / p + 1 := by sorry

end NUMINAMATH_CALUDE_C_formula_l1113_111309


namespace NUMINAMATH_CALUDE_gcf_of_60_and_90_l1113_111399

theorem gcf_of_60_and_90 : Nat.gcd 60 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_and_90_l1113_111399


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1113_111361

theorem min_value_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  1/a + 1/b + 1/c ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1113_111361


namespace NUMINAMATH_CALUDE_square_field_perimeter_l1113_111360

theorem square_field_perimeter (a p : ℝ) (h1 : a = a^2) (h2 : 6*a = 6*(2*p + 9)) :
  p = 36 := by sorry

end NUMINAMATH_CALUDE_square_field_perimeter_l1113_111360


namespace NUMINAMATH_CALUDE_matthews_sharing_solution_l1113_111362

/-- Represents the problem of Matthew sharing crackers and cakes with friends. -/
def MatthewsSharingProblem (total_crackers total_cakes items_per_friend : ℕ) : Prop :=
  let total_items := total_crackers + total_cakes
  let max_friends := total_items / items_per_friend
  max_friends = 3 ∧
  max_friends * items_per_friend ≤ total_items ∧
  (max_friends + 1) * items_per_friend > total_items

/-- Theorem stating the solution to Matthew's sharing problem. -/
theorem matthews_sharing_solution :
  MatthewsSharingProblem 14 21 10 :=
by
  sorry

#check matthews_sharing_solution

end NUMINAMATH_CALUDE_matthews_sharing_solution_l1113_111362


namespace NUMINAMATH_CALUDE_smallest_of_four_consecutive_integers_product_2520_l1113_111318

theorem smallest_of_four_consecutive_integers_product_2520 :
  ∃ (n : ℕ), n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 2520 ∧
  ∀ (m : ℕ), m > 0 → m * (m + 1) * (m + 2) * (m + 3) = 2520 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_of_four_consecutive_integers_product_2520_l1113_111318


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_rational_equation_l1113_111329

theorem negation_of_existence (P : ℚ → Prop) : 
  (¬ ∃ x : ℚ, P x) ↔ (∀ x : ℚ, ¬ P x) := by sorry

theorem negation_of_rational_equation : 
  (¬ ∃ x : ℚ, x - 2 = 0) ↔ (∀ x : ℚ, x - 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_rational_equation_l1113_111329


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_of_1000800000_l1113_111334

def n : ℕ := 1000800000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a > b ∧ b > c ∧ c > d ∧ d > e ∧
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    ∀ (x : ℕ), x ∣ n → x ≤ e ∨ x = d ∨ x = c ∨ x = b ∨ x = a ∨ x = n)

theorem fifth_largest_divisor_of_1000800000 :
  is_fifth_largest_divisor 62550000 :=
sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_of_1000800000_l1113_111334


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1113_111359

theorem complex_number_in_fourth_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := (2 - i) / (1 + i)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1113_111359


namespace NUMINAMATH_CALUDE_a_minus_b_equals_two_l1113_111384

theorem a_minus_b_equals_two (a b : ℝ) 
  (eq1 : 4 * a + 3 * b = 8) 
  (eq2 : 3 * a + 4 * b = 6) : 
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_two_l1113_111384


namespace NUMINAMATH_CALUDE_dodgeball_tournament_l1113_111344

theorem dodgeball_tournament (N : ℕ) : 
  (∃ W D : ℕ, 
    W + D = N * (N - 1) / 2 ∧ 
    15 * W + 22 * D = 1151) → 
  N = 12 := by
sorry

end NUMINAMATH_CALUDE_dodgeball_tournament_l1113_111344


namespace NUMINAMATH_CALUDE_student_average_mark_l1113_111352

/-- Given a student's marks in 5 subjects, prove that the average mark in 4 subjects
    (excluding physics) is 70, when the total marks are 280 more than the physics marks. -/
theorem student_average_mark (physics chemistry maths biology english : ℕ) :
  physics + chemistry + maths + biology + english = physics + 280 →
  (chemistry + maths + biology + english) / 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_student_average_mark_l1113_111352


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1113_111388

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 1) / (4 * x^2 - 1) = 3 / (2 * x + 1) - 4 / (4 * x - 2) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1113_111388


namespace NUMINAMATH_CALUDE_prime_counting_upper_bound_l1113_111313

open Real

/-- The prime counting function π(n) -/
noncomputable def prime_counting (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n > 55, π(n) < 3 ln 2 * (n / ln n) -/
theorem prime_counting_upper_bound (n : ℕ) (h : n > 55) :
  (prime_counting n : ℝ) < 3 * log 2 * (n / log n) := by
  sorry

end NUMINAMATH_CALUDE_prime_counting_upper_bound_l1113_111313


namespace NUMINAMATH_CALUDE_set_B_equals_l1113_111363

def A : Set Int := {-2, -1, 1, 2, 3, 4}

def B : Set Int := {x | ∃ t ∈ A, x = t^2}

theorem set_B_equals : B = {1, 4, 9, 16} := by sorry

end NUMINAMATH_CALUDE_set_B_equals_l1113_111363


namespace NUMINAMATH_CALUDE_billing_method_comparison_l1113_111326

/-- Cost calculation for Method A -/
def cost_A (x : ℝ) : ℝ := 8 + 0.2 * x

/-- Cost calculation for Method B -/
def cost_B (x : ℝ) : ℝ := 0.3 * x

/-- Theorem comparing billing methods based on call duration -/
theorem billing_method_comparison (x : ℝ) :
  (x < 80 → cost_B x < cost_A x) ∧
  (x = 80 → cost_A x = cost_B x) ∧
  (x > 80 → cost_A x < cost_B x) := by
  sorry

end NUMINAMATH_CALUDE_billing_method_comparison_l1113_111326


namespace NUMINAMATH_CALUDE_tower_combinations_l1113_111397

def red_cubes : ℕ := 2
def blue_cubes : ℕ := 4
def green_cubes : ℕ := 5
def tower_height : ℕ := 7

/-- The number of different towers with a height of 7 cubes that can be built
    with 2 red cubes, 4 blue cubes, and 5 green cubes. -/
def number_of_towers : ℕ := 420

theorem tower_combinations :
  (red_cubes + blue_cubes + green_cubes - tower_height = 2) →
  number_of_towers = 420 :=
by sorry

end NUMINAMATH_CALUDE_tower_combinations_l1113_111397


namespace NUMINAMATH_CALUDE_catch_up_time_l1113_111332

-- Define the velocities of objects A and B
def v_A (t : ℝ) : ℝ := 3 * t^2 + 1
def v_B (t : ℝ) : ℝ := 10 * t

-- Define the distances traveled by objects A and B
def d_A (t : ℝ) : ℝ := t^3 + t
def d_B (t : ℝ) : ℝ := 5 * t^2 + 5

-- Theorem: Object A catches up with object B at t = 5 seconds
theorem catch_up_time : 
  ∃ t : ℝ, t = 5 ∧ d_A t = d_B t :=
sorry

end NUMINAMATH_CALUDE_catch_up_time_l1113_111332


namespace NUMINAMATH_CALUDE_five_variable_inequality_l1113_111394

theorem five_variable_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 4*(x₁*x₂ + x₃*x₄ + x₅*x₁ + x₂*x₃ + x₄*x₅) := by
  sorry

end NUMINAMATH_CALUDE_five_variable_inequality_l1113_111394


namespace NUMINAMATH_CALUDE_class_size_l1113_111310

theorem class_size (N M S : ℕ) 
  (h1 : N - M = 10)
  (h2 : N - S = 15)
  (h3 : N - (M + S - 7) = 2)
  (h4 : M + S = N + 7) : N = 34 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1113_111310


namespace NUMINAMATH_CALUDE_mixed_number_multiplication_l1113_111342

theorem mixed_number_multiplication :
  99 * (24 / 25) * (-5) = -(499 + 4 / 5) := by sorry

end NUMINAMATH_CALUDE_mixed_number_multiplication_l1113_111342


namespace NUMINAMATH_CALUDE_field_trip_problem_l1113_111314

/-- Given a field trip with vans and buses, calculates the number of people in each van. -/
def peoplePerVan (numVans : ℕ) (numBuses : ℕ) (peoplePerBus : ℕ) (totalPeople : ℕ) : ℕ :=
  (totalPeople - numBuses * peoplePerBus) / numVans

theorem field_trip_problem :
  peoplePerVan 6 8 18 180 = 6 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_problem_l1113_111314


namespace NUMINAMATH_CALUDE_quadratic_real_root_l1113_111389

theorem quadratic_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ∈ Set.Ici 10 ∪ Set.Iic (-10) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l1113_111389


namespace NUMINAMATH_CALUDE_loan_amount_correct_l1113_111321

/-- The amount of money (in Rs.) that A lent to B -/
def loan_amount : ℝ := 3500

/-- B's net interest rate per annum (as a decimal) -/
def net_interest_rate : ℝ := 0.01

/-- B's gain in 3 years (in Rs.) -/
def gain_in_three_years : ℝ := 105

/-- Proves that the loan amount is correct given the conditions -/
theorem loan_amount_correct : 
  loan_amount * net_interest_rate * 3 = gain_in_three_years :=
by sorry

end NUMINAMATH_CALUDE_loan_amount_correct_l1113_111321


namespace NUMINAMATH_CALUDE_garden_perimeter_l1113_111390

/-- The perimeter of a rectangle given its length and breadth -/
def rectangle_perimeter (length : ℝ) (breadth : ℝ) : ℝ :=
  2 * (length + breadth)

/-- Theorem: The perimeter of a rectangular garden with length 140 m and breadth 100 m is 480 m -/
theorem garden_perimeter :
  rectangle_perimeter 140 100 = 480 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1113_111390
