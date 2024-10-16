import Mathlib

namespace NUMINAMATH_CALUDE_joan_balloons_l1279_127920

/-- Given that Joan initially has 9 blue balloons and loses 2 balloons,
    prove that she has 7 blue balloons remaining. -/
theorem joan_balloons : 
  let initial_balloons : ℕ := 9
  let lost_balloons : ℕ := 2
  initial_balloons - lost_balloons = 7 := by
sorry

end NUMINAMATH_CALUDE_joan_balloons_l1279_127920


namespace NUMINAMATH_CALUDE_initial_chairs_count_l1279_127990

theorem initial_chairs_count (initial_chairs : ℕ) 
  (h1 : initial_chairs - (initial_chairs - 3) = 12) : initial_chairs = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_chairs_count_l1279_127990


namespace NUMINAMATH_CALUDE_track_meet_attendance_l1279_127900

theorem track_meet_attendance :
  ∀ (total boys girls long_hair short_hair : ℕ),
    boys = 30 →
    total = boys + girls →
    long_hair = (3 * girls) / 5 →
    short_hair = (2 * girls) / 5 →
    short_hair = 10 →
    total = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_track_meet_attendance_l1279_127900


namespace NUMINAMATH_CALUDE_three_digit_congruence_solutions_l1279_127984

theorem three_digit_congruence_solutions : 
  (Finset.filter (fun y : ℕ => 100 ≤ y ∧ y ≤ 999 ∧ (1945 * y + 243) % 17 = 605 % 17) 
    (Finset.range 1000)).card = 53 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_congruence_solutions_l1279_127984


namespace NUMINAMATH_CALUDE_percentage_of_150_to_60_prove_percentage_l1279_127985

theorem percentage_of_150_to_60 : Real → Prop :=
  fun x => (150 / 60) * 100 = x

theorem prove_percentage :
  ∃ x, percentage_of_150_to_60 x ∧ x = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_150_to_60_prove_percentage_l1279_127985


namespace NUMINAMATH_CALUDE_fraction_problem_l1279_127925

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  (0.4 * N = 180) → 
  (f * (1/3) * (2/5) * N = 15) → 
  f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1279_127925


namespace NUMINAMATH_CALUDE_scientific_notation_of_2600000_l1279_127910

theorem scientific_notation_of_2600000 :
  2600000 = 2.6 * (10 ^ 6) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2600000_l1279_127910


namespace NUMINAMATH_CALUDE_distinct_values_count_l1279_127901

def standard_eval : ℕ := 3^(3^(3^3))

def alt_evals : List ℕ := [
  3^((3^3)^3),
  ((3^3)^3)^3,
  (3^(3^3))^3,
  (3^3)^(3^3)
]

theorem distinct_values_count :
  (alt_evals.filter (· ≠ standard_eval)).length = 1 := by sorry

end NUMINAMATH_CALUDE_distinct_values_count_l1279_127901


namespace NUMINAMATH_CALUDE_matching_color_probability_l1279_127961

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Abe's jelly bean distribution -/
def abe : JellyBeans :=
  { green := 2, red := 2, blue := 1, yellow := 0 }

/-- Clara's jelly bean distribution -/
def clara : JellyBeans :=
  { green := 3, red := 2, blue := 1, yellow := 2 }

/-- Calculates the probability of picking a specific color -/
def prob_color (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / jb.total

/-- Theorem: The probability of Abe and Clara showing the same color is 11/40 -/
theorem matching_color_probability :
  (prob_color abe abe.green * prob_color clara clara.green) +
  (prob_color abe abe.red * prob_color clara clara.red) +
  (prob_color abe abe.blue * prob_color clara clara.blue) = 11 / 40 := by
  sorry

end NUMINAMATH_CALUDE_matching_color_probability_l1279_127961


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1279_127952

theorem decimal_to_fraction (x : ℚ) (h : x = 3.68) : 
  ∃ (n d : ℕ), d ≠ 0 ∧ x = n / d ∧ n = 92 ∧ d = 25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1279_127952


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1279_127905

def arithmetic_sequence (a : ℝ) (n : ℕ) : ℝ := a + (n - 1) * (a + 1 - (a - 1))

theorem arithmetic_sequence_formula (a : ℝ) :
  (arithmetic_sequence a 1 = a - 1) ∧
  (arithmetic_sequence a 2 = a + 1) ∧
  (arithmetic_sequence a 3 = 2 * a + 3) →
  ∀ n : ℕ, arithmetic_sequence a n = 2 * n - 3 :=
by
  sorry

#check arithmetic_sequence_formula

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1279_127905


namespace NUMINAMATH_CALUDE_jillian_oranges_l1279_127941

/-- Given that Jillian divides oranges into pieces for her friends, 
    this theorem proves the number of oranges she had. -/
theorem jillian_oranges 
  (pieces_per_orange : ℕ) 
  (pieces_per_friend : ℕ) 
  (num_friends : ℕ) 
  (h1 : pieces_per_orange = 10) 
  (h2 : pieces_per_friend = 4) 
  (h3 : num_friends = 200) : 
  (num_friends * pieces_per_friend) / pieces_per_orange = 80 := by
  sorry

end NUMINAMATH_CALUDE_jillian_oranges_l1279_127941


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1279_127982

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {2, 4}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1279_127982


namespace NUMINAMATH_CALUDE_cube_sum_equals_110_l1279_127987

theorem cube_sum_equals_110 (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 6) : 
  x^3 + y^3 = 110 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_110_l1279_127987


namespace NUMINAMATH_CALUDE_parabola_chord_length_l1279_127937

/-- Given a parabola y² = 4x with a chord passing through its focus and endpoints A(x₁, y₁) and B(x₂, y₂),
    if x₁ + x₂ = 6, then the length of AB is 8. -/
theorem parabola_chord_length (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 4*x₁ → y₂^2 = 4*x₂ → x₁ + x₂ = 6 → 
  ∃ (AB : ℝ), AB = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ∧ AB = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l1279_127937


namespace NUMINAMATH_CALUDE_greatest_base_six_digit_sum_l1279_127908

/-- Represents a positive integer in base 6 as a list of digits (least significant first) -/
def BaseNRepr (n : ℕ) : List ℕ :=
  sorry

/-- Computes the sum of digits in a base 6 representation -/
def sumDigits (repr : List ℕ) : ℕ :=
  sorry

theorem greatest_base_six_digit_sum :
  (∀ n : ℕ, n > 0 → n < 2401 → sumDigits (BaseNRepr n) ≤ 12) ∧
  (∃ n : ℕ, n > 0 ∧ n < 2401 ∧ sumDigits (BaseNRepr n) = 12) :=
sorry

end NUMINAMATH_CALUDE_greatest_base_six_digit_sum_l1279_127908


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1279_127923

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  (3 / 4) * (8 / x - 15 * x^3 + 6 * x) = 6 / x - 45 / 4 * x^3 + 9 / 2 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1279_127923


namespace NUMINAMATH_CALUDE_inequality_theorem_l1279_127927

theorem inequality_theorem (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a + b + (1/2 : ℝ) ≥ Real.sqrt a + Real.sqrt b := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1279_127927


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l1279_127976

theorem modulo_eleven_residue : (305 + 7 * 44 + 9 * 176 + 6 * 18) % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l1279_127976


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1279_127970

-- Problem 1
theorem problem_1 (x y : ℝ) : (x + y)^2 + x * (x - 2*y) = 2*x^2 + y^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x ≠ 2 ∧ x ≠ 0) : 
  (x^2 - 6*x + 9) / (x - 2) / (x + 2 - (3*x - 4) / (x - 2)) = (x - 3) / x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1279_127970


namespace NUMINAMATH_CALUDE_chocolate_distribution_l1279_127911

theorem chocolate_distribution (total_chocolates : ℕ) (total_children : ℕ) 
  (boys : ℕ) (girls : ℕ) (chocolates_per_boy : ℕ) :
  total_chocolates = 3000 →
  total_children = 120 →
  boys = 60 →
  girls = 60 →
  chocolates_per_boy = 2 →
  (total_chocolates - boys * chocolates_per_boy) / girls = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l1279_127911


namespace NUMINAMATH_CALUDE_exists_real_geq_3_is_particular_l1279_127950

-- Define what a particular proposition is
def is_particular_proposition (p : Prop) : Prop :=
  ∃ (x : Type), p = ∃ (y : x), true

-- State the theorem
theorem exists_real_geq_3_is_particular : 
  is_particular_proposition (∃ (x : ℝ), x ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_exists_real_geq_3_is_particular_l1279_127950


namespace NUMINAMATH_CALUDE_inequality_and_system_solution_l1279_127960

theorem inequality_and_system_solution :
  (∀ x : ℝ, 2 * (-3 + x) > 3 * (x + 2) ↔ x < -12) ∧
  (∀ x : ℝ, (1/2 * (x + 1) < 2 ∧ (x + 2)/2 ≥ (x + 3)/3) ↔ 0 ≤ x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_system_solution_l1279_127960


namespace NUMINAMATH_CALUDE_henry_trays_problem_l1279_127998

theorem henry_trays_problem (trays_per_trip : ℕ) (trips : ℕ) (trays_second_table : ℕ) :
  trays_per_trip = 9 →
  trips = 9 →
  trays_second_table = 52 →
  trays_per_trip * trips - trays_second_table = 29 :=
by sorry

end NUMINAMATH_CALUDE_henry_trays_problem_l1279_127998


namespace NUMINAMATH_CALUDE_segment_distinctness_l1279_127916

theorem segment_distinctness (n : ℕ) (h : n ≥ 4) :
  ¬ ∀ (points : Fin (n + 1) → ℕ),
    (points 0 = 0 ∧ points (Fin.last n) = (n^2 + n) / 2) →
    (∀ i j : Fin (n + 1), i < j → points i < points j) →
    (∀ i j k l : Fin (n + 1), i < j ∧ k < l → 
      (points j - points i ≠ points l - points k ∨ (i = k ∧ j = l))) :=
by sorry

end NUMINAMATH_CALUDE_segment_distinctness_l1279_127916


namespace NUMINAMATH_CALUDE_four_fold_f_application_l1279_127904

-- Define the function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z ^ 2 else -(z ^ 2)

-- State the theorem
theorem four_fold_f_application :
  f (f (f (f (2 + 2*I)))) = -16777216 := by
  sorry

end NUMINAMATH_CALUDE_four_fold_f_application_l1279_127904


namespace NUMINAMATH_CALUDE_franks_breakfast_shopping_l1279_127931

/-- The cost of a bottle of milk in Frank's breakfast shopping -/
def milk_cost : ℝ := 2.5

/-- The cost of 10 buns -/
def buns_cost : ℝ := 1

/-- The number of bottles of milk Frank bought -/
def milk_bottles : ℕ := 1

/-- The cost of the carton of eggs -/
def eggs_cost : ℝ := 3 * milk_cost

/-- The total cost of Frank's breakfast shopping -/
def total_cost : ℝ := 11

theorem franks_breakfast_shopping :
  buns_cost + milk_bottles * milk_cost + eggs_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_franks_breakfast_shopping_l1279_127931


namespace NUMINAMATH_CALUDE_bottle_production_l1279_127949

/-- Given that 6 identical machines produce 240 bottles per minute at a constant rate,
    prove that 10 such machines will produce 1600 bottles in 4 minutes. -/
theorem bottle_production
  (machines : ℕ)
  (bottles_per_minute : ℕ)
  (h1 : machines = 6)
  (h2 : bottles_per_minute = 240)
  (constant_rate : ℕ → ℕ → ℕ) -- Function to calculate production based on number of machines and time
  (h3 : constant_rate machines 1 = bottles_per_minute) -- Production rate for given machines in 1 minute
  : constant_rate 10 4 = 1600 := by
  sorry


end NUMINAMATH_CALUDE_bottle_production_l1279_127949


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1279_127965

/-- A quadratic function with a negative leading coefficient -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The axis of symmetry of the quadratic function -/
def axis_of_symmetry (b c : ℝ) : ℝ := 2

theorem quadratic_inequality (b c : ℝ) :
  f b c (axis_of_symmetry b c + 2) < f b c (axis_of_symmetry b c - 1) ∧
  f b c (axis_of_symmetry b c - 1) < f b c (axis_of_symmetry b c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1279_127965


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l1279_127997

def A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -4; -3, 2]

theorem matrix_inverse_proof :
  ∃ (B : Matrix (Fin 2) (Fin 2) ℝ),
    B = !![1, 2; 1.5, 3.5] ∧ A * B = 1 ∧ B * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l1279_127997


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_68_l1279_127940

theorem smallest_n_divisible_by_68 :
  ∃ (n : ℕ), n^2 + 14*n + 13 ≡ 0 [MOD 68] ∧
  (∀ (m : ℕ), m < n → ¬(m^2 + 14*m + 13 ≡ 0 [MOD 68])) ∧
  n = 21 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_68_l1279_127940


namespace NUMINAMATH_CALUDE_pear_sales_l1279_127917

/-- Given a salesman who sold pears, prove that if he sold twice as much in the afternoon
    than in the morning, and 480 kilograms in total, then he sold 320 kilograms in the afternoon. -/
theorem pear_sales (morning_sales afternoon_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  morning_sales + afternoon_sales = 480 →
  afternoon_sales = 320 := by
  sorry

#check pear_sales

end NUMINAMATH_CALUDE_pear_sales_l1279_127917


namespace NUMINAMATH_CALUDE_min_value_expression_l1279_127963

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 2) :
  ∃ (m : ℝ), m = 25/2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 2 → (1 + 4*x + 3*y) / (x*y) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1279_127963


namespace NUMINAMATH_CALUDE_equation_solutions_l1279_127978

def equation (x : ℝ) : Prop :=
  x ≥ 1 ∧ Real.sqrt (x + 5 - 6 * Real.sqrt (x - 1)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 1)) = 3

theorem equation_solutions :
  {x : ℝ | equation x} = {5, 26} :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1279_127978


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l1279_127991

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_minimum_value 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hmin : ∀ x ∈ Set.Icc (-b / (2 * a)) ((2 * a - b) / (2 * a)), 
    quadratic_function a b c x ≠ (4 * a * c - b^2) / (4 * a)) :
  ∃ x ∈ Set.Icc (-b / (2 * a)) ((2 * a - b) / (2 * a)), 
    quadratic_function a b c x = (4 * a^2 + 4 * a * c - b^2) / (4 * a) ∧
    ∀ y ∈ Set.Icc (-b / (2 * a)) ((2 * a - b) / (2 * a)), 
      quadratic_function a b c y ≥ (4 * a^2 + 4 * a * c - b^2) / (4 * a) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l1279_127991


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1279_127981

-- Define the plane and lines
variable (α : Set (Real × Real × Real))
variable (m n : Set (Real × Real × Real))

-- Define the perpendicular and parallel relations
def perpendicular (l : Set (Real × Real × Real)) (p : Set (Real × Real × Real)) : Prop := sorry
def parallel (l1 l2 : Set (Real × Real × Real)) : Prop := sorry

-- State the theorem
theorem perpendicular_lines_parallel (h1 : perpendicular m α) (h2 : perpendicular n α) :
  parallel m n := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1279_127981


namespace NUMINAMATH_CALUDE_hidden_lattice_points_l1279_127956

theorem hidden_lattice_points (n : ℕ+) : 
  ∃ a b : ℤ, ∀ i j : ℕ, i < n ∧ j < n → Nat.gcd (Int.toNat (a + i)) (Int.toNat (b + j)) > 1 := by
  sorry

end NUMINAMATH_CALUDE_hidden_lattice_points_l1279_127956


namespace NUMINAMATH_CALUDE_sum_18_probability_l1279_127989

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The target sum we're aiming for -/
def target_sum : ℕ := 18

/-- The probability of rolling a sum of 18 with four standard 6-faced dice -/
def probability_sum_18 : ℚ := 5 / 216

/-- Theorem stating that the probability of rolling a sum of 18 with four standard 6-faced dice is 5/216 -/
theorem sum_18_probability : 
  probability_sum_18 = (num_favorable_outcomes : ℚ) / (num_faces ^ num_dice : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_sum_18_probability_l1279_127989


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1279_127995

theorem complex_equation_solution (a : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (1 + a * i) * i = -3 + i) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1279_127995


namespace NUMINAMATH_CALUDE_salary_distribution_l1279_127913

def salary : ℚ := 140000

def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5
def remaining : ℚ := 14000

def food_fraction : ℚ := 1/5

theorem salary_distribution :
  food_fraction * salary + rent_fraction * salary + clothes_fraction * salary + remaining = salary :=
by sorry

end NUMINAMATH_CALUDE_salary_distribution_l1279_127913


namespace NUMINAMATH_CALUDE_stamp_difference_l1279_127915

theorem stamp_difference (k a : ℕ) (h1 : k * 3 = a * 5) 
  (h2 : (k - 12) * 6 = (a + 12) * 8) : k - 12 = (a + 12) + 32 := by
  sorry

end NUMINAMATH_CALUDE_stamp_difference_l1279_127915


namespace NUMINAMATH_CALUDE_percentage_increase_l1279_127942

theorem percentage_increase (x : ℝ) (h : x = 105.6) :
  (x - 88) / 88 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1279_127942


namespace NUMINAMATH_CALUDE_factory_sampling_is_systematic_l1279_127979

/-- Represents a sampling method --/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a factory's production and inspection process --/
structure ProductionProcess where
  inspectionInterval : ℕ  -- Interval between inspections in minutes
  samplePosition : ℕ      -- Fixed position on the conveyor belt for sampling

/-- Determines the sampling method based on the production process --/
def determineSamplingMethod (process : ProductionProcess) : SamplingMethod :=
  if process.inspectionInterval > 0 ∧ process.samplePosition > 0 then
    SamplingMethod.Systematic
  else
    SamplingMethod.Other

/-- Theorem stating that the described process is systematic sampling --/
theorem factory_sampling_is_systematic (process : ProductionProcess) 
  (h1 : process.inspectionInterval = 10)
  (h2 : process.samplePosition > 0) :
  determineSamplingMethod process = SamplingMethod.Systematic := by
  sorry


end NUMINAMATH_CALUDE_factory_sampling_is_systematic_l1279_127979


namespace NUMINAMATH_CALUDE_fourth_quadrant_simplification_l1279_127967

/-- A point in the fourth quadrant of the Cartesian coordinate system -/
structure FourthQuadrantPoint where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_neg : b < 0

/-- Simplification of √(b²) + |b-a| for a point in the fourth quadrant -/
theorem fourth_quadrant_simplification (p : FourthQuadrantPoint) :
  Real.sqrt (p.b ^ 2) + abs (p.b - p.a) = p.a - 2 * p.b := by
  sorry


end NUMINAMATH_CALUDE_fourth_quadrant_simplification_l1279_127967


namespace NUMINAMATH_CALUDE_sine_cosine_roots_l1279_127955

theorem sine_cosine_roots (α : Real) (m : Real) : 
  α ∈ Set.Ioo 0 (2 * Real.pi) →
  (∃ (x y : Real), x = Real.sin α ∧ y = Real.cos α ∧
    2 * x^2 - (Real.sqrt 3 + 1) * x + m / 3 = 0 ∧
    2 * y^2 - (Real.sqrt 3 + 1) * y + m / 3 = 0) →
  m = 3 * Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_roots_l1279_127955


namespace NUMINAMATH_CALUDE_opposite_of_negative_nine_l1279_127933

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_nine : opposite (-9) = 9 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_nine_l1279_127933


namespace NUMINAMATH_CALUDE_bus_speed_l1279_127944

/-- Calculates the speed of a bus in kilometers per hour (kmph) given distance and time -/
theorem bus_speed (distance : Real) (time : Real) (conversion_factor : Real) : 
  distance = 900.072 ∧ time = 30 ∧ conversion_factor = 3.6 →
  (distance / time) * conversion_factor = 108.00864 := by
  sorry

#check bus_speed

end NUMINAMATH_CALUDE_bus_speed_l1279_127944


namespace NUMINAMATH_CALUDE_number_divided_by_constant_l1279_127946

theorem number_divided_by_constant (x : ℝ) : x / 0.06 = 16.666666666666668 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_constant_l1279_127946


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_5_pow_1000_without_zero_l1279_127930

theorem exists_number_divisible_by_5_pow_1000_without_zero : ∃ n : ℕ, 
  (5^1000 ∣ n) ∧ 
  (∀ d : ℕ, d < 10 → (n.digits 10).all (λ digit => digit ≠ d) → d ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_5_pow_1000_without_zero_l1279_127930


namespace NUMINAMATH_CALUDE_amount_after_two_years_l1279_127936

theorem amount_after_two_years (initial_amount : ℝ) : 
  initial_amount = 6400 →
  (initial_amount * (81 / 64) : ℝ) = 8100 := by
sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l1279_127936


namespace NUMINAMATH_CALUDE_base9_multiplication_l1279_127932

/-- Represents a number in base 9 --/
def Base9 : Type := ℕ

/-- Converts a base 9 number to a natural number --/
def to_nat (x : Base9) : ℕ := sorry

/-- Converts a natural number to a base 9 number --/
def from_nat (n : ℕ) : Base9 := sorry

/-- Multiplication operation for Base9 numbers --/
def mul_base9 (x y : Base9) : Base9 := sorry

theorem base9_multiplication :
  mul_base9 (from_nat 362) (from_nat 7) = from_nat 2875 :=
sorry

end NUMINAMATH_CALUDE_base9_multiplication_l1279_127932


namespace NUMINAMATH_CALUDE_expansion_coefficients_l1279_127957

theorem expansion_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) 
  (h : (2*(x-1)-1)^9 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                       a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9) : 
  a₂ = -144 ∧ a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 2 := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l1279_127957


namespace NUMINAMATH_CALUDE_fraction_addition_theorem_l1279_127918

theorem fraction_addition_theorem (a b c d x : ℚ) 
  (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : c ≠ d) 
  (h4 : (a + x) / (b + x) = c / d) : 
  x = (a * d - b * c) / (c - d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_theorem_l1279_127918


namespace NUMINAMATH_CALUDE_differential_equation_solution_l1279_127935

/-- The general solution to the differential equation dr - r dφ = 0 -/
theorem differential_equation_solution (r φ : ℝ → ℝ) (C : ℝ) :
  (∀ t, (deriv r t) - r t * (deriv φ t) = 0) ↔
  ∃ C, C > 0 ∧ ∀ t, r t = C * Real.exp (φ t) :=
sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l1279_127935


namespace NUMINAMATH_CALUDE_illumination_theorem_l1279_127962

/-- Represents a direction: North, South, East, or West -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a spotlight with a position and direction -/
structure Spotlight where
  position : Point
  direction : Direction

/-- Represents the configuration of 4 spotlights -/
def SpotlightConfiguration := Fin 4 → Spotlight

/-- Checks if a point is illuminated by a spotlight -/
def isIlluminated (p : Point) (s : Spotlight) : Prop :=
  match s.direction with
  | Direction.North => p.y ≥ s.position.y
  | Direction.South => p.y ≤ s.position.y
  | Direction.East => p.x ≥ s.position.x
  | Direction.West => p.x ≤ s.position.x

/-- The main theorem: there exists a configuration that illuminates the entire plane -/
theorem illumination_theorem (p1 p2 p3 p4 : Point) :
  ∃ (config : SpotlightConfiguration),
    ∀ (p : Point), ∃ (i : Fin 4), isIlluminated p (config i) := by
  sorry


end NUMINAMATH_CALUDE_illumination_theorem_l1279_127962


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l1279_127907

theorem rectangle_formation_count :
  let horizontal_lines := 5
  let vertical_lines := 4
  let horizontal_choices := Nat.choose horizontal_lines 2
  let vertical_choices := Nat.choose vertical_lines 2
  horizontal_choices * vertical_choices = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l1279_127907


namespace NUMINAMATH_CALUDE_max_value_theorem_equality_conditions_l1279_127969

theorem max_value_theorem (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a * b * c * d) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) ≤ 1 :=
sorry

theorem equality_conditions (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a * b * c * d) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) = 1 ↔ 
  ((a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1)) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_equality_conditions_l1279_127969


namespace NUMINAMATH_CALUDE_equation_solution_l1279_127986

theorem equation_solution :
  ∃ x : ℝ, (((3 * x - 1) / (x + 4)) > 0) ∧ 
            (((x + 4) / (3 * x - 1)) > 0) ∧
            (Real.sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3 * x - 1)) = 0) ∧
            (x = 5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1279_127986


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l1279_127999

/-- Given a set of observations with known properties, calculate the incorrect value --/
theorem incorrect_observation_value
  (n : ℕ)  -- Total number of observations
  (original_mean : ℝ)  -- Original mean of observations
  (correct_value : ℝ)  -- The correct value of the misrecorded observation
  (new_mean : ℝ)  -- New mean after correction
  (hn : n = 40)  -- There are 40 observations
  (hom : original_mean = 36)  -- The original mean was 36
  (hcv : correct_value = 34)  -- The correct value of the misrecorded observation is 34
  (hnm : new_mean = 36.45)  -- The new mean after correction is 36.45
  : ∃ (incorrect_value : ℝ), incorrect_value = 52 := by
  sorry


end NUMINAMATH_CALUDE_incorrect_observation_value_l1279_127999


namespace NUMINAMATH_CALUDE_inequality_proof_l1279_127909

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (b / (a - b) > c / (a - c)) ∧
  (a / (a + b) < (a + c) / (a + b + c)) ∧
  (1 / (a - b) + 1 / (b - c) ≥ 4 / (a - c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1279_127909


namespace NUMINAMATH_CALUDE_range_of_f_l1279_127966

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem range_of_f :
  ∃ (y_min y_max : ℝ), y_min = -1 ∧ y_max = 3 ∧
  (∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc y_min y_max) ∧
  (∀ y ∈ Set.Icc y_min y_max, ∃ x ∈ Set.Icc 0 3, f x = y) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1279_127966


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l1279_127919

/-- The volume of a sphere that circumscribes a rectangular solid with dimensions 3, 2, and 1 -/
theorem sphere_volume_circumscribing_rectangular_solid :
  let length : ℝ := 3
  let width : ℝ := 2
  let height : ℝ := 1
  let radius : ℝ := Real.sqrt (length^2 + width^2 + height^2) / 2
  let volume : ℝ := (4 / 3) * Real.pi * radius^3
  volume = (7 * Real.sqrt 14 * Real.pi) / 3 := by
  sorry

#check sphere_volume_circumscribing_rectangular_solid

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l1279_127919


namespace NUMINAMATH_CALUDE_xy_2yz_3zx_value_l1279_127902

theorem xy_2yz_3zx_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2/3 = 25)
  (eq2 : y^2/3 + z^2 = 9)
  (eq3 : z^2 + z*x + x^2 = 16) :
  x*y + 2*y*z + 3*z*x = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_xy_2yz_3zx_value_l1279_127902


namespace NUMINAMATH_CALUDE_garden_area_l1279_127914

/-- Calculates the area of a garden given property dimensions and garden proportions -/
theorem garden_area 
  (property_width : ℝ) 
  (property_length : ℝ) 
  (garden_width_ratio : ℝ) 
  (garden_length_ratio : ℝ) 
  (h1 : property_width = 1000) 
  (h2 : property_length = 2250) 
  (h3 : garden_width_ratio = 1 / 8) 
  (h4 : garden_length_ratio = 1 / 10) : 
  garden_width_ratio * property_width * garden_length_ratio * property_length = 28125 := by
  sorry

#check garden_area

end NUMINAMATH_CALUDE_garden_area_l1279_127914


namespace NUMINAMATH_CALUDE_average_of_subset_l1279_127980

theorem average_of_subset (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 63 →
  (list.get! 2 + list.get! 3) / 2 = 60 →
  (list.get! 0 + list.get! 1 + list.get! 4 + list.get! 5 + list.get! 6) / 5 = 64.2 := by
sorry

end NUMINAMATH_CALUDE_average_of_subset_l1279_127980


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1279_127912

theorem quadratic_inequality (x : ℝ) : x^2 - 10*x + 21 < 0 ↔ 3 < x ∧ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1279_127912


namespace NUMINAMATH_CALUDE_complex_product_polar_form_l1279_127983

-- Define the cis function
noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the problem statement
theorem complex_product_polar_form :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  (4 * cis (25 * π / 180)) * (-3 * cis (48 * π / 180)) = r * cis θ ∧
  r = 12 ∧ θ = 253 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_polar_form_l1279_127983


namespace NUMINAMATH_CALUDE_inequality_solution_set_not_equal_function_always_negative_implies_k_range_negation_of_inequality_solution_set_is_true_l1279_127943

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) * Real.sqrt (x^2 - 3*x + 2) ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ 2}

-- Define the function y
def y (k x : ℝ) : ℝ := k * x^2 - k * x - 1

theorem inequality_solution_set_not_equal : 
  {x : ℝ | inequality x} ≠ solution_set := by sorry

theorem function_always_negative_implies_k_range (k : ℝ) :
  (∀ x, y k x < 0) → -4 < k ∧ k ≤ 0 := by sorry

theorem negation_of_inequality_solution_set_is_true :
  ¬({x : ℝ | inequality x} = solution_set) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_not_equal_function_always_negative_implies_k_range_negation_of_inequality_solution_set_is_true_l1279_127943


namespace NUMINAMATH_CALUDE_smaller_factor_of_4582_l1279_127934

theorem smaller_factor_of_4582 :
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    a * b = 4582 ∧
    (∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 4582 → min x y = 21) :=
by sorry

end NUMINAMATH_CALUDE_smaller_factor_of_4582_l1279_127934


namespace NUMINAMATH_CALUDE_gcf_and_multiples_of_90_and_135_l1279_127988

theorem gcf_and_multiples_of_90_and_135 :
  ∃ (gcf : ℕ), 
    (Nat.gcd 90 135 = gcf) ∧ 
    (gcf = 45) ∧
    (45 ∣ gcf) ∧ 
    (90 ∣ gcf) ∧ 
    (135 ∣ gcf) := by
  sorry

end NUMINAMATH_CALUDE_gcf_and_multiples_of_90_and_135_l1279_127988


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1279_127974

/-- The speed of a boat in still water, given its speed with and against the stream -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ)
  (h1 : along_stream = 9)
  (h2 : against_stream = 5) :
  (along_stream + against_stream) / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1279_127974


namespace NUMINAMATH_CALUDE_democrat_count_l1279_127947

theorem democrat_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 870 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 145 := by
  sorry

end NUMINAMATH_CALUDE_democrat_count_l1279_127947


namespace NUMINAMATH_CALUDE_dog_owners_count_l1279_127958

-- Define the sets of people owning each type of pet
def C : Finset ℕ := sorry
def D : Finset ℕ := sorry
def R : Finset ℕ := sorry

-- Define the theorem
theorem dog_owners_count :
  (C ∪ D ∪ R).card = 60 ∧
  C.card = 30 ∧
  R.card = 16 ∧
  ((C ∩ D) ∪ (C ∩ R) ∪ (D ∩ R)).card - (C ∩ D ∩ R).card = 12 ∧
  (C ∩ D ∩ R).card = 7 →
  D.card = 40 := by
sorry


end NUMINAMATH_CALUDE_dog_owners_count_l1279_127958


namespace NUMINAMATH_CALUDE_solve_system_l1279_127938

theorem solve_system (y z : ℝ) 
  (h1 : y^2 - 6*y + 9 = 0) 
  (h2 : y + z = 11) : 
  y = 3 ∧ z = 8 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1279_127938


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1279_127994

-- Define the length of the train in meters
def train_length : ℝ := 83.33333333333334

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 5

-- Define the speed of the train in km/hr
def train_speed : ℝ := 60

-- Theorem to prove
theorem train_speed_calculation :
  train_speed = (train_length / 1000) / (crossing_time / 3600) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1279_127994


namespace NUMINAMATH_CALUDE_square_ratio_problem_l1279_127977

theorem square_ratio_problem (area_ratio : ℚ) (a b c : ℕ) :
  area_ratio = 48 / 125 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  a = 4 ∧ b = 15 ∧ c = 25 ∧ a + b + c = 44 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_problem_l1279_127977


namespace NUMINAMATH_CALUDE_linda_babysitting_hours_l1279_127929

/-- Linda's babysitting problem -/
theorem linda_babysitting_hours (babysitting_rate : ℚ) (application_fee : ℚ) (num_colleges : ℕ) :
  babysitting_rate = 10 →
  application_fee = 25 →
  num_colleges = 6 →
  (num_colleges * application_fee) / babysitting_rate = 15 :=
by sorry

end NUMINAMATH_CALUDE_linda_babysitting_hours_l1279_127929


namespace NUMINAMATH_CALUDE_lunch_percentage_boys_l1279_127953

theorem lunch_percentage_boys (C B G : ℝ) (P_b : ℝ) :
  B / G = 3 / 2 →
  C = B + G →
  0.52 * C = (P_b / 100) * B + (40 / 100) * G →
  P_b = 60 := by
  sorry

end NUMINAMATH_CALUDE_lunch_percentage_boys_l1279_127953


namespace NUMINAMATH_CALUDE_contractor_problem_l1279_127928

/-- Calculates the original number of days to complete a job given the original number of laborers,
    the number of absent laborers, and the number of days taken by the remaining laborers. -/
def original_completion_time (total_laborers : ℕ) (absent_laborers : ℕ) (actual_days : ℕ) : ℕ :=
  (total_laborers - absent_laborers) * actual_days / total_laborers

theorem contractor_problem (total_laborers absent_laborers actual_days : ℕ) 
  (h1 : total_laborers = 7)
  (h2 : absent_laborers = 3)
  (h3 : actual_days = 14) :
  original_completion_time total_laborers absent_laborers actual_days = 8 := by
  sorry

#eval original_completion_time 7 3 14

end NUMINAMATH_CALUDE_contractor_problem_l1279_127928


namespace NUMINAMATH_CALUDE_no_rectangle_with_sum_76_l1279_127959

theorem no_rectangle_with_sum_76 : ¬∃ (w : ℕ), w > 0 ∧ 2 * w^2 + 6 * w = 76 := by
  sorry

end NUMINAMATH_CALUDE_no_rectangle_with_sum_76_l1279_127959


namespace NUMINAMATH_CALUDE_solve_equation_l1279_127968

theorem solve_equation (x : ℝ) (h : 3 * x - 2 = 13) : 6 * x + 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1279_127968


namespace NUMINAMATH_CALUDE_radian_to_degree_conversion_l1279_127951

theorem radian_to_degree_conversion (π : ℝ) (h : π * (180 / π) = 180) :
  (23 / 12) * π * (180 / π) = 345 :=
sorry

end NUMINAMATH_CALUDE_radian_to_degree_conversion_l1279_127951


namespace NUMINAMATH_CALUDE_adam_apples_proof_l1279_127973

def monday_apples : ℕ := 15
def tuesday_multiplier : ℕ := 3
def wednesday_multiplier : ℕ := 4

def total_apples : ℕ := 
  monday_apples + 
  (tuesday_multiplier * monday_apples) + 
  (wednesday_multiplier * tuesday_multiplier * monday_apples)

theorem adam_apples_proof : total_apples = 240 := by
  sorry

end NUMINAMATH_CALUDE_adam_apples_proof_l1279_127973


namespace NUMINAMATH_CALUDE_range_of_m_l1279_127922

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b)
  (h : ∀ (a b : ℝ), a > 0 → b > 0 → a * b = 2 * a + b → a + 2 * b ≥ m^2 - 8 * m) :
  -1 ≤ m ∧ m ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1279_127922


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1279_127945

theorem cubic_root_sum (a b c : ℕ+) :
  let x : ℝ := (Real.rpow a (1/3 : ℝ) + Real.rpow b (1/3 : ℝ) + 2) / c
  27 * x^3 - 6 * x^2 - 6 * x - 2 = 0 →
  a + b + c = 75 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1279_127945


namespace NUMINAMATH_CALUDE_sequence_general_term_l1279_127975

/-- Given a sequence {aₙ} where a₁ = 1 and aₙ₊₁ - aₙ = 2ⁿ for all n ≥ 1,
    prove that the general term is given by aₙ = 2ⁿ - 1 -/
theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2^n) : 
    ∀ n : ℕ, n ≥ 1 → a n = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1279_127975


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l1279_127939

theorem square_perimeter_relation (perimeter_C : ℝ) (area_ratio : ℝ) : 
  perimeter_C = 40 →
  area_ratio = 1/3 →
  let side_C := perimeter_C / 4
  let area_C := side_C ^ 2
  let area_D := area_ratio * area_C
  let side_D := Real.sqrt area_D
  let perimeter_D := 4 * side_D
  perimeter_D = (40 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l1279_127939


namespace NUMINAMATH_CALUDE_vincent_animal_books_l1279_127924

/-- The number of books about animals Vincent bought -/
def num_animal_books : ℕ := sorry

/-- The cost of each book -/
def book_cost : ℕ := 16

/-- The total number of books about outer space and trains -/
def num_other_books : ℕ := 1 + 3

/-- The total amount Vincent spent on books -/
def total_spent : ℕ := 224

theorem vincent_animal_books : 
  num_animal_books = 10 := by sorry

end NUMINAMATH_CALUDE_vincent_animal_books_l1279_127924


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1279_127948

/-- Given a hyperbola with the specified properties, prove its equation -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 5 / 4
  let c := 5
  (c / a = e) →
  (c^2 = a^2 + b^2) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 16 - y^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1279_127948


namespace NUMINAMATH_CALUDE_infinite_solutions_cube_equation_l1279_127954

theorem infinite_solutions_cube_equation :
  ∃ f : ℕ → ℕ × ℕ × ℕ × ℕ, ∀ m : ℕ,
    let (n, x, y, z) := f m
    n > m ∧ (x + y + z)^3 = n^2 * x * y * z :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_cube_equation_l1279_127954


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l1279_127903

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x + 4}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the interval [4, +∞)
def interval_four_to_inf : Set ℝ := {x | x ≥ 4}

-- Theorem statement
theorem intersection_equals_interval : A_intersect_B = interval_four_to_inf := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l1279_127903


namespace NUMINAMATH_CALUDE_parabola_equation_l1279_127992

/-- Represents a parabola with focus (5,5) and directrix 4x + 9y = 36 -/
structure Parabola where
  focus : ℝ × ℝ := (5, 5)
  directrix : ℝ → ℝ → ℝ := fun x y => 4*x + 9*y - 36

/-- Represents the equation of a conic in general form -/
structure ConicEquation where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

def ConicEquation.isValid (eq : ConicEquation) : Prop :=
  eq.a > 0 ∧ Int.gcd eq.a.natAbs (Int.gcd eq.b.natAbs (Int.gcd eq.c.natAbs (Int.gcd eq.d.natAbs (Int.gcd eq.e.natAbs eq.f.natAbs)))) = 1

/-- The equation of the parabola matches the given conic equation -/
def equationMatches (p : Parabola) (eq : ConicEquation) : Prop :=
  ∀ x y : ℝ, eq.a * x^2 + eq.b * x * y + eq.c * y^2 + eq.d * x + eq.e * y + eq.f = 0 ↔
    (x - p.focus.1)^2 + (y - p.focus.2)^2 = ((4*x + 9*y - 36) / Real.sqrt 97)^2

theorem parabola_equation (p : Parabola) :
  ∃ eq : ConicEquation, eq.isValid ∧ equationMatches p eq ∧
    eq.a = 81 ∧ eq.b = -60 ∧ eq.c = 273 ∧ eq.d = -2162 ∧ eq.e = -5913 ∧ eq.f = 19407 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1279_127992


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1279_127921

/-- For a rectangle with sum of length and width equal to 28 meters, the perimeter is 56 meters. -/
theorem rectangle_perimeter (l w : ℝ) (h : l + w = 28) : 2 * (l + w) = 56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1279_127921


namespace NUMINAMATH_CALUDE_consecutive_seating_theorem_l1279_127926

/-- The number of people at the table -/
def total_people : ℕ := 12

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 4

/-- The number of biology majors -/
def biology_majors : ℕ := 3

/-- The probability of math and physics majors sitting in consecutive seats -/
def consecutive_seating_probability : ℚ := 7/240

theorem consecutive_seating_theorem :
  let total := total_people
  let math := math_majors
  let physics := physics_majors
  let bio := biology_majors
  total = math + physics + bio →
  consecutive_seating_probability = 7/240 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_seating_theorem_l1279_127926


namespace NUMINAMATH_CALUDE_product_of_differences_l1279_127971

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2006) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2007)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2006) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2007)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2006) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2007) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/2006 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l1279_127971


namespace NUMINAMATH_CALUDE_lesser_solution_quadratic_l1279_127993

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 10*x - 75 = 0 → (∃ y : ℝ, y^2 + 10*y - 75 = 0 ∧ y ≤ x) → x = -15 :=
sorry

end NUMINAMATH_CALUDE_lesser_solution_quadratic_l1279_127993


namespace NUMINAMATH_CALUDE_eric_egg_collection_days_l1279_127972

/-- Proves that Eric waited 3 days to collect 36 eggs from 4 chickens laying 3 eggs each per day -/
theorem eric_egg_collection_days (num_chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (total_eggs : ℕ) : 
  num_chickens = 4 → 
  eggs_per_chicken_per_day = 3 → 
  total_eggs = 36 → 
  (total_eggs / (num_chickens * eggs_per_chicken_per_day) : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_eric_egg_collection_days_l1279_127972


namespace NUMINAMATH_CALUDE_policeman_speed_l1279_127964

/-- Proves that given the initial conditions of a chase between a policeman and a thief,
    the policeman's speed is 64 km/hr. -/
theorem policeman_speed (initial_distance : ℝ) (thief_speed : ℝ) (thief_distance : ℝ) :
  initial_distance = 160 →
  thief_speed = 8 →
  thief_distance = 640 →
  ∃ (policeman_speed : ℝ), policeman_speed = 64 :=
by
  sorry


end NUMINAMATH_CALUDE_policeman_speed_l1279_127964


namespace NUMINAMATH_CALUDE_cookies_in_fridge_l1279_127906

/-- The number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 1024

/-- The number of cookies given to Tim -/
def tim_cookies : ℕ := 48

/-- The number of cookies given to Mike -/
def mike_cookies : ℕ := 58

/-- The number of cookies given to Sarah -/
def sarah_cookies : ℕ := 78

/-- The number of cookies given to Anna -/
def anna_cookies : ℕ := 2 * (tim_cookies + mike_cookies) - sarah_cookies / 2

/-- The number of cookies Uncle Jude put in the fridge -/
def fridge_cookies : ℕ := total_cookies - (tim_cookies + mike_cookies + sarah_cookies + anna_cookies)

theorem cookies_in_fridge : fridge_cookies = 667 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_fridge_l1279_127906


namespace NUMINAMATH_CALUDE_skin_cost_problem_l1279_127996

/-- Given two skins with a total value of 2250 rubles, sold with a total profit of 40%,
    where the profit from the first skin is 25% and the profit from the second skin is -50%,
    prove that the cost of the first skin is 2700 rubles and the cost of the second skin is 450 rubles. -/
theorem skin_cost_problem (x : ℝ) (h1 : x + (2250 - x) = 2250) 
  (h2 : 1.25 * x + 0.5 * (2250 - x) = 1.4 * 2250) : x = 2700 ∧ 2250 - x = 450 := by
  sorry

#check skin_cost_problem

end NUMINAMATH_CALUDE_skin_cost_problem_l1279_127996
