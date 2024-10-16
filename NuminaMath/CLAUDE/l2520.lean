import Mathlib

namespace NUMINAMATH_CALUDE_third_grade_sample_size_l2520_252028

/-- Calculates the number of students sampled from a specific grade in a stratified sampling. -/
def stratified_sample (total_students : ℕ) (grade_students : ℕ) (sample_size : ℕ) : ℕ :=
  (grade_students * sample_size) / total_students

/-- Theorem stating the result of stratified sampling for the third grade. -/
theorem third_grade_sample_size :
  let total_students : ℕ := 1000
  let third_grade_students : ℕ := 400
  let total_sample_size : ℕ := 40
  stratified_sample total_students third_grade_students total_sample_size = 16 := by
  sorry


end NUMINAMATH_CALUDE_third_grade_sample_size_l2520_252028


namespace NUMINAMATH_CALUDE_expression_value_l2520_252031

theorem expression_value : ∀ x : ℝ, x ≠ 5 →
  (x^2 - 3*x - 10) / (x - 5) = x + 2 ∧
  ((1^2 - 3*1 - 10) / (1 - 5) = 3) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2520_252031


namespace NUMINAMATH_CALUDE_base_ten_and_twelve_satisfy_conditions_l2520_252047

/-- Represents a number in a given base -/
def NumberInBase (n : ℕ) (base : ℕ) : ℕ := n

/-- Checks if a number is even in a given base -/
def IsEvenInBase (n : ℕ) (base : ℕ) : Prop :=
  ∃ k : ℕ, NumberInBase n base = 2 * k

/-- Checks if three numbers are consecutive in a given base -/
def AreConsecutiveInBase (a b c : ℕ) (base : ℕ) : Prop :=
  NumberInBase b base = NumberInBase a base + 1 ∧
  NumberInBase c base = NumberInBase b base + 1

/-- The main theorem to prove -/
theorem base_ten_and_twelve_satisfy_conditions :
  (NumberInBase 24 10 = NumberInBase 4 10 * NumberInBase 6 10 ∧
   ∃ a b c : ℕ, AreConsecutiveInBase a b c 10 ∧
   (IsEvenInBase a 10 ∧ IsEvenInBase b 10 ∧ IsEvenInBase c 10 ∨
    ¬IsEvenInBase a 10 ∧ ¬IsEvenInBase b 10 ∧ ¬IsEvenInBase c 10)) ∧
  (NumberInBase 24 12 = NumberInBase 4 12 * NumberInBase 6 12 ∧
   ∃ a b c : ℕ, AreConsecutiveInBase a b c 12 ∧
   (IsEvenInBase a 12 ∧ IsEvenInBase b 12 ∧ IsEvenInBase c 12 ∨
    ¬IsEvenInBase a 12 ∧ ¬IsEvenInBase b 12 ∧ ¬IsEvenInBase c 12)) :=
by sorry


end NUMINAMATH_CALUDE_base_ten_and_twelve_satisfy_conditions_l2520_252047


namespace NUMINAMATH_CALUDE_sebastians_age_l2520_252022

theorem sebastians_age (sebastian_age sister_age father_age : ℕ) : 
  (sebastian_age - 5) + (sister_age - 5) = 3 * (father_age - 5) / 4 →
  sebastian_age = sister_age + 10 →
  father_age = 85 →
  sebastian_age = 40 := by
sorry

end NUMINAMATH_CALUDE_sebastians_age_l2520_252022


namespace NUMINAMATH_CALUDE_solution_set_p_sufficient_condition_l2520_252008

-- Define the conditions
def p (x : ℝ) : Prop := -x^2 + 6*x + 16 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 4*x + 4 - m^2 ≤ 0

-- Theorem 1: The solution set of p is [-2, 8]
theorem solution_set_p : Set.Icc (-2 : ℝ) 8 = {x | p x} := by sorry

-- Theorem 2: If p is a sufficient but not necessary condition for q, then m ≥ 6
theorem sufficient_condition (h : ∀ x m, m > 0 → p x → q x m) :
  ∀ m, m > 0 → (∃ x, q x m ∧ ¬p x) → m ≥ 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_p_sufficient_condition_l2520_252008


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l2520_252005

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The shortest distance from a point on the hyperbola to one of its foci -/
def shortest_focal_distance (h : Hyperbola) : ℝ := 2

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on an asymptote of the hyperbola -/
def on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y / p.x = h.b / h.a ∨ p.y / p.x = -h.b / h.a

/-- The given point P -/
def P : Point := ⟨3, 4⟩

theorem hyperbola_parameters (h : Hyperbola) 
  (h_focal : shortest_focal_distance h = 2)
  (h_asymptote : on_asymptote h P) :
  h.a = 3 ∧ h.b = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l2520_252005


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2520_252060

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (half_dollar : CoinOutcome)

/-- The total number of possible outcomes when flipping 5 coins -/
def total_outcomes : ℕ := 32

/-- Predicate for the desired outcome (penny, nickel, and dime are heads) -/
def desired_outcome (cs : CoinSet) : Prop :=
  cs.penny = CoinOutcome.Heads ∧ cs.nickel = CoinOutcome.Heads ∧ cs.dime = CoinOutcome.Heads

/-- The number of outcomes satisfying the desired condition -/
def successful_outcomes : ℕ := 4

/-- The probability of the desired outcome -/
def probability : ℚ := 1 / 8

theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = probability :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2520_252060


namespace NUMINAMATH_CALUDE_sphere_volume_rectangular_solid_l2520_252083

theorem sphere_volume_rectangular_solid (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b = 1 →
  b * c = 2 →
  a * c = 2 →
  (4 / 3) * Real.pi * ((a^2 + b^2 + c^2).sqrt / 2)^3 = Real.pi * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_rectangular_solid_l2520_252083


namespace NUMINAMATH_CALUDE_not_all_regular_pentagons_congruent_l2520_252077

-- Define a regular pentagon
structure RegularPentagon where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

-- Define congruence for regular pentagons
def congruent (p1 p2 : RegularPentagon) : Prop :=
  p1.sideLength = p2.sideLength

-- Theorem statement
theorem not_all_regular_pentagons_congruent :
  ∃ (p1 p2 : RegularPentagon), ¬(congruent p1 p2) := by
  sorry

end NUMINAMATH_CALUDE_not_all_regular_pentagons_congruent_l2520_252077


namespace NUMINAMATH_CALUDE_max_player_salary_max_salary_is_512000_l2520_252003

/-- The maximum possible salary for a single player in a minor league soccer team -/
theorem max_player_salary (n : ℕ) (min_salary : ℕ) (max_total : ℕ) : ℕ :=
  let max_single_salary := max_total - (n - 1) * min_salary
  max_single_salary

/-- The maximum possible salary for a single player in the given scenario is $512,000 -/
theorem max_salary_is_512000 :
  max_player_salary 25 12000 800000 = 512000 := by
  sorry

end NUMINAMATH_CALUDE_max_player_salary_max_salary_is_512000_l2520_252003


namespace NUMINAMATH_CALUDE_largest_negative_integer_congruence_l2520_252072

theorem largest_negative_integer_congruence :
  ∃ (x : ℤ), x = -2 ∧ 
  (∀ (y : ℤ), y < 0 → 50 * y + 14 ≡ 10 [ZMOD 24] → y ≤ x) ∧
  50 * x + 14 ≡ 10 [ZMOD 24] := by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_congruence_l2520_252072


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2520_252063

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (-1, x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  parallel vector_a (vector_b x) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2520_252063


namespace NUMINAMATH_CALUDE_square_root_equation_l2520_252038

theorem square_root_equation (x : ℝ) : Real.sqrt (x - 5) = 7 → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l2520_252038


namespace NUMINAMATH_CALUDE_first_stop_passengers_l2520_252039

/-- The number of passengers who got on at the first stop of a bus route -/
def passengers_first_stop : ℕ :=
  sorry

/-- The net change in passengers at the second stop -/
def net_change_second_stop : ℤ := 2

/-- The net change in passengers at the third stop -/
def net_change_third_stop : ℤ := 2

/-- The total number of passengers after the third stop -/
def total_passengers : ℕ := 11

theorem first_stop_passengers :
  passengers_first_stop = 7 :=
sorry

end NUMINAMATH_CALUDE_first_stop_passengers_l2520_252039


namespace NUMINAMATH_CALUDE_fraction_simplification_l2520_252096

theorem fraction_simplification (m : ℝ) (h : m ≠ 3) :
  m^2 / (m - 3) + 9 / (3 - m) = m + 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2520_252096


namespace NUMINAMATH_CALUDE_extreme_values_and_three_roots_l2520_252017

/-- The function f(x) = x³ + ax² + bx + c -/
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_three_roots 
  (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, ∃ y₁ y₂ y₃, f a b c y₁ = 2*c ∧ f a b c y₂ = 2*c ∧ f a b c y₃ = 2*c ∧ y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₂ ≠ y₃) →
  (f' a b 1 = 0 ∧ f' a b (-2/3) = 0) →
  (a = -1/2 ∧ b = -2 ∧ 1/2 ≤ c ∧ c < 22/27) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_three_roots_l2520_252017


namespace NUMINAMATH_CALUDE_negative_a_sixth_divided_by_a_third_l2520_252095

theorem negative_a_sixth_divided_by_a_third (a : ℝ) : (-a)^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_sixth_divided_by_a_third_l2520_252095


namespace NUMINAMATH_CALUDE_gcd_1215_1995_l2520_252086

theorem gcd_1215_1995 : Nat.gcd 1215 1995 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1215_1995_l2520_252086


namespace NUMINAMATH_CALUDE_rainfall_ratio_l2520_252062

/-- Given the total rainfall over two weeks and the rainfall in the second week,
    prove the ratio of rainfall in the second week to the first week. -/
theorem rainfall_ratio (total : ℝ) (second_week : ℝ) 
    (h1 : total = 35)
    (h2 : second_week = 21) :
    second_week / (total - second_week) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_ratio_l2520_252062


namespace NUMINAMATH_CALUDE_polynomial_value_relation_l2520_252059

theorem polynomial_value_relation (m n : ℝ) : 
  -m^2 + 3*n = 2 → m^2 - 3*n - 1 = -3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_relation_l2520_252059


namespace NUMINAMATH_CALUDE_linear_equation_exponent_l2520_252076

theorem linear_equation_exponent (n : ℕ) : 
  (∀ x, ∃ a b, x^(2*n - 5) - 2 = a*x + b) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_l2520_252076


namespace NUMINAMATH_CALUDE_coin_trick_theorem_l2520_252004

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a row of 27 coins -/
def CoinRow := Vector CoinState 27

/-- Represents a selection of 5 coins from the row -/
def CoinSelection := Vector Nat 5

/-- Function to check if all coins in a selection are facing the same way -/
def allSameFacing (row : CoinRow) (selection : CoinSelection) : Prop :=
  ∀ i j, i < 5 → j < 5 → row.get (selection.get i) = row.get (selection.get j)

/-- The main theorem stating that it's always possible to select 10 coins facing the same way,
    such that 5 of them can determine the state of the other 5 -/
theorem coin_trick_theorem (row : CoinRow) :
  ∃ (selection1 selection2 : CoinSelection),
    allSameFacing row selection1 ∧
    allSameFacing row selection2 ∧
    (∀ i, i < 5 → selection1.get i ≠ selection2.get i) ∧
    (∃ f : CoinSelection → CoinSelection, f selection1 = selection2) :=
sorry

end NUMINAMATH_CALUDE_coin_trick_theorem_l2520_252004


namespace NUMINAMATH_CALUDE_triangle_collinearity_l2520_252058

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the orthocenter H
variable (H : ℝ × ℝ)

-- Define points M and N
variable (M N : ℝ × ℝ)

-- Define the circumcenter O of triangle HMN
variable (O : ℝ × ℝ)

-- Define point D
variable (D : ℝ × ℝ)

-- Define the conditions
def is_acute_triangle (A B C : ℝ × ℝ) : Prop := sorry

def angle_A_greater_than_60 (A B C : ℝ × ℝ) : Prop := sorry

def is_orthocenter (H A B C : ℝ × ℝ) : Prop := sorry

def on_side (M A B : ℝ × ℝ) : Prop := sorry

def angle_equals_60 (H M B : ℝ × ℝ) : Prop := sorry

def is_circumcenter (O H M N : ℝ × ℝ) : Prop := sorry

def forms_equilateral_triangle (D B C : ℝ × ℝ) : Prop := sorry

def same_side_as_A (D A B C : ℝ × ℝ) : Prop := sorry

def are_collinear (H O D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_collinearity 
  (h_acute : is_acute_triangle A B C)
  (h_angle_A : angle_A_greater_than_60 A B C)
  (h_orthocenter : is_orthocenter H A B C)
  (h_M_on_AB : on_side M A B)
  (h_N_on_AC : on_side N A C)
  (h_angle_HMB : angle_equals_60 H M B)
  (h_angle_HNC : angle_equals_60 H N C)
  (h_circumcenter : is_circumcenter O H M N)
  (h_equilateral : forms_equilateral_triangle D B C)
  (h_same_side : same_side_as_A D A B C) :
  are_collinear H O D :=
sorry

end NUMINAMATH_CALUDE_triangle_collinearity_l2520_252058


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l2520_252030

/-- The maximum distance from any point on the circle (x-1)² + (y-1)² = 2 to the line x + y - 4 = 0 is 2√2. -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 2}
  let line := {p : ℝ × ℝ | p.1 + p.2 - 4 = 0}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
    (∀ p ∈ circle, ∀ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ d) ∧
    (∃ p ∈ circle, ∃ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = d) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l2520_252030


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2520_252000

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first five terms of the sequence is 20. -/
def SumFirstFiveTerms (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 = 20

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arithmetic : IsArithmeticSequence a)
  (h_sum : SumFirstFiveTerms a) :
  a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2520_252000


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l2520_252041

theorem number_with_specific_remainders : ∃ N : ℕ, 
  N % 13 = 11 ∧ N % 17 = 9 ∧ N = 141 := by
sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l2520_252041


namespace NUMINAMATH_CALUDE_median_salary_is_40000_l2520_252016

/-- Represents a position in the company with its title, number of employees, and salary. -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company. -/
def positions : List Position := [
  ⟨"President", 1, 160000⟩,
  ⟨"Vice-President", 4, 105000⟩,
  ⟨"Director", 15, 80000⟩,
  ⟨"Associate Director", 10, 55000⟩,
  ⟨"Senior Manager", 20, 40000⟩,
  ⟨"Administrative Specialist", 50, 28000⟩
]

/-- The total number of employees in the company. -/
def totalEmployees : Nat := 100

/-- Calculates the median salary of the employees. -/
def medianSalary (pos : List Position) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary is $40,000. -/
theorem median_salary_is_40000 :
  medianSalary positions totalEmployees = 40000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_40000_l2520_252016


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l2520_252011

theorem middle_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ (x : ℚ), x > 0 ∧ a * x + b * x + c * x = total ∧ (∃ (n : ℕ), a * x = n ∨ b * x = n ∨ c * x = n) →
  b * x = 40 := by
  sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l2520_252011


namespace NUMINAMATH_CALUDE_larger_number_proof_l2520_252081

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 25) →
  (Nat.lcm a b = 4550) →
  (13 ∣ Nat.lcm a b) →
  (14 ∣ Nat.lcm a b) →
  max a b = 350 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2520_252081


namespace NUMINAMATH_CALUDE_alex_original_seat_l2520_252050

/-- Represents a seat in the movie theater --/
inductive Seat
| one | two | three | four | five | six | seven

/-- Represents the possible movements of friends --/
inductive Movement
| left : ℕ → Movement
| right : ℕ → Movement
| switch : Movement
| none : Movement

/-- Represents a friend in the theater --/
structure Friend :=
  (name : String)
  (initial_seat : Seat)
  (movement : Movement)

/-- The state of the theater --/
structure TheaterState :=
  (friends : List Friend)
  (alex_initial : Seat)
  (alex_final : Seat)

def is_end_seat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.seven

def move_left (s : Seat) (n : ℕ) : Seat :=
  match s, n with
  | Seat.one, _ => Seat.one
  | Seat.two, 1 => Seat.one
  | Seat.three, 1 => Seat.two
  | Seat.three, 2 => Seat.one
  | Seat.four, 1 => Seat.three
  | Seat.four, 2 => Seat.two
  | Seat.four, 3 => Seat.one
  | Seat.five, 1 => Seat.four
  | Seat.five, 2 => Seat.three
  | Seat.five, 3 => Seat.two
  | Seat.five, 4 => Seat.one
  | Seat.six, 1 => Seat.five
  | Seat.six, 2 => Seat.four
  | Seat.six, 3 => Seat.three
  | Seat.six, 4 => Seat.two
  | Seat.six, 5 => Seat.one
  | Seat.seven, 1 => Seat.six
  | Seat.seven, 2 => Seat.five
  | Seat.seven, 3 => Seat.four
  | Seat.seven, 4 => Seat.three
  | Seat.seven, 5 => Seat.two
  | Seat.seven, 6 => Seat.one
  | s, _ => s

theorem alex_original_seat (state : TheaterState) :
  state.friends = [
    ⟨"Bob", Seat.three, Movement.right 3⟩,
    ⟨"Cara", Seat.five, Movement.left 2⟩,
    ⟨"Dana", Seat.four, Movement.switch⟩,
    ⟨"Eve", Seat.two, Movement.switch⟩,
    ⟨"Fiona", Seat.six, Movement.right 1⟩,
    ⟨"Greg", Seat.seven, Movement.none⟩
  ] →
  is_end_seat state.alex_final →
  state.alex_initial = Seat.three :=
by sorry


end NUMINAMATH_CALUDE_alex_original_seat_l2520_252050


namespace NUMINAMATH_CALUDE_e_pow_f_neg_two_eq_half_l2520_252001

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem e_pow_f_neg_two_eq_half
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_log : ∀ x > 0, f x = Real.log x) :
  Real.exp (f (-2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_e_pow_f_neg_two_eq_half_l2520_252001


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2520_252042

theorem negation_of_universal_proposition (a : ℝ) (h : 0 < a ∧ a < 1) :
  (¬ ∀ x : ℝ, x < 0 → a^x > 1) ↔ (∃ x₀ : ℝ, x₀ < 0 ∧ a^x₀ ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2520_252042


namespace NUMINAMATH_CALUDE_regular_polygon_140_degrees_has_9_sides_l2520_252088

/-- A regular polygon with interior angles of 140 degrees has 9 sides -/
theorem regular_polygon_140_degrees_has_9_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 140 →
    (180 * (n - 2) : ℝ) = n * angle) →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_140_degrees_has_9_sides_l2520_252088


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2520_252074

theorem solution_set_inequality (x : ℝ) : (x - 2)^2 ≤ 2*x + 11 ↔ x ∈ Set.Icc (-1) 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2520_252074


namespace NUMINAMATH_CALUDE_percentage_not_sold_l2520_252091

def initial_stock : ℕ := 620
def monday_sales : ℕ := 50
def tuesday_sales : ℕ := 82
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

theorem percentage_not_sold (initial_stock monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ) :
  (initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100 =
  (620 - (50 + 82 + 60 + 48 + 40)) / 620 * 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_not_sold_l2520_252091


namespace NUMINAMATH_CALUDE_equation_has_seven_solutions_l2520_252006

/-- The function f(x) = |x² - 2x - 3| -/
def f (x : ℝ) : ℝ := |x^2 - 2*x - 3|

/-- The equation f³(x) - 4f²(x) - f(x) + 4 = 0 -/
def equation (x : ℝ) : Prop :=
  f x ^ 3 - 4 * (f x)^2 - f x + 4 = 0

/-- Theorem stating that the equation has exactly 7 solutions -/
theorem equation_has_seven_solutions :
  ∃! (s : Finset ℝ), s.card = 7 ∧ ∀ x, x ∈ s ↔ equation x :=
sorry

end NUMINAMATH_CALUDE_equation_has_seven_solutions_l2520_252006


namespace NUMINAMATH_CALUDE_melies_money_left_l2520_252051

/-- Calculates the amount of money left after buying meat. -/
def money_left (meat_amount : ℝ) (cost_per_kg : ℝ) (initial_money : ℝ) : ℝ :=
  initial_money - meat_amount * cost_per_kg

/-- Proves that Méliès has $16 left after buying meat. -/
theorem melies_money_left :
  let meat_amount : ℝ := 2
  let cost_per_kg : ℝ := 82
  let initial_money : ℝ := 180
  money_left meat_amount cost_per_kg initial_money = 16 := by
  sorry

end NUMINAMATH_CALUDE_melies_money_left_l2520_252051


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l2520_252013

/-- The volume of a tetrahedron with an inscribed sphere -/
theorem tetrahedron_volume (S₁ S₂ S₃ S₄ r : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) (h₃ : S₃ > 0) (h₄ : S₄ > 0) (hr : r > 0) :
  ∃ V : ℝ, V = (1/3) * (S₁ + S₂ + S₃ + S₄) * r ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l2520_252013


namespace NUMINAMATH_CALUDE_bo_learning_words_l2520_252048

/-- Calculates the number of words to learn per day given the total number of flashcards,
    the percentage of known words, and the number of days to learn. -/
def words_per_day (total_cards : ℕ) (known_percentage : ℚ) (days_to_learn : ℕ) : ℚ :=
  (total_cards - (known_percentage * total_cards)) / days_to_learn

/-- Proves that given 800 flashcards, 20% known words, and 40 days to learn,
    the number of words to learn per day is 16. -/
theorem bo_learning_words :
  words_per_day 800 (1/5) 40 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bo_learning_words_l2520_252048


namespace NUMINAMATH_CALUDE_jane_cans_count_l2520_252021

theorem jane_cans_count (total_seeds : ℝ) (seeds_per_can : ℕ) (h1 : total_seeds = 54.0) (h2 : seeds_per_can = 6) :
  (total_seeds / seeds_per_can : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_jane_cans_count_l2520_252021


namespace NUMINAMATH_CALUDE_identify_liars_in_two_questions_l2520_252069

/-- Represents a person who can be either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- Represents a position on a regular decagon -/
structure Position :=
  (angle : ℝ)

/-- Represents the state of the problem -/
structure DecagonState :=
  (people : Fin 10 → Person)
  (positions : Fin 10 → Position)

/-- Represents a question asked by the traveler -/
structure Question :=
  (position : Position)

/-- Represents an answer given by a person -/
structure Answer :=
  (distance : ℝ)

/-- Function to determine the answer given by a person -/
def getAnswer (state : DecagonState) (person : Fin 10) (q : Question) : Answer :=
  sorry

/-- Function to determine if a person is a liar based on their answer -/
def isLiar (state : DecagonState) (person : Fin 10) (q : Question) (a : Answer) : Bool :=
  sorry

/-- Theorem stating that at most 2 questions are needed to identify all liars -/
theorem identify_liars_in_two_questions (state : DecagonState) :
  ∃ (q1 q2 : Question), ∀ (person : Fin 10),
    isLiar state person q1 (getAnswer state person q1) ∨
    isLiar state person q2 (getAnswer state person q2) =
    (state.people person = Person.Liar) :=
  sorry

end NUMINAMATH_CALUDE_identify_liars_in_two_questions_l2520_252069


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l2520_252067

/-- Represents the fraction of pizza eaten by each sibling -/
structure PizzaConsumption where
  alex : Rat
  beth : Rat
  cyril : Rat
  dan : Rat

/-- Compares two rational numbers -/
def ratGreater (a b : Rat) : Prop := a > b

theorem pizza_consumption_order (pc : PizzaConsumption) : 
  pc.alex = 1/7 ∧ 
  pc.beth = 2/5 ∧ 
  pc.cyril = 3/10 ∧ 
  pc.dan = 2 * (1 - (pc.alex + pc.beth + pc.cyril)) →
  ratGreater pc.beth pc.dan ∧ 
  ratGreater pc.dan pc.cyril ∧ 
  ratGreater pc.cyril pc.alex :=
by sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l2520_252067


namespace NUMINAMATH_CALUDE_units_digits_divisible_by_eight_l2520_252019

theorem units_digits_divisible_by_eight :
  ∃! (digits : Finset Nat), 
    (∀ n : Nat, n % 8 = 0 → (n % 10) ∈ digits) ∧
    (∀ d ∈ digits, ∃ n : Nat, n % 8 = 0 ∧ n % 10 = d) ∧
    digits.card = 5 :=
by sorry

end NUMINAMATH_CALUDE_units_digits_divisible_by_eight_l2520_252019


namespace NUMINAMATH_CALUDE_f_properties_l2520_252026

def f (b c x : ℝ) : ℝ := x * abs x + b * x + c

theorem f_properties (b c : ℝ) :
  (∀ x, c = 0 → f b c (-x) = -f b c x) ∧
  (∀ x y, b = 0 → x < y → f b c x < f b c y) ∧
  (∀ x, f b c x - c = -(f b c (-x) - c)) ∧
  ¬(∀ b c, ∃ x y, f b c x = 0 ∧ f b c y = 0 ∧ ∀ z, f b c z = 0 → z = x ∨ z = y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2520_252026


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2520_252045

theorem child_ticket_cost (adult_price : ℕ) (total_people : ℕ) (total_revenue : ℕ) (num_children : ℕ) :
  adult_price = 11 →
  total_people = 23 →
  total_revenue = 246 →
  num_children = 7 →
  ∃ (child_price : ℕ), child_price = 10 ∧ 
    adult_price * (total_people - num_children) + child_price * num_children = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2520_252045


namespace NUMINAMATH_CALUDE_homework_difference_l2520_252053

theorem homework_difference (math reading history science : ℕ) 
  (h_math : math = 5)
  (h_reading : reading = 7)
  (h_history : history = 3)
  (h_science : science = 6) :
  (reading - math) + (science - history) = 5 :=
by sorry

end NUMINAMATH_CALUDE_homework_difference_l2520_252053


namespace NUMINAMATH_CALUDE_prob_two_dice_show_two_is_15_64_l2520_252092

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of at least one of two fair n-sided dice showing a specific number -/
def prob_at_least_one (n : ℕ) : ℚ :=
  1 - (n - 1)^2 / n^2

/-- The probability of at least one of two fair 8-sided dice showing a 2 -/
def prob_two_dice_show_two : ℚ := prob_at_least_one num_sides

theorem prob_two_dice_show_two_is_15_64 : 
  prob_two_dice_show_two = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_dice_show_two_is_15_64_l2520_252092


namespace NUMINAMATH_CALUDE_cage_cost_calculation_l2520_252044

def cat_toy_cost : ℝ := 10.22
def total_cost : ℝ := 21.95

theorem cage_cost_calculation : total_cost - cat_toy_cost = 11.73 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_calculation_l2520_252044


namespace NUMINAMATH_CALUDE_ab_length_is_two_l2520_252080

/-- Represents a point on a line --/
structure Point where
  position : ℝ

/-- Represents the distance between two points --/
def distance (p q : Point) : ℝ := abs (p.position - q.position)

/-- Theorem: Given points A, B, C, D on a line in order, if AC = 5, BD = 6, and CD = 3, then AB = 2 --/
theorem ab_length_is_two 
  (A B C D : Point) 
  (order : A.position < B.position ∧ B.position < C.position ∧ C.position < D.position)
  (ac_length : distance A C = 5)
  (bd_length : distance B D = 6)
  (cd_length : distance C D = 3) :
  distance A B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_length_is_two_l2520_252080


namespace NUMINAMATH_CALUDE_spherical_ball_radius_l2520_252093

/-- Given a cylindrical tub and a spherical iron ball, this theorem proves the radius of the ball
    based on the water level rise in the tub. -/
theorem spherical_ball_radius
  (tub_radius : ℝ)
  (water_rise : ℝ)
  (ball_radius : ℝ)
  (h1 : tub_radius = 12)
  (h2 : water_rise = 6.75)
  (h3 : (4 / 3) * Real.pi * ball_radius ^ 3 = Real.pi * tub_radius ^ 2 * water_rise) :
  ball_radius = 9 := by
  sorry

#check spherical_ball_radius

end NUMINAMATH_CALUDE_spherical_ball_radius_l2520_252093


namespace NUMINAMATH_CALUDE_tree_house_wood_needed_l2520_252064

-- Define the components of the tree house
structure TreeHouse where
  pillar_short : ℝ
  pillar_long : ℝ
  wall_short : ℝ
  wall_long : ℝ
  floor_avg : ℝ
  roof_first : ℝ
  roof_diff : ℝ

-- Define the function to calculate total wood needed
def total_wood (t : TreeHouse) : ℝ :=
  -- Pillars
  4 * t.pillar_short + 4 * t.pillar_long +
  -- Walls
  10 * t.wall_short + 10 * t.wall_long +
  -- Floor
  8 * t.floor_avg +
  -- Roof (arithmetic sequence sum formula)
  6 * t.roof_first + 15 * t.roof_diff

-- Theorem statement
theorem tree_house_wood_needed (t : TreeHouse) 
  (h1 : t.pillar_short = 4)
  (h2 : t.pillar_long = 5 * Real.sqrt t.pillar_short)
  (h3 : t.wall_short = 6)
  (h4 : t.wall_long = (2/3) * (t.wall_short ^ (3/2)))
  (h5 : t.floor_avg = 5.5)
  (h6 : t.roof_first = 2 * t.floor_avg)
  (h7 : t.roof_diff = (1/3) * t.pillar_short) :
  total_wood t = 344 := by
  sorry

end NUMINAMATH_CALUDE_tree_house_wood_needed_l2520_252064


namespace NUMINAMATH_CALUDE_parabola_single_intersection_l2520_252018

/-- A parabola with equation y = x^2 + 2x + k intersects the x-axis at only one point if and only if k = 1 -/
theorem parabola_single_intersection (k : ℝ) : 
  (∃! x, x^2 + 2*x + k = 0) ↔ k = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_single_intersection_l2520_252018


namespace NUMINAMATH_CALUDE_ball_max_height_l2520_252007

-- Define the height function
def h (t : ℝ) : ℝ := -4 * t^2 + 40 * t + 20

-- State the theorem
theorem ball_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 120 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l2520_252007


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2520_252078

/-- Given a total of 68 students in eighth grade with 28 girls, 
    the ratio of boys to girls is 10:7. -/
theorem boys_to_girls_ratio : 
  let total_students : ℕ := 68
  let girls : ℕ := 28
  let boys : ℕ := total_students - girls
  ∃ (a b : ℕ), a = 10 ∧ b = 7 ∧ boys * b = girls * a :=
by sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2520_252078


namespace NUMINAMATH_CALUDE_uncle_zhang_age_l2520_252009

/-- Represents the ages of Uncle Zhang and Uncle Li -/
structure UncleAges where
  zhang : ℕ
  li : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : UncleAges) : Prop :=
  ages.zhang + ages.li = 56 ∧
  ∃ (past_zhang : ℕ), past_zhang = ages.li / 2 ∧
  ages.zhang = ages.li - (ages.zhang - past_zhang)

/-- The theorem stating that Uncle Zhang's current age is 24 -/
theorem uncle_zhang_age :
  ∃ (ages : UncleAges), age_conditions ages ∧ ages.zhang = 24 := by
  sorry

end NUMINAMATH_CALUDE_uncle_zhang_age_l2520_252009


namespace NUMINAMATH_CALUDE_enemies_left_proof_l2520_252085

def enemies_left_undefeated (total_enemies : ℕ) (points_per_enemy : ℕ) (total_points : ℕ) : ℕ :=
  total_enemies - (total_points / points_per_enemy)

theorem enemies_left_proof (total_enemies : ℕ) (points_per_enemy : ℕ) (total_points : ℕ)
  (h1 : total_enemies = 11)
  (h2 : points_per_enemy = 9)
  (h3 : total_points = 72) :
  enemies_left_undefeated total_enemies points_per_enemy total_points = 3 :=
by
  sorry

#eval enemies_left_undefeated 11 9 72

end NUMINAMATH_CALUDE_enemies_left_proof_l2520_252085


namespace NUMINAMATH_CALUDE_lg_100_is_proposition_l2520_252015

/-- A proposition is a declarative sentence that can be judged to be true or false. -/
def IsProposition (s : String) : Prop := 
  ∃ (truthValue : Bool), (∀ (evaluation : String → Bool), evaluation s = truthValue)

/-- The statement "lg 100 = 2" -/
def statement : String := "lg 100 = 2"

/-- Theorem: The statement "lg 100 = 2" is a proposition -/
theorem lg_100_is_proposition : IsProposition statement := by
  sorry

end NUMINAMATH_CALUDE_lg_100_is_proposition_l2520_252015


namespace NUMINAMATH_CALUDE_faiths_weekly_earnings_l2520_252049

/-- Calculates the total weekly earnings for Faith given her work conditions --/
def total_weekly_earnings (
  hourly_wage : ℝ)
  (regular_hours_per_day : ℝ)
  (regular_days_per_week : ℝ)
  (overtime_hours_per_day : ℝ)
  (overtime_days_per_week : ℝ)
  (overtime_rate_multiplier : ℝ)
  (commission_rate : ℝ)
  (total_sales : ℝ) : ℝ :=
  let regular_earnings := hourly_wage * regular_hours_per_day * regular_days_per_week
  let overtime_earnings := hourly_wage * overtime_rate_multiplier * overtime_hours_per_day * overtime_days_per_week
  let commission := commission_rate * total_sales
  regular_earnings + overtime_earnings + commission

/-- Theorem stating that Faith's total weekly earnings are $1,062.50 --/
theorem faiths_weekly_earnings :
  total_weekly_earnings 13.5 8 5 2 5 1.5 0.1 3200 = 1062.5 := by
  sorry

end NUMINAMATH_CALUDE_faiths_weekly_earnings_l2520_252049


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l2520_252052

theorem exam_pass_percentage
  (failed_hindi : ℚ)
  (failed_english : ℚ)
  (failed_both : ℚ)
  (h1 : failed_hindi = 25 / 100)
  (h2 : failed_english = 48 / 100)
  (h3 : failed_both = 27 / 100) :
  1 - (failed_hindi + failed_english - failed_both) = 54 / 100 :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l2520_252052


namespace NUMINAMATH_CALUDE_min_box_value_l2520_252099

theorem min_box_value (a b Box : ℤ) : 
  (a ≠ b ∧ a ≠ Box ∧ b ≠ Box) →
  (∀ x, (a * x + b) * (b * x + a) = 31 * x^2 + Box * x + 31) →
  962 ≤ Box :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l2520_252099


namespace NUMINAMATH_CALUDE_odd_square_plus_four_odd_l2520_252025

theorem odd_square_plus_four_odd (p q : ℕ) 
  (hp : Odd p) (hq : Odd q) (hp_pos : 0 < p) (hq_pos : 0 < q) : 
  Odd (p^2 + 4*q) := by
  sorry

end NUMINAMATH_CALUDE_odd_square_plus_four_odd_l2520_252025


namespace NUMINAMATH_CALUDE_complex_modulus_l2520_252079

theorem complex_modulus (z : ℂ) :
  (((2 : ℂ) + 4 * I) / z = 1 + I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2520_252079


namespace NUMINAMATH_CALUDE_tyrah_sarah_pencil_ratio_l2520_252035

/-- Given that Tyrah has 12 pencils and Sarah has 2 pencils, 
    prove that the ratio of Tyrah's pencils to Sarah's pencils is 6. -/
theorem tyrah_sarah_pencil_ratio :
  ∀ (tyrah_pencils sarah_pencils : ℕ),
    tyrah_pencils = 12 →
    sarah_pencils = 2 →
    (tyrah_pencils : ℚ) / sarah_pencils = 6 := by
  sorry

end NUMINAMATH_CALUDE_tyrah_sarah_pencil_ratio_l2520_252035


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l2520_252055

def expression (x : ℝ) : ℝ :=
  5 * (x^2 - 2*x^4) + 3 * (2*x - 3*x^2 + 4*x^3) - 2 * (2*x^4 - 3*x^2)

theorem coefficient_of_x_squared :
  ∃ (a b c d e : ℝ), ∀ x, expression x = a*x^4 + b*x^3 + 2*x^2 + d*x + e :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l2520_252055


namespace NUMINAMATH_CALUDE_inequality_proof_l2520_252012

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 1) (h3 : 0 < n) :
  (1 + x)^n ≥ (1 - x)^n + 2 * n * x * (1 - x^2)^((n - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2520_252012


namespace NUMINAMATH_CALUDE_trigonometric_identities_and_circle_parametrization_l2520_252068

theorem trigonometric_identities_and_circle_parametrization (a t : ℝ) 
  (h : t = Real.tan (a / 2)) : 
  Real.cos a = (1 - t^2) / (1 + t^2) ∧ 
  Real.sin a = 2 * t / (1 + t^2) ∧ 
  Real.tan a = 2 * t / (1 - t^2) ∧ 
  ∀ x y : ℝ, x = (1 - t^2) / (1 + t^2) ∧ y = 2 * t / (1 + t^2) → x^2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_and_circle_parametrization_l2520_252068


namespace NUMINAMATH_CALUDE_gcf_lcm_60_72_l2520_252089

theorem gcf_lcm_60_72 : 
  (Nat.gcd 60 72 = 12) ∧ (Nat.lcm 60 72 = 360) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_60_72_l2520_252089


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2520_252090

theorem quadratic_inequality_solution_set (m : ℝ) :
  {x : ℝ | x^2 - (2*m + 1)*x + m^2 + m > 0} = {x : ℝ | x < m ∨ x > m + 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2520_252090


namespace NUMINAMATH_CALUDE_sequence_formula_correct_l2520_252027

def a (n : ℕ) : ℤ := (-1)^n * (4*n - 3)

theorem sequence_formula_correct : 
  (a 1 = -1) ∧ (a 2 = 5) ∧ (a 3 = -9) ∧ (a 4 = 13) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_correct_l2520_252027


namespace NUMINAMATH_CALUDE_ten_digit_numbers_with_repeats_l2520_252073

theorem ten_digit_numbers_with_repeats (n : ℕ) : n = 9 * 10^9 - 9 * Nat.factorial 9 :=
  by
    sorry

end NUMINAMATH_CALUDE_ten_digit_numbers_with_repeats_l2520_252073


namespace NUMINAMATH_CALUDE_fraction_change_l2520_252043

theorem fraction_change (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : 
  (2*a) * (2*b) / (2*(2*a) + 2*b) = 2 * (a * b / (2*a + b)) :=
sorry

end NUMINAMATH_CALUDE_fraction_change_l2520_252043


namespace NUMINAMATH_CALUDE_great_white_shark_teeth_l2520_252082

/-- The number of teeth of different shark species -/
def shark_teeth : ℕ → ℕ
| 0 => 180  -- tiger shark
| 1 => shark_teeth 0 / 6  -- hammerhead shark
| 2 => 2 * (shark_teeth 0 + shark_teeth 1)  -- great white shark
| _ => 0  -- other sharks (not relevant for this problem)

theorem great_white_shark_teeth : shark_teeth 2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_great_white_shark_teeth_l2520_252082


namespace NUMINAMATH_CALUDE_compute_expression_l2520_252084

theorem compute_expression : 15 * (30 / 6)^2 = 375 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l2520_252084


namespace NUMINAMATH_CALUDE_distance_product_l2520_252002

theorem distance_product (a₁ a₂ : ℝ) : 
  let p₁ := (3 * a₁, 2 * a₁ - 5)
  let p₂ := (6, -2)
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2 = (3 * Real.sqrt 17)^2 →
  let p₁ := (3 * a₂, 2 * a₂ - 5)
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2 = (3 * Real.sqrt 17)^2 →
  a₁ * a₂ = -2880 / 169 := by
sorry

end NUMINAMATH_CALUDE_distance_product_l2520_252002


namespace NUMINAMATH_CALUDE_equation_roots_range_l2520_252065

theorem equation_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 2*k*x₁^2 + (8*k+1)*x₁ = -8*k ∧ 2*k*x₂^2 + (8*k+1)*x₂ = -8*k) →
  (k > -1/16 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_range_l2520_252065


namespace NUMINAMATH_CALUDE_mary_sugar_addition_l2520_252037

/-- The amount of sugar Mary needs to add to her cake mix -/
def sugar_to_add (required_sugar : ℕ) (added_sugar : ℕ) : ℕ :=
  required_sugar - added_sugar

theorem mary_sugar_addition : sugar_to_add 11 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_mary_sugar_addition_l2520_252037


namespace NUMINAMATH_CALUDE_linda_savings_l2520_252071

theorem linda_savings (savings : ℝ) : (1 / 4 : ℝ) * savings = 220 → savings = 880 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_l2520_252071


namespace NUMINAMATH_CALUDE_b_reaches_a_in_120_minutes_l2520_252054

/-- Represents the walking scenario of two people A and B -/
structure WalkingScenario where
  speed_B : ℝ  -- B's speed in meters per minute
  initial_distance : ℝ  -- Initial distance between A and B in meters
  meeting_time : ℝ  -- Time when A and B meet in minutes

/-- Calculates the time for B to reach point A after A has reached point B -/
def time_for_B_to_reach_A (scenario : WalkingScenario) : ℝ :=
  -- We'll implement the calculation here
  sorry

/-- Theorem stating that given the conditions, B will take 120 minutes to reach A after A reaches B -/
theorem b_reaches_a_in_120_minutes (scenario : WalkingScenario) 
    (h1 : scenario.meeting_time = 60)
    (h2 : scenario.initial_distance = 4 * scenario.speed_B * scenario.meeting_time) : 
    time_for_B_to_reach_A scenario = 120 :=
  sorry

end NUMINAMATH_CALUDE_b_reaches_a_in_120_minutes_l2520_252054


namespace NUMINAMATH_CALUDE_circle_plus_inequality_equiv_l2520_252056

/-- The custom operation ⊕ defined on ℝ -/
def circle_plus (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the equivalence between the inequality and the range of x -/
theorem circle_plus_inequality_equiv (x : ℝ) :
  circle_plus (x - 1) (x + 2) < 0 ↔ x < -1 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_inequality_equiv_l2520_252056


namespace NUMINAMATH_CALUDE_factor_expression_l2520_252087

theorem factor_expression (y : ℝ) : 3 * y^2 - 75 = 3 * (y - 5) * (y + 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2520_252087


namespace NUMINAMATH_CALUDE_fiftieth_ring_l2520_252036

/-- Represents the number of squares in the nth ring -/
def S (n : ℕ) : ℕ := 10 * n - 2

/-- The properties of the sequence of rings -/
axiom first_ring : S 1 = 8
axiom second_ring : S 2 = 18
axiom ring_increase (n : ℕ) : n ≥ 2 → S (n + 1) - S n = 10

/-- The theorem stating the number of squares in the 50th ring -/
theorem fiftieth_ring : S 50 = 498 := by sorry

end NUMINAMATH_CALUDE_fiftieth_ring_l2520_252036


namespace NUMINAMATH_CALUDE_no_odd_white_columns_exists_odd_black_columns_l2520_252057

/-- Represents a 3x3x3 cube composed of white and black unit cubes -/
structure Cube :=
  (white_count : Nat)
  (black_count : Nat)
  (total_count : Nat)
  (is_valid : white_count + black_count = total_count ∧ total_count = 27)

/-- Represents a column in the cube -/
structure Column :=
  (white_count : Nat)
  (black_count : Nat)
  (is_valid : white_count + black_count = 3)

/-- Checks if a number is odd -/
def is_odd (n : Nat) : Prop := n % 2 = 1

/-- Theorem: It is impossible for each column to contain an odd number of white cubes -/
theorem no_odd_white_columns (c : Cube) (h : c.white_count = 14 ∧ c.black_count = 13) :
  ¬ (∀ col : Column, is_odd col.white_count) :=
sorry

/-- Theorem: It is possible for each column to contain an odd number of black cubes -/
theorem exists_odd_black_columns (c : Cube) (h : c.white_count = 14 ∧ c.black_count = 13) :
  ∃ (arrangement : List Column), (∀ col ∈ arrangement, is_odd col.black_count) ∧ 
    arrangement.length = 27 ∧ (arrangement.map Column.black_count).sum = 13 :=
sorry

end NUMINAMATH_CALUDE_no_odd_white_columns_exists_odd_black_columns_l2520_252057


namespace NUMINAMATH_CALUDE_pages_read_difference_l2520_252033

theorem pages_read_difference (total_pages : ℕ) (fraction_read : ℚ) : 
  total_pages = 90 → fraction_read = 2/3 → 
  (total_pages : ℚ) * fraction_read - (total_pages : ℚ) * (1 - fraction_read) = 30 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_difference_l2520_252033


namespace NUMINAMATH_CALUDE_cycle_gain_percentage_l2520_252032

/-- Calculate the overall gain percentage for three cycles given their purchase and sale prices -/
theorem cycle_gain_percentage
  (purchase_a purchase_b purchase_c : ℕ)
  (sale_a sale_b sale_c : ℕ)
  (h_purchase_a : purchase_a = 1000)
  (h_purchase_b : purchase_b = 3000)
  (h_purchase_c : purchase_c = 6000)
  (h_sale_a : sale_a = 2000)
  (h_sale_b : sale_b = 4500)
  (h_sale_c : sale_c = 8000) :
  (((sale_a + sale_b + sale_c) - (purchase_a + purchase_b + purchase_c)) * 100) / (purchase_a + purchase_b + purchase_c) = 45 := by
  sorry


end NUMINAMATH_CALUDE_cycle_gain_percentage_l2520_252032


namespace NUMINAMATH_CALUDE_greatest_power_of_two_dividing_expression_l2520_252061

theorem greatest_power_of_two_dividing_expression : ∃ k : ℕ, 
  (k = 1007 ∧ 
   2^k ∣ (10^1004 - 4^502) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1004 - 4^502) → m ≤ k) := by
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_dividing_expression_l2520_252061


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2520_252040

theorem opposite_of_negative_two : 
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2520_252040


namespace NUMINAMATH_CALUDE_trapezoid_area_equality_l2520_252075

/-- Given a square ABCD with side length a, and a trapezoid EBCF inside it with BE = CF = x,
    if the area of EBCF equals the area of ABCD minus twice the area of a rectangle JKHG 
    inside the square, then x = a/2 -/
theorem trapezoid_area_equality (a : ℝ) (x : ℝ) :
  (∃ (y z : ℝ), y + z = a ∧ x * a = a^2 - 2 * y * z) →
  x = a / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_equality_l2520_252075


namespace NUMINAMATH_CALUDE_log_inequality_iff_x_range_l2520_252023

-- Define the domain constraints
def domain (x : ℝ) : Prop := x > -2 ∧ x ≠ -1

-- Define the logarithmic inequality
def log_inequality (x : ℝ) : Prop :=
  Real.log (8 + x^3) / Real.log (2 + x) ≤ Real.log ((2 + x)^3) / Real.log (2 + x)

-- State the theorem
theorem log_inequality_iff_x_range (x : ℝ) :
  domain x → (log_inequality x ↔ (-2 < x ∧ x < -1) ∨ x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_iff_x_range_l2520_252023


namespace NUMINAMATH_CALUDE_athlete_stop_point_l2520_252010

/-- Represents a rectangular square with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a point on the perimeter of a rectangle -/
structure PerimeterPoint where
  distance : ℝ  -- Distance from a chosen starting point

/-- The athlete's run around the rectangular square -/
def athleteRun (rect : Rectangle) (start : PerimeterPoint) (distance : ℝ) : PerimeterPoint :=
  sorry

theorem athlete_stop_point (rect : Rectangle) (start : PerimeterPoint) :
  let totalDistance : ℝ := 15500  -- 15.5 km in meters
  rect.length = 900 ∧ rect.width = 600 ∧ start.distance = 550 →
  (athleteRun rect start totalDistance).distance = 150 :=
sorry

end NUMINAMATH_CALUDE_athlete_stop_point_l2520_252010


namespace NUMINAMATH_CALUDE_yoo_jeong_borrowed_nine_notebooks_l2520_252024

/-- The number of notebooks Min-young originally had -/
def original_notebooks : ℕ := 17

/-- The number of notebooks Min-young had left after lending -/
def remaining_notebooks : ℕ := 8

/-- The number of notebooks Yoo-jeong borrowed -/
def borrowed_notebooks : ℕ := original_notebooks - remaining_notebooks

theorem yoo_jeong_borrowed_nine_notebooks : borrowed_notebooks = 9 := by
  sorry

end NUMINAMATH_CALUDE_yoo_jeong_borrowed_nine_notebooks_l2520_252024


namespace NUMINAMATH_CALUDE_camp_distribution_correct_l2520_252098

/-- Represents a summer camp with three sub-camps -/
structure SummerCamp where
  totalStudents : Nat
  sampleSize : Nat
  firstDrawn : Nat
  campIEnd : Nat
  campIIEnd : Nat

/-- Calculates the number of students drawn from each camp -/
def campDistribution (camp : SummerCamp) : (Nat × Nat × Nat) :=
  sorry

/-- Theorem stating the correct distribution of sampled students across camps -/
theorem camp_distribution_correct (camp : SummerCamp) 
  (h1 : camp.totalStudents = 720)
  (h2 : camp.sampleSize = 60)
  (h3 : camp.firstDrawn = 4)
  (h4 : camp.campIEnd = 360)
  (h5 : camp.campIIEnd = 640) :
  campDistribution camp = (30, 24, 6) := by
  sorry

end NUMINAMATH_CALUDE_camp_distribution_correct_l2520_252098


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2520_252014

theorem cyclic_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_abc : a + b + c = 3) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) + 
  Real.sqrt 2 * (Real.sqrt (a / (b + c)) + Real.sqrt (b / (c + a)) + Real.sqrt (c / (a + b))) 
  ≥ 9/2 := by
sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2520_252014


namespace NUMINAMATH_CALUDE_reciprocal_of_25_l2520_252034

theorem reciprocal_of_25 (x : ℝ) : (1 / x = 25) → (x = 1 / 25) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_25_l2520_252034


namespace NUMINAMATH_CALUDE_class_size_l2520_252070

theorem class_size (mini_cupcakes : ℕ) (donut_holes : ℕ) (desserts_per_student : ℕ) : 
  mini_cupcakes = 14 → 
  donut_holes = 12 → 
  desserts_per_student = 2 → 
  (mini_cupcakes + donut_holes) / desserts_per_student = 13 := by
sorry

end NUMINAMATH_CALUDE_class_size_l2520_252070


namespace NUMINAMATH_CALUDE_triangle_inequality_l2520_252046

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2520_252046


namespace NUMINAMATH_CALUDE_smallest_divisible_by_three_l2520_252020

theorem smallest_divisible_by_three :
  ∃ (B : ℕ), B < 10 ∧ 
    (∀ (k : ℕ), k < B → ¬(800000 + 100000 * k + 4635) % 3 = 0) ∧
    (800000 + 100000 * B + 4635) % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_three_l2520_252020


namespace NUMINAMATH_CALUDE_log3_20_approximation_l2520_252066

-- Define the approximate values given in the problem
def log10_2_approx : ℝ := 0.301
def log10_3_approx : ℝ := 0.477

-- Define the target approximation
def log3_20_target : ℝ := 2.786

-- State the theorem
theorem log3_20_approximation :
  abs (Real.log 20 / Real.log 3 - log3_20_target) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_log3_20_approximation_l2520_252066


namespace NUMINAMATH_CALUDE_quadratic_function_max_abs_value_ge_one_l2520_252097

/-- Given a quadratic function f(x) = 2x^2 + mx + n, 
    prove that the maximum absolute value of f(1), f(2), and f(3) is at least 1. -/
theorem quadratic_function_max_abs_value_ge_one (m n : ℝ) : 
  let f := fun (x : ℝ) => 2 * x^2 + m * x + n
  max (|f 1|) (max (|f 2|) (|f 3|)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_max_abs_value_ge_one_l2520_252097


namespace NUMINAMATH_CALUDE_imaginary_sum_zero_l2520_252029

theorem imaginary_sum_zero (i : ℂ) (h : i^2 = -1) :
  i^15324 + i^15325 + i^15326 + i^15327 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_zero_l2520_252029


namespace NUMINAMATH_CALUDE_negation_equivalence_l2520_252094

variable (a : ℝ)

theorem negation_equivalence :
  (¬ ∀ x : ℝ, (x - a)^2 + 2 > 0) ↔ (∃ x : ℝ, (x - a)^2 + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2520_252094
