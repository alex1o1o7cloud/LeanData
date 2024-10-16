import Mathlib

namespace NUMINAMATH_CALUDE_odd_function_range_l1116_111658

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is periodic with period p if f(x + p) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_range (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : IsOdd f)
  (h_sum : ∀ x, f x + f (x + 3/2) = 0)
  (h_f1 : f 1 > 1)
  (h_f2 : f 2 = a) :
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_range_l1116_111658


namespace NUMINAMATH_CALUDE_transformation_result_l1116_111689

/-- Rotates a point (x, y) 180° counterclockwise around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2*h - x, 2*k - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqualX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem transformation_result (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectAboutYEqualX rotated.1 rotated.2
  final = (1, -4) → b - a = -3 := by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l1116_111689


namespace NUMINAMATH_CALUDE_festival_attendance_l1116_111669

theorem festival_attendance (total : ℕ) (first_day : ℕ) : 
  total = 2700 →
  first_day + (first_day / 2) + (3 * first_day) = total →
  first_day / 2 = 300 :=
by sorry

end NUMINAMATH_CALUDE_festival_attendance_l1116_111669


namespace NUMINAMATH_CALUDE_vector_subtraction_norm_l1116_111680

def a : Fin 3 → ℝ := ![1, 0, 2]
def b : Fin 3 → ℝ := ![0, 1, 2]

theorem vector_subtraction_norm : 
  ‖(fun i => a i - 2 * b i)‖ = 3 := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_norm_l1116_111680


namespace NUMINAMATH_CALUDE_vector_simplification_1_vector_simplification_2_l1116_111697

variable {V : Type*} [AddCommGroup V]

-- Define vectors
variable (A B C D E O : V)

-- Define the vector operations
def vec (X Y : V) := Y - X

-- Theorem statements
theorem vector_simplification_1 :
  (vec B A - vec B C) - (vec E D - vec E C) = vec D A := by sorry

theorem vector_simplification_2 :
  (vec A C + vec B O + vec O A) - (vec D C - vec D O - vec O B) = 0 := by sorry

end NUMINAMATH_CALUDE_vector_simplification_1_vector_simplification_2_l1116_111697


namespace NUMINAMATH_CALUDE_fraction_of_states_1790s_l1116_111661

/-- The number of states that joined the union during 1790-1799 -/
def states_joined_1790s : ℕ := 7

/-- The total number of states considered -/
def total_states : ℕ := 30

/-- The fraction of states that joined during 1790-1799 out of the first 30 states -/
theorem fraction_of_states_1790s :
  (states_joined_1790s : ℚ) / total_states = 7 / 30 := by sorry

end NUMINAMATH_CALUDE_fraction_of_states_1790s_l1116_111661


namespace NUMINAMATH_CALUDE_liquor_and_beer_cost_l1116_111622

/-- The price of one bottle of beer in yuan -/
def beer_price : ℚ := 2

/-- The price of one bottle of liquor in yuan -/
def liquor_price : ℚ := 16

/-- The total cost of 2 bottles of liquor and 12 bottles of beer in yuan -/
def total_cost : ℚ := 56

/-- The number of bottles of beer equivalent in price to one bottle of liquor -/
def liquor_to_beer_ratio : ℕ := 8

theorem liquor_and_beer_cost :
  (2 * liquor_price + 12 * beer_price = total_cost) →
  (liquor_price = liquor_to_beer_ratio * beer_price) →
  (liquor_price + beer_price = 18) := by
    sorry

end NUMINAMATH_CALUDE_liquor_and_beer_cost_l1116_111622


namespace NUMINAMATH_CALUDE_f_sin_75_eq_zero_l1116_111603

-- Define the function f
def f (a₄ a₃ a₂ a₁ a₀ : ℤ) (x : ℝ) : ℝ :=
  a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

-- State the theorem
theorem f_sin_75_eq_zero 
  (a₄ a₃ a₂ a₁ a₀ : ℤ) 
  (h₁ : f a₄ a₃ a₂ a₁ a₀ (Real.cos (75 * π / 180)) = 0) 
  (h₂ : a₄ ≠ 0) : 
  f a₄ a₃ a₂ a₁ a₀ (Real.sin (75 * π / 180)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sin_75_eq_zero_l1116_111603


namespace NUMINAMATH_CALUDE_max_distance_ellipse_point_l1116_111635

/-- 
Given an ellipse x²/a² + y²/b² = 1 with a > b > 0, and A(0, b),
the maximum value of |PA| for any point P on the ellipse is max(a²/√(a² - b²), 2b).
-/
theorem max_distance_ellipse_point (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let ellipse := {P : ℝ × ℝ | (P.1^2 / a^2) + (P.2^2 / b^2) = 1}
  let A := (0, b)
  let dist_PA (P : ℝ × ℝ) := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  (∀ P ∈ ellipse, dist_PA P ≤ max (a^2 / Real.sqrt (a^2 - b^2)) (2*b)) ∧
  (∃ P ∈ ellipse, dist_PA P = max (a^2 / Real.sqrt (a^2 - b^2)) (2*b))
:= by sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_point_l1116_111635


namespace NUMINAMATH_CALUDE_fixed_point_of_log_function_l1116_111681

-- Define the logarithm function with base a
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x + 2) + 3
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log_base a (x + 2) + 3

-- Theorem statement
theorem fixed_point_of_log_function (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
  f a (-1) = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_log_function_l1116_111681


namespace NUMINAMATH_CALUDE_train_speed_problem_l1116_111656

/-- Prove that given two trains of equal length 62.5 meters, where the faster train
    travels at 46 km/hr and passes the slower train in 45 seconds, the speed of
    the slower train is 36 km/hr. -/
theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 62.5 →
  faster_speed = 46 →
  passing_time = 45 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * (1000 / 3600) * passing_time = 2 * train_length :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1116_111656


namespace NUMINAMATH_CALUDE_circle_area_through_points_l1116_111613

/-- The area of a circle with center P(-3, 4) passing through Q(9, -3) is 193π square units. -/
theorem circle_area_through_points : 
  let P : ℝ × ℝ := (-3, 4)
  let Q : ℝ × ℝ := (9, -3)
  let r : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 193 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l1116_111613


namespace NUMINAMATH_CALUDE_largest_three_digit_geometric_sequence_l1116_111637

def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, b = a * r ∧ c = b * r

def digits_are_distinct (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem largest_three_digit_geometric_sequence :
  ∀ n : ℕ,
    100 ≤ n ∧ n < 1000 ∧
    (n / 100 = 8) ∧
    digits_are_distinct n ∧
    is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10) →
    n ≤ 842 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_geometric_sequence_l1116_111637


namespace NUMINAMATH_CALUDE_broken_glass_problem_l1116_111639

/-- The number of broken glass pieces during transportation --/
def broken_glass (total : ℕ) (safe_fee : ℕ) (compensation : ℕ) (total_fee : ℕ) : ℕ :=
  total - (total_fee + total * safe_fee) / (safe_fee + compensation)

theorem broken_glass_problem :
  broken_glass 100 3 5 260 = 5 := by
  sorry

end NUMINAMATH_CALUDE_broken_glass_problem_l1116_111639


namespace NUMINAMATH_CALUDE_cricket_problem_l1116_111676

/-- Represents the runs scored by each batsman in a cricket match -/
structure CricketScores where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ

/-- Theorem representing the cricket problem -/
theorem cricket_problem (scores : CricketScores) : scores.E = 20 :=
  by
  have h1 : scores.A + scores.B + scores.C + scores.D + scores.E = 180 := 
    sorry -- Average score is 36, so total is 5 * 36 = 180
  have h2 : scores.D = scores.E + 5 := 
    sorry -- D scored 5 more than E
  have h3 : scores.E = scores.A - 8 := 
    sorry -- E scored 8 fewer than A
  have h4 : scores.B = scores.D + scores.E := 
    sorry -- B scored as many as D and E combined
  have h5 : scores.B + scores.C = 107 := 
    sorry -- B and C scored 107 between them
  sorry -- Proof that E = 20

end NUMINAMATH_CALUDE_cricket_problem_l1116_111676


namespace NUMINAMATH_CALUDE_weight_of_ten_moles_C6H8O6_l1116_111631

/-- The weight of 10 moles of C6H8O6 -/
theorem weight_of_ten_moles_C6H8O6 
  (atomic_weight_C : ℝ) 
  (atomic_weight_H : ℝ) 
  (atomic_weight_O : ℝ) 
  (h1 : atomic_weight_C = 12.01)
  (h2 : atomic_weight_H = 1.008)
  (h3 : atomic_weight_O = 16.00) : 
  10 * (6 * atomic_weight_C + 8 * atomic_weight_H + 6 * atomic_weight_O) = 1761.24 := by
sorry

end NUMINAMATH_CALUDE_weight_of_ten_moles_C6H8O6_l1116_111631


namespace NUMINAMATH_CALUDE_happy_street_weekend_traffic_l1116_111648

/-- Number of cars passing Happy Street each day of the week -/
structure WeekTraffic where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  weekend_day : ℕ

/-- Conditions for the Happy Street traffic problem -/
def happy_street_conditions (w : WeekTraffic) : Prop :=
  w.tuesday = 25 ∧
  w.monday = w.tuesday - (w.tuesday / 5) ∧
  w.wednesday = w.monday + 2 ∧
  w.thursday = 10 ∧
  w.friday = 10 ∧
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + 2 * w.weekend_day = 97

theorem happy_street_weekend_traffic (w : WeekTraffic) 
  (h : happy_street_conditions w) : w.weekend_day = 5 := by
  sorry


end NUMINAMATH_CALUDE_happy_street_weekend_traffic_l1116_111648


namespace NUMINAMATH_CALUDE_max_remainder_of_division_by_11_l1116_111606

theorem max_remainder_of_division_by_11 (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A = 11 * B + C →
  C ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_of_division_by_11_l1116_111606


namespace NUMINAMATH_CALUDE_min_values_theorem_l1116_111623

theorem min_values_theorem :
  (∀ x > 1, x + 4 / (x - 1) ≥ 5) ∧
  (∀ a b, a > 0 → b > 0 → a + b = a * b → 9 * a + b ≥ 16) := by
sorry

end NUMINAMATH_CALUDE_min_values_theorem_l1116_111623


namespace NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l1116_111660

theorem unique_solution_for_rational_equation :
  ∃! x : ℝ, x ≠ 3 ∧ (x^2 - 9) / (x - 3) = 3 * x :=
by
  -- The unique solution is x = 3/2
  use 3/2
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l1116_111660


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_2017_power_plus_2017_l1116_111696

theorem infinite_primes_dividing_2017_power_plus_2017 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ 2017^(2^n) + 2017} := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_2017_power_plus_2017_l1116_111696


namespace NUMINAMATH_CALUDE_janice_started_sentences_l1116_111615

/-- Represents Janice's typing session --/
structure TypingSession where
  initial_speed : ℕ
  first_duration : ℕ
  second_speed : ℕ
  second_duration : ℕ
  third_speed : ℕ
  third_duration : ℕ
  erased_sentences : ℕ
  final_speed : ℕ
  final_duration : ℕ
  total_sentences : ℕ

/-- Calculates the number of sentences Janice started with --/
def sentences_started_with (session : TypingSession) : ℕ :=
  session.total_sentences -
  (session.initial_speed * session.first_duration +
   session.second_speed * session.second_duration +
   session.third_speed * session.third_duration -
   session.erased_sentences +
   session.final_speed * session.final_duration)

/-- Theorem stating that Janice started with 246 sentences --/
theorem janice_started_sentences (session : TypingSession)
  (h1 : session.initial_speed = 6)
  (h2 : session.first_duration = 10)
  (h3 : session.second_speed = 7)
  (h4 : session.second_duration = 10)
  (h5 : session.third_speed = 7)
  (h6 : session.third_duration = 15)
  (h7 : session.erased_sentences = 35)
  (h8 : session.final_speed = 5)
  (h9 : session.final_duration = 18)
  (h10 : session.total_sentences = 536) :
  sentences_started_with session = 246 := by
  sorry

end NUMINAMATH_CALUDE_janice_started_sentences_l1116_111615


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1116_111699

/-- For a > 0 and a ≠ 1, the function f(x) = a^(x-2) - 3 passes through the point (2, -2) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x : ℝ, a^(x - 2) - 3 = x ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1116_111699


namespace NUMINAMATH_CALUDE_polynomial_properties_l1116_111650

/-- Definition of the polynomial -/
def p (x y : ℝ) : ℝ := -5*x^2 - x*y^4 + 2^6*x*y + 3

/-- The number of terms in the polynomial -/
def num_terms : ℕ := 4

/-- The degree of the polynomial -/
def degree : ℕ := 5

/-- The coefficient of the highest degree term -/
def highest_coeff : ℝ := -1

/-- Theorem stating the properties of the polynomial -/
theorem polynomial_properties :
  (num_terms = 4) ∧ 
  (degree = 5) ∧ 
  (highest_coeff = -1) := by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1116_111650


namespace NUMINAMATH_CALUDE_checkers_tie_fraction_l1116_111647

theorem checkers_tie_fraction (ben_win_rate sara_win_rate : ℚ) 
  (h1 : ben_win_rate = 2/5)
  (h2 : sara_win_rate = 1/4) : 
  1 - (ben_win_rate + sara_win_rate) = 7/20 := by
sorry

end NUMINAMATH_CALUDE_checkers_tie_fraction_l1116_111647


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_geq_one_l1116_111629

theorem intersection_empty_implies_a_geq_one (a : ℝ) : 
  let A : Set ℝ := {0, 1}
  let B : Set ℝ := {x | x > a}
  A ∩ B = ∅ → a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_geq_one_l1116_111629


namespace NUMINAMATH_CALUDE_max_integer_value_of_expression_l1116_111679

theorem max_integer_value_of_expression (x : ℝ) : 
  (4 * x^2 + 8 * x + 21) / (4 * x^2 + 8 * x + 9) ≤ 7/3 ∧ 
  ∃ (y : ℝ), (4 * y^2 + 8 * y + 21) / (4 * y^2 + 8 * y + 9) > 2 ∧
  ∀ (z : ℝ), (4 * z^2 + 8 * z + 21) / (4 * z^2 + 8 * z + 9) < 3 := by
  sorry

end NUMINAMATH_CALUDE_max_integer_value_of_expression_l1116_111679


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1116_111693

theorem arithmetic_expression_equality : (4 * 12) - (4 + 12) = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1116_111693


namespace NUMINAMATH_CALUDE_skew_sufficient_not_necessary_for_non_intersecting_l1116_111627

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l₁ l₂ : Line3D) : Prop :=
  sorry

/-- Two lines intersect if they share a common point -/
def intersect (l₁ l₂ : Line3D) : Prop :=
  sorry

/-- Main theorem: Skew lines are sufficient but not necessary for non-intersecting lines -/
theorem skew_sufficient_not_necessary_for_non_intersecting :
  (∀ l₁ l₂ : Line3D, are_skew l₁ l₂ → ¬(intersect l₁ l₂)) ∧
  (∃ l₁ l₂ : Line3D, ¬(intersect l₁ l₂) ∧ ¬(are_skew l₁ l₂)) :=
by sorry

end NUMINAMATH_CALUDE_skew_sufficient_not_necessary_for_non_intersecting_l1116_111627


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1116_111618

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 3) = 9 → x = 84 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1116_111618


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1116_111619

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, S n = n * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1)
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Theorem: If S_10 = S_20 in an arithmetic sequence, then S_30 = 0 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (h : seq.S 10 = seq.S 20) :
  seq.S 30 = 0 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1116_111619


namespace NUMINAMATH_CALUDE_magical_card_stack_l1116_111609

/-- 
Given a stack of 2n cards numbered 1 to 2n, with the top n cards forming pile A 
and the rest forming pile B, prove that when restacked by alternating from 
piles B and A, the total number of cards where card 161 retains its original 
position is 482.
-/
theorem magical_card_stack (n : ℕ) : 
  (∃ (total : ℕ), 
    total = 2 * n ∧ 
    161 ≤ n ∧ 
    (∀ (k : ℕ), k ≤ total → k = 161 → (k - 1) / 2 = (n - 161))) → 
  2 * n = 482 :=
by sorry

end NUMINAMATH_CALUDE_magical_card_stack_l1116_111609


namespace NUMINAMATH_CALUDE_complex_distance_sum_l1116_111632

/-- Given a complex number z satisfying |z - 3 - 2i| = 7, 
    prove that |z - 2 + i|^2 + |z - 11 - 5i|^2 = 554 -/
theorem complex_distance_sum (z : ℂ) (h : Complex.abs (z - (3 + 2*I)) = 7) : 
  (Complex.abs (z - (2 - I)))^2 + (Complex.abs (z - (11 + 5*I)))^2 = 554 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_sum_l1116_111632


namespace NUMINAMATH_CALUDE_intersection_A_complementB_l1116_111649

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Define the complement of B with respect to U
def complementB : Set ℝ := U \ B

-- Theorem statement
theorem intersection_A_complementB : A ∩ complementB = {x | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complementB_l1116_111649


namespace NUMINAMATH_CALUDE_typist_salary_calculation_l1116_111690

theorem typist_salary_calculation (original_salary : ℝ) (raise_percentage : ℝ) (reduction_percentage : ℝ) : 
  original_salary = 2000 ∧ 
  raise_percentage = 10 ∧ 
  reduction_percentage = 5 → 
  original_salary * (1 + raise_percentage / 100) * (1 - reduction_percentage / 100) = 2090 :=
by sorry

end NUMINAMATH_CALUDE_typist_salary_calculation_l1116_111690


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1116_111634

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 3 / 2) : 
  let e := Real.sqrt (1 + (b / a) ^ 2)
  e = Real.sqrt 13 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1116_111634


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_factorization_4_l1116_111614

-- 1. 2x^2 + 2x = 2x(x+1)
theorem factorization_1 (x : ℝ) : 2*x^2 + 2*x = 2*x*(x+1) := by sorry

-- 2. a^3 - a = a(a+1)(a-1)
theorem factorization_2 (a : ℝ) : a^3 - a = a*(a+1)*(a-1) := by sorry

-- 3. (x-y)^2 - 4(x-y) + 4 = (x-y-2)^2
theorem factorization_3 (x y : ℝ) : (x-y)^2 - 4*(x-y) + 4 = (x-y-2)^2 := by sorry

-- 4. x^2 + 2xy + y^2 - 9 = (x+y+3)(x+y-3)
theorem factorization_4 (x y : ℝ) : x^2 + 2*x*y + y^2 - 9 = (x+y+3)*(x+y-3) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_factorization_4_l1116_111614


namespace NUMINAMATH_CALUDE_negation_equivalence_l1116_111605

theorem negation_equivalence :
  (¬ (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1116_111605


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1116_111695

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def B : Set ℝ := {x | 2*x - 3 > 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1116_111695


namespace NUMINAMATH_CALUDE_faye_pencils_l1116_111691

/-- The number of pencils Faye has in all sets -/
def total_pencils (rows_per_set : ℕ) (pencils_per_row : ℕ) (num_sets : ℕ) : ℕ :=
  rows_per_set * pencils_per_row * num_sets

/-- Theorem stating the total number of pencils Faye has -/
theorem faye_pencils :
  total_pencils 14 11 3 = 462 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencils_l1116_111691


namespace NUMINAMATH_CALUDE_probability_four_green_marbles_l1116_111611

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 8

/-- The number of purple marbles in the bag -/
def purple_marbles : ℕ := 4

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of draws -/
def num_draws : ℕ := 8

/-- The number of green marbles we want to draw -/
def target_green : ℕ := 4

/-- The probability of drawing exactly 'target_green' green marbles in 'num_draws' draws -/
def probability_exact_green : ℚ :=
  (Nat.choose num_draws target_green : ℚ) *
  (green_marbles ^ target_green * purple_marbles ^ (num_draws - target_green)) /
  (total_marbles ^ num_draws)

theorem probability_four_green_marbles :
  probability_exact_green = 1120 / 6561 :=
sorry

end NUMINAMATH_CALUDE_probability_four_green_marbles_l1116_111611


namespace NUMINAMATH_CALUDE_log_function_range_l1116_111655

theorem log_function_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ 2 → |Real.log x / Real.log a| > 1) ↔ 
  (a > 1/2 ∧ a < 1) ∨ (a > 1 ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_log_function_range_l1116_111655


namespace NUMINAMATH_CALUDE_helen_raisin_cookies_l1116_111638

/-- The number of raisin cookies Helen baked yesterday -/
def raisin_cookies_yesterday : ℕ := sorry

/-- The number of chocolate chip cookies Helen baked yesterday -/
def choc_chip_cookies_yesterday : ℕ := 519

/-- The number of raisin cookies Helen baked today -/
def raisin_cookies_today : ℕ := 280

/-- The number of chocolate chip cookies Helen baked today -/
def choc_chip_cookies_today : ℕ := 359

/-- Helen baked 20 more raisin cookies yesterday compared to today -/
axiom raisin_cookies_difference : raisin_cookies_yesterday = raisin_cookies_today + 20

theorem helen_raisin_cookies : raisin_cookies_yesterday = 300 := by sorry

end NUMINAMATH_CALUDE_helen_raisin_cookies_l1116_111638


namespace NUMINAMATH_CALUDE_class_division_l1116_111677

theorem class_division (total_students : ℕ) (x : ℕ) : 
  (total_students = 8 * x + 2) ∧ (total_students = 9 * x - 4) → x = 6 := by
sorry

end NUMINAMATH_CALUDE_class_division_l1116_111677


namespace NUMINAMATH_CALUDE_determinant_special_matrix_l1116_111645

theorem determinant_special_matrix (a y : ℝ) : 
  Matrix.det !![a, y, y; y, a, y; y, y, a] = a^3 - 2*a*y^2 + 2*y^3 := by
  sorry

end NUMINAMATH_CALUDE_determinant_special_matrix_l1116_111645


namespace NUMINAMATH_CALUDE_max_individual_award_l1116_111644

theorem max_individual_award 
  (total_prize : ℕ) 
  (num_winners : ℕ) 
  (min_award : ℕ) 
  (h1 : total_prize = 2500)
  (h2 : num_winners = 25)
  (h3 : min_award = 50)
  (h4 : (3 : ℚ) / 5 * total_prize = (2 : ℚ) / 5 * num_winners * max_award)
  : ∃ max_award : ℕ, max_award = 1300 := by
  sorry

end NUMINAMATH_CALUDE_max_individual_award_l1116_111644


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l1116_111685

/-- Given two points A(0,0) and B(x,-3) where the slope of AB is 4/5, 
    the sum of B's coordinates is -27/4 -/
theorem point_coordinate_sum (x : ℚ) : 
  let A : ℚ × ℚ := (0, 0)
  let B : ℚ × ℚ := (x, -3)
  let slope : ℚ := (B.2 - A.2) / (B.1 - A.1)
  slope = 4/5 → x + B.2 = -27/4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l1116_111685


namespace NUMINAMATH_CALUDE_garden_area_difference_l1116_111692

-- Define the dimensions of the gardens
def karl_length : ℝ := 20
def karl_width : ℝ := 45
def makenna_length : ℝ := 25
def makenna_width : ℝ := 40

-- Define the areas of the gardens
def karl_area : ℝ := karl_length * karl_width
def makenna_area : ℝ := makenna_length * makenna_width

-- Theorem to prove
theorem garden_area_difference : makenna_area - karl_area = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_difference_l1116_111692


namespace NUMINAMATH_CALUDE_pricing_scenario_l1116_111602

/-- The number of articles in a pricing scenario -/
def num_articles : ℕ := 50

/-- The number of articles used for selling price comparison -/
def comparison_articles : ℕ := 45

/-- The gain percentage as a rational number -/
def gain_percentage : ℚ := 1 / 9

theorem pricing_scenario :
  (∀ (cost_price selling_price : ℚ),
    cost_price * num_articles = selling_price * comparison_articles →
    selling_price = cost_price * (1 + gain_percentage)) →
  num_articles = 50 :=
sorry

end NUMINAMATH_CALUDE_pricing_scenario_l1116_111602


namespace NUMINAMATH_CALUDE_point_on_line_l1116_111675

/-- The line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_through_points (x₁ y₁ x₂ y₂ : ℚ) (x y : ℚ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

/-- The theorem stating that (-3/7, 8) lies on the line through (3, 0) and (0, 7) -/
theorem point_on_line : line_through_points 3 0 0 7 (-3/7) 8 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1116_111675


namespace NUMINAMATH_CALUDE_quiche_volume_l1116_111667

/-- Calculates the total volume of a quiche given the ingredients' volumes and spinach reduction factor. -/
theorem quiche_volume 
  (raw_spinach : ℝ) 
  (reduction_factor : ℝ) 
  (cream_cheese : ℝ) 
  (eggs : ℝ) 
  (h1 : 0 < reduction_factor) 
  (h2 : reduction_factor < 1) :
  raw_spinach * reduction_factor + cream_cheese + eggs = 
  (raw_spinach * reduction_factor + cream_cheese + eggs) := by
  sorry

end NUMINAMATH_CALUDE_quiche_volume_l1116_111667


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l1116_111641

/-- Given a point P with coordinates (2, -1) in the standard coordinate system
    and (b-1, a+3) in another coordinate system with the same origin,
    prove that a + b = -1 -/
theorem point_coordinate_sum (a b : ℝ) 
  (h1 : b - 1 = 2) 
  (h2 : a + 3 = -1) : 
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l1116_111641


namespace NUMINAMATH_CALUDE_spinner_final_direction_l1116_111604

/-- Represents the four cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a spinner with its current direction --/
structure Spinner :=
  (direction : Direction)

/-- Calculates the new direction after a given number of quarter turns clockwise --/
def new_direction_after_quarter_turns (initial : Direction) (quarter_turns : Int) : Direction :=
  sorry

/-- Converts revolutions to quarter turns --/
def revolutions_to_quarter_turns (revolutions : Rat) : Int :=
  sorry

theorem spinner_final_direction :
  let initial_spinner := Spinner.mk Direction.South
  let clockwise_turns := revolutions_to_quarter_turns (7/2)
  let counterclockwise_turns := revolutions_to_quarter_turns (9/4)
  let net_turns := clockwise_turns - counterclockwise_turns
  let final_direction := new_direction_after_quarter_turns initial_spinner.direction net_turns
  final_direction = Direction.West := by sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l1116_111604


namespace NUMINAMATH_CALUDE_min_xy_value_least_xy_value_l1116_111668

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  ∀ (a b : ℕ+), ((1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6) → (x : ℕ) * y ≤ (a : ℕ) * b :=
by
  sorry

theorem least_xy_value :
  ∃ (x y : ℕ+), ((1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) ∧ (x : ℕ) * y = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_least_xy_value_l1116_111668


namespace NUMINAMATH_CALUDE_annie_cookies_l1116_111651

/-- The number of cookies Annie ate over three days -/
def total_cookies (monday tuesday wednesday : ℕ) : ℕ :=
  monday + tuesday + wednesday

/-- Theorem: Annie ate 29 cookies over three days -/
theorem annie_cookies : ∃ (monday tuesday wednesday : ℕ),
  monday = 5 ∧
  tuesday = 2 * monday ∧
  wednesday = tuesday + (tuesday * 2 / 5) ∧
  total_cookies monday tuesday wednesday = 29 := by
  sorry

end NUMINAMATH_CALUDE_annie_cookies_l1116_111651


namespace NUMINAMATH_CALUDE_unique_triangle_solution_l1116_111616

/-- Represents the assignment of numbers to letters in the triangle puzzle -/
structure TriangleAssignment where
  A : Nat
  B : Nat
  C : Nat
  D : Nat
  E : Nat
  F : Nat

/-- The set of numbers used in the puzzle -/
def puzzleNumbers : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- Checks if the given assignment satisfies all conditions of the puzzle -/
def isValidAssignment (assignment : TriangleAssignment) : Prop :=
  assignment.A ∈ puzzleNumbers ∧
  assignment.B ∈ puzzleNumbers ∧
  assignment.C ∈ puzzleNumbers ∧
  assignment.D ∈ puzzleNumbers ∧
  assignment.E ∈ puzzleNumbers ∧
  assignment.F ∈ puzzleNumbers ∧
  assignment.D + assignment.E + assignment.B = 14 ∧
  assignment.A + assignment.C = 3 ∧
  assignment.A ≠ assignment.B ∧ assignment.A ≠ assignment.C ∧ assignment.A ≠ assignment.D ∧
  assignment.A ≠ assignment.E ∧ assignment.A ≠ assignment.F ∧
  assignment.B ≠ assignment.C ∧ assignment.B ≠ assignment.D ∧ assignment.B ≠ assignment.E ∧
  assignment.B ≠ assignment.F ∧
  assignment.C ≠ assignment.D ∧ assignment.C ≠ assignment.E ∧ assignment.C ≠ assignment.F ∧
  assignment.D ≠ assignment.E ∧ assignment.D ≠ assignment.F ∧
  assignment.E ≠ assignment.F

/-- The unique solution to the triangle puzzle -/
def triangleSolution : TriangleAssignment :=
  { A := 1, B := 3, C := 2, D := 5, E := 6, F := 4 }

/-- Theorem stating that the triangleSolution is the only valid assignment -/
theorem unique_triangle_solution :
  ∀ assignment : TriangleAssignment,
    isValidAssignment assignment → assignment = triangleSolution := by
  sorry

end NUMINAMATH_CALUDE_unique_triangle_solution_l1116_111616


namespace NUMINAMATH_CALUDE_total_vehicles_l1116_111630

theorem total_vehicles (motorcycles bicycles : ℕ) 
  (h1 : motorcycles = 2) 
  (h2 : bicycles = 5) : 
  motorcycles + bicycles = 7 :=
by sorry

end NUMINAMATH_CALUDE_total_vehicles_l1116_111630


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1116_111624

/-- The perimeter of a regular polygon with side length 8 and exterior angle 72 degrees is 40 units. -/
theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) : 
  side_length = 8 → 
  exterior_angle = 72 → 
  (360 / exterior_angle) * side_length = 40 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1116_111624


namespace NUMINAMATH_CALUDE_vet_donation_amount_l1116_111673

/-- Calculates the amount donated by the vet to an animal shelter during a pet adoption event. -/
theorem vet_donation_amount (dog_fee cat_fee : ℕ) (dog_adoptions cat_adoptions : ℕ) (donation_fraction : ℚ) : 
  dog_fee = 15 →
  cat_fee = 13 →
  dog_adoptions = 8 →
  cat_adoptions = 3 →
  donation_fraction = 1/3 →
  (dog_fee * dog_adoptions + cat_fee * cat_adoptions) * donation_fraction = 53 := by
  sorry

end NUMINAMATH_CALUDE_vet_donation_amount_l1116_111673


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l1116_111608

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance_theorem (A : ℝ × ℝ) :
  parabola A.1 A.2 →  -- A lies on the parabola
  (A.1 - focus.1)^2 + (A.2 - focus.2)^2 = (B.1 - focus.1)^2 + (B.2 - focus.2)^2 →  -- |AF| = |BF|
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8 :=  -- |AB| = 2√2
by sorry

end NUMINAMATH_CALUDE_parabola_distance_theorem_l1116_111608


namespace NUMINAMATH_CALUDE_distance_between_trees_l1116_111683

/-- Given a yard of length 1565 metres with 356 trees planted at equal distances
    (including one at each end), the distance between two consecutive trees
    is equal to 1565 / (356 - 1) metres. -/
theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) 
    (h1 : yard_length = 1565)
    (h2 : num_trees = 356) :
    (yard_length : ℚ) / (num_trees - 1) = 1565 / 355 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_trees_l1116_111683


namespace NUMINAMATH_CALUDE_intersection_count_theorem_m_value_theorem_l1116_111607

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 2

-- Define the line l
def l (x y : ℝ) : Prop := y = x ∧ x ≥ 0

-- Define the number of intersection points
def intersection_count : ℕ := 1

-- Define the equation for C₂ when θ = π/4
def C₂_equation (ρ m : ℝ) : Prop := ρ^2 - 3 * Real.sqrt 2 * ρ + 2 * m = 0

-- Theorem for the number of intersection points
theorem intersection_count_theorem :
  ∃! (x y : ℝ), C₁ x y ∧ l x y :=
sorry

-- Theorem for the value of m
theorem m_value_theorem (ρ₁ ρ₂ m : ℝ) :
  C₂_equation ρ₁ m ∧ C₂_equation ρ₂ m ∧ ρ₂ = 2 * ρ₁ → m = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_count_theorem_m_value_theorem_l1116_111607


namespace NUMINAMATH_CALUDE_xyz_value_l1116_111621

theorem xyz_value (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 195)
  (h2 : y * (z + x) = 204)
  (h3 : z * (x + y) = 213) : 
  x * y * z = 1029 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1116_111621


namespace NUMINAMATH_CALUDE_log2_derivative_l1116_111678

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
sorry

end NUMINAMATH_CALUDE_log2_derivative_l1116_111678


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1116_111688

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides. -/
def pentagon_sides : ℕ := 5

/-- The sum of the interior angles of a pentagon is 540 degrees. -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon_sides = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1116_111688


namespace NUMINAMATH_CALUDE_solution_set_of_decreasing_function_l1116_111657

open Set

def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem solution_set_of_decreasing_function
  (f : ℝ → ℝ)
  (h_decreasing : DecreasingFunction f)
  (h_f_0 : f 0 = 1)
  (h_f_1 : f 1 = 0) :
  {x : ℝ | f x > 0} = Iio 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_decreasing_function_l1116_111657


namespace NUMINAMATH_CALUDE_distance_to_x_axis_distance_M_to_x_axis_l1116_111646

/-- The distance from a point to the x-axis in a Cartesian coordinate system
    is equal to the absolute value of its y-coordinate. -/
theorem distance_to_x_axis (x y : ℝ) :
  let M : ℝ × ℝ := (x, y)
  abs y = dist M (x, 0) :=
by sorry

/-- The distance from the point M(-9,12) to the x-axis is 12. -/
theorem distance_M_to_x_axis :
  let M : ℝ × ℝ := (-9, 12)
  dist M (-9, 0) = 12 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_distance_M_to_x_axis_l1116_111646


namespace NUMINAMATH_CALUDE_shooting_sequences_l1116_111666

-- Define the number of targets in each column
def targets_A : ℕ := 3
def targets_B : ℕ := 2
def targets_C : ℕ := 3

-- Define the total number of targets
def total_targets : ℕ := targets_A + targets_B + targets_C

-- Theorem statement
theorem shooting_sequences :
  (total_targets.factorial) / (targets_A.factorial * targets_B.factorial * targets_C.factorial) = 560 := by
  sorry

end NUMINAMATH_CALUDE_shooting_sequences_l1116_111666


namespace NUMINAMATH_CALUDE_river_road_cars_l1116_111653

/-- Proves that the number of cars on River Road is 60 -/
theorem river_road_cars :
  ∀ (buses cars motorcycles : ℕ),
    (buses : ℚ) / cars = 1 / 3 →
    cars = buses + 40 →
    buses + cars + motorcycles = 720 →
    cars = 60 := by
  sorry

end NUMINAMATH_CALUDE_river_road_cars_l1116_111653


namespace NUMINAMATH_CALUDE_worksheets_graded_before_additional_l1116_111664

/-- The number of worksheets initially given to the teacher to grade. -/
def initial_worksheets : ℕ := 6

/-- The number of additional worksheets turned in later. -/
def additional_worksheets : ℕ := 18

/-- The total number of worksheets to grade after the additional ones were turned in. -/
def total_worksheets : ℕ := 20

/-- The number of worksheets graded before the additional ones were turned in. -/
def graded_worksheets : ℕ := 4

theorem worksheets_graded_before_additional :
  initial_worksheets - graded_worksheets + additional_worksheets = total_worksheets :=
sorry

end NUMINAMATH_CALUDE_worksheets_graded_before_additional_l1116_111664


namespace NUMINAMATH_CALUDE_complex_power_eight_l1116_111642

theorem complex_power_eight (z : ℂ) : z = (-Real.sqrt 3 + I) / 2 → z^8 = -1/2 - (Real.sqrt 3 / 2) * I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_eight_l1116_111642


namespace NUMINAMATH_CALUDE_investment_theorem_l1116_111654

/-- Calculates the total investment with interest after one year -/
def total_investment_with_interest (total_investment : ℝ) (amount_at_3_percent : ℝ) : ℝ :=
  let amount_at_5_percent := total_investment - amount_at_3_percent
  let interest_at_3_percent := amount_at_3_percent * 0.03
  let interest_at_5_percent := amount_at_5_percent * 0.05
  total_investment + interest_at_3_percent + interest_at_5_percent

/-- Theorem stating that the total investment with interest is $1,046 -/
theorem investment_theorem :
  total_investment_with_interest 1000 199.99999999999983 = 1046 := by
  sorry

end NUMINAMATH_CALUDE_investment_theorem_l1116_111654


namespace NUMINAMATH_CALUDE_counting_unit_relations_l1116_111612

/-- Represents a counting unit -/
inductive CountingUnit
| TenThousand
| HundredThousand
| Million
| TenMillion
| HundredMillion

/-- The progression rate between adjacent counting units -/
def progression_rate : ℕ := 10

/-- The value of a counting unit in terms of the base unit (assumed to be one) -/
def value : CountingUnit → ℕ
| CountingUnit.TenThousand => 10000
| CountingUnit.HundredThousand => 100000
| CountingUnit.Million => 1000000
| CountingUnit.TenMillion => 10000000
| CountingUnit.HundredMillion => 100000000

theorem counting_unit_relations :
  (value CountingUnit.HundredMillion = 10 * value CountingUnit.TenMillion) ∧
  (value CountingUnit.Million = 100 * value CountingUnit.TenThousand) :=
by sorry

end NUMINAMATH_CALUDE_counting_unit_relations_l1116_111612


namespace NUMINAMATH_CALUDE_future_value_proof_l1116_111694

/-- Calculates the future value of an investment with compound interest. -/
def future_value (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that given the specified conditions, the future value is $3600. -/
theorem future_value_proof :
  let principal : ℝ := 2500
  let rate : ℝ := 0.20
  let time : ℕ := 2
  future_value principal rate time = 3600 := by
sorry

#eval future_value 2500 0.20 2

end NUMINAMATH_CALUDE_future_value_proof_l1116_111694


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l1116_111672

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-1)^2) + Real.sqrt ((x-5)^2 + (y+3)^2) = 10

/-- The first focus of the ellipse -/
def focus1 : ℝ × ℝ := (0, 1)

/-- The second focus of the ellipse -/
def focus2 : ℝ × ℝ := (5, -3)

/-- The constant sum of distances from any point on the ellipse to the foci -/
def constant_sum : ℝ := 10

/-- Theorem stating that the given equation describes an ellipse -/
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), conic_equation x y ↔
    (x - (focus1.1 + focus2.1) / 2)^2 / a^2 +
    (y - (focus1.2 + focus2.2) / 2)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l1116_111672


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1116_111601

/-- Given an ellipse and a hyperbola with shared foci, prove the equation of the hyperbola -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    -- Ellipse equation
    (x^2 + 4*y^2 = 64) ∧
    -- Hyperbola shares foci with ellipse
    (a^2 + b^2 = 48) ∧
    -- Asymptote equation
    (x - Real.sqrt 3 * y = 0)) →
  -- Hyperbola equation
  x^2/36 - y^2/12 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1116_111601


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l1116_111671

/-- Proves that the cost price of one meter of cloth is 85 rupees -/
theorem cost_price_per_meter
  (total_length : ℕ)
  (total_selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 85)
  (h2 : total_selling_price = 8500)
  (h3 : profit_per_meter = 15) :
  (total_selling_price - total_length * profit_per_meter) / total_length = 85 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l1116_111671


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l1116_111665

/-- Proves that the average mark of excluded students is 40 given the conditions of the problem -/
theorem excluded_students_average_mark
  (total_students : ℕ)
  (total_average : ℚ)
  (remaining_average : ℚ)
  (excluded_count : ℕ)
  (h_total_students : total_students = 33)
  (h_total_average : total_average = 90)
  (h_remaining_average : remaining_average = 95)
  (h_excluded_count : excluded_count = 3) :
  let remaining_count := total_students - excluded_count
  let total_marks := total_students * total_average
  let remaining_marks := remaining_count * remaining_average
  let excluded_marks := total_marks - remaining_marks
  excluded_marks / excluded_count = 40 := by
  sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l1116_111665


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l1116_111617

theorem circle_diameter_ratio (C D : Real) (h1 : D = 20) 
  (h2 : C > 0 ∧ C < D) (h3 : (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 5) : 
  C = 10 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l1116_111617


namespace NUMINAMATH_CALUDE_shekar_social_studies_score_l1116_111663

theorem shekar_social_studies_score 
  (math_score : ℕ) 
  (science_score : ℕ) 
  (english_score : ℕ) 
  (biology_score : ℕ) 
  (average_score : ℕ) 
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 67)
  (h4 : biology_score = 95)
  (h5 : average_score = 77)
  (h6 : (math_score + science_score + english_score + biology_score + social_studies_score) / 5 = average_score) :
  social_studies_score = 82 :=
by
  sorry

#check shekar_social_studies_score

end NUMINAMATH_CALUDE_shekar_social_studies_score_l1116_111663


namespace NUMINAMATH_CALUDE_complex_number_problem_l1116_111633

theorem complex_number_problem (z : ℂ) : 
  Complex.abs z = 1 ∧ 
  (∃ (y : ℝ), (3 + 4*I) * z = y * I) → 
  z = Complex.mk (-4/5) (-3/5) ∨ 
  z = Complex.mk (4/5) (3/5) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1116_111633


namespace NUMINAMATH_CALUDE_factory_employees_count_l1116_111682

/-- Represents the profit calculation for a t-shirt factory --/
def factory_profit (num_employees : ℕ) : ℚ :=
  let shirts_per_employee := 20
  let shirt_price := 35
  let hourly_wage := 12
  let per_shirt_bonus := 5
  let hours_per_shift := 8
  let nonemployee_expenses := 1000
  let total_shirts := num_employees * shirts_per_employee
  let revenue := total_shirts * shirt_price
  let employee_pay := num_employees * (hourly_wage * hours_per_shift + per_shirt_bonus * shirts_per_employee)
  revenue - employee_pay - nonemployee_expenses

/-- The number of employees that results in the given profit --/
theorem factory_employees_count : 
  ∃ (n : ℕ), factory_profit n = 9080 ∧ n = 20 := by
  sorry


end NUMINAMATH_CALUDE_factory_employees_count_l1116_111682


namespace NUMINAMATH_CALUDE_prime_factors_equation_l1116_111610

/-- Given an expression (4^x) * (7^5) * (11^2) with 29 prime factors, prove x = 11 -/
theorem prime_factors_equation (x : ℕ) : 
  (2 * x + 5 + 2 = 29) → x = 11 := by sorry

end NUMINAMATH_CALUDE_prime_factors_equation_l1116_111610


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l1116_111687

theorem sum_of_five_consecutive_even_integers (m : ℤ) : 
  (m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l1116_111687


namespace NUMINAMATH_CALUDE_round_trip_with_car_percentage_l1116_111628

/-- The percentage of passengers with round-trip tickets who did not take their cars -/
def no_car_percentage : ℝ := 60

/-- The percentage of all passengers who held round-trip tickets -/
def round_trip_percentage : ℝ := 62.5

/-- The theorem to prove -/
theorem round_trip_with_car_percentage :
  (100 - no_car_percentage) * round_trip_percentage / 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_with_car_percentage_l1116_111628


namespace NUMINAMATH_CALUDE_necessary_condition_for_greater_than_five_l1116_111626

theorem necessary_condition_for_greater_than_five (x : ℝ) :
  x > 5 → x > 3 := by sorry

end NUMINAMATH_CALUDE_necessary_condition_for_greater_than_five_l1116_111626


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l1116_111686

theorem system_of_equations_solutions :
  -- First system of equations
  (∃ x y : ℝ, 2 * x - 3 * y = -2 ∧ 5 * x + 3 * y = 37 ∧ x = 5 ∧ y = 4) ∧
  -- Second system of equations
  (∃ x y : ℝ, 3 * x + 2 * y = 5 ∧ 4 * x - y = 3 ∧ x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l1116_111686


namespace NUMINAMATH_CALUDE_harry_last_mile_water_consumption_l1116_111684

/-- Represents the hike scenario --/
structure HikeScenario where
  totalDistance : ℝ
  initialWater : ℝ
  finalWater : ℝ
  timeTaken : ℝ
  leakRate : ℝ
  waterConsumptionFirstThreeMiles : ℝ

/-- Calculates the water consumed in the last mile of the hike --/
def waterConsumedLastMile (h : HikeScenario) : ℝ :=
  h.initialWater - h.finalWater - (h.leakRate * h.timeTaken) - (h.waterConsumptionFirstThreeMiles * (h.totalDistance - 1))

/-- Theorem stating that Harry drank 3 cups of water in the last mile --/
theorem harry_last_mile_water_consumption :
  let h : HikeScenario := {
    totalDistance := 4
    initialWater := 10
    finalWater := 2
    timeTaken := 2
    leakRate := 1
    waterConsumptionFirstThreeMiles := 1
  }
  waterConsumedLastMile h = 3 := by
  sorry


end NUMINAMATH_CALUDE_harry_last_mile_water_consumption_l1116_111684


namespace NUMINAMATH_CALUDE_train_length_calculation_l1116_111698

/-- The length of a train given its speed and time to cross a pole. -/
def train_length (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A train with speed 53.99999999999999 m/s that crosses a pole in 20 seconds has a length of 1080 meters. -/
theorem train_length_calculation :
  let speed : ℝ := 53.99999999999999
  let time : ℝ := 20
  train_length speed time = 1080 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1116_111698


namespace NUMINAMATH_CALUDE_reflection_of_line_over_x_axis_l1116_111670

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Reflects a line over the x-axis --/
def reflect_over_x_axis (l : Line) : Line :=
  { slope := -l.slope, intercept := -l.intercept }

theorem reflection_of_line_over_x_axis :
  let original_line : Line := { slope := 2, intercept := 3 }
  let reflected_line : Line := reflect_over_x_axis original_line
  reflected_line = { slope := -2, intercept := -3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_line_over_x_axis_l1116_111670


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1116_111620

/-- A geometric sequence with specific terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a3 : a 3 = 3)
  (h_a10 : a 10 = 384) :
  ∃ q : ℝ, ∀ n : ℕ, a n = 3 * q^(n - 3) ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1116_111620


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1116_111600

theorem rationalize_denominator :
  18 / (Real.sqrt 36 + Real.sqrt 2) = 54 / 17 - 9 * Real.sqrt 2 / 17 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1116_111600


namespace NUMINAMATH_CALUDE_square_difference_of_solutions_l1116_111659

theorem square_difference_of_solutions (α β : ℝ) : 
  α ≠ β ∧ α^2 = 3*α + 1 ∧ β^2 = 3*β + 1 → (α - β)^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_solutions_l1116_111659


namespace NUMINAMATH_CALUDE_ball_placement_count_ball_placement_proof_l1116_111643

theorem ball_placement_count : ℕ :=
  let n_balls : ℕ := 5
  let n_boxes : ℕ := 4
  let ways_to_divide : ℕ := Nat.choose n_balls (n_balls - n_boxes + 1)
  let ways_to_arrange : ℕ := Nat.factorial n_boxes
  ways_to_divide * ways_to_arrange

theorem ball_placement_proof :
  ball_placement_count = 240 := by
  sorry

end NUMINAMATH_CALUDE_ball_placement_count_ball_placement_proof_l1116_111643


namespace NUMINAMATH_CALUDE_sphere_radii_problem_l1116_111652

theorem sphere_radii_problem (r₁ r₂ r₃ : ℝ) : 
  -- Three spheres touch each other externally
  2 * Real.sqrt (r₁ * r₂) = 2 ∧
  2 * Real.sqrt (r₁ * r₃) = Real.sqrt 3 ∧
  2 * Real.sqrt (r₂ * r₃) = 1 ∧
  -- The spheres touch a plane at the vertices of a right triangle
  -- One leg of the triangle has length 1
  -- The angle opposite to the leg of length 1 is 30°
  -- (These conditions are implicitly satisfied by the equations above)
  -- The radii are positive
  r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0
  →
  -- The radii of the spheres are √3, 1/√3, and √3/4
  (r₁ = Real.sqrt 3 ∧ r₂ = 1 / Real.sqrt 3 ∧ r₃ = Real.sqrt 3 / 4) ∨
  (r₁ = Real.sqrt 3 ∧ r₂ = Real.sqrt 3 / 4 ∧ r₃ = 1 / Real.sqrt 3) ∨
  (r₁ = 1 / Real.sqrt 3 ∧ r₂ = Real.sqrt 3 ∧ r₃ = Real.sqrt 3 / 4) ∨
  (r₁ = 1 / Real.sqrt 3 ∧ r₂ = Real.sqrt 3 / 4 ∧ r₃ = Real.sqrt 3) ∨
  (r₁ = Real.sqrt 3 / 4 ∧ r₂ = Real.sqrt 3 ∧ r₃ = 1 / Real.sqrt 3) ∨
  (r₁ = Real.sqrt 3 / 4 ∧ r₂ = 1 / Real.sqrt 3 ∧ r₃ = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_radii_problem_l1116_111652


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l1116_111662

theorem least_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 56 = 3 ∧
  n % 78 = 3 ∧
  n % 9 = 0 ∧
  ∀ m : ℕ, m > 0 ∧ m % 56 = 3 ∧ m % 78 = 3 ∧ m % 9 = 0 → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l1116_111662


namespace NUMINAMATH_CALUDE_min_mobots_correct_l1116_111636

/-- Represents a lawn as a grid with dimensions m and n -/
structure Lawn where
  m : ℕ
  n : ℕ

/-- Represents a mobot that can mow a lawn -/
inductive Mobot
  | east  : Mobot  -- Mobot moving east
  | north : Mobot  -- Mobot moving north

/-- Function to calculate the minimum number of mobots required -/
def minMobotsRequired (lawn : Lawn) : ℕ := min lawn.m lawn.n

/-- Theorem stating that minMobotsRequired gives the correct minimum number of mobots -/
theorem min_mobots_correct (lawn : Lawn) :
  ∀ (mobots : List Mobot), (∀ row col, row < lawn.m ∧ col < lawn.n → 
    ∃ mobot ∈ mobots, (mobot = Mobot.east ∧ ∃ r, r ≤ row) ∨ 
                       (mobot = Mobot.north ∧ ∃ c, c ≤ col)) →
  mobots.length ≥ minMobotsRequired lawn :=
sorry

end NUMINAMATH_CALUDE_min_mobots_correct_l1116_111636


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1116_111640

/-- The amount of pie Erik ate -/
def erik_pie : ℚ := 0.6666666666666666

/-- The amount of pie Frank ate -/
def frank_pie : ℚ := 0.3333333333333333

/-- The difference between Erik's and Frank's pie consumption -/
def pie_difference : ℚ := erik_pie - frank_pie

theorem pie_eating_contest :
  pie_difference = 0.3333333333333333 := by sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1116_111640


namespace NUMINAMATH_CALUDE_final_week_hours_l1116_111674

def hours_worked : List ℕ := [14, 10, 13, 9, 12, 11]
def total_weeks : ℕ := 7
def required_average : ℕ := 12

theorem final_week_hours :
  ∃ (x : ℕ), (List.sum hours_worked + x) / total_weeks = required_average :=
by sorry

end NUMINAMATH_CALUDE_final_week_hours_l1116_111674


namespace NUMINAMATH_CALUDE_notebook_cost_l1116_111625

theorem notebook_cost (total_students : ℕ) (total_cost : ℕ) 
  (h1 : total_students = 36)
  (h2 : ∃ (buyers : ℕ) (notebooks_per_student : ℕ) (cost_per_notebook : ℕ),
    buyers > total_students / 2 ∧
    notebooks_per_student > 1 ∧
    cost_per_notebook > notebooks_per_student ∧
    buyers * notebooks_per_student * cost_per_notebook = total_cost)
  (h3 : total_cost = 2310) :
  ∃ (cost_per_notebook : ℕ), cost_per_notebook = 11 ∧
    ∃ (buyers : ℕ) (notebooks_per_student : ℕ),
      buyers > total_students / 2 ∧
      notebooks_per_student > 1 ∧
      cost_per_notebook > notebooks_per_student ∧
      buyers * notebooks_per_student * cost_per_notebook = total_cost :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l1116_111625
