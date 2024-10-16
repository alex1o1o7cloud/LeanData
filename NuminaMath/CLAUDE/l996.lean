import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l996_99677

theorem system_solution : 
  ∃! (x y z : ℝ), 
    x * (y + z) * (x + y + z) = 1170 ∧ 
    y * (z + x) * (x + y + z) = 1008 ∧ 
    z * (x + y) * (x + y + z) = 1458 ∧ 
    x = 5 ∧ y = 4 ∧ z = 9 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l996_99677


namespace NUMINAMATH_CALUDE_incorrect_expression_l996_99654

/-- A repeating decimal with non-repeating part X and repeating part Y -/
structure RepeatingDecimal where
  X : ℕ  -- non-repeating part
  Y : ℕ  -- repeating part
  t : ℕ  -- number of digits in X
  u : ℕ  -- number of digits in Y

/-- The value of a repeating decimal -/
def value (E : RepeatingDecimal) : ℚ :=
  sorry

/-- The statement that the expression is incorrect -/
theorem incorrect_expression (E : RepeatingDecimal) :
  ¬(10^E.t * (10^E.u - 1) * value E = E.Y * (E.X - 10)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_expression_l996_99654


namespace NUMINAMATH_CALUDE_cubic_decreasing_iff_a_leq_neg_three_l996_99623

/-- A cubic function f(x) = a x^3 + 3 x^2 - x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1

/-- A function is decreasing on ℝ if for all x, y in ℝ, x < y implies f(x) > f(y) -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

theorem cubic_decreasing_iff_a_leq_neg_three (a : ℝ) :
  IsDecreasing (f a) ↔ a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_cubic_decreasing_iff_a_leq_neg_three_l996_99623


namespace NUMINAMATH_CALUDE_total_over_budget_l996_99631

def project_budget (project : Char) : ℕ :=
  match project with
  | 'A' => 150000
  | 'B' => 120000
  | 'C' => 80000
  | _ => 0

def allocation_count (project : Char) : ℕ :=
  match project with
  | 'A' => 10
  | 'B' => 6
  | 'C' => 18
  | _ => 0

def allocation_period (project : Char) : ℕ :=
  match project with
  | 'A' => 2
  | 'B' => 3
  | 'C' => 1
  | _ => 0

def actual_spent (project : Char) : ℕ :=
  match project with
  | 'A' => 98450
  | 'B' => 72230
  | 'C' => 43065
  | _ => 0

def months_passed : ℕ := 9

def expected_expenditure (project : Char) : ℚ :=
  (project_budget project : ℚ) / (allocation_count project : ℚ) *
  ((months_passed : ℚ) / (allocation_period project : ℚ)).floor

def project_difference (project : Char) : ℚ :=
  (actual_spent project : ℚ) - expected_expenditure project

theorem total_over_budget :
  (project_difference 'A' + project_difference 'B' + project_difference 'C') = 38745 := by
  sorry

end NUMINAMATH_CALUDE_total_over_budget_l996_99631


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_13_after_subtraction_l996_99607

theorem smallest_number_divisible_by_13_after_subtraction (N : ℕ) : 
  (∃ k : ℕ, N - 10 = 13 * k) → N ≥ 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_13_after_subtraction_l996_99607


namespace NUMINAMATH_CALUDE_sequence_periodicity_implies_zero_l996_99698

theorem sequence_periodicity_implies_zero (a b c d : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + b n)
  (h2 : ∀ n, b (n + 1) = b n + c n)
  (h3 : ∀ n, c (n + 1) = c n + d n)
  (h4 : ∀ n, d (n + 1) = d n + a n)
  (h5 : ∃ k m : ℕ, k ≥ 1 ∧ m ≥ 1 ∧ 
    a (k + m) = a m ∧ 
    b (k + m) = b m ∧ 
    c (k + m) = c m ∧ 
    d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_implies_zero_l996_99698


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_is_13_l996_99606

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (yellow : ℕ)
  (white : ℕ)
  (green : ℕ)

/-- Calculates the minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFourSameColor (machine : GumballMachine) : ℕ :=
  sorry

/-- Theorem stating the minimum number of gumballs needed for the specific machine -/
theorem min_gumballs_for_four_same_color_is_13 :
  let machine : GumballMachine := ⟨10, 6, 8, 9⟩
  minGumballsForFourSameColor machine = 13 :=
sorry

end NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_is_13_l996_99606


namespace NUMINAMATH_CALUDE_remainder_of_123456789012_div_210_l996_99600

theorem remainder_of_123456789012_div_210 :
  123456789012 % 210 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_123456789012_div_210_l996_99600


namespace NUMINAMATH_CALUDE_complex_fraction_power_2000_l996_99664

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_power_2000 : ((1 - i) / (1 + i)) ^ 2000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_2000_l996_99664


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l996_99669

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, 3)
  are_parallel a b → x = 6 := by
    sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l996_99669


namespace NUMINAMATH_CALUDE_fraction_problem_l996_99699

theorem fraction_problem (numerator denominator : ℤ) (x : ℤ) : 
  denominator = numerator - 4 →
  denominator = 5 →
  numerator + x = 3 * denominator →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l996_99699


namespace NUMINAMATH_CALUDE_water_level_unchanged_l996_99613

/-- Represents the water level in a pool over time -/
def WaterLevel : Type :=
  { level : ℕ // level ≤ 1000 }

/-- The initial amount of water in gallons -/
def initial_water : WaterLevel :=
  ⟨300, by sorry⟩

/-- The evaporation rate in gallons per day -/
def evaporation_rate : ℕ := 1

/-- The amount of water added every 5 days in gallons -/
def water_addition : ℕ := 5

/-- The number of days to calculate the water level for -/
def days : ℕ := 35

/-- Calculates the water level after a given number of days -/
def water_after_days (initial : WaterLevel) (days : ℕ) : WaterLevel :=
  ⟨initial.val, by sorry⟩

/-- Theorem stating that the water level after 35 days is equal to the initial water level -/
theorem water_level_unchanged (initial : WaterLevel) :
  water_after_days initial days = initial :=
by sorry

end NUMINAMATH_CALUDE_water_level_unchanged_l996_99613


namespace NUMINAMATH_CALUDE_smallest_percent_increase_l996_99679

def question_values : List ℕ := [100, 300, 600, 1000, 1500, 2500, 4000, 6500]

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def consecutive_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  List.zip l (List.tail l)

theorem smallest_percent_increase :
  let pairs := consecutive_pairs question_values
  let increases := List.map (fun (p : ℕ × ℕ) => percent_increase p.1 p.2) pairs
  List.argmin id increases = some 3 := by sorry

end NUMINAMATH_CALUDE_smallest_percent_increase_l996_99679


namespace NUMINAMATH_CALUDE_simplify_cube_root_exponent_sum_l996_99618

theorem simplify_cube_root_exponent_sum (a b c : ℝ) : 
  ∃ (k : ℝ) (x y z : ℕ), 
    (∀ t : ℝ, t > 0 → (k * a^x * b^y * c^z)^3 * t = 40 * a^6 * b^9 * c^14) ∧ 
    x + y + z = 7 :=
sorry

end NUMINAMATH_CALUDE_simplify_cube_root_exponent_sum_l996_99618


namespace NUMINAMATH_CALUDE_expression_is_perfect_square_l996_99653

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- Define the expression
def expression : ℕ := 3^4 * 4^6 * 7^4

-- Define the prime factorization of 4
axiom four_factorization : 4 = 2^2

-- Theorem to prove
theorem expression_is_perfect_square : is_perfect_square expression := by
  sorry

end NUMINAMATH_CALUDE_expression_is_perfect_square_l996_99653


namespace NUMINAMATH_CALUDE_oliver_final_balance_l996_99655

def oliver_money_problem (initial_amount : ℝ) (allowance_savings : ℝ) (chore_earnings : ℝ)
  (frisbee_cost : ℝ) (puzzle_cost : ℝ) (sticker_cost : ℝ)
  (movie_ticket_price : ℝ) (movie_discount_percent : ℝ)
  (snack_price : ℝ) (snack_coupon : ℝ)
  (birthday_gift : ℝ) : Prop :=
  let total_expenses := frisbee_cost + puzzle_cost + sticker_cost
  let discounted_movie_price := movie_ticket_price * (1 - movie_discount_percent / 100)
  let snack_cost := snack_price - snack_coupon
  let final_balance := initial_amount + allowance_savings + chore_earnings - 
                       total_expenses - discounted_movie_price - snack_cost + birthday_gift
  final_balance = 9

theorem oliver_final_balance :
  oliver_money_problem 9 5 6 4 3 2 10 20 3 1 8 :=
by sorry

end NUMINAMATH_CALUDE_oliver_final_balance_l996_99655


namespace NUMINAMATH_CALUDE_inequality_proof_l996_99656

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l996_99656


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l996_99670

/-- The number to be expressed in scientific notation -/
def original_number : ℝ := 384000000

/-- The coefficient in scientific notation -/
def coefficient : ℝ := 3.84

/-- The exponent in scientific notation -/
def exponent : ℕ := 8

/-- Theorem stating that the original number is equal to its scientific notation form -/
theorem scientific_notation_equality :
  original_number = coefficient * (10 : ℝ) ^ exponent := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l996_99670


namespace NUMINAMATH_CALUDE_scientific_notation_3080000_l996_99619

theorem scientific_notation_3080000 :
  (3080000 : ℝ) = 3.08 * (10 ^ 6) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_3080000_l996_99619


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l996_99632

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Line → Prop)

-- Theorem 1: Two planes parallel to the same plane are parallel to each other
theorem planes_parallel_to_same_plane_are_parallel 
  (P Q R : Plane) (h1 : parallel P R) (h2 : parallel Q R) : parallel P Q :=
sorry

-- Theorem 2: Two planes perpendicular to the same line are parallel to each other
theorem planes_perpendicular_to_same_line_are_parallel 
  (P Q : Plane) (L : Line) (h1 : perpendicular P L) (h2 : perpendicular Q L) : parallel P Q :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l996_99632


namespace NUMINAMATH_CALUDE_stationery_cost_theorem_l996_99644

/-- Calculates the total cost of stationery given the number of boxes of pencils,
    pencils per box, cost per pencil, and cost per pen. -/
def total_stationery_cost (boxes : ℕ) (pencils_per_box : ℕ) (pencil_cost : ℕ) (pen_cost : ℕ) : ℕ :=
  let total_pencils := boxes * pencils_per_box
  let total_pens := 2 * total_pencils + 300
  let pencil_total_cost := total_pencils * pencil_cost
  let pen_total_cost := total_pens * pen_cost
  pencil_total_cost + pen_total_cost

/-- Theorem stating that the total cost of stationery under the given conditions is $18,300. -/
theorem stationery_cost_theorem :
  total_stationery_cost 15 80 4 5 = 18300 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_theorem_l996_99644


namespace NUMINAMATH_CALUDE_survey_problem_l996_99665

theorem survey_problem (A B C : ℝ) 
  (h_A : A = 50)
  (h_B : B = 30)
  (h_C : C = 20)
  (h_union : A + B + C - 17 = 78) 
  (h_multiple : 17 ≤ A + B + C - 78) :
  A + B + C - 78 = 5 := by
sorry

end NUMINAMATH_CALUDE_survey_problem_l996_99665


namespace NUMINAMATH_CALUDE_no_complex_root_for_integer_polynomial_l996_99674

/-- A polynomial of degree 4 with leading coefficient 1 and integer coefficients -/
def IntegerPolynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℤ, ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + d

/-- The property that a polynomial has two integer roots -/
def HasTwoIntegerRoots (P : ℝ → ℝ) : Prop :=
  ∃ p q : ℤ, P p = 0 ∧ P q = 0

/-- Complex number of the form (a + b*i)/2 where a and b are integers and b is non-zero -/
def ComplexRoot (z : ℂ) : Prop :=
  ∃ a b : ℤ, z = (a + b*Complex.I)/2 ∧ b ≠ 0

theorem no_complex_root_for_integer_polynomial (P : ℝ → ℝ) :
  IntegerPolynomial P → HasTwoIntegerRoots P →
  ¬∃ z : ℂ, ComplexRoot z ∧ (P z.re = 0 ∧ P z.im = 0) :=
sorry

end NUMINAMATH_CALUDE_no_complex_root_for_integer_polynomial_l996_99674


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l996_99633

-- Define the universal set U as the real numbers
def U := ℝ

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = 2^(|x|)}

-- Define set N
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.log (3 - x)}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = {t : ℝ | 1 ≤ t ∧ t < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l996_99633


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l996_99667

theorem largest_constant_inequality (x y : ℝ) :
  ∃ (C : ℝ), C = Real.sqrt 5 ∧
  (∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 10 ≥ C*(x + y + 2)) ∧
  (∀ (D : ℝ), D > C → ∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 10 < D*(x + y + 2)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l996_99667


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l996_99672

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a binary representation (list of bits). -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- The main theorem to prove -/
theorem binary_addition_subtraction :
  let a := [true, false, true, true]   -- 1101₂
  let b := [true, true, true]          -- 111₂
  let c := [false, true, false, true]  -- 1010₂
  let d := [true, false, false, true]  -- 1001₂
  let result := [true, true, true, false, false, true] -- 100111₂
  binary_to_nat a + binary_to_nat b - binary_to_nat c + binary_to_nat d =
  binary_to_nat result :=
by
  sorry


end NUMINAMATH_CALUDE_binary_addition_subtraction_l996_99672


namespace NUMINAMATH_CALUDE_binomial_7_2_l996_99639

theorem binomial_7_2 : (7 : ℕ).choose 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_2_l996_99639


namespace NUMINAMATH_CALUDE_office_age_problem_l996_99602

theorem office_age_problem (total_persons : Nat) (group1_persons : Nat) (group2_persons : Nat)
  (total_avg_age : ℝ) (group1_avg_age : ℝ) (group2_avg_age : ℝ)
  (h1 : total_persons = 16)
  (h2 : group1_persons = 5)
  (h3 : group2_persons = 9)
  (h4 : total_avg_age = 15)
  (h5 : group1_avg_age = 14)
  (h6 : group2_avg_age = 16)
  (h7 : group1_persons + group2_persons + 2 = total_persons) :
  ∃ (person15_age : ℝ),
    person15_age = total_persons * total_avg_age -
      (group1_persons * group1_avg_age + group2_persons * group2_avg_age) ∧
    person15_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_office_age_problem_l996_99602


namespace NUMINAMATH_CALUDE_michelle_score_l996_99617

/-- Michelle's basketball game record --/
theorem michelle_score (total_score : ℕ) (num_players : ℕ) (other_players : ℕ) (avg_other_score : ℕ) : 
  total_score = 72 →
  num_players = 8 →
  other_players = 7 →
  avg_other_score = 6 →
  total_score - (other_players * avg_other_score) = 30 := by
sorry

end NUMINAMATH_CALUDE_michelle_score_l996_99617


namespace NUMINAMATH_CALUDE_polynomial_factorization_l996_99643

theorem polynomial_factorization (k : ℝ) : 
  (∀ x : ℝ, x^2 - k*x - 6 = (x - 2)*(x + 3)) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l996_99643


namespace NUMINAMATH_CALUDE_book_pages_theorem_l996_99685

/-- A book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- The total number of pages in a book -/
def total_pages (b : Book) : ℕ := b.chapter1_pages + b.chapter2_pages

/-- Theorem: A book with 48 pages in the first chapter and 46 pages in the second chapter has 94 pages in total -/
theorem book_pages_theorem :
  ∀ (b : Book), b.chapter1_pages = 48 → b.chapter2_pages = 46 → total_pages b = 94 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l996_99685


namespace NUMINAMATH_CALUDE_parabola_symmetry_and_range_l996_99612

/-- A parabola with equation y = ax² + bx -/
structure Parabola where
  a : ℝ
  b : ℝ
  a_pos : a > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on a parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = para.a * p.x^2 + para.b * p.x

theorem parabola_symmetry_and_range 
  (para : Parabola) 
  (M N P : Point)
  (h_M : lies_on M para)
  (h_N : lies_on N para)
  (h_P : lies_on P para)
  (h_M_x : M.x = 2)
  (h_N_x : N.x = 4)
  (h_P_x : P.x = -1)
  (h_mn_neg : M.y * N.y < 0)
  (h_m_p_n : M.y < P.y ∧ P.y < N.y)
  (h_m_eq_n : M.y = N.y)
  (t : ℝ)
  (h_t_symm : t = -para.b / (2 * para.a)) :
  t = 3 ∧ 1 < t ∧ t < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_and_range_l996_99612


namespace NUMINAMATH_CALUDE_total_cost_after_increase_l996_99657

def price_increase : ℚ := 15 / 100

def original_orange_price : ℚ := 40
def original_mango_price : ℚ := 50

def new_orange_price : ℚ := original_orange_price * (1 + price_increase)
def new_mango_price : ℚ := original_mango_price * (1 + price_increase)

def total_cost : ℚ := 10 * new_orange_price + 10 * new_mango_price

theorem total_cost_after_increase :
  total_cost = 1035 := by sorry

end NUMINAMATH_CALUDE_total_cost_after_increase_l996_99657


namespace NUMINAMATH_CALUDE_angie_drinks_three_cups_per_day_l996_99647

/-- Represents the number of cups of coffee per pound -/
def cupsPerPound : ℕ := 40

/-- Represents the number of pounds of coffee bought -/
def poundsBought : ℕ := 3

/-- Represents the number of days the coffee lasts -/
def daysLasting : ℕ := 40

/-- Calculates the number of cups of coffee Angie drinks per day -/
def cupsPerDay : ℕ := (poundsBought * cupsPerPound) / daysLasting

/-- Theorem stating that Angie drinks 3 cups of coffee per day -/
theorem angie_drinks_three_cups_per_day : cupsPerDay = 3 := by
  sorry

end NUMINAMATH_CALUDE_angie_drinks_three_cups_per_day_l996_99647


namespace NUMINAMATH_CALUDE_sum_of_complex_equation_l996_99668

/-- Given real numbers x and y satisfying (2+i)x = 4+yi, prove that x + y = 4 -/
theorem sum_of_complex_equation (x y : ℝ) : 
  (Complex.I : ℂ) * x + 2 * x = 4 + (Complex.I : ℂ) * y → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_equation_l996_99668


namespace NUMINAMATH_CALUDE_point_not_on_line_l996_99695

theorem point_not_on_line (p q : ℝ) (h : p * q < 0) :
  -101 ≠ 21 * p + q := by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l996_99695


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l996_99616

/-- Calculates the total number of heartbeats during a race. -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  heart_rate * pace * race_distance

/-- Proves that the athlete's heart beats 28800 times during the 30-mile race. -/
theorem athlete_heartbeats :
  let heart_rate : ℕ := 160  -- heartbeats per minute
  let pace : ℕ := 6          -- minutes per mile
  let race_distance : ℕ := 30 -- miles
  total_heartbeats heart_rate pace race_distance = 28800 := by
  sorry


end NUMINAMATH_CALUDE_athlete_heartbeats_l996_99616


namespace NUMINAMATH_CALUDE_fencing_rate_proof_l996_99660

/-- Given a rectangular plot with the following properties:
    - The length is 10 meters more than the width
    - The perimeter is 340 meters
    - The total cost of fencing is 2210 Rs
    Prove that the rate per meter for fencing is 6.5 Rs -/
theorem fencing_rate_proof (width : ℝ) (length : ℝ) (perimeter : ℝ) (total_cost : ℝ) :
  length = width + 10 →
  perimeter = 340 →
  perimeter = 2 * (length + width) →
  total_cost = 2210 →
  total_cost / perimeter = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_rate_proof_l996_99660


namespace NUMINAMATH_CALUDE_fraction_simplification_implies_even_difference_l996_99673

theorem fraction_simplification_implies_even_difference 
  (a b c d : ℕ) (h1 : ∀ n : ℕ, c * n + d ≠ 0) 
  (h2 : ∀ n : ℕ, ∃ k : ℕ, a * n + b = 2 * k ∧ c * n + d = 2 * k) : 
  Even (a * d - b * c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_implies_even_difference_l996_99673


namespace NUMINAMATH_CALUDE_equation_solution_l996_99640

theorem equation_solution : ∃ x : ℝ, 
  x = 625 ∧ 
  Real.sqrt (3 + Real.sqrt (4 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l996_99640


namespace NUMINAMATH_CALUDE_equidistant_point_on_number_line_l996_99681

/-- Given points A (-1) and B (5) on a number line, if point P is equidistant from A and B, then P represents the number 2. -/
theorem equidistant_point_on_number_line :
  let a : ℝ := -1
  let b : ℝ := 5
  ∀ p : ℝ, |p - a| = |p - b| → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_number_line_l996_99681


namespace NUMINAMATH_CALUDE_iggy_thursday_miles_l996_99630

/-- Represents Iggy's running schedule for a week --/
structure RunningSchedule where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total miles run in a week --/
def totalMiles (schedule : RunningSchedule) : Nat :=
  schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday

/-- Calculates the total minutes run in a week given a pace in minutes per mile --/
def totalMinutes (schedule : RunningSchedule) (pace : Nat) : Nat :=
  (totalMiles schedule) * pace

/-- Theorem stating that Iggy ran 8 miles on Thursday --/
theorem iggy_thursday_miles :
  ∀ (schedule : RunningSchedule) (pace : Nat),
    schedule.monday = 3 →
    schedule.tuesday = 4 →
    schedule.wednesday = 6 →
    schedule.friday = 3 →
    pace = 10 →
    totalMinutes schedule pace = 4 * 60 →
    schedule.thursday = 8 := by
  sorry


end NUMINAMATH_CALUDE_iggy_thursday_miles_l996_99630


namespace NUMINAMATH_CALUDE_average_multiplication_l996_99688

theorem average_multiplication (numbers : Finset ℕ) (sum : ℕ) :
  numbers.card = 7 →
  sum / 7 = 24 →
  (5 * sum) / 7 = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_average_multiplication_l996_99688


namespace NUMINAMATH_CALUDE_corrected_mean_l996_99609

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value correct_value : ℝ) 
  (h1 : n = 50) 
  (h2 : original_mean = 41) 
  (h3 : incorrect_value = 23) 
  (h4 : correct_value = 48) : 
  (n : ℝ) * original_mean - incorrect_value + correct_value = n * 41.5 := by
  sorry

#check corrected_mean

end NUMINAMATH_CALUDE_corrected_mean_l996_99609


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l996_99692

theorem mark_and_carolyn_money_sum :
  let mark_money : ℚ := 7/8
  let carolyn_money : ℚ := 2/5
  (mark_money + carolyn_money : ℚ) = 1.275 := by sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l996_99692


namespace NUMINAMATH_CALUDE_parallel_condition_not_sufficient_nor_necessary_l996_99629

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Two lines are parallel -/
def parallel_lines (l m : Line3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

theorem parallel_condition_not_sufficient_nor_necessary 
  (l m : Line3D) (α : Plane3D) 
  (h_diff : l ≠ m) (h_parallel : parallel_lines l m) : 
  (¬ (∀ α, parallel_line_plane l α → parallel_line_plane m α)) ∧ 
  (¬ (∀ α, parallel_line_plane m α → parallel_line_plane l α)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_condition_not_sufficient_nor_necessary_l996_99629


namespace NUMINAMATH_CALUDE_circumradius_of_specific_isosceles_triangle_l996_99620

/-- An isosceles triangle with base 6 and side length 5 -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  is_isosceles : base = 6 ∧ side = 5

/-- The radius of the circumcircle of a triangle -/
def circumradius (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem: The radius of the circumcircle of an isosceles triangle with base 6 and side length 5 is 25/8 -/
theorem circumradius_of_specific_isosceles_triangle (t : IsoscelesTriangle) : 
  circumradius t = 25/8 := by sorry

end NUMINAMATH_CALUDE_circumradius_of_specific_isosceles_triangle_l996_99620


namespace NUMINAMATH_CALUDE_quadruple_work_time_l996_99634

-- Define the work rates for A and B
def work_rate_A : ℚ := 1 / 45
def work_rate_B : ℚ := 1 / 30

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Define the time to complete 4 times the work
def time_for_quadruple_work : ℚ := 4 / combined_work_rate

-- Theorem statement
theorem quadruple_work_time : time_for_quadruple_work = 9/2 := by sorry

end NUMINAMATH_CALUDE_quadruple_work_time_l996_99634


namespace NUMINAMATH_CALUDE_N_subset_M_l996_99625

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2 * p.1 + 1}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -(p.1^2)}

-- State the theorem
theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l996_99625


namespace NUMINAMATH_CALUDE_projection_coordinates_l996_99684

/-- The plane equation ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The projection of a point onto a plane -/
def projection (p : Point3D) (plane : Plane) : Point3D :=
  sorry

theorem projection_coordinates :
  let p := Point3D.mk 1 2 (-1)
  let plane := Plane.mk 3 (-1) 2 (-4)
  let proj := projection p plane
  proj.x = 29 / 14 ∧ proj.y = 23 / 14 ∧ proj.z = -2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_projection_coordinates_l996_99684


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l996_99676

theorem angle_sum_theorem (x : ℝ) : 
  (6 * x + 7 * x + 3 * x + 4 * x) * (π / 180) = 2 * π → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l996_99676


namespace NUMINAMATH_CALUDE_acute_triangle_contains_grid_point_l996_99663

-- Define a point on a graph paper grid
structure GridPoint where
  x : ℤ
  y : ℤ

-- Define a triangle on a graph paper grid
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

-- Define what it means for a triangle to be acute
def isAcute (t : GridTriangle) : Prop :=
  sorry -- Definition of acute triangle on a grid

-- Define what it means for a point to be inside or on the sides of a triangle
def isInsideOrOnSides (p : GridPoint) (t : GridTriangle) : Prop :=
  sorry -- Definition of a point being inside or on the sides of a triangle

-- The main theorem
theorem acute_triangle_contains_grid_point (t : GridTriangle) :
  isAcute t →
  ∃ p : GridPoint, p ≠ t.A ∧ p ≠ t.B ∧ p ≠ t.C ∧ isInsideOrOnSides p t :=
sorry

end NUMINAMATH_CALUDE_acute_triangle_contains_grid_point_l996_99663


namespace NUMINAMATH_CALUDE_larger_number_proof_l996_99649

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1311) (h3 : L = 11 * S + 11) : L = 1441 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l996_99649


namespace NUMINAMATH_CALUDE_white_paper_bunches_l996_99675

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The number of sheets in a bunch -/
def sheets_per_bunch : ℕ := 4

/-- The number of sheets in a bundle -/
def sheets_per_bundle : ℕ := 2

/-- The number of sheets in a heap -/
def sheets_per_heap : ℕ := 20

/-- The total number of sheets removed -/
def total_sheets_removed : ℕ := 114

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

theorem white_paper_bunches :
  white_bunches * sheets_per_bunch = 
    total_sheets_removed - 
    (colored_bundles * sheets_per_bundle + scrap_heaps * sheets_per_heap) :=
by sorry

end NUMINAMATH_CALUDE_white_paper_bunches_l996_99675


namespace NUMINAMATH_CALUDE_ratio_comparison_l996_99678

theorem ratio_comparison : ∀ (a b : ℕ), 
  a = 6 ∧ b = 7 →
  ∃ (x : ℕ), x = 3 ∧
  (a - x : ℚ) / (b - x : ℚ) < 3 / 4 ∧
  ∀ (y : ℕ), y < x →
  (a - y : ℚ) / (b - y : ℚ) ≥ 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ratio_comparison_l996_99678


namespace NUMINAMATH_CALUDE_dog_toy_cost_l996_99614

/-- The cost of dog toys with a "buy one get one half off" deal -/
theorem dog_toy_cost (regular_price : ℚ) (num_toys : ℕ) : 
  regular_price = 12 →
  num_toys = 4 →
  (num_toys / 2 : ℚ) * regular_price + (num_toys / 2 : ℚ) * (regular_price / 2) = 36 :=
by
  sorry

#check dog_toy_cost

end NUMINAMATH_CALUDE_dog_toy_cost_l996_99614


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l996_99683

theorem simplify_and_rationalize (x : ℝ) :
  x = 1 / (2 - 1 / (Real.sqrt 5 + 2)) →
  x = (4 + Real.sqrt 5) / 11 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l996_99683


namespace NUMINAMATH_CALUDE_cuboid_height_from_volume_and_base_area_l996_99611

/-- Represents the properties of a cuboid -/
structure Cuboid where
  volume : ℝ
  baseArea : ℝ
  height : ℝ

/-- Theorem stating that a cuboid with volume 144 and base area 18 has height 8 -/
theorem cuboid_height_from_volume_and_base_area :
  ∀ (c : Cuboid), c.volume = 144 → c.baseArea = 18 → c.height = 8 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_from_volume_and_base_area_l996_99611


namespace NUMINAMATH_CALUDE_duck_flock_size_l996_99610

/-- Calculates the total number of ducks in a combined flock after a given number of years -/
def combined_flock_size (initial_size : ℕ) (annual_increase : ℕ) (years : ℕ) (joining_flock : ℕ) : ℕ :=
  initial_size + annual_increase * years + joining_flock

/-- Theorem stating the combined flock size after 5 years -/
theorem duck_flock_size :
  combined_flock_size 100 10 5 150 = 300 := by
  sorry

#eval combined_flock_size 100 10 5 150

end NUMINAMATH_CALUDE_duck_flock_size_l996_99610


namespace NUMINAMATH_CALUDE_sector_central_angle_l996_99687

/-- Given a circular sector with perimeter 10 and area 4, prove that its central angle is 1/2 radians -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 10) (h2 : 1/2 * l * r = 4) :
  l / r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l996_99687


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l996_99693

/-- The speed of a man rowing in still water, given his downstream speed and current speed. -/
theorem mans_rowing_speed (downstream_distance : ℝ) (downstream_time : ℝ) (current_speed : ℝ) : 
  downstream_distance / downstream_time * 3600 / 1000 - current_speed = 6 :=
by
  sorry

#check mans_rowing_speed 110 44 3

end NUMINAMATH_CALUDE_mans_rowing_speed_l996_99693


namespace NUMINAMATH_CALUDE_ferry_speed_difference_l996_99603

/-- Represents the speed and time of a ferry journey -/
structure FerryJourney where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a ferry -/
def distance (journey : FerryJourney) : ℝ :=
  journey.speed * journey.time

theorem ferry_speed_difference :
  let ferryP : FerryJourney := { speed := 8, time := 3 }
  let ferryQ : FerryJourney := { speed := (3 * distance ferryP) / (ferryP.time + 5), time := ferryP.time + 5 }
  ferryQ.speed - ferryP.speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_ferry_speed_difference_l996_99603


namespace NUMINAMATH_CALUDE_object_speed_mph_l996_99638

-- Define the distance traveled in feet
def distance_feet : ℝ := 400

-- Define the time traveled in seconds
def time_seconds : ℝ := 4

-- Define the conversion factor from feet to miles
def feet_per_mile : ℝ := 5280

-- Define the conversion factor from seconds to hours
def seconds_per_hour : ℝ := 3600

-- Theorem statement
theorem object_speed_mph :
  let distance_miles := distance_feet / feet_per_mile
  let time_hours := time_seconds / seconds_per_hour
  let speed_mph := distance_miles / time_hours
  ∃ ε > 0, |speed_mph - 68.18| < ε :=
sorry

end NUMINAMATH_CALUDE_object_speed_mph_l996_99638


namespace NUMINAMATH_CALUDE_sample_customers_l996_99621

theorem sample_customers (samples_per_box : ℕ) (boxes_opened : ℕ) (samples_left : ℕ) : 
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_left = 5 →
  (samples_per_box * boxes_opened - samples_left) = 235 := by
  sorry

end NUMINAMATH_CALUDE_sample_customers_l996_99621


namespace NUMINAMATH_CALUDE_CH4_required_for_CCl4_l996_99635

-- Define the chemical species as real numbers (representing moles)
variable (CH4 CH2Cl2 CCl4 CHCl3 HCl Cl2 CH3Cl : ℝ)

-- Define the equilibrium constants
def K1 : ℝ := 1.2 * 10^2
def K2 : ℝ := 1.5 * 10^3
def K3 : ℝ := 3.4 * 10^4

-- Define the initial amounts of species
def initial_CH2Cl2 : ℝ := 2.5
def initial_CHCl3 : ℝ := 1.5
def initial_HCl : ℝ := 0.5
def initial_Cl2 : ℝ := 10
def initial_CH3Cl : ℝ := 0.2

-- Define the target amount of CCl4
def target_CCl4 : ℝ := 5

-- Theorem statement
theorem CH4_required_for_CCl4 :
  ∃ (required_CH4 : ℝ),
    required_CH4 = 2.5 ∧
    required_CH4 + initial_CH2Cl2 = target_CCl4 :=
sorry

end NUMINAMATH_CALUDE_CH4_required_for_CCl4_l996_99635


namespace NUMINAMATH_CALUDE_andrew_brought_40_chicken_nuggets_l996_99628

/-- Represents the number of appetizer portions Andrew brought -/
def total_appetizers : ℕ := 90

/-- Represents the number of hotdogs on sticks Andrew brought -/
def hotdogs : ℕ := 30

/-- Represents the number of bite-sized cheese pops Andrew brought -/
def cheese_pops : ℕ := 20

/-- Represents the number of chicken nuggets Andrew brought -/
def chicken_nuggets : ℕ := total_appetizers - hotdogs - cheese_pops

/-- Theorem stating that Andrew brought 40 pieces of chicken nuggets -/
theorem andrew_brought_40_chicken_nuggets : chicken_nuggets = 40 := by
  sorry

end NUMINAMATH_CALUDE_andrew_brought_40_chicken_nuggets_l996_99628


namespace NUMINAMATH_CALUDE_tangent_parallel_x_axis_tangent_parallel_line_l996_99627

-- Define the curve
def x (t : ℝ) : ℝ := t - 1
def y (t : ℝ) : ℝ := t^3 - 12*t + 1

-- Define the derivative of y with respect to x
def dy_dx (t : ℝ) : ℝ := 3*t^2 - 12

-- Define the slope of the line 9x + y + 3 = 0
def m : ℝ := -9

-- Theorem for points where tangent is parallel to x-axis
theorem tangent_parallel_x_axis :
  ∃ t₁ t₂ : ℝ, 
    t₁ ≠ t₂ ∧
    dy_dx t₁ = 0 ∧ dy_dx t₂ = 0 ∧
    x t₁ = 1 ∧ y t₁ = -15 ∧
    x t₂ = -3 ∧ y t₂ = 17 :=
sorry

-- Theorem for points where tangent is parallel to 9x + y + 3 = 0
theorem tangent_parallel_line :
  ∃ t₁ t₂ : ℝ,
    t₁ ≠ t₂ ∧
    dy_dx t₁ = m ∧ dy_dx t₂ = m ∧
    x t₁ = 0 ∧ y t₁ = -10 ∧
    x t₂ = -2 ∧ y t₂ = 12 :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_x_axis_tangent_parallel_line_l996_99627


namespace NUMINAMATH_CALUDE_successive_integers_product_l996_99646

theorem successive_integers_product (n : ℤ) : 
  n * (n + 1) = 7832 → n = 88 := by
  sorry

end NUMINAMATH_CALUDE_successive_integers_product_l996_99646


namespace NUMINAMATH_CALUDE_three_digit_number_puzzle_l996_99666

theorem three_digit_number_puzzle :
  ∃ (x y z : ℕ),
    0 ≤ x ∧ x ≤ 9 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    0 ≤ z ∧ z ≤ 9 ∧
    (100 * x + 10 * y + z) + (100 * z + 10 * y + x) = 1252 ∧
    x + y + z = 14 ∧
    x^2 + y^2 + z^2 = 84 ∧
    100 * x + 10 * y + z = 824 ∧
    100 * z + 10 * y + x = 428 :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_puzzle_l996_99666


namespace NUMINAMATH_CALUDE_unique_positive_solution_l996_99605

theorem unique_positive_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0)
  (eq₁ : x₁ + x₂ = x₃^2)
  (eq₂ : x₂ + x₃ = x₄^2)
  (eq₃ : x₃ + x₄ = x₅^2)
  (eq₄ : x₄ + x₅ = x₁^2)
  (eq₅ : x₅ + x₁ = x₂^2) :
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l996_99605


namespace NUMINAMATH_CALUDE_polynomial_simplification_l996_99696

theorem polynomial_simplification (m : ℝ) : 
  (∀ x y : ℝ, (2 * m * x^2 + 4 * x^2 + 3 * x + 1) - (6 * x^2 - 4 * y^2 + 3 * x) = 4 * y^2 + 1) ↔ 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l996_99696


namespace NUMINAMATH_CALUDE_four_by_four_cube_unpainted_l996_99658

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_units : Nat
  painted_per_face : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_units - (cube.size * cube.size * 6 - (cube.size - 2) * (cube.size - 2) * 6)

/-- Theorem stating that a 4x4x4 cube with 4 painted squares per face has 52 unpainted unit cubes -/
theorem four_by_four_cube_unpainted :
  let cube : PaintedCube := { size := 4, total_units := 64, painted_per_face := 4 }
  unpainted_cubes cube = 52 := by
  sorry

end NUMINAMATH_CALUDE_four_by_four_cube_unpainted_l996_99658


namespace NUMINAMATH_CALUDE_range_of_m_satisfying_condition_l996_99645

theorem range_of_m_satisfying_condition :
  {m : ℝ | ∀ x : ℝ, m * x^2 - (3 - m) * x + 1 > 0 ∨ m * x > 0} = {m : ℝ | 1/9 < m ∧ m < 1} := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_satisfying_condition_l996_99645


namespace NUMINAMATH_CALUDE_kamal_physics_marks_l996_99659

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average marks for a student -/
def average (marks : StudentMarks) : ℚ :=
  (marks.english + marks.mathematics + marks.physics + marks.chemistry + marks.biology) / 5

theorem kamal_physics_marks :
  ∀ (marks : StudentMarks),
    marks.english = 76 →
    marks.mathematics = 60 →
    marks.chemistry = 67 →
    marks.biology = 85 →
    average marks = 74 →
    marks.physics = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_kamal_physics_marks_l996_99659


namespace NUMINAMATH_CALUDE_rectangle_sides_l996_99697

theorem rectangle_sides (h b : ℝ) (h_pos : h > 0) (b_pos : b > 0) : 
  b = 3 * h → 
  h * b = 2 * (h + b) + Real.sqrt (h^2 + b^2) → 
  h = (8 + Real.sqrt 10) / 3 ∧ b = 8 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sides_l996_99697


namespace NUMINAMATH_CALUDE_complex_magnitude_l996_99662

theorem complex_magnitude (z : ℂ) (h : z = Complex.mk 2 (-1)) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l996_99662


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l996_99682

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  (∀ n, a (n + 1) > a n) →      -- increasing sequence
  a 2 = 2 →                     -- a_2 = 2
  a 4 - a 3 = 4 →               -- a_4 - a_3 = 4
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l996_99682


namespace NUMINAMATH_CALUDE_six_points_theorem_l996_99608

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  -- This is a simplified representation
  is_convex : Bool

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  -- Simplified representation of a line
  point1 : Point
  point2 : Point

/-- Calculates the vector between two points -/
def vector (p1 p2 : Point) : Point :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

/-- Checks if a point is on the side of a polygon -/
def is_on_side (p : Point) (poly : ConvexPolygon) : Prop :=
  sorry -- Define the condition for a point to be on the side of the polygon

/-- Calculates the distance between a line and a point -/
def distance_line_point (l : Line) (p : Point) : ℝ :=
  sorry -- Define the distance calculation

theorem six_points_theorem (H : ConvexPolygon) (a : ℝ) 
    (h1 : 0 < a) (h2 : a < 1) :
  ∃ (A1 A2 A3 A4 A5 A6 : Point),
    is_on_side A1 H ∧ is_on_side A2 H ∧ is_on_side A3 H ∧
    is_on_side A4 H ∧ is_on_side A5 H ∧ is_on_side A6 H ∧
    A1 ≠ A2 ∧ A2 ≠ A3 ∧ A3 ≠ A4 ∧ A4 ≠ A5 ∧ A5 ≠ A6 ∧ A6 ≠ A1 ∧
    vector A1 A2 = vector A5 A4 ∧
    vector A1 A2 = vector (Point.mk 0 0) (Point.mk (a * (A6.x - A3.x)) (a * (A6.y - A3.y))) ∧
    distance_line_point (Line.mk A1 A2) A3 = distance_line_point (Line.mk A5 A4) A3 :=
by
  sorry


end NUMINAMATH_CALUDE_six_points_theorem_l996_99608


namespace NUMINAMATH_CALUDE_lukes_stickers_l996_99615

theorem lukes_stickers (initial_stickers birthday_stickers given_to_sister used_on_card final_stickers : ℕ) :
  initial_stickers = 20 →
  birthday_stickers = 20 →
  given_to_sister = 5 →
  used_on_card = 8 →
  final_stickers = 39 →
  ∃ (bought_stickers : ℕ),
    bought_stickers = 12 ∧
    initial_stickers + birthday_stickers + bought_stickers = final_stickers + given_to_sister + used_on_card :=
by sorry

end NUMINAMATH_CALUDE_lukes_stickers_l996_99615


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l996_99686

theorem binomial_coefficient_problem (m : ℕ+) 
  (a b : ℕ) 
  (ha : a = Nat.choose (2 * m) m)
  (hb : b = Nat.choose (2 * m + 1) m)
  (h_eq : 13 * a = 7 * b) : 
  m = 6 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l996_99686


namespace NUMINAMATH_CALUDE_unique_solution_is_two_l996_99671

theorem unique_solution_is_two : 
  ∃! (x : ℝ), x > 0 ∧ x^(2^2) = 2^(x^2) ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_is_two_l996_99671


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l996_99622

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 7 → x ≥ 4 ∧ 4 < 3*4 - 7 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l996_99622


namespace NUMINAMATH_CALUDE_line_through_two_points_l996_99624

theorem line_through_two_points (m n p : ℝ) :
  (m = 3 * n + 5) ∧ (m + 2 = 3 * (n + p) + 5) → p = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_two_points_l996_99624


namespace NUMINAMATH_CALUDE_circle_sum_inequality_l996_99680

theorem circle_sum_inequality (a : Fin 100 → ℝ) (h : Function.Injective a) :
  ∃ i : Fin 100, a i + a ((i + 3) % 100) > a ((i + 1) % 100) + a ((i + 2) % 100) := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_inequality_l996_99680


namespace NUMINAMATH_CALUDE_roots_square_sum_l996_99637

theorem roots_square_sum (a b : ℝ) : 
  (∀ x, x^2 - 2*x - 1 = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_roots_square_sum_l996_99637


namespace NUMINAMATH_CALUDE_no_rational_solution_l996_99642

theorem no_rational_solution : ¬∃ (x y z : ℚ), (x + y + z = 0) ∧ (x^2 + y^2 + z^2 = 100) := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l996_99642


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l996_99626

/-- The minimum number of additional coins needed for distribution --/
def min_additional_coins (n : ℕ) (initial_coins : ℕ) : ℕ :=
  (n * (n + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex's distribution --/
theorem alex_coin_distribution :
  min_additional_coins 20 192 = 18 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l996_99626


namespace NUMINAMATH_CALUDE_point_on_bisector_coordinates_l996_99641

/-- A point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The bisector of the first and third quadrants -/
def firstThirdQuadrantBisector (p : Point) : Prop :=
  p.x = p.y

/-- Point P with coordinates (a, 2a-1) -/
def P (a : ℝ) : Point :=
  { x := a, y := 2 * a - 1 }

/-- Theorem stating that if P(a) is on the bisector, its coordinates are (1, 1) -/
theorem point_on_bisector_coordinates :
  ∀ a : ℝ, firstThirdQuadrantBisector (P a) → P a = { x := 1, y := 1 } :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_bisector_coordinates_l996_99641


namespace NUMINAMATH_CALUDE_terms_are_like_when_k_is_two_l996_99691

/-- Two monomials are like terms if they have the same variables raised to the same powers -/
def like_terms (term1 term2 : ℕ → ℕ) : Prop :=
  ∀ var, term1 var = term2 var

/-- The first term: -3x²y³ᵏ -/
def term1 (k : ℕ) : ℕ → ℕ
| 0 => 2  -- x has power 2
| 1 => 3 * k  -- y has power 3k
| _ => 0  -- other variables have power 0

/-- The second term: 4x²y⁶ -/
def term2 : ℕ → ℕ
| 0 => 2  -- x has power 2
| 1 => 6  -- y has power 6
| _ => 0  -- other variables have power 0

/-- Theorem: When k = 2, -3x²y³ᵏ and 4x²y⁶ are like terms -/
theorem terms_are_like_when_k_is_two : like_terms (term1 2) term2 := by
  sorry

end NUMINAMATH_CALUDE_terms_are_like_when_k_is_two_l996_99691


namespace NUMINAMATH_CALUDE_second_machine_rate_l996_99652

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Given two copy machines where the first makes 40 copies per minute,
    and together they make 2850 copies in half an hour,
    prove that the second machine makes 55 copies per minute -/
theorem second_machine_rate 
  (machine1 machine2 : CopyMachine)
  (h1 : machine1.copies_per_minute = 40)
  (h2 : machine1.copies_per_minute * 30 + machine2.copies_per_minute * 30 = 2850) :
  machine2.copies_per_minute = 55 := by
sorry

end NUMINAMATH_CALUDE_second_machine_rate_l996_99652


namespace NUMINAMATH_CALUDE_gcd_relation_l996_99689

theorem gcd_relation (a b : ℤ) : Int.gcd a b = 1 → Int.gcd (2*a + b) (a*(a + b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_relation_l996_99689


namespace NUMINAMATH_CALUDE_complex_equation_solution_l996_99650

theorem complex_equation_solution (Z : ℂ) : (2 + 4*I) / Z = 1 - I → Z = -1 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l996_99650


namespace NUMINAMATH_CALUDE_least_n_for_g_prime_product_l996_99636

def g (n : ℕ) : ℕ := n.choose 3

def isArithmeticProgression (p₁ p₂ p₃ : ℕ) (d : ℕ) : Prop :=
  p₂ = p₁ + d ∧ p₃ = p₂ + d

theorem least_n_for_g_prime_product : 
  ∃ (p₁ p₂ p₃ : ℕ),
    p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧
    p₁ < p₂ ∧ p₂ < p₃ ∧
    isArithmeticProgression p₁ p₂ p₃ 336 ∧
    g 2019 = p₁ * p₂ * p₃ ∧
    (∀ n < 2019, ¬∃ (q₁ q₂ q₃ : ℕ),
      q₁.Prime ∧ q₂.Prime ∧ q₃.Prime ∧
      q₁ < q₂ ∧ q₂ < q₃ ∧
      isArithmeticProgression q₁ q₂ q₃ 336 ∧
      g n = q₁ * q₂ * q₃) :=
by sorry

end NUMINAMATH_CALUDE_least_n_for_g_prime_product_l996_99636


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l996_99690

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (160 * π / 180) ^ 2) = Real.cos (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l996_99690


namespace NUMINAMATH_CALUDE_sabrina_cookies_l996_99661

/-- The number of cookies Sabrina had at the start -/
def initial_cookies : ℕ := 20

/-- The number of cookies Sabrina gave to her brother -/
def cookies_to_brother : ℕ := 10

/-- The number of cookies Sabrina's mother gave her -/
def cookies_from_mother : ℕ := cookies_to_brother / 2

/-- The fraction of cookies Sabrina gave to her sister -/
def fraction_to_sister : ℚ := 2 / 3

/-- The number of cookies Sabrina has left -/
def remaining_cookies : ℕ := 5

theorem sabrina_cookies :
  initial_cookies = cookies_to_brother + 
    (initial_cookies - cookies_to_brother + cookies_from_mother) * (1 - fraction_to_sister) :=
by sorry

end NUMINAMATH_CALUDE_sabrina_cookies_l996_99661


namespace NUMINAMATH_CALUDE_total_blue_marbles_l996_99601

-- Define the number of marbles collected by each friend
def jenny_red : ℕ := 30
def jenny_blue : ℕ := 25
def mary_red : ℕ := 2 * jenny_red
def anie_red : ℕ := mary_red + 20
def anie_blue : ℕ := 2 * jenny_blue
def mary_blue : ℕ := anie_blue / 2

-- Theorem to prove
theorem total_blue_marbles :
  jenny_blue + mary_blue + anie_blue = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_marbles_l996_99601


namespace NUMINAMATH_CALUDE_six_by_six_square_1x4_rectangles_impossible_l996_99648

theorem six_by_six_square_1x4_rectangles_impossible : ¬ ∃ (a b : ℕ), 
  a + 4*b = 6 ∧ 4*a + b = 6 :=
sorry

end NUMINAMATH_CALUDE_six_by_six_square_1x4_rectangles_impossible_l996_99648


namespace NUMINAMATH_CALUDE_mike_additional_money_needed_l996_99651

def phone_cost : ℝ := 1300
def smartwatch_cost : ℝ := 500
def phone_discount : ℝ := 0.10
def smartwatch_discount : ℝ := 0.15
def sales_tax : ℝ := 0.07
def mike_savings_percentage : ℝ := 0.40

theorem mike_additional_money_needed :
  let discounted_phone := phone_cost * (1 - phone_discount)
  let discounted_smartwatch := smartwatch_cost * (1 - smartwatch_discount)
  let total_before_tax := discounted_phone + discounted_smartwatch
  let total_with_tax := total_before_tax * (1 + sales_tax)
  let mike_savings := total_with_tax * mike_savings_percentage
  total_with_tax - mike_savings = 1023.99 := by sorry

end NUMINAMATH_CALUDE_mike_additional_money_needed_l996_99651


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l996_99604

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem smallest_fourth_number (x : ℕ) : 
  x ≥ 10 ∧ x < 100 →
  (sum_of_digits 21 + sum_of_digits 34 + sum_of_digits 65 + sum_of_digits x) * 4 = 
  (21 + 34 + 65 + x) →
  x ≥ 12 :=
by
  sorry

#eval sum_of_digits 21 + sum_of_digits 34 + sum_of_digits 65 + sum_of_digits 12
#eval 21 + 34 + 65 + 12

end NUMINAMATH_CALUDE_smallest_fourth_number_l996_99604


namespace NUMINAMATH_CALUDE_right_triangle_identification_l996_99694

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_identification :
  (is_right_triangle 1 (Real.sqrt 2) (Real.sqrt 3)) ∧
  ¬(is_right_triangle (Real.sqrt 2) (Real.sqrt 3) 2) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 9 16 25) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l996_99694
