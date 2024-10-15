import Mathlib

namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l63_6361

def N : ℕ := sorry  -- Definition of N as concatenation of integers from 34 to 76

theorem highest_power_of_three_dividing_N :
  ∃ k : ℕ, (3^k ∣ N) ∧ ¬(3^(k+1) ∣ N) ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l63_6361


namespace NUMINAMATH_CALUDE_eliza_height_difference_l63_6348

/-- Given the heights of Eliza and her siblings, prove that Eliza is 2 inches shorter than the tallest sibling -/
theorem eliza_height_difference (total_height : ℕ) (sibling1_height sibling2_height sibling3_height eliza_height : ℕ) :
  total_height = 330 ∧
  sibling1_height = 66 ∧
  sibling2_height = 66 ∧
  sibling3_height = 60 ∧
  eliza_height = 68 →
  ∃ (tallest_sibling_height : ℕ),
    tallest_sibling_height + sibling1_height + sibling2_height + sibling3_height + eliza_height = total_height ∧
    tallest_sibling_height - eliza_height = 2 :=
by sorry

end NUMINAMATH_CALUDE_eliza_height_difference_l63_6348


namespace NUMINAMATH_CALUDE_curve_to_line_equation_l63_6327

/-- Given a curve parameterized by (x, y) = (3t + 6, 5t - 7), where t is a real number,
    prove that the equation of the line in the form y = mx + b is y = (5/3)x - 17. -/
theorem curve_to_line_equation :
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 7 →
  ∃ (m b : ℝ), m = 5 / 3 ∧ b = -17 ∧ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_curve_to_line_equation_l63_6327


namespace NUMINAMATH_CALUDE_gcd_difference_square_l63_6322

theorem gcd_difference_square (x y z : ℕ+) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), (Nat.gcd x.val (Nat.gcd y.val z.val)) * (y.val - x.val) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_difference_square_l63_6322


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l63_6305

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l63_6305


namespace NUMINAMATH_CALUDE_minimum_score_for_average_l63_6315

def exam_scores : List ℕ := [92, 85, 89, 93]
def desired_average : ℕ := 90
def num_exams : ℕ := 5

theorem minimum_score_for_average (scores : List ℕ) (avg : ℕ) (n : ℕ) :
  scores.length + 1 = n →
  (scores.sum + (n * avg - scores.sum)) / n = avg →
  n * avg - scores.sum = 91 :=
by sorry

#check minimum_score_for_average exam_scores desired_average num_exams

end NUMINAMATH_CALUDE_minimum_score_for_average_l63_6315


namespace NUMINAMATH_CALUDE_inequality_system_solution_l63_6314

theorem inequality_system_solution (x : ℝ) :
  (1 / x < 1 ∧ |4 * x - 1| > 2) ↔ (x < -1/4 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l63_6314


namespace NUMINAMATH_CALUDE_power_quotient_square_l63_6316

theorem power_quotient_square : (19^12 / 19^8)^2 = 130321 := by sorry

end NUMINAMATH_CALUDE_power_quotient_square_l63_6316


namespace NUMINAMATH_CALUDE_quotient_of_powers_l63_6303

theorem quotient_of_powers (a b c : ℕ) (ha : a = 50) (hb : b = 25) (hc : c = 100) :
  (a ^ 50) / (b ^ 25) = c ^ 25 := by
  sorry

end NUMINAMATH_CALUDE_quotient_of_powers_l63_6303


namespace NUMINAMATH_CALUDE_sets_theorem_l63_6358

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

-- Define the theorem
theorem sets_theorem :
  -- Part 1
  (A (1/2) ∩ (Set.univ \ B (1/2)) = {x | 9/4 ≤ x ∧ x < 5/2}) ∧
  -- Part 2
  (∀ a : ℝ, Set.Subset (A a) (B a) ↔ -1/2 ≤ a ∧ a ≤ (3 - Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_sets_theorem_l63_6358


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l63_6346

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l63_6346


namespace NUMINAMATH_CALUDE_divisor_cube_eq_four_n_l63_6338

/-- The number of positive divisors of a positive integer -/
def d (n : ℕ+) : ℕ := sorry

/-- The theorem stating that the only positive integers n that satisfy d(n)^3 = 4n are 2, 128, and 2000 -/
theorem divisor_cube_eq_four_n : 
  ∀ n : ℕ+, d n ^ 3 = 4 * n ↔ n = 2 ∨ n = 128 ∨ n = 2000 := by sorry

end NUMINAMATH_CALUDE_divisor_cube_eq_four_n_l63_6338


namespace NUMINAMATH_CALUDE_solve_for_p_l63_6318

theorem solve_for_p (p q : ℚ) 
  (eq1 : 5 * p - 2 * q = 14) 
  (eq2 : 6 * p + q = 31) : 
  p = 76 / 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_p_l63_6318


namespace NUMINAMATH_CALUDE_tan_geq_one_range_l63_6395

open Set
open Real

theorem tan_geq_one_range (f : ℝ → ℝ) (h : ∀ x ∈ Ioo (-π/2) (π/2), f x = tan x) :
  {x ∈ Ioo (-π/2) (π/2) | f x ≥ 1} = Ico (π/4) (π/2) := by
  sorry

end NUMINAMATH_CALUDE_tan_geq_one_range_l63_6395


namespace NUMINAMATH_CALUDE_f_ordering_l63_6375

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem f_ordering : f (-π/3) > f (-1) ∧ f (-1) > f (π/11) := by
  sorry

end NUMINAMATH_CALUDE_f_ordering_l63_6375


namespace NUMINAMATH_CALUDE_sin_two_phi_l63_6300

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_l63_6300


namespace NUMINAMATH_CALUDE_f_at_5_l63_6368

def f (x : ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem f_at_5 : f 5 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_f_at_5_l63_6368


namespace NUMINAMATH_CALUDE_abc_is_246_l63_6351

/-- Represents a base-8 number with two digits --/
def BaseEight (a b : ℕ) : ℕ := 8 * a + b

/-- Converts a three-digit number to its decimal representation --/
def ToDecimal (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem abc_is_246 (A B C : ℕ) 
  (h1 : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
  (h2 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h3 : A < 8 ∧ B < 8)
  (h4 : C < 6)
  (h5 : BaseEight A B + C = BaseEight C 2)
  (h6 : BaseEight A B + BaseEight B A = BaseEight C C) :
  ToDecimal A B C = 246 := by
  sorry

end NUMINAMATH_CALUDE_abc_is_246_l63_6351


namespace NUMINAMATH_CALUDE_logo_scaling_l63_6331

theorem logo_scaling (w h W : ℝ) (hw : w > 0) (hh : h > 0) (hW : W > 0) :
  let scale := W / w
  let H := scale * h
  (W / w = H / h) ∧ (H = (W / w) * h) := by sorry

end NUMINAMATH_CALUDE_logo_scaling_l63_6331


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l63_6366

theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (1, 2) →
  b = (-1, x) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l63_6366


namespace NUMINAMATH_CALUDE_u_2023_equals_4_l63_6356

-- Define the function f
def f : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 2
| 4 => 1
| 5 => 4
| _ => 0  -- Default case for completeness

-- Define the sequence u
def u : ℕ → ℕ
| 0 => 5  -- u₀ = 5
| n + 1 => f (u n)  -- uₙ₊₁ = f(uₙ) for n ≥ 0

-- Theorem statement
theorem u_2023_equals_4 : u 2023 = 4 := by
  sorry

end NUMINAMATH_CALUDE_u_2023_equals_4_l63_6356


namespace NUMINAMATH_CALUDE_unique_last_digit_for_divisibility_by_6_l63_6304

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

def replace_last_digit (n : ℕ) (d : ℕ) : ℕ := (n / 10) * 10 + d

theorem unique_last_digit_for_divisibility_by_6 :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_6 (replace_last_digit 314270 d) ↔ d = last_digit 314274) :=
sorry

end NUMINAMATH_CALUDE_unique_last_digit_for_divisibility_by_6_l63_6304


namespace NUMINAMATH_CALUDE_field_width_calculation_l63_6360

/-- A rectangular football field with given dimensions and running conditions. -/
structure FootballField where
  length : ℝ
  width : ℝ
  laps : ℕ
  total_distance : ℝ

/-- The width of a football field given specific conditions. -/
def field_width (f : FootballField) : ℝ :=
  f.width

/-- Theorem stating the width of the field under given conditions. -/
theorem field_width_calculation (f : FootballField)
  (h1 : f.length = 100)
  (h2 : f.laps = 6)
  (h3 : f.total_distance = 1800)
  (h4 : f.total_distance = f.laps * (2 * f.length + 2 * f.width)) :
  field_width f = 50 := by
  sorry

#check field_width_calculation

end NUMINAMATH_CALUDE_field_width_calculation_l63_6360


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l63_6382

theorem arithmetic_mean_problem (a₁ a₂ a₃ a₄ a₅ a₆ A : ℝ) 
  (h_mean : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = A)
  (h_first_four : (a₁ + a₂ + a₃ + a₄) / 4 = A + 10)
  (h_last_four : (a₃ + a₄ + a₅ + a₆) / 4 = A - 7) :
  (a₁ + a₂ + a₅ + a₆) / 4 = A - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l63_6382


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l63_6302

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_equation_solutions :
  ∀ a b c : ℕ+,
    (factorial a.val + factorial b.val = 2^(factorial c.val)) ↔
    ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 2, 2)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l63_6302


namespace NUMINAMATH_CALUDE_max_sum_red_green_balls_l63_6325

theorem max_sum_red_green_balls :
  ∀ (total red green blue : ℕ),
    total = 28 →
    green = 12 →
    red + green + blue = total →
    red ≤ 11 →
    red + green ≤ 23 ∧ ∃ (red' : ℕ), red' ≤ 11 ∧ red' + green = 23 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_red_green_balls_l63_6325


namespace NUMINAMATH_CALUDE_cinema_selection_is_systematic_sampling_l63_6365

/-- Represents a cinema with a specific number of rows and seats per row. -/
structure Cinema where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a sampling method. -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | WithReplacement

/-- Represents the selection of seats in a cinema. -/
structure SeatSelection where
  cinema : Cinema
  seatNumber : Nat

/-- Determines if a sampling method is systematic based on the seat selection. -/
def isSystematicSampling (selection : SeatSelection) : Prop :=
  selection.cinema.rows > 0 ∧
  selection.cinema.seatsPerRow > 0 ∧
  selection.seatNumber < selection.cinema.seatsPerRow

/-- Theorem stating that the given seat selection is an example of systematic sampling. -/
theorem cinema_selection_is_systematic_sampling 
  (cinema : Cinema)
  (selection : SeatSelection)
  (h1 : cinema.rows = 50)
  (h2 : cinema.seatsPerRow = 60)
  (h3 : selection.seatNumber = 18)
  (h4 : selection.cinema = cinema) :
  isSystematicSampling selection ∧ 
  SamplingMethod.Systematic = SamplingMethod.Systematic :=
sorry


end NUMINAMATH_CALUDE_cinema_selection_is_systematic_sampling_l63_6365


namespace NUMINAMATH_CALUDE_raft_distance_l63_6384

/-- Given a motorboat that travels downstream and upstream in equal time,
    this theorem proves the distance a raft travels with the stream. -/
theorem raft_distance (t : ℝ) (vb vs : ℝ) : t > 0 →
  (vb + vs) * t = 90 →
  (vb - vs) * t = 70 →
  vs * t = 10 := by
  sorry

end NUMINAMATH_CALUDE_raft_distance_l63_6384


namespace NUMINAMATH_CALUDE_rectangular_garden_length_l63_6363

/-- The length of a rectangular garden with perimeter 900 m and breadth 190 m is 260 m. -/
theorem rectangular_garden_length : 
  ∀ (length breadth : ℝ),
  breadth = 190 →
  2 * (length + breadth) = 900 →
  length = 260 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_length_l63_6363


namespace NUMINAMATH_CALUDE_enrollment_increase_l63_6393

theorem enrollment_increase (e1991 e1992 e1993 : ℝ) 
  (h1 : e1993 = e1991 * (1 + 0.38))
  (h2 : e1993 = e1992 * (1 + 0.15)) :
  e1992 = e1991 * (1 + 0.2) := by
  sorry

end NUMINAMATH_CALUDE_enrollment_increase_l63_6393


namespace NUMINAMATH_CALUDE_first_month_sale_l63_6347

def average_sale : ℕ := 5500
def month2_sale : ℕ := 5927
def month3_sale : ℕ := 5855
def month4_sale : ℕ := 6230
def month5_sale : ℕ := 5562
def month6_sale : ℕ := 3991

theorem first_month_sale :
  let total_sale := 6 * average_sale
  let known_sales := month2_sale + month3_sale + month4_sale + month5_sale + month6_sale
  total_sale - known_sales = 5435 := by
sorry

end NUMINAMATH_CALUDE_first_month_sale_l63_6347


namespace NUMINAMATH_CALUDE_inscribed_square_area_ratio_l63_6336

/-- A circle with a square inscribed in it, where the square's vertices touch the circle
    and the side of the square intersects the circle such that each intersection segment
    equals twice the radius of the circle. -/
structure InscribedSquare where
  r : ℝ  -- radius of the circle
  s : ℝ  -- side length of the square
  h1 : s = r * Real.sqrt 2  -- relationship between side length and radius
  h2 : s * Real.sqrt 2 = 2 * r  -- diagonal of square equals diameter of circle

/-- The ratio of the area of the inscribed square to the area of the circle is 2/π. -/
theorem inscribed_square_area_ratio (square : InscribedSquare) :
  (square.s ^ 2) / (Real.pi * square.r ^ 2) = 2 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_ratio_l63_6336


namespace NUMINAMATH_CALUDE_number_of_office_workers_l63_6367

/-- Proves the number of office workers in company J --/
theorem number_of_office_workers :
  let factory_workers : ℕ := 15
  let factory_payroll : ℕ := 30000
  let office_payroll : ℕ := 75000
  let salary_difference : ℕ := 500
  let factory_avg_salary : ℕ := factory_payroll / factory_workers
  let office_avg_salary : ℕ := factory_avg_salary + salary_difference
  office_payroll / office_avg_salary = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_of_office_workers_l63_6367


namespace NUMINAMATH_CALUDE_equal_sums_exist_l63_6383

/-- Represents a 3x3 grid with values from {-1, 0, 1} -/
def Grid := Matrix (Fin 3) (Fin 3) (Fin 3)

/-- Computes the sum of a row in the grid -/
def rowSum (g : Grid) (i : Fin 3) : ℤ := sorry

/-- Computes the sum of a column in the grid -/
def colSum (g : Grid) (j : Fin 3) : ℤ := sorry

/-- Computes the sum of the main diagonal -/
def mainDiagSum (g : Grid) : ℤ := sorry

/-- Computes the sum of the anti-diagonal -/
def antiDiagSum (g : Grid) : ℤ := sorry

/-- All possible sums in the grid -/
def allSums (g : Grid) : List ℤ := 
  [rowSum g 0, rowSum g 1, rowSum g 2, 
   colSum g 0, colSum g 1, colSum g 2, 
   mainDiagSum g, antiDiagSum g]

theorem equal_sums_exist (g : Grid) : 
  ∃ (i j : Fin 8), i ≠ j ∧ (allSums g).get i = (allSums g).get j := by sorry

end NUMINAMATH_CALUDE_equal_sums_exist_l63_6383


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l63_6385

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l63_6385


namespace NUMINAMATH_CALUDE_valid_assignment_d_plus_5_l63_6376

/-- Represents a programming language variable --/
structure Variable where
  name : String

/-- Represents a programming language expression --/
inductive Expression where
  | Var : Variable → Expression
  | Const : Int → Expression
  | Add : Expression → Expression → Expression

/-- Represents an assignment statement --/
structure Assignment where
  lhs : Variable
  rhs : Expression

/-- Predicate to check if an assignment is valid --/
def is_valid_assignment (a : Assignment) : Prop :=
  ∃ (d : Variable), a.lhs = d ∧ 
    a.rhs = Expression.Add (Expression.Var d) (Expression.Const 5)

/-- Theorem stating that "d = d + 5" is a valid assignment --/
theorem valid_assignment_d_plus_5 :
  ∃ (a : Assignment), is_valid_assignment a :=
sorry

end NUMINAMATH_CALUDE_valid_assignment_d_plus_5_l63_6376


namespace NUMINAMATH_CALUDE_delaware_cell_phones_count_l63_6352

/-- The number of cell phones in Delaware -/
def delaware_cell_phones (population : ℕ) (phones_per_thousand : ℕ) : ℕ :=
  (population / 1000) * phones_per_thousand

/-- Proof that the number of cell phones in Delaware is 655,502 -/
theorem delaware_cell_phones_count :
  delaware_cell_phones 974000 673 = 655502 := by
  sorry

end NUMINAMATH_CALUDE_delaware_cell_phones_count_l63_6352


namespace NUMINAMATH_CALUDE_berries_to_buy_l63_6310

def total_needed : Nat := 21
def strawberries : Nat := 4
def blueberries : Nat := 8

theorem berries_to_buy (total_needed strawberries blueberries : Nat) : 
  total_needed - (strawberries + blueberries) = 9 :=
by sorry

end NUMINAMATH_CALUDE_berries_to_buy_l63_6310


namespace NUMINAMATH_CALUDE_consecutive_even_ages_l63_6342

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem consecutive_even_ages (a b c : ℕ) 
  (h1 : is_even a)
  (h2 : is_even b)
  (h3 : is_even c)
  (h4 : b = a + 2)
  (h5 : c = b + 2)
  (h6 : a + b + c = 48) :
  a = 14 ∧ c = 18 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_ages_l63_6342


namespace NUMINAMATH_CALUDE_toms_original_portion_l63_6388

theorem toms_original_portion (tom uma vicky : ℝ) : 
  tom + uma + vicky = 2000 →
  (tom - 200) + 3 * uma + 3 * vicky = 3500 →
  tom = 1150 := by
sorry

end NUMINAMATH_CALUDE_toms_original_portion_l63_6388


namespace NUMINAMATH_CALUDE_root_one_when_sum_zero_reciprocal_roots_l63_6392

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Theorem 1: If a + b + c = 0, then x = 1 is a root
theorem root_one_when_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hsum : a + b + c = 0) :
  quadratic a b c 1 := by sorry

-- Theorem 2: If x1 and x2 are roots of ax^2 + bx + c = 0 where x1 ≠ x2 ≠ 0,
-- then 1/x1 and 1/x2 are roots of cx^2 + bx + a = 0 (c ≠ 0)
theorem reciprocal_roots (a b c x1 x2 : ℝ) (ha : a ≠ 0) (hc : c ≠ 0)
  (hx1 : x1 ≠ 0) (hx2 : x2 ≠ 0) (hx1x2 : x1 ≠ x2)
  (hroot1 : quadratic a b c x1) (hroot2 : quadratic a b c x2) :
  quadratic c b a (1/x1) ∧ quadratic c b a (1/x2) := by sorry

end NUMINAMATH_CALUDE_root_one_when_sum_zero_reciprocal_roots_l63_6392


namespace NUMINAMATH_CALUDE_radford_distance_at_finish_l63_6326

/-- Represents the race between Radford and Peter -/
structure Race where
  radford_initial_lead : ℝ
  peter_lead_after_3min : ℝ
  race_duration : ℝ
  peter_speed_advantage : ℝ

/-- Calculates the distance between Radford and Peter at the end of the race -/
def final_distance (race : Race) : ℝ :=
  race.peter_lead_after_3min + race.peter_speed_advantage * (race.race_duration - 3)

/-- Theorem stating that Radford is 82 meters behind Peter at the end of the race -/
theorem radford_distance_at_finish (race : Race) 
  (h1 : race.radford_initial_lead = 30)
  (h2 : race.peter_lead_after_3min = 18)
  (h3 : race.race_duration = 7)
  (h4 : race.peter_speed_advantage = 16) :
  final_distance race = 82 := by
  sorry

end NUMINAMATH_CALUDE_radford_distance_at_finish_l63_6326


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l63_6324

/-- The value of p for which the axis of the parabola y^2 = 2px intersects
    the circle (x+1)^2 + y^2 = 4 at two points with distance 2√3 -/
theorem parabola_circle_intersection (p : ℝ) : p > 0 →
  (∃ A B : ℝ × ℝ,
    (A.1 + 1)^2 + A.2^2 = 4 ∧
    (B.1 + 1)^2 + B.2^2 = 4 ∧
    A.2^2 = 2 * p * A.1 ∧
    B.2^2 = 2 * p * B.1 ∧
    A.1 = B.1 ∧
    (A.2 - B.2)^2 = 12) →
  p = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l63_6324


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_less_than_neg_one_l63_6340

-- Define the function f(x) = ax + 1
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 1

-- State the theorem
theorem unique_solution_implies_a_less_than_neg_one :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_less_than_neg_one_l63_6340


namespace NUMINAMATH_CALUDE_circle_circumference_when_equal_to_area_l63_6313

/-- 
For a circle where the circumference and area are numerically equal,
if the diameter is 4, then the circumference is 4π.
-/
theorem circle_circumference_when_equal_to_area (d : ℝ) (C : ℝ) (A : ℝ) : 
  C = A →  -- Circumference equals area
  d = 4 →  -- Diameter is 4
  C = π * d →  -- Definition of circumference
  A = π * (d/2)^2 →  -- Definition of area
  C = 4 * π := by
sorry

end NUMINAMATH_CALUDE_circle_circumference_when_equal_to_area_l63_6313


namespace NUMINAMATH_CALUDE_boys_share_l63_6349

theorem boys_share (total_amount : ℕ) (total_children : ℕ) (num_boys : ℕ) (amount_per_girl : ℕ) 
  (h1 : total_amount = 460)
  (h2 : total_children = 41)
  (h3 : num_boys = 33)
  (h4 : amount_per_girl = 8) :
  (total_amount - (total_children - num_boys) * amount_per_girl) / num_boys = 12 := by
  sorry

end NUMINAMATH_CALUDE_boys_share_l63_6349


namespace NUMINAMATH_CALUDE_vector_problem_l63_6379

-- Define the vectors
def OA (k : ℝ) : ℝ × ℝ := (k, 12)
def OB : ℝ × ℝ := (4, 5)
def OC (k : ℝ) : ℝ × ℝ := (-k, 10)

-- Define collinearity
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, B - A = t • (C - A)

-- State the theorem
theorem vector_problem (k : ℝ) :
  collinear (OA k) OB (OC k) → k = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l63_6379


namespace NUMINAMATH_CALUDE_parallel_vectors_dot_product_l63_6370

/-- Given vectors a and b in ℝ², where a is parallel to b, prove their dot product is -5 -/
theorem parallel_vectors_dot_product (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, x - 1]
  let b : Fin 2 → ℝ := ![1, 2]
  (∃ (k : ℝ), a = k • b) →
  (a 0 * b 0 + a 1 * b 1 = -5) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_dot_product_l63_6370


namespace NUMINAMATH_CALUDE_prob_six_diff_tens_digits_l63_6333

/-- The probability of selecting 6 different integers between 10 and 99 (inclusive) 
    with different tens digits -/
def prob_diff_tens_digits : ℚ :=
  8000 / 5895

/-- The number of integers between 10 and 99, inclusive -/
def total_integers : ℕ := 90

/-- The number of possible tens digits -/
def num_tens_digits : ℕ := 9

/-- The number of integers to be selected -/
def num_selected : ℕ := 6

/-- The number of integers for each tens digit -/
def integers_per_tens : ℕ := 10

theorem prob_six_diff_tens_digits :
  prob_diff_tens_digits = 
    (Nat.choose num_tens_digits num_selected * integers_per_tens ^ num_selected) / 
    Nat.choose total_integers num_selected :=
sorry

end NUMINAMATH_CALUDE_prob_six_diff_tens_digits_l63_6333


namespace NUMINAMATH_CALUDE_simplify_fraction_simplify_harmonic_root1_simplify_harmonic_root2_calculate_expression_l63_6308

-- 1. Simplify fraction with square root
theorem simplify_fraction : (2 : ℝ) / (Real.sqrt 3 - 1) = Real.sqrt 3 + 1 := by sorry

-- 2. Simplify harmonic quadratic root (case 1)
theorem simplify_harmonic_root1 : Real.sqrt (4 + 2 * Real.sqrt 3) = Real.sqrt 3 + 1 := by sorry

-- 3. Simplify harmonic quadratic root (case 2)
theorem simplify_harmonic_root2 : Real.sqrt (6 - 2 * Real.sqrt 5) = Real.sqrt 5 - 1 := by sorry

-- 4. Calculate expression with harmonic quadratic roots
theorem calculate_expression (m n : ℝ) 
  (hm : m = 1 / Real.sqrt (5 + 2 * Real.sqrt 6))
  (hn : n = 1 / Real.sqrt (5 - 2 * Real.sqrt 6)) :
  (m - n) / (m + n) = -(Real.sqrt 6) / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_simplify_harmonic_root1_simplify_harmonic_root2_calculate_expression_l63_6308


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_increase_l63_6369

theorem equilateral_triangle_area_increase :
  ∀ s : ℝ,
  s > 0 →
  (s^2 * Real.sqrt 3) / 4 = 36 * Real.sqrt 3 →
  let new_s := s + 2
  let new_area := (new_s^2 * Real.sqrt 3) / 4
  let original_area := (s^2 * Real.sqrt 3) / 4
  new_area - original_area = 13 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_increase_l63_6369


namespace NUMINAMATH_CALUDE_octagon_diagonals_l63_6386

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 vertices -/
def octagon_vertices : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_vertices = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l63_6386


namespace NUMINAMATH_CALUDE_new_pressure_is_two_l63_6380

/-- Represents the pressure-volume relationship at constant temperature -/
structure GasState where
  pressure : ℝ
  volume : ℝ
  constant : ℝ

/-- The pressure-volume relationship is inversely proportional -/
axiom pressure_volume_constant (state : GasState) : state.pressure * state.volume = state.constant

/-- Initial state of the gas -/
def initial_state : GasState :=
  { pressure := 4
    volume := 3
    constant := 4 * 3 }

/-- New state of the gas after transfer -/
def new_state : GasState :=
  { pressure := 2  -- This is what we want to prove
    volume := 6
    constant := initial_state.constant }

/-- Theorem stating that the new pressure is 2 kPa -/
theorem new_pressure_is_two :
  new_state.pressure = 2 := by sorry

end NUMINAMATH_CALUDE_new_pressure_is_two_l63_6380


namespace NUMINAMATH_CALUDE_sum_of_ages_l63_6387

/-- Given the ages and relationships of Beckett, Olaf, Shannen, and Jack, prove that the sum of their ages is 71 years. -/
theorem sum_of_ages (beckett_age olaf_age shannen_age jack_age : ℕ) : 
  beckett_age = 12 →
  olaf_age = beckett_age + 3 →
  shannen_age = olaf_age - 2 →
  jack_age = 2 * shannen_age + 5 →
  beckett_age + olaf_age + shannen_age + jack_age = 71 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l63_6387


namespace NUMINAMATH_CALUDE_sum_of_three_integers_l63_6374

theorem sum_of_three_integers (large medium small : ℕ+) 
  (sum_large_medium : large + medium = 2003)
  (diff_medium_small : medium - small = 1000) :
  large + medium + small = 2004 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_integers_l63_6374


namespace NUMINAMATH_CALUDE_insurance_coverage_percentage_l63_6353

def mri_cost : ℝ := 1200
def doctor_rate : ℝ := 300
def doctor_time : ℝ := 0.5
def fee_for_seen : ℝ := 150
def tim_payment : ℝ := 300

def total_cost : ℝ := mri_cost + doctor_rate * doctor_time + fee_for_seen

def insurance_coverage : ℝ := total_cost - tim_payment

theorem insurance_coverage_percentage : 
  insurance_coverage / total_cost * 100 = 80 := by sorry

end NUMINAMATH_CALUDE_insurance_coverage_percentage_l63_6353


namespace NUMINAMATH_CALUDE_percentage_of_hindu_boys_l63_6306

theorem percentage_of_hindu_boys (total : ℕ) (muslim_percent : ℚ) (sikh_percent : ℚ) (other : ℕ) :
  total = 300 →
  muslim_percent = 44 / 100 →
  sikh_percent = 10 / 100 →
  other = 54 →
  (total - (muslim_percent * total + sikh_percent * total + other)) / total = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_hindu_boys_l63_6306


namespace NUMINAMATH_CALUDE_remaining_red_cards_l63_6396

theorem remaining_red_cards (total_cards : ℕ) (red_cards : ℕ) (removed_cards : ℕ) : 
  total_cards = 52 → 
  red_cards = total_cards / 2 →
  removed_cards = 10 →
  red_cards - removed_cards = 16 := by
  sorry

end NUMINAMATH_CALUDE_remaining_red_cards_l63_6396


namespace NUMINAMATH_CALUDE_max_winner_number_l63_6378

/-- Represents a wrestler in the tournament -/
structure Wrestler :=
  (number : ℕ)

/-- The tournament setup -/
def Tournament :=
  { wrestlers : Finset Wrestler // wrestlers.card = 512 }

/-- Predicate for the winning condition in a match -/
def wins (w1 w2 : Wrestler) : Prop :=
  w1.number < w2.number ∧ w2.number - w1.number > 2

/-- The winner of the tournament -/
def tournamentWinner (t : Tournament) : Wrestler :=
  sorry

/-- Theorem stating the maximum possible qualification number of the winner -/
theorem max_winner_number (t : Tournament) : 
  (tournamentWinner t).number ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_max_winner_number_l63_6378


namespace NUMINAMATH_CALUDE_chinese_chess_draw_probability_l63_6339

/-- Given the probabilities of winning and not losing for player A in Chinese chess,
    calculate the probability of a draw between player A and player B. -/
theorem chinese_chess_draw_probability
  (prob_win : ℝ) (prob_not_lose : ℝ)
  (h_win : prob_win = 0.4)
  (h_not_lose : prob_not_lose = 0.9) :
  prob_not_lose - prob_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chinese_chess_draw_probability_l63_6339


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l63_6359

/-- Calculate the interest rate given the principal, time, and total interest for a simple interest loan. -/
theorem simple_interest_rate_calculation
  (principal : ℝ)
  (time : ℝ)
  (total_interest : ℝ)
  (h_principal : principal = 5000)
  (h_time : time = 10)
  (h_total_interest : total_interest = 2000) :
  (total_interest * 100) / (principal * time) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l63_6359


namespace NUMINAMATH_CALUDE_die_rolls_for_most_likely_32_twos_l63_6354

/-- The number of rolls needed for the most likely number of twos to be 32 -/
theorem die_rolls_for_most_likely_32_twos :
  ∃ n : ℕ, 191 ≤ n ∧ n ≤ 197 ∧
  (∀ k : ℕ, (Nat.choose n k * (1/6)^k * (5/6)^(n-k)) ≤ (Nat.choose n 32 * (1/6)^32 * (5/6)^(n-32))) :=
by sorry

end NUMINAMATH_CALUDE_die_rolls_for_most_likely_32_twos_l63_6354


namespace NUMINAMATH_CALUDE_tricycle_count_l63_6391

/-- Represents the number of vehicles of each type -/
structure VehicleCounts where
  bicycles : ℕ
  tricycles : ℕ
  scooters : ℕ

/-- The total number of children -/
def totalChildren : ℕ := 10

/-- The total number of wheels -/
def totalWheels : ℕ := 25

/-- Calculates the total number of children given the vehicle counts -/
def countChildren (v : VehicleCounts) : ℕ :=
  v.bicycles + v.tricycles + v.scooters

/-- Calculates the total number of wheels given the vehicle counts -/
def countWheels (v : VehicleCounts) : ℕ :=
  2 * v.bicycles + 3 * v.tricycles + v.scooters

/-- Theorem stating that the number of tricycles is 5 -/
theorem tricycle_count :
  ∃ (v : VehicleCounts),
    countChildren v = totalChildren ∧
    countWheels v = totalWheels ∧
    v.tricycles = 5 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_count_l63_6391


namespace NUMINAMATH_CALUDE_sector_forms_cone_l63_6398

/-- Represents a circular sector -/
structure CircularSector where
  radius : ℝ
  angle : ℝ

/-- Represents a cone -/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ

/-- Given a circular sector, returns the cone formed by aligning its straight sides -/
def sectorToCone (sector : CircularSector) : Cone :=
  sorry

theorem sector_forms_cone :
  let sector : CircularSector := ⟨12, 270 * π / 180⟩
  let cone : Cone := sectorToCone sector
  cone.baseRadius = 9 ∧ cone.slantHeight = 12 := by
  sorry

end NUMINAMATH_CALUDE_sector_forms_cone_l63_6398


namespace NUMINAMATH_CALUDE_negation_equivalence_l63_6390

theorem negation_equivalence (x : ℝ) :
  ¬(x = 0 ∨ x = 1 → x^2 - x = 0) ↔ (x ≠ 0 ∧ x ≠ 1 → x^2 - x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l63_6390


namespace NUMINAMATH_CALUDE_emma_numbers_l63_6364

theorem emma_numbers (x y : ℤ) : 
  4 * x + 3 * y = 140 → (x = 20 ∨ y = 20) → x = 20 ∧ y = 20 := by
sorry

end NUMINAMATH_CALUDE_emma_numbers_l63_6364


namespace NUMINAMATH_CALUDE_identify_all_pairs_in_75_attempts_l63_6309

/-- Represents a door-key system with 100 doors and keys -/
structure DoorKeySystem :=
  (doors : Fin 100 → Nat)
  (keys : Fin 100 → Nat)
  (key_matches : ∀ i : Fin 100, (keys i = doors i) ∨ (keys i = doors i + 1) ∨ (keys i + 1 = doors i))

/-- Represents an attempt to match a key to a door -/
def Attempt := Fin 100 × Fin 100

/-- A function that determines if all key-door pairs can be identified within a given number of attempts -/
def can_identify_all_pairs (system : DoorKeySystem) (max_attempts : Nat) : Prop :=
  ∃ (attempts : List Attempt), 
    attempts.length ≤ max_attempts ∧ 
    (∀ i : Fin 100, ∃ j : Fin 100, (i, j) ∈ attempts ∨ (j, i) ∈ attempts) ∧
    (∀ i j : Fin 100, system.keys i = system.doors j → (i, j) ∈ attempts ∨ (j, i) ∈ attempts)

/-- Theorem stating that all key-door pairs can be identified within 75 attempts -/
theorem identify_all_pairs_in_75_attempts :
  ∀ system : DoorKeySystem, can_identify_all_pairs system 75 :=
sorry

end NUMINAMATH_CALUDE_identify_all_pairs_in_75_attempts_l63_6309


namespace NUMINAMATH_CALUDE_circle_diameter_l63_6328

theorem circle_diameter (C : ℝ) (h : C = 100) : C / π = 100 / π := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l63_6328


namespace NUMINAMATH_CALUDE_region_area_theorem_l63_6301

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bounded by two circles and the x-axis -/
def areaRegion (c1 c2 : Circle) : ℝ :=
  sorry

theorem region_area_theorem (c1 c2 : Circle) 
  (h1 : c1.center = (3, 5) ∧ c1.radius = 5)
  (h2 : c2.center = (13, 5) ∧ c2.radius = 5) : 
  areaRegion c1 c2 = 50 - 12.5 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_region_area_theorem_l63_6301


namespace NUMINAMATH_CALUDE_vector_sum_proof_l63_6397

def vector1 : Fin 2 → ℝ := ![5, -3]
def vector2 : Fin 2 → ℝ := ![-4, 6]
def vector3 : Fin 2 → ℝ := ![2, -8]

theorem vector_sum_proof :
  vector1 + vector2 + vector3 = ![3, -5] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l63_6397


namespace NUMINAMATH_CALUDE_remainder_of_2615_base12_div_9_l63_6381

/-- Converts a base-12 digit to its decimal equivalent -/
def base12ToDecimal (digit : ℕ) : ℕ := digit

/-- Calculates the decimal value of a base-12 number given its digits -/
def base12Value (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  base12ToDecimal d₃ * 12^3 + base12ToDecimal d₂ * 12^2 + 
  base12ToDecimal d₁ * 12^1 + base12ToDecimal d₀ * 12^0

/-- The base-12 number 2615₁₂ -/
def num : ℕ := base12Value 2 6 1 5

theorem remainder_of_2615_base12_div_9 :
  num % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_of_2615_base12_div_9_l63_6381


namespace NUMINAMATH_CALUDE_ellipse_range_and_logical_conditions_l63_6321

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (m + 1) + y^2 / (3 - m) = 1) → 
  (∃ a b : ℝ, a > b ∧ a^2 - b^2 = 3 - m - (m + 1) ∧ 
  ∀ t : ℝ, x^2 / (m + 1) + y^2 / (3 - m) = 1 → 
  (x = 0 → y^2 ≤ a^2) ∧ (y = 0 → x^2 ≤ b^2))

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 2*m + 3 ≠ 0

theorem ellipse_range_and_logical_conditions (m : ℝ) :
  (p m ↔ -1 < m ∧ m < 1) ∧
  ((¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ 1 ≤ m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_range_and_logical_conditions_l63_6321


namespace NUMINAMATH_CALUDE_series_sum_equals_one_l63_6399

open Real

noncomputable def seriesSum : ℝ := ∑' k, (k^2 : ℝ) / 3^k

theorem series_sum_equals_one : seriesSum = 1 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_l63_6399


namespace NUMINAMATH_CALUDE_matchsticks_left_six_matchsticks_left_l63_6345

/-- Calculates the number of matchsticks left after Elvis and Ralph create their squares --/
theorem matchsticks_left (total : ℕ) (elvis_max : ℕ) (ralph_max : ℕ) 
  (elvis_per_square : ℕ) (ralph_per_square : ℕ) : ℕ :=
  let elvis_squares := elvis_max / elvis_per_square
  let ralph_squares := ralph_max / ralph_per_square
  let elvis_used := elvis_squares * elvis_per_square
  let ralph_used := ralph_squares * ralph_per_square
  total - (elvis_used + ralph_used)

/-- Proves that 6 matchsticks are left under the given conditions --/
theorem six_matchsticks_left : 
  matchsticks_left 50 20 30 4 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_matchsticks_left_six_matchsticks_left_l63_6345


namespace NUMINAMATH_CALUDE_sugar_sacks_weight_l63_6337

theorem sugar_sacks_weight (x y : ℝ) 
  (h1 : y - x = 8)
  (h2 : x - 1 = 0.6 * (y + 1)) : 
  x + y = 40 := by
sorry

end NUMINAMATH_CALUDE_sugar_sacks_weight_l63_6337


namespace NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l63_6377

theorem range_of_a_for_false_proposition :
  {a : ℝ | ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + 2*a*x₀ + 2*a + 3 < 0} = Set.Ioi (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l63_6377


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l63_6373

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 5*x - 14 > 0) ∧
  (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 5*x - 14)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l63_6373


namespace NUMINAMATH_CALUDE_polynomial_remainder_l63_6323

theorem polynomial_remainder (x : ℝ) : 
  (5 * x^3 - 9 * x^2 + 3 * x + 17) % (x - 2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l63_6323


namespace NUMINAMATH_CALUDE_partition_16_into_8_pairs_eq_2027025_l63_6307

/-- The number of ways to partition 16 distinct elements into 8 unordered pairs -/
def partition_16_into_8_pairs : ℕ :=
  (Nat.factorial 16) / (Nat.pow 2 8 * Nat.factorial 8)

/-- Theorem stating that the number of ways to partition 16 distinct elements
    into 8 unordered pairs is equal to 2027025 -/
theorem partition_16_into_8_pairs_eq_2027025 :
  partition_16_into_8_pairs = 2027025 := by
  sorry

end NUMINAMATH_CALUDE_partition_16_into_8_pairs_eq_2027025_l63_6307


namespace NUMINAMATH_CALUDE_expression_evaluation_l63_6341

theorem expression_evaluation :
  let f (x : ℝ) := 2 * x^2 + 3 * x - 4
  f 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l63_6341


namespace NUMINAMATH_CALUDE_bob_initial_pennies_l63_6319

theorem bob_initial_pennies :
  ∀ (a b : ℕ),
  (b + 2 = 4 * (a - 2)) →
  (b - 2 = 3 * (a + 2)) →
  b = 62 := by
  sorry

end NUMINAMATH_CALUDE_bob_initial_pennies_l63_6319


namespace NUMINAMATH_CALUDE_restaurant_group_size_l63_6317

/-- Calculates the total number of people in a restaurant group given the following conditions:
  * The cost of an adult meal is $7
  * Kids eat for free
  * There are 9 kids in the group
  * The total cost for the group is $28
-/
theorem restaurant_group_size :
  let adult_meal_cost : ℕ := 7
  let kids_count : ℕ := 9
  let total_cost : ℕ := 28
  let adult_count : ℕ := total_cost / adult_meal_cost
  let total_people : ℕ := adult_count + kids_count
  total_people = 13 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_size_l63_6317


namespace NUMINAMATH_CALUDE_fraction_addition_l63_6357

theorem fraction_addition (d : ℝ) : (6 + 5*d) / 9 + 3 = (33 + 5*d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l63_6357


namespace NUMINAMATH_CALUDE_fractional_part_inequality_l63_6320

theorem fractional_part_inequality (α : ℝ) (h_α : 0 < α ∧ α < 1) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ ∀ n : ℕ+, α^(n : ℝ) < (n : ℝ) * x - ⌊(n : ℝ) * x⌋ := by
  sorry

end NUMINAMATH_CALUDE_fractional_part_inequality_l63_6320


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l63_6394

/-- A structure made of unit cubes -/
structure CubeStructure where
  /-- The number of unit cubes in the structure -/
  num_cubes : ℕ
  /-- The volume of the structure in cubic units -/
  volume : ℕ
  /-- The surface area of the structure in square units -/
  surface_area : ℕ
  /-- The structure has a central cube surrounded symmetrically on all faces except the bottom -/
  has_central_cube : Prop
  /-- The structure forms a large plus sign when viewed from the top -/
  is_plus_shaped : Prop

/-- The specific cube structure described in the problem -/
def plus_structure : CubeStructure :=
  { num_cubes := 9
  , volume := 9
  , surface_area := 31
  , has_central_cube := True
  , is_plus_shaped := True }

/-- The theorem stating that the ratio of volume to surface area for the plus_structure is 9/31 -/
theorem volume_to_surface_area_ratio (s : CubeStructure) (h1 : s = plus_structure) :
  (s.volume : ℚ) / s.surface_area = 9 / 31 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l63_6394


namespace NUMINAMATH_CALUDE_people_visited_neither_l63_6372

theorem people_visited_neither (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 100 →
  iceland = 55 →
  norway = 43 →
  both = 61 →
  total - (iceland + norway - both) = 63 := by
  sorry

end NUMINAMATH_CALUDE_people_visited_neither_l63_6372


namespace NUMINAMATH_CALUDE_geometric_mean_of_1_and_9_l63_6332

theorem geometric_mean_of_1_and_9 : 
  ∃ (x : ℝ), x^2 = 1 * 9 ∧ (x = 3 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_1_and_9_l63_6332


namespace NUMINAMATH_CALUDE_sum_of_roots_of_unity_l63_6350

def is_root_of_unity (z : ℂ) : Prop := ∃ n : ℕ, n > 0 ∧ z^n = 1

theorem sum_of_roots_of_unity (x y z : ℂ) :
  is_root_of_unity x ∧ is_root_of_unity y ∧ is_root_of_unity z →
  (is_root_of_unity (x + y + z) ↔ (x + y = 0 ∨ y + z = 0 ∨ z + x = 0)) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_unity_l63_6350


namespace NUMINAMATH_CALUDE_min_students_for_duplicate_vote_l63_6362

theorem min_students_for_duplicate_vote (n : ℕ) (h : n = 10) :
  let combinations := n.choose 2
  ∃ k : ℕ, k > combinations ∧
    ∀ m : ℕ, m < k → ∃ f : Fin m → Fin n × Fin n,
      Function.Injective f ∧
      ∀ i : Fin m, (f i).1 < (f i).2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_students_for_duplicate_vote_l63_6362


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l63_6335

def total_socks : ℕ := 7
def blue_socks : ℕ := 2
def other_socks : ℕ := 5
def socks_to_choose : ℕ := 4

def valid_combinations : ℕ := 30

theorem sock_selection_theorem :
  (Nat.choose blue_socks 2 * Nat.choose other_socks 2) +
  (Nat.choose blue_socks 2 * Nat.choose other_socks 1) +
  (Nat.choose blue_socks 1 * Nat.choose other_socks 2) = valid_combinations :=
by sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l63_6335


namespace NUMINAMATH_CALUDE_simplify_expression_l63_6343

theorem simplify_expression :
  (Real.sqrt 2 + 1) ^ (1 - Real.sqrt 3) / (Real.sqrt 2 - 1) ^ (1 + Real.sqrt 3) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l63_6343


namespace NUMINAMATH_CALUDE_set_intersection_example_l63_6311

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 4}

theorem set_intersection_example : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_example_l63_6311


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l63_6329

/-- Sarah's bowling score problem -/
theorem sarahs_bowling_score :
  ∀ (sarah greg : ℕ),
  sarah = greg + 60 →
  sarah + greg = 260 →
  sarah = 160 := by
sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l63_6329


namespace NUMINAMATH_CALUDE_factory_working_days_l63_6371

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 4340

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 2170

/-- The number of working days per week -/
def working_days : ℕ := toys_per_week / toys_per_day

theorem factory_working_days : working_days = 2 := by
  sorry

end NUMINAMATH_CALUDE_factory_working_days_l63_6371


namespace NUMINAMATH_CALUDE_magnet_cost_is_three_l63_6330

/-- The cost of the magnet at the garage sale -/
def magnet_cost (stuffed_animal_cost : ℚ) : ℚ :=
  (2 * stuffed_animal_cost) / 4

/-- Theorem stating that the magnet cost $3 -/
theorem magnet_cost_is_three :
  magnet_cost 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_magnet_cost_is_three_l63_6330


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l63_6389

theorem arithmetic_evaluation : 5 + 12 / 3 - 3^2 + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l63_6389


namespace NUMINAMATH_CALUDE_geometric_sequence_S_3_range_l63_6334

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the sum of the first three terms
def S_3 (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 3

-- Theorem statement
theorem geometric_sequence_S_3_range
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a2 : a 2 = 1) :
  ∃ y : ℝ, S_3 a = y ↔ y ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_S_3_range_l63_6334


namespace NUMINAMATH_CALUDE_time_difference_1200_miles_l63_6344

/-- Calculates the time difference for a 1200-mile trip between two given speeds -/
theorem time_difference_1200_miles (speed1 speed2 : ℝ) (h1 : speed1 > 0) (h2 : speed2 > 0) :
  (1200 / speed1 - 1200 / speed2) = 4 ↔ speed1 = 60 ∧ speed2 = 50 := by sorry

end NUMINAMATH_CALUDE_time_difference_1200_miles_l63_6344


namespace NUMINAMATH_CALUDE_amc_distinct_scores_l63_6312

/-- Represents the scoring system for an exam -/
structure ScoringSystem where
  totalQuestions : Nat
  correctPoints : Nat
  incorrectPoints : Nat
  unansweredPoints : Nat

/-- Calculates the number of distinct possible scores for a given scoring system -/
def distinctScores (s : ScoringSystem) : Nat :=
  sorry

/-- The AMC exam scoring system -/
def amcScoring : ScoringSystem :=
  { totalQuestions := 30
  , correctPoints := 5
  , incorrectPoints := 0
  , unansweredPoints := 2 }

/-- Theorem stating that the number of distinct possible scores for the AMC exam is 145 -/
theorem amc_distinct_scores : distinctScores amcScoring = 145 := by
  sorry

end NUMINAMATH_CALUDE_amc_distinct_scores_l63_6312


namespace NUMINAMATH_CALUDE_equation_solutions_l63_6355

/-- Given an equation a · b^x · c^(2x) = ∛(d)^(1/x) · ∜(e)^(1/x), 
    this theorem states that:
    1. When a = 2, b = 3, c = 5, d = 7, e = 11, there exist two real solutions.
    2. When a = 5, b = 3, c = 2, d = 1/7, e = 1/11, there are no real solutions. -/
theorem equation_solutions (a b c d e : ℝ) : 
  (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 7 ∧ e = 11 → 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * b^x₁ * c^(2*x₁) = (d^(1/3))^(1/x₁) * (e^(1/4))^(1/x₁) ∧
                   a * b^x₂ * c^(2*x₂) = (d^(1/3))^(1/x₂) * (e^(1/4))^(1/x₂)) ∧
  (a = 5 ∧ b = 3 ∧ c = 2 ∧ d = 1/7 ∧ e = 1/11 → 
    ¬∃ x : ℝ, a * b^x * c^(2*x) = (d^(1/3))^(1/x) * (e^(1/4))^(1/x)) :=
by sorry


end NUMINAMATH_CALUDE_equation_solutions_l63_6355
