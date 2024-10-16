import Mathlib

namespace NUMINAMATH_CALUDE_sum_can_equal_fifty_l272_27239

theorem sum_can_equal_fifty : ∃ (scenario : Type) (sum : scenario → ℝ), ∀ (s : scenario), sum s = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_can_equal_fifty_l272_27239


namespace NUMINAMATH_CALUDE_janet_owes_22000_l272_27248

/-- Calculates the total amount Janet owes for wages and taxes for one month -/
def total_owed (warehouse_workers : ℕ) (managers : ℕ) (warehouse_wage : ℚ) (manager_wage : ℚ)
  (days_per_month : ℕ) (hours_per_day : ℕ) (fica_tax_rate : ℚ) : ℚ :=
  let total_hours := days_per_month * hours_per_day
  let warehouse_total := warehouse_workers * warehouse_wage * total_hours
  let manager_total := managers * manager_wage * total_hours
  let total_wages := warehouse_total + manager_total
  let fica_taxes := total_wages * fica_tax_rate
  total_wages + fica_taxes

theorem janet_owes_22000 :
  total_owed 4 2 15 20 25 8 (1/10) = 22000 := by
  sorry

end NUMINAMATH_CALUDE_janet_owes_22000_l272_27248


namespace NUMINAMATH_CALUDE_total_hamburger_configurations_l272_27247

/-- The number of different condiments available. -/
def num_condiments : ℕ := 10

/-- The number of options for meat patties. -/
def meat_patty_options : ℕ := 4

/-- Theorem: The total number of different hamburger configurations. -/
theorem total_hamburger_configurations :
  (2 ^ num_condiments) * meat_patty_options = 4096 := by
  sorry

end NUMINAMATH_CALUDE_total_hamburger_configurations_l272_27247


namespace NUMINAMATH_CALUDE_power_of_two_geq_n_plus_one_l272_27233

theorem power_of_two_geq_n_plus_one (n : ℕ) (h : n ≥ 1) : 2^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_geq_n_plus_one_l272_27233


namespace NUMINAMATH_CALUDE_micheal_work_days_l272_27274

/-- Represents the total amount of work to be done -/
def W : ℝ := 1

/-- Represents the rate at which Micheal works (fraction of work done per day) -/
def M : ℝ := sorry

/-- Represents the rate at which Adam works (fraction of work done per day) -/
def A : ℝ := sorry

/-- Micheal and Adam can do the work together in 20 days -/
axiom combined_rate : M + A = W / 20

/-- After working together for 14 days, the remaining work is completed by Adam in 10 days -/
axiom remaining_work : A * 10 = W - 14 * (M + A)

theorem micheal_work_days : M = W / 50 := by sorry

end NUMINAMATH_CALUDE_micheal_work_days_l272_27274


namespace NUMINAMATH_CALUDE_triangle_properties_l272_27225

/-- Triangle ABC with given points and conditions -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  h_A : A = (-2, 1)
  h_B : B = (4, 3)

/-- The equation of a line in general form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point lies on a line -/
def lies_on (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Predicate to check if a line is perpendicular to another line -/
def perpendicular (l1 l2 : LineEquation) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem triangle_properties (t : Triangle) :
  (t.C = (3, -2) →
    ∃ (l : LineEquation), l.a = 1 ∧ l.b = 5 ∧ l.c = -3 ∧
    lies_on t.A l ∧
    ∃ (bc : LineEquation), lies_on t.B bc ∧ lies_on t.C bc ∧ perpendicular l bc) ∧
  (t.M = (3, 1) ∧ t.M.1 = (t.A.1 + t.C.1) / 2 ∧ t.M.2 = (t.A.2 + t.C.2) / 2 →
    ∃ (l : LineEquation), l.a = 1 ∧ l.b = 2 ∧ l.c = -10 ∧
    lies_on t.B l ∧ lies_on t.C l) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l272_27225


namespace NUMINAMATH_CALUDE_base5_98_to_base9_l272_27232

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base-9 --/
def decimalToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 9) ((m % 9) :: acc)
    aux n []

/-- Theorem: The base-9 representation of 98₍₅₎ is 58₍₉₎ --/
theorem base5_98_to_base9 :
  decimalToBase9 (base5ToDecimal [8, 9]) = [5, 8] :=
sorry

end NUMINAMATH_CALUDE_base5_98_to_base9_l272_27232


namespace NUMINAMATH_CALUDE_square_diff_over_hundred_l272_27259

theorem square_diff_over_hundred : (2200 - 2100)^2 / 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_over_hundred_l272_27259


namespace NUMINAMATH_CALUDE_odometer_sum_l272_27219

theorem odometer_sum (a b c : ℕ) : 
  a ≥ 1 → 
  a + b + c ≤ 9 → 
  (100 * c + 10 * a + b) - (100 * a + 10 * b + c) % 45 = 0 →
  100 * a + 10 * b + c + 100 * b + 10 * c + a + 100 * c + 10 * a + b = 999 :=
by sorry

end NUMINAMATH_CALUDE_odometer_sum_l272_27219


namespace NUMINAMATH_CALUDE_triangulations_count_l272_27290

/-- The number of triangulations of a convex n-gon with exactly two internal triangles -/
def triangulations_with_two_internal_triangles (n : ℕ) : ℕ :=
  n * Nat.choose (n - 4) 4 * 2^(n - 9)

/-- Theorem stating the number of triangulations of a convex n-gon with exactly two internal triangles -/
theorem triangulations_count (n : ℕ) (hn : n > 7) :
  triangulations_with_two_internal_triangles n =
    n * Nat.choose (n - 4) 4 * 2^(n - 9) := by
  sorry

end NUMINAMATH_CALUDE_triangulations_count_l272_27290


namespace NUMINAMATH_CALUDE_h_of_h_of_two_equals_91265_l272_27207

/-- Given a function h(x) = 3x^3 + 2x^2 - x + 1, prove that h(h(2)) = 91265 -/
theorem h_of_h_of_two_equals_91265 : 
  let h : ℝ → ℝ := fun x ↦ 3 * x^3 + 2 * x^2 - x + 1
  h (h 2) = 91265 := by
  sorry

end NUMINAMATH_CALUDE_h_of_h_of_two_equals_91265_l272_27207


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l272_27221

theorem product_and_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 9 → (1 / x) = (4 / y) → x + y = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l272_27221


namespace NUMINAMATH_CALUDE_no_prime_solution_l272_27283

theorem no_prime_solution : ¬∃ (p q : Nat), Prime p ∧ Prime q ∧ p > 5 ∧ q > 5 ∧ (p * q ∣ (5^p - 2^p) * (5^q - 2^q)) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l272_27283


namespace NUMINAMATH_CALUDE_power_function_above_identity_l272_27215

theorem power_function_above_identity {x α : ℝ} (hx : x ∈ Set.Ioo 0 1) (hα : α < 1) : x^α > x := by
  sorry

end NUMINAMATH_CALUDE_power_function_above_identity_l272_27215


namespace NUMINAMATH_CALUDE_M_intersect_N_l272_27246

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem M_intersect_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l272_27246


namespace NUMINAMATH_CALUDE_intersection_line_equation_l272_27252

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 20

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (A B : ℝ × ℝ),
  (circle1 A.1 A.2 ∧ circle2 A.1 A.2) →
  (circle1 B.1 B.2 ∧ circle2 B.1 B.2) →
  A ≠ B →
  line A.1 A.2 ∧ line B.1 B.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l272_27252


namespace NUMINAMATH_CALUDE_least_perimeter_l272_27282

/-- Triangle DEF with given cosine values -/
structure TriangleDEF where
  d : ℕ
  e : ℕ
  f : ℕ
  cos_d : Real
  cos_e : Real
  cos_f : Real
  h_cos_d : cos_d = 8 / 17
  h_cos_e : cos_e = 15 / 17
  h_cos_f : cos_f = -5 / 13

/-- The perimeter of triangle DEF -/
def perimeter (t : TriangleDEF) : ℕ := t.d + t.e + t.f

/-- The least possible perimeter of triangle DEF is 503 -/
theorem least_perimeter (t : TriangleDEF) : 
  (∀ t' : TriangleDEF, perimeter t ≤ perimeter t') → perimeter t = 503 := by
  sorry

end NUMINAMATH_CALUDE_least_perimeter_l272_27282


namespace NUMINAMATH_CALUDE_inequality_solution_l272_27268

theorem inequality_solution (x : ℝ) : 
  (x - 1) * (x - 4) * (x - 5)^2 / ((x - 3) * (x^2 - 9)) > 0 ↔ -3 < x ∧ x < 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l272_27268


namespace NUMINAMATH_CALUDE_school_girls_count_l272_27231

theorem school_girls_count (total_pupils : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_pupils = 485 → boys = 253 → girls = total_pupils - boys → girls = 232 := by
sorry

end NUMINAMATH_CALUDE_school_girls_count_l272_27231


namespace NUMINAMATH_CALUDE_area_relation_l272_27212

/-- A triangle is acute-angled if all its angles are less than 90 degrees. -/
def IsAcuteAngledTriangle (A B C : ℝ × ℝ) : Prop := sorry

/-- The orthocentre of a triangle is the point where all three altitudes intersect. -/
def Orthocentre (A B C H : ℝ × ℝ) : Prop := sorry

/-- The centroid of a triangle is the arithmetic mean position of all points in the triangle. -/
def Centroid (A B C G : ℝ × ℝ) : Prop := sorry

/-- The area of a triangle given its vertices. -/
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem area_relation (A B C H G₁ G₂ G₃ : ℝ × ℝ) :
  IsAcuteAngledTriangle A B C →
  Orthocentre A B C H →
  Centroid H B C G₁ →
  Centroid H C A G₂ →
  Centroid H A B G₃ →
  TriangleArea G₁ G₂ G₃ = 7 →
  TriangleArea A B C = 63 := by
  sorry

end NUMINAMATH_CALUDE_area_relation_l272_27212


namespace NUMINAMATH_CALUDE_vasya_reading_time_difference_l272_27258

/-- Represents the number of books Vasya planned to read each week -/
def planned_books_per_week : ℕ := sorry

/-- Represents the total number of books in the reading list -/
def total_books : ℕ := 12 * planned_books_per_week

/-- Represents the number of weeks it took Vasya to finish when reading one less book per week -/
def actual_weeks : ℕ := 12 + 3

theorem vasya_reading_time_difference :
  (total_books / (planned_books_per_week + 1) = 10) ∧
  (10 = 12 - 2) :=
by sorry

end NUMINAMATH_CALUDE_vasya_reading_time_difference_l272_27258


namespace NUMINAMATH_CALUDE_marias_number_l272_27234

theorem marias_number (x : ℝ) : ((3 * (x - 3) + 3) / 3 = 10) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_marias_number_l272_27234


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_shifted_roots_l272_27292

theorem sum_of_reciprocals_shifted_roots (a b c : ℂ) : 
  (a^3 - a - 2 = 0) → (b^3 - b - 2 = 0) → (c^3 - c - 2 = 0) →
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_shifted_roots_l272_27292


namespace NUMINAMATH_CALUDE_eight_divided_by_one_eighth_l272_27226

theorem eight_divided_by_one_eighth (x y : ℝ) : x = 8 ∧ y = 1/8 → x / y = 64 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_one_eighth_l272_27226


namespace NUMINAMATH_CALUDE_hall_reunion_attendance_l272_27254

/-- The number of people attending the Hall reunion -/
def hall_attendees (total_guests oates_attendees both_attendees : ℕ) : ℕ :=
  total_guests - (oates_attendees - both_attendees)

/-- Theorem stating the number of people attending the Hall reunion -/
theorem hall_reunion_attendance 
  (total_guests : ℕ) 
  (oates_attendees : ℕ) 
  (both_attendees : ℕ) 
  (h1 : total_guests = 100) 
  (h2 : oates_attendees = 40) 
  (h3 : both_attendees = 10) 
  (h4 : total_guests ≥ oates_attendees) 
  (h5 : oates_attendees ≥ both_attendees) : 
  hall_attendees total_guests oates_attendees both_attendees = 70 := by
  sorry

end NUMINAMATH_CALUDE_hall_reunion_attendance_l272_27254


namespace NUMINAMATH_CALUDE_number_percentage_problem_l272_27241

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 35 → 0.40 * N = 420 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_problem_l272_27241


namespace NUMINAMATH_CALUDE_power_calculation_l272_27243

theorem power_calculation : 10^6 * (10^2)^3 / 10^4 = 10^8 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l272_27243


namespace NUMINAMATH_CALUDE_certain_number_proof_l272_27288

theorem certain_number_proof : ∃ x : ℕ, 865 * 48 = 173 * x ∧ x = 240 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l272_27288


namespace NUMINAMATH_CALUDE_no_non_multiple_ghosts_l272_27278

/-- Definition of the sequence S -/
def S (p : ℕ) : ℕ → ℕ
  | n => if n < p then n else sorry

/-- A number is a ghost if it doesn't appear in S -/
def is_ghost (p : ℕ) (k : ℕ) : Prop :=
  ∀ n, S p n ≠ k

/-- Main theorem: There are no ghosts that are not multiples of p -/
theorem no_non_multiple_ghosts (p : ℕ) (hp : Prime p) (hp_odd : Odd p) :
  ∀ k, ¬(p ∣ k) → ¬(is_ghost p k) := by sorry

end NUMINAMATH_CALUDE_no_non_multiple_ghosts_l272_27278


namespace NUMINAMATH_CALUDE_first_candidate_marks_l272_27275

/-- Represents the total marks in the exam -/
def total_marks : ℝ := 600

/-- Represents the passing marks -/
def passing_marks : ℝ := 240

/-- Represents the percentage of marks obtained by the first candidate -/
def first_candidate_percentage : ℝ := 30

/-- Theorem stating the percentage of marks obtained by the first candidate -/
theorem first_candidate_marks :
  let second_candidate_marks := 0.45 * total_marks
  let first_candidate_marks := (first_candidate_percentage / 100) * total_marks
  (second_candidate_marks = passing_marks + 30) ∧
  (first_candidate_marks = passing_marks - 60) →
  first_candidate_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_first_candidate_marks_l272_27275


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l272_27272

theorem arithmetic_square_root_of_nine : ∃! x : ℝ, x ≥ 0 ∧ x^2 = 9 :=
  by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l272_27272


namespace NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_twelve_l272_27297

/-- The sum of the tens digit and the ones digit of (1+6)^12 is 1 -/
theorem sum_of_digits_of_seven_to_twelve : 
  (((1 + 6)^12 / 10) % 10 + (1 + 6)^12 % 10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_twelve_l272_27297


namespace NUMINAMATH_CALUDE_fifth_day_is_tuesday_l272_27245

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure DayInMonth where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Returns the day of the week for a given number of days after a reference day -/
def dayAfter (startDay : DayOfWeek) (daysAfter : Int) : DayOfWeek :=
  sorry

theorem fifth_day_is_tuesday
  (month : List DayInMonth)
  (h : ∃ d ∈ month, d.day = 20 ∧ d.dayOfWeek = DayOfWeek.Wednesday) :
  ∃ d ∈ month, d.day = 5 ∧ d.dayOfWeek = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_fifth_day_is_tuesday_l272_27245


namespace NUMINAMATH_CALUDE_composition_result_l272_27240

-- Define the functions f and g
def f (b : ℝ) (x : ℝ) : ℝ := 5 * x + b
def g (b : ℝ) (x : ℝ) : ℝ := b * x + 3

-- State the theorem
theorem composition_result (b e : ℝ) :
  (∀ x, f b (g b x) = 15 * x + e) → e = 18 := by
  sorry

end NUMINAMATH_CALUDE_composition_result_l272_27240


namespace NUMINAMATH_CALUDE_bike_sharing_growth_specific_bike_sharing_case_l272_27267

/-- Represents the growth of shared bicycles over three months -/
theorem bike_sharing_growth 
  (initial_bikes : ℕ) 
  (planned_increase : ℕ) 
  (growth_rate : ℝ) : 
  initial_bikes * (1 + growth_rate)^2 = initial_bikes + planned_increase :=
by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_bike_sharing_case (x : ℝ) : 
  1000 * (1 + x)^2 = 1000 + 440 :=
by
  sorry

end NUMINAMATH_CALUDE_bike_sharing_growth_specific_bike_sharing_case_l272_27267


namespace NUMINAMATH_CALUDE_exp_13pi_i_div_2_eq_i_l272_27228

-- Define the complex exponential function
noncomputable def complex_exp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem exp_13pi_i_div_2_eq_i : complex_exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_exp_13pi_i_div_2_eq_i_l272_27228


namespace NUMINAMATH_CALUDE_tan_period_l272_27294

theorem tan_period (x : ℝ) : 
  let f : ℝ → ℝ := fun x => Real.tan (3 * x / 4)
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ p = 4 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_period_l272_27294


namespace NUMINAMATH_CALUDE_complex_equality_l272_27220

-- Define the complex numbers
def z1 (x y : ℝ) : ℂ := x - 1 + y * Complex.I
def z2 (x : ℝ) : ℂ := Complex.I - 3 * x

-- Theorem statement
theorem complex_equality (x y : ℝ) :
  z1 x y = z2 x → x = 1/4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l272_27220


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l272_27269

theorem complex_expression_simplification :
  let i : ℂ := Complex.I
  3 * (4 - 2*i) + 2*i * (3 - 2*i) = 16 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l272_27269


namespace NUMINAMATH_CALUDE_prob_no_red_square_is_127_128_l272_27224

/-- Represents a 4-by-4 grid where each cell can be colored red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Returns true if the grid has a 3-by-3 red square starting at (i, j) -/
def has_red_square (g : Grid) (i j : Fin 2) : Prop :=
  ∀ (x y : Fin 3), g (i + x) (j + y) = true

/-- The probability of a grid not having any 3-by-3 red square -/
def prob_no_red_square : ℚ :=
  1 - (4 : ℚ) / 2^9

theorem prob_no_red_square_is_127_128 :
  prob_no_red_square = 127 / 128 := by sorry

#check prob_no_red_square_is_127_128

end NUMINAMATH_CALUDE_prob_no_red_square_is_127_128_l272_27224


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2011_unique_term_2011_l272_27209

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmeticSequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- Theorem stating that the 671st term of the sequence is 2011 -/
theorem arithmetic_sequence_2011 : arithmeticSequence 671 = 2011 := by sorry

/-- Theorem proving that 671 is the unique natural number n for which a_n = 2011 -/
theorem unique_term_2011 : ∀ n : ℕ, arithmeticSequence n = 2011 ↔ n = 671 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2011_unique_term_2011_l272_27209


namespace NUMINAMATH_CALUDE_horner_v4_value_l272_27281

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ := v * x + a

theorem horner_v4_value :
  let x := -4
  let v0 := 3
  let v1 := horner_step v0 x 5
  let v2 := horner_step v1 x 6
  let v3 := horner_step v2 x 79
  let v4 := horner_step v3 x (-8)
  v4 = 220 :=
by sorry

end NUMINAMATH_CALUDE_horner_v4_value_l272_27281


namespace NUMINAMATH_CALUDE_difference_y_coordinates_l272_27217

/-- Given a line with equation x = 2y + 5 and two points (m, n) and (m + 4, n + k) on this line,
    the value of k is 2. -/
theorem difference_y_coordinates (m n k : ℝ) : 
  (m = 2*n + 5) → (m + 4 = 2*(n + k) + 5) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_y_coordinates_l272_27217


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l272_27216

/-- Triangle ABC with given vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from a point to a line -/
def altitude (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Main theorem about the triangle ABC -/
theorem triangle_abc_properties :
  let t : Triangle := { A := (-2, 4), B := (-3, -1), C := (1, 3) }
  let alt_B_AC : ℝ → ℝ := altitude t.B (fun x => x - 1)  -- Line AC: y = x - 1
  ∀ x y, alt_B_AC x = y ↔ x + y - 2 = 0 ∧ triangleArea t = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l272_27216


namespace NUMINAMATH_CALUDE_jonahs_calorie_burn_l272_27200

/-- Calculates the difference in calories burned between two running durations -/
def calorie_difference (rate : ℕ) (duration1 duration2 : ℕ) : ℕ :=
  rate * duration2 - rate * duration1

/-- The problem statement -/
theorem jonahs_calorie_burn :
  let rate : ℕ := 30
  let short_duration : ℕ := 2
  let long_duration : ℕ := 5
  calorie_difference rate short_duration long_duration = 90 := by
  sorry

end NUMINAMATH_CALUDE_jonahs_calorie_burn_l272_27200


namespace NUMINAMATH_CALUDE_max_different_ages_l272_27222

theorem max_different_ages 
  (average_age : ℝ) 
  (std_dev : ℝ) 
  (average_age_eq : average_age = 31) 
  (std_dev_eq : std_dev = 8) : 
  ∃ (max_ages : ℕ), 
    max_ages = 17 ∧ 
    ∀ (age : ℕ), 
      (↑age ≥ average_age - std_dev ∧ ↑age ≤ average_age + std_dev) ↔ 
      (age ≥ 23 ∧ age ≤ 39) :=
by sorry

end NUMINAMATH_CALUDE_max_different_ages_l272_27222


namespace NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l272_27253

open Complex

theorem smallest_absolute_value_of_z (z : ℂ) (h : abs (z - 12) + abs (z - 5*I) = 13) :
  ∃ (w : ℂ), abs (z - 12) + abs (z - 5*I) = 13 ∧ abs w ≤ abs z ∧ abs w = 60 / 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l272_27253


namespace NUMINAMATH_CALUDE_subset_M_l272_27223

def M : Set ℕ := {x : ℕ | (1 : ℚ) / (x - 2 : ℚ) ≤ 0}

theorem subset_M : {1} ⊆ M := by sorry

end NUMINAMATH_CALUDE_subset_M_l272_27223


namespace NUMINAMATH_CALUDE_mod_equivalence_l272_27250

theorem mod_equivalence (n : ℕ) : 
  (179 * 933 / 7) % 50 = n ∧ 0 ≤ n ∧ n < 50 → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_l272_27250


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l272_27266

theorem polynomial_evaluation : 
  let x : ℝ := 2
  2 * x^3 + 3 * x^2 - 7 * x + 4 = 18 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l272_27266


namespace NUMINAMATH_CALUDE_original_price_calculation_l272_27244

theorem original_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 600) 
  (h2 : profit_percentage = 20) : 
  ∃ original_price : ℝ, 
    selling_price = original_price * (1 + profit_percentage / 100) ∧ 
    original_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l272_27244


namespace NUMINAMATH_CALUDE_function_values_l272_27296

/-- The linear function f(x) = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x - 3

theorem function_values :
  (f 4 = 5) ∧ (f (3/2) = 0) := by sorry

end NUMINAMATH_CALUDE_function_values_l272_27296


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l272_27284

theorem sum_of_squares_of_roots : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 2*x₁ + 4)^(x₁^2 - 2*x₁ + 3) = 625 ∧
  (x₂^2 - 2*x₂ + 4)^(x₂^2 - 2*x₂ + 3) = 625 ∧
  x₁ ≠ x₂ ∧
  x₁^2 + x₂^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l272_27284


namespace NUMINAMATH_CALUDE_y_coordinate_of_P_l272_27293

/-- A line through the origin equidistant from two points -/
structure EquidistantLine where
  slope : ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  origin_line : slope * P.1 = P.2 ∧ slope * Q.1 = Q.2
  equidistant : (P.1 - 0)^2 + (P.2 - 0)^2 = (Q.1 - 0)^2 + (Q.2 - 0)^2

/-- Theorem: Given the conditions, the y-coordinate of P is 3.2 -/
theorem y_coordinate_of_P (L : EquidistantLine)
  (h_slope : L.slope = 0.8)
  (h_x_coord : L.P.1 = 4) :
  L.P.2 = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_of_P_l272_27293


namespace NUMINAMATH_CALUDE_square_difference_l272_27229

theorem square_difference (a b : ℝ) (h1 : a + b = 5) (h2 : a - b = 2) : a^2 - b^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l272_27229


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l272_27299

/-- A quadratic equation ax^2 + bx + c = 0 has two distinct real roots if and only if its discriminant is positive -/
axiom quadratic_two_roots (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b^2 - 4*a*c > 0

/-- If c < 1/4, then the quadratic equation x^2 + 2x + 4c = 0 has two distinct real roots -/
theorem quadratic_roots_condition (c : ℝ) (h : c < 1/4) :
  ∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + 4*c = 0 ∧ y^2 + 2*y + 4*c = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l272_27299


namespace NUMINAMATH_CALUDE_flea_problem_l272_27208

/-- Represents the number of ways a flea can reach a point given the distance and number of jumps -/
def flea_jumps (distance : ℤ) (jumps : ℕ) : ℕ := sorry

/-- Represents whether it's possible for a flea to reach a point given the distance and number of jumps -/
def flea_can_reach (distance : ℤ) (jumps : ℕ) : Prop := sorry

theorem flea_problem :
  (flea_jumps 5 7 = 7) ∧
  (flea_jumps 5 9 = 36) ∧
  ¬(flea_can_reach 2013 2028) := by sorry

end NUMINAMATH_CALUDE_flea_problem_l272_27208


namespace NUMINAMATH_CALUDE_noah_lights_on_time_l272_27257

def bedroom_wattage : ℝ := 6
def office_wattage : ℝ := 3 * bedroom_wattage
def living_room_wattage : ℝ := 4 * bedroom_wattage
def total_energy_used : ℝ := 96

def total_wattage_per_hour : ℝ := bedroom_wattage + office_wattage + living_room_wattage

theorem noah_lights_on_time :
  total_energy_used / total_wattage_per_hour = 2 := by sorry

end NUMINAMATH_CALUDE_noah_lights_on_time_l272_27257


namespace NUMINAMATH_CALUDE_sufficient_lunks_for_bananas_l272_27280

/-- Represents the exchange rate between lunks and kunks -/
def lunk_to_kunk_rate : ℚ := 6 / 10

/-- Represents the exchange rate between kunks and bananas -/
def kunk_to_banana_rate : ℚ := 5 / 3

/-- The number of bananas we want to purchase -/
def target_bananas : ℕ := 24

/-- The number of lunks we claim is sufficient -/
def claimed_lunks : ℕ := 25

theorem sufficient_lunks_for_bananas :
  ∃ (kunks : ℚ),
    kunks * kunk_to_banana_rate ≥ target_bananas ∧
    kunks ≤ claimed_lunks * lunk_to_kunk_rate :=
by
  sorry

#check sufficient_lunks_for_bananas

end NUMINAMATH_CALUDE_sufficient_lunks_for_bananas_l272_27280


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l272_27271

theorem largest_solution_of_equation (x : ℝ) : 
  (3 * (10 * x^2 + 11 * x + 12) = x * (10 * x - 45)) →
  x ≤ (-39 + Real.sqrt 801) / 20 := by
sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l272_27271


namespace NUMINAMATH_CALUDE_league_games_count_l272_27263

def number_of_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem league_games_count :
  let total_teams : ℕ := 8
  let teams_per_game : ℕ := 2
  number_of_games total_teams = 28 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l272_27263


namespace NUMINAMATH_CALUDE_complement_of_M_l272_27295

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- State the theorem
theorem complement_of_M : 
  Set.compl M = {y : ℝ | y < -1} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l272_27295


namespace NUMINAMATH_CALUDE_smallest_non_special_number_l272_27256

def is_triangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k : ℕ, Nat.Prime p ∧ n = p ^ k

def is_prime_plus_one (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p + 1

def is_product_of_distinct_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q

theorem smallest_non_special_number : 
  (∀ n < 40, is_triangular n ∨ is_prime_power n ∨ is_prime_plus_one n ∨ is_product_of_distinct_primes n) ∧
  ¬(is_triangular 40 ∨ is_prime_power 40 ∨ is_prime_plus_one 40 ∨ is_product_of_distinct_primes 40) :=
sorry

end NUMINAMATH_CALUDE_smallest_non_special_number_l272_27256


namespace NUMINAMATH_CALUDE_power_function_m_value_l272_27242

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℝ), ∀ x, f x = a * x^n

theorem power_function_m_value (m : ℝ) :
  is_power_function (λ x => (3*m - 1) * x^m) → m = 2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_m_value_l272_27242


namespace NUMINAMATH_CALUDE_triangle_length_l272_27213

-- Define the curve y = x^3
def curve (x : ℝ) : ℝ := x^3

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions of the problem
structure ProblemConditions where
  triangle : Triangle
  on_curve : 
    curve triangle.A.1 = triangle.A.2 ∧
    curve triangle.B.1 = triangle.B.2 ∧
    curve triangle.C.1 = triangle.C.2
  A_at_origin : triangle.A = (0, 0)
  BC_parallel_x : triangle.B.2 = triangle.C.2
  area : ℝ

-- Define the theorem
theorem triangle_length (conditions : ProblemConditions) 
  (h : conditions.area = 125) : 
  let BC_length := |conditions.triangle.C.1 - conditions.triangle.B.1|
  BC_length = 10 := by sorry

end NUMINAMATH_CALUDE_triangle_length_l272_27213


namespace NUMINAMATH_CALUDE_proposition_truth_values_l272_27279

theorem proposition_truth_values (p q : Prop) (h1 : ¬p) (h2 : q) :
  ¬p ∧ ¬(p ∧ q) ∧ ¬(¬q) ∧ (p ∨ q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l272_27279


namespace NUMINAMATH_CALUDE_f_decreasing_iff_a_in_range_l272_27287

/-- The function f(x) defined as 2ax² + 4(a-3)x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x^2 + 4*(a-3)*x + 5

/-- The property of f(x) being decreasing on the interval (-∞, 3) -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 3 → y < 3 → f a x > f a y

/-- The theorem stating the range of a for which f(x) is decreasing on (-∞, 3) -/
theorem f_decreasing_iff_a_in_range :
  ∀ a, is_decreasing_on_interval a ↔ a ∈ Set.Icc 0 (3/4) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_iff_a_in_range_l272_27287


namespace NUMINAMATH_CALUDE_trigonometric_identities_l272_27270

variable (θ : Real)
variable (α : Real)

/-- Given tan θ = 2, prove the following statements -/
theorem trigonometric_identities (h : Real.tan θ = 2) :
  ((Real.sin α + Real.sqrt 2 * Real.cos α) / (Real.sin α - Real.sqrt 2 * Real.cos α) = 3 + 2 * Real.sqrt 2) ∧
  (Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l272_27270


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_is_4022_l272_27260

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the interval
def I : Set ℝ := Set.Icc (-2011) 2011

-- State the theorem
theorem sum_of_max_and_min_is_4022 
  (h1 : ∀ x ∈ I, ∀ y ∈ I, f (x + y) = f x + f y - 2011)
  (h2 : ∀ x > 0, x ∈ I → f x > 2011)
  : (⨆ x ∈ I, f x) + (⨅ x ∈ I, f x) = 4022 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_is_4022_l272_27260


namespace NUMINAMATH_CALUDE_g_of_3_l272_27210

def g (x : ℝ) : ℝ := -3 * x^4 + 4 * x^3 - 7 * x^2 + 5 * x - 2

theorem g_of_3 : g 3 = -185 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l272_27210


namespace NUMINAMATH_CALUDE_probability_order_l272_27235

-- Define the structure of a deck of cards
structure Card where
  suit : Fin 4
  rank : Fin 13

-- Define the deck
def standardDeck : Finset Card := sorry

-- Define the subsets of cards for each event
def fiveOfHearts : Finset Card := sorry
def jokers : Finset Card := sorry
def fives : Finset Card := sorry
def clubs : Finset Card := sorry
def redCards : Finset Card := sorry

-- Define the probability of drawing a card from a given set
def probability (subset : Finset Card) : ℚ :=
  (subset.card : ℚ) / (standardDeck.card : ℚ)

-- Theorem statement
theorem probability_order :
  probability fiveOfHearts < probability jokers ∧
  probability jokers < probability fives ∧
  probability fives < probability clubs ∧
  probability clubs < probability redCards :=
sorry

end NUMINAMATH_CALUDE_probability_order_l272_27235


namespace NUMINAMATH_CALUDE_neil_initial_games_neil_had_two_games_l272_27202

theorem neil_initial_games (henry_initial : ℕ) (games_given : ℕ) (henry_neil_ratio : ℕ) : ℕ :=
  let henry_final := henry_initial - games_given
  let neil_final := henry_final / henry_neil_ratio
  neil_final - games_given

theorem neil_had_two_games : neil_initial_games 33 5 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_neil_initial_games_neil_had_two_games_l272_27202


namespace NUMINAMATH_CALUDE_line_equation_correct_l272_27214

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Checks if a point (x, y) satisfies the equation of the line -/
def Line.satisfiesEquation (l : Line) (x y : ℝ) : Prop :=
  2 * x - y - 5 = 0

theorem line_equation_correct (l : Line) :
  l.slope = 2 ∧ l.point = (3, 1) →
  ∀ x y : ℝ, l.satisfiesEquation x y ↔ y - 1 = l.slope * (x - 3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_correct_l272_27214


namespace NUMINAMATH_CALUDE_car_speed_problem_l272_27206

/-- Given a car traveling for two hours with speeds x and 60 km/h, 
    prove that if the average speed is 102.5 km/h, then x must be 145 km/h. -/
theorem car_speed_problem (x : ℝ) :
  (x + 60) / 2 = 102.5 → x = 145 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l272_27206


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l272_27237

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l272_27237


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l272_27298

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 4)^2 - 6*(a 4) + 5 = 0 →
  (a 8)^2 - 6*(a 8) + 5 = 0 →
  a 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l272_27298


namespace NUMINAMATH_CALUDE_a_range_l272_27204

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > a then x + 2 else x^2 + 5*x + 2

-- Define g(x) in terms of f(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2*x

-- Define a predicate for g having exactly three distinct zeros
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    g a x = 0 ∧ g a y = 0 ∧ g a z = 0 ∧
    ∀ w : ℝ, g a w = 0 → w = x ∨ w = y ∨ w = z

-- The main theorem
theorem a_range (a : ℝ) :
  has_three_distinct_zeros a ↔ a ∈ Set.Icc (-1 : ℝ) 2 ∧ a ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_a_range_l272_27204


namespace NUMINAMATH_CALUDE_example_monomial_properties_l272_27276

/-- Represents a monomial with integer coefficient and variables x, y, and z -/
structure Monomial where
  coeff : Int
  x_exp : Nat
  y_exp : Nat
  z_exp : Nat

/-- Calculates the coefficient of a monomial -/
def coefficient (m : Monomial) : Int :=
  m.coeff

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : Nat :=
  m.x_exp + m.y_exp + m.z_exp

/-- The monomial -3^2 * x * y * z^2 -/
def example_monomial : Monomial :=
  { coeff := -9, x_exp := 1, y_exp := 1, z_exp := 2 }

theorem example_monomial_properties :
  (coefficient example_monomial = -9) ∧ (degree example_monomial = 4) := by
  sorry


end NUMINAMATH_CALUDE_example_monomial_properties_l272_27276


namespace NUMINAMATH_CALUDE_hyperbola_condition_l272_27289

theorem hyperbola_condition (m : ℝ) (h1 : -3 < m) (h2 : m < 0) :
  ∃ (x y : ℝ), (x^2 / (m - 2) + y^2 / (m + 3) = 1) ∧ 
  (∀ (a b : ℝ), a^2 / (m - 2) + b^2 / (m + 3) = 1 → 
    (a, b) ≠ (0, 0) ∧ (a / (m - 2), b / (m + 3)) ≠ (0, 0)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l272_27289


namespace NUMINAMATH_CALUDE_incorrect_height_calculation_l272_27236

theorem incorrect_height_calculation (n : ℕ) (initial_avg real_avg actual_height : ℝ) 
  (h1 : n = 35)
  (h2 : initial_avg = 185)
  (h3 : real_avg = 183)
  (h4 : actual_height = 106) :
  ∃ (incorrect_height : ℝ),
    incorrect_height = n * initial_avg - (n * real_avg - actual_height) ∧
    incorrect_height = 176 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_height_calculation_l272_27236


namespace NUMINAMATH_CALUDE_donovans_test_incorrect_answers_l272_27218

theorem donovans_test_incorrect_answers :
  ∀ (total : ℕ) (correct : ℕ) (percentage : ℚ),
    correct = 35 →
    percentage = 7292 / 10000 →
    (correct : ℚ) / (total : ℚ) = percentage →
    total - correct = 13 :=
  by sorry

end NUMINAMATH_CALUDE_donovans_test_incorrect_answers_l272_27218


namespace NUMINAMATH_CALUDE_equation_solution_l272_27203

theorem equation_solution (x y : ℝ) : ∃ z : ℝ, 0.65 * x * y - z = 0.2 * 747.50 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l272_27203


namespace NUMINAMATH_CALUDE_julio_has_more_soda_l272_27286

/-- Calculates the total liters of soda for a person given the number of orange and grape soda bottles -/
def totalSoda (orangeBottles grapeBottles : ℕ) : ℕ := 2 * (orangeBottles + grapeBottles)

theorem julio_has_more_soda : 
  let julioTotal := totalSoda 4 7
  let mateoTotal := totalSoda 1 3
  julioTotal - mateoTotal = 14 := by
  sorry

end NUMINAMATH_CALUDE_julio_has_more_soda_l272_27286


namespace NUMINAMATH_CALUDE_farm_oxen_count_l272_27285

/-- Represents the daily fodder consumption of one buffalo -/
def B : ℝ := sorry

/-- Represents the number of oxen on the farm -/
def O : ℕ := sorry

/-- The total amount of fodder available on the farm -/
def total_fodder : ℝ := sorry

theorem farm_oxen_count : O = 8 := by
  have h1 : 3 * B = 4 * (3/4 * B) := sorry
  have h2 : 3 * B = 2 * (3/2 * B) := sorry
  have h3 : total_fodder = (33 * B + 3/2 * O * B) * 48 := sorry
  have h4 : total_fodder = (108 * B + 3/2 * O * B) * 18 := sorry
  sorry

end NUMINAMATH_CALUDE_farm_oxen_count_l272_27285


namespace NUMINAMATH_CALUDE_parallel_vectors_l272_27273

/-- Given vectors a and b, find k such that (2a + b) is parallel to (1/2a + kb) -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (1, 2)) :
  ∃ k : ℝ, k = (1/4 : ℝ) ∧ 
  ∃ c : ℝ, c ≠ 0 ∧ c • (2 • a + b) = (1/2 • a + k • b) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l272_27273


namespace NUMINAMATH_CALUDE_perpendicular_to_countless_lines_perpendicular_to_intersection_perpendicular_to_plane_l272_27264

-- Define two perpendicular planes
axiom Plane1 : Type
axiom Plane2 : Type
axiom perpendicular_planes : Plane1 → Plane2 → Prop

-- Define a line
axiom Line : Type

-- Define a line being in a plane
axiom line_in_plane : Line → Plane1 → Prop
axiom line_in_plane2 : Line → Plane2 → Prop

-- Define perpendicularity between lines
axiom perpendicular_lines : Line → Line → Prop

-- Define perpendicularity between a line and a plane
axiom perpendicular_line_plane : Line → Plane1 → Prop
axiom perpendicular_line_plane2 : Line → Plane2 → Prop

-- Define the intersection line of two planes
axiom intersection_line : Plane1 → Plane2 → Line

-- Define a point
axiom Point : Type

-- Define a point being in a plane
axiom point_in_plane : Point → Plane1 → Prop

-- Define drawing a perpendicular line from a point to a line
axiom perpendicular_from_point : Point → Line → Line

-- Theorem 1: A line in one plane must be perpendicular to countless lines in the other plane
theorem perpendicular_to_countless_lines 
  (p1 : Plane1) (p2 : Plane2) (l : Line) 
  (h1 : perpendicular_planes p1 p2) 
  (h2 : line_in_plane l p1) : 
  ∃ (S : Set Line), (∀ l' ∈ S, line_in_plane2 l' p2 ∧ perpendicular_lines l l') ∧ Set.Infinite S :=
sorry

-- Theorem 2: If a perpendicular to the intersection line is drawn from any point in one plane, 
-- then this perpendicular must be perpendicular to the other plane
theorem perpendicular_to_intersection_perpendicular_to_plane 
  (p1 : Plane1) (p2 : Plane2) (pt : Point) 
  (h1 : perpendicular_planes p1 p2) 
  (h2 : point_in_plane pt p1) :
  let i := intersection_line p1 p2
  let perp := perpendicular_from_point pt i
  perpendicular_line_plane2 perp p2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_countless_lines_perpendicular_to_intersection_perpendicular_to_plane_l272_27264


namespace NUMINAMATH_CALUDE_count_valid_pairs_l272_27291

def validPair (x y : ℕ) : Prop :=
  2 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 16 ∧ 3 * x = y

theorem count_valid_pairs :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 4 ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs ↔ validPair p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l272_27291


namespace NUMINAMATH_CALUDE_golden_retriever_age_problem_l272_27201

/-- The age of a golden retriever given its weight gain per year and current weight -/
def golden_retriever_age (weight_gain_per_year : ℕ) (current_weight : ℕ) : ℕ :=
  current_weight / weight_gain_per_year

/-- Theorem: The age of a golden retriever that gains 11 pounds each year and currently weighs 88 pounds is 8 years -/
theorem golden_retriever_age_problem :
  golden_retriever_age 11 88 = 8 := by
  sorry

end NUMINAMATH_CALUDE_golden_retriever_age_problem_l272_27201


namespace NUMINAMATH_CALUDE_trigonometric_expression_value_l272_27230

theorem trigonometric_expression_value : 
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_value_l272_27230


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l272_27265

/-- The value of a when a line is tangent to a circle --/
theorem tangent_line_to_circle (a : ℝ) : 
  a > 0 →
  (∃ (x y : ℝ), x^2 + y^2 - a*x = 0 ∧ x - y - 1 = 0) →
  (∀ (x y : ℝ), x^2 + y^2 - a*x = 0 → x - y - 1 ≠ 0 ∨ 
    (∃ (x' y' : ℝ), x' ≠ x ∧ y' ≠ y ∧ x'^2 + y'^2 - a*x' = 0 ∧ x' - y' - 1 = 0)) →
  a = 2*(Real.sqrt 2 - 1) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l272_27265


namespace NUMINAMATH_CALUDE_star_calculation_l272_27227

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := x^3 - y

-- State the theorem
theorem star_calculation :
  star (3^(star 5 18)) (2^(star 2 9)) = 3^321 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l272_27227


namespace NUMINAMATH_CALUDE_classroom_size_l272_27277

theorem classroom_size (x : ℕ) 
  (h1 : (11 * x : ℝ) = (10 * (x - 1) + 30 : ℝ)) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_classroom_size_l272_27277


namespace NUMINAMATH_CALUDE_average_beef_sold_example_l272_27238

/-- Calculates the average amount of beef sold per day over three days -/
def average_beef_sold (day1 : ℕ) (day2_multiplier : ℕ) (day3 : ℕ) : ℚ :=
  (day1 + day1 * day2_multiplier + day3) / 3

theorem average_beef_sold_example :
  average_beef_sold 210 2 150 = 260 := by
  sorry

end NUMINAMATH_CALUDE_average_beef_sold_example_l272_27238


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l272_27205

theorem sine_cosine_inequality (a : ℝ) :
  (a < 0) →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔
  (a ≤ -2) := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l272_27205


namespace NUMINAMATH_CALUDE_problem_solution_l272_27251

def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1|

theorem problem_solution (a : ℝ) :
  (∀ x, f a x ≤ 2 ↔ -1/2 ≤ x ∧ x ≤ 3/2) →
  (a = 2 ∧
   ∀ x, f 2 x + f 2 (x/2 - 1) ≥ 5 ↔ x ≥ 3 ∨ x ≤ -1/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l272_27251


namespace NUMINAMATH_CALUDE_subtraction_relation_l272_27255

theorem subtraction_relation (minuend subtrahend difference : ℝ) 
  (h : subtrahend + difference = minuend) : 
  (minuend + subtrahend + difference) / minuend = 2 := by
sorry

end NUMINAMATH_CALUDE_subtraction_relation_l272_27255


namespace NUMINAMATH_CALUDE_radius_of_larger_circle_l272_27261

/-- Given a configuration of four circles of radius 2 that are externally tangent to two others
    and internally tangent to a larger circle, the radius of the larger circle is 2√3 + 2. -/
theorem radius_of_larger_circle (r : ℝ) (h1 : r > 0) :
  let small_radius : ℝ := 2
  let diagonal : ℝ := 4 * Real.sqrt 2
  let large_radius : ℝ := r
  (small_radius > 0) →
  (diagonal = 4 * Real.sqrt 2) →
  (large_radius = 2 * Real.sqrt 3 + 2) :=
by
  sorry

#check radius_of_larger_circle

end NUMINAMATH_CALUDE_radius_of_larger_circle_l272_27261


namespace NUMINAMATH_CALUDE_base_conversion_512_to_octal_l272_27211

theorem base_conversion_512_to_octal :
  (512 : ℕ) = 1 * 8^3 + 0 * 8^2 + 0 * 8^1 + 0 * 8^0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_512_to_octal_l272_27211


namespace NUMINAMATH_CALUDE_linear_decreasing_iff_k_lt_neg_half_l272_27262

/-- A function f: ℝ → ℝ is decreasing if for all x₁ < x₂, f(x₁) > f(x₂) -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

/-- The linear function y = (2k+1)x + b -/
def f (k b : ℝ) (x : ℝ) : ℝ := (2*k + 1)*x + b

theorem linear_decreasing_iff_k_lt_neg_half (k b : ℝ) :
  IsDecreasing (f k b) ↔ k < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_linear_decreasing_iff_k_lt_neg_half_l272_27262


namespace NUMINAMATH_CALUDE_workshop_percentage_approx_29_l272_27249

/-- Calculates the percentage of a work day spent in workshops -/
def workshop_percentage (work_day_hours : ℕ) (workshop1_minutes : ℕ) (workshop2_multiplier : ℕ) : ℚ :=
  let work_day_minutes : ℕ := work_day_hours * 60
  let workshop2_minutes : ℕ := workshop1_minutes * workshop2_multiplier
  let total_workshop_minutes : ℕ := workshop1_minutes + workshop2_minutes
  (total_workshop_minutes : ℚ) / (work_day_minutes : ℚ) * 100

/-- The percentage of the work day spent in workshops is approximately 29% -/
theorem workshop_percentage_approx_29 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |workshop_percentage 8 35 3 - 29| < ε :=
sorry

end NUMINAMATH_CALUDE_workshop_percentage_approx_29_l272_27249
