import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_sqrt_equation_l1835_183589

theorem no_solution_sqrt_equation : ¬ ∃ x : ℝ, Real.sqrt (3 * x - 2) + Real.sqrt (2 * x - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_sqrt_equation_l1835_183589


namespace NUMINAMATH_CALUDE_profit_ratio_proportional_to_investment_l1835_183595

/-- The ratio of profits for two investors is proportional to their investments -/
theorem profit_ratio_proportional_to_investment 
  (p_investment q_investment : ℕ) 
  (hp : p_investment = 40000) 
  (hq : q_investment = 60000) : 
  (p_investment : ℚ) / q_investment = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_profit_ratio_proportional_to_investment_l1835_183595


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l1835_183556

theorem rectangle_area_theorem (L W : ℝ) (h : 2 * L * (3 * W) = 1800) : L * W = 300 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l1835_183556


namespace NUMINAMATH_CALUDE_binomial_coefficient_arithmetic_sequence_l1835_183552

theorem binomial_coefficient_arithmetic_sequence (n : ℕ) : 
  (2 * Nat.choose n 9 = Nat.choose n 8 + Nat.choose n 10) ↔ (n = 14 ∨ n = 23) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_arithmetic_sequence_l1835_183552


namespace NUMINAMATH_CALUDE_probability_six_odd_in_eight_rolls_l1835_183543

theorem probability_six_odd_in_eight_rolls (n : ℕ) (p : ℚ) : 
  n = 8 →                   -- number of rolls
  p = 1/2 →                 -- probability of rolling an odd number
  (n.choose 6 : ℚ) * p^6 * (1 - p)^(n - 6) = 7/64 := by
sorry

end NUMINAMATH_CALUDE_probability_six_odd_in_eight_rolls_l1835_183543


namespace NUMINAMATH_CALUDE_investment_income_is_500_l1835_183533

/-- Calculates the yearly income from investments given the total amount,
    amounts invested at different rates, and their corresponding interest rates. -/
def yearly_income (total : ℝ) (amount1 : ℝ) (rate1 : ℝ) (amount2 : ℝ) (rate2 : ℝ) (rate3 : ℝ) : ℝ :=
  amount1 * rate1 + amount2 * rate2 + (total - amount1 - amount2) * rate3

/-- Theorem stating that the yearly income from the given investment scenario is $500 -/
theorem investment_income_is_500 :
  yearly_income 10000 4000 0.05 3500 0.04 0.064 = 500 := by
  sorry

#eval yearly_income 10000 4000 0.05 3500 0.04 0.064

end NUMINAMATH_CALUDE_investment_income_is_500_l1835_183533


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_sum_100_l1835_183504

theorem smallest_of_five_consecutive_sum_100 (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    a + b + c + d + e = 100 ∧ 
    b = a + 1 ∧ 
    c = a + 2 ∧ 
    d = a + 3 ∧ 
    e = a + 4) → 
  n = 18 := by sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_sum_100_l1835_183504


namespace NUMINAMATH_CALUDE_solve_equation_l1835_183500

-- Define the operation * based on the given condition
def star (a b : ℝ) : ℝ := a * (a * b - 7)

-- Theorem statement
theorem solve_equation : 
  (∃ x : ℝ, (star 3 x) = (star 2 (-8))) ∧ 
  (∀ x : ℝ, (star 3 x) = (star 2 (-8)) → x = -25/9) := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l1835_183500


namespace NUMINAMATH_CALUDE_smallest_multiple_l1835_183583

theorem smallest_multiple : ∃ (a : ℕ), 
  (∀ (n : ℕ), n > 0 ∧ 6 ∣ n ∧ 15 ∣ n ∧ n > 40 → a ≤ n) ∧
  6 ∣ a ∧ 15 ∣ a ∧ a > 40 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1835_183583


namespace NUMINAMATH_CALUDE_completing_square_transformation_l1835_183584

theorem completing_square_transformation (x : ℝ) : 
  (x^2 + 6*x - 4 = 0) ↔ ((x + 3)^2 = 13) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l1835_183584


namespace NUMINAMATH_CALUDE_contradiction_assumption_for_at_most_two_even_l1835_183593

theorem contradiction_assumption_for_at_most_two_even 
  (a b c : ℕ) : 
  (¬ (∃ (x y : ℕ), {a, b, c} \ {x, y} ⊆ {n : ℕ | Even n})) ↔ 
  (Even a ∧ Even b ∧ Even c) :=
sorry

end NUMINAMATH_CALUDE_contradiction_assumption_for_at_most_two_even_l1835_183593


namespace NUMINAMATH_CALUDE_garden_perimeter_l1835_183541

/-- The perimeter of a rectangular garden with the same area as a given playground -/
theorem garden_perimeter (garden_width playground_length playground_width : ℝ) :
  garden_width = 4 ∧
  playground_length = 16 ∧
  playground_width = 12 →
  (garden_width * (playground_length * playground_width / garden_width) + garden_width) * 2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1835_183541


namespace NUMINAMATH_CALUDE_point_coordinates_l1835_183567

/-- A point in the coordinate plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the coordinate plane. -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance from a point to the x-axis. -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis. -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: If a point P is in the second quadrant, its distance to the x-axis is 4,
    and its distance to the y-axis is 5, then its coordinates are (-5, 4). -/
theorem point_coordinates (P : Point) 
    (h1 : SecondQuadrant P) 
    (h2 : DistanceToXAxis P = 4) 
    (h3 : DistanceToYAxis P = 5) : 
    P.x = -5 ∧ P.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1835_183567


namespace NUMINAMATH_CALUDE_log_equality_implies_base_l1835_183569

theorem log_equality_implies_base (y : ℝ) (h : y > 0) :
  (Real.log 8 / Real.log y = Real.log 5 / Real.log 125) → y = 512 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_base_l1835_183569


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1835_183501

theorem fraction_to_decimal : (49 : ℚ) / (2^3 * 5^4) = 6.125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1835_183501


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1835_183510

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 1 / 2 →
  n + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1835_183510


namespace NUMINAMATH_CALUDE_quadratic_sum_equations_l1835_183549

/-- Given two quadratic equations and their roots, prove the equations for the sums of roots -/
theorem quadratic_sum_equations 
  (a b c α β γ : ℝ) 
  (h1 : a * α ≠ 0) 
  (h2 : b^2 - 4*a*c ≥ 0) 
  (h3 : β^2 - 4*α*γ ≥ 0) 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : x₁ ≤ x₂) 
  (hy : y₁ ≤ y₂) 
  (hx_roots : a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) 
  (hy_roots : α * y₁^2 + β * y₁ + γ = 0 ∧ α * y₂^2 + β * y₂ + γ = 0) :
  ∃ (d δ : ℝ), 
    d^2 = b^2 - 4*a*c ∧ 
    δ^2 = β^2 - 4*α*γ ∧
    (∀ z, 2*a*α*z^2 + 2*(a*β + α*b)*z + (2*a*γ + 2*α*c + b*β - d*δ) = 0 ↔ 
      (z = x₁ + y₁ ∨ z = x₂ + y₂)) ∧
    (∀ u, 2*a*α*u^2 + 2*(a*β + α*b)*u + (2*a*γ + 2*α*c + b*β + d*δ) = 0 ↔ 
      (u = x₁ + y₂ ∨ u = x₂ + y₁)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sum_equations_l1835_183549


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_sum_less_than_nineteen_nineteen_is_smallest_l1835_183566

theorem smallest_whole_number_above_sum : ℕ → Prop :=
  fun n => (∃ (m : ℕ), m > n ∧ 
    (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/6 : ℚ) + (6 + 1/12 : ℚ) < m) ∧
    (∀ (k : ℕ), k < n → 
    (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/6 : ℚ) + (6 + 1/12 : ℚ) ≥ k)

theorem sum_less_than_nineteen :
  (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/6 : ℚ) + (6 + 1/12 : ℚ) < 19 :=
by sorry

theorem nineteen_is_smallest : smallest_whole_number_above_sum 19 :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_sum_less_than_nineteen_nineteen_is_smallest_l1835_183566


namespace NUMINAMATH_CALUDE_total_attendance_percentage_l1835_183526

/-- Represents the departments in the company -/
inductive Department
  | IT
  | HR
  | Marketing

/-- Represents the genders of employees -/
inductive Gender
  | Male
  | Female

/-- Attendance rate for each department and gender -/
def attendance_rate (d : Department) (g : Gender) : ℝ :=
  match d, g with
  | Department.IT, Gender.Male => 0.25
  | Department.IT, Gender.Female => 0.60
  | Department.HR, Gender.Male => 0.30
  | Department.HR, Gender.Female => 0.50
  | Department.Marketing, Gender.Male => 0.10
  | Department.Marketing, Gender.Female => 0.45

/-- Employee composition for each department and gender -/
def employee_composition (d : Department) (g : Gender) : ℝ :=
  match d, g with
  | Department.IT, Gender.Male => 0.40
  | Department.IT, Gender.Female => 0.25
  | Department.HR, Gender.Male => 0.30
  | Department.HR, Gender.Female => 0.20
  | Department.Marketing, Gender.Male => 0.30
  | Department.Marketing, Gender.Female => 0.55

/-- Calculate the total attendance percentage -/
def total_attendance : ℝ :=
  (attendance_rate Department.IT Gender.Male * employee_composition Department.IT Gender.Male) +
  (attendance_rate Department.IT Gender.Female * employee_composition Department.IT Gender.Female) +
  (attendance_rate Department.HR Gender.Male * employee_composition Department.HR Gender.Male) +
  (attendance_rate Department.HR Gender.Female * employee_composition Department.HR Gender.Female) +
  (attendance_rate Department.Marketing Gender.Male * employee_composition Department.Marketing Gender.Male) +
  (attendance_rate Department.Marketing Gender.Female * employee_composition Department.Marketing Gender.Female)

/-- Theorem: The total attendance percentage is 71.75% -/
theorem total_attendance_percentage : total_attendance = 0.7175 := by
  sorry

end NUMINAMATH_CALUDE_total_attendance_percentage_l1835_183526


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1835_183519

theorem unique_triple_solution : 
  ∃! (a b c : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
    a^2 + b^2 + c^2 = 3 ∧ 
    (a + b + c) * (a^2*b + b^2*c + c^2*a) = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1835_183519


namespace NUMINAMATH_CALUDE_diophantine_solutions_l1835_183547

/-- Theorem: Solutions for the Diophantine equations 3a + 5b = 1, 3a + 5b = 4, and 183a + 117b = 3 -/
theorem diophantine_solutions :
  (∀ (a b : ℤ), 3*a + 5*b = 1 ↔ ∃ (k : ℤ), a = 2 - 5*k ∧ b = -1 + 3*k) ∧
  (∀ (a b : ℤ), 3*a + 5*b = 4 ↔ ∃ (k : ℤ), a = 8 - 5*k ∧ b = -4 + 3*k) ∧
  (∀ (a b : ℤ), 183*a + 117*b = 3 ↔ ∃ (k : ℤ), a = 16 - 39*k ∧ b = -25 + 61*k) :=
sorry

/-- Lemma: The solution set for 3a + 5b = 1 is correct -/
lemma solution_set_3a_5b_1 (a b : ℤ) :
  3*a + 5*b = 1 ↔ ∃ (k : ℤ), a = 2 - 5*k ∧ b = -1 + 3*k :=
sorry

/-- Lemma: The solution set for 3a + 5b = 4 is correct -/
lemma solution_set_3a_5b_4 (a b : ℤ) :
  3*a + 5*b = 4 ↔ ∃ (k : ℤ), a = 8 - 5*k ∧ b = -4 + 3*k :=
sorry

/-- Lemma: The solution set for 183a + 117b = 3 is correct -/
lemma solution_set_183a_117b_3 (a b : ℤ) :
  183*a + 117*b = 3 ↔ ∃ (k : ℤ), a = 16 - 39*k ∧ b = -25 + 61*k :=
sorry

end NUMINAMATH_CALUDE_diophantine_solutions_l1835_183547


namespace NUMINAMATH_CALUDE_tommy_pencil_case_erasers_l1835_183598

/-- Represents the contents of Tommy's pencil case -/
structure PencilCase where
  total_items : ℕ
  pencils : ℕ
  pens : ℕ
  erasers : ℕ

/-- Theorem stating the number of erasers in Tommy's pencil case -/
theorem tommy_pencil_case_erasers (pc : PencilCase)
    (h1 : pc.total_items = 13)
    (h2 : pc.pens = 2 * pc.pencils)
    (h3 : pc.pencils = 4)
    (h4 : pc.total_items = pc.pencils + pc.pens + pc.erasers) :
    pc.erasers = 1 := by
  sorry

end NUMINAMATH_CALUDE_tommy_pencil_case_erasers_l1835_183598


namespace NUMINAMATH_CALUDE_present_value_log_formula_l1835_183554

theorem present_value_log_formula (c s P k n : ℝ) (h_pos : 0 < 1 + k) :
  P = c * s / (1 + k) ^ n →
  n = (Real.log (c * s / P)) / (Real.log (1 + k)) :=
by sorry

end NUMINAMATH_CALUDE_present_value_log_formula_l1835_183554


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l1835_183522

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ k : ℕ+, Nat.gcd (13 * k + 4) (8 * k + 3) = 3) ∧
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l1835_183522


namespace NUMINAMATH_CALUDE_lines_perpendicular_l1835_183507

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the two lines
def line1 (t : Triangle) (x y : Real) : Prop :=
  x * Real.sin t.A + t.a * y + t.c = 0

def line2 (t : Triangle) (x y : Real) : Prop :=
  t.b * x - y * Real.sin t.B + Real.sin t.C = 0

-- Theorem statement
theorem lines_perpendicular (t : Triangle) : 
  (∀ x y, line1 t x y → line2 t x y → False) ∨ 
  (∃ x y, line1 t x y ∧ line2 t x y) :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_l1835_183507


namespace NUMINAMATH_CALUDE_triangle_side_length_l1835_183505

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if a = 3, C = 120°, and the area S = 15√3/4, then c = 7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = 3 →
  C = 2 * π / 3 →  -- 120° in radians
  S = 15 * Real.sqrt 3 / 4 →
  S = 1/2 * a * b * Real.sin C →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1835_183505


namespace NUMINAMATH_CALUDE_deposit_calculation_l1835_183568

theorem deposit_calculation (total_cost remaining_amount : ℝ) 
  (h1 : total_cost = 550)
  (h2 : remaining_amount = 495)
  (h3 : remaining_amount = total_cost - 0.1 * total_cost) :
  0.1 * total_cost = 55 := by
  sorry

end NUMINAMATH_CALUDE_deposit_calculation_l1835_183568


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l1835_183588

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 7 y = 15 ∧ y = 56 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l1835_183588


namespace NUMINAMATH_CALUDE_intersection_implies_b_range_l1835_183512

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_implies_b_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) →
  -Real.sqrt 6 / 2 ≤ b ∧ b ≤ Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_b_range_l1835_183512


namespace NUMINAMATH_CALUDE_fourth_root_of_207360000_l1835_183551

theorem fourth_root_of_207360000 : Real.sqrt (Real.sqrt 207360000) = 120 := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_207360000_l1835_183551


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l1835_183545

theorem three_digit_number_theorem (x y z : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ x ≠ 0 →
  let n := 100 * x + 10 * y + z
  let sum_digits := x + y + z
  n / sum_digits = 13 ∧ n % sum_digits = 15 →
  n = 106 ∨ n = 145 ∨ n = 184 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l1835_183545


namespace NUMINAMATH_CALUDE_digits_making_864n_divisible_by_4_l1835_183585

theorem digits_making_864n_divisible_by_4 : 
  ∃! (s : Finset Nat), 
    (∀ n ∈ s, n < 10) ∧ 
    (∀ n, n ∈ s ↔ (864 * 10 + n) % 4 = 0) ∧
    s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_digits_making_864n_divisible_by_4_l1835_183585


namespace NUMINAMATH_CALUDE_f_monotonicity_when_k_zero_f_nonnegative_condition_exponential_fraction_inequality_l1835_183596

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) - 1 - 2 * x - k * x^2

theorem f_monotonicity_when_k_zero :
  ∀ x : ℝ, x > 0 → (deriv (f 0)) x > 0 ∧
  ∀ x : ℝ, x < 0 → (deriv (f 0)) x < 0 := by sorry

theorem f_nonnegative_condition :
  ∀ k : ℝ, (∀ x : ℝ, x ≥ 0 → f k x ≥ 0) ↔ k ≤ 2 := by sorry

theorem exponential_fraction_inequality :
  ∀ n : ℕ+, (Real.exp (2 * ↑n) - 1) / (Real.exp 2 - 1) ≥ (2 * ↑n^3 + ↑n) / 3 := by sorry

end NUMINAMATH_CALUDE_f_monotonicity_when_k_zero_f_nonnegative_condition_exponential_fraction_inequality_l1835_183596


namespace NUMINAMATH_CALUDE_chess_club_girls_l1835_183599

theorem chess_club_girls (total_members present_members : ℕ) 
  (h_total : total_members = 32)
  (h_present : present_members = 20)
  (h_attendance : ∃ (boys girls : ℕ), 
    boys + girls = total_members ∧ 
    boys + girls / 2 = present_members) :
  ∃ (girls : ℕ), girls = 24 ∧ 
  ∃ (boys : ℕ), boys + girls = total_members ∧ 
                boys + girls / 2 = present_members :=
sorry

end NUMINAMATH_CALUDE_chess_club_girls_l1835_183599


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1835_183506

theorem inequality_solution_set (x : ℝ) :
  (((x + 5) / 2) - 2 < (3 * x + 2) / 2) ↔ (x > -1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1835_183506


namespace NUMINAMATH_CALUDE_got_percentage_is_fifty_percent_l1835_183597

/-- Represents the vote counts for three books -/
structure VoteCounts where
  got : ℕ  -- Game of Thrones
  twi : ℕ  -- Twilight
  aotd : ℕ  -- The Art of the Deal

/-- Calculates the altered vote counts after throwing away votes -/
def alterVotes (vc : VoteCounts) : VoteCounts :=
  { got := vc.got,
    twi := vc.twi / 2,
    aotd := vc.aotd - (vc.aotd * 4 / 5) }

/-- Calculates the percentage of votes for Game of Thrones after alteration -/
def gotPercentage (vc : VoteCounts) : ℚ :=
  let altered := alterVotes vc
  altered.got * 100 / (altered.got + altered.twi + altered.aotd)

/-- Theorem stating that for the given vote counts, the percentage of
    altered votes for Game of Thrones is 50% -/
theorem got_percentage_is_fifty_percent :
  let original := VoteCounts.mk 10 12 20
  gotPercentage original = 50 := by sorry

end NUMINAMATH_CALUDE_got_percentage_is_fifty_percent_l1835_183597


namespace NUMINAMATH_CALUDE_inverse_f_27_equals_3_l1835_183558

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem inverse_f_27_equals_3 :
  ∀ f_inv : ℝ → ℝ, 
  (∀ x : ℝ, f_inv (f x) = x) ∧ (∀ y : ℝ, f (f_inv y) = y) → 
  f_inv 27 = 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_f_27_equals_3_l1835_183558


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1835_183562

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, 12 * x^2 - a * x > a^2) ↔
    (a > 0 ∧ (x < -a/4 ∨ x > a/3)) ∨
    (a = 0 ∧ x ≠ 0) ∨
    (a < 0 ∧ (x < a/3 ∨ x > -a/4)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1835_183562


namespace NUMINAMATH_CALUDE_front_wheel_revolutions_l1835_183538

/-- Given a front wheel with perimeter 30 and a back wheel with perimeter 20,
    if the back wheel revolves 360 times, then the front wheel revolves 240 times. -/
theorem front_wheel_revolutions
  (front_perimeter : ℕ) (back_perimeter : ℕ) (back_revolutions : ℕ)
  (h1 : front_perimeter = 30)
  (h2 : back_perimeter = 20)
  (h3 : back_revolutions = 360) :
  (back_perimeter * back_revolutions) / front_perimeter = 240 := by
  sorry

end NUMINAMATH_CALUDE_front_wheel_revolutions_l1835_183538


namespace NUMINAMATH_CALUDE_participation_schemes_with_restriction_l1835_183564

-- Define the number of students and competitions
def num_students : ℕ := 4
def num_competitions : ℕ := 4

-- Define a function to calculate the number of participation schemes
def participation_schemes (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else Nat.factorial n / Nat.factorial (n - k)

-- Theorem statement
theorem participation_schemes_with_restriction :
  participation_schemes (num_students - 1) (num_competitions - 1) *
  (num_students - 1) = 18 :=
sorry

end NUMINAMATH_CALUDE_participation_schemes_with_restriction_l1835_183564


namespace NUMINAMATH_CALUDE_midpoint_locus_l1835_183559

/-- The locus of midpoints between a fixed point and points on a circle -/
theorem midpoint_locus (P : ℝ × ℝ) (c : Set (ℝ × ℝ)) :
  P = (4, -2) →
  c = {(x, y) | x^2 + y^2 = 4} →
  {(x, y) | ∃ (a : ℝ × ℝ), a ∈ c ∧ (x, y) = ((P.1 + a.1) / 2, (P.2 + a.2) / 2)} =
  {(x, y) | (x - 2)^2 + (y + 1)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_l1835_183559


namespace NUMINAMATH_CALUDE_factorial_difference_l1835_183582

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1835_183582


namespace NUMINAMATH_CALUDE_correct_match_probability_l1835_183576

/-- The probability of correctly matching all items when one match is known --/
theorem correct_match_probability (n : ℕ) (h : n = 4) : 
  (1 : ℚ) / Nat.factorial (n - 1) = (1 : ℚ) / 6 :=
sorry

end NUMINAMATH_CALUDE_correct_match_probability_l1835_183576


namespace NUMINAMATH_CALUDE_not_both_odd_l1835_183590

theorem not_both_odd (m n : ℕ) (h : (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 2020) : 
  ¬(Odd m ∧ Odd n) := by
sorry

end NUMINAMATH_CALUDE_not_both_odd_l1835_183590


namespace NUMINAMATH_CALUDE_boys_in_school_after_increase_l1835_183579

/-- The number of boys in a school after an increase -/
def boys_after_increase (initial_boys : ℕ) (additional_boys : ℕ) : ℕ :=
  initial_boys + additional_boys

theorem boys_in_school_after_increase :
  boys_after_increase 214 910 = 1124 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_school_after_increase_l1835_183579


namespace NUMINAMATH_CALUDE_average_salary_feb_to_may_l1835_183523

def average_salary_jan_to_apr : ℝ := 8000
def salary_may : ℝ := 6500
def salary_jan : ℝ := 5700

theorem average_salary_feb_to_may :
  let total_jan_to_apr := average_salary_jan_to_apr * 4
  let total_feb_to_apr := total_jan_to_apr - salary_jan
  let total_feb_to_may := total_feb_to_apr + salary_may
  (total_feb_to_may / 4 : ℝ) = 8200 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_feb_to_may_l1835_183523


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1835_183560

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1835_183560


namespace NUMINAMATH_CALUDE_unanswered_test_theorem_l1835_183591

/-- The number of ways to complete an unanswered multiple-choice test -/
def unanswered_test_completions (num_questions : ℕ) (num_choices : ℕ) : ℕ := 1

/-- Theorem: For a test with 4 questions and 5 choices per question, 
    there is only one way to complete it if all questions are unanswered -/
theorem unanswered_test_theorem : 
  unanswered_test_completions 4 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_test_theorem_l1835_183591


namespace NUMINAMATH_CALUDE_min_distance_to_plane_l1835_183535

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  d : ℝ

/-- The distance between a point and a plane -/
def distPointToPlane (p : Point3D) (plane : Plane) : ℝ :=
  sorry

/-- The distance between two points -/
def distBetweenPoints (p1 p2 : Point3D) : ℝ :=
  sorry

theorem min_distance_to_plane (α β γ : Plane) (A P : Point3D) :
  -- Planes are mutually perpendicular
  (α.normal.x * β.normal.x + α.normal.y * β.normal.y + α.normal.z * β.normal.z = 0) →
  (β.normal.x * γ.normal.x + β.normal.y * γ.normal.y + β.normal.z * γ.normal.z = 0) →
  (γ.normal.x * α.normal.x + γ.normal.y * α.normal.y + γ.normal.z * α.normal.z = 0) →
  -- A is on plane α
  (distPointToPlane A α = 0) →
  -- Distance from A to plane β is 3
  (distPointToPlane A β = 3) →
  -- Distance from A to plane γ is 3
  (distPointToPlane A γ = 3) →
  -- P is on plane α
  (distPointToPlane P α = 0) →
  -- Distance from P to plane β is twice the distance from P to point A
  (distPointToPlane P β = 2 * distBetweenPoints P A) →
  -- The minimum distance from points on the trajectory of P to plane γ is 3 - √3
  (∃ (P' : Point3D), distPointToPlane P' α = 0 ∧
    distPointToPlane P' β = 2 * distBetweenPoints P' A ∧
    distPointToPlane P' γ = 3 - Real.sqrt 3 ∧
    ∀ (P'' : Point3D), distPointToPlane P'' α = 0 →
      distPointToPlane P'' β = 2 * distBetweenPoints P'' A →
      distPointToPlane P'' γ ≥ 3 - Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_plane_l1835_183535


namespace NUMINAMATH_CALUDE_beaumont_high_school_science_classes_beaumont_high_school_main_theorem_l1835_183581

/-- The number of players taking at least two sciences at Beaumont High School -/
theorem beaumont_high_school_science_classes (total_players : ℕ) 
  (biology_players : ℕ) (chemistry_players : ℕ) (physics_players : ℕ) 
  (all_three_players : ℕ) : ℕ :=
by
  sorry

/-- The main theorem about Beaumont High School science classes -/
theorem beaumont_high_school_main_theorem : 
  beaumont_high_school_science_classes 30 15 10 5 3 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_beaumont_high_school_science_classes_beaumont_high_school_main_theorem_l1835_183581


namespace NUMINAMATH_CALUDE_expression_factorization_l1835_183550

theorem expression_factorization (y : ℝ) :
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l1835_183550


namespace NUMINAMATH_CALUDE_rational_equal_to_reciprocal_l1835_183516

theorem rational_equal_to_reciprocal (x : ℚ) : x = 1 ∨ x = -1 ↔ x = 1 / x := by sorry

end NUMINAMATH_CALUDE_rational_equal_to_reciprocal_l1835_183516


namespace NUMINAMATH_CALUDE_scientific_notation_of_1040000000_l1835_183574

theorem scientific_notation_of_1040000000 :
  (1040000000 : ℝ) = 1.04 * (10 : ℝ)^9 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1040000000_l1835_183574


namespace NUMINAMATH_CALUDE_not_right_triangle_3_5_7_l1835_183563

/-- A function that checks if three numbers can form the sides of a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that (3, 5, 7) cannot form the sides of a right triangle -/
theorem not_right_triangle_3_5_7 : ¬ is_right_triangle 3 5 7 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_3_5_7_l1835_183563


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1835_183539

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1835_183539


namespace NUMINAMATH_CALUDE_middle_number_is_nine_l1835_183537

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem middle_number_is_nine (a b c : ℤ) : 
  is_odd a ∧ is_odd b ∧ is_odd c ∧  -- a, b, c are odd numbers
  b = a + 2 ∧ c = b + 2 ∧            -- a, b, c are consecutive
  a + b + c = a + 20                 -- sum is 20 more than first number
  → b = 9 := by sorry

end NUMINAMATH_CALUDE_middle_number_is_nine_l1835_183537


namespace NUMINAMATH_CALUDE_ferry_time_difference_l1835_183515

/-- Proves that the difference in travel time between Ferry Q and Ferry P is 1 hour -/
theorem ferry_time_difference
  (time_p : ℝ) (speed_p : ℝ) (speed_difference : ℝ) (route_factor : ℝ) :
  time_p = 3 →
  speed_p = 8 →
  speed_difference = 4 →
  route_factor = 2 →
  let distance_p := time_p * speed_p
  let distance_q := route_factor * distance_p
  let speed_q := speed_p + speed_difference
  let time_q := distance_q / speed_q
  time_q - time_p = 1 := by
sorry

end NUMINAMATH_CALUDE_ferry_time_difference_l1835_183515


namespace NUMINAMATH_CALUDE_total_net_increase_l1835_183517

/-- Represents a time period with birth and death rates -/
structure TimePeriod where
  birthRate : Nat
  deathRate : Nat

/-- Calculates the net population increase for a given time period -/
def netIncrease (tp : TimePeriod) : Nat :=
  (tp.birthRate - tp.deathRate) * 10800

/-- The four time periods in a day -/
def dayPeriods : List TimePeriod := [
  { birthRate := 4, deathRate := 3 },
  { birthRate := 8, deathRate := 3 },
  { birthRate := 10, deathRate := 4 },
  { birthRate := 6, deathRate := 2 }
]

/-- Theorem: The total net population increase in one day is 172,800 -/
theorem total_net_increase : 
  (dayPeriods.map netIncrease).sum = 172800 := by
  sorry

end NUMINAMATH_CALUDE_total_net_increase_l1835_183517


namespace NUMINAMATH_CALUDE_krystiana_earnings_l1835_183540

/-- Represents the monthly earnings from an apartment building --/
def apartment_earnings (
  first_floor_price : ℕ)
  (second_floor_price : ℕ)
  (first_floor_rooms : ℕ)
  (second_floor_rooms : ℕ)
  (third_floor_rooms : ℕ)
  (third_floor_occupied : ℕ) : ℕ :=
  (first_floor_price * first_floor_rooms) +
  (second_floor_price * second_floor_rooms) +
  (2 * first_floor_price * third_floor_occupied)

/-- Krystiana's apartment building earnings theorem --/
theorem krystiana_earnings :
  apartment_earnings 15 20 3 3 3 2 = 165 := by
  sorry

#eval apartment_earnings 15 20 3 3 3 2

end NUMINAMATH_CALUDE_krystiana_earnings_l1835_183540


namespace NUMINAMATH_CALUDE_remaining_money_l1835_183532

def initial_amount : ℚ := 2 * 20 + 2 * 10 + 3 * 5 + 2 * 1 + 4.5
def cake_cost : ℚ := 17.5
def gift_cost : ℚ := 12.7
def donation : ℚ := 5.3

theorem remaining_money :
  initial_amount - (cake_cost + gift_cost + donation) = 46 :=
by sorry

end NUMINAMATH_CALUDE_remaining_money_l1835_183532


namespace NUMINAMATH_CALUDE_white_balls_count_l1835_183594

/-- The number of white balls in a bag with specific conditions -/
theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) : 
  total = 100 ∧ 
  green = 20 ∧ 
  yellow = 10 ∧ 
  red = 17 ∧ 
  purple = 3 ∧ 
  prob_not_red_purple = 4/5 →
  ∃ white : ℕ, white = 50 ∧ total = white + green + yellow + red + purple :=
by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l1835_183594


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_18_factorial_l1835_183548

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_power_of_three_dividing (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc x => acc + (if x % 3 = 0 then 1 else 0) + (if x % 9 = 0 then 1 else 0)) 0

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_three_dividing_18_factorial :
  ones_digit (3^(largest_power_of_three_dividing (factorial 18))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_18_factorial_l1835_183548


namespace NUMINAMATH_CALUDE_equation_solution_l1835_183570

theorem equation_solution (x : ℝ) : x ≠ 1 →
  ((3 * x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)) ↔ (x = -4 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1835_183570


namespace NUMINAMATH_CALUDE_video_library_space_per_hour_l1835_183520

/-- Given a video library with the following properties:
  * Contains 15 days of videos
  * Each day consists of 18 hours of videos
  * The entire library takes up 45,000 megabytes of disk space
  This theorem proves that the disk space required for one hour of video,
  when rounded to the nearest whole number, is 167 megabytes. -/
theorem video_library_space_per_hour :
  ∀ (days hours_per_day total_space : ℕ),
  days = 15 →
  hours_per_day = 18 →
  total_space = 45000 →
  round ((total_space : ℝ) / (days * hours_per_day : ℝ)) = 167 := by
  sorry

end NUMINAMATH_CALUDE_video_library_space_per_hour_l1835_183520


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_conditions_l1835_183508

theorem three_digit_numbers_with_conditions :
  ∃ (a b : ℕ),
    100 ≤ a ∧ a < b ∧ b < 1000 ∧
    ∃ (k : ℕ), a + b = 498 * k ∧
    ∃ (m : ℕ), b = 5 * m * a ∧
    a = 166 ∧ b = 830 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_conditions_l1835_183508


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1835_183571

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ∧ 
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1835_183571


namespace NUMINAMATH_CALUDE_f_is_even_l1835_183575

-- Define g as an odd function
def g_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^2)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_odd g) : ∀ x, f g (-x) = f g x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1835_183575


namespace NUMINAMATH_CALUDE_fraction_equality_l1835_183565

theorem fraction_equality (a b : ℝ) (x : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : x = a / b) :
  (a + 2*b) / (a - 2*b) = (x + 2) / (x - 2) := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1835_183565


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1835_183536

/-- An ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure Ellipse where
  k : ℝ
  eq : ∀ x y : ℝ, x^2 + k * y^2 = 2
  foci_on_y : True  -- This is a placeholder for the foci condition

/-- The range of k for a valid ellipse with foci on the y-axis -/
def valid_k_range (e : Ellipse) : Prop :=
  0 < e.k ∧ e.k < 1

/-- Theorem stating that for any ellipse with the given properties, k must be in (0, 1) -/
theorem ellipse_k_range (e : Ellipse) : valid_k_range e := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1835_183536


namespace NUMINAMATH_CALUDE_age_ratio_change_l1835_183524

theorem age_ratio_change (man_age son_age : ℕ) (h1 : man_age = 36) (h2 : son_age = 12) 
  (h3 : man_age = 3 * son_age) : 
  ∃ y : ℕ, man_age + y = 2 * (son_age + y) ∧ y = 12 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_change_l1835_183524


namespace NUMINAMATH_CALUDE_garden_width_l1835_183542

/-- A rectangular garden with given length and area has a specific width. -/
theorem garden_width (length area : ℝ) (h1 : length = 12) (h2 : area = 60) :
  area / length = 5 := by
  sorry

end NUMINAMATH_CALUDE_garden_width_l1835_183542


namespace NUMINAMATH_CALUDE_lines_not_parallel_l1835_183525

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem lines_not_parallel : 
  let m : ℝ := -1
  let l1 : Line := { a := 1, b := m, c := 6 }
  let l2 : Line := { a := m - 2, b := 3, c := 2 * m }
  ¬ parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_lines_not_parallel_l1835_183525


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1835_183561

/-- Given a geometric sequence with first term a₁ and common ratio q,
    if the sum of the first two terms S₂ = 3a₁, then q = 2 -/
theorem geometric_sequence_common_ratio (a₁ : ℝ) (q : ℝ) :
  a₁ ≠ 0 → a₁ + a₁ * q = 3 * a₁ → q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1835_183561


namespace NUMINAMATH_CALUDE_final_retail_price_l1835_183544

/-- Calculates the final retail price of a machine given wholesale price, markup, discount, and desired profit percentage. -/
theorem final_retail_price
  (wholesale_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (desired_profit_percentage : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : markup_percentage = 1)
  (h3 : discount_percentage = 0.2)
  (h4 : desired_profit_percentage = 0.6)
  : wholesale_price * (1 + markup_percentage) * (1 - discount_percentage) = 144 :=
by sorry

end NUMINAMATH_CALUDE_final_retail_price_l1835_183544


namespace NUMINAMATH_CALUDE_factorial_8_divisors_l1835_183527

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- The number of positive divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factorial_8_divisors :
  numDivisors (factorial 8) = 96 := by
  sorry

end NUMINAMATH_CALUDE_factorial_8_divisors_l1835_183527


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1835_183511

/-- Quadratic function -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Definition of N -/
def N (a b c : ℝ) : ℝ := |a + b + c| + |2*a - b|

/-- Definition of M -/
def M (a b c : ℝ) : ℝ := |a - b + c| + |2*a + b|

theorem quadratic_inequality (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : -b/(2*a) > 1) 
  (h3 : f a b c 0 = c) 
  (h4 : ∃ x, f a b c x > 0) : 
  M a b c < N a b c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1835_183511


namespace NUMINAMATH_CALUDE_radio_station_survey_l1835_183513

theorem radio_station_survey (total_listeners total_non_listeners male_non_listeners female_listeners : ℕ) 
  (h1 : total_listeners = 180)
  (h2 : total_non_listeners = 160)
  (h3 : male_non_listeners = 85)
  (h4 : female_listeners = 75) :
  total_listeners - female_listeners = 105 := by
  sorry

end NUMINAMATH_CALUDE_radio_station_survey_l1835_183513


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1835_183531

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (3 - x^2)}

-- Define the closed interval [-1, √3]
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ Real.sqrt 3}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = interval := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1835_183531


namespace NUMINAMATH_CALUDE_doubled_speed_cleaning_time_l1835_183573

def house_cleaning (bruce_rate anne_rate : ℝ) : Prop :=
  bruce_rate > 0 ∧ anne_rate > 0 ∧
  bruce_rate + anne_rate = 1 / 4 ∧
  anne_rate = 1 / 12

theorem doubled_speed_cleaning_time (bruce_rate anne_rate : ℝ) 
  (h : house_cleaning bruce_rate anne_rate) : 
  1 / (bruce_rate + 2 * anne_rate) = 3 := by
  sorry

end NUMINAMATH_CALUDE_doubled_speed_cleaning_time_l1835_183573


namespace NUMINAMATH_CALUDE_simplify_fraction_l1835_183509

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2 + 1) = 16250 / 601 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1835_183509


namespace NUMINAMATH_CALUDE_sqrt_sum_equivalence_l1835_183534

theorem sqrt_sum_equivalence (n : ℝ) (h : Real.sqrt 15 = n) :
  Real.sqrt 0.15 + Real.sqrt 1500 = (101 / 10) * n := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equivalence_l1835_183534


namespace NUMINAMATH_CALUDE_peters_parrots_l1835_183587

/-- Calculates the number of parrots Peter has based on the given conditions -/
theorem peters_parrots :
  let parakeet_consumption : ℕ := 2 -- grams per day
  let parrot_consumption : ℕ := 14 -- grams per day
  let finch_consumption : ℕ := parakeet_consumption / 2 -- grams per day
  let num_parakeets : ℕ := 3
  let num_finches : ℕ := 4
  let total_birdseed : ℕ := 266 -- grams for a week
  let days_in_week : ℕ := 7

  let parakeet_weekly_consumption : ℕ := num_parakeets * parakeet_consumption * days_in_week
  let finch_weekly_consumption : ℕ := num_finches * finch_consumption * days_in_week
  let remaining_birdseed : ℕ := total_birdseed - parakeet_weekly_consumption - finch_weekly_consumption
  let parrot_weekly_consumption : ℕ := parrot_consumption * days_in_week

  remaining_birdseed / parrot_weekly_consumption = 2 :=
by sorry

end NUMINAMATH_CALUDE_peters_parrots_l1835_183587


namespace NUMINAMATH_CALUDE_average_of_middle_two_l1835_183529

theorem average_of_middle_two (numbers : Fin 6 → ℝ) 
  (h_total_avg : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 6.40)
  (h_first_two_avg : (numbers 0 + numbers 1) / 2 = 6.2)
  (h_last_two_avg : (numbers 4 + numbers 5) / 2 = 6.9) :
  (numbers 2 + numbers 3) / 2 = 6.1 := by
  sorry

end NUMINAMATH_CALUDE_average_of_middle_two_l1835_183529


namespace NUMINAMATH_CALUDE_complex_calculation_l1835_183514

theorem complex_calculation : 550 - (104 / (Real.sqrt 20.8)^2)^3 = 425 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l1835_183514


namespace NUMINAMATH_CALUDE_family_birth_years_l1835_183553

def current_year : ℕ := 1967

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def satisfies_condition (birth_year : ℕ) (multiplier : ℕ) : Prop :=
  current_year - birth_year = multiplier * sum_of_digits birth_year

theorem family_birth_years :
  ∃ (grandpa eldest_son father pali brother mother grandfather grandmother : ℕ),
    satisfies_condition grandpa 3 ∧
    satisfies_condition eldest_son 3 ∧
    satisfies_condition father 3 = false ∧
    satisfies_condition (father - 1) 3 ∧
    satisfies_condition grandfather 3 = false ∧
    satisfies_condition (grandfather - 1) 3 ∧
    satisfies_condition grandmother 3 = false ∧
    satisfies_condition (grandmother + 1) 3 ∧
    satisfies_condition mother 2 = false ∧
    satisfies_condition (mother - 1) 2 ∧
    satisfies_condition pali 1 ∧
    satisfies_condition brother 1 = false ∧
    satisfies_condition (brother - 1) 1 ∧
    grandpa = 1889 ∧
    eldest_son = 1916 ∧
    father = 1928 ∧
    pali = 1951 ∧
    brother = 1947 ∧
    mother = 1934 ∧
    grandfather = 1896 ∧
    grandmother = 1909 :=
by
  sorry

end NUMINAMATH_CALUDE_family_birth_years_l1835_183553


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l1835_183580

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of quadruplets
def quadruplets : ℕ := 4

-- Define the number of starters to choose
def starters : ℕ := 6

-- Define the maximum number of quadruplets allowed in the starting lineup
def max_quadruplets_in_lineup : ℕ := 1

-- Theorem statement
theorem volleyball_team_starters :
  (Nat.choose (total_players - quadruplets) starters) +
  (quadruplets * Nat.choose (total_players - quadruplets) (starters - 1)) = 4092 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l1835_183580


namespace NUMINAMATH_CALUDE_transistors_in_1995_l1835_183577

/-- Moore's law states that the number of transistors doubles every 18 months -/
def moores_law_period : ℕ := 18

/-- Initial year when the count began -/
def initial_year : ℕ := 1985

/-- Initial number of transistors -/
def initial_transistors : ℕ := 500000

/-- Target year for calculation -/
def target_year : ℕ := 1995

/-- Calculate the number of transistors based on Moore's law -/
def calculate_transistors (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ (months / moores_law_period))

/-- Theorem stating that the number of transistors in 1995 is 32,000,000 -/
theorem transistors_in_1995 :
  calculate_transistors initial_transistors ((target_year - initial_year) * 12) = 32000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_1995_l1835_183577


namespace NUMINAMATH_CALUDE_corner_difference_divisible_by_six_l1835_183528

/-- A 9x9 table filled with numbers from 1 to 81 -/
def Table := Fin 9 → Fin 9 → Fin 81

/-- Check if two cells are adjacent -/
def adjacent (i j k l : Fin 9) : Prop :=
  (i = k ∧ (j = l + 1 ∨ j + 1 = l)) ∨ (j = l ∧ (i = k + 1 ∨ i + 1 = k))

/-- Check if a number is in a corner cell -/
def isCorner (i j : Fin 9) : Prop :=
  (i = 0 ∨ i = 8) ∧ (j = 0 ∨ j = 8)

/-- The main theorem -/
theorem corner_difference_divisible_by_six (t : Table) 
  (h1 : ∀ i j k l, adjacent i j k l → (t i j : ℕ) + 3 = t k l ∨ (t i j : ℕ) = (t k l : ℕ) + 3)
  (h2 : ∀ i j k l, i ≠ k ∨ j ≠ l → t i j ≠ t k l) :
  ∃ i j k l, isCorner i j ∧ isCorner k l ∧ 
    (∃ m : ℕ, (t i j : ℕ) - (t k l : ℕ) = 6 * m ∨ (t k l : ℕ) - (t i j : ℕ) = 6 * m) :=
sorry

end NUMINAMATH_CALUDE_corner_difference_divisible_by_six_l1835_183528


namespace NUMINAMATH_CALUDE_max_value_of_cyclic_sum_l1835_183502

theorem max_value_of_cyclic_sum (a b c d e f : ℝ) 
  (non_neg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ 0 ≤ f) 
  (sum_constraint : a + b + c + d + e + f = 6) : 
  a * b * c + b * c * d + c * d * e + d * e * f + e * f * a + f * a * b ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_cyclic_sum_l1835_183502


namespace NUMINAMATH_CALUDE_total_pencils_l1835_183586

/-- Given that each child has 2 pencils and there are 11 children, 
    prove that the total number of pencils is 22. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) 
  (h2 : num_children = 11) : 
  pencils_per_child * num_children = 22 := by
sorry

end NUMINAMATH_CALUDE_total_pencils_l1835_183586


namespace NUMINAMATH_CALUDE_dinos_remaining_balance_l1835_183572

/-- Represents a gig with hours worked per month and hourly rate -/
structure Gig where
  hours : ℕ
  rate : ℕ

/-- Calculates the monthly earnings from a gig -/
def monthlyEarnings (g : Gig) : ℕ := g.hours * g.rate

/-- Represents Dino's gigs -/
def dinos_gigs : List Gig := [
  ⟨20, 10⟩,
  ⟨30, 20⟩,
  ⟨5, 40⟩,
  ⟨15, 25⟩,
  ⟨10, 30⟩
]

/-- Dino's monthly expenses for each month -/
def monthly_expenses : List ℕ := [500, 550, 520, 480]

/-- The number of months -/
def num_months : ℕ := 4

theorem dinos_remaining_balance :
  (dinos_gigs.map monthlyEarnings).sum * num_months -
  monthly_expenses.sum = 4650 := by sorry

end NUMINAMATH_CALUDE_dinos_remaining_balance_l1835_183572


namespace NUMINAMATH_CALUDE_score_difference_l1835_183521

def blue_free_throws : ℕ := 18
def blue_two_pointers : ℕ := 25
def blue_three_pointers : ℕ := 6

def red_free_throws : ℕ := 15
def red_two_pointers : ℕ := 22
def red_three_pointers : ℕ := 5

def blue_score : ℕ := blue_free_throws + 2 * blue_two_pointers + 3 * blue_three_pointers
def red_score : ℕ := red_free_throws + 2 * red_two_pointers + 3 * red_three_pointers

theorem score_difference : blue_score - red_score = 12 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_l1835_183521


namespace NUMINAMATH_CALUDE_reading_time_difference_is_360_l1835_183530

/-- Calculates the difference in reading time between two people in minutes -/
def reading_time_difference (xanthia_speed molly_speed book_pages : ℕ) : ℕ :=
  let xanthia_time := book_pages / xanthia_speed
  let molly_time := book_pages / molly_speed
  (molly_time - xanthia_time) * 60

/-- The difference in reading time between Molly and Xanthia is 360 minutes -/
theorem reading_time_difference_is_360 :
  reading_time_difference 120 40 360 = 360 :=
by sorry

end NUMINAMATH_CALUDE_reading_time_difference_is_360_l1835_183530


namespace NUMINAMATH_CALUDE_find_m_l1835_183557

theorem find_m : ∃ m : ℝ, 
  (∀ x : ℝ, mx + 3 = x ↔ 5 - 2*x = 1) → m = -1/2 := by sorry

end NUMINAMATH_CALUDE_find_m_l1835_183557


namespace NUMINAMATH_CALUDE_center_is_eight_l1835_183578

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def isAdjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Check if a grid is valid according to the problem conditions --/
def isValidGrid (g : Grid) : Prop :=
  (∀ n : Fin 9, ∃! p : Fin 3 × Fin 3, g p.1 p.2 = n.val + 1) ∧
  (∀ n : Fin 8, ∃ p1 p2 : Fin 3 × Fin 3, 
    g p1.1 p1.2 = n.val + 1 ∧ 
    g p2.1 p2.2 = n.val + 2 ∧ 
    isAdjacent p1 p2) ∧
  (g 0 0 + g 0 2 + g 2 0 + g 2 2 = 24)

theorem center_is_eight (g : Grid) (h : isValidGrid g) : 
  g 1 1 = 8 := by sorry

end NUMINAMATH_CALUDE_center_is_eight_l1835_183578


namespace NUMINAMATH_CALUDE_valid_distributions_count_l1835_183546

-- Define the number of students
def num_students : ℕ := 4

-- Define the number of days
def num_days : ℕ := 2

-- Function to calculate the number of valid distributions
def count_valid_distributions (students : ℕ) (days : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem valid_distributions_count :
  count_valid_distributions num_students num_days = 14 := by
  sorry

end NUMINAMATH_CALUDE_valid_distributions_count_l1835_183546


namespace NUMINAMATH_CALUDE_white_ball_count_l1835_183555

/-- Given a bag with red and white balls, if the probability of drawing a red ball
    is 1/4 and there are 5 red balls, prove that there are 15 white balls. -/
theorem white_ball_count (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
    (h1 : red_balls = 5)
    (h2 : total_balls = red_balls + white_balls)
    (h3 : (red_balls : ℚ) / total_balls = 1 / 4) :
  white_balls = 15 := by
  sorry

end NUMINAMATH_CALUDE_white_ball_count_l1835_183555


namespace NUMINAMATH_CALUDE_reflection_distance_l1835_183518

/-- Given a point A with coordinates (1, -3), prove that the distance between A
    and its reflection A' over the y-axis is 2. -/
theorem reflection_distance : 
  let A : ℝ × ℝ := (1, -3)
  let A' : ℝ × ℝ := (-1, -3)  -- Reflection of A over y-axis
  ‖A - A'‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_reflection_distance_l1835_183518


namespace NUMINAMATH_CALUDE_range_of_a_l1835_183503

def S (a : ℝ) := {x : ℝ | x^2 ≤ a}

theorem range_of_a (a : ℝ) : (∅ ⊂ S a) → a ∈ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1835_183503


namespace NUMINAMATH_CALUDE_nearest_integer_to_sum_l1835_183592

theorem nearest_integer_to_sum (x y : ℝ) 
  (h1 : abs x - y = 5)
  (h2 : abs x * y - x^2 = -12) : 
  round (x + y) = -5 := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_sum_l1835_183592
