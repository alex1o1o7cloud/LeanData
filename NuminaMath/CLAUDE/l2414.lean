import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_asymptote_theorem_l2414_241483

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola a b) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The focus-to-asymptote distance equals the real axis length -/
def focus_asymptote_condition (h : Hyperbola a b) : Prop :=
  ∃ c, c^2 = a^2 + b^2 ∧ b * c / (a^2 + b^2).sqrt = 2 * a

/-- The equation of the asymptote -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = 2 * x ∨ y = -2 * x

/-- Theorem: If the focus-to-asymptote distance equals the real axis length,
    then the asymptote equation is y = ±2x -/
theorem hyperbola_asymptote_theorem (a b : ℝ) (h : Hyperbola a b) :
  focus_asymptote_condition h → ∀ x y, asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_theorem_l2414_241483


namespace NUMINAMATH_CALUDE_gcd_problem_l2414_241405

theorem gcd_problem : Nat.gcd 7260 540 - 12 + 5 = 53 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2414_241405


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l2414_241436

theorem power_of_power_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l2414_241436


namespace NUMINAMATH_CALUDE_arrangement_theorem_l2414_241460

/-- The number of ways to arrange 4 students into 2 classes out of 6 --/
def arrangement_count : ℕ := 90

/-- The total number of classes --/
def total_classes : ℕ := 6

/-- The number of classes to be selected --/
def selected_classes : ℕ := 2

/-- The total number of students to be arranged --/
def total_students : ℕ := 4

/-- The number of students per selected class --/
def students_per_class : ℕ := 2

theorem arrangement_theorem :
  arrangement_count = 
    (Nat.choose total_classes selected_classes) * 
    (Nat.choose total_students students_per_class) :=
sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l2414_241460


namespace NUMINAMATH_CALUDE_twelve_eat_both_l2414_241499

/-- Represents the eating habits in a family -/
structure FamilyEatingHabits where
  only_veg : ℕ
  only_non_veg : ℕ
  total_veg : ℕ

/-- Calculates the number of people who eat both veg and non-veg -/
def both_veg_and_non_veg (habits : FamilyEatingHabits) : ℕ :=
  habits.total_veg - habits.only_veg

/-- Theorem: In the given family, 12 people eat both veg and non-veg -/
theorem twelve_eat_both (habits : FamilyEatingHabits) 
    (h1 : habits.only_veg = 19)
    (h2 : habits.only_non_veg = 9)
    (h3 : habits.total_veg = 31) :
    both_veg_and_non_veg habits = 12 := by
  sorry

#eval both_veg_and_non_veg ⟨19, 9, 31⟩

end NUMINAMATH_CALUDE_twelve_eat_both_l2414_241499


namespace NUMINAMATH_CALUDE_marble_product_l2414_241458

theorem marble_product (red blue : ℕ) : 
  (red - blue = 12) →
  (red + blue = red - blue + 40) →
  red * blue = 640 := by
sorry

end NUMINAMATH_CALUDE_marble_product_l2414_241458


namespace NUMINAMATH_CALUDE_soda_quarters_l2414_241498

/-- Represents the number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- Represents the total amount paid in dollars -/
def total_paid : ℕ := 4

/-- Represents the number of quarters paid for chips -/
def quarters_for_chips : ℕ := 4

/-- Calculates the number of quarters paid for soda -/
def quarters_for_soda : ℕ := (total_paid - quarters_for_chips / quarters_per_dollar) * quarters_per_dollar

theorem soda_quarters : quarters_for_soda = 12 := by
  sorry

end NUMINAMATH_CALUDE_soda_quarters_l2414_241498


namespace NUMINAMATH_CALUDE_divisible_by_seventeen_l2414_241422

theorem divisible_by_seventeen (n : ℕ) : 17 ∣ (2^(5*n+3) + 5^n * 3^(n+2)) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seventeen_l2414_241422


namespace NUMINAMATH_CALUDE_division_problem_l2414_241485

theorem division_problem (dividend : Nat) (divisor : Nat) (remainder : Nat) (quotient : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 34 →
  divisor = 7 →
  remainder = 6 →
  quotient = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2414_241485


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l2414_241489

theorem min_value_sum_fractions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) ≥ 7 ∧
  ((x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) = 7 ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l2414_241489


namespace NUMINAMATH_CALUDE_anyas_age_l2414_241473

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem anyas_age :
  ∃ (age : ℕ), 
    110 ≤ sum_of_first_n age ∧ 
    sum_of_first_n age ≤ 130 ∧ 
    age = 15 := by
  sorry

end NUMINAMATH_CALUDE_anyas_age_l2414_241473


namespace NUMINAMATH_CALUDE_percentage_of_workday_in_meetings_l2414_241469

-- Define the workday duration in minutes
def workday_minutes : ℕ := 8 * 60

-- Define the duration of the first meeting
def first_meeting_duration : ℕ := 30

-- Define the duration of the second meeting
def second_meeting_duration : ℕ := 2 * first_meeting_duration

-- Define the duration of the third meeting
def third_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration

-- Define the total time spent in meetings
def total_meeting_time : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

-- Theorem to prove the percentage of workday spent in meetings
theorem percentage_of_workday_in_meetings :
  (total_meeting_time : ℚ) / workday_minutes * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_workday_in_meetings_l2414_241469


namespace NUMINAMATH_CALUDE_johnny_work_hours_l2414_241411

theorem johnny_work_hours (hourly_wage : ℝ) (total_earned : ℝ) (hours_worked : ℝ) : 
  hourly_wage = 2.35 →
  total_earned = 11.75 →
  hours_worked = total_earned / hourly_wage →
  hours_worked = 5 := by
sorry

end NUMINAMATH_CALUDE_johnny_work_hours_l2414_241411


namespace NUMINAMATH_CALUDE_power_inequality_l2414_241425

theorem power_inequality (x y : ℝ) (h : x^2013 + y^2013 > x^2012 + y^2012) :
  x^2014 + y^2014 > x^2013 + y^2013 :=
by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2414_241425


namespace NUMINAMATH_CALUDE_max_value_mx_plus_ny_l2414_241472

theorem max_value_mx_plus_ny (a b : ℝ) (m n x y : ℝ) 
  (h1 : m^2 + n^2 = a) (h2 : x^2 + y^2 = b) :
  (∃ (k : ℝ), k = m*x + n*y ∧ ∀ (p q : ℝ), p^2 + q^2 = a → ∀ (r s : ℝ), r^2 + s^2 = b → 
    p*r + q*s ≤ k) → k = Real.sqrt (a*b) :=
sorry

end NUMINAMATH_CALUDE_max_value_mx_plus_ny_l2414_241472


namespace NUMINAMATH_CALUDE_simplify_radical_sum_l2414_241461

theorem simplify_radical_sum : Real.sqrt 50 + Real.sqrt 18 = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l2414_241461


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2414_241424

theorem division_remainder_problem (a b q r : ℕ) 
  (h1 : a - b = 1335)
  (h2 : a = 1584)
  (h3 : a = q * b + r)
  (h4 : q = 6)
  (h5 : r < b) :
  r = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2414_241424


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2414_241475

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h_sum1 : a 2 + a 3 = 1) (h_sum2 : a 3 + a 4 = -2) :
  a 5 + a 6 + a 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2414_241475


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2414_241431

theorem other_root_of_quadratic (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (a+b+c)*x + ab + bc + ca
  let ab := a + b
  let ab_bc_ca := ab + bc + ca
  f ab = 0 →
  ∃ k, f k = 0 ∧ k = (ab + bc + ca) / (a + b) :=
by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2414_241431


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l2414_241491

-- Define the sets M and N
def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x > -1}

-- Define the universal set U
def U : Type := ℝ

-- State the theorem
theorem complement_intersection_M_N :
  (M ∩ N)ᶜ = {x : ℝ | x ≤ -1 ∨ x ≥ 3} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l2414_241491


namespace NUMINAMATH_CALUDE_line_segment_ratio_l2414_241451

theorem line_segment_ratio (a b c d : ℝ) :
  let O := 0
  let A := a
  let B := b
  let C := c
  let D := d
  ∀ P : ℝ, B < P ∧ P < C →
  (P - A) / (D - P) = (P - B) / (C - P) →
  P = (a * c - b * d) / (a - b + c - d) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l2414_241451


namespace NUMINAMATH_CALUDE_spelling_contest_questions_l2414_241447

/-- Represents the number of questions in a spelling contest -/
structure SpellingContest where
  drew_correct : Nat
  drew_wrong : Nat
  carla_correct : Nat
  carla_wrong : Nat

/-- The total number of questions in the spelling contest -/
def total_questions (contest : SpellingContest) : Nat :=
  contest.drew_correct + contest.drew_wrong + contest.carla_correct + contest.carla_wrong

/-- Theorem stating the total number of questions in the given spelling contest -/
theorem spelling_contest_questions : ∃ (contest : SpellingContest),
  contest.drew_correct = 20 ∧
  contest.drew_wrong = 6 ∧
  contest.carla_correct = 14 ∧
  contest.carla_wrong = 2 * contest.drew_wrong ∧
  total_questions contest = 52 := by
  sorry

end NUMINAMATH_CALUDE_spelling_contest_questions_l2414_241447


namespace NUMINAMATH_CALUDE_age_difference_proof_l2414_241423

theorem age_difference_proof : ∃! n : ℝ, n > 0 ∧ 
  ∃ A C : ℝ, A > 0 ∧ C > 0 ∧ 
    A = C + n ∧ 
    A - 2 = 4 * (C - 2) ∧ 
    A = C^3 ∧ 
    n = 1.875 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2414_241423


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2414_241409

theorem midpoint_trajectory (a b : ℝ) : 
  a^2 + b^2 = 1 → ∃ x y : ℝ, x = a ∧ y = b/2 ∧ x^2 + 4*y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2414_241409


namespace NUMINAMATH_CALUDE_x_plus_y_values_l2414_241478

theorem x_plus_y_values (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 40) :
  x + y = 2 + 2 * Real.sqrt 3 ∨ x + y = 2 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l2414_241478


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2414_241417

theorem complex_magnitude_problem (z : ℂ) (h : (1 - I) * z = 1 + I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2414_241417


namespace NUMINAMATH_CALUDE_alice_savings_l2414_241408

/-- Alice's savings problem -/
theorem alice_savings (B : ℕ) : 
  let first_month := 10
  let second_month := first_month + 30 + B
  let third_month := first_month + 60
  first_month + second_month + third_month = 120 + B :=
by sorry

end NUMINAMATH_CALUDE_alice_savings_l2414_241408


namespace NUMINAMATH_CALUDE_dimes_per_machine_is_100_l2414_241481

/-- Represents the number of machines in the launderette -/
def num_machines : ℕ := 3

/-- Represents the number of quarters in each machine -/
def quarters_per_machine : ℕ := 80

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the total amount of money from all machines in cents -/
def total_money : ℕ := 9000  -- $90 in cents

/-- Calculates the number of dimes in each machine -/
def dimes_per_machine : ℕ :=
  (total_money - num_machines * quarters_per_machine * quarter_value) / (num_machines * dime_value)

/-- Theorem stating that the number of dimes in each machine is 100 -/
theorem dimes_per_machine_is_100 : dimes_per_machine = 100 := by
  sorry

end NUMINAMATH_CALUDE_dimes_per_machine_is_100_l2414_241481


namespace NUMINAMATH_CALUDE_triangle_acute_from_sine_ratio_l2414_241421

theorem triangle_acute_from_sine_ratio (A B C : ℝ) (h_triangle : A + B + C = Real.pi)
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) (h_sine_ratio : ∃ k : ℝ, k > 0 ∧ Real.sin A = 5 * k ∧ Real.sin B = 11 * k ∧ Real.sin C = 13 * k) :
  A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_acute_from_sine_ratio_l2414_241421


namespace NUMINAMATH_CALUDE_intersection_P_M_l2414_241487

def P : Set ℤ := {x | 0 ≤ x ∧ x < 3}
def M : Set ℤ := {x | x^2 ≤ 9}

theorem intersection_P_M : P ∩ M = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_M_l2414_241487


namespace NUMINAMATH_CALUDE_alloy_mixture_theorem_l2414_241434

/-- The percentage of chromium in the first alloy -/
def chromium_percent_1 : ℝ := 12

/-- The percentage of chromium in the second alloy -/
def chromium_percent_2 : ℝ := 8

/-- The amount of the first alloy used (in kg) -/
def amount_1 : ℝ := 20

/-- The percentage of chromium in the new alloy -/
def new_chromium_percent : ℝ := 9.454545454545453

/-- The amount of the second alloy used (in kg) -/
def amount_2 : ℝ := 35

theorem alloy_mixture_theorem :
  chromium_percent_1 * amount_1 / 100 + chromium_percent_2 * amount_2 / 100 =
  new_chromium_percent * (amount_1 + amount_2) / 100 := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_theorem_l2414_241434


namespace NUMINAMATH_CALUDE_polynomial_value_at_zero_l2414_241463

def polynomial_condition (p : ℝ → ℝ) : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ),
    ∀ x, p x = a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

theorem polynomial_value_at_zero (p : ℝ → ℝ) :
  polynomial_condition p →
  (∀ n : Nat, n ≤ 7 → p (3^n) = 1 / 3^n) →
  p 0 = 3280 / 2187 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_at_zero_l2414_241463


namespace NUMINAMATH_CALUDE_density_of_cube_root_differences_l2414_241426

theorem density_of_cube_root_differences :
  ∀ ε > 0, ∀ x : ℝ, ∃ n m : ℕ, |x - (n^(1/3) - m^(1/3))| < ε :=
sorry

end NUMINAMATH_CALUDE_density_of_cube_root_differences_l2414_241426


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2414_241455

theorem repeating_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) + (6 / 9999 : ℚ) = 3793 / 9999 := by
  sorry

#check repeating_decimal_sum

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2414_241455


namespace NUMINAMATH_CALUDE_pipe_A_rate_l2414_241414

/-- The rate at which Pipe A fills the tank -/
def rate_A : ℝ := 40

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 800

/-- The rate at which Pipe B fills the tank in liters per minute -/
def rate_B : ℝ := 30

/-- The rate at which Pipe C drains the tank in liters per minute -/
def rate_C : ℝ := 20

/-- The time in minutes it takes to fill the tank -/
def fill_time : ℝ := 48

/-- The duration of one cycle in minutes -/
def cycle_duration : ℝ := 3

theorem pipe_A_rate : 
  rate_A = 40 ∧ 
  (fill_time / cycle_duration) * (rate_A + rate_B - rate_C) = tank_capacity := by
  sorry

end NUMINAMATH_CALUDE_pipe_A_rate_l2414_241414


namespace NUMINAMATH_CALUDE_least_months_to_triple_l2414_241464

/-- The initial borrowed amount in dollars -/
def initial_amount : ℝ := 1000

/-- The monthly interest rate as a decimal -/
def monthly_rate : ℝ := 0.06

/-- The function that calculates the amount owed after t months -/
def amount_owed (t : ℕ) : ℝ := initial_amount * (1 + monthly_rate) ^ t

/-- Theorem stating that 17 is the least number of months for which the amount owed exceeds three times the initial amount -/
theorem least_months_to_triple : 
  (∀ k : ℕ, k < 17 → amount_owed k ≤ 3 * initial_amount) ∧ 
  amount_owed 17 > 3 * initial_amount :=
sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l2414_241464


namespace NUMINAMATH_CALUDE_sally_balloons_count_l2414_241419

/-- The number of blue balloons Joan initially has -/
def initial_balloons : ℕ := 9

/-- The number of blue balloons Joan gives to Jessica -/
def balloons_given_away : ℕ := 2

/-- The number of blue balloons Joan has after all transactions -/
def final_balloons : ℕ := 12

/-- The number of blue balloons Sally gives to Joan -/
def balloons_from_sally : ℕ := 5

theorem sally_balloons_count : 
  initial_balloons + balloons_from_sally - balloons_given_away = final_balloons :=
by sorry

end NUMINAMATH_CALUDE_sally_balloons_count_l2414_241419


namespace NUMINAMATH_CALUDE_ten_attendants_used_both_l2414_241474

/-- The number of attendants who used both a pencil and a pen at a meeting -/
def attendants_using_both (pencil_users pen_users only_one_tool_users : ℕ) : ℕ :=
  (pencil_users + pen_users - only_one_tool_users) / 2

/-- Theorem stating that 10 attendants used both a pencil and a pen -/
theorem ten_attendants_used_both :
  attendants_using_both 25 15 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_attendants_used_both_l2414_241474


namespace NUMINAMATH_CALUDE_total_shingles_needed_l2414_241471

/-- Represents the dimensions of a rectangular roof side -/
structure RoofSide where
  length : ℕ
  width : ℕ

/-- Represents a roof with two identical slanted sides and shingle requirement -/
structure Roof where
  side : RoofSide
  shingles_per_sqft : ℕ

/-- Calculates the number of shingles needed for a roof -/
def shingles_needed (roof : Roof) : ℕ :=
  2 * roof.side.length * roof.side.width * roof.shingles_per_sqft

/-- The three roofs in the problem -/
def roof_A : Roof := { side := { length := 20, width := 40 }, shingles_per_sqft := 8 }
def roof_B : Roof := { side := { length := 25, width := 35 }, shingles_per_sqft := 10 }
def roof_C : Roof := { side := { length := 30, width := 30 }, shingles_per_sqft := 12 }

/-- Theorem stating the total number of shingles needed for all three roofs -/
theorem total_shingles_needed :
  shingles_needed roof_A + shingles_needed roof_B + shingles_needed roof_C = 51900 := by
  sorry

end NUMINAMATH_CALUDE_total_shingles_needed_l2414_241471


namespace NUMINAMATH_CALUDE_pointA_in_region_l2414_241407

/-- A point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The region defined by the inequality 2x + y - 6 < 0 -/
def region (p : Point) : Prop :=
  2 * p.x + p.y - 6 < 0

/-- The point (0, 1) -/
def pointA : Point :=
  { x := 0, y := 1 }

/-- Theorem: The point (0, 1) lies within the region defined by 2x + y - 6 < 0 -/
theorem pointA_in_region : region pointA := by
  sorry

end NUMINAMATH_CALUDE_pointA_in_region_l2414_241407


namespace NUMINAMATH_CALUDE_common_ratio_is_three_l2414_241418

/-- An arithmetic-geometric sequence with its properties -/
structure ArithGeomSeq where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  q : ℝ      -- Common ratio
  h1 : a 3 = 2 * S 2 + 1
  h2 : a 4 = 2 * S 3 + 1
  h3 : ∀ n : ℕ, n ≥ 2 → a (n+1) = q * a n

/-- The common ratio of the arithmetic-geometric sequence is 3 -/
theorem common_ratio_is_three (seq : ArithGeomSeq) : seq.q = 3 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_is_three_l2414_241418


namespace NUMINAMATH_CALUDE_center_sum_is_ten_l2414_241470

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def center (c : Circle) : ℝ × ℝ := sorry

/-- The sum of the coordinates of a point -/
def coord_sum (p : ℝ × ℝ) : ℝ := p.1 + p.2

/-- The specific circle from the problem -/
def problem_circle : Circle :=
  { equation := λ x y ↦ x^2 + y^2 - 6*x - 14*y + 24 = 0 }

theorem center_sum_is_ten :
  coord_sum (center problem_circle) = 10 := by sorry

end NUMINAMATH_CALUDE_center_sum_is_ten_l2414_241470


namespace NUMINAMATH_CALUDE_nineteenth_replacement_in_july_l2414_241453

/-- Represents the months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Calculates the number of months between two replacements -/
def monthsBetweenReplacements : ℕ := 7

/-- Calculates the total number of months after a given number of replacements -/
def totalMonthsAfter (replacements : ℕ) : ℕ :=
  monthsBetweenReplacements * (replacements - 1)

/-- Determines the month after a given number of months from January -/
def monthAfter (months : ℕ) : Month :=
  match months % 12 with
  | 0 => Month.January
  | 1 => Month.February
  | 2 => Month.March
  | 3 => Month.April
  | 4 => Month.May
  | 5 => Month.June
  | 6 => Month.July
  | 7 => Month.August
  | 8 => Month.September
  | 9 => Month.October
  | 10 => Month.November
  | _ => Month.December

/-- Theorem: The 19th replacement occurs in July -/
theorem nineteenth_replacement_in_july :
  monthAfter (totalMonthsAfter 19) = Month.July := by
  sorry

end NUMINAMATH_CALUDE_nineteenth_replacement_in_july_l2414_241453


namespace NUMINAMATH_CALUDE_triangle_theorem_triangle_range_theorem_l2414_241420

noncomputable section

def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (A + B + C = Real.pi) ∧ 
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem triangle_theorem 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_equation : (Real.cos B - 2 * Real.cos A) / (2 * a - b) = Real.cos C / c) :
  a / b = 2 := 
sorry

theorem triangle_range_theorem 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_equation : (Real.cos B - 2 * Real.cos A) / (2 * a - b) = Real.cos C / c)
  (h_obtuse : A > Real.pi / 2)
  (h_c : c = 3) :
  Real.sqrt 3 < b ∧ b < 3 := 
sorry

end NUMINAMATH_CALUDE_triangle_theorem_triangle_range_theorem_l2414_241420


namespace NUMINAMATH_CALUDE_spatial_relations_l2414_241435

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelPlaneLine : Plane → Line → Prop)
variable (perpendicularPlaneLine : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem spatial_relations 
  (m n l : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (h_distinct_planes : α ≠ β) :
  -- Define the propositions
  let p1 := ∀ m n α, parallel m n → contains α n → parallelPlaneLine α m
  let p2 := ∀ l m α β, perpendicularPlaneLine α l → perpendicularPlaneLine β m → perpendicular l m → perpendicularPlanes α β
  let p3 := ∀ l m n, perpendicular l n → perpendicular m n → parallel l m
  let p4 := ∀ m n α β, perpendicularPlanes α β → intersect α β m → contains β n → perpendicular n m → perpendicularPlaneLine α n
  -- The theorem statement
  (¬p1 ∧ p2 ∧ ¬p3 ∧ p4) :=
by
  sorry

end NUMINAMATH_CALUDE_spatial_relations_l2414_241435


namespace NUMINAMATH_CALUDE_kens_height_l2414_241492

/-- Given the heights of Ivan and Jackie, and the relationship between the averages,
    prove Ken's height. -/
theorem kens_height (h_ivan : ℝ) (h_jackie : ℝ) (h_ken : ℝ) :
  h_ivan = 175 →
  h_jackie = 175 →
  (h_ivan + h_jackie + h_ken) / 3 = 1.04 * (h_ivan + h_jackie) / 2 →
  h_ken = 196 := by
  sorry

end NUMINAMATH_CALUDE_kens_height_l2414_241492


namespace NUMINAMATH_CALUDE_perimeter_of_MNO_l2414_241446

/-- A solid right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Midpoint of an edge -/
structure Midpoint where
  edge_start : ℝ × ℝ × ℝ
  edge_end : ℝ × ℝ × ℝ

/-- Theorem: Perimeter of triangle MNO in a right prism -/
theorem perimeter_of_MNO (prism : RightPrism) 
  (M : Midpoint) (N : Midpoint) (O : Midpoint) : 
  prism.height = 20 → 
  prism.base_side_length = 10 → 
  -- Assumptions about M, N, O being midpoints of specific edges would be added here
  ∃ (perimeter : ℝ), perimeter = 5 * (2 * Real.sqrt 5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_MNO_l2414_241446


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l2414_241404

theorem average_of_a_and_b (a b c : ℝ) : 
  ((b + c) / 2 = 180) → 
  (a - c = 200) → 
  ((a + b) / 2 = 280) :=
by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l2414_241404


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2414_241438

/-- The quadratic equation (k+2)x^2 + 4x + 1 = 0 has two distinct real roots if and only if k < 2 and k ≠ -2 -/
theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k + 2) * x^2 + 4 * x + 1 = 0 ∧ (k + 2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2414_241438


namespace NUMINAMATH_CALUDE_polygon_sides_l2414_241448

/-- A convex polygon with n sides where the sum of all angles except one is 2970 degrees -/
structure ConvexPolygon where
  n : ℕ
  sum_except_one : ℝ
  convex : sum_except_one = 2970

theorem polygon_sides (p : ConvexPolygon) : p.n = 19 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2414_241448


namespace NUMINAMATH_CALUDE_cookie_price_calculation_l2414_241415

/-- Represents a neighborhood with homes and boxes sold per home -/
structure Neighborhood where
  homes : ℕ
  boxesPerHome : ℕ

/-- Calculates the total boxes sold in a neighborhood -/
def totalBoxesSold (n : Neighborhood) : ℕ :=
  n.homes * n.boxesPerHome

/-- The price per box of cookies -/
def pricePerBox : ℚ := 2

theorem cookie_price_calculation 
  (neighborhoodA neighborhoodB : Neighborhood)
  (hA : neighborhoodA = ⟨10, 2⟩)
  (hB : neighborhoodB = ⟨5, 5⟩)
  (hRevenue : 50 = pricePerBox * max (totalBoxesSold neighborhoodA) (totalBoxesSold neighborhoodB)) :
  pricePerBox = 2 := by
sorry

end NUMINAMATH_CALUDE_cookie_price_calculation_l2414_241415


namespace NUMINAMATH_CALUDE_contest_result_l2414_241456

/-- The number of times Frannie jumped -/
def frannies_jumps : ℕ := 53

/-- The difference between Meg's and Frannie's jumps -/
def jump_difference : ℕ := 18

/-- Meg's number of jumps -/
def megs_jumps : ℕ := frannies_jumps + jump_difference

theorem contest_result : megs_jumps = 71 := by
  sorry

end NUMINAMATH_CALUDE_contest_result_l2414_241456


namespace NUMINAMATH_CALUDE_exists_grade_to_move_l2414_241454

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem exists_grade_to_move :
  ∃ g ∈ group1,
    average (group1.filter (· ≠ g)) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end NUMINAMATH_CALUDE_exists_grade_to_move_l2414_241454


namespace NUMINAMATH_CALUDE_logistics_service_assignments_logistics_service_assignments_proof_l2414_241427

theorem logistics_service_assignments : ℕ :=
  let total_students : ℕ := 5
  let total_athletes : ℕ := 3
  let athlete_A_in_own_team : Bool := true

  50

theorem logistics_service_assignments_proof :
  logistics_service_assignments = 50 := by
  sorry

end NUMINAMATH_CALUDE_logistics_service_assignments_logistics_service_assignments_proof_l2414_241427


namespace NUMINAMATH_CALUDE_launderette_machines_l2414_241443

/-- Represents the number of quarters in each machine -/
def quarters_per_machine : ℕ := 80

/-- Represents the number of dimes in each machine -/
def dimes_per_machine : ℕ := 100

/-- Represents the total amount of money after emptying all machines (in cents) -/
def total_amount : ℕ := 9000  -- $90 in cents

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Theorem stating that the number of machines in the launderette is 3 -/
theorem launderette_machines : 
  ∃ (n : ℕ), n * (quarters_per_machine * quarter_value + dimes_per_machine * dime_value) = total_amount ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_launderette_machines_l2414_241443


namespace NUMINAMATH_CALUDE_min_n_for_inequality_l2414_241441

theorem min_n_for_inequality : 
  ∃ (n : ℕ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 ≤ n * (x^4 + y^4 + z^4)) ∧ 
  (∀ (m : ℕ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 ≤ m * (x^4 + y^4 + z^4)) → n ≤ m) ∧
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_n_for_inequality_l2414_241441


namespace NUMINAMATH_CALUDE_books_read_difference_l2414_241486

/-- Given a collection of books and the percentages read by different people,
    calculate the difference between Peter's read books and the combined total of others. -/
theorem books_read_difference (total_books : ℕ) 
  (peter_percent : ℚ) (brother_percent : ℚ) (sarah_percent : ℚ) (alex_percent : ℚ) :
  total_books = 80 →
  peter_percent = 70 / 100 →
  brother_percent = 35 / 100 →
  sarah_percent = 40 / 100 →
  alex_percent = 22 / 100 →
  (peter_percent * total_books : ℚ).floor - 
  ((brother_percent * total_books : ℚ).floor + 
   (sarah_percent * total_books : ℚ).floor + 
   (alex_percent * total_books : ℚ).ceil) = -22 := by
  sorry

#check books_read_difference

end NUMINAMATH_CALUDE_books_read_difference_l2414_241486


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2414_241412

theorem pure_imaginary_fraction (α : ℝ) : 
  (∃ (y : ℝ), (α + 3 * Complex.I) / (1 + 2 * Complex.I) = y * Complex.I) → α = -6 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2414_241412


namespace NUMINAMATH_CALUDE_two_machines_half_hour_copies_l2414_241410

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Calculates the number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.copies_per_minute * minutes

/-- Theorem: Two copy machines working together for half an hour will produce 3300 copies -/
theorem two_machines_half_hour_copies :
  let machine1 : CopyMachine := ⟨35⟩
  let machine2 : CopyMachine := ⟨75⟩
  let half_hour : ℕ := 30
  (copies_made machine1 half_hour) + (copies_made machine2 half_hour) = 3300 :=
by
  sorry

end NUMINAMATH_CALUDE_two_machines_half_hour_copies_l2414_241410


namespace NUMINAMATH_CALUDE_geometric_series_equality_l2414_241497

/-- Given real numbers p, q, and r, if the infinite geometric series
    (p/q) + (p/q^2) + (p/q^3) + ... equals 9, then the infinite geometric series
    (p/(p+r)) + (p/(p+r)^2) + (p/(p+r)^3) + ... equals 9(q-1) / (9q + r - 10) -/
theorem geometric_series_equality (p q r : ℝ) 
  (h : ∑' n, p / q^n = 9) :
  ∑' n, p / (p + r)^n = 9 * (q - 1) / (9 * q + r - 10) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l2414_241497


namespace NUMINAMATH_CALUDE_birthday_cars_count_l2414_241402

-- Define the initial number of cars
def initial_cars : ℕ := 14

-- Define the number of cars bought
def bought_cars : ℕ := 28

-- Define the number of cars given away
def given_away_cars : ℕ := 8 + 3

-- Define the final number of cars
def final_cars : ℕ := 43

-- Theorem to prove
theorem birthday_cars_count :
  ∃ (birthday_cars : ℕ), 
    initial_cars + bought_cars + birthday_cars - given_away_cars = final_cars ∧
    birthday_cars = 12 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cars_count_l2414_241402


namespace NUMINAMATH_CALUDE_equipment_theorem_l2414_241495

/-- Represents the sales data for equipment A and B -/
structure SalesData where
  a : ℕ  -- quantity of A
  b : ℕ  -- quantity of B
  total : ℕ  -- total amount in yuan

/-- Represents the problem setup -/
structure EquipmentProblem where
  sale1 : SalesData
  sale2 : SalesData
  totalPieces : ℕ
  maxRatio : ℕ  -- max ratio of A to B
  maxCost : ℕ

/-- The main theorem to prove -/
theorem equipment_theorem (p : EquipmentProblem) 
  (h1 : p.sale1 = ⟨20, 10, 1100⟩)
  (h2 : p.sale2 = ⟨25, 20, 1750⟩)
  (h3 : p.totalPieces = 50)
  (h4 : p.maxRatio = 2)
  (h5 : p.maxCost = 2000) :
  ∃ (priceA priceB : ℕ),
    priceA = 30 ∧ 
    priceB = 50 ∧ 
    (∃ (validPlans : Finset (ℕ × ℕ)),
      validPlans.card = 9 ∧
      ∀ (plan : ℕ × ℕ), plan ∈ validPlans ↔ 
        (plan.1 + plan.2 = p.totalPieces ∧
         plan.1 ≤ p.maxRatio * plan.2 ∧
         plan.1 * priceA + plan.2 * priceB ≤ p.maxCost)) :=
by sorry

end NUMINAMATH_CALUDE_equipment_theorem_l2414_241495


namespace NUMINAMATH_CALUDE_prob_at_least_one_women_pair_l2414_241442

/-- The number of young men in the group -/
def num_men : ℕ := 6

/-- The number of young women in the group -/
def num_women : ℕ := 6

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to pair up the group -/
def total_pairings : ℕ := (total_people.factorial) / (2^num_pairs * num_pairs.factorial)

/-- The number of ways to pair up without any all-women pairs -/
def pairings_without_women_pairs : ℕ := num_women.factorial

/-- The probability of at least one pair consisting of two women -/
theorem prob_at_least_one_women_pair :
  (total_pairings - pairings_without_women_pairs) / total_pairings = 9675 / 10395 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_women_pair_l2414_241442


namespace NUMINAMATH_CALUDE_sign_distribution_of_products_l2414_241429

theorem sign_distribution_of_products (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ((-a*b > 0 ∧ a*c < 0 ∧ b*d < 0 ∧ c*d < 0) ∨
   (-a*b < 0 ∧ a*c > 0 ∧ b*d > 0 ∧ c*d > 0) ∨
   (-a*b < 0 ∧ a*c < 0 ∧ b*d > 0 ∧ c*d > 0) ∨
   (-a*b < 0 ∧ a*c > 0 ∧ b*d < 0 ∧ c*d > 0) ∨
   (-a*b < 0 ∧ a*c > 0 ∧ b*d > 0 ∧ c*d < 0)) := by
  sorry

end NUMINAMATH_CALUDE_sign_distribution_of_products_l2414_241429


namespace NUMINAMATH_CALUDE_simplify_expression_l2414_241477

theorem simplify_expression (y : ℝ) : 3*y + 9*y^2 + 10 - (5 - 3*y - 9*y^2) = 18*y^2 + 6*y + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2414_241477


namespace NUMINAMATH_CALUDE_regular_pyramid_volume_l2414_241437

-- Define the properties of the pyramid
structure RegularPyramid where
  l : ℝ  -- lateral edge length
  interior_angle_sum : ℝ  -- sum of interior angles of the base polygon
  lateral_angle : ℝ  -- angle between lateral edge and height

-- Define the theorem
theorem regular_pyramid_volume 
  (p : RegularPyramid) 
  (h1 : p.interior_angle_sum = 720)
  (h2 : p.lateral_angle = 30) : 
  ∃ (v : ℝ), v = (3 * p.l ^ 3) / 16 := by
  sorry

end NUMINAMATH_CALUDE_regular_pyramid_volume_l2414_241437


namespace NUMINAMATH_CALUDE_bricklayer_team_size_l2414_241488

theorem bricklayer_team_size :
  ∀ (x : ℕ),
  (x > 4) →
  (432 / x + 9) * (x - 4) = 432 →
  x = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_bricklayer_team_size_l2414_241488


namespace NUMINAMATH_CALUDE_eight_additional_people_needed_l2414_241462

/-- The number of additional people needed to mow a lawn and trim its edges -/
def additional_people_needed (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) : ℕ :=
  let total_person_hours := initial_people * initial_time
  let people_mowing := total_person_hours / new_time
  let people_trimming := people_mowing / 3
  let total_people_needed := people_mowing + people_trimming
  total_people_needed - initial_people

/-- Theorem stating that 8 additional people are needed under the given conditions -/
theorem eight_additional_people_needed :
  additional_people_needed 8 3 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_additional_people_needed_l2414_241462


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_2_area_implies_a_eq_1_l2414_241406

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| + |x - a|

-- Theorem for the first part of the problem
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x > 6} = {x : ℝ | x > 3 ∨ x < -3} := by sorry

-- Theorem for the second part of the problem
theorem area_implies_a_eq_1 (a : ℝ) (h : a > 0) :
  (∫ x in {x | f a x ≤ 5}, (5 - f a x)) = 8 → a = 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_2_area_implies_a_eq_1_l2414_241406


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2414_241468

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -30 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2414_241468


namespace NUMINAMATH_CALUDE_equation_equivalence_l2414_241466

theorem equation_equivalence (a b c : ℝ) :
  (a * (b - c)) / (b + c) + (b * (c - a)) / (c + a) + (c * (a - b)) / (a + b) = 0 ↔
  (a^2 * (b - c)) / (b + c) + (b^2 * (c - a)) / (c + a) + (c^2 * (a - b)) / (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2414_241466


namespace NUMINAMATH_CALUDE_inequality_relationship_l2414_241416

theorem inequality_relationship :
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ x ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relationship_l2414_241416


namespace NUMINAMATH_CALUDE_find_a_and_b_l2414_241452

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 + 7 * x - 15 < 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ),
    (A ∩ B a b = ∅) ∧
    (A ∪ B a b = {x | -5 < x ∧ x ≤ 2}) ∧
    (a = -7/2) ∧
    (b = 3) := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l2414_241452


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2414_241413

theorem simplify_polynomial (y : ℝ) : 
  2*y*(4*y^3 - 3*y + 5) - 4*(y^3 - 3*y^2 + 4*y - 6) = 
  8*y^4 - 4*y^3 + 6*y^2 - 6*y + 24 := by
sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2414_241413


namespace NUMINAMATH_CALUDE_rhombus_diagonal_and_side_l2414_241449

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a rhombus -/
structure Rhombus where
  A : Point
  C : Point
  AB : Line

/-- Theorem about the diagonals and sides of a specific rhombus -/
theorem rhombus_diagonal_and_side 
  (ABCD : Rhombus)
  (h1 : ABCD.A = ⟨0, 2⟩)
  (h2 : ABCD.C = ⟨4, 6⟩)
  (h3 : ABCD.AB = ⟨3, -1, 2⟩) :
  ∃ (BD AD : Line),
    BD = ⟨1, 1, -6⟩ ∧
    AD = ⟨1, -3, 14⟩ := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_and_side_l2414_241449


namespace NUMINAMATH_CALUDE_value_of_p_l2414_241439

theorem value_of_p (x y : ℝ) (h : |x - 1/2| + Real.sqrt (y^2 - 1) = 0) : 
  |x| + |y| = 3/2 := by
sorry

end NUMINAMATH_CALUDE_value_of_p_l2414_241439


namespace NUMINAMATH_CALUDE_b_age_is_39_l2414_241430

/-- Represents a person's age --/
structure Age where
  value : ℕ

/-- Represents the ages of three people A, B, and C --/
structure AgeGroup where
  a : Age
  b : Age
  c : Age

/-- Checks if the given ages satisfy the conditions of the problem --/
def satisfiesConditions (ages : AgeGroup) : Prop :=
  (ages.a.value + 10 = 2 * (ages.b.value - 10)) ∧
  (ages.a.value = ages.b.value + 9) ∧
  (ages.c.value = ages.a.value + 4)

/-- Theorem stating that if the conditions are satisfied, B's age is 39 --/
theorem b_age_is_39 (ages : AgeGroup) :
  satisfiesConditions ages → ages.b.value = 39 := by
  sorry

#check b_age_is_39

end NUMINAMATH_CALUDE_b_age_is_39_l2414_241430


namespace NUMINAMATH_CALUDE_calculate_swimming_speed_triathlete_swimming_speed_l2414_241433

/-- Calculates the swimming speed given the total distance, running speed, and average speed -/
theorem calculate_swimming_speed 
  (total_distance : ℝ) 
  (running_distance : ℝ) 
  (running_speed : ℝ) 
  (average_speed : ℝ) : ℝ :=
  let swimming_distance := total_distance - running_distance
  let total_time := total_distance / average_speed
  let running_time := running_distance / running_speed
  let swimming_time := total_time - running_time
  swimming_distance / swimming_time

/-- Proves that the swimming speed is 6 miles per hour given the problem conditions -/
theorem triathlete_swimming_speed : 
  calculate_swimming_speed 6 3 10 7.5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculate_swimming_speed_triathlete_swimming_speed_l2414_241433


namespace NUMINAMATH_CALUDE_exactly_two_successes_in_four_trials_l2414_241484

/-- The probability of success on a single trial -/
def p : ℚ := 2/3

/-- The number of trials -/
def n : ℕ := 4

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The binomial coefficient function -/
def binomial_coeff (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

/-- The probability of exactly k successes in n trials with probability p -/
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coeff n k * p^k * (1-p)^(n-k)

theorem exactly_two_successes_in_four_trials : 
  binomial_prob n k p = 8/27 := by sorry

end NUMINAMATH_CALUDE_exactly_two_successes_in_four_trials_l2414_241484


namespace NUMINAMATH_CALUDE_f_cos_x_l2414_241465

theorem f_cos_x (f : ℝ → ℝ) (h : ∀ x, f (Real.sin x) = 3 - Real.cos (2 * x)) :
  ∀ x, f (Real.cos x) = 3 + Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_f_cos_x_l2414_241465


namespace NUMINAMATH_CALUDE_three_solutions_condition_l2414_241445

theorem three_solutions_condition (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    a * x₁ = |Real.log x₁| ∧ a * x₂ = |Real.log x₂| ∧ a * x₃ = |Real.log x₃|) ∧
  (∀ x : ℝ, a * x ≥ 0) ↔ 
  (-1 / Real.exp 1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_condition_l2414_241445


namespace NUMINAMATH_CALUDE_total_tires_is_101_l2414_241400

/-- The number of tires on a car -/
def car_tires : ℕ := 4

/-- The number of tires on a bicycle -/
def bicycle_tires : ℕ := 2

/-- The number of tires on a pickup truck -/
def pickup_truck_tires : ℕ := 4

/-- The number of tires on a tricycle -/
def tricycle_tires : ℕ := 3

/-- The number of cars Juan saw -/
def cars_seen : ℕ := 15

/-- The number of bicycles Juan saw -/
def bicycles_seen : ℕ := 3

/-- The number of pickup trucks Juan saw -/
def pickup_trucks_seen : ℕ := 8

/-- The number of tricycles Juan saw -/
def tricycles_seen : ℕ := 1

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 
  car_tires * cars_seen + 
  bicycle_tires * bicycles_seen + 
  pickup_truck_tires * pickup_trucks_seen + 
  tricycle_tires * tricycles_seen

theorem total_tires_is_101 : total_tires = 101 := by
  sorry

end NUMINAMATH_CALUDE_total_tires_is_101_l2414_241400


namespace NUMINAMATH_CALUDE_equation_equivalence_l2414_241490

theorem equation_equivalence (x : ℝ) : 
  (x + 1) / 0.3 - (2 * x - 1) / 0.7 = 1 ↔ (10 * x + 10) / 3 - (20 * x - 10) / 7 = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2414_241490


namespace NUMINAMATH_CALUDE_fraction_calculation_l2414_241480

theorem fraction_calculation : (8 / 15 - 7 / 9) + 3 / 4 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2414_241480


namespace NUMINAMATH_CALUDE_group_purchase_equation_l2414_241401

theorem group_purchase_equation (x : ℕ) : 
  (∀ (required : ℕ), 8 * x = required + 3 ∧ 7 * x = required - 4) → 
  8 * x - 3 = 7 * x + 4 := by
sorry

end NUMINAMATH_CALUDE_group_purchase_equation_l2414_241401


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l2414_241479

theorem least_number_with_remainders : ∃! n : ℕ, 
  n > 0 ∧
  n % 34 = 4 ∧
  n % 48 = 6 ∧
  n % 5 = 2 ∧
  ∀ m : ℕ, m > 0 ∧ m % 34 = 4 ∧ m % 48 = 6 ∧ m % 5 = 2 → n ≤ m :=
by
  use 4082
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l2414_241479


namespace NUMINAMATH_CALUDE_game_lives_distribution_l2414_241444

/-- Given a game with initial players, players who quit, and total lives among remaining players,
    calculates the number of lives each remaining player has. -/
def lives_per_player (initial_players quitters total_lives : ℕ) : ℕ :=
  total_lives / (initial_players - quitters)

/-- Theorem stating that in a game with 13 initial players, 8 quitters, and 30 total lives,
    each remaining player has 6 lives. -/
theorem game_lives_distribution :
  lives_per_player 13 8 30 = 6 := by
  sorry


end NUMINAMATH_CALUDE_game_lives_distribution_l2414_241444


namespace NUMINAMATH_CALUDE_function_symmetry_translation_l2414_241496

def symmetric_wrt_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

/-- If the graph of f(x+1) is symmetric to e^x with respect to the y-axis,
    then f(x) = e^(-(x+1)) -/
theorem function_symmetry_translation (f : ℝ → ℝ) :
  symmetric_wrt_y_axis (λ x => f (x + 1)) Real.exp →
  f = λ x => Real.exp (-(x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_symmetry_translation_l2414_241496


namespace NUMINAMATH_CALUDE_wild_animal_population_estimation_l2414_241403

/-- Represents the data for a sample plot -/
structure PlotData where
  x : ℝ  -- plant coverage area
  y : ℝ  -- number of wild animals

/-- Represents the statistical data for the sample -/
structure SampleStats where
  n : ℕ              -- number of sample plots
  total_plots : ℕ    -- total number of plots in the area
  sum_x : ℝ          -- sum of x values
  sum_y : ℝ          -- sum of y values
  sum_x_squared : ℝ  -- sum of (x - x̄)²
  sum_y_squared : ℝ  -- sum of (y - ȳ)²
  sum_xy : ℝ         -- sum of (x - x̄)(y - ȳ)

/-- Theorem statement for the wild animal population estimation problem -/
theorem wild_animal_population_estimation
  (stats : SampleStats)
  (h1 : stats.n = 20)
  (h2 : stats.total_plots = 200)
  (h3 : stats.sum_x = 60)
  (h4 : stats.sum_y = 1200)
  (h5 : stats.sum_x_squared = 80)
  (h6 : stats.sum_y_squared = 9000)
  (h7 : stats.sum_xy = 800) :
  let estimated_population := (stats.sum_y / stats.n) * stats.total_plots
  let correlation_coefficient := stats.sum_xy / Real.sqrt (stats.sum_x_squared * stats.sum_y_squared)
  estimated_population = 12000 ∧ abs (correlation_coefficient - 0.94) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_wild_animal_population_estimation_l2414_241403


namespace NUMINAMATH_CALUDE_three_algorithms_among_four_l2414_241493

/-- A statement describing a process or task -/
structure Statement where
  description : String

/-- Predicate to determine if a statement is an algorithm -/
def is_algorithm (s : Statement) : Prop :=
  -- This definition would typically include formal criteria for what constitutes an algorithm
  sorry

/-- The set of given statements -/
def given_statements : Finset Statement := sorry

theorem three_algorithms_among_four :
  ∃ (alg_statements : Finset Statement),
    alg_statements ⊆ given_statements ∧
    (∀ s ∈ alg_statements, is_algorithm s) ∧
    Finset.card alg_statements = 3 ∧
    Finset.card given_statements = 4 := by
  sorry

end NUMINAMATH_CALUDE_three_algorithms_among_four_l2414_241493


namespace NUMINAMATH_CALUDE_hcf_problem_l2414_241467

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2562) (h2 : Nat.lcm a b = 183) :
  Nat.gcd a b = 14 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l2414_241467


namespace NUMINAMATH_CALUDE_zeros_product_gt_e_squared_l2414_241450

/-- Given a function g(x) = ln x - kx with two distinct zeros x₁ and x₂, 
    prove that their product is greater than e^2. -/
theorem zeros_product_gt_e_squared 
  (k : ℝ) 
  (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_zero₁ : Real.log x₁ = k * x₁) 
  (h_zero₂ : Real.log x₂ = k * x₂) : 
  x₁ * x₂ > Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_product_gt_e_squared_l2414_241450


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2414_241432

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y → y ≤ z →
  y = 5 →
  (x + y + z) / 3 = x + 10 →
  (x + y + z) / 3 = z - 15 →
  x + y + z = 30 := by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2414_241432


namespace NUMINAMATH_CALUDE_equation_solution_l2414_241457

theorem equation_solution :
  ∃! x : ℚ, x ≠ -3 ∧ (x^2 + 4*x + 5) / (x + 3) = x + 6 :=
by
  use -13/5
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2414_241457


namespace NUMINAMATH_CALUDE_figure_area_l2414_241459

/-- Given a figure consisting of five identical squares, where the area of one part (gray) 
    is 0.6 cm² larger than another part (white), prove that the total area of the figure is 6 cm². -/
theorem figure_area (a : Real) 
  (h1 : a > 0) -- Side length of each square is positive
  (h2 : a^2 = (3/2 * a^2) + 0.6) -- Relationship between gray and white areas
  : 5 * a^2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_figure_area_l2414_241459


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_101_l2414_241476

theorem gcd_of_powers_of_101 : 
  Nat.Prime 101 → Nat.gcd (101^5 + 1) (101^5 + 101^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_101_l2414_241476


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2414_241482

theorem inequality_system_solution (m : ℝ) : 
  (0 ≤ m ∧ m < 1) →
  (∃! (x : ℕ), x > 0 ∧ x - m > 0 ∧ x - 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2414_241482


namespace NUMINAMATH_CALUDE_root_in_interval_l2414_241428

def f (x : ℝ) := x^2 - 1

theorem root_in_interval : ∃ x : ℝ, -2 < x ∧ x < 0 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2414_241428


namespace NUMINAMATH_CALUDE_divisible_by_perfect_cube_l2414_241440

theorem divisible_by_perfect_cube (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h_divides : (a^2 + 3*a*b + 3*b^2 - 1) ∣ (a + b^3)) :
  ∃ (n : ℕ), n > 1 ∧ (n^3 : ℕ) ∣ (a^2 + 3*a*b + 3*b^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_perfect_cube_l2414_241440


namespace NUMINAMATH_CALUDE_cable_on_hand_theorem_l2414_241494

/-- Given a total length of cable and a section length, calculates the number of sections. -/
def calculateSections (totalLength sectionLength : ℕ) : ℕ := totalLength / sectionLength

/-- Calculates the number of sections given away. -/
def sectionsGivenAway (totalSections : ℕ) : ℕ := totalSections / 4

/-- Calculates the number of sections remaining after giving some away. -/
def remainingSections (totalSections givenAway : ℕ) : ℕ := totalSections - givenAway

/-- Calculates the number of sections put in storage. -/
def sectionsInStorage (remainingSections : ℕ) : ℕ := remainingSections / 2

/-- Calculates the number of sections kept on hand. -/
def sectionsOnHand (remainingSections inStorage : ℕ) : ℕ := remainingSections - inStorage

/-- Calculates the total length of cable kept on hand. -/
def cableOnHand (sectionsOnHand sectionLength : ℕ) : ℕ := sectionsOnHand * sectionLength

theorem cable_on_hand_theorem (totalLength sectionLength : ℕ) 
    (h1 : totalLength = 1000)
    (h2 : sectionLength = 25) : 
  cableOnHand 
    (sectionsOnHand 
      (remainingSections 
        (calculateSections totalLength sectionLength) 
        (sectionsGivenAway (calculateSections totalLength sectionLength)))
      (sectionsInStorage 
        (remainingSections 
          (calculateSections totalLength sectionLength) 
          (sectionsGivenAway (calculateSections totalLength sectionLength)))))
    sectionLength = 375 := by
  sorry

end NUMINAMATH_CALUDE_cable_on_hand_theorem_l2414_241494
