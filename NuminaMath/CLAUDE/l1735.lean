import Mathlib

namespace NUMINAMATH_CALUDE_intersection_range_l1735_173574

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt ((x + 1)^2 + y^2) = 2

-- Define the line l
def l (k x y : ℝ) : Prop :=
  y = k * x + 1 - 2 * k

-- Theorem statement
theorem intersection_range (k : ℝ) :
  (∃ x y, C x y ∧ l k x y) → k ∈ Set.Icc (1/3 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l1735_173574


namespace NUMINAMATH_CALUDE_m_value_l1735_173503

/-- The function f(x) = 4x^2 + 5x + 6 -/
def f (x : ℝ) : ℝ := 4 * x^2 + 5 * x + 6

/-- The function g(x) = 2x^3 - mx + 8, parameterized by m -/
def g (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - m * x + 8

/-- Theorem stating that if f(5) - g(5) = 15, then m = 28.4 -/
theorem m_value (m : ℝ) : f 5 - g m 5 = 15 → m = 28.4 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l1735_173503


namespace NUMINAMATH_CALUDE_quiz_true_false_count_l1735_173506

theorem quiz_true_false_count :
  ∀ n : ℕ,
  (2^n - 2) * 16 = 224 →
  n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_true_false_count_l1735_173506


namespace NUMINAMATH_CALUDE_harkamal_payment_l1735_173501

/-- The amount Harkamal paid to the shopkeeper -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Proof that Harkamal paid 1145 to the shopkeeper -/
theorem harkamal_payment : total_amount_paid 8 70 9 65 = 1145 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l1735_173501


namespace NUMINAMATH_CALUDE_partition_exists_five_equal_parts_exist_l1735_173599

/-- Represents a geometric shape composed of squares and triangles -/
structure GeometricFigure where
  squares : ℕ
  triangles : ℕ

/-- Represents a partition of a geometric figure -/
structure Partition where
  parts : ℕ
  part_composition : GeometricFigure

/-- Predicate to check if a partition is valid for a given figure -/
def is_valid_partition (figure : GeometricFigure) (partition : Partition) : Prop :=
  figure.squares = partition.parts * partition.part_composition.squares ∧
  figure.triangles = partition.parts * partition.part_composition.triangles

/-- The specific figure from the problem -/
def problem_figure : GeometricFigure :=
  { squares := 10, triangles := 5 }

/-- The desired partition -/
def desired_partition : Partition :=
  { parts := 5, part_composition := { squares := 2, triangles := 1 } }

/-- Theorem stating that the desired partition is valid for the problem figure -/
theorem partition_exists : is_valid_partition problem_figure desired_partition := by
  sorry

/-- Main theorem proving the existence of the required partition -/
theorem five_equal_parts_exist : ∃ (p : Partition), 
  p.parts = 5 ∧ 
  p.part_composition.squares = 2 ∧ 
  p.part_composition.triangles = 1 ∧
  is_valid_partition problem_figure p := by
  sorry

end NUMINAMATH_CALUDE_partition_exists_five_equal_parts_exist_l1735_173599


namespace NUMINAMATH_CALUDE_no_primes_satisfying_conditions_l1735_173577

theorem no_primes_satisfying_conditions : ¬∃ (p q : ℕ), 
  Prime p ∧ Prime q ∧ p > 3 ∧ q > 3 ∧ 
  (q ∣ (p^2 - 1)) ∧ (p ∣ (q^2 - 1)) := by
sorry

end NUMINAMATH_CALUDE_no_primes_satisfying_conditions_l1735_173577


namespace NUMINAMATH_CALUDE_intersection_range_l1735_173570

theorem intersection_range (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ - 1 ∧ 
    y₂ = k * x₂ - 1 ∧ 
    x₁^2 - y₁^2 = 4 ∧ 
    x₂^2 - y₂^2 = 4 ∧ 
    x₁ > 0 ∧ 
    x₂ > 0) ↔ 
  (1 < k ∧ k < Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l1735_173570


namespace NUMINAMATH_CALUDE_land_increase_percentage_l1735_173563

theorem land_increase_percentage (A B C D E : ℝ) 
  (h1 : B = 1.5 * A)
  (h2 : C = 2 * A)
  (h3 : D = 2.5 * A)
  (h4 : E = 3 * A)
  (h5 : A > 0) :
  let initial_area := A + B + C + D + E
  let increase := 0.1 * A + (1 / 15) * B + 0.05 * C + 0.04 * D + (1 / 30) * E
  increase / initial_area = 0.05 := by
sorry

end NUMINAMATH_CALUDE_land_increase_percentage_l1735_173563


namespace NUMINAMATH_CALUDE_absolute_difference_mn_l1735_173593

theorem absolute_difference_mn (m n : ℝ) 
  (h1 : m * n = 6)
  (h2 : m + n = 7)
  (h3 : m^2 - n^2 = 13) : 
  |m - n| = 13/7 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_mn_l1735_173593


namespace NUMINAMATH_CALUDE_prop_p_and_q_implies_range_of_a_l1735_173568

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -2 ∨ a = 1

-- Theorem statement
theorem prop_p_and_q_implies_range_of_a :
  ∀ a : ℝ, p a ∧ q a → range_of_a a :=
by sorry

end NUMINAMATH_CALUDE_prop_p_and_q_implies_range_of_a_l1735_173568


namespace NUMINAMATH_CALUDE_mod_eq_two_l1735_173550

theorem mod_eq_two (n : ℤ) : 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_eq_two_l1735_173550


namespace NUMINAMATH_CALUDE_rebecca_groups_l1735_173509

def egg_count : Nat := 75
def banana_count : Nat := 99
def marble_count : Nat := 48
def apple_count : Nat := 6 * 12  -- 6 dozen
def orange_count : Nat := 6  -- half dozen

def egg_group_size : Nat := 4
def banana_group_size : Nat := 5
def marble_group_size : Nat := 6
def apple_group_size : Nat := 12
def orange_group_size : Nat := 2

def total_groups : Nat :=
  (egg_count + egg_group_size - 1) / egg_group_size +
  (banana_count + banana_group_size - 1) / banana_group_size +
  marble_count / marble_group_size +
  apple_count / apple_group_size +
  orange_count / orange_group_size

theorem rebecca_groups : total_groups = 54 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_groups_l1735_173509


namespace NUMINAMATH_CALUDE_plane_count_l1735_173514

theorem plane_count (total_wings : ℕ) (wings_per_plane : ℕ) (h1 : total_wings = 108) (h2 : wings_per_plane = 2) :
  total_wings / wings_per_plane = 54 := by
  sorry

end NUMINAMATH_CALUDE_plane_count_l1735_173514


namespace NUMINAMATH_CALUDE_negation_equivalence_l1735_173598

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x - 2) / x ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1735_173598


namespace NUMINAMATH_CALUDE_canyon_trail_length_l1735_173535

/-- Represents the hike on Canyon Trail -/
structure CanyonTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hike -/
def validHike (hike : CanyonTrail) : Prop :=
  hike.day1 + hike.day2 + hike.day3 = 36 ∧
  (hike.day2 + hike.day3 + hike.day4) / 3 = 14 ∧
  hike.day3 + hike.day4 + hike.day5 = 45 ∧
  hike.day1 + hike.day4 = 29

/-- The theorem stating the total length of the Canyon Trail -/
theorem canyon_trail_length (hike : CanyonTrail) (h : validHike hike) :
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 71 := by
  sorry

end NUMINAMATH_CALUDE_canyon_trail_length_l1735_173535


namespace NUMINAMATH_CALUDE_olivia_payment_l1735_173525

/-- Represents the number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- Represents the number of quarters Olivia pays for chips -/
def quarters_for_chips : ℕ := 4

/-- Represents the number of quarters Olivia pays for soda -/
def quarters_for_soda : ℕ := 12

/-- Calculates the total amount Olivia pays in dollars -/
def total_paid : ℚ :=
  (quarters_for_chips + quarters_for_soda) / quarters_per_dollar

theorem olivia_payment :
  total_paid = 4 := by sorry

end NUMINAMATH_CALUDE_olivia_payment_l1735_173525


namespace NUMINAMATH_CALUDE_symmetric_circle_l1735_173566

/-- Given a circle C and a line l, find the equation of the circle symmetric to C with respect to l -/
theorem symmetric_circle (x y : ℝ) : 
  (∀ x y, x^2 + y^2 = 4 → x - y - 3 = 0 → 
    ∃ a b r, (x - a)^2 + (y - b)^2 = r^2 ∧ 
    (x - a)^2 + (y - b)^2 = x^2 + y^2 - 6*x + 6*y + 14) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_l1735_173566


namespace NUMINAMATH_CALUDE_abscissa_range_theorem_l1735_173510

/-- The range of the abscissa of the center of circle M -/
def abscissa_range : Set ℝ := {a | a < 0 ∨ a > 12/5}

/-- The line on which the center of circle M lies -/
def center_line (x y : ℝ) : Prop := 2*x - y - 4 = 0

/-- The equation of circle M with center (a, 2a-4) and radius 1 -/
def circle_M (x y a : ℝ) : Prop := (x - a)^2 + (y - (2*a - 4))^2 = 1

/-- The condition that no point on circle M satisfies NO = 1/2 NA -/
def no_point_condition (x y : ℝ) : Prop := ¬(x^2 + y^2 = 1/4 * (x^2 + (y - 3)^2))

/-- The main theorem statement -/
theorem abscissa_range_theorem (a : ℝ) :
  (∀ x y : ℝ, center_line x y → circle_M x y a → no_point_condition x y) ↔ a ∈ abscissa_range :=
sorry

end NUMINAMATH_CALUDE_abscissa_range_theorem_l1735_173510


namespace NUMINAMATH_CALUDE_train_speed_l1735_173579

/-- 
Given a train with length 150 meters that crosses an electric pole in 3 seconds,
prove that its speed is 50 meters per second.
-/
theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) : 
  length = 150 ∧ 
  time = 3 ∧ 
  speed = length / time → 
  speed = 50 := by sorry

end NUMINAMATH_CALUDE_train_speed_l1735_173579


namespace NUMINAMATH_CALUDE_two_white_balls_possible_l1735_173556

/-- Represents the four types of ball replacements --/
inductive Replacement
  | ThreeBlackToOneBlack
  | TwoBlackOneWhiteToOneBlackOneWhite
  | OneBlackTwoWhiteToTwoWhite
  | ThreeWhiteToOneBlackOneWhite

/-- Represents the state of the box --/
structure BoxState :=
  (black : ℕ)
  (white : ℕ)

/-- Applies a single replacement to the box state --/
def applyReplacement (state : BoxState) (r : Replacement) : BoxState :=
  match r with
  | Replacement.ThreeBlackToOneBlack => 
      { black := state.black - 2, white := state.white }
  | Replacement.TwoBlackOneWhiteToOneBlackOneWhite => 
      { black := state.black - 1, white := state.white }
  | Replacement.OneBlackTwoWhiteToTwoWhite => 
      { black := state.black - 1, white := state.white - 1 }
  | Replacement.ThreeWhiteToOneBlackOneWhite => 
      { black := state.black + 1, white := state.white - 2 }

/-- Represents a sequence of replacements --/
def ReplacementSequence := List Replacement

/-- Applies a sequence of replacements to the initial box state --/
def applyReplacements (initial : BoxState) (seq : ReplacementSequence) : BoxState :=
  seq.foldl applyReplacement initial

/-- The theorem to be proved --/
theorem two_white_balls_possible : 
  ∃ (seq : ReplacementSequence), 
    (applyReplacements { black := 100, white := 100 } seq).white = 2 := by
  sorry


end NUMINAMATH_CALUDE_two_white_balls_possible_l1735_173556


namespace NUMINAMATH_CALUDE_work_completion_time_l1735_173588

theorem work_completion_time (b_time : ℝ) (joint_work_time : ℝ) (work_completed : ℝ) (a_time : ℝ) : 
  b_time = 20 →
  joint_work_time = 2 →
  work_completed = 0.2333333333333334 →
  joint_work_time * ((1 / a_time) + (1 / b_time)) = work_completed →
  a_time = 15 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1735_173588


namespace NUMINAMATH_CALUDE_geometric_arithmetic_geometric_progression_l1735_173543

theorem geometric_arithmetic_geometric_progression
  (a b c : ℝ) :
  (∃ q : ℝ, b = a * q ∧ c = a * q^2) →  -- Initial geometric progression
  (2 * (b + 2) = a + c) →               -- Arithmetic progression after increasing b by 2
  ((b + 2)^2 = a * (c + 9)) →           -- Geometric progression after increasing c by 9
  ((a = 4/25 ∧ b = -16/25 ∧ c = 64/25) ∨ (a = 4 ∧ b = 8 ∧ c = 16)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_geometric_progression_l1735_173543


namespace NUMINAMATH_CALUDE_inequality_theorem_l1735_173549

theorem inequality_theorem (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + b = c + d) (prod_gt : a * b > c * d) : 
  (Real.sqrt a + Real.sqrt b > Real.sqrt c + Real.sqrt d) ∧ 
  (|a - b| < |c - d|) := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1735_173549


namespace NUMINAMATH_CALUDE_max_quotient_value_l1735_173505

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 → 800 ≤ y ∧ y ≤ 1600 → y / x ≤ 16 / 3) ∧
  (∃ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 ∧ y / x = 16 / 3) :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l1735_173505


namespace NUMINAMATH_CALUDE_find_divisor_l1735_173572

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 18 + remainder) :
  ∃ (divisor : ℕ), dividend = quotient * divisor + remainder ∧ divisor = 18 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1735_173572


namespace NUMINAMATH_CALUDE_quadratic_intersects_negative_x_axis_l1735_173552

/-- A quadratic function f(x) = (m-2)x^2 - 4mx + 2m - 6 intersects with the negative x-axis at least once
    if and only if m is in the range 1 ≤ m < 2 or 2 < m < 3. -/
theorem quadratic_intersects_negative_x_axis (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (m - 2) * x^2 - 4 * m * x + 2 * m - 6 = 0) ↔
  (1 ≤ m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersects_negative_x_axis_l1735_173552


namespace NUMINAMATH_CALUDE_greatest_common_factor_of_palindromes_l1735_173583

def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ a b : ℕ, n = 100*a + 10*b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3 * k

def set_of_valid_palindromes : Set ℕ :=
  {n : ℕ | is_three_digit_palindrome n ∧ is_multiple_of_three n}

theorem greatest_common_factor_of_palindromes :
  ∃ g : ℕ, g > 0 ∧ 
    (∀ n ∈ set_of_valid_palindromes, g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n ∈ set_of_valid_palindromes, d ∣ n) → d ≤ g) ∧
    g = 3 :=
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_of_palindromes_l1735_173583


namespace NUMINAMATH_CALUDE_triangle_max_area_l1735_173533

open Real

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  c = sqrt 2 →
  b = a * sin C + c * A →
  C = π / 4 →
  ∃ (S : ℝ), S ≤ (1 + sqrt 2) / 2 ∧
    ∀ (S' : ℝ), S' = 1 / 2 * a * b * sin C → S' ≤ S :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1735_173533


namespace NUMINAMATH_CALUDE_line_equal_intercepts_l1735_173532

theorem line_equal_intercepts (a : ℝ) : 
  (∃ x y : ℝ, a * x + y - 2 - a = 0 ∧ 
   x = y ∧ 
   (x = 0 ∨ y = 0)) ↔ 
  (a = -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_line_equal_intercepts_l1735_173532


namespace NUMINAMATH_CALUDE_doctor_lindsay_daily_income_is_2200_l1735_173597

/-- Calculates the total money Doctor Lindsay receives in a typical 8-hour day -/
def doctor_lindsay_daily_income : ℕ := by
  -- Define the number of adult patients per hour
  let adult_patients_per_hour : ℕ := 4
  -- Define the number of child patients per hour
  let child_patients_per_hour : ℕ := 3
  -- Define the cost for an adult's office visit
  let adult_visit_cost : ℕ := 50
  -- Define the cost for a child's office visit
  let child_visit_cost : ℕ := 25
  -- Define the number of working hours per day
  let working_hours_per_day : ℕ := 8
  
  -- Calculate the total income
  exact adult_patients_per_hour * adult_visit_cost * working_hours_per_day + 
        child_patients_per_hour * child_visit_cost * working_hours_per_day

/-- Theorem stating that Doctor Lindsay's daily income is $2200 -/
theorem doctor_lindsay_daily_income_is_2200 : 
  doctor_lindsay_daily_income = 2200 := by
  sorry

end NUMINAMATH_CALUDE_doctor_lindsay_daily_income_is_2200_l1735_173597


namespace NUMINAMATH_CALUDE_contractor_absent_days_l1735_173521

/-- Proves the number of absent days for a contractor under specific conditions -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (daily_pay : ℚ) 
  (daily_fine : ℚ) 
  (total_received : ℚ) : 
  total_days = 30 → 
  daily_pay = 25 → 
  daily_fine = 7.5 → 
  total_received = 620 → 
  ∃ (days_worked : ℕ) (days_absent : ℕ), 
    days_worked + days_absent = total_days ∧ 
    (daily_pay * days_worked : ℚ) - (daily_fine * days_absent : ℚ) = total_received ∧ 
    days_absent = 8 := by
  sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l1735_173521


namespace NUMINAMATH_CALUDE_external_tangent_circle_distance_l1735_173557

theorem external_tangent_circle_distance 
  (O P : ℝ × ℝ) 
  (r₁ r₂ : ℝ) 
  (Q : ℝ × ℝ) 
  (T : ℝ × ℝ) 
  (Z : ℝ × ℝ) :
  r₁ = 10 →
  r₂ = 3 →
  dist O P = r₁ + r₂ →
  dist O T = r₁ →
  dist P Z = r₂ →
  (T.1 - O.1) * (Z.1 - T.1) + (T.2 - O.2) * (Z.2 - T.2) = 0 →
  (Z.1 - P.1) * (Z.1 - T.1) + (Z.2 - P.2) * (Z.2 - T.2) = 0 →
  dist O Z = 2 * Real.sqrt 145 :=
by sorry

end NUMINAMATH_CALUDE_external_tangent_circle_distance_l1735_173557


namespace NUMINAMATH_CALUDE_min_sum_squares_l1735_173500

-- Define the points A, B, C, D, E as real numbers representing their positions on a line
def A : ℝ := 0
def B : ℝ := 1
def C : ℝ := 3
def D : ℝ := 6
def E : ℝ := 10

-- Define the function to be minimized
def f (x : ℝ) : ℝ := (x - A)^2 + (x - B)^2 + (x - C)^2 + (x - D)^2 + (x - E)^2

-- State the theorem
theorem min_sum_squares :
  ∃ (min : ℝ), min = 60 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1735_173500


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l1735_173558

/-- Given a circle C: x^2 + y^2 + mx - 4 = 0 and two points on C symmetric 
    with respect to the line x - y + 3 = 0, prove that m = 6 -/
theorem circle_symmetry_line (m : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 + A.2^2 + m*A.1 - 4 = 0) ∧ 
    (B.1^2 + B.2^2 + m*B.1 - 4 = 0) ∧ 
    (A.1 - A.2 + 3 = B.1 - B.2 + 3)) → 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l1735_173558


namespace NUMINAMATH_CALUDE_pauls_initial_pens_l1735_173565

/-- Represents the number of items Paul has --/
structure PaulsItems where
  initialBooks : ℕ
  initialPens : ℕ
  finalBooks : ℕ
  finalPens : ℕ
  soldPens : ℕ

/-- Theorem stating that Paul's initial number of pens is 42 --/
theorem pauls_initial_pens (items : PaulsItems)
    (h1 : items.initialBooks = 143)
    (h2 : items.finalBooks = 113)
    (h3 : items.finalPens = 19)
    (h4 : items.soldPens = 23) :
    items.initialPens = 42 := by
  sorry

#check pauls_initial_pens

end NUMINAMATH_CALUDE_pauls_initial_pens_l1735_173565


namespace NUMINAMATH_CALUDE_vector_b_magnitude_l1735_173591

def a : ℝ × ℝ × ℝ := (-1, 2, -3)
def b : ℝ × ℝ × ℝ := (-4, -1, 2)

theorem vector_b_magnitude : ‖b‖ = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_vector_b_magnitude_l1735_173591


namespace NUMINAMATH_CALUDE_min_value_at_three_l1735_173555

/-- The quadratic function we want to minimize -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 27

/-- The theorem stating that f(x) is minimized when x = 3 -/
theorem min_value_at_three :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 := by sorry

end NUMINAMATH_CALUDE_min_value_at_three_l1735_173555


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1735_173592

theorem sum_of_x_and_y (x y S : ℝ) 
  (h1 : x + y = S) 
  (h2 : y - 3 * x = 7) 
  (h3 : y - x = 7.5) : 
  S = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1735_173592


namespace NUMINAMATH_CALUDE_smallest_possible_total_l1735_173508

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- The ratios given in the problem --/
def ninth_to_tenth_ratio : Rat := 7 / 4
def ninth_to_eleventh_ratio : Rat := 5 / 3

/-- The condition that the ratios are correct --/
def ratios_correct (gc : GradeCount) : Prop :=
  (gc.ninth : Rat) / gc.tenth = ninth_to_tenth_ratio ∧
  (gc.ninth : Rat) / gc.eleventh = ninth_to_eleventh_ratio

/-- The total number of students --/
def total_students (gc : GradeCount) : ℕ :=
  gc.ninth + gc.tenth + gc.eleventh

/-- The main theorem to prove --/
theorem smallest_possible_total : 
  ∃ (gc : GradeCount), ratios_correct gc ∧ 
    (∀ (gc' : GradeCount), ratios_correct gc' → total_students gc ≤ total_students gc') ∧
    total_students gc = 76 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_total_l1735_173508


namespace NUMINAMATH_CALUDE_largest_number_divisible_by_sum_of_digits_eight_eight_eight_satisfies_property_l1735_173587

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property we're looking for
def hasSumOfDigitsDivisibility (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

-- State the theorem
theorem largest_number_divisible_by_sum_of_digits :
  ∀ n : ℕ, n < 900 → hasSumOfDigitsDivisibility n → n ≤ 888 :=
by
  sorry

-- Prove that 888 satisfies the property
theorem eight_eight_eight_satisfies_property :
  hasSumOfDigitsDivisibility 888 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_divisible_by_sum_of_digits_eight_eight_eight_satisfies_property_l1735_173587


namespace NUMINAMATH_CALUDE_m_range_proof_l1735_173586

-- Define the conditions
def condition_p (x : ℝ) : Prop := x^2 - 3*x - 4 ≤ 0
def condition_q (x m : ℝ) : Prop := x^2 - 6*x + 9 - m^2 ≤ 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ -4 ∨ m ≥ 4

-- Theorem statement
theorem m_range_proof :
  (∀ x, condition_p x → condition_q x m) ∧
  (∃ x, condition_q x m ∧ ¬condition_p x) →
  m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_proof_l1735_173586


namespace NUMINAMATH_CALUDE_number_equation_solution_l1735_173536

theorem number_equation_solution :
  ∃ x : ℝ, (3/4 * x + 3^2 = 1/5 * (x - 8 * x^(1/3))) ∧ x = -27 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1735_173536


namespace NUMINAMATH_CALUDE_equation_solutions_l1735_173516

theorem equation_solutions :
  (∀ x : ℝ, (x - 2) * (x - 3) = x - 2 ↔ x = 2 ∨ x = 4) ∧
  (∀ x : ℝ, 2 * x^2 - 5 * x + 1 = 0 ↔ x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1735_173516


namespace NUMINAMATH_CALUDE_ball_selection_theorem_l1735_173511

def number_of_ways (total_red : ℕ) (total_white : ℕ) (balls_taken : ℕ) (min_score : ℕ) : ℕ :=
  let red_score := 2
  let white_score := 1
  (Finset.range (min total_red balls_taken + 1)).sum (fun red_taken =>
    let white_taken := balls_taken - red_taken
    if white_taken ≤ total_white ∧ red_taken * red_score + white_taken * white_score ≥ min_score
    then Nat.choose total_red red_taken * Nat.choose total_white white_taken
    else 0)

theorem ball_selection_theorem :
  number_of_ways 4 6 5 7 = 186 := by
  sorry

end NUMINAMATH_CALUDE_ball_selection_theorem_l1735_173511


namespace NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l1735_173548

/-- A positive integer n is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The largest integer that always divides n^4 - n for all composite n is 6 -/
theorem largest_divisor_n4_minus_n (n : ℕ) (h : IsComposite n) :
  (∀ k : ℕ, k > 6 → ∃ m : ℕ, IsComposite m ∧ (m^4 - m) % k ≠ 0) ∧
  (∀ n : ℕ, IsComposite n → (n^4 - n) % 6 = 0) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l1735_173548


namespace NUMINAMATH_CALUDE_system_solution_proof_l1735_173515

theorem system_solution_proof (x y : ℚ) : 
  (3 * x - y = 4 ∧ 6 * x - 3 * y = 10) ↔ (x = 2/3 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_proof_l1735_173515


namespace NUMINAMATH_CALUDE_sqrt_2x_plus_4_real_l1735_173585

theorem sqrt_2x_plus_4_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x + 4) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_plus_4_real_l1735_173585


namespace NUMINAMATH_CALUDE_root_product_theorem_l1735_173539

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^4 - x^3 + 2*x - 3

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 - 3

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ : ℝ) : 
  f x₁ = 0 → f x₂ = 0 → f x₃ = 0 → f x₄ = 0 → 
  g x₁ * g x₂ * g x₃ * g x₄ = 33 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1735_173539


namespace NUMINAMATH_CALUDE_complex_power_sum_l1735_173560

theorem complex_power_sum (i : ℂ) : i^2 = -1 → i^50 + 3 * i^303 - 2 * i^101 = -1 - 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1735_173560


namespace NUMINAMATH_CALUDE_geometry_theorem_l1735_173531

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations
def contains (p : Plane) (l : Line) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def intersection (p1 p2 : Plane) (l : Line) : Prop := sorry

-- State the theorem
theorem geometry_theorem 
  (m n l : Line) 
  (α β γ : Plane) 
  (hm : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (hα : α ≠ β ∧ α ≠ γ ∧ β ≠ γ) :
  (contains α m ∧ contains β n ∧ perpendicular_planes α β ∧ 
   intersection α β l ∧ perpendicular_lines m l → perpendicular_lines m n) ∧
  (parallel_planes α γ ∧ parallel_planes β γ → parallel_planes α β) := by
  sorry

end NUMINAMATH_CALUDE_geometry_theorem_l1735_173531


namespace NUMINAMATH_CALUDE_quadratic_propositions_l1735_173522

/-- Proposition p: The equation x^2 - 4mx + 1 = 0 has real solutions -/
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 4*m*x + 1 = 0

/-- Proposition q: There exists some x₀ ∈ ℝ such that mx₀² - 2x₀ - 1 > 0 -/
def q (m : ℝ) : Prop := ∃ x₀ : ℝ, m*x₀^2 - 2*x₀ - 1 > 0

theorem quadratic_propositions (m : ℝ) :
  (p m ↔ (m ≥ 1/2 ∨ m ≤ -1/2)) ∧
  (q m ↔ m > -1) ∧
  ((p m ↔ ¬q m) → (-1 < m ∧ m < 1/2)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_propositions_l1735_173522


namespace NUMINAMATH_CALUDE_second_expression_proof_l1735_173584

theorem second_expression_proof (a : ℝ) (x : ℝ) :
  a = 34 →
  ((2 * a + 16) + x) / 2 = 89 →
  x = 94 := by
sorry

end NUMINAMATH_CALUDE_second_expression_proof_l1735_173584


namespace NUMINAMATH_CALUDE_solution_system_l1735_173567

theorem solution_system (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 110) : 
  x^2 + y^2 = 8044 / 169 := by
  sorry

end NUMINAMATH_CALUDE_solution_system_l1735_173567


namespace NUMINAMATH_CALUDE_range_of_m_l1735_173524

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 10}

theorem range_of_m (m : ℝ) :
  (A m ∪ B = B) ↔ m ≤ 11/3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1735_173524


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l1735_173554

theorem opposite_of_negative_six (m : ℤ) : (m + (-6) = 0) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l1735_173554


namespace NUMINAMATH_CALUDE_balloon_arrangements_l1735_173537

theorem balloon_arrangements : 
  let total_letters : ℕ := 7
  let repeated_letters : ℕ := 2
  let repetitions_per_letter : ℕ := 2
  (total_letters.factorial) / (repetitions_per_letter.factorial ^ repeated_letters) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l1735_173537


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_pair_l1735_173520

theorem sum_of_reciprocal_pair (a b : ℝ) : 
  a > 0 → b > 0 → a * b = 1 → (3 * a + 2 * b) * (3 * b + 2 * a) = 295 → a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_pair_l1735_173520


namespace NUMINAMATH_CALUDE_doctor_engineer_ratio_l1735_173527

theorem doctor_engineer_ratio (d l e : ℕ) (avg_age : ℚ) : 
  avg_age = 45 →
  (40 * d + 55 * l + 50 * e : ℚ) / (d + l + e : ℚ) = avg_age →
  d = 3 * e :=
by sorry

end NUMINAMATH_CALUDE_doctor_engineer_ratio_l1735_173527


namespace NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l1735_173569

theorem squared_difference_of_quadratic_roots : ∀ Φ φ : ℝ, 
  Φ ≠ φ →
  Φ^2 = 2*Φ + 1 →
  φ^2 = 2*φ + 1 →
  (Φ - φ)^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l1735_173569


namespace NUMINAMATH_CALUDE_new_person_weight_l1735_173529

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 5 →
  replaced_weight = 40 →
  avg_increase = 10 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 90 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1735_173529


namespace NUMINAMATH_CALUDE_prime_fraction_solutions_l1735_173564

theorem prime_fraction_solutions (x y : ℕ) :
  (x > 0 ∧ y > 0) →
  (∃ p : ℕ, Nat.Prime p ∧ x * y^2 = p * (x + y)) ↔ 
  ((x = 2 ∧ y = 2) ∨ (x = 6 ∧ y = 2)) := by
sorry

end NUMINAMATH_CALUDE_prime_fraction_solutions_l1735_173564


namespace NUMINAMATH_CALUDE_complex_modulus_range_l1735_173542

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z = a + Complex.I) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l1735_173542


namespace NUMINAMATH_CALUDE_inverse_sum_reciprocals_l1735_173553

theorem inverse_sum_reciprocals (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (y * z + x * z + x * y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocals_l1735_173553


namespace NUMINAMATH_CALUDE_complex_radical_equation_l1735_173573

theorem complex_radical_equation : 
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 6 + 1) - Real.sqrt (4 - 2 * Real.sqrt 3)
  M = Real.sqrt 3 / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_radical_equation_l1735_173573


namespace NUMINAMATH_CALUDE_class_fraction_problem_l1735_173512

theorem class_fraction_problem (G : ℕ) (B : ℕ) :
  B = (5 * G) / 3 →
  (2 * G) / 3 = (1 / 4) * (B + G) :=
by sorry

end NUMINAMATH_CALUDE_class_fraction_problem_l1735_173512


namespace NUMINAMATH_CALUDE_point_on_transformed_plane_l1735_173518

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (pt : Point3D) (p : Plane) : Prop :=
  p.a * pt.x + p.b * pt.y + p.c * pt.z + p.d = 0

/-- The main theorem to prove -/
theorem point_on_transformed_plane :
  let A : Point3D := { x := -2, y := -1, z := 1 }
  let a : Plane := { a := 1, b := -2, c := 6, d := -10 }
  let k : ℝ := 3/5
  pointOnPlane A (transformPlane a k) := by sorry

end NUMINAMATH_CALUDE_point_on_transformed_plane_l1735_173518


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l1735_173544

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧
  N = 23 ∧
  (∀ (k : ℕ), k > N → 
    (1743 % k = 2019 % k ∧ 2019 % k = 3008 % k) → false) ∧
  1743 % N = 2019 % N ∧ 2019 % N = 3008 % N :=
by sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l1735_173544


namespace NUMINAMATH_CALUDE_second_box_capacity_l1735_173596

/-- Represents the amount of clay a box can hold based on its dimensions -/
def clay_capacity (height width length : ℝ) : ℝ := sorry

theorem second_box_capacity :
  let first_box_height : ℝ := 2
  let first_box_width : ℝ := 3
  let first_box_length : ℝ := 5
  let first_box_capacity : ℝ := 40
  let second_box_height : ℝ := 2 * first_box_height
  let second_box_width : ℝ := 3 * first_box_width
  let second_box_length : ℝ := first_box_length
  clay_capacity first_box_height first_box_width first_box_length = first_box_capacity →
  clay_capacity second_box_height second_box_width second_box_length = 240 :=
by sorry

end NUMINAMATH_CALUDE_second_box_capacity_l1735_173596


namespace NUMINAMATH_CALUDE_sqrt_seven_minus_one_half_less_than_one_l1735_173595

theorem sqrt_seven_minus_one_half_less_than_one :
  (Real.sqrt 7 - 1) / 2 < 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_minus_one_half_less_than_one_l1735_173595


namespace NUMINAMATH_CALUDE_total_votes_calculation_l1735_173530

theorem total_votes_calculation (V : ℝ) 
  (h1 : 0.3 * V + (0.3 * V + 1760) = V) : V = 4400 := by
  sorry

end NUMINAMATH_CALUDE_total_votes_calculation_l1735_173530


namespace NUMINAMATH_CALUDE_triangle_weights_equal_l1735_173575

/-- Given a triangle ABC with side weights x, y, and z, if the sum of weights on any two sides
    equals the weight on the third side multiplied by a constant k, then all weights are equal. -/
theorem triangle_weights_equal (x y z k : ℝ) 
  (h1 : x + y = k * z) 
  (h2 : y + z = k * x) 
  (h3 : z + x = k * y) : 
  x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_triangle_weights_equal_l1735_173575


namespace NUMINAMATH_CALUDE_carol_rectangle_length_l1735_173545

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem carol_rectangle_length 
  (jordan : Rectangle) 
  (carol : Rectangle) 
  (h1 : jordan.length = 3) 
  (h2 : jordan.width = 40) 
  (h3 : carol.width = 24) 
  (h4 : area jordan = area carol) : 
  carol.length = 5 := by
sorry

end NUMINAMATH_CALUDE_carol_rectangle_length_l1735_173545


namespace NUMINAMATH_CALUDE_difference_of_squares_535_465_l1735_173507

theorem difference_of_squares_535_465 : 535^2 - 465^2 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_535_465_l1735_173507


namespace NUMINAMATH_CALUDE_faster_train_speed_l1735_173581

/-- The speed of the faster train given two trains crossing each other -/
theorem faster_train_speed 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : train_length = 150)
  (h2 : crossing_time = 18)
  (h3 : speed_ratio = 3) : 
  ∃ (v : ℝ), v = 12.5 ∧ v = (2 * train_length) / (crossing_time * (1 + 1 / speed_ratio)) :=
by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l1735_173581


namespace NUMINAMATH_CALUDE_max_m_value_l1735_173504

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem max_m_value : 
  (∃ (m : ℝ), m = 4 ∧ 
    (∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m → f (x + t) ≤ x)) ∧
  (∀ (m : ℝ), m > 4 → 
    (∀ (t : ℝ), ∃ (x : ℝ), x ∈ Set.Icc 1 m ∧ f (x + t) > x)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1735_173504


namespace NUMINAMATH_CALUDE_cistern_filling_time_l1735_173580

theorem cistern_filling_time (partial_fill_time : ℝ) (partial_fill_fraction : ℝ) 
  (h1 : partial_fill_time = 5)
  (h2 : partial_fill_fraction = 1 / 11) :
  partial_fill_time / partial_fill_fraction = 55 := by
  sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l1735_173580


namespace NUMINAMATH_CALUDE_partner_capital_l1735_173590

/-- Given the profit distribution and profit rate change, calculate A's capital -/
theorem partner_capital (total_profit : ℝ) (a_profit_share : ℝ) (a_income_increase : ℝ) 
  (initial_rate : ℝ) (final_rate : ℝ) :
  (a_profit_share = 2/3) →
  (a_income_increase = 300) →
  (initial_rate = 0.05) →
  (final_rate = 0.07) →
  (a_income_increase = a_profit_share * total_profit * (final_rate - initial_rate)) →
  (∃ (a_capital : ℝ), a_capital = 300000 ∧ a_profit_share * total_profit = initial_rate * a_capital) :=
by sorry

end NUMINAMATH_CALUDE_partner_capital_l1735_173590


namespace NUMINAMATH_CALUDE_bank_teller_coins_l1735_173538

theorem bank_teller_coins (num_5c num_10c : ℕ) (total_value : ℚ) : 
  num_5c = 16 →
  num_10c = 16 →
  total_value = (5 * num_5c + 10 * num_10c) / 100 →
  total_value = 21/5 →
  num_5c + num_10c = 32 := by
sorry

end NUMINAMATH_CALUDE_bank_teller_coins_l1735_173538


namespace NUMINAMATH_CALUDE_ana_salary_calculation_l1735_173519

/-- Calculates the final salary after a raise, pay cut, and bonus -/
def final_salary (initial_salary : ℝ) (raise_percent : ℝ) (cut_percent : ℝ) (bonus : ℝ) : ℝ :=
  initial_salary * (1 + raise_percent) * (1 - cut_percent) + bonus

theorem ana_salary_calculation :
  final_salary 2500 0.25 0.25 200 = 2543.75 := by
  sorry

end NUMINAMATH_CALUDE_ana_salary_calculation_l1735_173519


namespace NUMINAMATH_CALUDE_franks_change_is_four_l1735_173578

/-- The amount of change Frank receives from his purchase. -/
def franks_change (chocolate_bars : ℕ) (chips : ℕ) (chocolate_price : ℚ) (chips_price : ℚ) (money_given : ℚ) : ℚ :=
  money_given - (chocolate_bars * chocolate_price + chips * chips_price)

/-- Theorem stating that Frank's change is $4 given the problem conditions. -/
theorem franks_change_is_four :
  franks_change 5 2 2 3 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_franks_change_is_four_l1735_173578


namespace NUMINAMATH_CALUDE_bicycle_speed_problem_l1735_173502

theorem bicycle_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_diff : ℝ) :
  distance = 12 →
  speed_ratio = 1.2 →
  time_diff = 1/6 →
  ∃ (speed_B : ℝ),
    distance / speed_B - distance / (speed_ratio * speed_B) = time_diff ∧
    speed_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_speed_problem_l1735_173502


namespace NUMINAMATH_CALUDE_equation_solutions_l1735_173541

def solution_set : Set (ℤ × ℤ) :=
  {(3, 2), (2, 3), (1, -1), (-1, 1), (0, -1), (-1, 0)}

def satisfies_equation (p : ℤ × ℤ) : Prop :=
  (p.1)^3 + (p.2)^3 + 1 = (p.1)^2 * (p.2)^2

theorem equation_solutions :
  ∀ (x y : ℤ), satisfies_equation (x, y) ↔ (x, y) ∈ solution_set := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1735_173541


namespace NUMINAMATH_CALUDE_xyz_value_l1735_173517

theorem xyz_value (a b c x y z : ℂ)
  (eq1 : a = (b + c) / (x - 2))
  (eq2 : b = (c + a) / (y - 2))
  (eq3 : c = (a + b) / (z - 2))
  (sum_prod : x * y + y * z + z * x = 67)
  (sum : x + y + z = 2010) :
  x * y * z = -5892 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1735_173517


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1735_173562

def A : Set Nat := {1, 2, 3, 5}
def B : Set Nat := {2, 3, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1735_173562


namespace NUMINAMATH_CALUDE_prob_three_heads_in_eight_tosses_l1735_173513

/-- A fair coin is tossed 8 times. -/
def num_tosses : ℕ := 8

/-- The number of heads we're looking for. -/
def target_heads : ℕ := 3

/-- The probability of getting heads on a single toss of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The probability of getting exactly 'target_heads' heads in 'num_tosses' tosses of a fair coin. -/
def probability_exact_heads : ℚ :=
  (Nat.choose num_tosses target_heads : ℚ) * prob_heads^target_heads * (1 - prob_heads)^(num_tosses - target_heads)

/-- Theorem stating that the probability of getting exactly 3 heads in 8 tosses of a fair coin is 7/32. -/
theorem prob_three_heads_in_eight_tosses : probability_exact_heads = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_heads_in_eight_tosses_l1735_173513


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l1735_173571

/-- Given a quadratic equation px^2 + qx + r = 0 with roots u and v,
    prove that qu + r and qv + r are roots of px^2 - (2pr-q)x + (pr-q^2+qr) = 0 -/
theorem quadratic_root_transformation (p q r u v : ℝ) 
  (hu : p * u^2 + q * u + r = 0)
  (hv : p * v^2 + q * v + r = 0) :
  p * (q * u + r)^2 - (2 * p * r - q) * (q * u + r) + (p * r - q^2 + q * r) = 0 ∧
  p * (q * v + r)^2 - (2 * p * r - q) * (q * v + r) + (p * r - q^2 + q * r) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l1735_173571


namespace NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l1735_173534

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- The x-intercept of a line. -/
def x_intercept (l : Line) : ℚ := -l.y_intercept / l.slope

/-- The slope of a line perpendicular to a given line. -/
def perpendicular_slope (m : ℚ) : ℚ := -1 / m

/-- The slope of the line 4x - 3y = 12. -/
def given_line_slope : ℚ := 4 / 3

theorem x_intercept_of_perpendicular_line :
  let perpendicular_line := Line.mk (perpendicular_slope given_line_slope) 4
  x_intercept perpendicular_line = 16 / 3 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l1735_173534


namespace NUMINAMATH_CALUDE_ellipse_m_range_l1735_173547

/-- Given an ellipse represented by the equation x²/(m-1) + y²/(2-m) = 1 with foci on the y-axis,
    the range of values for m is (1, 3/2). -/
theorem ellipse_m_range (x y m : ℝ) :
  (∀ x y, x^2 / (m - 1) + y^2 / (2 - m) = 1) →  -- Ellipse equation
  (∃ c : ℝ, ∀ x, x^2 / (m - 1) + 0^2 / (2 - m) = 1 → x^2 ≤ c^2) →  -- Foci on y-axis
  1 < m ∧ m < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l1735_173547


namespace NUMINAMATH_CALUDE_basketball_handshakes_l1735_173523

/-- Calculates the total number of handshakes in a basketball game scenario -/
theorem basketball_handshakes (team_size : ℕ) (referee_count : ℕ) : team_size = 7 ∧ referee_count = 3 →
  team_size * team_size + 2 * team_size * referee_count = 91 := by
  sorry

#check basketball_handshakes

end NUMINAMATH_CALUDE_basketball_handshakes_l1735_173523


namespace NUMINAMATH_CALUDE_savings_calculation_l1735_173559

/-- Represents the financial situation of a person in a particular month --/
structure FinancialSituation where
  k : ℝ  -- Constant factor
  x : ℝ  -- Variable for income
  y : ℝ  -- Variable for expenditure
  I : ℝ  -- Total income
  E : ℝ  -- Regular expenditure
  U : ℝ  -- Unplanned expense
  S : ℝ  -- Savings

/-- The conditions of the financial situation --/
def financial_conditions (fs : FinancialSituation) : Prop :=
  fs.I = fs.k * fs.x ∧
  fs.E = fs.k * fs.y ∧
  fs.x / fs.y = 5 / 4 ∧
  fs.U = 0.2 * fs.E ∧
  fs.I = 16000 ∧
  fs.S = fs.I - (fs.E + fs.U)

/-- The theorem stating that under the given conditions, the savings is 640 --/
theorem savings_calculation (fs : FinancialSituation) :
  financial_conditions fs → fs.S = 640 := by
  sorry


end NUMINAMATH_CALUDE_savings_calculation_l1735_173559


namespace NUMINAMATH_CALUDE_simplify_trig_ratio_l1735_173528

theorem simplify_trig_ratio : 
  (Real.sin (30 * π / 180) + Real.sin (40 * π / 180)) / 
  (Real.cos (30 * π / 180) + Real.cos (40 * π / 180)) = 
  Real.tan (35 * π / 180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_ratio_l1735_173528


namespace NUMINAMATH_CALUDE_olaf_boat_crew_size_l1735_173526

/-- Proves the number of men on Olaf's boat given the travel conditions -/
theorem olaf_boat_crew_size :
  ∀ (total_distance : ℝ) 
    (boat_speed : ℝ) 
    (water_per_man_per_day : ℝ) 
    (total_water : ℝ),
  total_distance = 4000 →
  boat_speed = 200 →
  water_per_man_per_day = 1/2 →
  total_water = 250 →
  (total_water / ((total_distance / boat_speed) * water_per_man_per_day) : ℝ) = 25 :=
by
  sorry


end NUMINAMATH_CALUDE_olaf_boat_crew_size_l1735_173526


namespace NUMINAMATH_CALUDE_trevors_age_when_brother_is_three_times_older_l1735_173551

theorem trevors_age_when_brother_is_three_times_older (trevor_current_age : ℕ) (brother_current_age : ℕ) :
  trevor_current_age = 11 →
  brother_current_age = 20 →
  ∃ (future_age : ℕ), future_age = 24 ∧ brother_current_age + future_age - trevor_current_age = 3 * trevor_current_age :=
by
  sorry

end NUMINAMATH_CALUDE_trevors_age_when_brother_is_three_times_older_l1735_173551


namespace NUMINAMATH_CALUDE_jake_weight_proof_l1735_173594

/-- Jake's weight in pounds -/
def jake_weight : ℝ := 230

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 111

/-- Jake's brother's weight in pounds -/
def brother_weight : ℝ := 139

theorem jake_weight_proof :
  -- Condition 1: If Jake loses 8 pounds, he will weigh twice as much as his sister
  jake_weight - 8 = 2 * sister_weight ∧
  -- Condition 2: Jake's brother is currently 6 pounds heavier than twice Jake's weight
  brother_weight = 2 * jake_weight + 6 ∧
  -- Condition 3: Together, all three of them now weigh 480 pounds
  jake_weight + sister_weight + brother_weight = 480 ∧
  -- Condition 4: The brother's weight is 125% of the sister's weight
  brother_weight = 1.25 * sister_weight →
  -- Conclusion: Jake's weight is 230 pounds
  jake_weight = 230 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_proof_l1735_173594


namespace NUMINAMATH_CALUDE_inverse_square_theorem_l1735_173561

def inverse_square_relation (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ y : ℝ, y ≠ 0 → f y = k / (y * y)

theorem inverse_square_theorem (f : ℝ → ℝ) 
  (h1 : inverse_square_relation f)
  (h2 : ∃ y : ℝ, f y = 1)
  (h3 : f 6 = 0.25) :
  f 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_square_theorem_l1735_173561


namespace NUMINAMATH_CALUDE_garage_sale_pants_price_l1735_173546

/-- Proves that the price of each pair of pants is $3 in Kekai's garage sale scenario --/
theorem garage_sale_pants_price (shirt_price : ℚ) (num_shirts num_pants : ℕ) (remaining_money : ℚ) :
  shirt_price = 1 →
  num_shirts = 5 →
  num_pants = 5 →
  remaining_money = 10 →
  ∃ (pants_price : ℚ),
    pants_price = 3 ∧
    remaining_money = (shirt_price * num_shirts + pants_price * num_pants) / 2 := by
  sorry


end NUMINAMATH_CALUDE_garage_sale_pants_price_l1735_173546


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l1735_173582

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem quadratic_inequality_negation : 
  (¬ ∃ x₀ : ℝ, x₀^2 + 1 > 3*x₀) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l1735_173582


namespace NUMINAMATH_CALUDE_books_together_l1735_173589

/-- The number of books Tim and Mike have together -/
def total_books (tim_books mike_books : ℕ) : ℕ := tim_books + mike_books

/-- Theorem stating that Tim and Mike have 42 books together -/
theorem books_together : total_books 22 20 = 42 := by
  sorry

end NUMINAMATH_CALUDE_books_together_l1735_173589


namespace NUMINAMATH_CALUDE_distance_to_nearest_city_l1735_173540

-- Define the distance to the nearest city
variable (d : ℝ)

-- Define the conditions based on the false statements
def alice_condition : Prop := d < 8
def bob_condition : Prop := d > 7
def charlie_condition : Prop := d > 5
def david_condition : Prop := d ≠ 3

-- Theorem statement
theorem distance_to_nearest_city :
  alice_condition d ∧ bob_condition d ∧ charlie_condition d ∧ david_condition d ↔ d ∈ Set.Ioo 7 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_nearest_city_l1735_173540


namespace NUMINAMATH_CALUDE_apples_per_pie_l1735_173576

theorem apples_per_pie 
  (initial_apples : ℕ) 
  (handed_out : ℕ) 
  (num_pies : ℕ) 
  (h1 : initial_apples = 62) 
  (h2 : handed_out = 8) 
  (h3 : num_pies = 6) 
  (h4 : num_pies ≠ 0) : 
  (initial_apples - handed_out) / num_pies = 9 := by
sorry

end NUMINAMATH_CALUDE_apples_per_pie_l1735_173576
