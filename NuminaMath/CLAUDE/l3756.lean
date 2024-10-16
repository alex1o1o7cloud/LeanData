import Mathlib

namespace NUMINAMATH_CALUDE_segment_length_after_reflection_l3756_375632

-- Define the points
def Z : ℝ × ℝ := (-5, 3)
def Z' : ℝ × ℝ := (5, 3)

-- Define the reflection over y-axis
def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Theorem statement
theorem segment_length_after_reflection :
  Z' = reflect_over_y_axis Z ∧ 
  Real.sqrt ((Z'.1 - Z.1)^2 + (Z'.2 - Z.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_after_reflection_l3756_375632


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l3756_375634

def B : Set ℕ := {n : ℕ | ∃ y : ℕ+, n = 4 * y + 2}

theorem gcd_of_B_is_two : ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l3756_375634


namespace NUMINAMATH_CALUDE_no_integer_roots_for_odd_coefficients_l3756_375656

theorem no_integer_roots_for_odd_coefficients (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) : 
  ¬ ∃ x : ℤ, a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_for_odd_coefficients_l3756_375656


namespace NUMINAMATH_CALUDE_condition_one_condition_two_condition_three_l3756_375601

-- Define set A
def A : Set ℝ := {x | x^2 + 2*x - 3 = 0}

-- Define set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | x = -1/(2*a)}

-- Theorem for condition ①
theorem condition_one : 
  ∀ a : ℝ, (A ∩ B a = B a) ↔ (a = 0 ∨ a = -1/2 ∨ a = 1/6) := by sorry

-- Theorem for condition ②
theorem condition_two :
  ∀ a : ℝ, ((Set.univ \ B a) ∩ A = {1}) ↔ (a = 1/6) := by sorry

-- Theorem for condition ③
theorem condition_three :
  ∀ a : ℝ, (A ∩ B a = ∅) ↔ (a ≠ 1/6 ∧ a ≠ -1/2) := by sorry

end NUMINAMATH_CALUDE_condition_one_condition_two_condition_three_l3756_375601


namespace NUMINAMATH_CALUDE_vacuum_tube_alignment_l3756_375641

theorem vacuum_tube_alignment (f : Fin 7 → Fin 7) (h : Function.Bijective f) :
  ∃ x : Fin 7, f x = x := by
  sorry

end NUMINAMATH_CALUDE_vacuum_tube_alignment_l3756_375641


namespace NUMINAMATH_CALUDE_area_of_trapezoid_l3756_375697

structure Triangle where
  area : ℝ

structure Trapezoid where
  area : ℝ

def isosceles_triangle (t : Triangle) : Prop := sorry

theorem area_of_trapezoid (PQR : Triangle) (smallest : Triangle) (QSTM : Trapezoid) :
  isosceles_triangle PQR →
  PQR.area = 100 →
  smallest.area = 2 →
  QSTM.area = 90 := by
  sorry

end NUMINAMATH_CALUDE_area_of_trapezoid_l3756_375697


namespace NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l3756_375623

/-- Converts a base 9 number to base 10 --/
def base9To10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 9 --/
def base10To9 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 9 number --/
def isThreeDigitBase9 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 888

theorem greatest_3digit_base9_divisible_by_7 :
  ∃ (n : ℕ), isThreeDigitBase9 n ∧ 
             (base9To10 n) % 7 = 0 ∧
             ∀ (m : ℕ), isThreeDigitBase9 m ∧ (base9To10 m) % 7 = 0 → m ≤ n ∧
             n = 888 := by
  sorry

end NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l3756_375623


namespace NUMINAMATH_CALUDE_symmetrical_line_equation_l3756_375695

/-- Given two lines in the plane, this function returns the equation of a line symmetrical to the first line with respect to the second line. -/
def symmetricalLine (l1 l2 : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The line with equation 2x - y - 2 = 0 -/
def line1 : ℝ → ℝ → Prop :=
  fun x y ↦ 2 * x - y - 2 = 0

/-- The line with equation x + y - 4 = 0 -/
def line2 : ℝ → ℝ → Prop :=
  fun x y ↦ x + y - 4 = 0

/-- The theorem stating that the symmetrical line has the equation x - 2y + 2 = 0 -/
theorem symmetrical_line_equation :
  symmetricalLine line1 line2 = fun x y ↦ x - 2 * y + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetrical_line_equation_l3756_375695


namespace NUMINAMATH_CALUDE_ratio_of_amounts_l3756_375635

theorem ratio_of_amounts (total : ℕ) (r_amount : ℕ) (h1 : total = 5000) (h2 : r_amount = 2000) :
  (r_amount : ℚ) / ((total - r_amount) : ℚ) = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_of_amounts_l3756_375635


namespace NUMINAMATH_CALUDE_students_who_got_off_l3756_375690

/-- Given a school bus scenario where some students get off at a stop, 
    this theorem proves the number of students who got off. -/
theorem students_who_got_off (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 10) (h2 : remaining = 7) : initial - remaining = 3 := by
  sorry

#check students_who_got_off

end NUMINAMATH_CALUDE_students_who_got_off_l3756_375690


namespace NUMINAMATH_CALUDE_amount_ratio_l3756_375679

def total_amount : ℕ := 7000
def r_amount : ℕ := 2800

theorem amount_ratio : 
  let pq_amount := total_amount - r_amount
  (r_amount : ℚ) / (pq_amount : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_amount_ratio_l3756_375679


namespace NUMINAMATH_CALUDE_least_five_digit_divisible_by_15_12_18_l3756_375638

theorem least_five_digit_divisible_by_15_12_18 :
  ∃ n : ℕ, 
    n ≥ 10000 ∧ 
    n < 100000 ∧ 
    n % 15 = 0 ∧ 
    n % 12 = 0 ∧ 
    n % 18 = 0 ∧
    (∀ m : ℕ, m ≥ 10000 ∧ m < n ∧ m % 15 = 0 ∧ m % 12 = 0 ∧ m % 18 = 0 → false) ∧
    n = 10080 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_divisible_by_15_12_18_l3756_375638


namespace NUMINAMATH_CALUDE_log_ratio_identity_l3756_375677

theorem log_ratio_identity 
  (x y a b : ℝ) 
  (hx : x > 0) (hy : y > 0) (ha : a > 0) (hb : b > 0) 
  (ha_neq : a ≠ 1) (hb_neq : b ≠ 1) : 
  (Real.log x / Real.log a) / (Real.log y / Real.log a) = 1 / (Real.log y / Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_identity_l3756_375677


namespace NUMINAMATH_CALUDE_room_rent_problem_l3756_375680

theorem room_rent_problem (total_rent_A total_rent_B : ℝ) 
  (rent_difference : ℝ) (h1 : total_rent_A = 4800) (h2 : total_rent_B = 4200) 
  (h3 : rent_difference = 30) :
  let rent_A := 240
  let rent_B := 210
  (total_rent_A / rent_A = total_rent_B / rent_B) ∧ 
  (rent_A = rent_B + rent_difference) := by
  sorry

end NUMINAMATH_CALUDE_room_rent_problem_l3756_375680


namespace NUMINAMATH_CALUDE_vector_sum_length_l3756_375619

/-- Given vectors a and b in ℝ², prove that |a + 2b| = √61 under specific conditions. -/
theorem vector_sum_length (a b : ℝ × ℝ) : 
  a = (3, -4) → 
  ‖b‖ = 2 → 
  (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) = 1/2 →
  ‖a + 2 • b‖ = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_length_l3756_375619


namespace NUMINAMATH_CALUDE_sum_of_integers_l3756_375600

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 15) 
  (h2 : x.val * y.val = 54) : 
  x.val + y.val = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3756_375600


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3756_375646

theorem polynomial_factorization (m : ℝ) : -4 * m^3 + 4 * m^2 - m = -m * (2*m - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3756_375646


namespace NUMINAMATH_CALUDE_triangle_centroid_property_l3756_375669

open Real

variable (A B C Q G' : ℝ × ℝ)

def is_inside_triangle (P A B C : ℝ × ℝ) : Prop := sorry

def distance_squared (P Q : ℝ × ℝ) : ℝ := 
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem triangle_centroid_property :
  is_inside_triangle G' A B C →
  G' = ((1/4 : ℝ) • A + (1/4 : ℝ) • B + (1/2 : ℝ) • C) →
  distance_squared Q A + distance_squared Q B + distance_squared Q C = 
  4 * distance_squared Q G' + distance_squared G' A + distance_squared G' B + distance_squared G' C :=
by sorry

end NUMINAMATH_CALUDE_triangle_centroid_property_l3756_375669


namespace NUMINAMATH_CALUDE_spider_position_after_2055_jumps_l3756_375606

/-- Represents the possible positions on the circle -/
inductive Position : Type
  | one | two | three | four | five | six | seven

/-- Defines the next position after a hop based on the current position -/
def nextPosition (p : Position) : Position :=
  match p with
  | Position.one => Position.two
  | Position.two => Position.five
  | Position.three => Position.four
  | Position.four => Position.seven
  | Position.five => Position.six
  | Position.six => Position.two
  | Position.seven => Position.one

/-- Calculates the position after n hops -/
def positionAfterNHops (start : Position) (n : ℕ) : Position :=
  match n with
  | 0 => start
  | n + 1 => nextPosition (positionAfterNHops start n)

theorem spider_position_after_2055_jumps :
  positionAfterNHops Position.six 2055 = Position.two :=
sorry

end NUMINAMATH_CALUDE_spider_position_after_2055_jumps_l3756_375606


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_three_l3756_375628

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Definition of line l₁ -/
def line_l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 1 = 0

/-- Definition of line l₂ -/
def line_l₂ (a : ℝ) (x y : ℝ) : Prop := 2 * x + (a + 1) * y + 1 = 0

/-- Theorem: If l₁ and l₂ are parallel, then a = -3 -/
theorem parallel_lines_imply_a_eq_neg_three (a : ℝ) :
  (∀ x y : ℝ, line_l₁ a x y ↔ line_l₂ a x y) → a = -3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_three_l3756_375628


namespace NUMINAMATH_CALUDE_intersection_points_count_l3756_375686

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

def opposite_signs_at_1_2 (f : ℝ → ℝ) : Prop := f 1 * f 2 < 0

-- Define the number of intersections with the x-axis
def num_intersections (f : ℝ → ℝ) : ℕ :=
  -- This is a placeholder definition
  -- In practice, this would be defined more rigorously
  2

-- State the theorem
theorem intersection_points_count
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono : monotone_increasing_pos f)
  (h_signs : opposite_signs_at_1_2 f) :
  num_intersections f = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_count_l3756_375686


namespace NUMINAMATH_CALUDE_stating_call_ratio_theorem_l3756_375674

/-- Represents the ratio of calls processed by team members -/
structure CallRatio where
  team_a : ℚ
  team_b : ℚ

/-- Represents the distribution of calls and agents between two teams -/
structure CallCenter where
  agent_ratio : ℚ  -- Ratio of team A agents to team B agents
  team_b_calls : ℚ -- Fraction of total calls processed by team B

/-- 
Given a call center with specified agent ratio and call distribution,
calculates the ratio of calls processed by each member of team A to team B
-/
def calculate_call_ratio (cc : CallCenter) : CallRatio :=
  { team_a := 7,
    team_b := 5 }

/-- 
Theorem stating that for a call center where team A has 5/8 as many agents as team B,
and team B processes 8/15 of the calls, the ratio of calls processed per agent
of team A to team B is 7:5
-/
theorem call_ratio_theorem (cc : CallCenter) 
  (h1 : cc.agent_ratio = 5 / 8)
  (h2 : cc.team_b_calls = 8 / 15) :
  calculate_call_ratio cc = { team_a := 7, team_b := 5 } := by
  sorry

end NUMINAMATH_CALUDE_stating_call_ratio_theorem_l3756_375674


namespace NUMINAMATH_CALUDE_stating_no_equal_area_division_for_n_gt_2_l3756_375608

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents an angle bisector in a triangle -/
structure AngleBisector where
  origin : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- 
  Given a triangle and a set of angle bisectors from one vertex, 
  checks if they divide the triangle into n equal-area parts
-/
def divideIntoEqualAreas (t : Triangle) (bisectors : List AngleBisector) (n : ℕ) : Prop :=
  sorry

/-- 
  Theorem stating that for all triangles and integers n > 2, 
  it is impossible for the angle bisectors of one of the triangle's vertices 
  to divide the triangle into n equal-area parts
-/
theorem no_equal_area_division_for_n_gt_2 :
  ∀ (t : Triangle) (n : ℕ), n > 2 → ¬∃ (bisectors : List AngleBisector), 
  divideIntoEqualAreas t bisectors n :=
sorry

end NUMINAMATH_CALUDE_stating_no_equal_area_division_for_n_gt_2_l3756_375608


namespace NUMINAMATH_CALUDE_no_five_consecutive_divisible_by_2025_l3756_375665

def x (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2025 :
  ∀ k : ℕ, ∃ i : Fin 5, ¬(2025 ∣ x (k + i.val)) :=
by sorry

end NUMINAMATH_CALUDE_no_five_consecutive_divisible_by_2025_l3756_375665


namespace NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l3756_375666

/-- Calculates the amount John paid out of pocket for a new computer and accessories,
    given the costs and the sale of his PlayStation. -/
theorem johns_out_of_pocket_expense (computer_cost accessories_cost playstation_value : ℝ)
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : playstation_value = 400)
  (discount_rate : ℝ)
  (h4 : discount_rate = 0.2) :
  computer_cost + accessories_cost - playstation_value * (1 - discount_rate) = 580 := by
sorry


end NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l3756_375666


namespace NUMINAMATH_CALUDE_basketball_game_scores_l3756_375667

/-- Represents the scores of a team in four quarters -/
structure Scores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the given scores form an increasing geometric sequence -/
def isGeometric (s : Scores) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if the given scores form an increasing arithmetic sequence -/
def isArithmetic (s : Scores) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a team -/
def totalScore (s : Scores) : ℕ := s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the halftime score for a team -/
def halftimeScore (s : Scores) : ℕ := s.q1 + s.q2

theorem basketball_game_scores (eagles lions : Scores) 
  (h1 : isGeometric eagles)
  (h2 : isArithmetic lions)
  (h3 : halftimeScore eagles = halftimeScore lions)
  (h4 : totalScore eagles = totalScore lions) :
  halftimeScore eagles + halftimeScore lions = 8 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_scores_l3756_375667


namespace NUMINAMATH_CALUDE_part1_part2_l3756_375671

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- Part 1
theorem part1 (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x, x ∈ Set.Icc 1 a ↔ f a x ∈ Set.Icc 1 a) : 
  a = 2 := by sorry

-- Part 2
theorem part2 (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x₁ x₂, x₁ ∈ Set.Icc 1 (a+1) → x₂ ∈ Set.Icc 1 (a+1) → 
    |f a x₁ - f a x₂| ≤ 4) : 
  1 < a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3756_375671


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l3756_375663

theorem unique_number_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    (∀ d : ℕ, d ∣ x → d = 1 ∨ d = p ∨ d = q ∨ d = 11 ∨ d = p*q ∨ d = p*11 ∨ d = q*11 ∨ d = p*q*11)) →
  11 ∣ x →
  x = 59048 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l3756_375663


namespace NUMINAMATH_CALUDE_drama_club_subjects_l3756_375636

/-- Given a group of students in a drama club, prove the number of students
    taking neither mathematics nor physics. -/
theorem drama_club_subjects (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ)
    (h1 : total = 60)
    (h2 : math = 40)
    (h3 : physics = 35)
    (h4 : both = 25) :
    total - (math + physics - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_subjects_l3756_375636


namespace NUMINAMATH_CALUDE_specific_hyperbola_real_axis_length_l3756_375670

/-- A hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  -- The hyperbola passes through this point
  point : ℝ × ℝ
  -- The equations of the asymptotes
  asymptote1 : ℝ → ℝ → ℝ
  asymptote2 : ℝ → ℝ → ℝ

/-- The length of the real axis of a hyperbola -/
def realAxisLength (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem stating the length of the real axis of the specific hyperbola -/
theorem specific_hyperbola_real_axis_length :
  ∃ (h : Hyperbola),
    h.point = (5, -2) ∧
    h.asymptote1 = (λ x y => x - 2*y) ∧
    h.asymptote2 = (λ x y => x + 2*y) ∧
    realAxisLength h = 6 :=
  sorry

end NUMINAMATH_CALUDE_specific_hyperbola_real_axis_length_l3756_375670


namespace NUMINAMATH_CALUDE_division_simplification_l3756_375660

theorem division_simplification (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (8 * a^3 * b - 4 * a^2 * b^2) / (4 * a * b) = 2 * a^2 - a * b :=
by sorry

end NUMINAMATH_CALUDE_division_simplification_l3756_375660


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_18_l3756_375672

theorem five_digit_divisible_by_18 (n : ℕ) : 
  n < 10 ∧ 
  73420 ≤ 7342 * 10 + n ∧ 
  7342 * 10 + n < 73430 ∧
  (7342 * 10 + n) % 18 = 0 
  ↔ n = 2 := by sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_18_l3756_375672


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3756_375654

theorem complex_equation_solution : ∃ (a b : ℝ), (Complex.mk a b) * (Complex.mk a b + Complex.I) * (Complex.mk a b + 2 * Complex.I) = 1001 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3756_375654


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_two_distinct_roots_l3756_375626

theorem at_least_one_equation_has_two_distinct_roots
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (4 * b^2 - 4 * a * c > 0) ∨ (4 * c^2 - 4 * a * b > 0) ∨ (4 * a^2 - 4 * b * c > 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_two_distinct_roots_l3756_375626


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3756_375637

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * y) + f (x + y) = f x * f y + f x + f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3756_375637


namespace NUMINAMATH_CALUDE_function_characterization_l3756_375661

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  f 0 ≠ 0 ∧
  ∀ x y : ℝ, f (x + y)^2 = 2 * f x * f y + max (f (x^2) + f (y^2)) (f (x^2 + y^2))

/-- The theorem stating that any function satisfying the equation must be either constant -1 or x - 1 -/
theorem function_characterization (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  (∀ x, f x = -1) ∨ (∀ x, f x = x - 1) := by sorry

end NUMINAMATH_CALUDE_function_characterization_l3756_375661


namespace NUMINAMATH_CALUDE_exp_function_inequality_l3756_375631

/-- Given an exponential function f(x) = a^x where 0 < a < 1, 
    prove that f(3) * f(2) < f(2) -/
theorem exp_function_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let f := fun (x : ℝ) => a^x
  f 3 * f 2 < f 2 := by
  sorry

end NUMINAMATH_CALUDE_exp_function_inequality_l3756_375631


namespace NUMINAMATH_CALUDE_exists_a_with_two_common_tangents_l3756_375659

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C₂ -/
def C₂ (x y a : ℝ) : Prop := (x - 4)^2 + (y + a)^2 = 64

/-- Condition for two circles to have exactly 2 common tangents -/
def has_two_common_tangents (a : ℝ) : Prop :=
  6 < Real.sqrt (16 + a^2) ∧ Real.sqrt (16 + a^2) < 10

/-- Theorem stating the existence of a positive integer a satisfying the conditions -/
theorem exists_a_with_two_common_tangents :
  ∃ a : ℕ+, has_two_common_tangents a.val := by sorry

end NUMINAMATH_CALUDE_exists_a_with_two_common_tangents_l3756_375659


namespace NUMINAMATH_CALUDE_length_a_prime_b_prime_l3756_375620

/-- Given points A, B, C, and the line y = x, prove that the length of A'B' is 4√2 -/
theorem length_a_prime_b_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 6) →
  B = (0, 10) →
  C = (3, 7) →
  (A'.1 = A'.2 ∧ B'.1 = B'.2) →  -- A' and B' are on the line y = x
  (∃ t : ℝ, A + t • (A' - A) = C) →  -- AA' passes through C
  (∃ s : ℝ, B + s • (B' - B) = C) →  -- BB' passes through C
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_length_a_prime_b_prime_l3756_375620


namespace NUMINAMATH_CALUDE_average_fuel_efficiency_l3756_375696

/-- Calculates the average fuel efficiency for a trip with multiple segments and different vehicles. -/
theorem average_fuel_efficiency 
  (total_distance : ℝ)
  (sedan_distance : ℝ)
  (truck_distance : ℝ)
  (detour_distance : ℝ)
  (sedan_efficiency : ℝ)
  (truck_efficiency : ℝ)
  (detour_efficiency : ℝ)
  (h1 : total_distance = sedan_distance + truck_distance + detour_distance)
  (h2 : sedan_distance = 150)
  (h3 : truck_distance = 150)
  (h4 : detour_distance = 50)
  (h5 : sedan_efficiency = 25)
  (h6 : truck_efficiency = 15)
  (h7 : detour_efficiency = 10) :
  ∃ (ε : ℝ), abs (total_distance / (sedan_distance / sedan_efficiency + 
                                    truck_distance / truck_efficiency + 
                                    detour_distance / detour_efficiency) - 16.67) < ε :=
by sorry

end NUMINAMATH_CALUDE_average_fuel_efficiency_l3756_375696


namespace NUMINAMATH_CALUDE_lee_soccer_game_probability_l3756_375624

theorem lee_soccer_game_probability (p : ℚ) (h : p = 5/9) :
  1 - p = 4/9 := by sorry

end NUMINAMATH_CALUDE_lee_soccer_game_probability_l3756_375624


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3756_375648

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 39) →
  (a 3 + a 6 + a 9 = 33) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3756_375648


namespace NUMINAMATH_CALUDE_parallel_line_not_through_point_l3756_375652

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.A * p.x + l.B * p.y + l.C = 0

theorem parallel_line_not_through_point
    (L : Line)
    (P : Point)
    (h_not_on : ¬ P.onLine L) :
    ∃ (k : ℝ),
      k ≠ 0 ∧
      (∀ (x y : ℝ),
        L.A * x + L.B * y + L.C + (L.A * P.x + L.B * P.y + L.C) = 0 ↔
        L.A * x + L.B * y + L.C + k = 0) ∧
      (L.A * P.x + L.B * P.y + L.C + k ≠ 0) :=
  sorry

end NUMINAMATH_CALUDE_parallel_line_not_through_point_l3756_375652


namespace NUMINAMATH_CALUDE_combined_weight_l3756_375643

theorem combined_weight (person baby nurse : ℝ)
  (h1 : person + baby = 78)
  (h2 : nurse + baby = 69)
  (h3 : person + nurse = 137) :
  person + nurse + baby = 142 :=
by sorry

end NUMINAMATH_CALUDE_combined_weight_l3756_375643


namespace NUMINAMATH_CALUDE_f_power_of_two_divides_l3756_375698

/-- f(d) is the smallest possible integer that has exactly d positive divisors -/
def f (d : ℕ) : ℕ := sorry

/-- Theorem: For every non-negative integer k, f(2^k) divides f(2^(k+1)) -/
theorem f_power_of_two_divides (k : ℕ) : 
  (f (2^k)) ∣ (f (2^(k+1))) := by sorry

end NUMINAMATH_CALUDE_f_power_of_two_divides_l3756_375698


namespace NUMINAMATH_CALUDE_right_triangle_product_divisible_by_30_l3756_375602

theorem right_triangle_product_divisible_by_30 (a b c : ℤ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (30 : ℤ) ∣ (a * b * c) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_product_divisible_by_30_l3756_375602


namespace NUMINAMATH_CALUDE_nested_average_equals_seven_ninths_l3756_375694

def average2 (a b : ℚ) : ℚ := (a + b) / 2

def average3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem nested_average_equals_seven_ninths :
  average3 (average3 2 2 0) (average2 0 2) 0 = 7/9 := by sorry

end NUMINAMATH_CALUDE_nested_average_equals_seven_ninths_l3756_375694


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3756_375685

theorem rectangle_area_change (initial_short : ℝ) (initial_long : ℝ) 
  (h1 : initial_short = 5)
  (h2 : initial_long = 7)
  (h3 : ∃ x, initial_short * (initial_long - x) = 24) :
  (initial_short * (initial_long - 2) = 25) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3756_375685


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3756_375629

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a →
  a 1 = 2 →
  a 2 + a 5 = 13 →
  a 5 + a 6 + a 7 = 33 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3756_375629


namespace NUMINAMATH_CALUDE_sum_mod_seven_l3756_375658

theorem sum_mod_seven : (5432 + 5433 + 5434 + 5435) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l3756_375658


namespace NUMINAMATH_CALUDE_money_split_proof_l3756_375676

/-- 
Given two people splitting money in a 2:3 ratio where the smaller share is $50,
prove that the total amount shared is $125.
-/
theorem money_split_proof (smaller_share : ℕ) (total : ℕ) : 
  smaller_share = 50 → 
  2 * total = 5 * smaller_share →
  total = 125 := by
sorry

end NUMINAMATH_CALUDE_money_split_proof_l3756_375676


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3756_375684

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 + a 5 = -6 →
  a 2 * a 6 = 8 →
  a 1 + a 7 = -9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3756_375684


namespace NUMINAMATH_CALUDE_function_property_l3756_375610

def IteratedFunction (f : ℕ+ → ℕ+) : ℕ → ℕ+ → ℕ+
  | 0, n => n
  | k+1, n => f (IteratedFunction f k n)

theorem function_property (f : ℕ+ → ℕ+) :
  (∀ (a b c : ℕ+), a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 →
    IteratedFunction f (a*b*c - a) (a*b*c) + 
    IteratedFunction f (a*b*c - b) (a*b*c) + 
    IteratedFunction f (a*b*c - c) (a*b*c) = a + b + c) →
  ∀ n : ℕ+, n ≥ 3 → f n = n - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3756_375610


namespace NUMINAMATH_CALUDE_solution_set_f_gt_x_range_of_a_when_f_plus_3_nonneg_l3756_375668

noncomputable section

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x - (2*a + 2)

-- Part 1: Solution set of f(x) > x
theorem solution_set_f_gt_x (a : ℝ) :
  (∀ x, f a x > x ↔ 
    (a > -3/2 ∧ (x > 2*a + 2 ∨ x < -1)) ∨
    (a = -3/2 ∧ x ≠ -1) ∨
    (a < -3/2 ∧ (x > -1 ∨ x < 2*a + 2))) :=
sorry

-- Part 2: Range of a when f(x) + 3 ≥ 0 for x ∈ (-1, +∞)
theorem range_of_a_when_f_plus_3_nonneg :
  (∀ x, x > -1 → f a x + 3 ≥ 0) ↔ a ≤ Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_x_range_of_a_when_f_plus_3_nonneg_l3756_375668


namespace NUMINAMATH_CALUDE_scientific_notation_of_28400_l3756_375647

theorem scientific_notation_of_28400 :
  28400 = 2.84 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_28400_l3756_375647


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3756_375689

theorem line_circle_intersection (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = k * A.1 + 2) ∧ 
    (B.2 = k * B.1 + 2) ∧ 
    ((A.1 - 3)^2 + (A.2 - 1)^2 = 9) ∧ 
    ((B.1 - 3)^2 + (B.2 - 1)^2 = 9) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 32)) →
  (k = 0 ∨ k = -3/4) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3756_375689


namespace NUMINAMATH_CALUDE_fruit_combination_count_l3756_375644

/-- The number of combinations when choosing k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of fruit types available -/
def fruit_types : ℕ := 4

/-- The number of fruits to be chosen -/
def fruits_to_choose : ℕ := 3

/-- Theorem: The number of combinations when choosing 3 fruits from 4 types is 4 -/
theorem fruit_combination_count : choose fruit_types fruits_to_choose = 4 := by
  sorry

end NUMINAMATH_CALUDE_fruit_combination_count_l3756_375644


namespace NUMINAMATH_CALUDE_omega_range_l3756_375609

/-- Given a function f(x) = 2sin(ωx) with ω > 0, if f(x) has a minimum value of -2 
    on the interval [-π/3, π/4], then 0 < ω ≤ 3/2 -/
theorem omega_range (ω : ℝ) (h1 : ω > 0) : 
  (∀ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω * x) ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω * x) = -2) →
  0 < ω ∧ ω ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_omega_range_l3756_375609


namespace NUMINAMATH_CALUDE_problem_solution_l3756_375605

theorem problem_solution (a : ℝ) : 3 ∈ ({a, a^2 - 2*a} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3756_375605


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3756_375649

theorem complex_equation_solution (z : ℂ) : 
  (3 + 4*I) / I = z / (1 + I) → z = 7 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3756_375649


namespace NUMINAMATH_CALUDE_no_solution_for_specific_p_range_l3756_375607

theorem no_solution_for_specific_p_range (p : ℝ) (h : 4/3 < p ∧ p < 2) :
  ¬∃ x : ℝ, Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_specific_p_range_l3756_375607


namespace NUMINAMATH_CALUDE_remainder_13_pow_1033_mod_50_l3756_375622

theorem remainder_13_pow_1033_mod_50 : 13^1033 % 50 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_13_pow_1033_mod_50_l3756_375622


namespace NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l3756_375653

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The number of trailing zeros in 30! is 7 -/
theorem trailing_zeros_30_factorial : trailingZeros 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l3756_375653


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l3756_375611

theorem quadratic_distinct_roots (n : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + n*x + 9 = 0 ∧ y^2 + n*y + 9 = 0) ↔ 
  (n < -6 ∨ n > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l3756_375611


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l3756_375621

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_15th_term 
  (a : ℕ → ℕ) 
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 12)
  (h_third : a 3 = 21) :
  a 15 = 129 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l3756_375621


namespace NUMINAMATH_CALUDE_dwayne_class_a_count_l3756_375633

/-- Proves that given the conditions from Mrs. Carter's and Mr. Dwayne's classes,
    the number of students who received an 'A' in Mr. Dwayne's class is 12. -/
theorem dwayne_class_a_count :
  let carter_total : ℕ := 20
  let carter_a_count : ℕ := 8
  let dwayne_total : ℕ := 30
  let ratio : ℚ := carter_a_count / carter_total
  ∃ (dwayne_a_count : ℕ), 
    (dwayne_a_count : ℚ) / dwayne_total = ratio ∧ 
    dwayne_a_count = 12 :=
by sorry

end NUMINAMATH_CALUDE_dwayne_class_a_count_l3756_375633


namespace NUMINAMATH_CALUDE_class_grade_average_l3756_375687

theorem class_grade_average (n : ℕ) (h : n > 0) :
  let first_quarter := n / 4
  let remaining := n - first_quarter
  let first_quarter_avg := 92
  let remaining_avg := 76
  let total_sum := first_quarter * first_quarter_avg + remaining * remaining_avg
  (total_sum : ℚ) / n = 80 := by
sorry

end NUMINAMATH_CALUDE_class_grade_average_l3756_375687


namespace NUMINAMATH_CALUDE_camp_cedar_counselors_l3756_375692

/-- The number of counselors needed at Camp Cedar --/
def counselors_needed (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  (num_boys / 6) + (num_girls / 10)

/-- Theorem: Camp Cedar needs 26 counselors --/
theorem camp_cedar_counselors :
  let num_boys : ℕ := 48
  let num_girls : ℕ := 4 * num_boys - 12
  counselors_needed num_boys num_girls = 26 := by
  sorry


end NUMINAMATH_CALUDE_camp_cedar_counselors_l3756_375692


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l3756_375642

/-- A circle tangent to the parabola y^2 = 2x (y > 0), its axis, and the x-axis -/
structure TangentCircle where
  /-- Center of the circle -/
  center : ℝ × ℝ
  /-- Radius of the circle -/
  radius : ℝ
  /-- The circle is tangent to the parabola y^2 = 2x (y > 0) -/
  tangent_to_parabola : center.2^2 = 2 * center.1
  /-- The circle is tangent to the x-axis -/
  tangent_to_x_axis : center.2 = radius
  /-- The circle's center is on the axis of the parabola (x-axis) -/
  on_parabola_axis : center.1 ≥ 0

/-- The equation of the circle is x^2 + y^2 - x - 2y + 1/4 = 0 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔
  x^2 + y^2 - x - 2*y + 1/4 = 0 := by
  sorry


end NUMINAMATH_CALUDE_tangent_circle_equation_l3756_375642


namespace NUMINAMATH_CALUDE_expand_binomials_l3756_375683

theorem expand_binomials (a : ℝ) : (a + 3) * (-a + 1) = -a^2 - 2*a + 3 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l3756_375683


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3756_375625

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3756_375625


namespace NUMINAMATH_CALUDE_morning_rowers_count_l3756_375645

def total_rowers : ℕ := 34
def afternoon_rowers : ℕ := 21

theorem morning_rowers_count : total_rowers - afternoon_rowers = 13 := by
  sorry

end NUMINAMATH_CALUDE_morning_rowers_count_l3756_375645


namespace NUMINAMATH_CALUDE_white_pieces_count_l3756_375655

/-- The number of possible arrangements of chess pieces -/
def total_arrangements : ℕ := 144

/-- The number of black chess pieces -/
def black_pieces : ℕ := 3

/-- Function to calculate the number of arrangements given white and black pieces -/
def arrangements (white : ℕ) (black : ℕ) : ℕ :=
  (Nat.factorial white) * (Nat.factorial black)

/-- Theorem stating that there are 4 white chess pieces -/
theorem white_pieces_count :
  ∃ (w : ℕ), w > 0 ∧ 
    arrangements w black_pieces = total_arrangements ∧ 
    (w = black_pieces ∨ w = black_pieces + 1) :=
by sorry

end NUMINAMATH_CALUDE_white_pieces_count_l3756_375655


namespace NUMINAMATH_CALUDE_jumping_contest_l3756_375639

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump mouse_jump : ℕ) 
  (h1 : grasshopper_jump = 25)
  (h2 : mouse_jump = 31) : 
  (grasshopper_jump + 32) - mouse_jump = 26 := by
  sorry


end NUMINAMATH_CALUDE_jumping_contest_l3756_375639


namespace NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l3756_375614

-- Define the variables
variable (x y z : ℝ)
-- Define constants of proportionality
variable (k j : ℝ)

-- Define the relationships
def x_varies_as_y_squared : Prop := ∃ k > 0, x = k * y^2
def y_varies_as_cube_root_z_squared : Prop := ∃ j > 0, y = j * (z^2)^(1/3)

-- State the theorem
theorem x_varies_as_four_thirds_power_of_z 
  (h1 : x_varies_as_y_squared x y) 
  (h2 : y_varies_as_cube_root_z_squared y z) : 
  ∃ m > 0, x = m * z^(4/3) := by
  sorry

end NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l3756_375614


namespace NUMINAMATH_CALUDE_multiply_586645_by_9999_l3756_375699

theorem multiply_586645_by_9999 : 586645 * 9999 = 5865864355 := by
  sorry

end NUMINAMATH_CALUDE_multiply_586645_by_9999_l3756_375699


namespace NUMINAMATH_CALUDE_sequence_uniqueness_l3756_375627

theorem sequence_uniqueness (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, n ≥ 1 → (a (n + 1))^2 = 1 + (n + 2021) * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = n + 2019 := by
  sorry

end NUMINAMATH_CALUDE_sequence_uniqueness_l3756_375627


namespace NUMINAMATH_CALUDE_pyramid_volume_l3756_375615

theorem pyramid_volume (h : ℝ) (h_parallel : ℝ) (cross_section_area : ℝ) :
  h = 8 →
  h_parallel = 3 →
  cross_section_area = 4 →
  (1/3 : ℝ) * (cross_section_area * (h / h_parallel)^2) * h = 2048/27 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3756_375615


namespace NUMINAMATH_CALUDE_sum_of_digits_square_of_nine_twos_l3756_375688

/-- The sum of digits of the square of a number consisting of n twos -/
def sum_of_digits_square_of_twos (n : ℕ) : ℕ := 2 * n^2

/-- The number of twos in our specific case -/
def num_twos : ℕ := 9

/-- Theorem: The sum of the digits of the square of a number consisting of 9 twos is 162 -/
theorem sum_of_digits_square_of_nine_twos :
  sum_of_digits_square_of_twos num_twos = 162 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_of_nine_twos_l3756_375688


namespace NUMINAMATH_CALUDE_cake_division_l3756_375630

theorem cake_division (cake_weight : ℝ) (pierre_ate : ℝ) : 
  cake_weight = 400 ∧ pierre_ate = 100 → 
  ∃ (n : ℕ), n = 8 ∧ cake_weight / n = pierre_ate / 2 := by
sorry

end NUMINAMATH_CALUDE_cake_division_l3756_375630


namespace NUMINAMATH_CALUDE_max_angle_A1MC1_is_pi_over_2_l3756_375613

/-- Represents a right square prism -/
structure RightSquarePrism where
  base_side : ℝ
  height : ℝ
  height_eq_half_base : height = base_side / 2

/-- Represents a point on an edge of the prism -/
structure EdgePoint where
  x : ℝ
  valid : 0 ≤ x ∧ x ≤ 1

/-- Calculates the angle A₁MC₁ given a point M on edge AB -/
def angle_A1MC1 (prism : RightSquarePrism) (M : EdgePoint) : ℝ := sorry

/-- Theorem: The maximum value of angle A₁MC₁ in a right square prism is π/2 -/
theorem max_angle_A1MC1_is_pi_over_2 (prism : RightSquarePrism) :
  ∃ M : EdgePoint, ∀ N : EdgePoint, angle_A1MC1 prism M ≥ angle_A1MC1 prism N ∧ 
  angle_A1MC1 prism M = π / 2 :=
sorry

end NUMINAMATH_CALUDE_max_angle_A1MC1_is_pi_over_2_l3756_375613


namespace NUMINAMATH_CALUDE_darkest_cell_value_l3756_375682

/-- Represents the grid structure -/
structure Grid :=
  (white1 white2 white3 white4 : Nat)
  (gray1 gray2 : Nat)
  (dark : Nat)

/-- The grid satisfies the problem conditions -/
def valid_grid (g : Grid) : Prop :=
  g.white1 > 1 ∧ g.white2 > 1 ∧ g.white3 > 1 ∧ g.white4 > 1 ∧
  g.white1 * g.white2 = 55 ∧
  g.white3 * g.white4 = 55 ∧
  g.gray1 = g.white1 * g.white3 ∧
  g.gray2 = g.white2 * g.white4 ∧
  g.dark = g.gray1 * g.gray2

theorem darkest_cell_value (g : Grid) :
  valid_grid g → g.dark = 245025 := by
  sorry

#check darkest_cell_value

end NUMINAMATH_CALUDE_darkest_cell_value_l3756_375682


namespace NUMINAMATH_CALUDE_factor_expression_l3756_375657

theorem factor_expression (x : ℝ) : x * (x + 4) + 3 * (x + 4) = (x + 4) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3756_375657


namespace NUMINAMATH_CALUDE_parallelogram_area_l3756_375651

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 inches and 20 inches is 100√3 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 20) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin ((180 - θ) * π / 180) = 100 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3756_375651


namespace NUMINAMATH_CALUDE_expression_evaluation_l3756_375616

theorem expression_evaluation :
  let a : ℚ := -1/2
  (a + 3)^2 - (a + 1) * (a - 1) - 2 * (2 * a + 4) = 1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3756_375616


namespace NUMINAMATH_CALUDE_gcd_digit_sum_theorem_l3756_375693

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem gcd_digit_sum_theorem : 
  let a := 4665 - 1305
  let b := 6905 - 4665
  let c := 6905 - 1305
  let gcd_result := Nat.gcd (Nat.gcd a b) c
  sum_of_digits gcd_result = 4 := by sorry

end NUMINAMATH_CALUDE_gcd_digit_sum_theorem_l3756_375693


namespace NUMINAMATH_CALUDE_girls_in_class_l3756_375618

theorem girls_in_class (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 56 → 
  4 * boys = 3 * girls → 
  total = girls + boys → 
  girls = 32 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l3756_375618


namespace NUMINAMATH_CALUDE_lcm_of_135_and_468_l3756_375664

theorem lcm_of_135_and_468 : Nat.lcm 135 468 = 7020 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_135_and_468_l3756_375664


namespace NUMINAMATH_CALUDE_binary_polynomial_form_l3756_375681

/-- A binary homogeneous polynomial of degree n -/
def BinaryHomogeneousPolynomial (n : ℕ) := ℝ → ℝ → ℝ

/-- The polynomial condition for all real numbers a, b, c -/
def SatisfiesCondition (P : BinaryHomogeneousPolynomial n) : Prop :=
  ∀ a b c : ℝ, P (a + b) c + P (b + c) a + P (c + a) b = 0

/-- The theorem stating the form of the polynomial P -/
theorem binary_polynomial_form (n : ℕ) (P : BinaryHomogeneousPolynomial n)
  (h1 : SatisfiesCondition P) (h2 : P 1 0 = 1) :
  ∃ f : ℝ → ℝ → ℝ, (∀ x y : ℝ, P x y = f x y * (x - 2*y)) ∧
                    (∀ x y : ℝ, f x y = (x + y)^(n-1)) :=
sorry

end NUMINAMATH_CALUDE_binary_polynomial_form_l3756_375681


namespace NUMINAMATH_CALUDE_f_composition_result_l3756_375678

noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z ^ 2 else -(z ^ 2)

theorem f_composition_result : f (f (f (f (2 + I)))) = 164833 + 354816 * I := by
  sorry

end NUMINAMATH_CALUDE_f_composition_result_l3756_375678


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3756_375691

/-- The function f(x) = x³ - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The point A through which the tangent line passes --/
def A : ℝ × ℝ := (0, 16)

/-- The point of tangency M --/
def M : ℝ × ℝ := (-2, f (-2))

theorem tangent_line_equation :
  ∀ x y : ℝ, (9:ℝ)*x - y + 16 = 0 ↔ 
  (y - M.2 = f' M.1 * (x - M.1) ∧ f M.1 = M.2 ∧ A.2 - M.2 = f' M.1 * (A.1 - M.1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3756_375691


namespace NUMINAMATH_CALUDE_least_integer_square_quadruple_l3756_375603

theorem least_integer_square_quadruple (x : ℤ) : x^2 = 4*x + 56 → x ≥ -7 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_quadruple_l3756_375603


namespace NUMINAMATH_CALUDE_calculation_proof_l3756_375640

theorem calculation_proof : (2.5 * (30.1 + 0.5)) / 1.5 = 51 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3756_375640


namespace NUMINAMATH_CALUDE_smallest_possible_M_l3756_375617

theorem smallest_possible_M (a b c d e : ℕ+) (h_sum : a + b + c + d + e = 2010) :
  let M := max (a + b) (max (b + c) (max (c + d) (d + e)))
  ∀ M', (∃ a' b' c' d' e' : ℕ+, a' + b' + c' + d' + e' = 2010 ∧
    M' = max (a' + b') (max (b' + c') (max (c' + d') (d' + e')))) →
  M' ≥ 671 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_M_l3756_375617


namespace NUMINAMATH_CALUDE_sticker_count_l3756_375604

/-- The total number of stickers Ryan, Steven, and Terry have altogether -/
def total_stickers (ryan_stickers : ℕ) (steven_multiplier : ℕ) (terry_extra : ℕ) : ℕ :=
  ryan_stickers + 
  (steven_multiplier * ryan_stickers) + 
  (steven_multiplier * ryan_stickers + terry_extra)

/-- Proof that the total number of stickers is 230 -/
theorem sticker_count : total_stickers 30 3 20 = 230 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l3756_375604


namespace NUMINAMATH_CALUDE_first_term_to_common_diff_ratio_l3756_375675

/-- An arithmetic progression with a specific property -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_15_eq_3sum_5 : (15 * a + 105 * d) = 3 * (5 * a + 10 * d)

/-- The ratio of the first term to the common difference is 5:1 -/
theorem first_term_to_common_diff_ratio 
  (ap : ArithmeticProgression) : ap.a / ap.d = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_term_to_common_diff_ratio_l3756_375675


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l3756_375650

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (new_player1_weight : ℕ) 
  (new_player2_weight : ℕ) 
  (new_average_weight : ℕ) 
  (h1 : original_players = 7)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_average_weight = 99) : 
  ∃ (original_average : ℕ), 
    (original_players * original_average + new_player1_weight + new_player2_weight) / 
    (original_players + 2) = new_average_weight ∧ 
    original_average = 103 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l3756_375650


namespace NUMINAMATH_CALUDE_solve_simultaneous_equations_l3756_375662

theorem solve_simultaneous_equations (a u : ℝ) 
  (eq1 : 3 / a + 1 / u = 7 / 2)
  (eq2 : 2 / a - 3 / u = 6) :
  a = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_solve_simultaneous_equations_l3756_375662


namespace NUMINAMATH_CALUDE_baker_cakes_l3756_375612

/-- The initial number of cakes Baker made -/
def initial_cakes : ℕ := 169

/-- The number of cakes Baker's friend bought -/
def bought_cakes : ℕ := 137

/-- The number of cakes Baker has left -/
def remaining_cakes : ℕ := 32

/-- Theorem stating that the initial number of cakes is equal to the sum of bought cakes and remaining cakes -/
theorem baker_cakes : initial_cakes = bought_cakes + remaining_cakes := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l3756_375612


namespace NUMINAMATH_CALUDE_max_sum_of_complex_numbers_l3756_375673

theorem max_sum_of_complex_numbers (a b : ℂ) : 
  a^2 + b^2 = 5 → 
  a^3 + b^3 = 7 → 
  (a + b).re ≤ (-1 + Real.sqrt 57) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_complex_numbers_l3756_375673
