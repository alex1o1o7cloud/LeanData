import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l1746_174656

theorem equation_solution (p : ℝ) (hp : p > 0) :
  ∃ x : ℝ, Real.sqrt (x^2 + 2*p*x - p^2) - Real.sqrt (x^2 - 2*p*x - p^2) = 1 ↔
  (|p| < 1/2 ∧ (x = Real.sqrt ((p^2 + 1/4) / (1 - 4*p^2)) ∨
               x = -Real.sqrt ((p^2 + 1/4) / (1 - 4*p^2)))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1746_174656


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1746_174621

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 7) + x = x^2 + 5*x - 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1746_174621


namespace NUMINAMATH_CALUDE_line_xz_plane_intersection_l1746_174604

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- The xz-plane -/
def xzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p.x = l.p1.x + t * (l.p2.x - l.p1.x) ∧
            p.y = l.p1.y + t * (l.p2.y - l.p1.y) ∧
            p.z = l.p1.z + t * (l.p2.z - l.p1.z)

theorem line_xz_plane_intersection :
  let l : Line3D := {
    p1 := { x := 2, y := -1, z := 3 },
    p2 := { x := 6, y := -4, z := 7 }
  }
  let intersectionPoint : Point3D := { x := 2/3, y := 0, z := 5/3 }
  (intersectionPoint ∈ xzPlane) ∧ 
  (pointOnLine intersectionPoint l) := by sorry

end NUMINAMATH_CALUDE_line_xz_plane_intersection_l1746_174604


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1746_174678

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 3) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 5 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = -3 ∧
    B = -2 ∧
    C = 0 ∧
    D = 1 ∧
    E = 30 ∧
    F = 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1746_174678


namespace NUMINAMATH_CALUDE_dynamic_load_calculation_l1746_174620

/-- Given an architectural formula for dynamic load on cylindrical columns -/
theorem dynamic_load_calculation (T H : ℝ) (hT : T = 3) (hH : H = 6) :
  (50 * T^3) / H^3 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_dynamic_load_calculation_l1746_174620


namespace NUMINAMATH_CALUDE_triangle_angle_solution_l1746_174668

theorem triangle_angle_solution (a b c : ℝ) (h1 : a = 40)
  (h2 : b = 3 * y) (h3 : c = y + 10) (h4 : a + b + c = 180) : y = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_solution_l1746_174668


namespace NUMINAMATH_CALUDE_debate_club_committee_compositions_l1746_174605

def total_candidates : ℕ := 20
def past_members : ℕ := 10
def committee_size : ℕ := 5
def min_past_members : ℕ := 3

theorem debate_club_committee_compositions :
  (Nat.choose past_members min_past_members * Nat.choose (total_candidates - past_members) (committee_size - min_past_members)) +
  (Nat.choose past_members (min_past_members + 1) * Nat.choose (total_candidates - past_members) (committee_size - (min_past_members + 1))) +
  (Nat.choose past_members committee_size) = 7752 := by
  sorry

end NUMINAMATH_CALUDE_debate_club_committee_compositions_l1746_174605


namespace NUMINAMATH_CALUDE_parities_of_E_10_11_12_l1746_174693

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => 2 * E (n + 2) + E n

theorem parities_of_E_10_11_12 :
  Even (E 10) ∧ Odd (E 11) ∧ Odd (E 12) := by
  sorry

end NUMINAMATH_CALUDE_parities_of_E_10_11_12_l1746_174693


namespace NUMINAMATH_CALUDE_max_sum_cubes_l1746_174639

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  (∃ (x y z w v : ℝ), x^2 + y^2 + z^2 + w^2 + v^2 = 5 ∧ 
   x^3 + y^3 + z^3 + w^3 + v^3 ≥ a^3 + b^3 + c^3 + d^3 + e^3) ∧
  (∀ (x y z w v : ℝ), x^2 + y^2 + z^2 + w^2 + v^2 = 5 → 
   x^3 + y^3 + z^3 + w^3 + v^3 ≤ 5 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l1746_174639


namespace NUMINAMATH_CALUDE_car_speed_proof_l1746_174679

/-- Proves that given the conditions of the car journey, the initial speed must be 75 km/hr -/
theorem car_speed_proof (v : ℝ) : 
  v > 0 →
  (320 / (160 / v + 160 / 80) = 77.4193548387097) →
  v = 75 := by
sorry

end NUMINAMATH_CALUDE_car_speed_proof_l1746_174679


namespace NUMINAMATH_CALUDE_circles_intersect_l1746_174629

theorem circles_intersect (r R d : ℝ) (hr : r = 4) (hR : R = 5) (hd : d = 6) :
  let sum := r + R
  let diff := R - r
  d > diff ∧ d < sum := by sorry

end NUMINAMATH_CALUDE_circles_intersect_l1746_174629


namespace NUMINAMATH_CALUDE_range_of_m_l1746_174608

-- Define the quadratic function
def f (m x : ℝ) := m * x^2 - m * x - 1

-- Define the solution set
def solution_set (m : ℝ) := {x : ℝ | f m x ≥ 0}

-- State the theorem
theorem range_of_m : 
  (∀ m : ℝ, solution_set m = ∅) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1746_174608


namespace NUMINAMATH_CALUDE_hypotenuse_product_squared_l1746_174670

/-- Right triangle with given area and side lengths -/
structure RightTriangle where
  area : ℝ
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  area_eq : area = (side1 * side2) / 2
  pythagorean : side1^2 + side2^2 = hypotenuse^2

/-- The problem statement -/
theorem hypotenuse_product_squared
  (T₁ T₂ : RightTriangle)
  (h_area₁ : T₁.area = 2)
  (h_area₂ : T₂.area = 3)
  (h_side_congruent : T₁.side1 = T₂.side1)
  (h_side_double : T₁.side2 = 2 * T₂.side2) :
  (T₁.hypotenuse * T₂.hypotenuse)^2 = 325 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_product_squared_l1746_174670


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_one_third_of_one_fourth_of_one_fifth_of_sixty_l1746_174618

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem one_third_of_one_fourth_of_one_fifth_of_sixty :
  (1 : ℚ) / 3 * (1 : ℚ) / 4 * (1 : ℚ) / 5 * 60 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_one_third_of_one_fourth_of_one_fifth_of_sixty_l1746_174618


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1746_174680

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1746_174680


namespace NUMINAMATH_CALUDE_prime_power_form_l1746_174633

theorem prime_power_form (n : ℕ) (h : Nat.Prime (4^n + 2^n + 1)) :
  ∃ k : ℕ, n = 3^k :=
by sorry

end NUMINAMATH_CALUDE_prime_power_form_l1746_174633


namespace NUMINAMATH_CALUDE_tile_border_ratio_l1746_174613

theorem tile_border_ratio (p b : ℝ) (h_positive : p > 0 ∧ b > 0) : 
  (225 * p^2) / ((15 * p + 30 * b)^2) = 49/100 → b/p = 4/7 := by
sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l1746_174613


namespace NUMINAMATH_CALUDE_problem_solution_l1746_174665

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + 2

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2*x + k/x

theorem problem_solution :
  -- Part 1: Average rate of change
  (f 2 - f 0) / 2 = 2 ∧
  -- Part 2: Parallel tangent lines
  (∃ k : ℝ, (deriv f 1 = deriv (g k) 1) → k = 1) ∧
  -- Part 3: Tangent line equation
  (∃ a b : ℝ, (∀ x : ℝ, a*x + b = 10*x - 14) ∧
              f 2 = a*2 + b ∧
              deriv f 2 = a) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1746_174665


namespace NUMINAMATH_CALUDE_remainder_equality_l1746_174635

theorem remainder_equality (A B D S S' : ℕ) (hA : A > B) :
  A % D = S →
  B % D = S' →
  (A + B) % D = (S + S') % D :=
by sorry

end NUMINAMATH_CALUDE_remainder_equality_l1746_174635


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1746_174674

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1746_174674


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l1746_174642

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 15 → 
    (isPalindrome n 2 ∧ isPalindrome n 4) → 
    n ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l1746_174642


namespace NUMINAMATH_CALUDE_zeros_after_decimal_for_40_pow_40_l1746_174676

/-- The number of zeros immediately following the decimal point in 1/(40^40) -/
def zeros_after_decimal (n : ℕ) : ℕ :=
  let base := 40
  let exponent := 40
  let denominator := base ^ exponent
  -- The actual computation of zeros is not implemented here
  sorry

/-- Theorem stating that the number of zeros after the decimal point in 1/(40^40) is 76 -/
theorem zeros_after_decimal_for_40_pow_40 : zeros_after_decimal 40 = 76 := by
  sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_for_40_pow_40_l1746_174676


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l1746_174685

/-- The number of ways to distribute n students to k universities, 
    with each university admitting at least one student -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: Distributing 5 students to 3 universities results in 150 different methods -/
theorem distribute_five_to_three : distribute_students 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l1746_174685


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1746_174612

theorem quadratic_roots_condition (a : ℝ) (h1 : a ≠ 0) (h2 : a < -1) :
  ∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ 
  (a * x^2 + 2 * x + 1 = 0) ∧ 
  (a * y^2 + 2 * y + 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1746_174612


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l1746_174636

/-- Represents the colors of the pegs -/
inductive Color
| Red
| Blue
| Green

/-- Represents a position on the triangular board -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the triangular board -/
def Board := List Position

/-- Defines a valid triangular board with 3 rows -/
def validBoard : Board :=
  [(Position.mk 1 1),
   (Position.mk 2 1), (Position.mk 2 2),
   (Position.mk 3 1), (Position.mk 3 2), (Position.mk 3 3)]

/-- Represents a peg placement on the board -/
structure Placement :=
  (pos : Position)
  (color : Color)

/-- Checks if a list of placements is valid according to the color restriction rule -/
def isValidPlacement (placements : List Placement) : Bool :=
  sorry

/-- Counts the number of valid arrangements of pegs on the board -/
def countValidArrangements (board : Board) (redPegs bluePegs greenPegs : Nat) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem valid_arrangements_count :
  countValidArrangements validBoard 4 3 2 = 6 :=
sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l1746_174636


namespace NUMINAMATH_CALUDE_min_correct_answers_to_advance_l1746_174649

/-- Given a math competition with the following conditions:
  * There are 25 questions in total
  * Each correct answer is worth 4 points
  * Each incorrect or unanswered question results in -1 point
  * A minimum of 60 points is required to advance
  This theorem proves that the minimum number of correctly answered questions
  to advance is 17. -/
theorem min_correct_answers_to_advance (total_questions : ℕ) (correct_points : ℤ) 
  (incorrect_points : ℤ) (min_points_to_advance : ℤ) :
  total_questions = 25 →
  correct_points = 4 →
  incorrect_points = -1 →
  min_points_to_advance = 60 →
  ∃ (min_correct : ℕ), 
    min_correct = 17 ∧ 
    (min_correct : ℤ) * correct_points + (total_questions - min_correct) * incorrect_points ≥ min_points_to_advance ∧
    ∀ (x : ℕ), x < min_correct → 
      (x : ℤ) * correct_points + (total_questions - x) * incorrect_points < min_points_to_advance :=
by sorry

end NUMINAMATH_CALUDE_min_correct_answers_to_advance_l1746_174649


namespace NUMINAMATH_CALUDE_batsman_average_after_15th_innings_l1746_174645

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored : ℚ) / (stats.innings + 1 : ℚ)

/-- Theorem: Batsman's average after 15th innings -/
theorem batsman_average_after_15th_innings
  (stats : BatsmanStats)
  (h1 : stats.innings = 14)
  (h2 : newAverage stats 85 = stats.average + 3)
  : newAverage stats 85 = 43 := by
  sorry

#check batsman_average_after_15th_innings

end NUMINAMATH_CALUDE_batsman_average_after_15th_innings_l1746_174645


namespace NUMINAMATH_CALUDE_fraction_repetend_correct_l1746_174619

/-- The repetend of the decimal representation of 7/19 -/
def repetend : List Nat := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]

/-- The fraction we're considering -/
def fraction : Rat := 7 / 19

theorem fraction_repetend_correct :
  ∃ (k : Nat), fraction = (k : Rat) / 10^repetend.length + 
    (List.sum (List.zipWith (λ (d i : Nat) => d * 10^(repetend.length - 1 - i)) repetend (List.range repetend.length)) : Rat) / 
    (10^repetend.length - 1) / 19 :=
sorry

end NUMINAMATH_CALUDE_fraction_repetend_correct_l1746_174619


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1746_174695

theorem perpendicular_vectors (m : ℝ) : 
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (m, 1)
  (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 0 → m = 7 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1746_174695


namespace NUMINAMATH_CALUDE_min_value_theorem_l1746_174657

theorem min_value_theorem (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < 1) :
  a^2 / x + b^2 / (1 - x) ≥ (a + b)^2 ∧ 
  ∃ y, 0 < y ∧ y < 1 ∧ a^2 / y + b^2 / (1 - y) = (a + b)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1746_174657


namespace NUMINAMATH_CALUDE_max_sum_constrained_max_sum_constrained_attained_l1746_174617

theorem max_sum_constrained (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 := by
sorry

theorem max_sum_constrained_attained :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  16 * x * y * z = (x + y)^2 * (x + z)^2 ∧
  x + y + z = 4 := by
sorry

end NUMINAMATH_CALUDE_max_sum_constrained_max_sum_constrained_attained_l1746_174617


namespace NUMINAMATH_CALUDE_f_symmetry_solutions_l1746_174671

def f_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = x^3 + 6

theorem f_symmetry_solutions (f : ℝ → ℝ) (hf : f_condition f) :
  {x : ℝ | x ≠ 0 ∧ f x = f (-x)} = {(1/2)^(1/6), -(1/2)^(1/6)} := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_solutions_l1746_174671


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1746_174640

theorem necessary_not_sufficient_condition (a : ℝ) :
  (((a - 1) * (a - 2) = 0) → (a = 2)) ∧
  ¬(∀ a : ℝ, ((a - 1) * (a - 2) = 0) ↔ (a = 2)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1746_174640


namespace NUMINAMATH_CALUDE_operation_result_l1746_174637

theorem operation_result (c : ℚ) : 
  2 * ((3 * c + 6 - 5 * c) / 3) = -4/3 * c + 4 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l1746_174637


namespace NUMINAMATH_CALUDE_remainder_problem_l1746_174616

theorem remainder_problem (n : ℕ) : 
  (n / 44 = 432 ∧ n % 44 = 0) → n % 38 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1746_174616


namespace NUMINAMATH_CALUDE_translation_theorem_l1746_174601

/-- A translation in the complex plane that moves 1 - 3i to 5 + 2i also moves 3 - 4i to 7 + i -/
theorem translation_theorem (t : ℂ → ℂ) :
  (t (1 - 3*I) = 5 + 2*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = z + w) →
  t (3 - 4*I) = 7 + I :=
by sorry

end NUMINAMATH_CALUDE_translation_theorem_l1746_174601


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l1746_174667

theorem arithmetic_square_root_of_16 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l1746_174667


namespace NUMINAMATH_CALUDE_game_strategies_l1746_174659

/-- The game state -/
structure GameState where
  board : ℝ
  turn : ℕ

/-- The game rules -/
def valid_move (x y : ℝ) : Prop :=
  0 < y - x ∧ y - x < 1

/-- The winning condition for the first variant -/
def winning_condition_1 (s : GameState) : Prop :=
  s.board ≥ 2010

/-- The winning condition for the second variant -/
def winning_condition_2 (s : GameState) : Prop :=
  s.board ≥ 2010 ∧ s.turn ≥ 2011

/-- The losing condition for the second variant -/
def losing_condition_2 (s : GameState) : Prop :=
  s.board ≥ 2010 ∧ s.turn ≤ 2010

/-- The theorem statement -/
theorem game_strategies :
  (∃ (strategy : ℕ → ℝ → ℝ),
    (∀ (n : ℕ) (x : ℝ), valid_move x (strategy n x)) ∧
    (∀ (play : ℕ → ℝ),
      (∀ (n : ℕ), valid_move (play n) (play (n+1))) →
      ∃ (k : ℕ), winning_condition_1 ⟨play k, k⟩ ∧
        k % 2 = 0)) ∧
  (∃ (strategy : ℕ → ℝ → ℝ),
    (∀ (n : ℕ) (x : ℝ), valid_move x (strategy n x)) ∧
    (∀ (play : ℕ → ℝ),
      (∀ (n : ℕ), valid_move (play n) (play (n+1))) →
      (∃ (k : ℕ), winning_condition_2 ⟨play k, k⟩ ∧
        k % 2 = 1) ∧
      (∀ (k : ℕ), k ≤ 2010 → ¬losing_condition_2 ⟨play k, k⟩))) :=
by sorry

end NUMINAMATH_CALUDE_game_strategies_l1746_174659


namespace NUMINAMATH_CALUDE_seat_difference_is_three_l1746_174638

/-- Represents a bus with seats on left and right sides, and a back seat. -/
structure Bus where
  leftSeats : Nat
  rightSeats : Nat
  backSeatCapacity : Nat
  seatCapacity : Nat
  totalCapacity : Nat

/-- The number of fewer seats on the right side compared to the left side. -/
def seatDifference (bus : Bus) : Nat :=
  bus.leftSeats - bus.rightSeats

/-- Theorem stating the difference in seats between left and right sides. -/
theorem seat_difference_is_three :
  ∃ (bus : Bus),
    bus.leftSeats = 15 ∧
    bus.seatCapacity = 3 ∧
    bus.backSeatCapacity = 10 ∧
    bus.totalCapacity = 91 ∧
    seatDifference bus = 3 := by
  sorry

#check seat_difference_is_three

end NUMINAMATH_CALUDE_seat_difference_is_three_l1746_174638


namespace NUMINAMATH_CALUDE_vector_b_exists_l1746_174602

def a : ℝ × ℝ := (1, -2)

theorem vector_b_exists : ∃ (b : ℝ × ℝ), 
  (∃ (k : ℝ), b = k • a) ∧ 
  (‖a + b‖ < ‖a‖) ∧
  b = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_b_exists_l1746_174602


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l1746_174663

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a3 : a 3 = 7)
  (h_a5_a2 : a 5 = a 2 + 6) :
  a 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l1746_174663


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1746_174681

def vector_a : ℝ × ℝ := (-5, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_vectors_x_value :
  perpendicular vector_a (vector_b x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1746_174681


namespace NUMINAMATH_CALUDE_committee_selection_l1746_174687

theorem committee_selection (boys girls : ℕ) (h1 : boys = 21) (h2 : girls = 14) :
  (Nat.choose (boys + girls) 4) - (Nat.choose boys 4 + Nat.choose girls 4) = 45374 :=
sorry

end NUMINAMATH_CALUDE_committee_selection_l1746_174687


namespace NUMINAMATH_CALUDE_scenario1_probability_scenario2_probability_l1746_174651

-- Define the probabilities
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define the number of shots for each scenario
def shots_scenario1 : ℕ := 3
def shots_scenario2 : ℕ := 2

-- Theorem for scenario 1
theorem scenario1_probability : 
  (1 - prob_A_hit ^ shots_scenario1) = 19/27 := by sorry

-- Theorem for scenario 2
theorem scenario2_probability : 
  (Nat.choose shots_scenario2 shots_scenario2 * prob_A_hit ^ shots_scenario2) *
  (Nat.choose shots_scenario2 1 * prob_B_hit ^ 1 * (1 - prob_B_hit) ^ (shots_scenario2 - 1)) = 1/6 := by sorry

end NUMINAMATH_CALUDE_scenario1_probability_scenario2_probability_l1746_174651


namespace NUMINAMATH_CALUDE_total_cleaning_time_is_180_l1746_174690

/-- The total time Matt and Alex spend cleaning their cars -/
def total_cleaning_time (matt_outside : ℕ) : ℕ :=
  let matt_inside := matt_outside / 4
  let matt_total := matt_outside + matt_inside
  let alex_outside := matt_outside / 2
  let alex_inside := matt_inside * 2
  let alex_total := alex_outside + alex_inside
  matt_total + alex_total

/-- Theorem stating that the total cleaning time is 180 minutes -/
theorem total_cleaning_time_is_180 :
  total_cleaning_time 80 = 180 := by sorry

end NUMINAMATH_CALUDE_total_cleaning_time_is_180_l1746_174690


namespace NUMINAMATH_CALUDE_selection_methods_count_l1746_174652

def num_type_a : ℕ := 3
def num_type_b : ℕ := 4
def total_selected : ℕ := 3

theorem selection_methods_count :
  (Finset.sum (Finset.range (total_selected + 1)) (λ k =>
    if k ≥ 1 ∧ (total_selected - k) ≥ 1 then
      (Nat.choose num_type_a k) * (Nat.choose num_type_b (total_selected - k))
    else
      0
  )) = 30 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l1746_174652


namespace NUMINAMATH_CALUDE_trig_expression_equals_sqrt_three_l1746_174669

theorem trig_expression_equals_sqrt_three : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_sqrt_three_l1746_174669


namespace NUMINAMATH_CALUDE_rice_purchase_l1746_174607

theorem rice_purchase (rice_price lentil_price total_weight total_cost : ℚ)
  (h1 : rice_price = 105/100)
  (h2 : lentil_price = 33/100)
  (h3 : total_weight = 30)
  (h4 : total_cost = 2340/100) :
  ∃ (rice_weight : ℚ),
    rice_weight + (total_weight - rice_weight) = total_weight ∧
    rice_price * rice_weight + lentil_price * (total_weight - rice_weight) = total_cost ∧
    rice_weight = 75/4 := by
  sorry

end NUMINAMATH_CALUDE_rice_purchase_l1746_174607


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1746_174675

/-- Proves that a train of given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 215) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1746_174675


namespace NUMINAMATH_CALUDE_jamie_flyer_earnings_l1746_174666

/-- Calculates Jamie's earnings from delivering flyers --/
def jamies_earnings (hourly_rate : ℕ) (days_per_week : ℕ) (hours_per_day : ℕ) (total_weeks : ℕ) : ℕ :=
  hourly_rate * days_per_week * hours_per_day * total_weeks

/-- Proves that Jamie's earnings after 6 weeks will be $360 --/
theorem jamie_flyer_earnings :
  jamies_earnings 10 2 3 6 = 360 := by
  sorry

#eval jamies_earnings 10 2 3 6

end NUMINAMATH_CALUDE_jamie_flyer_earnings_l1746_174666


namespace NUMINAMATH_CALUDE_fraction_over_65_l1746_174628

theorem fraction_over_65 (total : ℕ) (under_21 : ℕ) (over_65 : ℕ) : 
  (3 : ℚ) / 7 * total = under_21 →
  50 < total →
  total < 100 →
  under_21 = 33 →
  (over_65 : ℚ) / total = over_65 / 77 :=
by sorry

end NUMINAMATH_CALUDE_fraction_over_65_l1746_174628


namespace NUMINAMATH_CALUDE_sphere_properties_l1746_174650

/-- Given a sphere with volume 288π cubic inches, prove its surface area is 144π square inches and its diameter is 12 inches -/
theorem sphere_properties (r : ℝ) (h : (4/3) * Real.pi * r^3 = 288 * Real.pi) :
  (4 * Real.pi * r^2 = 144 * Real.pi) ∧ (2 * r = 12) := by
  sorry

end NUMINAMATH_CALUDE_sphere_properties_l1746_174650


namespace NUMINAMATH_CALUDE_angle_a1fb1_is_right_angle_l1746_174698

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Theorem: Angle A1FB1 is 90 degrees in a parabola -/
theorem angle_a1fb1_is_right_angle (parab : Parabola) 
  (focus : Point) 
  (directrix : ℝ) 
  (line : Line) 
  (a b : Point) 
  (a1 b1 : Point) :
  focus.x = parab.p / 2 →
  focus.y = 0 →
  directrix = -parab.p / 2 →
  parab.equation a.x a.y →
  parab.equation b.x b.y →
  line.p1 = focus →
  (line.p2 = a ∨ line.p2 = b) →
  a1.x = directrix →
  b1.x = directrix →
  a1.y = a.y →
  b1.y = b.y →
  -- The conclusion: ∠A1FB1 = 90°
  ∃ (angle : ℝ), angle = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_a1fb1_is_right_angle_l1746_174698


namespace NUMINAMATH_CALUDE_chromatic_number_iff_k_constructible_l1746_174677

/-- A graph is k-constructible if it can be built up from K_k by repeatedly adding a new vertex
    and joining it to a k-clique in the existing graph. -/
def is_k_constructible (G : SimpleGraph V) (k : ℕ) : Prop :=
  sorry

theorem chromatic_number_iff_k_constructible (G : SimpleGraph V) (k : ℕ) :
  G.chromaticNumber ≥ k ↔ ∃ H : SimpleGraph V, H ≤ G ∧ is_k_constructible H k :=
sorry

end NUMINAMATH_CALUDE_chromatic_number_iff_k_constructible_l1746_174677


namespace NUMINAMATH_CALUDE_jenny_egg_distribution_l1746_174627

theorem jenny_egg_distribution (n : ℕ) : 
  n ∣ 18 ∧ n ∣ 24 ∧ n ≥ 4 → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_jenny_egg_distribution_l1746_174627


namespace NUMINAMATH_CALUDE_parking_lot_perimeter_l1746_174615

theorem parking_lot_perimeter 
  (d : ℝ) (A : ℝ) (x y : ℝ) (P : ℝ) 
  (h1 : d = 20) 
  (h2 : A = 120) 
  (h3 : x = (2/3) * y) 
  (h4 : x^2 + y^2 = d^2) 
  (h5 : x * y = A) 
  (h6 : P = 2 * (x + y)) : 
  P = 20 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_perimeter_l1746_174615


namespace NUMINAMATH_CALUDE_jellybean_theorem_l1746_174632

def jellybean_problem (initial : ℕ) (samantha_took : ℕ) (shelby_ate : ℕ) : ℕ :=
  let remaining_after_samantha := initial - samantha_took
  let remaining_after_shelby := remaining_after_samantha - shelby_ate
  let total_removed := samantha_took + shelby_ate
  let shannon_added := total_removed / 2
  remaining_after_shelby + shannon_added

theorem jellybean_theorem :
  jellybean_problem 90 24 12 = 72 := by
  sorry

#eval jellybean_problem 90 24 12

end NUMINAMATH_CALUDE_jellybean_theorem_l1746_174632


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l1746_174646

/-- The function f(x) = x³ + ax² + 3x - 9 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

/-- The derivative of f(x) --/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l1746_174646


namespace NUMINAMATH_CALUDE_katherines_bananas_l1746_174684

theorem katherines_bananas (apples pears bananas total : ℕ) : 
  apples = 4 →
  pears = 3 * apples →
  total = 21 →
  total = apples + pears + bananas →
  bananas = 5 := by
sorry

end NUMINAMATH_CALUDE_katherines_bananas_l1746_174684


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1746_174673

theorem system_solution_ratio (a b x y : ℝ) (h1 : b ≠ 0) (h2 : 4*x - y = a) (h3 : 5*y - 20*x = b) : a / b = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1746_174673


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1746_174643

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * (x - 1)^3 = 24
def equation2 (x : ℝ) : Prop := (x - 3)^2 = 64

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 3 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x₁ x₂ : ℝ, equation2 x₁ ∧ equation2 x₂ ∧ x₁ = 11 ∧ x₂ = -5 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1746_174643


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l1746_174692

/-- The number of ways to distribute n identical balls into k distinct boxes, leaving no box empty -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- Theorem stating that there are 20 ways to distribute 7 identical balls into 4 distinct boxes with no empty box -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l1746_174692


namespace NUMINAMATH_CALUDE_bargaining_range_l1746_174631

def marked_price : ℝ := 100

def min_markup_percent : ℝ := 50
def max_markup_percent : ℝ := 100

def min_profit_percent : ℝ := 20

def lower_bound : ℝ := 60
def upper_bound : ℝ := 80

theorem bargaining_range :
  ∀ (cost_price : ℝ),
    (cost_price * (1 + min_markup_percent / 100) ≤ marked_price) →
    (cost_price * (1 + max_markup_percent / 100) ≥ marked_price) →
    (lower_bound ≥ cost_price * (1 + min_profit_percent / 100)) ∧
    (upper_bound ≤ marked_price) ∧
    (lower_bound ≤ upper_bound) :=
by sorry

end NUMINAMATH_CALUDE_bargaining_range_l1746_174631


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1746_174683

def polynomial (x : ℝ) : ℝ := 5 * (x^4 + 2*x^3 + 4*x^2 + 3)

theorem sum_of_squared_coefficients : 
  (5^2) + (10^2) + (20^2) + (0^2) + (15^2) = 750 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1746_174683


namespace NUMINAMATH_CALUDE_star_properties_l1746_174624

-- Define the * operation
def star (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

-- State the theorem
theorem star_properties :
  ∀ x y : ℝ,
  (star x y = star y x) ∧
  (star (x + 1) (x - 1) = x * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l1746_174624


namespace NUMINAMATH_CALUDE_max_value_constrained_sum_l1746_174672

theorem max_value_constrained_sum (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → 2*x + y + 2*z ≤ 2*a + b + 2*c) →
  2*a + b + 2*c = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constrained_sum_l1746_174672


namespace NUMINAMATH_CALUDE_seating_probability_l1746_174688

-- Define the number of boys in the class
def num_boys : ℕ := 9

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the derangement function for 4 elements
def derangement_4 : ℕ := 9

-- Define the probability we want to prove
def target_probability : ℚ := 1 / 32

-- Theorem statement
theorem seating_probability :
  (choose num_boys 3 * choose (num_boys - 3) 2 * derangement_4) / (Nat.factorial num_boys) = target_probability := by
  sorry

end NUMINAMATH_CALUDE_seating_probability_l1746_174688


namespace NUMINAMATH_CALUDE_total_spent_on_car_parts_l1746_174660

def speakers : ℚ := 235.87
def newTires : ℚ := 281.45
def steeringWheelCover : ℚ := 179.99
def seatCovers : ℚ := 122.31
def headlights : ℚ := 98.63

theorem total_spent_on_car_parts : 
  speakers + newTires + steeringWheelCover + seatCovers + headlights = 918.25 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_car_parts_l1746_174660


namespace NUMINAMATH_CALUDE_perimeter_of_square_figure_l1746_174664

/-- A figure composed of four identical squares -/
structure SquareFigure where
  -- Side length of each square
  side_length : ℝ
  -- Total area of the figure
  total_area : ℝ
  -- Number of vertical segments
  vertical_segments : ℕ
  -- Number of horizontal segments
  horizontal_segments : ℕ
  -- Condition: Total area is the area of four squares
  area_condition : total_area = 4 * side_length ^ 2

/-- The perimeter of the square figure -/
def perimeter (f : SquareFigure) : ℝ :=
  (f.vertical_segments + f.horizontal_segments) * f.side_length

/-- Theorem: If the total area is 144 cm² and the figure has 4 vertical and 6 horizontal segments,
    then the perimeter is 60 cm -/
theorem perimeter_of_square_figure (f : SquareFigure) 
    (h_area : f.total_area = 144) 
    (h_vertical : f.vertical_segments = 4) 
    (h_horizontal : f.horizontal_segments = 6) : 
    perimeter f = 60 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_square_figure_l1746_174664


namespace NUMINAMATH_CALUDE_line_equation_point_slope_l1746_174662

/-- A line passing through point (-1, 1) with slope 2 has the equation y = 2x + 3 -/
theorem line_equation_point_slope : 
  ∀ (x y : ℝ), y = 2*x + 3 ↔ (y - 1 = 2*(x - (-1)) ∧ (x, y) ≠ (-1, 1)) ∨ (x, y) = (-1, 1) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_point_slope_l1746_174662


namespace NUMINAMATH_CALUDE_electric_blankets_sold_l1746_174694

/-- Represents the number of electric blankets sold -/
def electric_blankets : ℕ := sorry

/-- Represents the number of hot-water bottles sold -/
def hot_water_bottles : ℕ := sorry

/-- Represents the number of thermometers sold -/
def thermometers : ℕ := sorry

/-- The price of a thermometer in dollars -/
def thermometer_price : ℕ := 2

/-- The price of a hot-water bottle in dollars -/
def hot_water_bottle_price : ℕ := 6

/-- The price of an electric blanket in dollars -/
def electric_blanket_price : ℕ := 10

/-- The total sales for all items in dollars -/
def total_sales : ℕ := 1800

theorem electric_blankets_sold :
  (thermometer_price * thermometers + 
   hot_water_bottle_price * hot_water_bottles + 
   electric_blanket_price * electric_blankets = total_sales) ∧
  (thermometers = 7 * hot_water_bottles) ∧
  (hot_water_bottles = 2 * electric_blankets) →
  electric_blankets = 36 := by sorry

end NUMINAMATH_CALUDE_electric_blankets_sold_l1746_174694


namespace NUMINAMATH_CALUDE_power_of_negative_one_2010_l1746_174691

theorem power_of_negative_one_2010 : ∃ x : ℕ, ((-1 : ℤ) ^ 2010 : ℤ) = x ∧ ∀ y : ℕ, y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_power_of_negative_one_2010_l1746_174691


namespace NUMINAMATH_CALUDE_fraction_relation_l1746_174653

theorem fraction_relation (x y z : ℚ) 
  (h1 : x / y = 3)
  (h2 : y / z = 5 / 2) :
  z / x = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relation_l1746_174653


namespace NUMINAMATH_CALUDE_real_part_of_reciprocal_l1746_174689

theorem real_part_of_reciprocal (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 1) :
  (1 / (z - Complex.I)).re = z.re / (2 - 2 * z.im) := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_reciprocal_l1746_174689


namespace NUMINAMATH_CALUDE_quadratic_radical_equivalence_l1746_174641

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem quadratic_radical_equivalence (m : ℕ) :
  (is_prime 2 ∧ is_prime (2023 - m)) → m = 2021 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equivalence_l1746_174641


namespace NUMINAMATH_CALUDE_min_fence_length_l1746_174654

theorem min_fence_length (w : ℝ) (l : ℝ) (area : ℝ) (perimeter : ℝ) : 
  w > 0 →
  l = 2 * w →
  area = l * w →
  area ≥ 500 →
  perimeter = 2 * (l + w) →
  perimeter ≥ 96 :=
by sorry

end NUMINAMATH_CALUDE_min_fence_length_l1746_174654


namespace NUMINAMATH_CALUDE_eleventh_flip_probability_l1746_174648

def is_fair_coin (coin : Type) : Prop := sorry

def probability_of_tails (coin : Type) : ℚ := sorry

def previous_flips_heads (coin : Type) (n : ℕ) : Prop := sorry

theorem eleventh_flip_probability (coin : Type) 
  (h_fair : is_fair_coin coin)
  (h_previous : previous_flips_heads coin 10) :
  probability_of_tails coin = 1/2 := by sorry

end NUMINAMATH_CALUDE_eleventh_flip_probability_l1746_174648


namespace NUMINAMATH_CALUDE_cost_of_graveling_specific_lawn_l1746_174686

/-- Calculates the cost of graveling two intersecting roads on a rectangular lawn. -/
def cost_of_graveling (lawn_length lawn_width road_width gravel_cost : ℝ) : ℝ :=
  let road_length_area := lawn_length * road_width
  let road_width_area := (lawn_width - road_width) * road_width
  let total_area := road_length_area + road_width_area
  total_area * gravel_cost

/-- The cost of graveling two intersecting roads on a 70m × 60m lawn with 10m wide roads at Rs. 3 per sq m is Rs. 3600. -/
theorem cost_of_graveling_specific_lawn :
  cost_of_graveling 70 60 10 3 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_graveling_specific_lawn_l1746_174686


namespace NUMINAMATH_CALUDE_triangle_properties_l1746_174600

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = Real.pi ∧
  (Real.cos A - 2 * Real.cos C) / Real.cos B = (2 * c - a) / b ∧
  Real.cos B = 1/4 ∧
  1/2 * a * c * Real.sin B = Real.sqrt 15 / 4 →
  Real.sin C / Real.sin A = 2 ∧
  a + b + c = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1746_174600


namespace NUMINAMATH_CALUDE_inequality_solution_implies_n_range_l1746_174623

theorem inequality_solution_implies_n_range (n : ℝ) : 
  (∀ x : ℝ, ((n - 3) * x > 2) ↔ (x < 2 / (n - 3))) → n < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_n_range_l1746_174623


namespace NUMINAMATH_CALUDE_max_value_of_a_l1746_174626

theorem max_value_of_a (a b c : ℝ) (sum_zero : a + b + c = 0) (sum_squares_six : a^2 + b^2 + c^2 = 6) :
  ∃ (max_a : ℝ), max_a = 2 ∧ ∀ x, (∃ y z, x + y + z = 0 ∧ x^2 + y^2 + z^2 = 6) → x ≤ max_a :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1746_174626


namespace NUMINAMATH_CALUDE_linda_classmates_l1746_174614

/-- The number of cookies each student receives -/
def cookies_per_student : ℕ := 10

/-- The number of cookies in one dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of dozens of cookies in each batch -/
def dozens_per_batch : ℕ := 4

/-- The number of batches of chocolate chip cookies Linda made -/
def chocolate_chip_batches : ℕ := 2

/-- The number of batches of oatmeal raisin cookies Linda made -/
def oatmeal_raisin_batches : ℕ := 1

/-- The number of additional batches Linda needs to bake -/
def additional_batches : ℕ := 2

/-- The total number of cookies Linda will have after baking all batches -/
def total_cookies : ℕ := 
  (chocolate_chip_batches + oatmeal_raisin_batches + additional_batches) * 
  dozens_per_batch * cookies_per_dozen

/-- The number of Linda's classmates -/
def number_of_classmates : ℕ := total_cookies / cookies_per_student

theorem linda_classmates : number_of_classmates = 24 := by
  sorry

end NUMINAMATH_CALUDE_linda_classmates_l1746_174614


namespace NUMINAMATH_CALUDE_binomial_cube_special_case_l1746_174634

theorem binomial_cube_special_case : 8^3 + 3*(8^2) + 3*8 + 1 = 729 := by
  sorry

end NUMINAMATH_CALUDE_binomial_cube_special_case_l1746_174634


namespace NUMINAMATH_CALUDE_inequality_range_l1746_174644

theorem inequality_range : 
  {a : ℝ | ∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a} = {a : ℝ | -1 ≤ a ∧ a ≤ 4} := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1746_174644


namespace NUMINAMATH_CALUDE_crayons_given_to_friends_l1746_174611

theorem crayons_given_to_friends (initial : ℕ) (lost : ℕ) (remaining : ℕ) 
  (h1 : initial = 440)
  (h2 : lost = 106)
  (h3 : remaining = 223) :
  initial - lost - remaining = 111 := by
  sorry

end NUMINAMATH_CALUDE_crayons_given_to_friends_l1746_174611


namespace NUMINAMATH_CALUDE_union_of_sets_l1746_174697

theorem union_of_sets : 
  let A : Set Nat := {1, 2, 4}
  let B : Set Nat := {2, 6}
  A ∪ B = {1, 2, 4, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1746_174697


namespace NUMINAMATH_CALUDE_min_product_of_three_min_product_is_neg_480_l1746_174696

def S : Finset Int := {-10, -7, -3, 1, 4, 6, 8}

theorem min_product_of_three (a b c : Int) : 
  a ∈ S → b ∈ S → c ∈ S → 
  a ≠ b → b ≠ c → a ≠ c →
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y → y ≠ z → x ≠ z →
  a * b * c ≤ x * y * z :=
by
  sorry

theorem min_product_is_neg_480 : 
  ∃ a b c : Int, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a * b * c = -480 ∧
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y → y ≠ z → x ≠ z →
   a * b * c ≤ x * y * z) :=
by
  sorry

end NUMINAMATH_CALUDE_min_product_of_three_min_product_is_neg_480_l1746_174696


namespace NUMINAMATH_CALUDE_right_triangle_7_24_25_l1746_174606

theorem right_triangle_7_24_25 (a b c : ℝ) :
  a = 7 ∧ b = 24 ∧ c = 25 → a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_7_24_25_l1746_174606


namespace NUMINAMATH_CALUDE_hallway_tiles_l1746_174622

/-- Calculates the total number of tiles used in a rectangular hallway with specific tiling patterns. -/
def total_tiles (length width : ℕ) : ℕ :=
  let outer_border := 2 * (length - 2) + 2 * (width - 2) + 4
  let second_border := 2 * ((length - 4) / 2) + 2 * ((width - 4) / 2)
  let inner_area := ((length - 6) * (width - 6)) / 9
  outer_border + second_border + inner_area

/-- Theorem stating that the total number of tiles used in a 20x30 foot rectangular hallway
    with specific tiling patterns is 175. -/
theorem hallway_tiles : total_tiles 30 20 = 175 := by
  sorry

end NUMINAMATH_CALUDE_hallway_tiles_l1746_174622


namespace NUMINAMATH_CALUDE_pitcher_distribution_l1746_174630

theorem pitcher_distribution (C : ℝ) (h : C > 0) : 
  let juice_amount : ℝ := (2/3) * C
  let cups : ℕ := 6
  let juice_per_cup : ℝ := juice_amount / cups
  juice_per_cup / C = 1/9 := by sorry

end NUMINAMATH_CALUDE_pitcher_distribution_l1746_174630


namespace NUMINAMATH_CALUDE_pencils_remaining_l1746_174658

theorem pencils_remaining (initial_pencils : ℕ) (pencils_removed : ℕ) : 
  initial_pencils = 9 → pencils_removed = 4 → initial_pencils - pencils_removed = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencils_remaining_l1746_174658


namespace NUMINAMATH_CALUDE_gadget_price_proof_l1746_174647

theorem gadget_price_proof (sticker_price : ℝ) : 
  (0.80 * sticker_price - 80) = (0.65 * sticker_price - 20) → sticker_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_gadget_price_proof_l1746_174647


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1746_174682

/-- Given a quadrilateral ABCD with extended sides, prove that A can be expressed
    as a linear combination of A'', B'', C'', D'' with specific coefficients. -/
theorem quadrilateral_reconstruction
  (A B C D A'' B'' C'' D'' : ℝ × ℝ) -- Points in 2D space
  (h1 : A'' - A = 2 * (B - A))      -- AA'' = 2AB
  (h2 : B'' - B = 3 * (C - B))      -- BB'' = 3BC
  (h3 : C'' - C = 2 * (D - C))      -- CC'' = 2CD
  (h4 : D'' - D = 2 * (A - D)) :    -- DD'' = 2DA
  A = (1/6 : ℝ) • A'' + (1/9 : ℝ) • B'' + (1/9 : ℝ) • C'' + (1/18 : ℝ) • D'' := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1746_174682


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1746_174655

theorem inequality_solution_set (x : ℝ) : (3 + x) * (2 - x) < 0 ↔ x > 2 ∨ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1746_174655


namespace NUMINAMATH_CALUDE_bob_weight_is_165_l1746_174609

def jim_weight : ℝ := sorry
def bob_weight : ℝ := sorry

axiom combined_weight : jim_weight + bob_weight = 220
axiom weight_relation : bob_weight - 2 * jim_weight = bob_weight / 3

theorem bob_weight_is_165 : bob_weight = 165 := by sorry

end NUMINAMATH_CALUDE_bob_weight_is_165_l1746_174609


namespace NUMINAMATH_CALUDE_collinear_points_sum_l1746_174603

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

/-- The main theorem -/
theorem collinear_points_sum (a b : ℝ) : 
  collinear (Point3D.mk 1 a b) (Point3D.mk a 2 3) (Point3D.mk a b 3) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l1746_174603


namespace NUMINAMATH_CALUDE_feb_29_is_sunday_l1746_174661

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February of a leap year -/
structure FebruaryDate :=
  (day : Nat)
  (isLeapYear : Bool)

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Advances the day of the week by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Main theorem: If February 11th is a Wednesday in a leap year, then February 29th is a Sunday -/
theorem feb_29_is_sunday (d : FebruaryDate) (dow : DayOfWeek) :
  d.day = 11 → d.isLeapYear = true → dow = DayOfWeek.Wednesday →
  advanceDays dow 18 = DayOfWeek.Sunday :=
by
  sorry


end NUMINAMATH_CALUDE_feb_29_is_sunday_l1746_174661


namespace NUMINAMATH_CALUDE_parabola_properties_l1746_174625

/-- Parabola C with vertex at origin and focus on y-axis -/
structure Parabola where
  focus : ℝ
  equation : ℝ → ℝ → Prop

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : C.equation x y

/-- Line segment on the parabola -/
structure LineSegmentOnParabola (C : Parabola) where
  A : PointOnParabola C
  B : PointOnParabola C

/-- Triangle on the parabola -/
structure TriangleOnParabola (C : Parabola) where
  A : PointOnParabola C
  B : PointOnParabola C
  D : PointOnParabola C

theorem parabola_properties (C : Parabola) (Q : PointOnParabola C) 
    (AB : LineSegmentOnParabola C) (M : ℝ) (ABD : TriangleOnParabola C) :
  Q.x = Real.sqrt 8 ∧ Q.y = 2 ∧ (Q.x - C.focus)^2 + Q.y^2 = 9 →
  (∃ (m : ℝ), m > 0 ∧ 
    (∃ (k : ℝ), AB.A.y = k * AB.A.x + m ∧ AB.B.y = k * AB.B.x + m) ∧
    AB.A.x * AB.B.x + AB.A.y * AB.B.y = 0) →
  ABD.D.x < ABD.A.x ∧ ABD.A.x < ABD.B.x ∧
  (ABD.B.x - ABD.A.x)^2 + (ABD.B.y - ABD.A.y)^2 = 
    (ABD.D.x - ABD.A.x)^2 + (ABD.D.y - ABD.A.y)^2 ∧
  (ABD.B.x - ABD.A.x) * (ABD.D.x - ABD.A.x) + 
    (ABD.B.y - ABD.A.y) * (ABD.D.y - ABD.A.y) = 0 →
  C.equation = (fun x y => x^2 = 4*y) ∧ 
  M = 4 ∧
  (∀ (ABD' : TriangleOnParabola C), 
    ABD'.D.x < ABD'.A.x ∧ ABD'.A.x < ABD'.B.x ∧
    (ABD'.B.x - ABD'.A.x)^2 + (ABD'.B.y - ABD'.A.y)^2 = 
      (ABD'.D.x - ABD'.A.x)^2 + (ABD'.D.y - ABD'.A.y)^2 ∧
    (ABD'.B.x - ABD'.A.x) * (ABD'.D.x - ABD'.A.x) + 
      (ABD'.B.y - ABD'.A.y) * (ABD'.D.y - ABD'.A.y) = 0 →
    (ABD'.B.x - ABD'.A.x) * (ABD'.D.y - ABD'.A.y) -
      (ABD'.B.y - ABD'.A.y) * (ABD'.D.x - ABD'.A.x) ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1746_174625


namespace NUMINAMATH_CALUDE_pages_per_day_l1746_174610

theorem pages_per_day (pages_per_book : ℕ) (days_per_book : ℕ) (h1 : pages_per_book = 249) (h2 : days_per_book = 3) :
  pages_per_book / days_per_book = 83 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l1746_174610


namespace NUMINAMATH_CALUDE_incircle_circumcircle_ratio_bound_incircle_circumcircle_ratio_bound_tight_l1746_174699

/-- The ratio of the incircle radius to the circumcircle radius of a right triangle is at most √2 - 1 -/
theorem incircle_circumcircle_ratio_bound (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  (a + b - c) / c ≤ Real.sqrt 2 - 1 :=
sorry

/-- The upper bound √2 - 1 is achievable for the ratio of incircle to circumcircle radius in a right triangle -/
theorem incircle_circumcircle_ratio_bound_tight :
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b - c) / c = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_incircle_circumcircle_ratio_bound_incircle_circumcircle_ratio_bound_tight_l1746_174699
