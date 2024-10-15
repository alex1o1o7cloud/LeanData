import Mathlib

namespace NUMINAMATH_CALUDE_angle_A_measure_perimeter_range_l2530_253055

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition given in the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 - 2*t.b*t.c*(Real.cos t.A) = (t.b + t.c)^2

/-- Theorem stating the measure of angle A -/
theorem angle_A_measure (t : Triangle) (h : satisfiesCondition t) : t.A = 2*π/3 := by
  sorry

/-- Theorem stating the range of the perimeter when a = 3 -/
theorem perimeter_range (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.a = 3) :
  6 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 2*Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_measure_perimeter_range_l2530_253055


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l2530_253073

theorem unequal_gender_probability :
  let n : ℕ := 12  -- Total number of children
  let p : ℚ := 1/2  -- Probability of each child being a boy (or girl)
  let total_outcomes : ℕ := 2^n  -- Total number of possible outcomes
  let equal_outcomes : ℕ := n.choose (n/2)  -- Number of outcomes with equal boys and girls
  (1 : ℚ) - (equal_outcomes : ℚ) / total_outcomes = 793/1024 :=
by sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l2530_253073


namespace NUMINAMATH_CALUDE_prob_divisible_by_eight_l2530_253036

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def die_sides : ℕ := 6

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 1 / 2

/-- The probability of rolling a 2 on a single die -/
def prob_two : ℚ := 1 / 6

/-- The probability of rolling a 4 on a single die -/
def prob_four : ℚ := 1 / 6

/-- The probability that the product of the rolls is divisible by 8 -/
theorem prob_divisible_by_eight : 
  (1 : ℚ) - (prob_odd ^ num_dice + 
    (num_dice.choose 1 : ℚ) * prob_two * prob_odd ^ (num_dice - 1) +
    (num_dice.choose 2 : ℚ) * prob_two ^ 2 * prob_odd ^ (num_dice - 2) +
    (num_dice.choose 1 : ℚ) * prob_four * prob_odd ^ (num_dice - 1)) = 65 / 72 := by
  sorry


end NUMINAMATH_CALUDE_prob_divisible_by_eight_l2530_253036


namespace NUMINAMATH_CALUDE_joint_club_afternoon_solution_l2530_253051

/-- Represents the joint club afternoon scenario with two classes -/
structure JointClubAfternoon where
  a : ℕ  -- number of students in class A
  b : ℕ  -- number of students in class B
  K : ℕ  -- the amount each student would pay if one class covered all costs

/-- Conditions for the joint club afternoon -/
def scenario (j : JointClubAfternoon) : Prop :=
  -- Total contribution for the first event
  5 * j.a + 3 * j.b = j.K * j.a
  ∧
  -- Total contribution for the second event
  4 * j.a + 6 * j.b = j.K * j.b
  ∧
  -- Class B has more students than class A
  j.b > j.a

/-- Theorem stating the solution to the problem -/
theorem joint_club_afternoon_solution (j : JointClubAfternoon) 
  (h : scenario j) : j.K = 9 ∧ j.b > j.a := by
  sorry


end NUMINAMATH_CALUDE_joint_club_afternoon_solution_l2530_253051


namespace NUMINAMATH_CALUDE_henry_lap_time_l2530_253005

theorem henry_lap_time (margo_lap_time henry_lap_time meet_time : ℕ) 
  (h1 : margo_lap_time = 12)
  (h2 : meet_time = 84)
  (h3 : meet_time % margo_lap_time = 0)
  (h4 : meet_time % henry_lap_time = 0)
  (h5 : henry_lap_time < margo_lap_time)
  : henry_lap_time = 7 :=
sorry

end NUMINAMATH_CALUDE_henry_lap_time_l2530_253005


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2530_253084

theorem diophantine_equation_solution (x y : ℕ) (h : 65 * x - 43 * y = 2) :
  ∃ t : ℤ, t ≤ 0 ∧ x = 4 - 43 * t ∧ y = 6 - 65 * t := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2530_253084


namespace NUMINAMATH_CALUDE_point_coordinates_l2530_253011

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance of a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (p : Point) 
  (h1 : SecondQuadrant p) 
  (h2 : DistanceToXAxis p = 3) 
  (h3 : DistanceToYAxis p = 1) : 
  p = Point.mk (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2530_253011


namespace NUMINAMATH_CALUDE_base4_77_last_digit_l2530_253038

def base4LastDigit (n : Nat) : Nat :=
  n % 4

theorem base4_77_last_digit :
  base4LastDigit 77 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base4_77_last_digit_l2530_253038


namespace NUMINAMATH_CALUDE_ratio_of_partial_fractions_l2530_253043

theorem ratio_of_partial_fractions (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 →
    P / (x + 6) + Q / (x^2 - 5*x) = (x^2 - 3*x + 15) / (x^3 + x^2 - 30*x)) →
  (Q : ℚ) / (P : ℚ) = 15 / 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_partial_fractions_l2530_253043


namespace NUMINAMATH_CALUDE_safe_flight_probability_l2530_253001

/-- Represents a rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.length * prism.width * prism.height

/-- Represents the problem setup -/
def problem_setup : Prop :=
  let outer_prism : RectangularPrism := { length := 5, width := 4, height := 3 }
  let inner_prism : RectangularPrism := { length := 3, width := 2, height := 1 }
  let outer_volume := volume outer_prism
  let inner_volume := volume inner_prism
  (inner_volume / outer_volume) = (1 : ℝ) / 10

/-- The main theorem to prove -/
theorem safe_flight_probability : problem_setup := by
  sorry

end NUMINAMATH_CALUDE_safe_flight_probability_l2530_253001


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l2530_253046

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 2) : 
  |x| + |y| ≤ 2 ∧ ∃ (a b : ℝ), a^2 + b^2 = 2 ∧ |a| + |b| = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l2530_253046


namespace NUMINAMATH_CALUDE_apple_price_l2530_253006

/-- The price of apples given Emmy's and Gerry's money and the total number of apples they can buy -/
theorem apple_price (emmy_money : ℝ) (gerry_money : ℝ) (total_apples : ℝ) 
  (h1 : emmy_money = 200)
  (h2 : gerry_money = 100)
  (h3 : total_apples = 150) :
  (emmy_money + gerry_money) / total_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_l2530_253006


namespace NUMINAMATH_CALUDE_paiges_files_l2530_253010

theorem paiges_files (deleted_files : ℕ) (files_per_folder : ℕ) (num_folders : ℕ) :
  deleted_files = 9 →
  files_per_folder = 6 →
  num_folders = 3 →
  deleted_files + (files_per_folder * num_folders) = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_paiges_files_l2530_253010


namespace NUMINAMATH_CALUDE_count_divisible_numbers_l2530_253058

theorem count_divisible_numbers : 
  (Finset.filter 
    (fun k : ℕ => k ≤ 267000 ∧ (k^2 - 1) % 267 = 0) 
    (Finset.range 267001)).card = 4000 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_l2530_253058


namespace NUMINAMATH_CALUDE_solve_bowling_problem_l2530_253023

def bowling_problem (score1 score2 average : ℕ) : Prop :=
  ∃ score3 : ℕ, 
    (score1 + score2 + score3) / 3 = average ∧
    score3 = 3 * average - score1 - score2

theorem solve_bowling_problem : 
  bowling_problem 113 85 106 → ∃ score3 : ℕ, score3 = 120 := by
  sorry

#check solve_bowling_problem

end NUMINAMATH_CALUDE_solve_bowling_problem_l2530_253023


namespace NUMINAMATH_CALUDE_museum_ticket_cost_l2530_253003

/-- The cost of tickets at a museum --/
theorem museum_ticket_cost (adult_price : ℝ) : 
  (7 * adult_price + 5 * (adult_price / 2) = 35) →
  (10 * adult_price + 8 * (adult_price / 2) = 51.58) := by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l2530_253003


namespace NUMINAMATH_CALUDE_building_cost_l2530_253064

/-- Calculates the total cost of all units in a building -/
def total_cost (total_units : ℕ) (cost_1bed : ℕ) (cost_2bed : ℕ) (num_2bed : ℕ) : ℕ :=
  let num_1bed := total_units - num_2bed
  num_1bed * cost_1bed + num_2bed * cost_2bed

/-- Proves that the total cost of all units in the given building is 4950 dollars -/
theorem building_cost : total_cost 12 360 450 7 = 4950 := by
  sorry

end NUMINAMATH_CALUDE_building_cost_l2530_253064


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_prism_height_isosceles_trapezoid_prism_height_proof_l2530_253022

/-- Represents a prism with a base that is an isosceles trapezoid inscribed around a circle -/
structure IsoscelesTrapezoidPrism where
  r : ℝ  -- radius of the inscribed circle
  α : ℝ  -- acute angle of the trapezoid

/-- 
Theorem: The height of the prism is 2r tan(α) given:
- The base is an isosceles trapezoid inscribed around a circle with radius r
- The acute angle of the trapezoid is α
- A plane passing through one side of the base and the acute angle endpoint 
  of the opposite side of the top plane forms an angle α with the base plane
-/
theorem isosceles_trapezoid_prism_height 
  (prism : IsoscelesTrapezoidPrism) : ℝ :=
  2 * prism.r * Real.tan prism.α

-- Proof
theorem isosceles_trapezoid_prism_height_proof
  (prism : IsoscelesTrapezoidPrism) :
  isosceles_trapezoid_prism_height prism = 2 * prism.r * Real.tan prism.α := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_prism_height_isosceles_trapezoid_prism_height_proof_l2530_253022


namespace NUMINAMATH_CALUDE_intersection_distance_l2530_253093

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The start point of the line -/
def startPoint : Point3D := ⟨1, 2, 3⟩

/-- The end point of the line -/
def endPoint : Point3D := ⟨3, 6, 7⟩

/-- The center of the unit sphere -/
def sphereCenter : Point3D := ⟨0, 0, 0⟩

/-- The radius of the unit sphere -/
def sphereRadius : ℝ := 1

/-- Theorem stating that the distance between the two intersection points of the line and the unit sphere is 12√145/33 -/
theorem intersection_distance : 
  ∃ (p1 p2 : Point3D), 
    (∃ (t1 t2 : ℝ), 
      p1 = ⟨startPoint.x + t1 * (endPoint.x - startPoint.x), 
            startPoint.y + t1 * (endPoint.y - startPoint.y), 
            startPoint.z + t1 * (endPoint.z - startPoint.z)⟩ ∧
      p2 = ⟨startPoint.x + t2 * (endPoint.x - startPoint.x), 
            startPoint.y + t2 * (endPoint.y - startPoint.y), 
            startPoint.z + t2 * (endPoint.z - startPoint.z)⟩ ∧
      (p1.x - sphereCenter.x)^2 + (p1.y - sphereCenter.y)^2 + (p1.z - sphereCenter.z)^2 = sphereRadius^2 ∧
      (p2.x - sphereCenter.x)^2 + (p2.y - sphereCenter.y)^2 + (p2.z - sphereCenter.z)^2 = sphereRadius^2) →
    ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2) = (12 * Real.sqrt 145 / 33)^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l2530_253093


namespace NUMINAMATH_CALUDE_phillips_apples_l2530_253090

theorem phillips_apples (ben phillip tom : ℕ) : 
  ben = phillip + 8 →
  3 * ben = 8 * tom →
  tom = 18 →
  phillip = 40 := by
sorry

end NUMINAMATH_CALUDE_phillips_apples_l2530_253090


namespace NUMINAMATH_CALUDE_functions_for_12_functions_for_2007_functions_for_2_pow_2007_l2530_253087

-- Define the functions
def φ : ℕ → ℕ := sorry
def σ : ℕ → ℕ := sorry
def τ : ℕ → ℕ := sorry

-- Theorem for n = 12
theorem functions_for_12 :
  φ 12 = 4 ∧ σ 12 = 28 ∧ τ 12 = 6 := by sorry

-- Theorem for n = 2007
theorem functions_for_2007 :
  φ 2007 = 1332 ∧ σ 2007 = 2912 ∧ τ 2007 = 6 := by sorry

-- Theorem for n = 2^2007
theorem functions_for_2_pow_2007 :
  φ (2^2007) = 2^2006 ∧ 
  σ (2^2007) = 2^2008 - 1 ∧ 
  τ (2^2007) = 2008 := by sorry

end NUMINAMATH_CALUDE_functions_for_12_functions_for_2007_functions_for_2_pow_2007_l2530_253087


namespace NUMINAMATH_CALUDE_number_problem_l2530_253060

theorem number_problem (x : ℝ) : 0.65 * x = 0.8 * x - 21 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2530_253060


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l2530_253047

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l2530_253047


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2530_253034

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

theorem symmetric_points_sum (m n : ℝ) :
  symmetric_wrt_origin (m, 5) (3, n) → m + n = -8 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2530_253034


namespace NUMINAMATH_CALUDE_abc_sum_sixteen_l2530_253061

theorem abc_sum_sixteen (a b c : ℤ) 
  (h1 : a ≥ 4) (h2 : b ≥ 4) (h3 : c ≥ 4)
  (h4 : ¬(a = b ∧ b = c))
  (h5 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) :
  a + b + c = 16 := by sorry

end NUMINAMATH_CALUDE_abc_sum_sixteen_l2530_253061


namespace NUMINAMATH_CALUDE_problem_solution_l2530_253017

-- Given equation
def equation (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0

-- Define the system of equations
def system (a b c x y : ℝ) : Prop :=
  a*x + b*y = 30 ∧ c*x + a*y = 28

-- Define the quadratic equation
def quadratic (a b m x : ℝ) : Prop :=
  a*x^2 + b*x + m = 0

theorem problem_solution :
  ∀ a b c : ℝ, equation a b c →
  (∃ x y : ℝ, (a = 3 ∧ b = 4 ∧ c = 5) → system a b c x y ∧ x = 2 ∧ y = 6) ∧
  (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) ∧
  (∀ m : ℝ, (∃ x : ℝ, quadratic a b m x) → m ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2530_253017


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2530_253096

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h1 : seq.S 2016 = 2016)
    (h2 : seq.S 2016 / 2016 - seq.S 16 / 16 = 2000) :
  seq.a 1 = -2014 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2530_253096


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2530_253099

theorem largest_n_satisfying_inequality :
  ∀ n : ℕ, n ≤ 9 ↔ (1 / 4 : ℚ) + (n / 8 : ℚ) < (3 / 2 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2530_253099


namespace NUMINAMATH_CALUDE_brian_white_stones_l2530_253095

/-- Represents Brian's stone collection -/
structure StoneCollection where
  white : ℕ
  black : ℕ
  grey : ℕ
  green : ℕ
  red : ℕ
  blue : ℕ

/-- Conditions of Brian's stone collection -/
def BrianCollection : StoneCollection → Prop := fun c =>
  c.white + c.black = 100 ∧
  c.grey + c.green = 100 ∧
  c.red + c.blue = 130 ∧
  c.white + c.black + c.grey + c.green + c.red + c.blue = 330 ∧
  c.white > c.black ∧
  c.white = c.grey ∧
  c.black = c.green ∧
  3 * c.blue = 2 * c.red ∧
  2 * (c.white + c.grey) = c.red

theorem brian_white_stones (c : StoneCollection) 
  (h : BrianCollection c) : c.white = 78 := by
  sorry

end NUMINAMATH_CALUDE_brian_white_stones_l2530_253095


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reversal_l2530_253072

/-- Returns true if n is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Returns true if n starts with the digit 3 -/
def starts_with_three (n : ℕ) : Prop := 30 ≤ n ∧ n ≤ 39

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The theorem stating that 53 is the smallest two-digit prime starting with 3 
    whose digit reversal is composite -/
theorem smallest_two_digit_prime_with_composite_reversal : 
  ∃ (n : ℕ), 
    is_two_digit n ∧ 
    starts_with_three n ∧ 
    Nat.Prime n ∧ 
    ¬(Nat.Prime (reverse_digits n)) ∧
    (∀ m : ℕ, m < n → 
      is_two_digit m → 
      starts_with_three m → 
      Nat.Prime m → 
      Nat.Prime (reverse_digits m)) ∧
    n = 53 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reversal_l2530_253072


namespace NUMINAMATH_CALUDE_max_gingerbread_production_l2530_253037

/-- The gingerbread production function -/
def gingerbread_production (k : ℝ) (t : ℝ) : ℝ := k * t * (24 - t)

/-- Theorem stating that gingerbread production is maximized at 16 hours of work -/
theorem max_gingerbread_production (k : ℝ) (h : k > 0) :
  ∃ (t : ℝ), t = 16 ∧ ∀ (s : ℝ), 0 ≤ s ∧ s ≤ 24 → gingerbread_production k s ≤ gingerbread_production k t :=
by
  sorry

#check max_gingerbread_production

end NUMINAMATH_CALUDE_max_gingerbread_production_l2530_253037


namespace NUMINAMATH_CALUDE_min_sum_at_6_l2530_253054

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = -14
  sum_of_5th_6th : a 5 + a 6 = -4
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_of_first_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: The sum of the first n terms takes its minimum value when n = 6 -/
theorem min_sum_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≠ 0 → sum_of_first_n_terms seq 6 ≤ sum_of_first_n_terms seq n := by
  sorry

end NUMINAMATH_CALUDE_min_sum_at_6_l2530_253054


namespace NUMINAMATH_CALUDE_carpool_arrangement_count_l2530_253086

def num_students : ℕ := 8
def num_grades : ℕ := 4
def students_per_grade : ℕ := 2
def car_capacity : ℕ := 4

def has_twin_sisters : Prop := true

theorem carpool_arrangement_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_carpool_arrangement_count_l2530_253086


namespace NUMINAMATH_CALUDE_exists_segment_with_sum_455_l2530_253031

/-- Represents a 10x10 table filled with numbers 1 to 100 as described in the problem -/
def Table := Matrix (Fin 10) (Fin 10) Nat

/-- Defines how the table is filled -/
def fillTable : Table :=
  fun i j => i.val * 10 + j.val + 1

/-- Represents a 7-cell segment in the specified form -/
structure Segment where
  center : Fin 10 × Fin 10
  direction : Bool  -- True for vertical, False for horizontal

/-- Calculates the sum of a segment -/
def segmentSum (t : Table) (s : Segment) : Nat :=
  let (i, j) := s.center
  if s.direction then
    t i j + t (i-1) j + t (i+1) j +
    t (i-1) (j-1) + t (i-1) (j+1) +
    t (i+1) (j-1) + t (i+1) (j+1)
  else
    t i j + t i (j-1) + t i (j+1) +
    t (i-1) (j-1) + t (i+1) (j-1) +
    t (i-1) (j+1) + t (i+1) (j+1)

/-- The main theorem to prove -/
theorem exists_segment_with_sum_455 :
  ∃ s : Segment, segmentSum fillTable s = 455 := by
  sorry

end NUMINAMATH_CALUDE_exists_segment_with_sum_455_l2530_253031


namespace NUMINAMATH_CALUDE_common_difference_is_four_l2530_253052

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_correct : ∀ n, S n = (n * (a 0 + a (n - 1))) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ := seq.a 1 - seq.a 0

theorem common_difference_is_four (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 - 3 * seq.S 2 = 12) : 
  common_difference seq = 4 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_four_l2530_253052


namespace NUMINAMATH_CALUDE_midnight_probability_l2530_253056

/-- Represents the words from which letters are selected -/
inductive Word
| ROAD
| LIGHTS
| TIME

/-- Represents the target word MIDNIGHT -/
def targetWord : String := "MIDNIGHT"

/-- Number of letters to select from each word -/
def selectCount (w : Word) : Nat :=
  match w with
  | .ROAD => 2
  | .LIGHTS => 3
  | .TIME => 4

/-- The probability of selecting the required letters from a given word -/
def selectionProbability (w : Word) : Rat :=
  match w with
  | .ROAD => 1 / 3
  | .LIGHTS => 1 / 20
  | .TIME => 1 / 4

/-- The total probability of selecting all required letters -/
def totalProbability : Rat :=
  (selectionProbability .ROAD) * (selectionProbability .LIGHTS) * (selectionProbability .TIME)

theorem midnight_probability : totalProbability = 1 / 240 := by
  sorry

end NUMINAMATH_CALUDE_midnight_probability_l2530_253056


namespace NUMINAMATH_CALUDE_triangle_problem_l2530_253049

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  (2 * b + Real.sqrt 3 * c) * Real.cos A + Real.sqrt 3 * a * Real.cos C = 0 →
  A = 5 * π / 6 ∧
  (a = 2 → 
    ∃ (lower upper : ℝ), lower = 2 ∧ upper = 2 * Real.sqrt 3 ∧
    ∀ (x : ℝ), (∃ (b' c' : ℝ), b' + Real.sqrt 3 * c' = x ∧
      b' / (Real.sin B) = c' / (Real.sin C) ∧
      (2 * b' + Real.sqrt 3 * c') * Real.cos A + Real.sqrt 3 * a * Real.cos C = 0) →
    lower < x ∧ x < upper) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2530_253049


namespace NUMINAMATH_CALUDE_sum_of_squares_l2530_253029

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 70) → (a + b + c = 17) → (a^2 + b^2 + c^2 = 149) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2530_253029


namespace NUMINAMATH_CALUDE_triangle_inequality_l2530_253059

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  2 * (a^2 + b^2) > c^2 ∧ 
  ∀ ε > 0, ∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    (2 - ε) * (a'^2 + b'^2) ≤ c'^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2530_253059


namespace NUMINAMATH_CALUDE_speaking_orders_eq_552_l2530_253080

/-- The number of students in the class -/
def total_students : ℕ := 7

/-- The number of students to be selected for speaking -/
def speakers : ℕ := 4

/-- Function to calculate the number of different speaking orders -/
def speaking_orders : ℕ :=
  let only_one_ab := 2 * (total_students - 2).choose (speakers - 1) * (speakers).factorial
  let both_ab := (total_students - 3).choose (speakers - 2) * 2 * 6
  only_one_ab + both_ab

/-- Theorem stating that the number of different speaking orders is 552 -/
theorem speaking_orders_eq_552 : speaking_orders = 552 := by
  sorry

end NUMINAMATH_CALUDE_speaking_orders_eq_552_l2530_253080


namespace NUMINAMATH_CALUDE_negative_three_triangle_four_equals_seven_l2530_253024

-- Define the ▲ operation
def triangle (a b : ℚ) : ℚ := -a + b

-- Theorem statement
theorem negative_three_triangle_four_equals_seven :
  triangle (-3) 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_triangle_four_equals_seven_l2530_253024


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_unit_interval_l2530_253083

-- Define the sets M and N
def M : Set ℝ := {x | x^2 ≤ 1}
def N : Set (ℝ × ℝ) := {p | p.2 ∈ M ∧ p.1 = p.2^2}

-- State the theorem
theorem M_intersect_N_eq_unit_interval :
  (M ∩ (N.image Prod.snd)) = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_unit_interval_l2530_253083


namespace NUMINAMATH_CALUDE_als_original_portion_l2530_253035

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1200 →
  a - 150 + 3 * b + 3 * c = 1800 →
  c = 2 * b →
  a = 825 :=
by sorry

end NUMINAMATH_CALUDE_als_original_portion_l2530_253035


namespace NUMINAMATH_CALUDE_m_value_l2530_253076

/-- Triangle DEF with median DG to side EF -/
structure TriangleDEF where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  DG : ℝ
  is_median : DE = 5 ∧ EF = 12 ∧ DF = 13 ∧ DG * DG = 2 * (m * m)

/-- The value of m in the equation DG = m√2 for the given triangle -/
def find_m (t : TriangleDEF) : ℝ := sorry

/-- Theorem stating that m = √266 / 2 for the given triangle -/
theorem m_value (t : TriangleDEF) : find_m t = Real.sqrt 266 / 2 := by sorry

end NUMINAMATH_CALUDE_m_value_l2530_253076


namespace NUMINAMATH_CALUDE_sum_cubic_over_power_of_three_l2530_253048

/-- The sum of the infinite series ∑_{k = 1}^∞ (k^3 / 3^k) is equal to 1.5 -/
theorem sum_cubic_over_power_of_three :
  ∑' k : ℕ, (k^3 : ℝ) / 3^k = (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sum_cubic_over_power_of_three_l2530_253048


namespace NUMINAMATH_CALUDE_investment_problem_l2530_253016

theorem investment_problem (x y : ℝ) : 
  x * 0.10 - y * 0.08 = 83 →
  y = 650 →
  x + y = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l2530_253016


namespace NUMINAMATH_CALUDE_absolute_value_square_l2530_253008

theorem absolute_value_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_l2530_253008


namespace NUMINAMATH_CALUDE_highest_power_of_six_in_twelve_factorial_l2530_253053

/-- The highest power of 6 that divides 12! is 6^5 -/
theorem highest_power_of_six_in_twelve_factorial :
  ∃ k : ℕ, (12 : ℕ).factorial = 6^5 * k ∧ ¬(∃ m : ℕ, (12 : ℕ).factorial = 6^6 * m) := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_six_in_twelve_factorial_l2530_253053


namespace NUMINAMATH_CALUDE_angela_height_l2530_253071

/-- Given the heights of five people with specific relationships, prove Angela's height. -/
theorem angela_height (carl becky amy helen angela : ℝ) 
  (h1 : carl = 120)
  (h2 : becky = 2 * carl)
  (h3 : amy = becky * 1.2)
  (h4 : helen = amy + 3)
  (h5 : angela = helen + 4) :
  angela = 295 := by
  sorry

end NUMINAMATH_CALUDE_angela_height_l2530_253071


namespace NUMINAMATH_CALUDE_only_valid_N_l2530_253082

theorem only_valid_N : 
  {N : ℕ+ | (∃ a b : ℕ, N = 2^a * 5^b) ∧ 
            (∃ k : ℕ, N + 25 = k^2)} = 
  {200, 2000} := by sorry

end NUMINAMATH_CALUDE_only_valid_N_l2530_253082


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l2530_253021

theorem square_area_equal_perimeter_triangle (s : ℝ) :
  let triangle_perimeter := 5.5 + 5.5 + 7
  let square_side := triangle_perimeter / 4
  s = square_side → s^2 = 20.25 := by
sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l2530_253021


namespace NUMINAMATH_CALUDE_total_wall_length_l2530_253004

/-- Represents the daily work rate of a bricklayer in meters per day -/
def daily_rate : ℕ := 8

/-- Represents the number of working days -/
def working_days : ℕ := 15

/-- Theorem: The total length of wall laid by a bricklayer in 15 days -/
theorem total_wall_length : daily_rate * working_days = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_wall_length_l2530_253004


namespace NUMINAMATH_CALUDE_sequence_problem_l2530_253097

theorem sequence_problem (a b : ℝ) : 
  (∃ r : ℝ, 10 * r = a ∧ a * r = 1/2) →  -- geometric sequence condition
  (∃ d : ℝ, a + d = 5 ∧ 5 + d = b) →    -- arithmetic sequence condition
  a = Real.sqrt 5 ∧ b = 10 - Real.sqrt 5 := by
sorry


end NUMINAMATH_CALUDE_sequence_problem_l2530_253097


namespace NUMINAMATH_CALUDE_project_completion_l2530_253019

theorem project_completion 
  (a b c d e : ℕ) 
  (f g : ℝ) 
  (h₁ : a > 0) 
  (h₂ : c > 0) 
  (h₃ : f > 0) 
  (h₄ : g > 0) :
  (d : ℝ) * (b : ℝ) * g * (e : ℝ) / ((c : ℝ) * (a : ℝ)) = 
  (b : ℝ) * (d : ℝ) * g * (e : ℝ) / ((c : ℝ) * (a : ℝ)) :=
by sorry

#check project_completion

end NUMINAMATH_CALUDE_project_completion_l2530_253019


namespace NUMINAMATH_CALUDE_paintable_area_is_1520_l2530_253015

/-- Calculates the total paintable area of walls in multiple bedrooms. -/
def total_paintable_area (num_bedrooms length width height unpaintable_area : ℝ) : ℝ :=
  num_bedrooms * ((2 * (length * height + width * height)) - unpaintable_area)

/-- Proves that the total paintable area of walls in 4 bedrooms is 1520 square feet. -/
theorem paintable_area_is_1520 :
  total_paintable_area 4 14 11 9 70 = 1520 := by
  sorry

end NUMINAMATH_CALUDE_paintable_area_is_1520_l2530_253015


namespace NUMINAMATH_CALUDE_beth_class_students_left_l2530_253063

/-- The number of students who left Beth's class in the final year -/
def students_left (initial : ℕ) (joined : ℕ) (final : ℕ) : ℕ :=
  initial + joined - final

theorem beth_class_students_left : 
  students_left 150 30 165 = 15 := by sorry

end NUMINAMATH_CALUDE_beth_class_students_left_l2530_253063


namespace NUMINAMATH_CALUDE_tan_equality_unique_solution_l2530_253078

theorem tan_equality_unique_solution : 
  ∃! (n : ℤ), -100 < n ∧ n < 100 ∧ Real.tan (n * π / 180) = Real.tan (216 * π / 180) :=
by
  -- The unique solution is n = 36
  use 36
  sorry

end NUMINAMATH_CALUDE_tan_equality_unique_solution_l2530_253078


namespace NUMINAMATH_CALUDE_hari_well_digging_time_l2530_253014

theorem hari_well_digging_time 
  (jake_time : ℝ) 
  (paul_time : ℝ) 
  (combined_time : ℝ) 
  (h : jake_time = 16)
  (i : paul_time = 24)
  (j : combined_time = 8)
  : ∃ (hari_time : ℝ), 
    1 / jake_time + 1 / paul_time + 1 / hari_time = 1 / combined_time ∧ 
    hari_time = 48 := by
  sorry

end NUMINAMATH_CALUDE_hari_well_digging_time_l2530_253014


namespace NUMINAMATH_CALUDE_special_function_zero_l2530_253098

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, |f a - f b| ≤ |a - b|) ∧ (f (f (f 0)) = 0)

/-- Theorem: If f is a special function, then f(0) = 0 -/
theorem special_function_zero (f : ℝ → ℝ) (h : special_function f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_zero_l2530_253098


namespace NUMINAMATH_CALUDE_tickets_left_l2530_253041

/-- The number of tickets Dave started with -/
def initial_tickets : ℕ := 98

/-- The number of tickets Dave spent on the stuffed tiger -/
def spent_tickets : ℕ := 43

/-- Theorem stating that Dave had 55 tickets left after spending on the stuffed tiger -/
theorem tickets_left : initial_tickets - spent_tickets = 55 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_l2530_253041


namespace NUMINAMATH_CALUDE_battle_station_staffing_l2530_253044

theorem battle_station_staffing (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (n.descFactorial k) = 30240 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l2530_253044


namespace NUMINAMATH_CALUDE_expected_value_of_three_marbles_l2530_253009

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sumOfThree (s : Finset ℕ) : ℕ := s.sum id

def allCombinations : Finset (Finset ℕ) :=
  marbles.powerset.filter (λ s => s.card = 3)

def expectedValue : ℚ :=
  (allCombinations.sum sumOfThree) / allCombinations.card

theorem expected_value_of_three_marbles :
  expectedValue = 21/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_three_marbles_l2530_253009


namespace NUMINAMATH_CALUDE_square_of_complex_l2530_253025

theorem square_of_complex (z : ℂ) (i : ℂ) : z = 5 + 3 * i → i^2 = -1 → z^2 = 16 + 30 * i := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_l2530_253025


namespace NUMINAMATH_CALUDE_sam_oatmeal_cookies_l2530_253018

/-- Given a total number of cookies and a ratio of three types of cookies,
    calculate the number of cookies of the second type. -/
def oatmealCookies (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) : ℕ :=
  let totalParts := ratio1 + ratio2 + ratio3
  let cookiesPerPart := total / totalParts
  ratio2 * cookiesPerPart

/-- Theorem stating that given 36 total cookies and a ratio of 2:3:4,
    the number of oatmeal cookies is 12. -/
theorem sam_oatmeal_cookies :
  oatmealCookies 36 2 3 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sam_oatmeal_cookies_l2530_253018


namespace NUMINAMATH_CALUDE_basketball_shooting_improvement_l2530_253062

theorem basketball_shooting_improvement (initial_shots : ℕ) (initial_made : ℕ) (next_game_shots : ℕ) (new_average : ℚ) : 
  initial_shots = 35 → 
  initial_made = 15 → 
  next_game_shots = 15 → 
  new_average = 11/20 → 
  ∃ (next_game_made : ℕ), 
    next_game_made = 13 ∧ 
    (initial_made + next_game_made : ℚ) / (initial_shots + next_game_shots : ℚ) = new_average :=
by sorry

#check basketball_shooting_improvement

end NUMINAMATH_CALUDE_basketball_shooting_improvement_l2530_253062


namespace NUMINAMATH_CALUDE_rational_abs_four_and_self_reciprocal_l2530_253081

theorem rational_abs_four_and_self_reciprocal :
  (∀ x : ℚ, |x| = 4 ↔ x = -4 ∨ x = 4) ∧
  (∀ x : ℝ, x⁻¹ = x ↔ x = -1 ∨ x = 1) := by sorry

end NUMINAMATH_CALUDE_rational_abs_four_and_self_reciprocal_l2530_253081


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l2530_253012

/-- If x^2 + 110x + d is equal to the square of a binomial, then d = 3025 -/
theorem quadratic_square_of_binomial (d : ℝ) :
  (∃ b : ℝ, ∀ x : ℝ, x^2 + 110*x + d = (x + b)^2) → d = 3025 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l2530_253012


namespace NUMINAMATH_CALUDE_symmetric_point_about_origin_l2530_253094

/-- Given a point P with coordinates (2, -4), this theorem proves that its symmetric point
    about the origin has coordinates (-2, 4). -/
theorem symmetric_point_about_origin :
  let P : ℝ × ℝ := (2, -4)
  let symmetric_point : ℝ × ℝ := (-P.1, -P.2)
  symmetric_point = (-2, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_origin_l2530_253094


namespace NUMINAMATH_CALUDE_alice_flour_measurement_l2530_253013

/-- The number of times Alice needs to fill her measuring cup to get the required amount of flour -/
def number_of_fills (total_flour : ℚ) (cup_capacity : ℚ) : ℚ :=
  total_flour / cup_capacity

/-- Theorem: Alice needs to fill her ⅓ cup measuring cup 10 times to get 3⅓ cups of flour -/
theorem alice_flour_measurement :
  number_of_fills (3 + 1/3) (1/3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_alice_flour_measurement_l2530_253013


namespace NUMINAMATH_CALUDE_function_value_at_negative_two_l2530_253057

/-- Given a function f(x) = ax^5 + bx^3 + cx + 1 where f(2) = -1, prove that f(-2) = 3 -/
theorem function_value_at_negative_two 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^5 + b * x^3 + c * x + 1)
  (h2 : f 2 = -1) : 
  f (-2) = 3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_two_l2530_253057


namespace NUMINAMATH_CALUDE_expression_factorization_l2530_253079

theorem expression_factorization (x : ℝ) : 
  (9 * x^4 - 138 * x^3 + 49 * x^2) - (-3 * x^4 + 27 * x^3 - 14 * x^2) = 
  3 * x^2 * (4 * x - 3) * (x - 7) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l2530_253079


namespace NUMINAMATH_CALUDE_nathan_writes_25_letters_per_hour_l2530_253040

/-- The number of letters Nathan can write in one hour -/
def nathan_letters_per_hour : ℕ := sorry

/-- The number of letters Jacob can write in one hour -/
def jacob_letters_per_hour : ℕ := sorry

/-- Jacob writes twice as fast as Nathan -/
axiom jacob_twice_as_fast : jacob_letters_per_hour = 2 * nathan_letters_per_hour

/-- Together, Jacob and Nathan can write 750 letters in 10 hours -/
axiom combined_output : 10 * (jacob_letters_per_hour + nathan_letters_per_hour) = 750

theorem nathan_writes_25_letters_per_hour : nathan_letters_per_hour = 25 := by
  sorry

end NUMINAMATH_CALUDE_nathan_writes_25_letters_per_hour_l2530_253040


namespace NUMINAMATH_CALUDE_selling_price_range_l2530_253089

/-- Represents the daily sales revenue as a function of the selling price --/
def revenue (x : ℝ) : ℝ := x * (45 - 3 * (x - 15))

/-- The minimum selling price in yuan --/
def min_price : ℝ := 15

/-- The theorem stating the range of selling prices that generate over 600 yuan in daily revenue --/
theorem selling_price_range :
  {x : ℝ | revenue x > 600 ∧ x ≥ min_price} = Set.Icc 15 20 := by sorry

end NUMINAMATH_CALUDE_selling_price_range_l2530_253089


namespace NUMINAMATH_CALUDE_joe_running_speed_l2530_253069

/-- Proves that Joe's running speed is 16 km/h given the problem conditions -/
theorem joe_running_speed (pete_speed : ℝ) : 
  pete_speed > 0 →
  (2 * pete_speed * (40 / 60) + pete_speed * (40 / 60) = 16) →
  2 * pete_speed = 16 := by
  sorry

#check joe_running_speed

end NUMINAMATH_CALUDE_joe_running_speed_l2530_253069


namespace NUMINAMATH_CALUDE_correct_tile_count_l2530_253028

/-- The dimensions of the room --/
def room_width : ℝ := 8
def room_height : ℝ := 12

/-- The dimensions of a tile --/
def tile_width : ℝ := 1.5
def tile_height : ℝ := 2

/-- The number of tiles needed to cover the room --/
def tiles_needed : ℕ := 32

/-- Theorem stating that the number of tiles needed is correct --/
theorem correct_tile_count : 
  (room_width * room_height) / (tile_width * tile_height) = tiles_needed := by
  sorry

end NUMINAMATH_CALUDE_correct_tile_count_l2530_253028


namespace NUMINAMATH_CALUDE_vector_parallel_sum_l2530_253007

/-- Given vectors a and b in ℝ², if a is parallel to (a + b), then the y-coordinate of b is -1/2. -/
theorem vector_parallel_sum (a b : ℝ × ℝ) (h : a = (4, -1)) (h' : b.1 = 2) :
  (∃ (k : ℝ), k • a = a + b) → b.2 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_sum_l2530_253007


namespace NUMINAMATH_CALUDE_karlson_candies_theorem_l2530_253074

/-- The number of ones initially on the board -/
def initial_ones : ℕ := 33

/-- The number of minutes Karlson has -/
def total_minutes : ℕ := 33

/-- The maximum number of candies Karlson can eat -/
def max_candies : ℕ := initial_ones.choose 2

/-- Theorem stating that the maximum number of candies Karlson can eat
    is equal to the number of unique pairs from the initial ones -/
theorem karlson_candies_theorem :
  max_candies = (initial_ones * (initial_ones - 1)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_karlson_candies_theorem_l2530_253074


namespace NUMINAMATH_CALUDE_unique_cube_root_property_l2530_253065

theorem unique_cube_root_property : ∃! (n : ℕ), 
  n > 0 ∧ 
  (∃ (m : ℕ), n = m^3 ∧ m = n / 1000) ∧
  n = 32768 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_root_property_l2530_253065


namespace NUMINAMATH_CALUDE_puzzle_solution_l2530_253067

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Calculate the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + (n - 1) * seq.diff

theorem puzzle_solution :
  ∀ (row col1 col2 : ArithmeticSequence),
    row.first = 28 →
    nthTerm row 4 = 25 →
    nthTerm row 5 = 32 →
    nthTerm col2 7 = -10 →
    col2.first = nthTerm row 7 →
    col1.first = 28 →
    col1.diff = 7 →
    col2.first = -6 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2530_253067


namespace NUMINAMATH_CALUDE_smaug_gold_coins_l2530_253000

/-- Represents the number of coins in Smaug's hoard -/
structure DragonHoard where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Represents the value of different coin types in terms of copper coins -/
structure CoinValues where
  silver_to_copper : ℕ
  gold_to_silver : ℕ

/-- Calculates the total value of a hoard in copper coins -/
def hoard_value (hoard : DragonHoard) (values : CoinValues) : ℕ :=
  hoard.gold * values.gold_to_silver * values.silver_to_copper +
  hoard.silver * values.silver_to_copper +
  hoard.copper

/-- Theorem stating that Smaug has 100 gold coins -/
theorem smaug_gold_coins : 
  ∀ (hoard : DragonHoard) (values : CoinValues),
    hoard.silver = 60 →
    hoard.copper = 33 →
    values.silver_to_copper = 8 →
    values.gold_to_silver = 3 →
    hoard_value hoard values = 2913 →
    hoard.gold = 100 := by
  sorry

end NUMINAMATH_CALUDE_smaug_gold_coins_l2530_253000


namespace NUMINAMATH_CALUDE_crayons_count_l2530_253020

/-- The number of rows of crayons -/
def num_rows : ℕ := 7

/-- The number of crayons in each row -/
def crayons_per_row : ℕ := 30

/-- The total number of crayons -/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem crayons_count : total_crayons = 210 := by
  sorry

end NUMINAMATH_CALUDE_crayons_count_l2530_253020


namespace NUMINAMATH_CALUDE_angle_A_is_pi_div_6_max_area_when_a_is_2_l2530_253075

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.cos t.C = (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A

-- Theorem 1: Prove that angle A = π/6
theorem angle_A_is_pi_div_6 (t : Triangle) (h : condition t) : t.A = π / 6 :=
sorry

-- Theorem 2: Prove that when a = 2, the maximum area of triangle ABC is 2 + √3
theorem max_area_when_a_is_2 (t : Triangle) (h1 : condition t) (h2 : t.a = 2) :
  (t.b * t.c * Real.sin t.A / 2) ≤ 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_div_6_max_area_when_a_is_2_l2530_253075


namespace NUMINAMATH_CALUDE_percentage_problem_l2530_253066

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 100 - 40 = 30 → P = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2530_253066


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l2530_253050

-- Define the original point
def original_point : ℝ × ℝ := (-1, 1)

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the symmetric point
def symmetric_point : ℝ × ℝ := (2, -2)

-- Theorem statement
theorem symmetric_point_correct : 
  let (x₁, y₁) := original_point
  let (x₂, y₂) := symmetric_point
  (line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) ∧ 
   (x₂ - x₁) = (y₂ - y₁) ∧
   (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4 * ((x₁ - (x₁ + x₂) / 2)^2 + (y₁ - (y₁ + y₂) / 2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l2530_253050


namespace NUMINAMATH_CALUDE_rect_to_polar_conversion_l2530_253091

/-- Conversion from rectangular coordinates to polar coordinates -/
theorem rect_to_polar_conversion :
  ∀ (x y : ℝ), x = -1 ∧ y = 1 →
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = Real.sqrt 2 ∧ θ = 3 * Real.pi / 4 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rect_to_polar_conversion_l2530_253091


namespace NUMINAMATH_CALUDE_train_distance_difference_l2530_253070

theorem train_distance_difference (v : ℝ) (h1 : v > 0) : 
  let d_ab := 7 * v
  let d_bc := 5 * v
  6 = d_ab + d_bc →
  d_ab - d_bc = 1 := by
sorry

end NUMINAMATH_CALUDE_train_distance_difference_l2530_253070


namespace NUMINAMATH_CALUDE_percent_employed_females_l2530_253088

/-- Given a town where 60% of the population are employed and 42% of the population are employed males,
    prove that 30% of the employed people are females. -/
theorem percent_employed_females (town : Type) 
  (total_population : ℕ) 
  (employed : ℕ) 
  (employed_males : ℕ) 
  (h1 : employed = (60 : ℚ) / 100 * total_population) 
  (h2 : employed_males = (42 : ℚ) / 100 * total_population) : 
  (employed - employed_males : ℚ) / employed = 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_percent_employed_females_l2530_253088


namespace NUMINAMATH_CALUDE_imaginary_part_of_inverse_one_plus_i_squared_l2530_253026

theorem imaginary_part_of_inverse_one_plus_i_squared (i : ℂ) (h : i * i = -1) :
  Complex.im (1 / ((1 : ℂ) + i)^2) = -(1/2) := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_inverse_one_plus_i_squared_l2530_253026


namespace NUMINAMATH_CALUDE_same_solution_implies_c_equals_four_l2530_253092

theorem same_solution_implies_c_equals_four :
  ∀ x c : ℝ,
  (3 * x + 9 = 0) →
  (c * x + 15 = 3) →
  c = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_equals_four_l2530_253092


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2530_253030

theorem min_value_of_expression (a : ℝ) (h1 : 1 < a) (h2 : a < 4) :
  a / (4 - a) + 1 / (a - 1) ≥ 2 ∧
  (a / (4 - a) + 1 / (a - 1) = 2 ↔ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2530_253030


namespace NUMINAMATH_CALUDE_concert_guests_combinations_l2530_253085

theorem concert_guests_combinations : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_concert_guests_combinations_l2530_253085


namespace NUMINAMATH_CALUDE_circle_radius_from_arc_length_and_angle_l2530_253042

/-- Given a circle with a sector having an arc length of 25 cm and a central angle of 45 degrees,
    the radius of the circle is equal to 100/π cm. -/
theorem circle_radius_from_arc_length_and_angle (L : ℝ) (θ : ℝ) (r : ℝ) :
  L = 25 →
  θ = 45 →
  L = (θ / 360) * (2 * π * r) →
  r = 100 / π :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_arc_length_and_angle_l2530_253042


namespace NUMINAMATH_CALUDE_sum_of_first_four_terms_l2530_253077

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem sum_of_first_four_terms 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 9)
  (h_a5 : a 5 = 243) :
  (a 1) + (a 2) + (a 3) + (a 4) = 120 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_four_terms_l2530_253077


namespace NUMINAMATH_CALUDE_quiz_score_problem_l2530_253039

theorem quiz_score_problem (total_questions : ℕ) 
  (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 20 ∧ 
  correct_points = 7 ∧ 
  incorrect_points = -4 ∧ 
  total_score = 100 → 
  ∃ (correct incorrect blank : ℕ), 
    correct + incorrect + blank = total_questions ∧ 
    correct_points * correct + incorrect_points * incorrect = total_score ∧ 
    blank = 1 :=
by sorry

end NUMINAMATH_CALUDE_quiz_score_problem_l2530_253039


namespace NUMINAMATH_CALUDE_project_hours_difference_l2530_253032

theorem project_hours_difference (total : ℕ) (k p m : ℕ) : 
  total = k + p + m →
  p = 2 * k →
  3 * p = m →
  total = 153 →
  m - k = 85 := by sorry

end NUMINAMATH_CALUDE_project_hours_difference_l2530_253032


namespace NUMINAMATH_CALUDE_mary_always_wins_l2530_253002

/-- Represents a player in the game -/
inductive Player : Type
| john : Player
| mary : Player

/-- Represents a move in the game -/
inductive Move : Type
| plus : Move
| minus : Move

/-- Represents the state of the game -/
structure GameState :=
(moves : List Move)

/-- The list of numbers in the game -/
def numbers : List Int := [-1, -2, -3, -4, -5, -6, -7, -8]

/-- Calculate the final sum based on the moves and numbers -/
def finalSum (state : GameState) : Int :=
  sorry

/-- Check if Mary wins given the final sum -/
def maryWins (sum : Int) : Prop :=
  sum = -4 ∨ sum = -2 ∨ sum = 0 ∨ sum = 2 ∨ sum = 4

/-- Mary's strategy function -/
def maryStrategy (state : GameState) : Move :=
  sorry

/-- Theorem stating that Mary always wins -/
theorem mary_always_wins :
  ∀ (game : List Move),
    game.length ≤ 8 →
    maryWins (finalSum { moves := game ++ [maryStrategy { moves := game }] }) :=
sorry

end NUMINAMATH_CALUDE_mary_always_wins_l2530_253002


namespace NUMINAMATH_CALUDE_circle_equation_range_l2530_253027

/-- A circle in the xy-plane can be represented by an equation of the form
    (x - h)^2 + (y - k)^2 = r^2, where (h, k) is the center and r is the radius. -/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ h k r, r > 0 ∧ ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The equation x^2 + y^2 + 2kx + 4y + 3k + 8 = 0 represents a circle for some real k -/
def equation (k : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8

/-- The range of k for which the equation represents a circle -/
def k_range (k : ℝ) : Prop :=
  k < -1 ∨ k > 4

theorem circle_equation_range :
  ∀ k, is_circle (equation k) ↔ k_range k :=
sorry

end NUMINAMATH_CALUDE_circle_equation_range_l2530_253027


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2530_253045

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2530_253045


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2530_253068

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    left focus F₁, right focus F₂, and a point P on C such that
    PF₁ ⟂ F₁F₂ and PF₁ = F₁F₂, prove that the eccentricity of C is √2 + 1. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (C : Set (ℝ × ℝ))
  (hC : C = {(x, y) | x^2 / a^2 - y^2 / b^2 = 1})
  (F₁ F₂ P : ℝ × ℝ)
  (hF₁ : F₁ ∈ C) (hF₂ : F₂ ∈ C) (hP : P ∈ C)
  (hLeft : (F₁.1 < F₂.1)) -- F₁ is left of F₂
  (hPerp : (P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2) = 0) -- PF₁ ⟂ F₁F₂
  (hEqual : (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) -- PF₁ = F₁F₂
  : (Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) / (2 * a)) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2530_253068


namespace NUMINAMATH_CALUDE_bird_count_l2530_253033

theorem bird_count (total_wings : ℕ) (wings_per_bird : ℕ) (h1 : total_wings = 20) (h2 : wings_per_bird = 2) :
  total_wings / wings_per_bird = 10 := by
sorry

end NUMINAMATH_CALUDE_bird_count_l2530_253033
