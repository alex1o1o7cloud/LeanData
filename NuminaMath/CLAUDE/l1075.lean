import Mathlib

namespace NUMINAMATH_CALUDE_hockey_season_games_l1075_107592

/-- The number of hockey games per month -/
def games_per_month : ℕ := 13

/-- The number of months in the season -/
def months_in_season : ℕ := 14

/-- The total number of hockey games in the season -/
def total_games : ℕ := games_per_month * months_in_season

theorem hockey_season_games : total_games = 182 := by
  sorry

end NUMINAMATH_CALUDE_hockey_season_games_l1075_107592


namespace NUMINAMATH_CALUDE_decreasing_function_range_l1075_107552

/-- A function f: ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_function_range (f : ℝ → ℝ) (m : ℝ) 
  (h1 : DecreasingFunction f)
  (h2 : ∀ x, f (-x) = -f x)
  (h3 : f (m - 1) + f (2*m - 1) > 0) :
  m < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_range_l1075_107552


namespace NUMINAMATH_CALUDE_distribute_6_balls_4_boxes_l1075_107540

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 84 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_6_balls_4_boxes :
  distribute_balls 6 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_distribute_6_balls_4_boxes_l1075_107540


namespace NUMINAMATH_CALUDE_wednesday_sales_proof_l1075_107512

def initial_stock : ℕ := 1300
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135
def unsold_percentage : ℚ := 69.07692307692308

theorem wednesday_sales_proof :
  ∃ (wednesday_sales : ℕ),
    (initial_stock : ℚ) * (1 - unsold_percentage / 100) =
    (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales : ℚ) ∧
    wednesday_sales = 64 := by sorry

end NUMINAMATH_CALUDE_wednesday_sales_proof_l1075_107512


namespace NUMINAMATH_CALUDE_cosine_angle_between_vectors_l1075_107526

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![1, 3]

theorem cosine_angle_between_vectors :
  let inner_product := (a 0) * (b 0) + (a 1) * (b 1)
  let magnitude_a := Real.sqrt ((a 0)^2 + (a 1)^2)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2)
  (inner_product / (magnitude_a * magnitude_b)) = (7 * Real.sqrt 2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_angle_between_vectors_l1075_107526


namespace NUMINAMATH_CALUDE_composite_shape_area_theorem_l1075_107547

/-- The composite shape formed by a hexagon and an octagon attached to an equilateral triangle --/
structure CompositeShape where
  sideLength : ℝ
  hexagonArea : ℝ
  octagonArea : ℝ

/-- Calculate the area of the composite shape --/
def compositeShapeArea (shape : CompositeShape) : ℝ :=
  shape.hexagonArea + shape.octagonArea

/-- The theorem stating the area of the composite shape --/
theorem composite_shape_area_theorem (shape : CompositeShape) 
  (h1 : shape.sideLength = 2)
  (h2 : shape.hexagonArea = 4 * Real.sqrt 3 + 6)
  (h3 : shape.octagonArea = 8 * (1 + Real.sqrt 2) - 12) :
  compositeShapeArea shape = 4 * Real.sqrt 3 + 8 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_composite_shape_area_theorem_l1075_107547


namespace NUMINAMATH_CALUDE_even_function_with_domain_l1075_107548

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem even_function_with_domain (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →
  (∀ x, f a b x ≠ 0 → a - 1 ≤ x ∧ x ≤ 2 * a) →
  (∃ c d : ℝ, ∀ x, -2/3 ≤ x ∧ x ≤ 2/3 → f a b x = 1/3 * x^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_even_function_with_domain_l1075_107548


namespace NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_value_l1075_107570

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  j : ℝ
  k : ℝ
  zeros : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → zeros i ≠ zeros j
  arithmetic_progression : ∃ (b d : ℝ), ∀ i, zeros i = b + d * i.val
  is_zero : ∀ x, x^4 + j*x^2 + k*x + 256 = (x - zeros 0) * (x - zeros 1) * (x - zeros 2) * (x - zeros 3)

/-- The value of j in an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_j_value (p : ArithmeticProgressionPolynomial) : p.j = -40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_value_l1075_107570


namespace NUMINAMATH_CALUDE_triangle_area_similarity_l1075_107504

-- Define the triangles
variable (A B C D E F : ℝ × ℝ)

-- Define the similarity relation
def similar (t1 t2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

-- Define the area function
def area (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

-- Define the side length function
def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_similarity :
  similar (A, B, C) (D, E, F) →
  side_length A B / side_length D E = 2 →
  area (A, B, C) = 8 →
  area (D, E, F) = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_similarity_l1075_107504


namespace NUMINAMATH_CALUDE_average_of_data_set_l1075_107501

def data_set : List ℝ := [9.8, 9.9, 10, 10.1, 10.2]

theorem average_of_data_set :
  (List.sum data_set) / (List.length data_set) = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_set_l1075_107501


namespace NUMINAMATH_CALUDE_ac_length_l1075_107562

/-- An isosceles trapezoid with specific measurements -/
structure IsoscelesTrapezoid where
  /-- Length of base AB -/
  ab : ℝ
  /-- Length of top side CD -/
  cd : ℝ
  /-- Length of leg AD (equal to BC) -/
  ad : ℝ
  /-- Constraint that AB > CD -/
  h_ab_gt_cd : ab > cd

/-- Theorem: In the given isosceles trapezoid, AC = 17 -/
theorem ac_length (t : IsoscelesTrapezoid) 
  (h_ab : t.ab = 21)
  (h_cd : t.cd = 9)
  (h_ad : t.ad = 10) : 
  Real.sqrt ((21 - 9) ^ 2 / 4 + 8 ^ 2) = 17 := by
  sorry


end NUMINAMATH_CALUDE_ac_length_l1075_107562


namespace NUMINAMATH_CALUDE_infinite_m_exist_l1075_107567

/-- A(n) is the number of subsets of {1,2,...,n} with sum of elements divisible by p -/
def A (p : ℕ) (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem infinite_m_exist (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p)
  (h_not_div : ¬(p^2 ∣ (2^(p-1) - 1))) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ (m : ℕ), m ∈ S →
      ∀ (k : ℤ), ∃ (q : ℤ), A p m - k = p * q := by
  sorry

end NUMINAMATH_CALUDE_infinite_m_exist_l1075_107567


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1075_107505

theorem exam_maximum_marks :
  let pass_percentage : ℚ := 90 / 100
  let student_marks : ℕ := 250
  let failing_margin : ℕ := 300
  let maximum_marks : ℕ := 612
  (pass_percentage * maximum_marks : ℚ) = (student_marks + failing_margin : ℚ) ∧
  maximum_marks = (student_marks + failing_margin : ℚ) / pass_percentage :=
by sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1075_107505


namespace NUMINAMATH_CALUDE_complex_number_problem_l1075_107509

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_problem (z : ℂ) 
  (h1 : is_purely_imaginary z) 
  (h2 : is_purely_imaginary ((z + 2)^2 - 8*I)) : 
  z = -2*I := by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1075_107509


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_three_l1075_107523

theorem fraction_meaningful_iff_not_three (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_three_l1075_107523


namespace NUMINAMATH_CALUDE_son_age_l1075_107555

theorem son_age (son_age father_age : ℕ) : 
  father_age = son_age + 20 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_son_age_l1075_107555


namespace NUMINAMATH_CALUDE_nested_function_evaluation_l1075_107544

-- Define the functions a and b
def a (k : ℕ) : ℕ := (k + 1) ^ 2
def b (k : ℕ) : ℕ := k ^ 3 - 2 * k + 1

-- State the theorem
theorem nested_function_evaluation :
  b (a (a (a (a 1)))) = 95877196142432 :=
by sorry

end NUMINAMATH_CALUDE_nested_function_evaluation_l1075_107544


namespace NUMINAMATH_CALUDE_greatest_third_side_length_l1075_107543

/-- The greatest integer length of the third side of a triangle with two sides of 7 cm and 10 cm. -/
theorem greatest_third_side_length : ℕ :=
  let a : ℝ := 7
  let b : ℝ := 10
  let c : ℝ := 16
  have triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b := by sorry
  have c_less_than_sum : c < a + b := by sorry
  have c_greatest_integer : ∀ n : ℕ, (n : ℝ) > c → (n : ℝ) ≥ a + b := by sorry
  16


end NUMINAMATH_CALUDE_greatest_third_side_length_l1075_107543


namespace NUMINAMATH_CALUDE_conference_attendees_l1075_107554

theorem conference_attendees (total : ℕ) (writers : ℕ) (editors : ℕ) (both : ℕ) 
    (h1 : total = 100)
    (h2 : writers = 40)
    (h3 : editors ≥ 39)
    (h4 : both ≤ 21) :
  total - (writers + editors - both) ≤ 42 := by
  sorry

end NUMINAMATH_CALUDE_conference_attendees_l1075_107554


namespace NUMINAMATH_CALUDE_number_of_values_l1075_107573

theorem number_of_values (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) (correct_mean : ℚ) :
  initial_mean = 180 →
  incorrect_value = 135 →
  correct_value = 155 →
  correct_mean = 180 + 2/3 →
  ∃ n : ℕ, 
    n * initial_mean = n * correct_mean - (correct_value - incorrect_value) ∧
    n = 60 :=
by sorry

end NUMINAMATH_CALUDE_number_of_values_l1075_107573


namespace NUMINAMATH_CALUDE_a_faster_than_b_l1075_107581

/-- Represents a person sawing wood -/
structure Sawyer where
  name : String
  sections : ℕ
  pieces : ℕ

/-- Calculates the number of cuts required for a single piece of wood -/
def cuts (s : Sawyer) : ℕ := s.sections - 1

/-- Calculates the total number of cuts made by a sawyer -/
def totalCuts (s : Sawyer) : ℕ := (s.pieces / s.sections) * cuts s

/-- Defines what it means for one sawyer to be faster than another -/
def isFasterThan (s1 s2 : Sawyer) : Prop := totalCuts s1 > totalCuts s2

theorem a_faster_than_b :
  let a : Sawyer := ⟨"A", 3, 24⟩
  let b : Sawyer := ⟨"B", 2, 28⟩
  isFasterThan a b := by sorry

end NUMINAMATH_CALUDE_a_faster_than_b_l1075_107581


namespace NUMINAMATH_CALUDE_contingency_fund_allocation_l1075_107516

def total_donation : ℚ := 240

def community_pantry_ratio : ℚ := 1/3
def local_crisis_ratio : ℚ := 1/2
def livelihood_ratio : ℚ := 1/4

def community_pantry : ℚ := total_donation * community_pantry_ratio
def local_crisis : ℚ := total_donation * local_crisis_ratio

def remaining_after_main : ℚ := total_donation - (community_pantry + local_crisis)
def livelihood : ℚ := remaining_after_main * livelihood_ratio

def contingency : ℚ := remaining_after_main - livelihood

theorem contingency_fund_allocation :
  contingency = 30 := by sorry

end NUMINAMATH_CALUDE_contingency_fund_allocation_l1075_107516


namespace NUMINAMATH_CALUDE_sum_f_2016_2017_2018_l1075_107507

/-- An odd periodic function with period 4 and f(1) = 1 -/
def f : ℝ → ℝ :=
  sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f has period 4 -/
axiom f_periodic : ∀ x, f (x + 4) = f x

/-- f(1) = 1 -/
axiom f_one : f 1 = 1

theorem sum_f_2016_2017_2018 : f 2016 + f 2017 + f 2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_2016_2017_2018_l1075_107507


namespace NUMINAMATH_CALUDE_strange_clock_time_l1075_107539

/-- Represents a hand on the strange clock -/
inductive ClockHand
| A
| B
| C

/-- Represents the position of a clock hand -/
structure HandPosition where
  exactHourMark : Bool
  slightlyBeforeHourMark : Bool

/-- Represents the strange clock -/
structure StrangeClock where
  hands : ClockHand → HandPosition
  sameLength : Bool
  noNumbers : Bool
  unclearTop : Bool

/-- Determines if a given time matches the strange clock configuration -/
def matchesClockConfiguration (clock : StrangeClock) (hours : Nat) (minutes : Nat) : Prop :=
  hours = 16 ∧ minutes = 50 ∧
  clock.hands ClockHand.A = { exactHourMark := true, slightlyBeforeHourMark := false } ∧
  clock.hands ClockHand.B = { exactHourMark := false, slightlyBeforeHourMark := true } ∧
  clock.hands ClockHand.C = { exactHourMark := false, slightlyBeforeHourMark := true } ∧
  clock.sameLength ∧ clock.noNumbers ∧ clock.unclearTop

theorem strange_clock_time (clock : StrangeClock) 
  (h1 : clock.hands ClockHand.A = { exactHourMark := true, slightlyBeforeHourMark := false })
  (h2 : clock.hands ClockHand.B = { exactHourMark := false, slightlyBeforeHourMark := true })
  (h3 : clock.hands ClockHand.C = { exactHourMark := false, slightlyBeforeHourMark := true })
  (h4 : clock.sameLength)
  (h5 : clock.noNumbers)
  (h6 : clock.unclearTop) :
  ∃ (hours minutes : Nat), matchesClockConfiguration clock hours minutes :=
by
  sorry

end NUMINAMATH_CALUDE_strange_clock_time_l1075_107539


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1075_107588

theorem quadratic_root_difference (s t : ℝ) (hs : s > 0) (ht : t > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + s*x₁ + t = 0 ∧
    x₂^2 + s*x₂ + t = 0 ∧
    |x₁ - x₂| = 2) →
  s = 2 * Real.sqrt (t + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1075_107588


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1075_107514

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 19*x - 48 = 0 → x ≤ 24 :=
by
  sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1075_107514


namespace NUMINAMATH_CALUDE_cows_gifted_is_eight_l1075_107511

/-- Calculates the number of cows given as a gift -/
def cows_gifted (initial : ℕ) (died : ℕ) (sold : ℕ) (increased : ℕ) (bought : ℕ) (total : ℕ) : ℕ :=
  total - (initial - died - sold + increased + bought)

/-- Theorem stating that the number of cows gifted is 8 -/
theorem cows_gifted_is_eight :
  cows_gifted 39 25 6 24 43 83 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cows_gifted_is_eight_l1075_107511


namespace NUMINAMATH_CALUDE_average_theorem_l1075_107529

theorem average_theorem (a b c : ℝ) :
  (a + b + c) / 3 = 12 →
  ((2*a + 1) + (2*b + 2) + (2*c + 3) + 2) / 4 = 20 := by
sorry

end NUMINAMATH_CALUDE_average_theorem_l1075_107529


namespace NUMINAMATH_CALUDE_fib_last_four_zeros_exist_l1075_107564

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Last four digits of a natural number -/
def lastFourDigits (n : ℕ) : ℕ :=
  n % 10000

/-- Theorem: There exists a term in the first 100,000,001 Fibonacci numbers whose last four digits are all zeros -/
theorem fib_last_four_zeros_exist : ∃ n : ℕ, n < 100000001 ∧ lastFourDigits (fib n) = 0 := by
  sorry


end NUMINAMATH_CALUDE_fib_last_four_zeros_exist_l1075_107564


namespace NUMINAMATH_CALUDE_pyramid_frustum_volume_l1075_107542

/-- Calculate the volume of a pyramid frustum given the dimensions of the original and smaller pyramids --/
theorem pyramid_frustum_volume
  (base_edge_original : ℝ)
  (altitude_original : ℝ)
  (base_edge_smaller : ℝ)
  (altitude_smaller : ℝ)
  (h_base_edge_original : base_edge_original = 18)
  (h_altitude_original : altitude_original = 12)
  (h_base_edge_smaller : base_edge_smaller = 12)
  (h_altitude_smaller : altitude_smaller = 8) :
  (1/3 * base_edge_original^2 * altitude_original) - (1/3 * base_edge_smaller^2 * altitude_smaller) = 912 := by
  sorry

#check pyramid_frustum_volume

end NUMINAMATH_CALUDE_pyramid_frustum_volume_l1075_107542


namespace NUMINAMATH_CALUDE_seeds_in_gray_parts_l1075_107575

theorem seeds_in_gray_parts (total_seeds : ℕ) 
  (white_seeds_circle1 : ℕ) (white_seeds_circle2 : ℕ) (white_seeds_each : ℕ)
  (h1 : white_seeds_circle1 = 87)
  (h2 : white_seeds_circle2 = 110)
  (h3 : white_seeds_each = 68) :
  (white_seeds_circle1 - white_seeds_each) + (white_seeds_circle2 - white_seeds_each) = 61 := by
  sorry

end NUMINAMATH_CALUDE_seeds_in_gray_parts_l1075_107575


namespace NUMINAMATH_CALUDE_train_passing_jogger_l1075_107536

/-- Theorem: Train passing jogger
  Given:
  - Jogger's speed: 9 kmph
  - Train's speed: 45 kmph
  - Train's length: 120 meters
  - Initial distance between jogger and train engine: 240 meters
  Prove: The time for the train to pass the jogger is 36 seconds
-/
theorem train_passing_jogger 
  (jogger_speed : Real) 
  (train_speed : Real) 
  (train_length : Real) 
  (initial_distance : Real) 
  (h1 : jogger_speed = 9) 
  (h2 : train_speed = 45) 
  (h3 : train_length = 120) 
  (h4 : initial_distance = 240) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_l1075_107536


namespace NUMINAMATH_CALUDE_midpoint_lines_perpendicular_l1075_107574

/-- A circle in which a quadrilateral is inscribed -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point on a circle -/
structure CirclePoint (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral (c : Circle) where
  A : CirclePoint c
  B : CirclePoint c
  C : CirclePoint c
  D : CirclePoint c

/-- Midpoint of an arc on a circle -/
def arcMidpoint (c : Circle) (p1 p2 : CirclePoint c) : CirclePoint c :=
  sorry

/-- Perpendicularity of two lines defined by four points -/
def arePerpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  sorry

/-- Theorem statement -/
theorem midpoint_lines_perpendicular (c : Circle) (quad : InscribedQuadrilateral c) :
  let M := arcMidpoint c quad.A quad.B
  let N := arcMidpoint c quad.B quad.C
  let P := arcMidpoint c quad.C quad.D
  let Q := arcMidpoint c quad.D quad.A
  arePerpendicular M.point P.point N.point Q.point :=
by sorry

end NUMINAMATH_CALUDE_midpoint_lines_perpendicular_l1075_107574


namespace NUMINAMATH_CALUDE_train_length_approx_l1075_107580

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length_approx (speed : ℝ) (time : ℝ) : 
  speed = 100 → time = 3.6 → ∃ (length : ℝ), 
  (abs (length - (speed * 1000 / 3600 * time)) < 0.5) ∧ 
  (round length = 100) := by
  sorry

#check train_length_approx

end NUMINAMATH_CALUDE_train_length_approx_l1075_107580


namespace NUMINAMATH_CALUDE_average_temperature_l1075_107597

def temperatures : List ℝ := [52, 64, 59, 60, 47]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 56.4 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l1075_107597


namespace NUMINAMATH_CALUDE_expression_simplification_l1075_107550

theorem expression_simplification (b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  let a : ℝ := 0.04
  1.24 * (Real.sqrt ((a * b * c + 4) / a + 4 * Real.sqrt (b * c / a))) / (Real.sqrt (a * b * c) + 2) = 6.2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1075_107550


namespace NUMINAMATH_CALUDE_polynomial_factorization_1_l1075_107582

theorem polynomial_factorization_1 (a : ℝ) : 
  a^7 + a^5 + 1 = (a^2 + a + 1) * (a^5 - a^4 + a^3 - a + 1) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_1_l1075_107582


namespace NUMINAMATH_CALUDE_sara_peaches_theorem_l1075_107590

/-- The number of peaches Sara picked initially -/
def initial_peaches : ℝ := 61

/-- The number of peaches Sara picked at the orchard -/
def orchard_peaches : ℝ := 24.0

/-- The total number of peaches Sara picked -/
def total_peaches : ℝ := 85

/-- Theorem stating that the initial number of peaches plus the orchard peaches equals the total peaches -/
theorem sara_peaches_theorem : initial_peaches + orchard_peaches = total_peaches := by
  sorry

end NUMINAMATH_CALUDE_sara_peaches_theorem_l1075_107590


namespace NUMINAMATH_CALUDE_min_yellow_marbles_l1075_107551

-- Define the total number of marbles
variable (n : ℕ)

-- Define the number of yellow marbles
variable (y : ℕ)

-- Define the conditions
def blue_marbles := n / 3
def red_marbles := n / 4
def green_marbles := 9
def white_marbles := 2 * y

-- Define the total number of marbles equation
def total_marbles_equation : Prop :=
  n = blue_marbles n + red_marbles n + green_marbles + y + white_marbles y

-- Theorem statement
theorem min_yellow_marbles :
  (∃ n : ℕ, total_marbles_equation n y) → y ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_yellow_marbles_l1075_107551


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1075_107584

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x + 3| < 8 ↔ -4.5 < x ∧ x < 3.5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1075_107584


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1075_107557

theorem fraction_sum_equality : (3 : ℚ) / 8 + 9 / 12 - 1 / 6 = 23 / 24 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1075_107557


namespace NUMINAMATH_CALUDE_smallest_multiple_of_7_4_5_l1075_107556

theorem smallest_multiple_of_7_4_5 : ∃ n : ℕ+, (∀ m : ℕ+, m.val % 7 = 0 ∧ m.val % 4 = 0 ∧ m.val % 5 = 0 → n ≤ m) ∧ n.val % 7 = 0 ∧ n.val % 4 = 0 ∧ n.val % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_7_4_5_l1075_107556


namespace NUMINAMATH_CALUDE_unpainted_area_crossed_boards_l1075_107565

/-- The area of the unpainted region when two boards cross at 45 degrees -/
theorem unpainted_area_crossed_boards (width1 width2 : ℝ) (angle : ℝ) :
  width1 = 5 →
  width2 = 7 →
  angle = π / 4 →
  let projected_length := width2
  let overlap_height := width2 * Real.cos angle
  let unpainted_area := projected_length * overlap_height
  unpainted_area = 49 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_area_crossed_boards_l1075_107565


namespace NUMINAMATH_CALUDE_bee_swarm_size_l1075_107508

theorem bee_swarm_size :
  ∃ n : ℕ,
    n > 0 ∧
    (n : ℝ) = (Real.sqrt ((n : ℝ) / 2)) + (8 / 9 * n) + 1 ∧
    n = 72 := by
  sorry

end NUMINAMATH_CALUDE_bee_swarm_size_l1075_107508


namespace NUMINAMATH_CALUDE_circular_garden_area_l1075_107585

-- Define the radius of the garden
def radius : ℝ := 16

-- Define the relationship between circumference and area
def fence_area_relation (circumference area : ℝ) : Prop :=
  circumference = (1/8) * area

-- Theorem statement
theorem circular_garden_area :
  let circumference := 2 * Real.pi * radius
  let area := Real.pi * radius^2
  fence_area_relation circumference area →
  area = 256 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_circular_garden_area_l1075_107585


namespace NUMINAMATH_CALUDE_floor_equality_sufficient_not_necessary_for_abs_diff_less_than_one_l1075_107594

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := 
  Int.floor x

-- Theorem statement
theorem floor_equality_sufficient_not_necessary_for_abs_diff_less_than_one :
  (∀ x y : ℝ, floor x = floor y → |x - y| < 1) ∧
  (∃ x y : ℝ, |x - y| < 1 ∧ floor x ≠ floor y) := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_sufficient_not_necessary_for_abs_diff_less_than_one_l1075_107594


namespace NUMINAMATH_CALUDE_flag_length_proof_l1075_107572

/-- Represents a rectangular piece of fabric -/
structure Fabric where
  length : ℝ
  width : ℝ

/-- The problem setup -/
def flag_problem : Prop :=
  let fabric1 : Fabric := ⟨8, 5⟩
  let fabric2 : Fabric := ⟨10, 7⟩
  let fabric3 : Fabric := ⟨5, 5⟩
  let desired_height : ℝ := 9
  ∃ (max_length : ℝ),
    max_length = 10 ∧
    max_length = max fabric1.length (max fabric2.length fabric3.length) ∧
    max_length ≥ desired_height

theorem flag_length_proof : flag_problem := by
  sorry

end NUMINAMATH_CALUDE_flag_length_proof_l1075_107572


namespace NUMINAMATH_CALUDE_albert_books_multiple_l1075_107587

theorem albert_books_multiple (stu_books : ℕ) (total_books : ℕ) (x : ℚ) : 
  stu_books = 9 →
  total_books = 45 →
  total_books = stu_books + stu_books * x →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_albert_books_multiple_l1075_107587


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_ratio_l1075_107538

theorem rectangle_area_diagonal_ratio (length width diagonal : ℝ) (k : ℝ) : 
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 4 / 3 →
  diagonal ^ 2 = length ^ 2 + width ^ 2 →
  length * width = k * diagonal ^ 2 →
  k = 12 / 25 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_ratio_l1075_107538


namespace NUMINAMATH_CALUDE_range_of_a_l1075_107522

def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

theorem range_of_a (a : ℝ) (h : A ∩ B a = B a) : a ≤ 0 ∨ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1075_107522


namespace NUMINAMATH_CALUDE_first_digit_853_base8_l1075_107569

/-- The first digit of the base 8 representation of a natural number -/
def firstDigitBase8 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let k := (Nat.log n 8).succ
    (n / 8^(k-1)) % 8

theorem first_digit_853_base8 :
  firstDigitBase8 853 = 1 := by
sorry

end NUMINAMATH_CALUDE_first_digit_853_base8_l1075_107569


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l1075_107510

theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let a := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  let r := (d1 * d2) / (4 * a)
  r = 60 / 13 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l1075_107510


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1075_107577

theorem polynomial_remainder (x : ℤ) : (x^15 - 1) % (x + 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1075_107577


namespace NUMINAMATH_CALUDE_hospital_staff_count_l1075_107500

theorem hospital_staff_count (total : ℕ) (d_ratio n_ratio : ℕ) (h1 : total = 456) (h2 : d_ratio = 8) (h3 : n_ratio = 11) : 
  ∃ (doctors nurses : ℕ), 
    doctors + nurses = total ∧ 
    doctors * n_ratio = nurses * d_ratio ∧ 
    nurses = 264 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l1075_107500


namespace NUMINAMATH_CALUDE_calculation_proof_l1075_107578

theorem calculation_proof : 
  100 - (25/8) / ((25/12) - (5/8)) * ((8/5) + (8/3)) = 636/7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1075_107578


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l1075_107591

/-- The number of ways to select a non-empty subset of fruits from a given number of identical apples and oranges, such that at least 2 oranges are selected. -/
def fruitBasketCombinations (apples oranges : ℕ) : ℕ :=
  (apples + 1) * (oranges - 1)

/-- Theorem stating that the number of fruit basket combinations with 4 apples and 12 oranges is 55. -/
theorem fruit_basket_problem :
  fruitBasketCombinations 4 12 = 55 := by
  sorry

#eval fruitBasketCombinations 4 12

end NUMINAMATH_CALUDE_fruit_basket_problem_l1075_107591


namespace NUMINAMATH_CALUDE_polynomial_sum_and_coefficient_sum_l1075_107521

theorem polynomial_sum_and_coefficient_sum (d : ℝ) (h : d ≠ 0) :
  (15 * d^3 + 12 * d + 7 + 18 * d^2) + (2 * d^3 + d - 6 + 3 * d^2) =
  17 * d^3 + 21 * d^2 + 13 * d + 1 ∧
  17 + 21 + 13 + 1 = 52 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_and_coefficient_sum_l1075_107521


namespace NUMINAMATH_CALUDE_marathon_remainder_l1075_107502

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def marathon : Marathon :=
  { miles := 26, yards := 385 }

def yardsPerMile : ℕ := 1760

def numMarathons : ℕ := 15

theorem marathon_remainder (m : ℕ) (y : ℕ) 
  (h : Distance.mk m y = 
    { miles := numMarathons * marathon.miles + (numMarathons * marathon.yards) / yardsPerMile,
      yards := (numMarathons * marathon.yards) % yardsPerMile }) :
  y = 495 := by
  sorry

end NUMINAMATH_CALUDE_marathon_remainder_l1075_107502


namespace NUMINAMATH_CALUDE_sum_342_78_base5_l1075_107524

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list representing a number in base 5 to a natural number in base 10 -/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 5 representation -/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_342_78_base5 :
  addBase5 (toBase5 342) (toBase5 78) = [3, 1, 4, 0] :=
sorry

end NUMINAMATH_CALUDE_sum_342_78_base5_l1075_107524


namespace NUMINAMATH_CALUDE_clean_80_cars_per_day_l1075_107515

/-- Represents the Super Clean Car Wash Company's operations -/
structure CarWash where
  price_per_car : ℕ  -- Price per car in dollars
  total_revenue : ℕ  -- Total revenue in dollars
  num_days : ℕ       -- Number of days

/-- Calculates the number of cars cleaned per day -/
def cars_per_day (cw : CarWash) : ℕ :=
  (cw.total_revenue / cw.price_per_car) / cw.num_days

/-- Theorem stating that the number of cars cleaned per day is 80 -/
theorem clean_80_cars_per_day (cw : CarWash)
  (h1 : cw.price_per_car = 5)
  (h2 : cw.total_revenue = 2000)
  (h3 : cw.num_days = 5) :
  cars_per_day cw = 80 := by
  sorry

#eval cars_per_day ⟨5, 2000, 5⟩

end NUMINAMATH_CALUDE_clean_80_cars_per_day_l1075_107515


namespace NUMINAMATH_CALUDE_system_solution_difference_l1075_107534

theorem system_solution_difference (x y : ℝ) : 
  1012 * x + 1016 * y = 1020 →
  1014 * x + 1018 * y = 1022 →
  x - y = 1.09 := by
sorry

end NUMINAMATH_CALUDE_system_solution_difference_l1075_107534


namespace NUMINAMATH_CALUDE_c_range_l1075_107541

def p (c : ℝ) : Prop := c^2 < c

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + 4*c*x + 1 > 0

def range_of_c (c : ℝ) : Prop :=
  (c > -1/2 ∧ c ≤ 0) ∨ (c ≥ 1/2 ∧ c < 1)

theorem c_range (c : ℝ) :
  (p c ∨ q c) ∧ ¬(p c ∧ q c) → range_of_c c :=
sorry

end NUMINAMATH_CALUDE_c_range_l1075_107541


namespace NUMINAMATH_CALUDE_reflected_triangle_angles_l1075_107537

-- Define the triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the reflection operation
def reflect (t : Triangle) : Triangle := sorry

-- Define the property of being acute
def is_acute (t : Triangle) : Prop := sorry

-- Define the property of being scalene
def is_scalene (t : Triangle) : Prop := sorry

-- Define the property of A₁B₁C₁ being inside ABC
def is_inside (t1 t2 : Triangle) : Prop := sorry

-- Define the property of A₂B₂C₂ being outside ABC
def is_outside (t1 t2 : Triangle) : Prop := sorry

-- Define the theorem
theorem reflected_triangle_angles 
  (ABC : Triangle) 
  (A₁B₁C₁ : Triangle) 
  (A₂B₂C₂ : Triangle) :
  is_acute ABC →
  is_scalene ABC →
  is_inside A₁B₁C₁ ABC →
  is_outside A₂B₂C₂ ABC →
  (A₂B₂C₂.a = 20 ∨ A₂B₂C₂.b = 20 ∨ A₂B₂C₂.c = 20) →
  (A₂B₂C₂.a = 70 ∨ A₂B₂C₂.b = 70 ∨ A₂B₂C₂.c = 70) →
  ((A₁B₁C₁.a = 60 ∧ A₁B₁C₁.b = 60 ∧ A₁B₁C₁.c = 60) ∨
   (A₁B₁C₁.a = 140/3 ∧ A₁B₁C₁.b = 60 ∧ A₁B₁C₁.c = 220/3) ∨
   (A₁B₁C₁.a = 60 ∧ A₁B₁C₁.b = 140/3 ∧ A₁B₁C₁.c = 220/3) ∨
   (A₁B₁C₁.a = 220/3 ∧ A₁B₁C₁.b = 60 ∧ A₁B₁C₁.c = 140/3)) :=
by sorry

end NUMINAMATH_CALUDE_reflected_triangle_angles_l1075_107537


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l1075_107553

theorem positive_integer_solutions_of_equation :
  {(x, y) : ℕ × ℕ | 2 * x^2 - 7 * x * y + 3 * y^3 = 0 ∧ x > 0 ∧ y > 0} =
  {(3, 1), (3, 2), (4, 2)} :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l1075_107553


namespace NUMINAMATH_CALUDE_amount_lent_to_C_is_correct_l1075_107576

/-- The amount of money A lent to C -/
def amount_lent_to_C : ℝ := 500

/-- The amount of money A lent to B -/
def amount_lent_to_B : ℝ := 5000

/-- The duration of the loan to B in years -/
def duration_B : ℝ := 2

/-- The duration of the loan to C in years -/
def duration_C : ℝ := 4

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.1

/-- The total interest received from both B and C -/
def total_interest : ℝ := 2200

theorem amount_lent_to_C_is_correct :
  amount_lent_to_C * interest_rate * duration_C +
  amount_lent_to_B * interest_rate * duration_B = total_interest :=
sorry

end NUMINAMATH_CALUDE_amount_lent_to_C_is_correct_l1075_107576


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l1075_107535

/-- The number of children in the family -/
def num_children : ℕ := 6

/-- The probability of a child being male (or female) -/
def gender_prob : ℚ := 1/2

/-- The probability of having an unequal number of sons and daughters -/
def unequal_gender_prob : ℚ := 11/16

theorem unequal_gender_probability :
  (1 : ℚ) - (Nat.choose num_children (num_children / 2) : ℚ) / (2 ^ num_children) = unequal_gender_prob :=
sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l1075_107535


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l1075_107589

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_non_factor_product (a b : ℕ) : 
  a ≠ b →
  a > 0 →
  b > 0 →
  is_factor a 48 →
  is_factor b 48 →
  ¬ is_factor (a * b) 48 →
  (∀ x y : ℕ, x ≠ y → x > 0 → y > 0 → is_factor x 48 → is_factor y 48 → 
    ¬ is_factor (x * y) 48 → a * b ≤ x * y) →
  a * b = 32 := by sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l1075_107589


namespace NUMINAMATH_CALUDE_twenty_first_term_of_ap_l1075_107546

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem twenty_first_term_of_ap (a₁ d : ℝ) (h₁ : a₁ = 3) (h₂ : d = 5) :
  arithmeticProgressionTerm a₁ d 21 = 103 :=
by sorry

end NUMINAMATH_CALUDE_twenty_first_term_of_ap_l1075_107546


namespace NUMINAMATH_CALUDE_second_number_is_984_l1075_107583

theorem second_number_is_984 (a b : ℕ) : 
  a < 10 ∧ b < 10 ∧ 
  a + b = 10 ∧
  (1000 + 300 + 10 * b + 7) % 11 = 0 →
  1000 + 300 + 10 * b + 7 - (400 + 10 * a + 3) = 984 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_984_l1075_107583


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1075_107549

/-- Given a hyperbola with the following properties:
    - The distance from the vertex to its asymptote is 2
    - The distance from the focus to the asymptote is 6
    Then the eccentricity of the hyperbola is 3 -/
theorem hyperbola_eccentricity (vertex_to_asymptote : ℝ) (focus_to_asymptote : ℝ) 
  (h1 : vertex_to_asymptote = 2)
  (h2 : focus_to_asymptote = 6) :
  let e := focus_to_asymptote / vertex_to_asymptote
  e = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1075_107549


namespace NUMINAMATH_CALUDE_zero_is_natural_number_zero_not_natural_is_false_l1075_107560

-- Define the set of natural numbers including 0
def NaturalNumbers : Set ℕ := {n : ℕ | True}

-- State the theorem
theorem zero_is_natural_number : (0 : ℕ) ∈ NaturalNumbers := by
  sorry

-- Prove that the statement "0 is not a natural number" is false
theorem zero_not_natural_is_false : ¬(0 ∉ NaturalNumbers) := by
  sorry

end NUMINAMATH_CALUDE_zero_is_natural_number_zero_not_natural_is_false_l1075_107560


namespace NUMINAMATH_CALUDE_unchanged_fraction_l1075_107503

theorem unchanged_fraction (x y : ℝ) : 
  (2 * x) / (3 * x - y) = (2 * (3 * x)) / (3 * (3 * x) - (3 * y)) :=
by sorry

end NUMINAMATH_CALUDE_unchanged_fraction_l1075_107503


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1075_107571

theorem arithmetic_sequence_sum (a₁ a₄ a₁₀ : ℚ) (n : ℕ) : 
  a₁ = -3 → a₄ = 4 → a₁₀ = 40 → n = 10 → 
  (n : ℚ) / 2 * (a₁ + a₁₀) = 285 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1075_107571


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l1075_107545

theorem cos_sum_of_complex_exponentials (α β : ℝ) : 
  Complex.exp (Complex.I * α) = (4:ℝ)/5 + Complex.I * (3:ℝ)/5 →
  Complex.exp (Complex.I * β) = -(5:ℝ)/13 + Complex.I * (12:ℝ)/13 →
  Real.cos (α + β) = -(7:ℝ)/13 := by sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l1075_107545


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1075_107559

theorem smallest_integer_with_remainders :
  ∃ (x : ℕ), x > 0 ∧
  x % 6 = 5 ∧
  x % 7 = 6 ∧
  x % 8 = 7 ∧
  ∀ (y : ℕ), y > 0 →
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1075_107559


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1075_107598

theorem cubic_equation_solution (a : ℝ) : a^3 = 21 * 25 * 35 * 63 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1075_107598


namespace NUMINAMATH_CALUDE_kendra_age_l1075_107533

/-- Proves that Kendra's age is 18 given the conditions in the problem -/
theorem kendra_age :
  ∀ (k s t : ℕ), -- k: Kendra's age, s: Sam's age, t: Sue's age
  s = 2 * t →    -- Sam is twice as old as Sue
  k = 3 * s →    -- Kendra is 3 times as old as Sam
  (k + 3) + (s + 3) + (t + 3) = 36 → -- Their total age in 3 years will be 36
  k = 18 := by
    sorry -- Proof omitted

end NUMINAMATH_CALUDE_kendra_age_l1075_107533


namespace NUMINAMATH_CALUDE_min_square_edge_for_circle_l1075_107528

-- Define the circumference of the circle
def circle_circumference : ℝ := 31.4

-- Define π as an approximation
def π : ℝ := 3.14

-- Define the theorem
theorem min_square_edge_for_circle :
  ∃ (edge_length : ℝ), 
    edge_length = circle_circumference / π ∧ 
    edge_length = 10 := by sorry

end NUMINAMATH_CALUDE_min_square_edge_for_circle_l1075_107528


namespace NUMINAMATH_CALUDE_roots_shifted_polynomial_l1075_107520

theorem roots_shifted_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 22*x + 19 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_shifted_polynomial_l1075_107520


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l1075_107527

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n^2 % 2 = 0 → n^2 % 3 = 0 → n^2 % 5 = 0 → n^2 ≥ 225 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l1075_107527


namespace NUMINAMATH_CALUDE_triangle_properties_l1075_107530

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  b = a * Real.cos C + (Real.sqrt 3 / 3) * a * Real.sin C →
  a = 2 →
  b + c ≥ 4 →
  A = π / 3 ∧ (1 / 2) * a * b * Real.sin C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1075_107530


namespace NUMINAMATH_CALUDE_log_equation_solution_l1075_107599

theorem log_equation_solution :
  ∃ x : ℝ, (Real.log x + 3 * Real.log 2 - 4 * Real.log 5 = 1) ∧ (x = 781.25) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1075_107599


namespace NUMINAMATH_CALUDE_equation_solutions_count_l1075_107518

open Real

theorem equation_solutions_count (a : ℝ) (h : a < 0) :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x ∈ s, -π < x ∧ x < π ∧ 
    (a - 1) * (sin (2 * x) + cos x) + (a - 1) * (sin x - cos (2 * x)) = 0) ∧
  (∀ x, -π < x → x < π → 
    (a - 1) * (sin (2 * x) + cos x) + (a - 1) * (sin x - cos (2 * x)) = 0 → x ∈ s) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l1075_107518


namespace NUMINAMATH_CALUDE_largest_band_size_l1075_107579

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Represents the band and its formations --/
structure Band where
  totalMembers : ℕ
  firstFormation : BandFormation
  secondFormation : BandFormation

/-- Checks if a band satisfies all given conditions --/
def satisfiesConditions (band : Band) : Prop :=
  band.totalMembers < 100 ∧
  band.totalMembers = band.firstFormation.rows * band.firstFormation.membersPerRow + 3 ∧
  band.totalMembers = band.secondFormation.rows * band.secondFormation.membersPerRow ∧
  band.secondFormation.rows = band.firstFormation.rows - 3 ∧
  band.secondFormation.membersPerRow = band.firstFormation.membersPerRow + 1

/-- The theorem stating that 75 is the largest possible number of band members --/
theorem largest_band_size :
  ∀ band : Band, satisfiesConditions band → band.totalMembers ≤ 75 :=
by sorry

end NUMINAMATH_CALUDE_largest_band_size_l1075_107579


namespace NUMINAMATH_CALUDE_animal_ages_sum_l1075_107558

/-- Represents the ages of the animals in the problem -/
structure AnimalAges where
  porcupine : ℕ
  owl : ℕ
  lion : ℕ

/-- Defines the conditions given in the problem -/
def valid_ages (ages : AnimalAges) : Prop :=
  ages.owl = 2 * ages.porcupine ∧
  ages.owl = ages.lion + 2 ∧
  ages.lion = ages.porcupine + 4

/-- The theorem to be proved -/
theorem animal_ages_sum (ages : AnimalAges) :
  valid_ages ages → ages.porcupine + ages.owl + ages.lion = 28 := by
  sorry


end NUMINAMATH_CALUDE_animal_ages_sum_l1075_107558


namespace NUMINAMATH_CALUDE_max_value_inequality_l1075_107566

theorem max_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ k : ℝ, (a + b + c) * (1 / a + 1 / (b + c)) ≥ k) ↔ k ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1075_107566


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1075_107593

theorem sum_of_fractions : 
  (1 : ℚ) / 3 + (1 : ℚ) / 2 + (-5 : ℚ) / 6 + (1 : ℚ) / 5 + (1 : ℚ) / 4 + (-9 : ℚ) / 20 + (-9 : ℚ) / 20 = (-9 : ℚ) / 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1075_107593


namespace NUMINAMATH_CALUDE_root_differences_ratio_l1075_107596

open Real

/-- Given quadratic trinomials and their root differences, prove the ratio of differences squared is 3 -/
theorem root_differences_ratio (a b : ℝ) : 
  let f₁ := fun x : ℝ => x^2 + a*x + 3
  let f₂ := fun x : ℝ => x^2 + 2*x - b
  let f₃ := fun x : ℝ => x^2 + 2*(a-1)*x + b + 6
  let f₄ := fun x : ℝ => x^2 + (4-a)*x - 2*b - 3
  let A := sqrt (a^2 - 12)
  let B := sqrt (4 + 4*b)
  let C := sqrt (4*a^2 - 8*a - 4*b - 20)
  let D := sqrt (a^2 - 8*a + 8*b + 28)
  A^2 ≠ B^2 →
  (C^2 - D^2) / (A^2 - B^2) = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_root_differences_ratio_l1075_107596


namespace NUMINAMATH_CALUDE_charles_total_earnings_l1075_107561

/-- Calculates Charles's total earnings from housesitting and dog walking -/
def charles_earnings (housesit_rate : ℕ) (dog_walk_rate : ℕ) (housesit_hours : ℕ) (dogs_walked : ℕ) : ℕ :=
  housesit_rate * housesit_hours + dog_walk_rate * dogs_walked

/-- Theorem stating that Charles's earnings are $216 given the specified rates and hours -/
theorem charles_total_earnings :
  charles_earnings 15 22 10 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_charles_total_earnings_l1075_107561


namespace NUMINAMATH_CALUDE_sara_quarters_l1075_107506

def cents : ℕ := 275
def cents_per_quarter : ℕ := 25

theorem sara_quarters : cents / cents_per_quarter = 11 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l1075_107506


namespace NUMINAMATH_CALUDE_rectangles_on_grid_l1075_107586

/-- The number of rectangles on a 4x4 grid with 5 points in each direction -/
def num_rectangles : ℕ := 100

/-- The number of points in each direction of the grid -/
def points_per_direction : ℕ := 5

/-- Theorem stating the number of rectangles on the grid -/
theorem rectangles_on_grid : 
  (Nat.choose points_per_direction 2) * (Nat.choose points_per_direction 2) = num_rectangles := by
  sorry

end NUMINAMATH_CALUDE_rectangles_on_grid_l1075_107586


namespace NUMINAMATH_CALUDE_bryden_receives_correct_amount_l1075_107517

/-- The amount Bryden receives for his state quarters -/
def bryden_amount (num_quarters : ℕ) (face_value : ℚ) (percentage : ℕ) (bonus_per_five : ℚ) : ℚ :=
  let base_amount := (num_quarters : ℚ) * face_value * (percentage : ℚ) / 100
  let num_bonuses := num_quarters / 5
  base_amount + (num_bonuses : ℚ) * bonus_per_five

/-- Theorem stating that Bryden will receive $45.75 for his seven state quarters -/
theorem bryden_receives_correct_amount :
  bryden_amount 7 0.25 2500 2 = 45.75 := by
  sorry

end NUMINAMATH_CALUDE_bryden_receives_correct_amount_l1075_107517


namespace NUMINAMATH_CALUDE_local_max_value_l1075_107563

/-- The function f(x) = x³ - 12x --/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- m is the point of local maximum for f --/
def is_local_max (m : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - m| < δ → f x ≤ f m

theorem local_max_value :
  ∃ m : ℝ, is_local_max m ∧ m = -2 :=
sorry

end NUMINAMATH_CALUDE_local_max_value_l1075_107563


namespace NUMINAMATH_CALUDE_sum_of_squares_from_means_l1075_107525

theorem sum_of_squares_from_means (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 2.5) :
  x^2 + y^2 + z^2 = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_from_means_l1075_107525


namespace NUMINAMATH_CALUDE_tens_digit_of_6_pow_19_l1075_107513

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_6_pow_19 : tens_digit (6^19) = 9 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_pow_19_l1075_107513


namespace NUMINAMATH_CALUDE_distributive_property_l1075_107532

theorem distributive_property (x y : ℝ) : x * (1 + y) = x + x * y := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_l1075_107532


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1075_107568

/-- Two arithmetic sequences a and b with their respective sums A and B -/
def arithmetic_sequences (a b : ℕ → ℚ) (A B : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, 
    (∃ d₁ d₂ : ℚ, ∀ k : ℕ, a (k + 1) = a k + d₁ ∧ b (k + 1) = b k + d₂) ∧
    A n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1) ∧
    B n = n * b 1 + n * (n - 1) / 2 * (b 2 - b 1)

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) (A B : ℕ → ℚ) 
  (h : arithmetic_sequences a b A B) 
  (h_ratio : ∀ n : ℕ, A n / B n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, a n / b n = (4 * n - 3) / (6 * n - 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1075_107568


namespace NUMINAMATH_CALUDE_tank_full_time_l1075_107531

/-- Represents the state of a water tank system -/
structure TankSystem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℤ

/-- Calculates the time needed to fill the tank -/
def time_to_fill (system : TankSystem) : ℕ :=
  sorry

/-- Theorem stating that the tank will be full after 56 minutes -/
theorem tank_full_time (system : TankSystem) 
  (h1 : system.capacity = 950)
  (h2 : system.pipeA_rate = 40)
  (h3 : system.pipeB_rate = 30)
  (h4 : system.pipeC_rate = -20) : 
  time_to_fill system = 56 :=
  sorry

end NUMINAMATH_CALUDE_tank_full_time_l1075_107531


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1075_107595

/-- Represents an ellipse with equation kx^2 + y^2 = 2 and focus on x-axis -/
structure Ellipse (k : ℝ) where
  equation : ∀ (x y : ℝ), k * x^2 + y^2 = 2
  focus_on_x_axis : True  -- This is a placeholder for the focus condition

/-- The range of k for an ellipse with equation kx^2 + y^2 = 2 and focus on x-axis is (0, 1) -/
theorem ellipse_k_range (k : ℝ) (e : Ellipse k) : 0 < k ∧ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1075_107595


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1075_107519

/-- The eccentricity of the hyperbola 9y² - 16x² = 144 is 5/4 -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 5/4 ∧ 
  ∀ (x y : ℝ), 9*y^2 - 16*x^2 = 144 → 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧
    y^2/a^2 - x^2/b^2 = 1 ∧
    c^2 = a^2 + b^2 ∧
    e = c/a :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1075_107519
