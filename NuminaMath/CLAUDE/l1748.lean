import Mathlib

namespace NUMINAMATH_CALUDE_prob_one_head_two_tails_l1748_174823

/-- The probability of getting one head and two tails when tossing three fair coins -/
theorem prob_one_head_two_tails : ℝ := by
  -- Define the number of possible outcomes when tossing three fair coins
  let total_outcomes : ℕ := 2^3

  -- Define the number of ways to get one head and two tails
  let favorable_outcomes : ℕ := 3

  -- Define the probability as the ratio of favorable outcomes to total outcomes
  let probability : ℝ := favorable_outcomes / total_outcomes

  -- Prove that this probability equals 3/8
  sorry

end NUMINAMATH_CALUDE_prob_one_head_two_tails_l1748_174823


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l1748_174814

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15) ∧ 
  x₁ ≠ x₂ ∧ 
  |x₁ - x₂| = 30 := by
sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l1748_174814


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_two_l1748_174862

theorem cube_root_sum_equals_two (x : ℝ) (h1 : x > 0) 
  (h2 : (2 - x^3)^(1/3) + (2 + x^3)^(1/3) = 2) : x^6 = 100/27 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_two_l1748_174862


namespace NUMINAMATH_CALUDE_sarah_bottle_caps_l1748_174802

/-- The total number of bottle caps Sarah has at the end of the week -/
def total_bottle_caps (initial : ℕ) (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  initial + day1 + day2 + day3

/-- Theorem stating that Sarah's total bottle caps at the end of the week
    is equal to her initial count plus all purchased bottle caps -/
theorem sarah_bottle_caps : 
  total_bottle_caps 450 175 95 220 = 940 := by
  sorry

end NUMINAMATH_CALUDE_sarah_bottle_caps_l1748_174802


namespace NUMINAMATH_CALUDE_max_value_product_l1748_174846

theorem max_value_product (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x/y + y/z + z/x + y/x + z/y + x/z = 9) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≤ 81/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l1748_174846


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1748_174853

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1748_174853


namespace NUMINAMATH_CALUDE_inequality_proof_l1748_174848

theorem inequality_proof (n : ℕ+) (k : ℝ) (hk : k > 0) :
  1 - 1/k ≤ n * (k^(1/n : ℝ) - 1) ∧ n * (k^(1/n : ℝ) - 1) ≤ k - 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1748_174848


namespace NUMINAMATH_CALUDE_best_of_three_match_probability_l1748_174847

/-- The probability of player A winning a single game against player B. -/
def p_win_game : ℚ := 1/3

/-- The probability of player A winning a best-of-three match against player B. -/
def p_win_match : ℚ := 7/27

/-- Theorem stating that if the probability of player A winning each game is 1/3,
    then the probability of A winning a best-of-three match is 7/27. -/
theorem best_of_three_match_probability :
  p_win_game = 1/3 → p_win_match = 7/27 := by sorry

end NUMINAMATH_CALUDE_best_of_three_match_probability_l1748_174847


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l1748_174885

def f (x : ℝ) : ℝ := x^2014

theorem triangle_angle_inequality (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : α + β > π/2) : 
  f (Real.sin α) > f (Real.cos β) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l1748_174885


namespace NUMINAMATH_CALUDE_ratio_equals_seven_l1748_174881

theorem ratio_equals_seven (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - 2 * z = 0)
  (eq2 : 2 * x + 6 * y - 21 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 4*x*y) / (y^2 + z^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equals_seven_l1748_174881


namespace NUMINAMATH_CALUDE_competition_result_competition_result_proof_l1748_174809

-- Define the type for students
inductive Student : Type
  | A | B | C | D | E

-- Define a type for the competition order
def CompetitionOrder := List Student

-- Define the first person's prediction
def firstPrediction : CompetitionOrder :=
  [Student.A, Student.B, Student.C, Student.D, Student.E]

-- Define the second person's prediction
def secondPrediction : CompetitionOrder :=
  [Student.D, Student.A, Student.E, Student.C, Student.B]

-- Function to check if a student is in the correct position
def correctPosition (actual : CompetitionOrder) (predicted : CompetitionOrder) (index : Nat) : Prop :=
  actual.get? index = predicted.get? index

-- Function to check if adjacent pairs are correct
def correctAdjacentPair (actual : CompetitionOrder) (predicted : CompetitionOrder) (index : Nat) : Prop :=
  actual.get? index = predicted.get? index ∧ actual.get? (index + 1) = predicted.get? (index + 1)

-- Main theorem
theorem competition_result (actual : CompetitionOrder) : Prop :=
  (actual.length = 5) ∧
  (∀ i, i < 5 → ¬correctPosition actual firstPrediction i) ∧
  (∀ i, i < 4 → ¬correctAdjacentPair actual firstPrediction i) ∧
  ((correctPosition actual secondPrediction 0 ∧ correctPosition actual secondPrediction 1) ∨
   (correctPosition actual secondPrediction 1 ∧ correctPosition actual secondPrediction 2) ∨
   (correctPosition actual secondPrediction 2 ∧ correctPosition actual secondPrediction 3) ∨
   (correctPosition actual secondPrediction 3 ∧ correctPosition actual secondPrediction 4)) ∧
  ((correctAdjacentPair actual secondPrediction 0 ∧ correctAdjacentPair actual secondPrediction 2) ∨
   (correctAdjacentPair actual secondPrediction 0 ∧ correctAdjacentPair actual secondPrediction 3) ∨
   (correctAdjacentPair actual secondPrediction 1 ∧ correctAdjacentPair actual secondPrediction 3)) ∧
  (actual = [Student.E, Student.D, Student.A, Student.C, Student.B])

-- Proof of the theorem
theorem competition_result_proof : ∃ actual, competition_result actual := by
  sorry


end NUMINAMATH_CALUDE_competition_result_competition_result_proof_l1748_174809


namespace NUMINAMATH_CALUDE_triangle_property_l1748_174829

-- Define the necessary types and structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Line :=
  (p1 p2 : Point)

-- Define the given conditions
def isAcute (t : Triangle) : Prop := sorry

def isOrthocenter (H : Point) (t : Triangle) : Prop := sorry

def lieOnSide (P : Point) (l : Line) : Prop := sorry

def angleEquals (A B C : Point) (angle : ℝ) : Prop := sorry

def intersectsAt (l1 l2 : Line) (P : Point) : Prop := sorry

def isCircumcenter (O : Point) (t : Triangle) : Prop := sorry

def sameSideAs (P Q : Point) (l : Line) : Prop := sorry

def collinear (P Q R : Point) : Prop := sorry

-- Define the theorem
theorem triangle_property 
  (ABC : Triangle) 
  (H M N P Q O E : Point) :
  isAcute ABC →
  isOrthocenter H ABC →
  lieOnSide M (Line.mk ABC.A ABC.B) →
  lieOnSide N (Line.mk ABC.A ABC.C) →
  angleEquals H M ABC.B (60 : ℝ) →
  angleEquals H N ABC.C (60 : ℝ) →
  intersectsAt (Line.mk H M) (Line.mk ABC.C ABC.A) P →
  intersectsAt (Line.mk H N) (Line.mk ABC.B ABC.A) Q →
  isCircumcenter O (Triangle.mk H M N) →
  angleEquals E ABC.B ABC.C (60 : ℝ) →
  sameSideAs E ABC.A (Line.mk ABC.B ABC.C) →
  collinear E O H →
  (Line.mk O H).p1 = (Line.mk P Q).p1 ∧ -- OH ⊥ PQ
  (Triangle.mk E ABC.B ABC.C).A = (Triangle.mk E ABC.B ABC.C).B ∧ 
  (Triangle.mk E ABC.B ABC.C).B = (Triangle.mk E ABC.B ABC.C).C -- Triangle EBC is equilateral
  := by sorry

end NUMINAMATH_CALUDE_triangle_property_l1748_174829


namespace NUMINAMATH_CALUDE_line_parameterization_l1748_174804

/-- Given a line y = 2x - 15 parameterized by (x, y) = (g(t), 10t + 5), prove that g(t) = 5t + 10 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t : ℝ, 10 * t + 5 = 2 * (g t) - 15) → 
  (∀ t : ℝ, g t = 5 * t + 10) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l1748_174804


namespace NUMINAMATH_CALUDE_min_sum_squares_l1748_174860

theorem min_sum_squares (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  ∃ (m : ℝ), m = 9 ∧ ∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 3 → a^2 + b^2 + c^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1748_174860


namespace NUMINAMATH_CALUDE_simplify_expression_l1748_174855

theorem simplify_expression (p : ℝ) (h1 : 1 < p) (h2 : p < 2) :
  Real.sqrt ((1 - p)^2) + (Real.sqrt (2 - p))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1748_174855


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1748_174824

/-- Represents a parabola y^2 = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- A point on the parabola -/
structure PointOnParabola (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * para.p * x

theorem parabola_focus_distance (para : Parabola) 
  (A : PointOnParabola para) (h_x : A.x = 2) (h_dist : Real.sqrt ((A.x - para.p/2)^2 + A.y^2) = 6) :
  para.p = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1748_174824


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1748_174852

/-- The area of the union of a square with side length 8 and a circle with radius 8
    centered at one of the square's vertices is 64 + 48π square units. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 8
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  let overlap_area := (1/4 : ℝ) * circle_area
  square_area + circle_area - overlap_area = 64 + 48 * π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1748_174852


namespace NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l1748_174877

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - Complex.I * 7) = 15) :
  ∃ (w : ℂ), Complex.abs w = 56 / 15 ∧ ∀ (v : ℂ), Complex.abs (v - 8) + Complex.abs (v - Complex.I * 7) = 15 → Complex.abs w ≤ Complex.abs v :=
by sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l1748_174877


namespace NUMINAMATH_CALUDE_monomial_division_l1748_174818

theorem monomial_division (x : ℝ) : 2 * x^3 / x^2 = 2 * x := by sorry

end NUMINAMATH_CALUDE_monomial_division_l1748_174818


namespace NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l1748_174861

/-- A police emergency number is a positive integer that ends with 133 in decimal representation. -/
def is_police_emergency_number (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7. -/
theorem police_emergency_number_has_large_prime_divisor (n : ℕ) :
  is_police_emergency_number n → ∃ p : ℕ, p > 7 ∧ Nat.Prime p ∧ p ∣ n :=
sorry

end NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l1748_174861


namespace NUMINAMATH_CALUDE_solve_for_a_l1748_174872

theorem solve_for_a : ∃ a : ℝ, (1/2 * 2 + a = -1) ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1748_174872


namespace NUMINAMATH_CALUDE_fraction_value_l1748_174869

/-- Represents the numerator of the fraction as a function of k -/
def numerator (k : ℕ) : ℕ := 10^k + 6 * (10^k - 1) / 9

/-- Represents the denominator of the fraction as a function of k -/
def denominator (k : ℕ) : ℕ := 60 * (10^k - 1) / 9 + 4

/-- The main theorem stating that the fraction is always 1/4 for any positive k -/
theorem fraction_value (k : ℕ) (h : k > 0) : 
  (numerator k : ℚ) / (denominator k : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1748_174869


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l1748_174822

/-- The original price of the coat -/
def original_price : ℝ := 120

/-- The first discount percentage -/
def first_discount : ℝ := 0.25

/-- The second discount percentage -/
def second_discount : ℝ := 0.20

/-- The final price after both discounts -/
def final_price : ℝ := 72

/-- Theorem stating that applying the two discounts sequentially results in the final price -/
theorem discounted_price_calculation :
  (1 - second_discount) * ((1 - first_discount) * original_price) = final_price :=
by sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l1748_174822


namespace NUMINAMATH_CALUDE_twenty_photos_needed_l1748_174830

/-- The minimum number of non-overlapping rectangular photos required to form a square -/
def min_photos_for_square (width : ℕ) (length : ℕ) : ℕ :=
  let square_side := Nat.lcm width length
  (square_side * square_side) / (width * length)

/-- Theorem stating that 20 photos of 12cm x 15cm are needed for the smallest square -/
theorem twenty_photos_needed : min_photos_for_square 12 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_photos_needed_l1748_174830


namespace NUMINAMATH_CALUDE_post_office_problem_l1748_174835

theorem post_office_problem (total_spent : ℚ) (letter_cost : ℚ) (package_cost : ℚ) 
  (h1 : total_spent = 449/100)
  (h2 : letter_cost = 37/100)
  (h3 : package_cost = 88/100)
  : ∃ (letters packages : ℕ), 
    letters = packages + 2 ∧ 
    letter_cost * letters + package_cost * packages = total_spent ∧
    letters = 5 := by
  sorry

end NUMINAMATH_CALUDE_post_office_problem_l1748_174835


namespace NUMINAMATH_CALUDE_position_2007_l1748_174889

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | DCBA
  | ADCB
  | BADC
  | CBAD

-- Define the transformation function
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ADCB
  | SquarePosition.ADCB => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.CBAD
  | SquarePosition.CBAD => SquarePosition.ABCD

-- Define the function to get the position after n transformations
def positionAfterN (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.DCBA
  | 2 => SquarePosition.ADCB
  | _ => SquarePosition.BADC

-- Theorem statement
theorem position_2007 : positionAfterN 2007 = SquarePosition.ADCB := by
  sorry


end NUMINAMATH_CALUDE_position_2007_l1748_174889


namespace NUMINAMATH_CALUDE_least_number_divisible_l1748_174894

theorem least_number_divisible (n : ℕ) : n = 857 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 24 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 32 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 36 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 54 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 7) = 24 * k₁ ∧ (n + 7) = 32 * k₂ ∧ (n + 7) = 36 * k₃ ∧ (n + 7) = 54 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_l1748_174894


namespace NUMINAMATH_CALUDE_calculation_proof_l1748_174888

theorem calculation_proof : (1/4 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * 72 + (1/8 : ℚ) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1748_174888


namespace NUMINAMATH_CALUDE_no_time_left_after_student_council_l1748_174849

/-- Represents the journey to school with various stops -/
structure SchoolJourney where
  totalTimeAvailable : ℕ
  travelTimeWithTraffic : ℕ
  timeToLibrary : ℕ
  timeToReturnBooks : ℕ
  extraTimeForManyBooks : ℕ
  timeToStudentCouncil : ℕ
  timeToSubmitProject : ℕ
  timeToClassroom : ℕ

/-- Calculates the time left after leaving the student council room -/
def timeLeftAfterStudentCouncil (journey : SchoolJourney) : Int :=
  journey.totalTimeAvailable - (journey.travelTimeWithTraffic + journey.timeToLibrary +
  journey.timeToReturnBooks + journey.extraTimeForManyBooks + journey.timeToStudentCouncil +
  journey.timeToSubmitProject)

/-- Theorem stating that in the worst-case scenario, there's no time left after leaving the student council room -/
theorem no_time_left_after_student_council (journey : SchoolJourney)
  (h1 : journey.totalTimeAvailable = 30)
  (h2 : journey.travelTimeWithTraffic = 25)
  (h3 : journey.timeToLibrary = 3)
  (h4 : journey.timeToReturnBooks = 2)
  (h5 : journey.extraTimeForManyBooks = 2)
  (h6 : journey.timeToStudentCouncil = 5)
  (h7 : journey.timeToSubmitProject = 3)
  (h8 : journey.timeToClassroom = 6) :
  timeLeftAfterStudentCouncil journey ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_time_left_after_student_council_l1748_174849


namespace NUMINAMATH_CALUDE_max_y_value_l1748_174880

theorem max_y_value (x y : ℝ) (h : x^2 + y^2 = 18*x + 40*y) : 
  ∃ (max_y : ℝ), max_y = 20 + Real.sqrt 481 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 18*x' + 40*y' → y' ≤ max_y := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l1748_174880


namespace NUMINAMATH_CALUDE_number_times_24_equals_173_times_240_l1748_174826

theorem number_times_24_equals_173_times_240 : ∃ x : ℕ, x * 24 = 173 * 240 ∧ x = 1730 := by
  sorry

end NUMINAMATH_CALUDE_number_times_24_equals_173_times_240_l1748_174826


namespace NUMINAMATH_CALUDE_seminar_chairs_l1748_174876

/-- Converts a number from base 6 to base 10 -/
def base6ToDecimal (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 36 + tens * 6 + ones

/-- Calculates the number of chairs needed given the number of participants and participants per chair -/
def calculateChairs (participants : Nat) (participantsPerChair : Nat) : Nat :=
  (participants + participantsPerChair - 1) / participantsPerChair

theorem seminar_chairs :
  let participantsBase6 : Nat := 315
  let participantsPerChair : Nat := 3
  let participantsDecimal := base6ToDecimal participantsBase6
  calculateChairs participantsDecimal participantsPerChair = 40 := by
  sorry

end NUMINAMATH_CALUDE_seminar_chairs_l1748_174876


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l1748_174874

theorem max_value_of_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  (∀ x' y', -6 ≤ x' ∧ x' ≤ -3 → 3 ≤ y' ∧ y' ≤ 5 → (x' - y') / y' ≤ (x - y) / y) →
  (x - y) / y = -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l1748_174874


namespace NUMINAMATH_CALUDE_sin_cos_sum_11_19_l1748_174834

theorem sin_cos_sum_11_19 : 
  Real.sin (11 * π / 180) * Real.cos (19 * π / 180) + 
  Real.cos (11 * π / 180) * Real.sin (19 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_11_19_l1748_174834


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l1748_174821

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 2|

-- State the theorem
theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_for_inequality_l1748_174821


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l1748_174898

def triangle_perimeter (a b c : ℕ) : ℕ := a + b + c

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem least_perimeter_triangle :
  ∃ (c : ℕ), 
    is_triangle 24 51 c ∧ 
    (∀ (x : ℕ), is_triangle 24 51 x → triangle_perimeter 24 51 c ≤ triangle_perimeter 24 51 x) ∧
    triangle_perimeter 24 51 c = 103 := by
  sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l1748_174898


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l1748_174883

theorem square_garden_perimeter (q p : ℝ) : 
  q = 49 → -- Area of the garden is 49 square feet
  q = p + 21 → -- Given relationship between q and p
  (4 * Real.sqrt q) = 28 -- Perimeter of the garden is 28 feet
:= by sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l1748_174883


namespace NUMINAMATH_CALUDE_kiley_ate_quarter_cheesecake_l1748_174812

/-- Represents the properties of a cheesecake and Kiley's consumption -/
structure CheesecakeConsumption where
  calories_per_slice : ℕ
  total_calories : ℕ
  slices_eaten : ℕ

/-- Calculates the percentage of cheesecake eaten -/
def percentage_eaten (c : CheesecakeConsumption) : ℚ :=
  (c.calories_per_slice * c.slices_eaten : ℚ) / c.total_calories * 100

/-- Theorem stating that Kiley ate 25% of the cheesecake -/
theorem kiley_ate_quarter_cheesecake :
  let c : CheesecakeConsumption := {
    calories_per_slice := 350,
    total_calories := 2800,
    slices_eaten := 2
  }
  percentage_eaten c = 25 := by
  sorry


end NUMINAMATH_CALUDE_kiley_ate_quarter_cheesecake_l1748_174812


namespace NUMINAMATH_CALUDE_octal_to_decimal_l1748_174854

theorem octal_to_decimal (r : ℕ) : 175 = 120 + r → r = 5 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_l1748_174854


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l1748_174805

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) :
  ∃ (m : ℝ), m = -1 ∧ ∀ x, (8 * x^2 + 10 * x + 6 = 2) → (3 * x + 2 ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l1748_174805


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1748_174825

def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

def last_term (seq : List ℕ) : ℕ :=
  match seq.getLast? with
  | some x => x
  | none => 0

theorem arithmetic_sequence_sum (a d : ℕ) :
  ∀ seq : List ℕ, seq = arithmetic_sequence a d seq.length →
  last_term seq = 50 →
  seq.sum = 442 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1748_174825


namespace NUMINAMATH_CALUDE_number_of_boys_in_school_l1748_174815

theorem number_of_boys_in_school : 
  ∃ (x : ℕ), 
    (x + (x * 900 / 100) = 900) ∧ 
    (x = 90) := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_in_school_l1748_174815


namespace NUMINAMATH_CALUDE_cottage_cheese_production_l1748_174866

/-- Represents the fat content balance in milk processing -/
def fat_balance (milk_mass : ℝ) (milk_fat : ℝ) (cheese_fat : ℝ) (whey_fat : ℝ) (cheese_mass : ℝ) : Prop :=
  milk_mass * milk_fat = cheese_mass * cheese_fat + (milk_mass - cheese_mass) * whey_fat

/-- Proves the amount of cottage cheese produced from milk -/
theorem cottage_cheese_production (milk_mass : ℝ) (milk_fat : ℝ) (cheese_fat : ℝ) (whey_fat : ℝ) 
  (h_milk_mass : milk_mass = 1)
  (h_milk_fat : milk_fat = 0.05)
  (h_cheese_fat : cheese_fat = 0.155)
  (h_whey_fat : whey_fat = 0.005) :
  ∃ cheese_mass : ℝ, cheese_mass = 0.3 ∧ fat_balance milk_mass milk_fat cheese_fat whey_fat cheese_mass :=
by
  sorry

#check cottage_cheese_production

end NUMINAMATH_CALUDE_cottage_cheese_production_l1748_174866


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_difference_l1748_174832

theorem prime_arithmetic_sequence_difference (p₁ p₂ p₃ d : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
  p₁ > 3 ∧ p₂ > 3 ∧ p₃ > 3 ∧
  p₂ = p₁ + d ∧ p₃ = p₂ + d →
  6 ∣ d := by
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_difference_l1748_174832


namespace NUMINAMATH_CALUDE_perpendicular_chords_intersection_distance_l1748_174890

theorem perpendicular_chords_intersection_distance (d r : ℝ) (AB CD : ℝ) (h1 : d = 10) (h2 : r = d / 2) (h3 : AB = 9) (h4 : CD = 8) :
  let S := r^2 - (AB/2)^2
  let R := r^2 - (CD/2)^2
  (S + R).sqrt = (55 : ℝ).sqrt / 2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_chords_intersection_distance_l1748_174890


namespace NUMINAMATH_CALUDE_max_underwear_is_four_l1748_174845

/-- Represents the washing machine and clothing weights --/
structure WashingMachine where
  limit : Nat
  sock_weight : Nat
  underwear_weight : Nat
  shirt_weight : Nat
  shorts_weight : Nat
  pants_weight : Nat

/-- Represents the clothes Tony is washing --/
structure ClothesInWash where
  pants : Nat
  shirts : Nat
  shorts : Nat
  socks : Nat

/-- Calculates the maximum number of additional pairs of underwear that can be added --/
def max_additional_underwear (wm : WashingMachine) (clothes : ClothesInWash) : Nat :=
  let current_weight := 
    clothes.pants * wm.pants_weight +
    clothes.shirts * wm.shirt_weight +
    clothes.shorts * wm.shorts_weight +
    clothes.socks * wm.sock_weight
  let remaining_weight := wm.limit - current_weight
  remaining_weight / wm.underwear_weight

/-- Theorem stating that the maximum number of additional pairs of underwear is 4 --/
theorem max_underwear_is_four :
  let wm : WashingMachine := {
    limit := 50,
    sock_weight := 2,
    underwear_weight := 4,
    shirt_weight := 5,
    shorts_weight := 8,
    pants_weight := 10
  }
  let clothes : ClothesInWash := {
    pants := 1,
    shirts := 2,
    shorts := 1,
    socks := 3
  }
  max_additional_underwear wm clothes = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_underwear_is_four_l1748_174845


namespace NUMINAMATH_CALUDE_log_x2y2_value_l1748_174884

theorem log_x2y2_value (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 2) :
  Real.log (x^2 * y^2) = 12/5 := by sorry

end NUMINAMATH_CALUDE_log_x2y2_value_l1748_174884


namespace NUMINAMATH_CALUDE_betty_order_total_payment_l1748_174828

/-- Calculates the total payment for Betty's order including shipping -/
def totalPayment (
  slipperPrice : Float) (slipperWeight : Float) (slipperCount : Nat)
  (lipstickPrice : Float) (lipstickWeight : Float) (lipstickCount : Nat)
  (hairColorPrice : Float) (hairColorWeight : Float) (hairColorCount : Nat)
  (sunglassesPrice : Float) (sunglassesWeight : Float) (sunglassesCount : Nat)
  (tshirtPrice : Float) (tshirtWeight : Float) (tshirtCount : Nat)
  : Float :=
  let totalCost := 
    slipperPrice * slipperCount.toFloat +
    lipstickPrice * lipstickCount.toFloat +
    hairColorPrice * hairColorCount.toFloat +
    sunglassesPrice * sunglassesCount.toFloat +
    tshirtPrice * tshirtCount.toFloat
  let totalWeight :=
    slipperWeight * slipperCount.toFloat +
    lipstickWeight * lipstickCount.toFloat +
    hairColorWeight * hairColorCount.toFloat +
    sunglassesWeight * sunglassesCount.toFloat +
    tshirtWeight * tshirtCount.toFloat
  let shippingCost :=
    if totalWeight ≤ 5 then 2
    else if totalWeight ≤ 10 then 4
    else 6
  totalCost + shippingCost

theorem betty_order_total_payment :
  totalPayment 2.5 0.3 6 1.25 0.05 4 3 0.2 8 5.75 0.1 3 12.25 0.5 4 = 114.25 := by
  sorry

end NUMINAMATH_CALUDE_betty_order_total_payment_l1748_174828


namespace NUMINAMATH_CALUDE_complex_division_equality_l1748_174839

theorem complex_division_equality : (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l1748_174839


namespace NUMINAMATH_CALUDE_vehicles_meeting_time_l1748_174844

-- Define the vehicles
structure Vehicle where
  id : Nat
  speed : ℝ

-- Define the meeting points
structure MeetingPoint where
  vehicle1 : Vehicle
  vehicle2 : Vehicle
  time : ℝ

-- Define the problem
theorem vehicles_meeting_time
  (v1 v2 v3 v4 : Vehicle)
  (m12 m13 m14 m24 m34 : MeetingPoint)
  (h1 : m12.vehicle1 = v1 ∧ m12.vehicle2 = v2 ∧ m12.time = 0)
  (h2 : m13.vehicle1 = v1 ∧ m13.vehicle2 = v3 ∧ m13.time = 220)
  (h3 : m14.vehicle1 = v1 ∧ m14.vehicle2 = v4 ∧ m14.time = 280)
  (h4 : m24.vehicle1 = v2 ∧ m24.vehicle2 = v4 ∧ m24.time = 240)
  (h5 : m34.vehicle1 = v3 ∧ m34.vehicle2 = v4 ∧ m34.time = 130)
  (h_constant_speed : ∀ v : Vehicle, v.speed > 0)
  : ∃ m23 : MeetingPoint, m23.vehicle1 = v2 ∧ m23.vehicle2 = v3 ∧ m23.time = 200 :=
sorry

end NUMINAMATH_CALUDE_vehicles_meeting_time_l1748_174844


namespace NUMINAMATH_CALUDE_bike_riders_count_l1748_174813

theorem bike_riders_count (total : ℕ) (difference : ℕ) :
  total = 676 →
  difference = 178 →
  ∃ (bikers hikers : ℕ),
    total = bikers + hikers ∧
    hikers = bikers + difference ∧
    bikers = 249 := by
  sorry

end NUMINAMATH_CALUDE_bike_riders_count_l1748_174813


namespace NUMINAMATH_CALUDE_correct_students_joined_l1748_174806

/-- The number of students who joined Beth's class -/
def students_joined : ℕ := 30

/-- The initial number of students -/
def initial_students : ℕ := 150

/-- The number of students who left in the final year -/
def students_left : ℕ := 15

/-- The final number of students -/
def final_students : ℕ := 165

/-- Theorem stating that the number of students who joined is correct -/
theorem correct_students_joined :
  initial_students + students_joined - students_left = final_students :=
by sorry

end NUMINAMATH_CALUDE_correct_students_joined_l1748_174806


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1748_174841

theorem quadratic_equation_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 9 / 2) : 
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1748_174841


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1748_174878

theorem polynomial_evaluation :
  let a : ℚ := 7/3
  (4 * a^2 - 11 * a + 7) * (2 * a - 3) = 140/27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1748_174878


namespace NUMINAMATH_CALUDE_circle_area_diameter_increase_l1748_174827

theorem circle_area_diameter_increase : 
  ∀ (A D A' D' : ℝ), 
  A > 0 → D > 0 → 
  A = (Real.pi / 4) * D^2 →
  A' = 4 * A →
  A' = (Real.pi / 4) * D'^2 →
  D' / D - 1 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_diameter_increase_l1748_174827


namespace NUMINAMATH_CALUDE_hockey_league_teams_l1748_174836

/-- The number of teams in a hockey league. -/
def num_teams : ℕ := 16

/-- The number of times each team faces every other team. -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season. -/
def total_games : ℕ := 1200

/-- Theorem stating that the number of teams is correct given the conditions. -/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) * games_per_pair) / 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_hockey_league_teams_l1748_174836


namespace NUMINAMATH_CALUDE_two_books_selection_ways_l1748_174819

/-- The number of ways to select two books of different subjects from three shelves -/
def select_two_books (chinese_books : ℕ) (math_books : ℕ) (english_books : ℕ) : ℕ :=
  chinese_books * math_books + chinese_books * english_books + math_books * english_books

/-- Theorem stating that selecting two books of different subjects from the given shelves results in 242 ways -/
theorem two_books_selection_ways :
  select_two_books 10 9 8 = 242 := by
  sorry

end NUMINAMATH_CALUDE_two_books_selection_ways_l1748_174819


namespace NUMINAMATH_CALUDE_no_primes_in_range_l1748_174879

theorem no_primes_in_range (n : ℕ) (hn : n > 2) :
  ∀ k : ℕ, n! + 2 < k ∧ k < n! + n → ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l1748_174879


namespace NUMINAMATH_CALUDE_snow_leopard_arrangements_l1748_174810

theorem snow_leopard_arrangements (n : ℕ) (h : n = 8) :
  2 * Nat.factorial (n - 2) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangements_l1748_174810


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l1748_174857

-- Define the universal set U
def U : Set ℝ := {x | x^2 - (5/2)*x + 1 ≥ 0}

-- Define set A
def A : Set ℝ := {x | |x - 1| > 1}

-- Define set B
def B : Set ℝ := {x | (x + 1)/(x - 2) ≥ 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | x ≤ -1 ∨ x > 2} := by sorry

-- Theorem for A ∪ (CᵤB)
theorem union_A_complement_B : A ∪ (U \ B) = U := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l1748_174857


namespace NUMINAMATH_CALUDE_min_value_theorem_l1748_174864

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  (x^2 + 3*x + 2) / x ≥ 2 * Real.sqrt 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1748_174864


namespace NUMINAMATH_CALUDE_battle_station_staffing_l1748_174817

theorem battle_station_staffing (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (n.factorial / (n - k).factorial) = 30240 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l1748_174817


namespace NUMINAMATH_CALUDE_wins_to_losses_ratio_l1748_174807

/-- Represents the statistics of a baseball team's season. -/
structure BaseballSeason where
  total_games : ℕ
  wins : ℕ
  losses : ℕ

/-- Defines the conditions for the baseball season. -/
def validSeason (s : BaseballSeason) : Prop :=
  s.total_games = 130 ∧
  s.wins = s.losses + 14 ∧
  s.wins = 101

/-- Theorem stating the ratio of wins to losses for the given conditions. -/
theorem wins_to_losses_ratio (s : BaseballSeason) (h : validSeason s) :
  s.wins = 101 ∧ s.losses = 87 := by
  sorry

#check wins_to_losses_ratio

end NUMINAMATH_CALUDE_wins_to_losses_ratio_l1748_174807


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_l1748_174851

theorem count_four_digit_numbers : 
  (Finset.range 4001).card = (Finset.Icc 1000 5000).card := by sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_l1748_174851


namespace NUMINAMATH_CALUDE_zoo_visitors_l1748_174895

theorem zoo_visitors (total_people : ℕ) (adult_price kid_price total_sales : ℚ)
  (h1 : total_people = 254)
  (h2 : adult_price = 28)
  (h3 : kid_price = 12)
  (h4 : total_sales = 3864) :
  ∃ (adults : ℕ), adults = 51 ∧
    ∃ (kids : ℕ), adults + kids = total_people ∧
      adult_price * adults + kid_price * kids = total_sales :=
by sorry

end NUMINAMATH_CALUDE_zoo_visitors_l1748_174895


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l1748_174868

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℤ) :
  arithmetic_sequence a → a 4 = -4 → a 8 = 4 → a 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l1748_174868


namespace NUMINAMATH_CALUDE_email_difference_is_six_l1748_174838

/-- Calculates the difference between morning and afternoon emails --/
def email_difference (early_morning late_morning early_afternoon late_afternoon : ℕ) : ℕ :=
  (early_morning + late_morning) - (early_afternoon + late_afternoon)

/-- Theorem stating the difference between morning and afternoon emails is 6 --/
theorem email_difference_is_six :
  email_difference 10 15 7 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_email_difference_is_six_l1748_174838


namespace NUMINAMATH_CALUDE_cubic_solution_sum_l1748_174837

theorem cubic_solution_sum (a b c : ℝ) : 
  (a^3 - 4*a^2 + 7*a = 15) ∧ 
  (b^3 - 4*b^2 + 7*b = 15) ∧ 
  (c^3 - 4*c^2 + 7*c = 15) →
  a*b/c + b*c/a + c*a/b = 49/15 := by
sorry

end NUMINAMATH_CALUDE_cubic_solution_sum_l1748_174837


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1748_174856

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 5 ∧ b = 12 ∧ c = 13

/-- A square inscribed in a right triangle with a vertex at the right angle -/
def squareAtRightAngle (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x < t.a ∧ x < t.b ∧ x / t.a = x / t.b

/-- A square inscribed in a right triangle with a side along the hypotenuse -/
def squareAlongHypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y < t.c ∧ (t.a - y)^2 + (t.b - y)^2 = y^2

theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ)
  (h1 : squareAtRightAngle t1 x) (h2 : squareAlongHypotenuse t2 y) :
  x / y = 144 / 85 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1748_174856


namespace NUMINAMATH_CALUDE_simplify_expression_l1748_174820

theorem simplify_expression (a : ℝ) (h : a < -3) :
  Real.sqrt ((2 * a - 1)^2) + Real.sqrt ((a + 3)^2) = -3 * a - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1748_174820


namespace NUMINAMATH_CALUDE_jane_egg_money_l1748_174899

/-- Calculates the money made from selling eggs over a period of weeks. -/
def money_from_eggs (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  (num_chickens * eggs_per_chicken * num_weeks : ℚ) / 12 * price_per_dozen

/-- Proves that Jane makes $20 in 2 weeks from selling eggs. -/
theorem jane_egg_money :
  money_from_eggs 10 6 2 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jane_egg_money_l1748_174899


namespace NUMINAMATH_CALUDE_plato_city_schools_l1748_174867

/-- The number of high schools in Plato City -/
def num_schools : ℕ := 21

/-- The total number of participants in the competition -/
def total_participants : ℕ := 3 * num_schools

/-- Charlie's rank in the competition -/
def charlie_rank : ℕ := (total_participants + 1) / 2

/-- Alice's rank in the competition -/
def alice_rank : ℕ := 45

/-- Bob's rank in the competition -/
def bob_rank : ℕ := 58

/-- Theorem stating that the number of schools satisfies all conditions -/
theorem plato_city_schools :
  num_schools = 21 ∧
  charlie_rank < alice_rank ∧
  charlie_rank < bob_rank ∧
  charlie_rank ≤ 45 ∧
  3 * num_schools ≥ bob_rank :=
sorry

end NUMINAMATH_CALUDE_plato_city_schools_l1748_174867


namespace NUMINAMATH_CALUDE_book_cost_price_l1748_174870

/-- The cost price of a book, given that selling it at 9% profit instead of 9% loss brings Rs 9 more -/
theorem book_cost_price (price : ℝ) : 
  (price * 1.09 - price * 0.91 = 9) → price = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l1748_174870


namespace NUMINAMATH_CALUDE_delta_phi_composition_l1748_174896

/-- Given two functions δ and φ, prove that δ(φ(x)) = 3 if and only if x = -19/20 -/
theorem delta_phi_composition (δ φ : ℝ → ℝ) (h1 : ∀ x, δ x = 4 * x + 6) (h2 : ∀ x, φ x = 5 * x + 4) :
  (∃ x, δ (φ x) = 3) ↔ (∃ x, x = -19/20) :=
by sorry

end NUMINAMATH_CALUDE_delta_phi_composition_l1748_174896


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1748_174800

-- Problem 1
theorem problem_1 : Real.sqrt 9 + |-2| - (-3)^2 + (π - 100)^0 = -3 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : 
  (x^2 + 1 = 5) ↔ (x = 2 ∨ x = -2) := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) :
  (x^2 = (x - 2)^2 + 7) ↔ (x = 11/4) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1748_174800


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_four_point_five_l1748_174843

theorem reciprocal_of_negative_four_point_five :
  ((-4.5)⁻¹ : ℝ) = -2/9 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_four_point_five_l1748_174843


namespace NUMINAMATH_CALUDE_derivative_of_f_l1748_174833

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x + x^2

theorem derivative_of_f :
  deriv f = λ x => 3 / x + 2 * x :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l1748_174833


namespace NUMINAMATH_CALUDE_smallest_m_divisibility_l1748_174811

theorem smallest_m_divisibility : ∃! m : ℕ,
  (∀ n : ℕ, Odd n → (148^n + m * 141^n) % 2023 = 0) ∧
  (∀ k : ℕ, k < m → ∃ n : ℕ, Odd n ∧ (148^n + k * 141^n) % 2023 ≠ 0) ∧
  m = 1735 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_divisibility_l1748_174811


namespace NUMINAMATH_CALUDE_root_sum_squares_l1748_174892

theorem root_sum_squares (a b c d : ℝ) : 
  (a^4 - 12*a^3 + 47*a^2 - 60*a + 24 = 0) →
  (b^4 - 12*b^3 + 47*b^2 - 60*b + 24 = 0) →
  (c^4 - 12*c^3 + 47*c^2 - 60*c + 24 = 0) →
  (d^4 - 12*d^3 + 47*d^2 - 60*d + 24 = 0) →
  (a+b)^2 + (b+c)^2 + (c+d)^2 + (d+a)^2 = 147 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_squares_l1748_174892


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l1748_174887

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 16

-- Define the line m that bisects the circle
def line_m (x y : ℝ) : Prop :=
  3*x - y = 0

-- Define the line l passing through D(0,-1) with slope k
def line_l (k x y : ℝ) : Prop :=
  y = k*x - 1

-- Theorem statement
theorem circle_and_line_intersection :
  -- Circle C passes through A(1,-1) and B(5,3)
  circle_C 1 (-1) ∧ circle_C 5 3 ∧
  -- Circle C is bisected by line m
  (∀ x y, circle_C x y → line_m x y → x = 1 ∧ y = 3) →
  -- Part 1: Prove the equation of circle C
  (∀ x y, circle_C x y ↔ (x - 1)^2 + (y - 3)^2 = 16) ∧
  -- Part 2: Prove the range of k for which line l intersects circle C at two distinct points
  (∀ k, (∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂) ↔
        (k < -8/15 ∨ k > 0)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l1748_174887


namespace NUMINAMATH_CALUDE_house_rooms_count_l1748_174897

/-- The number of rooms with 4 walls -/
def rooms_with_four_walls : ℕ := 5

/-- The number of rooms with 5 walls -/
def rooms_with_five_walls : ℕ := 4

/-- The number of walls each person should paint -/
def walls_per_person : ℕ := 8

/-- The number of people in Amanda's family -/
def family_members : ℕ := 5

/-- The total number of rooms in the house -/
def total_rooms : ℕ := rooms_with_four_walls + rooms_with_five_walls

theorem house_rooms_count : total_rooms = 9 := by
  sorry

end NUMINAMATH_CALUDE_house_rooms_count_l1748_174897


namespace NUMINAMATH_CALUDE_encounter_twelve_trams_l1748_174842

/-- Represents the tram system with given parameters -/
structure TramSystem where
  departure_interval : ℕ  -- Interval between tram departures in minutes
  journey_duration : ℕ    -- Duration of a full journey in minutes

/-- Calculates the number of trams encountered during a journey -/
def count_encountered_trams (system : TramSystem) : ℕ :=
  2 * (system.journey_duration / system.departure_interval)

/-- Theorem stating that in the given tram system, a passenger will encounter 12 trams -/
theorem encounter_twelve_trams (system : TramSystem) 
  (h1 : system.departure_interval = 10)
  (h2 : system.journey_duration = 60) : 
  count_encountered_trams system = 12 := by
  sorry

#eval count_encountered_trams ⟨10, 60⟩

end NUMINAMATH_CALUDE_encounter_twelve_trams_l1748_174842


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1748_174803

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b, b ≥ 0 → a^2 + b ≥ 0) ∧ 
  (∃ a b, a^2 + b ≥ 0 ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1748_174803


namespace NUMINAMATH_CALUDE_orange_tree_problem_l1748_174875

theorem orange_tree_problem (trees : ℕ) (picked_fraction : ℚ) (remaining : ℕ) :
  trees = 8 →
  picked_fraction = 2 / 5 →
  remaining = 960 →
  ∃ (initial : ℕ), initial = 200 ∧ 
    trees * (initial - picked_fraction * initial) = remaining :=
by sorry

end NUMINAMATH_CALUDE_orange_tree_problem_l1748_174875


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1748_174840

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x) + (9 / y) ≥ 16 ∧
  ((1 / x) + (9 / y) = 16 ↔ x = 1/4 ∧ y = 3/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1748_174840


namespace NUMINAMATH_CALUDE_system_solution_l1748_174816

theorem system_solution (x y : ℝ) : 
  (1 / (x^2 + y^2) + x^2 * y^2 = 5/4) ∧ 
  (2 * x^4 + 2 * y^4 + 5 * x^2 * y^2 = 9/4) ↔ 
  ((x = 1 / Real.sqrt 2 ∧ (y = 1 / Real.sqrt 2 ∨ y = -1 / Real.sqrt 2)) ∨
   (x = -1 / Real.sqrt 2 ∧ (y = 1 / Real.sqrt 2 ∨ y = -1 / Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1748_174816


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1748_174858

def M : Set Int := {-1, 0, 1}
def N : Set Int := {-2, -1, 0, 2}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1748_174858


namespace NUMINAMATH_CALUDE_function_inequality_l1748_174850

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)
variable (hf' : ∀ x, deriv f x < f x)

-- Define the theorem
theorem function_inequality (a : ℝ) (ha : a > 0) :
  f a < Real.exp a * f 0 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l1748_174850


namespace NUMINAMATH_CALUDE_reinforcement_calculation_l1748_174893

/-- Calculates the size of reinforcement given initial garrison size, provision days, and remaining days after reinforcement --/
def calculate_reinforcement (initial_garrison : ℕ) (initial_provision_days : ℕ) (days_before_reinforcement : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_provision_days
  let provisions_left := initial_garrison * (initial_provision_days - days_before_reinforcement)
  (provisions_left / remaining_days) - initial_garrison

theorem reinforcement_calculation (initial_garrison : ℕ) (initial_provision_days : ℕ) (days_before_reinforcement : ℕ) (remaining_days : ℕ) :
  initial_garrison = 1850 →
  initial_provision_days = 28 →
  days_before_reinforcement = 12 →
  remaining_days = 10 →
  calculate_reinforcement initial_garrison initial_provision_days days_before_reinforcement remaining_days = 1110 :=
by sorry

end NUMINAMATH_CALUDE_reinforcement_calculation_l1748_174893


namespace NUMINAMATH_CALUDE_intersection_P_Q_l1748_174863

-- Define set P
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- Define set Q
def Q : Set ℝ := {y | ∃ x : ℝ, y = x}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = {y : ℝ | y ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l1748_174863


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1748_174801

/-- An arithmetic sequence with first term 7, second term 11, and last term 95 has 23 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ → ℕ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
    a 0 = 7 →                                -- first term is 7
    a 1 = 11 →                               -- second term is 11
    (∃ m : ℕ, a m = 95 ∧ ∀ k > m, a k > 95) →  -- last term is 95
    ∃ n : ℕ, n = 23 ∧ a (n - 1) = 95 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1748_174801


namespace NUMINAMATH_CALUDE_sandwich_cost_l1748_174882

/-- The cost of Anna's sandwich given her breakfast and lunch expenses -/
theorem sandwich_cost (bagel_cost orange_juice_cost milk_cost lunch_difference : ℝ) : 
  bagel_cost = 0.95 →
  orange_juice_cost = 0.85 →
  milk_cost = 1.15 →
  lunch_difference = 4 →
  ∃ sandwich_cost : ℝ, 
    sandwich_cost + milk_cost = (bagel_cost + orange_juice_cost) + lunch_difference ∧
    sandwich_cost = 4.65 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_l1748_174882


namespace NUMINAMATH_CALUDE_chord_existence_l1748_174871

/-- A continuous curve in a 2D plane -/
def ContinuousCurve := Set (ℝ × ℝ)

/-- Defines if a curve connects two points -/
def connects (curve : ContinuousCurve) (A B : ℝ × ℝ) : Prop := sorry

/-- Defines if a curve has a chord of a given length parallel to a line segment -/
def has_parallel_chord (curve : ContinuousCurve) (A B : ℝ × ℝ) (length : ℝ) : Prop := sorry

/-- The distance between two points in 2D space -/
def distance (A B : ℝ × ℝ) : ℝ := sorry

theorem chord_existence (n : ℕ) (hn : n > 0) (A B : ℝ × ℝ) (curve : ContinuousCurve) :
  distance A B = 1 →
  connects curve A B →
  has_parallel_chord curve A B (1 / n) := by sorry

end NUMINAMATH_CALUDE_chord_existence_l1748_174871


namespace NUMINAMATH_CALUDE_mower_next_tangent_east_l1748_174859

/-- Represents the cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a circular garden with a mower --/
structure CircularGarden where
  garden_radius : ℝ
  mower_radius : ℝ
  initial_direction : Direction
  roll_direction : Bool  -- true for counterclockwise, false for clockwise

/-- 
  Determines the next tangent point where the mower's marker aims north again
  given a circular garden configuration
--/
def next_north_tangent (garden : CircularGarden) : Direction :=
  sorry

/-- The main theorem to be proved --/
theorem mower_next_tangent_east :
  let garden := CircularGarden.mk 15 5 Direction.North true
  next_north_tangent garden = Direction.East :=
sorry

end NUMINAMATH_CALUDE_mower_next_tangent_east_l1748_174859


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1748_174808

theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 7)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := distance^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1748_174808


namespace NUMINAMATH_CALUDE_quadratic_floor_existence_l1748_174865

theorem quadratic_floor_existence (x : ℝ) : 
  (∃ a b : ℤ, ∀ x : ℝ, x^2 + a*x + b ≠ 0 ∧ ∃ y : ℝ, ⌊y^2⌋ + a*y + b = 0) ∧
  (¬∃ a b : ℤ, ∀ x : ℝ, x^2 + 2*a*x + b ≠ 0 ∧ ∃ y : ℝ, ⌊y^2⌋ + 2*a*y + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_floor_existence_l1748_174865


namespace NUMINAMATH_CALUDE_fraction_decomposition_l1748_174831

theorem fraction_decomposition (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 / (x^2 - 1) = 1 / (x - 1) - 1 / (x + 1)) ∧
  (2 * x / (x^2 - 1) = 1 / (x - 1) + 1 / (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l1748_174831


namespace NUMINAMATH_CALUDE_modulus_of_complex_l1748_174873

theorem modulus_of_complex (z : ℂ) (h : z = 4 + 3*I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l1748_174873


namespace NUMINAMATH_CALUDE_vector_collinearity_l1748_174886

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 1]
def c : Fin 2 → ℝ := ![2, 1]

def collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, v = fun i => t * u i

theorem vector_collinearity (k : ℝ) :
  collinear (fun i => k * a i + b i) c → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1748_174886


namespace NUMINAMATH_CALUDE_biology_score_calculation_l1748_174891

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 62
def average_score : ℕ := 74
def total_subjects : ℕ := 5

theorem biology_score_calculation :
  let known_subjects_total := math_score + science_score + social_studies_score + english_score
  let all_subjects_total := average_score * total_subjects
  all_subjects_total - known_subjects_total = 85 := by
sorry

end NUMINAMATH_CALUDE_biology_score_calculation_l1748_174891
