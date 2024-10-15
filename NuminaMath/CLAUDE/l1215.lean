import Mathlib

namespace NUMINAMATH_CALUDE_one_fourths_in_five_thirds_l1215_121540

theorem one_fourths_in_five_thirds : (5 : ℚ) / 3 / (1 / 4) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_one_fourths_in_five_thirds_l1215_121540


namespace NUMINAMATH_CALUDE_negation_of_universal_inequality_l1215_121515

theorem negation_of_universal_inequality :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_inequality_l1215_121515


namespace NUMINAMATH_CALUDE_salesman_pear_sales_l1215_121554

/-- Represents the amount of pears sold by a salesman -/
structure PearSales where
  morning : ℕ
  afternoon : ℕ

/-- The total amount of pears sold in a day -/
def total_sales (sales : PearSales) : ℕ :=
  sales.morning + sales.afternoon

/-- Theorem stating the total sales of pears given the conditions -/
theorem salesman_pear_sales :
  ∃ (sales : PearSales),
    sales.afternoon = 340 ∧
    sales.afternoon = 2 * sales.morning ∧
    total_sales sales = 510 :=
by
  sorry

end NUMINAMATH_CALUDE_salesman_pear_sales_l1215_121554


namespace NUMINAMATH_CALUDE_polygon_with_1080_degrees_is_octagon_l1215_121557

/-- A polygon with interior angles summing to 1080° has 8 sides. -/
theorem polygon_with_1080_degrees_is_octagon :
  ∀ n : ℕ, (n - 2) * 180 = 1080 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_1080_degrees_is_octagon_l1215_121557


namespace NUMINAMATH_CALUDE_defective_draws_count_l1215_121568

/-- The number of ways to draw at least 2 defective products from a batch of 100 products
    containing 3 defective ones, when drawing 5 products. -/
def defectiveDraws : ℕ := sorry

/-- The total number of products -/
def totalProducts : ℕ := 100

/-- The number of defective products -/
def defectiveProducts : ℕ := 3

/-- The number of products drawn -/
def drawnProducts : ℕ := 5

theorem defective_draws_count :
  defectiveDraws = Nat.choose 3 2 * Nat.choose 97 3 + Nat.choose 3 3 * Nat.choose 97 2 := by
  sorry

end NUMINAMATH_CALUDE_defective_draws_count_l1215_121568


namespace NUMINAMATH_CALUDE_garden_ratio_l1215_121534

theorem garden_ratio (area width length : ℝ) : 
  area = 768 →
  width = 16 →
  area = length * width →
  length / width = 3 := by
sorry

end NUMINAMATH_CALUDE_garden_ratio_l1215_121534


namespace NUMINAMATH_CALUDE_students_per_van_l1215_121504

theorem students_per_van (num_vans : ℕ) (num_minibusses : ℕ) (students_per_minibus : ℕ) (total_students : ℕ) :
  num_vans = 6 →
  num_minibusses = 4 →
  students_per_minibus = 24 →
  total_students = 156 →
  (total_students - num_minibusses * students_per_minibus) / num_vans = 10 :=
by sorry

end NUMINAMATH_CALUDE_students_per_van_l1215_121504


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1215_121526

theorem absolute_value_simplification : |-4^2 + 6| = 10 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1215_121526


namespace NUMINAMATH_CALUDE_cubic_polynomial_condition_l1215_121509

/-- A polynomial is cubic if its highest degree term is of degree 3 -/
def IsCubicPolynomial (p : Polynomial ℝ) : Prop :=
  p.degree = 3

theorem cubic_polynomial_condition (m n : ℕ) :
  IsCubicPolynomial (X * Y^(m-n) + (n-2) * X^2 * Y^2 + 1) →
  m + 2*n = 8 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_condition_l1215_121509


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l1215_121521

theorem mary_baseball_cards 
  (initial_cards : ℕ) 
  (torn_cards : ℕ) 
  (cards_from_fred : ℕ) 
  (total_cards : ℕ) 
  (h1 : initial_cards = 18) 
  (h2 : torn_cards = 8) 
  (h3 : cards_from_fred = 26) 
  (h4 : total_cards = 84) : 
  total_cards - (initial_cards - torn_cards + cards_from_fred) = 48 := by
  sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l1215_121521


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_r_necessary_not_sufficient_for_p_l1215_121539

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | |3*x - 4| > 2}
def B : Set ℝ := {x : ℝ | 1 / (x^2 - x - 2) > 0}
def C (a : ℝ) : Set ℝ := {x : ℝ | (x - a) * (x - a - 1) ≥ 0}

-- Define the propositions p, q, and r
def p (x : ℝ) : Prop := x ∉ A
def q (x : ℝ) : Prop := x ∉ B
def r (a x : ℝ) : Prop := x ∈ C a

-- Theorem 1: p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (2/3 ≤ x ∧ x ≤ 2) → (-1 ≤ x ∧ x ≤ 2)) ∧
  (∃ x : ℝ, (-1 ≤ x ∧ x ≤ 2) ∧ ¬(2/3 ≤ x ∧ x ≤ 2)) :=
sorry

-- Theorem 2: r is a necessary but not sufficient condition for p
--            if and only if a ≥ 2 or a ≤ -1/3
theorem r_necessary_not_sufficient_for_p (a : ℝ) :
  ((∀ x : ℝ, p x → r a x) ∧ (∃ x : ℝ, r a x ∧ ¬(p x))) ↔ (a ≥ 2 ∨ a ≤ -1/3) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_r_necessary_not_sufficient_for_p_l1215_121539


namespace NUMINAMATH_CALUDE_parabolas_intersection_l1215_121562

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x_coords : Set ℝ :=
  {x | 2 * x^2 - 7 * x + 1 = 5 * x^2 - 2 * x - 2}

/-- The intersection points of two parabolas -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | p.1 ∈ intersection_x_coords ∧ p.2 = 2 * p.1^2 - 7 * p.1 + 1}

theorem parabolas_intersection :
  intersection_points = {((5 - Real.sqrt 61) / 6, 2 * ((5 - Real.sqrt 61) / 6)^2 - 7 * ((5 - Real.sqrt 61) / 6) + 1),
                         ((5 + Real.sqrt 61) / 6, 2 * ((5 + Real.sqrt 61) / 6)^2 - 7 * ((5 + Real.sqrt 61) / 6) + 1)} :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l1215_121562


namespace NUMINAMATH_CALUDE_food_supply_problem_l1215_121580

/-- Represents the food supply problem -/
theorem food_supply_problem (initial_men : ℕ) (additional_men : ℕ) (days_after_joining : ℕ) :
  initial_men = 760 →
  additional_men = 3040 →
  days_after_joining = 4 →
  ∃ (initial_days : ℕ),
    initial_days * initial_men = 
      (initial_days - 2) * initial_men + 
      days_after_joining * (initial_men + additional_men) ∧
    initial_days = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_food_supply_problem_l1215_121580


namespace NUMINAMATH_CALUDE_knight_reachability_l1215_121581

/-- Represents a position on an infinite chessboard -/
structure Position :=
  (x : ℤ)
  (y : ℤ)

/-- Represents a knight's move -/
def knight_move (p : Position) : Set Position :=
  {q : Position | (abs (q.x - p.x) = 2 ∧ abs (q.y - p.y) = 1) ∨
                  (abs (q.x - p.x) = 1 ∧ abs (q.y - p.y) = 2)}

/-- Represents the set of positions a knight can reach in exactly n moves -/
def reachable_in (start : Position) (n : ℕ) : Set Position :=
  match n with
  | 0 => {start}
  | n + 1 => ⋃ p ∈ reachable_in start n, knight_move p

/-- Represents whether a position is a black square -/
def is_black (p : Position) : Prop :=
  (p.x + p.y) % 2 = 0

/-- Represents the region described in the problem -/
def target_region (n : ℕ) (start : Position) : Set Position :=
  {p : Position | max (abs (p.x - start.x)) (abs (p.y - start.y)) ≤ 4*n + 1 ∧
                  abs (p.x - start.x) + abs (p.y - start.y) > 2*n}

/-- The main theorem to prove -/
theorem knight_reachability (n : ℕ) (start : Position) :
  ∀ p ∈ target_region n start, is_black p → p ∈ reachable_in start (2*n) :=
sorry

end NUMINAMATH_CALUDE_knight_reachability_l1215_121581


namespace NUMINAMATH_CALUDE_base_ten_and_twelve_satisfy_conditions_l1215_121551

/-- Represents a number in a given base -/
def NumberInBase (n : ℕ) (base : ℕ) : ℕ := n

/-- Checks if a number is even in a given base -/
def IsEvenInBase (n : ℕ) (base : ℕ) : Prop :=
  ∃ k : ℕ, NumberInBase n base = 2 * k

/-- Checks if three numbers are consecutive in a given base -/
def AreConsecutiveInBase (a b c : ℕ) (base : ℕ) : Prop :=
  NumberInBase b base = NumberInBase a base + 1 ∧
  NumberInBase c base = NumberInBase b base + 1

/-- The main theorem to prove -/
theorem base_ten_and_twelve_satisfy_conditions :
  (NumberInBase 24 10 = NumberInBase 4 10 * NumberInBase 6 10 ∧
   ∃ a b c : ℕ, AreConsecutiveInBase a b c 10 ∧
   (IsEvenInBase a 10 ∧ IsEvenInBase b 10 ∧ IsEvenInBase c 10 ∨
    ¬IsEvenInBase a 10 ∧ ¬IsEvenInBase b 10 ∧ ¬IsEvenInBase c 10)) ∧
  (NumberInBase 24 12 = NumberInBase 4 12 * NumberInBase 6 12 ∧
   ∃ a b c : ℕ, AreConsecutiveInBase a b c 12 ∧
   (IsEvenInBase a 12 ∧ IsEvenInBase b 12 ∧ IsEvenInBase c 12 ∨
    ¬IsEvenInBase a 12 ∧ ¬IsEvenInBase b 12 ∧ ¬IsEvenInBase c 12)) :=
by sorry


end NUMINAMATH_CALUDE_base_ten_and_twelve_satisfy_conditions_l1215_121551


namespace NUMINAMATH_CALUDE_base6_120_to_base2_l1215_121544

/-- Converts a number from base 6 to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 10 to base 2 --/
def base10ToBase2 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

theorem base6_120_to_base2 :
  base10ToBase2 (base6ToBase10 120) = [1, 1, 0, 0, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_base6_120_to_base2_l1215_121544


namespace NUMINAMATH_CALUDE_texas_california_plates_equal_l1215_121589

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_letters^3 * num_digits^4

/-- The number of possible California license plates -/
def california_plates : ℕ := num_digits * num_letters^3 * num_digits^3

/-- Theorem stating that Texas and California can issue the same number of license plates -/
theorem texas_california_plates_equal : texas_plates = california_plates := by
  sorry

end NUMINAMATH_CALUDE_texas_california_plates_equal_l1215_121589


namespace NUMINAMATH_CALUDE_sqrt_24_plus_3_bounds_l1215_121582

theorem sqrt_24_plus_3_bounds :
  (4 < Real.sqrt 24) ∧ (Real.sqrt 24 < 5) →
  (7 < Real.sqrt 24 + 3) ∧ (Real.sqrt 24 + 3 < 8) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_24_plus_3_bounds_l1215_121582


namespace NUMINAMATH_CALUDE_meeting_point_divides_segment_l1215_121523

/-- The meeting point of two people moving towards each other on a line --/
def meeting_point (x₁ y₁ x₂ y₂ : ℚ) (m n : ℕ) : ℚ × ℚ :=
  ((m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n))

/-- Theorem stating that the meeting point divides the line segment in the correct ratio --/
theorem meeting_point_divides_segment : 
  let mark_start : ℚ × ℚ := (2, 6)
  let sandy_start : ℚ × ℚ := (4, -2)
  let speed_ratio : ℕ × ℕ := (2, 1)
  let meet_point := meeting_point mark_start.1 mark_start.2 sandy_start.1 sandy_start.2 speed_ratio.1 speed_ratio.2
  meet_point = (8/3, 10/3) :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_divides_segment_l1215_121523


namespace NUMINAMATH_CALUDE_area_bisecting_line_property_l1215_121542

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(9, 0) -/
structure Triangle where
  P : ℝ × ℝ := (0, 10)
  Q : ℝ × ℝ := (3, 0)
  R : ℝ × ℝ := (9, 0)

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Function to check if a line bisects the area of the triangle -/
def bisects_area (t : Triangle) (l : Line) : Prop :=
  sorry -- Definition of area bisection

/-- Function to check if a line passes through a point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  sorry -- Definition of line passing through a point

/-- Theorem stating the property of the area-bisecting line -/
theorem area_bisecting_line_property (t : Triangle) :
  ∃ l : Line, bisects_area t l ∧ passes_through l t.Q ∧ l.slope + l.y_intercept = -20/3 :=
sorry

end NUMINAMATH_CALUDE_area_bisecting_line_property_l1215_121542


namespace NUMINAMATH_CALUDE_sufficient_unnecessary_condition_l1215_121512

theorem sufficient_unnecessary_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-4) 2 → (1/2) * x^2 - a ≥ 0) ↔ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_unnecessary_condition_l1215_121512


namespace NUMINAMATH_CALUDE_distance_to_reflection_l1215_121565

/-- The distance between a point (3, 1) and its reflection over the y-axis is 6 units. -/
theorem distance_to_reflection : ∃ (X X' : ℝ × ℝ),
  X = (3, 1) ∧
  X'.1 = -X.1 ∧
  X'.2 = X.2 ∧
  Real.sqrt ((X'.1 - X.1)^2 + (X'.2 - X.2)^2) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_to_reflection_l1215_121565


namespace NUMINAMATH_CALUDE_fraction_change_l1215_121547

theorem fraction_change (n d x : ℚ) : 
  d = 2 * n - 1 →
  n / d = 5 / 9 →
  (n + x) / (d + x) = 3 / 5 →
  x = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_change_l1215_121547


namespace NUMINAMATH_CALUDE_distance_not_equal_addition_l1215_121596

theorem distance_not_equal_addition : ∀ (a b : ℤ), 
  a = -3 → b = 10 → (abs (b - a) ≠ -3 + 10) :=
by
  sorry

end NUMINAMATH_CALUDE_distance_not_equal_addition_l1215_121596


namespace NUMINAMATH_CALUDE_hall_length_width_difference_l1215_121520

/-- Represents a rectangular hall with given properties -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  area : ℝ
  width_half_length : width = length / 2
  area_eq : area = length * width

/-- Theorem stating the difference between length and width of the hall -/
theorem hall_length_width_difference (hall : RectangularHall) 
  (h_area : hall.area = 578) : hall.length - hall.width = 17 := by
  sorry

end NUMINAMATH_CALUDE_hall_length_width_difference_l1215_121520


namespace NUMINAMATH_CALUDE_albums_needed_for_xiao_hong_l1215_121550

/-- Calculates the minimum number of complete photo albums needed to store a given number of photos. -/
def minimum_albums_needed (pages_per_album : ℕ) (photos_per_page : ℕ) (total_photos : ℕ) : ℕ :=
  (total_photos + pages_per_album * photos_per_page - 1) / (pages_per_album * photos_per_page)

/-- Proves that 6 albums are needed for the given conditions. -/
theorem albums_needed_for_xiao_hong : minimum_albums_needed 32 5 900 = 6 := by
  sorry

#eval minimum_albums_needed 32 5 900

end NUMINAMATH_CALUDE_albums_needed_for_xiao_hong_l1215_121550


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1215_121593

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the problem
theorem arithmetic_geometric_ratio
  (d : ℝ) (h_d : d ≠ 0)
  (h_geom : ∃ a₁ : ℝ, (arithmetic_sequence a₁ d 3)^2 = 
    (arithmetic_sequence a₁ d 2) * (arithmetic_sequence a₁ d 9)) :
  ∃ a₁ : ℝ, (arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 4) /
            (arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 5 + arithmetic_sequence a₁ d 6) = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1215_121593


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_not_exceeding_500_l1215_121553

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by
  sorry

theorem largest_n_not_exceeding_500 : 
  ∀ k : ℕ, (k * (k + 1) ≤ 1000 → k ≤ 31) ∧ 
           (31 * 32 ≤ 1000) ∧ 
           (32 * 33 > 1000) := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_not_exceeding_500_l1215_121553


namespace NUMINAMATH_CALUDE_ratio_problem_l1215_121586

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 3) :
  a / c = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1215_121586


namespace NUMINAMATH_CALUDE_fraction_equality_l1215_121594

theorem fraction_equality (a b : ℝ) (h : a ≠ 0) : b / a = (a * b) / (a * a) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1215_121594


namespace NUMINAMATH_CALUDE_max_product_of_ranged_functions_l1215_121597

/-- Given two functions f and g defined on ℝ with specific ranges, 
    prove that the maximum value of their product is -1 -/
theorem max_product_of_ranged_functions 
  (f g : ℝ → ℝ) 
  (hf : ∀ x, 1 ≤ f x ∧ f x ≤ 6) 
  (hg : ∀ x, -4 ≤ g x ∧ g x ≤ -1) : 
  (∀ x, f x * g x ≤ -1) ∧ (∃ x, f x * g x = -1) :=
sorry

end NUMINAMATH_CALUDE_max_product_of_ranged_functions_l1215_121597


namespace NUMINAMATH_CALUDE_inheritance_investment_percentage_l1215_121535

/-- Given an inheritance and investment scenario, prove the unknown investment percentage --/
theorem inheritance_investment_percentage 
  (total_inheritance : ℝ) 
  (known_investment : ℝ) 
  (known_rate : ℝ) 
  (total_interest : ℝ) 
  (h1 : total_inheritance = 4000)
  (h2 : known_investment = 1800)
  (h3 : known_rate = 0.065)
  (h4 : total_interest = 227)
  : ∃ (unknown_rate : ℝ), 
    known_investment * known_rate + (total_inheritance - known_investment) * unknown_rate = total_interest ∧ 
    unknown_rate = 0.05 := by
  sorry


end NUMINAMATH_CALUDE_inheritance_investment_percentage_l1215_121535


namespace NUMINAMATH_CALUDE_bo_learning_words_l1215_121552

/-- Calculates the number of words to learn per day given the total number of flashcards,
    the percentage of known words, and the number of days to learn. -/
def words_per_day (total_cards : ℕ) (known_percentage : ℚ) (days_to_learn : ℕ) : ℚ :=
  (total_cards - (known_percentage * total_cards)) / days_to_learn

/-- Proves that given 800 flashcards, 20% known words, and 40 days to learn,
    the number of words to learn per day is 16. -/
theorem bo_learning_words :
  words_per_day 800 (1/5) 40 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bo_learning_words_l1215_121552


namespace NUMINAMATH_CALUDE_max_subjects_per_teacher_l1215_121577

/-- Proves that the maximum number of subjects a teacher can teach is 4 -/
theorem max_subjects_per_teacher (total_subjects : ℕ) (min_teachers : ℕ) : 
  total_subjects = 16 → min_teachers = 4 → (total_subjects / min_teachers : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_subjects_per_teacher_l1215_121577


namespace NUMINAMATH_CALUDE_unique_operation_assignment_l1215_121518

-- Define the type for arithmetic operations
inductive ArithOp
| Add
| Sub
| Mul
| Div
| Eq

-- Define a function to apply an arithmetic operation
def apply_op (op : ArithOp) (x y : ℤ) : Prop :=
  match op with
  | ArithOp.Add => x + y = 0
  | ArithOp.Sub => x - y = 0
  | ArithOp.Mul => x * y = 0
  | ArithOp.Div => y ≠ 0 ∧ x / y = 0
  | ArithOp.Eq => x = y

-- Define the theorem
theorem unique_operation_assignment :
  ∃! (A B C D E : ArithOp),
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧
    (C ≠ D) ∧ (C ≠ E) ∧
    (D ≠ E) ∧
    apply_op A 4 2 ∧ apply_op B 2 2 ∧
    apply_op B 8 0 ∧ apply_op C 4 2 ∧
    apply_op D 2 3 ∧ apply_op B 5 5 ∧
    apply_op B 4 0 ∧ apply_op E 5 1 :=
sorry

end NUMINAMATH_CALUDE_unique_operation_assignment_l1215_121518


namespace NUMINAMATH_CALUDE_equation_solutions_l1215_121501

theorem equation_solutions :
  (∃ x : ℚ, 3 + 2 * x = 6 ∧ x = 3 / 2) ∧
  (∃ x : ℚ, 3 - 1 / 2 * x = 3 * x + 1 ∧ x = 4 / 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1215_121501


namespace NUMINAMATH_CALUDE_inequality_range_l1215_121590

theorem inequality_range (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) :
  (∀ k > 0, Real.log (Real.sin θ)^2 - Real.log (Real.cos θ)^2 ≤ k * Real.cos (2 * θ)) ↔
  θ ∈ Set.Ioo 0 (Real.pi / 4) ∪ Set.Icc (3 * Real.pi / 4) Real.pi ∪ 
      Set.Ioo Real.pi (5 * Real.pi / 4) ∪ Set.Icc (7 * Real.pi / 4) (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l1215_121590


namespace NUMINAMATH_CALUDE_multiply_times_theorem_l1215_121571

theorem multiply_times_theorem (n : ℝ) (x : ℝ) (h1 : n = 1) :
  x * n - 1 = 2 * n → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_times_theorem_l1215_121571


namespace NUMINAMATH_CALUDE_sin_theta_value_l1215_121549

theorem sin_theta_value (θ : Real) (h : Real.cos (π / 4 - θ / 2) = 2 / 3) : 
  Real.sin θ = -1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l1215_121549


namespace NUMINAMATH_CALUDE_impossible_to_flip_all_l1215_121527

/-- Represents the state of a coin: true if facing up, false if facing down -/
def Coin := Bool

/-- Represents the state of 5 coins -/
def CoinState := Fin 5 → Coin

/-- An operation that flips exactly 4 coins -/
def FlipFour (state : CoinState) : CoinState :=
  sorry

/-- The initial state where all coins are facing up -/
def initialState : CoinState := fun _ => true

/-- The target state where all coins are facing down -/
def targetState : CoinState := fun _ => false

/-- Predicate to check if a state can be reached from the initial state -/
def Reachable (state : CoinState) : Prop :=
  sorry

theorem impossible_to_flip_all :
  ¬ Reachable targetState :=
sorry

end NUMINAMATH_CALUDE_impossible_to_flip_all_l1215_121527


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_1001_l1215_121506

theorem modular_inverse_17_mod_1001 : ∃ x : ℕ, x ≤ 1000 ∧ (17 * x) % 1001 = 1 :=
by
  use 530
  sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_1001_l1215_121506


namespace NUMINAMATH_CALUDE_equation_has_seven_solutions_l1215_121583

/-- The function f(x) = |x² - 2x - 3| -/
def f (x : ℝ) : ℝ := |x^2 - 2*x - 3|

/-- The equation f³(x) - 4f²(x) - f(x) + 4 = 0 -/
def equation (x : ℝ) : Prop :=
  f x ^ 3 - 4 * (f x)^2 - f x + 4 = 0

/-- Theorem stating that the equation has exactly 7 solutions -/
theorem equation_has_seven_solutions :
  ∃! (s : Finset ℝ), s.card = 7 ∧ ∀ x, x ∈ s ↔ equation x :=
sorry

end NUMINAMATH_CALUDE_equation_has_seven_solutions_l1215_121583


namespace NUMINAMATH_CALUDE_consecutive_prime_even_triangular_product_l1215_121569

/-- A number is triangular if it can be represented as n * (n + 1) / 2 for some natural number n. -/
def IsTriangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

theorem consecutive_prime_even_triangular_product : ∃ a b c : ℕ,
  (a < 20 ∧ b < 20 ∧ c < 20) ∧
  (b = a + 1 ∧ c = b + 1) ∧
  Nat.Prime a ∧
  Even b ∧
  IsTriangular c ∧
  a * b * c = 2730 :=
sorry

end NUMINAMATH_CALUDE_consecutive_prime_even_triangular_product_l1215_121569


namespace NUMINAMATH_CALUDE_no_real_solutions_l1215_121546

theorem no_real_solutions :
  ¬ ∃ x : ℝ, x + Real.sqrt (x + 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1215_121546


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l1215_121555

theorem even_odd_sum_difference : 
  (Finset.sum (Finset.range 100) (fun i => 2 * (i + 1))) - 
  (Finset.sum (Finset.range 100) (fun i => 2 * i + 1)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l1215_121555


namespace NUMINAMATH_CALUDE_cost_for_23_days_l1215_121558

/-- Calculates the cost of staying in a student youth hostel for a given number of days. -/
def hostel_cost (days : ℕ) : ℚ :=
  let first_week_cost := 7 * 18
  let remaining_days := days - 7
  let additional_cost := remaining_days * 14
  first_week_cost + additional_cost

/-- Theorem stating that the cost for a 23-day stay is $350.00 -/
theorem cost_for_23_days : hostel_cost 23 = 350 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_23_days_l1215_121558


namespace NUMINAMATH_CALUDE_total_hired_is_31_l1215_121574

/-- Represents the daily pay for heavy operators -/
def heavy_operator_pay : ℕ := 129

/-- Represents the daily pay for general laborers -/
def general_laborer_pay : ℕ := 82

/-- Represents the total payroll -/
def total_payroll : ℕ := 3952

/-- Represents the number of general laborers hired -/
def laborers_hired : ℕ := 1

/-- Theorem stating that the total number of people hired is 31 -/
theorem total_hired_is_31 : 
  ∃ (heavy_operators : ℕ), 
    heavy_operator_pay * heavy_operators + general_laborer_pay * laborers_hired = total_payroll ∧
    heavy_operators + laborers_hired = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_hired_is_31_l1215_121574


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1215_121541

theorem trigonometric_identity (t : ℝ) (h : 3 * Real.cos (2 * t) - Real.sin (2 * t) ≠ 0) :
  (6 * Real.cos (2 * t)^3 + 2 * Real.sin (2 * t)^3) / (3 * Real.cos (2 * t) - Real.sin (2 * t))
  = Real.cos (4 * t) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1215_121541


namespace NUMINAMATH_CALUDE_rope_cost_minimum_l1215_121543

/-- The cost of one foot of rope in dollars -/
def cost_per_foot : ℚ := 5 / 4

/-- The length of rope needed in feet -/
def rope_length_needed : ℚ := 5

/-- The minimum cost to buy the required length of rope -/
def min_cost : ℚ := rope_length_needed * cost_per_foot

theorem rope_cost_minimum :
  min_cost = 25 / 4 := by sorry

end NUMINAMATH_CALUDE_rope_cost_minimum_l1215_121543


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1215_121566

/-- A line y = 2x + c is tangent to the parabola y^2 = 8x if and only if c = 1 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = 2 * p.1 + c ∧ p.2^2 = 8 * p.1) ↔ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1215_121566


namespace NUMINAMATH_CALUDE_megan_deleted_files_l1215_121599

/-- Calculates the number of deleted files given the initial number of files,
    the number of folders after organizing, and the number of files per folder. -/
def deleted_files (initial_files : ℕ) (num_folders : ℕ) (files_per_folder : ℕ) : ℕ :=
  initial_files - (num_folders * files_per_folder)

/-- Proves that Megan deleted 21 files given the problem conditions. -/
theorem megan_deleted_files :
  deleted_files 93 9 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_megan_deleted_files_l1215_121599


namespace NUMINAMATH_CALUDE_probability_different_digits_l1215_121576

def is_valid_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_different_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

def count_valid_numbers : ℕ := 999 - 100 + 1

def count_numbers_with_different_digits : ℕ := 9 * 9 * 8

theorem probability_different_digits :
  (count_numbers_with_different_digits : ℚ) / count_valid_numbers = 18 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_digits_l1215_121576


namespace NUMINAMATH_CALUDE_ball_max_height_l1215_121584

-- Define the height function
def h (t : ℝ) : ℝ := -4 * t^2 + 40 * t + 20

-- State the theorem
theorem ball_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 120 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l1215_121584


namespace NUMINAMATH_CALUDE_square_tile_side_length_l1215_121564

theorem square_tile_side_length (side : ℝ) (area : ℝ) : 
  area = 49 ∧ area = side * side → side = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_tile_side_length_l1215_121564


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_l1215_121592

theorem alcohol_water_ratio (alcohol_volume water_volume : ℚ) : 
  alcohol_volume = 2/7 → water_volume = 3/7 → 
  alcohol_volume / water_volume = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_l1215_121592


namespace NUMINAMATH_CALUDE_bicycle_business_loss_percentage_l1215_121503

/-- Calculates the overall loss percentage for a bicycle business -/
def overall_loss_percentage (cp1 sp1 cp2 sp2 cp3 sp3 : ℚ) : ℚ :=
  let tcp := cp1 + cp2 + cp3
  let tsp := sp1 + sp2 + sp3
  let loss := tcp - tsp
  (loss / tcp) * 100

/-- Theorem stating the overall loss percentage for the given bicycle business -/
theorem bicycle_business_loss_percentage :
  let cp1 := 1000
  let sp1 := 1080
  let cp2 := 1500
  let sp2 := 1100
  let cp3 := 2000
  let sp3 := 2200
  overall_loss_percentage cp1 sp1 cp2 sp2 cp3 sp3 = 2.67 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_business_loss_percentage_l1215_121503


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l1215_121514

/-- The height difference between Miki's sandcastle and her sister's sandcastle -/
theorem sandcastle_height_difference 
  (miki_height : ℝ) 
  (sister_height : ℝ) 
  (h1 : miki_height = 0.8333333333333334) 
  (h2 : sister_height = 0.5) : 
  miki_height - sister_height = 0.3333333333333334 := by
  sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l1215_121514


namespace NUMINAMATH_CALUDE_product_of_squares_l1215_121559

theorem product_of_squares : 
  (1 + 1 / 1^2) * (1 + 1 / 2^2) * (1 + 1 / 3^2) * (1 + 1 / 4^2) * (1 + 1 / 5^2) * (1 + 1 / 6^2) = 16661 / 3240 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_l1215_121559


namespace NUMINAMATH_CALUDE_frog_corner_prob_four_hops_l1215_121529

/-- Represents a position on the 4x4 grid -/
inductive Position
| Center
| Edge
| Corner

/-- Represents the state of the frog's movement -/
structure FrogState where
  position : Position
  hops : Nat

/-- The probability of moving to a corner in one hop from a given position -/
def cornerProbFromPosition (pos : Position) : ℚ :=
  match pos with
  | Position.Center => 0
  | Position.Edge => 1/8
  | Position.Corner => 1

/-- The probability of the frog being in a corner after n hops -/
def cornerProbAfterNHops (n : Nat) : ℚ :=
  sorry

theorem frog_corner_prob_four_hops :
  cornerProbAfterNHops 4 = 3/8 := by sorry

end NUMINAMATH_CALUDE_frog_corner_prob_four_hops_l1215_121529


namespace NUMINAMATH_CALUDE_points_from_lines_l1215_121500

/-- The number of lines formed by n points on a plane, where no three points are collinear -/
def num_lines (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that if 45 lines are formed by n points on a plane where no three are collinear, then n = 10 -/
theorem points_from_lines (n : ℕ) (h : num_lines n = 45) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_points_from_lines_l1215_121500


namespace NUMINAMATH_CALUDE_triangle_area_l1215_121588

-- Define the triangle ABC and point K
variable (A B C K : ℝ × ℝ)

-- Define the conditions
def is_on_line (P Q R : ℝ × ℝ) : Prop := sorry
def is_altitude (P Q R S : ℝ × ℝ) : Prop := sorry
def distance (P Q : ℝ × ℝ) : ℝ := sorry
def area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_area :
  is_on_line K B C →
  is_altitude A K B C →
  distance A C = 12 →
  distance B K = 9 →
  distance B C = 18 →
  area A B C = 27 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1215_121588


namespace NUMINAMATH_CALUDE_lucy_fish_purchase_l1215_121531

/-- The number of fish Lucy bought -/
def fish_bought (initial final : ℝ) : ℝ := final - initial

/-- Proof that Lucy bought 280 fish -/
theorem lucy_fish_purchase : fish_bought 212.0 492 = 280 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_purchase_l1215_121531


namespace NUMINAMATH_CALUDE_race_distance_l1215_121522

theorem race_distance (t1 t2 combined_time : ℝ) (h1 : t1 = 21) (h2 : t2 = 24) 
  (h3 : combined_time = 75) : 
  let d := (5 * t1 + 5 * t2) / combined_time
  d = 3 := by sorry

end NUMINAMATH_CALUDE_race_distance_l1215_121522


namespace NUMINAMATH_CALUDE_car_trip_distance_l1215_121538

theorem car_trip_distance (D : ℝ) 
  (h1 : D > 0)
  (h2 : (1/2) * D + (1/4) * ((1/2) * D) + (1/3) * ((1/2) * D - (1/4) * ((1/2) * D)) + 270 = D) :
  (1/4) * D = 270 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_distance_l1215_121538


namespace NUMINAMATH_CALUDE_cube_volume_l1215_121575

theorem cube_volume (s : ℝ) : 
  (s + 2) * (s - 2) * s = s^3 - 12 → s^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l1215_121575


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1215_121517

theorem complex_equation_solution (x y : ℝ) : 
  (x / (1 + Complex.I)) + (y / (1 + 2 * Complex.I)) = 5 / (1 + Complex.I) → y = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1215_121517


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l1215_121525

theorem number_of_divisors_36 : Finset.card (Nat.divisors 36) = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l1215_121525


namespace NUMINAMATH_CALUDE_proj_scale_proj_add_l1215_121513

-- Define the 2D vector type
def Vector2D := ℝ × ℝ

-- Define the projection operation on x-axis
def proj_x (v : Vector2D) : ℝ := v.1

-- Define the projection operation on y-axis
def proj_y (v : Vector2D) : ℝ := v.2

-- Define vector addition
def add (u v : Vector2D) : Vector2D := (u.1 + v.1, u.2 + v.2)

-- Define scalar multiplication
def scale (k : ℝ) (v : Vector2D) : Vector2D := (k * v.1, k * v.2)

-- Theorem for property 1 (scalar multiplication)
theorem proj_scale (k : ℝ) (v : Vector2D) :
  proj_x (scale k v) = k * proj_x v ∧ proj_y (scale k v) = k * proj_y v := by
  sorry

-- Theorem for property 2 (vector addition)
theorem proj_add (u v : Vector2D) :
  proj_x (add u v) = proj_x u + proj_x v ∧ proj_y (add u v) = proj_y u + proj_y v := by
  sorry

end NUMINAMATH_CALUDE_proj_scale_proj_add_l1215_121513


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1215_121561

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a > b → a > b - 1) ∧
  (∃ a b, a > b - 1 ∧ ¬(a > b)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1215_121561


namespace NUMINAMATH_CALUDE_equal_probability_sums_l1215_121591

/-- A standard die with faces labeled 1 to 6 -/
def StandardDie := Fin 6

/-- The number of dice being rolled -/
def numDice : ℕ := 9

/-- The sum we're comparing to -/
def compareSum : ℕ := 15

/-- The sum we're proving has the same probability -/
def targetSum : ℕ := 48

/-- A function to calculate the probability of a specific sum occurring when rolling n dice -/
noncomputable def probabilityOfSum (n : ℕ) (sum : ℕ) : ℝ := sorry

theorem equal_probability_sums :
  probabilityOfSum numDice compareSum = probabilityOfSum numDice targetSum :=
sorry

end NUMINAMATH_CALUDE_equal_probability_sums_l1215_121591


namespace NUMINAMATH_CALUDE_nina_savings_weeks_l1215_121570

/-- The number of weeks Nina needs to save to buy a video game -/
def weeks_to_save (game_cost : ℚ) (tax_rate : ℚ) (weekly_allowance : ℚ) (savings_rate : ℚ) : ℚ :=
  let total_cost := game_cost * (1 + tax_rate)
  let weekly_savings := weekly_allowance * savings_rate
  total_cost / weekly_savings

/-- Theorem: Nina needs 11 weeks to save for the video game -/
theorem nina_savings_weeks :
  weeks_to_save 50 0.1 10 0.5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_nina_savings_weeks_l1215_121570


namespace NUMINAMATH_CALUDE_min_cakes_to_recover_investment_l1215_121537

def investment : ℕ := 8000
def revenue_per_cake : ℕ := 15
def expense_per_cake : ℕ := 5

theorem min_cakes_to_recover_investment :
  ∀ n : ℕ, n * (revenue_per_cake - expense_per_cake) ≥ investment → n ≥ 800 :=
by sorry

end NUMINAMATH_CALUDE_min_cakes_to_recover_investment_l1215_121537


namespace NUMINAMATH_CALUDE_farm_spiders_l1215_121508

/-- Represents the number of animals on a farm --/
structure FarmAnimals where
  ducks : ℕ
  cows : ℕ
  spiders : ℕ

/-- Conditions of the farm problem --/
def farm_conditions (animals : FarmAnimals) : Prop :=
  2 * animals.ducks = 3 * animals.cows ∧
  2 * animals.ducks = 60 ∧
  2 * animals.ducks + 4 * animals.cows + 8 * animals.spiders = 270 ∧
  animals.ducks + animals.cows + animals.spiders = 70

/-- Theorem stating that under the given conditions, there are 20 spiders on the farm --/
theorem farm_spiders (animals : FarmAnimals) :
  farm_conditions animals → animals.spiders = 20 := by
  sorry

end NUMINAMATH_CALUDE_farm_spiders_l1215_121508


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1215_121595

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1215_121595


namespace NUMINAMATH_CALUDE_middle_elementary_students_l1215_121598

theorem middle_elementary_students (total : ℕ) 
  (h_total : total = 12000)
  (h_elementary : (15 : ℚ) / 16 * total = upper_elementary + middle_elementary)
  (h_not_upper : (1 : ℚ) / 2 * total = junior_high + middle_elementary)
  (h_groups : total = junior_high + upper_elementary + middle_elementary) :
  middle_elementary = 4875 := by
  sorry

end NUMINAMATH_CALUDE_middle_elementary_students_l1215_121598


namespace NUMINAMATH_CALUDE_jesse_pages_left_to_read_l1215_121573

/-- Represents a book with a given number of pages in the first 5 chapters -/
structure Book where
  chapter1 : ℕ
  chapter2 : ℕ
  chapter3 : ℕ
  chapter4 : ℕ
  chapter5 : ℕ

/-- The number of pages left to read in the book -/
def pagesLeftToRead (b : Book) : ℕ :=
  let pagesRead := b.chapter1 + b.chapter2 + b.chapter3 + b.chapter4 + b.chapter5
  let totalPages := pagesRead * 3
  totalPages - pagesRead

/-- Theorem stating the number of pages left to read in Jesse's book -/
theorem jesse_pages_left_to_read :
  let jessesBook : Book := {
    chapter1 := 10,
    chapter2 := 15,
    chapter3 := 27,
    chapter4 := 12,
    chapter5 := 19
  }
  pagesLeftToRead jessesBook = 166 := by
  sorry

end NUMINAMATH_CALUDE_jesse_pages_left_to_read_l1215_121573


namespace NUMINAMATH_CALUDE_roses_stolen_l1215_121572

/-- Given the initial number of roses, number of people, and roses per person,
    prove that the number of roses stolen is equal to the initial number of roses
    minus the product of the number of people and roses per person. -/
theorem roses_stolen (initial_roses : ℕ) (num_people : ℕ) (roses_per_person : ℕ) :
  initial_roses - (num_people * roses_per_person) =
  initial_roses - num_people * roses_per_person :=
by sorry

end NUMINAMATH_CALUDE_roses_stolen_l1215_121572


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l1215_121507

theorem largest_divisor_of_five_consecutive_integers (a : ℤ) :
  ∃ (k : ℤ), (a - 2) + (a - 1) + a + (a + 1) + (a + 2) = 5 * k :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l1215_121507


namespace NUMINAMATH_CALUDE_triangle_is_isosceles_l1215_121524

theorem triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a = 2 * b * Real.cos C →
  b = c :=
by sorry

end NUMINAMATH_CALUDE_triangle_is_isosceles_l1215_121524


namespace NUMINAMATH_CALUDE_nora_game_probability_l1215_121510

theorem nora_game_probability (p_lose : ℚ) (h1 : p_lose = 5/8) (h2 : ¬ ∃ p_tie : ℚ, p_tie > 0) :
  ∃ p_win : ℚ, p_win = 3/8 ∧ p_win + p_lose = 1 := by
  sorry

end NUMINAMATH_CALUDE_nora_game_probability_l1215_121510


namespace NUMINAMATH_CALUDE_pool_filling_time_l1215_121560

theorem pool_filling_time (pipe_a pipe_b pipe_c : ℝ) 
  (ha : pipe_a = 1 / 8)
  (hb : pipe_b = 1 / 12)
  (hc : pipe_c = 1 / 16) :
  1 / (pipe_a + pipe_b + pipe_c) = 48 / 13 :=
by sorry

end NUMINAMATH_CALUDE_pool_filling_time_l1215_121560


namespace NUMINAMATH_CALUDE_solve_for_x_l1215_121556

def star_op (x y : ℤ) : ℤ := x * y - 2 * (x + y)

theorem solve_for_x : ∃ x : ℤ, (∀ y : ℤ, star_op x y = x * y - 2 * (x + y)) ∧ star_op x (-3) = 1 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1215_121556


namespace NUMINAMATH_CALUDE_unique_parallel_line_l1215_121579

-- Define a type for points in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a type for lines in a plane
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define what it means for a point to be on a line
def PointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define what it means for two lines to be parallel
def ParallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

-- State the theorem
theorem unique_parallel_line (L : Line) (P : Point) 
  (h : ¬ PointOnLine P L) : 
  ∃! M : Line, ParallelLines M L ∧ PointOnLine P M :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_l1215_121579


namespace NUMINAMATH_CALUDE_max_different_sums_l1215_121519

def penny : ℚ := 1 / 100
def nickel : ℚ := 5 / 100
def dime : ℚ := 10 / 100
def half_dollar : ℚ := 50 / 100

def coin_set : Finset ℚ := {penny, nickel, nickel, dime, dime, half_dollar}

def sum_pairs (s : Finset ℚ) : Finset ℚ :=
  (s.product s).image (λ (x, y) => x + y)

theorem max_different_sums :
  (sum_pairs coin_set).card = 8 := by sorry

end NUMINAMATH_CALUDE_max_different_sums_l1215_121519


namespace NUMINAMATH_CALUDE_power_multiplication_l1215_121533

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1215_121533


namespace NUMINAMATH_CALUDE_exercise_gender_relation_l1215_121502

/-- Represents the contingency table data -/
structure ContingencyTable where
  male_regular : ℕ
  female_regular : ℕ
  male_not_regular : ℕ
  female_not_regular : ℕ

/-- Calculates the chi-square value -/
def chi_square (table : ContingencyTable) : ℚ :=
  let n := table.male_regular + table.female_regular + table.male_not_regular + table.female_not_regular
  let a := table.male_regular
  let b := table.female_regular
  let c := table.male_not_regular
  let d := table.female_not_regular
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The theorem to be proved -/
theorem exercise_gender_relation (total_students : ℕ) (prob_regular : ℚ) (male_regular : ℕ) (female_not_regular : ℕ) 
    (h_total : total_students = 100)
    (h_prob : prob_regular = 1/2)
    (h_male_regular : male_regular = 35)
    (h_female_not_regular : female_not_regular = 25)
    (h_critical_value : (2706 : ℚ)/1000 < (3841 : ℚ)/1000) :
  let table := ContingencyTable.mk 
    male_regular
    (total_students / 2 - male_regular)
    (total_students / 2 - female_not_regular)
    female_not_regular
  chi_square table > (2706 : ℚ)/1000 := by
  sorry

end NUMINAMATH_CALUDE_exercise_gender_relation_l1215_121502


namespace NUMINAMATH_CALUDE_slope_range_from_angle_of_inclination_l1215_121516

theorem slope_range_from_angle_of_inclination :
  ∀ a k : ℝ, 
    (π / 4 ≤ a ∧ a ≤ π / 2) →
    k = Real.tan a →
    (1 ≤ k ∧ ∀ y : ℝ, y ≥ 1 → ∃ x : ℝ, π / 4 ≤ x ∧ x ≤ π / 2 ∧ y = Real.tan x) :=
by sorry

end NUMINAMATH_CALUDE_slope_range_from_angle_of_inclination_l1215_121516


namespace NUMINAMATH_CALUDE_bryden_receives_45_dollars_l1215_121536

/-- The face value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The number of quarters Bryden has -/
def bryden_quarters : ℕ := 6

/-- The collector's offer as a percentage of face value -/
def collector_offer_percent : ℕ := 3000

/-- Calculate the amount Bryden will receive for his quarters -/
def bryden_received : ℚ :=
  (quarter_value * bryden_quarters) * (collector_offer_percent / 100)

theorem bryden_receives_45_dollars :
  bryden_received = 45 := by sorry

end NUMINAMATH_CALUDE_bryden_receives_45_dollars_l1215_121536


namespace NUMINAMATH_CALUDE_women_per_table_l1215_121505

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 7 →
  men_per_table = 2 →
  total_customers = 63 →
  ∃ women_per_table : ℕ, women_per_table * num_tables + men_per_table * num_tables = total_customers ∧ women_per_table = 7 :=
by sorry

end NUMINAMATH_CALUDE_women_per_table_l1215_121505


namespace NUMINAMATH_CALUDE_consecutive_points_theorem_l1215_121511

/-- Represents a point on a straight line -/
structure Point where
  x : ℝ

/-- Represents the distance between two points -/
def distance (p q : Point) : ℝ := q.x - p.x

theorem consecutive_points_theorem (a b c d e : Point)
  (consecutive : a.x < b.x ∧ b.x < c.x ∧ c.x < d.x ∧ d.x < e.x)
  (bc_eq_2cd : distance b c = 2 * distance c d)
  (de_eq_8 : distance d e = 8)
  (ac_eq_11 : distance a c = 11)
  (ae_eq_22 : distance a e = 22) :
  distance a b = 5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_points_theorem_l1215_121511


namespace NUMINAMATH_CALUDE_expansion_distinct_terms_l1215_121578

/-- The number of distinct terms in the expansion of (x+y)(a+b+c)(d+e+f) -/
def num_distinct_terms : ℕ := 18

/-- The first factor has 2 terms -/
def num_terms_factor1 : ℕ := 2

/-- The second factor has 3 terms -/
def num_terms_factor2 : ℕ := 3

/-- The third factor has 3 terms -/
def num_terms_factor3 : ℕ := 3

theorem expansion_distinct_terms :
  num_distinct_terms = num_terms_factor1 * num_terms_factor2 * num_terms_factor3 := by
  sorry

end NUMINAMATH_CALUDE_expansion_distinct_terms_l1215_121578


namespace NUMINAMATH_CALUDE_paint_cost_per_square_meter_l1215_121567

/-- Calculates the paint cost per square meter for a mural --/
theorem paint_cost_per_square_meter
  (mural_length : ℝ)
  (mural_width : ℝ)
  (painting_rate : ℝ)
  (labor_charge : ℝ)
  (total_cost : ℝ)
  (h1 : mural_length = 6)
  (h2 : mural_width = 3)
  (h3 : painting_rate = 1.5)
  (h4 : labor_charge = 10)
  (h5 : total_cost = 192) :
  (total_cost - (mural_length * mural_width / painting_rate) * labor_charge) / (mural_length * mural_width) = 4 :=
by sorry

end NUMINAMATH_CALUDE_paint_cost_per_square_meter_l1215_121567


namespace NUMINAMATH_CALUDE_adams_students_l1215_121532

/-- The number of students Adam teaches in 10 years -/
def students_in_ten_years (normal_students_per_year : ℕ) (first_year_students : ℕ) (total_years : ℕ) : ℕ :=
  first_year_students + (total_years - 1) * normal_students_per_year

/-- Theorem stating the total number of students Adam will teach in 10 years -/
theorem adams_students : 
  students_in_ten_years 50 40 10 = 490 := by
  sorry

end NUMINAMATH_CALUDE_adams_students_l1215_121532


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l1215_121585

theorem perpendicular_vectors_magnitude (a b : ℝ × ℝ) : 
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⟂ b
  (a.1^2 + a.2^2 = 4) →          -- |a| = 2
  (b.1^2 + b.2^2 = 4) →          -- |b| = 2
  ((2*a.1 - b.1)^2 + (2*a.2 - b.2)^2 = 20) := by  -- |2a - b| = 2√5
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l1215_121585


namespace NUMINAMATH_CALUDE_cos_420_plus_sin_330_eq_zero_l1215_121563

theorem cos_420_plus_sin_330_eq_zero :
  Real.cos (420 * π / 180) + Real.sin (330 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_420_plus_sin_330_eq_zero_l1215_121563


namespace NUMINAMATH_CALUDE_consecutive_sum_100_l1215_121530

theorem consecutive_sum_100 (n : ℕ) :
  (∃ (m : ℕ), m = n ∧ 
    n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_consecutive_sum_100_l1215_121530


namespace NUMINAMATH_CALUDE_actual_speed_is_30_l1215_121528

/-- Given that increasing the speed by 10 miles per hour reduces travel time by 1/4,
    prove that the actual average speed is 30 miles per hour. -/
theorem actual_speed_is_30 (v : ℝ) (h : v / (v + 10) = 3 / 4) : v = 30 := by
  sorry

end NUMINAMATH_CALUDE_actual_speed_is_30_l1215_121528


namespace NUMINAMATH_CALUDE_runner_parade_time_l1215_121545

/-- Calculates the time taken for a runner to travel from the front to the end of a moving parade. -/
theorem runner_parade_time (parade_length : ℝ) (parade_speed : ℝ) (runner_speed : ℝ) 
  (h1 : parade_length = 2)
  (h2 : parade_speed = 3)
  (h3 : runner_speed = 6) :
  (parade_length / (runner_speed - parade_speed)) * 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_runner_parade_time_l1215_121545


namespace NUMINAMATH_CALUDE_lorelai_jellybeans_count_l1215_121587

/-- The number of jellybeans Gigi has -/
def gigi_jellybeans : ℕ := 15

/-- The number of extra jellybeans Rory has compared to Gigi -/
def rory_extra_jellybeans : ℕ := 30

/-- The number of jellybeans Rory has -/
def rory_jellybeans : ℕ := gigi_jellybeans + rory_extra_jellybeans

/-- The total number of jellybeans both girls have -/
def total_girls_jellybeans : ℕ := gigi_jellybeans + rory_jellybeans

/-- The number of times Lorelai has eaten compared to both girls -/
def lorelai_multiplier : ℕ := 3

/-- The number of jellybeans Lorelai has eaten -/
def lorelai_jellybeans : ℕ := total_girls_jellybeans * lorelai_multiplier

theorem lorelai_jellybeans_count : lorelai_jellybeans = 180 := by
  sorry

end NUMINAMATH_CALUDE_lorelai_jellybeans_count_l1215_121587


namespace NUMINAMATH_CALUDE_equation_B_not_symmetric_l1215_121548

-- Define the equations
def equation_A (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equation_B (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equation_C (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1
def equation_D (x y : ℝ) : Prop := x + y^2 = -1

-- Define symmetry about x-axis
def symmetric_about_x_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ f x (-y)

-- Theorem statement
theorem equation_B_not_symmetric :
  ¬(symmetric_about_x_axis equation_B) ∧
  (symmetric_about_x_axis equation_A) ∧
  (symmetric_about_x_axis equation_C) ∧
  (symmetric_about_x_axis equation_D) :=
sorry

end NUMINAMATH_CALUDE_equation_B_not_symmetric_l1215_121548
