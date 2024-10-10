import Mathlib

namespace bird_count_proof_l3044_304472

/-- The number of storks on the fence -/
def num_storks : ℕ := 6

/-- The number of additional birds that joined -/
def additional_birds : ℕ := 3

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := 2

theorem bird_count_proof :
  initial_birds = 2 ∧
  num_storks = (initial_birds + additional_birds) + 1 :=
by sorry

end bird_count_proof_l3044_304472


namespace intersection_A_B_l3044_304400

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x : ℝ | 2 - x > 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end intersection_A_B_l3044_304400


namespace unique_special_number_l3044_304430

/-- A four-digit number with specific properties -/
def special_number : ℕ → Prop := λ n =>
  -- The number is four-digit
  1000 ≤ n ∧ n < 10000 ∧
  -- The unit digit is 2
  n % 10 = 2 ∧
  -- Moving the last digit to the front results in a number 108 less than the original
  (2000 + n / 10) = n - 108

theorem unique_special_number :
  ∃! n : ℕ, special_number n ∧ n = 2342 :=
sorry

end unique_special_number_l3044_304430


namespace wednesday_sales_l3044_304468

def initial_stock : ℕ := 600
def monday_sales : ℕ := 25
def tuesday_sales : ℕ := 70
def thursday_sales : ℕ := 110
def friday_sales : ℕ := 145
def unsold_percentage : ℚ := 1/4

theorem wednesday_sales :
  ∃ (wednesday_sales : ℕ),
    wednesday_sales = initial_stock * (1 - unsold_percentage) -
      (monday_sales + tuesday_sales + thursday_sales + friday_sales) ∧
    wednesday_sales = 100 := by
  sorry

end wednesday_sales_l3044_304468


namespace quadratic_roots_problem_l3044_304498

theorem quadratic_roots_problem (x₁ x₂ m : ℝ) : 
  (x₁^2 + 3*x₁ + m = 0) →
  (x₂^2 + 3*x₂ + m = 0) →
  (1/x₁ + 1/x₂ = 1) →
  m = -3 := by
sorry

end quadratic_roots_problem_l3044_304498


namespace correct_definition_in_list_correct_definition_unique_l3044_304440

/-- Definition of Digital Earth -/
def DigitalEarth : Type := String

/-- The correct definition of Digital Earth -/
def correct_definition : DigitalEarth :=
  "a technical system that digitizes the entire Earth's information and manages it through computer networks"

/-- Possible definitions of Digital Earth -/
def possible_definitions : List DigitalEarth :=
  [ "representing the size of the Earth with numbers"
  , correct_definition
  , "using the data of the latitude and longitude grid to represent the location of geographical entities"
  , "using GPS data to represent the location of various geographical entities on Earth"
  ]

/-- Theorem stating that the correct definition is in the list of possible definitions -/
theorem correct_definition_in_list : correct_definition ∈ possible_definitions :=
  by sorry

/-- Theorem stating that the correct definition is unique in the list -/
theorem correct_definition_unique :
  ∀ d ∈ possible_definitions, d = correct_definition ↔ d = possible_definitions[1] :=
  by sorry

end correct_definition_in_list_correct_definition_unique_l3044_304440


namespace odd_function_iff_m_and_n_l3044_304406

def f (m n x : ℝ) : ℝ := (m^2 - 1) * x^2 + (m - 1) * x + n + 2

theorem odd_function_iff_m_and_n (m n : ℝ) :
  (∀ x, f m n (-x) = -f m n x) ↔ ((m = 1 ∨ m = -1) ∧ n = 2) :=
sorry

end odd_function_iff_m_and_n_l3044_304406


namespace larger_root_of_quadratic_l3044_304435

theorem larger_root_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 40 = 0 → x ≤ 8 :=
by
  sorry

end larger_root_of_quadratic_l3044_304435


namespace pi_is_infinite_decimal_l3044_304447

-- Define the property of being an infinite decimal
def IsInfiniteDecimal (x : ℝ) : Prop := sorry

-- Define the property of being an irrational number
def IsIrrational (x : ℝ) : Prop := sorry

-- State the theorem
theorem pi_is_infinite_decimal :
  (∀ x : ℝ, IsIrrational x → IsInfiniteDecimal x) →  -- Condition: Irrational numbers are infinite decimals
  IsIrrational Real.pi →                             -- Condition: π is an irrational number
  IsInfiniteDecimal Real.pi :=                       -- Conclusion: π is an infinite decimal
by sorry

end pi_is_infinite_decimal_l3044_304447


namespace triangle_area_l3044_304412

/-- Given a triangle ABC with sides a, b, and c satisfying certain conditions, 
    prove that its area is 6. -/
theorem triangle_area (a b c : ℝ) : 
  (a + 4) / 3 = (b + 3) / 2 ∧ 
  (b + 3) / 2 = (c + 8) / 4 ∧ 
  a + b + c = 12 → 
  (1 / 2 : ℝ) * b * c = 6 := by sorry

end triangle_area_l3044_304412


namespace hyperbola_eccentricity_l3044_304416

theorem hyperbola_eccentricity (a : ℝ) : 
  a > 0 →
  (∃ x y : ℝ, x^2/a^2 - y^2/3^2 = 1) →
  (∃ e : ℝ, e = 2 ∧ e^2 = (a^2 + 3^2)/a^2) →
  a = Real.sqrt 3 := by
sorry

end hyperbola_eccentricity_l3044_304416


namespace dime_probability_l3044_304403

def coin_jar (quarter_value dime_value penny_value : ℚ)
             (total_quarter_value total_dime_value total_penny_value : ℚ) : Prop :=
  let quarter_count := total_quarter_value / quarter_value
  let dime_count := total_dime_value / dime_value
  let penny_count := total_penny_value / penny_value
  let total_coins := quarter_count + dime_count + penny_count
  dime_count / total_coins = 1 / 7

theorem dime_probability :
  coin_jar (25/100) (10/100) (1/100) (1250/100) (500/100) (250/100) := by
  sorry

end dime_probability_l3044_304403


namespace factors_imply_absolute_value_l3044_304495

def polynomial (h k : ℝ) (x : ℝ) : ℝ := 3 * x^4 - h * x^2 + k

theorem factors_imply_absolute_value (h k : ℝ) :
  (∀ x : ℝ, (x + 1 = 0 ∨ x - 2 = 0 ∨ x + 3 = 0) → polynomial h k x = 0) →
  |3 * h - 4 * k| = 3 := by
  sorry

end factors_imply_absolute_value_l3044_304495


namespace min_value_expression_l3044_304490

theorem min_value_expression (n : ℕ+) : 
  (3 * n : ℝ) / 4 + 32 / (n^2 : ℝ) ≥ 5 ∧ 
  (∃ n : ℕ+, (3 * n : ℝ) / 4 + 32 / (n^2 : ℝ) = 5) := by
  sorry

#check min_value_expression

end min_value_expression_l3044_304490


namespace geometric_sequence_common_ratio_l3044_304429

/-- A geometric sequence with first term 1 and third term 4 has a common ratio of ±2 -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ), 
  (∀ n : ℕ, a (n + 1) = a n * a 1) → -- Geometric sequence condition
  a 1 = 1 →
  a 3 = 4 →
  ∃ q : ℝ, a 1 * q^2 = a 3 ∧ q = 2 ∨ q = -2 := by
  sorry


end geometric_sequence_common_ratio_l3044_304429


namespace max_value_of_a_is_zero_l3044_304411

theorem max_value_of_a_is_zero (a : ℝ) 
  (h : ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → x * Real.log x - (1 + a) * x + 1 ≥ 0) : 
  a ≤ 0 ∧ ∀ ε > 0, ∃ x ∈ Set.Icc (1/2) 2, x * Real.log x - (1 + (a + ε)) * x + 1 < 0 := by
sorry

end max_value_of_a_is_zero_l3044_304411


namespace min_value_quadratic_min_value_quadratic_achieved_l3044_304455

theorem min_value_quadratic (x y : ℝ) : 
  2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x + 2 * y - 5 ≥ -10 := by
  sorry

theorem min_value_quadratic_achieved : 
  ∃ x y : ℝ, 2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x + 2 * y - 5 = -10 := by
  sorry

end min_value_quadratic_min_value_quadratic_achieved_l3044_304455


namespace john_bus_meet_once_l3044_304427

/-- Represents the movement of John and the bus on a straight path --/
structure Movement where
  johnSpeed : ℝ
  busSpeed : ℝ
  benchDistance : ℝ
  busStopTime : ℝ

/-- Calculates the number of times John and the bus meet --/
def meetingCount (m : Movement) : ℕ :=
  sorry

/-- Theorem stating that John and the bus meet exactly once --/
theorem john_bus_meet_once (m : Movement) 
  (h1 : m.johnSpeed = 6)
  (h2 : m.busSpeed = 15)
  (h3 : m.benchDistance = 300)
  (h4 : m.busStopTime = 45) :
  meetingCount m = 1 := by
  sorry

end john_bus_meet_once_l3044_304427


namespace octal_calculation_l3044_304443

/-- Represents a number in base 8 --/
def OctalNumber := Nat

/-- Addition of two octal numbers --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtraction of two octal numbers --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : Nat) : OctalNumber :=
  sorry

/-- Theorem: 24₈ + 53₈ - 17₈ = 60₈ in base 8 --/
theorem octal_calculation : 
  octal_sub (octal_add (to_octal 24) (to_octal 53)) (to_octal 17) = to_octal 60 :=
by sorry

end octal_calculation_l3044_304443


namespace magic_square_solution_l3044_304402

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℤ)

/-- The sum of any row, column, or diagonal in a magic square is constant -/
def MagicSquare.isMagic (s : MagicSquare) : Prop :=
  let sum := s.a11 + s.a12 + s.a13
  sum = s.a21 + s.a22 + s.a23 ∧
  sum = s.a31 + s.a32 + s.a33 ∧
  sum = s.a11 + s.a21 + s.a31 ∧
  sum = s.a12 + s.a22 + s.a32 ∧
  sum = s.a13 + s.a23 + s.a33 ∧
  sum = s.a11 + s.a22 + s.a33 ∧
  sum = s.a13 + s.a22 + s.a31

theorem magic_square_solution :
  ∀ (s : MagicSquare),
    s.isMagic →
    s.a11 = s.a11 ∧
    s.a12 = 23 ∧
    s.a13 = 84 ∧
    s.a21 = 3 →
    s.a11 = 175 := by
  sorry

#check magic_square_solution

end magic_square_solution_l3044_304402


namespace eighth_finger_number_l3044_304471

-- Define the function f
def f (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 4
  | 1 => 3
  | 2 => 6
  | 3 => 5
  | _ => 0  -- This case should never occur

-- Define a function to apply f n times
def apply_f_n_times (n : ℕ) : ℕ :=
  match n with
  | 0 => 4  -- Start with 4
  | n + 1 => f (apply_f_n_times n)

-- Theorem statement
theorem eighth_finger_number : apply_f_n_times 7 = 4 := by
  sorry

end eighth_finger_number_l3044_304471


namespace product_of_fractions_equals_one_l3044_304425

theorem product_of_fractions_equals_one :
  (7 / 3) * (10 / 6) * (35 / 21) * (20 / 12) * (49 / 21) * (18 / 30) * (45 / 27) * (24 / 40) = 1 := by
  sorry

end product_of_fractions_equals_one_l3044_304425


namespace sports_club_overlap_l3044_304413

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 42)
  (h2 : badminton = 20)
  (h3 : tennis = 23)
  (h4 : neither = 6) :
  badminton + tennis - (total - neither) = 7 :=
by sorry

end sports_club_overlap_l3044_304413


namespace mathematics_teacher_is_C_l3044_304418

-- Define the types for teachers and subjects
inductive Teacher : Type
  | A | B | C | D

inductive Subject : Type
  | Mathematics | Physics | Chemistry | English

-- Define a function to represent the ability to teach a subject
def canTeach : Teacher → Subject → Prop
  | Teacher.A, Subject.Physics => True
  | Teacher.A, Subject.Chemistry => True
  | Teacher.B, Subject.Mathematics => True
  | Teacher.B, Subject.English => True
  | Teacher.C, Subject.Mathematics => True
  | Teacher.C, Subject.Physics => True
  | Teacher.C, Subject.Chemistry => True
  | Teacher.D, Subject.Chemistry => True
  | _, _ => False

-- Define the assignment of teachers to subjects
def assignment : Subject → Teacher
  | Subject.Mathematics => Teacher.C
  | Subject.Physics => Teacher.A
  | Subject.Chemistry => Teacher.D
  | Subject.English => Teacher.B

-- Theorem statement
theorem mathematics_teacher_is_C :
  (∀ s : Subject, canTeach (assignment s) s) ∧
  (∀ t : Teacher, ∃! s : Subject, assignment s = t) ∧
  (∀ s : Subject, ∃! t : Teacher, assignment s = t) →
  assignment Subject.Mathematics = Teacher.C :=
by sorry

end mathematics_teacher_is_C_l3044_304418


namespace sum_increase_by_three_percent_l3044_304450

theorem sum_increase_by_three_percent : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
  (1.01 * x + 1.04 * y) = 1.03 * (x + y) := by
sorry

end sum_increase_by_three_percent_l3044_304450


namespace oliver_candy_boxes_l3044_304459

theorem oliver_candy_boxes (initial_boxes : ℕ) (total_boxes : ℕ) (boxes_bought_later : ℕ) :
  initial_boxes = 8 →
  total_boxes = 14 →
  boxes_bought_later = total_boxes - initial_boxes →
  boxes_bought_later = 6 := by
  sorry

end oliver_candy_boxes_l3044_304459


namespace first_sat_score_l3044_304489

/-- 
Given a 10% improvement from the first score to the second score, 
and a second score of 1100, prove that the first score must be 1000.
-/
theorem first_sat_score (second_score : ℝ) (improvement : ℝ) 
  (h1 : second_score = 1100)
  (h2 : improvement = 0.1)
  (h3 : second_score = (1 + improvement) * first_score) :
  first_score = 1000 := by
  sorry

end first_sat_score_l3044_304489


namespace product_of_w_and_x_is_zero_l3044_304405

theorem product_of_w_and_x_is_zero 
  (w x y : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / y) 
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) : 
  w * x = 0 := by
sorry

end product_of_w_and_x_is_zero_l3044_304405


namespace pants_and_coat_cost_l3044_304407

theorem pants_and_coat_cost (p s c : ℝ) 
  (h1 : p + s = 100)
  (h2 : c = 5 * s)
  (h3 : c = 180) : 
  p + c = 244 := by
  sorry

end pants_and_coat_cost_l3044_304407


namespace mans_speed_with_stream_l3044_304477

/-- Given a man's rowing speed against the stream and his rate in still water,
    calculate his speed with the stream. -/
theorem mans_speed_with_stream
  (speed_against_stream : ℝ)
  (rate_still_water : ℝ)
  (h1 : speed_against_stream = 4)
  (h2 : rate_still_water = 8) :
  rate_still_water + (rate_still_water - speed_against_stream) = 12 := by
  sorry

end mans_speed_with_stream_l3044_304477


namespace nickel_chocolates_l3044_304431

theorem nickel_chocolates (robert : ℕ) (nickel : ℕ) 
  (h1 : robert = 7)
  (h2 : robert = nickel + 2) : 
  nickel = 5 := by
sorry

end nickel_chocolates_l3044_304431


namespace fractional_equation_solution_l3044_304499

theorem fractional_equation_solution :
  ∃! x : ℝ, (3 - x) / (x - 4) + 1 / (4 - x) = 1 :=
by
  use 3
  sorry

end fractional_equation_solution_l3044_304499


namespace line_passes_through_quadrants_l3044_304414

/-- A line ax + by = c passes through the first, third, and fourth quadrants
    given that ab < 0 and bc < 0 -/
theorem line_passes_through_quadrants
  (a b c : ℝ) 
  (hab : a * b < 0) 
  (hbc : b * c < 0) : 
  ∃ (x y : ℝ), 
    (a * x + b * y = c) ∧ 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) :=
sorry

end line_passes_through_quadrants_l3044_304414


namespace starting_number_property_l3044_304474

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def subtractSumOfDigits (n : ℕ) : ℕ :=
  n - sumOfDigits n

def iterateSubtraction (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterateSubtraction (subtractSumOfDigits n) k

theorem starting_number_property (n : ℕ) (h : 100 ≤ n ∧ n ≤ 109) :
  iterateSubtraction n 11 = 0 :=
sorry

end starting_number_property_l3044_304474


namespace subset_implies_m_geq_one_l3044_304463

theorem subset_implies_m_geq_one (m : ℝ) : 
  ({x : ℝ | 0 < x ∧ x < 1} ⊆ {x : ℝ | 0 < x ∧ x < m}) → m ≥ 1 := by
  sorry

end subset_implies_m_geq_one_l3044_304463


namespace area_outside_overlapping_squares_l3044_304483

/-- The area of the region outside two overlapping squares within a larger square -/
theorem area_outside_overlapping_squares (large_side : ℝ) (small_side : ℝ) 
  (h_large : large_side = 9) 
  (h_small : small_side = 4) : 
  large_side^2 - 2 * small_side^2 = 49 := by
  sorry

end area_outside_overlapping_squares_l3044_304483


namespace max_value_expression_l3044_304442

open Real

theorem max_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (⨆ x : ℝ, 2 * (a - x) * (x - Real.sqrt (x^2 + b^2))) = b^2 := by
  sorry

end max_value_expression_l3044_304442


namespace eugene_model_house_l3044_304466

/-- The number of toothpicks Eugene uses per card -/
def toothpicks_per_card : ℕ := 75

/-- The total number of cards in the deck -/
def total_cards : ℕ := 52

/-- The number of cards Eugene did not use -/
def unused_cards : ℕ := 16

/-- The number of toothpicks in one box -/
def toothpicks_per_box : ℕ := 450

/-- The number of boxes of toothpicks Eugene used -/
def boxes_used : ℕ := 6

theorem eugene_model_house :
  boxes_used = (total_cards - unused_cards) * toothpicks_per_card / toothpicks_per_box :=
by sorry

end eugene_model_house_l3044_304466


namespace no_primes_divisible_by_91_l3044_304478

theorem no_primes_divisible_by_91 :
  ¬∃ p : ℕ, Nat.Prime p ∧ 91 ∣ p := by
  sorry

end no_primes_divisible_by_91_l3044_304478


namespace volunteer_team_statistics_l3044_304419

def frequencies : List ℕ := [10, 10, 10, 8, 8, 8, 8, 7, 7, 4]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

theorem volunteer_team_statistics :
  mode frequencies = 8 ∧
  median frequencies = 8 ∧
  mean frequencies = 8 := by sorry

end volunteer_team_statistics_l3044_304419


namespace total_oil_leak_l3044_304454

def initial_leak_A : ℕ := 6522
def initial_leak_B : ℕ := 3894
def initial_leak_C : ℕ := 1421

def leak_rate_A : ℕ := 257
def leak_rate_B : ℕ := 182
def leak_rate_C : ℕ := 97

def repair_time_A : ℕ := 20
def repair_time_B : ℕ := 15
def repair_time_C : ℕ := 12

theorem total_oil_leak :
  initial_leak_A + initial_leak_B + initial_leak_C +
  leak_rate_A * repair_time_A + leak_rate_B * repair_time_B + leak_rate_C * repair_time_C = 20871 := by
  sorry

end total_oil_leak_l3044_304454


namespace regular_polygon_diagonals_l3044_304451

theorem regular_polygon_diagonals (n : ℕ) : n > 2 → (n * (n - 3)) / 2 = 90 → n = 15 := by
  sorry

end regular_polygon_diagonals_l3044_304451


namespace intersection_points_sum_l3044_304460

-- Define the functions
def f (x : ℝ) : ℝ := (x - 2) * (x - 4)
def g (x : ℝ) : ℝ := -f x
def h (x : ℝ) : ℝ := f (-x)

-- Define c as the number of intersection points between f and g
def c : ℕ := 2

-- Define d as the number of intersection points between f and h
def d : ℕ := 1

-- Theorem to prove
theorem intersection_points_sum : 10 * c + d = 21 := by
  sorry

end intersection_points_sum_l3044_304460


namespace ellipse_equation_l3044_304437

/-- An ellipse with major axis three times the minor axis and focal distance 8 -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Half focal distance
  h_major_minor : a = 3 * b
  h_focal : c = 4
  h_positive : a > 0 ∧ b > 0
  h_ellipse : a^2 = b^2 + c^2

/-- The standard equation of the ellipse is either x²/18 + y²/2 = 1 or y²/18 + x²/2 = 1 -/
theorem ellipse_equation (e : Ellipse) :
  (∀ x y : ℝ, x^2 / 18 + y^2 / 2 = 1) ∨ (∀ x y : ℝ, y^2 / 18 + x^2 / 2 = 1) :=
sorry

end ellipse_equation_l3044_304437


namespace exam_score_problem_l3044_304439

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 120 →
  ∃ (correct_answers : ℕ) (wrong_answers : ℕ),
    correct_answers + wrong_answers = total_questions ∧
    correct_score * correct_answers + wrong_score * wrong_answers = total_score ∧
    correct_answers = 36 :=
by sorry

end exam_score_problem_l3044_304439


namespace midpoint_octagon_area_ratio_l3044_304448

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : Sorry

/-- The octagon formed by joining the midpoints of the sides of a regular octagon -/
def midpoint_octagon (oct : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (oct : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of the octagon formed by joining the midpoints of the sides
    of a regular octagon is 1/2 of the area of the original octagon -/
theorem midpoint_octagon_area_ratio (oct : RegularOctagon) :
  area (midpoint_octagon oct) = (1/2 : ℝ) * area oct :=
sorry

end midpoint_octagon_area_ratio_l3044_304448


namespace spade_equation_solution_l3044_304480

-- Define the ♠ operation
def spade (A B : ℝ) : ℝ := 4 * A + 3 * B + 6

-- Theorem statement
theorem spade_equation_solution :
  ∃! A : ℝ, spade A 5 = 79 ∧ A = 14.5 := by sorry

end spade_equation_solution_l3044_304480


namespace sin_alpha_value_l3044_304462

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 6) = 1 / 5) : 
  Real.sin α = (6 * Real.sqrt 2 - 1) / 10 := by
sorry

end sin_alpha_value_l3044_304462


namespace crease_length_eq_sqrt_six_over_four_l3044_304479

/-- An isosceles right triangle with hypotenuse 1 -/
structure IsoscelesRightTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is 1 -/
  hypotenuse_eq_one : hypotenuse = 1

/-- The crease formed by folding one vertex to the other on the hypotenuse -/
def crease_length (t : IsoscelesRightTriangle) : ℝ :=
  sorry  -- Definition of crease length calculation

theorem crease_length_eq_sqrt_six_over_four (t : IsoscelesRightTriangle) :
  crease_length t = Real.sqrt 6 / 4 := by
  sorry

end crease_length_eq_sqrt_six_over_four_l3044_304479


namespace butter_left_is_two_l3044_304497

/-- Calculates the amount of butter left after making three types of cookies. -/
def butter_left (total : ℚ) (choc_chip_frac : ℚ) (peanut_butter_frac : ℚ) (sugar_frac : ℚ) : ℚ :=
  let remaining_after_two := total - (choc_chip_frac * total) - (peanut_butter_frac * total)
  remaining_after_two - (sugar_frac * remaining_after_two)

/-- Proves that given the specified conditions, the amount of butter left is 2 kilograms. -/
theorem butter_left_is_two :
  butter_left 10 (1/2) (1/5) (1/3) = 2 := by
  sorry

end butter_left_is_two_l3044_304497


namespace four_digit_sum_product_divisible_by_11_l3044_304457

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Converts four digits to a four-digit number -/
def toNumber (w x y z : Digit) : ℕ :=
  1000 * w.val + 100 * x.val + 10 * y.val + z.val

theorem four_digit_sum_product_divisible_by_11 
  (w x y z : Digit) 
  (hw : w ≠ x) (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ w) 
  (hwx : w ≠ y) (hwy : w ≠ z) (hxy : x ≠ z) : 
  11 ∣ (toNumber w x y z + toNumber z y x w + toNumber w x y z * toNumber z y x w) :=
sorry

end four_digit_sum_product_divisible_by_11_l3044_304457


namespace gcd_38_23_is_1_l3044_304401

/-- The method of continued subtraction for calculating GCD -/
def continuedSubtractionGCD (a b : ℕ) : ℕ :=
  if a = 0 then b
  else if b = 0 then a
  else if a ≥ b then continuedSubtractionGCD (a - b) b
  else continuedSubtractionGCD a (b - a)

/-- Theorem: The GCD of 38 and 23 is 1 using the method of continued subtraction -/
theorem gcd_38_23_is_1 : continuedSubtractionGCD 38 23 = 1 := by
  sorry

end gcd_38_23_is_1_l3044_304401


namespace sock_matching_probability_l3044_304481

def total_socks : ℕ := 8
def black_socks : ℕ := 6
def white_socks : ℕ := 2

def total_combinations : ℕ := total_socks.choose 2
def matching_combinations : ℕ := black_socks.choose 2 + 1

theorem sock_matching_probability :
  (matching_combinations : ℚ) / total_combinations = 2 / 7 :=
by sorry

end sock_matching_probability_l3044_304481


namespace geometric_progression_ratio_l3044_304444

theorem geometric_progression_ratio (x y z r : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h_geometric : ∃ (a : ℝ), a ≠ 0 ∧ 
    x * (y - z) = a ∧ 
    y * (z - x) = a * r ∧ 
    z * (x - y) = a * r^2) :
  r^2 + r + 1 = 0 := by
sorry

end geometric_progression_ratio_l3044_304444


namespace square_ratio_side_length_sum_l3044_304486

theorem square_ratio_side_length_sum (area_ratio : ℚ) :
  area_ratio = 27 / 50 →
  ∃ (a b c : ℕ), 
    (a * (b.sqrt : ℝ) / c : ℝ) = (area_ratio : ℝ).sqrt ∧
    a = 3 ∧ b = 6 ∧ c = 10 ∧
    a + b + c = 19 := by
  sorry

end square_ratio_side_length_sum_l3044_304486


namespace arithmetic_expression_evaluation_l3044_304494

theorem arithmetic_expression_evaluation : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_expression_evaluation_l3044_304494


namespace apple_preference_percentage_l3044_304428

-- Define the fruit categories
inductive Fruit
| Apple
| Banana
| Cherry
| Orange
| Pear

-- Define the function that gives the frequency for each fruit
def frequency (f : Fruit) : ℕ :=
  match f with
  | Fruit.Apple => 75
  | Fruit.Banana => 80
  | Fruit.Cherry => 45
  | Fruit.Orange => 100
  | Fruit.Pear => 50

-- Define the total number of responses
def total_responses : ℕ := 
  frequency Fruit.Apple + frequency Fruit.Banana + frequency Fruit.Cherry + 
  frequency Fruit.Orange + frequency Fruit.Pear

-- Theorem: The percentage of people who preferred apples is 21%
theorem apple_preference_percentage : 
  (frequency Fruit.Apple : ℚ) / (total_responses : ℚ) * 100 = 21 := by
  sorry

end apple_preference_percentage_l3044_304428


namespace seven_lines_29_regions_l3044_304432

/-- The number of regions created by n lines in a plane, where no two lines are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1) / 2)

/-- Seven straight lines in a plane with no two parallel and no three concurrent divide the plane into 29 regions -/
theorem seven_lines_29_regions : num_regions 7 = 29 := by
  sorry

end seven_lines_29_regions_l3044_304432


namespace at_least_seven_stay_probability_l3044_304421

def total_friends : ℕ := 8
def sure_friends : ℕ := 3
def unsure_friends : ℕ := 5
def stay_probability : ℚ := 1/3

def probability_at_least_seven_stay : ℚ :=
  (Nat.choose unsure_friends 4 * stay_probability^4 * (1 - stay_probability)^1) +
  (stay_probability^5)

theorem at_least_seven_stay_probability :
  probability_at_least_seven_stay = 11/243 :=
sorry

end at_least_seven_stay_probability_l3044_304421


namespace problem_statement_l3044_304467

theorem problem_statement (x y : ℝ) (h : x - 2*y = -5) : 2 - x + 2*y = 7 := by
  sorry

end problem_statement_l3044_304467


namespace monogram_count_l3044_304484

theorem monogram_count : ∀ n : ℕ, n = 12 → (n.choose 2) = 66 := by
  sorry

end monogram_count_l3044_304484


namespace chess_game_probability_l3044_304438

theorem chess_game_probability (draw_prob : ℚ) (b_win_prob : ℚ) (a_win_prob : ℚ) : 
  draw_prob = 1/2 → b_win_prob = 1/3 → a_win_prob = 1 - draw_prob - b_win_prob → a_win_prob = 1/6 := by
  sorry

end chess_game_probability_l3044_304438


namespace expression_evaluation_l3044_304456

theorem expression_evaluation (x y : ℝ) (hx : x = 3) (hy : y = 5) : 
  (3 * x^4 + 2 * y^2 + 10) / 8 = 303 / 8 := by
  sorry

end expression_evaluation_l3044_304456


namespace subset_implies_m_geq_two_l3044_304434

def set_A (m : ℝ) : Set ℝ := {x | x ≤ m}
def set_B : Set ℝ := {x | -2 < x ∧ x ≤ 2}

theorem subset_implies_m_geq_two (m : ℝ) :
  set_B ⊆ set_A m → m ≥ 2 := by
  sorry

end subset_implies_m_geq_two_l3044_304434


namespace fish_comparison_l3044_304482

theorem fish_comparison (x g s r : ℕ) : 
  x > 0 ∧ 
  x = g + s + r ∧ 
  x - g = (2 * x) / 3 - 1 ∧ 
  x - r = (2 * x) / 3 + 4 → 
  s = g + 2 := by
sorry

end fish_comparison_l3044_304482


namespace sum_of_squares_problem_l3044_304415

theorem sum_of_squares_problem (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0)
  (sum_of_squares : a^2 + b^2 + c^2 = 52)
  (sum_of_products : a*b + b*c + c*a = 28) :
  a + b + c = 6 * Real.sqrt 3 := by
sorry

end sum_of_squares_problem_l3044_304415


namespace inequality_proof_l3044_304492

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  a / b + b / c > a / c + c / a := by
sorry

end inequality_proof_l3044_304492


namespace no_solution_for_equation_l3044_304449

theorem no_solution_for_equation : 
  ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 1 / (2 * a + 3 * b)) := by
  sorry

end no_solution_for_equation_l3044_304449


namespace helen_gas_consumption_l3044_304469

/-- Represents the gas consumption for Helen's lawn maintenance --/
def lawn_maintenance_gas_consumption : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ := sorry

/-- The number of times the large lawn is cut --/
def large_lawn_cuts : ℕ := 18

/-- The number of times the small lawn is cut --/
def small_lawn_cuts : ℕ := 14

/-- The number of times the suburban lawn is trimmed --/
def suburban_trims : ℕ := 6

/-- The number of times the leaf blower is used --/
def leaf_blower_uses : ℕ := 2

theorem helen_gas_consumption :
  lawn_maintenance_gas_consumption large_lawn_cuts small_lawn_cuts suburban_trims leaf_blower_uses 3 2 = 22 := by sorry

end helen_gas_consumption_l3044_304469


namespace total_sums_attempted_l3044_304493

/-- Given a student's performance on a set of math problems, calculate the total number of problems attempted. -/
theorem total_sums_attempted
  (correct : ℕ)  -- Number of sums solved correctly
  (h1 : correct = 12)  -- The student solved 12 sums correctly
  (h2 : ∃ wrong : ℕ, wrong = 2 * correct)  -- The student got twice as many sums wrong as right
  : ∃ total : ℕ, total = 3 * correct :=
by sorry

end total_sums_attempted_l3044_304493


namespace divisibility_condition_l3044_304461

theorem divisibility_condition (n : ℕ) : n ≥ 1 → (n^2 ∣ 2^n + 1) ↔ n = 1 ∨ n = 3 := by
  sorry

end divisibility_condition_l3044_304461


namespace game_tie_fraction_l3044_304485

theorem game_tie_fraction (jack_wins emily_wins : ℚ) 
  (h1 : jack_wins = 5 / 12)
  (h2 : emily_wins = 1 / 4) : 
  1 - (jack_wins + emily_wins) = 1 / 3 := by
sorry

end game_tie_fraction_l3044_304485


namespace weight_of_ton_l3044_304424

/-- The weight of a ton in pounds -/
def ton_weight : ℝ := 2000

theorem weight_of_ton (elephant_weight : ℝ) (donkey_weight : ℝ) 
  (h1 : elephant_weight = 3 * ton_weight)
  (h2 : donkey_weight = 0.1 * elephant_weight)
  (h3 : elephant_weight + donkey_weight = 6600) :
  ton_weight = 2000 := by
  sorry

#check weight_of_ton

end weight_of_ton_l3044_304424


namespace sin_cos_difference_equals_half_l3044_304496

theorem sin_cos_difference_equals_half : 
  Real.sin (36 * π / 180) * Real.cos (6 * π / 180) - 
  Real.sin (54 * π / 180) * Real.cos (84 * π / 180) = 1/2 := by
  sorry

end sin_cos_difference_equals_half_l3044_304496


namespace expression_evaluation_l3044_304488

theorem expression_evaluation (a b : ℝ) (h : a^2 + b^2 - 2*a + 4*b = -5) :
  (a - 2*b)*(a^2 + 2*a*b + 4*b^2) - a*(a - 5*b)*(a + 3*b) = 120 := by
  sorry

end expression_evaluation_l3044_304488


namespace trajectory_is_ellipse_l3044_304487

/-- The definition of an ellipse in 2D space -/
def is_ellipse (S : Set (ℝ × ℝ)) (F₁ F₂ : ℝ × ℝ) (c : ℝ) : Prop :=
  ∀ M ∈ S, dist M F₁ + dist M F₂ = c

/-- The set of points M satisfying the given condition -/
def trajectory (F₁ F₂ : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {M | dist M F₁ + dist M F₂ = 8}

theorem trajectory_is_ellipse (F₁ F₂ : ℝ × ℝ) (h : dist F₁ F₂ = 6) :
  is_ellipse (trajectory F₁ F₂) F₁ F₂ 8 := by
  sorry

end trajectory_is_ellipse_l3044_304487


namespace tan_double_angle_l3044_304491

theorem tan_double_angle (α : Real) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2 →
  Real.tan (2 * α) = 3/4 := by
  sorry

end tan_double_angle_l3044_304491


namespace twentyFifth_is_221_l3044_304410

/-- Converts a natural number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The 25th number in base-3 counting sequence -/
def twentyFifthBase3 : List ℕ := toBase3 25

theorem twentyFifth_is_221 : twentyFifthBase3 = [2, 2, 1] := by
  sorry

#eval twentyFifthBase3

end twentyFifth_is_221_l3044_304410


namespace inverse_proportion_l3044_304422

/-- Given that α is inversely proportional to β, prove that if α = -3 when β = -6, 
    then α = 9/4 when β = 8. -/
theorem inverse_proportion (α β : ℝ → ℝ) (k : ℝ) 
    (h1 : ∀ x, α x * β x = k)  -- α is inversely proportional to β
    (h2 : α (-6) = -3)         -- α = -3 when β = -6
    (h3 : β (-6) = -6)         -- β = -6 when β = -6 (implicit in the problem)
    : α 8 = 9/4 := by
  sorry

end inverse_proportion_l3044_304422


namespace min_sum_squares_l3044_304470

theorem min_sum_squares (a b : ℝ) : 
  let A : ℝ × ℝ := (a, 2)
  let B : ℝ × ℝ := (3, b)
  let C : ℝ × ℝ := (2, 3)
  let O : ℝ × ℝ := (0, 0)
  let OB : ℝ × ℝ := (3 - 0, b - 0)
  let AC : ℝ × ℝ := (2 - a, 3 - 2)
  (OB.1 * AC.1 + OB.2 * AC.2 = 0) →
  (∃ (x : ℝ), ∀ (a b : ℝ), a^2 + b^2 ≥ x ∧ (∃ (a₀ b₀ : ℝ), a₀^2 + b₀^2 = x)) ∧
  (∀ (x : ℝ), (∃ (a b : ℝ), a^2 + b^2 = x) → x ≥ 18/5) :=
by sorry

end min_sum_squares_l3044_304470


namespace soda_cost_l3044_304433

/-- The cost of items in cents -/
structure ItemCosts where
  burger : ℕ
  soda : ℕ
  fries : ℕ

/-- Represents the purchase combinations -/
inductive Purchase
  | uri1 : Purchase
  | gen1 : Purchase
  | uri2 : Purchase
  | gen2 : Purchase

/-- The cost of each purchase in cents -/
def purchaseCost (p : Purchase) (costs : ItemCosts) : ℕ :=
  match p with
  | .uri1 => 3 * costs.burger + costs.soda
  | .gen1 => 2 * costs.burger + 3 * costs.soda
  | .uri2 => costs.burger + 2 * costs.fries
  | .gen2 => costs.soda + 3 * costs.fries

theorem soda_cost (costs : ItemCosts) 
  (h1 : purchaseCost .uri1 costs = 390)
  (h2 : purchaseCost .gen1 costs = 440)
  (h3 : purchaseCost .uri2 costs = 230)
  (h4 : purchaseCost .gen2 costs = 270) :
  costs.soda = 234 := by
  sorry

end soda_cost_l3044_304433


namespace decreasing_implies_a_leq_neg_three_l3044_304408

/-- A quadratic function f(x) that is decreasing on (-∞, 4] -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The property that f is decreasing on (-∞, 4] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 4 → f a x > f a y

/-- Theorem stating that if f is decreasing on (-∞, 4], then a ≤ -3 -/
theorem decreasing_implies_a_leq_neg_three (a : ℝ) :
  is_decreasing_on_interval a → a ≤ -3 := by sorry

end decreasing_implies_a_leq_neg_three_l3044_304408


namespace author_paperback_percentage_is_six_percent_l3044_304417

/-- Represents the book sales problem --/
structure BookSales where
  paperback_copies : ℕ
  paperback_price : ℚ
  hardcover_copies : ℕ
  hardcover_price : ℚ
  hardcover_percentage : ℚ
  total_earnings : ℚ

/-- Calculates the author's percentage from paperback sales --/
def paperback_percentage (sales : BookSales) : ℚ :=
  let paperback_sales := sales.paperback_copies * sales.paperback_price
  let hardcover_sales := sales.hardcover_copies * sales.hardcover_price
  let hardcover_earnings := sales.hardcover_percentage * hardcover_sales
  let paperback_earnings := sales.total_earnings - hardcover_earnings
  paperback_earnings / paperback_sales

/-- Theorem stating that the author's percentage from paperback sales is 6% --/
theorem author_paperback_percentage_is_six_percent (sales : BookSales) 
  (h1 : sales.paperback_copies = 32000)
  (h2 : sales.paperback_price = 1/5)
  (h3 : sales.hardcover_copies = 15000)
  (h4 : sales.hardcover_price = 2/5)
  (h5 : sales.hardcover_percentage = 12/100)
  (h6 : sales.total_earnings = 1104) :
  paperback_percentage sales = 6/100 := by
  sorry


end author_paperback_percentage_is_six_percent_l3044_304417


namespace equation_solution_l3044_304475

theorem equation_solution :
  ∃! x : ℝ, x ≠ 3 ∧
    ∃ z : ℝ, z = (x^2 - 9) / (x - 3) ∧ z = 3 * x ∧
    x = 3 / 2 :=
by
  sorry

end equation_solution_l3044_304475


namespace fixed_point_of_exponential_function_l3044_304420

theorem fixed_point_of_exponential_function :
  let f : ℝ → ℝ := λ x => 2^(x + 2) + 1
  f (-2) = 2 := by sorry

end fixed_point_of_exponential_function_l3044_304420


namespace parabola_point_and_focus_l3044_304453

theorem parabola_point_and_focus (m : ℝ) (p : ℝ) : 
  p > 0 →
  ((-3)^2 = 2 * p * m) →
  (m + p / 2)^2 + (3 - p / 2)^2 = 5^2 →
  ((m = 1/2 ∧ p = 9) ∨ (m = 9/2 ∧ p = 1)) :=
by sorry

end parabola_point_and_focus_l3044_304453


namespace least_product_of_two_primes_above_ten_l3044_304409

theorem least_product_of_two_primes_above_ten (p q : ℕ) : 
  Prime p → Prime q → p > 10 → q > 10 → p ≠ q → 
  ∀ r s : ℕ, Prime r → Prime s → r > 10 → s > 10 → r ≠ s → 
  p * q ≤ r * s → p * q = 143 := by sorry

end least_product_of_two_primes_above_ten_l3044_304409


namespace square_plus_reciprocal_square_l3044_304464

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 4) :
  x^2 + 1/x^2 = 14 := by
  sorry

end square_plus_reciprocal_square_l3044_304464


namespace stratified_sample_size_l3044_304445

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  stratum_size : ℕ
  stratum_sample : ℕ
  total_sample : ℕ

/-- 
Theorem: In a stratified sampling scenario with a total population of 750,
where one stratum has 250 members and 5 are sampled from this stratum,
the total sample size is 15.
-/
theorem stratified_sample_size 
  (s : StratifiedSample) 
  (h1 : s.total_population = 750) 
  (h2 : s.stratum_size = 250) 
  (h3 : s.stratum_sample = 5) 
  (h4 : s.stratum_sample / s.stratum_size = s.total_sample / s.total_population) : 
  s.total_sample = 15 := by
sorry

end stratified_sample_size_l3044_304445


namespace geometric_sequence_sixth_term_l3044_304465

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a2 : a 2 = 1)
  (h_a8 : a 8 = a 6 + 2 * a 4) :
  a 6 = 4 := by
sorry


end geometric_sequence_sixth_term_l3044_304465


namespace winnie_lollipops_left_l3044_304423

/-- The number of lollipops left after equal distribution -/
def lollipops_left (total : ℕ) (friends : ℕ) : ℕ :=
  total % friends

theorem winnie_lollipops_left :
  let cherry := 60
  let wintergreen := 145
  let grape := 10
  let shrimp_cocktail := 295
  let total := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 13
  lollipops_left total friends = 3 := by
  sorry

end winnie_lollipops_left_l3044_304423


namespace modified_tic_tac_toe_tie_probability_l3044_304426

theorem modified_tic_tac_toe_tie_probability 
  (amy_win_prob : ℚ)
  (lily_win_prob : ℚ)
  (ben_win_prob : ℚ)
  (h1 : amy_win_prob = 4 / 15)
  (h2 : lily_win_prob = 1 / 5)
  (h3 : ben_win_prob = 1 / 6)
  (h4 : amy_win_prob + lily_win_prob + ben_win_prob < 1) : 
  1 - (amy_win_prob + lily_win_prob + ben_win_prob) = 11 / 30 := by
sorry

end modified_tic_tac_toe_tie_probability_l3044_304426


namespace volleyball_match_probability_l3044_304473

/-- The probability of Team A winning a single set -/
def p_A : ℚ := 2/3

/-- The probability of Team B winning a single set -/
def p_B : ℚ := 1 - p_A

/-- The number of sets Team B has won at the start -/
def initial_B_wins : ℕ := 2

/-- The number of sets needed to win the match -/
def sets_to_win : ℕ := 3

/-- The probability of Team B winning the match given they lead 2:0 -/
def p_B_wins : ℚ := p_B + p_A * p_B + p_A * p_A * p_B

theorem volleyball_match_probability :
  p_B_wins = 19/27 := by sorry

end volleyball_match_probability_l3044_304473


namespace repeating_decimal_equiv_fraction_l3044_304441

def repeating_decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 990 + a / 10

theorem repeating_decimal_equiv_fraction :
  repeating_decimal_to_fraction 2 1 3 = 523 / 2475 ∧
  (∀ m n : ℕ, m ≠ 0 → n ≠ 0 → m / n = 523 / 2475 → m ≥ 523 ∧ n ≥ 2475) :=
sorry

end repeating_decimal_equiv_fraction_l3044_304441


namespace jack_stairs_problem_l3044_304404

/-- Represents the number of steps in each flight of stairs -/
def steps_per_flight : ℕ := 12

/-- Represents the height of each step in inches -/
def step_height : ℕ := 8

/-- Represents the number of flights Jack went down -/
def flights_down : ℕ := 6

/-- Represents how much further down Jack ended up, in feet -/
def final_position : ℕ := 24

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the height of one flight of stairs in inches -/
def flight_height : ℕ := steps_per_flight * step_height

/-- Represents the number of flights Jack went up initially -/
def flights_up : ℕ := 9

theorem jack_stairs_problem :
  flights_up * flight_height = 
  flights_down * flight_height + feet_to_inches final_position :=
by sorry

end jack_stairs_problem_l3044_304404


namespace probability_two_heads_two_tails_prove_probability_two_heads_two_tails_l3044_304458

/-- The probability of getting exactly two heads and two tails when tossing four fair coins -/
theorem probability_two_heads_two_tails : ℚ :=
  3/8

/-- Proof that the probability of getting exactly two heads and two tails
    when tossing four fair coins is 3/8 -/
theorem prove_probability_two_heads_two_tails :
  probability_two_heads_two_tails = 3/8 := by
  sorry

end probability_two_heads_two_tails_prove_probability_two_heads_two_tails_l3044_304458


namespace platform_length_proof_l3044_304446

/-- Proves that given a train with specified speed and length, crossing a platform in a certain time, the platform length is approximately 165 meters. -/
theorem platform_length_proof (train_speed : Real) (train_length : Real) (crossing_time : Real) :
  train_speed = 132 * 1000 / 3600 →
  train_length = 110 →
  crossing_time = 7.499400047996161 →
  ∃ (platform_length : Real),
    (platform_length + train_length) = train_speed * crossing_time ∧
    abs (platform_length - 165) < 1 := by
  sorry

end platform_length_proof_l3044_304446


namespace leibniz_theorem_l3044_304436

/-- Leibniz's Theorem -/
theorem leibniz_theorem (A B C M : ℝ × ℝ) : 
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  3 * dist M G ^ 2 = 
    dist M A ^ 2 + dist M B ^ 2 + dist M C ^ 2 - 
    (1/3) * (dist A B ^ 2 + dist B C ^ 2 + dist C A ^ 2) := by
  sorry

#check leibniz_theorem

end leibniz_theorem_l3044_304436


namespace x_forty_percent_greater_than_88_l3044_304476

theorem x_forty_percent_greater_than_88 :
  ∀ x : ℝ, x = 88 * (1 + 0.4) → x = 123.2 :=
by
  sorry

end x_forty_percent_greater_than_88_l3044_304476


namespace polar_coordinate_equivalence_l3044_304452

/-- 
Given a point in polar coordinates (-5, 5π/7), prove that it is equivalent 
to the point (5, 12π/7) in standard polar coordinate representation, 
where r > 0 and 0 ≤ θ < 2π.
-/
theorem polar_coordinate_equivalence :
  ∀ (r θ : ℝ), 
  r = -5 ∧ θ = (5 * Real.pi) / 7 →
  ∃ (r' θ' : ℝ),
    r' > 0 ∧ 
    0 ≤ θ' ∧ 
    θ' < 2 * Real.pi ∧
    r' = 5 ∧ 
    θ' = (12 * Real.pi) / 7 ∧
    (r * (Real.cos θ), r * (Real.sin θ)) = (r' * (Real.cos θ'), r' * (Real.sin θ')) :=
by sorry

end polar_coordinate_equivalence_l3044_304452
