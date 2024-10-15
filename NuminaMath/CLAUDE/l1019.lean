import Mathlib

namespace NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l1019_101969

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 175 → ∃ (a b : ℕ), a^2 - b^2 = 175 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 625 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l1019_101969


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1019_101943

/-- 
Given an isosceles triangle with two sides of length 15 and a perimeter of 40,
prove that the length of the third side (base) is 10.
-/
theorem isosceles_triangle_base_length 
  (a b c : ℝ) 
  (h1 : a = 15) 
  (h2 : b = 15) 
  (h3 : a + b + c = 40) : 
  c = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1019_101943


namespace NUMINAMATH_CALUDE_rotation_effect_l1019_101924

-- Define a type for the shapes
inductive Shape
  | Triangle
  | Circle
  | Square
  | Pentagon

-- Define a function to represent the initial arrangement
def initial_position (s : Shape) : ℕ :=
  match s with
  | Shape.Triangle => 0
  | Shape.Circle => 1
  | Shape.Square => 2
  | Shape.Pentagon => 3

-- Define a function to represent the position after rotation
def rotated_position (s : Shape) : ℕ :=
  match s with
  | Shape.Triangle => 1
  | Shape.Circle => 2
  | Shape.Square => 3
  | Shape.Pentagon => 0

-- Theorem stating that each shape moves to the next position after rotation
theorem rotation_effect :
  ∀ s : Shape, (rotated_position s) = ((initial_position s) + 1) % 4 :=
by sorry

end NUMINAMATH_CALUDE_rotation_effect_l1019_101924


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1019_101960

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 4 + a 7 = 2)
  (h_product : a 5 * a 6 = -3) :
  a 1 * a 10 = -323 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1019_101960


namespace NUMINAMATH_CALUDE_gmat_question_percentages_l1019_101992

/-- Given percentages of test takers answering questions correctly or incorrectly, 
    prove the percentage that answered both questions correctly. -/
theorem gmat_question_percentages 
  (first_correct : ℝ) 
  (second_correct : ℝ) 
  (neither_correct : ℝ) 
  (h1 : first_correct = 85) 
  (h2 : second_correct = 70) 
  (h3 : neither_correct = 5) : 
  first_correct + second_correct - (100 - neither_correct) = 60 := by
  sorry


end NUMINAMATH_CALUDE_gmat_question_percentages_l1019_101992


namespace NUMINAMATH_CALUDE_prob_sum_div_three_is_seven_ninths_l1019_101919

/-- Represents a biased die where even numbers are twice as likely as odd numbers -/
structure BiasedDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℝ
  /-- Probability of rolling an even number -/
  even_prob : ℝ
  /-- The probabilities sum to 1 -/
  sum_to_one : odd_prob + even_prob = 1
  /-- Even numbers are twice as likely as odd numbers -/
  even_twice_odd : even_prob = 2 * odd_prob

/-- The probability that the sum of three rolls of a biased die is divisible by 3 -/
def prob_sum_div_three (d : BiasedDie) : ℝ :=
  d.even_prob^3 + d.odd_prob^3 + 3 * d.even_prob^2 * d.odd_prob

/-- Theorem: The probability that the sum of three rolls of the biased die is divisible by 3 is 7/9 -/
theorem prob_sum_div_three_is_seven_ninths (d : BiasedDie) :
    prob_sum_div_three d = 7/9 := by
  sorry


end NUMINAMATH_CALUDE_prob_sum_div_three_is_seven_ninths_l1019_101919


namespace NUMINAMATH_CALUDE_ice_cream_profit_l1019_101917

/-- Proves the number of ice cream cones needed to be sold for a specific profit -/
theorem ice_cream_profit (cone_price : ℚ) (expense_ratio : ℚ) (target_profit : ℚ) :
  cone_price = 5 →
  expense_ratio = 4/5 →
  target_profit = 200 →
  (target_profit / (1 - expense_ratio)) / cone_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_profit_l1019_101917


namespace NUMINAMATH_CALUDE_exam_score_proof_l1019_101928

theorem exam_score_proof (total_questions : ℕ) 
                         (correct_score wrong_score : ℤ) 
                         (total_score : ℤ) 
                         (h1 : total_questions = 100)
                         (h2 : correct_score = 5)
                         (h3 : wrong_score = -2)
                         (h4 : total_score = 150) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + 
    wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 50 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_proof_l1019_101928


namespace NUMINAMATH_CALUDE_percent_equivalence_l1019_101934

theorem percent_equivalence : ∃ x : ℚ, (60 / 100 * 500 : ℚ) = x / 100 * 600 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_equivalence_l1019_101934


namespace NUMINAMATH_CALUDE_max_value_sqrt_function_l1019_101978

theorem max_value_sqrt_function (x : ℝ) (h1 : 2 < x) (h2 : x < 5) :
  ∃ (M : ℝ), M = 4 * Real.sqrt 3 ∧ ∀ y, 2 < y → y < 5 → Real.sqrt (3 * y * (8 - y)) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_function_l1019_101978


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_in_binomial_difference_x_cubed_coefficient_is_negative_ten_l1019_101977

theorem x_cubed_coefficient_in_binomial_difference : ℤ :=
  let n₁ : ℕ := 5
  let n₂ : ℕ := 6
  let k : ℕ := 3
  let coeff₁ : ℤ := (Nat.choose n₁ k : ℤ)
  let coeff₂ : ℤ := (Nat.choose n₂ k : ℤ)
  coeff₁ - coeff₂

theorem x_cubed_coefficient_is_negative_ten :
  x_cubed_coefficient_in_binomial_difference = -10 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_in_binomial_difference_x_cubed_coefficient_is_negative_ten_l1019_101977


namespace NUMINAMATH_CALUDE_johns_initial_marbles_l1019_101989

/-- Given that:
    - Ben had 18 marbles initially
    - John had an unknown number of marbles initially
    - Ben gave half of his marbles to John
    - After the transfer, John had 17 more marbles than Ben
    Prove that John had 17 marbles initially -/
theorem johns_initial_marbles :
  ∀ (john_initial : ℕ),
  let ben_initial : ℕ := 18
  let ben_gave : ℕ := ben_initial / 2
  let ben_final : ℕ := ben_initial - ben_gave
  let john_final : ℕ := john_initial + ben_gave
  john_final = ben_final + 17 →
  john_initial = 17 := by
sorry

end NUMINAMATH_CALUDE_johns_initial_marbles_l1019_101989


namespace NUMINAMATH_CALUDE_orange_crates_pigeonhole_l1019_101963

theorem orange_crates_pigeonhole (total_crates : ℕ) (min_oranges max_oranges : ℕ) :
  total_crates = 200 →
  min_oranges = 100 →
  max_oranges = 130 →
  ∃ (n : ℕ), n ≥ 7 ∧ 
    ∃ (k : ℕ), min_oranges ≤ k ∧ k ≤ max_oranges ∧
      (∃ (crates : Finset (Fin total_crates)), crates.card = n ∧ 
        ∀ c ∈ crates, ∃ f : Fin total_crates → ℕ, f c = k) :=
by sorry

end NUMINAMATH_CALUDE_orange_crates_pigeonhole_l1019_101963


namespace NUMINAMATH_CALUDE_total_distinct_students_l1019_101985

/-- Represents the number of distinct students in the mathematics competition --/
def distinct_students (germain newton young germain_newton_overlap germain_young_overlap : ℕ) : ℕ :=
  germain + newton + young - germain_newton_overlap - germain_young_overlap

/-- Theorem stating that the total number of distinct students is 32 --/
theorem total_distinct_students :
  distinct_students 13 10 12 2 1 = 32 := by
  sorry

#eval distinct_students 13 10 12 2 1

end NUMINAMATH_CALUDE_total_distinct_students_l1019_101985


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l1019_101926

/-- The swimming speed of a person in still water, given their performance against a current. -/
theorem swimming_speed_in_still_water (water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : water_speed = 4)
  (h2 : distance = 8)
  (h3 : time = 2)
  (h4 : (swimming_speed - water_speed) * time = distance) :
  swimming_speed = 8 :=
sorry

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l1019_101926


namespace NUMINAMATH_CALUDE_iron_bar_width_is_48_l1019_101970

-- Define the dimensions of the iron bar
def iron_bar_length : ℝ := 12
def iron_bar_height : ℝ := 6

-- Define the number of iron bars and iron balls
def num_iron_bars : ℕ := 10
def num_iron_balls : ℕ := 720

-- Define the volume of each iron ball
def iron_ball_volume : ℝ := 8

-- Theorem statement
theorem iron_bar_width_is_48 (w : ℝ) :
  (num_iron_bars : ℝ) * (iron_bar_length * w * iron_bar_height) =
  (num_iron_balls : ℝ) * iron_ball_volume →
  w = 48 := by sorry

end NUMINAMATH_CALUDE_iron_bar_width_is_48_l1019_101970


namespace NUMINAMATH_CALUDE_carbon_atoms_count_l1019_101952

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Atomic weights of elements in atomic mass units (amu) -/
def atomic_weight : CompoundComposition → ℕ
  | ⟨c, h, o⟩ => 12 * c + 1 * h + 16 * o

/-- The compound has 1 Hydrogen and 1 Oxygen atom -/
def compound_constraints (comp : CompoundComposition) : Prop :=
  comp.hydrogen = 1 ∧ comp.oxygen = 1

/-- The molecular weight of the compound is 65 amu -/
def molecular_weight_constraint (comp : CompoundComposition) : Prop :=
  atomic_weight comp = 65

theorem carbon_atoms_count :
  ∀ comp : CompoundComposition,
    compound_constraints comp →
    molecular_weight_constraint comp →
    comp.carbon = 4 :=
by sorry

end NUMINAMATH_CALUDE_carbon_atoms_count_l1019_101952


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1019_101987

/-- Two lines are parallel if their normal vectors are proportional -/
def parallel (a b c d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a = k * d ∧ b = k * e

/-- Two lines coincide if their coefficients are proportional -/
def coincide (a b c d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a = k * d ∧ b = k * e ∧ c = k * f

theorem parallel_lines_a_value (a : ℝ) :
  parallel a (a + 2) 2 1 a 1 ∧ ¬ coincide a (a + 2) 2 1 a 1 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1019_101987


namespace NUMINAMATH_CALUDE_f_properties_l1019_101982

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) / (sin x + 2)

theorem f_properties (a : ℝ) (h : a ≥ -2) :
  (∀ x ∈ Set.Icc 0 (π/2), Monotone (f π)) ∧
  (∀ x ∈ Set.Icc 0 (π/2), f a x ≤ π/6 - a/3) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1019_101982


namespace NUMINAMATH_CALUDE_lucas_book_purchase_l1019_101902

theorem lucas_book_purchase (total_money : ℚ) (total_books : ℕ) (book_price : ℚ) 
    (h1 : total_money > 0)
    (h2 : total_books > 0)
    (h3 : book_price > 0)
    (h4 : (1 / 4 : ℚ) * total_money = (1 / 2 : ℚ) * total_books * book_price) : 
  total_money - (total_books * book_price) = (1 / 2 : ℚ) * total_money := by
sorry

end NUMINAMATH_CALUDE_lucas_book_purchase_l1019_101902


namespace NUMINAMATH_CALUDE_standard_deck_three_card_selections_l1019_101994

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cardsPerSuit : Nat)
  (redSuits : Nat)
  (blackSuits : Nat)

/-- A standard deck of 52 cards -/
def standardDeck : Deck :=
  { cards := 52
  , suits := 4
  , cardsPerSuit := 13
  , redSuits := 2
  , blackSuits := 2 }

/-- The number of ways to select three different cards from a deck, where order matters -/
def threeCardSelections (d : Deck) : Nat :=
  d.cards * (d.cards - 1) * (d.cards - 2)

/-- Theorem stating the number of ways to select three different cards from a standard deck -/
theorem standard_deck_three_card_selections :
  threeCardSelections standardDeck = 132600 := by
  sorry

end NUMINAMATH_CALUDE_standard_deck_three_card_selections_l1019_101994


namespace NUMINAMATH_CALUDE_triangle_area_l1019_101981

/-- Given a triangle with perimeter 28 and inradius 2.5, prove that its area is 35 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
    (h1 : perimeter = 28) 
    (h2 : inradius = 2.5) 
    (h3 : area = inradius * (perimeter / 2)) : area = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1019_101981


namespace NUMINAMATH_CALUDE_teacher_age_l1019_101921

theorem teacher_age (n : ℕ) (initial_avg : ℚ) (transfer_age : ℕ) (new_avg : ℚ) :
  n = 45 →
  initial_avg = 14 →
  transfer_age = 15 →
  new_avg = 14.66 →
  let remaining_students := n - 1
  let total_age := n * initial_avg
  let remaining_age := total_age - transfer_age
  let teacher_age := (remaining_students + 1) * new_avg - remaining_age
  (∀ p : ℕ, Prime p → p > teacher_age → p ≥ 17) ∧
  Prime 17 ∧
  17 > teacher_age :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l1019_101921


namespace NUMINAMATH_CALUDE_vector_AB_and_magnitude_l1019_101986

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 3)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem vector_AB_and_magnitude :
  vector_AB = (1, 1) ∧ 
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_AB_and_magnitude_l1019_101986


namespace NUMINAMATH_CALUDE_triangle_side_length_l1019_101912

-- Define the triangle ABC
def triangle (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions to ensure it's a valid triangle
  true

-- Define the angle measure in degrees
def angle_measure (A B C : ℝ × ℝ) : ℝ := 
  -- Add definition for angle measure
  0

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ :=
  -- Add definition for distance
  0

theorem triangle_side_length 
  (A B C : ℝ × ℝ) 
  (h_triangle : triangle A B C)
  (h_angle_B : angle_measure A B C = 45)
  (h_angle_C : angle_measure B C A = 80)
  (h_side_AC : distance A C = 5) :
  distance B C = (10 * Real.sin (55 * π / 180)) / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1019_101912


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l1019_101958

theorem complex_sum_theorem (B Q R T : ℂ) : 
  B = 3 - 2*I ∧ Q = -5 + 3*I ∧ R = 2*I ∧ T = -1 + 2*I →
  B - Q + R + T = 7 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l1019_101958


namespace NUMINAMATH_CALUDE_pencil_cost_l1019_101949

/-- The cost of a pencil given initial and remaining amounts -/
theorem pencil_cost (initial : ℕ) (remaining : ℕ) (h : initial = 15 ∧ remaining = 4) :
  initial - remaining = 11 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l1019_101949


namespace NUMINAMATH_CALUDE_mercury_radius_scientific_notation_l1019_101959

/-- Given a number in decimal notation, returns its scientific notation as a pair (a, n) where a is the coefficient and n is the exponent. -/
def toScientificNotation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem mercury_radius_scientific_notation :
  toScientificNotation 2440000 = (2.44, 6) :=
sorry

end NUMINAMATH_CALUDE_mercury_radius_scientific_notation_l1019_101959


namespace NUMINAMATH_CALUDE_max_sum_surrounding_45_l1019_101909

theorem max_sum_surrounding_45 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℕ) : 
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧ a₁ ≠ a₈ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧ a₂ ≠ a₈ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧ a₃ ≠ a₈ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧ a₄ ≠ a₈ ∧
  a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧ a₅ ≠ a₈ ∧
  a₆ ≠ a₇ ∧ a₆ ≠ a₈ ∧
  a₇ ≠ a₈ ∧
  0 < a₁ ∧ 0 < a₂ ∧ 0 < a₃ ∧ 0 < a₄ ∧ 0 < a₅ ∧ 0 < a₆ ∧ 0 < a₇ ∧ 0 < a₈ ∧
  a₁ * 45 * a₅ = 3240 ∧
  a₂ * 45 * a₆ = 3240 ∧
  a₃ * 45 * a₇ = 3240 ∧
  a₄ * 45 * a₈ = 3240 →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ ≤ 160 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_surrounding_45_l1019_101909


namespace NUMINAMATH_CALUDE_existence_of_a_sequence_l1019_101923

theorem existence_of_a_sequence (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ (i j k : Fin n), x i j + x j k + x k i = 0) :
  ∃ (a : Fin n → ℝ), ∀ (i j : Fin n), x i j = a i - a j := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_sequence_l1019_101923


namespace NUMINAMATH_CALUDE_stratified_sample_elderly_count_l1019_101945

/-- Represents the number of teachers in a sample -/
structure TeacherSample where
  young : ℕ
  elderly : ℕ

/-- Represents the ratio of young to elderly teachers -/
structure TeacherRatio where
  young : ℕ
  elderly : ℕ

/-- 
Given a stratified sample of teachers where:
- The ratio of young to elderly teachers is 16:9
- There are 320 young teachers in the sample
Prove that there are 180 elderly teachers in the sample
-/
theorem stratified_sample_elderly_count 
  (ratio : TeacherRatio) 
  (sample : TeacherSample) :
  ratio.young = 16 →
  ratio.elderly = 9 →
  sample.young = 320 →
  (ratio.young : ℚ) / ratio.elderly = sample.young / sample.elderly →
  sample.elderly = 180 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_elderly_count_l1019_101945


namespace NUMINAMATH_CALUDE_simplify_expression_l1019_101965

theorem simplify_expression (w : ℝ) : w - 2*w + 4*w - 5*w + 3 - 5 + 7 - 9 = -2*w - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1019_101965


namespace NUMINAMATH_CALUDE_jenny_walking_distance_l1019_101962

theorem jenny_walking_distance (ran_distance : ℝ) (extra_distance : ℝ) :
  ran_distance = 0.6 →
  extra_distance = 0.2 →
  ran_distance = (ran_distance - extra_distance) + extra_distance →
  ran_distance - extra_distance = 0.4 :=
by
  sorry

end NUMINAMATH_CALUDE_jenny_walking_distance_l1019_101962


namespace NUMINAMATH_CALUDE_monday_calls_l1019_101948

/-- Represents the number of calls answered on each day of the work week -/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day -/
def averageCalls : ℕ := 40

/-- The number of working days in a week -/
def workDays : ℕ := 5

/-- Jean's call data for the week -/
def jeanCalls : WeekCalls := {
  monday := 0,  -- We don't know this value yet
  tuesday := 46,
  wednesday := 27,
  thursday := 61,
  friday := 31
}

theorem monday_calls : jeanCalls.monday = 35 := by sorry

end NUMINAMATH_CALUDE_monday_calls_l1019_101948


namespace NUMINAMATH_CALUDE_larger_number_proof_l1019_101903

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 17) (h2 : x - y = 7) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1019_101903


namespace NUMINAMATH_CALUDE_y_value_when_x_is_zero_l1019_101935

theorem y_value_when_x_is_zero (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = 0) : 
  y = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_y_value_when_x_is_zero_l1019_101935


namespace NUMINAMATH_CALUDE_reciprocal_and_abs_of_negative_one_sixth_l1019_101976

theorem reciprocal_and_abs_of_negative_one_sixth :
  let x : ℚ := -1/6
  let reciprocal : ℚ := 1/x
  (reciprocal = -6) ∧ (abs reciprocal = 6) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_abs_of_negative_one_sixth_l1019_101976


namespace NUMINAMATH_CALUDE_angle_FAG_measure_l1019_101991

-- Define the structure of the geometric configuration
structure GeometricConfiguration where
  -- Triangle ABC is equilateral
  triangle_ABC_equilateral : Bool
  -- BCDFG is a regular pentagon
  pentagon_BCDFG_regular : Bool
  -- Triangle ABC and pentagon BCDFG share side BC
  shared_side_BC : Bool

-- Define the theorem
theorem angle_FAG_measure (config : GeometricConfiguration) 
  (h1 : config.triangle_ABC_equilateral = true)
  (h2 : config.pentagon_BCDFG_regular = true)
  (h3 : config.shared_side_BC = true) :
  ∃ (angle_FAG : ℝ), angle_FAG = 36 := by
  sorry

end NUMINAMATH_CALUDE_angle_FAG_measure_l1019_101991


namespace NUMINAMATH_CALUDE_prime_power_equation_solutions_l1019_101999

theorem prime_power_equation_solutions :
  ∀ (p x y : ℕ),
    Prime p →
    x > 0 →
    y > 0 →
    p^x = y^3 + 1 →
    ((p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_equation_solutions_l1019_101999


namespace NUMINAMATH_CALUDE_no_valid_covering_exists_l1019_101971

/-- Represents a unit square in the chain --/
structure UnitSquare where
  id : Nat
  left_neighbor : Option Nat
  right_neighbor : Option Nat

/-- Represents the chain of squares --/
def SquareChain := List UnitSquare

/-- Represents a vertex on the cube --/
structure CubeVertex where
  x : Fin 4
  y : Fin 4
  z : Fin 4

/-- Represents the 3x3x3 cube --/
def Cube := Set CubeVertex

/-- A covering is a mapping from squares to positions on the cube surface --/
def Covering := UnitSquare → Option CubeVertex

/-- Checks if a covering is valid according to the problem constraints --/
def is_valid_covering (chain : SquareChain) (cube : Cube) (covering : Covering) : Prop :=
  sorry

/-- The main theorem stating that no valid covering exists --/
theorem no_valid_covering_exists (chain : SquareChain) (cube : Cube) :
  chain.length = 54 → ¬∃ (covering : Covering), is_valid_covering chain cube covering :=
sorry

end NUMINAMATH_CALUDE_no_valid_covering_exists_l1019_101971


namespace NUMINAMATH_CALUDE_range_of_g_l1019_101980

noncomputable def g (x : ℝ) : ℝ := 1 / x^2 + 3

theorem range_of_g :
  Set.range g = Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l1019_101980


namespace NUMINAMATH_CALUDE_line_intercept_l1019_101966

/-- Given a line y = ax + b passing through the points (3, -2) and (7, 14), prove that b = -14 -/
theorem line_intercept (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x + b) →   -- Definition of the line
  (-2 : ℝ) = a * 3 + b →         -- Line passes through (3, -2)
  (14 : ℝ) = a * 7 + b →         -- Line passes through (7, 14)
  b = -14 := by sorry

end NUMINAMATH_CALUDE_line_intercept_l1019_101966


namespace NUMINAMATH_CALUDE_inverse_f_f_condition_l1019_101964

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2)^2

-- State the theorem
theorem inverse_f (x : ℝ) (h : x ≥ -1) : 
  f⁻¹ (x + 1) = 2 - Real.sqrt (x + 1) := by sorry

-- Define the domain of f
def dom_f : Set ℝ := {x : ℝ | x ≤ 2}

-- State the condition given in the problem
theorem f_condition (x : ℝ) (h : x ≤ 1) : 
  f (x + 1) = (x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_inverse_f_f_condition_l1019_101964


namespace NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l1019_101953

/-- A regular polygon with interior angles measuring 150° has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ θ : ℝ, θ = 150 → n * θ = (n - 2) * 180) →
  n = 12 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l1019_101953


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l1019_101967

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_parallel
  (a b : Line) (α β : Plane)
  (different_lines : a ≠ b)
  (different_planes : α ≠ β)
  (a_perp_α : perpendicular a α)
  (b_perp_β : perpendicular b β)
  (α_parallel_β : parallel_planes α β) :
  parallel_lines a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l1019_101967


namespace NUMINAMATH_CALUDE_binomial_floor_divisibility_l1019_101925

theorem binomial_floor_divisibility (n p : ℕ) (h1 : n ≥ p) (h2 : Nat.Prime (50 * p)) : 
  p ∣ (Nat.choose n p - n / p) :=
by sorry

end NUMINAMATH_CALUDE_binomial_floor_divisibility_l1019_101925


namespace NUMINAMATH_CALUDE_system_solution_l1019_101947

theorem system_solution (x y k : ℝ) : 
  x + 3*y = 2*k + 1 → 
  x - y = 1 → 
  x = -y → 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1019_101947


namespace NUMINAMATH_CALUDE_all_propositions_true_l1019_101988

-- Define the original proposition
def original_proposition (x : ℝ) : Prop :=
  (x = 2 ∨ x = -3) → (x - 2) * (x + 3) = 0

-- Define the converse
def converse (x : ℝ) : Prop :=
  (x - 2) * (x + 3) = 0 → (x = 2 ∨ x = -3)

-- Define the inverse
def inverse (x : ℝ) : Prop :=
  (x ≠ 2 ∧ x ≠ -3) → (x - 2) * (x + 3) ≠ 0

-- Define the contrapositive
def contrapositive (x : ℝ) : Prop :=
  (x - 2) * (x + 3) ≠ 0 → (x ≠ 2 ∧ x ≠ -3)

-- Theorem stating that all propositions are true for all real numbers
theorem all_propositions_true :
  ∀ x : ℝ, original_proposition x ∧ converse x ∧ inverse x ∧ contrapositive x :=
by sorry


end NUMINAMATH_CALUDE_all_propositions_true_l1019_101988


namespace NUMINAMATH_CALUDE_triangle_side_length_l1019_101904

theorem triangle_side_length (n : ℕ) : 
  (7 + 11 + n > 35) ∧ 
  (7 + 11 > n) ∧ 
  (7 + n > 11) ∧ 
  (11 + n > 7) →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1019_101904


namespace NUMINAMATH_CALUDE_history_book_cost_l1019_101931

theorem history_book_cost 
  (total_books : ℕ) 
  (math_book_cost : ℕ) 
  (total_price : ℕ) 
  (math_books_bought : ℕ) 
  (h1 : total_books = 80)
  (h2 : math_book_cost = 4)
  (h3 : total_price = 390)
  (h4 : math_books_bought = 10) :
  (total_price - math_books_bought * math_book_cost) / (total_books - math_books_bought) = 5 := by
sorry

end NUMINAMATH_CALUDE_history_book_cost_l1019_101931


namespace NUMINAMATH_CALUDE_phone_not_answered_probability_l1019_101979

theorem phone_not_answered_probability 
  (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.1) 
  (h2 : p2 = 0.25) 
  (h3 : p3 = 0.45) : 
  1 - (p1 + p2 + p3) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_phone_not_answered_probability_l1019_101979


namespace NUMINAMATH_CALUDE_rational_function_value_l1019_101972

-- Define the function types
def linear_function (α : Type*) [Ring α] := α → α
def quadratic_function (α : Type*) [Ring α] := α → α
def rational_function (α : Type*) [Field α] := α → α

-- Define the properties of the rational function
def has_vertical_asymptotes (f : rational_function ℝ) (a b : ℝ) : Prop :=
  ∀ (x : ℝ), x ≠ a ∧ x ≠ b → f x ≠ 0

def passes_through (f : rational_function ℝ) (x y : ℝ) : Prop :=
  f x = y

-- Main theorem
theorem rational_function_value
  (p : linear_function ℝ)
  (q : quadratic_function ℝ)
  (f : rational_function ℝ)
  (h1 : ∀ (x : ℝ), f x = p x / q x)
  (h2 : has_vertical_asymptotes f (-1) 4)
  (h3 : passes_through f 0 0)
  (h4 : passes_through f 1 (-3)) :
  p (-2) / q (-2) = -6 :=
sorry

end NUMINAMATH_CALUDE_rational_function_value_l1019_101972


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1019_101929

-- Define the solution set
def solution_set : Set ℝ := {x | x > 3 ∨ x < -1}

-- State the theorem
theorem absolute_value_inequality :
  {x : ℝ | |x - 1| > 2} = solution_set := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1019_101929


namespace NUMINAMATH_CALUDE_constant_molecular_weight_l1019_101933

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight : ℝ := 3264

/-- The number of moles of the compound -/
def number_of_moles : ℝ := 8

/-- Theorem: The molecular weight of a compound remains constant regardless of the number of moles -/
theorem constant_molecular_weight : 
  molecular_weight = molecular_weight * (number_of_moles / number_of_moles) :=
by sorry

end NUMINAMATH_CALUDE_constant_molecular_weight_l1019_101933


namespace NUMINAMATH_CALUDE_canoe_weight_problem_l1019_101954

theorem canoe_weight_problem (canoe_capacity : ℕ) (dog_weight_ratio : ℚ) (total_weight : ℕ) :
  canoe_capacity = 6 →
  dog_weight_ratio = 1/4 →
  total_weight = 595 →
  ∃ (person_weight : ℕ),
    person_weight = 140 ∧
    (↑(2 * canoe_capacity) / 3 : ℚ).floor * person_weight + 
    (dog_weight_ratio * person_weight).num / (dog_weight_ratio * person_weight).den = total_weight :=
by sorry

end NUMINAMATH_CALUDE_canoe_weight_problem_l1019_101954


namespace NUMINAMATH_CALUDE_egg_transfer_proof_l1019_101974

/-- Proves that transferring 24 eggs from basket B to basket A will make the number of eggs in basket A twice the number of eggs in basket B -/
theorem egg_transfer_proof (initial_A initial_B transferred : ℕ) 
  (h1 : initial_A = 54)
  (h2 : initial_B = 63)
  (h3 : transferred = 24) :
  initial_A + transferred = 2 * (initial_B - transferred) := by
  sorry

end NUMINAMATH_CALUDE_egg_transfer_proof_l1019_101974


namespace NUMINAMATH_CALUDE_set_equality_and_range_of_a_l1019_101905

-- Define the sets
def M : Set ℝ := {x | (x + 3)^2 ≤ 0}
def N : Set ℝ := {x | x^2 + x - 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 5 - a}
def A : Set ℝ := (Set.univ \ M) ∩ N

-- State the theorem
theorem set_equality_and_range_of_a :
  (A = {2}) ∧
  (∀ a : ℝ, (A ∪ B a = A) ↔ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_and_range_of_a_l1019_101905


namespace NUMINAMATH_CALUDE_exact_sixty_possible_greater_than_sixty_possible_l1019_101908

/-- Represents the number of pieces a single piece of paper can be cut into -/
inductive Cut
  | eight : Cut
  | twelve : Cut

/-- Represents a sequence of cuts applied to the original piece of paper -/
def CutSequence := List Cut

/-- Calculates the number of pieces resulting from applying a sequence of cuts -/
def num_pieces (cuts : CutSequence) : ℕ :=
  cuts.foldl (λ acc cut => match cut with
    | Cut.eight => acc * 8
    | Cut.twelve => acc * 12) 1

/-- Theorem stating that it's possible to obtain exactly 60 pieces -/
theorem exact_sixty_possible : ∃ (cuts : CutSequence), num_pieces cuts = 60 := by
  sorry

/-- Theorem stating that it's possible to obtain any number of pieces greater than 60 -/
theorem greater_than_sixty_possible (n : ℕ) (h : n > 60) : 
  ∃ (cuts : CutSequence), num_pieces cuts = n := by
  sorry

end NUMINAMATH_CALUDE_exact_sixty_possible_greater_than_sixty_possible_l1019_101908


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1019_101914

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the line L
def line_L (k x y : ℝ) : Prop := y = k*x - 3*k + 1

-- Theorem statement
theorem line_intersects_circle :
  ∀ (k : ℝ), ∃ (x y : ℝ), circle_C x y ∧ line_L k x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1019_101914


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l1019_101901

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces -/
structure RectangularPrism where
  -- We don't need to define any specific properties here

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def num_corners (rp : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- Theorem: The sum of edges, corners, and faces of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  num_edges rp + num_corners rp + num_faces rp = 26 := by
  sorry

#check rectangular_prism_sum

end NUMINAMATH_CALUDE_rectangular_prism_sum_l1019_101901


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1019_101940

/-- A function f from non-negative reals to reals satisfying f(x + y) = f(x) * f(y) -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≥ 0 → y ≥ 0 → f (x + y) = f x * f y

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h1 : FunctionalEquation f)
  (h2 : ∀ x, x ≥ 0 → f x ≥ 0)
  (h3 : f 3 = f 1 ^ 3) :
  ∀ c : ℝ, c ≥ 0 → ∃ g : ℝ → ℝ, FunctionalEquation g ∧ g 1 = c ∧ g 3 = g 1 ^ 3 :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1019_101940


namespace NUMINAMATH_CALUDE_seven_students_distribution_l1019_101950

/-- The number of ways to distribute n students into two dormitories,
    with each dormitory having at least m students -/
def distribution_count (n : ℕ) (m : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 112 ways to distribute 7 students into two dormitories,
    with each dormitory having at least 2 students -/
theorem seven_students_distribution :
  distribution_count 7 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_seven_students_distribution_l1019_101950


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1019_101990

-- Define the function f
variable {f : ℝ → ℝ}

-- Define what it means for f to have an extreme value at x
def has_extreme_value (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x

-- Define what it means for f to be differentiable
def is_differentiable (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, DifferentiableAt ℝ f x

-- Define the proposition p
def p (f : ℝ → ℝ) (x : ℝ) : Prop :=
  has_extreme_value f x

-- Define the proposition q
def q (f : ℝ → ℝ) (x : ℝ) : Prop :=
  is_differentiable f ∧ deriv f x = 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ f : ℝ → ℝ, ∀ x : ℝ, p f x → q f x) ∧
  (∃ f : ℝ → ℝ, ∃ x : ℝ, q f x ∧ ¬p f x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1019_101990


namespace NUMINAMATH_CALUDE_prob_diameter_given_smoothness_correct_prob_smoothness_given_diameter_correct_l1019_101920

/-- Represents the total number of circular parts -/
def total_parts : ℕ := 100

/-- Represents the number of parts with qualified diameters -/
def qualified_diameter : ℕ := 98

/-- Represents the number of parts with qualified smoothness -/
def qualified_smoothness : ℕ := 96

/-- Represents the number of parts with both qualified diameter and smoothness -/
def qualified_both : ℕ := 94

/-- Calculates the probability of qualified diameter given qualified smoothness -/
def prob_diameter_given_smoothness : ℚ :=
  qualified_both / qualified_smoothness

/-- Calculates the probability of qualified smoothness given qualified diameter -/
def prob_smoothness_given_diameter : ℚ :=
  qualified_both / qualified_diameter

theorem prob_diameter_given_smoothness_correct :
  prob_diameter_given_smoothness = 94 / 96 := by sorry

theorem prob_smoothness_given_diameter_correct :
  prob_smoothness_given_diameter = 94 / 98 := by sorry

end NUMINAMATH_CALUDE_prob_diameter_given_smoothness_correct_prob_smoothness_given_diameter_correct_l1019_101920


namespace NUMINAMATH_CALUDE_clara_cookie_sales_clara_total_cookies_l1019_101955

theorem clara_cookie_sales : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | cookies_per_box1, cookies_per_box2, cookies_per_box3,
    boxes_sold1, boxes_sold2, boxes_sold3 =>
  cookies_per_box1 * boxes_sold1 +
  cookies_per_box2 * boxes_sold2 +
  cookies_per_box3 * boxes_sold3

theorem clara_total_cookies :
  clara_cookie_sales 12 20 16 50 80 70 = 3320 := by
  sorry

end NUMINAMATH_CALUDE_clara_cookie_sales_clara_total_cookies_l1019_101955


namespace NUMINAMATH_CALUDE_reading_assignment_valid_l1019_101916

/-- Represents the reading assignment for Alice, Bob, and Chandra -/
structure ReadingAssignment where
  alice_pages : ℕ
  bob_pages : ℕ
  chandra_pages : ℕ
  alice_speed : ℕ
  bob_speed : ℕ
  chandra_speed : ℕ
  total_pages : ℕ

/-- Calculates the time spent reading for a given number of pages and reading speed -/
def reading_time (pages : ℕ) (speed : ℕ) : ℕ := pages * speed

/-- Proves that the given reading assignment satisfies the conditions -/
theorem reading_assignment_valid (ra : ReadingAssignment) 
  (h_alice : ra.alice_pages = 416)
  (h_bob : ra.bob_pages = 208)
  (h_chandra : ra.chandra_pages = 276)
  (h_alice_speed : ra.alice_speed = 18)
  (h_bob_speed : ra.bob_speed = 36)
  (h_chandra_speed : ra.chandra_speed = 27)
  (h_total : ra.total_pages = 900) : 
  ra.alice_pages + ra.bob_pages + ra.chandra_pages = ra.total_pages ∧
  reading_time ra.alice_pages ra.alice_speed = reading_time ra.bob_pages ra.bob_speed ∧
  reading_time ra.bob_pages ra.bob_speed = reading_time ra.chandra_pages ra.chandra_speed :=
by sorry


end NUMINAMATH_CALUDE_reading_assignment_valid_l1019_101916


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l1019_101941

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of unique ice cream flavors -/
def ice_cream_flavors : ℕ := distribute 5 4

theorem ice_cream_flavors_count : ice_cream_flavors = 56 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l1019_101941


namespace NUMINAMATH_CALUDE_distance_to_complex_point_l1019_101907

open Complex

theorem distance_to_complex_point :
  let z : ℂ := 3 / (2 - I)^2
  abs z = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_distance_to_complex_point_l1019_101907


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1019_101906

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31/13 - 1/13 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1019_101906


namespace NUMINAMATH_CALUDE_linear_function_point_values_l1019_101937

theorem linear_function_point_values (a m n b : ℝ) :
  (∃ (m n : ℝ), n = 2 * m + b ∧ a = 2 * (1/2) + b) →
  (∀ (m n : ℝ), n = 2 * m + b → m * n ≥ -8) →
  (∃ (m n : ℝ), n = 2 * m + b ∧ m * n = -8) →
  (a = -7 ∨ a = 9) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_point_values_l1019_101937


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1019_101983

theorem sum_of_numbers (x y : ℝ) : y = 2 * x - 3 ∧ y = 33 → x + y = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1019_101983


namespace NUMINAMATH_CALUDE_resort_tips_fraction_l1019_101998

theorem resort_tips_fraction (average_tips : ℝ) (h : average_tips > 0) :
  let other_months_total := 6 * average_tips
  let august_tips := 6 * average_tips
  let total_tips := other_months_total + august_tips
  august_tips / total_tips = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_resort_tips_fraction_l1019_101998


namespace NUMINAMATH_CALUDE_game_cost_l1019_101944

/-- 
Given:
- Frank's initial money: 11 dollars
- Frank's allowance: 14 dollars
- Frank's final money: 22 dollars

Prove that the cost of the new game is 3 dollars.
-/
theorem game_cost (initial_money : ℕ) (allowance : ℕ) (final_money : ℕ)
  (h1 : initial_money = 11)
  (h2 : allowance = 14)
  (h3 : final_money = 22) :
  initial_money - (final_money - allowance) = 3 :=
by sorry

end NUMINAMATH_CALUDE_game_cost_l1019_101944


namespace NUMINAMATH_CALUDE_license_plate_count_l1019_101930

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of characters (letters + digits) -/
def num_chars : ℕ := num_letters + num_digits

/-- The number of possible license plates -/
def num_license_plates : ℕ := num_letters * num_chars * 1 * num_digits

theorem license_plate_count :
  num_license_plates = 9360 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l1019_101930


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1019_101911

def f (a b x : ℝ) : ℝ := 2 * x^2 + a * x + b

theorem quadratic_function_theorem (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) → -- f is an even function
  f a b 1 = -3 →                -- f(1) = -3
  (∀ x, f a b x = 2 * x^2 - 5) ∧ -- f(x) = 2x² - 5
  {x : ℝ | 2 * x^2 - 5 ≥ 3 * x + 4} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3} := by
sorry


end NUMINAMATH_CALUDE_quadratic_function_theorem_l1019_101911


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1019_101956

/-- An arithmetic sequence with given properties has 13 terms -/
theorem arithmetic_sequence_terms (a d : ℝ) (n : ℕ) 
  (h1 : 3 * a + 3 * d = 34)
  (h2 : 3 * a + 3 * d * (n - 1) = 146)
  (h3 : n * (2 * a + (n - 1) * d) / 2 = 390) :
  n = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1019_101956


namespace NUMINAMATH_CALUDE_not_right_triangle_l1019_101984

theorem not_right_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 3 * C) 
  (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l1019_101984


namespace NUMINAMATH_CALUDE_mary_cut_ten_roses_l1019_101995

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Proof that Mary cut 10 roses from her garden -/
theorem mary_cut_ten_roses (initial_roses final_roses : ℕ)
  (h1 : initial_roses = 6)
  (h2 : final_roses = 16) :
  roses_cut initial_roses final_roses = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_cut_ten_roses_l1019_101995


namespace NUMINAMATH_CALUDE_solve_for_y_l1019_101997

theorem solve_for_y (x y n : ℝ) (h : x ≠ y) (h_n : n = (3 * x * y) / (x - y)) :
  y = (n * x) / (3 * x + n) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1019_101997


namespace NUMINAMATH_CALUDE_students_neither_football_nor_cricket_l1019_101942

theorem students_neither_football_nor_cricket 
  (total : ℕ) 
  (football : ℕ) 
  (cricket : ℕ) 
  (both : ℕ) 
  (h1 : total = 450) 
  (h2 : football = 325) 
  (h3 : cricket = 175) 
  (h4 : both = 100) : 
  total - (football + cricket - both) = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_neither_football_nor_cricket_l1019_101942


namespace NUMINAMATH_CALUDE_crayons_lost_l1019_101900

/-- Given that Paul gave away 52 crayons and lost or gave away a total of 587 crayons,
    prove that the number of crayons he lost is 535. -/
theorem crayons_lost (crayons_given_away : ℕ) (total_lost_or_given_away : ℕ)
    (h1 : crayons_given_away = 52)
    (h2 : total_lost_or_given_away = 587) :
    total_lost_or_given_away - crayons_given_away = 535 := by
  sorry

end NUMINAMATH_CALUDE_crayons_lost_l1019_101900


namespace NUMINAMATH_CALUDE_stating_retirement_benefit_formula_l1019_101975

/-- Represents the retirement benefit calculation for a teacher. -/
structure TeacherBenefit where
  /-- The number of years the teacher has taught. -/
  y : ℝ
  /-- The proportionality constant for the benefit calculation. -/
  k : ℝ
  /-- The additional years in the first scenario. -/
  c : ℝ
  /-- The additional years in the second scenario. -/
  d : ℝ
  /-- The benefit increase in the first scenario. -/
  r : ℝ
  /-- The benefit increase in the second scenario. -/
  s : ℝ
  /-- Ensures that c and d are different. -/
  h_c_neq_d : c ≠ d
  /-- The benefit is proportional to the square root of years taught. -/
  h_benefit : k * Real.sqrt y > 0
  /-- The equation for the first scenario. -/
  h_eq1 : k * Real.sqrt (y + c) = k * Real.sqrt y + r
  /-- The equation for the second scenario. -/
  h_eq2 : k * Real.sqrt (y + d) = k * Real.sqrt y + s

/-- 
Theorem stating that the original annual retirement benefit 
is equal to (s² - r²) / (2(s - r)) given the conditions.
-/
theorem retirement_benefit_formula (tb : TeacherBenefit) : 
  tb.k * Real.sqrt tb.y = (tb.s^2 - tb.r^2) / (2 * (tb.s - tb.r)) := by
  sorry


end NUMINAMATH_CALUDE_stating_retirement_benefit_formula_l1019_101975


namespace NUMINAMATH_CALUDE_minimum_transport_cost_l1019_101936

theorem minimum_transport_cost
  (total_trees : ℕ)
  (chinese_scholar_trees : ℕ)
  (white_pines : ℕ)
  (type_a_capacity_chinese : ℕ)
  (type_a_capacity_pine : ℕ)
  (type_b_capacity : ℕ)
  (type_a_cost : ℕ)
  (type_b_cost : ℕ)
  (total_trucks : ℕ)
  (h1 : total_trees = 320)
  (h2 : chinese_scholar_trees = white_pines + 80)
  (h3 : chinese_scholar_trees + white_pines = total_trees)
  (h4 : type_a_capacity_chinese = 40)
  (h5 : type_a_capacity_pine = 10)
  (h6 : type_b_capacity = 20)
  (h7 : type_a_cost = 400)
  (h8 : type_b_cost = 360)
  (h9 : total_trucks = 8) :
  ∃ (type_a_trucks : ℕ) (type_b_trucks : ℕ),
    type_a_trucks + type_b_trucks = total_trucks ∧
    type_a_trucks * type_a_capacity_chinese + type_b_trucks * type_b_capacity ≥ chinese_scholar_trees ∧
    type_a_trucks * type_a_capacity_pine + type_b_trucks * type_b_capacity ≥ white_pines ∧
    type_a_trucks * type_a_cost + type_b_trucks * type_b_cost = 2960 ∧
    ∀ (other_a : ℕ) (other_b : ℕ),
      other_a + other_b = total_trucks →
      other_a * type_a_capacity_chinese + other_b * type_b_capacity ≥ chinese_scholar_trees →
      other_a * type_a_capacity_pine + other_b * type_b_capacity ≥ white_pines →
      other_a * type_a_cost + other_b * type_b_cost ≥ 2960 :=
by sorry

end NUMINAMATH_CALUDE_minimum_transport_cost_l1019_101936


namespace NUMINAMATH_CALUDE_max_value_of_f_l1019_101968

def f (x : ℝ) : ℝ := -(x + 1)^2 + 5

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1019_101968


namespace NUMINAMATH_CALUDE_diamond_expression_result_l1019_101915

/-- The diamond operation defined as a ⋄ b = a - 1/b -/
def diamond (a b : ℚ) : ℚ := a - 1 / b

/-- Theorem stating the result of the given expression -/
theorem diamond_expression_result :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 := by
  sorry

end NUMINAMATH_CALUDE_diamond_expression_result_l1019_101915


namespace NUMINAMATH_CALUDE_marias_stamps_l1019_101922

theorem marias_stamps (S : ℕ) : 
  S > 1 ∧ 
  S % 9 = 1 ∧ 
  S % 10 = 1 ∧ 
  S % 11 = 1 ∧
  (∀ T : ℕ, T > 1 ∧ T % 9 = 1 ∧ T % 10 = 1 ∧ T % 11 = 1 → S ≤ T) → 
  S = 991 := by
sorry

end NUMINAMATH_CALUDE_marias_stamps_l1019_101922


namespace NUMINAMATH_CALUDE_triangle_area_proof_l1019_101973

noncomputable def f (x : ℝ) := Real.cos (2 * x) + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem triangle_area_proof (A B C : ℝ) (hA : f A = 1) (ha : Real.sqrt 3 = A) (hbc : B + C = 3) :
  (1 / 2 : ℝ) * B * C * Real.sin A = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l1019_101973


namespace NUMINAMATH_CALUDE_sin_105_times_sin_15_l1019_101938

theorem sin_105_times_sin_15 : Real.sin (105 * π / 180) * Real.sin (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_105_times_sin_15_l1019_101938


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1019_101910

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1)}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem intersection_complement_equality :
  N ∩ (U \ M) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1019_101910


namespace NUMINAMATH_CALUDE_profit_equation_correct_l1019_101961

/-- Represents the profit calculation for a bicycle sale --/
def profit_equation (x : ℝ) : Prop :=
  0.8 * (1 + 0.45) * x - x = 50

/-- Theorem stating that the profit equation correctly represents the given scenario --/
theorem profit_equation_correct (x : ℝ) : profit_equation x ↔ 
  (∃ (markup discount profit : ℝ),
    markup = 0.45 ∧
    discount = 0.2 ∧
    profit = 50 ∧
    (1 - discount) * (1 + markup) * x - x = profit) :=
by sorry

end NUMINAMATH_CALUDE_profit_equation_correct_l1019_101961


namespace NUMINAMATH_CALUDE_initial_fraction_is_half_l1019_101927

/-- Represents a journey with two parts at different speeds -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  initialSpeed : ℝ
  remainingSpeed : ℝ
  initialFraction : ℝ

/-- The conditions of the journey -/
def journeyConditions (j : Journey) : Prop :=
  j.initialSpeed = 40 ∧
  j.remainingSpeed = 20 ∧
  j.initialFraction * j.totalDistance = j.initialSpeed * (j.totalTime / 3) ∧
  (1 - j.initialFraction) * j.totalDistance = j.remainingSpeed * (2 * j.totalTime / 3) ∧
  j.totalDistance > 0 ∧
  j.totalTime > 0

/-- The theorem stating that under the given conditions, the initial fraction is 1/2 -/
theorem initial_fraction_is_half (j : Journey) :
  journeyConditions j → j.initialFraction = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_initial_fraction_is_half_l1019_101927


namespace NUMINAMATH_CALUDE_stock_price_change_l1019_101957

theorem stock_price_change (total_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : ∀ s : Fin total_stocks, ∃ (price_yesterday price_today : ℝ), price_yesterday ≠ price_today)
  (h3 : ∃ (higher lower : ℕ), 
    higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5)) :
  ∃ (higher : ℕ), higher = 1080 ∧ 
    ∃ (lower : ℕ), higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5) := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l1019_101957


namespace NUMINAMATH_CALUDE_elephant_giraffe_jade_ratio_l1019_101996

/-- The amount of jade (in grams) needed for a giraffe statue -/
def giraffe_jade : ℝ := 120

/-- The selling price of a giraffe statue -/
def giraffe_price : ℝ := 150

/-- The selling price of an elephant statue -/
def elephant_price : ℝ := 350

/-- The total amount of jade Nancy has -/
def total_jade : ℝ := 1920

/-- The additional revenue from making all elephants instead of giraffes -/
def additional_revenue : ℝ := 400

/-- The ratio of jade used for an elephant statue to a giraffe statue -/
def jade_ratio : ℝ := 2

theorem elephant_giraffe_jade_ratio :
  let elephant_jade := giraffe_jade * jade_ratio
  let giraffe_count := total_jade / giraffe_jade
  let elephant_count := total_jade / elephant_jade
  giraffe_count * giraffe_price + additional_revenue = elephant_count * elephant_price :=
sorry

end NUMINAMATH_CALUDE_elephant_giraffe_jade_ratio_l1019_101996


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_plus_one_l1019_101993

theorem smallest_three_digit_multiple_plus_one : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  (∃ k : ℕ, n = 3 * k + 1) ∧
  (∃ k : ℕ, n = 4 * k + 1) ∧
  (∃ k : ℕ, n = 5 * k + 1) ∧
  (∃ k : ℕ, n = 7 * k + 1) ∧
  (∃ k : ℕ, n = 8 * k + 1) ∧
  (∀ m : ℕ, m < n →
    ¬(100 ≤ m ∧ m < 1000 ∧
      (∃ k : ℕ, m = 3 * k + 1) ∧
      (∃ k : ℕ, m = 4 * k + 1) ∧
      (∃ k : ℕ, m = 5 * k + 1) ∧
      (∃ k : ℕ, m = 7 * k + 1) ∧
      (∃ k : ℕ, m = 8 * k + 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_plus_one_l1019_101993


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1019_101939

def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (Set.univ \ B) = Set.Icc 1 2 ∩ Set.Iio 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1019_101939


namespace NUMINAMATH_CALUDE_fraction_equality_l1019_101918

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1019_101918


namespace NUMINAMATH_CALUDE_rocket_heights_sum_l1019_101946

/-- The height of the first rocket in feet -/
def first_rocket_height : ℝ := 500

/-- The height of the second rocket in feet -/
def second_rocket_height : ℝ := 2 * first_rocket_height

/-- The combined height of both rockets in feet -/
def combined_height : ℝ := first_rocket_height + second_rocket_height

theorem rocket_heights_sum :
  combined_height = 1500 := by
  sorry

end NUMINAMATH_CALUDE_rocket_heights_sum_l1019_101946


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l1019_101913

/-- The ratio of average speed to still water speed for a boat trip --/
theorem boat_speed_ratio 
  (still_water_speed : ℝ) 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (h1 : still_water_speed = 20)
  (h2 : current_speed = 4)
  (h3 : downstream_distance = 5)
  (h4 : upstream_distance = 3) :
  let downstream_speed := still_water_speed + current_speed
  let upstream_speed := still_water_speed - current_speed
  let total_distance := downstream_distance + upstream_distance
  let total_time := downstream_distance / downstream_speed + upstream_distance / upstream_speed
  let average_speed := total_distance / total_time
  average_speed / still_water_speed = 96 / 95 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l1019_101913


namespace NUMINAMATH_CALUDE_initial_workers_count_l1019_101951

/-- Represents the work done in digging a hole -/
def work (workers : ℕ) (hours : ℕ) (depth : ℕ) : ℕ := workers * hours * depth

theorem initial_workers_count :
  ∀ (W : ℕ),
  (∃ (k : ℕ), k > 0 ∧
    work W 8 30 = k * 30 ∧
    work (W + 35) 6 40 = k * 40) →
  W = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_workers_count_l1019_101951


namespace NUMINAMATH_CALUDE_max_primitive_dinosaur_cells_l1019_101932

/-- An animal is a connected figure consisting of equal-sized square cells. -/
structure Animal where
  cells : ℕ
  is_connected : Bool

/-- A dinosaur is an animal with at least 2007 cells. -/
def Dinosaur (a : Animal) : Prop :=
  a.cells ≥ 2007

/-- A primitive dinosaur cannot be partitioned into two or more dinosaurs. -/
def PrimitiveDinosaur (a : Animal) : Prop :=
  Dinosaur a ∧ ¬∃ (b c : Animal), Dinosaur b ∧ Dinosaur c ∧ b.cells + c.cells ≤ a.cells

/-- The maximum number of cells in a primitive dinosaur is 8025. -/
theorem max_primitive_dinosaur_cells :
  ∃ (a : Animal), PrimitiveDinosaur a ∧ a.cells = 8025 ∧
  ∀ (b : Animal), PrimitiveDinosaur b → b.cells ≤ 8025 := by
  sorry


end NUMINAMATH_CALUDE_max_primitive_dinosaur_cells_l1019_101932
