import Mathlib

namespace NUMINAMATH_CALUDE_halfway_fraction_l1016_101640

theorem halfway_fraction (a b c d : ℤ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 6) :
  (a : ℚ) / b + ((c : ℚ) / d - (a : ℚ) / b) / 2 = 19 / 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1016_101640


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l1016_101604

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 12 = 0) 
  (h2 : n ≥ 24) : ∃ (blue red green yellow : ℕ),
  blue = n / 3 ∧ 
  red = n / 4 ∧ 
  green = 6 ∧ 
  yellow = n - (blue + red + green) ∧ 
  blue + red + green + yellow = n ∧
  yellow ≥ 4 ∧
  (∀ m : ℕ, m < n → ¬(∃ b r g y : ℕ, 
    b = m / 3 ∧ 
    r = m / 4 ∧ 
    g = 6 ∧ 
    y = m - (b + r + g) ∧ 
    b + r + g + y = m ∧ 
    y ≥ 4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l1016_101604


namespace NUMINAMATH_CALUDE_bob_spending_theorem_l1016_101677

def spending_problem (initial_amount : ℚ) : ℚ :=
  let after_monday := initial_amount / 2
  let after_tuesday := after_monday - (after_monday / 5)
  let after_wednesday := after_tuesday - (after_tuesday * 3 / 8)
  after_wednesday

theorem bob_spending_theorem :
  spending_problem 80 = 20 := by sorry

end NUMINAMATH_CALUDE_bob_spending_theorem_l1016_101677


namespace NUMINAMATH_CALUDE_worker_payment_l1016_101658

/-- Given a sum of money that can pay worker A for 18 days, worker B for 12 days, 
    and worker C for 24 days, prove that it can pay all three workers together for 5 days. -/
theorem worker_payment (S : ℚ) (A B C : ℚ) (hA : S = 18 * A) (hB : S = 12 * B) (hC : S = 24 * C) :
  ∃ D : ℕ, D = 5 ∧ S = D * (A + B + C) :=
sorry

end NUMINAMATH_CALUDE_worker_payment_l1016_101658


namespace NUMINAMATH_CALUDE_black_cards_count_l1016_101621

theorem black_cards_count (total_cards : Nat) (red_cards : Nat) (clubs : Nat)
  (h_total : total_cards = 13)
  (h_red : red_cards = 6)
  (h_clubs : clubs = 6)
  (h_suits : ∃ (spades diamonds hearts : Nat), 
    spades + diamonds + hearts + clubs = total_cards ∧
    diamonds = 2 * spades ∧
    hearts = 2 * diamonds) :
  clubs + (total_cards - red_cards - clubs) = 7 := by
  sorry

end NUMINAMATH_CALUDE_black_cards_count_l1016_101621


namespace NUMINAMATH_CALUDE_workers_wage_increase_l1016_101696

/-- If a worker's daily wage is increased by 40% resulting in a new wage of $35 per day, 
    then the original daily wage was $25. -/
theorem workers_wage_increase (original_wage : ℝ) 
  (h1 : original_wage * 1.4 = 35) : original_wage = 25 := by
  sorry

end NUMINAMATH_CALUDE_workers_wage_increase_l1016_101696


namespace NUMINAMATH_CALUDE_number_ratio_l1016_101695

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 69) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l1016_101695


namespace NUMINAMATH_CALUDE_expression_evaluation_l1016_101673

-- Define the expression as a function
def f (x : ℚ) : ℚ := (4 + x * (4 + x) - 4^2) / (x - 4 + x^2 + 2*x)

-- State the theorem
theorem expression_evaluation :
  f (-3) = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1016_101673


namespace NUMINAMATH_CALUDE_total_digits_in_books_l1016_101606

/-- Calculate the number of digits used to number pages in a book -/
def digitsInBook (pages : ℕ) : ℕ :=
  let singleDigitPages := min pages 9
  let doubleDigitPages := min (pages - 9) 90
  let tripleDigitPages := min (pages - 99) 900
  let quadrupleDigitPages := max (pages - 999) 0
  singleDigitPages * 1 +
  doubleDigitPages * 2 +
  tripleDigitPages * 3 +
  quadrupleDigitPages * 4

/-- The total number of digits used to number pages in the collection of books -/
def totalDigits : ℕ :=
  digitsInBook 450 + digitsInBook 675 + digitsInBook 1125 + digitsInBook 2430

theorem total_digits_in_books :
  totalDigits = 15039 := by sorry

end NUMINAMATH_CALUDE_total_digits_in_books_l1016_101606


namespace NUMINAMATH_CALUDE_aquarium_count_l1016_101663

/-- Given a total number of saltwater animals and the number of animals per aquarium,
    calculate the number of aquariums. -/
def calculate_aquariums (total_animals : ℕ) (animals_per_aquarium : ℕ) : ℕ :=
  total_animals / animals_per_aquarium

theorem aquarium_count :
  let total_animals : ℕ := 52
  let animals_per_aquarium : ℕ := 2
  calculate_aquariums total_animals animals_per_aquarium = 26 := by
  sorry

#eval calculate_aquariums 52 2

end NUMINAMATH_CALUDE_aquarium_count_l1016_101663


namespace NUMINAMATH_CALUDE_line_AB_equation_l1016_101679

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the point P
def point_P : ℝ × ℝ := (3, 1)

-- Define the existence of a circle passing through P and tangent to C at A and B
def tangent_circle_exists : Prop :=
  ∃ (center_x center_y radius : ℝ) (A B : ℝ × ℝ),
    -- The circle passes through P
    (point_P.1 - center_x)^2 + (point_P.2 - center_y)^2 = radius^2 ∧
    -- A and B are on circle C
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    -- The new circle is tangent to C at A and B
    (A.1 - center_x)^2 + (A.2 - center_y)^2 = radius^2 ∧
    (B.1 - center_x)^2 + (B.2 - center_y)^2 = radius^2

-- Theorem: The equation of line AB is 2x + y - 3 = 0
theorem line_AB_equation :
  tangent_circle_exists →
  ∃ (A B : ℝ × ℝ), circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    ∀ (x y : ℝ), (2 * x + y - 3 = 0) ↔ (∃ t : ℝ, x = A.1 + t * (B.1 - A.1) ∧ y = A.2 + t * (B.2 - A.2)) :=
by sorry

end NUMINAMATH_CALUDE_line_AB_equation_l1016_101679


namespace NUMINAMATH_CALUDE_sand_pit_fill_theorem_l1016_101602

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents a sand pit with its dimensions and current fill level -/
structure SandPit where
  dimensions : PrismDimensions
  fillLevel : ℝ  -- Represents the fraction of the pit that is filled (0 to 1)

/-- Calculates the additional sand volume needed to fill the pit completely -/
def additionalSandNeeded (pit : SandPit) : ℝ :=
  (1 - pit.fillLevel) * prismVolume pit.dimensions

theorem sand_pit_fill_theorem (pit : SandPit) 
    (h1 : pit.dimensions.length = 10)
    (h2 : pit.dimensions.width = 2)
    (h3 : pit.dimensions.height = 0.5)
    (h4 : pit.fillLevel = 0.5) :
    additionalSandNeeded pit = 5 := by
  sorry

#eval additionalSandNeeded {
  dimensions := { length := 10, width := 2, height := 0.5 },
  fillLevel := 0.5
}

end NUMINAMATH_CALUDE_sand_pit_fill_theorem_l1016_101602


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1016_101650

-- Define the present ages
def sons_present_age : ℕ := 24
def mans_present_age : ℕ := sons_present_age + 26

-- Define the ages in two years
def sons_future_age : ℕ := sons_present_age + 2
def mans_future_age : ℕ := mans_present_age + 2

-- Define the ratio
def age_ratio : ℚ := mans_future_age / sons_future_age

theorem age_ratio_is_two_to_one : age_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1016_101650


namespace NUMINAMATH_CALUDE_third_grade_volunteers_l1016_101685

/-- Calculates the number of volunteers to be recruited from a specific grade --/
def volunteers_from_grade (total_students : ℕ) (grade_students : ℕ) (total_volunteers : ℕ) : ℕ :=
  (grade_students * total_volunteers) / total_students

theorem third_grade_volunteers :
  let total_students : ℕ := 2040
  let first_grade : ℕ := 680
  let second_grade : ℕ := 850
  let third_grade : ℕ := 510
  let total_volunteers : ℕ := 12
  volunteers_from_grade total_students third_grade total_volunteers = 3 := by
  sorry


end NUMINAMATH_CALUDE_third_grade_volunteers_l1016_101685


namespace NUMINAMATH_CALUDE_ball_count_difference_l1016_101687

theorem ball_count_difference (total : ℕ) (white : ℕ) : 
  total = 100 →
  white = 16 →
  ∃ (blue red : ℕ),
    blue > white ∧
    red = 2 * blue ∧
    red + blue + white = total ∧
    blue - white = 12 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_difference_l1016_101687


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l1016_101633

-- Define the curve
def curve (a x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def curve_derivative (a x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a_value (a : ℝ) :
  curve a (-1) = a + 2 →
  curve_derivative a (-1) = 8 →
  a = -6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l1016_101633


namespace NUMINAMATH_CALUDE_relationship_abc_l1016_101680

theorem relationship_abc (a b c : ℝ) 
  (ha : a = 1 / 2022)
  (hb : b = Real.exp (-2021 / 2022))
  (hc : c = Real.log (2023 / 2022)) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1016_101680


namespace NUMINAMATH_CALUDE_rug_area_is_24_l1016_101632

/-- Calculates the area of a rug on a rectangular floor with uncovered strips along the edges -/
def rugArea (floorLength floorWidth stripWidth : ℝ) : ℝ :=
  (floorLength - 2 * stripWidth) * (floorWidth - 2 * stripWidth)

/-- Theorem stating that the area of the rug is 24 square meters given the specific dimensions -/
theorem rug_area_is_24 :
  rugArea 10 8 2 = 24 := by
  sorry

#eval rugArea 10 8 2

end NUMINAMATH_CALUDE_rug_area_is_24_l1016_101632


namespace NUMINAMATH_CALUDE_modular_congruence_unique_solution_l1016_101692

theorem modular_congruence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ 24938 ≡ n [ZMOD 25] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_unique_solution_l1016_101692


namespace NUMINAMATH_CALUDE_composition_equation_solution_l1016_101676

def α : ℝ → ℝ := λ x ↦ 4 * x + 9
def β : ℝ → ℝ := λ x ↦ 9 * x + 6

theorem composition_equation_solution :
  ∃! x : ℝ, α (β x) = 8 ∧ x = -25/36 := by sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l1016_101676


namespace NUMINAMATH_CALUDE_two_numbers_sum_l1016_101698

theorem two_numbers_sum (x y : ℝ) 
  (sum_eq : x + y = 5)
  (diff_eq : x - y = 10)
  (square_diff_eq : x^2 - y^2 = 50) : 
  x + y = 5 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_l1016_101698


namespace NUMINAMATH_CALUDE_equality_equivalence_l1016_101620

theorem equality_equivalence (a b c : ℝ) : 
  (a * (b - c)) / (b + c) + (b * (c - a)) / (c + a) + (c * (a - b)) / (a + b) = 0 ↔ 
  (a^2 * (b - c)) / (b + c) + (b^2 * (c - a)) / (c + a) + (c^2 * (a - b)) / (a + b) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equality_equivalence_l1016_101620


namespace NUMINAMATH_CALUDE_reach_one_l1016_101674

/-- Represents the two possible operations in the game -/
inductive Operation
  | EraseUnitsDigit
  | MultiplyByTwo

/-- Defines a step in the game as applying an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.EraseUnitsDigit => n / 10
  | Operation.MultiplyByTwo => n * 2

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a number -/
def applySequence (n : ℕ) (seq : OperationSequence) : ℕ :=
  seq.foldl applyOperation n

/-- The main theorem stating that for any positive natural number,
    there exists a sequence of operations that transforms it to 1 -/
theorem reach_one (n : ℕ) (h : n > 0) :
  ∃ (seq : OperationSequence), applySequence n seq = 1 := by
  sorry

end NUMINAMATH_CALUDE_reach_one_l1016_101674


namespace NUMINAMATH_CALUDE_evaluate_expression_l1016_101607

theorem evaluate_expression : 1234562 - (12 * 3 * (2 + 7)) = 1234238 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1016_101607


namespace NUMINAMATH_CALUDE_a_seven_value_l1016_101636

/-- An arithmetic sequence where the reciprocals of terms form an arithmetic sequence -/
def reciprocal_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, 1 / a (n + 1) - 1 / a n = d

theorem a_seven_value (a : ℕ → ℝ) 
  (h_seq : reciprocal_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 4) :
  a 7 = -2 :=
sorry

end NUMINAMATH_CALUDE_a_seven_value_l1016_101636


namespace NUMINAMATH_CALUDE_sandy_worked_five_days_l1016_101613

/-- The number of days Sandy worked -/
def days_worked (total_hours : ℕ) (hours_per_day : ℕ) : ℚ :=
  total_hours / hours_per_day

/-- Proof that Sandy worked 5 days -/
theorem sandy_worked_five_days (total_hours : ℕ) (hours_per_day : ℕ) 
  (h1 : total_hours = 45)
  (h2 : hours_per_day = 9) :
  days_worked total_hours hours_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_sandy_worked_five_days_l1016_101613


namespace NUMINAMATH_CALUDE_min_distance_C₁_C₂_l1016_101654

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop :=
  (x - Real.sqrt 3 / 2)^2 + (y - 1/2)^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y - 8 = 0

-- State the theorem
theorem min_distance_C₁_C₂ :
  ∃ d : ℝ, d = 2 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ → C₂ x₂ y₂ →
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_C₁_C₂_l1016_101654


namespace NUMINAMATH_CALUDE_exp_inequality_equivalence_l1016_101601

theorem exp_inequality_equivalence (x : ℝ) : 1 < Real.exp x ∧ Real.exp x < 2 ↔ 0 < x ∧ x < Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_exp_inequality_equivalence_l1016_101601


namespace NUMINAMATH_CALUDE_lcm_of_9_16_21_l1016_101657

theorem lcm_of_9_16_21 : Nat.lcm 9 (Nat.lcm 16 21) = 1008 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_16_21_l1016_101657


namespace NUMINAMATH_CALUDE_essay_pages_theorem_l1016_101626

/-- Calculates the number of pages needed for a given number of words -/
def pages_needed (words : ℕ) : ℕ := (words + 259) / 260

/-- Represents the essay writing scenario -/
def essay_pages : Prop :=
  let johnny_words : ℕ := 150
  let madeline_words : ℕ := 2 * johnny_words
  let timothy_words : ℕ := madeline_words + 30
  let total_pages : ℕ := pages_needed johnny_words + pages_needed madeline_words + pages_needed timothy_words
  total_pages = 5

theorem essay_pages_theorem : essay_pages := by
  sorry

end NUMINAMATH_CALUDE_essay_pages_theorem_l1016_101626


namespace NUMINAMATH_CALUDE_cone_radius_theorem_l1016_101628

/-- For a cone with radius r, slant height 2r, and lateral surface area equal to half its volume, prove that r = 4√3 -/
theorem cone_radius_theorem (r : ℝ) (h : ℝ) : 
  r > 0 → 
  h > 0 → 
  (2 * r)^2 = r^2 + h^2 →  -- Pythagorean theorem for the slant height
  π * r * (2 * r) = (1/2) * ((1/3) * π * r^2 * h) →  -- Lateral surface area = 1/2 * Volume
  r = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_radius_theorem_l1016_101628


namespace NUMINAMATH_CALUDE_max_length_special_arithmetic_progression_l1016_101639

/-- An arithmetic progression of natural numbers with common difference 2 -/
def ArithmeticProgression (a₁ : ℕ) (n : ℕ) : Fin n → ℕ :=
  λ i => a₁ + 2 * i.val

/-- Predicate to check if a number is prime -/
def IsPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The maximum length of the special arithmetic progression -/
def MaxLength : ℕ := 3

theorem max_length_special_arithmetic_progression :
  ∀ a₁ n : ℕ,
    (∀ k : Fin n, IsPrime ((ArithmeticProgression a₁ n k)^2 + 1)) →
    n ≤ MaxLength :=
by sorry

end NUMINAMATH_CALUDE_max_length_special_arithmetic_progression_l1016_101639


namespace NUMINAMATH_CALUDE_coefficient_a5_equals_6_l1016_101653

theorem coefficient_a5_equals_6 
  (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, x^6 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6) →
  a₅ = 6 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a5_equals_6_l1016_101653


namespace NUMINAMATH_CALUDE_birds_on_fence_two_plus_four_birds_l1016_101655

/-- The number of birds on a fence after more birds join -/
theorem birds_on_fence (initial : Nat) (joined : Nat) : 
  initial + joined = initial + joined :=
by sorry

/-- The specific case of 2 initial birds and 4 joined birds -/
theorem two_plus_four_birds : 2 + 4 = 6 :=
by sorry

end NUMINAMATH_CALUDE_birds_on_fence_two_plus_four_birds_l1016_101655


namespace NUMINAMATH_CALUDE_inverse_difference_theorem_l1016_101667

theorem inverse_difference_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ - (b / 3)⁻¹) = -(a⁻¹ * b⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_inverse_difference_theorem_l1016_101667


namespace NUMINAMATH_CALUDE_value_after_two_years_approximation_l1016_101682

/-- Calculates the value after n years given an initial value and annual increase rate -/
def value_after_n_years (initial_value : ℝ) (increase_rate : ℝ) (n : ℕ) : ℝ :=
  initial_value * (1 + increase_rate) ^ n

/-- The problem statement -/
theorem value_after_two_years_approximation :
  let initial_value : ℝ := 64000
  let increase_rate : ℝ := 1 / 9
  let years : ℕ := 2
  let final_value := value_after_n_years initial_value increase_rate years
  abs (final_value - 79012.36) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_value_after_two_years_approximation_l1016_101682


namespace NUMINAMATH_CALUDE_chris_mixture_problem_l1016_101683

/-- Given the conditions of Chris's mixture of raisins and nuts, prove that the number of pounds of nuts is 4. -/
theorem chris_mixture_problem (raisin_pounds : ℝ) (nut_pounds : ℝ) (raisin_cost : ℝ) (nut_cost : ℝ) :
  raisin_pounds = 3 →
  nut_cost = 2 * raisin_cost →
  (raisin_pounds * raisin_cost) = (3 / 11) * (raisin_pounds * raisin_cost + nut_pounds * nut_cost) →
  nut_pounds = 4 := by
sorry

end NUMINAMATH_CALUDE_chris_mixture_problem_l1016_101683


namespace NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l1016_101664

-- Define the circles
def circle1 : Real × Real × Real := (0, 0, 3)  -- (center_x, center_y, radius)
def circle2 : Real × Real × Real := (12, 0, 5)  -- (center_x, center_y, radius)

-- Define the theorem
theorem tangent_intersection_x_coordinate :
  ∃ (x : Real),
    x > 0 ∧  -- Intersection to the right of origin
    (let (x1, y1, r1) := circle1
     let (x2, y2, r2) := circle2
     (x - x1) / (x - x2) = r1 / r2) ∧
    x = 18 := by
  sorry


end NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l1016_101664


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1016_101612

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + (3^a - 3) * y = 0 → 
    ∃ k : ℝ, y = (-3 / (3^a - 3)) * x + k) →
  (∀ x y : ℝ, 2 * x - y - 3 = 0 → 
    ∃ k : ℝ, y = 2 * x + k) →
  (∀ m₁ m₂ : ℝ, m₁ * m₂ = -1 → 
    m₁ = -3 / (3^a - 3) ∧ m₂ = 2) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1016_101612


namespace NUMINAMATH_CALUDE_cube_sum_equals_thirteen_l1016_101637

theorem cube_sum_equals_thirteen (a b : ℝ) 
  (h1 : a^3 - 3*a*b^2 = 39)
  (h2 : b^3 - 3*a^2*b = 26) :
  a^2 + b^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_thirteen_l1016_101637


namespace NUMINAMATH_CALUDE_s_bounds_l1016_101672

theorem s_bounds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  let s := Real.sqrt (a * b / ((b + c) * (c + a))) +
           Real.sqrt (b * c / ((c + a) * (a + b))) +
           Real.sqrt (c * a / ((a + b) * (b + c)))
  1 ≤ s ∧ s ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_s_bounds_l1016_101672


namespace NUMINAMATH_CALUDE_original_selling_price_l1016_101681

/-- Given an article with a cost price of $15000, prove that the original selling price
    that would result in an 8% profit if discounted by 10% is $18000. -/
theorem original_selling_price (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  cost_price = 15000 →
  discount_rate = 0.1 →
  profit_rate = 0.08 →
  ∃ (selling_price : ℝ),
    selling_price * (1 - discount_rate) = cost_price * (1 + profit_rate) ∧
    selling_price = 18000 := by
  sorry

end NUMINAMATH_CALUDE_original_selling_price_l1016_101681


namespace NUMINAMATH_CALUDE_vertical_line_no_slope_l1016_101631

/-- A line parallel to the y-axis has no defined slope -/
theorem vertical_line_no_slope (a : ℝ) : 
  ¬ ∃ (m : ℝ), ∀ (x y : ℝ), x = a → (∀ ε > 0, ∃ δ > 0, ∀ x' y', |x' - x| < δ → |y' - y| < ε * |x' - x|) :=
by
  sorry

end NUMINAMATH_CALUDE_vertical_line_no_slope_l1016_101631


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1016_101694

theorem negation_of_existence (f : ℝ → Prop) :
  (¬ ∃ x : ℝ, f x) ↔ ∀ x : ℝ, ¬ f x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 3 > 0) ↔ (∀ x : ℝ, x^2 - 2*x + 3 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1016_101694


namespace NUMINAMATH_CALUDE_naza_market_averages_l1016_101660

/-- Represents an electronic shop with TV sets and models -/
structure Shop where
  name : Char
  tv_sets : ℕ
  tv_models : ℕ

/-- The list of shops in the Naza market -/
def naza_shops : List Shop := [
  ⟨'A', 20, 3⟩,
  ⟨'B', 30, 4⟩,
  ⟨'C', 60, 5⟩,
  ⟨'D', 80, 6⟩,
  ⟨'E', 50, 2⟩,
  ⟨'F', 40, 4⟩,
  ⟨'G', 70, 3⟩
]

/-- The total number of shops -/
def total_shops : ℕ := naza_shops.length

/-- Calculates the average of a list of natural numbers -/
def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

/-- Theorem stating the average number of TV sets and models in Naza market shops -/
theorem naza_market_averages :
  average (naza_shops.map Shop.tv_sets) = 50 ∧
  average (naza_shops.map Shop.tv_models) = 27 / 7 := by
  sorry

end NUMINAMATH_CALUDE_naza_market_averages_l1016_101660


namespace NUMINAMATH_CALUDE_three_card_selection_count_l1016_101684

/-- The number of ways to select 3 different cards in order from a set of 13 cards -/
def select_three_cards : ℕ := 13 * 12 * 11

/-- Theorem stating that selecting 3 different cards in order from 13 cards results in 1716 possibilities -/
theorem three_card_selection_count : select_three_cards = 1716 := by
  sorry

end NUMINAMATH_CALUDE_three_card_selection_count_l1016_101684


namespace NUMINAMATH_CALUDE_garden_width_is_correct_garden_area_is_correct_l1016_101617

/-- Represents a rectangular flower garden -/
structure FlowerGarden where
  length : ℝ
  width : ℝ
  area : ℝ

/-- The flower garden has the given dimensions -/
def garden : FlowerGarden where
  length := 4
  width := 35.8
  area := 143.2

/-- Theorem: The width of the flower garden is 35.8 meters -/
theorem garden_width_is_correct : garden.width = 35.8 := by
  sorry

/-- Theorem: The area of the garden is equal to length times width -/
theorem garden_area_is_correct : garden.area = garden.length * garden.width := by
  sorry

end NUMINAMATH_CALUDE_garden_width_is_correct_garden_area_is_correct_l1016_101617


namespace NUMINAMATH_CALUDE_tom_final_book_count_l1016_101614

/-- The number of books Tom has after selling some and buying new ones -/
def final_book_count (initial_books sold_books new_books : ℕ) : ℕ :=
  initial_books - sold_books + new_books

/-- Theorem stating that Tom ends up with 39 books -/
theorem tom_final_book_count :
  final_book_count 5 4 38 = 39 := by
  sorry

end NUMINAMATH_CALUDE_tom_final_book_count_l1016_101614


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1016_101619

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x * y = -2) 
  (h2 : x + y = 4) : 
  x^2 * y + x * y^2 = -8 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1016_101619


namespace NUMINAMATH_CALUDE_strategy_game_cost_l1016_101642

/-- The cost of Tom's video game purchases -/
def total_cost : ℚ := 35.52

/-- The cost of the football game -/
def football_cost : ℚ := 14.02

/-- The cost of the Batman game -/
def batman_cost : ℚ := 12.04

/-- The cost of the strategy game -/
def strategy_cost : ℚ := total_cost - football_cost - batman_cost

theorem strategy_game_cost :
  strategy_cost = 9.46 := by sorry

end NUMINAMATH_CALUDE_strategy_game_cost_l1016_101642


namespace NUMINAMATH_CALUDE_ellas_raise_percentage_l1016_101648

/-- Calculates the percentage raise given the conditions of Ella's babysitting earnings and expenses. -/
theorem ellas_raise_percentage 
  (video_game_percentage : Real) 
  (last_year_video_game_expense : Real) 
  (new_salary : Real) 
  (h1 : video_game_percentage = 0.40)
  (h2 : last_year_video_game_expense = 100)
  (h3 : new_salary = 275) : 
  (new_salary - (last_year_video_game_expense / video_game_percentage)) / (last_year_video_game_expense / video_game_percentage) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ellas_raise_percentage_l1016_101648


namespace NUMINAMATH_CALUDE_complex_roots_isosceles_triangle_l1016_101641

theorem complex_roots_isosceles_triangle (a b z₁ z₂ : ℂ) :
  z₁^2 + a*z₁ + b = 0 →
  z₂^2 + a*z₂ + b = 0 →
  z₂ = Complex.exp (Real.pi * Complex.I / 4) * z₁ →
  a^2 / b = 4 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_isosceles_triangle_l1016_101641


namespace NUMINAMATH_CALUDE_original_number_l1016_101669

theorem original_number (N : ℕ) : (∀ k : ℕ, N - 7 ≠ 12 * k) ∧ (∃ k : ℕ, N - 7 = 12 * k) → N = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1016_101669


namespace NUMINAMATH_CALUDE_line_rotation_theorem_l1016_101603

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a line counterclockwise around a point --/
def rotateLine (l : Line) (θ : ℝ) (p : ℝ × ℝ) : Line :=
  sorry

/-- Finds the intersection of a line with the x-axis --/
def xAxisIntersection (l : Line) : ℝ × ℝ :=
  sorry

theorem line_rotation_theorem (l : Line) :
  l.a = 2 ∧ l.b = -1 ∧ l.c = -4 →
  let p := xAxisIntersection l
  let l' := rotateLine l (π/4) p
  l'.a = 3 ∧ l'.b = 1 ∧ l'.c = -6 :=
sorry

end NUMINAMATH_CALUDE_line_rotation_theorem_l1016_101603


namespace NUMINAMATH_CALUDE_total_money_l1016_101645

theorem total_money (r p q : ℕ) (h1 : r = 1600) (h2 : r = (2 * (p + q)) / 3) : 
  p + q + r = 4000 := by
sorry

end NUMINAMATH_CALUDE_total_money_l1016_101645


namespace NUMINAMATH_CALUDE_keith_missed_four_games_l1016_101646

/-- The number of football games Keith missed, given the total number of games and the number of games he attended. -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Theorem stating that Keith missed 4 football games. -/
theorem keith_missed_four_games :
  let total_games : ℕ := 8
  let attended_games : ℕ := 4
  games_missed total_games attended_games = 4 := by
sorry

end NUMINAMATH_CALUDE_keith_missed_four_games_l1016_101646


namespace NUMINAMATH_CALUDE_unique_right_triangle_l1016_101622

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- Theorem stating that among the given sets, only {1, 1, √2} forms a right triangle --/
theorem unique_right_triangle :
  ¬ is_right_triangle 4 5 6 ∧
  is_right_triangle 1 1 (Real.sqrt 2) ∧
  ¬ is_right_triangle 6 8 11 ∧
  ¬ is_right_triangle 5 12 23 :=
by sorry

#check unique_right_triangle

end NUMINAMATH_CALUDE_unique_right_triangle_l1016_101622


namespace NUMINAMATH_CALUDE_product_of_x_values_l1016_101644

theorem product_of_x_values (x : ℝ) : 
  (|12 / x + 3| = 2) → (∃ y : ℝ, (|12 / y + 3| = 2) ∧ x * y = 144 / 5) :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l1016_101644


namespace NUMINAMATH_CALUDE_equal_area_segment_property_l1016_101647

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  base_difference : longer_base = shorter_base + 150
  midpoint_ratio : ℝ
  midpoint_area_ratio : midpoint_ratio = 3 / 4

/-- The length of the segment that divides the trapezoid into two equal-area regions -/
def equal_area_segment (t : Trapezoid) : ℝ :=
  t.shorter_base + 150

/-- Theorem stating the property of the equal area segment -/
theorem equal_area_segment_property (t : Trapezoid) :
  ⌊(equal_area_segment t)^3 / 1000⌋ = 142 := by
  sorry

#check equal_area_segment_property

end NUMINAMATH_CALUDE_equal_area_segment_property_l1016_101647


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1016_101634

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 3^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1016_101634


namespace NUMINAMATH_CALUDE_point_classification_l1016_101670

-- Define the plane region
def in_region (x y : ℝ) : Prop := x + y - 1 ≤ 0

-- Theorem stating that (-1,3) is not in the region, while the other points are
theorem point_classification :
  ¬(in_region (-1) 3) ∧ 
  in_region 0 0 ∧ 
  in_region (-1) 1 ∧ 
  in_region 2 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_classification_l1016_101670


namespace NUMINAMATH_CALUDE_arrangement_counts_l1016_101666

/-- The number of boys in the arrangement -/
def num_boys : ℕ := 3

/-- The number of girls in the arrangement -/
def num_girls : ℕ := 4

/-- The total number of people to be arranged -/
def total_people : ℕ := num_boys + num_girls

/-- Calculates the number of arrangements with Person A and Person B at the ends -/
def arrangements_ends : ℕ := sorry

/-- Calculates the number of arrangements with all boys standing together -/
def arrangements_boys_together : ℕ := sorry

/-- Calculates the number of arrangements with no two boys standing next to each other -/
def arrangements_boys_separated : ℕ := sorry

/-- Calculates the number of arrangements with exactly one person between Person A and Person B -/
def arrangements_one_between : ℕ := sorry

theorem arrangement_counts :
  arrangements_ends = 240 ∧
  arrangements_boys_together = 720 ∧
  arrangements_boys_separated = 1440 ∧
  arrangements_one_between = 1200 := by sorry

end NUMINAMATH_CALUDE_arrangement_counts_l1016_101666


namespace NUMINAMATH_CALUDE_original_to_half_ratio_l1016_101693

theorem original_to_half_ratio (x : ℝ) (h : x / 2 = 9) : x / (x / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_original_to_half_ratio_l1016_101693


namespace NUMINAMATH_CALUDE_range_of_a_l1016_101678

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 2| + 1) → 
  1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1016_101678


namespace NUMINAMATH_CALUDE_inequality_proof_l1016_101624

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1016_101624


namespace NUMINAMATH_CALUDE_inequality_proof_l1016_101699

theorem inequality_proof (x y : ℝ) : 
  ((x * y - y^2) / (x^2 + 4 * x + 5))^3 ≤ ((x^2 - x * y) / (x^2 + 4 * x + 5))^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1016_101699


namespace NUMINAMATH_CALUDE_cube_side_ratio_l1016_101675

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 16 → a / b = 4 := by
sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l1016_101675


namespace NUMINAMATH_CALUDE_lap_time_improvement_l1016_101608

/-- Represents the performance data for a runner -/
structure Performance where
  laps : ℕ
  time : ℕ  -- time in minutes

/-- Calculates the lap time in seconds given a Performance -/
def lapTimeInSeconds (p : Performance) : ℚ :=
  (p.time * 60) / p.laps

theorem lap_time_improvement (initial : Performance) (current : Performance) 
  (h1 : initial.laps = 8) (h2 : initial.time = 36)
  (h3 : current.laps = 10) (h4 : current.time = 35) :
  lapTimeInSeconds initial - lapTimeInSeconds current = 60 := by
  sorry

end NUMINAMATH_CALUDE_lap_time_improvement_l1016_101608


namespace NUMINAMATH_CALUDE_problem_statement_l1016_101691

theorem problem_statement : 
  (∀ x y : ℝ, (Real.sqrt x + Real.sqrt y = 0) → (x = 0 ∧ y = 0)) ∨
  (∀ x : ℝ, (x^2 + 4*x - 5 = 0) → (x = -5)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1016_101691


namespace NUMINAMATH_CALUDE_clara_total_earnings_l1016_101652

/-- Represents a staff member at the cake shop -/
structure Staff :=
  (name : String)
  (hourlyRate : ℝ)
  (holidayBonus : ℝ)

/-- Calculates the total earnings for a staff member -/
def totalEarnings (s : Staff) (hoursWorked : ℝ) : ℝ :=
  s.hourlyRate * hoursWorked + s.holidayBonus

/-- Theorem: Clara's total earnings for the 2-month period -/
theorem clara_total_earnings :
  let clara : Staff := { name := "Clara", hourlyRate := 13, holidayBonus := 60 }
  let standardHours : ℝ := 20 * 8  -- 20 hours per week for 8 weeks
  let vacationHours : ℝ := 20 * 1.5  -- 10 days vacation (1.5 weeks)
  let claraHours : ℝ := standardHours - vacationHours
  totalEarnings clara claraHours = 1750 := by
  sorry

end NUMINAMATH_CALUDE_clara_total_earnings_l1016_101652


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l1016_101668

/-- Given an isosceles triangle with two equal sides of 15 cm and a base of 24 cm,
    prove that a similar triangle with a base of 60 cm has a perimeter of 135 cm. -/
theorem similar_triangle_perimeter 
  (original_equal_sides : ℝ)
  (original_base : ℝ)
  (similar_base : ℝ)
  (h_isosceles : original_equal_sides = 15)
  (h_original_base : original_base = 24)
  (h_similar_base : similar_base = 60) :
  let scale_factor := similar_base / original_base
  let similar_equal_sides := original_equal_sides * scale_factor
  similar_equal_sides * 2 + similar_base = 135 :=
by
  sorry

#check similar_triangle_perimeter

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l1016_101668


namespace NUMINAMATH_CALUDE_bluegrass_percentage_in_x_l1016_101690

/-- Represents a seed mixture with percentages of different grass types -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
def finalMixture (x y : SeedMixture) (xProportion : ℝ) : SeedMixture :=
  { ryegrass := x.ryegrass * xProportion + y.ryegrass * (1 - xProportion)
  , bluegrass := x.bluegrass * xProportion + y.bluegrass * (1 - xProportion)
  , fescue := x.fescue * xProportion + y.fescue * (1 - xProportion) }

theorem bluegrass_percentage_in_x 
  (x : SeedMixture) 
  (y : SeedMixture)
  (h1 : x.ryegrass = 0.4)
  (h2 : x.ryegrass + x.bluegrass = 1)
  (h3 : y.ryegrass = 0.25)
  (h4 : y.fescue = 0.75)
  (h5 : (finalMixture x y (1/3)).ryegrass = 0.3) :
  x.bluegrass = 0.6 := by
sorry

end NUMINAMATH_CALUDE_bluegrass_percentage_in_x_l1016_101690


namespace NUMINAMATH_CALUDE_max_b_minus_a_l1016_101651

/-- Given a function f and a constant a, finds the maximum value of b-a -/
theorem max_b_minus_a (a : ℝ) (f : ℝ → ℝ) (h1 : a > -1) 
  (h2 : ∀ x, f x = Real.exp x - a * x + (1/2) * x^2) 
  (h3 : ∀ x b, f x ≥ (1/2) * x^2 + x + b) :
  ∃ (b : ℝ), b - a ≤ 1 + Real.exp (-1) ∧ 
  (∀ c, (∀ x, f x ≥ (1/2) * x^2 + x + c) → c - a ≤ 1 + Real.exp (-1)) :=
sorry

end NUMINAMATH_CALUDE_max_b_minus_a_l1016_101651


namespace NUMINAMATH_CALUDE_tim_vocabulary_proof_l1016_101625

/-- Proves that given the conditions of Tim's word learning, his original vocabulary was 14600 words --/
theorem tim_vocabulary_proof (words_per_day : ℕ) (learning_days : ℕ) (increase_percentage : ℚ) : 
  words_per_day = 10 →
  learning_days = 730 →
  increase_percentage = 1/2 →
  (words_per_day * learning_days : ℚ) = increase_percentage * (words_per_day * learning_days + 14600) :=
by
  sorry

end NUMINAMATH_CALUDE_tim_vocabulary_proof_l1016_101625


namespace NUMINAMATH_CALUDE_segment_division_problem_l1016_101688

/-- The problem of determining the number of parts a unit segment is divided into -/
theorem segment_division_problem (min_distance : ℚ) (h1 : min_distance = 0.02857142857142857) : 
  (1 : ℚ) / min_distance = 35 := by
  sorry

end NUMINAMATH_CALUDE_segment_division_problem_l1016_101688


namespace NUMINAMATH_CALUDE_inequality_solution_l1016_101656

theorem inequality_solution (x : ℝ) : 
  1 / (x^3 + 1) > 4 / x + 2 / 5 ↔ -1 < x ∧ x < 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1016_101656


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l1016_101686

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 3 / 5 →  -- ratio of angles is 3:5
  |a - b| = 22.5 :=  -- positive difference is 22.5°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l1016_101686


namespace NUMINAMATH_CALUDE_sam_seashells_count_l1016_101609

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := 47

/-- The total number of seashells Sam and Mary found together -/
def total_seashells : ℕ := 65

/-- The number of seashells Sam found -/
def sam_seashells : ℕ := total_seashells - mary_seashells

theorem sam_seashells_count : sam_seashells = 18 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_count_l1016_101609


namespace NUMINAMATH_CALUDE_polynomial_division_l1016_101649

theorem polynomial_division (x : ℝ) :
  8 * x^4 - 4 * x^3 + 5 * x^2 - 9 * x + 3 = (x - 1) * (8 * x^3 - 4 * x^2 + 9 * x - 18) + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l1016_101649


namespace NUMINAMATH_CALUDE_set_equality_l1016_101638

def A : Set ℕ := {0, 1, 2, 4, 5, 7}
def B : Set ℕ := {1, 3, 6, 8, 9}
def C : Set ℕ := {3, 7, 8}

theorem set_equality : (A ∩ B) ∪ C = (A ∪ C) ∩ (B ∪ C) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l1016_101638


namespace NUMINAMATH_CALUDE_segment_properties_l1016_101618

/-- Given two points A(1, 2) and B(9, 14), prove the distance between them and their midpoint. -/
theorem segment_properties : 
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (9, 14)
  let distance := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (distance = 16) ∧ (midpoint = (5, 8)) := by
  sorry

end NUMINAMATH_CALUDE_segment_properties_l1016_101618


namespace NUMINAMATH_CALUDE_cube_difference_given_difference_l1016_101630

theorem cube_difference_given_difference (x : ℝ) (h : x - 1/x = 5) :
  x^3 - 1/x^3 = 140 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_given_difference_l1016_101630


namespace NUMINAMATH_CALUDE_quadratic_passes_through_point_l1016_101643

/-- A quadratic function passing through (-1, 0) given a - b + c = 0 -/
theorem quadratic_passes_through_point
  (a b c : ℝ) -- Coefficients of the quadratic function
  (h : a - b + c = 0) -- Given condition
  : let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c -- Definition of the quadratic function
    f (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_passes_through_point_l1016_101643


namespace NUMINAMATH_CALUDE_proposition_b_correct_l1016_101627

theorem proposition_b_correct :
  (∃ x : ℕ, x^3 ≤ x^2) ∧
  ((∀ x : ℝ, x > 1 → x^2 > 1) ∧ (∃ x : ℝ, x ≤ 1 ∧ x^2 > 1)) ∧
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_b_correct_l1016_101627


namespace NUMINAMATH_CALUDE_lineup_count_l1016_101629

/-- The number of ways to choose a lineup from a basketball team with specific constraints. -/
def chooseLineup (totalPlayers : ℕ) (twinCount : ℕ) (tripletCount : ℕ) (lineupSize : ℕ) : ℕ :=
  let nonSpecialPlayers := totalPlayers - twinCount - tripletCount
  let noSpecial := Nat.choose nonSpecialPlayers lineupSize
  let oneTriplet := tripletCount * Nat.choose nonSpecialPlayers (lineupSize - 1)
  let oneTwin := twinCount * Nat.choose nonSpecialPlayers (lineupSize - 1)
  let oneTripletOneTwin := tripletCount * twinCount * Nat.choose nonSpecialPlayers (lineupSize - 2)
  noSpecial + oneTriplet + oneTwin + oneTripletOneTwin

/-- The theorem stating the number of ways to choose the lineup under given constraints. -/
theorem lineup_count :
  chooseLineup 16 2 3 5 = 3102 :=
by sorry

end NUMINAMATH_CALUDE_lineup_count_l1016_101629


namespace NUMINAMATH_CALUDE_total_interest_calculation_l1016_101659

/-- Calculate the total interest after 10 years given the following conditions:
    1. The simple interest on the initial principal for 10 years is 400.
    2. The principal is trebled after 5 years. -/
theorem total_interest_calculation (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 400) 
  (h2 : P > 0) 
  (h3 : R > 0) : 
  P * R * 5 / 100 + 3 * P * R * 5 / 100 = 1000 := by
  sorry

#check total_interest_calculation

end NUMINAMATH_CALUDE_total_interest_calculation_l1016_101659


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l1016_101689

theorem min_value_quadratic_sum (a b c t k : ℝ) (hsum : a + b + c = t) (hk : k > 0) :
  k * a^2 + b^2 + k * c^2 ≥ k * t^2 / (k + 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l1016_101689


namespace NUMINAMATH_CALUDE_preimage_of_one_l1016_101605

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem preimage_of_one (x : ℝ) : f x = 1 ↔ x = -1 ∨ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_one_l1016_101605


namespace NUMINAMATH_CALUDE_exists_divisible_by_three_l1016_101662

/-- A circular sequence of natural numbers satisfying specific neighboring conditions -/
structure CircularSequence where
  nums : Fin 99 → ℕ
  neighbor_condition : ∀ i : Fin 99, 
    (nums i.succ = 2 * nums i) ∨ 
    (nums i.succ = nums i + 1) ∨ 
    (nums i.succ = nums i + 2)

/-- Theorem: In any CircularSequence, there exists a number divisible by 3 -/
theorem exists_divisible_by_three (seq : CircularSequence) :
  ∃ i : Fin 99, 3 ∣ seq.nums i := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_three_l1016_101662


namespace NUMINAMATH_CALUDE_solve_age_problem_l1016_101697

def age_problem (rona_age : ℕ) : Prop :=
  let rachel_age := 2 * rona_age
  let collete_age := rona_age / 2
  let tommy_age := collete_age + rona_age
  rachel_age + rona_age + collete_age + tommy_age = 40

theorem solve_age_problem : age_problem 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_age_problem_l1016_101697


namespace NUMINAMATH_CALUDE_yellow_given_popped_prob_l1016_101610

-- Define the probabilities of kernel colors in the bag
def white_prob : ℚ := 1/2
def yellow_prob : ℚ := 1/3
def blue_prob : ℚ := 1/6

-- Define the probabilities of popping for each color
def white_pop_prob : ℚ := 2/3
def yellow_pop_prob : ℚ := 1/2
def blue_pop_prob : ℚ := 3/4

-- State the theorem
theorem yellow_given_popped_prob :
  let total_pop_prob := white_prob * white_pop_prob + yellow_prob * yellow_pop_prob + blue_prob * blue_pop_prob
  (yellow_prob * yellow_pop_prob) / total_pop_prob = 4/23 := by
  sorry

end NUMINAMATH_CALUDE_yellow_given_popped_prob_l1016_101610


namespace NUMINAMATH_CALUDE_product_prs_is_54_l1016_101611

theorem product_prs_is_54 (p r s : ℕ) : 
  3^p + 3^5 = 270 → 
  2^r + 58 = 122 → 
  7^2 + 5^s = 2504 → 
  p * r * s = 54 := by
sorry

end NUMINAMATH_CALUDE_product_prs_is_54_l1016_101611


namespace NUMINAMATH_CALUDE_physics_players_l1016_101615

def total_players : ℕ := 30
def math_players : ℕ := 15
def both_subjects : ℕ := 6

theorem physics_players :
  ∃ (physics_players : ℕ),
    physics_players = total_players - (math_players - both_subjects) ∧
    physics_players = 21 :=
by sorry

end NUMINAMATH_CALUDE_physics_players_l1016_101615


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1016_101616

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.sin y = 2008)
  (h2 : x + 2008 * Real.cos y = 2007)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1016_101616


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1016_101635

theorem arithmetic_expression_equality : 
  (0.15 : ℝ)^3 - (0.06 : ℝ)^3 / (0.15 : ℝ)^2 + 0.009 + (0.06 : ℝ)^2 = 0.006375 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1016_101635


namespace NUMINAMATH_CALUDE_five_teachers_three_types_l1016_101671

/-- The number of ways to assign teachers to question types. -/
def assignment_count (teachers : ℕ) (question_types : ℕ) : ℕ :=
  -- Number of ways to assign teachers to question types
  -- with at least one teacher per type
  sorry

/-- Theorem stating the number of assignment methods for 5 teachers and 3 question types. -/
theorem five_teachers_three_types : assignment_count 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_teachers_three_types_l1016_101671


namespace NUMINAMATH_CALUDE_binomial_10_2_l1016_101600

theorem binomial_10_2 : (10 : ℕ).choose 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_2_l1016_101600


namespace NUMINAMATH_CALUDE_expand_triple_product_l1016_101665

theorem expand_triple_product (x y z : ℝ) :
  (x + 8) * (3 * y + 12) * (2 * z + 4) =
  6 * x * y * z + 12 * x * z + 24 * y * z + 12 * x * y + 48 * x + 96 * y + 96 * z + 384 := by
  sorry

end NUMINAMATH_CALUDE_expand_triple_product_l1016_101665


namespace NUMINAMATH_CALUDE_prob_different_colors_specific_l1016_101623

/-- The probability of drawing two chips of different colors from a bag --/
def prob_different_colors (blue red yellow : ℕ) : ℚ :=
  let total := blue + red + yellow
  let prob_blue := blue / total
  let prob_red := red / total
  let prob_yellow := yellow / total
  let prob_not_blue := (red + yellow) / (total - 1)
  let prob_not_red := (blue + yellow) / (total - 1)
  let prob_not_yellow := (blue + red) / (total - 1)
  prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow

/-- Theorem: The probability of drawing two chips of different colors from a bag with 7 blue, 6 red, and 5 yellow chips is 122/153 --/
theorem prob_different_colors_specific : prob_different_colors 7 6 5 = 122 / 153 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_specific_l1016_101623


namespace NUMINAMATH_CALUDE_monthly_salary_proof_l1016_101661

/-- Proves that a person's monthly salary is 1000 Rs, given the conditions -/
theorem monthly_salary_proof (salary : ℝ) : salary = 1000 :=
  let initial_savings_rate : ℝ := 0.25
  let initial_expense_rate : ℝ := 1 - initial_savings_rate
  let expense_increase_rate : ℝ := 0.10
  let new_savings_amount : ℝ := 175

  have h1 : initial_savings_rate * salary = 
            salary - initial_expense_rate * salary := by sorry

  have h2 : new_savings_amount = 
            salary - (initial_expense_rate * salary * (1 + expense_increase_rate)) := by sorry

  sorry

end NUMINAMATH_CALUDE_monthly_salary_proof_l1016_101661
