import Mathlib

namespace NUMINAMATH_CALUDE_min_surface_area_angle_l724_72423

/-- The angle that minimizes the surface area of a rotated right triangle -/
theorem min_surface_area_angle (AC BC CD : ℝ) (h1 : AC = 3) (h2 : BC = 4) (h3 : CD = 10) :
  let α := Real.arctan (2 / 3)
  let surface_area (θ : ℝ) := π * (240 - 12 * (2 * Real.sin θ + 3 * Real.cos θ))
  ∀ θ, surface_area α ≤ surface_area θ := by
sorry


end NUMINAMATH_CALUDE_min_surface_area_angle_l724_72423


namespace NUMINAMATH_CALUDE_lateral_surface_area_cylinder_l724_72453

/-- The lateral surface area of a cylinder with radius 1 and height 2 is 4π. -/
theorem lateral_surface_area_cylinder : 
  let r : ℝ := 1
  let h : ℝ := 2
  2 * Real.pi * r * h = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_lateral_surface_area_cylinder_l724_72453


namespace NUMINAMATH_CALUDE_best_approximation_l724_72485

/-- The function representing the sum of squared differences between measurements and x -/
def y (x : ℝ) : ℝ := (x - 6.5)^2 + (x - 5.9)^2 + (x - 6.0)^2 + (x - 6.7)^2 + (x - 4.5)^2

/-- The theorem stating that 5.92 minimizes the function y -/
theorem best_approximation :
  ∀ x : ℝ, y 5.92 ≤ y x :=
sorry

end NUMINAMATH_CALUDE_best_approximation_l724_72485


namespace NUMINAMATH_CALUDE_square_diagonal_side_area_l724_72450

/-- Given a square with diagonal length 4, prove its side length and area. -/
theorem square_diagonal_side_area :
  ∃ (side_length area : ℝ),
    4^2 = 2 * side_length^2 ∧
    side_length = 2 * Real.sqrt 2 ∧
    area = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_side_area_l724_72450


namespace NUMINAMATH_CALUDE_complex_real_condition_l724_72468

theorem complex_real_condition (i : ℂ) (m : ℝ) : 
  i * i = -1 →
  (1 / (2 + i) + m * i).im = 0 →
  m = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l724_72468


namespace NUMINAMATH_CALUDE_most_trailing_zeros_l724_72405

-- Define a function to count trailing zeros
def countTrailingZeros (n : ℕ) : ℕ := sorry

-- Define the arithmetic expressions
def expr1 : ℕ := 300 + 60
def expr2 : ℕ := 22 * 5
def expr3 : ℕ := 25 * 4
def expr4 : ℕ := 400 / 8

-- Theorem statement
theorem most_trailing_zeros :
  countTrailingZeros expr3 ≥ countTrailingZeros expr1 ∧
  countTrailingZeros expr3 ≥ countTrailingZeros expr2 ∧
  countTrailingZeros expr3 ≥ countTrailingZeros expr4 :=
by sorry

end NUMINAMATH_CALUDE_most_trailing_zeros_l724_72405


namespace NUMINAMATH_CALUDE_illuminated_area_ratio_l724_72402

theorem illuminated_area_ratio (r : ℝ) (h : r > 0) :
  let sphere_radius := r
  let light_distance := 3 * r
  let illuminated_area := 2 * Real.pi * r * (r - r / 4)
  let cone_base_radius := r / 4 * Real.sqrt 15
  let cone_slant_height := r * Real.sqrt 15
  let cone_lateral_area := Real.pi * cone_base_radius * cone_slant_height
  illuminated_area / cone_lateral_area = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_illuminated_area_ratio_l724_72402


namespace NUMINAMATH_CALUDE_sector_angle_l724_72451

/-- Given a circular sector with arc length 4 and area 4, 
    prove that the absolute value of its central angle in radians is 2. -/
theorem sector_angle (r : ℝ) (θ : ℝ) (h1 : r * θ = 4) (h2 : (1/2) * r^2 * θ = 4) : 
  |θ| = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_angle_l724_72451


namespace NUMINAMATH_CALUDE_total_treats_is_275_l724_72463

/-- The total number of treats Mary, John, and Sue have -/
def total_treats (chewing_gums chocolate_bars lollipops cookies other_candies : ℕ) : ℕ :=
  chewing_gums + chocolate_bars + lollipops + cookies + other_candies

/-- Theorem stating that the total number of treats is 275 -/
theorem total_treats_is_275 :
  total_treats 60 55 70 50 40 = 275 := by
  sorry

end NUMINAMATH_CALUDE_total_treats_is_275_l724_72463


namespace NUMINAMATH_CALUDE_money_lent_to_B_l724_72488

/-- Proves that the amount lent to B is 4000, given the problem conditions --/
theorem money_lent_to_B (total : ℕ) (rate_A rate_B : ℚ) (years : ℕ) (interest_diff : ℕ) :
  total = 10000 →
  rate_A = 15 / 100 →
  rate_B = 18 / 100 →
  years = 2 →
  interest_diff = 360 →
  ∃ (amount_A amount_B : ℕ),
    amount_A + amount_B = total ∧
    amount_A * rate_A * years = (amount_B * rate_B * years + interest_diff) ∧
    amount_B = 4000 :=
by sorry

end NUMINAMATH_CALUDE_money_lent_to_B_l724_72488


namespace NUMINAMATH_CALUDE_mod_fifteen_equivalence_l724_72401

theorem mod_fifteen_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 14567 [ZMOD 15] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_fifteen_equivalence_l724_72401


namespace NUMINAMATH_CALUDE_brendans_hourly_wage_l724_72424

-- Define Brendan's work schedule
def hours_per_week : ℕ := 2 * 8 + 1 * 12

-- Define Brendan's hourly tip rate
def hourly_tips : ℚ := 12

-- Define the fraction of tips reported to IRS
def reported_tips_fraction : ℚ := 1 / 3

-- Define the tax rate
def tax_rate : ℚ := 1 / 5

-- Define the weekly tax amount
def weekly_tax : ℚ := 56

-- Theorem to prove Brendan's hourly wage
theorem brendans_hourly_wage :
  ∃ (hourly_wage : ℚ),
    hourly_wage * hours_per_week +
    reported_tips_fraction * (hourly_tips * hours_per_week) =
    weekly_tax / tax_rate ∧
    hourly_wage = 6 := by
  sorry

end NUMINAMATH_CALUDE_brendans_hourly_wage_l724_72424


namespace NUMINAMATH_CALUDE_max_pairs_after_loss_l724_72456

/-- Represents the types of shoes -/
inductive ShoeType
| Sneaker
| Sandal
| Boot

/-- Represents the colors of shoes -/
inductive ShoeColor
| Red
| Blue
| Green
| Black

/-- Represents the sizes of shoes -/
inductive ShoeSize
| Size6
| Size7
| Size8

/-- Represents a shoe with its type, color, and size -/
structure Shoe :=
  (type : ShoeType)
  (color : ShoeColor)
  (size : ShoeSize)

/-- Represents the initial collection of shoes -/
def initial_collection : Finset Shoe := sorry

/-- The number of shoes lost -/
def shoes_lost : Nat := 9

/-- Theorem stating the maximum number of complete pairs after losing shoes -/
theorem max_pairs_after_loss :
  ∃ (remaining_collection : Finset Shoe),
    remaining_collection ⊆ initial_collection ∧
    (initial_collection.card - remaining_collection.card = shoes_lost) ∧
    (∀ (s : Shoe), s ∈ remaining_collection →
      ∃ (s' : Shoe), s' ∈ remaining_collection ∧ s ≠ s' ∧
        s.type = s'.type ∧ s.color = s'.color ∧ s.size = s'.size) ∧
    remaining_collection.card = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_pairs_after_loss_l724_72456


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l724_72495

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, 2^x > 2 → 1/x < 1) ∧ 
  (∃ x, 1/x < 1 ∧ 2^x ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l724_72495


namespace NUMINAMATH_CALUDE_houses_in_block_l724_72437

theorem houses_in_block (junk_mail_per_house : ℕ) (total_junk_mail : ℕ) 
  (h1 : junk_mail_per_house = 2) 
  (h2 : total_junk_mail = 14) : 
  total_junk_mail / junk_mail_per_house = 7 := by
sorry

end NUMINAMATH_CALUDE_houses_in_block_l724_72437


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l724_72459

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) = (4 * Real.pi * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l724_72459


namespace NUMINAMATH_CALUDE_matrix_transformation_l724_72404

/-- Given a 2x2 matrix M with eigenvector [1, 1] corresponding to eigenvalue 8,
    prove that the transformation of point (-1, 2) by M results in (-2, 4) -/
theorem matrix_transformation (a b : ℝ) : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; 4, b]
  (M.mulVec ![1, 1] = ![8, 8]) →
  (M.mulVec ![-1, 2] = ![-2, 4]) := by
sorry

end NUMINAMATH_CALUDE_matrix_transformation_l724_72404


namespace NUMINAMATH_CALUDE_largest_negative_integer_congruence_l724_72415

theorem largest_negative_integer_congruence :
  ∃ x : ℤ, x < 0 ∧ 
    (42 * x + 30) % 24 = 26 % 24 ∧
    x % 12 = (-2) % 12 ∧
    ∀ y : ℤ, y < 0 → (42 * y + 30) % 24 = 26 % 24 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_integer_congruence_l724_72415


namespace NUMINAMATH_CALUDE_matrix_sum_equals_result_l724_72436

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 0; 1, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-5, -7; 4, -9]

theorem matrix_sum_equals_result : A + B = !![(-2), (-7); 5, (-7)] := by sorry

end NUMINAMATH_CALUDE_matrix_sum_equals_result_l724_72436


namespace NUMINAMATH_CALUDE_triangle_area_sum_l724_72411

theorem triangle_area_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + y^2 = 3^2)
  (h2 : y^2 + y*z + z^2 = 4^2)
  (h3 : x^2 + Real.sqrt 3 * x*z + z^2 = 5^2) :
  2*x*y + x*z + Real.sqrt 3 * y*z = 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_sum_l724_72411


namespace NUMINAMATH_CALUDE_log_relation_l724_72440

theorem log_relation (a b : ℝ) (h1 : a = Real.log 625 / Real.log 4) (h2 : b = Real.log 25 / Real.log 5) :
  a = 4 / b := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l724_72440


namespace NUMINAMATH_CALUDE_curve_C_equation_l724_72457

/-- The equation of curve C -/
def curve_C (a : ℝ) (x y : ℝ) : Prop :=
  a * x^2 + a * y^2 - 2 * a^2 * x - 4 * y = 0

/-- The equation of line l -/
def line_l (x y : ℝ) : Prop :=
  y = -2 * x + 4

/-- M and N are distinct intersection points of curve C and line l -/
def intersection_points (a : ℝ) (M N : ℝ × ℝ) : Prop :=
  M ≠ N ∧ curve_C a M.1 M.2 ∧ curve_C a N.1 N.2 ∧ line_l M.1 M.2 ∧ line_l N.1 N.2

/-- The distance from origin O to M is equal to the distance from O to N -/
def equal_distances (M N : ℝ × ℝ) : Prop :=
  M.1^2 + M.2^2 = N.1^2 + N.2^2

theorem curve_C_equation (a : ℝ) (M N : ℝ × ℝ) :
  a ≠ 0 →
  intersection_points a M N →
  equal_distances M N →
  ∃ x y : ℝ, x^2 + y^2 - 4*x - 2*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_curve_C_equation_l724_72457


namespace NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_l724_72410

/-- Theorem: For an ellipse with center (3, -5), semi-major axis length 7, and semi-minor axis length 4,
    the sum of its center coordinates and axis lengths is 9. -/
theorem ellipse_sum_coordinates_and_axes :
  ∀ (h k a b : ℝ),
    h = 3 →
    k = -5 →
    a = 7 →
    b = 4 →
    h + k + a + b = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_l724_72410


namespace NUMINAMATH_CALUDE_distance_to_nearest_town_l724_72428

theorem distance_to_nearest_town (d : ℝ) 
  (h1 : ¬(d ≥ 8))  -- Alice's statement is false
  (h2 : ¬(d ≤ 7))  -- Bob's statement is false
  (h3 : d ≠ 5)     -- Charlie's statement is false
  : 7 < d ∧ d < 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_nearest_town_l724_72428


namespace NUMINAMATH_CALUDE_paper_folding_l724_72455

theorem paper_folding (n : ℕ) : 2^n = 128 → n = 7 := by sorry

end NUMINAMATH_CALUDE_paper_folding_l724_72455


namespace NUMINAMATH_CALUDE_biology_quiz_probability_l724_72477

/-- The number of questions in the quiz --/
def total_questions : ℕ := 20

/-- The number of questions Jessica guesses randomly --/
def guessed_questions : ℕ := 5

/-- The number of answer choices for each question --/
def answer_choices : ℕ := 4

/-- The probability of getting a single question correct by random guessing --/
def prob_correct : ℚ := 1 / answer_choices

/-- The probability of getting at least two questions correct out of five randomly guessed questions --/
def prob_at_least_two_correct : ℚ := 47 / 128

theorem biology_quiz_probability :
  (1 : ℚ) - (Nat.choose guessed_questions 0 * (1 - prob_correct)^guessed_questions +
             Nat.choose guessed_questions 1 * (1 - prob_correct)^(guessed_questions - 1) * prob_correct) =
  prob_at_least_two_correct :=
sorry

end NUMINAMATH_CALUDE_biology_quiz_probability_l724_72477


namespace NUMINAMATH_CALUDE_mental_health_survey_is_comprehensive_l724_72482

/-- Represents a survey --/
structure Survey where
  description : String
  population : Set String
  environment : String

/-- Conditions for a comprehensive survey --/
def is_comprehensive (s : Survey) : Prop :=
  s.population.Finite ∧
  s.population.Nonempty ∧
  (∀ x ∈ s.population, ∃ y, y = x) ∧
  s.environment = "Contained"

/-- The survey on students' mental health --/
def mental_health_survey : Survey :=
  { description := "Survey on the current status of students' mental health in a school in Huicheng District"
  , population := {"Students in a school in Huicheng District"}
  , environment := "Contained" }

/-- Theorem stating that the mental health survey is comprehensive --/
theorem mental_health_survey_is_comprehensive :
  is_comprehensive mental_health_survey :=
sorry

end NUMINAMATH_CALUDE_mental_health_survey_is_comprehensive_l724_72482


namespace NUMINAMATH_CALUDE_sum_of_ABC_values_l724_72430

/-- A function that represents the number A5B79C given digits A, B, and C -/
def number (A B C : ℕ) : ℕ := A * 100000 + 5 * 10000 + B * 1000 + 7 * 100 + 9 * 10 + C

/-- Predicate to check if a number is a single digit -/
def is_single_digit (n : ℕ) : Prop := n ≤ 9

/-- The sum of all possible values of A+B+C given the conditions -/
def sum_of_possible_values : ℕ := 29

/-- The main theorem -/
theorem sum_of_ABC_values (A B C : ℕ) 
  (hA : is_single_digit A) (hB : is_single_digit B) (hC : is_single_digit C)
  (h_div : (number A B C) % 11 = 0) : 
  sum_of_possible_values = 29 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ABC_values_l724_72430


namespace NUMINAMATH_CALUDE_intersection_of_parallel_lines_l724_72480

/-- The number of parallelograms formed by the intersection of two sets of parallel lines -/
def parallelograms (n m : ℕ) : ℕ := n.choose 2 * m

/-- Given two sets of parallel lines intersecting in a plane, 
    where one set has 8 lines and they form 280 parallelograms, 
    prove that the other set must have 10 lines -/
theorem intersection_of_parallel_lines : 
  ∃ (n : ℕ), n > 0 ∧ parallelograms n 8 = 280 ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_parallel_lines_l724_72480


namespace NUMINAMATH_CALUDE_exchange_process_duration_l724_72470

/-- Represents the number of children of each gender -/
def n : ℕ := 10

/-- Calculates the sum of the first n even numbers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- Calculates the sum of the first n natural numbers -/
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the total number of swaps required to move boys from even positions to the first n positions -/
def total_swaps (n : ℕ) : ℕ := sum_even n - sum_natural n

theorem exchange_process_duration :
  total_swaps n = 55 ∧ total_swaps n < 60 := by sorry

end NUMINAMATH_CALUDE_exchange_process_duration_l724_72470


namespace NUMINAMATH_CALUDE_most_likely_genotype_combination_l724_72407

/-- Represents the alleles for rabbit fur type -/
inductive Allele
| H  -- Dominant hairy allele
| h  -- Recessive hairy allele
| S  -- Dominant smooth allele
| s  -- Recessive smooth allele

/-- Represents the genotype of a rabbit -/
structure Genotype where
  allele1 : Allele
  allele2 : Allele

/-- Determines if a rabbit has hairy fur based on its genotype -/
def hasHairyFur (g : Genotype) : Bool :=
  match g.allele1, g.allele2 with
  | Allele.H, _ => true
  | _, Allele.H => true
  | _, _ => false

/-- The probability of the hairy fur allele in the population -/
def p : ℝ := 0.1

/-- Represents the result of mating two rabbits -/
structure MatingResult where
  parent1 : Genotype
  parent2 : Genotype
  offspringCount : Nat
  allOffspringHairy : Bool

/-- The theorem to be proved -/
theorem most_likely_genotype_combination (result : MatingResult) 
  (h1 : result.parent1.allele1 = Allele.H ∨ result.parent1.allele2 = Allele.H)
  (h2 : result.parent2.allele1 = Allele.S ∨ result.parent2.allele2 = Allele.S)
  (h3 : result.offspringCount = 4)
  (h4 : result.allOffspringHairy = true) :
  (result.parent1 = Genotype.mk Allele.H Allele.H ∧ 
   result.parent2 = Genotype.mk Allele.S Allele.h) :=
sorry

end NUMINAMATH_CALUDE_most_likely_genotype_combination_l724_72407


namespace NUMINAMATH_CALUDE_original_number_proof_l724_72422

theorem original_number_proof : ∃! N : ℕ, N > 0 ∧ (N - 5) % 13 = 0 ∧ ∀ M : ℕ, M > 0 → (M - 5) % 13 = 0 → M ≥ N :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l724_72422


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l724_72418

theorem simplify_and_evaluate (x : ℝ) (h : x = 4) : 
  (2 - 2*x/(x-2)) / ((x^2 - 4) / (x^2 - 4*x + 4)) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l724_72418


namespace NUMINAMATH_CALUDE_sticker_distribution_l724_72467

theorem sticker_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (Nat.choose (n + k - 1) (k - 1)) = 1001 :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l724_72467


namespace NUMINAMATH_CALUDE_second_tract_width_l724_72497

/-- Given two rectangular tracts of land, prove that the width of the second tract is 630 meters -/
theorem second_tract_width (length1 width1 length2 combined_area : ℝ)
  (h1 : length1 = 300)
  (h2 : width1 = 500)
  (h3 : length2 = 250)
  (h4 : combined_area = 307500)
  (h5 : combined_area = length1 * width1 + length2 * (combined_area - length1 * width1) / length2) :
  (combined_area - length1 * width1) / length2 = 630 := by
  sorry

end NUMINAMATH_CALUDE_second_tract_width_l724_72497


namespace NUMINAMATH_CALUDE_factor_expression_l724_72465

theorem factor_expression :
  ∀ x : ℝ, 63 * x^2 + 42 = 21 * (3 * x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l724_72465


namespace NUMINAMATH_CALUDE_rectangle_not_stable_l724_72412

-- Define the shape type
inductive Shape
| AcuteTriangle
| Rectangle
| RightTriangle
| IsoscelesTriangle

-- Define stability property
def IsStable (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => False
  | _ => True

-- State the theorem
theorem rectangle_not_stable :
  ∀ (s : Shape), ¬(IsStable s) ↔ s = Shape.Rectangle :=
by sorry

end NUMINAMATH_CALUDE_rectangle_not_stable_l724_72412


namespace NUMINAMATH_CALUDE_unique_solution_l724_72421

/-- Represents a six-digit number with distinct digits -/
structure SixDigitNumber where
  digits : Fin 6 → Fin 10
  distinct : ∀ i j, i ≠ j → digits i ≠ digits j

/-- The equation 6 × AOBMEP = 7 × MEPAOB -/
def EquationHolds (n : SixDigitNumber) : Prop :=
  6 * (100000 * n.digits 0 + 10000 * n.digits 1 + 1000 * n.digits 2 +
       100 * n.digits 3 + 10 * n.digits 4 + n.digits 5) =
  7 * (100000 * n.digits 3 + 10000 * n.digits 4 + 1000 * n.digits 5 +
       100 * n.digits 0 + 10 * n.digits 1 + n.digits 2)

/-- The unique solution to the equation -/
def Solution : SixDigitNumber where
  digits := fun i => match i with
    | 0 => 5  -- A
    | 1 => 3  -- O
    | 2 => 8  -- B
    | 3 => 4  -- M
    | 4 => 6  -- E
    | 5 => 1  -- P
  distinct := by sorry

theorem unique_solution :
  ∀ n : SixDigitNumber, EquationHolds n ↔ n = Solution := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l724_72421


namespace NUMINAMATH_CALUDE_rice_and_grain_separation_l724_72454

/-- Represents the amount of rice in dan -/
def total_rice : ℕ := 1536

/-- Represents the sample size in grains -/
def sample_size : ℕ := 256

/-- Represents the number of mixed grain in the sample -/
def mixed_grain_sample : ℕ := 18

/-- Calculates the amount of mixed grain in the entire batch -/
def mixed_grain_total : ℕ := total_rice * mixed_grain_sample / sample_size

theorem rice_and_grain_separation :
  mixed_grain_total = 108 := by
  sorry

end NUMINAMATH_CALUDE_rice_and_grain_separation_l724_72454


namespace NUMINAMATH_CALUDE_set_intersection_complement_l724_72466

theorem set_intersection_complement (U A B : Set ℤ) : 
  U = Set.univ ∧ A = {-1, 1, 2} ∧ B = {-1, 1} → A ∩ (U \ B) = {2} :=
by
  sorry

end NUMINAMATH_CALUDE_set_intersection_complement_l724_72466


namespace NUMINAMATH_CALUDE_inequality_proof_l724_72442

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  a^2 / (b - 1) + b^2 / (a - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l724_72442


namespace NUMINAMATH_CALUDE_prize_problem_l724_72416

-- Define the types of prizes
inductive PrizeType
| A
| B

-- Define the unit prices
def unit_price : PrizeType → ℕ
| PrizeType.A => 10
| PrizeType.B => 15

-- Define the total number of prizes
def total_prizes : ℕ := 100

-- Define the maximum total cost
def max_total_cost : ℕ := 1160

-- Define the condition for the quantity of type A prizes
def type_a_condition (a b : ℕ) : Prop := a ≤ 3 * b

-- Define the cost function
def cost (a b : ℕ) : ℕ := a * unit_price PrizeType.A + b * unit_price PrizeType.B

-- Define the valid purchasing plan
def valid_plan (a b : ℕ) : Prop :=
  a + b = total_prizes ∧
  cost a b ≤ max_total_cost ∧
  type_a_condition a b

-- Theorem statement
theorem prize_problem :
  (3 * unit_price PrizeType.A + 2 * unit_price PrizeType.B = 60) ∧
  (unit_price PrizeType.A + 3 * unit_price PrizeType.B = 55) ∧
  (∃ (plans : List (ℕ × ℕ)), 
    plans.length = 8 ∧
    (∀ p ∈ plans, valid_plan p.1 p.2) ∧
    (∃ (a b : ℕ), (a, b) ∈ plans ∧ cost a b = 1125 ∧ 
      ∀ (x y : ℕ), valid_plan x y → cost x y ≥ 1125)) :=
by sorry

end NUMINAMATH_CALUDE_prize_problem_l724_72416


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l724_72444

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 9 → b = 9 → c = 4 →
  (a + b > c ∧ b + c > a ∧ a + c > b) →  -- Triangle inequality
  a + b + c = 22 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l724_72444


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l724_72429

/-- Given vectors a and b in ℝ², and c = a + k*b, prove that if a ⊥ c, then k = -10/3 -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (3, 1))
  (h2 : b = (1, 0))
  (h3 : c = a + k • b)
  (h4 : a.1 * c.1 + a.2 * c.2 = 0) : 
  k = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l724_72429


namespace NUMINAMATH_CALUDE_min_balls_for_fifteen_guarantee_l724_72474

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n of a single color -/
def minBallsForGuarantee (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem min_balls_for_fifteen_guarantee : 
  let counts : BallCounts := {
    red := 28,
    green := 20,
    yellow := 19,
    blue := 13,
    white := 11,
    black := 9
  }
  minBallsForGuarantee counts 15 = 76 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_fifteen_guarantee_l724_72474


namespace NUMINAMATH_CALUDE_train_passing_jogger_l724_72434

theorem train_passing_jogger (jogger_speed train_speed : ℝ) 
  (train_length initial_distance : ℝ) :
  jogger_speed = 9 →
  train_speed = 45 →
  train_length = 120 →
  initial_distance = 240 →
  (train_speed - jogger_speed) * (5 / 18) * 
    ((initial_distance + train_length) / ((train_speed - jogger_speed) * (5 / 18))) = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_l724_72434


namespace NUMINAMATH_CALUDE_max_colors_is_two_l724_72487

/-- A coloring function for integers -/
def Coloring := ℕ → ℕ

/-- Checks if a coloring is valid for the given range and condition -/
def is_valid_coloring (χ : Coloring) : Prop :=
  ∀ a b c : ℕ, 49 ≤ a ∧ a ≤ 94 → 49 ≤ b ∧ b ≤ 94 → 49 ≤ c ∧ c ≤ 94 →
    χ a = χ b → χ c ≠ χ a → ¬(c ∣ (a + b))

/-- The maximum number of colors used in a valid coloring -/
def max_colors : ℕ := 2

/-- The main theorem: The maximum number of colors in a valid coloring is 2 -/
theorem max_colors_is_two :
  ∀ χ : Coloring, is_valid_coloring χ →
    ∃ n : ℕ, n ≤ max_colors ∧
      (∀ k : ℕ, 49 ≤ k ∧ k ≤ 94 → χ k < n) ∧
      (∀ m : ℕ, (∀ k : ℕ, 49 ≤ k ∧ k ≤ 94 → χ k < m) → max_colors ≤ m) :=
sorry

end NUMINAMATH_CALUDE_max_colors_is_two_l724_72487


namespace NUMINAMATH_CALUDE_largest_number_with_conditions_l724_72499

def is_valid_digit (d : Nat) : Prop := d = 1 ∨ d = 2 ∨ d = 3

def digits_sum_to_13 (n : Nat) : Prop :=
  (Nat.digits 10 n).sum = 13

def all_digits_valid (n : Nat) : Prop :=
  ∀ d ∈ Nat.digits 10 n, is_valid_digit d

def is_largest_with_conditions (n : Nat) : Prop :=
  digits_sum_to_13 n ∧ all_digits_valid n ∧
  ∀ m : Nat, digits_sum_to_13 m → all_digits_valid m → m ≤ n

theorem largest_number_with_conditions :
  is_largest_with_conditions 322222 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_conditions_l724_72499


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l724_72484

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 18)
  (h2 : a * b + c + d = 85)
  (h3 : a * d + b * c = 187)
  (h4 : c * d = 110) :
  a^2 + b^2 + c^2 + d^2 ≤ 120 ∧ 
  ∃ (a' b' c' d' : ℝ), 
    a' + b' = 18 ∧ 
    a' * b' + c' + d' = 85 ∧ 
    a' * d' + b' * c' = 187 ∧ 
    c' * d' = 110 ∧ 
    a'^2 + b'^2 + c'^2 + d'^2 = 120 :=
by sorry

#check max_sum_of_squares

end NUMINAMATH_CALUDE_max_sum_of_squares_l724_72484


namespace NUMINAMATH_CALUDE_problem_statement_l724_72460

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : Real.exp a + Real.log b = 1) :
  (a + Real.log b < 0) ∧ (Real.exp a + b > 2) ∧ (a + b > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l724_72460


namespace NUMINAMATH_CALUDE_greatest_triangle_perimeter_l724_72431

theorem greatest_triangle_perimeter : ∃ (a b c : ℕ),
  (a > 0 ∧ b > 0 ∧ c > 0) ∧  -- positive integer side lengths
  (b = 4 * a) ∧              -- one side is four times as long as a second side
  (c = 20) ∧                 -- the third side has length 20
  (a + b > c ∧ b + c > a ∧ c + a > b) ∧  -- triangle inequality
  (∀ (x y z : ℕ), (x > 0 ∧ y > 0 ∧ z > 0) → 
    (y = 4 * x) → (z = 20) → 
    (x + y > z ∧ y + z > x ∧ z + x > y) →
    (x + y + z ≤ a + b + c)) ∧
  (a + b + c = 50) :=
by sorry

end NUMINAMATH_CALUDE_greatest_triangle_perimeter_l724_72431


namespace NUMINAMATH_CALUDE_triangle_side_ratio_sum_l724_72403

/-- Given a triangle with side lengths a, b, and c, 
    the sum of the ratios of each side length to the difference between 
    the sum of the other two sides and itself is greater than or equal to 3. -/
theorem triangle_side_ratio_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_sum_l724_72403


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l724_72448

/-- The inradius of a right triangle with sides 9, 40, and 41 is 4 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 40 ∧ c = 41 →
  a^2 + b^2 = c^2 →
  r = (a * b) / (2 * (a + b + c)) →
  r = 4 := by sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l724_72448


namespace NUMINAMATH_CALUDE_complex_multiplication_l724_72461

theorem complex_multiplication (z₁ z₂ : ℂ) (h₁ : z₁ = 1 - I) (h₂ : z₂ = 2 + I) :
  z₁ * z₂ = 3 - I := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l724_72461


namespace NUMINAMATH_CALUDE_f_2009_is_zero_l724_72409

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2009_is_zero (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : is_odd_function (fun x ↦ f (x - 1))) : 
  f 2009 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2009_is_zero_l724_72409


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l724_72426

theorem partial_fraction_decomposition :
  ∀ (x : ℝ) (A B C : ℝ), 
    A = -6 ∧ B = 4 ∧ C = 5 →
    x ≠ 0 →
    (-2 * x^2 + 5 * x - 6) / (x^3 + x) = A / x + (B * x + C) / (x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l724_72426


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l724_72438

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l724_72438


namespace NUMINAMATH_CALUDE_difference_of_squares_l724_72492

theorem difference_of_squares : (502 : ℤ) * 502 - (501 : ℤ) * 503 = 1 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l724_72492


namespace NUMINAMATH_CALUDE_size_relationship_l724_72469

theorem size_relationship : 
  let a : ℝ := 1 + Real.sqrt 7
  let b : ℝ := Real.sqrt 3 + Real.sqrt 5
  let c : ℝ := 4
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_size_relationship_l724_72469


namespace NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l724_72489

/-- Proves that for a rectangular hall with width equal to half its length
    and an area of 800 square meters, the difference between its length
    and width is 20 meters. -/
theorem rectangular_hall_dimension_difference
  (length width : ℝ)
  (h1 : width = length / 2)
  (h2 : length * width = 800) :
  length - width = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l724_72489


namespace NUMINAMATH_CALUDE_alice_above_quota_l724_72498

def alice_sales (adidas_price nike_price reebok_price : ℕ)
                (adidas_qty nike_qty reebok_qty : ℕ)
                (quota : ℕ) : ℤ :=
  (adidas_price * adidas_qty + nike_price * nike_qty + reebok_price * reebok_qty) - quota

theorem alice_above_quota :
  alice_sales 45 60 35 6 8 9 1000 = 65 := by
  sorry

end NUMINAMATH_CALUDE_alice_above_quota_l724_72498


namespace NUMINAMATH_CALUDE_card_sum_theorem_l724_72481

theorem card_sum_theorem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end NUMINAMATH_CALUDE_card_sum_theorem_l724_72481


namespace NUMINAMATH_CALUDE_max_value_implies_a_l724_72414

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * (x - 2)^2

theorem max_value_implies_a (a : ℝ) (h1 : a ≠ 0) :
  (∃ (M : ℝ), M = 16/9 ∧ ∀ (x : ℝ), f a x ≤ M) → a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l724_72414


namespace NUMINAMATH_CALUDE_unique_solution_to_exponential_equation_l724_72439

theorem unique_solution_to_exponential_equation :
  ∀ x y z : ℕ, 3^x + 4^y = 5^z ↔ x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_exponential_equation_l724_72439


namespace NUMINAMATH_CALUDE_otherSideHeadsProbabilityIsCorrect_l724_72472

/-- Represents the three types of coins -/
inductive Coin
  | Normal
  | DoubleHeads
  | DoubleTails

/-- Represents the possible outcomes of a coin flip -/
inductive FlipResult
  | Heads
  | Tails

/-- The probability of selecting each coin -/
def coinProbability : Coin → ℚ
  | Coin.Normal => 1/3
  | Coin.DoubleHeads => 1/3
  | Coin.DoubleTails => 1/3

/-- The probability of getting heads given a specific coin -/
def headsGivenCoin : Coin → ℚ
  | Coin.Normal => 1/2
  | Coin.DoubleHeads => 1
  | Coin.DoubleTails => 0

/-- The probability that the other side is heads given that heads was observed -/
def otherSideHeadsProbability : ℚ := by sorry

theorem otherSideHeadsProbabilityIsCorrect :
  otherSideHeadsProbability = 2/3 := by sorry

end NUMINAMATH_CALUDE_otherSideHeadsProbabilityIsCorrect_l724_72472


namespace NUMINAMATH_CALUDE_min_value_theorem_l724_72486

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + 4*b^2 + 9*c^2 = 4*b + 12*c - 2) :
  1/a + 2/b + 3/c ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l724_72486


namespace NUMINAMATH_CALUDE_vector_computation_l724_72400

theorem vector_computation :
  4 • !![3, -9] - 3 • !![2, -8] + 2 • !![1, -6] = !![8, -24] := by
  sorry

end NUMINAMATH_CALUDE_vector_computation_l724_72400


namespace NUMINAMATH_CALUDE_sum_of_roots_is_eight_l724_72420

theorem sum_of_roots_is_eight : 
  ∃ (N₁ N₂ : ℝ), N₁ * (N₁ - 8) = 4 ∧ N₂ * (N₂ - 8) = 4 ∧ N₁ + N₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_eight_l724_72420


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l724_72443

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l724_72443


namespace NUMINAMATH_CALUDE_lcm_28_72_l724_72462

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end NUMINAMATH_CALUDE_lcm_28_72_l724_72462


namespace NUMINAMATH_CALUDE_circle_M_and_common_chord_l724_72475

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 8 = 0

-- Define the circle N
def circle_N (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 25

-- Define the line on which the center of M lies
def center_line (x y : ℝ) : Prop :=
  x - y = 0

-- Theorem statement
theorem circle_M_and_common_chord :
  -- Circle M passes through (0,-2) and (4,0)
  circle_M 0 (-2) ∧ circle_M 4 0 ∧
  -- The center of M lies on the line x-y=0
  ∃ (cx cy : ℝ), center_line cx cy ∧ 
    ∀ (x y : ℝ), circle_M x y ↔ (x - cx)^2 + (y - cy)^2 = cx^2 + cy^2 + 8 →
  -- 1. The equation of circle M is x^2 + y^2 - 2x - 2y - 8 = 0
  (∀ (x y : ℝ), circle_M x y ↔ x^2 + y^2 - 2*x - 2*y - 8 = 0) ∧
  -- 2. The length of the common chord between M and N is 2√5
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_M x1 y1 ∧ circle_N x1 y1 ∧
    circle_M x2 y2 ∧ circle_N x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_circle_M_and_common_chord_l724_72475


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l724_72419

theorem arithmetic_simplification :
  2 - (-3) - 6 - (-8) - 10 - (-12) = 9 := by sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l724_72419


namespace NUMINAMATH_CALUDE_negation_equivalence_l724_72435

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 1 < 0) ↔ (∀ x : ℝ, x^2 - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l724_72435


namespace NUMINAMATH_CALUDE_rectangle_area_irrational_l724_72432

-- Define a rectangle with rational length and irrational width
structure Rectangle where
  length : ℚ
  width : ℝ
  width_irrational : Irrational width

-- Define the area of the rectangle
def area (rect : Rectangle) : ℝ := (rect.length : ℝ) * rect.width

-- Theorem statement
theorem rectangle_area_irrational (rect : Rectangle) : Irrational (area rect) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_irrational_l724_72432


namespace NUMINAMATH_CALUDE_gas_bill_calculation_l724_72494

/-- Represents the household bills and payments -/
structure HouseholdBills where
  electricity : ℕ
  water : ℕ
  internet : ℕ
  gas : ℕ
  gasPaidFraction : ℚ
  gasAdditionalPayment : ℕ
  remainingPayment : ℕ

/-- Theorem stating that given the household bill conditions, the gas bill is $120 -/
theorem gas_bill_calculation (bills : HouseholdBills) 
  (h1 : bills.electricity = 60)
  (h2 : bills.water = 40)
  (h3 : bills.internet = 25)
  (h4 : bills.gasPaidFraction = 3/4)
  (h5 : bills.gasAdditionalPayment = 5)
  (h6 : bills.remainingPayment = 30)
  (h7 : bills.water / 2 + (bills.internet - 4 * 5) + (bills.gas * (1 - bills.gasPaidFraction) - bills.gasAdditionalPayment) = bills.remainingPayment) :
  bills.gas = 120 := by
  sorry

#check gas_bill_calculation

end NUMINAMATH_CALUDE_gas_bill_calculation_l724_72494


namespace NUMINAMATH_CALUDE_pythagorean_triple_check_l724_72493

def isPythagoreanTriple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem pythagorean_triple_check :
  ¬(isPythagoreanTriple 2 3 4) ∧
  (isPythagoreanTriple 3 4 5) ∧
  (isPythagoreanTriple 6 8 10) ∧
  (isPythagoreanTriple 5 12 13) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_check_l724_72493


namespace NUMINAMATH_CALUDE_simplify_expression_l724_72406

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) :
  (y - 1) * x⁻¹ - y = -((y * x - y + 1) / x) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l724_72406


namespace NUMINAMATH_CALUDE_gasoline_quantity_reduction_l724_72408

/-- Proves that a 25% price increase and 10% budget increase results in a 12% reduction in quantity --/
theorem gasoline_quantity_reduction 
  (P : ℝ) -- Original price of gasoline
  (Q : ℝ) -- Original quantity of gasoline
  (h1 : P > 0) -- Assumption: Price is positive
  (h2 : Q > 0) -- Assumption: Quantity is positive
  : 
  let new_price := 1.25 * P -- 25% price increase
  let new_budget := 1.10 * (P * Q) -- 10% budget increase
  let new_quantity := new_budget / new_price -- New quantity calculation
  (1 - new_quantity / Q) * 100 = 12 -- Percentage reduction in quantity
  := by sorry

end NUMINAMATH_CALUDE_gasoline_quantity_reduction_l724_72408


namespace NUMINAMATH_CALUDE_quartic_roots_l724_72452

/-- The value of N in the quartic equation --/
def N : ℝ := 10^10

/-- The quartic function --/
def f (x : ℝ) : ℝ := x^4 - (2*N + 1)*x^2 - x + N^2 + N - 1

/-- The first approximate root --/
def root1 : ℝ := 99999.9984

/-- The second approximate root --/
def root2 : ℝ := 100000.0016

/-- Theorem stating that the quartic equation has two approximate roots --/
theorem quartic_roots : 
  ∃ (r1 r2 : ℝ), 
    (abs (r1 - root1) < 0.00005) ∧ 
    (abs (r2 - root2) < 0.00005) ∧ 
    f r1 = 0 ∧ 
    f r2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quartic_roots_l724_72452


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l724_72445

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a - 1) * x^2

-- Define the property of f being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Main theorem
theorem solution_set_of_inequality (a : ℝ) :
  (is_odd_function (f a)) →
  {x : ℝ | f a (a * x) > f a (a - x)} = {x : ℝ | x > 1/2} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l724_72445


namespace NUMINAMATH_CALUDE_a_5_equals_31_l724_72471

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_5_equals_31 (a : ℕ → ℝ) :
  geometric_sequence (λ n => 1 + a n) →
  (∀ n : ℕ, (1 + a (n + 1)) = 2 * (1 + a n)) →
  a 1 = 1 →
  a 5 = 31 := by
sorry

end NUMINAMATH_CALUDE_a_5_equals_31_l724_72471


namespace NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l724_72427

/-- Proves that Joel will be 27 years old when his dad is twice his age, given their current ages. -/
theorem joel_age_when_dad_twice_as_old (joel_current_age : ℕ) (dad_current_age : ℕ) : 
  joel_current_age = 5 → dad_current_age = 32 → 
  ∃ (years_passed : ℕ), 
    joel_current_age + years_passed = 27 ∧ 
    dad_current_age + years_passed = 2 * (joel_current_age + years_passed) := by
  sorry

end NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l724_72427


namespace NUMINAMATH_CALUDE_parabola_point_coordinate_l724_72417

/-- The x-coordinate of a point on the parabola y^2 = 6x that is twice as far from the focus as from the y-axis -/
theorem parabola_point_coordinate :
  ∀ (x y : ℝ),
  y^2 = 6*x →
  (x + 3/2)^2 + y^2 = 4 * x^2 →
  x = 3/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_coordinate_l724_72417


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l724_72447

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 < 4) ↔ (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l724_72447


namespace NUMINAMATH_CALUDE_garbage_collection_total_l724_72490

/-- The total amount of garbage collected by two groups, where one group collected 387 pounds and the other collected 39 pounds less. -/
theorem garbage_collection_total : 
  let lizzie_group := 387
  let other_group := lizzie_group - 39
  lizzie_group + other_group = 735 := by
  sorry

end NUMINAMATH_CALUDE_garbage_collection_total_l724_72490


namespace NUMINAMATH_CALUDE_weaker_correlation_as_r_approaches_zero_l724_72491

/-- The correlation coefficient type -/
def CorrelationCoefficient := { r : ℝ // -1 < r ∧ r < 1 }

/-- A measure of correlation strength -/
def correlationStrength (r : CorrelationCoefficient) : ℝ := |r.val|

/-- Theorem: As the absolute value of the correlation coefficient approaches 0, 
    the correlation between two variables becomes weaker -/
theorem weaker_correlation_as_r_approaches_zero 
  (r : CorrelationCoefficient) : 
  ∀ ε > 0, ∃ δ > 0, ∀ r' : CorrelationCoefficient, 
    correlationStrength r' < δ → correlationStrength r' < ε :=
sorry

end NUMINAMATH_CALUDE_weaker_correlation_as_r_approaches_zero_l724_72491


namespace NUMINAMATH_CALUDE_arun_weight_upper_bound_l724_72476

-- Define the weight range according to Arun's opinion
def arun_lower_bound : ℝ := 66
def arun_upper_bound : ℝ := 72

-- Define the weight range according to Arun's brother's opinion
def brother_lower_bound : ℝ := 60
def brother_upper_bound : ℝ := 70

-- Define the average weight
def average_weight : ℝ := 68

-- Define mother's upper bound (to be proven)
def mother_upper_bound : ℝ := 70

-- Theorem statement
theorem arun_weight_upper_bound :
  ∀ w : ℝ,
  (w > arun_lower_bound ∧ w < arun_upper_bound) →
  (w > brother_lower_bound ∧ w < brother_upper_bound) →
  (w ≤ mother_upper_bound) →
  (∃ w_min w_max : ℝ, 
    w_min > max arun_lower_bound brother_lower_bound ∧
    w_max < min arun_upper_bound brother_upper_bound ∧
    (w_min + w_max) / 2 = average_weight) →
  mother_upper_bound = 70 := by
sorry

end NUMINAMATH_CALUDE_arun_weight_upper_bound_l724_72476


namespace NUMINAMATH_CALUDE_finite_decimal_fractions_l724_72496

/-- A fraction a/b can be expressed as a finite decimal if and only if
    b in its simplest form is composed of only the prime factors 2 and 5 -/
def is_finite_decimal (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), b = 2^x * 5^y

/-- The set of natural numbers n for which both 1/n and 1/(n+1) are finite decimals -/
def S : Set ℕ := {n : ℕ | is_finite_decimal 1 n ∧ is_finite_decimal 1 (n+1)}

theorem finite_decimal_fractions : S = {1, 4} := by sorry

end NUMINAMATH_CALUDE_finite_decimal_fractions_l724_72496


namespace NUMINAMATH_CALUDE_color_tv_price_l724_72449

theorem color_tv_price (x : ℝ) : 
  (1 + 0.4) * x * 0.8 - x = 144 → x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_color_tv_price_l724_72449


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_max_value_at_zero_l724_72478

theorem max_value_of_sum_of_roots (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (x + 15) + Real.sqrt (17 - x) + Real.sqrt x ≤ Real.sqrt 15 + Real.sqrt 17 :=
by sorry

theorem max_value_at_zero :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 17 ∧
  Real.sqrt (x + 15) + Real.sqrt (17 - x) + Real.sqrt x = Real.sqrt 15 + Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_max_value_at_zero_l724_72478


namespace NUMINAMATH_CALUDE_grade_assignment_count_l724_72473

/-- The number of ways to assign grades to students. -/
def assignGrades (numStudents : ℕ) (numGrades : ℕ) : ℕ :=
  numGrades ^ numStudents

/-- Theorem: The number of ways to assign 4 different grades to 15 students is 4^15. -/
theorem grade_assignment_count :
  assignGrades 15 4 = 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l724_72473


namespace NUMINAMATH_CALUDE_second_volume_pages_l724_72446

/-- Calculates the number of digits used to number pages up to n --/
def digits_used (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + (n - 9) * 2
  else 189 + (n - 99) * 3

/-- Represents the properties of the two volumes --/
structure TwoVolumes :=
  (first : ℕ)
  (second : ℕ)
  (total_digits : ℕ)
  (page_difference : ℕ)

/-- The main theorem about the number of pages in the second volume --/
theorem second_volume_pages (v : TwoVolumes) 
  (h1 : v.total_digits = 888)
  (h2 : v.second = v.first + v.page_difference)
  (h3 : v.page_difference = 8)
  (h4 : digits_used v.first + digits_used v.second = v.total_digits) :
  v.second = 170 := by
  sorry

#check second_volume_pages

end NUMINAMATH_CALUDE_second_volume_pages_l724_72446


namespace NUMINAMATH_CALUDE_cube_root_square_l724_72458

theorem cube_root_square (y : ℝ) : (y + 5) ^ (1/3 : ℝ) = 3 → (y + 5)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_square_l724_72458


namespace NUMINAMATH_CALUDE_laser_path_distance_correct_l724_72441

/-- The total distance traveled by a laser beam with specified bounces -/
def laser_path_distance : ℝ := 12

/-- Starting point of the laser -/
def start_point : ℝ × ℝ := (4, 6)

/-- Final point of the laser -/
def end_point : ℝ × ℝ := (8, 6)

/-- Theorem stating that the laser path distance is correct -/
theorem laser_path_distance_correct :
  let path := laser_path_distance
  let start := start_point
  let end_ := end_point
  (path = ‖(start.1 + end_.1, start.2 - end_.2)‖) ∧
  (path > 0) ∧
  (start.1 > 0) ∧
  (start.2 > 0) ∧
  (end_.1 > 0) ∧
  (end_.2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_laser_path_distance_correct_l724_72441


namespace NUMINAMATH_CALUDE_initial_cards_proof_l724_72483

/-- The number of baseball cards Nell initially had -/
def initial_cards : ℕ := 455

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := 301

/-- The number of cards Nell has left -/
def cards_left : ℕ := 154

/-- Theorem stating that the initial number of cards is equal to the sum of cards given away and cards left -/
theorem initial_cards_proof : initial_cards = cards_given + cards_left := by
  sorry

end NUMINAMATH_CALUDE_initial_cards_proof_l724_72483


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l724_72479

theorem quadratic_inequality_solution_set (x : ℝ) :
  {x : ℝ | x^2 - 3*x + 2 ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l724_72479


namespace NUMINAMATH_CALUDE_circle_tangents_l724_72433

-- Define the circles
def circle_C (m : ℝ) (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8 - m
def circle_D (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 1

-- Define the property of having three common tangents
def has_three_common_tangents (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (circle_C m x1 y1 ∧ circle_D x1 y1) ∧
    (circle_C m x2 y2 ∧ circle_D x2 y2) ∧
    (circle_C m x3 y3 ∧ circle_D x3 y3) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3)

-- Theorem statement
theorem circle_tangents (m : ℝ) :
  has_three_common_tangents m → m = -8 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangents_l724_72433


namespace NUMINAMATH_CALUDE_correct_answer_points_l724_72413

/-- Represents the scoring system for a math competition --/
structure ScoringSystem where
  total_problems : ℕ
  wang_score : ℤ
  zhang_score : ℤ
  correct_points : ℕ
  incorrect_points : ℕ

/-- Theorem stating that the given scoring system results in 25 points for correct answers --/
theorem correct_answer_points (s : ScoringSystem) : 
  s.total_problems = 20 ∧ 
  s.wang_score = 328 ∧ 
  s.zhang_score = 27 ∧ 
  s.correct_points ≥ 10 ∧ s.correct_points ≤ 99 ∧
  s.incorrect_points ≥ 10 ∧ s.incorrect_points ≤ 99 →
  s.correct_points = 25 := by
  sorry


end NUMINAMATH_CALUDE_correct_answer_points_l724_72413


namespace NUMINAMATH_CALUDE_min_sum_a_b_l724_72425

theorem min_sum_a_b (a b : ℕ+) (h : 45 * a + b = 2021) : 
  (∀ (x y : ℕ+), 45 * x + y = 2021 → a + b ≤ x + y) ∧ a + b = 85 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l724_72425


namespace NUMINAMATH_CALUDE_third_number_in_sum_l724_72464

theorem third_number_in_sum (a b c : ℝ) (h1 : a = 3.15) (h2 : b = 0.014) (h3 : a + b + c = 3.622) : c = 0.458 := by
  sorry

end NUMINAMATH_CALUDE_third_number_in_sum_l724_72464
