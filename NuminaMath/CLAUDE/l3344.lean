import Mathlib

namespace NUMINAMATH_CALUDE_garden_dimensions_l3344_334420

/-- Represents a rectangular garden with a surrounding walkway -/
structure GardenWithWalkway where
  garden_width : ℝ
  walkway_width : ℝ

/-- The combined area of the garden and walkway is 432 square meters -/
axiom total_area (g : GardenWithWalkway) : 
  (g.garden_width + 2 * g.walkway_width) * (3 * g.garden_width + 2 * g.walkway_width) = 432

/-- The area of the walkway alone is 108 square meters -/
axiom walkway_area (g : GardenWithWalkway) : 
  (g.garden_width + 2 * g.walkway_width) * (3 * g.garden_width + 2 * g.walkway_width) - 
  3 * g.garden_width * g.garden_width = 108

/-- The dimensions of the garden are 6√3 and 18√3 meters -/
theorem garden_dimensions (g : GardenWithWalkway) : 
  g.garden_width = 6 * Real.sqrt 3 ∧ 3 * g.garden_width = 18 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_garden_dimensions_l3344_334420


namespace NUMINAMATH_CALUDE_rebecca_egg_groups_l3344_334444

/-- Given a total number of eggs and the number of eggs per group, 
    calculate the number of groups that can be created. -/
def calculate_groups (total_eggs : ℕ) (eggs_per_group : ℕ) : ℕ :=
  total_eggs / eggs_per_group

/-- Theorem stating that with 15 eggs and 5 eggs per group, 
    the number of groups is 3. -/
theorem rebecca_egg_groups : 
  calculate_groups 15 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_egg_groups_l3344_334444


namespace NUMINAMATH_CALUDE_arcsin_sqrt3_div2_l3344_334472

theorem arcsin_sqrt3_div2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by sorry

end NUMINAMATH_CALUDE_arcsin_sqrt3_div2_l3344_334472


namespace NUMINAMATH_CALUDE_binary_arithmetic_l3344_334413

/-- Addition and subtraction of binary numbers --/
theorem binary_arithmetic : 
  let a := 0b1101
  let b := 0b10
  let c := 0b101
  let d := 0b11
  let result := 0b1011
  (a + b + c) - d = result :=
by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_l3344_334413


namespace NUMINAMATH_CALUDE_inequality_proof_l3344_334426

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ (a + b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3344_334426


namespace NUMINAMATH_CALUDE_coin_collection_average_l3344_334492

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℕ → ℝ
  | 0 => a₁
  | k + 1 => arithmetic_sequence a₁ d n k + d

theorem coin_collection_average :
  let a₁ : ℝ := 5
  let d : ℝ := 6
  let n : ℕ := 7
  let seq := arithmetic_sequence a₁ d n
  (seq 0 + seq (n - 1)) / 2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_coin_collection_average_l3344_334492


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3344_334469

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The original number to be represented -/
def original_number : ℝ := 43050000

/-- The scientific notation representation -/
def scientific_repr : ScientificNotation :=
  { coefficient := 4.305,
    exponent := 7,
    h1 := by sorry }

/-- Theorem stating that the original number equals its scientific notation representation -/
theorem original_equals_scientific :
  original_number = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3344_334469


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l3344_334493

/-- Given six rectangles with width 2 and lengths 1, 4, 9, 16, 25, and 36, 
    prove that the sum of their areas is 182. -/
theorem sum_of_rectangle_areas : 
  let width : ℕ := 2
  let lengths : List ℕ := [1, 4, 9, 16, 25, 36]
  let areas : List ℕ := lengths.map (λ l => l * width)
  areas.sum = 182 := by
sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l3344_334493


namespace NUMINAMATH_CALUDE_part_one_part_two_l3344_334409

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

-- Part 1
theorem part_one :
  (Set.compl (B (1/2))) ∩ (A (1/2)) = {x : ℝ | 9/4 ≤ x ∧ x < 5/2} := by sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, (A a ⊆ B a) ↔ (a ≥ -1/2 ∧ a ≤ (3 - Real.sqrt 5) / 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3344_334409


namespace NUMINAMATH_CALUDE_fold_paper_sum_l3344_334423

/-- The fold line equation --/
def fold_line (x y : ℝ) : Prop := y = 2 * x - 4

/-- The relation between (8,4) and (m,n) --/
def point_relation (m n : ℝ) : Prop := 2 * n - 8 = -m + 8

/-- The theorem stating that m + n = 32/3 --/
theorem fold_paper_sum (m n : ℝ) 
  (h1 : fold_line ((1 + 5) / 2) ((3 + 1) / 2))
  (h2 : fold_line ((8 + m) / 2) ((4 + n) / 2))
  (h3 : point_relation m n) :
  m + n = 32 / 3 := by sorry

end NUMINAMATH_CALUDE_fold_paper_sum_l3344_334423


namespace NUMINAMATH_CALUDE_total_points_after_perfect_games_l3344_334411

/-- A perfect score in a game -/
def perfect_score : ℕ := 21

/-- The number of perfect games played -/
def games_played : ℕ := 3

/-- Theorem: The total points after 3 perfect games is 63 -/
theorem total_points_after_perfect_games : 
  perfect_score * games_played = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_points_after_perfect_games_l3344_334411


namespace NUMINAMATH_CALUDE_cos_difference_value_l3344_334438

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.cos A + Real.cos B = 1/2) 
  (h2 : Real.sin A + Real.sin B = 3/2) : 
  Real.cos (A - B) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_value_l3344_334438


namespace NUMINAMATH_CALUDE_historicalFictionNewReleasesFractionIs12_47_l3344_334419

/-- Represents a bookstore inventory --/
structure Inventory where
  total : ℕ
  historicalFiction : ℕ
  historicalFictionNewReleases : ℕ
  otherNewReleases : ℕ

/-- Conditions of the bookstore inventory --/
def validInventory (i : Inventory) : Prop :=
  i.historicalFiction = (30 * i.total) / 100 ∧
  i.historicalFictionNewReleases = (40 * i.historicalFiction) / 100 ∧
  i.otherNewReleases = (50 * (i.total - i.historicalFiction)) / 100

/-- The fraction of new releases that are historical fiction --/
def historicalFictionNewReleasesFraction (i : Inventory) : ℚ :=
  i.historicalFictionNewReleases / (i.historicalFictionNewReleases + i.otherNewReleases)

/-- Theorem stating the fraction of new releases that are historical fiction --/
theorem historicalFictionNewReleasesFractionIs12_47 (i : Inventory) 
  (h : validInventory i) : historicalFictionNewReleasesFraction i = 12 / 47 := by
  sorry

end NUMINAMATH_CALUDE_historicalFictionNewReleasesFractionIs12_47_l3344_334419


namespace NUMINAMATH_CALUDE_base_difference_equals_59_l3344_334464

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def base_6_number : List Nat := [5, 2, 3]
def base_5_number : List Nat := [1, 3, 2]

theorem base_difference_equals_59 :
  to_base_10 base_6_number 6 - to_base_10 base_5_number 5 = 59 := by
  sorry

end NUMINAMATH_CALUDE_base_difference_equals_59_l3344_334464


namespace NUMINAMATH_CALUDE_first_team_speed_calculation_l3344_334440

/-- The speed of the first team in miles per hour -/
def first_team_speed : ℝ := 20

/-- The speed of the second team in miles per hour -/
def second_team_speed : ℝ := 30

/-- The radio range in miles -/
def radio_range : ℝ := 125

/-- The time until radio contact is lost in hours -/
def time_until_lost_contact : ℝ := 2.5

theorem first_team_speed_calculation :
  first_team_speed = (radio_range / time_until_lost_contact) - second_team_speed := by
  sorry

#check first_team_speed_calculation

end NUMINAMATH_CALUDE_first_team_speed_calculation_l3344_334440


namespace NUMINAMATH_CALUDE_total_amount_calculation_l3344_334412

theorem total_amount_calculation (x y z : ℝ) : 
  x > 0 → 
  y = 0.45 * x → 
  z = 0.30 * x → 
  y = 36 → 
  x + y + z = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l3344_334412


namespace NUMINAMATH_CALUDE_evaluate_power_l3344_334476

-- Define the problem
theorem evaluate_power : (81 : ℝ) ^ (11/4) = 177147 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_power_l3344_334476


namespace NUMINAMATH_CALUDE_currency_notes_count_l3344_334451

theorem currency_notes_count 
  (total_amount : ℕ) 
  (denomination_1 : ℕ) 
  (denomination_2 : ℕ) 
  (count_denomination_2 : ℕ) : 
  total_amount = 5000 ∧ 
  denomination_1 = 95 ∧ 
  denomination_2 = 45 ∧ 
  count_denomination_2 = 71 → 
  ∃ (count_denomination_1 : ℕ), 
    count_denomination_1 * denomination_1 + count_denomination_2 * denomination_2 = total_amount ∧ 
    count_denomination_1 + count_denomination_2 = 90 :=
by sorry

end NUMINAMATH_CALUDE_currency_notes_count_l3344_334451


namespace NUMINAMATH_CALUDE_total_remaining_candle_life_l3344_334482

/-- Calculates the total remaining candle life in a house given the number of candles and their remaining life percentages in different rooms. -/
theorem total_remaining_candle_life
  (bedroom_candles : ℕ)
  (bedroom_life : ℚ)
  (living_room_candles : ℕ)
  (living_room_life : ℚ)
  (hallway_candles : ℕ)
  (hallway_life : ℚ)
  (study_room_life : ℚ)
  (h1 : bedroom_candles = 20)
  (h2 : living_room_candles = bedroom_candles / 2)
  (h3 : hallway_candles = 20)
  (h4 : bedroom_life = 60 / 100)
  (h5 : living_room_life = 80 / 100)
  (h6 : hallway_life = 50 / 100)
  (h7 : study_room_life = 70 / 100) :
  let study_room_candles := bedroom_candles + living_room_candles + 5
  (bedroom_candles : ℚ) * bedroom_life +
  (living_room_candles : ℚ) * living_room_life +
  (hallway_candles : ℚ) * hallway_life +
  (study_room_candles : ℚ) * study_room_life = 54.5 :=
sorry

end NUMINAMATH_CALUDE_total_remaining_candle_life_l3344_334482


namespace NUMINAMATH_CALUDE_mock_exam_is_systematic_sampling_l3344_334450

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Cluster

/-- Represents an examination room --/
structure ExamRoom where
  seats : Fin 30 → Nat
  selected_seat : Nat

/-- Represents the mock exam setup --/
structure MockExam where
  rooms : Fin 80 → ExamRoom
  selection_method : SamplingMethod

/-- The mock exam setup as described in the problem --/
def mock_exam : MockExam :=
  { rooms := λ _ => { seats := λ _ => Nat.succ (Nat.zero), selected_seat := 15 },
    selection_method := SamplingMethod.Systematic }

/-- Theorem stating that the sampling method used in the mock exam is systematic sampling --/
theorem mock_exam_is_systematic_sampling :
  mock_exam.selection_method = SamplingMethod.Systematic :=
by sorry

end NUMINAMATH_CALUDE_mock_exam_is_systematic_sampling_l3344_334450


namespace NUMINAMATH_CALUDE_sum_of_real_roots_of_quartic_l3344_334456

theorem sum_of_real_roots_of_quartic (x : ℝ) :
  let f : ℝ → ℝ := λ x => x^4 - 6*x^2 - 2*x - 1
  ∃ (r₁ r₂ : ℝ), (∀ r : ℝ, f r = 0 → r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_real_roots_of_quartic_l3344_334456


namespace NUMINAMATH_CALUDE_valid_range_for_square_root_fraction_l3344_334401

theorem valid_range_for_square_root_fraction (x : ℝ) :
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 1)) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_valid_range_for_square_root_fraction_l3344_334401


namespace NUMINAMATH_CALUDE_probability_ratio_l3344_334422

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def num_numbers : ℕ := 10

/-- The number of slips for each number from 1 to 5 -/
def slips_per_low_number : ℕ := 5

/-- The number of slips for each number from 6 to 10 -/
def slips_per_high_number : ℕ := 3

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability that all four drawn slips bear the same number (only possible for numbers 1 to 5) -/
def r : ℚ := (slips_per_low_number.choose drawn_slips * 5 : ℚ) / total_slips.choose drawn_slips

/-- The probability that two slips bear a number c (1 to 5) and two slips bear a number d ≠ c (6 to 10) -/
def s : ℚ := (5 * 5 * slips_per_low_number.choose 2 * slips_per_high_number.choose 2 : ℚ) / total_slips.choose drawn_slips

theorem probability_ratio : s / r = 30 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l3344_334422


namespace NUMINAMATH_CALUDE_B_2_2_l3344_334471

def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m+1, 0 => B m 2
| m+1, n+1 => B m (B (m+1) n)

theorem B_2_2 : B 2 2 = 8 := by sorry

end NUMINAMATH_CALUDE_B_2_2_l3344_334471


namespace NUMINAMATH_CALUDE_max_value_with_constraint_l3344_334466

theorem max_value_with_constraint (x y z : ℝ) (h : 4 * x^2 + y^2 + 16 * z^2 = 1) :
  7 * x + 2 * y + 8 * z ≤ 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_with_constraint_l3344_334466


namespace NUMINAMATH_CALUDE_cristine_lemons_l3344_334479

theorem cristine_lemons (initial_lemons : ℕ) (given_away_fraction : ℚ) (exchanged_fraction : ℚ) : 
  initial_lemons = 12 →
  given_away_fraction = 1/4 →
  exchanged_fraction = 1/3 →
  (initial_lemons - initial_lemons * given_away_fraction) * (1 - exchanged_fraction) = 6 := by
sorry

end NUMINAMATH_CALUDE_cristine_lemons_l3344_334479


namespace NUMINAMATH_CALUDE_triangle_area_is_25_over_3_l3344_334431

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The area of a triangle given three lines that form its sides -/
def triangleArea (l1 l2 l3 : Line) : ℝ :=
  sorry

/-- The three lines that form the triangle -/
def line1 : Line := { slope := 2, intercept := 4 }
def line2 : Line := { slope := -1, intercept := 3 }
def line3 : Line := { slope := 0, intercept := 0 }

theorem triangle_area_is_25_over_3 :
  triangleArea line1 line2 line3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_25_over_3_l3344_334431


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_l3344_334416

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line type -/
structure Line where
  passes_through : ℝ × ℝ → Prop

/-- Intersection point of a line and a parabola -/
structure IntersectionPoint where
  point : ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_intersection_distance 
  (p : Parabola)
  (l : Line)
  (A B : IntersectionPoint)
  (h1 : p.equation = fun x y => y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : l.passes_through p.focus)
  (h4 : l.passes_through A.point ∧ l.passes_through B.point)
  (h5 : distance A.point p.focus = 4)
  : distance B.point p.focus = 4/3 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_l3344_334416


namespace NUMINAMATH_CALUDE_population_average_age_l3344_334487

/-- Given a population with females and males, calculate the average age -/
theorem population_average_age
  (female_ratio male_ratio : ℕ)
  (female_avg_age male_avg_age : ℝ)
  (h_ratio : female_ratio = 11 ∧ male_ratio = 10)
  (h_female_age : female_avg_age = 34)
  (h_male_age : male_avg_age = 32) :
  let total_people := female_ratio + male_ratio
  let total_age_sum := female_ratio * female_avg_age + male_ratio * male_avg_age
  total_age_sum / total_people = 33 + 1 / 21 :=
by sorry

end NUMINAMATH_CALUDE_population_average_age_l3344_334487


namespace NUMINAMATH_CALUDE_library_sunday_visitors_l3344_334454

/-- Calculates the average number of visitors on Sundays in a library -/
theorem library_sunday_visitors
  (total_days : ℕ) 
  (non_sunday_visitors : ℕ) 
  (overall_average : ℕ) 
  (h1 : total_days = 30)
  (h2 : non_sunday_visitors = 240)
  (h3 : overall_average = 285) :
  let sundays : ℕ := total_days / 7 + 1
  let non_sundays : ℕ := total_days - sundays
  let sunday_visitors : ℕ := (overall_average * total_days - non_sunday_visitors * non_sundays) / sundays
  sunday_visitors = 510 := by
sorry

end NUMINAMATH_CALUDE_library_sunday_visitors_l3344_334454


namespace NUMINAMATH_CALUDE_greenSpaceAfterThreeYears_l3344_334406

/-- Calculates the green space after a given number of years with a fixed annual increase rate -/
def greenSpaceAfterYears (initialSpace : ℝ) (annualIncrease : ℝ) (years : ℕ) : ℝ :=
  initialSpace * (1 + annualIncrease) ^ years

/-- Theorem stating that the green space after 3 years with initial 1000 acres and 10% annual increase is 1331 acres -/
theorem greenSpaceAfterThreeYears :
  greenSpaceAfterYears 1000 0.1 3 = 1331 := by sorry

end NUMINAMATH_CALUDE_greenSpaceAfterThreeYears_l3344_334406


namespace NUMINAMATH_CALUDE_consecutive_square_roots_l3344_334403

theorem consecutive_square_roots (x : ℝ) (n : ℕ) :
  (∃ (m : ℕ), n = m^2 ∧ x = Real.sqrt n) →
  Real.sqrt (n + 1) = Real.sqrt (x^2 + 1) := by
sorry

end NUMINAMATH_CALUDE_consecutive_square_roots_l3344_334403


namespace NUMINAMATH_CALUDE_heesu_has_greatest_sum_l3344_334468

def sora_numbers : Fin 2 → ℕ
| 0 => 4
| 1 => 6

def heesu_numbers : Fin 2 → ℕ
| 0 => 7
| 1 => 5

def jiyeon_numbers : Fin 2 → ℕ
| 0 => 3
| 1 => 8

def sum_numbers (numbers : Fin 2 → ℕ) : ℕ :=
  (numbers 0) + (numbers 1)

theorem heesu_has_greatest_sum :
  sum_numbers heesu_numbers > sum_numbers sora_numbers ∧
  sum_numbers heesu_numbers > sum_numbers jiyeon_numbers :=
by sorry

end NUMINAMATH_CALUDE_heesu_has_greatest_sum_l3344_334468


namespace NUMINAMATH_CALUDE_ordered_triples_solution_l3344_334495

theorem ordered_triples_solution :
  ∀ a b c : ℝ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (⌊a⌋ * b * c = 3 ∧ a * ⌊b⌋ * c = 4 ∧ a * b * ⌊c⌋ = 5) →
  ((a = Real.sqrt 30 / 3 ∧ b = Real.sqrt 30 / 4 ∧ c = 2 * Real.sqrt 30 / 5) ∨
   (a = Real.sqrt 30 / 3 ∧ b = Real.sqrt 30 / 2 ∧ c = Real.sqrt 30 / 5)) :=
by sorry

end NUMINAMATH_CALUDE_ordered_triples_solution_l3344_334495


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3344_334484

theorem negation_of_existence_proposition :
  (¬ ∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔
  (∀ c : ℝ, c > 0 → ¬ ∃ x : ℝ, x^2 - x + c = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3344_334484


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3344_334439

theorem linear_equation_solution (m : ℝ) : 
  (1 : ℝ) * m - 3 = 3 → m = 6 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3344_334439


namespace NUMINAMATH_CALUDE_expand_expression_l3344_334497

theorem expand_expression (x y : ℝ) : (2*x + 3) * (5*y + 7) = 10*x*y + 14*x + 15*y + 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3344_334497


namespace NUMINAMATH_CALUDE_consecutive_even_count_l3344_334453

def is_consecutive_even (a b : ℕ) : Prop := b = a + 2

def sum_consecutive_even (start : ℕ) (count : ℕ) : ℕ :=
  (count * (2 * start + count - 1))

theorem consecutive_even_count :
  ∃ (count : ℕ), 
    sum_consecutive_even 80 count = 246 ∧
    count = 3 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_count_l3344_334453


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3344_334473

theorem cubic_equation_root (a b : ℚ) :
  (∃ x : ℂ, x^3 + a*x^2 + b*x + 15 = 0 ∧ x = 2 + Real.sqrt 5) →
  b = 29 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3344_334473


namespace NUMINAMATH_CALUDE_barn_painted_area_l3344_334455

/-- Calculates the total area to be painted in a barn with given dimensions and conditions -/
def total_painted_area (length width height : ℝ) (window_side : ℝ) (num_windows : ℕ) : ℝ :=
  let long_wall_area := length * height
  let wide_wall_area := width * height
  let ceiling_area := length * width
  let window_area := window_side * window_side * num_windows
  let total_wall_area := 2 * (2 * long_wall_area + 2 * wide_wall_area - window_area)
  total_wall_area + ceiling_area

/-- The total area to be painted in the barn is 796 square yards -/
theorem barn_painted_area :
  total_painted_area 12 15 6 2 2 = 796 := by
  sorry

end NUMINAMATH_CALUDE_barn_painted_area_l3344_334455


namespace NUMINAMATH_CALUDE_fraction_bounds_l3344_334449

theorem fraction_bounds (n : ℕ+) : 1/2 ≤ (n : ℚ) / (n + 1) ∧ (n : ℚ) / (n + 1) < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_bounds_l3344_334449


namespace NUMINAMATH_CALUDE_williams_tips_l3344_334474

/-- Williams works at a resort for 7 months. Let A be the average monthly tips for 6 of these months.
In August, he made 8 times the average of the other months. -/
theorem williams_tips (A : ℚ) : 
  let august_tips := 8 * A
  let total_tips := 15 * A
  august_tips / total_tips = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_williams_tips_l3344_334474


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l3344_334445

theorem simplify_product_of_square_roots (y : ℝ) (hy : y > 0) :
  Real.sqrt (50 * y^3) * Real.sqrt (18 * y) * Real.sqrt (98 * y^5) = 210 * y^4 * Real.sqrt (2 * y) :=
by sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l3344_334445


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3344_334475

theorem complex_number_quadrant (z : ℂ) (m : ℝ) 
  (h1 : z * Complex.I = Complex.I + m)
  (h2 : z.im = 1) : 
  0 < z.re :=
sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3344_334475


namespace NUMINAMATH_CALUDE_samson_sandwich_difference_l3344_334480

/-- The number of sandwiches Samson ate for lunch on Monday -/
def monday_lunch : ℕ := 3

/-- The number of sandwiches Samson ate for dinner on Monday -/
def monday_dinner : ℕ := 2 * monday_lunch

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_breakfast : ℕ := 1

/-- The total number of sandwiches Samson ate on Monday -/
def monday_total : ℕ := monday_lunch + monday_dinner

/-- The difference between the number of sandwiches Samson ate on Monday and Tuesday -/
def sandwich_difference : ℕ := monday_total - tuesday_breakfast

theorem samson_sandwich_difference : sandwich_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_samson_sandwich_difference_l3344_334480


namespace NUMINAMATH_CALUDE_function_composition_ratio_l3344_334405

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 2 * x - 1

theorem function_composition_ratio :
  f (g (f 3)) / g (f (g 3)) = 79 / 37 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l3344_334405


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l3344_334433

/-- The maximum marks for an exam -/
def maximum_marks : ℝ := sorry

/-- The passing mark as a percentage of the maximum marks -/
def passing_percentage : ℝ := 0.45

/-- The marks obtained by the student -/
def student_marks : ℝ := 150

/-- The number of marks by which the student failed -/
def failing_margin : ℝ := 30

theorem exam_maximum_marks : 
  (passing_percentage * maximum_marks = student_marks + failing_margin) → 
  maximum_marks = 400 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l3344_334433


namespace NUMINAMATH_CALUDE_mikes_work_days_l3344_334490

theorem mikes_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days : ℕ) : 
  hours_per_day = 3 →
  total_hours = 15 →
  days * hours_per_day = total_hours →
  days = 5 := by
sorry

end NUMINAMATH_CALUDE_mikes_work_days_l3344_334490


namespace NUMINAMATH_CALUDE_fraction_addition_l3344_334470

theorem fraction_addition : (2 : ℚ) / 5 + (1 : ℚ) / 3 = (11 : ℚ) / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3344_334470


namespace NUMINAMATH_CALUDE_intersection_condition_minimum_condition_l3344_334488

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x^3 - x^2 else a * x * Real.exp x

-- Theorem for the range of m
theorem intersection_condition (a : ℝ) (h : a > 0) :
  ∀ m : ℝ, (∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ f a x = m) ↔ (0 ≤ m ∧ m ≤ 4) ∨ m = -4/27 :=
sorry

-- Theorem for the range of a
theorem minimum_condition :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ -a) ↔ a ≥ 4/27 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_minimum_condition_l3344_334488


namespace NUMINAMATH_CALUDE_triangle_area_at_most_half_parallelogram_l3344_334442

/-- A parallelogram in a 2D plane -/
structure Parallelogram :=
  (P Q R S : ℝ × ℝ)

/-- A triangle in a 2D plane -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Calculate the area of a parallelogram -/
def area_parallelogram (p : Parallelogram) : ℝ :=
  sorry

/-- Calculate the area of a triangle -/
def area_triangle (t : Triangle) : ℝ :=
  sorry

/-- Check if a triangle is inside a parallelogram -/
def is_inside (t : Triangle) (p : Parallelogram) : Prop :=
  sorry

/-- Theorem: The area of a triangle inside a parallelogram is at most half the area of the parallelogram -/
theorem triangle_area_at_most_half_parallelogram (p : Parallelogram) (t : Triangle) 
  (h : is_inside t p) : area_triangle t ≤ (1/2) * area_parallelogram p :=
  sorry

end NUMINAMATH_CALUDE_triangle_area_at_most_half_parallelogram_l3344_334442


namespace NUMINAMATH_CALUDE_point_on_y_axis_l3344_334491

/-- A point P with coordinates (m+2, 2m-4) that lies on the y-axis has coordinates (0, -8). -/
theorem point_on_y_axis (m : ℝ) :
  (m + 2 = 0) → (m + 2, 2 * m - 4) = (0, -8) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l3344_334491


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l3344_334481

theorem discount_percentage_calculation (marked_price : ℝ) (h1 : marked_price > 0) : 
  let cost_price := 0.64 * marked_price
  let gain := 0.375 * cost_price
  let selling_price := cost_price + gain
  let discount := marked_price - selling_price
  (discount / marked_price) * 100 = 12 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l3344_334481


namespace NUMINAMATH_CALUDE_solution_mixture_percentage_l3344_334483

/-- Proves that in a mixture of solutions X and Y, where X is 40% chemical A and Y is 50% chemical A,
    if the final mixture is 47% chemical A, then the percentage of solution X in the mixture is 30%. -/
theorem solution_mixture_percentage (x y : ℝ) :
  x + y = 100 →
  0.40 * x + 0.50 * y = 47 →
  x = 30 := by sorry

end NUMINAMATH_CALUDE_solution_mixture_percentage_l3344_334483


namespace NUMINAMATH_CALUDE_intersection_condition_l3344_334436

-- Define the line l
def line (k x y : ℝ) : Prop := y + k*x + 2 = 0

-- Define the curve C in polar coordinates
def curve (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the curve C in Cartesian coordinates
def curve_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 2*x

-- Theorem stating the condition for intersection
theorem intersection_condition (k : ℝ) :
  (∃ x y : ℝ, line k x y ∧ curve_cartesian x y) → k ≤ -3/4 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l3344_334436


namespace NUMINAMATH_CALUDE_some_number_value_l3344_334428

theorem some_number_value (x y n : ℝ) 
  (h1 : x / (2 * y) = 3 / n) 
  (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) : 
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3344_334428


namespace NUMINAMATH_CALUDE_pictures_picked_out_l3344_334429

def total_pictures : ℕ := 10
def jim_bought : ℕ := 3
def probability : ℚ := 7/15

theorem pictures_picked_out :
  ∃ n : ℕ, n > 0 ∧ n < total_pictures ∧
  (Nat.choose (total_pictures - jim_bought) n : ℚ) / (Nat.choose total_pictures n : ℚ) = probability ∧
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_pictures_picked_out_l3344_334429


namespace NUMINAMATH_CALUDE_figurine_cost_l3344_334465

def televisions : ℕ := 5
def television_cost : ℕ := 50
def figurines : ℕ := 10
def total_spent : ℕ := 260

theorem figurine_cost :
  (total_spent - televisions * television_cost) / figurines = 1 :=
sorry

end NUMINAMATH_CALUDE_figurine_cost_l3344_334465


namespace NUMINAMATH_CALUDE_equation_root_implies_m_value_l3344_334448

theorem equation_root_implies_m_value (x m : ℝ) :
  x > 0 →
  (x - 1) / (x - 5) = m * x / (10 - 2 * x) →
  m = -8/5 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_implies_m_value_l3344_334448


namespace NUMINAMATH_CALUDE_seashells_found_l3344_334461

/-- The number of seashells found by Joan and Jessica -/
theorem seashells_found (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
  sorry

end NUMINAMATH_CALUDE_seashells_found_l3344_334461


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l3344_334418

/-- The radius of a sphere tangent to a truncated cone -/
theorem sphere_radius_in_truncated_cone (r₁ r₂ : ℝ) (hr₁ : r₁ = 25) (hr₂ : r₂ = 5) :
  let h := Real.sqrt ((r₁ - r₂)^2 + (r₁ + r₂)^2)
  (h / 2 : ℝ) = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l3344_334418


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3344_334496

theorem repeating_decimal_sum : 
  (∃ (x y z : ℚ), 
    (1000 * x - x = 123) ∧ 
    (10000 * y - y = 4567) ∧ 
    (100 * z - z = 89) ∧ 
    (x + y + z = 14786 / 9999)) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3344_334496


namespace NUMINAMATH_CALUDE_solve_for_y_l3344_334498

theorem solve_for_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) (h4 : x / y = 81) : 
  y = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l3344_334498


namespace NUMINAMATH_CALUDE_difference_C₁_C₂_l3344_334430

/-- Triangle ABC with given angle measures and altitude from C --/
structure TriangleABC where
  A : ℝ
  B : ℝ
  C : ℝ
  C₁ : ℝ
  C₂ : ℝ
  angleA_eq : A = 30
  angleB_eq : B = 70
  sum_angles : A + B + C = 180
  C_split : C = C₁ + C₂
  right_angle_AC₁ : A + C₁ + 90 = 180
  right_angle_BC₂ : B + C₂ + 90 = 180

/-- Theorem: In the given triangle, C₁ - C₂ = 40° --/
theorem difference_C₁_C₂ (t : TriangleABC) : t.C₁ - t.C₂ = 40 := by
  sorry

end NUMINAMATH_CALUDE_difference_C₁_C₂_l3344_334430


namespace NUMINAMATH_CALUDE_four_dogs_food_consumption_l3344_334441

/-- The total daily food consumption of four dogs -/
def total_dog_food_consumption (dog1 dog2 dog3 dog4 : ℚ) : ℚ :=
  dog1 + dog2 + dog3 + dog4

/-- Theorem stating the total daily food consumption of four specific dogs -/
theorem four_dogs_food_consumption :
  total_dog_food_consumption (1/8) (1/4) (3/8) (1/2) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_four_dogs_food_consumption_l3344_334441


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l3344_334489

theorem perfect_square_polynomial (x a : ℝ) :
  (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) + a^4 = (x^2 + 5 * a * x + 5 * a^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l3344_334489


namespace NUMINAMATH_CALUDE_mark_total_eggs_l3344_334485

/-- The number of people sharing the eggs -/
def num_people : ℕ := 4

/-- The number of eggs each person gets when distributed equally -/
def eggs_per_person : ℕ := 6

/-- The total number of eggs Mark has -/
def total_eggs : ℕ := num_people * eggs_per_person

theorem mark_total_eggs : total_eggs = 24 := by
  sorry

end NUMINAMATH_CALUDE_mark_total_eggs_l3344_334485


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_equations_l3344_334477

/-- Definition of an ellipse with given properties -/
def Ellipse (e : ℝ) (d : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (f₁ f₂ : ℝ × ℝ), 
    f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    (p.1 - f₁.1)^2 + p.2^2 + (p.1 - f₂.1)^2 + p.2^2 = d^2 ∧
    (f₁.1 - f₂.1)^2 = (e * d)^2}

/-- Definition of a hyperbola with given properties -/
def Hyperbola (c : ℝ) (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (f₁ f₂ : ℝ × ℝ),
    f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    (f₁.1 - f₂.1)^2 = 4 * c^2 ∧
    (p.2 = k * p.1 → p.1^2 * (1 + k^2) = c^2 * (1 + k^2)^2)}

/-- Main theorem statement -/
theorem ellipse_hyperbola_equations :
  ∀ (x y : ℝ),
    (x, y) ∈ Ellipse (1/2) 8 ↔ x^2/16 + y^2/12 = 1 ∧
    (x, y) ∈ Hyperbola 2 (Real.sqrt 3) ↔ x^2 - y^2/3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_equations_l3344_334477


namespace NUMINAMATH_CALUDE_parabola_area_l3344_334486

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the roots of the parabola
def root1 : ℝ := 1
def root2 : ℝ := 3

-- State the theorem
theorem parabola_area : 
  (∫ (x : ℝ) in root1..root2, -f x) = 4/3 := by sorry

end NUMINAMATH_CALUDE_parabola_area_l3344_334486


namespace NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l3344_334457

/-- Given 100 pounds of cucumbers with initial 99% water composition by weight,
    prove that after water evaporation resulting in 95% water composition,
    the new weight is 20 pounds. -/
theorem cucumber_weight_after_evaporation
  (initial_weight : ℝ)
  (initial_water_percentage : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_weight = 100)
  (h2 : initial_water_percentage = 0.99)
  (h3 : final_water_percentage = 0.95) :
  let solid_weight := initial_weight * (1 - initial_water_percentage)
  let final_weight := solid_weight / (1 - final_water_percentage)
  final_weight = 20 :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l3344_334457


namespace NUMINAMATH_CALUDE_competition_scores_l3344_334447

def student_scores : List ℝ := [80, 84, 86, 90]

theorem competition_scores (fifth_score : ℝ) 
  (h1 : (fifth_score :: student_scores).length = 5)
  (h2 : (fifth_score :: student_scores).sum / 5 = 87) :
  fifth_score = 95 ∧ 
  let all_scores := fifth_score :: student_scores
  (all_scores.map (λ x => (x - 87)^2)).sum / 5 = 26.4 := by
  sorry

end NUMINAMATH_CALUDE_competition_scores_l3344_334447


namespace NUMINAMATH_CALUDE_bill_sunday_miles_l3344_334446

/-- Represents the number of miles run by Bill and Julia over two days --/
structure RunningMiles where
  billSaturday : ℕ
  billSunday : ℕ
  juliaSunday : ℕ

/-- The conditions of the running problem --/
def runningProblem (r : RunningMiles) : Prop :=
  r.billSunday = r.billSaturday + 4 ∧
  r.juliaSunday = 2 * r.billSunday ∧
  r.billSaturday + r.billSunday + r.juliaSunday = 36

theorem bill_sunday_miles (r : RunningMiles) :
  runningProblem r → r.billSunday = 10 := by
  sorry

end NUMINAMATH_CALUDE_bill_sunday_miles_l3344_334446


namespace NUMINAMATH_CALUDE_set_B_proof_l3344_334458

def U : Finset Nat := {1,2,3,4,5,6,7,8}

theorem set_B_proof (A B : Finset Nat) 
  (h1 : A ∩ (U \ B) = {1,3})
  (h2 : U \ (A ∪ B) = {2,4}) :
  B = {5,6,7,8} := by
sorry

end NUMINAMATH_CALUDE_set_B_proof_l3344_334458


namespace NUMINAMATH_CALUDE_f_positive_solution_set_m_upper_bound_l3344_334494

def f (x : ℝ) := |x - 2| - |2*x + 1|

theorem f_positive_solution_set :
  {x : ℝ | f x > 0} = Set.Ioo (-3) (1/3) :=
sorry

theorem m_upper_bound (m : ℝ) :
  (∃ x₀ : ℝ, f x₀ > 2*m + 1) → m < 3/4 :=
sorry

end NUMINAMATH_CALUDE_f_positive_solution_set_m_upper_bound_l3344_334494


namespace NUMINAMATH_CALUDE_total_fish_count_l3344_334437

/-- The number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- The number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish per multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def white_ducks : ℕ := 3

/-- The number of black ducks -/
def black_ducks : ℕ := 7

/-- The number of multicolor ducks -/
def multicolor_ducks : ℕ := 6

/-- The total number of fish in the lake -/
def total_fish : ℕ := fish_per_white_duck * white_ducks + 
                      fish_per_black_duck * black_ducks + 
                      fish_per_multicolor_duck * multicolor_ducks

theorem total_fish_count : total_fish = 157 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l3344_334437


namespace NUMINAMATH_CALUDE_bisection_uses_all_structures_l3344_334424

/-- Represents the different algorithm structures -/
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents the bisection method for a specific equation -/
structure BisectionMethod where
  equation : ℝ → ℝ
  approximateRoot : ℝ → ℝ → ℝ → ℝ

/-- The bisection method for x^2 - 10 = 0 -/
def bisectionForXSquaredMinus10 : BisectionMethod :=
  { equation := λ x => x^2 - 10,
    approximateRoot := sorry }

/-- Checks if a given algorithm structure is used in the bisection method -/
def usesStructure (b : BisectionMethod) (s : AlgorithmStructure) : Prop :=
  match s with
  | AlgorithmStructure.Sequential => sorry
  | AlgorithmStructure.Conditional => sorry
  | AlgorithmStructure.Loop => sorry

theorem bisection_uses_all_structures :
  ∀ s : AlgorithmStructure, usesStructure bisectionForXSquaredMinus10 s := by
  sorry

end NUMINAMATH_CALUDE_bisection_uses_all_structures_l3344_334424


namespace NUMINAMATH_CALUDE_pool_length_is_ten_l3344_334459

/-- Represents a rectangular swimming pool with a surrounding deck -/
structure PoolWithDeck where
  poolLength : ℝ
  poolWidth : ℝ
  deckWidth : ℝ

/-- Calculates the total area of the pool and deck -/
def totalArea (p : PoolWithDeck) : ℝ :=
  (p.poolLength + 2 * p.deckWidth) * (p.poolWidth + 2 * p.deckWidth)

/-- Theorem: The length of the pool is 10 feet given the specified conditions -/
theorem pool_length_is_ten :
  ∃ (p : PoolWithDeck),
    p.poolWidth = 12 ∧
    p.deckWidth = 4 ∧
    totalArea p = 360 ∧
    p.poolLength = 10 := by
  sorry

end NUMINAMATH_CALUDE_pool_length_is_ten_l3344_334459


namespace NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l3344_334414

theorem choose_three_cooks_from_ten (n : ℕ) (k : ℕ) : n = 10 ∧ k = 3 → Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l3344_334414


namespace NUMINAMATH_CALUDE_chloe_wins_l3344_334415

/-- Given that the ratio of Chloe's wins to Max's wins is 8:3 and Max won 9 times,
    prove that Chloe won 24 times. -/
theorem chloe_wins (ratio_chloe : ℕ) (ratio_max : ℕ) (max_wins : ℕ) 
    (h1 : ratio_chloe = 8)
    (h2 : ratio_max = 3)
    (h3 : max_wins = 9) : 
  (ratio_chloe * max_wins) / ratio_max = 24 := by
  sorry

#check chloe_wins

end NUMINAMATH_CALUDE_chloe_wins_l3344_334415


namespace NUMINAMATH_CALUDE_student_not_asked_probability_l3344_334499

/-- The probability of a student not being asked in either of two consecutive lessons -/
theorem student_not_asked_probability
  (total_students : ℕ)
  (selected_students : ℕ)
  (previous_lesson_pool : ℕ)
  (h1 : total_students = 30)
  (h2 : selected_students = 3)
  (h3 : previous_lesson_pool = 10)
  : ℚ :=
  11 / 30

/-- The proof of the theorem -/
lemma student_not_asked_probability_proof :
  student_not_asked_probability 30 3 10 rfl rfl rfl = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_student_not_asked_probability_l3344_334499


namespace NUMINAMATH_CALUDE_product_last_two_digits_perfect_square_even_l3344_334435

theorem product_last_two_digits_perfect_square_even (n : ℤ) : 
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ (n^2 % 100 = 10 * a + b) ∧ Even (a * b) :=
sorry

end NUMINAMATH_CALUDE_product_last_two_digits_perfect_square_even_l3344_334435


namespace NUMINAMATH_CALUDE_pot_temperature_celsius_l3344_334410

/-- Converts temperature from Fahrenheit to Celsius -/
def fahrenheit_to_celsius (f : ℚ) : ℚ :=
  (f - 32) * (5/9)

/-- The temperature of the pot of water in Fahrenheit -/
def pot_temperature_f : ℚ := 122

theorem pot_temperature_celsius :
  fahrenheit_to_celsius pot_temperature_f = 50 := by
  sorry

end NUMINAMATH_CALUDE_pot_temperature_celsius_l3344_334410


namespace NUMINAMATH_CALUDE_solution_pairs_l3344_334460

theorem solution_pairs (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (3 * y - Real.sqrt (y / x) - 6 * Real.sqrt (x * y) + 2 = 0 ∧
   x^2 + 81 * x^2 * y^4 = 2 * y^2) ↔
  ((x = Real.sqrt (Real.sqrt 31) / 12 ∧ y = Real.sqrt (Real.sqrt 31) / 3) ∨
   (x = 1/3 ∧ y = 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l3344_334460


namespace NUMINAMATH_CALUDE_binomial_distribution_not_equivalent_to_expansion_l3344_334417

-- Define the binomial distribution formula
def binomial_distribution (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Define the general term of binomial expansion
def binomial_expansion_term (n k : ℕ) (a b : ℝ) : ℝ :=
  (n.choose k) * a^k * b^(n-k)

-- Theorem statement
theorem binomial_distribution_not_equivalent_to_expansion :
  ∃ n k : ℕ, ∃ p : ℝ, 
    binomial_distribution n k p ≠ binomial_expansion_term n k p (1-p) :=
sorry

end NUMINAMATH_CALUDE_binomial_distribution_not_equivalent_to_expansion_l3344_334417


namespace NUMINAMATH_CALUDE_points_needed_theorem_l3344_334467

/-- Represents the points scored in each game -/
structure GameScores where
  lastHome : ℕ
  firstAway : ℕ
  secondAway : ℕ
  thirdAway : ℕ

/-- Calculates the points needed in the next game -/
def pointsNeededNextGame (scores : GameScores) : ℕ :=
  4 * scores.lastHome - (scores.lastHome + scores.firstAway + scores.secondAway + scores.thirdAway)

/-- Theorem stating the conditions and the result to be proved -/
theorem points_needed_theorem (scores : GameScores) 
  (h1 : scores.lastHome = 2 * scores.firstAway)
  (h2 : scores.secondAway = scores.firstAway + 18)
  (h3 : scores.thirdAway = scores.secondAway + 2)
  (h4 : scores.lastHome = 62) :
  pointsNeededNextGame scores = 55 := by
  sorry

#eval pointsNeededNextGame ⟨62, 31, 49, 51⟩

end NUMINAMATH_CALUDE_points_needed_theorem_l3344_334467


namespace NUMINAMATH_CALUDE_geometric_sum_first_7_terms_l3344_334434

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_7_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/2
  geometric_sum a r 7 = 127/192 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_first_7_terms_l3344_334434


namespace NUMINAMATH_CALUDE_mike_fred_salary_ratio_l3344_334400

/-- Proves that Mike earned 11 times more money than Fred five months ago -/
theorem mike_fred_salary_ratio :
  ∀ (fred_salary mike_salary_now : ℕ),
    fred_salary = 1000 →
    mike_salary_now = 15400 →
    ∃ (mike_salary_before : ℕ),
      mike_salary_now = (140 * mike_salary_before) / 100 ∧
      mike_salary_before = 11 * fred_salary :=
by sorry

end NUMINAMATH_CALUDE_mike_fred_salary_ratio_l3344_334400


namespace NUMINAMATH_CALUDE_perpendicular_vectors_exist_minimum_dot_product_l3344_334425

/-- Given vectors in 2D space -/
def OA : Fin 2 → ℝ := ![5, 1]
def OB : Fin 2 → ℝ := ![1, 7]
def OC : Fin 2 → ℝ := ![4, 2]

/-- Vector OM as a function of t -/
def OM (t : ℝ) : Fin 2 → ℝ := fun i => t * OC i

/-- Vector MA as a function of t -/
def MA (t : ℝ) : Fin 2 → ℝ := fun i => OA i - OM t i

/-- Vector MB as a function of t -/
def MB (t : ℝ) : Fin 2 → ℝ := fun i => OB i - OM t i

/-- Dot product of MA and MB -/
def MA_dot_MB (t : ℝ) : ℝ := (MA t 0) * (MB t 0) + (MA t 1) * (MB t 1)

theorem perpendicular_vectors_exist :
  ∃ t : ℝ, MA_dot_MB t = 0 ∧ t = (5 + Real.sqrt 10) / 5 ∨ t = (5 - Real.sqrt 10) / 5 := by
  sorry

theorem minimum_dot_product :
  ∃ t : ℝ, ∀ s : ℝ, MA_dot_MB t ≤ MA_dot_MB s ∧ OM t = OC := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_exist_minimum_dot_product_l3344_334425


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l3344_334478

theorem vector_dot_product_problem (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b = (-1, 2)) :
  (a + 2 • b) • b = 14 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l3344_334478


namespace NUMINAMATH_CALUDE_extra_interest_proof_l3344_334452

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem extra_interest_proof (principal : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) :
  principal = 15000 →
  rate1 = 0.15 →
  rate2 = 0.12 →
  time = 2 →
  simple_interest principal rate1 time - simple_interest principal rate2 time = 900 := by
  sorry

end NUMINAMATH_CALUDE_extra_interest_proof_l3344_334452


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l3344_334427

theorem candy_sampling_percentage (caught_percent : ℝ) (not_caught_ratio : ℝ) 
  (h1 : caught_percent = 22)
  (h2 : not_caught_ratio = 0.2) : 
  (caught_percent / (1 - not_caught_ratio)) = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l3344_334427


namespace NUMINAMATH_CALUDE_floor_sum_equals_negative_one_l3344_334421

theorem floor_sum_equals_negative_one : ⌊(19.7 : ℝ)⌋ + ⌊(-19.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_equals_negative_one_l3344_334421


namespace NUMINAMATH_CALUDE_total_cars_sold_l3344_334432

def cars_sold_day1 : ℕ := 14
def cars_sold_day2 : ℕ := 16
def cars_sold_day3 : ℕ := 27

theorem total_cars_sold : cars_sold_day1 + cars_sold_day2 + cars_sold_day3 = 57 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_sold_l3344_334432


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2009_n_42_divisible_by_2009_exists_unique_smallest_n_l3344_334408

theorem smallest_n_divisible_by_2009 :
  ∀ n : ℕ, n > 1 → n^2 * (n - 1) % 2009 = 0 → n ≥ 42 :=
by
  sorry

theorem n_42_divisible_by_2009 : 42^2 * (42 - 1) % 2009 = 0 :=
by
  sorry

theorem exists_unique_smallest_n :
  ∃! n : ℕ, n > 1 ∧ n^2 * (n - 1) % 2009 = 0 ∧ ∀ m : ℕ, m > 1 → m^2 * (m - 1) % 2009 = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2009_n_42_divisible_by_2009_exists_unique_smallest_n_l3344_334408


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l3344_334443

def vector_a : Fin 2 → ℝ := ![4, 2]
def vector_b (y : ℝ) : Fin 2 → ℝ := ![6, y]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (∀ i, u i = k * v i)

theorem parallel_vectors_y_value :
  parallel vector_a (vector_b y) → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l3344_334443


namespace NUMINAMATH_CALUDE_infinite_solutions_exist_l3344_334462

theorem infinite_solutions_exist :
  ∃ f : ℕ → ℕ → ℕ × ℕ × ℕ,
    ∀ u v : ℕ, u > 1 → v > 1 →
      let (x, y, z) := f u v
      x^2015 + y^2015 = z^2016 ∧
      x ≠ y ∧ y ≠ z ∧ x ≠ z :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_exist_l3344_334462


namespace NUMINAMATH_CALUDE_altitude_to_base_l3344_334463

/-- Given a triangle ABC with known sides and area, prove the altitude to base AB -/
theorem altitude_to_base (a b c area h : ℝ) : 
  a = 30 → b = 17 → c = 25 → area = 120 → 
  area = (1/2) * a * h → h = 8 := by sorry

end NUMINAMATH_CALUDE_altitude_to_base_l3344_334463


namespace NUMINAMATH_CALUDE_supermarket_profit_analysis_l3344_334404

/-- Represents the daily sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -20 * x + 1600

/-- Represents the daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The cost price per box -/
def cost_price : ℝ := 40

/-- The minimum selling price per box -/
def min_selling_price : ℝ := 45

/-- The sales volume at the minimum selling price -/
def base_sales : ℝ := 700

/-- The decrease in sales for each yuan increase in price -/
def sales_decrease : ℝ := 20

theorem supermarket_profit_analysis :
  (∀ x ≥ min_selling_price, sales_volume x = -20 * x + 1600) ∧
  (∃ x ≥ min_selling_price, daily_profit x = 6000 ∧ x = 50) ∧
  (∃ x ≥ min_selling_price, ∀ y ≥ min_selling_price, daily_profit x ≥ daily_profit y ∧ x = 60 ∧ daily_profit x = 8000) := by
  sorry


end NUMINAMATH_CALUDE_supermarket_profit_analysis_l3344_334404


namespace NUMINAMATH_CALUDE_chess_tournament_score_change_l3344_334402

/-- Represents a chess tournament with 2n players -/
structure ChessTournament (n : ℕ) where
  players : Fin (2 * n)
  score : Fin (2 * n) → ℝ
  score_change : Fin (2 * n) → ℝ

/-- The theorem to be proved -/
theorem chess_tournament_score_change (n : ℕ) (tournament : ChessTournament n) :
  (∀ p, tournament.score_change p ≥ n) →
  (∀ p, tournament.score_change p = n) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_score_change_l3344_334402


namespace NUMINAMATH_CALUDE_quadrilateral_circumscribed_l3344_334407

-- Define the structure for a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the property of being convex
def is_convex (q : Quadrilateral) : Prop := sorry

-- Define the property of a point being inside a quadrilateral
def is_interior_point (P : Point) (q : Quadrilateral) : Prop := sorry

-- Define angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- Define the property of being circumscribed
def is_circumscribed (q : Quadrilateral) : Prop := sorry

theorem quadrilateral_circumscribed 
  (q : Quadrilateral) (P : Point) 
  (h_convex : is_convex q)
  (h_interior : is_interior_point P q)
  (h_angle1 : angle_measure q.A P q.B + angle_measure q.C P q.D = 
              angle_measure q.B P q.C + angle_measure q.D P q.A)
  (h_angle2 : angle_measure P q.A q.D + angle_measure P q.C q.D = 
              angle_measure P q.A q.B + angle_measure P q.C q.B)
  (h_angle3 : angle_measure P q.D q.C + angle_measure P q.B q.C = 
              angle_measure P q.D q.A + angle_measure P q.B q.A) :
  is_circumscribed q :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_circumscribed_l3344_334407
