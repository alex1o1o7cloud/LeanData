import Mathlib

namespace NUMINAMATH_CALUDE_fraction_power_product_l3763_376378

theorem fraction_power_product : (2/3 : ℚ)^2023 * (-3/2 : ℚ)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l3763_376378


namespace NUMINAMATH_CALUDE_robie_cards_l3763_376303

/-- The number of cards in each box -/
def cards_per_box : ℕ := 10

/-- The number of cards not placed in a box -/
def loose_cards : ℕ := 5

/-- The number of boxes Robie gave away -/
def boxes_given_away : ℕ := 2

/-- The number of boxes Robie has left -/
def boxes_left : ℕ := 5

/-- The total number of cards Robie had in the beginning -/
def total_cards : ℕ := (boxes_given_away + boxes_left) * cards_per_box + loose_cards

theorem robie_cards : total_cards = 75 := by
  sorry

end NUMINAMATH_CALUDE_robie_cards_l3763_376303


namespace NUMINAMATH_CALUDE_product_of_successive_numbers_l3763_376308

theorem product_of_successive_numbers : 
  let n : Real := 88.49858755935034
  let product := n * (n + 1)
  ∃ ε > 0, |product - 7913| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_successive_numbers_l3763_376308


namespace NUMINAMATH_CALUDE_clay_pot_flower_cost_difference_clay_pot_flower_cost_difference_proof_l3763_376313

/-- The cost difference between a clay pot and flowers -/
theorem clay_pot_flower_cost_difference : ℝ → ℝ → ℝ → Prop :=
  fun flower_cost clay_pot_cost soil_cost =>
    flower_cost = 9 ∧
    clay_pot_cost > flower_cost ∧
    soil_cost = flower_cost - 2 ∧
    flower_cost + clay_pot_cost + soil_cost = 45 →
    clay_pot_cost - flower_cost = 20

/-- Proof of the clay_pot_flower_cost_difference theorem -/
theorem clay_pot_flower_cost_difference_proof :
  ∃ (flower_cost clay_pot_cost soil_cost : ℝ),
    clay_pot_flower_cost_difference flower_cost clay_pot_cost soil_cost :=
by
  sorry

end NUMINAMATH_CALUDE_clay_pot_flower_cost_difference_clay_pot_flower_cost_difference_proof_l3763_376313


namespace NUMINAMATH_CALUDE_identify_scientists_l3763_376331

/-- Represents the type of scientist: chemist or alchemist -/
inductive ScientistType
| Chemist
| Alchemist

/-- Represents a scientist at the conference -/
structure Scientist where
  id : Nat
  type : ScientistType

/-- Represents the conference of scientists -/
structure Conference where
  scientists : List Scientist
  num_chemists : Nat
  num_alchemists : Nat
  chemists_outnumber_alchemists : num_chemists > num_alchemists

/-- Represents a question asked by the mathematician -/
def Question := Scientist → Scientist → ScientistType

/-- The main theorem to be proved -/
theorem identify_scientists (conf : Conference) :
  ∃ (questions : List Question), questions.length ≤ 2 * conf.scientists.length - 2 ∧
  (∀ s : Scientist, s ∈ conf.scientists → 
    ∃ (determined_type : ScientistType), determined_type = s.type) :=
sorry

end NUMINAMATH_CALUDE_identify_scientists_l3763_376331


namespace NUMINAMATH_CALUDE_sqrt_9801_minus_39_cube_l3763_376345

theorem sqrt_9801_minus_39_cube (a b : ℕ+) :
  (Real.sqrt 9801 - 39 : ℝ) = (Real.sqrt a.val - b.val : ℝ)^3 →
  a.val + b.val = 13 := by
sorry

end NUMINAMATH_CALUDE_sqrt_9801_minus_39_cube_l3763_376345


namespace NUMINAMATH_CALUDE_percentage_literate_inhabitants_l3763_376382

theorem percentage_literate_inhabitants (total_inhabitants : ℕ) 
  (male_percentage : ℚ) (literate_male_percentage : ℚ) (literate_female_percentage : ℚ)
  (h1 : total_inhabitants = 1000)
  (h2 : male_percentage = 60 / 100)
  (h3 : literate_male_percentage = 20 / 100)
  (h4 : literate_female_percentage = 325 / 1000) : 
  (↑(total_inhabitants * (male_percentage * literate_male_percentage * total_inhabitants + 
    (1 - male_percentage) * literate_female_percentage * total_inhabitants)) / 
    (↑total_inhabitants * 1000) : ℚ) = 25 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_literate_inhabitants_l3763_376382


namespace NUMINAMATH_CALUDE_all_points_on_line_l3763_376329

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The set of n points on the plane -/
def PointSet (n : ℕ) := Fin n → Point

/-- The property that any line through two points contains at least one more point -/
def ThreePointProperty (points : PointSet n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j →
    ∃ (l : Line), pointOnLine (points i) l ∧ pointOnLine (points j) l →
      ∃ (m : Fin n), m ≠ i ∧ m ≠ j ∧ pointOnLine (points m) l

/-- The theorem statement -/
theorem all_points_on_line (n : ℕ) (points : PointSet n) 
  (h : ThreePointProperty points) : 
  ∃ (l : Line), ∀ (i : Fin n), pointOnLine (points i) l :=
sorry

end NUMINAMATH_CALUDE_all_points_on_line_l3763_376329


namespace NUMINAMATH_CALUDE_five_divides_cube_iff_five_divides_l3763_376384

theorem five_divides_cube_iff_five_divides (a : ℤ) : 
  (5 : ℤ) ∣ a^3 ↔ (5 : ℤ) ∣ a := by
  sorry

end NUMINAMATH_CALUDE_five_divides_cube_iff_five_divides_l3763_376384


namespace NUMINAMATH_CALUDE_number_of_choices_l3763_376357

-- Define the total number of subjects
def total_subjects : ℕ := 6

-- Define the number of science subjects
def science_subjects : ℕ := 3

-- Define the number of humanities subjects
def humanities_subjects : ℕ := 3

-- Define the number of subjects to be chosen
def subjects_to_choose : ℕ := 3

-- Define the minimum number of science subjects to be chosen
def min_science_subjects : ℕ := 2

-- Theorem statement
theorem number_of_choices :
  (Nat.choose science_subjects 2 * Nat.choose humanities_subjects 1) +
  (Nat.choose science_subjects 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_choices_l3763_376357


namespace NUMINAMATH_CALUDE_hall_of_mirrors_glass_area_l3763_376325

/-- Calculates the total area of glass needed for James' hall of mirrors --/
theorem hall_of_mirrors_glass_area :
  let wall1_length : ℝ := 30
  let wall1_width : ℝ := 12
  let wall2_length : ℝ := 30
  let wall2_width : ℝ := 12
  let wall3_length : ℝ := 20
  let wall3_width : ℝ := 12
  let wall1_area := wall1_length * wall1_width
  let wall2_area := wall2_length * wall2_width
  let wall3_area := wall3_length * wall3_width
  let total_area := wall1_area + wall2_area + wall3_area
  total_area = 960 := by sorry

end NUMINAMATH_CALUDE_hall_of_mirrors_glass_area_l3763_376325


namespace NUMINAMATH_CALUDE_trains_meeting_problem_l3763_376368

/-- Theorem: Two trains meeting problem
    Given two trains starting 450 miles apart and traveling towards each other
    at 50 miles per hour each, the distance traveled by one train when they meet
    is 225 miles. -/
theorem trains_meeting_problem (distance_between_stations : ℝ) 
                                (speed_train_a : ℝ) 
                                (speed_train_b : ℝ) : ℝ :=
  by
  have h1 : distance_between_stations = 450 := by sorry
  have h2 : speed_train_a = 50 := by sorry
  have h3 : speed_train_b = 50 := by sorry
  
  -- Calculate the combined speed of the trains
  let combined_speed := speed_train_a + speed_train_b
  
  -- Calculate the time until the trains meet
  let time_to_meet := distance_between_stations / combined_speed
  
  -- Calculate the distance traveled by Train A
  let distance_traveled_by_a := speed_train_a * time_to_meet
  
  -- Prove that the distance traveled by Train A is 225 miles
  have h4 : distance_traveled_by_a = 225 := by sorry
  
  exact distance_traveled_by_a


end NUMINAMATH_CALUDE_trains_meeting_problem_l3763_376368


namespace NUMINAMATH_CALUDE_no_roots_lost_l3763_376317

theorem no_roots_lost (x : ℝ) : 
  (x^4 + x^3 + x^2 + x + 1 = 0) ↔ (x^2 + x + 1 + 1/x + 1/x^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_roots_lost_l3763_376317


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3763_376332

theorem equation_solution_exists : ∃ x : ℝ, (0.75 : ℝ) ^ x + 2 = 8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3763_376332


namespace NUMINAMATH_CALUDE_birthday_on_sunday_l3763_376350

/-- Represents days of the week -/
inductive Day : Type
  | sunday : Day
  | monday : Day
  | tuesday : Day
  | wednesday : Day
  | thursday : Day
  | friday : Day
  | saturday : Day

/-- Returns the day before the given day -/
def dayBefore (d : Day) : Day :=
  match d with
  | Day.sunday => Day.saturday
  | Day.monday => Day.sunday
  | Day.tuesday => Day.monday
  | Day.wednesday => Day.tuesday
  | Day.thursday => Day.wednesday
  | Day.friday => Day.thursday
  | Day.saturday => Day.friday

/-- Returns the day that is n days after the given day -/
def daysAfter (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => daysAfter (match d with
    | Day.sunday => Day.monday
    | Day.monday => Day.tuesday
    | Day.tuesday => Day.wednesday
    | Day.wednesday => Day.thursday
    | Day.thursday => Day.friday
    | Day.friday => Day.saturday
    | Day.saturday => Day.sunday) m

theorem birthday_on_sunday (today : Day) (birthday : Day) : 
  today = Day.thursday → 
  daysAfter (dayBefore birthday) 2 = dayBefore (dayBefore (dayBefore today)) → 
  birthday = Day.sunday := by
  sorry

end NUMINAMATH_CALUDE_birthday_on_sunday_l3763_376350


namespace NUMINAMATH_CALUDE_magic_square_x_free_l3763_376326

/-- Represents a 3x3 magic square with given entries -/
structure MagicSquare where
  x : ℝ
  sum : ℝ
  top_middle : ℝ
  top_right : ℝ
  middle_left : ℝ
  is_magic : sum = x + top_middle + top_right
           ∧ sum = x + middle_left + (sum - x - middle_left)
           ∧ sum = top_right + (sum - top_right - (sum - x - middle_left))

/-- Theorem stating that x can be any real number in the given magic square -/
theorem magic_square_x_free (m : MagicSquare) (h : m.top_middle = 35 ∧ m.top_right = 58 ∧ m.middle_left = 8 ∧ m.sum = 85) :
  ∀ y : ℝ, ∃ m' : MagicSquare, m'.x = y ∧ m'.top_middle = m.top_middle ∧ m'.top_right = m.top_right ∧ m'.middle_left = m.middle_left ∧ m'.sum = m.sum :=
sorry

end NUMINAMATH_CALUDE_magic_square_x_free_l3763_376326


namespace NUMINAMATH_CALUDE_lcm_of_12_16_15_l3763_376380

theorem lcm_of_12_16_15 : Nat.lcm (Nat.lcm 12 16) 15 = 240 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_16_15_l3763_376380


namespace NUMINAMATH_CALUDE_unique_power_of_two_product_l3763_376359

theorem unique_power_of_two_product (a b : ℕ) :
  (∃ k : ℕ, (a + b^2) * (b + a^2) = 2^k) ↔ (a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_power_of_two_product_l3763_376359


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l3763_376354

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l3763_376354


namespace NUMINAMATH_CALUDE_g_of_three_l3763_376305

theorem g_of_three (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3*x - 2) = 4*x + 1) : g 3 = 23/3 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_l3763_376305


namespace NUMINAMATH_CALUDE_function_min_value_l3763_376386

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem function_min_value 
  (h_max : ∃ (m : ℝ), ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3 ∧ ∃ y ∈ Set.Icc (-2 : ℝ) 2, f y m = 3) :
  ∃ (m : ℝ), ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ -37 ∧ ∃ y ∈ Set.Icc (-2 : ℝ) 2, f y m = -37 :=
sorry

end NUMINAMATH_CALUDE_function_min_value_l3763_376386


namespace NUMINAMATH_CALUDE_x_greater_than_ln_one_plus_x_l3763_376396

theorem x_greater_than_ln_one_plus_x {x : ℝ} (h : x > 0) : x > Real.log (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_ln_one_plus_x_l3763_376396


namespace NUMINAMATH_CALUDE_max_different_digits_is_eight_l3763_376343

/-- A natural number satisfying the divisibility condition -/
def DivisibleNumber (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ Finset.range 10 → d ≠ 0 → (n.digits 10).contains d → n % d = 0

/-- The maximum number of different digits in a DivisibleNumber -/
def MaxDifferentDigits : ℕ := 8

/-- Theorem stating the maximum number of different digits in a DivisibleNumber -/
theorem max_different_digits_is_eight :
  ∃ n : ℕ, DivisibleNumber n ∧ (n.digits 10).card = MaxDifferentDigits ∧
  ∀ m : ℕ, DivisibleNumber m → (m.digits 10).card ≤ MaxDifferentDigits :=
sorry

end NUMINAMATH_CALUDE_max_different_digits_is_eight_l3763_376343


namespace NUMINAMATH_CALUDE_relationship_abc_l3763_376373

theorem relationship_abc : ∃ (a b c : ℝ), 
  a = 2^(2/5) ∧ b = 9^(1/5) ∧ c = 3^(3/4) ∧ a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3763_376373


namespace NUMINAMATH_CALUDE_roundness_of_24300000_l3763_376394

/-- Roundness of a positive integer is the sum of the exponents of its prime factors. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The number we're analyzing -/
def number : ℕ+ := 24300000

theorem roundness_of_24300000 : roundness number = 15 := by sorry

end NUMINAMATH_CALUDE_roundness_of_24300000_l3763_376394


namespace NUMINAMATH_CALUDE_ambika_candles_l3763_376351

theorem ambika_candles (ambika : ℕ) (aniyah : ℕ) : 
  aniyah = 6 * ambika →
  (ambika + aniyah) / 2 = 14 →
  ambika = 4 := by
sorry

end NUMINAMATH_CALUDE_ambika_candles_l3763_376351


namespace NUMINAMATH_CALUDE_determinant_of_cubic_roots_l3763_376398

theorem determinant_of_cubic_roots (p q : ℝ) (a b c : ℂ) : 
  (a^3 + p*a + q = 0) → 
  (b^3 + p*b + q = 0) → 
  (c^3 + p*c + q = 0) → 
  (Complex.abs a ≠ 0) →
  (Complex.abs b ≠ 0) →
  (Complex.abs c ≠ 0) →
  let matrix := !![2 + a^2, 1, 1; 1, 2 + b^2, 1; 1, 1, 2 + c^2]
  Matrix.det matrix = (2*p^2 : ℂ) - 4*q + q^2 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_cubic_roots_l3763_376398


namespace NUMINAMATH_CALUDE_part1_part2_l3763_376340

-- Part 1
theorem part1 (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0)
  (hab : a + b = 10) (hxy : a / x + b / y = 1) (hmin : ∀ x' y', x' > 0 → y' > 0 → a / x' + b / y' = 1 → x' + y' ≥ 18) :
  (a = 2 ∧ b = 8) ∨ (a = 8 ∧ b = 2) := by sorry

-- Part 2
theorem part2 :
  ∃ a : ℝ, a > 0 ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * Real.sqrt (2 * x * y) ≤ a * (x + y)) ∧
  (∀ a' : ℝ, a' > 0 → (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * Real.sqrt (2 * x * y) ≤ a' * (x + y)) → a ≤ a') ∧
  a = 2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3763_376340


namespace NUMINAMATH_CALUDE_square_difference_l3763_376366

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : 
  (x - y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3763_376366


namespace NUMINAMATH_CALUDE_only_statement4_correct_l3763_376349

-- Define the structure of input/output statements
inductive Statement
| Input (vars : List String)
| InputAssign (var : String) (value : Nat)
| Print (expr : String)
| PrintMultiple (values : List Nat)

-- Define the rules for correct statements
def isCorrectInput (s : Statement) : Prop :=
  match s with
  | Statement.Input vars => vars.length > 0
  | Statement.InputAssign _ _ => false
  | _ => false

def isCorrectOutput (s : Statement) : Prop :=
  match s with
  | Statement.Print _ => false
  | Statement.PrintMultiple values => values.length > 0
  | _ => false

def isCorrect (s : Statement) : Prop :=
  isCorrectInput s ∨ isCorrectOutput s

-- Define the given statements
def statement1 : Statement := Statement.Input ["a", "b", "c"]
def statement2 : Statement := Statement.Print "a=1"
def statement3 : Statement := Statement.InputAssign "x" 2
def statement4 : Statement := Statement.PrintMultiple [20, 4]

-- Theorem to prove
theorem only_statement4_correct :
  ¬ isCorrect statement1 ∧
  ¬ isCorrect statement2 ∧
  ¬ isCorrect statement3 ∧
  isCorrect statement4 :=
sorry

end NUMINAMATH_CALUDE_only_statement4_correct_l3763_376349


namespace NUMINAMATH_CALUDE_root_sum_squares_l3763_376374

theorem root_sum_squares (a b c d : ℂ) : 
  (a^4 - 24*a^3 + 50*a^2 - 35*a + 7 = 0) →
  (b^4 - 24*b^3 + 50*b^2 - 35*b + 7 = 0) →
  (c^4 - 24*c^3 + 50*c^2 - 35*c + 7 = 0) →
  (d^4 - 24*d^3 + 50*d^2 - 35*d + 7 = 0) →
  (a+b+c)^2 + (b+c+d)^2 + (c+d+a)^2 + (d+a+b)^2 = 2104 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_squares_l3763_376374


namespace NUMINAMATH_CALUDE_fungi_at_128pm_l3763_376348

/-- The number of fungi at a given time, given an initial population and doubling time -/
def fungiBehavior (initialPopulation : ℕ) (doublingTime : ℕ) (elapsedTime : ℕ) : ℕ :=
  initialPopulation * 2 ^ (elapsedTime / doublingTime)

/-- Theorem stating the number of fungi at 1:28 p.m. given the initial conditions -/
theorem fungi_at_128pm (initialPopulation : ℕ) (doublingTime : ℕ) (elapsedTime : ℕ) :
  initialPopulation = 30 → doublingTime = 4 → elapsedTime = 28 →
  fungiBehavior initialPopulation doublingTime elapsedTime = 3840 := by
  sorry

#check fungi_at_128pm

end NUMINAMATH_CALUDE_fungi_at_128pm_l3763_376348


namespace NUMINAMATH_CALUDE_quadratic_function_nonnegative_constraint_l3763_376376

theorem quadratic_function_nonnegative_constraint (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → x^2 + a*x + 3 - a ≥ 0) → 
  a ∈ Set.Icc (-7) 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_nonnegative_constraint_l3763_376376


namespace NUMINAMATH_CALUDE_balloon_count_l3763_376301

theorem balloon_count (green blue yellow red : ℚ) (total : ℕ) : 
  green = 2/9 →
  blue = 1/3 →
  yellow = 1/4 →
  red = 7/36 →
  green + blue + yellow + red = 1 →
  (yellow * total / 2 : ℚ) = 50 →
  total = 400 := by
sorry

end NUMINAMATH_CALUDE_balloon_count_l3763_376301


namespace NUMINAMATH_CALUDE_sum_of_ages_l3763_376375

theorem sum_of_ages (age1 age2 age3 : ℕ) : 
  age1 = 9 → age2 = 9 → age3 = 11 → age1 + age2 + age3 = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3763_376375


namespace NUMINAMATH_CALUDE_soda_bottles_ordered_l3763_376379

/-- The number of bottles of soda ordered by a store owner in April and May -/
theorem soda_bottles_ordered (april_cases may_cases bottles_per_case : ℕ) 
  (h1 : april_cases = 20)
  (h2 : may_cases = 30)
  (h3 : bottles_per_case = 20) :
  (april_cases + may_cases) * bottles_per_case = 1000 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottles_ordered_l3763_376379


namespace NUMINAMATH_CALUDE_small_rectangle_perimeter_l3763_376322

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the problem setup -/
structure ProblemSetup where
  large_rectangle : Rectangle
  num_vertical_cuts : ℕ
  num_horizontal_cuts : ℕ
  total_cut_length : ℝ

/-- Theorem stating the solution to the problem -/
theorem small_rectangle_perimeter
  (setup : ProblemSetup)
  (h1 : setup.large_rectangle.perimeter = 100)
  (h2 : setup.num_vertical_cuts = 6)
  (h3 : setup.num_horizontal_cuts = 9)
  (h4 : setup.total_cut_length = 405)
  (h5 : (setup.num_vertical_cuts + 1) * (setup.num_horizontal_cuts + 1) = 70) :
  let small_rectangle := Rectangle.mk
    (setup.large_rectangle.width / (setup.num_vertical_cuts + 1))
    (setup.large_rectangle.height / (setup.num_horizontal_cuts + 1))
  small_rectangle.perimeter = 13 := by
  sorry

end NUMINAMATH_CALUDE_small_rectangle_perimeter_l3763_376322


namespace NUMINAMATH_CALUDE_sum_inequality_l3763_376341

theorem sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  let S := a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c)
  0 < S ∧ S < 1 := by sorry

end NUMINAMATH_CALUDE_sum_inequality_l3763_376341


namespace NUMINAMATH_CALUDE_simplify_expression_l3763_376364

theorem simplify_expression (x : ℝ) : 5 * x + 7 * x - 3 * x = 9 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3763_376364


namespace NUMINAMATH_CALUDE_smallest_integer_larger_than_sqrt3_minus_sqrt2_to_6th_l3763_376352

theorem smallest_integer_larger_than_sqrt3_minus_sqrt2_to_6th :
  ∃ n : ℤ, (n = 133 ∧ (∀ m : ℤ, m > (Real.sqrt 3 - Real.sqrt 2)^6 → m ≥ n)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_larger_than_sqrt3_minus_sqrt2_to_6th_l3763_376352


namespace NUMINAMATH_CALUDE_combined_distance_is_122_l3763_376314

-- Define the fuel-to-distance ratios for both cars
def car_A_ratio : Rat := 4 / 7
def car_B_ratio : Rat := 3 / 5

-- Define the amount of fuel used by each car
def car_A_fuel : ℕ := 44
def car_B_fuel : ℕ := 27

-- Function to calculate distance given fuel and ratio
def calculate_distance (fuel : ℕ) (ratio : Rat) : ℚ :=
  (fuel : ℚ) * (ratio.den : ℚ) / (ratio.num : ℚ)

-- Theorem stating the combined distance is 122 miles
theorem combined_distance_is_122 :
  (calculate_distance car_A_fuel car_A_ratio + calculate_distance car_B_fuel car_B_ratio) = 122 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_is_122_l3763_376314


namespace NUMINAMATH_CALUDE_isosceles_triangle_legs_l3763_376383

/-- An isosceles triangle with integer side lengths and perimeter 12 -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ
  perimeter_eq : leg + leg + base = 12
  triangle_inequality : base < leg + leg ∧ leg < base + leg

/-- The possible leg lengths of an isosceles triangle with perimeter 12 -/
def possibleLegLengths : Set ℕ :=
  {n : ℕ | ∃ (t : IsoscelesTriangle), t.leg = n}

theorem isosceles_triangle_legs :
  possibleLegLengths = {4, 5} := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_legs_l3763_376383


namespace NUMINAMATH_CALUDE_sum_of_complex_numbers_l3763_376323

theorem sum_of_complex_numbers :
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := 4 - 7*I
  let z₃ : ℂ := -2 + 3*I
  z₁ + z₂ + z₃ = 5 + I := by sorry

end NUMINAMATH_CALUDE_sum_of_complex_numbers_l3763_376323


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_1_f_extrema_on_interval_2_l3763_376389

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Theorem for the interval [-2, 0]
theorem f_extrema_on_interval_1 :
  (∀ x ∈ Set.Icc (-2) 0, f x ≤ 2) ∧
  (∀ x ∈ Set.Icc (-2) 0, f x ≥ 10) ∧
  (∃ x ∈ Set.Icc (-2) 0, f x = 2) ∧
  (∃ x ∈ Set.Icc (-2) 0, f x = 10) :=
sorry

-- Theorem for the interval [2, 3]
theorem f_extrema_on_interval_2 :
  (∀ x ∈ Set.Icc 2 3, f x ≤ 5) ∧
  (∀ x ∈ Set.Icc 2 3, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 2 3, f x = 5) ∧
  (∃ x ∈ Set.Icc 2 3, f x = 2) :=
sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_1_f_extrema_on_interval_2_l3763_376389


namespace NUMINAMATH_CALUDE_exists_special_function_l3763_376307

theorem exists_special_function :
  ∃ f : ℕ → ℕ,
    (∀ m n : ℕ, m < n → f m < f n) ∧
    f 1 = 2 ∧
    ∀ n : ℕ, f (f n) = f n + n :=
by sorry

end NUMINAMATH_CALUDE_exists_special_function_l3763_376307


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_plus_i_l3763_376361

theorem imaginary_part_of_one_over_one_plus_i :
  let z : ℂ := 1 / (1 + Complex.I)
  Complex.im z = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_plus_i_l3763_376361


namespace NUMINAMATH_CALUDE_inequality_implies_values_l3763_376347

theorem inequality_implies_values (a b : ℤ) 
  (h : ∀ x : ℝ, x ≤ 0 → (a * x + 2) * (x^2 + 2 * b) ≤ 0) : 
  a = 1 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_values_l3763_376347


namespace NUMINAMATH_CALUDE_max_value_m_l3763_376395

theorem max_value_m (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, m / (3 * a + b) - 3 / a - 1 / b ≤ 0) →
  (∃ m : ℝ, m = 16 ∧ ∀ m' : ℝ, m' / (3 * a + b) - 3 / a - 1 / b ≤ 0 → m' ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_max_value_m_l3763_376395


namespace NUMINAMATH_CALUDE_triangle_side_value_l3763_376310

theorem triangle_side_value (m : ℝ) : m > 0 → 
  (3 + 4 > m ∧ 3 + m > 4 ∧ 4 + m > 3) →
  (m = 1 ∨ m = 5 ∨ m = 7 ∨ m = 9) →
  m = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_side_value_l3763_376310


namespace NUMINAMATH_CALUDE_average_of_data_set_l3763_376338

def data_set : List ℤ := [7, 5, -2, 5, 10]

theorem average_of_data_set :
  (data_set.sum : ℚ) / data_set.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_set_l3763_376338


namespace NUMINAMATH_CALUDE_integral_3x_plus_sin_x_l3763_376391

theorem integral_3x_plus_sin_x (x : Real) : 
  ∫ x in (0)..(π/2), (3*x + Real.sin x) = 3*π^2/8 + 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_3x_plus_sin_x_l3763_376391


namespace NUMINAMATH_CALUDE_stella_unpaid_leave_l3763_376355

/-- Calculates the number of months of unpaid leave taken by an employee given their monthly income and actual annual income. -/
def unpaid_leave_months (monthly_income : ℕ) (actual_annual_income : ℕ) : ℕ :=
  12 - actual_annual_income / monthly_income

/-- Proves that given Stella's monthly income of 4919 dollars and her actual annual income of 49190 dollars, the number of months of unpaid leave she took is 2. -/
theorem stella_unpaid_leave :
  unpaid_leave_months 4919 49190 = 2 := by
  sorry

end NUMINAMATH_CALUDE_stella_unpaid_leave_l3763_376355


namespace NUMINAMATH_CALUDE_expression_simplification_l3763_376369

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (x^2 + 1) / (x^2 - 1) - (x - 2) / (x - 1) / ((x - 2) / x) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3763_376369


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3763_376312

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x = -2) → 
  (-1 : ℝ) ∈ {x : ℝ | 3 * x^2 + m * x = -2} → 
  (-2/3 : ℝ) ∈ {x : ℝ | 3 * x^2 + m * x = -2} :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3763_376312


namespace NUMINAMATH_CALUDE_subtraction_of_like_terms_l3763_376306

theorem subtraction_of_like_terms (a b : ℝ) : 5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_like_terms_l3763_376306


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3763_376372

theorem solve_linear_equation (x y a : ℚ) : 
  x = 2 → y = a → 2 * x - 3 * y = 5 → a = -1/3 := by sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3763_376372


namespace NUMINAMATH_CALUDE_gcd_lcm_product_300_l3763_376328

theorem gcd_lcm_product_300 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 300) :
  ∃! (s : Finset ℕ), s.card = 8 ∧ ∀ d, d ∈ s ↔ ∃ (x y : ℕ+), x * y = 300 ∧ Nat.gcd x y = d :=
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_300_l3763_376328


namespace NUMINAMATH_CALUDE_jessica_quarters_l3763_376393

theorem jessica_quarters (initial borrowed current : ℕ) : 
  borrowed = 3 → current = 5 → initial = current + borrowed :=
by sorry

end NUMINAMATH_CALUDE_jessica_quarters_l3763_376393


namespace NUMINAMATH_CALUDE_john_smith_payment_l3763_376339

-- Define the number of cakes
def num_cakes : ℕ := 3

-- Define the cost per cake in cents (to avoid floating-point numbers)
def cost_per_cake : ℕ := 1200

-- Define the number of people sharing the cost
def num_people : ℕ := 2

-- Theorem to prove
theorem john_smith_payment :
  (num_cakes * cost_per_cake) / num_people = 1800 := by
  sorry

end NUMINAMATH_CALUDE_john_smith_payment_l3763_376339


namespace NUMINAMATH_CALUDE_pythagorean_triple_parity_l3763_376370

theorem pythagorean_triple_parity (m n : ℤ) 
  (h_succ : m = n + 1 ∨ n = m + 1)
  (a b c : ℤ) 
  (h_a : a = m^2 - n^2)
  (h_b : b = 2*m*n)
  (h_c : c = m^2 + n^2)
  (h_coprime : ¬(2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c)) :
  Odd c ∧ Even b ∧ Odd a := by
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_parity_l3763_376370


namespace NUMINAMATH_CALUDE_exists_counterexample_1_fraction_inequality_implies_exists_counterexample_3_fraction_inequality_implies_product_l3763_376365

-- Statement 1
theorem exists_counterexample_1 : ∃ (a b c d : ℝ), a > b ∧ c = d ∧ a * c ≤ b * d := by sorry

-- Statement 2
theorem fraction_inequality_implies (a b c : ℝ) (h : c ≠ 0) : a / c^2 < b / c^2 → a < b := by sorry

-- Statement 3
theorem exists_counterexample_3 : ∃ (a b c d : ℝ), a > b ∧ c > d ∧ a - c ≤ b - d := by sorry

-- Statement 4
theorem fraction_inequality_implies_product (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : c / a > d / b → b * c > a * d := by sorry

end NUMINAMATH_CALUDE_exists_counterexample_1_fraction_inequality_implies_exists_counterexample_3_fraction_inequality_implies_product_l3763_376365


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3763_376302

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^2 + 7 * x - 3) - (x^2 + 5 * x - 12) = x^2 + 2 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3763_376302


namespace NUMINAMATH_CALUDE_prob_all_white_value_l3763_376309

/-- The number of small cubes forming the larger cube -/
def num_cubes : ℕ := 8

/-- The probability of a single small cube showing a white face after flipping -/
def prob_white_face : ℚ := 5/6

/-- The probability of all surfaces of the larger cube becoming white after flipping -/
def prob_all_white : ℚ := (prob_white_face ^ num_cubes).num / (prob_white_face ^ num_cubes).den

theorem prob_all_white_value : prob_all_white = 390625/1679616 := by sorry

end NUMINAMATH_CALUDE_prob_all_white_value_l3763_376309


namespace NUMINAMATH_CALUDE_tan_sum_special_case_l3763_376333

theorem tan_sum_special_case :
  let tan55 := Real.tan (55 * π / 180)
  let tan65 := Real.tan (65 * π / 180)
  tan55 + tan65 - Real.sqrt 3 * tan55 * tan65 = -Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tan_sum_special_case_l3763_376333


namespace NUMINAMATH_CALUDE_shopping_trip_tax_percentage_l3763_376324

/-- Calculate the total tax percentage given spending percentages and tax rates -/
theorem shopping_trip_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.45)
  (h2 : food_percent = 0.45)
  (h3 : other_percent = 0.1)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.05)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.1) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.0325 := by
  sorry

#check shopping_trip_tax_percentage

end NUMINAMATH_CALUDE_shopping_trip_tax_percentage_l3763_376324


namespace NUMINAMATH_CALUDE_flour_requirement_undetermined_l3763_376397

/-- Represents the recipe requirements and current state of baking --/
structure BakingScenario where
  sugar_required : ℕ
  sugar_added : ℕ
  flour_added : ℕ

/-- Represents the unknown total flour required by the recipe --/
def total_flour_required : ℕ → Prop := fun _ => True

/-- Theorem stating that the total flour required cannot be determined --/
theorem flour_requirement_undetermined (scenario : BakingScenario) 
  (h1 : scenario.sugar_required = 11)
  (h2 : scenario.sugar_added = 10)
  (h3 : scenario.flour_added = 12) :
  ∀ n : ℕ, total_flour_required n :=
by sorry

end NUMINAMATH_CALUDE_flour_requirement_undetermined_l3763_376397


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l3763_376315

theorem greatest_multiple_of_four (x : ℕ) : 
  x % 4 = 0 → 
  x > 0 → 
  x^3 < 5000 → 
  x ≤ 16 ∧ 
  ∃ y : ℕ, y % 4 = 0 ∧ y > 0 ∧ y^3 < 5000 ∧ y = 16 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l3763_376315


namespace NUMINAMATH_CALUDE_range_of_difference_l3763_376311

theorem range_of_difference (a b : ℝ) (ha : 12 < a ∧ a < 60) (hb : 15 < b ∧ b < 36) :
  -24 < a - b ∧ a - b < 45 := by
  sorry

end NUMINAMATH_CALUDE_range_of_difference_l3763_376311


namespace NUMINAMATH_CALUDE_unique_grid_placement_l3763_376304

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- The sum of adjacent numbers is less than 12 --/
def valid_sum (g : Grid) : Prop :=
  ∀ p1 p2 : Fin 3 × Fin 3, adjacent p1 p2 → g p1.1 p1.2 + g p2.1 p2.2 < 12

/-- The grid contains all numbers from 1 to 9 --/
def contains_all_numbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n.val + 1

/-- The given positions of odd numbers --/
def given_positions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7 ∧ g 0 2 = 9

/-- The theorem to be proved --/
theorem unique_grid_placement :
  ∀ g : Grid,
    valid_sum g →
    contains_all_numbers g →
    given_positions g →
    g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_grid_placement_l3763_376304


namespace NUMINAMATH_CALUDE_reciprocal_of_difference_l3763_376367

-- Define repeating decimals
def repeating_decimal_1 : ℚ := 1/9
def repeating_decimal_6 : ℚ := 2/3

-- State the theorem
theorem reciprocal_of_difference : (repeating_decimal_6 - repeating_decimal_1)⁻¹ = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_difference_l3763_376367


namespace NUMINAMATH_CALUDE_trig_simplification_l3763_376371

theorem trig_simplification (α : Real) (h : α ≠ 0) (h' : α ≠ π / 2) :
  (1 / Real.sin α + 1 / Real.tan α) * (1 - Real.cos α) = Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3763_376371


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l3763_376316

/-- Given that 20 cows eat 20 bags of husk in 20 days, prove that one cow will eat one bag of husk in 20 days. -/
theorem cow_husk_consumption (num_cows : ℕ) (num_bags : ℕ) (num_days : ℕ) 
  (h1 : num_cows = 20) 
  (h2 : num_bags = 20) 
  (h3 : num_days = 20) : 
  (num_days : ℚ) = (num_cows : ℚ) * (num_bags : ℚ) / ((num_cows : ℚ) * (num_bags : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l3763_376316


namespace NUMINAMATH_CALUDE_units_digit_of_47_to_47_l3763_376353

theorem units_digit_of_47_to_47 : (47^47 % 10 = 3) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_47_to_47_l3763_376353


namespace NUMINAMATH_CALUDE_smallest_n_perfect_powers_l3763_376336

theorem smallest_n_perfect_powers : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℕ), 3 * n = x^4) ∧ 
  (∃ (y : ℕ), 2 * n = y^5) ∧ 
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 3 * m = x^4) → 
    (∃ (y : ℕ), 2 * m = y^5) → 
    n ≤ m) ∧
  n = 432 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_powers_l3763_376336


namespace NUMINAMATH_CALUDE_negation_of_forall_inequality_negation_of_specific_inequality_l3763_376388

theorem negation_of_forall_inequality (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, x ≤ 1 → p x) ↔ (∃ x : ℝ, x ≤ 1 ∧ ¬(p x)) := by sorry

theorem negation_of_specific_inequality :
  (¬ ∀ x : ℝ, x ≤ 1 → x^2 - 2*x + 1 ≥ 0) ↔ (∃ x : ℝ, x ≤ 1 ∧ x^2 - 2*x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_inequality_negation_of_specific_inequality_l3763_376388


namespace NUMINAMATH_CALUDE_sean_charles_whistle_difference_l3763_376344

/-- 
Given that Sean has 45 whistles and Charles has 13 whistles, 
prove that Sean has 32 more whistles than Charles.
-/
theorem sean_charles_whistle_difference :
  ∀ (sean_whistles charles_whistles : ℕ),
    sean_whistles = 45 →
    charles_whistles = 13 →
    sean_whistles - charles_whistles = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_sean_charles_whistle_difference_l3763_376344


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l3763_376319

/-- Given a cylinder with original volume of 15 cubic feet, prove that tripling its radius and doubling its height results in a new volume of 270 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 15 → π * (3*r)^2 * (2*h) = 270 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l3763_376319


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l3763_376346

def quadratic_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem roots_of_quadratic_equation :
  let a : ℝ := 1
  let b : ℝ := -7
  let c : ℝ := 12
  (quadratic_equation a b c 3 = 0) ∧ (quadratic_equation a b c 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l3763_376346


namespace NUMINAMATH_CALUDE_sqrt_three_expression_equality_l3763_376334

theorem sqrt_three_expression_equality : 
  (Real.sqrt 3 + 1)^2 - Real.sqrt 12 + 2 * Real.sqrt (1/3) = 4 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_expression_equality_l3763_376334


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l3763_376330

theorem rectangle_area_equals_perimeter (x : ℝ) : 
  let length : ℝ := 4 * x
  let width : ℝ := 2 * x + 6
  length > 0 ∧ width > 0 →
  length * width = 2 * (length + width) →
  x = (-3 + Real.sqrt 33) / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l3763_376330


namespace NUMINAMATH_CALUDE_output_for_input_12_l3763_376337

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 35 then
    step1 + 10
  else
    step1 - 7

theorem output_for_input_12 :
  function_machine 12 = 29 := by sorry

end NUMINAMATH_CALUDE_output_for_input_12_l3763_376337


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3763_376385

theorem quadratic_inequality_solution (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, ax^2 - (a + 1)*x + 1 < 0 ↔ 
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨ 
     (a > 1 ∧ 1/a < x ∧ x < 1))) ∧
  (a = 1 → ∀ x : ℝ, ¬(x^2 - 2*x + 1 < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3763_376385


namespace NUMINAMATH_CALUDE_largest_remaining_number_l3763_376363

/-- Represents the original number sequence as a list of digits -/
def originalSequence : List Nat := sorry

/-- Represents the result after removing 100 digits -/
def resultSequence : List Nat := sorry

/-- The number of digits to remove -/
def digitsToRemove : Nat := 100

/-- Checks if a sequence is a valid subsequence of another sequence -/
def isValidSubsequence (sub seq : List Nat) : Prop := sorry

/-- Checks if a number represented as a list of digits is greater than another -/
def isGreaterThan (a b : List Nat) : Prop := sorry

theorem largest_remaining_number :
  isValidSubsequence resultSequence originalSequence ∧
  resultSequence.length = originalSequence.length - digitsToRemove ∧
  (∀ (other : List Nat), 
    isValidSubsequence other originalSequence → 
    other.length = originalSequence.length - digitsToRemove →
    isGreaterThan resultSequence other ∨ resultSequence = other) :=
sorry

end NUMINAMATH_CALUDE_largest_remaining_number_l3763_376363


namespace NUMINAMATH_CALUDE_anya_initial_seat_l3763_376392

def Friend := Fin 5

structure SeatingArrangement where
  seats : Friend → Fin 5
  bijective : Function.Bijective seats

def move_right (n : Nat) (s : Fin 5) : Fin 5 :=
  ⟨(s.val + n) % 5, by sorry⟩

def move_left (n : Nat) (s : Fin 5) : Fin 5 :=
  ⟨(s.val + 5 - n % 5) % 5, by sorry⟩

def swap (s1 s2 : Fin 5) (s : Fin 5) : Fin 5 :=
  if s = s1 then s2
  else if s = s2 then s1
  else s

theorem anya_initial_seat (initial final : SeatingArrangement) 
  (anya varya galya diana ellya : Friend) :
  initial.seats anya ≠ 1 →
  initial.seats anya ≠ 5 →
  final.seats anya = 1 ∨ final.seats anya = 5 →
  final.seats varya = move_right 1 (initial.seats varya) →
  final.seats galya = move_left 3 (initial.seats galya) →
  final.seats diana = initial.seats ellya →
  final.seats ellya = initial.seats diana →
  initial.seats anya = 3 := by sorry

end NUMINAMATH_CALUDE_anya_initial_seat_l3763_376392


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3763_376318

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x ^ (1/4) = 15 / (8 - x ^ (1/4))) ↔ (x = 625 ∨ x = 81) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3763_376318


namespace NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l3763_376356

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ y - a * x^2
  ∃! p : ℝ × ℝ, f p = 0 ∧ p.1 = 0 ∧ p.2 = 1 / (4 * a) :=
sorry

/-- The focus of the parabola y = 4x^2 is at (0, 1/16) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ y - 4 * x^2
  ∃! p : ℝ × ℝ, f p = 0 ∧ p.1 = 0 ∧ p.2 = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l3763_376356


namespace NUMINAMATH_CALUDE_equation_solution_l3763_376335

theorem equation_solution (x : ℚ) (h : 2 * x + 1 = 8) : 4 * x + 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3763_376335


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3763_376360

open Function Real

theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, x + y + z = 0 → f (x^3) + f y^3 + f z^3 = 3*x*y*z) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3763_376360


namespace NUMINAMATH_CALUDE_sqrt_equation_sum_l3763_376320

theorem sqrt_equation_sum (a t : ℝ) (ha : a > 0) (ht : t > 0) :
  Real.sqrt (6 + a / t) = 6 * Real.sqrt (a / t) → t + a = 41 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_sum_l3763_376320


namespace NUMINAMATH_CALUDE_condition_equivalence_l3763_376327

theorem condition_equivalence (α β : ℝ) :
  (α > β) ↔ (α + Real.sin α * Real.cos β > β + Real.sin β * Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_condition_equivalence_l3763_376327


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l3763_376390

/-- Calculates the total ticket sales for a theater performance --/
theorem theater_ticket_sales 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_attendance : ℕ) 
  (child_attendance : ℕ) : 
  adult_price = 8 → 
  child_price = 1 → 
  total_attendance = 22 → 
  child_attendance = 18 → 
  (total_attendance - child_attendance) * adult_price + child_attendance * child_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l3763_376390


namespace NUMINAMATH_CALUDE_problem_1_l3763_376321

theorem problem_1 : (-8) + 10 - 2 + (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_problem_1_l3763_376321


namespace NUMINAMATH_CALUDE_covered_number_value_l3763_376300

theorem covered_number_value : ∃ a : ℝ, 
  (∀ x : ℝ, (x - a) / 2 = x + 3 ↔ x = -7) ∧ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_covered_number_value_l3763_376300


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l3763_376362

theorem existence_of_special_sequence (n : ℕ) : 
  ∃ (a : ℕ → ℕ), 
    (∀ i j, i < j → j ≤ n → a i > a j) ∧ 
    (∀ i, i < n → a i ∣ (a (i + 1))^2) ∧
    (∀ i j, i ≠ j → i ≤ n → j ≤ n → ¬(a i ∣ a j)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l3763_376362


namespace NUMINAMATH_CALUDE_least_four_digit_solution_l3763_376399

theorem least_four_digit_solution (x : ℕ) : x = 1163 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 15 [ZMOD 20] ∧
     3 * y + 10 ≡ 19 [ZMOD 14] ∧
     -3 * y + 4 ≡ 2 * y [ZMOD 35] ∧
     y + 1 ≡ 0 [ZMOD 11]) →
    x ≤ y) ∧
  (5 * x ≡ 15 [ZMOD 20]) ∧
  (3 * x + 10 ≡ 19 [ZMOD 14]) ∧
  (-3 * x + 4 ≡ 2 * x [ZMOD 35]) ∧
  (x + 1 ≡ 0 [ZMOD 11]) := by
  sorry

end NUMINAMATH_CALUDE_least_four_digit_solution_l3763_376399


namespace NUMINAMATH_CALUDE_loss_equates_to_five_balls_l3763_376342

/-- Given the sale of 20 balls at Rs. 720 with a loss equal to the cost price of some balls,
    and the cost price of a ball being Rs. 48, prove that the loss equates to 5 balls. -/
theorem loss_equates_to_five_balls 
  (total_balls : ℕ) 
  (selling_price : ℕ) 
  (cost_price_per_ball : ℕ) 
  (h1 : total_balls = 20)
  (h2 : selling_price = 720)
  (h3 : cost_price_per_ball = 48) :
  (total_balls * cost_price_per_ball - selling_price) / cost_price_per_ball = 5 :=
by sorry

end NUMINAMATH_CALUDE_loss_equates_to_five_balls_l3763_376342


namespace NUMINAMATH_CALUDE_probability_y_div_x_geq_4_probability_equals_one_eighth_l3763_376377

/-- The probability that y/x ≥ 4 when x and y are randomly selected from [0,2] -/
theorem probability_y_div_x_geq_4 : Real :=
  let total_area := 4
  let favorable_area := 1/2
  favorable_area / total_area

/-- The probability is equal to 1/8 -/
theorem probability_equals_one_eighth : probability_y_div_x_geq_4 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_y_div_x_geq_4_probability_equals_one_eighth_l3763_376377


namespace NUMINAMATH_CALUDE_line_length_after_erasing_l3763_376358

/-- Proves that erasing 33 cm from a 1 m line results in a 67 cm line -/
theorem line_length_after_erasing :
  ∀ (initial_length : ℝ) (erased_length : ℝ),
  initial_length = 1 →
  erased_length = 33 / 100 →
  (initial_length - erased_length) * 100 = 67 := by
sorry

end NUMINAMATH_CALUDE_line_length_after_erasing_l3763_376358


namespace NUMINAMATH_CALUDE_friday_temperature_l3763_376387

def monday_temp : ℝ := 40

theorem friday_temperature 
  (h1 : (monday_temp + tuesday_temp + wednesday_temp + thursday_temp) / 4 = 48)
  (h2 : (tuesday_temp + wednesday_temp + thursday_temp + friday_temp) / 4 = 46) :
  friday_temp = 32 := by
sorry

end NUMINAMATH_CALUDE_friday_temperature_l3763_376387


namespace NUMINAMATH_CALUDE_cubic_inequality_l3763_376381

theorem cubic_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3763_376381
