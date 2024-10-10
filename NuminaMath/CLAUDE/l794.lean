import Mathlib

namespace set_equivalence_l794_79443

theorem set_equivalence : 
  {x : ℕ | 8 < x ∧ x < 12} = {9, 10, 11} := by
sorry

end set_equivalence_l794_79443


namespace angle_rotation_l794_79472

def first_quadrant (α : Real) : Prop :=
  0 < α ∧ α < Real.pi / 2

def third_quadrant (α : Real) : Prop :=
  Real.pi < α ∧ α < 3 * Real.pi / 2

theorem angle_rotation (α : Real) :
  first_quadrant α → third_quadrant (α + Real.pi) := by
  sorry

end angle_rotation_l794_79472


namespace limit_of_sequence_a_l794_79444

def a (n : ℕ) : ℚ := (1 - 2 * n^2) / (2 + 4 * n^2)

theorem limit_of_sequence_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-1/2)| < ε :=
sorry

end limit_of_sequence_a_l794_79444


namespace only_fourth_statement_correct_l794_79487

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the theorem
theorem only_fourth_statement_correct 
  (a b : Line) 
  (α β : Plane) 
  (distinct_lines : a ≠ b) 
  (distinct_planes : α ≠ β) :
  (∃ (a b : Line) (α β : Plane),
    perpendicular a b ∧ 
    perpendicular_plane a α ∧ 
    perpendicular_plane b β → 
    perpendicular_planes α β) ∧
  (¬∃ (a b : Line) (α : Plane),
    perpendicular a b ∧ 
    parallel a α → 
    parallel b α) ∧
  (¬∃ (a : Line) (α β : Plane),
    parallel a α ∧ 
    perpendicular_planes α β → 
    perpendicular_plane a β) ∧
  (¬∃ (a : Line) (α β : Plane),
    perpendicular_plane a β ∧ 
    perpendicular_planes α β → 
    parallel a α) :=
by sorry

end only_fourth_statement_correct_l794_79487


namespace people_per_column_second_arrangement_l794_79419

/-- 
Given a group of people that can be arranged in two ways:
1. 16 columns with 30 people per column
2. 8 columns with an unknown number of people per column

This theorem proves that the number of people per column in the second arrangement is 60.
-/
theorem people_per_column_second_arrangement 
  (total_people : ℕ) 
  (columns_first : ℕ) 
  (people_per_column_first : ℕ) 
  (columns_second : ℕ) : 
  columns_first = 16 → 
  people_per_column_first = 30 → 
  columns_second = 8 → 
  total_people = columns_first * people_per_column_first → 
  total_people / columns_second = 60 := by
  sorry

end people_per_column_second_arrangement_l794_79419


namespace new_average_income_l794_79446

/-- Given a family with 3 earning members and an average monthly income,
    calculate the new average income after one member passes away. -/
theorem new_average_income
  (initial_members : ℕ)
  (initial_average : ℚ)
  (deceased_income : ℚ)
  (h1 : initial_members = 3)
  (h2 : initial_average = 735)
  (h3 : deceased_income = 905) :
  let total_income := initial_members * initial_average
  let remaining_income := total_income - deceased_income
  let remaining_members := initial_members - 1
  remaining_income / remaining_members = 650 := by
sorry

end new_average_income_l794_79446


namespace polynomial_coefficient_sum_l794_79422

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 36 := by
sorry

end polynomial_coefficient_sum_l794_79422


namespace volunteer_selection_l794_79497

/-- The number of ways to select 3 volunteers from 5, with at most one of A and B --/
def select_volunteers (total : ℕ) (to_select : ℕ) (special : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose (total - special) (to_select - special)

/-- Theorem stating that selecting 3 from 5 with at most one of two special volunteers results in 7 ways --/
theorem volunteer_selection :
  select_volunteers 5 3 2 = 7 := by
  sorry

end volunteer_selection_l794_79497


namespace xyz_product_l794_79473

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 168)
  (h2 : y * (z + x) = 186)
  (h3 : z * (x + y) = 194) :
  x * y * z = 860 := by
sorry

end xyz_product_l794_79473


namespace new_average_age_l794_79461

def initial_people : ℕ := 8
def initial_average_age : ℚ := 35
def leaving_person_age : ℕ := 25
def remaining_people : ℕ := 7

theorem new_average_age :
  let total_age : ℚ := initial_people * initial_average_age
  let remaining_age : ℚ := total_age - leaving_person_age
  remaining_age / remaining_people = 36.42857 := by
  sorry

end new_average_age_l794_79461


namespace complex_magnitude_l794_79460

theorem complex_magnitude (z : ℂ) (h : (1 + Complex.I) * z = 1 - 7 * Complex.I) : 
  Complex.abs z = 4 * Real.sqrt 2 := by
  sorry

end complex_magnitude_l794_79460


namespace square_root_problem_l794_79447

theorem square_root_problem (m n : ℝ) 
  (h1 : (5*m - 2)^(1/3) = -3) 
  (h2 : Real.sqrt (3*m + 2*n - 1) = 4) : 
  Real.sqrt (2*m + n + 10) = 4 ∨ Real.sqrt (2*m + n + 10) = -4 := by
  sorry

end square_root_problem_l794_79447


namespace max_k_for_inequality_l794_79439

theorem max_k_for_inequality : 
  (∃ k : ℤ, ∀ x y : ℝ, x > 0 → y > 0 → 4 * x^2 + 9 * y^2 ≥ 2^k * x * y) ∧ 
  (∀ k : ℤ, k > 3 → ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x^2 + 9 * y^2 < 2^k * x * y) :=
sorry

end max_k_for_inequality_l794_79439


namespace conic_section_eccentricity_l794_79484

theorem conic_section_eccentricity (m : ℝ) : 
  (m^2 = 2 * 8) →
  (∃ (e : ℝ), (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧
    ((m > 0 → e = Real.sqrt 3 / 2) ∧
     (m < 0 → e = Real.sqrt 5)) ∧
    (∀ (x y : ℝ), x^2 + y^2 / m = 1 → 
      (m > 0 → e^2 = 1 - (1 / m)) ∧
      (m < 0 → e^2 = 1 + (1 / m)))) :=
by sorry

end conic_section_eccentricity_l794_79484


namespace reinforcement_arrival_theorem_l794_79421

/-- The number of days after which the reinforcement arrived -/
def reinforcement_arrival_day : ℕ := 15

/-- The initial number of men in the garrison -/
def initial_garrison : ℕ := 2000

/-- The number of days the initial provisions would last -/
def initial_provision_days : ℕ := 54

/-- The number of men in the reinforcement -/
def reinforcement : ℕ := 1900

/-- The number of days the provisions last after reinforcement -/
def remaining_days : ℕ := 20

theorem reinforcement_arrival_theorem :
  initial_garrison * (initial_provision_days - reinforcement_arrival_day) =
  (initial_garrison + reinforcement) * remaining_days :=
by sorry

end reinforcement_arrival_theorem_l794_79421


namespace five_hundred_billion_scientific_notation_l794_79475

/-- Express 500 billion in scientific notation -/
theorem five_hundred_billion_scientific_notation :
  (500000000000 : ℝ) = 5 * 10^11 := by
  sorry

end five_hundred_billion_scientific_notation_l794_79475


namespace f_satisfies_conditions_l794_79405

-- Define the function f
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x, f x = f (-x)) ∧                   -- f is an even function
  (∀ x y, 0 ≤ x ∧ x ≤ y → f y ≤ f x) ∧    -- f is monotonically decreasing on [0,+∞)
  (∀ y, ∃ x, f x = y ↔ y ≤ 0) :=          -- Range of f is (-∞,0]
by sorry

end f_satisfies_conditions_l794_79405


namespace parabola_line_intersection_l794_79415

/-- The parabola y = 2x^2 intersects the line y = x - 4 at exactly one point when
    shifted right by p units or down by q units, where p = q = 31/8 -/
theorem parabola_line_intersection (p q : ℝ) : 
  (∀ x y : ℝ, y = 2*(x - p)^2 ∧ y = x - 4 → (∃! z : ℝ, z = x)) ∧
  (∀ x y : ℝ, y = 2*x^2 - q ∧ y = x - 4 → (∃! z : ℝ, z = x)) →
  p = 31/8 ∧ q = 31/8 := by
sorry


end parabola_line_intersection_l794_79415


namespace range_of_f_l794_79450

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 4)

-- State the theorem
theorem range_of_f :
  let S := {y : ℝ | ∃ x : ℝ, x ≠ -8 ∧ f x = y}
  S = {y : ℝ | y < -36 ∨ y > -36} :=
sorry

end range_of_f_l794_79450


namespace min_value_theorem_l794_79455

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : (a + c) * (a + b) = 6 - 2 * Real.sqrt 5) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 := by
  sorry

end min_value_theorem_l794_79455


namespace solution_system_l794_79440

theorem solution_system (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := by
sorry

end solution_system_l794_79440


namespace gold_beads_undetermined_l794_79488

/-- Represents the types of beads used in the corset --/
inductive BeadType
  | Purple
  | Blue
  | Gold

/-- Represents a row of beads --/
structure BeadRow where
  beadType : BeadType
  beadsPerRow : ℕ
  rowCount : ℕ

/-- Represents the corset design --/
structure CorsetDesign where
  purpleRows : BeadRow
  blueRows : BeadRow
  goldBeads : ℕ
  totalCost : ℚ

def carlyDesign : CorsetDesign :=
  { purpleRows := { beadType := BeadType.Purple, beadsPerRow := 20, rowCount := 50 }
  , blueRows := { beadType := BeadType.Blue, beadsPerRow := 18, rowCount := 40 }
  , goldBeads := 0  -- This is what we're trying to determine
  , totalCost := 180 }

/-- The theorem stating that the number of gold beads cannot be determined --/
theorem gold_beads_undetermined (design : CorsetDesign) : 
  design.purpleRows.beadsPerRow = carlyDesign.purpleRows.beadsPerRow ∧ 
  design.purpleRows.rowCount = carlyDesign.purpleRows.rowCount ∧
  design.blueRows.beadsPerRow = carlyDesign.blueRows.beadsPerRow ∧
  design.blueRows.rowCount = carlyDesign.blueRows.rowCount ∧
  design.totalCost = carlyDesign.totalCost →
  ∃ (x y : ℕ), x ≠ y ∧ 
    (∃ (design1 design2 : CorsetDesign), 
      design1.goldBeads = x ∧ 
      design2.goldBeads = y ∧
      design1.purpleRows = design.purpleRows ∧
      design1.blueRows = design.blueRows ∧
      design1.totalCost = design.totalCost ∧
      design2.purpleRows = design.purpleRows ∧
      design2.blueRows = design.blueRows ∧
      design2.totalCost = design.totalCost) :=
by
  sorry

end gold_beads_undetermined_l794_79488


namespace odd_digits_base4_157_l794_79412

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of digits -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 157 is 3 -/
theorem odd_digits_base4_157 : countOddDigits (toBase4 157) = 3 := by
  sorry

end odd_digits_base4_157_l794_79412


namespace notebook_final_price_l794_79489

def initial_price : ℝ := 15
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.25

def price_after_first_discount : ℝ := initial_price * (1 - first_discount_rate)
def final_price : ℝ := price_after_first_discount * (1 - second_discount_rate)

theorem notebook_final_price : final_price = 9 := by
  sorry

end notebook_final_price_l794_79489


namespace train_length_calculation_l794_79495

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (time_to_pass : ℝ) : 
  train_speed = 21 → 
  bridge_length = 130 → 
  time_to_pass = 142.2857142857143 → 
  ∃ (train_length : ℝ), (abs (train_length - 700) < 0.1) ∧ 
    (train_length + bridge_length = train_speed * (1000 / 3600) * time_to_pass) := by
  sorry

end train_length_calculation_l794_79495


namespace jungkook_has_fewest_erasers_l794_79402

def jungkook_erasers : ℕ := 6

def jimin_erasers (j : ℕ) : ℕ := j + 4

def seokjin_erasers (j : ℕ) : ℕ := j - 3

theorem jungkook_has_fewest_erasers :
  ∀ (j s : ℕ), 
    j = jimin_erasers jungkook_erasers →
    s = seokjin_erasers j →
    jungkook_erasers ≤ j ∧ jungkook_erasers ≤ s :=
by sorry

end jungkook_has_fewest_erasers_l794_79402


namespace product_of_numbers_l794_79459

theorem product_of_numbers (x y : ℝ) : 
  x - y = 12 → x^2 + y^2 = 250 → x * y = 52.7364 := by
  sorry

end product_of_numbers_l794_79459


namespace retail_price_calculation_l794_79425

theorem retail_price_calculation (total_cost : ℕ) (price_difference : ℕ) (additional_books : ℕ) :
  total_cost = 48 ∧ price_difference = 2 ∧ additional_books = 4 →
  ∃ (n : ℕ), n > 0 ∧ total_cost / n = 6 ∧ 
  (total_cost / n - price_difference) * (n + additional_books) = total_cost :=
by sorry

end retail_price_calculation_l794_79425


namespace quadratic_condition_l794_79410

theorem quadratic_condition (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) →
  (a > 0 ∧ b^2 - 4*a*c < 0) ∧
  ¬(a > 0 ∧ b^2 - 4*a*c < 0 → ∀ x : ℝ, a * x^2 + b * x + c > 0) :=
by sorry

end quadratic_condition_l794_79410


namespace arithmetic_sequence_seventh_term_l794_79437

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 4 + a 9 = 24) 
  (h_sixth : a 6 = 11) : 
  a 7 = 13 := by
sorry

end arithmetic_sequence_seventh_term_l794_79437


namespace power_of_256_l794_79466

theorem power_of_256 : (256 : ℝ) ^ (5/4 : ℝ) = 1024 := by
  sorry

end power_of_256_l794_79466


namespace quadratic_inequality_range_l794_79403

theorem quadratic_inequality_range (a b c : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 3 → -1 < a * x^2 + b * x + c ∧ a * x^2 + b * x + c < 1) ∧
  (∀ x, x ∉ Set.Ioo (-1 : ℝ) 3 → a * x^2 + b * x + c ≤ -1 ∨ a * x^2 + b * x + c ≥ 1) →
  -1/2 < a ∧ a < 1/2 := by
  sorry

end quadratic_inequality_range_l794_79403


namespace total_sleep_week_is_366_l794_79479

def connor_sleep : ℕ := 6
def luke_sleep : ℕ := connor_sleep + 2
def emma_sleep : ℕ := connor_sleep - 1
def ava_sleep (day : ℕ) : ℕ := 5 + (day - 1) / 2
def puppy_sleep : ℕ := 2 * luke_sleep
def cat_sleep : ℕ := 4 + 7

def total_sleep_week : ℕ :=
  7 * connor_sleep +
  7 * luke_sleep +
  7 * emma_sleep +
  (ava_sleep 1 + ava_sleep 2 + ava_sleep 3 + ava_sleep 4 + ava_sleep 5 + ava_sleep 6 + ava_sleep 7) +
  7 * puppy_sleep +
  7 * cat_sleep

theorem total_sleep_week_is_366 : total_sleep_week = 366 := by
  sorry

end total_sleep_week_is_366_l794_79479


namespace circle_radius_l794_79469

theorem circle_radius (P Q : ℝ) (h : P / Q = 15) : 
  ∃ r : ℝ, r > 0 ∧ P = π * r^2 ∧ Q = 2 * π * r ∧ r = 30 := by
  sorry

end circle_radius_l794_79469


namespace smaller_solution_quadratic_l794_79411

theorem smaller_solution_quadratic (x : ℝ) : 
  (x^2 + 17*x - 72 = 0) → (x = -24 ∨ x = 3) → x = min (-24) 3 := by
  sorry

end smaller_solution_quadratic_l794_79411


namespace index_card_area_l794_79449

theorem index_card_area (length width : ℝ) 
  (h1 : length = 5 ∧ width = 7)
  (h2 : ∃ (shortened_side : ℝ), 
    (shortened_side = length - 2 ∨ shortened_side = width - 2) ∧
    shortened_side * (if shortened_side = length - 2 then width else length) = 21) :
  length * (width - 1) = 30 :=
by sorry

end index_card_area_l794_79449


namespace pencil_profit_l794_79430

def pencil_problem (pencils : ℕ) (buy_price : ℚ) (sell_price : ℚ) : Prop :=
  let cost := (pencils : ℚ) * buy_price / 4
  let revenue := (pencils : ℚ) * sell_price / 5
  let profit := revenue - cost
  profit = 60

theorem pencil_profit : 
  pencil_problem 1200 3 4 :=
sorry

end pencil_profit_l794_79430


namespace S_excludes_A_and_B_only_l794_79404

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, -2)

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ((p.1 - 1)^2 + (p.2 - 1)^2) * ((p.1 - 2)^2 + (p.2 + 2)^2) ≠ 0}

theorem S_excludes_A_and_B_only :
  ∀ p : ℝ × ℝ, p ∉ S ↔ p = A ∨ p = B := by sorry

end S_excludes_A_and_B_only_l794_79404


namespace quadratic_function_properties_l794_79462

/-- A quadratic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + b + 2

/-- The theorem statement -/
theorem quadratic_function_properties (a b : ℝ) :
  a > 0 →
  (∀ x ∈ Set.Icc 0 1, f a b x ≤ f a b 0) →
  (∀ x ∈ Set.Icc 0 1, f a b x ≥ f a b 1) →
  f a b 0 - f a b 1 = 3 →
  f a b 1 = 0 →
  a = 3 ∧ b = 1 ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc (1/3) 2, f a b x < m * x^2 + 1) ↔ m > 3) :=
by sorry


end quadratic_function_properties_l794_79462


namespace unpainted_cubes_count_l794_79468

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Represents the number of painted faces on a unit cube -/
def painted_faces (c : Cube 4) (unit_cube : Fin 64) : ℕ := sorry

theorem unpainted_cubes_count (c : Cube 4) :
  (Finset.univ.filter (fun unit_cube => painted_faces c unit_cube = 0)).card = 58 := by
  sorry

end unpainted_cubes_count_l794_79468


namespace min_value_reciprocal_sum_l794_79474

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1/x + 4/y ≥ 9 := by sorry

end min_value_reciprocal_sum_l794_79474


namespace max_sum_on_circle_l794_79496

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 36) : x + y ≤ 9 := by
  sorry

end max_sum_on_circle_l794_79496


namespace john_double_sam_age_l794_79434

/-- The number of years until John is twice as old as Sam -/
def years_until_double : ℕ := 9

/-- Sam's current age -/
def sam_age : ℕ := 9

/-- John's current age -/
def john_age : ℕ := 3 * sam_age

theorem john_double_sam_age :
  john_age + years_until_double = 2 * (sam_age + years_until_double) :=
by sorry

end john_double_sam_age_l794_79434


namespace smallest_four_digit_divisible_by_9_with_two_even_two_odd_l794_79498

/-- A function that checks if a number has two even and two odd digits -/
def hasTwoEvenTwoOddDigits (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.filter (·.mod 2 = 0)).length = 2 ∧ 
  (digits.filter (·.mod 2 = 1)).length = 2

/-- The smallest positive four-digit number divisible by 9 with two even and two odd digits -/
def smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd : ℕ := 1089

theorem smallest_four_digit_divisible_by_9_with_two_even_two_odd :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n.mod 9 = 0 ∧ hasTwoEvenTwoOddDigits n →
    smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd ≤ n) ∧
  1000 ≤ smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd ∧
  smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd < 10000 ∧
  smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd.mod 9 = 0 ∧
  hasTwoEvenTwoOddDigits smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd :=
by
  sorry

end smallest_four_digit_divisible_by_9_with_two_even_two_odd_l794_79498


namespace no_consecutive_integers_with_square_diff_2000_l794_79478

theorem no_consecutive_integers_with_square_diff_2000 :
  ¬ ∃ (a : ℤ), (a + 1)^2 - a^2 = 2000 := by
  sorry

end no_consecutive_integers_with_square_diff_2000_l794_79478


namespace quadratic_equation_roots_l794_79463

theorem quadratic_equation_roots (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (2*k + 1)*x + k^2 + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (k > 3/4) ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ + x₂ = -x₁ * x₂ → k = 2) :=
by sorry


end quadratic_equation_roots_l794_79463


namespace initial_value_proof_l794_79448

theorem initial_value_proof : 
  ∃! x : ℕ, x ≥ 0 ∧ (∀ y : ℕ, y ≥ 0 → (y + 37) % 3 = 0 ∧ (y + 37) % 5 = 0 ∧ (y + 37) % 7 = 0 ∧ (y + 37) % 8 = 0 → x ≤ y) ∧
  (x + 37) % 3 = 0 ∧ (x + 37) % 5 = 0 ∧ (x + 37) % 7 = 0 ∧ (x + 37) % 8 = 0 :=
by sorry

end initial_value_proof_l794_79448


namespace vincent_outer_space_books_l794_79483

/-- The number of books about outer space Vincent bought -/
def outer_space_books : ℕ := 1

/-- The number of books about animals Vincent bought -/
def animal_books : ℕ := 10

/-- The number of books about trains Vincent bought -/
def train_books : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 16

/-- The total amount spent on books in dollars -/
def total_spent : ℕ := 224

theorem vincent_outer_space_books :
  outer_space_books = 1 ∧
  animal_books * book_cost + outer_space_books * book_cost + train_books * book_cost = total_spent :=
sorry

end vincent_outer_space_books_l794_79483


namespace sum_gcd_lcm_6_15_30_l794_79407

def gcd_three (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm_three (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem sum_gcd_lcm_6_15_30 :
  gcd_three 6 15 30 + lcm_three 6 15 30 = 33 := by
  sorry

end sum_gcd_lcm_6_15_30_l794_79407


namespace trolley_passengers_l794_79417

/-- The number of people on a trolley after three stops -/
def people_on_trolley (initial_pickup : ℕ) (second_stop_off : ℕ) (second_stop_on : ℕ) 
  (third_stop_off : ℕ) (third_stop_on : ℕ) : ℕ :=
  initial_pickup - second_stop_off + second_stop_on - third_stop_off + third_stop_on

/-- Theorem stating the number of people on the trolley after three stops -/
theorem trolley_passengers : 
  people_on_trolley 10 3 20 18 2 = 11 := by
  sorry

end trolley_passengers_l794_79417


namespace estate_division_l794_79481

theorem estate_division (E : ℝ) 
  (h1 : ∃ (x : ℝ), 6 * x = 2/3 * E)  -- Two sons and daughter receive 2/3 of estate in 3:2:1 ratio
  (h2 : ∃ (x : ℝ), 3 * x = E - (9 * x + 750))  -- Wife's share is 3x, where x is daughter's share
  (h3 : 750 ≤ E)  -- Butler's share is $750
  : E = 7500 := by
  sorry

end estate_division_l794_79481


namespace range_positive_iff_l794_79492

/-- The quadratic function f(x) = ax^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

/-- The range of f is a subset of positive real numbers -/
def range_subset_positive (a : ℝ) : Prop :=
  ∀ x, f a x > 0

/-- The necessary and sufficient condition for the range of f to be a subset of positive real numbers -/
theorem range_positive_iff (a : ℝ) :
  range_subset_positive a ↔ 0 ≤ a ∧ a < 4 :=
sorry

end range_positive_iff_l794_79492


namespace at_least_two_unusual_numbers_l794_79401

/-- A hundred-digit number is unusual if its cube ends with itself but its square does not. -/
def IsUnusual (n : ℕ) : Prop :=
  n ^ 3 % 10^100 = n % 10^100 ∧ n ^ 2 % 10^100 ≠ n % 10^100

/-- There are at least two hundred-digit unusual numbers. -/
theorem at_least_two_unusual_numbers : ∃ n₁ n₂ : ℕ,
  n₁ ≠ n₂ ∧
  10^99 ≤ n₁ ∧ n₁ < 10^100 ∧
  10^99 ≤ n₂ ∧ n₂ < 10^100 ∧
  IsUnusual n₁ ∧ IsUnusual n₂ := by
  sorry

end at_least_two_unusual_numbers_l794_79401


namespace find_number_l794_79491

theorem find_number : ∃! x : ℝ, ((((x - 74) * 15) / 5) + 16) - 15 = 58 := by
  sorry

end find_number_l794_79491


namespace unique_pair_satisfying_conditions_l794_79482

theorem unique_pair_satisfying_conditions :
  ∃! (n p : ℕ+), 
    (Nat.Prime p.val) ∧ 
    (-↑n : ℤ) ≤ 2 * ↑p ∧
    (↑p - 1 : ℤ) ^ n.val + 1 ∣ ↑n ^ (p.val - 1) ∧
    n = 3 ∧ p = 3 := by
  sorry

end unique_pair_satisfying_conditions_l794_79482


namespace simplification_and_exponent_sum_l794_79414

-- Define the expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^7 * z^9) ^ (1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z^3 * (5 * x^2 * y) ^ (1/3)

-- Define the sum of exponents outside the radical
def sum_of_exponents : ℕ := 1 + 1 + 3

-- Theorem statement
theorem simplification_and_exponent_sum :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
  original_expression x y z = simplified_expression x y z ∧
  sum_of_exponents = 5 := by sorry

end simplification_and_exponent_sum_l794_79414


namespace time_interval_is_two_seconds_l794_79409

/-- The time interval for birth and death rates in a city --/
def time_interval (birth_rate death_rate net_increase_per_day seconds_per_day : ℕ) : ℚ :=
  seconds_per_day / (net_increase_per_day / (birth_rate - death_rate))

/-- Theorem: The time interval for birth and death rates is 2 seconds --/
theorem time_interval_is_two_seconds :
  time_interval 4 2 86400 86400 = 2 := by
  sorry

#eval time_interval 4 2 86400 86400

end time_interval_is_two_seconds_l794_79409


namespace min_score_for_average_increase_l794_79458

/-- Given 4 tests with an average score of 68, prove that a score of at least 78 on the 5th test 
    is necessary to achieve an average score of more than 70 over all 5 tests. -/
theorem min_score_for_average_increase (current_tests : Nat) (current_average : ℝ) 
  (target_average : ℝ) (min_score : ℝ) : 
  current_tests = 4 → 
  current_average = 68 → 
  target_average > 70 → 
  min_score ≥ 78 → 
  (current_tests * current_average + min_score) / (current_tests + 1) > target_average :=
by sorry

end min_score_for_average_increase_l794_79458


namespace certain_number_proof_l794_79432

theorem certain_number_proof (p q x : ℝ) 
  (h1 : 3 / p = x)
  (h2 : 3 / q = 15)
  (h3 : p - q = 0.3) :
  x = 6 := by
  sorry

end certain_number_proof_l794_79432


namespace range_of_f_l794_79470

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- Define the domain
def domain : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 1 ≤ y ∧ y ≤ 5} :=
sorry

end range_of_f_l794_79470


namespace no_solution_iff_m_eq_neg_one_l794_79445

theorem no_solution_iff_m_eq_neg_one (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (3 - 2*x)/(x - 3) + (2 + m*x)/(3 - x) ≠ -1) ↔ m = -1 := by
  sorry

end no_solution_iff_m_eq_neg_one_l794_79445


namespace second_smallest_hot_dog_packs_l794_79453

theorem second_smallest_hot_dog_packs : 
  (∃ n : ℕ, n > 0 ∧ (12 * n) % 8 = (8 - 7) % 8 ∧ 
   (∀ m : ℕ, m > 0 ∧ m < n → (12 * m) % 8 ≠ (8 - 7) % 8)) → 
  (∃ n : ℕ, n > 0 ∧ (12 * n) % 8 = (8 - 7) % 8 ∧ 
   (∃! m : ℕ, m > 0 ∧ m < n ∧ (12 * m) % 8 = (8 - 7) % 8) ∧ n = 5) :=
by sorry

end second_smallest_hot_dog_packs_l794_79453


namespace chris_donuts_l794_79451

theorem chris_donuts (initial_donuts : ℕ) : 
  (initial_donuts : ℝ) * 0.9 - 4 = 23 → initial_donuts = 30 := by
  sorry

end chris_donuts_l794_79451


namespace dog_eaten_cost_l794_79436

/-- Represents the cost of ingredients for a cake -/
structure CakeIngredients where
  flour : Float
  sugar : Float
  eggs : Float
  butter : Float

/-- Represents the cake and its consumption -/
structure Cake where
  ingredients : CakeIngredients
  totalSlices : Nat
  slicesEatenByMother : Nat

def totalCost (c : CakeIngredients) : Float :=
  c.flour + c.sugar + c.eggs + c.butter

def costPerSlice (cake : Cake) : Float :=
  totalCost cake.ingredients / cake.totalSlices.toFloat

def slicesEatenByDog (cake : Cake) : Nat :=
  cake.totalSlices - cake.slicesEatenByMother

theorem dog_eaten_cost (cake : Cake) 
  (h1 : cake.ingredients = { flour := 4, sugar := 2, eggs := 0.5, butter := 2.5 })
  (h2 : cake.totalSlices = 6)
  (h3 : cake.slicesEatenByMother = 2) :
  costPerSlice cake * (slicesEatenByDog cake).toFloat = 6 := by
  sorry

#check dog_eaten_cost

end dog_eaten_cost_l794_79436


namespace line_points_k_value_l794_79454

theorem line_points_k_value (m n k : ℝ) : 
  (m = 2 * n + 5) → 
  (m + 4 = 2 * (n + k) + 5) → 
  k = 2 := by
sorry

end line_points_k_value_l794_79454


namespace inequality_equivalence_l794_79457

theorem inequality_equivalence (x : ℝ) : 3 * x^2 + 2 * x - 3 > 12 - 2 * x ↔ x < -3 ∨ x > 5/3 := by
  sorry

end inequality_equivalence_l794_79457


namespace fraction_simplification_l794_79400

theorem fraction_simplification (a b x : ℝ) :
  (Real.sqrt (a^2 + b^2 + x^2) - (x^2 - a^2 - b^2) / Real.sqrt (a^2 + b^2 + x^2)) / (a^2 + b^2 + x^2) = 
  2 * (a^2 + b^2) / (a^2 + b^2 + x^2)^(3/2) := by
  sorry

end fraction_simplification_l794_79400


namespace no_real_solution_for_log_equation_l794_79420

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 4*x - 15) :=
by sorry

end no_real_solution_for_log_equation_l794_79420


namespace factorial_divisibility_l794_79485

theorem factorial_divisibility (m : ℕ) (h : m > 1) :
  (m - 1).factorial % m = 0 ↔ ¬ Nat.Prime m := by sorry

end factorial_divisibility_l794_79485


namespace simple_random_sampling_probability_l794_79477

-- Define the population size
def population_size : ℕ := 100

-- Define the sample size
def sample_size : ℕ := 5

-- Define the probability of an individual being drawn
def prob_individual_drawn (n : ℕ) (k : ℕ) : ℚ := k / n

-- Theorem statement
theorem simple_random_sampling_probability :
  prob_individual_drawn population_size sample_size = 1 / 20 := by
  sorry

end simple_random_sampling_probability_l794_79477


namespace added_value_theorem_l794_79413

theorem added_value_theorem (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 8) :
  x + y = 128 * (1/x) → y = 8 := by
sorry

end added_value_theorem_l794_79413


namespace new_person_weight_l794_79423

theorem new_person_weight
  (initial_count : ℕ)
  (average_increase : ℝ)
  (replaced_weight : ℝ)
  (hcount : initial_count = 7)
  (hincrease : average_increase = 3.5)
  (hreplaced : replaced_weight = 75)
  : ℝ :=
by
  -- The weight of the new person
  sorry

#check new_person_weight

end new_person_weight_l794_79423


namespace cans_given_away_equals_2500_l794_79441

/-- Represents the food bank's inventory and distribution --/
structure FoodBank where
  initialStock : Nat
  day1People : Nat
  day1CansPerPerson : Nat
  day1Restock : Nat
  day2People : Nat
  day2CansPerPerson : Nat
  day2Restock : Nat

/-- Calculates the total number of cans given away --/
def totalCansGivenAway (fb : FoodBank) : Nat :=
  fb.day1People * fb.day1CansPerPerson + fb.day2People * fb.day2CansPerPerson

/-- Theorem stating that given the specific conditions, 2500 cans were given away --/
theorem cans_given_away_equals_2500 (fb : FoodBank) 
  (h1 : fb.initialStock = 2000)
  (h2 : fb.day1People = 500)
  (h3 : fb.day1CansPerPerson = 1)
  (h4 : fb.day1Restock = 1500)
  (h5 : fb.day2People = 1000)
  (h6 : fb.day2CansPerPerson = 2)
  (h7 : fb.day2Restock = 3000) :
  totalCansGivenAway fb = 2500 := by
  sorry

end cans_given_away_equals_2500_l794_79441


namespace base_of_power_l794_79433

theorem base_of_power (b : ℝ) (x y : ℤ) 
  (h1 : b^x * 4^y = 531441)
  (h2 : x - y = 12)
  (h3 : x = 12) : 
  b = 3 := by sorry

end base_of_power_l794_79433


namespace reciprocal_roots_quadratic_l794_79465

theorem reciprocal_roots_quadratic (a b : ℂ) : 
  (a^2 + 4*a + 8 = 0) ∧ (b^2 + 4*b + 8 = 0) → 
  (8*(1/a)^2 + 4*(1/a) + 1 = 0) ∧ (8*(1/b)^2 + 4*(1/b) + 1 = 0) := by
  sorry

end reciprocal_roots_quadratic_l794_79465


namespace woodworker_tables_l794_79427

/-- Proves the number of tables made by a woodworker given the total number of furniture legs and chairs made -/
theorem woodworker_tables (total_legs : ℕ) (chairs : ℕ) : 
  total_legs = 40 → 
  chairs = 6 → 
  ∃ (tables : ℕ), 
    tables * 4 + chairs * 4 = total_legs ∧ 
    tables = 4 := by
  sorry

end woodworker_tables_l794_79427


namespace quadratic_function_m_value_l794_79476

theorem quadratic_function_m_value :
  ∃! m : ℝ, (abs (m - 1) = 2) ∧ (m - 3 ≠ 0) ∧ (m = -1) :=
by sorry

end quadratic_function_m_value_l794_79476


namespace circle_equation_l794_79494

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := y = x - 1

-- Define the line l₂
def l₂ (x : ℝ) : Prop := x = -1

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 1 = 0

-- Define the circle equation
def circle_eq (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

theorem circle_equation 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : l₁ x₁ y₁) 
  (h₂ : l₁ x₂ y₂) 
  (h₃ : quadratic_eq x₁) 
  (h₄ : quadratic_eq x₂) :
  ∃ (a b r : ℝ), 
    (circle_eq x₁ y₁ a b r ∧ 
     circle_eq x₂ y₂ a b r ∧ 
     (a = 3 ∧ b = 2 ∧ r = 4) ∨ 
     (a = 11 ∧ b = -6 ∧ r = 12)) ∧
    ∀ (x : ℝ), l₂ x → (x - a)^2 = r^2 :=
sorry

end circle_equation_l794_79494


namespace duck_cow_problem_l794_79428

theorem duck_cow_problem (D C : ℕ) : 
  2 * D + 4 * C = 2 * (D + C) + 28 → C = 14 := by
sorry

end duck_cow_problem_l794_79428


namespace book_arrangement_l794_79406

theorem book_arrangement (k m n : ℕ) :
  (∃ (f : ℕ → ℕ), f 0 = 3 * k.factorial * m.factorial * n.factorial) ∧
  (∃ (g : ℕ → ℕ), g 0 = (m + n).factorial * (m + n + 1) * k.factorial) := by
  sorry

end book_arrangement_l794_79406


namespace range_of_a_l794_79464

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : A ∩ B a = B a → a = 1 ∨ a ≤ -1 := by
  sorry

end range_of_a_l794_79464


namespace sum_inequality_l794_79480

/-- Given real numbers x₁, x₂, x₃ such that the sum of any two is greater than the third,
    prove that (2/3) * (∑ xᵢ) * (∑ xᵢ²) > ∑ xᵢ³ + x₁x₂x₃ -/
theorem sum_inequality (x₁ x₂ x₃ : ℝ) 
    (h₁ : x₁ + x₂ > x₃) (h₂ : x₂ + x₃ > x₁) (h₃ : x₃ + x₁ > x₂) :
    2/3 * (x₁ + x₂ + x₃) * (x₁^2 + x₂^2 + x₃^2) > x₁^3 + x₂^3 + x₃^3 + x₁*x₂*x₃ := by
  sorry

end sum_inequality_l794_79480


namespace lcm_5_6_10_12_l794_79435

theorem lcm_5_6_10_12 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 12)) = 60 := by
  sorry

end lcm_5_6_10_12_l794_79435


namespace shirt_cost_l794_79431

/-- Given the cost equations for jeans, shirts, and hats, prove the cost of a shirt. -/
theorem shirt_cost (j s h : ℚ) 
  (eq1 : 3 * j + 2 * s + h = 89)
  (eq2 : 2 * j + 3 * s + 2 * h = 102)
  (eq3 : 4 * j + s + 3 * h = 125) :
  s = 12.53 := by
  sorry

end shirt_cost_l794_79431


namespace cat_sale_theorem_l794_79429

/-- Represents the count of cats for each breed -/
structure CatCounts where
  siamese : Nat
  persian : Nat
  house : Nat
  maineCoon : Nat

/-- Represents the number of pairs sold for each breed -/
structure SoldPairs where
  siamese : Nat
  persian : Nat
  maineCoon : Nat

/-- Calculates the remaining cats after the sale -/
def remainingCats (initial : CatCounts) (sold : SoldPairs) : CatCounts :=
  { siamese := initial.siamese - sold.siamese,
    persian := initial.persian - sold.persian,
    house := initial.house,
    maineCoon := initial.maineCoon - sold.maineCoon }

theorem cat_sale_theorem (initial : CatCounts) (sold : SoldPairs) :
  initial.siamese = 25 →
  initial.persian = 18 →
  initial.house = 12 →
  initial.maineCoon = 10 →
  sold.siamese = 6 →
  sold.persian = 4 →
  sold.maineCoon = 3 →
  let remaining := remainingCats initial sold
  remaining.siamese = 19 ∧
  remaining.persian = 14 ∧
  remaining.house = 12 ∧
  remaining.maineCoon = 7 :=
by sorry

end cat_sale_theorem_l794_79429


namespace distinct_roots_condition_roots_when_k_is_one_l794_79490

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := x^2 + (2*k + 3)*x + k^2 + 5*k

-- Theorem for part 1
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
   quadratic_equation k x = 0 ∧ 
   quadratic_equation k y = 0) →
  k < 9/8 :=
sorry

-- Theorem for part 2
theorem roots_when_k_is_one :
  quadratic_equation 1 (-2) = 0 ∧ 
  quadratic_equation 1 (-3) = 0 :=
sorry

end distinct_roots_condition_roots_when_k_is_one_l794_79490


namespace hyperbola_equation_l794_79452

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → y = 2*x) →
  (∃ x₀ : ℝ, x₀ = 5 ∧ (∀ y : ℝ, y^2 = 20*x₀ → (x₀^2 / a^2 - y^2 / b^2 = 1))) →
  a^2 = 5 ∧ b^2 = 20 := by
sorry

end hyperbola_equation_l794_79452


namespace vector_sum_triangle_l794_79486

-- Define a triangle ABC in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vector addition
def vectorAdd (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector subtraction (used to represent directed edges)
def vectorSub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Theorem statement
theorem vector_sum_triangle (t : Triangle) : 
  vectorAdd (vectorAdd (vectorSub t.B t.A) (vectorSub t.C t.B)) (vectorSub t.A t.C) = (0, 0) := by
  sorry

end vector_sum_triangle_l794_79486


namespace remainder_problem_l794_79456

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 38) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
sorry

end remainder_problem_l794_79456


namespace inequality_region_l794_79416

theorem inequality_region (x y : ℝ) : 
  x + 3*y - 1 < 0 → (x < 1 - 3*y) ∧ (y < (1 - x)/3) := by
  sorry

end inequality_region_l794_79416


namespace jonny_stairs_l794_79499

theorem jonny_stairs :
  ∀ (j : ℕ),
  (j + (j / 3 - 7) = 1685) →
  j = 1521 := by
sorry

end jonny_stairs_l794_79499


namespace sum_of_coordinates_symmetric_points_l794_79424

/-- Two points A(a, 2022) and A'(-2023, b) are symmetric with respect to the origin if and only if
    their coordinates satisfy the given conditions. -/
def symmetric_points (a b : ℝ) : Prop :=
  a = 2023 ∧ b = -2022

/-- The sum of a and b is 1 when A(a, 2022) and A'(-2023, b) are symmetric with respect to the origin. -/
theorem sum_of_coordinates_symmetric_points (a b : ℝ) 
    (h : symmetric_points a b) : a + b = 1 := by
  sorry

#check sum_of_coordinates_symmetric_points

end sum_of_coordinates_symmetric_points_l794_79424


namespace walking_speed_l794_79418

theorem walking_speed (distance : Real) (time_minutes : Real) (speed : Real) : 
  distance = 500 ∧ time_minutes = 6 → speed = 5000 := by
  sorry

end walking_speed_l794_79418


namespace line_problem_l794_79442

theorem line_problem (front_position back_position total : ℕ) 
  (h1 : front_position = 8)
  (h2 : back_position = 6)
  (h3 : total = front_position + back_position - 1) :
  total = 13 := by
  sorry

end line_problem_l794_79442


namespace biology_group_specimen_exchange_l794_79408

/-- Represents the number of specimens exchanged in a biology interest group --/
def specimens_exchanged (x : ℕ) : ℕ := x * (x - 1)

/-- Theorem stating that the equation x(x-1) = 110 correctly represents the situation --/
theorem biology_group_specimen_exchange (x : ℕ) :
  specimens_exchanged x = 110 ↔ x * (x - 1) = 110 := by
  sorry

end biology_group_specimen_exchange_l794_79408


namespace complex_square_pure_imaginary_l794_79471

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_square_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((1 + a * Complex.I) ^ 2) → a = 1 ∨ a = -1 := by
  sorry

end complex_square_pure_imaginary_l794_79471


namespace imaginary_power_sum_zero_l794_79426

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum_zero : 
  i^14762 + i^14763 + i^14764 + i^14765 = 0 :=
by
  sorry

-- Define the property of i
axiom i_squared : i^2 = -1

end imaginary_power_sum_zero_l794_79426


namespace greatest_a_value_l794_79438

theorem greatest_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = -12) → 
  (a > 0) → 
  (∀ b : ℤ, b > a → ¬(∃ y : ℤ, y^2 + b*y = -12)) → 
  a = 13 := by
  sorry

end greatest_a_value_l794_79438


namespace james_savings_proof_l794_79467

def weekly_allowance : ℕ := 10
def savings_weeks : ℕ := 4
def video_game_fraction : ℚ := 1 / 2
def book_fraction : ℚ := 1 / 4

theorem james_savings_proof :
  let total_savings := weekly_allowance * savings_weeks
  let after_video_game := total_savings - (video_game_fraction * total_savings)
  let final_amount := after_video_game - (book_fraction * after_video_game)
  final_amount = 15 := by sorry

end james_savings_proof_l794_79467


namespace binomial_expansion_sum_l794_79493

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end binomial_expansion_sum_l794_79493
