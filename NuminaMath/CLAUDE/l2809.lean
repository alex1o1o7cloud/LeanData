import Mathlib

namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2809_280987

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^4 + 12 * b^4 + 50 * c^4 + 1 / (9 * a * b * c) ≥ 2 * Real.sqrt (20 / 3) :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (8 * a^4 + 12 * b^4 + 50 * c^4 + 1 / (9 * a * b * c) = 2 * Real.sqrt (20 / 3)) ↔
  (a = (3/2)^(1/4) * b ∧ b = (25/6)^(1/4) * c ∧ c = (4/25)^(1/4) * a) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2809_280987


namespace NUMINAMATH_CALUDE_late_arrivals_count_l2809_280985

/-- Represents the number of people per lollipop -/
def people_per_lollipop : ℕ := 5

/-- Represents the initial number of people -/
def initial_people : ℕ := 45

/-- Represents the total number of lollipops given away -/
def total_lollipops : ℕ := 12

/-- Calculates the number of people who came in later -/
def late_arrivals : ℕ := total_lollipops * people_per_lollipop - initial_people

theorem late_arrivals_count : late_arrivals = 15 := by
  sorry

end NUMINAMATH_CALUDE_late_arrivals_count_l2809_280985


namespace NUMINAMATH_CALUDE_solve_exam_problem_l2809_280942

def exam_problem (exam_A_total exam_B_total exam_A_wrong exam_B_correct_diff : ℕ) : Prop :=
  let exam_A_correct := exam_A_total - exam_A_wrong
  let exam_B_correct := exam_A_correct + exam_B_correct_diff
  let exam_B_wrong := exam_B_total - exam_B_correct
  exam_A_wrong + exam_B_wrong = 9

theorem solve_exam_problem :
  exam_problem 12 15 4 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exam_problem_l2809_280942


namespace NUMINAMATH_CALUDE_fare_660_equals_3_miles_unique_distance_for_660_l2809_280974

/-- Calculates the taxi fare for a given distance -/
def taxi_fare (distance : ℚ) : ℚ :=
  1 + 0.4 * (5 * distance - 1)

/-- Proves that a fare of $6.60 corresponds to a distance of 3 miles -/
theorem fare_660_equals_3_miles :
  taxi_fare 3 = 6.6 :=
sorry

/-- Proves that 3 miles is the unique distance that results in a fare of $6.60 -/
theorem unique_distance_for_660 :
  ∀ d : ℚ, taxi_fare d = 6.6 → d = 3 :=
sorry

end NUMINAMATH_CALUDE_fare_660_equals_3_miles_unique_distance_for_660_l2809_280974


namespace NUMINAMATH_CALUDE_sum_of_unit_vector_magnitudes_l2809_280938

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given two unit vectors, prove that the sum of their magnitudes is 2 -/
theorem sum_of_unit_vector_magnitudes
  (a₀ b₀ : E) 
  (ha : ‖a₀‖ = 1) 
  (hb : ‖b₀‖ = 1) : 
  ‖a₀‖ + ‖b₀‖ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_unit_vector_magnitudes_l2809_280938


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2809_280984

theorem absolute_value_inequality (x : ℝ) : 
  abs (x - 1) + abs (x + 2) < 5 ↔ -3 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2809_280984


namespace NUMINAMATH_CALUDE_money_ratio_l2809_280923

def money_problem (total : ℚ) (rene : ℚ) : Prop :=
  ∃ (isha florence : ℚ) (k : ℕ),
    isha = (1/3) * total ∧
    florence = (1/2) * isha ∧
    florence = k * rene ∧
    total = isha + florence + rene ∧
    rene = 300 ∧
    total = 1650 ∧
    florence / rene = 3/2

theorem money_ratio :
  money_problem 1650 300 := by sorry

end NUMINAMATH_CALUDE_money_ratio_l2809_280923


namespace NUMINAMATH_CALUDE_percentage_to_pass_l2809_280901

/-- Given a student's marks and passing conditions, prove the percentage needed to pass -/
theorem percentage_to_pass
  (marks_obtained : ℕ)
  (marks_to_pass : ℕ)
  (max_marks : ℕ)
  (h1 : marks_obtained = 130)
  (h2 : marks_to_pass = marks_obtained + 14)
  (h3 : max_marks = 400) :
  (marks_to_pass : ℚ) / max_marks * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l2809_280901


namespace NUMINAMATH_CALUDE_modular_congruence_l2809_280945

theorem modular_congruence (x : ℤ) :
  (5 * x + 9) % 16 = 3 → (3 * x + 8) % 16 = 14 := by
sorry

end NUMINAMATH_CALUDE_modular_congruence_l2809_280945


namespace NUMINAMATH_CALUDE_ellipse_m_value_l2809_280973

/-- An ellipse with equation x²/10 + y²/m = 1, foci on y-axis, and major axis length 8 has m = 16 -/
theorem ellipse_m_value (m : ℝ) : 
  (∀ x y : ℝ, x^2 / 10 + y^2 / m = 1) →  -- Ellipse equation
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 = 10 ∧ b^2 = m) →  -- Standard form of ellipse
  (∀ x : ℝ, x^2 / 10 + 0^2 / m ≠ 1) →  -- Foci on y-axis
  (2 * Real.sqrt m = 8) →  -- Major axis length
  m = 16 := by
sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l2809_280973


namespace NUMINAMATH_CALUDE_equality_check_l2809_280912

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  ((-1)^3 = -1^3) ∧ 
  ((2/3)^2 ≠ 2^2/3) ∧ 
  ((-2)^2 ≠ -2^2) := by
  sorry

end NUMINAMATH_CALUDE_equality_check_l2809_280912


namespace NUMINAMATH_CALUDE_f_monotonic_range_l2809_280935

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 3

-- Define the property of being monotonic on an interval
def IsMonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ f x > f y)

-- Theorem statement
theorem f_monotonic_range (m : ℝ) :
  IsMonotonicOn f m (m + 4) → m ∈ Set.Iic (-5) ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_range_l2809_280935


namespace NUMINAMATH_CALUDE_parallelogram_cyclic_equidistant_implies_bisector_l2809_280960

-- Define the necessary structures and functions
structure Point := (x y : ℝ)

def Line := Point → Point → Prop

def parallelogram (A B C D : Point) : Prop := sorry

def cyclic_quadrilateral (B C E D : Point) : Prop := sorry

def intersects_interior (l : Line) (A B : Point) (F : Point) : Prop := sorry

def intersects (l : Line) (A B : Point) (G : Point) : Prop := sorry

def distance (P Q : Point) : ℝ := sorry

def angle_bisector (l : Line) (A B C : Point) : Prop := sorry

-- State the theorem
theorem parallelogram_cyclic_equidistant_implies_bisector
  (A B C D E F G : Point) (ℓ : Line) :
  parallelogram A B C D →
  cyclic_quadrilateral B C E D →
  ℓ A F →
  ℓ A G →
  intersects_interior ℓ D C F →
  intersects ℓ B C G →
  distance E F = distance E G →
  distance E F = distance E C →
  angle_bisector ℓ D A B :=
sorry

end NUMINAMATH_CALUDE_parallelogram_cyclic_equidistant_implies_bisector_l2809_280960


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l2809_280959

theorem sin_thirteen_pi_sixths : Real.sin (13 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l2809_280959


namespace NUMINAMATH_CALUDE_dishes_bananas_difference_is_ten_l2809_280920

/-- The number of pears Charles picked -/
def pears_picked : ℕ := 50

/-- The number of dishes Sandrine washed -/
def dishes_washed : ℕ := 160

/-- The number of bananas Charles cooked -/
def bananas_cooked : ℕ := 3 * pears_picked

/-- The difference between dishes washed and bananas cooked -/
def dishes_bananas_difference : ℕ := dishes_washed - bananas_cooked

theorem dishes_bananas_difference_is_ten :
  dishes_bananas_difference = 10 := by sorry

end NUMINAMATH_CALUDE_dishes_bananas_difference_is_ten_l2809_280920


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2809_280913

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  geometric_subsequence : (a 3) ^ 2 = a 2 * a 7
  initial_condition : 2 * a 1 + a 2 = 1

/-- The general term of the arithmetic sequence is 5/3 - n -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n, seq.a n = 5/3 - n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2809_280913


namespace NUMINAMATH_CALUDE_blood_expires_same_day_l2809_280972

/- Define the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/- Define the expiration time in seconds (8!) -/
def expiration_time : ℕ := Nat.factorial 8

/- Theorem: The blood expires in less than one day -/
theorem blood_expires_same_day : 
  (expiration_time : ℚ) / seconds_per_day < 1 := by
  sorry


end NUMINAMATH_CALUDE_blood_expires_same_day_l2809_280972


namespace NUMINAMATH_CALUDE_correct_assembly_rates_l2809_280992

/-- Represents the assembly and disassembly rates of coffee grinders for two robots -/
structure CoffeeGrinderRates where
  hubert_assembly : ℝ     -- Hubert's assembly rate (grinders per hour)
  robert_assembly : ℝ     -- Robert's assembly rate (grinders per hour)

/-- Checks if the given rates satisfy the problem conditions -/
def satisfies_conditions (rates : CoffeeGrinderRates) : Prop :=
  -- Each assembles four times faster than the other disassembles
  rates.hubert_assembly = 4 * (rates.robert_assembly / 4) ∧
  rates.robert_assembly = 4 * (rates.hubert_assembly / 4) ∧
  -- Morning shift conditions
  (rates.hubert_assembly - rates.robert_assembly / 4) * 3 = 27 ∧
  -- Afternoon shift conditions
  (rates.robert_assembly - rates.hubert_assembly / 4) * 6 = 120

/-- The theorem stating the correct assembly rates for Hubert and Robert -/
theorem correct_assembly_rates :
  ∃ (rates : CoffeeGrinderRates),
    satisfies_conditions rates ∧
    rates.hubert_assembly = 12 ∧
    rates.robert_assembly = 80 / 3 := by
  sorry


end NUMINAMATH_CALUDE_correct_assembly_rates_l2809_280992


namespace NUMINAMATH_CALUDE_ethanol_in_full_tank_l2809_280928

def tank_capacity : ℝ := 212
def fuel_A_volume : ℝ := 98
def fuel_A_ethanol_percentage : ℝ := 0.12
def fuel_B_ethanol_percentage : ℝ := 0.16

theorem ethanol_in_full_tank : 
  let fuel_B_volume := tank_capacity - fuel_A_volume
  let ethanol_in_A := fuel_A_volume * fuel_A_ethanol_percentage
  let ethanol_in_B := fuel_B_volume * fuel_B_ethanol_percentage
  ethanol_in_A + ethanol_in_B = 30 := by
sorry

end NUMINAMATH_CALUDE_ethanol_in_full_tank_l2809_280928


namespace NUMINAMATH_CALUDE_ratio_x_to_w_l2809_280953

/-- Given the relationships between x, y, z, and w, prove that the ratio of x to w is 0.486 -/
theorem ratio_x_to_w (x y z w : ℝ) 
  (h1 : x = 1.20 * y)
  (h2 : y = 0.30 * z)
  (h3 : z = 1.35 * w) :
  x / w = 0.486 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_w_l2809_280953


namespace NUMINAMATH_CALUDE_no_two_digit_reverse_sum_twice_square_l2809_280915

theorem no_two_digit_reverse_sum_twice_square : 
  ¬ ∃ (N : ℕ), 
    (10 ≤ N ∧ N ≤ 99) ∧ 
    ∃ (k : ℕ), 
      N + (10 * (N % 10) + N / 10) = 2 * k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_reverse_sum_twice_square_l2809_280915


namespace NUMINAMATH_CALUDE_tangent_line_and_minimum_l2809_280956

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

theorem tangent_line_and_minimum (a : ℝ) :
  (∃ y, x - 4 * y + 4 * Real.log 2 - 4 = 0 ↔ 
    y = f 1 x ∧ x = 2) ∧
  (a ≤ 0 → ∀ x ∈ Set.Ioo 0 (Real.exp 1), ∃ y ∈ Set.Ioo 0 (Real.exp 1), f a y < f a x) ∧
  (0 < a → a < Real.exp 1 → ∀ x ∈ Set.Ioo 0 (Real.exp 1), f a a ≤ f a x ∧ f a a = Real.log a) ∧
  (Real.exp 1 ≤ a → ∀ x ∈ Set.Ioo 0 (Real.exp 1), a / Real.exp 1 ≤ f a x ∧ a / Real.exp 1 = f a (Real.exp 1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_minimum_l2809_280956


namespace NUMINAMATH_CALUDE_last_colored_cell_position_l2809_280969

/-- Represents a position in the grid --/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the dimensions of the rectangle --/
structure Dimensions :=
  (width : Nat)
  (height : Nat)

/-- Represents the direction of movement in the spiral --/
inductive Direction
  | Right
  | Down
  | Left
  | Up

/-- Function to determine the next position in the spiral --/
def nextPosition (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Right => { row := pos.row,     col := pos.col + 1 }
  | Direction.Down  => { row := pos.row + 1, col := pos.col }
  | Direction.Left  => { row := pos.row,     col := pos.col - 1 }
  | Direction.Up    => { row := pos.row - 1, col := pos.col }

/-- Function to determine if a position is within the rectangle --/
def isWithinBounds (pos : Position) (dim : Dimensions) : Bool :=
  pos.row ≥ 1 && pos.row ≤ dim.height && pos.col ≥ 1 && pos.col ≤ dim.width

/-- Theorem stating that the last colored cell in a 200x100 rectangle,
    colored in a spiral pattern, is at position (51, 50) --/
theorem last_colored_cell_position :
  ∃ (coloringProcess : Nat → Position),
    (coloringProcess 0 = { row := 1, col := 1 }) →
    (∀ n, isWithinBounds (coloringProcess n) { width := 200, height := 100 }) →
    (∀ n, ∃ dir, nextPosition (coloringProcess n) dir = coloringProcess (n + 1)) →
    (∃ lastStep, ∀ m > lastStep, ¬isWithinBounds (coloringProcess m) { width := 200, height := 100 }) →
    (coloringProcess lastStep = { row := 51, col := 50 }) :=
by sorry


end NUMINAMATH_CALUDE_last_colored_cell_position_l2809_280969


namespace NUMINAMATH_CALUDE_log_xy_value_l2809_280962

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the theorem
theorem log_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : log (x * y^2) = 2) (h2 : log (x^3 * y) = 3) : 
  log (x * y) = 7/5 := by
  sorry


end NUMINAMATH_CALUDE_log_xy_value_l2809_280962


namespace NUMINAMATH_CALUDE_ninas_pet_eyes_l2809_280982

/-- The total number of eyes among Nina's pet insects -/
theorem ninas_pet_eyes : 
  let spider_count : ℕ := 3
  let ant_count : ℕ := 50
  let eyes_per_spider : ℕ := 8
  let eyes_per_ant : ℕ := 2
  let total_eyes : ℕ := spider_count * eyes_per_spider + ant_count * eyes_per_ant
  total_eyes = 124 := by sorry

end NUMINAMATH_CALUDE_ninas_pet_eyes_l2809_280982


namespace NUMINAMATH_CALUDE_apple_bags_problem_l2809_280971

theorem apple_bags_problem (A B C : ℕ) 
  (h1 : A + B + C = 24)
  (h2 : B + C = 18)
  (h3 : A + C = 19) :
  A + B = 11 := by
sorry

end NUMINAMATH_CALUDE_apple_bags_problem_l2809_280971


namespace NUMINAMATH_CALUDE_log_equality_l2809_280946

theorem log_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 4*y^2 = 12*x*y) :
  Real.log (x + 2*y) - 2 * Real.log 2 = 0.5 * (Real.log x + Real.log y) := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l2809_280946


namespace NUMINAMATH_CALUDE_max_expression_value_l2809_280963

def expression (a b c d : ℕ) : ℕ := d * (c^a - b)

theorem max_expression_value :
  ∃ (a b c d : ℕ),
    a ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    b ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    c ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    d ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    expression a b c d = 126 ∧
    ∀ (x y z w : ℕ),
      x ∈ ({1, 2, 3, 4} : Set ℕ) →
      y ∈ ({1, 2, 3, 4} : Set ℕ) →
      z ∈ ({1, 2, 3, 4} : Set ℕ) →
      w ∈ ({1, 2, 3, 4} : Set ℕ) →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
      expression x y z w ≤ 126 :=
by
  sorry


end NUMINAMATH_CALUDE_max_expression_value_l2809_280963


namespace NUMINAMATH_CALUDE_milk_cost_l2809_280902

/-- If 4 boxes of milk cost 26 yuan, then 6 boxes of the same milk will cost 39 yuan. -/
theorem milk_cost (cost : ℕ) (boxes : ℕ) (h1 : cost = 26) (h2 : boxes = 4) :
  (cost / boxes) * 6 = 39 :=
by sorry

end NUMINAMATH_CALUDE_milk_cost_l2809_280902


namespace NUMINAMATH_CALUDE_gcd_378_90_l2809_280958

theorem gcd_378_90 : Nat.gcd 378 90 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_378_90_l2809_280958


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2809_280929

theorem necessary_but_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x ≤ 1/2 ∧ y ≤ 1/2 → x + y ≤ 1) ∧
  (∃ x y : ℝ, x + y ≤ 1 ∧ ¬(x ≤ 1/2 ∧ y ≤ 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2809_280929


namespace NUMINAMATH_CALUDE_unique_nonnegative_solution_l2809_280922

theorem unique_nonnegative_solution (x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
  x + y + z = 3 * x * y →
  x^2 + y^2 + z^2 = 3 * x * z →
  x^3 + y^3 + z^3 = 3 * y * z →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_nonnegative_solution_l2809_280922


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2809_280944

theorem expand_and_simplify (y : ℝ) : -2 * (5 * y^3 - 4 * y^2 + 3 * y - 6) = -10 * y^3 + 8 * y^2 - 6 * y + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2809_280944


namespace NUMINAMATH_CALUDE_namjoon_used_seven_pencils_l2809_280918

/-- Represents the number of pencils each person has at different stages --/
structure PencilCount where
  initial : Nat
  after_taehyung_gives : Nat
  final : Nat

/-- The problem setup --/
def problem : PencilCount × PencilCount := 
  ({ initial := 10, after_taehyung_gives := 7, final := 6 },  -- Taehyung's pencils
   { initial := 10, after_taehyung_gives := 13, final := 6 }) -- Namjoon's pencils

/-- Calculates the number of pencils Namjoon used --/
def pencils_namjoon_used (p : PencilCount × PencilCount) : Nat :=
  p.2.after_taehyung_gives - p.2.final

/-- Theorem stating that Namjoon used 7 pencils --/
theorem namjoon_used_seven_pencils :
  pencils_namjoon_used problem = 7 := by
  sorry

end NUMINAMATH_CALUDE_namjoon_used_seven_pencils_l2809_280918


namespace NUMINAMATH_CALUDE_total_popsicle_sticks_l2809_280998

/-- The total number of popsicle sticks owned by Gino, you, and Nick is 195. -/
theorem total_popsicle_sticks :
  let gino_sticks : ℕ := 63
  let your_sticks : ℕ := 50
  let nick_sticks : ℕ := 82
  gino_sticks + your_sticks + nick_sticks = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_popsicle_sticks_l2809_280998


namespace NUMINAMATH_CALUDE_possible_x_values_l2809_280994

def M (x : ℝ) : Set ℝ := {-2, 3*x^2 + 3*x - 4, x^2 + x - 4}

theorem possible_x_values (x : ℝ) : 2 ∈ M x → x = 2 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_possible_x_values_l2809_280994


namespace NUMINAMATH_CALUDE_lawn_mowing_time_l2809_280970

/-- Calculates the time required to mow a rectangular lawn -/
theorem lawn_mowing_time 
  (length width : ℝ) 
  (effective_swath : ℝ) 
  (mowing_speed : ℝ) : 
  length = 120 → 
  width = 200 → 
  effective_swath = 2 → 
  mowing_speed = 4000 → 
  (width / effective_swath) * length / mowing_speed = 3 := by
  sorry

#check lawn_mowing_time

end NUMINAMATH_CALUDE_lawn_mowing_time_l2809_280970


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2809_280999

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 → a * b = 0)) ↔ (a * b ≠ 0 → a ≠ 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2809_280999


namespace NUMINAMATH_CALUDE_point_on_600_degree_angle_l2809_280964

/-- Prove that if a point (-4, a) lies on the terminal side of an angle measuring 600°, then a = -4√3. -/
theorem point_on_600_degree_angle (a : ℝ) : 
  (∃ θ : ℝ, θ = 600 * π / 180 ∧ Real.tan θ = a / (-4)) → a = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_600_degree_angle_l2809_280964


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_coincide_l2809_280989

/-- The squared distance from the center to a focus of a hyperbola -/
def hyperbola_c_squared (a b : ℝ) : ℝ := a^2 + b^2

/-- The squared distance from the center to a focus of an ellipse -/
def ellipse_c_squared (a b : ℝ) : ℝ := a^2 - b^2

theorem ellipse_hyperbola_foci_coincide :
  let ellipse_a_squared : ℝ := 16
  let hyperbola_a_squared : ℝ := 144 / 25
  let hyperbola_b_squared : ℝ := 81 / 25
  ∀ b_squared : ℝ,
    hyperbola_c_squared hyperbola_a_squared hyperbola_b_squared =
    ellipse_c_squared ellipse_a_squared b_squared →
    b_squared = 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_coincide_l2809_280989


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l2809_280951

theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 50)
  (h2 : faster_speed = 14)
  (h3 : additional_distance = 20) :
  ∃ x : ℝ, x > 0 ∧ actual_distance / x = (actual_distance + additional_distance) / faster_speed ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l2809_280951


namespace NUMINAMATH_CALUDE_percentage_problem_l2809_280957

theorem percentage_problem (x : ℝ) : x * 0.0005 = 6.178 → x = 12356 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2809_280957


namespace NUMINAMATH_CALUDE_range_of_a_l2809_280980

/-- The range of values for real number a given specific conditions -/
theorem range_of_a (a : ℝ) : 
  (∃ x, x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0) →
  (∃ x, x^2 + 2*x - 8 > 0) →
  (∀ x, (x^2 - 4*a*x + 3*a^2 ≥ 0 ∨ a ≥ 0) → (x^2 + 2*x - 8 ≤ 0)) →
  (∃ x, (x^2 + 2*x - 8 ≤ 0) ∧ (x^2 - 4*a*x + 3*a^2 ≥ 0 ∨ a ≥ 0)) →
  a ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2809_280980


namespace NUMINAMATH_CALUDE_rational_roots_of_polynomial_l2809_280930

theorem rational_roots_of_polynomial (x : ℚ) :
  (4 * x^4 - 3 * x^3 - 13 * x^2 + 5 * x + 2 = 0) ↔ (x = 2 ∨ x = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_of_polynomial_l2809_280930


namespace NUMINAMATH_CALUDE_oranges_per_group_l2809_280916

/-- Given the total number of oranges and the number of orange groups,
    prove that the number of oranges per group is 2. -/
theorem oranges_per_group (total_oranges : ℕ) (orange_groups : ℕ) 
  (h1 : total_oranges = 356) (h2 : orange_groups = 178) :
  total_oranges / orange_groups = 2 := by
  sorry


end NUMINAMATH_CALUDE_oranges_per_group_l2809_280916


namespace NUMINAMATH_CALUDE_burger_cost_is_13_l2809_280903

/-- The cost of a single burger given the conditions of Alice's burger purchases in June. -/
def burger_cost (burgers_per_day : ℕ) (days_in_june : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (burgers_per_day * days_in_june)

/-- Theorem stating that the cost of each burger is 13 dollars under the given conditions. -/
theorem burger_cost_is_13 :
  burger_cost 4 30 1560 = 13 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_is_13_l2809_280903


namespace NUMINAMATH_CALUDE_smaller_square_area_percentage_l2809_280933

/-- A circle with an inscribed square and a smaller square -/
structure CircleWithSquares where
  /-- Radius of the circle -/
  r : ℝ
  /-- Side length of the larger square (inscribed in the circle) -/
  large_side : ℝ
  /-- Side length of the smaller square -/
  small_side : ℝ
  /-- The larger square is inscribed in the circle -/
  large_inscribed : large_side = 2 * r
  /-- The smaller square shares one side with the larger square -/
  shared_side : small_side ≤ large_side
  /-- Two vertices of the smaller square are on the circle -/
  vertices_on_circle : small_side^2 + (large_side/2 + small_side/2)^2 = r^2

/-- The area of the smaller square is 0% of the area of the larger square -/
theorem smaller_square_area_percentage (c : CircleWithSquares) :
  (c.small_side^2) / (c.large_side^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_area_percentage_l2809_280933


namespace NUMINAMATH_CALUDE_megatek_graph_is_pie_chart_l2809_280981

-- Define the properties of the graph
structure EmployeeGraph where
  -- The graph is circular
  isCircular : Bool
  -- The angle of each sector is proportional to the quantity it represents
  isSectorProportional : Bool
  -- The manufacturing sector angle
  manufacturingAngle : ℝ
  -- The percentage of employees in manufacturing
  manufacturingPercentage : ℝ

-- Define a pie chart
def isPieChart (graph : EmployeeGraph) : Prop :=
  graph.isCircular ∧ 
  graph.isSectorProportional ∧
  graph.manufacturingAngle = 144 ∧
  graph.manufacturingPercentage = 40

-- Theorem to prove
theorem megatek_graph_is_pie_chart (graph : EmployeeGraph) 
  (h1 : graph.isCircular = true)
  (h2 : graph.isSectorProportional = true)
  (h3 : graph.manufacturingAngle = 144)
  (h4 : graph.manufacturingPercentage = 40) :
  isPieChart graph :=
sorry

end NUMINAMATH_CALUDE_megatek_graph_is_pie_chart_l2809_280981


namespace NUMINAMATH_CALUDE_quadratic_sum_l2809_280966

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ), 
  (15 * x^2 + 150 * x + 2250 = a * (x + b)^2 + c) ∧ (a + b + c = 1895) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2809_280966


namespace NUMINAMATH_CALUDE_james_initial_milk_l2809_280926

def ounces_drank : ℕ := 13
def ounces_per_gallon : ℕ := 128
def ounces_left : ℕ := 371

def initial_gallons : ℚ :=
  (ounces_left + ounces_drank) / ounces_per_gallon

theorem james_initial_milk : initial_gallons = 3 := by
  sorry

end NUMINAMATH_CALUDE_james_initial_milk_l2809_280926


namespace NUMINAMATH_CALUDE_min_draw_for_red_card_l2809_280965

theorem min_draw_for_red_card (total : ℕ) (blue yellow red : ℕ) :
  total = 20 →
  blue + yellow + red = total →
  blue = yellow / 6 →
  red < yellow →
  15 = total - red + 1 :=
by sorry

end NUMINAMATH_CALUDE_min_draw_for_red_card_l2809_280965


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2809_280995

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 + 2 * a^2 - 5 * a - 15 = 0) ∧
  (3 * b^3 + 2 * b^2 - 5 * b - 15 = 0) ∧
  (3 * c^3 + 2 * c^2 - 5 * c - 15 = 0) →
  a^2 + b^2 + c^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2809_280995


namespace NUMINAMATH_CALUDE_percentage_of_x_l2809_280904

theorem percentage_of_x (x y : ℝ) (h1 : x / y = 4) (h2 : y ≠ 0) :
  (2 * x - y) / x * 100 = 175 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_x_l2809_280904


namespace NUMINAMATH_CALUDE_round_trip_average_speed_river_boat_average_speed_l2809_280906

/-- The average speed of a round trip given upstream and downstream speeds -/
theorem round_trip_average_speed (upstream_speed downstream_speed : ℝ) 
  (upstream_speed_pos : 0 < upstream_speed)
  (downstream_speed_pos : 0 < downstream_speed) :
  (2 * upstream_speed * downstream_speed) / (upstream_speed + downstream_speed) =
  (2 * 6 * 8) / (6 + 8) :=
by sorry

/-- The specific case for the river boat problem -/
theorem river_boat_average_speed :
  (2 * 6 * 8) / (6 + 8) = 48 / 7 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_river_boat_average_speed_l2809_280906


namespace NUMINAMATH_CALUDE_square_difference_l2809_280908

theorem square_difference : (625 : ℤ)^2 - (375 : ℤ)^2 = 250000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2809_280908


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_million_l2809_280919

def T (n : ℕ) : ℕ := n * 2^(n-1)

theorem smallest_n_exceeding_million :
  (∀ k < 20, T k ≤ 10^6) ∧ T 20 > 10^6 := by sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_million_l2809_280919


namespace NUMINAMATH_CALUDE_sum_equals_42_l2809_280911

/-- An increasing geometric sequence with specific properties -/
structure IncreasingGeometricSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  geometric : ∃ r : ℝ, r > 1 ∧ ∀ n, a (n + 1) = r * a n
  sum_condition : a 1 + a 3 + a 5 = 21
  a3_value : a 3 = 6

/-- The sum of specific terms in the sequence equals 42 -/
theorem sum_equals_42 (seq : IncreasingGeometricSequence) : seq.a 5 + seq.a 3 + seq.a 9 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_42_l2809_280911


namespace NUMINAMATH_CALUDE_correct_algebraic_operation_l2809_280932

variable (x y : ℝ)

theorem correct_algebraic_operation : y * x - 3 * x * y = -2 * x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_algebraic_operation_l2809_280932


namespace NUMINAMATH_CALUDE_special_polygon_area_l2809_280925

/-- A polygon with special properties -/
structure SpecialPolygon where
  sides : ℕ
  perimeter : ℝ
  is_decomposable_into_rectangles : Prop
  all_sides_congruent : Prop
  sides_perpendicular : Prop

/-- The area of a special polygon -/
def area (p : SpecialPolygon) : ℝ := sorry

/-- Theorem stating the area of the specific polygon described in the problem -/
theorem special_polygon_area :
  ∀ (p : SpecialPolygon),
    p.sides = 24 ∧
    p.perimeter = 48 ∧
    p.is_decomposable_into_rectangles ∧
    p.all_sides_congruent ∧
    p.sides_perpendicular →
    area p = 32 := by sorry

end NUMINAMATH_CALUDE_special_polygon_area_l2809_280925


namespace NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l2809_280979

def base9ToDecimal (n : Nat) : Nat :=
  (n / 100) * 9^2 + ((n / 10) % 10) * 9 + (n % 10)

theorem greatest_3digit_base9_divisible_by_7 :
  ∃ (n : Nat), 
    n < 1000 ∧ 
    base9ToDecimal n % 7 = 0 ∧
    (∀ m : Nat, m < 1000 → base9ToDecimal m % 7 = 0 → m ≤ n) ∧
    n = 888 := by
  sorry

end NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l2809_280979


namespace NUMINAMATH_CALUDE_rebecca_eggs_l2809_280949

/-- The number of eggs Rebecca has -/
def number_of_eggs : ℕ := 3 * 3

/-- The size of each group of eggs -/
def group_size : ℕ := 3

/-- The number of groups Rebecca created -/
def number_of_groups : ℕ := 3

theorem rebecca_eggs : 
  number_of_eggs = group_size * number_of_groups := by sorry

end NUMINAMATH_CALUDE_rebecca_eggs_l2809_280949


namespace NUMINAMATH_CALUDE_max_travel_distance_proof_l2809_280907

/-- The distance (in km) a tire can travel on the front wheel before wearing out -/
def front_tire_lifespan : ℝ := 25000

/-- The distance (in km) a tire can travel on the rear wheel before wearing out -/
def rear_tire_lifespan : ℝ := 15000

/-- The maximum distance (in km) a motorcycle can travel before its tires are completely worn out,
    given that the tires are exchanged between front and rear wheels at the optimal time -/
def max_travel_distance : ℝ := 18750

/-- Theorem stating that the calculated maximum travel distance is correct -/
theorem max_travel_distance_proof :
  max_travel_distance = (front_tire_lifespan * rear_tire_lifespan) / (front_tire_lifespan / 2 + rear_tire_lifespan / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_travel_distance_proof_l2809_280907


namespace NUMINAMATH_CALUDE_workshop_attendance_l2809_280988

theorem workshop_attendance : 
  ∀ (total wolf_laureates wolf_and_nobel_laureates nobel_laureates : ℕ),
    wolf_laureates = 31 →
    wolf_and_nobel_laureates = 16 →
    nobel_laureates = 27 →
    ∃ (non_wolf_nobel non_wolf_non_nobel : ℕ),
      non_wolf_nobel = nobel_laureates - wolf_and_nobel_laureates ∧
      non_wolf_nobel = non_wolf_non_nobel + 3 ∧
      total = wolf_laureates + non_wolf_nobel + non_wolf_non_nobel →
      total = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_attendance_l2809_280988


namespace NUMINAMATH_CALUDE_intersection_M_N_l2809_280967

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2809_280967


namespace NUMINAMATH_CALUDE_circle_radius_from_tangents_l2809_280950

/-- A circle with two parallel tangents and a third tangent -/
structure CircleWithTangents where
  r : ℝ  -- radius of the circle
  xy : ℝ  -- length of tangent XY
  xpyp : ℝ  -- length of tangent X'Y'

/-- The theorem stating the relationship between the tangents and the radius -/
theorem circle_radius_from_tangents (c : CircleWithTangents) 
  (h1 : c.xy = 7)
  (h2 : c.xpyp = 12) :
  c.r = 4 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_tangents_l2809_280950


namespace NUMINAMATH_CALUDE_f_behavior_l2809_280952

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

def has_min_value (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  ∀ x, a ≤ x → x ≤ b → c ≤ f x

-- State the theorem
theorem f_behavior :
  is_even f →
  increasing_on f 5 7 →
  has_min_value f 5 7 6 →
  decreasing_on f (-7) (-5) ∧ has_min_value f (-7) (-5) 6 :=
sorry

end NUMINAMATH_CALUDE_f_behavior_l2809_280952


namespace NUMINAMATH_CALUDE_middle_term_value_l2809_280954

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i j k : ℕ, i < j → j < k → a j - a i = a k - a j

-- Define our specific sequence
def our_sequence (n : ℕ) : ℝ :=
  match n with
  | 0 => 23
  | 1 => 0  -- x (unknown)
  | 2 => 0  -- y (to be proven)
  | 3 => 0  -- z (unknown)
  | 4 => 47
  | _ => 0  -- other terms are not relevant

-- State the theorem
theorem middle_term_value :
  is_arithmetic_sequence our_sequence →
  our_sequence 2 = 35 :=
by sorry

end NUMINAMATH_CALUDE_middle_term_value_l2809_280954


namespace NUMINAMATH_CALUDE_no_real_solutions_quadratic_inequality_l2809_280986

theorem no_real_solutions_quadratic_inequality :
  ¬∃ (x : ℝ), 3 * x^2 + 9 * x ≤ -12 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_quadratic_inequality_l2809_280986


namespace NUMINAMATH_CALUDE_six_minutes_to_hours_l2809_280990

-- Define the conversion factor from minutes to hours
def minutes_to_hours (minutes : ℚ) : ℚ := minutes / 60

-- State the theorem
theorem six_minutes_to_hours : 
  minutes_to_hours 6 = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_six_minutes_to_hours_l2809_280990


namespace NUMINAMATH_CALUDE_set_intersection_empty_implies_a_range_l2809_280947

theorem set_intersection_empty_implies_a_range (a : ℝ) : 
  let A := {x : ℝ | a - 1 < x ∧ x < 2*a + 1}
  let B := {x : ℝ | 0 < x ∧ x < 1}
  (A ∩ B = ∅) → (a ≤ -1/2 ∨ a ≥ 2) := by
sorry

end NUMINAMATH_CALUDE_set_intersection_empty_implies_a_range_l2809_280947


namespace NUMINAMATH_CALUDE_marbles_given_proof_l2809_280975

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 776 - 593

/-- Connie's initial number of marbles -/
def initial_marbles : ℕ := 776

/-- The number of marbles Connie has left -/
def remaining_marbles : ℕ := 593

theorem marbles_given_proof : 
  marbles_given = initial_marbles - remaining_marbles :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_proof_l2809_280975


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l2809_280976

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem line_parallel_to_intersection_of_parallel_planes
  (a : Line) (α β : Plane) (b : Line)
  (h1 : parallel_line_plane a α)
  (h2 : parallel_line_plane a β)
  (h3 : intersect α β = b) :
  parallel_line_line a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l2809_280976


namespace NUMINAMATH_CALUDE_tan_product_ninth_pi_l2809_280910

theorem tan_product_ninth_pi : Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_ninth_pi_l2809_280910


namespace NUMINAMATH_CALUDE_gcd_547_323_l2809_280968

theorem gcd_547_323 : Nat.gcd 547 323 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_547_323_l2809_280968


namespace NUMINAMATH_CALUDE_expression_value_l2809_280927

theorem expression_value : 
  (7 - (540 : ℚ) / 9) - (5 - (330 : ℚ) * 2 / 11) + (2 - (260 : ℚ) * 3 / 13) = -56 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2809_280927


namespace NUMINAMATH_CALUDE_fraction_value_l2809_280936

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 2 * d) : 
  (a * c) / (b * d) = 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l2809_280936


namespace NUMINAMATH_CALUDE_only_positive_number_l2809_280905

theorem only_positive_number (numbers : Set ℝ) : 
  numbers = {0, 5, -1/2, -Real.sqrt 2} → 
  (∃ x ∈ numbers, x > 0) ∧ (∀ y ∈ numbers, y > 0 → y = 5) := by
sorry

end NUMINAMATH_CALUDE_only_positive_number_l2809_280905


namespace NUMINAMATH_CALUDE_lcm_54_198_l2809_280941

theorem lcm_54_198 : Nat.lcm 54 198 = 594 := by
  sorry

end NUMINAMATH_CALUDE_lcm_54_198_l2809_280941


namespace NUMINAMATH_CALUDE_initial_men_correct_l2809_280939

/-- The initial number of men working on a project -/
def initial_men : ℕ := 15

/-- The number of days to complete the work with the initial group -/
def initial_days : ℕ := 40

/-- The number of men who leave the project -/
def men_leaving : ℕ := 14

/-- The number of days worked before some men leave -/
def days_before_leaving : ℕ := 16

/-- The number of days to complete the remaining work after some men leave -/
def remaining_days : ℕ := 40

/-- Theorem stating that the initial number of men is correct given the conditions -/
theorem initial_men_correct : 
  (initial_men : ℚ) * initial_days * (initial_days - days_before_leaving) = 
  (initial_men - men_leaving) * initial_days * remaining_days :=
sorry

end NUMINAMATH_CALUDE_initial_men_correct_l2809_280939


namespace NUMINAMATH_CALUDE_triangle_property_l2809_280993

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that for a triangle satisfying certain conditions, 
    angle B is 2π/3 and the area is (3√3)/2. -/
theorem triangle_property (t : Triangle) 
  (h1 : t.a * Real.sin t.B + t.b * Real.sin t.A = t.b * Real.sin t.C - t.c * Real.sin t.B)
  (h2 : t.b = Real.sqrt 13)
  (h3 : t.a + t.c = 4) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l2809_280993


namespace NUMINAMATH_CALUDE_expression_evaluation_l2809_280955

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := -2
  (3*x - 2*y)^2 - (2*y + x)*(2*y - x) - 2*x*(5*x - 6*y + x*y) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2809_280955


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l2809_280921

theorem students_playing_neither_sport 
  (total : ℕ) 
  (football : ℕ) 
  (tennis : ℕ) 
  (both : ℕ) 
  (h1 : total = 60) 
  (h2 : football = 36) 
  (h3 : tennis = 30) 
  (h4 : both = 22) : 
  total - (football + tennis - both) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l2809_280921


namespace NUMINAMATH_CALUDE_women_in_room_l2809_280924

theorem women_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (initial_men + 2) = 14 →
  2 * (initial_women - 3) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_women_in_room_l2809_280924


namespace NUMINAMATH_CALUDE_fraction_simplification_l2809_280934

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) (h3 : a ≠ 2) :
  (a + 1) / (a^2 - 1) / ((a^2 - 4) / (a^2 + a - 2)) - (1 - a) / (a - 2) = a / (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2809_280934


namespace NUMINAMATH_CALUDE_difference_of_squares_81_49_l2809_280940

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_81_49_l2809_280940


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2809_280943

theorem constant_term_expansion (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f x = (x + 2) * (1/x - a*x)^7 ∧ 
   ∃ (g : ℝ → ℝ), (∀ x ≠ 0, f x = g x) ∧ g 0 = -280) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2809_280943


namespace NUMINAMATH_CALUDE_find_a_l2809_280978

/-- The value of a that satisfies the given inequality system -/
def a : ℝ := 4

/-- The system of inequalities -/
def inequality_system (x a : ℝ) : Prop :=
  2 * x + 1 > 3 ∧ a - x > 1

/-- The solution set of the inequality system -/
def solution_set (x : ℝ) : Prop :=
  1 < x ∧ x < 3

theorem find_a :
  (∀ x, inequality_system x a ↔ solution_set x) →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_find_a_l2809_280978


namespace NUMINAMATH_CALUDE_time_per_question_l2809_280961

/-- Proves that given a test with 100 questions, where 40 questions are left unanswered
    and 2 hours are spent answering, the time taken for each answered question is 2 minutes. -/
theorem time_per_question (total_questions : Nat) (unanswered_questions : Nat) (time_spent : Nat) :
  total_questions = 100 →
  unanswered_questions = 40 →
  time_spent = 120 →
  (time_spent : ℚ) / ((total_questions - unanswered_questions) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_per_question_l2809_280961


namespace NUMINAMATH_CALUDE_expression_evaluation_l2809_280977

theorem expression_evaluation : -1^2008 + (-1)^2009 + 1^2010 - 1^2011 = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2809_280977


namespace NUMINAMATH_CALUDE_total_evaluations_is_2680_l2809_280914

/-- Represents a class with its exam components and student count -/
structure ExamClass where
  students : ℕ
  multipleChoice : ℕ
  shortAnswer : ℕ
  essay : ℕ
  otherEvaluations : ℕ

/-- Calculates the total evaluations for a single class -/
def classEvaluations (c : ExamClass) : ℕ :=
  c.students * (c.multipleChoice + c.shortAnswer + c.essay) + c.otherEvaluations

/-- The exam classes as defined in the problem -/
def examClasses : List ExamClass := [
  ⟨30, 12, 0, 3, 30⟩,  -- Class A
  ⟨25, 15, 5, 2, 5⟩,   -- Class B
  ⟨35, 10, 0, 3, 5⟩,   -- Class C
  ⟨40, 11, 4, 3, 40⟩,  -- Class D
  ⟨20, 14, 5, 2, 5⟩    -- Class E
]

/-- The theorem stating that the total evaluations equal 2680 -/
theorem total_evaluations_is_2680 :
  (examClasses.map classEvaluations).sum = 2680 := by
  sorry

end NUMINAMATH_CALUDE_total_evaluations_is_2680_l2809_280914


namespace NUMINAMATH_CALUDE_distance_to_FA_l2809_280931

/-- RegularHexagon represents a regular hexagon with a point inside -/
structure RegularHexagon where
  -- Point inside the hexagon
  P : Point
  -- Distances from P to each side
  dist_AB : ℝ
  dist_BC : ℝ
  dist_CD : ℝ
  dist_DE : ℝ
  dist_EF : ℝ
  dist_FA : ℝ

/-- Theorem stating the distance from P to FA in the given hexagon -/
theorem distance_to_FA (h : RegularHexagon)
  (h_AB : h.dist_AB = 1)
  (h_BC : h.dist_BC = 2)
  (h_CD : h.dist_CD = 5)
  (h_DE : h.dist_DE = 7)
  (h_EF : h.dist_EF = 6)
  : h.dist_FA = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_FA_l2809_280931


namespace NUMINAMATH_CALUDE_average_stickers_per_album_l2809_280991

def album_stickers : List ℕ := [5, 7, 9, 14, 19, 12, 26, 18, 11, 15]

theorem average_stickers_per_album :
  (album_stickers.sum : ℚ) / album_stickers.length = 13.6 := by
  sorry

end NUMINAMATH_CALUDE_average_stickers_per_album_l2809_280991


namespace NUMINAMATH_CALUDE_jack_sugar_calculation_l2809_280937

/-- Given Jack's sugar operations, prove the final amount is correct. -/
theorem jack_sugar_calculation (initial : ℕ) (used : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 65 → used = 18 → bought = 50 → final = 97 → 
  final = initial - used + bought :=
by
  sorry

end NUMINAMATH_CALUDE_jack_sugar_calculation_l2809_280937


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2809_280996

/-- The Stewart farm problem -/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
  sheep * 7 = horses * 6 →
  horses * 230 = 12880 →
  sheep = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2809_280996


namespace NUMINAMATH_CALUDE_multiplication_correction_l2809_280948

theorem multiplication_correction (n : ℕ) : 
  n * 987 = 559981 → 
  (∃ a b : ℕ, a ≠ 9 ∧ b ≠ 8 ∧ n * 987 = 5 * 100000 + a * 10000 + b * 1000 + 981) → 
  n * 987 = 559989 :=
by sorry

end NUMINAMATH_CALUDE_multiplication_correction_l2809_280948


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2809_280900

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation that z satisfies
def equation (z : ℂ) : Prop := (-1 + i) * z = (1 + i)^2

-- Define what it means for a complex number to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant :
  ∃ z : ℂ, equation z ∧ in_fourth_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2809_280900


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2809_280909

theorem smallest_integer_with_remainders : 
  ∃ x : ℕ, 
    (x > 0) ∧ 
    (x % 5 = 4) ∧ 
    (x % 6 = 5) ∧ 
    (x % 7 = 6) ∧ 
    (∀ y : ℕ, y > 0 → y % 5 = 4 → y % 6 = 5 → y % 7 = 6 → x ≤ y) ∧
    x = 209 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2809_280909


namespace NUMINAMATH_CALUDE_greatest_triangle_perimeter_l2809_280983

theorem greatest_triangle_perimeter : 
  ∀ a b c : ℕ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (b = 4 * a) →
  (c = 20) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (∀ x y z : ℕ, 
    (x > 0 ∧ y > 0 ∧ z > 0) →
    (y = 4 * x) →
    (z = 20) →
    (x + y > z ∧ y + z > x ∧ z + x > y) →
    (a + b + c ≥ x + y + z)) →
  a + b + c = 50 :=
by sorry

end NUMINAMATH_CALUDE_greatest_triangle_perimeter_l2809_280983


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l2809_280997

/-- The line y = kx - k + 1 intersects the ellipse x²/9 + y²/4 = 1 for all real k -/
theorem line_intersects_ellipse (k : ℝ) : ∃ (x y : ℝ), 
  y = k * x - k + 1 ∧ x^2 / 9 + y^2 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l2809_280997


namespace NUMINAMATH_CALUDE_problem_solution_l2809_280917

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

-- Define M(a) and m(a)
def M (a : ℝ) : ℝ := max (f a 1) (f a 2)
def m (a : ℝ) : ℝ := min (f a 1) (f a 2)

-- Define h(a)
def h (a : ℝ) : ℝ := M a - m a

theorem problem_solution :
  (∀ a : ℝ, f' a 0 = 3 → a = 1/2) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → f a x + f a (-x) ≥ 12 * Real.log x) →
    a ≤ -1 - Real.exp (-1)) ∧
  (∀ a : ℝ, a > 1 →
    (∃ min_h : ℝ, min_h = 8/27 ∧
      ∀ a' : ℝ, a' > 1 → h a' ≥ min_h)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2809_280917
