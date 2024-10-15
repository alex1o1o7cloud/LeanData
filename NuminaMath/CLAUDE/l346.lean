import Mathlib

namespace NUMINAMATH_CALUDE_library_tables_count_l346_34610

/-- The number of pupils that can be seated at a rectangular table -/
def rectangular_table_capacity : ℕ := 10

/-- The number of pupils that can be seated at a square table -/
def square_table_capacity : ℕ := 4

/-- The number of square tables needed in the library -/
def square_tables_needed : ℕ := 5

/-- The total number of pupils that need to be seated -/
def total_pupils : ℕ := 90

/-- The number of rectangular tables in the library -/
def rectangular_tables : ℕ := 7

theorem library_tables_count :
  rectangular_tables * rectangular_table_capacity +
  square_tables_needed * square_table_capacity = total_pupils :=
by sorry

end NUMINAMATH_CALUDE_library_tables_count_l346_34610


namespace NUMINAMATH_CALUDE_max_dot_product_ellipse_l346_34626

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def center : ℝ × ℝ := (0, 0)

noncomputable def left_focus : ℝ × ℝ := (-1, 0)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem max_dot_product_ellipse :
  ∃ (max : ℝ), max = 6 ∧
  ∀ (P : ℝ × ℝ), ellipse P.1 P.2 →
  dot_product (P.1 - center.1, P.2 - center.2) (P.1 - left_focus.1, P.2 - left_focus.2) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_ellipse_l346_34626


namespace NUMINAMATH_CALUDE_time_spent_on_type_a_l346_34666

/-- Represents the time allocation for an exam with three problem types. -/
structure ExamTime where
  totalTime : ℕ  -- Total exam time in minutes
  totalQuestions : ℕ  -- Total number of questions
  typeACount : ℕ  -- Number of Type A problems
  typeBCount : ℕ  -- Number of Type B problems
  typeCCount : ℕ  -- Number of Type C problems
  typeATime : ℚ  -- Time for one Type A problem
  typeBTime : ℚ  -- Time for one Type B problem
  typeCTime : ℚ  -- Time for one Type C problem

/-- Theorem stating the time spent on Type A problems in the given exam scenario. -/
theorem time_spent_on_type_a (exam : ExamTime) : 
  exam.totalTime = 240 ∧ 
  exam.totalQuestions = 300 ∧ 
  exam.typeACount = 25 ∧ 
  exam.typeBCount = 100 ∧ 
  exam.typeCCount = 175 ∧ 
  exam.typeATime = 4 * exam.typeBTime ∧ 
  exam.typeBTime = 2 * exam.typeCTime ∧ 
  exam.typeACount * exam.typeATime + exam.typeBCount * exam.typeBTime = exam.totalTime / 2 
  → exam.typeACount * exam.typeATime = 60 := by
  sorry

end NUMINAMATH_CALUDE_time_spent_on_type_a_l346_34666


namespace NUMINAMATH_CALUDE_northward_distance_l346_34674

/-- Calculates the northward distance given total driving time, speed, and westward distance -/
theorem northward_distance 
  (total_time : ℝ) 
  (speed : ℝ) 
  (westward_distance : ℝ) 
  (h1 : total_time = 6) 
  (h2 : speed = 25) 
  (h3 : westward_distance = 95) : 
  speed * total_time - westward_distance = 55 := by
sorry

end NUMINAMATH_CALUDE_northward_distance_l346_34674


namespace NUMINAMATH_CALUDE_inhabitable_earth_surface_fraction_l346_34675

theorem inhabitable_earth_surface_fraction :
  let total_surface := 1
  let land_fraction := (1 : ℚ) / 3
  let inhabitable_land_fraction := (2 : ℚ) / 3
  land_fraction * inhabitable_land_fraction = (2 : ℚ) / 9 :=
by sorry

end NUMINAMATH_CALUDE_inhabitable_earth_surface_fraction_l346_34675


namespace NUMINAMATH_CALUDE_integer_solutions_system_l346_34614

theorem integer_solutions_system :
  ∀ x y z : ℤ,
  (x^2 - y^2 = z ∧ 3*x*y + (x-y)*z = z^2) →
  ((x = 2 ∧ y = 1 ∧ z = 3) ∨
   (x = 1 ∧ y = 2 ∧ z = -3) ∨
   (x = 1 ∧ y = 0 ∧ z = 1) ∨
   (x = 0 ∧ y = 1 ∧ z = -1) ∨
   (x = 0 ∧ y = 0 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_system_l346_34614


namespace NUMINAMATH_CALUDE_g_2010_equals_342_l346_34681

/-- The function g satisfies the given property for positive integers -/
def g_property (g : ℕ+ → ℕ) : Prop :=
  ∀ (x y m : ℕ+), x + y = 2^(m : ℕ) → g x + g y = 3 * m^2

/-- The main theorem stating that g(2010) = 342 -/
theorem g_2010_equals_342 (g : ℕ+ → ℕ) (h : g_property g) : g 2010 = 342 := by
  sorry

end NUMINAMATH_CALUDE_g_2010_equals_342_l346_34681


namespace NUMINAMATH_CALUDE_inequality_proof_l346_34623

variable (x y z : ℝ)

def condition (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z + x * y + y * z + z * x = x + y + z + 1

theorem inequality_proof (h : condition x y z) :
  (1 / 3) * (Real.sqrt ((1 + x^2) / (1 + x)) + 
             Real.sqrt ((1 + y^2) / (1 + y)) + 
             Real.sqrt ((1 + z^2) / (1 + z))) ≤ ((x + y + z) / 3) ^ (5/8) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l346_34623


namespace NUMINAMATH_CALUDE_max_area_is_10000_l346_34648

/-- Represents a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- The perimeter of the garden is 400 feet -/
def perimeterConstraint (g : Garden) : Prop :=
  2 * g.length + 2 * g.width = 400

/-- The length of the garden is at least 100 feet -/
def lengthConstraint (g : Garden) : Prop :=
  g.length ≥ 100

/-- The width of the garden is at least 50 feet -/
def widthConstraint (g : Garden) : Prop :=
  g.width ≥ 50

/-- The area of the garden -/
def area (g : Garden) : ℝ :=
  g.length * g.width

/-- The maximum area of the garden satisfying all constraints is 10000 square feet -/
theorem max_area_is_10000 :
  ∃ (g : Garden),
    perimeterConstraint g ∧
    lengthConstraint g ∧
    widthConstraint g ∧
    area g = 10000 ∧
    ∀ (g' : Garden),
      perimeterConstraint g' ∧
      lengthConstraint g' ∧
      widthConstraint g' →
      area g' ≤ 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_area_is_10000_l346_34648


namespace NUMINAMATH_CALUDE_coin_machine_possible_amount_l346_34680

theorem coin_machine_possible_amount :
  ∃ (m n p : ℕ), 298 = 5 + 25 * m + 2 * n + 29 * p :=
sorry

end NUMINAMATH_CALUDE_coin_machine_possible_amount_l346_34680


namespace NUMINAMATH_CALUDE_gcd_1785_840_l346_34699

theorem gcd_1785_840 : Nat.gcd 1785 840 = 105 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1785_840_l346_34699


namespace NUMINAMATH_CALUDE_triangle_larger_segment_l346_34682

theorem triangle_larger_segment (a b c h x : ℝ) : 
  a = 30 → b = 70 → c = 80 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 65 :=
by sorry

end NUMINAMATH_CALUDE_triangle_larger_segment_l346_34682


namespace NUMINAMATH_CALUDE_adlai_animal_legs_l346_34624

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The number of dogs Adlai has -/
def adlai_dogs : ℕ := 2

/-- The number of chickens Adlai has -/
def adlai_chickens : ℕ := 1

/-- The total number of animal legs Adlai has -/
def total_legs : ℕ := adlai_dogs * dog_legs + adlai_chickens * chicken_legs

theorem adlai_animal_legs : total_legs = 10 := by
  sorry

end NUMINAMATH_CALUDE_adlai_animal_legs_l346_34624


namespace NUMINAMATH_CALUDE_triangle_angle_c_l346_34638

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_c (t : Triangle) :
  t.a = 1 ∧ t.A = Real.pi / 3 ∧ t.c = Real.sqrt 3 / 3 → t.C = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l346_34638


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l346_34628

/-- The length of the bridge in meters -/
def bridge_length : ℝ := 200

/-- The time it takes for the train to cross the bridge in seconds -/
def bridge_crossing_time : ℝ := 10

/-- The time it takes for the train to pass a lamp post on the bridge in seconds -/
def lamppost_passing_time : ℝ := 5

/-- The length of the train in meters -/
def train_length : ℝ := 200

theorem bridge_length_calculation :
  bridge_length = train_length := by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l346_34628


namespace NUMINAMATH_CALUDE_checkerboard_coverable_iff_even_area_uncoverable_checkerboards_l346_34694

/-- A checkerboard -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- The property of a checkerboard being completely coverable by dominoes -/
def is_coverable (board : Checkerboard) : Prop :=
  (board.rows * board.cols) % 2 = 0

/-- Theorem stating that a checkerboard is coverable iff its area is even -/
theorem checkerboard_coverable_iff_even_area (board : Checkerboard) :
  is_coverable board ↔ Even (board.rows * board.cols) :=
sorry

/-- Function to check if a checkerboard is coverable -/
def check_coverable (board : Checkerboard) : Bool :=
  (board.rows * board.cols) % 2 = 0

/-- Theorem stating which of the given checkerboards are not coverable -/
theorem uncoverable_checkerboards :
  let boards := [
    Checkerboard.mk 4 4,
    Checkerboard.mk 5 5,
    Checkerboard.mk 5 7,
    Checkerboard.mk 6 6,
    Checkerboard.mk 7 3
  ]
  (boards.filter (λ b => ¬check_coverable b)).map (λ b => (b.rows, b.cols)) =
    [(5, 5), (5, 7), (7, 3)] :=
sorry

end NUMINAMATH_CALUDE_checkerboard_coverable_iff_even_area_uncoverable_checkerboards_l346_34694


namespace NUMINAMATH_CALUDE_expression_evaluation_l346_34672

theorem expression_evaluation : 
  (2 + 1/4)^(1/2) - 0.3^0 - 16^(-3/4) = 3/8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l346_34672


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l346_34654

theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, |x - 4| - |x + 5| ≤ m) → m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l346_34654


namespace NUMINAMATH_CALUDE_power_one_sixth_equals_one_l346_34670

def is_greatest_power_of_two_factor (a : ℕ) : Prop :=
  2^a ∣ 180 ∧ ∀ k > a, ¬(2^k ∣ 180)

def is_greatest_power_of_three_factor (b : ℕ) : Prop :=
  3^b ∣ 180 ∧ ∀ k > b, ¬(3^k ∣ 180)

theorem power_one_sixth_equals_one (a b : ℕ) 
  (h1 : is_greatest_power_of_two_factor a) 
  (h2 : is_greatest_power_of_three_factor b) : 
  (1/6 : ℚ)^(b - a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_one_sixth_equals_one_l346_34670


namespace NUMINAMATH_CALUDE_ball_events_properties_l346_34655

-- Define the sample space
def Ω : Type := Fin 8

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A, B, and C
def A : Set Ω := sorry
def B : Set Ω := sorry
def C : Set Ω := sorry

-- Theorem statement
theorem ball_events_properties :
  (P (A ∩ C) = 0) ∧
  (P (A ∩ B) = P A * P B) ∧
  (P (B ∩ C) = P B * P C) := by
  sorry

end NUMINAMATH_CALUDE_ball_events_properties_l346_34655


namespace NUMINAMATH_CALUDE_certain_value_proof_l346_34652

theorem certain_value_proof (n : ℝ) (v : ℝ) (h1 : n = 45) (h2 : (1/3) * n - v = 10) : v = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l346_34652


namespace NUMINAMATH_CALUDE_initial_typists_count_l346_34607

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 20

/-- The number of letters typed by the initial group in 20 minutes -/
def letters_20min : ℕ := 60

/-- The number of typists in the second group -/
def second_group_typists : ℕ := 30

/-- The number of letters typed by the second group in 1 hour -/
def letters_1hour : ℕ := 270

/-- The time ratio between 1 hour and 20 minutes -/
def time_ratio : ℚ := 3

theorem initial_typists_count :
  initial_typists * second_group_typists * letters_20min * time_ratio = letters_1hour * initial_typists :=
by sorry

end NUMINAMATH_CALUDE_initial_typists_count_l346_34607


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l346_34647

theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a → Prime b → Prime c → 
  a * b * c = 399 → 
  2 * (a * b + b * c + c * a) = 422 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l346_34647


namespace NUMINAMATH_CALUDE_square_value_l346_34651

theorem square_value : ∃ (square : ℝ), 
  ((11.2 - 1.2 * square) / 4 + 51.2 * square) * 0.1 = 9.1 ∧ square = 1.568 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l346_34651


namespace NUMINAMATH_CALUDE_melissa_oranges_l346_34635

theorem melissa_oranges (initial_oranges : ℕ) (taken_oranges : ℕ) 
  (h1 : initial_oranges = 70) (h2 : taken_oranges = 19) : 
  initial_oranges - taken_oranges = 51 := by
  sorry

end NUMINAMATH_CALUDE_melissa_oranges_l346_34635


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l346_34613

theorem quadratic_roots_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c < 0) :
  ∃ (x₁ x₂ : ℝ), 
    (a * x₁^2 + b * x₁ + c = 0) ∧ 
    (a * x₂^2 + b * x₂ + c = 0) ∧ 
    (x₁ > 0) ∧ 
    (x₂ < 0) ∧ 
    (|x₂| > |x₁|) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l346_34613


namespace NUMINAMATH_CALUDE_parallel_line_equation_l346_34649

/-- The equation of a line passing through (3, 2) and parallel to 4x + y - 2 = 0 -/
theorem parallel_line_equation : 
  ∀ (x y : ℝ), 
  (∃ (m b : ℝ), 
    -- The line passes through (3, 2)
    2 = m * 3 + b ∧ 
    -- The line is parallel to 4x + y - 2 = 0
    m = -4 ∧ 
    -- The equation of the line
    y = m * x + b) 
  ↔ 
  -- The resulting equation
  4 * x + y - 14 = 0 := by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l346_34649


namespace NUMINAMATH_CALUDE_cube_coplanar_probability_l346_34664

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of vertices we choose -/
def chosen_vertices : ℕ := 4

/-- The number of ways to choose 4 vertices that lie in the same plane -/
def coplanar_choices : ℕ := 12

/-- The total number of ways to choose 4 vertices from 8 -/
def total_choices : ℕ := Nat.choose cube_vertices chosen_vertices

/-- The probability that 4 randomly chosen vertices of a cube lie in the same plane -/
theorem cube_coplanar_probability : 
  (coplanar_choices : ℚ) / total_choices = 6 / 35 := by sorry

end NUMINAMATH_CALUDE_cube_coplanar_probability_l346_34664


namespace NUMINAMATH_CALUDE_abs_z_squared_l346_34668

-- Define a complex number z
variable (z : ℂ)

-- State the theorem
theorem abs_z_squared (h : z + Complex.abs z = 2 + 8*I) : Complex.abs z ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_squared_l346_34668


namespace NUMINAMATH_CALUDE_joy_tonight_outcomes_l346_34644

/-- The number of letters in mailbox A -/
def mailbox_A : Nat := 30

/-- The number of letters in mailbox B -/
def mailbox_B : Nat := 20

/-- The total number of different outcomes for selecting a lucky star and two lucky partners -/
def total_outcomes : Nat := mailbox_A * (mailbox_A - 1) * mailbox_B + mailbox_B * (mailbox_B - 1) * mailbox_A

theorem joy_tonight_outcomes : total_outcomes = 28800 := by
  sorry

end NUMINAMATH_CALUDE_joy_tonight_outcomes_l346_34644


namespace NUMINAMATH_CALUDE_orange_trees_remaining_fruit_l346_34689

theorem orange_trees_remaining_fruit (num_trees : ℕ) (fruits_per_tree : ℕ) (fraction_picked : ℚ) : 
  num_trees = 8 → 
  fruits_per_tree = 200 → 
  fraction_picked = 2/5 → 
  (num_trees * fruits_per_tree) - (num_trees * fruits_per_tree * fraction_picked) = 960 := by
sorry

end NUMINAMATH_CALUDE_orange_trees_remaining_fruit_l346_34689


namespace NUMINAMATH_CALUDE_triangle_cut_theorem_l346_34616

theorem triangle_cut_theorem : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → 
    ((9 - y : ℤ) + (12 - y) ≤ (20 - y) ∧
     (9 - y : ℤ) + (20 - y) ≤ (12 - y) ∧
     (12 - y : ℤ) + (20 - y) ≤ (9 - y)) → 
    y ≥ x) ∧
  (9 - x : ℤ) + (12 - x) ≤ (20 - x) ∧
  (9 - x : ℤ) + (20 - x) ≤ (12 - x) ∧
  (12 - x : ℤ) + (20 - x) ≤ (9 - x) ∧
  x = 17 :=
by sorry

end NUMINAMATH_CALUDE_triangle_cut_theorem_l346_34616


namespace NUMINAMATH_CALUDE_lowest_class_size_l346_34696

theorem lowest_class_size (n : ℕ) : n > 0 ∧ 12 ∣ n ∧ 24 ∣ n → n ≥ 24 :=
by sorry

end NUMINAMATH_CALUDE_lowest_class_size_l346_34696


namespace NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_l346_34656

theorem sum_of_real_and_imag_parts : ∃ (z : ℂ), z = (Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I)) ∧ z.re + z.im = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_l346_34656


namespace NUMINAMATH_CALUDE_rationalize_denominator_l346_34620

theorem rationalize_denominator : 35 / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l346_34620


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l346_34605

/-- Given a square with diagonal length 14√2 cm, its area is 196 cm² -/
theorem square_area_from_diagonal : ∀ s : ℝ,
  s > 0 →
  s * s * 2 = (14 * Real.sqrt 2) ^ 2 →
  s * s = 196 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l346_34605


namespace NUMINAMATH_CALUDE_half_day_division_ways_l346_34698

/-- The number of ways to express 43200 as a product of two positive integers -/
def num_factor_pairs : ℕ := 72

/-- Half a day in seconds -/
def half_day_seconds : ℕ := 43200

theorem half_day_division_ways :
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = half_day_seconds) (Finset.product (Finset.range (half_day_seconds + 1)) (Finset.range (half_day_seconds + 1)))).card = num_factor_pairs :=
sorry

end NUMINAMATH_CALUDE_half_day_division_ways_l346_34698


namespace NUMINAMATH_CALUDE_total_interest_earned_l346_34684

def initial_investment : ℝ := 2000
def interest_rate : ℝ := 0.12
def time_period : ℕ := 4

theorem total_interest_earned :
  let final_amount := initial_investment * (1 + interest_rate) ^ time_period
  final_amount - initial_investment = 1147.04 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_earned_l346_34684


namespace NUMINAMATH_CALUDE_period_2_gym_class_size_l346_34650

theorem period_2_gym_class_size : ℕ → Prop :=
  fun x => (2 * x - 5 = 11) → x = 8

#check period_2_gym_class_size

end NUMINAMATH_CALUDE_period_2_gym_class_size_l346_34650


namespace NUMINAMATH_CALUDE_base_10_648_equals_base_7_1614_l346_34637

/-- Converts a base-10 integer to its representation in base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to its decimal representation --/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldr (λ d acc => d + 7 * acc) 0

theorem base_10_648_equals_base_7_1614 :
  fromBase7 [4, 1, 6, 1] = 648 :=
by sorry

end NUMINAMATH_CALUDE_base_10_648_equals_base_7_1614_l346_34637


namespace NUMINAMATH_CALUDE_ken_steak_change_l346_34676

/-- Calculates the change Ken will receive when buying steak -/
def calculate_change (price_per_pound : ℕ) (pounds_bought : ℕ) (payment : ℕ) : ℕ :=
  payment - (price_per_pound * pounds_bought)

/-- Proves that Ken will receive $6 in change -/
theorem ken_steak_change :
  let price_per_pound : ℕ := 7
  let pounds_bought : ℕ := 2
  let payment : ℕ := 20
  calculate_change price_per_pound pounds_bought payment = 6 := by
sorry

end NUMINAMATH_CALUDE_ken_steak_change_l346_34676


namespace NUMINAMATH_CALUDE_log_equation_solution_l346_34686

theorem log_equation_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq2 : q ≠ 2) :
  Real.log p + Real.log q = Real.log (2 * (p + q)) → p = (2 * (q - 1)) / (q - 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l346_34686


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l346_34646

theorem isosceles_triangle_base_angle (apex_angle : ℝ) (base_angle : ℝ) :
  apex_angle = 100 → -- The apex angle is 100°
  apex_angle + 2 * base_angle = 180 → -- Sum of angles in a triangle is 180°
  base_angle = 40 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l346_34646


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_formula_arithmetic_sequence_possible_formulas_l346_34608

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_unique_formula 
  (a₁ d : ℤ) 
  (h1 : arithmetic_sequence a₁ d 11 = 0) 
  (h2 : sum_arithmetic_sequence a₁ d 14 = 98) :
  ∀ n : ℕ, arithmetic_sequence a₁ d n = 22 - 2 * n :=
sorry

theorem arithmetic_sequence_possible_formulas 
  (a₁ d : ℤ) 
  (h1 : a₁ ≥ 6) 
  (h2 : arithmetic_sequence a₁ d 11 > 0) 
  (h3 : sum_arithmetic_sequence a₁ d 14 ≤ 77) :
  (∀ n : ℕ, arithmetic_sequence a₁ d n = 12 - n) ∨ 
  (∀ n : ℕ, arithmetic_sequence a₁ d n = 13 - n) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_formula_arithmetic_sequence_possible_formulas_l346_34608


namespace NUMINAMATH_CALUDE_triangle_cosine_proof_l346_34622

/-- Given a triangle ABC with A = 2B, a = 6, and b = 4, prove that cos B = 3/4 -/
theorem triangle_cosine_proof (A B C : ℝ) (a b c : ℝ) : 
  A = 2 * B → 
  a = 6 → 
  b = 4 → 
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  Real.cos B = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_proof_l346_34622


namespace NUMINAMATH_CALUDE_three_power_gt_cube_l346_34673

theorem three_power_gt_cube (n : ℕ) (h : n ≠ 3) : 3^n > n^3 := by
  sorry

end NUMINAMATH_CALUDE_three_power_gt_cube_l346_34673


namespace NUMINAMATH_CALUDE_larger_tv_diagonal_l346_34636

theorem larger_tv_diagonal (d : ℝ) : 
  d > 0 → 
  (d / Real.sqrt 2) ^ 2 = (25 / Real.sqrt 2) ^ 2 + 79.5 → 
  d = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_larger_tv_diagonal_l346_34636


namespace NUMINAMATH_CALUDE_q_factor_change_l346_34695

theorem q_factor_change (w h z : ℝ) (q : ℝ → ℝ → ℝ → ℝ) 
  (hq : q w h z = 5 * w / (4 * h * z^2)) :
  q (4*w) (2*h) (3*z) = (2/9) * q w h z := by
sorry

end NUMINAMATH_CALUDE_q_factor_change_l346_34695


namespace NUMINAMATH_CALUDE_max_sum_cubes_l346_34690

theorem max_sum_cubes (a b c d : ℝ) 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 16)
  (distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d) :
  a^3 + b^3 + c^3 + d^3 ≤ 64 ∧ 
  ∃ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 16 ∧ 
                   x ≠ y ∧ y ≠ z ∧ z ≠ w ∧
                   x^3 + y^3 + z^3 + w^3 = 64 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l346_34690


namespace NUMINAMATH_CALUDE_train_length_l346_34658

theorem train_length (crossing_time : ℝ) (speed_kmh : ℝ) : 
  crossing_time = 20 → speed_kmh = 36 → 
  (speed_kmh * 1000 / 3600) * crossing_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l346_34658


namespace NUMINAMATH_CALUDE_tangent_line_sin_at_pi_l346_34615

theorem tangent_line_sin_at_pi (x y : ℝ) :
  let f : ℝ → ℝ := λ t => Real.sin t
  let f' : ℝ → ℝ := λ t => Real.cos t
  let tangent_point : ℝ × ℝ := (π, 0)
  let slope : ℝ := f' tangent_point.1
  x + y - π = 0 ↔ y - tangent_point.2 = slope * (x - tangent_point.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_sin_at_pi_l346_34615


namespace NUMINAMATH_CALUDE_boat_license_plates_l346_34691

def letter_choices : ℕ := 3
def digit_choices : ℕ := 10
def num_digits : ℕ := 4

theorem boat_license_plates :
  letter_choices * digit_choices^num_digits = 30000 :=
sorry

end NUMINAMATH_CALUDE_boat_license_plates_l346_34691


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l346_34600

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 - 4*x - 5 = 0} = {-1, 5} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l346_34600


namespace NUMINAMATH_CALUDE_equation_solution_l346_34643

theorem equation_solution (x y z : ℝ) 
  (eq1 : 4*x - 5*y - z = 0)
  (eq2 : x + 5*y - 18*z = 0)
  (h : z ≠ 0) :
  (x^2 + 4*x*y) / (y^2 + z^2) = 3622 / 9256 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l346_34643


namespace NUMINAMATH_CALUDE_dog_drying_time_l346_34627

/-- Time to dry a short-haired dog in minutes -/
def short_hair_time : ℕ := 10

/-- Time to dry a full-haired dog in minutes -/
def full_hair_time : ℕ := 2 * short_hair_time

/-- Number of short-haired dogs -/
def num_short_hair : ℕ := 6

/-- Number of full-haired dogs -/
def num_full_hair : ℕ := 9

/-- Total time to dry all dogs in hours -/
def total_time_hours : ℚ := (num_short_hair * short_hair_time + num_full_hair * full_hair_time) / 60

theorem dog_drying_time : total_time_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_dog_drying_time_l346_34627


namespace NUMINAMATH_CALUDE_decimal_difference_l346_34653

theorem decimal_difference : (8.1 : ℝ) - (8.01 : ℝ) ≠ 0.1 := by sorry

end NUMINAMATH_CALUDE_decimal_difference_l346_34653


namespace NUMINAMATH_CALUDE_subtraction_in_base_8_l346_34667

def base_8_to_decimal (n : ℕ) : ℕ := sorry

def decimal_to_base_8 (n : ℕ) : ℕ := sorry

theorem subtraction_in_base_8 :
  decimal_to_base_8 (base_8_to_decimal 2101 - base_8_to_decimal 1245) = 634 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_in_base_8_l346_34667


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l346_34661

theorem repeating_decimal_subtraction : 
  ∃ (a b c : ℚ), 
    (∀ n : ℕ, a = (234 / 10^3 + 234 / (10^3 * (1000^n - 1)))) ∧
    (∀ n : ℕ, b = (567 / 10^3 + 567 / (10^3 * (1000^n - 1)))) ∧
    (∀ n : ℕ, c = (891 / 10^3 + 891 / (10^3 * (1000^n - 1)))) ∧
    a - b - c = -408 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l346_34661


namespace NUMINAMATH_CALUDE_stratified_sample_medium_supermarkets_l346_34619

/-- Given a population of supermarkets with the following properties:
  * total_supermarkets: The total number of supermarkets
  * medium_supermarkets: The number of medium-sized supermarkets
  * sample_size: The size of the stratified sample to be taken
  
  This theorem proves that the number of medium-sized supermarkets
  to be selected in the sample is equal to the expected value. -/
theorem stratified_sample_medium_supermarkets
  (total_supermarkets : ℕ)
  (medium_supermarkets : ℕ)
  (sample_size : ℕ)
  (h1 : total_supermarkets = 2000)
  (h2 : medium_supermarkets = 400)
  (h3 : sample_size = 100)
  : (medium_supermarkets * sample_size) / total_supermarkets = 20 := by
  sorry

#check stratified_sample_medium_supermarkets

end NUMINAMATH_CALUDE_stratified_sample_medium_supermarkets_l346_34619


namespace NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l346_34685

/-- Proves that the total amount of ethanol in a fuel tank is 30 gallons given specific conditions -/
theorem ethanol_in_fuel_tank (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) 
  (h1 : tank_capacity = 218)
  (h2 : fuel_a_volume = 122)
  (h3 : fuel_a_ethanol_percent = 0.12)
  (h4 : fuel_b_ethanol_percent = 0.16) :
  fuel_a_volume * fuel_a_ethanol_percent + (tank_capacity - fuel_a_volume) * fuel_b_ethanol_percent = 30 := by
  sorry

end NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l346_34685


namespace NUMINAMATH_CALUDE_devin_teaching_years_difference_l346_34618

/-- Proves that Devin has been teaching for 5 years less than half of Tom's teaching years -/
theorem devin_teaching_years_difference (total_years devin_years tom_years : ℕ) : 
  total_years = 70 → tom_years = 50 → devin_years = total_years - tom_years → 
  tom_years / 2 - devin_years = 5 := by
  sorry

end NUMINAMATH_CALUDE_devin_teaching_years_difference_l346_34618


namespace NUMINAMATH_CALUDE_f_monotone_and_max_a_l346_34659

noncomputable def f (a x : ℝ) : ℝ := (x - a - 1) * Real.exp (x - 1) - (1/2) * x^2 + a * x

theorem f_monotone_and_max_a :
  (∀ x y : ℝ, x < y → f 1 x < f 1 y) ∧
  (∃ a : ℝ, a = Real.exp 1 / 2 - 1 ∧
    (∀ b : ℝ, (∃ x : ℝ, x > 0 ∧ f b x = -1/2) →
      (∀ y : ℝ, y > 0 → f b y ≥ -1/2) →
      b ≤ a)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_and_max_a_l346_34659


namespace NUMINAMATH_CALUDE_percentage_of_men_employees_l346_34629

theorem percentage_of_men_employees (men_attendance : ℝ) (women_attendance : ℝ) (total_attendance : ℝ) :
  men_attendance = 0.2 →
  women_attendance = 0.4 →
  total_attendance = 0.34 →
  ∃ (men_percentage : ℝ),
    men_percentage + (1 - men_percentage) = 1 ∧
    men_attendance * men_percentage + women_attendance * (1 - men_percentage) = total_attendance ∧
    men_percentage = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_men_employees_l346_34629


namespace NUMINAMATH_CALUDE_banana_problem_solution_l346_34603

/-- Represents the banana purchase and sale problem --/
def banana_problem (purchase_pounds : ℚ) (purchase_price : ℚ) 
                   (sale_pounds : ℚ) (sale_price : ℚ) 
                   (profit : ℚ) (total_pounds : ℚ) : Prop :=
  -- Cost price per pound
  let cp_per_pound := purchase_price / purchase_pounds
  -- Selling price per pound
  let sp_per_pound := sale_price / sale_pounds
  -- Total cost
  let total_cost := total_pounds * cp_per_pound
  -- Total revenue
  let total_revenue := total_pounds * sp_per_pound
  -- Profit calculation
  (total_revenue - total_cost = profit) ∧
  -- Ensure the total pounds is positive
  (total_pounds > 0)

/-- Theorem stating the solution to the banana problem --/
theorem banana_problem_solution :
  banana_problem 3 0.5 4 1 6 432 := by
  sorry

end NUMINAMATH_CALUDE_banana_problem_solution_l346_34603


namespace NUMINAMATH_CALUDE_f_not_tangent_to_x_axis_max_a_for_monotone_g_l346_34602

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x - (a / 2) * x^2

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - a * x

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2 * x

-- Define the derivative of g(x)
def g_deriv (a : ℝ) (x : ℝ) : ℝ := f_deriv a x + 2

-- Theorem 1: f(x) cannot be tangent to the x-axis for any a
theorem f_not_tangent_to_x_axis (a : ℝ) : ¬∃ x : ℝ, f a x = 0 ∧ f_deriv a x = 0 := by
  sorry

-- Theorem 2: The maximum integer value of a for which g(x) is monotonically increasing is 1
theorem max_a_for_monotone_g : 
  ∀ a : ℤ, (∀ x : ℝ, g_deriv a x ≥ 0) → a ≤ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_f_not_tangent_to_x_axis_max_a_for_monotone_g_l346_34602


namespace NUMINAMATH_CALUDE_common_remainder_problem_l346_34601

theorem common_remainder_problem (n : ℕ) : 
  n > 1 ∧ 
  n % 25 = n % 7 ∧ 
  n = 175 ∧ 
  ∀ m : ℕ, (m > 1 ∧ m % 25 = m % 7) → m ≥ n → 
  n % 25 = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_remainder_problem_l346_34601


namespace NUMINAMATH_CALUDE_billys_restaurant_bill_l346_34625

/-- Calculates the total bill for a group at Billy's Restaurant -/
theorem billys_restaurant_bill (adults children meal_cost : ℕ) : 
  adults = 2 → children = 5 → meal_cost = 3 → 
  (adults + children) * meal_cost = 21 := by
sorry

end NUMINAMATH_CALUDE_billys_restaurant_bill_l346_34625


namespace NUMINAMATH_CALUDE_remainder_n_cubed_plus_three_l346_34617

theorem remainder_n_cubed_plus_three (n : ℕ) (h : n > 2) :
  (n^3 + 3) % (n + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_n_cubed_plus_three_l346_34617


namespace NUMINAMATH_CALUDE_max_crayfish_revenue_l346_34663

/-- The revenue function for selling crayfish -/
def revenue (total : ℕ) (sold : ℕ) : ℝ :=
  (total - sold : ℝ) * ((total - sold : ℝ) - 4.5) * sold

/-- The statement that proves the maximum revenue and number of crayfish sold -/
theorem max_crayfish_revenue :
  let total := 32
  ∃ (max_sold : ℕ) (max_revenue : ℝ),
    max_sold = 14 ∧
    max_revenue = 189 ∧
    ∀ (sold : ℕ), sold ≤ total → revenue total sold ≤ max_revenue :=
by sorry

end NUMINAMATH_CALUDE_max_crayfish_revenue_l346_34663


namespace NUMINAMATH_CALUDE_cos_alpha_value_l346_34697

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 1/3) 
  (h2 : Real.pi/2 ≤ α) 
  (h3 : α ≤ Real.pi) : 
  Real.cos α = -2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l346_34697


namespace NUMINAMATH_CALUDE_max_area_rectangle_max_area_520_perimeter_l346_34609

/-- The perimeter of the rectangle in meters -/
def perimeter : ℝ := 520

/-- Theorem: Maximum area of a rectangle with given perimeter -/
theorem max_area_rectangle (l w : ℝ) (h1 : l > 0) (h2 : w > 0) (h3 : 2 * l + 2 * w = perimeter) :
  l * w ≤ (perimeter / 4) ^ 2 :=
sorry

/-- Corollary: The maximum area of a rectangle with perimeter 520 meters is 16900 square meters -/
theorem max_area_520_perimeter :
  ∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * l + 2 * w = perimeter ∧ l * w = 16900 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_max_area_520_perimeter_l346_34609


namespace NUMINAMATH_CALUDE_not_equal_to_seven_thirds_l346_34611

theorem not_equal_to_seven_thirds : ∃ x, x ≠ 7/3 ∧ 
  (x = 3 + 1/9) ∧ 
  (14/6 = 7/3) ∧ 
  (2 + 1/3 = 7/3) ∧ 
  (2 + 4/12 = 7/3) := by
  sorry

end NUMINAMATH_CALUDE_not_equal_to_seven_thirds_l346_34611


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l346_34688

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l346_34688


namespace NUMINAMATH_CALUDE_apartment_length_l346_34645

/-- Proves that the length of an apartment with given specifications is 16 feet -/
theorem apartment_length : 
  ∀ (width : ℝ) (total_rooms : ℕ) (living_room_size : ℝ),
    width = 10 →
    total_rooms = 6 →
    living_room_size = 60 →
    ∃ (room_size : ℝ),
      room_size = living_room_size / 3 ∧
      width * 16 = living_room_size + (total_rooms - 1) * room_size :=
by sorry

end NUMINAMATH_CALUDE_apartment_length_l346_34645


namespace NUMINAMATH_CALUDE_spade_combination_l346_34634

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_combination : spade 5 (spade 2 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spade_combination_l346_34634


namespace NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l346_34633

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function g is even if g(-x) = g(x) for all x in its domain -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

/-- The product of an odd function and the absolute value of an even function is odd -/
theorem odd_times_abs_even_is_odd (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) :
  IsOdd (fun x ↦ f x * |g x|) := by sorry

end NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l346_34633


namespace NUMINAMATH_CALUDE_hyperbola_and_angle_bisector_l346_34660

-- Define the hyperbola Γ
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the line l
def line (x y : ℝ) : Prop :=
  x + y - 2 = 0

-- Define that l is parallel to one of the asymptotes and passes through a focus
def line_properties (a b : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), line x₀ y₀ ∧ 
  ((x₀ = a ∧ y₀ = 0) ∨ (x₀ = -a ∧ y₀ = 0)) ∧
  (∀ x y : ℝ, line x y → y = x ∨ y = -x)

-- Main theorem
theorem hyperbola_and_angle_bisector 
  (a b : ℝ) 
  (h : line_properties a b) :
  (∀ x y : ℝ, hyperbola a b x y ↔ x^2 - y^2 = 2) ∧
  (∃ P : ℝ × ℝ, 
    hyperbola a b P.1 P.2 ∧ 
    line P.1 P.2 ∧
    ∀ x y : ℝ, 3*x - y - 4 = 0 ↔ 
      (∃ t : ℝ, x = t*P.1 + (1-t)*(-2) ∧ y = t*P.2) ∨
      (∃ t : ℝ, x = t*P.1 + (1-t)*2 ∧ y = t*P.2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_angle_bisector_l346_34660


namespace NUMINAMATH_CALUDE_gcd_9009_13860_l346_34631

theorem gcd_9009_13860 : Nat.gcd 9009 13860 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9009_13860_l346_34631


namespace NUMINAMATH_CALUDE_equation_solution_l346_34679

theorem equation_solution (x y : ℝ) 
  (h : |x - Real.log y| + Real.sin (π * x) = x + Real.log y) : 
  x = 0 ∧ Real.exp (-1/2) ≤ y ∧ y ≤ Real.exp (1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l346_34679


namespace NUMINAMATH_CALUDE_oranges_remaining_proof_l346_34693

/-- The number of oranges Michaela needs to eat to get full -/
def michaela_oranges : ℕ := 20

/-- The number of oranges Cassandra needs to eat to get full -/
def cassandra_oranges : ℕ := 2 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 90

/-- The number of oranges remaining after both Michaela and Cassandra eat until they're full -/
def remaining_oranges : ℕ := total_oranges - (michaela_oranges + cassandra_oranges)

theorem oranges_remaining_proof : remaining_oranges = 30 := by
  sorry

end NUMINAMATH_CALUDE_oranges_remaining_proof_l346_34693


namespace NUMINAMATH_CALUDE_tangent_through_origin_l346_34692

/-- Given a curve y = x^a + 1 where a is a real number,
    if the tangent line to this curve at the point (1, 2) passes through the origin,
    then a = 2. -/
theorem tangent_through_origin (a : ℝ) : 
  (∀ x y : ℝ, y = x^a + 1) →
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * (x - 1) + 2 ∧ y = m * x + b) →
  (0 = 0 * 0 + b) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_through_origin_l346_34692


namespace NUMINAMATH_CALUDE_binary_1100_is_12_l346_34687

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1100_is_12 :
  binary_to_decimal [false, false, true, true] = 12 := by
  sorry

end NUMINAMATH_CALUDE_binary_1100_is_12_l346_34687


namespace NUMINAMATH_CALUDE_cube_root_of_product_powers_l346_34630

theorem cube_root_of_product_powers (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (m^4 * n^4)^(1/3) = (m * n)^(4/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_product_powers_l346_34630


namespace NUMINAMATH_CALUDE_proposition_analysis_l346_34612

theorem proposition_analysis :
  (¬(∀ x y a : ℝ, 1 < y ∧ y < x ∧ 0 < a ∧ a < 1 → a^(1/x) < a^(1/y))) ∧
  (∀ x y a : ℝ, 1 < y ∧ y < x ∧ a < 0 → x^a < y^a) ∧
  ((¬(∀ x y a : ℝ, 1 < y ∧ y < x ∧ 0 < a ∧ a < 1 → a^(1/x) < a^(1/y))) ∨
   (∀ x y a : ℝ, 1 < y ∧ y < x ∧ a < 0 → x^a < y^a)) ∧
  (¬(∀ x y a : ℝ, 1 < y ∧ y < x ∧ 0 < a ∧ a < 1 → a^(1/x) < a^(1/y))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_analysis_l346_34612


namespace NUMINAMATH_CALUDE_max_a_for_monotone_cubic_l346_34621

/-- Given a > 0 and f(x) = x^3 - ax is monotonically increasing on [1, +∞),
    the maximum value of a is 3. -/
theorem max_a_for_monotone_cubic (a : ℝ) (h1 : a > 0) :
  (∀ x ≥ 1, Monotone (fun x => x^3 - a*x)) →
  a ≤ 3 ∧ ∀ ε > 0, ∃ x ≥ 1, ¬Monotone (fun x => x^3 - (3 + ε)*x) := by
  sorry

end NUMINAMATH_CALUDE_max_a_for_monotone_cubic_l346_34621


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l346_34677

theorem modular_congruence_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 27 ∧ n ≡ -3456 [ZMOD 28] ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l346_34677


namespace NUMINAMATH_CALUDE_basketball_free_throws_count_l346_34669

/-- Represents the scoring of a basketball team -/
structure BasketballScore where
  two_points : ℕ
  three_points : ℕ
  free_throws : ℕ

/-- Calculates the total score of a basketball team -/
def total_score (score : BasketballScore) : ℕ :=
  2 * score.two_points + 3 * score.three_points + score.free_throws

/-- Theorem: Given the conditions, the number of free throws is 12 -/
theorem basketball_free_throws_count 
  (score : BasketballScore) 
  (h1 : 3 * score.three_points = 2 * (2 * score.two_points))
  (h2 : score.free_throws = score.two_points + 1)
  (h3 : total_score score = 79) : 
  score.free_throws = 12 := by
  sorry

#check basketball_free_throws_count

end NUMINAMATH_CALUDE_basketball_free_throws_count_l346_34669


namespace NUMINAMATH_CALUDE_impossible_digit_assignment_l346_34632

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  vertices : ℕ
  sides_eq : sides = n
  vertices_eq : vertices = n

/-- Assignment of digits to vertices -/
def DigitAssignment (n : ℕ) := Fin n → Fin 10

/-- Predicate to check if an assignment satisfies the condition -/
def SatisfiesCondition (n : ℕ) (assignment : DigitAssignment n) : Prop :=
  ∀ (i j : Fin 10), i ≠ j → 
    ∃ (v w : Fin n), v.val + 1 = w.val ∨ (v.val = n - 1 ∧ w.val = 0) ∧ 
      assignment v = i ∧ assignment w = j

theorem impossible_digit_assignment :
  ¬ ∃ (assignment : DigitAssignment 45), SatisfiesCondition 45 assignment := by
  sorry

end NUMINAMATH_CALUDE_impossible_digit_assignment_l346_34632


namespace NUMINAMATH_CALUDE_square_sum_xy_l346_34671

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 30) 
  (h2 : y * (x + y) = 60) : 
  (x + y)^2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l346_34671


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l346_34662

/-- An ellipse with axes parallel to the coordinate axes, tangent to the x-axis at (6, 0) and tangent to the y-axis at (0, 2) -/
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ -- semi-major axis
  b : ℝ -- semi-minor axis
  h_tangent_x : center.1 - a = 6
  h_tangent_y : center.2 - b = 2
  h_a_gt_b : a > b
  h_positive : a > 0 ∧ b > 0

/-- The distance between the foci of the ellipse is 8√2 -/
theorem ellipse_foci_distance (e : Ellipse) : Real.sqrt 128 = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l346_34662


namespace NUMINAMATH_CALUDE_square_sum_identity_l346_34642

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l346_34642


namespace NUMINAMATH_CALUDE_total_revenue_equals_4452_4_l346_34640

def calculate_revenue (price : ℝ) (quantity : ℕ) (discount : ℝ) (tax : ℝ) (surcharge : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  let final_price := taxed_price * (1 + surcharge)
  final_price * quantity

def total_revenue : ℝ :=
  calculate_revenue 25 60 0.1 0.05 0 +
  calculate_revenue 25 10 0 0 0.03 +
  calculate_revenue 25 20 0.05 0.02 0 +
  calculate_revenue 25 44 0.15 0 0.04 +
  calculate_revenue 25 66 0.2 0 0

theorem total_revenue_equals_4452_4 :
  total_revenue = 4452.4 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_equals_4452_4_l346_34640


namespace NUMINAMATH_CALUDE_sheela_bank_deposit_l346_34606

theorem sheela_bank_deposit (monthly_income : ℝ) (deposit_percentage : ℝ) (deposit_amount : ℝ) :
  monthly_income = 11875 →
  deposit_percentage = 32 →
  deposit_amount = (deposit_percentage / 100) * monthly_income →
  deposit_amount = 3796 := by
  sorry

end NUMINAMATH_CALUDE_sheela_bank_deposit_l346_34606


namespace NUMINAMATH_CALUDE_new_average_weight_l346_34641

/-- Given a group of students and a new student joining, calculate the new average weight -/
theorem new_average_weight 
  (initial_count : ℕ) 
  (initial_average : ℝ) 
  (new_student_weight : ℝ) : 
  initial_count = 29 → 
  initial_average = 28 → 
  new_student_weight = 7 → 
  let total_weight := initial_count * initial_average + new_student_weight
  let new_count := initial_count + 1
  (total_weight / new_count : ℝ) = 27.3 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l346_34641


namespace NUMINAMATH_CALUDE_positive_intervals_l346_34639

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 2)

theorem positive_intervals (x : ℝ) : 
  f x > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_positive_intervals_l346_34639


namespace NUMINAMATH_CALUDE_determinant_minimization_l346_34678

theorem determinant_minimization (a b : ℤ) : 
  let Δ := 36 * a - 81 * b
  ∃ (c : ℕ+), 
    (∀ k : ℕ+, Δ = k → c ≤ k) ∧ 
    Δ = c ∧
    c = 9 ∧
    (∀ a' b' : ℕ+, 36 * a' - 81 * b' = c → a + b ≤ a' + b') ∧
    a = 7 ∧ 
    b = 3 := by
  sorry

end NUMINAMATH_CALUDE_determinant_minimization_l346_34678


namespace NUMINAMATH_CALUDE_max_x_squared_y_value_l346_34683

theorem max_x_squared_y_value (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x^3 + y^3 + 3*x*y = 1) :
  ∃ (M : ℝ), M = 4/27 ∧ x^2 * y ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_x_squared_y_value_l346_34683


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l346_34665

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a, if a₁ + a₂ + a₃ = 32 and a₁₁ + a₁₂ + a₁₃ = 118, then a₄ + a₁₀ = 50. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_sum1 : a 1 + a 2 + a 3 = 32) 
    (h_sum2 : a 11 + a 12 + a 13 = 118) : 
  a 4 + a 10 = 50 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l346_34665


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_specific_a_l346_34604

theorem polynomial_equality_implies_specific_a (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 15) + 4 = (x + b) * (x + c)) →
  (a = 10 ∨ a = 25) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_specific_a_l346_34604


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l346_34657

/-- Given two hyperbolas with equations (x²/9) - (y²/16) = 1 and (y²/25) - ((x-4)²/M) = 1,
    if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - (x - 4)^2 / M = 1) →
  (∀ x y, y = (4/3) * x ↔ y = (5/Real.sqrt M) * (x - 4)) →
  M = 225 / 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l346_34657
