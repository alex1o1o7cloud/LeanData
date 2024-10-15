import Mathlib

namespace NUMINAMATH_CALUDE_unique_number_with_special_divisors_l2788_278843

def has_twelve_divisors (N : ℕ) : Prop :=
  ∃ (d : Fin 12 → ℕ), 
    (∀ i j, i < j → d i < d j) ∧
    (∀ i, d i ∣ N) ∧
    (∀ m, m ∣ N → ∃ i, d i = m) ∧
    d 0 = 1 ∧ d 11 = N

theorem unique_number_with_special_divisors :
  ∃! N : ℕ, has_twelve_divisors N ∧
    ∃ (d : Fin 12 → ℕ), 
      (∀ i j, i < j → d i < d j) ∧
      (∀ i, d i ∣ N) ∧
      (d 0 = 1) ∧
      (d (d 3 - 2) = (d 0 + d 1 + d 3) * d 7) ∧
      N = 1989 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_special_divisors_l2788_278843


namespace NUMINAMATH_CALUDE_area_of_M_l2788_278818

-- Define the set M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (abs x + abs (4 - x) ≤ 4) ∧
               ((x^2 - 4*x - 2*y + 2) / (y - x + 3) ≥ 0) ∧
               (0 ≤ x ∧ x ≤ 4)}

-- Define the area function for sets in ℝ²
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_M : area M = 4 := by sorry

end NUMINAMATH_CALUDE_area_of_M_l2788_278818


namespace NUMINAMATH_CALUDE_imaginary_part_of_square_l2788_278806

theorem imaginary_part_of_square : Complex.im ((1 - 4 * Complex.I) ^ 2) = -8 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_square_l2788_278806


namespace NUMINAMATH_CALUDE_min_sum_of_product_l2788_278869

theorem min_sum_of_product (a b : ℤ) (h : a * b = 150) : 
  ∀ x y : ℤ, x * y = 150 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 150 ∧ a₀ + b₀ = -151 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l2788_278869


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2788_278822

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2788_278822


namespace NUMINAMATH_CALUDE_harvard_attendance_l2788_278898

def total_applicants : ℕ := 20000
def acceptance_rate : ℚ := 5 / 100
def attendance_rate : ℚ := 90 / 100

theorem harvard_attendance : 
  ⌊(total_applicants : ℚ) * acceptance_rate * attendance_rate⌋ = 900 := by
  sorry

end NUMINAMATH_CALUDE_harvard_attendance_l2788_278898


namespace NUMINAMATH_CALUDE_ball_box_difference_l2788_278833

theorem ball_box_difference : 
  let white_balls : ℕ := 30
  let red_balls : ℕ := 18
  let balls_per_box : ℕ := 6
  let white_boxes := white_balls / balls_per_box
  let red_boxes := red_balls / balls_per_box
  white_boxes - red_boxes = 2 := by
sorry

end NUMINAMATH_CALUDE_ball_box_difference_l2788_278833


namespace NUMINAMATH_CALUDE_circle_B_radius_l2788_278886

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

def internally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius - c2.radius)^2

def congruent (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

def passes_through_center (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = c1.radius^2

theorem circle_B_radius
  (A B C D E : Circle)
  (h1 : externally_tangent A B)
  (h2 : externally_tangent A C)
  (h3 : externally_tangent A E)
  (h4 : externally_tangent B C)
  (h5 : externally_tangent B E)
  (h6 : externally_tangent C E)
  (h7 : internally_tangent A D)
  (h8 : internally_tangent B D)
  (h9 : internally_tangent C D)
  (h10 : internally_tangent E D)
  (h11 : congruent B C)
  (h12 : congruent A E)
  (h13 : A.radius = 2)
  (h14 : passes_through_center A D)
  : B.radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_B_radius_l2788_278886


namespace NUMINAMATH_CALUDE_bobby_candy_count_l2788_278804

/-- The number of candy pieces Bobby ate initially -/
def initial_candy : ℕ := 26

/-- The number of additional candy pieces Bobby ate -/
def additional_candy : ℕ := 17

/-- The total number of candy pieces Bobby ate -/
def total_candy : ℕ := initial_candy + additional_candy

theorem bobby_candy_count : total_candy = 43 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_count_l2788_278804


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l2788_278800

/-- Given a cuboid with edges of 4 cm, x cm, and 6 cm, and a volume of 120 cm³, prove that x = 5 cm. -/
theorem cuboid_edge_length (x : ℝ) : 
  x > 0 → 4 * x * 6 = 120 → x = 5 := by sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l2788_278800


namespace NUMINAMATH_CALUDE_company_merger_profit_l2788_278838

theorem company_merger_profit (x : ℝ) (h1 : 0.4 * x = 60000) (h2 : 0 < x) : 0.6 * x = 90000 := by
  sorry

end NUMINAMATH_CALUDE_company_merger_profit_l2788_278838


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2788_278848

theorem inequality_system_solution :
  ∀ x : ℝ,
  (x + 1 > 7 - 2*x ∧ x ≤ (4 + 2*x) / 3) ↔ (2 < x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2788_278848


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2788_278808

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 2)
  f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2788_278808


namespace NUMINAMATH_CALUDE_pablo_puzzle_completion_time_l2788_278844

/-- The number of days Pablo needs to complete all puzzles -/
def days_to_complete_puzzles (
  pieces_per_hour : ℕ
  ) (
  puzzles_300 : ℕ
  ) (
  puzzles_500 : ℕ
  ) (
  max_hours_per_day : ℕ
  ) : ℕ :=
  let total_pieces := puzzles_300 * 300 + puzzles_500 * 500
  let pieces_per_day := pieces_per_hour * max_hours_per_day
  (total_pieces + pieces_per_day - 1) / pieces_per_day

theorem pablo_puzzle_completion_time :
  days_to_complete_puzzles 100 8 5 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_pablo_puzzle_completion_time_l2788_278844


namespace NUMINAMATH_CALUDE_community_service_arrangements_l2788_278861

def volunteers : ℕ := 8
def service_days : ℕ := 5

def arrangements (n m : ℕ) : ℕ := sorry

theorem community_service_arrangements :
  let total_arrangements := 
    (arrangements 2 1 * arrangements 6 4 * arrangements 5 5) + 
    (arrangements 2 2 * arrangements 6 3 * arrangements 4 2)
  total_arrangements = 5040 := by sorry

end NUMINAMATH_CALUDE_community_service_arrangements_l2788_278861


namespace NUMINAMATH_CALUDE_total_area_of_triangular_houses_l2788_278823

/-- The total area of three similar triangular houses -/
theorem total_area_of_triangular_houses (base : ℝ) (height : ℝ) (num_houses : ℕ) :
  base = 40 ∧ height = 20 ∧ num_houses = 3 →
  num_houses * (base * height / 2) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_area_of_triangular_houses_l2788_278823


namespace NUMINAMATH_CALUDE_gcd_g_102_103_l2788_278814

def g (x : ℤ) : ℤ := x^2 - x + 2007

theorem gcd_g_102_103 : Int.gcd (g 102) (g 103) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_102_103_l2788_278814


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l2788_278805

theorem shortest_side_of_right_triangle (a b c : ℝ) :
  a = 5 →
  b = 12 →
  c^2 = a^2 + b^2 →
  c ≥ a ∧ c ≥ b →
  a = min a b := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l2788_278805


namespace NUMINAMATH_CALUDE_savings_percentage_l2788_278836

/-- Proves that a person saves 20% of their salary given specific conditions -/
theorem savings_percentage (salary : ℝ) (savings_after_increase : ℝ) 
  (h1 : salary = 6500)
  (h2 : savings_after_increase = 260)
  (h3 : ∃ (original_expenses : ℝ), 
    salary = original_expenses + (salary * 0.2) 
    ∧ savings_after_increase = salary - (original_expenses * 1.2)) :
  (salary - savings_after_increase) / salary * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l2788_278836


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficient_l2788_278813

theorem cubic_expansion_coefficient (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficient_l2788_278813


namespace NUMINAMATH_CALUDE_square_of_binomial_equivalence_l2788_278849

theorem square_of_binomial_equivalence (x : ℝ) : (-3 - x) * (3 - x) = (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_equivalence_l2788_278849


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2788_278888

theorem complex_fraction_simplification :
  let z : ℂ := (4 - 9*I) / (3 + 4*I)
  z = -24/25 - 43/25*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2788_278888


namespace NUMINAMATH_CALUDE_squared_sum_inequality_l2788_278868

theorem squared_sum_inequality (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a^3 + b^3 = 2*a*b) :
  a^2 + b^2 ≤ 1 + a*b := by sorry

end NUMINAMATH_CALUDE_squared_sum_inequality_l2788_278868


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2788_278892

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {3, 4}

-- Theorem statement
theorem intersection_with_complement : A ∩ (U \ B) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2788_278892


namespace NUMINAMATH_CALUDE_largest_number_on_board_l2788_278887

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

def set_of_numbers : Set ℕ := {n | is_two_digit n ∧ n % 6 = 0 ∧ ends_in_four n}

theorem largest_number_on_board : 
  ∃ (m : ℕ), m ∈ set_of_numbers ∧ ∀ (n : ℕ), n ∈ set_of_numbers → n ≤ m ∧ m = 84 :=
sorry

end NUMINAMATH_CALUDE_largest_number_on_board_l2788_278887


namespace NUMINAMATH_CALUDE_premium_percentage_is_twenty_percent_l2788_278856

/-- Calculates the premium percentage on shares given investment details. -/
def calculate_premium_percentage (total_investment : ℚ) (face_value : ℚ) (dividend_rate : ℚ) (dividend_received : ℚ) : ℚ :=
  let num_shares := dividend_received / (dividend_rate * face_value / 100)
  let share_cost := total_investment / num_shares
  (share_cost - face_value) / face_value * 100

/-- Proves that the premium percentage is 20% given the specified conditions. -/
theorem premium_percentage_is_twenty_percent :
  let total_investment : ℚ := 14400
  let face_value : ℚ := 100
  let dividend_rate : ℚ := 5
  let dividend_received : ℚ := 600
  calculate_premium_percentage total_investment face_value dividend_rate dividend_received = 20 := by
  sorry

end NUMINAMATH_CALUDE_premium_percentage_is_twenty_percent_l2788_278856


namespace NUMINAMATH_CALUDE_equation_solution_l2788_278884

theorem equation_solution (x : ℝ) : x^2 + x = 5 + Real.sqrt 5 ↔ x = Real.sqrt 5 ∨ x = -Real.sqrt 5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2788_278884


namespace NUMINAMATH_CALUDE_arithmetic_equation_l2788_278858

theorem arithmetic_equation : 50 + 5 * 12 / (180 / 3) = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l2788_278858


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l2788_278894

theorem triangle_max_perimeter :
  ∀ (x : ℕ),
  x > 0 →
  x ≤ 6 →
  x + 4*x > 20 →
  4*x + 20 > x →
  x + 20 > 4*x →
  (∀ y : ℕ, y > 0 → y ≤ 6 → y + 4*y > 20 → 4*y + 20 > y → y + 20 > 4*y → x + 4*x + 20 ≥ y + 4*y + 20) →
  x + 4*x + 20 = 50 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l2788_278894


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l2788_278865

theorem butterflies_in_garden (total : ℕ) (flew_away_fraction : ℚ) (left : ℕ) : 
  total = 150 →
  flew_away_fraction = 11 / 13 →
  left = total - Int.floor (↑total * flew_away_fraction) →
  left = 23 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l2788_278865


namespace NUMINAMATH_CALUDE_equations_same_graph_l2788_278882

-- Define the three equations
def equation_I (x y : ℝ) : Prop := y = x^2 - 1
def equation_II (x y : ℝ) : Prop := x ≠ 1 → y = (x^3 - x) / (x - 1)
def equation_III (x y : ℝ) : Prop := (x - 1) * y = x^3 - x

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq1 x y ↔ eq2 x y

-- Theorem statement
theorem equations_same_graph :
  (same_graph equation_II equation_III) ∧
  (¬ same_graph equation_I equation_II) ∧
  (¬ same_graph equation_I equation_III) :=
sorry

end NUMINAMATH_CALUDE_equations_same_graph_l2788_278882


namespace NUMINAMATH_CALUDE_max_value_theorem_l2788_278819

theorem max_value_theorem (k : ℝ) (hk : k > 0) :
  (3 * k^3 + 3 * k) / ((3/2 * k^2 + 14) * (14 * k^2 + 3/2)) ≤ Real.sqrt 21 / 175 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2788_278819


namespace NUMINAMATH_CALUDE_smallest_b_value_l2788_278866

theorem smallest_b_value (a b c : ℕ) : 
  (a * b * c = 360) → 
  (1 < a) → (a < b) → (b < c) → 
  (∀ b' : ℕ, (∃ a' c' : ℕ, a' * b' * c' = 360 ∧ 1 < a' ∧ a' < b' ∧ b' < c') → b ≤ b') → 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2788_278866


namespace NUMINAMATH_CALUDE_circle_condition_l2788_278828

theorem circle_condition (k : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0) ↔ (k > 4 ∨ k < -1) :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l2788_278828


namespace NUMINAMATH_CALUDE_tank_dimension_proof_l2788_278855

/-- Proves that the second dimension of a rectangular tank is 5 feet -/
theorem tank_dimension_proof (w : ℝ) : 
  w > 0 → -- w is positive
  4 * w * 3 > 0 → -- tank volume is positive
  2 * (4 * w + 4 * 3 + w * 3) = 1880 / 20 → -- surface area equation
  w = 5 := by
  sorry

end NUMINAMATH_CALUDE_tank_dimension_proof_l2788_278855


namespace NUMINAMATH_CALUDE_min_students_with_brown_eyes_and_lunch_box_l2788_278881

/-- Given a class with the following properties:
  * There are 30 students in total
  * 12 students have brown eyes
  * 20 students have a lunch box
  This theorem proves that the minimum number of students
  who have both brown eyes and a lunch box is 2. -/
theorem min_students_with_brown_eyes_and_lunch_box
  (total_students : ℕ)
  (brown_eyes : ℕ)
  (lunch_box : ℕ)
  (h1 : total_students = 30)
  (h2 : brown_eyes = 12)
  (h3 : lunch_box = 20) :
  brown_eyes + lunch_box - total_students ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_students_with_brown_eyes_and_lunch_box_l2788_278881


namespace NUMINAMATH_CALUDE_train_length_l2788_278872

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 48 → time_s = 9 → ∃ length_m : ℝ, abs (length_m - 119.97) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2788_278872


namespace NUMINAMATH_CALUDE_half_recipe_flour_l2788_278895

-- Define the original amount of flour in the recipe
def original_flour : ℚ := 4 + 1/2

-- Define the fraction of the recipe we're making
def recipe_fraction : ℚ := 1/2

-- Theorem to prove
theorem half_recipe_flour :
  recipe_fraction * original_flour = 2 + 1/4 :=
by sorry

end NUMINAMATH_CALUDE_half_recipe_flour_l2788_278895


namespace NUMINAMATH_CALUDE_min_megabytes_for_plan_y_l2788_278839

/-- The cost in cents for Plan X given m megabytes -/
def plan_x_cost (m : ℕ) : ℕ := 15 * m

/-- The cost in cents for Plan Y given m megabytes -/
def plan_y_cost (m : ℕ) : ℕ := 3000 + 7 * m

/-- Predicate to check if Plan Y is cheaper for a given number of megabytes -/
def plan_y_cheaper (m : ℕ) : Prop := plan_y_cost m < plan_x_cost m

theorem min_megabytes_for_plan_y : ∀ m : ℕ, m ≥ 376 → plan_y_cheaper m ∧ ∀ n : ℕ, n < 376 → ¬plan_y_cheaper n :=
  sorry

end NUMINAMATH_CALUDE_min_megabytes_for_plan_y_l2788_278839


namespace NUMINAMATH_CALUDE_soccer_game_scoring_l2788_278860

/-- Soccer game scoring theorem -/
theorem soccer_game_scoring
  (team_a_first_half : ℕ)
  (team_b_first_half : ℕ)
  (team_a_second_half : ℕ)
  (team_b_second_half : ℕ)
  (h1 : team_a_first_half = 8)
  (h2 : team_b_second_half = team_a_first_half)
  (h3 : team_a_second_half = team_b_second_half - 2)
  (h4 : team_a_first_half + team_b_first_half + team_a_second_half + team_b_second_half = 26) :
  team_b_first_half / team_a_first_half = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_soccer_game_scoring_l2788_278860


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_real_range_l2788_278864

theorem sqrt_x_plus_one_real_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_real_range_l2788_278864


namespace NUMINAMATH_CALUDE_kim_math_test_probability_l2788_278871

theorem kim_math_test_probability (p : ℚ) (h : p = 4/7) :
  1 - p = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_kim_math_test_probability_l2788_278871


namespace NUMINAMATH_CALUDE_expression_evaluation_l2788_278857

theorem expression_evaluation (x : ℝ) (h : x^2 - 3*x - 2 = 0) :
  (x + 1) * (x - 1) - (x + 3)^2 + 2*x^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2788_278857


namespace NUMINAMATH_CALUDE_curve_translation_l2788_278876

-- Define the original curve
def original_curve (x y : ℝ) : Prop :=
  y * Real.cos x + 2 * y - 1 = 0

-- Define the translated curve
def translated_curve (x y : ℝ) : Prop :=
  (y + 1) * Real.sin x + 2 * y + 1 = 0

-- Theorem statement
theorem curve_translation :
  ∀ x y : ℝ, original_curve (x - π/2) (y + 1) ↔ translated_curve x y :=
by sorry

end NUMINAMATH_CALUDE_curve_translation_l2788_278876


namespace NUMINAMATH_CALUDE_open_book_is_random_event_l2788_278874

/-- Represents the possible classifications of events --/
inductive EventType
  | Certain
  | Random
  | Impossible
  | Determined

/-- Represents a book --/
structure Book where
  grade : Nat
  subject : String
  publisher : String

/-- Represents the event of opening a book to a specific page --/
structure OpenBookEvent where
  book : Book
  page : Nat
  intentional : Bool

/-- Definition of a certain event --/
def is_certain_event (e : OpenBookEvent) : Prop :=
  e.intentional ∧ ∀ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- Definition of a random event --/
def is_random_event (e : OpenBookEvent) : Prop :=
  ¬e.intentional ∧ ∃ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- Definition of an impossible event --/
def is_impossible_event (e : OpenBookEvent) : Prop :=
  ¬∃ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- Definition of a determined event --/
def is_determined_event (e : OpenBookEvent) : Prop :=
  e.intentional ∧ ∃ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- The main theorem to prove --/
theorem open_book_is_random_event (e : OpenBookEvent) 
  (h1 : e.book.grade = 9)
  (h2 : e.book.subject = "mathematics")
  (h3 : e.book.publisher = "East China Normal University")
  (h4 : e.page = 50)
  (h5 : ¬e.intentional) :
  is_random_event e :=
sorry

end NUMINAMATH_CALUDE_open_book_is_random_event_l2788_278874


namespace NUMINAMATH_CALUDE_a_minus_b_equals_two_l2788_278867

-- Define the functions f, g, h, and h_inv
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)
def h_inv (x : ℝ) : ℝ := x + 3

-- State the theorem
theorem a_minus_b_equals_two (a b : ℝ) : 
  (∀ x, h a b x = x - 3) → 
  (∀ x, h a b (h_inv x) = x) → 
  a - b = 2 := by
  sorry


end NUMINAMATH_CALUDE_a_minus_b_equals_two_l2788_278867


namespace NUMINAMATH_CALUDE_strawberry_sale_revenue_difference_l2788_278840

/-- Represents the sale of strawberries at a supermarket -/
structure StrawberrySale where
  pints_sold : ℕ
  sale_revenue : ℕ
  price_difference : ℕ

/-- Calculates the revenue difference between regular and sale prices -/
def revenue_difference (sale : StrawberrySale) : ℕ :=
  let sale_price := sale.sale_revenue / sale.pints_sold
  let regular_price := sale_price + sale.price_difference
  regular_price * sale.pints_sold - sale.sale_revenue

/-- Theorem stating the revenue difference for the given scenario -/
theorem strawberry_sale_revenue_difference :
  ∃ (sale : StrawberrySale),
    sale.pints_sold = 54 ∧
    sale.sale_revenue = 216 ∧
    sale.price_difference = 2 ∧
    revenue_difference sale = 108 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_sale_revenue_difference_l2788_278840


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2788_278885

/-- Given a quadratic function f(x) = ax² + bx + c, 
    if f(0) = f(4) > f(1), then a > 0 and 4a + b = 0 -/
theorem quadratic_function_property (a b c : ℝ) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  (f 0 = f 4 ∧ f 0 > f 1) → (a > 0 ∧ 4 * a + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2788_278885


namespace NUMINAMATH_CALUDE_cube_root_eight_times_sixth_root_sixtyfour_equals_four_l2788_278862

theorem cube_root_eight_times_sixth_root_sixtyfour_equals_four :
  (8 : ℝ) ^ (1/3) * (64 : ℝ) ^ (1/6) = 4 := by sorry

end NUMINAMATH_CALUDE_cube_root_eight_times_sixth_root_sixtyfour_equals_four_l2788_278862


namespace NUMINAMATH_CALUDE_two_distinct_roots_condition_l2788_278870

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := k * x^2 - 2 * x - 3

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  quadratic_equation k x₁ = 0 ∧ 
  quadratic_equation k x₂ = 0

-- Theorem statement
theorem two_distinct_roots_condition (k : ℝ) :
  has_two_distinct_real_roots k ↔ k > -1/3 ∧ k ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_roots_condition_l2788_278870


namespace NUMINAMATH_CALUDE_students_per_bus_l2788_278810

theorem students_per_bus 
  (total_students : ℕ) 
  (num_buses : ℕ) 
  (students_in_cars : ℕ) 
  (h1 : total_students = 396) 
  (h2 : num_buses = 7) 
  (h3 : students_in_cars = 4) 
  (h4 : num_buses > 0) : 
  (total_students - students_in_cars) / num_buses = 56 := by
sorry

end NUMINAMATH_CALUDE_students_per_bus_l2788_278810


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_half_l2788_278820

noncomputable def f (x : ℝ) : ℝ := Real.sin x / (Real.sin x + Real.cos x)

theorem derivative_f_at_pi_half :
  deriv f (π / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_half_l2788_278820


namespace NUMINAMATH_CALUDE_terry_current_age_l2788_278883

/-- Terry's current age in years -/
def terry_age : ℕ := sorry

/-- Nora's current age in years -/
def nora_age : ℕ := 10

/-- The number of years in the future when Terry's age will be 4 times Nora's current age -/
def years_future : ℕ := 10

theorem terry_current_age : 
  terry_age = 30 :=
by
  have h1 : terry_age + years_future = 4 * nora_age := sorry
  sorry

end NUMINAMATH_CALUDE_terry_current_age_l2788_278883


namespace NUMINAMATH_CALUDE_population_model_steps_l2788_278826

/- Define the steps as an inductive type -/
inductive ModelingStep
  | observe : ModelingStep
  | test : ModelingStep
  | propose : ModelingStep
  | express : ModelingStep

/- Define a function to represent the correct order of steps -/
def correct_order : List ModelingStep :=
  [ModelingStep.observe, ModelingStep.propose, ModelingStep.express, ModelingStep.test]

/- Define a predicate to check if a given order is correct -/
def is_correct_order (order : List ModelingStep) : Prop :=
  order = correct_order

/- Theorem stating that the specified order is correct -/
theorem population_model_steps :
  is_correct_order [ModelingStep.observe, ModelingStep.propose, ModelingStep.express, ModelingStep.test] :=
by sorry

end NUMINAMATH_CALUDE_population_model_steps_l2788_278826


namespace NUMINAMATH_CALUDE_rockham_soccer_league_members_l2788_278837

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 4

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 5

/-- The total cost for all members in dollars -/
def total_cost : ℕ := 2366

/-- The number of pairs of socks each member needs -/
def socks_per_member : ℕ := 2

/-- The number of T-shirts each member needs -/
def tshirts_per_member : ℕ := 2

/-- The cost of equipment for one member -/
def member_cost : ℕ := socks_per_member * sock_cost + 
                       tshirts_per_member * (sock_cost + tshirt_additional_cost)

/-- The number of members in the Rockham Soccer League -/
def number_of_members : ℕ := total_cost / member_cost

theorem rockham_soccer_league_members : number_of_members = 91 := by
  sorry

end NUMINAMATH_CALUDE_rockham_soccer_league_members_l2788_278837


namespace NUMINAMATH_CALUDE_part1_part2_l2788_278802

-- Define the concept of l-increasing function
def is_l_increasing (f : ℝ → ℝ) (D : Set ℝ) (M : Set ℝ) (l : ℝ) : Prop :=
  l ≠ 0 ∧ (∀ x ∈ M, x + l ∈ D ∧ f (x + l) ≥ f x)

-- Part 1
theorem part1 (f : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Set.Ici (-1), f x = x^2) →
  is_l_increasing f (Set.Ici (-1)) (Set.Ici (-1)) m →
  m ≥ 2 := by sorry

-- Part 2
theorem part2 (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (-x) = -f x) →
  (∀ x ≥ 0, f x = |x - a^2| - a^2) →
  is_l_increasing f Set.univ Set.univ 8 →
  -2 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2788_278802


namespace NUMINAMATH_CALUDE_blue_marbles_count_l2788_278850

theorem blue_marbles_count (red blue : ℕ) : 
  red + blue = 6000 →
  (red + blue) - (blue - red) = 4800 →
  blue > red →
  blue = 3600 := by
sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l2788_278850


namespace NUMINAMATH_CALUDE_only_zhong_symmetrical_l2788_278801

/-- Represents a Chinese character --/
inductive ChineseCharacter
| ai    -- 爱
| wo    -- 我
| zhong -- 中
| guo   -- 国

/-- Determines if a Chinese character is symmetrical --/
def is_symmetrical (c : ChineseCharacter) : Prop :=
  match c with
  | ChineseCharacter.zhong => True
  | _ => False

/-- Theorem stating that among the given characters, only 中 (zhong) is symmetrical --/
theorem only_zhong_symmetrical :
  ∀ c : ChineseCharacter,
    is_symmetrical c ↔ c = ChineseCharacter.zhong :=
by sorry

end NUMINAMATH_CALUDE_only_zhong_symmetrical_l2788_278801


namespace NUMINAMATH_CALUDE_xiao_ming_correct_answers_l2788_278880

theorem xiao_ming_correct_answers 
  (total_questions : ℕ) 
  (correct_points : ℤ) 
  (wrong_points : ℤ) 
  (total_score : ℤ) 
  (h1 : total_questions = 20)
  (h2 : correct_points = 5)
  (h3 : wrong_points = -1)
  (h4 : total_score = 76) :
  ∃ (correct_answers : ℕ), 
    correct_answers ≤ total_questions ∧ 
    correct_points * correct_answers + wrong_points * (total_questions - correct_answers) = total_score ∧
    correct_answers = 16 := by
  sorry

#check xiao_ming_correct_answers

end NUMINAMATH_CALUDE_xiao_ming_correct_answers_l2788_278880


namespace NUMINAMATH_CALUDE_junk_mail_delivery_l2788_278863

/-- Calculates the total pieces of junk mail delivered given the number of houses with white and red mailboxes -/
def total_junk_mail (total_houses : ℕ) (white_mailboxes : ℕ) (red_mailboxes : ℕ) (mail_per_house : ℕ) : ℕ :=
  (white_mailboxes + red_mailboxes) * mail_per_house

/-- Proves that the total junk mail delivered is 30 pieces given the specified conditions -/
theorem junk_mail_delivery :
  total_junk_mail 8 2 3 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_delivery_l2788_278863


namespace NUMINAMATH_CALUDE_suzanna_bike_ride_l2788_278845

/-- Calculates the distance traveled given a constant speed and time -/
def distance_traveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- Represents Suzanna's bike ride -/
theorem suzanna_bike_ride (speed : ℚ) (time : ℚ) (h1 : speed = 1 / 6) (h2 : time = 40) :
  distance_traveled speed time = 6 := by
  sorry

#check suzanna_bike_ride

end NUMINAMATH_CALUDE_suzanna_bike_ride_l2788_278845


namespace NUMINAMATH_CALUDE_no_such_hexagon_exists_l2788_278846

-- Define a hexagon as a collection of 6 points in 2D space
def Hexagon := (Fin 6 → ℝ × ℝ)

-- Define a predicate for convexity
def is_convex (h : Hexagon) : Prop := sorry

-- Define a predicate for a point being inside a hexagon
def is_inside (p : ℝ × ℝ) (h : Hexagon) : Prop := sorry

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the length of a hexagon side
def side_length (h : Hexagon) (i : Fin 6) : ℝ := 
  distance (h i) (h ((i + 1) % 6))

-- Theorem statement
theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon) (m : ℝ × ℝ),
    is_convex h ∧
    is_inside m h ∧
    (∀ i : Fin 6, side_length h i > 1) ∧
    (∀ i : Fin 6, distance m (h i) < 1) :=
sorry

end NUMINAMATH_CALUDE_no_such_hexagon_exists_l2788_278846


namespace NUMINAMATH_CALUDE_acme_vowel_soup_combinations_l2788_278879

/-- Represents the number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- Represents the number of times each vowel appears in the soup -/
def vowel_occurrences : ℕ := 6

/-- Represents the number of wildcard characters in the soup -/
def num_wildcards : ℕ := 1

/-- Represents the length of the words to be formed -/
def word_length : ℕ := 6

/-- Represents the total number of character choices for each position in the word -/
def choices_per_position : ℕ := num_vowels + num_wildcards

/-- Theorem stating that the number of possible six-letter words is 46656 -/
theorem acme_vowel_soup_combinations :
  choices_per_position ^ word_length = 46656 := by
  sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_combinations_l2788_278879


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2788_278852

/-- Definition of a quadratic equation in standard form -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2788_278852


namespace NUMINAMATH_CALUDE_sara_lunch_bill_total_l2788_278854

/-- The total cost of Sara's lunch bill --/
def lunch_bill (hotdog_cost salad_cost drink_cost side_item_cost : ℚ) : ℚ :=
  hotdog_cost + salad_cost + drink_cost + side_item_cost

/-- Theorem stating that Sara's lunch bill totals $16.71 --/
theorem sara_lunch_bill_total :
  lunch_bill 5.36 5.10 2.50 3.75 = 16.71 := by
  sorry

end NUMINAMATH_CALUDE_sara_lunch_bill_total_l2788_278854


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2788_278878

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi →
  V = (4/3) * Real.pi * r^3 →
  A = 4 * Real.pi * r^2 →
  A = 36 * Real.pi * 2^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2788_278878


namespace NUMINAMATH_CALUDE_triangle_cosine_proof_l2788_278896

theorem triangle_cosine_proof (A B C : ℝ) (a b c : ℝ) : 
  a = 4 → c = 9 → (Real.sin A) * (Real.sin C) = (Real.sin B)^2 → 
  Real.cos B = 61/72 := by sorry

end NUMINAMATH_CALUDE_triangle_cosine_proof_l2788_278896


namespace NUMINAMATH_CALUDE_circle_radius_with_tangent_and_secant_l2788_278897

/-- A circle with a tangent and a secant drawn from an external point -/
structure CircleWithTangentAndSecant where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the tangent -/
  tangent_length : ℝ
  /-- The length of the internal segment of the secant -/
  secant_internal_length : ℝ
  /-- The tangent and secant are mutually perpendicular -/
  perpendicular : True

/-- Theorem: If a circle has a tangent of length 12 and a secant with internal segment of length 10,
    and the tangent and secant are mutually perpendicular, then the radius of the circle is 13 -/
theorem circle_radius_with_tangent_and_secant 
  (c : CircleWithTangentAndSecant) 
  (h1 : c.tangent_length = 12) 
  (h2 : c.secant_internal_length = 10) :
  c.radius = 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_tangent_and_secant_l2788_278897


namespace NUMINAMATH_CALUDE_c_paisa_per_a_rupee_l2788_278803

/-- Represents the share of money for each person in rupees -/
structure Shares where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : Shares) : Prop :=
  s.b = 0.65 * s.a ∧  -- For each Rs. A has, B has 65 paisa
  s.c = 32 ∧  -- C's share is Rs. 32
  s.a + s.b + s.c = 164  -- The total sum of money is Rs. 164

/-- The theorem to be proved -/
theorem c_paisa_per_a_rupee (s : Shares) 
  (h : problem_conditions s) : (s.c * 100) / s.a = 40 := by
  sorry


end NUMINAMATH_CALUDE_c_paisa_per_a_rupee_l2788_278803


namespace NUMINAMATH_CALUDE_some_number_value_l2788_278893

theorem some_number_value (x : ℝ) : x * 6000 = 480 * 10^5 → x = 8000 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2788_278893


namespace NUMINAMATH_CALUDE_david_money_left_l2788_278827

/-- Calculates the amount of money David has left after his trip -/
def money_left (initial_amount : ℕ) (difference : ℕ) : ℕ :=
  initial_amount - (initial_amount - difference) / 2

theorem david_money_left :
  money_left 1800 800 = 500 := by
  sorry

end NUMINAMATH_CALUDE_david_money_left_l2788_278827


namespace NUMINAMATH_CALUDE_min_value_inequality_l2788_278851

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 9) :
  (x^2 + y^2)/(3*(x + y)) + (x^2 + z^2)/(3*(x + z)) + (y^2 + z^2)/(3*(y + z)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2788_278851


namespace NUMINAMATH_CALUDE_lcm_of_1716_924_1260_l2788_278859

theorem lcm_of_1716_924_1260 : Nat.lcm (Nat.lcm 1716 924) 1260 = 13860 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_1716_924_1260_l2788_278859


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l2788_278830

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (x + 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  (a₁ + 3*a₃ + 5*a₅ + 7*a₇ + 9*a₉)^2 - (2*a₂ + 4*a₄ + 6*a₆ + 8*a₈)^2 = 3^12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l2788_278830


namespace NUMINAMATH_CALUDE_primle_is_79_l2788_278877

def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_in_tens_place (n : ℕ) (d : ℕ) : Prop := n / 10 = d

def digit_in_ones_place (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem primle_is_79 (primle : ℕ) 
  (h1 : is_prime primle)
  (h2 : is_two_digit primle)
  (h3 : digit_in_tens_place primle 7)
  (h4 : ¬ digit_in_ones_place primle 7)
  (h5 : ¬ (digit_in_tens_place primle 1 ∨ digit_in_ones_place primle 1))
  (h6 : ¬ (digit_in_tens_place primle 3 ∨ digit_in_ones_place primle 3))
  (h7 : ¬ (digit_in_tens_place primle 4 ∨ digit_in_ones_place primle 4)) :
  primle = 79 := by
sorry

end NUMINAMATH_CALUDE_primle_is_79_l2788_278877


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_675_l2788_278815

theorem sin_n_equals_cos_675 (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) :
  Real.sin (n * π / 180) = Real.cos (675 * π / 180) → n = 45 := by
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_675_l2788_278815


namespace NUMINAMATH_CALUDE_probability_of_point_in_region_l2788_278831

-- Define the lines
def line1 (x : ℝ) : ℝ := -2 * x + 8
def line2 (x : ℝ) : ℝ := -3 * x + 9

-- Define the region of interest
def region_of_interest (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ y ≤ line1 x ∧ y ≥ line2 x

-- Define the total area under line1 in the first quadrant
def total_area : ℝ := 16

-- Define the area of the region of interest
def area_of_interest : ℝ := 14.5

-- Theorem statement
theorem probability_of_point_in_region :
  (area_of_interest / total_area) = 0.90625 :=
sorry

end NUMINAMATH_CALUDE_probability_of_point_in_region_l2788_278831


namespace NUMINAMATH_CALUDE_geometric_progression_identity_l2788_278812

/-- If a, b, c form a geometric progression, then (a+b+c)(a-b+c) = a^2 + b^2 + c^2 -/
theorem geometric_progression_identity (a b c : ℝ) (h : b^2 = a*c) :
  (a + b + c) * (a - b + c) = a^2 + b^2 + c^2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_progression_identity_l2788_278812


namespace NUMINAMATH_CALUDE_power_equality_l2788_278841

theorem power_equality : (8 : ℕ) ^ 8 = (4 : ℕ) ^ 12 ∧ (8 : ℕ) ^ 8 = (2 : ℕ) ^ 24 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2788_278841


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_three_l2788_278807

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes are equal -/
def parallel_lines (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ A1 / B1 = A2 / B2

/-- The first line: 3x + ay + 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + a * y + 1 = 0

/-- The second line: (a+2)x + y + a = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  (a + 2) * x + y + a = 0

/-- The theorem stating that the lines are parallel if and only if a = -3 -/
theorem lines_parallel_iff_a_eq_neg_three :
  ∃ (a : ℝ), parallel_lines 3 a 1 (a + 2) 1 a ↔ a = -3 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_three_l2788_278807


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2788_278811

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 * a 1 - 10 * a 1 + 16 = 0) →
  (a 19 * a 19 - 10 * a 19 + 16 = 0) →
  a 8 * a 10 * a 12 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2788_278811


namespace NUMINAMATH_CALUDE_probability_of_even_sum_l2788_278825

theorem probability_of_even_sum (p1 p2 : ℝ) 
  (h1 : p1 = 1/2)  -- Probability of even number from first wheel
  (h2 : p2 = 1/3)  -- Probability of even number from second wheel
  : p1 * p2 + (1 - p1) * (1 - p2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_even_sum_l2788_278825


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2788_278824

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2788_278824


namespace NUMINAMATH_CALUDE_grape_rate_proof_l2788_278873

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 70

/-- The weight of grapes purchased in kg -/
def grape_weight : ℝ := 7

/-- The weight of mangoes purchased in kg -/
def mango_weight : ℝ := 9

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 55

/-- The total amount paid -/
def total_paid : ℝ := 985

theorem grape_rate_proof : 
  grape_rate * grape_weight + mango_rate * mango_weight = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_proof_l2788_278873


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l2788_278847

/-- Represents the savings from a coupon given a price -/
def CouponSavings (price : ℝ) : (ℝ → ℝ) → ℝ := fun coupon => coupon price

/-- Coupon A: 20% off the listed price -/
def CouponA (price : ℝ) : ℝ := 0.2 * price

/-- Coupon B: $30 off the listed price -/
def CouponB (price : ℝ) : ℝ := 30

/-- Coupon C: 20% off the amount exceeding $100 -/
def CouponC (price : ℝ) : ℝ := 0.2 * (price - 100)

/-- The lowest price where Coupon A saves at least as much as Coupon B or C -/
def x : ℝ := 150

/-- The highest price where Coupon A saves at least as much as Coupon B or C -/
def y : ℝ := 300

theorem coupon_savings_difference :
  ∀ price : ℝ, price > 100 →
  (x ≤ price ∧ price ≤ y) ↔
  (CouponSavings price CouponA ≥ CouponSavings price CouponB ∧
   CouponSavings price CouponA ≥ CouponSavings price CouponC) →
  y - x = 150 := by sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l2788_278847


namespace NUMINAMATH_CALUDE_inequality_proof_l2788_278809

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2788_278809


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l2788_278899

/-- Given a paint mixture with a ratio of blue:green:yellow as 4:3:5,
    if 15 quarts of yellow paint are used, then 9 quarts of green paint should be used. -/
theorem paint_mixture_ratio (blue green yellow : ℚ) :
  blue / green = 4 / 3 →
  green / yellow = 3 / 5 →
  yellow = 15 →
  green = 9 := by
sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l2788_278899


namespace NUMINAMATH_CALUDE_marble_difference_l2788_278834

theorem marble_difference (seokjin_marbles : ℕ) (yuna_marbles : ℕ) (jimin_marbles : ℕ) : 
  seokjin_marbles = 3 →
  yuna_marbles = seokjin_marbles - 1 →
  jimin_marbles = 2 * seokjin_marbles →
  jimin_marbles - yuna_marbles = 4 := by
sorry

end NUMINAMATH_CALUDE_marble_difference_l2788_278834


namespace NUMINAMATH_CALUDE_limit_at_negative_four_l2788_278816

/-- The limit of (2x^2 + 6x - 8)/(x + 4) as x approaches -4 is -10 -/
theorem limit_at_negative_four :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x + 4| ∧ |x + 4| < δ →
    |(2*x^2 + 6*x - 8)/(x + 4) + 10| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_at_negative_four_l2788_278816


namespace NUMINAMATH_CALUDE_vector_equality_l2788_278832

def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (x - 2, x)

theorem vector_equality (x : ℝ) :
  let a := vector_a x
  let b := vector_b x
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = (a.1 - b.1)^2 + (a.2 - b.2)^2 →
  x = 1 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_vector_equality_l2788_278832


namespace NUMINAMATH_CALUDE_sum_of_squares_l2788_278829

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 28) : x^2 + y^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2788_278829


namespace NUMINAMATH_CALUDE_compound_interest_problem_l2788_278842

theorem compound_interest_problem (P : ℝ) (t : ℝ) : 
  P * (1 + 0.1)^t = 2420 → 
  P * (1 + 0.1)^(t+3) = 2662 → 
  t = 3 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l2788_278842


namespace NUMINAMATH_CALUDE_max_a_value_l2788_278890

/-- A lattice point in an xy-coordinate system is any point (x, y) where both x and y are integers. -/
def is_lattice_point (x y : ℤ) : Prop := True

/-- The equation y = mx + 3 -/
def equation (m : ℚ) (x y : ℤ) : Prop := y = m * x + 3

/-- The condition that the equation has no lattice point solutions for 0 < x ≤ 150 -/
def no_lattice_solutions (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 150 → is_lattice_point x y → ¬equation m x y

/-- The theorem stating that 101/150 is the maximum value of a satisfying the given conditions -/
theorem max_a_value : 
  (∃ a : ℚ, a = 101/150 ∧ 
    (∀ m : ℚ, 2/3 < m → m < a → no_lattice_solutions m) ∧
    (∀ b : ℚ, b > a → ∃ m : ℚ, 2/3 < m ∧ m < b ∧ ¬no_lattice_solutions m)) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2788_278890


namespace NUMINAMATH_CALUDE_initial_typists_count_l2788_278891

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 25

/-- The number of letters the initial group can type in 20 minutes -/
def letters_in_20_min : ℕ := 60

/-- The number of typists in the second group -/
def second_group_typists : ℕ := 75

/-- The number of letters the second group can type in 60 minutes -/
def letters_in_60_min : ℕ := 540

/-- The time ratio between the two scenarios -/
def time_ratio : ℚ := 3

theorem initial_typists_count :
  initial_typists * second_group_typists * letters_in_20_min * time_ratio = 
  letters_in_60_min * initial_typists * time_ratio :=
sorry

end NUMINAMATH_CALUDE_initial_typists_count_l2788_278891


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2788_278889

-- Problem 1
theorem problem_1 : 3 + (-1) - (-3) + 2 = 10 := by sorry

-- Problem 2
theorem problem_2 : 12 + |(-6)| - (-8) * 3 = 42 := by sorry

-- Problem 3
theorem problem_3 : (2/3 - 1/4 - 3/8) * 24 = 1 := by sorry

-- Problem 4
theorem problem_4 : -1^2021 - (-3 * (2/3)^2 - 4/3 / 2^2) = 2/3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2788_278889


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2788_278821

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℕ)
  (h_geom : GeometricSequence a)
  (h_first : a 1 = 3)
  (h_sixth : a 6 = 972) :
  a 7 = 2187 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2788_278821


namespace NUMINAMATH_CALUDE_avalon_quest_probability_l2788_278817

theorem avalon_quest_probability :
  let total_players : ℕ := 10
  let bad_players : ℕ := 4
  let quest_size : ℕ := 3
  let good_players : ℕ := total_players - bad_players
  let total_quests : ℕ := Nat.choose total_players quest_size
  let failed_quests : ℕ := total_quests - Nat.choose good_players quest_size
  let one_bad_quests : ℕ := Nat.choose bad_players 1 * Nat.choose good_players (quest_size - 1)
  (failed_quests > 0) →
  (one_bad_quests : ℚ) / failed_quests = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_avalon_quest_probability_l2788_278817


namespace NUMINAMATH_CALUDE_range_of_a_l2788_278875

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | x ≤ a}
def Q : Set ℝ := {y | ∃ θ : ℝ, y = Real.sin θ}

-- State the theorem
theorem range_of_a (a : ℝ) : P a ⊇ Q → a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2788_278875


namespace NUMINAMATH_CALUDE_card_deck_size_l2788_278835

theorem card_deck_size (n : ℕ) (h1 : n ≥ 6) 
  (h2 : Nat.choose n 6 = 6 * Nat.choose n 3) : n = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_card_deck_size_l2788_278835


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l2788_278853

def pizza_problem (total_slices : ℕ) (plain_cost anchovy_cost onion_cost : ℚ)
  (anchovy_slices onion_slices : ℕ) (jerry_plain_slices : ℕ) : Prop :=
  let total_cost := plain_cost + anchovy_cost + onion_cost
  let cost_per_slice := total_cost / total_slices
  let jerry_slices := anchovy_slices + onion_slices + jerry_plain_slices
  let tom_slices := total_slices - jerry_slices
  let jerry_cost := cost_per_slice * jerry_slices
  let tom_cost := cost_per_slice * tom_slices
  jerry_cost - tom_cost = 11.36

theorem pizza_payment_difference :
  pizza_problem 12 12 3 2 4 4 2 := by sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l2788_278853
