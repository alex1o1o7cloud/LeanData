import Mathlib

namespace NUMINAMATH_CALUDE_sum_divisible_by_3_probability_l2499_249923

/-- Represents the outcome of rolling a fair 6-sided die -/
def DieRoll : Type := Fin 6

/-- The sample space of rolling a fair die three times -/
def SampleSpace : Type := DieRoll × DieRoll × DieRoll

/-- The number of possible outcomes in the sample space -/
def totalOutcomes : Nat := 216

/-- Predicate for outcomes where the sum is divisible by 3 -/
def sumDivisibleBy3 (outcome : SampleSpace) : Prop :=
  (outcome.1.val + outcome.2.1.val + outcome.2.2.val + 3) % 3 = 0

/-- The number of favorable outcomes (sum divisible by 3) -/
def favorableOutcomes : Nat := 72

/-- The probability of the sum being divisible by 3 -/
def probability : ℚ := favorableOutcomes / totalOutcomes

theorem sum_divisible_by_3_probability :
  probability = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_sum_divisible_by_3_probability_l2499_249923


namespace NUMINAMATH_CALUDE_p_or_q_true_not_imply_p_and_q_true_l2499_249921

theorem p_or_q_true_not_imply_p_and_q_true (p q : Prop) : 
  (p ∨ q) → ¬(p ∧ q → True) :=
by sorry

end NUMINAMATH_CALUDE_p_or_q_true_not_imply_p_and_q_true_l2499_249921


namespace NUMINAMATH_CALUDE_trapezoid_de_length_l2499_249911

/-- Represents a trapezoid ABCD formed by a rectangle ABCE and a right triangle EDF -/
structure Trapezoid where
  /-- Length of side AB of the rectangle -/
  ab : ℝ
  /-- Length of side BC of the rectangle -/
  bc : ℝ
  /-- Length of side DE of the trapezoid -/
  de : ℝ
  /-- Length of side EF of the triangle -/
  ef : ℝ
  /-- Condition that AB = 7 -/
  ab_eq : ab = 7
  /-- Condition that BC = 8 -/
  bc_eq : bc = 8
  /-- Condition that DE is twice EF -/
  de_twice_ef : de = 2 * ef
  /-- Condition that the areas of the rectangle and triangle are equal -/
  areas_equal : ab * bc = (1 / 2) * de * ef

/-- Theorem stating that the length of DE in the described trapezoid is 4√14 -/
theorem trapezoid_de_length (t : Trapezoid) : t.de = 4 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_de_length_l2499_249911


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2499_249987

theorem geometric_sequence_problem (a b c : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ -2 = -2 * r ∧ a = -2 * r^2 ∧ b = -2 * r^3 ∧ c = -2 * r^4 ∧ -8 = -2 * r^5) →
  b = -4 ∧ a * c = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2499_249987


namespace NUMINAMATH_CALUDE_joshua_bottle_caps_l2499_249926

theorem joshua_bottle_caps (initial : ℕ) (final : ℕ) (bought : ℕ) : 
  initial = 40 → final = 47 → final = initial + bought → bought = 7 := by
  sorry

end NUMINAMATH_CALUDE_joshua_bottle_caps_l2499_249926


namespace NUMINAMATH_CALUDE_runner_ends_in_quadrant_A_l2499_249964

/-- Represents the quadrants of the circular track -/
inductive Quadrant
  | A
  | B
  | C
  | D

/-- Represents a point on the circular track -/
structure Point where
  angle : ℝ  -- Angle in radians from the starting point S

/-- The circular track -/
structure Track where
  circumference : ℝ
  start : Point

/-- A runner on the track -/
structure Runner where
  position : Point
  distance_run : ℝ

/-- Function to determine which quadrant a point is in -/
def point_to_quadrant (p : Point) : Quadrant :=
  sorry

/-- Function to update a runner's position after running a certain distance -/
def update_position (r : Runner) (d : ℝ) (t : Track) : Runner :=
  sorry

/-- Main theorem: After running one mile, the runner ends up in quadrant A -/
theorem runner_ends_in_quadrant_A (t : Track) (r : Runner) :
  t.circumference = 60 ∧ 
  r.position = t.start ∧
  (update_position r 5280 t).position = t.start →
  point_to_quadrant ((update_position r 5280 t).position) = Quadrant.A :=
  sorry

end NUMINAMATH_CALUDE_runner_ends_in_quadrant_A_l2499_249964


namespace NUMINAMATH_CALUDE_handshakes_five_people_l2499_249949

/-- The number of handshakes between n people, where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := Nat.choose n 2

/-- There are 5 people in the room. -/
def num_people : ℕ := 5

theorem handshakes_five_people : handshakes num_people = 10 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_five_people_l2499_249949


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2499_249918

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (((3 / (x - 1)) - x - 1) / ((x^2 - 4*x + 4) / (x - 1))) = (2 + x) / (2 - x) ∧
  (((3 / (0 - 1)) - 0 - 1) / ((0^2 - 4*0 + 4) / (0 - 1))) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2499_249918


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2499_249908

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + a*x - 2 = 0 ↔ x = x₁ ∨ x = x₂) →  -- equation has exactly two roots
  x₁ ≠ x₂ →  -- roots are distinct
  x₁ < -1 →
  x₂ > 1 →
  -1 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2499_249908


namespace NUMINAMATH_CALUDE_partner_q_investment_time_l2499_249924

/-- Represents the investment and profit information for a business partnership --/
structure Partnership where
  investmentRatio : Fin 3 → ℚ
  profitRatio : Fin 3 → ℚ
  investmentTime : Fin 3 → ℚ

/-- Theorem stating the investment time for partner Q given the conditions --/
theorem partner_q_investment_time (p : Partnership) :
  p.investmentRatio 0 = 3 ∧
  p.investmentRatio 1 = 4 ∧
  p.investmentRatio 2 = 5 ∧
  p.profitRatio 0 = 9 ∧
  p.profitRatio 1 = 16 ∧
  p.profitRatio 2 = 25 ∧
  p.investmentTime 0 = 4 ∧
  p.investmentTime 2 = 10 →
  p.investmentTime 1 = 8 :=
by sorry

end NUMINAMATH_CALUDE_partner_q_investment_time_l2499_249924


namespace NUMINAMATH_CALUDE_quadratic_always_intersects_x_axis_l2499_249974

theorem quadratic_always_intersects_x_axis (a : ℝ) (ha : a ≠ 0) :
  ∃ x : ℝ, a * x^2 - (3*a + 1) * x + 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_intersects_x_axis_l2499_249974


namespace NUMINAMATH_CALUDE_farmers_field_planted_fraction_l2499_249962

theorem farmers_field_planted_fraction 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (leg_lengths : a = 5 ∧ b = 12) 
  (square_side : ℝ) 
  (square_distance_to_hypotenuse : ℝ) 
  (square_distance_condition : square_distance_to_hypotenuse = 3) 
  (square_tangent : square_side ≤ a ∧ square_side ≤ b) 
  (area_equation : (1/2) * c * square_distance_to_hypotenuse = (1/2) * a * b - square_side^2) :
  (((1/2) * a * b - square_side^2) / ((1/2) * a * b)) = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_farmers_field_planted_fraction_l2499_249962


namespace NUMINAMATH_CALUDE_sum_three_numbers_l2499_249972

theorem sum_three_numbers (a b c N : ℝ) 
  (sum_eq : a + b + c = 105)
  (a_eq : a - 5 = N)
  (b_eq : b + 10 = N)
  (c_eq : 5 * c = N) : 
  N = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l2499_249972


namespace NUMINAMATH_CALUDE_book_price_calculation_l2499_249947

theorem book_price_calculation (P : ℝ) : 
  P * 0.85 * 1.40 = 476 → P = 400 := by
  sorry

end NUMINAMATH_CALUDE_book_price_calculation_l2499_249947


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2499_249995

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (3 * α - 2 * β) = 1 / 2)
  (h2 : Real.tan (5 * α - 4 * β) = 1 / 4) : 
  Real.tan α = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2499_249995


namespace NUMINAMATH_CALUDE_nail_salon_revenue_l2499_249958

/-- Calculates the total money made from manicures in a nail salon --/
def total_manicure_money (manicure_cost : ℝ) (total_fingers : ℕ) (fingers_per_person : ℕ) (non_clients : ℕ) : ℝ :=
  let total_people : ℕ := total_fingers / fingers_per_person
  let clients : ℕ := total_people - non_clients
  (clients : ℝ) * manicure_cost

/-- Theorem stating the total money made from manicures in the given scenario --/
theorem nail_salon_revenue :
  total_manicure_money 20 210 10 11 = 200 := by
  sorry

end NUMINAMATH_CALUDE_nail_salon_revenue_l2499_249958


namespace NUMINAMATH_CALUDE_josh_marbles_l2499_249925

/-- The number of marbles Josh lost -/
def marbles_lost : ℕ := sorry

/-- The number of marbles Josh initially had -/
def initial_marbles : ℕ := 7

/-- The number of new marbles Josh found -/
def new_marbles : ℕ := 10

/-- The difference between marbles found and marbles lost -/
def difference : ℕ := 2

theorem josh_marbles : marbles_lost = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l2499_249925


namespace NUMINAMATH_CALUDE_max_cards_from_poster_board_l2499_249903

/-- Represents the dimensions of a rectangular object in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of small rectangles that can fit into a larger square -/
def maxRectangles (square_side : ℕ) (card : Dimensions) : ℕ :=
  (square_side / card.length) * (square_side / card.width)

theorem max_cards_from_poster_board :
  let poster_board_side : ℕ := 12  -- 1 foot = 12 inches
  let card : Dimensions := { length := 2, width := 3 }
  maxRectangles poster_board_side card = 24 := by
sorry

end NUMINAMATH_CALUDE_max_cards_from_poster_board_l2499_249903


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2499_249910

theorem solution_set_quadratic_inequality :
  {x : ℝ | 6 * x^2 + 5 * x < 4} = {x : ℝ | -4/3 < x ∧ x < 1/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2499_249910


namespace NUMINAMATH_CALUDE_power_equation_solution_l2499_249970

theorem power_equation_solution :
  ∀ m : ℤ, 3 * 2^2000 - 5 * 2^1999 + 4 * 2^1998 - 2^1997 = m * 2^1997 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2499_249970


namespace NUMINAMATH_CALUDE_min_value_fraction_l2499_249930

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 2) : 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 → (x + y) / (x * y * z) ≤ (a + b) / (a * b * c)) →
  (x + y) / (x * y * z) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2499_249930


namespace NUMINAMATH_CALUDE_transform_G_to_cup_l2499_249942

-- Define the set of shapes (including letters and symbols)
def Shape : Type := String

-- Define the transformations
def T₁ (s : Shape) : Shape := sorry
def T₂ (s : Shape) : Shape := sorry

-- Define the composition of transformations
def T (s : Shape) : Shape := T₂ (T₁ s)

-- State the theorem
theorem transform_G_to_cup (h1 : T₁ "R" = "y") (h2 : T₂ "y" = "B")
                           (h3 : T₁ "L" = "⌝") (h4 : T₂ "⌝" = "Γ") :
  T "G" = "∪" := by sorry

end NUMINAMATH_CALUDE_transform_G_to_cup_l2499_249942


namespace NUMINAMATH_CALUDE_inscribed_square_area_bound_l2499_249928

-- Define an acute triangle
def AcuteTriangle (A B C : Point) : Prop := sorry

-- Define a square
def Square (M N P Q : Point) : Prop := sorry

-- Define a point being on a line segment
def PointOnSegment (P A B : Point) : Prop := sorry

-- Define the area of a polygon
def Area (polygon : Set Point) : ℝ := sorry

theorem inscribed_square_area_bound 
  (A B C M N P Q : Point) 
  (h_acute : AcuteTriangle A B C)
  (h_square : Square M N P Q)
  (h_inscribed : PointOnSegment M B C ∧ PointOnSegment N B C ∧ 
                 PointOnSegment P A C ∧ PointOnSegment Q A B) :
  Area {M, N, P, Q} ≤ (1/2) * Area {A, B, C} := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_bound_l2499_249928


namespace NUMINAMATH_CALUDE_test_series_count_l2499_249957

/-- The number of tests in Professor Tester's series -/
def n : ℕ := 8

/-- John's average score if he scored 97 on the last test -/
def avg_with_97 : ℚ := 90

/-- John's average score if he scored 73 on the last test -/
def avg_with_73 : ℚ := 87

/-- The score difference between the two scenarios -/
def score_diff : ℚ := 97 - 73

/-- The average difference between the two scenarios -/
def avg_diff : ℚ := avg_with_97 - avg_with_73

theorem test_series_count :
  score_diff / (n + 1 : ℚ) = avg_diff :=
sorry

end NUMINAMATH_CALUDE_test_series_count_l2499_249957


namespace NUMINAMATH_CALUDE_power_sum_problem_l2499_249994

theorem power_sum_problem (a b : ℝ) 
  (h1 : a^5 + b^5 = 3) 
  (h2 : a^15 + b^15 = 9) : 
  a^10 + b^10 = 5 := by
sorry

end NUMINAMATH_CALUDE_power_sum_problem_l2499_249994


namespace NUMINAMATH_CALUDE_consecutive_odd_power_sum_divisible_l2499_249965

-- Define consecutive odd numbers
def ConsecutiveOddNumbers (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = 2*k + 1 ∧ b = 2*k + 3

-- Define divisibility
def Divides (d n : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Theorem statement
theorem consecutive_odd_power_sum_divisible (a b : ℕ) :
  ConsecutiveOddNumbers a b → Divides (a + b) (a^b + b^a) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_power_sum_divisible_l2499_249965


namespace NUMINAMATH_CALUDE_addition_problem_l2499_249901

theorem addition_problem : ∃! x : ℝ, 8 + x = -5 ∧ x = -13 := by sorry

end NUMINAMATH_CALUDE_addition_problem_l2499_249901


namespace NUMINAMATH_CALUDE_population_growth_rate_exists_and_unique_l2499_249982

theorem population_growth_rate_exists_and_unique :
  ∃! r : ℝ, 0 < r ∧ r < 1 ∧ 20000 * (1 + r)^3 = 26620 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_exists_and_unique_l2499_249982


namespace NUMINAMATH_CALUDE_greatest_power_of_two_l2499_249948

theorem greatest_power_of_two (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (12^603 - 8^402) ∧ 
   ∀ m : ℕ, 2^m ∣ (12^603 - 8^402) → m ≤ k) → 
  n = 1209 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_l2499_249948


namespace NUMINAMATH_CALUDE_remaining_fuel_after_three_hours_l2499_249944

/-- Represents the remaining fuel in a car's tank after driving for a certain time -/
def remaining_fuel (initial_fuel : ℝ) (consumption_rate : ℝ) (hours : ℝ) : ℝ :=
  initial_fuel - consumption_rate * hours

/-- Theorem stating that the remaining fuel after 3 hours matches the expression a-3b -/
theorem remaining_fuel_after_three_hours (a b : ℝ) :
  remaining_fuel a b 3 = a - 3 * b := by
  sorry

end NUMINAMATH_CALUDE_remaining_fuel_after_three_hours_l2499_249944


namespace NUMINAMATH_CALUDE_min_perimeter_special_triangle_l2499_249946

def triangle_perimeter (a b c : ℕ) : ℕ := a + b + c

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem min_perimeter_special_triangle :
  ∃ (c : ℕ), 
    is_valid_triangle 24 51 c ∧ 
    (∀ (x : ℕ), is_valid_triangle 24 51 x → triangle_perimeter 24 51 c ≤ triangle_perimeter 24 51 x) ∧
    triangle_perimeter 24 51 c = 103 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_special_triangle_l2499_249946


namespace NUMINAMATH_CALUDE_tv_and_radio_clients_l2499_249917

def total_clients : ℕ := 180
def tv_clients : ℕ := 115
def radio_clients : ℕ := 110
def magazine_clients : ℕ := 130
def tv_and_magazine : ℕ := 85
def radio_and_magazine : ℕ := 95
def all_three : ℕ := 80

theorem tv_and_radio_clients : 
  total_clients = tv_clients + radio_clients + magazine_clients - tv_and_magazine - radio_and_magazine - (tv_clients + radio_clients - total_clients) + all_three := by
  sorry

end NUMINAMATH_CALUDE_tv_and_radio_clients_l2499_249917


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2499_249905

/-- Given a sum at simple interest for 10 years, if increasing the interest rate by 5%
    results in Rs. 200 more interest, then the original sum is Rs. 2000. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 200 → P = 2000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2499_249905


namespace NUMINAMATH_CALUDE_log_sum_problem_l2499_249975

theorem log_sum_problem (x y : ℝ) (h1 : Real.log x / Real.log 4 + Real.log y / Real.log 4 = 1/2) (h2 : x = 12) :
  y = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_problem_l2499_249975


namespace NUMINAMATH_CALUDE_equation_solutions_l2499_249913

theorem equation_solutions :
  (∃ (x : ℝ), (1/2) * (2*x - 5)^2 - 2 = 0 ↔ x = 7/2 ∨ x = 3/2) ∧
  (∃ (x : ℝ), x^2 - 4*x - 4 = 0 ↔ x = 2 + 2*Real.sqrt 2 ∨ x = 2 - 2*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2499_249913


namespace NUMINAMATH_CALUDE_external_diagonals_inequality_five_seven_ten_not_valid_l2499_249922

/-- External diagonals of a right regular prism -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ
  a_le_b : a ≤ b
  b_le_c : b ≤ c
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- Theorem: For valid external diagonals of a right regular prism, a² + b² > c² -/
theorem external_diagonals_inequality (d : ExternalDiagonals) : d.a^2 + d.b^2 > d.c^2 := by
  sorry

/-- The set {5, 7, 10} cannot be the lengths of external diagonals of a right regular prism -/
theorem five_seven_ten_not_valid : ¬∃ (d : ExternalDiagonals), d.a = 5 ∧ d.b = 7 ∧ d.c = 10 := by
  sorry

end NUMINAMATH_CALUDE_external_diagonals_inequality_five_seven_ten_not_valid_l2499_249922


namespace NUMINAMATH_CALUDE_girls_in_class_l2499_249937

theorem girls_in_class (total : Nat) (prob : Rat) : 
  total = 25 → 
  prob = 3/25 → 
  (fun n : Nat => n * (n - 1) = prob * (total * (total - 1))) 9 → 
  total - 9 = 16 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_class_l2499_249937


namespace NUMINAMATH_CALUDE_no_quadratic_cycle_l2499_249955

theorem no_quadratic_cycle (f : ℝ → ℝ) (h : ∃ a b c : ℝ, f x = a * x^2 + b * x + c) :
  ¬ ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = b ∧ f b = c ∧ f c = a :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_cycle_l2499_249955


namespace NUMINAMATH_CALUDE_seashells_problem_l2499_249971

theorem seashells_problem (given_away : ℕ) (remaining : ℕ) :
  given_away = 18 → remaining = 17 → given_away + remaining = 35 :=
by sorry

end NUMINAMATH_CALUDE_seashells_problem_l2499_249971


namespace NUMINAMATH_CALUDE_calculate_value_probability_l2499_249906

def calculate_letters : Finset Char := {'C', 'A', 'L', 'C', 'U', 'L', 'A', 'T', 'E'}
def value_letters : Finset Char := {'V', 'A', 'L', 'U', 'E'}

theorem calculate_value_probability :
  (calculate_letters.filter (λ c => c ∈ value_letters)).card / calculate_letters.card = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_calculate_value_probability_l2499_249906


namespace NUMINAMATH_CALUDE_cubic_function_extreme_points_l2499_249907

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- Predicate stating that f has exactly two extreme points -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f' a x = 0 ∧ f' a y = 0 ∧
  ∀ z : ℝ, f' a z = 0 → z = x ∨ z = y

theorem cubic_function_extreme_points (a : ℝ) :
  has_two_extreme_points a → a < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_extreme_points_l2499_249907


namespace NUMINAMATH_CALUDE_total_books_l2499_249998

theorem total_books (sam_books joan_books : ℕ) 
  (h1 : sam_books = 110) 
  (h2 : joan_books = 102) : 
  sam_books + joan_books = 212 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l2499_249998


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l2499_249940

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (2/3 : ℂ) + (5/8 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (2/3 : ℂ) - (5/8 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l2499_249940


namespace NUMINAMATH_CALUDE_mixed_fruit_juice_cost_l2499_249927

/-- The cost per litre of the superfruit juice cocktail -/
def superfruit_cost : ℝ := 1399.45

/-- The cost per litre of the açaí berry juice -/
def acai_cost : ℝ := 3104.35

/-- The volume of mixed fruit juice used -/
def mixed_fruit_volume : ℝ := 33

/-- The volume of açaí berry juice used -/
def acai_volume : ℝ := 22

/-- The cost per litre of the mixed fruit juice -/
def mixed_fruit_cost : ℝ := 256.79

theorem mixed_fruit_juice_cost : 
  mixed_fruit_volume * mixed_fruit_cost + acai_volume * acai_cost = 
  (mixed_fruit_volume + acai_volume) * superfruit_cost :=
by sorry

end NUMINAMATH_CALUDE_mixed_fruit_juice_cost_l2499_249927


namespace NUMINAMATH_CALUDE_vector_equality_vector_equation_solution_l2499_249938

/-- Two vectors in ℝ² are equal if their corresponding components are equal -/
theorem vector_equality (a b c d : ℝ) : (a, b) = (c, d) ↔ a = c ∧ b = d := by sorry

/-- Definition of Vector1 -/
def Vector1 (u : ℝ) : ℝ × ℝ := (3 + 5*u, -1 - 3*u)

/-- Definition of Vector2 -/
def Vector2 (v : ℝ) : ℝ × ℝ := (0 - 3*v, 2 + 4*v)

theorem vector_equation_solution :
  ∃ (u v : ℝ), Vector1 u = Vector2 v ∧ u = -3/11 ∧ v = -16/11 := by sorry

end NUMINAMATH_CALUDE_vector_equality_vector_equation_solution_l2499_249938


namespace NUMINAMATH_CALUDE_complex_equality_implication_l2499_249990

theorem complex_equality_implication (x y : ℝ) : 
  (Complex.I * x + 2 = y - Complex.I) → (x - y = -3) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implication_l2499_249990


namespace NUMINAMATH_CALUDE_set_membership_and_inclusion_l2499_249935

def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}
def B : Set ℤ := {x | ∃ k : ℤ, x = 2*k + 1}

theorem set_membership_and_inclusion :
  (8 ∈ A ∧ 9 ∈ A ∧ 10 ∉ A) ∧ (∀ x : ℤ, x ∈ A → x ∈ B) := by sorry

end NUMINAMATH_CALUDE_set_membership_and_inclusion_l2499_249935


namespace NUMINAMATH_CALUDE_cone_height_l2499_249983

-- Define the cone
structure Cone where
  surfaceArea : ℝ
  centralAngle : ℝ

-- Theorem statement
theorem cone_height (c : Cone) 
  (h1 : c.surfaceArea = π) 
  (h2 : c.centralAngle = 2 * π / 3) : 
  ∃ h : ℝ, h = Real.sqrt 2 ∧ h > 0 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l2499_249983


namespace NUMINAMATH_CALUDE_problem_proof_l2499_249909

theorem problem_proof : Real.sqrt 8 - 4 * Real.sin (π / 4) - (1 / 3)⁻¹ = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2499_249909


namespace NUMINAMATH_CALUDE_library_configuration_count_l2499_249919

/-- The number of different configurations for 8 identical books in a library,
    where at least one book must remain in the library and at least one must be checked out. -/
def library_configurations : ℕ := 7

/-- The total number of books in the library -/
def total_books : ℕ := 8

/-- Proposition that there are exactly 7 different configurations for the books in the library -/
theorem library_configuration_count :
  (∀ config : ℕ, 1 ≤ config ∧ config ≤ total_books - 1) →
  (∀ config : ℕ, config ≤ total_books - config) →
  library_configurations = (total_books - 1) := by
  sorry

end NUMINAMATH_CALUDE_library_configuration_count_l2499_249919


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l2499_249978

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l2499_249978


namespace NUMINAMATH_CALUDE_exists_twelve_digit_non_cube_l2499_249997

theorem exists_twelve_digit_non_cube : ∃ n : ℕ, (10^11 ≤ n ∧ n < 10^12) ∧ ¬∃ k : ℕ, n = k^3 := by
  sorry

end NUMINAMATH_CALUDE_exists_twelve_digit_non_cube_l2499_249997


namespace NUMINAMATH_CALUDE_min_omega_value_l2499_249943

/-- Given a function f(x) = 2 * sin(ω * x) where ω > 0, and f(x) has a minimum value of -2
    in the interval [-π/3, π/6], prove that the minimum value of ω is 3/2. -/
theorem min_omega_value (ω : ℝ) : 
  (ω > 0) →
  (∀ x ∈ Set.Icc (-π/3) (π/6), 2 * Real.sin (ω * x) ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/6), 2 * Real.sin (ω * x) = -2) →
  ω ≥ 3/2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_value_l2499_249943


namespace NUMINAMATH_CALUDE_find_s_value_l2499_249904

/-- Given a relationship between R, S, and T, prove that S = 3/2 when R = 18 and T = 2 -/
theorem find_s_value (k : ℝ) : 
  (2 = k * 1^2 / 8) →  -- When R = 2, S = 1, and T = 8
  (18 = k * S^2 / 2) →  -- When R = 18 and T = 2
  S = 3/2 := by sorry

end NUMINAMATH_CALUDE_find_s_value_l2499_249904


namespace NUMINAMATH_CALUDE_quadrilateral_is_rhombus_l2499_249916

theorem quadrilateral_is_rhombus (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = a*b + b*c + c*d + d*a) : 
  a = b ∧ b = c ∧ c = d := by
  sorry

-- The theorem states that if the given condition is true,
-- then all sides of the quadrilateral are equal,
-- which is the definition of a rhombus.

end NUMINAMATH_CALUDE_quadrilateral_is_rhombus_l2499_249916


namespace NUMINAMATH_CALUDE_min_value_of_M_l2499_249960

theorem min_value_of_M (a b : ℕ+) : 
  ∃ (m : ℕ), m = 3 * a.val ^ 2 - a.val * b.val ^ 2 - 2 * b.val - 4 ∧ 
  m ≥ 2 ∧ 
  ∀ (k : ℕ), k = 3 * a.val ^ 2 - a.val * b.val ^ 2 - 2 * b.val - 4 → k ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_M_l2499_249960


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2499_249986

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | x^2 - x - 2 > 0}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2499_249986


namespace NUMINAMATH_CALUDE_swords_per_orc_l2499_249967

theorem swords_per_orc (total_swords : ℕ) (num_squads : ℕ) (orcs_per_squad : ℕ) :
  total_swords = 1200 →
  num_squads = 10 →
  orcs_per_squad = 8 →
  total_swords / (num_squads * orcs_per_squad) = 15 := by
  sorry

end NUMINAMATH_CALUDE_swords_per_orc_l2499_249967


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2499_249929

/-- Given a square ABCD with side length 1, E is the midpoint of AB, 
    F is the intersection of ED and AC, and G is the intersection of EC and BD. 
    The radius r of the circle inscribed in quadrilateral EFPG is equal to |EF| - |FP|. -/
theorem inscribed_circle_radius (A B C D E F G P : ℝ × ℝ) (r : ℝ) : 
  A = (0, 1) →
  B = (1, 1) →
  C = (1, 0) →
  D = (0, 0) →
  E = (1/2, 1) →
  F = (0, 1) →
  G = (2/3, 2/3) →
  P = (1/2, 1/2) →
  r = |EF| - |FP| :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2499_249929


namespace NUMINAMATH_CALUDE_half_circle_is_300_clerts_l2499_249996

-- Define the number of clerts in a full circle
def full_circle_clerts : ℕ := 600

-- Define a half-circle as half of a full circle
def half_circle_clerts : ℕ := full_circle_clerts / 2

-- Theorem to prove
theorem half_circle_is_300_clerts : half_circle_clerts = 300 := by
  sorry

end NUMINAMATH_CALUDE_half_circle_is_300_clerts_l2499_249996


namespace NUMINAMATH_CALUDE_fraction_equality_implies_sum_l2499_249991

theorem fraction_equality_implies_sum (C D : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
    (D * x - 17) / (x^2 - 8*x + 15) = C / (x - 3) + 2 / (x - 5)) →
  C + D = 32/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_sum_l2499_249991


namespace NUMINAMATH_CALUDE_pipe_A_fill_time_l2499_249936

/-- The time (in hours) taken by pipe B to empty the full cistern -/
def time_B : ℝ := 25

/-- The time (in hours) taken to fill the cistern when both pipes are opened -/
def time_both : ℝ := 99.99999999999999

/-- The time (in hours) taken by pipe A to fill the cistern -/
def time_A : ℝ := 20

/-- Theorem stating that the time taken by pipe A to fill the cistern is 20 hours -/
theorem pipe_A_fill_time :
  (1 / time_A - 1 / time_B) * time_both = 1 :=
sorry

end NUMINAMATH_CALUDE_pipe_A_fill_time_l2499_249936


namespace NUMINAMATH_CALUDE_parallel_lines_circle_distance_l2499_249977

theorem parallel_lines_circle_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ),
    chord1 = 38 ∧ 
    chord2 = 38 ∧ 
    chord3 = 34 ∧
    chord1 * 38 * chord1 / 4 + (d / 2) * 38 * (d / 2) = chord1 * r^2 ∧
    chord3 * 34 * chord3 / 4 + (3 * d / 2) * 34 * (3 * d / 2) = chord3 * r^2) →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_circle_distance_l2499_249977


namespace NUMINAMATH_CALUDE_product_real_iff_condition_l2499_249985

/-- For complex numbers z₁ = a + bi and z₂ = c + di, where a, b, c, and d are real numbers,
    the product z₁ * z₂ is real if and only if ad + bc = 0. -/
theorem product_real_iff_condition (a b c d : ℝ) :
  (Complex.I * Complex.I = -1) →
  let z₁ : ℂ := Complex.mk a b
  let z₂ : ℂ := Complex.mk c d
  (z₁ * z₂).im = 0 ↔ a * d + b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_real_iff_condition_l2499_249985


namespace NUMINAMATH_CALUDE_rainwater_solution_l2499_249999

/-- A tank collecting rainwater over three days -/
structure RainwaterTank where
  capacity : ℝ
  initialFill : ℝ
  day1Collection : ℝ
  day2Collection : ℝ
  day3Excess : ℝ

/-- The conditions of the rainwater collection problem -/
def rainProblem (tank : RainwaterTank) : Prop :=
  tank.capacity = 100 ∧
  tank.initialFill = 2/5 * tank.capacity ∧
  tank.day2Collection = tank.day1Collection + 5 ∧
  tank.initialFill + tank.day1Collection + tank.day2Collection = tank.capacity ∧
  tank.day3Excess = 25

/-- The theorem stating the solution to the rainwater problem -/
theorem rainwater_solution (tank : RainwaterTank) 
  (h : rainProblem tank) : tank.day1Collection = 27.5 := by
  sorry


end NUMINAMATH_CALUDE_rainwater_solution_l2499_249999


namespace NUMINAMATH_CALUDE_apple_distribution_l2499_249934

theorem apple_distribution (boxes : Nat) (apples_per_box : Nat) (rotten_apples : Nat) (people : Nat) :
  boxes = 7 →
  apples_per_box = 9 →
  rotten_apples = 7 →
  people = 8 →
  (boxes * apples_per_box - rotten_apples) / people = 7 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l2499_249934


namespace NUMINAMATH_CALUDE_probability_green_jellybean_l2499_249988

def total_jellybeans : ℕ := 7 + 9 + 8 + 10 + 6
def green_jellybeans : ℕ := 9

theorem probability_green_jellybean :
  (green_jellybeans : ℚ) / (total_jellybeans : ℚ) = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_jellybean_l2499_249988


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2499_249968

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = 3 * 2^n + k,
    prove that if {a_n} is a geometric sequence, then k = -3. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  (∀ n, S n = 3 * 2^n + k) →
  (∀ n, a n = S n - S (n-1)) →
  (∀ n, n ≥ 2 → a n * a (n-2) = (a (n-1))^2) →
  k = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2499_249968


namespace NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l2499_249952

theorem four_digit_multiples_of_seven (n : ℕ) : 
  (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 7 = 0) ↔ 
  (n ∈ Finset.range 1286 ∧ ∃ k : ℕ, n = 7 * k + 1001) :=
sorry

end NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l2499_249952


namespace NUMINAMATH_CALUDE_curve_transformation_l2499_249984

theorem curve_transformation (x : ℝ) : 2 * Real.cos (2 * (x - π/3)) = Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l2499_249984


namespace NUMINAMATH_CALUDE_gummy_worms_problem_l2499_249980

theorem gummy_worms_problem (x : ℝ) : (x / 2^4 = 4) → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_gummy_worms_problem_l2499_249980


namespace NUMINAMATH_CALUDE_factorization_left_to_right_l2499_249939

theorem factorization_left_to_right : 
  ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_factorization_left_to_right_l2499_249939


namespace NUMINAMATH_CALUDE_right_triangle_leg_ratio_l2499_249979

theorem right_triangle_leg_ratio (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_projection : (c - b^2 / c) / (b^2 / c) = 4) : b / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_ratio_l2499_249979


namespace NUMINAMATH_CALUDE_function_not_satisfying_condition_l2499_249989

theorem function_not_satisfying_condition :
  ∃ f : ℝ → ℝ, (∀ x, f x = x + 1) ∧ (∃ x, f (2 * x) ≠ 2 * f x) := by
  sorry

end NUMINAMATH_CALUDE_function_not_satisfying_condition_l2499_249989


namespace NUMINAMATH_CALUDE_max_value_of_one_minus_cos_l2499_249941

open Real

theorem max_value_of_one_minus_cos (x : ℝ) :
  ∃ (k : ℤ), (∀ y : ℝ, 1 - cos y ≤ 1 - cos (π + 2 * π * ↑k)) ∧
              (1 - cos x = 1 - cos (π + 2 * π * ↑k) ↔ ∃ m : ℤ, x = π + 2 * π * ↑m) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_one_minus_cos_l2499_249941


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_I_simplify_trigonometric_expression_II_l2499_249951

-- Part I
theorem simplify_trigonometric_expression_I :
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (20 * π / 180))) /
  (Real.sin (160 * π / 180) - Real.sqrt (1 - Real.sin (20 * π / 180) ^ 2)) = -1 := by sorry

-- Part II
theorem simplify_trigonometric_expression_II (α : Real) (h : π / 2 < α ∧ α < π) :
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) +
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) =
  Real.sin α - Real.cos α := by sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_I_simplify_trigonometric_expression_II_l2499_249951


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l2499_249969

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 9 = 5 ∧ ∀ m, m < 100 ∧ m % 9 = 5 → m ≤ n ↔ n = 95 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l2499_249969


namespace NUMINAMATH_CALUDE_catrionas_aquarium_l2499_249963

/-- The number of goldfish in Catriona's aquarium -/
def num_goldfish : ℕ := 8

/-- The number of angelfish in Catriona's aquarium -/
def num_angelfish : ℕ := num_goldfish + 4

/-- The number of guppies in Catriona's aquarium -/
def num_guppies : ℕ := 2 * num_angelfish

/-- The total number of fish in Catriona's aquarium -/
def total_fish : ℕ := num_goldfish + num_angelfish + num_guppies

theorem catrionas_aquarium : total_fish = 44 := by
  sorry

end NUMINAMATH_CALUDE_catrionas_aquarium_l2499_249963


namespace NUMINAMATH_CALUDE_largest_number_l2499_249953

/-- Represents a repeating decimal number with an integer part and a fractional part -/
structure RepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : ℕ
  repeatingPart : ℕ
  nonRepeatingDigits : ℕ
  repeatingDigits : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (r : RepeatingDecimal) : ℚ :=
  sorry

/-- The number 7.45678 -/
def a : ℚ := 745678 / 100000

/-- The number 7.456̅7 -/
def b : RepeatingDecimal := ⟨7, 456, 7, 3, 1⟩

/-- The number 7.45̅67 -/
def c : RepeatingDecimal := ⟨7, 45, 67, 2, 2⟩

/-- The number 7.4̅567 -/
def d : RepeatingDecimal := ⟨7, 4, 567, 1, 3⟩

/-- The number 7.̅4567 -/
def e : RepeatingDecimal := ⟨7, 0, 4567, 0, 4⟩

theorem largest_number :
  b.toRational > a ∧
  b.toRational > c.toRational ∧
  b.toRational > d.toRational ∧
  b.toRational > e.toRational :=
sorry

end NUMINAMATH_CALUDE_largest_number_l2499_249953


namespace NUMINAMATH_CALUDE_parabola_point_order_l2499_249954

-- Define the parabola function
def f (x : ℝ) : ℝ := -2 * (x + 1)^2 - 1

-- Define the points A, B, and C
def A : ℝ × ℝ := (-3, f (-3))
def B : ℝ × ℝ := (-2, f (-2))
def C : ℝ × ℝ := (2, f 2)

-- Extract y-coordinates
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

-- Theorem statement
theorem parabola_point_order : y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_order_l2499_249954


namespace NUMINAMATH_CALUDE_temperature_difference_l2499_249973

theorem temperature_difference (M L N : ℝ) : 
  (M = L + N) →  -- Minneapolis is N degrees warmer than St. Louis at noon
  (|((L + N) - 6) - (L + 4)| = 3) →  -- Temperature difference at 5:00 PM
  (N = 13 ∨ N = 7) ∧ (13 * 7 = 91) :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_l2499_249973


namespace NUMINAMATH_CALUDE_john_index_cards_l2499_249931

/-- Given that John buys 2 packs for each student, has 6 classes, and each class has 30 students,
    prove that the total number of packs John bought is 360. -/
theorem john_index_cards (packs_per_student : ℕ) (num_classes : ℕ) (students_per_class : ℕ)
  (h1 : packs_per_student = 2)
  (h2 : num_classes = 6)
  (h3 : students_per_class = 30) :
  packs_per_student * num_classes * students_per_class = 360 := by
  sorry

end NUMINAMATH_CALUDE_john_index_cards_l2499_249931


namespace NUMINAMATH_CALUDE_number_comparison_l2499_249959

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 2 to base 10 -/
def base2ToBase10 (n : ℕ) : ℕ := sorry

theorem number_comparison :
  let a : ℕ := 33
  let b : ℕ := base6ToBase10 52
  let c : ℕ := base2ToBase10 11111
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_number_comparison_l2499_249959


namespace NUMINAMATH_CALUDE_min_area_theorem_l2499_249932

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Represents the configuration of shapes within the larger square -/
structure ShapeConfiguration where
  largeSquare : Square
  rectangle1 : Rectangle
  square1 : Square
  rectangleR : Rectangle

/-- The theorem statement -/
theorem min_area_theorem (config : ShapeConfiguration) : 
  config.rectangle1.width = 1 ∧ 
  config.rectangle1.height = 4 ∧
  config.square1.side = 1 ∧
  config.largeSquare.side ≥ 4 →
  config.largeSquare.side ^ 2 ≥ 16 ∧
  config.rectangleR.width * config.rectangleR.height = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_area_theorem_l2499_249932


namespace NUMINAMATH_CALUDE_rose_bushes_count_l2499_249900

/-- The number of rose bushes in the park after planting -/
def final_roses : ℕ := 6

/-- The number of new rose bushes to be planted -/
def new_roses : ℕ := 4

/-- The number of rose bushes currently in the park -/
def current_roses : ℕ := final_roses - new_roses

theorem rose_bushes_count : current_roses = 2 := by
  sorry

end NUMINAMATH_CALUDE_rose_bushes_count_l2499_249900


namespace NUMINAMATH_CALUDE_ab_is_zero_l2499_249981

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Given complex equation -/
def complex_equation (a b : ℝ) : Prop :=
  (1 + i) / (1 - i) = (a : ℂ) + b * i

/-- Theorem stating that if the complex equation holds, then ab = 0 -/
theorem ab_is_zero (a b : ℝ) (h : complex_equation a b) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_is_zero_l2499_249981


namespace NUMINAMATH_CALUDE_number_division_remainder_l2499_249950

theorem number_division_remainder (N : ℕ) : 
  (N / 5 = 5 ∧ N % 5 = 0) → N % 11 = 3 := by
sorry

end NUMINAMATH_CALUDE_number_division_remainder_l2499_249950


namespace NUMINAMATH_CALUDE_multiply_by_eleven_l2499_249961

theorem multiply_by_eleven (A B : Nat) (h1 : A < 10) (h2 : B < 10) (h3 : A + B < 10) :
  (10 * A + B) * 11 = 100 * A + 10 * (A + B) + B := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_eleven_l2499_249961


namespace NUMINAMATH_CALUDE_three_digit_multiple_of_seven_l2499_249992

theorem three_digit_multiple_of_seven :
  ∃! D : ℕ, D < 10 ∧ (400 + 10 * D + 5) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_multiple_of_seven_l2499_249992


namespace NUMINAMATH_CALUDE_grid_coloring_4x2011_l2499_249966

/-- Represents the number of ways to color a 4 × n grid with the given constraints -/
def coloringWays (n : ℕ) : ℕ :=
  64 * 3^(2*n)

/-- The problem statement -/
theorem grid_coloring_4x2011 :
  coloringWays 2011 = 64 * 3^4020 :=
by sorry

end NUMINAMATH_CALUDE_grid_coloring_4x2011_l2499_249966


namespace NUMINAMATH_CALUDE_bedevir_will_participate_l2499_249945

/-- The combat skill of the n-th opponent -/
def opponent_skill (n : ℕ) : ℚ := 1 / (2^(n+1) - 1)

/-- The probability of Sir Bedevir winning against the n-th opponent -/
def win_probability (n : ℕ) : ℚ := 1 / (1 + opponent_skill n)

/-- Theorem: Sir Bedevir's probability of winning is greater than 1/2 for any opponent -/
theorem bedevir_will_participate (k : ℕ) (h : k > 1) :
  ∀ n, n < k → win_probability n > 1/2 := by sorry

end NUMINAMATH_CALUDE_bedevir_will_participate_l2499_249945


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2499_249956

theorem complex_expression_simplification :
  (3/2)^0 - (1 - 0.5^(-2)) / ((27/8)^(2/3)) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2499_249956


namespace NUMINAMATH_CALUDE_kids_waiting_swings_is_three_l2499_249915

/-- The number of kids waiting for the swings -/
def kids_waiting_swings : ℕ := sorry

/-- The number of kids waiting for the slide -/
def kids_waiting_slide : ℕ := 2 * kids_waiting_swings

/-- The wait time for the swings in seconds -/
def wait_time_swings : ℕ := 120 * kids_waiting_swings

/-- The wait time for the slide in seconds -/
def wait_time_slide : ℕ := 15 * kids_waiting_slide

/-- The difference between the longer and shorter wait times -/
def wait_time_difference : ℕ := 270

theorem kids_waiting_swings_is_three :
  kids_waiting_swings = 3 ∧
  kids_waiting_slide = 2 * kids_waiting_swings ∧
  wait_time_swings = 120 * kids_waiting_swings ∧
  wait_time_slide = 15 * kids_waiting_slide ∧
  wait_time_swings - wait_time_slide = wait_time_difference :=
by sorry

end NUMINAMATH_CALUDE_kids_waiting_swings_is_three_l2499_249915


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l2499_249976

-- Define the function f
def f (x a : ℝ) := |2*x - a| + |x - 3*a|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ |x - a/2| + a^2 + 1} = 
  {a : ℝ | (-2 ≤ a ∧ a ≤ -1/2) ∨ (1/2 ≤ a ∧ a ≤ 2)} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l2499_249976


namespace NUMINAMATH_CALUDE_fraction_equality_l2499_249933

theorem fraction_equality : (5 * 3 + 4) / 7 = 19 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2499_249933


namespace NUMINAMATH_CALUDE_initial_deadlift_weight_l2499_249912

def initial_squat : ℝ := 700
def initial_bench : ℝ := 400
def squat_loss_percentage : ℝ := 30
def deadlift_loss : ℝ := 200
def new_total : ℝ := 1490

theorem initial_deadlift_weight :
  ∃ (initial_deadlift : ℝ),
    initial_deadlift - deadlift_loss +
    initial_bench +
    initial_squat * (1 - squat_loss_percentage / 100) = new_total ∧
    initial_deadlift = 800 := by
  sorry

end NUMINAMATH_CALUDE_initial_deadlift_weight_l2499_249912


namespace NUMINAMATH_CALUDE_inverse_proportion_increasing_l2499_249902

/-- For an inverse proportion function y = (m-5)/x, if y increases as x increases on each branch of its graph, then m < 5 -/
theorem inverse_proportion_increasing (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≠ 0 → x₂ ≠ 0 → x₁ < x₂ → (m - 5) / x₁ < (m - 5) / x₂) → 
  m < 5 :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_increasing_l2499_249902


namespace NUMINAMATH_CALUDE_flow_rate_difference_l2499_249920

/-- Proves that the difference between 0.6 times the original flow rate and the reduced flow rate is 1 gallon per minute -/
theorem flow_rate_difference (original_rate reduced_rate : ℝ) 
  (h1 : original_rate = 5.0)
  (h2 : reduced_rate = 2) : 
  0.6 * original_rate - reduced_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_flow_rate_difference_l2499_249920


namespace NUMINAMATH_CALUDE_students_doing_hula_hoops_l2499_249914

theorem students_doing_hula_hoops 
  (jumping_rope : ℕ) 
  (hula_hoop_ratio : ℕ) 
  (h1 : jumping_rope = 7)
  (h2 : hula_hoop_ratio = 5) :
  jumping_rope * hula_hoop_ratio = 35 := by
  sorry

end NUMINAMATH_CALUDE_students_doing_hula_hoops_l2499_249914


namespace NUMINAMATH_CALUDE_equation_solution_l2499_249993

theorem equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  let k := 5 * (m^2 - n^2)
  let x := (5 * m^2 - 9 * n^2) / (4 * m + 6 * n)
  (x + 2 * m)^2 - (x - 3 * n)^2 = k :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2499_249993
