import Mathlib

namespace NUMINAMATH_CALUDE_quotient_problem_l3955_395548

theorem quotient_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ n : ℤ, a / b = n) (h4 : a / b = a / 2 ∨ a / b = 6 * b) : 
  a / b = 12 := by
sorry

end NUMINAMATH_CALUDE_quotient_problem_l3955_395548


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3955_395509

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = (2 + Real.sqrt 14) / 2 ∧ 
  x₂ = (2 - Real.sqrt 14) / 2 ∧ 
  2 * x₁^2 - 4 * x₁ - 5 = 0 ∧ 
  2 * x₂^2 - 4 * x₂ - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3955_395509


namespace NUMINAMATH_CALUDE_binomial_coefficient_fifth_power_fourth_term_l3955_395523

theorem binomial_coefficient_fifth_power_fourth_term : 
  Nat.choose 5 3 = 10 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_fifth_power_fourth_term_l3955_395523


namespace NUMINAMATH_CALUDE_thirtieth_term_is_59_l3955_395504

/-- A sequence where each term is 2 more than the previous term, starting with 1 -/
def counting_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => counting_sequence n + 2

/-- The 30th term of the counting sequence is 59 -/
theorem thirtieth_term_is_59 : counting_sequence 29 = 59 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_is_59_l3955_395504


namespace NUMINAMATH_CALUDE_original_car_price_l3955_395584

theorem original_car_price (used_price : ℝ) (percentage : ℝ) (original_price : ℝ) : 
  used_price = 15000 →
  percentage = 0.40 →
  used_price = percentage * original_price →
  original_price = 37500 := by
sorry

end NUMINAMATH_CALUDE_original_car_price_l3955_395584


namespace NUMINAMATH_CALUDE_nested_root_equality_l3955_395555

theorem nested_root_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ 7) ^ (1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_nested_root_equality_l3955_395555


namespace NUMINAMATH_CALUDE_factorization_of_4a_minus_a_cubed_l3955_395574

theorem factorization_of_4a_minus_a_cubed (a : ℝ) : 4*a - a^3 = a*(2-a)*(2+a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4a_minus_a_cubed_l3955_395574


namespace NUMINAMATH_CALUDE_diagonals_in_30_sided_polygon_l3955_395569

theorem diagonals_in_30_sided_polygon : ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_in_30_sided_polygon_l3955_395569


namespace NUMINAMATH_CALUDE_inequality_solution_l3955_395582

theorem inequality_solution (a : ℝ) :
  (a = 0 → ¬∃ x, (1 - a * x)^2 < 1) ∧
  (a < 0 → ∀ x, (1 - a * x)^2 < 1 ↔ (2 / a < x ∧ x < 0)) ∧
  (a > 0 → ∀ x, (1 - a * x)^2 < 1 ↔ (0 < x ∧ x < 2 / a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3955_395582


namespace NUMINAMATH_CALUDE_fruit_garden_ratio_l3955_395533

/-- Given a garden with the specified conditions, prove the ratio of fruit section to whole garden --/
theorem fruit_garden_ratio 
  (total_area : ℝ) 
  (fruit_quarter : ℝ) 
  (h1 : total_area = 64) 
  (h2 : fruit_quarter = 8) : 
  (4 * fruit_quarter) / total_area = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_garden_ratio_l3955_395533


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3955_395517

/-- Given a polynomial P(x) that satisfies P(x) - 3x^2 = x^2 - (1/2)x + 1,
    prove that (-3x^2) * P(x) = -12x^4 + (3/2)x^3 - 3x^2 -/
theorem polynomial_multiplication (x : ℝ) (P : ℝ → ℝ) 
    (h : P x - 3 * x^2 = x^2 - (1/2) * x + 1) :
  (-3 * x^2) * P x = -12 * x^4 + (3/2) * x^3 - 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3955_395517


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3955_395588

/-- A quadratic function passing through given points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  point_1 : a * (-3)^2 + b * (-3) + c = 0
  point_2 : a * (-2)^2 + b * (-2) + c = -3
  point_3 : a * (-1)^2 + b * (-1) + c = -4
  point_4 : c = -3

/-- Statements about the quadratic function -/
def statements (f : QuadraticFunction) : Fin 4 → Prop
  | 0 => f.a * f.c < 0
  | 1 => ∀ x > 1, ∀ y > x, f.a * y^2 + f.b * y + f.c > f.a * x^2 + f.b * x + f.c
  | 2 => f.a * (-4)^2 + (f.b - 4) * (-4) + f.c = 0
  | 3 => ∀ x, -1 < x → x < 0 → f.a * x^2 + (f.b - 1) * x + f.c + 3 > 0

/-- The main theorem -/
theorem quadratic_function_theorem (f : QuadraticFunction) :
  ∃ (S : Finset (Fin 4)), S.card = 2 ∧ (∀ i, i ∈ S ↔ statements f i) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3955_395588


namespace NUMINAMATH_CALUDE_cube_root_scaling_l3955_395513

theorem cube_root_scaling (a b c d : ℝ) (ha : a > 0) (hc : c > 0) :
  (a^(1/3) = b) → (c^(1/3) = d) →
  ((1000 * a)^(1/3) = 10 * b) ∧ ((-0.001 * c)^(1/3) = -0.1 * d) := by
  sorry

/- The theorem above captures the essence of the problem without directly using the specific numbers.
   It shows the scaling properties of cube roots that are used to solve the original problem. -/

end NUMINAMATH_CALUDE_cube_root_scaling_l3955_395513


namespace NUMINAMATH_CALUDE_marble_bag_size_l3955_395521

/-- Represents a bag of marbles with blue, red, and white colors. -/
structure MarbleBag where
  total : ℕ
  blue : ℕ
  red : ℕ
  white : ℕ

/-- The probability of selecting a red or white marble from the bag. -/
def redOrWhiteProbability (bag : MarbleBag) : ℚ :=
  (bag.red + bag.white : ℚ) / bag.total

theorem marble_bag_size :
  ∃ (bag : MarbleBag),
    bag.blue = 5 ∧
    bag.red = 7 ∧
    redOrWhiteProbability bag = 3/4 ∧
    bag.total = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_bag_size_l3955_395521


namespace NUMINAMATH_CALUDE_first_day_is_saturday_l3955_395573

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure MonthDay where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the day of the week for a given day number -/
def getDayOfWeek (dayNumber : Nat) : DayOfWeek := sorry

/-- Theorem stating that if the 25th is a Tuesday, the 1st is a Saturday -/
theorem first_day_is_saturday 
  (h : getDayOfWeek 25 = DayOfWeek.Tuesday) : 
  getDayOfWeek 1 = DayOfWeek.Saturday := by
  sorry

end NUMINAMATH_CALUDE_first_day_is_saturday_l3955_395573


namespace NUMINAMATH_CALUDE_circle_area_relation_l3955_395568

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_relation :
  ∀ (r_A r_B : ℝ),
  circle_area r_A = 9 →
  r_A = r_B / 2 →
  circle_area r_B = 36 := by
sorry

end NUMINAMATH_CALUDE_circle_area_relation_l3955_395568


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3955_395539

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The arithmetic sequence condition -/
def ArithmeticCondition (a : ℕ → ℝ) : Prop :=
  2 * ((1/2) * a 3) = a 1 + 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticCondition a →
  (∀ n : ℕ, a n > 0) →
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3955_395539


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3955_395540

-- Define set M
def M : Set ℝ := {x | x * (x - 5) ≤ 6}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3955_395540


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3955_395572

theorem polynomial_factorization (x : ℝ) :
  4 * (x + 5) * (x + 6) * (x + 10) * (x + 12) - 3 * x^2 =
  (2 * x^2 + 35 * x + 120) * (x + 8) * (2 * x + 15) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3955_395572


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3955_395516

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + x - 2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l3955_395516


namespace NUMINAMATH_CALUDE_tim_pay_per_task_l3955_395571

/-- Represents the pay per task for Tim's work --/
def pay_per_task (tasks_per_day : ℕ) (days_per_week : ℕ) (weekly_pay : ℚ) : ℚ :=
  weekly_pay / (tasks_per_day * days_per_week)

/-- Theorem stating that Tim's pay per task is $1.20 --/
theorem tim_pay_per_task :
  pay_per_task 100 6 720 = 1.20 := by
  sorry

end NUMINAMATH_CALUDE_tim_pay_per_task_l3955_395571


namespace NUMINAMATH_CALUDE_continuity_at_negative_one_l3955_395570

noncomputable def f (x : ℝ) : ℝ := (x^3 - x^2 + x + 1) / (x^2 - 1)

theorem continuity_at_negative_one :
  Filter.Tendsto f (nhds (-1)) (nhds (-2)) :=
sorry

end NUMINAMATH_CALUDE_continuity_at_negative_one_l3955_395570


namespace NUMINAMATH_CALUDE_train_speed_l3955_395514

-- Define the length of the train in meters
def train_length : ℝ := 130

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 3.249740020798336

-- Define the conversion factor from m/s to km/hr
def ms_to_kmhr : ℝ := 3.6

-- Theorem to prove the train's speed
theorem train_speed : 
  (train_length / crossing_time) * ms_to_kmhr = 144 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3955_395514


namespace NUMINAMATH_CALUDE_average_monthly_balance_l3955_395594

def january_balance : ℝ := 120
def february_balance : ℝ := 250
def march_balance : ℝ := 200
def april_balance : ℝ := 200
def may_balance : ℝ := 180
def num_months : ℝ := 5

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance + may_balance) / num_months = 190 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l3955_395594


namespace NUMINAMATH_CALUDE_class_size_possibilities_l3955_395503

theorem class_size_possibilities (N : ℕ) : 
  (∃ k : ℕ, N = 8 + k) →  -- Total students is 8 bullies plus some honor students
  (7 : ℚ) / (N - 1 : ℚ) < (1 : ℚ) / 3 →  -- Bullies' condition
  (8 : ℚ) / (N - 1 : ℚ) ≥ (1 : ℚ) / 3 →  -- Honor students' condition
  N ∈ ({23, 24, 25} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_class_size_possibilities_l3955_395503


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3955_395501

theorem opposite_of_2023 : 
  (∀ x : ℤ, x + 2023 = 0 → x = -2023) ∧ (-2023 + 2023 = 0) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3955_395501


namespace NUMINAMATH_CALUDE_smallest_divisible_by_20_and_36_l3955_395547

theorem smallest_divisible_by_20_and_36 : ∃ n : ℕ, n > 0 ∧ 20 ∣ n ∧ 36 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 20 ∣ m ∧ 36 ∣ m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_20_and_36_l3955_395547


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3955_395585

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3955_395585


namespace NUMINAMATH_CALUDE_total_jump_sequences_l3955_395598

-- Define a regular hexagon
structure RegularHexagon :=
  (vertices : Fin 6 → Point)

-- Define a frog's jump
inductive Jump
| clockwise
| counterclockwise

-- Define a sequence of jumps
def JumpSequence := List Jump

-- Define the result of a jump sequence
inductive JumpResult
| reachedD
| notReachedD

-- Function to determine the result of a jump sequence
def jumpSequenceResult (h : RegularHexagon) (js : JumpSequence) : JumpResult :=
  sorry

-- Function to count valid jump sequences
def countValidJumpSequences (h : RegularHexagon) : Nat :=
  sorry

-- The main theorem
theorem total_jump_sequences (h : RegularHexagon) :
  countValidJumpSequences h = 26 :=
sorry

end NUMINAMATH_CALUDE_total_jump_sequences_l3955_395598


namespace NUMINAMATH_CALUDE_third_element_in_tenth_bracket_l3955_395552

/-- The number of elements in the nth bracket -/
def bracket_size (n : ℕ) : ℕ := n

/-- The sum of elements in the first n brackets -/
def sum_bracket_sizes (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The last element in the nth bracket -/
def last_element_in_bracket (n : ℕ) : ℕ := sum_bracket_sizes n

theorem third_element_in_tenth_bracket :
  ∃ (k : ℕ), k = last_element_in_bracket 9 + 3 ∧ k = 48 :=
sorry

end NUMINAMATH_CALUDE_third_element_in_tenth_bracket_l3955_395552


namespace NUMINAMATH_CALUDE_helen_cookies_l3955_395545

/-- The number of raisin cookies Helen baked this morning -/
def raisin_cookies : ℕ := 231

/-- The difference between chocolate chip cookies and raisin cookies -/
def cookie_difference : ℕ := 25

/-- The number of chocolate chip cookies Helen baked this morning -/
def choc_chip_cookies : ℕ := raisin_cookies + cookie_difference

theorem helen_cookies : choc_chip_cookies = 256 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l3955_395545


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_10_and_6_l3955_395512

theorem smallest_common_multiple_of_10_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, (10 ∣ m) ∧ (6 ∣ m) → n ≤ m) ∧ (10 ∣ n) ∧ (6 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_10_and_6_l3955_395512


namespace NUMINAMATH_CALUDE_min_draws_for_all_colors_l3955_395527

theorem min_draws_for_all_colors (white black yellow : ℕ) 
  (hw : white = 8) (hb : black = 9) (hy : yellow = 7) :
  (white + black + yellow - (white + black - 1)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_all_colors_l3955_395527


namespace NUMINAMATH_CALUDE_function_zero_points_theorem_l3955_395506

open Real

theorem function_zero_points_theorem (f : ℝ → ℝ) (a : ℝ) (x₁ x₂ : ℝ) 
  (h_f : ∀ x, f x = log x - a * x)
  (h_zero : f x₁ = 0 ∧ f x₂ = 0)
  (h_distinct : x₁ < x₂) :
  (0 < a ∧ a < 1 / Real.exp 1) ∧ 
  (2 / (x₁ + x₂) < a) := by
  sorry

end NUMINAMATH_CALUDE_function_zero_points_theorem_l3955_395506


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3955_395592

theorem max_value_on_circle (x y : ℝ) : 
  x^2 + y^2 = 1 → (y / (x + 2) ≤ Real.sqrt 3 / 3) ∧ 
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 1 ∧ y₀ / (x₀ + 2) = Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3955_395592


namespace NUMINAMATH_CALUDE_perimeter_folded_square_l3955_395587

/-- Given a square ABCD with side length 2, where A is folded to meet BC at A' such that A'C = 1/2,
    the perimeter of triangle A'BD is (3 + √17)/2 + 2√2. -/
theorem perimeter_folded_square (A B C D A' : ℝ × ℝ) : 
  (∀ (X Y : ℝ × ℝ), ‖X - Y‖ = 2 → (X = A ∧ Y = B) ∨ (X = B ∧ Y = C) ∨ (X = C ∧ Y = D) ∨ (X = D ∧ Y = A)) →
  A'.1 = B.1 + 3/2 →
  A'.2 = B.2 →
  C.1 = B.1 + 2 →
  C.2 = B.2 →
  ‖A' - C‖ = 1/2 →
  ‖A' - B‖ + ‖B - D‖ + ‖D - A'‖ = (3 + Real.sqrt 17) / 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_folded_square_l3955_395587


namespace NUMINAMATH_CALUDE_total_fruits_eaten_l3955_395522

/-- Prove that the total number of fruits eaten by three dogs is 240 given the specified conditions -/
theorem total_fruits_eaten (dog1_apples dog2_blueberries dog3_bonnies : ℕ) : 
  dog3_bonnies = 60 →
  dog2_blueberries = (3 * dog3_bonnies) / 4 →
  dog1_apples = 3 * dog2_blueberries →
  dog1_apples + dog2_blueberries + dog3_bonnies = 240 := by
  sorry

#check total_fruits_eaten

end NUMINAMATH_CALUDE_total_fruits_eaten_l3955_395522


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l3955_395590

/-- The perpendicular bisector of a line segment passing through two points -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) (b : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = b

theorem perpendicular_bisector_value : 
  perpendicular_bisector 2 4 6 8 10 := by
  sorry

#check perpendicular_bisector_value

end NUMINAMATH_CALUDE_perpendicular_bisector_value_l3955_395590


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l3955_395546

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The symmetric point with respect to the y-axis --/
def symmetricYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The original point (2,5) --/
def originalPoint : Point :=
  { x := 2, y := 5 }

/-- The expected symmetric point (-2,5) --/
def expectedSymmetricPoint : Point :=
  { x := -2, y := 5 }

theorem symmetric_point_y_axis :
  symmetricYAxis originalPoint = expectedSymmetricPoint := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l3955_395546


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l3955_395581

def number_of_people : ℕ := 4

def total_arrangements (n : ℕ) : ℕ := n.factorial

def arrangements_with_pair_together (n : ℕ) : ℕ := (n - 1).factorial * 2

theorem seating_arrangements_with_restriction :
  total_arrangements number_of_people - arrangements_with_pair_together number_of_people = 12 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l3955_395581


namespace NUMINAMATH_CALUDE_area_of_overlapping_squares_l3955_395556

/-- The area covered by two overlapping congruent squares -/
theorem area_of_overlapping_squares (side_length : ℝ) (h : side_length = 12) :
  let square_area := side_length ^ 2
  let overlap_area := square_area / 4
  let total_area := 2 * square_area - overlap_area
  total_area = 252 := by sorry

end NUMINAMATH_CALUDE_area_of_overlapping_squares_l3955_395556


namespace NUMINAMATH_CALUDE_product_equals_three_eighths_l3955_395595

-- Define the fractions and mixed number
def a : ℚ := 1/2
def b : ℚ := 2/3
def c : ℚ := 3/4
def d : ℚ := 3/2  -- 1.5 as a fraction

-- State the theorem
theorem product_equals_three_eighths :
  a * b * c * d = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_three_eighths_l3955_395595


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_l3955_395502

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpoint_octagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

theorem midpoint_octagon_area (o : RegularOctagon) :
  area (midpoint_octagon o) = (1/4) * area o := by
  sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_l3955_395502


namespace NUMINAMATH_CALUDE_max_reverse_sum_theorem_l3955_395531

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def reverse_number (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem max_reverse_sum_theorem (a b : ℕ) 
  (h1 : is_three_digit a) 
  (h2 : is_three_digit b) 
  (h3 : a % 10 ≠ 0) 
  (h4 : b % 10 ≠ 0) 
  (h5 : a + b = 1372) : 
  ∃ (max : ℕ), reverse_number a + reverse_number b ≤ max ∧ max = 1372 := by
  sorry

end NUMINAMATH_CALUDE_max_reverse_sum_theorem_l3955_395531


namespace NUMINAMATH_CALUDE_basketball_handshakes_l3955_395543

theorem basketball_handshakes :
  let team_size : ℕ := 5
  let num_teams : ℕ := 2
  let num_referees : ℕ := 3
  let inter_team_handshakes := team_size * team_size
  let player_referee_handshakes := (team_size * num_teams) * num_referees
  inter_team_handshakes + player_referee_handshakes = 55 :=
by sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l3955_395543


namespace NUMINAMATH_CALUDE_inequality_proof_l3955_395593

theorem inequality_proof (a b c : ℝ) 
  (ha : 1 - a^2 ≥ 0) (hb : 1 - b^2 ≥ 0) (hc : 1 - c^2 ≥ 0) : 
  Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) ≤ 
  Real.sqrt (9 - (a + b + c)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3955_395593


namespace NUMINAMATH_CALUDE_equal_power_implies_equal_l3955_395530

theorem equal_power_implies_equal (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 0 < a) (h4 : a < 1) (h5 : a^b = b^a) : a = b := by
  sorry

end NUMINAMATH_CALUDE_equal_power_implies_equal_l3955_395530


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3955_395524

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 16*x + 15 = 0 → ∃ s₁ s₂ : ℝ, s₁^2 + s₂^2 = 226 ∧ (x = s₁ ∨ x = s₂) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3955_395524


namespace NUMINAMATH_CALUDE_simone_apple_fraction_l3955_395566

theorem simone_apple_fraction (x : ℚ) : 
  (16 * x + 15 * (1 / 3 : ℚ) = 13) → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simone_apple_fraction_l3955_395566


namespace NUMINAMATH_CALUDE_identical_solutions_l3955_395575

/-- Two equations have identical solutions when k = 0 -/
theorem identical_solutions (x y k : ℝ) : 
  (y = x^2 ∧ y = 3*x^2 + k) ↔ k = 0 :=
by sorry

end NUMINAMATH_CALUDE_identical_solutions_l3955_395575


namespace NUMINAMATH_CALUDE_parabola_two_distinct_roots_l3955_395538

/-- Given a real number m, the quadratic equation x^2 - (2m-1)x + (m^2 - m) = 0 has two distinct real roots. -/
theorem parabola_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁^2 - (2*m - 1)*x₁ + (m^2 - m) = 0 ∧
    x₂^2 - (2*m - 1)*x₂ + (m^2 - m) = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_two_distinct_roots_l3955_395538


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3955_395599

theorem quadratic_inequality_solution (a b : ℝ) (h : Set ℝ) : 
  (∀ x, x ∈ h ↔ (a * x^2 - 3*x + 6 > 4 ∧ (x < 1 ∨ x > b))) →
  (a = 1 ∧ b = 2) ∧
  (∀ c, 
    (c > 2 → {x | 2 < x ∧ x < c} = {x | x^2 - (2 + c)*x + 2*c < 0}) ∧
    (c < 2 → {x | c < x ∧ x < 2} = {x | x^2 - (2 + c)*x + 2*c < 0}) ∧
    (c = 2 → (∅ : Set ℝ) = {x | x^2 - (2 + c)*x + 2*c < 0})) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3955_395599


namespace NUMINAMATH_CALUDE_time_difference_is_56_minutes_l3955_395528

def minnie_uphill_distance : ℝ := 12
def minnie_flat_distance : ℝ := 18
def minnie_downhill_distance : ℝ := 22
def minnie_uphill_speed : ℝ := 4
def minnie_flat_speed : ℝ := 25
def minnie_downhill_speed : ℝ := 32

def penny_downhill_distance : ℝ := 22
def penny_flat_distance : ℝ := 18
def penny_uphill_distance : ℝ := 12
def penny_downhill_speed : ℝ := 15
def penny_flat_speed : ℝ := 35
def penny_uphill_speed : ℝ := 8

theorem time_difference_is_56_minutes :
  let minnie_time := minnie_uphill_distance / minnie_uphill_speed +
                     minnie_flat_distance / minnie_flat_speed +
                     minnie_downhill_distance / minnie_downhill_speed
  let penny_time := penny_downhill_distance / penny_downhill_speed +
                    penny_flat_distance / penny_flat_speed +
                    penny_uphill_distance / penny_uphill_speed
  (minnie_time - penny_time) * 60 = 56 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_is_56_minutes_l3955_395528


namespace NUMINAMATH_CALUDE_courtyard_paving_l3955_395551

/-- Calculates the number of bricks required to pave a courtyard -/
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width : ℝ) :
  courtyard_length = 35 ∧ 
  courtyard_width = 24 ∧ 
  brick_length = 0.15 ∧ 
  brick_width = 0.08 →
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 70000 := by
  sorry

#check courtyard_paving

end NUMINAMATH_CALUDE_courtyard_paving_l3955_395551


namespace NUMINAMATH_CALUDE_percentage_relation_l3955_395557

theorem percentage_relation (j k l m x : ℝ) 
  (h1 : 1.25 * j = 0.25 * k)
  (h2 : 1.5 * k = x / 100 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l3955_395557


namespace NUMINAMATH_CALUDE_different_subject_book_choices_l3955_395505

def chinese_books : ℕ := 8
def math_books : ℕ := 6
def english_books : ℕ := 5

theorem different_subject_book_choices :
  chinese_books * math_books + 
  chinese_books * english_books + 
  math_books * english_books = 118 := by
  sorry

end NUMINAMATH_CALUDE_different_subject_book_choices_l3955_395505


namespace NUMINAMATH_CALUDE_second_hose_spray_rate_l3955_395563

/-- Calculates the spray rate of the second hose needed to fill a pool --/
theorem second_hose_spray_rate 
  (pool_capacity : ℝ) 
  (first_hose_rate : ℝ) 
  (total_time : ℝ) 
  (second_hose_time : ℝ) 
  (h1 : pool_capacity = 390)
  (h2 : first_hose_rate = 50)
  (h3 : total_time = 5)
  (h4 : second_hose_time = 2)
  : ∃ (second_hose_rate : ℝ), 
    second_hose_rate * second_hose_time + first_hose_rate * total_time = pool_capacity ∧ 
    second_hose_rate = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_hose_spray_rate_l3955_395563


namespace NUMINAMATH_CALUDE_abc_value_l3955_395559

def is_valid_abc (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > b ∧ b > c ∧
  (10 * a + b) + (10 * b + a) = 55 ∧
  1300 < 222 * (a + b + c) ∧ 222 * (a + b + c) < 1400

theorem abc_value :
  ∀ a b c : ℕ, is_valid_abc a b c → a = 3 ∧ b = 2 ∧ c = 1 :=
sorry

end NUMINAMATH_CALUDE_abc_value_l3955_395559


namespace NUMINAMATH_CALUDE_school_students_count_l3955_395586

/-- The number of students who play football -/
def football : ℕ := 325

/-- The number of students who play cricket -/
def cricket : ℕ := 175

/-- The number of students who play neither football nor cricket -/
def neither : ℕ := 50

/-- The number of students who play both football and cricket -/
def both : ℕ := 140

/-- The total number of students in the school -/
def total_students : ℕ := football + cricket - both + neither

theorem school_students_count :
  total_students = 410 := by sorry

end NUMINAMATH_CALUDE_school_students_count_l3955_395586


namespace NUMINAMATH_CALUDE_exists_integers_satisfying_inequality_l3955_395544

theorem exists_integers_satisfying_inequality :
  ∃ (A B : ℤ), (0.999 : ℝ) < (A : ℝ) + (B : ℝ) * Real.sqrt 2 ∧ (A : ℝ) + (B : ℝ) * Real.sqrt 2 < 1 :=
by sorry

end NUMINAMATH_CALUDE_exists_integers_satisfying_inequality_l3955_395544


namespace NUMINAMATH_CALUDE_conjugate_complex_abs_l3955_395537

theorem conjugate_complex_abs (α β : ℂ) : 
  (∃ (x y : ℝ), α = x + y * I ∧ β = x - y * I) →  -- α and β are conjugates
  (∃ (r : ℝ), α / β^2 = r) →                     -- α/β² is real
  Complex.abs (α - β) = 4 * Real.sqrt 3 →        -- |α - β| = 4√3
  Complex.abs α = 4 :=                           -- |α| = 4
by sorry

end NUMINAMATH_CALUDE_conjugate_complex_abs_l3955_395537


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l3955_395541

theorem fenced_area_calculation (yard_length yard_width cutout_side : ℕ) 
  (h1 : yard_length = 20)
  (h2 : yard_width = 18)
  (h3 : cutout_side = 4) :
  yard_length * yard_width - cutout_side * cutout_side = 344 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l3955_395541


namespace NUMINAMATH_CALUDE_fine_on_fifth_day_l3955_395520

/-- Calculates the fine for a given day based on the previous day's fine -/
def nextDayFine (previousFine : ℚ) : ℚ :=
  min (previousFine + 0.3) (previousFine * 2)

/-- Calculates the fine for a given number of days overdue -/
def fineFordaysOverdue (days : ℕ) : ℚ :=
  match days with
  | 0 => 0
  | 1 => 0.05
  | n + 1 => nextDayFine (fineFordaysOverdue n)

theorem fine_on_fifth_day :
  fineFordaysOverdue 5 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_fine_on_fifth_day_l3955_395520


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l3955_395550

theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_milk : ℝ) 
  (new_ratio : ℝ) :
  total_volume = 20 →
  added_milk = 5 →
  new_ratio = 4 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_volume ∧
    (initial_milk + added_milk) / initial_water = new_ratio ∧
    initial_milk / initial_water = 3 := by
sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l3955_395550


namespace NUMINAMATH_CALUDE_cell_growth_problem_l3955_395579

/-- Calculates the number of cells after a given number of days, 
    given an initial population and growth rate every two days. -/
def cell_population (initial_cells : ℕ) (growth_rate : ℕ) (days : ℕ) : ℕ :=
  initial_cells * growth_rate ^ (days / 2)

/-- Theorem stating that given the specific conditions of the problem,
    the cell population after 10 days is 1215. -/
theorem cell_growth_problem : cell_population 5 3 10 = 1215 := by
  sorry


end NUMINAMATH_CALUDE_cell_growth_problem_l3955_395579


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l3955_395562

theorem dining_bill_calculation (number_of_people : ℕ) (individual_payment : ℚ) (tip_percentage : ℚ) 
  (h1 : number_of_people = 6)
  (h2 : individual_payment = 25.48)
  (h3 : tip_percentage = 0.10) :
  (number_of_people : ℚ) * individual_payment / (1 + tip_percentage) = 139.89 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l3955_395562


namespace NUMINAMATH_CALUDE_local_extremum_implies_a_equals_four_l3955_395526

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_implies_a_equals_four :
  ∀ a b : ℝ,
  (f a b 1 = 10) →  -- f(1) = 10
  (f' a b 1 = 0) →  -- f'(1) = 0 (condition for local extremum)
  (∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → f a b x ≤ f a b 1) →  -- local maximum condition
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_local_extremum_implies_a_equals_four_l3955_395526


namespace NUMINAMATH_CALUDE_computer_pricing_l3955_395565

/-- Proves that if a selling price of $2240 yields a 40% profit on cost, 
    then a selling price of $2560 yields a 60% profit on the same cost. -/
theorem computer_pricing (cost : ℝ) 
  (h1 : 2240 = cost + 0.4 * cost) 
  (h2 : 2560 = cost + 0.6 * cost) : 
  2240 = cost * 1.4 ∧ 2560 = cost * 1.6 := by
  sorry

#check computer_pricing

end NUMINAMATH_CALUDE_computer_pricing_l3955_395565


namespace NUMINAMATH_CALUDE_shortest_wire_length_l3955_395591

/-- The length of the shortest wire around two circular poles -/
theorem shortest_wire_length (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 18) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let straight_section := 2 * Real.sqrt ((r1 + r2)^2 - (r2 - r1)^2)
  let small_circle_arc := 2 * π * r1 * (1/3)
  let large_circle_arc := 2 * π * r2 * (2/3)
  straight_section + small_circle_arc + large_circle_arc = 12 * Real.sqrt 3 + 14 * π :=
by sorry

end NUMINAMATH_CALUDE_shortest_wire_length_l3955_395591


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l3955_395596

theorem gcd_special_numbers : Nat.gcd 333333333 555555555 = 111111111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l3955_395596


namespace NUMINAMATH_CALUDE_rectangle_circles_l3955_395583

theorem rectangle_circles (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circles_l3955_395583


namespace NUMINAMATH_CALUDE_bookstore_shipment_size_l3955_395564

theorem bookstore_shipment_size (displayed_percentage : ℚ) (stored_amount : ℕ) : 
  displayed_percentage = 1/4 →
  stored_amount = 225 →
  ∃ total : ℕ, total = 300 ∧ (1 - displayed_percentage) * total = stored_amount :=
by
  sorry

end NUMINAMATH_CALUDE_bookstore_shipment_size_l3955_395564


namespace NUMINAMATH_CALUDE_mean_variance_preserved_l3955_395597

def initial_set : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
def new_set : List Int := [-5, -5, -3, -2, -1, 0, 1, 1, 2, 3, 4, 5]

def mean (s : List Int) : ℚ :=
  (s.sum : ℚ) / s.length

def variance (s : List Int) : ℚ :=
  let m := mean s
  (s.map (fun x => ((x : ℚ) - m) ^ 2)).sum / s.length

theorem mean_variance_preserved :
  mean initial_set = mean new_set ∧
  variance initial_set = variance new_set := by
  sorry

#eval mean initial_set
#eval mean new_set
#eval variance initial_set
#eval variance new_set

end NUMINAMATH_CALUDE_mean_variance_preserved_l3955_395597


namespace NUMINAMATH_CALUDE_grants_room_count_l3955_395507

def danielles_rooms : ℕ := 6

def heidis_rooms (danielles_rooms : ℕ) : ℕ := 3 * danielles_rooms

def grants_rooms (heidis_rooms : ℕ) : ℚ := (1 : ℚ) / 9 * heidis_rooms

theorem grants_room_count :
  grants_rooms (heidis_rooms danielles_rooms) = 2 := by
  sorry

end NUMINAMATH_CALUDE_grants_room_count_l3955_395507


namespace NUMINAMATH_CALUDE_jimmy_cards_ratio_l3955_395510

def jimmy_cards_problem (initial_cards : ℕ) (cards_to_bob : ℕ) (cards_left : ℕ) : Prop :=
  let cards_to_mary := initial_cards - cards_left - cards_to_bob
  (cards_to_mary : ℚ) / cards_to_bob = 2 / 1

theorem jimmy_cards_ratio : jimmy_cards_problem 18 3 9 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_cards_ratio_l3955_395510


namespace NUMINAMATH_CALUDE_lidia_money_is_66_l3955_395578

/-- The amount of money Lidia has for buying apps -/
def lidia_money (app_cost : ℝ) (num_apps : ℕ) (remaining : ℝ) : ℝ :=
  app_cost * (num_apps : ℝ) + remaining

/-- Theorem stating that Lidia has $66 for buying apps -/
theorem lidia_money_is_66 :
  lidia_money 4 15 6 = 66 := by
  sorry

end NUMINAMATH_CALUDE_lidia_money_is_66_l3955_395578


namespace NUMINAMATH_CALUDE_power_multiplication_l3955_395511

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3955_395511


namespace NUMINAMATH_CALUDE_exist_four_cells_l3955_395577

/-- Represents a cell in the grid -/
structure Cell :=
  (x : Fin 17)
  (y : Fin 17)
  (value : Fin 70)

/-- The type of the grid -/
def Grid := Fin 17 → Fin 17 → Fin 70

/-- Predicate to check if all numbers from 1 to 70 appear exactly once in the grid -/
def valid_grid (g : Grid) : Prop :=
  ∀ n : Fin 70, ∃! (x y : Fin 17), g x y = n

/-- Distance between two cells -/
def distance (a b : Cell) : ℕ :=
  (a.x - b.x) ^ 2 + (a.y - b.y) ^ 2

/-- Sum of values in two cells -/
def sum_values (a b : Cell) : ℕ :=
  a.value.val + b.value.val

/-- Main theorem -/
theorem exist_four_cells (g : Grid) (h : valid_grid g) :
  ∃ (a b c d : Cell),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    distance a b = distance c d ∧
    distance a d = distance b c ∧
    sum_values a c = sum_values b d :=
  sorry

end NUMINAMATH_CALUDE_exist_four_cells_l3955_395577


namespace NUMINAMATH_CALUDE_system_solution_l3955_395554

theorem system_solution (x y : ℝ) : 
  x^3 + y^3 = 7 ∧ x*y*(x + y) = -2 ↔ (x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3955_395554


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l3955_395519

theorem problems_left_to_grade (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) :
  total_worksheets = 14 →
  problems_per_worksheet = 2 →
  graded_worksheets = 7 →
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 14 :=
by sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l3955_395519


namespace NUMINAMATH_CALUDE_prove_additional_cans_l3955_395534

/-- The number of additional cans Alyssa and Abigail need to collect. -/
def additional_cans_needed (total_needed alyssa_collected abigail_collected : ℕ) : ℕ :=
  total_needed - (alyssa_collected + abigail_collected)

/-- Theorem: Given the conditions, the additional cans needed is 27. -/
theorem prove_additional_cans : additional_cans_needed 100 30 43 = 27 := by
  sorry

end NUMINAMATH_CALUDE_prove_additional_cans_l3955_395534


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3955_395500

theorem quadratic_roots_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + a*x + b = 0 ↔ x = x₁ ∨ x = x₂) ∧
    (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 2*x₁ ∨ x = 2*x₂)) →
  a / c = 1 / 8 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_relation_l3955_395500


namespace NUMINAMATH_CALUDE_pants_price_proof_l3955_395542

-- Define the total cost
def total_cost : ℝ := 70.93

-- Define the price difference between belt and pants
def price_difference : ℝ := 2.93

-- Define the price of pants
def price_of_pants : ℝ := 34.00

-- Theorem statement
theorem pants_price_proof :
  ∃ (belt_price : ℝ),
    price_of_pants + belt_price = total_cost ∧
    price_of_pants = belt_price - price_difference :=
by sorry

end NUMINAMATH_CALUDE_pants_price_proof_l3955_395542


namespace NUMINAMATH_CALUDE_percent_problem_l3955_395560

theorem percent_problem : ∃ x : ℝ, (1 / 100) * x = 123.56 ∧ x = 12356 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l3955_395560


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l3955_395558

theorem baseball_card_value_decrease : 
  let initial_value : ℝ := 100
  let year1_decrease : ℝ := 0.60
  let year2_decrease : ℝ := 0.30
  let year3_decrease : ℝ := 0.20
  let year4_decrease : ℝ := 0.10
  
  let value_after_year1 : ℝ := initial_value * (1 - year1_decrease)
  let value_after_year2 : ℝ := value_after_year1 * (1 - year2_decrease)
  let value_after_year3 : ℝ := value_after_year2 * (1 - year3_decrease)
  let value_after_year4 : ℝ := value_after_year3 * (1 - year4_decrease)
  
  let total_decrease : ℝ := (initial_value - value_after_year4) / initial_value * 100

  total_decrease = 79.84 := by sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l3955_395558


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l3955_395576

/-- The motion equation of an object -/
def s (t : ℝ) : ℝ := 7 * t^2 - 13 * t + 8

/-- The instantaneous velocity (derivative of s with respect to t) -/
def v (t : ℝ) : ℝ := 14 * t - 13

/-- Theorem: If the instantaneous velocity at t₀ is 1, then t₀ = 1 -/
theorem instantaneous_velocity_at_one (t₀ : ℝ) : v t₀ = 1 → t₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l3955_395576


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3955_395532

theorem solution_satisfies_system :
  let f (x y : ℝ) := x + y + 2 - 4*x*y
  ∀ (x y z : ℝ), 
    (f x y = 0 ∧ f y z = 0 ∧ f z x = 0) →
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3955_395532


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l3955_395529

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Murtha's pebble collection problem -/
theorem murtha_pebble_collection : arithmetic_sum 1 2 12 = 144 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebble_collection_l3955_395529


namespace NUMINAMATH_CALUDE_conference_trip_distance_l3955_395553

/-- Conference Trip Problem -/
theorem conference_trip_distance :
  ∀ (d : ℝ) (t : ℝ),
    -- Initial speed
    let v₁ : ℝ := 40
    -- Speed increase
    let v₂ : ℝ := 20
    -- Time late if continued at initial speed
    let t_late : ℝ := 0.75
    -- Time early with speed increase
    let t_early : ℝ := 0.25
    -- Distance equation at initial speed
    d = v₁ * (t + t_late) →
    -- Distance equation with speed increase
    d - v₁ = (v₁ + v₂) * (t - 1 - t_early) →
    -- Conclusion: distance is 160 miles
    d = 160 := by
  sorry

end NUMINAMATH_CALUDE_conference_trip_distance_l3955_395553


namespace NUMINAMATH_CALUDE_assignments_for_twenty_points_l3955_395525

/-- Calculates the number of assignments required for a given number of points -/
def assignments_required (points : ℕ) : ℕ :=
  let segments := (points + 3) / 4
  (segments * (segments + 1) * 2) 

/-- The theorem stating that 60 assignments are required for 20 points -/
theorem assignments_for_twenty_points :
  assignments_required 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_assignments_for_twenty_points_l3955_395525


namespace NUMINAMATH_CALUDE_salary_increase_l3955_395536

/-- Given an original salary and a salary increase, 
    proves that the new salary is $90,000 if the percent increase is 38.46153846153846% --/
theorem salary_increase (S : ℝ) (increase : ℝ) : 
  increase = 25000 →
  (increase / S) * 100 = 38.46153846153846 →
  S + increase = 90000 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l3955_395536


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l3955_395535

theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = breadth + 24 →
  perimeter = 2 * (length + breadth) →
  26.50 * perimeter = 5300 →
  length = 62 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l3955_395535


namespace NUMINAMATH_CALUDE_f_decreasing_when_a_negative_l3955_395515

-- Define the function f(x) = ax^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3

-- Theorem statement
theorem f_decreasing_when_a_negative (a : ℝ) (h1 : a ≠ 0) (h2 : a < 0) :
  ∀ x y : ℝ, x < y → f a x > f a y :=
by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_when_a_negative_l3955_395515


namespace NUMINAMATH_CALUDE_problem_solution_l3955_395508

theorem problem_solution (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -1) : 
  (3 * a + a * b + 3 * b = 5) ∧ (a^2 + b^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3955_395508


namespace NUMINAMATH_CALUDE_friction_coefficient_inclined_plane_l3955_395589

/-- The coefficient of kinetic friction for a block sliding down an inclined plane,
    given that it reaches the bottom simultaneously with a hollow cylinder rolling without slipping -/
theorem friction_coefficient_inclined_plane (θ : Real) (g : Real) 
  (h1 : 0 < θ) (h2 : θ < π / 2) (h3 : g > 0) :
  let μ := (1 / 2) * Real.tan θ
  let a_cylinder := (1 / 2) * g * Real.sin θ
  let a_block := g * Real.sin θ - μ * g * Real.cos θ
  a_cylinder = a_block :=
by sorry

end NUMINAMATH_CALUDE_friction_coefficient_inclined_plane_l3955_395589


namespace NUMINAMATH_CALUDE_range_of_a_l3955_395561

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (Real.exp x + x - a + 1)

theorem range_of_a (a : ℝ) :
  (∃ x₀ y₀ : ℝ, y₀ = Real.cos x₀ ∧ f a (f a y₀) = y₀) →
  2 ≤ a ∧ a ≤ Real.exp 1 + 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3955_395561


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3955_395518

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = (1 - Real.sqrt 3) / 2) 
  (h2 : α ∈ Set.Ioo 0 Real.pi) : 
  Real.tan α = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3955_395518


namespace NUMINAMATH_CALUDE_interior_triangle_perimeter_is_715_l3955_395549

/-- Triangle ABC with parallel lines forming interior triangle XYZ -/
structure ParallelLineTriangle where
  /-- Side length of AB -/
  ab : ℝ
  /-- Side length of BC -/
  bc : ℝ
  /-- Side length of AC -/
  ac : ℝ
  /-- Length of intersection of ℓA with interior of triangle ABC -/
  ℓa_intersection : ℝ
  /-- Length of intersection of ℓB with interior of triangle ABC -/
  ℓb_intersection : ℝ
  /-- Length of intersection of ℓC with interior of triangle ABC -/
  ℓc_intersection : ℝ

/-- Perimeter of the interior triangle XYZ formed by lines ℓA, ℓB, and ℓC -/
def interior_triangle_perimeter (t : ParallelLineTriangle) : ℝ := sorry

/-- Theorem stating that the perimeter of the interior triangle is 715 for the given conditions -/
theorem interior_triangle_perimeter_is_715 (t : ParallelLineTriangle) 
  (h1 : t.ab = 120)
  (h2 : t.bc = 220)
  (h3 : t.ac = 180)
  (h4 : t.ℓa_intersection = 55)
  (h5 : t.ℓb_intersection = 45)
  (h6 : t.ℓc_intersection = 15) :
  interior_triangle_perimeter t = 715 := by sorry

end NUMINAMATH_CALUDE_interior_triangle_perimeter_is_715_l3955_395549


namespace NUMINAMATH_CALUDE_max_rooks_on_chessboard_l3955_395567

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a rook placement on a chessboard --/
structure RookPlacement :=
  (board : Chessboard)
  (num_rooks : ℕ)

/-- Predicate to check if a rook placement satisfies the condition --/
def satisfies_condition (placement : RookPlacement) : Prop :=
  ∀ (removed : ℕ), removed < placement.num_rooks →
    ∃ (square : ℕ × ℕ), 
      square.1 ≤ placement.board.size ∧ 
      square.2 ≤ placement.board.size ∧
      (∀ (rook : ℕ × ℕ), rook ≠ removed → 
        (rook.1 ≠ square.1 ∧ rook.2 ≠ square.2))

/-- The main theorem --/
theorem max_rooks_on_chessboard :
  ∃ (placement : RookPlacement),
    placement.board.size = 10 ∧
    placement.num_rooks = 81 ∧
    satisfies_condition placement ∧
    (∀ (other_placement : RookPlacement),
      other_placement.board.size = 10 →
      satisfies_condition other_placement →
      other_placement.num_rooks ≤ 81) :=
sorry

end NUMINAMATH_CALUDE_max_rooks_on_chessboard_l3955_395567


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3955_395580

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^12 - 1) % (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3955_395580
