import Mathlib

namespace NUMINAMATH_CALUDE_remainder_of_n_l2399_239918

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 11 = 9) (h2 : n^3 % 11 = 5) : n % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_l2399_239918


namespace NUMINAMATH_CALUDE_max_value_expression_l2399_239983

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (⨆ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + b*x + c^2))) = 
    3/2 * (a^2 + a*b + b^2/4 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2399_239983


namespace NUMINAMATH_CALUDE_bianca_winning_strategy_l2399_239907

/-- Represents a game state with two piles of marbles. -/
structure GameState where
  a : ℕ
  b : ℕ
  sum_eq_100 : a + b = 100

/-- Predicate to check if a move is valid. -/
def valid_move (s : GameState) (pile : ℕ) (remove : ℕ) : Prop :=
  (pile = s.a ∨ pile = s.b) ∧ 0 < remove ∧ remove ≤ pile / 2

/-- Predicate to check if a game state is a winning position for Bianca. -/
def is_winning_for_bianca (s : GameState) : Prop :=
  (s.a = 50 ∧ s.b = 50) ∨
  (s.a = 67 ∧ s.b = 33) ∨
  (s.a = 33 ∧ s.b = 67) ∨
  (s.a = 95 ∧ s.b = 5) ∨
  (s.a = 5 ∧ s.b = 95)

/-- Theorem stating that Bianca has a winning strategy if and only if
    the game state is one of the specified winning positions. -/
theorem bianca_winning_strategy (s : GameState) :
  (∀ (pile remove : ℕ), valid_move s pile remove →
    ∃ (new_s : GameState), ¬is_winning_for_bianca new_s) ↔
  is_winning_for_bianca s :=
sorry

end NUMINAMATH_CALUDE_bianca_winning_strategy_l2399_239907


namespace NUMINAMATH_CALUDE_volcano_theorem_l2399_239954

def volcano_problem (initial_volcanoes : ℕ) (first_explosion_rate : ℚ) 
  (mid_year_explosion_rate : ℚ) (end_year_explosion_rate : ℚ) (intact_volcanoes : ℕ) : Prop :=
  let remaining_after_first := initial_volcanoes - (initial_volcanoes * first_explosion_rate).floor
  let remaining_after_mid := remaining_after_first - (remaining_after_first * mid_year_explosion_rate).floor
  let final_exploded := (remaining_after_mid * end_year_explosion_rate).floor
  initial_volcanoes - intact_volcanoes = 
    (initial_volcanoes * first_explosion_rate).floor + 
    (remaining_after_first * mid_year_explosion_rate).floor + 
    final_exploded

theorem volcano_theorem : 
  volcano_problem 200 (20/100) (40/100) (50/100) 48 := by
  sorry

end NUMINAMATH_CALUDE_volcano_theorem_l2399_239954


namespace NUMINAMATH_CALUDE_product_remainder_seven_l2399_239948

theorem product_remainder_seven (a b : ℕ) (ha : a = 326) (hb : b = 57) :
  (a * b) % 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_seven_l2399_239948


namespace NUMINAMATH_CALUDE_function_derivative_l2399_239963

/-- Given a function f(x) = α² - cos(x), prove that its derivative f'(x) = sin(x) -/
theorem function_derivative (α : ℝ) : 
  let f : ℝ → ℝ := λ x => α^2 - Real.cos x
  deriv f = λ x => Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_l2399_239963


namespace NUMINAMATH_CALUDE_triple_base_exponent_l2399_239944

theorem triple_base_exponent (a b : ℤ) (x : ℚ) (h1 : b ≠ 0) :
  (3 * a) ^ (3 * b) = a ^ b * x ^ b → x = 27 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triple_base_exponent_l2399_239944


namespace NUMINAMATH_CALUDE_range_of_x_l2399_239928

theorem range_of_x (x y : ℝ) (h1 : 2*x - y = 4) (h2 : -2 < y) (h3 : y ≤ 3) :
  1 < x ∧ x ≤ 7/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2399_239928


namespace NUMINAMATH_CALUDE_smallest_integer_solution_inequality_l2399_239986

theorem smallest_integer_solution_inequality (x : ℤ) :
  (∀ y : ℤ, 3 * y ≥ y - 5 → y ≥ -2) ∧
  (3 * (-2) ≥ -2 - 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_inequality_l2399_239986


namespace NUMINAMATH_CALUDE_karthik_weight_average_l2399_239991

def karthik_weight_range (w : ℝ) : Prop :=
  55 < w ∧ w < 62 ∧ 50 < w ∧ w < 60 ∧ w < 58

theorem karthik_weight_average :
  ∃ (min max : ℝ),
    (∀ w, karthik_weight_range w → min ≤ w ∧ w ≤ max) ∧
    (∃ w₁ w₂, karthik_weight_range w₁ ∧ karthik_weight_range w₂ ∧ w₁ = min ∧ w₂ = max) ∧
    (min + max) / 2 = 56.5 :=
sorry

end NUMINAMATH_CALUDE_karthik_weight_average_l2399_239991


namespace NUMINAMATH_CALUDE_inscribed_circle_square_side_length_l2399_239902

theorem inscribed_circle_square_side_length 
  (circle_area : ℝ) 
  (h_area : circle_area = 3848.4510006474966) : 
  let square_side := 70
  let π := Real.pi
  circle_area = π * (square_side / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_side_length_l2399_239902


namespace NUMINAMATH_CALUDE_parabola_directrix_l2399_239904

/-- The equation of a parabola -/
def parabola_eq (x y : ℝ) : Prop := y = 4 * x^2 + 4 * x + 1

/-- The equation of the directrix -/
def directrix_eq (y : ℝ) : Prop := y = 11/16

/-- Theorem stating that the given directrix is correct for the parabola -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_eq x y → ∃ d : ℝ, directrix_eq d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
   ∃ f : ℝ × ℝ, (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.2 - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2399_239904


namespace NUMINAMATH_CALUDE_like_terms_exponent_l2399_239961

theorem like_terms_exponent (m n : ℕ) : 
  (∃ (x y : ℝ), -7 * x^(m+2) * y^2 = -3 * x^3 * y^n) → m^n = 1 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_l2399_239961


namespace NUMINAMATH_CALUDE_function_symmetry_theorem_l2399_239908

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define the concept of symmetry about y-axis
def symmetric_about_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_symmetry_theorem :
  symmetric_about_y_axis (fun x ↦ f (x - 1)) exp →
  f = fun x ↦ exp (-x - 1) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_theorem_l2399_239908


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2399_239996

theorem complex_magnitude_problem (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : a / (1 - i) = 1 - b * i) : 
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2399_239996


namespace NUMINAMATH_CALUDE_complex_modulus_l2399_239970

theorem complex_modulus (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) : 
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_l2399_239970


namespace NUMINAMATH_CALUDE_unique_integer_product_l2399_239959

/-- A function that returns true if the given number uses each digit from the given list exactly once -/
def uses_digits_once (n : ℕ) (digits : List ℕ) : Prop :=
  sorry

/-- A function that combines two natural numbers into a single number -/
def combine_numbers (a b : ℕ) : ℕ :=
  sorry

theorem unique_integer_product : ∃! n : ℕ, 
  uses_digits_once (combine_numbers (4 * n) (5 * n)) [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧ 
  n = 2469 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_product_l2399_239959


namespace NUMINAMATH_CALUDE_fifteenth_term_of_modified_arithmetic_sequence_l2399_239955

/-- Given an arithmetic sequence with first term 3, second term 15, and third term 27,
    prove that the 15th term is 339 when the common difference is doubled. -/
theorem fifteenth_term_of_modified_arithmetic_sequence :
  ∀ (a : ℕ → ℝ),
    a 1 = 3 →
    a 2 = 15 →
    a 3 = 27 →
    (∀ n : ℕ, a (n + 1) - a n = 2 * (a 2 - a 1)) →
    a 15 = 339 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_modified_arithmetic_sequence_l2399_239955


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2399_239937

/-- Expresses the sum of three repeating decimals as a rational number -/
theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + (3 : ℚ) / 99 + (4 : ℚ) / 9999 = 283 / 11111 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2399_239937


namespace NUMINAMATH_CALUDE_square_hole_reassembly_l2399_239940

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Represents a square with a square hole -/
structure SquareWithHole :=
  (outer_side : ℝ)
  (hole_side : ℝ)
  (hole_position : ℝ × ℝ)

/-- Function to divide a square with a hole into four quadrilaterals -/
def divide_square (s : SquareWithHole) : Fin 4 → Quadrilateral :=
  sorry

/-- Function to check if a set of quadrilaterals can form a square with a hole -/
def can_form_square_with_hole (quads : Fin 4 → Quadrilateral) : Prop :=
  sorry

/-- The main theorem to be proved -/
theorem square_hole_reassembly 
  (s : SquareWithHole) : 
  can_form_square_with_hole (divide_square s) :=
sorry

end NUMINAMATH_CALUDE_square_hole_reassembly_l2399_239940


namespace NUMINAMATH_CALUDE_bookcase_shelves_l2399_239906

theorem bookcase_shelves (initial_books : ℕ) (books_bought : ℕ) (books_per_shelf : ℕ) (books_left_over : ℕ) : 
  initial_books = 56 →
  books_bought = 26 →
  books_per_shelf = 20 →
  books_left_over = 2 →
  (initial_books + books_bought - books_left_over) / books_per_shelf = 4 := by
sorry

end NUMINAMATH_CALUDE_bookcase_shelves_l2399_239906


namespace NUMINAMATH_CALUDE_louise_picture_hanging_l2399_239929

/-- Given a total number of pictures, the number hung horizontally, and the number hung haphazardly,
    calculate the number of pictures hung vertically. -/
def verticalPictures (total horizontal haphazard : ℕ) : ℕ :=
  total - horizontal - haphazard

/-- Theorem stating that given 30 total pictures, with half hung horizontally and 5 haphazardly,
    the number of vertically hung pictures is 10. -/
theorem louise_picture_hanging :
  let total := 30
  let horizontal := total / 2
  let haphazard := 5
  verticalPictures total horizontal haphazard = 10 := by
  sorry

end NUMINAMATH_CALUDE_louise_picture_hanging_l2399_239929


namespace NUMINAMATH_CALUDE_f_negative_one_value_l2399_239914

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_negative_one_value :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is an odd function
  (∀ x : ℝ, x > 0 → f x = 2*x - 1) →  -- Definition of f for positive x
  f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_one_value_l2399_239914


namespace NUMINAMATH_CALUDE_combined_drying_time_l2399_239998

-- Define the driers' capacities and individual drying times
def drier1_capacity : ℚ := 1/2
def drier2_capacity : ℚ := 3/4
def drier3_capacity : ℚ := 1

def drier1_time : ℚ := 24
def drier2_time : ℚ := 2
def drier3_time : ℚ := 8

-- Define the combined drying rate
def combined_rate : ℚ := 
  drier1_capacity / drier1_time + 
  drier2_capacity / drier2_time + 
  drier3_capacity / drier3_time

-- Theorem statement
theorem combined_drying_time : 
  1 / combined_rate = 3/2 := by sorry

end NUMINAMATH_CALUDE_combined_drying_time_l2399_239998


namespace NUMINAMATH_CALUDE_simplified_robot_ratio_l2399_239951

/-- The number of animal robots Michael has -/
def michaels_robots : ℕ := 8

/-- The number of animal robots Tom has -/
def toms_robots : ℕ := 16

/-- The ratio of Tom's robots to Michael's robots -/
def robot_ratio : Rat := toms_robots / michaels_robots

theorem simplified_robot_ratio : robot_ratio = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_robot_ratio_l2399_239951


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2399_239926

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2399_239926


namespace NUMINAMATH_CALUDE_intersection_M_N_l2399_239975

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x, y = x + 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2399_239975


namespace NUMINAMATH_CALUDE_product_of_y_coordinates_l2399_239999

def point_P (y : ℝ) : ℝ × ℝ := (-3, y)

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem product_of_y_coordinates (k : ℝ) (h : k > 0) :
  ∃ y1 y2 : ℝ,
    distance_squared (point_P y1) (5, 2) = k^2 ∧
    distance_squared (point_P y2) (5, 2) = k^2 ∧
    y1 * y2 = 68 - k^2 :=
  sorry

end NUMINAMATH_CALUDE_product_of_y_coordinates_l2399_239999


namespace NUMINAMATH_CALUDE_problem_solution_l2399_239981

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 3*x + 3/x + 1/x^2 = 26)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2399_239981


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2399_239900

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 100 ∧ 
  white = 20 ∧ 
  green = 30 ∧ 
  red = 37 ∧ 
  purple = 3 ∧ 
  prob = 6/10 ∧ 
  (white + green : ℚ) / total + (total - white - green - red - purple : ℚ) / total = prob →
  total - white - green - red - purple = 10 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2399_239900


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2399_239976

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2399_239976


namespace NUMINAMATH_CALUDE_expression_evaluation_l2399_239910

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  3 * x^y + 4 * y^x - 2 * x * y = 47 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2399_239910


namespace NUMINAMATH_CALUDE_batsman_average_l2399_239973

/-- Represents a batsman's performance --/
structure Batsman where
  innings : ℕ
  runsInLastInning : ℕ
  averageIncrease : ℕ

/-- Calculates the average runs per inning after the last inning --/
def finalAverage (b : Batsman) : ℕ :=
  let previousAverage := b.runsInLastInning - b.averageIncrease
  previousAverage + b.averageIncrease

/-- Theorem: The batsman's average after 17 innings is 40 runs --/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 17)
  (h2 : b.runsInLastInning = 200)
  (h3 : b.averageIncrease = 10) : 
  finalAverage b = 40 := by
  sorry

#eval finalAverage { innings := 17, runsInLastInning := 200, averageIncrease := 10 }

end NUMINAMATH_CALUDE_batsman_average_l2399_239973


namespace NUMINAMATH_CALUDE_division_by_negative_l2399_239960

theorem division_by_negative : 15 / (-3 : ℤ) = -5 := by sorry

end NUMINAMATH_CALUDE_division_by_negative_l2399_239960


namespace NUMINAMATH_CALUDE_partnership_contribution_time_l2399_239925

/-- Proves that given the conditions of the partnership problem, A contributed for 8 months -/
theorem partnership_contribution_time (a_contribution b_contribution total_profit a_share : ℚ)
  (b_time : ℕ) :
  a_contribution = 5000 →
  b_contribution = 6000 →
  b_time = 5 →
  total_profit = 8400 →
  a_share = 4800 →
  ∃ (a_time : ℕ),
    a_time = 8 ∧
    a_share / total_profit = (a_contribution * a_time) / (a_contribution * a_time + b_contribution * b_time) :=
by sorry

end NUMINAMATH_CALUDE_partnership_contribution_time_l2399_239925


namespace NUMINAMATH_CALUDE_max_value_A_l2399_239994

/-- The function A(x, y) as defined in the problem -/
def A (x y : ℝ) : ℝ := x^4*y + x*y^4 + x^3*y + x*y^3 + x^2*y + x*y^2

/-- The theorem stating the maximum value of A(x, y) under the given constraint -/
theorem max_value_A :
  ∀ x y : ℝ, x + y = 1 → A x y ≤ 7/16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_A_l2399_239994


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_parallel_transitive_perpendicular_l2399_239930

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_parallel_implies_perpendicular
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : parallel_line_plane n α)
  (h3 : m ≠ n) :
  perpendicular_lines m n :=
sorry

-- Theorem 2
theorem parallel_transitive_perpendicular
  (m : Line) (α β γ : Plane)
  (h1 : parallel_plane α β)
  (h2 : parallel_plane β γ)
  (h3 : perpendicular m α)
  (h4 : α ≠ β) (h5 : β ≠ γ) (h6 : α ≠ γ) :
  perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_parallel_transitive_perpendicular_l2399_239930


namespace NUMINAMATH_CALUDE_sausages_problem_l2399_239962

def sausages_left (initial : ℕ) : ℕ :=
  let after_monday := initial - (2 * initial / 5)
  let after_tuesday := after_monday - (after_monday / 2)
  let after_wednesday := after_tuesday - (after_tuesday / 4)
  let after_thursday := after_wednesday - (after_wednesday / 3)
  after_thursday - (3 * after_thursday / 5)

theorem sausages_problem : sausages_left 1200 = 72 := by
  sorry

end NUMINAMATH_CALUDE_sausages_problem_l2399_239962


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l2399_239974

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (a * 10 + b : ℚ) / 99

/-- The fraction representation of 0.overline(63) -/
def frac63 : ℚ := RepeatingDecimal 6 3

/-- The fraction representation of 0.overline(21) -/
def frac21 : ℚ := RepeatingDecimal 2 1

/-- Proves that the division of 0.overline(63) by 0.overline(21) equals 3 -/
theorem repeating_decimal_division : frac63 / frac21 = 3 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l2399_239974


namespace NUMINAMATH_CALUDE_buses_passed_count_l2399_239927

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a bus schedule -/
structure BusSchedule where
  startTime : Time
  interval : Nat

/-- Calculates the number of buses passed during a journey -/
def busesPassed (departureTime : Time) (journeyDuration : Nat) (cityASchedule : BusSchedule) (cityBSchedule : BusSchedule) : Nat :=
  sorry

theorem buses_passed_count :
  let cityASchedule : BusSchedule := ⟨⟨6, 0, by sorry, by sorry⟩, 2⟩
  let cityBSchedule : BusSchedule := ⟨⟨6, 30, by sorry, by sorry⟩, 1⟩
  let departureTime : Time := ⟨14, 30, by sorry, by sorry⟩
  let journeyDuration : Nat := 8
  busesPassed departureTime journeyDuration cityASchedule cityBSchedule = 5 := by
  sorry

end NUMINAMATH_CALUDE_buses_passed_count_l2399_239927


namespace NUMINAMATH_CALUDE_cheryl_pesto_production_l2399_239946

/-- Prove that Cheryl can make 32 cups of pesto given the harvesting conditions --/
theorem cheryl_pesto_production (basil_per_pesto : ℕ) (basil_per_week : ℕ) (weeks : ℕ)
  (h1 : basil_per_pesto = 4)
  (h2 : basil_per_week = 16)
  (h3 : weeks = 8) :
  (basil_per_week * weeks) / basil_per_pesto = 32 := by
  sorry


end NUMINAMATH_CALUDE_cheryl_pesto_production_l2399_239946


namespace NUMINAMATH_CALUDE_exam_proctoring_arrangements_l2399_239934

def female_teachers : ℕ := 2
def male_teachers : ℕ := 5
def total_teachers : ℕ := female_teachers + male_teachers
def stationary_positions : ℕ := 2

theorem exam_proctoring_arrangements :
  (female_teachers * (total_teachers - 1).choose stationary_positions) = 42 := by
  sorry

end NUMINAMATH_CALUDE_exam_proctoring_arrangements_l2399_239934


namespace NUMINAMATH_CALUDE_rain_given_northeast_wind_l2399_239968

/-- Probability of northeast winds blowing -/
def P_A : ℝ := 0.7

/-- Probability of rain -/
def P_B : ℝ := 0.8

/-- Probability of both northeast winds blowing and rain -/
def P_AB : ℝ := 0.65

/-- Theorem: The conditional probability of rain given northeast winds is 13/14 -/
theorem rain_given_northeast_wind :
  P_AB / P_A = 13 / 14 := by sorry

end NUMINAMATH_CALUDE_rain_given_northeast_wind_l2399_239968


namespace NUMINAMATH_CALUDE_complex_calculation_theorem_logarithm_calculation_theorem_l2399_239952

theorem complex_calculation_theorem :
  (2 ^ (1/3) * 3 ^ (1/2)) ^ 6 + (2 * 2 ^ (1/2)) ^ (4/3) - 4 * (16/49) ^ (-1/2) - 2 ^ (1/4) * 8 ^ 0.25 - (-2005) ^ 0 = 100 :=
by sorry

theorem logarithm_calculation_theorem :
  ((1 - Real.log 3 / Real.log 6) ^ 2 + Real.log 2 / Real.log 6 * Real.log 18 / Real.log 6) / (Real.log 4 / Real.log 6) = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_theorem_logarithm_calculation_theorem_l2399_239952


namespace NUMINAMATH_CALUDE_shaded_region_angle_l2399_239941

/-- Given two concentric circles with radii 1 and 2, if the area of the shaded region
    between them is three times smaller than the area of the larger circle,
    then the angle subtending this shaded region at the center is 8π/9 radians. -/
theorem shaded_region_angle (r₁ r₂ : ℝ) (A_shaded A_large : ℝ) (θ : ℝ) :
  r₁ = 1 →
  r₂ = 2 →
  A_large = π * r₂^2 →
  A_shaded = (1/3) * A_large →
  A_shaded = (θ / (2 * π)) * (π * r₂^2 - π * r₁^2) →
  θ = (8 * π) / 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_angle_l2399_239941


namespace NUMINAMATH_CALUDE_vector_bc_coordinates_l2399_239913

/-- Given points A, B, and vector AC, prove that vector BC has specific coordinates -/
theorem vector_bc_coordinates (A B C : ℝ × ℝ) (h1 : A = (0, 1)) (h2 : B = (3, 2)) 
  (h3 : C.1 - A.1 = -4 ∧ C.2 - A.2 = -3) : 
  (C.1 - B.1, C.2 - B.2) = (-7, -4) := by
  sorry

#check vector_bc_coordinates

end NUMINAMATH_CALUDE_vector_bc_coordinates_l2399_239913


namespace NUMINAMATH_CALUDE_relative_error_approximation_l2399_239988

theorem relative_error_approximation (y : ℝ) (h : |y| < 1) :
  let f := fun x => 1 / (1 + x)
  let approx := fun x => 1 - x
  let relative_error := fun x => (f x - approx x) / f x
  relative_error y = y^2 := by
  sorry

end NUMINAMATH_CALUDE_relative_error_approximation_l2399_239988


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l2399_239938

theorem scientific_notation_proof : 
  284000000 = 2.84 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l2399_239938


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2399_239957

-- Define the function f(x) = ax^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3

-- Theorem statement
theorem cubic_function_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, x < y → a < 0 → f a x > f a y) ∧
  (∃ x : ℝ, x ≠ 1 ∧ f a x = 3 * a * x - 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2399_239957


namespace NUMINAMATH_CALUDE_arithmetic_sequence_18th_term_l2399_239979

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_18th_term (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_sum : a 2 + a 5 + a 8 = 33)
  (h_geom_mean : (a 5 + 1)^2 = (a 2 + 1) * (a 8 + 7)) :
  a 18 = 37 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_18th_term_l2399_239979


namespace NUMINAMATH_CALUDE_circle_center_on_line_l2399_239953

/-- Given a circle with equation x² + y² - 2ax + 4y - 6 = 0,
    if its center (h, k) satisfies h + 2k + 1 = 0, then a = 3 -/
theorem circle_center_on_line (a : ℝ) :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 - 2*a*x + 4*y - 6 = 0
  let center := fun (h k : ℝ) => ∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + (k+2)^2 + 10)
  let on_line := fun (h k : ℝ) => h + 2*k + 1 = 0
  (∃ h k, center h k ∧ on_line h k) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_on_line_l2399_239953


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2399_239969

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line
  (given_line : Line)
  (given_point : Point)
  (result_line : Line) :
  given_line = Line.mk 1 2 (-1) →
  given_point = Point.mk 1 2 →
  result_line = Line.mk 1 2 (-5) →
  pointOnLine given_point result_line ∧ parallel given_line result_line := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2399_239969


namespace NUMINAMATH_CALUDE_min_value_of_function_l2399_239915

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  x + 9 / (x - 2) ≥ 8 ∧ ∃ y > 2, y + 9 / (y - 2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2399_239915


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l2399_239945

theorem same_solution_implies_c_value (x c : ℚ) : 
  (3 * x + 9 = 0) ∧ (c * x^2 - 7 = 6) → c = 13/9 := by sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l2399_239945


namespace NUMINAMATH_CALUDE_total_pay_per_episode_l2399_239920

def tv_show_pay (main_characters minor_characters minor_pay major_pay_ratio : ℕ) : ℕ :=
  let minor_total := minor_characters * minor_pay
  let major_total := main_characters * (major_pay_ratio * minor_pay)
  minor_total + major_total

theorem total_pay_per_episode :
  tv_show_pay 5 4 15000 3 = 285000 :=
by
  sorry

end NUMINAMATH_CALUDE_total_pay_per_episode_l2399_239920


namespace NUMINAMATH_CALUDE_saltwater_concentration_l2399_239950

/-- The final concentration of saltwater in a cup after partial overflow and refilling -/
theorem saltwater_concentration (initial_concentration : ℝ) 
  (overflow_ratio : ℝ) (h1 : initial_concentration = 0.16) 
  (h2 : overflow_ratio = 0.1) : 
  initial_concentration * (1 - overflow_ratio) = 8/75 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_concentration_l2399_239950


namespace NUMINAMATH_CALUDE_sqrt_ab_eq_a_plus_b_iff_zero_l2399_239972

theorem sqrt_ab_eq_a_plus_b_iff_zero (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt (a * b) = a + b ↔ a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ab_eq_a_plus_b_iff_zero_l2399_239972


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l2399_239911

theorem sqrt_difference_equality : Real.sqrt (49 + 49) - Real.sqrt (36 + 25) = 7 * Real.sqrt 2 - Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l2399_239911


namespace NUMINAMATH_CALUDE_cone_volume_l2399_239964

theorem cone_volume (slant_height height : ℝ) (h1 : slant_height = 15) (h2 : height = 9) :
  let radius := Real.sqrt (slant_height^2 - height^2)
  (1 / 3 : ℝ) * π * radius^2 * height = 432 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l2399_239964


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2399_239967

theorem sum_of_four_numbers : 1.84 + 5.23 + 2.41 + 8.64 = 18.12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2399_239967


namespace NUMINAMATH_CALUDE_solution_set_for_a_2_a_value_for_even_function_l2399_239905

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solution_set_for_a_2 :
  {x : ℝ | f 2 x ≥ 2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 5/2} := by sorry

-- Part 2
theorem a_value_for_even_function :
  (∀ x, f a x = f a (-x)) → a = -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_2_a_value_for_even_function_l2399_239905


namespace NUMINAMATH_CALUDE_triangle_trig_max_value_l2399_239924

theorem triangle_trig_max_value (A B C : ℝ) : 
  A = π / 4 → 
  A + B + C = π → 
  0 < B → 
  B < π → 
  0 < C → 
  C < π → 
  ∃ (x : ℝ), x = 2 * Real.cos B + Real.sin (2 * C) ∧ x ≤ 3 / 2 ∧ 
  ∀ (y : ℝ), y = 2 * Real.cos B + Real.sin (2 * C) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_triangle_trig_max_value_l2399_239924


namespace NUMINAMATH_CALUDE_games_per_box_l2399_239936

theorem games_per_box (initial_games : ℕ) (sold_games : ℕ) (num_boxes : ℕ) 
  (h1 : initial_games = 35)
  (h2 : sold_games = 19)
  (h3 : num_boxes = 2)
  (h4 : initial_games > sold_games) :
  (initial_games - sold_games) / num_boxes = 8 := by
sorry

end NUMINAMATH_CALUDE_games_per_box_l2399_239936


namespace NUMINAMATH_CALUDE_count_arrangements_11250_l2399_239978

def digits : List Nat := [1, 1, 2, 5, 0]

def is_multiple_of_two (n : Nat) : Bool :=
  n % 2 = 0

def is_five_digit (n : Nat) : Bool :=
  n ≥ 10000 ∧ n < 100000

def count_valid_arrangements (ds : List Nat) : Nat :=
  sorry

theorem count_arrangements_11250 : 
  count_valid_arrangements digits = 24 := by sorry

end NUMINAMATH_CALUDE_count_arrangements_11250_l2399_239978


namespace NUMINAMATH_CALUDE_local_value_in_product_l2399_239923

/-- The face value of a digit is the digit itself. -/
def faceValue (digit : ℕ) : ℕ := digit

/-- The local value of a digit in a number is the digit multiplied by its place value. -/
def localValue (digit : ℕ) (placeValue : ℕ) : ℕ := digit * placeValue

/-- The product of two numbers. -/
def product (a b : ℕ) : ℕ := a * b

/-- The theorem stating that the local value of 6 in the product of the face value of 7
    and the local value of 8 in 7098060 is equal to 60. -/
theorem local_value_in_product :
  let number := 7098060
  let faceValue7 := faceValue 7
  let localValue8 := localValue 8 1000
  let prod := product faceValue7 localValue8
  localValue 6 10 = 60 :=
by sorry

end NUMINAMATH_CALUDE_local_value_in_product_l2399_239923


namespace NUMINAMATH_CALUDE_combined_grade4_percent_is_16_l2399_239995

/-- Represents the number of students in Pinegrove school -/
def pinegrove_students : ℕ := 120

/-- Represents the number of students in Maplewood school -/
def maplewood_students : ℕ := 180

/-- Represents the percentage of grade 4 students in Pinegrove school -/
def pinegrove_grade4_percent : ℚ := 10 / 100

/-- Represents the percentage of grade 4 students in Maplewood school -/
def maplewood_grade4_percent : ℚ := 20 / 100

/-- Represents the total number of students in both schools -/
def total_students : ℕ := pinegrove_students + maplewood_students

/-- Theorem stating that the percentage of grade 4 students in the combined schools is 16% -/
theorem combined_grade4_percent_is_16 : 
  (pinegrove_grade4_percent * pinegrove_students + maplewood_grade4_percent * maplewood_students) / total_students = 16 / 100 := by
  sorry

end NUMINAMATH_CALUDE_combined_grade4_percent_is_16_l2399_239995


namespace NUMINAMATH_CALUDE_triple_sum_power_divisibility_l2399_239993

theorem triple_sum_power_divisibility (a b c : ℤ) (h : a + b + c = 0) :
  ∃ k : ℤ, a^1999 + b^1999 + c^1999 = 6 * k :=
by sorry

end NUMINAMATH_CALUDE_triple_sum_power_divisibility_l2399_239993


namespace NUMINAMATH_CALUDE_workers_wage_increase_l2399_239966

theorem workers_wage_increase (original_wage new_wage : ℝ) : 
  (new_wage = original_wage * 1.5) → (new_wage = 51) → (original_wage = 34) := by
  sorry

end NUMINAMATH_CALUDE_workers_wage_increase_l2399_239966


namespace NUMINAMATH_CALUDE_triangle_angle_properties_l2399_239949

theorem triangle_angle_properties (α : Real) 
  (h1 : 0 < α ∧ α < π) -- α is an internal angle of a triangle
  (h2 : Real.sin α + Real.cos α = 1/5) :
  Real.tan α = -4/3 ∧ 
  1 / (Real.cos α ^ 2 - Real.sin α ^ 2) = -25/7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_properties_l2399_239949


namespace NUMINAMATH_CALUDE_rectangle_area_is_eight_l2399_239917

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*y^2 + 8*x - 16*y + 32 = 0

/-- The rectangle's height is twice the diameter of the circle -/
def rectangle_height_condition (height diameter : ℝ) : Prop :=
  height = 2 * diameter

/-- One pair of sides of the rectangle is parallel to the x-axis -/
def rectangle_orientation : Prop :=
  True  -- This condition is implicitly assumed and doesn't affect the calculation

/-- The area of the rectangle given its height and width -/
def rectangle_area (height width : ℝ) : ℝ :=
  height * width

/-- The main theorem stating that the area of the rectangle is 8 square units -/
theorem rectangle_area_is_eight :
  ∃ (x y height width diameter : ℝ),
    circle_equation x y ∧
    rectangle_height_condition height diameter ∧
    rectangle_orientation ∧
    rectangle_area height width = 8 :=
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_eight_l2399_239917


namespace NUMINAMATH_CALUDE_triangle_side_length_l2399_239990

-- Define the triangle PQR
structure Triangle (P Q R : ℝ) where
  angleSum : P + Q + R = Real.pi
  positive : 0 < P ∧ 0 < Q ∧ 0 < R

-- Define the side lengths
def sideLength (a b : ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_side_length 
  (P Q R : ℝ) 
  (tri : Triangle P Q R) 
  (h1 : Real.cos (2 * P - Q) + Real.sin (P + 2 * Q) = 1)
  (h2 : sideLength P Q = 5)
  (h3 : sideLength P Q + sideLength Q R + sideLength R P = 12) :
  sideLength Q R = 3.5 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2399_239990


namespace NUMINAMATH_CALUDE_a_negative_sufficient_not_necessary_l2399_239985

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem a_negative_sufficient_not_necessary :
  (∀ a : ℝ, a < 0 → ∃ x : ℝ, x < 0 ∧ f a x = 0) ∧
  (∃ a : ℝ, a ≥ 0 ∧ ∃ x : ℝ, x < 0 ∧ f a x = 0) := by
  sorry

end NUMINAMATH_CALUDE_a_negative_sufficient_not_necessary_l2399_239985


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l2399_239965

-- Define the days of the week
inductive Day :=
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

def add_days (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => next_day (add_days d m)

theorem tomorrow_is_saturday 
  (h : add_days (next_day (next_day Day.Wednesday)) 5 = Day.Monday) : 
  next_day Day.Friday = Day.Saturday :=
by
  sorry

#check tomorrow_is_saturday

end NUMINAMATH_CALUDE_tomorrow_is_saturday_l2399_239965


namespace NUMINAMATH_CALUDE_gcd_8512_13832_l2399_239931

theorem gcd_8512_13832 : Nat.gcd 8512 13832 = 1064 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8512_13832_l2399_239931


namespace NUMINAMATH_CALUDE_arc_MTN_range_l2399_239942

/-- Represents a circle rolling along the base of an isosceles triangle -/
structure RollingCircle where
  -- Radius of the circle (equal to the altitude of the triangle)
  radius : ℝ
  -- Base angle of the isosceles triangle
  base_angle : ℝ
  -- Position of the tangent point T along AB (0 ≤ t ≤ 1)
  t : ℝ
  -- Constraint: 0 ≤ t ≤ 1
  t_range : 0 ≤ t ∧ t ≤ 1

/-- Calculates the angle of arc MTN for a given position of the rolling circle -/
def arcMTN (circle : RollingCircle) : ℝ :=
  sorry

/-- Theorem stating that arc MTN varies from 0° to 80° -/
theorem arc_MTN_range (circle : RollingCircle) :
  0 ≤ arcMTN circle ∧ arcMTN circle ≤ 80 ∧
  (∃ c1 : RollingCircle, arcMTN c1 = 0) ∧
  (∃ c2 : RollingCircle, arcMTN c2 = 80) :=
sorry

end NUMINAMATH_CALUDE_arc_MTN_range_l2399_239942


namespace NUMINAMATH_CALUDE_gcd_1855_1120_l2399_239982

theorem gcd_1855_1120 : Nat.gcd 1855 1120 = 35 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1855_1120_l2399_239982


namespace NUMINAMATH_CALUDE_total_cost_new_puppy_l2399_239916

def adoption_fee : ℝ := 20
def dog_food : ℝ := 20
def treat_bag_price : ℝ := 2.5
def num_treat_bags : ℕ := 2
def toys : ℝ := 15
def crate : ℝ := 20
def bed : ℝ := 20
def collar_leash : ℝ := 15
def discount_rate : ℝ := 0.2

theorem total_cost_new_puppy :
  let supplies_cost := dog_food + treat_bag_price * num_treat_bags + toys + crate + bed + collar_leash
  let discounted_supplies_cost := supplies_cost * (1 - discount_rate)
  adoption_fee + discounted_supplies_cost = 96 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_new_puppy_l2399_239916


namespace NUMINAMATH_CALUDE_min_value_theorem_l2399_239947

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : 2 * m - n * (-2) - 2 = 0) :
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 2 * m' - n' * (-2) - 2 = 0 → 
    1 / m + 2 / n ≤ 1 / m' + 2 / n') ∧ 
  (1 / m + 2 / n = 3 + 2 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2399_239947


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2399_239939

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 7*x + 10 = 0

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  is_isosceles : base ≠ leg

-- Define the triangle with sides from the quadratic equation
def triangle_from_equation : IsoscelesTriangle :=
  { base := 2,
    leg := 5,
    is_isosceles := by norm_num }

-- State the theorem
theorem isosceles_triangle_perimeter :
  quadratic_equation triangle_from_equation.base ∧
  quadratic_equation triangle_from_equation.leg →
  triangle_from_equation.base + 2 * triangle_from_equation.leg = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2399_239939


namespace NUMINAMATH_CALUDE_fraction_independence_l2399_239932

theorem fraction_independence (a b c a₁ b₁ c₁ : ℝ) (h₁ : a₁ ≠ 0) :
  (∀ x, (a * x^2 + b * x + c) / (a₁ * x^2 + b₁ * x + c₁) = (a / a₁)) ↔ 
  (a / a₁ = b / b₁ ∧ b / b₁ = c / c₁) :=
sorry

end NUMINAMATH_CALUDE_fraction_independence_l2399_239932


namespace NUMINAMATH_CALUDE_greater_solution_quadratic_l2399_239943

theorem greater_solution_quadratic (x : ℝ) : 
  x^2 + 20*x - 96 = 0 → x ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_greater_solution_quadratic_l2399_239943


namespace NUMINAMATH_CALUDE_f_increasing_condition_f_max_min_on_interval_log_inequality_l2399_239997

noncomputable section

variables (a : ℝ) (x : ℝ) (n : ℕ)

def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x

theorem f_increasing_condition (h : a > 0) :
  (∀ x ≥ 1, Monotone (f a)) ↔ a ≥ 1 := by sorry

theorem f_max_min_on_interval (h : a = 1) :
  (∀ x ∈ Set.Icc (1/2) 2, f a x ≤ 1 - Real.log 2) ∧
  (∀ x ∈ Set.Icc (1/2) 2, f a x ≥ 0) ∧
  (∃ x ∈ Set.Icc (1/2) 2, f a x = 1 - Real.log 2) ∧
  (∃ x ∈ Set.Icc (1/2) 2, f a x = 0) := by sorry

theorem log_inequality (h : a = 1) (hn : n > 1) :
  Real.log (n / (n - 1 : ℝ)) > 1 / n := by sorry

end

end NUMINAMATH_CALUDE_f_increasing_condition_f_max_min_on_interval_log_inequality_l2399_239997


namespace NUMINAMATH_CALUDE_grid_paths_count_l2399_239992

/-- Represents a grid of roads between two locations -/
structure Grid where
  north_paths : Nat
  east_paths : Nat

/-- Calculates the total number of paths in a grid -/
def total_paths (g : Grid) : Nat :=
  g.north_paths * g.east_paths

/-- Theorem stating that the total number of paths in the given grid is 15 -/
theorem grid_paths_count : 
  ∀ g : Grid, g.north_paths = 3 → g.east_paths = 5 → total_paths g = 15 := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_count_l2399_239992


namespace NUMINAMATH_CALUDE_max_value_implies_a_l2399_239984

def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

theorem max_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, f a x ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 2, f a x = 3) → 
  a = 5 - Real.sqrt 10 ∨ a = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l2399_239984


namespace NUMINAMATH_CALUDE_simplify_expression_l2399_239922

theorem simplify_expression : 
  (625 : ℝ) ^ (1/4 : ℝ) * (256 : ℝ) ^ (1/2 : ℝ) = 80 :=
by
  have h1 : (625 : ℝ) = 5^4 := by norm_num
  have h2 : (256 : ℝ) = 2^8 := by norm_num
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2399_239922


namespace NUMINAMATH_CALUDE_platform_length_l2399_239987

/-- The length of a platform given a train's speed, length, and crossing time -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 →
  train_length = 230.0384 →
  crossing_time = 24 →
  (train_speed * 1000 / 3600) * crossing_time - train_length = 249.9616 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2399_239987


namespace NUMINAMATH_CALUDE_partnership_profit_share_l2399_239909

/-- 
Given a partnership where:
- A invests 3 times as much as B
- B invests two-thirds of what C invests
- The total profit is 6600
Prove that B's share of the profit is 1200
-/
theorem partnership_profit_share 
  (c_investment : ℝ) 
  (total_profit : ℝ) 
  (h1 : total_profit = 6600) 
  (h2 : c_investment > 0) : 
  let b_investment := (2/3) * c_investment
  let a_investment := 3 * b_investment
  let total_investment := a_investment + b_investment + c_investment
  b_investment / total_investment * total_profit = 1200 := by
sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l2399_239909


namespace NUMINAMATH_CALUDE_min_value_plus_argmin_l2399_239921

open Real

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * cos (2 * x) + 16) - sin x ^ 2

theorem min_value_plus_argmin (m n : ℝ) 
  (hm : ∀ x, f x ≥ m)
  (hn : f n = m)
  (hp : ∀ x, 0 < x → x < n → f x > m) : 
  m + n = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_plus_argmin_l2399_239921


namespace NUMINAMATH_CALUDE_y_divisibility_l2399_239989

def y : ℕ := 128 + 192 + 256 + 320 + 576 + 704 + 6464

theorem y_divisibility :
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 16 * k) ∧
  (∃ k : ℕ, y = 32 * k) ∧
  (∃ k : ℕ, y = 64 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l2399_239989


namespace NUMINAMATH_CALUDE_salt_water_evaporation_l2399_239956

/-- Given a salt water solution with initial weight of 200 grams and 5% salt concentration,
    if the salt concentration becomes 8% after evaporation,
    then 75 grams of water has evaporated. -/
theorem salt_water_evaporation (initial_weight : ℝ) (initial_concentration : ℝ) 
    (final_concentration : ℝ) (evaporated_water : ℝ) : 
  initial_weight = 200 →
  initial_concentration = 0.05 →
  final_concentration = 0.08 →
  initial_weight * initial_concentration = 
    (initial_weight - evaporated_water) * final_concentration →
  evaporated_water = 75 := by
  sorry

#check salt_water_evaporation

end NUMINAMATH_CALUDE_salt_water_evaporation_l2399_239956


namespace NUMINAMATH_CALUDE_ratio_antecedent_proof_l2399_239903

theorem ratio_antecedent_proof (ratio_antecedent ratio_consequent consequent : ℚ) : 
  ratio_antecedent = 4 →
  ratio_consequent = 6 →
  consequent = 75 →
  (ratio_antecedent / ratio_consequent) * consequent = 50 := by
sorry

end NUMINAMATH_CALUDE_ratio_antecedent_proof_l2399_239903


namespace NUMINAMATH_CALUDE_al_mass_percentage_in_mixture_l2399_239980

/-- The mass percentage of aluminum in a mixture of AlCl3, Al2(SO4)3, and Al(OH)3 --/
theorem al_mass_percentage_in_mixture (m_AlCl3 m_Al2SO4_3 m_AlOH3 : ℝ)
  (molar_mass_Al molar_mass_AlCl3 molar_mass_Al2SO4_3 molar_mass_AlOH3 : ℝ)
  (h1 : m_AlCl3 = 50)
  (h2 : m_Al2SO4_3 = 70)
  (h3 : m_AlOH3 = 40)
  (h4 : molar_mass_Al = 26.98)
  (h5 : molar_mass_AlCl3 = 133.33)
  (h6 : molar_mass_Al2SO4_3 = 342.17)
  (h7 : molar_mass_AlOH3 = 78.01) :
  let m_Al_AlCl3 := m_AlCl3 / molar_mass_AlCl3 * molar_mass_Al
  let m_Al_Al2SO4_3 := m_Al2SO4_3 / molar_mass_Al2SO4_3 * (2 * molar_mass_Al)
  let m_Al_AlOH3 := m_AlOH3 / molar_mass_AlOH3 * molar_mass_Al
  let total_m_Al := m_Al_AlCl3 + m_Al_Al2SO4_3 + m_Al_AlOH3
  let total_m_mixture := m_AlCl3 + m_Al2SO4_3 + m_AlOH3
  let mass_percentage := total_m_Al / total_m_mixture * 100
  ∃ ε > 0, |mass_percentage - 21.87| < ε :=
by sorry

end NUMINAMATH_CALUDE_al_mass_percentage_in_mixture_l2399_239980


namespace NUMINAMATH_CALUDE_fraction_value_at_x_equals_one_l2399_239977

theorem fraction_value_at_x_equals_one :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x^2 - 4)
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_x_equals_one_l2399_239977


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2399_239958

theorem trigonometric_identity :
  (Real.sin (160 * π / 180) + Real.sin (40 * π / 180)) *
  (Real.sin (140 * π / 180) + Real.sin (20 * π / 180)) +
  (Real.sin (50 * π / 180) - Real.sin (70 * π / 180)) *
  (Real.sin (130 * π / 180) - Real.sin (110 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2399_239958


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2399_239919

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (1/2) * a * x^2 - a * x + 2 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2399_239919


namespace NUMINAMATH_CALUDE_f_properties_l2399_239901

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 + 1/2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-1/2) (Real.sqrt 3 / 2) ↔
    ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/4) ∧ f x = y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2399_239901


namespace NUMINAMATH_CALUDE_number_of_ways_is_132_l2399_239971

/-- Represents a girl --/
inductive Girl
| Amy
| Beth
| Jo

/-- Represents a song --/
inductive Song
| One
| Two
| Three
| Four

/-- Represents whether a girl likes a song --/
def Likes : Girl → Song → Prop := sorry

/-- No song is liked by all three girls --/
def NoSongLikedByAll : Prop :=
  ∀ s : Song, ¬(Likes Girl.Amy s ∧ Likes Girl.Beth s ∧ Likes Girl.Jo s)

/-- For each pair of girls, there is at least one song liked by those two but disliked by the third --/
def PairwiseLikedSong : Prop :=
  (∃ s : Song, Likes Girl.Amy s ∧ Likes Girl.Beth s ∧ ¬Likes Girl.Jo s) ∧
  (∃ s : Song, Likes Girl.Beth s ∧ Likes Girl.Jo s ∧ ¬Likes Girl.Amy s) ∧
  (∃ s : Song, Likes Girl.Jo s ∧ Likes Girl.Amy s ∧ ¬Likes Girl.Beth s)

/-- The number of ways the girls can like the songs satisfying the conditions --/
def NumberOfWays : ℕ := sorry

/-- The theorem to be proved --/
theorem number_of_ways_is_132 
  (h1 : NoSongLikedByAll) 
  (h2 : PairwiseLikedSong) : 
  NumberOfWays = 132 := by sorry

end NUMINAMATH_CALUDE_number_of_ways_is_132_l2399_239971


namespace NUMINAMATH_CALUDE_horner_eval_23_l2399_239912

def horner_polynomial (a b c d x : ℤ) : ℤ := ((a * x + b) * x + c) * x + d

theorem horner_eval_23 :
  let f : ℤ → ℤ := λ x => 7 * x^3 + 3 * x^2 - 5 * x + 11
  let horner : ℤ → ℤ := horner_polynomial 7 3 (-5) 11
  (∀ step : ℤ, step ≠ 85169 → (step = 7 ∨ step = 164 ∨ step = 3762 ∨ step = 86537)) ∧
  f 23 = horner 23 ∧
  f 23 = 86537 := by
sorry

end NUMINAMATH_CALUDE_horner_eval_23_l2399_239912


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2399_239935

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ (7^n ≡ n^5 [ZMOD 3]) ∧ 
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬(7^m ≡ m^5 [ZMOD 3]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2399_239935


namespace NUMINAMATH_CALUDE_bricks_required_l2399_239933

/-- The number of bricks required to pave a rectangular courtyard -/
theorem bricks_required (courtyard_length courtyard_width brick_length brick_width : ℝ) :
  courtyard_length = 28 →
  courtyard_width = 13 →
  brick_length = 0.22 →
  brick_width = 0.12 →
  ↑(⌈(courtyard_length * courtyard_width * 10000) / (brick_length * brick_width)⌉) = 13788 := by
  sorry

end NUMINAMATH_CALUDE_bricks_required_l2399_239933
