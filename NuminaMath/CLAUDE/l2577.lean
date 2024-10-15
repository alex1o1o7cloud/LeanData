import Mathlib

namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2577_257773

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 64 * π) :
  2 * π * r^2 + π * r^2 = 192 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2577_257773


namespace NUMINAMATH_CALUDE_distinct_sums_count_l2577_257733

/-- Represents the number of coins of each denomination -/
structure CoinSet :=
  (one_yuan : ℕ)
  (half_yuan : ℕ)

/-- Represents a selection of coins -/
structure CoinSelection :=
  (one_yuan : ℕ)
  (half_yuan : ℕ)

/-- Calculates the sum of face values for a given coin selection -/
def sumFaceValues (selection : CoinSelection) : ℚ :=
  selection.one_yuan + selection.half_yuan / 2

/-- Generates all possible coin selections given a coin set and total number of coins to select -/
def possibleSelections (coins : CoinSet) (total : ℕ) : List CoinSelection :=
  sorry

/-- Calculates the number of distinct sums from all possible selections -/
def distinctSums (coins : CoinSet) (total : ℕ) : ℕ :=
  (possibleSelections coins total).map sumFaceValues |> List.eraseDups |> List.length

/-- The main theorem stating that there are exactly 7 distinct sums when selecting 6 coins from 5 one-yuan and 6 half-yuan coins -/
theorem distinct_sums_count :
  distinctSums (CoinSet.mk 5 6) 6 = 7 := by sorry

end NUMINAMATH_CALUDE_distinct_sums_count_l2577_257733


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l2577_257781

/-- Given two solutions x and y, where:
  - x has A% of chemical a and 90% of chemical b
  - y has 20% of chemical a and 80% of chemical b
  - A mixture of x and y is 12% chemical a
  - The mixture is 80% solution x and 20% solution y
  Prove that A = 10 -/
theorem chemical_mixture_problem (A : ℝ) : 
  A + 90 = 100 →
  0.8 * A + 0.2 * 20 = 12 →
  A = 10 := by sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l2577_257781


namespace NUMINAMATH_CALUDE_square_sum_product_equals_k_squared_l2577_257799

theorem square_sum_product_equals_k_squared (k : ℕ) : 
  2012^2 + 2010 * 2011 * 2013 * 2014 = k^2 ∧ k > 0 → k = 4048142 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_equals_k_squared_l2577_257799


namespace NUMINAMATH_CALUDE_other_solution_quadratic_equation_l2577_257777

theorem other_solution_quadratic_equation :
  let f (x : ℚ) := 42 * x^2 + 2 * x + 31 - (73 * x + 4)
  (f (3/7) = 0) → (f (3/2) = 0) := by sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_equation_l2577_257777


namespace NUMINAMATH_CALUDE_remaining_macaroons_formula_l2577_257709

/-- The number of remaining macaroons after Fran eats some -/
def remaining_macaroons (k : ℚ) : ℚ :=
  let red := 50
  let green := 40
  let blue := 30
  let yellow := 20
  let orange := 10
  let total_baked := red + green + blue + yellow + orange
  let eaten_green := k
  let eaten_red := 2 * k
  let eaten_blue := 3 * k
  let eaten_yellow := (1 / 2) * k * yellow
  let eaten_orange := (1 / 5) * k
  let total_eaten := eaten_green + eaten_red + eaten_blue + eaten_yellow + eaten_orange
  total_baked - total_eaten

theorem remaining_macaroons_formula (k : ℚ) :
  remaining_macaroons k = 150 - (81 * k / 5) := by
  sorry

end NUMINAMATH_CALUDE_remaining_macaroons_formula_l2577_257709


namespace NUMINAMATH_CALUDE_plywood_cutting_result_l2577_257770

/-- Represents the cutting of a square plywood into smaller squares. -/
structure PlywoodCutting where
  side : ℝ
  small_square_side : ℝ
  large_square_side : ℝ
  total_cut_length : ℝ

/-- Calculates the total number of squares obtained from cutting the plywood. -/
def total_squares (cut : PlywoodCutting) : ℕ :=
  sorry

/-- Theorem stating that for the given plywood cutting specifications, 
    the total number of squares obtained is 16. -/
theorem plywood_cutting_result : 
  let cut := PlywoodCutting.mk 50 10 20 280
  total_squares cut = 16 := by
  sorry

end NUMINAMATH_CALUDE_plywood_cutting_result_l2577_257770


namespace NUMINAMATH_CALUDE_seminar_handshakes_l2577_257750

/-- The number of people attending the seminar -/
def n : ℕ := 12

/-- The number of pairs of people who don't shake hands -/
def excluded_pairs : ℕ := 1

/-- The total number of handshakes in the seminar -/
def total_handshakes : ℕ := n.choose 2 - excluded_pairs

/-- Theorem stating the total number of handshakes in the seminar -/
theorem seminar_handshakes : total_handshakes = 65 := by
  sorry

end NUMINAMATH_CALUDE_seminar_handshakes_l2577_257750


namespace NUMINAMATH_CALUDE_student_rabbit_difference_l2577_257708

/-- Given 4 classrooms, each with 18 students and 2 rabbits, prove that the difference
    between the total number of students and rabbits is 64. -/
theorem student_rabbit_difference (num_classrooms : ℕ) (students_per_class : ℕ) (rabbits_per_class : ℕ)
    (h1 : num_classrooms = 4)
    (h2 : students_per_class = 18)
    (h3 : rabbits_per_class = 2) :
    num_classrooms * students_per_class - num_classrooms * rabbits_per_class = 64 := by
  sorry

end NUMINAMATH_CALUDE_student_rabbit_difference_l2577_257708


namespace NUMINAMATH_CALUDE_complex_fraction_l2577_257761

theorem complex_fraction (z : ℂ) (h : z = 1 - 2*I) :
  (z + 2) / (z - 1) = 1 + (3/2)*I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_l2577_257761


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2577_257774

/-- Calculates the cost price per meter of cloth given total meters, selling price, and profit per meter -/
def cost_price_per_meter (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_meters) / total_meters

/-- Proves that the cost price of one meter of cloth is 5 Rs. given the problem conditions -/
theorem cloth_cost_price :
  cost_price_per_meter 66 660 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2577_257774


namespace NUMINAMATH_CALUDE_figure_to_square_possible_l2577_257791

/-- A figure composed of unit squares -/
structure UnitSquareFigure where
  area : ℕ

/-- A part of a figure after cutting -/
structure FigurePart where
  area : ℕ

/-- Represents a square -/
structure Square where
  side_length : ℕ

/-- Theorem stating that a UnitSquareFigure can be cut into three parts to form a square -/
theorem figure_to_square_possible (fig : UnitSquareFigure) 
  (h : ∃ n : ℕ, n * n = fig.area) : 
  ∃ (part1 part2 part3 : FigurePart) (sq : Square),
    part1.area + part2.area + part3.area = fig.area ∧
    sq.side_length * sq.side_length = fig.area :=
sorry

end NUMINAMATH_CALUDE_figure_to_square_possible_l2577_257791


namespace NUMINAMATH_CALUDE_buffet_combinations_l2577_257740

def num_meat_options : ℕ := 4
def num_vegetable_options : ℕ := 5
def num_vegetables_to_choose : ℕ := 3
def num_dessert_options : ℕ := 4
def num_desserts_to_choose : ℕ := 2

def choose (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem buffet_combinations :
  (num_meat_options) *
  (choose num_vegetable_options num_vegetables_to_choose) *
  (choose num_dessert_options num_desserts_to_choose) = 240 := by
  sorry

end NUMINAMATH_CALUDE_buffet_combinations_l2577_257740


namespace NUMINAMATH_CALUDE_bat_lifespan_solution_l2577_257741

def bat_lifespan_problem (bat_lifespan : ℕ) : Prop :=
  let hamster_lifespan := bat_lifespan - 6
  let frog_lifespan := 4 * hamster_lifespan
  bat_lifespan + hamster_lifespan + frog_lifespan = 30

theorem bat_lifespan_solution :
  ∃ (bat_lifespan : ℕ), bat_lifespan_problem bat_lifespan ∧ bat_lifespan = 10 := by
  sorry

end NUMINAMATH_CALUDE_bat_lifespan_solution_l2577_257741


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2577_257792

-- Define the plane and lines
variable (α : Set (Real × Real × Real))
variable (m n : Set (Real × Real × Real))

-- Define the perpendicular and parallel relations
def perpendicular (l : Set (Real × Real × Real)) (p : Set (Real × Real × Real)) : Prop := sorry
def parallel (l1 l2 : Set (Real × Real × Real)) : Prop := sorry

-- State the theorem
theorem perpendicular_lines_parallel (h1 : perpendicular m α) (h2 : perpendicular n α) :
  parallel m n := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2577_257792


namespace NUMINAMATH_CALUDE_bugs_meet_on_bc_l2577_257745

/-- Triangle with side lengths -/
structure Triangle where
  ab : ℝ
  bc : ℝ
  ac : ℝ

/-- Bug with starting position and speed -/
structure Bug where
  start : ℕ  -- 0 for A, 1 for B, 2 for C
  speed : ℝ
  clockwise : Bool

/-- The point where bugs meet -/
def MeetingPoint (t : Triangle) (bugA bugC : Bug) : ℝ := sorry

theorem bugs_meet_on_bc (t : Triangle) (bugA bugC : Bug) :
  t.ab = 5 ∧ t.bc = 6 ∧ t.ac = 7 ∧
  bugA.start = 0 ∧ bugA.speed = 1 ∧ bugA.clockwise = true ∧
  bugC.start = 2 ∧ bugC.speed = 2 ∧ bugC.clockwise = false →
  MeetingPoint t bugA bugC = 1 := by sorry

end NUMINAMATH_CALUDE_bugs_meet_on_bc_l2577_257745


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2577_257721

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁ > x₂) ∧ 
  (|2 * x₁ - 3| = 14) ∧ 
  (|2 * x₂ - 3| = 14) ∧ 
  (x₁ - x₂ = 14) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2577_257721


namespace NUMINAMATH_CALUDE_chessboard_determinability_l2577_257713

/-- Represents a square on the chessboard -/
structure Square where
  row : Nat
  col : Nat

/-- Represents the chessboard and the game state -/
structure Chessboard (n : Nat) where
  selected : Set Square
  adjacent_counts : Square → Nat

/-- Defines when a number is "beautiful" (remainder 1 when divided by 3) -/
def is_beautiful (k : Nat) : Bool :=
  k % 3 = 1

/-- Defines when a square is beautiful -/
def beautiful_square (s : Square) : Bool :=
  is_beautiful s.row ∧ is_beautiful s.col

/-- Defines when Bianka can uniquely determine Aranka's selection -/
def can_determine (n : Nat) (board : Chessboard n) : Prop :=
  ∀ (alt_board : Chessboard n),
    (∀ (s : Square), board.adjacent_counts s = alt_board.adjacent_counts s) →
    board.selected = alt_board.selected

/-- The main theorem to be proved -/
theorem chessboard_determinability (n : Nat) :
  (∃ (k : Nat), n = 3 * k + 1) → (∀ (board : Chessboard n), can_determine n board) ∧
  (∃ (k : Nat), n = 3 * k + 2) → (∃ (board : Chessboard n), ¬can_determine n board) :=
sorry

end NUMINAMATH_CALUDE_chessboard_determinability_l2577_257713


namespace NUMINAMATH_CALUDE_orange_harvest_theorem_l2577_257783

-- Define the number of days for the harvest
def harvest_days : ℕ := 4

-- Define the total number of sacks harvested
def total_sacks : ℕ := 56

-- Define the function to calculate sacks per day
def sacks_per_day (total : ℕ) (days : ℕ) : ℕ := total / days

-- Theorem statement
theorem orange_harvest_theorem : 
  sacks_per_day total_sacks harvest_days = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_theorem_l2577_257783


namespace NUMINAMATH_CALUDE_parallelogram_area_and_magnitude_l2577_257703

/-- Given a complex number z with positive real part, if the parallelogram formed by 
    0, z, z², and z + z² has area 20/29, then the smallest possible value of |z² + z| 
    is (r² + r), where r³|sin θ| = 20/29 and z = r(cos θ + i sin θ). -/
theorem parallelogram_area_and_magnitude (z : ℂ) (r θ : ℝ) (h1 : z.re > 0) 
  (h2 : z = r * Complex.exp (θ * Complex.I)) 
  (h3 : r > 0) 
  (h4 : r^3 * |Real.sin θ| = 20/29) 
  (h5 : Complex.abs (z * z - z) = 20/29) : 
  ∃ (d : ℝ), d = r^2 + r ∧ 
  ∀ (w : ℂ), Complex.abs (w^2 + w) ≥ d := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_and_magnitude_l2577_257703


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_3007_l2577_257743

-- Define a function to get the last two digits of a number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define the sequence of last two digits of 13^n
def lastTwoDigitsOf13Pow (n : ℕ) : ℕ :=
  match n % 10 with
  | 0 => 49
  | 1 => 37
  | 2 => 81
  | 3 => 53
  | 4 => 89
  | 5 => 57
  | 6 => 41
  | 7 => 17
  | 8 => 21
  | 9 => 73
  | _ => 0  -- This case should never occur

-- Theorem statement
theorem tens_digit_of_13_pow_3007 :
  (lastTwoDigitsOf13Pow 3007) / 10 = 1 := by
  sorry


end NUMINAMATH_CALUDE_tens_digit_of_13_pow_3007_l2577_257743


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l2577_257748

/-- A triangle with sides a, b, and c is either isosceles or right-angled if (a-b)(a^2+b^2-c^2) = 0 --/
theorem triangle_isosceles_or_right_angled 
  {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_equation : (a - b) * (a^2 + b^2 - c^2) = 0) : 
  (a = b ∨ a = c ∨ b = c) ∨ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l2577_257748


namespace NUMINAMATH_CALUDE_abs_neg_three_equals_three_l2577_257785

theorem abs_neg_three_equals_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_equals_three_l2577_257785


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2577_257744

theorem least_positive_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 5 = 4 ∧ 
  n % 7 = 6 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 5 = 4 ∧ m % 7 = 6 → n ≤ m :=
by
  use 69
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2577_257744


namespace NUMINAMATH_CALUDE_product_equality_l2577_257715

theorem product_equality (h : 213 * 16 = 3408) : 0.16 * 2.13 = 0.3408 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2577_257715


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2577_257763

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2577_257763


namespace NUMINAMATH_CALUDE_prime_factors_of_n_l2577_257729

theorem prime_factors_of_n (n : ℕ) (h1 : n > 0) (h2 : n < 200) (h3 : ∃ k : ℕ, 14 * n = 60 * k) :
  ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ∣ n ∧ q ∣ n ∧ r ∣ n :=
sorry

end NUMINAMATH_CALUDE_prime_factors_of_n_l2577_257729


namespace NUMINAMATH_CALUDE_f_n_has_real_root_l2577_257796

def f (x : ℝ) : ℝ := x^2 + 2007*x + 1

def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => f (f_n n x)

theorem f_n_has_real_root (n : ℕ+) : ∃ x : ℝ, f_n n x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_n_has_real_root_l2577_257796


namespace NUMINAMATH_CALUDE_art_arrangement_probability_l2577_257793

/-- The probability of arranging art pieces with specific conditions -/
theorem art_arrangement_probability (total_pieces : ℕ) (dali_paintings : ℕ) : 
  total_pieces = 12 →
  dali_paintings = 4 →
  (7 : ℚ) / 1485 = (
    (total_pieces - dali_paintings)  -- non-Dali pieces for first position
    * (total_pieces - dali_paintings)  -- positions for Dali group after first piece
    * (Nat.factorial (total_pieces - dali_paintings + 1))  -- arrangements of remaining pieces
  ) / (Nat.factorial total_pieces) := by
  sorry

#check art_arrangement_probability

end NUMINAMATH_CALUDE_art_arrangement_probability_l2577_257793


namespace NUMINAMATH_CALUDE_coefficient_x_squared_eq_five_l2577_257760

/-- The coefficient of x^2 in the expansion of (1/x^2 + x)^5 -/
def coefficient_x_squared : ℕ :=
  (Nat.choose 5 1)

theorem coefficient_x_squared_eq_five : coefficient_x_squared = 5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_eq_five_l2577_257760


namespace NUMINAMATH_CALUDE_jeans_final_price_l2577_257751

/-- Calculates the final price of jeans after summer and Wednesday discounts --/
theorem jeans_final_price (original_price : ℝ) (summer_discount_percent : ℝ) (wednesday_discount : ℝ) :
  original_price = 49 →
  summer_discount_percent = 50 →
  wednesday_discount = 10 →
  original_price * (1 - summer_discount_percent / 100) - wednesday_discount = 14.5 := by
  sorry

#check jeans_final_price

end NUMINAMATH_CALUDE_jeans_final_price_l2577_257751


namespace NUMINAMATH_CALUDE_hundred_hours_before_seven_am_l2577_257767

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the time a given number of hours before a specified time -/
def timeBefore (t : TimeOfDay) (h : Nat) : TimeOfDay :=
  sorry

/-- Theorem: 100 hours before 7:00 a.m. is 3:00 a.m. -/
theorem hundred_hours_before_seven_am :
  let start_time : TimeOfDay := ⟨7, 0, by sorry⟩
  let end_time : TimeOfDay := ⟨3, 0, by sorry⟩
  timeBefore start_time 100 = end_time := by
  sorry

end NUMINAMATH_CALUDE_hundred_hours_before_seven_am_l2577_257767


namespace NUMINAMATH_CALUDE_remainder_problem_l2577_257786

theorem remainder_problem (n : ℤ) : n % 8 = 3 → (4 * n - 9) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2577_257786


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_subset_implies_m_range_l2577_257755

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 2)*(x - m - 2) ≤ 0}

-- Theorem 1
theorem intersection_implies_m_value :
  ∀ m : ℝ, A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3} → m = 2 :=
sorry

-- Theorem 2
theorem subset_implies_m_range :
  ∀ m : ℝ, A ⊆ (Set.univ : Set ℝ) \ B m → m < -3 ∨ m > 5 :=
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_subset_implies_m_range_l2577_257755


namespace NUMINAMATH_CALUDE_tennis_ball_price_is_6_l2577_257702

/-- The price of a tennis ball in yuan -/
def tennis_ball_price : ℝ := 6

/-- The price of a tennis racket in yuan -/
def tennis_racket_price : ℝ := tennis_ball_price + 83

/-- The total cost of 2 tennis rackets and 7 tennis balls in yuan -/
def total_cost : ℝ := 220

theorem tennis_ball_price_is_6 :
  (2 * tennis_racket_price + 7 * tennis_ball_price = total_cost) ∧
  (tennis_racket_price = tennis_ball_price + 83) →
  tennis_ball_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_tennis_ball_price_is_6_l2577_257702


namespace NUMINAMATH_CALUDE_no_solution_inequality_l2577_257762

theorem no_solution_inequality : ¬∃ (x : ℝ), 2 - 3*x + 2*x^2 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_no_solution_inequality_l2577_257762


namespace NUMINAMATH_CALUDE_multiplication_problem_l2577_257766

theorem multiplication_problem : ∃ x : ℕ, 582964 * x = 58293485180 ∧ x = 100000 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l2577_257766


namespace NUMINAMATH_CALUDE_retail_price_increase_l2577_257768

theorem retail_price_increase (wholesale_cost employee_paid : ℝ) (employee_discount : ℝ) :
  wholesale_cost = 200 →
  employee_discount = 0.15 →
  employee_paid = 204 →
  ∃ (retail_price_increase : ℝ),
    retail_price_increase = 0.20 ∧
    employee_paid = wholesale_cost * (1 + retail_price_increase) * (1 - employee_discount) :=
by
  sorry

end NUMINAMATH_CALUDE_retail_price_increase_l2577_257768


namespace NUMINAMATH_CALUDE_specific_plot_fencing_cost_l2577_257720

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculates the total cost of fencing for a given rectangular plot. -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot. -/
theorem specific_plot_fencing_cost :
  let plot : RectangularPlot := {
    length := 56,
    breadth := 56 - 12,
    fencing_cost_per_meter := 26.5
  }
  total_fencing_cost plot = 5300 := by
  sorry


end NUMINAMATH_CALUDE_specific_plot_fencing_cost_l2577_257720


namespace NUMINAMATH_CALUDE_square_side_prime_l2577_257775

/-- Given an integer 'a' representing the side length of a square, if it's impossible to construct 
    a rectangle with the same area as the square where both sides of the rectangle are integers 
    greater than 1, then 'a' must be a prime number. -/
theorem square_side_prime (a : ℕ) (h : a > 1) : 
  (∀ m n : ℕ, m > 1 → n > 1 → m * n ≠ a * a) → Nat.Prime a := by
  sorry

end NUMINAMATH_CALUDE_square_side_prime_l2577_257775


namespace NUMINAMATH_CALUDE_computer_price_increase_l2577_257704

theorem computer_price_increase (c : ℝ) : 
  c + c * 0.3 = 351 → c + 351 = 621 :=
by sorry

end NUMINAMATH_CALUDE_computer_price_increase_l2577_257704


namespace NUMINAMATH_CALUDE_quadratic_negative_range_l2577_257754

/-- The quadratic function f(x) = ax^2 + 2ax + m -/
def f (a m : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + m

theorem quadratic_negative_range (a m : ℝ) (h1 : a < 0) (h2 : f a m 2 = 0) :
  {x : ℝ | f a m x < 0} = {x : ℝ | x < -4 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_range_l2577_257754


namespace NUMINAMATH_CALUDE_additional_amount_needed_for_free_shipping_l2577_257746

def free_shipping_threshold : ℝ := 50.00

def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00

def first_two_discount : ℝ := 0.25
def total_discount : ℝ := 0.10

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_price : ℝ :=
  discounted_price book1_price first_two_discount +
  discounted_price book2_price first_two_discount +
  book3_price + book4_price

def final_price : ℝ :=
  discounted_price total_price total_discount

theorem additional_amount_needed_for_free_shipping :
  free_shipping_threshold - final_price = 13.10 := by
  sorry

end NUMINAMATH_CALUDE_additional_amount_needed_for_free_shipping_l2577_257746


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2577_257706

theorem regular_polygon_sides (interior_angle : ℝ) : 
  interior_angle = 140 → (360 / (180 - interior_angle) : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2577_257706


namespace NUMINAMATH_CALUDE_sin_cos_sum_identity_l2577_257752

theorem sin_cos_sum_identity (x y : ℝ) :
  Real.sin (x + y) * Real.cos (2 * y) + Real.cos (x + y) * Real.sin (2 * y) = Real.sin (x + 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_identity_l2577_257752


namespace NUMINAMATH_CALUDE_player_a_wins_two_player_player_b_wins_three_player_l2577_257772

/-- Represents a player in the Lazy Checkers game -/
inductive Player : Type
| A : Player
| B : Player
| C : Player

/-- Represents a position on the 5x5 board -/
structure Position :=
(row : Fin 5)
(col : Fin 5)

/-- Represents the state of the Lazy Checkers game -/
structure GameState :=
(board : Position → Option Player)
(current_player : Player)

/-- Represents a winning strategy for a player -/
def WinningStrategy (p : Player) : Type :=
GameState → Position

/-- The rules of Lazy Checkers ensure a valid game state -/
def ValidGameState (state : GameState) : Prop :=
sorry

/-- Theorem: In a two-player Lazy Checkers game, Player A has a winning strategy -/
theorem player_a_wins_two_player :
  ∃ (strategy : WinningStrategy Player.A),
    ∀ (initial_state : GameState),
      ValidGameState initial_state →
      initial_state.current_player = Player.A →
      -- The strategy leads to a win for Player A
      sorry :=
sorry

/-- Theorem: In a three-player Lazy Checkers game, Player B has a winning strategy when cooperating with Player C -/
theorem player_b_wins_three_player :
  ∃ (strategy_b : WinningStrategy Player.B) (strategy_c : WinningStrategy Player.C),
    ∀ (initial_state : GameState),
      ValidGameState initial_state →
      initial_state.current_player = Player.A →
      -- The strategies lead to a win for Player B
      sorry :=
sorry

end NUMINAMATH_CALUDE_player_a_wins_two_player_player_b_wins_three_player_l2577_257772


namespace NUMINAMATH_CALUDE_larger_rhombus_side_length_l2577_257757

/-- Two similar rhombi sharing a diagonal -/
structure SimilarRhombi where
  small_area : ℝ
  large_area : ℝ
  shared_diagonal : ℝ
  similar : small_area > 0 ∧ large_area > 0

/-- The side length of a rhombus -/
def side_length (r : SimilarRhombi) : ℝ → ℝ := sorry

/-- Theorem: The side length of the larger rhombus is √15 -/
theorem larger_rhombus_side_length (r : SimilarRhombi) 
  (h1 : r.small_area = 1) 
  (h2 : r.large_area = 9) : 
  side_length r r.large_area = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_larger_rhombus_side_length_l2577_257757


namespace NUMINAMATH_CALUDE_original_price_calculation_l2577_257758

theorem original_price_calculation (discount_percentage : ℝ) (selling_price : ℝ) 
  (h1 : discount_percentage = 20)
  (h2 : selling_price = 14) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_percentage / 100) = selling_price ∧ 
    original_price = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2577_257758


namespace NUMINAMATH_CALUDE_target_breaking_orders_l2577_257731

/-- The number of targets in the first column -/
def column_A : ℕ := 4

/-- The number of targets in the second column -/
def column_B : ℕ := 3

/-- The number of targets in the third column -/
def column_C : ℕ := 3

/-- The total number of targets -/
def total_targets : ℕ := column_A + column_B + column_C

/-- The number of different orders to break the targets -/
def break_orders : ℕ := (Nat.factorial total_targets) / 
  (Nat.factorial column_A * Nat.factorial column_B * Nat.factorial column_C)

theorem target_breaking_orders : break_orders = 4200 := by
  sorry

end NUMINAMATH_CALUDE_target_breaking_orders_l2577_257731


namespace NUMINAMATH_CALUDE_expansion_terms_count_l2577_257726

theorem expansion_terms_count (G1 G2 : Finset (Fin 4)) 
  (hG1 : G1.card = 4) (hG2 : G2.card = 4) :
  (G1.product G2).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l2577_257726


namespace NUMINAMATH_CALUDE_exactly_two_out_of_three_germinate_l2577_257705

def seed_germination_probability : ℚ := 3/5

def exactly_two_out_of_three_probability : ℚ :=
  3 * seed_germination_probability^2 * (1 - seed_germination_probability)

theorem exactly_two_out_of_three_germinate :
  exactly_two_out_of_three_probability = 54/125 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_out_of_three_germinate_l2577_257705


namespace NUMINAMATH_CALUDE_product_of_differences_l2577_257787

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2006) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2007)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2006) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2007)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2006) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2007) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/2006 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l2577_257787


namespace NUMINAMATH_CALUDE_stamp_difference_l2577_257788

theorem stamp_difference (k a : ℕ) (h1 : k * 3 = a * 5) 
  (h2 : (k - 12) * 6 = (a + 12) * 8) : k - 12 = (a + 12) + 32 := by
  sorry

end NUMINAMATH_CALUDE_stamp_difference_l2577_257788


namespace NUMINAMATH_CALUDE_beans_remaining_fraction_l2577_257776

/-- Given a jar and coffee beans, where:
  1. The weight of the jar is 10% of the total weight when filled with beans.
  2. After removing some beans, the weight of the jar and remaining beans is 60% of the original total weight.
  Prove that the fraction of beans remaining in the jar is 5/9. -/
theorem beans_remaining_fraction (jar_weight : ℝ) (full_beans_weight : ℝ) 
  (remaining_beans_weight : ℝ) : 
  (jar_weight = 0.1 * (jar_weight + full_beans_weight)) →
  (jar_weight + remaining_beans_weight = 0.6 * (jar_weight + full_beans_weight)) →
  (remaining_beans_weight / full_beans_weight = 5 / 9) :=
by sorry

end NUMINAMATH_CALUDE_beans_remaining_fraction_l2577_257776


namespace NUMINAMATH_CALUDE_proportion_problem_l2577_257790

theorem proportion_problem (y : ℝ) : 
  (0.25 : ℝ) / 0.75 = y / 6 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l2577_257790


namespace NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l2577_257797

def is_valid_abcba (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_abcba_divisible_by_13 :
  ∀ n : ℕ, is_valid_abcba n → n ≤ 96769 ∧ 96769 % 13 = 0 ∧ is_valid_abcba 96769 :=
sorry

end NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l2577_257797


namespace NUMINAMATH_CALUDE_difference_of_two_numbers_l2577_257724

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) :
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_difference_of_two_numbers_l2577_257724


namespace NUMINAMATH_CALUDE_max_satisfying_all_is_50_l2577_257771

/-- Represents the youth summer village population --/
structure Village where
  total : ℕ
  notWorking : ℕ
  withFamily : ℕ
  singingInShower : ℕ

/-- The conditions of the problem --/
def problemVillage : Village :=
  { total := 100
  , notWorking := 50
  , withFamily := 25
  , singingInShower := 75 }

/-- The maximum number of people satisfying all conditions --/
def maxSatisfyingAll (v : Village) : ℕ :=
  min (v.total - v.notWorking) (min (v.total - v.withFamily) v.singingInShower)

/-- Theorem stating the maximum number of people satisfying all conditions --/
theorem max_satisfying_all_is_50 :
  maxSatisfyingAll problemVillage = 50 := by sorry

end NUMINAMATH_CALUDE_max_satisfying_all_is_50_l2577_257771


namespace NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l2577_257701

/-- Given a real number a, we define a function f on (-∞, a] such that f(x) = x + 1.
    We assume that for all x and y in (-∞, a], f(x+y) ≤ 2f(x) - 3f(y).
    This theorem states that under these conditions, a must be less than or equal to -2. -/
theorem function_inequality_implies_upper_bound (a : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, x ≤ a → f x = x + 1)
  (h2 : ∀ x y, x ≤ a → y ≤ a → f (x + y) ≤ 2 * f x - 3 * f y) :
  a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l2577_257701


namespace NUMINAMATH_CALUDE_cyclist_final_speed_l2577_257711

/-- Calculates the final speed of a cyclist given initial speed, acceleration, and time. -/
def final_speed (initial_speed : ℝ) (acceleration : ℝ) (time : ℝ) : ℝ :=
  initial_speed + acceleration * time

/-- Converts speed from m/s to km/h. -/
def ms_to_kmh (speed_ms : ℝ) : ℝ :=
  speed_ms * 3.6

theorem cyclist_final_speed :
  let initial_speed := 16 -- m/s
  let acceleration := 0.5 -- m/s²
  let time := 2 * 3600 -- 2 hours in seconds
  let final_speed_ms := final_speed initial_speed acceleration time
  let final_speed_kmh := ms_to_kmh final_speed_ms
  final_speed_kmh = 13017.6 := by sorry

end NUMINAMATH_CALUDE_cyclist_final_speed_l2577_257711


namespace NUMINAMATH_CALUDE_cinema_lineup_ways_l2577_257747

def number_of_people : ℕ := 8
def number_of_windows : ℕ := 2

theorem cinema_lineup_ways :
  (2 ^ number_of_people) * (Nat.factorial number_of_people) = 10321920 := by
  sorry

end NUMINAMATH_CALUDE_cinema_lineup_ways_l2577_257747


namespace NUMINAMATH_CALUDE_pradeep_marks_l2577_257739

theorem pradeep_marks (total_marks : ℕ) (pass_percentage : ℚ) (fail_margin : ℕ) : 
  total_marks = 840 → 
  pass_percentage = 1/4 → 
  (total_marks * pass_percentage).floor - fail_margin = 185 :=
by sorry

end NUMINAMATH_CALUDE_pradeep_marks_l2577_257739


namespace NUMINAMATH_CALUDE_system_solution_l2577_257736

theorem system_solution :
  ∃ (x y : ℚ), 
    (5 * x - 3 * y = -7) ∧ 
    (4 * x + 6 * y = 34) ∧ 
    (x = 10 / 7) ∧ 
    (y = 33 / 7) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2577_257736


namespace NUMINAMATH_CALUDE_honey_water_percentage_l2577_257722

/-- Given that 1.5 kg of flower-nectar yields 1 kg of honey and nectar contains 50% water,
    prove that the resulting honey contains 25% water. -/
theorem honey_water_percentage :
  ∀ (nectar_mass honey_mass water_percentage_nectar : ℝ),
    nectar_mass = 1.5 →
    honey_mass = 1 →
    water_percentage_nectar = 50 →
    (honey_mass - (nectar_mass * (1 - water_percentage_nectar / 100))) / honey_mass * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_honey_water_percentage_l2577_257722


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l2577_257782

/-- The capacity of a water tank in liters -/
def tank_capacity : ℝ := 72

/-- The amount of water in the tank when it's 40% full -/
def water_at_40_percent : ℝ := 0.4 * tank_capacity

/-- The amount of water in the tank when it's 10% empty (90% full) -/
def water_at_90_percent : ℝ := 0.9 * tank_capacity

/-- Theorem stating the tank capacity based on the given condition -/
theorem tank_capacity_proof :
  water_at_90_percent - water_at_40_percent = 36 ∧
  tank_capacity = 72 :=
sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l2577_257782


namespace NUMINAMATH_CALUDE_arithmetic_square_root_when_negative_root_is_five_l2577_257732

theorem arithmetic_square_root_when_negative_root_is_five (x : ℝ) : 
  ((-5 : ℝ)^2 = x) → Real.sqrt x = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_when_negative_root_is_five_l2577_257732


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2577_257725

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 20th term is 15 
    and the 21st term is 18, the 5th term is -30. -/
theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℤ) (h : ArithmeticSequence a) 
  (h20 : a 20 = 15) (h21 : a 21 = 18) : 
  a 5 = -30 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2577_257725


namespace NUMINAMATH_CALUDE_remainder_8423_div_9_l2577_257723

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The digital root of a natural number (iterative sum of digits until a single digit is reached) -/
def digital_root (n : ℕ) : ℕ := sorry

theorem remainder_8423_div_9 : 8423 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_8423_div_9_l2577_257723


namespace NUMINAMATH_CALUDE_min_total_cost_both_measures_l2577_257712

/-- Represents the cost and effectiveness of a preventive measure -/
structure PreventiveMeasure where
  cost : ℝ
  effectiveness : ℝ

/-- Calculates the total cost given the initial probability, potential loss,
    and a list of implemented preventive measures -/
def totalCost (initialProb : ℝ) (potentialLoss : ℝ) (measures : List PreventiveMeasure) : ℝ :=
  let measuresCost := measures.foldl (fun acc m => acc + m.cost) 0
  let finalProb := measures.foldl (fun acc m => acc * (1 - m.effectiveness)) initialProb
  measuresCost + finalProb * potentialLoss

/-- Theorem stating that the minimum total cost is achieved by implementing both measures -/
theorem min_total_cost_both_measures
  (initialProb : ℝ)
  (potentialLoss : ℝ)
  (measureA : PreventiveMeasure)
  (measureB : PreventiveMeasure)
  (h_initialProb : initialProb = 0.3)
  (h_potentialLoss : potentialLoss = 400)
  (h_measureA : measureA = { cost := 0.45, effectiveness := 0.9 })
  (h_measureB : measureB = { cost := 0.3, effectiveness := 0.85 }) :
  (totalCost initialProb potentialLoss [measureA, measureB] ≤ 
   min (totalCost initialProb potentialLoss [])
      (min (totalCost initialProb potentialLoss [measureA])
           (totalCost initialProb potentialLoss [measureB]))) ∧
  (totalCost initialProb potentialLoss [measureA, measureB] = 81) := by
  sorry

#check min_total_cost_both_measures

end NUMINAMATH_CALUDE_min_total_cost_both_measures_l2577_257712


namespace NUMINAMATH_CALUDE_unique_polygon_diagonals_l2577_257727

/-- The number of diagonals in a convex polygon with k sides -/
def numDiagonals (k : ℕ) : ℚ := (k * (k - 3)) / 2

/-- The condition for the number of diagonals in the two polygons -/
def diagonalCondition (n : ℕ) : Prop :=
  numDiagonals (3 * n + 2) = (1 - 0.615) * numDiagonals (5 * n - 2)

theorem unique_polygon_diagonals : ∃! (n : ℕ), n > 0 ∧ diagonalCondition n :=
  sorry

end NUMINAMATH_CALUDE_unique_polygon_diagonals_l2577_257727


namespace NUMINAMATH_CALUDE_log_difference_equals_six_l2577_257780

theorem log_difference_equals_six : 
  ∀ (log₄ : ℝ → ℝ),
  (log₄ 256 = 4) →
  (log₄ (1/16) = -2) →
  (log₄ 256 - log₄ (1/16) = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_six_l2577_257780


namespace NUMINAMATH_CALUDE_negation_equivalence_l2577_257749

theorem negation_equivalence : 
  (¬∃ x₀ : ℝ, x₀^2 - 2*x₀ + 4 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2577_257749


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l2577_257734

theorem min_value_abs_sum (x : ℝ) (h : 0 ≤ x ∧ x ≤ 10) :
  |x - 4| + |x + 2| + |x - 5| + |3*x - 1| + |2*x + 6| ≥ 17.333 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l2577_257734


namespace NUMINAMATH_CALUDE_probability_product_72_l2577_257730

/-- A function representing the possible outcomes of rolling a standard die -/
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := (standardDie.card) ^ 3

/-- The number of favorable outcomes (combinations that multiply to 72) -/
def favorableOutcomes : ℕ := 6

/-- The probability of rolling three dice such that their product is 72 -/
def probabilityProductIs72 : ℚ := favorableOutcomes / totalOutcomes

theorem probability_product_72 : probabilityProductIs72 = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_product_72_l2577_257730


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l2577_257778

/-- Two triangles are similar if they have the same shape but not necessarily the same size. -/
def are_similar (t1 t2 : Triangle) : Prop := sorry

/-- An equilateral triangle has all sides equal and all angles equal to 60°. -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- An isosceles triangle with a 120° angle has two equal sides and two base angles of 30°. -/
def is_isosceles_120 (t : Triangle) : Prop := sorry

/-- Two triangles are congruent if they have the same shape and size. -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- A right triangle has one angle of 90°. -/
def is_right_triangle (t : Triangle) : Prop := sorry

theorem triangle_similarity_theorem :
  ∀ t1 t2 : Triangle,
  (is_equilateral t1 ∧ is_equilateral t2) → are_similar t1 t2 ∧
  (is_isosceles_120 t1 ∧ is_isosceles_120 t2) → are_similar t1 t2 ∧
  are_congruent t1 t2 → are_similar t1 t2 ∧
  ∃ t3 t4 : Triangle, is_right_triangle t3 ∧ is_right_triangle t4 ∧ ¬ are_similar t3 t4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l2577_257778


namespace NUMINAMATH_CALUDE_regression_validity_l2577_257753

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Represents the sample means of x and y -/
structure SampleMeans where
  x : ℝ
  y : ℝ

/-- Checks if the linear regression equation is valid for the given sample means -/
def isValidRegression (reg : LinearRegression) (means : SampleMeans) : Prop :=
  means.y = reg.slope * means.x + reg.intercept

/-- Theorem stating that the given linear regression is valid for the provided sample means -/
theorem regression_validity (means : SampleMeans) 
    (h_corr : 0 < 0.4) -- Positive correlation between x and y
    (h_means_x : means.x = 3)
    (h_means_y : means.y = 3.5) :
    isValidRegression ⟨0.4, 2.3⟩ means := by
  sorry

end NUMINAMATH_CALUDE_regression_validity_l2577_257753


namespace NUMINAMATH_CALUDE_johnny_works_four_and_half_hours_l2577_257716

/-- Represents Johnny's dog walking business --/
structure DogWalker where
  dogs_per_walk : ℕ
  pay_30min : ℕ
  pay_60min : ℕ
  long_walks_per_day : ℕ
  work_days_per_week : ℕ
  weekly_earnings : ℕ

/-- Calculates the number of hours Johnny works per day --/
def hours_worked_per_day (dw : DogWalker) : ℚ :=
  let long_walk_earnings := dw.pay_60min * (dw.long_walks_per_day / dw.dogs_per_walk)
  let weekly_long_walk_earnings := long_walk_earnings * dw.work_days_per_week
  let weekly_short_walk_earnings := dw.weekly_earnings - weekly_long_walk_earnings
  let short_walks_per_week := weekly_short_walk_earnings / dw.pay_30min
  let short_walks_per_day := short_walks_per_week / dw.work_days_per_week
  let short_walk_sets_per_day := short_walks_per_day / dw.dogs_per_walk
  ((dw.long_walks_per_day / dw.dogs_per_walk) * 60 + short_walk_sets_per_day * 30) / 60

/-- Theorem stating that Johnny works 4.5 hours per day --/
theorem johnny_works_four_and_half_hours
  (johnny : DogWalker)
  (h1 : johnny.dogs_per_walk = 3)
  (h2 : johnny.pay_30min = 15)
  (h3 : johnny.pay_60min = 20)
  (h4 : johnny.long_walks_per_day = 6)
  (h5 : johnny.work_days_per_week = 5)
  (h6 : johnny.weekly_earnings = 1500) :
  hours_worked_per_day johnny = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_johnny_works_four_and_half_hours_l2577_257716


namespace NUMINAMATH_CALUDE_chili_paste_can_difference_l2577_257756

def large_can_size : ℕ := 25
def small_can_size : ℕ := 15
def large_cans_needed : ℕ := 45

theorem chili_paste_can_difference :
  (large_cans_needed * large_can_size) / small_can_size - large_cans_needed = 30 :=
by sorry

end NUMINAMATH_CALUDE_chili_paste_can_difference_l2577_257756


namespace NUMINAMATH_CALUDE_factorization_equality_l2577_257717

theorem factorization_equality (x : ℝ) : 
  (3 * x^3 + 48 * x^2 - 14) - (-9 * x^3 + 2 * x^2 - 14) = 2 * x^2 * (6 * x + 23) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2577_257717


namespace NUMINAMATH_CALUDE_bracket_difference_l2577_257707

theorem bracket_difference (a b c : ℝ) : (a - (b - c)) - ((a - b) - c) = 2 * c := by
  sorry

end NUMINAMATH_CALUDE_bracket_difference_l2577_257707


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2577_257765

/-- Represents an ellipse with equation kx^2 + y^2 = 2 and focus on x-axis -/
structure Ellipse (k : ℝ) where
  equation : ∀ (x y : ℝ), k * x^2 + y^2 = 2
  focus_on_x_axis : True  -- This is a placeholder for the focus condition

/-- The range of k for an ellipse with equation kx^2 + y^2 = 2 and focus on x-axis is (0, 1) -/
theorem ellipse_k_range (k : ℝ) (e : Ellipse k) : 0 < k ∧ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2577_257765


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2577_257700

theorem subtraction_preserves_inequality (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2577_257700


namespace NUMINAMATH_CALUDE_age_difference_l2577_257719

/-- Given three people A, B, and C, where the total age of A and B is 18 years more than
    the total age of B and C, prove that C is 18 years younger than A. -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 18) : A = C + 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2577_257719


namespace NUMINAMATH_CALUDE_parallelogram_area_l2577_257714

/-- Represents a parallelogram ABCD with specific properties -/
structure Parallelogram where
  -- Length of side AB
  a : ℝ
  -- Height of the parallelogram
  v : ℝ
  -- Ensures a and v are positive
  a_pos : 0 < a
  v_pos : 0 < v
  -- When F is 1/5 of BD from D, shaded area is 1 cm² greater than when F is 2/5 of BD from D
  area_difference : (17/50 - 13/50) * (a * v) = 1

/-- The area of a parallelogram with the given properties is 12.5 cm² -/
theorem parallelogram_area (p : Parallelogram) : p.a * p.v = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2577_257714


namespace NUMINAMATH_CALUDE_remaining_work_time_l2577_257728

theorem remaining_work_time (a_rate b_rate : ℚ) (b_work_days : ℕ) : 
  a_rate = 1 / 12 →
  b_rate = 1 / 15 →
  b_work_days = 10 →
  (1 - b_rate * b_work_days) / a_rate = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_work_time_l2577_257728


namespace NUMINAMATH_CALUDE_power_function_through_point_l2577_257718

/-- A power function passing through the point (33, 3) has exponent 3 -/
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x^α) →  -- f is a power function with exponent α
  f 33 = 3 →          -- f passes through the point (33, 3)
  α = 3 :=             -- the exponent α is equal to 3
by
  sorry


end NUMINAMATH_CALUDE_power_function_through_point_l2577_257718


namespace NUMINAMATH_CALUDE_coins_divisible_by_six_l2577_257735

theorem coins_divisible_by_six (n : ℕ) : 
  (∃ (a b c : ℕ), n = 2*a + 2*b + 2*c) ∧ 
  (∃ (x y : ℕ), n = 3*x + 3*y) → 
  ∃ (z : ℕ), n = 6*z :=
sorry

end NUMINAMATH_CALUDE_coins_divisible_by_six_l2577_257735


namespace NUMINAMATH_CALUDE_two_triangles_exist_l2577_257737

/-- Given a side length, ratio of other sides, and circumradius, prove existence of two triangles -/
theorem two_triangles_exist (a : ℝ) (k : ℝ) (R : ℝ) 
    (h_a : a > 0) (h_k : k > 0) (h_R : R > 0) (h_aR : a < 2*R) : 
  ∃ (b₁ c₁ b₂ c₂ : ℝ), 
    (b₁ > 0 ∧ c₁ > 0 ∧ b₂ > 0 ∧ c₂ > 0) ∧ 
    (b₁/c₁ = k ∧ b₂/c₂ = k) ∧
    (a + b₁ > c₁ ∧ b₁ + c₁ > a ∧ c₁ + a > b₁) ∧
    (a + b₂ > c₂ ∧ b₂ + c₂ > a ∧ c₂ + a > b₂) ∧
    (b₁ ≠ b₂ ∨ c₁ ≠ c₂) ∧
    (4 * R * R * (a + b₁ + c₁) = (a * b₁ * c₁) / R) ∧
    (4 * R * R * (a + b₂ + c₂) = (a * b₂ * c₂) / R) :=
by sorry

end NUMINAMATH_CALUDE_two_triangles_exist_l2577_257737


namespace NUMINAMATH_CALUDE_quadratic_zero_condition_l2577_257759

/-- A quadratic function f(x) = x^2 - 2x + m has a zero in (-1, 0) if and only if -3 < m < 0 -/
theorem quadratic_zero_condition (m : ℝ) : 
  (∃ x : ℝ, -1 < x ∧ x < 0 ∧ x^2 - 2*x + m = 0) ↔ -3 < m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_zero_condition_l2577_257759


namespace NUMINAMATH_CALUDE_zhang_ning_match_results_l2577_257795

/-- Represents the outcome of a badminton match --/
inductive MatchOutcome
  | WinTwoZero
  | WinTwoOne
  | LoseTwoOne
  | LoseTwoZero

/-- Probability of Xie Xingfang winning a single set in the first two sets --/
def p_xie : ℝ := 0.6

/-- Probability of Zhang Ning winning the third set if the score reaches 1:1 --/
def p_zhang_third : ℝ := 0.6

/-- Calculates the probability of Zhang Ning winning with a score of 2:1 --/
def prob_zhang_win_two_one : ℝ :=
  2 * (1 - p_xie) * p_xie * p_zhang_third

/-- Calculates the expected value of Zhang Ning's net winning sets --/
def expected_net_wins : ℝ :=
  -2 * (p_xie * p_xie) +
  -1 * (2 * (1 - p_xie) * p_xie * (1 - p_zhang_third)) +
  1 * prob_zhang_win_two_one +
  2 * ((1 - p_xie) * (1 - p_xie))

/-- Theorem stating the probability of Zhang Ning winning 2:1 and her expected net winning sets --/
theorem zhang_ning_match_results :
  prob_zhang_win_two_one = 0.288 ∧ expected_net_wins = 0.496 := by
  sorry


end NUMINAMATH_CALUDE_zhang_ning_match_results_l2577_257795


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l2577_257742

theorem triangle_side_ratio (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ x y, (x = a ∧ y = b) ∨ (x = a ∧ y = c) ∨ (x = b ∧ y = c) ∧
  ((Real.sqrt 5 - 1) / 2 ≤ x / y) ∧ (x / y ≤ (Real.sqrt 5 + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l2577_257742


namespace NUMINAMATH_CALUDE_solution_ratio_l2577_257710

theorem solution_ratio (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_ratio_l2577_257710


namespace NUMINAMATH_CALUDE_antonios_meatballs_l2577_257784

/-- Antonio's meatball problem -/
theorem antonios_meatballs (recipe_amount : ℚ) (family_members : ℕ) (total_hamburger : ℚ) : 
  recipe_amount = 1/8 →
  family_members = 8 →
  total_hamburger = 4 →
  (total_hamburger / recipe_amount) / family_members = 4 :=
by sorry

end NUMINAMATH_CALUDE_antonios_meatballs_l2577_257784


namespace NUMINAMATH_CALUDE_number_of_classes_for_histogram_l2577_257798

theorem number_of_classes_for_histogram (tallest_height shortest_height class_interval : ℝ)
  (h1 : tallest_height = 186)
  (h2 : shortest_height = 154)
  (h3 : class_interval = 5)
  : Int.ceil ((tallest_height - shortest_height) / class_interval) = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_classes_for_histogram_l2577_257798


namespace NUMINAMATH_CALUDE_fifth_root_over_sixth_root_of_eleven_l2577_257769

theorem fifth_root_over_sixth_root_of_eleven (x : ℝ) :
  (11 ^ (1/5)) / (11 ^ (1/6)) = 11 ^ (1/30) :=
sorry

end NUMINAMATH_CALUDE_fifth_root_over_sixth_root_of_eleven_l2577_257769


namespace NUMINAMATH_CALUDE_rakesh_distance_rakesh_walked_approx_distance_l2577_257779

/-- Proves that Rakesh walked approximately 28.29 kilometers given the conditions of the problem. -/
theorem rakesh_distance (hiro_distance : ℝ) : ℝ :=
  let rakesh_distance := 4 * hiro_distance - 10
  let sanjay_distance := 2 * hiro_distance + 3
  have total_distance : hiro_distance + rakesh_distance + sanjay_distance = 60 := by sorry
  have hiro_calc : hiro_distance = 67 / 7 := by sorry
  rakesh_distance

/-- The approximate distance Rakesh walked -/
def rakesh_approx_distance : ℝ := 28.29

theorem rakesh_walked_approx_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |rakesh_distance (67 / 7) - rakesh_approx_distance| < ε :=
by sorry

end NUMINAMATH_CALUDE_rakesh_distance_rakesh_walked_approx_distance_l2577_257779


namespace NUMINAMATH_CALUDE_triangle_area_is_two_l2577_257794

/-- A line in 2D space --/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The area of a triangle given by three points --/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- The intersection point of two lines --/
def lineIntersection (l1 l2 : Line) : ℝ × ℝ := sorry

/-- The main theorem --/
theorem triangle_area_is_two :
  let line1 : Line := { slope := 3/4, point := (3, 3) }
  let line2 : Line := { slope := -1, point := (3, 3) }
  let line3 : Line := { slope := -1, point := (0, 14) }
  let p1 := (3, 3)
  let p2 := lineIntersection line1 line3
  let p3 := lineIntersection line2 line3
  triangleArea p1 p2 p3 = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_two_l2577_257794


namespace NUMINAMATH_CALUDE_segment_distinctness_l2577_257789

theorem segment_distinctness (n : ℕ) (h : n ≥ 4) :
  ¬ ∀ (points : Fin (n + 1) → ℕ),
    (points 0 = 0 ∧ points (Fin.last n) = (n^2 + n) / 2) →
    (∀ i j : Fin (n + 1), i < j → points i < points j) →
    (∀ i j k l : Fin (n + 1), i < j ∧ k < l → 
      (points j - points i ≠ points l - points k ∨ (i = k ∧ j = l))) :=
by sorry

end NUMINAMATH_CALUDE_segment_distinctness_l2577_257789


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2577_257738

def M : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def N : Set ℝ := {x | 2*x + 1 < 5}

theorem union_of_M_and_N : M ∪ N = {x | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2577_257738


namespace NUMINAMATH_CALUDE_floor_equality_sufficient_not_necessary_for_abs_diff_less_than_one_l2577_257764

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := 
  Int.floor x

-- Theorem statement
theorem floor_equality_sufficient_not_necessary_for_abs_diff_less_than_one :
  (∀ x y : ℝ, floor x = floor y → |x - y| < 1) ∧
  (∃ x y : ℝ, |x - y| < 1 ∧ floor x ≠ floor y) := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_sufficient_not_necessary_for_abs_diff_less_than_one_l2577_257764
