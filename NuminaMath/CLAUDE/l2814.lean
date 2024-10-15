import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_proposition_l2814_281453

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → 3 * x^2 - x - 2 > 0) ↔ 
  (∃ x : ℝ, x > 0 ∧ 3 * x^2 - x - 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2814_281453


namespace NUMINAMATH_CALUDE_tangent_and_trig_identity_l2814_281491

theorem tangent_and_trig_identity (α : Real) (h : Real.tan α = -2) : 
  Real.tan (α - 7 * Real.pi) = -2 ∧ 
  (2 * Real.sin (Real.pi - α) * Real.sin (α - Real.pi / 2)) / 
  (Real.sin α ^ 2 - 2 * Real.cos α ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_trig_identity_l2814_281491


namespace NUMINAMATH_CALUDE_overlaid_triangles_result_l2814_281430

/-- Represents a transparent sheet with shaded triangles -/
structure Sheet :=
  (total_triangles : Nat)
  (shaded_triangles : Nat)

/-- Calculates the number of visible shaded triangles when sheets are overlaid -/
def visible_shaded_triangles (sheets : List Sheet) : Nat :=
  sorry

/-- Theorem stating the result for the specific problem -/
theorem overlaid_triangles_result :
  let sheets := [
    { total_triangles := 49, shaded_triangles := 16 },
    { total_triangles := 49, shaded_triangles := 16 },
    { total_triangles := 49, shaded_triangles := 16 }
  ]
  visible_shaded_triangles sheets = 31 := by
  sorry

end NUMINAMATH_CALUDE_overlaid_triangles_result_l2814_281430


namespace NUMINAMATH_CALUDE_rectangle_same_color_l2814_281474

-- Define the color type
def Color := Fin

-- Define a point on the plane
structure Point where
  x : Int
  y : Int

-- Define a coloring function
def coloring (p : Nat) : Point → Color p :=
  sorry

-- The main theorem
theorem rectangle_same_color (p : Nat) :
  ∃ (a b c d : Point), 
    (a.x < b.x ∧ a.y < c.y) ∧ 
    (b.x - a.x = d.x - c.x) ∧ 
    (c.y - a.y = d.y - b.y) ∧
    (coloring p a = coloring p b) ∧
    (coloring p b = coloring p c) ∧
    (coloring p c = coloring p d) :=
  sorry

end NUMINAMATH_CALUDE_rectangle_same_color_l2814_281474


namespace NUMINAMATH_CALUDE_darias_savings_correct_l2814_281489

/-- Calculates the weekly savings amount needed to reach a target --/
def weekly_savings (total_cost : ℕ) (initial_savings : ℕ) (weeks : ℕ) : ℕ :=
  (total_cost - initial_savings) / weeks

/-- Proves that Daria's weekly savings amount is correct --/
theorem darias_savings_correct (total_cost initial_savings weeks : ℕ)
  (h1 : total_cost = 120)
  (h2 : initial_savings = 20)
  (h3 : weeks = 10) :
  weekly_savings total_cost initial_savings weeks = 10 := by
  sorry

end NUMINAMATH_CALUDE_darias_savings_correct_l2814_281489


namespace NUMINAMATH_CALUDE_equal_color_distribution_l2814_281441

/-- The number of balls -/
def n : ℕ := 8

/-- The probability of a ball being painted black or white -/
def p : ℚ := 1/2

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of having exactly 4 black and 4 white balls -/
def prob_four_black_four_white : ℚ :=
  (choose n (n/2) : ℚ) * p^n

theorem equal_color_distribution :
  prob_four_black_four_white = 35/128 :=
sorry

end NUMINAMATH_CALUDE_equal_color_distribution_l2814_281441


namespace NUMINAMATH_CALUDE_largest_factorial_divisor_l2814_281466

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem largest_factorial_divisor :
  ∀ m : ℕ, m > 98 → ¬(factorial m ∣ factorial 100 + factorial 99 + factorial 98) ∧
  (factorial 98 ∣ factorial 100 + factorial 99 + factorial 98) := by
  sorry

end NUMINAMATH_CALUDE_largest_factorial_divisor_l2814_281466


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2814_281422

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x + 3 > 2 → -x < 6) ∧ 
  (∃ x : ℝ, -x < 6 ∧ ¬(x + 3 > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2814_281422


namespace NUMINAMATH_CALUDE_employee_pay_l2814_281418

theorem employee_pay (total : ℝ) (ratio : ℝ) (lower_pay : ℝ) : 
  total = 580 →
  ratio = 1.5 →
  total = lower_pay + ratio * lower_pay →
  lower_pay = 232 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_l2814_281418


namespace NUMINAMATH_CALUDE_four_teacher_proctoring_l2814_281452

/-- Represents the number of teachers and classes -/
def n : ℕ := 4

/-- The number of ways to arrange n teachers to proctor n classes, where no teacher proctors their own class -/
def derangement (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of ways to arrange 4 teachers to proctor 4 classes, where no teacher proctors their own class, is equal to 9 -/
theorem four_teacher_proctoring : derangement n = 9 := by sorry

end NUMINAMATH_CALUDE_four_teacher_proctoring_l2814_281452


namespace NUMINAMATH_CALUDE_cookie_sheet_length_l2814_281449

/-- Given a rectangle with width 10 inches and perimeter 24 inches, prove its length is 2 inches. -/
theorem cookie_sheet_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 10 → perimeter = 24 → perimeter = 2 * (length + width) → length = 2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sheet_length_l2814_281449


namespace NUMINAMATH_CALUDE_inverse_proportion_points_l2814_281423

/-- Given that (2,3) lies on the graph of y = k/x (k ≠ 0), prove that (1,6) also lies on the same graph. -/
theorem inverse_proportion_points : ∀ k : ℝ, k ≠ 0 → (3 = k / 2) → (6 = k / 1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_l2814_281423


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2814_281427

theorem arithmetic_calculation : 5 * 6 - 2 * 3 + 7 * 4 + 9 * 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2814_281427


namespace NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_per_day_l2814_281433

/-- Represents the number of blocks Ray walks in each segment of his route -/
structure RouteSegments where
  toPark : ℕ
  toHighSchool : ℕ
  toHome : ℕ

/-- Calculates the total number of blocks walked in one complete route -/
def totalBlocksPerWalk (route : RouteSegments) : ℕ :=
  route.toPark + route.toHighSchool + route.toHome

/-- Represents Ray's daily dog walking routine -/
structure DailyWalk where
  route : RouteSegments
  frequency : ℕ

/-- Calculates the total number of blocks walked per day -/
def totalBlocksPerDay (daily : DailyWalk) : ℕ :=
  (totalBlocksPerWalk daily.route) * daily.frequency

/-- Theorem: Ray's dog walks 66 blocks each day -/
theorem rays_dog_walks_66_blocks_per_day :
  ∀ (daily : DailyWalk),
    daily.route.toPark = 4 →
    daily.route.toHighSchool = 7 →
    daily.route.toHome = 11 →
    daily.frequency = 3 →
    totalBlocksPerDay daily = 66 := by
  sorry


end NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_per_day_l2814_281433


namespace NUMINAMATH_CALUDE_not_center_of_symmetry_l2814_281476

/-- Given that the centers of symmetry for tan(x) are of the form (kπ/2, 0) where k is any integer,
    prove that (-π/18, 0) is not a center of symmetry for the function t = tan(3x + π/3) -/
theorem not_center_of_symmetry :
  ¬ (∃ (k : ℤ), -π/18 = k*π/6 - π/9) := by sorry

end NUMINAMATH_CALUDE_not_center_of_symmetry_l2814_281476


namespace NUMINAMATH_CALUDE_same_solution_implies_c_equals_6_l2814_281463

theorem same_solution_implies_c_equals_6 (x : ℝ) (c : ℝ) : 
  (3 * x + 6 = 0) → (c * x + 15 = 3) → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_equals_6_l2814_281463


namespace NUMINAMATH_CALUDE_max_pairs_correct_l2814_281428

def max_pairs (n : ℕ) : ℕ :=
  let k := (8037 : ℕ) / 5
  k

theorem max_pairs_correct (n : ℕ) (h : n = 4019) :
  ∀ (k : ℕ) (pairs : List (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
    pairs.length ≤ max_pairs n :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_correct_l2814_281428


namespace NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_8_pow_2003_l2814_281490

/-- The sum of the tens digit and the units digit in the decimal representation of 8^2003 is 2 -/
theorem sum_of_tens_and_units_digits_of_8_pow_2003 : ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ a + b = 2 ∧ 
  (∃ (k : ℕ), 8^2003 = k * 100 + a * 10 + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_8_pow_2003_l2814_281490


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2814_281431

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 + a 13 + a 14 + a 15 = 8) :
  5 * a 7 - 2 * a 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2814_281431


namespace NUMINAMATH_CALUDE_max_checkers_theorem_l2814_281470

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  size : Nat
  white_checkers : Nat
  black_checkers : Nat

/-- Checks if a chessboard configuration is valid -/
def is_valid_config (c : ChessboardConfig) : Prop :=
  c.size = 8 ∧
  c.white_checkers = 2 * c.black_checkers ∧
  c.white_checkers + c.black_checkers ≤ c.size * c.size

/-- The maximum number of checkers that can be placed -/
def max_checkers : Nat := 48

/-- Theorem: The maximum number of checkers that can be placed on an 8x8 chessboard,
    such that each row and column contains twice as many white checkers as black ones, is 48 -/
theorem max_checkers_theorem (c : ChessboardConfig) :
  is_valid_config c → c.white_checkers + c.black_checkers ≤ max_checkers :=
by
  sorry

#check max_checkers_theorem

end NUMINAMATH_CALUDE_max_checkers_theorem_l2814_281470


namespace NUMINAMATH_CALUDE_prob_first_second_win_eq_three_tenths_l2814_281401

/-- Represents a lottery with winning and non-winning tickets -/
structure Lottery where
  total_tickets : ℕ
  winning_tickets : ℕ
  people : ℕ
  h_winning_le_total : winning_tickets ≤ total_tickets
  h_people_le_total : people ≤ total_tickets

/-- The probability of drawing a winning ticket -/
def prob_win (L : Lottery) : ℚ :=
  L.winning_tickets / L.total_tickets

/-- The probability of both the first and second person drawing a winning ticket -/
def prob_first_second_win (L : Lottery) : ℚ :=
  (L.winning_tickets / L.total_tickets) * ((L.winning_tickets - 1) / (L.total_tickets - 1))

/-- Theorem stating the probability of both first and second person drawing a winning ticket -/
theorem prob_first_second_win_eq_three_tenths (L : Lottery) 
    (h_total : L.total_tickets = 5)
    (h_winning : L.winning_tickets = 3)
    (h_people : L.people = 5) :
    prob_first_second_win L = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_prob_first_second_win_eq_three_tenths_l2814_281401


namespace NUMINAMATH_CALUDE_students_disliking_menu_l2814_281402

theorem students_disliking_menu (total : ℕ) (liked : ℕ) (h1 : total = 400) (h2 : liked = 235) :
  total - liked = 165 := by
  sorry

end NUMINAMATH_CALUDE_students_disliking_menu_l2814_281402


namespace NUMINAMATH_CALUDE_machine_parts_replacement_l2814_281451

theorem machine_parts_replacement (num_machines : ℕ) (parts_per_machine : ℕ)
  (fail_rate_week1 : ℚ) (fail_rate_week2 : ℚ) (fail_rate_week3 : ℚ) :
  num_machines = 500 →
  parts_per_machine = 6 →
  fail_rate_week1 = 1/10 →
  fail_rate_week2 = 3/10 →
  fail_rate_week3 = 6/10 →
  (fail_rate_week1 + fail_rate_week2 + fail_rate_week3 = 1) →
  (num_machines * parts_per_machine * fail_rate_week3 +
   (num_machines * parts_per_machine * fail_rate_week2 * fail_rate_week3) +
   (num_machines * parts_per_machine * fail_rate_week1 * fail_rate_week2 * fail_rate_week3) : ℚ) = 1983 := by
  sorry


end NUMINAMATH_CALUDE_machine_parts_replacement_l2814_281451


namespace NUMINAMATH_CALUDE_catherine_pencil_distribution_l2814_281485

theorem catherine_pencil_distribution (initial_pens : ℕ) (initial_pencils : ℕ) 
  (friends : ℕ) (pens_per_friend : ℕ) (total_left : ℕ) :
  initial_pens = 60 →
  initial_pencils = initial_pens →
  friends = 7 →
  pens_per_friend = 8 →
  total_left = 22 →
  ∃ (pencils_per_friend : ℕ),
    pencils_per_friend * friends = initial_pencils - (total_left - (initial_pens - pens_per_friend * friends)) ∧
    pencils_per_friend = 6 :=
by sorry

end NUMINAMATH_CALUDE_catherine_pencil_distribution_l2814_281485


namespace NUMINAMATH_CALUDE_ratio_transformation_l2814_281400

theorem ratio_transformation (x y : ℝ) (h : x / y = 7 / 3) : 
  (x + y) / (x - y) = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_transformation_l2814_281400


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2814_281414

theorem absolute_value_equality (y : ℝ) : |y| = |y - 3| → y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2814_281414


namespace NUMINAMATH_CALUDE_oliver_tickets_used_l2814_281460

theorem oliver_tickets_used (ferris_rides bumper_rides ticket_cost : ℕ) 
  (h1 : ferris_rides = 5)
  (h2 : bumper_rides = 4)
  (h3 : ticket_cost = 7) :
  (ferris_rides + bumper_rides) * ticket_cost = 63 := by
  sorry

end NUMINAMATH_CALUDE_oliver_tickets_used_l2814_281460


namespace NUMINAMATH_CALUDE_f_properties_l2814_281497

/-- The function f(x) that attains an extremum of 0 at x = -1 -/
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a - 1

/-- Theorem stating the properties of f(x) -/
theorem f_properties (a b : ℝ) :
  (f a b (-1) = 0 ∧ (deriv (f a b)) (-1) = 0) →
  (a = 1 ∧ b = 1 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 1 x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2814_281497


namespace NUMINAMATH_CALUDE_problem_statement_l2814_281475

theorem problem_statement (m n : ℤ) (h : 2*m - 3*n = 7) : 8 - 2*m + 3*n = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2814_281475


namespace NUMINAMATH_CALUDE_parallelogram_xy_product_l2814_281493

/-- A parallelogram with side lengths specified by parameters -/
structure Parallelogram (x y : ℝ) where
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ
  ef_eq : ef = 42
  fg_eq : fg = 4 * y^2
  gh_eq : gh = 3 * x + 6
  he_eq : he = 32
  opposite_sides_equal : ef = gh ∧ fg = he

/-- The product of x and y in the specified parallelogram is 24√2 -/
theorem parallelogram_xy_product (x y : ℝ) (p : Parallelogram x y) :
  x * y = 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_xy_product_l2814_281493


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2814_281415

theorem fraction_equation_solution : ∃ x : ℚ, (1 / 3 - 1 / 4 : ℚ) = 1 / x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2814_281415


namespace NUMINAMATH_CALUDE_power_calculation_l2814_281481

theorem power_calculation : (8^5 / 8^2) * 4^6 = 2^21 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2814_281481


namespace NUMINAMATH_CALUDE_inequality_theorem_l2814_281445

theorem inequality_theorem (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  a / (a - c) > b / (b - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2814_281445


namespace NUMINAMATH_CALUDE_essay_word_count_l2814_281410

theorem essay_word_count 
  (intro_length : ℕ) 
  (body_section_length : ℕ) 
  (num_body_sections : ℕ) 
  (h1 : intro_length = 450)
  (h2 : num_body_sections = 4)
  (h3 : body_section_length = 800) : 
  intro_length + 3 * intro_length + num_body_sections * body_section_length = 5000 :=
by sorry

end NUMINAMATH_CALUDE_essay_word_count_l2814_281410


namespace NUMINAMATH_CALUDE_car_distance_theorem_l2814_281461

theorem car_distance_theorem (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 56 →
  ∃ (distance : ℝ),
    distance = new_speed * (3/2 * initial_time) ∧
    distance = 504 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l2814_281461


namespace NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l2814_281494

theorem quadratic_discriminant_nonnegative (x : ℤ) :
  x^2 * (49 - 40*x^2) ≥ 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l2814_281494


namespace NUMINAMATH_CALUDE_rectangular_field_width_l2814_281468

/-- 
Given a rectangular field where the length is 7/5 of its width and the perimeter is 360 meters,
prove that the width of the field is 75 meters.
-/
theorem rectangular_field_width (width length perimeter : ℝ) : 
  length = (7/5) * width → 
  perimeter = 2 * length + 2 * width → 
  perimeter = 360 → 
  width = 75 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l2814_281468


namespace NUMINAMATH_CALUDE_max_flights_theorem_l2814_281478

/-- Represents the number of cities with each airport type -/
structure AirportCounts where
  small : ℕ
  medium : ℕ
  big : ℕ

/-- Calculates the maximum number of flights given the airport counts -/
def max_flights (counts : AirportCounts) : ℕ :=
  counts.small * counts.medium +
  counts.small * counts.big +
  counts.medium * counts.big +
  25

/-- The theorem stating the maximum number of flights -/
theorem max_flights_theorem :
  ∃ (counts : AirportCounts),
    counts.small + counts.medium + counts.big = 28 ∧
    max_flights counts = 286 ∧
    ∀ (other_counts : AirportCounts),
      other_counts.small + other_counts.medium + other_counts.big = 28 →
      max_flights other_counts ≤ 286 :=
by
  sorry

#eval max_flights { small := 9, medium := 10, big := 9 }

end NUMINAMATH_CALUDE_max_flights_theorem_l2814_281478


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2814_281444

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 4 - y^2 / 9 = 0) ↔ (y = (3/2) * x ∨ y = -(3/2) * x) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2814_281444


namespace NUMINAMATH_CALUDE_f_integer_values_l2814_281408

def f (a b : ℕ+) : ℚ :=
  (a.val^2 + a.val * b.val + b.val^2) / (a.val * b.val - 1)

theorem f_integer_values (a b : ℕ+) (h : a.val * b.val ≠ 1) :
  ∃ (n : ℤ), n ∈ ({4, 7} : Set ℤ) ∧ f a b = n := by
  sorry

end NUMINAMATH_CALUDE_f_integer_values_l2814_281408


namespace NUMINAMATH_CALUDE_sample_size_equals_selected_students_l2814_281448

/-- Represents the sample size of a survey -/
def sample_size : ℕ := 1200

/-- Represents the number of students selected for the investigation -/
def selected_students : ℕ := 1200

/-- Theorem stating that the sample size is equal to the number of selected students -/
theorem sample_size_equals_selected_students : sample_size = selected_students := by
  sorry

end NUMINAMATH_CALUDE_sample_size_equals_selected_students_l2814_281448


namespace NUMINAMATH_CALUDE_seven_equal_parts_exist_l2814_281456

/- Define a rectangle with integer dimensions -/
structure Rectangle where
  height : ℕ
  width : ℕ

/- Define a cut as either horizontal or vertical -/
inductive Cut
| Horizontal : ℕ → Cut
| Vertical : ℕ → Cut

/- Define a division of a rectangle -/
def Division := List Cut

/- Function to check if a division results in equal parts -/
def resultsInEqualParts (r : Rectangle) (d : Division) : Prop :=
  sorry

/- Main theorem -/
theorem seven_equal_parts_exist :
  ∃ (d : Division), resultsInEqualParts (Rectangle.mk 7 1) d ∧ d.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_seven_equal_parts_exist_l2814_281456


namespace NUMINAMATH_CALUDE_square_equality_solution_l2814_281425

theorem square_equality_solution : ∃ (N : ℕ+), (36 ^ 2 * 72 ^ 2 : ℕ) = 12 ^ 2 * N ^ 2 ∧ N = 216 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_solution_l2814_281425


namespace NUMINAMATH_CALUDE_circle_tangency_l2814_281471

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1_center : ℝ × ℝ) (c1_radius : ℝ) 
                       (c2_center : ℝ × ℝ) (c2_radius : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (c1_radius + c2_radius)^2

theorem circle_tangency (a : ℝ) (h : a > 0) :
  externally_tangent (a, 0) 2 (0, Real.sqrt 5) 3 → a = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_l2814_281471


namespace NUMINAMATH_CALUDE_probability_no_adjacent_same_rolls_l2814_281477

-- Define the number of people
def num_people : ℕ := 5

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Define the probability of adjacent people rolling different numbers
def prob_diff_adjacent : ℚ := (die_sides - 1) / die_sides

-- Define the probability of no two adjacent people rolling the same number
def prob_no_adjacent_same : ℚ := prob_diff_adjacent ^ (num_people - 1)

-- Theorem statement
theorem probability_no_adjacent_same_rolls :
  prob_no_adjacent_same = 625 / 1296 := by sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_same_rolls_l2814_281477


namespace NUMINAMATH_CALUDE_estimate_sqrt_difference_l2814_281467

theorem estimate_sqrt_difference (ε : Real) (h : ε > 0) : 
  |Real.sqrt 58 - Real.sqrt 55 - 0.20| < ε :=
sorry

end NUMINAMATH_CALUDE_estimate_sqrt_difference_l2814_281467


namespace NUMINAMATH_CALUDE_kangaroo_cant_reach_far_l2814_281440

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a valid jump for the kangaroo -/
def validJump (p q : Point) : Prop :=
  (q.x = p.x + 1 ∧ q.y = p.y - 1) ∨ (q.x = p.x - 5 ∧ q.y = p.y + 7)

/-- Defines if a point is in the first quadrant -/
def inFirstQuadrant (p : Point) : Prop :=
  p.x ≥ 0 ∧ p.y ≥ 0

/-- Defines if a point is at least 1000 units away from the origin -/
def farFromOrigin (p : Point) : Prop :=
  p.x^2 + p.y^2 ≥ 1000000

/-- Defines if a point can be reached through a sequence of valid jumps -/
def canReach (start target : Point) : Prop :=
  ∃ (n : ℕ) (path : ℕ → Point), 
    path 0 = start ∧ 
    path n = target ∧ 
    ∀ i < n, validJump (path i) (path (i+1)) ∧ inFirstQuadrant (path (i+1))

/-- The main theorem to be proved -/
theorem kangaroo_cant_reach_far (p : Point) 
  (h1 : inFirstQuadrant p) 
  (h2 : p.x + p.y ≤ 4) : 
  ¬∃ q : Point, canReach p q ∧ farFromOrigin q :=
sorry

end NUMINAMATH_CALUDE_kangaroo_cant_reach_far_l2814_281440


namespace NUMINAMATH_CALUDE_unique_multiplication_with_repeated_digit_l2814_281412

theorem unique_multiplication_with_repeated_digit :
  ∃! (a b c d e f g h i j z : ℕ),
    (0 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧
    (0 ≤ d ∧ d ≤ 9) ∧ (0 ≤ e ∧ e ≤ 9) ∧ (0 ≤ f ∧ f ≤ 9) ∧
    (0 ≤ g ∧ g ≤ 9) ∧ (0 ≤ h ∧ h ≤ 9) ∧ (0 ≤ i ∧ i ≤ 9) ∧
    (0 ≤ j ∧ j ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    (a * 1000000 + b * 100000 + z * 10000 + c * 1000 + d * 100 + e * 10 + z) *
    (f * 100000 + g * 10000 + h * 1000 + i * 100 + z * 10 + j) =
    423416204528 :=
by sorry

end NUMINAMATH_CALUDE_unique_multiplication_with_repeated_digit_l2814_281412


namespace NUMINAMATH_CALUDE_probability_two_black_balls_is_three_tenths_l2814_281480

def total_balls : ℕ := 16
def black_balls : ℕ := 9

def probability_two_black_balls : ℚ :=
  (black_balls.choose 2) / (total_balls.choose 2)

theorem probability_two_black_balls_is_three_tenths :
  probability_two_black_balls = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_black_balls_is_three_tenths_l2814_281480


namespace NUMINAMATH_CALUDE_max_time_digit_sum_l2814_281417

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours ≤ 23
  minute_valid : minutes ≤ 59

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits in a 24-hour digital watch display is 24 -/
theorem max_time_digit_sum : 
  ∃ (t : Time24), ∀ (t' : Time24), timeDigitSum t' ≤ timeDigitSum t ∧ timeDigitSum t = 24 :=
sorry

end NUMINAMATH_CALUDE_max_time_digit_sum_l2814_281417


namespace NUMINAMATH_CALUDE_sons_age_l2814_281436

theorem sons_age (man daughter son : ℕ) 
  (h1 : man = son + 30)
  (h2 : daughter = son - 8)
  (h3 : man + 2 = 3 * (son + 2))
  (h4 : man + 2 = 2 * (daughter + 2)) :
  son = 13 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l2814_281436


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2814_281420

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : a + b + c = 1) : 
  (a^2 * b^2 / ((a^2 + b*c) * (b^2 + a*c))) + 
  (a^2 * c^2 / ((a^2 + b*c) * (c^2 + a*b))) + 
  (b^2 * c^2 / ((b^2 + a*c) * (c^2 + a*b))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2814_281420


namespace NUMINAMATH_CALUDE_remainder_of_repeated_12_l2814_281495

def repeated_digit_number (n : ℕ) : ℕ := 
  -- Function to generate the number with n repetitions of "12"
  -- Implementation details omitted for brevity
  sorry

theorem remainder_of_repeated_12 (n : ℕ) :
  repeated_digit_number 150 % 99 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_repeated_12_l2814_281495


namespace NUMINAMATH_CALUDE_unique_positive_x_exists_l2814_281442

/-- Given a > b > 0, there exists a unique positive x such that 
    f(x) = ((a^(1/3) + b^(1/3)) / 2)^3, where f(x) = (2(a+b)x + 2ab) / (4x + a + b) -/
theorem unique_positive_x_exists (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃! x : ℝ, x > 0 ∧ (2 * (a + b) * x + 2 * a * b) / (4 * x + a + b) = ((a^(1/3) + b^(1/3)) / 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_x_exists_l2814_281442


namespace NUMINAMATH_CALUDE_lcm_812_3214_l2814_281406

theorem lcm_812_3214 : Nat.lcm 812 3214 = 1304124 := by
  sorry

end NUMINAMATH_CALUDE_lcm_812_3214_l2814_281406


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_l2814_281413

theorem smallest_four_digit_multiple : ∃ (n : ℕ), 
  (n = 1119) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < n → ¬(((m + 1) % 5 = 0) ∧ ((m + 1) % 7 = 0) ∧ ((m + 1) % 8 = 0))) ∧
  ((n + 1) % 5 = 0) ∧ ((n + 1) % 7 = 0) ∧ ((n + 1) % 8 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_l2814_281413


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_3_and_8_l2814_281498

theorem smallest_four_digit_divisible_by_3_and_8 :
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n < 10000) ∧ 
    n % 3 = 0 ∧ 
    n % 8 = 0 ∧
    (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) → m % 3 = 0 → m % 8 = 0 → n ≤ m) ∧
    n = 1008 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_3_and_8_l2814_281498


namespace NUMINAMATH_CALUDE_interest_period_calculation_l2814_281405

theorem interest_period_calculation 
  (initial_amount : ℝ) 
  (rate_A : ℝ) 
  (rate_B : ℝ) 
  (gain_B : ℝ) 
  (h1 : initial_amount = 2800)
  (h2 : rate_A = 0.15)
  (h3 : rate_B = 0.185)
  (h4 : gain_B = 294) :
  ∃ t : ℝ, t = 3 ∧ initial_amount * (rate_B - rate_A) * t = gain_B :=
sorry

end NUMINAMATH_CALUDE_interest_period_calculation_l2814_281405


namespace NUMINAMATH_CALUDE_family_egg_count_l2814_281458

/-- Calculates the final number of eggs a family has after using some and chickens laying new ones. -/
def finalEggCount (initialEggs usedEggs chickens eggsPerChicken : ℕ) : ℕ :=
  initialEggs - usedEggs + chickens * eggsPerChicken

/-- Proves that for the given scenario, the family ends up with 11 eggs. -/
theorem family_egg_count : finalEggCount 10 5 2 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_family_egg_count_l2814_281458


namespace NUMINAMATH_CALUDE_evelyns_remaining_bottle_caps_l2814_281429

/-- The number of bottle caps Evelyn has left after losing some -/
def bottle_caps_left (initial : ℝ) (lost : ℝ) : ℝ := initial - lost

/-- Theorem: Evelyn's remaining bottle caps -/
theorem evelyns_remaining_bottle_caps :
  bottle_caps_left 63.75 18.36 = 45.39 := by
  sorry

end NUMINAMATH_CALUDE_evelyns_remaining_bottle_caps_l2814_281429


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2814_281457

/-- Given a line with equation y - 3 = -3(x - 5), 
    the sum of its x-intercept and y-intercept is 24 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 3 = -3 * (x - 5)) → 
  ∃ (x_int y_int : ℝ), 
    (y_int - 3 = -3 * (x_int - 5)) ∧ 
    (0 - 3 = -3 * (x_int - 5)) ∧ 
    (y_int - 3 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 24) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2814_281457


namespace NUMINAMATH_CALUDE_min_river_width_for_race_l2814_281416

/-- The width of a river that can accommodate a boat race -/
def river_width (num_boats : ℕ) (boat_width : ℕ) (space_between : ℕ) : ℕ :=
  num_boats * boat_width + (num_boats - 1) * space_between + 2 * space_between

/-- Theorem stating the minimum width of the river for the given conditions -/
theorem min_river_width_for_race : river_width 8 3 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_river_width_for_race_l2814_281416


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l2814_281407

theorem tan_sum_simplification : 
  Real.tan (π / 8) + Real.tan (5 * π / 24) = 
    2 * Real.sin (13 * π / 24) / Real.sqrt ((2 + Real.sqrt 2) * (2 + Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l2814_281407


namespace NUMINAMATH_CALUDE_weight_2019_is_9_5_l2814_281421

/-- The weight of a single stick in kilograms -/
def stick_weight : ℝ := 0.5

/-- The number of sticks used to form each digit -/
def sticks_per_digit : Fin 10 → ℕ
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0  -- We only care about digits 0, 1, 2, and 9 for this problem

/-- The weight of the number 2019 in kilograms -/
def weight_2019 : ℝ :=
  (sticks_per_digit 2 + sticks_per_digit 0 + sticks_per_digit 1 + sticks_per_digit 9) * stick_weight

/-- The theorem stating that the weight of 2019 is 9.5 kg -/
theorem weight_2019_is_9_5 : weight_2019 = 9.5 := by
  sorry

#eval weight_2019

end NUMINAMATH_CALUDE_weight_2019_is_9_5_l2814_281421


namespace NUMINAMATH_CALUDE_rectangle_area_l2814_281499

/-- Proves that a rectangle with width to height ratio of 7:5 and perimeter 48 cm has an area of 140 cm² -/
theorem rectangle_area (width height : ℝ) : 
  width / height = 7 / 5 →
  2 * (width + height) = 48 →
  width * height = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2814_281499


namespace NUMINAMATH_CALUDE_f_properties_l2814_281472

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x) + Real.log (1 + x) + x^4 - 2*x^2

theorem f_properties :
  (∀ x, f x ≠ 0 → -1 < x ∧ x < 1) ∧
  (∀ x, f (-x) = f x) ∧
  (∀ y, (∃ x, f x = y) → y ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2814_281472


namespace NUMINAMATH_CALUDE_ac_plus_one_lt_a_plus_c_l2814_281446

theorem ac_plus_one_lt_a_plus_c (a c : ℝ) (ha : 0 < a ∧ a < 1) (hc : c > 1) :
  a * c + 1 < a + c := by
  sorry

end NUMINAMATH_CALUDE_ac_plus_one_lt_a_plus_c_l2814_281446


namespace NUMINAMATH_CALUDE_min_filtration_processes_correct_l2814_281403

/-- The reduction rate of impurities for each filtration process -/
def reduction_rate : ℝ := 0.20

/-- The target percentage of impurities after filtration -/
def target_percentage : ℝ := 0.05

/-- Approximation of log₂ -/
def log2_approx : ℝ := 0.3010

/-- The minimum number of filtration processes required -/
def min_filtration_processes : ℕ := 14

/-- Theorem stating the minimum number of filtration processes required -/
theorem min_filtration_processes_correct :
  ∀ n : ℕ,
  (1 - reduction_rate) ^ n < target_percentage →
  n ≥ min_filtration_processes :=
sorry

end NUMINAMATH_CALUDE_min_filtration_processes_correct_l2814_281403


namespace NUMINAMATH_CALUDE_a_decreasing_l2814_281419

open BigOperators

def a (n : ℕ) : ℚ := ∑ k in Finset.range n, 1 / (k * (n + 1 - k))

theorem a_decreasing (n : ℕ) (h : n ≥ 2) : a (n + 1) < a n := by
  sorry

end NUMINAMATH_CALUDE_a_decreasing_l2814_281419


namespace NUMINAMATH_CALUDE_three_number_sum_l2814_281462

theorem three_number_sum : ∀ a b c : ℝ,
  a ≤ b → b ≤ c →
  b = 7 →
  (a + b + c) / 3 = a + 8 →
  (a + b + c) / 3 = c - 20 →
  a + b + c = 57 := by
sorry

end NUMINAMATH_CALUDE_three_number_sum_l2814_281462


namespace NUMINAMATH_CALUDE_cos_is_omega_2_on_0_1_sin_omega_t_characterization_sin_sum_range_for_omega_functions_l2814_281411

/-- Definition of Ω(t) function -/
def is_omega_t_function (f : ℝ → ℝ) (t a b : ℝ) : Prop :=
  a < b ∧ t > 0 ∧
  ((∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y)) ∧
  ((∀ x y, a + t ≤ x ∧ x < y ∧ y ≤ b + t → f x ≤ f y) ∨ (∀ x y, a + t ≤ x ∧ x < y ∧ y ≤ b + t → f x ≥ f y))

/-- Theorem: cos x is an Ω(2) function on [0,1] -/
theorem cos_is_omega_2_on_0_1 : is_omega_t_function Real.cos 2 0 1 := by sorry

/-- Theorem: Characterization of t for sin x to be an Ω(t) function on [-π/2, π/2] -/
theorem sin_omega_t_characterization (t : ℝ) : 
  is_omega_t_function Real.sin t (-π/2) (π/2) ↔ ∃ k : ℤ, t = 2 * π * k ∧ k > 0 := by sorry

/-- Theorem: Range of sin α + sin β for Ω functions -/
theorem sin_sum_range_for_omega_functions (α β : ℝ) :
  (∃ a B, is_omega_t_function Real.sin β a (α + B) ∧ is_omega_t_function Real.sin α B (α + β)) →
  (0 < Real.sin α + Real.sin β ∧ Real.sin α + Real.sin β ≤ 1) ∨ Real.sin α + Real.sin β = 2 := by sorry

end NUMINAMATH_CALUDE_cos_is_omega_2_on_0_1_sin_omega_t_characterization_sin_sum_range_for_omega_functions_l2814_281411


namespace NUMINAMATH_CALUDE_nell_remaining_cards_l2814_281459

/-- Proves that Nell has 276 cards after giving away 28 cards from her initial 304 cards. -/
theorem nell_remaining_cards (initial_cards : ℕ) (cards_given : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 304 → cards_given = 28 → remaining_cards = initial_cards - cards_given → remaining_cards = 276 := by
  sorry

end NUMINAMATH_CALUDE_nell_remaining_cards_l2814_281459


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l2814_281454

-- Define set A
def A : Set ℝ := {x | |x - 1| ≤ 1}

-- Define set B with parameter a
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem for part 1
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for part 2
theorem subset_condition (a : ℝ) :
  A ⊆ B a ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l2814_281454


namespace NUMINAMATH_CALUDE_complex_circle_theorem_l2814_281424

def complex_circle_problem (a₁ a₂ a₃ a₄ a₅ : ℂ) (s : ℝ) : Prop :=
  (a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0) ∧
  (a₂ / a₁ = a₃ / a₂) ∧ (a₃ / a₂ = a₄ / a₃) ∧ (a₄ / a₃ = a₅ / a₄) ∧
  (a₁ + a₂ + a₃ + a₄ + a₅ = 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅)) ∧
  (a₁ + a₂ + a₃ + a₄ + a₅ = s) ∧
  (Complex.abs s ≤ 2) →
  Complex.abs a₁ = 2 ∧ Complex.abs a₂ = 2 ∧ Complex.abs a₃ = 2 ∧ Complex.abs a₄ = 2 ∧ Complex.abs a₅ = 2

theorem complex_circle_theorem (a₁ a₂ a₃ a₄ a₅ : ℂ) (s : ℝ) :
  complex_circle_problem a₁ a₂ a₃ a₄ a₅ s := by
  sorry

end NUMINAMATH_CALUDE_complex_circle_theorem_l2814_281424


namespace NUMINAMATH_CALUDE_sarahs_flour_purchase_l2814_281483

/-- Sarah's flour purchase problem -/
theorem sarahs_flour_purchase
  (rye : ℝ)
  (chickpea : ℝ)
  (pastry : ℝ)
  (total : ℝ)
  (h_rye : rye = 5)
  (h_chickpea : chickpea = 3)
  (h_pastry : pastry = 2)
  (h_total : total = 20)
  : total - (rye + chickpea + pastry) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_flour_purchase_l2814_281483


namespace NUMINAMATH_CALUDE_amy_earnings_l2814_281465

theorem amy_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (tips : ℝ) :
  hourly_wage = 2 → hours_worked = 7 → tips = 9 →
  hourly_wage * hours_worked + tips = 23 := by
  sorry

end NUMINAMATH_CALUDE_amy_earnings_l2814_281465


namespace NUMINAMATH_CALUDE_interior_diagonal_sum_for_specific_box_l2814_281488

/-- Represents a rectangular box with given surface area and edge length sum -/
structure RectangularBox where
  surface_area : ℝ
  edge_length_sum : ℝ

/-- Calculates the sum of lengths of all interior diagonals of a rectangular box -/
def interior_diagonal_sum (box : RectangularBox) : ℝ :=
  sorry

/-- Theorem: For a rectangular box with surface area 130 and edge length sum 56,
    the sum of interior diagonal lengths is 4√66 -/
theorem interior_diagonal_sum_for_specific_box :
  let box : RectangularBox := { surface_area := 130, edge_length_sum := 56 }
  interior_diagonal_sum box = 4 * Real.sqrt 66 := by
  sorry

end NUMINAMATH_CALUDE_interior_diagonal_sum_for_specific_box_l2814_281488


namespace NUMINAMATH_CALUDE_combined_mean_of_three_sets_l2814_281443

theorem combined_mean_of_three_sets (set1_count : ℕ) (set1_mean : ℚ)
                                    (set2_count : ℕ) (set2_mean : ℚ)
                                    (set3_count : ℕ) (set3_mean : ℚ) :
  set1_count = 7 ∧ set1_mean = 15 ∧
  set2_count = 8 ∧ set2_mean = 20 ∧
  set3_count = 5 ∧ set3_mean = 12 →
  (set1_count * set1_mean + set2_count * set2_mean + set3_count * set3_mean) /
  (set1_count + set2_count + set3_count) = 325 / 20 := by
sorry

end NUMINAMATH_CALUDE_combined_mean_of_three_sets_l2814_281443


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2814_281432

theorem container_volume_ratio (volume_first volume_second : ℚ) : 
  volume_first > 0 →
  volume_second > 0 →
  (4 / 5 : ℚ) * volume_first = (2 / 3 : ℚ) * volume_second →
  volume_first / volume_second = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2814_281432


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2814_281438

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 8 * a 9 * a 10 = -a 13 ^ 2) →
  (a 8 * a 9 * a 10 = -1000) →
  a 10 * a 12 = 100 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2814_281438


namespace NUMINAMATH_CALUDE_chess_player_win_loss_difference_l2814_281447

theorem chess_player_win_loss_difference
  (total_games : ℕ)
  (total_points : ℚ)
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)
  (h1 : total_games = 40)
  (h2 : total_points = 25)
  (h3 : wins + draws + losses = total_games)
  (h4 : wins + (1/2 : ℚ) * draws = total_points) :
  wins - losses = 10 := by
  sorry

end NUMINAMATH_CALUDE_chess_player_win_loss_difference_l2814_281447


namespace NUMINAMATH_CALUDE_series_rationality_characterization_l2814_281450

/-- Represents a sequence of coefficients for the series -/
def CoefficientSequence := ℕ → ℕ

/-- The series sum for a given coefficient sequence -/
noncomputable def SeriesSum (a : CoefficientSequence) : ℝ :=
  ∑' n, (a n : ℝ) / n.factorial

/-- Condition that all coefficients from N onwards are zero -/
def AllZeroFrom (a : CoefficientSequence) (N : ℕ) : Prop :=
  ∀ n ≥ N, a n = 0

/-- Condition that all coefficients from N onwards are n-1 -/
def AllNMinusOneFrom (a : CoefficientSequence) (N : ℕ) : Prop :=
  ∀ n ≥ N, a n = n - 1

/-- The main theorem statement -/
theorem series_rationality_characterization (a : CoefficientSequence) 
  (h : ∀ n ≥ 2, 0 ≤ a n ∧ a n ≤ n - 1) :
  (∃ q : ℚ, SeriesSum a = q) ↔ 
  (∃ N : ℕ, AllZeroFrom a N ∨ AllNMinusOneFrom a N) := by
  sorry

end NUMINAMATH_CALUDE_series_rationality_characterization_l2814_281450


namespace NUMINAMATH_CALUDE_shooting_challenge_sequences_l2814_281486

theorem shooting_challenge_sequences : ℕ := by
  -- Define the total number of targets
  let total_targets : ℕ := 10

  -- Define the number of targets in each column
  let targets_A : ℕ := 4
  let targets_B : ℕ := 4
  let targets_C : ℕ := 2

  -- Assert that the sum of targets in all columns equals the total targets
  have h1 : targets_A + targets_B + targets_C = total_targets := by sorry

  -- Define the number of different sequences
  let num_sequences : ℕ := (Nat.factorial total_targets) / 
    ((Nat.factorial targets_A) * (Nat.factorial targets_B) * (Nat.factorial targets_C))

  -- Prove that the number of sequences equals 3150
  have h2 : num_sequences = 3150 := by sorry

  -- Return the result
  exact 3150

end NUMINAMATH_CALUDE_shooting_challenge_sequences_l2814_281486


namespace NUMINAMATH_CALUDE_sum_interior_angles_convex_polygon_l2814_281409

theorem sum_interior_angles_convex_polygon (n : ℕ) (h : n = 10) :
  (∃ (v : ℕ), v + 3 = n ∧ v = 7) →
  (n - 2) * 180 = 1440 :=
sorry

end NUMINAMATH_CALUDE_sum_interior_angles_convex_polygon_l2814_281409


namespace NUMINAMATH_CALUDE_log_sum_equals_three_l2814_281426

theorem log_sum_equals_three : Real.log 50 + Real.log 20 = 3 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_l2814_281426


namespace NUMINAMATH_CALUDE_inverse_sum_product_l2814_281473

theorem inverse_sum_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h_sum : 3*x + y/3 ≠ 0) : 
  (3*x + y/3)⁻¹ * ((3*x)⁻¹ + (y/3)⁻¹) = (x*y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_product_l2814_281473


namespace NUMINAMATH_CALUDE_complex_product_modulus_l2814_281455

theorem complex_product_modulus (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = t - 3 * Complex.I →
  t = 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_modulus_l2814_281455


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2814_281479

theorem cubic_equation_solution : 
  ∀ x y : ℕ+, 
  (x : ℝ)^3 + (y : ℝ)^3 = 4 * ((x : ℝ)^2 * (y : ℝ) + (x : ℝ) * (y : ℝ)^2 - 5) → 
  ((x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 1)) := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2814_281479


namespace NUMINAMATH_CALUDE_coin_fraction_missing_l2814_281492

theorem coin_fraction_missing (x : ℚ) : x > 0 → 
  let lost := x / 2
  let found := (3 / 4) * lost
  x - (x - lost + found) = x / 8 := by
sorry

end NUMINAMATH_CALUDE_coin_fraction_missing_l2814_281492


namespace NUMINAMATH_CALUDE_inscribed_square_diagonal_l2814_281439

theorem inscribed_square_diagonal (length width : ℝ) (h1 : length = 8) (h2 : width = 6) :
  let inscribed_square_side := width
  let inscribed_square_area := inscribed_square_side ^ 2
  let third_square_area := 9 * inscribed_square_area
  let third_square_side := Real.sqrt third_square_area
  let third_square_diagonal := third_square_side * Real.sqrt 2
  third_square_diagonal = 18 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_diagonal_l2814_281439


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l2814_281469

def is_prime (n : ℕ) : Prop := sorry

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k, k ≥ start ∧ k < start + count → ¬ is_prime k

theorem smallest_prime_after_seven_nonprimes :
  ∃ n : ℕ, 
    (consecutive_nonprimes n 7) ∧ 
    (is_prime (n + 7)) ∧
    (∀ m : ℕ, m < n → ¬(consecutive_nonprimes m 7 ∧ is_prime (m + 7))) ∧
    (n + 7 = 97) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l2814_281469


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2814_281437

theorem sum_of_fractions : 
  (1 / (1 * 2 : ℚ)) + (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + 
  (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2814_281437


namespace NUMINAMATH_CALUDE_town_population_male_count_l2814_281482

theorem town_population_male_count (total_population : ℕ) (num_groups : ℕ) (male_groups : ℕ) : 
  total_population = 480 →
  num_groups = 4 →
  male_groups = 2 →
  (total_population / num_groups) * male_groups = 240 := by
sorry

end NUMINAMATH_CALUDE_town_population_male_count_l2814_281482


namespace NUMINAMATH_CALUDE_composition_equality_l2814_281487

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 4
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem composition_equality : f (g (f 3)) = 108 := by
  sorry

end NUMINAMATH_CALUDE_composition_equality_l2814_281487


namespace NUMINAMATH_CALUDE_smallest_distance_to_2i_l2814_281434

theorem smallest_distance_to_2i (z : ℂ) (h : Complex.abs (z^2 + 3 + Complex.I) = Complex.abs (z * (z + 1 + 3 * Complex.I))) :
  Complex.abs (z - 2 * Complex.I) ≥ (1 : ℝ) / 2 ∧
  ∃ w : ℂ, Complex.abs (w^2 + 3 + Complex.I) = Complex.abs (w * (w + 1 + 3 * Complex.I)) ∧
           Complex.abs (w - 2 * Complex.I) = (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_to_2i_l2814_281434


namespace NUMINAMATH_CALUDE_apple_price_36kg_l2814_281435

/-- The price of apples with a two-tier pricing system -/
def apple_price (l q : ℚ) (kg : ℚ) : ℚ :=
  if kg ≤ 30 then l * kg
  else l * 30 + q * (kg - 30)

theorem apple_price_36kg (l q : ℚ) :
  (apple_price l q 33 = 360) →
  (apple_price l q 25 = 250) →
  (apple_price l q 36 = 420) :=
by sorry

end NUMINAMATH_CALUDE_apple_price_36kg_l2814_281435


namespace NUMINAMATH_CALUDE_book_problem_solution_l2814_281464

/-- Represents the cost and quantity relationships between two types of books -/
structure BookProblem where
  cost_diff : ℕ             -- Cost difference between type B and type A
  total_cost_A : ℕ          -- Total cost for type A books
  total_cost_B : ℕ          -- Total cost for type B books
  total_books : ℕ           -- Total number of books to purchase
  max_total_cost : ℕ        -- Maximum total cost allowed

/-- Calculates the cost of type A books given the problem parameters -/
def cost_A (p : BookProblem) : ℕ :=
  p.total_cost_A * p.total_cost_B / (p.total_cost_B - p.total_cost_A * p.cost_diff)

/-- Calculates the cost of type B books given the problem parameters -/
def cost_B (p : BookProblem) : ℕ :=
  cost_A p + p.cost_diff

/-- Calculates the minimum number of type A books to purchase -/
def min_books_A (p : BookProblem) : ℕ :=
  (p.total_books * cost_B p - p.max_total_cost) / (cost_B p - cost_A p)

/-- Theorem stating the solution to the book purchasing problem -/
theorem book_problem_solution (p : BookProblem) 
  (h : p = { cost_diff := 20, total_cost_A := 540, total_cost_B := 780, 
             total_books := 70, max_total_cost := 3550 }) : 
  cost_A p = 45 ∧ cost_B p = 65 ∧ min_books_A p = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_problem_solution_l2814_281464


namespace NUMINAMATH_CALUDE_lcm_problem_l2814_281484

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 16) (h2 : a * b = 2560) :
  Nat.lcm a b = 160 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2814_281484


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2814_281496

/-- The solution set of the inequality |x - 1| < kx contains exactly three integers -/
def has_three_integer_solutions (k : ℝ) : Prop :=
  ∃ (a b c : ℤ), a < b ∧ b < c ∧
  (∀ x : ℝ, |x - 1| < k * x ↔ (x > a ∧ x < c)) ∧
  (∀ n : ℤ, |n - 1| < k * n ↔ (n = a + 1 ∨ n = b ∨ n = c - 1))

/-- The main theorem -/
theorem inequality_solution_range (k : ℝ) :
  has_three_integer_solutions k → k ∈ Set.Ioo (2/3) (3/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2814_281496


namespace NUMINAMATH_CALUDE_linear_function_condition_passes_through_origin_l2814_281404

/-- A linear function of x with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := (2*m + 1)*x + m - 3

theorem linear_function_condition (m : ℝ) :
  (∀ x, ∃ y, f m x = y) ↔ m ≠ -1/2 :=
sorry

theorem passes_through_origin (m : ℝ) :
  f m 0 = 0 ↔ m = 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_condition_passes_through_origin_l2814_281404
