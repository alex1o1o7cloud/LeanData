import Mathlib

namespace NUMINAMATH_CALUDE_david_recreation_spending_l1585_158567

theorem david_recreation_spending :
  ∀ (last_week_wages : ℝ) (last_week_percent : ℝ),
    last_week_percent > 0 →
    (0.7 * last_week_wages * 0.2) = (0.7 * (last_week_percent / 100) * last_week_wages) →
    last_week_percent = 20 := by
  sorry

end NUMINAMATH_CALUDE_david_recreation_spending_l1585_158567


namespace NUMINAMATH_CALUDE_distribute_books_correct_l1585_158508

/-- The number of ways to distribute 6 different books among three people. -/
def distribute_books : Nat × Nat × Nat × Nat :=
  let n : Nat := 6  -- Total number of books
  let k : Nat := 3  -- Number of people

  -- Scenario 1: One person gets 1 book, another gets 2 books, the last gets 3 books
  let scenario1 : Nat := k.factorial * n.choose 1 * (n - 1).choose 2 * (n - 3).choose 3

  -- Scenario 2: Books are evenly distributed, each person getting 2 books
  let scenario2 : Nat := (n.choose 2 * (n - 2).choose 2 * (n - 4).choose 2) / k.factorial

  -- Scenario 3: One part gets 4 books, other two parts get 1 book each
  let scenario3 : Nat := n.choose 4

  -- Scenario 4: A gets 1 book, B gets 1 book, C gets 4 books
  let scenario4 : Nat := n.choose 4 * (n - 4).choose 1

  (scenario1, scenario2, scenario3, scenario4)

theorem distribute_books_correct :
  distribute_books = (360, 90, 15, 30) := by
  sorry

end NUMINAMATH_CALUDE_distribute_books_correct_l1585_158508


namespace NUMINAMATH_CALUDE_swimming_pool_width_l1585_158548

/-- Proves that the width of a rectangular swimming pool is 20 feet -/
theorem swimming_pool_width :
  ∀ (length width : ℝ) (water_removed : ℝ) (depth_lowered : ℝ),
    length = 60 →
    water_removed = 4500 →
    depth_lowered = 0.5 →
    water_removed / 7.5 = length * width * depth_lowered →
    width = 20 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_width_l1585_158548


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l1585_158530

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_even_digit_multiple_of_9_under_1000 :
  ∃ (n : ℕ), n = 360 ∧
    n < 1000 ∧
    has_only_even_digits n ∧
    n % 9 = 0 ∧
    ∀ (m : ℕ), m < 1000 → has_only_even_digits m → m % 9 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_under_1000_l1585_158530


namespace NUMINAMATH_CALUDE_soap_brand_survey_l1585_158551

theorem soap_brand_survey (total : ℕ) (only_A : ℕ) (both : ℕ) (only_B_ratio : ℕ) :
  total = 260 →
  only_A = 60 →
  both = 30 →
  only_B_ratio = 3 →
  total - (only_A + both + only_B_ratio * both) = 80 :=
by sorry

end NUMINAMATH_CALUDE_soap_brand_survey_l1585_158551


namespace NUMINAMATH_CALUDE_curve_classification_l1585_158513

-- Define the curve equation
def curve_equation (x y m : ℝ) : Prop :=
  x^2 / (5 - m) + y^2 / (2 - m) = 1

-- Define the condition for m
def m_condition (m : ℝ) : Prop := m < 3

-- Define the result for different ranges of m
def curve_type (m : ℝ) : Prop :=
  (m < 2 → ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ∀ x y : ℝ, curve_equation x y m ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  (2 < m → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    ∀ x y : ℝ, curve_equation x y m ↔ (x^2 / a^2 - y^2 / b^2 = 1))

-- Theorem statement
theorem curve_classification (m : ℝ) :
  m_condition m → curve_type m :=
sorry

end NUMINAMATH_CALUDE_curve_classification_l1585_158513


namespace NUMINAMATH_CALUDE_player_B_most_consistent_l1585_158552

/-- Represents a player in the rope skipping test -/
inductive Player : Type
  | A : Player
  | B : Player
  | C : Player
  | D : Player

/-- Returns the variance of a player's performance -/
def variance (p : Player) : ℝ :=
  match p with
  | Player.A => 0.023
  | Player.B => 0.018
  | Player.C => 0.020
  | Player.D => 0.021

/-- States that Player B has the most consistent performance -/
theorem player_B_most_consistent :
  ∀ p : Player, p ≠ Player.B → variance Player.B < variance p :=
by sorry

end NUMINAMATH_CALUDE_player_B_most_consistent_l1585_158552


namespace NUMINAMATH_CALUDE_a_in_second_quadrant_l1585_158515

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the second quadrant of a rectangular coordinate system -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point A with coordinates dependent on x -/
def A (x : ℝ) : Point :=
  { x := 6 - 2*x, y := x - 5 }

/-- Theorem stating the condition for point A to be in the second quadrant -/
theorem a_in_second_quadrant :
  ∀ x : ℝ, SecondQuadrant (A x) ↔ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_a_in_second_quadrant_l1585_158515


namespace NUMINAMATH_CALUDE_modulus_of_z_l1585_158525

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1585_158525


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1585_158519

theorem simplify_and_evaluate (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  5 * (3 * x^2 * y - x * y^2) - (x * y^2 + 3 * x^2 * y) = 36 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1585_158519


namespace NUMINAMATH_CALUDE_brian_watching_time_l1585_158501

/-- The total time Brian spends watching animal videos -/
def total_watching_time (cat_video_length : ℕ) : ℕ :=
  let dog_video_length := 2 * cat_video_length
  let gorilla_video_length := 2 * (cat_video_length + dog_video_length)
  cat_video_length + dog_video_length + gorilla_video_length

/-- Theorem stating that Brian spends 36 minutes watching animal videos -/
theorem brian_watching_time : total_watching_time 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_brian_watching_time_l1585_158501


namespace NUMINAMATH_CALUDE_cow_spots_problem_l1585_158522

theorem cow_spots_problem (left_spots right_spots total_spots additional_spots : ℕ) :
  left_spots = 16 →
  right_spots = 3 * left_spots + additional_spots →
  total_spots = left_spots + right_spots →
  total_spots = 71 →
  additional_spots = 7 := by
  sorry

end NUMINAMATH_CALUDE_cow_spots_problem_l1585_158522


namespace NUMINAMATH_CALUDE_second_project_length_l1585_158538

/-- Represents a digging project with depth, length, and breadth measurements. -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of a digging project. -/
def volume (project : DiggingProject) : ℝ :=
  project.depth * project.length * project.breadth

theorem second_project_length : 
  ∀ (project1 project2 : DiggingProject),
  project1.depth = 100 →
  project1.length = 25 →
  project1.breadth = 30 →
  project2.depth = 75 →
  project2.breadth = 50 →
  volume project1 = volume project2 →
  project2.length = 20 := by
  sorry

#check second_project_length

end NUMINAMATH_CALUDE_second_project_length_l1585_158538


namespace NUMINAMATH_CALUDE_jenny_reading_speed_l1585_158526

/-- Represents Jenny's reading challenge --/
structure ReadingChallenge where
  days : ℕ
  books : ℕ
  book1_words : ℕ
  book2_words : ℕ
  book3_words : ℕ
  reading_minutes_per_day : ℕ

/-- Calculates the reading speed in words per hour --/
def calculate_reading_speed (challenge : ReadingChallenge) : ℕ :=
  let total_words := challenge.book1_words + challenge.book2_words + challenge.book3_words
  let words_per_day := total_words / challenge.days
  let reading_hours_per_day := challenge.reading_minutes_per_day / 60
  words_per_day / reading_hours_per_day

/-- Jenny's specific reading challenge --/
def jenny_challenge : ReadingChallenge :=
  { days := 10
  , books := 3
  , book1_words := 200
  , book2_words := 400
  , book3_words := 300
  , reading_minutes_per_day := 54
  }

/-- Theorem stating that Jenny's reading speed is 100 words per hour --/
theorem jenny_reading_speed :
  calculate_reading_speed jenny_challenge = 100 := by
  sorry

end NUMINAMATH_CALUDE_jenny_reading_speed_l1585_158526


namespace NUMINAMATH_CALUDE_evaluate_expression_l1585_158505

theorem evaluate_expression : (1 / ((-5^3)^4)) * ((-5)^15) * (5^2) = -3125 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1585_158505


namespace NUMINAMATH_CALUDE_system_solution_exists_iff_l1585_158580

theorem system_solution_exists_iff (k : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - y = 4 ∧ k * x^2 + y = 5) ↔ k > -1/36 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_iff_l1585_158580


namespace NUMINAMATH_CALUDE_inequality_proof_l1585_158527

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (h : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1585_158527


namespace NUMINAMATH_CALUDE_line_intersection_plane_parallel_l1585_158585

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the intersection of lines
variable (intersect : Line → Line → Prop)

-- Define parallel planes
variable (parallel : Plane → Plane → Prop)

-- Define the statement
theorem line_intersection_plane_parallel 
  (l m : Line) (α β : Plane) 
  (h1 : subset l α) (h2 : subset m β) :
  (¬ intersect l m → parallel α β) ∧ 
  ¬ (¬ intersect l m → parallel α β) ∧ 
  (parallel α β → ¬ intersect l m) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_plane_parallel_l1585_158585


namespace NUMINAMATH_CALUDE_gcd_of_60_and_75_l1585_158581

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_60_and_75_l1585_158581


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1585_158573

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  1 / x + 1 / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1585_158573


namespace NUMINAMATH_CALUDE_smallest_n_for_pie_distribution_l1585_158516

theorem smallest_n_for_pie_distribution (N : ℕ) : 
  N > 70 → (21 * N) % 70 = 0 → (∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N) → N = 80 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_pie_distribution_l1585_158516


namespace NUMINAMATH_CALUDE_sum_of_abc_l1585_158597

theorem sum_of_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l1585_158597


namespace NUMINAMATH_CALUDE_negative_among_expressions_l1585_158589

theorem negative_among_expressions : 
  (|(-3)| > 0) ∧ (-(-3) > 0) ∧ ((-3)^2 > 0) ∧ (-Real.sqrt 3 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_among_expressions_l1585_158589


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l1585_158568

theorem quadratic_form_ratio (a d : ℝ) : 
  (∀ x, x^2 + 500*x + 2500 = (x + a)^2 + d) →
  d / a = -240 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l1585_158568


namespace NUMINAMATH_CALUDE_area_of_region_l1585_158571

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 17 ∧ 
   A = Real.pi * (Real.sqrt ((x + 1)^2 + (y - 2)^2))^2 ∧
   x^2 + y^2 + 2*x - 4*y = 12) :=
by sorry

end NUMINAMATH_CALUDE_area_of_region_l1585_158571


namespace NUMINAMATH_CALUDE_sin_20_sqrt3_plus_tan_50_equals_one_l1585_158534

theorem sin_20_sqrt3_plus_tan_50_equals_one :
  Real.sin (20 * π / 180) * (Real.sqrt 3 + Real.tan (50 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_20_sqrt3_plus_tan_50_equals_one_l1585_158534


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1585_158509

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a+1, a^2+3}

-- State the theorem
theorem intersection_implies_a_value :
  ∀ a : ℝ, (A a ∩ B a = {-3}) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1585_158509


namespace NUMINAMATH_CALUDE_equation_roots_l1585_158504

theorem equation_roots : 
  ∀ x : ℝ, (21 / (x^2 - 9) - 3 / (x - 3) = 1) ↔ (x = 3 ∨ x = -7) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l1585_158504


namespace NUMINAMATH_CALUDE_coefficient_x_sqrt_x_in_expansion_l1585_158590

theorem coefficient_x_sqrt_x_in_expansion :
  let expansion := (λ x : ℝ => (Real.sqrt x - 1)^5)
  ∃ c : ℝ, ∀ x : ℝ, x > 0 →
    expansion x = c * x * Real.sqrt x + (λ y => y - c * x * Real.sqrt x) (expansion x) ∧
    c = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_sqrt_x_in_expansion_l1585_158590


namespace NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l1585_158543

/-- The number of chocolate bars in the large box -/
def total_chocolate_bars (num_small_boxes : ℕ) (bars_per_small_box : ℕ) : ℕ :=
  num_small_boxes * bars_per_small_box

/-- Theorem stating the total number of chocolate bars in the large box -/
theorem chocolate_bars_in_large_box :
  total_chocolate_bars 20 32 = 640 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l1585_158543


namespace NUMINAMATH_CALUDE_certain_number_problem_l1585_158523

theorem certain_number_problem : ∃ x : ℚ, (1206 / 3 : ℚ) = 3 * x ∧ x = 134 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1585_158523


namespace NUMINAMATH_CALUDE_no_polynomial_exists_l1585_158593

theorem no_polynomial_exists : ¬∃ (P : ℝ → ℝ → ℝ), 
  (∀ x y, x ∈ ({1, 2, 3} : Set ℝ) → y ∈ ({1, 2, 3} : Set ℝ) → 
    P x y ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 10} : Set ℝ)) ∧ 
  (∀ v, v ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 10} : Set ℝ) → 
    ∃! (x y : ℝ), x ∈ ({1, 2, 3} : Set ℝ) ∧ y ∈ ({1, 2, 3} : Set ℝ) ∧ P x y = v) ∧
  (∃ (a b c d e f : ℝ), ∀ x y, P x y = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f) :=
by sorry

end NUMINAMATH_CALUDE_no_polynomial_exists_l1585_158593


namespace NUMINAMATH_CALUDE_first_group_size_l1585_158544

def work_rate (people : ℕ) (time : ℕ) : ℚ := 1 / (people * time)

theorem first_group_size :
  ∀ (p : ℕ),
  (work_rate p 60 = work_rate 16 30) →
  p = 8 := by
sorry

end NUMINAMATH_CALUDE_first_group_size_l1585_158544


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1585_158536

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1585_158536


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1585_158564

-- Problem 1
theorem problem_1 : (-81) - (1/4 : ℚ) + (-7) - (3/4 : ℚ) - (-22) = -67 := by sorry

-- Problem 2
theorem problem_2 : -(4^2) / ((-2)^3) - 2^2 * (-1/2 : ℚ) = 4 := by sorry

-- Problem 3
theorem problem_3 : -(1^2023) - 24 * ((1/2 : ℚ) - 2/3 + 3/8) = -6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1585_158564


namespace NUMINAMATH_CALUDE_total_tickets_sold_l1585_158500

theorem total_tickets_sold (child_cost adult_cost total_revenue child_count : ℕ) 
  (h1 : child_cost = 6)
  (h2 : adult_cost = 9)
  (h3 : total_revenue = 1875)
  (h4 : child_count = 50) :
  child_count + (total_revenue - child_cost * child_count) / adult_cost = 225 :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l1585_158500


namespace NUMINAMATH_CALUDE_rectangle_ratio_golden_ratio_l1585_158553

/-- Given a unit square AEFD, prove that if the ratio of length to width of rectangle ABCD
    equals the ratio of length to width of rectangle BCFE, then the length of AB (W) is (1 + √5) / 2. -/
theorem rectangle_ratio_golden_ratio (W : ℝ) : 
  W > 0 ∧ W / 1 = 1 / (W - 1) → W = (1 + Real.sqrt 5) / 2 := by
  sorry

#check rectangle_ratio_golden_ratio

end NUMINAMATH_CALUDE_rectangle_ratio_golden_ratio_l1585_158553


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l1585_158560

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (large : Rectangle) (small : Rectangle) :
  large.width = 12 ∧ 
  large.height = 12 ∧ 
  small.width = 6 ∧ 
  small.height = 4 ∧ 
  large.area - small.area = 144 →
  Rectangle.perimeter { width := large.width - small.width, height := large.height } +
  Rectangle.perimeter { width := large.width, height := large.height - small.height } = 28 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l1585_158560


namespace NUMINAMATH_CALUDE_parabola_vertex_l1585_158563

/-- The vertex of the parabola y = x^2 - 2x + 4 has coordinates (1, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = x^2 - 2*x + 4 → (∃ (h k : ℝ), h = 1 ∧ k = 3 ∧ 
    ∀ (x' : ℝ), x'^2 - 2*x' + 4 ≥ k ∧ 
    (x'^2 - 2*x' + 4 = k ↔ x' = h)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1585_158563


namespace NUMINAMATH_CALUDE_butterfat_mixture_proof_l1585_158584

-- Define the initial quantities and percentages
def initial_volume : ℝ := 8
def initial_butterfat_percentage : ℝ := 0.35
def added_butterfat_percentage : ℝ := 0.10
def target_butterfat_percentage : ℝ := 0.20

-- Define the volume of milk to be added
def added_volume : ℝ := 12

-- Theorem statement
theorem butterfat_mixture_proof :
  let total_volume := initial_volume + added_volume
  let total_butterfat := initial_volume * initial_butterfat_percentage + added_volume * added_butterfat_percentage
  total_butterfat / total_volume = target_butterfat_percentage := by
sorry


end NUMINAMATH_CALUDE_butterfat_mixture_proof_l1585_158584


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1585_158566

/-- The expansion of (1 - 1/(2x))^6 in terms of 1/x -/
def expansion (x : ℝ) (a : Fin 7 → ℝ) : Prop :=
  (1 - 1/(2*x))^6 = a 0 + a 1 * (1/x) + a 2 * (1/x)^2 + a 3 * (1/x)^3 + 
                    a 4 * (1/x)^4 + a 5 * (1/x)^5 + a 6 * (1/x)^6

/-- The sum of the coefficients a_3 and a_4 is equal to -25/16 -/
theorem sum_of_coefficients (x : ℝ) (a : Fin 7 → ℝ) 
  (h : expansion x a) : a 3 + a 4 = -25/16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1585_158566


namespace NUMINAMATH_CALUDE_pie_eaten_after_seven_trips_l1585_158565

def eat_pie (n : ℕ) : ℚ :=
  1 - (2/3)^n

theorem pie_eaten_after_seven_trips :
  eat_pie 7 = 1093 / 2187 :=
by sorry

end NUMINAMATH_CALUDE_pie_eaten_after_seven_trips_l1585_158565


namespace NUMINAMATH_CALUDE_cosine_function_parameters_l1585_158532

/-- Proves that for y = a cos(bx), if max is 3 at x=0 and first zero at x=π/6, then a=3 and b=3 -/
theorem cosine_function_parameters (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x, a * Real.cos (b * x) ≤ 3) → 
  a * Real.cos 0 = 3 → 
  a * Real.cos (b * (π / 6)) = 0 → 
  a = 3 ∧ b = 3 := by
  sorry


end NUMINAMATH_CALUDE_cosine_function_parameters_l1585_158532


namespace NUMINAMATH_CALUDE_triple_percent_40_l1585_158517

/-- The operation % defined on real numbers -/
def percent (M : ℝ) : ℝ := 0.4 * M + 2

/-- Theorem stating that applying the percent operation three times to 40 results in 5.68 -/
theorem triple_percent_40 : percent (percent (percent 40)) = 5.68 := by
  sorry

end NUMINAMATH_CALUDE_triple_percent_40_l1585_158517


namespace NUMINAMATH_CALUDE_ball_selection_limit_l1585_158583

open Real

/-- The probability of selecting n₁ white balls and n₂ black balls without replacement from an urn -/
noncomputable def P (M₁ M₂ n₁ n₂ : ℕ) : ℝ :=
  (Nat.choose M₁ n₁ * Nat.choose M₂ n₂ : ℝ) / Nat.choose (M₁ + M₂) (n₁ + n₂)

/-- The limit of the probability as M and M₁ approach infinity -/
theorem ball_selection_limit (n₁ n₂ : ℕ) (p : ℝ) (h_p : 0 < p ∧ p < 1) :
  ∀ ε > 0, ∃ N : ℕ, ∀ M₁ M₂ : ℕ,
    M₁ ≥ N → M₂ ≥ N →
    |P M₁ M₂ n₁ n₂ - (Nat.choose (n₁ + n₂) n₁ : ℝ) * p^n₁ * (1 - p)^n₂| < ε :=
by sorry

end NUMINAMATH_CALUDE_ball_selection_limit_l1585_158583


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l1585_158521

/-- The number of x-intercepts of the parabola x = -3y^2 + 2y + 2 -/
theorem parabola_x_intercepts :
  let f : ℝ → ℝ := λ y => -3 * y^2 + 2 * y + 2
  ∃! x : ℝ, ∃ y : ℝ, f y = x ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l1585_158521


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1585_158503

theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = Real.sqrt 8) :
  let side := (4 * Real.sqrt 6) / 3
  let area := (1 / 2) * side * h
  area = 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1585_158503


namespace NUMINAMATH_CALUDE_rhombus_diagonal_theorem_l1585_158591

/-- Represents a rhombus with given properties -/
structure Rhombus where
  diagonal1 : ℝ
  perimeter : ℝ
  diagonal2 : ℝ

/-- Theorem stating the relationship between the diagonals and perimeter of a rhombus -/
theorem rhombus_diagonal_theorem (r : Rhombus) (h1 : r.diagonal1 = 24) (h2 : r.perimeter = 52) :
  r.diagonal2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_theorem_l1585_158591


namespace NUMINAMATH_CALUDE_total_covered_area_is_72_l1585_158546

/-- Represents a rectangular strip with length and width -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular strip -/
def Strip.area (s : Strip) : ℝ := s.length * s.width

/-- Calculates the area of overlap between two perpendicular strips -/
def overlap_area (s : Strip) : ℝ := s.width * s.width

/-- The setup of the problem with four strips and overlaps -/
structure StripSetup where
  strips : Fin 4 → Strip
  num_overlaps : ℕ

/-- Theorem: The total area covered by four strips with given dimensions and overlaps is 72 -/
theorem total_covered_area_is_72 (setup : StripSetup) 
  (h1 : ∀ i, (setup.strips i).length = 12)
  (h2 : ∀ i, (setup.strips i).width = 2)
  (h3 : setup.num_overlaps = 6) :
  (Finset.sum Finset.univ (λ i => (setup.strips i).area)) - 
  (setup.num_overlaps : ℝ) * overlap_area (setup.strips 0) = 72 := by
  sorry


end NUMINAMATH_CALUDE_total_covered_area_is_72_l1585_158546


namespace NUMINAMATH_CALUDE_approximation_accuracy_l1585_158575

theorem approximation_accuracy : 
  abs (84 * Real.sqrt 7 - 222 * (2 / (2 + 7))) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_approximation_accuracy_l1585_158575


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1585_158535

theorem arithmetic_sequence_common_difference
  (a₁ : ℚ)    -- first term
  (aₙ : ℚ)    -- last term
  (S  : ℚ)    -- sum of all terms
  (h₁ : a₁ = 3)
  (h₂ : aₙ = 34)
  (h₃ : S = 222) :
  ∃ (n : ℕ) (d : ℚ), n > 1 ∧ d = 31/11 ∧ 
    aₙ = a₁ + (n - 1) * d ∧
    S = n * (a₁ + aₙ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1585_158535


namespace NUMINAMATH_CALUDE_journey_speed_l1585_158562

/-- Proves the required speed for the second part of a journey given the total distance, total time, initial speed, and initial time. -/
theorem journey_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (h1 : total_distance = 24) 
  (h2 : total_time = 8) 
  (h3 : initial_speed = 4) 
  (h4 : initial_time = 4) 
  : 
  (total_distance - initial_speed * initial_time) / (total_time - initial_time) = 2 := by
  sorry

#check journey_speed

end NUMINAMATH_CALUDE_journey_speed_l1585_158562


namespace NUMINAMATH_CALUDE_factorial_square_root_theorem_l1585_158550

theorem factorial_square_root_theorem : 
  (Real.sqrt ((Nat.factorial 5 * Nat.factorial 4) / Nat.factorial 3))^2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_theorem_l1585_158550


namespace NUMINAMATH_CALUDE_value_of_S_l1585_158518

theorem value_of_S : let S : ℝ := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 1 / (Real.sqrt 12 - 3)
  S = 7 := by sorry

end NUMINAMATH_CALUDE_value_of_S_l1585_158518


namespace NUMINAMATH_CALUDE_vector_triangle_inequality_l1585_158545

/-- Given two vectors AB and AC in a Euclidean space, with |AB| = 3 and |AC| = 6,
    prove that the magnitude of BC is between 3 and 9 inclusive. -/
theorem vector_triangle_inequality (A B C : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖B - A‖ = 3) (h2 : ‖C - A‖ = 6) : 
  3 ≤ ‖C - B‖ ∧ ‖C - B‖ ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_vector_triangle_inequality_l1585_158545


namespace NUMINAMATH_CALUDE_trig_expression_equality_l1585_158555

theorem trig_expression_equality : 
  (Real.sin (110 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (155 * π / 180)^2 - Real.sin (155 * π / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l1585_158555


namespace NUMINAMATH_CALUDE_sum_of_w_and_y_l1585_158587

theorem sum_of_w_and_y (W X Y Z : ℤ) : 
  W ∈ ({1, 2, 5, 6} : Set ℤ) →
  X ∈ ({1, 2, 5, 6} : Set ℤ) →
  Y ∈ ({1, 2, 5, 6} : Set ℤ) →
  Z ∈ ({1, 2, 5, 6} : Set ℤ) →
  W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
  (W : ℚ) / X + (Y : ℚ) / Z = 3 →
  W + Y = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_w_and_y_l1585_158587


namespace NUMINAMATH_CALUDE_smallest_n_value_l1585_158588

theorem smallest_n_value (r g b : ℕ) (n : ℕ) : 
  (∃ m : ℕ, m > 0 ∧ 10 * r = m ∧ 18 * g = m ∧ 24 * b = m ∧ 25 * n = m) →
  n ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1585_158588


namespace NUMINAMATH_CALUDE_composite_shape_sum_l1585_158547

/-- Represents a 3D geometric shape with faces, edges, and vertices -/
structure Shape where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- The initial triangular prism -/
def triangularPrism : Shape := ⟨5, 9, 6⟩

/-- Attaches a regular pentagonal prism to a quadrilateral face of the given shape -/
def attachPentagonalPrism (s : Shape) : Shape :=
  ⟨s.faces - 1 + 7, s.edges + 10, s.vertices + 5⟩

/-- Adds a pyramid to a pentagonal face of the given shape -/
def addPyramid (s : Shape) : Shape :=
  ⟨s.faces - 1 + 5, s.edges + 5, s.vertices + 1⟩

/-- Calculates the sum of faces, edges, and vertices of a shape -/
def sumFeatures (s : Shape) : ℕ :=
  s.faces + s.edges + s.vertices

/-- Theorem stating that the sum of features of the final composite shape is 51 -/
theorem composite_shape_sum :
  sumFeatures (addPyramid (attachPentagonalPrism triangularPrism)) = 51 := by
  sorry

end NUMINAMATH_CALUDE_composite_shape_sum_l1585_158547


namespace NUMINAMATH_CALUDE_total_vehicles_l1585_158599

theorem total_vehicles (motorcycles bicycles : ℕ) 
  (h1 : motorcycles = 2) 
  (h2 : bicycles = 5) : 
  motorcycles + bicycles = 7 :=
by sorry

end NUMINAMATH_CALUDE_total_vehicles_l1585_158599


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1585_158557

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 20) :
  let side_length := perimeter / 4
  let area := side_length * side_length
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l1585_158557


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l1585_158595

/-- Given two parabolas that form a kite when intersecting the coordinate axes, 
    prove that the sum of their coefficients is 1/50 if the kite area is 20 -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    (a * x₁^2 + 3 = 0 ∧ 5 - b * x₁^2 = 0) ∧ 
    (a * x₂^2 + 3 = 0 ∧ 5 - b * x₂^2 = 0) ∧
    (y₁ = a * 0^2 + 3 ∧ y₂ = 5 - b * 0^2) ∧
    (1/2 * (x₂ - x₁) * (y₂ - y₁) = 20)) →
  a + b = 1/50 := by
sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l1585_158595


namespace NUMINAMATH_CALUDE_survey_b_count_l1585_158554

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => (firstSample + i * (populationSize / sampleSize)) % populationSize + 1)

/-- Count elements in a list that fall within a given range -/
def countInRange (list : List ℕ) (lower upper : ℕ) : ℕ :=
  list.filter (fun x => lower ≤ x ∧ x ≤ upper) |>.length

theorem survey_b_count :
  let populationSize := 480
  let sampleSize := 16
  let firstSample := 8
  let surveyBLower := 161
  let surveyBUpper := 320
  let sampledNumbers := systematicSample populationSize sampleSize firstSample
  countInRange sampledNumbers surveyBLower surveyBUpper = 5 := by
  sorry


end NUMINAMATH_CALUDE_survey_b_count_l1585_158554


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l1585_158507

/-- Represents the decimal expansion of a rational number -/
def DecimalExpansion (q : ℚ) : List ℕ := sorry

/-- Checks if a list contains three consecutive identical elements -/
def hasThreeConsecutiveIdentical (l : List ℕ) : Prop := sorry

/-- Checks if a list is entirely composed of identical elements -/
def isEntirelyIdentical (l : List ℕ) : Prop := sorry

/-- Checks if a natural number satisfies the given conditions -/
def satisfiesConditions (n : ℕ) : Prop :=
  let expansion := DecimalExpansion (1 / n)
  hasThreeConsecutiveIdentical expansion ∧ ¬isEntirelyIdentical expansion

theorem smallest_satisfying_number :
  satisfiesConditions 157 ∧ ∀ m < 157, ¬satisfiesConditions m := by sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l1585_158507


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1585_158582

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) * (2 - Complex.I)
  (z.re = 0) → m = -2 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1585_158582


namespace NUMINAMATH_CALUDE_permutation_square_sum_bounds_l1585_158561

def is_permutation (a : Fin 10 → ℕ) : Prop :=
  ∀ i : Fin 10, ∃ j : Fin 10, a j = i.val + 1

theorem permutation_square_sum_bounds 
  (a b : Fin 10 → ℕ) 
  (ha : is_permutation a) 
  (hb : is_permutation b) :
  (∃ k : Fin 10, a k ^ 2 + b k ^ 2 ≥ 101) ∧
  (∃ k : Fin 10, a k ^ 2 + b k ^ 2 ≤ 61) :=
sorry

end NUMINAMATH_CALUDE_permutation_square_sum_bounds_l1585_158561


namespace NUMINAMATH_CALUDE_cid_oil_changes_l1585_158514

/-- Represents the mechanic shop's pricing and services -/
structure MechanicShop where
  oil_change_price : ℕ
  repair_price : ℕ
  car_wash_price : ℕ
  repaired_cars : ℕ
  washed_cars : ℕ
  total_earnings : ℕ

/-- Calculates the number of oil changes given the shop's data -/
def calculate_oil_changes (shop : MechanicShop) : ℕ :=
  (shop.total_earnings - shop.repair_price * shop.repaired_cars - shop.car_wash_price * shop.washed_cars) / shop.oil_change_price

/-- Theorem stating that Cid changed the oil for 5 cars -/
theorem cid_oil_changes :
  let shop : MechanicShop := {
    oil_change_price := 20,
    repair_price := 30,
    car_wash_price := 5,
    repaired_cars := 10,
    washed_cars := 15,
    total_earnings := 475
  }
  calculate_oil_changes shop = 5 := by sorry

end NUMINAMATH_CALUDE_cid_oil_changes_l1585_158514


namespace NUMINAMATH_CALUDE_factorization_of_4_minus_4x_squared_l1585_158531

theorem factorization_of_4_minus_4x_squared (x : ℝ) : 4 - 4*x^2 = 4*(1+x)*(1-x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4_minus_4x_squared_l1585_158531


namespace NUMINAMATH_CALUDE_circplus_neg_three_eight_l1585_158559

/-- The ⊕ operation for rational numbers -/
def circplus (a b : ℚ) : ℚ := a * b + (a - b)

/-- Theorem stating that (-3) ⊕ 8 = -35 -/
theorem circplus_neg_three_eight : circplus (-3) 8 = -35 := by sorry

end NUMINAMATH_CALUDE_circplus_neg_three_eight_l1585_158559


namespace NUMINAMATH_CALUDE_kim_morning_routine_time_l1585_158533

/-- Represents the types of employees in Kim's office. -/
inductive EmployeeType
  | Senior
  | Junior
  | Intern

/-- Represents whether an employee worked overtime or not. -/
inductive OvertimeStatus
  | Overtime
  | NoOvertime

/-- Calculates the total time for Kim's morning routine. -/
def morning_routine_time (
  senior_count junior_count intern_count : Nat
  ) (
  senior_overtime junior_overtime intern_overtime : Nat
  ) : Nat :=
  let coffee_time := 5
  let status_update_time :=
    3 * senior_count + 2 * junior_count + 1 * intern_count
  let payroll_update_time :=
    4 * senior_overtime + 2 * (senior_count - senior_overtime) +
    3 * junior_overtime + 1 * (junior_count - junior_overtime) +
    2 * intern_overtime + 1 -- 1 minute for 2 interns without overtime (30 seconds each)
  let task_allocation_time :=
    4 * senior_count + 3 * junior_count + 2 * intern_count
  let additional_tasks_time := 10 + 8 + 6 + 5

  coffee_time + status_update_time + payroll_update_time +
  task_allocation_time + additional_tasks_time

/-- Theorem stating that Kim's morning routine takes 101 minutes. -/
theorem kim_morning_routine_time :
  morning_routine_time 3 3 3 2 3 1 = 101 := by
  sorry

end NUMINAMATH_CALUDE_kim_morning_routine_time_l1585_158533


namespace NUMINAMATH_CALUDE_cookie_distribution_l1585_158592

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) :
  total_cookies = 24 →
  num_people = 6 →
  cookies_per_person = total_cookies / num_people →
  cookies_per_person = 4 :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1585_158592


namespace NUMINAMATH_CALUDE_print_shop_charge_difference_l1585_158598

/-- The charge difference for color copies between two print shops -/
theorem print_shop_charge_difference 
  (price_X : ℚ) -- Price per copy at shop X
  (price_Y : ℚ) -- Price per copy at shop Y
  (num_copies : ℕ) -- Number of copies
  (h1 : price_X = 1.25) -- Shop X charges $1.25 per copy
  (h2 : price_Y = 2.75) -- Shop Y charges $2.75 per copy
  (h3 : num_copies = 60) -- We're considering 60 copies
  : (price_Y - price_X) * num_copies = 90 := by
  sorry

#check print_shop_charge_difference

end NUMINAMATH_CALUDE_print_shop_charge_difference_l1585_158598


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1585_158594

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 3) = 9 → x = 84 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1585_158594


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l1585_158524

/-- A six-digit number starting with 1 -/
def SixDigitNumber (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 1

/-- Function to move the first digit of a six-digit number to the last position -/
def MoveFirstToLast (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

/-- The main theorem -/
theorem unique_six_digit_number : 
  ∀ n : ℕ, SixDigitNumber n → (MoveFirstToLast n = 3 * n) → n = 142857 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l1585_158524


namespace NUMINAMATH_CALUDE_c_work_time_l1585_158579

-- Define the work rates of a, b, and c
variable (A B C : ℝ)

-- Define the conditions
def condition1 : Prop := A + B = 1 / 6
def condition2 : Prop := B + C = 1 / 8
def condition3 : Prop := C + A = 1 / 12

-- Theorem statement
theorem c_work_time (h1 : condition1 A B) (h2 : condition2 B C) (h3 : condition3 C A) :
  1 / C = 48 := by sorry

end NUMINAMATH_CALUDE_c_work_time_l1585_158579


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l1585_158510

/-- Given a parabola passing through points (-2,0) and (4,0), 
    its axis of symmetry is the line x = 1 -/
theorem parabola_axis_of_symmetry :
  ∀ (f : ℝ → ℝ),
  (f (-2) = 0) →
  (f 4 = 0) →
  (∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c) →
  (∀ x, f (1 + x) = f (1 - x)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l1585_158510


namespace NUMINAMATH_CALUDE_sum_of_specific_S_l1585_158506

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

theorem sum_of_specific_S : S 17 + S 33 + S 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_l1585_158506


namespace NUMINAMATH_CALUDE_pencil_distribution_l1585_158569

theorem pencil_distribution (initial_pencils : ℕ) (containers : ℕ) (additional_pencils : ℕ)
  (h1 : initial_pencils = 150)
  (h2 : containers = 5)
  (h3 : additional_pencils = 30) :
  (initial_pencils + additional_pencils) / containers = 36 :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1585_158569


namespace NUMINAMATH_CALUDE_problem_solution_l1585_158558

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) : x^7 - 6*x^5 + 5*x^3 - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1585_158558


namespace NUMINAMATH_CALUDE_complex_division_result_l1585_158541

theorem complex_division_result : (1 - Complex.I) / (1 + Complex.I) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l1585_158541


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1585_158577

theorem simple_interest_rate_calculation (total_amount interest_difference amount_B : ℚ)
  (h1 : total_amount = 10000)
  (h2 : interest_difference = 360)
  (h3 : amount_B = 4000) :
  let amount_A := total_amount - amount_B
  let rate_A := 15 / 100
  let time := 2
  let interest_A := amount_A * rate_A * time
  let interest_B := interest_A - interest_difference
  let rate_B := interest_B / (amount_B * time)
  rate_B = 18 / 100 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1585_158577


namespace NUMINAMATH_CALUDE_arithmetic_parabola_common_point_l1585_158512

/-- Represents a parabola with coefficients forming an arithmetic progression -/
structure ArithmeticParabola where
  a : ℝ
  d : ℝ

/-- The equation of the parabola given by y = ax^2 + bx + c where b = a + d and c = a + 2d -/
def ArithmeticParabola.equation (p : ArithmeticParabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + (p.a + p.d) * x + (p.a + 2 * p.d)

/-- Theorem stating that all arithmetic parabolas pass through the point (-2, 0) -/
theorem arithmetic_parabola_common_point (p : ArithmeticParabola) :
  p.equation (-2) 0 := by sorry

end NUMINAMATH_CALUDE_arithmetic_parabola_common_point_l1585_158512


namespace NUMINAMATH_CALUDE_root_of_equations_l1585_158572

theorem root_of_equations (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + p = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + p * m + q = 0) :
  m^5 = q / p ∧ (p = q → ∃ k : Fin 5, m = Complex.exp (2 * Real.pi * I * (k : ℝ) / 5)) :=
sorry

end NUMINAMATH_CALUDE_root_of_equations_l1585_158572


namespace NUMINAMATH_CALUDE_adult_ticket_cost_adult_ticket_cost_proof_l1585_158528

/-- The cost of an adult ticket for a play, given the following conditions:
  * Child tickets cost 1 dollar
  * 22 people attended the performance
  * Total ticket sales were 50 dollars
  * 18 children attended the play
-/
theorem adult_ticket_cost : ℝ → Prop :=
  fun adult_cost =>
    let child_cost : ℝ := 1
    let total_attendance : ℕ := 22
    let total_sales : ℝ := 50
    let children_attendance : ℕ := 18
    let adult_attendance : ℕ := total_attendance - children_attendance
    adult_cost * adult_attendance + child_cost * children_attendance = total_sales ∧
    adult_cost = 8

/-- Proof of the adult ticket cost theorem -/
theorem adult_ticket_cost_proof : ∃ (cost : ℝ), adult_ticket_cost cost := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_adult_ticket_cost_proof_l1585_158528


namespace NUMINAMATH_CALUDE_circle_condition_l1585_158539

/-- A circle in the xy-plane can be represented by the equation (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius. -/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ h k r, r > 0 ∧ ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The equation of the form x^2 + y^2 + dx + ey + f = 0 -/
def general_quadratic (d e f : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + d*x + e*y + f

theorem circle_condition (m : ℝ) :
  is_circle (general_quadratic (-2) (-4) m) → m < 5 := by
  sorry


end NUMINAMATH_CALUDE_circle_condition_l1585_158539


namespace NUMINAMATH_CALUDE_no_square_base_l1585_158574

theorem no_square_base (b : ℕ) (h : b > 0) : ¬∃ (n : ℕ), b^2 + 3*b + 2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_square_base_l1585_158574


namespace NUMINAMATH_CALUDE_integer_1200_in_column_B_l1585_158529

/-- The column type representing the six columns A, B, C, D, E, F --/
inductive Column
| A | B | C | D | E | F

/-- The function that maps a positive integer to its corresponding column in the zigzag pattern --/
def columnFor (n : ℕ) : Column :=
  match (n - 1) % 10 with
  | 0 => Column.A
  | 1 => Column.B
  | 2 => Column.C
  | 3 => Column.D
  | 4 => Column.E
  | 5 => Column.F
  | 6 => Column.E
  | 7 => Column.D
  | 8 => Column.C
  | _ => Column.B

/-- Theorem stating that the integer 1200 will be placed in column B --/
theorem integer_1200_in_column_B : columnFor 1200 = Column.B := by
  sorry

end NUMINAMATH_CALUDE_integer_1200_in_column_B_l1585_158529


namespace NUMINAMATH_CALUDE_prob_heart_spade_king_two_draws_l1585_158556

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ := 52)
  (target_cards : ℕ := 28)

/-- Calculates the probability of drawing at least one target card in two draws with replacement -/
def prob_at_least_one_target (d : Deck) : ℚ :=
  1 - (1 - d.target_cards / d.total_cards) ^ 2

/-- The probability of drawing at least one heart, spade, or king in two draws with replacement from a standard deck is 133/169 -/
theorem prob_heart_spade_king_two_draws :
  prob_at_least_one_target (Deck.mk 52 28) = 133 / 169 := by
  sorry

#eval prob_at_least_one_target (Deck.mk 52 28)

end NUMINAMATH_CALUDE_prob_heart_spade_king_two_draws_l1585_158556


namespace NUMINAMATH_CALUDE_distance_between_signs_l1585_158540

theorem distance_between_signs 
  (total_distance : ℕ) 
  (distance_to_first_sign : ℕ) 
  (distance_after_second_sign : ℕ) 
  (h1 : total_distance = 1000)
  (h2 : distance_to_first_sign = 350)
  (h3 : distance_after_second_sign = 275) :
  total_distance - distance_to_first_sign - distance_after_second_sign = 375 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_signs_l1585_158540


namespace NUMINAMATH_CALUDE_equal_roots_condition_l1585_158586

theorem equal_roots_condition (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 2*x - (m^2 + 2)) / ((x^2 - 2)*(m - 2)) = x / m) → 
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l1585_158586


namespace NUMINAMATH_CALUDE_emerald_density_conversion_l1585_158570

/-- Density of a material in g/cm³ -/
def density : ℝ := 2.7

/-- Conversion factor from grams to carats -/
def gramsToCarat : ℝ := 5

/-- Conversion factor from cubic centimeters to cubic inches -/
def cmCubedToInchCubed : ℝ := 16.387

/-- Density of emerald in carats per cubic inch -/
def emeraldDensityCaratsPerCubicInch : ℝ :=
  density * gramsToCarat * cmCubedToInchCubed

theorem emerald_density_conversion :
  ⌊emeraldDensityCaratsPerCubicInch⌋ = 221 := by sorry

end NUMINAMATH_CALUDE_emerald_density_conversion_l1585_158570


namespace NUMINAMATH_CALUDE_squido_oysters_l1585_158520

theorem squido_oysters (squido crabby : ℕ) : 
  crabby ≥ 2 * squido →
  squido + crabby = 600 →
  squido = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_squido_oysters_l1585_158520


namespace NUMINAMATH_CALUDE_complement_of_complement_l1585_158596

def V : Finset Nat := {1, 2, 3, 4, 5}

def C_VN : Finset Nat := {2, 4}

def N : Finset Nat := {1, 3, 5}

theorem complement_of_complement (V C_VN N : Finset Nat) 
  (hV : V = {1, 2, 3, 4, 5})
  (hC_VN : C_VN = {2, 4})
  (hN : N = {1, 3, 5}) :
  N = V \ C_VN :=
by sorry

end NUMINAMATH_CALUDE_complement_of_complement_l1585_158596


namespace NUMINAMATH_CALUDE_village_population_l1585_158537

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.15) = 2553 → P = 3162 :=
by sorry

end NUMINAMATH_CALUDE_village_population_l1585_158537


namespace NUMINAMATH_CALUDE_function_equation_solution_l1585_158502

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_equation_solution :
  (∀ x : ℝ, 2 * f (x - 1) - 3 * f (1 - x) = 5 * x) →
  (∀ x : ℝ, f x = x - 5) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l1585_158502


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1585_158576

theorem max_value_of_expression (x y z : ℝ) 
  (h : 2 * x^2 + y^2 + z^2 = 2 * x - 4 * y + 2 * x * z - 5) : 
  ∃ (M : ℝ), M = 4 ∧ ∀ (a b c : ℝ), 2 * a^2 + b^2 + c^2 = 2 * a - 4 * b + 2 * a * c - 5 → 
  a - b + c ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1585_158576


namespace NUMINAMATH_CALUDE_root_sum_of_coefficients_l1585_158578

theorem root_sum_of_coefficients (a b : ℝ) : 
  (Complex.I * 2 + 1) ^ 2 + a * (Complex.I * 2 + 1) + b = 0 → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_of_coefficients_l1585_158578


namespace NUMINAMATH_CALUDE_lcm_gcd_equality_l1585_158549

/-- For positive integers a, b, c, prove that 
    [a,b,c]^2 / ([a,b][b,c][c,a]) = (a,b,c)^2 / ((a,b)(b,c)(c,a)) -/
theorem lcm_gcd_equality (a b c : ℕ+) : 
  (Nat.lcm (Nat.lcm a b) c)^2 / (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a) = 
  (Nat.gcd (Nat.gcd a b) c)^2 / (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_equality_l1585_158549


namespace NUMINAMATH_CALUDE_farm_hens_count_l1585_158542

/-- Represents the number of animals on a farm. -/
structure FarmAnimals where
  hens : ℕ
  cows : ℕ
  goats : ℕ

/-- Calculates the total number of heads for all animals on the farm. -/
def totalHeads (farm : FarmAnimals) : ℕ :=
  farm.hens + farm.cows + farm.goats

/-- Calculates the total number of feet for all animals on the farm. -/
def totalFeet (farm : FarmAnimals) : ℕ :=
  2 * farm.hens + 4 * farm.cows + 4 * farm.goats

/-- Theorem stating that given the conditions, there are 66 hens on the farm. -/
theorem farm_hens_count (farm : FarmAnimals) 
  (head_count : totalHeads farm = 120) 
  (feet_count : totalFeet farm = 348) : 
  farm.hens = 66 := by
  sorry

end NUMINAMATH_CALUDE_farm_hens_count_l1585_158542


namespace NUMINAMATH_CALUDE_f_strictly_increasing_when_a_eq_one_f_increasing_intervals_l1585_158511

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x

-- Theorem for part (I)
theorem f_strictly_increasing_when_a_eq_one :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f 1 x₁ < f 1 x₂ := by sorry

-- Theorem for part (II)
theorem f_increasing_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (0 < a → a < 1 → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < a → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f a x₁ < f a x₂)) ∧
  (a = 1 → ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (1 < a → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂ : ℝ, a < x₁ → x₁ < x₂ → f a x₁ < f a x₂)) := by sorry

end

end NUMINAMATH_CALUDE_f_strictly_increasing_when_a_eq_one_f_increasing_intervals_l1585_158511
