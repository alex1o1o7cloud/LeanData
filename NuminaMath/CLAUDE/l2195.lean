import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l2195_219531

theorem simplify_expression (a c d x y z : ℝ) (h : cx + dz ≠ 0) :
  (c*x*(a^3*x^3 + 3*a^3*y^3 + c^3*z^3) + d*z*(a^3*x^3 + 3*c^3*x^3 + c^3*z^3)) / (c*x + d*z) =
  a^3*x^3 + c^3*z^3 + (3*c*x*a^3*y^3)/(c*x + d*z) + (3*d*z*c^3*x^3)/(c*x + d*z) := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l2195_219531


namespace NUMINAMATH_CALUDE_production_decrease_l2195_219557

theorem production_decrease (x : ℝ) : 
  (1 - x / 100) * (1 - x / 100) = 0.49 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_production_decrease_l2195_219557


namespace NUMINAMATH_CALUDE_product_units_digit_base_6_l2195_219523

-- Define the base-10 numbers
def a : ℕ := 217
def b : ℕ := 45

-- Define the base of the target representation
def base : ℕ := 6

-- Theorem statement
theorem product_units_digit_base_6 :
  (a * b) % base = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_units_digit_base_6_l2195_219523


namespace NUMINAMATH_CALUDE_prime_square_product_equality_l2195_219520

theorem prime_square_product_equality (p : ℕ) (x y : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2)
  (h_x_range : x ∈ Finset.range ((p - 1) / 2 + 1) \ {0})
  (h_y_range : y ∈ Finset.range ((p - 1) / 2 + 1) \ {0})
  (h_square : ∃ k : ℕ, x * (p - x) * y * (p - y) = k^2) :
  x = y := by
sorry

end NUMINAMATH_CALUDE_prime_square_product_equality_l2195_219520


namespace NUMINAMATH_CALUDE_opposite_signs_sum_and_max_difference_l2195_219519

theorem opposite_signs_sum_and_max_difference (m n : ℤ) : 
  (|m| = 1 ∧ |n| = 4) → 
  ((m > 0 ∧ n < 0) ∨ (m < 0 ∧ n > 0) → (m + n = -3 ∨ m + n = 3)) ∧
  (∀ (a b : ℤ), |a| = 1 ∧ |b| = 4 → m - n ≥ a - b) :=
by sorry

end NUMINAMATH_CALUDE_opposite_signs_sum_and_max_difference_l2195_219519


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2195_219578

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 4*m^2 + 3*m = 8*n^3 + 12*n^2 + 6*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2195_219578


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l2195_219518

def line (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 1

def opposite_sides (a : ℝ) : Prop :=
  line a 1 1 > 0 ∧ line a 0 (-2) < 0

def range_of_a : Set ℝ := { a | a < -2 ∨ a > 1/2 }

theorem range_of_a_theorem :
  ∀ a : ℝ, opposite_sides a ↔ a ∈ range_of_a := by sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l2195_219518


namespace NUMINAMATH_CALUDE_min_toothpicks_to_remove_for_given_figure_l2195_219535

/-- Represents a figure made of toothpicks -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  triangles : ℕ
  squares : ℕ

/-- The minimum number of toothpicks to remove to eliminate all shapes -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ := sorry

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_to_remove_for_given_figure :
  ∃ (figure : ToothpickFigure),
    figure.total_toothpicks = 40 ∧
    figure.triangles > 20 ∧
    figure.squares = 10 ∧
    min_toothpicks_to_remove figure = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_toothpicks_to_remove_for_given_figure_l2195_219535


namespace NUMINAMATH_CALUDE_thermometer_to_bottle_ratio_l2195_219574

/-- Proves that the ratio of thermometers sold to hot-water bottles sold is 7:1 given the problem conditions -/
theorem thermometer_to_bottle_ratio :
  ∀ (T H : ℕ), 
    (2 * T + 6 * H = 1200) →  -- Total sales equation
    (H = 60) →                -- Number of hot-water bottles sold
    (T : ℚ) / H = 7 / 1 :=    -- Ratio of thermometers to hot-water bottles
by
  sorry

#check thermometer_to_bottle_ratio

end NUMINAMATH_CALUDE_thermometer_to_bottle_ratio_l2195_219574


namespace NUMINAMATH_CALUDE_candy_boxes_l2195_219541

theorem candy_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (h1 : pieces_per_box = 500) (h2 : total_pieces = 3000) :
  total_pieces / pieces_per_box = 6 := by
sorry

end NUMINAMATH_CALUDE_candy_boxes_l2195_219541


namespace NUMINAMATH_CALUDE_max_distance_to_line_l2195_219559

/-- Given a line ax + by + c = 0 where a, b, and c form an arithmetic sequence,
    the maximum distance from the origin (0, 0) to this line is √5. -/
theorem max_distance_to_line (a b c : ℝ) :
  (a + c = 2 * b) →  -- arithmetic sequence condition
  (∃ (x y : ℝ), a * x + b * y + c = 0) →  -- line exists
  (∀ (x y : ℝ), a * x + b * y + c = 0 → (x^2 + y^2 : ℝ) ≤ 5) ∧
  (∃ (x y : ℝ), a * x + b * y + c = 0 ∧ x^2 + y^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l2195_219559


namespace NUMINAMATH_CALUDE_find_a_plus_c_l2195_219506

theorem find_a_plus_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 42)
  (h2 : b + d = 6)
  (h3 : b * d = 5) :
  a + c = 7 := by
sorry

end NUMINAMATH_CALUDE_find_a_plus_c_l2195_219506


namespace NUMINAMATH_CALUDE_zoo_sea_lions_l2195_219560

theorem zoo_sea_lions (sea_lions : ℕ) (penguins : ℕ) : 
  (sea_lions : ℚ) / penguins = 4 / 11 →
  penguins = sea_lions + 84 →
  sea_lions = 48 := by
sorry

end NUMINAMATH_CALUDE_zoo_sea_lions_l2195_219560


namespace NUMINAMATH_CALUDE_quadruple_sum_product_l2195_219589

theorem quadruple_sum_product : 
  ∀ (x₁ x₂ x₃ x₄ : ℝ),
  (x₁ + x₂ * x₃ * x₄ = 2 ∧
   x₂ + x₁ * x₃ * x₄ = 2 ∧
   x₃ + x₁ * x₂ * x₄ = 2 ∧
   x₄ + x₁ * x₂ * x₃ = 2) →
  ((x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = 3) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = 3 ∧ x₄ = -1) ∨
   (x₁ = -1 ∧ x₂ = 3 ∧ x₃ = -1 ∧ x₄ = -1) ∨
   (x₁ = 3 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = -1)) :=
by sorry


end NUMINAMATH_CALUDE_quadruple_sum_product_l2195_219589


namespace NUMINAMATH_CALUDE_justin_and_tim_games_l2195_219594

theorem justin_and_tim_games (total_players : ℕ) (h1 : total_players = 8) :
  Nat.choose (total_players - 2) 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_justin_and_tim_games_l2195_219594


namespace NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l2195_219549

/-- Given a rectangle with sides y and z, and a square with side x placed against
    the shorter side y, the perimeter of one of the four congruent rectangles
    formed in the remaining space is equal to 2y + 2z - 4x. -/
theorem congruent_rectangle_perimeter 
  (y z x : ℝ) 
  (h1 : y > 0) 
  (h2 : z > 0) 
  (h3 : x > 0) 
  (h4 : x < y) 
  (h5 : x < z) : 
  2*y + 2*z - 4*x = 2*((y - x) + (z - x)) := by
  sorry


end NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l2195_219549


namespace NUMINAMATH_CALUDE_mangoes_per_box_l2195_219528

/-- Given a total of 4320 mangoes distributed equally among 36 boxes,
    prove that there are 10 dozens of mangoes in each box. -/
theorem mangoes_per_box (total_mangoes : Nat) (num_boxes : Nat) 
    (h1 : total_mangoes = 4320) (h2 : num_boxes = 36) :
    (total_mangoes / (12 * num_boxes) : Nat) = 10 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_per_box_l2195_219528


namespace NUMINAMATH_CALUDE_b_completes_in_12_days_l2195_219539

/-- The number of days B takes to complete the remaining work after A works for 5 days -/
def days_B_completes_work (a_rate b_rate : ℚ) (a_days : ℕ) : ℚ :=
  (1 - a_rate * a_days) / b_rate

theorem b_completes_in_12_days :
  let a_rate : ℚ := 1 / 15
  let b_rate : ℚ := 1 / 18
  let a_days : ℕ := 5
  days_B_completes_work a_rate b_rate a_days = 12 := by
sorry

end NUMINAMATH_CALUDE_b_completes_in_12_days_l2195_219539


namespace NUMINAMATH_CALUDE_tan_quadruple_angle_l2195_219553

theorem tan_quadruple_angle (θ : Real) (h : Real.tan θ = 3) : 
  Real.tan (4 * θ) = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_quadruple_angle_l2195_219553


namespace NUMINAMATH_CALUDE_second_number_is_22_l2195_219504

theorem second_number_is_22 (x y : ℝ) (h1 : x + y = 33) (h2 : y = 2 * x) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_22_l2195_219504


namespace NUMINAMATH_CALUDE_product_pattern_l2195_219582

theorem product_pattern (n : ℕ) : (10 * n + 3) * (10 * n + 7) = 100 * n * (n + 1) + 21 := by
  sorry

end NUMINAMATH_CALUDE_product_pattern_l2195_219582


namespace NUMINAMATH_CALUDE_bluray_movies_returned_l2195_219502

/-- Represents the number of movies returned -/
def movies_returned (initial_dvd : ℕ) (initial_bluray : ℕ) (final_dvd : ℕ) (final_bluray : ℕ) : ℕ :=
  initial_bluray - final_bluray

theorem bluray_movies_returned :
  ∀ (initial_dvd initial_bluray final_dvd final_bluray : ℕ),
    initial_dvd + initial_bluray = 378 →
    initial_dvd * 4 = initial_bluray * 17 →
    final_dvd * 2 = final_bluray * 9 →
    final_dvd = initial_dvd →
    movies_returned initial_dvd initial_bluray final_dvd final_bluray = 4 := by
  sorry

end NUMINAMATH_CALUDE_bluray_movies_returned_l2195_219502


namespace NUMINAMATH_CALUDE_certain_number_value_l2195_219517

theorem certain_number_value (x y z : ℝ) 
  (h1 : 0.5 * x = y + z) 
  (h2 : x - 2 * y = 40) : 
  z = 20 := by sorry

end NUMINAMATH_CALUDE_certain_number_value_l2195_219517


namespace NUMINAMATH_CALUDE_right_triangle_area_l2195_219561

theorem right_triangle_area (h : ℝ) (α : ℝ) (A : ℝ) :
  h = 8 * Real.sqrt 2 →
  α = 45 * π / 180 →
  A = (h^2 / 4) →
  A = 32 :=
by
  sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_right_triangle_area_l2195_219561


namespace NUMINAMATH_CALUDE_lines_not_form_triangle_l2195_219507

/-- A line in the xy-plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The three lines in the problem -/
def l1 (m : ℝ) : Line := ⟨3, m, -1⟩
def l2 : Line := ⟨3, -2, -5⟩
def l3 : Line := ⟨6, 1, -5⟩

/-- Theorem stating the conditions under which the three lines cannot form a triangle -/
theorem lines_not_form_triangle (m : ℝ) : 
  (¬(∃ x y : ℝ, 3*x + m*y - 1 = 0 ∧ 3*x - 2*y - 5 = 0 ∧ 6*x + y - 5 = 0)) ↔ 
  (m = -2 ∨ m = 1/2) :=
sorry

end NUMINAMATH_CALUDE_lines_not_form_triangle_l2195_219507


namespace NUMINAMATH_CALUDE_common_root_value_l2195_219511

-- Define the polynomials
def poly1 (x C : ℝ) : ℝ := x^3 + C*x^2 + 15
def poly2 (x D : ℝ) : ℝ := x^3 + D*x + 35

-- Theorem statement
theorem common_root_value (C D : ℝ) :
  ∃ (p : ℝ), 
    (poly1 p C = 0 ∧ poly2 p D = 0) ∧ 
    (∃ (q r : ℝ), p * q * r = -15) ∧
    (∃ (s t : ℝ), p * s * t = -35) →
    p = Real.rpow 525 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_common_root_value_l2195_219511


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2195_219583

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of participants excluding the 12 lowest-scoring players
  total_points : ℕ → ℕ → ℚ  -- Function to calculate total points between two groups of players
  lowest_twelve_points : ℚ  -- Points earned by the 12 lowest-scoring players among themselves

/-- The theorem stating the total number of participants in the tournament -/
theorem chess_tournament_participants (t : ChessTournament) : 
  (t.n + 12 = 24) ∧ 
  (t.total_points t.n 12 = t.total_points t.n t.n / 2) ∧
  (t.lowest_twelve_points = 66) ∧
  (t.total_points (t.n + 12) (t.n + 12) / 2 = t.total_points t.n t.n + 2 * t.lowest_twelve_points) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2195_219583


namespace NUMINAMATH_CALUDE_committee_selection_ways_l2195_219521

theorem committee_selection_ways (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 5) :
  Nat.choose n k = 118755 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l2195_219521


namespace NUMINAMATH_CALUDE_sample_size_example_l2195_219543

/-- Represents the sample size of a survey --/
def sample_size (population : ℕ) (selected : ℕ) : ℕ := selected

/-- Theorem stating that for a population of 300 students with 50 selected, the sample size is 50 --/
theorem sample_size_example : sample_size 300 50 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_example_l2195_219543


namespace NUMINAMATH_CALUDE_sin_230_minus_sqrt3_tan_170_l2195_219586

theorem sin_230_minus_sqrt3_tan_170 : 
  Real.sin (230 * π / 180) * (1 - Real.sqrt 3 * Real.tan (170 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_230_minus_sqrt3_tan_170_l2195_219586


namespace NUMINAMATH_CALUDE_erased_odd_number_l2195_219516

/-- The sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ := n^2

/-- The sequence of odd numbers -/
def odd_sequence (n : ℕ) : ℕ := 2*n - 1

theorem erased_odd_number :
  ∃ (n : ℕ) (k : ℕ), k < n ∧ sum_odd_numbers n - odd_sequence k = 1998 →
  odd_sequence k = 27 :=
sorry

end NUMINAMATH_CALUDE_erased_odd_number_l2195_219516


namespace NUMINAMATH_CALUDE_smallest_class_size_l2195_219585

/-- Represents a class of students who took a test -/
structure TestClass where
  n : ℕ                -- number of students
  scores : Fin n → ℕ   -- scores of each student
  test_max : ℕ         -- maximum possible score on the test

/-- Conditions for our specific test class -/
def SatisfiesConditions (c : TestClass) : Prop :=
  c.test_max = 100 ∧
  (∃ (i₁ i₂ i₃ i₄ : Fin c.n), i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₃ ≠ i₄ ∧
    c.scores i₁ = 90 ∧ c.scores i₂ = 90 ∧ c.scores i₃ = 90 ∧ c.scores i₄ = 90) ∧
  (∀ i, c.scores i ≥ 70) ∧
  (Finset.sum (Finset.univ : Finset (Fin c.n)) c.scores / c.n = 80)

/-- The main theorem stating that the smallest possible class size is 8 -/
theorem smallest_class_size (c : TestClass) (h : SatisfiesConditions c) :
  c.n ≥ 8 ∧ ∃ (c' : TestClass), SatisfiesConditions c' ∧ c'.n = 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2195_219585


namespace NUMINAMATH_CALUDE_two_digit_sum_problem_l2195_219591

theorem two_digit_sum_problem :
  ∃! (x y z : ℕ), 
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    11 * x + 11 * y + 11 * z = 100 * x + 10 * y + z :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_problem_l2195_219591


namespace NUMINAMATH_CALUDE_triangle_function_properties_l2195_219576

/-- Given a triangle ABC with side lengths a, b, c, where c > a > 0 and c > b > 0,
    and a function f(x) = a^x + b^x - c^x, prove that:
    1. For all x < 1, f(x) > 0
    2. There exists x > 0 such that xa^x, b^x, c^x cannot form a triangle
    3. If ABC is obtuse, then there exists x ∈ (1, 2) such that f(x) = 0 -/
theorem triangle_function_properties (a b c : ℝ) (h1 : c > a) (h2 : a > 0) (h3 : c > b) (h4 : b > 0)
  (h5 : a + b > c) (f : ℝ → ℝ) (hf : ∀ x, f x = a^x + b^x - c^x) :
  (∀ x < 1, f x > 0) ∧
  (∃ x > 0, ¬ (xa^x + b^x > c^x ∧ xa^x + c^x > b^x ∧ b^x + c^x > xa^x)) ∧
  (a^2 + b^2 < c^2 → ∃ x ∈ Set.Ioo 1 2, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_function_properties_l2195_219576


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2195_219546

theorem parallelogram_side_length 
  (s : ℝ) 
  (side1 : ℝ) 
  (side2 : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h : side1 = 3 * s) 
  (h' : side2 = s) 
  (h'' : angle = π / 3) 
  (h''' : area = 9 * Real.sqrt 3) 
  (h'''' : area = side1 * side2 * Real.sin angle) : 
  s = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2195_219546


namespace NUMINAMATH_CALUDE_binomial_320_320_l2195_219526

theorem binomial_320_320 : Nat.choose 320 320 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_320_320_l2195_219526


namespace NUMINAMATH_CALUDE_larry_stickers_l2195_219500

/-- The number of stickers Larry loses -/
def lost_stickers : ℕ := 6

/-- The number of stickers Larry ends up with -/
def final_stickers : ℕ := 87

/-- The initial number of stickers Larry had -/
def initial_stickers : ℕ := final_stickers + lost_stickers

theorem larry_stickers : initial_stickers = 93 := by
  sorry

end NUMINAMATH_CALUDE_larry_stickers_l2195_219500


namespace NUMINAMATH_CALUDE_sum_of_possible_values_l2195_219598

theorem sum_of_possible_values (p q r ℓ : ℂ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → p ≠ q → q ≠ r → r ≠ p →
  p / (1 - q^2) = ℓ → q / (1 - r^2) = ℓ → r / (1 - p^2) = ℓ →
  ∃ (ℓ₁ ℓ₂ : ℂ), (∀ x : ℂ, x = ℓ → x = ℓ₁ ∨ x = ℓ₂) ∧ ℓ₁ + ℓ₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_possible_values_l2195_219598


namespace NUMINAMATH_CALUDE_min_value_expression_l2195_219558

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 5) : x^2 + y^2 + 2*z^2 - x^2*y^2*z ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2195_219558


namespace NUMINAMATH_CALUDE_special_line_equation_l2195_219530

/-- A line passing through (-2, 2) forming a triangle with area 1 with the coordinate axes -/
structure SpecialLine where
  /-- Slope of the line -/
  k : ℝ
  /-- The line passes through (-2, 2) -/
  passes_through : 2 = k * (-2) + 2
  /-- The area of the triangle formed with the axes is 1 -/
  triangle_area : |4 + 2/k + 2*k| = 1

/-- The equation of a SpecialLine is either x + 2y - 2 = 0 or 2x + y + 2 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.k = -1/2 ∧ ∀ x y, x + 2*y - 2 = 0 ↔ y = l.k * x + 2) ∨
  (l.k = -2 ∧ ∀ x y, 2*x + y + 2 = 0 ↔ y = l.k * x + 2) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l2195_219530


namespace NUMINAMATH_CALUDE_coin_exchange_terminates_l2195_219514

-- Define the Dwarf type
structure Dwarf where
  id : Nat
  coins : Nat
  acquaintances : List Nat

-- Define the Clan type
def Clan := List Dwarf

-- Function to represent a single day's coin exchange
def exchangeCoins (clan : Clan) : Clan :=
  sorry

-- Theorem statement
theorem coin_exchange_terminates (initialClan : Clan) :
  ∃ n : Nat, ∀ m : Nat, m ≥ n → exchangeCoins^[m] initialClan = exchangeCoins^[n] initialClan :=
sorry

end NUMINAMATH_CALUDE_coin_exchange_terminates_l2195_219514


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2195_219508

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 1 → x - 1 ≥ Real.log x) ↔ (∃ x : ℝ, x > 1 ∧ x - 1 < Real.log x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2195_219508


namespace NUMINAMATH_CALUDE_percentage_of_300_is_66_l2195_219581

theorem percentage_of_300_is_66 : 
  (66 : ℝ) / 300 * 100 = 22 := by sorry

end NUMINAMATH_CALUDE_percentage_of_300_is_66_l2195_219581


namespace NUMINAMATH_CALUDE_scallops_per_person_is_two_l2195_219565

-- Define the constants from the problem
def scallops_per_pound : ℕ := 8
def cost_per_pound : ℚ := 24
def number_of_people : ℕ := 8
def total_cost : ℚ := 48

-- Define the function to calculate scallops per person
def scallops_per_person : ℚ :=
  (total_cost / cost_per_pound * scallops_per_pound) / number_of_people

-- Theorem to prove
theorem scallops_per_person_is_two : scallops_per_person = 2 := by
  sorry

end NUMINAMATH_CALUDE_scallops_per_person_is_two_l2195_219565


namespace NUMINAMATH_CALUDE_elvis_song_writing_time_l2195_219554

/-- Proves that Elvis spent 15 minutes writing each song given the conditions of his album production. -/
theorem elvis_song_writing_time :
  let total_songs : ℕ := 10
  let total_studio_time : ℕ := 5 * 60  -- in minutes
  let recording_time_per_song : ℕ := 12
  let editing_time_all_songs : ℕ := 30
  let total_recording_time := total_songs * recording_time_per_song
  let remaining_time := total_studio_time - total_recording_time - editing_time_all_songs
  let writing_time_per_song := remaining_time / total_songs
  writing_time_per_song = 15 := by
    sorry

#check elvis_song_writing_time

end NUMINAMATH_CALUDE_elvis_song_writing_time_l2195_219554


namespace NUMINAMATH_CALUDE_means_inequality_l2195_219579

theorem means_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  Real.sqrt (a * b) > Real.rpow (a * b * c) (1/3) ∧
  Real.rpow (a * b * c) (1/3) > (2 * b * c) / (b + c) := by
sorry

end NUMINAMATH_CALUDE_means_inequality_l2195_219579


namespace NUMINAMATH_CALUDE_product_mod_23_l2195_219544

theorem product_mod_23 :
  (2003 * 2004 * 2005 * 2006 * 2007 * 2008) % 23 = 3 := by sorry

end NUMINAMATH_CALUDE_product_mod_23_l2195_219544


namespace NUMINAMATH_CALUDE_melinda_paid_759_l2195_219555

-- Define the cost of items
def doughnut_cost : ℚ := 0.45
def coffee_cost : ℚ := (4.91 - 3 * doughnut_cost) / 4

-- Define Melinda's purchase
def melinda_doughnuts : ℕ := 5
def melinda_coffees : ℕ := 6

-- Define Melinda's total cost
def melinda_total_cost : ℚ := melinda_doughnuts * doughnut_cost + melinda_coffees * coffee_cost

-- Theorem to prove
theorem melinda_paid_759 : melinda_total_cost = 7.59 := by
  sorry

end NUMINAMATH_CALUDE_melinda_paid_759_l2195_219555


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l2195_219580

theorem smallest_lcm_with_gcd_five (k l : ℕ) : 
  k ≥ 10000 ∧ k < 100000 ∧ 
  l ≥ 10000 ∧ l < 100000 ∧ 
  Nat.gcd k l = 5 → 
  Nat.lcm k l ≥ 20010000 := by
sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l2195_219580


namespace NUMINAMATH_CALUDE_complex_magnitude_l2195_219563

theorem complex_magnitude (z : ℂ) (h1 : z.im = 2) (h2 : (z^2 + 3).re = 0) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2195_219563


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2195_219562

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 1) :
  1 / a + 1 / b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2195_219562


namespace NUMINAMATH_CALUDE_modified_cube_edges_l2195_219533

/-- Represents a modified cube -/
structure ModifiedCube where
  sideLength : ℕ
  smallCubeRemoved1 : ℕ
  smallCubeRemoved2 : ℕ
  largeCubeRemoved : ℕ

/-- Calculates the number of edges in a modified cube -/
def edgeCount (c : ModifiedCube) : ℕ := sorry

/-- Theorem stating that a specific modified cube has 22 edges -/
theorem modified_cube_edges :
  let c : ModifiedCube := {
    sideLength := 4,
    smallCubeRemoved1 := 1,
    smallCubeRemoved2 := 1,
    largeCubeRemoved := 2
  }
  edgeCount c = 22 := by sorry

end NUMINAMATH_CALUDE_modified_cube_edges_l2195_219533


namespace NUMINAMATH_CALUDE_least_frood_drop_beats_eat_l2195_219524

def frood_drop_score (n : ℕ) : ℕ := n * (n + 1) / 2
def frood_eat_score (n : ℕ) : ℕ := 15 * n

theorem least_frood_drop_beats_eat :
  ∀ k : ℕ, k < 30 → frood_drop_score k ≤ frood_eat_score k ∧
  frood_drop_score 30 > frood_eat_score 30 :=
sorry

end NUMINAMATH_CALUDE_least_frood_drop_beats_eat_l2195_219524


namespace NUMINAMATH_CALUDE_jaylens_vegetables_l2195_219573

theorem jaylens_vegetables (x y z g : ℚ) : 
  x = 5/3 * y → 
  z = 2 * (1/2 * y) → 
  g = (1/2 * (x/4)) - 3 → 
  20 = x/4 → 
  x + y + z + g = 183 := by
  sorry

end NUMINAMATH_CALUDE_jaylens_vegetables_l2195_219573


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2195_219534

/-- Two lines in the plane, represented by their equations --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel --/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- Check if two lines are identical --/
def identical (l₁ l₂ : Line) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ l₁.a = k * l₂.a ∧ l₁.b = k * l₂.b ∧ l₁.c = k * l₂.c

theorem parallel_lines_a_value (a : ℝ) :
  let l₁ : Line := { a := a, b := 3, c := 1 }
  let l₂ : Line := { a := 2, b := a + 1, c := 1 }
  parallel l₁ l₂ ∧ ¬ identical l₁ l₂ → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2195_219534


namespace NUMINAMATH_CALUDE_curve_tangent_l2195_219556

/-- Given a curve C defined by x = √2 cos(φ) and y = sin(φ), prove that for a point M on C,
    if the angle between OM and the positive x-axis is π/3, then tan(φ) = √6. -/
theorem curve_tangent (φ : ℝ) : 
  let M : ℝ × ℝ := (Real.sqrt 2 * Real.cos φ, Real.sin φ)
  (M.2 / M.1 = Real.tan (π / 3)) → Real.tan φ = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_l2195_219556


namespace NUMINAMATH_CALUDE_ab_value_l2195_219592

/-- Given that p and q are integers satisfying the equation and other conditions, prove that ab = 10^324 -/
theorem ab_value (p q : ℤ) (a b : ℝ) 
  (hp : p = Real.sqrt (Real.log a))
  (hq : q = Real.sqrt (Real.log b))
  (ha : a = 10^(p^2))
  (hb : b = 10^(q^2))
  (heq : 2*p + 2*q + (Real.log a)/2 + (Real.log b)/2 + p * q = 200) :
  a * b = 10^324 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2195_219592


namespace NUMINAMATH_CALUDE_jonathan_distance_l2195_219564

theorem jonathan_distance (J : ℝ) 
  (mercedes_distance : ℝ → ℝ)
  (davonte_distance : ℝ → ℝ)
  (h1 : mercedes_distance J = 2 * J)
  (h2 : davonte_distance J = mercedes_distance J + 2)
  (h3 : mercedes_distance J + davonte_distance J = 32) :
  J = 7.5 := by
sorry

end NUMINAMATH_CALUDE_jonathan_distance_l2195_219564


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l2195_219550

theorem sum_of_coefficients_zero (x y : ℝ) : 
  (fun x y => (3 * x^2 - 5 * x * y + 2 * y^2)^5) 1 1 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l2195_219550


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l2195_219566

theorem units_digit_of_7_power_2023 : (7^2023 : ℕ) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l2195_219566


namespace NUMINAMATH_CALUDE_lemon_bag_mass_l2195_219542

theorem lemon_bag_mass (max_load : ℕ) (num_bags : ℕ) (remaining_capacity : ℕ) 
  (h1 : max_load = 900)
  (h2 : num_bags = 100)
  (h3 : remaining_capacity = 100) :
  (max_load - remaining_capacity) / num_bags = 8 := by
  sorry

end NUMINAMATH_CALUDE_lemon_bag_mass_l2195_219542


namespace NUMINAMATH_CALUDE_inequality_solution_l2195_219515

theorem inequality_solution (x : ℕ) : 
  (x + 3 : ℚ) / (x^2 - 4) - 1 / (x + 2) < 2 * x / (2 * x - x^2) ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2195_219515


namespace NUMINAMATH_CALUDE_sum_less_than_six_for_735_l2195_219548

def is_less_than_six (n : ℕ) : Bool :=
  n < 6

def sum_less_than_six (cards : List ℕ) : ℕ :=
  (cards.filter is_less_than_six).sum

theorem sum_less_than_six_for_735 : 
  ∃ (cards : List ℕ), 
    cards.length = 3 ∧ 
    (∀ n ∈ cards, 1 ≤ n ∧ n ≤ 9) ∧
    cards.foldl (λ acc d => acc * 10 + d) 0 = 735 ∧
    sum_less_than_six cards = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_less_than_six_for_735_l2195_219548


namespace NUMINAMATH_CALUDE_min_values_l2195_219590

theorem min_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 3/b = 1 → x + 3*y ≤ a + 3*b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 3/b = 1 → x*y ≤ a*b) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 3/b = 1 ∧ x + 3*y = a + 3*b ∧ x*y = a*b) :=
by sorry

end NUMINAMATH_CALUDE_min_values_l2195_219590


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l2195_219505

theorem incorrect_observation_value 
  (n : ℕ) 
  (original_mean : ℝ) 
  (new_mean : ℝ) 
  (correct_value : ℝ) 
  (h1 : n = 50) 
  (h2 : original_mean = 36) 
  (h3 : new_mean = 36.5) 
  (h4 : correct_value = 45) :
  ∃ (incorrect_value : ℝ), 
    (n : ℝ) * original_mean = (n : ℝ) * new_mean - correct_value + incorrect_value ∧ 
    incorrect_value = 20 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l2195_219505


namespace NUMINAMATH_CALUDE_sureshs_speed_l2195_219552

/-- Suresh's walking speed problem -/
theorem sureshs_speed (track_circumference : ℝ) (meeting_time : ℝ) (wife_speed : ℝ) 
  (h1 : track_circumference = 726) 
  (h2 : meeting_time = 5.28)
  (h3 : wife_speed = 3.75) : 
  ∃ (suresh_speed : ℝ), abs (suresh_speed - 4.5054) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_sureshs_speed_l2195_219552


namespace NUMINAMATH_CALUDE_fredrickson_chickens_l2195_219577

/-- Given a total number of chickens, calculates the number of chickens that do not lay eggs. -/
def chickens_not_laying_eggs (total : ℕ) : ℕ :=
  let roosters := total / 4
  let hens := total - roosters
  let laying_hens := (hens * 3) / 4
  roosters + (hens - laying_hens)

/-- Theorem stating that for 80 chickens, where 1/4 are roosters and 3/4 of hens lay eggs,
    the number of chickens not laying eggs is 35. -/
theorem fredrickson_chickens :
  chickens_not_laying_eggs 80 = 35 := by
  sorry

#eval chickens_not_laying_eggs 80

end NUMINAMATH_CALUDE_fredrickson_chickens_l2195_219577


namespace NUMINAMATH_CALUDE_correct_factorization_l2195_219538

theorem correct_factorization (x y : ℝ) : x * (x - y) - y * (x - y) = (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l2195_219538


namespace NUMINAMATH_CALUDE_theater_seat_count_l2195_219527

/-- Calculates the total number of seats in a theater with the given configuration. -/
def theater_seats (total_rows : ℕ) (odd_row_seats : ℕ) (even_row_seats : ℕ) : ℕ :=
  let odd_rows := (total_rows + 1) / 2
  let even_rows := total_rows / 2
  odd_rows * odd_row_seats + even_rows * even_row_seats

/-- Theorem stating that a theater with 11 rows, where odd rows have 15 seats
    and even rows have 16 seats, has a total of 170 seats. -/
theorem theater_seat_count :
  theater_seats 11 15 16 = 170 := by
  sorry

#eval theater_seats 11 15 16

end NUMINAMATH_CALUDE_theater_seat_count_l2195_219527


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2195_219529

/-- A geometric sequence with given first and fourth terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 1)
  (h_fourth : a 4 = 27) :
  a 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2195_219529


namespace NUMINAMATH_CALUDE_pie_chart_highlights_part_whole_l2195_219596

/-- Enumeration of statistical graph types --/
inductive StatisticalGraph
  | BarGraph
  | PieChart
  | LineGraph
  | FrequencyDistributionHistogram

/-- Function to determine if a graph type highlights part-whole relationships --/
def highlights_part_whole_relationship (graph : StatisticalGraph) : Prop :=
  match graph with
  | StatisticalGraph.PieChart => True
  | _ => False

/-- Theorem stating that the Pie chart is the graph that highlights part-whole relationships --/
theorem pie_chart_highlights_part_whole :
  ∀ (graph : StatisticalGraph),
    highlights_part_whole_relationship graph ↔ graph = StatisticalGraph.PieChart :=
by sorry

end NUMINAMATH_CALUDE_pie_chart_highlights_part_whole_l2195_219596


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2195_219570

theorem complex_fraction_equality : (5 * Complex.I) / (1 - 2 * Complex.I) = -2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2195_219570


namespace NUMINAMATH_CALUDE_angle_relation_l2195_219597

theorem angle_relation (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.tan (α - β) = 1/2) (h4 : Real.tan β = -1/7) :
  2*α - β = -3*π/4 := by sorry

end NUMINAMATH_CALUDE_angle_relation_l2195_219597


namespace NUMINAMATH_CALUDE_teena_speed_calculation_l2195_219588

/-- Teena's speed in miles per hour -/
def teena_speed : ℝ := 55

/-- Roe's speed in miles per hour -/
def roe_speed : ℝ := 40

/-- Initial distance Teena is behind Roe in miles -/
def initial_distance_behind : ℝ := 7.5

/-- Time elapsed in hours -/
def time_elapsed : ℝ := 1.5

/-- Final distance Teena is ahead of Roe in miles -/
def final_distance_ahead : ℝ := 15

theorem teena_speed_calculation :
  teena_speed * time_elapsed = 
    roe_speed * time_elapsed + initial_distance_behind + final_distance_ahead := by
  sorry

#check teena_speed_calculation

end NUMINAMATH_CALUDE_teena_speed_calculation_l2195_219588


namespace NUMINAMATH_CALUDE_odd_prime_condition_l2195_219501

theorem odd_prime_condition (p : ℕ) (h_prime : Nat.Prime p) : 
  (∃! k : ℕ, Even k ∧ k ∣ (14 * p)) → Odd p :=
sorry

end NUMINAMATH_CALUDE_odd_prime_condition_l2195_219501


namespace NUMINAMATH_CALUDE_adjusted_target_heart_rate_for_30_year_old_l2195_219551

/-- Calculates the adjusted target heart rate for a runner --/
def adjustedTargetHeartRate (age : ℕ) : ℕ :=
  let maxHeartRate : ℕ := 220 - age
  let initialTargetRate : ℚ := 0.7 * maxHeartRate
  let adjustment : ℚ := 0.1 * initialTargetRate
  let adjustedRate : ℚ := initialTargetRate + adjustment
  (adjustedRate + 0.5).floor.toNat

/-- Theorem stating that for a 30-year-old runner, the adjusted target heart rate is 146 bpm --/
theorem adjusted_target_heart_rate_for_30_year_old :
  adjustedTargetHeartRate 30 = 146 := by
  sorry

#eval adjustedTargetHeartRate 30

end NUMINAMATH_CALUDE_adjusted_target_heart_rate_for_30_year_old_l2195_219551


namespace NUMINAMATH_CALUDE_product_equals_square_l2195_219540

theorem product_equals_square : 500 * 2019 * 0.0505 * 20 = (2019 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l2195_219540


namespace NUMINAMATH_CALUDE_abs_sum_inequality_positive_reals_inequality_l2195_219545

-- Problem 1
theorem abs_sum_inequality (x : ℝ) :
  |x - 1| + |x + 1| ≤ 4 ↔ x ∈ Set.Icc (-2 : ℝ) 2 := by sorry

-- Problem 2
theorem positive_reals_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / a^2 + 1 / b^2 + 1 / c^2 ≥ a + b + c := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_positive_reals_inequality_l2195_219545


namespace NUMINAMATH_CALUDE_sqrt_49_is_7_l2195_219567

theorem sqrt_49_is_7 : Real.sqrt 49 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_is_7_l2195_219567


namespace NUMINAMATH_CALUDE_potato_wedges_count_l2195_219525

/-- The number of wedges one potato can be cut into -/
def wedges_per_potato : ℕ := sorry

/-- The total number of potatoes harvested -/
def total_potatoes : ℕ := 67

/-- The number of potatoes cut into wedges -/
def wedge_potatoes : ℕ := 13

/-- The number of potato chips one potato can make -/
def chips_per_potato : ℕ := 20

/-- The difference between the number of potato chips and wedges -/
def chip_wedge_difference : ℕ := 436

theorem potato_wedges_count :
  wedges_per_potato = 8 ∧
  (total_potatoes - wedge_potatoes) / 2 * chips_per_potato - wedge_potatoes * wedges_per_potato = chip_wedge_difference :=
by sorry

end NUMINAMATH_CALUDE_potato_wedges_count_l2195_219525


namespace NUMINAMATH_CALUDE_q_min_at_two_l2195_219572

/-- The function q(x) defined as (x - 5)^2 + (x + 1)^2 - 6 -/
def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

/-- Theorem stating that q(x) has a minimum value when x = 2 -/
theorem q_min_at_two : 
  ∀ x : ℝ, q 2 ≤ q x := by sorry

end NUMINAMATH_CALUDE_q_min_at_two_l2195_219572


namespace NUMINAMATH_CALUDE_flowchart_connection_is_flow_line_l2195_219568

-- Define the basic elements of a flowchart
inductive FlowchartElement
  | ConnectionPoint
  | DecisionBox
  | FlowLine
  | ProcessBox

-- Define a property for connecting steps in a flowchart
def connects_steps (element : FlowchartElement) : Prop :=
  element = FlowchartElement.FlowLine

-- Theorem statement
theorem flowchart_connection_is_flow_line :
  ∃ (element : FlowchartElement), connects_steps element :=
sorry

end NUMINAMATH_CALUDE_flowchart_connection_is_flow_line_l2195_219568


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l2195_219510

theorem quadratic_function_proof (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = -2 ∨ x = 4) →
  (∃ x, ∀ y, a * y^2 + b * y + c ≤ a * x^2 + b * x + c) →
  (∃ x, a * x^2 + b * x + c = 9) →
  (∀ x, a * x^2 + b * x + c = -x^2 + 2*x + 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l2195_219510


namespace NUMINAMATH_CALUDE_class_selection_theorem_l2195_219569

theorem class_selection_theorem (n m k a : ℕ) (h1 : n = 10) (h2 : m = 4) (h3 : k = 4) (h4 : a = 2) :
  (Nat.choose m a) * (Nat.choose (n - m) (k - a)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_class_selection_theorem_l2195_219569


namespace NUMINAMATH_CALUDE_sheep_buying_problem_sheep_buying_problem_unique_l2195_219512

/-- The number of people buying the sheep -/
def num_people : ℕ := 21

/-- The price of the sheep in coins -/
def sheep_price : ℕ := 150

/-- Theorem stating the solution to the sheep-buying problem -/
theorem sheep_buying_problem :
  (∃ (n : ℕ) (p : ℕ),
    n = num_people ∧
    p = sheep_price ∧
    5 * n + 45 = p ∧
    7 * n + 3 = p) :=
by sorry

/-- Theorem proving the uniqueness of the solution -/
theorem sheep_buying_problem_unique :
  ∀ (n : ℕ) (p : ℕ),
    5 * n + 45 = p ∧
    7 * n + 3 = p →
    n = num_people ∧
    p = sheep_price :=
by sorry

end NUMINAMATH_CALUDE_sheep_buying_problem_sheep_buying_problem_unique_l2195_219512


namespace NUMINAMATH_CALUDE_max_consecutive_interesting_numbers_l2195_219584

/-- A function that checks if a number is interesting (has at least one digit divisible by 3) -/
def is_interesting (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d % 3 = 0

/-- The theorem stating the maximum number of consecutive interesting three-digit numbers -/
theorem max_consecutive_interesting_numbers :
  ∃ start : ℕ,
    start ≥ 100 ∧
    start + 121 ≤ 999 ∧
    (∀ k : ℕ, k ∈ Finset.range 122 → is_interesting (start + k)) ∧
    (∀ m : ℕ, m > 122 →
      ¬∃ s : ℕ, s ≥ 100 ∧ s + m - 1 ≤ 999 ∧
        ∀ j : ℕ, j ∈ Finset.range m → is_interesting (s + j)) :=
by
  sorry


end NUMINAMATH_CALUDE_max_consecutive_interesting_numbers_l2195_219584


namespace NUMINAMATH_CALUDE_john_total_pay_this_year_l2195_219575

/-- John's annual bonus calculation -/
def johnBonus (baseSalaryLastYear : ℝ) (firstBonusLastYear : ℝ) (baseSalaryThisYear : ℝ) 
              (bonusGrowthRate : ℝ) (projectBonus : ℝ) (projectsCompleted : ℕ) : ℝ :=
  let firstBonusThisYear := firstBonusLastYear * (1 + bonusGrowthRate)
  let secondBonus := projectBonus * projectsCompleted
  baseSalaryThisYear + firstBonusThisYear + secondBonus

theorem john_total_pay_this_year :
  johnBonus 100000 10000 200000 0.05 2000 8 = 226500 := by
  sorry

end NUMINAMATH_CALUDE_john_total_pay_this_year_l2195_219575


namespace NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_root_l2195_219509

theorem unique_magnitude_of_quadratic_root : ∃! m : ℝ, ∃ z : ℂ, z^2 - 6*z + 25 = 0 ∧ Complex.abs z = m := by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_root_l2195_219509


namespace NUMINAMATH_CALUDE_flight_speed_l2195_219503

/-- Given a flight distance and time, calculate the speed -/
theorem flight_speed (distance : ℝ) (time : ℝ) (h1 : distance = 256) (h2 : time = 8) :
  distance / time = 32 := by
  sorry

end NUMINAMATH_CALUDE_flight_speed_l2195_219503


namespace NUMINAMATH_CALUDE_sufficient_conditions_for_x_squared_less_than_one_l2195_219593

theorem sufficient_conditions_for_x_squared_less_than_one :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 < 1) ∧
  (∀ x : ℝ, -1 < x ∧ x < 0 → x^2 < 1) ∧
  (∀ x : ℝ, -1 < x ∧ x < 1 → x^2 < 1) ∧
  (∃ x : ℝ, x < 1 ∧ x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_conditions_for_x_squared_less_than_one_l2195_219593


namespace NUMINAMATH_CALUDE_unique_solution_l2195_219513

/-- A function from positive reals to positive reals -/
def PositiveFunction := {f : ℝ → ℝ // ∀ x, x > 0 → f x > 0}

/-- The functional equation -/
def SatisfiesEquation (f : PositiveFunction) (c : ℝ) : Prop :=
  c > 0 ∧ ∀ x y, x > 0 → y > 0 → f.val ((c + 1) * x + f.val y) = f.val (x + 2 * y) + 2 * c * x

/-- The theorem statement -/
theorem unique_solution (f : PositiveFunction) (c : ℝ) 
  (h : SatisfiesEquation f c) : 
  ∀ x, x > 0 → f.val x = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2195_219513


namespace NUMINAMATH_CALUDE_triangle_area_l2195_219595

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is 3 under the given conditions. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  a = Real.sqrt 5 →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  (1 / 2) * a * c * Real.sin B = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2195_219595


namespace NUMINAMATH_CALUDE_tricia_age_is_five_l2195_219537

-- Define the ages as natural numbers
def tricia_age : ℕ := 5
def amilia_age : ℕ := 3 * tricia_age
def yorick_age : ℕ := 4 * amilia_age
def eugene_age : ℕ := yorick_age / 2
def khloe_age : ℕ := eugene_age / 3
def rupert_age : ℕ := khloe_age + 10
def vincent_age : ℕ := 22

-- State the theorem
theorem tricia_age_is_five :
  tricia_age = 5 ∧
  amilia_age = 3 * tricia_age ∧
  yorick_age = 4 * amilia_age ∧
  yorick_age = 2 * eugene_age ∧
  khloe_age = eugene_age / 3 ∧
  rupert_age = khloe_age + 10 ∧
  vincent_age = 22 ∧
  rupert_age < vincent_age →
  tricia_age = 5 := by
sorry

end NUMINAMATH_CALUDE_tricia_age_is_five_l2195_219537


namespace NUMINAMATH_CALUDE_legoland_animals_l2195_219599

theorem legoland_animals (kangaroos koalas pandas : ℕ) : 
  kangaroos = 567 →
  kangaroos = 9 * koalas →
  koalas = 7 * pandas →
  kangaroos + koalas + pandas = 639 := by
sorry

end NUMINAMATH_CALUDE_legoland_animals_l2195_219599


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2195_219571

theorem minimum_value_theorem (x : ℝ) (h : x > 3) :
  (x + 18) / Real.sqrt (x - 3) ≥ 2 * Real.sqrt 21 ∧
  (∃ x₀ : ℝ, x₀ > 3 ∧ (x₀ + 18) / Real.sqrt (x₀ - 3) = 2 * Real.sqrt 21 ∧ x₀ = 24) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2195_219571


namespace NUMINAMATH_CALUDE_trajectory_of_P_l2195_219522

-- Define the coordinate system
variable (O : ℝ × ℝ)  -- Origin
variable (A B P : ℝ → ℝ × ℝ)  -- Points as functions of time

-- Define the conditions
axiom origin : O = (0, 0)
axiom A_on_x_axis : ∀ t, (A t).2 = 0
axiom B_on_y_axis : ∀ t, (B t).1 = 0
axiom AB_length : ∀ t, Real.sqrt ((A t).1^2 + (B t).2^2) = 3
axiom P_position : ∀ t, P t = (2/3 • A t) + (1/3 • B t)

-- State the theorem
theorem trajectory_of_P :
  ∀ t, (P t).1^2 / 4 + (P t).2^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l2195_219522


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2195_219536

theorem expand_and_simplify (x : ℝ) : (17*x - 9) * 3*x = 51*x^2 - 27*x := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2195_219536


namespace NUMINAMATH_CALUDE_parallel_line_distance_in_circle_l2195_219587

/-- Given a circle intersected by four equally spaced parallel lines creating chords of lengths 44, 44, 40, and 40, the distance between two adjacent parallel lines is 8/√23. -/
theorem parallel_line_distance_in_circle : ∀ (r : ℝ) (d : ℝ),
  (44 + (1/4) * d^2 = r^2) →
  (40 + (27/16) * d^2 = r^2) →
  d = 8 / Real.sqrt 23 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_distance_in_circle_l2195_219587


namespace NUMINAMATH_CALUDE_garden_length_l2195_219547

/-- Proves that a rectangular garden with perimeter 80 meters and width 15 meters has a length of 25 meters -/
theorem garden_length (perimeter width : ℝ) (h1 : perimeter = 80) (h2 : width = 15) :
  let length := (perimeter / 2) - width
  length = 25 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l2195_219547


namespace NUMINAMATH_CALUDE_unique_self_opposite_l2195_219532

theorem unique_self_opposite : ∃! x : ℝ, x = -x := by sorry

end NUMINAMATH_CALUDE_unique_self_opposite_l2195_219532
