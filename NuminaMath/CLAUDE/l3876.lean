import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3876_387689

theorem complex_modulus_problem (z : ℂ) (h : z * (2 - Complex.I) = 1 + 7 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3876_387689


namespace NUMINAMATH_CALUDE_hazel_received_six_l3876_387621

/-- The number of shirts Hazel received -/
def hazel_shirts : ℕ := sorry

/-- The number of shirts Razel received -/
def razel_shirts : ℕ := sorry

/-- The total number of shirts Hazel and Razel have -/
def total_shirts : ℕ := 18

/-- Razel received twice the number of shirts as Hazel -/
axiom razel_twice_hazel : razel_shirts = 2 * hazel_shirts

/-- The total number of shirts is the sum of Hazel's and Razel's shirts -/
axiom total_is_sum : total_shirts = hazel_shirts + razel_shirts

/-- Theorem: Hazel received 6 shirts -/
theorem hazel_received_six : hazel_shirts = 6 := by sorry

end NUMINAMATH_CALUDE_hazel_received_six_l3876_387621


namespace NUMINAMATH_CALUDE_watermelon_price_in_units_l3876_387608

/-- The price of a watermelon in won -/
def watermelon_price : ℝ := 5000 - 200

/-- The conversion factor from won to units of 1000 won -/
def conversion_factor : ℝ := 1000

theorem watermelon_price_in_units : watermelon_price / conversion_factor = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_price_in_units_l3876_387608


namespace NUMINAMATH_CALUDE_roots_of_equation_l3876_387637

theorem roots_of_equation (x : ℝ) : (x - 1)^2 = 1 ↔ x = 0 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3876_387637


namespace NUMINAMATH_CALUDE_field_ratio_l3876_387632

/-- Proves that a rectangular field with perimeter 336 meters and width 70 meters has a length-to-width ratio of 7:5 -/
theorem field_ratio (perimeter width : ℝ) (h1 : perimeter = 336) (h2 : width = 70) :
  (perimeter / 2 - width) / width = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_l3876_387632


namespace NUMINAMATH_CALUDE_brother_d_payment_l3876_387676

theorem brother_d_payment (n : ℕ) (a₁ d : ℚ) (h₁ : n = 5) (h₂ : a₁ = 300) 
  (h₃ : n / 2 * (2 * a₁ + (n - 1) * d) = 1000) : a₁ + 3 * d = 450 := by
  sorry

end NUMINAMATH_CALUDE_brother_d_payment_l3876_387676


namespace NUMINAMATH_CALUDE_quadratic_floor_existence_l3876_387617

theorem quadratic_floor_existence (x : ℝ) : 
  (∃ a b : ℤ, ∀ x : ℝ, x^2 + a*x + b ≠ 0 ∧ ∃ y : ℝ, ⌊y^2⌋ + a*y + b = 0) ∧
  (¬∃ a b : ℤ, ∀ x : ℝ, x^2 + 2*a*x + b ≠ 0 ∧ ∃ y : ℝ, ⌊y^2⌋ + 2*a*y + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_floor_existence_l3876_387617


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3876_387670

theorem min_value_expression (x : ℝ) (h : x > -1) :
  2 * x + 1 / (x + 1) ≥ 2 * Real.sqrt 2 - 2 :=
by sorry

theorem min_value_achievable :
  ∃ x > -1, 2 * x + 1 / (x + 1) = 2 * Real.sqrt 2 - 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3876_387670


namespace NUMINAMATH_CALUDE_point_on_line_for_all_k_l3876_387684

/-- The point P lies on the line (k+2)x + (1-k)y - 4k - 5 = 0 for all values of k. -/
theorem point_on_line_for_all_k :
  ∀ (k : ℝ), (k + 2) * 3 + (1 - k) * (-1) - 4 * k - 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_for_all_k_l3876_387684


namespace NUMINAMATH_CALUDE_triangle_side_relation_l3876_387636

theorem triangle_side_relation (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Positive lengths
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  a^2 + 4*a*c + 3*c^2 - 3*a*b - 7*b*c + 2*b^2 = 0 →
  a + c - 2*b = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l3876_387636


namespace NUMINAMATH_CALUDE_square_of_95_l3876_387649

theorem square_of_95 : (95 : ℤ)^2 = 100^2 - 2 * 100 * 5 + 5^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_95_l3876_387649


namespace NUMINAMATH_CALUDE_equation_solutions_l3876_387654

def equation (x y : ℤ) : ℤ := 
  4*x^3 + 4*x^2*y - 15*x*y^2 - 18*y^3 - 12*x^2 + 6*x*y + 36*y^2 + 5*x - 10*y

def solution_set : Set (ℤ × ℤ) :=
  {p | p.1 = 1 ∧ p.2 = 1} ∪ {p | ∃ (y : ℕ), p.1 = 2*y ∧ p.2 = y}

theorem equation_solutions :
  ∀ (x y : ℤ), x > 0 ∧ y > 0 →
    (equation x y = 0 ↔ (x, y) ∈ solution_set) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3876_387654


namespace NUMINAMATH_CALUDE_gnomon_magic_diagonal_sums_equal_l3876_387669

/-- Represents a 3x3 square --/
def Square := Matrix (Fin 3) (Fin 3) ℝ

/-- Checks if a 3x3 square is gnomon-magic --/
def is_gnomon_magic (s : Square) : Prop :=
  let sum1 := s 1 1 + s 1 2 + s 2 1 + s 2 2
  let sum2 := s 1 2 + s 1 3 + s 2 2 + s 2 3
  let sum3 := s 2 1 + s 2 2 + s 3 1 + s 3 2
  let sum4 := s 2 2 + s 2 3 + s 3 2 + s 3 3
  sum1 = sum2 ∧ sum2 = sum3 ∧ sum3 = sum4

/-- Calculates the sum of the main diagonal --/
def main_diagonal_sum (s : Square) : ℝ :=
  s 1 1 + s 2 2 + s 3 3

/-- Calculates the sum of the anti-diagonal --/
def anti_diagonal_sum (s : Square) : ℝ :=
  s 1 3 + s 2 2 + s 3 1

/-- Theorem: In a 3x3 gnomon-magic square, the sums of numbers along the two diagonals are equal --/
theorem gnomon_magic_diagonal_sums_equal (s : Square) (h : is_gnomon_magic s) :
  main_diagonal_sum s = anti_diagonal_sum s := by
  sorry

end NUMINAMATH_CALUDE_gnomon_magic_diagonal_sums_equal_l3876_387669


namespace NUMINAMATH_CALUDE_curve_points_difference_l3876_387679

theorem curve_points_difference (a b : ℝ) : 
  (a ≠ b) → 
  ((a : ℝ)^2 + (Real.sqrt π)^4 = 2 * (Real.sqrt π)^2 * a + 1) → 
  ((b : ℝ)^2 + (Real.sqrt π)^4 = 2 * (Real.sqrt π)^2 * b + 1) → 
  |a - b| = 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_points_difference_l3876_387679


namespace NUMINAMATH_CALUDE_inequality_range_l3876_387665

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, |x + 1| - |x - 2| > k) → k < -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3876_387665


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l3876_387666

theorem real_part_of_complex_number (z : ℂ) 
  (h1 : Complex.abs z = 1)
  (h2 : Complex.abs (z - 1.45) = 1.05) :
  z.re = 20 / 29 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l3876_387666


namespace NUMINAMATH_CALUDE_max_single_game_schedules_max_n_value_l3876_387678

/-- Represents a chess tournament between two teams -/
structure ChessTournament where
  team_size : ℕ
  total_games : ℕ
  games_played : ℕ

/-- Creates a chess tournament with the given parameters -/
def create_tournament (size : ℕ) : ChessTournament :=
  { team_size := size
  , total_games := size * size
  , games_played := 0 }

/-- Theorem stating the maximum number of ways to schedule a single game -/
theorem max_single_game_schedules (t : ChessTournament) (h1 : t.team_size = 15) :
  (t.total_games - t.games_played) ≤ 120 := by
  sorry

/-- Main theorem proving the maximum value of N -/
theorem max_n_value :
  ∃ (t : ChessTournament), t.team_size = 15 ∧ (t.total_games - t.games_played) = 120 := by
  sorry

end NUMINAMATH_CALUDE_max_single_game_schedules_max_n_value_l3876_387678


namespace NUMINAMATH_CALUDE_age_of_25th_student_l3876_387607

/-- The age of the 25th student in a class with specific age distributions -/
theorem age_of_25th_student (total_students : ℕ) (avg_age : ℝ) 
  (group1_count : ℕ) (group1_avg : ℝ) (group2_count : ℕ) (group2_avg : ℝ) :
  total_students = 25 →
  avg_age = 25 →
  group1_count = 10 →
  group1_avg = 22 →
  group2_count = 14 →
  group2_avg = 28 →
  (total_students * avg_age) - (group1_count * group1_avg + group2_count * group2_avg) = 13 :=
by sorry

end NUMINAMATH_CALUDE_age_of_25th_student_l3876_387607


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3876_387693

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 1057 ∧ m = 23) :
  ∃ x : ℕ, x = 1 ∧
  (∀ y : ℕ, (n + y) % m = 0 → y ≥ x) ∧
  (n + x) % m = 0 :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3876_387693


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3876_387653

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmeticSequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3876_387653


namespace NUMINAMATH_CALUDE_scooter_travel_time_l3876_387680

/-- The time it takes for a scooter to travel a given distance, given the following conditions:
  * The distance between two points A and B is 50 miles
  * A bicycle travels 1/2 mile per hour slower than the scooter
  * The bicycle takes 45 minutes (3/4 hour) more than the scooter to make the trip
  * x is the scooter's rate of speed in miles per hour
-/
theorem scooter_travel_time (x : ℝ) : 
  (∃ y : ℝ, y > 0 ∧ y = x - 1/2) →  -- Bicycle speed exists and is positive
  50 / (x - 1/2) - 50 / x = 3/4 →   -- Time difference equation
  50 / x = 50 / x :=                -- Conclusion (trivial here, but represents the result)
by sorry

end NUMINAMATH_CALUDE_scooter_travel_time_l3876_387680


namespace NUMINAMATH_CALUDE_equation_solution_l3876_387622

theorem equation_solution : ∃ (x₁ x₂ : ℚ), 
  (x₁ = 1/9 ∧ x₂ = 1/18) ∧ 
  (∀ x : ℚ, (101*x^2 - 18*x + 1)^2 - 121*x^2*(101*x^2 - 18*x + 1) + 2020*x^4 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3876_387622


namespace NUMINAMATH_CALUDE_rope_length_proof_l3876_387614

/-- The length of a rope after being folded in half twice -/
def folded_length : ℝ := 10

/-- The number of times the rope is folded in half -/
def fold_count : ℕ := 2

/-- Calculates the original length of the rope before folding -/
def original_length : ℝ := folded_length * (2 ^ fold_count)

/-- Proves that the original length of the rope is 40 centimeters -/
theorem rope_length_proof : original_length = 40 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_proof_l3876_387614


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l3876_387604

/-- Given a square with an inscribed circle, if the perimeter of the square (in inches) 
    equals the area of the circle (in square inches), then the diameter of the circle 
    is 16/π inches. -/
theorem inscribed_circle_diameter (s : ℝ) (r : ℝ) (h : s > 0) :
  (4 * s = π * r^2) → (2 * r = 16 / π) := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l3876_387604


namespace NUMINAMATH_CALUDE_cos_330_degrees_l3876_387677

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l3876_387677


namespace NUMINAMATH_CALUDE_min_output_no_losses_l3876_387639

-- Define the total cost function
def total_cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the selling price per unit
def selling_price : ℝ := 25

-- Define the condition for no losses
def no_losses (x : ℕ) : Prop := selling_price * x ≥ total_cost x

-- State the theorem
theorem min_output_no_losses :
  ∃ (x : ℕ), x > 0 ∧ x < 240 ∧ no_losses x ∧
  ∀ (y : ℕ), y > 0 ∧ y < x → ¬(no_losses y) ∧
  x = 150 :=
sorry

end NUMINAMATH_CALUDE_min_output_no_losses_l3876_387639


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3876_387651

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 2 → b = 3 → c^2 = a^2 + b^2 → c = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3876_387651


namespace NUMINAMATH_CALUDE_class_composition_after_adding_boys_l3876_387661

theorem class_composition_after_adding_boys (initial_boys initial_girls added_boys : ℕ) 
  (h1 : initial_boys = 11)
  (h2 : initial_girls = 13)
  (h3 : added_boys = 1) :
  (initial_girls : ℚ) / ((initial_boys + initial_girls + added_boys) : ℚ) = 52 / 100 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_after_adding_boys_l3876_387661


namespace NUMINAMATH_CALUDE_a_work_days_l3876_387687

/-- The number of days B takes to finish the work alone -/
def b_days : ℝ := 15

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B works alone after A leaves -/
def b_alone_days : ℝ := 7

/-- The total amount of work to be done -/
def total_work : ℝ := 1

-- The theorem to prove
theorem a_work_days : 
  ∃ (x : ℝ), 
    x > 0 ∧ 
    together_days * (1/x + 1/b_days) + b_alone_days * (1/b_days) = total_work ∧ 
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_a_work_days_l3876_387687


namespace NUMINAMATH_CALUDE_different_winning_scores_l3876_387641

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  /-- The number of runners in each team -/
  runners_per_team : Nat
  /-- The total number of runners -/
  total_runners : Nat
  /-- The sum of all positions -/
  total_sum : Nat
  /-- The lowest possible winning score -/
  min_winning_score : Nat
  /-- The highest possible winning score -/
  max_winning_score : Nat
  /-- Assertion that there are two teams -/
  two_teams : total_runners = 2 * runners_per_team
  /-- Assertion that the total sum is correct -/
  sum_correct : total_sum = (total_runners * (total_runners + 1)) / 2
  /-- Assertion that the minimum winning score is correct -/
  min_score_correct : min_winning_score = (runners_per_team * (runners_per_team + 1)) / 2
  /-- Assertion that the maximum winning score is less than half the total sum -/
  max_score_correct : max_winning_score = (total_sum / 2) - 1

/-- The main theorem stating the number of different winning scores -/
theorem different_winning_scores (meet : CrossCountryMeet) (h : meet.runners_per_team = 5) :
  (meet.max_winning_score - meet.min_winning_score + 1) = 13 := by
  sorry

end NUMINAMATH_CALUDE_different_winning_scores_l3876_387641


namespace NUMINAMATH_CALUDE_min_value_of_g_l3876_387660

theorem min_value_of_g (φ : Real) (h1 : 0 < φ) (h2 : φ < π) : 
  let f := fun x => Real.sqrt 3 * Real.sin (2 * x + φ) + Real.cos (2 * x + φ)
  let g := fun x => f (x - 3 * π / 4)
  (∀ y, f (π / 12 - y) = f (π / 12 + y)) →
  (∃ x ∈ Set.Icc (-π / 4) (π / 6), g x = -1) ∧
  (∀ x ∈ Set.Icc (-π / 4) (π / 6), g x ≥ -1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_g_l3876_387660


namespace NUMINAMATH_CALUDE_derivative_inequality_l3876_387674

theorem derivative_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : ∀ x, f' x + f x > 0) (x₁ x₂ : ℝ) (h3 : x₁ < x₂) :
  Real.exp x₁ * f x₁ < Real.exp x₂ * f x₂ := by
  sorry

end NUMINAMATH_CALUDE_derivative_inequality_l3876_387674


namespace NUMINAMATH_CALUDE_fisherman_multiple_is_three_l3876_387655

/-- The multiple of fish caught by the fisherman compared to the pelican and kingfisher combined -/
def fisherman_multiple (pelican_fish kingfisher_fish fisherman_fish : ℕ) : ℚ :=
  fisherman_fish / (pelican_fish + kingfisher_fish)

/-- Theorem stating the multiple of fish caught by the fisherman -/
theorem fisherman_multiple_is_three :
  ∀ (pelican_fish kingfisher_fish fisherman_fish : ℕ),
    pelican_fish = 13 →
    kingfisher_fish = pelican_fish + 7 →
    fisherman_fish = pelican_fish + 86 →
    fisherman_multiple pelican_fish kingfisher_fish fisherman_fish = 3 := by
  sorry

#eval fisherman_multiple 13 20 99

end NUMINAMATH_CALUDE_fisherman_multiple_is_three_l3876_387655


namespace NUMINAMATH_CALUDE_xyz_sum_product_bounds_l3876_387686

theorem xyz_sum_product_bounds (x y z : ℝ) 
  (h : 3 * (x + y + z) = x^2 + y^2 + z^2) : 
  let f := x*y + x*z + y*z
  ∃ (M m : ℝ), 
    (∀ a b c : ℝ, 3*(a + b + c) = a^2 + b^2 + c^2 → a*b + a*c + b*c ≤ M) ∧
    (∀ a b c : ℝ, 3*(a + b + c) = a^2 + b^2 + c^2 → m ≤ a*b + a*c + b*c) ∧
    f ≤ M ∧ 
    m ≤ f ∧
    M = 27 ∧ 
    m = -9/8 ∧ 
    M + 5*m = 126/8 :=
by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_product_bounds_l3876_387686


namespace NUMINAMATH_CALUDE_f_ratio_is_integer_l3876_387658

/-- Sequence a_n defined recursively -/
def a (r s : ℕ+) : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => r * a r s (n + 1) + s * a r s n

/-- Product f_n of the first n terms of a_n -/
def f (r s : ℕ+) : ℕ → ℕ
  | 0 => 1
  | (n + 1) => f r s n * a r s (n + 1)

/-- Main theorem: f_n / (f_k * f_(n-k)) is an integer for 0 < k < n -/
theorem f_ratio_is_integer (r s : ℕ+) (n k : ℕ) (h1 : 0 < k) (h2 : k < n) :
  ∃ m : ℕ, f r s n = m * (f r s k * f r s (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_f_ratio_is_integer_l3876_387658


namespace NUMINAMATH_CALUDE_french_exam_vocab_study_l3876_387611

/-- Represents the French exam vocabulary problem -/
theorem french_exam_vocab_study (total_words : ℕ) (recall_rate : ℚ) (guess_rate : ℚ) (target_score : ℚ) :
  let min_words : ℕ := 712
  total_words = 800 ∧ recall_rate = 1 ∧ guess_rate = 1/10 ∧ target_score = 9/10 →
  (↑min_words : ℚ) + guess_rate * (total_words - min_words) ≥ target_score * total_words ∧
  ∀ (x : ℕ), x < min_words →
    (↑x : ℚ) + guess_rate * (total_words - x) < target_score * total_words :=
by sorry

end NUMINAMATH_CALUDE_french_exam_vocab_study_l3876_387611


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3876_387682

theorem min_value_of_expression :
  ∃ (x : ℝ), (8 - x) * (6 - x) * (8 + x) * (6 + x) = -196 ∧
  ∀ (y : ℝ), (8 - y) * (6 - y) * (8 + y) * (6 + y) ≥ -196 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3876_387682


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l3876_387600

/-- The x-coordinate of the intersection point of y = 9 / (x^2 + 3) and x + y = 3 is 0 -/
theorem intersection_x_coordinate : ∃ y : ℝ, 
  y = 9 / (0^2 + 3) ∧ 0 + y = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l3876_387600


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3876_387648

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 8 * x + 1 < 0 ↔ -1/3 < x ∧ x < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3876_387648


namespace NUMINAMATH_CALUDE_mn_gcd_lcm_equation_l3876_387695

theorem mn_gcd_lcm_equation (m n : ℕ+) :
  m * n = (Nat.gcd m n)^2 + Nat.lcm m n →
  (m = 2 ∧ n = 4) ∨ (m = 4 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_mn_gcd_lcm_equation_l3876_387695


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_plus_constant_l3876_387667

theorem no_solution_absolute_value_plus_constant :
  ∀ x : ℝ, ¬(|5*x| + 7 = 0) :=
sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_plus_constant_l3876_387667


namespace NUMINAMATH_CALUDE_jasons_betta_fish_count_jasons_betta_fish_count_is_five_l3876_387659

/-- The number of betta fish Jason has, given:
  1. The moray eel eats 20 guppies per day.
  2. Each betta fish eats 7 guppies per day.
  3. The total number of guppies needed per day is 55. -/
theorem jasons_betta_fish_count : ℕ :=
  let moray_eel_guppies : ℕ := 20
  let betta_fish_guppies_per_day : ℕ := 7
  let total_guppies_per_day : ℕ := 55
  let betta_fish_count := (total_guppies_per_day - moray_eel_guppies) / betta_fish_guppies_per_day
  5

/-- Proof that Jason has 5 betta fish -/
theorem jasons_betta_fish_count_is_five : jasons_betta_fish_count = 5 := by
  sorry

end NUMINAMATH_CALUDE_jasons_betta_fish_count_jasons_betta_fish_count_is_five_l3876_387659


namespace NUMINAMATH_CALUDE_smallest_student_group_l3876_387698

theorem smallest_student_group (n : ℕ) : 
  (n % 6 = 3) ∧ 
  (n % 7 = 4) ∧ 
  (n % 8 = 5) ∧ 
  (n % 9 = 2) ∧ 
  (∀ m : ℕ, m < n → ¬(m % 6 = 3 ∧ m % 7 = 4 ∧ m % 8 = 5 ∧ m % 9 = 2)) → 
  n = 765 := by
sorry

end NUMINAMATH_CALUDE_smallest_student_group_l3876_387698


namespace NUMINAMATH_CALUDE_smallest_even_integer_abs_inequality_l3876_387619

theorem smallest_even_integer_abs_inequality :
  ∃ (x : ℤ), 
    (∀ (y : ℤ), (y % 2 = 0 ∧ |3*y - 4| ≤ 20) → x ≤ y) ∧
    (x % 2 = 0) ∧
    (|3*x - 4| ≤ 20) ∧
    x = -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_even_integer_abs_inequality_l3876_387619


namespace NUMINAMATH_CALUDE_cos_330_degrees_l3876_387664

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l3876_387664


namespace NUMINAMATH_CALUDE_base_5_representation_of_89_l3876_387652

def to_base_5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: to_base_5 (n / 5)

theorem base_5_representation_of_89 :
  to_base_5 89 = [4, 2, 3] :=
by sorry

end NUMINAMATH_CALUDE_base_5_representation_of_89_l3876_387652


namespace NUMINAMATH_CALUDE_orange_tree_problem_l3876_387624

theorem orange_tree_problem (trees : ℕ) (picked_fraction : ℚ) (remaining : ℕ) :
  trees = 8 →
  picked_fraction = 2 / 5 →
  remaining = 960 →
  ∃ (initial : ℕ), initial = 200 ∧ 
    trees * (initial - picked_fraction * initial) = remaining :=
by sorry

end NUMINAMATH_CALUDE_orange_tree_problem_l3876_387624


namespace NUMINAMATH_CALUDE_negation_equivalence_l3876_387647

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3876_387647


namespace NUMINAMATH_CALUDE_angle_RPQ_measure_l3876_387694

-- Define the points
variable (P Q R S : Point)

-- Define the angle measure function
variable (angle_measure : Point → Point → Point → ℝ)

-- Define the conditions
variable (h1 : angle_measure S Q P = angle_measure P Q R)
variable (h2 : QP = PR)
variable (h3 : angle_measure R S Q = 3 * y)
variable (h4 : angle_measure R P Q = 2 * y)
variable (h5 : P ∈ line R S)

-- Define y as a real number
variable (y : ℝ)

-- State the theorem
theorem angle_RPQ_measure :
  angle_measure R P Q = 72 := by sorry

end NUMINAMATH_CALUDE_angle_RPQ_measure_l3876_387694


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3876_387685

/-- Represents the speed of a swimmer in various conditions -/
structure SwimmerSpeed where
  stillWater : ℝ
  stream : ℝ

/-- Calculates the effective speed of the swimmer -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.stillWater + s.stream else s.stillWater - s.stream

/-- Theorem: Given the conditions, the swimmer's speed in still water is 5 km/h -/
theorem swimmer_speed_in_still_water
  (s : SwimmerSpeed)
  (h1 : effectiveSpeed s true * 3 = 18)  -- Downstream condition
  (h2 : effectiveSpeed s false * 3 = 12) -- Upstream condition
  : s.stillWater = 5 := by
  sorry


end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3876_387685


namespace NUMINAMATH_CALUDE_highlight_film_average_time_l3876_387668

/-- Given the footage times for 5 players, prove that the average time each player gets in the highlight film is 2 minutes -/
theorem highlight_film_average_time (point_guard shooting_guard small_forward power_forward center : ℕ) 
  (h1 : point_guard = 130)
  (h2 : shooting_guard = 145)
  (h3 : small_forward = 85)
  (h4 : power_forward = 60)
  (h5 : center = 180) :
  (point_guard + shooting_guard + small_forward + power_forward + center) / (5 * 60) = 2 := by
  sorry

end NUMINAMATH_CALUDE_highlight_film_average_time_l3876_387668


namespace NUMINAMATH_CALUDE_divisor_sum_theorem_l3876_387683

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ := (a * (r^(n+1) - 1)) / (r - 1)

theorem divisor_sum_theorem (k m : ℕ) :
  (sum_of_geometric_series 1 2 k) * (sum_of_geometric_series 1 5 m) = 930 →
  k + m = 6 := by
sorry

end NUMINAMATH_CALUDE_divisor_sum_theorem_l3876_387683


namespace NUMINAMATH_CALUDE_fraction_value_l3876_387643

/-- Represents the numerator of the fraction as a function of k -/
def numerator (k : ℕ) : ℕ := 10^k + 6 * (10^k - 1) / 9

/-- Represents the denominator of the fraction as a function of k -/
def denominator (k : ℕ) : ℕ := 60 * (10^k - 1) / 9 + 4

/-- The main theorem stating that the fraction is always 1/4 for any positive k -/
theorem fraction_value (k : ℕ) (h : k > 0) : 
  (numerator k : ℚ) / (denominator k : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3876_387643


namespace NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l3876_387688

theorem alcohol_percentage_after_dilution :
  let initial_volume : ℝ := 15
  let initial_alcohol_percentage : ℝ := 20
  let added_water : ℝ := 2
  let initial_alcohol_volume : ℝ := initial_volume * (initial_alcohol_percentage / 100)
  let new_total_volume : ℝ := initial_volume + added_water
  let new_alcohol_percentage : ℝ := (initial_alcohol_volume / new_total_volume) * 100
  ∀ ε > 0, |new_alcohol_percentage - 17.65| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l3876_387688


namespace NUMINAMATH_CALUDE_heine_biscuits_l3876_387699

/-- The number of biscuits Mrs. Heine needs to buy for her dogs -/
def total_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog

/-- Theorem stating that Mrs. Heine needs to buy 6 biscuits -/
theorem heine_biscuits : total_biscuits 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_heine_biscuits_l3876_387699


namespace NUMINAMATH_CALUDE_remove_matches_no_rectangle_l3876_387672

-- Define the structure of the grid
def Grid := List (List Bool)

-- Define a function to check if a grid contains a rectangle
def containsRectangle (grid : Grid) : Bool := sorry

-- Define the initial 4x4 grid
def initialGrid : Grid := sorry

-- Define a function to remove matches from the grid
def removeMatches (grid : Grid) (numToRemove : Nat) : Grid := sorry

-- Theorem statement
theorem remove_matches_no_rectangle :
  ∃ (finalGrid : Grid),
    (removeMatches initialGrid 11 = finalGrid) ∧
    (containsRectangle finalGrid = false) := by
  sorry

end NUMINAMATH_CALUDE_remove_matches_no_rectangle_l3876_387672


namespace NUMINAMATH_CALUDE_chocolate_boxes_given_away_tom_chocolate_boxes_l3876_387635

theorem chocolate_boxes_given_away (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) : ℕ :=
  let total_pieces := total_boxes * pieces_per_box
  let pieces_given_away := total_pieces - pieces_left
  pieces_given_away / pieces_per_box

theorem tom_chocolate_boxes :
  chocolate_boxes_given_away 14 3 18 = 8 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_boxes_given_away_tom_chocolate_boxes_l3876_387635


namespace NUMINAMATH_CALUDE_jeremy_watermelon_consumption_l3876_387629

/-- The number of watermelons Jeremy eats per week, given the total number of watermelons,
    the number of weeks they last, and the number given away each week. -/
def watermelons_eaten_per_week (total : ℕ) (weeks : ℕ) (given_away_per_week : ℕ) : ℕ :=
  (total - weeks * given_away_per_week) / weeks

/-- Theorem stating that given 30 watermelons lasting 6 weeks, 
    with 2 given away each week, Jeremy eats 3 watermelons per week. -/
theorem jeremy_watermelon_consumption :
  watermelons_eaten_per_week 30 6 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_watermelon_consumption_l3876_387629


namespace NUMINAMATH_CALUDE_cara_card_is_five_l3876_387640

def is_valid_sequence (a b c d : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧ a + b + c + d = 20

def alan_statement (a : ℕ) : Prop :=
  ∃ b c d, is_valid_sequence a b c d ∧
  ∃ b' c' d', b' ≠ b ∧ is_valid_sequence a b' c' d'

def bella_statement (a b : ℕ) : Prop :=
  ∃ c d, is_valid_sequence a b c d ∧
  ∃ c' d', c' ≠ c ∧ is_valid_sequence a b c' d'

def cara_statement (a b c : ℕ) : Prop :=
  ∃ d, is_valid_sequence a b c d ∧
  ∃ d', d' ≠ d ∧ is_valid_sequence a b c d'

def david_statement (a b c d : ℕ) : Prop :=
  is_valid_sequence a b c d ∧
  ∃ a' b' c', a' ≠ a ∧ is_valid_sequence a' b' c' d

theorem cara_card_is_five :
  ∀ a b c d : ℕ,
    is_valid_sequence a b c d →
    alan_statement a →
    bella_statement a b →
    cara_statement a b c →
    david_statement a b c d →
    c = 5 := by
  sorry

end NUMINAMATH_CALUDE_cara_card_is_five_l3876_387640


namespace NUMINAMATH_CALUDE_walters_pocket_percentage_l3876_387656

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of pennies Walter has -/
def num_pennies : ℕ := 2

/-- The number of nickels Walter has -/
def num_nickels : ℕ := 3

/-- The number of dimes Walter has -/
def num_dimes : ℕ := 2

/-- The total value of coins in Walter's pocket in cents -/
def total_value : ℕ := 
  num_pennies * penny_value + num_nickels * nickel_value + num_dimes * dime_value

/-- The percentage of one dollar that Walter has in his pocket -/
theorem walters_pocket_percentage :
  (total_value : ℚ) / 100 * 100 = 37 := by sorry

end NUMINAMATH_CALUDE_walters_pocket_percentage_l3876_387656


namespace NUMINAMATH_CALUDE_power_inequality_l3876_387657

theorem power_inequality (x y a : ℝ) (hx : 0 < x) (hy : x < y) (hy1 : y < 1) (ha : 0 < a) (ha1 : a < 1) :
  x^a < y^a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3876_387657


namespace NUMINAMATH_CALUDE_z_value_for_given_w_and_v_l3876_387610

/-- Given a relationship between z, w, and v, prove that z equals 7.5 when w = 4 and v = 8 -/
theorem z_value_for_given_w_and_v (k : ℝ) :
  (3 * 15 = k * 4 / 2^2) →  -- Initial condition
  (∀ z w v : ℝ, 3 * z = k * v / w^2) →  -- General relationship
  ∃ z : ℝ, (3 * z = k * 8 / 4^2) ∧ z = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_z_value_for_given_w_and_v_l3876_387610


namespace NUMINAMATH_CALUDE_arithmetic_square_root_problem_l3876_387638

theorem arithmetic_square_root_problem (n : ℝ) (x : ℝ) 
  (h1 : n > 0) 
  (h2 : Real.sqrt n = x + 1) 
  (h3 : Real.sqrt n = 2*x - 4) : 
  Real.sqrt n = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_problem_l3876_387638


namespace NUMINAMATH_CALUDE_twelfth_term_value_l3876_387631

-- Define the sequence
def a (n : ℕ) : ℚ := n / (n^2 + 1) * (-1)^(n+1)

-- State the theorem
theorem twelfth_term_value : a 12 = -12 / 145 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_value_l3876_387631


namespace NUMINAMATH_CALUDE_solve_watermelon_problem_l3876_387627

def watermelon_problem (michael_weight : ℝ) (clay_multiplier : ℝ) (john_fraction : ℝ) : Prop :=
  let clay_weight := michael_weight * clay_multiplier
  let john_weight := clay_weight * john_fraction
  john_weight = 12

theorem solve_watermelon_problem :
  watermelon_problem 8 3 (1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_watermelon_problem_l3876_387627


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3876_387634

-- Define the plane
variable (Plane : Type)

-- Define points in the plane
variable (O P A B P1 P2 : Plane)

-- Define the angle
variable (angle : Plane → Plane → Plane → Prop)

-- Define the property of being inside an angle
variable (inside_angle : Plane → Plane → Plane → Plane → Prop)

-- Define the property of a point being on a line
variable (on_line : Plane → Plane → Plane → Prop)

-- Define the reflection of a point over a line
variable (reflect : Plane → Plane → Plane → Plane)

-- Define the perimeter of a triangle
variable (perimeter : Plane → Plane → Plane → ℝ)

-- Define the theorem
theorem min_perimeter_triangle 
  (h_acute : angle O A B)
  (h_inside : inside_angle O A B P)
  (h_P1 : P1 = reflect O A P)
  (h_P2 : P2 = reflect O B P)
  (h_A_on_side : on_line O A A)
  (h_B_on_side : on_line O B B)
  (h_A_on_P1P2 : on_line P1 P2 A)
  (h_B_on_P1P2 : on_line P1 P2 B) :
  ∀ A' B', on_line O A A' → on_line O B B' → 
    perimeter P A B ≤ perimeter P A' B' :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3876_387634


namespace NUMINAMATH_CALUDE_cube_pyramid_plane_pairs_l3876_387650

/-- A solid formed by a cube and a pyramid --/
structure CubePyramidSolid where
  cube_edges : Finset (Fin 12)
  pyramid_edges : Finset (Fin 5)

/-- Function to count pairs of edges that determine a plane --/
def count_plane_determining_pairs (solid : CubePyramidSolid) : ℕ :=
  sorry

/-- Theorem stating the number of edge pairs determining a plane --/
theorem cube_pyramid_plane_pairs :
  ∀ (solid : CubePyramidSolid),
  count_plane_determining_pairs solid = 82 :=
sorry

end NUMINAMATH_CALUDE_cube_pyramid_plane_pairs_l3876_387650


namespace NUMINAMATH_CALUDE_battery_mass_problem_l3876_387691

theorem battery_mass_problem (x y : ℝ) 
  (eq1 : 2 * x + 2 * y = 72)
  (eq2 : 3 * x + 2 * y = 96) :
  x = 24 := by
sorry

end NUMINAMATH_CALUDE_battery_mass_problem_l3876_387691


namespace NUMINAMATH_CALUDE_element_in_set_l3876_387673

theorem element_in_set : ∀ (M : Set ℕ), M = {0, 1, 2} → 0 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l3876_387673


namespace NUMINAMATH_CALUDE_function_graph_overlap_l3876_387615

theorem function_graph_overlap (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a^(x/2) = 2^(-x/2)) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_graph_overlap_l3876_387615


namespace NUMINAMATH_CALUDE_statement_analysis_l3876_387612

-- Define the types of statements
inductive StatementType
  | Universal
  | Existential

-- Define a structure to represent a statement
structure Statement where
  content : String
  type : StatementType
  isTrue : Bool

-- Define the statements
def statement1 : Statement := {
  content := "The diagonals of a square are perpendicular bisectors of each other",
  type := StatementType.Universal,
  isTrue := true
}

def statement2 : Statement := {
  content := "All Chinese people speak Chinese",
  type := StatementType.Universal,
  isTrue := false
}

def statement3 : Statement := {
  content := "Some numbers are greater than their squares",
  type := StatementType.Existential,
  isTrue := true
}

def statement4 : Statement := {
  content := "Some real numbers have irrational square roots",
  type := StatementType.Existential,
  isTrue := true
}

-- Theorem to prove
theorem statement_analysis : 
  (statement1.type = StatementType.Universal ∧ statement1.isTrue) ∧
  (statement2.type = StatementType.Universal ∧ ¬statement2.isTrue) ∧
  (statement3.type = StatementType.Existential ∧ statement3.isTrue) ∧
  (statement4.type = StatementType.Existential ∧ statement4.isTrue) := by
  sorry


end NUMINAMATH_CALUDE_statement_analysis_l3876_387612


namespace NUMINAMATH_CALUDE_fifteenth_prime_l3876_387601

def is_prime (n : ℕ) : Prop := sorry

def nth_prime (n : ℕ) : ℕ := sorry

theorem fifteenth_prime :
  (nth_prime 8 = 19) → (nth_prime 15 = 47) := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_prime_l3876_387601


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l3876_387609

theorem stratified_sampling_problem (total_students : ℕ) (sample_size : ℕ) (major_c_students : ℕ) :
  total_students = 1000 →
  sample_size = 40 →
  major_c_students = 400 →
  (major_c_students * sample_size) / total_students = 16 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l3876_387609


namespace NUMINAMATH_CALUDE_exchange_indifference_l3876_387605

/-- Represents the number of rubles a tourist plans to exchange. -/
def rubles : ℕ := 140

/-- Represents the exchange rate (in tugriks) for the first office. -/
def rate1 : ℕ := 3000

/-- Represents the exchange rate (in tugriks) for the second office. -/
def rate2 : ℕ := 2950

/-- Represents the commission fee (in tugriks) for the first office. -/
def commission : ℕ := 7000

theorem exchange_indifference :
  rate1 * rubles - commission = rate2 * rubles :=
by sorry

end NUMINAMATH_CALUDE_exchange_indifference_l3876_387605


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l3876_387623

theorem max_value_of_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  (∀ x' y', -6 ≤ x' ∧ x' ≤ -3 → 3 ≤ y' ∧ y' ≤ 5 → (x' - y') / y' ≤ (x - y) / y) →
  (x - y) / y = -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l3876_387623


namespace NUMINAMATH_CALUDE_sales_solution_l3876_387625

def sales_problem (month1 month3 month4 month5 month6 average_sale : ℕ) : Prop :=
  ∃ (month2 : ℕ),
    (month1 + month2 + month3 + month4 + month5 + month6) / 6 = average_sale ∧
    month2 = 6927

theorem sales_solution :
  sales_problem 6435 6855 7230 6562 7991 7000 :=
sorry

end NUMINAMATH_CALUDE_sales_solution_l3876_387625


namespace NUMINAMATH_CALUDE_rectangular_pen_max_area_l3876_387616

/-- The perimeter of the rectangular pen -/
def perimeter : ℝ := 60

/-- The maximum possible area of a rectangular pen with the given perimeter -/
def max_area : ℝ := 225

/-- Theorem: The maximum area of a rectangular pen with a perimeter of 60 feet is 225 square feet -/
theorem rectangular_pen_max_area : 
  ∀ (width height : ℝ), 
  width > 0 → height > 0 → 
  2 * (width + height) = perimeter → 
  width * height ≤ max_area :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_pen_max_area_l3876_387616


namespace NUMINAMATH_CALUDE_delta_power_of_three_l3876_387692

/-- The forward difference operator -/
def forwardDifference (f : ℕ → ℝ) : ℕ → ℝ :=
  fun n => f (n + 1) - f n

/-- Iterated forward difference operator -/
def iteratedForwardDifference (f : ℕ → ℝ) : ℕ → ℕ → ℝ
  | 0, n => f n
  | (k + 1), n => forwardDifference (iteratedForwardDifference f k) n

/-- The main theorem -/
theorem delta_power_of_three (k : ℕ) (n : ℕ) (h : k ≥ 1) :
  iteratedForwardDifference (fun n => 3^n) k n = 2^k * 3^n := by
  sorry


end NUMINAMATH_CALUDE_delta_power_of_three_l3876_387692


namespace NUMINAMATH_CALUDE_midpoint_square_area_l3876_387662

theorem midpoint_square_area (A : ℝ) (h : A = 144) : 
  let s := Real.sqrt A
  let midpoint_side := Real.sqrt ((s/2)^2 + (s/2)^2)
  let midpoint_area := midpoint_side^2
  midpoint_area = 72 := by sorry

end NUMINAMATH_CALUDE_midpoint_square_area_l3876_387662


namespace NUMINAMATH_CALUDE_book_cost_price_l3876_387644

/-- The cost price of a book, given that selling it at 9% profit instead of 9% loss brings Rs 9 more -/
theorem book_cost_price (price : ℝ) : 
  (price * 1.09 - price * 0.91 = 9) → price = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l3876_387644


namespace NUMINAMATH_CALUDE_our_equation_is_linear_l3876_387620

/-- Definition of a linear equation in two variables -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The specific equation we want to prove is linear -/
def our_equation (x y : ℝ) : ℝ := x + y - 5

theorem our_equation_is_linear :
  is_linear_equation our_equation :=
sorry

end NUMINAMATH_CALUDE_our_equation_is_linear_l3876_387620


namespace NUMINAMATH_CALUDE_closest_point_l3876_387613

/-- The vector that depends on the scalar parameter s -/
def u (s : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3 + 4*s
  | 1 => -2 - 6*s
  | 2 => 1 + 2*s

/-- The constant vector b -/
def b : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 1
  | 1 => 5
  | 2 => -3

/-- The direction vector of the line -/
def v : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 4
  | 1 => -6
  | 2 => 2

/-- Theorem stating that s = 13/8 minimizes the distance between u and b -/
theorem closest_point (s : ℝ) :
  (∀ i, (u s i - b i) * v i = 0) ↔ s = 13/8 := by
  sorry

end NUMINAMATH_CALUDE_closest_point_l3876_387613


namespace NUMINAMATH_CALUDE_regular_polygon_tiling_l3876_387696

theorem regular_polygon_tiling (x y z : ℕ) (hx : x > 2) (hy : y > 2) (hz : z > 2) :
  (((x - 2 : ℝ) / x + (y - 2 : ℝ) / y + (z - 2 : ℝ) / z) = 2) →
  (1 / x + 1 / y + 1 / z : ℝ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_tiling_l3876_387696


namespace NUMINAMATH_CALUDE_joans_kittens_l3876_387690

theorem joans_kittens (initial_kittens : ℕ) (given_away : ℕ) (remaining : ℕ) 
  (h1 : given_away = 2) 
  (h2 : remaining = 6) 
  (h3 : initial_kittens = remaining + given_away) : initial_kittens = 8 :=
by sorry

end NUMINAMATH_CALUDE_joans_kittens_l3876_387690


namespace NUMINAMATH_CALUDE_intersection_A_B_l3876_387642

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | (x + 4) * (x - 1) < 0}
def B : Set ℝ := {x : ℝ | x^2 - 2*x = 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3876_387642


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3876_387681

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| + 3 * x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3876_387681


namespace NUMINAMATH_CALUDE_magnitude_of_scaled_complex_l3876_387626

theorem magnitude_of_scaled_complex (z : ℂ) :
  z = 3 - 2 * Complex.I →
  Complex.abs (-1/3 * z) = Real.sqrt 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_scaled_complex_l3876_387626


namespace NUMINAMATH_CALUDE_teacher_health_survey_l3876_387671

theorem teacher_health_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_high_bp : high_bp = 90)
  (h_heart_trouble : heart_trouble = 60)
  (h_both : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_teacher_health_survey_l3876_387671


namespace NUMINAMATH_CALUDE_cottage_cheese_production_l3876_387618

/-- Represents the fat content balance in milk processing -/
def fat_balance (milk_mass : ℝ) (milk_fat : ℝ) (cheese_fat : ℝ) (whey_fat : ℝ) (cheese_mass : ℝ) : Prop :=
  milk_mass * milk_fat = cheese_mass * cheese_fat + (milk_mass - cheese_mass) * whey_fat

/-- Proves the amount of cottage cheese produced from milk -/
theorem cottage_cheese_production (milk_mass : ℝ) (milk_fat : ℝ) (cheese_fat : ℝ) (whey_fat : ℝ) 
  (h_milk_mass : milk_mass = 1)
  (h_milk_fat : milk_fat = 0.05)
  (h_cheese_fat : cheese_fat = 0.155)
  (h_whey_fat : whey_fat = 0.005) :
  ∃ cheese_mass : ℝ, cheese_mass = 0.3 ∧ fat_balance milk_mass milk_fat cheese_fat whey_fat cheese_mass :=
by
  sorry

#check cottage_cheese_production

end NUMINAMATH_CALUDE_cottage_cheese_production_l3876_387618


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3876_387630

theorem fractional_equation_solution : 
  ∃ x : ℝ, (x ≠ 0 ∧ x ≠ -1) ∧ (6 / (x + 1) = (x + 5) / (x * (x + 1))) ∧ x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3876_387630


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3876_387606

theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) :
  train_length = 250 →
  crossing_time = 25 →
  train_speed_kmh = 57.6 →
  ∃ (bridge_length : ℝ),
    bridge_length = 150 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3876_387606


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l3876_387633

theorem larger_solution_of_quadratic (x : ℝ) : 
  (2 * x^2 - 14 * x - 84 = 0) → (∃ y : ℝ, 2 * y^2 - 14 * y - 84 = 0 ∧ y ≠ x) → 
  (x = 14 ∨ x = -3) → (x = 14 ∨ (x = -3 ∧ 14 > x)) :=
sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l3876_387633


namespace NUMINAMATH_CALUDE_child_ticket_cost_l3876_387663

theorem child_ticket_cost (adult_price : ℕ) (total_attendees : ℕ) (total_revenue : ℕ) (child_attendees : ℕ) : 
  adult_price = 60 →
  total_attendees = 280 →
  total_revenue = 14000 →
  child_attendees = 80 →
  ∃ (child_price : ℕ), child_price = 25 ∧
    total_revenue = adult_price * (total_attendees - child_attendees) + child_price * child_attendees :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l3876_387663


namespace NUMINAMATH_CALUDE_parabola_coefficient_l3876_387603

/-- Given a parabola y = ax^2 + bx + c with vertex (p, -p) and y-intercept (0, p), where p ≠ 0, 
    the value of b is -4. -/
theorem parabola_coefficient (a b c p : ℝ) : p ≠ 0 → 
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ (x - p)^2 = (y + p) / a) → 
  c = p → 
  b = -4 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l3876_387603


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3876_387697

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (h1 : leg = 15) (h2 : angle = 45) :
  let hypotenuse := leg * Real.sqrt 2
  hypotenuse = 15 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3876_387697


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l3876_387602

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l3876_387602


namespace NUMINAMATH_CALUDE_zeros_of_f_l3876_387675

noncomputable section

-- Define the piecewise function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1/2 then x - 2/x else x^2 + 2*x + a - 1

-- Define the set of zeros for f
def zeros (a : ℝ) : Set ℝ :=
  {x : ℝ | f a x = 0}

-- Theorem statement
theorem zeros_of_f (a : ℝ) (h : a > 0) :
  zeros a = 
    if a > 2 then {Real.sqrt 2}
    else if a = 2 then {Real.sqrt 2, -1}
    else {Real.sqrt 2, -1 + Real.sqrt (2-a), -1 - Real.sqrt (2-a)} :=
by sorry

end

end NUMINAMATH_CALUDE_zeros_of_f_l3876_387675


namespace NUMINAMATH_CALUDE_age_difference_l3876_387646

theorem age_difference (alice_age carol_age betty_age : ℕ) : 
  carol_age = 5 * alice_age →
  carol_age = 2 * betty_age →
  betty_age = 6 →
  carol_age - alice_age = 10 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3876_387646


namespace NUMINAMATH_CALUDE_sum_of_perpendiculars_eq_twice_side_l3876_387628

/-- A square with side length s -/
structure Square (s : ℝ) where
  side : s > 0

/-- A point inside a square -/
structure PointInSquare (s : ℝ) where
  x : ℝ
  y : ℝ
  x_bound : 0 ≤ x ∧ x ≤ s
  y_bound : 0 ≤ y ∧ y ≤ s

/-- The sum of perpendiculars from a point to the sides of a square -/
def sumOfPerpendiculars (s : ℝ) (p : PointInSquare s) : ℝ :=
  p.x + (s - p.x) + p.y + (s - p.y)

/-- Theorem: The sum of perpendiculars from any point inside a square to its sides
    is equal to twice the side length of the square -/
theorem sum_of_perpendiculars_eq_twice_side {s : ℝ} (sq : Square s) (p : PointInSquare s) :
  sumOfPerpendiculars s p = 2 * s := by
  sorry


end NUMINAMATH_CALUDE_sum_of_perpendiculars_eq_twice_side_l3876_387628


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3876_387645

theorem logarithm_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - 5 ^ (Real.log 3 / Real.log 5) = -1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3876_387645
