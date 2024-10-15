import Mathlib

namespace NUMINAMATH_CALUDE_current_speed_l1129_112923

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 10) :
  ∃ (current_speed : ℝ), current_speed = 2.5 ∧ 
    speed_with_current = speed_against_current + 2 * current_speed :=
by sorry

end NUMINAMATH_CALUDE_current_speed_l1129_112923


namespace NUMINAMATH_CALUDE_square_calculation_l1129_112957

theorem square_calculation :
  (41 ^ 2 = 40 ^ 2 + 81) ∧ (39 ^ 2 = 40 ^ 2 - 79) := by
  sorry

end NUMINAMATH_CALUDE_square_calculation_l1129_112957


namespace NUMINAMATH_CALUDE_mass_of_copper_sulfate_pentahydrate_l1129_112961

-- Define the constants
def volume : ℝ := 0.5 -- in L
def concentration : ℝ := 1 -- in mol/L
def molar_mass : ℝ := 250 -- in g/mol

-- Theorem statement
theorem mass_of_copper_sulfate_pentahydrate (volume concentration molar_mass : ℝ) : 
  volume * concentration * molar_mass = 125 := by
  sorry

#check mass_of_copper_sulfate_pentahydrate

end NUMINAMATH_CALUDE_mass_of_copper_sulfate_pentahydrate_l1129_112961


namespace NUMINAMATH_CALUDE_joe_age_difference_l1129_112969

theorem joe_age_difference (joe_age : ℕ) (james_age : ℕ) : joe_age = 22 → 2 * (joe_age + 8) = 3 * (james_age + 8) → joe_age - james_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_joe_age_difference_l1129_112969


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_l1129_112937

theorem pencil_eraser_cost : ∃ (p e : ℕ), 
  15 * p + 5 * e = 200 ∧ 
  p ≥ 2 * e ∧ 
  p + e = 14 := by
sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_l1129_112937


namespace NUMINAMATH_CALUDE_exists_h_not_divisible_l1129_112936

theorem exists_h_not_divisible : ∃ h : ℝ, ∀ n : ℕ, 
  ¬(⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
sorry

end NUMINAMATH_CALUDE_exists_h_not_divisible_l1129_112936


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1129_112989

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (9 - 2 * x) = 8 → x = -55 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1129_112989


namespace NUMINAMATH_CALUDE_total_food_amount_l1129_112942

-- Define the number of boxes
def num_boxes : ℕ := 388

-- Define the amount of food per box in kilograms
def food_per_box : ℕ := 2

-- Theorem to prove the total amount of food
theorem total_food_amount : num_boxes * food_per_box = 776 := by
  sorry

end NUMINAMATH_CALUDE_total_food_amount_l1129_112942


namespace NUMINAMATH_CALUDE_physics_marks_l1129_112950

theorem physics_marks (P C M : ℝ) 
  (h1 : (P + C + M) / 3 = 85)
  (h2 : (P + M) / 2 = 90)
  (h3 : (P + C) / 2 = 70) :
  P = 65 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l1129_112950


namespace NUMINAMATH_CALUDE_arithmetic_sequence_l1129_112951

theorem arithmetic_sequence (a b c : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃ x : ℝ, bx^2 - 4*b*x + 2*(a+c) = 0 ∧ (∀ y : ℝ, bx^2 - 4*b*x + 2*(a+c) = 0 → y = x)) :
  b - a = c - b := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_l1129_112951


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l1129_112964

/-- Given that x₁ and x₂ are extremal points of f(x) = (1/3)ax³ - (1/2)ax² - x,
    prove that the line passing through A(x₁, 1/x₁) and B(x₂, 1/x₂)
    intersects the ellipse x²/2 + y² = 1 --/
theorem line_intersects_ellipse (a : ℝ) (x₁ x₂ : ℝ) :
  (x₁ ≠ x₂) →
  (∀ x, (a*x^2 - a*x - 1 = 0) ↔ (x = x₁ ∨ x = x₂)) →
  ∃ x y : ℝ, (y - 1/x₁ = (1/x₂ - 1/x₁)/(x₂ - x₁) * (x - x₁)) ∧
             (x^2/2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l1129_112964


namespace NUMINAMATH_CALUDE_alice_probability_after_three_turns_l1129_112984

-- Define the probabilities
def alice_to_bob : ℚ := 2/3
def alice_keeps : ℚ := 1/3
def bob_to_alice : ℚ := 1/3
def bob_keeps : ℚ := 2/3

-- Define the game state after three turns
def alice_has_ball_after_three_turns : ℚ :=
  alice_to_bob * bob_to_alice +
  alice_to_bob * bob_keeps * bob_to_alice +
  alice_keeps * alice_to_bob * bob_to_alice +
  alice_keeps * alice_keeps

-- Theorem statement
theorem alice_probability_after_three_turns :
  alice_has_ball_after_three_turns = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_alice_probability_after_three_turns_l1129_112984


namespace NUMINAMATH_CALUDE_sine_cosine_identity_l1129_112904

theorem sine_cosine_identity : 
  Real.sin (50 * π / 180) * Real.cos (170 * π / 180) - 
  Real.sin (40 * π / 180) * Real.sin (170 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_identity_l1129_112904


namespace NUMINAMATH_CALUDE_function_max_value_l1129_112933

theorem function_max_value (x : ℝ) (h : x < 5/4) :
  (4 * x - 2 + 1 / (4 * x - 5)) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_function_max_value_l1129_112933


namespace NUMINAMATH_CALUDE_sum_of_divisors_330_l1129_112958

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_330 : sum_of_divisors 330 = 864 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_330_l1129_112958


namespace NUMINAMATH_CALUDE_jacob_final_score_l1129_112921

/-- Represents the score for a quiz contest -/
structure QuizScore where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  correct_points : ℚ
  incorrect_points : ℚ

/-- Calculates the final score for a quiz contest -/
def final_score (qs : QuizScore) : ℚ :=
  qs.correct * qs.correct_points + qs.incorrect * qs.incorrect_points

/-- Jacob's quiz score -/
def jacob_score : QuizScore :=
  { correct := 20
    incorrect := 10
    unanswered := 5
    correct_points := 1
    incorrect_points := -1/2 }

/-- Theorem: Jacob's final score is 15 points -/
theorem jacob_final_score :
  final_score jacob_score = 15 := by sorry

end NUMINAMATH_CALUDE_jacob_final_score_l1129_112921


namespace NUMINAMATH_CALUDE_remainder_nine_333_mod_50_l1129_112975

theorem remainder_nine_333_mod_50 : 9^333 % 50 = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_nine_333_mod_50_l1129_112975


namespace NUMINAMATH_CALUDE_tenth_grader_average_score_l1129_112967

/-- Represents a chess tournament between 9th and 10th graders -/
structure ChessTournament where
  ninth_graders : ℕ
  tenth_graders : ℕ
  tournament_points : ℕ

/-- The number of 10th graders is 10 times the number of 9th graders -/
axiom tenth_grader_count (t : ChessTournament) : t.tenth_graders = 10 * t.ninth_graders

/-- Each player plays every other player exactly once -/
axiom total_games (t : ChessTournament) : t.tournament_points = (t.ninth_graders + t.tenth_graders) * (t.ninth_graders + t.tenth_graders - 1) / 2

/-- The average score of a 10th grader is 10 points -/
theorem tenth_grader_average_score (t : ChessTournament) :
  t.tournament_points / t.tenth_graders = 10 :=
sorry

end NUMINAMATH_CALUDE_tenth_grader_average_score_l1129_112967


namespace NUMINAMATH_CALUDE_ratio_q_p_l1129_112918

def total_slips : ℕ := 40
def distinct_numbers : ℕ := 10
def slips_per_number : ℕ := 4
def drawn_slips : ℕ := 4

def p : ℚ := (distinct_numbers * 1) / Nat.choose total_slips drawn_slips
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

theorem ratio_q_p : q / p = 162 := by
  sorry

end NUMINAMATH_CALUDE_ratio_q_p_l1129_112918


namespace NUMINAMATH_CALUDE_librarian_took_two_books_l1129_112970

/-- The number of books the librarian took -/
def librarian_took (total_books : ℕ) (shelves_needed : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books - (shelves_needed * books_per_shelf)

/-- Theorem stating that the librarian took 2 books -/
theorem librarian_took_two_books :
  librarian_took 14 4 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_librarian_took_two_books_l1129_112970


namespace NUMINAMATH_CALUDE_last_digit_fifth_power_l1129_112946

theorem last_digit_fifth_power (R : ℤ) : 10 ∣ (R^5 - R) := by
  sorry

end NUMINAMATH_CALUDE_last_digit_fifth_power_l1129_112946


namespace NUMINAMATH_CALUDE_market_spending_l1129_112986

theorem market_spending (total_amount mildred_spent candice_spent : ℕ) 
  (h1 : total_amount = 100)
  (h2 : mildred_spent = 25)
  (h3 : candice_spent = 35) :
  total_amount - (mildred_spent + candice_spent) = 40 := by
  sorry

end NUMINAMATH_CALUDE_market_spending_l1129_112986


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_l1129_112932

/-- The length of one side of the largest equilateral triangle created from a 78 cm string -/
def triangle_side_length : ℝ := 26

/-- The total length of the string used to create the triangle -/
def string_length : ℝ := 78

/-- Theorem stating that the length of one side of the largest equilateral triangle
    created from a 78 cm string is 26 cm -/
theorem equilateral_triangle_side (s : ℝ) :
  s = triangle_side_length ↔ s * 3 = string_length :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_l1129_112932


namespace NUMINAMATH_CALUDE_hurricane_damage_calculation_l1129_112911

/-- Calculates the total hurricane damage in Canadian dollars, including a recovery tax -/
theorem hurricane_damage_calculation (damage_usd : ℝ) (assets_cad : ℝ) (exchange_rate : ℝ) (tax_rate : ℝ) :
  damage_usd = 45000000 →
  assets_cad = 15000000 →
  exchange_rate = 1.25 →
  tax_rate = 0.1 →
  let damage_cad := damage_usd * exchange_rate + assets_cad
  let total_with_tax := damage_cad * (1 + tax_rate)
  total_with_tax = 78375000 := by
sorry

end NUMINAMATH_CALUDE_hurricane_damage_calculation_l1129_112911


namespace NUMINAMATH_CALUDE_range_of_x_l1129_112909

theorem range_of_x (x : ℝ) : 
  (|x - 1| + |x - 2| = 1) → (1 ≤ x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1129_112909


namespace NUMINAMATH_CALUDE_angies_age_equation_angie_is_eight_years_old_l1129_112953

/-- Angie's age in years -/
def angiesAge : ℕ := 8

/-- The equation representing the given condition -/
theorem angies_age_equation : 2 * angiesAge + 4 = 20 := by sorry

/-- Proof that Angie's age is 8 years old -/
theorem angie_is_eight_years_old : angiesAge = 8 := by sorry

end NUMINAMATH_CALUDE_angies_age_equation_angie_is_eight_years_old_l1129_112953


namespace NUMINAMATH_CALUDE_seventy_seven_base4_non_consecutive_digits_l1129_112903

/-- Converts a decimal number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of non-consecutive digits in a list of digits -/
def countNonConsecutiveDigits (digits : List ℕ) : ℕ :=
  sorry

theorem seventy_seven_base4_non_consecutive_digits :
  let base4Repr := toBase4 77
  countNonConsecutiveDigits base4Repr = 3 :=
by sorry

end NUMINAMATH_CALUDE_seventy_seven_base4_non_consecutive_digits_l1129_112903


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1129_112990

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 3) :
  (1/a + 1/b) ≥ 1 + 2*Real.sqrt 2/3 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 3 ∧ 1/a₀ + 1/b₀ = 1 + 2*Real.sqrt 2/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1129_112990


namespace NUMINAMATH_CALUDE_no_equidistant_points_l1129_112925

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane, represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Configuration of a circle and two parallel tangent lines -/
structure CircleWithTangents where
  circle : Circle
  tangent1 : Line
  tangent2 : Line
  d : ℝ  -- distance from circle center to each tangent

/-- Predicate for a point being equidistant from a circle and a line -/
def isEquidistant (p : ℝ × ℝ) (c : Circle) (l : Line) : Prop := sorry

/-- Main theorem: No equidistant points exist when d > r -/
theorem no_equidistant_points (config : CircleWithTangents) 
  (h : config.d > config.circle.radius) :
  ¬∃ p : ℝ × ℝ, isEquidistant p config.circle config.tangent1 ∧ 
                isEquidistant p config.circle config.tangent2 := by
  sorry

end NUMINAMATH_CALUDE_no_equidistant_points_l1129_112925


namespace NUMINAMATH_CALUDE_quadratic_factorization_and_perfect_square_discriminant_l1129_112915

/-- A quadratic expression of the form 15x^2 + ax + 15 can be factored into two linear binomial 
factors with integer coefficients, and its discriminant is a perfect square when a = 34 -/
theorem quadratic_factorization_and_perfect_square_discriminant :
  ∃ (m n p q : ℤ), 
    (15 : ℤ) * m * p = 15 ∧ 
    m * q + n * p = 34 ∧ 
    n * q = 15 ∧
    ∃ (k : ℤ), 34^2 - 4 * 15 * 15 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_and_perfect_square_discriminant_l1129_112915


namespace NUMINAMATH_CALUDE_average_score_five_subjects_l1129_112935

theorem average_score_five_subjects 
  (avg_three : ℝ) 
  (score_four : ℝ) 
  (score_five : ℝ) 
  (h1 : avg_three = 92) 
  (h2 : score_four = 90) 
  (h3 : score_five = 95) : 
  (3 * avg_three + score_four + score_five) / 5 = 92.2 := by
sorry

end NUMINAMATH_CALUDE_average_score_five_subjects_l1129_112935


namespace NUMINAMATH_CALUDE_probability_one_white_ball_l1129_112927

/-- The probability of drawing exactly one white ball when randomly selecting two balls from a bag containing 2 white and 3 black balls -/
theorem probability_one_white_ball (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) :
  total_balls = white_balls + black_balls →
  total_balls = 5 →
  white_balls = 2 →
  black_balls = 3 →
  (Nat.choose white_balls 1 * Nat.choose black_balls 1 : ℚ) / Nat.choose total_balls 2 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_white_ball_l1129_112927


namespace NUMINAMATH_CALUDE_bench_press_increase_factor_l1129_112991

theorem bench_press_increase_factor 
  (initial_weight : ℝ) 
  (injury_decrease_percent : ℝ) 
  (final_weight : ℝ) 
  (h1 : initial_weight = 500) 
  (h2 : injury_decrease_percent = 80) 
  (h3 : final_weight = 300) : 
  final_weight / (initial_weight * (1 - injury_decrease_percent / 100)) = 3 := by
sorry

end NUMINAMATH_CALUDE_bench_press_increase_factor_l1129_112991


namespace NUMINAMATH_CALUDE_orange_bin_problem_l1129_112982

theorem orange_bin_problem (initial : ℕ) (removed : ℕ) (added : ℕ) : 
  initial = 50 → removed = 40 → added = 24 → initial - removed + added = 34 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_problem_l1129_112982


namespace NUMINAMATH_CALUDE_gmat_scores_l1129_112998

theorem gmat_scores (u v : ℝ) (hu : u > 0) (hv : v > 0) (h : v / u = 1 / 3) :
  (u + v) / 2 = (2 / 3) * u := by
  sorry

end NUMINAMATH_CALUDE_gmat_scores_l1129_112998


namespace NUMINAMATH_CALUDE_powers_of_two_start_with_any_digits_l1129_112966

theorem powers_of_two_start_with_any_digits (A m : ℕ) : 
  ∃ n : ℕ+, (10 ^ m * A : ℝ) < (2 : ℝ) ^ (n : ℝ) ∧ (2 : ℝ) ^ (n : ℝ) < (10 ^ m * (A + 1) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_powers_of_two_start_with_any_digits_l1129_112966


namespace NUMINAMATH_CALUDE_balloon_problem_l1129_112917

/-- Represents the balloon counts for a person -/
structure BalloonCount where
  red : Nat
  blue : Nat

/-- Calculates the total cost of balloons -/
def totalCost (redCount blue_count : Nat) (redCost blueCost : Nat) : Nat :=
  redCount * redCost + blue_count * blueCost

/-- Theorem statement for the balloon problem -/
theorem balloon_problem 
  (fred sam dan : BalloonCount)
  (redCost blueCost : Nat)
  (h1 : fred = ⟨10, 5⟩)
  (h2 : sam = ⟨46, 20⟩)
  (h3 : dan = ⟨16, 12⟩)
  (h4 : redCost = 10)
  (h5 : blueCost = 5) :
  let totalRed := fred.red + sam.red + dan.red
  let totalBlue := fred.blue + sam.blue + dan.blue
  let totalCost := totalCost totalRed totalBlue redCost blueCost
  totalRed = 72 ∧ totalBlue = 37 ∧ totalCost = 905 := by
  sorry


end NUMINAMATH_CALUDE_balloon_problem_l1129_112917


namespace NUMINAMATH_CALUDE_gretchen_desk_work_time_l1129_112914

/-- Represents the ratio of walking time to sitting time -/
def walkingSittingRatio : ℚ := 10 / 90

/-- Represents the total walking time in minutes -/
def totalWalkingTime : ℕ := 40

/-- Represents the time spent working at the desk in hours -/
def deskWorkTime : ℚ := 6

theorem gretchen_desk_work_time :
  walkingSittingRatio * (deskWorkTime * 60) = totalWalkingTime :=
sorry

end NUMINAMATH_CALUDE_gretchen_desk_work_time_l1129_112914


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1129_112931

theorem gcd_polynomial_and_multiple (b : ℤ) (h : ∃ k : ℤ, b = 342 * k) :
  Nat.gcd (Int.natAbs (5*b^3 + b^2 + 8*b + 38)) (Int.natAbs b) = 38 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l1129_112931


namespace NUMINAMATH_CALUDE_bus_capacity_is_193_l1129_112983

/-- Represents the seating capacity of a double-decker bus -/
def double_decker_bus_capacity (lower_left : ℕ) (lower_right : ℕ) (regular_seat_capacity : ℕ)
  (priority_seats : ℕ) (priority_seat_capacity : ℕ) (upper_left : ℕ) (upper_right : ℕ)
  (upper_seat_capacity : ℕ) (upper_back : ℕ) : ℕ :=
  (lower_left + lower_right) * regular_seat_capacity +
  priority_seats * priority_seat_capacity +
  (upper_left + upper_right) * upper_seat_capacity +
  upper_back

/-- Theorem stating the total seating capacity of the given double-decker bus -/
theorem bus_capacity_is_193 :
  double_decker_bus_capacity 15 12 2 4 1 20 20 3 15 = 193 := by
  sorry


end NUMINAMATH_CALUDE_bus_capacity_is_193_l1129_112983


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1129_112987

-- Define the complex number z as (m+1)(1-i)
def z (m : ℝ) : ℂ := (m + 1) * (1 - Complex.I)

-- Theorem statement
theorem pure_imaginary_condition (m : ℝ) :
  (z m).re = 0 → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1129_112987


namespace NUMINAMATH_CALUDE_larry_initial_amount_l1129_112956

def larry_problem (initial_amount lunch_cost brother_gift current_amount : ℕ) : Prop :=
  initial_amount = lunch_cost + brother_gift + current_amount

theorem larry_initial_amount :
  ∃ (initial_amount : ℕ), larry_problem initial_amount 5 2 15 ∧ initial_amount = 22 := by
  sorry

end NUMINAMATH_CALUDE_larry_initial_amount_l1129_112956


namespace NUMINAMATH_CALUDE_quadratic_equation_root_relation_l1129_112995

theorem quadratic_equation_root_relation (m : ℝ) (hm : m ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁^2 - x₁ + m = 0 ∧ x₂^2 - x₂ + 3*m = 0 ∧ x₂ = 2*x₁) →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_relation_l1129_112995


namespace NUMINAMATH_CALUDE_hilt_book_profit_l1129_112907

/-- The difference in total amount between selling and buying books -/
def book_profit (num_books : ℕ) (buy_price sell_price : ℚ) : ℚ :=
  num_books * sell_price - num_books * buy_price

/-- Theorem stating the profit from buying and selling books -/
theorem hilt_book_profit :
  book_profit 15 11 25 = 210 := by
  sorry

end NUMINAMATH_CALUDE_hilt_book_profit_l1129_112907


namespace NUMINAMATH_CALUDE_corner_removed_cube_surface_area_l1129_112980

/-- Represents a cube with corner cubes removed -/
structure CornerRemovedCube where
  side_length : ℝ
  corner_size : ℝ

/-- Calculates the surface area of a cube with corner cubes removed -/
def surface_area (cube : CornerRemovedCube) : ℝ :=
  6 * cube.side_length^2

/-- Theorem stating that a 4x4x4 cube with corner cubes removed has surface area 96 sq.cm -/
theorem corner_removed_cube_surface_area :
  let cube : CornerRemovedCube := ⟨4, 1⟩
  surface_area cube = 96 := by
  sorry

end NUMINAMATH_CALUDE_corner_removed_cube_surface_area_l1129_112980


namespace NUMINAMATH_CALUDE_linear_function_k_value_l1129_112938

/-- A linear function passing through a specific point -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 3

/-- The point through which the function passes -/
def point : ℝ × ℝ := (2, 5)

theorem linear_function_k_value :
  ∃ k : ℝ, linear_function k (point.1) = point.2 ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l1129_112938


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l1129_112988

def total_balls : ℕ := 9
def white_balls : ℕ := 4
def black_balls : ℕ := 5
def drawn_balls : ℕ := 3

theorem ball_drawing_theorem :
  -- 1. Total number of ways to choose 3 balls from 9 balls
  Nat.choose total_balls drawn_balls = 84 ∧
  -- 2. Number of ways to choose 2 white and 1 black
  (Nat.choose white_balls 2 * Nat.choose black_balls 1) = 30 ∧
  -- 3. Number of ways to choose at least 2 white balls
  (Nat.choose white_balls 2 * Nat.choose black_balls 1 + Nat.choose white_balls 3) = 34 ∧
  -- 4. Probability of choosing 2 white and 1 black
  (↑(Nat.choose white_balls 2 * Nat.choose black_balls 1) / ↑(Nat.choose total_balls drawn_balls) : ℚ) = 30 / 84 ∧
  -- 5. Probability of choosing at least 2 white balls
  (↑(Nat.choose white_balls 2 * Nat.choose black_balls 1 + Nat.choose white_balls 3) / ↑(Nat.choose total_balls drawn_balls) : ℚ) = 34 / 84 :=
by sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l1129_112988


namespace NUMINAMATH_CALUDE_mirella_orange_books_l1129_112973

/-- The number of pages in each purple book -/
def purple_pages : ℕ := 230

/-- The number of pages in each orange book -/
def orange_pages : ℕ := 510

/-- The number of purple books Mirella read -/
def purple_books_read : ℕ := 5

/-- The difference between orange and purple pages Mirella read -/
def page_difference : ℕ := 890

/-- The number of orange books Mirella read -/
def orange_books_read : ℕ := 4

theorem mirella_orange_books :
  orange_books_read * orange_pages = 
  purple_books_read * purple_pages + page_difference := by
  sorry

end NUMINAMATH_CALUDE_mirella_orange_books_l1129_112973


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l1129_112943

/-- The amount of money Chris had before his birthday. -/
def money_before_birthday (grandmother_gift aunt_uncle_gift parents_gift total_now : ℕ) : ℕ :=
  total_now - (grandmother_gift + aunt_uncle_gift + parents_gift)

/-- Theorem stating that Chris had $239 before his birthday. -/
theorem chris_money_before_birthday :
  money_before_birthday 25 20 75 359 = 239 := by
  sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l1129_112943


namespace NUMINAMATH_CALUDE_cone_volume_l1129_112906

/-- Given a cone with base radius 1 and lateral surface area √5π, its volume is 2π/3 -/
theorem cone_volume (r h : ℝ) (lateral_area : ℝ) : 
  r = 1 → 
  lateral_area = Real.sqrt 5 * Real.pi → 
  2 * Real.pi * r * h = lateral_area → 
  (1/3) * Real.pi * r^2 * h = (2/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1129_112906


namespace NUMINAMATH_CALUDE_target_hit_probability_l1129_112913

theorem target_hit_probability (p_A p_B : ℝ) (h_A : p_A = 9/10) (h_B : p_B = 8/9) :
  1 - (1 - p_A) * (1 - p_B) = 89/90 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1129_112913


namespace NUMINAMATH_CALUDE_line_parallel_plane_transitivity_l1129_112981

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_plane_transitivity 
  (a b : Line) (α : Plane) :
  parallel a α → subset b α → parallel a α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_plane_transitivity_l1129_112981


namespace NUMINAMATH_CALUDE_inequality_proof_l1129_112945

theorem inequality_proof (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) (h₅ : a₅ > 1) : 
  16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) > (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1129_112945


namespace NUMINAMATH_CALUDE_value_of_w_l1129_112934

theorem value_of_w (j p t q s w : ℝ) 
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (w / 100) * p)
  (h4 : q = 1.15 * p)
  (h5 : q = 0.70 * j)
  (h6 : s = 1.40 * t)
  (h7 : s = 0.90 * q) :
  w = 6.25 := by
sorry

end NUMINAMATH_CALUDE_value_of_w_l1129_112934


namespace NUMINAMATH_CALUDE_d_range_l1129_112955

/-- Circle C with center (3,4) and radius 1 -/
def CircleC (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

/-- Point A -/
def A : ℝ × ℝ := (0, 1)

/-- Point B -/
def B : ℝ × ℝ := (0, -1)

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- The function d for a point P on the circle -/
def d (x y : ℝ) : ℝ := distanceSquared (x, y) A + distanceSquared (x, y) B

theorem d_range :
  ∀ x y : ℝ, CircleC x y → 34 ≤ d x y ∧ d x y ≤ 74 :=
sorry

end NUMINAMATH_CALUDE_d_range_l1129_112955


namespace NUMINAMATH_CALUDE_restaurant_bill_problem_l1129_112900

theorem restaurant_bill_problem (kate_bill : ℝ) (bob_discount : ℝ) (kate_discount : ℝ) (total_after_discount : ℝ) :
  kate_bill = 25 →
  bob_discount = 0.05 →
  kate_discount = 0.02 →
  total_after_discount = 53 →
  ∃ bob_bill : ℝ, 
    bob_bill * (1 - bob_discount) + kate_bill * (1 - kate_discount) = total_after_discount ∧
    bob_bill = 30 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_problem_l1129_112900


namespace NUMINAMATH_CALUDE_min_votes_class_president_l1129_112910

/-- Represents the minimum number of votes needed to win an election -/
def min_votes_to_win (total_votes : ℕ) (num_candidates : ℕ) : ℕ :=
  (total_votes / num_candidates) + 1

/-- Theorem: In an election with 4 candidates and 61 votes, the minimum number of votes to win is 16 -/
theorem min_votes_class_president : min_votes_to_win 61 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_votes_class_president_l1129_112910


namespace NUMINAMATH_CALUDE_unique_solution_exists_l1129_112974

theorem unique_solution_exists : ∃! (a b : ℝ), 2 * a + b = 7 ∧ a - b = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l1129_112974


namespace NUMINAMATH_CALUDE_cubic_inequality_l1129_112999

theorem cubic_inequality (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + a + b + c ∧
  ((a^3 + b^3 + c^3) / 3 = a * b * c + a + b + c ↔ 
   (b = a + 1 ∧ c = b + 1) ∨ (a = b + 1 ∧ c = a + 1) ∨ (a = c + 1 ∧ b = a + 1)) :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1129_112999


namespace NUMINAMATH_CALUDE_sqrt_nine_equals_three_l1129_112902

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_equals_three_l1129_112902


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_5_cubed_plus_10_to_4_l1129_112952

theorem greatest_prime_factor_of_5_cubed_plus_10_to_4 :
  ∃ p : ℕ, p.Prime ∧ p ∣ (5^3 + 10^4) ∧ ∀ q : ℕ, q.Prime → q ∣ (5^3 + 10^4) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_5_cubed_plus_10_to_4_l1129_112952


namespace NUMINAMATH_CALUDE_exponent_equality_l1129_112908

theorem exponent_equality (a : ℝ) (m n : ℕ) (h : 0 < a) :
  (a^12 = (a^3)^m) ∧ (a^12 = a^2 * a^n) → m = 4 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l1129_112908


namespace NUMINAMATH_CALUDE_triangle_strike_interval_l1129_112963

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem triangle_strike_interval (cymbal_interval triangle_interval coincidence_interval : ℕ) :
  cymbal_interval = 7 →
  is_factor triangle_interval coincidence_interval →
  is_factor cymbal_interval coincidence_interval →
  coincidence_interval = 14 →
  triangle_interval ≠ cymbal_interval →
  triangle_interval > 0 →
  triangle_interval = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_strike_interval_l1129_112963


namespace NUMINAMATH_CALUDE_equation_solution_l1129_112924

theorem equation_solution :
  ∃ y : ℝ, (7 * (4 * y + 3) - 3 = -3 * (2 - 9 * y)) ∧ (y = -24) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1129_112924


namespace NUMINAMATH_CALUDE_prob_no_rain_five_days_l1129_112977

/-- The probability of no rain for n consecutive days, given the probability of rain on each day is p -/
def prob_no_rain (n : ℕ) (p : ℚ) : ℚ := (1 - p) ^ n

theorem prob_no_rain_five_days :
  prob_no_rain 5 (1/2) = 1/32 := by sorry

end NUMINAMATH_CALUDE_prob_no_rain_five_days_l1129_112977


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_one_l1129_112922

theorem product_of_fractions_equals_one :
  (7 / 4 : ℚ) * (8 / 14 : ℚ) * (21 / 12 : ℚ) * (16 / 28 : ℚ) *
  (49 / 28 : ℚ) * (24 / 42 : ℚ) * (63 / 36 : ℚ) * (32 / 56 : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_one_l1129_112922


namespace NUMINAMATH_CALUDE_average_visitors_theorem_l1129_112939

/-- Calculates the average number of visitors per day in a 30-day month starting with a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalDays : ℕ := 30
  let sundays : ℕ := 4
  let otherDays : ℕ := totalDays - sundays
  let totalVisitors : ℕ := sundayVisitors * sundays + otherDayVisitors * otherDays
  (totalVisitors : ℚ) / totalDays

theorem average_visitors_theorem (sundayVisitors otherDayVisitors : ℕ) 
  (h1 : sundayVisitors = 510) (h2 : otherDayVisitors = 240) : 
  averageVisitorsPerDay sundayVisitors otherDayVisitors = 276 := by
  sorry

#eval averageVisitorsPerDay 510 240

end NUMINAMATH_CALUDE_average_visitors_theorem_l1129_112939


namespace NUMINAMATH_CALUDE_coffee_maker_capacity_l1129_112965

/-- A cylindrical coffee maker with capacity x cups contains 30 cups when 25% full. -/
theorem coffee_maker_capacity (x : ℝ) (h : 0.25 * x = 30) : x = 120 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_capacity_l1129_112965


namespace NUMINAMATH_CALUDE_paper_width_is_four_l1129_112976

/-- Given a rectangular paper surrounded by a wall photo, this theorem proves
    that the width of the paper is 4 inches under certain conditions. -/
theorem paper_width_is_four 
  (photo_width : ℝ) 
  (paper_length : ℝ) 
  (photo_area : ℝ) 
  (h1 : photo_width = 2)
  (h2 : paper_length = 8)
  (h3 : photo_area = 96)
  (h4 : photo_area = (paper_length + 2 * photo_width) * (paper_width + 2 * photo_width)) :
  paper_width = 4 :=
by
  sorry

#check paper_width_is_four

end NUMINAMATH_CALUDE_paper_width_is_four_l1129_112976


namespace NUMINAMATH_CALUDE_escalator_problem_solution_l1129_112944

/-- The time taken to descend an escalator under different conditions -/
def EscalatorProblem (s : ℝ) : Prop :=
  let t_standing := (3/2 : ℝ)  -- Time taken when standing on moving escalator
  let t_running_stationary := (1 : ℝ)  -- Time taken when running on stationary escalator
  let v_escalator := s / t_standing  -- Speed of escalator
  let v_running := s / t_running_stationary  -- Speed of running
  let v_combined := v_escalator + v_running  -- Combined speed
  let t_running_moving := s / v_combined  -- Time taken when running on moving escalator
  t_running_moving = (3/5 : ℝ)

/-- Theorem stating the solution to the escalator problem -/
theorem escalator_problem_solution :
  ∀ s : ℝ, s > 0 → EscalatorProblem s :=
by
  sorry

end NUMINAMATH_CALUDE_escalator_problem_solution_l1129_112944


namespace NUMINAMATH_CALUDE_partition_theorem_l1129_112968

theorem partition_theorem (a m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n < 1/m) :
  let x := a * (n - 1) / (m * n - 1)
  let first_partition := (m * x, a - m * x)
  let second_partition := (x, n * (a - m * x))
  first_partition.1 + first_partition.2 = a ∧
  second_partition.1 + second_partition.2 = a ∧
  first_partition.1 = m * second_partition.1 ∧
  second_partition.2 = n * first_partition.2 :=
by sorry

end NUMINAMATH_CALUDE_partition_theorem_l1129_112968


namespace NUMINAMATH_CALUDE_share_difference_l1129_112930

theorem share_difference (total amount_a amount_b amount_c : ℕ) : 
  total = 120 →
  amount_b = 20 →
  amount_a + amount_b + amount_c = total →
  amount_a = amount_c - 20 →
  amount_a - amount_b = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_share_difference_l1129_112930


namespace NUMINAMATH_CALUDE_john_pennies_l1129_112920

/-- Given that Kate has 223 pennies, John has more pennies than Kate,
    and the difference between their pennies is 165,
    prove that John has 388 pennies. -/
theorem john_pennies (kate_pennies : ℕ) (john_more : ℕ) (difference : ℕ)
    (h1 : kate_pennies = 223)
    (h2 : john_more > kate_pennies)
    (h3 : john_more - kate_pennies = difference)
    (h4 : difference = 165) :
    john_more = 388 := by
  sorry

end NUMINAMATH_CALUDE_john_pennies_l1129_112920


namespace NUMINAMATH_CALUDE_sum_digits_2_5_power_1997_l1129_112948

/-- The number of decimal digits in a positive integer -/
def num_digits (x : ℕ) : ℕ := sorry

/-- Theorem: The sum of the number of decimal digits in 2^1997 and 5^1997 is 1998 -/
theorem sum_digits_2_5_power_1997 : num_digits (2^1997) + num_digits (5^1997) = 1998 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_2_5_power_1997_l1129_112948


namespace NUMINAMATH_CALUDE_hawkeye_fewer_maine_coons_l1129_112949

/-- Proves that Hawkeye owns 1 fewer Maine Coon than Gordon --/
theorem hawkeye_fewer_maine_coons (jamie_persians jamie_maine_coons gordon_persians gordon_maine_coons hawkeye_maine_coons : ℕ) :
  jamie_persians = 4 →
  jamie_maine_coons = 2 →
  gordon_persians = jamie_persians / 2 →
  gordon_maine_coons = jamie_maine_coons + 1 →
  jamie_persians + jamie_maine_coons + gordon_persians + gordon_maine_coons + hawkeye_maine_coons = 13 →
  gordon_maine_coons - hawkeye_maine_coons = 1 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_fewer_maine_coons_l1129_112949


namespace NUMINAMATH_CALUDE_students_walking_home_l1129_112971

theorem students_walking_home (bus_fraction automobile_fraction bicycle_fraction : ℚ)
  (h1 : bus_fraction = 1/2)
  (h2 : automobile_fraction = 1/4)
  (h3 : bicycle_fraction = 1/10) :
  1 - (bus_fraction + automobile_fraction + bicycle_fraction) = 3/20 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l1129_112971


namespace NUMINAMATH_CALUDE_average_stickers_per_pack_l1129_112929

def sticker_counts : List ℕ := [5, 8, 0, 12, 15, 20, 22, 25, 30, 35]

def num_packs : ℕ := 10

theorem average_stickers_per_pack :
  (sticker_counts.sum : ℚ) / num_packs = 17.2 := by
  sorry

end NUMINAMATH_CALUDE_average_stickers_per_pack_l1129_112929


namespace NUMINAMATH_CALUDE_triangle_inequality_third_side_length_l1129_112962

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  (a < b + c ∧ b < c + a ∧ c < a + b) :=
sorry

theorem third_side_length (x : ℝ) : 
  x > 0 → 
  5 > 0 → 
  8 > 0 → 
  (5 + 8 > x ∧ 8 + x > 5 ∧ x + 5 > 8) → 
  (3 < x ∧ x < 13) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_third_side_length_l1129_112962


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1129_112947

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 35)
  (sum2 : b + c = 47)
  (sum3 : c + a = 58) : 
  a + b + c = 70 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1129_112947


namespace NUMINAMATH_CALUDE_harkamal_payment_l1129_112919

/-- The total amount paid by Harkamal to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1100 to the shopkeeper -/
theorem harkamal_payment :
  total_amount 8 70 9 60 = 1100 := by
  sorry

#eval total_amount 8 70 9 60

end NUMINAMATH_CALUDE_harkamal_payment_l1129_112919


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l1129_112912

theorem consecutive_integers_product (a b c d e : ℤ) : 
  (a + b + c + d + e) / 5 = 17 ∧ 
  d = 12 ∧ 
  e = 22 ∧ 
  (∃ n : ℤ, a = n ∧ b = n + 1 ∧ c = n + 2) →
  a * b * c = 4896 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l1129_112912


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1129_112972

-- Problem 1
theorem problem_1 : (π - 3.14)^0 + Real.sqrt 16 + |1 - Real.sqrt 2| = 4 + Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : ∃ (x y : ℝ), x - y = 2 ∧ 2*x + 3*y = 9 ∧ x = 3 ∧ y = 1 := by sorry

-- Problem 3
theorem problem_3 : ∃ (x y : ℝ), 5*(x-1) + 2*y = 4*(1-y) + 3 ∧ x/3 + y/2 = 1 ∧ x = 0 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1129_112972


namespace NUMINAMATH_CALUDE_sqrt_sum_value_l1129_112994

theorem sqrt_sum_value (a b : ℝ) (h : (Real.sqrt a + Real.sqrt b) * (Real.sqrt a + Real.sqrt b - 2) = 3) :
  Real.sqrt a + Real.sqrt b = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_value_l1129_112994


namespace NUMINAMATH_CALUDE_sara_golf_balls_l1129_112978

/-- The number of golf balls in a dozen -/
def dozen : ℕ := 12

/-- The total number of golf balls Sara has -/
def total_golf_balls : ℕ := 108

/-- The number of dozens of golf balls Sara has -/
def dozens_of_golf_balls : ℕ := total_golf_balls / dozen

theorem sara_golf_balls : dozens_of_golf_balls = 9 := by
  sorry

end NUMINAMATH_CALUDE_sara_golf_balls_l1129_112978


namespace NUMINAMATH_CALUDE_no_complete_set_in_matrix_l1129_112905

/-- Definition of the matrix A --/
def A (n : ℕ) (i j : ℕ) : ℕ :=
  if (i + j - 1) % n = 0 then n else (i + j - 1) % n

/-- Theorem statement --/
theorem no_complete_set_in_matrix (n : ℕ) (h_even : Even n) (h_pos : 0 < n) :
  ¬ ∃ σ : Fin n → Fin n, Function.Bijective σ ∧ (∀ i : Fin n, A n i.val (σ i).val = i.val + 1) :=
sorry

end NUMINAMATH_CALUDE_no_complete_set_in_matrix_l1129_112905


namespace NUMINAMATH_CALUDE_smallest_non_prime_digit_divisible_by_all_single_digit_primes_l1129_112940

def is_prime (n : ℕ) : Prop := sorry

def single_digit_primes : List ℕ := [2, 3, 5, 7]

def digits (n : ℕ) : List ℕ := sorry

theorem smallest_non_prime_digit_divisible_by_all_single_digit_primes :
  ∃ (N : ℕ),
    (∀ d ∈ digits N, ¬ is_prime d) ∧
    (∀ p ∈ single_digit_primes, N % p = 0) ∧
    (∀ m < N, ¬(∀ d ∈ digits m, ¬ is_prime d) ∨ ¬(∀ p ∈ single_digit_primes, m % p = 0)) ∧
    N = 840 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_prime_digit_divisible_by_all_single_digit_primes_l1129_112940


namespace NUMINAMATH_CALUDE_smallest_base_for_150_base_6_works_for_150_smallest_base_is_6_l1129_112959

theorem smallest_base_for_150 :
  ∀ b : ℕ, b > 0 → (b^2 ≤ 150 ∧ 150 < b^3) → b ≥ 6 :=
by sorry

theorem base_6_works_for_150 :
  6^2 ≤ 150 ∧ 150 < 6^3 :=
by sorry

theorem smallest_base_is_6 :
  ∃! b : ℕ, b > 0 ∧ b^2 ≤ 150 ∧ 150 < b^3 ∧ ∀ c : ℕ, (c > 0 ∧ c^2 ≤ 150 ∧ 150 < c^3) → c ≥ b :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_150_base_6_works_for_150_smallest_base_is_6_l1129_112959


namespace NUMINAMATH_CALUDE_magnitude_relationship_l1129_112993

theorem magnitude_relationship :
  let α : ℝ := Real.cos 4
  let b : ℝ := Real.cos (4 * π / 5)
  let c : ℝ := Real.sin (7 * π / 6)
  b < α ∧ α < c := by
  sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l1129_112993


namespace NUMINAMATH_CALUDE_smallest_m_no_real_solutions_l1129_112997

theorem smallest_m_no_real_solutions : ∃ (m : ℕ), 
  (m > 0) ∧ 
  (∀ (x : ℝ), m * x^2 - 3 * x + 1 ≠ 0) ∧
  (∀ (k : ℕ), k > 0 → k < m → ∃ (x : ℝ), k * x^2 - 3 * x + 1 = 0) ∧
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_no_real_solutions_l1129_112997


namespace NUMINAMATH_CALUDE_teachers_not_adjacent_l1129_112928

/-- The number of ways to arrange 2 teachers and 3 students in a row, 
    such that the teachers are not adjacent -/
def arrangement_count : ℕ := 72

/-- The number of teachers -/
def teacher_count : ℕ := 2

/-- The number of students -/
def student_count : ℕ := 3

theorem teachers_not_adjacent : 
  arrangement_count = 
    (Nat.factorial student_count) * (Nat.factorial (student_count + 1)) / 
    (Nat.factorial (student_count + 1 - teacher_count)) := by
  sorry

end NUMINAMATH_CALUDE_teachers_not_adjacent_l1129_112928


namespace NUMINAMATH_CALUDE_sin_alpha_equals_half_l1129_112996

theorem sin_alpha_equals_half (α : Real) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin α = Real.cos (2 * α)) : 
  Real.sin α = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_equals_half_l1129_112996


namespace NUMINAMATH_CALUDE_bakery_bread_rolls_l1129_112960

/-- Given a bakery with a total of 90 items, 19 croissants, and 22 bagels,
    prove that the number of bread rolls is 49. -/
theorem bakery_bread_rolls :
  let total_items : ℕ := 90
  let croissants : ℕ := 19
  let bagels : ℕ := 22
  let bread_rolls : ℕ := total_items - croissants - bagels
  bread_rolls = 49 := by
  sorry

end NUMINAMATH_CALUDE_bakery_bread_rolls_l1129_112960


namespace NUMINAMATH_CALUDE_caroline_citrus_drinks_l1129_112926

/-- The number of citrus drinks Caroline can make from a given number of oranges -/
def citrus_drinks (oranges : ℕ) : ℕ :=
  8 * oranges / 3

/-- Theorem stating that Caroline can make 56 citrus drinks from 21 oranges -/
theorem caroline_citrus_drinks :
  citrus_drinks 21 = 56 := by
  sorry

end NUMINAMATH_CALUDE_caroline_citrus_drinks_l1129_112926


namespace NUMINAMATH_CALUDE_tobys_friends_l1129_112985

theorem tobys_friends (total_friends : ℕ) (boy_friends : ℕ) (girl_friends : ℕ) : 
  (boy_friends : ℚ) / (total_friends : ℚ) = 55 / 100 →
  boy_friends = 33 →
  girl_friends = 27 :=
by sorry

end NUMINAMATH_CALUDE_tobys_friends_l1129_112985


namespace NUMINAMATH_CALUDE_min_value_z_l1129_112901

/-- The minimum value of z = x - y given the specified constraints -/
theorem min_value_z (x y : ℝ) (h1 : x + y - 2 ≥ 0) (h2 : x ≤ 4) (h3 : y ≤ 5) :
  ∀ (x' y' : ℝ), x' + y' - 2 ≥ 0 → x' ≤ 4 → y' ≤ 5 → x - y ≤ x' - y' :=
by sorry

end NUMINAMATH_CALUDE_min_value_z_l1129_112901


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1129_112941

-- Problem 1
theorem problem_1 : 23 + (-16) - (-7) = 14 := by sorry

-- Problem 2
theorem problem_2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by sorry

-- Problem 3
theorem problem_3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -10/3 := by sorry

-- Problem 4
theorem problem_4 : -(1^4) - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1129_112941


namespace NUMINAMATH_CALUDE_unique_starting_number_l1129_112992

def operation (n : ℕ) : ℕ :=
  if n % 3 = 0 then n / 3 else n + 1

def iterate_operation (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => iterate_operation (operation n) k

theorem unique_starting_number : 
  ∃! n : ℕ, iterate_operation n 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_starting_number_l1129_112992


namespace NUMINAMATH_CALUDE_lcm_gcd_difference_times_min_l1129_112916

theorem lcm_gcd_difference_times_min (a b : ℕ) (ha : a = 8) (hb : b = 12) :
  (Nat.lcm a b - Nat.gcd a b) * min a b = 160 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_difference_times_min_l1129_112916


namespace NUMINAMATH_CALUDE_students_favoring_both_issues_l1129_112979

/-- The number of students who voted in favor of both issues in a school referendum -/
theorem students_favoring_both_issues 
  (total_students : ℕ) 
  (favor_first : ℕ) 
  (favor_second : ℕ) 
  (against_both : ℕ) 
  (h1 : total_students = 215)
  (h2 : favor_first = 160)
  (h3 : favor_second = 132)
  (h4 : against_both = 40) : 
  favor_first + favor_second - (total_students - against_both) = 117 :=
by sorry

end NUMINAMATH_CALUDE_students_favoring_both_issues_l1129_112979


namespace NUMINAMATH_CALUDE_potato_distribution_l1129_112954

theorem potato_distribution (num_people : ℕ) (bag_weight : ℝ) (bag_cost : ℝ) (total_cost : ℝ) :
  num_people = 40 →
  bag_weight = 20 →
  bag_cost = 5 →
  total_cost = 15 →
  (total_cost / bag_cost * bag_weight) / num_people = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_potato_distribution_l1129_112954
