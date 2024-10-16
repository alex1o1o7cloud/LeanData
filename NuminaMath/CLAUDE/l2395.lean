import Mathlib

namespace NUMINAMATH_CALUDE_salary_calculation_l2395_239543

/-- Represents the number of turbans given as part of the salary -/
def turbans : ℕ := sorry

/-- The annual base salary in rupees -/
def base_salary : ℕ := 90

/-- The price of each turban in rupees -/
def turban_price : ℕ := 70

/-- The number of months the servant worked -/
def months_worked : ℕ := 9

/-- The amount in rupees the servant received when leaving -/
def amount_received : ℕ := 50

/-- The total annual salary in rupees -/
def total_annual_salary : ℕ := base_salary + turbans * turban_price

/-- The fraction of the year the servant worked -/
def fraction_worked : ℚ := 3 / 4

theorem salary_calculation :
  (fraction_worked * total_annual_salary : ℚ) = (amount_received + turban_price : ℕ) → turbans = 1 :=
by sorry

end NUMINAMATH_CALUDE_salary_calculation_l2395_239543


namespace NUMINAMATH_CALUDE_problem_statement_l2395_239509

theorem problem_statement (m n : ℝ) (h : 3 * m - n = 1) :
  9 * m^2 - n^2 - 2 * n = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2395_239509


namespace NUMINAMATH_CALUDE_not_p_and_not_q_l2395_239508

-- Define proposition p
def p : Prop := ∃ t : ℝ, t > 0 ∧ t^2 - 2*t + 2 = 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp x - x - 1 ≥ -1

-- Theorem statement
theorem not_p_and_not_q : (¬p) ∧ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_l2395_239508


namespace NUMINAMATH_CALUDE_january_31_is_wednesday_l2395_239578

/-- Days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to advance a day by n days -/
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

/-- Theorem: If January 1 is a Monday, then January 31 is a Wednesday -/
theorem january_31_is_wednesday : 
  advanceDay DayOfWeek.Monday 30 = DayOfWeek.Wednesday := by
  sorry

end NUMINAMATH_CALUDE_january_31_is_wednesday_l2395_239578


namespace NUMINAMATH_CALUDE_power_of_power_l2395_239511

theorem power_of_power (x : ℝ) : (x^5)^2 = x^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2395_239511


namespace NUMINAMATH_CALUDE_book_env_intersection_l2395_239527

/-- The number of people participating in both Book Club and Environmental Theme Painting --/
def intersection_book_env (total participants : ℕ) 
  (book_club fun_sports env_painting : ℕ) 
  (book_fun fun_env : ℕ) : ℕ :=
  book_club + fun_sports + env_painting - total - book_fun - fun_env

/-- Theorem stating the number of people participating in both Book Club and Environmental Theme Painting --/
theorem book_env_intersection : 
  ∀ (total participants : ℕ) 
    (book_club fun_sports env_painting : ℕ) 
    (book_fun fun_env : ℕ),
  total = 120 →
  book_club = 80 →
  fun_sports = 50 →
  env_painting = 40 →
  book_fun = 20 →
  fun_env = 10 →
  intersection_book_env total participants book_club fun_sports env_painting book_fun fun_env = 20 := by
  sorry

#check book_env_intersection

end NUMINAMATH_CALUDE_book_env_intersection_l2395_239527


namespace NUMINAMATH_CALUDE_right_triangle_area_l2395_239597

/-- The area of a right-angled triangle with sides 30 cm and 40 cm adjacent to the right angle is 600 cm². -/
theorem right_triangle_area (a b : ℝ) (h1 : a = 30) (h2 : b = 40) : 
  (1/2) * a * b = 600 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2395_239597


namespace NUMINAMATH_CALUDE_yellow_flowers_count_l2395_239526

/-- Represents the number of flowers of each color in the garden -/
structure FlowerCounts where
  total : Nat
  green : Nat
  red : Nat
  blue : Nat
  yellow : Nat

/-- The conditions of the flower garden problem -/
def gardenConditions : FlowerCounts → Prop := fun c =>
  c.total = 96 ∧
  c.green = 9 ∧
  c.red = 3 * c.green ∧
  c.blue = c.total / 2 ∧
  c.yellow = c.total - c.green - c.red - c.blue

/-- Theorem stating that under the given conditions, there are 12 yellow flowers -/
theorem yellow_flowers_count (c : FlowerCounts) : 
  gardenConditions c → c.yellow = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_flowers_count_l2395_239526


namespace NUMINAMATH_CALUDE_next_occurrence_sqrt_l2395_239589

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 60 * 60

/-- The time difference in seconds between two consecutive occurrences -/
def time_difference : ℕ := seconds_per_day + seconds_per_hour

theorem next_occurrence_sqrt (S : ℕ) (h : S = time_difference) : 
  Real.sqrt (S : ℝ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_next_occurrence_sqrt_l2395_239589


namespace NUMINAMATH_CALUDE_fruit_cost_theorem_l2395_239528

/-- Calculates the cost of remaining fruit after discount and loss -/
def remaining_fruit_cost (pear_price apple_price pineapple_price plum_price : ℚ)
                         (pear_qty apple_qty pineapple_qty plum_qty : ℕ)
                         (apple_discount : ℚ) (fruit_loss_ratio : ℚ) : ℚ :=
  let total_cost := pear_price * pear_qty + 
                    apple_price * apple_qty * (1 - apple_discount) + 
                    pineapple_price * pineapple_qty + 
                    plum_price * plum_qty
  total_cost * (1 - fruit_loss_ratio)

/-- The theorem to be proved -/
theorem fruit_cost_theorem : 
  remaining_fruit_cost 1.5 0.75 2 0.5 6 4 2 1 0.25 0.5 = 7.88 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_theorem_l2395_239528


namespace NUMINAMATH_CALUDE_tetrahedron_edges_form_two_triangles_l2395_239515

-- Define a tetrahedron as a structure with 6 edges
structure Tetrahedron where
  edges : Fin 6 → ℝ
  edge_positive : ∀ i, edges i > 0

-- Define a predicate for valid triangles
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b + c ∧ b < a + c ∧ c < a + b

-- Theorem statement
theorem tetrahedron_edges_form_two_triangles (t : Tetrahedron) :
  ∃ (i₁ i₂ i₃ i₄ i₅ i₆ : Fin 6),
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₁ ≠ i₅ ∧ i₁ ≠ i₆ ∧
    i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₂ ≠ i₅ ∧ i₂ ≠ i₆ ∧
    i₃ ≠ i₄ ∧ i₃ ≠ i₅ ∧ i₃ ≠ i₆ ∧
    i₄ ≠ i₅ ∧ i₄ ≠ i₆ ∧
    i₅ ≠ i₆ ∧
    is_valid_triangle (t.edges i₁) (t.edges i₂) (t.edges i₃) ∧
    is_valid_triangle (t.edges i₄) (t.edges i₅) (t.edges i₆) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_edges_form_two_triangles_l2395_239515


namespace NUMINAMATH_CALUDE_initial_acorns_l2395_239536

def acorns_given_away : ℕ := 7
def acorns_left : ℕ := 9

theorem initial_acorns : 
  acorns_given_away + acorns_left = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_acorns_l2395_239536


namespace NUMINAMATH_CALUDE_det_dilation_matrix_l2395_239568

/-- A 3x3 matrix representing a dilation with scale factor 5 -/
def E : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => 5)

/-- Theorem stating that the determinant of E is 125 -/
theorem det_dilation_matrix : Matrix.det E = 125 := by
  sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_l2395_239568


namespace NUMINAMATH_CALUDE_female_democrats_count_l2395_239523

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 750 →
  female + male = total →
  female / 2 + male / 4 = total / 3 →
  female / 2 = 125 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l2395_239523


namespace NUMINAMATH_CALUDE_distributions_five_balls_four_boxes_l2395_239538

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def totalDistributions (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes,
    with the condition that one specific box must contain at least one ball -/
def distributionsWithConstraint (n k : ℕ) : ℕ :=
  totalDistributions n k - totalDistributions n (k - 1)

theorem distributions_five_balls_four_boxes :
  distributionsWithConstraint 5 4 = 781 := by
  sorry

end NUMINAMATH_CALUDE_distributions_five_balls_four_boxes_l2395_239538


namespace NUMINAMATH_CALUDE_problems_per_page_l2395_239588

theorem problems_per_page 
  (math_pages : ℕ) 
  (reading_pages : ℕ) 
  (total_problems : ℕ) 
  (h1 : math_pages = 6) 
  (h2 : reading_pages = 4) 
  (h3 : total_problems = 30) : 
  total_problems / (math_pages + reading_pages) = 3 := by
sorry

end NUMINAMATH_CALUDE_problems_per_page_l2395_239588


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2395_239592

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

/-- Theorem: If a_3 + a_4 + a_5 + a_6 + a_7 = 20 in an arithmetic sequence, then S_9 = 36 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h : seq.a 3 + seq.a 4 + seq.a 5 + seq.a 6 + seq.a 7 = 20) :
  seq.S 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2395_239592


namespace NUMINAMATH_CALUDE_abc_sum_product_range_l2395_239514

theorem abc_sum_product_range (a b c : ℝ) (h : a + b + c = 3) :
  ∃ S : Set ℝ, S = Set.Iic 3 ∧ ∀ x : ℝ, x ∈ S ↔ ∃ a' b' c' : ℝ, a' + b' + c' = 3 ∧ a' * b' + a' * c' + b' * c' = x :=
sorry

end NUMINAMATH_CALUDE_abc_sum_product_range_l2395_239514


namespace NUMINAMATH_CALUDE_expression_equals_two_l2395_239535

theorem expression_equals_two :
  (-1)^2023 - 2 * Real.sin (π / 3) + |(-Real.sqrt 3)| + (1/3)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l2395_239535


namespace NUMINAMATH_CALUDE_cot_150_degrees_l2395_239582

theorem cot_150_degrees : Real.cos (150 * π / 180) / Real.sin (150 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_150_degrees_l2395_239582


namespace NUMINAMATH_CALUDE_money_problem_l2395_239524

theorem money_problem (c d : ℝ) (h1 : 7 * c + d > 84) (h2 : 5 * c - d = 35) :
  c > 9.92 ∧ d > 14.58 := by
sorry

end NUMINAMATH_CALUDE_money_problem_l2395_239524


namespace NUMINAMATH_CALUDE_prob_neither_chooses_D_l2395_239541

/-- Represents the four projects --/
inductive Project : Type
  | A
  | B
  | C
  | D

/-- Represents the outcome of both students' choices --/
structure Outcome :=
  (fanfan : Project)
  (lelle : Project)

/-- The set of all possible outcomes --/
def all_outcomes : Finset Outcome :=
  sorry

/-- The set of outcomes where neither student chooses project D --/
def outcomes_without_D : Finset Outcome :=
  sorry

/-- The probability of an event is the number of favorable outcomes
    divided by the total number of outcomes --/
def probability (event : Finset Outcome) : Rat :=
  (event.card : Rat) / (all_outcomes.card : Rat)

/-- The main theorem: probability of neither student choosing D is 1/2 --/
theorem prob_neither_chooses_D :
  probability outcomes_without_D = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_prob_neither_chooses_D_l2395_239541


namespace NUMINAMATH_CALUDE_sandys_number_l2395_239584

theorem sandys_number : ∃! x : ℝ, (3 * x + 20)^2 = 2500 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sandys_number_l2395_239584


namespace NUMINAMATH_CALUDE_clubsuit_symmetry_forms_intersecting_lines_l2395_239506

-- Define the operation ♣
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Theorem statement
theorem clubsuit_symmetry_forms_intersecting_lines :
  ∃ (l₁ l₂ : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), clubsuit x y = clubsuit y x ↔ (x, y) ∈ l₁ ∪ l₂) ∧
    (l₁ ≠ l₂) ∧
    (∃ (p : ℝ × ℝ), p ∈ l₁ ∧ p ∈ l₂) :=
sorry


end NUMINAMATH_CALUDE_clubsuit_symmetry_forms_intersecting_lines_l2395_239506


namespace NUMINAMATH_CALUDE_exists_non_one_same_first_digit_l2395_239558

/-- Given a natural number n, returns the first digit of n -/
def firstDigit (n : ℕ) : ℕ := sorry

/-- Returns true if all numbers in the list start with the same digit -/
def sameFirstDigit (numbers : List ℕ) : Bool := sorry

theorem exists_non_one_same_first_digit :
  ∃ x : ℕ, x > 0 ∧ 
  let powers := List.range 2015 |>.map (λ i => x^(i+1))
  sameFirstDigit powers ∧ 
  firstDigit x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_non_one_same_first_digit_l2395_239558


namespace NUMINAMATH_CALUDE_harmonic_progression_solutions_l2395_239574

def is_harmonic_progression (a b c : ℕ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (1 / a + 1 / c = 2 / b)

def valid_harmonic_progression (a b c : ℕ) : Prop :=
  is_harmonic_progression a b c ∧ a < b ∧ b < c ∧ a = 20 ∧ c % b = 0

theorem harmonic_progression_solutions :
  {(b, c) : ℕ × ℕ | valid_harmonic_progression 20 b c} =
    {(30, 60), (35, 140), (36, 180), (38, 380), (39, 780)} := by
  sorry

end NUMINAMATH_CALUDE_harmonic_progression_solutions_l2395_239574


namespace NUMINAMATH_CALUDE_bike_ride_time_l2395_239534

/-- Given a constant speed where 2 miles are covered in 8 minutes, 
    prove that the time required to cover 5 miles is 20 minutes. -/
theorem bike_ride_time (speed : ℝ) (distance_to_julia : ℝ) (time_to_julia : ℝ) 
  (distance_to_bernard : ℝ) : 
  distance_to_julia = 2 →
  time_to_julia = 8 →
  distance_to_bernard = 5 →
  speed = distance_to_julia / time_to_julia →
  distance_to_bernard / speed = 20 := by
  sorry

#check bike_ride_time

end NUMINAMATH_CALUDE_bike_ride_time_l2395_239534


namespace NUMINAMATH_CALUDE_cell_phone_providers_l2395_239573

theorem cell_phone_providers (n : ℕ) (k : ℕ) : n = 25 ∧ k = 4 → (n - 0) * (n - 1) * (n - 2) * (n - 3) = 303600 := by
  sorry

end NUMINAMATH_CALUDE_cell_phone_providers_l2395_239573


namespace NUMINAMATH_CALUDE_shooter_probability_l2395_239513

theorem shooter_probability (p_a p_b : ℝ) 
  (h_p_a : p_a = 3/4)
  (h_p_b : p_b = 2/3)
  (h_p_a_range : 0 ≤ p_a ∧ p_a ≤ 1)
  (h_p_b_range : 0 ≤ p_b ∧ p_b ≤ 1) :
  p_a * (1 - p_b) * (1 - p_b) + (1 - p_a) * p_b * (1 - p_b) + (1 - p_a) * (1 - p_b) * p_b = 7/36 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l2395_239513


namespace NUMINAMATH_CALUDE_three_red_and_at_least_one_white_mutually_exclusive_and_complementary_l2395_239566

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing 3 balls -/
def Outcome := (Color × Color × Color)

/-- The sample space of all possible outcomes when drawing 3 balls from a bag with 5 red and 5 white balls -/
def SampleSpace : Set Outcome := sorry

/-- Event: Draw three red balls -/
def ThreeRedBalls (outcome : Outcome) : Prop := 
  outcome = (Color.Red, Color.Red, Color.Red)

/-- Event: Draw three balls with at least one white ball -/
def AtLeastOneWhiteBall (outcome : Outcome) : Prop := 
  outcome.1 = Color.White ∨ outcome.2.1 = Color.White ∨ outcome.2.2 = Color.White

theorem three_red_and_at_least_one_white_mutually_exclusive_and_complementary :
  (∀ outcome ∈ SampleSpace, ¬(ThreeRedBalls outcome ∧ AtLeastOneWhiteBall outcome)) ∧ 
  (∀ outcome ∈ SampleSpace, ThreeRedBalls outcome ∨ AtLeastOneWhiteBall outcome) := by
  sorry

end NUMINAMATH_CALUDE_three_red_and_at_least_one_white_mutually_exclusive_and_complementary_l2395_239566


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2395_239594

theorem inequality_solution_set (x : ℝ) : 
  (x + 1) / (x - 2) + (x + 3) / (2 * x + 1) ≤ 2 ↔ x ∈ Set.Icc (-1/2) 1 ∪ Set.Icc 2 9 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2395_239594


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l2395_239549

/-- Given an angle θ in the second quadrant satisfying the equation
    cos(θ/2) - sin(θ/2) = √(1 - sin(θ)), prove that θ/2 is in the third quadrant. -/
theorem angle_in_third_quadrant (θ : Real) 
  (h1 : π < θ ∧ θ < 3*π/2) -- θ is in the second quadrant
  (h2 : Real.cos (θ/2) - Real.sin (θ/2) = Real.sqrt (1 - Real.sin θ)) :
  π < θ/2 ∧ θ/2 < 3*π/2 := by
  sorry


end NUMINAMATH_CALUDE_angle_in_third_quadrant_l2395_239549


namespace NUMINAMATH_CALUDE_ratio_equality_l2395_239557

theorem ratio_equality : 
  ∀ (a b c d x : ℚ), 
    a = 3 / 5 → 
    b = 6 / 7 → 
    c = 2 / 3 → 
    d = 7 / 15 → 
    (a / b = d / c) → 
    x = d := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2395_239557


namespace NUMINAMATH_CALUDE_solution_difference_l2395_239505

theorem solution_difference (r s : ℝ) : 
  r ≠ s →
  (r - 5) * (r + 5) = 25 * r - 125 →
  (s - 5) * (s + 5) = 25 * s - 125 →
  r > s →
  r - s = 15 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l2395_239505


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2395_239567

/-- The quadratic inequality function -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 1) * x + 2

/-- The solution set when m = 0 -/
def solution_set_m_zero : Set ℝ := {x | -2 < x ∧ x < 1}

/-- The range of m for which the solution set is ℝ -/
def m_range : Set ℝ := {m | 1 ≤ m ∧ m < 9}

theorem quadratic_inequality_theorem :
  (∀ x, x ∈ solution_set_m_zero ↔ f 0 x > 0) ∧
  (∀ m, (∀ x, f m x > 0) ↔ m ∈ m_range) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2395_239567


namespace NUMINAMATH_CALUDE_sodium_chloride_percentage_l2395_239583

theorem sodium_chloride_percentage
  (tank_capacity : ℝ)
  (fill_ratio : ℝ)
  (evaporation_rate : ℝ)
  (time : ℝ)
  (final_water_concentration : ℝ)
  (h1 : tank_capacity = 24)
  (h2 : fill_ratio = 1/4)
  (h3 : evaporation_rate = 0.4)
  (h4 : time = 6)
  (h5 : final_water_concentration = 1/2) :
  let initial_volume := tank_capacity * fill_ratio
  let evaporated_water := evaporation_rate * time
  let final_volume := initial_volume - evaporated_water
  let initial_sodium_chloride_percentage := 
    100 * (initial_volume - (final_volume * final_water_concentration)) / initial_volume
  initial_sodium_chloride_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_sodium_chloride_percentage_l2395_239583


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_squares_product_l2395_239504

theorem quadratic_equation_sum_squares_product (k : ℚ) : 
  (∃ a b : ℚ, 3 * a^2 + 7 * a + k = 0 ∧ 3 * b^2 + 7 * b + k = 0 ∧ a ≠ b) →
  (∀ a b : ℚ, 3 * a^2 + 7 * a + k = 0 → 3 * b^2 + 7 * b + k = 0 → a^2 + b^2 = 3 * a * b) ↔
  k = 49 / 15 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_squares_product_l2395_239504


namespace NUMINAMATH_CALUDE_percentage_difference_l2395_239553

theorem percentage_difference (x y : ℝ) (h : y = 1.25 * x) : 
  x = 0.8 * y := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2395_239553


namespace NUMINAMATH_CALUDE_earth_central_angle_special_case_l2395_239522

/-- Represents a point on Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- The Earth, assumed to be a perfect sphere -/
structure Earth where
  center : Point
  radius : Real

/-- Calculates the central angle between two points on Earth -/
def centralAngle (earth : Earth) (p1 p2 : EarthPoint) : Real :=
  sorry

theorem earth_central_angle_special_case (earth : Earth) :
  let a : EarthPoint := { latitude := 0, longitude := 100 }
  let b : EarthPoint := { latitude := 30, longitude := -90 }
  centralAngle earth a b = 180 := by sorry

end NUMINAMATH_CALUDE_earth_central_angle_special_case_l2395_239522


namespace NUMINAMATH_CALUDE_square_of_1023_l2395_239598

theorem square_of_1023 : (1023 : ℕ)^2 = 1046529 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1023_l2395_239598


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l2395_239537

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 15) :
  a^4 + b^4 + c^4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l2395_239537


namespace NUMINAMATH_CALUDE_base_prime_441_l2395_239500

/-- Base prime representation of a natural number --/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- 441 in base prime representation --/
theorem base_prime_441 : base_prime_repr 441 = [0, 2, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_441_l2395_239500


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2395_239559

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 6*a*x + 9

-- Define the theorem
theorem quadratic_function_properties (a : ℝ) :
  -- Part (1)
  (f a 2 = 7 → 
    a = 1/2 ∧ 
    ∃ (x y : ℝ), x = 3/2 ∧ y = 27/4 ∧ ∀ (t : ℝ), f a t ≥ f a x) ∧
  -- Part (2)
  (f a 2 = 7 → 
    ∀ (x : ℝ), -1 ≤ x ∧ x < 3 → 27/4 ≤ f a x ∧ f a x ≤ 13) ∧
  -- Part (3)
  (∀ (x : ℝ), x ≥ 3 → ∀ (y : ℝ), y > x → f a y > f a x) →
  (∀ (x₁ x₂ : ℝ), 3*a - 2 ≤ x₁ ∧ x₁ ≤ 5 ∧ 3*a - 2 ≤ x₂ ∧ x₂ ≤ 5 → 
    f a x₁ - f a x₂ ≤ 9*a^2 + 20) →
  1/6 ≤ a ∧ a ≤ 1 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l2395_239559


namespace NUMINAMATH_CALUDE_consecutive_composites_exist_l2395_239575

-- Define a function to check if a number is composite
def isComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

-- Define a function to check if a sequence of numbers is all composite
def allComposite (start : ℕ) (length : ℕ) : Prop :=
  ∀ i, i ∈ Finset.range length → isComposite (start + i)

-- Theorem statement
theorem consecutive_composites_exist :
  (∃ start : ℕ, start ≤ 500 - 9 + 1 ∧ allComposite start 9) ∧
  (∃ start : ℕ, start ≤ 500 - 11 + 1 ∧ allComposite start 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_composites_exist_l2395_239575


namespace NUMINAMATH_CALUDE_christine_savings_510_l2395_239576

/-- Calculates Christine's savings based on her commission rates, sales, and allocation percentages. -/
def christine_savings (
  electronics_rate : ℚ)
  (clothing_rate : ℚ)
  (furniture_rate : ℚ)
  (electronics_sales : ℚ)
  (clothing_sales : ℚ)
  (furniture_sales : ℚ)
  (personal_needs_rate : ℚ)
  (investment_rate : ℚ) : ℚ :=
  let total_commission := 
    electronics_rate * electronics_sales + 
    clothing_rate * clothing_sales + 
    furniture_rate * furniture_sales
  let personal_needs := personal_needs_rate * total_commission
  let investments := investment_rate * total_commission
  total_commission - personal_needs - investments

/-- Theorem stating that Christine's savings for the month equal $510. -/
theorem christine_savings_510 : 
  christine_savings (15/100) (10/100) (20/100) 12000 8000 4000 (55/100) (30/100) = 510 := by
  sorry

end NUMINAMATH_CALUDE_christine_savings_510_l2395_239576


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2395_239593

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity 
  (h : Hyperbola a b) 
  (F A B P O : Point) -- F is the right focus, O is the origin
  (right_branch : Point → Prop) -- Predicate for points on the right branch
  (on_hyperbola : Point → Prop) -- Predicate for points on the hyperbola
  (line_through : Point → Point → Point → Prop) -- Predicate for collinear points
  (symmetric_to : Point → Point → Point → Prop) -- Predicate for point symmetry
  (perpendicular : Point → Point → Point → Point → Prop) -- Predicate for perpendicular lines
  (h_right_focus : F.x > 0 ∧ F.y = 0)
  (h_AB_on_C : on_hyperbola A ∧ on_hyperbola B)
  (h_AB_right : right_branch A ∧ right_branch B)
  (h_line_FAB : line_through F A B)
  (h_A_symmetric : symmetric_to A O P)
  (h_PF_perp_AB : perpendicular P F A B)
  (h_BF_3AF : distance B F = 3 * distance A F)
  : eccentricity h = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2395_239593


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2395_239595

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : geometric_sequence a) :
  a₃ * a₅ * a₇ * a₉ * a₁₁ = 243 → a₁₀^2 / a₁₃ = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2395_239595


namespace NUMINAMATH_CALUDE_remainder_problem_l2395_239571

theorem remainder_problem (d r : ℤ) : 
  d > 1 → 
  1122 % d = r → 
  1540 % d = r → 
  2455 % d = r → 
  d - r = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2395_239571


namespace NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l2395_239501

theorem unique_solution_3x_4y_5z :
  ∀ (x y z : ℕ), 3^x + 4^y = 5^z → (x = 2 ∧ y = 2 ∧ z = 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l2395_239501


namespace NUMINAMATH_CALUDE_rectangular_box_existence_l2395_239532

theorem rectangular_box_existence : ∃ (a b c : ℕ), 
  a * b * c ≥ 1995 ∧ 2 * (a * b + b * c + a * c) = 958 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_existence_l2395_239532


namespace NUMINAMATH_CALUDE_platform_length_calculation_l2395_239544

/-- Calculates the length of a platform given train parameters --/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmph = 80 →
  crossing_time = 22 →
  ∃ (platform_length : ℝ), abs (platform_length - 288.84) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_platform_length_calculation_l2395_239544


namespace NUMINAMATH_CALUDE_chess_club_games_l2395_239560

/-- 
Given a chess club with the following properties:
- There are 15 participants in total
- 5 of the participants are instructors
- Each member plays one game against each instructor

The total number of games played is 70.
-/
theorem chess_club_games (total_participants : ℕ) (instructors : ℕ) :
  total_participants = 15 →
  instructors = 5 →
  instructors * (total_participants - 1) = 70 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_games_l2395_239560


namespace NUMINAMATH_CALUDE_smallest_angle_tangent_equality_l2395_239599

theorem smallest_angle_tangent_equality (x : ℝ) : 
  (x > 0) → 
  (x * (180 / π) = 5.625) → 
  (Real.tan (6 * x) = (Real.cos (2 * x) - Real.sin (2 * x)) / (Real.cos (2 * x) + Real.sin (2 * x))) → 
  ∀ y : ℝ, (y > 0) → 
    (Real.tan (6 * y) = (Real.cos (2 * y) - Real.sin (2 * y)) / (Real.cos (2 * y) + Real.sin (2 * y))) → 
    (y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_tangent_equality_l2395_239599


namespace NUMINAMATH_CALUDE_solving_linear_equations_count_l2395_239547

/-- Given a total number of math homework problems, calculate the number of
    solving linear equations problems, knowing that 40% are Algebra problems
    and half of the Algebra problems are solving linear equations. -/
def solvingLinearEquationsProblems (total : ℕ) : ℕ :=
  (total * 40 / 100) / 2

/-- Proof that for 140 total math homework problems, the number of
    solving linear equations problems is 28. -/
theorem solving_linear_equations_count :
  solvingLinearEquationsProblems 140 = 28 := by
  sorry

#eval solvingLinearEquationsProblems 140

end NUMINAMATH_CALUDE_solving_linear_equations_count_l2395_239547


namespace NUMINAMATH_CALUDE_oviparous_produces_significant_differences_l2395_239539

-- Define the modes of reproduction
inductive ReproductionMode
  | Vegetative
  | Oviparous
  | Fission
  | Budding

-- Define the reproduction categories
inductive ReproductionCategory
  | Sexual
  | Asexual

-- Define a function to categorize reproduction modes
def categorizeReproduction : ReproductionMode → ReproductionCategory
  | ReproductionMode.Vegetative => ReproductionCategory.Asexual
  | ReproductionMode.Oviparous => ReproductionCategory.Sexual
  | ReproductionMode.Fission => ReproductionCategory.Asexual
  | ReproductionMode.Budding => ReproductionCategory.Asexual

-- Define a property for producing offspring with significant differences
def produceSignificantDifferences (mode : ReproductionMode) : Prop :=
  categorizeReproduction mode = ReproductionCategory.Sexual

theorem oviparous_produces_significant_differences :
  ∀ (mode : ReproductionMode),
    produceSignificantDifferences mode ↔ mode = ReproductionMode.Oviparous :=
by sorry

end NUMINAMATH_CALUDE_oviparous_produces_significant_differences_l2395_239539


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_inequality_l2395_239545

theorem least_integer_absolute_value_inequality :
  ∃ (x : ℤ), (3 * |x| + 4 < 19) ∧ (∀ (y : ℤ), y < x → 3 * |y| + 4 ≥ 19) :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_inequality_l2395_239545


namespace NUMINAMATH_CALUDE_evaluate_expression_l2395_239525

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2395_239525


namespace NUMINAMATH_CALUDE_parabola_and_line_intersection_l2395_239519

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the distance between focus and directrix
def focus_directrix_distance (p : ℝ) : ℝ := 2*p

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through two points
def line_equation (A B : Point) (x y : ℝ) : Prop :=
  (y - A.y) * (B.x - A.x) = (x - A.x) * (B.y - A.y)

-- Define the midpoint of a line segment
def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- Theorem statement
theorem parabola_and_line_intersection 
  (p : ℝ) (A B : Point) (h1 : p > 0) (h2 : focus_directrix_distance p = 4) 
  (h3 : parabola p A.x A.y) (h4 : parabola p B.x B.y)
  (h5 : is_midpoint (Point.mk 1 (-1)) A B) :
  (∀ x y, parabola p x y ↔ y^2 = 8*x) ∧
  (∀ x y, line_equation A B x y ↔ 4*x + y - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_intersection_l2395_239519


namespace NUMINAMATH_CALUDE_oven_temperature_increase_l2395_239581

/-- Given an oven with a current temperature and a required temperature,
    calculate the temperature increase needed. -/
def temperature_increase_needed (current_temp required_temp : ℕ) : ℕ :=
  required_temp - current_temp

/-- Theorem stating that for an oven at 150 degrees that needs to reach 546 degrees,
    the temperature increase needed is 396 degrees. -/
theorem oven_temperature_increase :
  temperature_increase_needed 150 546 = 396 := by
  sorry

end NUMINAMATH_CALUDE_oven_temperature_increase_l2395_239581


namespace NUMINAMATH_CALUDE_product_abc_equals_195_l2395_239555

/-- Given the conditions on products of variables a, b, c, d, e, f,
    prove that a * b * c equals 195. -/
theorem product_abc_equals_195
  (h1 : b * c * d = 65)
  (h2 : c * d * e = 1000)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 3/4)
  : a * b * c = 195 := by
  sorry

end NUMINAMATH_CALUDE_product_abc_equals_195_l2395_239555


namespace NUMINAMATH_CALUDE_gingers_children_l2395_239565

/-- The number of cakes Ginger bakes for each child per year -/
def cakes_per_child : ℕ := 4

/-- The number of cakes Ginger bakes for her husband per year -/
def cakes_for_husband : ℕ := 6

/-- The number of cakes Ginger bakes for her parents per year -/
def cakes_for_parents : ℕ := 2

/-- The total number of cakes Ginger bakes in 10 years -/
def total_cakes : ℕ := 160

/-- The number of years over which the total cakes are counted -/
def years : ℕ := 10

/-- Ginger's number of children -/
def num_children : ℕ := 2

theorem gingers_children :
  num_children * cakes_per_child * years + cakes_for_husband * years + cakes_for_parents * years = total_cakes :=
by sorry

end NUMINAMATH_CALUDE_gingers_children_l2395_239565


namespace NUMINAMATH_CALUDE_min_moves_to_align_cups_l2395_239554

/-- Represents the state of cups on a table -/
structure CupState where
  totalCups : Nat
  upsideCups : Nat
  downsideCups : Nat

/-- Represents a move that flips exactly 3 cups -/
def flipThreeCups (state : CupState) : CupState :=
  { totalCups := state.totalCups,
    upsideCups := state.upsideCups + 3 - 2 * min 3 state.upsideCups,
    downsideCups := state.downsideCups + 3 - 2 * min 3 state.downsideCups }

/-- Predicate to check if all cups are facing the same direction -/
def allSameDirection (state : CupState) : Prop :=
  state.upsideCups = 0 ∨ state.upsideCups = state.totalCups

/-- The main theorem stating the minimum number of moves required -/
theorem min_moves_to_align_cups : 
  ∃ (n : Nat), 
    (∀ (state : CupState), 
      state.totalCups = 10 → 
      state.upsideCups = 5 → 
      state.downsideCups = 5 → 
      ∃ (moves : List (CupState → CupState)), 
        moves.length ≤ n ∧ 
        allSameDirection (moves.foldl (fun s m => m s) state) ∧
        ∀ m, m ∈ moves → m = flipThreeCups) ∧
    (∀ (k : Nat), 
      k < n → 
      ∃ (state : CupState), 
        state.totalCups = 10 ∧ 
        state.upsideCups = 5 ∧ 
        state.downsideCups = 5 ∧ 
        ∀ (moves : List (CupState → CupState)), 
          moves.length ≤ k → 
          (∀ m, m ∈ moves → m = flipThreeCups) → 
          ¬allSameDirection (moves.foldl (fun s m => m s) state)) ∧
    n = 3 :=
sorry

end NUMINAMATH_CALUDE_min_moves_to_align_cups_l2395_239554


namespace NUMINAMATH_CALUDE_sequence_properties_l2395_239586

def a (n : ℕ) : ℤ := 15 * n + 2 + (15 * n - 32) * 16^(n - 1)

theorem sequence_properties :
  (∀ n : ℕ, (15^3 : ℤ) ∣ a n) ∧
  (∀ n : ℕ, (1991 : ℤ) ∣ a n ∧ (1991 : ℤ) ∣ a (n + 1) ∧ (1991 : ℤ) ∣ a (n + 2) ↔ ∃ k : ℕ, n = 89595 * k) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2395_239586


namespace NUMINAMATH_CALUDE_probability_same_color_is_half_l2395_239591

def num_red_balls : ℕ := 2
def num_white_balls : ℕ := 2
def total_balls : ℕ := num_red_balls + num_white_balls

def num_possible_outcomes : ℕ := total_balls * total_balls
def num_same_color_outcomes : ℕ := num_red_balls * num_red_balls + num_white_balls * num_white_balls

def probability_same_color : ℚ := num_same_color_outcomes / num_possible_outcomes

theorem probability_same_color_is_half : probability_same_color = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_half_l2395_239591


namespace NUMINAMATH_CALUDE_zoo_layout_l2395_239542

theorem zoo_layout (tiger_enclosures : ℕ) (zebra_enclosures_per_tiger : ℕ) (giraffe_enclosure_ratio : ℕ)
  (tigers_per_enclosure : ℕ) (zebras_per_enclosure : ℕ) (total_animals : ℕ)
  (h1 : tiger_enclosures = 4)
  (h2 : zebra_enclosures_per_tiger = 2)
  (h3 : giraffe_enclosure_ratio = 3)
  (h4 : tigers_per_enclosure = 4)
  (h5 : zebras_per_enclosure = 10)
  (h6 : total_animals = 144) :
  (total_animals - (tiger_enclosures * tigers_per_enclosure + tiger_enclosures * zebra_enclosures_per_tiger * zebras_per_enclosure)) / 
  (giraffe_enclosure_ratio * tiger_enclosures * zebra_enclosures_per_tiger) = 2 :=
by sorry

end NUMINAMATH_CALUDE_zoo_layout_l2395_239542


namespace NUMINAMATH_CALUDE_twenty_two_oclock_is_ten_pm_l2395_239507

/-- Converts 24-hour time format to 12-hour time format -/
def convert_24_to_12 (hour : ℕ) : ℕ × String :=
  if hour < 12 then (if hour = 0 then 12 else hour, "AM")
  else ((if hour = 12 then 12 else hour - 12), "PM")

/-- Theorem stating that 22:00 in 24-hour format is equivalent to 10:00 PM in 12-hour format -/
theorem twenty_two_oclock_is_ten_pm :
  convert_24_to_12 22 = (10, "PM") :=
sorry

end NUMINAMATH_CALUDE_twenty_two_oclock_is_ten_pm_l2395_239507


namespace NUMINAMATH_CALUDE_only_solutions_for_exponential_equation_l2395_239550

theorem only_solutions_for_exponential_equation :
  ∀ a b : ℕ+, a ^ b.val = b.val ^ (a.val ^ 2) →
    ((a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 16) ∨ (a = 3 ∧ b = 27)) := by
  sorry

end NUMINAMATH_CALUDE_only_solutions_for_exponential_equation_l2395_239550


namespace NUMINAMATH_CALUDE_sum_of_other_y_coordinates_l2395_239516

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- 
Given a rectangle with two opposite vertices at (2, 10) and (8, -6),
the sum of the y-coordinates of the other two vertices is 4.
-/
theorem sum_of_other_y_coordinates (rect : Rectangle) 
  (h1 : rect.v1 = (2, 10))
  (h2 : rect.v3 = (8, -6))
  (h_opposite : (rect.v1 = (2, 10) ∧ rect.v3 = (8, -6)) ∨ 
                (rect.v2 = (2, 10) ∧ rect.v4 = (8, -6))) :
  (rect.v2).2 + (rect.v4).2 = 4 := by
  sorry

#check sum_of_other_y_coordinates

end NUMINAMATH_CALUDE_sum_of_other_y_coordinates_l2395_239516


namespace NUMINAMATH_CALUDE_equation_system_solution_l2395_239579

theorem equation_system_solution :
  ∃ (x y z : ℝ),
    (x / 6) * 12 = 10 ∧
    (y / 4) * 8 = x ∧
    (z / 3) * 5 + y = 20 ∧
    x = 5 ∧
    y = 5 / 2 ∧
    z = 21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l2395_239579


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2395_239562

-- System 1
theorem system_one_solution : 
  ∃ (x y : ℝ), y = x - 4 ∧ x + y = 6 ∧ x = 5 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution :
  ∃ (x y : ℝ), 2*x + y = 1 ∧ 4*x - y = 5 ∧ x = 1 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2395_239562


namespace NUMINAMATH_CALUDE_odd_function_zero_value_l2395_239518

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_zero_value (f : ℝ → ℝ) (h : OddFunction f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_value_l2395_239518


namespace NUMINAMATH_CALUDE_locus_of_Q_is_ellipse_l2395_239517

/-- The locus of point Q is an ellipse centered at P with the same eccentricity as the original ellipse -/
theorem locus_of_Q_is_ellipse (m n x₀ y₀ : ℝ) (hm : m > 0) (hn : n > 0) 
  (hP : m * x₀^2 + n * y₀^2 < 1) : 
  ∃ (x y : ℝ), m * (x - x₀)^2 + n * (y - y₀)^2 = m * x₀^2 + n * y₀^2 := by
  sorry

#check locus_of_Q_is_ellipse

end NUMINAMATH_CALUDE_locus_of_Q_is_ellipse_l2395_239517


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_l2395_239587

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of all digits displayed on the watch -/
def totalSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The largest possible sum of digits on a 24-hour format digital watch -/
def maxSumOfDigits : Nat := 23

theorem largest_sum_of_digits :
  ∀ t : Time24, totalSumOfDigits t ≤ maxSumOfDigits :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_of_digits_l2395_239587


namespace NUMINAMATH_CALUDE_a_range_l2395_239540

/-- The function f(x) = x^2 - 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

/-- f is increasing on [1, +∞) -/
def f_increasing (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f a x < f a y

theorem a_range (a : ℝ) (h : f_increasing a) : a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l2395_239540


namespace NUMINAMATH_CALUDE_power_17_2023_mod_26_l2395_239585

theorem power_17_2023_mod_26 : 17^2023 % 26 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_17_2023_mod_26_l2395_239585


namespace NUMINAMATH_CALUDE_product_87_93_l2395_239572

theorem product_87_93 : 87 * 93 = 8091 := by
  sorry

end NUMINAMATH_CALUDE_product_87_93_l2395_239572


namespace NUMINAMATH_CALUDE_clock_hands_alignment_l2395_239530

/-- The number of whole seconds remaining in an hour when the clock hands make equal angles with the vertical -/
def remaining_seconds : ℕ := by sorry

/-- The angle (in degrees) that the hour hand and minute hand make with the vertical when they align -/
def alignment_angle : ℚ := by sorry

theorem clock_hands_alignment :
  (alignment_angle * 120 : ℚ) = (360 - alignment_angle) * 10 ∧
  remaining_seconds = 3600 - Int.floor (alignment_angle * 120) := by sorry

end NUMINAMATH_CALUDE_clock_hands_alignment_l2395_239530


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l2395_239510

-- Define the number of volunteers and venues
def num_volunteers : ℕ := 5
def num_venues : ℕ := 3

-- Define a function to calculate the number of ways to distribute volunteers
def distribute_volunteers (volunteers : ℕ) (venues : ℕ) : ℕ :=
  -- This function should calculate the number of ways to distribute volunteers
  -- to venues, ensuring each venue gets at least one volunteer
  sorry

-- Theorem stating that the number of ways to distribute 5 volunteers to 3 venues is 150
theorem distribute_five_to_three :
  distribute_volunteers num_volunteers num_venues = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l2395_239510


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2395_239512

-- Define the custom operation
def otimes (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

-- Theorem statement
theorem unique_solution_exists :
  ∃! y : ℝ, otimes 2 y = 20 := by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2395_239512


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l2395_239570

def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - 2

theorem f_decreasing_on_interval (a b : ℝ) :
  (∀ x ∈ Set.Icc (1 + a) 2, f a b x = f a b (-x)) →
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, x < y → f a b x > f a b y) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l2395_239570


namespace NUMINAMATH_CALUDE_race_finish_orders_l2395_239596

def number_of_permutations (n : ℕ) : ℕ := Nat.factorial n

theorem race_finish_orders : number_of_permutations 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l2395_239596


namespace NUMINAMATH_CALUDE_regression_line_not_fixed_point_l2395_239551

/-- A regression line in the form y = bx + a -/
structure RegressionLine where
  a : ℝ
  b : ℝ

/-- Predicate to check if a point lies on a regression line -/
def lies_on (line : RegressionLine) (x y : ℝ) : Prop :=
  y = line.b * x + line.a

theorem regression_line_not_fixed_point :
  ∃ (line : RegressionLine), 
    (¬ lies_on line 0 0) ∧ 
    (∀ x, ¬ lies_on line x 0) ∧ 
    (∀ x y, ¬ lies_on line x y) :=
sorry

end NUMINAMATH_CALUDE_regression_line_not_fixed_point_l2395_239551


namespace NUMINAMATH_CALUDE_perimeter_semicircular_arcs_on_square_l2395_239561

/-- The perimeter of a region bounded by semicircular arcs on a square's sides -/
theorem perimeter_semicircular_arcs_on_square (side_length : Real) :
  side_length = 4 / Real.pi →
  (4 : Real) * (Real.pi * side_length / 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_semicircular_arcs_on_square_l2395_239561


namespace NUMINAMATH_CALUDE_new_drive_usage_percentage_l2395_239590

def initial_free_space : ℝ := 324
def initial_used_space : ℝ := 850
def initial_document_size : ℝ := 180
def initial_photo_size : ℝ := 380
def initial_video_size : ℝ := 290
def document_compression_ratio : ℝ := 0.05
def photo_compression_ratio : ℝ := 0.12
def video_compression_ratio : ℝ := 0.20
def deleted_photo_size : ℝ := 65.9
def deleted_video_size : ℝ := 98.1
def added_document_size : ℝ := 20.4
def added_photo_size : ℝ := 37.6
def new_drive_size : ℝ := 1500

theorem new_drive_usage_percentage (ε : ℝ) (hε : ε > 0) :
  ∃ (percentage : ℝ),
    abs (percentage - 43.56) < ε ∧
    percentage = 
      (((initial_document_size + added_document_size) * (1 - document_compression_ratio) +
        (initial_photo_size - deleted_photo_size + added_photo_size) * (1 - photo_compression_ratio) +
        (initial_video_size - deleted_video_size) * (1 - video_compression_ratio)) /
       new_drive_size) * 100 :=
by sorry

end NUMINAMATH_CALUDE_new_drive_usage_percentage_l2395_239590


namespace NUMINAMATH_CALUDE_ladybugs_without_spots_l2395_239577

theorem ladybugs_without_spots (total : ℕ) (with_spots : ℕ) (without_spots : ℕ) : 
  total = 67082 → with_spots = 12170 → without_spots = total - with_spots → without_spots = 54912 := by
  sorry

end NUMINAMATH_CALUDE_ladybugs_without_spots_l2395_239577


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l2395_239533

theorem merchant_profit_percentage (C S : ℝ) (h : C > 0) :
  20 * C = 15 * S →
  (S - C) / C * 100 = 100/3 :=
by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l2395_239533


namespace NUMINAMATH_CALUDE_investment_problem_l2395_239569

/-- Proves that Raghu's investment is 2656.25 given the conditions of the problem -/
theorem investment_problem (raghu : ℝ) : 
  let trishul := 0.9 * raghu
  let vishal := 1.1 * trishul
  let chandni := 1.15 * vishal
  (raghu + trishul + vishal + chandni = 10700) →
  raghu = 2656.25 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l2395_239569


namespace NUMINAMATH_CALUDE_parallel_vectors_l2395_239531

/-- Two vectors in ℝ² -/
def a (m : ℝ) : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (2, -3)

/-- Definition of parallel vectors in ℝ² -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

/-- Theorem: If a(m) is parallel to b, then m = -3/2 -/
theorem parallel_vectors (m : ℝ) :
  parallel (a m) b → m = -3/2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2395_239531


namespace NUMINAMATH_CALUDE_school_trip_seats_l2395_239529

/-- Given a total number of students and buses, calculate the number of seats per bus -/
def seatsPerBus (students : ℕ) (buses : ℕ) : ℚ :=
  (students : ℚ) / (buses : ℚ)

/-- Theorem: Given 14 students and 7 buses, the number of seats on each bus is 2 -/
theorem school_trip_seats :
  seatsPerBus 14 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_seats_l2395_239529


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2395_239502

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^15 + 11^21) ∧ ∀ q, Nat.Prime q → q ∣ (3^15 + 11^21) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2395_239502


namespace NUMINAMATH_CALUDE_late_fee_is_124_l2395_239503

/-- Calculates the late fee per month for the second bill given the total amount owed and details of three bills. -/
def calculate_late_fee (total_owed : ℚ) (bill1_amount : ℚ) (bill1_interest_rate : ℚ) (bill1_months : ℕ)
  (bill2_amount : ℚ) (bill2_months : ℕ) (bill3_fee1 : ℚ) (bill3_fee2 : ℚ) : ℚ :=
  let bill1_total := bill1_amount + bill1_amount * bill1_interest_rate * bill1_months
  let bill3_total := bill3_fee1 + bill3_fee2
  let bill2_total := total_owed - bill1_total - bill3_total
  (bill2_total - bill2_amount) / bill2_months

/-- Theorem stating that the late fee per month for the second bill is $124. -/
theorem late_fee_is_124 :
  calculate_late_fee 1234 200 (1/10) 2 130 6 40 80 = 124 := by
  sorry

end NUMINAMATH_CALUDE_late_fee_is_124_l2395_239503


namespace NUMINAMATH_CALUDE_f_monotone_increasing_iff_l2395_239552

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1) + 1

theorem f_monotone_increasing_iff (x : ℝ) :
  StrictMono (fun y => f y) ↔ x ∈ Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_iff_l2395_239552


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l2395_239580

theorem exactly_one_positive_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 20 - 5 * n > 12 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l2395_239580


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2395_239521

theorem intersection_of_sets : 
  let M : Set Char := {a, b, c}
  let N : Set Char := {b, c, d}
  M ∩ N = {b, c} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2395_239521


namespace NUMINAMATH_CALUDE_first_quarter_homework_points_l2395_239520

theorem first_quarter_homework_points :
  ∀ (homework quiz test : ℕ),
    homework + quiz + test = 265 →
    test = 4 * quiz →
    quiz = homework + 5 →
    homework = 40 := by
  sorry

end NUMINAMATH_CALUDE_first_quarter_homework_points_l2395_239520


namespace NUMINAMATH_CALUDE_consecutive_triangle_sides_l2395_239546

theorem consecutive_triangle_sides (n : ℕ) (h : n ≥ 1) :
  ∀ (a : ℕ → ℕ), (∀ i, i < 6*n → a (i+1) = a i + 1) →
  ∃ i, i + 2 < 6*n ∧ 
    a i + a (i+1) > a (i+2) ∧
    a i + a (i+2) > a (i+1) ∧
    a (i+1) + a (i+2) > a i :=
sorry

end NUMINAMATH_CALUDE_consecutive_triangle_sides_l2395_239546


namespace NUMINAMATH_CALUDE_prime_iff_divides_factorial_plus_one_l2395_239548

theorem prime_iff_divides_factorial_plus_one (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ n ∣ (Nat.factorial (n - 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_iff_divides_factorial_plus_one_l2395_239548


namespace NUMINAMATH_CALUDE_unique_non_range_value_l2395_239556

/-- The function f defined by the given properties -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- Theorem stating that 58 is the unique number not in the range of f -/
theorem unique_non_range_value
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h_19 : f a b c d 19 = 19)
  (h_97 : f a b c d 97 = 97)
  (h_inverse : ∀ x ≠ -d/c, f a b c d (f a b c d x) = x) :
  ∃! y, ∀ x, f a b c d x ≠ y ∧ y = 58 := by
  sorry

end NUMINAMATH_CALUDE_unique_non_range_value_l2395_239556


namespace NUMINAMATH_CALUDE_cos_165_degrees_l2395_239563

theorem cos_165_degrees : Real.cos (165 * π / 180) = -((Real.sqrt 6 + Real.sqrt 2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_cos_165_degrees_l2395_239563


namespace NUMINAMATH_CALUDE_point_symmetry_product_l2395_239564

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given that:
    - Point A lies on the y-axis with coordinates (3a-8, -3)
    - Points A and B(0, b) are symmetric with respect to the x-axis
    Prove that ab = 8 -/
theorem point_symmetry_product (a b : ℝ) : 
  let A : Point := ⟨3*a - 8, -3⟩
  let B : Point := ⟨0, b⟩
  (A.x = 0) →  -- A lies on the y-axis
  (A.y = -B.y) →  -- A and B are symmetric with respect to the x-axis
  a * b = 8 := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_product_l2395_239564
