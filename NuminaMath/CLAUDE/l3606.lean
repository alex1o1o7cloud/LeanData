import Mathlib

namespace NUMINAMATH_CALUDE_circle_configurations_exist_l3606_360668

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the theorem
theorem circle_configurations_exist :
  ∃ (A B : ℝ × ℝ) (a b : ℝ),
    a > b ∧
    ∃ (AB : ℝ),
      AB > 0 ∧
      (a - b < AB) ∧
      (∃ AB', AB' > 0 ∧ a + b = AB') ∧
      (∃ AB'', AB'' > 0 ∧ a + b < AB'') ∧
      (∃ AB''', AB''' > 0 ∧ a - b = AB''') :=
by sorry

end NUMINAMATH_CALUDE_circle_configurations_exist_l3606_360668


namespace NUMINAMATH_CALUDE_water_left_l3606_360638

theorem water_left (initial_water : ℚ) (used_water : ℚ) (water_left : ℚ) : 
  initial_water = 3 ∧ used_water = 11/4 → water_left = initial_water - used_water → water_left = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_water_left_l3606_360638


namespace NUMINAMATH_CALUDE_product_95_105_l3606_360630

theorem product_95_105 : 95 * 105 = 9975 := by
  sorry

end NUMINAMATH_CALUDE_product_95_105_l3606_360630


namespace NUMINAMATH_CALUDE_jerrys_breakfast_l3606_360672

theorem jerrys_breakfast (pancake_calories : ℕ) (bacon_calories : ℕ) (cereal_calories : ℕ) 
  (total_calories : ℕ) (bacon_strips : ℕ) :
  pancake_calories = 120 →
  bacon_calories = 100 →
  cereal_calories = 200 →
  total_calories = 1120 →
  bacon_strips = 2 →
  ∃ (num_pancakes : ℕ), 
    num_pancakes * pancake_calories + 
    bacon_strips * bacon_calories + 
    cereal_calories = total_calories ∧
    num_pancakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_breakfast_l3606_360672


namespace NUMINAMATH_CALUDE_complex_simplification_l3606_360616

/-- Given that i is the imaginary unit, prove that 
    (4*I)/((1-I)^2 + 2) + I^2018 = -2 + I -/
theorem complex_simplification :
  (4 * I) / ((1 - I)^2 + 2) + I^2018 = -2 + I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l3606_360616


namespace NUMINAMATH_CALUDE_largest_increase_2006_2007_l3606_360640

def students : Fin 5 → ℕ
  | 0 => 80  -- 2003
  | 1 => 88  -- 2004
  | 2 => 94  -- 2005
  | 3 => 106 -- 2006
  | 4 => 130 -- 2007

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseYears : Fin 4 := 3

theorem largest_increase_2006_2007 :
  ∀ i : Fin 4, percentageIncrease (students i) (students (i + 1)) ≤ 
    percentageIncrease (students largestIncreaseYears) (students (largestIncreaseYears + 1)) :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_2006_2007_l3606_360640


namespace NUMINAMATH_CALUDE_num_grade_assignments_l3606_360620

/-- The number of students in the class -/
def num_students : ℕ := 10

/-- The number of possible grades (A, B, C) -/
def num_grades : ℕ := 3

/-- Theorem: The number of ways to assign grades to all students -/
theorem num_grade_assignments : (num_grades ^ num_students : ℕ) = 59049 := by
  sorry

end NUMINAMATH_CALUDE_num_grade_assignments_l3606_360620


namespace NUMINAMATH_CALUDE_students_studying_both_subjects_l3606_360622

theorem students_studying_both_subjects (total : ℕ) 
  (physics_min physics_max chemistry_min chemistry_max : ℕ) : 
  total = 2500 →
  physics_min = 1750 →
  physics_max = 1875 →
  chemistry_min = 875 →
  chemistry_max = 1125 →
  ∃ (m M : ℕ),
    m = physics_min + chemistry_min - total ∧
    M = physics_max + chemistry_max - total ∧
    M - m = 375 :=
by sorry

end NUMINAMATH_CALUDE_students_studying_both_subjects_l3606_360622


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3606_360603

theorem fractional_equation_solution :
  ∃! x : ℚ, x ≠ 1 ∧ x ≠ -1 ∧ (x / (x + 1) - 1 = 3 / (x - 1)) :=
by
  use (-1/2)
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3606_360603


namespace NUMINAMATH_CALUDE_flagpole_shadow_length_l3606_360618

/-- Given a flagpole and a building under similar conditions, prove the length of the flagpole's shadow. -/
theorem flagpole_shadow_length 
  (flagpole_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : building_height = 26)
  (h3 : building_shadow = 65)
  : ∃ (flagpole_shadow : ℝ), flagpole_shadow = 45 ∧ 
    flagpole_height / flagpole_shadow = building_height / building_shadow :=
by sorry

end NUMINAMATH_CALUDE_flagpole_shadow_length_l3606_360618


namespace NUMINAMATH_CALUDE_tangent_line_logarithm_l3606_360617

theorem tangent_line_logarithm (x y : ℝ) :
  let f : ℝ → ℝ := λ t => Real.log t
  let tangent_line : ℝ → ℝ → Prop := λ a b => x - y - 1 = 0
  let perpendicular_line : ℝ → ℝ → Prop := λ a b => b = -a
  ∃ x₀ y₀ : ℝ,
    (y₀ = f x₀) ∧
    (∀ t : ℝ, (deriv f) x₀ * (deriv (λ s => -(s : ℝ))) t = -1) ∧
    tangent_line x₀ y₀ :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_line_logarithm_l3606_360617


namespace NUMINAMATH_CALUDE_team_formation_theorem_l3606_360613

/-- The number of ways to form a team with at least one female student -/
def team_formation_count (male_count : ℕ) (female_count : ℕ) (team_size : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to form the team under given conditions -/
theorem team_formation_theorem :
  team_formation_count 5 3 4 = 780 :=
sorry

end NUMINAMATH_CALUDE_team_formation_theorem_l3606_360613


namespace NUMINAMATH_CALUDE_student_activity_arrangements_l3606_360653

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of arrangements for distributing students between two activities -/
def total_arrangements (n : ℕ) : ℕ :=
  choose n 4 + choose n 3 + choose n 2

theorem student_activity_arrangements :
  total_arrangements 6 = 50 := by sorry

end NUMINAMATH_CALUDE_student_activity_arrangements_l3606_360653


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l3606_360650

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 - |x+a| -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 - |x + a|

/-- If f(x) = x^2 - |x+a| is an even function, then a = 0 -/
theorem even_function_implies_a_zero (a : ℝ) :
  IsEven (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l3606_360650


namespace NUMINAMATH_CALUDE_balloon_permutations_count_l3606_360609

/-- The number of distinct permutations of the letters in "BALLOON" -/
def balloon_permutations : ℕ := 1260

/-- The total number of letters in "BALLOON" -/
def total_letters : ℕ := 7

/-- The number of repeated 'L's in "BALLOON" -/
def repeated_L : ℕ := 2

/-- The number of repeated 'O's in "BALLOON" -/
def repeated_O : ℕ := 2

/-- Theorem stating that the number of distinct permutations of the letters in "BALLOON" is 1260 -/
theorem balloon_permutations_count :
  balloon_permutations = Nat.factorial total_letters / (Nat.factorial repeated_L * Nat.factorial repeated_O) :=
by sorry

end NUMINAMATH_CALUDE_balloon_permutations_count_l3606_360609


namespace NUMINAMATH_CALUDE_sticks_for_800_hexagons_l3606_360639

/-- The number of sticks required to form a row of n hexagons -/
def sticksForHexagons (n : ℕ) : ℕ :=
  if n = 0 then 0 else 6 + 5 * (n - 1)

/-- Theorem: The number of sticks required for 800 hexagons is 4001 -/
theorem sticks_for_800_hexagons : sticksForHexagons 800 = 4001 := by
  sorry

#eval sticksForHexagons 800  -- To verify the result

end NUMINAMATH_CALUDE_sticks_for_800_hexagons_l3606_360639


namespace NUMINAMATH_CALUDE_man_age_difference_l3606_360651

/-- Proves that a man is 22 years older than his son given certain conditions -/
theorem man_age_difference (man_age son_age : ℕ) : 
  son_age = 20 →
  man_age > son_age →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_man_age_difference_l3606_360651


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3606_360660

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 2 + a 4 + a 9 + a 11 = 32 →
  a 6 + a 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3606_360660


namespace NUMINAMATH_CALUDE_class_size_l3606_360691

theorem class_size (dog_video_percentage : ℚ) (dog_movie_percentage : ℚ) (dog_preference_count : ℕ) :
  dog_video_percentage = 1/2 →
  dog_movie_percentage = 1/10 →
  dog_preference_count = 18 →
  (dog_video_percentage + dog_movie_percentage) * ↑dog_preference_count / (dog_video_percentage + dog_movie_percentage) = 30 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l3606_360691


namespace NUMINAMATH_CALUDE_hex_B1C_equals_2844_l3606_360635

/-- Converts a hexadecimal digit to its decimal value -/
def hexToDecimal (c : Char) : Nat :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => c.toString.toNat!

/-- Converts a hexadecimal string to its decimal value -/
def hexStringToDecimal (s : String) : Nat :=
  s.foldl (fun acc c => 16 * acc + hexToDecimal c) 0

/-- The hexadecimal number B1C is equal to 2844 in decimal -/
theorem hex_B1C_equals_2844 : hexStringToDecimal "B1C" = 2844 := by
  sorry

end NUMINAMATH_CALUDE_hex_B1C_equals_2844_l3606_360635


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l3606_360633

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1 : ℤ) * d

theorem fiftieth_term_of_sequence (a₁ d : ℤ) (h₁ : a₁ = 48) (h₂ : d = -2) :
  arithmeticSequenceTerm a₁ d 50 = -50 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l3606_360633


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l3606_360689

/-- Given points A, B, and C in a plane, with D as the midpoint of AC, 
    prove that the sum of the slope and y-intercept of the line through C and D is 36/5 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 8) → B = (0, 0) → C = (10, 0) → 
  D = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  (D.2 - C.2) / (D.1 - C.1) + (D.2 - (D.2 - C.2) / (D.1 - C.1) * D.1) = 36 / 5 :=
by sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l3606_360689


namespace NUMINAMATH_CALUDE_jacksons_vacuuming_time_l3606_360600

/-- Represents the problem of calculating Jackson's vacuuming time --/
theorem jacksons_vacuuming_time (vacuum_time : ℝ) : 
  vacuum_time = 2 :=
by
  have hourly_rate : ℝ := 5
  have dish_washing_time : ℝ := 0.5
  have bathroom_cleaning_time : ℝ := 3 * dish_washing_time
  have total_earnings : ℝ := 30
  have total_chore_time : ℝ := 2 * vacuum_time + dish_washing_time + bathroom_cleaning_time
  
  have h1 : hourly_rate * total_chore_time = total_earnings :=
    sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_jacksons_vacuuming_time_l3606_360600


namespace NUMINAMATH_CALUDE_cos_2x_value_l3606_360636

theorem cos_2x_value (x : Real) (h : 2 * Real.sin x + Real.cos (π / 2 - x) = 1) :
  Real.cos (2 * x) = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_cos_2x_value_l3606_360636


namespace NUMINAMATH_CALUDE_fraction_unchanged_l3606_360619

theorem fraction_unchanged (x y : ℝ) : 
  (2 * x) / (3 * x - 2 * y) = (4 * x) / (6 * x - 4 * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l3606_360619


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_l3606_360681

/-- If x^2 - 10x + m is a perfect square trinomial, then m = 25 -/
theorem perfect_square_trinomial_m (m : ℝ) : 
  (∃ a b : ℝ, ∀ x, x^2 - 10*x + m = (a*x + b)^2) → m = 25 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_l3606_360681


namespace NUMINAMATH_CALUDE_point_symmetry_l3606_360697

/-- Given three points A, B, and P in a 2D Cartesian coordinate system,
    prove that if B has coordinates (1, 2), P is symmetric to A with respect to the x-axis,
    and P is symmetric to B with respect to the y-axis, then A has coordinates (-1, -2). -/
theorem point_symmetry (A B P : ℝ × ℝ) : 
  B = (1, 2) → 
  P.1 = A.1 ∧ P.2 = -A.2 →  -- P is symmetric to A with respect to x-axis
  P.1 = -B.1 ∧ P.2 = B.2 →  -- P is symmetric to B with respect to y-axis
  A = (-1, -2) := by
sorry

end NUMINAMATH_CALUDE_point_symmetry_l3606_360697


namespace NUMINAMATH_CALUDE_min_sum_of_squares_on_line_l3606_360657

theorem min_sum_of_squares_on_line (m n : ℝ) (h : m + n = 1) : 
  m^2 + n^2 ≥ 4 ∧ ∃ (m₀ n₀ : ℝ), m₀ + n₀ = 1 ∧ m₀^2 + n₀^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_on_line_l3606_360657


namespace NUMINAMATH_CALUDE_parabola_through_point_l3606_360698

-- Define a parabola type
structure Parabola where
  -- A parabola is defined by its equation
  equation : ℝ → ℝ → Prop

-- Define the condition that a parabola passes through a point
def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

-- Define the two possible standard forms of a parabola
def vertical_parabola (a : ℝ) : Parabola :=
  ⟨λ x y => y^2 = 4*a*x⟩

def horizontal_parabola (b : ℝ) : Parabola :=
  ⟨λ x y => x^2 = 4*b*y⟩

-- The theorem to be proved
theorem parabola_through_point :
  ∃ (p : Parabola), passes_through p (-2) 4 ∧
    ((∃ a : ℝ, p = vertical_parabola a ∧ a = -2) ∨
     (∃ b : ℝ, p = horizontal_parabola b ∧ b = 1/4)) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_point_l3606_360698


namespace NUMINAMATH_CALUDE_difference_x_y_l3606_360631

theorem difference_x_y (x y : ℤ) 
  (sum_eq : x + y = 20)
  (diff_eq : x - y = 10)
  (x_val : x = 15) :
  x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_x_y_l3606_360631


namespace NUMINAMATH_CALUDE_multiple_solutions_exist_l3606_360682

theorem multiple_solutions_exist : ∃ p₁ p₂ : ℝ, 
  p₁ ≠ p₂ ∧ 
  p₁ ∈ Set.Ioo 0 1 ∧ 
  p₂ ∈ Set.Ioo 0 1 ∧
  10 * p₁^3 * (1 - p₁)^2 = 144/625 ∧
  10 * p₂^3 * (1 - p₂)^2 = 144/625 :=
sorry

end NUMINAMATH_CALUDE_multiple_solutions_exist_l3606_360682


namespace NUMINAMATH_CALUDE_total_crickets_l3606_360659

/-- The total number of crickets given an initial and additional amount -/
theorem total_crickets (initial : ℝ) (additional : ℝ) :
  initial = 7.5 → additional = 11.25 → initial + additional = 18.75 :=
by sorry

end NUMINAMATH_CALUDE_total_crickets_l3606_360659


namespace NUMINAMATH_CALUDE_alyssas_soccer_games_l3606_360601

theorem alyssas_soccer_games (games_this_year games_next_year total_games : ℕ) 
  (h1 : games_this_year = 11)
  (h2 : games_next_year = 15)
  (h3 : total_games = 39) :
  total_games - games_this_year - games_next_year = 13 := by
  sorry

end NUMINAMATH_CALUDE_alyssas_soccer_games_l3606_360601


namespace NUMINAMATH_CALUDE_inequality_condition_l3606_360680

theorem inequality_condition (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 2| + |x - 5| + |x - 10| < b) ↔ b > 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3606_360680


namespace NUMINAMATH_CALUDE_target_avg_income_l3606_360656

def past_income : List ℝ := [406, 413, 420, 436, 395]
def next_weeks : ℕ := 5
def total_weeks : ℕ := 10
def next_avg_income : ℝ := 586

theorem target_avg_income :
  let past_total := past_income.sum
  let next_total := next_avg_income * next_weeks
  let total_income := past_total + next_total
  (total_income / total_weeks : ℝ) = 500 := by sorry

end NUMINAMATH_CALUDE_target_avg_income_l3606_360656


namespace NUMINAMATH_CALUDE_sum_of_digits_l3606_360646

def is_valid_arrangement (digits : Finset ℕ) (vertical horizontal : Finset ℕ) : Prop :=
  digits.card = 7 ∧ 
  digits ⊆ Finset.range 9 ∧ 
  vertical.card = 4 ∧ 
  horizontal.card = 4 ∧ 
  (vertical ∩ horizontal).card = 1 ∧
  vertical ⊆ digits ∧ 
  horizontal ⊆ digits

theorem sum_of_digits 
  (digits : Finset ℕ) 
  (vertical horizontal : Finset ℕ) 
  (h_valid : is_valid_arrangement digits vertical horizontal)
  (h_vertical_sum : vertical.sum id = 26)
  (h_horizontal_sum : horizontal.sum id = 20) :
  digits.sum id = 32 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l3606_360646


namespace NUMINAMATH_CALUDE_inequality_relation_l3606_360683

theorem inequality_relation (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, (1 / x < a ∧ x ≤ 1 / a)) ∧
  (∀ x : ℝ, x > 1 / a → 1 / x < a) :=
sorry

end NUMINAMATH_CALUDE_inequality_relation_l3606_360683


namespace NUMINAMATH_CALUDE_total_height_calculation_l3606_360610

-- Define the heights in inches
def sculpture_height_inches : ℚ := 34
def base_height_inches : ℚ := 2

-- Define the conversion factor from inches to centimeters
def inches_to_cm : ℚ := 2.54

-- Define the total height in centimeters
def total_height_cm : ℚ := (sculpture_height_inches + base_height_inches) * inches_to_cm

-- Theorem statement
theorem total_height_calculation :
  total_height_cm = 91.44 := by sorry

end NUMINAMATH_CALUDE_total_height_calculation_l3606_360610


namespace NUMINAMATH_CALUDE_books_sold_to_store_l3606_360625

def book_problem (initial_books : ℕ) (book_club_months : ℕ) (bookstore_books : ℕ) 
  (yard_sale_books : ℕ) (daughter_books : ℕ) (mother_books : ℕ) (donated_books : ℕ) 
  (final_books : ℕ) : ℕ :=
  let total_acquired := initial_books + book_club_months + bookstore_books + 
                        yard_sale_books + daughter_books + mother_books
  let before_selling := total_acquired - donated_books
  before_selling - final_books

theorem books_sold_to_store : 
  book_problem 72 12 5 2 1 4 12 81 = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_to_store_l3606_360625


namespace NUMINAMATH_CALUDE_positive_solution_sum_l3606_360627

theorem positive_solution_sum (a b : ℕ+) (x : ℤ) : 
  x > 0 → x = Int.sqrt a - b → x^2 - 10*x = 39 → a + b = 69 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_sum_l3606_360627


namespace NUMINAMATH_CALUDE_digit_equation_solutions_l3606_360663

theorem digit_equation_solutions (n : ℕ) (hn : n ≥ 2) :
  let a (x : ℕ) := x * (10^n - 1) / 9
  let b (y : ℕ) := y * (10^n - 1) / 9
  let c (z : ℕ) := z * (10^(2*n) - 1) / 9
  ∀ x y z : ℕ, (a x)^2 + b y = c z →
    ((x = 3 ∧ y = 2 ∧ z = 1) ∨
     (x = 6 ∧ y = 8 ∧ z = 4) ∨
     (x = 8 ∧ y = 3 ∧ z = 7)) :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solutions_l3606_360663


namespace NUMINAMATH_CALUDE_john_drinks_two_cups_per_day_l3606_360688

-- Define the constants
def gallon_to_ounce : ℚ := 128
def cup_to_ounce : ℚ := 8
def days_between_purchases : ℚ := 4
def gallons_per_purchase : ℚ := 1/2

-- Define the function to calculate cups per day
def cups_per_day : ℚ :=
  (gallons_per_purchase * gallon_to_ounce) / (days_between_purchases * cup_to_ounce)

-- Theorem statement
theorem john_drinks_two_cups_per_day :
  cups_per_day = 2 := by sorry

end NUMINAMATH_CALUDE_john_drinks_two_cups_per_day_l3606_360688


namespace NUMINAMATH_CALUDE_concentric_circles_chords_l3606_360621

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two consecutive chords is 60°, then the number of chords needed to
    complete a full rotation is 3. -/
theorem concentric_circles_chords (angle : ℝ) (n : ℕ) : 
  angle = 60 → n * angle = 360 → n = 3 := by sorry

end NUMINAMATH_CALUDE_concentric_circles_chords_l3606_360621


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l3606_360604

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a)) ≥ 3 / Real.rpow 162 (1/3) :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b) + b / (6 * c) + c / (9 * a)) = 3 / Real.rpow 162 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l3606_360604


namespace NUMINAMATH_CALUDE_team_selection_l3606_360655

theorem team_selection (m : ℕ) : 
  (0 ≤ 14 - m) ∧ 
  (14 - m ≤ 2 * m) ∧ 
  (0 ≤ 5 * m - 11) ∧ 
  (5 * m - 11 ≤ 3 * m) →
  (m = 5) ∧ 
  (Nat.choose 10 9 * Nat.choose 15 14 = 150) := by
sorry


end NUMINAMATH_CALUDE_team_selection_l3606_360655


namespace NUMINAMATH_CALUDE_garden_length_l3606_360645

/-- Given a square playground and a rectangular garden, proves the length of the garden
    when the total fencing is known. -/
theorem garden_length
  (playground_side : ℕ)
  (garden_width : ℕ)
  (total_fencing : ℕ)
  (h1 : playground_side = 27)
  (h2 : garden_width = 9)
  (h3 : total_fencing = 150)
  (h4 : 4 * playground_side + 2 * garden_width + 2 * (total_fencing - 4 * playground_side - 2 * garden_width) / 2 = total_fencing) :
  (total_fencing - 4 * playground_side - 2 * garden_width) / 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_garden_length_l3606_360645


namespace NUMINAMATH_CALUDE_area_of_inscribed_hexagon_l3606_360674

/-- The area of a regular hexagon inscribed in a circle with radius 3 units is 13.5√3 square units. -/
theorem area_of_inscribed_hexagon : 
  let r : ℝ := 3  -- radius of the circle
  let hexagon_area : ℝ := 6 * (r^2 * Real.sqrt 3 / 4)  -- area of hexagon as 6 times the area of an equilateral triangle
  hexagon_area = 13.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribed_hexagon_l3606_360674


namespace NUMINAMATH_CALUDE_remainder_s_mod_6_l3606_360667

theorem remainder_s_mod_6 (s t : ℕ) (hs : s > t) (h_mod : (s - t) % 6 = 5) : s % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_s_mod_6_l3606_360667


namespace NUMINAMATH_CALUDE_count_monomials_l3606_360643

-- Define what a monomial is
def is_monomial (expr : String) : Bool :=
  match expr with
  | "0" => true
  | "2x-1" => false
  | "a" => true
  | "1/x" => false
  | "-2/3" => true
  | "(x-y)/2" => false
  | "2x/5" => true
  | _ => false

-- Define the set of expressions
def expressions : List String :=
  ["0", "2x-1", "a", "1/x", "-2/3", "(x-y)/2", "2x/5"]

-- Theorem statement
theorem count_monomials :
  (expressions.filter is_monomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_count_monomials_l3606_360643


namespace NUMINAMATH_CALUDE_sum_of_angles_equals_540_l3606_360692

-- Define the angles as real numbers
variable (a b c d e f g : ℝ)

-- Define the straight lines (we don't need to explicitly define them, 
-- but we'll use their properties in the theorem statement)

-- State the theorem
theorem sum_of_angles_equals_540 :
  a + b + c + d + e + f + g = 540 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_angles_equals_540_l3606_360692


namespace NUMINAMATH_CALUDE_no_real_solution_log_equation_l3606_360664

theorem no_real_solution_log_equation :
  ¬∃ x : ℝ, (Real.log (x + 6) + Real.log (x - 2) = Real.log (x^2 - 3*x - 18)) ∧
             (x + 6 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 3*x - 18 > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_log_equation_l3606_360664


namespace NUMINAMATH_CALUDE_no_four_digit_perfect_square_with_condition_l3606_360606

theorem no_four_digit_perfect_square_with_condition : ¬ ∃ (abcd : ℕ), 
  (1000 ≤ abcd ∧ abcd ≤ 9999) ∧  -- four-digit number
  (∃ (n : ℕ), abcd = n^2) ∧  -- perfect square
  (∃ (ab cd : ℕ), 
    (10 ≤ ab ∧ ab ≤ 99) ∧  -- ab is two-digit
    (10 ≤ cd ∧ cd ≤ 99) ∧  -- cd is two-digit
    (abcd = 100 * ab + cd) ∧  -- abcd is composed of ab and cd
    (ab = cd / 4)) :=  -- given condition
by sorry


end NUMINAMATH_CALUDE_no_four_digit_perfect_square_with_condition_l3606_360606


namespace NUMINAMATH_CALUDE_total_games_in_season_l3606_360658

/-- The number of hockey games per month -/
def games_per_month : ℕ := 13

/-- The number of months in the hockey season -/
def months_in_season : ℕ := 14

/-- The total number of hockey games in the season -/
def total_games : ℕ := games_per_month * months_in_season

/-- Theorem stating that the total number of hockey games in the season is 182 -/
theorem total_games_in_season : total_games = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_season_l3606_360658


namespace NUMINAMATH_CALUDE_race_heartbeats_l3606_360647

/-- Calculates the total number of heartbeats during a race given the race distance, cycling pace, and heart rate. -/
def total_heartbeats (race_distance : ℕ) (cycling_pace : ℕ) (heart_rate : ℕ) : ℕ :=
  race_distance * cycling_pace * heart_rate

/-- Theorem stating that for a 100-mile race, with a cycling pace of 4 minutes per mile and a heart rate of 120 beats per minute, the total number of heartbeats is 48000. -/
theorem race_heartbeats :
  total_heartbeats 100 4 120 = 48000 := by
  sorry

end NUMINAMATH_CALUDE_race_heartbeats_l3606_360647


namespace NUMINAMATH_CALUDE_age_problem_l3606_360629

theorem age_problem (age : ℕ) : 5 * (age + 5) - 5 * (age - 5) = age → age = 50 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l3606_360629


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l3606_360675

/-- Given a bag with red and yellow balls, calculate the probability of drawing a red ball -/
theorem probability_of_red_ball (num_red : ℕ) (num_yellow : ℕ) :
  num_red = 6 → num_yellow = 3 →
  (num_red : ℚ) / (num_red + num_yellow : ℚ) = 2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l3606_360675


namespace NUMINAMATH_CALUDE_point_on_line_with_equal_distances_quadrant_l3606_360648

/-- A point with coordinates (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line y = 2x + 3 -/
def lineEquation (p : Point) : Prop :=
  p.y = 2 * p.x + 3

/-- Equal distance to both coordinate axes -/
def equalDistanceToAxes (p : Point) : Prop :=
  abs p.x = abs p.y

/-- Second quadrant -/
def inSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Third quadrant -/
def inThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: A point on the line y = 2x + 3 with equal distances to both axes is in the second or third quadrant -/
theorem point_on_line_with_equal_distances_quadrant (p : Point) 
  (h1 : lineEquation p) (h2 : equalDistanceToAxes p) : 
  inSecondQuadrant p ∨ inThirdQuadrant p := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_with_equal_distances_quadrant_l3606_360648


namespace NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l3606_360676

/-- The function f(x) defined as x^2 + 3x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

/-- Theorem stating that the largest value of c such that -2 is in the range of f(x) = x^2 + 3x + c is 1/4 -/
theorem largest_c_for_negative_two_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), (∃ (x : ℝ), f d x = -2) → d ≤ c) ∧
  (∃ (x : ℝ), f (1/4) x = -2) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l3606_360676


namespace NUMINAMATH_CALUDE_unique_quadrilateral_perimeter_unique_perimeter_value_l3606_360686

/-- Represents a quadrilateral with integer side lengths -/
structure Quadrilateral where
  AB : ℕ+
  BC : ℕ+
  CD : ℕ+
  AD : ℕ+

/-- The perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℕ :=
  q.AB.val + q.BC.val + q.CD.val + q.AD.val

/-- Theorem stating that there is a unique quadrilateral satisfying the given conditions -/
theorem unique_quadrilateral_perimeter :
  ∃! (q : Quadrilateral),
    q.AB = 3 ∧
    q.BC = q.AD - 1 ∧
    q.BC = q.CD - 1 ∧
    (q.AB ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    (q.CD ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    perimeter q = 17 :=
  sorry

/-- Corollary: The perimeter of the unique quadrilateral is 17 -/
theorem unique_perimeter_value (p : ℕ) :
  (∃ (q : Quadrilateral),
    q.AB = 3 ∧
    q.BC = q.AD - 1 ∧
    q.BC = q.CD - 1 ∧
    (q.AB ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    (q.CD ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    perimeter q = p) →
  p = 17 :=
  sorry

end NUMINAMATH_CALUDE_unique_quadrilateral_perimeter_unique_perimeter_value_l3606_360686


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l3606_360634

def is_vertex (f : ℝ → ℝ) (x₀ y₀ : ℝ) : Prop :=
  ∀ x, f x ≥ f x₀ ∧ f x₀ = y₀

def has_vertical_symmetry_axis (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f (x₀ + x) = f (x₀ - x)

theorem quadratic_coefficients 
  (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = a * x^2 + b * x + c) →
  is_vertex f (-2) 5 →
  has_vertical_symmetry_axis f (-2) →
  f 0 = 9 →
  a = 1 ∧ b = 4 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l3606_360634


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l3606_360614

theorem wrong_mark_calculation (n : ℕ) (initial_avg correct_avg : ℚ) (correct_mark : ℕ) :
  n = 30 ∧ 
  initial_avg = 60 ∧ 
  correct_avg = 57.5 ∧ 
  correct_mark = 15 →
  ∃ wrong_mark : ℕ,
    (n * initial_avg - wrong_mark + correct_mark) / n = correct_avg ∧
    wrong_mark = 90 := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l3606_360614


namespace NUMINAMATH_CALUDE_staircase_carpet_cost_l3606_360687

/-- Represents the dimensions and cost parameters of a staircase -/
structure Staircase where
  num_steps : ℕ
  step_height : ℝ
  step_depth : ℝ
  width : ℝ
  carpet_cost_per_sqm : ℝ

/-- Calculates the cost of carpeting a staircase -/
def carpet_cost (s : Staircase) : ℝ :=
  let total_height := s.num_steps * s.step_height
  let total_depth := s.num_steps * s.step_depth
  let combined_length := total_height + total_depth
  let total_area := combined_length * s.width
  total_area * s.carpet_cost_per_sqm

/-- Theorem: The cost of carpeting the given staircase is 1512 yuan -/
theorem staircase_carpet_cost :
  let s : Staircase := {
    num_steps := 15,
    step_height := 0.16,
    step_depth := 0.26,
    width := 3,
    carpet_cost_per_sqm := 80
  }
  carpet_cost s = 1512 := by
  sorry

end NUMINAMATH_CALUDE_staircase_carpet_cost_l3606_360687


namespace NUMINAMATH_CALUDE_family_heights_l3606_360699

/-- Given the heights of a family, prove the calculated heights of specific members -/
theorem family_heights (cary bill jan tim sara : ℝ) : 
  cary = 72 →
  bill = 0.8 * cary →
  jan = bill + 5 →
  tim = (bill + jan) / 2 - 4 →
  sara = 1.2 * ((cary + bill + jan + tim) / 4) →
  (bill = 57.6 ∧ jan = 62.6 ∧ tim = 56.1 ∧ sara = 74.49) := by
  sorry

end NUMINAMATH_CALUDE_family_heights_l3606_360699


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3606_360611

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x)^(1/3 : ℝ) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3606_360611


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3606_360666

theorem negation_of_universal_proposition :
  (¬ ∀ x y : ℝ, x < 0 → y < 0 → x + y ≤ -2 * Real.sqrt (x * y)) ↔
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x + y > -2 * Real.sqrt (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3606_360666


namespace NUMINAMATH_CALUDE_tim_score_l3606_360608

/-- Represents the scores of players in a basketball game -/
structure BasketballScores where
  joe : ℕ
  tim : ℕ
  ken : ℕ

/-- Theorem: Tim's score is 30 points given the conditions of the basketball game -/
theorem tim_score (scores : BasketballScores) : scores.tim = 30 :=
  by
  have h1 : scores.tim = scores.joe + 20 := by sorry
  have h2 : scores.tim * 2 = scores.ken := by sorry
  have h3 : scores.joe + scores.tim + scores.ken = 100 := by sorry
  sorry

#check tim_score

end NUMINAMATH_CALUDE_tim_score_l3606_360608


namespace NUMINAMATH_CALUDE_c_is_positive_l3606_360665

theorem c_is_positive (a b c d e f : ℤ) 
  (h1 : a * b + c * d * e * f < 0)
  (h2 : a < 0)
  (h3 : b < 0)
  (h4 : d < 0)
  (h5 : e < 0)
  (h6 : f < 0) :
  c > 0 := by
sorry

end NUMINAMATH_CALUDE_c_is_positive_l3606_360665


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3606_360642

def is_odd (n : ℕ) : Bool := n % 2 = 1

def is_even (n : ℕ) : Bool := n % 2 = 0

def digit_count (n : ℕ) : ℕ := (String.length (toString n))

def sum_of_digits (n : ℕ) : ℕ :=
  (toString n).toList.map (fun c => c.toNat - '0'.toNat) |>.sum

def is_valid_number (n : ℕ) : Bool :=
  digit_count n = 4 ∧
  n % 9 = 0 ∧
  (is_odd (n / 1000 % 10) + is_odd (n / 100 % 10) + is_odd (n / 10 % 10) + is_odd (n % 10) = 3) ∧
  (is_even (n / 1000 % 10) + is_even (n / 100 % 10) + is_even (n / 10 % 10) + is_even (n % 10) = 1)

theorem smallest_valid_number : 
  (∀ m : ℕ, 1000 ≤ m ∧ m < 1215 → ¬ is_valid_number m) ∧ is_valid_number 1215 := by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3606_360642


namespace NUMINAMATH_CALUDE_find_A_l3606_360632

theorem find_A (A : ℕ) (h : A % 9 = 6 ∧ A / 9 = 2) : A = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l3606_360632


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l3606_360679

theorem cube_sum_reciprocal (r : ℝ) (h : (r + 1/r)^2 = 3) : r^3 + 1/r^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l3606_360679


namespace NUMINAMATH_CALUDE_jake_biking_speed_l3606_360669

/-- Represents the distance to the water park -/
def distance_to_waterpark : ℝ := 22

/-- Represents Jake's dad's driving time in hours -/
def dad_driving_time : ℝ := 0.5

/-- Represents Jake's dad's first half speed in miles per hour -/
def dad_speed1 : ℝ := 28

/-- Represents Jake's dad's second half speed in miles per hour -/
def dad_speed2 : ℝ := 60

/-- Represents Jake's biking time in hours -/
def jake_biking_time : ℝ := 2

/-- Theorem stating that Jake's biking speed is 11 miles per hour -/
theorem jake_biking_speed : 
  distance_to_waterpark / jake_biking_time = 11 := by
  sorry

/-- Lemma showing that the distance is correctly calculated -/
lemma distance_calculation : 
  distance_to_waterpark = 
    dad_speed1 * (dad_driving_time / 2) + 
    dad_speed2 * (dad_driving_time / 2) := by
  sorry

end NUMINAMATH_CALUDE_jake_biking_speed_l3606_360669


namespace NUMINAMATH_CALUDE_triangle_height_l3606_360678

theorem triangle_height (A B C : Real) (a b c : Real) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  1 + 2 * Real.cos (B + C) = 0 →
  ∃ h : Real, h = b * Real.sin C ∧ h = (Real.sqrt 3 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_l3606_360678


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3606_360649

theorem arithmetic_sequence_length (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 2 →
  aₙ = 3006 →
  d = 4 →
  aₙ = a₁ + (n - 1) * d →
  n = 752 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3606_360649


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3606_360684

theorem arithmetic_square_root_of_16 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3606_360684


namespace NUMINAMATH_CALUDE_gravel_cost_theorem_l3606_360654

/-- Calculates the cost of gravelling a path inside a rectangular plot -/
def gravel_cost (length width path_width gravel_cost_per_sqm : ℝ) : ℝ :=
  let total_area := length * width
  let inner_area := (length - 2 * path_width) * (width - 2 * path_width)
  let path_area := total_area - inner_area
  path_area * gravel_cost_per_sqm

/-- Theorem stating the cost of gravelling the path -/
theorem gravel_cost_theorem :
  gravel_cost 100 70 2.5 0.9 = 742.5 := by
sorry

end NUMINAMATH_CALUDE_gravel_cost_theorem_l3606_360654


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3606_360695

theorem min_value_of_expression (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 ≥ 2018 ∧
  ∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 = 2018 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3606_360695


namespace NUMINAMATH_CALUDE_total_tickets_sold_l3606_360637

/-- Represents the number of tickets sold for a theater performance --/
structure TheaterTickets where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total revenue from ticket sales --/
def totalRevenue (tickets : TheaterTickets) : ℕ :=
  12 * tickets.orchestra + 8 * tickets.balcony

/-- Theorem stating the total number of tickets sold given the conditions --/
theorem total_tickets_sold : 
  ∃ (tickets : TheaterTickets), 
    totalRevenue tickets = 3320 ∧ 
    tickets.balcony = tickets.orchestra + 140 ∧
    tickets.orchestra + tickets.balcony = 360 := by
  sorry

#check total_tickets_sold

end NUMINAMATH_CALUDE_total_tickets_sold_l3606_360637


namespace NUMINAMATH_CALUDE_square_side_length_l3606_360628

theorem square_side_length : ∃ (s : ℝ), s > 0 ∧ s^2 + s - 4*s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3606_360628


namespace NUMINAMATH_CALUDE_solve_pet_sitting_problem_l3606_360677

def pet_sitting_problem (hourly_rate : ℝ) (hours_this_week : ℝ) (total_earnings : ℝ) : Prop :=
  let earnings_this_week := hourly_rate * hours_this_week
  let earnings_last_week := total_earnings - earnings_this_week
  let hours_last_week := earnings_last_week / hourly_rate
  hourly_rate = 5 ∧ hours_this_week = 30 ∧ total_earnings = 250 → hours_last_week = 20

theorem solve_pet_sitting_problem :
  pet_sitting_problem 5 30 250 := by
  sorry

end NUMINAMATH_CALUDE_solve_pet_sitting_problem_l3606_360677


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l3606_360671

theorem part_to_whole_ratio (N A : ℕ) (h1 : N = 48) (h2 : A = 15) : 
  ∃ P : ℕ, P + A = 27 → P * 4 = N := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l3606_360671


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3606_360615

/-- Theorem: Rectangle Dimension Change
  Given a rectangle with length L and breadth B,
  if the length is increased by 10% and the area is increased by 37.5%,
  then the breadth must be increased by 25%.
-/
theorem rectangle_dimension_change
  (L B : ℝ)  -- Original length and breadth
  (L' B' : ℝ) -- New length and breadth
  (h1 : L' = 1.1 * L)  -- Length increased by 10%
  (h2 : L' * B' = 1.375 * (L * B))  -- Area increased by 37.5%
  : B' = 1.25 * B := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3606_360615


namespace NUMINAMATH_CALUDE_range_of_m_l3606_360694

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  monotonically_decreasing f →
  f (m + 1) < f (3 - 2 * m) →
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), m = -(Real.sin x)^2 - 2 * Real.sin x + 1) →
  m ∈ Set.Ioo (2/3) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3606_360694


namespace NUMINAMATH_CALUDE_factorization_equality_l3606_360605

theorem factorization_equality (x y : ℝ) : 1 - 2*(x - y) + (x - y)^2 = (1 - x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3606_360605


namespace NUMINAMATH_CALUDE_triangle_inequality_special_l3606_360602

theorem triangle_inequality_special (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x^2 + y^2 - x*y) + Real.sqrt (y^2 + z^2 - y*z) ≥ Real.sqrt (z^2 + x^2 - z*x) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_special_l3606_360602


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l3606_360623

theorem complex_purely_imaginary (a : ℝ) :
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l3606_360623


namespace NUMINAMATH_CALUDE_selection_test_results_l3606_360661

/-- Represents the probability of A answering a question correctly -/
def prob_A_correct : ℚ := 3/5

/-- Represents the number of questions B can answer correctly out of 10 -/
def B_correct_answers : ℕ := 5

/-- Represents the total number of questions in the pool -/
def total_questions : ℕ := 10

/-- Represents the number of questions in each exam -/
def exam_questions : ℕ := 3

/-- Represents the score for a correct answer -/
def correct_score : ℤ := 10

/-- Represents the score deduction for an incorrect answer -/
def incorrect_score : ℤ := -5

/-- Represents the minimum score required for selection -/
def selection_threshold : ℤ := 15

/-- The expected score for A -/
def expected_score_A : ℚ := 12

/-- The probability that both A and B are selected -/
def prob_both_selected : ℚ := 81/250

theorem selection_test_results :
  (prob_A_correct = 3/5) →
  (B_correct_answers = 5) →
  (total_questions = 10) →
  (exam_questions = 3) →
  (correct_score = 10) →
  (incorrect_score = -5) →
  (selection_threshold = 15) →
  (expected_score_A = 12) ∧
  (prob_both_selected = 81/250) := by
  sorry

end NUMINAMATH_CALUDE_selection_test_results_l3606_360661


namespace NUMINAMATH_CALUDE_congruence_problem_l3606_360662

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 200 ∧ (150 * n) % 199 = 110 % 199 → n % 199 = 157 % 199 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3606_360662


namespace NUMINAMATH_CALUDE_savings_percentage_l3606_360690

/-- Represents the financial situation of a person over two years --/
structure FinancialSituation where
  income_year1 : ℝ
  savings_year1 : ℝ
  income_year2 : ℝ
  savings_year2 : ℝ

/-- Conditions for the financial situation --/
def ValidFinancialSituation (fs : FinancialSituation) : Prop :=
  fs.income_year1 > 0 ∧
  fs.savings_year1 > 0 ∧
  fs.income_year2 = 1.35 * fs.income_year1 ∧
  fs.savings_year2 = 2 * fs.savings_year1 ∧
  (fs.income_year1 - fs.savings_year1) + (fs.income_year2 - fs.savings_year2) = 2 * (fs.income_year1 - fs.savings_year1)

/-- Theorem stating the percentage of income saved in the first year --/
theorem savings_percentage (fs : FinancialSituation) 
  (h : ValidFinancialSituation fs) : 
  fs.savings_year1 / fs.income_year1 = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l3606_360690


namespace NUMINAMATH_CALUDE_driving_equation_correct_l3606_360624

/-- Represents a driving scenario where the actual speed is faster than planned. -/
structure DrivingScenario where
  distance : ℝ
  planned_speed : ℝ
  actual_speed : ℝ
  time_saved : ℝ

/-- The equation correctly represents the driving scenario. -/
theorem driving_equation_correct (scenario : DrivingScenario) 
  (h1 : scenario.distance = 240)
  (h2 : scenario.actual_speed = 1.5 * scenario.planned_speed)
  (h3 : scenario.time_saved = 1)
  (h4 : scenario.planned_speed > 0) :
  scenario.distance / scenario.planned_speed - scenario.distance / scenario.actual_speed = scenario.time_saved := by
  sorry

#check driving_equation_correct

end NUMINAMATH_CALUDE_driving_equation_correct_l3606_360624


namespace NUMINAMATH_CALUDE_triangle_properties_l3606_360641

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem combining all parts of the problem --/
theorem triangle_properties (t : Triangle) (p : ℝ) :
  (Real.sqrt 3 * Real.sin t.B - Real.cos t.B) * (Real.sqrt 3 * Real.sin t.C - Real.cos t.C) = 4 * Real.cos t.B * Real.cos t.C →
  t.A = π / 3 ∧
  (t.a = 2 → 0 < (1/2 * t.b * t.c * Real.sin t.A) ∧ (1/2 * t.b * t.c * Real.sin t.A) ≤ Real.sqrt 3) ∧
  (Real.sin t.B = p * Real.sin t.C → 1/2 < p ∧ p < 2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_properties_l3606_360641


namespace NUMINAMATH_CALUDE_evaluate_expression_l3606_360670

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3606_360670


namespace NUMINAMATH_CALUDE_probability_of_dime_l3606_360696

/-- Represents the types of coins in the jar -/
inductive Coin
| Dime
| Nickel
| Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
| Coin.Dime => 10
| Coin.Nickel => 5
| Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def total_value : Coin → ℕ
| Coin.Dime => 500
| Coin.Nickel => 300
| Coin.Penny => 200

/-- The number of coins of each type in the jar -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Dime + coin_count Coin.Nickel + coin_count Coin.Penny

/-- The probability of randomly selecting a dime from the jar -/
theorem probability_of_dime : 
  (coin_count Coin.Dime : ℚ) / total_coins = 5 / 31 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_dime_l3606_360696


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3606_360652

/-- Parabola intersecting a line --/
structure ParabolaIntersection where
  p : ℝ
  chord_length : ℝ
  h_p_pos : p > 0
  h_chord : chord_length = 3 * Real.sqrt 5

/-- The result of the intersection --/
def ParabolaIntersectionResult (pi : ParabolaIntersection) : Prop :=
  -- Part I: The equation of the parabola is y² = 4x
  (pi.p = 2) ∧
  -- Part II: The maximum distance from a point on the circumcircle of triangle ABF to line AB
  (∃ (max_distance : ℝ), max_distance = (9 * Real.sqrt 5) / 2)

/-- Main theorem --/
theorem parabola_intersection_theorem (pi : ParabolaIntersection) :
  ParabolaIntersectionResult pi :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3606_360652


namespace NUMINAMATH_CALUDE_trisomy21_caused_by_sperm_l3606_360685

/-- Represents a genotype for the STR marker on chromosome 21 -/
inductive Genotype
  | Negative
  | Positive
  | DoublePositive

/-- Represents a person with their genotype -/
structure Person where
  genotype : Genotype

/-- Represents a family with a child, father, and mother -/
structure Family where
  child : Person
  father : Person
  mother : Person

/-- Defines Trisomy 21 syndrome -/
def hasTrisomy21 (p : Person) : Prop := p.genotype = Genotype.DoublePositive

/-- Defines the condition of sperm having 2 chromosome 21s -/
def spermHasTwoChromosome21 (f : Family) : Prop :=
  f.father.genotype = Genotype.Positive ∧
  f.mother.genotype = Genotype.Negative ∧
  f.child.genotype = Genotype.DoublePositive

/-- Theorem stating that given the family's genotypes, the child's Trisomy 21 is caused by sperm with 2 chromosome 21s -/
theorem trisomy21_caused_by_sperm (f : Family)
  (h_child : f.child.genotype = Genotype.DoublePositive)
  (h_father : f.father.genotype = Genotype.Positive)
  (h_mother : f.mother.genotype = Genotype.Negative) :
  hasTrisomy21 f.child ∧ spermHasTwoChromosome21 f := by
  sorry


end NUMINAMATH_CALUDE_trisomy21_caused_by_sperm_l3606_360685


namespace NUMINAMATH_CALUDE_winter_olympics_theorem_l3606_360693

/-- Represents the scoring system for the Winter Olympics knowledge competition. -/
structure ScoringSystem where
  num_questions : ℕ
  correct_points : ℕ
  incorrect_points : ℤ

/-- Calculates the total score given the number of correct and incorrect answers. -/
def calculate_score (system : ScoringSystem) (correct : ℕ) (incorrect : ℕ) : ℤ :=
  (correct : ℤ) * system.correct_points - incorrect * system.incorrect_points

/-- Calculates the minimum number of students required for at least 3 to have the same score. -/
def min_students_for_same_score (system : ScoringSystem) : ℕ :=
  (system.num_questions * system.correct_points + 1) * 2 + 1

/-- The Winter Olympics knowledge competition theorem. -/
theorem winter_olympics_theorem (system : ScoringSystem)
  (h_num_questions : system.num_questions = 10)
  (h_correct_points : system.correct_points = 5)
  (h_incorrect_points : system.incorrect_points = 1)
  (h_xiao_ming_correct : ℕ)
  (h_xiao_ming_incorrect : ℕ)
  (h_xiao_ming_total : h_xiao_ming_correct + h_xiao_ming_incorrect = system.num_questions)
  (h_xiao_ming_correct_8 : h_xiao_ming_correct = 8)
  (h_xiao_ming_incorrect_2 : h_xiao_ming_incorrect = 2) :
  (calculate_score system h_xiao_ming_correct h_xiao_ming_incorrect = 38) ∧
  (min_students_for_same_score system = 23) := by
  sorry

end NUMINAMATH_CALUDE_winter_olympics_theorem_l3606_360693


namespace NUMINAMATH_CALUDE_cone_height_from_circular_sector_l3606_360626

/-- The height of a cone formed by rolling one of four congruent sectors cut from a circular sheet of paper. -/
theorem cone_height_from_circular_sector (r : ℝ) (h : r = 10) :
  let sector_angle : ℝ := 2 * Real.pi / 4
  let base_radius : ℝ := r * sector_angle / (2 * Real.pi)
  let height : ℝ := Real.sqrt (r^2 - base_radius^2)
  height = (5 * Real.sqrt 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_from_circular_sector_l3606_360626


namespace NUMINAMATH_CALUDE_fraction_percent_of_x_l3606_360607

theorem fraction_percent_of_x (x : ℝ) (h : x > 0) : (x / 10 + x / 25) / x * 100 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_percent_of_x_l3606_360607


namespace NUMINAMATH_CALUDE_horner_method_correctness_horner_method_equivalence_l3606_360673

/-- Horner's Method evaluation for a specific polynomial -/
def horner_eval (x : ℝ) : ℝ := 
  (((((4 * x - 3) * x + 4) * x - 2) * x - 2) * x + 3)

/-- Count of multiplication operations in Horner's Method for this polynomial -/
def horner_mult_count : ℕ := 5

/-- Count of addition operations in Horner's Method for this polynomial -/
def horner_add_count : ℕ := 5

/-- Theorem stating the correctness of Horner's Method for the given polynomial -/
theorem horner_method_correctness : 
  horner_eval 3 = 816 ∧ 
  horner_mult_count = 5 ∧ 
  horner_add_count = 5 := by sorry

/-- Theorem stating that Horner's Method gives the same result as direct polynomial evaluation -/
theorem horner_method_equivalence (x : ℝ) : 
  horner_eval x = 4 * x^5 - 3 * x^4 + 4 * x^3 - 2 * x^2 - 2 * x + 3 := by sorry

end NUMINAMATH_CALUDE_horner_method_correctness_horner_method_equivalence_l3606_360673


namespace NUMINAMATH_CALUDE_abs_difference_of_product_and_sum_l3606_360644

theorem abs_difference_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 6) 
  (h2 : p + q = 7) : 
  |p - q| = Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_abs_difference_of_product_and_sum_l3606_360644


namespace NUMINAMATH_CALUDE_weighted_average_calculation_l3606_360612

/-- Calculates the weighted average of exam scores for a class -/
theorem weighted_average_calculation (total_students : ℕ) 
  (math_perfect_scores math_zero_scores : ℕ)
  (science_perfect_scores science_zero_scores : ℕ)
  (math_average_rest science_average_rest : ℚ)
  (math_weight science_weight : ℚ) :
  total_students = 30 →
  math_perfect_scores = 3 →
  math_zero_scores = 4 →
  math_average_rest = 50 →
  science_perfect_scores = 2 →
  science_zero_scores = 5 →
  science_average_rest = 60 →
  math_weight = 2/5 →
  science_weight = 3/5 →
  (((math_perfect_scores * 100 + 
    (total_students - math_perfect_scores - math_zero_scores) * math_average_rest) * math_weight +
   (science_perfect_scores * 100 + 
    (total_students - science_perfect_scores - science_zero_scores) * science_average_rest) * science_weight) / total_students) = 1528/30 :=
by sorry

end NUMINAMATH_CALUDE_weighted_average_calculation_l3606_360612
