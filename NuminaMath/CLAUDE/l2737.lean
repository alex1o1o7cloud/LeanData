import Mathlib

namespace NUMINAMATH_CALUDE_temperature_decrease_fraction_l2737_273732

theorem temperature_decrease_fraction (current_temp : ℝ) (decrease : ℝ) 
  (h1 : current_temp = 84)
  (h2 : decrease = 21) :
  (current_temp - decrease) / current_temp = 3/4 := by
sorry

end NUMINAMATH_CALUDE_temperature_decrease_fraction_l2737_273732


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_threes_l2737_273737

/-- Given a natural number n, this function returns the number composed of 2n digits of 1 -/
def two_n_ones (n : ℕ) : ℕ := (10^(2*n) - 1) / 9

/-- Given a natural number n, this function returns the number composed of n digits of 2 -/
def n_twos (n : ℕ) : ℕ := 2 * ((10^n - 1) / 9)

/-- Given a natural number n, this function returns the number composed of n digits of 3 -/
def n_threes (n : ℕ) : ℕ := (10^n - 1) / 3

theorem sqrt_difference_equals_threes (n : ℕ) : 
  Real.sqrt (two_n_ones n - n_twos n) = n_threes n := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_threes_l2737_273737


namespace NUMINAMATH_CALUDE_melanie_books_before_l2737_273747

/-- The number of books Melanie had before the yard sale -/
def books_before : ℕ := sorry

/-- The number of books Melanie bought at the yard sale -/
def books_bought : ℕ := 46

/-- The total number of books Melanie has after the yard sale -/
def books_after : ℕ := 87

/-- Theorem stating that Melanie had 41 books before the yard sale -/
theorem melanie_books_before : books_before = 41 := by sorry

end NUMINAMATH_CALUDE_melanie_books_before_l2737_273747


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l2737_273772

theorem quadratic_equation_solution_sum :
  ∀ c d : ℝ,
  (∃ x : ℝ, x^2 - 6*x + 11 = 23 ∧ (x = c ∨ x = d)) →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l2737_273772


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l2737_273717

theorem square_perimeter_relation (x y : Real) 
  (hx : x > 0) 
  (hy : y > 0) 
  (perimeter_x : 4 * x = 32) 
  (area_relation : y^2 = (1/3) * x^2) : 
  4 * y = (32 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l2737_273717


namespace NUMINAMATH_CALUDE_largest_interesting_number_l2737_273720

def is_interesting (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length ≥ 3 ∧
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

theorem largest_interesting_number :
  (∀ m : ℕ, is_interesting m → m ≤ 96433469) ∧ is_interesting 96433469 := by
  sorry

end NUMINAMATH_CALUDE_largest_interesting_number_l2737_273720


namespace NUMINAMATH_CALUDE_number_of_indoor_players_l2737_273768

/-- Given a group of players with outdoor, indoor, and both categories, 
    calculate the number of indoor players. -/
theorem number_of_indoor_players 
  (total : ℕ) 
  (outdoor : ℕ) 
  (both : ℕ) 
  (h1 : total = 400) 
  (h2 : outdoor = 350) 
  (h3 : both = 60) : 
  ∃ indoor : ℕ, indoor = 110 ∧ total = outdoor + indoor - both :=
sorry

end NUMINAMATH_CALUDE_number_of_indoor_players_l2737_273768


namespace NUMINAMATH_CALUDE_f_simplification_f_value_in_third_quadrant_l2737_273703

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by
  sorry

theorem f_value_in_third_quadrant (α : Real) 
  (h1 : α > Real.pi ∧ α < 3 * Real.pi / 2) 
  (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) : 
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_f_simplification_f_value_in_third_quadrant_l2737_273703


namespace NUMINAMATH_CALUDE_total_laces_is_6x_l2737_273787

/-- Given a number of shoe pairs, calculate the total number of laces needed -/
def total_laces (x : ℕ) : ℕ :=
  let lace_sets_per_pair := 2
  let color_options := 3
  x * lace_sets_per_pair * color_options

/-- Theorem stating that the total number of laces is 6x -/
theorem total_laces_is_6x (x : ℕ) : total_laces x = 6 * x := by
  sorry

end NUMINAMATH_CALUDE_total_laces_is_6x_l2737_273787


namespace NUMINAMATH_CALUDE_solve_for_a_l2737_273773

-- Define the operation *
def star_op (a b : ℚ) : ℚ := 2*a - b^2

-- Theorem statement
theorem solve_for_a :
  ∀ a : ℚ, star_op a 7 = -20 → a = 29/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2737_273773


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2737_273784

theorem quadratic_coefficient (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9) - 36 = 0) → b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2737_273784


namespace NUMINAMATH_CALUDE_max_value_of_f_l2737_273741

noncomputable def f (x : ℝ) : ℝ := x * Real.sqrt (18 - x) + Real.sqrt (18 * x - x ^ 3)

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 18 ∧
  f x = 2 * Real.sqrt 17 ∧
  ∀ y ∈ Set.Icc 0 18, f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2737_273741


namespace NUMINAMATH_CALUDE_catch_up_theorem_l2737_273744

/-- The distance traveled by both tourists when the second catches up to the first -/
def catch_up_distance : ℝ := 56

/-- The speed of the first tourist on bicycle in km/h -/
def speed_bicycle : ℝ := 16

/-- The speed of the second tourist on motorcycle in km/h -/
def speed_motorcycle : ℝ := 56

/-- The initial travel time of the first tourist before the break in hours -/
def initial_travel_time : ℝ := 1.5

/-- The break time of the first tourist in hours -/
def break_time : ℝ := 1.5

/-- The time delay between the start of the first and second tourist in hours -/
def start_delay : ℝ := 4

theorem catch_up_theorem :
  ∃ t : ℝ, t > 0 ∧
  speed_bicycle * (initial_travel_time + t) = 
  speed_motorcycle * t ∧
  catch_up_distance = speed_motorcycle * t :=
sorry

end NUMINAMATH_CALUDE_catch_up_theorem_l2737_273744


namespace NUMINAMATH_CALUDE_expression_evaluation_l2737_273796

theorem expression_evaluation :
  let a : ℝ := 1
  let b : ℝ := -1
  3 * a^2 * b - 2 * (a * b - 3/2 * a^2 * b) + a * b - 2 * a^2 * b = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2737_273796


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l2737_273754

theorem five_digit_multiple_of_nine : ∃ (d : ℕ), d < 10 ∧ 56170 + d ≡ 0 [MOD 9] := by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l2737_273754


namespace NUMINAMATH_CALUDE_root_product_sum_l2737_273735

theorem root_product_sum (a b c : ℂ) : 
  (5 * a^3 - 4 * a^2 + 15 * a - 12 = 0) →
  (5 * b^3 - 4 * b^2 + 15 * b - 12 = 0) →
  (5 * c^3 - 4 * c^2 + 15 * c - 12 = 0) →
  a * b + a * c + b * c = -3 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l2737_273735


namespace NUMINAMATH_CALUDE_existence_of_sequence_l2737_273709

theorem existence_of_sequence (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ) 
  (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  ∃ a : Fin (n + 1) → ℝ,
    (a 0 + a (Fin.last n) = 0) ∧
    (∀ i, |a i| ≤ 1) ∧
    (∀ i : Fin n, |a i.succ - a i| = x i) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l2737_273709


namespace NUMINAMATH_CALUDE_optimal_price_and_profit_l2737_273793

/-- Represents the daily sales volume as a function of the selling price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 740

/-- Represents the daily profit as a function of the selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The cost price of each book -/
def cost_price : ℝ := 40

/-- The minimum selling price -/
def min_price : ℝ := 44

/-- The maximum selling price based on the profit margin constraint -/
def max_price : ℝ := 52

theorem optimal_price_and_profit :
  ∀ x : ℝ, min_price ≤ x ∧ x ≤ max_price →
  daily_profit x ≤ daily_profit max_price ∧
  daily_profit max_price = 2640 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_and_profit_l2737_273793


namespace NUMINAMATH_CALUDE_coffee_cost_l2737_273759

/-- The cost of each coffee Jon buys, given his spending habits in April. -/
theorem coffee_cost (coffees_per_day : ℕ) (total_spent : ℕ) (days_in_april : ℕ) :
  coffees_per_day = 2 →
  total_spent = 120 →
  days_in_april = 30 →
  total_spent / (coffees_per_day * days_in_april) = 2 :=
by sorry

end NUMINAMATH_CALUDE_coffee_cost_l2737_273759


namespace NUMINAMATH_CALUDE_subtract_linear_equations_l2737_273762

theorem subtract_linear_equations :
  let eq1 : ℝ → ℝ → ℝ := λ x y => 2 * x + 3 * y
  let eq2 : ℝ → ℝ → ℝ := λ x y => 5 * x + 3 * y
  let result : ℝ → ℝ := λ x => -3 * x
  (∀ x y, eq1 x y = 11) →
  (∀ x y, eq2 x y = -7) →
  (∀ x, result x = 18) →
  ∀ x y, eq1 x y - eq2 x y = result x :=
by
  sorry

end NUMINAMATH_CALUDE_subtract_linear_equations_l2737_273762


namespace NUMINAMATH_CALUDE_nancy_homework_pages_l2737_273771

def homework_pages (total_problems : ℕ) (finished_problems : ℕ) (problems_per_page : ℕ) : ℕ :=
  (total_problems - finished_problems) / problems_per_page

theorem nancy_homework_pages : 
  homework_pages 101 47 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_nancy_homework_pages_l2737_273771


namespace NUMINAMATH_CALUDE_impossible_table_l2737_273745

/-- Represents a 6x6 table of integers -/
def Table := Fin 6 → Fin 6 → ℤ

/-- Checks if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  ∀ i j k l, (i ≠ k ∨ j ≠ l) → t i j ≠ t k l

/-- Checks if the sum of a 1x5 rectangle is valid (2022 or 2023) -/
def valid_sum (s : ℤ) : Prop := s = 2022 ∨ s = 2023

/-- Checks if all 1x5 rectangles (horizontal and vertical) have valid sums -/
def all_rectangles_valid (t : Table) : Prop :=
  (∀ i j, valid_sum (t i j + t i (j+1) + t i (j+2) + t i (j+3) + t i (j+4))) ∧
  (∀ i j, valid_sum (t i j + t (i+1) j + t (i+2) j + t (i+3) j + t (i+4) j))

/-- The main theorem: it's impossible to fill the table satisfying all conditions -/
theorem impossible_table : ¬∃ (t : Table), all_distinct t ∧ all_rectangles_valid t := by
  sorry

end NUMINAMATH_CALUDE_impossible_table_l2737_273745


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2737_273755

-- Define the repeating decimals
def repeating_six : ℚ := 2/3
def repeating_seven : ℚ := 7/9

-- State the theorem
theorem sum_of_repeating_decimals : 
  repeating_six + repeating_seven = 13/9 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2737_273755


namespace NUMINAMATH_CALUDE_ap_has_twelve_terms_l2737_273731

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  n : ℕ
  a : ℝ
  d : ℝ
  odd_sum : ℝ
  even_sum : ℝ
  last_term : ℝ
  third_term : ℝ

/-- The conditions of the arithmetic progression -/
def APConditions (ap : ArithmeticProgression) : Prop :=
  Even ap.n ∧
  ap.odd_sum = 36 ∧
  ap.even_sum = 42 ∧
  ap.last_term = ap.a + 12 ∧
  ap.third_term = 6 ∧
  ap.third_term = ap.a + 2 * ap.d ∧
  ap.odd_sum = (ap.n / 2 : ℝ) * (ap.a + (ap.a + (ap.n - 2) * ap.d)) ∧
  ap.even_sum = (ap.n / 2 : ℝ) * ((ap.a + ap.d) + (ap.a + (ap.n - 1) * ap.d))

/-- The theorem to be proved -/
theorem ap_has_twelve_terms (ap : ArithmeticProgression) :
  APConditions ap → ap.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ap_has_twelve_terms_l2737_273731


namespace NUMINAMATH_CALUDE_event_selection_methods_l2737_273776

def total_students : ℕ := 5
def selected_students : ℕ := 4
def num_days : ℕ := 3
def friday_attendees : ℕ := 2
def saturday_attendees : ℕ := 1
def sunday_attendees : ℕ := 1

theorem event_selection_methods :
  (Nat.choose total_students friday_attendees) *
  (Nat.choose (total_students - friday_attendees) saturday_attendees) *
  (Nat.choose (total_students - friday_attendees - saturday_attendees) sunday_attendees) = 60 := by
  sorry

end NUMINAMATH_CALUDE_event_selection_methods_l2737_273776


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2737_273756

theorem inequality_equivalence (x y : ℝ) :
  (2 * y + 3 * x > Real.sqrt (9 * x^2)) ↔
  ((x ≥ 0 ∧ y > 0) ∨ (x < 0 ∧ y > -3 * x)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2737_273756


namespace NUMINAMATH_CALUDE_sock_drawer_probability_l2737_273761

/-- The total number of socks in the drawer -/
def total_socks : ℕ := 2016

/-- The number of copper socks -/
def copper_socks : ℕ := 2000

/-- The number of colors other than copper -/
def other_colors : ℕ := 8

/-- The number of socks for each color other than copper -/
def socks_per_color : ℕ := 2

/-- The probability of drawing two socks of the same color or one red and one green sock -/
def probability : ℚ := 1999012 / 2031120

theorem sock_drawer_probability :
  (copper_socks.choose 2 + other_colors * socks_per_color.choose 2 + socks_per_color ^ 2) /
  total_socks.choose 2 = probability := by sorry

end NUMINAMATH_CALUDE_sock_drawer_probability_l2737_273761


namespace NUMINAMATH_CALUDE_tangent_perpendicular_line_l2737_273783

theorem tangent_perpendicular_line (x₀ y₀ c : ℝ) : 
  y₀ = Real.exp x₀ →                     -- P is on the curve y = e^x
  x₀ + 2 * y₀ + c = 0 →                  -- Line passes through P
  2 * Real.exp x₀ = -1 →                 -- Line is perpendicular to tangent
  c = -4 - Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_line_l2737_273783


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_simplest_form_l2737_273766

theorem subtraction_of_fractions : 
  (9 : ℚ) / 23 - (5 : ℚ) / 69 = (22 : ℚ) / 69 := by
  sorry

theorem simplest_form : 
  ∀ (a b : ℤ), a ≠ 0 → b > 0 → (22 : ℚ) / 69 = (a : ℚ) / b → a = 22 ∧ b = 69 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_simplest_form_l2737_273766


namespace NUMINAMATH_CALUDE_dispatch_plans_count_l2737_273757

-- Define the total number of students
def total_students : ℕ := 6

-- Define the number of students needed for each day
def sunday_students : ℕ := 2
def friday_students : ℕ := 1
def saturday_students : ℕ := 1

-- Define the total number of students needed
def total_needed : ℕ := sunday_students + friday_students + saturday_students

-- Theorem statement
theorem dispatch_plans_count : 
  (Nat.choose total_students sunday_students) * 
  (Nat.choose (total_students - sunday_students) friday_students) * 
  (Nat.choose (total_students - sunday_students - friday_students) saturday_students) = 180 :=
by sorry

end NUMINAMATH_CALUDE_dispatch_plans_count_l2737_273757


namespace NUMINAMATH_CALUDE_cohen_bird_count_l2737_273774

/-- The total number of fish-eater birds Cohen saw over three days -/
def total_birds (day1 : ℕ) (day2_factor : ℕ) (day3_reduction : ℕ) : ℕ :=
  day1 + day1 * day2_factor + (day1 * day2_factor - day3_reduction)

/-- Theorem stating the total number of fish-eater birds Cohen saw over three days -/
theorem cohen_bird_count :
  total_birds 300 2 200 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_cohen_bird_count_l2737_273774


namespace NUMINAMATH_CALUDE_number_fraction_theorem_l2737_273710

theorem number_fraction_theorem (number : ℚ) (fraction : ℚ) : 
  number = 64 →
  number = number * fraction + 40 →
  fraction = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_theorem_l2737_273710


namespace NUMINAMATH_CALUDE_distance_P_to_y_axis_l2737_273712

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate. -/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- The point P has coordinates (3, -5). -/
def P : ℝ × ℝ := (3, -5)

/-- Theorem: The distance from point P(3, -5) to the y-axis is 3. -/
theorem distance_P_to_y_axis : distance_to_y_axis P.1 P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_y_axis_l2737_273712


namespace NUMINAMATH_CALUDE_torn_pages_theorem_l2737_273715

/-- Represents a set of consecutive pages torn from a book --/
structure TornPages where
  first : ℕ  -- First page number
  count : ℕ  -- Number of pages torn out

/-- The sum of consecutive integers from n to n + k - 1 --/
def sum_consecutive (n : ℕ) (k : ℕ) : ℕ :=
  k * (2 * n + k - 1) / 2

theorem torn_pages_theorem (pages : TornPages) :
  sum_consecutive pages.first pages.count = 344 →
  (344 = 2^3 * 43 ∧
   pages.first + (pages.first + pages.count - 1) = 43 ∧
   pages.count = 16) := by
  sorry


end NUMINAMATH_CALUDE_torn_pages_theorem_l2737_273715


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2737_273752

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2737_273752


namespace NUMINAMATH_CALUDE_digit_count_of_2_15_3_2_5_12_l2737_273746

theorem digit_count_of_2_15_3_2_5_12 : 
  (Nat.digits 10 (2^15 * 3^2 * 5^12)).length = 14 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_of_2_15_3_2_5_12_l2737_273746


namespace NUMINAMATH_CALUDE_percentage_of_french_speakers_l2737_273729

theorem percentage_of_french_speakers (total_employees : ℝ) (h1 : total_employees > 0) :
  let men_percentage : ℝ := 70
  let women_percentage : ℝ := 100 - men_percentage
  let men_french_speakers_percentage : ℝ := 50
  let women_non_french_speakers_percentage : ℝ := 83.33333333333331
  let men : ℝ := (men_percentage / 100) * total_employees
  let women : ℝ := (women_percentage / 100) * total_employees
  let men_french_speakers : ℝ := (men_french_speakers_percentage / 100) * men
  let women_french_speakers : ℝ := (1 - women_non_french_speakers_percentage / 100) * women
  let total_french_speakers : ℝ := men_french_speakers + women_french_speakers
  (total_french_speakers / total_employees) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_french_speakers_l2737_273729


namespace NUMINAMATH_CALUDE_max_guaranteed_winning_score_l2737_273722

/-- Represents a 9x9 grid game board -/
def GameBoard := Fin 9 → Fin 9 → Bool

/-- Counts the number of rows and columns where crosses outnumber noughts -/
def countCrossDominance (board : GameBoard) : ℕ :=
  sorry

/-- Counts the number of rows and columns where noughts outnumber crosses -/
def countNoughtDominance (board : GameBoard) : ℕ :=
  sorry

/-- Calculates the winning score for the first player -/
def winningScore (board : GameBoard) : ℤ :=
  (countCrossDominance board : ℤ) - (countNoughtDominance board : ℤ)

/-- Represents a strategy for playing the game -/
def Strategy := GameBoard → Fin 9 × Fin 9

/-- The theorem stating that the maximum guaranteed winning score is 2 -/
theorem max_guaranteed_winning_score :
  ∃ (strategyFirst : Strategy),
    ∀ (strategySecond : Strategy),
      ∃ (finalBoard : GameBoard),
        (winningScore finalBoard ≥ 2) ∧
        ∀ (otherFinalBoard : GameBoard),
          winningScore otherFinalBoard ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_guaranteed_winning_score_l2737_273722


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l2737_273764

theorem quadratic_root_ratio (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ 0 ∧ y = 2 * x ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  2 * b^2 = 9 * a * c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l2737_273764


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l2737_273777

theorem ratio_sum_problem (x y z a : ℚ) : 
  (∃ (k : ℚ), x = 3 * k ∧ y = 4 * k ∧ z = 6 * k) →
  y = 15 * a + 5 →
  x + y + z = 52 →
  a = 11 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l2737_273777


namespace NUMINAMATH_CALUDE_prop_2_prop_4_l2737_273742

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define evenness for a function
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define symmetry about a vertical line
def SymmetricAbout (g : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, g (a + x) = g (a - x)

-- Proposition ②
theorem prop_2 (h : IsEven (fun x ↦ f (x + 2))) : SymmetricAbout f 2 := by sorry

-- Proposition ④
theorem prop_4 : SymmetricAbout (fun x ↦ f (x - 2)) 2 ∧ SymmetricAbout (fun x ↦ f (2 - x)) 2 := by sorry

end NUMINAMATH_CALUDE_prop_2_prop_4_l2737_273742


namespace NUMINAMATH_CALUDE_prob_at_least_one_success_l2737_273728

/-- The probability of success for a single attempt -/
def p : ℝ := 0.5

/-- The number of attempts for each athlete -/
def attempts_per_athlete : ℕ := 2

/-- The total number of attempts -/
def total_attempts : ℕ := 2 * attempts_per_athlete

/-- The probability of at least one successful attempt out of the total attempts -/
theorem prob_at_least_one_success :
  1 - (1 - p) ^ total_attempts = 0.9375 := by
  sorry


end NUMINAMATH_CALUDE_prob_at_least_one_success_l2737_273728


namespace NUMINAMATH_CALUDE_special_polynomial_p_count_l2737_273725

/-- Represents a polynomial of degree 4 with specific properties -/
structure SpecialPolynomial where
  m : ℤ
  n : ℤ
  p : ℤ
  zeros : Fin 4 → ℝ
  is_zero : ∀ i, (zeros i)^4 - 2004 * (zeros i)^3 + m * (zeros i)^2 + n * (zeros i) + p = 0
  distinct_zeros : ∀ i j, i ≠ j → zeros i ≠ zeros j
  positive_zeros : ∀ i, zeros i > 0
  integer_zero : ∃ i, ∃ k : ℤ, zeros i = k
  sum_property : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ zeros i = zeros j + zeros k
  product_property : ∃ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧ zeros i = zeros j * zeros k * zeros l

/-- The number of possible values for p in a SpecialPolynomial -/
def count_p_values : ℕ := 63000

/-- Theorem stating that there are exactly 63000 possible values for p -/
theorem special_polynomial_p_count :
  (∃ f : Set SpecialPolynomial → ℕ, f {sp | sp.p = p} = count_p_values) :=
sorry

end NUMINAMATH_CALUDE_special_polynomial_p_count_l2737_273725


namespace NUMINAMATH_CALUDE_sequence_product_l2737_273785

/-- An arithmetic sequence where no term is zero -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m ∧ a n ≠ 0

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem sequence_product (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  a 4 - 2 * (a 7)^2 + 3 * a 8 = 0 →
  b 7 = a 7 →
  b 3 * b 7 * b 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l2737_273785


namespace NUMINAMATH_CALUDE_clerk_forms_per_hour_l2737_273743

theorem clerk_forms_per_hour 
  (total_forms : ℕ) 
  (work_hours : ℕ) 
  (num_clerks : ℕ) 
  (h1 : total_forms = 2400) 
  (h2 : work_hours = 8) 
  (h3 : num_clerks = 12) : 
  (total_forms / work_hours) / num_clerks = 25 := by
sorry

end NUMINAMATH_CALUDE_clerk_forms_per_hour_l2737_273743


namespace NUMINAMATH_CALUDE_inequality_proof_l2737_273740

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) :
  a^2 / b + b^2 / c + c^2 / d + d^2 / a ≥ 4 + (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2737_273740


namespace NUMINAMATH_CALUDE_smallest_inverse_mod_735_l2737_273750

theorem smallest_inverse_mod_735 : 
  ∀ n : ℕ, n > 2 → (∃ m : ℕ, n * m ≡ 1 [MOD 735]) → n ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_inverse_mod_735_l2737_273750


namespace NUMINAMATH_CALUDE_product_of_1001_2_and_121_3_l2737_273708

/-- Converts a number from base 2 to base 10 -/
def base2To10 (n : List Bool) : Nat :=
  n.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a number from base 3 to base 10 -/
def base3To10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The problem statement -/
theorem product_of_1001_2_and_121_3 :
  let n1 := base2To10 [true, false, false, true]
  let n2 := base3To10 [1, 2, 1]
  n1 * n2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_product_of_1001_2_and_121_3_l2737_273708


namespace NUMINAMATH_CALUDE_football_players_count_l2737_273733

theorem football_players_count (total_players cricket_players hockey_players softball_players : ℕ) :
  total_players = 55 →
  cricket_players = 15 →
  hockey_players = 12 →
  softball_players = 15 →
  total_players = cricket_players + hockey_players + softball_players + 13 :=
by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l2737_273733


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l2737_273780

theorem least_number_with_remainder (n : ℕ) : n = 125 →
  (∃ k : ℕ, n = 20 * k + 5) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m = 20 * k + 5)) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l2737_273780


namespace NUMINAMATH_CALUDE_distance_between_points_l2737_273799

theorem distance_between_points (A B : ℝ) : 
  (|A| = 2 ∧ |B| = 7) → (|A - B| = 5 ∨ |A - B| = 9) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2737_273799


namespace NUMINAMATH_CALUDE_red_balloon_probability_l2737_273724

/-- Calculates the probability of selecting a red balloon given the initial and additional counts of red and blue balloons. -/
theorem red_balloon_probability
  (initial_red : ℕ)
  (initial_blue : ℕ)
  (additional_red : ℕ)
  (additional_blue : ℕ)
  (h1 : initial_red = 2)
  (h2 : initial_blue = 4)
  (h3 : additional_red = 2)
  (h4 : additional_blue = 2) :
  (initial_red + additional_red : ℚ) / ((initial_red + additional_red + initial_blue + additional_blue) : ℚ) = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_red_balloon_probability_l2737_273724


namespace NUMINAMATH_CALUDE_power_five_sum_greater_than_mixed_products_l2737_273763

theorem power_five_sum_greater_than_mixed_products {a b : ℝ} 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  a^5 + b^5 > a^3 * b^2 + a^2 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_five_sum_greater_than_mixed_products_l2737_273763


namespace NUMINAMATH_CALUDE_eight_chairs_bought_l2737_273753

/-- Represents the chair purchase scenario at Big Lots --/
structure ChairPurchase where
  normalPrice : ℝ
  initialDiscount : ℝ
  additionalDiscount : ℝ
  totalCost : ℝ
  minChairsForAdditionalDiscount : ℕ

/-- Calculates the number of chairs bought given the purchase conditions --/
def calculateChairsBought (purchase : ChairPurchase) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, 8 chairs were bought --/
theorem eight_chairs_bought : 
  let purchase : ChairPurchase := {
    normalPrice := 20,
    initialDiscount := 0.25,
    additionalDiscount := 1/3,
    totalCost := 105,
    minChairsForAdditionalDiscount := 5
  }
  calculateChairsBought purchase = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_chairs_bought_l2737_273753


namespace NUMINAMATH_CALUDE_no_all_ones_sum_l2737_273765

def has_no_zero_digit (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

def is_rearrangement (n m : ℕ) : Prop :=
  n.digits 10 ≠ [] ∧ Multiset.ofList (n.digits 10) = Multiset.ofList (m.digits 10)

def all_ones (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 1

theorem no_all_ones_sum (N : ℕ) (hN : has_no_zero_digit N) :
  ∀ M : ℕ, is_rearrangement N M → ¬ all_ones (N + M) :=
sorry

end NUMINAMATH_CALUDE_no_all_ones_sum_l2737_273765


namespace NUMINAMATH_CALUDE_harold_wrapping_cost_l2737_273736

/-- Represents the number of shirt boxes that can be wrapped with one roll of paper -/
def shirt_boxes_per_roll : ℕ := 5

/-- Represents the number of XL boxes that can be wrapped with one roll of paper -/
def xl_boxes_per_roll : ℕ := 3

/-- Represents the number of shirt boxes Harold needs to wrap -/
def harold_shirt_boxes : ℕ := 20

/-- Represents the number of XL boxes Harold needs to wrap -/
def harold_xl_boxes : ℕ := 12

/-- Represents the cost of one roll of wrapping paper in cents -/
def cost_per_roll : ℕ := 400

/-- Theorem stating that Harold will spend $32.00 to wrap all boxes -/
theorem harold_wrapping_cost : 
  (((harold_shirt_boxes + shirt_boxes_per_roll - 1) / shirt_boxes_per_roll) + 
   ((harold_xl_boxes + xl_boxes_per_roll - 1) / xl_boxes_per_roll)) * 
  cost_per_roll = 3200 := by
  sorry

end NUMINAMATH_CALUDE_harold_wrapping_cost_l2737_273736


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l2737_273792

/-- Represents the different employee categories in the company -/
inductive EmployeeCategory
  | Senior
  | Intermediate
  | General

/-- Represents the company's employee distribution -/
structure CompanyDistribution where
  total : Nat
  senior : Nat
  intermediate : Nat
  general : Nat
  senior_count : senior ≤ total
  intermediate_count : intermediate ≤ total
  general_count : general ≤ total
  total_sum : senior + intermediate + general = total

/-- Represents a sampling method -/
inductive SamplingMethod
  | Simple
  | Stratified
  | Cluster
  | Systematic

/-- Determines the most appropriate sampling method given a company distribution and sample size -/
def mostAppropriateSamplingMethod (dist : CompanyDistribution) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is the most appropriate method for the given scenario -/
theorem stratified_sampling_most_appropriate (dist : CompanyDistribution) (sampleSize : Nat) :
  dist.total = 150 ∧ dist.senior = 15 ∧ dist.intermediate = 45 ∧ dist.general = 90 ∧ sampleSize = 30 →
  mostAppropriateSamplingMethod dist sampleSize = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l2737_273792


namespace NUMINAMATH_CALUDE_equal_distribution_possible_l2737_273770

/-- Represents the setup of the ball distribution problem -/
structure BallDistribution where
  num_colors : ℕ
  num_students : ℕ
  total_balls : ℕ
  min_balls_per_color : ℕ
  min_balls_per_box : ℕ

/-- Checks if the ball distribution is valid -/
def is_valid_distribution (d : BallDistribution) : Prop :=
  d.num_colors = 20 ∧
  d.num_students = 20 ∧
  d.total_balls = 800 ∧
  d.min_balls_per_color ≥ 10 ∧
  d.min_balls_per_box ≥ 10

/-- Theorem stating that a valid distribution allows equal distribution of balls among students -/
theorem equal_distribution_possible (d : BallDistribution) 
  (h : is_valid_distribution d) : 
  ∃ (balls_per_student : ℕ), 
    balls_per_student * d.num_students = d.total_balls ∧ 
    ∃ (num_boxes : ℕ), 
      num_boxes % d.num_students = 0 ∧
      num_boxes * d.min_balls_per_box ≥ d.total_balls := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_possible_l2737_273770


namespace NUMINAMATH_CALUDE_trains_meeting_point_l2737_273706

/-- The speed of the Bombay Express in km/h -/
def bombay_speed : ℝ := 60

/-- The speed of the Rajdhani Express in km/h -/
def rajdhani_speed : ℝ := 80

/-- The time difference between the departures of the two trains in hours -/
def time_difference : ℝ := 2

/-- The meeting point of the two trains -/
def meeting_point : ℝ := 480

theorem trains_meeting_point :
  ∃ t : ℝ, t > 0 ∧ bombay_speed * (t + time_difference) = rajdhani_speed * t ∧
  rajdhani_speed * t = meeting_point := by sorry

end NUMINAMATH_CALUDE_trains_meeting_point_l2737_273706


namespace NUMINAMATH_CALUDE_unique_perfect_square_sum_diff_l2737_273779

theorem unique_perfect_square_sum_diff (a : ℕ) : 
  (∃ b : ℕ, a * a = (b + 1) * (b + 1) - b * b ∧ 
            a * a = b * b + (b + 1) * (b + 1)) ∧ 
  a * a < 20000 ↔ 
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_sum_diff_l2737_273779


namespace NUMINAMATH_CALUDE_transformed_area_l2737_273797

-- Define the matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 2; 4, -5]

-- Define the area of the original region R
def area_R : ℝ := 15

-- Theorem statement
theorem transformed_area :
  let det_A := Matrix.det A
  area_R * |det_A| = 345 := by
sorry

end NUMINAMATH_CALUDE_transformed_area_l2737_273797


namespace NUMINAMATH_CALUDE_sin_greater_cos_range_l2737_273788

theorem sin_greater_cos_range (x : ℝ) : 
  x ∈ Set.Ioo (0 : ℝ) (2 * Real.pi) → 
  (Real.sin x > Real.cos x ↔ x ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_sin_greater_cos_range_l2737_273788


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l2737_273749

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_specific_proposition :
  (¬ ∃ x : ℝ, x^2 - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l2737_273749


namespace NUMINAMATH_CALUDE_min_lifts_for_equal_weight_l2737_273713

/-- The minimum number of lifts required to match or exceed the initial total weight -/
def min_lifts (initial_weight : ℕ) (initial_reps : ℕ) (new_weight : ℕ) (new_count : ℕ) : ℕ :=
  ((initial_weight * initial_reps + new_weight - 1) / new_weight : ℕ)

theorem min_lifts_for_equal_weight :
  min_lifts 75 10 80 4 = 10 := by sorry

end NUMINAMATH_CALUDE_min_lifts_for_equal_weight_l2737_273713


namespace NUMINAMATH_CALUDE_max_value_constraint_l2737_273790

/-- Given a point (3,1) lying on the line mx + ny + 1 = 0 where mn > 0, 
    the maximum value of 3/m + 1/n is -16. -/
theorem max_value_constraint (m n : ℝ) : 
  m * n > 0 → 
  3 * m + n = -1 → 
  (3 / m + 1 / n) ≤ -16 ∧ 
  ∃ m₀ n₀ : ℝ, m₀ * n₀ > 0 ∧ 3 * m₀ + n₀ = -1 ∧ 3 / m₀ + 1 / n₀ = -16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2737_273790


namespace NUMINAMATH_CALUDE_sin_cos_difference_77_47_l2737_273794

theorem sin_cos_difference_77_47 :
  Real.sin (77 * π / 180) * Real.cos (47 * π / 180) -
  Real.cos (77 * π / 180) * Real.sin (47 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_77_47_l2737_273794


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l2737_273798

/-- Simple interest calculation -/
theorem simple_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : principal = 10000)
  (h2 : rate = 0.04)
  (h3 : time = 1) :
  principal * rate * time = 400 := by
  sorry

#check simple_interest_calculation

end NUMINAMATH_CALUDE_simple_interest_calculation_l2737_273798


namespace NUMINAMATH_CALUDE_expression_evaluation_l2737_273789

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 5 + 1/2
  ((x^2 / (x - 1) - x + 1) / ((4*x^2 - 4*x + 1) / (1 - x))) = -Real.sqrt 5 / 10 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2737_273789


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l2737_273714

/-- Given a quadratic function f(x) = 3x^2 + 2x + 4, when shifted 3 units to the left,
    it becomes g(x) = a*x^2 + b*x + c. This theorem proves that a + b + c = 60. -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, 3*(x+3)^2 + 2*(x+3) + 4 = a*x^2 + b*x + c) → 
  a + b + c = 60 := by
sorry


end NUMINAMATH_CALUDE_quadratic_shift_sum_l2737_273714


namespace NUMINAMATH_CALUDE_parabola_focus_l2737_273767

/-- The focus of the parabola y^2 = 8x is at the point (2, 0) -/
theorem parabola_focus (x y : ℝ) : 
  (∀ x y, y^2 = 8*x ↔ (x - 2)^2 + y^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l2737_273767


namespace NUMINAMATH_CALUDE_extremum_condition_l2737_273791

open Real

-- Define a differentiable function
variable (f : ℝ → ℝ) (hf : Differentiable ℝ f)

-- Define the concept of an extremum
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

-- Theorem statement
theorem extremum_condition (x₀ : ℝ) :
  (HasExtremumAt f x₀ → deriv f x₀ = 0) ∧
  ∃ g : ℝ → ℝ, Differentiable ℝ g ∧ deriv g 0 = 0 ∧ ¬HasExtremumAt g 0 := by
  sorry

end NUMINAMATH_CALUDE_extremum_condition_l2737_273791


namespace NUMINAMATH_CALUDE_peter_pizza_fraction_l2737_273758

theorem peter_pizza_fraction :
  ∀ (total_slices : ℕ) (peter_own_slices : ℕ) (shared_slices : ℕ),
  total_slices = 16 →
  peter_own_slices = 2 →
  shared_slices = 2 →
  (peter_own_slices : ℚ) / total_slices + (shared_slices / 2 : ℚ) / total_slices = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_peter_pizza_fraction_l2737_273758


namespace NUMINAMATH_CALUDE_james_muffins_count_l2737_273795

/-- The number of muffins Arthur baked -/
def arthur_muffins : ℕ := 115

/-- The factor by which James baked more muffins than Arthur -/
def james_factor : ℕ := 12

/-- The number of muffins James baked -/
def james_muffins : ℕ := arthur_muffins * james_factor

theorem james_muffins_count : james_muffins = 1380 := by
  sorry

end NUMINAMATH_CALUDE_james_muffins_count_l2737_273795


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l2737_273738

open InnerProductSpace

theorem max_sum_squared_distances {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b c d : V) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2 ≤ 16 ∧
  ∃ (a' b' c' d' : V), ‖a'‖ = 1 ∧ ‖b'‖ = 1 ∧ ‖c'‖ = 1 ∧ ‖d'‖ = 1 ∧
    ‖a' - b'‖^2 + ‖a' - c'‖^2 + ‖a' - d'‖^2 + ‖b' - c'‖^2 + ‖b' - d'‖^2 + ‖c' - d'‖^2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l2737_273738


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1001st_term_l2737_273719

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (p q : ℝ) : ℕ → ℝ
  | 0 => p
  | 1 => 9
  | 2 => 3*p - q + 7
  | 3 => 3*p + q + 2
  | n + 4 => ArithmeticSequence p q 3 + (n + 1) * (ArithmeticSequence p q 3 - ArithmeticSequence p q 2)

/-- Theorem stating that the 1001st term of the sequence is 5004 -/
theorem arithmetic_sequence_1001st_term (p q : ℝ) :
  ArithmeticSequence p q 1000 = 5004 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1001st_term_l2737_273719


namespace NUMINAMATH_CALUDE_mt_everest_summit_distance_l2737_273700

/-- The distance from the base camp to the summit of Mt. Everest --/
def summit_distance : ℝ := 5800

/-- Hillary's climbing rate in feet per hour --/
def hillary_climb_rate : ℝ := 800

/-- Eddy's climbing rate in feet per hour --/
def eddy_climb_rate : ℝ := 500

/-- Hillary's descent rate in feet per hour --/
def hillary_descent_rate : ℝ := 1000

/-- The distance in feet that Hillary stops short of the summit --/
def hillary_stop_distance : ℝ := 1000

/-- The time in hours from start until Hillary and Eddy pass each other --/
def time_until_pass : ℝ := 6

theorem mt_everest_summit_distance :
  summit_distance = 
    hillary_climb_rate * time_until_pass + hillary_stop_distance ∧
  summit_distance = 
    eddy_climb_rate * time_until_pass + 
    hillary_descent_rate * (time_until_pass - hillary_climb_rate * time_until_pass / hillary_descent_rate) :=
by sorry

end NUMINAMATH_CALUDE_mt_everest_summit_distance_l2737_273700


namespace NUMINAMATH_CALUDE_product_mod_600_l2737_273726

theorem product_mod_600 : (1234 * 2047) % 600 = 198 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_600_l2737_273726


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2737_273718

def a (x : ℝ) : ℝ × ℝ := (x - 1, x)
def b : ℝ × ℝ := (-1, 2)

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), a x = k • b) → x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2737_273718


namespace NUMINAMATH_CALUDE_subway_speed_difference_l2737_273760

/-- The speed function of the subway train -/
def speed (s : ℝ) : ℝ := s^2 + 2*s

/-- The theorem stating the existence of the time when the train was 28 km/h slower -/
theorem subway_speed_difference :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 7 ∧ 
  speed 7 - speed t = 28 ∧ 
  t = 5 := by sorry

end NUMINAMATH_CALUDE_subway_speed_difference_l2737_273760


namespace NUMINAMATH_CALUDE_min_distance_on_hyperbola_l2737_273739

theorem min_distance_on_hyperbola :
  ∀ x y : ℝ, (x^2 / 8 - y^2 / 4 = 1) → (∀ x' y' : ℝ, (x'^2 / 8 - y'^2 / 4 = 1) → |x - y| ≤ |x' - y'|) →
  |x - y| = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_on_hyperbola_l2737_273739


namespace NUMINAMATH_CALUDE_span_equality_iff_multiple_l2737_273730

theorem span_equality_iff_multiple (α₁ β₁ γ₁ α₂ β₂ γ₂ : ℝ) 
  (h₁ : α₁ + β₁ + γ₁ ≠ 0) (h₂ : α₂ + β₂ + γ₂ ≠ 0) :
  Submodule.span ℝ {(α₁, β₁, γ₁)} = Submodule.span ℝ {(α₂, β₂, γ₂)} ↔ 
  ∃ (k : ℝ), k ≠ 0 ∧ (α₁, β₁, γ₁) = (k * α₂, k * β₂, k * γ₂) :=
by sorry

end NUMINAMATH_CALUDE_span_equality_iff_multiple_l2737_273730


namespace NUMINAMATH_CALUDE_sugar_needed_proof_l2737_273711

/-- Given a recipe requiring a total amount of flour, with some flour already added,
    and the remaining flour needed being 2 cups more than the sugar needed,
    prove that the amount of sugar needed is correct. -/
theorem sugar_needed_proof 
  (total_flour : ℕ)  -- Total flour needed
  (added_flour : ℕ)  -- Flour already added
  (h1 : total_flour = 11)  -- Total flour is 11 cups
  (h2 : added_flour = 2)   -- 2 cups of flour already added
  : 
  total_flour - added_flour - 2 = 7  -- Sugar needed is 7 cups
  := by sorry

end NUMINAMATH_CALUDE_sugar_needed_proof_l2737_273711


namespace NUMINAMATH_CALUDE_triangle_perimeter_not_72_l2737_273778

theorem triangle_perimeter_not_72 (a b c : ℝ) : 
  a = 10 → b = 25 → a + b + c = 72 → ¬(a + c > b ∧ b + c > a ∧ a + b > c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_not_72_l2737_273778


namespace NUMINAMATH_CALUDE_point_move_upward_l2737_273734

def Point := ℝ × ℝ

def move_upward (p : Point) (units : ℝ) : Point :=
  (p.1, p.2 + units)

theorem point_move_upward (A B : Point) (h : ℝ) :
  A = (1, -2) →
  h = 1 →
  B = move_upward A h →
  B = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_point_move_upward_l2737_273734


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l2737_273775

theorem simplify_radical_expression :
  Real.sqrt 18 - Real.sqrt 50 + 3 * Real.sqrt (1/2) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l2737_273775


namespace NUMINAMATH_CALUDE_sine_equality_l2737_273707

/-- Given three nonzero real numbers a, b, c and three real angles α, β, γ,
    if a sin α + b sin β + c sin γ = 0 and a cos α + b cos β + c cos γ = 0,
    then sin(β - γ)/a = sin(γ - α)/b = sin(α - β)/c -/
theorem sine_equality (a b c α β γ : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a * Real.sin α + b * Real.sin β + c * Real.sin γ = 0)
  (h2 : a * Real.cos α + b * Real.cos β + c * Real.cos γ = 0) :
  Real.sin (β - γ) / a = Real.sin (γ - α) / b ∧ 
  Real.sin (γ - α) / b = Real.sin (α - β) / c :=
by sorry

end NUMINAMATH_CALUDE_sine_equality_l2737_273707


namespace NUMINAMATH_CALUDE_cuboid_ratio_simplification_l2737_273721

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℕ
  breadth : ℕ
  height : ℕ

/-- Calculates the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Simplifies the ratio of three numbers by dividing each by their GCD -/
def simplifyRatio (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
  let d := gcd3 a b c
  (a / d, b / d, c / d)

/-- The main theorem stating that the given cuboid dimensions simplify to the ratio 6:5:4 -/
theorem cuboid_ratio_simplification (c : CuboidDimensions) 
  (h1 : c.length = 90) 
  (h2 : c.breadth = 75) 
  (h3 : c.height = 60) : 
  simplifyRatio c.length c.breadth c.height = (6, 5, 4) := by
  sorry

end NUMINAMATH_CALUDE_cuboid_ratio_simplification_l2737_273721


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_ten_zeros_l2737_273705

theorem smallest_multiplier_for_ten_zeros (n : ℕ) : 
  (∀ m : ℕ, m < 78125000 → ¬(∃ k : ℕ, 128 * m = k * 10^10)) ∧ 
  (∃ k : ℕ, 128 * 78125000 = k * 10^10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_ten_zeros_l2737_273705


namespace NUMINAMATH_CALUDE_orchids_cut_l2737_273786

theorem orchids_cut (initial_red : ℕ) (initial_white : ℕ) (final_red : ℕ) : 
  final_red - initial_red = final_red - initial_red :=
by
  sorry

#check orchids_cut 9 3 15

end NUMINAMATH_CALUDE_orchids_cut_l2737_273786


namespace NUMINAMATH_CALUDE_composition_equation_solution_l2737_273727

theorem composition_equation_solution (δ φ : ℝ → ℝ) (h1 : ∀ x, δ x = 2 * x + 5) 
  (h2 : ∀ x, φ x = 9 * x + 6) (h3 : δ (φ x) = 3) : x = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l2737_273727


namespace NUMINAMATH_CALUDE_count_multiples_of_ten_l2737_273702

theorem count_multiples_of_ten : ∃ n : ℕ, n = (Finset.filter (λ x => x % 10 = 0 ∧ x > 9 ∧ x < 101) (Finset.range 101)).card ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_ten_l2737_273702


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2737_273704

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 + 8x - 6 -/
def a : ℝ := 5
def b : ℝ := 8
def c : ℝ := -6

/-- Theorem: The discriminant of 5x^2 + 8x - 6 is 184 -/
theorem quadratic_discriminant : discriminant a b c = 184 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2737_273704


namespace NUMINAMATH_CALUDE_picnic_attendance_l2737_273781

theorem picnic_attendance (total_students : ℕ) (picnic_attendees : ℕ) 
  (girls : ℕ) (boys : ℕ) : 
  total_students = 1500 →
  picnic_attendees = 975 →
  total_students = girls + boys →
  picnic_attendees = (3 * girls / 4) + (3 * boys / 5) →
  (3 * girls / 4 : ℕ) = 375 :=
by sorry

end NUMINAMATH_CALUDE_picnic_attendance_l2737_273781


namespace NUMINAMATH_CALUDE_inequality_proof_l2737_273769

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2737_273769


namespace NUMINAMATH_CALUDE_sock_combination_count_l2737_273716

/-- Represents the color of a sock pair -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents the pattern of a sock pair -/
inductive Pattern
  | Striped
  | Dotted
  | Checkered
  | Plain

/-- Represents a pair of socks -/
structure SockPair :=
  (color : Color)
  (pattern : Pattern)

def total_pairs : ℕ := 12
def red_pairs : ℕ := 4
def blue_pairs : ℕ := 4
def green_pairs : ℕ := 4

def sock_collection : List SockPair := sorry

/-- Checks if two socks form a valid combination according to the constraints -/
def is_valid_combination (sock1 sock2 : SockPair) : Bool := sorry

/-- Counts the number of valid combinations -/
def count_valid_combinations (socks : List SockPair) : ℕ := sorry

theorem sock_combination_count :
  count_valid_combinations sock_collection = 12 := by sorry

end NUMINAMATH_CALUDE_sock_combination_count_l2737_273716


namespace NUMINAMATH_CALUDE_special_function_property_l2737_273782

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (f 2 = 2) ∧ 
  (∀ x y : ℝ, f (x * y + f x) = x * f y + f x + x^2)

/-- The number of possible values for f(1/2) -/
def num_values (f : ℝ → ℝ) : ℕ :=
  sorry

/-- The sum of all possible values for f(1/2) -/
def sum_values (f : ℝ → ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  (num_values f : ℝ) * sum_values f = -2 :=
sorry

end NUMINAMATH_CALUDE_special_function_property_l2737_273782


namespace NUMINAMATH_CALUDE_actual_car_mass_l2737_273751

/-- The mass of a scaled object given the original mass and scale factor. -/
def scaled_mass (original_mass : ℝ) (scale_factor : ℝ) : ℝ :=
  original_mass * scale_factor^3

/-- The mass of the actual car body is 1024 kg. -/
theorem actual_car_mass (model_mass : ℝ) (scale_factor : ℝ) 
  (h1 : model_mass = 2)
  (h2 : scale_factor = 8) :
  scaled_mass model_mass scale_factor = 1024 := by
  sorry

end NUMINAMATH_CALUDE_actual_car_mass_l2737_273751


namespace NUMINAMATH_CALUDE_selection_theorem_l2737_273701

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of workers -/
def total_workers : ℕ := 11

/-- The number of workers who can do typesetting -/
def typesetting_workers : ℕ := 7

/-- The number of workers who can do printing -/
def printing_workers : ℕ := 6

/-- The number of workers to be selected for each task -/
def workers_per_task : ℕ := 4

/-- The number of ways to select workers for typesetting and printing -/
def selection_ways : ℕ := 
  choose typesetting_workers workers_per_task * 
  choose (total_workers - workers_per_task) workers_per_task +
  choose (printing_workers - workers_per_task + 1) (printing_workers - workers_per_task) * 
  choose 2 1 * 
  choose (typesetting_workers - 1) workers_per_task +
  choose (printing_workers - workers_per_task + 2) (printing_workers - workers_per_task) * 
  choose (typesetting_workers - 2) workers_per_task * 
  choose 2 2

theorem selection_theorem : selection_ways = 185 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l2737_273701


namespace NUMINAMATH_CALUDE_determinant_expansion_second_column_l2737_273748

theorem determinant_expansion_second_column 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) : 
  let M := ![![a₁, 3, b₁], ![a₂, 2, b₂], ![a₃, -2, b₃]]
  Matrix.det M = 3 * Matrix.det ![![a₂, b₂], ![a₃, b₃]] + 
                 2 * Matrix.det ![![a₁, b₁], ![a₃, b₃]] - 
                 2 * Matrix.det ![![a₁, b₁], ![a₂, b₂]] := by
  sorry

end NUMINAMATH_CALUDE_determinant_expansion_second_column_l2737_273748


namespace NUMINAMATH_CALUDE_ab_value_l2737_273723

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2737_273723
