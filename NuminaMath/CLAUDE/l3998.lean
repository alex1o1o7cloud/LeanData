import Mathlib

namespace NUMINAMATH_CALUDE_smallest_x_for_cube_l3998_399893

theorem smallest_x_for_cube (x M : ℕ+) : 
  (∀ y : ℕ+, y < x → ¬∃ N : ℕ+, 720 * y = N^3) → 
  (∃ N : ℕ+, 720 * x = N^3) → 
  x = 300 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_cube_l3998_399893


namespace NUMINAMATH_CALUDE_obtuse_angle_range_l3998_399828

def vector_a : ℝ × ℝ := (2, -1)
def vector_b (t : ℝ) : ℝ × ℝ := (t, 3)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def is_obtuse (v w : ℝ × ℝ) : Prop := dot_product v w < 0

theorem obtuse_angle_range (t : ℝ) :
  is_obtuse vector_a (vector_b t) →
  t ∈ (Set.Iio (-6) ∪ Set.Ioo (-6) (3/2)) :=
sorry

end NUMINAMATH_CALUDE_obtuse_angle_range_l3998_399828


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l3998_399879

/-- The nth positive integer that is both odd and a multiple of 5 -/
def oddMultipleOf5 (n : ℕ) : ℕ := 2 * n * 5 - 5

theorem fifteenth_odd_multiple_of_5 : oddMultipleOf5 15 = 145 := by sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l3998_399879


namespace NUMINAMATH_CALUDE_square_side_length_average_l3998_399852

theorem square_side_length_average : 
  let areas : List ℝ := [25, 64, 121, 196]
  let side_lengths := areas.map Real.sqrt
  (side_lengths.sum / side_lengths.length : ℝ) = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l3998_399852


namespace NUMINAMATH_CALUDE_distance_traveled_l3998_399863

-- Define the velocity function
def velocity (t : ℝ) : ℝ := 2 * t + 3

-- Define the theorem
theorem distance_traveled (a b : ℝ) (ha : a = 3) (hb : b = 5) :
  ∫ x in a..b, velocity x = 22 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3998_399863


namespace NUMINAMATH_CALUDE_unique_integer_divisibility_l3998_399878

theorem unique_integer_divisibility (n : ℕ) : 
  n > 1 → (∃ k : ℕ, (2^n + 1) = k * n^2) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisibility_l3998_399878


namespace NUMINAMATH_CALUDE_triangle_equilateral_condition_l3998_399896

theorem triangle_equilateral_condition (A B C : Real) (a b c : Real) :
  (A + B + C = Real.pi) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a / Real.cos A = b / Real.cos B) →
  (b / Real.cos B = c / Real.cos C) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (A = B ∧ B = C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_condition_l3998_399896


namespace NUMINAMATH_CALUDE_book_pages_theorem_l3998_399807

/-- Calculates the number of pages with text in a book with given specifications. -/
def pages_with_text (total_pages : ℕ) (image_pages : ℕ) (intro_pages : ℕ) : ℕ :=
  let remaining_pages := total_pages - image_pages - intro_pages
  remaining_pages / 2

/-- Theorem stating that a book with 98 pages, half images, 11 intro pages, 
    and remaining pages split equally between blank and text, has 19 pages of text. -/
theorem book_pages_theorem : 
  pages_with_text 98 (98 / 2) 11 = 19 := by
sorry

#eval pages_with_text 98 (98 / 2) 11

end NUMINAMATH_CALUDE_book_pages_theorem_l3998_399807


namespace NUMINAMATH_CALUDE_decagon_triangles_l3998_399814

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def triangles_from_decagon : ℕ := Nat.choose decagon_vertices triangle_vertices

theorem decagon_triangles :
  triangles_from_decagon = 120 := by sorry

end NUMINAMATH_CALUDE_decagon_triangles_l3998_399814


namespace NUMINAMATH_CALUDE_prob_at_least_one_multiple_of_four_l3998_399838

/-- The number of integers from 1 to 60 inclusive -/
def total_numbers : ℕ := 60

/-- The number of multiples of 4 from 1 to 60 inclusive -/
def multiples_of_four : ℕ := 15

/-- The probability of choosing a number that is not a multiple of 4 -/
def prob_not_multiple_of_four : ℚ := (total_numbers - multiples_of_four) / total_numbers

theorem prob_at_least_one_multiple_of_four :
  1 - prob_not_multiple_of_four ^ 2 = 7 / 16 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_multiple_of_four_l3998_399838


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l3998_399854

/-- Given a polynomial g(x) with three distinct roots that are also roots of f(x),
    prove that f(2) = -16342.5 -/
theorem polynomial_root_problem (p q d : ℝ) : 
  let g : ℝ → ℝ := λ x => x^3 + p*x^2 + 2*x + 15
  let f : ℝ → ℝ := λ x => x^4 + 2*x^3 + q*x^2 + 150*x + d
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g r₁ = 0 ∧ g r₂ = 0 ∧ g r₃ = 0 ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  f 2 = -16342.5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l3998_399854


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l3998_399813

theorem no_solutions_in_interval (x : ℝ) :
  -π ≤ x ∧ x ≤ 3*π →
  ¬(1 / Real.sin x + 1 / Real.cos x = 4) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l3998_399813


namespace NUMINAMATH_CALUDE_Ann_age_is_6_l3998_399849

/-- Ann's current age -/
def Ann_age : ℕ := sorry

/-- Tom's current age -/
def Tom_age : ℕ := 2 * Ann_age

/-- The sum of their ages 10 years later -/
def sum_ages_later : ℕ := Ann_age + 10 + Tom_age + 10

theorem Ann_age_is_6 : Ann_age = 6 := by
  have h1 : sum_ages_later = 38 := sorry
  sorry

end NUMINAMATH_CALUDE_Ann_age_is_6_l3998_399849


namespace NUMINAMATH_CALUDE_library_books_problem_l3998_399864

theorem library_books_problem (initial_books : ℕ) : 
  initial_books - 227 + 56 - 35 = 29 → initial_books = 235 :=
by sorry

end NUMINAMATH_CALUDE_library_books_problem_l3998_399864


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3998_399874

theorem sum_of_coefficients (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₁*(x-1)^4 + a₂*(x-1)^3 + a₃*(x-1)^2 + a₄*(x-1) + a₅ = x^4) →
  a₂ + a₃ + a₄ = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3998_399874


namespace NUMINAMATH_CALUDE_garden_area_l3998_399808

theorem garden_area (side_length : ℝ) (h1 : 30 * side_length = 1500) (h2 : 8 * (4 * side_length) = 1500) :
  side_length ^ 2 = 2197.265625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l3998_399808


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3998_399811

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 63 = (X - 3) * q + 24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3998_399811


namespace NUMINAMATH_CALUDE_number_always_divisible_by_396_l3998_399894

/-- Represents a permutation of digits 0 to 9 -/
def DigitPermutation := Fin 10 → Fin 10

/-- Constructs the number based on the given permutation -/
def constructNumber (p : DigitPermutation) : ℕ :=
  -- Implementation details omitted
  sorry

/-- The theorem to be proved -/
theorem number_always_divisible_by_396 (p : DigitPermutation) :
  396 ∣ constructNumber p := by
  sorry

end NUMINAMATH_CALUDE_number_always_divisible_by_396_l3998_399894


namespace NUMINAMATH_CALUDE_total_pokemon_cards_l3998_399871

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 4

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 14

/-- The total number of Pokemon cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_pokemon_cards : total_cards = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_pokemon_cards_l3998_399871


namespace NUMINAMATH_CALUDE_rex_cards_left_l3998_399898

theorem rex_cards_left (nicole_cards : ℕ) (cindy_cards : ℕ) (rex_cards : ℕ) : 
  nicole_cards = 400 →
  cindy_cards = 2 * nicole_cards →
  rex_cards = (nicole_cards + cindy_cards) / 2 →
  rex_cards / 4 = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_rex_cards_left_l3998_399898


namespace NUMINAMATH_CALUDE_solve_equation_l3998_399840

theorem solve_equation (x : ℝ) : 3 * x = (20 - x) + 20 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3998_399840


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_perfect_square_l3998_399842

theorem consecutive_integers_product_plus_one_is_perfect_square (n : ℤ) :
  ∃ m : ℤ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_perfect_square_l3998_399842


namespace NUMINAMATH_CALUDE_sunshine_car_rentals_rate_l3998_399832

theorem sunshine_car_rentals_rate (sunshine_daily_rate city_daily_rate city_mile_rate : ℚ)
  (equal_cost_miles : ℕ) :
  sunshine_daily_rate = 17.99 ∧
  city_daily_rate = 18.95 ∧
  city_mile_rate = 0.16 ∧
  equal_cost_miles = 48 →
  ∃ sunshine_mile_rate : ℚ,
    sunshine_mile_rate = 0.18 ∧
    sunshine_daily_rate + sunshine_mile_rate * equal_cost_miles =
    city_daily_rate + city_mile_rate * equal_cost_miles :=
by sorry

end NUMINAMATH_CALUDE_sunshine_car_rentals_rate_l3998_399832


namespace NUMINAMATH_CALUDE_bills_weights_theorem_l3998_399853

/-- The total weight of sand in two jugs filled partially with sand -/
def total_weight (jug_capacity : ℝ) (fill_percentage : ℝ) (num_jugs : ℕ) (sand_density : ℝ) : ℝ :=
  jug_capacity * fill_percentage * (num_jugs : ℝ) * sand_density

/-- Theorem stating the total weight of sand in Bill's improvised weights -/
theorem bills_weights_theorem :
  total_weight 2 0.7 2 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_bills_weights_theorem_l3998_399853


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3998_399829

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) : 
  -- The triangle is isosceles
  (α = β ∨ β = γ ∨ α = γ) →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- One angle is 70°
  (α = 70 ∨ β = 70 ∨ γ = 70) →
  -- The vertex angle (the one that's not equal to the other two) is either 70° or 40°
  (((α ≠ β ∧ α ≠ γ) → α = 70 ∨ α = 40) ∧
   ((β ≠ α ∧ β ≠ γ) → β = 70 ∨ β = 40) ∧
   ((γ ≠ α ∧ γ ≠ β) → γ = 70 ∨ γ = 40)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3998_399829


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3998_399822

theorem solution_set_inequality (x : ℝ) :
  (x + 2) * (x - 1) > 0 ↔ x < -2 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3998_399822


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3998_399820

/-- Given a function f(x) = x^3 + ax^2 + x with a tangent line at (1, f(1)) having slope 6, 
    prove that a = 1. -/
theorem tangent_line_slope (a : ℝ) : 
  let f := λ x : ℝ => x^3 + a*x^2 + x
  let f' := λ x : ℝ => 3*x^2 + 2*a*x + 1
  f' 1 = 6 → a = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3998_399820


namespace NUMINAMATH_CALUDE_ben_hours_per_shift_l3998_399882

/-- Represents the time it takes Ben to build one rocking chair -/
def time_per_chair : ℕ := 5

/-- Represents the number of chairs Ben builds in 10 days -/
def chairs_in_ten_days : ℕ := 16

/-- Represents the number of days Ben works -/
def work_days : ℕ := 10

/-- Represents the number of shifts Ben works per day -/
def shifts_per_day : ℕ := 1

/-- Theorem stating that Ben works 8 hours per shift -/
theorem ben_hours_per_shift : 
  (chairs_in_ten_days * time_per_chair) / work_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_ben_hours_per_shift_l3998_399882


namespace NUMINAMATH_CALUDE_carpenters_completion_time_l3998_399870

/-- The time it takes for two carpenters to complete a job together -/
theorem carpenters_completion_time 
  (rate1 : ℚ) -- Work rate of the first carpenter
  (rate2 : ℚ) -- Work rate of the second carpenter
  (h1 : rate1 = 1 / 7) -- First carpenter's work rate
  (h2 : rate2 = 1 / (35/2)) -- Second carpenter's work rate
  : (1 : ℚ) / (rate1 + rate2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_carpenters_completion_time_l3998_399870


namespace NUMINAMATH_CALUDE_prob_class1_drew_two_mc_correct_expected_rounds_correct_l3998_399890

-- Define the boxes
structure Box where
  multiple_choice : ℕ
  fill_in_blank : ℕ

-- Define the game
structure Game where
  box_a : Box
  box_b : Box
  class_6_first_win_prob : ℚ
  next_win_prob : ℚ

-- Define the problem
def chinese_culture_competition : Game :=
  { box_a := { multiple_choice := 5, fill_in_blank := 3 }
  , box_b := { multiple_choice := 4, fill_in_blank := 3 }
  , class_6_first_win_prob := 3/5
  , next_win_prob := 2/5
  }

-- Part 1: Probability calculation
def prob_class1_drew_two_mc (g : Game) : ℚ :=
  20/49

-- Part 2: Expected value calculation
def expected_rounds (g : Game) : ℚ :=
  537/125

-- Theorem statements
theorem prob_class1_drew_two_mc_correct (g : Game) :
  g = chinese_culture_competition →
  prob_class1_drew_two_mc g = 20/49 := by sorry

theorem expected_rounds_correct (g : Game) :
  g = chinese_culture_competition →
  expected_rounds g = 537/125 := by sorry

end NUMINAMATH_CALUDE_prob_class1_drew_two_mc_correct_expected_rounds_correct_l3998_399890


namespace NUMINAMATH_CALUDE_f_equals_g_l3998_399881

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l3998_399881


namespace NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l3998_399869

theorem consecutive_four_product_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l3998_399869


namespace NUMINAMATH_CALUDE_perpendicular_condition_l3998_399855

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicularity for two lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line (a+2)x+3ay+1=0 -/
def line1 (a : ℝ) : Line :=
  { a := a + 2, b := 3 * a, c := 1 }

/-- The second line (a-2)x+(a+2)y-3=0 -/
def line2 (a : ℝ) : Line :=
  { a := a - 2, b := a + 2, c := -3 }

theorem perpendicular_condition (a : ℝ) :
  (a = -2 → perpendicular (line1 a) (line2 a)) ∧
  (∃ b : ℝ, b ≠ -2 ∧ perpendicular (line1 b) (line2 b)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l3998_399855


namespace NUMINAMATH_CALUDE_lemonade_production_l3998_399810

/-- Given that John can prepare 15 lemonades from 3 lemons, 
    prove that he can make 90 lemonades from 18 lemons. -/
theorem lemonade_production (initial_lemons : ℕ) (initial_lemonades : ℕ) (new_lemons : ℕ) : 
  initial_lemons = 3 → initial_lemonades = 15 → new_lemons = 18 →
  (new_lemons * initial_lemonades / initial_lemons : ℕ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_production_l3998_399810


namespace NUMINAMATH_CALUDE_book_selling_price_l3998_399877

/-- Proves that the selling price of each book is $1.50 --/
theorem book_selling_price (total_books : ℕ) (records_bought : ℕ) (record_price : ℚ) (money_left : ℚ) :
  total_books = 200 →
  records_bought = 75 →
  record_price = 3 →
  money_left = 75 →
  (total_books : ℚ) * (1.5 : ℚ) = records_bought * record_price + money_left :=
by
  sorry

#check book_selling_price

end NUMINAMATH_CALUDE_book_selling_price_l3998_399877


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_negative_l3998_399883

/-- 
Given a real number m, prove that if the solution set of the inequality 
(mx-1)(x-2) > 0 is {x | 1/m < x < 2}, then m < 0.
-/
theorem inequality_solution_implies_m_negative (m : ℝ) : 
  (∀ x, (m * x - 1) * (x - 2) > 0 ↔ 1/m < x ∧ x < 2) → m < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_negative_l3998_399883


namespace NUMINAMATH_CALUDE_total_boys_in_class_l3998_399815

/-- Given a circular arrangement of students, if the 10th and 40th positions
    are opposite each other and only every other student is counted,
    then the total number of boys in the class is 30. -/
theorem total_boys_in_class (n : ℕ) 
  (circular_arrangement : n > 0)
  (opposite_positions : 40 - 10 = n / 2)
  (count_every_other : n % 2 = 0) : 
  n / 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_boys_in_class_l3998_399815


namespace NUMINAMATH_CALUDE_coin_difference_is_eight_l3998_399837

/-- Represents the available coin denominations in cents -/
def coin_denominations : List Nat := [5, 10, 20, 25]

/-- The amount to be paid in cents -/
def amount_to_pay : Nat := 50

/-- Calculates the minimum number of coins needed to make the given amount -/
def min_coins (amount : Nat) (denominations : List Nat) : Nat :=
  sorry

/-- Calculates the maximum number of coins needed to make the given amount -/
def max_coins (amount : Nat) (denominations : List Nat) : Nat :=
  sorry

/-- Proves that the difference between the maximum and minimum number of coins
    needed to make 50 cents using the given denominations is 8 -/
theorem coin_difference_is_eight :
  max_coins amount_to_pay coin_denominations - min_coins amount_to_pay coin_denominations = 8 :=
by sorry

end NUMINAMATH_CALUDE_coin_difference_is_eight_l3998_399837


namespace NUMINAMATH_CALUDE_quadratic_sum_bound_l3998_399826

/-- Represents a quadratic function of the form y = x^2 - (a+2)x + 2a + 1 -/
def QuadraticFunction (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - (a + 2) * x + 2 * a + 1

/-- Theorem: For a quadratic function passing through (-1, y₀) where y₀ is the minimum,
    any two different points A(m, n) and B(2-m, p) on the parabola satisfy n + p > -8 -/
theorem quadratic_sum_bound
  (a : ℝ)
  (y₀ : ℝ)
  (h1 : QuadraticFunction a (-1) = y₀)
  (h2 : ∀ x y, y = QuadraticFunction a x → y ≥ y₀)
  (m n p : ℝ)
  (h3 : n = QuadraticFunction a m)
  (h4 : p = QuadraticFunction a (2 - m))
  (h5 : m ≠ 2 - m) :
  n + p > -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_bound_l3998_399826


namespace NUMINAMATH_CALUDE_max_value_of_function_l3998_399899

theorem max_value_of_function (x : ℝ) (h : x^2 + x + 1 ≠ 0) :
  ∃ (M : ℝ), M = 13/3 ∧ ∀ (y : ℝ), (3*x^2 + 3*x + 4) / (x^2 + x + 1) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3998_399899


namespace NUMINAMATH_CALUDE_students_in_multiple_activities_l3998_399844

theorem students_in_multiple_activities 
  (total_students : ℕ) 
  (debate_only : ℕ) 
  (singing_only : ℕ) 
  (dance_only : ℕ) 
  (no_activity : ℕ) 
  (h1 : total_students = 55)
  (h2 : debate_only = 10)
  (h3 : singing_only = 18)
  (h4 : dance_only = 8)
  (h5 : no_activity = 5) :
  total_students - (debate_only + singing_only + dance_only + no_activity) = 14 := by
  sorry

end NUMINAMATH_CALUDE_students_in_multiple_activities_l3998_399844


namespace NUMINAMATH_CALUDE_equation_solution_l3998_399830

theorem equation_solution :
  ∃ y : ℚ, (3 / y - (5 / y) / (7 / y) = 1.2) ∧ (y = 105 / 67) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3998_399830


namespace NUMINAMATH_CALUDE_max_triangle_area_is_sqrt3_div_2_l3998_399848

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line in 2D space -/
structure Line where
  k : ℝ
  m : ℝ

/-- The maximum area of triangle AOB for the given ellipse and line conditions -/
def max_triangle_area (e : Ellipse) (l : Line) : ℝ :=
  sorry

/-- Main theorem: The maximum area of triangle AOB is √3/2 under the given conditions -/
theorem max_triangle_area_is_sqrt3_div_2 
  (e : Ellipse) 
  (h_vertex : e.b = 1)
  (h_eccentricity : Real.sqrt (e.a^2 - e.b^2) / e.a = Real.sqrt 6 / 3)
  (l : Line)
  (h_distance : |l.m| / Real.sqrt (1 + l.k^2) = Real.sqrt 3 / 2) :
  max_triangle_area e l = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_is_sqrt3_div_2_l3998_399848


namespace NUMINAMATH_CALUDE_grocery_store_costs_l3998_399888

theorem grocery_store_costs (total_cost : ℝ) (salary_fraction : ℝ) (delivery_fraction : ℝ) 
  (h1 : total_cost = 4000)
  (h2 : salary_fraction = 2/5)
  (h3 : delivery_fraction = 1/4) : 
  total_cost * (1 - salary_fraction) * (1 - delivery_fraction) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_costs_l3998_399888


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3998_399834

open Real

/-- Given a function g: ℝ → ℝ satisfying the functional equation
    (g(x) * g(y) - g(x*y)) / 5 = x + y + 4 for all x, y ∈ ℝ,
    prove that g(x) = x + 5 for all x ∈ ℝ. -/
theorem functional_equation_solution (g : ℝ → ℝ) 
    (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = x + y + 4) :
  ∀ x : ℝ, g x = x + 5 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3998_399834


namespace NUMINAMATH_CALUDE_total_age_is_47_l3998_399892

/-- Given three people A, B, and C, where A is two years older than B, B is twice as old as C, 
    and B is 18 years old, prove that the total of their ages is 47 years. -/
theorem total_age_is_47 (A B C : ℕ) : 
  B = 18 → A = B + 2 → B = 2 * C → A + B + C = 47 := by sorry

end NUMINAMATH_CALUDE_total_age_is_47_l3998_399892


namespace NUMINAMATH_CALUDE_complex_number_property_l3998_399843

theorem complex_number_property (w : ℂ) (h : w + 1 / w = 2 * Real.cos (π / 4)) :
  w^12 + 1 / w^12 = -2 := by sorry

end NUMINAMATH_CALUDE_complex_number_property_l3998_399843


namespace NUMINAMATH_CALUDE_student_count_l3998_399835

theorem student_count (total : ℕ) 
  (h1 : total / 5 + total / 4 + total / 2 + 30 = total) : total = 600 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l3998_399835


namespace NUMINAMATH_CALUDE_fourth_term_constant_implies_n_equals_5_l3998_399802

theorem fourth_term_constant_implies_n_equals_5 (n : ℕ) (x : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (Nat.choose n 3) * (-1/2)^3 * x^((n-5)/2) = k) →
  n = 5 :=
sorry

end NUMINAMATH_CALUDE_fourth_term_constant_implies_n_equals_5_l3998_399802


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3998_399841

/-- Given a boat that travels 32 km along a stream and 12 km against the same stream
    in one hour each, its speed in still water is 22 km/hr. -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ) 
  (h1 : along_stream = 32) 
  (h2 : against_stream = 12) : 
  (along_stream + against_stream) / 2 = 22 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3998_399841


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3998_399886

theorem lcm_from_product_and_hcf (a b : ℕ+) :
  a * b = 82500 → Nat.gcd a b = 55 → Nat.lcm a b = 1500 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3998_399886


namespace NUMINAMATH_CALUDE_cows_eating_husk_l3998_399824

/-- The number of bags of husk eaten by a group of cows in 30 days -/
def bags_eaten (num_cows : ℕ) (bags_per_cow : ℕ) : ℕ :=
  num_cows * bags_per_cow

/-- Theorem: 30 cows eat 30 bags of husk in 30 days -/
theorem cows_eating_husk :
  bags_eaten 30 1 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cows_eating_husk_l3998_399824


namespace NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_l3998_399861

-- Problem 1
theorem problem_one (α : Real) (h : Real.tan α = 3) :
  (3 * Real.sin α + Real.cos α) / (Real.sin α - 2 * Real.cos α) = 10 := by sorry

-- Problem 2
theorem problem_two (α : Real) :
  (-Real.sin (π + α) + Real.sin (-α) - Real.tan (2*π + α)) /
  (Real.tan (α + π) + Real.cos (-α) + Real.cos (π - α)) = -1 := by sorry

-- Problem 3
theorem problem_three (α : Real) (h1 : Real.sin α + Real.cos α = 1/2) (h2 : 0 < α) (h3 : α < π) :
  Real.sin α * Real.cos α = -3/8 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_l3998_399861


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3998_399818

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 + 2 * Complex.I → z = -2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3998_399818


namespace NUMINAMATH_CALUDE_silverware_probability_l3998_399880

/-- The number of each type and color of silverware in the drawer -/
def num_each : ℕ := 8

/-- The total number of pieces of silverware in the drawer -/
def total_pieces : ℕ := 6 * num_each

/-- The number of ways to choose any 3 items from the drawer -/
def total_ways : ℕ := Nat.choose total_pieces 3

/-- The number of ways to choose one fork, one spoon, and one knife of different colors -/
def favorable_ways : ℕ := 2 * (num_each * num_each * num_each)

/-- The probability of selecting one fork, one spoon, and one knife of different colors -/
def probability : ℚ := favorable_ways / total_ways

theorem silverware_probability :
  probability = 32 / 541 := by sorry

end NUMINAMATH_CALUDE_silverware_probability_l3998_399880


namespace NUMINAMATH_CALUDE_merry_and_brother_lambs_l3998_399827

/-- The number of lambs Merry and her brother have in total -/
def total_lambs (merry_lambs : ℕ) (brother_extra : ℕ) : ℕ :=
  merry_lambs + (merry_lambs + brother_extra)

/-- Theorem stating the total number of lambs Merry and her brother have -/
theorem merry_and_brother_lambs :
  total_lambs 10 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_merry_and_brother_lambs_l3998_399827


namespace NUMINAMATH_CALUDE_total_monthly_pay_is_12708_l3998_399805

-- Define the structure for an employee
structure Employee where
  name : String
  hours_per_week : ℕ
  hourly_rate : ℕ

-- Define the list of employees
def employees : List Employee := [
  { name := "Fiona", hours_per_week := 40, hourly_rate := 20 },
  { name := "John", hours_per_week := 30, hourly_rate := 22 },
  { name := "Jeremy", hours_per_week := 25, hourly_rate := 18 },
  { name := "Katie", hours_per_week := 35, hourly_rate := 21 },
  { name := "Matt", hours_per_week := 28, hourly_rate := 19 }
]

-- Define the number of weeks in a month
def weeks_in_month : ℕ := 4

-- Calculate the monthly pay for all employees
def total_monthly_pay : ℕ :=
  employees.foldl (fun acc e => acc + e.hours_per_week * e.hourly_rate * weeks_in_month) 0

-- Theorem stating that the total monthly pay is $12,708
theorem total_monthly_pay_is_12708 : total_monthly_pay = 12708 := by
  sorry

end NUMINAMATH_CALUDE_total_monthly_pay_is_12708_l3998_399805


namespace NUMINAMATH_CALUDE_redo_horseshoe_profit_l3998_399839

/-- Calculates the profit for Redo's horseshoe manufacturing --/
def calculate_profit (initial_outlay : ℕ) (cost_per_set : ℕ) (price_per_set : ℕ) (num_sets : ℕ) : ℤ :=
  let total_cost : ℕ := initial_outlay + cost_per_set * num_sets
  let revenue : ℕ := price_per_set * num_sets
  (revenue : ℤ) - (total_cost : ℤ)

theorem redo_horseshoe_profit :
  calculate_profit 10000 20 50 500 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_redo_horseshoe_profit_l3998_399839


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l3998_399862

theorem sqrt_meaningful_iff_geq_two (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l3998_399862


namespace NUMINAMATH_CALUDE_line_l_satisfies_conditions_l3998_399856

-- Define the points
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (5, 4)
def C : ℝ × ℝ := (3, 7)
def D : ℝ × ℝ := (7, 1)
def E : ℝ × ℝ := (10, 2)
def F : ℝ × ℝ := (8, 6)

-- Define the line l
def l (x y : ℝ) : Prop := 10 * x - 2 * y - 55 = 0

-- Define the line DF
def DF (x y : ℝ) : Prop := y = 5 * x - 34

-- Define the triangles ABC and DEF
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | ∃ (t₁ t₂ t₃ : ℝ), t₁ ≥ 0 ∧ t₂ ≥ 0 ∧ t₃ ≥ 0 ∧ t₁ + t₂ + t₃ = 1 ∧
    p = (t₁ * A.1 + t₂ * B.1 + t₃ * C.1, t₁ * A.2 + t₂ * B.2 + t₃ * C.2)}

def triangle_DEF : Set (ℝ × ℝ) :=
  {p | ∃ (t₁ t₂ t₃ : ℝ), t₁ ≥ 0 ∧ t₂ ≥ 0 ∧ t₃ ≥ 0 ∧ t₁ + t₂ + t₃ = 1 ∧
    p = (t₁ * D.1 + t₂ * E.1 + t₃ * F.1, t₁ * D.2 + t₂ * E.2 + t₃ * F.2)}

-- Define the distance function
def distance (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  sorry -- Implementation of distance function

theorem line_l_satisfies_conditions :
  (∀ x y, l x y → DF x y) ∧ -- l is parallel to DF
  (∃ d : ℝ, 
    (∀ p ∈ triangle_ABC, distance p l ≥ d) ∧
    (∃ p₁ ∈ triangle_ABC, distance p₁ l = d) ∧
    (∀ p ∈ triangle_DEF, distance p l ≥ d) ∧
    (∃ p₂ ∈ triangle_DEF, distance p₂ l = d)) :=
by sorry

end NUMINAMATH_CALUDE_line_l_satisfies_conditions_l3998_399856


namespace NUMINAMATH_CALUDE_square_difference_equals_one_l3998_399858

theorem square_difference_equals_one : (825 : ℤ) * 825 - 824 * 826 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_one_l3998_399858


namespace NUMINAMATH_CALUDE_money_distribution_sum_l3998_399803

/-- Represents the share of money for each person --/
structure Share where
  amount : ℝ

/-- Represents the distribution of money among A, B, and C --/
structure Distribution where
  a : Share
  b : Share
  c : Share

/-- The conditions of the problem --/
def satisfiesConditions (d : Distribution) : Prop :=
  d.b.amount = 0.65 * d.a.amount ∧
  d.c.amount = 0.40 * d.a.amount ∧
  d.c.amount = 32

/-- The total sum of money --/
def totalSum (d : Distribution) : ℝ :=
  d.a.amount + d.b.amount + d.c.amount

/-- The theorem to prove --/
theorem money_distribution_sum :
  ∀ d : Distribution, satisfiesConditions d → totalSum d = 164 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_sum_l3998_399803


namespace NUMINAMATH_CALUDE_rain_probability_l3998_399825

theorem rain_probability (monday_rain : ℝ) (tuesday_rain : ℝ) (no_rain : ℝ)
  (h1 : monday_rain = 0.7)
  (h2 : tuesday_rain = 0.5)
  (h3 : no_rain = 0.2) :
  monday_rain + tuesday_rain - (1 - no_rain) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l3998_399825


namespace NUMINAMATH_CALUDE_diminished_number_divisibility_l3998_399809

def smallest_number : ℕ := 1013
def diminished_number : ℕ := smallest_number - 5

def divisors : Set ℕ := {1, 2, 3, 4, 6, 7, 8, 9, 12, 14, 16, 18, 21, 24, 28, 36, 42, 48, 56, 63, 72, 84, 96, 112, 126, 144, 168, 192, 252, 336, 504, 1008}

theorem diminished_number_divisibility :
  (∀ n ∈ divisors, diminished_number % n = 0) ∧
  (∀ m : ℕ, m > 0 → m ∉ divisors → diminished_number % m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_diminished_number_divisibility_l3998_399809


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3998_399868

theorem contrapositive_equivalence :
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔
  (∀ a b : ℝ, a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3998_399868


namespace NUMINAMATH_CALUDE_lisa_scenery_photos_l3998_399812

-- Define the variables
def animal_photos : ℕ := 10
def flower_photos : ℕ := 3 * animal_photos
def scenery_photos : ℕ := flower_photos - 10
def total_photos : ℕ := 45

-- Theorem to prove
theorem lisa_scenery_photos :
  scenery_photos = 20 ∧
  animal_photos + flower_photos + scenery_photos = total_photos :=
by sorry

end NUMINAMATH_CALUDE_lisa_scenery_photos_l3998_399812


namespace NUMINAMATH_CALUDE_max_weight_difference_is_0_6_l3998_399845

/-- Represents the weight range of a flour bag -/
structure FlourBag where
  center : ℝ
  tolerance : ℝ

/-- Calculates the maximum weight of a flour bag -/
def max_weight (bag : FlourBag) : ℝ := bag.center + bag.tolerance

/-- Calculates the minimum weight of a flour bag -/
def min_weight (bag : FlourBag) : ℝ := bag.center - bag.tolerance

/-- Theorem: The maximum difference in weights between any two bags is 0.6 kg -/
theorem max_weight_difference_is_0_6 (bag1 bag2 bag3 : FlourBag)
  (h1 : bag1 = ⟨25, 0.1⟩)
  (h2 : bag2 = ⟨25, 0.2⟩)
  (h3 : bag3 = ⟨25, 0.3⟩) :
  (max_weight bag3 - min_weight bag3) = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_max_weight_difference_is_0_6_l3998_399845


namespace NUMINAMATH_CALUDE_alpha_gamma_relation_l3998_399846

theorem alpha_gamma_relation (α β γ : ℝ) 
  (h1 : β = 10^(1 / (1 - Real.log α)))
  (h2 : γ = 10^(1 / (1 - Real.log β))) :
  α = 10^(1 / (1 - Real.log γ)) := by
  sorry

end NUMINAMATH_CALUDE_alpha_gamma_relation_l3998_399846


namespace NUMINAMATH_CALUDE_quadratic_no_intersection_l3998_399867

/-- A quadratic function that doesn't intersect the x-axis has c > 1 -/
theorem quadratic_no_intersection (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + c ≠ 0) → c > 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_intersection_l3998_399867


namespace NUMINAMATH_CALUDE_warehouse_capacity_is_510_l3998_399857

/-- The total capacity of a grain-storage warehouse --/
def warehouse_capacity (total_bins : ℕ) (large_bins : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) : ℕ :=
  large_bins * large_capacity + (total_bins - large_bins) * small_capacity

/-- Theorem: The warehouse capacity is 510 tons --/
theorem warehouse_capacity_is_510 :
  warehouse_capacity 30 12 20 15 = 510 :=
by sorry

end NUMINAMATH_CALUDE_warehouse_capacity_is_510_l3998_399857


namespace NUMINAMATH_CALUDE_smallest_n_greater_than_threshold_l3998_399833

/-- The first term of the arithmetic sequence -/
def a₁ : ℕ := 11

/-- The common difference of the arithmetic sequence -/
def d : ℕ := 6

/-- The threshold value -/
def threshold : ℕ := 2017

/-- The n-th term of the arithmetic sequence -/
def aₙ (n : ℕ) : ℕ := a₁ + (n - 1) * d

/-- The proposition to be proved -/
theorem smallest_n_greater_than_threshold :
  (∀ k ≥ 336, aₙ k > threshold) ∧
  (∀ m < 336, ∃ l ≥ m, aₙ l ≤ threshold) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_greater_than_threshold_l3998_399833


namespace NUMINAMATH_CALUDE_tourist_journey_times_l3998_399821

-- Define the speeds of the tourists
variable (v1 v2 : ℝ)

-- Define the time (in minutes) it takes the second tourist to travel the distance the first tourist covers in 120 minutes
variable (x : ℝ)

-- Define the total journey times for each tourist
def first_tourist_time : ℝ := 120 + x + 28
def second_tourist_time : ℝ := 60 + x

-- State the theorem
theorem tourist_journey_times 
  (h1 : x * v2 = 120 * v1) -- Distance equality at meeting point
  (h2 : v2 * (x + 60) = 120 * v1 + v1 * (x + 28)) -- Total distance equality
  : first_tourist_time = 220 ∧ second_tourist_time = 132 := by
  sorry


end NUMINAMATH_CALUDE_tourist_journey_times_l3998_399821


namespace NUMINAMATH_CALUDE_evaluate_expression_l3998_399806

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) :
  y * (y - 2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3998_399806


namespace NUMINAMATH_CALUDE_max_value_cos_sin_sum_l3998_399885

theorem max_value_cos_sin_sum :
  ∃ (M : ℝ), M = 5 ∧ ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_sum_l3998_399885


namespace NUMINAMATH_CALUDE_average_salary_feb_to_may_l3998_399850

def average_salary_jan_to_apr : ℝ := 8000
def average_salary_some_months : ℝ := 8800
def salary_may : ℝ := 6500
def salary_jan : ℝ := 3300

theorem average_salary_feb_to_may :
  let total_salary_jan_to_apr := average_salary_jan_to_apr * 4
  let total_salary_feb_to_apr := total_salary_jan_to_apr - salary_jan
  let total_salary_feb_to_may := total_salary_feb_to_apr + salary_may
  total_salary_feb_to_may / 4 = average_salary_some_months :=
by sorry

end NUMINAMATH_CALUDE_average_salary_feb_to_may_l3998_399850


namespace NUMINAMATH_CALUDE_T_properties_l3998_399823

-- Define the set T
def T : Set ℤ := {x | ∃ n : ℤ, x = n^2 + (n+2)^2 + (n+4)^2}

-- Statement to prove
theorem T_properties :
  (∀ x ∈ T, ¬(4 ∣ x)) ∧ (∃ x ∈ T, 13 ∣ x) := by sorry

end NUMINAMATH_CALUDE_T_properties_l3998_399823


namespace NUMINAMATH_CALUDE_geometric_sequence_cubic_root_count_l3998_399831

/-- Given a, b, c form a geometric sequence, the equation ax³ + bx² + cx = 0 has exactly one real root -/
theorem geometric_sequence_cubic_root_count 
  (a b c : ℝ) 
  (h_geom : ∃ (r : ℝ), b = a * r ∧ c = b * r ∧ r ≠ 0) :
  (∃! x : ℝ, a * x^3 + b * x^2 + c * x = 0) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_cubic_root_count_l3998_399831


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l3998_399865

-- Define proposition p
def p : Prop := ∀ (x a : ℝ), x^2 + a*x + a^2 ≥ 0

-- Define proposition q
def q : Prop := ∃ (x₀ : ℕ), x₀ > 0 ∧ 2*x₀^2 - 1 ≤ 0

-- Theorem statement
theorem p_or_q_is_true : p ∨ q := by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l3998_399865


namespace NUMINAMATH_CALUDE_order_of_roots_l3998_399836

theorem order_of_roots (m n p : ℝ) : 
  m = (1/3)^(1/5) → n = (1/4)^(1/3) → p = (1/5)^(1/4) → n < p ∧ p < m := by
  sorry

end NUMINAMATH_CALUDE_order_of_roots_l3998_399836


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3998_399887

theorem max_value_trig_expression :
  ∀ x y z : ℝ,
  (Real.sin (3 * x) + Real.sin (2 * y) + Real.sin z) *
  (Real.cos (3 * x) + Real.cos (2 * y) + Real.cos z) ≤ 4.5 ∧
  ∃ a b c : ℝ,
  (Real.sin (3 * a) + Real.sin (2 * b) + Real.sin c) *
  (Real.cos (3 * a) + Real.cos (2 * b) + Real.cos c) = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3998_399887


namespace NUMINAMATH_CALUDE_equal_interval_points_ratio_l3998_399875

theorem equal_interval_points_ratio : 
  ∀ (s S : ℝ), 
  (∃ d : ℝ, s = 9 * d ∧ S = 99 * d) → 
  S / s = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_interval_points_ratio_l3998_399875


namespace NUMINAMATH_CALUDE_bernoullis_inequality_l3998_399876

theorem bernoullis_inequality (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : n > 0) :
  (1 + x)^n ≥ 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_bernoullis_inequality_l3998_399876


namespace NUMINAMATH_CALUDE_congruence_problem_l3998_399860

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3998_399860


namespace NUMINAMATH_CALUDE_triangle_formation_l3998_399851

/-- A line in 2D space represented by ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines form a triangle --/
def form_triangle (l1 l2 l3 : Line) : Prop :=
  ∃ (x y : ℝ), (l1.a * x + l1.b * y = l1.c) ∧ 
                (l2.a * x + l2.b * y = l2.c) ∧ 
                (l3.a * x + l3.b * y = l3.c)

theorem triangle_formation (m : ℝ) : 
  ¬(form_triangle 
      ⟨1, 1, 2⟩  -- x + y = 2
      ⟨m, 1, 0⟩  -- mx + y = 0
      ⟨1, -1, 4⟩ -- x - y = 4
    ) ↔ m = 1/3 ∨ m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_formation_l3998_399851


namespace NUMINAMATH_CALUDE_mod_9_sum_of_digits_mod_9_sum_mod_9_product_l3998_399819

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Property 1
theorem mod_9_sum_of_digits (n : ℕ) : n % 9 = sumOfDigits n % 9 := by
  sorry

-- Property 2
theorem mod_9_sum (ns : List ℕ) : 
  (ns.sum % 9) = (ns.map (· % 9)).sum % 9 := by
  sorry

-- Property 3
theorem mod_9_product (ns : List ℕ) : 
  (ns.prod % 9) = (ns.map (· % 9)).prod % 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_9_sum_of_digits_mod_9_sum_mod_9_product_l3998_399819


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l3998_399895

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f 2 x < 4} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 7/2} := by sorry

-- Part 2: Range of a when f(x) ≥ 2 for all x
theorem range_of_a :
  (∀ x, f a x ≥ 2) ↔ a ≤ -1 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l3998_399895


namespace NUMINAMATH_CALUDE_cubic_root_cube_relation_l3998_399891

/-- Given a cubic polynomial f(x) = x^3 - 2x^2 + 5x - 3 with three distinct roots,
    and another cubic polynomial g(x) = x^3 + bx^2 + cx + d whose roots are
    the cubes of the roots of f(x), prove that b = -2, c = -5, and d = 3. -/
theorem cubic_root_cube_relation :
  let f (x : ℝ) := x^3 - 2*x^2 + 5*x - 3
  let g (x : ℝ) := x^3 + b*x^2 + c*x + d
  ∀ (b c d : ℝ),
  (∀ r : ℝ, f r = 0 → g (r^3) = 0) →
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  b = -2 ∧ c = -5 ∧ d = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_cube_relation_l3998_399891


namespace NUMINAMATH_CALUDE_find_other_number_l3998_399872

theorem find_other_number (A B : ℕ) (h1 : A = 24) (h2 : Nat.gcd A B = 17) (h3 : Nat.lcm A B = 312) :
  B = 221 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l3998_399872


namespace NUMINAMATH_CALUDE_amusement_park_tickets_l3998_399804

theorem amusement_park_tickets 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_paid : ℕ) 
  (child_tickets : ℕ) : 
  adult_price = 8 → 
  child_price = 5 → 
  total_paid = 201 → 
  child_tickets = 21 → 
  ∃ (adult_tickets : ℕ), 
    adult_price * adult_tickets + child_price * child_tickets = total_paid ∧ 
    adult_tickets + child_tickets = 33 :=
by
  sorry

#check amusement_park_tickets

end NUMINAMATH_CALUDE_amusement_park_tickets_l3998_399804


namespace NUMINAMATH_CALUDE_angle_with_special_supplementary_complementary_relation_l3998_399800

theorem angle_with_special_supplementary_complementary_relation :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 90 →
  (180 - x = 3 * (90 - x)) →
  x = 45 :=
by sorry

end NUMINAMATH_CALUDE_angle_with_special_supplementary_complementary_relation_l3998_399800


namespace NUMINAMATH_CALUDE_olaf_collection_l3998_399859

def total_cars (initial : ℕ) (uncle : ℕ) : ℕ :=
  let grandpa := 2 * uncle
  let dad := 10
  let mum := dad + 5
  let auntie := 6
  let cousin_liam := dad / 2
  let cousin_emma := uncle / 3
  let grandmother := 3 * auntie
  initial + grandpa + dad + mum + auntie + uncle + cousin_liam + cousin_emma + grandmother

theorem olaf_collection (initial : ℕ) (uncle : ℕ) 
  (h1 : initial = 150)
  (h2 : uncle = 5)
  (h3 : auntie = uncle + 1) :
  total_cars initial uncle = 220 := by
  sorry

#eval total_cars 150 5

end NUMINAMATH_CALUDE_olaf_collection_l3998_399859


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3998_399884

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - 2*x)}
def N : Set ℝ := {y | ∃ x, y = Real.exp x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 0 < x ∧ x < 1/2} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3998_399884


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l3998_399817

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The squared distance between two points in a 2D plane --/
def squaredDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The squared distance between the intersection points of two specific circles --/
theorem intersection_distance_squared (c1 c2 : Circle) 
  (h1 : c1 = ⟨(3, -2), 5⟩) 
  (h2 : c2 = ⟨(3, 4), 3⟩) : 
  ∃ (p1 p2 : ℝ × ℝ), 
    squaredDistance p1 c1.center = c1.radius^2 ∧ 
    squaredDistance p1 c2.center = c2.radius^2 ∧
    squaredDistance p2 c1.center = c1.radius^2 ∧ 
    squaredDistance p2 c2.center = c2.radius^2 ∧
    squaredDistance p1 p2 = 224/9 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l3998_399817


namespace NUMINAMATH_CALUDE_intersection_A_B_l3998_399889

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | x^2 + 3*x - 4 < 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3998_399889


namespace NUMINAMATH_CALUDE_max_lessons_l3998_399897

/-- Represents the number of clothing items the instructor has -/
structure Wardrobe where
  shirts : ℕ
  trousers : ℕ
  shoes : ℕ
  jackets : ℕ
  jackets_eq_two : jackets = 2

/-- Calculates the number of possible lesson outfits -/
def lesson_outfits (w : Wardrobe) : ℕ := 3 * w.shirts * w.trousers * w.shoes

/-- Represents the additional lessons possible with one more item -/
structure AdditionalLessons where
  with_shirt : ℕ
  with_trousers : ℕ
  with_shoes : ℕ
  shirt_eq : with_shirt = 18
  trousers_eq : with_trousers = 63
  shoes_eq : with_shoes = 42

theorem max_lessons (w : Wardrobe) (a : AdditionalLessons) :
  lesson_outfits w = 126 := by
  sorry

end NUMINAMATH_CALUDE_max_lessons_l3998_399897


namespace NUMINAMATH_CALUDE_classic_rock_collections_l3998_399816

/-- The number of albums in either Andrew's or Bob's collection, but not both -/
def albums_not_shared (andrew_total : ℕ) (bob_not_andrew : ℕ) (shared : ℕ) : ℕ :=
  (andrew_total - shared) + bob_not_andrew

theorem classic_rock_collections :
  let andrew_total := 20
  let bob_not_andrew := 8
  let shared := 11
  albums_not_shared andrew_total bob_not_andrew shared = 17 := by sorry

end NUMINAMATH_CALUDE_classic_rock_collections_l3998_399816


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l3998_399847

theorem isosceles_triangle_condition (A B C : Real) :
  (A > 0) → (B > 0) → (C > 0) → (A + B + C = π) →
  (Real.log (Real.sin A) - Real.log (Real.cos B) - Real.log (Real.sin C) = Real.log 2) →
  ∃ (x y : Real), (x = y) ∧ 
  ((A = x ∧ B = y ∧ C = y) ∨ (A = y ∧ B = x ∧ C = y) ∨ (A = y ∧ B = y ∧ C = x)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l3998_399847


namespace NUMINAMATH_CALUDE_is_projection_matrix_l3998_399866

def projection_matrix (A : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * A = A

theorem is_projection_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![20/49, 20/49; 20/49, 29/49]
  projection_matrix A := by
  sorry

end NUMINAMATH_CALUDE_is_projection_matrix_l3998_399866


namespace NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l3998_399801

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, 5)

theorem reflection_of_P_across_x_axis :
  reflect_x P = (-2, -5) := by sorry

end NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l3998_399801


namespace NUMINAMATH_CALUDE_power_three_thirds_of_675_l3998_399873

theorem power_three_thirds_of_675 : (675 : ℝ) ^ (3/3) = 675 := by
  sorry

end NUMINAMATH_CALUDE_power_three_thirds_of_675_l3998_399873
