import Mathlib

namespace NUMINAMATH_CALUDE_max_value_plus_cos_squared_l1258_125896

theorem max_value_plus_cos_squared (x : ℝ) (M : ℝ) : 
  0 ≤ x → x ≤ π / 2 → 
  (∀ y, 0 ≤ y ∧ y ≤ π / 2 → 
    3 * Real.sin y ^ 2 + 8 * Real.sin y * Real.cos y + 9 * Real.cos y ^ 2 ≤ M) →
  (3 * Real.sin x ^ 2 + 8 * Real.sin x * Real.cos x + 9 * Real.cos x ^ 2 = M) →
  M + 100 * Real.cos x ^ 2 = 91 := by
sorry

end NUMINAMATH_CALUDE_max_value_plus_cos_squared_l1258_125896


namespace NUMINAMATH_CALUDE_pig_price_calculation_l1258_125801

theorem pig_price_calculation (num_cows : ℕ) (num_pigs : ℕ) (price_per_cow : ℕ) (total_revenue : ℕ) :
  num_cows = 20 →
  num_pigs = 4 * num_cows →
  price_per_cow = 800 →
  total_revenue = 48000 →
  (total_revenue - num_cows * price_per_cow) / num_pigs = 400 := by
  sorry

end NUMINAMATH_CALUDE_pig_price_calculation_l1258_125801


namespace NUMINAMATH_CALUDE_unique_numbers_l1258_125892

/-- Checks if a three-digit number has distinct digits in ascending order -/
def has_ascending_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a < b ∧ b < c

/-- Checks if a three-digit number has identical digits -/
def has_identical_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a : ℕ), n = 100 * a + 10 * a + a

/-- Checks if all words in the name of a number start with the same letter -/
def name_starts_same_letter (n : ℕ) : Prop :=
  -- This is a placeholder for the actual condition
  n = 147

/-- Checks if all words in the name of a number start with different letters -/
def name_starts_different_letters (n : ℕ) : Prop :=
  -- This is a placeholder for the actual condition
  n = 111

theorem unique_numbers :
  (∃! n : ℕ, has_ascending_digits n ∧ name_starts_same_letter n) ∧
  (∃! n : ℕ, has_identical_digits n ∧ name_starts_different_letters n) :=
sorry

end NUMINAMATH_CALUDE_unique_numbers_l1258_125892


namespace NUMINAMATH_CALUDE_sandwich_combinations_count_l1258_125876

/-- Represents the number of toppings available -/
def num_toppings : ℕ := 7

/-- Represents the number of bread types available -/
def num_bread_types : ℕ := 3

/-- Represents the number of filling types available -/
def num_filling_types : ℕ := 3

/-- Represents the maximum number of filling layers -/
def max_filling_layers : ℕ := 2

/-- Calculates the total number of sandwich combinations -/
def total_sandwich_combinations : ℕ :=
  (2^num_toppings) * num_bread_types * (num_filling_types + num_filling_types^2)

/-- Theorem stating that the total number of sandwich combinations is 4608 -/
theorem sandwich_combinations_count :
  total_sandwich_combinations = 4608 := by sorry

end NUMINAMATH_CALUDE_sandwich_combinations_count_l1258_125876


namespace NUMINAMATH_CALUDE_point_on_inverse_graph_and_coordinate_sum_l1258_125806

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Theorem statement
theorem point_on_inverse_graph_and_coordinate_sum 
  (h : f 3 = 5/3) : 
  (f_inv (5/3) = 3) ∧ 
  ((1/3) * (f_inv (5/3)) = 1) ∧ 
  (5/3 + 1 = 8/3) := by
sorry

end NUMINAMATH_CALUDE_point_on_inverse_graph_and_coordinate_sum_l1258_125806


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1258_125840

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_eq : a 1 + 3 * a 8 = 1560) :
  2 * a 9 - a 10 = 507 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1258_125840


namespace NUMINAMATH_CALUDE_range_of_a_l1258_125829

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (h_xy : x + y + 8 = x * y)
  (h_ineq : ∀ x y : ℝ, x > 0 → y > 0 → x + y + 8 = x * y → (x + y)^2 - a*(x + y) + 1 ≥ 0) :
  a ≤ 65/8 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1258_125829


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l1258_125818

def red_lamps : ℕ := 4
def blue_lamps : ℕ := 3
def green_lamps : ℕ := 3
def total_lamps : ℕ := red_lamps + blue_lamps + green_lamps
def lamps_turned_on : ℕ := 5

def probability_leftmost_green_off_second_right_blue_on : ℚ := 63 / 100

theorem lava_lamp_probability :
  let total_arrangements := Nat.choose total_lamps red_lamps * Nat.choose (total_lamps - red_lamps) blue_lamps
  let leftmost_green_arrangements := Nat.choose (total_lamps - 1) (green_lamps - 1) * Nat.choose (total_lamps - green_lamps) red_lamps * Nat.choose (total_lamps - green_lamps - red_lamps) (blue_lamps - 1)
  let second_right_blue_on_arrangements := Nat.choose (total_lamps - 2) (blue_lamps - 1)
  let remaining_on_lamps := Nat.choose (total_lamps - 2) (lamps_turned_on - 1)
  (leftmost_green_arrangements * second_right_blue_on_arrangements * remaining_on_lamps : ℚ) / (total_arrangements * Nat.choose total_lamps lamps_turned_on) = probability_leftmost_green_off_second_right_blue_on :=
by sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l1258_125818


namespace NUMINAMATH_CALUDE_largest_last_digit_is_two_l1258_125871

/-- A string of digits satisfying the given conditions -/
structure SpecialString :=
  (digits : Fin 1003 → Nat)
  (first_digit : digits 0 = 2)
  (consecutive_divisible : ∀ i : Fin 1002, 
    (digits i * 10 + digits (i.succ)) % 17 = 0 ∨ 
    (digits i * 10 + digits (i.succ)) % 23 = 0)

/-- The largest possible last digit in the special string -/
def largest_last_digit : Nat := 2

/-- Theorem stating that the largest possible last digit is 2 -/
theorem largest_last_digit_is_two :
  ∀ s : SpecialString, s.digits 1002 ≤ largest_last_digit :=
sorry

end NUMINAMATH_CALUDE_largest_last_digit_is_two_l1258_125871


namespace NUMINAMATH_CALUDE_circle_M_properties_l1258_125804

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 4

-- Define the line that contains the center of the circle
def center_line (x y : ℝ) : Prop :=
  x + y = 2

-- Define the points C and D
def point_C : ℝ × ℝ := (1, -1)
def point_D : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem circle_M_properties :
  (∀ x y, circle_M x y → center_line x y) ∧
  circle_M point_C.1 point_C.2 ∧
  circle_M point_D.1 point_D.2 ∧
  (∀ x y, circle_M x y → 2 - 2 * Real.sqrt 2 ≤ x + y ∧ x + y ≤ 2 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_circle_M_properties_l1258_125804


namespace NUMINAMATH_CALUDE_triangle_sum_of_squares_l1258_125861

-- Define an equilateral triangle ABC with side length 10
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 10 ∧ dist B C = 10 ∧ dist C A = 10

-- Define points P and Q on AB and AC respectively
def PointP (A B P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B ∧ dist A P = 2

def PointQ (A C Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • A + t • C ∧ dist A Q = 2

-- Theorem statement
theorem triangle_sum_of_squares 
  (A B C P Q : ℝ × ℝ) 
  (h_triangle : Triangle A B C) 
  (h_P : PointP A B P) 
  (h_Q : PointQ A C Q) : 
  (dist C P)^2 + (dist C Q)^2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_of_squares_l1258_125861


namespace NUMINAMATH_CALUDE_fish_pond_population_l1258_125897

-- Define the parameters
def initial_tagged : ℕ := 40
def second_catch : ℕ := 50
def tagged_in_second : ℕ := 2

-- Define the theorem
theorem fish_pond_population :
  let total_fish : ℕ := (initial_tagged * second_catch) / tagged_in_second
  total_fish = 1000 := by
  sorry

end NUMINAMATH_CALUDE_fish_pond_population_l1258_125897


namespace NUMINAMATH_CALUDE_binomial_distribution_parameters_l1258_125862

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_parameters 
  (X : BinomialDistribution) 
  (h_expectation : expectation X = 2)
  (h_variance : variance X = 4) :
  X.n = 12 ∧ X.p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_parameters_l1258_125862


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1258_125857

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  ∃ (max_val : ℝ), (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 - 2*x'*y' + 3*y'^2 = 10 
    → x'^2 + 2*x'*y' + 3*y'^2 ≤ max_val) ∧ max_val = 20 + 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1258_125857


namespace NUMINAMATH_CALUDE_container_volume_ratio_l1258_125856

theorem container_volume_ratio :
  ∀ (v1 v2 : ℚ),
  v1 > 0 → v2 > 0 →
  (5 / 6 : ℚ) * v1 = (3 / 4 : ℚ) * v2 →
  v1 / v2 = (9 / 10 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l1258_125856


namespace NUMINAMATH_CALUDE_remaining_rolls_to_sell_l1258_125898

/-- Calculates the remaining rolls of gift wrap Nellie needs to sell -/
theorem remaining_rolls_to_sell 
  (total_rolls : ℕ) 
  (sold_to_grandmother : ℕ) 
  (sold_to_uncle : ℕ) 
  (sold_to_neighbor : ℕ) 
  (h1 : total_rolls = 45)
  (h2 : sold_to_grandmother = 1)
  (h3 : sold_to_uncle = 10)
  (h4 : sold_to_neighbor = 6) :
  total_rolls - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end NUMINAMATH_CALUDE_remaining_rolls_to_sell_l1258_125898


namespace NUMINAMATH_CALUDE_distinct_collections_count_l1258_125894

/-- Represents the number of each letter in BIOLOGY --/
structure LetterCount where
  o : Nat
  i : Nat
  y : Nat
  b : Nat
  g : Nat

/-- The initial count of letters in BIOLOGY --/
def initial_count : LetterCount :=
  { o := 2, i := 1, y := 1, b := 1, g := 2 }

/-- A collection of letters that can be put in the bag --/
structure BagCollection where
  vowels : Nat
  consonants : Nat

/-- Check if a collection is valid (3 vowels and 2 consonants) --/
def is_valid_collection (c : BagCollection) : Prop :=
  c.vowels = 3 ∧ c.consonants = 2

/-- Count the number of distinct vowel combinations --/
def count_vowel_combinations (lc : LetterCount) : Nat :=
  sorry

/-- Count the number of distinct consonant combinations --/
def count_consonant_combinations (lc : LetterCount) : Nat :=
  sorry

/-- The main theorem: there are 12 distinct possible collections --/
theorem distinct_collections_count :
  count_vowel_combinations initial_count * count_consonant_combinations initial_count = 12 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l1258_125894


namespace NUMINAMATH_CALUDE_reading_time_difference_l1258_125821

/-- The reading problem setup -/
structure ReadingProblem where
  xanthia_rate : ℕ  -- pages per hour
  molly_rate : ℕ    -- pages per hour
  book_pages : ℕ
  
/-- Calculate the time difference in minutes -/
def time_difference (p : ReadingProblem) : ℕ :=
  ((p.book_pages / p.molly_rate - p.book_pages / p.xanthia_rate) * 60 : ℕ)

/-- The main theorem -/
theorem reading_time_difference (p : ReadingProblem) 
  (h1 : p.xanthia_rate = 120)
  (h2 : p.molly_rate = 60)
  (h3 : p.book_pages = 360) : 
  time_difference p = 180 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l1258_125821


namespace NUMINAMATH_CALUDE_parabola_vertex_m_value_l1258_125808

theorem parabola_vertex_m_value (m : ℝ) :
  let f (x : ℝ) := 3 * x^2 + 6 * Real.sqrt m * x + 36
  let vertex_y := f (-(Real.sqrt m) / 3)
  vertex_y = 33 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_m_value_l1258_125808


namespace NUMINAMATH_CALUDE_shaded_area_division_l1258_125889

/-- Represents a grid in the first quadrant -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)
  (shaded_squares : ℕ)

/-- Represents a line passing through (0,0) and (8,c) -/
structure Line :=
  (c : ℝ)

/-- Checks if a line divides the shaded area of a grid into two equal parts -/
def divides_equally (g : Grid) (l : Line) : Prop :=
  ∃ (area : ℝ), area > 0 ∧ area * 2 = g.shaded_squares

theorem shaded_area_division (g : Grid) (l : Line) :
  g.width = 8 ∧ g.height = 6 ∧ g.shaded_squares = 32 →
  divides_equally g l ↔ l.c = 4 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_division_l1258_125889


namespace NUMINAMATH_CALUDE_enclosed_area_equals_four_l1258_125878

-- Define the functions for the line and curve
def f (x : ℝ) : ℝ := 4 * x
def g (x : ℝ) : ℝ := x^3

-- Define the intersection points
def x₁ : ℝ := 0
def x₂ : ℝ := 2

-- State the theorem
theorem enclosed_area_equals_four :
  (∫ x in x₁..x₂, f x - g x) = 4 := by sorry

end NUMINAMATH_CALUDE_enclosed_area_equals_four_l1258_125878


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1258_125846

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1258_125846


namespace NUMINAMATH_CALUDE_total_games_in_season_l1258_125832

def total_teams : ℕ := 200
def num_sub_leagues : ℕ := 10
def teams_per_sub_league : ℕ := 20
def regular_season_matches : ℕ := 8
def teams_to_intermediate : ℕ := 5
def teams_to_playoff : ℕ := 2

def regular_season_games (n : ℕ) : ℕ := n * (n - 1) / 2 * regular_season_matches

def intermediate_round_games (n : ℕ) : ℕ := n * (n - 1) / 2

def playoff_round_games (n : ℕ) : ℕ := (n * (n - 1) / 2 - num_sub_leagues * (num_sub_leagues - 1) / 2) * 2

theorem total_games_in_season :
  regular_season_games teams_per_sub_league * num_sub_leagues +
  intermediate_round_games (teams_to_intermediate * num_sub_leagues) +
  playoff_round_games (teams_to_playoff * num_sub_leagues) = 16715 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_season_l1258_125832


namespace NUMINAMATH_CALUDE_intersection_M_N_l1258_125873

open Set

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complementN : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := U \ complementN

-- Theorem statement
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1258_125873


namespace NUMINAMATH_CALUDE_simplify_expression_l1258_125819

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 256) - Real.sqrt (13/2))^2 = (45 - 8 * Real.sqrt 26) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1258_125819


namespace NUMINAMATH_CALUDE_circle_E_equation_l1258_125872

-- Define the circle E
def circle_E (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the condition that E passes through A(0,0) and B(1,1)
def passes_through_A_and_B (E : Set (ℝ × ℝ)) : Prop :=
  (0, 0) ∈ E ∧ (1, 1) ∈ E

-- Define the three additional conditions
def condition_1 (E : Set (ℝ × ℝ)) : Prop :=
  (2, 0) ∈ E

def condition_2 (E : Set (ℝ × ℝ)) : Prop :=
  ∀ m : ℝ, ∃ p q : ℝ × ℝ, p ∈ E ∧ q ∈ E ∧
    p.2 = m * (p.1 - 1) ∧ q.2 = m * (q.1 - 1) ∧
    p ≠ q

def condition_3 (E : Set (ℝ × ℝ)) : Prop :=
  ∃ y : ℝ, (0, y) ∈ E ∧ ∀ t : ℝ, t ≠ y → (0, t) ∉ E

-- The main theorem
theorem circle_E_equation :
  ∀ E : Set (ℝ × ℝ),
  passes_through_A_and_B E →
  (condition_1 E ∨ condition_2 E ∨ condition_3 E) →
  E = circle_E (1, 0) 1 :=
sorry

end NUMINAMATH_CALUDE_circle_E_equation_l1258_125872


namespace NUMINAMATH_CALUDE_cookies_per_bag_l1258_125814

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 703) (h2 : num_bags = 37) :
  total_cookies / num_bags = 19 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l1258_125814


namespace NUMINAMATH_CALUDE_identity_proof_l1258_125826

theorem identity_proof (a b c : ℝ) 
  (h1 : (a - c) / (a + c) ≠ 0)
  (h2 : (b - c) / (b + c) ≠ 0)
  (h3 : (a + c) / (a - c) + (b + c) / (b - c) ≠ 0) :
  ((((a - c) / (a + c) + (b - c) / (b + c)) / ((a + c) / (a - c) + (b + c) / (b - c))) ^ 2) = 
  ((((a - c) / (a + c)) ^ 2 + ((b - c) / (b + c)) ^ 2) / (((a + c) / (a - c)) ^ 2 + ((b + c) / (b - c)) ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_identity_proof_l1258_125826


namespace NUMINAMATH_CALUDE_quadratic_functions_equality_l1258_125838

/-- Given a quadratic function f(x) = x² + bx + 8 with b ≠ 0 and two distinct real roots x₁ and x₂,
    and a quadratic function g(x) with quadratic coefficient 1 and roots x₁ + 1/x₂ and x₂ + 1/x₁,
    prove that if g(1) = f(1), then g(1) = -8. -/
theorem quadratic_functions_equality (b : ℝ) (x₁ x₂ : ℝ) :
  b ≠ 0 →
  x₁ ≠ x₂ →
  (∀ x, x^2 + b*x + 8 = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ c d : ℝ, ∀ x, (x - (x₁ + 1/x₂)) * (x - (x₂ + 1/x₁)) = x^2 + c*x + d) →
  (1^2 + b*1 + 8 = 1^2 + c*1 + d) →
  1^2 + c*1 + d = -8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_functions_equality_l1258_125838


namespace NUMINAMATH_CALUDE_rectangle_area_l1258_125813

-- Define the radius of the inscribed circle
def circle_radius : ℝ := 7

-- Define the ratio of length to width
def length_width_ratio : ℝ := 2

-- Theorem statement
theorem rectangle_area (width : ℝ) (length : ℝ) 
  (h1 : width = 2 * circle_radius) 
  (h2 : length = length_width_ratio * width) : 
  width * length = 392 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_l1258_125813


namespace NUMINAMATH_CALUDE_project_completion_theorem_l1258_125809

theorem project_completion_theorem (a b c x y z : ℝ) 
  (ha : a / x = 1 / y + 1 / z)
  (hb : b / y = 1 / x + 1 / z)
  (hc : c / z = 1 / x + 1 / y)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 1 := by
sorry


end NUMINAMATH_CALUDE_project_completion_theorem_l1258_125809


namespace NUMINAMATH_CALUDE_derivative_at_one_l1258_125881

/-- Given a differentiable function f: ℝ → ℝ where x > 0, 
    if f(x) = 2e^x * f'(1) + 3ln(x), then f'(1) = 3 / (1 - 2e) -/
theorem derivative_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : ∀ x > 0, f x = 2 * Real.exp x * deriv f 1 + 3 * Real.log x) : 
  deriv f 1 = 3 / (1 - 2 * Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1258_125881


namespace NUMINAMATH_CALUDE_inner_circle_radius_l1258_125827

/-- Given a circle of radius R and a point A on its diameter at distance a from the center,
    the radius of the circle that touches the diameter at A and is internally tangent to the given circle
    is (R^2 - a^2) / (2R). -/
theorem inner_circle_radius (R a : ℝ) (h₁ : R > 0) (h₂ : 0 < a ∧ a < R) :
  ∃ x : ℝ, x > 0 ∧ x = (R^2 - a^2) / (2*R) ∧
  x^2 + a^2 = (R - x)^2 :=
sorry

end NUMINAMATH_CALUDE_inner_circle_radius_l1258_125827


namespace NUMINAMATH_CALUDE_hotdog_cost_l1258_125895

/-- The total cost of hot dogs given the number of hot dogs and the price per hot dog. -/
def total_cost (num_hotdogs : ℕ) (price_per_hotdog : ℚ) : ℚ :=
  num_hotdogs * price_per_hotdog

/-- Theorem stating that the total cost of 6 hot dogs at 50 cents each is $3.00 -/
theorem hotdog_cost : total_cost 6 (50 / 100) = 3 := by
  sorry

end NUMINAMATH_CALUDE_hotdog_cost_l1258_125895


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1258_125835

theorem smallest_solution_of_equation (x : ℝ) :
  x^4 - 64*x^2 + 576 = 0 →
  x ≥ -2 * Real.sqrt 6 ∧
  (∃ y : ℝ, y^4 - 64*y^2 + 576 = 0 ∧ y = -2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1258_125835


namespace NUMINAMATH_CALUDE_arc_length_300_degrees_l1258_125820

/-- The length of an arc in a circle with radius 2 and central angle 300° is 10π/3 -/
theorem arc_length_300_degrees (r : ℝ) (θ : ℝ) : 
  r = 2 → θ = 300 * π / 180 → r * θ = 10 * π / 3 := by sorry

end NUMINAMATH_CALUDE_arc_length_300_degrees_l1258_125820


namespace NUMINAMATH_CALUDE_xiao_ming_arrival_time_l1258_125888

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : minutes < 60 := by sorry

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60
  , minutes := totalMinutes % 60 }

theorem xiao_ming_arrival_time 
  (departure_time : Time)
  (journey_duration : Nat)
  (h1 : departure_time.hours = 6)
  (h2 : departure_time.minutes = 55)
  (h3 : journey_duration = 30) :
  addMinutes departure_time journey_duration = { hours := 7, minutes := 25 } := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_arrival_time_l1258_125888


namespace NUMINAMATH_CALUDE_parabola_max_vertex_sum_l1258_125852

theorem parabola_max_vertex_sum (a T : ℤ) (h_T : T ≠ 0) : 
  let parabola (x y : ℝ) := ∃ b c : ℝ, y = a * x^2 + b * x + c
  let passes_through (x y : ℝ) := parabola x y
  let vertex_sum := 
    let h : ℝ := T
    let k : ℝ := -a * T^2
    h + k
  (passes_through 0 0) ∧ 
  (passes_through (2 * T) 0) ∧ 
  (passes_through (T + 2) 32) →
  (∀ N : ℝ, N = vertex_sum → N ≤ 68) ∧ 
  (∃ N : ℝ, N = vertex_sum ∧ N = 68) :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_vertex_sum_l1258_125852


namespace NUMINAMATH_CALUDE_baker_sales_difference_l1258_125815

theorem baker_sales_difference (cakes_made pastries_made cakes_sold pastries_sold : ℕ) :
  cakes_made = 157 →
  pastries_made = 169 →
  cakes_sold = 158 →
  pastries_sold = 147 →
  cakes_sold - pastries_sold = 11 := by
sorry

end NUMINAMATH_CALUDE_baker_sales_difference_l1258_125815


namespace NUMINAMATH_CALUDE_tourists_escape_theorem_l1258_125870

/-- Represents the color of a hat -/
inductive HatColor
  | Black
  | White

/-- Represents a tourist in the line -/
structure Tourist where
  position : Nat
  hatColor : HatColor

/-- Represents the line of tourists -/
def TouristLine := List Tourist

/-- A strategy is a function that takes the visible hats and previous guesses
    and returns a guess for the current tourist's hat color -/
def Strategy := (visibleHats : List HatColor) → (previousGuesses : List HatColor) → HatColor

/-- Applies the strategy to a line of tourists and returns the number of correct guesses -/
def applyStrategy (line : TouristLine) (strategy : Strategy) : Nat :=
  sorry

/-- There exists a strategy that guarantees at least 9 out of 10 tourists can correctly guess their hat color -/
theorem tourists_escape_theorem :
  ∃ (strategy : Strategy),
    ∀ (line : TouristLine),
      line.length = 10 →
      applyStrategy line strategy ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_tourists_escape_theorem_l1258_125870


namespace NUMINAMATH_CALUDE_problem_solution_l1258_125834

theorem problem_solution (x y : ℝ) 
  (hx : 2 < x ∧ x < 3) 
  (hy : -2 < y ∧ y < -1) 
  (hxy : x < y ∧ y < 0) : 
  (0 < x + y ∧ x + y < 2) ∧ 
  (3 < x - y ∧ x - y < 5) ∧ 
  (-6 < x * y ∧ x * y < -2) ∧
  (x^2 + y^2) * (x - y) > (x^2 - y^2) * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1258_125834


namespace NUMINAMATH_CALUDE_class_size_proof_l1258_125877

theorem class_size_proof :
  ∃ (n : ℕ), 
    n > 0 ∧
    (n / 2 : ℕ) > 0 ∧
    (n / 4 : ℕ) > 0 ∧
    (n / 7 : ℕ) > 0 ∧
    n - (n / 2) - (n / 4) - (n / 7) < 6 ∧
    n = 28 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l1258_125877


namespace NUMINAMATH_CALUDE_math_book_cost_l1258_125879

theorem math_book_cost (total_books : ℕ) (math_books : ℕ) (history_book_cost : ℕ) (total_cost : ℕ) :
  total_books = 80 →
  math_books = 32 →
  history_book_cost = 5 →
  total_cost = 368 →
  ∃ (math_book_cost : ℕ),
    math_book_cost * math_books + (total_books - math_books) * history_book_cost = total_cost ∧
    math_book_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_math_book_cost_l1258_125879


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1258_125847

theorem min_value_trig_expression (α β : ℝ) : 
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1258_125847


namespace NUMINAMATH_CALUDE_cubic_term_of_line_l1258_125807

-- Define the line equation
def line_equation (x : ℝ) : ℝ := x^2 - x^3

-- State the theorem
theorem cubic_term_of_line : 
  ∃ (a b c d : ℝ), 
    (∀ x, line_equation x = a*x^3 + b*x^2 + c*x + d) ∧ 
    (a = -1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_term_of_line_l1258_125807


namespace NUMINAMATH_CALUDE_triangle_area_is_16_l1258_125874

/-- The area of a triangle formed by three lines in a 2D plane --/
def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ :=
  sorry

/-- The first line: y = 6 --/
def line1 (x : ℝ) : ℝ := 6

/-- The second line: y = 2 + x --/
def line2 (x : ℝ) : ℝ := 2 + x

/-- The third line: y = 2 - x --/
def line3 (x : ℝ) : ℝ := 2 - x

theorem triangle_area_is_16 : triangleArea line1 line2 line3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_16_l1258_125874


namespace NUMINAMATH_CALUDE_friend_payment_is_five_l1258_125839

/-- The cost per person when splitting a restaurant bill -/
def cost_per_person (num_friends : ℕ) (hamburger_price : ℚ) (num_hamburgers : ℕ)
  (fries_price : ℚ) (num_fries : ℕ) (soda_price : ℚ) (num_sodas : ℕ)
  (spaghetti_price : ℚ) (num_spaghetti : ℕ) : ℚ :=
  (hamburger_price * num_hamburgers + fries_price * num_fries +
   soda_price * num_sodas + spaghetti_price * num_spaghetti) / num_friends

/-- Theorem: Each friend pays $5 when splitting the bill equally -/
theorem friend_payment_is_five :
  cost_per_person 5 3 5 (6/5) 4 (1/2) 5 (27/10) 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_friend_payment_is_five_l1258_125839


namespace NUMINAMATH_CALUDE_sum_of_integers_l1258_125830

theorem sum_of_integers (a b : ℕ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) 
  (h4 : a^2 - b^2 = 2018 - 2*a) : a + b = 672 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1258_125830


namespace NUMINAMATH_CALUDE_system_solutions_correct_l1258_125851

theorem system_solutions_correct : 
  (∃ (x y : ℝ), 3*x - y = -1 ∧ x + 2*y = 9 ∧ x = 1 ∧ y = 4) ∧
  (∃ (x y : ℝ), x/4 + y/3 = 4/3 ∧ 5*(x - 9) = 4*(y - 13/4) ∧ x = 6 ∧ y = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l1258_125851


namespace NUMINAMATH_CALUDE_nap_start_time_l1258_125860

def minutes_past_midnight (hours minutes : ℕ) : ℕ :=
  hours * 60 + minutes

def time_from_minutes (total_minutes : ℕ) : ℕ × ℕ :=
  (total_minutes / 60, total_minutes % 60)

theorem nap_start_time 
  (nap_duration : ℕ) 
  (wake_up_hours wake_up_minutes : ℕ) 
  (h1 : nap_duration = 65)
  (h2 : wake_up_hours = 13)
  (h3 : wake_up_minutes = 30) :
  time_from_minutes (minutes_past_midnight wake_up_hours wake_up_minutes - nap_duration) = (12, 25) := by
  sorry

end NUMINAMATH_CALUDE_nap_start_time_l1258_125860


namespace NUMINAMATH_CALUDE_no_prime_with_perfect_square_131_base_l1258_125805

theorem no_prime_with_perfect_square_131_base : ¬∃ n : ℕ, 
  (5 ≤ n ∧ n ≤ 15) ∧ 
  Nat.Prime n ∧ 
  ∃ m : ℕ, n^2 + 3*n + 1 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_with_perfect_square_131_base_l1258_125805


namespace NUMINAMATH_CALUDE_equation_solution_l1258_125875

theorem equation_solution (x : ℝ) (h : x ≠ 3) :
  (x^2 - 9) / (x - 3) = 3 * x ↔ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1258_125875


namespace NUMINAMATH_CALUDE_min_value_quadratic_function_l1258_125884

/-- Given a quadratic function f(x) = ax² - 4x + c with range [0, +∞),
    prove that the minimum value of 1/c + 9/a is 3 -/
theorem min_value_quadratic_function (a c : ℝ) (h_pos_a : a > 0) (h_pos_c : c > 0)
  (h_range : Set.range (fun x => a * x^2 - 4*x + c) = Set.Ici 0)
  (h_ac : a * c = 4) :
  (∀ y, 1/c + 9/a ≥ y) ∧ (∃ y, 1/c + 9/a = y) ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_function_l1258_125884


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_229_l1258_125843

theorem inverse_of_3_mod_229 : ∃ x : ℕ, x < 229 ∧ (3 * x) % 229 = 1 :=
  by
    use 153
    sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_229_l1258_125843


namespace NUMINAMATH_CALUDE_waiter_tips_fraction_l1258_125864

theorem waiter_tips_fraction (base_salary : ℚ) : 
  let tips := (5 / 4) * base_salary
  let total_income := base_salary + tips
  let expenses := (1 / 8) * base_salary
  let taxes := (1 / 5) * total_income
  let after_tax_income := total_income - taxes
  (tips / after_tax_income) = 25 / 36 :=
by sorry

end NUMINAMATH_CALUDE_waiter_tips_fraction_l1258_125864


namespace NUMINAMATH_CALUDE_peter_train_probability_l1258_125845

theorem peter_train_probability (p : ℚ) (h : p = 5/12) : 1 - p = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_peter_train_probability_l1258_125845


namespace NUMINAMATH_CALUDE_chlorine_cost_l1258_125842

-- Define the pool dimensions
def pool_length : ℝ := 10
def pool_width : ℝ := 8
def pool_depth : ℝ := 6

-- Define the chlorine requirement
def cubic_feet_per_quart : ℝ := 120

-- Define the cost of chlorine
def cost_per_quart : ℝ := 3

-- Theorem statement
theorem chlorine_cost : 
  let pool_volume : ℝ := pool_length * pool_width * pool_depth
  let quarts_needed : ℝ := pool_volume / cubic_feet_per_quart
  let total_cost : ℝ := quarts_needed * cost_per_quart
  total_cost = 12 := by sorry

end NUMINAMATH_CALUDE_chlorine_cost_l1258_125842


namespace NUMINAMATH_CALUDE_circle_center_sum_l1258_125816

/-- Given a circle with equation x^2 + y^2 = 6x + 8y + 2, 
    the sum of the coordinates of its center is 7. -/
theorem circle_center_sum : ∃ (h k : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 6*x + 8*y + 2 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 2)) ∧
  h + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1258_125816


namespace NUMINAMATH_CALUDE_oil_price_reduction_l1258_125811

/-- Proves that a 25% reduction in oil price resulting in 5 kg more for Rs. 900 leads to a reduced price of Rs. 45 per kg -/
theorem oil_price_reduction (original_price : ℝ) (original_quantity : ℝ) : 
  (original_quantity * original_price = 900) →
  ((original_quantity + 5) * (0.75 * original_price) = 900) →
  (0.75 * original_price = 45) :=
by sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l1258_125811


namespace NUMINAMATH_CALUDE_obesity_probability_l1258_125848

theorem obesity_probability (P_obese_male P_obese_female : ℝ) 
  (ratio_male_female : ℚ) :
  P_obese_male = 1/5 →
  P_obese_female = 1/10 →
  ratio_male_female = 3/2 →
  let P_male := ratio_male_female / (1 + ratio_male_female)
  let P_female := 1 - P_male
  let P_obese := P_male * P_obese_male + P_female * P_obese_female
  (P_male * P_obese_male) / P_obese = 3/4 := by
sorry

end NUMINAMATH_CALUDE_obesity_probability_l1258_125848


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l1258_125812

theorem salary_reduction_percentage (x : ℝ) : 
  (100 - x) * (1 + 53.84615384615385 / 100) = 100 → x = 35 := by
  sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l1258_125812


namespace NUMINAMATH_CALUDE_area_BEIH_l1258_125828

/-- Given a 2×2 square ABCD with B at (0,0), E is the midpoint of AB, 
    F is the midpoint of BC, I is the intersection of AF and DE, 
    and H is the intersection of BD and AF. -/
def square_setup (A B C D E F H I : ℝ × ℝ) : Prop :=
  B = (0, 0) ∧ 
  C = (2, 0) ∧ 
  D = (2, 2) ∧ 
  A = (0, 2) ∧
  E = (0, 1) ∧
  F = (1, 0) ∧
  H.1 = H.2 ∧ -- H is on the diagonal BD
  I.2 = -2 * I.1 + 2 ∧ -- I is on line AF
  I.2 = (1/2) * I.1 + 1 -- I is on line DE

/-- The area of quadrilateral BEIH is 7/15 -/
theorem area_BEIH (A B C D E F H I : ℝ × ℝ) 
  (h : square_setup A B C D E F H I) : 
  let area := (1/2) * abs ((E.1 * I.2 + I.1 * H.2 + H.1 * B.2 + B.1 * E.2) - 
                           (E.2 * I.1 + I.2 * H.1 + H.2 * B.1 + B.2 * E.1))
  area = 7/15 := by
sorry

end NUMINAMATH_CALUDE_area_BEIH_l1258_125828


namespace NUMINAMATH_CALUDE_equation_solution_l1258_125865

theorem equation_solution : ∃ x : ℝ, 2 * (3 * x - 1) = 7 - (x - 5) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1258_125865


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l1258_125863

theorem quadratic_form_equivalence :
  ∀ x y : ℝ, y = (1/2) * x^2 - 2*x + 1 ↔ y = (1/2) * (x - 2)^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l1258_125863


namespace NUMINAMATH_CALUDE_power_function_properties_l1258_125803

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (2 * m^2 - 2 * m - 3) * x^2

-- State the theorem
theorem power_function_properties (m : ℝ) :
  (∀ x y, 0 < x ∧ x < y → f m y < f m x) →  -- f is monotonically decreasing on (0, +∞)
  (f m 8 = Real.sqrt 2 / 4) ∧               -- f(8) = √2/4
  (∀ x, f m (x^2 + 2*x) < f m (x + 6) ↔ x ∈ Set.Ioo (-6) (-3) ∪ Set.Ioi 2) :=
by sorry


end NUMINAMATH_CALUDE_power_function_properties_l1258_125803


namespace NUMINAMATH_CALUDE_sons_age_l1258_125887

/-- Proves that the son's current age is 16 years given the specified conditions -/
theorem sons_age (son_age father_age : ℕ) : 
  father_age = 4 * son_age →
  (son_age - 10) + (father_age - 10) = 60 →
  son_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l1258_125887


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1258_125885

theorem quadratic_max_value (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 + 15 * x + 9
  ∃ (max : ℝ), max = (111 : ℝ) / 4 ∧ ∀ y, f y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1258_125885


namespace NUMINAMATH_CALUDE_simplify_expression_l1258_125855

theorem simplify_expression (a b : ℝ) : (25*a + 70*b) + (15*a + 34*b) - (12*a + 55*b) = 28*a + 49*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1258_125855


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1258_125869

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1258_125869


namespace NUMINAMATH_CALUDE_roots_expression_l1258_125810

theorem roots_expression (p q : ℝ) (α β γ δ : ℝ) : 
  (α^2 + p*α - 2 = 0) → 
  (β^2 + p*β - 2 = 0) → 
  (γ^2 + q*γ - 3 = 0) → 
  (δ^2 + q*δ - 3 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 3*(q^2 - p^2) - 2*q + 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_l1258_125810


namespace NUMINAMATH_CALUDE_divisibility_condition_l1258_125891

theorem divisibility_condition (n : ℤ) : (3 * n + 7) ∣ (5 * n + 13) ↔ n ∈ ({-3, -2, -1} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1258_125891


namespace NUMINAMATH_CALUDE_nancy_balloons_l1258_125867

theorem nancy_balloons (mary_balloons : ℕ) (nancy_balloons : ℕ) : 
  mary_balloons = 28 → 
  mary_balloons = 4 * nancy_balloons → 
  nancy_balloons = 7 := by
sorry

end NUMINAMATH_CALUDE_nancy_balloons_l1258_125867


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l1258_125817

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l1258_125817


namespace NUMINAMATH_CALUDE_flag_design_count_l1258_125893

/-- The number of possible colors for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

theorem flag_design_count :
  num_flag_designs = 27 :=
sorry

end NUMINAMATH_CALUDE_flag_design_count_l1258_125893


namespace NUMINAMATH_CALUDE_hcl_moles_formed_l1258_125824

-- Define the chemical equation
structure ChemicalEquation where
  reactants : List (String × ℕ)
  products : List (String × ℕ)

-- Define the reaction
def reaction : ChemicalEquation :=
  { reactants := [("CH4", 1), ("Cl2", 4)],
    products := [("CCl4", 1), ("HCl", 4)] }

-- Define the initial quantities
def initialQuantities : List (String × ℕ) :=
  [("CH4", 1), ("Cl2", 4)]

-- Theorem to prove
theorem hcl_moles_formed (reaction : ChemicalEquation) (initialQuantities : List (String × ℕ)) :
  reaction.reactants = [("CH4", 1), ("Cl2", 4)] →
  reaction.products = [("CCl4", 1), ("HCl", 4)] →
  initialQuantities = [("CH4", 1), ("Cl2", 4)] →
  (List.find? (λ p => p.1 = "HCl") reaction.products).map Prod.snd = some 4 := by
  sorry

end NUMINAMATH_CALUDE_hcl_moles_formed_l1258_125824


namespace NUMINAMATH_CALUDE_molecular_weight_of_C2H5Cl2O2_l1258_125850

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of C2H5Cl2O2 in g/mol -/
def molecular_weight : ℝ := 2 * carbon_weight + 5 * hydrogen_weight + 2 * chlorine_weight + 2 * oxygen_weight

/-- Theorem stating that the molecular weight of C2H5Cl2O2 is 132.96 g/mol -/
theorem molecular_weight_of_C2H5Cl2O2 : molecular_weight = 132.96 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_of_C2H5Cl2O2_l1258_125850


namespace NUMINAMATH_CALUDE_subtract_three_from_binary_l1258_125854

/-- Converts a binary number (represented as a list of bits) to decimal --/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts a decimal number to binary (represented as a list of bits) --/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2) :: acc)
  aux n []

theorem subtract_three_from_binary :
  let M : List Nat := [0, 1, 0, 1, 0, 1]  -- 101010 in binary
  let M_decimal : Nat := binary_to_decimal M
  let result : List Nat := decimal_to_binary (M_decimal - 3)
  result = [1, 1, 1, 0, 0, 1] -- 100111 in binary
  := by sorry

end NUMINAMATH_CALUDE_subtract_three_from_binary_l1258_125854


namespace NUMINAMATH_CALUDE_quadratic_radicals_combination_l1258_125841

theorem quadratic_radicals_combination (x : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ 3 * x + 5 = k * (2 * x + 7)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_combination_l1258_125841


namespace NUMINAMATH_CALUDE_ratio_problem_l1258_125859

theorem ratio_problem (a b c : ℝ) (h1 : a / b = 11 / 3) (h2 : a / c = 0.7333333333333333) : 
  b / c = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1258_125859


namespace NUMINAMATH_CALUDE_C_not_necessarily_necessary_for_A_C_not_necessarily_sufficient_for_A_l1258_125802

-- Define propositions A, B, and C
variable (A B C : Prop)

-- C is a necessary condition for B
axiom C_necessary_for_B : B → C

-- B is a sufficient condition for A
axiom B_sufficient_for_A : B → A

-- Theorem: C is not necessarily a necessary condition for A
theorem C_not_necessarily_necessary_for_A : ¬(A → C) := by sorry

-- Theorem: C is not necessarily a sufficient condition for A
theorem C_not_necessarily_sufficient_for_A : ¬(C → A) := by sorry

end NUMINAMATH_CALUDE_C_not_necessarily_necessary_for_A_C_not_necessarily_sufficient_for_A_l1258_125802


namespace NUMINAMATH_CALUDE_inequality_proof_l1258_125831

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 3) :
  a * b + b * c + c * a ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c ∧ Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1258_125831


namespace NUMINAMATH_CALUDE_damaged_books_count_damaged_books_proof_l1258_125825

theorem damaged_books_count : ℕ → ℕ → Prop :=
  fun obsolete damaged =>
    (obsolete = 6 * damaged - 8) →
    (obsolete + damaged = 69) →
    (damaged = 11)

-- The proof is omitted
theorem damaged_books_proof : damaged_books_count 58 11 := by sorry

end NUMINAMATH_CALUDE_damaged_books_count_damaged_books_proof_l1258_125825


namespace NUMINAMATH_CALUDE_mountain_temperature_l1258_125866

theorem mountain_temperature (T : ℝ) 
  (h1 : T * (3/4) = T - 21) : T = 84 := by
  sorry

end NUMINAMATH_CALUDE_mountain_temperature_l1258_125866


namespace NUMINAMATH_CALUDE_ratio_proof_l1258_125886

theorem ratio_proof (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 2) : (a + b) / (b + c) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_proof_l1258_125886


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1258_125823

theorem fraction_equivalence :
  ∀ (n : ℚ), (4 + n) / (7 + n) = 3 / 4 ↔ n = 5 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1258_125823


namespace NUMINAMATH_CALUDE_f_1993_of_3_eq_one_fifth_l1258_125858

-- Define the function f
def f (x : ℚ) : ℚ := (1 + x) / (1 - 3 * x)

-- Define the iterated function f_n recursively
def f_n : ℕ → (ℚ → ℚ)
  | 0 => id
  | n + 1 => f ∘ (f_n n)

-- State the theorem
theorem f_1993_of_3_eq_one_fifth : f_n 1993 3 = 1/5 := by sorry

end NUMINAMATH_CALUDE_f_1993_of_3_eq_one_fifth_l1258_125858


namespace NUMINAMATH_CALUDE_complex_polynomial_root_l1258_125899

theorem complex_polynomial_root (a b c d : ℤ) : 
  (a * (Complex.I + 3) ^ 5 + b * (Complex.I + 3) ^ 4 + c * (Complex.I + 3) ^ 3 + 
   d * (Complex.I + 3) ^ 2 + b * (Complex.I + 3) + a = 0) →
  (Int.gcd a (Int.gcd b (Int.gcd c d)) = 1) →
  d.natAbs = 167 := by
sorry

end NUMINAMATH_CALUDE_complex_polynomial_root_l1258_125899


namespace NUMINAMATH_CALUDE_students_with_dogs_and_amphibians_but_not_cats_l1258_125837

theorem students_with_dogs_and_amphibians_but_not_cats (total_students : ℕ) 
  (students_with_dogs : ℕ) (students_with_cats : ℕ) (students_with_amphibians : ℕ) 
  (students_without_pets : ℕ) :
  total_students = 40 →
  students_with_dogs = 24 →
  students_with_cats = 10 →
  students_with_amphibians = 8 →
  students_without_pets = 6 →
  (∃ (x y z : ℕ),
    x + y = students_with_dogs ∧
    y + z = students_with_amphibians ∧
    x + y + z = total_students - students_without_pets ∧
    y = 0) :=
by sorry

end NUMINAMATH_CALUDE_students_with_dogs_and_amphibians_but_not_cats_l1258_125837


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l1258_125844

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/2, 1/3 + 1/9, 1/3 + 1/8]
  (∀ x ∈ sums, x ≤ 1/3 + 1/2) ∧ (1/3 + 1/2 = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l1258_125844


namespace NUMINAMATH_CALUDE_min_triangle_area_l1258_125836

/-- An acute-angled triangle with an inscribed square --/
structure TriangleWithSquare where
  /-- The base length of the triangle --/
  b : ℝ
  /-- The height of the triangle --/
  h : ℝ
  /-- The side length of the inscribed square --/
  s : ℝ
  /-- The triangle is acute-angled --/
  acute : 0 < b ∧ 0 < h
  /-- The square is inscribed as described --/
  square_inscribed : s = (b * h) / (b + h)

/-- The theorem stating the minimum area of the triangle --/
theorem min_triangle_area (t : TriangleWithSquare) (h_area : t.s^2 = 2017) :
  2 * t.s^2 ≤ (t.b * t.h) / 2 ∧
  ∃ (t' : TriangleWithSquare), t'.s^2 = 2017 ∧ (t'.b * t'.h) / 2 = 2 * t'.s^2 := by
  sorry

#check min_triangle_area

end NUMINAMATH_CALUDE_min_triangle_area_l1258_125836


namespace NUMINAMATH_CALUDE_max_circumference_in_standard_parabola_l1258_125822

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a parabola in the form x^2 = 4y -/
def standardParabola : Set (ℝ × ℝ) :=
  {p | p.1^2 = 4 * p.2}

/-- Checks if a circle passes through the vertex of the standard parabola -/
def passesVertexStandardParabola (c : Circle) : Prop :=
  c.center.1^2 + c.center.2^2 = c.radius^2

/-- Checks if a circle is entirely inside the standard parabola -/
def insideStandardParabola (c : Circle) : Prop :=
  ∀ (x y : ℝ), (x - c.center.1)^2 + (y - c.center.2)^2 ≤ c.radius^2 → x^2 ≤ 4 * y

/-- The maximum circumference theorem -/
theorem max_circumference_in_standard_parabola :
  ∃ (c : Circle),
    passesVertexStandardParabola c ∧
    insideStandardParabola c ∧
    (∀ (c' : Circle),
      passesVertexStandardParabola c' ∧
      insideStandardParabola c' →
      2 * π * c'.radius ≤ 2 * π * c.radius) ∧
    2 * π * c.radius = 4 * π :=
sorry

end NUMINAMATH_CALUDE_max_circumference_in_standard_parabola_l1258_125822


namespace NUMINAMATH_CALUDE_probability_of_condition_l1258_125849

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ (m : ℕ), 4 ∣ m ∧ a * b + a + b = m - 1

def total_valid_pairs : ℕ := Nat.choose 60 2

def satisfying_pairs : ℕ := 1350

theorem probability_of_condition :
  (satisfying_pairs : ℚ) / total_valid_pairs = 45 / 59 :=
sorry

end NUMINAMATH_CALUDE_probability_of_condition_l1258_125849


namespace NUMINAMATH_CALUDE_expression_value_l1258_125853

theorem expression_value (x y z : ℝ) 
  (eq1 : 4*x - 6*y - 2*z = 0)
  (eq2 : x + 2*y - 10*z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 - x*y) / (y^2 + z^2) = 26/25 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1258_125853


namespace NUMINAMATH_CALUDE_complex_square_l1258_125882

theorem complex_square (z : ℂ) (i : ℂ) (h1 : z = 5 - 3*i) (h2 : i^2 = -1) :
  z^2 = 16 - 30*i := by sorry

end NUMINAMATH_CALUDE_complex_square_l1258_125882


namespace NUMINAMATH_CALUDE_curve_equation_and_m_range_l1258_125868

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ Real.sqrt ((p.1 - 1)^2 + p.2^2) - p.1 = 1}

-- Define the function for the dot product of vectors FA and FB
def dotProductFAFB (m : ℝ) (A B : ℝ × ℝ) : ℝ :=
  (A.1 - 1) * (B.1 - 1) + A.2 * B.2

theorem curve_equation_and_m_range :
  -- Part 1: The equation of curve C
  (∀ p : ℝ × ℝ, p ∈ C ↔ p.1 > 0 ∧ p.2^2 = 4 * p.1) ∧
  -- Part 2: Existence of m
  (∃ m : ℝ, m > 0 ∧
    ∀ A B : ℝ × ℝ, A ∈ C → B ∈ C →
      (∃ t : ℝ, A.1 = t * A.2 + m ∧ B.1 = t * B.2 + m) →
        dotProductFAFB m A B < 0) ∧
  -- Part 3: Range of m
  (∀ m : ℝ, (∀ A B : ℝ × ℝ, A ∈ C → B ∈ C →
    (∃ t : ℝ, A.1 = t * A.2 + m ∧ B.1 = t * B.2 + m) →
      dotProductFAFB m A B < 0) ↔
        m > 3 - 2 * Real.sqrt 2 ∧ m < 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_curve_equation_and_m_range_l1258_125868


namespace NUMINAMATH_CALUDE_remainder_problem_l1258_125883

theorem remainder_problem (m : ℤ) : (((8 - m) + (m + 4)) % 5) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1258_125883


namespace NUMINAMATH_CALUDE_power_equation_solution_l1258_125890

theorem power_equation_solution (x y : ℕ) :
  (3 : ℝ) ^ x * (4 : ℝ) ^ y = 59049 ∧ x = 10 → x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1258_125890


namespace NUMINAMATH_CALUDE_literary_readers_count_l1258_125880

theorem literary_readers_count (total : ℕ) (sci_fi : ℕ) (both : ℕ) (literary : ℕ) : 
  total = 150 → sci_fi = 120 → both = 60 → literary = total - sci_fi + both → literary = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_literary_readers_count_l1258_125880


namespace NUMINAMATH_CALUDE_intersection_y_coordinate_l1258_125833

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the tangent line at a point (a, 4a^2) on the parabola
def tangent_line (a : ℝ) (x : ℝ) : ℝ := 8 * a * x - 4 * a^2

-- Define the condition for perpendicular tangents
def perpendicular_tangents (a b : ℝ) : Prop := 8 * a * 8 * b = -1

-- Theorem statement
theorem intersection_y_coordinate (a b : ℝ) : 
  a ≠ b → 
  perpendicular_tangents a b →
  ∃ x, tangent_line a x = tangent_line b x ∧ tangent_line a x = -1/4 :=
sorry

end NUMINAMATH_CALUDE_intersection_y_coordinate_l1258_125833


namespace NUMINAMATH_CALUDE_container_cubes_theorem_l1258_125800

/-- Represents the dimensions of a rectangular container -/
structure ContainerDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℕ

/-- Calculate the number of cubes that can fit in the container -/
def cubesFit (container : ContainerDimensions) (cube : CubeDimensions) : ℕ :=
  (container.length / cube.side) * (container.width / cube.side) * (container.height / cube.side)

/-- Calculate the volume of the container -/
def containerVolume (container : ContainerDimensions) : ℕ :=
  container.length * container.width * container.height

/-- Calculate the volume of a single cube -/
def cubeVolume (cube : CubeDimensions) : ℕ :=
  cube.side * cube.side * cube.side

/-- Calculate the fraction of the container volume occupied by cubes -/
def occupiedFraction (container : ContainerDimensions) (cube : CubeDimensions) : ℚ :=
  (cubesFit container cube * cubeVolume cube : ℚ) / containerVolume container

theorem container_cubes_theorem (container : ContainerDimensions) (cube : CubeDimensions) 
  (h1 : container.length = 8)
  (h2 : container.width = 4)
  (h3 : container.height = 9)
  (h4 : cube.side = 2) :
  cubesFit container cube = 32 ∧ occupiedFraction container cube = 8/9 := by
  sorry

#eval cubesFit ⟨8, 4, 9⟩ ⟨2⟩
#eval occupiedFraction ⟨8, 4, 9⟩ ⟨2⟩

end NUMINAMATH_CALUDE_container_cubes_theorem_l1258_125800
