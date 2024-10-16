import Mathlib

namespace NUMINAMATH_CALUDE_intersection_implies_a_greater_than_one_l2766_276630

-- Define the sets A and B
def A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a}

def B (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = b^p.1 + 1}

-- State the theorem
theorem intersection_implies_a_greater_than_one 
  (a b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (A a ∩ B b).Nonempty → a > 1 := by
  sorry


end NUMINAMATH_CALUDE_intersection_implies_a_greater_than_one_l2766_276630


namespace NUMINAMATH_CALUDE_number_of_recitation_orders_l2766_276654

/-- The number of high school seniors --/
def total_students : ℕ := 7

/-- The number of students to be selected --/
def selected_students : ℕ := 4

/-- The number of special students (A, B, C) --/
def special_students : ℕ := 3

/-- Function to calculate the number of recitation orders --/
def recitation_orders : ℕ := sorry

/-- Theorem stating the number of recitation orders --/
theorem number_of_recitation_orders :
  recitation_orders = 768 := by sorry

end NUMINAMATH_CALUDE_number_of_recitation_orders_l2766_276654


namespace NUMINAMATH_CALUDE_fib_sum_equality_l2766_276600

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of first n terms of Fibonacci sequence -/
def fibSum (n : ℕ) : ℕ :=
  (List.range n).map fib |>.sum

/-- Theorem: S_2016 + S_2015 - S_2014 - S_2013 = a_2018 for Fibonacci sequence -/
theorem fib_sum_equality :
  fibSum 2016 + fibSum 2015 - fibSum 2014 - fibSum 2013 = fib 2018 := by
  sorry

end NUMINAMATH_CALUDE_fib_sum_equality_l2766_276600


namespace NUMINAMATH_CALUDE_unique_polynomial_l2766_276650

/-- A polynomial satisfying the given conditions -/
def p : ℝ → ℝ := λ x => x^2 + 1

/-- The theorem stating that p is the unique polynomial satisfying the conditions -/
theorem unique_polynomial :
  (p 3 = 10) ∧
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 3) ∧
  (∀ q : ℝ → ℝ, (q 3 = 10 ∧ ∀ x y : ℝ, q x * q y = q x + q y + q (x * y) - 3) → q = p) :=
by sorry

end NUMINAMATH_CALUDE_unique_polynomial_l2766_276650


namespace NUMINAMATH_CALUDE_probability_non_defective_pencils_l2766_276613

/-- The probability of selecting 3 non-defective pencils from a box of 9 pencils with 2 defective pencils -/
theorem probability_non_defective_pencils :
  let total_pencils : ℕ := 9
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations : ℕ := Nat.choose total_pencils selected_pencils
  let non_defective_combinations : ℕ := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 5 / 12 :=
by sorry

end NUMINAMATH_CALUDE_probability_non_defective_pencils_l2766_276613


namespace NUMINAMATH_CALUDE_exists_valid_configuration_two_thirds_not_exists_valid_configuration_three_fourths_not_exists_valid_configuration_seven_tenths_l2766_276628

/-- Represents the fraction of difficult problems and well-performing students -/
structure TestConfiguration (α : ℚ) where
  difficultProblems : ℚ
  wellPerformingStudents : ℚ
  difficultProblems_ge : difficultProblems ≥ α
  wellPerformingStudents_ge : wellPerformingStudents ≥ α

/-- Theorem stating the existence of a valid configuration for α = 2/3 -/
theorem exists_valid_configuration_two_thirds :
  ∃ (config : TestConfiguration (2/3)), True :=
sorry

/-- Theorem stating the non-existence of a valid configuration for α = 3/4 -/
theorem not_exists_valid_configuration_three_fourths :
  ¬ ∃ (config : TestConfiguration (3/4)), True :=
sorry

/-- Theorem stating the non-existence of a valid configuration for α = 7/10 -/
theorem not_exists_valid_configuration_seven_tenths :
  ¬ ∃ (config : TestConfiguration (7/10)), True :=
sorry

end NUMINAMATH_CALUDE_exists_valid_configuration_two_thirds_not_exists_valid_configuration_three_fourths_not_exists_valid_configuration_seven_tenths_l2766_276628


namespace NUMINAMATH_CALUDE_cookies_per_pan_l2766_276648

theorem cookies_per_pan (total_pans : ℕ) (total_cookies : ℕ) (h1 : total_pans = 5) (h2 : total_cookies = 40) :
  total_cookies / total_pans = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_pan_l2766_276648


namespace NUMINAMATH_CALUDE_sum_2012_terms_equals_negative_2012_l2766_276653

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_2012_terms_equals_negative_2012 :
  let a₁ : ℤ := -2012
  let d : ℤ := 2
  let n : ℕ := 2012
  sum_arithmetic_sequence a₁ d n = -2012 := by sorry

end NUMINAMATH_CALUDE_sum_2012_terms_equals_negative_2012_l2766_276653


namespace NUMINAMATH_CALUDE_lcm_problem_l2766_276624

theorem lcm_problem (m n : ℕ+) :
  m - n = 189 →
  Nat.lcm m n = 133866 →
  m = 22311 ∧ n = 22122 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2766_276624


namespace NUMINAMATH_CALUDE_sqrt_3600_div_15_l2766_276617

theorem sqrt_3600_div_15 : Real.sqrt 3600 / 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3600_div_15_l2766_276617


namespace NUMINAMATH_CALUDE_sixth_root_of_594823321_l2766_276608

theorem sixth_root_of_594823321 : (594823321 : ℝ) ^ (1/6 : ℝ) = 51 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_594823321_l2766_276608


namespace NUMINAMATH_CALUDE_siblings_selection_probability_l2766_276621

/-- Given the probabilities of selection for three siblings, prove that the probability of all three being selected is 3/28 -/
theorem siblings_selection_probability
  (p_ram : ℚ) (p_ravi : ℚ) (p_rani : ℚ)
  (h_ram : p_ram = 5 / 7)
  (h_ravi : p_ravi = 1 / 5)
  (h_rani : p_rani = 3 / 4) :
  p_ram * p_ravi * p_rani = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_siblings_selection_probability_l2766_276621


namespace NUMINAMATH_CALUDE_modular_inverse_three_mod_187_l2766_276639

theorem modular_inverse_three_mod_187 :
  ∃ x : ℕ, x < 187 ∧ (3 * x) % 187 = 1 :=
by
  use 125
  sorry

end NUMINAMATH_CALUDE_modular_inverse_three_mod_187_l2766_276639


namespace NUMINAMATH_CALUDE_lynne_book_cost_l2766_276637

/-- Proves that the cost of each book is $7 given the conditions of Lynne's purchase -/
theorem lynne_book_cost (num_books : ℕ) (num_magazines : ℕ) (magazine_cost : ℚ) (total_spent : ℚ) :
  num_books = 9 →
  num_magazines = 3 →
  magazine_cost = 4 →
  total_spent = 75 →
  (num_books * (total_spent - num_magazines * magazine_cost) / num_books : ℚ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_lynne_book_cost_l2766_276637


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l2766_276672

theorem cost_price_of_ball (selling_price : ℕ) (num_balls : ℕ) (loss_balls : ℕ) :
  selling_price = 720 ∧ num_balls = 17 ∧ loss_balls = 5 →
  ∃ (cost_price : ℕ), cost_price * num_balls - cost_price * loss_balls = selling_price ∧ cost_price = 60 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l2766_276672


namespace NUMINAMATH_CALUDE_remaining_fuel_fraction_l2766_276636

def tank_capacity : ℚ := 12
def round_trip_distance : ℚ := 20
def miles_per_gallon : ℚ := 5

theorem remaining_fuel_fraction :
  (tank_capacity - round_trip_distance / miles_per_gallon) / tank_capacity = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_fuel_fraction_l2766_276636


namespace NUMINAMATH_CALUDE_magnitude_of_vector_l2766_276601

/-- Given two unit vectors e₁ and e₂ on a plane with an angle of 60° between them,
    and a vector OP = 3e₁ + 2e₂, prove that the magnitude of OP is √19. -/
theorem magnitude_of_vector (e₁ e₂ : ℝ × ℝ) : 
  (e₁.1^2 + e₁.2^2 = 1) →  -- e₁ is a unit vector
  (e₂.1^2 + e₂.2^2 = 1) →  -- e₂ is a unit vector
  (e₁.1 * e₂.1 + e₁.2 * e₂.2 = 1/2) →  -- angle between e₁ and e₂ is 60°
  let OP := (3 * e₁.1 + 2 * e₂.1, 3 * e₁.2 + 2 * e₂.2)
  (OP.1^2 + OP.2^2 = 19) :=
by sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_l2766_276601


namespace NUMINAMATH_CALUDE_winner_for_10_winner_for_12_winner_for_15_winner_for_30_l2766_276605

/-- Represents the outcome of the game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Represents the game state -/
structure GameState where
  n : Nat
  circled : List Nat

/-- Checks if two numbers are relatively prime -/
def isRelativelyPrime (a b : Nat) : Bool :=
  Nat.gcd a b = 1

/-- Checks if a number can be circled given the current game state -/
def canCircle (state : GameState) (num : Nat) : Bool :=
  num ≤ state.n &&
  num ∉ state.circled &&
  state.circled.all (isRelativelyPrime num)

/-- Determines the winner of the game given the initial value of N -/
def determineWinner (n : Nat) : GameOutcome :=
  sorry

/-- Theorem stating the game outcome for N = 10 -/
theorem winner_for_10 : determineWinner 10 = GameOutcome.FirstPlayerWins := by sorry

/-- Theorem stating the game outcome for N = 12 -/
theorem winner_for_12 : determineWinner 12 = GameOutcome.FirstPlayerWins := by sorry

/-- Theorem stating the game outcome for N = 15 -/
theorem winner_for_15 : determineWinner 15 = GameOutcome.SecondPlayerWins := by sorry

/-- Theorem stating the game outcome for N = 30 -/
theorem winner_for_30 : determineWinner 30 = GameOutcome.FirstPlayerWins := by sorry

end NUMINAMATH_CALUDE_winner_for_10_winner_for_12_winner_for_15_winner_for_30_l2766_276605


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2766_276643

/-- A Mersenne number is of the form 2^p - 1 for some positive integer p -/
def mersenne_number (p : ℕ) : ℕ := 2^p - 1

/-- A Mersenne prime is a Mersenne number that is also prime -/
def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = mersenne_number p ∧ Nat.Prime n

theorem largest_mersenne_prime_under_500 :
  ∀ n : ℕ, is_mersenne_prime n ∧ n < 500 → n ≤ 127 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2766_276643


namespace NUMINAMATH_CALUDE_x_equals_n_l2766_276681

def x : ℕ → ℚ
  | 0 => 0
  | n + 1 => ((n^2 + n + 1) * x n + 1) / (n^2 + n + 1 - x n)

theorem x_equals_n (n : ℕ) : x n = n := by
  sorry

end NUMINAMATH_CALUDE_x_equals_n_l2766_276681


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2766_276623

/-- Given a rhombus with area 80 cm² and one diagonal 16 cm, prove the other diagonal is 10 cm -/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) : 
  area = 80 → d1 = 16 → area = (d1 * d2) / 2 → d2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2766_276623


namespace NUMINAMATH_CALUDE_least_value_quadratic_l2766_276618

theorem least_value_quadratic (a : ℝ) :
  (∀ x : ℝ, x^2 - 12*x + 35 ≤ 0 → a ≤ x) ↔ a = 5 := by sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l2766_276618


namespace NUMINAMATH_CALUDE_kim_candy_bars_saved_l2766_276663

/-- The number of candy bars Kim's dad buys her per week -/
def candyBarsPerWeek : ℕ := 2

/-- The number of weeks it takes Kim to eat one candy bar -/
def weeksPerCandyBar : ℕ := 4

/-- The total number of weeks -/
def totalWeeks : ℕ := 16

/-- The number of candy bars Kim saved after the total number of weeks -/
def candyBarsSaved : ℕ := totalWeeks * candyBarsPerWeek - totalWeeks / weeksPerCandyBar

theorem kim_candy_bars_saved : candyBarsSaved = 28 := by
  sorry

end NUMINAMATH_CALUDE_kim_candy_bars_saved_l2766_276663


namespace NUMINAMATH_CALUDE_units_digit_27_times_36_l2766_276665

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_27_times_36 : units_digit (27 * 36) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_27_times_36_l2766_276665


namespace NUMINAMATH_CALUDE_odd_sum_of_odd_square_plus_cube_l2766_276652

theorem odd_sum_of_odd_square_plus_cube (n m : ℤ) : 
  Odd (n^2 + m^3) → Odd (n + m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_odd_square_plus_cube_l2766_276652


namespace NUMINAMATH_CALUDE_shirt_costs_15_l2766_276687

/-- The cost of one pair of jeans -/
def jeans_cost : ℚ := sorry

/-- The cost of one shirt -/
def shirt_cost : ℚ := sorry

/-- First condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jeans_cost + 2 * shirt_cost = 69

/-- Second condition: 2 pairs of jeans and 3 shirts cost $71 -/
axiom condition2 : 2 * jeans_cost + 3 * shirt_cost = 71

/-- Theorem: The cost of one shirt is $15 -/
theorem shirt_costs_15 : shirt_cost = 15 := by sorry

end NUMINAMATH_CALUDE_shirt_costs_15_l2766_276687


namespace NUMINAMATH_CALUDE_circle_equation_l2766_276640

/-- Given two circles C1 and C2 where:
    1. C1 has equation (x-1)^2 + (y-1)^2 = 1
    2. The coordinate axes are common tangents of C1 and C2
    3. The distance between the centers of C1 and C2 is 3√2
    Then the equation of C2 must be one of:
    (x-4)^2 + (y-4)^2 = 16
    (x+2)^2 + (y+2)^2 = 4
    (x-2√2)^2 + (y+2√2)^2 = 8
    (x+2√2)^2 + (y-2√2)^2 = 8 -/
theorem circle_equation (C1 C2 : Set (ℝ × ℝ)) : 
  (∀ x y, (x-1)^2 + (y-1)^2 = 1 ↔ (x, y) ∈ C1) →
  (∀ x, (x, 0) ∈ C1 → (x, 0) ∈ C2) →
  (∀ y, (0, y) ∈ C1 → (0, y) ∈ C2) →
  (∃ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ C1 ∧ (x₂, y₂) ∈ C2 ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 18) →
  (∀ x y, (x, y) ∈ C2 ↔ 
    ((x-4)^2 + (y-4)^2 = 16) ∨
    ((x+2)^2 + (y+2)^2 = 4) ∨
    ((x-2*Real.sqrt 2)^2 + (y+2*Real.sqrt 2)^2 = 8) ∨
    ((x+2*Real.sqrt 2)^2 + (y-2*Real.sqrt 2)^2 = 8)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2766_276640


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2766_276690

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_2 : a + b + c = 2) : 
  1 / (a + 3 * b) + 1 / (b + 3 * c) + 1 / (c + 3 * a) ≥ 27 / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2766_276690


namespace NUMINAMATH_CALUDE_blocks_differing_in_three_ways_l2766_276611

/-- Represents the number of options for each attribute of a block -/
structure BlockOptions :=
  (materials : Nat)
  (sizes : Nat)
  (colors : Nat)
  (shapes : Nat)

/-- Calculates the number of blocks that differ in exactly k ways from a specific block -/
def countDifferingBlocks (options : BlockOptions) (k : Nat) : Nat :=
  sorry

/-- The specific block options for our problem -/
def ourBlockOptions : BlockOptions :=
  { materials := 2, sizes := 4, colors := 4, shapes := 4 }

/-- The main theorem: 45 blocks differ in exactly 3 ways from a specific block -/
theorem blocks_differing_in_three_ways :
  countDifferingBlocks ourBlockOptions 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_blocks_differing_in_three_ways_l2766_276611


namespace NUMINAMATH_CALUDE_log_equation_solution_l2766_276607

-- Define the logarithm function with base 1/3
noncomputable def log_one_third (x : ℝ) : ℝ := Real.log x / Real.log (1/3)

-- State the theorem
theorem log_equation_solution :
  ∃! x : ℝ, x > 1 ∧ log_one_third (x^2 + 3*x - 4) = log_one_third (2*x + 2) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2766_276607


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_not_regular_polygon_l2766_276606

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- The triangle has two equal angles -/
  has_two_equal_angles : Bool
  /-- The triangle has a right angle -/
  has_right_angle : Bool
  /-- All isosceles right triangles are similar -/
  always_similar : Bool
  /-- The triangle has two equal sides -/
  has_two_equal_sides : Bool

/-- A regular polygon -/
structure RegularPolygon where
  /-- All sides are equal -/
  equilateral : Bool
  /-- All angles are equal -/
  equiangular : Bool

/-- Theorem: Isosceles right triangles are not regular polygons -/
theorem isosceles_right_triangle_not_regular_polygon (t : IsoscelesRightTriangle) : 
  ¬∃(p : RegularPolygon), (t.has_two_equal_angles ∧ t.has_right_angle ∧ t.always_similar ∧ t.has_two_equal_sides) → 
  (p.equilateral ∧ p.equiangular) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_not_regular_polygon_l2766_276606


namespace NUMINAMATH_CALUDE_complex_number_magnitude_squared_l2766_276634

theorem complex_number_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 4 - 7*I) :
  Complex.abs z^2 = 65/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_squared_l2766_276634


namespace NUMINAMATH_CALUDE_sarah_hair_products_usage_l2766_276612

/-- Given Sarah's daily shampoo and conditioner usage, calculate the total volume used in 14 days -/
theorem sarah_hair_products_usage 
  (shampoo_daily : ℝ) 
  (conditioner_daily : ℝ) 
  (h1 : shampoo_daily = 1) 
  (h2 : conditioner_daily = shampoo_daily / 2) 
  (days : ℕ) 
  (h3 : days = 14) : 
  shampoo_daily * days + conditioner_daily * days = 21 := by
  sorry


end NUMINAMATH_CALUDE_sarah_hair_products_usage_l2766_276612


namespace NUMINAMATH_CALUDE_total_heads_count_l2766_276633

/-- Proves that the total number of heads is 48 given the conditions of the problem -/
theorem total_heads_count (hens cows : ℕ) : 
  hens = 28 →
  2 * hens + 4 * cows = 136 →
  hens + cows = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_heads_count_l2766_276633


namespace NUMINAMATH_CALUDE_sequence_property_l2766_276638

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a n - a (n + 1) = (a n * a (n + 1)) / (2^(n - 1))

theorem sequence_property (a : ℕ → ℚ) (k : ℕ) 
  (h1 : RecurrenceSequence a) 
  (h2 : a 2 = -1)
  (h3 : a k = 16 * a 8)
  (h4 : k > 0) : 
  k = 12 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l2766_276638


namespace NUMINAMATH_CALUDE_percentage_of_women_in_non_union_l2766_276685

theorem percentage_of_women_in_non_union (total_employees : ℝ) 
  (h1 : total_employees > 0)
  (h2 : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ p * total_employees = number_of_male_employees)
  (h3 : 0.6 * total_employees = number_of_unionized_employees)
  (h4 : 0.7 * number_of_unionized_employees = number_of_male_unionized_employees)
  (h5 : 0.9 * (total_employees - number_of_unionized_employees) = number_of_female_non_unionized_employees) :
  (number_of_female_non_unionized_employees / (total_employees - number_of_unionized_employees)) = 0.9 := by
sorry


end NUMINAMATH_CALUDE_percentage_of_women_in_non_union_l2766_276685


namespace NUMINAMATH_CALUDE_sample_size_is_40_l2766_276686

/-- Represents a frequency distribution histogram -/
structure Histogram where
  num_bars : ℕ
  central_freq : ℕ
  other_freq : ℕ

/-- Calculates the sample size of a histogram -/
def sample_size (h : Histogram) : ℕ :=
  h.central_freq + h.other_freq

/-- Theorem stating the sample size for the given histogram -/
theorem sample_size_is_40 (h : Histogram) 
  (h_bars : h.num_bars = 7)
  (h_central : h.central_freq = 8)
  (h_ratio : h.central_freq = h.other_freq / 4) :
  sample_size h = 40 := by
  sorry

#check sample_size_is_40

end NUMINAMATH_CALUDE_sample_size_is_40_l2766_276686


namespace NUMINAMATH_CALUDE_xy_positive_iff_fraction_positive_and_am_gm_inequality_l2766_276669

theorem xy_positive_iff_fraction_positive_and_am_gm_inequality :
  (∀ x y : ℝ, x * y > 0 ↔ x / y > 0) ∧
  (∀ a b : ℝ, a * b ≤ ((a + b) / 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_xy_positive_iff_fraction_positive_and_am_gm_inequality_l2766_276669


namespace NUMINAMATH_CALUDE_unique_two_digit_ratio_l2766_276646

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem unique_two_digit_ratio :
  ∃! n : ℕ, is_two_digit n ∧ (n : ℚ) / (reverse_digits n : ℚ) = 7 / 4 :=
by
  use 21
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_ratio_l2766_276646


namespace NUMINAMATH_CALUDE_sqrt2_plus_1_power_l2766_276603

theorem sqrt2_plus_1_power (n : ℕ+) :
  ∃ m : ℕ+, (Real.sqrt 2 + 1) ^ n.val = Real.sqrt m.val + Real.sqrt (m.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_plus_1_power_l2766_276603


namespace NUMINAMATH_CALUDE_hockey_players_count_l2766_276632

theorem hockey_players_count (total_players cricket_players football_players softball_players : ℕ) 
  (h1 : total_players = 77)
  (h2 : cricket_players = 22)
  (h3 : football_players = 21)
  (h4 : softball_players = 19) :
  total_players - (cricket_players + football_players + softball_players) = 15 := by
sorry

end NUMINAMATH_CALUDE_hockey_players_count_l2766_276632


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l2766_276677

theorem largest_n_satisfying_conditions : ∃ (n : ℤ), n = 181 ∧ 
  (∃ (m : ℤ), n^2 = (m+1)^3 - m^3) ∧ 
  (∃ (k : ℤ), 2*n + 79 = k^2) ∧
  (∀ (n' : ℤ), n' > n → 
    (¬∃ (m : ℤ), n'^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ (k : ℤ), 2*n' + 79 = k^2)) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l2766_276677


namespace NUMINAMATH_CALUDE_intersection_equality_l2766_276698

def M : Set ℤ := {-1, 0, 1}

def N (a : ℤ) : Set ℤ := {a, a^2}

theorem intersection_equality (a : ℤ) : M ∩ N a = N a ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l2766_276698


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2766_276645

theorem quadratic_root_in_unit_interval
  (a b c m : ℝ)
  (ha : a > 0)
  (hm : m > 0)
  (h_sum : a / (m + 2) + b / (m + 1) + c / m = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2766_276645


namespace NUMINAMATH_CALUDE_two_point_distribution_max_value_l2766_276625

/-- A random variable following a two-point distribution -/
structure TwoPointDistribution where
  p : ℝ
  hp : 0 < p ∧ p < 1

/-- The expected value of a two-point distribution -/
def expectedValue (ξ : TwoPointDistribution) : ℝ := ξ.p

/-- The variance of a two-point distribution -/
def variance (ξ : TwoPointDistribution) : ℝ := ξ.p * (1 - ξ.p)

/-- The theorem stating the maximum value of (2D(ξ)-1)/E(ξ) for a two-point distribution -/
theorem two_point_distribution_max_value (ξ : TwoPointDistribution) :
  (∃ (c : ℝ), ∀ (η : TwoPointDistribution), (2 * variance η - 1) / expectedValue η ≤ c) ∧
  (∃ (ξ_max : TwoPointDistribution), (2 * variance ξ_max - 1) / expectedValue ξ_max = 2 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_two_point_distribution_max_value_l2766_276625


namespace NUMINAMATH_CALUDE_pharmacist_weights_existence_l2766_276678

theorem pharmacist_weights_existence :
  ∃ (a b c : ℝ), 
    a < b ∧ b < c ∧
    a + b = 100 ∧
    a + c = 101 ∧
    b + c = 102 ∧
    a < 90 ∧ b < 90 ∧ c < 90 := by
  sorry

end NUMINAMATH_CALUDE_pharmacist_weights_existence_l2766_276678


namespace NUMINAMATH_CALUDE_fill_pipe_fraction_l2766_276658

/-- Represents the fraction of a cistern that can be filled in a given time -/
def FractionFilled (time : ℝ) : ℝ := sorry

theorem fill_pipe_fraction :
  let fill_time : ℝ := 30
  let fraction := FractionFilled fill_time
  (∃ (f : ℝ), FractionFilled fill_time = f ∧ f * fill_time = fill_time) →
  fraction = 1 := by sorry

end NUMINAMATH_CALUDE_fill_pipe_fraction_l2766_276658


namespace NUMINAMATH_CALUDE_certain_yellow_ball_pick_l2766_276610

theorem certain_yellow_ball_pick (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) (m : ℕ) : 
  total_balls = 8 →
  red_balls = 3 →
  yellow_balls = 5 →
  total_balls = red_balls + yellow_balls →
  m ≤ red_balls →
  yellow_balls = total_balls - m →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_yellow_ball_pick_l2766_276610


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l2766_276660

/-- For an arithmetic sequence with first term a₁ and common difference d,
    the condition 2a₁ + 11d > 0 is sufficient but not necessary for 2a₁ + 11d ≥ 0 -/
theorem arithmetic_sequence_condition (a₁ d : ℝ) :
  (∃ x y : ℝ, (x > y) ∧ (x ≥ 0) ∧ (y < 0)) ∧
  (2 * a₁ + 11 * d > 0 → 2 * a₁ + 11 * d ≥ 0) ∧
  ¬(2 * a₁ + 11 * d ≥ 0 → 2 * a₁ + 11 * d > 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l2766_276660


namespace NUMINAMATH_CALUDE_robot_sorting_problem_l2766_276694

/-- Represents the sorting capacity of robots -/
structure RobotSorting where
  typeA : ℕ  -- Number of type A robots
  typeB : ℕ  -- Number of type B robots
  totalPackages : ℕ  -- Total packages sorted per hour

/-- Theorem representing the robot sorting problem -/
theorem robot_sorting_problem 
  (scenario1 : RobotSorting)
  (scenario2 : RobotSorting)
  (h1 : scenario1.typeA = 80 ∧ scenario1.typeB = 100 ∧ scenario1.totalPackages = 8200)
  (h2 : scenario2.typeA = 50 ∧ scenario2.typeB = 50 ∧ scenario2.totalPackages = 4500)
  (totalNewRobots : ℕ)
  (h3 : totalNewRobots = 200)
  (minNewPackages : ℕ)
  (h4 : minNewPackages = 9000) :
  ∃ (maxTypeA : ℕ),
    maxTypeA ≤ totalNewRobots ∧
    ∀ (newTypeA : ℕ),
      newTypeA ≤ totalNewRobots →
      (40 * newTypeA + 50 * (totalNewRobots - newTypeA) ≥ minNewPackages →
       newTypeA ≤ maxTypeA) ∧
    40 * maxTypeA + 50 * (totalNewRobots - maxTypeA) ≥ minNewPackages ∧
    maxTypeA = 100 :=
sorry

end NUMINAMATH_CALUDE_robot_sorting_problem_l2766_276694


namespace NUMINAMATH_CALUDE_calvin_chips_days_l2766_276602

/-- The number of days per week Calvin buys chips -/
def days_per_week : ℕ := sorry

/-- The cost of one pack of chips in dollars -/
def cost_per_pack : ℚ := 1/2

/-- The number of weeks Calvin has been buying chips -/
def num_weeks : ℕ := 4

/-- The total amount Calvin has spent on chips in dollars -/
def total_spent : ℚ := 10

theorem calvin_chips_days :
  days_per_week * num_weeks * cost_per_pack = total_spent ∧
  days_per_week = 5 := by sorry

end NUMINAMATH_CALUDE_calvin_chips_days_l2766_276602


namespace NUMINAMATH_CALUDE_binomial_coefficient_1000_1000_l2766_276667

theorem binomial_coefficient_1000_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1000_1000_l2766_276667


namespace NUMINAMATH_CALUDE_correct_snow_globes_count_l2766_276683

/-- The number of snow globes in each box of Christmas decorations -/
def snow_globes_per_box : ℕ := 5

/-- The number of pieces of tinsel in each box -/
def tinsel_per_box : ℕ := 4

/-- The number of Christmas trees in each box -/
def trees_per_box : ℕ := 1

/-- The total number of boxes distributed -/
def total_boxes : ℕ := 12

/-- The total number of decorations handed out -/
def total_decorations : ℕ := 120

/-- Theorem stating that the number of snow globes per box is correct -/
theorem correct_snow_globes_count :
  snow_globes_per_box = (total_decorations - total_boxes * (tinsel_per_box + trees_per_box)) / total_boxes :=
by sorry

end NUMINAMATH_CALUDE_correct_snow_globes_count_l2766_276683


namespace NUMINAMATH_CALUDE_lasagna_pieces_sum_to_six_l2766_276675

/-- Represents the number of lasagna pieces each person eats -/
structure LasagnaPieces where
  manny : ℚ
  aaron : ℚ
  kai : ℚ
  raphael : ℚ
  lisa : ℚ

/-- Calculates the total number of lasagna pieces eaten -/
def total_pieces (pieces : LasagnaPieces) : ℚ :=
  pieces.manny + pieces.aaron + pieces.kai + pieces.raphael + pieces.lisa

/-- Theorem stating the total number of lasagna pieces equals 6 -/
theorem lasagna_pieces_sum_to_six : ∃ (pieces : LasagnaPieces), 
  pieces.manny = 1 ∧ 
  pieces.aaron = 0 ∧ 
  pieces.kai = 2 * pieces.manny ∧ 
  pieces.raphael = pieces.manny / 2 ∧ 
  pieces.lisa = 2 + pieces.raphael ∧ 
  total_pieces pieces = 6 := by
  sorry

end NUMINAMATH_CALUDE_lasagna_pieces_sum_to_six_l2766_276675


namespace NUMINAMATH_CALUDE_whole_number_between_bounds_l2766_276604

theorem whole_number_between_bounds (M : ℤ) : 9 < (M : ℚ) / 4 ∧ (M : ℚ) / 4 < 9.5 → M = 37 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_bounds_l2766_276604


namespace NUMINAMATH_CALUDE_nonagon_arithmetic_mean_property_l2766_276609

/-- Represents a vertex of the nonagon with its assigned number -/
structure Vertex where
  index : Fin 9
  value : Nat

/-- Checks if three vertices form an equilateral triangle in a regular nonagon -/
def isEquilateralTriangle (v1 v2 v3 : Vertex) : Prop :=
  (v2.index - v1.index) % 3 = 0 ∧ (v3.index - v2.index) % 3 = 0 ∧ (v1.index - v3.index) % 3 = 0

/-- Checks if one number is the arithmetic mean of the other two -/
def isArithmeticMean (a b c : Nat) : Prop :=
  2 * b = a + c

/-- The arrangement of numbers on the nonagon -/
def arrangement : List Vertex :=
  List.map (fun i => ⟨i, 2016 + i⟩) (List.range 9)

/-- The main theorem to prove -/
theorem nonagon_arithmetic_mean_property :
  ∀ v1 v2 v3 : Vertex,
    v1 ∈ arrangement →
    v2 ∈ arrangement →
    v3 ∈ arrangement →
    isEquilateralTriangle v1 v2 v3 →
    isArithmeticMean v1.value v2.value v3.value ∨
    isArithmeticMean v2.value v3.value v1.value ∨
    isArithmeticMean v3.value v1.value v2.value :=
  sorry

end NUMINAMATH_CALUDE_nonagon_arithmetic_mean_property_l2766_276609


namespace NUMINAMATH_CALUDE_rectangle_quadrilateral_area_l2766_276619

/-- Given a rectangle with sides 5 cm and 48 cm, where the longer side is divided into three equal parts
    and the midpoint of the shorter side is connected to the first division point on the longer side,
    the area of the resulting smaller quadrilateral is 90 cm². -/
theorem rectangle_quadrilateral_area :
  let short_side : ℝ := 5
  let long_side : ℝ := 48
  let division_point : ℝ := long_side / 3
  let midpoint : ℝ := short_side / 2
  let total_area : ℝ := short_side * long_side
  let part_area : ℝ := short_side * division_point
  let quadrilateral_area : ℝ := part_area + (part_area / 2)
  quadrilateral_area = 90
  := by sorry

end NUMINAMATH_CALUDE_rectangle_quadrilateral_area_l2766_276619


namespace NUMINAMATH_CALUDE_base_5_to_base_7_conversion_l2766_276642

def base_5_to_decimal (n : ℕ) : ℕ := 
  2 * 5^0 + 1 * 5^1 + 4 * 5^2

def decimal_to_base_7 (n : ℕ) : List ℕ :=
  [2, 1, 2]

theorem base_5_to_base_7_conversion :
  decimal_to_base_7 (base_5_to_decimal 412) = [2, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_5_to_base_7_conversion_l2766_276642


namespace NUMINAMATH_CALUDE_boxes_sold_theorem_l2766_276655

/-- Represents the number of boxes sold on each day --/
structure BoxesSold where
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Calculates the total number of boxes sold over three days --/
def totalBoxesSold (boxes : BoxesSold) : ℕ :=
  boxes.friday + boxes.saturday + boxes.sunday

/-- Theorem stating the total number of boxes sold over three days --/
theorem boxes_sold_theorem (boxes : BoxesSold) 
  (h1 : boxes.friday = 40)
  (h2 : boxes.saturday = 2 * boxes.friday - 10)
  (h3 : boxes.sunday = boxes.saturday / 2) :
  totalBoxesSold boxes = 145 := by
  sorry

#check boxes_sold_theorem

end NUMINAMATH_CALUDE_boxes_sold_theorem_l2766_276655


namespace NUMINAMATH_CALUDE_painter_can_blacken_all_cells_l2766_276671

/-- Represents a cell on the board -/
structure Cell :=
  (x : Nat) (y : Nat)

/-- Represents the color of a cell -/
inductive Color
  | Black
  | White

/-- Represents the board -/
def Board := Cell → Color

/-- Represents the painter's position -/
structure PainterPosition :=
  (cell : Cell)

/-- Function to change the color of a cell -/
def changeColor (color : Color) : Color :=
  match color with
  | Color.Black => Color.White
  | Color.White => Color.Black

/-- Function to check if a cell is on the border of the board -/
def isBorderCell (cell : Cell) (rows : Nat) (cols : Nat) : Prop :=
  cell.x = 0 ∨ cell.x = rows - 1 ∨ cell.y = 0 ∨ cell.y = cols - 1

/-- The main theorem -/
theorem painter_can_blacken_all_cells :
  ∀ (initialBoard : Board) (startPos : PainterPosition),
    (∀ (cell : Cell), cell.x < 2012 ∧ cell.y < 2013) →  -- Board dimensions
    (startPos.cell.x = 0 ∨ startPos.cell.x = 2011) ∧ (startPos.cell.y = 0 ∨ startPos.cell.y = 2012) →  -- Start from corner
    (∀ (cell : Cell), (cell.x + cell.y) % 2 = 0 → initialBoard cell = Color.Black) →  -- Initial checkerboard pattern
    (∀ (cell : Cell), (cell.x + cell.y) % 2 = 1 → initialBoard cell = Color.White) →
    ∃ (finalBoard : Board) (endPos : PainterPosition),
      (∀ (cell : Cell), finalBoard cell = Color.Black) ∧  -- All cells are black
      isBorderCell endPos.cell 2012 2013 :=  -- End on border
by sorry

end NUMINAMATH_CALUDE_painter_can_blacken_all_cells_l2766_276671


namespace NUMINAMATH_CALUDE_lottery_probability_l2766_276662

theorem lottery_probability (total_tickets : Nat) (winning_tickets : Nat) (buyers : Nat) :
  total_tickets = 10 →
  winning_tickets = 3 →
  buyers = 5 →
  let prob_at_least_one_wins := 1 - (Nat.choose (total_tickets - winning_tickets) buyers / Nat.choose total_tickets buyers)
  prob_at_least_one_wins = 77 / 84 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l2766_276662


namespace NUMINAMATH_CALUDE_factorial_ones_divisibility_l2766_276651

/-- Definition of [n]! as the product of numbers consisting of n ones -/
def factorial_ones (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => (10^(i+1) - 1) / 9)

/-- Theorem stating that [n+m]! is divisible by [n]! * [m]! -/
theorem factorial_ones_divisibility (n m : ℕ) :
  ∃ k : ℕ, factorial_ones (n + m) = k * (factorial_ones n * factorial_ones m) := by
  sorry

end NUMINAMATH_CALUDE_factorial_ones_divisibility_l2766_276651


namespace NUMINAMATH_CALUDE_free_throw_contest_ratio_l2766_276641

theorem free_throw_contest_ratio (alex sandra hector : ℕ) : 
  alex = 8 →
  sandra = 3 * alex →
  alex + sandra + hector = 80 →
  hector / sandra = 2 := by
sorry

end NUMINAMATH_CALUDE_free_throw_contest_ratio_l2766_276641


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2766_276695

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2003 : ℝ) ^ x + (2004 : ℝ) ^ x = (2005 : ℝ) ^ x := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2766_276695


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2766_276693

theorem line_circle_intersection (r : ℝ) (A B : ℝ × ℝ) (h_r : r > 0) : 
  (∀ (x y : ℝ), 3*x - 4*y + 5 = 0 → x^2 + y^2 = r^2) →
  (A.1^2 + A.2^2 = r^2) →
  (B.1^2 + B.2^2 = r^2) →
  (3*A.1 - 4*A.2 + 5 = 0) →
  (3*B.1 - 4*B.2 + 5 = 0) →
  (A.1 * B.1 + A.2 * B.2 = -r^2/2) →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2766_276693


namespace NUMINAMATH_CALUDE_total_birds_and_storks_l2766_276673

def birds_and_storks (initial_birds : ℕ) (initial_storks : ℕ) (additional_storks : ℕ) : ℕ :=
  initial_birds + initial_storks + additional_storks

theorem total_birds_and_storks :
  birds_and_storks 3 4 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_and_storks_l2766_276673


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2766_276684

/-- Given a triangle ABC with vertices A(4, 4), B(-4, 2), and C(2, 0) -/
def triangle_ABC : Set (ℝ × ℝ) := {(4, 4), (-4, 2), (2, 0)}

/-- The equation of a line ax + by + c = 0 is represented by the triple (a, b, c) -/
def Line := ℝ × ℝ × ℝ

/-- The median CD of triangle ABC -/
def median_CD : Line := sorry

/-- The altitude from C to AB -/
def altitude_C : Line := sorry

/-- The centroid G of triangle ABC -/
def centroid_G : ℝ × ℝ := sorry

theorem triangle_ABC_properties :
  (median_CD = (3, 2, -6)) ∧
  (altitude_C = (4, 1, -8)) ∧
  (centroid_G = (2/3, 2)) := by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2766_276684


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2766_276668

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 1/2 ∧ x₂ = 1 ∧ 
  2 * x₁^2 - 3 * x₁ + 1 = 0 ∧ 
  2 * x₂^2 - 3 * x₂ + 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2766_276668


namespace NUMINAMATH_CALUDE_tray_trips_l2766_276666

theorem tray_trips (capacity : ℕ) (total_trays : ℕ) (h1 : capacity = 8) (h2 : total_trays = 16) :
  (total_trays + capacity - 1) / capacity = 2 := by
  sorry

end NUMINAMATH_CALUDE_tray_trips_l2766_276666


namespace NUMINAMATH_CALUDE_line_through_parabola_focus_l2766_276674

/-- The value of 'a' for a line ax - y + 1 = 0 passing through the focus of the parabola y^2 = 4x -/
theorem line_through_parabola_focus (a : ℝ) : 
  (∃ x y : ℝ, y^2 = 4*x ∧ a*x - y + 1 = 0 ∧ x = 1 ∧ y = 0) → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_through_parabola_focus_l2766_276674


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l2766_276622

theorem rectangle_length_proof (width : ℝ) (small_area : ℝ) : 
  width = 20 → small_area = 200 → ∃ (length : ℝ), 
    length = 40 ∧ 
    (length / 2) * (width / 2) = small_area := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l2766_276622


namespace NUMINAMATH_CALUDE_equation_properties_l2766_276679

def p (x : ℝ) := x^4 - x^3 - 1

theorem equation_properties :
  (∃ (r₁ r₂ : ℝ), p r₁ = 0 ∧ p r₂ = 0 ∧ r₁ ≠ r₂ ∧
    (∀ (r : ℝ), p r = 0 → r = r₁ ∨ r = r₂)) ∧
  (∀ (r₁ r₂ : ℝ), p r₁ = 0 → p r₂ = 0 → r₁ + r₂ > 6/11) ∧
  (∀ (r₁ r₂ : ℝ), p r₁ = 0 → p r₂ = 0 → r₁ * r₂ < -11/10) :=
by sorry

end NUMINAMATH_CALUDE_equation_properties_l2766_276679


namespace NUMINAMATH_CALUDE_probability_between_C_and_D_l2766_276649

/-- Given a line segment AB with points C, D, and E such that AB = 4AD = 4BE
    and AD = DC = CE = EB, the probability of a random point on AB being
    between C and D is 1/4. -/
theorem probability_between_C_and_D (A B C D E : ℝ) : 
  A < C ∧ C < D ∧ D < E ∧ E < B →
  B - A = 4 * (D - A) →
  B - A = 4 * (B - E) →
  D - A = C - D →
  C - D = E - C →
  E - C = B - E →
  (D - C) / (B - A) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_between_C_and_D_l2766_276649


namespace NUMINAMATH_CALUDE_train_speed_l2766_276696

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 250.00000000000003)
  (h2 : time = 15) : 
  ∃ (speed : ℝ), abs (speed - 60) < 0.00000000000001 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2766_276696


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l2766_276657

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 - 3*z + 2) ≤ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l2766_276657


namespace NUMINAMATH_CALUDE_f_properties_l2766_276670

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem f_properties :
  -1 < a → a < 0 → x > 0 →
  (∃ (max_val min_val : ℝ),
    (a = -1/2 → 
      (∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y ≤ max_val) ∧
      (∃ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y = max_val) ∧
      max_val = 1/2 + (Real.exp 1)^2/4) ∧
    (a = -1/2 → 
      (∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y ≥ min_val) ∧
      (∃ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a y = min_val) ∧
      min_val = 5/4)) ∧
  (∀ y z, 0 < y → y < Real.sqrt (-a/(a+1)) → z ≥ Real.sqrt (-a/(a+1)) → 
    f a y ≥ f a (Real.sqrt (-a/(a+1))) ∧ f a z ≥ f a (Real.sqrt (-a/(a+1)))) ∧
  (∀ y, y > 0 → f a y > 1 + a/2 * Real.log (-a) ↔ 1/Real.exp 1 - 1 < a) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l2766_276670


namespace NUMINAMATH_CALUDE_saturday_visitors_count_l2766_276699

def friday_visitors : ℕ := 12315
def saturday_multiplier : ℕ := 7

theorem saturday_visitors_count : friday_visitors * saturday_multiplier = 86205 := by
  sorry

end NUMINAMATH_CALUDE_saturday_visitors_count_l2766_276699


namespace NUMINAMATH_CALUDE_sum_of_roots_l2766_276682

theorem sum_of_roots (k m : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : 2 * y₁^2 - k * y₁ - m = 0)
  (h₂ : 2 * y₂^2 - k * y₂ - m = 0)
  (h₃ : y₁ ≠ y₂) : 
  y₁ + y₂ = k / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2766_276682


namespace NUMINAMATH_CALUDE_problem_solution_l2766_276615

theorem problem_solution (a : ℝ) (h : a^2 - 4*a + 3 = 0) :
  (9 - 3*a) / (2*a - 4) / (a + 2 - 5 / (a - 2)) = -3/8 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2766_276615


namespace NUMINAMATH_CALUDE_curve_symmetric_line_k_l2766_276626

/-- The curve equation --/
def curve (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 6*y + 1 = 0

/-- The line equation --/
def line (k x y : ℝ) : Prop :=
  k*x + 2*y - 4 = 0

/-- Two points are symmetric with respect to a line --/
def symmetric (P Q : ℝ × ℝ) (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line k x y ∧ 
    (P.1 + Q.1 = 2*x) ∧ (P.2 + Q.2 = 2*y)

theorem curve_symmetric_line_k (P Q : ℝ × ℝ) (k : ℝ) :
  P ≠ Q →
  curve P.1 P.2 →
  curve Q.1 Q.2 →
  symmetric P Q k →
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetric_line_k_l2766_276626


namespace NUMINAMATH_CALUDE_equation_solution_l2766_276635

theorem equation_solution : ∃ x : ℝ, 4*x + 4 - x - 2*x + 2 - 2 - x + 2 + 6 = 0 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2766_276635


namespace NUMINAMATH_CALUDE_last_digit_of_n_l2766_276691

/-- Represents a natural number with its digits -/
structure DigitNumber where
  value : ℕ
  num_digits : ℕ
  greater_than_ten : value > 10

/-- Represents the transformation from N to M -/
structure Transformation where
  increase_by_two : ℕ  -- position of the digit increased by 2
  increase_by_odd : List ℕ  -- list of odd numbers added to other digits

/-- Main theorem statement -/
theorem last_digit_of_n (N M : DigitNumber) (t : Transformation) :
  M.value = 3 * N.value →
  M.num_digits = N.num_digits →
  (∃ (transformed_N : ℕ), transformed_N = N.value + t.increase_by_two + t.increase_by_odd.sum) →
  N.value % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_n_l2766_276691


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2766_276644

theorem inequality_system_solution (k : ℝ) : 
  (∀ x : ℝ, (2*x + 9 > 6*x + 1 ∧ x - k < 1) ↔ x < 2) → 
  k ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2766_276644


namespace NUMINAMATH_CALUDE_rearrangement_maintains_ratio_l2766_276627

/-- Represents a figure made of sticks -/
structure StickFigure where
  num_sticks : ℕ
  area : ℝ

/-- The total number of sticks -/
def total_sticks : ℕ := 20

/-- The number of fixed sticks -/
def fixed_sticks : ℕ := 12

/-- The initial figure with 14 sticks -/
def initial_figure_14 : StickFigure := ⟨14, 3⟩

/-- The initial figure with 6 sticks -/
def initial_figure_6 : StickFigure := ⟨6, 1⟩

/-- The rearranged figure with 7 sticks -/
def rearranged_figure_7 : StickFigure := ⟨7, 1⟩

/-- The rearranged figure with 13 sticks -/
def rearranged_figure_13 : StickFigure := ⟨13, 3⟩

/-- Theorem stating that the rearrangement maintains the area ratio -/
theorem rearrangement_maintains_ratio :
  (initial_figure_14.area / initial_figure_6.area = 
   rearranged_figure_13.area / rearranged_figure_7.area) ∧
  (total_sticks = initial_figure_14.num_sticks + initial_figure_6.num_sticks) ∧
  (total_sticks = rearranged_figure_13.num_sticks + rearranged_figure_7.num_sticks) ∧
  (fixed_sticks + rearranged_figure_13.num_sticks - rearranged_figure_7.num_sticks = initial_figure_14.num_sticks) :=
by sorry

end NUMINAMATH_CALUDE_rearrangement_maintains_ratio_l2766_276627


namespace NUMINAMATH_CALUDE_no_rectangle_with_half_perimeter_and_area_l2766_276664

theorem no_rectangle_with_half_perimeter_and_area 
  (a b : ℝ) (h_ab : 0 < a ∧ a < b) : 
  ¬∃ (x y : ℝ), 
    0 < x ∧ x < b ∧
    0 < y ∧ y < b ∧
    x + y = a + b ∧
    x * y = (a * b) / 2 := by
sorry

end NUMINAMATH_CALUDE_no_rectangle_with_half_perimeter_and_area_l2766_276664


namespace NUMINAMATH_CALUDE_no_function_with_double_application_plus_2019_l2766_276688

-- Statement of the theorem
theorem no_function_with_double_application_plus_2019 :
  ¬ (∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2019) := by
  sorry

end NUMINAMATH_CALUDE_no_function_with_double_application_plus_2019_l2766_276688


namespace NUMINAMATH_CALUDE_x_range_for_quartic_equation_l2766_276692

theorem x_range_for_quartic_equation (k x : ℝ) :
  x^4 - 2*k*x^2 + k^2 + 2*k - 3 = 0 → -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_quartic_equation_l2766_276692


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l2766_276614

theorem greatest_integer_solution : 
  ∃ (n : ℤ), (∀ (x : ℤ), 6*x^2 + 5*x - 8 < 3*x^2 - 4*x + 1 → x ≤ n) ∧ 
  (6*n^2 + 5*n - 8 < 3*n^2 - 4*n + 1) ∧ 
  n = 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l2766_276614


namespace NUMINAMATH_CALUDE_arithmetic_sequence_l2766_276689

theorem arithmetic_sequence (a b c : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃ x : ℝ, bx^2 - 4*b*x + 2*(a+c) = 0 ∧ (∀ y : ℝ, bx^2 - 4*b*x + 2*(a+c) = 0 → y = x)) :
  b - a = c - b := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_l2766_276689


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_four_l2766_276697

/-- A line in 2D space defined by parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if two lines are parallel --/
def are_parallel (l1 l2 : ParametricLine) : Prop :=
  ∃ (k : ℝ), ∀ (t : ℝ), 
    (l1.x t - l1.x 0) * (l2.y t - l2.y 0) = k * (l1.y t - l1.y 0) * (l2.x t - l2.x 0)

/-- The first line l₁ --/
def l1 : ParametricLine where
  x := λ s => 2 * s + 1
  y := λ s => s

/-- The second line l₂ --/
def l2 (a : ℝ) : ParametricLine where
  x := λ t => a * t
  y := λ t => 2 * t - 1

/-- Theorem: If l₁ and l₂ are parallel, then a = 4 --/
theorem parallel_lines_a_equals_four :
  are_parallel l1 (l2 a) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_four_l2766_276697


namespace NUMINAMATH_CALUDE_pentagon_area_l2766_276656

theorem pentagon_area (a b c d e : ℝ) (h₁ : a = 18) (h₂ : b = 25) (h₃ : c = 30) (h₄ : d = 28) (h₅ : e = 25)
  (h₆ : a * b / 2 + (b + c) * d / 2 = 995) : 
  ∃ (pentagon_area : ℝ), pentagon_area = 995 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l2766_276656


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l2766_276676

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l2766_276676


namespace NUMINAMATH_CALUDE_value_of_b_l2766_276631

theorem value_of_b (p q r : ℝ) (b : ℝ) 
  (h1 : p - q = 2) 
  (h2 : p - r = 1) 
  (h3 : b = (r - q) * ((p - q)^2 + (p - q)*(p - r) + (p - r)^2)) :
  b = 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l2766_276631


namespace NUMINAMATH_CALUDE_trapezoid_area_l2766_276616

/-- Given a trapezoid with bases a and b, prove that its area is 150 -/
theorem trapezoid_area (a b : ℝ) : 
  ((a + b) / 2) * ((a - b) / 2) = 25 →
  ∃ h : ℝ, h = 3 * (a - b) →
  (1 / 2) * (a + b) * h = 150 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2766_276616


namespace NUMINAMATH_CALUDE_sherman_weekly_driving_time_l2766_276680

/-- Calculates the total weekly driving time for Sherman given his commute and weekend driving schedules. -/
theorem sherman_weekly_driving_time 
  (weekday_commute_minutes : ℕ) -- Daily commute time (round trip) in minutes
  (weekdays : ℕ) -- Number of weekdays
  (weekend_driving_hours : ℕ) -- Daily weekend driving time in hours
  (weekend_days : ℕ) -- Number of weekend days
  (h1 : weekday_commute_minutes = 60) -- 30 minutes to office + 30 minutes back home
  (h2 : weekdays = 5) -- 5 weekdays in a week
  (h3 : weekend_driving_hours = 2) -- 2 hours of driving each weekend day
  (h4 : weekend_days = 2) -- 2 days in a weekend
  : (weekday_commute_minutes * weekdays) / 60 + weekend_driving_hours * weekend_days = 9 :=
by sorry

end NUMINAMATH_CALUDE_sherman_weekly_driving_time_l2766_276680


namespace NUMINAMATH_CALUDE_bridge_painting_l2766_276629

theorem bridge_painting (painted_section : ℚ) : 
  (1.2 * painted_section = 1/2) → 
  (1/2 - painted_section = 1/12) := by
sorry

end NUMINAMATH_CALUDE_bridge_painting_l2766_276629


namespace NUMINAMATH_CALUDE_boys_participation_fraction_l2766_276661

-- Define the total number of students
def total_students : ℕ := 800

-- Define the number of participating students
def participating_students : ℕ := 550

-- Define the number of participating girls
def participating_girls : ℕ := 150

-- Define the fraction of girls who participated
def girls_participation_fraction : ℚ := 3/4

-- Theorem to prove
theorem boys_participation_fraction :
  let total_girls : ℕ := participating_girls * 4 / 3
  let total_boys : ℕ := total_students - total_girls
  let participating_boys : ℕ := participating_students - participating_girls
  (participating_boys : ℚ) / total_boys = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_boys_participation_fraction_l2766_276661


namespace NUMINAMATH_CALUDE_school_population_l2766_276620

theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) : 
  b + g + t = 26 * t := by sorry

end NUMINAMATH_CALUDE_school_population_l2766_276620


namespace NUMINAMATH_CALUDE_sports_club_members_l2766_276647

theorem sports_club_members (badminton tennis both neither : ℕ) 
  (h1 : badminton = 18)
  (h2 : tennis = 19)
  (h3 : both = 9)
  (h4 : neither = 2) :
  badminton + tennis - both + neither = 30 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l2766_276647


namespace NUMINAMATH_CALUDE_iron_percentage_in_alloy_l2766_276659

/-- The percentage of alloy in the ore -/
def alloy_percentage : ℝ := 0.25

/-- The total amount of ore in kg -/
def total_ore : ℝ := 266.6666666666667

/-- The amount of pure iron obtained in kg -/
def pure_iron : ℝ := 60

/-- The percentage of iron in the alloy -/
def iron_percentage : ℝ := 0.9

theorem iron_percentage_in_alloy :
  alloy_percentage * total_ore * iron_percentage = pure_iron :=
sorry

end NUMINAMATH_CALUDE_iron_percentage_in_alloy_l2766_276659
