import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l1658_165854

/-- Represents a three-digit number with distinct digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The set of available digits -/
def availableDigits : Finset Nat := {5, 5, 6, 6, 6, 7, 8, 8, 9}

/-- Theorem stating the minimum value of A + B - C -/
theorem min_value_theorem (A B C : ThreeDigitNumber) 
  (h1 : A.hundreds ∈ availableDigits)
  (h2 : A.tens ∈ availableDigits)
  (h3 : A.ones ∈ availableDigits)
  (h4 : B.hundreds ∈ availableDigits)
  (h5 : B.tens ∈ availableDigits)
  (h6 : B.ones ∈ availableDigits)
  (h7 : C.hundreds ∈ availableDigits)
  (h8 : C.tens ∈ availableDigits)
  (h9 : C.ones ∈ availableDigits)
  (h10 : A.toNat + B.toNat - C.toNat ≥ 149) :
  ∃ (A' B' C' : ThreeDigitNumber),
    A'.hundreds ∈ availableDigits ∧
    A'.tens ∈ availableDigits ∧
    A'.ones ∈ availableDigits ∧
    B'.hundreds ∈ availableDigits ∧
    B'.tens ∈ availableDigits ∧
    B'.ones ∈ availableDigits ∧
    C'.hundreds ∈ availableDigits ∧
    C'.tens ∈ availableDigits ∧
    C'.ones ∈ availableDigits ∧
    A'.toNat + B'.toNat - C'.toNat = 149 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1658_165854


namespace NUMINAMATH_CALUDE_fixed_point_coordinates_l1658_165894

/-- Given that for any real number k, the line (3+k)x + (1-2k)y + 1 + 5k = 0
    passes through a fixed point A, prove that the coordinates of A are (-1, 2). -/
theorem fixed_point_coordinates (A : ℝ × ℝ) :
  (∀ k : ℝ, (3 + k) * A.1 + (1 - 2*k) * A.2 + 1 + 5*k = 0) →
  A = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_coordinates_l1658_165894


namespace NUMINAMATH_CALUDE_max_tokens_on_chessboard_max_tokens_on_chessboard_with_diagonals_l1658_165876

/-- Represents a chessboard configuration -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Checks if a chessboard configuration is valid according to the given constraints -/
def is_valid_configuration (board : Chessboard) : Prop :=
  -- At most 4 tokens per row
  (∀ row, (Finset.filter (λ col => board row col) Finset.univ).card ≤ 4) ∧
  -- At most 4 tokens per column
  (∀ col, (Finset.filter (λ row => board row col) Finset.univ).card ≤ 4)

/-- Checks if a chessboard configuration is valid including diagonal constraints -/
def is_valid_configuration_with_diagonals (board : Chessboard) : Prop :=
  is_valid_configuration board ∧
  -- At most 4 tokens on main diagonal
  (Finset.filter (λ i => board i i) Finset.univ).card ≤ 4 ∧
  -- At most 4 tokens on anti-diagonal
  (Finset.filter (λ i => board i (7 - i)) Finset.univ).card ≤ 4

/-- The total number of tokens on the board -/
def token_count (board : Chessboard) : Nat :=
  (Finset.filter (λ p => board p.1 p.2) (Finset.univ.product Finset.univ)).card

theorem max_tokens_on_chessboard :
  (∃ board : Chessboard, is_valid_configuration board ∧ token_count board = 32) ∧
  (∀ board : Chessboard, is_valid_configuration board → token_count board ≤ 32) :=
sorry

theorem max_tokens_on_chessboard_with_diagonals :
  (∃ board : Chessboard, is_valid_configuration_with_diagonals board ∧ token_count board = 32) ∧
  (∀ board : Chessboard, is_valid_configuration_with_diagonals board → token_count board ≤ 32) :=
sorry

end NUMINAMATH_CALUDE_max_tokens_on_chessboard_max_tokens_on_chessboard_with_diagonals_l1658_165876


namespace NUMINAMATH_CALUDE_concatenated_integers_divisible_by_55_l1658_165898

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of concatenating integers from 1 to n
  sorry

theorem concatenated_integers_divisible_by_55 :
  ∃ k : ℕ, concatenate_integers 55 = 55 * k := by
  sorry

end NUMINAMATH_CALUDE_concatenated_integers_divisible_by_55_l1658_165898


namespace NUMINAMATH_CALUDE_chinese_books_probability_l1658_165869

theorem chinese_books_probability (total_books : ℕ) (chinese_books : ℕ) (math_books : ℕ) :
  total_books = chinese_books + math_books →
  chinese_books = 3 →
  math_books = 2 →
  (Nat.choose chinese_books 2 : ℚ) / (Nat.choose total_books 2) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_chinese_books_probability_l1658_165869


namespace NUMINAMATH_CALUDE_koch_snowflake_area_l1658_165884

/-- Given a sequence of curves P₀, P₁, P₂, ..., where:
    1. P₀ is an equilateral triangle with area 1
    2. Pₖ₊₁ is obtained from Pₖ by trisecting each side, constructing an equilateral 
       triangle on the middle segment, and removing the middle segment
    3. Sₙ is the area enclosed by curve Pₙ
    
    This theorem states the formula for Sₙ and its limit as n approaches infinity. -/
theorem koch_snowflake_area (n : ℕ) : 
  ∃ (S : ℕ → ℝ), 
    (∀ k, S k = (47/20) * (1 - (4/9)^k)) ∧ 
    (∀ ε > 0, ∃ N, ∀ n ≥ N, |S n - 47/20| < ε) := by
  sorry

end NUMINAMATH_CALUDE_koch_snowflake_area_l1658_165884


namespace NUMINAMATH_CALUDE_unique_divisibility_pair_l1658_165895

/-- A predicate that checks if there are infinitely many positive integers k 
    for which (k^n + k^2 - 1) divides (k^m + k - 1) -/
def InfinitelyManyDivisors (m n : ℕ) : Prop :=
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ (k^n + k^2 - 1) ∣ (k^m + k - 1)

/-- The theorem stating that (5,3) is the only pair of integers (m,n) 
    satisfying the given conditions -/
theorem unique_divisibility_pair :
  ∀ m n : ℕ, m > 2 → n > 2 → InfinitelyManyDivisors m n → m = 5 ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_divisibility_pair_l1658_165895


namespace NUMINAMATH_CALUDE_increasing_sine_function_bound_l1658_165888

open Real

/-- A function f is increasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The main theorem: If f(x) = x + a*sin(x) is increasing on ℝ, then -1 ≤ a ≤ 1 -/
theorem increasing_sine_function_bound (a : ℝ) :
  IncreasingOn (fun x => x + a * sin x) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_sine_function_bound_l1658_165888


namespace NUMINAMATH_CALUDE_sailboat_speed_at_max_power_l1658_165826

/-- The speed of a sailboat when the wind power is maximized -/
theorem sailboat_speed_at_max_power 
  (C S ρ : ℝ) 
  (v₀ : ℝ) 
  (h_positive : C > 0 ∧ S > 0 ∧ ρ > 0 ∧ v₀ > 0) :
  ∃ (v : ℝ), 
    v = v₀ / 3 ∧ 
    (∀ (u : ℝ), 
      u * (C * S * ρ * (v₀ - u)^2) / 2 ≤ v * (C * S * ρ * (v₀ - v)^2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sailboat_speed_at_max_power_l1658_165826


namespace NUMINAMATH_CALUDE_ruel_stamps_count_l1658_165845

theorem ruel_stamps_count : ∀ (books_of_10 books_of_15 : ℕ),
  books_of_10 = 4 →
  books_of_15 = 6 →
  books_of_10 * 10 + books_of_15 * 15 = 130 :=
by
  sorry

end NUMINAMATH_CALUDE_ruel_stamps_count_l1658_165845


namespace NUMINAMATH_CALUDE_trigonometric_roots_theorem_l1658_165844

-- Define the equation
def equation (x m : ℝ) : Prop := 2 * x^2 - (Real.sqrt 3 - 1) * x + m = 0

-- Define the theorem
theorem trigonometric_roots_theorem (θ : ℝ) (m : ℝ) 
  (h1 : equation (Real.sin θ) m)
  (h2 : equation (Real.cos θ) m)
  (h3 : θ > 3/2 * Real.pi ∧ θ < 2 * Real.pi) :
  m = -Real.sqrt 3 / 2 ∧
  (Real.sin θ / (1 - 1 / Real.tan θ) + Real.cos θ / (1 - Real.tan θ) = (Real.sqrt 3 - 1) / 2) ∧
  Real.cos (2 * θ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_roots_theorem_l1658_165844


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1658_165805

theorem algebraic_simplification (y : ℝ) (h : y ≠ 0) :
  (-24 * y^3) * (5 * y^2) * (1 / (2*y)^3) = -15 * y^2 :=
sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1658_165805


namespace NUMINAMATH_CALUDE_minimal_tile_placement_l1658_165812

/-- Represents a tile placement on a grid -/
structure TilePlacement where
  tiles : ℕ
  grid_size : ℕ
  is_valid : Bool

/-- Checks if a tile placement is valid -/
def is_valid_placement (p : TilePlacement) : Prop :=
  p.is_valid ∧ 
  p.grid_size = 8 ∧ 
  p.tiles > 0 ∧ 
  p.tiles ≤ 32 ∧
  ∀ (t : TilePlacement), t.tiles < p.tiles → ¬t.is_valid

theorem minimal_tile_placement : 
  ∃ (p : TilePlacement), is_valid_placement p ∧ p.tiles = 28 := by
  sorry

end NUMINAMATH_CALUDE_minimal_tile_placement_l1658_165812


namespace NUMINAMATH_CALUDE_equation_arrangements_l1658_165885

def word : String := "equation"

def letter_count : Nat := word.length

theorem equation_arrangements :
  let distinct_letters : Nat := 8
  let qu_as_unit : Nat := 1
  let remaining_letters : Nat := distinct_letters - 2
  let units_to_arrange : Nat := qu_as_unit + remaining_letters
  let letters_to_select : Nat := 5 - 2
  let ways_to_select : Nat := Nat.choose remaining_letters letters_to_select
  let ways_to_arrange : Nat := Nat.factorial (letters_to_select + 1)
  ways_to_select * ways_to_arrange = 480 := by
  sorry

end NUMINAMATH_CALUDE_equation_arrangements_l1658_165885


namespace NUMINAMATH_CALUDE_pet_store_puppies_l1658_165870

theorem pet_store_puppies (sold : ℕ) (puppies_per_cage : ℕ) (num_cages : ℕ) :
  sold = 3 ∧ puppies_per_cage = 5 ∧ num_cages = 3 →
  sold + num_cages * puppies_per_cage = 18 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l1658_165870


namespace NUMINAMATH_CALUDE_exists_increasing_interval_l1658_165866

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := log x + 1 / log x

-- State the theorem
theorem exists_increasing_interval :
  ∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > x₀, deriv f x > 0 :=
sorry

end NUMINAMATH_CALUDE_exists_increasing_interval_l1658_165866


namespace NUMINAMATH_CALUDE_double_age_in_years_until_double_l1658_165883

/-- The number of years until I'm twice my brother's age -/
def years_until_double : ℕ := 10

/-- My current age -/
def my_current_age : ℕ := 20

/-- My brother's current age -/
def brothers_current_age : ℕ := my_current_age - years_until_double

theorem double_age_in_years_until_double :
  (my_current_age + years_until_double) = 2 * (brothers_current_age + years_until_double) ∧
  (my_current_age + years_until_double) + (brothers_current_age + years_until_double) = 45 :=
by sorry

end NUMINAMATH_CALUDE_double_age_in_years_until_double_l1658_165883


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1658_165899

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^2 - 3) - 4*(x^2 - 3*x + 6) = 8*x^3 - 4*x^2 + 6*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1658_165899


namespace NUMINAMATH_CALUDE_integer_solutions_of_system_l1658_165861

theorem integer_solutions_of_system (x y z t : ℤ) : 
  (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
  ((x = 1 ∧ y = 0 ∧ z = 3 ∧ t = 1) ∨
   (x = 1 ∧ y = 0 ∧ z = 3 ∧ t = -1) ∨
   (x = 1 ∧ y = 0 ∧ z = -3 ∧ t = 1) ∨
   (x = 1 ∧ y = 0 ∧ z = -3 ∧ t = -1) ∨
   (x = -1 ∧ y = 0 ∧ z = 3 ∧ t = 1) ∨
   (x = -1 ∧ y = 0 ∧ z = 3 ∧ t = -1) ∨
   (x = -1 ∧ y = 0 ∧ z = -3 ∧ t = 1) ∨
   (x = -1 ∧ y = 0 ∧ z = -3 ∧ t = -1) ∨
   (x = 3 ∧ y = 1 ∧ z = 1 ∧ t = 0) ∨
   (x = 3 ∧ y = 1 ∧ z = -1 ∧ t = 0) ∨
   (x = 3 ∧ y = -1 ∧ z = 1 ∧ t = 0) ∨
   (x = 3 ∧ y = -1 ∧ z = -1 ∧ t = 0) ∨
   (x = -3 ∧ y = 1 ∧ z = 1 ∧ t = 0) ∨
   (x = -3 ∧ y = 1 ∧ z = -1 ∧ t = 0) ∨
   (x = -3 ∧ y = -1 ∧ z = 1 ∧ t = 0) ∨
   (x = -3 ∧ y = -1 ∧ z = -1 ∧ t = 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_system_l1658_165861


namespace NUMINAMATH_CALUDE_rope_jumps_percentage_l1658_165806

def rope_jumps : List ℕ := [50, 77, 83, 91, 93, 101, 87, 102, 111, 63, 117, 89, 121, 130, 133, 146, 88, 158, 177, 188]

def total_students : ℕ := 20

def in_range (x : ℕ) : Bool := 80 ≤ x ∧ x ≤ 100

def count_in_range (l : List ℕ) : ℕ := (l.filter in_range).length

theorem rope_jumps_percentage :
  (count_in_range rope_jumps : ℚ) / total_students * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rope_jumps_percentage_l1658_165806


namespace NUMINAMATH_CALUDE_circle_line_tangent_l1658_165840

/-- A circle C in the xy-plane -/
def Circle (a : ℝ) (x y : ℝ) : Prop :=
  x^2 - 2*a*x + y^2 = 0

/-- A line l in the xy-plane -/
def Line (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + 3 = 0

/-- The circle and line are tangent if they intersect at exactly one point -/
def Tangent (a : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, Circle a p.1 p.2 ∧ Line p.1 p.2

theorem circle_line_tangent (a : ℝ) (h1 : a > 0) (h2 : Tangent a) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_tangent_l1658_165840


namespace NUMINAMATH_CALUDE_parallelism_sufficiency_not_necessity_l1658_165879

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ ≠ 0 ∧ m₂ ≠ 0

/-- The condition for parallelism of the given lines -/
def parallelism_condition (a : ℝ) : Prop :=
  are_parallel 2 a (-2) (a + 1) 1 (-a)

theorem parallelism_sufficiency_not_necessity :
  (∀ a : ℝ, a = 1 → parallelism_condition a) ∧
  ¬(∀ a : ℝ, parallelism_condition a → a = 1) :=
by sorry

end NUMINAMATH_CALUDE_parallelism_sufficiency_not_necessity_l1658_165879


namespace NUMINAMATH_CALUDE_apple_group_addition_l1658_165897

/-- Given a basket of apples divided among a group, prove the number of people who joined --/
theorem apple_group_addition (total_apples : ℕ) (original_per_person : ℕ) (new_per_person : ℕ) :
  total_apples = 1430 →
  original_per_person = 22 →
  new_per_person = 13 →
  ∃ (original_group : ℕ) (joined_group : ℕ),
    original_group * original_per_person = total_apples ∧
    (original_group + joined_group) * new_per_person = total_apples ∧
    joined_group = 45 := by
  sorry


end NUMINAMATH_CALUDE_apple_group_addition_l1658_165897


namespace NUMINAMATH_CALUDE_jake_snakes_l1658_165842

/-- The number of eggs each snake lays -/
def eggs_per_snake : ℕ := 2

/-- The price of a regular baby snake in dollars -/
def regular_price : ℕ := 250

/-- The price of the rare baby snake in dollars -/
def rare_price : ℕ := 4 * regular_price

/-- The total amount Jake received from selling the snakes in dollars -/
def total_revenue : ℕ := 2250

/-- The number of snakes Jake has -/
def num_snakes : ℕ := 3

theorem jake_snakes :
  num_snakes * eggs_per_snake * regular_price + (rare_price - regular_price) = total_revenue :=
sorry

end NUMINAMATH_CALUDE_jake_snakes_l1658_165842


namespace NUMINAMATH_CALUDE_prob_sum_eight_two_dice_l1658_165810

def dice_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ := 5

theorem prob_sum_eight_two_dice : 
  (favorable_outcomes : ℚ) / dice_outcomes = 5 / 36 := by sorry

end NUMINAMATH_CALUDE_prob_sum_eight_two_dice_l1658_165810


namespace NUMINAMATH_CALUDE_maximum_marks_l1658_165828

theorem maximum_marks (victor_marks : ℕ) (max_marks : ℕ) (h1 : victor_marks = 368) (h2 : 92 * max_marks = 100 * victor_marks) : max_marks = 400 := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_l1658_165828


namespace NUMINAMATH_CALUDE_triangle_side_ratio_bounds_l1658_165814

theorem triangle_side_ratio_bounds (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  let t := (a + b + c) / Real.sqrt (a * b + b * c + c * a)
  Real.sqrt 3 ≤ t ∧ t < 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_bounds_l1658_165814


namespace NUMINAMATH_CALUDE_seating_arrangements_l1658_165893

/-- The number of seats in the row -/
def total_seats : ℕ := 8

/-- The number of people to be seated -/
def people : ℕ := 3

/-- The number of seats that must be left empty at the ends -/
def end_seats : ℕ := 2

/-- The number of seats available for seating after accounting for end seats -/
def available_seats : ℕ := total_seats - end_seats

/-- The number of gaps between seated people (including before first and after last) -/
def gaps : ℕ := people + 1

/-- Theorem stating the number of seating arrangements -/
theorem seating_arrangements :
  (Nat.choose available_seats gaps) * (Nat.factorial people) = 24 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1658_165893


namespace NUMINAMATH_CALUDE_remainder_7459_div_9_l1658_165862

theorem remainder_7459_div_9 : 
  7459 % 9 = (7 + 4 + 5 + 9) % 9 := by sorry

end NUMINAMATH_CALUDE_remainder_7459_div_9_l1658_165862


namespace NUMINAMATH_CALUDE_range_a_theorem_l1658_165873

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -2 ∨ a = 1

-- State the theorem
theorem range_a_theorem : 
  ∀ a : ℝ, (p a ∧ q a) → range_of_a a := by
  sorry

end NUMINAMATH_CALUDE_range_a_theorem_l1658_165873


namespace NUMINAMATH_CALUDE_eighth_term_value_l1658_165847

theorem eighth_term_value (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, S n = n^2) : a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l1658_165847


namespace NUMINAMATH_CALUDE_smallest_positive_b_squared_l1658_165827

-- Define the circles w₁ and w₂
def w₁ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 6*y - 23 = 0
def w₂ (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 6*y + 41 = 0

-- Define the condition for a point (x, y) to be on the line y = bx
def on_line (x y b : ℝ) : Prop := y = b * x

-- Define the condition for a circle to be externally tangent to w₂ and internally tangent to w₁
def tangent_condition (x y r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    w₁ x₁ y₁ ∧ w₂ x₂ y₂ ∧
    (x - x₂)^2 + (y - y₂)^2 = (r + Real.sqrt 10)^2 ∧
    (x - x₁)^2 + (y - y₁)^2 = (Real.sqrt 50 - r)^2

-- Main theorem
theorem smallest_positive_b_squared (b : ℝ) :
  (∀ b' : ℝ, b' > 0 ∧ b' < b →
    ¬∃ (x y r : ℝ), on_line x y b' ∧ tangent_condition x y r) →
  (∃ (x y r : ℝ), on_line x y b ∧ tangent_condition x y r) →
  b^2 = 21/16 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_b_squared_l1658_165827


namespace NUMINAMATH_CALUDE_range_of_a_range_of_b_l1658_165858

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a + 3
def g (b : ℝ) (x : ℝ) : ℝ := b*x + 5 - 2*b

-- Theorem for part 1
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 0) →
  -8 ≤ a ∧ a ≤ 0 :=
sorry

-- Theorem for part 2
theorem range_of_b (b : ℝ) :
  (∀ x₁ ∈ Set.Icc (1 : ℝ) 4, ∃ x₂ ∈ Set.Icc (1 : ℝ) 4, g b x₁ = f 3 x₂) →
  -1 ≤ b ∧ b ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_b_l1658_165858


namespace NUMINAMATH_CALUDE_factor_tree_value_l1658_165821

/-- Given a factor tree with the following relationships:
  X = Y * Z
  Y = 7 * F
  Z = 11 * G
  F = 7 * 2
  G = 3 * 2
  Prove that X = 12936 -/
theorem factor_tree_value (X Y Z F G : ℕ) 
  (h1 : X = Y * Z)
  (h2 : Y = 7 * F)
  (h3 : Z = 11 * G)
  (h4 : F = 7 * 2)
  (h5 : G = 3 * 2) : 
  X = 12936 := by
  sorry

#check factor_tree_value

end NUMINAMATH_CALUDE_factor_tree_value_l1658_165821


namespace NUMINAMATH_CALUDE_octal_minus_septenary_in_decimal_l1658_165852

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem octal_minus_septenary_in_decimal : 
  let octal := [2, 1, 3]
  let septenary := [1, 4, 2]
  to_base_10 octal 8 - to_base_10 septenary 7 = 60 := by
  sorry


end NUMINAMATH_CALUDE_octal_minus_septenary_in_decimal_l1658_165852


namespace NUMINAMATH_CALUDE_composition_equality_l1658_165834

/-- Given two functions f and g, prove that their composition at x = 3 equals 103 -/
theorem composition_equality (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 4 * x + 3) 
  (hg : ∀ x, g x = (x + 2) ^ 2) : 
  f (g 3) = 103 := by
  sorry

end NUMINAMATH_CALUDE_composition_equality_l1658_165834


namespace NUMINAMATH_CALUDE_a_equals_three_iff_parallel_not_coincident_l1658_165820

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ × ℝ → Prop := fun (x, y) ↦ a * x + 2 * y + 3 * a = 0
  line2 : ℝ × ℝ → Prop := fun (x, y) ↦ 3 * x + (a - 1) * y + 7 - a = 0

/-- Condition for two lines to be parallel and not coincident -/
def parallel_not_coincident (l : TwoLines) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    (∀ (x y : ℝ), l.line1 (x, y) ↔ l.line2 (k * x + l.a, k * y + 2))

/-- The main theorem -/
theorem a_equals_three_iff_parallel_not_coincident (l : TwoLines) :
  l.a = 3 ↔ parallel_not_coincident l :=
sorry

end NUMINAMATH_CALUDE_a_equals_three_iff_parallel_not_coincident_l1658_165820


namespace NUMINAMATH_CALUDE_dice_prob_same_color_l1658_165851

def prob_same_color (d1_sides d2_sides : ℕ)
  (d1_maroon d1_teal d1_cyan d1_sparkly : ℕ)
  (d2_maroon d2_teal d2_cyan d2_sparkly : ℕ) : ℚ :=
  let p_maroon := (d1_maroon : ℚ) / d1_sides * (d2_maroon : ℚ) / d2_sides
  let p_teal := (d1_teal : ℚ) / d1_sides * (d2_teal : ℚ) / d2_sides
  let p_cyan := (d1_cyan : ℚ) / d1_sides * (d2_cyan : ℚ) / d2_sides
  let p_sparkly := (d1_sparkly : ℚ) / d1_sides * (d2_sparkly : ℚ) / d2_sides
  p_maroon + p_teal + p_cyan + p_sparkly

theorem dice_prob_same_color :
  prob_same_color 20 16 5 8 6 1 4 6 5 1 = 99 / 320 := by
  sorry

end NUMINAMATH_CALUDE_dice_prob_same_color_l1658_165851


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l1658_165867

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l1658_165867


namespace NUMINAMATH_CALUDE_prob_less_than_one_third_l1658_165896

/-- The probability that a number randomly selected from (0, 1/2) is less than 1/3 is 2/3. -/
theorem prob_less_than_one_third : 
  ∀ (P : Set ℝ → ℝ) (Ω : Set ℝ),
    (∀ a b, a < b → P (Set.Ioo a b) = b - a) →  -- P is a uniform probability measure
    Ω = Set.Ioo 0 (1/2) →                       -- Ω is the interval (0, 1/2)
    P {x ∈ Ω | x < 1/3} / P Ω = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_one_third_l1658_165896


namespace NUMINAMATH_CALUDE_negative_one_fourth_minus_bracket_l1658_165860

theorem negative_one_fourth_minus_bracket : -1^4 - (2 - (-3)^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_fourth_minus_bracket_l1658_165860


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1658_165836

/-- An ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure Ellipse (k : ℝ) where
  eq : ∀ (x y : ℝ), x^2 + k * y^2 = 2
  foci_on_y : True  -- This is a placeholder for the foci condition

/-- The range of k for an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
theorem ellipse_k_range (k : ℝ) (e : Ellipse k) : 0 < k ∧ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1658_165836


namespace NUMINAMATH_CALUDE_star_calculation_l1658_165864

def star (a b : ℚ) : ℚ := (a + b) / 4

theorem star_calculation : star (star 3 8) 6 = 35 / 16 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l1658_165864


namespace NUMINAMATH_CALUDE_rob_doubles_l1658_165846

/-- Rob has some baseball cards, and Jess has 5 times as many doubles as Rob. 
    Jess has 40 doubles baseball cards. -/
theorem rob_doubles (rob_cards : ℕ) (rob_doubles : ℕ) (jess_doubles : ℕ) 
    (h1 : rob_cards ≥ rob_doubles)
    (h2 : jess_doubles = 5 * rob_doubles)
    (h3 : jess_doubles = 40) : 
  rob_doubles = 8 := by
  sorry

end NUMINAMATH_CALUDE_rob_doubles_l1658_165846


namespace NUMINAMATH_CALUDE_max_rectangles_l1658_165818

/-- Represents a cell in the figure -/
inductive Cell
| White
| Black

/-- Represents the figure as a 2D array of cells -/
def Figure := Array (Array Cell)

/-- Checks if a figure has alternating black and white cells -/
def hasAlternatingColors (fig : Figure) : Prop := sorry

/-- Checks if the middle diagonal of a figure is black -/
def hasBlackDiagonal (fig : Figure) : Prop := sorry

/-- Counts the number of black cells in a figure -/
def countBlackCells (fig : Figure) : Nat := sorry

/-- Represents a 1x2 rectangle placement in the figure -/
structure Rectangle where
  row : Nat
  col : Nat

/-- Checks if a rectangle placement is valid (spans one black and one white cell) -/
def isValidRectangle (fig : Figure) (rect : Rectangle) : Prop := sorry

/-- The main theorem -/
theorem max_rectangles (fig : Figure) 
  (h1 : hasAlternatingColors fig)
  (h2 : hasBlackDiagonal fig) :
  (∃ (rects : List Rectangle), 
    (∀ r ∈ rects, isValidRectangle fig r) ∧ 
    rects.length = countBlackCells fig) ∧
  (∀ (rects : List Rectangle), 
    (∀ r ∈ rects, isValidRectangle fig r) → 
    rects.length ≤ countBlackCells fig) := by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_l1658_165818


namespace NUMINAMATH_CALUDE_hotel_rooms_available_l1658_165889

theorem hotel_rooms_available (total_floors : ℕ) (rooms_per_floor : ℕ) (unavailable_floors : ℕ) :
  total_floors = 10 →
  rooms_per_floor = 10 →
  unavailable_floors = 1 →
  (total_floors - unavailable_floors) * rooms_per_floor = 90 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_available_l1658_165889


namespace NUMINAMATH_CALUDE_arccos_sin_three_l1658_165868

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_three_l1658_165868


namespace NUMINAMATH_CALUDE_inequality_problems_l1658_165863

theorem inequality_problems (x : ℝ) :
  ((-x^2 + 4*x - 4 < 0) ↔ (x ≠ 2)) ∧
  ((((1 - x) / (x - 5)) > 0) ↔ (1 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problems_l1658_165863


namespace NUMINAMATH_CALUDE_horner_eval_not_28_l1658_165822

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 2x^4 + 3x^3 + 4x^2 + 5x + 6 -/
def f_coeffs : List ℝ := [1, 2, 3, 4, 5, 6]

/-- Theorem: 28 does not appear in Horner's method calculation for f(2) -/
theorem horner_eval_not_28 :
  ∀ n : ℕ, n ≤ 5 →
  (horner_eval (f_coeffs.take n) 2) ≠ 28 :=
by
  sorry

#eval horner_eval f_coeffs 2

end NUMINAMATH_CALUDE_horner_eval_not_28_l1658_165822


namespace NUMINAMATH_CALUDE_modulo_17_intercepts_l1658_165881

/-- Prove the x-intercept, y-intercept, and their sum for the equation 5x ≡ 3y - 1 (mod 17) -/
theorem modulo_17_intercepts :
  ∃ (x₀ y₀ : ℕ), 
    x₀ < 17 ∧ 
    y₀ < 17 ∧
    (5 * x₀) % 17 = 16 ∧ 
    (3 * y₀) % 17 = 1 ∧
    x₀ = 1 ∧ 
    y₀ = 6 ∧ 
    x₀ + y₀ = 7 :=
by sorry

end NUMINAMATH_CALUDE_modulo_17_intercepts_l1658_165881


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_5_intersection_equals_b_iff_a_in_range_l1658_165878

def A : Set ℝ := {x | 1 < x - 1 ∧ x - 1 < 3}
def B (a : ℝ) : Set ℝ := {x | (x - 3) * (x - a) < 0}

theorem intersection_and_union_when_a_is_5 :
  (A ∩ B 5 = {x | 3 < x ∧ x < 4}) ∧
  (A ∪ B 5 = {x | 2 < x ∧ x < 5}) := by sorry

theorem intersection_equals_b_iff_a_in_range :
  ∀ a : ℝ, A ∩ B a = B a ↔ 2 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_5_intersection_equals_b_iff_a_in_range_l1658_165878


namespace NUMINAMATH_CALUDE_special_sequence_sum_5_l1658_165829

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  roots_property : a 2 * a 4 = 3 ∧ a 2 + a 4 = 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : SpecialArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem: S_5 = 5/2 for the special arithmetic sequence -/
theorem special_sequence_sum_5 (seq : SpecialArithmeticSequence) : 
  sum_n seq 5 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_sum_5_l1658_165829


namespace NUMINAMATH_CALUDE_scooter_distance_l1658_165811

/-- Proves that a scooter traveling 5/8 as fast as a motorcycle going 96 miles per hour will cover 40 miles in 40 minutes. -/
theorem scooter_distance (motorcycle_speed : ℝ) (scooter_ratio : ℝ) (travel_time : ℝ) :
  motorcycle_speed = 96 →
  scooter_ratio = 5/8 →
  travel_time = 40/60 →
  scooter_ratio * motorcycle_speed * travel_time = 40 :=
by sorry

end NUMINAMATH_CALUDE_scooter_distance_l1658_165811


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1658_165850

theorem triangle_angle_measure (angle1 angle2 angle3 angle4 : ℝ) :
  angle1 = 34 →
  angle2 = 53 →
  angle3 = 27 →
  angle1 + angle2 + angle3 + angle4 = 180 →
  angle4 = 114 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1658_165850


namespace NUMINAMATH_CALUDE_R_24_divisible_by_R_4_Q_only_ones_and_zeros_zeros_in_Q_l1658_165856

/-- R_k represents an integer whose decimal representation consists of k consecutive 1s -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- The quotient Q is defined as R_24 divided by R_4 -/
def Q : ℕ := R 24 / R 4

/-- count_zeros counts the number of zeros in the decimal representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem R_24_divisible_by_R_4 : R 24 % R 4 = 0 := sorry

theorem Q_only_ones_and_zeros : ∀ d : ℕ, d ∈ Q.digits 10 → d = 0 ∨ d = 1 := sorry

theorem zeros_in_Q : count_zeros Q = 15 := by sorry

end NUMINAMATH_CALUDE_R_24_divisible_by_R_4_Q_only_ones_and_zeros_zeros_in_Q_l1658_165856


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1658_165887

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 5 > 2*x} = {x : ℝ | x > 5} ∪ {x : ℝ | x < -1} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1658_165887


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1658_165853

theorem triangle_angle_proof (a b c A B C : ℝ) (S_ABC : ℝ) : 
  b = 2 →
  S_ABC = 2 * Real.sqrt 3 →
  c * Real.cos B + b * Real.cos C - 2 * a * Real.cos A = 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  S_ABC = (1 / 2) * a * b * Real.sin C →
  S_ABC = (1 / 2) * b * c * Real.sin A →
  S_ABC = (1 / 2) * c * a * Real.sin B →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C →
  C = π / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1658_165853


namespace NUMINAMATH_CALUDE_man_to_boy_work_ratio_l1658_165872

/-- The daily work done by a man -/
def M : ℝ := sorry

/-- The daily work done by a boy -/
def B : ℝ := sorry

/-- The total amount of work to be done -/
def total_work : ℝ := sorry

/-- The first condition: 12 men and 16 boys can do the work in 5 days -/
axiom condition1 : 5 * (12 * M + 16 * B) = total_work

/-- The second condition: 13 men and 24 boys can do the work in 4 days -/
axiom condition2 : 4 * (13 * M + 24 * B) = total_work

/-- The theorem stating that the ratio of daily work done by a man to that of a boy is 2:1 -/
theorem man_to_boy_work_ratio : M / B = 2 := by sorry

end NUMINAMATH_CALUDE_man_to_boy_work_ratio_l1658_165872


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l1658_165886

-- Define the total number of people that can ride at once
def total_riders : ℕ := 4

-- Define the capacity of each seat
def seat_capacity : ℕ := 2

-- Define the number of seats on the Ferris wheel
def num_seats : ℕ := total_riders / seat_capacity

-- Theorem statement
theorem ferris_wheel_seats : num_seats = 2 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l1658_165886


namespace NUMINAMATH_CALUDE_complement_of_A_l1658_165807

-- Define the universal set U
def U : Finset ℕ := {1,2,3,4,5,6,7}

-- Define set A
def A : Finset ℕ := Finset.filter (fun x => 1 ≤ x ∧ x ≤ 6) U

-- Theorem statement
theorem complement_of_A : (U \ A) = {7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1658_165807


namespace NUMINAMATH_CALUDE_correct_mark_is_90_l1658_165819

/-- Proves that the correct mark is 90 given the problem conditions --/
theorem correct_mark_is_90 (n : ℕ) (initial_avg correct_avg wrong_mark : ℚ) :
  n = 10 →
  initial_avg = 100 →
  correct_avg = 96 →
  wrong_mark = 50 →
  ∃ x : ℚ, (n * initial_avg - wrong_mark + x) / n = correct_avg ∧ x = 90 :=
by sorry

end NUMINAMATH_CALUDE_correct_mark_is_90_l1658_165819


namespace NUMINAMATH_CALUDE_fundraising_goal_l1658_165824

/-- Fundraising goal calculation for a school's community outreach program -/
theorem fundraising_goal (families_20 families_10 families_5 : ℕ) 
  (donation_20 donation_10 donation_5 : ℕ) (additional_needed : ℕ) : 
  families_20 = 2 → 
  families_10 = 8 → 
  families_5 = 10 → 
  donation_20 = 20 → 
  donation_10 = 10 → 
  donation_5 = 5 → 
  additional_needed = 30 → 
  families_20 * donation_20 + families_10 * donation_10 + families_5 * donation_5 + additional_needed = 200 := by
sorry

#eval 2 * 20 + 8 * 10 + 10 * 5 + 30

end NUMINAMATH_CALUDE_fundraising_goal_l1658_165824


namespace NUMINAMATH_CALUDE_max_value_of_f_l1658_165848

/-- The quadratic function f(x) = -2x^2 + 4x - 6 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 6

/-- The maximum value of f(x) is -4 -/
theorem max_value_of_f :
  ∃ (m : ℝ), m = -4 ∧ ∀ (x : ℝ), f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1658_165848


namespace NUMINAMATH_CALUDE_opposite_sign_sum_three_l1658_165832

theorem opposite_sign_sum_three (x y : ℝ) :
  (|x^2 - 4*x + 4| * (2*x - y - 3).sqrt < 0) →
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_opposite_sign_sum_three_l1658_165832


namespace NUMINAMATH_CALUDE_missing_digit_in_103rd_rising_number_l1658_165804

/-- A rising number is a positive integer each digit of which is larger than each of the digits to its left. -/
def IsRisingNumber (n : ℕ) : Prop := sorry

/-- The set of all five-digit rising numbers using digits from 1 to 9. -/
def FiveDigitRisingNumbers : Set ℕ := {n : ℕ | IsRisingNumber n ∧ n ≥ 10000 ∧ n < 100000}

/-- The 103rd element in the ordered set of five-digit rising numbers. -/
def OneHundredThirdRisingNumber : ℕ := sorry

theorem missing_digit_in_103rd_rising_number :
  ¬ (∃ (d : ℕ), d = 5 ∧ 10 * (OneHundredThirdRisingNumber / 10) + d = OneHundredThirdRisingNumber) :=
sorry

end NUMINAMATH_CALUDE_missing_digit_in_103rd_rising_number_l1658_165804


namespace NUMINAMATH_CALUDE_maddie_tshirt_cost_l1658_165843

/-- Calculates the total cost of T-shirts bought by Maddie -/
def total_cost (white_packs blue_packs : ℕ) (white_per_pack blue_per_pack : ℕ) (cost_per_shirt : ℕ) : ℕ :=
  let total_shirts := white_packs * white_per_pack + blue_packs * blue_per_pack
  total_shirts * cost_per_shirt

/-- Theorem stating that Maddie spent $66 on T-shirts -/
theorem maddie_tshirt_cost :
  total_cost 2 4 5 3 3 = 66 := by
  sorry

end NUMINAMATH_CALUDE_maddie_tshirt_cost_l1658_165843


namespace NUMINAMATH_CALUDE_remainder_of_n_l1658_165874

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 5 = 1) (h2 : n^4 % 5 = 1) :
  n % 5 = 1 ∨ n % 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_l1658_165874


namespace NUMINAMATH_CALUDE_original_photo_dimensions_l1658_165892

/-- Represents the dimensions of a rectangular photo frame --/
structure PhotoFrame where
  width : ℕ
  height : ℕ

/-- Calculates the number of squares needed for a frame --/
def squares_for_frame (frame : PhotoFrame) : ℕ :=
  2 * (frame.width + frame.height)

/-- Theorem stating the dimensions of the original photo --/
theorem original_photo_dimensions 
  (original_squares : ℕ) 
  (cut_squares : ℕ) 
  (h1 : original_squares = 1812)
  (h2 : cut_squares = 2018) :
  ∃ (frame : PhotoFrame), 
    squares_for_frame frame = original_squares ∧ 
    frame.width = 803 ∧ 
    frame.height = 101 ∧
    cut_squares - original_squares = 2 * frame.height :=
sorry


end NUMINAMATH_CALUDE_original_photo_dimensions_l1658_165892


namespace NUMINAMATH_CALUDE_tabitha_honey_days_l1658_165890

/-- Represents the number of days Tabitha can enjoy honey in her tea --/
def honey_days (servings_per_cup : ℕ) (evening_cups : ℕ) (morning_cups : ℕ) 
               (container_ounces : ℕ) (servings_per_ounce : ℕ) : ℕ :=
  (container_ounces * servings_per_ounce) / (servings_per_cup * (evening_cups + morning_cups))

/-- Theorem stating that Tabitha can enjoy honey in her tea for 32 days --/
theorem tabitha_honey_days : 
  honey_days 1 2 1 16 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_tabitha_honey_days_l1658_165890


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l1658_165802

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- all terms are positive
  ∃ d : ℝ, a = b - d ∧ c = b + d →  -- forms an arithmetic sequence
  a * b * c = 125 →  -- product is 125
  b ≥ 5 ∧ (∀ b' : ℝ, b' ≥ 5 → b' = 5) :=  -- b is at least 5, and 5 is the smallest such value
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l1658_165802


namespace NUMINAMATH_CALUDE_vertex_locus_is_circle_l1658_165801

/-- A triangle with a fixed base and a median of constant length --/
structure TriangleWithMedian where
  /-- The length of the fixed base AB --/
  base_length : ℝ
  /-- The length of the median from A to side BC --/
  median_length : ℝ

/-- The locus of vertex C in a triangle with a fixed base and constant median length --/
def vertex_locus (t : TriangleWithMedian) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {p | ∃ (A B : EuclideanSpace ℝ (Fin 2)), 
    ‖B - A‖ = t.base_length ∧ 
    ‖p - A‖ = t.median_length}

/-- The theorem stating that the locus of vertex C is a circle --/
theorem vertex_locus_is_circle (t : TriangleWithMedian) 
  (h : t.base_length = 6 ∧ t.median_length = 3) : 
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ),
    vertex_locus t = {p | ‖p - center‖ = radius} ∧ radius = 3 :=
sorry

end NUMINAMATH_CALUDE_vertex_locus_is_circle_l1658_165801


namespace NUMINAMATH_CALUDE_zoo_total_revenue_l1658_165891

def monday_children : Nat := 7
def monday_adults : Nat := 5
def tuesday_children : Nat := 4
def tuesday_adults : Nat := 2
def child_ticket_cost : Nat := 3
def adult_ticket_cost : Nat := 4

theorem zoo_total_revenue : 
  (monday_children + tuesday_children) * child_ticket_cost + 
  (monday_adults + tuesday_adults) * adult_ticket_cost = 61 := by
  sorry

#eval (monday_children + tuesday_children) * child_ticket_cost + 
      (monday_adults + tuesday_adults) * adult_ticket_cost

end NUMINAMATH_CALUDE_zoo_total_revenue_l1658_165891


namespace NUMINAMATH_CALUDE_jacket_markup_percentage_l1658_165831

theorem jacket_markup_percentage 
  (purchase_price : ℝ)
  (selling_price : ℝ)
  (markup_percentage : ℝ)
  (discount_rate : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 56)
  (h2 : selling_price = purchase_price + markup_percentage * selling_price)
  (h3 : discount_rate = 0.2)
  (h4 : gross_profit = 8)
  (h5 : gross_profit = (1 - discount_rate) * selling_price - purchase_price) :
  markup_percentage = 0.3 := by
sorry

end NUMINAMATH_CALUDE_jacket_markup_percentage_l1658_165831


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_40_l1658_165817

theorem consecutive_integers_around_sqrt_40 (a b : ℤ) : 
  (a + 1 = b) → (a < Real.sqrt 40) → (Real.sqrt 40 < b) → (a + b = 13) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_40_l1658_165817


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1658_165871

theorem complex_modulus_problem (z : ℂ) : 
  z = (1 + 2*I)^2 / (-I + 2) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1658_165871


namespace NUMINAMATH_CALUDE_inequality_system_solution_iff_l1658_165833

theorem inequality_system_solution_iff (a : ℝ) :
  (∃ x : ℝ, x ≥ -1 ∧ 2 * x < a) ↔ a > -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_iff_l1658_165833


namespace NUMINAMATH_CALUDE_picasso_paintings_probability_l1658_165835

/-- The probability of placing 4 Picasso paintings consecutively among 12 art pieces -/
theorem picasso_paintings_probability (total_pieces : ℕ) (picasso_paintings : ℕ) :
  total_pieces = 12 →
  picasso_paintings = 4 →
  (picasso_paintings.factorial * (total_pieces - picasso_paintings + 1).factorial) / total_pieces.factorial = 1 / 55 :=
by sorry

end NUMINAMATH_CALUDE_picasso_paintings_probability_l1658_165835


namespace NUMINAMATH_CALUDE_john_pill_payment_john_pays_54_dollars_l1658_165839

/-- The amount John pays for pills in a 30-day month, given the specified conditions. -/
theorem john_pill_payment (pills_per_day : ℕ) (cost_per_pill : ℚ) 
  (insurance_coverage_percent : ℚ) (days_in_month : ℕ) : ℚ :=
  let total_cost := (pills_per_day : ℚ) * cost_per_pill * days_in_month
  let insurance_coverage := total_cost * (insurance_coverage_percent / 100)
  total_cost - insurance_coverage

/-- Proof that John pays $54 for his pills in a 30-day month. -/
theorem john_pays_54_dollars : 
  john_pill_payment 2 (3/2) 40 30 = 54 := by
  sorry

end NUMINAMATH_CALUDE_john_pill_payment_john_pays_54_dollars_l1658_165839


namespace NUMINAMATH_CALUDE_square_difference_equality_l1658_165816

theorem square_difference_equality : 1004^2 - 996^2 - 1000^2 + 1000^2 = 16000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1658_165816


namespace NUMINAMATH_CALUDE_ratio_e_to_f_l1658_165875

theorem ratio_e_to_f (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.75) :
  e / f = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_e_to_f_l1658_165875


namespace NUMINAMATH_CALUDE_janessas_gift_to_dexter_l1658_165859

def cards_given_to_dexter (initial_cards : ℕ) (cards_from_father : ℕ) (cards_ordered : ℕ) (bad_cards : ℕ) (cards_kept : ℕ) : ℕ :=
  initial_cards + cards_from_father + cards_ordered - bad_cards - cards_kept

theorem janessas_gift_to_dexter :
  cards_given_to_dexter 4 13 36 4 20 = 29 := by
  sorry

end NUMINAMATH_CALUDE_janessas_gift_to_dexter_l1658_165859


namespace NUMINAMATH_CALUDE_gcd_100_450_l1658_165882

theorem gcd_100_450 : Nat.gcd 100 450 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gcd_100_450_l1658_165882


namespace NUMINAMATH_CALUDE_metaPopulation2050_l1658_165803

-- Define the initial population and year
def initialPopulation : ℕ := 150
def initialYear : ℕ := 2005

-- Define the doubling period and target year
def doublingPeriod : ℕ := 20
def targetYear : ℕ := 2050

-- Define the population growth function
def populationGrowth (years : ℕ) : ℕ :=
  initialPopulation * (2 ^ (years / doublingPeriod))

-- Theorem statement
theorem metaPopulation2050 :
  populationGrowth (targetYear - initialYear) = 600 := by
  sorry

end NUMINAMATH_CALUDE_metaPopulation2050_l1658_165803


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l1658_165880

theorem least_subtrahend_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem problem_solution :
  let n := 13603
  let d := 87
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 ∧ k = 31 :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l1658_165880


namespace NUMINAMATH_CALUDE_pushkin_pension_is_survivor_l1658_165800

-- Define the types of pensions
inductive PensionType
| Retirement
| Disability
| Survivor

-- Define a structure for a pension
structure Pension where
  recipient : String
  year_assigned : Nat
  is_lifelong : Bool
  type : PensionType

-- Define Pushkin's family pension
def pushkin_family_pension : Pension :=
  { recipient := "Pushkin's wife and daughters"
  , year_assigned := 1837
  , is_lifelong := true
  , type := PensionType.Survivor }

-- Theorem statement
theorem pushkin_pension_is_survivor :
  pushkin_family_pension.type = PensionType.Survivor :=
by sorry

end NUMINAMATH_CALUDE_pushkin_pension_is_survivor_l1658_165800


namespace NUMINAMATH_CALUDE_manuscript_cost_calculation_l1658_165857

/-- Calculates the total cost of typing a manuscript with given conditions -/
def manuscript_typing_cost (total_pages : ℕ) (first_typing_rate : ℕ) (revision_rate : ℕ) 
  (pages_revised_once : ℕ) (pages_revised_twice : ℕ) : ℕ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost := total_pages * first_typing_rate
  let first_revision_cost := pages_revised_once * revision_rate
  let second_revision_cost := pages_revised_twice * revision_rate * 2
  initial_typing_cost + first_revision_cost + second_revision_cost

theorem manuscript_cost_calculation :
  manuscript_typing_cost 100 10 5 30 20 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_calculation_l1658_165857


namespace NUMINAMATH_CALUDE_largest_band_size_l1658_165877

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (original : BandFormation) (total : ℕ) : Prop :=
  total < 100 ∧
  total = original.rows * original.membersPerRow + 3 ∧
  total = (original.rows - 3) * (original.membersPerRow + 1)

/-- Finds the largest valid band formation --/
def largestValidFormation : Option (BandFormation × ℕ) :=
  sorry

theorem largest_band_size :
  ∀ bf : BandFormation,
  ∀ m : ℕ,
  isValidFormation bf m →
  m ≤ 75 :=
sorry

end NUMINAMATH_CALUDE_largest_band_size_l1658_165877


namespace NUMINAMATH_CALUDE_function_property_l1658_165823

/-- Given a function f(x) = a ln x + bx + 1, prove that a - b = 10 under specific conditions -/
theorem function_property (a b : ℝ) : 
  (∀ x, x > 0 → ∃ f : ℝ → ℝ, f x = a * Real.log x + b * x + 1) → 
  (∃ f' : ℝ → ℝ, ∀ x, x > 0 → f' x = a / x + b) →
  (a + b = -2) →
  (3/2 * a + b = 0) →
  a - b = 10 := by
sorry

end NUMINAMATH_CALUDE_function_property_l1658_165823


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1658_165813

theorem trigonometric_equation_solution (z : ℂ) : 
  (Complex.sin z + Complex.sin (2 * z) + Complex.sin (3 * z) = 
   Complex.cos z + Complex.cos (2 * z) + Complex.cos (3 * z)) ↔ 
  (∃ (k : ℤ), z = (2 / 3 : ℂ) * π * (3 * k + 1) ∨ z = (2 / 3 : ℂ) * π * (3 * k - 1)) ∨
  (∃ (n : ℤ), z = (π / 8 : ℂ) * (4 * n + 1)) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1658_165813


namespace NUMINAMATH_CALUDE_initial_price_increase_l1658_165865

theorem initial_price_increase (P : ℝ) (x : ℝ) : 
  P * (1 + x / 100) * (1 - 10 / 100) = P * (1 + 12.5 / 100) → 
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_initial_price_increase_l1658_165865


namespace NUMINAMATH_CALUDE_sum_of_three_squares_squared_l1658_165837

theorem sum_of_three_squares_squared (a b c : ℕ) :
  ∃ (x y z : ℕ), (a^2 + b^2 + c^2)^2 = x^2 + y^2 + z^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_squared_l1658_165837


namespace NUMINAMATH_CALUDE_tan_75_degrees_l1658_165815

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_75_degrees_l1658_165815


namespace NUMINAMATH_CALUDE_range_of_f_l1658_165838

noncomputable def f (x : ℝ) : ℝ := 1 - x - 9 / x

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ 0 ∧ f x = y) ↔ y ≤ -5 ∨ y ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1658_165838


namespace NUMINAMATH_CALUDE_yellow_apples_probability_l1658_165855

/-- The probability of choosing 2 yellow apples out of 10 apples, where 4 are yellow -/
theorem yellow_apples_probability (total_apples : ℕ) (yellow_apples : ℕ) (chosen_apples : ℕ)
  (h1 : total_apples = 10)
  (h2 : yellow_apples = 4)
  (h3 : chosen_apples = 2) :
  (yellow_apples.choose chosen_apples : ℚ) / (total_apples.choose chosen_apples) = 2 / 15 :=
by sorry

end NUMINAMATH_CALUDE_yellow_apples_probability_l1658_165855


namespace NUMINAMATH_CALUDE_percentage_increase_l1658_165841

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 100 → final = 150 → (final - initial) / initial * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1658_165841


namespace NUMINAMATH_CALUDE_friend_team_assignment_count_l1658_165825

-- Define the number of friends and teams
def num_friends : ℕ := 6
def num_teams : ℕ := 4

-- Theorem statement
theorem friend_team_assignment_count :
  (num_teams ^ num_friends : ℕ) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_friend_team_assignment_count_l1658_165825


namespace NUMINAMATH_CALUDE_passing_percentage_l1658_165808

def total_marks : ℕ := 400
def student_marks : ℕ := 150
def failed_by : ℕ := 30

theorem passing_percentage : 
  (((student_marks + failed_by : ℚ) / total_marks) * 100 = 45) := by sorry

end NUMINAMATH_CALUDE_passing_percentage_l1658_165808


namespace NUMINAMATH_CALUDE_correct_result_l1658_165830

theorem correct_result (x : ℝ) (h : x / 3 = 45) : 3 * x = 405 := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l1658_165830


namespace NUMINAMATH_CALUDE_root_value_theorem_l1658_165849

theorem root_value_theorem (m : ℝ) : m^2 - 6*m - 5 = 0 → 11 + 6*m - m^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l1658_165849


namespace NUMINAMATH_CALUDE_area_ratio_PQRV_ABCD_l1658_165809

-- Define the squares and points
variable (A B C D P Q R V : ℝ × ℝ)

-- Define the properties of the squares
def is_square (A B C D : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    ‖B - A‖ = s ∧ ‖C - B‖ = s ∧ ‖D - C‖ = s ∧ ‖A - D‖ = s ∧
    (B - A) • (C - B) = 0

-- Define that P is on side AB
def P_on_AB (A B P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = A + t • (B - A)

-- Define the condition AP = 3 * PB
def AP_eq_3PB (A B P : ℝ × ℝ) : Prop :=
  ‖P - A‖ = 3 * ‖B - P‖

-- Define the area of a square
def area (A B C D : ℝ × ℝ) : ℝ :=
  ‖B - A‖^2

-- Theorem statement
theorem area_ratio_PQRV_ABCD 
  (h1 : is_square A B C D)
  (h2 : is_square P Q R V)
  (h3 : P_on_AB A B P)
  (h4 : AP_eq_3PB A B P) :
  area P Q R V / area A B C D = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_PQRV_ABCD_l1658_165809
