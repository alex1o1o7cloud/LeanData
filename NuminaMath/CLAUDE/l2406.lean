import Mathlib

namespace NUMINAMATH_CALUDE_walking_speed_problem_l2406_240661

theorem walking_speed_problem (x : ℝ) (h1 : x > 0) : 
  (100 / x * 12 = 100 + 20) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l2406_240661


namespace NUMINAMATH_CALUDE_smallest_norm_given_condition_l2406_240612

/-- Given a vector v in ℝ², prove that the smallest possible value of its norm,
    given that the norm of v + (4, 2) is 10, is 10 - 2√5. -/
theorem smallest_norm_given_condition (v : ℝ × ℝ) 
  (h : ‖v + (4, 2)‖ = 10) : 
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ 
  ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ :=
sorry

end NUMINAMATH_CALUDE_smallest_norm_given_condition_l2406_240612


namespace NUMINAMATH_CALUDE_triangle_area_special_case_l2406_240607

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
prove that under the given conditions, the area of the triangle is √15.
-/
theorem triangle_area_special_case (A B C : ℝ) (a b c : ℝ) : 
  a = 2 →
  2 * Real.sin A = Real.sin C →
  π / 2 < B → B < π →
  Real.cos (2 * C) = -1/4 →
  (1/2) * a * c * Real.sin B = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_special_case_l2406_240607


namespace NUMINAMATH_CALUDE_f_of_4_equals_9_l2406_240638

/-- The function f is defined as f(x) = x^2 - 2x + 1 for all x. -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem: The value of f(4) is 9. -/
theorem f_of_4_equals_9 : f 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_4_equals_9_l2406_240638


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l2406_240686

/-- A rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ :=
  4 * 3

theorem rectangular_prism_parallel_edges :
  ∀ (prism : RectangularPrism),
    prism.length = 4 ∧ prism.width = 3 ∧ prism.height = 2 →
    parallel_edge_pairs prism = 12 := by
  sorry

#check rectangular_prism_parallel_edges

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l2406_240686


namespace NUMINAMATH_CALUDE_ludwig_earnings_l2406_240615

/-- Calculates the weekly earnings of a worker with given work schedule and daily salary. -/
def weeklyEarnings (totalDays : ℕ) (halfDays : ℕ) (dailySalary : ℚ) : ℚ :=
  let fullDays := totalDays - halfDays
  fullDays * dailySalary + halfDays * (dailySalary / 2)

/-- Theorem stating that under the given conditions, the weekly earnings are $55. -/
theorem ludwig_earnings :
  weeklyEarnings 7 3 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ludwig_earnings_l2406_240615


namespace NUMINAMATH_CALUDE_f_2007_equals_zero_l2406_240614

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function g : ℝ → ℝ is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem f_2007_equals_zero
  (f g : ℝ → ℝ)
  (h_even : IsEven f)
  (h_odd : IsOdd g)
  (h_fg : ∀ x, g x = f (x - 1)) :
  f 2007 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2007_equals_zero_l2406_240614


namespace NUMINAMATH_CALUDE_hostel_rate_is_15_l2406_240692

/-- Represents the lodging problem for Jimmy's vacation --/
def lodging_problem (hostel_rate : ℚ) : Prop :=
  let hostel_nights : ℕ := 3
  let cabin_nights : ℕ := 2
  let cabin_rate : ℚ := 45
  let cabin_people : ℕ := 3
  let total_cost : ℚ := 75
  (hostel_nights : ℚ) * hostel_rate + 
    (cabin_nights : ℚ) * (cabin_rate / cabin_people) = total_cost

/-- Theorem stating that the hostel rate is $15 per night --/
theorem hostel_rate_is_15 : 
  lodging_problem 15 := by sorry

end NUMINAMATH_CALUDE_hostel_rate_is_15_l2406_240692


namespace NUMINAMATH_CALUDE_triangle_formation_l2406_240606

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  triangle_inequality 2 3 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l2406_240606


namespace NUMINAMATH_CALUDE_cinema_entry_cost_l2406_240654

def totalEntryCost (totalStudents : ℕ) (regularPrice : ℕ) (discountInterval : ℕ) (freeInterval : ℕ) : ℕ :=
  let discountedStudents := totalStudents / discountInterval
  let freeStudents := totalStudents / freeInterval
  let fullPriceStudents := totalStudents - discountedStudents - freeStudents
  let fullPriceCost := fullPriceStudents * regularPrice
  let discountedCost := discountedStudents * (regularPrice / 2)
  fullPriceCost + discountedCost

theorem cinema_entry_cost :
  totalEntryCost 84 50 12 35 = 3925 := by
  sorry

end NUMINAMATH_CALUDE_cinema_entry_cost_l2406_240654


namespace NUMINAMATH_CALUDE_gcd_of_45139_34481_4003_l2406_240693

theorem gcd_of_45139_34481_4003 : Nat.gcd 45139 (Nat.gcd 34481 4003) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45139_34481_4003_l2406_240693


namespace NUMINAMATH_CALUDE_point_position_l2406_240636

def line (x y : ℝ) := x + 2 * y = 2

def point_below_left (P : ℝ × ℝ) : Prop :=
  P.1 + 2 * P.2 < 2

theorem point_position :
  let P : ℝ × ℝ := (1/12, 33/36)
  point_below_left P := by sorry

end NUMINAMATH_CALUDE_point_position_l2406_240636


namespace NUMINAMATH_CALUDE_seedling_purchase_solution_l2406_240633

/-- Represents the unit prices and maximum purchase of seedlings --/
structure SeedlingPurchase where
  price_a : ℝ  -- Unit price of type A seedlings
  price_b : ℝ  -- Unit price of type B seedlings
  max_a : ℕ    -- Maximum number of type A seedlings that can be purchased

/-- Theorem statement for the seedling purchase problem --/
theorem seedling_purchase_solution :
  ∃ (sp : SeedlingPurchase),
    -- Condition 1: 30 bundles of A and 10 bundles of B cost 380 yuan
    30 * sp.price_a + 10 * sp.price_b = 380 ∧
    -- Condition 2: 50 bundles of A and 30 bundles of B cost 740 yuan
    50 * sp.price_a + 30 * sp.price_b = 740 ∧
    -- Condition 3: Budget constraint with discount
    sp.price_a * 0.9 * sp.max_a + sp.price_b * 0.9 * (100 - sp.max_a) ≤ 828 ∧
    -- Solution 1: Unit prices
    sp.price_a = 10 ∧ sp.price_b = 8 ∧
    -- Solution 2: Maximum number of type A seedlings
    sp.max_a = 60 := by
  sorry

end NUMINAMATH_CALUDE_seedling_purchase_solution_l2406_240633


namespace NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l2406_240698

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_line_segments := Q.vertices.choose 2
  let face_diagonals := Q.quadrilateral_faces * 2
  total_line_segments - Q.edges - face_diagonals

/-- The main theorem stating the number of space diagonals in the given polyhedron -/
theorem space_diagonals_of_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 42,
    triangular_faces := 30,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 341 := by sorry

end NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l2406_240698


namespace NUMINAMATH_CALUDE_max_a_value_l2406_240667

theorem max_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = -30) → 
  (a > 0) → 
  a ≤ 31 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l2406_240667


namespace NUMINAMATH_CALUDE_polyhedron_surface_area_l2406_240685

/-- Represents a polyhedron with three orthographic views --/
structure Polyhedron where
  front_view : Set (ℝ × ℝ)
  side_view : Set (ℝ × ℝ)
  top_view : Set (ℝ × ℝ)

/-- Calculates the surface area of a polyhedron --/
noncomputable def surface_area (p : Polyhedron) : ℝ := sorry

/-- Theorem stating that the surface area of the given polyhedron is 8 --/
theorem polyhedron_surface_area (p : Polyhedron) : surface_area p = 8 := by sorry

end NUMINAMATH_CALUDE_polyhedron_surface_area_l2406_240685


namespace NUMINAMATH_CALUDE_wolf_winning_strategy_wolf_wins_l2406_240688

/-- Represents a player in the game -/
inductive Player
| Wolf
| Hare

/-- Represents the state of the game board -/
structure GameState where
  number : Nat
  currentPlayer : Player

/-- Defines a valid move in the game -/
def isValidMove (n : Nat) (digit : Nat) : Prop :=
  digit > 0 ∧ digit ≤ 9 ∧ digit ≤ n

/-- Applies a move to the game state -/
def applyMove (state : GameState) (digit : Nat) : GameState :=
  { number := state.number - digit,
    currentPlayer := match state.currentPlayer with
      | Player.Wolf => Player.Hare
      | Player.Hare => Player.Wolf }

/-- Defines the winning condition -/
def isWinningState (state : GameState) : Prop :=
  state.number = 0

/-- Theorem: There exists a winning strategy for Wolf starting with 1234 -/
theorem wolf_winning_strategy :
  ∃ (strategy : GameState → Nat),
    (∀ (state : GameState), isValidMove state.number (strategy state)) →
    (∀ (state : GameState),
      state.currentPlayer = Player.Wolf →
      isWinningState (applyMove state (strategy state)) ∨
      ∃ (hareMove : Nat),
        isValidMove (applyMove state (strategy state)).number hareMove →
        isWinningState (applyMove (applyMove state (strategy state)) hareMove)) :=
sorry

/-- The initial game state -/
def initialState : GameState :=
  { number := 1234, currentPlayer := Player.Wolf }

/-- Corollary: Wolf wins the game starting from 1234 -/
theorem wolf_wins : ∃ (moves : List Nat), 
  isWinningState (moves.foldl applyMove initialState) ∧
  moves.length % 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_wolf_winning_strategy_wolf_wins_l2406_240688


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_properties_l2406_240653

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

def arithmetic_sequence (b₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := b₁ + (n - 1) * d

theorem geometric_arithmetic_sequence_properties
  (a₁ : ℝ) (b₁ : ℝ) (q : ℝ) (d : ℝ) 
  (h1 : q = -2/3)
  (h2 : b₁ = 12)
  (h3 : geometric_sequence a₁ q 9 > arithmetic_sequence b₁ d 9)
  (h4 : geometric_sequence a₁ q 10 > arithmetic_sequence b₁ d 10) :
  (geometric_sequence a₁ q 9 * geometric_sequence a₁ q 10 < 0) ∧
  (arithmetic_sequence b₁ d 9 > arithmetic_sequence b₁ d 10) :=
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_properties_l2406_240653


namespace NUMINAMATH_CALUDE_pencil_cost_l2406_240605

-- Define the cost of a pen and a pencil in cents
variable (p q : ℚ)

-- Define the conditions from the problem
def condition1 : Prop := 3 * p + 4 * q = 287
def condition2 : Prop := 5 * p + 2 * q = 236

-- Theorem to prove
theorem pencil_cost (h1 : condition1 p q) (h2 : condition2 p q) : q = 52 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l2406_240605


namespace NUMINAMATH_CALUDE_outfit_choices_l2406_240644

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 8

/-- The number of shirts available -/
def num_shirts : ℕ := num_colors

/-- The number of pants available -/
def num_pants : ℕ := num_colors

/-- The number of hats available -/
def num_hats : ℕ := num_colors

/-- Calculate the total number of outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- Calculate the number of outfits where shirt and pants are the same color -/
def same_color_combinations : ℕ := num_colors * num_hats

/-- Calculate the number of valid outfit choices -/
def valid_outfits : ℕ := total_combinations - same_color_combinations

theorem outfit_choices : valid_outfits = 448 := by
  sorry

end NUMINAMATH_CALUDE_outfit_choices_l2406_240644


namespace NUMINAMATH_CALUDE_share_of_A_l2406_240603

theorem share_of_A (total : ℝ) (a b c : ℝ) : 
  total = 116000 →
  a + b + c = total →
  a / b = 3 / 4 →
  b / c = 5 / 6 →
  a = 116000 * 15 / 59 :=
by sorry

end NUMINAMATH_CALUDE_share_of_A_l2406_240603


namespace NUMINAMATH_CALUDE_value_standard_deviations_below_mean_l2406_240674

/-- For a normal distribution with mean 14.5 and standard deviation 1.5,
    the value 11.5 is 2 standard deviations less than the mean. -/
theorem value_standard_deviations_below_mean
  (μ : ℝ) (σ : ℝ) (x : ℝ)
  (h_mean : μ = 14.5)
  (h_std_dev : σ = 1.5)
  (h_value : x = 11.5) :
  (μ - x) / σ = 2 := by
sorry

end NUMINAMATH_CALUDE_value_standard_deviations_below_mean_l2406_240674


namespace NUMINAMATH_CALUDE_sine_inequality_unique_solution_l2406_240689

theorem sine_inequality_unique_solution :
  ∀ y ∈ Set.Icc 0 (Real.pi / 2),
    (∀ x ∈ Set.Icc 0 Real.pi, Real.sin (x + y) < Real.sin x + Real.sin y) ↔
    y = 0 :=
by sorry

end NUMINAMATH_CALUDE_sine_inequality_unique_solution_l2406_240689


namespace NUMINAMATH_CALUDE_power_division_sum_difference_equals_sixteen_l2406_240616

theorem power_division_sum_difference_equals_sixteen :
  (5 ^ 6 / 5 ^ 4) + 3 ^ 3 - 6 ^ 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_division_sum_difference_equals_sixteen_l2406_240616


namespace NUMINAMATH_CALUDE_august_has_five_tuesdays_l2406_240623

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  first_day : DayOfWeek

/-- Given a month, returns the number of occurrences of each day of the week -/
def count_days (m : Month) : DayOfWeek → Nat :=
  sorry

/-- Returns true if the given day occurs exactly five times in the month -/
def occurs_five_times (d : DayOfWeek) (m : Month) : Prop :=
  count_days m d = 5

/-- Theorem: If July has five Fridays, then August must have five Tuesdays -/
theorem august_has_five_tuesdays
  (july : Month)
  (august : Month)
  (h1 : july.days = 31)
  (h2 : august.days = 31)
  (h3 : occurs_five_times DayOfWeek.Friday july) :
  occurs_five_times DayOfWeek.Tuesday august :=
sorry

end NUMINAMATH_CALUDE_august_has_five_tuesdays_l2406_240623


namespace NUMINAMATH_CALUDE_cylinder_height_comparison_l2406_240628

theorem cylinder_height_comparison (r₁ r₂ h₁ h₂ : ℝ) :
  r₁ > 0 ∧ r₂ > 0 ∧ h₁ > 0 ∧ h₂ > 0 →
  r₂ = 1.1 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.21 * h₂ :=
by sorry

#check cylinder_height_comparison

end NUMINAMATH_CALUDE_cylinder_height_comparison_l2406_240628


namespace NUMINAMATH_CALUDE_money_division_l2406_240666

theorem money_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 406)
  (h_a : a = b / 2)
  (h_b : b = c / 2)
  (h_sum : a + b + c = total) : c = 232 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l2406_240666


namespace NUMINAMATH_CALUDE_product_xyz_is_one_ninth_l2406_240699

theorem product_xyz_is_one_ninth 
  (x y z : ℝ) 
  (h1 : x + 1/y = 3) 
  (h2 : y + 1/z = 5) : 
  x * y * z = 1/9 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_is_one_ninth_l2406_240699


namespace NUMINAMATH_CALUDE_cos_shift_equivalence_l2406_240659

open Real

theorem cos_shift_equivalence (x : ℝ) :
  cos (2 * (x + π / 6) - π / 3) = cos (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_cos_shift_equivalence_l2406_240659


namespace NUMINAMATH_CALUDE_notepad_lasts_four_days_l2406_240642

/-- Represents the number of pieces of letter-size paper used --/
def letter_size_papers : ℕ := 5

/-- Represents the number of times each paper is folded --/
def folds : ℕ := 3

/-- Represents the number of notes written per day --/
def notes_per_day : ℕ := 10

/-- Calculates the number of note-size papers produced from one letter-size paper --/
def note_papers_per_letter_paper : ℕ := 2^folds

/-- Calculates the total number of note-size papers in a notepad --/
def total_note_papers : ℕ := letter_size_papers * note_papers_per_letter_paper

/-- Represents how long a notepad lasts in days --/
def notepad_duration : ℕ := total_note_papers / notes_per_day

theorem notepad_lasts_four_days : notepad_duration = 4 := by
  sorry

end NUMINAMATH_CALUDE_notepad_lasts_four_days_l2406_240642


namespace NUMINAMATH_CALUDE_linear_function_proof_l2406_240643

/-- A linear function passing through points (1, 3) and (-2, 12) -/
def f (x : ℝ) : ℝ := -3 * x + 6

theorem linear_function_proof :
  (f 1 = 3 ∧ f (-2) = 12) ∧
  (∀ a : ℝ, f (2 * a) ≠ -6 * a + 8) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_proof_l2406_240643


namespace NUMINAMATH_CALUDE_soccer_ball_inflation_time_l2406_240678

/-- The time in minutes it takes to inflate one soccer ball -/
def inflationTime : ℕ := 20

/-- The number of balls Alexia inflates -/
def alexiaBalls : ℕ := 20

/-- The number of additional balls Ermias inflates compared to Alexia -/
def ermiasDifference : ℕ := 5

/-- The total number of balls Ermias inflates -/
def ermiasBalls : ℕ := alexiaBalls + ermiasDifference

/-- The total number of balls inflated by both Alexia and Ermias -/
def totalBalls : ℕ := alexiaBalls + ermiasBalls

/-- The total time in minutes taken to inflate all soccer balls -/
def totalTime : ℕ := totalBalls * inflationTime

theorem soccer_ball_inflation_time : totalTime = 900 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_inflation_time_l2406_240678


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2406_240694

theorem imaginary_part_of_complex_fraction : Complex.im (2 * Complex.I / (1 - Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2406_240694


namespace NUMINAMATH_CALUDE_grapes_and_orange_cost_l2406_240675

/-- Represents the prices of items in John's purchase -/
structure Prices where
  peanuts : ℝ
  grapes : ℝ
  orange : ℝ
  chocolates : ℝ

/-- The conditions of John's purchase -/
def purchase_conditions (p : Prices) : Prop :=
  p.peanuts + p.grapes + p.orange + p.chocolates = 25 ∧
  p.chocolates = 2 * p.peanuts ∧
  p.orange = p.peanuts - p.grapes

/-- The theorem stating the cost of grapes and orange -/
theorem grapes_and_orange_cost (p : Prices) 
  (h : purchase_conditions p) : p.grapes + p.orange = 6.25 := by
  sorry

#check grapes_and_orange_cost

end NUMINAMATH_CALUDE_grapes_and_orange_cost_l2406_240675


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficient_l2406_240683

theorem quadratic_equation_coefficient : ∀ a b c : ℝ,
  (∀ x, 3 * x^2 + 1 = 6 * x ↔ a * x^2 + b * x + c = 0) →
  a = 3 →
  b = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficient_l2406_240683


namespace NUMINAMATH_CALUDE_no_valid_mapping_divisible_by_1010_l2406_240671

/-- Represents a mapping from letters to digits -/
def LetterToDigitMap := Char → Fin 10

/-- Checks if a mapping is valid for the word INNOPOLIS -/
def is_valid_mapping (m : LetterToDigitMap) : Prop :=
  m 'I' ≠ m 'N' ∧ m 'I' ≠ m 'O' ∧ m 'I' ≠ m 'P' ∧ m 'I' ≠ m 'L' ∧ m 'I' ≠ m 'S' ∧
  m 'N' ≠ m 'O' ∧ m 'N' ≠ m 'P' ∧ m 'N' ≠ m 'L' ∧ m 'N' ≠ m 'S' ∧
  m 'O' ≠ m 'P' ∧ m 'O' ≠ m 'L' ∧ m 'O' ≠ m 'S' ∧
  m 'P' ≠ m 'L' ∧ m 'P' ≠ m 'S' ∧
  m 'L' ≠ m 'S'

/-- Converts the word INNOPOLIS to a number using the given mapping -/
def word_to_number (m : LetterToDigitMap) : ℕ :=
  m 'I' * 100000000 + m 'N' * 10000000 + m 'N' * 1000000 + 
  m 'O' * 100000 + m 'P' * 10000 + m 'O' * 1000 + 
  m 'L' * 100 + m 'I' * 10 + m 'S'

/-- The main theorem stating that no valid mapping exists that makes the number divisible by 1010 -/
theorem no_valid_mapping_divisible_by_1010 :
  ¬ ∃ (m : LetterToDigitMap), is_valid_mapping m ∧ (word_to_number m % 1010 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_mapping_divisible_by_1010_l2406_240671


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2406_240662

theorem quadratic_discriminant : 
  let a : ℚ := 5
  let b : ℚ := 5 + 1/5
  let c : ℚ := 1/5
  let discriminant := b^2 - 4*a*c
  discriminant = 576/25 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2406_240662


namespace NUMINAMATH_CALUDE_g_of_6_l2406_240664

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 37*x^2 - 18*x - 80

theorem g_of_6 : g 6 = 712 := by
  sorry

end NUMINAMATH_CALUDE_g_of_6_l2406_240664


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2406_240640

theorem quadratic_inequality_equivalence (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4*a*c ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2406_240640


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l2406_240690

theorem common_internal_tangent_length
  (center_distance : ℝ)
  (small_radius : ℝ)
  (large_radius : ℝ)
  (h1 : center_distance = 41)
  (h2 : small_radius = 4)
  (h3 : large_radius = 5) :
  Real.sqrt (center_distance^2 - (small_radius + large_radius)^2) = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l2406_240690


namespace NUMINAMATH_CALUDE_olympiad_problem_selection_l2406_240604

theorem olympiad_problem_selection (total_initial : ℕ) (final_count : ℕ) :
  total_initial = 27 →
  final_count = 10 →
  ∃ (alina_problems masha_problems : ℕ),
    alina_problems + masha_problems = total_initial ∧
    alina_problems / 2 + 2 * masha_problems / 3 = total_initial - final_count ∧
    masha_problems - alina_problems = 15 :=
by sorry

end NUMINAMATH_CALUDE_olympiad_problem_selection_l2406_240604


namespace NUMINAMATH_CALUDE_x₁_plus_x₂_pos_l2406_240634

noncomputable section

variables (a : ℝ) (x x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.log (a * x + 1) - a * x - Real.log a

axiom a_pos : a > 0

axiom x_domain : x > -1/a

axiom x₁_domain : -1/a < x₁ ∧ x₁ < 0

axiom x₂_domain : x₂ > 0

axiom f_roots : f a x₁ = 0 ∧ f a x₂ = 0

theorem x₁_plus_x₂_pos : x₁ + x₂ > 0 := by sorry

end NUMINAMATH_CALUDE_x₁_plus_x₂_pos_l2406_240634


namespace NUMINAMATH_CALUDE_election_vote_difference_l2406_240621

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 6450 →
  candidate_percentage = 31 / 100 →
  ⌊(1 - candidate_percentage) * total_votes⌋ - ⌊candidate_percentage * total_votes⌋ = 2451 :=
by sorry

end NUMINAMATH_CALUDE_election_vote_difference_l2406_240621


namespace NUMINAMATH_CALUDE_merchant_problem_l2406_240622

theorem merchant_problem (n : ℕ) : 
  (100 * n^2 : ℕ) / 100 * (2 * n) = 2662 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_merchant_problem_l2406_240622


namespace NUMINAMATH_CALUDE_fruit_basket_count_l2406_240635

def total_fruits (mangoes pears pawpaws kiwis lemons : ℕ) : ℕ :=
  mangoes + pears + pawpaws + kiwis + lemons

theorem fruit_basket_count : 
  ∀ (kiwis : ℕ),
  kiwis = 9 →
  total_fruits 18 10 12 kiwis 9 = 58 := by
sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l2406_240635


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l2406_240620

theorem point_on_terminal_side (y : ℝ) (β : ℝ) : 
  (- Real.sqrt 3 : ℝ) ^ 2 + y ^ 2 > 0 →  -- Point P is not at the origin
  Real.sin β = Real.sqrt 13 / 13 →      -- Given condition for sin β
  y > 0 →                               -- y is positive (terminal side in first quadrant)
  y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l2406_240620


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l2406_240602

theorem subtraction_of_fractions : 1 / 210 - 17 / 35 = -101 / 210 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l2406_240602


namespace NUMINAMATH_CALUDE_charity_event_result_l2406_240610

/-- Represents the number of books of each type brought by a participant -/
structure BookContribution where
  encyclopedias : ℕ
  fiction : ℕ
  reference : ℕ

/-- Represents the total number of books collected -/
structure TotalBooks where
  encyclopedias : ℕ
  fiction : ℕ
  reference : ℕ

/-- Represents how books are distributed on shelves -/
structure ShelfDistribution where
  first_shelf : ℕ
  second_shelf : ℕ

def charity_event_books (total : TotalBooks) (shelf : ShelfDistribution) : Prop :=
  -- Each participant brings either 1 encyclopedia, 3 fiction books, or 2 reference books
  ∃ (participants : ℕ) (encyc_part fict_part ref_part : ℕ),
    participants = encyc_part + fict_part + ref_part ∧
    total.encyclopedias = encyc_part * 1 ∧
    total.fiction = fict_part * 3 ∧
    total.reference = ref_part * 2 ∧
    -- 150 encyclopedias were collected
    total.encyclopedias = 150 ∧
    -- Two bookshelves were filled with an equal number of books
    shelf.first_shelf = shelf.second_shelf ∧
    -- The first shelf contained 1/5 of all reference books, 1/7 of all fiction books, and all encyclopedias
    shelf.first_shelf = total.encyclopedias + total.reference / 5 + total.fiction / 7 ∧
    -- Total books on both shelves
    shelf.first_shelf + shelf.second_shelf = total.encyclopedias + total.fiction + total.reference

theorem charity_event_result :
  ∀ (total : TotalBooks) (shelf : ShelfDistribution),
    charity_event_books total shelf →
    ∃ (participants : ℕ),
      participants = 416 ∧
      total.encyclopedias + total.fiction + total.reference = 738 :=
sorry

end NUMINAMATH_CALUDE_charity_event_result_l2406_240610


namespace NUMINAMATH_CALUDE_defective_items_count_l2406_240647

def total_products : ℕ := 100
def defective_items : ℕ := 2
def items_to_draw : ℕ := 3

def ways_with_defective : ℕ := Nat.choose total_products items_to_draw - Nat.choose (total_products - defective_items) items_to_draw

theorem defective_items_count : ways_with_defective = 9472 := by
  sorry

end NUMINAMATH_CALUDE_defective_items_count_l2406_240647


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2406_240663

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (aₙ : ℚ) (n : ℕ) : ℚ := n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_problem (a₁ a₂ a₅ aₙ : ℚ) (n : ℕ) :
  a₁ = 1/3 →
  a₂ + a₅ = 4 →
  aₙ = 33 →
  (∃ d : ℚ, ∀ k : ℕ, arithmetic_sequence a₁ d k = a₁ + (k - 1 : ℚ) * d) →
  n = 50 ∧ sum_arithmetic_sequence a₁ aₙ n = 850 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2406_240663


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2406_240650

theorem simplify_square_roots : 
  (Real.sqrt 392 / Real.sqrt 352) + (Real.sqrt 180 / Real.sqrt 120) = 
  (7 * Real.sqrt 6 + 6 * Real.sqrt 11) / (2 * Real.sqrt 66) := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2406_240650


namespace NUMINAMATH_CALUDE_original_number_is_nine_l2406_240658

theorem original_number_is_nine (x : ℝ) : (x - 5) / 4 = (x - 4) / 5 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_nine_l2406_240658


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2406_240655

/-- If x^2 + mx + n is a perfect square, then n = (|m| / 2)^2 -/
theorem perfect_square_condition (m n : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + m*x + n = k^2) → n = (|m| / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2406_240655


namespace NUMINAMATH_CALUDE_sufficient_implies_necessary_l2406_240672

theorem sufficient_implies_necessary (A B : Prop) :
  (A → B) → (¬B → ¬A) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_implies_necessary_l2406_240672


namespace NUMINAMATH_CALUDE_mario_orange_consumption_l2406_240657

/-- Represents the amount of fruit eaten by each person in ounces -/
structure FruitConsumption where
  mario : ℕ
  lydia : ℕ
  nicolai : ℕ

/-- Converts pounds to ounces -/
def poundsToOunces (pounds : ℕ) : ℕ := pounds * 16

/-- Theorem: Given the conditions, Mario ate 8 ounces of oranges -/
theorem mario_orange_consumption (total : ℕ) (fc : FruitConsumption) 
  (h1 : poundsToOunces total = fc.mario + fc.lydia + fc.nicolai)
  (h2 : total = 8)
  (h3 : fc.lydia = 24)
  (h4 : fc.nicolai = poundsToOunces 6) :
  fc.mario = 8 := by
  sorry

#check mario_orange_consumption

end NUMINAMATH_CALUDE_mario_orange_consumption_l2406_240657


namespace NUMINAMATH_CALUDE_expand_product_l2406_240600

theorem expand_product (x : ℝ) : 3 * (x + 4) * (x + 5) = 3 * x^2 + 27 * x + 60 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2406_240600


namespace NUMINAMATH_CALUDE_willow_football_time_l2406_240691

/-- Proves that Willow played football for 60 minutes given the conditions -/
theorem willow_football_time :
  ∀ (total_time basketball_time football_time : ℕ),
  total_time = 120 →
  basketball_time = 60 →
  total_time = basketball_time + football_time →
  football_time = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_willow_football_time_l2406_240691


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_slope_l2406_240687

/-- Given a quadrilateral ABCD inscribed in an ellipse, with three sides AB, BC, CD parallel to fixed directions,
    the slope of the fourth side DA is determined by the slopes of the other three sides and the ellipse parameters. -/
theorem inscribed_quadrilateral_slope (a b : ℝ) (m₁ m₂ m₃ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ m : ℝ, m = (b^2 * (m₁ + m₃ - m₂) + a^2 * m₁ * m₂ * m₃) / (b^2 + a^2 * (m₁ * m₂ + m₂ * m₃ - m₁ * m₃)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_slope_l2406_240687


namespace NUMINAMATH_CALUDE_count_integers_with_factors_l2406_240648

theorem count_integers_with_factors : 
  ∃! n : ℕ, 200 ≤ n ∧ n ≤ 500 ∧ 22 ∣ n ∧ 16 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_factors_l2406_240648


namespace NUMINAMATH_CALUDE_initial_savings_theorem_l2406_240625

def calculate_initial_savings (repair_fee : ℕ) (remaining_savings : ℕ) : ℕ :=
  let corner_light := 2 * repair_fee
  let brake_disk := 3 * corner_light
  let floor_mats := brake_disk
  let steering_wheel_cover := corner_light / 2
  let seat_covers := 2 * floor_mats
  let total_expenses := repair_fee + corner_light + 2 * brake_disk + floor_mats + steering_wheel_cover + seat_covers
  remaining_savings + total_expenses

theorem initial_savings_theorem (repair_fee : ℕ) (remaining_savings : ℕ) :
  repair_fee = 10 ∧ remaining_savings = 480 →
  calculate_initial_savings repair_fee remaining_savings = 820 :=
by sorry

end NUMINAMATH_CALUDE_initial_savings_theorem_l2406_240625


namespace NUMINAMATH_CALUDE_f_bounds_and_solution_set_l2406_240601

noncomputable def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem f_bounds_and_solution_set :
  (∀ x : ℝ, -3 ≤ f x ∧ f x ≤ 3) ∧
  {x : ℝ | f x ≥ x^2 - 8*x + 14} = {x : ℝ | 3 ≤ x ∧ x ≤ 4 + Real.sqrt 5} :=
by sorry

end NUMINAMATH_CALUDE_f_bounds_and_solution_set_l2406_240601


namespace NUMINAMATH_CALUDE_total_faces_is_198_l2406_240631

/-- The total number of faces on all dice and geometrical shapes -/
def total_faces : ℕ := sorry

/-- Number of six-sided dice -/
def six_sided_dice : ℕ := 4

/-- Number of eight-sided dice -/
def eight_sided_dice : ℕ := 5

/-- Number of twelve-sided dice -/
def twelve_sided_dice : ℕ := 3

/-- Number of twenty-sided dice -/
def twenty_sided_dice : ℕ := 2

/-- Number of cubes -/
def cubes : ℕ := 1

/-- Number of tetrahedrons -/
def tetrahedrons : ℕ := 3

/-- Number of icosahedrons -/
def icosahedrons : ℕ := 2

/-- Theorem stating that the total number of faces is 198 -/
theorem total_faces_is_198 : total_faces = 198 := by sorry

end NUMINAMATH_CALUDE_total_faces_is_198_l2406_240631


namespace NUMINAMATH_CALUDE_triangle_property_l2406_240608

/-- Given a triangle ABC with angles A, B, C satisfying the given condition,
    prove that A = π/3 and the maximum area is 3√3/4 when the circumradius is 1 -/
theorem triangle_property (A B C : Real) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (eq : (Real.sin A - Real.sin B + Real.sin C) / Real.sin C = 
        Real.sin B / (Real.sin A + Real.sin B - Real.sin C)) :
  A = π/3 ∧ 
  (∀ S : Real, S ≤ 3 * Real.sqrt 3 / 4 ∧ 
    ∃ a b c : Real, 0 < a ∧ 0 < b ∧ 0 < c ∧
      a^2 + b^2 + c^2 = 2 * (a*b + b*c + c*a) ∧
      S = (Real.sin A * b * c) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l2406_240608


namespace NUMINAMATH_CALUDE_flight_cost_a_to_b_l2406_240668

/-- Represents the cost of a flight between two cities -/
structure FlightCost where
  bookingFee : ℝ
  ratePerKm : ℝ

/-- Calculates the total cost of a flight -/
def calculateFlightCost (distance : ℝ) (cost : FlightCost) : ℝ :=
  cost.bookingFee + cost.ratePerKm * distance

/-- The problem statement -/
theorem flight_cost_a_to_b :
  let distanceAB : ℝ := 3500
  let flightCost : FlightCost := { bookingFee := 120, ratePerKm := 0.12 }
  calculateFlightCost distanceAB flightCost = 540 := by
  sorry


end NUMINAMATH_CALUDE_flight_cost_a_to_b_l2406_240668


namespace NUMINAMATH_CALUDE_algebraic_simplification_l2406_240637

theorem algebraic_simplification (m n : ℝ) :
  (3 * m^2 - m * n + 5) - 2 * (5 * m * n - 4 * m^2 + 2) = 11 * m^2 - 11 * m * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l2406_240637


namespace NUMINAMATH_CALUDE_inverse_proportional_solution_l2406_240682

def inverse_proportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportional_solution (x y : ℝ) 
  (h1 : inverse_proportional x y) 
  (h2 : x + y = 30) 
  (h3 : x - y = 10) : 
  (∃ y' : ℝ, inverse_proportional 8 y' ∧ y' = 25) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportional_solution_l2406_240682


namespace NUMINAMATH_CALUDE_circle_center_sum_l2406_240649

/-- Given a circle with equation x^2 + y^2 = 6x + 4y + 4, prove that the sum of the coordinates of its center is 5. -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 4*y + 4 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 4)) → 
  h + k = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2406_240649


namespace NUMINAMATH_CALUDE_probability_one_heads_three_coins_l2406_240697

theorem probability_one_heads_three_coins :
  let n : ℕ := 3  -- number of coins
  let p : ℚ := 1/2  -- probability of heads for a fair coin
  let k : ℕ := 1  -- number of heads we want
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k) = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_heads_three_coins_l2406_240697


namespace NUMINAMATH_CALUDE_fish_difference_l2406_240646

theorem fish_difference (goldfish : ℕ) (angelfish : ℕ) (guppies : ℕ) : 
  goldfish = 8 →
  guppies = 2 * angelfish →
  goldfish + angelfish + guppies = 44 →
  angelfish - goldfish = 4 := by
sorry

end NUMINAMATH_CALUDE_fish_difference_l2406_240646


namespace NUMINAMATH_CALUDE_petya_wins_petya_wins_game_l2406_240696

/-- Represents the game between Petya and Vasya with two boxes of candies. -/
structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

/-- Defines the game setup with the given conditions. -/
def game : CandyGame :=
  { total_candies := 25
  , prob_two_caramels := 0.54 }

/-- Theorem stating that Petya has a higher chance of winning the game. -/
theorem petya_wins (g : CandyGame) : g.prob_two_caramels > 0.5 → 
  (1 - g.prob_two_caramels) < 0.5 := by
  sorry

/-- Corollary proving that Petya wins the specific game instance. -/
theorem petya_wins_game : (1 - game.prob_two_caramels) < 0.5 := by
  sorry

end NUMINAMATH_CALUDE_petya_wins_petya_wins_game_l2406_240696


namespace NUMINAMATH_CALUDE_union_A_B_when_m_1_intersection_A_B_empty_union_A_B_equals_A_l2406_240641

-- Define sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ 2 * m + 1}

-- Theorem 1: Union of A and B when m = 1
theorem union_A_B_when_m_1 :
  A ∪ B 1 = {x | -2 < x ∧ x ≤ 3} := by sorry

-- Theorem 2: Condition for A ∩ B = ∅
theorem intersection_A_B_empty (m : ℝ) :
  A ∩ B m = ∅ ↔ m ≤ -3/2 ∨ m ≥ 4 := by sorry

-- Theorem 3: Condition for A ∪ B = A
theorem union_A_B_equals_A (m : ℝ) :
  A ∪ B m = A ↔ m < -3 ∨ (0 < m ∧ m < 1/2) := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_1_intersection_A_B_empty_union_A_B_equals_A_l2406_240641


namespace NUMINAMATH_CALUDE_min_value_3a_plus_1_l2406_240609

theorem min_value_3a_plus_1 (a : ℝ) (h : 8 * a^2 + 9 * a + 6 = 2) :
  ∃ (x : ℝ), (3 * a + 1 ≥ x) ∧ (∀ y, 3 * a + 1 ≥ y → x ≥ y) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_3a_plus_1_l2406_240609


namespace NUMINAMATH_CALUDE_tangent_perpendicular_range_l2406_240695

/-- The range of a when the tangent lines of two specific curves are perpendicular -/
theorem tangent_perpendicular_range (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 (3/2) ∧ 
    ((a * x₀ + a - 1) * (x₀ - 2) = -1)) → 
  a ∈ Set.Icc 1 (3/2) := by
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_range_l2406_240695


namespace NUMINAMATH_CALUDE_ratio_equation_solution_product_l2406_240630

theorem ratio_equation_solution_product (x : ℝ) : 
  (((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → x = 0) ∧ 
  (∃ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_product_l2406_240630


namespace NUMINAMATH_CALUDE_oxford_high_school_principals_l2406_240676

/-- Oxford High School Problem -/
theorem oxford_high_school_principals 
  (total_people : ℕ) 
  (teachers : ℕ) 
  (classes : ℕ) 
  (students_per_class : ℕ) 
  (h1 : total_people = 349) 
  (h2 : teachers = 48) 
  (h3 : classes = 15) 
  (h4 : students_per_class = 20) :
  total_people - (teachers + classes * students_per_class) = 1 :=
by sorry

end NUMINAMATH_CALUDE_oxford_high_school_principals_l2406_240676


namespace NUMINAMATH_CALUDE_square_odd_digits_iff_one_or_three_l2406_240617

/-- A function that checks if a natural number consists of only odd digits -/
def hasOnlyOddDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d % 2 = 1

/-- Theorem stating that n^2 has only odd digits if and only if n is 1 or 3 -/
theorem square_odd_digits_iff_one_or_three (n : ℕ) :
  n > 0 → (hasOnlyOddDigits (n^2) ↔ n = 1 ∨ n = 3) :=
by sorry

end NUMINAMATH_CALUDE_square_odd_digits_iff_one_or_three_l2406_240617


namespace NUMINAMATH_CALUDE_courtyard_length_is_20_l2406_240681

/-- The width of the courtyard in meters -/
def courtyard_width : ℝ := 16.5

/-- The number of paving stones required to cover the courtyard -/
def num_paving_stones : ℕ := 66

/-- The length of a paving stone in meters -/
def paving_stone_length : ℝ := 2.5

/-- The width of a paving stone in meters -/
def paving_stone_width : ℝ := 2

/-- The theorem stating that the length of the courtyard is 20 meters -/
theorem courtyard_length_is_20 : 
  (courtyard_width * (num_paving_stones * paving_stone_length * paving_stone_width) / courtyard_width) = 20 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_is_20_l2406_240681


namespace NUMINAMATH_CALUDE_consecutive_sum_formula_l2406_240652

def consecutive_sum (n : ℤ) : ℤ := (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)

theorem consecutive_sum_formula (n : ℤ) : consecutive_sum n = 5 * n + 20 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_formula_l2406_240652


namespace NUMINAMATH_CALUDE_square_area_problem_l2406_240679

theorem square_area_problem (a b : ℝ) (h : a > b) :
  let diagonal_I := a - b
  let area_I := (diagonal_I^2) / 2
  let area_II := 3 * area_I
  area_II = (3 * (a - b)^2) / 2 := by
sorry

end NUMINAMATH_CALUDE_square_area_problem_l2406_240679


namespace NUMINAMATH_CALUDE_binary_sequence_eventually_periodic_l2406_240670

/-- A sequence of 0s and 1s -/
def BinarySequence := ℕ → Fin 2

/-- A block of n consecutive terms in a sequence -/
def Block (s : BinarySequence) (n : ℕ) (start : ℕ) : Fin n → Fin 2 :=
  fun i => s (start + i)

/-- A sequence is eventually periodic if there exist positive integers p and N such that
    for all k ≥ N, s(k + p) = s(k) -/
def EventuallyPeriodic (s : BinarySequence) : Prop :=
  ∃ (p N : ℕ), p > 0 ∧ ∀ k ≥ N, s (k + p) = s k

/-- The main theorem: if a binary sequence contains only n different blocks of
    n consecutive terms, where n is a positive integer, then it is eventually periodic -/
theorem binary_sequence_eventually_periodic
  (s : BinarySequence) (n : ℕ) (hn : n > 0)
  (h_blocks : ∃ (blocks : Finset (Fin n → Fin 2)),
    blocks.card = n ∧
    ∀ k, ∃ b ∈ blocks, Block s n k = b) :
  EventuallyPeriodic s :=
sorry

end NUMINAMATH_CALUDE_binary_sequence_eventually_periodic_l2406_240670


namespace NUMINAMATH_CALUDE_max_sum_ITEST_l2406_240651

theorem max_sum_ITEST (I T E S : ℕ+) : 
  I ≠ T ∧ I ≠ E ∧ I ≠ S ∧ T ≠ E ∧ T ≠ S ∧ E ≠ S →
  I * T * E * S * T = 2006 →
  (∀ (I' T' E' S' : ℕ+), 
    I' ≠ T' ∧ I' ≠ E' ∧ I' ≠ S' ∧ T' ≠ E' ∧ T' ≠ S' ∧ E' ≠ S' →
    I' * T' * E' * S' * T' = 2006 →
    I + T + E + S + T + 2006 ≥ I' + T' + E' + S' + T' + 2006) →
  I + T + E + S + T + 2006 = 2086 := by
sorry

end NUMINAMATH_CALUDE_max_sum_ITEST_l2406_240651


namespace NUMINAMATH_CALUDE_leo_laundry_problem_l2406_240626

theorem leo_laundry_problem (total_bill : ℕ) (shirt_cost : ℕ) (trouser_cost : ℕ) (total_shirts : ℕ) :
  total_bill = 140 →
  shirt_cost = 5 →
  trouser_cost = 9 →
  total_shirts = 10 →
  ∃ (num_trousers : ℕ), num_trousers = 10 ∧ total_bill = shirt_cost * total_shirts + trouser_cost * num_trousers :=
by
  sorry

end NUMINAMATH_CALUDE_leo_laundry_problem_l2406_240626


namespace NUMINAMATH_CALUDE_juice_bar_problem_l2406_240632

theorem juice_bar_problem (total_spent : ℕ) (mango_juice_cost : ℕ) (other_juice_total : ℕ) (total_people : ℕ) :
  total_spent = 94 →
  mango_juice_cost = 5 →
  other_juice_total = 54 →
  total_people = 17 →
  ∃ (other_juice_cost : ℕ),
    other_juice_cost = 6 ∧
    other_juice_cost * (total_people - (total_spent - other_juice_total) / mango_juice_cost) = other_juice_total :=
by sorry

end NUMINAMATH_CALUDE_juice_bar_problem_l2406_240632


namespace NUMINAMATH_CALUDE_sum_of_even_and_odd_is_odd_l2406_240656

def P : Set ℤ := {x | ∃ k, x = 2 * k}
def Q : Set ℤ := {x | ∃ k, x = 2 * k + 1}
def R : Set ℤ := {x | ∃ k, x = 4 * k + 1}

theorem sum_of_even_and_odd_is_odd (a b : ℤ) (ha : a ∈ P) (hb : b ∈ Q) : 
  a + b ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_and_odd_is_odd_l2406_240656


namespace NUMINAMATH_CALUDE_polynomial_form_l2406_240629

/-- A polynomial that satisfies the given condition -/
noncomputable def satisfying_polynomial (P : ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), P (a + b - 2*c) + P (b + c - 2*a) + P (c + a - 2*b) = 
                  3*P (a - b) + 3*P (b - c) + 3*P (c - a)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_form (P : ℝ → ℝ) (hP : satisfying_polynomial P) :
  ∃ (a b : ℝ), ∀ x, P x = a * x^2 + b * x :=
sorry

end NUMINAMATH_CALUDE_polynomial_form_l2406_240629


namespace NUMINAMATH_CALUDE_notebook_distribution_l2406_240639

theorem notebook_distribution (x : ℕ) : 
  (x > 0) → 
  ((x - 1) % 3 = 0) → 
  ((x + 2) % 4 = 0) → 
  ((x - 1) / 3 : ℚ) = ((x + 2) / 4 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_notebook_distribution_l2406_240639


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_l2406_240618

theorem sqrt_sum_squares : Real.sqrt (2^4 + 2^4 + 4^2) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_l2406_240618


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l2406_240611

theorem a_gt_one_sufficient_not_necessary_for_a_squared_gt_one :
  (∃ a : ℝ, a > 1 ∧ a^2 > 1) ∧ 
  (∃ a : ℝ, a^2 > 1 ∧ ¬(a > 1)) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l2406_240611


namespace NUMINAMATH_CALUDE_july_green_tea_price_l2406_240645

/-- Represents the price of tea and coffee in June and July -/
structure PriceData where
  june_price : ℝ
  july_coffee_price : ℝ
  july_tea_price : ℝ

/-- Represents the mixture of tea and coffee -/
structure Mixture where
  tea_quantity : ℝ
  coffee_quantity : ℝ
  total_weight : ℝ
  total_cost : ℝ

/-- Theorem stating the price of green tea in July -/
theorem july_green_tea_price (p : PriceData) (m : Mixture) : 
  p.june_price > 0 ∧ 
  p.july_coffee_price = 2 * p.june_price ∧ 
  p.july_tea_price = 0.1 * p.june_price ∧
  m.tea_quantity = m.coffee_quantity ∧
  m.total_weight = 3 ∧
  m.total_cost = 3.15 ∧
  m.total_cost = m.tea_quantity * p.july_tea_price + m.coffee_quantity * p.july_coffee_price →
  p.july_tea_price = 0.1 := by
sorry


end NUMINAMATH_CALUDE_july_green_tea_price_l2406_240645


namespace NUMINAMATH_CALUDE_monotonic_implies_not_even_but_not_conversely_l2406_240677

-- Define the properties of a function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsMonotonic (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f x ≤ f y

-- State the theorem
theorem monotonic_implies_not_even_but_not_conversely :
  (∃ f : ℝ → ℝ, IsMonotonic f → ¬IsEven f) ∧
  (∃ g : ℝ → ℝ, ¬IsEven g ∧ ¬IsMonotonic g) :=
sorry

end NUMINAMATH_CALUDE_monotonic_implies_not_even_but_not_conversely_l2406_240677


namespace NUMINAMATH_CALUDE_sqrt_450_simplified_l2406_240665

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplified_l2406_240665


namespace NUMINAMATH_CALUDE_line_equation_through_intersection_and_parallel_l2406_240669

/-- Given two lines in the plane and a third line parallel to one of them,
    this theorem proves the equation of the third line. -/
theorem line_equation_through_intersection_and_parallel
  (l₁ l₂ l₃ l : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 3 * x + 5 * y - 4 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 6 * x - y + 3 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ 2 * x + 3 * y + 5 = 0)
  (h_intersect : ∃ x y, l₁ x y ∧ l₂ x y ∧ l x y)
  (h_parallel : ∃ k ≠ 0, ∀ x y, l x y ↔ 2 * k * x + 3 * k * y + (k * 5 + c) = 0) :
  ∀ x y, l x y ↔ 6 * x + 9 * y - 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_equation_through_intersection_and_parallel_l2406_240669


namespace NUMINAMATH_CALUDE_sonya_fell_six_times_l2406_240627

/-- The number of times Steven fell while ice skating -/
def steven_falls : ℕ := 3

/-- The number of times Stephanie fell while ice skating -/
def stephanie_falls : ℕ := steven_falls + 13

/-- The number of times Sonya fell while ice skating -/
def sonya_falls : ℕ := stephanie_falls / 2 - 2

/-- Theorem stating that Sonya fell 6 times -/
theorem sonya_fell_six_times : sonya_falls = 6 := by sorry

end NUMINAMATH_CALUDE_sonya_fell_six_times_l2406_240627


namespace NUMINAMATH_CALUDE_inequality_of_distinct_positive_numbers_l2406_240673

theorem inequality_of_distinct_positive_numbers (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hcd : c ≠ d) (hda : d ≠ a) :
  a^2 / b + b^2 / c + c^2 / d + d^2 / a > a + b + c + d :=
sorry

end NUMINAMATH_CALUDE_inequality_of_distinct_positive_numbers_l2406_240673


namespace NUMINAMATH_CALUDE_two_fifths_300_minus_three_fifths_125_l2406_240680

theorem two_fifths_300_minus_three_fifths_125 : 
  (2 : ℚ) / 5 * 300 - (3 : ℚ) / 5 * 125 = 45 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_300_minus_three_fifths_125_l2406_240680


namespace NUMINAMATH_CALUDE_rational_solution_quadratic_l2406_240660

theorem rational_solution_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + 4 * k = 0) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_rational_solution_quadratic_l2406_240660


namespace NUMINAMATH_CALUDE_sum_rounded_to_hundredth_l2406_240624

-- Define the numbers
def a : Float := 45.378
def b : Float := 13.897
def c : Float := 29.4567

-- Define the sum
def sum : Float := a + b + c

-- Define a function to round to the nearest hundredth
def round_to_hundredth (x : Float) : Float :=
  (x * 100).round / 100

-- Theorem statement
theorem sum_rounded_to_hundredth :
  round_to_hundredth sum = 88.74 := by sorry

end NUMINAMATH_CALUDE_sum_rounded_to_hundredth_l2406_240624


namespace NUMINAMATH_CALUDE_calculate_savings_l2406_240619

/-- Given a person's income and expenditure ratio, and their income, calculate their savings -/
theorem calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) 
  (h1 : income_ratio = 7)
  (h2 : expenditure_ratio = 6)
  (h3 : income = 14000) :
  income - (expenditure_ratio * income / income_ratio) = 2000 := by
  sorry

#check calculate_savings

end NUMINAMATH_CALUDE_calculate_savings_l2406_240619


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2406_240613

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : ℚ
  common_diff : ℚ
  seq_def : ∀ n, a n = first_term + (n - 1) * common_diff

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.first_term + (n - 1) * seq.common_diff) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, n > 0 → sum_n_terms a n / sum_n_terms b n = (2 * n + 3 : ℚ) / (3 * n - 1)) →
  a.a 9 / b.a 9 = 37 / 50 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2406_240613


namespace NUMINAMATH_CALUDE_oreo_distribution_l2406_240684

/-- The number of Oreos Jordan has -/
def jordans_oreos : ℕ := 11

/-- The number of Oreos James has -/
def james_oreos (j : ℕ) : ℕ := 2 * j + 3

/-- The total number of Oreos -/
def total_oreos : ℕ := 36

theorem oreo_distribution : 
  james_oreos jordans_oreos + jordans_oreos = total_oreos :=
by sorry

end NUMINAMATH_CALUDE_oreo_distribution_l2406_240684
