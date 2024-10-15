import Mathlib

namespace NUMINAMATH_CALUDE_solve_pocket_money_problem_l2652_265203

def pocket_money_problem (P : ℝ) : Prop :=
  let tteokbokki_cost : ℝ := P / 2
  let remaining_after_tteokbokki : ℝ := P - tteokbokki_cost
  let pencil_cost : ℝ := (3 / 8) * remaining_after_tteokbokki
  let final_remaining : ℝ := remaining_after_tteokbokki - pencil_cost
  (final_remaining = 2500) → (tteokbokki_cost = 4000)

theorem solve_pocket_money_problem :
  ∃ P : ℝ, pocket_money_problem P :=
sorry

end NUMINAMATH_CALUDE_solve_pocket_money_problem_l2652_265203


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2652_265297

-- Problem 1
theorem problem_1 (a b : ℝ) (h1 : a > b) (h2 : a^2 + b^2 = 13) (h3 : a * b = 6) :
  a - b = 1 := by sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : a^2 + b^2 = c^2)  -- Pythagorean theorem for right triangle
  (h3 : a^2 + b^2 + 25 = 6*a + 8*b) :
  a + b + c = 12 ∨ a + b + c = 7 + Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2652_265297


namespace NUMINAMATH_CALUDE_sum_of_four_digit_numbers_eq_93324_l2652_265299

def digits : List Nat := [2, 4, 5, 3]

/-- The sum of all four-digit numbers formed by using the digits 2, 4, 5, and 3 once each -/
def sum_of_four_digit_numbers : Nat :=
  let sum_of_digits := digits.sum
  let count_per_place := Nat.factorial 4 / 4
  sum_of_digits * count_per_place * (1000 + 100 + 10 + 1)

theorem sum_of_four_digit_numbers_eq_93324 :
  sum_of_four_digit_numbers = 93324 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_numbers_eq_93324_l2652_265299


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_count_l2652_265258

-- Define the people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person

-- Define a seating arrangement as a function from position to person
def SeatingArrangement := Fin 5 → Person

-- Define the condition that two people cannot sit next to each other
def CannotSitNextTo (p1 p2 : Person) (arrangement : SeatingArrangement) : Prop :=
  ∀ i : Fin 4, arrangement i ≠ p1 ∨ arrangement (Fin.succ i) ≠ p2

-- Define a valid seating arrangement
def ValidArrangement (arrangement : SeatingArrangement) : Prop :=
  (CannotSitNextTo Person.Alice Person.Bob arrangement) ∧
  (CannotSitNextTo Person.Alice Person.Carla arrangement) ∧
  (CannotSitNextTo Person.Carla Person.Bob arrangement) ∧
  (CannotSitNextTo Person.Carla Person.Derek arrangement) ∧
  (CannotSitNextTo Person.Derek Person.Eric arrangement)

-- The main theorem
theorem valid_seating_arrangements_count :
  ∃ arrangements : Finset SeatingArrangement,
    (∀ arr ∈ arrangements, ValidArrangement arr) ∧
    (∀ arr, ValidArrangement arr → arr ∈ arrangements) ∧
    arrangements.card = 12 :=
sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_count_l2652_265258


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solution_l2652_265230

theorem cubic_equation_integer_solution (m : ℤ) :
  (∃ x : ℤ, x^3 - m*x^2 + m*x - (m^2 + 1) = 0) ↔ (m = -3 ∨ m = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solution_l2652_265230


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2652_265253

/-- Given a curve C in polar coordinates with equation ρ = 6 * cos(θ),
    prove that its equivalent Cartesian equation is x² + y² = 6x -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 6 * Real.cos θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  x^2 + y^2 = 6*x := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2652_265253


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2652_265286

theorem inequality_solution_set (x : ℝ) :
  (x^2 - 5 * abs x + 6 < 0) ↔ ((-3 < x ∧ x < -2) ∨ (2 < x ∧ x < 3)) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2652_265286


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2652_265270

theorem opposite_of_negative_2023 : -(-(2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2652_265270


namespace NUMINAMATH_CALUDE_square_land_area_l2652_265262

/-- A square land plot with side length 40 units has an area of 1600 square units. -/
theorem square_land_area : 
  ∀ (side_length area : ℝ), 
  side_length = 40 → 
  area = side_length ^ 2 → 
  area = 1600 :=
by sorry

end NUMINAMATH_CALUDE_square_land_area_l2652_265262


namespace NUMINAMATH_CALUDE_word_permutations_l2652_265205

-- Define the number of distinct letters in the word
def num_distinct_letters : ℕ := 6

-- Define the function to calculate factorial
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem word_permutations :
  factorial num_distinct_letters = 720 := by
  sorry

end NUMINAMATH_CALUDE_word_permutations_l2652_265205


namespace NUMINAMATH_CALUDE_first_divisor_problem_l2652_265292

theorem first_divisor_problem :
  ∃ (d : ℕ+) (x k m : ℤ),
    x = k * d.val + 11 ∧
    x = 9 * m + 2 ∧
    d.val < 11 ∧
    9 % d.val = 0 ∧
    d = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l2652_265292


namespace NUMINAMATH_CALUDE_redToGreenGrapeRatio_l2652_265255

/-- Represents the composition of a fruit salad --/
structure FruitSalad where
  raspberries : ℕ
  greenGrapes : ℕ
  redGrapes : ℕ

/-- The properties of our specific fruit salad --/
def fruitSaladProperties (fs : FruitSalad) : Prop :=
  fs.raspberries = fs.greenGrapes - 5 ∧
  fs.raspberries + fs.greenGrapes + fs.redGrapes = 102 ∧
  fs.redGrapes = 67

/-- The theorem stating the ratio of red grapes to green grapes --/
theorem redToGreenGrapeRatio (fs : FruitSalad) 
  (h : fruitSaladProperties fs) : 
  fs.redGrapes * 20 = fs.greenGrapes * 67 := by
  sorry

#check redToGreenGrapeRatio

end NUMINAMATH_CALUDE_redToGreenGrapeRatio_l2652_265255


namespace NUMINAMATH_CALUDE_unique_function_solution_l2652_265248

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (1 + x * y) - f (x + y) = f x * f y) ∧ 
  (f (-1) ≠ 0) → 
  (∀ x : ℝ, f x = x - 1) :=
sorry

end NUMINAMATH_CALUDE_unique_function_solution_l2652_265248


namespace NUMINAMATH_CALUDE_percentage_non_mutated_frogs_l2652_265257

def total_frogs : ℕ := 250
def extra_legs : ℕ := 32
def two_heads : ℕ := 21
def bright_red : ℕ := 16
def skin_abnormalities : ℕ := 12
def extra_eyes : ℕ := 7

theorem percentage_non_mutated_frogs :
  let mutated_frogs := extra_legs + two_heads + bright_red + skin_abnormalities + extra_eyes
  let non_mutated_frogs := total_frogs - mutated_frogs
  (non_mutated_frogs : ℚ) / total_frogs * 100 = 648 / 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_non_mutated_frogs_l2652_265257


namespace NUMINAMATH_CALUDE_intersection_point_l2652_265239

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 3 = 0

-- Define a point on the y-axis
def on_y_axis (x y : ℝ) : Prop := x = 0

-- Theorem statement
theorem intersection_point :
  ∃ (y : ℝ), line_equation 0 y ∧ on_y_axis 0 y ∧ y = 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l2652_265239


namespace NUMINAMATH_CALUDE_problem_solution_l2652_265226

-- Define the propositions
def P (x : ℝ) : Prop := x^2 - 3*x + 2 = 0
def Q (x : ℝ) : Prop := x = 1
def R (x : ℝ) : Prop := x^2 + x + 1 < 0
def S (x : ℝ) : Prop := x > 2
def T (x : ℝ) : Prop := x^2 - 3*x + 2 > 0

-- Theorem statement
theorem problem_solution :
  (∀ x, (¬Q x → ¬P x) ↔ (P x → Q x)) ∧
  (¬(∃ x, R x) ↔ (∀ x, x^2 + x + 1 ≥ 0)) ∧
  ((∀ x, S x → T x) ∧ ¬(∀ x, T x → S x)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2652_265226


namespace NUMINAMATH_CALUDE_geometric_sequence_partial_sums_zero_property_l2652_265220

/-- A geometric sequence of real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Partial sums of a sequence -/
def partial_sums (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => partial_sums a n + a (n + 1)

/-- The main theorem -/
theorem geometric_sequence_partial_sums_zero_property
  (a : ℕ → ℝ) (h : geometric_sequence a) :
  (∀ n : ℕ, partial_sums a n ≠ 0) ∨
  (∀ m : ℕ, ∃ n : ℕ, n ≥ m ∧ partial_sums a n = 0) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_partial_sums_zero_property_l2652_265220


namespace NUMINAMATH_CALUDE_investment_growth_l2652_265277

/-- Proves that an initial investment of $400, when compounded annually at 12% interest for 5 years, results in a final value of $704.98. -/
theorem investment_growth (initial_investment : ℝ) (interest_rate : ℝ) (years : ℕ) (final_value : ℝ) :
  initial_investment = 400 →
  interest_rate = 0.12 →
  years = 5 →
  final_value = 704.98 →
  final_value = initial_investment * (1 + interest_rate) ^ years := by
  sorry


end NUMINAMATH_CALUDE_investment_growth_l2652_265277


namespace NUMINAMATH_CALUDE_rachel_homework_difference_l2652_265211

/-- Rachel's homework problem -/
theorem rachel_homework_difference (math_pages reading_pages : ℕ) : 
  math_pages = 8 →
  reading_pages = 14 →
  reading_pages > math_pages →
  reading_pages - math_pages = 6 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_difference_l2652_265211


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_and_exclusion_l2652_265254

theorem systematic_sampling_interval_and_exclusion 
  (total_stores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_stores = 92) 
  (h2 : sample_size = 30) :
  ∃ (interval : ℕ) (excluded : ℕ),
    interval * sample_size + excluded = total_stores ∧ 
    interval = 3 ∧ 
    excluded = 2 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_and_exclusion_l2652_265254


namespace NUMINAMATH_CALUDE_runs_ratio_l2652_265282

/-- Represents the runs scored by each player -/
structure Runs where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The conditions of the cricket match -/
def cricket_match (r : Runs) : Prop :=
  r.a + r.b + r.c = 95 ∧
  r.c = 75 ∧
  r.a * 3 = r.b * 1

/-- The theorem to prove -/
theorem runs_ratio (r : Runs) (h : cricket_match r) : 
  r.b * 5 = r.c * 1 := by
sorry


end NUMINAMATH_CALUDE_runs_ratio_l2652_265282


namespace NUMINAMATH_CALUDE_max_scheduling_ways_after_15_games_l2652_265210

/-- Represents a chess tournament between schoolchildren and students. -/
structure ChessTournament where
  schoolchildren : Nat
  students : Nat
  total_games : Nat
  scheduled_games : Nat

/-- The maximum number of ways to schedule one game in the next round. -/
def max_scheduling_ways (tournament : ChessTournament) : Nat :=
  tournament.total_games - tournament.scheduled_games

/-- The theorem stating the maximum number of ways to schedule one game
    after uniquely scheduling 15 games in a tournament with 15 schoolchildren
    and 15 students. -/
theorem max_scheduling_ways_after_15_games
  (tournament : ChessTournament)
  (h1 : tournament.schoolchildren = 15)
  (h2 : tournament.students = 15)
  (h3 : tournament.total_games = tournament.schoolchildren * tournament.students)
  (h4 : tournament.scheduled_games = 15) :
  max_scheduling_ways tournament = 120 := by
  sorry


end NUMINAMATH_CALUDE_max_scheduling_ways_after_15_games_l2652_265210


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2652_265207

/-- Two 2D vectors are parallel if and only if their determinant is zero -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 - v 1 * w 0 = 0

theorem parallel_vectors_x_value :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-3, x]
  are_parallel a b → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2652_265207


namespace NUMINAMATH_CALUDE_repeating_decimal_inequality_l2652_265280

/-- Represents a repeating decimal with non-repeating part P and repeating part Q -/
structure RepeatingDecimal where
  P : ℕ  -- non-repeating part
  Q : ℕ  -- repeating part
  r : ℕ  -- number of digits in P
  s : ℕ  -- number of digits in Q

/-- The value of the repeating decimal as a real number -/
noncomputable def decimal_value (D : RepeatingDecimal) : ℝ :=
  sorry

/-- Statement: The equation 10^r(10^s + 1)D = Q(P + 1) is false for repeating decimals -/
theorem repeating_decimal_inequality (D : RepeatingDecimal) :
  (10^D.r * (10^D.s + 1)) * (decimal_value D) ≠ D.Q * (D.P + 1) :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_inequality_l2652_265280


namespace NUMINAMATH_CALUDE_small_cube_edge_length_l2652_265214

/-- Given a cube with volume 1000 cm³, prove that cutting off 8 small cubes
    of equal size from its corners, resulting in a remaining volume of 488 cm³,
    yields small cubes with edge length 4 cm. -/
theorem small_cube_edge_length 
  (initial_volume : ℝ) 
  (remaining_volume : ℝ) 
  (num_small_cubes : ℕ) 
  (h_initial : initial_volume = 1000)
  (h_remaining : remaining_volume = 488)
  (h_num_cubes : num_small_cubes = 8) :
  ∃ (edge_length : ℝ), 
    edge_length = 4 ∧ 
    initial_volume - num_small_cubes * edge_length ^ 3 = remaining_volume :=
by sorry

end NUMINAMATH_CALUDE_small_cube_edge_length_l2652_265214


namespace NUMINAMATH_CALUDE_pentagon_area_fraction_is_five_eighths_l2652_265244

/-- Represents the tiling pattern of a large square -/
structure TilingPattern where
  total_divisions : Nat
  pentagon_count : Nat
  square_count : Nat

/-- Calculates the fraction of area covered by pentagons in the tiling pattern -/
def pentagon_area_fraction (pattern : TilingPattern) : Rat :=
  pattern.pentagon_count / pattern.total_divisions

/-- Theorem stating that the fraction of area covered by pentagons is 5/8 -/
theorem pentagon_area_fraction_is_five_eighths (pattern : TilingPattern) :
  pattern.total_divisions = 16 →
  pattern.pentagon_count = 10 →
  pattern.square_count = 6 →
  pentagon_area_fraction pattern = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_fraction_is_five_eighths_l2652_265244


namespace NUMINAMATH_CALUDE_multiply_54_46_l2652_265276

theorem multiply_54_46 : 54 * 46 = 2484 := by
  sorry

end NUMINAMATH_CALUDE_multiply_54_46_l2652_265276


namespace NUMINAMATH_CALUDE_constant_sum_area_one_iff_identical_l2652_265237

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 1000
  col : Fin 1000
  value : ℝ

/-- Represents the entire 1000 × 1000 grid -/
def Grid := Cell → ℝ

/-- A rectangle within the grid -/
structure Rectangle where
  top_left : Cell
  bottom_right : Cell

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  (r.bottom_right.row - r.top_left.row + 1) * (r.bottom_right.col - r.top_left.col + 1)

/-- The sum of values in a rectangle -/
def sum_rectangle (g : Grid) (r : Rectangle) : ℝ := sorry

/-- Predicate: all rectangles of area s have the same sum -/
def constant_sum_for_area (g : Grid) (s : ℕ) : Prop :=
  ∀ r₁ r₂ : Rectangle, area r₁ = s → area r₂ = s → sum_rectangle g r₁ = sum_rectangle g r₂

/-- Predicate: all cells in the grid have the same value -/
def all_cells_identical (g : Grid) : Prop :=
  ∀ c₁ c₂ : Cell, g c₁ = g c₂

/-- Main theorem: constant sum for area 1 implies all cells are identical -/
theorem constant_sum_area_one_iff_identical :
  ∀ g : Grid, constant_sum_for_area g 1 ↔ all_cells_identical g :=
sorry

end NUMINAMATH_CALUDE_constant_sum_area_one_iff_identical_l2652_265237


namespace NUMINAMATH_CALUDE_reflect_point_1_2_l2652_265287

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The theorem states that reflecting the point (1,2) across the x-axis results in (1,-2) -/
theorem reflect_point_1_2 : reflect_x (1, 2) = (1, -2) := by sorry

end NUMINAMATH_CALUDE_reflect_point_1_2_l2652_265287


namespace NUMINAMATH_CALUDE_diagonal_passes_through_600_cubes_l2652_265290

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem: An internal diagonal of a 120 × 280 × 360 rectangular solid passes through 600 cubes -/
theorem diagonal_passes_through_600_cubes :
  cubes_passed_by_diagonal 120 280 360 = 600 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_600_cubes_l2652_265290


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l2652_265260

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l2652_265260


namespace NUMINAMATH_CALUDE_exists_small_angle_between_diagonals_l2652_265208

/-- A convex dodecagon -/
structure ConvexDodecagon where
  -- We don't need to define the structure explicitly for this problem

/-- A diagonal in a polygon -/
structure Diagonal where
  -- We don't need to define the structure explicitly for this problem

/-- The angle between two diagonals -/
def angle_between_diagonals (d1 d2 : Diagonal) : ℝ := sorry

/-- The number of sides in a dodecagon -/
def dodecagon_sides : ℕ := 12

/-- The number of diagonals in a dodecagon -/
def dodecagon_diagonals : ℕ := 54

/-- The sum of angles in a plane around a point -/
def sum_of_angles : ℝ := 360

/-- Theorem: In any convex dodecagon, there exist two diagonals forming an angle not exceeding 3° -/
theorem exists_small_angle_between_diagonals (d : ConvexDodecagon) :
  ∃ (d1 d2 : Diagonal), angle_between_diagonals d1 d2 ≤ 3 := by sorry

end NUMINAMATH_CALUDE_exists_small_angle_between_diagonals_l2652_265208


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2652_265240

/-- Proves that the ratio of Rommel's age to Tim's age is 3:1 -/
theorem age_ratio_proof (tim_age : ℕ) (rommel_age : ℕ) (jenny_age : ℕ) : 
  tim_age = 5 →
  jenny_age = rommel_age + 2 →
  jenny_age = tim_age + 12 →
  rommel_age / tim_age = 3 := by
  sorry

#check age_ratio_proof

end NUMINAMATH_CALUDE_age_ratio_proof_l2652_265240


namespace NUMINAMATH_CALUDE_carol_ate_twelve_cakes_l2652_265289

/-- The number of cakes Sara bakes per day -/
def cakes_per_day : ℕ := 10

/-- The number of days Sara bakes cakes -/
def baking_days : ℕ := 5

/-- The number of cans of frosting needed to frost a single cake -/
def cans_per_cake : ℕ := 2

/-- The number of cans of frosting Sara needs for the remaining cakes -/
def cans_needed : ℕ := 76

/-- The number of cakes Carol ate -/
def cakes_eaten_by_carol : ℕ := cakes_per_day * baking_days - cans_needed / cans_per_cake

theorem carol_ate_twelve_cakes : cakes_eaten_by_carol = 12 := by
  sorry

end NUMINAMATH_CALUDE_carol_ate_twelve_cakes_l2652_265289


namespace NUMINAMATH_CALUDE_second_train_length_l2652_265293

/-- Calculates the length of the second train given the parameters of two trains approaching each other. -/
theorem second_train_length 
  (length_train1 : ℝ) 
  (speed_train1 : ℝ) 
  (speed_train2 : ℝ) 
  (clear_time : ℝ) 
  (h1 : length_train1 = 120)
  (h2 : speed_train1 = 42)
  (h3 : speed_train2 = 30)
  (h4 : clear_time = 20.99832013438925) :
  ∃ (length_train2 : ℝ), 
    abs (length_train2 - 299.97) < 0.01 ∧ 
    length_train1 + length_train2 = (speed_train1 + speed_train2) * (1000 / 3600) * clear_time := by
  sorry

end NUMINAMATH_CALUDE_second_train_length_l2652_265293


namespace NUMINAMATH_CALUDE_angle_measure_l2652_265219

theorem angle_measure (x : ℝ) : 
  (180 - x = 2 * (90 - x) + 20) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l2652_265219


namespace NUMINAMATH_CALUDE_lemonade_glasses_served_l2652_265223

/-- The number of glasses of lemonade that can be served from a given number of pitchers. -/
def glasses_served (glasses_per_pitcher : ℕ) (num_pitchers : ℕ) : ℕ :=
  glasses_per_pitcher * num_pitchers

/-- Theorem stating that 6 pitchers of lemonade, each serving 5 glasses, can serve 30 glasses in total. -/
theorem lemonade_glasses_served :
  glasses_served 5 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_glasses_served_l2652_265223


namespace NUMINAMATH_CALUDE_unique_solution_is_identity_l2652_265279

open Set
open Function
open Real

/-- The functional equation that f must satisfy for all positive real numbers x, y, z -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y z, x > 0 → y > 0 → z > 0 →
    (z + 1) * f (x + y) = f (x * f z + y) + f (y * f z + x)

/-- The main theorem stating that the only function satisfying the equation is the identity function -/
theorem unique_solution_is_identity :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
    satisfies_equation f →
    ∀ x, x > 0 → f x = x :=
by sorry


end NUMINAMATH_CALUDE_unique_solution_is_identity_l2652_265279


namespace NUMINAMATH_CALUDE_count_numbers_satisfying_conditions_l2652_265229

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def starts_with_nine (n : ℕ) : Prop := ∃ (a b : ℕ), n = 900 + 90 * a + b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_three_digit n ∧
  starts_with_nine n ∧
  digit_sum n = 27 ∧
  Even n

theorem count_numbers_satisfying_conditions : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_conditions n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_satisfying_conditions_l2652_265229


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l2652_265284

theorem unique_solution_for_exponential_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (9 * x)^18 = (27 * x)^9 ∧ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l2652_265284


namespace NUMINAMATH_CALUDE_ella_age_l2652_265200

/-- Given the ages of Sam, Tim, and Ella, prove that Ella is 15 years old. -/
theorem ella_age (s t e : ℕ) : 
  (s + t + e) / 3 = 12 →  -- The average of their ages is 12
  e - 5 = s →             -- Five years ago, Ella was the same age as Sam is now
  t + 4 = (3 * (s + 4)) / 4 →  -- In 4 years, Tim's age will be 3/4 of Sam's age at that time
  e = 15 := by
sorry


end NUMINAMATH_CALUDE_ella_age_l2652_265200


namespace NUMINAMATH_CALUDE_triangle_area_l2652_265201

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A + B + C = π →
  a = 2 * Real.sqrt 3 →
  b + c = 4 →
  Real.cos B * Real.cos C - Real.sin B * Real.sin C = 1/2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2652_265201


namespace NUMINAMATH_CALUDE_line_A2A3_tangent_to_circle_M_l2652_265212

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola_C A.1 A.2

-- Define a line tangent to the circle M
def line_tangent_to_circle_M (A B : ℝ × ℝ) : Prop :=
  let d := abs ((B.2 - A.2) * 2 - (B.1 - A.1) * 0 + (A.1 * B.2 - B.1 * A.2)) /
            Real.sqrt ((B.2 - A.2)^2 + (B.1 - A.1)^2)
  d = 1

-- State the theorem
theorem line_A2A3_tangent_to_circle_M 
  (A₁ A₂ A₃ : ℝ × ℝ)
  (h₁ : point_on_parabola A₁)
  (h₂ : point_on_parabola A₂)
  (h₃ : point_on_parabola A₃)
  (h₄ : line_tangent_to_circle_M A₁ A₂)
  (h₅ : line_tangent_to_circle_M A₁ A₃) :
  line_tangent_to_circle_M A₂ A₃ := by sorry

end NUMINAMATH_CALUDE_line_A2A3_tangent_to_circle_M_l2652_265212


namespace NUMINAMATH_CALUDE_weight_plates_theorem_l2652_265242

/-- Calculates the effective weight of plates when lowered, considering technology and incline effects -/
def effectiveWeight (numPlates : ℕ) (plateWeight : ℝ) (techIncrease : ℝ) (inclineIncrease : ℝ) : ℝ :=
  let baseWeight := numPlates * plateWeight
  let withTech := baseWeight * (1 + techIncrease)
  withTech * (1 + inclineIncrease)

/-- Theorem: The effective weight of 10 plates of 30 pounds each, with 20% tech increase and 15% incline increase, is 414 pounds -/
theorem weight_plates_theorem :
  effectiveWeight 10 30 0.2 0.15 = 414 := by
  sorry


end NUMINAMATH_CALUDE_weight_plates_theorem_l2652_265242


namespace NUMINAMATH_CALUDE_seminar_discount_percentage_l2652_265271

/-- Calculates the discount percentage for early registration of a seminar --/
theorem seminar_discount_percentage
  (regular_fee : ℝ)
  (num_teachers : ℕ)
  (food_allowance : ℝ)
  (total_spent : ℝ)
  (h1 : regular_fee = 150)
  (h2 : num_teachers = 10)
  (h3 : food_allowance = 10)
  (h4 : total_spent = 1525)
  : (1 - (total_spent - num_teachers * food_allowance) / (num_teachers * regular_fee)) * 100 = 5 := by
  sorry

#check seminar_discount_percentage

end NUMINAMATH_CALUDE_seminar_discount_percentage_l2652_265271


namespace NUMINAMATH_CALUDE_fun_run_ratio_l2652_265298

def runners_last_year : ℕ := 200 - 40
def runners_this_year : ℕ := 320

theorem fun_run_ratio : 
  (runners_this_year : ℚ) / (runners_last_year : ℚ) = 2 := by sorry

end NUMINAMATH_CALUDE_fun_run_ratio_l2652_265298


namespace NUMINAMATH_CALUDE_dentists_age_l2652_265252

theorem dentists_age : ∃ (x : ℕ), 
  (x - 8) / 6 = (x + 8) / 10 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_dentists_age_l2652_265252


namespace NUMINAMATH_CALUDE_max_value_of_h_l2652_265218

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the function h as the sum of f and g
def h (x : ℝ) : ℝ := f x + g x

-- State the theorem
theorem max_value_of_h :
  (∀ x, -7 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -3 ≤ g x ∧ g x ≤ 2) →
  (∃ x, h x = 6) ∧ (∀ x, h x ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_h_l2652_265218


namespace NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l2652_265222

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Check if four points form a rectangle --/
def isRectangle (r : Rectangle) : Prop :=
  let (x1, y1) := r.v1
  let (x2, y2) := r.v2
  let (x3, y3) := r.v3
  let (x4, y4) := r.v4
  (x1 = x3 ∧ x2 = x4 ∧ y1 = y2 ∧ y3 = y4) ∨
  (x1 = x2 ∧ x3 = x4 ∧ y1 = y3 ∧ y2 = y4)

/-- The main theorem --/
theorem fourth_vertex_of_rectangle :
  ∀ (r : Rectangle),
    r.v1 = (1, 1) →
    r.v2 = (5, 1) →
    r.v3 = (1, 7) →
    isRectangle r →
    r.v4 = (5, 7) := by
  sorry


end NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l2652_265222


namespace NUMINAMATH_CALUDE_triangle_angle_b_l2652_265236

/-- In a triangle ABC, given that a cos B - b cos A = c and C = π/5, prove that B = 3π/10 -/
theorem triangle_angle_b (a b c A B C : ℝ) : 
  a * Real.cos B - b * Real.cos A = c →
  C = π / 5 →
  B = 3 * π / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_b_l2652_265236


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l2652_265245

theorem greatest_integer_radius (A : ℝ) (h : A < 100 * Real.pi) :
  ∃ (r : ℕ), r^2 * Real.pi ≤ A ∧ ∀ (s : ℕ), s^2 * Real.pi ≤ A → s ≤ r ∧ r = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l2652_265245


namespace NUMINAMATH_CALUDE_overtime_calculation_l2652_265204

/-- A worker's pay structure and hours worked --/
structure WorkerPay where
  ordinary_rate : ℚ  -- Rate for ordinary time in cents per hour
  overtime_rate : ℚ  -- Rate for overtime in cents per hour
  total_pay : ℚ      -- Total pay for the week in cents
  total_hours : ℕ    -- Total hours worked in the week

/-- Calculate the number of overtime hours --/
def overtime_hours (w : WorkerPay) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the overtime hours are 8 --/
theorem overtime_calculation (w : WorkerPay) 
  (h1 : w.ordinary_rate = 60)
  (h2 : w.overtime_rate = 90)
  (h3 : w.total_pay = 3240)
  (h4 : w.total_hours = 50) :
  overtime_hours w = 8 := by
  sorry

end NUMINAMATH_CALUDE_overtime_calculation_l2652_265204


namespace NUMINAMATH_CALUDE_percentage_of_men_l2652_265256

/-- Represents the composition of employees in a company -/
structure Company where
  men : ℝ
  women : ℝ
  men_french : ℝ
  women_french : ℝ

/-- The company satisfies the given conditions -/
def valid_company (c : Company) : Prop :=
  c.men + c.women = 100 ∧
  c.men_french = 0.6 * c.men ∧
  c.women_french = 0.35 * c.women ∧
  c.men_french + c.women_french = 50

/-- The theorem stating that 60% of the company employees are men -/
theorem percentage_of_men (c : Company) (h : valid_company c) : c.men = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_men_l2652_265256


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2652_265261

theorem fraction_sum_equals_decimal : 
  2/10 - 5/100 + 3/1000 + 8/10000 = 0.1538 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2652_265261


namespace NUMINAMATH_CALUDE_x_bounds_and_sqrt2_inequality_l2652_265295

theorem x_bounds_and_sqrt2_inequality :
  ∃ x : ℝ,
    (x = (x^2 + 1) / 198) ∧
    (1/198 < x) ∧
    (x < 197.99494949) ∧
    (Real.sqrt 2 < 1.41421356) := by
  sorry

end NUMINAMATH_CALUDE_x_bounds_and_sqrt2_inequality_l2652_265295


namespace NUMINAMATH_CALUDE_expression_simplification_l2652_265225

theorem expression_simplification (x : ℝ) :
  14 * (150 / 3 + 35 / 7 + 16 / 32 + x) = 777 + 14 * x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2652_265225


namespace NUMINAMATH_CALUDE_equilateral_triangle_count_is_twenty_l2652_265246

/-- Represents a point in the hexagonal lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The hexagonal lattice with 19 points -/
def HexagonalLattice : Set LatticePoint :=
  sorry

/-- Predicate to check if three points form an equilateral triangle -/
def IsEquilateralTriangle (p1 p2 p3 : LatticePoint) : Prop :=
  sorry

/-- The number of equilateral triangles in the lattice -/
def EquilateralTriangleCount : ℕ :=
  sorry

/-- Theorem stating that there are exactly 20 equilateral triangles in the lattice -/
theorem equilateral_triangle_count_is_twenty :
  EquilateralTriangleCount = 20 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_count_is_twenty_l2652_265246


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2652_265285

theorem decimal_to_fraction : (3.56 : ℚ) = 89 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2652_265285


namespace NUMINAMATH_CALUDE_unique_solution_l2652_265296

def a : Fin 3 → ℝ := ![2, 2, 2]
def b : Fin 3 → ℝ := ![3, -2, 1]
def c : Fin 3 → ℝ := ![3, 3, -4]

def orthogonal (u v : Fin 3 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) + (u 2) * (v 2) = 0

theorem unique_solution :
  orthogonal a b ∧ orthogonal b c ∧ orthogonal a c →
  ∃! (p q r : ℝ), ∀ i : Fin 3,
    (![3, -1, 8] i) = p * (a i) + q * (b i) + r * (c i) ∧
    p = 5/3 ∧ q = 0 ∧ r = -10/17 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2652_265296


namespace NUMINAMATH_CALUDE_relationship_between_m_and_a_l2652_265217

theorem relationship_between_m_and_a (m : ℕ) (a : ℝ) 
  (h1 : m > 0) (h2 : a > 0) :
  ((∀ n : ℕ, n > m → (1 : ℝ) / n < a) ∧ 
   (∀ n : ℕ, 0 < n ∧ n ≤ m → (1 : ℝ) / n ≥ a)) ↔ 
  m = ⌊(1 : ℝ) / a⌋ := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_m_and_a_l2652_265217


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_set_l2652_265221

-- Define the cubic polynomial function
def f (x : ℝ) : ℝ := -3 * x^3 + 5 * x^2 - 2 * x + 1

-- State the theorem
theorem cubic_inequality_solution_set :
  ∀ x : ℝ, f x > 0 ↔ (x > -1 ∧ x < 1/3) ∨ x > 1 :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_set_l2652_265221


namespace NUMINAMATH_CALUDE_andrews_age_l2652_265274

theorem andrews_age (a g s : ℝ) 
  (h1 : g = 10 * a)
  (h2 : g - s = a + 45)
  (h3 : s = 5) :
  a = 50 / 9 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l2652_265274


namespace NUMINAMATH_CALUDE_problem_I3_1_l2652_265249

theorem problem_I3_1 (w x y z : ℝ) (hw : w > 0) 
  (h1 : w * x * y * z = 4) (h2 : w - x * y * z = 3) : w = 4 := by
sorry


end NUMINAMATH_CALUDE_problem_I3_1_l2652_265249


namespace NUMINAMATH_CALUDE_certain_number_problem_l2652_265278

theorem certain_number_problem (x : ℝ) : 0.7 * x = 0.6 * 80 + 22 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2652_265278


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2652_265227

/-- A curve in the xy-plane defined by x = t^2 - 4 and y = t^3 - 6t + 4 for all real t. -/
def curve (t : ℝ) : ℝ × ℝ :=
  (t^2 - 4, t^3 - 6*t + 4)

/-- The point where the curve crosses itself. -/
def self_intersection_point : ℝ × ℝ := (2, 4)

/-- Theorem stating that the curve crosses itself at the point (2, 4). -/
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = self_intersection_point :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2652_265227


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2652_265231

theorem inequality_solution_range (a : ℝ) : 
  (∀ x, (a - 1) * x < 1 ↔ x > 1 / (a - 1)) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2652_265231


namespace NUMINAMATH_CALUDE_pet_store_birds_l2652_265234

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 9

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds :
  total_birds = 36 := by sorry

end NUMINAMATH_CALUDE_pet_store_birds_l2652_265234


namespace NUMINAMATH_CALUDE_min_value_trig_expression_greatest_lower_bound_trig_expression_l2652_265268

theorem min_value_trig_expression (x : ℝ) : 
  (4 * Real.sin x * Real.cos x + 3) / (Real.cos x)^2 ≥ 5/3 :=
by sorry

theorem greatest_lower_bound_trig_expression :
  ∃ x : ℝ, (4 * Real.sin x * Real.cos x + 3) / (Real.cos x)^2 = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_greatest_lower_bound_trig_expression_l2652_265268


namespace NUMINAMATH_CALUDE_solve_equation_l2652_265238

theorem solve_equation (x : ℝ) (h : (40 / x) - 1 = 19) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2652_265238


namespace NUMINAMATH_CALUDE_modulus_of_complex_expression_l2652_265265

theorem modulus_of_complex_expression :
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I) + 2 * Complex.I
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_expression_l2652_265265


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2652_265250

/-- The perimeter of an isosceles triangle given specific conditions -/
theorem isosceles_triangle_perimeter : 
  ∀ (equilateral_perimeter isosceles_base : ℝ),
  equilateral_perimeter = 60 →
  isosceles_base = 15 →
  ∃ (isosceles_perimeter : ℝ),
  isosceles_perimeter = equilateral_perimeter / 3 + equilateral_perimeter / 3 + isosceles_base ∧
  isosceles_perimeter = 55 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2652_265250


namespace NUMINAMATH_CALUDE_jens_ducks_l2652_265291

theorem jens_ducks (chickens ducks : ℕ) : 
  ducks = 4 * chickens + 10 →
  chickens + ducks = 185 →
  ducks = 150 := by
sorry

end NUMINAMATH_CALUDE_jens_ducks_l2652_265291


namespace NUMINAMATH_CALUDE_ariel_birth_year_l2652_265216

/-- Calculates the birth year of a person given their fencing start year, years of fencing, and current age. -/
def birth_year (fencing_start_year : ℕ) (years_fencing : ℕ) (current_age : ℕ) : ℕ :=
  fencing_start_year - (current_age - years_fencing)

/-- Proves that Ariel's birth year is 1992 given the provided conditions. -/
theorem ariel_birth_year :
  let fencing_start_year : ℕ := 2006
  let years_fencing : ℕ := 16
  let current_age : ℕ := 30
  birth_year fencing_start_year years_fencing current_age = 1992 := by
  sorry

#eval birth_year 2006 16 30

end NUMINAMATH_CALUDE_ariel_birth_year_l2652_265216


namespace NUMINAMATH_CALUDE_train_length_calculation_l2652_265259

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 30 → time_s = 6 → 
  ∃ (length_m : ℝ), (abs (length_m - 50) < 1 ∧ length_m = speed_kmh * (1000 / 3600) * time_s) :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2652_265259


namespace NUMINAMATH_CALUDE_charlie_fruit_bags_l2652_265267

theorem charlie_fruit_bags : 0.17 + 0.17 + 0.33 = 0.67 := by sorry

end NUMINAMATH_CALUDE_charlie_fruit_bags_l2652_265267


namespace NUMINAMATH_CALUDE_correct_time_per_lap_l2652_265288

/-- The time in minutes for one lap around the playground -/
def time_per_lap : ℝ := 19.2

/-- The number of laps cycled -/
def num_laps : ℕ := 5

/-- The total time in minutes for cycling the given number of laps -/
def total_time : ℝ := 96

theorem correct_time_per_lap : 
  time_per_lap * num_laps = total_time := by sorry

end NUMINAMATH_CALUDE_correct_time_per_lap_l2652_265288


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2652_265266

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 9 / b ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2652_265266


namespace NUMINAMATH_CALUDE_remaining_length_is_21_l2652_265263

/-- A polygon with perpendicular adjacent sides -/
structure PerpendicularPolygon where
  left : ℝ
  top : ℝ
  right : ℝ
  bottom_removed : List ℝ

/-- The total length of remaining segments after removal -/
def remaining_length (p : PerpendicularPolygon) : ℝ :=
  p.left + p.top + p.right

theorem remaining_length_is_21 (p : PerpendicularPolygon)
  (h1 : p.left = 10)
  (h2 : p.top = 3)
  (h3 : p.right = 8)
  (h4 : p.bottom_removed = [2, 1, 2]) :
  remaining_length p = 21 := by
  sorry

end NUMINAMATH_CALUDE_remaining_length_is_21_l2652_265263


namespace NUMINAMATH_CALUDE_sodas_bought_example_l2652_265206

/-- Given a total cost, sandwich price, number of sandwiches, and soda price,
    calculate the number of sodas bought. -/
def sodas_bought (total_cost sandwich_price num_sandwiches soda_price : ℚ) : ℚ :=
  (total_cost - num_sandwiches * sandwich_price) / soda_price

theorem sodas_bought_example : 
  sodas_bought 8.38 2.45 2 0.87 = 4 := by sorry

end NUMINAMATH_CALUDE_sodas_bought_example_l2652_265206


namespace NUMINAMATH_CALUDE_halloween_jelly_beans_l2652_265251

theorem halloween_jelly_beans 
  (initial_jelly_beans : ℕ)
  (total_children : ℕ)
  (jelly_beans_per_child : ℕ)
  (remaining_jelly_beans : ℕ)
  (h1 : initial_jelly_beans = 100)
  (h2 : total_children = 40)
  (h3 : jelly_beans_per_child = 2)
  (h4 : remaining_jelly_beans = 36)
  : (((initial_jelly_beans - remaining_jelly_beans) / jelly_beans_per_child) / total_children) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_halloween_jelly_beans_l2652_265251


namespace NUMINAMATH_CALUDE_least_number_of_cans_l2652_265209

theorem least_number_of_cans (maaza pepsi sprite : ℕ) 
  (h_maaza : maaza = 10)
  (h_pepsi : pepsi = 144)
  (h_sprite : sprite = 368) :
  let gcd_all := Nat.gcd maaza (Nat.gcd pepsi sprite)
  ∃ (can_size : ℕ), 
    can_size = gcd_all ∧ 
    can_size > 0 ∧
    maaza % can_size = 0 ∧ 
    pepsi % can_size = 0 ∧ 
    sprite % can_size = 0 ∧
    (maaza / can_size + pepsi / can_size + sprite / can_size) = 261 :=
by sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l2652_265209


namespace NUMINAMATH_CALUDE_expression_value_l2652_265233

theorem expression_value : (49 + 5)^2 - (5^2 + 49^2) = 490 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2652_265233


namespace NUMINAMATH_CALUDE_sample_size_theorem_l2652_265264

theorem sample_size_theorem (frequency_sum : ℝ) (frequency_ratio : ℝ) 
  (h1 : frequency_sum = 20) 
  (h2 : frequency_ratio = 0.4) : 
  frequency_sum / frequency_ratio = 50 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_theorem_l2652_265264


namespace NUMINAMATH_CALUDE_equation_equivalence_l2652_265247

theorem equation_equivalence (x y : ℝ) :
  y^2 - 2*x*y + x^2 - 1 = 0 ↔ (y = x + 1 ∨ y = x - 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2652_265247


namespace NUMINAMATH_CALUDE_bianca_deleted_files_l2652_265235

/-- The number of pictures Bianca deleted -/
def pictures : ℕ := 5

/-- The number of songs Bianca deleted -/
def songs : ℕ := 12

/-- The number of text files Bianca deleted -/
def text_files : ℕ := 10

/-- The number of video files Bianca deleted -/
def video_files : ℕ := 6

/-- The total number of files Bianca deleted -/
def total_files : ℕ := pictures + songs + text_files + video_files

theorem bianca_deleted_files : total_files = 33 := by
  sorry

end NUMINAMATH_CALUDE_bianca_deleted_files_l2652_265235


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_144_l2652_265273

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_144_l2652_265273


namespace NUMINAMATH_CALUDE_hash_2_3_4_l2652_265213

/-- The # operation defined on three real numbers -/
def hash (a b c : ℝ) : ℝ := (b + 1)^2 - 4*a*(c - 1)

/-- Theorem stating that #(2, 3, 4) = -8 -/
theorem hash_2_3_4 : hash 2 3 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_3_4_l2652_265213


namespace NUMINAMATH_CALUDE_sum_of_i_powers_2021_to_2024_l2652_265283

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem: The sum of i^2021, i^2022, i^2023, and i^2024 is equal to 0 -/
theorem sum_of_i_powers_2021_to_2024 : i^2021 + i^2022 + i^2023 + i^2024 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_2021_to_2024_l2652_265283


namespace NUMINAMATH_CALUDE_price_difference_l2652_265232

/-- Calculates the difference between the final retail price and the average price
    customers paid for the first 150 garments under a special pricing scheme. -/
theorem price_difference (original_price : ℝ) (first_increase : ℝ) (second_increase : ℝ)
  (special_rate1 : ℝ) (special_rate2 : ℝ) (special_quantity1 : ℕ) (special_quantity2 : ℕ)
  (h1 : original_price = 50)
  (h2 : first_increase = 0.3)
  (h3 : second_increase = 0.15)
  (h4 : special_rate1 = 0.7)
  (h5 : special_rate2 = 0.85)
  (h6 : special_quantity1 = 50)
  (h7 : special_quantity2 = 100) :
  let final_price := original_price * (1 + first_increase) * (1 + second_increase)
  let special_price1 := final_price * special_rate1
  let special_price2 := final_price * special_rate2
  let total_special_price := special_price1 * special_quantity1 + special_price2 * special_quantity2
  let avg_special_price := total_special_price / (special_quantity1 + special_quantity2)
  final_price - avg_special_price = 14.95 := by
sorry

end NUMINAMATH_CALUDE_price_difference_l2652_265232


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2652_265269

theorem min_value_quadratic (x y : ℝ) : 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2652_265269


namespace NUMINAMATH_CALUDE_crocus_bulbs_count_l2652_265241

/-- Represents the number of crocus bulbs that can be bought given the constraints -/
def crocus_bulbs : ℕ := 22

/-- Represents the number of daffodil bulbs that can be bought given the constraints -/
def daffodil_bulbs : ℕ := 55 - crocus_bulbs

/-- The total number of bulbs -/
def total_bulbs : ℕ := 55

/-- The cost of a single crocus bulb in cents -/
def crocus_cost : ℕ := 35

/-- The cost of a single daffodil bulb in cents -/
def daffodil_cost : ℕ := 65

/-- The total budget in cents -/
def total_budget : ℕ := 2915

theorem crocus_bulbs_count : 
  crocus_bulbs = 22 ∧ 
  crocus_bulbs + daffodil_bulbs = total_bulbs ∧ 
  crocus_bulbs * crocus_cost + daffodil_bulbs * daffodil_cost = total_budget := by
  sorry

end NUMINAMATH_CALUDE_crocus_bulbs_count_l2652_265241


namespace NUMINAMATH_CALUDE_quiz_probability_theorem_l2652_265228

/-- The number of questions in the quiz -/
def total_questions : ℕ := 30

/-- The number of answer choices for each question -/
def choices_per_question : ℕ := 6

/-- The number of questions Emily guesses randomly -/
def guessed_questions : ℕ := 5

/-- The probability of guessing a single question correctly -/
def prob_correct : ℚ := 1 / choices_per_question

/-- The probability of guessing a single question incorrectly -/
def prob_incorrect : ℚ := 1 - prob_correct

/-- The probability of guessing at least two out of five questions correctly -/
def prob_at_least_two_correct : ℚ := 763 / 3888

theorem quiz_probability_theorem :
  (1 : ℚ) - (prob_incorrect ^ guessed_questions + 
    (guessed_questions : ℚ) * prob_correct * prob_incorrect ^ (guessed_questions - 1)) = 
  prob_at_least_two_correct :=
sorry

end NUMINAMATH_CALUDE_quiz_probability_theorem_l2652_265228


namespace NUMINAMATH_CALUDE_projection_matrix_values_l2652_265294

def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  Q ^ 2 = Q

theorem projection_matrix_values :
  ∀ (x y : ℝ),
  let Q : Matrix (Fin 2) (Fin 2) ℝ := !![x, 1/5; y, 4/5]
  is_projection_matrix Q ↔ x = 1 ∧ y = 0 := by
sorry


end NUMINAMATH_CALUDE_projection_matrix_values_l2652_265294


namespace NUMINAMATH_CALUDE_smallest_advantageous_discount_l2652_265202

def is_more_advantageous (n : ℕ) : Prop :=
  (1 - n / 100 : ℝ) < (1 - 0.2)^2 ∧
  (1 - n / 100 : ℝ) < (1 - 0.15)^3 ∧
  (1 - n / 100 : ℝ) < (1 - 0.3) * (1 - 0.1)

theorem smallest_advantageous_discount : 
  (∀ m : ℕ, m < 39 → ¬(is_more_advantageous m)) ∧ 
  is_more_advantageous 39 := by
  sorry

end NUMINAMATH_CALUDE_smallest_advantageous_discount_l2652_265202


namespace NUMINAMATH_CALUDE_problem_solution_l2652_265243

noncomputable def f (m n x : ℝ) : ℝ := m * x + 1 / (n * x) + 1 / 2

theorem problem_solution (m n : ℝ) 
  (h1 : f m n 1 = 2) 
  (h2 : f m n 2 = 11 / 4) :
  (m = 1 ∧ n = 2) ∧ 
  (∀ x y, 1 ≤ x → x < y → f m n x < f m n y) ∧
  (∀ x : ℝ, f m n (1 + 2 * x^2) > f m n (x^2 - 2 * x + 4) ↔ x < -3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2652_265243


namespace NUMINAMATH_CALUDE_linear_system_solution_l2652_265215

theorem linear_system_solution (x y m : ℝ) : 
  (2 * x + y = 7) → 
  (x + 2 * y = m - 3) → 
  (x - y = 2) → 
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l2652_265215


namespace NUMINAMATH_CALUDE_limit_proof_l2652_265275

open Real

theorem limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, 0 < |x - 1/2| ∧ |x - 1/2| < δ →
    |(2*x^2 - 5*x + 2)/(x - 1/2) + 3| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_proof_l2652_265275


namespace NUMINAMATH_CALUDE_line_intersects_y_axis_l2652_265272

/-- A line in 2D space defined by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The y-axis, represented as a vertical line with x-coordinate 0 -/
def yAxis : Line := { point1 := (0, 0), point2 := (0, 1) }

/-- Function to determine if a point lies on a given line -/
def pointOnLine (l : Line) (p : ℝ × ℝ) : Prop :=
  let (x1, y1) := l.point1
  let (x2, y2) := l.point2
  let (x, y) := p
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

/-- Function to determine if a point lies on the y-axis -/
def pointOnYAxis (p : ℝ × ℝ) : Prop :=
  p.1 = 0

/-- The main theorem to be proved -/
theorem line_intersects_y_axis :
  let l : Line := { point1 := (2, 3), point2 := (6, -9) }
  pointOnLine l (0, 9) ∧ pointOnYAxis (0, 9) := by sorry

end NUMINAMATH_CALUDE_line_intersects_y_axis_l2652_265272


namespace NUMINAMATH_CALUDE_batsman_matches_l2652_265281

theorem batsman_matches (total_matches : ℕ) (first_set_matches : ℕ) (first_set_avg : ℝ) 
                         (second_set_avg : ℝ) (total_avg : ℝ) :
  total_matches = 30 →
  first_set_matches = 20 →
  first_set_avg = 30 →
  second_set_avg = 15 →
  total_avg = 25 →
  (total_matches - first_set_matches : ℝ) = 10 := by
  sorry


end NUMINAMATH_CALUDE_batsman_matches_l2652_265281


namespace NUMINAMATH_CALUDE_divisor_sum_theorem_l2652_265224

def sum_of_divisors (i j k : ℕ) : ℕ :=
  (2^(i+1) - 1) * (3^(j+1) - 1) * (5^(k+1) - 1) / ((2-1) * (3-1) * (5-1))

theorem divisor_sum_theorem (i j k : ℕ) :
  sum_of_divisors i j k = 1200 → i + j + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_theorem_l2652_265224
