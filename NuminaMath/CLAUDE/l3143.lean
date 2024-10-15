import Mathlib

namespace NUMINAMATH_CALUDE_fly_path_length_l3143_314303

theorem fly_path_length (r : ℝ) (path_end : ℝ) (h1 : r = 100) (h2 : path_end = 120) : 
  let diameter := 2 * r
  let chord := Real.sqrt (diameter^2 - path_end^2)
  diameter + chord + path_end = 480 := by sorry

end NUMINAMATH_CALUDE_fly_path_length_l3143_314303


namespace NUMINAMATH_CALUDE_house_transaction_loss_l3143_314340

def house_transaction (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : ℝ :=
  let first_sale := initial_value * (1 - loss_percent)
  let second_sale := first_sale * (1 + gain_percent)
  initial_value - second_sale

theorem house_transaction_loss :
  house_transaction 9000 0.1 0.1 = 810 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_loss_l3143_314340


namespace NUMINAMATH_CALUDE_point_divides_segment_l3143_314391

/-- Given two points A and B in 2D space, and a point P, this function checks if P divides the line segment AB in the given ratio m:n -/
def divides_segment (A B P : ℝ × ℝ) (m n : ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x, y) := P
  x = (m * x₂ + n * x₁) / (m + n) ∧
  y = (m * y₂ + n * y₁) / (m + n)

/-- The theorem states that the point (3.5, 8.5) divides the line segment between (2, 10) and (8, 4) in the ratio 1:3 -/
theorem point_divides_segment :
  divides_segment (2, 10) (8, 4) (3.5, 8.5) 1 3 := by
  sorry

end NUMINAMATH_CALUDE_point_divides_segment_l3143_314391


namespace NUMINAMATH_CALUDE_fraction_units_and_exceed_l3143_314355

def fraction_units (numerator denominator : ℕ) : ℕ := numerator

def units_to_exceed (start target : ℕ) : ℕ :=
  if start ≥ target then 1 else target - start + 1

theorem fraction_units_and_exceed :
  (fraction_units 5 8 = 5) ∧
  (units_to_exceed 5 8 = 4) := by sorry

end NUMINAMATH_CALUDE_fraction_units_and_exceed_l3143_314355


namespace NUMINAMATH_CALUDE_angle_value_l3143_314369

theorem angle_value (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan (α + β) = 3 →
  Real.tan β = 1/2 →
  α = π/4 := by sorry

end NUMINAMATH_CALUDE_angle_value_l3143_314369


namespace NUMINAMATH_CALUDE_square_diagonals_are_equal_l3143_314333

-- Define the basic shapes
class Square
class Parallelogram

-- Define the property of having equal diagonals
def has_equal_diagonals (α : Type*) := Prop

-- State the given conditions
axiom square_equal_diagonals : has_equal_diagonals Square
axiom parallelogram_equal_diagonals : has_equal_diagonals Parallelogram
axiom square_is_parallelogram : Square → Parallelogram

-- State the theorem to be proved
theorem square_diagonals_are_equal : has_equal_diagonals Square := by
  sorry

end NUMINAMATH_CALUDE_square_diagonals_are_equal_l3143_314333


namespace NUMINAMATH_CALUDE_smallest_n_not_prime_l3143_314365

theorem smallest_n_not_prime (n : ℕ) : 
  (∀ k < n, Nat.Prime (2^k + 1)) ∧ ¬(Nat.Prime (2^n + 1)) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_not_prime_l3143_314365


namespace NUMINAMATH_CALUDE_tournament_committee_count_l3143_314334

/-- The number of teams in the frisbee league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the host team for the committee -/
def host_committee_size : ℕ := 4

/-- The number of members selected from each non-host team for the committee -/
def non_host_committee_size : ℕ := 3

/-- The total size of the tournament committee -/
def total_committee_size : ℕ := 13

/-- The number of possible tournament committees -/
def num_possible_committees : ℕ := 3443073600

theorem tournament_committee_count :
  (num_teams * (Nat.choose team_size host_committee_size) * 
   (Nat.choose team_size non_host_committee_size) ^ (num_teams - 1)) = num_possible_committees :=
by sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l3143_314334


namespace NUMINAMATH_CALUDE_bread_baking_pattern_l3143_314358

/-- A sequence of bread loaves baked over 6 days -/
def BreadSequence : Type := Fin 6 → ℕ

/-- The condition that the daily increase grows by 1 each day -/
def IncreasingDifference (s : BreadSequence) : Prop :=
  ∀ i : Fin 4, s (i + 1) - s i < s (i + 2) - s (i + 1)

/-- The known values for days 1, 2, 3, 4, and 6 -/
def KnownValues (s : BreadSequence) : Prop :=
  s 0 = 5 ∧ s 1 = 7 ∧ s 2 = 10 ∧ s 3 = 14 ∧ s 5 = 25

theorem bread_baking_pattern (s : BreadSequence) 
  (h1 : IncreasingDifference s) (h2 : KnownValues s) : s 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_bread_baking_pattern_l3143_314358


namespace NUMINAMATH_CALUDE_celeste_song_probability_l3143_314332

/-- Represents the collection of songs on Celeste's o-Pod -/
structure SongCollection where
  total_songs : Nat
  shortest_song : Nat
  song_increment : Nat
  favorite_song_length : Nat
  time_limit : Nat

/-- Calculates the probability of not hearing the entire favorite song 
    within the time limit for a given song collection -/
def probability_not_hearing_favorite (sc : SongCollection) : Rat :=
  1 - (Nat.factorial (sc.total_songs - 1) + Nat.factorial (sc.total_songs - 2)) / 
      Nat.factorial sc.total_songs

/-- The main theorem stating the probability for Celeste's specific case -/
theorem celeste_song_probability : 
  let sc : SongCollection := {
    total_songs := 12,
    shortest_song := 45,
    song_increment := 15,
    favorite_song_length := 240,
    time_limit := 300
  }
  probability_not_hearing_favorite sc = 10 / 11 := by
  sorry


end NUMINAMATH_CALUDE_celeste_song_probability_l3143_314332


namespace NUMINAMATH_CALUDE_square_fencing_cost_l3143_314347

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The cost of fencing each side of the square in dollars -/
def cost_per_side : ℕ := 79

/-- The total cost of fencing a square -/
def total_cost : ℕ := square_sides * cost_per_side

theorem square_fencing_cost : total_cost = 316 := by
  sorry

end NUMINAMATH_CALUDE_square_fencing_cost_l3143_314347


namespace NUMINAMATH_CALUDE_simplify_expression_l3143_314399

theorem simplify_expression (y : ℝ) : 3*y + 4*y^2 + 2 - (5 - (3*y + 4*y^2) - 8) = 8*y^2 + 6*y + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3143_314399


namespace NUMINAMATH_CALUDE_bags_sold_thursday_l3143_314383

/-- Calculates the number of bags sold on Thursday given the total stock and sales on other days --/
theorem bags_sold_thursday (total_stock : ℕ) (monday_sales tuesday_sales wednesday_sales friday_sales : ℕ)
  (h1 : total_stock = 600)
  (h2 : monday_sales = 25)
  (h3 : tuesday_sales = 70)
  (h4 : wednesday_sales = 100)
  (h5 : friday_sales = 145)
  (h6 : (total_stock : ℚ) * (25 : ℚ) / 100 = total_stock - (monday_sales + tuesday_sales + wednesday_sales + friday_sales + 110)) :
  110 = total_stock - (monday_sales + tuesday_sales + wednesday_sales + friday_sales + (total_stock : ℚ) * (25 : ℚ) / 100) :=
by sorry

end NUMINAMATH_CALUDE_bags_sold_thursday_l3143_314383


namespace NUMINAMATH_CALUDE_angle_FDE_l3143_314386

theorem angle_FDE (BAC : Real) (h : BAC = 70) : ∃ FDE : Real, FDE = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_FDE_l3143_314386


namespace NUMINAMATH_CALUDE_anyas_age_l3143_314329

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem anyas_age :
  ∃ (age : ℕ), 
    110 ≤ sum_of_first_n age ∧ 
    sum_of_first_n age ≤ 130 ∧ 
    age = 15 := by
  sorry

end NUMINAMATH_CALUDE_anyas_age_l3143_314329


namespace NUMINAMATH_CALUDE_phi_value_for_symmetric_sine_l3143_314362

theorem phi_value_for_symmetric_sine (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  f (π / 3) = 1 / 2 →
  f (-π / 3) = 1 / 2 →
  ∃ k : ℤ, φ = 2 * k * π - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_for_symmetric_sine_l3143_314362


namespace NUMINAMATH_CALUDE_equation_real_solutions_l3143_314313

theorem equation_real_solutions (a b x : ℝ) : 
  (∃ x : ℝ, Real.sqrt (2 * a + b + 2 * x) + Real.sqrt (10 * a + 9 * b - 6 * x) = 2 * Real.sqrt (2 * a + b - 2 * x)) ↔ 
  ((0 ≤ a ∧ -a ≤ b ∧ b ≤ 0 ∧ (x = Real.sqrt (a * (a + b)) ∨ x = -Real.sqrt (a * (a + b)))) ∨
   (a ≥ -8/9 * a ∧ -8/9 * a ≥ b ∧ b ≤ 0 ∧ x = -Real.sqrt (a * (a + b)))) :=
by sorry

end NUMINAMATH_CALUDE_equation_real_solutions_l3143_314313


namespace NUMINAMATH_CALUDE_circle_not_intersecting_diagonal_probability_l3143_314335

/-- The probability that a circle of radius 1 randomly placed inside a 15 × 36 rectangle
    does not intersect the diagonal of the rectangle -/
theorem circle_not_intersecting_diagonal_probability : ℝ := by
  -- Define the rectangle dimensions
  let rectangle_width : ℝ := 15
  let rectangle_height : ℝ := 36
  
  -- Define the circle radius
  let circle_radius : ℝ := 1
  
  -- Define the valid region for circle center
  let valid_region_width : ℝ := rectangle_width - 2 * circle_radius
  let valid_region_height : ℝ := rectangle_height - 2 * circle_radius
  
  -- Calculate the area of the valid region
  let valid_region_area : ℝ := valid_region_width * valid_region_height
  
  -- Define the safe area where the circle doesn't intersect the diagonal
  let safe_area : ℝ := 375
  
  -- Calculate the probability
  let probability : ℝ := safe_area / valid_region_area
  
  -- Prove that the probability equals 375/442
  sorry

#eval (375 : ℚ) / 442

end NUMINAMATH_CALUDE_circle_not_intersecting_diagonal_probability_l3143_314335


namespace NUMINAMATH_CALUDE_bakery_chairs_count_l3143_314301

theorem bakery_chairs_count :
  let indoor_tables : ℕ := 8
  let outdoor_tables : ℕ := 12
  let chairs_per_indoor_table : ℕ := 3
  let chairs_per_outdoor_table : ℕ := 3
  let total_chairs := indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table
  total_chairs = 60 := by
  sorry

end NUMINAMATH_CALUDE_bakery_chairs_count_l3143_314301


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3143_314359

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, 2 * x + 1 ≤ 0) ↔ (∀ x : ℝ, 2 * x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3143_314359


namespace NUMINAMATH_CALUDE_door_crank_time_l3143_314368

/-- Represents the time taken for various parts of the game show challenge -/
structure GameShowTimes where
  firstRun : Nat  -- Time for first run in seconds
  secondRun : Nat -- Time for second run in seconds
  totalTime : Nat -- Total time for the entire event in seconds

/-- Calculates the time taken to crank open the door -/
def timeToCrankDoor (times : GameShowTimes) : Nat :=
  times.totalTime - (times.firstRun + times.secondRun)

/-- Theorem stating that the time to crank open the door is 73 seconds -/
theorem door_crank_time (times : GameShowTimes) 
  (h1 : times.firstRun = 7 * 60 + 23)
  (h2 : times.secondRun = 5 * 60 + 58)
  (h3 : times.totalTime = 874) :
  timeToCrankDoor times = 73 := by
  sorry

#eval timeToCrankDoor { firstRun := 7 * 60 + 23, secondRun := 5 * 60 + 58, totalTime := 874 }

end NUMINAMATH_CALUDE_door_crank_time_l3143_314368


namespace NUMINAMATH_CALUDE_tank_capacity_correct_l3143_314360

/-- The capacity of a tank in gallons -/
def tank_capacity : ℝ := 32

/-- The total amount of oil in gallons -/
def total_oil : ℝ := 728

/-- The number of tanks needed -/
def num_tanks : ℕ := 23

/-- Theorem stating that the tank capacity is approximately correct -/
theorem tank_capacity_correct : 
  ∃ ε > 0, ε < 1 ∧ |tank_capacity - total_oil / num_tanks| < ε :=
sorry

end NUMINAMATH_CALUDE_tank_capacity_correct_l3143_314360


namespace NUMINAMATH_CALUDE_profit_percent_when_cost_is_40_percent_of_selling_price_l3143_314396

theorem profit_percent_when_cost_is_40_percent_of_selling_price :
  ∀ (selling_price : ℝ), selling_price > 0 →
  let cost_price := 0.4 * selling_price
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  profit_percent = 150 := by
sorry

end NUMINAMATH_CALUDE_profit_percent_when_cost_is_40_percent_of_selling_price_l3143_314396


namespace NUMINAMATH_CALUDE_vector_simplification_l3143_314357

variable {V : Type*} [AddCommGroup V]

variable (A B C M O : V)

theorem vector_simplification :
  (A - B + M - B) + (B - O + B - C) + (O - M) = A - C :=
by sorry

end NUMINAMATH_CALUDE_vector_simplification_l3143_314357


namespace NUMINAMATH_CALUDE_ones_digit_of_8_power_50_l3143_314306

theorem ones_digit_of_8_power_50 : 8^50 % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_8_power_50_l3143_314306


namespace NUMINAMATH_CALUDE_intersection_M_N_l3143_314322

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3143_314322


namespace NUMINAMATH_CALUDE_m_range_for_three_integer_solutions_l3143_314389

def inequality_system (x m : ℝ) : Prop :=
  2 * x - 1 ≤ 5 ∧ x - m > 0

def has_three_integer_solutions (m : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℤ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    inequality_system x₁ m ∧ inequality_system x₂ m ∧ inequality_system x₃ m ∧
    ∀ x : ℤ, inequality_system x m → x = x₁ ∨ x = x₂ ∨ x = x₃

theorem m_range_for_three_integer_solutions :
  ∀ m : ℝ, has_three_integer_solutions m ↔ 0 ≤ m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_m_range_for_three_integer_solutions_l3143_314389


namespace NUMINAMATH_CALUDE_solution_difference_l3143_314381

theorem solution_difference (r s : ℝ) : 
  ((r - 5) * (r + 5) = 26 * r - 130) →
  ((s - 5) * (s + 5) = 26 * s - 130) →
  r ≠ s →
  r > s →
  r - s = 16 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l3143_314381


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l3143_314350

-- Problem 1
theorem problem_one : 4 * Real.sqrt 2 + Real.sqrt 8 - Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_two : Real.sqrt (1 + 1/3) / Real.sqrt (2 + 1/3) * Real.sqrt (1 + 2/5) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l3143_314350


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l3143_314388

def is_arithmetic (s : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, s (n + 1) = s n + d

def seq1 (n : ℕ) : ℚ := n + 4
def seq2 (n : ℕ) : ℚ := if n % 2 = 0 then 3 - 3 * (n / 2) else 0
def seq3 (n : ℕ) : ℚ := 0
def seq4 (n : ℕ) : ℚ := (n + 1) / 10

theorem arithmetic_sequence_count :
  (is_arithmetic seq1) ∧
  (¬ is_arithmetic seq2) ∧
  (is_arithmetic seq3) ∧
  (is_arithmetic seq4) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l3143_314388


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_l3143_314377

theorem smallest_x_absolute_value (x : ℝ) : 
  (∀ y : ℝ, |5*y + 15| = 40 → y ≥ x) ↔ x = -11 ∧ |5*x + 15| = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_l3143_314377


namespace NUMINAMATH_CALUDE_fence_panels_count_l3143_314376

/-- Represents the components of a fence panel -/
structure FencePanel where
  sheets : Nat
  beams : Nat

/-- Represents the composition of sheets and beams -/
structure MetalComposition where
  rods_per_sheet : Nat
  rods_per_beam : Nat

/-- Calculates the number of fence panels given the total rods and composition -/
def calculate_fence_panels (total_rods : Nat) (panel : FencePanel) (comp : MetalComposition) : Nat :=
  total_rods / (panel.sheets * comp.rods_per_sheet + panel.beams * comp.rods_per_beam)

theorem fence_panels_count (total_rods : Nat) (panel : FencePanel) (comp : MetalComposition) :
  total_rods = 380 →
  panel.sheets = 3 →
  panel.beams = 2 →
  comp.rods_per_sheet = 10 →
  comp.rods_per_beam = 4 →
  calculate_fence_panels total_rods panel comp = 10 := by
  sorry

#eval calculate_fence_panels 380 ⟨3, 2⟩ ⟨10, 4⟩

end NUMINAMATH_CALUDE_fence_panels_count_l3143_314376


namespace NUMINAMATH_CALUDE_rectangle_division_theorem_l3143_314305

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

theorem rectangle_division_theorem :
  ∃ (original : Rectangle) (largest smallest : Rectangle),
    (∃ (other1 other2 : Rectangle),
      area original = area largest + area smallest + area other1 + area other2) ∧
    perimeter largest = 28 ∧
    perimeter smallest = 12 ∧
    area original = 96 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_division_theorem_l3143_314305


namespace NUMINAMATH_CALUDE_playground_area_l3143_314356

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playgroundArea : ℝ

/-- The landscape satisfies the given conditions -/
def validLandscape (l : Landscape) : Prop :=
  l.length = 8 * l.breadth ∧
  l.length = 240 ∧
  l.playgroundArea = (1 / 6) * (l.length * l.breadth)

/-- Theorem: The playground area is 1200 square meters -/
theorem playground_area (l : Landscape) (h : validLandscape l) : 
  l.playgroundArea = 1200 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_l3143_314356


namespace NUMINAMATH_CALUDE_equation_solution_l3143_314373

theorem equation_solution : ∃ (x : ℝ), 
  x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -6 ∧
  (x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2) ∧
  (3 * x + 6) / (x^2 + 5*x - 6) = (x - 3) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3143_314373


namespace NUMINAMATH_CALUDE_rope_cutting_l3143_314309

theorem rope_cutting (total_length : ℝ) (ratio_short : ℝ) (ratio_long : ℝ) :
  total_length = 35 →
  ratio_short = 3 →
  ratio_long = 4 →
  (ratio_long / (ratio_short + ratio_long)) * total_length = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_l3143_314309


namespace NUMINAMATH_CALUDE_magician_performances_l3143_314366

/-- The number of performances a magician has put on --/
def num_performances : ℕ := 100

/-- The probability that an audience member never reappears --/
def prob_never_reappear : ℚ := 1/10

/-- The probability that two people reappear instead of one --/
def prob_two_reappear : ℚ := 1/5

/-- The total number of people who have reappeared --/
def total_reappeared : ℕ := 110

theorem magician_performances :
  (1 - prob_never_reappear - prob_two_reappear) * num_performances +
  2 * prob_two_reappear * num_performances = total_reappeared :=
by sorry

end NUMINAMATH_CALUDE_magician_performances_l3143_314366


namespace NUMINAMATH_CALUDE_suraj_average_after_17th_innings_l3143_314387

def average_after_17th_innings (initial_average : ℝ) (score_17th : ℝ) (average_increase : ℝ) : Prop :=
  let total_runs_16 := 16 * initial_average
  let total_runs_17 := total_runs_16 + score_17th
  let new_average := total_runs_17 / 17
  new_average = initial_average + average_increase

theorem suraj_average_after_17th_innings :
  ∃ (initial_average : ℝ),
    average_after_17th_innings initial_average 112 6 ∧
    initial_average + 6 = 16 :=
by
  sorry

#check suraj_average_after_17th_innings

end NUMINAMATH_CALUDE_suraj_average_after_17th_innings_l3143_314387


namespace NUMINAMATH_CALUDE_linear_function_derivative_l3143_314315

/-- Given a linear function f(x) = ax + 3 where f'(1) = 3, prove that a = 3 -/
theorem linear_function_derivative (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = a * x + 3) ∧ (deriv f 1 = 3)) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_derivative_l3143_314315


namespace NUMINAMATH_CALUDE_five_digit_subtraction_l3143_314320

theorem five_digit_subtraction (a b c d e : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 →
  a > 0 →
  (a * 10000 + b * 1000 + c * 100 + d * 10 + e) - 
  (e * 10000 + d * 1000 + c * 100 + b * 10 + a) = 
  (10072 : ℕ) →
  a > e →
  (∀ a' e' : ℕ, a' < 10 ∧ e' < 10 ∧ a' > e' → a' - e' ≥ a - e) →
  a = 9 ∧ e = 7 := by
sorry

end NUMINAMATH_CALUDE_five_digit_subtraction_l3143_314320


namespace NUMINAMATH_CALUDE_subtract_to_one_l3143_314331

theorem subtract_to_one : ∃ x : ℤ, (-5) - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_subtract_to_one_l3143_314331


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_18_l3143_314361

/-- A function that returns the sum of n consecutive integers starting from a -/
def consecutive_sum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A predicate that checks if a set of consecutive integers sums to 18 -/
def is_valid_set (a n : ℕ) : Prop :=
  n ≥ 2 ∧ consecutive_sum a n = 18

theorem unique_consecutive_sum_18 :
  ∃! p : ℕ × ℕ, is_valid_set p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_18_l3143_314361


namespace NUMINAMATH_CALUDE_third_hour_speed_l3143_314352

/-- Calculates the average speed for the third hour given total distance, total time, and speeds for the first two hours -/
def average_speed_third_hour (total_distance : ℝ) (total_time : ℝ) (speed_first_hour : ℝ) (speed_second_hour : ℝ) : ℝ :=
  let distance_first_two_hours := speed_first_hour + speed_second_hour
  let distance_third_hour := total_distance - distance_first_two_hours
  distance_third_hour

/-- Proves that the average speed for the third hour is 30 mph given the problem conditions -/
theorem third_hour_speed : 
  let total_distance : ℝ := 120
  let total_time : ℝ := 3
  let speed_first_hour : ℝ := 40
  let speed_second_hour : ℝ := 50
  average_speed_third_hour total_distance total_time speed_first_hour speed_second_hour = 30 := by
  sorry


end NUMINAMATH_CALUDE_third_hour_speed_l3143_314352


namespace NUMINAMATH_CALUDE_xyz_product_l3143_314351

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 100 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l3143_314351


namespace NUMINAMATH_CALUDE_ticket_probabilities_l3143_314325

/-- Represents a group of tickets -/
structure TicketGroup where
  football : ℕ
  volleyball : ℕ

/-- The probability of drawing a football ticket from a group -/
def football_prob (group : TicketGroup) : ℚ :=
  group.football / (group.football + group.volleyball)

/-- The setup of the ticket drawing scenario -/
def ticket_scenario : Prop :=
  ∃ (group1 group2 : TicketGroup),
    group1.football = 6 ∧ group1.volleyball = 4 ∧
    group2.football = 4 ∧ group2.volleyball = 6

theorem ticket_probabilities (h : ticket_scenario) :
  ∃ (group1 group2 : TicketGroup),
    (football_prob group1 * football_prob group2 = 6/25) ∧
    (1 - (1 - football_prob group1) * (1 - football_prob group2) = 19/25) :=
by sorry

end NUMINAMATH_CALUDE_ticket_probabilities_l3143_314325


namespace NUMINAMATH_CALUDE_expression_value_l3143_314324

theorem expression_value : (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3143_314324


namespace NUMINAMATH_CALUDE_mechanism_efficiency_problem_l3143_314349

theorem mechanism_efficiency_problem (t_combined t_partial t_remaining : ℝ) 
  (h_combined : t_combined = 30)
  (h_partial : t_partial = 6)
  (h_remaining : t_remaining = 40) :
  ∃ (t1 t2 : ℝ),
    t1 = 75 ∧ 
    t2 = 50 ∧ 
    (1 / t1 + 1 / t2 = 1 / t_combined) ∧
    (t_partial * (1 / t1 + 1 / t2) + t_remaining / t2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_mechanism_efficiency_problem_l3143_314349


namespace NUMINAMATH_CALUDE_sequence_sum_l3143_314336

theorem sequence_sum (A B C D E F G H I : ℤ) : 
  E = 7 →
  A + B + C + D = 40 →
  B + C + D + E = 40 →
  C + D + E + F = 40 →
  D + E + F + G = 40 →
  E + F + G + H = 40 →
  F + G + H + I = 40 →
  A + I = 40 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3143_314336


namespace NUMINAMATH_CALUDE_combined_tennis_preference_l3143_314312

/-- Calculates the combined percentage of students preferring tennis across three schools -/
theorem combined_tennis_preference (north_students : ℕ) (north_tennis_pct : ℚ)
  (south_students : ℕ) (south_tennis_pct : ℚ)
  (valley_students : ℕ) (valley_tennis_pct : ℚ)
  (h1 : north_students = 1800)
  (h2 : north_tennis_pct = 25 / 100)
  (h3 : south_students = 3000)
  (h4 : south_tennis_pct = 50 / 100)
  (h5 : valley_students = 800)
  (h6 : valley_tennis_pct = 30 / 100) :
  (north_students * north_tennis_pct +
   south_students * south_tennis_pct +
   valley_students * valley_tennis_pct) /
  (north_students + south_students + valley_students) =
  39 / 100 := by
  sorry

end NUMINAMATH_CALUDE_combined_tennis_preference_l3143_314312


namespace NUMINAMATH_CALUDE_ned_friend_games_l3143_314346

/-- The number of games Ned bought from his friend -/
def games_from_friend : ℕ := 50

/-- The number of games Ned bought at the garage sale -/
def garage_sale_games : ℕ := 27

/-- The number of games that didn't work -/
def non_working_games : ℕ := 74

/-- The number of good games Ned ended up with -/
def good_games : ℕ := 3

/-- Theorem stating that the number of games Ned bought from his friend is 50 -/
theorem ned_friend_games : 
  games_from_friend = 50 ∧
  games_from_friend + garage_sale_games = non_working_games + good_games :=
by sorry

end NUMINAMATH_CALUDE_ned_friend_games_l3143_314346


namespace NUMINAMATH_CALUDE_equation_solution_l3143_314363

theorem equation_solution : ∃ m : ℝ, (243 : ℝ) ^ (1/5) = 3^m ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3143_314363


namespace NUMINAMATH_CALUDE_pyramid_theorem_l3143_314385

/-- Regular quadrangular pyramid with a plane through diagonal of base and height -/
structure RegularQuadPyramid where
  /-- Side length of the base -/
  a : ℝ
  /-- Angle between opposite slant heights -/
  α : ℝ
  /-- Ratio of section area to lateral surface area -/
  k : ℝ
  /-- Base side length is positive -/
  a_pos : 0 < a
  /-- Angle is between 0 and π -/
  α_range : 0 < α ∧ α < π
  /-- k is positive -/
  k_pos : 0 < k

/-- Theorem about the cosine of the angle between slant heights and permissible k values -/
theorem pyramid_theorem (p : RegularQuadPyramid) :
  (Real.cos p.α = 64 * p.k^2 - 1) ∧ 
  (p.k ≤ Real.sqrt 2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_theorem_l3143_314385


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3143_314341

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 3 x ≥ x + 9} = {x : ℝ | x < -11/3 ∨ x > 7} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3143_314341


namespace NUMINAMATH_CALUDE_existence_of_multiple_1984_l3143_314374

theorem existence_of_multiple_1984 (a : Fin 97 → ℕ+) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ i j k l : Fin 97, i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧
    1984 ∣ ((a i).val - (a j).val) * ((a k).val - (a l).val) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_multiple_1984_l3143_314374


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1500th_term_l3143_314367

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_1500th_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_term1 : a 1 = m)
  (h_term2 : a 2 = m + 2*n)
  (h_term3 : a 3 = 5*m - n)
  (h_term4 : a 4 = 3*m + 3*n)
  (h_term5 : a 5 = 7*m - n)
  : a 1500 = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1500th_term_l3143_314367


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l3143_314353

/-- The shortest altitude of a triangle with sides 9, 12, and 15 is 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 → 
  a^2 + b^2 = c^2 → 
  h * c = 2 * (a * b / 2) → 
  h ≤ a ∧ h ≤ b → 
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l3143_314353


namespace NUMINAMATH_CALUDE_positive_root_of_equation_l3143_314338

theorem positive_root_of_equation (x : ℝ) : 
  x > 0 ∧ (1/3) * (4*x^2 - 2) = (x^2 - 35*x - 7) * (x^2 + 20*x + 4) → 
  x = (35 + Real.sqrt 1257) / 2 := by
sorry

end NUMINAMATH_CALUDE_positive_root_of_equation_l3143_314338


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l3143_314307

/-- The line y = kx + 1 (k ∈ ℝ) always has a common point with the curve x²/5 + y²/m = 1
    if and only if m ≥ 1 and m ≠ 5, where m is a non-negative real number. -/
theorem line_ellipse_intersection (m : ℝ) (h_m_nonneg : m ≥ 0) :
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1) ↔ m ≥ 1 ∧ m ≠ 5 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l3143_314307


namespace NUMINAMATH_CALUDE_comparison_of_radicals_and_fractions_l3143_314393

theorem comparison_of_radicals_and_fractions : 
  (2 * Real.sqrt 7 < 4 * Real.sqrt 2) ∧ ((Real.sqrt 5 - 1) / 2 > 0.5) := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_radicals_and_fractions_l3143_314393


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l3143_314345

theorem sqrt_sum_fractions : Real.sqrt ((1 : ℝ) / 8 + 1 / 18) = Real.sqrt 26 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l3143_314345


namespace NUMINAMATH_CALUDE_ten_attendants_used_both_l3143_314330

/-- The number of attendants who used both a pencil and a pen at a meeting -/
def attendants_using_both (pencil_users pen_users only_one_tool_users : ℕ) : ℕ :=
  (pencil_users + pen_users - only_one_tool_users) / 2

/-- Theorem stating that 10 attendants used both a pencil and a pen -/
theorem ten_attendants_used_both :
  attendants_using_both 25 15 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_attendants_used_both_l3143_314330


namespace NUMINAMATH_CALUDE_highest_power_of_1991_l3143_314354

theorem highest_power_of_1991 :
  let n : ℕ := 1990^(1991^1992) + 1992^(1991^1990)
  ∃ k : ℕ, k = 2 ∧ (1991 : ℕ)^k ∣ n ∧ ∀ m : ℕ, m > k → ¬((1991 : ℕ)^m ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_1991_l3143_314354


namespace NUMINAMATH_CALUDE_isosceles_max_angle_diff_l3143_314348

/-- An isosceles triangle has two equal angles -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_180 : a + b + c = 180
  isosceles : (a = b) ∨ (b = c) ∨ (a = c)

/-- Given an isosceles triangle with one angle 50°, prove that the maximum difference between the other two angles is 30° -/
theorem isosceles_max_angle_diff (t : IsoscelesTriangle) (h : t.a = 50 ∨ t.b = 50 ∨ t.c = 50) :
  ∃ (x y : ℝ), ((x = t.a ∧ y = t.b) ∨ (x = t.b ∧ y = t.c) ∨ (x = t.a ∧ y = t.c)) ∧
  (∀ (x' y' : ℝ), ((x' = t.a ∧ y' = t.b) ∨ (x' = t.b ∧ y' = t.c) ∨ (x' = t.a ∧ y' = t.c)) →
  |x' - y'| ≤ |x - y|) ∧ |x - y| = 30 :=
sorry

end NUMINAMATH_CALUDE_isosceles_max_angle_diff_l3143_314348


namespace NUMINAMATH_CALUDE_exponent_square_negative_product_l3143_314397

theorem exponent_square_negative_product (a b : ℝ) : (-a^3 * b)^2 = a^6 * b^2 := by sorry

end NUMINAMATH_CALUDE_exponent_square_negative_product_l3143_314397


namespace NUMINAMATH_CALUDE_parallelogram_area_32_22_l3143_314314

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_32_22 :
  parallelogram_area 32 22 = 704 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_32_22_l3143_314314


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_7_l3143_314304

def is_divisible_by_7 (n : ℕ) : Prop := ∃ k, n = 7 * k

def number_with_digit (x : ℕ) : ℕ := 5200 + 10 * x + 4

theorem smallest_digit_divisible_by_7 :
  (∃ x : ℕ, x ≤ 9 ∧ is_divisible_by_7 (number_with_digit x)) ∧
  (∀ y : ℕ, y < 2 → ¬is_divisible_by_7 (number_with_digit y)) ∧
  is_divisible_by_7 (number_with_digit 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_7_l3143_314304


namespace NUMINAMATH_CALUDE_equation_solution_l3143_314379

theorem equation_solution : ∃ x : ℝ, (9 / (x + 3 / 0.75) = 1) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3143_314379


namespace NUMINAMATH_CALUDE_negative_one_third_less_than_negative_point_three_l3143_314344

theorem negative_one_third_less_than_negative_point_three : -1/3 < -0.3 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_third_less_than_negative_point_three_l3143_314344


namespace NUMINAMATH_CALUDE_pool_width_l3143_314321

/-- Proves the width of a rectangular pool given its draining rate, dimensions, initial capacity, and time to drain. -/
theorem pool_width
  (drain_rate : ℝ)
  (length depth : ℝ)
  (initial_capacity : ℝ)
  (drain_time : ℝ)
  (h1 : drain_rate = 60)
  (h2 : length = 150)
  (h3 : depth = 10)
  (h4 : initial_capacity = 0.8)
  (h5 : drain_time = 800) :
  ∃ (width : ℝ), width = 40 ∧ 
    drain_rate * drain_time = initial_capacity * (length * width * depth) :=
by sorry

end NUMINAMATH_CALUDE_pool_width_l3143_314321


namespace NUMINAMATH_CALUDE_nested_radical_value_l3143_314317

/-- The value of the infinite nested radical √(6 + √(6 + √(6 + ...))) -/
noncomputable def nested_radical : ℝ :=
  Real.sqrt (6 + Real.sqrt (6 + Real.sqrt (6 + Real.sqrt 6)))

/-- The nested radical equals 3 -/
theorem nested_radical_value : nested_radical = 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l3143_314317


namespace NUMINAMATH_CALUDE_max_value_mx_plus_ny_l3143_314328

theorem max_value_mx_plus_ny (a b : ℝ) (m n x y : ℝ) 
  (h1 : m^2 + n^2 = a) (h2 : x^2 + y^2 = b) :
  (∃ (k : ℝ), k = m*x + n*y ∧ ∀ (p q : ℝ), p^2 + q^2 = a → ∀ (r s : ℝ), r^2 + s^2 = b → 
    p*r + q*s ≤ k) → k = Real.sqrt (a*b) :=
sorry

end NUMINAMATH_CALUDE_max_value_mx_plus_ny_l3143_314328


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3143_314390

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, (0 < x ∧ x < 2) → (x^2 - x - 2 < 0)) ∧ 
  (∃ x, (x^2 - x - 2 < 0) ∧ ¬(0 < x ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3143_314390


namespace NUMINAMATH_CALUDE_range_of_f_l3143_314311

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y | ∃ x ≥ 0, f x = y} = {y | y ≥ 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3143_314311


namespace NUMINAMATH_CALUDE_sum_a_b_equals_31_l3143_314323

/-- The number of divisors of a positive integer -/
def num_divisors (x : ℕ+) : ℕ := sorry

/-- The product of the smallest ⌈n/2⌉ divisors of x -/
def f (x : ℕ+) : ℕ := sorry

/-- The least value of x such that f(x) is a multiple of x -/
def a : ℕ+ := sorry

/-- The least value of n such that there exists y with n factors and f(y) is a multiple of y -/
def b : ℕ := sorry

theorem sum_a_b_equals_31 : (a : ℕ) + b = 31 := by sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_31_l3143_314323


namespace NUMINAMATH_CALUDE_calculation_proof_l3143_314310

theorem calculation_proof : (0.0077 * 4.5) / (0.05 * 0.1 * 0.007) = 989.2857142857143 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3143_314310


namespace NUMINAMATH_CALUDE_parabola_vertex_l3143_314308

/-- The parabola defined by y = -(x+2)^2 + 6 has its vertex at (-2, 6) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -(x + 2)^2 + 6 → (∀ t : ℝ, y ≤ -(t + 2)^2 + 6) → (x = -2 ∧ y = 6) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3143_314308


namespace NUMINAMATH_CALUDE_sampling_more_suitable_for_large_population_l3143_314337

/-- Represents a survey method -/
inductive SurveyMethod
  | Comprehensive
  | Sampling

/-- Represents the characteristics of a survey -/
structure SurveyCharacteristics where
  populationSize : ℕ
  isSurveyingLargePopulation : Bool

/-- Determines the most suitable survey method based on survey characteristics -/
def mostSuitableSurveyMethod (characteristics : SurveyCharacteristics) : SurveyMethod :=
  if characteristics.isSurveyingLargePopulation then
    SurveyMethod.Sampling
  else
    SurveyMethod.Comprehensive

/-- Theorem: For a large population survey, sampling is more suitable than comprehensive -/
theorem sampling_more_suitable_for_large_population 
  (characteristics : SurveyCharacteristics) 
  (h : characteristics.isSurveyingLargePopulation = true) : 
  mostSuitableSurveyMethod characteristics = SurveyMethod.Sampling :=
by
  sorry

end NUMINAMATH_CALUDE_sampling_more_suitable_for_large_population_l3143_314337


namespace NUMINAMATH_CALUDE_petes_walking_distance_closest_to_2800_l3143_314372

/-- Represents a pedometer with a maximum step count --/
structure Pedometer :=
  (max_count : ℕ)

/-- Represents Pete's walking data for a year --/
structure YearlyWalkingData :=
  (pedometer : Pedometer)
  (flips : ℕ)
  (final_reading : ℕ)
  (steps_per_mile : ℕ)

/-- Calculates the total steps walked in a year --/
def total_steps (data : YearlyWalkingData) : ℕ :=
  data.flips * (data.pedometer.max_count + 1) + data.final_reading

/-- Calculates the total miles walked in a year --/
def total_miles (data : YearlyWalkingData) : ℚ :=
  (total_steps data : ℚ) / data.steps_per_mile

/-- Theorem stating that Pete's walking distance is closest to 2800 miles --/
theorem petes_walking_distance_closest_to_2800 (data : YearlyWalkingData) 
  (h1 : data.pedometer.max_count = 89999)
  (h2 : data.flips = 55)
  (h3 : data.final_reading = 30000)
  (h4 : data.steps_per_mile = 1800) :
  ∃ (n : ℕ), n ≤ 50 ∧ |total_miles data - 2800| < |total_miles data - (2800 - n)| ∧ 
             |total_miles data - 2800| < |total_miles data - (2800 + n)| :=
  sorry

#eval total_miles { pedometer := { max_count := 89999 }, flips := 55, final_reading := 30000, steps_per_mile := 1800 }

end NUMINAMATH_CALUDE_petes_walking_distance_closest_to_2800_l3143_314372


namespace NUMINAMATH_CALUDE_max_sum_is_38_l3143_314398

/-- Represents the setup of numbers in the grid -/
structure Grid :=
  (a b c d e : ℕ)

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {2, 3, 8, 9, 14, 15}

/-- Checks if the grid satisfies the equality condition -/
def isValidGrid (g : Grid) : Prop :=
  g.a + g.b + g.e = g.a + g.c + g.e ∧
  g.a + g.c + g.e = g.b + g.d + g.e

/-- Checks if the grid uses numbers from the available set -/
def usesAvailableNumbers (g : Grid) : Prop :=
  g.a ∈ availableNumbers ∧
  g.b ∈ availableNumbers ∧
  g.c ∈ availableNumbers ∧
  g.d ∈ availableNumbers ∧
  g.e ∈ availableNumbers

/-- Calculates the sum of the grid -/
def gridSum (g : Grid) : ℕ := g.a + g.b + g.e

/-- Theorem: The maximum sum of a valid grid using the available numbers is 38 -/
theorem max_sum_is_38 :
  ∃ (g : Grid), isValidGrid g ∧ usesAvailableNumbers g ∧
  (∀ (h : Grid), isValidGrid h ∧ usesAvailableNumbers h → gridSum h ≤ gridSum g) ∧
  gridSum g = 38 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_is_38_l3143_314398


namespace NUMINAMATH_CALUDE_unique_integer_pair_existence_l3143_314375

theorem unique_integer_pair_existence (a b : ℤ) :
  ∃! (x y : ℤ), (x + 2*y - a)^2 + (2*x - y - b)^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_pair_existence_l3143_314375


namespace NUMINAMATH_CALUDE_boys_playing_marbles_count_l3143_314370

/-- The number of marbles Haley has -/
def total_marbles : ℕ := 26

/-- The number of marbles each boy receives -/
def marbles_per_boy : ℕ := 2

/-- The number of boys who love to play marbles -/
def boys_playing_marbles : ℕ := total_marbles / marbles_per_boy

theorem boys_playing_marbles_count : boys_playing_marbles = 13 := by
  sorry

end NUMINAMATH_CALUDE_boys_playing_marbles_count_l3143_314370


namespace NUMINAMATH_CALUDE_three_officers_from_six_people_l3143_314364

/-- The number of ways to choose three distinct officers from a group of people. -/
def chooseThreeOfficers (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- Theorem stating that choosing three distinct officers from 6 people results in 120 ways. -/
theorem three_officers_from_six_people : chooseThreeOfficers 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_three_officers_from_six_people_l3143_314364


namespace NUMINAMATH_CALUDE_discount_problem_l3143_314384

theorem discount_problem (list_price : ℝ) (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  list_price = 70 →
  final_price = 59.85 →
  discount1 = 10 →
  (list_price * (1 - discount1 / 100) * (1 - discount2 / 100) = final_price) →
  discount2 = 5 := by
sorry

end NUMINAMATH_CALUDE_discount_problem_l3143_314384


namespace NUMINAMATH_CALUDE_inscribed_cube_edge_length_l3143_314316

theorem inscribed_cube_edge_length (S : Real) (r : Real) (x : Real) :
  S = 4 * Real.pi →  -- Surface area of the sphere
  S = 4 * Real.pi * r^2 →  -- Formula for surface area of a sphere
  x * Real.sqrt 3 = 2 * r →  -- Relationship between cube diagonal and sphere diameter
  x = 2 * Real.sqrt 3 / 3 :=  -- Edge length of the inscribed cube
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_edge_length_l3143_314316


namespace NUMINAMATH_CALUDE_smallest_x_for_simplified_fractions_l3143_314342

theorem smallest_x_for_simplified_fractions : ∃ (x : ℕ), x > 0 ∧
  (∀ (k : ℕ), k ≥ 1 ∧ k ≤ 40 → Nat.gcd (3*x + k) (k + 7) = 1) ∧
  (∀ (y : ℕ), y > 0 ∧ y < x → ∃ (k : ℕ), k ≥ 1 ∧ k ≤ 40 ∧ Nat.gcd (3*y + k) (k + 7) ≠ 1) ∧
  x = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_simplified_fractions_l3143_314342


namespace NUMINAMATH_CALUDE_train_passing_time_l3143_314394

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 150 →
  train_speed = 62 * (1000 / 3600) →
  man_speed = 8 * (1000 / 3600) →
  (train_length / (train_speed - man_speed)) = 10 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l3143_314394


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_not_necessary_condition_l3143_314302

theorem sufficient_condition_for_inequality (x : ℝ) :
  (-1 < x ∧ x < 5) → (6 / (x + 1) ≥ 1) :=
by
  sorry

theorem not_necessary_condition (x : ℝ) :
  (6 / (x + 1) ≥ 1) → ¬(-1 < x ∧ x < 5) :=
by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_not_necessary_condition_l3143_314302


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3143_314300

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), 100 * x = 56 + x ∧ n * x = x) → x = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3143_314300


namespace NUMINAMATH_CALUDE_second_player_wins_l3143_314392

-- Define the graph structure
structure GameGraph where
  vertices : Finset Char
  edges : Finset (Char × Char)
  start : Char
  degree : Char → Nat

-- Define the game rules
structure GameRules where
  graph : GameGraph
  current_player : Nat
  used_edges : Finset (Char × Char)
  current_position : Char

-- Define a move
def valid_move (rules : GameRules) (next : Char) : Prop :=
  (rules.current_position, next) ∈ rules.graph.edges ∧
  (rules.current_position, next) ∉ rules.used_edges

-- Define the winning condition
def is_winning_position (rules : GameRules) : Prop :=
  ∀ next, ¬(valid_move rules next)

-- Theorem statement
theorem second_player_wins (g : GameGraph)
  (h1 : g.vertices = {'A', 'B', 'C', 'D', 'E', 'F'})
  (h2 : g.start = 'A')
  (h3 : g.degree 'A' = 4)
  (h4 : g.degree 'B' = 5)
  (h5 : g.degree 'C' = 5)
  (h6 : g.degree 'D' = 3)
  (h7 : g.degree 'E' = 3)
  (h8 : g.degree 'F' = 5)
  : ∃ (strategy : GameRules → Char),
    ∀ (rules : GameRules),
      rules.graph = g →
      rules.current_player = 2 →
      (valid_move rules (strategy rules) ∧
       is_winning_position
         { graph := rules.graph,
           current_player := 1,
           used_edges := insert (rules.current_position, strategy rules) rules.used_edges,
           current_position := strategy rules }) :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l3143_314392


namespace NUMINAMATH_CALUDE_first_number_in_ratio_l3143_314319

/-- Given two positive integers a and b with a ratio of 3:4 and LCM 180, prove that a = 45 -/
theorem first_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → 
  Nat.lcm a b = 180 → 
  a = 45 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_ratio_l3143_314319


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3143_314318

theorem diophantine_equation_solutions :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S → (p.1^2 + p.2^2 = 26 * p.1)) ∧ S.card ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3143_314318


namespace NUMINAMATH_CALUDE_logarithm_sum_equality_logarithm_product_equality_l3143_314327

-- Part 1
theorem logarithm_sum_equality : 2 * Real.log 10 / Real.log 2 + Real.log 0.04 / Real.log 2 = 2 := by sorry

-- Part 2
theorem logarithm_product_equality : 
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 5 / Real.log 3 + Real.log 5 / Real.log 9) * 
  (Real.log 2 / Real.log 5 + Real.log 2 / Real.log 25) = 15/8 := by sorry

end NUMINAMATH_CALUDE_logarithm_sum_equality_logarithm_product_equality_l3143_314327


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3143_314380

theorem combined_mean_of_two_sets (set1_count set2_count : ℕ) 
  (set1_mean set2_mean : ℚ) : 
  set1_count = 7 → 
  set2_count = 8 → 
  set1_mean = 15 → 
  set2_mean = 30 → 
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 23 := by
sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3143_314380


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3143_314343

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (containedIn : Line → Plane → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (parallelLineToPlane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (l : Line) (α β : Plane)
  (h1 : parallelPlanes α β)
  (h2 : containedIn l α) :
  parallelLineToPlane l β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3143_314343


namespace NUMINAMATH_CALUDE_unique_lcm_triple_l3143_314395

theorem unique_lcm_triple : ∃! (x y z : ℕ+), 
  (Nat.lcm x.val y.val = 108) ∧ 
  (Nat.lcm x.val z.val = 400) ∧ 
  (Nat.lcm y.val z.val = 450) := by
  sorry

end NUMINAMATH_CALUDE_unique_lcm_triple_l3143_314395


namespace NUMINAMATH_CALUDE_slipper_cost_theorem_l3143_314326

/-- Calculate the total cost of slippers with embroidery and shipping --/
def calculate_slipper_cost (original_price : ℝ) (discount_rate : ℝ) 
  (embroidery_cost_multiple : ℝ) (num_initials : ℕ) (base_shipping : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let embroidery_cost := 2 * (embroidery_cost_multiple * num_initials)
  let total_cost := discounted_price + embroidery_cost + base_shipping
  total_cost

/-- Theorem stating the total cost of the slippers --/
theorem slipper_cost_theorem :
  calculate_slipper_cost 50 0.1 4.5 3 10 = 82 :=
by sorry

end NUMINAMATH_CALUDE_slipper_cost_theorem_l3143_314326


namespace NUMINAMATH_CALUDE_tenth_root_unity_l3143_314382

theorem tenth_root_unity : 
  ∃ (n : ℕ) (h : n < 10), 
    (Complex.tan (Real.pi / 5) + Complex.I) / (Complex.tan (Real.pi / 5) - Complex.I) = 
    Complex.exp (Complex.I * (2 * ↑n * Real.pi / 10)) :=
by sorry

end NUMINAMATH_CALUDE_tenth_root_unity_l3143_314382


namespace NUMINAMATH_CALUDE_f_decreasing_l3143_314378

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + b

-- State the theorem
theorem f_decreasing (b : ℝ) : 
  ∀ x ∈ Set.Icc (-2) 2, 
    ∀ y ∈ Set.Icc (-2) 2, 
      x < y → f b x > f b y :=
by
  sorry


end NUMINAMATH_CALUDE_f_decreasing_l3143_314378


namespace NUMINAMATH_CALUDE_jumping_contest_l3143_314371

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) 
  (h1 : grasshopper_jump = 25)
  (h2 : mouse_jump = 31)
  (h3 : mouse_jump + 26 = frog_jump)
  (h4 : frog_jump > grasshopper_jump) :
  frog_jump - grasshopper_jump = 32 := by
sorry


end NUMINAMATH_CALUDE_jumping_contest_l3143_314371


namespace NUMINAMATH_CALUDE_max_servings_is_56_l3143_314339

/-- Ingredients required for one serving of salad -/
structure SaladServing where
  cucumbers : ℕ := 2
  tomatoes : ℕ := 2
  brynza : ℕ := 75  -- in grams
  peppers : ℕ := 1

/-- Ingredients available in the warehouse -/
structure Warehouse where
  cucumbers : ℕ := 117
  tomatoes : ℕ := 116
  brynza : ℕ := 4200  -- converted from 4.2 kg to grams
  peppers : ℕ := 60

/-- Calculate the maximum number of servings that can be made -/
def maxServings (w : Warehouse) (s : SaladServing) : ℕ :=
  min (w.cucumbers / s.cucumbers)
      (min (w.tomatoes / s.tomatoes)
           (min (w.brynza / s.brynza)
                (w.peppers / s.peppers)))

/-- Theorem: The maximum number of servings that can be made is 56 -/
theorem max_servings_is_56 (w : Warehouse) (s : SaladServing) :
  maxServings w s = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_is_56_l3143_314339
