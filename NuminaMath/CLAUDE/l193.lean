import Mathlib

namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_l193_19329

theorem smallest_k_with_remainder (k : ℕ) : k = 135 ↔ 
  (k > 1) ∧ 
  (∃ a : ℕ, k = 11 * a + 3) ∧ 
  (∃ b : ℕ, k = 4 * b + 3) ∧ 
  (∃ c : ℕ, k = 3 * c + 3) ∧ 
  (∀ m : ℕ, m > 1 → 
    ((∃ x : ℕ, m = 11 * x + 3) ∧ 
     (∃ y : ℕ, m = 4 * y + 3) ∧ 
     (∃ z : ℕ, m = 3 * z + 3)) → 
    m ≥ k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_l193_19329


namespace NUMINAMATH_CALUDE_smallest_non_prime_sums_l193_19303

theorem smallest_non_prime_sums : ∃ (n : ℕ), n = 7 ∧
  (∀ m : ℕ, m < n →
    (Prime (m + 1 + m + 2 + m + 3) ∨
     Prime (m + m + 2 + m + 3) ∨
     Prime (m + m + 1 + m + 3) ∨
     Prime (m + m + 1 + m + 2))) ∧
  (¬ Prime (n + 1 + n + 2 + n + 3) ∧
   ¬ Prime (n + n + 2 + n + 3) ∧
   ¬ Prime (n + n + 1 + n + 3) ∧
   ¬ Prime (n + n + 1 + n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_sums_l193_19303


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_l193_19305

/-- Calculates the sum of digits of a number in base 10 -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Calculates the sum of binary digits of a number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is less than 500 -/
def lessThan500 (n : ℕ) : Prop := n < 500

/-- Checks if the cost is the same for both options -/
def equalCost (n : ℕ) : Prop := sumDigits n = sumBinaryDigits n

theorem largest_equal_cost_number :
  ∀ n : ℕ, lessThan500 n → equalCost n → n ≤ 247 :=
sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_l193_19305


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l193_19365

/-- Given an ellipse and a hyperbola with coinciding foci, prove that b^2 = 75/4 for the ellipse -/
theorem ellipse_hyperbola_foci (b : ℝ) : 
  (∀ x y : ℝ, x^2/25 + y^2/b^2 = 1 → x^2/64 - y^2/36 = 1/16) →
  (∃ c : ℝ, c^2 = 25 - b^2 ∧ c^2 = 64 - 36) →
  b^2 = 75/4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l193_19365


namespace NUMINAMATH_CALUDE_work_completion_time_l193_19343

/-- Given Johnson's and Vincent's individual work rates, calculates the time required for them to complete the work together -/
theorem work_completion_time (johnson_rate vincent_rate : ℚ) 
  (h1 : johnson_rate = 1 / 10)
  (h2 : vincent_rate = 1 / 40) :
  1 / (johnson_rate + vincent_rate) = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l193_19343


namespace NUMINAMATH_CALUDE_winning_condition_l193_19326

/-- Represents a chessboard of size n × n -/
structure Chessboard (n : ℕ) where
  size : n > 0

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- Represents the result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- The game played on a chessboard -/
def game (n : ℕ) (board : Chessboard n) : GameResult := sorry

/-- Theorem stating the winning condition based on the parity of n -/
theorem winning_condition (n : ℕ) (board : Chessboard n) :
  game n board = GameResult.FirstPlayerWins ↔ Even n := by sorry

end NUMINAMATH_CALUDE_winning_condition_l193_19326


namespace NUMINAMATH_CALUDE_max_green_socks_l193_19332

/-- Represents the number of socks in a drawer -/
structure SockDrawer where
  green : ℕ
  yellow : ℕ
  total_bound : green + yellow ≤ 2500

/-- The probability of choosing two socks of the same color -/
def same_color_probability (d : SockDrawer) : ℚ :=
  let t := d.green + d.yellow
  (d.green * (d.green - 1) + d.yellow * (d.yellow - 1)) / (t * (t - 1))

/-- The theorem stating the maximum number of green socks possible -/
theorem max_green_socks (d : SockDrawer) 
  (h : same_color_probability d = 2/3) : 
  d.green ≤ 1275 ∧ ∃ d' : SockDrawer, d'.green = 1275 ∧ same_color_probability d' = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_max_green_socks_l193_19332


namespace NUMINAMATH_CALUDE_negative_two_less_than_negative_three_halves_l193_19395

theorem negative_two_less_than_negative_three_halves : -2 < -3/2 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_less_than_negative_three_halves_l193_19395


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l193_19379

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 8 = 15 - a 5 → a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l193_19379


namespace NUMINAMATH_CALUDE_sum_of_greatest_b_values_l193_19368

theorem sum_of_greatest_b_values (c : ℝ) (h : c ≠ 0) :
  ∃ (b₁ b₂ : ℝ), b₁ > b₂ ∧ b₂ > 0 ∧
  (4 * b₁^4 - 41 * b₁^2 + 100) * c = 0 ∧
  (4 * b₂^4 - 41 * b₂^2 + 100) * c = 0 ∧
  ∀ (b : ℝ), (4 * b^4 - 41 * b^2 + 100) * c = 0 → b ≤ b₁ ∧
  b₁ + b₂ = 4.5 :=
sorry

end NUMINAMATH_CALUDE_sum_of_greatest_b_values_l193_19368


namespace NUMINAMATH_CALUDE_batsman_average_increase_l193_19322

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  total_runs : Nat
  average : Rat

/-- Calculates the increase in average after a new inning -/
def average_increase (b : Batsman) (new_runs : Nat) : Rat :=
  let new_total := b.total_runs + new_runs
  let new_average : Rat := new_total / (b.innings + 1)
  new_average - b.average

/-- Theorem: The batsman's average increases by 3 -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 16 →
    (b.total_runs + 86) / 17 = 38 →
    average_increase b 86 = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l193_19322


namespace NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l193_19362

theorem smallest_area_of_2020th_square (n : ℕ) : 
  n > 0 →
  n^2 = 2019 + (n^2 - 2019) →
  (∀ i : Fin 2019, 1 = 1) →
  n^2 - 2019 ≠ 1 →
  n^2 - 2019 ≥ 6 ∧ 
  ∀ m : ℕ, m > 0 → m^2 = 2019 + (m^2 - 2019) → (∀ i : Fin 2019, 1 = 1) → m^2 - 2019 ≠ 1 → m^2 - 2019 ≥ n^2 - 2019 :=
by sorry

#check smallest_area_of_2020th_square

end NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l193_19362


namespace NUMINAMATH_CALUDE_magic_deck_problem_l193_19373

/-- Given a magician selling magic card decks, this theorem proves
    the number of decks left unsold at the end of the day. -/
theorem magic_deck_problem (initial_decks : ℕ) (price_per_deck : ℕ) (total_earnings : ℕ) :
  initial_decks = 16 →
  price_per_deck = 7 →
  total_earnings = 56 →
  initial_decks - (total_earnings / price_per_deck) = 8 := by
  sorry

end NUMINAMATH_CALUDE_magic_deck_problem_l193_19373


namespace NUMINAMATH_CALUDE_quadrilateral_perpendicular_diagonals_l193_19382

/-- Given a quadrilateral ABCD in the complex plane, construct points O₁, O₂, O₃, O₄
    and prove that O₁O₃ is perpendicular and equal to O₂O₄ -/
theorem quadrilateral_perpendicular_diagonals
  (a b c d : ℂ) : 
  let g₁ : ℂ := (a + d) / 2
  let g₂ : ℂ := (b + a) / 2
  let g₃ : ℂ := (c + b) / 2
  let g₄ : ℂ := (d + c) / 2
  let o₁ : ℂ := g₁ + (d - a) / 2 * Complex.I
  let o₂ : ℂ := g₂ + (a - b) / 2 * Complex.I
  let o₃ : ℂ := g₃ + (c - b) / 2 * Complex.I
  let o₄ : ℂ := g₄ + (d - c) / 2 * Complex.I
  (o₃ - o₁) = (o₄ - o₂) * Complex.I ∧ Complex.abs (o₃ - o₁) = Complex.abs (o₄ - o₂) :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perpendicular_diagonals_l193_19382


namespace NUMINAMATH_CALUDE_fraction_sum_l193_19386

theorem fraction_sum (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l193_19386


namespace NUMINAMATH_CALUDE_researcher_can_reach_oasis_l193_19375

/-- Represents a traveler in the desert -/
structure Traveler where
  food : ℕ
  position : ℕ

/-- Represents the state of the journey -/
structure JourneyState where
  researcher : Traveler
  porters : List Traveler
  day : ℕ

def oasisDistance : ℕ := 380
def dailyTravel : ℕ := 60
def maxFood : ℕ := 4

def canReachOasis (initialState : JourneyState) : Prop :=
  ∃ (finalState : JourneyState),
    finalState.researcher.position = oasisDistance ∧
    finalState.day ≤ initialState.researcher.food * maxFood ∧
    ∀ porter ∈ finalState.porters, porter.position = 0

theorem researcher_can_reach_oasis :
  ∃ (initialState : JourneyState),
    initialState.researcher.food = maxFood ∧
    initialState.researcher.position = 0 ∧
    initialState.porters.length = 2 ∧
    (∀ porter ∈ initialState.porters, porter.food = maxFood ∧ porter.position = 0) ∧
    initialState.day = 0 ∧
    canReachOasis initialState :=
  sorry

end NUMINAMATH_CALUDE_researcher_can_reach_oasis_l193_19375


namespace NUMINAMATH_CALUDE_f_nonnegative_range_l193_19347

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (Real.exp x - a) - a^2 * x

theorem f_nonnegative_range (a : ℝ) :
  (∀ x, f a x ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.exp (3/4)) 1 :=
by sorry

end NUMINAMATH_CALUDE_f_nonnegative_range_l193_19347


namespace NUMINAMATH_CALUDE_inscribed_square_area_l193_19380

/-- The parabola function y = x^2 - 10x + 21 --/
def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A square inscribed in the region bound by the parabola and the x-axis --/
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  side : ℝ    -- length of the square's side
  h1 : center - side/2 ≥ 0  -- Left side of square is non-negative
  h2 : center + side/2 ≤ 10 -- Right side of square is at most the x-intercept
  h3 : parabola (center - side/2) = 0  -- Left bottom corner on x-axis
  h4 : parabola (center + side/2) = 0  -- Right bottom corner on x-axis
  h5 : parabola center = side          -- Top of square touches parabola

/-- The theorem stating the area of the inscribed square --/
theorem inscribed_square_area (s : InscribedSquare) :
  s.side^2 = 24 - 8*Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l193_19380


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_cone_l193_19354

/-- The surface area of a sphere containing a cone with base radius 1 and height √3 -/
theorem sphere_surface_area_with_cone (R : ℝ) : 
  (R : ℝ) > 0 → -- Radius is positive
  R^2 = (R - Real.sqrt 3)^2 + 1 → -- Cone geometry condition
  4 * π * R^2 = 16 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_cone_l193_19354


namespace NUMINAMATH_CALUDE_stating_count_numbers_with_five_or_six_in_base_eight_l193_19381

/-- 
Given a positive integer n and a base b, returns the number of integers 
from 1 to n (inclusive) in base b that contain at least one digit d or e.
-/
def count_numbers_with_digits (n : ℕ) (b : ℕ) (d e : ℕ) : ℕ :=
  sorry

/-- 
Theorem stating that the number of integers from 1 to 512 (inclusive) 
in base 8 that contain at least one digit 5 or 6 is equal to 296.
-/
theorem count_numbers_with_five_or_six_in_base_eight : 
  count_numbers_with_digits 512 8 5 6 = 296 := by
  sorry

end NUMINAMATH_CALUDE_stating_count_numbers_with_five_or_six_in_base_eight_l193_19381


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l193_19337

def polynomial (x m : ℝ) : ℝ := 3 * x^2 - 9 * x + m

theorem polynomial_divisibility (m : ℝ) : 
  (∃ q : ℝ → ℝ, ∀ x, polynomial x m = (x - 2) * q x) ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l193_19337


namespace NUMINAMATH_CALUDE_no_integer_solution_l193_19333

theorem no_integer_solution : ∀ x y : ℤ, x^2 - 37*y ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l193_19333


namespace NUMINAMATH_CALUDE_remaining_pie_portion_l193_19315

theorem remaining_pie_portion (carlos_share : Real) (maria_fraction : Real) : 
  carlos_share = 0.8 →
  maria_fraction = 0.25 →
  (1 - carlos_share) * (1 - maria_fraction) = 0.15 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_pie_portion_l193_19315


namespace NUMINAMATH_CALUDE_taxi_fare_for_8_2km_l193_19360

/-- Calculates the taxi fare for a given distance -/
def taxiFare (distance : Float) : Float :=
  let baseFare := 6
  let midRateDistance := 4
  let midRate := 1
  let highRate := 0.8
  let baseDistance := 3
  let midDistanceEnd := 7
  if distance ≤ baseDistance then
    baseFare
  else if distance ≤ midDistanceEnd then
    baseFare + midRate * (Float.ceil (distance - baseDistance))
  else
    baseFare + midRate * midRateDistance + highRate * (Float.ceil (distance - midDistanceEnd))

theorem taxi_fare_for_8_2km :
  taxiFare 8.2 = 11.6 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_for_8_2km_l193_19360


namespace NUMINAMATH_CALUDE_a3_value_geometric_sequence_max_sum_value_l193_19398

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the conditions for the sequence
def SequenceConditions (a : Sequence) : Prop :=
  (∀ n ≥ 2, a n ≥ 0) ∧
  (∀ n ≥ 2, (2 * a n = a (n+1) + a (n-1)) ∨ (2 * a (n+1) = a n + a (n-1)))

-- Theorem 1
theorem a3_value (a : Sequence) (h : SequenceConditions a) :
  a 1 = 5 ∧ a 2 = 3 ∧ a 4 = 2 → a 3 = 1 :=
sorry

-- Theorem 2
theorem geometric_sequence (a : Sequence) (h : SequenceConditions a) :
  a 1 = 0 ∧ a 4 = 0 ∧ a 7 = 0 ∧ a 2 > 0 ∧ a 5 > 0 ∧ a 8 > 0 →
  ∃ q : ℝ, q = 1/4 ∧ a 5 = a 2 * q ∧ a 8 = a 5 * q :=
sorry

-- Theorem 3
theorem max_sum_value (a : Sequence) (h : SequenceConditions a) :
  a 1 = 1 ∧ a 2 = 2 ∧
  (∃ r s t : ℕ, 2 < r ∧ r < s ∧ s < t ∧ a r = 0 ∧ a s = 0 ∧ a t = 0 ∧
    (∀ n : ℕ, n ≠ r ∧ n ≠ s ∧ n ≠ t → a n ≠ 0)) →
  (∀ r s t : ℕ, 2 < r ∧ r < s ∧ s < t ∧ a r = 0 ∧ a s = 0 ∧ a t = 0 →
    a (r+1) + a (s+1) + a (t+1) ≤ 21/64) :=
sorry

end NUMINAMATH_CALUDE_a3_value_geometric_sequence_max_sum_value_l193_19398


namespace NUMINAMATH_CALUDE_tanks_needed_l193_19324

def existing_tanks : ℕ := 3
def existing_capacity : ℕ := 15
def new_capacity : ℕ := 10
def total_fish : ℕ := 75

theorem tanks_needed : 
  (total_fish - existing_tanks * existing_capacity) / new_capacity = 3 := by
  sorry

end NUMINAMATH_CALUDE_tanks_needed_l193_19324


namespace NUMINAMATH_CALUDE_savings_percentage_l193_19372

/-- Represents a person's financial situation over two years --/
structure FinancialSituation where
  income_year1 : ℝ
  savings_year1 : ℝ
  income_year2 : ℝ
  savings_year2 : ℝ

/-- Calculates the expenditure for a given year --/
def expenditure (income : ℝ) (savings : ℝ) : ℝ :=
  income - savings

/-- Theorem stating the conditions and the result to be proved --/
theorem savings_percentage (f : FinancialSituation) 
  (h1 : f.income_year2 = 1.2 * f.income_year1)
  (h2 : f.savings_year2 = 2 * f.savings_year1)
  (h3 : expenditure f.income_year1 f.savings_year1 + 
        expenditure f.income_year2 f.savings_year2 = 
        2 * expenditure f.income_year1 f.savings_year1) :
  f.savings_year1 / f.income_year1 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l193_19372


namespace NUMINAMATH_CALUDE_complement_of_55_degrees_l193_19311

def angle_A : ℝ := 55

def complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_55_degrees :
  complement angle_A = 35 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_55_degrees_l193_19311


namespace NUMINAMATH_CALUDE_min_z_value_l193_19361

theorem min_z_value (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → Odd y → Odd z → 
  z - x ≥ 9 → 
  (∀ w, w < z → ¬(x < w ∧ w < z ∧ w - x > 5 ∧ Odd w)) →
  z = 9 := by
sorry

end NUMINAMATH_CALUDE_min_z_value_l193_19361


namespace NUMINAMATH_CALUDE_frog_jumps_l193_19369

/-- A jump sequence represents the frog's movements, where
    true represents a jump to the right and false represents a jump to the left. -/
def JumpSequence := List Bool

/-- The position after following a jump sequence -/
def position (p q : ℕ) (jumps : JumpSequence) : ℤ :=
  jumps.foldl (λ acc jump => if jump then acc + p else acc - q) 0

/-- A jump sequence is valid if it starts and ends at 0 -/
def is_valid_sequence (p q : ℕ) (jumps : JumpSequence) : Prop :=
  position p q jumps = 0

theorem frog_jumps (p q : ℕ) (jumps : JumpSequence) (d : ℕ) :
  Nat.Coprime p q →
  is_valid_sequence p q jumps →
  d < p + q →
  ∃ (i j : ℕ), i < jumps.length ∧ j < jumps.length ∧
    abs (position p q (jumps.take i) - position p q (jumps.take j)) = d :=
sorry

end NUMINAMATH_CALUDE_frog_jumps_l193_19369


namespace NUMINAMATH_CALUDE_parallelogram_area_l193_19321

/-- The area of a parallelogram with base 36 cm and height 24 cm is 864 square centimeters. -/
theorem parallelogram_area : 
  let base : ℝ := 36
  let height : ℝ := 24
  let area := base * height
  area = 864 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l193_19321


namespace NUMINAMATH_CALUDE_election_win_percentage_l193_19358

/-- The required percentage to win an election --/
def required_percentage_to_win (total_votes : ℕ) (candidate_votes : ℕ) (additional_votes_needed : ℕ) : ℚ :=
  (candidate_votes + additional_votes_needed : ℚ) / total_votes * 100

/-- Theorem stating the required percentage to win the election --/
theorem election_win_percentage 
  (total_votes : ℕ) 
  (candidate_votes : ℕ) 
  (additional_votes_needed : ℕ) 
  (h1 : total_votes = 6000)
  (h2 : candidate_votes = total_votes / 100)
  (h3 : additional_votes_needed = 3000) :
  required_percentage_to_win total_votes candidate_votes additional_votes_needed = 51 := by
sorry

#eval required_percentage_to_win 6000 60 3000

end NUMINAMATH_CALUDE_election_win_percentage_l193_19358


namespace NUMINAMATH_CALUDE_droid_weekly_usage_l193_19327

/-- Represents the daily coffee bean usage in Droid's coffee shop -/
structure DailyUsage where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- Calculates the total daily usage -/
def totalDailyUsage (usage : DailyUsage) : ℕ :=
  usage.morning + usage.afternoon + usage.evening

/-- Represents the weekly coffee bean usage in Droid's coffee shop -/
structure WeeklyUsage where
  weekday : DailyUsage
  saturday : DailyUsage
  sunday : DailyUsage

/-- Calculates the total weekly usage -/
def totalWeeklyUsage (usage : WeeklyUsage) : ℕ :=
  5 * totalDailyUsage usage.weekday + totalDailyUsage usage.saturday + totalDailyUsage usage.sunday

/-- The coffee bean usage pattern for Droid's coffee shop -/
def droidUsage : WeeklyUsage where
  weekday := { morning := 3, afternoon := 9, evening := 6 }
  saturday := { morning := 4, afternoon := 8, evening := 6 }
  sunday := { morning := 2, afternoon := 2, evening := 2 }

theorem droid_weekly_usage : totalWeeklyUsage droidUsage = 114 := by
  sorry

end NUMINAMATH_CALUDE_droid_weekly_usage_l193_19327


namespace NUMINAMATH_CALUDE_wall_length_proof_l193_19346

/-- Proves that the length of a wall is 900 cm given specific brick and wall dimensions --/
theorem wall_length_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
                          (wall_height : ℝ) (wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_height = 600 →
  wall_width = 22.5 →
  num_bricks = 7200 →
  (brick_length * brick_width * brick_height * num_bricks) / (wall_height * wall_width) = 900 :=
by
  sorry

#check wall_length_proof

end NUMINAMATH_CALUDE_wall_length_proof_l193_19346


namespace NUMINAMATH_CALUDE_rotated_rectangle_height_l193_19340

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The configuration of three rectangles with the middle one rotated -/
structure RectangleConfiguration where
  left : Rectangle
  middle : Rectangle
  right : Rectangle
  rotated : Bool

/-- Calculate the height of the top vertex of the middle rectangle when rotated -/
def heightOfRotatedMiddle (config : RectangleConfiguration) : ℝ :=
  if config.rotated then config.middle.width else config.middle.height

/-- The main theorem stating that the height of the rotated middle rectangle is 2 inches -/
theorem rotated_rectangle_height
  (config : RectangleConfiguration)
  (h1 : config.left.width = 2 ∧ config.left.height = 1)
  (h2 : config.middle.width = 2 ∧ config.middle.height = 1)
  (h3 : config.right.width = 2 ∧ config.right.height = 1)
  (h4 : config.rotated = true) :
  heightOfRotatedMiddle config = 2 := by
  sorry

end NUMINAMATH_CALUDE_rotated_rectangle_height_l193_19340


namespace NUMINAMATH_CALUDE_monthly_interest_payment_l193_19367

/-- Calculate the monthly interest payment given the annual interest rate and investment amount -/
theorem monthly_interest_payment 
  (annual_rate : ℝ) 
  (investment : ℝ) 
  (h1 : annual_rate = 0.09) 
  (h2 : investment = 28800) : 
  (investment * annual_rate) / 12 = 216 := by
  sorry

end NUMINAMATH_CALUDE_monthly_interest_payment_l193_19367


namespace NUMINAMATH_CALUDE_cosine_function_triangle_constraint_l193_19364

open Real

theorem cosine_function_triangle_constraint (ω : ℝ) : 
  ω > 0 →
  let f : ℝ → ℝ := λ x => cos (ω * x)
  let A : ℝ × ℝ := (2 * π / ω, 1)
  let B : ℝ × ℝ := (π / ω, -1)
  let O : ℝ × ℝ := (0, 0)
  (∀ x > 0, x < 2 * π / ω → f x ≤ 1) →
  (∀ x > 0, x < π / ω → f x ≥ -1) →
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) > 0 →
  (O.1 - A.1) * (B.1 - A.1) + (O.2 - A.2) * (B.2 - A.2) > 0 →
  (O.1 - B.1) * (A.1 - B.1) + (O.2 - B.2) * (A.2 - B.2) > 0 →
  sqrt 2 * π / 2 < ω ∧ ω < sqrt 2 * π :=
by sorry

end NUMINAMATH_CALUDE_cosine_function_triangle_constraint_l193_19364


namespace NUMINAMATH_CALUDE_gift_price_proof_l193_19388

def gift_price_calculation (lisa_savings : ℝ) (mother_fraction : ℝ) (brother_multiplier : ℝ) (price_difference : ℝ) : Prop :=
  let mother_contribution := mother_fraction * lisa_savings
  let brother_contribution := brother_multiplier * mother_contribution
  let total_amount := lisa_savings + mother_contribution + brother_contribution
  let gift_price := total_amount + price_difference
  gift_price = 3760

theorem gift_price_proof :
  gift_price_calculation 1200 (3/5) 2 400 := by
  sorry

end NUMINAMATH_CALUDE_gift_price_proof_l193_19388


namespace NUMINAMATH_CALUDE_betty_afternoon_flies_l193_19310

/-- The number of flies Betty caught in the afternoon before one escaped -/
def afternoon_flies : ℕ := 6

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of flies the frog eats per day -/
def flies_per_day : ℕ := 2

/-- The number of flies Betty caught in the morning -/
def morning_flies : ℕ := 5

/-- The number of additional flies Betty needs for the whole week -/
def additional_flies_needed : ℕ := 4

/-- The number of flies that escaped when Betty removed the lid -/
def escaped_flies : ℕ := 1

theorem betty_afternoon_flies :
  afternoon_flies = 
    days_in_week * flies_per_day - morning_flies - additional_flies_needed + escaped_flies :=
by sorry

end NUMINAMATH_CALUDE_betty_afternoon_flies_l193_19310


namespace NUMINAMATH_CALUDE_liliane_alice_relationship_l193_19334

/-- Represents the amount of soda each person has -/
structure SodaAmounts where
  jacqueline : ℝ
  liliane : ℝ
  alice : ℝ
  bruno : ℝ

/-- The conditions of the soda problem -/
def SodaProblem (amounts : SodaAmounts) : Prop :=
  amounts.liliane = amounts.jacqueline * 1.6 ∧
  amounts.alice = amounts.jacqueline * 1.4 ∧
  amounts.bruno = amounts.jacqueline * 0.8

/-- The theorem stating the relationship between Liliane's and Alice's soda amounts -/
theorem liliane_alice_relationship (amounts : SodaAmounts) 
  (h : SodaProblem amounts) : 
  ∃ ε > 0, ε < 0.005 ∧ amounts.liliane = amounts.alice * (1 + 0.15 + ε) :=
sorry

end NUMINAMATH_CALUDE_liliane_alice_relationship_l193_19334


namespace NUMINAMATH_CALUDE_max_portions_is_two_l193_19339

/-- Represents the number of bags for each ingredient -/
structure Ingredients :=
  (nuts : ℕ)
  (dried_fruit : ℕ)
  (chocolate : ℕ)
  (coconut : ℕ)

/-- Represents the ratio of ingredients in each portion -/
structure Ratio :=
  (nuts : ℕ)
  (dried_fruit : ℕ)
  (chocolate : ℕ)
  (coconut : ℕ)

/-- Calculates the maximum number of portions that can be made -/
def max_portions (ingredients : Ingredients) (ratio : Ratio) : ℕ :=
  min (ingredients.nuts / ratio.nuts)
      (min (ingredients.dried_fruit / ratio.dried_fruit)
           (min (ingredients.chocolate / ratio.chocolate)
                (ingredients.coconut / ratio.coconut)))

/-- Proves that the maximum number of portions is 2 -/
theorem max_portions_is_two :
  let ingredients := Ingredients.mk 16 6 8 4
  let ratio := Ratio.mk 4 3 2 1
  max_portions ingredients ratio = 2 :=
by
  sorry

#eval max_portions (Ingredients.mk 16 6 8 4) (Ratio.mk 4 3 2 1)

end NUMINAMATH_CALUDE_max_portions_is_two_l193_19339


namespace NUMINAMATH_CALUDE_bugs_eating_flowers_l193_19396

/-- Given 3 bugs, each eating 2 flowers, the total number of flowers eaten is 6. -/
theorem bugs_eating_flowers :
  let num_bugs : ℕ := 3
  let flowers_per_bug : ℕ := 2
  num_bugs * flowers_per_bug = 6 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eating_flowers_l193_19396


namespace NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l193_19301

/-- The number of weeks in the school year -/
def school_weeks : ℕ := 36

/-- The number of days per week Jackson could eat peanut butter and jelly sandwiches -/
def pbj_days_per_week : ℕ := 2

/-- The number of Wednesdays Jackson missed -/
def missed_wednesdays : ℕ := 1

/-- The number of Fridays Jackson missed -/
def missed_fridays : ℕ := 2

/-- The total number of peanut butter and jelly sandwiches Jackson ate -/
def total_pbj_sandwiches : ℕ := school_weeks * pbj_days_per_week - (missed_wednesdays + missed_fridays)

theorem jackson_pbj_sandwiches :
  total_pbj_sandwiches = 69 := by
  sorry

end NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l193_19301


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l193_19325

/-- Given a > 0 and a ≠ 1, prove that the function f(x) = 2 - a^(x+1) always passes through the point (-1, 1) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ 2 - a^(x + 1)
  f (-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l193_19325


namespace NUMINAMATH_CALUDE_wednesday_kids_count_l193_19357

def monday_kids : ℕ := 17
def tuesday_kids : ℕ := 15
def total_kids : ℕ := 34

theorem wednesday_kids_count : total_kids - monday_kids - tuesday_kids = 2 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_kids_count_l193_19357


namespace NUMINAMATH_CALUDE_dog_count_l193_19314

theorem dog_count (total : ℕ) (cats : ℕ) (h1 : total = 17) (h2 : cats = 8) :
  total - cats = 9 := by
  sorry

end NUMINAMATH_CALUDE_dog_count_l193_19314


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l193_19394

/-- The number of combinations of k items chosen from a set of n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of available pizza toppings -/
def n : ℕ := 7

/-- The number of toppings to be chosen -/
def k : ℕ := 3

/-- Theorem: The number of combinations of 3 toppings chosen from 7 available toppings is 35 -/
theorem pizza_toppings_combinations : binomial n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l193_19394


namespace NUMINAMATH_CALUDE_square_remainder_sum_quotient_l193_19385

theorem square_remainder_sum_quotient : 
  let squares := List.map (fun n => n^2) (List.range 6)
  let remainders := List.map (fun x => x % 13) squares
  let distinct_remainders := List.eraseDups remainders
  let m := distinct_remainders.sum
  m / 13 = 3 := by
sorry

end NUMINAMATH_CALUDE_square_remainder_sum_quotient_l193_19385


namespace NUMINAMATH_CALUDE_mean_height_is_70_625_l193_19350

def heights : List ℝ := [58, 59, 60, 61, 64, 65, 68, 70, 73, 73, 75, 76, 77, 78, 78, 79]

theorem mean_height_is_70_625 :
  (heights.sum / heights.length : ℝ) = 70.625 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_is_70_625_l193_19350


namespace NUMINAMATH_CALUDE_average_equals_50y_implies_y_value_l193_19304

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem average_equals_50y_implies_y_value :
  let n := 99
  let sum_1_to_99 := sum_to_n n
  ∀ y : ℚ, (sum_1_to_99 + y) / (n + 1 : ℚ) = 50 * y → y = 4950 / 4999 := by
sorry

end NUMINAMATH_CALUDE_average_equals_50y_implies_y_value_l193_19304


namespace NUMINAMATH_CALUDE_range_of_a_l193_19359

def prop_A (a : ℝ) : Prop :=
  ∀ x, x^2 + (a - 1) * x + a^2 > 0

def prop_B (a : ℝ) : Prop :=
  ∀ x y, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

def exclusive_or (P Q : Prop) : Prop :=
  (P ∧ ¬Q) ∨ (¬P ∧ Q)

theorem range_of_a : 
  {a : ℝ | exclusive_or (prop_A a) (prop_B a)} = 
  {a : ℝ | -1 ≤ a ∧ a < -1/2 ∨ 1/3 < a ∧ a ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l193_19359


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sin_sum_l193_19352

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_sum 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = 2 * Real.pi) : 
  Real.sin (a 2 + a 8) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sin_sum_l193_19352


namespace NUMINAMATH_CALUDE_sqrt_1_0201_l193_19389

theorem sqrt_1_0201 (h : Real.sqrt 102.01 = 10.1) : Real.sqrt 1.0201 = 1.01 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_1_0201_l193_19389


namespace NUMINAMATH_CALUDE_problem_solution_l193_19374

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.sin y = 2021)
  (h2 : x + 2021 * Real.cos y = 2020)
  (h3 : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2020 + π / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l193_19374


namespace NUMINAMATH_CALUDE_problem_solution_l193_19320

theorem problem_solution (x : ℝ) :
  x + Real.sqrt (x^2 - 1) + 1 / (x - Real.sqrt (x^2 - 1)) = 20 →
  x^2 + Real.sqrt (x^4 - 1) + 1 / (x^2 + Real.sqrt (x^4 - 1)) = 10201 / 200 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l193_19320


namespace NUMINAMATH_CALUDE_agricultural_equipment_problem_l193_19351

theorem agricultural_equipment_problem 
  (cost_2A_1B : ℝ) 
  (cost_1A_3B : ℝ) 
  (total_budget : ℝ) :
  cost_2A_1B = 4.2 →
  cost_1A_3B = 5.1 →
  total_budget = 10 →
  ∃ (cost_A cost_B : ℝ) (max_units_A : ℕ),
    cost_A = 1.5 ∧
    cost_B = 1.2 ∧
    max_units_A = 3 ∧
    2 * cost_A + cost_B = cost_2A_1B ∧
    cost_A + 3 * cost_B = cost_1A_3B ∧
    (∀ m : ℕ, m * cost_A + (2 * m - 3) * cost_B ≤ total_budget → m ≤ max_units_A) :=
by sorry

end NUMINAMATH_CALUDE_agricultural_equipment_problem_l193_19351


namespace NUMINAMATH_CALUDE_library_book_count_l193_19371

/-- The number of shelves in the library -/
def num_shelves : ℕ := 14240

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 8

/-- The total number of books in the library -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem library_book_count : total_books = 113920 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l193_19371


namespace NUMINAMATH_CALUDE_dvd_packs_calculation_l193_19335

theorem dvd_packs_calculation (total_money : ℕ) (pack_cost : ℕ) (h1 : total_money = 104) (h2 : pack_cost = 26) :
  total_money / pack_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_dvd_packs_calculation_l193_19335


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l193_19345

-- Define the two lines
def line1 (x : ℝ) : ℝ := 3 * x + 4
def line2 (x y : ℝ) : Prop := 3 * x + y = 25

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line2 x y ∧ y = line1 x ∧ x = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l193_19345


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l193_19312

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - x + 1 > 0) ↔ a > (1/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l193_19312


namespace NUMINAMATH_CALUDE_ze_and_triplet_ages_l193_19317

/-- Represents the ages of Zé Roberto and his children -/
structure FamilyAges where
  ze : ℕ
  twin : ℕ
  triplet : ℕ

/-- Conditions for Zé Roberto's family ages -/
def valid_family_ages (f : FamilyAges) : Prop :=
  -- Zé's current age equals the sum of his children's ages
  f.ze = 2 * f.twin + 3 * f.triplet ∧
  -- In 15 years, the sum of the children's ages will be twice Zé's age
  2 * (f.ze + 15) = 2 * (f.twin + 15) + 3 * (f.triplet + 15) ∧
  -- In 15 years, the sum of the twins' ages will equal the sum of the triplets' ages
  2 * (f.twin + 15) = 3 * (f.triplet + 15)

/-- Theorem stating Zé's current age and the age of each triplet -/
theorem ze_and_triplet_ages (f : FamilyAges) (h : valid_family_ages f) :
  f.ze = 45 ∧ f.triplet = 5 := by
  sorry


end NUMINAMATH_CALUDE_ze_and_triplet_ages_l193_19317


namespace NUMINAMATH_CALUDE_thirtieth_digit_of_sum_l193_19323

def fraction_sum (a b c : ℚ) : ℚ := a + b + c

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem thirtieth_digit_of_sum :
  nth_digit_after_decimal (fraction_sum (1/7) (1/3) (1/11)) 30 = 9 :=
sorry

end NUMINAMATH_CALUDE_thirtieth_digit_of_sum_l193_19323


namespace NUMINAMATH_CALUDE_sqrt_200_range_l193_19338

theorem sqrt_200_range : 14 < Real.sqrt 200 ∧ Real.sqrt 200 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_range_l193_19338


namespace NUMINAMATH_CALUDE_smallest_norm_v_l193_19348

theorem smallest_norm_v (v : ℝ × ℝ) (h : ‖v + (4, 2)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w + (4, 2)‖ = 10 ∧ ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ ∧ ‖w‖ = 10 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_norm_v_l193_19348


namespace NUMINAMATH_CALUDE_modulus_of_z_l193_19330

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = Complex.abs (2 + Complex.I) + 2 * Complex.I) : 
  Complex.abs z = 3 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_z_l193_19330


namespace NUMINAMATH_CALUDE_star_product_six_equals_twentyfour_l193_19384

/-- Custom operation definition -/
def star (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

/-- Theorem stating that if a * b = 6, then a ¤ b = 24 -/
theorem star_product_six_equals_twentyfour (a b : ℝ) (h : a * b = 6) : star a b = 24 := by
  sorry

end NUMINAMATH_CALUDE_star_product_six_equals_twentyfour_l193_19384


namespace NUMINAMATH_CALUDE_average_people_per_hour_rounded_l193_19353

def people_moving : ℕ := 3000
def days : ℕ := 4
def hours_per_day : ℕ := 24

def average_per_hour : ℚ :=
  people_moving / (days * hours_per_day)

theorem average_people_per_hour_rounded :
  round average_per_hour = 31 := by
  sorry

end NUMINAMATH_CALUDE_average_people_per_hour_rounded_l193_19353


namespace NUMINAMATH_CALUDE_diagonal_passes_through_600_cubes_l193_19392

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (a b c : ℕ) : ℕ :=
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c)

/-- Theorem: For a 120 × 270 × 300 rectangular solid, the internal diagonal passes through 600 cubes -/
theorem diagonal_passes_through_600_cubes :
  cubes_passed_by_diagonal 120 270 300 = 600 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_600_cubes_l193_19392


namespace NUMINAMATH_CALUDE_joan_took_25_marbles_l193_19336

-- Define the initial number of yellow marbles
def initial_yellow_marbles : ℕ := 86

-- Define the remaining number of yellow marbles
def remaining_yellow_marbles : ℕ := 61

-- Define the number of yellow marbles Joan took
def marbles_taken : ℕ := initial_yellow_marbles - remaining_yellow_marbles

-- Theorem to prove
theorem joan_took_25_marbles : marbles_taken = 25 := by
  sorry

end NUMINAMATH_CALUDE_joan_took_25_marbles_l193_19336


namespace NUMINAMATH_CALUDE_center_is_nine_l193_19331

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Check if a grid satisfies the consecutive number condition --/
def consecutiveAdjacent (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, (g i j).succ = g k l → adjacent (i, j) (k, l)

/-- Sum of corner numbers in the grid --/
def cornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- The center number in the grid --/
def centerNumber (g : Grid) : Nat := g 1 1

/-- All numbers from 1 to 9 are used in the grid --/
def usesAllNumbers (g : Grid) : Prop :=
  ∀ n : Nat, n ≥ 1 → n ≤ 9 → ∃ i j : Fin 3, g i j = n

theorem center_is_nine (g : Grid) 
    (h1 : usesAllNumbers g)
    (h2 : consecutiveAdjacent g)
    (h3 : cornerSum g = 20) :
  centerNumber g = 9 := by
  sorry

end NUMINAMATH_CALUDE_center_is_nine_l193_19331


namespace NUMINAMATH_CALUDE_no_two_digit_even_square_palindromes_l193_19313

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem no_two_digit_even_square_palindromes :
  ¬ ∃ n : ℕ, is_two_digit n ∧ is_perfect_square n ∧ 
    (∃ m : ℕ, n = m * m ∧ is_even m) ∧ is_palindrome n :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_even_square_palindromes_l193_19313


namespace NUMINAMATH_CALUDE_sugar_amount_is_one_cup_l193_19302

/-- Represents the ratio of ingredients in a recipe --/
structure Ratio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio --/
def originalRatio : Ratio :=
  { flour := 7, water := 2, sugar := 1 }

/-- The new recipe ratio --/
def newRatio : Ratio :=
  { flour := originalRatio.flour * 2,
    water := originalRatio.water,
    sugar := originalRatio.sugar * 2 }

/-- The amount of water in the new recipe (in cups) --/
def newWaterAmount : ℚ := 2

/-- Calculates the amount of sugar needed in the new recipe --/
def sugarNeeded (r : Ratio) (waterAmount : ℚ) : ℚ :=
  (waterAmount * r.sugar) / r.water

/-- Theorem stating that the amount of sugar needed in the new recipe is 1 cup --/
theorem sugar_amount_is_one_cup :
  sugarNeeded newRatio newWaterAmount = 1 := by
  sorry


end NUMINAMATH_CALUDE_sugar_amount_is_one_cup_l193_19302


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_min_value_is_nine_l193_19318

theorem min_value_sum_of_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  ∀ x y, x > 0 → y > 0 → 2/a + 1/b ≤ 2/x + 1/y :=
by
  sorry

theorem min_value_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  ∃ x y, x > 0 ∧ y > 0 ∧ 2/x + 1/y = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_min_value_is_nine_l193_19318


namespace NUMINAMATH_CALUDE_geometric_sequence_decreasing_l193_19355

def geometric_sequence (n : ℕ) : ℝ := 4 * (3 ^ (1 - n))

theorem geometric_sequence_decreasing :
  ∀ n : ℕ, geometric_sequence (n + 1) < geometric_sequence n :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_decreasing_l193_19355


namespace NUMINAMATH_CALUDE_combined_mean_score_l193_19308

/-- Given two classes with different average scores and a ratio of students, 
    calculate the combined mean score. -/
theorem combined_mean_score (avg1 avg2 : ℝ) (ratio1 ratio2 : ℕ) : 
  avg1 = 90 →
  avg2 = 75 →
  ratio1 = 2 →
  ratio2 = 3 →
  (avg1 * ratio1 + avg2 * ratio2) / (ratio1 + ratio2) = 81 := by
sorry

end NUMINAMATH_CALUDE_combined_mean_score_l193_19308


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l193_19393

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((Complex.I / (1 + 2 * Complex.I)) * Complex.I) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l193_19393


namespace NUMINAMATH_CALUDE_triangle_angle_b_sixty_degrees_l193_19376

theorem triangle_angle_b_sixty_degrees 
  (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (condition : c / (a + b) + a / (b + c) = 1) : 
  angle_b = π / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_b_sixty_degrees_l193_19376


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l193_19316

/-- Given a real number a, if f(x) = x^3 + ax^2 + (a - 3)x and its derivative f'(x) is an even function,
    then the equation of the tangent line to the curve y = f(x) at the origin is 2x + y = 0. -/
theorem tangent_line_at_origin (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a - 3)*x
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 + 2*a*x + (a - 3)
  (∀ x, f' x = f' (-x)) →  -- f' is an even function
  (λ x y ↦ 2*x + y = 0) = (λ x y ↦ y = f' 0 * x) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l193_19316


namespace NUMINAMATH_CALUDE_batsman_boundaries_l193_19356

theorem batsman_boundaries (total_runs : ℕ) (sixes : ℕ) (run_percentage : ℚ) : 
  total_runs = 120 →
  sixes = 8 →
  run_percentage = 1/2 →
  (∃ (boundaries : ℕ), 
    total_runs = run_percentage * total_runs + sixes * 6 + boundaries * 4 ∧
    boundaries = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_batsman_boundaries_l193_19356


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l193_19319

theorem smallest_prime_divisor_of_sum (n : ℕ) (h : n = 3^15 + 5^21) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → p ≤ q :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l193_19319


namespace NUMINAMATH_CALUDE_sequence_not_periodic_l193_19378

/-- The sequence (a_n) defined by a_n = ⌊x^(n+1)⌋ - x⌊x^n⌋ is not periodic for any real x > 1 that is not an integer. -/
theorem sequence_not_periodic (x : ℝ) (hx : x > 1) (hx_not_int : ¬ ∃ n : ℤ, x = n) :
  ¬ ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, 
    (⌊x^(n+1)⌋ - x * ⌊x^n⌋ : ℝ) = (⌊x^(n+p+1)⌋ - x * ⌊x^(n+p)⌋ : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_periodic_l193_19378


namespace NUMINAMATH_CALUDE_hall_length_proof_l193_19397

theorem hall_length_proof (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = breadth + 5 →
  area = length * breadth →
  area = 750 →
  length = 30 := by
sorry

end NUMINAMATH_CALUDE_hall_length_proof_l193_19397


namespace NUMINAMATH_CALUDE_inequality_solutions_l193_19344

def inequality (a x : ℝ) := a * x^2 - (a + 2) * x + 2 < 0

theorem inequality_solutions :
  ∀ a : ℝ,
    (a = -1 → {x : ℝ | inequality a x} = {x : ℝ | x < -2 ∨ x > 1}) ∧
    (a = 0 → {x : ℝ | inequality a x} = {x : ℝ | x > 1}) ∧
    (a < 0 → {x : ℝ | inequality a x} = {x : ℝ | x < 2/a ∨ x > 1}) ∧
    (0 < a ∧ a < 2 → {x : ℝ | inequality a x} = {x : ℝ | 1 < x ∧ x < 2/a}) ∧
    (a = 2 → {x : ℝ | inequality a x} = ∅) ∧
    (a > 2 → {x : ℝ | inequality a x} = {x : ℝ | 2/a < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l193_19344


namespace NUMINAMATH_CALUDE_binomial_properties_l193_19349

-- Define the binomial distribution
def binomial_distribution (n : ℕ) (p : ℝ) : ℕ → ℝ := sorry

-- Define the probability of X being odd
def prob_odd (n : ℕ) (p : ℝ) : ℝ := sorry

-- Define the probability of X being even
def prob_even (n : ℕ) (p : ℝ) : ℝ := sorry

theorem binomial_properties (n : ℕ) (p : ℝ) 
  (h1 : n > 0) (h2 : 0 < p) (h3 : p < 1) :
  -- 1. The sum of probabilities of X being odd and even equals 1
  (prob_odd n p + prob_even n p = 1) ∧ 
  -- 2. When p = 1/2, the probability of X being odd equals the probability of X being even
  (p = 1/2 → prob_odd n p = prob_even n p) ∧ 
  -- 3. When 0 < p < 1/2, the probability of X being odd increases as n increases
  (p < 1/2 → ∀ m, n < m → prob_odd n p < prob_odd m p) :=
by sorry

end NUMINAMATH_CALUDE_binomial_properties_l193_19349


namespace NUMINAMATH_CALUDE_system_solution_l193_19309

/-- Given a system of equations:
    1) x = 1.12 * y + 52.8
    2) x = y + 50
    Prove that the solution is approximately x ≈ 26.67 and y ≈ -23.33 -/
theorem system_solution :
  ∃ (x y : ℝ),
    (x = 1.12 * y + 52.8) ∧
    (x = y + 50) ∧
    (abs (x - 26.67) < 0.01) ∧
    (abs (y + 23.33) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l193_19309


namespace NUMINAMATH_CALUDE_mishas_current_money_l193_19399

/-- Misha's current amount of money in dollars -/
def current_money : ℕ := sorry

/-- The amount Misha needs to earn in dollars -/
def money_to_earn : ℕ := 13

/-- The total amount Misha will have after earning more money, in dollars -/
def total_money : ℕ := 47

/-- Theorem stating Misha's current amount of money -/
theorem mishas_current_money : current_money = 34 := by sorry

end NUMINAMATH_CALUDE_mishas_current_money_l193_19399


namespace NUMINAMATH_CALUDE_pages_difference_l193_19390

/-- Represents the number of pages in a purple book -/
def purple_pages : ℕ := 230

/-- Represents the number of pages in an orange book -/
def orange_pages : ℕ := 510

/-- Represents the number of purple books Mirella read -/
def purple_books_read : ℕ := 5

/-- Represents the number of orange books Mirella read -/
def orange_books_read : ℕ := 4

/-- Theorem stating the difference in pages read between orange and purple books -/
theorem pages_difference : 
  orange_pages * orange_books_read - purple_pages * purple_books_read = 890 := by
  sorry

end NUMINAMATH_CALUDE_pages_difference_l193_19390


namespace NUMINAMATH_CALUDE_gcd_product_is_square_l193_19306

theorem gcd_product_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, (Nat.gcd x y).gcd z * x * y * z = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_product_is_square_l193_19306


namespace NUMINAMATH_CALUDE_total_weight_of_baskets_l193_19342

def basket_weight : ℕ := 30
def num_baskets : ℕ := 8

theorem total_weight_of_baskets : basket_weight * num_baskets = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_baskets_l193_19342


namespace NUMINAMATH_CALUDE_triangle_ratio_l193_19328

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2b*sin(2A) = 3a*sin(B) and c = 2b, then a/b = √2 -/
theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * b * Real.sin (2 * A) = 3 * a * Real.sin B →
  c = 2 * b →
  a / b = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l193_19328


namespace NUMINAMATH_CALUDE_right_triangle_area_l193_19383

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 10 →
  angle = 30 * π / 180 →
  let shorter_leg := hypotenuse / 2
  let longer_leg := shorter_leg * Real.sqrt 3
  let area := (shorter_leg * longer_leg) / 2
  area = (25 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l193_19383


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l193_19341

theorem imaginary_part_of_complex_number (z : ℂ) : z = 1 + 1 / Complex.I → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l193_19341


namespace NUMINAMATH_CALUDE_total_books_count_l193_19377

/-- Given that Sandy has 10 books, Benny has 24 books, and Tim has 33 books,
    prove that they have 67 books in total. -/
theorem total_books_count (sandy_books benny_books tim_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : benny_books = 24)
  (h3 : tim_books = 33) : 
  sandy_books + benny_books + tim_books = 67 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l193_19377


namespace NUMINAMATH_CALUDE_special_integers_count_l193_19366

/-- Sum of all positive divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- The number of integers j such that 1 ≤ j ≤ 5041 and g(j) = 1 + √j + j -/
def count_special_integers : ℕ := sorry

theorem special_integers_count :
  count_special_integers = 20 := by sorry

end NUMINAMATH_CALUDE_special_integers_count_l193_19366


namespace NUMINAMATH_CALUDE_valid_seven_digit_integers_l193_19300

-- Define the recurrence relation
def a : ℕ → ℕ
  | 0 => 0  -- Base case (not used)
  | 1 => 4  -- a₁ = 4
  | 2 => 17 -- a₂ = 17
  | n + 3 => 4 * a (n + 2) + 2 * a (n + 1)

-- Theorem statement
theorem valid_seven_digit_integers : a 7 = 29776 := by
  sorry

end NUMINAMATH_CALUDE_valid_seven_digit_integers_l193_19300


namespace NUMINAMATH_CALUDE_fish_population_estimation_l193_19307

/-- Calculates the number of fish in a lake on May 1st given certain conditions --/
theorem fish_population_estimation (marked_may : ℕ) (caught_sept : ℕ) (marked_sept : ℕ)
  (death_rate : ℚ) (new_fish_rate : ℚ) :
  marked_may = 60 →
  caught_sept = 70 →
  marked_sept = 3 →
  death_rate = 1/4 →
  new_fish_rate = 2/5 →
  ∃ (fish_may : ℕ), fish_may = 840 ∧ 
    (fish_may : ℚ) * (1 - death_rate) * (marked_sept : ℚ) / (caught_sept : ℚ) = 
    (marked_may : ℚ) * (1 - death_rate) ∧
    (fish_may : ℚ) * (1 - death_rate) / (1 - new_fish_rate) = 
    (fish_may : ℚ) * (1 - death_rate) * (marked_sept : ℚ) / (caught_sept : ℚ) / (1 - new_fish_rate) :=
by
  sorry


end NUMINAMATH_CALUDE_fish_population_estimation_l193_19307


namespace NUMINAMATH_CALUDE_prank_combinations_l193_19363

theorem prank_combinations (choices : List Nat) : 
  choices = [2, 3, 0, 6, 1] → List.prod choices = 0 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l193_19363


namespace NUMINAMATH_CALUDE_max_duck_moves_l193_19387

/-- 
Given positive integers a, b, and c representing the number of ducks 
picking rock, paper, and scissors respectively in a circular arrangement, 
the maximum number of possible moves according to the rock-paper-scissors 
switching rules is max(a × b, b × c, c × a).
-/
theorem max_duck_moves (a b c : ℕ+) : 
  ∃ (max_moves : ℕ), max_moves = max (a * b) (max (b * c) (c * a)) ∧
  ∀ (moves : ℕ), moves ≤ max_moves := by
sorry


end NUMINAMATH_CALUDE_max_duck_moves_l193_19387


namespace NUMINAMATH_CALUDE_parabola_equation_and_vertex_l193_19370

/-- A parabola passing through points (1, 0) and (3, 0) -/
def Parabola (x y : ℝ) : Prop :=
  ∃ b c : ℝ, y = -x^2 + b*x + c ∧ 0 = -1 + b + c ∧ 0 = -9 + 3*b + c

theorem parabola_equation_and_vertex :
  (∀ x y : ℝ, Parabola x y ↔ y = -x^2 + 4*x - 3) ∧
  (∃ x y : ℝ, Parabola x y ∧ x = 2 ∧ y = 1 ∧
    ∀ x' y' : ℝ, Parabola x' y' → y' ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_and_vertex_l193_19370


namespace NUMINAMATH_CALUDE_inequality_equivalence_l193_19391

/-- The inequality holds for all positive q if and only if p is in the interval [0, 2) -/
theorem inequality_equivalence (p : ℝ) : 
  (∀ q : ℝ, q > 0 → (4 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 5 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 
  (0 ≤ p ∧ p < 2) := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l193_19391
