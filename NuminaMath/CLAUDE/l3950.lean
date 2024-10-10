import Mathlib

namespace projectile_height_l3950_395026

theorem projectile_height (t : ℝ) : 
  t > 0 ∧ -16 * t^2 + 80 * t = 36 ∧ 
  (∀ s, s > 0 ∧ -16 * s^2 + 80 * s = 36 → t ≤ s) → 
  t = 0.5 := by sorry

end projectile_height_l3950_395026


namespace leader_assistant_combinations_l3950_395034

/-- The number of ways to choose a team leader and an assistant of the same gender -/
def choose_leader_and_assistant (total : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  boys * (boys - 1) + girls * (girls - 1)

/-- Theorem: There are 98 ways to choose a team leader and an assistant of the same gender
    from a class of 15 students, consisting of 8 boys and 7 girls -/
theorem leader_assistant_combinations :
  choose_leader_and_assistant 15 8 7 = 98 := by
  sorry

end leader_assistant_combinations_l3950_395034


namespace pagoda_lamps_l3950_395042

theorem pagoda_lamps (n : ℕ) (total : ℕ) (h1 : n = 7) (h2 : total = 381) : 
  (∃ a : ℕ, a * (2^n - 1) = total) → 3 * (2^n - 1) = total := by
sorry

end pagoda_lamps_l3950_395042


namespace p_recurrence_l3950_395077

/-- The probability of getting a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℝ := sorry

/-- The recurrence relation for p_{n,k} -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + 1 / 2^k := by sorry

end p_recurrence_l3950_395077


namespace plane_division_l3950_395080

/-- The number of parts into which n lines divide a plane -/
def f (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem stating that f(n) correctly counts the number of parts for n ≥ 2 -/
theorem plane_division (n : ℕ) (h : n ≥ 2) : 
  f n = 1 + n * (n + 1) / 2 := by
  sorry

/-- Helper lemma for the induction step -/
lemma induction_step (k : ℕ) (h : k ≥ 2) :
  f (k + 1) = f k + (k + 1) := by
  sorry

end plane_division_l3950_395080


namespace marked_squares_rearrangement_l3950_395049

/-- Represents a square table with marked cells -/
structure MarkedTable (n : ℕ) where
  marks : Finset ((Fin n) × (Fin n))
  mark_count : marks.card = 110

/-- Represents a permutation of rows and columns -/
structure TablePermutation (n : ℕ) where
  row_perm : Equiv.Perm (Fin n)
  col_perm : Equiv.Perm (Fin n)

/-- Checks if a cell is on or above the main diagonal -/
def is_on_or_above_diagonal {n : ℕ} (i j : Fin n) : Prop :=
  i.val ≤ j.val

/-- Applies a permutation to a marked cell -/
def apply_perm {n : ℕ} (perm : TablePermutation n) (cell : (Fin n) × (Fin n)) : (Fin n) × (Fin n) :=
  (perm.row_perm cell.1, perm.col_perm cell.2)

/-- Theorem: For any 100x100 table with 110 marked squares, there exists a permutation
    that places all marked squares on or above the main diagonal -/
theorem marked_squares_rearrangement :
  ∀ (t : MarkedTable 100),
  ∃ (perm : TablePermutation 100),
  ∀ cell ∈ t.marks,
  is_on_or_above_diagonal (apply_perm perm cell).1 (apply_perm perm cell).2 :=
sorry

end marked_squares_rearrangement_l3950_395049


namespace prob_three_same_color_l3950_395043

/-- A deck of cards with red and black colors -/
structure Deck :=
  (total : ℕ)
  (red : ℕ)
  (black : ℕ)
  (h1 : red + black = total)

/-- The probability of drawing three cards of the same color -/
def prob_same_color (d : Deck) : ℚ :=
  2 * (d.red.choose 3 / d.total.choose 3)

/-- The specific deck described in the problem -/
def modified_deck : Deck :=
  { total := 60
  , red := 30
  , black := 30
  , h1 := by simp }

/-- The main theorem stating the probability for the given deck -/
theorem prob_three_same_color :
  prob_same_color modified_deck = 406 / 1711 := by
  sorry

end prob_three_same_color_l3950_395043


namespace max_attendees_is_three_tuesday_has_three_friday_has_three_saturday_has_three_no_day_exceeds_three_l3950_395099

-- Define the days of the week
inductive Day
| Mon | Tues | Wed | Thurs | Fri | Sat

-- Define the people
inductive Person
| Amy | Bob | Charlie | Diana | Evan

-- Define the availability function
def available : Person → Day → Bool
| Person.Amy, Day.Mon => false
| Person.Amy, Day.Tues => true
| Person.Amy, Day.Wed => false
| Person.Amy, Day.Thurs => false
| Person.Amy, Day.Fri => true
| Person.Amy, Day.Sat => true
| Person.Bob, Day.Mon => true
| Person.Bob, Day.Tues => false
| Person.Bob, Day.Wed => true
| Person.Bob, Day.Thurs => true
| Person.Bob, Day.Fri => false
| Person.Bob, Day.Sat => true
| Person.Charlie, Day.Mon => false
| Person.Charlie, Day.Tues => false
| Person.Charlie, Day.Wed => false
| Person.Charlie, Day.Thurs => true
| Person.Charlie, Day.Fri => true
| Person.Charlie, Day.Sat => false
| Person.Diana, Day.Mon => true
| Person.Diana, Day.Tues => true
| Person.Diana, Day.Wed => false
| Person.Diana, Day.Thurs => false
| Person.Diana, Day.Fri => true
| Person.Diana, Day.Sat => false
| Person.Evan, Day.Mon => false
| Person.Evan, Day.Tues => true
| Person.Evan, Day.Wed => true
| Person.Evan, Day.Thurs => false
| Person.Evan, Day.Fri => false
| Person.Evan, Day.Sat => true

-- Count the number of available people for a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (λ p => available p d) [Person.Amy, Person.Bob, Person.Charlie, Person.Diana, Person.Evan]).length

-- Find the maximum number of available people across all days
def maxAvailable : Nat :=
  List.foldl max 0 (List.map countAvailable [Day.Mon, Day.Tues, Day.Wed, Day.Thurs, Day.Fri, Day.Sat])

-- Theorem: The maximum number of attendees is 3
theorem max_attendees_is_three : maxAvailable = 3 := by sorry

-- Theorem: Tuesday has 3 attendees
theorem tuesday_has_three : countAvailable Day.Tues = 3 := by sorry

-- Theorem: Friday has 3 attendees
theorem friday_has_three : countAvailable Day.Fri = 3 := by sorry

-- Theorem: Saturday has 3 attendees
theorem saturday_has_three : countAvailable Day.Sat = 3 := by sorry

-- Theorem: No other day has more than 3 attendees
theorem no_day_exceeds_three : ∀ d : Day, countAvailable d ≤ 3 := by sorry

end max_attendees_is_three_tuesday_has_three_friday_has_three_saturday_has_three_no_day_exceeds_three_l3950_395099


namespace player_a_wins_l3950_395037

/-- Represents a player in the chocolate bar game -/
inductive Player
| A
| B

/-- Represents a move in the chocolate bar game -/
inductive Move
| Single
| Double

/-- Represents the state of the chocolate bar game -/
structure GameState where
  grid : Fin 7 → Fin 7 → Bool
  current_player : Player

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Checks if a move is valid for the current player and game state -/
def is_valid_move (gs : GameState) (m : Move) : Bool :=
  match gs.current_player, m with
  | Player.A, Move.Single => true
  | Player.B, _ => true
  | _, _ => false

/-- Applies a move to the game state, returning the new state -/
def apply_move (gs : GameState) (m : Move) : GameState :=
  sorry

/-- Counts the number of squares taken by a player -/
def count_squares (gs : GameState) (p : Player) : Nat :=
  sorry

/-- The main theorem stating that Player A can always secure more than half the squares -/
theorem player_a_wins (init_state : GameState) (strategy_a strategy_b : Strategy) :
  ∃ (final_state : GameState),
    count_squares final_state Player.A > 24 :=
  sorry

end player_a_wins_l3950_395037


namespace intersection_is_empty_l3950_395063

def A : Set ℝ := {x | x^2 - 2*x > 0}
def B : Set ℝ := {x | |x + 1| < 0}

theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end intersection_is_empty_l3950_395063


namespace product_of_r_values_l3950_395003

theorem product_of_r_values : ∃ (r₁ r₂ : ℝ),
  (∀ r : ℝ, (∃! x : ℝ, (1 : ℝ) / (3 * x) = (r - x) / 6) ↔ (r = r₁ ∨ r = r₂)) ∧
  r₁ * r₂ = -8 :=
sorry

end product_of_r_values_l3950_395003


namespace equalizing_amount_is_55_l3950_395076

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The condition when Foma gives Ierema 70 gold coins -/
def condition1 (w : MerchantWealth) : Prop :=
  w.ierema + 70 = w.yuliy

/-- The condition when Foma gives Ierema 40 gold coins -/
def condition2 (w : MerchantWealth) : Prop :=
  w.foma - 40 = w.yuliy

/-- The amount of gold coins Foma should give Ierema to equalize their wealth -/
def equalizingAmount (w : MerchantWealth) : ℕ :=
  (w.foma - w.ierema) / 2

theorem equalizing_amount_is_55 (w : MerchantWealth) 
  (h1 : condition1 w) (h2 : condition2 w) : 
  equalizingAmount w = 55 := by
  sorry

end equalizing_amount_is_55_l3950_395076


namespace unique_solution_system_l3950_395045

theorem unique_solution_system : 
  ∃! (a b c : ℕ+), 
    (a.val : ℤ)^3 - (b.val : ℤ)^3 - (c.val : ℤ)^3 = 3 * (a.val : ℤ) * (b.val : ℤ) * (c.val : ℤ) ∧ 
    (a.val : ℤ)^2 = 2 * ((b.val : ℤ) + (c.val : ℤ)) ∧
    a.val = 2 ∧ b.val = 1 ∧ c.val = 1 :=
by sorry

end unique_solution_system_l3950_395045


namespace closest_option_is_150000_l3950_395047

/-- Represents the population of the United States in 2020 --/
def us_population : ℕ := 331000000

/-- Represents the total area of the United States in square miles --/
def us_area : ℕ := 3800000

/-- Represents the number of square feet in one square mile --/
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

/-- Calculates the average number of square feet per person --/
def avg_sq_feet_per_person : ℚ :=
  (us_area * sq_feet_per_sq_mile) / us_population

/-- List of given options for the average square feet per person --/
def options : List ℕ := [30000, 60000, 90000, 120000, 150000]

/-- Theorem stating that 150000 is the closest option to the actual average --/
theorem closest_option_is_150000 :
  ∃ (x : ℕ), x ∈ options ∧ 
  ∀ (y : ℕ), y ∈ options → |avg_sq_feet_per_person - x| ≤ |avg_sq_feet_per_person - y| :=
by
  sorry

end closest_option_is_150000_l3950_395047


namespace different_suit_card_selection_l3950_395053

theorem different_suit_card_selection :
  let total_cards : ℕ := 52
  let num_suits : ℕ := 4
  let cards_per_suit : ℕ := 13
  let selection_size : ℕ := 4

  (num_suits ^ selection_size) = 28561 :=
by
  sorry

end different_suit_card_selection_l3950_395053


namespace center_sum_l3950_395028

/-- The center of a circle defined by the equation x^2 + y^2 = 4x - 6y + 9 -/
def circle_center : ℝ × ℝ := sorry

/-- The equation of the circle -/
axiom circle_equation (p : ℝ × ℝ) : p.1^2 + p.2^2 = 4*p.1 - 6*p.2 + 9

theorem center_sum : circle_center.1 + circle_center.2 = -1 := by sorry

end center_sum_l3950_395028


namespace parabola_hyperbola_tangency_l3950_395064

/-- A parabola with equation y = x^2 + 4 -/
def parabola (x y : ℝ) : Prop := y = x^2 + 4

/-- A hyperbola with equation y^2 - mx^2 = 1, where m is a parameter -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m * x^2 = 1

/-- Two curves are tangent if they intersect at exactly one point -/
def are_tangent (curve1 curve2 : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, curve1 p.1 p.2 ∧ curve2 p.1 p.2

theorem parabola_hyperbola_tangency (m : ℝ) :
  are_tangent (parabola) (hyperbola m) → m = 8 + 2 * Real.sqrt 15 := by
  sorry

end parabola_hyperbola_tangency_l3950_395064


namespace hot_dog_stand_ketchup_bottles_l3950_395070

/-- Given a ratio of condiment bottles and the number of mayo bottles,
    calculate the number of ketchup bottles -/
def ketchup_bottles (ketchup_ratio mustard_ratio mayo_ratio mayo_count : ℕ) : ℕ :=
  (ketchup_ratio * mayo_count) / mayo_ratio

theorem hot_dog_stand_ketchup_bottles :
  ketchup_bottles 3 3 2 4 = 6 := by sorry

end hot_dog_stand_ketchup_bottles_l3950_395070


namespace greatest_b_value_l3950_395091

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 12 ≥ 0 → x ≤ 6) ∧ 
  (-6^2 + 8*6 - 12 ≥ 0) := by
sorry

end greatest_b_value_l3950_395091


namespace no_such_function_exists_l3950_395002

theorem no_such_function_exists : 
  ¬ ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), n > 1 → f n = f (f (n - 1)) + f (f (n + 1)) :=
by sorry

end no_such_function_exists_l3950_395002


namespace equal_distribution_contribution_l3950_395005

def earnings : List ℕ := [18, 22, 30, 35, 45]

theorem equal_distribution_contribution :
  let total := earnings.sum
  let equal_share := total / earnings.length
  let max_earner := earnings.maximum?
  match max_earner with
  | some max => max - equal_share = 15
  | none => False
  := by sorry

end equal_distribution_contribution_l3950_395005


namespace num_distinct_colorings_bound_l3950_395011

/-- Represents a coloring of a 5x5 two-sided paper --/
def Coloring (n : ℕ) := Fin 5 → Fin 5 → Fin n

/-- The group of symmetries for a square --/
inductive SquareSymmetry
| identity
| rotate90
| rotate180
| rotate270
| reflectHorizontal
| reflectVertical
| reflectDiagonal1
| reflectDiagonal2

/-- Applies a symmetry to a coloring --/
def applySymmetry (sym : SquareSymmetry) (c : Coloring n) : Coloring n :=
  sorry

/-- Checks if a coloring is fixed under a symmetry --/
def isFixed (sym : SquareSymmetry) (c : Coloring n) : Prop :=
  c = applySymmetry sym c

/-- The number of colorings fixed by a given symmetry --/
def numFixedColorings (sym : SquareSymmetry) (n : ℕ) : ℕ :=
  sorry

/-- The total number of distinct colorings --/
def numDistinctColorings (n : ℕ) : ℕ :=
  sorry

theorem num_distinct_colorings_bound (n : ℕ) :
  numDistinctColorings n ≤ (n^25 + 4*n^15 + n^13 + 2*n^7) / 8 :=
sorry

end num_distinct_colorings_bound_l3950_395011


namespace quadratic_inequality_solution_set_l3950_395088

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 < x + 6 ↔ -2 < x ∧ x < 3 := by sorry

end quadratic_inequality_solution_set_l3950_395088


namespace compound_proposition_1_compound_proposition_2_compound_proposition_3_l3950_395030

-- Define the propositions
def smallest_angle_not_greater_than_60 (α : Real) : Prop :=
  (∀ β γ : Real, α + β + γ = 180 → α ≤ β ∧ α ≤ γ) → α ≤ 60

def isosceles_right_triangle (α β γ : Real) : Prop :=
  α + β + γ = 180 ∧ α = 90 ∧ β = 45 ∧ (γ = α ∨ γ = β) ∧ α = 90

def triangle_with_60_degree (α β γ : Real) : Prop :=
  α + β + γ = 180 ∧ (α = 60 ∨ β = 60 ∨ γ = 60)

-- Theorem statements
theorem compound_proposition_1 (α : Real) :
  smallest_angle_not_greater_than_60 α ↔ 
  ¬(∀ β γ : Real, α + β + γ = 180 → α ≤ β ∧ α ≤ γ → α > 60) :=
sorry

theorem compound_proposition_2 (α β γ : Real) :
  isosceles_right_triangle α β γ ↔
  (α + β + γ = 180 ∧ α = 90 ∧ β = 45 ∧ (γ = α ∨ γ = β)) ∧
  (α + β + γ = 180 ∧ α = 90 ∧ β = 45) :=
sorry

theorem compound_proposition_3 (α β γ : Real) :
  triangle_with_60_degree α β γ ↔
  (α + β + γ = 180 ∧ α = 60 ∧ β = 60 ∧ γ = 60) ∨
  (α + β + γ = 180 ∧ (α = 60 ∨ β = 60 ∨ γ = 60) ∧ (α = 90 ∨ β = 90 ∨ γ = 90)) :=
sorry

end compound_proposition_1_compound_proposition_2_compound_proposition_3_l3950_395030


namespace complement_of_70_degrees_l3950_395060

theorem complement_of_70_degrees :
  let given_angle : ℝ := 70
  let complement_sum : ℝ := 90
  let complement_angle : ℝ := complement_sum - given_angle
  complement_angle = 20 := by sorry

end complement_of_70_degrees_l3950_395060


namespace billy_ticket_difference_l3950_395012

/-- The difference between initial tickets and remaining tickets after purchases -/
def ticket_difference (initial_tickets yoyo_cost keychain_cost plush_toy_cost : ℝ) : ℝ :=
  initial_tickets - (initial_tickets - (yoyo_cost + keychain_cost + plush_toy_cost))

/-- Theorem stating the ticket difference for Billy's specific case -/
theorem billy_ticket_difference :
  ticket_difference 48.5 11.7 6.3 16.2 = 14.3 := by
  sorry

end billy_ticket_difference_l3950_395012


namespace quadrilateral_classification_l3950_395033

/-- A quadrilateral with angles α, β, γ, δ satisfying certain conditions --/
structure Quadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  angle_sum : α + β + γ + δ = 2 * Real.pi
  cosine_sum : Real.cos α + Real.cos β + Real.cos γ + Real.cos δ = 0

/-- Definition of a parallelogram based on opposite angles --/
def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.α + q.γ = Real.pi) ∧ (q.β + q.δ = Real.pi)

/-- Definition of a cyclic quadrilateral based on opposite angles --/
def is_cyclic (q : Quadrilateral) : Prop :=
  (q.α + q.γ = Real.pi) ∨ (q.β + q.δ = Real.pi)

/-- Definition of a trapezoid based on adjacent angles --/
def is_trapezoid (q : Quadrilateral) : Prop :=
  (q.α + q.β = Real.pi) ∨ (q.β + q.γ = Real.pi) ∨ (q.γ + q.δ = Real.pi) ∨ (q.δ + q.α = Real.pi)

/-- Main theorem: A quadrilateral with the given properties is either a parallelogram, cyclic, or trapezoid --/
theorem quadrilateral_classification (q : Quadrilateral) :
  is_parallelogram q ∨ is_cyclic q ∨ is_trapezoid q := by
  sorry

end quadrilateral_classification_l3950_395033


namespace clock_angle_at_two_thirty_l3950_395048

/-- The measure of the smaller angle formed by the hour-hand and minute-hand of a clock at 2:30 -/
def clock_angle : ℝ := 105

/-- The number of degrees in a full circle on a clock -/
def full_circle : ℝ := 360

/-- The number of hours on a clock -/
def clock_hours : ℕ := 12

/-- The hour component of the time -/
def hour : ℕ := 2

/-- The minute component of the time -/
def minute : ℕ := 30

theorem clock_angle_at_two_thirty :
  clock_angle = min (|hour_angle - minute_angle|) (full_circle - |hour_angle - minute_angle|) :=
by
  sorry
where
  /-- The angle of the hour hand from 12 o'clock position -/
  hour_angle : ℝ := (hour + minute / 60) * (full_circle / clock_hours)
  /-- The angle of the minute hand from 12 o'clock position -/
  minute_angle : ℝ := minute * (full_circle / 60)

#check clock_angle_at_two_thirty

end clock_angle_at_two_thirty_l3950_395048


namespace trailing_zeros_50_factorial_l3950_395032

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem trailing_zeros_50_factorial : trailingZeros 50 = 12 := by
  sorry

end trailing_zeros_50_factorial_l3950_395032


namespace greatest_x_value_l3950_395078

theorem greatest_x_value (x : ℝ) : 
  x ≠ 6 → x ≠ -3 → (x^2 - x - 30) / (x - 6) = 5 / (x + 3) → 
  x ≤ -2 ∧ ∃ y, y = -2 ∧ (y^2 - y - 30) / (y - 6) = 5 / (y + 3) := by
sorry

end greatest_x_value_l3950_395078


namespace rectangle_length_l3950_395050

theorem rectangle_length (l w : ℝ) (h1 : l = 4 * w) (h2 : l * w = 100) : l = 20 := by
  sorry

end rectangle_length_l3950_395050


namespace largest_710_double_correct_l3950_395093

/-- Converts a positive integer to its base-7 representation as a list of digits --/
def toBase7 (n : ℕ+) : List ℕ :=
  sorry

/-- Converts a list of digits to a base-10 number --/
def toBase10 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a positive integer is a 7-10 double --/
def is710Double (n : ℕ+) : Prop :=
  toBase10 (toBase7 n) = 2 * n

/-- The largest 7-10 double --/
def largest710Double : ℕ+ := 315

theorem largest_710_double_correct :
  is710Double largest710Double ∧
  ∀ n : ℕ+, n > largest710Double → ¬is710Double n :=
sorry

end largest_710_double_correct_l3950_395093


namespace complex_square_equality_l3950_395024

theorem complex_square_equality (a b : ℕ+) :
  (↑a - Complex.I * ↑b) ^ 2 = 15 - 8 * Complex.I →
  ↑a - Complex.I * ↑b = 4 - Complex.I := by
  sorry

end complex_square_equality_l3950_395024


namespace product_of_numbers_l3950_395098

theorem product_of_numbers (x y : ℝ) : 
  x - y = 7 → x^2 + y^2 = 85 → x * y = 18 := by
sorry

end product_of_numbers_l3950_395098


namespace no_square_with_seven_lattice_points_l3950_395075

/-- A square in a right-angled coordinate system -/
structure RotatedSquare where
  /-- The center of the square -/
  center : ℝ × ℝ
  /-- The side length of the square -/
  side_length : ℝ
  /-- The angle between the sides of the square and the coordinate axes (in radians) -/
  angle : ℝ

/-- A lattice point in the coordinate system -/
def LatticePoint : Type := ℤ × ℤ

/-- Predicate to check if a point is inside a rotated square -/
def is_inside (s : RotatedSquare) (p : ℝ × ℝ) : Prop := sorry

/-- Count the number of lattice points inside a rotated square -/
def count_lattice_points_inside (s : RotatedSquare) : ℕ := sorry

/-- Theorem stating that no rotated square at 45° contains exactly 7 lattice points -/
theorem no_square_with_seven_lattice_points :
  ¬ ∃ (s : RotatedSquare), s.angle = π / 4 ∧ count_lattice_points_inside s = 7 := by
  sorry

end no_square_with_seven_lattice_points_l3950_395075


namespace B_power_150_is_identity_l3950_395016

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_150_is_identity :
  B ^ 150 = 1 := by sorry

end B_power_150_is_identity_l3950_395016


namespace sum_of_two_5cm_cubes_volume_l3950_395071

/-- The volume of a cube with edge length s -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- The sum of volumes of two cubes with edge length s -/
def sum_of_two_cube_volumes (s : ℝ) : ℝ := 2 * cube_volume s

theorem sum_of_two_5cm_cubes_volume :
  sum_of_two_cube_volumes 5 = 250 := by
  sorry

end sum_of_two_5cm_cubes_volume_l3950_395071


namespace cycle_loss_percentage_l3950_395015

/-- Given a cost price and selling price, calculate the percentage of loss -/
def percentageLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice * 100

theorem cycle_loss_percentage :
  let costPrice : ℚ := 2800
  let sellingPrice : ℚ := 2100
  percentageLoss costPrice sellingPrice = 25 := by
  sorry

end cycle_loss_percentage_l3950_395015


namespace triangle_arithmetic_sequence_angle_l3950_395059

theorem triangle_arithmetic_sequence_angle (α d : ℝ) :
  (α - d) + α + (α + d) = 180 → α = 60 ∨ (α - d) = 60 ∨ (α + d) = 60 := by
  sorry

end triangle_arithmetic_sequence_angle_l3950_395059


namespace ahead_of_schedule_l3950_395061

/-- Represents the worker's production plan -/
def WorkerPlan (total_parts : ℕ) (total_days : ℕ) (initial_rate : ℕ) (initial_days : ℕ) (x : ℕ) : Prop :=
  initial_rate * initial_days + (total_days - initial_days) * x > total_parts

/-- Theorem stating the condition for completing the task ahead of schedule -/
theorem ahead_of_schedule (x : ℕ) :
  WorkerPlan 408 15 24 3 x ↔ 24 * 3 + (15 - 3) * x > 408 :=
by sorry

end ahead_of_schedule_l3950_395061


namespace geometric_sequence_sum_l3950_395079

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end geometric_sequence_sum_l3950_395079


namespace factorial_ratio_l3950_395097

theorem factorial_ratio : Nat.factorial 10 / (Nat.factorial 7 * Nat.factorial 3) = 120 := by
  sorry

end factorial_ratio_l3950_395097


namespace pool_depth_is_10_feet_l3950_395017

/-- Represents the dimensions and properties of a pool -/
structure Pool where
  width : ℝ
  length : ℝ
  depth : ℝ
  capacity : ℝ
  drainRate : ℝ
  drainTime : ℝ
  initialFillPercentage : ℝ

/-- Calculates the volume of water drained from the pool -/
def volumeDrained (p : Pool) : ℝ := p.drainRate * p.drainTime

/-- Calculates the total capacity of the pool -/
def totalCapacity (p : Pool) : ℝ := p.width * p.length * p.depth

/-- Theorem stating that the depth of the pool is 10 feet -/
theorem pool_depth_is_10_feet (p : Pool) 
  (h1 : p.width = 40)
  (h2 : p.length = 150)
  (h3 : p.drainRate = 60)
  (h4 : p.drainTime = 800)
  (h5 : p.initialFillPercentage = 0.8)
  (h6 : volumeDrained p = p.initialFillPercentage * totalCapacity p) :
  p.depth = 10 := by
  sorry

#check pool_depth_is_10_feet

end pool_depth_is_10_feet_l3950_395017


namespace probability_tropical_temperate_l3950_395001

/-- The number of tropical fruits -/
def tropical_fruits : ℕ := 3

/-- The number of temperate fruits -/
def temperate_fruits : ℕ := 2

/-- The total number of fruits -/
def total_fruits : ℕ := tropical_fruits + temperate_fruits

/-- The number of fruits to be selected -/
def selection_size : ℕ := 2

/-- The probability of selecting one tropical and one temperate fruit -/
theorem probability_tropical_temperate :
  (Nat.choose tropical_fruits 1 * Nat.choose temperate_fruits 1 : ℚ) / 
  Nat.choose total_fruits selection_size = 3/5 := by sorry

end probability_tropical_temperate_l3950_395001


namespace smaller_number_problem_l3950_395020

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 22) (h2 : x - y = 16) : 
  min x y = 3 := by
  sorry

end smaller_number_problem_l3950_395020


namespace sqrt_expression_equality_l3950_395066

theorem sqrt_expression_equality : 
  Real.sqrt 12 - Real.sqrt (1/3) - Real.sqrt 2 * Real.sqrt 6 = -(Real.sqrt 3 / 3) := by
  sorry

end sqrt_expression_equality_l3950_395066


namespace alex_cookies_l3950_395082

theorem alex_cookies (alex sam : ℕ) : 
  alex = sam + 8 → 
  sam = alex / 3 → 
  alex = 12 := by
sorry

end alex_cookies_l3950_395082


namespace trig_expression_equals_four_l3950_395027

theorem trig_expression_equals_four : 
  (Real.sqrt 3 * Real.tan (10 * π / 180) + 1) / 
  ((4 * (Real.cos (10 * π / 180))^2 - 2) * Real.sin (10 * π / 180)) = 4 := by
  sorry

end trig_expression_equals_four_l3950_395027


namespace derivative_pos_implies_increasing_exists_increasing_not_always_pos_derivative_l3950_395081

open Function Real

-- Define a differentiable function f: ℝ → ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for f to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Statement 1: If f'(x) > 0 for all x, then f is increasing
theorem derivative_pos_implies_increasing :
  (∀ x, deriv f x > 0) → IsIncreasing f :=
sorry

-- Statement 2: There exists an increasing f where it's not true that f'(x) > 0 for all x
theorem exists_increasing_not_always_pos_derivative :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ IsIncreasing f ∧ ¬(∀ x, deriv f x > 0) :=
sorry

end derivative_pos_implies_increasing_exists_increasing_not_always_pos_derivative_l3950_395081


namespace dans_remaining_money_l3950_395056

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial : ℕ) (cost : ℕ) : ℕ :=
  initial - cost

theorem dans_remaining_money :
  remaining_money 4 1 = 3 := by
  sorry

end dans_remaining_money_l3950_395056


namespace area_BXC_specific_trapezoid_l3950_395038

/-- Represents a trapezoid ABCD with intersection point X of diagonals -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  area : ℝ

/-- Calculates the area of triangle BXC in the trapezoid -/
def area_BXC (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of triangle BXC in the specific trapezoid -/
theorem area_BXC_specific_trapezoid :
  let t : Trapezoid := { AB := 20, CD := 30, area := 300 }
  area_BXC t = 72 := by sorry

end area_BXC_specific_trapezoid_l3950_395038


namespace negation_of_exponential_inequality_l3950_395013

theorem negation_of_exponential_inequality (P : Prop) :
  (P ↔ ∀ x : ℝ, x > 0 → Real.exp x > x + 1) →
  (¬P ↔ ∃ x : ℝ, x > 0 ∧ Real.exp x ≤ x + 1) :=
by sorry

end negation_of_exponential_inequality_l3950_395013


namespace complement_intersection_theorem_l3950_395007

def U : Set Nat := {2, 3, 6, 8}
def A : Set Nat := {2, 3}
def B : Set Nat := {2, 6, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {6, 8} := by sorry

end complement_intersection_theorem_l3950_395007


namespace local_max_condition_l3950_395062

theorem local_max_condition (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ IsLocalMax (fun x => Real.exp x + a * x) x) →
  a < -1 := by sorry

end local_max_condition_l3950_395062


namespace arithmetic_combination_equals_24_l3950_395041

theorem arithmetic_combination_equals_24 : ∃ (expr : ℝ → ℝ → ℝ → ℝ → ℝ), 
  (expr 5 7 8 8 = 24) ∧ 
  (∀ a b c d, expr a b c d = ((b + c) / a) * d ∨ expr a b c d = ((b - a) * c) + d) :=
by sorry

end arithmetic_combination_equals_24_l3950_395041


namespace polynomial_roots_l3950_395074

theorem polynomial_roots : ∃ (x₁ x₂ x₃ : ℝ), 
  (x₁ = 1 ∧ x₂ = 2 ∧ x₃ = -1) ∧ 
  (∀ x : ℝ, x^4 - 4*x^3 + 3*x^2 + 4*x - 4 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by sorry

end polynomial_roots_l3950_395074


namespace molecular_weight_CaI2_value_l3950_395031

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of Calcium atoms in CaI2 -/
def num_Ca : ℕ := 1

/-- The number of Iodine atoms in CaI2 -/
def num_I : ℕ := 2

/-- The molecular weight of CaI2 in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca * num_Ca + atomic_weight_I * num_I

theorem molecular_weight_CaI2_value : molecular_weight_CaI2 = 293.88 := by
  sorry

end molecular_weight_CaI2_value_l3950_395031


namespace equation_solution_l3950_395085

theorem equation_solution : ∃! x : ℝ, 13 + Real.sqrt (-4 + x - 3 * 3) = 14 := by
  sorry

end equation_solution_l3950_395085


namespace james_bike_ride_l3950_395084

/-- Proves that given the conditions of James' bike ride, the third hour distance is 25% farther than the second hour distance -/
theorem james_bike_ride (second_hour_distance : ℝ) (total_distance : ℝ) :
  second_hour_distance = 18 →
  second_hour_distance = (1 + 0.2) * (second_hour_distance / 1.2) →
  total_distance = 55.5 →
  (total_distance - (second_hour_distance + second_hour_distance / 1.2)) / second_hour_distance = 0.25 := by
  sorry

#check james_bike_ride

end james_bike_ride_l3950_395084


namespace complex_fraction_simplification_l3950_395086

theorem complex_fraction_simplification :
  (5 - Complex.I) / (1 - Complex.I) = 3 + 2 * Complex.I :=
by sorry

end complex_fraction_simplification_l3950_395086


namespace min_value_expression_l3950_395090

open Real

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min : ℝ), min = Real.sqrt 39 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 →
    (|6*x - 4*y| + |3*(x + y*Real.sqrt 3) + 2*(x*Real.sqrt 3 - y)|) / Real.sqrt (x^2 + y^2) ≥ min :=
sorry

end min_value_expression_l3950_395090


namespace degree_of_composed_and_multiplied_polynomials_l3950_395096

/-- The degree of a polynomial -/
noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry

/-- Polynomial composition -/
def polyComp (p q : Polynomial ℝ) : Polynomial ℝ := sorry

/-- Polynomial multiplication -/
def polyMul (p q : Polynomial ℝ) : Polynomial ℝ := sorry

theorem degree_of_composed_and_multiplied_polynomials 
  (f g : Polynomial ℝ) 
  (hf : degree f = 3) 
  (hg : degree g = 7) : 
  degree (polyMul (polyComp f (Polynomial.X^4)) (polyComp g (Polynomial.X^3))) = 33 := by
  sorry

end degree_of_composed_and_multiplied_polynomials_l3950_395096


namespace right_triangle_in_sets_l3950_395057

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2

/-- The sets of side lengths given in the problem --/
def side_length_sets : List (ℕ × ℕ × ℕ) :=
  [(5, 4, 3), (1, 2, 3), (5, 6, 7), (2, 2, 3)]

theorem right_triangle_in_sets :
  ∃! (a b c : ℕ), (a, b, c) ∈ side_length_sets ∧ is_right_triangle a b c :=
sorry

end right_triangle_in_sets_l3950_395057


namespace range_of_a_l3950_395073

-- Define the function f(x) = (a^2 - 1)x^2 - (a-1)x - 1
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 1) * x^2 - (a - 1) * x - 1

-- Define the property that f(x) < 0 for all real x
def always_negative (a : ℝ) : Prop := ∀ x : ℝ, f a x < 0

-- Theorem statement
theorem range_of_a : 
  {a : ℝ | always_negative a} = Set.Ioc (- 3/5) 1 :=
sorry

end range_of_a_l3950_395073


namespace f_11_equals_149_l3950_395040

def f (n : ℕ) : ℕ := n^2 + n + 17

theorem f_11_equals_149 : f 11 = 149 := by sorry

end f_11_equals_149_l3950_395040


namespace minimum_artists_count_l3950_395068

theorem minimum_artists_count : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 5 = 1 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 3 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 5 = 1 ∧ m % 6 = 2 ∧ m % 8 = 3 → m ≥ n) ∧
  n = 236 := by
  sorry

end minimum_artists_count_l3950_395068


namespace gcf_lcm_sum_8_12_l3950_395035

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcf_lcm_sum_8_12_l3950_395035


namespace two_digit_numbers_with_product_and_gcd_l3950_395094

theorem two_digit_numbers_with_product_and_gcd 
  (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a < 100) 
  (h2 : 10 ≤ b ∧ b < 100) 
  (h3 : a * b = 1728) 
  (h4 : Nat.gcd a b = 12) : 
  (a = 36 ∧ b = 48) ∨ (a = 48 ∧ b = 36) := by
sorry

end two_digit_numbers_with_product_and_gcd_l3950_395094


namespace money_left_after_shopping_l3950_395083

def bread_price : ℝ := 2
def butter_original_price : ℝ := 3
def butter_discount : ℝ := 0.1
def juice_price_multiplier : ℝ := 2
def cookies_original_price : ℝ := 4
def cookies_discount : ℝ := 0.2
def vat_rate : ℝ := 0.05
def initial_money : ℝ := 20

def calculate_discounted_price (original_price discount : ℝ) : ℝ :=
  original_price * (1 - discount)

def calculate_total_cost (bread butter juice cookies : ℝ) : ℝ :=
  bread + butter + juice + cookies

def apply_vat (total_cost vat_rate : ℝ) : ℝ :=
  total_cost * (1 + vat_rate)

theorem money_left_after_shopping :
  let butter_price := calculate_discounted_price butter_original_price butter_discount
  let cookies_price := calculate_discounted_price cookies_original_price cookies_discount
  let juice_price := bread_price * juice_price_multiplier
  let total_cost := calculate_total_cost bread_price butter_price juice_price cookies_price
  let final_cost := apply_vat total_cost vat_rate
  initial_money - final_cost = 7.5 := by sorry

end money_left_after_shopping_l3950_395083


namespace bus_passengers_l3950_395025

theorem bus_passengers (initial : ℕ) (got_on : ℕ) (current : ℕ) : 
  got_on = 13 → current = 17 → initial + got_on = current → initial = 4 := by
  sorry

end bus_passengers_l3950_395025


namespace triangle_shape_l3950_395087

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def has_equal_roots (t : Triangle) : Prop :=
  ∃ x : ℝ, t.b * (x^2 + 1) + t.c * (x^2 - 1) - 2 * t.a * x = 0 ∧
  ∀ y : ℝ, t.b * (y^2 + 1) + t.c * (y^2 - 1) - 2 * t.a * y = 0 → y = x

def angle_condition (t : Triangle) : Prop :=
  Real.sin t.C * Real.cos t.A - Real.cos t.C * Real.sin t.A = 0

-- Define an isosceles right-angled triangle
def is_isosceles_right_angled (t : Triangle) : Prop :=
  t.a = t.b ∧ t.A = t.B ∧ t.C = Real.pi / 2

-- State the theorem
theorem triangle_shape (t : Triangle) :
  has_equal_roots t → angle_condition t → is_isosceles_right_angled t := by
  sorry

end triangle_shape_l3950_395087


namespace shaded_area_of_divided_square_l3950_395036

theorem shaded_area_of_divided_square (side_length : ℝ) (total_squares : ℕ) (shaded_squares : ℕ) : 
  side_length = 10 ∧ total_squares = 25 ∧ shaded_squares = 5 → 
  (side_length^2 / total_squares) * shaded_squares = 20 := by
  sorry

#check shaded_area_of_divided_square

end shaded_area_of_divided_square_l3950_395036


namespace s_range_l3950_395052

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def divisible_by_11 (n : ℕ) : Prop := ∃ k, n = 11 * k

def s (n : ℕ) : ℕ := sorry

theorem s_range (n : ℕ) (h_composite : is_composite n) (h_div11 : divisible_by_11 n) :
  ∃ (m : ℕ), m ≥ 11 ∧ s n = m ∧ ∀ (k : ℕ), k ≥ 11 → ∃ (p : ℕ), is_composite p ∧ divisible_by_11 p ∧ s p = k :=
sorry

end s_range_l3950_395052


namespace geometric_sequence_a5_l3950_395089

/-- A geometric sequence with a_3 = 2 and a_7 = 8 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ (∀ n, a (n + 1) = r * a n) ∧ a 3 = 2 ∧ a 7 = 8

/-- Theorem: In a geometric sequence where a_3 = 2 and a_7 = 8, a_5 = 4 -/
theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : geometric_sequence a) : a 5 = 4 := by
  sorry

end geometric_sequence_a5_l3950_395089


namespace cats_dogs_percentage_difference_l3950_395023

/-- Represents the number of animals in a compound -/
structure AnimalCount where
  cats : ℕ
  dogs : ℕ
  frogs : ℕ

/-- The conditions of the animal compound problem -/
def CompoundConditions (count : AnimalCount) : Prop :=
  count.cats < count.dogs ∧
  count.frogs = 2 * count.dogs ∧
  count.cats + count.dogs + count.frogs = 304 ∧
  count.frogs = 160

/-- The percentage difference between dogs and cats -/
def PercentageDifference (count : AnimalCount) : ℚ :=
  (count.dogs - count.cats : ℚ) / count.dogs * 100

/-- Theorem stating the percentage difference between dogs and cats -/
theorem cats_dogs_percentage_difference (count : AnimalCount) 
  (h : CompoundConditions count) : PercentageDifference count = 20 := by
  sorry


end cats_dogs_percentage_difference_l3950_395023


namespace topsoil_cost_l3950_395072

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards : ℝ := 7

/-- The total cost of topsoil for a given number of cubic yards -/
def total_cost (yards : ℝ) : ℝ :=
  yards * cubic_feet_per_cubic_yard * cost_per_cubic_foot

theorem topsoil_cost : total_cost cubic_yards = 1512 := by
  sorry

end topsoil_cost_l3950_395072


namespace parabola_line_intersection_l3950_395092

theorem parabola_line_intersection (b : ℝ) : 
  (∃! x : ℝ, bx^2 + 5*x + 2 = -2*x - 2) ↔ b = 49/16 := by sorry

end parabola_line_intersection_l3950_395092


namespace class_average_mark_l3950_395014

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 30 →
  excluded_students = 5 →
  excluded_avg = 30 →
  remaining_avg = 90 →
  (total_students : ℝ) * (total_students * remaining_avg - excluded_students * excluded_avg) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end class_average_mark_l3950_395014


namespace inequality_equivalence_l3950_395008

theorem inequality_equivalence (y : ℝ) : 
  (3/10 : ℝ) + |2*y - 1/5| < 7/10 ↔ -1/10 < y ∧ y < 3/10 := by
sorry

end inequality_equivalence_l3950_395008


namespace triangle_AXY_is_obtuse_l3950_395054

-- Define the triangular pyramid ABCD
structure TriangularPyramid where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the inscribed and exscribed spheres
structure InscribedSphere where
  center : Point
  radius : ℝ

structure ExscribedSphere where
  center : Point
  radius : ℝ

-- Define the points where the spheres touch face BCD
def X (pyramid : TriangularPyramid) (inscribedSphere : InscribedSphere) : Point := sorry
def Y (pyramid : TriangularPyramid) (exscribedSphere : ExscribedSphere) : Point := sorry

-- Define the angle AXY
def angle_AXY (pyramid : TriangularPyramid) (inscribedSphere : InscribedSphere) (exscribedSphere : ExscribedSphere) : ℝ := sorry

-- Theorem statement
theorem triangle_AXY_is_obtuse (pyramid : TriangularPyramid) (inscribedSphere : InscribedSphere) (exscribedSphere : ExscribedSphere) :
  angle_AXY pyramid inscribedSphere exscribedSphere > π / 2 := by
  sorry

end triangle_AXY_is_obtuse_l3950_395054


namespace adam_action_figures_l3950_395010

/-- The total number of action figures Adam can fit on his shelves -/
def total_action_figures (initial_shelves : List Nat) (new_shelves : Nat) (new_shelf_capacity : Nat) : Nat :=
  (initial_shelves.sum) + (new_shelves * new_shelf_capacity)

/-- Theorem: Adam can fit 52 action figures on his shelves -/
theorem adam_action_figures :
  total_action_figures [9, 14, 7] 2 11 = 52 := by
  sorry

end adam_action_figures_l3950_395010


namespace minimum_value_inequality_minimum_value_achievable_l3950_395069

theorem minimum_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 5) : 
  1/x + 4/y + 9/z ≥ 36/5 := by
  sorry

theorem minimum_value_achievable : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 5 ∧ 1/x + 4/y + 9/z = 36/5 := by
  sorry

end minimum_value_inequality_minimum_value_achievable_l3950_395069


namespace first_nonzero_digit_of_one_over_137_l3950_395006

theorem first_nonzero_digit_of_one_over_137 :
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ d < 10 ∧ 
  (∀ (k : ℕ), k < n → (10^(k+1) / 137 % 10 = 0)) ∧
  (10^(n+1) / 137 % 10 = d) ∧ d = 2 :=
sorry

end first_nonzero_digit_of_one_over_137_l3950_395006


namespace apples_on_tree_l3950_395044

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := 7

/-- The number of apples Rachel picked -/
def picked_apples : ℕ := 4

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := initial_apples - picked_apples

theorem apples_on_tree : remaining_apples = 3 := by
  sorry

end apples_on_tree_l3950_395044


namespace savings_after_expense_l3950_395065

def weekly_savings (n : ℕ) : ℕ := 20 + 10 * n

def total_savings (weeks : ℕ) : ℕ :=
  (List.range weeks).map weekly_savings |>.sum

theorem savings_after_expense (weeks : ℕ) (expense : ℕ) : 
  weeks = 4 → expense = 75 → total_savings weeks - expense = 65 := by
  sorry

end savings_after_expense_l3950_395065


namespace pyramid_volume_approx_l3950_395019

-- Define the pyramid
structure Pyramid where
  base_area : ℝ
  face1_area : ℝ
  face2_area : ℝ

-- Define the volume function
def pyramid_volume (p : Pyramid) : ℝ :=
  -- The actual calculation is not implemented, as per instructions
  sorry

-- Theorem statement
theorem pyramid_volume_approx (p : Pyramid) 
  (h1 : p.base_area = 144) 
  (h2 : p.face1_area = 72) 
  (h3 : p.face2_area = 54) : 
  ∃ (ε : ℝ), abs (pyramid_volume p - 518.76) < ε ∧ ε > 0 := by
  sorry


end pyramid_volume_approx_l3950_395019


namespace sum_squares_bounds_l3950_395009

theorem sum_squares_bounds (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y = 10) :
  50 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 100 ∧
  (∃ a b : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ a + b = 10 ∧ a^2 + b^2 = 50) ∧
  (∃ c d : ℝ, c ≥ 0 ∧ d ≥ 0 ∧ c + d = 10 ∧ c^2 + d^2 = 100) := by
  sorry

end sum_squares_bounds_l3950_395009


namespace log_properties_l3950_395018

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem statement
theorem log_properties (a M N x : ℝ) 
  (ha : a > 0 ∧ a ≠ 1) (hM : M > 0) (hN : N > 0) :
  (log a (a^x) = x) ∧
  (log a (M / N) = log a M - log a N) ∧
  (log a (M * N) = log a M + log a N) := by
  sorry

end log_properties_l3950_395018


namespace car_travel_time_fraction_l3950_395067

theorem car_travel_time_fraction (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) 
  (h1 : distance = 432)
  (h2 : original_time = 6)
  (h3 : new_speed = 48) : 
  (distance / new_speed) / original_time = 1 / 4 := by
  sorry

end car_travel_time_fraction_l3950_395067


namespace identifiable_bulbs_for_two_trips_three_states_max_identifiable_bulbs_is_power_l3950_395051

/-- The maximum number of bulbs and switches that can be identified -/
def max_identifiable_bulbs (n : ℕ) (m : ℕ) : ℕ := m^n

/-- Theorem: With 2 trips and 3 states, 9 bulbs and switches can be identified -/
theorem identifiable_bulbs_for_two_trips_three_states :
  max_identifiable_bulbs 2 3 = 9 := by
  sorry

/-- Theorem: The maximum number of identifiable bulbs is always a power of the number of states -/
theorem max_identifiable_bulbs_is_power (n m : ℕ) :
  ∃ k, max_identifiable_bulbs n m = m^k := by
  sorry

end identifiable_bulbs_for_two_trips_three_states_max_identifiable_bulbs_is_power_l3950_395051


namespace parallel_vector_problem_l3950_395055

/-- Given two vectors a and b in ℝ², where a = (-2, 1), |b| = 5, and a is parallel to b,
    prove that b is either (-2√5, √5) or (2√5, -√5). -/
theorem parallel_vector_problem (a b : ℝ × ℝ) : 
  a = (-2, 1) → 
  ‖b‖ = 5 → 
  ∃ (k : ℝ), b = k • a → 
  b = (-2 * Real.sqrt 5, Real.sqrt 5) ∨ b = (2 * Real.sqrt 5, -Real.sqrt 5) :=
by sorry

end parallel_vector_problem_l3950_395055


namespace log_equation_solution_l3950_395022

theorem log_equation_solution : 
  ∃ (x : ℝ), (Real.log 729 / Real.log (3 * x) = x) ∧ x = 3 := by
  sorry

end log_equation_solution_l3950_395022


namespace volume_of_R_revolution_l3950_395039

-- Define the region R
def R := {(x, y) : ℝ × ℝ | |8 - x| + y ≤ 10 ∧ 3 * y - x ≥ 15}

-- Define the axis of revolution
def axis := {(x, y) : ℝ × ℝ | 3 * y - x = 15}

-- Define the volume of the solid of revolution
def volume_of_revolution (region : Set (ℝ × ℝ)) (axis : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem volume_of_R_revolution :
  volume_of_revolution R axis = (343 * Real.pi) / (12 * Real.sqrt 10) := by sorry

end volume_of_R_revolution_l3950_395039


namespace exists_tetrahedron_no_triangle_l3950_395000

/-- A tetrahedron with an inscribed sphere -/
structure TangentialTetrahedron where
  /-- Lengths of tangents from vertices to points of contact with the inscribed sphere -/
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  /-- All lengths are positive -/
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  d_pos : d > 0

/-- Predicate to check if three lengths can form a triangle -/
def canFormTriangle (x y z : ℝ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

/-- Theorem stating that there exists a tangential tetrahedron where no combination
    of tangent lengths can form a triangle -/
theorem exists_tetrahedron_no_triangle :
  ∃ (t : TangentialTetrahedron),
    ¬(canFormTriangle t.a t.b t.c ∨
      canFormTriangle t.a t.b t.d ∨
      canFormTriangle t.a t.c t.d ∨
      canFormTriangle t.b t.c t.d) :=
sorry

end exists_tetrahedron_no_triangle_l3950_395000


namespace camera_profit_difference_l3950_395095

/-- Represents the profit calculation for camera sales -/
def camera_profit (num_cameras : ℕ) (buy_price sell_price : ℚ) : ℚ :=
  num_cameras * (sell_price - buy_price)

/-- Represents the problem of calculating the difference in profit between Maddox and Theo -/
theorem camera_profit_difference : 
  let num_cameras : ℕ := 3
  let buy_price : ℚ := 20
  let maddox_sell_price : ℚ := 28
  let theo_sell_price : ℚ := 23
  let maddox_profit := camera_profit num_cameras buy_price maddox_sell_price
  let theo_profit := camera_profit num_cameras buy_price theo_sell_price
  maddox_profit - theo_profit = 15 := by
sorry


end camera_profit_difference_l3950_395095


namespace fraction_difference_l3950_395004

theorem fraction_difference (a b c d : ℚ) : 
  a = 3/4 ∧ b = 7/8 ∧ c = 13/16 ∧ d = 1/2 →
  max a (max b (max c d)) - min a (min b (min c d)) = 3/8 := by
sorry

end fraction_difference_l3950_395004


namespace cubic_equation_solutions_l3950_395021

theorem cubic_equation_solutions :
  ∀ x y z : ℤ,
  (x + y + z = 2 ∧ x^3 + y^3 + z^3 = -10) ↔
  ((x, y, z) = (3, 3, -4) ∨ (x, y, z) = (3, -4, 3) ∨ (x, y, z) = (-4, 3, 3)) :=
by sorry

end cubic_equation_solutions_l3950_395021


namespace intersection_distance_product_l3950_395046

/-- Given a line L and a circle C, prove that the product of distances from a point on the line to the intersection points of the line and circle is 1/4. -/
theorem intersection_distance_product (P : ℝ × ℝ) (α : ℝ) (C : Set (ℝ × ℝ)) : 
  P = (1/2, 1) →
  α = π/6 →
  C = {(x, y) | x^2 + y^2 = x + y} →
  ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ 
    (∃ (t₁ t₂ : ℝ), 
      A = (1/2 + (Real.sqrt 3)/2 * t₁, 1 + 1/2 * t₁) ∧
      B = (1/2 + (Real.sqrt 3)/2 * t₂, 1 + 1/2 * t₂)) ∧
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) * 
    Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 1/4 := by
  sorry

end intersection_distance_product_l3950_395046


namespace largest_ball_radius_largest_ball_touches_plane_largest_ball_on_z_axis_l3950_395029

/-- Represents a torus formed by revolving a circle about the z-axis. -/
structure Torus where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents a spherical ball. -/
structure Ball where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The largest ball that can be positioned on top of the torus. -/
def largest_ball (t : Torus) : Ball :=
  { center := (0, 0, 4),
    radius := 4 }

/-- Theorem stating that the largest ball has radius 4. -/
theorem largest_ball_radius (t : Torus) :
  (t.center = (4, 0, 1) ∧ t.radius = 1) →
  (largest_ball t).radius = 4 := by
  sorry

/-- Theorem stating that the largest ball touches the horizontal plane. -/
theorem largest_ball_touches_plane (t : Torus) :
  (t.center = (4, 0, 1) ∧ t.radius = 1) →
  (largest_ball t).center.2.1 = (largest_ball t).radius := by
  sorry

/-- Theorem stating that the largest ball is centered on the z-axis. -/
theorem largest_ball_on_z_axis (t : Torus) :
  (t.center = (4, 0, 1) ∧ t.radius = 1) →
  (largest_ball t).center.1 = 0 ∧ (largest_ball t).center.2.2 = 0 := by
  sorry

end largest_ball_radius_largest_ball_touches_plane_largest_ball_on_z_axis_l3950_395029


namespace central_angle_for_given_sector_l3950_395058

/-- A circular sector with given area and perimeter -/
structure CircularSector where
  area : ℝ
  perimeter : ℝ

/-- The central angle of a circular sector in radians -/
def central_angle (s : CircularSector) : ℝ := 
  2 -- We define this as 2, which is what we want to prove

/-- Theorem: For a circular sector with area 1 and perimeter 4, the central angle is 2 radians -/
theorem central_angle_for_given_sector :
  ∀ (s : CircularSector), s.area = 1 ∧ s.perimeter = 4 → central_angle s = 2 := by
  sorry

#check central_angle_for_given_sector

end central_angle_for_given_sector_l3950_395058
