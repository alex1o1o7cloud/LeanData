import Mathlib

namespace NUMINAMATH_CALUDE_initial_investment_rate_l1897_189783

-- Define the initial investment
def initial_investment : ℝ := 1400

-- Define the additional investment
def additional_investment : ℝ := 700

-- Define the interest rate of the additional investment
def additional_rate : ℝ := 0.08

-- Define the total investment
def total_investment : ℝ := initial_investment + additional_investment

-- Define the desired total annual income rate
def total_income_rate : ℝ := 0.06

-- Define the function that calculates the total annual income
def total_annual_income (r : ℝ) : ℝ := 
  initial_investment * r + additional_investment * additional_rate

-- Theorem statement
theorem initial_investment_rate : 
  ∃ r : ℝ, total_annual_income r = total_income_rate * total_investment ∧ r = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_initial_investment_rate_l1897_189783


namespace NUMINAMATH_CALUDE_rational_root_of_cubic_l1897_189773

/-- Given a cubic polynomial with rational coefficients, if 3 + √5 is a root
    and another root is rational, then the rational root is -6 -/
theorem rational_root_of_cubic (a b c : ℚ) :
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 3 + Real.sqrt 5) →
  (∃ r : ℚ, r^3 + a*r^2 + b*r + c = 0) →
  (∃ r : ℚ, r^3 + a*r^2 + b*r + c = 0 ∧ r = -6) :=
by sorry

end NUMINAMATH_CALUDE_rational_root_of_cubic_l1897_189773


namespace NUMINAMATH_CALUDE_satellite_has_24_units_l1897_189755

/-- Represents a satellite with modular units and sensors. -/
structure Satellite where
  units : ℕ
  non_upgraded_per_unit : ℕ
  total_upgraded : ℕ

/-- The conditions of the satellite problem. -/
def satellite_conditions (s : Satellite) : Prop :=
  -- Condition 2: non-upgraded sensors per unit is 1/6 of total upgraded
  s.non_upgraded_per_unit = s.total_upgraded / 6 ∧
  -- Condition 3: 20% of all sensors are upgraded
  s.total_upgraded = (s.total_upgraded + s.units * s.non_upgraded_per_unit) / 5

/-- The theorem stating that a satellite satisfying the given conditions has 24 units. -/
theorem satellite_has_24_units (s : Satellite) (h : satellite_conditions s) : s.units = 24 := by
  sorry


end NUMINAMATH_CALUDE_satellite_has_24_units_l1897_189755


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l1897_189788

theorem wrapping_paper_fraction (total_fraction : Rat) (num_presents : Nat) 
  (h1 : total_fraction = 5 / 12)
  (h2 : num_presents = 4) :
  total_fraction / num_presents = 5 / 48 := by
sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l1897_189788


namespace NUMINAMATH_CALUDE_man_wage_is_350_l1897_189780

/-- The daily wage of a man -/
def man_wage : ℝ := 350

/-- The daily wage of a woman -/
def woman_wage : ℝ := 200

/-- The total number of men -/
def num_men : ℕ := 24

/-- The total number of women -/
def num_women : ℕ := 16

/-- The total daily wages -/
def total_wages : ℝ := 11600

theorem man_wage_is_350 :
  (num_men * man_wage + num_women * woman_wage = total_wages) ∧
  ((num_men / 2) * man_wage + 37 * woman_wage = total_wages) →
  man_wage = 350 := by
  sorry

end NUMINAMATH_CALUDE_man_wage_is_350_l1897_189780


namespace NUMINAMATH_CALUDE_polygon_angle_ratio_l1897_189767

theorem polygon_angle_ratio (n : ℕ) : 
  (((n - 2) * 180) / 360 : ℚ) = 9/2 ↔ n = 11 := by sorry

end NUMINAMATH_CALUDE_polygon_angle_ratio_l1897_189767


namespace NUMINAMATH_CALUDE_base_ten_to_four_156_base_four_to_ten_2130_l1897_189720

/-- Converts a natural number from base 10 to base 4 --/
def toBaseFour (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Converts a list of digits in base 4 to a natural number in base 10 --/
def fromBaseFour (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ (digits.length - 1 - i))) 0

theorem base_ten_to_four_156 : toBaseFour 156 = [2, 1, 3, 0] := by sorry

theorem base_four_to_ten_2130 : fromBaseFour [2, 1, 3, 0] = 156 := by sorry

end NUMINAMATH_CALUDE_base_ten_to_four_156_base_four_to_ten_2130_l1897_189720


namespace NUMINAMATH_CALUDE_max_value_ad_minus_bc_l1897_189772

theorem max_value_ad_minus_bc :
  ∀ a b c d : ℤ,
  a ∈ ({-1, 1, 2} : Set ℤ) →
  b ∈ ({-1, 1, 2} : Set ℤ) →
  c ∈ ({-1, 1, 2} : Set ℤ) →
  d ∈ ({-1, 1, 2} : Set ℤ) →
  (∀ x y z w : ℤ,
    x ∈ ({-1, 1, 2} : Set ℤ) →
    y ∈ ({-1, 1, 2} : Set ℤ) →
    z ∈ ({-1, 1, 2} : Set ℤ) →
    w ∈ ({-1, 1, 2} : Set ℤ) →
    x * w - y * z ≤ 6) ∧
  (∃ x y z w : ℤ,
    x ∈ ({-1, 1, 2} : Set ℤ) ∧
    y ∈ ({-1, 1, 2} : Set ℤ) ∧
    z ∈ ({-1, 1, 2} : Set ℤ) ∧
    w ∈ ({-1, 1, 2} : Set ℤ) ∧
    x * w - y * z = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_value_ad_minus_bc_l1897_189772


namespace NUMINAMATH_CALUDE_geometric_sequence_property_geometric_sequence_sum_l1897_189710

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m

/-- The property that if m + n = p + q, then a_m * a_n = a_p * a_q for a geometric sequence -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q :=
sorry

theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) 
  (h_sum : a 4 + a 8 = -3) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_geometric_sequence_sum_l1897_189710


namespace NUMINAMATH_CALUDE_simplify_expression_l1897_189735

theorem simplify_expression : 3 * (((1 + 2 + 3 + 4) * 3) + ((1 * 4 + 16) / 4)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1897_189735


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l1897_189743

/-- A quadratic polynomial -/
def QuadraticPolynomial (R : Type*) [Field R] := R → R

/-- Property that p(n) = 1/n^2 for n = 1, 2, 3 -/
def SatisfiesCondition (p : QuadraticPolynomial ℝ) : Prop :=
  p 1 = 1 ∧ p 2 = 1/4 ∧ p 3 = 1/9

theorem quadratic_polynomial_property (p : QuadraticPolynomial ℝ) 
  (h : SatisfiesCondition p) : p 4 = -9/16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l1897_189743


namespace NUMINAMATH_CALUDE_solution_proof_l1897_189701

-- Part 1: System of equations
def satisfies_system (x y : ℝ) : Prop :=
  2 * x - y = 5 ∧ 3 * x + 4 * y = 2

-- Part 2: System of inequalities
def satisfies_inequalities (x : ℝ) : Prop :=
  -2 * x < 6 ∧ 3 * (x - 2) ≤ x - 4

-- Part 3: Integer solutions
def is_integer_solution (x : ℤ) : Prop :=
  -3 < (x : ℝ) ∧ (x : ℝ) ≤ 1

theorem solution_proof :
  -- Part 1
  satisfies_system 2 (-1) ∧
  -- Part 2
  (∀ x : ℝ, satisfies_inequalities x ↔ -3 < x ∧ x ≤ 1) ∧
  -- Part 3
  (∀ x : ℤ, is_integer_solution x ↔ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_proof_l1897_189701


namespace NUMINAMATH_CALUDE_songs_to_learn_l1897_189708

/-- Given that Billy can play 24 songs and his music book contains 52 songs,
    prove that the number of songs he still needs to learn is 28. -/
theorem songs_to_learn (songs_can_play : ℕ) (total_songs : ℕ) 
  (h1 : songs_can_play = 24) (h2 : total_songs = 52) : 
  total_songs - songs_can_play = 28 := by
  sorry

end NUMINAMATH_CALUDE_songs_to_learn_l1897_189708


namespace NUMINAMATH_CALUDE_sum_of_integers_l1897_189785

theorem sum_of_integers (s l : ℤ) : 
  s = 10 → 2 * l = 5 * s - 10 → s + l = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1897_189785


namespace NUMINAMATH_CALUDE_flip_colors_iff_even_l1897_189771

/-- Represents the color of a square on the board -/
inductive Color
| White
| Black
| Orange

/-- Represents a 3n × 3n board -/
def Board (n : ℕ) := Fin (3*n) → Fin (3*n) → Color

/-- Initial coloring of the board -/
def initialBoard (n : ℕ) : Board n :=
  λ i j => if (i.val + j.val) % 3 = 2 then Color.Black else Color.White

/-- A move on the board -/
def move (b : Board n) (i j : Fin (3*n)) : Board n :=
  λ x y => if x.val ∈ [i.val, i.val+1] ∧ y.val ∈ [j.val, j.val+1]
           then match b x y with
                | Color.White => Color.Orange
                | Color.Orange => Color.Black
                | Color.Black => Color.White
           else b x y

/-- The goal state of the board -/
def goalBoard (n : ℕ) : Board n :=
  λ i j => if (i.val + j.val) % 3 = 2 then Color.White else Color.Black

/-- A sequence of moves -/
def MoveSequence (n : ℕ) := List (Fin (3*n) × Fin (3*n))

/-- Apply a sequence of moves to a board -/
def applyMoves (b : Board n) (moves : MoveSequence n) : Board n :=
  moves.foldl (λ board (i, j) => move board i j) b

theorem flip_colors_iff_even (n : ℕ) (h : n > 0) :
  (∃ (moves : MoveSequence n), applyMoves (initialBoard n) moves = goalBoard n) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_flip_colors_iff_even_l1897_189771


namespace NUMINAMATH_CALUDE_pauls_erasers_l1897_189760

/-- The number of erasers Paul got for his birthday -/
def erasers : ℕ := 0  -- We'll prove this is actually 457

/-- The number of crayons Paul got for his birthday -/
def initial_crayons : ℕ := 617

/-- The number of crayons Paul had left at the end of the school year -/
def remaining_crayons : ℕ := 523

/-- The difference between the number of crayons and erasers left -/
def crayon_eraser_difference : ℕ := 66

theorem pauls_erasers : 
  erasers = 457 ∧ 
  initial_crayons = 617 ∧
  remaining_crayons = 523 ∧
  crayon_eraser_difference = 66 ∧
  remaining_crayons = erasers + crayon_eraser_difference :=
sorry

end NUMINAMATH_CALUDE_pauls_erasers_l1897_189760


namespace NUMINAMATH_CALUDE_rabbits_distance_specific_rabbits_distance_l1897_189784

/-- The distance between two rabbits' homes given their resting patterns --/
theorem rabbits_distance (white_rest_interval : ℕ) (gray_rest_interval : ℕ) 
  (rest_difference : ℕ) : ℕ :=
  let meeting_point := white_rest_interval * gray_rest_interval * rest_difference / 
    (white_rest_interval - gray_rest_interval)
  2 * meeting_point

/-- Proof of the specific rabbit problem --/
theorem specific_rabbits_distance : 
  rabbits_distance 30 20 15 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_distance_specific_rabbits_distance_l1897_189784


namespace NUMINAMATH_CALUDE_f_monotonic_increasing_interval_l1897_189728

-- Define the function f(x) = -x^2 + 1
def f (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem f_monotonic_increasing_interval :
  ∀ x y, x < y ∧ y ≤ 0 → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_f_monotonic_increasing_interval_l1897_189728


namespace NUMINAMATH_CALUDE_unique_solution_when_a_is_one_l1897_189722

-- Define the equation
def equation (a x : ℝ) : Prop :=
  (5 : ℝ) ^ (x^2 + 2*a*x + a^2) = a*x^2 + 2*a^2*x + a^3 + a^2 - 6*a + 6

-- Theorem statement
theorem unique_solution_when_a_is_one :
  ∃! a : ℝ, ∃! x : ℝ, equation a x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_when_a_is_one_l1897_189722


namespace NUMINAMATH_CALUDE_solution_set_correct_range_of_b_l1897_189730

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + 2

-- Define the solution set for f(x) > 0
def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 then Set.Ioo (1/a) 2
  else if a = 0 then Set.Iio 2
  else if 0 < a ∧ a < 1/2 then Set.Iio 2 ∪ Set.Ioi (1/a)
  else if a = 1/2 then Set.Iio 2 ∪ Set.Ioi 2
  else Set.Iio (1/a) ∪ Set.Ioi 2

-- State the theorem for the solution set
theorem solution_set_correct (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ f a x > 0 :=
sorry

-- State the theorem for the range of b
theorem range_of_b :
  ∀ x ∈ Set.Icc (1/3) 1,
  ∀ m ∈ Set.Icc 1 4,
  f 1 (1/x) + (3 - 2*m)/x ≤ b^2 - 2*b - 2 →
  b ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_correct_range_of_b_l1897_189730


namespace NUMINAMATH_CALUDE_parabola_equation_l1897_189792

/-- The standard equation of a parabola with vertex (0,0) and focus (3,0) -/
theorem parabola_equation (x y : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ y^2 = 4 * p * x ∧ 3 = p) → y^2 = 12 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1897_189792


namespace NUMINAMATH_CALUDE_little_twelve_games_l1897_189751

/-- Represents a basketball conference with two divisions -/
structure BasketballConference :=
  (teams_per_division : ℕ)
  (divisions : ℕ)
  (intra_division_games : ℕ)
  (inter_division_games : ℕ)

/-- Calculates the total number of games in the conference -/
def total_games (conf : BasketballConference) : ℕ :=
  let total_teams := conf.teams_per_division * conf.divisions
  let games_per_team := (conf.teams_per_division - 1) * conf.intra_division_games + 
                        conf.teams_per_division * conf.inter_division_games
  total_teams * games_per_team / 2

/-- Theorem stating that the Little Twelve Basketball Conference schedules 96 games -/
theorem little_twelve_games : 
  ∀ (conf : BasketballConference), 
    conf.teams_per_division = 6 ∧ 
    conf.divisions = 2 ∧ 
    conf.intra_division_games = 2 ∧ 
    conf.inter_division_games = 1 → 
    total_games conf = 96 := by
  sorry

end NUMINAMATH_CALUDE_little_twelve_games_l1897_189751


namespace NUMINAMATH_CALUDE_suresh_job_time_l1897_189702

/-- The time it takes Ashutosh to complete the job alone (in hours) -/
def ashutosh_time : ℝ := 15

/-- The time Suresh works on the job (in hours) -/
def suresh_work_time : ℝ := 9

/-- The time Ashutosh works to complete the remaining job (in hours) -/
def ashutosh_remaining_time : ℝ := 6

/-- The time it takes Suresh to complete the job alone (in hours) -/
def suresh_time : ℝ := 15

theorem suresh_job_time :
  suresh_time * (1 / ashutosh_time * ashutosh_remaining_time + 1 / suresh_time * suresh_work_time) = suresh_time := by
  sorry

#check suresh_job_time

end NUMINAMATH_CALUDE_suresh_job_time_l1897_189702


namespace NUMINAMATH_CALUDE_gym_membership_cost_l1897_189791

/-- Calculates the total cost of gym memberships for the first year -/
theorem gym_membership_cost (cheap_monthly_fee : ℕ) (cheap_signup_fee : ℕ) 
  (expensive_monthly_multiplier : ℕ) (expensive_signup_months : ℕ) (months_per_year : ℕ) : 
  cheap_monthly_fee = 10 →
  cheap_signup_fee = 50 →
  expensive_monthly_multiplier = 3 →
  expensive_signup_months = 4 →
  months_per_year = 12 →
  (cheap_monthly_fee * months_per_year + cheap_signup_fee) + 
  (cheap_monthly_fee * expensive_monthly_multiplier * months_per_year + 
   cheap_monthly_fee * expensive_monthly_multiplier * expensive_signup_months) = 650 := by
  sorry

#check gym_membership_cost

end NUMINAMATH_CALUDE_gym_membership_cost_l1897_189791


namespace NUMINAMATH_CALUDE_intercept_length_min_distance_l1897_189734

-- Define the family of curves C
def C (m : ℝ) (x y : ℝ) : Prop :=
  4 * x^2 + 5 * y^2 - 8 * m * x - 20 * m * y + 24 * m^2 - 20 = 0

-- Define the circle
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * x + 4 * Real.sqrt 6 * y + 30 = 0

-- Define the lines
def Line1 (x y : ℝ) : Prop := y = 2 * x + 2
def Line2 (x y : ℝ) : Prop := y = 2 * x - 2

-- Theorem for part 1
theorem intercept_length (m : ℝ) :
  ∀ x y, C m x y → (Line1 x y ∨ Line2 x y) →
  ∃ x1 y1 x2 y2, C m x1 y1 ∧ C m x2 y2 ∧
  ((Line1 x1 y1 ∧ Line1 x2 y2) ∨ (Line2 x1 y1 ∧ Line2 x2 y2)) ∧
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 5 * Real.sqrt 5 / 3 :=
sorry

-- Theorem for part 2
theorem min_distance :
  ∀ m x1 y1 x2 y2, C m x1 y1 → Circle x2 y2 →
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) ≥ 2 * Real.sqrt 5 - 1 :=
sorry

end NUMINAMATH_CALUDE_intercept_length_min_distance_l1897_189734


namespace NUMINAMATH_CALUDE_distance_circle_C_to_line_l_l1897_189770

/-- Circle C with center (1, 0) and radius 1 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}

/-- Line l with equation x + y + 2√2 - 1 = 0 -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + 2 * Real.sqrt 2 - 1 = 0}

/-- Center of circle C -/
def center_C : ℝ × ℝ := (1, 0)

/-- Distance from a point to a line -/
def point_to_line_distance (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem distance_circle_C_to_line_l :
  point_to_line_distance center_C line_l = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_circle_C_to_line_l_l1897_189770


namespace NUMINAMATH_CALUDE_central_high_school_ratio_l1897_189733

theorem central_high_school_ratio (f s : ℚ) 
  (h1 : f > 0) (h2 : s > 0)
  (h3 : (3/7) * f = (2/3) * s) : f / s = 14/9 := by
  sorry

end NUMINAMATH_CALUDE_central_high_school_ratio_l1897_189733


namespace NUMINAMATH_CALUDE_probability_kings_or_aces_value_l1897_189732

/-- A standard deck of cards. -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)
  (num_kings : ℕ)

/-- The probability of drawing either three kings or at least 2 aces
    when 3 cards are selected randomly from a standard deck. -/
def probability_kings_or_aces (d : Deck) : ℚ :=
  sorry

/-- The theorem stating the probability of drawing either three kings or at least 2 aces
    when 3 cards are selected randomly from a standard deck. -/
theorem probability_kings_or_aces_value (d : Deck) 
  (h1 : d.total_cards = 52)
  (h2 : d.num_aces = 4)
  (h3 : d.num_kings = 4) :
  probability_kings_or_aces d = 74 / 5525 :=
sorry

end NUMINAMATH_CALUDE_probability_kings_or_aces_value_l1897_189732


namespace NUMINAMATH_CALUDE_least_value_cubic_equation_l1897_189746

theorem least_value_cubic_equation :
  let f : ℝ → ℝ := λ y => 3 * y^3 + 3 * y^2 + 5 * y + 1
  ∃ y_min : ℝ,
    f y_min = 5 ∧
    ∀ y : ℝ, f y = 5 → y ≥ y_min ∧
    y_min = 1 :=
by sorry

end NUMINAMATH_CALUDE_least_value_cubic_equation_l1897_189746


namespace NUMINAMATH_CALUDE_fourth_largest_common_divisor_l1897_189798

def is_divisor (d n : ℕ) : Prop := n % d = 0

def common_divisors (a b : ℕ) : Set ℕ :=
  {d : ℕ | is_divisor d a ∧ is_divisor d b}

theorem fourth_largest_common_divisor :
  let cd := common_divisors 72 120
  ∃ (l : List ℕ), (∀ x ∈ cd, x ∈ l) ∧
                  (∀ x ∈ l, x ∈ cd) ∧
                  l.Sorted (· > ·) ∧
                  l.get? 3 = some 6 :=
sorry

end NUMINAMATH_CALUDE_fourth_largest_common_divisor_l1897_189798


namespace NUMINAMATH_CALUDE_least_odd_number_satisfying_conditions_l1897_189725

theorem least_odd_number_satisfying_conditions : ∃ (m₁ m₂ n₁ n₂ : ℕ+), 
  let a : ℕ := 261
  (a = m₁.val ^ 2 + n₁.val ^ 2) ∧
  (a ^ 2 = m₂.val ^ 2 + n₂.val ^ 2) ∧
  (m₁.val - n₁.val = m₂.val - n₂.val) ∧
  (∀ (b : ℕ) (k₁ k₂ l₁ l₂ : ℕ+), b < a → b % 2 = 1 → b > 5 →
    (b = k₁.val ^ 2 + l₁.val ^ 2 ∧
     b ^ 2 = k₂.val ^ 2 + l₂.val ^ 2 ∧
     k₁.val - l₁.val = k₂.val - l₂.val) → False) :=
by sorry

end NUMINAMATH_CALUDE_least_odd_number_satisfying_conditions_l1897_189725


namespace NUMINAMATH_CALUDE_fold_square_problem_l1897_189753

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  let distAB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  distAB = 8 ∧ 
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2

-- Define point E as the midpoint of AD
def Midpoint (E A D : ℝ × ℝ) : Prop :=
  E.1 = (A.1 + D.1) / 2 ∧ E.2 = (A.2 + D.2) / 2

-- Define point F on BD such that BF = EF
def PointF (F B D E : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  F.1 = B.1 + t * (D.1 - B.1) ∧ 
  F.2 = B.2 + t * (D.2 - B.2) ∧
  (F.1 - B.1)^2 + (F.2 - B.2)^2 = (F.1 - E.1)^2 + (F.2 - E.2)^2

-- Theorem statement
theorem fold_square_problem (A B C D E F : ℝ × ℝ) :
  Square A B C D → Midpoint E A D → PointF F B D E →
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 3^2 := by
  sorry

end NUMINAMATH_CALUDE_fold_square_problem_l1897_189753


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_product_l1897_189739

theorem largest_common_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∀ m : ℕ, m > 315 → ∃ k : ℕ, k > 0 ∧ Even k ∧ 
    ¬(m ∣ (k+1)*(k+3)*(k+5)*(k+7)*(k+9)*(k+11)*(k+13))) ∧
  (315 ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_product_l1897_189739


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1897_189778

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i : ℂ) / (1 - i) = -1/2 + (1/2 : ℂ) * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1897_189778


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1897_189713

theorem complex_fraction_simplification :
  (2 + 4 * Complex.I) / ((1 + Complex.I)^2) = 2 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1897_189713


namespace NUMINAMATH_CALUDE_first_stick_length_l1897_189712

theorem first_stick_length (stick1 stick2 stick3 : ℝ) : 
  stick2 = 2 * stick1 →
  stick3 = stick2 - 1 →
  stick1 + stick2 + stick3 = 14 →
  stick1 = 3 := by
sorry

end NUMINAMATH_CALUDE_first_stick_length_l1897_189712


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_series_l1897_189719

def fibonacci_factorial_series := [2, 3, 5, 8, 13, 21]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem sum_of_last_two_digits_of_series : 
  (fibonacci_factorial_series.map (λ x => last_two_digits (factorial x))).sum = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_series_l1897_189719


namespace NUMINAMATH_CALUDE_distance_A_l1897_189748

def A : ℝ × ℝ := (0, 15)
def B : ℝ × ℝ := (0, 18)
def C : ℝ × ℝ := (4, 10)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

theorem distance_A'B' :
  ∀ (A' B' : ℝ × ℝ),
    on_line_y_eq_x A' →
    on_line_y_eq_x B' →
    collinear A A' C →
    collinear B B' C →
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_l1897_189748


namespace NUMINAMATH_CALUDE_james_walking_distance_l1897_189762

def base7_to_base10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem james_walking_distance :
  base7_to_base10 3 6 5 2 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_james_walking_distance_l1897_189762


namespace NUMINAMATH_CALUDE_smallest_h_divisible_by_primes_l1897_189721

theorem smallest_h_divisible_by_primes : ∃ (h : ℕ), h > 0 ∧ 
  (∀ (h' : ℕ), h' < h → ¬∃ (k : ℤ), (13 ∣ (h' + k)) ∧ (17 ∣ (h' + k)) ∧ (29 ∣ (h' + k))) ∧
  ∃ (k : ℤ), (13 ∣ (h + k)) ∧ (17 ∣ (h + k)) ∧ (29 ∣ (h + k)) :=
by sorry

#check smallest_h_divisible_by_primes

end NUMINAMATH_CALUDE_smallest_h_divisible_by_primes_l1897_189721


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_26_l1897_189744

/-- Calculates the net rate of pay for a driver given specific conditions --/
theorem driver_net_pay_rate (hours : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (ac_efficiency_decrease : ℝ) (pay_per_mile : ℝ) (gas_price : ℝ) 
  (gas_price_increase : ℝ) : ℝ :=
  let distance := hours * speed
  let adjusted_fuel_efficiency := fuel_efficiency * (1 - ac_efficiency_decrease)
  let gas_used := distance / adjusted_fuel_efficiency
  let earnings := pay_per_mile * distance
  let new_gas_price := gas_price * (1 + gas_price_increase)
  let gas_cost := new_gas_price * gas_used
  let net_earnings := earnings - gas_cost
  let net_rate := net_earnings / hours
  net_rate

/-- Proves that the driver's net rate of pay is $26 per hour under given conditions --/
theorem driver_net_pay_is_26 :
  driver_net_pay_rate 3 50 30 0.1 0.6 2 0.2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_26_l1897_189744


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1897_189764

theorem circle_radius_from_area_circumference_ratio 
  (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) (h3 : P / Q = 10) : 
  ∃ r : ℝ, r > 0 ∧ P = π * r^2 ∧ Q = 2 * π * r ∧ r = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1897_189764


namespace NUMINAMATH_CALUDE_opposite_numbers_proof_l1897_189718

theorem opposite_numbers_proof : 
  (-(5^2) = -((5^2))) ∧ ((5^2) = (-5)^2) → 
  (-(5^2) = -(((-5)^2))) ∧ (-(5^2) ≠ (-5)^2) := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_proof_l1897_189718


namespace NUMINAMATH_CALUDE_team_a_wins_l1897_189752

theorem team_a_wins (total_matches : ℕ) (team_a_points : ℕ) : 
  total_matches = 10 → 
  team_a_points = 22 → 
  ∃ (wins draws : ℕ), 
    wins + draws = total_matches ∧ 
    3 * wins + draws = team_a_points ∧ 
    wins = 6 :=
by sorry

end NUMINAMATH_CALUDE_team_a_wins_l1897_189752


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1897_189740

theorem sqrt_inequality (n : ℕ+) : Real.sqrt (n + 1) - Real.sqrt n < 1 / (2 * Real.sqrt n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1897_189740


namespace NUMINAMATH_CALUDE_kishore_rent_expenditure_l1897_189729

def monthly_salary (savings : ℕ) : ℕ := savings * 10

def total_expenses (salary : ℕ) : ℕ := (salary * 9) / 10

def other_expenses : ℕ := 1500 + 4500 + 2500 + 2000 + 3940

def rent_expenditure (total_exp other_exp : ℕ) : ℕ := total_exp - other_exp

theorem kishore_rent_expenditure (savings : ℕ) (h : savings = 2160) :
  rent_expenditure (total_expenses (monthly_salary savings)) other_expenses = 5000 := by
  sorry

end NUMINAMATH_CALUDE_kishore_rent_expenditure_l1897_189729


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l1897_189781

theorem largest_multiple_of_8_under_100 :
  ∃ n : ℕ, n * 8 = 96 ∧ n * 8 < 100 ∧ ∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l1897_189781


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l1897_189711

theorem impossible_coin_probabilities :
  ¬∃ (p₁ p₂ : ℝ), 0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧
    (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
    p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l1897_189711


namespace NUMINAMATH_CALUDE_five_classes_in_school_l1897_189706

/-- Represents the number of students in each class -/
def class_sizes (n : ℕ) : ℕ → ℕ
  | 0 => 25
  | i + 1 => class_sizes n i - 2

/-- The total number of students in the school -/
def total_students (n : ℕ) : ℕ :=
  (List.range n).map (class_sizes n) |>.sum

/-- The theorem stating that there are 5 classes in the school -/
theorem five_classes_in_school :
  ∃ n : ℕ, n > 0 ∧ total_students n = 105 ∧ n = 5 :=
sorry

end NUMINAMATH_CALUDE_five_classes_in_school_l1897_189706


namespace NUMINAMATH_CALUDE_correct_minus_incorrect_l1897_189704

/-- Calculates the result following the order of operations -/
def J : ℤ := 12 - (3 * 4)

/-- Calculates the result ignoring parentheses and going from left to right -/
def A : ℤ := (12 - 3) * 4

/-- The difference between the correct calculation and the incorrect one -/
theorem correct_minus_incorrect : J - A = -36 := by sorry

end NUMINAMATH_CALUDE_correct_minus_incorrect_l1897_189704


namespace NUMINAMATH_CALUDE_train_crossing_signal_pole_l1897_189776

/-- Given a train and a platform with the following properties:
  * The train is 300 meters long
  * The platform is 250 meters long
  * The train crosses the platform in 33 seconds
  This theorem proves that the time taken for the train to cross a signal pole is 18 seconds. -/
theorem train_crossing_signal_pole
  (train_length : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 250)
  (h3 : platform_crossing_time = 33)
  : ℝ :=
let total_distance := train_length + platform_length
let train_speed := total_distance / platform_crossing_time
let signal_pole_crossing_time := train_length / train_speed
18

/-- The proof of the theorem -/
lemma train_crossing_signal_pole_proof
  (train_length : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 250)
  (h3 : platform_crossing_time = 33)
  : train_crossing_signal_pole train_length platform_length platform_crossing_time h1 h2 h3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_signal_pole_l1897_189776


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1897_189738

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The arithmetic sequence -/
def a : ℕ → ℝ := sorry

theorem arithmetic_sequence_sum :
  (S 7 = 28) → (S 11 = 66) → (S 9 = 45) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1897_189738


namespace NUMINAMATH_CALUDE_min_value_H_negative_reals_l1897_189789

-- Define the concept of an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function H
def H (a b : ℝ) (f g : ℝ → ℝ) (x : ℝ) : ℝ := a * f x + b * g x + 1

-- State the theorem
theorem min_value_H_negative_reals 
  (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : OddFunction f) (hg : OddFunction g)
  (hmax : ∃ M, M = 5 ∧ ∀ x > 0, H a b f g x ≤ M) :
  ∃ m, m = -3 ∧ ∀ x < 0, H a b f g x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_H_negative_reals_l1897_189789


namespace NUMINAMATH_CALUDE_picnic_gender_difference_l1897_189709

/-- Given a group of people at a picnic, prove the difference between men and women -/
theorem picnic_gender_difference 
  (total : ℕ) 
  (men : ℕ) 
  (women : ℕ) 
  (children : ℕ) 
  (h1 : total = 200)
  (h2 : men + women = children + 20)
  (h3 : men = 65)
  (h4 : total = men + women + children) :
  men - women = 20 := by
sorry

end NUMINAMATH_CALUDE_picnic_gender_difference_l1897_189709


namespace NUMINAMATH_CALUDE_uncertain_roots_l1897_189741

/-- Given that mx² - 2(m+2)x + m + 5 = 0 has no real roots, 
    prove that the number of real roots of (m-5)x² - 2(m+2)x + m = 0 is uncertain. -/
theorem uncertain_roots (m : ℝ) 
  (h : ∀ x : ℝ, m * x^2 - 2*(m+2)*x + m + 5 ≠ 0) : 
  ∃ m₁ m₂ : ℝ, 
    (∃! x : ℝ, (m₁-5) * x^2 - 2*(m₁+2)*x + m₁ = 0) ∧ 
    (∃ x y : ℝ, x ≠ y ∧ (m₂-5) * x^2 - 2*(m₂+2)*x + m₂ = 0 ∧ (m₂-5) * y^2 - 2*(m₂+2)*y + m₂ = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_uncertain_roots_l1897_189741


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_positive_l1897_189795

theorem negation_of_existence_squared_positive :
  (¬ ∃ x : ℝ, x^2 > 0) ↔ (∀ x : ℝ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_positive_l1897_189795


namespace NUMINAMATH_CALUDE_probability_matching_letter_l1897_189703

def word1 : String := "MATHEMATICS"
def word2 : String := "CALCULUS"

def is_in_word2 (c : Char) : Bool :=
  word2.contains c

def count_matching_letters : Nat :=
  word1.toList.filter is_in_word2 |>.length

theorem probability_matching_letter :
  (count_matching_letters : ℚ) / word1.length = 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_matching_letter_l1897_189703


namespace NUMINAMATH_CALUDE_remainder_96_104_div_9_l1897_189790

theorem remainder_96_104_div_9 : (96 * 104) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_96_104_div_9_l1897_189790


namespace NUMINAMATH_CALUDE_paintbrush_cost_calculation_l1897_189737

/-- The cost of the paintbrush Rose wants to buy -/
def paintbrush_cost (paints_cost easel_cost rose_has rose_needs : ℚ) : ℚ :=
  (rose_has + rose_needs) - (paints_cost + easel_cost)

/-- Theorem stating the cost of the paintbrush Rose wants to buy -/
theorem paintbrush_cost_calculation :
  paintbrush_cost 9.20 6.50 7.10 11 = 2.40 := by sorry

end NUMINAMATH_CALUDE_paintbrush_cost_calculation_l1897_189737


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1897_189777

/-- An arithmetic sequence {a_n} with its partial sums S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Theorem: For an arithmetic sequence, if S_4 = 25 and S_8 = 100, then S_12 = 225 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.S 4 = 25) (h2 : seq.S 8 = 100) : seq.S 12 = 225 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1897_189777


namespace NUMINAMATH_CALUDE_inequality_range_l1897_189768

theorem inequality_range (x : ℝ) : 
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) ↔ (x < -1 ∨ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1897_189768


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l1897_189715

/-- The average speed of a round trip given outbound and return speeds -/
theorem round_trip_average_speed
  (outbound_speed : ℝ)
  (return_speed : ℝ)
  (h1 : outbound_speed = 60)
  (h2 : return_speed = 40)
  : (2 / (1 / outbound_speed + 1 / return_speed)) = 48 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l1897_189715


namespace NUMINAMATH_CALUDE_log_sqrt3_729sqrt3_l1897_189750

theorem log_sqrt3_729sqrt3 : Real.log (729 * Real.sqrt 3) / Real.log (Real.sqrt 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt3_729sqrt3_l1897_189750


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1897_189716

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₂ = 3 and a₅ + a₇ = 10, prove that a₁ + a₁₀ = 9.5 -/
theorem arithmetic_sequence_sum (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a2 : a 2 = 3) 
  (h_sum : a 5 + a 7 = 10) : 
  a 1 + a 10 = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1897_189716


namespace NUMINAMATH_CALUDE_geometric_sum_proof_l1897_189794

theorem geometric_sum_proof : 
  let a₁ : ℚ := 3/4
  let r : ℚ := 3/4
  let n : ℕ := 10
  let S := a₁ * (1 - r^n) / (1 - r)
  S = 2968581/1048576 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_proof_l1897_189794


namespace NUMINAMATH_CALUDE_wednesday_earnings_l1897_189714

/-- Represents the working hours and earnings of Jack and Bob on a particular Wednesday -/
structure WorkDay where
  t : ℝ
  jack_hours : ℝ := t - 2
  jack_rate : ℝ := 3 * t - 2
  bob_hours : ℝ := 1.5 * (t - 2)
  bob_rate : ℝ := (3 * t - 2) - (2 * t - 7)
  tax : ℝ := 10

/-- The theorem stating that t = 19/3 is the only valid solution -/
theorem wednesday_earnings (w : WorkDay) : 
  (w.jack_hours * w.jack_rate - w.tax = w.bob_hours * w.bob_rate - w.tax) ∧ 
  (w.jack_hours > 0) ∧ (w.bob_hours > 0) → 
  w.t = 19/3 := by
  sorry

#check wednesday_earnings

end NUMINAMATH_CALUDE_wednesday_earnings_l1897_189714


namespace NUMINAMATH_CALUDE_odd_function_values_and_monotonicity_and_inequality_l1897_189761

noncomputable def f (a b x : ℝ) : ℝ := (b - 2^x) / (2^(x+1) + a)

theorem odd_function_values_and_monotonicity_and_inequality (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (a = 2 ∧ b = 1) ∧
  (∀ x y, x < y → f 2 1 x > f 2 1 y) ∧
  (∀ k, (∀ x ≥ 1, f 2 1 (k * 3^x) + f 2 1 (3^x - 9^x + 2) > 0) ↔ k < 4/3) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_values_and_monotonicity_and_inequality_l1897_189761


namespace NUMINAMATH_CALUDE_oblomov_weight_change_l1897_189747

theorem oblomov_weight_change : 
  let spring_factor : ℝ := 0.75
  let summer_factor : ℝ := 1.20
  let autumn_factor : ℝ := 0.90
  let winter_factor : ℝ := 1.20
  spring_factor * summer_factor * autumn_factor * winter_factor < 1 := by
sorry

end NUMINAMATH_CALUDE_oblomov_weight_change_l1897_189747


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l1897_189736

/-- A parallelogram with vertices A, B, C, D in 2D space -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The sum of coordinates of a point -/
def sumCoordinates (p : ℝ × ℝ) : ℝ := p.1 + p.2

/-- Theorem: In parallelogram ABCD with A(-1,2), B(3,-4), C(7,3), and A,C opposite, 
    the sum of coordinates of D is 12 -/
theorem parallelogram_vertex_sum (ABCD : Parallelogram) 
    (hA : ABCD.A = (-1, 2))
    (hB : ABCD.B = (3, -4))
    (hC : ABCD.C = (7, 3))
    (hAC_opposite : ABCD.A = (-ABCD.C.1, -ABCD.C.2)) :
    sumCoordinates ABCD.D = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l1897_189736


namespace NUMINAMATH_CALUDE_f_bijection_l1897_189731

def f (n : ℤ) : ℤ := 2 * n

theorem f_bijection : Function.Bijective f := by sorry

end NUMINAMATH_CALUDE_f_bijection_l1897_189731


namespace NUMINAMATH_CALUDE_batsman_average_l1897_189787

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  (previous_total = 16 * previous_average) →
  (previous_total + 87) / 17 = previous_average + 3 →
  (previous_total + 87) / 17 = 39 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l1897_189787


namespace NUMINAMATH_CALUDE_range_of_f_l1897_189779

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the function we want to find the range of
def f (x y : ℝ) : ℝ := 2*x + y

-- Theorem statement
theorem range_of_f :
  (∃ (x y : ℝ), ellipse x y ∧ f x y = Real.sqrt 17) ∧
  (∃ (x y : ℝ), ellipse x y ∧ f x y = -Real.sqrt 17) ∧
  (∀ (x y : ℝ), ellipse x y → f x y ≤ Real.sqrt 17) ∧
  (∀ (x y : ℝ), ellipse x y → f x y ≥ -Real.sqrt 17) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1897_189779


namespace NUMINAMATH_CALUDE_inequality_implication_l1897_189759

theorem inequality_implication (a b c : ℝ) (h1 : a / c^2 > b / c^2) (h2 : c ≠ 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1897_189759


namespace NUMINAMATH_CALUDE_last_three_digits_of_8_105_l1897_189724

theorem last_three_digits_of_8_105 : 8^105 ≡ 992 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_8_105_l1897_189724


namespace NUMINAMATH_CALUDE_max_modulus_complex_l1897_189745

theorem max_modulus_complex (z : ℂ) : 
  ∀ z, Complex.abs (z + z⁻¹) = 1 → Complex.abs z ≤ (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_modulus_complex_l1897_189745


namespace NUMINAMATH_CALUDE_savings_of_eight_hundred_bills_l1897_189774

/-- The total savings amount when exchanged into a given number of $100 bills -/
def savings_amount (num_bills : ℕ) : ℕ := 100 * num_bills

/-- Theorem: If a person has 8 $100 bills after exchanging all their savings, 
    their total savings amount to $800 -/
theorem savings_of_eight_hundred_bills : 
  savings_amount 8 = 800 := by sorry

end NUMINAMATH_CALUDE_savings_of_eight_hundred_bills_l1897_189774


namespace NUMINAMATH_CALUDE_triangle_max_value_l1897_189754

/-- In a triangle ABC, given the conditions, prove the maximum value of (1/2)b + a -/
theorem triangle_max_value (a b c : ℝ) (h1 : a^2 + b^2 = c^2 + a*b) (h2 : c = 1) :
  (∃ (x y : ℝ), x^2 + y^2 = 1^2 + x*y ∧ (1/2)*y + x ≤ (1/2)*b + a) ∧
  (∀ (x y : ℝ), x^2 + y^2 = 1^2 + x*y → (1/2)*y + x ≤ (1/2)*b + a) →
  (1/2)*b + a = Real.sqrt 21 / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_value_l1897_189754


namespace NUMINAMATH_CALUDE_carnation_fraction_l1897_189700

/-- Represents a flower bouquet with pink and red roses and carnations -/
structure Bouquet where
  pink_roses : ℚ
  red_roses : ℚ
  pink_carnations : ℚ
  red_carnations : ℚ

/-- The fraction of carnations in the bouquet is 7/10 -/
theorem carnation_fraction (b : Bouquet) : 
  b.pink_roses + b.red_roses + b.pink_carnations + b.red_carnations = 1 →
  b.pink_roses + b.pink_carnations = 6/10 →
  b.pink_roses = 1/3 * (b.pink_roses + b.pink_carnations) →
  b.red_carnations = 3/4 * (b.red_roses + b.red_carnations) →
  b.pink_carnations + b.red_carnations = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_carnation_fraction_l1897_189700


namespace NUMINAMATH_CALUDE_fraction_scaling_l1897_189765

theorem fraction_scaling (x y : ℝ) :
  (3*x + 3*y) / ((3*x)^2 + (3*y)^2) = (1/3) * ((x + y) / (x^2 + y^2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_scaling_l1897_189765


namespace NUMINAMATH_CALUDE_church_cookie_baking_l1897_189742

theorem church_cookie_baking (members : ℕ) (sheets_per_member : ℕ) (cookies_per_sheet : ℕ)
  (h1 : members = 100)
  (h2 : sheets_per_member = 10)
  (h3 : cookies_per_sheet = 16) :
  members * sheets_per_member * cookies_per_sheet = 16000 := by
  sorry

end NUMINAMATH_CALUDE_church_cookie_baking_l1897_189742


namespace NUMINAMATH_CALUDE_inscribed_polygon_limit_l1897_189757

noncomputable def a (n : ℕ) : ℝ :=
  Real.sqrt (2 - Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2)))))))

theorem inscribed_polygon_limit :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |2^n * a n - π| < ε :=
by sorry

end NUMINAMATH_CALUDE_inscribed_polygon_limit_l1897_189757


namespace NUMINAMATH_CALUDE_larger_number_proof_l1897_189793

/-- Given two positive integers with HCF 23 and LCM factors 13 and 14, prove the larger number is 322 -/
theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) → 
  (∃ (k : ℕ+), Nat.lcm a b = 23 * 13 * 14 * k) → 
  (max a b = 322) := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1897_189793


namespace NUMINAMATH_CALUDE_power_difference_equals_multiple_of_thirty_power_l1897_189796

theorem power_difference_equals_multiple_of_thirty_power : 
  (5^1002 + 6^1001)^2 - (5^1002 - 6^1001)^2 = 24 * 30^1001 := by
sorry

end NUMINAMATH_CALUDE_power_difference_equals_multiple_of_thirty_power_l1897_189796


namespace NUMINAMATH_CALUDE_log_function_fixed_point_l1897_189769

theorem log_function_fixed_point (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log a + 1
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_function_fixed_point_l1897_189769


namespace NUMINAMATH_CALUDE_f_at_negative_two_equals_six_l1897_189723

-- Define the functions f and g
def f (a b c x : ℝ) : ℝ := a * x^2 + 2 * b * x + c
def g (a b c x : ℝ) : ℝ := (a + 1) * x^2 + 2 * (b + 2) * x + (c + 4)

-- Define the discriminant function
def discriminant (a b c : ℝ) : ℝ := 4 * b^2 - 4 * a * c

-- State the theorem
theorem f_at_negative_two_equals_six (a b c : ℝ) :
  discriminant a b c - discriminant (a + 1) (b + 2) (c + 4) = 24 →
  f a b c (-2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_f_at_negative_two_equals_six_l1897_189723


namespace NUMINAMATH_CALUDE_pond_fish_problem_l1897_189797

/-- Represents the number of fish in a pond -/
def total_fish : ℕ := 500

/-- Represents the number of fish initially tagged -/
def tagged_fish : ℕ := 50

/-- Represents the number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- Represents the number of tagged fish found in the second catch -/
def tagged_in_second_catch : ℕ := 5

theorem pond_fish_problem :
  (tagged_in_second_catch : ℚ) / second_catch = tagged_fish / total_fish :=
sorry

end NUMINAMATH_CALUDE_pond_fish_problem_l1897_189797


namespace NUMINAMATH_CALUDE_zoe_bought_eight_roses_l1897_189786

/-- Calculates the number of roses bought given the total spent, cost per flower, and number of daisies. -/
def roses_bought (total_spent : ℕ) (cost_per_flower : ℕ) (num_daisies : ℕ) : ℕ :=
  (total_spent - cost_per_flower * num_daisies) / cost_per_flower

/-- Proves that Zoe bought 8 roses given the problem conditions. -/
theorem zoe_bought_eight_roses (total_spent : ℕ) (cost_per_flower : ℕ) (num_daisies : ℕ) 
    (h1 : total_spent = 30)
    (h2 : cost_per_flower = 3)
    (h3 : num_daisies = 2) : 
  roses_bought total_spent cost_per_flower num_daisies = 8 := by
  sorry

#eval roses_bought 30 3 2  -- Should output 8

end NUMINAMATH_CALUDE_zoe_bought_eight_roses_l1897_189786


namespace NUMINAMATH_CALUDE_bedroom_curtain_width_l1897_189775

theorem bedroom_curtain_width :
  let initial_width : ℝ := 16
  let initial_height : ℝ := 12
  let living_room_width : ℝ := 4
  let living_room_height : ℝ := 6
  let bedroom_height : ℝ := 4
  let remaining_area : ℝ := 160
  let total_area := initial_width * initial_height
  let living_room_area := living_room_width * living_room_height
  let bedroom_width := (total_area - living_room_area - remaining_area) / bedroom_height
  bedroom_width = 2 := by sorry

end NUMINAMATH_CALUDE_bedroom_curtain_width_l1897_189775


namespace NUMINAMATH_CALUDE_digit_2009_is_zero_l1897_189707

/-- The sequence of digits obtained by writing natural numbers successively -/
def digit_sequence : ℕ → ℕ := sorry

/-- The number of digits used to write numbers from 1 to n -/
def digits_count (n : ℕ) : ℕ := sorry

/-- The 2009th digit in the sequence -/
def digit_2009 : ℕ := digit_sequence 2009

theorem digit_2009_is_zero : digit_2009 = 0 := by sorry

end NUMINAMATH_CALUDE_digit_2009_is_zero_l1897_189707


namespace NUMINAMATH_CALUDE_total_pages_calculation_l1897_189727

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 9

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 49

/-- The total number of pages in all booklets -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem total_pages_calculation :
  total_pages = 441 :=
by sorry

end NUMINAMATH_CALUDE_total_pages_calculation_l1897_189727


namespace NUMINAMATH_CALUDE_complement_of_union_equals_four_l1897_189717

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_equals_four :
  (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_four_l1897_189717


namespace NUMINAMATH_CALUDE_balance_condition1_balance_condition2_triangular_weight_is_60_l1897_189726

/-- The weight of a rectangular weight in grams -/
def rectangular_weight : ℝ := 90

/-- The weight of a round weight in grams -/
def round_weight : ℝ := 30

/-- The weight of a triangular weight in grams -/
def triangular_weight : ℝ := 60

/-- First balance condition: 1 round + 1 triangular = 3 round -/
theorem balance_condition1 : round_weight + triangular_weight = 3 * round_weight := by sorry

/-- Second balance condition: 4 round + 1 triangular = 1 triangular + 1 round + 1 rectangular -/
theorem balance_condition2 : 4 * round_weight + triangular_weight = triangular_weight + round_weight + rectangular_weight := by sorry

/-- Proof that the triangular weight is 60 grams -/
theorem triangular_weight_is_60 : triangular_weight = 60 := by sorry

end NUMINAMATH_CALUDE_balance_condition1_balance_condition2_triangular_weight_is_60_l1897_189726


namespace NUMINAMATH_CALUDE_average_value_function_m_range_l1897_189758

/-- Definition of an average value function -/
def is_average_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ : ℝ, a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

/-- The function we're considering -/
def f (m : ℝ) : ℝ → ℝ := λ x => x^2 - m*x - 1

/-- The theorem statement -/
theorem average_value_function_m_range :
  ∀ m : ℝ, is_average_value_function (f m) (-1) 1 → 0 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_average_value_function_m_range_l1897_189758


namespace NUMINAMATH_CALUDE_function_range_l1897_189756

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

-- Define the domain
def domain : Set ℝ := {x | -2 < x ∧ x < 1}

-- State the theorem
theorem function_range :
  {y | ∃ x ∈ domain, f x = y} = {y | -2 ≤ y ∧ y < 2} := by sorry

end NUMINAMATH_CALUDE_function_range_l1897_189756


namespace NUMINAMATH_CALUDE_cone_with_hole_volume_l1897_189749

/-- The volume of a cone with a cylindrical hole -/
theorem cone_with_hole_volume
  (cone_diameter : ℝ)
  (cone_height : ℝ)
  (hole_diameter : ℝ)
  (h_cone_diameter : cone_diameter = 12)
  (h_cone_height : cone_height = 12)
  (h_hole_diameter : hole_diameter = 4) :
  (1/3 * π * (cone_diameter/2)^2 * cone_height) - (π * (hole_diameter/2)^2 * cone_height) = 96 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_with_hole_volume_l1897_189749


namespace NUMINAMATH_CALUDE_no_base_all_prime_l1897_189766

/-- For any base b ≥ 2, there exists a number of the form 11...1 
    with (b^2 - 1) ones in base b that is not prime. -/
theorem no_base_all_prime (b : ℕ) (hb : b ≥ 2) : 
  ∃ N : ℕ, (∃ k : ℕ, N = (b^(2*k) - 1) / (b^2 - 1)) ∧ ¬ Prime N := by
  sorry

end NUMINAMATH_CALUDE_no_base_all_prime_l1897_189766


namespace NUMINAMATH_CALUDE_triangle_problem_l1897_189705

noncomputable section

/-- Given a triangle ABC with the following properties:
  BC = √5
  AC = 3
  sin C = 2 * sin A
  Prove that:
  1. AB = 2√5
  2. sin(2A - π/4) = √2/10
-/
theorem triangle_problem (A B C : ℝ) (h1 : Real.sqrt 5 = BC)
  (h2 : 3 = AC) (h3 : Real.sin C = 2 * Real.sin A) :
  AB = 2 * Real.sqrt 5 ∧ Real.sin (2 * A - π / 4) = Real.sqrt 2 / 10 :=
by sorry

end

end NUMINAMATH_CALUDE_triangle_problem_l1897_189705


namespace NUMINAMATH_CALUDE_square_sum_inequality_l1897_189799

theorem square_sum_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l1897_189799


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l1897_189782

theorem tangent_line_intersection (a : ℝ) : 
  (∃ x : ℝ, x + Real.log x = a * x^2 + (a + 2) * x + 1 ∧ 
   2 * x - 1 = a * x^2 + (a + 2) * x + 1) ↔ 
  a = 8 := by sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l1897_189782


namespace NUMINAMATH_CALUDE_meat_for_community_event_l1897_189763

/-- The amount of meat (in pounds) needed to make a given number of hamburgers. -/
def meat_needed (hamburgers : ℕ) : ℚ :=
  (5 : ℚ) * hamburgers / 10

/-- Theorem stating that 15 pounds of meat are needed for 30 hamburgers. -/
theorem meat_for_community_event : meat_needed 30 = 15 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_community_event_l1897_189763
