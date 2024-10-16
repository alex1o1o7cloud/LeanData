import Mathlib

namespace NUMINAMATH_CALUDE_time_to_paint_one_house_l2338_233851

/-- Given that 9 houses can be painted in 3 hours, prove that it takes 20 minutes to paint one house. -/
theorem time_to_paint_one_house :
  let total_houses : ℕ := 9
  let total_hours : ℕ := 3
  let minutes_per_hour : ℕ := 60
  total_hours * minutes_per_hour / total_houses = 20 := by
  sorry

end NUMINAMATH_CALUDE_time_to_paint_one_house_l2338_233851


namespace NUMINAMATH_CALUDE_opposite_sign_implies_y_power_x_25_l2338_233811

theorem opposite_sign_implies_y_power_x_25 (x y : ℝ) : 
  (((x - 2)^2 > 0 ∧ |5 + y| < 0) ∨ ((x - 2)^2 < 0 ∧ |5 + y| > 0)) → y^x = 25 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_implies_y_power_x_25_l2338_233811


namespace NUMINAMATH_CALUDE_power_two_divides_power_odd_minus_one_l2338_233868

theorem power_two_divides_power_odd_minus_one (k n : ℕ) (h_k_odd : Odd k) (h_n_ge_one : n ≥ 1) :
  ∃ m : ℤ, k^(2^n) - 1 = 2^(n+2) * m :=
sorry

end NUMINAMATH_CALUDE_power_two_divides_power_odd_minus_one_l2338_233868


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l2338_233841

/-- Given that 5(log_a x)^2 + 9(log_b x)^2 = (20(log x)^2) / (log a log b) and a, b, x > 1,
    prove that b = a^((20+√220)/10) or b = a^((20-√220)/10) -/
theorem logarithmic_equation_solution (a b x : ℝ) (ha : a > 1) (hb : b > 1) (hx : x > 1)
  (h : 5 * (Real.log x / Real.log a)^2 + 9 * (Real.log x / Real.log b)^2 = 20 * (Real.log x)^2 / (Real.log a * Real.log b)) :
  b = a^((20 + Real.sqrt 220) / 10) ∨ b = a^((20 - Real.sqrt 220) / 10) := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l2338_233841


namespace NUMINAMATH_CALUDE_initial_men_count_l2338_233816

/-- Proves that the initial number of men is 760, given the food supply conditions. -/
theorem initial_men_count (M : ℕ) : 
  (M * 22 = (M + 40) * 19 + M * 2) → M = 760 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l2338_233816


namespace NUMINAMATH_CALUDE_division_addition_equality_l2338_233865

theorem division_addition_equality : (-180) / (-45) + (-9) = -5 := by
  sorry

end NUMINAMATH_CALUDE_division_addition_equality_l2338_233865


namespace NUMINAMATH_CALUDE_contractor_absent_days_l2338_233895

/-- Proves the number of absent days for a contractor under specific conditions -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (daily_pay : ℚ) 
  (daily_fine : ℚ) 
  (total_received : ℚ) 
  (h1 : total_days = 30)
  (h2 : daily_pay = 25)
  (h3 : daily_fine = 7.5)
  (h4 : total_received = 360) :
  ∃ (absent_days : ℕ), 
    (absent_days : ℚ) * daily_fine + (total_days - absent_days : ℚ) * daily_pay = total_received ∧ 
    absent_days = 12 := by
  sorry


end NUMINAMATH_CALUDE_contractor_absent_days_l2338_233895


namespace NUMINAMATH_CALUDE_polar_equation_graph_l2338_233870

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a graph in polar coordinates -/
inductive PolarGraph
  | Circle : PolarGraph
  | Ray : PolarGraph
  | Both : PolarGraph

/-- The equation (ρ-3)(θ-π/2)=0 with ρ≥0 -/
def polarEquation (p : PolarPoint) : Prop :=
  (p.ρ - 3) * (p.θ - Real.pi / 2) = 0 ∧ p.ρ ≥ 0

/-- The theorem stating that the equation represents a circle and a ray -/
theorem polar_equation_graph : 
  (∃ p : PolarPoint, polarEquation p) → PolarGraph.Both = PolarGraph.Both :=
sorry

end NUMINAMATH_CALUDE_polar_equation_graph_l2338_233870


namespace NUMINAMATH_CALUDE_kyler_wins_two_l2338_233833

/-- Represents a chess player --/
inductive Player
| Peter
| Emma
| Kyler

/-- Represents the number of games won and lost by a player --/
structure GameRecord where
  player : Player
  wins : ℕ
  losses : ℕ

/-- The total number of games in the tournament --/
def totalGames : ℕ := 6

theorem kyler_wins_two (peter_record : GameRecord) (emma_record : GameRecord) (kyler_record : GameRecord) :
  peter_record.player = Player.Peter ∧
  peter_record.wins = 5 ∧
  peter_record.losses = 4 ∧
  emma_record.player = Player.Emma ∧
  emma_record.wins = 2 ∧
  emma_record.losses = 5 ∧
  kyler_record.player = Player.Kyler ∧
  kyler_record.losses = 4 →
  kyler_record.wins = 2 := by
  sorry

end NUMINAMATH_CALUDE_kyler_wins_two_l2338_233833


namespace NUMINAMATH_CALUDE_highest_score_is_179_l2338_233866

/-- Represents a batsman's statistics --/
structure BatsmanStats where
  totalInnings : ℕ
  overallAverage : ℚ
  highLowDifference : ℕ
  averageExcludingHighLow : ℚ

/-- Calculates the highest score of a batsman given their statistics --/
def highestScore (stats : BatsmanStats) : ℕ :=
  sorry

/-- Theorem stating that the highest score is 179 for the given conditions --/
theorem highest_score_is_179 (stats : BatsmanStats) 
  (h1 : stats.totalInnings = 46)
  (h2 : stats.overallAverage = 60)
  (h3 : stats.highLowDifference = 150)
  (h4 : stats.averageExcludingHighLow = 58) :
  highestScore stats = 179 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_is_179_l2338_233866


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2338_233818

theorem root_sum_theorem (a b : ℝ) (ha : a ≠ 0) (h : a^2 + b*a - 2*a = 0) : a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2338_233818


namespace NUMINAMATH_CALUDE_urn_contents_l2338_233856

/-- Represents the contents of an urn with yellow, white, and red balls. -/
structure Urn :=
  (yellow : ℕ)
  (white : ℕ)
  (red : ℕ)

/-- Calculates the probability of drawing balls of given colors from the urn. -/
def probability (u : Urn) (colors : List ℕ) : ℚ :=
  (colors.sum : ℚ) / ((u.yellow + u.white + u.red) : ℚ)

/-- The main theorem about the urn contents. -/
theorem urn_contents : 
  ∀ (u : Urn), 
    u.yellow = 18 →
    probability u [u.white, u.red] = probability u [u.white, u.yellow] - 1/15 →
    probability u [u.red, u.yellow] = probability u [u.white, u.yellow] * 11/10 →
    u.white = 27 ∧ u.red = 16 := by
  sorry

end NUMINAMATH_CALUDE_urn_contents_l2338_233856


namespace NUMINAMATH_CALUDE_five_ruble_coins_count_l2338_233844

/-- Given the total number of coins and the number of coins that are not of each other denomination,
    prove that the number of five-ruble coins is 5. -/
theorem five_ruble_coins_count
  (total_coins : ℕ)
  (not_two_ruble : ℕ)
  (not_ten_ruble : ℕ)
  (not_one_ruble : ℕ)
  (h1 : total_coins = 25)
  (h2 : not_two_ruble = 19)
  (h3 : not_ten_ruble = 20)
  (h4 : not_one_ruble = 16) :
  total_coins - ((total_coins - not_two_ruble) + (total_coins - not_ten_ruble) + (total_coins - not_one_ruble)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_five_ruble_coins_count_l2338_233844


namespace NUMINAMATH_CALUDE_rhombus_trapezoids_l2338_233805

/-- Represents a rhombus with side length n and angle A of 60° -/
structure Rhombus (n : ℕ) where
  side_length : ℕ
  angle_A : ℝ
  side_length_eq : side_length = n
  angle_A_eq : angle_A = 60

/-- The number of trapezoids formed in the rhombus configuration -/
def num_trapezoids (n : ℕ) : ℕ := n * (n^2 - 1) * (2*n + 1) / 3

/-- Theorem stating the number of trapezoids in the given rhombus configuration -/
theorem rhombus_trapezoids (n : ℕ) (r : Rhombus n) :
  ∃ (s : ℕ → ℕ), s n = num_trapezoids n := by
  sorry

end NUMINAMATH_CALUDE_rhombus_trapezoids_l2338_233805


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2338_233892

theorem remainder_divisibility (N : ℤ) : 
  N % 2 = 1 → N % 35 = 1 → N % 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2338_233892


namespace NUMINAMATH_CALUDE_min_cars_theorem_l2338_233821

/-- Calculates the minimum number of cars needed for a family where each car must rest one day a week and all adults want to drive daily. -/
def min_cars_needed (num_adults : ℕ) : ℕ :=
  if num_adults ≤ 6 then
    num_adults + 1
  else
    (num_adults * 7 + 5) / 6

theorem min_cars_theorem (num_adults : ℕ) :
  (num_adults = 5 → min_cars_needed num_adults = 6) ∧
  (num_adults = 8 → min_cars_needed num_adults = 10) :=
by sorry

#eval min_cars_needed 5  -- Should output 6
#eval min_cars_needed 8  -- Should output 10

end NUMINAMATH_CALUDE_min_cars_theorem_l2338_233821


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2338_233823

theorem complex_absolute_value (t : ℝ) : 
  t > 0 → Complex.abs (-5 + t * Complex.I) = 3 * Real.sqrt 13 → t = 2 * Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2338_233823


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2338_233837

-- Define the type for a point in 2D space
def Point := ℝ × ℝ

-- Define the type for a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Define a function to check if two lines are perpendicular
def areLinesPerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Theorem statement
theorem perpendicular_line_through_point 
  (l : Line) 
  (h1 : isPointOnLine (-1, 2) l) 
  (h2 : areLinesPerpendicular l ⟨1, -3, 5⟩) : 
  l = ⟨3, 1, 1⟩ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2338_233837


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2338_233874

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l2338_233874


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2338_233876

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 - 8 * x^2 + d * x - 7
  (g 2 = -9) ∧ (g (-1) = -19) → c = 19/3 ∧ d = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2338_233876


namespace NUMINAMATH_CALUDE_small_square_area_l2338_233885

-- Define the tile and its components
def TileArea : ℝ := 49
def HypotenuseLength : ℝ := 5
def NumTriangles : ℕ := 8

-- Theorem statement
theorem small_square_area :
  ∀ (small_square_area : ℝ),
    small_square_area = TileArea - NumTriangles * (HypotenuseLength^2 / 2) →
    small_square_area = 1 :=
by sorry

end NUMINAMATH_CALUDE_small_square_area_l2338_233885


namespace NUMINAMATH_CALUDE_molecular_weight_C8H10N4O6_l2338_233835

/-- The atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Carbon atoms in C8H10N4O6 -/
def num_C : ℕ := 8

/-- The number of Hydrogen atoms in C8H10N4O6 -/
def num_H : ℕ := 10

/-- The number of Nitrogen atoms in C8H10N4O6 -/
def num_N : ℕ := 4

/-- The number of Oxygen atoms in C8H10N4O6 -/
def num_O : ℕ := 6

/-- The molecular weight of C8H10N4O6 in g/mol -/
def molecular_weight : ℝ :=
  num_C * atomic_weight_C +
  num_H * atomic_weight_H +
  num_N * atomic_weight_N +
  num_O * atomic_weight_O

theorem molecular_weight_C8H10N4O6 : molecular_weight = 258.22 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_C8H10N4O6_l2338_233835


namespace NUMINAMATH_CALUDE_parabola_vector_sum_implies_magnitude_sum_l2338_233867

noncomputable section

-- Define the parabola
def is_on_parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the vector from focus to a point
def vec_from_focus (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - focus.1, p.2 - focus.2)

-- Define the magnitude of a vector
def vec_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem parabola_vector_sum_implies_magnitude_sum
  (A B C : ℝ × ℝ)
  (hA : is_on_parabola A)
  (hB : is_on_parabola B)
  (hC : is_on_parabola C)
  (h_sum : vec_from_focus A + 2 • vec_from_focus B + 3 • vec_from_focus C = (0, 0)) :
  vec_magnitude (vec_from_focus A) + 2 * vec_magnitude (vec_from_focus B) + 3 * vec_magnitude (vec_from_focus C) = 12 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vector_sum_implies_magnitude_sum_l2338_233867


namespace NUMINAMATH_CALUDE_washers_remaining_l2338_233803

/-- Calculates the number of washers remaining after a plumbing job. -/
theorem washers_remaining
  (total_pipe_length : ℕ)
  (pipe_per_bolt : ℕ)
  (washers_per_bolt : ℕ)
  (initial_washers : ℕ)
  (h1 : total_pipe_length = 40)
  (h2 : pipe_per_bolt = 5)
  (h3 : washers_per_bolt = 2)
  (h4 : initial_washers = 20) :
  initial_washers - (total_pipe_length / pipe_per_bolt * washers_per_bolt) = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_washers_remaining_l2338_233803


namespace NUMINAMATH_CALUDE_womens_doubles_handshakes_l2338_233831

/-- The number of handshakes in a women's doubles tennis tournament -/
theorem womens_doubles_handshakes (num_teams : ℕ) (team_size : ℕ) : 
  num_teams = 4 → team_size = 2 → num_teams * team_size * (num_teams * team_size - team_size) / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_womens_doubles_handshakes_l2338_233831


namespace NUMINAMATH_CALUDE_p_nonneg_iff_equal_l2338_233882

/-- The polynomial p(x) defined in terms of a, b, and c -/
def p (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

/-- Theorem stating that p(x) is non-negative for all real x if and only if a = b = c -/
theorem p_nonneg_iff_equal (a b c : ℝ) :
  (∀ x : ℝ, p a b c x ≥ 0) ↔ (a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_p_nonneg_iff_equal_l2338_233882


namespace NUMINAMATH_CALUDE_polynomial_roots_l2338_233814

def p (x : ℝ) : ℝ := x^3 - 3*x^2 - 4*x + 12

theorem polynomial_roots :
  (∀ x : ℝ, p x = 0 ↔ x = 2 ∨ x = -2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2338_233814


namespace NUMINAMATH_CALUDE_equation_solution_l2338_233884

theorem equation_solution (x : ℚ) : 
  x ≠ -5 → ((x^2 + 3*x + 4) / (x + 5) = x + 6 ↔ x = -13/4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2338_233884


namespace NUMINAMATH_CALUDE_marias_paper_count_l2338_233836

theorem marias_paper_count : 
  ∀ (desk_sheets backpack_sheets : ℕ),
    desk_sheets = 50 →
    backpack_sheets = 41 →
    desk_sheets + backpack_sheets = 91 :=
by
  sorry

end NUMINAMATH_CALUDE_marias_paper_count_l2338_233836


namespace NUMINAMATH_CALUDE_larger_number_proof_l2338_233898

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 60) 
  (h2 : Nat.lcm a b = 60 * 11 * 15) : max a b = 900 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2338_233898


namespace NUMINAMATH_CALUDE_max_fraction_value_l2338_233807

theorem max_fraction_value : 
  ∃ (a b c d e f : ℕ), 
    a ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    f ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a / b + c / d) / (e / f) = 14 ∧
    ∀ (x y z w u v : ℕ),
      x ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      y ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      z ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      w ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      u ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      v ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ u ∧ x ≠ v ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ u ∧ y ≠ v ∧
      z ≠ w ∧ z ≠ u ∧ z ≠ v ∧
      w ≠ u ∧ w ≠ v ∧
      u ≠ v →
      (x / y + z / w) / (u / v) ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_value_l2338_233807


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l2338_233824

/-- The line L in the xy-plane --/
def line_L (m : ℝ) (x y : ℝ) : Prop :=
  5 * y + (2 * m - 4) * x - 10 * m = 0

/-- The rectangle OABC --/
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 10 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

/-- Point D on OA --/
def point_D (m : ℝ) : ℝ × ℝ := (0, 2 * m)

/-- Point E on BC --/
def point_E (m : ℝ) : ℝ × ℝ := (10, 8 - 2 * m)

/-- Area of quadrilateral ADEB --/
def area_ADEB (m : ℝ) : ℝ := 20

/-- Area of rectangle OABC --/
def area_OABC : ℝ := 60

/-- Parallel line that divides the rectangle into three equal areas --/
def parallel_line (m : ℝ) (x y : ℝ) : Prop :=
  y = ((4 - 2 * m) / 5) * x + (2 * m - 2)

theorem line_intersection_theorem (m : ℝ) :
  (1 ≤ m ∧ m ≤ 3) ∧
  (area_ADEB m = (1 / 3) * area_OABC) ∧
  (∀ x y, parallel_line m x y → 
    ∃ F G, F ∈ rectangle ∧ G ∈ rectangle ∧
    line_L m F.1 F.2 ∧ line_L m G.1 G.2 ∧
    area_ADEB m = area_OABC / 3) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_theorem_l2338_233824


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l2338_233864

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 8
  let total_democrats : ℕ := 10
  let subcommittee_republicans : ℕ := 3
  let subcommittee_democrats : ℕ := 2
  let ways_to_choose_republicans : ℕ := Nat.choose total_republicans subcommittee_republicans
  let ways_to_choose_chair : ℕ := total_democrats
  let ways_to_choose_other_democrat : ℕ := Nat.choose (total_democrats - 1) (subcommittee_democrats - 1)
  ways_to_choose_republicans * ways_to_choose_chair * ways_to_choose_other_democrat = 5040 :=
by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l2338_233864


namespace NUMINAMATH_CALUDE_bruce_fruit_shopping_l2338_233869

def grapes_quantity : ℝ := 8
def grapes_price : ℝ := 70
def mangoes_quantity : ℝ := 11
def mangoes_price : ℝ := 55
def oranges_quantity : ℝ := 5
def oranges_price : ℝ := 45
def apples_quantity : ℝ := 3
def apples_price : ℝ := 90
def cherries_quantity : ℝ := 4.5
def cherries_price : ℝ := 120

def total_cost : ℝ := grapes_quantity * grapes_price + 
                      mangoes_quantity * mangoes_price + 
                      oranges_quantity * oranges_price + 
                      apples_quantity * apples_price + 
                      cherries_quantity * cherries_price

theorem bruce_fruit_shopping : total_cost = 2200 := by
  sorry

end NUMINAMATH_CALUDE_bruce_fruit_shopping_l2338_233869


namespace NUMINAMATH_CALUDE_rhombus_transformations_l2338_233873

/-- Represents a point transformation on the plane -/
def PointTransformation := (ℤ × ℤ) → (ℤ × ℤ)

/-- Transformation of type (i) -/
def transform_i (α : ℤ) : PointTransformation :=
  λ (x, y) => (x, α * x + y)

/-- Transformation of type (ii) -/
def transform_ii (α : ℤ) : PointTransformation :=
  λ (x, y) => (x + α * y, y)

/-- A rhombus with integer-coordinate vertices -/
structure IntegerRhombus :=
  (v1 v2 v3 v4 : ℤ × ℤ)

/-- Checks if a quadrilateral is a square -/
def is_square (q : IntegerRhombus) : Prop := sorry

/-- Checks if a quadrilateral is a non-square rectangle -/
def is_non_square_rectangle (q : IntegerRhombus) : Prop := sorry

/-- Applies a series of transformations to a rhombus -/
def apply_transformations (r : IntegerRhombus) (ts : List PointTransformation) : IntegerRhombus := sorry

/-- Main theorem statement -/
theorem rhombus_transformations :
  (¬ ∃ (r : IntegerRhombus) (ts : List PointTransformation),
     is_square (apply_transformations r ts)) ∧
  (∃ (r : IntegerRhombus) (ts : List PointTransformation),
     is_non_square_rectangle (apply_transformations r ts)) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_transformations_l2338_233873


namespace NUMINAMATH_CALUDE_pizza_price_problem_l2338_233817

/-- Proves the price of large pizza slices given the conditions of the problem -/
theorem pizza_price_problem (small_price : ℕ) (total_slices : ℕ) (total_revenue : ℕ) (small_slices : ℕ) :
  small_price = 150 →
  total_slices = 5000 →
  total_revenue = 1050000 →
  small_slices = 2000 →
  (total_revenue - small_price * small_slices) / (total_slices - small_slices) = 250 := by
sorry

end NUMINAMATH_CALUDE_pizza_price_problem_l2338_233817


namespace NUMINAMATH_CALUDE_union_A_complement_B_equals_geq_neg_two_l2338_233812

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem union_A_complement_B_equals_geq_neg_two :
  A ∪ (U \ B) = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_equals_geq_neg_two_l2338_233812


namespace NUMINAMATH_CALUDE_sum_of_min_max_T_l2338_233839

theorem sum_of_min_max_T (B M T : ℝ) 
  (h1 : B^2 + M^2 + T^2 = 2022) 
  (h2 : B + M + T = 72) : 
  ∃ (Tmin Tmax : ℝ), 
    (∀ T' : ℝ, (∃ B' M' : ℝ, B'^2 + M'^2 + T'^2 = 2022 ∧ B' + M' + T' = 72) → Tmin ≤ T' ∧ T' ≤ Tmax) ∧
    Tmin + Tmax = 48 :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_max_T_l2338_233839


namespace NUMINAMATH_CALUDE_cars_per_row_in_section_G_l2338_233862

/-- The number of rows in Section G -/
def section_G_rows : ℕ := 15

/-- The number of rows in Section H -/
def section_H_rows : ℕ := 20

/-- The number of cars per row in Section H -/
def section_H_cars_per_row : ℕ := 9

/-- The number of cars Nate can walk past per minute -/
def cars_per_minute : ℕ := 11

/-- The number of minutes Nate spent searching -/
def search_time : ℕ := 30

/-- The number of cars per row in Section G -/
def section_G_cars_per_row : ℕ := 10

theorem cars_per_row_in_section_G :
  section_G_cars_per_row = 
    (cars_per_minute * search_time - section_H_rows * section_H_cars_per_row) / section_G_rows :=
by sorry

end NUMINAMATH_CALUDE_cars_per_row_in_section_G_l2338_233862


namespace NUMINAMATH_CALUDE_bucket_weight_l2338_233843

/-- Given a bucket with the following properties:
    1. When three-fourths full, it weighs p kilograms.
    2. When one-third full, it weighs q kilograms.
    This theorem states that when the bucket is five-sixths full, 
    it weighs (6p - q) / 5 kilograms. -/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let weight_three_fourths := p
  let weight_one_third := q
  let weight_five_sixths := (6 * p - q) / 5
  weight_five_sixths

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l2338_233843


namespace NUMINAMATH_CALUDE_ratio_problem_l2338_233849

theorem ratio_problem (A B C : ℝ) (h1 : A + B + C = 98) (h2 : A / B = 2 / 3) (h3 : B = 30) :
  B / C = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2338_233849


namespace NUMINAMATH_CALUDE_penalty_kicks_count_l2338_233825

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) : 
  total_players = 25 → goalies = 4 → (total_players - goalies) * goalies = 96 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kicks_count_l2338_233825


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2338_233891

/-- Represents a two-stage sampling process -/
structure TwoStageSampling where
  initial_count : ℕ
  excluded_count : ℕ
  selected_count : ℕ

/-- Calculates the probability of selection in a two-stage sampling process -/
def selection_probability (sampling : TwoStageSampling) : ℚ :=
  sampling.selected_count / (sampling.initial_count - sampling.excluded_count)

/-- Theorem stating that the selection probability is equal for all students and is 50/2000 -/
theorem equal_selection_probability (sampling : TwoStageSampling) 
  (h1 : sampling.initial_count = 2011)
  (h2 : sampling.excluded_count = 11)
  (h3 : sampling.selected_count = 50) :
  selection_probability sampling = 50 / 2000 := by
  sorry

#eval selection_probability ⟨2011, 11, 50⟩

end NUMINAMATH_CALUDE_equal_selection_probability_l2338_233891


namespace NUMINAMATH_CALUDE_trishul_investment_percentage_l2338_233808

/-- Proves that Trishul invested 10% less than Raghu -/
theorem trishul_investment_percentage (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →  -- Vishal invested 10% more than Trishul
  vishal + trishul + raghu = 6069 →  -- Total sum of investments
  raghu = 2100 →  -- Raghu's investment
  (raghu - trishul) / raghu = 0.1 :=  -- Trishul invested 10% less than Raghu
by sorry

end NUMINAMATH_CALUDE_trishul_investment_percentage_l2338_233808


namespace NUMINAMATH_CALUDE_soccer_ball_weight_l2338_233897

theorem soccer_ball_weight :
  ∀ (soccer_ball_weight bicycle_weight : ℝ),
    8 * soccer_ball_weight = 5 * bicycle_weight →
    4 * bicycle_weight = 120 →
    soccer_ball_weight = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_weight_l2338_233897


namespace NUMINAMATH_CALUDE_sin_pi_sixth_minus_two_alpha_l2338_233845

theorem sin_pi_sixth_minus_two_alpha (α : ℝ) 
  (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.sin (π / 6 - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_sixth_minus_two_alpha_l2338_233845


namespace NUMINAMATH_CALUDE_round_0_6457_to_hundredth_l2338_233815

/-- Rounds a number to the nearest hundredth -/
def roundToHundredth (x : ℚ) : ℚ :=
  (⌊x * 100 + 0.5⌋ : ℚ) / 100

/-- The theorem states that rounding 0.6457 to the nearest hundredth results in 0.65 -/
theorem round_0_6457_to_hundredth :
  roundToHundredth (6457 / 10000) = 65 / 100 := by sorry

end NUMINAMATH_CALUDE_round_0_6457_to_hundredth_l2338_233815


namespace NUMINAMATH_CALUDE_expand_expression_l2338_233883

theorem expand_expression (x y : ℝ) : -12 * (3 * x - 4 + 2 * y) = -36 * x + 48 - 24 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2338_233883


namespace NUMINAMATH_CALUDE_cubic_equality_solution_l2338_233852

theorem cubic_equality_solution : ∃ n : ℤ, 3^3 - 5 = 4^2 + n ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equality_solution_l2338_233852


namespace NUMINAMATH_CALUDE_rod_cutting_l2338_233878

/-- Given a rod of 17 meters long from which 20 pieces can be cut,
    prove that the length of each piece is 85 centimeters. -/
theorem rod_cutting (rod_length : ℝ) (num_pieces : ℕ) (piece_length_cm : ℝ) :
  rod_length = 17 →
  num_pieces = 20 →
  piece_length_cm = (rod_length / num_pieces) * 100 →
  piece_length_cm = 85 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l2338_233878


namespace NUMINAMATH_CALUDE_perpendicular_iff_m_eq_neg_two_thirds_l2338_233813

/-- Two lines in the cartesian plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line in the problem -/
def line1 (m : ℝ) : Line :=
  ⟨1, m + 1, m - 2⟩

/-- The second line in the problem -/
def line2 (m : ℝ) : Line :=
  ⟨m, 2, 8⟩

/-- The theorem to be proved -/
theorem perpendicular_iff_m_eq_neg_two_thirds (m : ℝ) :
  perpendicular (line1 m) (line2 m) ↔ m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_iff_m_eq_neg_two_thirds_l2338_233813


namespace NUMINAMATH_CALUDE_provisions_problem_l2338_233875

/-- The number of days the provisions last for the initial group -/
def initial_days : ℝ := 12

/-- The number of additional men joining the group -/
def additional_men : ℕ := 300

/-- The number of days the provisions last after the additional men join -/
def new_days : ℝ := 9.662337662337663

/-- The initial number of men in the group -/
def initial_men : ℕ := 1240

theorem provisions_problem :
  ∃ (M : ℝ), 
    (M ≥ 0) ∧ 
    (abs (M - initial_men) < 1) ∧
    (M * initial_days = (M + additional_men) * new_days) :=
by sorry

end NUMINAMATH_CALUDE_provisions_problem_l2338_233875


namespace NUMINAMATH_CALUDE_shirt_price_proof_l2338_233828

theorem shirt_price_proof (P : ℝ) : 
  (0.75 * (0.75 * P) = 18) → P = 32 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_proof_l2338_233828


namespace NUMINAMATH_CALUDE_expression_evaluation_l2338_233819

theorem expression_evaluation (x y : ℚ) (hx : x = -1) (hy : y = -1/3) :
  (3 * x^2 + x * y + 2 * y) - 2 * (5 * x * y - 4 * x^2 + y) = 8 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2338_233819


namespace NUMINAMATH_CALUDE_min_sum_of_squares_on_line_l2338_233887

theorem min_sum_of_squares_on_line :
  ∀ (x y : ℝ), x + y = 4 → ∀ (a b : ℝ), a + b = 4 → x^2 + y^2 ≤ a^2 + b^2 ∧ ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 4 ∧ x₀^2 + y₀^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_on_line_l2338_233887


namespace NUMINAMATH_CALUDE_insurance_coverage_percentage_l2338_233802

theorem insurance_coverage_percentage
  (frames_cost : ℝ)
  (lenses_cost : ℝ)
  (coupon_value : ℝ)
  (final_cost : ℝ)
  (h1 : frames_cost = 200)
  (h2 : lenses_cost = 500)
  (h3 : coupon_value = 50)
  (h4 : final_cost = 250) :
  (((frames_cost + lenses_cost - coupon_value) - final_cost) / lenses_cost) * 100 = 80 :=
by sorry

end NUMINAMATH_CALUDE_insurance_coverage_percentage_l2338_233802


namespace NUMINAMATH_CALUDE_inverse_function_symmetry_l2338_233826

def symmetric_about (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

theorem inverse_function_symmetry 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h₁ : Function.Bijective f) 
  (h₂ : Function.RightInverse g f) 
  (h₃ : Function.LeftInverse g f)
  (h₄ : symmetric_about f (0, 1)) : 
  ∀ a : ℝ, g a + g (2 - a) = 0 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_symmetry_l2338_233826


namespace NUMINAMATH_CALUDE_modified_pyramid_volume_l2338_233838

/-- Given a pyramid with a square base and volume of 60 cubic inches, 
    if the base side length is tripled and the height is decreased by 25%, 
    the new volume will be 405 cubic inches. -/
theorem modified_pyramid_volume 
  (s : ℝ) (h : ℝ) 
  (original_volume : (1/3 : ℝ) * s^2 * h = 60) 
  (s_positive : s > 0) 
  (h_positive : h > 0) : 
  (1/3 : ℝ) * (3*s)^2 * (0.75*h) = 405 :=
by sorry

end NUMINAMATH_CALUDE_modified_pyramid_volume_l2338_233838


namespace NUMINAMATH_CALUDE_remainder_123456789012_mod_180_l2338_233894

theorem remainder_123456789012_mod_180 : 123456789012 % 180 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_123456789012_mod_180_l2338_233894


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2338_233859

/-- Two arithmetic sequences and their sums -/
def arithmetic_sequences (a b : ℕ → ℝ) (A B : ℕ → ℝ) : Prop :=
  (∀ n, A n = (n * (a 1 + a n)) / 2) ∧
  (∀ n, B n = (n * (b 1 + b n)) / 2) ∧
  (∀ n, a (n + 1) - a n = a 2 - a 1) ∧
  (∀ n, b (n + 1) - b n = b 2 - b 1)

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℝ) (A B : ℕ → ℝ) 
  (h : arithmetic_sequences a b A B) 
  (h_ratio : ∀ n : ℕ, A n / B n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, a n / b n = (4 * n - 3) / (6 * n - 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2338_233859


namespace NUMINAMATH_CALUDE_factorial_236_trailing_zeros_l2338_233881

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 236! has 57 trailing zeros -/
theorem factorial_236_trailing_zeros :
  trailingZeros 236 = 57 := by sorry

end NUMINAMATH_CALUDE_factorial_236_trailing_zeros_l2338_233881


namespace NUMINAMATH_CALUDE_rain_period_end_time_l2338_233822

def start_time : ℕ := 8  -- 8 am
def rain_duration : ℕ := 4
def no_rain_duration : ℕ := 5

def total_duration : ℕ := rain_duration + no_rain_duration

def end_time : ℕ := start_time + total_duration

theorem rain_period_end_time :
  end_time = 17  -- 5 pm in 24-hour format
:= by sorry

end NUMINAMATH_CALUDE_rain_period_end_time_l2338_233822


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l2338_233872

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeBalls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 95 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distributeBalls 7 3 = 95 := by sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l2338_233872


namespace NUMINAMATH_CALUDE_horseback_trip_speed_l2338_233899

/-- The speed of Barry and Jim on the first day of their horseback riding trip -/
def first_day_speed : ℝ := 5

/-- The total distance traveled during the three-day trip -/
def total_distance : ℝ := 115

/-- The duration of travel on the first day -/
def first_day_duration : ℝ := 7

/-- The distance traveled on the second day -/
def second_day_distance : ℝ := 36 + 9

/-- The distance traveled on the third day -/
def third_day_distance : ℝ := 35

theorem horseback_trip_speed :
  first_day_speed * first_day_duration + second_day_distance + third_day_distance = total_distance :=
sorry

end NUMINAMATH_CALUDE_horseback_trip_speed_l2338_233899


namespace NUMINAMATH_CALUDE_regular_hexagon_interior_angle_measure_l2338_233855

/-- The measure of an interior angle of a regular hexagon -/
def regular_hexagon_interior_angle : ℝ := 120

/-- A regular hexagon has 6 sides -/
def regular_hexagon_sides : ℕ := 6

/-- Theorem: The measure of each interior angle of a regular hexagon is 120 degrees -/
theorem regular_hexagon_interior_angle_measure :
  regular_hexagon_interior_angle = (((regular_hexagon_sides - 2) * 180) : ℝ) / regular_hexagon_sides :=
by sorry

end NUMINAMATH_CALUDE_regular_hexagon_interior_angle_measure_l2338_233855


namespace NUMINAMATH_CALUDE_complex_number_location_l2338_233801

theorem complex_number_location (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (1 + i) / i
  z = 1 - i ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2338_233801


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2338_233840

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (10, 6)

/-- The line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - Q.2 = m * (p.1 - Q.1)}

/-- The condition for non-intersection -/
def no_intersection (m : ℝ) : Prop :=
  line_through_Q m ∩ P = ∅

theorem parabola_line_intersection :
  ∃ (r s : ℝ), (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) → r + s = 40 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2338_233840


namespace NUMINAMATH_CALUDE_lucas_future_age_l2338_233846

def age_problem (gladys_age billy_age lucas_age : ℕ) : Prop :=
  (gladys_age = 30) ∧
  (gladys_age = 3 * billy_age) ∧
  (gladys_age = 2 * (billy_age + lucas_age))

theorem lucas_future_age 
  (gladys_age billy_age lucas_age : ℕ) 
  (h : age_problem gladys_age billy_age lucas_age) : 
  lucas_age + 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_lucas_future_age_l2338_233846


namespace NUMINAMATH_CALUDE_defective_units_shipped_percentage_l2338_233857

theorem defective_units_shipped_percentage
  (total_units : ℝ)
  (defective_rate : ℝ)
  (defective_shipped_rate : ℝ)
  (h1 : defective_rate = 0.1)
  (h2 : defective_shipped_rate = 0.005)
  (h3 : total_units > 0) :
  (defective_shipped_rate / defective_rate) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_defective_units_shipped_percentage_l2338_233857


namespace NUMINAMATH_CALUDE_sum_of_max_min_is_negative_one_l2338_233880

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem sum_of_max_min_is_negative_one :
  ∃ (max min : ℝ), 
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max + min = -1 := by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_is_negative_one_l2338_233880


namespace NUMINAMATH_CALUDE_symmetric_points_l2338_233834

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

/-- The theorem stating that (4, -3) is symmetric to (-4, 3) with respect to the origin -/
theorem symmetric_points : symmetric_wrt_origin (4, -3) (-4, 3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_l2338_233834


namespace NUMINAMATH_CALUDE_mikes_muffins_l2338_233863

/-- The number of muffins in a dozen -/
def dozen : ℕ := 12

/-- The number of boxes Mike needs to pack all his muffins -/
def boxes : ℕ := 8

/-- Mike's muffins theorem -/
theorem mikes_muffins : dozen * boxes = 96 := by
  sorry

end NUMINAMATH_CALUDE_mikes_muffins_l2338_233863


namespace NUMINAMATH_CALUDE_sarah_toy_cars_l2338_233888

def initial_amount : ℕ := 53
def toy_car_cost : ℕ := 11
def scarf_cost : ℕ := 10
def beanie_cost : ℕ := 14
def remaining_amount : ℕ := 7

theorem sarah_toy_cars :
  ∃ (num_cars : ℕ),
    num_cars * toy_car_cost + scarf_cost + beanie_cost = initial_amount - remaining_amount ∧
    num_cars = 2 :=
by sorry

end NUMINAMATH_CALUDE_sarah_toy_cars_l2338_233888


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l2338_233842

open Real

noncomputable def f (x : ℝ) : ℝ := exp x * cos x

theorem tangent_slope_angle_at_zero (α : ℝ) :
  (∀ x, HasDerivAt f (exp x * (cos x - sin x)) x) →
  HasDerivAt f 1 0 →
  0 ≤ α →
  α < π →
  tan α = 1 →
  α = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_zero_l2338_233842


namespace NUMINAMATH_CALUDE_rest_area_location_l2338_233829

theorem rest_area_location (city_a city_b rest_area : ℝ) : 
  city_a = 50 →
  city_b = 230 →
  rest_area - city_a = (5/8) * (city_b - city_a) →
  rest_area = 162.5 := by
sorry

end NUMINAMATH_CALUDE_rest_area_location_l2338_233829


namespace NUMINAMATH_CALUDE_base_b_is_four_l2338_233850

theorem base_b_is_four : 
  ∃ (b : ℕ), 
    b > 0 ∧ 
    (b - 1) * (b - 1) * b = 72 ∧ 
    b = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_b_is_four_l2338_233850


namespace NUMINAMATH_CALUDE_more_than_half_inside_l2338_233861

/-- A triangle with an inscribed circle -/
structure InscribedTriangle where
  /-- The triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The inscribed circle -/
  circle : Set (ℝ × ℝ)
  /-- The circle is inscribed in the triangle -/
  inscribed : circle ⊆ triangle

/-- A square circumscribed around a circle -/
structure CircumscribedSquare where
  /-- The square -/
  square : Set (ℝ × ℝ)
  /-- The circumscribed circle -/
  circle : Set (ℝ × ℝ)
  /-- The square is circumscribed around the circle -/
  circumscribed : circle ⊆ square

/-- The perimeter of a square -/
def squarePerimeter (s : CircumscribedSquare) : ℝ := sorry

/-- The length of the square's perimeter segments inside the triangle -/
def insidePerimeterLength (t : InscribedTriangle) (s : CircumscribedSquare) : ℝ := sorry

/-- Main theorem: More than half of the square's perimeter is inside the triangle -/
theorem more_than_half_inside (t : InscribedTriangle) (s : CircumscribedSquare) 
  (h : t.circle = s.circle) : 
  insidePerimeterLength t s > squarePerimeter s / 2 := by sorry

end NUMINAMATH_CALUDE_more_than_half_inside_l2338_233861


namespace NUMINAMATH_CALUDE_arrangement_count_l2338_233806

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of teachers in each group -/
def teachers_per_group : ℕ := 1

/-- The number of students in each group -/
def students_per_group : ℕ := 2

/-- The number of groups -/
def num_groups : ℕ := 2

/-- The total number of arrangements -/
def total_arrangements : ℕ := 12

theorem arrangement_count :
  (Nat.choose num_teachers teachers_per_group) *
  (Nat.choose num_students students_per_group) = total_arrangements :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l2338_233806


namespace NUMINAMATH_CALUDE_adult_admission_price_l2338_233854

theorem adult_admission_price
  (total_people : ℕ)
  (total_receipts : ℕ)
  (num_children : ℕ)
  (child_price : ℕ)
  (h1 : total_people = 610)
  (h2 : total_receipts = 960)
  (h3 : num_children = 260)
  (h4 : child_price = 1) :
  (total_receipts - num_children * child_price) / (total_people - num_children) = 2 :=
by sorry

end NUMINAMATH_CALUDE_adult_admission_price_l2338_233854


namespace NUMINAMATH_CALUDE_sum_of_triangles_eq_22_l2338_233830

/-- Represents the value of a triangle with vertices a, b, and c -/
def triangle_value (a b c : ℕ) : ℕ := a * b + c

/-- The sum of the values of two specific triangles -/
def sum_of_triangles : ℕ :=
  triangle_value 3 2 5 + triangle_value 4 1 7

theorem sum_of_triangles_eq_22 : sum_of_triangles = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangles_eq_22_l2338_233830


namespace NUMINAMATH_CALUDE_unique_solution_l2338_233820

/-- Represents the ages of the grandchildren --/
structure GrandchildrenAges where
  martinka : ℕ
  tomasek : ℕ
  jaromir : ℕ
  kacka : ℕ
  ida : ℕ
  verka : ℕ

/-- The conditions given in the problem --/
def satisfiesConditions (ages : GrandchildrenAges) : Prop :=
  ages.martinka = ages.tomasek + 8 ∧
  ages.verka = ages.ida + 7 ∧
  ages.martinka = ages.jaromir + 1 ∧
  ages.kacka = ages.tomasek + 11 ∧
  ages.jaromir = ages.ida + 4 ∧
  ages.tomasek + ages.jaromir = 13

/-- The theorem stating that there is a unique solution satisfying all conditions --/
theorem unique_solution : ∃! ages : GrandchildrenAges, satisfiesConditions ages ∧
  ages.martinka = 11 ∧
  ages.tomasek = 3 ∧
  ages.jaromir = 10 ∧
  ages.kacka = 14 ∧
  ages.ida = 6 ∧
  ages.verka = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2338_233820


namespace NUMINAMATH_CALUDE_gravitational_force_on_space_station_l2338_233847

/-- Gravitational force model -/
structure GravitationalModel where
  k : ℝ
  force : ℝ → ℝ
  h_inverse_square : ∀ d, d > 0 → force d = k / (d^2)

/-- Problem statement -/
theorem gravitational_force_on_space_station
  (model : GravitationalModel)
  (h_surface : model.force 6000 = 800)
  : model.force 360000 = 2/9 := by
  sorry


end NUMINAMATH_CALUDE_gravitational_force_on_space_station_l2338_233847


namespace NUMINAMATH_CALUDE_difference_m_n_l2338_233893

theorem difference_m_n (m n : ℕ+) (h : 10 * 2^(m : ℕ) = 2^(n : ℕ) + 2^((n : ℕ) + 2)) :
  (n : ℕ) - (m : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_m_n_l2338_233893


namespace NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l2338_233896

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 10) + Complex.abs (z + 3*I) = 17) :
  ∃ (w : ℂ), Complex.abs (z - 10) + Complex.abs (z + 3*I) = 17 ∧ Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 30/17 :=
by sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l2338_233896


namespace NUMINAMATH_CALUDE_wire_service_coverage_l2338_233858

/-- The percentage of reporters covering local politics in country x -/
def local_politics_coverage : ℝ := 12

/-- The percentage of reporters not covering politics -/
def non_politics_coverage : ℝ := 80

/-- The percentage of reporters covering politics but not local politics in country x -/
def politics_not_local : ℝ := 40

/-- Theorem stating that given the conditions, the percentage of reporters
    who cover politics but not local politics in country x is 40% -/
theorem wire_service_coverage :
  local_politics_coverage = 12 →
  non_politics_coverage = 80 →
  politics_not_local = 40 :=
by
  sorry

#check wire_service_coverage

end NUMINAMATH_CALUDE_wire_service_coverage_l2338_233858


namespace NUMINAMATH_CALUDE_f_zero_range_l2338_233810

/-- The function f(x) = x^3 + 2x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2*x - a

/-- The theorem stating that if f(x) has exactly one zero in (1, 2), then a is in (3, 12) -/
theorem f_zero_range (a : ℝ) : 
  (∃! x, x ∈ (Set.Ioo 1 2) ∧ f a x = 0) → a ∈ Set.Ioo 3 12 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_range_l2338_233810


namespace NUMINAMATH_CALUDE_weekly_earnings_calculation_l2338_233800

/- Define the basic fees and attendance -/
def kidFee : ℚ := 3
def adultFee : ℚ := 6
def weekdayKids : ℕ := 8
def weekdayAdults : ℕ := 10
def weekendKids : ℕ := 12
def weekendAdults : ℕ := 15

/- Define the discounts and special rates -/
def weekendRate : ℚ := 1.5
def groupDiscountRate : ℚ := 0.8
def membershipDiscountRate : ℚ := 0.9
def weekdayGroupBookings : ℕ := 2
def weekendMemberships : ℕ := 8

/- Calculate earnings -/
def weekdayEarnings : ℚ := 5 * (weekdayKids * kidFee + weekdayAdults * adultFee)
def weekendEarnings : ℚ := 2 * (weekendKids * kidFee * weekendRate + weekendAdults * adultFee * weekendRate)

/- Calculate discounts -/
def weekdayGroupDiscount : ℚ := 5 * weekdayGroupBookings * (kidFee + adultFee) * (1 - groupDiscountRate)
def weekendMembershipDiscount : ℚ := 2 * weekendMemberships * adultFee * weekendRate * (1 - membershipDiscountRate)

/- Define the total weekly earnings -/
def totalWeeklyEarnings : ℚ := weekdayEarnings + weekendEarnings - weekdayGroupDiscount - weekendMembershipDiscount

/- The theorem to prove -/
theorem weekly_earnings_calculation : totalWeeklyEarnings = 738.6 := by
  sorry


end NUMINAMATH_CALUDE_weekly_earnings_calculation_l2338_233800


namespace NUMINAMATH_CALUDE_intersection_slope_l2338_233832

/-- Given two lines p and q that intersect at (1, 1), prove that the slope of q is -3 -/
theorem intersection_slope (k : ℝ) : 
  (∀ x y : ℝ, y = -2*x + 3 → y = k*x + 4) → -- Line p: y = -2x + 3, Line q: y = kx + 4
  1 = -2*1 + 3 →                            -- (1, 1) satisfies line p
  1 = k*1 + 4 →                             -- (1, 1) satisfies line q
  k = -3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_l2338_233832


namespace NUMINAMATH_CALUDE_removed_term_is_last_l2338_233877

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) : ℕ → ℚ :=
  fun n => a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem removed_term_is_last
  (a₁ : ℚ)
  (avg_11 : ℚ)
  (avg_10 : ℚ)
  (h₁ : a₁ = -5)
  (h₂ : avg_11 = 5)
  (h₃ : avg_10 = 4)
  (h₄ : ∃ d : ℚ, sum_arithmetic_sequence a₁ d 11 = 11 * avg_11) :
  arithmetic_sequence a₁ ((sum_arithmetic_sequence a₁ 2 11 - sum_arithmetic_sequence a₁ 2 10) / 1) 11 =
  sum_arithmetic_sequence a₁ 2 11 - 10 * avg_10 :=
by sorry

end NUMINAMATH_CALUDE_removed_term_is_last_l2338_233877


namespace NUMINAMATH_CALUDE_special_function_sum_property_l2338_233827

/-- A function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (f 0 = 0) ∧
  (∀ x y, x ∈ ({x | x < -1} ∪ {x | x > 1}) → 
          y ∈ ({x | x < -1} ∪ {x | x > 1}) → 
          f (1/x) + f (1/y) = f ((x+y)/(1+x*y))) ∧
  (∀ x, x ∈ {x | -1 < x ∧ x < 0} → f x > 0)

/-- The theorem to be proved -/
theorem special_function_sum_property (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∑' (n : ℕ), f (1 / (n^2 + 7*n + 11)) > f (1/2) := by
  sorry

end NUMINAMATH_CALUDE_special_function_sum_property_l2338_233827


namespace NUMINAMATH_CALUDE_petes_number_l2338_233890

theorem petes_number : ∃ x : ℚ, 3 * (3 * x - 5) = 96 ∧ x = 111 / 9 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l2338_233890


namespace NUMINAMATH_CALUDE_clown_count_l2338_233860

/-- The number of clown mobiles -/
def num_mobiles : ℕ := 357

/-- The number of clowns in each mobile -/
def clowns_per_mobile : ℕ := 842

/-- The total number of clowns in all mobiles -/
def total_clowns : ℕ := num_mobiles * clowns_per_mobile

theorem clown_count : total_clowns = 300534 := by
  sorry

end NUMINAMATH_CALUDE_clown_count_l2338_233860


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_2772_l2338_233809

/-- The product of the digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Check if a number is a five-digit integer -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_with_product_2772 :
  ∀ n : ℕ, is_five_digit n → digit_product n = 2772 → n ≤ 98721 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_2772_l2338_233809


namespace NUMINAMATH_CALUDE_complex_multiplication_l2338_233804

theorem complex_multiplication : (1 + Complex.I) * (2 - Complex.I) = 3 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2338_233804


namespace NUMINAMATH_CALUDE_function_inequality_and_logarithm_comparison_l2338_233879

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

-- State the theorem
theorem function_inequality_and_logarithm_comparison (m : ℝ) 
  (h : ∀ x : ℝ, (1 / m) - 4 ≥ f m x) : 
  m > 0 ∧ Real.log (m + 2) / Real.log (m + 1) > Real.log (m + 3) / Real.log (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_and_logarithm_comparison_l2338_233879


namespace NUMINAMATH_CALUDE_charging_pile_growth_l2338_233853

/-- Represents the growth of smart charging piles over two months -/
theorem charging_pile_growth 
  (initial_count : ℕ) 
  (final_count : ℕ) 
  (growth_rate : ℝ) 
  (h1 : initial_count = 301)
  (h2 : final_count = 500)
  : initial_count * (1 + growth_rate)^2 = final_count := by
  sorry

#check charging_pile_growth

end NUMINAMATH_CALUDE_charging_pile_growth_l2338_233853


namespace NUMINAMATH_CALUDE_exists_index_sum_inequality_l2338_233889

theorem exists_index_sum_inequality (a : Fin 100 → ℝ) 
  (h_distinct : ∀ i j : Fin 100, i ≠ j → a i ≠ a j) :
  ∃ i : Fin 100, a i + a ((i + 3) % 100) > a ((i + 1) % 100) + a ((i + 2) % 100) := by
  sorry

end NUMINAMATH_CALUDE_exists_index_sum_inequality_l2338_233889


namespace NUMINAMATH_CALUDE_consecutive_integers_with_unique_prime_factors_l2338_233848

theorem consecutive_integers_with_unique_prime_factors (n : ℕ) (hn : n > 0) :
  ∃ x : ℤ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    ∃ (p : ℕ) (k : ℕ), Prime p ∧ (x + i : ℤ) = p * k ∧ ¬(p ∣ k) :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_with_unique_prime_factors_l2338_233848


namespace NUMINAMATH_CALUDE_computer_game_cost_l2338_233886

/-- The cost of the computer game Mr. Grey purchased, given the following conditions:
  * He bought 3 polo shirts for $26 each
  * He bought 2 necklaces for $83 each
  * He received a $12 rebate
  * The total cost after the rebate was $322
-/
theorem computer_game_cost : ℕ := by
  let polo_shirt_cost : ℕ := 26
  let polo_shirt_count : ℕ := 3
  let necklace_cost : ℕ := 83
  let necklace_count : ℕ := 2
  let rebate : ℕ := 12
  let total_cost_after_rebate : ℕ := 322

  have h1 : polo_shirt_cost * polo_shirt_count + necklace_cost * necklace_count + 90 = total_cost_after_rebate + rebate := by sorry

  exact 90

end NUMINAMATH_CALUDE_computer_game_cost_l2338_233886


namespace NUMINAMATH_CALUDE_mo_drinking_difference_l2338_233871

/-- Mo's drinking habits and last week's data --/
structure MoDrinkingData where
  n : ℕ  -- Number of hot chocolate cups on rainy days
  total_cups : ℕ  -- Total cups of tea and hot chocolate last week
  rainy_days : ℕ  -- Number of rainy days last week

/-- Theorem stating the difference between tea and hot chocolate cups --/
theorem mo_drinking_difference (data : MoDrinkingData) : 
  data.n ≤ 2 ∧ 
  data.total_cups = 20 ∧ 
  data.rainy_days = 2 → 
  (7 - data.rainy_days) * 3 - data.rainy_days * data.n = 11 := by
  sorry

#check mo_drinking_difference

end NUMINAMATH_CALUDE_mo_drinking_difference_l2338_233871
