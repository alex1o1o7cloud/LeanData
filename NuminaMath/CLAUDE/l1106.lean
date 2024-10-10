import Mathlib

namespace distance_to_point_l1106_110650

theorem distance_to_point : Real.sqrt (9^2 + (-40)^2) = 41 := by sorry

end distance_to_point_l1106_110650


namespace equation_is_parabola_and_ellipse_l1106_110647

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation y^4 - 6x^4 = 3y^2 + 1 -/
def equation (p : Point2D) : Prop :=
  p.y^4 - 6*p.x^4 = 3*p.y^2 + 1

/-- Represents a parabola in 2D space -/
def isParabola (S : Set Point2D) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ p ∈ S, p.y = a*p.x^2 + b*p.x + c

/-- Represents an ellipse in 2D space -/
def isEllipse (S : Set Point2D) : Prop :=
  ∃ h k a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ p ∈ S, 
    ((p.x - h)^2 / a^2) + ((p.y - k)^2 / b^2) = 1

/-- The set of all points satisfying the equation -/
def S : Set Point2D :=
  {p : Point2D | equation p}

/-- Theorem stating that the equation represents the union of a parabola and an ellipse -/
theorem equation_is_parabola_and_ellipse :
  ∃ P E : Set Point2D, isParabola P ∧ isEllipse E ∧ S = P ∪ E :=
sorry

end equation_is_parabola_and_ellipse_l1106_110647


namespace find_a_l1106_110637

/-- The value of a that satisfies the given inequality system -/
def a : ℝ := 4

/-- The system of inequalities -/
def inequality_system (x a : ℝ) : Prop :=
  2 * x + 1 > 3 ∧ a - x > 1

/-- The solution set of the inequality system -/
def solution_set (x : ℝ) : Prop :=
  1 < x ∧ x < 3

theorem find_a :
  (∀ x, inequality_system x a ↔ solution_set x) →
  a = 4 := by sorry

end find_a_l1106_110637


namespace total_ants_l1106_110660

def ants_problem (abe beth cece duke emily frances : ℕ) : Prop :=
  abe = 4 ∧
  beth = 2 * abe ∧
  cece = 3 * abe ∧
  duke = abe / 2 ∧
  emily = abe + (75 * abe) / 100 ∧
  frances = 2 * cece ∧
  abe + beth + cece + duke + emily + frances = 57

theorem total_ants : ∃ (abe beth cece duke emily frances : ℕ),
  ants_problem abe beth cece duke emily frances :=
sorry

end total_ants_l1106_110660


namespace c_wins_probability_l1106_110627

/-- Represents a player in the backgammon tournament -/
inductive Player := | A | B | C

/-- Represents the state of the tournament -/
structure TournamentState where
  lastWinner : Player
  lastLoser : Player

/-- The probability of a player winning a single game -/
def winProbability : ℚ := 1 / 2

/-- The probability of player C winning the tournament -/
def probCWins : ℚ := 2 / 7

/-- Theorem stating that the probability of player C winning the tournament is 2/7 -/
theorem c_wins_probability : 
  probCWins = 2 / 7 := by sorry

end c_wins_probability_l1106_110627


namespace max_rectangle_area_l1106_110670

/-- The equation that the vertices' coordinates must satisfy -/
def vertex_equation (x y : ℝ) : Prop :=
  |y + 1| * (y^2 + 2*y + 28) + |x - 2| = 9 * (y^2 + 2*y + 4)

/-- The area function of the rectangle -/
def rectangle_area (x : ℝ) : ℝ :=
  -4 * x * (x - 3)^3

/-- The theorem stating the maximum area of the rectangle -/
theorem max_rectangle_area :
  ∃ (x y : ℝ), vertex_equation x y ∧
  ∀ (x' y' : ℝ), vertex_equation x' y' → rectangle_area x ≥ rectangle_area x' ∧
  rectangle_area x = 34.171875 := by
sorry

end max_rectangle_area_l1106_110670


namespace a_range_when_p_false_l1106_110612

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Ioo 2 3, x^2 + 5 > a*x

-- Define the range of a
def a_range : Set ℝ := Set.Ici (2 * Real.sqrt 5)

-- Theorem statement
theorem a_range_when_p_false :
  (∃ a : ℝ, ¬(p a)) ↔ ∃ a ∈ a_range, True :=
sorry

end a_range_when_p_false_l1106_110612


namespace calculate_expression_l1106_110622

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end calculate_expression_l1106_110622


namespace arithmetic_sequence_inequality_l1106_110693

theorem arithmetic_sequence_inequality (a₁ : ℝ) (d : ℝ) :
  (∀ n : Fin 8, a₁ + (n : ℕ) * d > 0) →
  d ≠ 0 →
  (a₁ * (a₁ + 7 * d)) < ((a₁ + 3 * d) * (a₁ + 4 * d)) := by
  sorry

end arithmetic_sequence_inequality_l1106_110693


namespace quadratic_equation_general_form_l1106_110696

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (x + 3) * (x - 1) = 2 * x - 4 ↔ x^2 + 1 = 0 := by sorry

end quadratic_equation_general_form_l1106_110696


namespace unique_solution_system_l1106_110649

theorem unique_solution_system (x y : ℝ) :
  (2 * x + y + 8 ≤ 0) ∧
  (x^4 + 2 * x^2 * y^2 + y^4 + 9 - 10 * x^2 - 10 * y^2 = 8 * x * y) →
  x = -3 ∧ y = -2 :=
by sorry

end unique_solution_system_l1106_110649


namespace circle_tangent_and_shortest_chord_l1106_110698

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define points P and M
def P : ℝ × ℝ := (2, 5)
def M : ℝ × ℝ := (5, 0)

-- Define the line with shortest chord length
def shortest_chord_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the tangent lines
def tangent_line1 (x y : ℝ) : Prop := 3*x + 4*y - 15 = 0
def tangent_line2 (x : ℝ) : Prop := x = 5

theorem circle_tangent_and_shortest_chord :
  (∀ x y, C x y → shortest_chord_line x y → (x, y) = P ∨ (C x y ∧ shortest_chord_line x y)) ∧
  (∀ x y, C x y → tangent_line1 x y → (x, y) = M ∨ (C x y ∧ tangent_line1 x y)) ∧
  (∀ x y, C x y → tangent_line2 x → (x, y) = M ∨ (C x y ∧ tangent_line2 x)) := by
  sorry

end circle_tangent_and_shortest_chord_l1106_110698


namespace simplify_expression_l1106_110621

theorem simplify_expression (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ -b) :
  let M := a - b
  (2 * a) / (a^2 - b^2) - 1 / M = 1 / (a + b) := by
sorry

end simplify_expression_l1106_110621


namespace valid_basis_l1106_110626

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (3, -1)
def a : ℝ × ℝ := (3, 4)

theorem valid_basis :
  ∃ (x y : ℝ), x • e₁ + y • e₂ = a ∧ ¬(∃ (k : ℝ), e₁ = k • e₂) :=
sorry

end valid_basis_l1106_110626


namespace complement_A_intersection_B_range_l1106_110681

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- Define set B with parameter a
def B (a : ℝ) : Set ℝ := {x : ℝ | |x| > a}

-- Define the complement of A with respect to U
def complementA : Set ℝ := Set.Icc (-1) 3

theorem complement_A_intersection_B_range :
  (∀ a : ℝ, (complementA ∩ B a).Nonempty) ↔ a ∈ Set.Icc 0 2 :=
sorry

end complement_A_intersection_B_range_l1106_110681


namespace hockey_league_games_l1106_110625

/-- Represents a hockey league with two divisions -/
structure HockeyLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of games played in the hockey league -/
def total_games (league : HockeyLeague) : Nat :=
  let intra_games := league.divisions * (league.teams_per_division * (league.teams_per_division - 1) / 2) * league.intra_division_games
  let inter_games := league.divisions * league.teams_per_division * league.teams_per_division * league.inter_division_games
  intra_games + inter_games

/-- Theorem stating that the total number of games in the described hockey league is 192 -/
theorem hockey_league_games :
  let league : HockeyLeague := {
    divisions := 2,
    teams_per_division := 6,
    intra_division_games := 4,
    inter_division_games := 2
  }
  total_games league = 192 := by sorry

end hockey_league_games_l1106_110625


namespace deal_or_no_deal_elimination_l1106_110678

theorem deal_or_no_deal_elimination (total_boxes : ℕ) (high_value_boxes : ℕ) 
  (elimination_target : ℚ) :
  total_boxes = 30 →
  high_value_boxes = 9 →
  elimination_target = 1/3 →
  ∃ (boxes_to_eliminate : ℕ),
    boxes_to_eliminate = 3 ∧
    (total_boxes - boxes_to_eliminate : ℚ) * elimination_target ≤ high_value_boxes ∧
    ∀ (n : ℕ), n < boxes_to_eliminate →
      (total_boxes - n : ℚ) * elimination_target > high_value_boxes :=
by sorry

end deal_or_no_deal_elimination_l1106_110678


namespace simplify_expression_l1106_110676

theorem simplify_expression (a : ℝ) : (a + 4) * (a - 4) - (a - 1)^2 = 2 * a - 17 := by
  sorry

end simplify_expression_l1106_110676


namespace coefficients_of_equation_l1106_110677

/-- Given a quadratic equation ax² + bx + c = 0, this function returns its coefficients (a, b, c) -/
def quadratic_coefficients (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

/-- The coefficients of the quadratic equation 4x² - 6x + 1 = 0 are (4, -6, 1) -/
theorem coefficients_of_equation : quadratic_coefficients 4 (-6) 1 = (4, -6, 1) := by sorry

end coefficients_of_equation_l1106_110677


namespace rectangle_length_l1106_110628

/-- Given a rectangle where the length is three times the breadth and the area is 6075 square meters,
    prove that the length of the rectangle is 135 meters. -/
theorem rectangle_length (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth → 
  area = length * breadth → 
  area = 6075 → 
  length = 135 := by sorry

end rectangle_length_l1106_110628


namespace expansion_binomial_coefficients_l1106_110663

theorem expansion_binomial_coefficients (n : ℕ) : 
  (∃ a d : ℚ, (n.choose 1 : ℚ) = a ∧ 
               (n.choose 2 : ℚ) = a + d ∧ 
               (n.choose 3 : ℚ) = a + 2*d) → 
  n = 7 := by
sorry

end expansion_binomial_coefficients_l1106_110663


namespace base10_729_equals_base7_261_l1106_110658

-- Define a function to convert a base-7 number to base-10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

-- Define the base-7 representation of 261₇
def base7_261 : List Nat := [1, 6, 2]

-- Theorem statement
theorem base10_729_equals_base7_261 :
  base7ToBase10 base7_261 = 729 := by
  sorry

end base10_729_equals_base7_261_l1106_110658


namespace min_distance_to_vertex_l1106_110619

/-- A right circular cone with base radius 1 and slant height 3 -/
structure Cone where
  base_radius : ℝ
  slant_height : ℝ
  base_radius_eq : base_radius = 1
  slant_height_eq : slant_height = 3

/-- A point on the shortest path between two points on the base circumference -/
def ShortestPathPoint (c : Cone) := ℝ

/-- The distance from the vertex to a point on the shortest path -/
def distance_to_vertex (c : Cone) (p : ShortestPathPoint c) : ℝ := sorry

/-- The theorem stating the minimum distance from the vertex to a point on the shortest path -/
theorem min_distance_to_vertex (c : Cone) : 
  ∃ (p : ShortestPathPoint c), distance_to_vertex c p = 3/2 ∧ 
  ∀ (q : ShortestPathPoint c), distance_to_vertex c q ≥ 3/2 := by sorry

end min_distance_to_vertex_l1106_110619


namespace distinct_selections_count_l1106_110629

/-- Represents the counts of each letter in "MATHEMATICAL" --/
structure LetterCounts where
  a : Nat
  e : Nat
  i : Nat
  m : Nat
  t : Nat
  h : Nat
  c : Nat
  l : Nat

/-- The initial letter counts in "MATHEMATICAL" --/
def initial_counts : LetterCounts := {
  a := 3, e := 1, i := 1, m := 2, t := 2, h := 1, c := 1, l := 1
}

/-- Counts the number of distinct ways to choose 3 vowels and 4 consonants
    from the word "MATHEMATICAL" with indistinguishable T's, M's, and A's --/
def count_distinct_selections (counts : LetterCounts) : Nat :=
  sorry

theorem distinct_selections_count :
  count_distinct_selections initial_counts = 64 := by
  sorry

end distinct_selections_count_l1106_110629


namespace gcd_547_323_l1106_110606

theorem gcd_547_323 : Nat.gcd 547 323 = 1 := by
  sorry

end gcd_547_323_l1106_110606


namespace product_equals_2010_l1106_110624

def sequence_product (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (n + 1 : ℚ) / n * sequence_product (n - 1)

theorem product_equals_2010 :
  sequence_product 2009 = 2010 := by
  sorry

end product_equals_2010_l1106_110624


namespace unique_quadratic_solution_l1106_110630

theorem unique_quadratic_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b ≠ 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 3/b) * x + c = 0) ↔ 
  c = 3/2 := by
sorry

end unique_quadratic_solution_l1106_110630


namespace distance_after_four_hours_l1106_110656

/-- The distance between two students walking in opposite directions -/
def distance_between_students (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem: The distance between two students walking in opposite directions for 4 hours,
    with speeds of 6 km/hr and 9 km/hr respectively, is 60 km. -/
theorem distance_after_four_hours :
  distance_between_students 6 9 4 = 60 := by
  sorry

#eval distance_between_students 6 9 4

end distance_after_four_hours_l1106_110656


namespace log_cube_exp_inequality_l1106_110610

theorem log_cube_exp_inequality (x : ℝ) (h : 0 < x ∧ x < 1) :
  Real.log x / Real.log 3 < x^3 ∧ x^3 < 3^x := by
  sorry

end log_cube_exp_inequality_l1106_110610


namespace A_intersection_B_equals_A_l1106_110692

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define set A
def A : Set ℝ := {x | f x = x}

-- Define set B
def B : Set ℝ := {x | f (f x) = x}

-- Theorem statement
theorem A_intersection_B_equals_A : A ∩ B = A := by sorry

end A_intersection_B_equals_A_l1106_110692


namespace tan_15_degrees_l1106_110615

theorem tan_15_degrees : Real.tan (15 * π / 180) = 2 - Real.sqrt 3 := by
  sorry

end tan_15_degrees_l1106_110615


namespace sqrt_21_position_l1106_110669

theorem sqrt_21_position (n : ℕ) : 
  (∀ k : ℕ, k > 0 → ∃ a : ℝ, a = Real.sqrt (2 * k - 1)) → 
  Real.sqrt 21 = Real.sqrt (2 * 11 - 1) := by
sorry

end sqrt_21_position_l1106_110669


namespace last_colored_cell_position_l1106_110607

/-- Represents a position in the grid --/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the dimensions of the rectangle --/
structure Dimensions :=
  (width : Nat)
  (height : Nat)

/-- Represents the direction of movement in the spiral --/
inductive Direction
  | Right
  | Down
  | Left
  | Up

/-- Function to determine the next position in the spiral --/
def nextPosition (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Right => { row := pos.row,     col := pos.col + 1 }
  | Direction.Down  => { row := pos.row + 1, col := pos.col }
  | Direction.Left  => { row := pos.row,     col := pos.col - 1 }
  | Direction.Up    => { row := pos.row - 1, col := pos.col }

/-- Function to determine if a position is within the rectangle --/
def isWithinBounds (pos : Position) (dim : Dimensions) : Bool :=
  pos.row ≥ 1 && pos.row ≤ dim.height && pos.col ≥ 1 && pos.col ≤ dim.width

/-- Theorem stating that the last colored cell in a 200x100 rectangle,
    colored in a spiral pattern, is at position (51, 50) --/
theorem last_colored_cell_position :
  ∃ (coloringProcess : Nat → Position),
    (coloringProcess 0 = { row := 1, col := 1 }) →
    (∀ n, isWithinBounds (coloringProcess n) { width := 200, height := 100 }) →
    (∀ n, ∃ dir, nextPosition (coloringProcess n) dir = coloringProcess (n + 1)) →
    (∃ lastStep, ∀ m > lastStep, ¬isWithinBounds (coloringProcess m) { width := 200, height := 100 }) →
    (coloringProcess lastStep = { row := 51, col := 50 }) :=
by sorry


end last_colored_cell_position_l1106_110607


namespace calculation_proof_l1106_110689

theorem calculation_proof :
  (1 : ℚ) * (5 / 7 : ℚ) * (-4 - 2/3 : ℚ) / (1 + 2/3 : ℚ) = -2 ∧
  (-2 - 1/7 : ℚ) / (-1.2 : ℚ) * (-1 - 2/5 : ℚ) = -5/2 := by
  sorry

end calculation_proof_l1106_110689


namespace lemonade_amount_l1106_110679

/-- Represents the components of lemonade -/
structure LemonadeComponents where
  water : ℝ
  syrup : ℝ
  lemon_juice : ℝ

/-- Calculates the total amount of lemonade -/
def total_lemonade (c : LemonadeComponents) : ℝ :=
  c.water + c.syrup + c.lemon_juice

/-- Theorem stating the amount of lemonade made given the conditions -/
theorem lemonade_amount (c : LemonadeComponents) 
  (h1 : c.water = 4 * c.syrup) 
  (h2 : c.syrup = 2 * c.lemon_juice)
  (h3 : c.lemon_juice = 3) : 
  total_lemonade c = 24 := by
  sorry

#check lemonade_amount

end lemonade_amount_l1106_110679


namespace complex_equation_solution_l1106_110686

theorem complex_equation_solution (z : ℂ) : z * (2 + I) = 1 + 3 * I → z = 1 + I := by
  sorry

end complex_equation_solution_l1106_110686


namespace arithmetic_and_geometric_sequence_l1106_110616

theorem arithmetic_and_geometric_sequence (a b c : ℝ) : 
  (∃ d : ℝ, b - a = d ∧ c - b = d) →  -- arithmetic sequence condition
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- geometric sequence condition
  (a = b ∧ b = c ∧ a ≠ 0) :=
by sorry

end arithmetic_and_geometric_sequence_l1106_110616


namespace determinant_sum_l1106_110667

theorem determinant_sum (x y : ℝ) (h1 : x ≠ y) 
  (h2 : Matrix.det ![![2, 6, 12], ![4, x, y], ![4, y, x]] = 0) : 
  x + y = 36 := by
  sorry

end determinant_sum_l1106_110667


namespace table_relationship_l1106_110674

def f (x : ℝ) : ℝ := 200 - 3*x - 6*x^2

theorem table_relationship : 
  (f 0 = 200) ∧ 
  (f 2 = 152) ∧ 
  (f 4 = 80) ∧ 
  (f 6 = -16) ∧ 
  (f 8 = -128) := by
  sorry

end table_relationship_l1106_110674


namespace total_oranges_in_boxes_l1106_110602

def box1_capacity : ℕ := 80
def box2_capacity : ℕ := 50
def box1_fill_ratio : ℚ := 3/4
def box2_fill_ratio : ℚ := 3/5

theorem total_oranges_in_boxes :
  (↑box1_capacity * box1_fill_ratio).floor + (↑box2_capacity * box2_fill_ratio).floor = 90 := by
  sorry

end total_oranges_in_boxes_l1106_110602


namespace vloggers_earnings_per_view_l1106_110687

/-- Represents the earnings and viewership of a vlogger -/
structure Vlogger where
  name : String
  daily_viewers : ℕ
  weekly_earnings : ℚ

/-- Calculates the earnings per view for a vlogger -/
def earnings_per_view (v : Vlogger) : ℚ :=
  v.weekly_earnings / (v.daily_viewers * 7)

theorem vloggers_earnings_per_view 
  (voltaire leila : Vlogger)
  (h1 : voltaire.daily_viewers = 50)
  (h2 : leila.daily_viewers = 2 * voltaire.daily_viewers)
  (h3 : leila.weekly_earnings = 350) :
  earnings_per_view voltaire = earnings_per_view leila ∧ 
  earnings_per_view voltaire = 1/2 := by
  sorry

#check vloggers_earnings_per_view

end vloggers_earnings_per_view_l1106_110687


namespace expression_evaluation_l1106_110636

theorem expression_evaluation : -1^2008 + (-1)^2009 + 1^2010 - 1^2011 = -2 := by
  sorry

end expression_evaluation_l1106_110636


namespace bucket_weight_l1106_110684

/-- 
Given:
- p: weight when bucket is three-quarters full
- q: weight when bucket is one-third full
- r: weight of empty bucket
Prove: weight of full bucket is (4p - r) / 3
-/
theorem bucket_weight (p q r : ℝ) : ℝ :=
  let three_quarters_full := p
  let one_third_full := q
  let empty_bucket := r
  let full_bucket := (4 * p - r) / 3
  full_bucket

#check bucket_weight

end bucket_weight_l1106_110684


namespace equal_savings_time_l1106_110605

/-- Proves that Jim and Sara will have saved the same amount after 820 weeks -/
theorem equal_savings_time (sara_initial : ℕ) (sara_weekly : ℕ) (jim_initial : ℕ) (jim_weekly : ℕ)
  (h1 : sara_initial = 4100)
  (h2 : sara_weekly = 10)
  (h3 : jim_initial = 0)
  (h4 : jim_weekly = 15) :
  ∃ w : ℕ, w = 820 ∧ sara_initial + w * sara_weekly = jim_initial + w * jim_weekly :=
by
  sorry

end equal_savings_time_l1106_110605


namespace find_k_l1106_110665

theorem find_k : ∃ k : ℚ, (2 * 2 - 3 * k * (-1) = 1) ∧ k = -1 := by
  sorry

end find_k_l1106_110665


namespace prob_nine_successes_possible_l1106_110634

/-- The number of trials -/
def n : ℕ := 10

/-- The success probability -/
def p : ℝ := 0.9

/-- The binomial probability mass function -/
def binomial_pmf (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem stating that the probability of exactly 9 successes is between 0 and 1 -/
theorem prob_nine_successes_possible :
  0 < binomial_pmf 9 ∧ binomial_pmf 9 < 1 := by
  sorry

end prob_nine_successes_possible_l1106_110634


namespace fixed_point_on_line_l1106_110683

theorem fixed_point_on_line (a b : ℝ) (h : a + 2 * b = 1) :
  a * (1/2) + 3 * (-1/6) + b = 0 := by sorry

end fixed_point_on_line_l1106_110683


namespace basketball_team_selection_count_l1106_110638

/-- The number of ways to select a basketball team lineup with specific roles. -/
theorem basketball_team_selection_count :
  let total_members : ℕ := 15
  let leadership_material : ℕ := 6
  let positions_to_fill : ℕ := 5
  
  -- Number of ways to select captain and vice-captain
  let leadership_selection : ℕ := leadership_material * (leadership_material - 1)
  
  -- Number of ways to select 5 position players from remaining members
  let position_selection : ℕ := 
    (total_members - 2) * (total_members - 3) * (total_members - 4) * 
    (total_members - 5) * (total_members - 6)
  
  leadership_selection * position_selection = 3326400 :=
by sorry

end basketball_team_selection_count_l1106_110638


namespace perpendicular_lines_slope_equation_l1106_110601

theorem perpendicular_lines_slope_equation (k₁ k₂ n : ℝ) : 
  (2 * k₁^2 + 8 * k₁ + n = 0) →
  (2 * k₂^2 + 8 * k₂ + n = 0) →
  (k₁ * k₂ = -1) →
  n = -2 :=
by sorry

end perpendicular_lines_slope_equation_l1106_110601


namespace seventh_twenty_ninth_712th_digit_l1106_110653

def decimal_representation (n d : ℕ) : List ℕ :=
  sorry

theorem seventh_twenty_ninth_712th_digit :
  let repr := decimal_representation 7 29
  let cycle_length := 29
  let digit_position := 712 % cycle_length
  List.get! repr digit_position = 1 := by
  sorry

end seventh_twenty_ninth_712th_digit_l1106_110653


namespace divide_42_problem_l1106_110652

theorem divide_42_problem (x : ℚ) (h : 35 / x = 5) : 42 / x = 6 := by
  sorry

end divide_42_problem_l1106_110652


namespace count_divisible_numbers_eq_179_l1106_110618

/-- The count of five-digit numbers exactly divisible by 6, 7, 8, and 9 -/
def count_divisible_numbers : ℕ :=
  let lcm := Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))
  let lower_bound := ((10000 + lcm - 1) / lcm : ℕ)
  let upper_bound := (99999 / lcm : ℕ)
  upper_bound - lower_bound + 1

theorem count_divisible_numbers_eq_179 : count_divisible_numbers = 179 := by
  sorry

end count_divisible_numbers_eq_179_l1106_110618


namespace no_prime_solution_l1106_110675

theorem no_prime_solution (p : ℕ) (hp : Prime p) : ¬(2^p + p ∣ 3^p + p) := by
  sorry

end no_prime_solution_l1106_110675


namespace school_population_after_new_students_l1106_110697

theorem school_population_after_new_students (initial_avg_age initial_num_students new_students new_avg_age avg_decrease : ℝ) :
  initial_avg_age = 48 →
  new_students = 120 →
  new_avg_age = 32 →
  avg_decrease = 4 →
  (initial_avg_age * initial_num_students + new_avg_age * new_students) / (initial_num_students + new_students) = initial_avg_age - avg_decrease →
  initial_num_students + new_students = 480 := by
sorry

end school_population_after_new_students_l1106_110697


namespace only_solution_is_two_l1106_110685

theorem only_solution_is_two : 
  ∀ n : ℕ, n > 0 → ((n + 1) ∣ (2 * n^2 + 5 * n)) ↔ n = 2 := by
  sorry

end only_solution_is_two_l1106_110685


namespace smith_family_buffet_cost_l1106_110609

/-- Calculates the total cost of a family buffet given the pricing structure and family composition. -/
def familyBuffetCost (adultPrice childPrice : ℚ) (seniorDiscount : ℚ) 
  (numAdults numChildren numSeniors : ℕ) : ℚ :=
  (numAdults : ℚ) * adultPrice + 
  (numChildren : ℚ) * childPrice + 
  (numSeniors : ℚ) * (adultPrice * (1 - seniorDiscount))

/-- Theorem stating that Mr. Smith's family buffet cost is $162 -/
theorem smith_family_buffet_cost : 
  familyBuffetCost 30 15 (1/10) 3 3 1 = 162 := by
  sorry

end smith_family_buffet_cost_l1106_110609


namespace chairs_for_play_l1106_110694

theorem chairs_for_play (rows : ℕ) (chairs_per_row : ℕ) 
  (h1 : rows = 27) (h2 : chairs_per_row = 16) : 
  rows * chairs_per_row = 432 := by
  sorry

end chairs_for_play_l1106_110694


namespace lawn_mowing_time_l1106_110608

/-- Calculates the time required to mow a rectangular lawn -/
theorem lawn_mowing_time 
  (length width : ℝ) 
  (effective_swath : ℝ) 
  (mowing_speed : ℝ) : 
  length = 120 → 
  width = 200 → 
  effective_swath = 2 → 
  mowing_speed = 4000 → 
  (width / effective_swath) * length / mowing_speed = 3 := by
  sorry

#check lawn_mowing_time

end lawn_mowing_time_l1106_110608


namespace apex_angle_of_identical_cones_l1106_110688

/-- The apex angle of a cone is the angle between its generatrices in the axial section. -/
def apex_angle (cone : Type) : ℝ := sorry

/-- A cone with apex at point A -/
structure Cone (A : Type) where
  apex : A
  angle : ℝ

/-- Three cones touch each other externally -/
def touch_externally (c1 c2 c3 : Cone A) : Prop := sorry

/-- A cone touches another cone internally -/
def touch_internally (c1 c2 : Cone A) : Prop := sorry

theorem apex_angle_of_identical_cones 
  (A : Type) 
  (c1 c2 c3 c4 : Cone A) 
  (h1 : touch_externally c1 c2 c3)
  (h2 : c1.angle = c2.angle)
  (h3 : c3.angle = π / 3)
  (h4 : touch_internally c1 c4)
  (h5 : touch_internally c2 c4)
  (h6 : touch_internally c3 c4)
  (h7 : c4.angle = 5 * π / 6) :
  c1.angle = 2 * Real.arctan (Real.sqrt 3 - 1) := by sorry

end apex_angle_of_identical_cones_l1106_110688


namespace quadratic_equation_two_distinct_roots_l1106_110654

theorem quadratic_equation_two_distinct_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + 1 = 0 ∧ x₂^2 - 3*x₂ + 1 = 0 := by
  sorry

end quadratic_equation_two_distinct_roots_l1106_110654


namespace triangle_problem_l1106_110632

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (4 * a * Real.cos B = c^2 - 4 * b * Real.cos A) →
  (C = π / 3) →
  (a + b = 4 * Real.sqrt 2) →
  -- Conclusions
  (c = 4) ∧
  (1/2 * a * b * Real.sin C = (4 * Real.sqrt 3) / 3) :=
by sorry

end triangle_problem_l1106_110632


namespace chocolate_cost_450_l1106_110680

/-- The cost of buying a specific number of chocolate candies, given the cost and quantity per box. -/
def chocolate_cost (candies_per_box : ℕ) (cost_per_box : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem: The cost of 450 chocolate candies is $120, given that a box of 30 candies costs $8. -/
theorem chocolate_cost_450 : chocolate_cost 30 8 450 = 120 := by
  sorry

end chocolate_cost_450_l1106_110680


namespace shaded_area_is_18_l1106_110620

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a point that divides a line segment -/
structure DivisionPoint where
  x : ℝ
  y : ℝ

/-- Calculate the area of the shaded region in a rectangle -/
def shadedArea (rect : Rectangle) (numDivisions : ℕ) : ℝ :=
  sorry

/-- Theorem stating that the shaded area in the given rectangle is 18 -/
theorem shaded_area_is_18 :
  let rect : Rectangle := { length := 9, width := 5 }
  let numDivisions : ℕ := 5
  shadedArea rect numDivisions = 18 := by sorry

end shaded_area_is_18_l1106_110620


namespace right_triangles_count_l1106_110661

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a line of points -/
def Line := List Point

/-- Creates a line of points with given y-coordinate -/
def createLine (y : ℕ) : Line :=
  List.map (fun x => ⟨x, y⟩) (List.range 73)

/-- Checks if three points form a right triangle -/
def isRightTriangle (p1 p2 p3 : Point) : Bool :=
  -- Implementation omitted for brevity
  sorry

/-- Counts the number of right triangles formed by points from two lines -/
def countRightTriangles (line1 line2 : Line) : ℕ :=
  -- Implementation omitted for brevity
  sorry

/-- The main theorem to prove -/
theorem right_triangles_count :
  let line1 := createLine 3
  let line2 := createLine 4
  countRightTriangles line1 line2 = 10654 := by
  sorry

end right_triangles_count_l1106_110661


namespace exponential_regression_model_l1106_110635

/-- Given a model y = ce^(kx) and a linear equation z = 0.3x + 4 where z = ln y,
    prove that c = e^4 and k = 0.3 -/
theorem exponential_regression_model (c k : ℝ) :
  (∀ x y : ℝ, y = c * Real.exp (k * x)) →
  (∀ x z : ℝ, z = 0.3 * x + 4) →
  (∀ y : ℝ, z = Real.log y) →
  c = Real.exp 4 ∧ k = 0.3 := by
sorry

end exponential_regression_model_l1106_110635


namespace intersection_M_N_l1106_110617

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l1106_110617


namespace f_properties_l1106_110691

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (2 * x) + (a - 2) * Real.exp x - x

theorem f_properties (a : ℝ) :
  (∀ x, a ≤ 0 → (deriv (f a)) x < 0) ∧
  (a > 0 → ∀ x, x < -Real.log a → (deriv (f a)) x < 0) ∧
  (a > 0 → ∀ x, x > -Real.log a → (deriv (f a)) x > 0) ∧
  (a ≥ 1 → ∀ x, f a x ≥ 0) :=
by sorry

end f_properties_l1106_110691


namespace B_equals_roster_l1106_110614

def A : Set Int := {-2, 2, 3, 4}

def B : Set Int := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_roster : B = {4, 9, 16} := by sorry

end B_equals_roster_l1106_110614


namespace not_rhombus_from_equal_adjacent_sides_l1106_110641

/-- A quadrilateral is a polygon with four sides -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A rhombus is a quadrilateral with all sides equal -/
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, dist (q.vertices i) (q.vertices ((i + 1) % 4)) = 
                 dist (q.vertices j) (q.vertices ((j + 1) % 4))

/-- Two sides are adjacent if they share a vertex -/
def are_adjacent_sides (q : Quadrilateral) (i j : Fin 4) : Prop :=
  (j = (i + 1) % 4) ∨ (i = (j + 1) % 4)

/-- A pair of adjacent sides are equal -/
def has_equal_adjacent_sides (q : Quadrilateral) : Prop :=
  ∃ i j : Fin 4, are_adjacent_sides q i j ∧ 
    dist (q.vertices i) (q.vertices ((i + 1) % 4)) = 
    dist (q.vertices j) (q.vertices ((j + 1) % 4))

/-- The statement to be proved false -/
theorem not_rhombus_from_equal_adjacent_sides :
  ¬(∀ q : Quadrilateral, has_equal_adjacent_sides q → is_rhombus q) :=
sorry

end not_rhombus_from_equal_adjacent_sides_l1106_110641


namespace tip_percentage_is_ten_percent_l1106_110633

/-- Calculates the tip percentage given the total bill, number of people, and amount paid per person. -/
def calculate_tip_percentage (total_bill : ℚ) (num_people : ℕ) (amount_per_person : ℚ) : ℚ :=
  let total_paid := num_people * amount_per_person
  let tip_amount := total_paid - total_bill
  (tip_amount / total_bill) * 100

/-- Proves that for a bill of $139.00 split among 8 people, if each pays $19.1125, the tip is 10%. -/
theorem tip_percentage_is_ten_percent :
  calculate_tip_percentage 139 8 (19 + 9/80) = 10 := by
  sorry

end tip_percentage_is_ten_percent_l1106_110633


namespace bananas_count_l1106_110682

/-- Represents the contents of a fruit bowl -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Conditions for the fruit bowl problem -/
def fruitBowlConditions (bowl : FruitBowl) : Prop :=
  bowl.pears = bowl.apples + 2 ∧
  bowl.bananas = bowl.pears + 3 ∧
  bowl.apples + bowl.pears + bowl.bananas = 19

/-- Theorem stating that under the given conditions, the number of bananas is 9 -/
theorem bananas_count (bowl : FruitBowl) : 
  fruitBowlConditions bowl → bowl.bananas = 9 := by
  sorry


end bananas_count_l1106_110682


namespace hannah_savings_l1106_110611

theorem hannah_savings (a₁ : ℕ) (r : ℕ) (n : ℕ) (last_term : ℕ) :
  a₁ = 4 → r = 2 → n = 4 → last_term = 20 →
  (a₁ * (r^n - 1) / (r - 1)) + last_term = 80 := by
  sorry

end hannah_savings_l1106_110611


namespace quadratic_roots_product_l1106_110642

theorem quadratic_roots_product (a b : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + b = 0 → (x = a ∨ x = b)) → 
  a + b = 5 → 
  a * b = 6 → 
  a * b = 6 := by
sorry

end quadratic_roots_product_l1106_110642


namespace effective_CAGR_l1106_110699

/-- The effective Compound Annual Growth Rate (CAGR) for an investment with stepped interest rates, inflation, and currency exchange rate changes. -/
theorem effective_CAGR 
  (R1 R2 R3 R4 I C : ℝ) 
  (h_growth : (3/5 : ℝ) = (1 + R1/100)^(5/2) * (1 + R2/100)^(5/2) * (1 + R3/100)^(5/2) * (1 + R4/100)^(5/2)) :
  ∃ CAGR : ℝ, 
    CAGR = ((1 + R1/100)^(5/2) * (1 + R2/100)^(5/2) * (1 + R3/100)^(5/2) * (1 + R4/100)^(5/2) / (1 + I/100)^10 * (1 + C/100)^10)^(1/10) - 1 := by
  sorry

end effective_CAGR_l1106_110699


namespace sphere_radius_in_unit_cube_l1106_110604

/-- The radius of a sphere satisfying specific conditions in a unit cube -/
theorem sphere_radius_in_unit_cube : ∃ r : ℝ,
  (r > 0) ∧ 
  (r^4 - 4*r^3 + 6*r^2 - 8*r + 4 = 0) ∧
  ((0 - r)^2 + (0 - r)^2 + (0 - (1 - r))^2 = r^2) ∧ -- Sphere passes through A(0,0,0)
  ((1 - r)^2 + (1 - r)^2 + (0 - (1 - r))^2 = r^2) ∧ -- Sphere passes through C(1,1,0)
  ((1 - r)^2 + (0 - r)^2 = r^2) ∧                   -- Sphere touches edge through B(1,0,0)
  (1 - (1 - r) = r)                                 -- Sphere touches top face (z=1)
  := by sorry

end sphere_radius_in_unit_cube_l1106_110604


namespace proposition_and_related_l1106_110613

theorem proposition_and_related (a b : ℝ) : 
  (a + b = 1 → a * b ≤ 1/4) ∧ 
  (a * b > 1/4 → a + b ≠ 1) ∧ 
  ¬(a * b ≤ 1/4 → a + b = 1) ∧ 
  ¬(a + b ≠ 1 → a * b > 1/4) := by
sorry

end proposition_and_related_l1106_110613


namespace unique_solution_congruence_system_l1106_110600

theorem unique_solution_congruence_system :
  ∀ x y z : ℤ,
  2 ≤ x ∧ x ≤ y ∧ y ≤ z →
  (x * y) % z = 1 →
  (x * z) % y = 1 →
  (y * z) % x = 1 →
  x = 2 ∧ y = 3 ∧ z = 5 := by
sorry

end unique_solution_congruence_system_l1106_110600


namespace intersection_of_lines_l1106_110673

theorem intersection_of_lines :
  let x : ℚ := 77 / 32
  let y : ℚ := 57 / 20
  (8 * x - 5 * y = 10) ∧ (9 * x + y^2 = 25) := by
  sorry

end intersection_of_lines_l1106_110673


namespace x_equals_zero_l1106_110631

theorem x_equals_zero (a : ℝ) (x : ℝ) 
  (h1 : a > 0) 
  (h2 : (10 : ℝ) ^ x = Real.log (10 * a) + Real.log (a⁻¹)) : 
  x = 0 := by
  sorry

end x_equals_zero_l1106_110631


namespace exists_palindromic_product_l1106_110643

/-- A natural number is palindromic in base 10 if it reads the same forward and backward. -/
def IsPalindromic (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), n = digits.foldl (λ acc d => 10 * acc + d) 0 ∧ digits = digits.reverse

/-- For any natural number not divisible by 10, there exists another natural number
    such that their product is palindromic in base 10. -/
theorem exists_palindromic_product (x : ℕ) (hx : ¬ 10 ∣ x) :
  ∃ y : ℕ, IsPalindromic (x * y) := by
  sorry

end exists_palindromic_product_l1106_110643


namespace geometric_sequence_proof_l1106_110639

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) > a n

theorem geometric_sequence_proof (a : ℕ → ℝ) 
  (h1 : geometric_sequence a)
  (h2 : increasing_sequence a)
  (h3 : a 5 ^ 2 = a 10)
  (h4 : ∀ (n : ℕ), 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  ∀ (n : ℕ), a n = 2^n :=
sorry

end geometric_sequence_proof_l1106_110639


namespace find_original_number_l1106_110659

theorem find_original_number : ∃ x : ℕ, 
  (x : ℚ) / 25 * 85 = x * 67 / 25 + 3390 ∧ x > 0 := by
  sorry

end find_original_number_l1106_110659


namespace area_of_triangle_PAB_l1106_110666

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

-- Define the tangent line
def tangent_line (x y m : ℝ) : Prop := y = Real.sqrt 3 * x + m ∧ m > 0

-- Define the point of tangency P
def point_P (x y : ℝ) : Prop := circle_O x y ∧ ∃ m, tangent_line x y m

-- Define points A and B as intersections of circle O and line y = x
def point_A_B (xa ya xb yb : ℝ) : Prop :=
  circle_O xa ya ∧ line_y_eq_x xa ya ∧
  circle_O xb yb ∧ line_y_eq_x xb yb ∧
  (xa ≠ xb ∨ ya ≠ yb)

-- Theorem statement
theorem area_of_triangle_PAB :
  ∀ (xa ya xb yb xp yp : ℝ),
  point_A_B xa ya xb yb →
  point_P xp yp →
  ∃ (area : ℝ), area = Real.sqrt 6 + Real.sqrt 2 :=
sorry

end area_of_triangle_PAB_l1106_110666


namespace line_MN_tangent_to_circle_l1106_110664

-- Define the necessary types
variable (Point Line Circle : Type)

-- Define the necessary relations and functions
variable (on_line : Point → Line → Prop)
variable (on_circle : Point → Circle → Prop)
variable (center : Circle → Point)
variable (tangent_line : Line → Circle → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Point)
variable (line_through : Point → Point → Line)

-- Define the given points, lines, and circle
variable (A B C O E F R P N M : Point)
variable (AB AC PR EF MN : Line)
variable (ω : Circle)

-- State the theorem
theorem line_MN_tangent_to_circle (h1 : ¬ on_line A (line_through B C))
  (h2 : center ω = O)
  (h3 : tangent_line AC ω)
  (h4 : tangent_line AB ω)
  (h5 : on_line E AC)
  (h6 : on_line F AB)
  (h7 : on_line R EF)
  (h8 : parallel (line_through O P) EF)
  (h9 : on_line P AB)
  (h10 : N = intersect PR AC)
  (h11 : M = intersect AB (line_through R C))
  (h12 : parallel (line_through R C) AC) :
  tangent_line MN ω :=
sorry

end line_MN_tangent_to_circle_l1106_110664


namespace family_trip_eggs_l1106_110657

theorem family_trip_eggs (adults girls : ℕ) (total_eggs : ℕ) : 
  adults = 3 →
  girls = 7 →
  total_eggs = 36 →
  ∃ (boys : ℕ), 
    adults * 3 + girls * 1 + boys * 2 = total_eggs ∧
    boys = 10 :=
by sorry

end family_trip_eggs_l1106_110657


namespace cryptarithm_unique_solution_l1106_110603

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Represents the cryptarithm KIC + KCI = ICK -/
def cryptarithm (K I C : Digit) : Prop :=
  100 * K.val + 10 * I.val + C.val +
  100 * K.val + 10 * C.val + I.val =
  100 * I.val + 10 * C.val + K.val

/-- The cryptarithm has a unique solution -/
theorem cryptarithm_unique_solution :
  ∃! (K I C : Digit), cryptarithm K I C ∧ K ≠ I ∧ K ≠ C ∧ I ≠ C ∧
  K.val = 4 ∧ I.val = 9 ∧ C.val = 5 := by sorry

end cryptarithm_unique_solution_l1106_110603


namespace parking_arrangement_equality_parking_spaces_count_l1106_110623

/-- Number of arrangements of k elements from n elements -/
def A (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of parking spaces -/
def n : ℕ := sorry

/-- Theorem stating the equality of probabilities for different parking arrangements -/
theorem parking_arrangement_equality : A (n - 2) 3 = A 3 2 * A (n - 2) 2 := by sorry

/-- Theorem proving that n equals 10 -/
theorem parking_spaces_count : n = 10 := by sorry

end parking_arrangement_equality_parking_spaces_count_l1106_110623


namespace arithmetic_geometric_sequence_a1_l1106_110672

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) ^ 2 = a n * a (n + 2)

theorem arithmetic_geometric_sequence_a1 (a : ℕ → ℚ) 
  (h_seq : ArithmeticGeometricSequence a) 
  (h_sum : a 1 + a 6 = 11) 
  (h_prod : a 3 * a 4 = 32 / 9) : 
  a 1 = 32 / 3 ∨ a 1 = 1 / 3 := by
  sorry

end arithmetic_geometric_sequence_a1_l1106_110672


namespace height_ratio_of_isosceles_triangles_l1106_110690

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base_angle : ℝ
  side_length : ℝ
  base_length : ℝ
  height : ℝ

/-- The problem statement -/
theorem height_ratio_of_isosceles_triangles
  (triangle_A triangle_B : IsoscelesTriangle)
  (h_vertical_angle : 180 - 2 * triangle_A.base_angle = 180 - 2 * triangle_B.base_angle)
  (h_base_angle_A : triangle_A.base_angle = 40)
  (h_base_angle_B : triangle_B.base_angle = 50)
  (h_side_ratio : triangle_B.side_length / triangle_A.side_length = 5 / 3)
  (h_area_ratio : (triangle_B.base_length * triangle_B.height) / (triangle_A.base_length * triangle_A.height) = 25 / 9) :
  triangle_B.height / triangle_A.height = 5 / 3 := by
sorry

end height_ratio_of_isosceles_triangles_l1106_110690


namespace remainder_3_167_mod_11_l1106_110645

theorem remainder_3_167_mod_11 : 3^167 % 11 = 9 := by
  sorry

end remainder_3_167_mod_11_l1106_110645


namespace sara_pumpkins_left_l1106_110671

/-- Given that Sara grew 43 pumpkins and rabbits ate 23 pumpkins, 
    prove that Sara has 20 pumpkins left. -/
theorem sara_pumpkins_left : 
  let total_grown : ℕ := 43
  let eaten_by_rabbits : ℕ := 23
  let pumpkins_left := total_grown - eaten_by_rabbits
  pumpkins_left = 20 := by sorry

end sara_pumpkins_left_l1106_110671


namespace dan_destroyed_balloons_l1106_110646

/-- The number of red balloons destroyed by Dan -/
def balloons_destroyed (fred_balloons sam_balloons remaining_balloons : ℝ) : ℝ :=
  fred_balloons + sam_balloons - remaining_balloons

theorem dan_destroyed_balloons :
  balloons_destroyed 10.0 46.0 40 = 16.0 := by
  sorry

end dan_destroyed_balloons_l1106_110646


namespace perpendicular_line_through_point_l1106_110695

/-- Given a line L1 with equation x - 2y + 3 = 0, prove that the line L2 with equation 2x + y - 3 = 0
    passes through the point (1, 1) and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x - 2*y + 3 = 0) →  -- L1 equation
  (2*1 + 1 - 3 = 0) ∧  -- L2 passes through (1, 1)
  (1 * 2 + (-2) * 1 = 0)  -- L1 and L2 are perpendicular (slope product = -1)
  :=
by sorry

end perpendicular_line_through_point_l1106_110695


namespace problem_statement_l1106_110651

theorem problem_statement (m : ℝ) : 
  (∀ x₁ ∈ Set.Icc 0 2, ∃ x₂ ∈ Set.Icc 1 2, x₁^2 ≥ (1/2)^x₂ - m) → 
  m ≥ 1/4 := by
  sorry

end problem_statement_l1106_110651


namespace delta_flight_price_l1106_110655

theorem delta_flight_price (delta_discount : Real) (united_price : Real) (united_discount : Real) (price_difference : Real) :
  delta_discount = 0.20 →
  united_price = 1100 →
  united_discount = 0.30 →
  price_difference = 90 →
  ∃ (original_delta_price : Real),
    original_delta_price * (1 - delta_discount) = 
    united_price * (1 - united_discount) - price_difference ∧
    original_delta_price = 850 := by
  sorry

end delta_flight_price_l1106_110655


namespace ellipse_hyperbola_foci_coincide_l1106_110640

/-- The squared distance from the center to a focus of a hyperbola -/
def hyperbola_c_squared (a b : ℝ) : ℝ := a^2 + b^2

/-- The squared distance from the center to a focus of an ellipse -/
def ellipse_c_squared (a b : ℝ) : ℝ := a^2 - b^2

theorem ellipse_hyperbola_foci_coincide :
  let ellipse_a_squared : ℝ := 16
  let hyperbola_a_squared : ℝ := 144 / 25
  let hyperbola_b_squared : ℝ := 81 / 25
  ∀ b_squared : ℝ,
    hyperbola_c_squared hyperbola_a_squared hyperbola_b_squared =
    ellipse_c_squared ellipse_a_squared b_squared →
    b_squared = 7 := by
  sorry

end ellipse_hyperbola_foci_coincide_l1106_110640


namespace min_value_xy_l1106_110668

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y + 6 = x*y) : 
  x*y ≥ 18 := by
  sorry

end min_value_xy_l1106_110668


namespace a_5_equals_9_l1106_110644

-- Define the sequence and its sum
def S (n : ℕ) := n^2

-- Define the general term of the sequence
def a (n : ℕ) : ℕ := S n - S (n-1)

-- Theorem statement
theorem a_5_equals_9 : a 5 = 9 := by sorry

end a_5_equals_9_l1106_110644


namespace power_division_l1106_110648

theorem power_division (a : ℝ) : a^7 / a = a^6 := by
  sorry

end power_division_l1106_110648


namespace puppy_weight_l1106_110662

/-- Given the weights of a puppy and two cats satisfying certain conditions,
    prove that the puppy weighs 12 pounds. -/
theorem puppy_weight (a b c : ℝ) 
    (h1 : a + b + c = 36)
    (h2 : a + c = 3 * b)
    (h3 : a + b = c + 6) :
    a = 12 := by
  sorry

end puppy_weight_l1106_110662
