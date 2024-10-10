import Mathlib

namespace reps_before_high_elevation_pushups_l3505_350547

/-- Calculates the number of reps reached before moving to the next push-up type -/
def repsBeforeNextType (totalWeeks : ℕ) (typesOfPushups : ℕ) (daysPerWeek : ℕ) (repsAddedPerDay : ℕ) (initialReps : ℕ) : ℕ :=
  let weeksPerType : ℕ := totalWeeks / typesOfPushups
  let totalDays : ℕ := weeksPerType * daysPerWeek
  initialReps + (totalDays * repsAddedPerDay)

theorem reps_before_high_elevation_pushups :
  repsBeforeNextType 9 4 5 1 1 = 11 := by
  sorry

end reps_before_high_elevation_pushups_l3505_350547


namespace seven_boys_without_calculators_l3505_350523

/-- Represents the number of boys who didn't bring calculators to Mrs. Luna's math class -/
def boys_without_calculators (total_boys : ℕ) (total_with_calculators : ℕ) (girls_with_calculators : ℕ) : ℕ :=
  total_boys - (total_with_calculators - girls_with_calculators)

/-- Theorem stating that 7 boys didn't bring their calculators to Mrs. Luna's math class -/
theorem seven_boys_without_calculators :
  boys_without_calculators 20 28 15 = 7 := by
  sorry

end seven_boys_without_calculators_l3505_350523


namespace parallelogram_distance_l3505_350512

/-- A parallelogram with given dimensions -/
structure Parallelogram where
  side1 : ℝ  -- Length of one pair of parallel sides
  side2 : ℝ  -- Length of the other pair of parallel sides
  height1 : ℝ  -- Height corresponding to side1
  height2 : ℝ  -- Height corresponding to side2 (to be proved)

/-- Theorem stating the relationship between the dimensions of the parallelogram -/
theorem parallelogram_distance (p : Parallelogram) 
  (h1 : p.side1 = 20) 
  (h2 : p.side2 = 75) 
  (h3 : p.height1 = 60) : 
  p.height2 = 16 := by
  sorry

end parallelogram_distance_l3505_350512


namespace michaels_investment_l3505_350532

theorem michaels_investment (total_investment : ℝ) (thrifty_rate : ℝ) (rich_rate : ℝ) 
  (years : ℕ) (final_amount : ℝ) (thrifty_investment : ℝ) :
  total_investment = 1500 →
  thrifty_rate = 0.04 →
  rich_rate = 0.06 →
  years = 3 →
  final_amount = 1738.84 →
  thrifty_investment * (1 + thrifty_rate) ^ years + 
    (total_investment - thrifty_investment) * (1 + rich_rate) ^ years = final_amount →
  thrifty_investment = 720.84 := by
sorry

end michaels_investment_l3505_350532


namespace quadratic_one_root_l3505_350510

theorem quadratic_one_root (a b c d : ℝ) : 
  b = a - d →
  c = a - 3*d →
  a ≥ b →
  b ≥ c →
  c ≥ 0 →
  (∃! x : ℝ, a*x^2 + b*x + c = 0) →
  (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ x = -(1 + 3*Real.sqrt 22) / 6) :=
by sorry

end quadratic_one_root_l3505_350510


namespace system_ratio_value_l3505_350569

/-- Given a system of linear equations with a nontrivial solution,
    prove that the ratio xy/z^2 has a specific value. -/
theorem system_ratio_value (x y z k : ℝ) : 
  x ≠ 0 →
  y ≠ 0 →
  z ≠ 0 →
  x + k*y + 4*z = 0 →
  3*x + k*y - 3*z = 0 →
  2*x + 5*y - 3*z = 0 →
  -- The condition for nontrivial solution is implicitly included in the equations
  ∃ (c : ℝ), x*y / (z^2) = c :=
by
  sorry


end system_ratio_value_l3505_350569


namespace value_set_of_t_l3505_350594

/-- The value set of t given the conditions -/
theorem value_set_of_t (t : ℝ) : 
  (∀ y, y > 2 * 1 - t + 1 → (1, y) = (1, t)) → 
  (∀ x, x^2 + (2*t - 4)*x + 4 > 0) → 
  3 < t ∧ t < 4 := by
  sorry

end value_set_of_t_l3505_350594


namespace wall_height_proof_l3505_350503

/-- The height of a wall built with a specific number of bricks of given dimensions. -/
theorem wall_height_proof (brick_length brick_width brick_height : ℝ)
  (wall_length wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.08 →
  wall_length = 10 →
  wall_width = 24.5 →
  num_bricks = 12250 →
  ∃ (h : ℝ), h = 0.08 ∧ num_bricks * (brick_length * brick_width * brick_height) = wall_length * h * wall_width :=
by sorry

end wall_height_proof_l3505_350503


namespace alice_has_winning_strategy_l3505_350554

/-- Represents the state of the game with three piles of coins. -/
structure GameState :=
  (pile1 : Nat) (pile2 : Nat) (pile3 : Nat)

/-- Represents a player in the game. -/
inductive Player
  | Alice | Bob | Charlie

/-- Represents a move in the game. -/
structure Move :=
  (pile : Fin 3) (coins : Fin 3)

/-- Defines if a game state is terminal (no coins left). -/
def isTerminal (state : GameState) : Prop :=
  state.pile1 = 0 ∧ state.pile2 = 0 ∧ state.pile3 = 0

/-- Defines a valid move in the game. -/
def validMove (state : GameState) (move : Move) : Prop :=
  match move.pile with
  | 0 => state.pile1 ≥ move.coins
  | 1 => state.pile2 ≥ move.coins
  | 2 => state.pile3 ≥ move.coins

/-- Applies a move to a game state. -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move.pile with
  | 0 => { state with pile1 := state.pile1 - move.coins }
  | 1 => { state with pile2 := state.pile2 - move.coins }
  | 2 => { state with pile3 := state.pile3 - move.coins }

/-- Defines the next player in turn. -/
def nextPlayer : Player → Player
  | Player.Alice => Player.Bob
  | Player.Bob => Player.Charlie
  | Player.Charlie => Player.Alice

/-- Theorem: Alice has a winning strategy in the game starting with piles of 5, 7, and 8 coins. -/
theorem alice_has_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (game : GameState → Player → Prop),
      (∀ s p, game s p → ¬isTerminal s → ∃ m, validMove s m ∧ game (applyMove s m) (nextPlayer p)) →
      (∀ s, isTerminal s → game s Player.Charlie) →
      game { pile1 := 5, pile2 := 7, pile3 := 8 } Player.Alice :=
by sorry

end alice_has_winning_strategy_l3505_350554


namespace nested_expression_value_l3505_350516

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end nested_expression_value_l3505_350516


namespace no_order_for_seven_l3505_350567

def f (x : ℕ) : ℕ := x^2 % 13

def iterate_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem no_order_for_seven :
  ¬ ∃ n : ℕ, n > 0 ∧ iterate_f n 7 = 7 :=
sorry

end no_order_for_seven_l3505_350567


namespace scientific_notation_correct_l3505_350534

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Check if a ScientificNotation represents a given real number -/
def represents (sn : ScientificNotation) (x : ℝ) : Prop :=
  x = sn.coefficient * (10 : ℝ) ^ sn.exponent

/-- The number we want to represent in scientific notation -/
def target_number : ℝ := 37000000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 3.7
    exponent := 7
    h_coeff_range := by sorry }

theorem scientific_notation_correct :
  represents proposed_notation target_number := by sorry

end scientific_notation_correct_l3505_350534


namespace parabola_directrix_l3505_350529

/-- A parabola is defined by its equation relating x and y coordinates. -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- The directrix of a parabola is a line perpendicular to its axis of symmetry. -/
structure Directrix where
  equation : ℝ → ℝ → Prop

/-- For a parabola with equation x² = (1/4)y, its directrix has equation y = -1/16. -/
theorem parabola_directrix (p : Parabola) (d : Directrix) :
  (∀ x y, p.equation x y ↔ x^2 = (1/4) * y) →
  (∀ x y, d.equation x y ↔ y = -1/16) :=
sorry

end parabola_directrix_l3505_350529


namespace isosceles_right_triangle_l3505_350583

theorem isosceles_right_triangle
  (a b c : ℝ)
  (h1 : ∀ x, (b + c) * x^2 - 2 * a * x + (c - b) = 0 → (∃! y, x = y))
  (h2 : Real.sin b * Real.cos a - Real.cos b * Real.sin a = 0) :
  a = b ∧ a^2 + b^2 = c^2 := by
sorry

end isosceles_right_triangle_l3505_350583


namespace sequence_sum_l3505_350573

theorem sequence_sum : 
  let a₁ : ℚ := 4/3
  let a₂ : ℚ := 7/5
  let a₃ : ℚ := 11/8
  let a₄ : ℚ := 19/15
  let a₅ : ℚ := 35/27
  let a₆ : ℚ := 67/52
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ - 9 = -17312.5 / 7020 :=
by sorry

end sequence_sum_l3505_350573


namespace remainder_problem_l3505_350530

theorem remainder_problem : (11^7 + 9^8 + 7^9) % 7 = 1 := by
  sorry

end remainder_problem_l3505_350530


namespace problem_solution_l3505_350570

theorem problem_solution (x y : ℝ) (h1 : x^2 + 4 = y - 2) (h2 : x = 6) : y = 42 := by
  sorry

end problem_solution_l3505_350570


namespace rectangle_area_l3505_350577

/-- The area of a rectangle with width w, length 3w, and diagonal d is (3/10)d^2 -/
theorem rectangle_area (w d : ℝ) (h1 : w > 0) (h2 : d > 0) (h3 : w^2 + (3*w)^2 = d^2) :
  w * (3*w) = (3/10) * d^2 := by
  sorry

end rectangle_area_l3505_350577


namespace max_value_of_expression_l3505_350535

theorem max_value_of_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) :
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 4 ∧ (2*a - b)^2 + (2*b - c)^2 + (2*c - a)^2 > (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2) →
  (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2 ≤ 28 :=
by sorry

end max_value_of_expression_l3505_350535


namespace standard_equation_of_M_no_B_on_circle_and_M_l3505_350508

-- Define the ellipse M
def ellipse_M (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the focus F₁ and vertex C
def F₁ : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (-2, 0)

-- Theorem for part I
theorem standard_equation_of_M : 
  ∀ x y : ℝ, ellipse_M x y ↔ x^2 / 4 + y^2 / 3 = 1 :=
sorry

-- Theorem for part II
theorem no_B_on_circle_and_M :
  ¬ ∃ x₀ y₀ : ℝ, 
    ellipse_M x₀ y₀ ∧ 
    -2 < x₀ ∧ x₀ < 2 ∧
    (x₀ + 1)^2 + y₀^2 = (x₀ + 2)^2 + y₀^2 :=
sorry

end standard_equation_of_M_no_B_on_circle_and_M_l3505_350508


namespace participation_schemes_count_l3505_350522

/-- Represents the number of students -/
def num_students : ℕ := 5

/-- Represents the number of competitions -/
def num_competitions : ℕ := 4

/-- Represents the number of competitions student A cannot participate in -/
def restricted_competitions : ℕ := 2

/-- Calculates the number of different competition participation schemes -/
def participation_schemes : ℕ := 
  (num_competitions - restricted_competitions) * 
  (Nat.factorial num_students / Nat.factorial (num_students - (num_competitions - 1)))

/-- Theorem stating the number of different competition participation schemes -/
theorem participation_schemes_count : participation_schemes = 72 := by
  sorry

end participation_schemes_count_l3505_350522


namespace max_triangles_for_three_families_of_ten_l3505_350589

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : Nat)

/-- Represents the configuration of three families of parallel lines -/
structure LineConfiguration :=
  (family1 : LineFamily)
  (family2 : LineFamily)
  (family3 : LineFamily)

/-- Calculates the maximum number of triangles formed by the given line configuration -/
def maxTriangles (config : LineConfiguration) : Nat :=
  sorry

/-- Theorem stating the maximum number of triangles formed by three families of 10 parallel lines each -/
theorem max_triangles_for_three_families_of_ten :
  ∃ (config : LineConfiguration),
    config.family1.count = 10 ∧
    config.family2.count = 10 ∧
    config.family3.count = 10 ∧
    maxTriangles config = 150 :=
  sorry

end max_triangles_for_three_families_of_ten_l3505_350589


namespace ellipse_equation_and_intersection_l3505_350546

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ := (0, 0)
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- Theorem about the ellipse equation and intersection with a line -/
theorem ellipse_equation_and_intersection
  (e : Ellipse)
  (h1 : e.center = (0, 0))
  (h2 : e.foci_on_x_axis = true)
  (h3 : e.eccentricity = Real.sqrt 3 / 2)
  (h4 : e.passes_through = (4, 1)) :
  (∃ (a b : ℝ), ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 20 + y^2 / 5 = 1)) ∧
  (∀ m : ℝ, (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    (x₁^2 / 20 + y₁^2 / 5 = 1) ∧ (y₁ = x₁ + m) ∧
    (x₂^2 / 20 + y₂^2 / 5 = 1) ∧ (y₂ = x₂ + m)) ↔
   (-5 < m ∧ m < 5)) :=
by sorry

end ellipse_equation_and_intersection_l3505_350546


namespace min_blocks_for_cube_l3505_350539

/-- The length of the rectangular block -/
def block_length : ℕ := 5

/-- The width of the rectangular block -/
def block_width : ℕ := 4

/-- The height of the rectangular block -/
def block_height : ℕ := 3

/-- The side length of the cube formed by the blocks -/
def cube_side : ℕ := Nat.lcm (Nat.lcm block_length block_width) block_height

/-- The volume of the cube -/
def cube_volume : ℕ := cube_side ^ 3

/-- The volume of a single block -/
def block_volume : ℕ := block_length * block_width * block_height

/-- The number of blocks needed to form the cube -/
def blocks_needed : ℕ := cube_volume / block_volume

theorem min_blocks_for_cube : blocks_needed = 3600 := by
  sorry

end min_blocks_for_cube_l3505_350539


namespace intersection_complement_theorem_l3505_350580

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 > 4}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem intersection_complement_theorem :
  N ∩ (Set.univ \ M) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end intersection_complement_theorem_l3505_350580


namespace set_intersection_proof_l3505_350557

def A : Set ℝ := {x : ℝ | |2*x - 1| < 6}
def B : Set ℝ := {-3, 0, 1, 2, 3, 4}

theorem set_intersection_proof : A ∩ B = {0, 1, 2, 3} := by
  sorry

end set_intersection_proof_l3505_350557


namespace triangle_angle_b_triangle_sides_l3505_350597

/-- Theorem: In an acute triangle ABC, if b*cos(C) + √3*b*sin(C) = a + c, then B = π/3 -/
theorem triangle_angle_b (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b * Real.cos C + Real.sqrt 3 * b * Real.sin C = a + c →
  B = π/3 := by
  sorry

/-- Corollary: If b = 2 and the area of triangle ABC is √3, then a = 2 and c = 2 -/
theorem triangle_sides (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b = 2 →
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →
  a = 2 ∧ c = 2 := by
  sorry

end triangle_angle_b_triangle_sides_l3505_350597


namespace fixed_point_of_exponential_plus_constant_l3505_350592

/-- For any positive real number a, the function f(x) = a^x + 4 passes through the point (0, 5) -/
theorem fixed_point_of_exponential_plus_constant (a : ℝ) (ha : a > 0) :
  let f := fun (x : ℝ) => a^x + 4
  f 0 = 5 := by
  sorry

end fixed_point_of_exponential_plus_constant_l3505_350592


namespace cricketer_average_score_l3505_350544

/-- 
Given a cricketer whose average score increases by 4 after scoring 95 runs in the 19th inning,
this theorem proves that the cricketer's average score after 19 innings is 23 runs per inning.
-/
theorem cricketer_average_score 
  (initial_average : ℝ) 
  (score_increase : ℝ) 
  (runs_19th_inning : ℕ) :
  score_increase = 4 →
  runs_19th_inning = 95 →
  (18 * initial_average + runs_19th_inning) / 19 = initial_average + score_increase →
  initial_average + score_increase = 23 :=
by
  sorry

#check cricketer_average_score

end cricketer_average_score_l3505_350544


namespace impossible_assembly_l3505_350533

theorem impossible_assembly (p q r : ℕ) : ¬∃ (x y z : ℕ),
  (2 * p + 2 * r + 2 = 2 * x) ∧
  (2 * p + q + 1 = 2 * x + y) ∧
  (q + r = y + z) :=
by sorry

end impossible_assembly_l3505_350533


namespace exactly_two_integers_satisfy_l3505_350524

-- Define the circle
def circle_center : ℝ × ℝ := (3, -3)
def circle_radius : ℝ := 8

-- Define the point (x, x+2)
def point (x : ℤ) : ℝ × ℝ := (x, x + 2)

-- Define the condition for a point to be inside or on the circle
def inside_or_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 ≤ circle_radius^2

-- Theorem statement
theorem exactly_two_integers_satisfy :
  ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x : ℤ, x ∈ s ↔ inside_or_on_circle (point x) :=
sorry

end exactly_two_integers_satisfy_l3505_350524


namespace triangle_properties_l3505_350506

noncomputable section

-- Define the triangle ABC
variable (A B C : Real)
variable (a b c : Real)

-- Define the conditions
axiom triangle_angles : A + B + C = Real.pi
axiom cos_A : Real.cos A = 1/3
axiom side_a : a = Real.sqrt 3

-- Define the theorem
theorem triangle_properties :
  (Real.sin ((B + C) / 2))^2 + Real.cos (2 * A) = -1/9 ∧
  (∀ x y : Real, x * y ≤ 9/4 ∧ (x = b ∧ y = c → x * y = 9/4)) :=
sorry

end triangle_properties_l3505_350506


namespace lucy_father_age_twice_l3505_350528

theorem lucy_father_age_twice (lucy_birth_year father_birth_year : ℕ) 
  (h1 : lucy_birth_year = 2000) 
  (h2 : father_birth_year = 1960) : 
  ∃ (year : ℕ), year = 2040 ∧ 
  (year - father_birth_year = 2 * (year - lucy_birth_year)) :=
sorry

end lucy_father_age_twice_l3505_350528


namespace first_five_pages_drawings_l3505_350581

def drawings_on_page (page : Nat) : Nat :=
  5 * 2^(page - 1)

def total_drawings (n : Nat) : Nat :=
  (List.range n).map drawings_on_page |>.sum

theorem first_five_pages_drawings : total_drawings 5 = 155 := by
  sorry

end first_five_pages_drawings_l3505_350581


namespace least_positive_integer_to_multiple_of_three_l3505_350540

theorem least_positive_integer_to_multiple_of_three : 
  ∃ (n : ℕ), n > 0 ∧ (575 + n) % 3 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (575 + m) % 3 = 0 → n ≤ m :=
by sorry

end least_positive_integer_to_multiple_of_three_l3505_350540


namespace ball_count_equals_hex_sum_ball_count_2010_l3505_350563

/-- Converts a natural number to its hexadecimal representation -/
def toHex (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball-placing process for n steps -/
def ballCount (n : ℕ) : ℕ :=
  sorry

theorem ball_count_equals_hex_sum (n : ℕ) : 
  ballCount n = sumDigits (toHex n) := by
  sorry

theorem ball_count_2010 : 
  ballCount 2010 = 30 := by
  sorry

end ball_count_equals_hex_sum_ball_count_2010_l3505_350563


namespace max_factorable_n_is_largest_l3505_350505

/-- A polynomial of the form 3x^2 + nx + 72 can be factored as (3x + A)(x + B) where A and B are integers -/
def is_factorable (n : ℤ) : Prop :=
  ∃ A B : ℤ, 3 * B + A = n ∧ A * B = 72

/-- The maximum value of n for which 3x^2 + nx + 72 can be factored as the product of two linear factors with integer coefficients -/
def max_factorable_n : ℤ := 217

/-- Theorem stating that max_factorable_n is the largest value of n for which the polynomial is factorable -/
theorem max_factorable_n_is_largest :
  is_factorable max_factorable_n ∧
  ∀ m : ℤ, m > max_factorable_n → ¬is_factorable m :=
by sorry

end max_factorable_n_is_largest_l3505_350505


namespace garden_rectangle_length_l3505_350518

theorem garden_rectangle_length :
  ∀ (perimeter width length base_triangle height_triangle : ℝ),
    perimeter = 480 →
    width = 2 * base_triangle →
    base_triangle = 50 →
    height_triangle = 100 →
    perimeter = 2 * (length + width) →
    length = 140 := by
  sorry

end garden_rectangle_length_l3505_350518


namespace stratified_sampling_is_most_suitable_l3505_350555

structure Population where
  male : ℕ
  female : ℕ

structure Sample where
  male : ℕ
  female : ℕ

def isStratifiedSampling (pop : Population) (samp : Sample) : Prop :=
  (pop.male : ℚ) / (pop.female : ℚ) = (samp.male : ℚ) / (samp.female : ℚ)

def isMostSuitableMethod (method : String) (pop : Population) (samp : Sample) : Prop :=
  method = "Stratified sampling" ∧ isStratifiedSampling pop samp

theorem stratified_sampling_is_most_suitable :
  let pop : Population := { male := 500, female := 400 }
  let samp : Sample := { male := 25, female := 20 }
  isMostSuitableMethod "Stratified sampling" pop samp :=
by
  sorry

#check stratified_sampling_is_most_suitable

end stratified_sampling_is_most_suitable_l3505_350555


namespace no_solution_exists_l3505_350500

theorem no_solution_exists : ¬∃ (s c : ℕ), 
  15 ≤ s ∧ s ≤ 35 ∧ c > 0 ∧ 30 * s + 31 * c = 1200 :=
by sorry

end no_solution_exists_l3505_350500


namespace coin_exchange_problem_l3505_350525

theorem coin_exchange_problem :
  ∃! (one_cent two_cent five_cent ten_cent : ℕ),
    two_cent = (3 * one_cent) / 5 ∧
    five_cent = (3 * two_cent) / 5 ∧
    ten_cent = (3 * five_cent) / 5 - 7 ∧
    50 < (one_cent + 2 * two_cent + 5 * five_cent + 10 * ten_cent) / 100 ∧
    (one_cent + 2 * two_cent + 5 * five_cent + 10 * ten_cent) / 100 < 100 ∧
    one_cent = 1375 ∧
    two_cent = 825 ∧
    five_cent = 495 ∧
    ten_cent = 290 := by
  sorry

end coin_exchange_problem_l3505_350525


namespace interval_eq_set_representation_l3505_350585

-- Define the interval (-3, 2]
def interval : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 2}

-- Define the set representation
def set_representation : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 2}

-- Theorem stating that the interval and set representation are equal
theorem interval_eq_set_representation : interval = set_representation := by
  sorry

end interval_eq_set_representation_l3505_350585


namespace a_minus_c_value_l3505_350509

/-- Given that A = 742, B = A + 397, and B = C + 693, prove that A - C = 296 -/
theorem a_minus_c_value (A B C : ℤ) 
  (h1 : A = 742)
  (h2 : B = A + 397)
  (h3 : B = C + 693) : 
  A - C = 296 := by
sorry

end a_minus_c_value_l3505_350509


namespace ned_remaining_lives_l3505_350582

/-- Given that Ned started with 83 lives and lost 13 lives, prove that he now has 70 lives. -/
theorem ned_remaining_lives (initial_lives : ℕ) (lost_lives : ℕ) (remaining_lives : ℕ) : 
  initial_lives = 83 → lost_lives = 13 → remaining_lives = initial_lives - lost_lives → remaining_lives = 70 := by
  sorry

end ned_remaining_lives_l3505_350582


namespace words_removed_during_editing_l3505_350543

theorem words_removed_during_editing 
  (yvonne_words : ℕ)
  (janna_words : ℕ)
  (words_removed : ℕ)
  (words_added : ℕ)
  (h1 : yvonne_words = 400)
  (h2 : janna_words = yvonne_words + 150)
  (h3 : words_added = 2 * words_removed)
  (h4 : yvonne_words + janna_words - words_removed + words_added + 30 = 1000) :
  words_removed = 20 := by
  sorry

end words_removed_during_editing_l3505_350543


namespace money_difference_l3505_350561

/-- Given Eliza has 7q + 3 quarters and Tom has 2q + 8 quarters, where every 5 quarters
    over the count of the other person are converted into nickels, the difference in
    their money is 5(q - 1) cents. -/
theorem money_difference (q : ℤ) : 
  let eliza_quarters := 7 * q + 3
  let tom_quarters := 2 * q + 8
  let quarter_difference := eliza_quarters - tom_quarters
  let nickel_groups := quarter_difference / 5
  nickel_groups * 5 = 5 * (q - 1) := by sorry

end money_difference_l3505_350561


namespace sufficient_not_necessary_l3505_350565

theorem sufficient_not_necessary (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  (∃ b : ℝ, b ≠ 2 ∧ (b - 1) * (b - 2) = 0) := by
sorry

end sufficient_not_necessary_l3505_350565


namespace fraction_division_problem_solution_l3505_350504

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem problem_solution : (3 : ℚ) / 4 / ((2 : ℚ) / 5) = 15 / 8 := by sorry

end fraction_division_problem_solution_l3505_350504


namespace impossible_coin_probabilities_l3505_350548

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end impossible_coin_probabilities_l3505_350548


namespace chord_length_concentric_circles_l3505_350511

theorem chord_length_concentric_circles 
  (area_ring : ℝ) 
  (radius_small : ℝ) 
  (chord_length : ℝ) :
  area_ring = 50 * Real.pi ∧ 
  radius_small = 5 →
  chord_length = 10 * Real.sqrt 2 :=
by sorry

end chord_length_concentric_circles_l3505_350511


namespace Q_subset_P_l3505_350549

def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {y | y < 1}

theorem Q_subset_P : Q ⊆ P := by sorry

end Q_subset_P_l3505_350549


namespace lcm_gcf_ratio_l3505_350527

theorem lcm_gcf_ratio : 
  (Nat.lcm 180 594) / (Nat.gcd 180 594) = 330 := by sorry

end lcm_gcf_ratio_l3505_350527


namespace sugar_for_muffins_l3505_350574

/-- Given a recipe for muffins, calculate the required sugar for a larger batch -/
theorem sugar_for_muffins (original_muffins original_sugar target_muffins : ℕ) :
  original_muffins > 0 →
  original_sugar > 0 →
  target_muffins > 0 →
  (original_sugar * target_muffins) / original_muffins = 
    (3 * 72) / 24 :=
by
  sorry

#eval (3 * 72) / 24  -- This should output 9

end sugar_for_muffins_l3505_350574


namespace right_triangle_median_on_hypotenuse_l3505_350566

theorem right_triangle_median_on_hypotenuse (a b : ℝ) (h : a = 5 ∧ b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  (c / 2) = 13 / 2 := by
  sorry

end right_triangle_median_on_hypotenuse_l3505_350566


namespace exactly_one_greater_than_one_l3505_350596

theorem exactly_one_greater_than_one (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (prod_one : a * b * c = 1)
  (sum_greater : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end exactly_one_greater_than_one_l3505_350596


namespace total_traffic_tickets_l3505_350514

/-- The total number of traffic tickets Mark and Sarah have -/
def total_tickets (mark_parking : ℕ) (sarah_parking : ℕ) (mark_speeding : ℕ) (sarah_speeding : ℕ) : ℕ :=
  mark_parking + sarah_parking + mark_speeding + sarah_speeding

/-- Theorem stating the total number of traffic tickets Mark and Sarah have -/
theorem total_traffic_tickets :
  ∀ (mark_parking sarah_parking mark_speeding sarah_speeding : ℕ),
  mark_parking = 2 * sarah_parking →
  mark_speeding = sarah_speeding →
  sarah_speeding = 6 →
  mark_parking = 8 →
  total_tickets mark_parking sarah_parking mark_speeding sarah_speeding = 24 := by
  sorry

end total_traffic_tickets_l3505_350514


namespace video_game_discount_savings_l3505_350517

theorem video_game_discount_savings (original_price : ℚ) 
  (flat_discount : ℚ) (percentage_discount : ℚ) : 
  original_price = 60 →
  flat_discount = 10 →
  percentage_discount = 0.25 →
  (original_price - flat_discount) * (1 - percentage_discount) - 
  (original_price * (1 - percentage_discount) - flat_discount) = 
  250 / 100 := by
  sorry

end video_game_discount_savings_l3505_350517


namespace intersection_point_l3505_350550

/-- A parametric curve in 2D space -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given parametric curve -/
def givenCurve : ParametricCurve where
  x := fun t => -2 + 5 * t
  y := fun t => 1 - 2 * t

/-- Theorem: The point (1/2, 0) is the intersection of the given curve with the x-axis -/
theorem intersection_point : 
  ∃ t : ℝ, givenCurve.x t = 1/2 ∧ givenCurve.y t = 0 :=
by sorry

end intersection_point_l3505_350550


namespace parking_lot_problem_l3505_350553

theorem parking_lot_problem (initial cars_left cars_entered final : ℕ) : 
  cars_left = 13 →
  cars_entered = cars_left + 5 →
  final = 85 →
  final = initial - cars_left + cars_entered →
  initial = 80 := by sorry

end parking_lot_problem_l3505_350553


namespace complex_symmetry_quotient_l3505_350576

theorem complex_symmetry_quotient : 
  ∀ (z₁ z₂ : ℂ), 
  (z₁.im = -z₂.im) → 
  (z₁.re = z₂.re) → 
  (z₁ = 2 + I) → 
  (z₁ / z₂ = (3/5 : ℂ) + (4/5 : ℂ) * I) := by
sorry

end complex_symmetry_quotient_l3505_350576


namespace distance_between_points_l3505_350562

/-- The distance between two points on a plane is the square root of the sum of squares of differences in their coordinates. -/
theorem distance_between_points (A B : ℝ × ℝ) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 26 :=
by
  -- Given points A(2,1) and B(-3,2)
  have hA : A = (2, 1) := by sorry
  have hB : B = (-3, 2) := by sorry
  
  -- Proof goes here
  sorry

end distance_between_points_l3505_350562


namespace password_decryption_probability_l3505_350526

theorem password_decryption_probability 
  (p1 : ℝ) (p2 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/4) 
  (h3 : 0 ≤ p1 ∧ p1 ≤ 1) (h4 : 0 ≤ p2 ∧ p2 ≤ 1) : 
  1 - (1 - p1) * (1 - p2) = 0.4 := by
sorry

end password_decryption_probability_l3505_350526


namespace opposite_signs_and_sum_negative_l3505_350575

theorem opposite_signs_and_sum_negative (a b : ℚ) 
  (h1 : a * b < 0) 
  (h2 : a + b < 0) : 
  a > 0 ∧ b < 0 ∧ |b| > a := by
  sorry

end opposite_signs_and_sum_negative_l3505_350575


namespace concentration_reduction_proof_l3505_350552

def initial_concentration : ℝ := 0.9
def target_concentration : ℝ := 0.1
def concentration_reduction_factor : ℝ := 0.9

def minimum_operations : ℕ := 21

theorem concentration_reduction_proof :
  (∀ n : ℕ, n < minimum_operations → initial_concentration * concentration_reduction_factor ^ n ≥ target_concentration) ∧
  initial_concentration * concentration_reduction_factor ^ minimum_operations < target_concentration :=
by sorry

end concentration_reduction_proof_l3505_350552


namespace boat_distance_upstream_l3505_350564

/-- Proves that the distance travelled upstream is 10 km given the conditions of the boat problem -/
theorem boat_distance_upstream 
  (boat_speed : ℝ) 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 25) 
  (h2 : upstream_time = 1) 
  (h3 : downstream_time = 0.25) : 
  (boat_speed - ((boat_speed * upstream_time - boat_speed * downstream_time) / (upstream_time + downstream_time))) * upstream_time = 10 := by
  sorry

end boat_distance_upstream_l3505_350564


namespace quadratic_negative_root_l3505_350541

theorem quadratic_negative_root (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ x^2 + m*x - 4 = 0) ↔ m > 0 := by
  sorry

end quadratic_negative_root_l3505_350541


namespace composite_transformation_matrix_l3505_350578

/-- The dilation matrix with scale factor 2 -/
def dilationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 0],
    ![0, 2]]

/-- The rotation matrix for 90 degrees counterclockwise rotation -/
def rotationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1],
    ![1,  0]]

/-- The expected result matrix -/
def resultMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -2],
    ![2,  0]]

theorem composite_transformation_matrix :
  rotationMatrix * dilationMatrix = resultMatrix := by
  sorry

end composite_transformation_matrix_l3505_350578


namespace cubic_sum_and_product_l3505_350551

theorem cubic_sum_and_product (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (h : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 ∧ a*b + b*c + c*a = -(a^3 + 12) / a := by
  sorry

end cubic_sum_and_product_l3505_350551


namespace giant_slide_rides_count_l3505_350521

/-- Represents the carnival scenario with given ride times and planned rides --/
structure CarnivalScenario where
  total_time : ℕ  -- Total time in minutes
  roller_coaster_time : ℕ
  tilt_a_whirl_time : ℕ
  giant_slide_time : ℕ
  vortex_time : ℕ
  bumper_cars_time : ℕ
  roller_coaster_rides : ℕ
  tilt_a_whirl_rides : ℕ
  vortex_rides : ℕ
  bumper_cars_rides : ℕ

/-- Theorem stating that the number of giant slide rides is equal to tilt-a-whirl rides --/
theorem giant_slide_rides_count (scenario : CarnivalScenario) : 
  scenario.total_time = 240 ∧
  scenario.roller_coaster_time = 30 ∧
  scenario.tilt_a_whirl_time = 60 ∧
  scenario.giant_slide_time = 15 ∧
  scenario.vortex_time = 45 ∧
  scenario.bumper_cars_time = 25 ∧
  scenario.roller_coaster_rides = 4 ∧
  scenario.tilt_a_whirl_rides = 2 ∧
  scenario.vortex_rides = 1 ∧
  scenario.bumper_cars_rides = 3 →
  scenario.tilt_a_whirl_rides = 2 :=
by sorry

end giant_slide_rides_count_l3505_350521


namespace hyperbola_vertices_distance_l3505_350536

/-- The distance between the vertices of the hyperbola x²/121 - y²/36 = 1 is 22 -/
theorem hyperbola_vertices_distance : 
  let a : ℝ := Real.sqrt 121
  let b : ℝ := Real.sqrt 36
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 121 - y^2 / 36 = 1
  2 * a = 22 := by sorry

end hyperbola_vertices_distance_l3505_350536


namespace sample_for_x_24_possible_x_for_87_l3505_350542

/-- Represents a systematic sampling method for a population of 1000 individuals. -/
def systematicSample (x : Nat) : List Nat :=
  List.range 10
    |>.map (fun k => (x + 33 * k) % 1000)

/-- Checks if a number ends with given digits. -/
def endsWithDigits (n : Nat) (digits : Nat) : Bool :=
  n % 100 = digits

/-- Theorem for the first part of the problem. -/
theorem sample_for_x_24 :
    systematicSample 24 = [24, 157, 290, 323, 456, 589, 622, 755, 888, 921] := by
  sorry

/-- Theorem for the second part of the problem. -/
theorem possible_x_for_87 :
    {x : Nat | ∃ n ∈ systematicSample x, endsWithDigits n 87} =
    {87, 54, 21, 88, 55, 22, 89, 56, 23, 90} := by
  sorry

end sample_for_x_24_possible_x_for_87_l3505_350542


namespace jellybean_problem_l3505_350513

/-- The number of jellybeans remaining after eating 25% --/
def eat_jellybeans (n : ℝ) : ℝ := 0.75 * n

/-- The number of jellybeans Jenny has initially --/
def initial_jellybeans : ℝ := 80

/-- The number of jellybeans added after the first day --/
def added_jellybeans : ℝ := 20

/-- The number of jellybeans remaining after three days --/
def remaining_jellybeans : ℝ := 
  eat_jellybeans (eat_jellybeans (eat_jellybeans initial_jellybeans + added_jellybeans))

theorem jellybean_problem : remaining_jellybeans = 45 := by sorry

end jellybean_problem_l3505_350513


namespace min_value_E_l3505_350571

theorem min_value_E (E : ℝ) : 
  (∃ x : ℝ, ∀ y : ℝ, |E| + |y + 7| + |y - 5| ≥ |E| + |x + 7| + |x - 5| ∧ |E| + |x + 7| + |x - 5| = 12) →
  |E| ≥ 0 ∧ ∀ δ > 0, ∃ x : ℝ, |E| + |x + 7| + |x - 5| < 12 + δ :=
by sorry

end min_value_E_l3505_350571


namespace proposition_and_converse_l3505_350502

theorem proposition_and_converse (a b : ℝ) :
  (a + b ≥ 2 → max a b ≥ 1) ∧
  ¬(max a b ≥ 1 → a + b ≥ 2) := by
sorry

end proposition_and_converse_l3505_350502


namespace square_root_existence_l3505_350537

theorem square_root_existence : 
  (∃ x : ℝ, x^2 = (-3)^2) ∧ 
  (∃ x : ℝ, x^2 = 0) ∧ 
  (∃ x : ℝ, x^2 = 1/8) ∧ 
  (¬∃ x : ℝ, x^2 = -6^3) := by
  sorry

end square_root_existence_l3505_350537


namespace elevator_weight_problem_l3505_350558

/-- Given 6 people with an average weight of 156 lbs, if a 7th person enters and
    the new average weight becomes 151 lbs, then the weight of the 7th person is 121 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (new_avg_weight : ℝ) (seventh_person_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 156 →
  new_avg_weight = 151 →
  (initial_people * initial_avg_weight + seventh_person_weight) / (initial_people + 1) = new_avg_weight →
  seventh_person_weight = 121 := by
  sorry

end elevator_weight_problem_l3505_350558


namespace trailing_zeros_50_factorial_l3505_350501

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem trailing_zeros_50_factorial :
  trailingZeros 50 = 12 := by
  sorry


end trailing_zeros_50_factorial_l3505_350501


namespace product_sum_relation_l3505_350591

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 14 → b = 8 → b - a = 3 := by sorry

end product_sum_relation_l3505_350591


namespace car_journey_distance_l3505_350584

/-- Proves that given a car that travels a certain distance in 9 hours for the forward journey,
    and returns with a speed increased by 20 km/hr in 6 hours, the distance traveled is 360 km. -/
theorem car_journey_distance : ∀ (v : ℝ),
  v * 9 = (v + 20) * 6 →
  v * 9 = 360 :=
by
  sorry

end car_journey_distance_l3505_350584


namespace absolute_value_difference_l3505_350559

theorem absolute_value_difference (m n : ℝ) (hm : m < 0) (hmn : m * n < 0) :
  |n - m + 1| - |m - n - 5| = -4 := by
sorry

end absolute_value_difference_l3505_350559


namespace no_real_arithmetic_progression_l3505_350515

theorem no_real_arithmetic_progression : ¬ ∃ (a b : ℝ), 
  (b - a = a - 12) ∧ (ab - b = b - a) := by
  sorry

end no_real_arithmetic_progression_l3505_350515


namespace grain_transfer_theorem_transfer_valid_l3505_350588

/-- The amount of grain to be transferred from Warehouse B to Warehouse A -/
def transfer : ℕ := 15

/-- The initial amount of grain in Warehouse A -/
def initial_A : ℕ := 540

/-- The initial amount of grain in Warehouse B -/
def initial_B : ℕ := 200

/-- Theorem stating that transferring the specified amount will result in
    Warehouse A having three times the grain of Warehouse B -/
theorem grain_transfer_theorem :
  (initial_A + transfer) = 3 * (initial_B - transfer) := by
  sorry

/-- Proof that the transfer amount is non-negative and not greater than
    the initial amount in Warehouse B -/
theorem transfer_valid :
  0 ≤ transfer ∧ transfer ≤ initial_B := by
  sorry

end grain_transfer_theorem_transfer_valid_l3505_350588


namespace equation_solution_l3505_350568

theorem equation_solution : ∃ x : ℝ, x > 0 ∧ 90 + 5 * 12 / (180 / x) = 91 ∧ x = 3 := by sorry

end equation_solution_l3505_350568


namespace divisibility_properties_l3505_350590

theorem divisibility_properties (n : ℕ) :
  (∃ k : ℤ, 2^n - 1 = 7 * k) ↔ (∃ m : ℕ, n = 3 * m) ∧
  ¬(∃ k : ℤ, 2^n + 1 = 7 * k) :=
by sorry

end divisibility_properties_l3505_350590


namespace rectangle_triangle_equal_area_l3505_350507

theorem rectangle_triangle_equal_area (perimeter : ℝ) (height : ℝ) (x : ℝ) : 
  perimeter = 60 →
  height = 30 →
  ∃ a b : ℝ, 
    a + b = 30 ∧
    a * b = (1/2) * height * x →
  x = 15 :=
by sorry

end rectangle_triangle_equal_area_l3505_350507


namespace mango_rate_calculation_l3505_350572

/-- The rate of mangoes per kg given the purchase details --/
theorem mango_rate_calculation (grape_quantity : ℕ) (grape_rate : ℕ) 
  (mango_quantity : ℕ) (total_paid : ℕ) : 
  grape_quantity = 8 →
  grape_rate = 70 →
  mango_quantity = 9 →
  total_paid = 965 →
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 45 :=
by sorry

end mango_rate_calculation_l3505_350572


namespace conservation_center_turtles_l3505_350598

/-- The number of green turtles -/
def green_turtles : ℕ := 800

/-- The number of hawksbill turtles -/
def hawksbill_turtles : ℕ := 2 * green_turtles + green_turtles

/-- The total number of turtles in the conservation center -/
def total_turtles : ℕ := green_turtles + hawksbill_turtles

theorem conservation_center_turtles : total_turtles = 3200 := by
  sorry

end conservation_center_turtles_l3505_350598


namespace northton_capsule_depth_l3505_350556

/-- The depth of Northton's time capsule given Southton's depth and the relationship between them. -/
theorem northton_capsule_depth (southton_depth : ℕ) (h1 : southton_depth = 15) :
  southton_depth * 4 - 12 = 48 := by
  sorry

end northton_capsule_depth_l3505_350556


namespace point_distance_inequality_l3505_350587

theorem point_distance_inequality (x : ℝ) : 
  (|x - 0| > |x - (-1)|) → x < -1/2 := by
  sorry

end point_distance_inequality_l3505_350587


namespace travel_time_calculation_l3505_350595

/-- Calculates the time required to travel between two cities given a map scale, distance on the map, and car speed. -/
theorem travel_time_calculation (scale : ℚ) (map_distance : ℚ) (car_speed : ℚ) :
  scale = 1 / 3000000 →
  map_distance = 6 →
  car_speed = 30 →
  (map_distance * scale * 100000) / car_speed = 6000 := by
  sorry

#check travel_time_calculation

end travel_time_calculation_l3505_350595


namespace inverse_matrices_sum_l3505_350599

def matrix1 (a b c d e : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![a, 1, b, 2],
    ![2, 3, 4, 3],
    ![c, 5, d, 3],
    ![2, 4, 1, e]]

def matrix2 (f g h i j k : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![-7, f, -13, 3],
    ![g, -15, h, 2],
    ![3, i, 5, 1],
    ![2, j, 4, k]]

theorem inverse_matrices_sum (a b c d e f g h i j k : ℝ) :
  (matrix1 a b c d e) * (matrix2 f g h i j k) = 1 →
  a + b + c + d + e + f + g + h + i + j + k = 22 := by
  sorry

end inverse_matrices_sum_l3505_350599


namespace range_of_a_l3505_350545

theorem range_of_a (A B : Set ℝ) (a : ℝ) :
  A = {x : ℝ | x ≤ 1} →
  B = {x : ℝ | x ≥ a} →
  A ∪ B = Set.univ →
  Set.Iic 1 = {a | ∀ x, (x ∈ A ∪ B ↔ x ∈ Set.univ)} :=
by sorry

end range_of_a_l3505_350545


namespace debby_bottles_left_l3505_350593

/-- Calculates the number of water bottles left after drinking a certain amount per day for a number of days. -/
def bottles_left (total : ℕ) (per_day : ℕ) (days : ℕ) : ℕ :=
  total - (per_day * days)

/-- Theorem stating that given 264 initial bottles, drinking 15 per day for 11 days leaves 99 bottles. -/
theorem debby_bottles_left : bottles_left 264 15 11 = 99 := by
  sorry

end debby_bottles_left_l3505_350593


namespace ellipse_sum_a_k_l3505_350520

-- Define the ellipse
def Ellipse (f₁ f₂ p : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2)
  let d₂ := Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2)
  let c := Real.sqrt ((f₂.1 - f₁.1)^2 + (f₂.2 - f₁.2)^2) / 2
  let a := (d₁ + d₂) / 2
  let b := Real.sqrt (a^2 - c^2)
  let h := (f₁.1 + f₂.1) / 2
  let k := (f₁.2 + f₂.2) / 2
  ∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem ellipse_sum_a_k :
  let f₁ : ℝ × ℝ := (2, 1)
  let f₂ : ℝ × ℝ := (2, 5)
  let p : ℝ × ℝ := (-3, 3)
  Ellipse f₁ f₂ p →
  let a := (Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
            Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2)) / 2
  let k := (f₁.2 + f₂.2) / 2
  a + k = Real.sqrt 29 + 3 := by
  sorry

end ellipse_sum_a_k_l3505_350520


namespace base_equivalence_l3505_350519

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The theorem stating the equivalence of the numbers in different bases --/
theorem base_equivalence (k : Nat) :
  toBase10 [5, 2, 4] 8 = toBase10 [6, 6, 4] k → k = 7 ∧ toBase10 [6, 6, 4] 7 = toBase10 [5, 2, 4] 8 := by
  sorry

end base_equivalence_l3505_350519


namespace linear_function_point_sum_l3505_350586

/-- If the point A(m, n) lies on the line y = -2x + 1, then 4m + 2n + 2022 = 2024 -/
theorem linear_function_point_sum (m n : ℝ) : n = -2 * m + 1 → 4 * m + 2 * n + 2022 = 2024 := by
  sorry

end linear_function_point_sum_l3505_350586


namespace not_increasing_on_interval_l3505_350579

-- Define the function f(x) = -x²
def f (x : ℝ) : ℝ := -x^2

-- Define what it means for a function to be increasing on an interval
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem not_increasing_on_interval : ¬ IsIncreasing f 0 2 := by
  sorry

end not_increasing_on_interval_l3505_350579


namespace polynomial_coefficient_bound_l3505_350560

theorem polynomial_coefficient_bound (a b c d : ℝ) : 
  (∀ x : ℝ, |x| < 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) →
  |a| + |b| + |c| + |d| ≤ 7 := by
  sorry

end polynomial_coefficient_bound_l3505_350560


namespace min_value_of_exponential_sum_l3505_350538

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 2*y = 1) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 2 ∧ ∀ z, z = 2^x + 4^y → z ≥ min :=
sorry

end min_value_of_exponential_sum_l3505_350538


namespace boat_speed_in_still_water_l3505_350531

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 6 →
  downstream_distance = 35.2 →
  downstream_time = 44 / 60 →
  ∃ (boat_speed : ℝ), boat_speed = 42 ∧ 
    downstream_distance = (boat_speed + current_speed) * downstream_time :=
by
  sorry


end boat_speed_in_still_water_l3505_350531
