import Mathlib

namespace characterize_valid_pairs_l524_52469

def is_valid_pair (n p : ℕ+) : Prop :=
  Nat.Prime p.val ∧ n.val ≤ 2 * p.val ∧ (p.val - 1)^n.val + 1 ∣ n.val^2

theorem characterize_valid_pairs :
  ∀ (n p : ℕ+), is_valid_pair n p ↔
    (n = 2 ∧ p = 2) ∨
    (n = 3 ∧ p = 3) ∨
    (n = 1 ∧ Nat.Prime p.val) :=
sorry

end characterize_valid_pairs_l524_52469


namespace solution_approximation_l524_52487

/-- The solution to the equation (0.0077 * 4.5) / (x * 0.1 * 0.007) = 990 is approximately 28571.42 -/
theorem solution_approximation : ∃ x : ℝ, 
  (0.0077 * 4.5) / (x * 0.1 * 0.007) = 990 ∧ 
  abs (x - 28571.42) < 0.01 := by
  sorry

end solution_approximation_l524_52487


namespace journey_speed_l524_52402

/-- Proves that given a journey of 200 km completed in 10 hours with constant speed throughout, the speed of travel is 20 km/hr. -/
theorem journey_speed (total_distance : ℝ) (total_time : ℝ) (speed : ℝ) 
  (h1 : total_distance = 200) 
  (h2 : total_time = 10) 
  (h3 : speed * total_time = total_distance) : 
  speed = 20 := by
  sorry

end journey_speed_l524_52402


namespace binary_93_l524_52422

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Theorem: The binary representation of 93 is [true, false, true, true, true, false, true] -/
theorem binary_93 : toBinary 93 = [true, false, true, true, true, false, true] := by
  sorry

end binary_93_l524_52422


namespace day_365_is_tuesday_l524_52478

/-- Given a year with 365 days, if the 15th day is a Tuesday, then the 365th day is also a Tuesday. -/
theorem day_365_is_tuesday (year : ℕ) (h1 : year % 7 = 2) : (365 % 7 = year % 7) := by
  sorry

#check day_365_is_tuesday

end day_365_is_tuesday_l524_52478


namespace kelly_cheese_days_l524_52495

/-- The number of weeks Kelly needs to cover -/
def weeks : ℕ := 4

/-- The number of packages of string cheese Kelly buys -/
def packages : ℕ := 2

/-- The number of string cheeses in each package -/
def cheeses_per_package : ℕ := 30

/-- The number of string cheeses the oldest child needs per day -/
def oldest_child_cheeses : ℕ := 2

/-- The number of string cheeses the youngest child needs per day -/
def youngest_child_cheeses : ℕ := 1

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: Kelly puts string cheeses in her kids' lunches 5 days per week -/
theorem kelly_cheese_days : 
  (packages * cheeses_per_package) / (oldest_child_cheeses + youngest_child_cheeses) / weeks = 5 := by
  sorry

end kelly_cheese_days_l524_52495


namespace f_inequality_l524_52436

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x + 1 else Real.exp (x * Real.log 2)

theorem f_inequality (x : ℝ) : 
  f x + f (x - 1/2) > 1 ↔ x > -1/4 := by sorry

end f_inequality_l524_52436


namespace product_of_decimals_l524_52444

theorem product_of_decimals : (0.5 : ℝ) * 0.8 = 0.4 := by
  sorry

end product_of_decimals_l524_52444


namespace cos_arcsin_seven_twentyfifths_l524_52448

theorem cos_arcsin_seven_twentyfifths : 
  Real.cos (Real.arcsin (7 / 25)) = 24 / 25 := by
  sorry

end cos_arcsin_seven_twentyfifths_l524_52448


namespace fraction_product_is_one_l524_52409

theorem fraction_product_is_one :
  (4 / 2) * (3 / 6) * (10 / 5) * (15 / 30) * (20 / 10) * (45 / 90) * (50 / 25) * (60 / 120) = 1 := by
  sorry

end fraction_product_is_one_l524_52409


namespace mikes_pumpkins_l524_52460

theorem mikes_pumpkins (sandy_pumpkins : ℕ) (total_pumpkins : ℕ) (mike_pumpkins : ℕ) : 
  sandy_pumpkins = 51 → total_pumpkins = 74 → mike_pumpkins = total_pumpkins - sandy_pumpkins → mike_pumpkins = 23 := by
  sorry

end mikes_pumpkins_l524_52460


namespace geometric_mean_problem_l524_52497

theorem geometric_mean_problem : 
  let a := 7 + 3 * Real.sqrt 5
  let b := 7 - 3 * Real.sqrt 5
  ∃ x : ℝ, x^2 = a * b ∧ (x = 2 ∨ x = -2) :=
by sorry

end geometric_mean_problem_l524_52497


namespace multiple_problem_l524_52407

theorem multiple_problem (m : ℝ) : 38 + m * 43 = 124 ↔ m = 2 := by sorry

end multiple_problem_l524_52407


namespace assignment_methods_count_l524_52400

/-- The number of companies available for internship --/
def num_companies : ℕ := 4

/-- The number of interns to be assigned --/
def num_interns : ℕ := 5

/-- The number of ways to assign interns to companies --/
def assignment_count : ℕ := num_companies ^ num_interns

/-- Theorem stating that the number of assignment methods is 1024 --/
theorem assignment_methods_count : assignment_count = 1024 := by
  sorry

end assignment_methods_count_l524_52400


namespace croissant_fold_time_l524_52446

/-- Represents the time taken for croissant making process -/
structure CroissantTime where
  total_time : ℕ           -- Total time in minutes
  fold_count : ℕ           -- Number of times dough is folded
  rest_time : ℕ            -- Rest time for each fold in minutes
  mix_time : ℕ             -- Time to mix ingredients in minutes
  bake_time : ℕ            -- Time to bake in minutes
  fold_time : ℕ            -- Time to fold dough each time in minutes

/-- Theorem stating the time to fold the dough each time -/
theorem croissant_fold_time (c : CroissantTime) 
  (h1 : c.total_time = 6 * 60)  -- 6 hours in minutes
  (h2 : c.fold_count = 4)
  (h3 : c.rest_time = 75)
  (h4 : c.mix_time = 10)
  (h5 : c.bake_time = 30)
  (h6 : c.total_time = c.mix_time + c.bake_time + c.fold_count * c.rest_time + c.fold_count * c.fold_time) :
  c.fold_time = 5 := by
  sorry


end croissant_fold_time_l524_52446


namespace equation_solution_l524_52494

theorem equation_solution :
  ∃ (x : ℝ), x ≠ 1 ∧ (x^2 - x + 2) / (x - 1) = x + 3 ∧ x = 5/3 :=
by
  sorry

end equation_solution_l524_52494


namespace sector_radius_and_angle_l524_52432

/-- Given a sector with perimeter 4 and area 1, prove its radius is 1 and central angle is 2 -/
theorem sector_radius_and_angle (r θ : ℝ) 
  (h_perimeter : 2 * r + θ * r = 4)
  (h_area : 1/2 * θ * r^2 = 1) : 
  r = 1 ∧ θ = 2 := by sorry

end sector_radius_and_angle_l524_52432


namespace circle_C_properties_l524_52428

-- Define the circle C
def circle_C (x y : ℝ) := (x - 3)^2 + (y - 1)^2 = 1

-- Define the line l
def line_l (x y m : ℝ) := x + 2*y + m = 0

theorem circle_C_properties :
  -- Circle C passes through (2,1) and (3,2)
  circle_C 2 1 ∧ circle_C 3 2 ∧
  -- Circle C is symmetric with respect to x-3y=0
  (∀ x y, circle_C x y → circle_C (3*y) y) →
  -- The standard equation of C is (x-3)^2 + (y-1)^2 = 1
  (∀ x y, circle_C x y ↔ (x - 3)^2 + (y - 1)^2 = 1) ∧
  -- If C intersects line_l at A and B with |AB| = 4√5/5, then m = -4 or m = -6
  (∀ m : ℝ, (∃ A B : ℝ × ℝ, 
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    line_l A.1 A.2 m ∧ line_l B.1 B.2 m ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4*Real.sqrt 5/5)^2) →
    m = -4 ∨ m = -6) :=
sorry

end circle_C_properties_l524_52428


namespace impossible_to_cover_modified_chessboard_l524_52454

/-- Represents a chessboard square --/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Defines the color of a square on the chessboard --/
def squareColor (s : Square) : Bool :=
  (s.row + s.col) % 2 = 0

/-- Represents the modified chessboard after removing two squares --/
def ModifiedChessboard : Set Square :=
  { s : Square | s ≠ ⟨0, 0⟩ ∧ s ≠ ⟨7, 7⟩ }

/-- A domino covers two adjacent squares --/
def validDomino (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.col = s2.col ∧ s1.row.val + 1 = s2.row.val)

/-- A valid domino placement on the modified chessboard --/
def validPlacement (placement : Set (Square × Square)) : Prop :=
  ∀ (s1 s2 : Square), (s1, s2) ∈ placement →
    s1 ∈ ModifiedChessboard ∧ s2 ∈ ModifiedChessboard ∧ validDomino s1 s2

/-- The main theorem stating that it's impossible to cover the modified chessboard with dominos --/
theorem impossible_to_cover_modified_chessboard :
  ¬∃ (placement : Set (Square × Square)),
    validPlacement placement ∧
    (∀ s ∈ ModifiedChessboard, ∃ s1 s2, (s1, s2) ∈ placement ∧ (s = s1 ∨ s = s2)) :=
sorry

end impossible_to_cover_modified_chessboard_l524_52454


namespace quadratic_value_range_l524_52468

-- Define the set of x that satisfies the inequality
def S : Set ℝ := {x : ℝ | x^2 - 7*x + 12 < 0}

-- State the theorem
theorem quadratic_value_range : 
  ∀ x ∈ S, 0 < x^2 - 5*x + 6 ∧ x^2 - 5*x + 6 < 2 :=
by sorry

end quadratic_value_range_l524_52468


namespace triangle_perimeter_proof_l524_52467

theorem triangle_perimeter_proof :
  ∀ a : ℕ,
  a % 2 = 0 →
  2 < a →
  a < 14 →
  6 + 8 + a = 24 :=
by
  sorry

end triangle_perimeter_proof_l524_52467


namespace difference_of_squares_l524_52493

theorem difference_of_squares (m : ℝ) : m^2 - 1 = (m - 1) * (m + 1) := by
  sorry

end difference_of_squares_l524_52493


namespace planar_figures_l524_52415

-- Define the types of figures
inductive Figure
  | TwoSegmentPolyline
  | ThreeSegmentPolyline
  | TriangleClosed
  | QuadrilateralEqualOppositeSides
  | Trapezoid

-- Define what it means for a figure to be planar
def isPlanar (f : Figure) : Prop :=
  match f with
  | Figure.TwoSegmentPolyline => true
  | Figure.ThreeSegmentPolyline => false
  | Figure.TriangleClosed => true
  | Figure.QuadrilateralEqualOppositeSides => false
  | Figure.Trapezoid => true

-- Theorem statement
theorem planar_figures :
  (∀ f : Figure, isPlanar f ↔ (f = Figure.TwoSegmentPolyline ∨ f = Figure.TriangleClosed ∨ f = Figure.Trapezoid)) :=
by sorry

end planar_figures_l524_52415


namespace geometric_sequence_property_l524_52450

-- Define a geometric sequence
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) (h : isGeometric a) :
  a 4 * a 6 * a 8 * a 10 * a 12 = 32 → a 10^2 / a 12 = 2 := by
  sorry

end geometric_sequence_property_l524_52450


namespace selection_theorem_l524_52480

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 5

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 3

/-- Represents the number of course representatives to be selected -/
def num_representatives : ℕ := 5

/-- Calculates the number of ways to select representatives under condition I -/
def selection_ways_I : ℕ := sorry

/-- Calculates the number of ways to select representatives under condition II -/
def selection_ways_II : ℕ := sorry

/-- Calculates the number of ways to select representatives under condition III -/
def selection_ways_III : ℕ := sorry

/-- Calculates the number of ways to select representatives under condition IV -/
def selection_ways_IV : ℕ := sorry

theorem selection_theorem :
  selection_ways_I = 840 ∧
  selection_ways_II = 3360 ∧
  selection_ways_III = 5400 ∧
  selection_ways_IV = 1080 := by sorry

end selection_theorem_l524_52480


namespace arithmetic_sequence_first_term_l524_52473

/-- An arithmetic sequence with common difference 1 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 1

/-- The nth term of the sequence -/
def a (n : ℕ) : ℝ := sorry

theorem arithmetic_sequence_first_term :
  arithmetic_sequence a ∧ a 5 ^ 2 = a 3 * a 11 → a 1 = 1 := by
  sorry

end arithmetic_sequence_first_term_l524_52473


namespace quadratic_function_property_l524_52445

/-- Given a quadratic function f(x) = ax² + bx + 1 with two distinct points
    (m, 2023) and (n, 2023) on its graph, prove that f(m + n) = 1 -/
theorem quadratic_function_property
  (a b m n : ℝ)
  (hm : a * m^2 + b * m + 1 = 2023)
  (hn : a * n^2 + b * n + 1 = 2023)
  (hd : m ≠ n) :
  a * (m + n)^2 + b * (m + n) + 1 = 1 := by
  sorry

end quadratic_function_property_l524_52445


namespace max_intersections_theorem_l524_52482

/-- The number of intersection points for k lines in a plane -/
def num_intersections (k : ℕ) : ℕ := sorry

/-- The maximum number of intersection points after adding one more line to k lines -/
def max_intersections_after_adding_line (k : ℕ) : ℕ := sorry

/-- Theorem: The maximum number of intersection points after adding one more line
    to k lines is equal to the number of intersection points for k lines plus k -/
theorem max_intersections_theorem (k : ℕ) :
  max_intersections_after_adding_line k = num_intersections k + k := by sorry

end max_intersections_theorem_l524_52482


namespace blackboard_numbers_l524_52435

def can_be_written (n : ℕ) : Prop :=
  ∃ x y : ℕ, n + 1 = 2^x * 3^y

theorem blackboard_numbers (n : ℕ) :
  can_be_written n ↔ 
  (n = 13121 ∨ (∃ a b : ℕ, can_be_written a ∧ can_be_written b ∧ n = a * b + a + b)) :=
sorry

end blackboard_numbers_l524_52435


namespace sqrt_inequality_l524_52485

theorem sqrt_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 / y) + Real.sqrt (y^2 / x) ≥ Real.sqrt x + Real.sqrt y := by
  sorry

end sqrt_inequality_l524_52485


namespace sandcastle_ratio_l524_52463

theorem sandcastle_ratio : 
  ∀ (j : ℕ), 
    20 + 200 + j + 5 * j = 580 →
    j / 20 = 3 / 1 := by
  sorry

end sandcastle_ratio_l524_52463


namespace triangle_area_decomposition_l524_52464

/-- Given a triangle with area T and a point inside it, through which lines are drawn parallel to each side,
    dividing the triangle into smaller parallelograms and triangles, with the areas of the resulting
    smaller triangles being T₁, T₂, and T₃, prove that √T₁ + √T₂ + √T₃ = √T. -/
theorem triangle_area_decomposition (T T₁ T₂ T₃ : ℝ) 
  (h₁ : T > 0) (h₂ : T₁ > 0) (h₃ : T₂ > 0) (h₄ : T₃ > 0) :
  Real.sqrt T₁ + Real.sqrt T₂ + Real.sqrt T₃ = Real.sqrt T := by
  sorry

end triangle_area_decomposition_l524_52464


namespace billy_age_l524_52423

theorem billy_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 60) : 
  billy = 45 := by
sorry

end billy_age_l524_52423


namespace complex_pairs_sum_l524_52418

theorem complex_pairs_sum : ∃ (a₁ b₁ a₂ b₂ : ℕ+), 
  a₁ < b₁ ∧ 
  a₂ < b₂ ∧ 
  (a₁ + Complex.I * b₁) * (b₁ - Complex.I * a₁) = 2020 ∧
  (a₂ + Complex.I * b₂) * (b₂ - Complex.I * a₂) = 2020 ∧
  (a₁ : ℕ) + (b₁ : ℕ) + (a₂ : ℕ) + (b₂ : ℕ) = 714 ∧
  (a₁, b₁) ≠ (a₂, b₂) :=
by sorry

end complex_pairs_sum_l524_52418


namespace jessies_cars_l524_52411

theorem jessies_cars (tommy : ℕ) (total : ℕ) (brother_extra : ℕ) :
  tommy = 3 →
  brother_extra = 5 →
  total = 17 →
  ∃ (jessie : ℕ), jessie = 3 ∧ tommy + jessie + (tommy + jessie + brother_extra) = total :=
by sorry

end jessies_cars_l524_52411


namespace arrangements_equal_24_l524_52433

/-- Represents the number of traditional Chinese paintings -/
def traditional_paintings : Nat := 3

/-- Represents the number of oil paintings -/
def oil_paintings : Nat := 2

/-- Represents the number of ink paintings -/
def ink_paintings : Nat := 1

/-- Calculates the number of arrangements for the paintings -/
def calculate_arrangements : Nat :=
  -- The actual calculation is not provided here
  -- It should consider the constraints mentioned in the problem
  sorry

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_equal_24 : calculate_arrangements = 24 := by
  sorry

end arrangements_equal_24_l524_52433


namespace rectangle_square_diagonal_intersection_l524_52439

/-- Given a square and a rectangle with the same perimeter and a common corner,
    prove that the intersection of the rectangle's diagonals lies on the square's diagonal. -/
theorem rectangle_square_diagonal_intersection
  (s a b : ℝ) 
  (h_perimeter : 4 * s = 2 * a + 2 * b) 
  (h_positive : s > 0 ∧ a > 0 ∧ b > 0) :
  a / 2 = b / 2 := by sorry

end rectangle_square_diagonal_intersection_l524_52439


namespace f_properties_l524_52412

noncomputable def f (x : ℝ) : ℝ := Real.log (x * (Real.exp x - Real.exp (-x)) / 2)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end f_properties_l524_52412


namespace mersenne_fermat_prime_composite_l524_52417

theorem mersenne_fermat_prime_composite (n : ℕ) (h : n > 2) :
  (Nat.Prime (2^n - 1) → ¬Nat.Prime (2^n + 1)) ∧
  (Nat.Prime (2^n + 1) → ¬Nat.Prime (2^n - 1)) :=
sorry

end mersenne_fermat_prime_composite_l524_52417


namespace towel_loads_l524_52475

theorem towel_loads (towels_per_load : ℕ) (total_towels : ℕ) (h1 : towels_per_load = 7) (h2 : total_towels = 42) :
  total_towels / towels_per_load = 6 := by
  sorry

end towel_loads_l524_52475


namespace video_game_lives_l524_52458

/-- Given an initial number of players, additional players, and total lives,
    calculate the number of lives per player. -/
def lives_per_player (initial_players : ℕ) (additional_players : ℕ) (total_lives : ℕ) : ℕ :=
  total_lives / (initial_players + additional_players)

/-- Theorem: In the video game scenario, each player has 6 lives. -/
theorem video_game_lives : lives_per_player 2 2 24 = 6 := by
  sorry

#eval lives_per_player 2 2 24

end video_game_lives_l524_52458


namespace ski_prices_solution_l524_52476

theorem ski_prices_solution (x y : ℝ) :
  (2 * x + y = 340) ∧ (3 * x + 2 * y = 570) ↔ x = 110 ∧ y = 120 := by
  sorry

end ski_prices_solution_l524_52476


namespace dragons_volleyball_games_l524_52405

theorem dragons_volleyball_games :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (initial_games * 55 / 100) →
    (initial_wins + 8) = ((initial_games + 12) * 60 / 100) →
    initial_games + 12 = 28 :=
by
  sorry

end dragons_volleyball_games_l524_52405


namespace sequence_general_term_l524_52492

/-- Given a sequence {a_n} where the sum of the first n terms S_n satisfies S_n = (3/2)a_n - 3,
    prove that the general term formula is a_n = 2 * 3^n. -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (3/2) * a n - 3) →
  ∃ C, ∀ n, a n = C * 3^n :=
by sorry

end sequence_general_term_l524_52492


namespace abc_sum_bounds_l524_52472

theorem abc_sum_bounds (a b c d : ℝ) (h : a + b + c = -d) (h_d : d ≠ 0) :
  0 ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c ≤ d^2 / 2 := by
  sorry

end abc_sum_bounds_l524_52472


namespace soccer_tournament_arrangements_l524_52443

/-- The number of teams in the tournament -/
def num_teams : ℕ := 6

/-- The number of matches each team plays -/
def matches_per_team : ℕ := 2

/-- The total number of possible arrangements of matches -/
def total_arrangements : ℕ := 70

/-- Theorem stating the number of possible arrangements for the given conditions -/
theorem soccer_tournament_arrangements :
  ∀ (n : ℕ) (m : ℕ),
    n = num_teams →
    m = matches_per_team →
    (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (k : ℕ), k ≤ 1) →
    (∀ (i : ℕ), i < n → ∃ (s : Finset ℕ), s.card = m ∧ ∀ (j : ℕ), j ∈ s → j < n ∧ j ≠ i) →
    total_arrangements = 70 :=
by sorry

end soccer_tournament_arrangements_l524_52443


namespace inequality_preservation_l524_52483

theorem inequality_preservation (x y : ℝ) (h : x > y) : x - 2 > y - 2 := by
  sorry

end inequality_preservation_l524_52483


namespace total_amount_proof_l524_52498

/-- The total amount shared among p, q, and r -/
def total_amount : ℝ := 5400.000000000001

/-- The amount r has -/
def r_amount : ℝ := 3600.0000000000005

/-- Theorem stating that given r has two-thirds of the total amount and r's amount is 3600.0000000000005,
    the total amount is 5400.000000000001 -/
theorem total_amount_proof :
  (2 / 3 : ℝ) * total_amount = r_amount →
  total_amount = 5400.000000000001 := by
sorry

end total_amount_proof_l524_52498


namespace triangle_tangent_determinant_l524_52437

/-- Given angles A, B, C of a non-right triangle, the determinant of the matrix
    | tan²A  1      1     |
    | 1      tan²B  1     |
    | 1      1      tan²C |
    is equal to 2. -/
theorem triangle_tangent_determinant (A B C : Real) 
  (h : A + B + C = π) 
  (h_non_right : A ≠ π/2 ∧ B ≠ π/2 ∧ C ≠ π/2) : 
  let M : Matrix (Fin 3) (Fin 3) Real := 
    !![Real.tan A ^ 2, 1, 1; 
       1, Real.tan B ^ 2, 1; 
       1, 1, Real.tan C ^ 2]
  Matrix.det M = 2 := by
sorry

end triangle_tangent_determinant_l524_52437


namespace f_properties_l524_52419

open Real

noncomputable def f (x : ℝ) := exp x - (1/2) * x^2

theorem f_properties :
  (∃ (m b : ℝ), m = 1 ∧ b = -1 ∧ ∀ x y, y = f x → m * x + b * y + 1 = 0) ∧
  (3/2 < f (log 2) ∧ f (log 2) < 2) ∧
  (∃! x, f x = 0) := by
  sorry

end f_properties_l524_52419


namespace expression_evaluation_l524_52420

theorem expression_evaluation : 
  11 - 10 / 2 + (8 * 3) - 7 / 1 + 9 - 6 * 2 + 4 - 3 = 21 := by
  sorry

end expression_evaluation_l524_52420


namespace smallest_x_quadratic_l524_52413

theorem smallest_x_quadratic : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + 8 * x + 3
  ∃ x : ℝ, f x = 9 ∧ ∀ y : ℝ, f y = 9 → x ≤ y ∧ x = (-8 - 2 * Real.sqrt 46) / 10 :=
by sorry

end smallest_x_quadratic_l524_52413


namespace geometric_sequence_third_term_squared_l524_52403

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sum of the first five terms of the sequence equals 27. -/
def SumFirstFiveIs27 (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 = 27

/-- The sum of the reciprocals of the first five terms of the sequence equals 3. -/
def SumReciprocalFirstFiveIs3 (a : ℕ → ℝ) : Prop :=
  1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = 3

theorem geometric_sequence_third_term_squared
  (a : ℕ → ℝ)
  (h_geometric : IsGeometricSequence a)
  (h_sum : SumFirstFiveIs27 a)
  (h_sum_reciprocal : SumReciprocalFirstFiveIs3 a) :
  (a 3) ^ 2 = 9 := by
  sorry

end geometric_sequence_third_term_squared_l524_52403


namespace right_triangle_rotation_volume_l524_52434

/-- Given a right-angled triangle with legs b and c, and hypotenuse a, 
    where b + c = 25 and angle α = 61°55'40", 
    the volume of the solid formed by rotating the triangle around its hypotenuse 
    is approximately 887. -/
theorem right_triangle_rotation_volume 
  (b c a : ℝ) (α : Real) 
  (h_right_angle : b^2 + c^2 = a^2)
  (h_sum : b + c = 25)
  (h_angle : α = Real.pi * (61 + 55/60 + 40/3600) / 180) :
  ∃ (V : ℝ), abs (V - 887) < 1 ∧ V = (1/3) * Real.pi * c * (a * b / Real.sqrt (a^2 + b^2))^2 := by
  sorry

end right_triangle_rotation_volume_l524_52434


namespace cone_lateral_surface_area_cone_lateral_surface_area_proof_l524_52474

/-- The lateral surface area of a cone with base radius 3 and lateral surface that unfolds into a semicircle -/
theorem cone_lateral_surface_area : Real :=
  let base_radius : Real := 3
  let lateral_surface_is_semicircle : Prop := True  -- This is a placeholder for the condition
  18 * Real.pi

/-- Proof of the lateral surface area of the cone -/
theorem cone_lateral_surface_area_proof :
  cone_lateral_surface_area = 18 * Real.pi :=
by sorry

end cone_lateral_surface_area_cone_lateral_surface_area_proof_l524_52474


namespace paint_mixture_intensity_l524_52426

/-- Calculates the intensity of a paint mixture -/
def mixturePaintIntensity (originalIntensity : ℚ) (addedIntensity : ℚ) (replacedFraction : ℚ) : ℚ :=
  (1 - replacedFraction) * originalIntensity + replacedFraction * addedIntensity

/-- Theorem stating that mixing 50% intensity paint with 20% intensity paint in a 2:1 ratio results in 40% intensity -/
theorem paint_mixture_intensity :
  mixturePaintIntensity (1/2) (1/5) (1/3) = (2/5) := by
  sorry

#eval mixturePaintIntensity (1/2) (1/5) (1/3)

end paint_mixture_intensity_l524_52426


namespace infinitely_many_satisfying_points_l524_52425

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- A circle with center at the origin and radius 2 -/
def Circle := {p : Point | p.x^2 + p.y^2 ≤ 4}

/-- The endpoints of a diameter of the circle -/
def diameterEndpoints : (Point × Point) :=
  ({x := -2, y := 0}, {x := 2, y := 0})

/-- The condition for a point P to satisfy the sum of squares property -/
def satisfiesSumOfSquares (p : Point) : Prop :=
  let (a, b) := diameterEndpoints
  distanceSquared p a + distanceSquared p b = 8

/-- The set of points satisfying the condition -/
def SatisfyingPoints : Set Point :=
  {p ∈ Circle | satisfiesSumOfSquares p}

theorem infinitely_many_satisfying_points :
  Set.Infinite SatisfyingPoints :=
sorry

end infinitely_many_satisfying_points_l524_52425


namespace triangle_to_pentagon_area_ratio_l524_52453

/-- The ratio of the area of an equilateral triangle to the area of a pentagon formed by
    placing the triangle atop a square (where the triangle's base equals the square's side) -/
theorem triangle_to_pentagon_area_ratio :
  let s : ℝ := 1  -- Assume unit length for simplicity
  let square_area := s^2
  let triangle_area := (s^2 * Real.sqrt 3) / 4
  let pentagon_area := square_area + triangle_area
  triangle_area / pentagon_area = (4 * Real.sqrt 3 - 3) / 13 := by
  sorry

end triangle_to_pentagon_area_ratio_l524_52453


namespace expression_evaluation_l524_52461

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = -1) :
  (x - 2*y)^2 - (x - 3*y)*(x + 3*y) - 4*y^2 = 17 := by
  sorry

end expression_evaluation_l524_52461


namespace root_quadratic_equation_l524_52410

theorem root_quadratic_equation (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → m^2 - 2*m + 2020 = 2023 := by
  sorry

end root_quadratic_equation_l524_52410


namespace quadratic_rotate_translate_l524_52414

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Rotation of a function by 180 degrees around the origin -/
def Rotate180 (f : ℝ → ℝ) : ℝ → ℝ := fun x ↦ -f x

/-- Translation of a function upwards by d units -/
def TranslateUp (f : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := fun x ↦ f x + d

/-- The theorem stating the result of rotating a quadratic function 180 degrees
    around the origin and then translating it upwards -/
theorem quadratic_rotate_translate (a b c d : ℝ) :
  (TranslateUp (Rotate180 (QuadraticFunction a b c)) d) =
  QuadraticFunction (-a) (-b) (-c + d) :=
sorry

end quadratic_rotate_translate_l524_52414


namespace no_solution_exists_l524_52477

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Repeated application of S function n times -/
def repeated_S (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The sum of n and repeated applications of S up to n times -/
def sum_with_repeated_S (n : ℕ) : ℕ := 
  n + (Finset.range n).sum (λ k => repeated_S n k)

/-- Theorem stating that there is no n satisfying the equation -/
theorem no_solution_exists : ¬ ∃ n : ℕ, sum_with_repeated_S n = 2000000 := by
  sorry

end no_solution_exists_l524_52477


namespace square_sum_theorem_l524_52489

theorem square_sum_theorem (x y : ℝ) (h1 : x - y = 5) (h2 : -x*y = 4) : x^2 + y^2 = 17 := by
  sorry

end square_sum_theorem_l524_52489


namespace symmetric_points_l524_52465

/-- Given a point M with coordinates (x, y), this theorem proves the coordinates
    of points symmetric to M with respect to x-axis, y-axis, and origin. -/
theorem symmetric_points (x y : ℝ) :
  let M : ℝ × ℝ := (x, y)
  let M_x_sym : ℝ × ℝ := (x, -y)  -- Symmetric to x-axis
  let M_y_sym : ℝ × ℝ := (-x, y)  -- Symmetric to y-axis
  let M_origin_sym : ℝ × ℝ := (-x, -y)  -- Symmetric to origin
  (M_x_sym = (x, -y)) ∧
  (M_y_sym = (-x, y)) ∧
  (M_origin_sym = (-x, -y)) := by
sorry


end symmetric_points_l524_52465


namespace number_of_children_l524_52449

theorem number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) 
  (h1 : crayons_per_child = 12) 
  (h2 : total_crayons = 216) : 
  total_crayons / crayons_per_child = 18 := by
  sorry

end number_of_children_l524_52449


namespace square_difference_l524_52456

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : 
  (x - y)^2 = 1 := by
  sorry

end square_difference_l524_52456


namespace triangle_longest_side_l524_52488

/-- Given a triangle with sides 8, y+5, and 3y+2, and perimeter 45, the longest side is 24.5 -/
theorem triangle_longest_side (y : ℝ) :
  8 + (y + 5) + (3 * y + 2) = 45 →
  max 8 (max (y + 5) (3 * y + 2)) = 24.5 := by
  sorry

end triangle_longest_side_l524_52488


namespace candy_bar_cost_l524_52496

/-- The cost of a candy bar given initial and remaining amounts -/
theorem candy_bar_cost (initial_amount remaining_amount : ℕ) :
  initial_amount = 5 ∧ remaining_amount = 3 →
  initial_amount - remaining_amount = 2 := by
  sorry

end candy_bar_cost_l524_52496


namespace money_distribution_l524_52466

theorem money_distribution (a b c d : ℤ) : 
  a + b + c + d = 600 →
  a + c = 200 →
  b + c = 350 →
  a + d = 300 →
  a ≥ 2 * b →
  c = 150 :=
by sorry

end money_distribution_l524_52466


namespace sum_reciprocals_eq_2823_div_7_l524_52429

/-- The function f(n) that returns the integer closest to the fourth root of n -/
def f (n : ℕ) : ℕ := sorry

/-- The sum of 1/f(k) for k from 1 to 2018 -/
def sum_reciprocals : ℚ :=
  (Finset.range 2018).sum (fun k => 1 / (f (k + 1) : ℚ))

/-- The theorem stating that the sum of reciprocals equals 2823/7 -/
theorem sum_reciprocals_eq_2823_div_7 : sum_reciprocals = 2823 / 7 := by sorry

end sum_reciprocals_eq_2823_div_7_l524_52429


namespace faculty_reduction_proof_l524_52484

/-- The original number of faculty members before reduction -/
def original_faculty : ℕ := 253

/-- The percentage of faculty remaining after reduction -/
def remaining_percentage : ℚ := 77 / 100

/-- The number of faculty members after reduction -/
def reduced_faculty : ℕ := 195

/-- Theorem stating that the original faculty count, when reduced by 23%, 
    results in approximately 195 professors -/
theorem faculty_reduction_proof : 
  ⌊(original_faculty : ℚ) * remaining_percentage⌋ = reduced_faculty :=
sorry

end faculty_reduction_proof_l524_52484


namespace all_options_satisfy_statement_l524_52471

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem all_options_satisfy_statement : 
  ∀ n ∈ ({54, 81, 99, 108} : Set ℕ), 
    (sum_of_digits n) % 9 = 0 → n % 9 = 0 ∧ n % 3 = 0 := by
  sorry

end all_options_satisfy_statement_l524_52471


namespace x_squared_plus_reciprocal_l524_52486

theorem x_squared_plus_reciprocal (x : ℝ) (h : 35 = x^6 + 1/x^6) :
  x^2 + 1/x^2 = 37 := by
  sorry

end x_squared_plus_reciprocal_l524_52486


namespace smallest_common_multiple_tutors_smallest_group_l524_52438

theorem smallest_common_multiple (n : ℕ) : n > 0 ∧ n % 14 = 0 ∧ n % 10 = 0 ∧ n % 15 = 0 → n ≥ 210 := by
  sorry

theorem tutors_smallest_group : ∃ (n : ℕ), n > 0 ∧ n % 14 = 0 ∧ n % 10 = 0 ∧ n % 15 = 0 ∧ n = 210 := by
  sorry

end smallest_common_multiple_tutors_smallest_group_l524_52438


namespace no_consecutive_ones_eq_fib_l524_52431

/-- The number of binary sequences of length n with no two consecutive 1s -/
def no_consecutive_ones (n : ℕ) : ℕ :=
  sorry

/-- The nth Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: The number of binary sequences of length n with no two consecutive 1s
    is equal to the (n+2)th Fibonacci number -/
theorem no_consecutive_ones_eq_fib (n : ℕ) : no_consecutive_ones n = fib (n + 2) := by
  sorry

end no_consecutive_ones_eq_fib_l524_52431


namespace lime_bottom_implies_magenta_top_l524_52406

-- Define the colors
inductive Color
| Purple
| Cyan
| Magenta
| Lime
| Silver
| Black

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  left : Face
  right : Face

-- Define the property that all faces have different colors
def has_unique_colors (c : Cube) : Prop :=
  c.top.color ≠ c.bottom.color ∧
  c.top.color ≠ c.front.color ∧
  c.top.color ≠ c.back.color ∧
  c.top.color ≠ c.left.color ∧
  c.top.color ≠ c.right.color ∧
  c.bottom.color ≠ c.front.color ∧
  c.bottom.color ≠ c.back.color ∧
  c.bottom.color ≠ c.left.color ∧
  c.bottom.color ≠ c.right.color ∧
  c.front.color ≠ c.back.color ∧
  c.front.color ≠ c.left.color ∧
  c.front.color ≠ c.right.color ∧
  c.back.color ≠ c.left.color ∧
  c.back.color ≠ c.right.color ∧
  c.left.color ≠ c.right.color

-- Theorem statement
theorem lime_bottom_implies_magenta_top (c : Cube) 
  (h1 : has_unique_colors c) 
  (h2 : c.bottom.color = Color.Lime) : 
  c.top.color = Color.Magenta :=
sorry

end lime_bottom_implies_magenta_top_l524_52406


namespace prob_both_3_l524_52408

-- Define the number of sides for each die
def die1_sides : ℕ := 6
def die2_sides : ℕ := 7

-- Define the probability of rolling a 3 on each die
def prob_3_die1 : ℚ := 1 / die1_sides
def prob_3_die2 : ℚ := 1 / die2_sides

-- Theorem: The probability of rolling a 3 on both dice is 1/42
theorem prob_both_3 : prob_3_die1 * prob_3_die2 = 1 / 42 := by
  sorry

end prob_both_3_l524_52408


namespace smallest_prime_factor_of_1729_l524_52462

theorem smallest_prime_factor_of_1729 :
  (Nat.minFac 1729 = 7) := by sorry

end smallest_prime_factor_of_1729_l524_52462


namespace smallest_solution_abs_equation_l524_52441

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 4 ∧
  (∀ (y : ℝ), y * |y| = 3 * y + 4 → x ≤ y) ∧
  x = 4 := by
  sorry

end smallest_solution_abs_equation_l524_52441


namespace students_wearing_other_colors_l524_52404

theorem students_wearing_other_colors 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (h1 : total_students = 900) 
  (h2 : blue_percent = 44 / 100) 
  (h3 : red_percent = 28 / 100) 
  (h4 : green_percent = 10 / 100) : 
  ℕ := by
  
  sorry

#check students_wearing_other_colors

end students_wearing_other_colors_l524_52404


namespace strawberry_weight_theorem_l524_52401

/-- The weight of Marco's strawberries in pounds -/
def marco_weight : ℕ := 19

/-- The difference in weight between Marco's dad's strawberries and Marco's strawberries in pounds -/
def weight_difference : ℕ := 34

/-- The weight of Marco's dad's strawberries in pounds -/
def dad_weight : ℕ := marco_weight + weight_difference

/-- The total weight of Marco's and his dad's strawberries in pounds -/
def total_weight : ℕ := marco_weight + dad_weight

theorem strawberry_weight_theorem :
  total_weight = 72 := by sorry

end strawberry_weight_theorem_l524_52401


namespace counterexample_exists_l524_52447

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end counterexample_exists_l524_52447


namespace binaryOp_solution_l524_52442

/-- A binary operation on positive real numbers -/
def binaryOp : (ℝ → ℝ → ℝ) := sorry

/-- The binary operation is continuous -/
axiom binaryOp_continuous : Continuous (Function.uncurry binaryOp)

/-- The binary operation is commutative -/
axiom binaryOp_comm : ∀ a b : ℝ, a > 0 → b > 0 → binaryOp a b = binaryOp b a

/-- The binary operation is distributive across multiplication -/
axiom binaryOp_distrib : ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
  binaryOp a (b * c) = (binaryOp a b) * (binaryOp a c)

/-- The binary operation satisfies 2 ⊗ 2 = 4 -/
axiom binaryOp_two_two : binaryOp 2 2 = 4

/-- The main theorem: if x ⊗ y = x for x > 1, then y = √2 -/
theorem binaryOp_solution {x y : ℝ} (hx : x > 1) (h : binaryOp x y = x) : 
  y = Real.sqrt 2 := by sorry

end binaryOp_solution_l524_52442


namespace intersection_equals_B_intersection_with_complement_empty_l524_52470

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a - 2 ≤ x ∧ x ≤ 2*a + 3}
def B : Set ℝ := {x : ℝ | x^2 - 6*x + 5 ≤ 0}

-- Theorem for the first question
theorem intersection_equals_B (a : ℝ) :
  (A a) ∩ B = B ↔ a ∈ Set.Icc 1 3 := by sorry

-- Theorem for the second question
theorem intersection_with_complement_empty (a : ℝ) :
  (A a) ∩ (Bᶜ) = ∅ ↔ a < -5 := by sorry

end intersection_equals_B_intersection_with_complement_empty_l524_52470


namespace smallest_a_for_composite_f_l524_52424

/-- A function that represents x^4 + a^2 --/
def f (x a : ℤ) : ℤ := x^4 + a^2

/-- Definition of a composite number --/
def is_composite (n : ℤ) : Prop := ∃ (a b : ℤ), a ≠ 1 ∧ a ≠ -1 ∧ a ≠ n ∧ a ≠ -n ∧ n = a * b

/-- The main theorem --/
theorem smallest_a_for_composite_f :
  ∀ x : ℤ, is_composite (f x 8) ∧
  ∀ a : ℕ, a > 0 ∧ a < 8 → ∃ x : ℤ, ¬is_composite (f x a) :=
sorry

end smallest_a_for_composite_f_l524_52424


namespace no_equal_roots_for_quadratic_l524_52416

/-- The quadratic equation x^2 - (p+1)x + (p-1) = 0 has no real values of p for which its roots are equal. -/
theorem no_equal_roots_for_quadratic :
  ¬ ∃ p : ℝ, ∃ x : ℝ, x^2 - (p + 1) * x + (p - 1) = 0 ∧
    ∀ y : ℝ, y^2 - (p + 1) * y + (p - 1) = 0 → y = x :=
by sorry

end no_equal_roots_for_quadratic_l524_52416


namespace base_sum_problem_l524_52490

theorem base_sum_problem (G₁ G₂ : ℚ) : ∃! (S₁ S₂ : ℕ+),
  (G₁ = (4 * S₁ + 8) / (S₁^2 - 1) ∧ G₁ = (3 * S₂ + 6) / (S₂^2 - 1)) ∧
  (G₂ = (8 * S₁ + 4) / (S₁^2 - 1) ∧ G₂ = (6 * S₂ + 3) / (S₂^2 - 1)) ∧
  S₁ + S₂ = 23 := by
  sorry

end base_sum_problem_l524_52490


namespace product_72516_9999_l524_52481

theorem product_72516_9999 : 72516 * 9999 = 724987484 := by
  sorry

end product_72516_9999_l524_52481


namespace money_sharing_problem_l524_52430

theorem money_sharing_problem (john jose binoy : ℕ) 
  (h1 : john + jose + binoy > 0)  -- Ensure total is positive
  (h2 : jose = 2 * john)          -- Ratio condition for Jose
  (h3 : binoy = 3 * john)         -- Ratio condition for Binoy
  (h4 : john = 2200)              -- John's share
  : john + jose + binoy = 13200 := by
  sorry

end money_sharing_problem_l524_52430


namespace number_multiplication_l524_52421

theorem number_multiplication (x : ℝ) : x - 7 = 9 → 5 * x = 80 := by
  sorry

end number_multiplication_l524_52421


namespace dodecahedron_edge_probability_l524_52499

/-- A regular dodecahedron -/
structure Dodecahedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 20
  edge_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of randomly selecting two vertices that are endpoints of an edge -/
def edge_endpoint_probability (d : Dodecahedron) : ℚ :=
  (d.edges.card : ℚ) / (d.vertices.card.choose 2 : ℚ)

/-- The main theorem -/
theorem dodecahedron_edge_probability (d : Dodecahedron) :
  edge_endpoint_probability d = 3/19 := by
  sorry

end dodecahedron_edge_probability_l524_52499


namespace min_value_expression_l524_52479

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 3) :
  (2 * a^2 + 1) / a + (b^2 - 2) / (b + 2) ≥ 13 / 5 :=
sorry

end min_value_expression_l524_52479


namespace constant_dot_product_l524_52440

/-- The ellipse E -/
def ellipse_E (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- The hyperbola C -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/2 = 1

/-- The foci of ellipse E and hyperbola C coincide -/
axiom foci_coincide : ∀ x y : ℝ, ellipse_E x y → hyperbola_C x y → x^2 - y^2 = 3

/-- The minor axis endpoints and one focus of ellipse E form an equilateral triangle -/
axiom equilateral_triangle : ∀ x y : ℝ, ellipse_E x y → x^2 + y^2 = 1 → x^2 = 3/4

/-- The dot product MP · MQ is constant when m = 17/8 -/
theorem constant_dot_product :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  ellipse_E x₁ y₁ → ellipse_E x₂ y₂ →
  ∃ k : ℝ, y₁ = k * (x₁ - 1) ∧ y₂ = k * (x₂ - 1) →
  (17/8 - x₁) * (17/8 - x₂) + y₁ * y₂ = 33/64 :=
sorry

end constant_dot_product_l524_52440


namespace stocking_stuffers_l524_52491

/-- Calculates the total cost of stocking stuffers for all kids and the number of unique combinations of books and toys for each kid's stocking. -/
theorem stocking_stuffers (num_kids : ℕ) (num_candy_canes : ℕ) (candy_cane_price : ℚ)
  (num_beanie_babies : ℕ) (beanie_baby_price : ℚ) (num_books : ℕ) (book_price : ℚ)
  (num_toys_per_stocking : ℕ) (num_toy_options : ℕ) (toy_price : ℚ) (gift_card_value : ℚ) :
  num_kids = 4 →
  num_candy_canes = 4 →
  candy_cane_price = 1/2 →
  num_beanie_babies = 2 →
  beanie_baby_price = 3 →
  num_books = 5 →
  book_price = 5 →
  num_toys_per_stocking = 3 →
  num_toy_options = 10 →
  toy_price = 1 →
  gift_card_value = 10 →
  (num_kids * (num_candy_canes * candy_cane_price +
               num_beanie_babies * beanie_baby_price +
               book_price +
               num_toys_per_stocking * toy_price +
               gift_card_value) = 104) ∧
  (num_books * (num_toy_options.choose num_toys_per_stocking) = 600) :=
by sorry

end stocking_stuffers_l524_52491


namespace statue_carving_l524_52452

theorem statue_carving (initial_weight : ℝ) (first_week_cut : ℝ) (second_week_cut : ℝ) (final_weight : ℝ) :
  initial_weight = 250 →
  first_week_cut = 0.3 →
  second_week_cut = 0.2 →
  final_weight = 105 →
  let weight_after_first_week := initial_weight * (1 - first_week_cut)
  let weight_after_second_week := weight_after_first_week * (1 - second_week_cut)
  let third_week_cut := (weight_after_second_week - final_weight) / weight_after_second_week
  third_week_cut = 0.25 := by
sorry

end statue_carving_l524_52452


namespace coefficient_x_squared_proof_l524_52457

/-- The coefficient of x^2 in the expansion of (x - 2/x)^4 * (x - 2) -/
def coefficient_x_squared : ℤ := 16

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_x_squared_proof :
  coefficient_x_squared = 
    (-(binomial 4 1 : ℤ) * 2) * (-2 : ℤ) := by sorry

end coefficient_x_squared_proof_l524_52457


namespace sum_abs_zero_implies_a_minus_abs_2a_l524_52451

theorem sum_abs_zero_implies_a_minus_abs_2a (a : ℝ) : a + |a| = 0 → a - |2*a| = 3*a := by
  sorry

end sum_abs_zero_implies_a_minus_abs_2a_l524_52451


namespace triangle_properties_l524_52459

open Real

/-- Given a triangle ABC with angle C = 2π/3 and c² = 5a² + ab, prove the following:
    1. sin B / sin A = 2
    2. The maximum value of sin A * sin B is 1/4 -/
theorem triangle_properties (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angle : angle_C = 2 * π / 3)
  (h_side : c^2 = 5 * a^2 + a * b) :
  (sin angle_B / sin angle_A = 2) ∧
  (∀ x y : ℝ, 0 < x ∧ x < π / 3 → sin x * sin y ≤ 1 / 4) :=
by sorry


end triangle_properties_l524_52459


namespace ratio_to_thirteen_l524_52455

theorem ratio_to_thirteen : ∃ x : ℚ, (5 : ℚ) / 1 = x / 13 ∧ x = 65 := by
  sorry

end ratio_to_thirteen_l524_52455


namespace max_a_for_quadratic_inequality_l524_52427

theorem max_a_for_quadratic_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 3 = x * y) :
  ∃ (a_max : ℝ), ∀ (a : ℝ), 
    (∀ (x y : ℝ), x > 0 → y > 0 → x + y + 3 = x * y → 
      (x + y)^2 - a*(x + y) + 1 ≥ 0) ↔ a ≤ a_max ∧ a_max = 37/6 := by
  sorry

end max_a_for_quadratic_inequality_l524_52427
