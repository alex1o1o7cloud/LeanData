import Mathlib

namespace intersection_shape_circumference_l2909_290931

/-- The circumference of the shape formed by intersecting quarter circles in a square -/
theorem intersection_shape_circumference (π : ℝ) (side_length : ℝ) : 
  π = 3.141 → side_length = 2 → (4 * π) / 3 = 4.188 := by sorry

end intersection_shape_circumference_l2909_290931


namespace unruly_quadratic_max_sum_of_roots_l2909_290963

/-- A quadratic polynomial of the form q(x) = (x-r)^2 - s -/
def QuadraticPolynomial (r s : ℝ) (x : ℝ) : ℝ := (x - r)^2 - s

/-- The composition of a quadratic polynomial with itself -/
def ComposedQuadratic (r s : ℝ) (x : ℝ) : ℝ :=
  QuadraticPolynomial r s (QuadraticPolynomial r s x)

/-- Predicate for an unruly quadratic polynomial -/
def IsUnruly (r s : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), 
    (ComposedQuadratic r s x₁ = 0 ∧
     ComposedQuadratic r s x₂ = 0 ∧
     ComposedQuadratic r s x₃ = 0) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    (∃ (x₄ : ℝ), ComposedQuadratic r s x₄ = 0 ∧
                 (∀ (x : ℝ), ComposedQuadratic r s x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄))

/-- The sum of roots of q(x) = 0 -/
def SumOfRoots (r s : ℝ) : ℝ := 2 * r

theorem unruly_quadratic_max_sum_of_roots :
  ∃ (r s : ℝ), IsUnruly r s ∧
    (∀ (r' s' : ℝ), IsUnruly r' s' → SumOfRoots r s ≥ SumOfRoots r' s') ∧
    QuadraticPolynomial r s 1 = 7/4 :=
sorry

end unruly_quadratic_max_sum_of_roots_l2909_290963


namespace smallest_a_value_l2909_290936

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) :
  17 ≤ a ∧ ∀ a' : ℝ, (0 ≤ a' ∧ (∀ x : ℤ, Real.sin (a' * x + b) = Real.sin (17 * x))) → a' ≥ 17 :=
by sorry

end smallest_a_value_l2909_290936


namespace game_ends_in_one_round_l2909_290908

/-- Represents a player in the game -/
inductive Player : Type
  | A | B | C | D

/-- The state of the game, containing the token count for each player -/
structure GameState :=
  (tokens : Player → Nat)

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := fun p => match p with
    | Player.A => 8
    | Player.B => 9
    | Player.C => 10
    | Player.D => 11 }

/-- Determines if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Prop :=
  ∃ p, state.tokens p = 0

/-- Determines the player with the most tokens -/
def playerWithMostTokens (state : GameState) : Player :=
  sorry

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Theorem: The game ends after 1 round -/
theorem game_ends_in_one_round :
  gameEnded (playRound initialState) :=
sorry

end game_ends_in_one_round_l2909_290908


namespace parabola_translation_l2909_290913

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := 2 * p.a * h + p.b
  , c := p.a * h^2 + p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let p := Parabola.mk 1 4 (-4)
  let p_translated := translate p 2 (-3)
  y = x^2 + 4*x - 4 →
  y = (x + 4)^2 - 11 ↔
  y = p_translated.a * x^2 + p_translated.b * x + p_translated.c :=
by sorry

end parabola_translation_l2909_290913


namespace parking_lot_motorcycles_l2909_290976

theorem parking_lot_motorcycles :
  let total_vehicles : ℕ := 24
  let total_wheels : ℕ := 86
  let car_wheels : ℕ := 4
  let motorcycle_wheels : ℕ := 3
  ∃ (cars motorcycles : ℕ),
    cars + motorcycles = total_vehicles ∧
    car_wheels * cars + motorcycle_wheels * motorcycles = total_wheels ∧
    motorcycles = 10 :=
by sorry

end parking_lot_motorcycles_l2909_290976


namespace triangle_problem_l2909_290971

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
  -- Given condition
  2 * b * Real.cos C = a * Real.cos C + c * Real.cos A →
  -- Part 1: Prove C = π/3
  C = π/3 ∧
  -- Part 2: Given additional conditions
  (b = 2 ∧ c = Real.sqrt 7 →
    -- Prove a = 3
    a = 3 ∧
    -- Prove area = 3√3/2
    1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
by sorry

end triangle_problem_l2909_290971


namespace diophantine_equation_solutions_l2909_290926

def is_solution (x y z : ℕ+) : Prop :=
  (x + y) * (y + z) * (z + x) = x * y * z * (x + y + z) ∧
  Nat.gcd x.val y.val = 1 ∧ Nat.gcd y.val z.val = 1 ∧ Nat.gcd z.val x.val = 1

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+, is_solution x y z ↔ 
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ 
     (x = 1 ∧ y = 1 ∧ z = 2) ∨ 
     (x = 1 ∧ y = 2 ∧ z = 3)) :=
sorry

end diophantine_equation_solutions_l2909_290926


namespace angle_complement_half_supplement_l2909_290930

theorem angle_complement_half_supplement (x : ℝ) : 
  (90 - x) = (1/2) * (180 - x) → x = 0 := by
  sorry

end angle_complement_half_supplement_l2909_290930


namespace smallest_three_digit_congruence_l2909_290990

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    n = 100 ∧ 
    100 ≤ n ∧ n < 1000 ∧
    75 * n % 450 = 300 % 450 ∧
    (∀ m : ℕ, 100 ≤ m ∧ m < n → 75 * m % 450 ≠ 300 % 450) := by
  sorry

end smallest_three_digit_congruence_l2909_290990


namespace student_preferences_l2909_290903

/-- In a class of 30 students, prove that the sum of students who like maths and history is 15,
    given the distribution of student preferences. -/
theorem student_preferences (total : ℕ) (maths_ratio science_ratio history_ratio : ℚ) : 
  total = 30 ∧ 
  maths_ratio = 3/10 ∧ 
  science_ratio = 1/4 ∧ 
  history_ratio = 2/5 → 
  ∃ (maths science history literature : ℕ),
    maths = ⌊maths_ratio * total⌋ ∧
    science = ⌊science_ratio * (total - maths)⌋ ∧
    history = ⌊history_ratio * (total - maths - science)⌋ ∧
    literature = total - maths - science - history ∧
    maths + history = 15 :=
by sorry


end student_preferences_l2909_290903


namespace deposit_withdrawal_amount_l2909_290943

/-- Proves that the total amount withdrawn after 4 years of annual deposits
    with compound interest is equal to (a/p) * ((1+p)^5 - (1+p)),
    where a is the annual deposit amount and p is the interest rate. -/
theorem deposit_withdrawal_amount (a p : ℝ) (h₁ : a > 0) (h₂ : p > 0) :
  a * (1 + p)^4 + a * (1 + p)^3 + a * (1 + p)^2 + a * (1 + p) + a = 
  (a / p) * ((1 + p)^5 - (1 + p)) :=
sorry

end deposit_withdrawal_amount_l2909_290943


namespace fraction_simplification_l2909_290966

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1/2 := by
  sorry

end fraction_simplification_l2909_290966


namespace arithmetic_calculation_l2909_290942

theorem arithmetic_calculation : 2354 + 240 / 60 - 354 * 2 = 1650 := by
  sorry

end arithmetic_calculation_l2909_290942


namespace complex_number_in_fourth_quadrant_l2909_290994

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (3 - 5*I) / (1 - I)
  (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end complex_number_in_fourth_quadrant_l2909_290994


namespace percentage_difference_l2909_290909

theorem percentage_difference : (40 / 100 * 60) - (4 / 5 * 25) = 4 := by sorry

end percentage_difference_l2909_290909


namespace probability_not_losing_l2909_290933

theorem probability_not_losing (p_win p_draw : ℝ) 
  (h_win : p_win = 0.3) 
  (h_draw : p_draw = 0.2) : 
  p_win + p_draw = 0.5 := by
  sorry

end probability_not_losing_l2909_290933


namespace zoe_winter_clothing_l2909_290981

/-- The number of boxes of winter clothing Zoe has. -/
def num_boxes : ℕ := 8

/-- The number of scarves in each box. -/
def scarves_per_box : ℕ := 4

/-- The number of mittens in each box. -/
def mittens_per_box : ℕ := 6

/-- The total number of pieces of winter clothing Zoe has. -/
def total_pieces : ℕ := num_boxes * (scarves_per_box + mittens_per_box)

theorem zoe_winter_clothing :
  total_pieces = 80 :=
by sorry

end zoe_winter_clothing_l2909_290981


namespace green_pill_cost_proof_l2909_290911

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℝ := 15

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℝ := green_pill_cost - 2

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 588

theorem green_pill_cost_proof :
  green_pill_cost = 15 ∧
  pink_pill_cost = green_pill_cost - 2 ∧
  treatment_days * (green_pill_cost + pink_pill_cost) = total_cost :=
by sorry

end green_pill_cost_proof_l2909_290911


namespace block_running_difference_l2909_290973

theorem block_running_difference (inner_side_length outer_side_length : ℝ) 
  (h1 : inner_side_length = 450)
  (h2 : outer_side_length = inner_side_length + 50) : 
  4 * outer_side_length - 4 * inner_side_length = 200 :=
by sorry

end block_running_difference_l2909_290973


namespace rhombus_side_length_l2909_290939

/-- Given a rhombus with area K and diagonals d and 3d, prove its side length. -/
theorem rhombus_side_length (K d : ℝ) (h1 : K > 0) (h2 : d > 0) : ∃ s : ℝ,
  (K = (3 * d^2) / 2) →  -- Area formula for rhombus
  (s^2 = (d^2 / 4) + ((3 * d)^2 / 4)) →  -- Pythagorean theorem for side length
  s = Real.sqrt ((5 * K) / 3) :=
sorry

end rhombus_side_length_l2909_290939


namespace set_equality_implies_subset_l2909_290901

theorem set_equality_implies_subset (A B C : Set α) :
  A ∪ B = B ∩ C → A ⊆ C := by
  sorry

end set_equality_implies_subset_l2909_290901


namespace ceiling_neg_sqrt_64_over_9_l2909_290948

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end ceiling_neg_sqrt_64_over_9_l2909_290948


namespace necessary_not_sufficient_l2909_290975

theorem necessary_not_sufficient (a b c d : ℝ) (h : c > d) :
  (∀ a b, (a - c > b - d) → (a > b)) ∧
  (∃ a b, (a > b) ∧ ¬(a - c > b - d)) :=
by sorry

end necessary_not_sufficient_l2909_290975


namespace sector_central_angle_l2909_290965

/-- Given a circular sector with perimeter 8 and area 4, prove that its central angle is 2 radians -/
theorem sector_central_angle (r : ℝ) (α : ℝ) : 
  r + r + r * α = 8 → -- perimeter condition
  (1/2) * r^2 * α = 4 → -- area condition
  α = 2 := by sorry

end sector_central_angle_l2909_290965


namespace initial_birds_on_fence_l2909_290951

theorem initial_birds_on_fence :
  ∀ (initial_birds additional_birds total_birds : ℕ),
    additional_birds = 4 →
    total_birds = 6 →
    total_birds = initial_birds + additional_birds →
    initial_birds = 2 := by
  sorry

end initial_birds_on_fence_l2909_290951


namespace angus_patrick_diff_l2909_290945

/-- The number of fish caught by Ollie -/
def ollie_catch : ℕ := 5

/-- The number of fish caught by Patrick -/
def patrick_catch : ℕ := 8

/-- The difference between Angus and Ollie's catch -/
def angus_ollie_diff : ℕ := 7

/-- The number of fish caught by Angus -/
def angus_catch : ℕ := ollie_catch + angus_ollie_diff

/-- Theorem: The difference between Angus and Patrick's fish catch is 4 -/
theorem angus_patrick_diff : angus_catch - patrick_catch = 4 := by
  sorry

end angus_patrick_diff_l2909_290945


namespace count_odd_numbers_between_150_and_350_l2909_290941

theorem count_odd_numbers_between_150_and_350 : 
  (Finset.filter (fun n => n % 2 = 1 ∧ 150 < n ∧ n < 350) (Finset.range 350)).card = 100 :=
by sorry

end count_odd_numbers_between_150_and_350_l2909_290941


namespace willow_play_time_l2909_290953

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time Willow played football in minutes -/
def football_time : ℕ := 60

/-- The time Willow played basketball in minutes -/
def basketball_time : ℕ := 60

/-- The total time Willow played in hours -/
def total_time_hours : ℚ := (football_time + basketball_time) / minutes_per_hour

theorem willow_play_time : total_time_hours = 2 := by
  sorry

end willow_play_time_l2909_290953


namespace max_value_constraint_l2909_290920

theorem max_value_constraint (x y : ℝ) (h : 5 * x^2 + 4 * y^2 = 10 * x) : x^2 + y^2 ≤ 4 := by
  sorry

end max_value_constraint_l2909_290920


namespace cube_with_cut_corners_has_36_edges_l2909_290923

/-- A cube with cut corners is a polyhedron resulting from cutting off each corner of a cube
    such that the cutting planes do not intersect within or on the cube. -/
structure CubeWithCutCorners where
  -- We don't need to define the structure explicitly for this problem

/-- The number of edges in a cube with cut corners -/
def num_edges_cube_with_cut_corners : ℕ := 36

/-- Theorem stating that a cube with cut corners has 36 edges -/
theorem cube_with_cut_corners_has_36_edges (c : CubeWithCutCorners) :
  num_edges_cube_with_cut_corners = 36 := by
  sorry

end cube_with_cut_corners_has_36_edges_l2909_290923


namespace certain_number_proof_l2909_290959

theorem certain_number_proof (D S X : ℤ) : 
  D = 20 → S = 55 → X + (D - S) = 3 * D - 90 → X = 5 := by
  sorry

end certain_number_proof_l2909_290959


namespace complement_A_in_U_l2909_290972

def U : Set ℕ := {x | x ≥ 3}
def A : Set ℕ := {x | x^2 ≥ 10}

theorem complement_A_in_U : U \ A = {3} := by sorry

end complement_A_in_U_l2909_290972


namespace some_number_value_l2909_290916

theorem some_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = x * 25 * 45 * 49) : x = 3 := by
  sorry

end some_number_value_l2909_290916


namespace aeroplane_speed_l2909_290987

theorem aeroplane_speed (distance : ℝ) (time1 : ℝ) (time2 : ℝ) (speed2 : ℝ) :
  time1 = 6 →
  time2 = 14 / 3 →
  speed2 = 540 →
  distance = speed2 * time2 →
  distance = (distance / time1) * time1 →
  distance / time1 = 420 := by
sorry

end aeroplane_speed_l2909_290987


namespace regular_pyramid_cross_section_l2909_290985

/-- Regular pyramid with inscribed cross-section --/
structure RegularPyramid where
  -- Base side length
  base_side : ℝ
  -- Ratio of edge division by plane
  edge_ratio : ℝ × ℝ
  -- Ratio of volumes divided by plane
  volume_ratio : ℝ × ℝ
  -- Distance from sphere center to plane
  sphere_center_distance : ℝ
  -- Perimeter of cross-section
  cross_section_perimeter : ℝ

/-- Theorem about regular pyramid with specific cross-section --/
theorem regular_pyramid_cross_section 
  (p : RegularPyramid) 
  (h_base : p.base_side = 2) 
  (h_perimeter : p.cross_section_perimeter = 32/5) :
  p.edge_ratio = (2, 3) ∧ 
  p.volume_ratio = (26, 9) ∧ 
  p.sphere_center_distance = (22 * Real.sqrt 14) / (35 * Real.sqrt 15) := by
  sorry

end regular_pyramid_cross_section_l2909_290985


namespace partnership_profit_l2909_290992

/-- Partnership profit calculation -/
theorem partnership_profit (a b c : ℚ) (b_share : ℚ) : 
  a = 3 * b ∧ b = (2/3) * c ∧ b_share = 600 →
  (11/2) * b_share = 3300 :=
by sorry

end partnership_profit_l2909_290992


namespace arithmetic_sequence_proof_l2909_290935

/-- Proves that 1, 3, and 5 form a monotonically increasing arithmetic sequence with -1 and 7 -/
theorem arithmetic_sequence_proof : 
  let sequence := [-1, 1, 3, 5, 7]
  (∀ i : Fin 4, sequence[i] < sequence[i+1]) ∧ 
  (∃ d : ℤ, ∀ i : Fin 4, sequence[i+1] - sequence[i] = d) := by
  sorry

end arithmetic_sequence_proof_l2909_290935


namespace intersection_complement_equals_set_l2909_290957

def I : Set ℕ := Set.univ

def A : Set ℕ := {x | 2 ≤ x ∧ x ≤ 10}

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def B : Set ℕ := {x | isPrime x}

theorem intersection_complement_equals_set : A ∩ (I \ B) = {4, 6, 8, 9, 10} := by sorry

end intersection_complement_equals_set_l2909_290957


namespace cube_edge_length_l2909_290983

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 16 * Real.pi) :
  ∃ (a : ℝ), a > 0 ∧ a = (4 * Real.sqrt 3) / 3 ∧ 
  surface_area = 4 * Real.pi * ((Real.sqrt 3 * a) / 2)^2 := by
  sorry

end cube_edge_length_l2909_290983


namespace rectangle_perimeter_l2909_290970

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : ℝ × ℝ)

-- Define the folding and crease
structure Folding (rect : Rectangle) :=
  (A' : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)

-- Define the given dimensions
def given_dimensions (rect : Rectangle) (fold : Folding rect) : Prop :=
  let (ax, ay) := rect.A
  let (ex, ey) := fold.E
  let (fx, fy) := fold.F
  let (cx, cy) := rect.C
  Real.sqrt ((ax - ex)^2 + (ay - ey)^2) = 6 ∧
  Real.sqrt ((ex - rect.B.1)^2 + (ey - rect.B.2)^2) = 15 ∧
  Real.sqrt ((cx - fx)^2 + (cy - fy)^2) = 5

-- Define the theorem
theorem rectangle_perimeter (rect : Rectangle) (fold : Folding rect) :
  given_dimensions rect fold →
  (let perimeter := 2 * (Real.sqrt ((rect.A.1 - rect.B.1)^2 + (rect.A.2 - rect.B.2)^2) +
                         Real.sqrt ((rect.B.1 - rect.C.1)^2 + (rect.B.2 - rect.C.2)^2))
   perimeter = 808) := by
  sorry


end rectangle_perimeter_l2909_290970


namespace twentyfour_game_solution_l2909_290932

/-- A type representing the allowed arithmetic operations -/
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

/-- A type representing an arithmetic expression -/
inductive Expr
  | Const (n : Int)
  | BinOp (op : Operation) (e1 e2 : Expr)

/-- Evaluate an expression -/
def eval : Expr → Int
  | Expr.Const n => n
  | Expr.BinOp Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.BinOp Operation.Sub e1 e2 => eval e1 - eval e2
  | Expr.BinOp Operation.Mul e1 e2 => eval e1 * eval e2
  | Expr.BinOp Operation.Div e1 e2 => eval e1 / eval e2

/-- Check if an expression uses all given numbers exactly once -/
def usesAllNumbers (e : Expr) (nums : List Int) : Bool :=
  match e with
  | Expr.Const n => nums == [n]
  | Expr.BinOp _ e1 e2 =>
    let nums1 := nums.filter (λ n => n ∉ collectNumbers e2)
    let nums2 := nums.filter (λ n => n ∉ collectNumbers e1)
    usesAllNumbers e1 nums1 && usesAllNumbers e2 nums2
where
  collectNumbers : Expr → List Int
    | Expr.Const n => [n]
    | Expr.BinOp _ e1 e2 => collectNumbers e1 ++ collectNumbers e2

theorem twentyfour_game_solution :
  ∃ (e : Expr), usesAllNumbers e [3, -5, 6, -8] ∧ eval e = 24 := by
  sorry

end twentyfour_game_solution_l2909_290932


namespace first_investment_rate_l2909_290946

/-- Represents the interest rate problem --/
structure InterestRateProblem where
  firstInvestment : ℝ
  secondInvestment : ℝ
  totalInterest : ℝ
  knownRate : ℝ
  firstRate : ℝ

/-- The interest rate problem satisfies the given conditions --/
def validProblem (p : InterestRateProblem) : Prop :=
  p.secondInvestment = p.firstInvestment - 100 ∧
  p.secondInvestment = 400 ∧
  p.knownRate = 0.07 ∧
  p.totalInterest = 73 ∧
  p.firstInvestment * p.firstRate + p.secondInvestment * p.knownRate = p.totalInterest

/-- The theorem stating that the first investment's interest rate is 0.15 --/
theorem first_investment_rate (p : InterestRateProblem) 
  (h : validProblem p) : p.firstRate = 0.15 := by
  sorry


end first_investment_rate_l2909_290946


namespace intersection_equals_subset_implies_a_values_l2909_290928

def A (a : ℝ) : Set ℝ := {x | x - a = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem intersection_equals_subset_implies_a_values (a : ℝ) 
  (h : A a ∩ B a = B a) : 
  a = 1 ∨ a = -1 ∨ a = 0 := by
  sorry

end intersection_equals_subset_implies_a_values_l2909_290928


namespace gregs_dog_walking_rate_l2909_290967

/-- Greg's dog walking earnings problem -/
theorem gregs_dog_walking_rate :
  ∀ (rate : ℚ),
  (20 + 10 * rate) +   -- One dog for 10 minutes
  2 * (20 + 7 * rate) +  -- Two dogs for 7 minutes each
  3 * (20 + 9 * rate) = 171  -- Three dogs for 9 minutes each
  →
  rate = 1 := by
sorry

end gregs_dog_walking_rate_l2909_290967


namespace geometric_sequence_property_l2909_290902

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : geometric_sequence a) (h4 : a 4 = 5) :
  a 1 * a 7 = 25 := by
  sorry

end geometric_sequence_property_l2909_290902


namespace matrix_N_property_l2909_290997

theorem matrix_N_property :
  ∃ (N : Matrix (Fin 3) (Fin 3) ℝ),
    (∀ (u : Fin 3 → ℝ), N.mulVec u = (3 : ℝ) • u) ∧
    N = !![3, 0, 0; 0, 3, 0; 0, 0, 3] := by
  sorry

end matrix_N_property_l2909_290997


namespace geometric_mean_of_3_and_12_l2909_290986

theorem geometric_mean_of_3_and_12 :
  let b : ℝ := 3
  let c : ℝ := 12
  Real.sqrt (b * c) = 6 := by sorry

end geometric_mean_of_3_and_12_l2909_290986


namespace cos_150_degrees_l2909_290914

theorem cos_150_degrees : Real.cos (150 * π / 180) = -1/2 := by
  sorry

end cos_150_degrees_l2909_290914


namespace absolute_value_not_positive_l2909_290960

theorem absolute_value_not_positive (x : ℚ) : 
  |4 * x - 2| ≤ 0 ↔ x = 1/2 := by sorry

end absolute_value_not_positive_l2909_290960


namespace range_of_m_for_union_equality_l2909_290969

/-- The set A of solutions to x^2 - 3x + 2 = 0 -/
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}

/-- The set B of solutions to x^2 - 2x + m = 0, parameterized by m -/
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + m = 0}

/-- The theorem stating the range of m for which A ∪ B = A -/
theorem range_of_m_for_union_equality :
  {m : ℝ | A ∪ B m = A} = {m : ℝ | m ≥ 1} := by sorry

end range_of_m_for_union_equality_l2909_290969


namespace banana_apple_ratio_l2909_290954

/-- Represents the number of fruits in a basket -/
structure FruitBasket where
  oranges : ℕ
  apples : ℕ
  bananas : ℕ
  peaches : ℕ

/-- Checks if the fruit basket satisfies the given conditions -/
def validBasket (basket : FruitBasket) : Prop :=
  basket.oranges = 6 ∧
  basket.apples = basket.oranges - 2 ∧
  basket.peaches * 2 = basket.bananas ∧
  basket.oranges + basket.apples + basket.bananas + basket.peaches = 28

/-- Theorem stating that in a valid fruit basket, the ratio of bananas to apples is 3:1 -/
theorem banana_apple_ratio (basket : FruitBasket) (h : validBasket basket) :
  basket.bananas = 3 * basket.apples := by
  sorry

end banana_apple_ratio_l2909_290954


namespace lawsuit_probability_comparison_l2909_290927

theorem lawsuit_probability_comparison :
  let p1_win : ℝ := 0.30
  let p2_win : ℝ := 0.50
  let p3_win : ℝ := 0.40
  let p4_win : ℝ := 0.25
  
  let p1_lose : ℝ := 1 - p1_win
  let p2_lose : ℝ := 1 - p2_win
  let p3_lose : ℝ := 1 - p3_win
  let p4_lose : ℝ := 1 - p4_win
  
  let p_win_all : ℝ := p1_win * p2_win * p3_win * p4_win
  let p_lose_all : ℝ := p1_lose * p2_lose * p3_lose * p4_lose
  
  (p_lose_all - p_win_all) / p_win_all = 9.5
:= by sorry

end lawsuit_probability_comparison_l2909_290927


namespace contrapositive_zero_product_l2909_290905

theorem contrapositive_zero_product (a b : ℝ) :
  (¬(a = 0 ∨ b = 0) → ab ≠ 0) ↔ (ab = 0 → a = 0 ∨ b = 0) :=
by sorry

end contrapositive_zero_product_l2909_290905


namespace rectangle_division_exists_l2909_290947

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a division of a rectangle into parts -/
structure RectangleDivision where
  parts : List ℝ

/-- Checks if a division is valid for a given rectangle -/
def isValidDivision (r : Rectangle) (d : RectangleDivision) : Prop :=
  d.parts.length = 4 ∧ d.parts.sum = r.width * r.height

/-- The main theorem to be proved -/
theorem rectangle_division_exists : ∃ (d : RectangleDivision), 
  isValidDivision ⟨6, 10⟩ d ∧ 
  d.parts = [8, 12, 16, 24] := by
  sorry


end rectangle_division_exists_l2909_290947


namespace rectangular_to_polar_conversion_l2909_290956

theorem rectangular_to_polar_conversion :
  let x : ℝ := 2
  let y : ℝ := -2 * Real.sqrt 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 && y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  r = 2 * Real.sqrt 3 ∧ θ = 5 * Real.pi / 4 :=
by sorry

end rectangular_to_polar_conversion_l2909_290956


namespace black_or_white_probability_l2909_290915

/-- The probability of drawing a red ball from the box -/
def prob_red : ℝ := 0.45

/-- The probability of drawing a white ball from the box -/
def prob_white : ℝ := 0.25

/-- The probability of drawing either a black ball or a white ball from the box -/
def prob_black_or_white : ℝ := 1 - prob_red

theorem black_or_white_probability : prob_black_or_white = 0.55 := by
  sorry

end black_or_white_probability_l2909_290915


namespace odd_function_zero_l2909_290999

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_zero (f : ℝ → ℝ) (h : OddFunction f) : f 0 = 0 := by
  sorry

end odd_function_zero_l2909_290999


namespace bread_cost_l2909_290988

def total_cost : ℕ := 42
def banana_cost : ℕ := 12
def milk_cost : ℕ := 7
def apple_cost : ℕ := 14

theorem bread_cost : 
  total_cost - (banana_cost + milk_cost + apple_cost) = 9 := by
  sorry

end bread_cost_l2909_290988


namespace cube_volume_from_space_diagonal_l2909_290950

/-- The volume of a cube given its space diagonal length. -/
theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  (d / Real.sqrt 3) ^ 3 = 216 := by
  sorry

end cube_volume_from_space_diagonal_l2909_290950


namespace cricket_runs_l2909_290906

theorem cricket_runs (a b c : ℕ) (h1 : 3 * a = b) (h2 : 5 * b = c) (h3 : a + b + c = 95) :
  c = 75 := by
  sorry

end cricket_runs_l2909_290906


namespace monotonic_range_of_a_l2909_290938

/-- The piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a^2 - 1) * Real.exp (a * x)

/-- The function f is monotonic on ℝ -/
def is_monotonic (a : ℝ) : Prop :=
  Monotone (f a) ∨ StrictMono (f a)

/-- The theorem stating the range of a for which f is monotonic -/
theorem monotonic_range_of_a :
  ∀ a : ℝ, is_monotonic a ↔ a ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ioo 1 (Real.sqrt 2) := by
  sorry

end monotonic_range_of_a_l2909_290938


namespace remainder_problem_l2909_290993

theorem remainder_problem (d : ℤ) (r : ℤ) 
  (h1 : d > 1)
  (h2 : 1237 % d = r)
  (h3 : 1694 % d = r)
  (h4 : 2791 % d = r) :
  d - r = 134 := by
sorry

end remainder_problem_l2909_290993


namespace unique_integer_solution_l2909_290918

theorem unique_integer_solution : 
  ∀ x y : ℤ, x^2 - 2*x*y + 2*y^2 - 4*y^3 = 0 → x = 0 ∧ y = 0 := by
  sorry

end unique_integer_solution_l2909_290918


namespace Z_set_eq_roster_l2909_290964

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the set we want to prove
def Z_set : Set ℂ := {z | ∃ n : ℤ, z = i^n + i^(-n)}

-- The theorem to prove
theorem Z_set_eq_roster : Z_set = {0, 2, -2} := by sorry

end Z_set_eq_roster_l2909_290964


namespace remainder_of_2007_pow_2008_mod_10_l2909_290995

theorem remainder_of_2007_pow_2008_mod_10 : 2007^2008 % 10 = 1 := by
  sorry

end remainder_of_2007_pow_2008_mod_10_l2909_290995


namespace susan_age_l2909_290961

theorem susan_age (susan joe billy : ℕ) 
  (h1 : susan = 2 * joe)
  (h2 : susan + joe + billy = 60)
  (h3 : billy = joe + 10) :
  susan = 25 := by sorry

end susan_age_l2909_290961


namespace a_plus_b_value_l2909_290925

theorem a_plus_b_value (a b : ℝ) (ha : |a| = 5) (hb : |b| = 2) (hab : a < b) :
  a + b = -3 := by
  sorry

end a_plus_b_value_l2909_290925


namespace total_nails_calculation_l2909_290958

/-- The number of nails left at each station -/
def nails_per_station : ℕ := 7

/-- The number of stations visited -/
def stations_visited : ℕ := 20

/-- The total number of nails brought -/
def total_nails : ℕ := nails_per_station * stations_visited

theorem total_nails_calculation : total_nails = 140 := by
  sorry

end total_nails_calculation_l2909_290958


namespace marys_age_l2909_290974

theorem marys_age :
  ∃! x : ℕ, 
    (∃ n : ℕ, x - 2 = n^2) ∧ 
    (∃ m : ℕ, x + 2 = m^3) ∧ 
    x = 6 := by
  sorry

end marys_age_l2909_290974


namespace min_value_sum_of_squares_l2909_290952

theorem min_value_sum_of_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_eq_9 : x + y + z = 9) : 
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end min_value_sum_of_squares_l2909_290952


namespace wade_drink_cost_l2909_290962

/-- The cost of each drink given Wade's purchases -/
theorem wade_drink_cost (total_spent : ℝ) (sandwich_cost : ℝ) (num_sandwiches : ℕ) (num_drinks : ℕ) 
  (h1 : total_spent = 26)
  (h2 : sandwich_cost = 6)
  (h3 : num_sandwiches = 3)
  (h4 : num_drinks = 2) :
  (total_spent - num_sandwiches * sandwich_cost) / num_drinks = 4 := by
  sorry

end wade_drink_cost_l2909_290962


namespace polynomial_identity_sum_of_squares_l2909_290991

theorem polynomial_identity_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x, 729 * x^3 + 64 = (a*x^2 + b*x + c) * (d*x^2 + e*x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 := by
sorry

end polynomial_identity_sum_of_squares_l2909_290991


namespace max_value_of_f_l2909_290904

open Real

theorem max_value_of_f (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∃ (max_val : ℝ), max_val = 3 * Real.sqrt 3 ∧
  ∀ y ∈ Set.Ioo 0 (π / 2), 8 * sin y - tan y ≤ max_val :=
by
  sorry

end max_value_of_f_l2909_290904


namespace twelfth_even_multiple_of_4_l2909_290919

/-- The nth term in the sequence of positive integers that are both even and multiples of 4 -/
def evenMultipleOf4 (n : ℕ) : ℕ := 4 * n

/-- Theorem stating that the 12th term in the sequence of positive integers 
    that are both even and multiples of 4 is equal to 48 -/
theorem twelfth_even_multiple_of_4 : evenMultipleOf4 12 = 48 := by
  sorry

end twelfth_even_multiple_of_4_l2909_290919


namespace distinct_triangles_count_l2909_290929

/-- The maximum exponent for the line segment lengths -/
def max_exponent : ℕ := 10

/-- The set of line segment lengths -/
def segment_lengths : Set ℕ := {n | ∃ k : ℕ, 0 ≤ k ∧ k ≤ max_exponent ∧ n = 2^k}

/-- A function to check if three lengths can form a nondegenerate triangle -/
def is_nondegenerate_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The number of distinct nondegenerate triangles -/
def num_distinct_triangles : ℕ := Nat.choose (max_exponent + 1) 2

theorem distinct_triangles_count :
  num_distinct_triangles = 55 := by sorry

end distinct_triangles_count_l2909_290929


namespace football_yards_lost_l2909_290977

theorem football_yards_lost (yards_gained yards_progress : ℤ) 
  (h1 : yards_gained = 8)
  (h2 : yards_progress = 3) :
  ∃ yards_lost : ℤ, yards_lost + yards_gained = yards_progress ∧ yards_lost = -5 :=
by
  sorry

end football_yards_lost_l2909_290977


namespace bite_size_samples_per_half_l2909_290922

def total_pies : ℕ := 13
def halves_per_pie : ℕ := 2
def total_tasters : ℕ := 130

theorem bite_size_samples_per_half : 
  (total_tasters / (total_pies * halves_per_pie) : ℚ) = 5 := by
  sorry

end bite_size_samples_per_half_l2909_290922


namespace inequality_theorem_equality_theorem_l2909_290996

-- Define the condition
def condition (x y : ℝ) : Prop := (x + 1) * (y + 2) = 8

-- Define the main theorem
theorem inequality_theorem (x y : ℝ) (h : condition x y) :
  (x * y - 10)^2 ≥ 64 ∧
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) :=
by sorry

-- Define the equality cases
def equality_cases (x y : ℝ) : Prop :=
  (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)

-- Theorem for the equality cases
theorem equality_theorem (x y : ℝ) (h : condition x y) :
  (x * y - 10)^2 = 64 ↔ equality_cases x y :=
by sorry

end inequality_theorem_equality_theorem_l2909_290996


namespace modified_geometric_series_sum_l2909_290989

/-- The sum of a modified geometric series -/
theorem modified_geometric_series_sum 
  (a r : ℝ) 
  (h_r : -1 < r ∧ r < 1) :
  let series_sum : ℕ → ℝ := λ n => a^2 * r^(3*n)
  ∑' n, series_sum n = a^2 / (1 - r^3) := by
  sorry

end modified_geometric_series_sum_l2909_290989


namespace polynomial_division_remainder_l2909_290900

theorem polynomial_division_remainder :
  ∃ (q r : Polynomial ℝ),
    x^4 + 5 = (x^2 - 4*x + 7) * q + r ∧
    r.degree < (x^2 - 4*x + 7).degree ∧
    r = 8*x - 58 := by
  sorry

end polynomial_division_remainder_l2909_290900


namespace binomial_12_11_l2909_290921

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end binomial_12_11_l2909_290921


namespace remainder_x_50_divided_by_x2_minus_4x_plus_3_l2909_290998

theorem remainder_x_50_divided_by_x2_minus_4x_plus_3 (x : ℝ) :
  ∃ (Q : ℝ → ℝ), x^50 = (x^2 - 4*x + 3) * Q x + ((3^50 - 1)/2 * x + (5 - 3^50)/2) :=
by sorry

end remainder_x_50_divided_by_x2_minus_4x_plus_3_l2909_290998


namespace arithmetic_mean_of_4_and_16_l2909_290944

theorem arithmetic_mean_of_4_and_16 (x : ℝ) :
  x = (4 + 16) / 2 → x = 10 := by
  sorry

end arithmetic_mean_of_4_and_16_l2909_290944


namespace investment_equation_l2909_290937

/-- Proves that the total amount invested satisfies the given equation based on the problem conditions -/
theorem investment_equation (total_interest : ℝ) (higher_rate_fraction : ℝ) 
  (lower_rate : ℝ) (higher_rate : ℝ) :
  total_interest = 1440 →
  higher_rate_fraction = 0.55 →
  lower_rate = 0.06 →
  higher_rate = 0.09 →
  ∃ T : ℝ, 0.0765 * T = 1440 :=
by
  sorry

end investment_equation_l2909_290937


namespace union_complement_equality_l2909_290980

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_equality : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_equality_l2909_290980


namespace son_work_time_l2909_290982

-- Define the work rates
def man_rate : ℚ := 1 / 6
def combined_rate : ℚ := 1 / 3

-- Define the son's work rate
def son_rate : ℚ := combined_rate - man_rate

-- Theorem to prove
theorem son_work_time : (1 : ℚ) / son_rate = 6 := by sorry

end son_work_time_l2909_290982


namespace cube_face_sum_l2909_290940

/-- Represents the six positive integers on the faces of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- Calculates the sum of vertex labels given the face values -/
def vertexSum (faces : CubeFaces) : ℕ :=
  (faces.a * faces.b * faces.c) + (faces.a * faces.e * faces.c) +
  (faces.a * faces.b * faces.f) + (faces.a * faces.e * faces.f) +
  (faces.d * faces.b * faces.c) + (faces.d * faces.e * faces.c) +
  (faces.d * faces.b * faces.f) + (faces.d * faces.e * faces.f)

/-- Calculates the sum of all face values -/
def faceSum (faces : CubeFaces) : ℕ :=
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f

/-- Theorem: If the vertex sum is 1452, then the face sum is 47 -/
theorem cube_face_sum (faces : CubeFaces) :
  vertexSum faces = 1452 → faceSum faces = 47 := by
  sorry

end cube_face_sum_l2909_290940


namespace star_three_neg_four_star_not_commutative_l2909_290934

-- Define the new operation "*" for rational numbers
def star (a b : ℚ) : ℚ := 2 * a - 1 + b

-- Theorem 1: 3 * (-4) = 1
theorem star_three_neg_four : star 3 (-4) = 1 := by sorry

-- Theorem 2: 7 * (-3) ≠ (-3) * 7
theorem star_not_commutative : star 7 (-3) ≠ star (-3) 7 := by sorry

end star_three_neg_four_star_not_commutative_l2909_290934


namespace womens_haircut_cost_l2909_290949

theorem womens_haircut_cost :
  let childrens_haircut_cost : ℝ := 36
  let num_children : ℕ := 2
  let tip_percentage : ℝ := 0.20
  let tip_amount : ℝ := 24
  let womens_haircut_cost : ℝ := 48
  tip_amount = tip_percentage * (womens_haircut_cost + num_children * childrens_haircut_cost) :=
by
  sorry

end womens_haircut_cost_l2909_290949


namespace exists_negative_greater_than_neg_half_l2909_290984

theorem exists_negative_greater_than_neg_half : ∃ x : ℚ, -1/2 < x ∧ x < 0 := by
  sorry

end exists_negative_greater_than_neg_half_l2909_290984


namespace joan_pinball_spending_l2909_290955

/-- The amount of money in dollars represented by a half-dollar -/
def half_dollar_value : ℚ := 0.5

/-- The total amount spent in dollars given the number of half-dollars spent each day -/
def total_spent (wed thur fri : ℕ) : ℚ :=
  half_dollar_value * (wed + thur + fri : ℚ)

/-- Theorem stating that if Joan spent 4 half-dollars on Wednesday, 14 on Thursday,
    and 8 on Friday, then the total amount she spent playing pinball is $13.00 -/
theorem joan_pinball_spending :
  total_spent 4 14 8 = 13 := by sorry

end joan_pinball_spending_l2909_290955


namespace fraction_equality_with_different_numerator_denominator_relations_l2909_290912

theorem fraction_equality_with_different_numerator_denominator_relations : 
  ∃ (a b c d : ℤ), a < b ∧ c > d ∧ (a : ℚ) / b = (c : ℚ) / d := by
  sorry

end fraction_equality_with_different_numerator_denominator_relations_l2909_290912


namespace curve_decomposition_l2909_290907

-- Define the curve
def curve (x y : ℝ) : Prop := (x + y - 1) * Real.sqrt (x - 1) = 0

-- Define the line x = 1
def line (x y : ℝ) : Prop := x = 1

-- Define the ray x + y - 1 = 0 where x ≥ 1
def ray (x y : ℝ) : Prop := x + y - 1 = 0 ∧ x ≥ 1

-- Theorem statement
theorem curve_decomposition :
  ∀ x y : ℝ, x ≥ 1 → (curve x y ↔ line x y ∨ ray x y) :=
sorry

end curve_decomposition_l2909_290907


namespace equation_system_solutions_l2909_290978

/-- A system of two equations with two unknowns x and y -/
def equation_system (x y : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3) = 0 ∧
  (|x - 1| + |y - 1|) * (|x - 2| + |y - 2|) * (|x - 3| + |y - 4|) = 0

/-- The theorem stating that the equation system has only three specific solutions -/
theorem equation_system_solutions :
  ∀ x y : ℝ, equation_system x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 3 ∧ y = 4) :=
by sorry

end equation_system_solutions_l2909_290978


namespace total_balls_correct_l2909_290924

/-- The number of yellow balls in the bag -/
def yellow_balls : ℕ := 6

/-- The probability of drawing a yellow ball -/
def yellow_probability : ℚ := 3/10

/-- The total number of balls in the bag -/
def total_balls : ℕ := 20

/-- Theorem stating that the total number of balls is correct given the conditions -/
theorem total_balls_correct : 
  (yellow_balls : ℚ) / total_balls = yellow_probability :=
by sorry

end total_balls_correct_l2909_290924


namespace paper_folding_height_l2909_290979

/-- Given a square piece of paper with side length 100 cm, 
    with cuts from each corner starting 8 cm from the corner and meeting at 45°,
    prove that the perpendicular height of the folded shape is 8 cm. -/
theorem paper_folding_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 100 →
  cut_distance = 8 →
  cut_angle = 45 →
  let diagonal_length := side_length * Real.sqrt 2
  let cut_length := cut_distance * Real.sqrt 2
  let height := Real.sqrt (cut_length^2 - (cut_length / 2)^2)
  height = 8 := by
  sorry

end paper_folding_height_l2909_290979


namespace extremum_and_nonnegative_conditions_l2909_290968

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - x) - Real.log x

theorem extremum_and_nonnegative_conditions (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ∈ Set.Ioo (1 - ε) (1 + ε) → f a x ≥ f a 1) →
  a = 1 ∧
  (∀ (x : ℝ), x ≥ 1 → f a x ≥ 0) ↔ a ≥ 1 :=
by sorry

end extremum_and_nonnegative_conditions_l2909_290968


namespace profit_percentage_l2909_290917

theorem profit_percentage (selling_price : ℝ) (cost_price : ℝ) (h : cost_price = 0.89 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (1 / 0.89 - 1) * 100 := by
  sorry

end profit_percentage_l2909_290917


namespace unique_solution_iff_nonzero_l2909_290910

theorem unique_solution_iff_nonzero (a : ℝ) :
  (a ≠ 0) ↔ (∃! x : ℝ, a * x = 1) :=
by sorry

end unique_solution_iff_nonzero_l2909_290910
