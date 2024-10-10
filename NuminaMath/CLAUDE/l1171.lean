import Mathlib

namespace line_intersects_x_axis_l1171_117198

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point2D
  p2 : Point2D

/-- Checks if a point lies on the x-axis -/
def isOnXAxis (p : Point2D) : Prop :=
  p.y = 0

/-- Checks if a point lies on a given line -/
def isOnLine (l : Line) (p : Point2D) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

theorem line_intersects_x_axis (l : Line) : 
  l.p1 = ⟨3, -1⟩ → l.p2 = ⟨7, 3⟩ → 
  ∃ p : Point2D, isOnLine l p ∧ isOnXAxis p ∧ p = ⟨4, 0⟩ := by
  sorry

end line_intersects_x_axis_l1171_117198


namespace farmer_water_capacity_l1171_117126

/-- Calculates the total water capacity for a farmer's trucks -/
def total_water_capacity (num_trucks : ℕ) (tanks_per_truck : ℕ) (liters_per_tank : ℕ) : ℕ :=
  num_trucks * tanks_per_truck * liters_per_tank

/-- Theorem stating the total water capacity for the farmer's specific setup -/
theorem farmer_water_capacity :
  total_water_capacity 3 3 150 = 1350 := by
  sorry

end farmer_water_capacity_l1171_117126


namespace abs_equation_solution_set_l1171_117147

theorem abs_equation_solution_set (x : ℝ) :
  |2*x - 1| = |x| + |x - 1| ↔ x ≤ 0 ∨ x ≥ 1 := by
  sorry

end abs_equation_solution_set_l1171_117147


namespace sum_of_roots_l1171_117156

theorem sum_of_roots (k d x₁ x₂ : ℝ) 
  (h₁ : x₁ ≠ x₂)
  (h₂ : 5 * x₁^2 - k * x₁ = d)
  (h₃ : 5 * x₂^2 - k * x₂ = d) :
  x₁ + x₂ = k / 5 := by
sorry

end sum_of_roots_l1171_117156


namespace sum_of_x1_and_x2_l1171_117141

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- State the theorem
theorem sum_of_x1_and_x2 (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ → f x₁ = 101 → f x₂ = 101 → x₁ + x₂ = 2 := by
  sorry

end sum_of_x1_and_x2_l1171_117141


namespace stratified_sampling_elderly_count_l1171_117189

def total_population : ℕ := 180
def elderly_population : ℕ := 30
def sample_size : ℕ := 36

theorem stratified_sampling_elderly_count :
  (elderly_population * sample_size) / total_population = 6 :=
sorry

end stratified_sampling_elderly_count_l1171_117189


namespace nickel_piles_count_l1171_117101

/-- Represents the number of coins in each pile -/
def coins_per_pile : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the number of piles of quarters -/
def quarter_piles : ℕ := 4

/-- Represents the number of piles of dimes -/
def dime_piles : ℕ := 6

/-- Represents the number of piles of pennies -/
def penny_piles : ℕ := 5

/-- Represents the total value Rocco has in cents -/
def total_value : ℕ := 2100

/-- Theorem stating that the number of piles of nickels is 9 -/
theorem nickel_piles_count : 
  ∃ (nickel_piles : ℕ), 
    nickel_piles = 9 ∧
    quarter_piles * coins_per_pile * quarter_value + 
    dime_piles * coins_per_pile * dime_value + 
    nickel_piles * coins_per_pile * nickel_value + 
    penny_piles * coins_per_pile * penny_value = 
    total_value :=
by sorry

end nickel_piles_count_l1171_117101


namespace arithmetic_sequence_value_increasing_sequence_set_l1171_117104

def sequence_sum (a : ℝ) (n : ℕ) : ℝ := sorry

def sequence_term (a : ℝ) (n : ℕ) : ℝ := sorry

axiom sequence_sum_property (a : ℝ) (n : ℕ) :
  n ≥ 2 → (sequence_sum a n)^2 = 3 * n^2 * (sequence_term a n) + (sequence_sum a (n-1))^2

axiom nonzero_terms (a : ℝ) (n : ℕ) : sequence_term a n ≠ 0

def is_arithmetic_sequence (a : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → sequence_term a (n+1) - sequence_term a n = sequence_term a n - sequence_term a (n-1)

def is_increasing_sequence (a : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → sequence_term a n < sequence_term a (n+1)

theorem arithmetic_sequence_value (a : ℝ) :
  is_arithmetic_sequence a → a = 3 := sorry

theorem increasing_sequence_set :
  {a : ℝ | is_increasing_sequence a} = Set.Ioo (9/4) (15/4) := sorry

end arithmetic_sequence_value_increasing_sequence_set_l1171_117104


namespace simplify_expression_l1171_117133

theorem simplify_expression (x y : ℝ) : 7 * x + 8 - 3 * x + 15 - 2 * y = 4 * x - 2 * y + 23 := by
  sorry

end simplify_expression_l1171_117133


namespace right_triangle_circle_and_trajectory_l1171_117127

/-- Right triangle ABC with hypotenuse AB, where A(-1,0) and B(3,0) -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  hA : A = (-1, 0)
  hB : B = (3, 0)
  isRightTriangle : sorry -- Assume this triangle is right-angled

/-- The general equation of a circle -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The equation of a trajectory -/
def TrajectoryEquation (a b r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

theorem right_triangle_circle_and_trajectory 
  (triangle : RightTriangle) (x y : ℝ) (hy : y ≠ 0) :
  (CircleEquation 1 0 2 x y ↔ x^2 + y^2 - 2*x - 3 = 0) ∧
  (TrajectoryEquation 2 0 1 x y ↔ (x-2)^2 + y^2 = 1) := by
  sorry

end right_triangle_circle_and_trajectory_l1171_117127


namespace parabola_translation_l1171_117123

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally -/
def translate_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c }

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b
  , c := p.c + v }

theorem parabola_translation :
  let p1 : Parabola := { a := 2, b := 4, c := -3 }  -- y = 2(x+1)^2 - 3
  let p2 : Parabola := translate_vertical (translate_horizontal p1 1) 3
  p2 = { a := 2, b := 0, c := 0 }  -- y = 2x^2
  := by sorry

end parabola_translation_l1171_117123


namespace quadratic_inequality_range_l1171_117145

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 4 * a * x + 3 > 0) → 0 ≤ a ∧ a < 3/4 := by
sorry

end quadratic_inequality_range_l1171_117145


namespace gold_medals_count_l1171_117172

theorem gold_medals_count (total : ℕ) (silver : ℕ) (bronze : ℕ) (h1 : total = 67) (h2 : silver = 32) (h3 : bronze = 16) :
  total - silver - bronze = 19 := by
  sorry

end gold_medals_count_l1171_117172


namespace one_common_root_sum_l1171_117110

theorem one_common_root_sum (a b : ℝ) :
  (∃! x, x^2 + a*x + b = 0 ∧ x^2 + b*x + a = 0) →
  a + b = -1 :=
by sorry

end one_common_root_sum_l1171_117110


namespace possible_regions_l1171_117146

/-- The number of lines dividing the plane -/
def num_lines : ℕ := 99

/-- The function that calculates the number of regions based on the number of parallel lines -/
def num_regions (k : ℕ) : ℕ := (k + 1) * (100 - k)

/-- The theorem stating the possible values of n less than 199 -/
theorem possible_regions :
  ∀ n : ℕ, n < 199 →
    (∃ k : ℕ, k ≤ num_lines ∧ n = num_regions k) →
    n = 100 ∨ n = 198 := by
  sorry

end possible_regions_l1171_117146


namespace sandy_average_price_per_book_l1171_117169

/-- Represents a bookshop visit with the number of books bought and the total price paid -/
structure BookshopVisit where
  books : ℕ
  price : ℚ

/-- Calculates the average price per book given a list of bookshop visits -/
def averagePricePerBook (visits : List BookshopVisit) : ℚ :=
  (visits.map (λ v => v.price)).sum / (visits.map (λ v => v.books)).sum

/-- The theorem statement for Sandy's bookshop visits -/
theorem sandy_average_price_per_book :
  let visits : List BookshopVisit := [
    { books := 65, price := 1080 },
    { books := 55, price := 840 },
    { books := 45, price := 765 },
    { books := 35, price := 630 }
  ]
  averagePricePerBook visits = 16575 / 1000 := by
  sorry


end sandy_average_price_per_book_l1171_117169


namespace final_week_hours_l1171_117139

def hours_worked : List ℕ := [14, 10, 13, 9, 12, 11]
def total_weeks : ℕ := 7
def required_average : ℕ := 12

theorem final_week_hours :
  ∃ (x : ℕ), (List.sum hours_worked + x) / total_weeks = required_average :=
by sorry

end final_week_hours_l1171_117139


namespace point_on_line_l1171_117140

/-- The line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_through_points (x₁ y₁ x₂ y₂ : ℚ) (x y : ℚ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

/-- The theorem stating that (-3/7, 8) lies on the line through (3, 0) and (0, 7) -/
theorem point_on_line : line_through_points 3 0 0 7 (-3/7) 8 := by
  sorry

end point_on_line_l1171_117140


namespace fraction_equality_l1171_117171

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 2 / 3) :
  t / q = 1 := by sorry

end fraction_equality_l1171_117171


namespace marble_count_l1171_117132

/-- Represents a bag of marbles with red, blue, and green colors -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Theorem: Given a bag of marbles with the specified conditions, 
    prove the total number of marbles and the number of red marbles -/
theorem marble_count (bag : MarbleBag) 
  (ratio : bag.red * 3 * 4 = bag.blue * 2 * 4 ∧ bag.blue * 2 * 4 = bag.green * 2 * 3)
  (green_count : bag.green = 36) :
  bag.red + bag.blue + bag.green = 81 ∧ bag.red = 18 := by
  sorry

end marble_count_l1171_117132


namespace inverse_function_point_sum_l1171_117137

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverse functions
axiom inverse_relation : ∀ x, f_inv (f x) = x

-- Define the condition that (1,2) is on the graph of y = f(x)/2
axiom point_on_graph : f 1 = 4

-- Theorem to prove
theorem inverse_function_point_sum :
  ∃ a b : ℝ, f_inv a = 2*b ∧ a + b = 9/2 :=
sorry

end inverse_function_point_sum_l1171_117137


namespace bowl_glass_pairings_l1171_117181

/-- The number of bowl colors -/
def numBowls : ℕ := 5

/-- The number of glass colors -/
def numGlasses : ℕ := 4

/-- The total number of possible pairings without restrictions -/
def totalPairings : ℕ := numBowls * numGlasses

/-- The number of restricted pairings (purple bowl with green glass) -/
def restrictedPairings : ℕ := 1

/-- The number of valid pairings -/
def validPairings : ℕ := totalPairings - restrictedPairings

theorem bowl_glass_pairings :
  validPairings = 19 :=
sorry

end bowl_glass_pairings_l1171_117181


namespace complement_of_A_l1171_117193

def A : Set ℝ := {x | (x - 1) / (x - 2) ≥ 0}

theorem complement_of_A : (Set.univ \ A) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end complement_of_A_l1171_117193


namespace expression_evaluation_l1171_117168

theorem expression_evaluation (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  ((((x+2)^2 * (x^2-2*x+4)^2) / (x^3+8)^2)^2) * ((((x-2)^2 * (x^2+2*x+4)^2) / (x^3-8)^2)^2) = 1 :=
by sorry

end expression_evaluation_l1171_117168


namespace longest_side_l1171_117108

-- Define the triangle
def triangle (x : ℝ) := {a : ℝ // a = 7 ∨ a = x^2 + 4 ∨ a = 3*x + 1}

-- Define the perimeter condition
def perimeter_condition (x : ℝ) : Prop := 7 + (x^2 + 4) + (3*x + 1) = 45

-- State the theorem
theorem longest_side (x : ℝ) (h : perimeter_condition x) : 
  ∀ (side : triangle x), side.val ≤ x^2 + 4 :=
sorry

end longest_side_l1171_117108


namespace vasya_has_winning_strategy_l1171_117184

/-- Represents a game state -/
structure GameState where
  board : List Nat
  currentPlayer : Bool  -- true for Petya, false for Vasya

/-- Checks if a list of numbers contains an arithmetic progression -/
def hasArithmeticProgression (numbers : List Nat) : Bool :=
  sorry

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Nat) : Bool :=
  move ≤ 2018 ∧ move ∉ state.board

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Nat) : GameState :=
  { board := move :: state.board
  , currentPlayer := ¬state.currentPlayer }

/-- Represents a strategy for a player -/
def Strategy := GameState → Nat

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (strategy : Strategy) (player : Bool) : Prop :=
  ∀ (initialState : GameState),
    initialState.currentPlayer = player →
    ∃ (finalState : GameState),
      (finalState.board.length ≥ 3 ∧
       hasArithmeticProgression finalState.board) ∧
      finalState.currentPlayer = player

/-- The main theorem stating that Vasya (the second player) has a winning strategy -/
theorem vasya_has_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy strategy false :=
sorry

end vasya_has_winning_strategy_l1171_117184


namespace point_movement_and_linear_function_l1171_117150

theorem point_movement_and_linear_function (k : ℝ) : 
  let initial_point : ℝ × ℝ := (5, 3)
  let new_point : ℝ × ℝ := (initial_point.1 - 4, initial_point.2 - 1)
  new_point.2 = k * new_point.1 - 2 → k = 4 :=
by
  sorry

end point_movement_and_linear_function_l1171_117150


namespace equation_represents_two_lines_l1171_117131

/-- The equation represents two lines if it can be rewritten in the form of two linear equations -/
def represents_two_lines (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), ∀ x y, f x y = 0 ↔ (x = a * y + b ∨ x = c * y + d)

/-- The given equation -/
def equation (x y : ℝ) : ℝ :=
  x^2 - 25 * y^2 - 10 * x + 50

theorem equation_represents_two_lines :
  represents_two_lines equation :=
sorry

end equation_represents_two_lines_l1171_117131


namespace vet_donation_amount_l1171_117157

/-- Calculates the amount donated by the vet to an animal shelter during a pet adoption event. -/
theorem vet_donation_amount (dog_fee cat_fee : ℕ) (dog_adoptions cat_adoptions : ℕ) (donation_fraction : ℚ) : 
  dog_fee = 15 →
  cat_fee = 13 →
  dog_adoptions = 8 →
  cat_adoptions = 3 →
  donation_fraction = 1/3 →
  (dog_fee * dog_adoptions + cat_fee * cat_adoptions) * donation_fraction = 53 := by
  sorry

end vet_donation_amount_l1171_117157


namespace fixed_point_of_exponential_function_l1171_117199

/-- For a > 0 and a ≠ 1, the function f(x) = a^(x-2) - 3 passes through the point (2, -2) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x : ℝ, a^(x - 2) - 3 = x ∧ x = 2 := by
  sorry

end fixed_point_of_exponential_function_l1171_117199


namespace fraction_expression_value_l1171_117102

theorem fraction_expression_value (m n p : ℝ) (h : m + n - p = 0) :
  m * (1 / n - 1 / p) + n * (1 / m - 1 / p) - p * (1 / m + 1 / n) = -3 := by
  sorry

end fraction_expression_value_l1171_117102


namespace rectangular_to_polar_conversion_l1171_117188

theorem rectangular_to_polar_conversion :
  ∀ (x y r θ : ℝ),
  x = 8 ∧ y = 2 * Real.sqrt 3 →
  r = Real.sqrt (x^2 + y^2) →
  θ = Real.arctan (y / x) →
  r > 0 →
  0 ≤ θ ∧ θ < 2 * Real.pi →
  r = 2 * Real.sqrt 19 ∧ θ = Real.arctan (Real.sqrt 3 / 4) :=
by sorry

end rectangular_to_polar_conversion_l1171_117188


namespace arithmetic_sequence_difference_l1171_117138

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_difference : 
  let a := -8  -- First term
  let d := 7   -- Common difference (derived from -1 - (-8))
  let seq := arithmeticSequence a d
  (seq 110 - seq 100).natAbs = 70 := by sorry

end arithmetic_sequence_difference_l1171_117138


namespace fourth_term_is_27_l1171_117129

def S (n : ℕ) : ℤ := 4 * n^2 - n - 8

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem fourth_term_is_27 : a 4 = 27 := by
  sorry

end fourth_term_is_27_l1171_117129


namespace log_sum_two_five_equals_one_l1171_117190

theorem log_sum_two_five_equals_one : Real.log 2 + Real.log 5 = 1 := by
  sorry

end log_sum_two_five_equals_one_l1171_117190


namespace circular_garden_radius_l1171_117122

theorem circular_garden_radius (r : ℝ) : r > 0 → 2 * Real.pi * r = (1 / 5) * Real.pi * r^2 → r = 10 := by
  sorry

end circular_garden_radius_l1171_117122


namespace triangle_sine_problem_l1171_117155

theorem triangle_sine_problem (D E F : ℝ) (h_area : (1/2) * D * E * Real.sin F = 100) 
  (h_geom_mean : Real.sqrt (D * E) = 15) : Real.sin F = 8/9 := by
  sorry

end triangle_sine_problem_l1171_117155


namespace arthur_arrival_speed_l1171_117107

theorem arthur_arrival_speed :
  ∀ (distance : ℝ) (n : ℝ),
    (distance / 60 = distance / n + 1/12) →
    (distance / 90 = distance / n - 1/12) →
    n = 72 := by
  sorry

end arthur_arrival_speed_l1171_117107


namespace events_mutually_exclusive_not_complementary_l1171_117148

-- Define the set of people
inductive Person : Type
  | A
  | B
  | C

-- Define the set of cards
inductive Card : Type
  | Red
  | Yellow
  | Blue

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person B gets the red card"
def event_B (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A d ∧ event_B d)) ∧
  (∃ d : Distribution, ¬event_A d ∧ ¬event_B d) :=
sorry

end events_mutually_exclusive_not_complementary_l1171_117148


namespace equal_face_parallelepiped_implies_rhombus_l1171_117167

/-- A parallelepiped with equal parallelogram faces -/
structure EqualFaceParallelepiped where
  /-- The length of the first edge -/
  a : ℝ
  /-- The length of the second edge -/
  b : ℝ
  /-- The length of the third edge -/
  c : ℝ
  /-- All edges have positive length -/
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  /-- All faces have equal area -/
  equal_faces : a * b = b * c ∧ b * c = a * c

/-- A rhombus is a quadrilateral with all sides equal -/
def is_rhombus (s₁ s₂ s₃ s₄ : ℝ) : Prop :=
  s₁ = s₂ ∧ s₂ = s₃ ∧ s₃ = s₄

/-- If all 6 faces of a parallelepiped are equal parallelograms, then they are rhombuses -/
theorem equal_face_parallelepiped_implies_rhombus (P : EqualFaceParallelepiped) :
  is_rhombus P.a P.a P.a P.a ∧
  is_rhombus P.b P.b P.b P.b ∧
  is_rhombus P.c P.c P.c P.c :=
sorry

end equal_face_parallelepiped_implies_rhombus_l1171_117167


namespace teacher_assignment_count_l1171_117118

/-- The number of ways to assign teachers to classes -/
def assign_teachers (n : ℕ) (m : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- The number of intern teachers -/
def num_teachers : ℕ := 4

/-- The number of classes -/
def num_classes : ℕ := 3

/-- Theorem stating that the number of ways to assign 4 teachers to 3 classes,
    with each class having at least 1 teacher, is 36 -/
theorem teacher_assignment_count :
  assign_teachers num_teachers num_classes = 36 :=
sorry

end teacher_assignment_count_l1171_117118


namespace exponential_max_greater_than_min_l1171_117187

theorem exponential_max_greater_than_min (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x ∈ Set.Icc 1 2 ∧ y ∈ Set.Icc 1 2 ∧ a^x > a^y :=
sorry

end exponential_max_greater_than_min_l1171_117187


namespace expression_bounds_l1171_117176

theorem expression_bounds (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  let expr := |x + y + z| / (|x| + |y| + |z|)
  (∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 → |a + b + c| / (|a| + |b| + |c|) ≤ 1) ∧
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ |a + b + c| / (|a| + |b| + |c|) = 0) ∧
  (1 - 0 = 1) := by
sorry

end expression_bounds_l1171_117176


namespace largest_divisor_of_m_l1171_117117

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 72 ∣ m^3) : 
  ∀ k : ℕ, k ∣ m → k ≤ 6 ∧ 6 ∣ m :=
sorry

end largest_divisor_of_m_l1171_117117


namespace shirt_pricing_theorem_l1171_117166

/-- Represents the monthly sales volume as a function of price -/
def sales_volume (x : ℝ) : ℝ := -20 * x + 2600

/-- Represents the profit as a function of price -/
def profit (x : ℝ) : ℝ := (x - 50) * (sales_volume x)

/-- The cost price of each shirt -/
def cost_price : ℝ := 50

/-- The constraint that selling price is not less than cost price -/
def price_constraint (x : ℝ) : Prop := x ≥ cost_price

/-- The constraint that profit per unit should not exceed 30% of cost price -/
def profit_constraint (x : ℝ) : Prop := (x - cost_price) / cost_price ≤ 0.3

theorem shirt_pricing_theorem :
  ∃ (x : ℝ), price_constraint x ∧ profit x = 24000 ∧ x = 70 ∧
  ∃ (y : ℝ), price_constraint y ∧ profit_constraint y ∧
    (∀ z, price_constraint z → profit_constraint z → profit z ≤ profit y) ∧
    y = 65 ∧ profit y = 19500 := by sorry

end shirt_pricing_theorem_l1171_117166


namespace prob_square_divisor_15_factorial_l1171_117134

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of positive integer divisors of n that are perfect squares -/
def num_square_divisors (n : ℕ) : ℕ := sorry

/-- The total number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The probability of choosing a perfect square divisor from the positive integer divisors of n -/
def prob_square_divisor (n : ℕ) : ℚ :=
  (num_square_divisors n : ℚ) / (num_divisors n : ℚ)

theorem prob_square_divisor_15_factorial :
  prob_square_divisor (factorial 15) = 1 / 36 := by sorry

end prob_square_divisor_15_factorial_l1171_117134


namespace jelly_bean_division_l1171_117125

theorem jelly_bean_division (initial_amount : ℕ) (eaten_amount : ℕ) (num_piles : ℕ) :
  initial_amount = 36 →
  eaten_amount = 6 →
  num_piles = 3 →
  (initial_amount - eaten_amount) / num_piles = 10 :=
by
  sorry

end jelly_bean_division_l1171_117125


namespace parabola_equation_l1171_117170

/-- A parabola passing through two points on the x-axis -/
structure Parabola where
  a : ℝ
  b : ℝ
  eval : ℝ → ℝ := fun x => a * x^2 + b * x - 5

/-- The parabola passes through the points (-1,0) and (5,0) -/
def passes_through (p : Parabola) : Prop :=
  p.eval (-1) = 0 ∧ p.eval 5 = 0

/-- The theorem stating that the parabola passing through (-1,0) and (5,0) has the equation y = x² - 4x - 5 -/
theorem parabola_equation (p : Parabola) (h : passes_through p) :
  p.a = 1 ∧ p.b = -4 := by sorry

end parabola_equation_l1171_117170


namespace negative_division_rule_div_negative_64_negative_32_l1171_117183

theorem negative_division_rule (x y : ℤ) (hy : y ≠ 0) : (-x) / (-y) = x / y := by sorry

theorem div_negative_64_negative_32 : (-64) / (-32) = 2 := by sorry

end negative_division_rule_div_negative_64_negative_32_l1171_117183


namespace wang_loss_is_97_l1171_117106

-- Define the relevant quantities
def gift_cost : ℕ := 18
def gift_price : ℕ := 21
def payment : ℕ := 100
def change_given : ℕ := 79
def counterfeit_bill : ℕ := 100
def neighbor_repayment : ℕ := 100

-- Define Mr. Wang's loss
def wang_loss : ℕ := change_given + gift_cost + neighbor_repayment - payment

-- Theorem statement
theorem wang_loss_is_97 : wang_loss = 97 := by
  sorry

end wang_loss_is_97_l1171_117106


namespace different_color_chips_probability_l1171_117151

theorem different_color_chips_probability :
  let total_chips : ℕ := 9
  let blue_chips : ℕ := 6
  let yellow_chips : ℕ := 3
  let prob_blue_then_yellow : ℚ := (blue_chips / total_chips) * (yellow_chips / (total_chips - 1))
  let prob_yellow_then_blue : ℚ := (yellow_chips / total_chips) * (blue_chips / (total_chips - 1))
  prob_blue_then_yellow + prob_yellow_then_blue = 1 / 2 := by
sorry

end different_color_chips_probability_l1171_117151


namespace square_circle_relation_l1171_117120

theorem square_circle_relation (s : ℝ) (h : s > 0) :
  (4 * s = π * (s / Real.sqrt 2)^2) → s = 8 / π := by
  sorry

end square_circle_relation_l1171_117120


namespace jessica_probability_is_37_966_l1171_117130

/-- Represents the problem of distributing textbooks into boxes. -/
structure TextbookDistribution where
  total_books : Nat
  english_books : Nat
  box1_capacity : Nat
  box2_capacity : Nat
  box3_capacity : Nat
  box4_capacity : Nat

/-- The specific textbook distribution problem given in the question. -/
def jessica_distribution : TextbookDistribution :=
  { total_books := 15
  , english_books := 4
  , box1_capacity := 3
  , box2_capacity := 4
  , box3_capacity := 5
  , box4_capacity := 3
  }

/-- Calculates the probability of all English textbooks ending up in the third box. -/
def probability_all_english_in_third_box (d : TextbookDistribution) : Rat :=
  sorry

/-- Theorem stating that the probability for Jessica's distribution is 37/966. -/
theorem jessica_probability_is_37_966 :
  probability_all_english_in_third_box jessica_distribution = 37 / 966 := by
  sorry

end jessica_probability_is_37_966_l1171_117130


namespace not_divisible_by_100_l1171_117162

theorem not_divisible_by_100 : ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by
  sorry

end not_divisible_by_100_l1171_117162


namespace largest_square_area_l1171_117161

theorem largest_square_area (x y z : ℝ) (h1 : x^2 + y^2 = z^2) 
  (h2 : x^2 + y^2 + 2*z^2 = 722) : z^2 = 722/3 := by
  sorry

end largest_square_area_l1171_117161


namespace balloon_difference_l1171_117128

theorem balloon_difference (yellow_balloons : ℕ) (total_balloons : ℕ) (school_balloons : ℕ) :
  yellow_balloons = 3414 →
  total_balloons % 10 = 0 →
  total_balloons / 10 = school_balloons →
  school_balloons = 859 →
  total_balloons > 2 * yellow_balloons →
  total_balloons - 2 * yellow_balloons = 1762 := by
  sorry

end balloon_difference_l1171_117128


namespace problem_statement_l1171_117115

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a^2 + 3*b^2 ≥ 2*b*(a + b)) ∧ 
  ((1/a + 2/b = 1) → (2*a + b ≥ 8)) :=
by sorry

end problem_statement_l1171_117115


namespace fruits_in_buckets_l1171_117164

/-- The number of fruits in three buckets -/
def total_fruits (a b c : ℕ) : ℕ := a + b + c

/-- Theorem: The total number of fruits in three buckets is 37 -/
theorem fruits_in_buckets :
  ∀ (a b c : ℕ),
  c = 9 →
  b = c + 3 →
  a = b + 4 →
  total_fruits a b c = 37 :=
by
  sorry

end fruits_in_buckets_l1171_117164


namespace intersection_M_N_l1171_117196

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l1171_117196


namespace geometric_sequence_sum_l1171_117149

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q ^ (n - 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 3 + a 5 = 6 →
  a 5 + a 7 + a 9 = 28 := by
  sorry

end geometric_sequence_sum_l1171_117149


namespace complex_power_of_sqrt2i_l1171_117185

theorem complex_power_of_sqrt2i :
  ∀ z : ℂ, z = Complex.I * Real.sqrt 2 → z^4 = 4 := by
  sorry

end complex_power_of_sqrt2i_l1171_117185


namespace pentagon_angle_sum_l1171_117144

theorem pentagon_angle_sum (P Q R a b : ℝ) : 
  P = 34 → Q = 82 → R = 30 → 
  (P + Q + (360 - a) + 90 + (120 - b) = 540) → 
  a + b = 146 := by sorry

end pentagon_angle_sum_l1171_117144


namespace sum_of_sqrt_greater_than_one_l1171_117160

theorem sum_of_sqrt_greater_than_one 
  (x y z t : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0)
  (hxy : x ≠ y) (hxz : x ≠ z) (hxt : x ≠ t)
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t)
  (hsum : x + y + z + t = 1) :
  (Real.sqrt x + Real.sqrt y > 1) ∨
  (Real.sqrt x + Real.sqrt z > 1) ∨
  (Real.sqrt x + t > 1) ∨
  (Real.sqrt y + Real.sqrt z > 1) ∨
  (Real.sqrt y + Real.sqrt t > 1) ∨
  (Real.sqrt z + Real.sqrt t > 1) := by
  sorry

end sum_of_sqrt_greater_than_one_l1171_117160


namespace square_difference_l1171_117192

theorem square_difference (x y : ℝ) : (x - y) * (x - y) = x^2 - 2*x*y + y^2 := by
  sorry

end square_difference_l1171_117192


namespace bank_transfer_result_l1171_117100

def initial_balance : ℕ := 27004
def transfer_amount : ℕ := 69

theorem bank_transfer_result :
  initial_balance - transfer_amount = 26935 :=
by sorry

end bank_transfer_result_l1171_117100


namespace dryer_ball_savings_l1171_117116

/-- Calculates the savings from using wool dryer balls instead of dryer sheets over two years -/
theorem dryer_ball_savings :
  let loads_per_month : ℕ := 4 + 5 + 6 + 7
  let loads_per_year : ℕ := loads_per_month * 12
  let sheets_per_box : ℕ := 104
  let boxes_per_year : ℕ := (loads_per_year + sheets_per_box - 1) / sheets_per_box
  let initial_box_price : ℝ := 5.50
  let price_increase_rate : ℝ := 0.025
  let dryer_ball_price : ℝ := 15

  let first_year_cost : ℝ := boxes_per_year * initial_box_price
  let second_year_cost : ℝ := boxes_per_year * (initial_box_price * (1 + price_increase_rate))
  let total_sheet_cost : ℝ := first_year_cost + second_year_cost

  let savings : ℝ := total_sheet_cost - dryer_ball_price

  savings = 18.4125 := by sorry

end dryer_ball_savings_l1171_117116


namespace mode_most_relevant_for_sales_volume_l1171_117109

/-- Represents a shoe size -/
def ShoeSize := ℕ

/-- Represents a list of shoe sizes sold -/
def SalesList := List ShoeSize

/-- Calculates the mode of a list of shoe sizes -/
def mode (sales : SalesList) : ShoeSize :=
  sorry

/-- Represents the relevance of a statistical measure for determining the shoe size with highest sales volume -/
inductive Relevance
| Low : Relevance
| Medium : Relevance
| High : Relevance

/-- Determines the relevance of a statistical measure for sales volume prediction -/
def relevanceForSalesVolume (measure : String) : Relevance :=
  sorry

theorem mode_most_relevant_for_sales_volume :
  relevanceForSalesVolume "mode" = Relevance.High ∧
  (∀ m : String, m ≠ "mode" → relevanceForSalesVolume m ≠ Relevance.High) :=
sorry

end mode_most_relevant_for_sales_volume_l1171_117109


namespace sweets_distribution_l1171_117135

theorem sweets_distribution (total_children : Nat) (absent_children : Nat) (extra_sweets : Nat) 
  (h1 : total_children = 256)
  (h2 : absent_children = 64)
  (h3 : extra_sweets = 12) :
  let original_sweets := (total_children - absent_children) * extra_sweets / absent_children
  original_sweets = 36 := by
sorry

end sweets_distribution_l1171_117135


namespace cube_edge_length_is_ten_l1171_117136

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

/-- Checks if a point is inside a cube -/
def isInside (p : Point3D) (c : Cube) : Prop :=
  0 < p.x ∧ p.x < c.edgeLength ∧
  0 < p.y ∧ p.y < c.edgeLength ∧
  0 < p.z ∧ p.z < c.edgeLength

/-- Theorem: If there exists an interior point with specific distances from four vertices of a cube,
    then the edge length of the cube is 10 -/
theorem cube_edge_length_is_ten (c : Cube) (p : Point3D) 
    (v1 v2 v3 v4 : Point3D) : 
    isInside p c →
    squaredDistance p v1 = 50 →
    squaredDistance p v2 = 70 →
    squaredDistance p v3 = 90 →
    squaredDistance p v4 = 110 →
    (v1.x = 0 ∨ v1.x = c.edgeLength) ∧ 
    (v1.y = 0 ∨ v1.y = c.edgeLength) ∧ 
    (v1.z = 0 ∨ v1.z = c.edgeLength) →
    (v2.x = 0 ∨ v2.x = c.edgeLength) ∧ 
    (v2.y = 0 ∨ v2.y = c.edgeLength) ∧ 
    (v2.z = 0 ∨ v2.z = c.edgeLength) →
    (v3.x = 0 ∨ v3.x = c.edgeLength) ∧ 
    (v3.y = 0 ∨ v3.y = c.edgeLength) ∧ 
    (v3.z = 0 ∨ v3.z = c.edgeLength) →
    (v4.x = 0 ∨ v4.x = c.edgeLength) ∧ 
    (v4.y = 0 ∨ v4.y = c.edgeLength) ∧ 
    (v4.z = 0 ∨ v4.z = c.edgeLength) →
    (v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4) →
    c.edgeLength = 10 := by
  sorry


end cube_edge_length_is_ten_l1171_117136


namespace equal_distribution_of_sweets_l1171_117153

/-- Proves that each student receives 4 sweet treats given the conditions -/
theorem equal_distribution_of_sweets
  (cookies : ℕ) (cupcakes : ℕ) (brownies : ℕ) (students : ℕ)
  (h_cookies : cookies = 20)
  (h_cupcakes : cupcakes = 25)
  (h_brownies : brownies = 35)
  (h_students : students = 20)
  : (cookies + cupcakes + brownies) / students = 4 := by
  sorry

end equal_distribution_of_sweets_l1171_117153


namespace difference_ones_zeros_235_l1171_117124

def base_10_to_base_2 (n : ℕ) : List ℕ :=
  sorry

def count_zeros (l : List ℕ) : ℕ :=
  sorry

def count_ones (l : List ℕ) : ℕ :=
  sorry

theorem difference_ones_zeros_235 :
  let binary_235 := base_10_to_base_2 235
  let w := count_ones binary_235
  let z := count_zeros binary_235
  w - z = 2 := by sorry

end difference_ones_zeros_235_l1171_117124


namespace intersection_of_lines_l1171_117163

theorem intersection_of_lines : ∃! p : ℚ × ℚ, 
  8 * p.1 - 5 * p.2 = 20 ∧ 6 * p.1 + 2 * p.2 = 18 :=
by
  -- The proof would go here
  sorry

end intersection_of_lines_l1171_117163


namespace ages_sum_l1171_117103

theorem ages_sum (diane_future_age diane_current_age : ℕ) 
  (h1 : diane_future_age = 30)
  (h2 : diane_current_age = 16) : ∃ (alex_age allison_age : ℕ), 
  (diane_future_age = alex_age / 2) ∧ 
  (diane_future_age = allison_age * 2) ∧
  (alex_age + allison_age = 47) :=
by sorry

end ages_sum_l1171_117103


namespace max_distance_to_line_l1171_117180

/-- The maximum distance from the point (1, 1) to the line x*cos(θ) + y*sin(θ) = 2 is 2 + √2 -/
theorem max_distance_to_line : 
  let P : ℝ × ℝ := (1, 1)
  let line (θ : ℝ) (x y : ℝ) := x * Real.cos θ + y * Real.sin θ = 2
  ∃ (d : ℝ), d = 2 + Real.sqrt 2 ∧ 
    ∀ (θ : ℝ), d ≥ Real.sqrt ((P.1 * Real.cos θ + P.2 * Real.sin θ - 2) ^ 2) :=
by sorry

end max_distance_to_line_l1171_117180


namespace unique_solution_mod_151_l1171_117178

theorem unique_solution_mod_151 :
  ∃! n : ℤ, 0 ≤ n ∧ n < 151 ∧ (150 * n) % 151 = 93 % 151 ∧ n = 58 := by
  sorry

end unique_solution_mod_151_l1171_117178


namespace function_identity_l1171_117114

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ m n, m < n → f m < f n

theorem function_identity (f : ℕ → ℕ) 
  (h_increasing : StrictlyIncreasing f)
  (h_two : f 2 = 2)
  (h_coprime : ∀ m n, Nat.Coprime m n → f (m * n) = f m * f n) :
  ∀ n, f n = n :=
sorry

end function_identity_l1171_117114


namespace integer_root_iff_a_value_l1171_117173

def polynomial (a x : ℤ) : ℤ := x^4 + 4*x^3 + a*x^2 + 8

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial a x = 0

theorem integer_root_iff_a_value :
  ∀ a : ℤ, has_integer_root a ↔ a = -14 ∨ a = -13 ∨ a = -5 ∨ a = 2 :=
sorry

end integer_root_iff_a_value_l1171_117173


namespace cricket_problem_l1171_117159

/-- Represents the runs scored by each batsman in a cricket match -/
structure CricketScores where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ

/-- Theorem representing the cricket problem -/
theorem cricket_problem (scores : CricketScores) : scores.E = 20 :=
  by
  have h1 : scores.A + scores.B + scores.C + scores.D + scores.E = 180 := 
    sorry -- Average score is 36, so total is 5 * 36 = 180
  have h2 : scores.D = scores.E + 5 := 
    sorry -- D scored 5 more than E
  have h3 : scores.E = scores.A - 8 := 
    sorry -- E scored 8 fewer than A
  have h4 : scores.B = scores.D + scores.E := 
    sorry -- B scored as many as D and E combined
  have h5 : scores.B + scores.C = 107 := 
    sorry -- B and C scored 107 between them
  sorry -- Proof that E = 20

end cricket_problem_l1171_117159


namespace patio_length_l1171_117191

theorem patio_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = 4 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 100 →
  length = 40 := by
sorry

end patio_length_l1171_117191


namespace janet_cat_collars_l1171_117179

/-- The number of inches of nylon needed for a dog collar -/
def dog_collar_nylon : ℕ := 18

/-- The number of inches of nylon needed for a cat collar -/
def cat_collar_nylon : ℕ := 10

/-- The total number of inches of nylon Janet needs -/
def total_nylon : ℕ := 192

/-- The number of dog collars Janet needs to make -/
def num_dog_collars : ℕ := 9

/-- Theorem stating that Janet needs to make 3 cat collars -/
theorem janet_cat_collars : 
  (total_nylon - num_dog_collars * dog_collar_nylon) / cat_collar_nylon = 3 := by
  sorry

end janet_cat_collars_l1171_117179


namespace student_rank_theorem_l1171_117182

/-- Given a group of students, calculate the rank from left based on total students and rank from right -/
def rankFromLeft (totalStudents : ℕ) (rankFromRight : ℕ) : ℕ :=
  totalStudents - rankFromRight + 1

/-- Theorem stating that in a group of 10 students, the 6th from right is 5th from left -/
theorem student_rank_theorem :
  let totalStudents : ℕ := 10
  let rankFromRight : ℕ := 6
  rankFromLeft totalStudents rankFromRight = 5 := by
  sorry


end student_rank_theorem_l1171_117182


namespace factor_calculation_l1171_117105

theorem factor_calculation (original : ℝ) (factor : ℝ) : 
  original = 5 → 
  (2 * original + 9) * factor = 57 → 
  factor = 3 := by
sorry

end factor_calculation_l1171_117105


namespace jasper_sold_31_drinks_l1171_117152

/-- Represents the number of items sold by Jasper -/
structure JasperSales where
  chips : ℕ
  hot_dogs : ℕ
  drinks : ℕ

/-- Calculates the number of drinks sold by Jasper -/
def calculate_drinks (sales : JasperSales) : ℕ :=
  sales.chips - 8 + 12

/-- Theorem stating that Jasper sold 31 drinks -/
theorem jasper_sold_31_drinks (sales : JasperSales) 
  (h1 : sales.chips = 27)
  (h2 : sales.hot_dogs = sales.chips - 8)
  (h3 : sales.drinks = sales.hot_dogs + 12) :
  sales.drinks = 31 := by
  sorry

end jasper_sold_31_drinks_l1171_117152


namespace fraction_simplification_l1171_117121

theorem fraction_simplification :
  (1 - 2 - 4 + 8 + 16 + 32 - 64 + 128 - 256) /
  (2 - 4 - 8 + 16 + 32 + 64 - 128 + 256) = 1 / 2 := by
  sorry

end fraction_simplification_l1171_117121


namespace units_digit_17_pow_28_l1171_117197

theorem units_digit_17_pow_28 : (17^28) % 10 = 1 := by
  sorry

end units_digit_17_pow_28_l1171_117197


namespace find_extremes_l1171_117194

/-- Represents the result of a weighing -/
inductive CompareResult
  | Less : CompareResult
  | Equal : CompareResult
  | Greater : CompareResult

/-- Represents a weight -/
structure Weight where
  id : Nat

/-- Represents a weighing operation -/
def weighing (w1 w2 : Weight) : CompareResult := sorry

/-- Represents the set of 5 weights -/
def Weights : Type := Fin 5 → Weight

/-- The heaviest weight in the set -/
def heaviest (ws : Weights) : Weight := sorry

/-- The lightest weight in the set -/
def lightest (ws : Weights) : Weight := sorry

/-- Axiom: Three weights have the same weight -/
axiom three_same_weight (ws : Weights) : 
  ∃ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    weighing (ws i) (ws j) = CompareResult.Equal ∧ 
    weighing (ws j) (ws k) = CompareResult.Equal

/-- Axiom: One weight is heavier than the three identical weights -/
axiom one_heavier (ws : Weights) : 
  ∃ (h : Fin 5), ∀ (i : Fin 5), 
    weighing (ws i) (ws h) = CompareResult.Less ∨ 
    weighing (ws i) (ws h) = CompareResult.Equal

/-- Axiom: One weight is lighter than the three identical weights -/
axiom one_lighter (ws : Weights) : 
  ∃ (l : Fin 5), ∀ (i : Fin 5), 
    weighing (ws l) (ws i) = CompareResult.Less ∨ 
    weighing (ws l) (ws i) = CompareResult.Equal

/-- Theorem: It's possible to determine the heaviest and lightest weights in at most three weighings -/
theorem find_extremes (ws : Weights) : 
  ∃ (w1 w2 w3 w4 w5 w6 : Weight), 
    (weighing w1 w2 = CompareResult.Less ∨ 
     weighing w1 w2 = CompareResult.Equal ∨ 
     weighing w1 w2 = CompareResult.Greater) ∧
    (weighing w3 w4 = CompareResult.Less ∨ 
     weighing w3 w4 = CompareResult.Equal ∨ 
     weighing w3 w4 = CompareResult.Greater) ∧
    (weighing w5 w6 = CompareResult.Less ∨ 
     weighing w5 w6 = CompareResult.Equal ∨ 
     weighing w5 w6 = CompareResult.Greater) →
    (heaviest ws = heaviest ws ∧ lightest ws = lightest ws) :=
  sorry

end find_extremes_l1171_117194


namespace extreme_value_implies_a_range_l1171_117158

/-- A function f with an extreme value only at x = 0 -/
def f (a b x : ℝ) : ℝ := x^4 + a*x^3 + 2*x^2 + b

/-- f has an extreme value only at x = 0 -/
def has_extreme_only_at_zero (a b : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → ¬(∀ y : ℝ, f a b y ≤ f a b x ∨ ∀ y : ℝ, f a b y ≥ f a b x)

/-- The main theorem: if f has an extreme value only at x = 0, then -8/3 ≤ a ≤ 8/3 -/
theorem extreme_value_implies_a_range (a b : ℝ) :
  has_extreme_only_at_zero a b → -8/3 ≤ a ∧ a ≤ 8/3 := by
  sorry

end extreme_value_implies_a_range_l1171_117158


namespace solution_equation1_solution_equation2_l1171_117165

-- Define the equations
def equation1 (x : ℝ) : Prop := 6 * x - 7 = 4 * x - 5
def equation2 (x : ℝ) : Prop := (x + 1) / 2 - 1 = 2 + (2 - x) / 4

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by
  sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = 4 := by
  sorry

end solution_equation1_solution_equation2_l1171_117165


namespace property_P_lower_bound_l1171_117195

/-- Property P for a function f: ℝ → ℝ -/
def has_property_P (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, Real.sqrt (2 * f x) - Real.sqrt (2 * f x - f (2 * x)) ≥ 2

/-- The theorem stating that if f has property P, then f(x) ≥ 12 + 8√2 for all real x -/
theorem property_P_lower_bound (f : ℝ → ℝ) (h : has_property_P f) :
  ∀ x : ℝ, f x ≥ 12 + 8 * Real.sqrt 2 := by
  sorry

end property_P_lower_bound_l1171_117195


namespace smallest_a_value_l1171_117142

theorem smallest_a_value (a b d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) :
  (∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x + d)) →
  ∃ k : ℤ, a = 17 - 2 * Real.pi * ↑k ∧ 
    ∀ a' : ℝ, (0 ≤ a' ∧ ∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x + d)) → 
      17 - 2 * Real.pi * ↑k ≤ a' :=
by sorry

end smallest_a_value_l1171_117142


namespace no_real_roots_for_nonzero_k_l1171_117143

theorem no_real_roots_for_nonzero_k :
  ∀ k : ℝ, k ≠ 0 → ¬∃ x : ℝ, x^2 + k*x + k^2 = 0 := by
sorry

end no_real_roots_for_nonzero_k_l1171_117143


namespace space_diagonals_of_Q_l1171_117119

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - (2 * Q.quadrilateral_faces)

/-- Theorem: The number of space diagonals in the given polyhedron Q is 315 -/
theorem space_diagonals_of_Q :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 72 ∧
    Q.faces = 44 ∧
    Q.triangular_faces = 20 ∧
    Q.quadrilateral_faces = 24 ∧
    space_diagonals Q = 315 :=
sorry

end space_diagonals_of_Q_l1171_117119


namespace yellow_square_area_percentage_l1171_117113

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  /-- Side length of the square flag -/
  side : ℝ
  /-- Width of each arm of the cross (equal to side length of yellow square) -/
  crossWidth : ℝ
  /-- Assumption that the cross width is positive and less than the flag side -/
  crossWidthValid : 0 < crossWidth ∧ crossWidth < side

/-- The area of the entire flag -/
def SquareFlag.area (flag : SquareFlag) : ℝ := flag.side ^ 2

/-- The area of the cross (including yellow center) -/
def SquareFlag.crossArea (flag : SquareFlag) : ℝ :=
  4 * flag.side * flag.crossWidth - 3 * flag.crossWidth ^ 2

/-- The area of the yellow square at the center -/
def SquareFlag.yellowArea (flag : SquareFlag) : ℝ := flag.crossWidth ^ 2

/-- Theorem stating that if the cross occupies 49% of the flag's area, 
    then the yellow square occupies 12.25% of the flag's area -/
theorem yellow_square_area_percentage (flag : SquareFlag) 
  (h : flag.crossArea = 0.49 * flag.area) : 
  flag.yellowArea / flag.area = 0.1225 := by
  sorry

end yellow_square_area_percentage_l1171_117113


namespace prob_non_defective_product_l1171_117154

theorem prob_non_defective_product (prob_grade_b prob_grade_c : ℝ) 
  (h1 : prob_grade_b = 0.03)
  (h2 : prob_grade_c = 0.01)
  (h3 : 0 ≤ prob_grade_b ∧ prob_grade_b ≤ 1)
  (h4 : 0 ≤ prob_grade_c ∧ prob_grade_c ≤ 1)
  (h5 : prob_grade_b + prob_grade_c ≤ 1) :
  1 - (prob_grade_b + prob_grade_c) = 0.96 := by
sorry

end prob_non_defective_product_l1171_117154


namespace quadratic_one_root_l1171_117175

/-- A quadratic function f(x) = x^2 - 2x + m has exactly one root if and only if m = 1 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x, x^2 - 2*x + m = 0) ↔ m = 1 := by
  sorry

end quadratic_one_root_l1171_117175


namespace triangle_radius_inequalities_l1171_117186

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define perimeter, circumradius, and inradius
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c
def circumradius (t : Triangle) : ℝ := sorry
def inradius (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_radius_inequalities :
  ∃ t1 t2 t3 : Triangle,
    ¬(perimeter t1 > circumradius t1 + inradius t1) ∧
    ¬(perimeter t2 ≤ circumradius t2 + inradius t2) ∧
    ¬(perimeter t3 / 6 < circumradius t3 + inradius t3 ∧ circumradius t3 + inradius t3 < 6 * perimeter t3) :=
  sorry

end triangle_radius_inequalities_l1171_117186


namespace cycle_selling_price_l1171_117111

/-- Calculates the selling price of an item given its cost price and gain percentage. -/
def selling_price (cost : ℕ) (gain_percent : ℕ) : ℕ :=
  cost + (cost * gain_percent) / 100

/-- Theorem: If a cycle is bought for Rs. 1000 and sold with a 100% gain, the selling price is Rs. 2000. -/
theorem cycle_selling_price :
  selling_price 1000 100 = 2000 := by
  sorry

#eval selling_price 1000 100

end cycle_selling_price_l1171_117111


namespace boris_bowls_l1171_117174

def candy_distribution (initial_candy : ℕ) (daughter_eats : ℕ) (boris_takes : ℕ) (remaining_in_bowl : ℕ) : ℕ :=
  let remaining_candy := initial_candy - daughter_eats
  let pieces_per_bowl := remaining_in_bowl + boris_takes
  remaining_candy / pieces_per_bowl

theorem boris_bowls :
  candy_distribution 100 8 3 20 = 4 := by
  sorry

end boris_bowls_l1171_117174


namespace f_derivative_at_one_l1171_117112

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.log x

theorem f_derivative_at_one : 
  deriv f 1 = Real.cos 1 + 1 := by sorry

end f_derivative_at_one_l1171_117112


namespace expand_expression_l1171_117177

theorem expand_expression (x : ℝ) : (x + 3) * (4 * x - 8) - 2 * x = 4 * x^2 + 2 * x - 24 := by
  sorry

end expand_expression_l1171_117177
