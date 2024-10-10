import Mathlib

namespace mod_equivalence_unique_solution_l3773_377382

theorem mod_equivalence_unique_solution :
  ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end mod_equivalence_unique_solution_l3773_377382


namespace base4_to_decimal_example_l3773_377347

/-- Converts a base-4 number to decimal --/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base-4 representation of the number --/
def base4Number : List Nat := [2, 1, 0, 0, 3]

/-- Theorem: The base-4 number 30012₍₄₎ is equal to 774 in decimal notation --/
theorem base4_to_decimal_example : base4ToDecimal base4Number = 774 := by
  sorry

end base4_to_decimal_example_l3773_377347


namespace congruence_problem_l3773_377352

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ 123456 [ZMOD 7] := by
  sorry

end congruence_problem_l3773_377352


namespace even_painted_faces_count_l3773_377399

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Represents a cube cut from the block -/
structure Cube where
  x : Nat
  y : Nat
  z : Nat

/-- Returns the number of painted faces for a cube in the given position -/
def numPaintedFaces (b : Block) (c : Cube) : Nat :=
  sorry

/-- Returns true if the number is even -/
def isEven (n : Nat) : Bool :=
  sorry

/-- Counts the number of cubes with an even number of painted faces -/
def countEvenPaintedFaces (b : Block) : Nat :=
  sorry

/-- Theorem: In a 6x3x2 inch block painted on all sides and cut into 1 inch cubes,
    the number of cubes with an even number of painted faces is 20 -/
theorem even_painted_faces_count (b : Block) :
  b.length = 6 → b.width = 3 → b.height = 2 →
  countEvenPaintedFaces b = 20 :=
by sorry

end even_painted_faces_count_l3773_377399


namespace midpoint_distance_to_y_axis_l3773_377358

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the property of line m passing through focus and intersecting the parabola
def line_intersects_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A ∧ parabola B ∧ 
  (A.1 - focus.1) * (B.2 - focus.2) = (B.1 - focus.1) * (A.2 - focus.2)

-- Define the condition |AF| + |BF| = 10
def distance_sum_condition (A B : ℝ × ℝ) : Prop :=
  Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2) +
  Real.sqrt ((B.1 - focus.1)^2 + (B.2 - focus.2)^2) = 10

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (A B : ℝ × ℝ) 
  (h1 : line_intersects_parabola A B) 
  (h2 : distance_sum_condition A B) : 
  (A.1 + B.1) / 2 = 4 := by
  sorry

end midpoint_distance_to_y_axis_l3773_377358


namespace jackson_decorations_given_l3773_377376

/-- Given that Mrs. Jackson has 4 boxes of Christmas decorations with 15 decorations in each box
    and she used 35 decorations, prove that she gave 25 decorations to her neighbor. -/
theorem jackson_decorations_given (boxes : ℕ) (decorations_per_box : ℕ) (used_decorations : ℕ)
    (h1 : boxes = 4)
    (h2 : decorations_per_box = 15)
    (h3 : used_decorations = 35) :
    boxes * decorations_per_box - used_decorations = 25 := by
  sorry

end jackson_decorations_given_l3773_377376


namespace problem_1_problem_2_problem_3_problem_4_l3773_377374

-- Problem 1
theorem problem_1 : (1) - 6 - 13 + (-24) = -43 := by sorry

-- Problem 2
theorem problem_2 : (-6) / (3/7) * (-7) = 98 := by sorry

-- Problem 3
theorem problem_3 : (2/3 - 1/12 - 1/15) * (-60) = -31 := by sorry

-- Problem 4
theorem problem_4 : -1^4 - 1/6 * (2 - (-3)^2) = 1/6 := by sorry

end problem_1_problem_2_problem_3_problem_4_l3773_377374


namespace clothing_price_theorem_l3773_377315

/-- The price per item of clothing in yuan -/
def price : ℝ := 110

/-- The number of items sold in the first scenario -/
def quantity1 : ℕ := 10

/-- The number of items sold in the second scenario -/
def quantity2 : ℕ := 11

/-- The discount percentage in the first scenario -/
def discount_percent : ℝ := 0.08

/-- The discount amount in yuan for the second scenario -/
def discount_amount : ℝ := 30

theorem clothing_price_theorem :
  quantity1 * (price * (1 - discount_percent)) = quantity2 * (price - discount_amount) :=
sorry

end clothing_price_theorem_l3773_377315


namespace product_of_two_numbers_with_sum_100_l3773_377339

theorem product_of_two_numbers_with_sum_100 (a : ℝ) : 
  let b := 100 - a
  (a + b = 100) → (a * b = a * (100 - a)) := by
  sorry

end product_of_two_numbers_with_sum_100_l3773_377339


namespace expression_value_l3773_377371

theorem expression_value : 3 * ((18 + 7)^2 - (7^2 + 18^2)) = 756 := by
  sorry

end expression_value_l3773_377371


namespace tempo_original_value_l3773_377312

theorem tempo_original_value (insurance_ratio : ℚ) (premium_rate : ℚ) (premium : ℚ) :
  insurance_ratio = 5 / 7 →
  premium_rate = 3 / 100 →
  premium = 300 →
  ∃ (original_value : ℚ), original_value = 14000 ∧ 
    premium = premium_rate * insurance_ratio * original_value :=
by sorry

end tempo_original_value_l3773_377312


namespace smallest_multiple_37_3_mod_97_l3773_377333

theorem smallest_multiple_37_3_mod_97 : ∃ n : ℕ, 
  n > 0 ∧ 
  37 ∣ n ∧ 
  n % 97 = 3 ∧ 
  ∀ m : ℕ, m > 0 → 37 ∣ m → m % 97 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_multiple_37_3_mod_97_l3773_377333


namespace triangle_properties_l3773_377384

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  (t.a + t.c)^2 - t.b^2 = 3 * t.a * t.c ∧
  t.b = 6 ∧
  Real.sin t.C = 2 * Real.sin t.A

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 3 ∧ (1/2 * t.a * t.b : ℝ) = 6 * Real.sqrt 3 := by
  sorry

end triangle_properties_l3773_377384


namespace jack_collection_books_per_author_l3773_377397

/-- Represents Jack's classic book collection -/
structure ClassicCollection where
  authors : Nat
  total_books : Nat

/-- Calculates the number of books per author in a classic collection -/
def books_per_author (c : ClassicCollection) : Nat :=
  c.total_books / c.authors

/-- Theorem: In Jack's collection of 6 authors and 198 books, each author has 33 books -/
theorem jack_collection_books_per_author :
  let jack_collection : ClassicCollection := { authors := 6, total_books := 198 }
  books_per_author jack_collection = 33 := by
  sorry

end jack_collection_books_per_author_l3773_377397


namespace interest_rate_calculation_l3773_377326

theorem interest_rate_calculation (total_sum : ℝ) (second_part : ℝ) : 
  total_sum = 2665 →
  second_part = 1332.5 →
  let first_part := total_sum - second_part
  let interest_first := first_part * 0.03 * 5
  let interest_second := second_part * 0.03 * 3 * (5 : ℝ) / 3
  interest_first = interest_second →
  5 = 100 * interest_second / (second_part * 3) := by
sorry

end interest_rate_calculation_l3773_377326


namespace value_of_expression_l3773_377361

theorem value_of_expression (x y : ℚ) (hx : x = 2/3) (hy : y = 3/2) :
  (1/3) * x^5 * y^6 = 3/2 := by
  sorry

end value_of_expression_l3773_377361


namespace smooth_transition_iff_tangent_l3773_377383

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define tangency
def isTangent (c : Circle) (l : Line) (p : Point) : Prop :=
  -- The point lies on both the circle and the line
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  p.2 = l.slope * p.1 + l.intercept ∧
  -- The line is perpendicular to the radius at the point of tangency
  l.slope * (p.1 - c.center.1) = -(p.2 - c.center.2)

-- Define smooth transition
def smoothTransition (c : Circle) (l : Line) (p : Point) : Prop :=
  -- The velocity vector is continuous at the transition point
  isTangent c l p

-- Theorem statement
theorem smooth_transition_iff_tangent (c : Circle) (l : Line) (p : Point) :
  smoothTransition c l p ↔ isTangent c l p :=
sorry

end smooth_transition_iff_tangent_l3773_377383


namespace syrup_volume_l3773_377392

/-- The final volume of syrup after reduction and sugar addition -/
theorem syrup_volume (y : ℝ) : 
  let initial_volume : ℝ := 6 * 4  -- 6 quarts to cups
  let reduced_volume : ℝ := initial_volume * (1 / 12)
  let volume_with_sugar : ℝ := reduced_volume + 1
  let final_volume : ℝ := volume_with_sugar * y
  final_volume = 3 * y :=
by sorry

end syrup_volume_l3773_377392


namespace f_properties_implications_l3773_377369

/-- A function satisfying the given properties -/
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧  -- even function
  (∀ x, f x = f (2 - x)) ∧  -- symmetric about x = 1
  (∀ x₁ x₂, x₁ ∈ Set.Icc 0 (1/2) → x₂ ∈ Set.Icc 0 (1/2) → f (x₁ + x₂) = f x₁ * f x₂) ∧
  f 1 = 2

theorem f_properties_implications {f : ℝ → ℝ} (hf : f_properties f) :
  f (1/2) = Real.sqrt 2 ∧ f (1/4) = Real.sqrt (Real.sqrt 2) ∧ ∀ x, f x = f (x + 2) := by
  sorry

end f_properties_implications_l3773_377369


namespace equation_solution_equivalence_l3773_377337

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(-1, 2, 5), (-1, -5, -2), (-2, 1, 5), (-2, -5, -1), (5, 1, -2), (5, 2, -1)}

def satisfies_equations (x y z : ℝ) : Prop :=
  x - y + z = 2 ∧ x^2 + y^2 + z^2 = 30 ∧ x^3 - y^3 + z^3 = 116

theorem equation_solution_equivalence :
  ∀ x y z : ℝ, satisfies_equations x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end equation_solution_equivalence_l3773_377337


namespace trig_identity_l3773_377364

theorem trig_identity (α : Real) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end trig_identity_l3773_377364


namespace tangent_line_triangle_area_l3773_377311

/-- A line tangent to both y = x² and y = -1/x -/
structure TangentLine where
  -- The slope of the tangent line
  m : ℝ
  -- The y-intercept of the tangent line
  b : ℝ
  -- The x-coordinate of the point of tangency on y = x²
  x₁ : ℝ
  -- The x-coordinate of the point of tangency on y = -1/x
  x₂ : ℝ
  -- Condition: The line is tangent to y = x² at (x₁, x₁²)
  h₁ : m * x₁ + b = x₁^2
  -- Condition: The slope at the point of tangency on y = x² is correct
  h₂ : m = 2 * x₁
  -- Condition: The line is tangent to y = -1/x at (x₂, -1/x₂)
  h₃ : m * x₂ + b = -1 / x₂
  -- Condition: The slope at the point of tangency on y = -1/x is correct
  h₄ : m = 1 / x₂^2

/-- The area of the triangle formed by a tangent line and the coordinate axes is 2 -/
theorem tangent_line_triangle_area (l : TangentLine) : 
  (1 / 2) * (1 / l.m) * (-l.b) = 2 := by sorry

end tangent_line_triangle_area_l3773_377311


namespace track_length_proof_l3773_377388

/-- The length of the circular track -/
def track_length : ℝ := 600

/-- The distance Brenda runs before the first meeting -/
def brenda_first_distance : ℝ := 120

/-- The additional distance Sally runs between the first and second meeting -/
def sally_additional_distance : ℝ := 180

/-- Theorem stating the length of the track given the meeting conditions -/
theorem track_length_proof :
  ∃ (brenda_speed sally_speed : ℝ),
    brenda_speed > 0 ∧ sally_speed > 0 ∧
    brenda_first_distance / (track_length / 2 - brenda_first_distance) = brenda_speed / sally_speed ∧
    (track_length / 2 - brenda_first_distance + sally_additional_distance) / (brenda_first_distance + track_length / 2 - (track_length / 2 - brenda_first_distance + sally_additional_distance)) = sally_speed / brenda_speed :=
by
  sorry

end track_length_proof_l3773_377388


namespace marble_selection_probability_l3773_377323

/-- The number of blue marbles -/
def blue_marbles : ℕ := 7

/-- The number of yellow marbles -/
def yellow_marbles : ℕ := 5

/-- The total number of selections -/
def total_selections : ℕ := 7

/-- The number of blue marbles we want to select after the first yellow -/
def target_blue : ℕ := 3

/-- The probability of the described event -/
def probability : ℚ := 214375 / 1492992

theorem marble_selection_probability :
  (yellow_marbles : ℚ) / (yellow_marbles + blue_marbles) *
  (Nat.choose (total_selections - 1) target_blue : ℚ) *
  (blue_marbles ^ target_blue * yellow_marbles ^ (total_selections - target_blue - 1) : ℚ) /
  ((yellow_marbles + blue_marbles) ^ (total_selections - 1)) = probability :=
sorry

end marble_selection_probability_l3773_377323


namespace basketball_team_selection_l3773_377306

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def players_to_choose : ℕ := 8
def max_quadruplets : ℕ := 2

theorem basketball_team_selection :
  (Nat.choose total_players players_to_choose) -
  (Nat.choose quadruplets 3 * Nat.choose (total_players - quadruplets) (players_to_choose - 3) +
   Nat.choose quadruplets 4 * Nat.choose (total_players - quadruplets) (players_to_choose - 4)) = 34749 := by
  sorry

end basketball_team_selection_l3773_377306


namespace min_value_sqrt_reciprocal_min_value_achieved_l3773_377317

theorem min_value_sqrt_reciprocal (x : ℝ) (h : x > 0) : 
  3 * Real.sqrt (2 * x) + 4 / x ≥ 8 := by
  sorry

theorem min_value_achieved (x : ℝ) (h : x > 0) : 
  ∃ y > 0, 3 * Real.sqrt (2 * y) + 4 / y = 8 := by
  sorry

end min_value_sqrt_reciprocal_min_value_achieved_l3773_377317


namespace distance_ratio_car_a_to_b_l3773_377381

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

/-- Theorem: The ratio of distances covered by Car A and Car B is 3:1 -/
theorem distance_ratio_car_a_to_b (car_a car_b : Car)
    (h_speed_a : car_a.speed = 50)
    (h_time_a : car_a.time = 6)
    (h_speed_b : car_b.speed = 100)
    (h_time_b : car_b.time = 1) :
    distance car_a / distance car_b = 3 := by
  sorry

#check distance_ratio_car_a_to_b

end distance_ratio_car_a_to_b_l3773_377381


namespace complement_A_in_U_l3773_377303

def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

theorem complement_A_in_U : (U \ A) = {4} := by sorry

end complement_A_in_U_l3773_377303


namespace ratio_change_l3773_377355

theorem ratio_change (x y : ℤ) (n : ℤ) : 
  y = 72 → x / y = 1 / 4 → (x + n) / y = 1 / 3 → n = 6 := by
  sorry

end ratio_change_l3773_377355


namespace equal_roots_quadratic_l3773_377375

theorem equal_roots_quadratic (k B : ℝ) : 
  k = 1 → (∃ x : ℝ, 2 * k * x^2 + B * x + 2 = 0 ∧ 
    ∀ y : ℝ, 2 * k * y^2 + B * y + 2 = 0 → y = x) → 
  B = 4 ∨ B = -4 := by
  sorry

end equal_roots_quadratic_l3773_377375


namespace angle_ratio_not_right_triangle_l3773_377340

/-- Triangle ABC with angles A, B, and C in the ratio 3:4:5 is not necessarily a right triangle -/
theorem angle_ratio_not_right_triangle (A B C : ℝ) : 
  A / B = 3 / 4 ∧ B / C = 4 / 5 ∧ A + B + C = π → 
  ¬ (A = π / 2 ∨ B = π / 2 ∨ C = π / 2) :=
by sorry

end angle_ratio_not_right_triangle_l3773_377340


namespace bad_carrots_count_bad_carrots_problem_l3773_377304

theorem bad_carrots_count (nancy_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : ℕ :=
  let total_carrots := nancy_carrots + mom_carrots
  total_carrots - good_carrots

theorem bad_carrots_problem :
  bad_carrots_count 38 47 71 = 14 := by
  sorry

end bad_carrots_count_bad_carrots_problem_l3773_377304


namespace probability_standard_bulb_l3773_377349

/-- Probability of a bulb being from factory 1 -/
def p_factory1 : ℝ := 0.45

/-- Probability of a bulb being from factory 2 -/
def p_factory2 : ℝ := 0.40

/-- Probability of a bulb being from factory 3 -/
def p_factory3 : ℝ := 0.15

/-- Probability of a bulb from factory 1 being standard -/
def p_standard_factory1 : ℝ := 0.70

/-- Probability of a bulb from factory 2 being standard -/
def p_standard_factory2 : ℝ := 0.80

/-- Probability of a bulb from factory 3 being standard -/
def p_standard_factory3 : ℝ := 0.81

/-- The probability of purchasing a standard bulb from the store -/
theorem probability_standard_bulb :
  p_factory1 * p_standard_factory1 + 
  p_factory2 * p_standard_factory2 + 
  p_factory3 * p_standard_factory3 = 0.7565 := by
  sorry

end probability_standard_bulb_l3773_377349


namespace tan_function_property_l3773_377362

theorem tan_function_property (a : ℝ) : 
  let f : ℝ → ℝ := λ x => 1 + Real.tan x
  f a = 3 → f (-a) = -1 := by
sorry

end tan_function_property_l3773_377362


namespace candy_distribution_l3773_377366

/-- Represents the number of candies eaten by each person -/
structure CandyEaten where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Represents the relative eating rates of the three people -/
structure EatingRates where
  andrey_boris : ℚ  -- Ratio of Andrey's rate to Boris's rate
  andrey_denis : ℚ  -- Ratio of Andrey's rate to Denis's rate

/-- Given the eating rates and total candies eaten, calculate how many candies each person ate -/
def calculate_candy_eaten (rates : EatingRates) (total : ℕ) : CandyEaten :=
  sorry

/-- The main theorem to prove -/
theorem candy_distribution (rates : EatingRates) (total : ℕ) :
  rates.andrey_boris = 4/3 →
  rates.andrey_denis = 6/7 →
  total = 70 →
  let result := calculate_candy_eaten rates total
  result.andrey = 24 ∧ result.boris = 18 ∧ result.denis = 28 :=
sorry

end candy_distribution_l3773_377366


namespace quadratic_always_positive_l3773_377302

theorem quadratic_always_positive : ∀ x : ℝ, 3 * x^2 - 2 * x + 1 > 0 := by
  sorry

end quadratic_always_positive_l3773_377302


namespace red_balloons_count_l3773_377341

theorem red_balloons_count (total : ℕ) (green : ℕ) (h1 : total = 17) (h2 : green = 9) :
  total - green = 8 := by
  sorry

end red_balloons_count_l3773_377341


namespace f_max_value_l3773_377310

def f (x : ℝ) := x^3 - x^2 - x + 2

theorem f_max_value :
  (∃ x, f x = 1 ∧ ∀ y, f y ≥ f x) →
  (∃ x, f x = 59/27 ∧ ∀ y, f y ≤ f x) :=
by sorry

end f_max_value_l3773_377310


namespace negation_of_proposition_l3773_377344

theorem negation_of_proposition (A B : Set α) :
  ¬(∀ x, x ∈ A ∩ B → x ∈ A ∨ x ∈ B) ↔ (∃ x, x ∉ A ∩ B ∧ x ∉ A ∧ x ∉ B) :=
by sorry

end negation_of_proposition_l3773_377344


namespace victoria_snack_money_l3773_377324

theorem victoria_snack_money (initial_amount : ℕ) 
  (pizza_cost : ℕ) (pizza_quantity : ℕ)
  (juice_cost : ℕ) (juice_quantity : ℕ) :
  initial_amount = 50 →
  pizza_cost = 12 →
  pizza_quantity = 2 →
  juice_cost = 2 →
  juice_quantity = 2 →
  initial_amount - (pizza_cost * pizza_quantity + juice_cost * juice_quantity) = 22 := by
sorry


end victoria_snack_money_l3773_377324


namespace corner_sum_possibilities_l3773_377328

/-- Represents the color of a cell on the board -/
inductive CellColor
| Gold
| Silver

/-- Represents the board configuration -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (cellColor : Nat → Nat → CellColor)
  (vertexValue : Nat → Nat → Fin 2)

/-- Checks if a cell satisfies the sum condition based on its color -/
def validCell (b : Board) (row col : Nat) : Prop :=
  let sum := b.vertexValue row col + b.vertexValue row (col+1) +
             b.vertexValue (row+1) col + b.vertexValue (row+1) (col+1)
  match b.cellColor row col with
  | CellColor.Gold => sum % 2 = 0
  | CellColor.Silver => sum % 2 = 1

/-- Checks if the entire board configuration is valid -/
def validBoard (b : Board) : Prop :=
  b.rows = 2016 ∧ b.cols = 2017 ∧
  (∀ row col, row < b.rows → col < b.cols → validCell b row col) ∧
  (∀ row col, (row + col) % 2 = 0 → b.cellColor row col = CellColor.Gold) ∧
  (∀ row col, (row + col) % 2 = 1 → b.cellColor row col = CellColor.Silver)

/-- The sum of the four corner vertices of the board -/
def cornerSum (b : Board) : Nat :=
  b.vertexValue 0 0 + b.vertexValue 0 b.cols +
  b.vertexValue b.rows 0 + b.vertexValue b.rows b.cols

/-- Theorem stating the possible sums of the four corner vertices -/
theorem corner_sum_possibilities (b : Board) (h : validBoard b) :
  cornerSum b = 0 ∨ cornerSum b = 2 ∨ cornerSum b = 4 := by
  sorry

end corner_sum_possibilities_l3773_377328


namespace projection_of_a_onto_b_l3773_377348

def a : Fin 2 → ℝ := ![3, 4]
def b : Fin 2 → ℝ := ![1, 2]

theorem projection_of_a_onto_b :
  let proj := (((a 0) * (b 0) + (a 1) * (b 1)) / ((b 0)^2 + (b 1)^2)) • b
  proj 0 = 11/5 ∧ proj 1 = 22/5 := by
  sorry

end projection_of_a_onto_b_l3773_377348


namespace specific_tetrahedron_volume_l3773_377322

/-- Represents a tetrahedron with vertices P, Q, R, S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 3.5 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := 3
  }
  tetrahedronVolume t = 3.5 := by
  sorry

end specific_tetrahedron_volume_l3773_377322


namespace problem_1_problem_2_l3773_377327

-- Problem 1
theorem problem_1 : (1 - Real.sqrt 3) ^ 0 - |-(Real.sqrt 2)| + ((-27) ^ (1/3 : ℝ)) - ((-1/2) ^ (-1 : ℝ)) = -(Real.sqrt 2) := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x ≠ 2) : 
  ((x^2 - 1) / (x - 2) - x - 1) / ((x + 1) / (x^2 - 4*x + 4)) = x - 2 := by
  sorry

end problem_1_problem_2_l3773_377327


namespace sum_of_exponents_15_factorial_l3773_377367

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_perfect_square_divisor (n : ℕ) : ℕ :=
  sorry

def prime_factors (n : ℕ) : List ℕ :=
  sorry

def exponents_of_prime_factors (n : ℕ) : List ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  let n := factorial 15
  let largest_square := largest_perfect_square_divisor n
  let square_root := Nat.sqrt largest_square
  let exponents := exponents_of_prime_factors square_root
  List.sum exponents = 9 := by
  sorry

end sum_of_exponents_15_factorial_l3773_377367


namespace arithmetic_calculation_l3773_377391

theorem arithmetic_calculation : 
  (1 + 0.23 + 0.34) * (0.23 + 0.34 + 0.45) - (1 + 0.23 + 0.34 + 0.45) * (0.23 + 0.34) = 0.45 := by
  sorry

end arithmetic_calculation_l3773_377391


namespace johns_base_salary_l3773_377396

/-- John's monthly savings rate as a decimal -/
def savings_rate : ℝ := 0.10

/-- John's monthly savings amount in dollars -/
def savings_amount : ℝ := 400

/-- Theorem stating John's monthly base salary -/
theorem johns_base_salary :
  ∀ (base_salary : ℝ),
  base_salary * savings_rate = savings_amount →
  base_salary = 4000 := by
  sorry

end johns_base_salary_l3773_377396


namespace complement_A_intersect_B_l3773_377363

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {-1, 0, 2}

theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = {2} := by sorry

end complement_A_intersect_B_l3773_377363


namespace smallest_n_for_99n_all_threes_l3773_377313

def is_all_threes (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 3

theorem smallest_n_for_99n_all_threes :
  ∃! N : ℕ, (N > 0) ∧ 
    (is_all_threes (99 * N)) ∧ 
    (∀ m : ℕ, m > 0 → is_all_threes (99 * m) → N ≤ m) ∧
    N = 3367 :=
sorry

end smallest_n_for_99n_all_threes_l3773_377313


namespace inequality_theorem_l3773_377316

theorem inequality_theorem (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ((a < c ∧ c < b) ∨ (a < d ∧ d < b) ∨ (c < a ∧ a < d) ∨ (c < b ∧ b < d)) →
  Real.sqrt ((a + b) * (c + d)) > Real.sqrt (a * b) + Real.sqrt (c * d) :=
by sorry

end inequality_theorem_l3773_377316


namespace household_survey_l3773_377380

/-- Households survey problem -/
theorem household_survey (neither : ℕ) (only_w : ℕ) (both : ℕ) 
  (h1 : neither = 80)
  (h2 : only_w = 60)
  (h3 : both = 40) :
  neither + only_w + 3 * both + both = 300 := by
  sorry

end household_survey_l3773_377380


namespace common_factor_of_polynomial_l3773_377345

theorem common_factor_of_polynomial (a b : ℤ) :
  ∃ (k : ℤ), (6 * a^2 * b - 3 * a * b^2) = k * (3 * a * b) :=
by sorry

end common_factor_of_polynomial_l3773_377345


namespace jerry_feathers_left_l3773_377385

def feathers_left (hawk_feathers : ℕ) (eagle_ratio : ℕ) (given_away : ℕ) : ℕ :=
  let total_feathers := hawk_feathers + eagle_ratio * hawk_feathers
  let remaining_after_gift := total_feathers - given_away
  remaining_after_gift / 2

theorem jerry_feathers_left : feathers_left 6 17 10 = 49 := by
  sorry

end jerry_feathers_left_l3773_377385


namespace factorization_count_l3773_377386

theorem factorization_count : 
  ∃! (S : Finset ℤ), 
    (∀ m : ℤ, m ∈ S ↔ 
      ∃ a b : ℤ, ∀ x : ℝ, x^2 + m*x - 16 = (x + a)*(x + b)) ∧ 
    S.card = 5 := by
  sorry

end factorization_count_l3773_377386


namespace range_of_a_l3773_377331

-- Define the propositions p and q
def p (x a : ℝ) : Prop := |x - a| < 3
def q (x : ℝ) : Prop := x^2 - 2*x - 3 < 0

-- Define the theorem
theorem range_of_a :
  (∀ x, q x → p x a) ∧ 
  (∃ x, p x a ∧ ¬q x) →
  0 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l3773_377331


namespace exp_gt_one_plus_x_l3773_377378

theorem exp_gt_one_plus_x : ∀ x : ℝ, x > 0 → Real.exp x > 1 + x := by sorry

end exp_gt_one_plus_x_l3773_377378


namespace roger_has_more_candy_l3773_377305

-- Define the number of bags for Sandra and Roger
def sandra_bags : ℕ := 2
def roger_bags : ℕ := 2

-- Define the number of candies in each of Sandra's bags
def sandra_candy_per_bag : ℕ := 6

-- Define the number of candies in Roger's bags
def roger_candy_bag1 : ℕ := 11
def roger_candy_bag2 : ℕ := 3

-- Calculate the total number of candies for Sandra and Roger
def sandra_total : ℕ := sandra_bags * sandra_candy_per_bag
def roger_total : ℕ := roger_candy_bag1 + roger_candy_bag2

-- State the theorem
theorem roger_has_more_candy : roger_total = sandra_total + 2 := by
  sorry

end roger_has_more_candy_l3773_377305


namespace quilt_remaining_squares_l3773_377314

/-- Calculates the number of remaining squares to sew in a quilt -/
theorem quilt_remaining_squares (squares_per_side : ℕ) (sewn_percentage : ℚ) : 
  squares_per_side = 16 → sewn_percentage = 1/4 → 
  (2 * squares_per_side : ℚ) * (1 - sewn_percentage) = 24 := by
  sorry


end quilt_remaining_squares_l3773_377314


namespace project_completion_time_l3773_377319

theorem project_completion_time (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0) :
  let total_people := m + n
  let days_for_total := m
  let days_for_n := m * total_people / n
  (∀ (person : ℕ), person ≤ total_people → person > 0 → 
    (1 : ℚ) / (total_people * days_for_total : ℚ) = 
    (1 : ℚ) / (person * (total_people * days_for_total / person) : ℚ)) →
  days_for_n * n = m * total_people :=
sorry

end project_completion_time_l3773_377319


namespace horner_method_v3_l3773_377325

def f (x : ℝ) : ℝ := 3*x^5 - 2*x^4 + 2*x^3 - 4*x^2 - 7

def horner_v3 (a : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x - 2
  let v2 := v1 * x + 2
  v2 * x - 4

theorem horner_method_v3 :
  horner_v3 f 2 = 16 := by
  sorry

end horner_method_v3_l3773_377325


namespace three_positions_from_six_people_l3773_377387

/-- The number of ways to select 3 distinct positions from a group of 6 people. -/
def select_positions (n : ℕ) (k : ℕ) : ℕ :=
  n * (n - 1) * (n - 2)

/-- Theorem stating that selecting 3 distinct positions from 6 people results in 120 ways. -/
theorem three_positions_from_six_people :
  select_positions 6 3 = 120 := by
  sorry

end three_positions_from_six_people_l3773_377387


namespace other_factor_is_five_l3773_377301

def w : ℕ := 120

theorem other_factor_is_five :
  ∀ (product : ℕ),
  (∃ (k : ℕ), product = 936 * w * k) →
  (∃ (m : ℕ), product = 2^5 * 3^3 * m) →
  (∀ (x : ℕ), x < w → ¬(∃ (y : ℕ), 936 * x * y = product)) →
  (∃ (n : ℕ), product = 936 * w * 5 * n) :=
by sorry

end other_factor_is_five_l3773_377301


namespace parabola_translation_l3773_377346

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + k }

theorem parabola_translation :
  let original := Parabola.mk 1 0 0  -- y = x²
  let translated := translate original (-1) (-2)  -- 1 unit left, 2 units down
  translated = Parabola.mk 1 2 (-2)  -- y = (x+1)² - 2
  := by sorry

end parabola_translation_l3773_377346


namespace solution_implies_relationship_l3773_377307

theorem solution_implies_relationship (a b c : ℝ) :
  (a * (-3) + c * (-2) = 1) →
  (c * (-3) - b * (-2) = 2) →
  9 * a + 4 * b = 1 := by
sorry

end solution_implies_relationship_l3773_377307


namespace p_necessary_not_sufficient_for_q_l3773_377390

-- Define the sets corresponding to p and q
def set_p : Set ℝ := {x | (1 - x^2) / (|x| - 2) < 0}
def set_q : Set ℝ := {x | x^2 + x - 6 > 0}

-- State the theorem
theorem p_necessary_not_sufficient_for_q :
  (set_q ⊆ set_p) ∧ (set_q ≠ set_p) := by
  sorry

end p_necessary_not_sufficient_for_q_l3773_377390


namespace initial_eggs_count_l3773_377395

/-- The number of eggs initially in the box -/
def initial_eggs : ℕ := sorry

/-- The number of eggs Daniel adds to the box -/
def added_eggs : ℕ := 4

/-- The total number of eggs after Daniel adds eggs -/
def total_eggs : ℕ := 11

/-- Theorem stating that the initial number of eggs is 7 -/
theorem initial_eggs_count : initial_eggs = 7 := by
  sorry

end initial_eggs_count_l3773_377395


namespace dimes_in_tip_jar_l3773_377351

def nickel_value : ℚ := 0.05
def dime_value : ℚ := 0.10
def half_dollar_value : ℚ := 0.50

def shining_nickels : ℕ := 3
def shining_dimes : ℕ := 13
def tip_jar_half_dollars : ℕ := 9

def total_amount : ℚ := 6.65

theorem dimes_in_tip_jar :
  ∃ (tip_jar_dimes : ℕ),
    (shining_nickels * nickel_value + shining_dimes * dime_value +
     tip_jar_dimes * dime_value + tip_jar_half_dollars * half_dollar_value = total_amount) ∧
    tip_jar_dimes = 7 :=
by sorry

end dimes_in_tip_jar_l3773_377351


namespace sum_in_base8_l3773_377309

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

theorem sum_in_base8 :
  let a := 53
  let b := 27
  let sum := base10_to_base8 (base8_to_base10 a + base8_to_base10 b)
  sum = 102 := by sorry

end sum_in_base8_l3773_377309


namespace non_decreasing_sequence_count_l3773_377389

theorem non_decreasing_sequence_count :
  let max_value : ℕ := 1003
  let seq_length : ℕ := 7
  let sequence_count := Nat.choose 504 seq_length
  ∀ (b : Fin seq_length → ℕ),
    (∀ i j : Fin seq_length, i ≤ j → b i ≤ b j) →
    (∀ i : Fin seq_length, b i ≤ max_value) →
    (∀ i : Fin seq_length, Odd (b i - i.val.succ)) →
    (∃! c : ℕ, c = sequence_count) :=
by sorry

end non_decreasing_sequence_count_l3773_377389


namespace purely_imaginary_iff_a_equals_one_l3773_377372

theorem purely_imaginary_iff_a_equals_one (a : ℝ) :
  let z : ℂ := a^2 - 1 + (a + 1) * I
  (z.re = 0 ∧ z.im ≠ 0) ↔ a = 1 := by
  sorry

end purely_imaginary_iff_a_equals_one_l3773_377372


namespace apple_cost_problem_l3773_377357

theorem apple_cost_problem (l q : ℝ) : 
  (30 * l + 3 * q = 168) →
  (30 * l + 6 * q = 186) →
  (∀ k, k ≤ 30 → k * l = k * 5) →
  20 * l = 100 := by
sorry

end apple_cost_problem_l3773_377357


namespace ice_cream_pudding_cost_difference_l3773_377330

-- Define the quantities and prices
def ice_cream_quantity : ℕ := 15
def pudding_quantity : ℕ := 5
def ice_cream_price : ℕ := 5
def pudding_price : ℕ := 2

-- Define the theorem
theorem ice_cream_pudding_cost_difference :
  (ice_cream_quantity * ice_cream_price) - (pudding_quantity * pudding_price) = 65 := by
  sorry

end ice_cream_pudding_cost_difference_l3773_377330


namespace min_disks_required_l3773_377320

/-- Represents the number of files of each size --/
structure FileDistribution :=
  (large : Nat)  -- 0.9 MB files
  (medium : Nat) -- 0.8 MB files
  (small : Nat)  -- 0.5 MB files

/-- Represents the problem setup --/
def diskProblem : FileDistribution :=
  { large := 5
    medium := 15
    small := 20 }

/-- Disk capacity in MB --/
def diskCapacity : Rat := 2

/-- File sizes in MB --/
def largeFileSize : Rat := 9/10
def mediumFileSize : Rat := 4/5
def smallFileSize : Rat := 1/2

/-- The theorem stating the minimum number of disks required --/
theorem min_disks_required (fd : FileDistribution) 
  (h1 : fd.large + fd.medium + fd.small = 40)
  (h2 : fd = diskProblem) :
  ∃ (n : Nat), n = 18 ∧ 
  (∀ (m : Nat), m < n → 
    m * diskCapacity < 
    fd.large * largeFileSize + fd.medium * mediumFileSize + fd.small * smallFileSize) :=
  sorry

end min_disks_required_l3773_377320


namespace combined_weight_is_6600_l3773_377300

/-- The weight of the elephant in tons -/
def elephant_weight_tons : ℝ := 3

/-- The weight of a ton in pounds -/
def pounds_per_ton : ℝ := 2000

/-- The percentage of the elephant's weight that the donkey weighs less -/
def donkey_weight_percentage : ℝ := 0.9

/-- The combined weight of the elephant and donkey in pounds -/
def combined_weight_pounds : ℝ :=
  elephant_weight_tons * pounds_per_ton +
  elephant_weight_tons * pounds_per_ton * (1 - donkey_weight_percentage)

theorem combined_weight_is_6600 :
  combined_weight_pounds = 6600 :=
sorry

end combined_weight_is_6600_l3773_377300


namespace cubic_roots_sum_l3773_377350

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 3*a - 1 = 0) → 
  (b^3 - 3*b - 1 = 0) → 
  (c^3 - 3*c - 1 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -9 := by
sorry

end cubic_roots_sum_l3773_377350


namespace sufficient_not_necessary_l3773_377368

theorem sufficient_not_necessary (a b : ℝ) : 
  (a^2 + b^2 = 0 → a * b = 0) ∧ 
  ∃ a b : ℝ, a * b = 0 ∧ a^2 + b^2 ≠ 0 := by
  sorry

end sufficient_not_necessary_l3773_377368


namespace arccos_sum_equals_arcsin_l3773_377365

theorem arccos_sum_equals_arcsin (x : ℝ) : 
  Real.arccos x + Real.arccos (1 - x) = Real.arcsin x →
  (x = 0 ∨ x = 1 ∨ x = (1 + Real.sqrt 5) / 2) := by
sorry

end arccos_sum_equals_arcsin_l3773_377365


namespace mansion_rooms_less_than_55_l3773_377359

/-- Represents the number of rooms with a specific type of bouquet -/
structure BouquetRooms where
  roses : ℕ
  carnations : ℕ
  chrysanthemums : ℕ

/-- Represents the number of rooms with combinations of bouquets -/
structure OverlapRooms where
  carnations_chrysanthemums : ℕ
  chrysanthemums_roses : ℕ
  carnations_roses : ℕ

theorem mansion_rooms_less_than_55 (b : BouquetRooms) (o : OverlapRooms) 
    (h1 : b.roses = 30)
    (h2 : b.carnations = 20)
    (h3 : b.chrysanthemums = 10)
    (h4 : o.carnations_chrysanthemums = 2)
    (h5 : o.chrysanthemums_roses = 3)
    (h6 : o.carnations_roses = 4) :
    b.roses + b.carnations + b.chrysanthemums - 
    o.carnations_chrysanthemums - o.chrysanthemums_roses - o.carnations_roses < 55 := by
  sorry


end mansion_rooms_less_than_55_l3773_377359


namespace triangle_angles_l3773_377332

open Real

theorem triangle_angles (a b c : ℝ) (A B C : ℝ) : 
  let m : ℝ × ℝ := (sqrt 3, -1)
  let n : ℝ × ℝ := (cos A, sin A)
  (m.1 * n.1 + m.2 * n.2 = 0) →  -- m ⊥ n
  (a * cos B + b * cos A = c * sin C) →  -- given condition
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- positive side lengths
  (A > 0 ∧ B > 0 ∧ C > 0) →  -- positive angles
  (A + B + C = π) →  -- sum of angles in a triangle
  (A = π/3 ∧ B = π/6) :=
by sorry

end triangle_angles_l3773_377332


namespace x_plus_y_equals_48_l3773_377398

-- Define the arithmetic sequence
def arithmetic_sequence : List ℝ := [3, 9, 15, 33]

-- Define x and y as the last two terms before 33
def x : ℝ := arithmetic_sequence[arithmetic_sequence.length - 3]
def y : ℝ := arithmetic_sequence[arithmetic_sequence.length - 2]

-- Theorem to prove
theorem x_plus_y_equals_48 : x + y = 48 := by
  sorry

end x_plus_y_equals_48_l3773_377398


namespace soda_transaction_result_l3773_377336

def soda_transaction (initial_cans : ℕ) : ℕ × ℕ :=
  let jeff_takes := 6
  let jeff_returns := jeff_takes / 2
  let after_jeff := initial_cans - jeff_takes + jeff_returns
  let tim_buys := after_jeff / 3
  let store_bonus := tim_buys / 4
  let after_store := after_jeff + tim_buys + store_bonus
  let sarah_takes := after_store / 5
  let end_of_day := after_store - sarah_takes
  let sarah_returns := sarah_takes * 2
  let next_day := end_of_day + sarah_returns
  (end_of_day, next_day)

theorem soda_transaction_result :
  soda_transaction 22 = (21, 31) := by sorry

end soda_transaction_result_l3773_377336


namespace power_inequality_l3773_377342

theorem power_inequality (n : ℕ) (hn : n > 3) : n^(n+1) > (n+1)^n := by
  sorry

end power_inequality_l3773_377342


namespace component_service_life_probability_l3773_377334

theorem component_service_life_probability 
  (P_exceed_1_year : ℝ) 
  (P_exceed_2_years : ℝ) 
  (h1 : P_exceed_1_year = 0.6) 
  (h2 : P_exceed_2_years = 0.3) :
  (P_exceed_2_years / P_exceed_1_year) = 0.5 := by
  sorry

end component_service_life_probability_l3773_377334


namespace range_sum_of_bounds_l3773_377360

open Set Real

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

theorem range_sum_of_bounds (a b : ℝ) :
  (∀ y ∈ Set.range h, a < y ∧ y ≤ b) ∧
  (∀ ε > 0, ∃ y ∈ Set.range h, y < a + ε) ∧
  (b ∈ Set.range h) →
  a + b = 1 := by sorry

end range_sum_of_bounds_l3773_377360


namespace two_stage_discount_l3773_377353

/-- Calculate the actual discount and difference from claimed discount in a two-stage discount scenario -/
theorem two_stage_discount (initial_discount additional_discount claimed_discount : ℝ) :
  initial_discount = 0.4 →
  additional_discount = 0.25 →
  claimed_discount = 0.6 →
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_additional := remaining_after_initial * (1 - additional_discount)
  let actual_discount := 1 - remaining_after_additional
  actual_discount = 0.55 ∧ claimed_discount - actual_discount = 0.05 := by
  sorry

end two_stage_discount_l3773_377353


namespace lcm_150_294_l3773_377393

theorem lcm_150_294 : Nat.lcm 150 294 = 7350 := by
  sorry

end lcm_150_294_l3773_377393


namespace cyclist_speed_l3773_377356

theorem cyclist_speed (speed : ℝ) : 
  (speed ≥ 0) →                           -- Non-negative speed
  (5 * speed + 5 * speed = 50) →          -- Total distance after 5 hours
  (speed = 5) :=                          -- Speed of each cyclist
by sorry

end cyclist_speed_l3773_377356


namespace cos_difference_l3773_377329

theorem cos_difference (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
  sorry

end cos_difference_l3773_377329


namespace arrangement_count_l3773_377318

/-- Represents the number of people in the arrangement. -/
def total_people : ℕ := 6

/-- Represents the number of people who have a specific position requirement (Jia, Bing, and Yi). -/
def specific_people : ℕ := 3

/-- Calculates the number of arrangements where one person stands between two others in a line of n people. -/
def arrangements (n : ℕ) : ℕ :=
  (Nat.factorial (n - specific_people + 1)) * 2

/-- Theorem stating that the number of arrangements for 6 people with the given condition is 48. -/
theorem arrangement_count :
  arrangements total_people = 48 := by
  sorry

end arrangement_count_l3773_377318


namespace right_triangle_hypotenuse_l3773_377394

theorem right_triangle_hypotenuse : ∀ x₁ x₂ : ℝ,
  x₁^2 - 36*x₁ + 70 = 0 →
  x₂^2 - 36*x₂ + 70 = 0 →
  x₁ ≠ x₂ →
  Real.sqrt (x₁^2 + x₂^2) = 34 := by
sorry

end right_triangle_hypotenuse_l3773_377394


namespace wheel_probability_l3773_377308

theorem wheel_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 →
  p_B = 1/3 →
  p_C = 1/6 →
  p_A + p_B + p_C + p_D = 1 →
  p_D = 1/4 := by
sorry

end wheel_probability_l3773_377308


namespace composite_product_quotient_l3773_377338

def first_five_composites : List ℕ := [21, 22, 24, 25, 26]
def next_five_composites : List ℕ := [27, 28, 30, 32, 33]

def product_list (l : List ℕ) : ℕ := l.foldl (·*·) 1

theorem composite_product_quotient :
  (product_list first_five_composites : ℚ) / (product_list next_five_composites) = 1 / 1964 := by
  sorry

end composite_product_quotient_l3773_377338


namespace fraction_addition_l3773_377379

theorem fraction_addition (d : ℝ) : (5 + 2 * d) / 8 + 3 = (29 + 2 * d) / 8 := by
  sorry

end fraction_addition_l3773_377379


namespace rectangular_to_polar_conversion_l3773_377343

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 3 * Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = Real.sqrt 91 ∧ θ = Real.arctan (3 * Real.sqrt 3 / 8) :=
by sorry

end rectangular_to_polar_conversion_l3773_377343


namespace ratio_of_a_to_b_l3773_377370

theorem ratio_of_a_to_b (A B : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) 
  (h : (2 / 3) * A = (3 / 4) * B) : A / B = 9 / 8 := by
  sorry

end ratio_of_a_to_b_l3773_377370


namespace nine_chapters_problem_l3773_377321

/-- Represents the worth of animals in taels of gold -/
structure AnimalWorth where
  cow : ℝ
  sheep : ℝ

/-- Represents the total worth of a group of animals -/
def groupWorth (w : AnimalWorth) (cows sheep : ℕ) : ℝ :=
  cows * w.cow + sheep * w.sheep

/-- The problem statement from "The Nine Chapters on the Mathematical Art" -/
theorem nine_chapters_problem (w : AnimalWorth) : 
  (groupWorth w 5 2 = 10 ∧ groupWorth w 2 5 = 8) ↔ 
  (5 * w.cow + 2 * w.sheep = 10 ∧ 2 * w.cow + 5 * w.sheep = 8) := by
sorry

end nine_chapters_problem_l3773_377321


namespace dave_tray_capacity_l3773_377373

theorem dave_tray_capacity (trays_table1 trays_table2 num_trips : ℕ) 
  (h1 : trays_table1 = 17)
  (h2 : trays_table2 = 55)
  (h3 : num_trips = 8) :
  (trays_table1 + trays_table2) / num_trips = 9 := by
  sorry

#check dave_tray_capacity

end dave_tray_capacity_l3773_377373


namespace swimming_speed_in_still_water_l3773_377354

/-- Proves that a person's swimming speed in still water is 4 km/h given the conditions -/
theorem swimming_speed_in_still_water 
  (water_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : water_speed = 2)
  (h2 : distance = 8)
  (h3 : time = 4)
  (h4 : (swimming_speed - water_speed) * time = distance) :
  swimming_speed = 4 :=
by
  sorry

#check swimming_speed_in_still_water

end swimming_speed_in_still_water_l3773_377354


namespace number_of_grey_stones_l3773_377377

/-- Given a collection of stones with specific properties, prove the number of grey stones. -/
theorem number_of_grey_stones 
  (total_stones : ℕ) 
  (white_stones : ℕ) 
  (green_stones : ℕ) 
  (h1 : total_stones = 100)
  (h2 : white_stones = 60)
  (h3 : green_stones = 60)
  (h4 : white_stones > total_stones - white_stones)
  (h5 : (white_stones : ℚ) / (total_stones - white_stones) = (grey_stones : ℚ) / green_stones) :
  grey_stones = 90 :=
by
  sorry

#check number_of_grey_stones

end number_of_grey_stones_l3773_377377


namespace solve_for_k_l3773_377335

theorem solve_for_k (k : ℚ) : 
  (∃ x : ℚ, 3 * x + (2 * k - 1) = x - 6 * (3 * k + 2)) ∧ 
  (3 * 1 + (2 * k - 1) = 1 - 6 * (3 * k + 2)) → 
  k = -13/20 := by
sorry

end solve_for_k_l3773_377335
