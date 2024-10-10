import Mathlib

namespace carlton_zoo_total_l869_86928

/-- Represents the number of animals in each zoo -/
structure ZooAnimals :=
  (rhinoceroses : ℕ)
  (elephants : ℕ)
  (lions : ℕ)
  (monkeys : ℕ)
  (penguins : ℕ)

/-- Defines the relationship between Bell Zoo and Carlton Zoo -/
def zoo_relationship (bell : ZooAnimals) (carlton : ZooAnimals) : Prop :=
  bell.rhinoceroses = carlton.lions ∧
  bell.elephants = carlton.lions + 3 ∧
  bell.elephants = carlton.rhinoceroses ∧
  carlton.elephants = carlton.rhinoceroses + 2 ∧
  carlton.monkeys = 2 * (carlton.rhinoceroses + carlton.elephants + carlton.lions) ∧
  carlton.penguins = carlton.monkeys + 2 ∧
  bell.monkeys = 2 * carlton.penguins / 3 ∧
  bell.penguins = bell.monkeys + 2 ∧
  bell.lions * 2 = bell.penguins ∧
  bell.rhinoceroses + bell.elephants + bell.lions + bell.monkeys + bell.penguins = 48

theorem carlton_zoo_total (bell : ZooAnimals) (carlton : ZooAnimals) 
  (h : zoo_relationship bell carlton) : 
  carlton.rhinoceroses + carlton.elephants + carlton.lions + carlton.monkeys + carlton.penguins = 57 :=
by sorry


end carlton_zoo_total_l869_86928


namespace wizard_elixir_combinations_l869_86908

/-- Represents the number of roots available to the wizard. -/
def num_roots : ℕ := 4

/-- Represents the number of minerals available to the wizard. -/
def num_minerals : ℕ := 5

/-- Represents the number of incompatible pairs of roots and minerals. -/
def num_incompatible_pairs : ℕ := 3

/-- Theorem stating the number of possible combinations for the wizard's elixir. -/
theorem wizard_elixir_combinations : 
  num_roots * num_minerals - num_incompatible_pairs = 17 := by
  sorry

end wizard_elixir_combinations_l869_86908


namespace disjunction_implies_conjunction_false_l869_86915

theorem disjunction_implies_conjunction_false : 
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by
  sorry

end disjunction_implies_conjunction_false_l869_86915


namespace fruit_filled_mooncake_probability_l869_86989

def num_fruits : ℕ := 5
def num_meats : ℕ := 4

def combinations (n : ℕ) : ℕ := n * (n - 1) / 2

theorem fruit_filled_mooncake_probability :
  let total_combinations := combinations num_fruits + combinations num_meats
  let fruit_combinations := combinations num_fruits
  (fruit_combinations : ℚ) / total_combinations = 5 / 8 := by sorry

end fruit_filled_mooncake_probability_l869_86989


namespace rotate_triangle_forms_cone_l869_86914

/-- A right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ
  hypotenuse : ℝ
  right_angle : base^2 + height^2 = hypotenuse^2

/-- A cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- The solid formed by rotating a right-angled triangle around one of its right-angle sides -/
def rotateTriangle (t : RightTriangle) : Cone :=
  { radius := t.base, height := t.height }

/-- Theorem: Rotating a right-angled triangle around one of its right-angle sides forms a cone -/
theorem rotate_triangle_forms_cone (t : RightTriangle) :
  ∃ (c : Cone), rotateTriangle t = c :=
sorry

end rotate_triangle_forms_cone_l869_86914


namespace marias_trip_distance_l869_86927

/-- Proves that the total distance of Maria's trip is 450 miles -/
theorem marias_trip_distance (D : ℝ) 
  (first_stop : D - D/3 = 2/3 * D)
  (second_stop : 2/3 * D - 1/4 * (2/3 * D) = 1/2 * D)
  (third_stop : 1/2 * D - 1/5 * (1/2 * D) = 2/5 * D)
  (final_distance : 2/5 * D = 180) :
  D = 450 := by sorry

end marias_trip_distance_l869_86927


namespace pancake_cost_l869_86976

/-- The cost of a stack of pancakes satisfies the given conditions -/
theorem pancake_cost (pancake_stacks : ℕ) (bacon_slices : ℕ) (bacon_price : ℚ) (total_raised : ℚ) :
  pancake_stacks = 60 →
  bacon_slices = 90 →
  bacon_price = 2 →
  total_raised = 420 →
  ∃ (P : ℚ), P * pancake_stacks + bacon_price * bacon_slices = total_raised ∧ P = 4 :=
by sorry

end pancake_cost_l869_86976


namespace periodic_points_measure_l869_86982

open MeasureTheory

theorem periodic_points_measure (f : ℝ → ℝ) (hf : Continuous f) (hf0 : f 0 = 0) (hf1 : f 1 = 0) :
  let A := {h ∈ Set.Icc 0 1 | ∃ x ∈ Set.Icc 0 1, f (x + h) = f x}
  Measurable A ∧ volume A ≥ 1/2 := by
sorry

end periodic_points_measure_l869_86982


namespace tenth_term_of_specific_arithmetic_sequence_l869_86993

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- State the theorem
theorem tenth_term_of_specific_arithmetic_sequence : 
  ∃ (a d : ℝ), 
    arithmetic_sequence a d 3 = 10 ∧ 
    arithmetic_sequence a d 6 = 16 ∧ 
    arithmetic_sequence a d 10 = 24 := by
  sorry

end tenth_term_of_specific_arithmetic_sequence_l869_86993


namespace triangles_in_ten_point_config_l869_86921

/-- Represents a configuration of points on a circle with chords --/
structure CircleConfiguration where
  numPoints : ℕ
  numChords : ℕ
  numIntersections : ℕ

/-- Calculates the number of triangles formed by chord intersections --/
def numTriangles (config : CircleConfiguration) : ℕ :=
  sorry

/-- The specific configuration for our problem --/
def tenPointConfig : CircleConfiguration :=
  { numPoints := 10
  , numChords := 45
  , numIntersections := 210 }

/-- Theorem stating that the number of triangles in the given configuration is 120 --/
theorem triangles_in_ten_point_config :
  numTriangles tenPointConfig = 120 :=
sorry

end triangles_in_ten_point_config_l869_86921


namespace set_equivalence_l869_86960

theorem set_equivalence : 
  {x : ℕ+ | x - 3 < 2} = {1, 2, 3, 4} := by sorry

end set_equivalence_l869_86960


namespace sqrt_meaningful_range_l869_86955

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - 3 * x) ↔ x ≤ 1 / 3 := by
  sorry

end sqrt_meaningful_range_l869_86955


namespace total_money_is_305_l869_86946

/-- The value of a gold coin in dollars -/
def gold_coin_value : ℕ := 50

/-- The value of a silver coin in dollars -/
def silver_coin_value : ℕ := 25

/-- The number of gold coins -/
def num_gold_coins : ℕ := 3

/-- The number of silver coins -/
def num_silver_coins : ℕ := 5

/-- The amount of cash in dollars -/
def cash : ℕ := 30

/-- The total amount of money in dollars -/
def total_money : ℕ := gold_coin_value * num_gold_coins + silver_coin_value * num_silver_coins + cash

theorem total_money_is_305 : total_money = 305 := by
  sorry

end total_money_is_305_l869_86946


namespace apples_given_away_l869_86934

/-- Given that Joan picked a certain number of apples and now has fewer,
    prove that the number of apples she gave away is the difference between
    the initial and current number of apples. -/
theorem apples_given_away (initial current : ℕ) (h : current ≤ initial) :
  initial - current = initial - current := by sorry

end apples_given_away_l869_86934


namespace range_of_m_l869_86985

def p (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ a = 16 - m ∧ b = m - 4 ∧ a > 0 ∧ b > 0

def q (m : ℝ) : Prop :=
  (m - 10)^2 + 3^2 < 13

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  (4 < m ∧ m ≤ 8) ∨ (10 ≤ m ∧ m < 12) :=
sorry

end range_of_m_l869_86985


namespace carmen_sculpture_height_l869_86912

/-- Represents a measurement in feet and inches -/
structure FeetInches where
  feet : ℕ
  inches : ℕ
  h_valid : inches < 12

/-- Converts inches to a FeetInches measurement -/
def inchesToFeetInches (totalInches : ℕ) : FeetInches :=
  { feet := totalInches / 12,
    inches := totalInches % 12,
    h_valid := by sorry }

/-- Adds two FeetInches measurements -/
def addFeetInches (a b : FeetInches) : FeetInches :=
  inchesToFeetInches (a.feet * 12 + a.inches + b.feet * 12 + b.inches)

theorem carmen_sculpture_height :
  let rectangular_prism_height : ℕ := 8
  let cylinder_height : ℕ := 15
  let pyramid_height : ℕ := 10
  let base_height : ℕ := 10
  let sculpture_height := rectangular_prism_height + cylinder_height + pyramid_height
  let sculpture_feet_inches := inchesToFeetInches sculpture_height
  let base_feet_inches := inchesToFeetInches base_height
  let combined_height := addFeetInches sculpture_feet_inches base_feet_inches
  combined_height = { feet := 3, inches := 7, h_valid := by sorry } := by sorry

end carmen_sculpture_height_l869_86912


namespace explicit_formula_l869_86969

noncomputable section

variable (f : ℝ → ℝ)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = (deriv f 1) * Real.exp (x - 1) - (f 0) * x + (1/2) * x^2

theorem explicit_formula (h : satisfies_condition f) :
  ∀ x, f x = Real.exp x - x + (1/2) * x^2 := by
  sorry

end

end explicit_formula_l869_86969


namespace ellipse_triangle_perimeter_l869_86990

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ -- Semi-major axis
  b : ℝ -- Semi-minor axis
  f1 : Point -- Focus 1
  f2 : Point -- Focus 2

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop := sorry

theorem ellipse_triangle_perimeter 
  (e : Ellipse) 
  (A B : Point) 
  (h1 : e.a = 5)
  (h2 : isOnEllipse e A)
  (h3 : isOnEllipse e B) :
  distance A B + distance A e.f2 + distance B e.f2 = 4 * e.a := by sorry

end ellipse_triangle_perimeter_l869_86990


namespace convex_polygon_symmetry_l869_86930

-- Define a convex polygon
structure ConvexPolygon where
  -- (Add necessary fields for a convex polygon)

-- Define a point inside the polygon
structure InnerPoint (P : ConvexPolygon) where
  point : ℝ × ℝ
  isInside : Bool -- Predicate to check if the point is inside the polygon

-- Define a line passing through a point
structure Line (P : ℝ × ℝ) where
  slope : ℝ
  -- The line is represented by y = slope * (x - P.1) + P.2

-- Function to check if a line divides the polygon into equal areas
def dividesEqualAreas (P : ConvexPolygon) (O : InnerPoint P) (l : Line O.point) : Prop :=
  -- (Add logic to check if the line divides the polygon into equal areas)
  sorry

-- Function to check if a point is the center of symmetry
def isCenterOfSymmetry (P : ConvexPolygon) (O : InnerPoint P) : Prop :=
  -- (Add logic to check if O is the center of symmetry)
  sorry

-- The main theorem
theorem convex_polygon_symmetry (P : ConvexPolygon) (O : InnerPoint P) :
  (∀ l : Line O.point, dividesEqualAreas P O l) → isCenterOfSymmetry P O :=
by
  sorry

end convex_polygon_symmetry_l869_86930


namespace rectangle_to_square_l869_86961

theorem rectangle_to_square (area : ℝ) (reduction : ℝ) (side : ℝ) : 
  area = 600 →
  reduction = 10 →
  (side + reduction) * side = area →
  side * side = area →
  side = 20 :=
by sorry

end rectangle_to_square_l869_86961


namespace proportion_fourth_term_l869_86977

theorem proportion_fourth_term (x y : ℝ) : 
  (0.75 : ℝ) / 1.2 = 5 / y → y = 8 := by
  sorry

end proportion_fourth_term_l869_86977


namespace non_intersecting_to_concentric_l869_86949

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- An inversion transformation --/
structure Inversion where
  center : ℝ × ℝ
  power : ℝ
  power_pos : power > 0

/-- Two circles are non-intersecting --/
def non_intersecting (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

/-- Two circles are concentric --/
def concentric (c1 c2 : Circle) : Prop :=
  c1.center = c2.center

/-- The image of a circle under inversion --/
def inversion_image (i : Inversion) (c : Circle) : Circle :=
  sorry

/-- The main theorem --/
theorem non_intersecting_to_concentric :
  ∀ (S1 S2 : Circle), non_intersecting S1 S2 →
  ∃ (i : Inversion), concentric (inversion_image i S1) (inversion_image i S2) :=
sorry

end non_intersecting_to_concentric_l869_86949


namespace smallest_angle_measure_l869_86929

theorem smallest_angle_measure (ABC ABD : ℝ) (h1 : ABC = 24) (h2 : ABD = 20) :
  ∃ CBD : ℝ, CBD = ABC - ABD ∧ CBD = 4 ∧ ∀ x : ℝ, x ≥ 0 → x ≥ CBD := by
  sorry

end smallest_angle_measure_l869_86929


namespace intersection_exists_l869_86991

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 / 4 = 1

/-- The line equation -/
def line (k x y : ℝ) : Prop := y = k * x

/-- The theorem statement -/
theorem intersection_exists : ∃ k : ℝ, 0 < k ∧ k < 2 ∧ 
  ∃ x y : ℝ, hyperbola x y ∧ line k x y :=
sorry

end intersection_exists_l869_86991


namespace problem_solution_l869_86926

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x else Real.log x / x

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem problem_solution (a : ℕ → ℝ) : 
  is_geometric_sequence a →
  a 3 * a 4 * a 5 = 1 →
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1 →
  a 1 = Real.exp 2 := by
  sorry

end problem_solution_l869_86926


namespace rogers_money_l869_86972

theorem rogers_money (x : ℤ) : 
  x - 20 + 46 = 71 → x = 45 := by
sorry

end rogers_money_l869_86972


namespace calculate_unknown_interest_rate_l869_86967

/-- Proves that for a given principal, time period, and interest rate difference, 
    the unknown rate can be calculated. -/
theorem calculate_unknown_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (known_rate : ℝ) 
  (interest_difference : ℝ) 
  (unknown_rate : ℝ)
  (h1 : principal = 7000)
  (h2 : time = 2)
  (h3 : known_rate = 18)
  (h4 : interest_difference = 840)
  (h5 : principal * (known_rate / 100) * time - principal * (unknown_rate / 100) * time = interest_difference) :
  unknown_rate = 12 := by
  sorry

end calculate_unknown_interest_rate_l869_86967


namespace cut_to_square_iff_perfect_square_l869_86933

/-- Represents a figure on a grid -/
structure GridFigure where
  area : ℕ

/-- Represents a cut of the figure -/
inductive Cut
  | Line : Cut

/-- Represents the result of cutting the figure -/
structure CutResult where
  parts : Fin 3 → GridFigure

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- Can form a square from the cut parts -/
def can_form_square (cr : CutResult) : Prop :=
  ∃ side : ℕ, (cr.parts 0).area + (cr.parts 1).area + (cr.parts 2).area = side * side

/-- The main theorem: a figure can be cut into three parts to form a square
    if and only if its area is a perfect square -/
theorem cut_to_square_iff_perfect_square (f : GridFigure) :
  (∃ cuts : List Cut, ∃ cr : CutResult, can_form_square cr) ↔ is_perfect_square f.area :=
sorry

end cut_to_square_iff_perfect_square_l869_86933


namespace rectangle_area_increase_l869_86979

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let new_length := 1.3 * l
  let new_width := 1.2 * w
  let original_area := l * w
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.56 := by
sorry

end rectangle_area_increase_l869_86979


namespace second_player_strategy_exists_first_player_strategy_exists_l869_86953

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a 4-digit number -/
def FourDigitNumber := Fin 10000

/-- The game state, representing the current partially filled subtraction problem -/
structure GameState where
  minuend : FourDigitNumber
  subtrahend : FourDigitNumber

/-- A player's move, either calling out a digit or placing a digit -/
inductive Move
  | CallDigit : Digit → Move
  | PlaceDigit : Digit → Nat → Move

/-- The result of the game -/
def gameResult (finalState : GameState) : Int :=
  (finalState.minuend.val : Int) - (finalState.subtrahend.val : Int)

/-- A strategy for a player -/
def Strategy := GameState → Move

/-- Theorem: There exists a strategy for the second player to keep the difference ≤ 4000 -/
theorem second_player_strategy_exists : 
  ∃ (s : Strategy), ∀ (g : GameState), gameResult g ≤ 4000 := by sorry

/-- Theorem: There exists a strategy for the first player to keep the difference ≥ 4000 -/
theorem first_player_strategy_exists :
  ∃ (s : Strategy), ∀ (g : GameState), gameResult g ≥ 4000 := by sorry

end second_player_strategy_exists_first_player_strategy_exists_l869_86953


namespace arithmetic_sequence_properties_l869_86954

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  h1 : a 2 + a 6 = 6
  h2 : (5 * (a 1 + a 5)) / 2 = 35 / 3

/-- The sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n * (seq.a 1 + seq.a n)) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = (2 / 3) * n + 1 / 3) ∧
  (∀ n : ℕ, S seq n ≥ 1) ∧
  (S seq 1 = 1) := by
  sorry

end arithmetic_sequence_properties_l869_86954


namespace smallest_fraction_between_l869_86981

theorem smallest_fraction_between (a b c d : ℕ) (h1 : a < b) (h2 : c < d) :
  ∃ (x y : ℕ), 
    (x : ℚ) / y > (a : ℚ) / b ∧ 
    (x : ℚ) / y < (c : ℚ) / d ∧ 
    (∀ (p q : ℕ), (p : ℚ) / q > (a : ℚ) / b ∧ (p : ℚ) / q < (c : ℚ) / d → y ≤ q) ∧
    x = 2 ∧ y = 7 :=
by sorry

end smallest_fraction_between_l869_86981


namespace obtuse_triangle_count_and_largest_perimeter_l869_86905

/-- Represents a triangle with side lengths in arithmetic progression --/
structure ArithmeticTriangle where
  a : ℕ  -- middle length
  d : ℕ  -- common difference

/-- Checks if the triangle is obtuse --/
def ArithmeticTriangle.isObtuse (t : ArithmeticTriangle) : Prop :=
  (t.a - t.d)^2 + t.a^2 < (t.a + t.d)^2

/-- Checks if the triangle satisfies the given conditions --/
def ArithmeticTriangle.isValid (t : ArithmeticTriangle) : Prop :=
  t.d > 0 ∧ t.a > t.d ∧ t.a + t.d ≤ 50

/-- Counts the number of valid obtuse triangles --/
def countValidObtuseTriangles : ℕ := sorry

/-- Finds the triangle with the largest perimeter --/
def largestPerimeterTriangle : ArithmeticTriangle := sorry

theorem obtuse_triangle_count_and_largest_perimeter :
  countValidObtuseTriangles = 157 ∧
  let t := largestPerimeterTriangle
  t.a - t.d = 29 ∧ t.a = 39 ∧ t.a + t.d = 50 := by sorry

end obtuse_triangle_count_and_largest_perimeter_l869_86905


namespace common_factor_l869_86920

def expression (m n : ℕ) : ℤ := 4 * m^3 * n - 9 * m * n^3

theorem common_factor (m n : ℕ) : 
  ∃ (k : ℤ), expression m n = m * n * k ∧ 
  ¬∃ (l : ℤ), l ≠ 1 ∧ l ≠ -1 ∧ 
  ∃ (p : ℤ), expression m n = (m * n * l) * p :=
sorry

end common_factor_l869_86920


namespace bob_pie_count_l869_86986

/-- The radius of Tom's circular pies in cm -/
def tom_radius : ℝ := 8

/-- The number of pies Tom can make in one batch -/
def tom_batch_size : ℕ := 6

/-- The length of one leg of Bob's right-angled triangular pies in cm -/
def bob_leg1 : ℝ := 6

/-- The length of the other leg of Bob's right-angled triangular pies in cm -/
def bob_leg2 : ℝ := 8

/-- The number of pies Bob can make with the same amount of dough as Tom -/
def bob_batch_size : ℕ := 50

theorem bob_pie_count :
  bob_batch_size = ⌊(tom_radius^2 * Real.pi * tom_batch_size) / (bob_leg1 * bob_leg2 / 2)⌋ := by
  sorry

end bob_pie_count_l869_86986


namespace katie_total_marbles_l869_86962

/-- The number of marbles Katie has -/
def total_marbles (pink orange purple : ℕ) : ℕ := pink + orange + purple

/-- The properties of Katie's marble collection -/
def katie_marbles (pink orange purple : ℕ) : Prop :=
  pink = 13 ∧ orange = pink - 9 ∧ purple = 4 * orange

theorem katie_total_marbles :
  ∀ pink orange purple : ℕ,
    katie_marbles pink orange purple →
    total_marbles pink orange purple = 33 :=
by
  sorry

end katie_total_marbles_l869_86962


namespace zero_only_number_unchanged_by_integer_multiplication_l869_86956

theorem zero_only_number_unchanged_by_integer_multiplication :
  ∀ n : ℤ, (∀ m : ℤ, n * m = n) → n = 0 := by
  sorry

end zero_only_number_unchanged_by_integer_multiplication_l869_86956


namespace max_silver_tokens_l869_86942

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules --/
inductive ExchangeRule
  | RedToSilver : ExchangeRule  -- 3 red → 2 silver + 1 blue
  | BlueToSilver : ExchangeRule -- 4 blue → 1 silver + 2 red

/-- Applies an exchange rule to a token count --/
def applyExchange (tc : TokenCount) (rule : ExchangeRule) : Option TokenCount :=
  match rule with
  | ExchangeRule.RedToSilver =>
      if tc.red ≥ 3 then
        some ⟨tc.red - 3, tc.blue + 1, tc.silver + 2⟩
      else
        none
  | ExchangeRule.BlueToSilver =>
      if tc.blue ≥ 4 then
        some ⟨tc.red + 2, tc.blue - 4, tc.silver + 1⟩
      else
        none

/-- Determines if any exchange is possible --/
def canExchange (tc : TokenCount) : Bool :=
  tc.red ≥ 3 ∨ tc.blue ≥ 4

/-- The main theorem to prove --/
theorem max_silver_tokens :
  ∃ (final : TokenCount),
    final.silver = 113 ∧
    ¬(canExchange final) ∧
    (∀ (tc : TokenCount),
      tc.red = 100 ∧ tc.blue = 50 ∧ tc.silver = 0 →
      (∃ (exchanges : List ExchangeRule),
        (exchanges.foldl (λ acc rule => (applyExchange acc rule).getD acc) tc) = final)) :=
  sorry


end max_silver_tokens_l869_86942


namespace min_value_reciprocal_sum_l869_86938

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / a + 4 / b ≥ 9 / 2 := by sorry

end min_value_reciprocal_sum_l869_86938


namespace initial_distance_is_one_mile_l869_86910

/-- Two boats moving towards each other -/
structure BoatSystem where
  boat1_speed : ℝ
  boat2_speed : ℝ
  distance_before_collision : ℝ
  time_before_collision : ℝ

/-- The initial distance between the boats -/
def initial_distance (bs : BoatSystem) : ℝ :=
  bs.distance_before_collision + (bs.boat1_speed + bs.boat2_speed) * bs.time_before_collision

/-- Theorem stating the initial distance between the boats -/
theorem initial_distance_is_one_mile :
  ∀ (bs : BoatSystem),
    bs.boat1_speed = 5 ∧
    bs.boat2_speed = 25 ∧
    bs.distance_before_collision = 0.5 ∧
    bs.time_before_collision = 1 / 60 →
    initial_distance bs = 1 := by
  sorry

end initial_distance_is_one_mile_l869_86910


namespace intersection_M_N_l869_86907

-- Define the sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {y | y > -1}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo (-1) 1 := by sorry

end intersection_M_N_l869_86907


namespace smallest_p_is_three_l869_86983

theorem smallest_p_is_three (p q s r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime s → Nat.Prime r →
  p + q + s = r →
  2 < p → p < q → q < s →
  ∀ p' : ℕ, (Nat.Prime p' ∧ 
             (∃ q' s' r' : ℕ, Nat.Prime q' ∧ Nat.Prime s' ∧ Nat.Prime r' ∧
                              p' + q' + s' = r' ∧
                              2 < p' ∧ p' < q' ∧ q' < s')) →
            p' ≥ 3 :=
by sorry

end smallest_p_is_three_l869_86983


namespace square_area_increase_l869_86980

theorem square_area_increase (a : ℝ) (ha : a > 0) : 
  let side_b := 2 * a
  let side_c := side_b * 1.4
  let area_a := a ^ 2
  let area_b := side_b ^ 2
  let area_c := side_c ^ 2
  (area_c - (area_a + area_b)) / (area_a + area_b) = 0.568 := by
  sorry

end square_area_increase_l869_86980


namespace largest_angle_in_triangle_l869_86924

/-- Given a triangle PQR with angles 3x, x, and 6x, prove that the largest angle is 108° -/
theorem largest_angle_in_triangle (x : ℝ) : 
  x > 0 ∧ 3*x + x + 6*x = 180 → 
  max (3*x) (max x (6*x)) = 108 := by
sorry

end largest_angle_in_triangle_l869_86924


namespace smallest_two_digit_prime_with_composite_reverse_l869_86922

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def has_tens_digit_2 (n : ℕ) : Prop := n ≥ 20 ∧ n < 30

theorem smallest_two_digit_prime_with_composite_reverse :
  ∃ n : ℕ, is_prime n ∧ 
           is_composite (reverse_digits n) ∧ 
           has_tens_digit_2 n ∧
           (∀ m : ℕ, m < n → ¬(is_prime m ∧ is_composite (reverse_digits m) ∧ has_tens_digit_2 m)) ∧
           n = 23 :=
by sorry

end smallest_two_digit_prime_with_composite_reverse_l869_86922


namespace gift_wrap_sales_l869_86902

theorem gift_wrap_sales (solid_price print_price total_rolls total_amount : ℝ) 
  (h1 : solid_price = 4)
  (h2 : print_price = 6)
  (h3 : total_rolls = 480)
  (h4 : total_amount = 2340)
  : ∃ (solid_rolls print_rolls : ℝ),
    solid_rolls + print_rolls = total_rolls ∧
    solid_price * solid_rolls + print_price * print_rolls = total_amount ∧
    print_rolls = 210 := by
  sorry

end gift_wrap_sales_l869_86902


namespace hyperbola_asymptotes_equation_l869_86978

/-- Given a parabola and a hyperbola in the Cartesian coordinate plane, 
    with a point of intersection and a condition on the focus of the parabola, 
    prove that the asymptotes of the hyperbola have a specific equation. -/
theorem hyperbola_asymptotes_equation (b : ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ) :
  b > 0 →
  A.2^2 = 4 * A.1 →
  A.1^2 / 4 - A.2^2 / b^2 = 1 →
  F = (1, 0) →
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = 25 →
  ∃ (k : ℝ), k = 2 * Real.sqrt 3 / 3 ∧
    (∀ (x y : ℝ), (x^2 / 4 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x)) :=
by sorry

end hyperbola_asymptotes_equation_l869_86978


namespace math_test_blank_questions_l869_86935

theorem math_test_blank_questions 
  (total_questions : ℕ) 
  (word_problems : ℕ) 
  (addition_subtraction_problems : ℕ)
  (questions_answered : ℕ) 
  (h1 : total_questions = 45)
  (h2 : word_problems = 17)
  (h3 : addition_subtraction_problems = 28)
  (h4 : questions_answered = 38)
  (h5 : word_problems + addition_subtraction_problems = total_questions) :
  total_questions - questions_answered = 7 := by
  sorry

end math_test_blank_questions_l869_86935


namespace square_perimeter_l869_86984

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (3 * s = 40) → (4 * s = 160 / 3) := by
  sorry

end square_perimeter_l869_86984


namespace marys_potatoes_l869_86944

/-- 
Given that Mary has some initial number of potatoes, rabbits ate 3 potatoes, 
and Mary now has 5 potatoes left, prove that Mary initially had 8 potatoes.
-/
theorem marys_potatoes (initial : ℕ) (eaten : ℕ) (remaining : ℕ) : 
  eaten = 3 → remaining = 5 → initial = eaten + remaining → initial = 8 := by
sorry

end marys_potatoes_l869_86944


namespace preimage_of_3_1_l869_86992

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x + 2*y, x - 2*y)

theorem preimage_of_3_1 (x y : ℝ) :
  f (x, y) = (3, 1) → (x, y) = (2, 1/2) := by
  sorry

end preimage_of_3_1_l869_86992


namespace log_simplification_l869_86958

theorem log_simplification (u v w t : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) (ht : t > 0) :
  Real.log (u / v) + Real.log (v / (2 * w)) + Real.log (w / (4 * t)) - Real.log (u / t) = Real.log (1 / 8) := by
  sorry

end log_simplification_l869_86958


namespace unique_positive_solution_l869_86959

theorem unique_positive_solution : 
  ∃! (x : ℝ), x > 0 ∧ x^101 + 100^99 = x^99 + 100^101 :=
by sorry

end unique_positive_solution_l869_86959


namespace peony_count_l869_86925

theorem peony_count (n : ℕ) 
  (h1 : ∃ (x : ℕ), n = 4*x + 2*x + 6*x) 
  (h2 : ∃ (y : ℕ), 6*y - 4*y = 30) 
  (h3 : ∃ (z : ℕ), 4 + 2 + 6 = 12) : n = 180 := by
  sorry

end peony_count_l869_86925


namespace max_product_sum_2004_l869_86963

theorem max_product_sum_2004 :
  (∃ (a b : ℤ), a + b = 2004 ∧ a * b = 1004004) ∧
  (∀ (x y : ℤ), x + y = 2004 → x * y ≤ 1004004) := by
  sorry

end max_product_sum_2004_l869_86963


namespace fraction_simplification_fraction_value_at_one_l869_86932

theorem fraction_simplification (x : ℤ) (h1 : -2 < x) (h2 : x < 2) (h3 : x ≠ 0) :
  (((x^2 - 1) / (x^2 + 2*x + 1)) / ((1 / (x + 1)) - 1)) = -(x - 1) / x :=
sorry

theorem fraction_value_at_one :
  (((1^2 - 1) / (1^2 + 2*1 + 1)) / ((1 / (1 + 1)) - 1)) = 0 :=
sorry

end fraction_simplification_fraction_value_at_one_l869_86932


namespace smallest_five_digit_number_with_conditions_l869_86970

theorem smallest_five_digit_number_with_conditions : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- 5-digit number
  (n % 32 = 0) ∧              -- divisible by 32
  (n % 45 = 0) ∧              -- divisible by 45
  (n % 54 = 0) ∧              -- divisible by 54
  (30 % n = 0) ∧              -- factor of 30
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m % 32 = 0) ∧ (m % 45 = 0) ∧ (m % 54 = 0) ∧ (30 % m = 0) → n ≤ m) ∧
  n = 12960 :=
by
  sorry

end smallest_five_digit_number_with_conditions_l869_86970


namespace fraction_addition_l869_86917

theorem fraction_addition : 
  (7 : ℚ) / 12 + (11 : ℚ) / 16 = (61 : ℚ) / 48 :=
by sorry

end fraction_addition_l869_86917


namespace line_through_point_l869_86941

/-- Given a line described by the equation 2 - kx = -4y that contains the point (3, 1),
    prove that k = 2. -/
theorem line_through_point (k : ℝ) : 
  (2 - k * 3 = -4 * 1) → k = 2 := by
  sorry

end line_through_point_l869_86941


namespace square_root_of_10_factorial_div_210_l869_86948

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem square_root_of_10_factorial_div_210 :
  ∃ (x : ℝ), x > 0 ∧ x^2 = (factorial 10 : ℝ) / 210 ∧ x = 24 * Real.sqrt 30 := by
  sorry

end square_root_of_10_factorial_div_210_l869_86948


namespace f_negative_three_value_l869_86998

/-- Given a function f(x) = a*sin(x) + b*tan(x) + x^3 + 1, 
    if f(3) = 7, then f(-3) = -5 -/
theorem f_negative_three_value 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * Real.sin x + b * Real.tan x + x^3 + 1) 
  (h2 : f 3 = 7) : 
  f (-3) = -5 := by sorry

end f_negative_three_value_l869_86998


namespace prime_divisibility_condition_l869_86918

theorem prime_divisibility_condition (p : ℕ) (x : ℕ) :
  Prime p →
  1 ≤ x ∧ x ≤ 2 * p →
  (x^(p-1) ∣ (p-1)^x + 1) ↔ 
  ((p = 2 ∧ x = 2) ∨ (p = 3 ∧ x = 3) ∨ (x = 1)) :=
by sorry

end prime_divisibility_condition_l869_86918


namespace eight_factorial_equals_product_l869_86936

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem eight_factorial_equals_product : 4 * 6 * 3 * 560 = factorial 8 := by
  sorry


end eight_factorial_equals_product_l869_86936


namespace sqrt_difference_product_l869_86951

theorem sqrt_difference_product : (Real.sqrt 6 + Real.sqrt 11) * (Real.sqrt 6 - Real.sqrt 11) = -5 := by
  sorry

end sqrt_difference_product_l869_86951


namespace stirling_duality_l869_86931

/-- Stirling number of the second kind -/
def stirling2 (N n : ℕ) : ℕ := sorry

/-- Stirling number of the first kind -/
def stirling1 (n M : ℕ) : ℤ := sorry

/-- Kronecker delta -/
def kroneckerDelta (N M : ℕ) : ℕ :=
  if N = M then 1 else 0

/-- The duality property of Stirling numbers -/
theorem stirling_duality (N M : ℕ) :
  (∑' n, (stirling2 N n : ℤ) * stirling1 n M) = kroneckerDelta N M := by
  sorry

end stirling_duality_l869_86931


namespace log_product_equals_one_third_l869_86919

theorem log_product_equals_one_third :
  Real.log 2 / Real.log 3 *
  Real.log 3 / Real.log 4 *
  Real.log 4 / Real.log 5 *
  Real.log 5 / Real.log 6 *
  Real.log 6 / Real.log 7 *
  Real.log 7 / Real.log 8 = 1/3 := by
  sorry

end log_product_equals_one_third_l869_86919


namespace set_operation_result_l869_86903

def X : Set ℕ := {0, 1, 2, 4, 5, 7}
def Y : Set ℕ := {1, 3, 6, 8, 9}
def Z : Set ℕ := {3, 7, 8}

theorem set_operation_result : (X ∩ Y) ∪ Z = {1, 3, 7, 8} := by sorry

end set_operation_result_l869_86903


namespace distance_P_to_xaxis_l869_86916

/-- The distance from a point to the x-axis is the absolute value of its y-coordinate -/
def distanceToXAxis (p : ℝ × ℝ) : ℝ := |p.2|

/-- Point P with coordinates (-3, 1) -/
def P : ℝ × ℝ := (-3, 1)

/-- Theorem: The distance from point P to the x-axis is 1 -/
theorem distance_P_to_xaxis : distanceToXAxis P = 1 := by
  sorry

end distance_P_to_xaxis_l869_86916


namespace sequence_sum_bound_l869_86973

/-- Given a sequence of positive integers satisfying certain conditions, 
    prove that the sum of its first n terms is at most n². -/
theorem sequence_sum_bound (n : ℕ) (a : ℕ → ℕ) : n > 0 →
  (∀ i, a (i + n) = a i) →
  (∀ i ∈ Finset.range n, a i > 0) →
  (∀ i ∈ Finset.range (n - 1), a i ≤ a (i + 1)) →
  a n ≤ a 1 + n →
  (∀ i ∈ Finset.range n, a (a i) ≤ n + i) →
  (Finset.range n).sum a ≤ n^2 := by
  sorry


end sequence_sum_bound_l869_86973


namespace special_ap_sums_l869_86947

/-- An arithmetic progression with special properties -/
structure SpecialAP where
  m : ℕ
  n : ℕ
  sum_m_terms : ℕ
  sum_n_terms : ℕ
  h1 : sum_m_terms = n
  h2 : sum_n_terms = m

/-- The sum of (m+n) terms and (m-n) terms for a SpecialAP -/
def special_sums (ap : SpecialAP) : ℤ × ℚ :=
  (-(ap.m + ap.n : ℤ), (ap.m - ap.n : ℚ) * (2 * ap.n + ap.m) / ap.m)

/-- Theorem stating the sums of (m+n) and (m-n) terms for a SpecialAP -/
theorem special_ap_sums (ap : SpecialAP) :
  special_sums ap = (-(ap.m + ap.n : ℤ), (ap.m - ap.n : ℚ) * (2 * ap.n + ap.m) / ap.m) := by
  sorry

#check special_ap_sums

end special_ap_sums_l869_86947


namespace greatest_number_with_odd_factors_has_odd_factors_196_is_less_than_200_196_greatest_number_less_than_200_with_odd_factors_l869_86966

def has_odd_number_of_factors (n : ℕ) : Prop :=
  Odd (Finset.card (Finset.filter (·∣n) (Finset.range (n + 1))))

theorem greatest_number_with_odd_factors : 
  ∀ n : ℕ, n < 200 → has_odd_number_of_factors n → n ≤ 196 :=
by sorry

theorem has_odd_factors_196 : has_odd_number_of_factors 196 :=
by sorry

theorem is_less_than_200_196 : 196 < 200 :=
by sorry

theorem greatest_number_less_than_200_with_odd_factors :
  ∃ n : ℕ, n < 200 ∧ has_odd_number_of_factors n ∧
  ∀ m : ℕ, m < 200 → has_odd_number_of_factors m → m ≤ n :=
by sorry

end greatest_number_with_odd_factors_has_odd_factors_196_is_less_than_200_196_greatest_number_less_than_200_with_odd_factors_l869_86966


namespace max_common_segment_length_theorem_l869_86997

/-- The maximum length of the common initial segment of two sequences with coprime periods -/
def max_common_segment_length (m n : ℕ) : ℕ :=
  m + n - 2

/-- Theorem stating that for two sequences with coprime periods m and n,
    the maximum length of their common initial segment is m + n - 2 -/
theorem max_common_segment_length_theorem (m n : ℕ) (h : Nat.Coprime m n) :
  max_common_segment_length m n = m + n - 2 := by
  sorry


end max_common_segment_length_theorem_l869_86997


namespace intersection_equality_l869_86987

theorem intersection_equality (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
  let B : Set ℝ := {x | a*x - 1 = 0}
  (A ∩ B = B) → (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by
  sorry

end intersection_equality_l869_86987


namespace john_squat_difference_l869_86900

/-- Given John's raw squat weight, the weight added by sleeves, and the percentage added by wraps,
    calculate the difference between the weight added by wraps and sleeves. -/
def weight_difference (raw_squat : ℝ) (sleeve_addition : ℝ) (wrap_percentage : ℝ) : ℝ :=
  raw_squat * wrap_percentage - sleeve_addition

/-- Prove that the difference between the weight added by wraps and sleeves to John's squat is 120 pounds. -/
theorem john_squat_difference :
  weight_difference 600 30 0.25 = 120 := by
  sorry

end john_squat_difference_l869_86900


namespace double_iced_subcubes_count_l869_86950

/-- Represents a 3D cube with icing on some faces -/
structure IcedCube where
  size : Nat
  top_iced : Bool
  front_iced : Bool
  right_iced : Bool

/-- Counts the number of 1x1x1 subcubes with icing on exactly two faces -/
def count_double_iced_subcubes (cube : IcedCube) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem double_iced_subcubes_count (cake : IcedCube) : 
  cake.size = 5 ∧ cake.top_iced ∧ cake.front_iced ∧ cake.right_iced →
  count_double_iced_subcubes cake = 32 :=
by sorry

end double_iced_subcubes_count_l869_86950


namespace intersection_equals_open_closed_interval_l869_86975

-- Define set A
def A : Set ℝ := {x | x^2 - 1 ≤ 0}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_open_closed_interval :
  A_intersect_B = Set.Ioo 0 1 ∪ Set.Ioc 1 1 :=
sorry

end intersection_equals_open_closed_interval_l869_86975


namespace seven_by_seven_dissection_l869_86906

theorem seven_by_seven_dissection :
  ∀ (a b : ℕ),
  (3 * a + 4 * b = 7 * 7) →
  (b = 1) := by
sorry

end seven_by_seven_dissection_l869_86906


namespace attitude_gender_relationship_expected_value_X_l869_86904

-- Define the survey data
def total_sample : ℕ := 200
def male_agree : ℕ := 70
def male_disagree : ℕ := 30
def female_agree : ℕ := 50
def female_disagree : ℕ := 50

-- Define the chi-square function
def chi_square (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% confidence
def critical_value : ℚ := 6635 / 1000

-- Define the probability of agreeing
def p_agree : ℚ := (male_agree + female_agree) / total_sample

-- Theorem 1: Relationship between attitudes and gender
theorem attitude_gender_relationship :
  chi_square total_sample male_agree female_agree male_disagree female_disagree > critical_value :=
sorry

-- Theorem 2: Expected value of X
theorem expected_value_X :
  (3 : ℚ) * p_agree = 9 / 5 :=
sorry

end attitude_gender_relationship_expected_value_X_l869_86904


namespace f_composition_eq_one_l869_86911

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 1 else x - 3

theorem f_composition_eq_one (x : ℝ) :
  f (f x) = 1 ↔ x ∈ Set.union (Set.Icc 0 1) (Set.union (Set.Icc 3 4) {7}) := by
  sorry

end f_composition_eq_one_l869_86911


namespace sam_seashells_l869_86952

theorem sam_seashells (initial_seashells : ℕ) (given_away : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 35)
  (h2 : given_away = 18)
  (h3 : remaining_seashells = initial_seashells - given_away) :
  remaining_seashells = 17 := by
  sorry

end sam_seashells_l869_86952


namespace total_cans_collected_l869_86964

def saturday_bags : ℕ := 3
def sunday_bags : ℕ := 4
def cans_per_bag : ℕ := 9

theorem total_cans_collected :
  saturday_bags * cans_per_bag + sunday_bags * cans_per_bag = 63 :=
by sorry

end total_cans_collected_l869_86964


namespace james_and_louise_ages_l869_86968

theorem james_and_louise_ages :
  ∀ (james louise : ℕ),
  james = louise + 6 →
  james + 8 = 4 * (louise - 4) →
  james + louise = 26 :=
by
  sorry

end james_and_louise_ages_l869_86968


namespace no_valid_arrangement_l869_86940

theorem no_valid_arrangement :
  ¬ ∃ (x y : ℕ), 
    90 = x * y ∧ 
    5 ≤ x ∧ x ≤ 20 ∧ 
    Even y :=
by sorry

end no_valid_arrangement_l869_86940


namespace modulus_of_complex_number_l869_86988

theorem modulus_of_complex_number :
  let z : ℂ := (1 + 2*Complex.I) / Complex.I^2
  Complex.abs z = Real.sqrt 5 := by sorry

end modulus_of_complex_number_l869_86988


namespace michael_digging_time_l869_86945

/-- Given the conditions of Michael's and his father's hole digging, prove that Michael will take 700 hours to dig his hole. -/
theorem michael_digging_time (father_rate : ℝ) (father_time : ℝ) (michael_depth_diff : ℝ) :
  father_rate = 4 →
  father_time = 400 →
  michael_depth_diff = 400 →
  (2 * (father_rate * father_time) - michael_depth_diff) / father_rate = 700 :=
by sorry

end michael_digging_time_l869_86945


namespace set_equality_implies_sum_l869_86913

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → a^2002 + b^2003 = 1 := by
  sorry

end set_equality_implies_sum_l869_86913


namespace train_crossing_time_l869_86901

/-- Calculates the time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 100 →
  train_speed = 54.99520038396929 →
  man_speed = 5 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 6 := by
  sorry

end train_crossing_time_l869_86901


namespace distance_against_current_equals_swimming_speed_l869_86937

/-- The distance swam against the current given swimming speed in still water and current speed -/
def distanceAgainstCurrent (swimmingSpeed currentSpeed : ℝ) : ℝ :=
  swimmingSpeed

theorem distance_against_current_equals_swimming_speed
  (swimmingSpeed currentSpeed : ℝ)
  (h1 : swimmingSpeed = 12)
  (h2 : currentSpeed = 10)
  (h3 : swimmingSpeed > currentSpeed) :
  distanceAgainstCurrent swimmingSpeed currentSpeed = 12 := by
  sorry

#eval distanceAgainstCurrent 12 10

end distance_against_current_equals_swimming_speed_l869_86937


namespace net_amount_is_2550_l869_86923

/-- Calculates the net amount received from selling puppies given the specified conditions -/
def calculate_net_amount (first_litter : ℕ) (second_litter : ℕ) 
  (first_price : ℕ) (second_price : ℕ) (raising_cost : ℕ) : ℕ :=
  let sold_first := (first_litter * 3) / 4
  let sold_second := (second_litter * 3) / 4
  let revenue := sold_first * first_price + sold_second * second_price
  let expenses := (first_litter + second_litter) * raising_cost
  revenue - expenses

/-- The net amount received from selling puppies under the given conditions is $2550 -/
theorem net_amount_is_2550 : 
  calculate_net_amount 10 12 200 250 50 = 2550 := by
  sorry

end net_amount_is_2550_l869_86923


namespace arrangements_eq_combinations_l869_86939

/-- The number of ways to arrange nine 1s and four 0s in a row, where no two 0s are adjacent -/
def arrangements : ℕ := sorry

/-- The number of ways to choose 4 items from 10 items -/
def combinations : ℕ := Nat.choose 10 4

/-- Theorem stating that the number of arrangements is equal to the number of combinations -/
theorem arrangements_eq_combinations : arrangements = combinations := by sorry

end arrangements_eq_combinations_l869_86939


namespace max_sum_of_factors_l869_86909

theorem max_sum_of_factors (diamond delta : ℕ) : 
  diamond * delta = 36 → (∀ x y : ℕ, x * y = 36 → x + y ≤ diamond + delta) → diamond + delta = 37 := by
  sorry

end max_sum_of_factors_l869_86909


namespace unique_assignment_l869_86965

-- Define the students and authors as enums
inductive Student : Type
| ZhangBoyuan : Student
| GaoJiaming : Student
| LiuYuheng : Student

inductive Author : Type
| Shakespeare : Author
| Hugo : Author
| CaoXueqin : Author

-- Define the assignment of authors to students
def Assignment := Student → Author

-- Define the condition that each student has a different author
def all_different (a : Assignment) : Prop :=
  ∀ s1 s2 : Student, s1 ≠ s2 → a s1 ≠ a s2

-- Define Teacher Liu's guesses
def guess1 (a : Assignment) : Prop := a Student.ZhangBoyuan = Author.Shakespeare
def guess2 (a : Assignment) : Prop := a Student.LiuYuheng ≠ Author.CaoXueqin
def guess3 (a : Assignment) : Prop := a Student.GaoJiaming ≠ Author.Shakespeare

-- Define the condition that only one guess is correct
def only_one_correct (a : Assignment) : Prop :=
  (guess1 a ∧ ¬guess2 a ∧ ¬guess3 a) ∨
  (¬guess1 a ∧ guess2 a ∧ ¬guess3 a) ∨
  (¬guess1 a ∧ ¬guess2 a ∧ guess3 a)

-- The main theorem
theorem unique_assignment :
  ∃! a : Assignment,
    all_different a ∧
    only_one_correct a ∧
    a Student.ZhangBoyuan = Author.CaoXueqin ∧
    a Student.GaoJiaming = Author.Shakespeare ∧
    a Student.LiuYuheng = Author.Hugo :=
  sorry

end unique_assignment_l869_86965


namespace cone_volume_l869_86974

/-- Given a cone with slant height 1 and lateral surface area 2π/3, its volume is 4√5π/81 -/
theorem cone_volume (s : Real) (A : Real) (V : Real) : 
  s = 1 → A = (2/3) * Real.pi → V = (4 * Real.sqrt 5 / 81) * Real.pi :=
by sorry

end cone_volume_l869_86974


namespace employee_hire_year_l869_86971

/-- Represents the rule of 70 provision for retirement eligibility -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Calculates the year an employee was hired given their retirement eligibility year and years of employment -/
def hire_year (retirement_eligibility_year : ℕ) (years_employed : ℕ) : ℕ :=
  retirement_eligibility_year - years_employed

theorem employee_hire_year :
  ∀ (retirement_eligibility_year : ℕ) (hire_age : ℕ),
    hire_age = 32 →
    retirement_eligibility_year = 2009 →
    (∃ (years_employed : ℕ), rule_of_70 (hire_age + years_employed) years_employed) →
    hire_year retirement_eligibility_year (retirement_eligibility_year - (hire_age + 32)) = 1971 :=
by sorry

end employee_hire_year_l869_86971


namespace maxwell_age_proof_l869_86994

/-- Maxwell's current age --/
def maxwell_age : ℕ := 6

/-- Maxwell's sister's current age --/
def sister_age : ℕ := 2

/-- Years into the future when the age relationship holds --/
def years_future : ℕ := 2

theorem maxwell_age_proof :
  maxwell_age = 6 ∧
  sister_age = 2 ∧
  maxwell_age + years_future = 2 * (sister_age + years_future) :=
by sorry

end maxwell_age_proof_l869_86994


namespace largest_prime_factor_is_29_l869_86995

def numbers : List Nat := [145, 187, 221, 299, 169]

/-- Returns the largest prime factor of a natural number -/
def largestPrimeFactor (n : Nat) : Nat :=
  sorry

theorem largest_prime_factor_is_29 : 
  ∀ n ∈ numbers, largestPrimeFactor n ≤ 29 ∧ 
  ∃ m ∈ numbers, largestPrimeFactor m = 29 :=
sorry

end largest_prime_factor_is_29_l869_86995


namespace peanut_butter_jars_l869_86943

/-- Given the total amount of peanut butter and jar sizes, calculate the number of jars. -/
def number_of_jars (total_ounces : ℕ) (jar_sizes : List ℕ) : ℕ :=
  if jar_sizes.length = 0 then 0
  else
    let jars_per_size := total_ounces / (jar_sizes.sum)
    jars_per_size * jar_sizes.length

/-- Theorem stating that given 252 ounces of peanut butter in equal numbers of 16, 28, and 40 ounce jars, the total number of jars is 9. -/
theorem peanut_butter_jars :
  number_of_jars 252 [16, 28, 40] = 9 := by
  sorry

end peanut_butter_jars_l869_86943


namespace vector_addition_l869_86999

/-- Given two vectors OA and AB in 2D space, prove that OB is their sum. -/
theorem vector_addition (OA AB : ℝ × ℝ) (h1 : OA = (-2, 3)) (h2 : AB = (-1, -4)) :
  OA + AB = (-3, -1) := by
  sorry

end vector_addition_l869_86999


namespace dasha_ate_one_bowl_l869_86996

/-- The number of bowls of porridge eaten by each monkey -/
structure MonkeyPorridge where
  masha : ℕ
  dasha : ℕ
  glasha : ℕ
  natasha : ℕ

/-- The conditions of the monkey porridge problem -/
def MonkeyPorridgeConditions (mp : MonkeyPorridge) : Prop :=
  mp.masha + mp.dasha + mp.glasha + mp.natasha = 16 ∧
  mp.glasha + mp.natasha = 9 ∧
  mp.masha > mp.dasha ∧
  mp.masha > mp.glasha ∧
  mp.masha > mp.natasha

theorem dasha_ate_one_bowl (mp : MonkeyPorridge) 
  (h : MonkeyPorridgeConditions mp) : mp.dasha = 1 := by
  sorry

end dasha_ate_one_bowl_l869_86996


namespace power_calculations_l869_86957

theorem power_calculations :
  ((-2 : ℤ) ^ (0 : ℕ) = 1) ∧
  ((-3 : ℚ) ^ (-3 : ℤ) = -1/27) := by
  sorry

end power_calculations_l869_86957
