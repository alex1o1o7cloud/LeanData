import Mathlib

namespace NUMINAMATH_CALUDE_unique_perfect_square_P_l2949_294962

def P (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 3*x + 31

theorem unique_perfect_square_P :
  ∃! x : ℤ, ∃ y : ℤ, P x = y^2 :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_P_l2949_294962


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l2949_294936

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define a point (x, y) on the graph of y = f(x)
variable (x y : ℝ)

-- Define the reflection transformation across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- State the theorem
theorem reflection_across_y_axis :
  y = f x ↔ y = f (-(-x)) :=
sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l2949_294936


namespace NUMINAMATH_CALUDE_shopping_cost_calculation_l2949_294971

/-- Calculates the total cost of a shopping trip and determines the additional amount needed --/
theorem shopping_cost_calculation (shirts_count sunglasses_count skirts_count sandals_count hats_count bags_count earrings_count : ℕ)
  (shirt_price sunglasses_price skirt_price sandal_price hat_price bag_price earring_price : ℚ)
  (discount_rate tax_rate : ℚ) (payment : ℚ) :
  let subtotal := shirts_count * shirt_price + sunglasses_count * sunglasses_price + 
                  skirts_count * skirt_price + sandals_count * sandal_price + 
                  hats_count * hat_price + bags_count * bag_price + 
                  earrings_count * earring_price
  let discounted_total := subtotal * (1 - discount_rate)
  let final_total := discounted_total * (1 + tax_rate)
  let change_needed := final_total - payment
  shirts_count = 10 ∧ sunglasses_count = 2 ∧ skirts_count = 4 ∧
  sandals_count = 3 ∧ hats_count = 5 ∧ bags_count = 7 ∧ earrings_count = 6 ∧
  shirt_price = 5 ∧ sunglasses_price = 12 ∧ skirt_price = 18 ∧
  sandal_price = 3 ∧ hat_price = 8 ∧ bag_price = 14 ∧ earring_price = 6 ∧
  discount_rate = 1/10 ∧ tax_rate = 13/200 ∧ payment = 300 →
  change_needed = 307/20 := by
  sorry

end NUMINAMATH_CALUDE_shopping_cost_calculation_l2949_294971


namespace NUMINAMATH_CALUDE_nth_equation_holds_l2949_294969

theorem nth_equation_holds (n : ℕ) :
  1 - 1 / ((n + 1: ℚ) ^ 2) = (n / (n + 1 : ℚ)) * ((n + 2) / (n + 1 : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_holds_l2949_294969


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2949_294931

theorem quadratic_equation_result (x : ℝ) (h : x^2 - 2*x + 1 = 0) : 2*x^2 - 4*x = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2949_294931


namespace NUMINAMATH_CALUDE_exists_uncovered_cell_l2949_294949

/-- Represents a grid cell --/
structure Cell :=
  (x : Nat) (y : Nat)

/-- Represents a rectangle --/
structure Rectangle :=
  (width : Nat) (height : Nat)

/-- The dimensions of the grid --/
def gridWidth : Nat := 11
def gridHeight : Nat := 1117

/-- The dimensions of the cutting rectangle --/
def cuttingRectangle : Rectangle := { width := 6, height := 1 }

/-- A function to check if a cell is covered by a rectangle --/
def isCovered (c : Cell) (r : Rectangle) (position : Cell) : Prop :=
  c.x ≥ position.x ∧ c.x < position.x + r.width ∧
  c.y ≥ position.y ∧ c.y < position.y + r.height

/-- The main theorem --/
theorem exists_uncovered_cell :
  ∃ (c : Cell), c.x < gridWidth ∧ c.y < gridHeight ∧
  ∀ (arrangements : List Cell),
    ∃ (p : Cell), p ∈ arrangements →
      ¬(isCovered c cuttingRectangle p) :=
sorry

end NUMINAMATH_CALUDE_exists_uncovered_cell_l2949_294949


namespace NUMINAMATH_CALUDE_net_cash_change_l2949_294919

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  ownsHouse : Bool

/-- Represents a house transaction -/
inductive Transaction
  | Rent : Transaction
  | BuyHouse : Int → Transaction
  | SellHouse : Int → Transaction

def initialValueA : Int := 15000
def initialValueB : Int := 20000
def initialHouseValue : Int := 15000
def rentAmount : Int := 2000

def applyTransaction (state : FinancialState) (transaction : Transaction) : FinancialState :=
  match transaction with
  | Transaction.Rent => 
      if state.ownsHouse then 
        { cash := state.cash + rentAmount, ownsHouse := state.ownsHouse }
      else 
        { cash := state.cash - rentAmount, ownsHouse := state.ownsHouse }
  | Transaction.BuyHouse price => 
      { cash := state.cash - price, ownsHouse := true }
  | Transaction.SellHouse price => 
      { cash := state.cash + price, ownsHouse := false }

def transactions : List Transaction := [
  Transaction.Rent,
  Transaction.SellHouse 18000,
  Transaction.BuyHouse 17000
]

theorem net_cash_change 
  (initialA : FinancialState) 
  (initialB : FinancialState) 
  (finalA : FinancialState) 
  (finalB : FinancialState) :
  initialA = { cash := initialValueA, ownsHouse := true } →
  initialB = { cash := initialValueB, ownsHouse := false } →
  finalA = transactions.foldl applyTransaction initialA →
  finalB = transactions.foldl applyTransaction initialB →
  finalA.cash - initialA.cash = 3000 ∧ 
  finalB.cash - initialB.cash = -3000 :=
sorry

end NUMINAMATH_CALUDE_net_cash_change_l2949_294919


namespace NUMINAMATH_CALUDE_correct_num_clowns_l2949_294964

/-- The number of clowns attending a carousel --/
def num_clowns : ℕ := 4

/-- The number of children attending the carousel --/
def num_children : ℕ := 30

/-- The total number of candies initially --/
def total_candies : ℕ := 700

/-- The number of candies given to each person --/
def candies_per_person : ℕ := 20

/-- The number of candies left after distribution --/
def candies_left : ℕ := 20

/-- Theorem stating that the number of clowns is correct given the conditions --/
theorem correct_num_clowns :
  num_clowns * candies_per_person + num_children * candies_per_person + candies_left = total_candies :=
by sorry

end NUMINAMATH_CALUDE_correct_num_clowns_l2949_294964


namespace NUMINAMATH_CALUDE_truck_rental_problem_l2949_294930

/-- The total number of trucks on Monday morning -/
def total_trucks : ℕ := 30

/-- The number of trucks rented out during the week -/
def rented_trucks : ℕ := 20

/-- The number of trucks returned by Saturday morning -/
def returned_trucks : ℕ := rented_trucks / 2

/-- The number of trucks on the lot Saturday morning -/
def saturday_trucks : ℕ := returned_trucks

theorem truck_rental_problem :
  (returned_trucks = rented_trucks / 2) →
  (saturday_trucks ≥ 10) →
  (rented_trucks = 20) →
  (total_trucks = rented_trucks + (rented_trucks - returned_trucks)) :=
by sorry

end NUMINAMATH_CALUDE_truck_rental_problem_l2949_294930


namespace NUMINAMATH_CALUDE_parallel_line_through_point_main_theorem_l2949_294925

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (given_line : Line) 
  (point : Point) 
  (result_line : Line) : Prop :=
  given_line.a = 2 ∧ 
  given_line.b = -1 ∧ 
  given_line.c = 1 ∧ 
  point.x = -1 ∧ 
  point.y = 0 ∧ 
  result_line.a = 2 ∧ 
  result_line.b = -1 ∧ 
  result_line.c = 2 ∧ 
  point.liesOn result_line ∧ 
  result_line.isParallel given_line

/-- The main theorem stating that the resulting line equation is correct -/
theorem main_theorem : ∃ (given_line result_line : Line) (point : Point), 
  parallel_line_through_point given_line point result_line := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_main_theorem_l2949_294925


namespace NUMINAMATH_CALUDE_max_value_of_f_l2949_294908

def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = -2) →
  (∃ x ∈ Set.Icc 0 1, f a x = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2949_294908


namespace NUMINAMATH_CALUDE_simplify_expression_l2949_294929

theorem simplify_expression : 80 - (5 - (6 + 2 * (7 - 8 - 5))) = 69 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2949_294929


namespace NUMINAMATH_CALUDE_ellipse_trajectory_l2949_294966

/-- The trajectory of point Q given an ellipse and its properties -/
theorem ellipse_trajectory (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∀ (x y : ℝ), 
  ((-x/2)^2 / a^2 + (-y/2)^2 / b^2 = 1) →
  (x^2 / (4*a^2) + y^2 / (4*b^2) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_trajectory_l2949_294966


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2949_294950

open Set

/-- Solution set for the quadratic inequality ax^2 + (1-a)x - 1 > 0 -/
def SolutionSet (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 1}
  else if a > 0 then {x | x < -1/a ∨ x > 1}
  else if -1 < a then {x | 1 < x ∧ x < -1/a}
  else univ

theorem quadratic_inequality_solution_set :
  (∀ x, x ∈ SolutionSet 2 ↔ (x < -1/2 ∨ x > 1)) ∧
  (∀ a, a > -1 → ∀ x, x ∈ SolutionSet a ↔
    (a = 0 ∧ x > 1) ∨
    (a > 0 ∧ (x < -1/a ∨ x > 1)) ∨
    (-1 < a ∧ a < 0 ∧ 1 < x ∧ x < -1/a)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2949_294950


namespace NUMINAMATH_CALUDE_marathon_remainder_l2949_294934

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def marathon : Marathon :=
  { miles := 26, yards := 385 }

def yardsPerMile : ℕ := 1760

def numMarathons : ℕ := 15

theorem marathon_remainder (m : ℕ) (y : ℕ) 
  (h : Distance.mk m y = 
    { miles := numMarathons * marathon.miles + (numMarathons * marathon.yards) / yardsPerMile,
      yards := (numMarathons * marathon.yards) % yardsPerMile }) :
  y = 495 := by
  sorry

end NUMINAMATH_CALUDE_marathon_remainder_l2949_294934


namespace NUMINAMATH_CALUDE_triangle_area_similarity_l2949_294909

-- Define the triangles
variable (A B C D E F : ℝ × ℝ)

-- Define the similarity relation
def similar (t1 t2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

-- Define the area function
def area (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

-- Define the side length function
def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_similarity :
  similar (A, B, C) (D, E, F) →
  side_length A B / side_length D E = 2 →
  area (A, B, C) = 8 →
  area (D, E, F) = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_similarity_l2949_294909


namespace NUMINAMATH_CALUDE_committee_formation_theorem_l2949_294972

/-- The number of ways to form a committee with leaders --/
def committee_formation_ways (n m k : ℕ) : ℕ :=
  (Nat.choose n m) * (2^m - 2)

/-- Theorem stating the number of ways to form the committee --/
theorem committee_formation_theorem :
  committee_formation_ways 10 5 4 = 7560 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_theorem_l2949_294972


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l2949_294958

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 20 = 12 →
  arithmetic_sequence a₁ d 21 = 16 →
  arithmetic_sequence a₁ d 5 = -48 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l2949_294958


namespace NUMINAMATH_CALUDE_max_black_cells_l2949_294943

/-- Represents a board with black and white cells -/
def Board (n : ℕ) := Fin (2*n+1) → Fin (2*n+1) → Bool

/-- Checks if a 2x2 sub-board has at most 2 black cells -/
def ValidSubBoard (b : Board n) (i j : Fin (2*n)) : Prop :=
  (b i j).toNat + (b i (j+1)).toNat + (b (i+1) j).toNat + (b (i+1) (j+1)).toNat ≤ 2

/-- A board is valid if all its 2x2 sub-boards have at most 2 black cells -/
def ValidBoard (b : Board n) : Prop :=
  ∀ i j, ValidSubBoard b i j

/-- Counts the number of black cells in a board -/
def CountBlackCells (b : Board n) : ℕ :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => (b i j).toNat))

/-- The maximum number of black cells in a valid (2n+1) × (2n+1) board is (2n+1)(n+1) -/
theorem max_black_cells (n : ℕ) :
  (∃ b : Board n, ValidBoard b ∧ CountBlackCells b = (2*n+1)*(n+1)) ∧
  (∀ b : Board n, ValidBoard b → CountBlackCells b ≤ (2*n+1)*(n+1)) := by
  sorry

end NUMINAMATH_CALUDE_max_black_cells_l2949_294943


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l2949_294905

theorem binomial_expansion_constant_term (n : ℕ+) :
  (Nat.choose n 2 = Nat.choose n 4) →
  (Nat.choose n (n / 2) = 20) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l2949_294905


namespace NUMINAMATH_CALUDE_derek_savings_l2949_294933

theorem derek_savings (P : ℚ) : P * 2^11 = 4096 → P = 2 := by
  sorry

end NUMINAMATH_CALUDE_derek_savings_l2949_294933


namespace NUMINAMATH_CALUDE_subcommittee_count_l2949_294942

def committee_size : ℕ := 7
def subcommittee_size : ℕ := 3

theorem subcommittee_count : 
  Nat.choose committee_size subcommittee_size = 35 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l2949_294942


namespace NUMINAMATH_CALUDE_shipment_composition_l2949_294923

/-- Represents a shipment of boxes with two possible weights -/
structure Shipment where
  total_boxes : ℕ
  weight1 : ℕ
  weight2 : ℕ
  count1 : ℕ
  count2 : ℕ
  initial_avg : ℚ

/-- Theorem about the composition of a specific shipment -/
theorem shipment_composition (s : Shipment) 
  (h1 : s.total_boxes = 30)
  (h2 : s.weight1 = 10)
  (h3 : s.weight2 = 20)
  (h4 : s.initial_avg = 18)
  (h5 : s.count1 + s.count2 = s.total_boxes)
  (h6 : s.weight1 * s.count1 + s.weight2 * s.count2 = s.initial_avg * s.total_boxes) :
  s.count1 = 6 ∧ s.count2 = 24 := by
  sorry

/-- Function to calculate the number of heavy boxes to remove to reach a target average -/
def boxes_to_remove (s : Shipment) (target_avg : ℚ) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_shipment_composition_l2949_294923


namespace NUMINAMATH_CALUDE_relay_race_ratio_l2949_294928

theorem relay_race_ratio (total_members : Nat) (other_members : Nat) (other_distance : ℝ) (total_distance : ℝ) :
  total_members = 5 →
  other_members = 4 →
  other_distance = 3 →
  total_distance = 18 →
  (total_distance - other_members * other_distance) / other_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_ratio_l2949_294928


namespace NUMINAMATH_CALUDE_parabola_points_relationship_l2949_294915

/-- Proves that for points A(2, y₁), B(3, y₂), and C(-1, y₃) lying on the parabola 
    y = ax² - 4ax + c where a > 0, the relationship y₁ < y₂ < y₃ holds. -/
theorem parabola_points_relationship (a c y₁ y₂ y₃ : ℝ) 
  (ha : a > 0)
  (h1 : y₁ = a * 2^2 - 4 * a * 2 + c)
  (h2 : y₂ = a * 3^2 - 4 * a * 3 + c)
  (h3 : y₃ = a * (-1)^2 - 4 * a * (-1) + c) :
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

#check parabola_points_relationship

end NUMINAMATH_CALUDE_parabola_points_relationship_l2949_294915


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2949_294993

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2949_294993


namespace NUMINAMATH_CALUDE_sara_quarters_l2949_294940

def cents : ℕ := 275
def cents_per_quarter : ℕ := 25

theorem sara_quarters : cents / cents_per_quarter = 11 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l2949_294940


namespace NUMINAMATH_CALUDE_base8_246_equals_base10_166_l2949_294961

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (d2 d1 d0 : ℕ) : ℕ :=
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

/-- The base 8 number 246₈ is equal to 166 in base 10 --/
theorem base8_246_equals_base10_166 : base8_to_base10 2 4 6 = 166 := by
  sorry

end NUMINAMATH_CALUDE_base8_246_equals_base10_166_l2949_294961


namespace NUMINAMATH_CALUDE_orange_bin_count_l2949_294977

/-- Given an initial quantity of oranges, a number of oranges removed, and a number of oranges added,
    calculate the final quantity of oranges. -/
def final_orange_count (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that given the specific values from the problem,
    the final orange count is 31. -/
theorem orange_bin_count : final_orange_count 5 2 28 = 31 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_count_l2949_294977


namespace NUMINAMATH_CALUDE_distance_X_to_CD_l2949_294974

/-- Square with side length 2s and quarter-circle arcs -/
structure SquareWithArcs (s : ℝ) :=
  (A B C D : ℝ × ℝ)
  (X : ℝ × ℝ)
  (h_square : A = (0, 0) ∧ B = (2*s, 0) ∧ C = (2*s, 2*s) ∧ D = (0, 2*s))
  (h_arc_A : (X.1 - A.1)^2 + (X.2 - A.2)^2 = (2*s)^2)
  (h_arc_B : (X.1 - B.1)^2 + (X.2 - B.2)^2 = (2*s)^2)
  (h_X_inside : 0 < X.1 ∧ X.1 < 2*s ∧ 0 < X.2 ∧ X.2 < 2*s)

/-- The distance from X to side CD in a SquareWithArcs is 2s(2 - √3) -/
theorem distance_X_to_CD (s : ℝ) (sq : SquareWithArcs s) :
  2*s - sq.X.2 = 2*s*(2 - Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_distance_X_to_CD_l2949_294974


namespace NUMINAMATH_CALUDE_coefficient_of_x4_l2949_294954

theorem coefficient_of_x4 (x : ℝ) : 
  let expression := 2*(x^2 - x^4 + 2*x^3) + 4*(x^4 - x^3 + x^2 + 2*x^5 - x^6) + 3*(2*x^3 + x^4 - 4*x^2)
  ∃ (a b c d e f : ℝ), expression = a*x^6 + b*x^5 + 5*x^4 + c*x^3 + d*x^2 + e*x + f :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x4_l2949_294954


namespace NUMINAMATH_CALUDE_inequalities_given_sum_positive_l2949_294985

theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_sum_positive_l2949_294985


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2949_294907

theorem max_value_of_expression (x y z : ℝ) (h : x + y + 2*z = 5) :
  ∃ (max : ℝ), max = 25/6 ∧ ∀ (a b c : ℝ), a + b + 2*c = 5 → a*b + a*c + b*c ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2949_294907


namespace NUMINAMATH_CALUDE_not_both_divisible_by_seven_l2949_294992

theorem not_both_divisible_by_seven (a b : ℝ) : 
  (¬ ∃ k : ℤ, a * b = 7 * k) → (¬ ∃ m : ℤ, a = 7 * m) ∧ (¬ ∃ n : ℤ, b = 7 * n) := by
  sorry

end NUMINAMATH_CALUDE_not_both_divisible_by_seven_l2949_294992


namespace NUMINAMATH_CALUDE_exact_fourth_power_implies_zero_coefficients_exact_square_implies_perfect_square_l2949_294953

-- Part (a)
theorem exact_fourth_power_implies_zero_coefficients 
  (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^4) :
  a = 0 ∧ b = 0 := by sorry

-- Part (b)
theorem exact_square_implies_perfect_square 
  (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ z : ℤ, a * x^2 + b * x + c = z^2) :
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 := by sorry

end NUMINAMATH_CALUDE_exact_fourth_power_implies_zero_coefficients_exact_square_implies_perfect_square_l2949_294953


namespace NUMINAMATH_CALUDE_emma_last_page_l2949_294946

/-- Represents a reader with their reading speed in seconds per page -/
structure Reader where
  name : String
  speed : ℕ

/-- Represents the novel reading scenario -/
structure NovelReading where
  totalPages : ℕ
  emma : Reader
  liam : Reader
  noah : Reader
  noahPages : ℕ

/-- Calculates the last page Emma should read -/
def lastPageForEmma (scenario : NovelReading) : ℕ :=
  sorry

/-- Theorem stating that the last page Emma should read is 525 -/
theorem emma_last_page (scenario : NovelReading) 
  (h1 : scenario.totalPages = 900)
  (h2 : scenario.emma = ⟨"Emma", 15⟩)
  (h3 : scenario.liam = ⟨"Liam", 45⟩)
  (h4 : scenario.noah = ⟨"Noah", 30⟩)
  (h5 : scenario.noahPages = 200)
  : lastPageForEmma scenario = 525 := by
  sorry

end NUMINAMATH_CALUDE_emma_last_page_l2949_294946


namespace NUMINAMATH_CALUDE_curve_family_point_condition_l2949_294960

/-- A point (x, y) lies on at least one curve of the family y = p^2 + (2p - 1)x + 2x^2 
    if and only if y ≥ x^2 - x -/
theorem curve_family_point_condition (x y : ℝ) : 
  (∃ p : ℝ, y = p^2 + (2*p - 1)*x + 2*x^2) ↔ y ≥ x^2 - x := by
sorry

end NUMINAMATH_CALUDE_curve_family_point_condition_l2949_294960


namespace NUMINAMATH_CALUDE_not_arithmetic_sequence_sqrt_2_3_5_l2949_294996

theorem not_arithmetic_sequence_sqrt_2_3_5 : ¬∃ (a b c : ℝ), 
  (a = Real.sqrt 2) ∧ 
  (b = Real.sqrt 3) ∧ 
  (c = Real.sqrt 5) ∧ 
  (b - a = c - b) :=
by sorry

end NUMINAMATH_CALUDE_not_arithmetic_sequence_sqrt_2_3_5_l2949_294996


namespace NUMINAMATH_CALUDE_function_identities_equivalence_l2949_294914

theorem function_identities_equivalence (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ↔
  (∀ x y : ℝ, f (x * y + x + y) = f (x * y) + f x + f y) :=
by sorry

end NUMINAMATH_CALUDE_function_identities_equivalence_l2949_294914


namespace NUMINAMATH_CALUDE_rectangle_area_l2949_294944

theorem rectangle_area (y : ℝ) (h : y > 0) :
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ l = 3 * w ∧ y^2 = l^2 + w^2 ∧ w * l = (3 * y^2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2949_294944


namespace NUMINAMATH_CALUDE_no_solution_equation_l2949_294913

theorem no_solution_equation : ¬∃ (x : ℝ), x - 7 / (x - 3) = 3 - 7 / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2949_294913


namespace NUMINAMATH_CALUDE_andrew_kept_stickers_l2949_294983

def total_stickers : ℕ := 2000

def daniel_stickers : ℕ := (total_stickers * 5) / 100

def fred_stickers : ℕ := daniel_stickers + 120

def emily_stickers : ℕ := ((daniel_stickers + fred_stickers) * 50) / 100

def gina_stickers : ℕ := 80

def hannah_stickers : ℕ := ((emily_stickers + gina_stickers) * 20) / 100

def total_given_away : ℕ := daniel_stickers + fred_stickers + emily_stickers + gina_stickers + hannah_stickers

theorem andrew_kept_stickers : total_stickers - total_given_away = 1392 := by
  sorry

end NUMINAMATH_CALUDE_andrew_kept_stickers_l2949_294983


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l2949_294975

/-- A geometric sequence with a specific sum formula -/
structure GeometricSequence where
  a : ℕ → ℝ
  sum : ℕ → ℝ
  sum_formula : ∀ n, sum n = 3^(n + 1) + a 1
  is_geometric : ∀ n, a (n + 2) * a n = (a (n + 1))^2

/-- The value of 'a' in the sum formula is -3 -/
theorem geometric_sequence_sum_constant (seq : GeometricSequence) : seq.a 1 - 9 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_constant_l2949_294975


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_nine_l2949_294968

theorem three_digit_divisible_by_nine :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 3 ∧ 
  (n / 100) % 10 = 5 ∧ 
  n % 9 = 0 ∧
  n = 513 :=
sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_nine_l2949_294968


namespace NUMINAMATH_CALUDE_a_range_l2949_294981

theorem a_range (a : ℝ) (h : a^(3/2) < a^(Real.sqrt 2)) : 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l2949_294981


namespace NUMINAMATH_CALUDE_quadrangular_pyramid_edge_length_l2949_294955

/-- A quadrangular pyramid with equal edge lengths -/
structure QuadrangularPyramid where
  edge_length : ℝ
  sum_of_edges : ℝ
  edge_sum_eq : sum_of_edges = 8 * edge_length

/-- Theorem: In a quadrangular pyramid with equal edge lengths, 
    if the sum of edge lengths is 14.8 meters, then each edge is 1.85 meters long -/
theorem quadrangular_pyramid_edge_length 
  (pyramid : QuadrangularPyramid) 
  (h : pyramid.sum_of_edges = 14.8) : 
  pyramid.edge_length = 1.85 := by
  sorry

#check quadrangular_pyramid_edge_length

end NUMINAMATH_CALUDE_quadrangular_pyramid_edge_length_l2949_294955


namespace NUMINAMATH_CALUDE_lottery_probabilities_l2949_294952

/-- Represents the probability of winning a single lottery event -/
def p : ℝ := 0.05

/-- The probability of winning both lotteries -/
def win_both : ℝ := p * p

/-- The probability of winning exactly one lottery -/
def win_one : ℝ := p * (1 - p) + (1 - p) * p

/-- The probability of winning at least one lottery -/
def win_at_least_one : ℝ := win_both + win_one

theorem lottery_probabilities :
  win_both = 0.0025 ∧ win_one = 0.095 ∧ win_at_least_one = 0.0975 := by
  sorry


end NUMINAMATH_CALUDE_lottery_probabilities_l2949_294952


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2949_294999

theorem simplify_and_evaluate (x : ℝ) (h : x ≠ 2) :
  (x + 2 + 3 / (x - 2)) / ((1 + 2*x + x^2) / (x - 2)) = (x - 1) / (x + 1) ∧
  (4 + 2 + 3 / (4 - 2)) / ((1 + 2*4 + 4^2) / (4 - 2)) = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2949_294999


namespace NUMINAMATH_CALUDE_harry_lost_nineteen_pencils_l2949_294973

/-- The number of pencils Anna has -/
def anna_pencils : ℕ := 50

/-- The number of pencils Harry initially had -/
def harry_initial_pencils : ℕ := 2 * anna_pencils

/-- The number of pencils Harry has left -/
def harry_remaining_pencils : ℕ := 81

/-- The number of pencils Harry lost -/
def harry_lost_pencils : ℕ := harry_initial_pencils - harry_remaining_pencils

theorem harry_lost_nineteen_pencils : harry_lost_pencils = 19 := by
  sorry

end NUMINAMATH_CALUDE_harry_lost_nineteen_pencils_l2949_294973


namespace NUMINAMATH_CALUDE_number_puzzle_l2949_294917

theorem number_puzzle : ∃ x : ℤ, x - 29 + 64 = 76 ∧ x = 41 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l2949_294917


namespace NUMINAMATH_CALUDE_area_of_AGKIJEFB_l2949_294956

-- Define the hexagons and point K
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ
  area : ℝ

def Point := ℝ × ℝ

-- Define the problem setup
axiom hexagon1 : Hexagon
axiom hexagon2 : Hexagon
axiom K : Point

-- State the conditions
axiom shared_side : hexagon1.vertices 4 = hexagon2.vertices 4 ∧ hexagon1.vertices 5 = hexagon2.vertices 5
axiom equal_areas : hexagon1.area = 36 ∧ hexagon2.area = 36
axiom K_on_AB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ K = (1 - t) • hexagon1.vertices 0 + t • hexagon1.vertices 1
axiom AK_KB_ratio : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a / b = 1 / 2 ∧
  K = (b / (a + b)) • hexagon1.vertices 0 + (a / (a + b)) • hexagon1.vertices 1
axiom K_midpoint_GH : K = (1 / 2) • hexagon2.vertices 0 + (1 / 2) • hexagon2.vertices 1

-- Define the polygon AGKIJEFB
def polygon_AGKIJEFB_area : ℝ := sorry

-- State the theorem to be proved
theorem area_of_AGKIJEFB : polygon_AGKIJEFB_area = 36 + Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_area_of_AGKIJEFB_l2949_294956


namespace NUMINAMATH_CALUDE_admission_cutoff_score_admission_cutoff_score_is_96_l2949_294912

theorem admission_cutoff_score (total_average : ℝ) (admitted_fraction : ℝ) 
  (admitted_score_diff : ℝ) (non_admitted_score_diff : ℝ) : ℝ :=
  let cutoff := total_average + (admitted_fraction * admitted_score_diff - 
    (1 - admitted_fraction) * non_admitted_score_diff)
  cutoff

theorem admission_cutoff_score_is_96 :
  admission_cutoff_score 90 (2/5) 15 20 = 96 := by
  sorry

end NUMINAMATH_CALUDE_admission_cutoff_score_admission_cutoff_score_is_96_l2949_294912


namespace NUMINAMATH_CALUDE_charity_event_revenue_l2949_294902

theorem charity_event_revenue (total_tickets : Nat) (total_revenue : Nat) 
  (full_price_tickets : Nat) (discount_tickets : Nat) (full_price : Nat) :
  total_tickets = 190 →
  total_revenue = 2871 →
  full_price_tickets + discount_tickets = total_tickets →
  full_price_tickets * full_price + discount_tickets * (full_price / 3) = total_revenue →
  full_price_tickets * full_price = 1900 :=
by sorry

end NUMINAMATH_CALUDE_charity_event_revenue_l2949_294902


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l2949_294979

/-- A line parallel to y = -3x - 6 passing through (3, -1) has y-intercept 8 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b x = y ↔ ∃ k, y = -3 * x + k) →  -- b is parallel to y = -3x - 6
  b 3 = -1 →                               -- b passes through (3, -1)
  ∃ k, b 0 = k ∧ k = 8 :=                  -- y-intercept of b is 8
by sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l2949_294979


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2949_294980

theorem trigonometric_simplification :
  let sin15 := Real.sin (15 * π / 180)
  let sin30 := Real.sin (30 * π / 180)
  let sin45 := Real.sin (45 * π / 180)
  let sin60 := Real.sin (60 * π / 180)
  let sin75 := Real.sin (75 * π / 180)
  let cos15 := Real.cos (15 * π / 180)
  let cos30 := Real.cos (30 * π / 180)
  (sin15 + sin30 + sin45 + sin60 + sin75) / (sin15 * cos15 * cos30) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2949_294980


namespace NUMINAMATH_CALUDE_rotation_180_transforms_rectangle_l2949_294916

-- Define the points of rectangle ABCD
def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (-1, 5)
def D : ℝ × ℝ := (-3, 5)

-- Define the points of rectangle A'B'C'D'
def A' : ℝ × ℝ := (3, -2)
def B' : ℝ × ℝ := (1, -2)
def C' : ℝ × ℝ := (1, -5)
def D' : ℝ × ℝ := (3, -5)

-- Define the 180° rotation transformation
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement
theorem rotation_180_transforms_rectangle :
  rotate180 A = A' ∧
  rotate180 B = B' ∧
  rotate180 C = C' ∧
  rotate180 D = D' := by
  sorry


end NUMINAMATH_CALUDE_rotation_180_transforms_rectangle_l2949_294916


namespace NUMINAMATH_CALUDE_box_volume_increase_l2949_294984

/-- Theorem about the volume of a rectangular box after increasing dimensions --/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 4500)
  (surface_area : 2 * (l * w + l * h + w * h) = 1800)
  (edge_sum : 4 * (l + w + h) = 216) :
  (l + 1) * (w + 1) * (h + 1) = 5455 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2949_294984


namespace NUMINAMATH_CALUDE_lucy_fish_total_l2949_294982

theorem lucy_fish_total (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 212 → additional = 68 → total = initial + additional → total = 280 := by
sorry

end NUMINAMATH_CALUDE_lucy_fish_total_l2949_294982


namespace NUMINAMATH_CALUDE_equation_solutions_l2949_294990

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 16 = 0 ↔ x = 4 ∨ x = -4) ∧
  (∀ x : ℝ, (x + 10)^3 + 27 = 0 ↔ x = -13) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2949_294990


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l2949_294978

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : p ≠ 1) (h2 : r ≠ 1) (h3 : p ≠ r) 
  (h4 : k ≠ 0) (h5 : k * p^3 - k * r^3 = 3 * (k * p - k * r)) :
  p + r = Real.sqrt 3 ∨ p + r = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l2949_294978


namespace NUMINAMATH_CALUDE_binomial_1000_1000_l2949_294935

theorem binomial_1000_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1000_1000_l2949_294935


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2949_294945

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) → (a > 3 ∨ a < -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2949_294945


namespace NUMINAMATH_CALUDE_triangle_side_length_l2949_294921

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a = 2 → b = 3 → C = Real.pi / 3 → c = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2949_294921


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2949_294948

def U : Finset ℕ := {0, 1, 2, 3, 4}
def M : Finset ℕ := {0, 1, 2}
def N : Finset ℕ := {2, 3}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2949_294948


namespace NUMINAMATH_CALUDE_chemical_solution_mixing_l2949_294932

theorem chemical_solution_mixing (initial_concentration : ℝ) 
  (replacement_concentration : ℝ) (replaced_portion : ℝ) 
  (resulting_concentration : ℝ) : 
  initial_concentration = 0.85 →
  replacement_concentration = 0.20 →
  replaced_portion = 0.6923076923076923 →
  resulting_concentration = 
    (initial_concentration * (1 - replaced_portion) + 
     replacement_concentration * replaced_portion) →
  resulting_concentration = 0.40 := by
sorry

end NUMINAMATH_CALUDE_chemical_solution_mixing_l2949_294932


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_one_l2949_294986

-- Define the points
def A : ℝ × ℝ := (0, -3)
def B : ℝ × ℝ := (3, 3)
def C : ℝ → ℝ × ℝ := λ x ↦ (x, -1)

-- Define the vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC (x : ℝ) : ℝ × ℝ := ((C x).1 - A.1, (C x).2 - A.2)

-- Theorem statement
theorem parallel_vectors_imply_x_equals_one :
  ∀ x : ℝ, (∃ k : ℝ, AB = k • (AC x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_one_l2949_294986


namespace NUMINAMATH_CALUDE_circle_tangent_vector_theorem_l2949_294963

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the point A
def A : ℝ × ℝ := (3, 4)

-- Define the vector equation
def VectorEquation (P M N : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), A.1 = x * M.1 + y * N.1 ∧ A.2 = x * M.2 + y * N.2

-- Define the trajectory equation
def TrajectoryEquation (P : ℝ × ℝ) : Prop :=
  P.1 ≠ 0 ∧ P.2 ≠ 0 ∧ P.1^2 / 16 + P.2^2 / 9 = (P.1 + P.2 - 1)^2

theorem circle_tangent_vector_theorem :
  ∀ (P M N : ℝ × ℝ),
    P ∈ Circle (0, 0) 1 →
    VectorEquation P M N →
    TrajectoryEquation P ∧ (∀ x y : ℝ, 9 * x^2 + 16 * y^2 ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_vector_theorem_l2949_294963


namespace NUMINAMATH_CALUDE_least_number_with_divisibility_property_l2949_294965

def is_divisible (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

def is_least_with_property (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a + 1 = b ∧ b ≤ 20 ∧
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 20 ∧ k ≠ a ∧ k ≠ b → is_divisible n k) ∧
  (∀ m : ℕ, m < n → ¬∃ (c d : ℕ), c + 1 = d ∧ d ≤ 20 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 20 ∧ k ≠ c ∧ k ≠ d → is_divisible m k))

theorem least_number_with_divisibility_property :
  is_least_with_property 12252240 := by sorry

end NUMINAMATH_CALUDE_least_number_with_divisibility_property_l2949_294965


namespace NUMINAMATH_CALUDE_diagonals_of_120_degree_polygon_l2949_294903

/-- The number of diagonals in a regular polygon with 120° interior angles is 9. -/
theorem diagonals_of_120_degree_polygon : ∃ (n : ℕ), 
  (∀ (i : ℕ), i < n → (180 * (n - 2) : ℝ) / n = 120) → 
  (n * (n - 3) : ℝ) / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_of_120_degree_polygon_l2949_294903


namespace NUMINAMATH_CALUDE_max_value_xyz_l2949_294937

theorem max_value_xyz (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 3 * x + 2 * y + 6 * z = 1) :
  x^4 * y^3 * z^2 ≤ 1 / 372008 :=
by sorry

end NUMINAMATH_CALUDE_max_value_xyz_l2949_294937


namespace NUMINAMATH_CALUDE_system_solution_l2949_294988

theorem system_solution (x y : ℝ) : x = 1 ∧ y = -2 → x + y = -1 ∧ x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2949_294988


namespace NUMINAMATH_CALUDE_rabbit_carrot_problem_l2949_294995

theorem rabbit_carrot_problem (initial_carrots : ℕ) : 
  (((initial_carrots * 3 - 30) * 3 - 30) * 3 - 30) * 3 - 30 = 0 → 
  initial_carrots = 15 := by
sorry

end NUMINAMATH_CALUDE_rabbit_carrot_problem_l2949_294995


namespace NUMINAMATH_CALUDE_product_magnitude_l2949_294924

theorem product_magnitude (z₁ z₂ : ℂ) (h1 : Complex.abs z₁ = 3) (h2 : z₂ = Complex.mk 2 1) :
  Complex.abs (z₁ * z₂) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_product_magnitude_l2949_294924


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2949_294900

/-- Definition of a quadratic equation in standard form -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l2949_294900


namespace NUMINAMATH_CALUDE_total_travel_time_is_156_hours_l2949_294906

/-- Represents the total travel time of a car journey with specific conditions. -/
def total_travel_time (time_ngapara_zipra : ℝ) : ℝ :=
  let time_ningi_zipra : ℝ := 0.8 * time_ngapara_zipra
  let time_zipra_varnasi : ℝ := 0.75 * time_ningi_zipra
  let delay_time : ℝ := 0.25 * time_ningi_zipra
  time_ngapara_zipra + time_ningi_zipra + delay_time + time_zipra_varnasi

/-- Theorem stating that the total travel time is 156 hours given the specified conditions. -/
theorem total_travel_time_is_156_hours :
  total_travel_time 60 = 156 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_is_156_hours_l2949_294906


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l2949_294967

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 13 ∣ n → n ≥ 104 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l2949_294967


namespace NUMINAMATH_CALUDE_semicircle_in_rectangle_radius_l2949_294904

/-- Given a rectangle with a semi-circle inscribed, prove that the radius is 27 cm -/
theorem semicircle_in_rectangle_radius (L W r : ℝ) : 
  L > 0 → W > 0 → r > 0 →
  2 * L + 2 * W = 216 → -- Perimeter of rectangle is 216 cm
  W = 2 * r → -- Width is twice the radius
  L = 2 * r → -- Length is diameter (twice the radius)
  r = 27 := by
sorry

end NUMINAMATH_CALUDE_semicircle_in_rectangle_radius_l2949_294904


namespace NUMINAMATH_CALUDE_sum_of_specific_sequence_l2949_294998

def arithmetic_sequence_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem sum_of_specific_sequence :
  arithmetic_sequence_sum 102 492 10 = 11880 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_sequence_l2949_294998


namespace NUMINAMATH_CALUDE_no_a_in_either_subject_l2949_294911

theorem no_a_in_either_subject (total_students : ℕ) (physics_a : ℕ) (chemistry_a : ℕ) (both_a : ℕ)
  (h1 : total_students = 40)
  (h2 : physics_a = 10)
  (h3 : chemistry_a = 18)
  (h4 : both_a = 6) :
  total_students - (physics_a + chemistry_a - both_a) = 18 :=
by sorry

end NUMINAMATH_CALUDE_no_a_in_either_subject_l2949_294911


namespace NUMINAMATH_CALUDE_complex_argument_of_two_plus_two_i_sqrt_three_l2949_294970

/-- For the complex number z = 2 + 2i√3, when expressed in the form re^(iθ), θ = π/3 -/
theorem complex_argument_of_two_plus_two_i_sqrt_three :
  let z : ℂ := 2 + 2 * Complex.I * Real.sqrt 3
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_complex_argument_of_two_plus_two_i_sqrt_three_l2949_294970


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l2949_294910

theorem exam_maximum_marks :
  let pass_percentage : ℚ := 90 / 100
  let student_marks : ℕ := 250
  let failing_margin : ℕ := 300
  let maximum_marks : ℕ := 612
  (pass_percentage * maximum_marks : ℚ) = (student_marks + failing_margin : ℚ) ∧
  maximum_marks = (student_marks + failing_margin : ℚ) / pass_percentage :=
by sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l2949_294910


namespace NUMINAMATH_CALUDE_min_value_of_f_l2949_294901

theorem min_value_of_f (x : ℝ) (h : x > 0) : 
  let f := fun x => 1 / x^2 + 2 * x
  (∀ y > 0, f y ≥ 3) ∧ (∃ z > 0, f z = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2949_294901


namespace NUMINAMATH_CALUDE_expression_simplification_l2949_294959

theorem expression_simplification (x : ℝ) :
  x = (1/2)⁻¹ + (π - 1)^0 →
  ((x - 3) / (x^2 - 1) - 2 / (x + 1)) / (x / (x^2 - 2*x + 1)) = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2949_294959


namespace NUMINAMATH_CALUDE_quadratic_roots_uniqueness_l2949_294947

/-- Given two quadratic polynomials with specific root relationships, 
    prove that there is only one set of values for the roots and coefficients. -/
theorem quadratic_roots_uniqueness (p q u v : ℝ) : 
  p ≠ 0 ∧ q ≠ 0 ∧ u ≠ 0 ∧ v ≠ 0 ∧  -- non-zero roots
  p ≠ q ∧ u ≠ v ∧  -- distinct roots
  (∀ x, x^2 + u*x - v = (x - p)*(x - q)) ∧  -- first polynomial
  (∀ x, x^2 + p*x - q = (x - u)*(x - v)) →  -- second polynomial
  p = -1 ∧ q = 2 ∧ u = -1 ∧ v = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_uniqueness_l2949_294947


namespace NUMINAMATH_CALUDE_town_population_increase_l2949_294918

/-- Calculates the average percent increase of population per year given initial and final populations over a decade. -/
def average_percent_increase (initial_population final_population : ℕ) : ℚ :=
  let total_increase : ℕ := final_population - initial_population
  let average_annual_increase : ℚ := total_increase / 10
  (average_annual_increase / initial_population) * 100

/-- Theorem stating that the average percent increase of population per year is 7% for the given conditions. -/
theorem town_population_increase :
  average_percent_increase 175000 297500 = 7 := by
  sorry

#eval average_percent_increase 175000 297500

end NUMINAMATH_CALUDE_town_population_increase_l2949_294918


namespace NUMINAMATH_CALUDE_rental_ratio_proof_l2949_294938

/-- Represents the ratio of dramas to action movies rented during a two-week period -/
def drama_action_ratio : ℚ := 37 / 8

/-- Theorem stating the ratio of dramas to action movies given the rental conditions -/
theorem rental_ratio_proof (T : ℝ) (a : ℝ) (h1 : T > 0) (h2 : a > 0) : 
  (0.64 * T = 10 * a) →  -- Condition: 64% of rentals are comedies, and comedies = 10a
  (∃ d : ℝ, d > 0 ∧ 0.36 * T = a + d) →  -- Condition: Remaining 36% are dramas and action movies
  (∃ s : ℝ, s > 0 ∧ ∃ d : ℝ, d = s * a) →  -- Condition: Dramas are some times action movies
  drama_action_ratio = 37 / 8 :=
sorry

end NUMINAMATH_CALUDE_rental_ratio_proof_l2949_294938


namespace NUMINAMATH_CALUDE_john_profit_calculation_l2949_294939

/-- Calculates John's profit from selling newspapers, magazines, and books --/
theorem john_profit_calculation :
  let newspaper_count : ℕ := 500
  let magazine_count : ℕ := 300
  let book_count : ℕ := 200
  let newspaper_price : ℚ := 2
  let magazine_price : ℚ := 4
  let book_price : ℚ := 10
  let newspaper_sold_ratio : ℚ := 0.80
  let magazine_sold_ratio : ℚ := 0.75
  let book_sold_ratio : ℚ := 0.60
  let newspaper_discount : ℚ := 0.75
  let magazine_discount : ℚ := 0.60
  let book_discount : ℚ := 0.45
  let tax_rate : ℚ := 0.08
  let shipping_fee : ℚ := 25
  let commission_rate : ℚ := 0.05

  let newspaper_cost := newspaper_price * (1 - newspaper_discount)
  let magazine_cost := magazine_price * (1 - magazine_discount)
  let book_cost := book_price * (1 - book_discount)

  let total_cost_before_tax := 
    newspaper_count * newspaper_cost +
    magazine_count * magazine_cost +
    book_count * book_cost

  let total_cost_after_tax_and_shipping :=
    total_cost_before_tax * (1 + tax_rate) + shipping_fee

  let total_revenue_before_commission :=
    newspaper_count * newspaper_sold_ratio * newspaper_price +
    magazine_count * magazine_sold_ratio * magazine_price +
    book_count * book_sold_ratio * book_price

  let total_revenue_after_commission :=
    total_revenue_before_commission * (1 - commission_rate)

  let profit := total_revenue_after_commission - total_cost_after_tax_and_shipping

  profit = 753.60 := by sorry

end NUMINAMATH_CALUDE_john_profit_calculation_l2949_294939


namespace NUMINAMATH_CALUDE_chess_tournament_players_l2949_294922

theorem chess_tournament_players (total_games : ℕ) (h1 : total_games = 30) : ∃ n : ℕ, n > 0 ∧ total_games = n * (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l2949_294922


namespace NUMINAMATH_CALUDE_sum_f_2016_2017_2018_l2949_294941

/-- An odd periodic function with period 4 and f(1) = 1 -/
def f : ℝ → ℝ :=
  sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f has period 4 -/
axiom f_periodic : ∀ x, f (x + 4) = f x

/-- f(1) = 1 -/
axiom f_one : f 1 = 1

theorem sum_f_2016_2017_2018 : f 2016 + f 2017 + f 2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_2016_2017_2018_l2949_294941


namespace NUMINAMATH_CALUDE_v2_equals_22_at_neg_4_l2949_294976

/-- Horner's Rule for a specific polynomial -/
def horner_polynomial (x : ℝ) : ℝ := 
  ((((x * x + 6) * x + 9) * x + 0) * x + 0) * x + 208

/-- v2 calculation in Horner's Rule -/
def v2 (x : ℝ) : ℝ := 
  (1 * x * x) + 6

/-- Theorem: v2 equals 22 when x = -4 for the given polynomial -/
theorem v2_equals_22_at_neg_4 : 
  v2 (-4) = 22 := by sorry

end NUMINAMATH_CALUDE_v2_equals_22_at_neg_4_l2949_294976


namespace NUMINAMATH_CALUDE_ellipse_other_intersection_l2949_294920

/-- Define an ellipse with foci at (0,0) and (4,0) that intersects the x-axis at (1,0) -/
def ellipse (x : ℝ) : Prop :=
  (|x| + |x - 4|) = 4

/-- The other point of intersection of the ellipse with the x-axis -/
def other_intersection : ℝ := 4

/-- Theorem stating that the other point of intersection is (4,0) -/
theorem ellipse_other_intersection :
  ellipse other_intersection ∧ other_intersection ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_other_intersection_l2949_294920


namespace NUMINAMATH_CALUDE_inequality_proof_l2949_294951

theorem inequality_proof (x y : ℝ) (h1 : x > y) (h2 : x * y = 1) :
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2949_294951


namespace NUMINAMATH_CALUDE_ellipse_point_distance_l2949_294997

/-- The ellipse with equation x²/9 + y²/6 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) + (p.2^2 / 6) = 1}

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_point_distance :
  P ∈ Ellipse →
  angle F₁ P F₂ = Real.arccos (3/5) →
  distance P O = Real.sqrt 30 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_point_distance_l2949_294997


namespace NUMINAMATH_CALUDE_evaluate_expression_l2949_294987

theorem evaluate_expression : (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2949_294987


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l2949_294989

theorem cow_husk_consumption 
  (cows bags days : ℕ) 
  (h : cows = 45 ∧ bags = 45 ∧ days = 45) : 
  (1 : ℕ) * days = 45 := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l2949_294989


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_3_l2949_294957

/-- The function f(x) = x^2 + 4ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4*a*x + 2

/-- The theorem stating that if f(x) is monotonically decreasing in (-∞, 6), then a ≤ 3 -/
theorem monotone_decreasing_implies_a_leq_3 (a : ℝ) :
  (∀ x y, x < y → y < 6 → f a x > f a y) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_3_l2949_294957


namespace NUMINAMATH_CALUDE_arthur_muffins_arthur_muffins_proof_l2949_294926

theorem arthur_muffins : ℕ → Prop :=
  fun initial_muffins =>
    initial_muffins + 48 = 83 → initial_muffins = 35

-- Proof
theorem arthur_muffins_proof : arthur_muffins 35 := by
  sorry

end NUMINAMATH_CALUDE_arthur_muffins_arthur_muffins_proof_l2949_294926


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2949_294994

/-- Proves that given a parabola y^2 = 8x whose latus rectum passes through a focus of a hyperbola
    x^2/a^2 - y^2/b^2 = 1 (a > 0, b > 0), and one asymptote of the hyperbola is x + √3y = 0,
    the equation of the hyperbola is x^2/3 - y^2 = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y^2 = 8*x ∧ x = -2) →  -- Latus rectum of parabola
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  (∃ (x y : ℝ), x + Real.sqrt 3 * y = 0) →  -- Asymptote equation
  (∀ (x y : ℝ), x^2/3 - y^2 = 1) :=  -- Resulting hyperbola equation
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l2949_294994


namespace NUMINAMATH_CALUDE_cara_friends_photo_l2949_294927

theorem cara_friends_photo (n : ℕ) (k : ℕ) : n = 7 → k = 2 → Nat.choose n k = 21 := by
  sorry

end NUMINAMATH_CALUDE_cara_friends_photo_l2949_294927


namespace NUMINAMATH_CALUDE_congruence_solution_l2949_294991

theorem congruence_solution (n : ℕ) : n ∈ Finset.range 47 ∧ 13 * n ≡ 9 [MOD 47] ↔ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2949_294991
