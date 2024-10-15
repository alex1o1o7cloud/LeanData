import Mathlib

namespace NUMINAMATH_CALUDE_batsman_highest_score_l1226_122606

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46) 
  (h1 : average = 60) 
  (h2 : score_difference = 190) 
  (h3 : average_excluding_extremes = 58) : 
  ∃ (highest_score lowest_score : ℕ), 
    highest_score - lowest_score = score_difference ∧ 
    (total_innings : ℚ) * average = (total_innings - 2 : ℚ) * average_excluding_extremes + highest_score + lowest_score ∧
    highest_score = 199 :=
by sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l1226_122606


namespace NUMINAMATH_CALUDE_original_rectangle_perimeter_l1226_122696

/-- Given a rectangle with sides a and b, prove that if it's cut diagonally
    and then one piece is cut parallel to its shorter sides at the midpoints,
    resulting in a rectangle with perimeter 129 cm, then the perimeter of the
    original rectangle was 258 cm. -/
theorem original_rectangle_perimeter
  (a b : ℝ) 
  (h_positive : a > 0 ∧ b > 0)
  (h_final_perimeter : 2 * (a / 2 + b / 2) = 129) :
  2 * (a + b) = 258 :=
sorry

end NUMINAMATH_CALUDE_original_rectangle_perimeter_l1226_122696


namespace NUMINAMATH_CALUDE_sum_has_even_digit_l1226_122649

def reverse_number (n : List Nat) : List Nat :=
  n.reverse

def sum_digits (n m : List Nat) : List Nat :=
  sorry

theorem sum_has_even_digit (n : List Nat) (h : n.length = 17) :
  ∃ (d : Nat), d ∈ sum_digits n (reverse_number n) ∧ Even d :=
sorry

end NUMINAMATH_CALUDE_sum_has_even_digit_l1226_122649


namespace NUMINAMATH_CALUDE_brown_eyes_light_brown_skin_l1226_122676

/-- Represents the characteristics of the group of girls -/
structure GirlGroup where
  total : Nat
  blue_eyes_fair_skin : Nat
  light_brown_skin : Nat
  brown_eyes : Nat

/-- Theorem stating the number of girls with brown eyes and light brown skin -/
theorem brown_eyes_light_brown_skin (g : GirlGroup) 
  (h1 : g.total = 50)
  (h2 : g.blue_eyes_fair_skin = 14)
  (h3 : g.light_brown_skin = 31)
  (h4 : g.brown_eyes = 18) :
  g.brown_eyes - (g.total - g.light_brown_skin - g.blue_eyes_fair_skin) = 13 := by
  sorry

#check brown_eyes_light_brown_skin

end NUMINAMATH_CALUDE_brown_eyes_light_brown_skin_l1226_122676


namespace NUMINAMATH_CALUDE_angle_triple_complement_l1226_122672

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l1226_122672


namespace NUMINAMATH_CALUDE_income_increase_is_fifty_percent_l1226_122647

/-- Represents the financial situation of a person over two years -/
structure FinancialData where
  income1 : ℝ
  savingsRate1 : ℝ
  incomeIncrease : ℝ

/-- The conditions of the problem -/
def problemConditions (d : FinancialData) : Prop :=
  d.savingsRate1 = 0.5 ∧
  d.income1 > 0 ∧
  d.incomeIncrease > 0 ∧
  let savings1 := d.savingsRate1 * d.income1
  let expenditure1 := d.income1 - savings1
  let income2 := d.income1 * (1 + d.incomeIncrease)
  let savings2 := 2 * savings1
  let expenditure2 := income2 - savings2
  expenditure1 + expenditure2 = 2 * expenditure1

/-- The theorem stating that under the given conditions, 
    the income increase in the second year is 50% -/
theorem income_increase_is_fifty_percent (d : FinancialData) :
  problemConditions d → d.incomeIncrease = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_income_increase_is_fifty_percent_l1226_122647


namespace NUMINAMATH_CALUDE_l_shape_area_l1226_122678

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The large rectangle -/
def large_rectangle : Rectangle := { length := 10, width := 6 }

/-- The small rectangle to be subtracted -/
def small_rectangle : Rectangle := { length := 4, width := 3 }

/-- The number of small rectangles to be subtracted -/
def num_small_rectangles : ℕ := 2

/-- Theorem: The area of the L-shape is 36 square units -/
theorem l_shape_area : 
  area large_rectangle - num_small_rectangles * area small_rectangle = 36 := by
  sorry

end NUMINAMATH_CALUDE_l_shape_area_l1226_122678


namespace NUMINAMATH_CALUDE_sarah_marriage_age_l1226_122636

def game_prediction (name_length : ℕ) (current_age : ℕ) : ℕ :=
  name_length + 2 * current_age

theorem sarah_marriage_age :
  game_prediction 5 9 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sarah_marriage_age_l1226_122636


namespace NUMINAMATH_CALUDE_king_probability_l1226_122623

/-- Custom deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (ranks : Nat)
  (one_card_per_rank_suit : cards = suits * ranks)

/-- Probability of drawing a specific rank -/
def prob_draw_rank (d : Deck) (rank_count : Nat) : ℚ :=
  rank_count / d.cards

theorem king_probability (d : Deck) (h1 : d.cards = 65) (h2 : d.suits = 5) (h3 : d.ranks = 13) :
  prob_draw_rank d d.suits = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_king_probability_l1226_122623


namespace NUMINAMATH_CALUDE_double_march_earnings_cars_l1226_122613

/-- Represents the earnings of a car salesman -/
structure CarSalesmanEarnings where
  baseSalary : ℕ
  commissionPerCar : ℕ
  marchEarnings : ℕ

/-- Calculates the number of cars needed to be sold to reach a target earning -/
def carsNeededForTarget (e : CarSalesmanEarnings) (targetEarnings : ℕ) : ℕ :=
  ((targetEarnings - e.baseSalary) + e.commissionPerCar - 1) / e.commissionPerCar

/-- Theorem: The number of cars needed to double March earnings is 15 -/
theorem double_march_earnings_cars (e : CarSalesmanEarnings) 
    (h1 : e.baseSalary = 1000)
    (h2 : e.commissionPerCar = 200)
    (h3 : e.marchEarnings = 2000) : 
    carsNeededForTarget e (2 * e.marchEarnings) = 15 := by
  sorry

end NUMINAMATH_CALUDE_double_march_earnings_cars_l1226_122613


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l1226_122643

/-- Distance between centers of two pulleys -/
theorem pulley_centers_distance 
  (r1 : ℝ) (r2 : ℝ) (contact_distance : ℝ)
  (h1 : r1 = 10)
  (h2 : r2 = 6)
  (h3 : contact_distance = 30) :
  ∃ (center_distance : ℝ), 
    center_distance = 2 * Real.sqrt 229 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l1226_122643


namespace NUMINAMATH_CALUDE_equation_roots_l1226_122658

theorem equation_roots : 
  {x : ℝ | Real.sqrt (x^2) + 3 * x⁻¹ = 4} = {3, -3, 1, -1} :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l1226_122658


namespace NUMINAMATH_CALUDE_extreme_value_condition_l1226_122683

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- State the theorem
theorem extreme_value_condition (a b : ℝ) :
  (f a b 1 = 4) ∧ (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1) →
  a * b = -27 ∨ a * b = -2 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l1226_122683


namespace NUMINAMATH_CALUDE_correct_statements_count_l1226_122653

-- Define a structure to represent a statement
structure GeometricStatement :=
  (id : Nat)
  (content : String)
  (isCorrect : Bool)

-- Define the four statements
def statement1 : GeometricStatement :=
  { id := 1
  , content := "The prism with the least number of faces has 6 vertices"
  , isCorrect := true }

def statement2 : GeometricStatement :=
  { id := 2
  , content := "A frustum is the middle part of a cone cut by two parallel planes"
  , isCorrect := false }

def statement3 : GeometricStatement :=
  { id := 3
  , content := "A plane passing through the vertex of a cone cuts the cone into a section that is an isosceles triangle"
  , isCorrect := true }

def statement4 : GeometricStatement :=
  { id := 4
  , content := "Equal angles remain equal in perspective drawings"
  , isCorrect := false }

-- Define the list of all statements
def allStatements : List GeometricStatement :=
  [statement1, statement2, statement3, statement4]

-- Theorem to prove
theorem correct_statements_count :
  (allStatements.filter (·.isCorrect)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l1226_122653


namespace NUMINAMATH_CALUDE_both_normal_l1226_122601

-- Define a type for people
inductive Person : Type
| MrA : Person
| MrsA : Person

-- Define what it means to be normal
def normal (p : Person) : Prop := True

-- Define the statement made by each person
def statement (p : Person) : Prop :=
  match p with
  | Person.MrA => normal Person.MrsA
  | Person.MrsA => normal Person.MrA

-- Theorem: There exists a consistent interpretation where both are normal
theorem both_normal :
  ∃ (interp : Person → Prop),
    (∀ p, interp p ↔ normal p) ∧
    (∀ p, interp p → statement p) :=
sorry

end NUMINAMATH_CALUDE_both_normal_l1226_122601


namespace NUMINAMATH_CALUDE_shifted_line_not_in_third_quadrant_l1226_122677

/-- The original line equation -/
def original_line (x : ℝ) : ℝ := -2 * x - 1

/-- The shifted line equation -/
def shifted_line (x : ℝ) : ℝ := -2 * x + 5

/-- The shift amount -/
def shift : ℝ := 3

/-- Theorem: The shifted line does not intersect the third quadrant -/
theorem shifted_line_not_in_third_quadrant :
  ∀ x y : ℝ, y = shifted_line x → ¬(x < 0 ∧ y < 0) :=
sorry

end NUMINAMATH_CALUDE_shifted_line_not_in_third_quadrant_l1226_122677


namespace NUMINAMATH_CALUDE_perimeter_of_triangle_from_unit_square_l1226_122639

/-- Represents a triangle formed from a unit square --/
structure TriangleFromUnitSquare where
  /-- The base of the isosceles triangle --/
  base : ℝ
  /-- One leg of the isosceles triangle --/
  leg : ℝ
  /-- The triangle is isosceles --/
  isIsosceles : base = 2 * leg
  /-- The base is formed by two sides of the unit square --/
  baseFromSquare : base = Real.sqrt 2
  /-- Each leg is formed by one side of the unit square --/
  legFromSquare : leg = Real.sqrt 2 / 2

/-- The perimeter of the triangle formed from a unit square is 2√2 --/
theorem perimeter_of_triangle_from_unit_square (t : TriangleFromUnitSquare) :
  t.base + 2 * t.leg = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_triangle_from_unit_square_l1226_122639


namespace NUMINAMATH_CALUDE_little_red_journey_l1226_122695

/-- The distance from Little Red's house to school in kilometers -/
def distance_to_school : ℝ := 1.5

/-- Little Red's average speed uphill in kilometers per hour -/
def speed_uphill : ℝ := 2

/-- Little Red's average speed downhill in kilometers per hour -/
def speed_downhill : ℝ := 3

/-- The total time taken for the journey in minutes -/
def total_time : ℝ := 18

/-- The system of equations describing Little Red's journey to school -/
def journey_equations (x y : ℝ) : Prop :=
  (speed_uphill / 60 * x + speed_downhill / 60 * y = distance_to_school) ∧
  (x + y = total_time)

theorem little_red_journey :
  ∀ x y : ℝ, journey_equations x y ↔
    (2 / 60 * x + 3 / 60 * y = 1.5) ∧ (x + y = 18) :=
sorry

end NUMINAMATH_CALUDE_little_red_journey_l1226_122695


namespace NUMINAMATH_CALUDE_blue_cube_problem_l1226_122645

theorem blue_cube_problem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 :=
by sorry

end NUMINAMATH_CALUDE_blue_cube_problem_l1226_122645


namespace NUMINAMATH_CALUDE_phones_sold_is_four_l1226_122616

/-- Calculates the total number of cell phones sold given the initial and final counts
    of Samsung phones and iPhones, as well as the number of damaged/defective phones. -/
def total_phones_sold (initial_samsung : ℕ) (final_samsung : ℕ) (initial_iphone : ℕ) 
                      (final_iphone : ℕ) (damaged_samsung : ℕ) (defective_iphone : ℕ) : ℕ :=
  (initial_samsung - final_samsung - damaged_samsung) + 
  (initial_iphone - final_iphone - defective_iphone)

/-- Theorem stating that the total number of cell phones sold is 4 given the specific
    initial and final counts, and the number of damaged/defective phones. -/
theorem phones_sold_is_four : 
  total_phones_sold 14 10 8 5 2 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_phones_sold_is_four_l1226_122616


namespace NUMINAMATH_CALUDE_expression_equals_sum_l1226_122663

theorem expression_equals_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let numerator := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let denominator := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  numerator / denominator = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_sum_l1226_122663


namespace NUMINAMATH_CALUDE_fair_distribution_theorem_l1226_122681

/-- Represents the outcome of a chess game -/
inductive GameOutcome
  | AWin
  | BWin

/-- Represents the state of the chess competition -/
structure ChessCompetition where
  total_games : Nat
  games_played : Nat
  a_wins : Nat
  b_wins : Nat
  prize_money : Nat

/-- Calculates the probability of player A winning the competition -/
def probability_a_wins (comp : ChessCompetition) : Rat :=
  sorry

/-- Calculates the fair distribution of prize money -/
def fair_distribution (comp : ChessCompetition) : Nat × Nat :=
  sorry

/-- Theorem stating the fair distribution of prize money -/
theorem fair_distribution_theorem (comp : ChessCompetition) 
  (h1 : comp.total_games = 7)
  (h2 : comp.games_played = 5)
  (h3 : comp.a_wins = 3)
  (h4 : comp.b_wins = 2)
  (h5 : comp.prize_money = 10000) :
  fair_distribution comp = (7500, 2500) :=
sorry

end NUMINAMATH_CALUDE_fair_distribution_theorem_l1226_122681


namespace NUMINAMATH_CALUDE_dumbbell_weight_problem_l1226_122665

theorem dumbbell_weight_problem (total_weight : ℝ) (first_pair_weight : ℝ) (third_pair_weight : ℝ) 
  (h1 : total_weight = 32)
  (h2 : first_pair_weight = 3)
  (h3 : third_pair_weight = 8) :
  total_weight - 2 * first_pair_weight - 2 * third_pair_weight = 10 := by
  sorry

end NUMINAMATH_CALUDE_dumbbell_weight_problem_l1226_122665


namespace NUMINAMATH_CALUDE_power_difference_l1226_122642

theorem power_difference (m n : ℕ) (h1 : 2^m = 32) (h2 : 3^n = 81) : 5^(m-n) = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l1226_122642


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l1226_122622

theorem parallel_lines_condition (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : a₁^2 + b₁^2 ≠ 0) (h₂ : a₂^2 + b₂^2 ≠ 0) :
  ¬(∀ (x y : ℝ), (a₁*x + b₁*y + c₁ = 0 ↔ a₂*x + b₂*y + c₂ = 0) ↔ 
    (a₁*b₂ - a₂*b₁ ≠ 0)) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l1226_122622


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l1226_122609

theorem magnitude_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.abs ((1 - i) / (2 * i + 1)) = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l1226_122609


namespace NUMINAMATH_CALUDE_chocolates_on_square_perimeter_l1226_122648

/-- The number of chocolates on one side of the square -/
def chocolates_per_side : ℕ := 6

/-- The number of sides in a square -/
def sides_of_square : ℕ := 4

/-- The number of corners in a square -/
def corners_of_square : ℕ := 4

/-- The total number of chocolates around the perimeter of the square -/
def chocolates_on_perimeter : ℕ := chocolates_per_side * sides_of_square - corners_of_square

theorem chocolates_on_square_perimeter : chocolates_on_perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_on_square_perimeter_l1226_122648


namespace NUMINAMATH_CALUDE_race_distance_l1226_122646

theorem race_distance (total_length : Real) (part1 : Real) (part2 : Real) (part3 : Real)
  (h1 : total_length = 74.5)
  (h2 : part1 = 15.5)
  (h3 : part2 = 21.5)
  (h4 : part3 = 21.5) :
  total_length - (part1 + part2 + part3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l1226_122646


namespace NUMINAMATH_CALUDE_projection_closed_l1226_122635

open Set
open Topology

-- Define the projection function
def proj_y (p : ℝ × ℝ) : ℝ := p.2

-- State the theorem
theorem projection_closed {a b : ℝ} {S : Set (ℝ × ℝ)} 
  (hS : IsClosed S) 
  (hSub : S ⊆ {p : ℝ × ℝ | a < p.1 ∧ p.1 < b}) :
  IsClosed (proj_y '' S) := by
  sorry

end NUMINAMATH_CALUDE_projection_closed_l1226_122635


namespace NUMINAMATH_CALUDE_ellipse_b_plus_k_l1226_122657

/-- Definition of an ellipse with given foci and a point on the curve -/
def Ellipse (f1 f2 p : ℝ × ℝ) :=
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
    (p.1 - h)^2 / a^2 + (p.2 - k)^2 / b^2 = 1 ∧
    (f1.1 - h)^2 / a^2 + (f1.2 - k)^2 / b^2 = 1 ∧
    (f2.1 - h)^2 / a^2 + (f2.2 - k)^2 / b^2 = 1

/-- Theorem stating the sum of b and k for the given ellipse -/
theorem ellipse_b_plus_k :
  ∀ (a b h k : ℝ),
    Ellipse (2, 3) (2, 7) (6, 5) →
    a > 0 →
    b > 0 →
    (6 - h)^2 / a^2 + (5 - k)^2 / b^2 = 1 →
    b + k = 4 * Real.sqrt 5 + 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_b_plus_k_l1226_122657


namespace NUMINAMATH_CALUDE_sqrt2_similarity_l1226_122631

-- Define similarity for quadratic surds
def similar_quadratic_surds (a b : ℝ) : Prop :=
  ∃ (r : ℚ), r ≠ 0 ∧ a = r * b

-- Theorem statement
theorem sqrt2_similarity (r : ℚ) (h : r ≠ 0) :
  similar_quadratic_surds (r * Real.sqrt 2) (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_sqrt2_similarity_l1226_122631


namespace NUMINAMATH_CALUDE_point_in_intersection_l1226_122605

def U : Set (ℝ × ℝ) := Set.univ

def A (m : ℝ) : Set (ℝ × ℝ) := U \ {p : ℝ × ℝ | p.1 + p.2 > m}

def B (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ n}

def C_U (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := U \ S

theorem point_in_intersection (m n : ℝ) :
  (1, 2) ∈ (C_U (A m) ∩ B n) ↔ m ≥ 3 ∧ n ≥ 5 := by sorry

end NUMINAMATH_CALUDE_point_in_intersection_l1226_122605


namespace NUMINAMATH_CALUDE_cubic_polynomial_value_l1226_122644

/-- The given polynomial h -/
def h (x : ℝ) : ℝ := x^3 - x^2 + x - 1

/-- The roots of h -/
def roots_h : Set ℝ := {x | h x = 0}

/-- The theorem statement -/
theorem cubic_polynomial_value (p : ℝ → ℝ) :
  (∃ a b c : ℝ, roots_h = {a, b, c}) →  -- h has three distinct roots
  (∀ x, x ∈ roots_h → x^3 ∈ {y | p y = 0}) →  -- roots of p are cubes of roots of h
  (∀ x, p (p x) = p (p (p x))) →  -- p is a cubic polynomial
  p 1 = 2 →  -- given condition
  p 8 = 1008 := by  -- conclusion to prove
sorry


end NUMINAMATH_CALUDE_cubic_polynomial_value_l1226_122644


namespace NUMINAMATH_CALUDE_birdseed_mix_l1226_122691

/-- Given two brands of birdseed and their composition, prove the percentage of sunflower in Brand A -/
theorem birdseed_mix (x : ℝ) : 
  (0.4 + x / 100 = 1) →  -- Brand A composition
  (0.65 + 0.35 = 1) →  -- Brand B composition
  (0.6 * x / 100 + 0.4 * 0.35 = 0.5) →  -- Mix composition
  x = 60 := by sorry

end NUMINAMATH_CALUDE_birdseed_mix_l1226_122691


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l1226_122630

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) →
  (a ≥ 1/7 ∧ a < 1/3) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l1226_122630


namespace NUMINAMATH_CALUDE_unique_K_value_l1226_122632

theorem unique_K_value : ∃! K : ℕ, 
  (∃ Z : ℕ, 1000 < Z ∧ Z < 8000 ∧ K > 2 ∧ Z = K * K^2) ∧ 
  (∃ a b : ℕ, K^3 = a^2 ∧ K^3 = b^3) ∧
  K = 16 :=
sorry

end NUMINAMATH_CALUDE_unique_K_value_l1226_122632


namespace NUMINAMATH_CALUDE_lg_graph_property_l1226_122692

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define what it means for a point to be on the graph of y = lg x
def on_lg_graph (p : ℝ × ℝ) : Prop := p.2 = lg p.1

-- Theorem statement
theorem lg_graph_property (a b : ℝ) (h1 : on_lg_graph (a, b)) (h2 : a ≠ 1) :
  on_lg_graph (a^2, 2*b) :=
sorry

end NUMINAMATH_CALUDE_lg_graph_property_l1226_122692


namespace NUMINAMATH_CALUDE_base_7_units_digit_of_sum_l1226_122697

theorem base_7_units_digit_of_sum (a b : ℕ) (ha : a = 156) (hb : b = 97) :
  (a + b) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_7_units_digit_of_sum_l1226_122697


namespace NUMINAMATH_CALUDE_ellipse_on_y_axis_l1226_122626

/-- Given real numbers m and n where m > n > 0, the equation mx² + ny² = 1 represents an ellipse with foci on the y-axis -/
theorem ellipse_on_y_axis (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  (∃ (c : ℝ), c > 0 ∧ c^2 = a^2 - b^2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_on_y_axis_l1226_122626


namespace NUMINAMATH_CALUDE_cone_volume_ratio_l1226_122611

theorem cone_volume_ratio : 
  let r_C : ℝ := 20
  let h_C : ℝ := 50
  let r_D : ℝ := 25
  let h_D : ℝ := 40
  let V_C := (1/3) * π * r_C^2 * h_C
  let V_D := (1/3) * π * r_D^2 * h_D
  V_C / V_D = 4/5 := by sorry

end NUMINAMATH_CALUDE_cone_volume_ratio_l1226_122611


namespace NUMINAMATH_CALUDE_fish_sample_properties_l1226_122668

/-- Represents the mass categories of fish -/
inductive MassCategory
  | Mass1 : MassCategory
  | Mass2 : MassCategory
  | Mass3 : MassCategory
  | Mass4 : MassCategory

/-- Maps mass categories to their actual mass values -/
def massValue : MassCategory → Float
  | MassCategory.Mass1 => 1.0
  | MassCategory.Mass2 => 1.2
  | MassCategory.Mass3 => 1.5
  | MassCategory.Mass4 => 1.8

/-- Represents the frequency of each mass category -/
def frequency : MassCategory → Nat
  | MassCategory.Mass1 => 4
  | MassCategory.Mass2 => 5
  | MassCategory.Mass3 => 8
  | MassCategory.Mass4 => 3

/-- The total number of fish in the sample -/
def sampleSize : Nat := 20

/-- The number of marked fish recaptured -/
def markedRecaptured : Nat := 2

/-- The total number of fish recaptured -/
def totalRecaptured : Nat := 100

/-- Theorem stating the properties of the fish sample -/
theorem fish_sample_properties :
  (∃ median : Float, median = 1.5) ∧
  (∃ mean : Float, mean = 1.37) ∧
  (∃ totalMass : Float, totalMass = 1370) := by
  sorry

end NUMINAMATH_CALUDE_fish_sample_properties_l1226_122668


namespace NUMINAMATH_CALUDE_janes_blouses_l1226_122664

theorem janes_blouses (skirt_price : ℕ) (blouse_price : ℕ) (num_skirts : ℕ) (total_paid : ℕ) (change : ℕ) : 
  skirt_price = 13 →
  blouse_price = 6 →
  num_skirts = 2 →
  total_paid = 100 →
  change = 56 →
  (total_paid - change - (num_skirts * skirt_price)) / blouse_price = 3 :=
by sorry

end NUMINAMATH_CALUDE_janes_blouses_l1226_122664


namespace NUMINAMATH_CALUDE_youngest_not_first_or_last_l1226_122679

/-- The number of ways to arrange n people in a line -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line with one specific person at the start or end -/
def restrictedArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of people in the line -/
def n : ℕ := 5

theorem youngest_not_first_or_last :
  totalArrangements n - restrictedArrangements n = 72 := by
  sorry

#eval totalArrangements n - restrictedArrangements n

end NUMINAMATH_CALUDE_youngest_not_first_or_last_l1226_122679


namespace NUMINAMATH_CALUDE_two_p_plus_q_l1226_122617

theorem two_p_plus_q (p q : ℚ) (h : p / q = 3 / 5) : 2 * p + q = (11 / 5) * q := by
  sorry

end NUMINAMATH_CALUDE_two_p_plus_q_l1226_122617


namespace NUMINAMATH_CALUDE_latest_score_is_68_l1226_122660

def scores : List Int := [68, 75, 83, 94]

def is_integer_average (subset : List Int) : Prop :=
  subset.sum % subset.length = 0

theorem latest_score_is_68 :
  (∀ subset : List Int, subset ⊆ scores → is_integer_average subset) →
  scores.head? = some 68 :=
by sorry

end NUMINAMATH_CALUDE_latest_score_is_68_l1226_122660


namespace NUMINAMATH_CALUDE_area1_is_linear_area2_is_quadratic_l1226_122659

-- Define the rectangles
def rectangle1 (x : ℝ) : ℝ × ℝ := (10 - x, 5)
def rectangle2 (x : ℝ) : ℝ × ℝ := (30 + x, 20 + x)

-- Define the area functions
def area1 (x : ℝ) : ℝ := (rectangle1 x).1 * (rectangle1 x).2
def area2 (x : ℝ) : ℝ := (rectangle2 x).1 * (rectangle2 x).2

-- Theorem statements
theorem area1_is_linear : ∃ (m b : ℝ), ∀ x, area1 x = m * x + b := by sorry

theorem area2_is_quadratic : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, area2 x = a * x^2 + b * x + c) := by sorry

end NUMINAMATH_CALUDE_area1_is_linear_area2_is_quadratic_l1226_122659


namespace NUMINAMATH_CALUDE_derivative_log2_l1226_122602

open Real

theorem derivative_log2 (x : ℝ) (h : x > 0) :
  deriv (fun x => log x / log 2) x = 1 / (x * log 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_log2_l1226_122602


namespace NUMINAMATH_CALUDE_milk_volume_calculation_l1226_122669

def milk_volumes : List ℝ := [2.35, 1.75, 0.9, 0.75, 0.5, 0.325, 0.25]

theorem milk_volume_calculation :
  let total_volume := milk_volumes.sum
  let average_volume := total_volume / milk_volumes.length
  total_volume = 6.825 ∧ average_volume = 0.975 := by sorry

end NUMINAMATH_CALUDE_milk_volume_calculation_l1226_122669


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1226_122618

/-- Given a line in vector form, proves that its slope-intercept form has specific m and b values -/
theorem line_vector_to_slope_intercept :
  let vector_form := fun (x y : ℝ) => -3 * (x - 5) + 2 * (y + 1) = 0
  let slope_intercept_form := fun (x y : ℝ) => y = (3/2) * x - 17/2
  (∀ x y, vector_form x y ↔ slope_intercept_form x y) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1226_122618


namespace NUMINAMATH_CALUDE_equation_solutions_l1226_122670

def has_different_divisors (a b : ℤ) : Prop :=
  ∃ d : ℤ, (d ∣ a ∧ ¬(d ∣ b)) ∨ (d ∣ b ∧ ¬(d ∣ a))

theorem equation_solutions :
  ∀ a b : ℤ, has_different_divisors a b → a^2 + a = b^3 + b →
  ((a = 1 ∧ b = 1) ∨ (a = -2 ∧ b = 1) ∨ (a = 5 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1226_122670


namespace NUMINAMATH_CALUDE_trig_identity_l1226_122656

theorem trig_identity (α : ℝ) : 
  -Real.sin α + Real.sqrt 3 * Real.cos α = 2 * Real.sin (α + 2 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l1226_122656


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l1226_122654

theorem fruit_basket_problem :
  Nat.gcd (Nat.gcd 15 9) 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l1226_122654


namespace NUMINAMATH_CALUDE_average_price_rahim_l1226_122624

/-- Represents a book purchase from a shop -/
structure BookPurchase where
  quantity : ℕ
  totalPrice : ℕ

/-- Calculates the average price per book given a list of book purchases -/
def averagePrice (purchases : List BookPurchase) : ℚ :=
  let totalBooks := purchases.map (fun p => p.quantity) |>.sum
  let totalCost := purchases.map (fun p => p.totalPrice) |>.sum
  (totalCost : ℚ) / (totalBooks : ℚ)

theorem average_price_rahim (purchases : List BookPurchase) 
  (h1 : purchases = [
    ⟨40, 600⟩,  -- Shop A
    ⟨20, 240⟩,  -- Shop B
    ⟨15, 180⟩,  -- Shop C
    ⟨25, 325⟩   -- Shop D
  ]) : 
  averagePrice purchases = 1345 / 100 := by
  sorry

#eval (1345 : ℚ) / 100  -- To verify the result is indeed 13.45

end NUMINAMATH_CALUDE_average_price_rahim_l1226_122624


namespace NUMINAMATH_CALUDE_calculation_proof_l1226_122650

theorem calculation_proof :
  ((125 + 17) * 8 = 1136) ∧ ((458 - (85 + 28)) / 23 = 15) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1226_122650


namespace NUMINAMATH_CALUDE_updated_average_weight_average_weight_proof_l1226_122661

theorem updated_average_weight (initial_avg : ℝ) (second_avg : ℝ) (third_avg : ℝ) 
  (correction1 : ℝ) (correction2 : ℝ) (correction3 : ℝ) : ℝ :=
  let initial_total := initial_avg * 5
  let second_total := second_avg * 9
  let third_total := third_avg * 12
  let corrected_total := third_total + correction1 + correction2 + correction3
  corrected_total / 12

theorem average_weight_proof :
  updated_average_weight 60 63 64 5 5 5 = 64.4167 := by
  sorry

end NUMINAMATH_CALUDE_updated_average_weight_average_weight_proof_l1226_122661


namespace NUMINAMATH_CALUDE_lakers_win_probability_l1226_122699

/-- The probability of a team winning a single game in the NBA finals -/
def win_prob : ℚ := 1/4

/-- The number of wins needed to win the NBA finals -/
def wins_needed : ℕ := 4

/-- The total number of games in a 7-game series -/
def total_games : ℕ := 7

/-- The probability of the Lakers winning the NBA finals in exactly 7 games -/
def lakers_win_in_seven : ℚ := 135/4096

theorem lakers_win_probability :
  lakers_win_in_seven = (Nat.choose 6 3 : ℚ) * win_prob^3 * (1 - win_prob)^3 * win_prob :=
by sorry

end NUMINAMATH_CALUDE_lakers_win_probability_l1226_122699


namespace NUMINAMATH_CALUDE_equation_solutions_l1226_122638

def is_solution (X Y Z : ℕ) : Prop :=
  X^Y + Y^Z = X * Y * Z

theorem equation_solutions :
  ∀ X Y Z : ℕ,
    is_solution X Y Z ↔
      (X = 1 ∧ Y = 1 ∧ Z = 2) ∨
      (X = 2 ∧ Y = 2 ∧ Z = 2) ∨
      (X = 2 ∧ Y = 2 ∧ Z = 3) ∨
      (X = 4 ∧ Y = 2 ∧ Z = 3) ∨
      (X = 4 ∧ Y = 2 ∧ Z = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1226_122638


namespace NUMINAMATH_CALUDE_circle_equation_l1226_122610

/-- The standard equation of a circle with center (-3, 4) and radius √5 -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-3, 4)
  let radius : ℝ := Real.sqrt 5
  (x + 3)^2 + (y - 4)^2 = 5 ↔
    ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1226_122610


namespace NUMINAMATH_CALUDE_motorcyclist_cyclist_problem_l1226_122652

/-- The distance between two points A and B, given the conditions of the problem -/
def distance_AB : ℝ := 20

theorem motorcyclist_cyclist_problem (x : ℝ) 
  (h1 : x > 0) -- Ensure distance is positive
  (h2 : x - 4 > 0) -- Ensure meeting point is between A and B
  (h3 : (x - 4) / 4 = x / (x - 15)) -- Ratio of speeds equation
  : x = distance_AB := by
  sorry

end NUMINAMATH_CALUDE_motorcyclist_cyclist_problem_l1226_122652


namespace NUMINAMATH_CALUDE_unique_intersection_point_l1226_122620

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 9*x^2 + 27*x - 14

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, 
    (∀ x, f x = (p.2) ↔ x = p.1) ∧ 
    p.1 = p.2 ∧ 
    p = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l1226_122620


namespace NUMINAMATH_CALUDE_percentage_problem_l1226_122655

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.15 * 1500 - 30 → x = 780 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1226_122655


namespace NUMINAMATH_CALUDE_smallest_coinciding_triangle_l1226_122651

/-- Represents the type of isosceles triangle -/
inductive TriangleType
  | Acute
  | Right

/-- Returns the vertex angle of a triangle based on its type -/
def vertexAngle (t : TriangleType) : ℕ :=
  match t with
  | TriangleType.Acute => 30
  | TriangleType.Right => 90

/-- Returns the type of the n-th triangle in the sequence -/
def nthTriangleType (n : ℕ) : TriangleType :=
  if n % 3 = 0 then TriangleType.Right else TriangleType.Acute

/-- Calculates the sum of vertex angles for the first n triangles -/
def sumOfAngles (n : ℕ) : ℕ :=
  List.range n |> List.map (fun i => vertexAngle (nthTriangleType (i + 1))) |> List.sum

/-- The main theorem to prove -/
theorem smallest_coinciding_triangle : 
  (∀ k < 23, sumOfAngles k % 360 ≠ 0) ∧ sumOfAngles 23 % 360 = 0 := by
  sorry


end NUMINAMATH_CALUDE_smallest_coinciding_triangle_l1226_122651


namespace NUMINAMATH_CALUDE_initial_order_is_60_l1226_122615

/-- Represents the cog production scenario with two production rates and an overall average --/
def CogProduction (initial_rate : ℝ) (increased_rate : ℝ) (additional_cogs : ℝ) (average_output : ℝ) : Prop :=
  ∃ (initial_order : ℝ),
    initial_order > 0 ∧
    (initial_order + additional_cogs) / (initial_order / initial_rate + additional_cogs / increased_rate) = average_output

/-- Theorem stating that given the specific production rates and average, the initial order is 60 cogs --/
theorem initial_order_is_60 :
  CogProduction 15 60 60 24 → ∃ (x : ℝ), x = 60 ∧ CogProduction 15 60 60 24 := by
  sorry

#check initial_order_is_60

end NUMINAMATH_CALUDE_initial_order_is_60_l1226_122615


namespace NUMINAMATH_CALUDE_investment_growth_l1226_122614

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.10

/-- The number of years the investment grows -/
def years : ℕ := 4

/-- The initial investment amount -/
def initial_investment : ℝ := 300

/-- The final value after compounding -/
def final_value : ℝ := 439.23

/-- Theorem stating that the initial investment grows to the final value 
    when compounded annually at the given interest rate for the specified number of years -/
theorem investment_growth :
  initial_investment * (1 + interest_rate) ^ years = final_value := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l1226_122614


namespace NUMINAMATH_CALUDE_tangent_length_is_six_l1226_122667

/-- A circle passing through three points -/
structure Circle3Points where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- The length of the tangent from a point to a circle -/
def tangentLength (origin : ℝ × ℝ) (circle : Circle3Points) : ℝ :=
  sorry

/-- Theorem: The length of the tangent from the origin to the specific circle is 6 -/
theorem tangent_length_is_six : 
  let origin : ℝ × ℝ := (0, 0)
  let circle : Circle3Points := { 
    p1 := (2, 3),
    p2 := (4, 6),
    p3 := (6, 15)
  }
  tangentLength origin circle = 6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_is_six_l1226_122667


namespace NUMINAMATH_CALUDE_obtuse_triangle_consecutive_sides_l1226_122619

/-- An obtuse triangle with consecutive natural number side lengths has sides 2, 3, and 4 -/
theorem obtuse_triangle_consecutive_sides : 
  ∀ (a b c : ℕ), 
  (a < b) ∧ (b < c) ∧  -- consecutive
  (c = a + 2) ∧        -- consecutive
  (c^2 > a^2 + b^2) →  -- obtuse (by law of cosines)
  a = 2 ∧ b = 3 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_consecutive_sides_l1226_122619


namespace NUMINAMATH_CALUDE_tangent_line_fixed_point_l1226_122633

/-- The function f(x) = x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

/-- The derivative of f(x) -/
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 2*x + m

/-- Theorem: The tangent line to f(x) at x = 2 passes through (0, -3) for all m -/
theorem tangent_line_fixed_point (m : ℝ) : 
  let x₀ : ℝ := 2
  let y₀ : ℝ := f m x₀
  let slope : ℝ := f_derivative m x₀
  ∃ (k : ℝ), k * slope = y₀ + 3 ∧ k * (-1) = x₀ := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_fixed_point_l1226_122633


namespace NUMINAMATH_CALUDE_max_balloons_with_promotion_orvin_max_balloons_l1226_122687

/-- The maximum number of balloons that can be bought given a promotion --/
theorem max_balloons_with_promotion (full_price_balloons : ℕ) : ℕ :=
  let discounted_sets := (full_price_balloons * 2) / 3
  discounted_sets * 2

/-- Proof that given the conditions, the maximum number of balloons Orvin can buy is 52 --/
theorem orvin_max_balloons : max_balloons_with_promotion 40 = 52 := by
  sorry

end NUMINAMATH_CALUDE_max_balloons_with_promotion_orvin_max_balloons_l1226_122687


namespace NUMINAMATH_CALUDE_quadratic_zeros_imply_a_range_l1226_122603

/-- A quadratic function f(x) = x^2 - 2ax + 4 with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 4

/-- The property that f has two zeros in the interval (1, +∞) -/
def has_two_zeros_after_one (a : ℝ) : Prop :=
  ∃ x y, 1 < x ∧ x < y ∧ f a x = 0 ∧ f a y = 0

/-- If f(x) = x^2 - 2ax + 4 has two zeros in (1, +∞), then 2 < a < 5/2 -/
theorem quadratic_zeros_imply_a_range (a : ℝ) : 
  has_two_zeros_after_one a → 2 < a ∧ a < 5/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_zeros_imply_a_range_l1226_122603


namespace NUMINAMATH_CALUDE_problem_statement_l1226_122694

theorem problem_statement (a b c d : ℕ+) 
  (h1 : a^3 = b^2) 
  (h2 : c^4 = d^3) 
  (h3 : c - a = 31) : 
  d - b = 229 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1226_122694


namespace NUMINAMATH_CALUDE_pet_shop_limbs_l1226_122629

/-- The total number of legs and arms in the pet shop -/
def total_limbs : ℕ :=
  4 * 2 +  -- birds
  6 * 4 +  -- dogs
  5 * 0 +  -- snakes
  2 * 8 +  -- spiders
  3 * 4 +  -- horses
  7 * 4 +  -- rabbits
  2 * 8 +  -- octopuses
  8 * 6 +  -- ants
  1 * 12   -- unique creature

/-- Theorem stating that the total number of legs and arms in the pet shop is 164 -/
theorem pet_shop_limbs : total_limbs = 164 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_limbs_l1226_122629


namespace NUMINAMATH_CALUDE_equation_solution_l1226_122680

theorem equation_solution :
  let f (n : ℝ) := (3 - 2*n) / (n + 2) + (3*n - 9) / (3 - 2*n)
  let n₁ := (25 + Real.sqrt 13) / 18
  let n₂ := (25 - Real.sqrt 13) / 18
  f n₁ = 2 ∧ f n₂ = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1226_122680


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1226_122621

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 2 * a 3 = 5 →
  a 7 * a 8 * a 9 = 10 →
  a 4 * a 5 * a 6 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1226_122621


namespace NUMINAMATH_CALUDE_platform_length_l1226_122625

/-- Calculates the length of a platform given train parameters --/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : 
  train_length = 175 →
  train_speed_kmph = 36 →
  crossing_time = 40 →
  (train_speed_kmph * 1000 / 3600 * crossing_time) - train_length = 225 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l1226_122625


namespace NUMINAMATH_CALUDE_matt_current_age_is_65_l1226_122640

def james_age_3_years_ago : ℕ := 27
def years_since_james_27 : ℕ := 3
def years_until_matt_twice_james : ℕ := 5

def james_current_age : ℕ := james_age_3_years_ago + years_since_james_27

def james_age_in_5_years : ℕ := james_current_age + years_until_matt_twice_james

def matt_age_in_5_years : ℕ := 2 * james_age_in_5_years

theorem matt_current_age_is_65 : matt_age_in_5_years - years_until_matt_twice_james = 65 := by
  sorry

end NUMINAMATH_CALUDE_matt_current_age_is_65_l1226_122640


namespace NUMINAMATH_CALUDE_z_sixth_power_l1226_122673

theorem z_sixth_power (z : ℂ) : z = (-Real.sqrt 5 + Complex.I) / 2 → z^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_z_sixth_power_l1226_122673


namespace NUMINAMATH_CALUDE_cross_product_perpendicular_l1226_122666

/-- The cross product of two 3D vectors -/
def cross_product (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => match i with
    | 0 => v 1 * w 2 - v 2 * w 1
    | 1 => v 2 * w 0 - v 0 * w 2
    | 2 => v 0 * w 1 - v 1 * w 0

/-- The dot product of two 3D vectors -/
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

theorem cross_product_perpendicular (v w : Fin 3 → ℝ) :
  let v1 : Fin 3 → ℝ := fun i => match i with
    | 0 => 3
    | 1 => -2
    | 2 => 4
  let v2 : Fin 3 → ℝ := fun i => match i with
    | 0 => 1
    | 1 => 5
    | 2 => -3
  let cp := cross_product v1 v2
  cp 0 = -14 ∧ cp 1 = 13 ∧ cp 2 = 17 ∧
  dot_product cp v1 = 0 ∧ dot_product cp v2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cross_product_perpendicular_l1226_122666


namespace NUMINAMATH_CALUDE_fourth_grid_shaded_fraction_initial_shaded_squares_shaded_squares_arithmetic_l1226_122628

/-- Represents the number of shaded squares in the nth grid -/
def shaded_squares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the total number of squares in the nth grid -/
def total_squares (n : ℕ) : ℕ := n ^ 2

/-- The main theorem stating the fraction of shaded squares in the fourth grid -/
theorem fourth_grid_shaded_fraction :
  (shaded_squares 4 : ℚ) / (total_squares 4 : ℚ) = 7 / 16 := by
  sorry

/-- Verifies that the first three grids have 1, 3, and 5 shaded squares respectively -/
theorem initial_shaded_squares :
  shaded_squares 1 = 1 ∧ shaded_squares 2 = 3 ∧ shaded_squares 3 = 5 := by
  sorry

/-- Verifies that the sequence of shaded squares is arithmetic -/
theorem shaded_squares_arithmetic :
  ∀ n : ℕ, shaded_squares (n + 1) - shaded_squares n = 
           shaded_squares (n + 2) - shaded_squares (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_fourth_grid_shaded_fraction_initial_shaded_squares_shaded_squares_arithmetic_l1226_122628


namespace NUMINAMATH_CALUDE_suit_price_increase_l1226_122662

theorem suit_price_increase (original_price : ℝ) (discounted_price : ℝ) :
  original_price = 160 →
  discounted_price = 150 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 25 ∧
    discounted_price = (original_price * (1 + increase_percentage / 100)) * 0.75 :=
by sorry

end NUMINAMATH_CALUDE_suit_price_increase_l1226_122662


namespace NUMINAMATH_CALUDE_max_rectangular_pen_area_l1226_122634

/-- Given 50 feet of fencing with 5 feet used for a non-enclosing gate,
    the maximum area of a rectangular pen enclosed by the remaining fencing
    is 126.5625 square feet. -/
theorem max_rectangular_pen_area : 
  ∀ (width height : ℝ),
    width > 0 → height > 0 →
    width + height = (50 - 5) / 2 →
    width * height ≤ 126.5625 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangular_pen_area_l1226_122634


namespace NUMINAMATH_CALUDE_distinct_triangles_in_grid_l1226_122682

/-- The number of points in a 3 x 2 grid -/
def total_points : ℕ := 6

/-- The number of points needed to form a triangle -/
def points_per_triangle : ℕ := 3

/-- The number of rows in the grid -/
def num_rows : ℕ := 3

/-- Function to calculate combinations -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (List.range n).foldl (λ acc i => acc * (n - i) / (i + 1)) 1

/-- The number of degenerate cases (collinear points in rows) -/
def degenerate_cases : ℕ := num_rows

/-- Theorem: The number of distinct triangles in a 3 x 2 grid is 17 -/
theorem distinct_triangles_in_grid :
  choose total_points points_per_triangle - degenerate_cases = 17 := by
  sorry


end NUMINAMATH_CALUDE_distinct_triangles_in_grid_l1226_122682


namespace NUMINAMATH_CALUDE_solution_count_l1226_122686

-- Define the equations
def equation1 (x y : ℂ) : Prop := y = (x + 2)^3
def equation2 (x y : ℂ) : Prop := x * y + 2 * y = 2

-- Define a solution pair
def is_solution (x y : ℂ) : Prop := equation1 x y ∧ equation2 x y

-- Define the count of real and imaginary solutions
def real_solution_count : ℕ := 2
def imaginary_solution_count : ℕ := 2

-- Theorem statement
theorem solution_count :
  (∃ (s : Finset (ℂ × ℂ)), s.card = real_solution_count + imaginary_solution_count ∧
    (∀ (p : ℂ × ℂ), p ∈ s ↔ is_solution p.1 p.2) ∧
    (∃ (r : Finset (ℂ × ℂ)), r ⊆ s ∧ r.card = real_solution_count ∧
      (∀ (p : ℂ × ℂ), p ∈ r → p.1.im = 0 ∧ p.2.im = 0)) ∧
    (∃ (i : Finset (ℂ × ℂ)), i ⊆ s ∧ i.card = imaginary_solution_count ∧
      (∀ (p : ℂ × ℂ), p ∈ i → p.1.im ≠ 0 ∨ p.2.im ≠ 0))) :=
sorry

end NUMINAMATH_CALUDE_solution_count_l1226_122686


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l1226_122604

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the perimeter function
def perimeter (q : Quadrilateral) : ℝ := sorry

-- Define the perpendicular function
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_perimeter 
  (ABCD : Quadrilateral)
  (perp_AB_BC : perpendicular (ABCD.B - ABCD.A) (ABCD.C - ABCD.B))
  (perp_DC_BC : perpendicular (ABCD.C - ABCD.D) (ABCD.C - ABCD.B))
  (AB_length : distance ABCD.A ABCD.B = 15)
  (DC_length : distance ABCD.D ABCD.C = 6)
  (BC_length : distance ABCD.B ABCD.C = 10)
  (AB_eq_AD : distance ABCD.A ABCD.B = distance ABCD.A ABCD.D) :
  perimeter ABCD = 31 + Real.sqrt 181 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l1226_122604


namespace NUMINAMATH_CALUDE_angle_rotation_and_trig_identity_l1226_122689

theorem angle_rotation_and_trig_identity 
  (initial_angle : Real) 
  (rotations : Nat) 
  (α : Real) 
  (h1 : initial_angle = 30 * Real.pi / 180)
  (h2 : rotations = 3)
  (h3 : Real.sin (-Real.pi/2 - α) = -1/3)
  (h4 : Real.tan α < 0) :
  (initial_angle + rotations * 2 * Real.pi) * 180 / Real.pi = 1110 ∧ 
  Real.cos (3 * Real.pi / 2 + α) = -2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_rotation_and_trig_identity_l1226_122689


namespace NUMINAMATH_CALUDE_linear_function_k_value_l1226_122698

/-- Given a linear function y = kx + 1 passing through the point (-1, 0), prove that k = 1 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1) → -- The function is linear with equation y = kx + 1
  (0 = k * (-1) + 1) →         -- The graph passes through the point (-1, 0)
  k = 1                        -- Conclusion: k equals 1
:= by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l1226_122698


namespace NUMINAMATH_CALUDE_cube_surface_area_l1226_122688

/-- Given a cube with volume x^3, its surface area is 6x^2 -/
theorem cube_surface_area (x : ℝ) (h : x > 0) :
  (6 : ℝ) * x^2 = 6 * (x^3)^((2:ℝ)/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1226_122688


namespace NUMINAMATH_CALUDE_trains_crossing_time_l1226_122600

/-- Proves the time taken for two trains to cross each other -/
theorem trains_crossing_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (h1 : length = 120)
  (h2 : time1 = 5)
  (h3 : time2 = 15)
  : (2 * length) / ((length / time1) + (length / time2)) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_trains_crossing_time_l1226_122600


namespace NUMINAMATH_CALUDE_mountaineer_arrangements_l1226_122693

theorem mountaineer_arrangements (total : ℕ) (familiar : ℕ) (groups : ℕ) (familiar_per_group : ℕ) :
  total = 10 →
  familiar = 4 →
  groups = 2 →
  familiar_per_group = 2 →
  (familiar.choose familiar_per_group) * ((total - familiar).choose familiar_per_group) * groups = 120 :=
by sorry

end NUMINAMATH_CALUDE_mountaineer_arrangements_l1226_122693


namespace NUMINAMATH_CALUDE_tank_capacity_l1226_122675

theorem tank_capacity (initial_fullness : Rat) (final_fullness : Rat) (added_water : Rat) :
  initial_fullness = 1/4 →
  final_fullness = 2/3 →
  added_water = 120 →
  (final_fullness - initial_fullness) * (added_water / (final_fullness - initial_fullness)) = 288 :=
by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l1226_122675


namespace NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l1226_122674

theorem vector_subtraction_scalar_multiplication :
  let v1 : Fin 2 → ℝ := ![3, -8]
  let v2 : Fin 2 → ℝ := ![-2, 6]
  let scalar : ℝ := 5
  v1 - scalar • v2 = ![13, -38] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l1226_122674


namespace NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l1226_122690

theorem polynomial_value_at_n_plus_one (n : ℕ) (P : Polynomial ℝ) 
  (h_degree : P.degree ≤ n) 
  (h_values : ∀ k : ℕ, k ≤ n → P.eval (k : ℝ) = k / (k + 1)) :
  P.eval ((n + 1 : ℕ) : ℝ) = (n + 1 + (-1)^(n + 1)) / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l1226_122690


namespace NUMINAMATH_CALUDE_shortest_chord_equation_l1226_122671

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 24 = 0

/-- Line l -/
def line_l (x y k : ℝ) : Prop := y = k*(x - 2) - 1

/-- Line AB -/
def line_AB (x y : ℝ) : Prop := x - y - 3 = 0

/-- The theorem statement -/
theorem shortest_chord_equation (k : ℝ) :
  (∃ A B : ℝ × ℝ, 
    (circle_C A.1 A.2 ∧ circle_C B.1 B.2) ∧ 
    (line_l A.1 A.2 k ∧ line_l B.1 B.2 k) ∧
    (∀ P Q : ℝ × ℝ, circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧ 
      line_l P.1 P.2 k ∧ line_l Q.1 Q.2 k →
      (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ (P.1 - Q.1)^2 + (P.2 - Q.2)^2)) →
  (∀ x y : ℝ, line_AB x y ↔ (circle_C x y ∧ line_l x y k)) :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_equation_l1226_122671


namespace NUMINAMATH_CALUDE_greg_age_l1226_122608

/-- Given the ages and relationships of siblings, prove Greg's age -/
theorem greg_age (cindy_age : ℕ) (jan_age : ℕ) (marcia_age : ℕ) (greg_age : ℕ)
  (h1 : cindy_age = 5)
  (h2 : jan_age = cindy_age + 2)
  (h3 : marcia_age = 2 * jan_age)
  (h4 : greg_age = marcia_age + 2) :
  greg_age = 16 := by
sorry

end NUMINAMATH_CALUDE_greg_age_l1226_122608


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1226_122637

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 1/y ≥ 2 ∧ (1/x + 1/y = 2 ↔ x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1226_122637


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1226_122612

theorem arithmetic_computation : -10 * 3 - (-4 * -2) + (-12 * -4) / 2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1226_122612


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1226_122607

theorem chess_tournament_games (n : ℕ) (h : n = 8) : 
  n * (n - 1) = 56 ∧ 2 * (n * (n - 1)) = 112 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l1226_122607


namespace NUMINAMATH_CALUDE_carrot_cost_correct_l1226_122627

/-- Represents the cost of carrots for all students -/
def carrot_cost : ℚ := 185

/-- Represents the number of third grade classes -/
def third_grade_classes : ℕ := 5

/-- Represents the number of students in each third grade class -/
def third_grade_students_per_class : ℕ := 30

/-- Represents the number of fourth grade classes -/
def fourth_grade_classes : ℕ := 4

/-- Represents the number of students in each fourth grade class -/
def fourth_grade_students_per_class : ℕ := 28

/-- Represents the number of fifth grade classes -/
def fifth_grade_classes : ℕ := 4

/-- Represents the number of students in each fifth grade class -/
def fifth_grade_students_per_class : ℕ := 27

/-- Represents the cost of a hamburger -/
def hamburger_cost : ℚ := 21/10

/-- Represents the cost of a cookie -/
def cookie_cost : ℚ := 1/5

/-- Represents the total cost of lunch for all students -/
def total_lunch_cost : ℚ := 1036

/-- Theorem stating that the cost of carrots is correct given the conditions -/
theorem carrot_cost_correct : 
  let total_students := third_grade_classes * third_grade_students_per_class + 
                        fourth_grade_classes * fourth_grade_students_per_class + 
                        fifth_grade_classes * fifth_grade_students_per_class
  total_lunch_cost = total_students * (hamburger_cost + cookie_cost) + carrot_cost :=
by sorry

end NUMINAMATH_CALUDE_carrot_cost_correct_l1226_122627


namespace NUMINAMATH_CALUDE_rachel_total_steps_l1226_122684

/-- Represents a landmark with its stair information -/
structure Landmark where
  name : String
  flightsUp : Nat
  flightsDown : Nat
  stepsPerFlight : Nat

/-- Calculates the total steps for a single landmark -/
def stepsForLandmark (l : Landmark) : Nat :=
  (l.flightsUp + l.flightsDown) * l.stepsPerFlight

/-- The list of landmarks Rachel visited -/
def landmarks : List Landmark := [
  { name := "Eiffel Tower", flightsUp := 347, flightsDown := 216, stepsPerFlight := 10 },
  { name := "Notre-Dame Cathedral", flightsUp := 178, flightsDown := 165, stepsPerFlight := 12 },
  { name := "Leaning Tower of Pisa", flightsUp := 294, flightsDown := 172, stepsPerFlight := 8 },
  { name := "Colosseum", flightsUp := 122, flightsDown := 93, stepsPerFlight := 15 },
  { name := "Sagrada Familia", flightsUp := 267, flightsDown := 251, stepsPerFlight := 11 },
  { name := "Park Güell", flightsUp := 134, flightsDown := 104, stepsPerFlight := 9 }
]

/-- Calculates the total steps for all landmarks -/
def totalSteps : Nat :=
  landmarks.map stepsForLandmark |>.sum

theorem rachel_total_steps :
  totalSteps = 24539 := by
  sorry

end NUMINAMATH_CALUDE_rachel_total_steps_l1226_122684


namespace NUMINAMATH_CALUDE_horner_v₂_value_l1226_122641

def horner_polynomial (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 - 3*x^3 + 1.8*x^2 + 0.35*x + 2

def v₀ : ℝ := 1

def v₁ (x : ℝ) : ℝ := v₀ * x - 5

def v₂ (x : ℝ) : ℝ := v₁ x * x + 6

theorem horner_v₂_value :
  v₂ (-1) = 12 :=
by sorry

end NUMINAMATH_CALUDE_horner_v₂_value_l1226_122641


namespace NUMINAMATH_CALUDE_last_digit_is_three_l1226_122685

/-- Represents a four-digit number -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  h1 : d1 < 10
  h2 : d2 < 10
  h3 : d3 < 10
  h4 : d4 < 10

/-- Predicate for the first clue -/
def clue1 (n : FourDigitNumber) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧ 
    (n.d1 = (Vector.get ⟨[1,3,5,9], rfl⟩ i) ∧ i ≠ 0) ∧
    (n.d2 = (Vector.get ⟨[1,3,5,9], rfl⟩ j) ∧ j ≠ 1)

/-- Predicate for the second clue -/
def clue2 (n : FourDigitNumber) : Prop :=
  n.d1 = 9 ∨ n.d2 = 0 ∨ n.d3 = 1 ∨ n.d4 = 3

/-- Predicate for the third clue -/
def clue3 (n : FourDigitNumber) : Prop :=
  (n.d1 = 9 ∧ (n.d2 = 0 ∨ n.d3 = 1 ∨ n.d4 = 3)) ∨
  (n.d2 = 0 ∧ (n.d1 = 9 ∨ n.d3 = 1 ∨ n.d4 = 3)) ∨
  (n.d3 = 1 ∧ (n.d1 = 9 ∨ n.d2 = 0 ∨ n.d4 = 3)) ∨
  (n.d4 = 3 ∧ (n.d1 = 9 ∨ n.d2 = 0 ∨ n.d3 = 1))

/-- Predicate for the fourth clue -/
def clue4 (n : FourDigitNumber) : Prop :=
  (n.d2 = 1 ∨ n.d3 = 1 ∨ n.d4 = 1) ∧ n.d1 ≠ 1

/-- Predicate for the fifth clue -/
def clue5 (n : FourDigitNumber) : Prop :=
  n.d1 ≠ 7 ∧ n.d1 ≠ 6 ∧ n.d1 ≠ 4 ∧ n.d1 ≠ 2 ∧
  n.d2 ≠ 7 ∧ n.d2 ≠ 6 ∧ n.d2 ≠ 4 ∧ n.d2 ≠ 2 ∧
  n.d3 ≠ 7 ∧ n.d3 ≠ 6 ∧ n.d3 ≠ 4 ∧ n.d3 ≠ 2 ∧
  n.d4 ≠ 7 ∧ n.d4 ≠ 6 ∧ n.d4 ≠ 4 ∧ n.d4 ≠ 2

theorem last_digit_is_three (n : FourDigitNumber) 
  (h1 : clue1 n) (h2 : clue2 n) (h3 : clue3 n) (h4 : clue4 n) (h5 : clue5 n) : 
  n.d4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_is_three_l1226_122685
