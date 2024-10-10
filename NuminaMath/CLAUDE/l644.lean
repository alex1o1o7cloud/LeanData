import Mathlib

namespace matrix_rank_theorem_l644_64437

theorem matrix_rank_theorem (m n : ℕ) (A : Matrix (Fin m) (Fin n) ℚ) 
  (h : ∃ (S : Finset ℕ), S.card ≥ m + n ∧ 
    (∀ p ∈ S, Nat.Prime p ∧ ∃ (i : Fin m) (j : Fin n), |A i j| = p)) : 
  Matrix.rank A ≥ 2 := by
sorry

end matrix_rank_theorem_l644_64437


namespace inequality_proof_l644_64469

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 ∧ ((a + b) / 2)^2 ≥ a * b :=
by sorry

end inequality_proof_l644_64469


namespace solve_nested_function_l644_64435

def f (p : ℝ) : ℝ := 2 * p + 20

theorem solve_nested_function : ∃ p : ℝ, f (f (f p)) = -4 ∧ p = -18 := by
  sorry

end solve_nested_function_l644_64435


namespace sheets_per_ream_l644_64498

theorem sheets_per_ream (cost_per_ream : ℕ) (sheets_needed : ℕ) (total_cost : ℕ) :
  cost_per_ream = 27 →
  sheets_needed = 5000 →
  total_cost = 270 →
  (sheets_needed / (total_cost / cost_per_ream) : ℕ) = 500 := by
  sorry

end sheets_per_ream_l644_64498


namespace sum_of_seven_odds_mod_twelve_l644_64462

theorem sum_of_seven_odds_mod_twelve (n : ℕ) (h : n = 10331) : 
  (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) + (n + 12)) % 12 = 7 := by
  sorry

end sum_of_seven_odds_mod_twelve_l644_64462


namespace modulus_of_complex_expression_l644_64488

theorem modulus_of_complex_expression :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_complex_expression_l644_64488


namespace marble_selection_ways_l644_64429

def total_marbles : ℕ := 15
def red_marbles : ℕ := 2
def green_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def marbles_to_choose : ℕ := 5

theorem marble_selection_ways :
  (red_marbles.choose 1) * (green_marbles.choose 1) *
  ((total_marbles - red_marbles - green_marbles + 2).choose (marbles_to_choose - 2)) = 660 := by
  sorry

end marble_selection_ways_l644_64429


namespace bicycle_profit_problem_l644_64420

theorem bicycle_profit_problem (initial_cost final_price : ℝ) : 
  (initial_cost * 1.25 * 1.25 = final_price) →
  (final_price = 225) →
  (initial_cost = 144) := by
sorry

end bicycle_profit_problem_l644_64420


namespace person_savings_l644_64425

/-- Calculates a person's savings given their income and income-to-expenditure ratio --/
theorem person_savings (income : ℚ) (ratio_income ratio_expenditure : ℕ) 
  (h1 : income = 18000)
  (h2 : ratio_income = 9)
  (h3 : ratio_expenditure = 8) : 
  income - (income * ratio_expenditure / ratio_income) = 2000 := by
  sorry

end person_savings_l644_64425


namespace cone_from_sector_l644_64451

/-- Represents a circular sector -/
structure CircularSector where
  radius : ℝ
  angle : ℝ

/-- Represents a cone -/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ
  height : ℝ

/-- Checks if a cone can be formed from a given circular sector -/
def canFormCone (sector : CircularSector) (cone : Cone) : Prop :=
  -- The slant height of the cone equals the radius of the sector
  cone.slantHeight = sector.radius ∧
  -- The arc length of the sector equals the circumference of the cone's base
  (sector.angle / 360) * (2 * Real.pi * sector.radius) = 2 * Real.pi * cone.baseRadius ∧
  -- The Pythagorean theorem holds for the cone's dimensions
  cone.slantHeight ^ 2 = cone.baseRadius ^ 2 + cone.height ^ 2

/-- Theorem stating that a specific cone can be formed from a given sector -/
theorem cone_from_sector :
  let sector := CircularSector.mk 15 300
  let cone := Cone.mk 12 15 9
  canFormCone sector cone := by
  sorry

end cone_from_sector_l644_64451


namespace chris_age_l644_64473

theorem chris_age (c m : ℕ) : c = 3 * m - 22 → c + m = 70 → c = 47 := by
  sorry

end chris_age_l644_64473


namespace perpendicular_parallel_transitivity_l644_64428

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_transitivity 
  (l : Line) (α β : Plane) :
  perp l α → para α β → perp l β :=
by sorry

end perpendicular_parallel_transitivity_l644_64428


namespace system_of_equations_solution_l644_64483

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (6 * x - 3 * y = -3) ∧ (5 * x - 9 * y = -35) ∧ x = 2 ∧ y = 5 := by
  sorry

end system_of_equations_solution_l644_64483


namespace farm_water_consumption_l644_64426

theorem farm_water_consumption 
  (num_cows : ℕ)
  (cow_daily_water : ℕ)
  (sheep_cow_ratio : ℕ)
  (sheep_water_ratio : ℚ)
  (days_in_week : ℕ)
  (h1 : num_cows = 40)
  (h2 : cow_daily_water = 80)
  (h3 : sheep_cow_ratio = 10)
  (h4 : sheep_water_ratio = 1/4)
  (h5 : days_in_week = 7) :
  (num_cows * cow_daily_water * days_in_week) + 
  (num_cows * sheep_cow_ratio * (sheep_water_ratio * cow_daily_water) * days_in_week) = 78400 :=
by sorry

end farm_water_consumption_l644_64426


namespace geometric_series_properties_l644_64432

theorem geometric_series_properties (q : ℝ) (b₁ : ℝ) (h_q : |q| < 1) :
  (b₁ / (1 - q) = 16) →
  (b₁^2 / (1 - q^2) = 153.6) →
  (b₁ * q^3 = 32/9 ∧ q = 2/3) :=
by sorry

end geometric_series_properties_l644_64432


namespace set_equation_solution_l644_64401

theorem set_equation_solution (A X Y : Set α) 
  (h1 : X ∪ Y = A) 
  (h2 : X ∩ A = Y) : 
  X = A ∧ Y = A := by
  sorry

end set_equation_solution_l644_64401


namespace distance_is_1760_l644_64443

/-- The distance between Péter's and Károly's houses in meters. -/
def distance_between_houses : ℝ := 1760

/-- The distance from Péter's house to the first meeting point in meters. -/
def first_meeting_distance : ℝ := 720

/-- The distance from Károly's house to the second meeting point in meters. -/
def second_meeting_distance : ℝ := 400

/-- Theorem stating that the distance between the houses is 1760 meters. -/
theorem distance_is_1760 :
  let x := distance_between_houses
  let d1 := first_meeting_distance
  let d2 := second_meeting_distance
  (d1 / (x - d1) = (x - d2) / (x + d2)) →
  x = 1760 := by
  sorry


end distance_is_1760_l644_64443


namespace lcm_problem_l644_64424

theorem lcm_problem (a b c : ℕ+) (h1 : a = 10) (h2 : c = 20) (h3 : Nat.lcm a (Nat.lcm b c) = 140) : b = 7 := by
  sorry

end lcm_problem_l644_64424


namespace fraction_simplification_l644_64407

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  a^2 / (a * b) = a / b := by
  sorry

end fraction_simplification_l644_64407


namespace correlation_identification_l644_64484

-- Define the relationships
inductive Relationship
| AgeAndFat
| CurvePoints
| FruitProduction
| StudentAndID

-- Define the property of having a correlation
def HasCorrelation : Relationship → Prop :=
  fun r => match r with
  | Relationship.AgeAndFat => true
  | Relationship.CurvePoints => false
  | Relationship.FruitProduction => true
  | Relationship.StudentAndID => false

-- Define the property of being a functional relationship
def IsFunctionalRelationship : Relationship → Prop :=
  fun r => match r with
  | Relationship.AgeAndFat => false
  | Relationship.CurvePoints => true
  | Relationship.FruitProduction => false
  | Relationship.StudentAndID => true

-- Theorem statement
theorem correlation_identification :
  (∀ r : Relationship, HasCorrelation r ↔ ¬(IsFunctionalRelationship r)) ∧
  (HasCorrelation Relationship.AgeAndFat ∧ HasCorrelation Relationship.FruitProduction) ∧
  (¬HasCorrelation Relationship.CurvePoints ∧ ¬HasCorrelation Relationship.StudentAndID) :=
by sorry

end correlation_identification_l644_64484


namespace modulus_of_complex_l644_64402

theorem modulus_of_complex (z : ℂ) : (Complex.I * z = 3 + 4 * Complex.I) → Complex.abs z = 5 := by
  sorry

end modulus_of_complex_l644_64402


namespace log_stacks_total_l644_64454

def first_stack_start : ℕ := 15
def first_stack_end : ℕ := 4
def second_stack_start : ℕ := 5
def second_stack_end : ℕ := 10

def total_logs : ℕ := 159

theorem log_stacks_total :
  (first_stack_start - first_stack_end + 1) * (first_stack_start + first_stack_end) / 2 +
  (second_stack_end - second_stack_start + 1) * (second_stack_start + second_stack_end) / 2 =
  total_logs := by
  sorry

end log_stacks_total_l644_64454


namespace book_purchase_remaining_money_l644_64408

theorem book_purchase_remaining_money (m : ℚ) (n : ℕ) (b : ℚ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : b > 0) 
  (h4 : (1/4) * m = (1/2) * n * b) : 
  m - n * b = (1/2) * m := by
sorry

end book_purchase_remaining_money_l644_64408


namespace probability_is_half_l644_64470

/-- A circular field with six equally spaced radial roads -/
structure CircularField :=
  (radius : ℝ)
  (num_roads : ℕ)
  (h_num_roads : num_roads = 6)

/-- A geologist traveling on one of the roads -/
structure Geologist :=
  (speed : ℝ)
  (road : ℕ)
  (h_speed : speed = 5)
  (h_road : road ∈ Finset.range 6)

/-- The distance between two geologists after one hour -/
def distance (field : CircularField) (g1 g2 : Geologist) : ℝ :=
  sorry

/-- The probability of two geologists being more than 8 km apart -/
def probability (field : CircularField) : ℝ :=
  sorry

/-- Main theorem: The probability is 0.5 -/
theorem probability_is_half (field : CircularField) :
  probability field = 0.5 :=
sorry

end probability_is_half_l644_64470


namespace number_reversal_property_l644_64444

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Counts the number of zero digits in a natural number -/
def countZeroDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is of the form 1099...989 with k repetitions of 99 -/
def isSpecialForm (n : ℕ) : Prop := ∃ k : ℕ, n = 10^(2*k+1) + 9 * (10^(2*k) - 1) / 99

theorem number_reversal_property (N : ℕ) :
  (9 * N = reverseDigits N) ∧ (countZeroDigits N ≤ 1) ↔ N = 0 ∨ isSpecialForm N :=
sorry

end number_reversal_property_l644_64444


namespace limit_tan_sin_ratio_l644_64411

open Real

noncomputable def f (x : ℝ) : ℝ := tan (6 * x) / sin (3 * x)

theorem limit_tan_sin_ratio :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - 2| < ε :=
sorry

end limit_tan_sin_ratio_l644_64411


namespace flower_pots_theorem_l644_64406

/-- Represents a set of items with increasing prices -/
structure IncreasingPriceSet where
  num_items : ℕ
  price_difference : ℚ
  total_cost : ℚ

/-- The cost of the most expensive item in the set -/
def most_expensive_item_cost (s : IncreasingPriceSet) : ℚ :=
  (s.total_cost - (s.num_items - 1) * s.num_items * s.price_difference / 2) / s.num_items + (s.num_items - 1) * s.price_difference

/-- Theorem: For a set of 6 items with $0.15 price difference and $8.25 total cost, 
    the most expensive item costs $1.75 -/
theorem flower_pots_theorem : 
  let s : IncreasingPriceSet := ⟨6, 15/100, 825/100⟩
  most_expensive_item_cost s = 175/100 := by
  sorry

end flower_pots_theorem_l644_64406


namespace negation_equivalence_exists_false_conjunction_true_component_negation_implication_l644_64489

-- Define the propositions
def p : Prop := ∃ x : ℝ, x^2 + x - 1 < 0
def q : Prop := ∃ x : ℝ, x^2 - 3*x + 2 = 0

-- Statement 1
theorem negation_equivalence : (¬p) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by sorry

-- Statement 2
theorem exists_false_conjunction_true_component :
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) := by sorry

-- Statement 3
theorem negation_implication :
  ¬(q → ∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x = 2) ≠
  (q → ∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 2) := by sorry

end negation_equivalence_exists_false_conjunction_true_component_negation_implication_l644_64489


namespace photo_arrangements_l644_64486

def num_boys : ℕ := 4
def num_girls : ℕ := 3

def arrangements_girls_at_ends : ℕ := 720
def arrangements_no_adjacent_girls : ℕ := 1440
def arrangements_girl_A_right_of_B : ℕ := 2520

theorem photo_arrangements :
  (num_boys = 4 ∧ num_girls = 3) →
  (arrangements_girls_at_ends = 720 ∧
   arrangements_no_adjacent_girls = 1440 ∧
   arrangements_girl_A_right_of_B = 2520) :=
by sorry

end photo_arrangements_l644_64486


namespace inequality_solution_set_l644_64445

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x + 2| - |x - 1| < a) → a > -3 := by
  sorry

end inequality_solution_set_l644_64445


namespace hours_per_day_l644_64416

theorem hours_per_day (days : ℕ) (total_hours : ℕ) (h1 : days = 6) (h2 : total_hours = 18) :
  total_hours / days = 3 := by
  sorry

end hours_per_day_l644_64416


namespace triangle_incenter_properties_l644_64464

/-- 
Given a right-angled triangle ABC with angle A = 90°, sides BC = a, AC = b, AB = c,
and a line d passing through the incenter intersecting AB at P and AC at Q.
-/
theorem triangle_incenter_properties 
  (a b c : ℝ) 
  (h_right_angle : a^2 = b^2 + c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (P Q : ℝ × ℝ) 
  (h_P_on_AB : P.1 ≥ 0 ∧ P.1 ≤ c ∧ P.2 = 0)
  (h_Q_on_AC : Q.1 = 0 ∧ Q.2 ≥ 0 ∧ Q.2 ≤ b)
  (h_PQ_through_incenter : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
    P = (t * c, 0) ∧ 
    Q = (0, (1 - t) * b) ∧ 
    t * c / (a + b + c) = (1 - t) * b / (a + b + c)) :
  (b * (c - P.1) / P.1 + c * (b - Q.2) / Q.2 = a) ∧
  (∃ (m : ℝ), ∀ (x y : ℝ), 
    x ≥ 0 ∧ x ≤ c ∧ y ≥ 0 ∧ y ≤ b →
    ((c - x) / x)^2 + ((b - y) / y)^2 ≥ 1) := by
  sorry

end triangle_incenter_properties_l644_64464


namespace men_to_women_ratio_l644_64419

theorem men_to_women_ratio (men : ℝ) (women : ℝ) (h : women = 0.9 * men) :
  (men / women) * 100 = (1 / 0.9) * 100 := by
sorry

end men_to_women_ratio_l644_64419


namespace not_eventually_periodic_l644_64496

/-- The rightmost non-zero digit in the decimal representation of n! -/
def rightmost_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The sequence of rightmost non-zero digits of factorials -/
def a : ℕ → ℕ := rightmost_nonzero_digit

/-- The sequence (a_n)_{n ≥ 0} is not periodic from any certain point onwards -/
theorem not_eventually_periodic :
  ∀ p q : ℕ, ∃ n : ℕ, n ≥ q ∧ a n ≠ a (n + p) :=
sorry

end not_eventually_periodic_l644_64496


namespace towel_area_decrease_l644_64440

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let original_area := L * B
  let new_length := L * (1 - 0.2)
  let new_breadth := B * (1 - 0.1)
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area * 100 = 28 := by
sorry

end towel_area_decrease_l644_64440


namespace point_satisfies_inequalities_l644_64413

-- Define the system of inequalities
def satisfies_inequalities (x y : ℝ) : Prop :=
  (x - 2*y + 5 > 0) ∧ (x - y + 3 ≤ 0)

-- Theorem statement
theorem point_satisfies_inequalities : 
  satisfies_inequalities (-2 : ℝ) (1 : ℝ) :=
by
  sorry

end point_satisfies_inequalities_l644_64413


namespace card_area_theorem_l644_64450

/-- Represents a rectangular card with length and width in inches -/
structure Card where
  length : ℝ
  width : ℝ

/-- Calculates the area of a card in square inches -/
def area (c : Card) : ℝ := c.length * c.width

/-- The original card -/
def original_card : Card := { length := 5, width := 7 }

/-- Theorem: If shortening one side of the original 5x7 card by 2 inches
    results in an area of 21 square inches, then shortening the other side
    by 1 inch results in an area of 30 square inches -/
theorem card_area_theorem :
  (∃ (c : Card), (c.length = original_card.length - 2 ∨ c.width = original_card.width - 2) ∧
                 area c = 21) →
  area { length := original_card.length,
         width := original_card.width - 1 } = 30 := by
  sorry

end card_area_theorem_l644_64450


namespace round_robin_tournament_teams_l644_64446

/-- Represents the total points in a round-robin tournament -/
def totalPoints (n : ℕ) : ℕ := n * (n - 1)

/-- The set of reported total points and their averages -/
def reportedPoints : Finset ℕ := {3086, 2018, 1238, 2162, 2552, 1628, 2114}

/-- Theorem stating that if one of the reported points is correct, then there are 47 teams -/
theorem round_robin_tournament_teams :
  ∃ (p : ℕ), p ∈ reportedPoints ∧ totalPoints 47 = p :=
sorry

end round_robin_tournament_teams_l644_64446


namespace simple_interest_rate_l644_64477

/-- Given a principal amount and a simple interest rate, if the amount after 12 years
    is 9/6 of the principal, then the rate is 100/24 -/
theorem simple_interest_rate (P R : ℝ) (P_pos : P > 0) : 
  P * (1 + R * 12 / 100) = P * (9 / 6) → R = 100 / 24 := by
  sorry

end simple_interest_rate_l644_64477


namespace corrected_mean_l644_64481

theorem corrected_mean (n : ℕ) (incorrect_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 50 ∧ incorrect_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 45 →
  (n : ℚ) * incorrect_mean - incorrect_value + correct_value = 36.44 * n :=
by sorry

end corrected_mean_l644_64481


namespace equation_solutions_count_l644_64471

theorem equation_solutions_count :
  let f : ℝ → ℝ := λ x => 2 * Real.sqrt 2 * (Real.sin (π * x / 4))^3 - Real.cos (π * (1 - x) / 4)
  ∃! (solutions : Finset ℝ),
    (∀ x ∈ solutions, f x = 0 ∧ 0 ≤ x ∧ x ≤ 2020) ∧
    (∀ x, f x = 0 ∧ 0 ≤ x ∧ x ≤ 2020 → x ∈ solutions) ∧
    Finset.card solutions = 505 :=
by sorry

end equation_solutions_count_l644_64471


namespace sparrow_grains_l644_64430

theorem sparrow_grains : ∃ (x : ℕ), 
  (9 * x < 1001) ∧ 
  (10 * x > 1100) ∧ 
  (x = 111) := by
sorry

end sparrow_grains_l644_64430


namespace problem_statement_inequality_statement_l644_64421

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x
def g (a x : ℝ) : ℝ := -x^2 + a*x - 3

theorem problem_statement (a : ℝ) : 
  (∀ x > 0, 2 * f x ≥ g a x) → a ≤ 4 :=
sorry

theorem inequality_statement : 
  ∀ x > 0, Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end problem_statement_inequality_statement_l644_64421


namespace bracelet_selling_price_l644_64438

def number_of_bracelets : ℕ := 12
def cost_per_bracelet : ℚ := 1
def cost_of_cookies : ℚ := 3
def money_left : ℚ := 3

def total_cost : ℚ := number_of_bracelets * cost_per_bracelet
def total_revenue : ℚ := cost_of_cookies + money_left + total_cost

theorem bracelet_selling_price :
  (total_revenue / number_of_bracelets : ℚ) = 1.75 := by
  sorry

end bracelet_selling_price_l644_64438


namespace num_fm_pairs_is_four_l644_64405

/-- The number of possible (f,m) pairs for 7 people at a round table -/
def num_fm_pairs : ℕ :=
  let people : ℕ := 7
  4

/-- Theorem: The number of possible (f,m) pairs for 7 people at a round table is 4 -/
theorem num_fm_pairs_is_four :
  num_fm_pairs = 4 := by sorry

end num_fm_pairs_is_four_l644_64405


namespace anthony_balloons_l644_64452

theorem anthony_balloons (tom_balloons luke_balloons anthony_balloons : ℕ) :
  tom_balloons = 3 * luke_balloons →
  luke_balloons = anthony_balloons / 4 →
  tom_balloons = 33 →
  anthony_balloons = 44 :=
by sorry

end anthony_balloons_l644_64452


namespace midpoint_trajectory_l644_64466

/-- Given a curve defined by 2x^2 - y = 0, prove that the midpoint of the line segment
    connecting (0, -1) and any point on the curve satisfies y = 4x^2 - 1/2 -/
theorem midpoint_trajectory (x₁ y₁ x y : ℝ) :
  (2 * x₁^2 = y₁) →  -- P(x₁, y₁) is on the curve
  (x = x₁ / 2) →     -- x-coordinate of midpoint
  (y = (y₁ - 1) / 2) -- y-coordinate of midpoint
  → y = 4 * x^2 - 1/2 := by
sorry

end midpoint_trajectory_l644_64466


namespace vector_equality_l644_64490

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, -1)
def c : ℝ × ℝ := (-1, 2)

theorem vector_equality : c = a - b := by sorry

end vector_equality_l644_64490


namespace function_min_max_values_l644_64455

/-- The function f(x) = x^3 - 3x + m has a minimum value of -1 and a maximum value of 3 -/
theorem function_min_max_values (m : ℝ) : 
  (∃ x₀ : ℝ, ∀ x : ℝ, x^3 - 3*x + m ≥ x₀^3 - 3*x₀ + m ∧ x₀^3 - 3*x₀ + m = -1) →
  (∃ x₁ : ℝ, ∀ x : ℝ, x^3 - 3*x + m ≤ x₁^3 - 3*x₁ + m ∧ x₁^3 - 3*x₁ + m = 3) :=
by sorry

end function_min_max_values_l644_64455


namespace total_available_seats_l644_64493

/-- Represents a bus with its seating configuration and broken seats -/
structure Bus where
  columns : ℕ
  rows_left : ℕ
  rows_right : ℕ
  broken_seats : ℕ

/-- Calculates the number of available seats in a bus -/
def available_seats (bus : Bus) : ℕ :=
  bus.columns * (bus.rows_left + bus.rows_right) - bus.broken_seats

/-- The list of buses with their configurations -/
def buses : List Bus := [
  ⟨4, 10, 0, 2⟩,   -- Bus 1
  ⟨5, 8, 0, 4⟩,    -- Bus 2
  ⟨3, 12, 0, 3⟩,   -- Bus 3
  ⟨4, 6, 8, 1⟩,    -- Bus 4
  ⟨6, 8, 10, 5⟩,   -- Bus 5
  ⟨5, 8, 2, 4⟩     -- Bus 6 (2 rows with 2 seats each unavailable)
]

/-- Theorem stating that the total number of available seats is 311 -/
theorem total_available_seats :
  (buses.map available_seats).sum = 311 := by
  sorry


end total_available_seats_l644_64493


namespace feed_lasts_longer_when_selling_feed_lasts_shorter_when_buying_nils_has_300_geese_l644_64467

/-- Represents the number of geese Nils currently has -/
def current_geese : ℕ := sorry

/-- Represents the number of days the feed lasts with the current number of geese -/
def current_feed_duration : ℕ := sorry

/-- Represents the amount of feed one goose consumes per day -/
def feed_per_goose_per_day : ℚ := sorry

/-- Represents the total amount of feed available -/
def total_feed : ℚ := sorry

/-- The feed lasts 20 days longer when 75 geese are sold -/
theorem feed_lasts_longer_when_selling : 
  total_feed / (feed_per_goose_per_day * (current_geese - 75)) = current_feed_duration + 20 := by sorry

/-- The feed lasts 15 days shorter when 100 geese are bought -/
theorem feed_lasts_shorter_when_buying : 
  total_feed / (feed_per_goose_per_day * (current_geese + 100)) = current_feed_duration - 15 := by sorry

/-- The main theorem proving that Nils has 300 geese -/
theorem nils_has_300_geese : current_geese = 300 := by sorry

end feed_lasts_longer_when_selling_feed_lasts_shorter_when_buying_nils_has_300_geese_l644_64467


namespace exactly_two_absent_probability_l644_64439

-- Define the probability of a student being absent
def prob_absent : ℚ := 1 / 20

-- Define the probability of a student being present
def prob_present : ℚ := 1 - prob_absent

-- Define the number of students we're considering
def num_students : ℕ := 3

-- Define the number of absent students we're looking for
def num_absent : ℕ := 2

-- Theorem statement
theorem exactly_two_absent_probability :
  (prob_absent ^ num_absent * prob_present ^ (num_students - num_absent)) * (num_students.choose num_absent) = 7125 / 1000000 := by
  sorry

end exactly_two_absent_probability_l644_64439


namespace largest_consecutive_sum_l644_64417

theorem largest_consecutive_sum (n : ℕ) (a : ℕ) (h1 : n > 1) 
  (h2 : n * a + n * (n - 1) / 2 = 2016) : 
  a + (n - 1) ≤ 673 := by
sorry

end largest_consecutive_sum_l644_64417


namespace sphere_volume_ratio_not_square_of_radii_ratio_l644_64414

theorem sphere_volume_ratio_not_square_of_radii_ratio (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) :
  (4 * π * r₁^3 / 3) / (4 * π * r₂^3 / 3) ≠ (r₁ / r₂)^2 :=
by sorry

end sphere_volume_ratio_not_square_of_radii_ratio_l644_64414


namespace no_real_roots_l644_64463

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + m) - a n = m * (a 1 - a 0)

-- Define the problem statement
theorem no_real_roots
  (a : ℕ → ℝ)
  (h_arithmetic : isArithmeticSequence a)
  (h_sum : a 2 + a 5 + a 8 = 9) :
  ∀ x : ℝ, x^2 + (a 4 + a 6) * x + 10 ≠ 0 :=
by
  sorry


end no_real_roots_l644_64463


namespace joes_lift_l644_64465

theorem joes_lift (first_lift second_lift : ℝ)
  (h1 : first_lift + second_lift = 1800)
  (h2 : 2 * first_lift = second_lift + 300) :
  first_lift = 700 := by
sorry

end joes_lift_l644_64465


namespace system_solution_l644_64497

theorem system_solution : 
  ∀ x y : ℚ, 
  x^2 - 9*y^2 = 0 ∧ x + y = 1 → 
  (x = 3/4 ∧ y = 1/4) ∨ (x = 3/2 ∧ y = -1/2) := by
sorry

end system_solution_l644_64497


namespace smallest_perfect_square_with_remainders_l644_64404

theorem smallest_perfect_square_with_remainders : ∃ n : ℕ, 
  n > 1 ∧
  n % 3 = 2 ∧
  n % 7 = 2 ∧
  n % 8 = 2 ∧
  ∃ k : ℕ, n = k^2 ∧
  ∀ m : ℕ, m > 1 → m % 3 = 2 → m % 7 = 2 → m % 8 = 2 → (∃ j : ℕ, m = j^2) → m ≥ n :=
by sorry

end smallest_perfect_square_with_remainders_l644_64404


namespace equation_one_real_root_l644_64410

theorem equation_one_real_root (t : ℝ) : 
  (∃! x : ℝ, 3 * x + 7 * t - 2 + (2 * t * x^2 + 7 * t^2 - 9) / (x - t) = 0) ↔ 
  (t = -3 ∨ t = -7/2 ∨ t = 1) := by sorry

end equation_one_real_root_l644_64410


namespace range_of_f_l644_64442

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_of_f :
  Set.range f = Set.Ioo 0 1 := by sorry

end range_of_f_l644_64442


namespace sum_of_divisors_36_l644_64415

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_36 : sum_of_divisors 36 = 91 := by
  sorry

end sum_of_divisors_36_l644_64415


namespace weight_of_new_person_l644_64478

/-- Given a group of 4 persons with a total weight W, if replacing a person
    weighing 65 kg with a new person increases the average weight by 1.5 kg,
    then the weight of the new person is 71 kg. -/
theorem weight_of_new_person (W : ℝ) : 
  (W - 65 + 71) / 4 = W / 4 + 1.5 := by sorry

end weight_of_new_person_l644_64478


namespace simplify_and_evaluate_l644_64447

theorem simplify_and_evaluate (a : ℝ) (h : a = 1 - Real.sqrt 2) :
  a * (a - 9) - (a + 3) * (a - 3) = 9 * Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l644_64447


namespace catch_up_time_is_55_minutes_l644_64460

/-- The time it takes for Bob to catch up with John -/
def catch_up_time (john_speed bob_speed initial_distance stop_time : ℚ) : ℚ :=
  let relative_speed := bob_speed - john_speed
  let time_without_stop := initial_distance / relative_speed
  (time_without_stop + stop_time / 60) * 60

theorem catch_up_time_is_55_minutes :
  catch_up_time 2 6 3 10 = 55 := by sorry

end catch_up_time_is_55_minutes_l644_64460


namespace coordinate_axes_equiv_product_zero_l644_64433

/-- The set of points on the coordinate axes in a Cartesian coordinate system -/
def CoordinateAxes : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

/-- The set of points where the product of coordinates is zero -/
def ProductZeroSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 * p.2 = 0}

theorem coordinate_axes_equiv_product_zero :
  CoordinateAxes = ProductZeroSet :=
sorry

end coordinate_axes_equiv_product_zero_l644_64433


namespace daily_wage_of_c_l644_64409

/-- Represents the daily wage and work days of a worker -/
structure Worker where
  dailyWage : ℚ
  workDays : ℕ

theorem daily_wage_of_c (a b c : Worker) 
  (ratio_a_b : a.dailyWage / b.dailyWage = 3 / 4)
  (ratio_b_c : b.dailyWage / c.dailyWage = 4 / 5)
  (work_days : a.workDays = 6 ∧ b.workDays = 9 ∧ c.workDays = 4)
  (total_earning : a.dailyWage * a.workDays + b.dailyWage * b.workDays + c.dailyWage * c.workDays = 1850) :
  c.dailyWage = 625 / 3 := by
  sorry


end daily_wage_of_c_l644_64409


namespace smallest_number_divisible_l644_64492

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < 1038239 → ¬(618 ∣ (m + 1) ∧ 3648 ∣ (m + 1) ∧ 60 ∣ (m + 1))) ∧ 
  (618 ∣ (1038239 + 1) ∧ 3648 ∣ (1038239 + 1) ∧ 60 ∣ (1038239 + 1)) := by
  sorry


end smallest_number_divisible_l644_64492


namespace shirt_price_ratio_l644_64457

theorem shirt_price_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 2 / 5
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_price : ℝ := selling_price * (4 / 5)
  cost_price / marked_price = 12 / 25 := by
sorry

end shirt_price_ratio_l644_64457


namespace least_addition_for_divisibility_l644_64436

theorem least_addition_for_divisibility : 
  (∃ x : ℕ, x ≥ 0 ∧ (228712 + x) % (2 * 3 * 5) = 0) ∧ 
  (∀ y : ℕ, y ≥ 0 ∧ (228712 + y) % (2 * 3 * 5) = 0 → y ≥ 8) :=
by sorry

end least_addition_for_divisibility_l644_64436


namespace inverse_exponential_point_l644_64487

theorem inverse_exponential_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (Function.invFun (fun x ↦ a^x) 9 = 2) → a = 3 := by
  sorry

end inverse_exponential_point_l644_64487


namespace problem_1_problem_2_problem_3_problem_4_l644_64423

-- Problem 1
theorem problem_1 : 25 - 9 + (-12) - (-7) = 4 := by sorry

-- Problem 2
theorem problem_2 : 1/9 * (-2)^3 / (2/3)^2 = -2 := by sorry

-- Problem 3
theorem problem_3 : (5/12 + 2/3 - 3/4) * (-12) = -4 := by sorry

-- Problem 4
theorem problem_4 : -1^4 + (-2) / (-1/3) - |(-9)| = -4 := by sorry

end problem_1_problem_2_problem_3_problem_4_l644_64423


namespace specific_pyramid_height_l644_64468

/-- A right pyramid with a square base -/
structure RightPyramid where
  /-- The perimeter of the square base in inches -/
  base_perimeter : ℝ
  /-- The distance from the apex to any vertex of the base in inches -/
  apex_to_vertex : ℝ

/-- The height of a right pyramid from its apex to the center of its square base -/
def pyramid_height (p : RightPyramid) : ℝ :=
  sorry

/-- Theorem stating the height of the specific pyramid -/
theorem specific_pyramid_height :
  let p := RightPyramid.mk 40 15
  pyramid_height p = 5 * Real.sqrt 7 := by
  sorry

end specific_pyramid_height_l644_64468


namespace unique_solution_condition_l644_64412

theorem unique_solution_condition (j : ℝ) : 
  (∃! x : ℝ, (2*x + 7)*(x - 5) = -43 + j*x) ↔ (j = 5 ∨ j = -11) := by
  sorry

end unique_solution_condition_l644_64412


namespace blackboard_number_increase_l644_64480

theorem blackboard_number_increase (n k : ℕ+) :
  let new_k := k + Nat.gcd k n
  (new_k - k = 1) ∨ (Nat.Prime (new_k - k)) :=
sorry

end blackboard_number_increase_l644_64480


namespace total_movies_in_five_years_l644_64479

-- Define the number of movies L&J Productions makes per year
def lj_movies_per_year : ℕ := 220

-- Define the percentage increase for Johnny TV
def johnny_tv_increase_percent : ℕ := 25

-- Define the number of years
def years : ℕ := 5

-- Statement to prove
theorem total_movies_in_five_years :
  (lj_movies_per_year + (lj_movies_per_year * johnny_tv_increase_percent) / 100 + lj_movies_per_year) * years = 2475 := by
  sorry

end total_movies_in_five_years_l644_64479


namespace cos_315_degrees_l644_64494

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_315_degrees_l644_64494


namespace zoo_animals_l644_64482

theorem zoo_animals (X : ℕ) : 
  X - 6 + 1 + 3 + 8 + 16 = 90 → X = 68 := by
sorry

end zoo_animals_l644_64482


namespace unique_zero_point_l644_64422

open Real

noncomputable def f (x : ℝ) := exp x + x - 2 * exp 1

theorem unique_zero_point :
  ∃! x, f x = 0 :=
sorry

end unique_zero_point_l644_64422


namespace orange_difference_l644_64475

/-- The number of oranges and apples picked by George and Amelia -/
structure FruitPicking where
  george_oranges : ℕ
  george_apples : ℕ
  amelia_oranges : ℕ
  amelia_apples : ℕ

/-- The conditions of the fruit picking problem -/
def fruit_picking_conditions (fp : FruitPicking) : Prop :=
  fp.george_oranges = 45 ∧
  fp.george_apples = fp.amelia_apples + 5 ∧
  fp.amelia_oranges < fp.george_oranges ∧
  fp.amelia_apples = 15 ∧
  fp.george_oranges + fp.george_apples + fp.amelia_oranges + fp.amelia_apples = 107

/-- The theorem stating the difference in orange count -/
theorem orange_difference (fp : FruitPicking) 
  (h : fruit_picking_conditions fp) : 
  fp.george_oranges - fp.amelia_oranges = 18 := by
  sorry

end orange_difference_l644_64475


namespace blocks_with_two_differences_eq_28_l644_64418

/-- Represents the number of options for each category of block attributes -/
structure BlockCategories where
  materials : Nat
  sizes : Nat
  colors : Nat
  shapes : Nat

/-- Calculates the number of blocks differing in exactly two ways from a reference block -/
def blocksWithTwoDifferences (categories : BlockCategories) : Nat :=
  sorry

/-- The specific categories for the given problem -/
def problemCategories : BlockCategories :=
  { materials := 2
  , sizes := 3
  , colors := 5
  , shapes := 4
  }

/-- Theorem stating that the number of blocks differing in exactly two ways is 28 -/
theorem blocks_with_two_differences_eq_28 :
  blocksWithTwoDifferences problemCategories = 28 := by
  sorry

end blocks_with_two_differences_eq_28_l644_64418


namespace max_value_a_l644_64400

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 4 * d)
  (h4 : d < 100) :
  a ≤ 2367 ∧ ∃ (a' b' c' d' : ℕ+), a' = 2367 ∧ 
    a' < 2 * b' ∧ b' < 3 * c' ∧ c' < 4 * d' ∧ d' < 100 :=
by sorry

end max_value_a_l644_64400


namespace not_exp_ix_always_one_l644_64485

open Complex

theorem not_exp_ix_always_one (x : ℝ) : ¬ ∀ x, exp (I * x) = 1 := by
  sorry

/-- e^(ix) is a periodic function with period 2π -/
axiom exp_ix_periodic : ∀ x : ℝ, exp (I * x) = exp (I * (x + 2 * Real.pi))

/-- e^(ix) = e^(i(x + 2πk)) for any integer k -/
axiom exp_ix_shift : ∀ (x : ℝ) (k : ℤ), exp (I * x) = exp (I * (x + 2 * Real.pi * ↑k))

end not_exp_ix_always_one_l644_64485


namespace inequality_theorem_l644_64448

theorem inequality_theorem (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  c / a > c / b :=
by sorry

end inequality_theorem_l644_64448


namespace simplify_and_evaluate_expression_l644_64476

theorem simplify_and_evaluate_expression (m : ℚ) (h : m = 5) :
  (m + 2 - 5 / (m - 2)) / ((3 * m - m^2) / (m - 2)) = -8 / 5 := by
  sorry

end simplify_and_evaluate_expression_l644_64476


namespace no_money_left_l644_64461

theorem no_money_left (total_money : ℝ) (total_items : ℝ) (h1 : total_money > 0) (h2 : total_items > 0) :
  (1 / 3 : ℝ) * total_money = (1 / 3 : ℝ) * total_items * (total_money / total_items) →
  total_money - total_items * (total_money / total_items) = 0 := by
sorry

end no_money_left_l644_64461


namespace pricing_theorem_l644_64431

/-- Proves that for an item with a marked price 50% above its cost price,
    a discount of 23.33% on the marked price results in a 15% profit,
    and the final selling price is 115% of the cost price. -/
theorem pricing_theorem (cost_price : ℝ) (cost_price_pos : cost_price > 0) :
  let marked_price := cost_price * 1.5
  let discount_percentage := 23.33 / 100
  let selling_price := marked_price * (1 - discount_percentage)
  selling_price = cost_price * 1.15 ∧ 
  (selling_price - cost_price) / cost_price = 0.15 := by
  sorry

#check pricing_theorem

end pricing_theorem_l644_64431


namespace isosceles_triangle_perimeter_l644_64441

/-- An isosceles triangle with two sides of length 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = b ∧ (a = 3 ∧ c = 7 ∨ a = 7 ∧ c = 3)) →
  a + b + c = 17 :=
by sorry

end isosceles_triangle_perimeter_l644_64441


namespace percentage_equation_l644_64474

theorem percentage_equation (x : ℝ) : (65 / 100 * x = 20 / 100 * 747.50) → x = 230 := by
  sorry

end percentage_equation_l644_64474


namespace sum_of_prime_factors_l644_64499

theorem sum_of_prime_factors (n : ℕ) : 
  n > 0 ∧ n < 1000 ∧ (∃ k : ℤ, 42 * n = 180 * k) →
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id = 10) :=
by sorry

end sum_of_prime_factors_l644_64499


namespace lines_parallel_to_same_line_are_parallel_l644_64491

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_to_same_line_are_parallel
  (a b c : Line) :
  parallel a c → parallel b c → parallel a b :=
sorry

end lines_parallel_to_same_line_are_parallel_l644_64491


namespace point_order_on_line_l644_64449

/-- Given points on a line, prove their y-coordinates are ordered. -/
theorem point_order_on_line (b : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 3 * (-3) - b)
  (h₂ : y₂ = 3 * 1 - b)
  (h₃ : y₃ = 3 * (-1) - b) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end point_order_on_line_l644_64449


namespace three_primes_sum_l644_64403

theorem three_primes_sum (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r →
  p * q * r = 31 * (p + q + r) →
  p + q + r = 51 := by
sorry

end three_primes_sum_l644_64403


namespace store_max_profit_l644_64495

/-- Represents the profit function for a store selling seasonal goods -/
def profit_function (x : ℝ) : ℝ := -5 * x^2 + 500 * x + 20000

/-- The maximum profit achieved by the store -/
def max_profit : ℝ := 32500

theorem store_max_profit :
  ∃ (x : ℝ), 
    (∀ y : ℝ, profit_function y ≤ profit_function x) ∧ 
    profit_function x = max_profit :=
by sorry

end store_max_profit_l644_64495


namespace dress_count_proof_l644_64434

def total_dresses (emily melissa debora sophia : ℕ) : ℕ :=
  emily + melissa + debora + sophia

theorem dress_count_proof 
  (emily : ℕ) 
  (h_emily : emily = 16)
  (melissa : ℕ) 
  (h_melissa : melissa = emily / 2)
  (debora : ℕ)
  (h_debora : debora = melissa + 12)
  (sophia : ℕ)
  (h_sophia : sophia = debora * 3 / 4) :
  total_dresses emily melissa debora sophia = 59 := by
sorry

end dress_count_proof_l644_64434


namespace points_earned_l644_64472

def points_per_enemy : ℕ := 3
def total_enemies : ℕ := 6
def enemies_not_defeated : ℕ := 2

theorem points_earned : 
  (total_enemies - enemies_not_defeated) * points_per_enemy = 12 := by
  sorry

end points_earned_l644_64472


namespace rectangle_area_theorem_l644_64458

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

def Rectangle.area (r : Rectangle) : ℝ := sorry

def angle (p1 p2 p3 : Point) : ℝ := sorry

def distance (p1 p2 : Point) : ℝ := sorry

def foldPoint (p : Point) (line : Point × Point) : Point := sorry

theorem rectangle_area_theorem (ABCD : Rectangle) (E F : Point) (B' C' : Point) :
  E.x = ABCD.A.x ∧ F.x = ABCD.D.x →
  distance ABCD.B E < distance ABCD.C F →
  B' = foldPoint ABCD.B (E, F) →
  C' = foldPoint ABCD.C (E, F) →
  C'.x = ABCD.A.x →
  angle ABCD.A B' C' = 2 * angle B' E ABCD.A →
  distance ABCD.A B' = 8 →
  distance ABCD.B E = 15 →
  ∃ (a b c : ℕ), 
    Rectangle.area ABCD = a + b * Real.sqrt c ∧
    a = 100 ∧ b = 4 ∧ c = 23 ∧
    a + b + c = 127 ∧
    ∀ (p : ℕ), Prime p → c % (p * p) ≠ 0 :=
by sorry

end rectangle_area_theorem_l644_64458


namespace infinitely_many_triangular_pentagonal_pairs_l644_64459

/-- A pair of positive integers (n, m) is a triangular-pentagonal pair if n(n+1) = m(3m-1) -/
def IsTriangularPentagonalPair (n m : ℕ) : Prop :=
  n > 0 ∧ m > 0 ∧ n * (n + 1) = m * (3 * m - 1)

/-- There exist infinitely many triangular-pentagonal pairs -/
theorem infinitely_many_triangular_pentagonal_pairs :
  ∀ k : ℕ, ∃ n m : ℕ, n > k ∧ m > k ∧ IsTriangularPentagonalPair n m :=
sorry

end infinitely_many_triangular_pentagonal_pairs_l644_64459


namespace complex_number_problem_l644_64456

def complex_i : ℂ := Complex.I

theorem complex_number_problem (z₁ z₂ : ℂ) 
  (h1 : (z₁ - 2) * (1 + complex_i) = 1 - complex_i)
  (h2 : z₂.im = 2)
  (h3 : (z₁ * z₂).im = 0) :
  z₁ = 2 - complex_i ∧ z₂ = 4 + 2 * complex_i := by
  sorry

end complex_number_problem_l644_64456


namespace alice_wins_second_attempt_prob_l644_64453

-- Define the number of cards in the deck
def deckSize : ℕ := 20

-- Define the probability of a correct guess in each turn
def probFirst : ℚ := 1 / deckSize
def probSecond : ℚ := 1 / (deckSize - 1)
def probThird : ℚ := 1 / (deckSize - 2)

-- Define the probability of Alice winning on her second attempt
def aliceWinsSecondAttempt : ℚ := (1 - probFirst) * (1 - probSecond) * probThird

-- Theorem to prove
theorem alice_wins_second_attempt_prob :
  aliceWinsSecondAttempt = 1 / deckSize := by
  sorry


end alice_wins_second_attempt_prob_l644_64453


namespace number_puzzle_l644_64427

theorem number_puzzle : ∃ x : ℝ, (2 * x) / 16 = 25 ∧ x = 200 := by sorry

end number_puzzle_l644_64427
