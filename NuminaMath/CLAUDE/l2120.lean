import Mathlib

namespace file_download_rate_l2120_212038

/-- Given a file download scenario, prove the download rate for the latter part. -/
theorem file_download_rate 
  (file_size : ℝ) 
  (initial_rate : ℝ) 
  (initial_size : ℝ) 
  (total_time : ℝ) 
  (h1 : file_size = 90) 
  (h2 : initial_rate = 5) 
  (h3 : initial_size = 60) 
  (h4 : total_time = 15) : 
  (file_size - initial_size) / (total_time - initial_size / initial_rate) = 10 := by
  sorry

end file_download_rate_l2120_212038


namespace sufficient_not_necessary_l2120_212057

theorem sufficient_not_necessary (a b : ℝ) :
  (a > 2 ∧ b > 1 → a + b > 3 ∧ a * b > 2) ∧
  ∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 2 ∧ b > 1) :=
by sorry

end sufficient_not_necessary_l2120_212057


namespace cubic_inequality_l2120_212053

theorem cubic_inequality (a b c : ℝ) :
  a^6 + b^6 + c^6 - 3*a^2*b^2*c^2 ≥ (1/2) * (a-b)^2 * (b-c)^2 * (c-a)^2 := by
  sorry

end cubic_inequality_l2120_212053


namespace fraction_value_l2120_212067

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 2 * d) : 
  (a * c) / (b * d) = 8 := by
sorry

end fraction_value_l2120_212067


namespace bed_sheet_problem_l2120_212026

/-- Calculates the length of a bed-sheet in meters given the total cutting time, 
    time per cut, and length of each piece. -/
def bed_sheet_length (total_time : ℕ) (time_per_cut : ℕ) (piece_length : ℕ) : ℚ :=
  (total_time / time_per_cut) * piece_length / 100

/-- Proves that a bed-sheet cut into 20cm pieces, taking 5 minutes per cut and 
    245 minutes total, is 9.8 meters long. -/
theorem bed_sheet_problem : bed_sheet_length 245 5 20 = 9.8 := by
  sorry

#eval bed_sheet_length 245 5 20

end bed_sheet_problem_l2120_212026


namespace salary_change_l2120_212037

theorem salary_change (original : ℝ) (h : original > 0) :
  ∃ (increase : ℝ),
    (original * 0.7 * (1 + increase) = original * 0.91) ∧
    increase = 0.3 := by
  sorry

end salary_change_l2120_212037


namespace phone_bill_percentage_abigail_phone_bill_l2120_212045

theorem phone_bill_percentage (initial_amount : ℝ) (food_percentage : ℝ) 
  (entertainment_cost : ℝ) (final_amount : ℝ) : ℝ :=
  let food_cost := initial_amount * food_percentage
  let after_food := initial_amount - food_cost
  let before_phone_bill := after_food - entertainment_cost
  let phone_bill_cost := before_phone_bill - final_amount
  let phone_bill_percentage := (phone_bill_cost / after_food) * 100
  phone_bill_percentage

theorem abigail_phone_bill : 
  phone_bill_percentage 200 0.6 20 40 = 25 := by
  sorry

end phone_bill_percentage_abigail_phone_bill_l2120_212045


namespace correct_reasoning_combination_l2120_212032

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| PartToWhole
| GeneralToSpecific
| SpecificToSpecific

-- Define a function that maps reasoning types to their directions
def reasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating that the correct combination is Inductive, Deductive, and Analogical
theorem correct_reasoning_combination :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
sorry

end correct_reasoning_combination_l2120_212032


namespace difference_of_squares_division_l2120_212016

theorem difference_of_squares_division : (315^2 - 291^2) / 24 = 606 := by
  sorry

end difference_of_squares_division_l2120_212016


namespace shifted_function_l2120_212020

def g (x : ℝ) : ℝ := 5 * x^2

def f (x : ℝ) : ℝ := 5 * (x - 3)^2 - 2

theorem shifted_function (x : ℝ) : 
  f x = g (x - 3) - 2 := by sorry

end shifted_function_l2120_212020


namespace composition_result_l2120_212048

/-- Given two functions f and g, prove that f(g(2)) = 169 -/
theorem composition_result (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = x^2)
  (hg : ∀ x, g x = 2*x^2 + x + 3) : 
  f (g 2) = 169 := by
  sorry

end composition_result_l2120_212048


namespace difference_of_squares_l2120_212003

theorem difference_of_squares (x y : ℝ) 
  (sum_eq : x + y = 20) 
  (diff_eq : x - y = 8) : 
  x^2 - y^2 = 160 := by
sorry

end difference_of_squares_l2120_212003


namespace imaginary_unit_power_sum_l2120_212061

theorem imaginary_unit_power_sum : ∀ i : ℂ, i^2 = -1 →
  i^15300 + i^15301 + i^15302 + i^15303 + i^15304 = 1 := by
  sorry

end imaginary_unit_power_sum_l2120_212061


namespace sum_of_triple_products_of_roots_l2120_212064

theorem sum_of_triple_products_of_roots (p q r s : ℂ) : 
  (4 * p^4 - 8 * p^3 + 18 * p^2 - 14 * p + 7 = 0) →
  (4 * q^4 - 8 * q^3 + 18 * q^2 - 14 * q + 7 = 0) →
  (4 * r^4 - 8 * r^3 + 18 * r^2 - 14 * r + 7 = 0) →
  (4 * s^4 - 8 * s^3 + 18 * s^2 - 14 * s + 7 = 0) →
  p * q * r + p * q * s + p * r * s + q * r * s = 7 / 2 := by
sorry

end sum_of_triple_products_of_roots_l2120_212064


namespace blue_paint_calculation_l2120_212079

/-- Given a ratio of blue paint to white paint and the amount of white paint used,
    calculate the amount of blue paint required. -/
def blue_paint_required (blue_ratio : ℚ) (white_ratio : ℚ) (white_amount : ℚ) : ℚ :=
  (blue_ratio / white_ratio) * white_amount

/-- Theorem stating that given a 5:6 ratio of blue to white paint and 18 quarts of white paint,
    15 quarts of blue paint are required. -/
theorem blue_paint_calculation :
  let blue_ratio : ℚ := 5
  let white_ratio : ℚ := 6
  let white_amount : ℚ := 18
  blue_paint_required blue_ratio white_ratio white_amount = 15 := by
  sorry

end blue_paint_calculation_l2120_212079


namespace s13_is_constant_l2120_212068

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Given S_4 + a_25 = 5, S_13 is constant -/
theorem s13_is_constant (seq : ArithmeticSequence) 
    (h : seq.S 4 + seq.a 25 = 5) : 
  ∃ c : ℝ, seq.S 13 = c := by
  sorry

end s13_is_constant_l2120_212068


namespace two_yellow_marbles_prob_l2120_212031

/-- Represents the number of marbles of each color in the box -/
structure MarbleBox where
  blue : ℕ
  yellow : ℕ
  orange : ℕ

/-- Calculates the total number of marbles in the box -/
def totalMarbles (box : MarbleBox) : ℕ :=
  box.blue + box.yellow + box.orange

/-- Calculates the probability of drawing a yellow marble -/
def probYellow (box : MarbleBox) : ℚ :=
  box.yellow / (totalMarbles box)

/-- The probability of drawing two yellow marbles in succession with replacement -/
def probTwoYellow (box : MarbleBox) : ℚ :=
  (probYellow box) * (probYellow box)

theorem two_yellow_marbles_prob :
  let box : MarbleBox := { blue := 4, yellow := 5, orange := 6 }
  probTwoYellow box = 1 / 9 := by
  sorry

end two_yellow_marbles_prob_l2120_212031


namespace geometric_sequence_a2_l2120_212098

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Theorem: In a geometric sequence where a_7 = 1/4 and a_3 * a_5 = 4(a_4 - 1), a_2 = 8 -/
theorem geometric_sequence_a2 (a : ℕ → ℚ) 
    (h_geom : GeometricSequence a) 
    (h_a7 : a 7 = 1/4) 
    (h_a3a5 : a 3 * a 5 = 4 * (a 4 - 1)) : 
  a 2 = 8 := by
  sorry

end geometric_sequence_a2_l2120_212098


namespace triangle_rotation_l2120_212039

theorem triangle_rotation (a₁ a₂ a₃ : ℝ) (h1 : 12 * a₁ = 360) (h2 : 6 * a₂ = 360) (h3 : a₁ + a₂ + a₃ = 180) :
  ∃ n : ℕ, n * a₃ ≥ 360 ∧ ∀ m : ℕ, m * a₃ ≥ 360 → n ≤ m ∧ n = 4 :=
sorry

end triangle_rotation_l2120_212039


namespace factor_tree_X_value_l2120_212025

theorem factor_tree_X_value :
  ∀ (X Y Z F G : ℕ),
    X = Y * Z →
    Y = 7 * F →
    Z = 11 * G →
    F = 7 * 3 →
    G = 11 * 3 →
    X = 53361 := by
  sorry

end factor_tree_X_value_l2120_212025


namespace special_function_property_l2120_212012

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (f 2 = 2) ∧ 
  (∀ x y : ℝ, f (x * y + f x) = x * f y + f x + x^2)

/-- The number of possible values for f(1/2) -/
def num_values (f : ℝ → ℝ) : ℕ :=
  sorry

/-- The sum of all possible values for f(1/2) -/
def sum_values (f : ℝ → ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  (num_values f : ℝ) * sum_values f = -2 :=
sorry

end special_function_property_l2120_212012


namespace quadratic_expression_equality_l2120_212027

theorem quadratic_expression_equality (x y : ℝ) 
  (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 
  10 * x^2 + 19 * x * y + 10 * y^2 = 153 := by sorry

end quadratic_expression_equality_l2120_212027


namespace max_harmonious_t_patterns_l2120_212097

/-- Represents a coloring of an 8x8 grid --/
def Coloring := Fin 8 → Fin 8 → Bool

/-- Represents a T-shaped pattern in the grid --/
structure TPattern where
  row : Fin 8
  col : Fin 8
  orientation : Fin 4

/-- The total number of T-shaped patterns in an 8x8 grid --/
def total_t_patterns : Nat := 168

/-- Checks if a T-pattern is harmonious under a given coloring --/
def is_harmonious (c : Coloring) (t : TPattern) : Bool :=
  sorry

/-- Counts the number of harmonious T-patterns for a given coloring --/
def count_harmonious (c : Coloring) : Nat :=
  sorry

/-- The maximum number of harmonious T-patterns possible --/
def max_harmonious : Nat := 132

theorem max_harmonious_t_patterns :
  ∃ (c : Coloring), count_harmonious c = max_harmonious ∧
  ∀ (c' : Coloring), count_harmonious c' ≤ max_harmonious :=
sorry

end max_harmonious_t_patterns_l2120_212097


namespace base_number_proof_l2120_212077

theorem base_number_proof (x : ℝ) (n : ℕ) 
  (h1 : 4 * x^(2*n) = 4^18) (h2 : n = 17) : x = 2 := by
  sorry

end base_number_proof_l2120_212077


namespace mirror_pieces_l2120_212002

theorem mirror_pieces : ∃ P : ℕ, 
  (P > 0) ∧ 
  (P / 2 - 3 > 0) ∧
  ((P / 2 - 3) / 3 = 9) ∧
  (P = 60) := by
  sorry

end mirror_pieces_l2120_212002


namespace cakes_dinner_today_l2120_212099

def cakes_lunch_today : ℕ := 5
def cakes_yesterday : ℕ := 3
def total_cakes : ℕ := 14

theorem cakes_dinner_today : ∃ x : ℕ, x = total_cakes - cakes_lunch_today - cakes_yesterday ∧ x = 6 := by
  sorry

end cakes_dinner_today_l2120_212099


namespace second_investment_rate_l2120_212059

def contest_winnings : ℝ := 5000
def first_investment : ℝ := 1800
def first_interest_rate : ℝ := 0.05
def total_interest : ℝ := 298

def second_investment : ℝ := 2 * first_investment - 400

def first_interest : ℝ := first_investment * first_interest_rate

def second_interest : ℝ := total_interest - first_interest

theorem second_investment_rate (second_rate : ℝ) : 
  second_rate * second_investment = second_interest → second_rate = 0.065 := by
  sorry

end second_investment_rate_l2120_212059


namespace circle_equation_implies_y_to_x_equals_nine_l2120_212051

theorem circle_equation_implies_y_to_x_equals_nine (x y : ℝ) : 
  x^2 + y^2 - 4*x + 6*y + 13 = 0 → y^x = 9 := by
  sorry

end circle_equation_implies_y_to_x_equals_nine_l2120_212051


namespace quadratic_rational_solutions_l2120_212046

theorem quadratic_rational_solutions : 
  ∃! (c₁ c₂ : ℕ+), 
    (∃ (x : ℚ), 7 * x^2 + 15 * x + c₁.val = 0) ∧ 
    (∃ (y : ℚ), 7 * y^2 + 15 * y + c₂.val = 0) ∧ 
    c₁ ≠ c₂ ∧ 
    c₁.val * c₂.val = 8 := by
  sorry

end quadratic_rational_solutions_l2120_212046


namespace expression_evaluation_l2120_212081

theorem expression_evaluation (x y z : ℝ) (hx : x = -6) (hy : y = -3) (hz : z = 1/2) :
  4 * z * (x - y)^2 - (x * z) / y + 3 * Real.sin (y * z) = 17 + 3 * Real.sin (-3/2) := by
  sorry

end expression_evaluation_l2120_212081


namespace negation_existence_to_universal_negation_of_existence_proposition_l2120_212056

theorem negation_existence_to_universal (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) := by sorry

end negation_existence_to_universal_negation_of_existence_proposition_l2120_212056


namespace sled_total_distance_l2120_212083

/-- The distance traveled by a sled in n seconds, given initial distance and acceleration -/
def sledDistance (initialDistance : ℕ) (acceleration : ℕ) (n : ℕ) : ℕ :=
  n * (2 * initialDistance + (n - 1) * acceleration) / 2

/-- Theorem stating the total distance traveled by the sled -/
theorem sled_total_distance :
  sledDistance 8 10 40 = 8120 := by
  sorry

end sled_total_distance_l2120_212083


namespace triangle_area_l2120_212055

theorem triangle_area (A B C : Real) (a b c : Real) : 
  -- Triangle ABC exists with sides a, b, c opposite to angles A, B, C
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  -- Given conditions
  (Real.cos B = 1/4) →
  (b = 3) →
  (Real.sin C = 2 * Real.sin A) →
  -- Conclusion
  (1/2 * a * c * Real.sin B = Real.sqrt 15 / 4) := by
sorry

end triangle_area_l2120_212055


namespace smallest_x_abs_equation_l2120_212075

theorem smallest_x_abs_equation : 
  (∃ x : ℝ, |x - 8| = 9) ∧ (∀ x : ℝ, |x - 8| = 9 → x ≥ -1) ∧ |-1 - 8| = 9 := by
  sorry

end smallest_x_abs_equation_l2120_212075


namespace proposition_1_proposition_2_proposition_3_proposition_4_l2120_212076

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (parallelPlanes : Plane → Plane → Prop)

-- Notation
local notation:50 l1:50 " ⊥ " l2:50 => perpendicular l1 l2
local notation:50 l1:50 " ∥ " l2:50 => parallel l1 l2
local notation:50 l:50 " ⊥ " p:50 => perpendicularToPlane l p
local notation:50 l:50 " ∥ " p:50 => parallelToPlane l p
local notation:50 p1:50 " ⊥ " p2:50 => perpendicularPlanes p1 p2
local notation:50 p1:50 " ∥ " p2:50 => parallelPlanes p1 p2

-- Theorem statements
theorem proposition_1 (m n : Line) (α β : Plane) :
  (m ⊥ α) → (n ⊥ β) → (m ⊥ n) → (α ⊥ β) := by sorry

theorem proposition_2 (m n : Line) (α β : Plane) :
  ¬ ((m ∥ α) → (n ∥ β) → (m ∥ n) → (α ∥ β)) := by sorry

theorem proposition_3 (m n : Line) (α β : Plane) :
  ¬ ((m ⊥ α) → (n ∥ β) → (m ⊥ n) → (α ⊥ β)) := by sorry

theorem proposition_4 (m n : Line) (α β : Plane) :
  (m ⊥ α) → (n ∥ β) → (m ∥ n) → (α ⊥ β) := by sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l2120_212076


namespace equation_solutions_l2120_212030

theorem equation_solutions :
  (∃ x : ℚ, x - 2 * (x - 4) = 3 * (1 - x) ∧ x = -5/2) ∧
  (∃ x : ℚ, (2 * x + 1) / 3 - (5 * x - 1) / 60 = 1 ∧ x = 39/35) :=
by sorry

end equation_solutions_l2120_212030


namespace solution_equivalence_l2120_212036

theorem solution_equivalence :
  (∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2) :=
by sorry

end solution_equivalence_l2120_212036


namespace sum_of_cube_difference_l2120_212049

theorem sum_of_cube_difference (a b c : ℕ+) :
  (a + b + c : ℕ+)^3 - a^3 - b^3 - c^3 = 210 →
  (a : ℕ) + b + c = 11 := by
  sorry

end sum_of_cube_difference_l2120_212049


namespace first_term_of_constant_ratio_l2120_212088

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℕ) * d) / 2

theorem first_term_of_constant_ratio (a : ℚ) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → arithmetic_sum a 5 (3 * n) / arithmetic_sum a 5 n = c) →
  a = 5 / 2 :=
by sorry

end first_term_of_constant_ratio_l2120_212088


namespace cuboid_ratio_simplification_l2120_212008

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℕ
  breadth : ℕ
  height : ℕ

/-- Calculates the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Simplifies the ratio of three numbers by dividing each by their GCD -/
def simplifyRatio (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
  let d := gcd3 a b c
  (a / d, b / d, c / d)

/-- The main theorem stating that the given cuboid dimensions simplify to the ratio 6:5:4 -/
theorem cuboid_ratio_simplification (c : CuboidDimensions) 
  (h1 : c.length = 90) 
  (h2 : c.breadth = 75) 
  (h3 : c.height = 60) : 
  simplifyRatio c.length c.breadth c.height = (6, 5, 4) := by
  sorry

end cuboid_ratio_simplification_l2120_212008


namespace conference_games_l2120_212071

theorem conference_games (total_teams : Nat) (divisions : Nat) (teams_per_division : Nat)
  (h1 : total_teams = 12)
  (h2 : divisions = 3)
  (h3 : teams_per_division = 4)
  (h4 : total_teams = divisions * teams_per_division) :
  (total_teams * (3 * (teams_per_division - 1) + 2 * (total_teams - teams_per_division))) / 2 = 84 := by
  sorry

end conference_games_l2120_212071


namespace theater_attendance_l2120_212017

/-- The number of men who spent Rs. 3 each on tickets -/
def num_men_standard : ℕ := 8

/-- The amount spent by each of the standard-paying men -/
def standard_price : ℚ := 3

/-- The total amount spent by all men -/
def total_spent : ℚ := 29.25

/-- The extra amount spent by the last man compared to the average -/
def extra_spent : ℚ := 2

theorem theater_attendance :
  ∃ (n : ℕ), n > 0 ∧
  (n : ℚ) * (total_spent / n) = 
    num_men_standard * standard_price + (total_spent / n + extra_spent) ∧
  n = 9 := by
sorry

end theater_attendance_l2120_212017


namespace right_triangle_area_l2120_212023

/-- The area of a right triangle with base 12 cm and height 15 cm is 90 square centimeters. -/
theorem right_triangle_area (base height area : ℝ) : 
  base = 12 → height = 15 → area = (1 / 2) * base * height → area = 90 := by
  sorry

end right_triangle_area_l2120_212023


namespace arithmetic_sequence_fifth_term_l2120_212082

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 2 + a 7 = 18)
  (h_fourth : a 4 = 3) :
  a 5 = 15 := by
sorry

end arithmetic_sequence_fifth_term_l2120_212082


namespace quadratic_roots_sum_of_squares_l2120_212035

theorem quadratic_roots_sum_of_squares (m : ℝ) 
  (h1 : ∃ x₁ x₂ : ℝ, x₁^2 - m*x₁ + 2*m - 1 = 0 ∧ x₂^2 - m*x₂ + 2*m - 1 = 0)
  (h2 : ∃ x₁ x₂ : ℝ, x₁^2 + x₂^2 = 23 ∧ x₁^2 - m*x₁ + 2*m - 1 = 0 ∧ x₂^2 - m*x₂ + 2*m - 1 = 0) :
  m = -3 := by
sorry

end quadratic_roots_sum_of_squares_l2120_212035


namespace consecutive_integers_product_812_sum_57_l2120_212080

theorem consecutive_integers_product_812_sum_57 :
  ∀ x y : ℕ,
    x > 0 →
    y = x + 1 →
    x * y = 812 →
    x + y = 57 := by
  sorry

end consecutive_integers_product_812_sum_57_l2120_212080


namespace inequality_proof_l2120_212001

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt ((z + x) * (z + y)) - z ≥ Real.sqrt (x * y) := by
  sorry

end inequality_proof_l2120_212001


namespace athletic_groups_l2120_212022

/-- Given a number of athletes and groups satisfying certain conditions,
    prove that there are 59 athletes and 8 groups. -/
theorem athletic_groups (x y : ℤ) 
  (eq1 : 7 * y + 3 = x) 
  (eq2 : 8 * y - 5 = x) : 
  x = 59 ∧ y = 8 := by
  sorry

end athletic_groups_l2120_212022


namespace sin_sum_product_l2120_212042

theorem sin_sum_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end sin_sum_product_l2120_212042


namespace circle_equation_proof_l2120_212040

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line on which the center of the required circle lies
def centerLine (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

-- Define the equation of the required circle
def requiredCircle (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 13

-- Theorem statement
theorem circle_equation_proof :
  ∀ x y : ℝ,
  (circle1 x y ∧ circle2 x y) →
  ∃ h k : ℝ,
  centerLine h k ∧
  requiredCircle x y ∧
  (x - h)^2 + (y - k)^2 = (x + 1)^2 + (y - 1)^2 :=
sorry

end circle_equation_proof_l2120_212040


namespace brendas_age_l2120_212029

theorem brendas_age (addison janet brenda : ℝ) 
  (h1 : addison = 4 * brenda) 
  (h2 : janet = brenda + 10) 
  (h3 : addison = janet) : 
  brenda = 10 / 3 := by
sorry

end brendas_age_l2120_212029


namespace sine_HAC_specific_prism_l2120_212069

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  a : ℝ  -- length
  b : ℝ  -- width
  c : ℝ  -- height
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D
  F : Point3D
  G : Point3D
  H : Point3D

/-- Calculate the sine of the angle HAC in a rectangular prism -/
def sineHAC (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem: The sine of angle HAC in the given rectangular prism is √143 / 13 -/
theorem sine_HAC_specific_prism :
  let prism : RectangularPrism := {
    a := 2,
    b := 2,
    c := 3,
    A := ⟨0, 0, 0⟩,
    B := ⟨2, 0, 0⟩,
    C := ⟨2, 2, 0⟩,
    D := ⟨0, 2, 0⟩,
    E := ⟨0, 0, 3⟩,
    F := ⟨2, 0, 3⟩,
    G := ⟨2, 2, 3⟩,
    H := ⟨0, 2, 3⟩
  }
  sineHAC prism = Real.sqrt 143 / 13 := by
  sorry

end sine_HAC_specific_prism_l2120_212069


namespace savings_calculation_l2120_212066

/-- Calculates the savings of a person given their income and the ratio of income to expenditure -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given a person's income of 14000 and income to expenditure ratio of 7:6, their savings are 2000 -/
theorem savings_calculation :
  calculate_savings 14000 7 6 = 2000 := by
  sorry

end savings_calculation_l2120_212066


namespace jane_sequin_count_l2120_212047

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ n d : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

/-- Represents the sequin count problem for Jane's costume -/
def sequinCount : Prop :=
  let blueStars := 10 * 12
  let purpleSquares := 8 * 15
  let greenHexagons := 14 * 20
  let redCircles := arithmeticSum 10 5 5
  blueStars + purpleSquares + greenHexagons + redCircles = 620

theorem jane_sequin_count : sequinCount := by
  sorry

end jane_sequin_count_l2120_212047


namespace complex_root_modulus_one_iff_divisible_by_six_l2120_212034

theorem complex_root_modulus_one_iff_divisible_by_six (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (n + 2) % 6 = 0 := by
  sorry

end complex_root_modulus_one_iff_divisible_by_six_l2120_212034


namespace polygon_25_diagonals_l2120_212096

/-- The number of diagonals in a convex polygon with n sides,
    where each vertex connects only to vertices at least k places apart. -/
def diagonals (n : ℕ) (k : ℕ) : ℕ :=
  (n * (n - (2*k + 1))) / 2

/-- Theorem: A convex 25-sided polygon where each vertex connects only to
    vertices at least 2 places apart in sequence has 250 diagonals. -/
theorem polygon_25_diagonals :
  diagonals 25 2 = 250 := by
  sorry

end polygon_25_diagonals_l2120_212096


namespace diamond_value_l2120_212091

/-- The diamond operation for non-zero integers -/
def diamond (a b : ℤ) : ℚ := (1 : ℚ) / a + (2 : ℚ) / b

/-- Theorem stating the value of a ◇ b given the conditions -/
theorem diamond_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 10) (h4 : a * b = 24) :
  diamond a b = 2 / 3 := by
  sorry

end diamond_value_l2120_212091


namespace metallic_sheet_length_l2120_212021

/-- The length of a rectangular metallic sheet, given its width and the dimensions of an open box formed from it. -/
theorem metallic_sheet_length (w h v : ℝ) (hw : w = 36) (hh : h = 8) (hv : v = 5440) : ∃ l : ℝ,
  l = 50 ∧ v = (l - 2 * h) * (w - 2 * h) * h :=
by sorry

end metallic_sheet_length_l2120_212021


namespace necessary_and_sufficient_condition_l2120_212014

/-- The universal set U is the set of positive integers less than or equal to a -/
def U (a : ℝ) : Set ℕ := {x : ℕ | x > 0 ∧ x ≤ ⌊a⌋}

/-- Set P -/
def P : Set ℕ := {1, 2, 3}

/-- Set Q -/
def Q : Set ℕ := {4, 5, 6}

/-- The complement of set A in the universal set U -/
def complement (a : ℝ) (A : Set ℕ) : Set ℕ := (U a) \ A

theorem necessary_and_sufficient_condition (a : ℝ) :
  (6 ≤ a ∧ a < 7) ↔ complement a P = Q := by sorry

end necessary_and_sufficient_condition_l2120_212014


namespace bolt_width_calculation_l2120_212086

/-- The width of a bolt of fabric given specific cuts and remaining area --/
theorem bolt_width_calculation (living_room_length living_room_width bedroom_length bedroom_width bolt_length remaining_fabric : ℝ) 
  (h1 : living_room_length = 4)
  (h2 : living_room_width = 6)
  (h3 : bedroom_length = 2)
  (h4 : bedroom_width = 4)
  (h5 : bolt_length = 12)
  (h6 : remaining_fabric = 160) :
  (remaining_fabric + living_room_length * living_room_width + bedroom_length * bedroom_width) / bolt_length = 16 := by
  sorry

end bolt_width_calculation_l2120_212086


namespace meaningful_expression_range_l2120_212054

theorem meaningful_expression_range (m : ℝ) : 
  (∃ (x : ℝ), x = (m - 1).sqrt / (m - 2)) ↔ (m ≥ 1 ∧ m ≠ 2) :=
by sorry

end meaningful_expression_range_l2120_212054


namespace kirills_height_l2120_212062

theorem kirills_height (h_kirill : ℕ) (h_brother : ℕ) 
  (height_difference : h_brother = h_kirill + 14)
  (total_height : h_kirill + h_brother = 112) : 
  h_kirill = 49 := by
  sorry

end kirills_height_l2120_212062


namespace inequality_proof_l2120_212058

theorem inequality_proof (m n a : ℝ) (h : m > n) : a - m < a - n := by
  sorry

end inequality_proof_l2120_212058


namespace star_symmetric_set_eq_three_lines_l2120_212078

/-- The star operation -/
def star (a b : ℝ) : ℝ := a^2 * b + a * b^2

/-- The set of points (x, y) where x ★ y = y ★ x -/
def star_symmetric_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

/-- The union of three lines: x = 0, y = 0, and x + y = 0 -/
def three_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 + p.2 = 0}

theorem star_symmetric_set_eq_three_lines :
  star_symmetric_set = three_lines := by sorry

end star_symmetric_set_eq_three_lines_l2120_212078


namespace correct_calculation_l2120_212093

theorem correct_calculation (x : ℝ) (h : x + 20 = 180) : x / 20 = 8 := by
  sorry

end correct_calculation_l2120_212093


namespace largest_angle_in_special_triangle_l2120_212095

theorem largest_angle_in_special_triangle (A B C : ℝ) (h_triangle : A + B + C = π)
  (h_ratio : (Real.sin B + Real.sin C) / (Real.sin C + Real.sin A) = 4/5 ∧
             (Real.sin C + Real.sin A) / (Real.sin A + Real.sin B) = 5/6) :
  max A (max B C) = 2*π/3 :=
by sorry

end largest_angle_in_special_triangle_l2120_212095


namespace king_arthur_advisors_l2120_212089

theorem king_arthur_advisors (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) : 
  let q := 1 - p
  let prob_correct_two_advisors := p^2 + 2*p*q*(1/2)
  prob_correct_two_advisors = p :=
by sorry

end king_arthur_advisors_l2120_212089


namespace perimeter_of_specific_arrangement_l2120_212072

/-- Represents the arrangement of unit squares in the figure -/
def SquareArrangement : Type := Unit  -- Placeholder for the specific arrangement

/-- Calculates the perimeter of the given square arrangement -/
def perimeter (arrangement : SquareArrangement) : ℕ :=
  26  -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the perimeter of the given square arrangement is 26 -/
theorem perimeter_of_specific_arrangement :
  ∀ (arrangement : SquareArrangement), perimeter arrangement = 26 := by
  sorry

#check perimeter_of_specific_arrangement

end perimeter_of_specific_arrangement_l2120_212072


namespace three_point_five_million_scientific_notation_l2120_212024

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- Definition of 3.5 million -/
def three_point_five_million : ℝ := 3.5e6

/-- Theorem stating that 3.5 million can be expressed as 3.5 × 10^6 in scientific notation -/
theorem three_point_five_million_scientific_notation :
  ∃ (sn : ScientificNotation), three_point_five_million = sn.a * (10 : ℝ) ^ sn.n :=
sorry

end three_point_five_million_scientific_notation_l2120_212024


namespace triangle_property_l2120_212005

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    proves that under certain conditions, angle A is π/3 and the area is 3√3. -/
theorem triangle_property (a b c A B C : ℝ) : 
  0 < A ∧ A < π ∧   -- A is in (0, π)
  0 < B ∧ B < π ∧   -- B is in (0, π)
  0 < C ∧ C < π ∧   -- C is in (0, π)
  A + B + C = π ∧   -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧   -- Positive side lengths
  a * Real.sin C = Real.sqrt 3 * c * Real.cos A ∧   -- Given condition
  a = Real.sqrt 13 ∧   -- Given value of a
  c = 3 →   -- Given value of c
  A = π / 3 ∧   -- Angle A is 60°
  (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3   -- Area of triangle ABC
  := by sorry

end triangle_property_l2120_212005


namespace b_complete_time_l2120_212007

/-- The time it takes for A to complete the work alone -/
def a_time : ℚ := 14 / 3

/-- The time A and B work together -/
def together_time : ℚ := 1

/-- The time B works alone after A leaves -/
def b_remaining_time : ℚ := 41 / 14

/-- The time it takes for B to complete the work alone -/
def b_time : ℚ := 5

theorem b_complete_time : 
  (1 / a_time + 1 / b_time) * together_time + 
  (1 / b_time) * b_remaining_time = 1 := by sorry

end b_complete_time_l2120_212007


namespace smallest_integer_with_remainder_one_l2120_212074

theorem smallest_integer_with_remainder_one : ∃ k : ℕ,
  k > 1 ∧
  k % 13 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  k % 5 = 1 ∧
  (∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 ∧ m % 5 = 1 → k ≤ m) ∧
  k = 1366 :=
by sorry

end smallest_integer_with_remainder_one_l2120_212074


namespace perfect_square_binomial_l2120_212063

theorem perfect_square_binomial (x : ℝ) (k : ℝ) : 
  (∃ b : ℝ, ∀ x, x^2 + 24*x + k = (x + b)^2) ↔ k = 144 := by
  sorry

end perfect_square_binomial_l2120_212063


namespace factoring_expression_l2120_212018

theorem factoring_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end factoring_expression_l2120_212018


namespace luke_stickers_to_sister_l2120_212052

/-- Calculates the number of stickers Luke gave to his sister -/
def stickers_given_to_sister (initial : ℕ) (bought : ℕ) (birthday : ℕ) (used : ℕ) (final : ℕ) : ℕ :=
  initial + bought + birthday - used - final

/-- Theorem stating the number of stickers Luke gave to his sister -/
theorem luke_stickers_to_sister :
  stickers_given_to_sister 20 12 20 8 39 = 5 := by
  sorry

end luke_stickers_to_sister_l2120_212052


namespace apartment_occupancy_l2120_212006

theorem apartment_occupancy (stories : ℕ) (apartments_per_floor : ℕ) (total_people : ℕ) 
  (h1 : stories = 25)
  (h2 : apartments_per_floor = 4)
  (h3 : total_people = 200) :
  total_people / (stories * apartments_per_floor) = 2 := by
sorry

end apartment_occupancy_l2120_212006


namespace distinct_prime_factors_of_divisor_product_60_l2120_212087

/-- The number of distinct prime factors of the product of divisors of 60 -/
theorem distinct_prime_factors_of_divisor_product_60 : ∃ (B : ℕ), 
  (∀ d : ℕ, d ∣ 60 → d ∣ B) ∧ 
  (∀ n : ℕ, (∀ d : ℕ, d ∣ 60 → d ∣ n) → B ∣ n) ∧
  (Nat.card {p : ℕ | Nat.Prime p ∧ p ∣ B} = 3) :=
sorry

end distinct_prime_factors_of_divisor_product_60_l2120_212087


namespace mixed_number_multiplication_l2120_212019

theorem mixed_number_multiplication :
  99 * (24 / 25) * (-5) = -(499 + 4 / 5) :=
by sorry

end mixed_number_multiplication_l2120_212019


namespace max_guaranteed_winning_score_l2120_212009

/-- Represents a 9x9 grid game board -/
def GameBoard := Fin 9 → Fin 9 → Bool

/-- Counts the number of rows and columns where crosses outnumber noughts -/
def countCrossDominance (board : GameBoard) : ℕ :=
  sorry

/-- Counts the number of rows and columns where noughts outnumber crosses -/
def countNoughtDominance (board : GameBoard) : ℕ :=
  sorry

/-- Calculates the winning score for the first player -/
def winningScore (board : GameBoard) : ℤ :=
  (countCrossDominance board : ℤ) - (countNoughtDominance board : ℤ)

/-- Represents a strategy for playing the game -/
def Strategy := GameBoard → Fin 9 × Fin 9

/-- The theorem stating that the maximum guaranteed winning score is 2 -/
theorem max_guaranteed_winning_score :
  ∃ (strategyFirst : Strategy),
    ∀ (strategySecond : Strategy),
      ∃ (finalBoard : GameBoard),
        (winningScore finalBoard ≥ 2) ∧
        ∀ (otherFinalBoard : GameBoard),
          winningScore otherFinalBoard ≤ 2 :=
sorry

end max_guaranteed_winning_score_l2120_212009


namespace hexagon_four_identical_shapes_l2120_212033

/-- A regular hexagon -/
structure RegularHexagon where
  -- Add necessary fields

/-- A line segment representing a cut in the hexagon -/
structure Cut where
  -- Add necessary fields

/-- Represents a shape resulting from cuts in the hexagon -/
structure Shape where
  -- Add necessary fields

/-- Checks if two shapes are identical -/
def are_identical (s1 s2 : Shape) : Prop := sorry

/-- Checks if a cut is along a symmetry axis of the hexagon -/
def is_symmetry_axis_cut (h : RegularHexagon) (c : Cut) : Prop := sorry

/-- Theorem: A regular hexagon can be divided into four identical shapes by cutting along its symmetry axes -/
theorem hexagon_four_identical_shapes (h : RegularHexagon) :
  ∃ (c1 c2 : Cut) (s1 s2 s3 s4 : Shape),
    is_symmetry_axis_cut h c1 ∧
    is_symmetry_axis_cut h c2 ∧
    are_identical s1 s2 ∧
    are_identical s1 s3 ∧
    are_identical s1 s4 :=
  sorry

end hexagon_four_identical_shapes_l2120_212033


namespace square_roots_problem_l2120_212028

theorem square_roots_problem (n : ℝ) (h_pos : n > 0) :
  (∃ x : ℝ, (x + 1)^2 = n ∧ (4 - 2*x)^2 = n) → n = 36 := by
  sorry

end square_roots_problem_l2120_212028


namespace largest_common_value_less_than_1000_l2120_212000

/-- 
Given two arithmetic progressions:
1) {5, 9, 13, 17, ...} with common difference 4
2) {4, 12, 20, 28, ...} with common difference 8
This theorem states that their largest common value less than 1000 is 993.
-/
theorem largest_common_value_less_than_1000 :
  let seq1 := fun n : ℕ => 5 + 4 * n
  let seq2 := fun n : ℕ => 4 + 8 * n
  ∃ (k1 k2 : ℕ), seq1 k1 = seq2 k2 ∧ 
                 seq1 k1 < 1000 ∧
                 ∀ (m1 m2 : ℕ), seq1 m1 = seq2 m2 → seq1 m1 < 1000 → seq1 m1 ≤ seq1 k1 ∧
                 seq1 k1 = 993 :=
by sorry


end largest_common_value_less_than_1000_l2120_212000


namespace abs_condition_for_log_half_condition_l2120_212060

-- Define the logarithm with base 1/2
noncomputable def log_half (x : ℝ) := Real.log x / Real.log (1/2)

-- Statement of the theorem
theorem abs_condition_for_log_half_condition (x : ℝ) :
  (∀ x, |x - 2| < 1 → log_half (x + 2) < 0) ∧
  (∃ x, log_half (x + 2) < 0 ∧ |x - 2| ≥ 1) :=
by sorry

end abs_condition_for_log_half_condition_l2120_212060


namespace symmetry_about_xOy_plane_l2120_212041

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry about the xOy plane -/
def symmetricAboutXOy (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = q.y ∧ p.z = -q.z

theorem symmetry_about_xOy_plane :
  let p := Point3D.mk 1 3 (-5)
  let q := Point3D.mk 1 3 5
  symmetricAboutXOy p q :=
by
  sorry

#check symmetry_about_xOy_plane

end symmetry_about_xOy_plane_l2120_212041


namespace probability_walking_200_or_less_l2120_212085

/-- Number of gates in the airport --/
def num_gates : ℕ := 20

/-- Distance between adjacent gates in feet --/
def gate_distance : ℕ := 50

/-- Maximum walking distance in feet --/
def max_distance : ℕ := 200

/-- Calculate the number of favorable outcomes --/
def favorable_outcomes : ℕ := sorry

/-- Calculate the total number of possible outcomes --/
def total_outcomes : ℕ := num_gates * (num_gates - 1)

/-- The probability of walking 200 feet or less --/
theorem probability_walking_200_or_less :
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 19 := by sorry

end probability_walking_200_or_less_l2120_212085


namespace initial_cost_of_article_l2120_212043

/-- 
Proves that the initial cost of an article is 3000, given the conditions of two successive discounts.
-/
theorem initial_cost_of_article (price_after_first_discount : ℕ) 
  (final_price : ℕ) (h1 : price_after_first_discount = 2100) 
  (h2 : final_price = 1050) : ℕ :=
  by
    sorry

#check initial_cost_of_article

end initial_cost_of_article_l2120_212043


namespace simplify_power_l2120_212092

theorem simplify_power (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by sorry

end simplify_power_l2120_212092


namespace vector_magnitude_problem_l2120_212044

/-- Given two plane vectors a and b with an angle of 120° between them,
    |a| = 1, |b| = 2, and a vector m satisfying m · a = m · b = 1,
    prove that |m| = √21/3 -/
theorem vector_magnitude_problem (a b m : ℝ × ℝ) :
  (∃ θ : ℝ, θ = 2 * π / 3 ∧ a.1 * b.1 + a.2 * b.2 = ‖a‖ * ‖b‖ * Real.cos θ) →
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  m • a = 1 →
  m • b = 1 →
  ‖m‖ = Real.sqrt 21 / 3 := by
  sorry

#check vector_magnitude_problem

end vector_magnitude_problem_l2120_212044


namespace student_combinations_l2120_212011

/-- The number of possible combinations when n people each have 2 choices -/
def combinations (n : ℕ) : ℕ := 2^n

/-- There are 5 students -/
def num_students : ℕ := 5

/-- Theorem: The number of combinations for 5 students with 2 choices each is 32 -/
theorem student_combinations : combinations num_students = 32 := by
  sorry

end student_combinations_l2120_212011


namespace cheeseburger_cost_is_three_l2120_212015

def restaurant_problem (cheeseburger_cost : ℝ) : Prop :=
  let jim_money : ℝ := 20
  let cousin_money : ℝ := 10
  let total_money : ℝ := jim_money + cousin_money
  let spent_percentage : ℝ := 0.8
  let milkshake_cost : ℝ := 5
  let cheese_fries_cost : ℝ := 8
  let total_spent : ℝ := total_money * spent_percentage
  let num_cheeseburgers : ℕ := 2
  let num_milkshakes : ℕ := 2
  total_spent = num_cheeseburgers * cheeseburger_cost + num_milkshakes * milkshake_cost + cheese_fries_cost

theorem cheeseburger_cost_is_three :
  restaurant_problem 3 := by sorry

end cheeseburger_cost_is_three_l2120_212015


namespace christmas_day_is_saturday_l2120_212065

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in November or December -/
structure Date where
  month : Nat
  day : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (date : Date) : DayOfWeek := sorry

/-- Function to add days to a given date -/
def addDays (date : Date) (days : Nat) : Date := sorry

theorem christmas_day_is_saturday 
  (thanksgiving : Date)
  (h1 : thanksgiving.month = 11)
  (h2 : thanksgiving.day = 25)
  (h3 : dayOfWeek thanksgiving = DayOfWeek.Thursday) :
  dayOfWeek (Date.mk 12 25) = DayOfWeek.Saturday := by sorry

end christmas_day_is_saturday_l2120_212065


namespace regular_hexagon_perimeter_l2120_212004

/-- The perimeter of a regular hexagon with side length 5 cm is 30 cm. -/
theorem regular_hexagon_perimeter :
  ∀ (side_length : ℝ),
  side_length = 5 →
  (6 : ℝ) * side_length = 30 :=
by sorry

end regular_hexagon_perimeter_l2120_212004


namespace competition_results_l2120_212094

def scores_8_1 : List ℕ := [70, 70, 75, 75, 75, 75, 80, 80, 80, 85, 90, 90, 90, 90, 90, 95, 95, 95, 100, 100]
def scores_8_2 : List ℕ := [75, 75, 80, 80, 80, 80, 80, 85, 85, 85, 85, 85, 85, 85, 85, 90, 90, 95, 95, 100]

def median (l : List ℕ) : ℚ := sorry
def mean (l : List ℕ) : ℚ := sorry
def variance (l : List ℕ) : ℚ := sorry

theorem competition_results :
  (median scores_8_1 = 87.5) ∧
  (mean scores_8_2 = 85) ∧
  (variance scores_8_2 < variance scores_8_1) := by
  sorry

end competition_results_l2120_212094


namespace product_of_three_numbers_l2120_212084

theorem product_of_three_numbers (a b c : ℚ) : 
  a + b + c = 30 →
  a = 2 * (b + c) →
  b = 5 * c →
  a * b * c = 2500 / 9 := by
sorry

end product_of_three_numbers_l2120_212084


namespace tangent_perpendicular_line_l2120_212013

theorem tangent_perpendicular_line (x₀ y₀ c : ℝ) : 
  y₀ = Real.exp x₀ →                     -- P is on the curve y = e^x
  x₀ + 2 * y₀ + c = 0 →                  -- Line passes through P
  2 * Real.exp x₀ = -1 →                 -- Line is perpendicular to tangent
  c = -4 - Real.log 2 := by
sorry

end tangent_perpendicular_line_l2120_212013


namespace arithmetic_sequence_nth_term_l2120_212090

/-- 
Given an arithmetic sequence where:
- The first term is 3x - 4
- The second term is 7x - 15
- The third term is 4x + 2
- The nth term is 4018

Prove that n = 803
-/
theorem arithmetic_sequence_nth_term (x : ℚ) (n : ℕ) :
  (3 * x - 4 : ℚ) = (7 * x - 15 : ℚ) - (3 * x - 4 : ℚ) ∧
  (7 * x - 15 : ℚ) = (4 * x + 2 : ℚ) - (7 * x - 15 : ℚ) ∧
  (8 : ℚ) + (n - 1 : ℚ) * 5 = 4018 →
  n = 803 := by
sorry

end arithmetic_sequence_nth_term_l2120_212090


namespace isabel_circuit_length_l2120_212050

/-- The length of Isabel's running circuit in meters. -/
def circuit_length : ℕ := 365

/-- The number of times Isabel runs the circuit in the morning. -/
def morning_runs : ℕ := 7

/-- The number of times Isabel runs the circuit in the afternoon. -/
def afternoon_runs : ℕ := 3

/-- The total distance Isabel runs in a week, in meters. -/
def weekly_distance : ℕ := 25550

/-- The number of days in a week. -/
def days_in_week : ℕ := 7

theorem isabel_circuit_length :
  circuit_length * (morning_runs + afternoon_runs) * days_in_week = weekly_distance :=
sorry

end isabel_circuit_length_l2120_212050


namespace emily_cards_l2120_212010

theorem emily_cards (x : ℕ) : x + 7 = 70 → x = 63 := by
  sorry

end emily_cards_l2120_212010


namespace power_ranger_stickers_l2120_212073

theorem power_ranger_stickers (box1 box2 total : ℕ) : 
  box1 = 23 →
  box2 = box1 + 12 →
  total = box1 + box2 →
  total = 58 := by sorry

end power_ranger_stickers_l2120_212073


namespace quadratic_one_solution_l2120_212070

/-- The quadratic equation bx^2 - 12x + 9 = 0 has exactly one solution when b = 4 -/
theorem quadratic_one_solution (b : ℝ) : 
  (∃! x, b * x^2 - 12 * x + 9 = 0) ↔ b = 4 := by
  sorry

end quadratic_one_solution_l2120_212070
