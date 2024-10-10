import Mathlib

namespace fraction_less_than_two_l2662_266280

theorem fraction_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end fraction_less_than_two_l2662_266280


namespace area_ABCD_less_than_one_l2662_266222

-- Define the quadrilateral ABCD
variable (A B C D M P Q : ℝ × ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def diagonals_intersect_at (A B C D M : ℝ × ℝ) : Prop := sorry

def area_triangle (X Y Z : ℝ × ℝ) : ℝ := sorry

def is_midpoint (P X Y : ℝ × ℝ) : Prop := sorry

def distance (X Y : ℝ × ℝ) : ℝ := sorry

def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ABCD_less_than_one
  (h_convex : is_convex_quadrilateral A B C D)
  (h_diagonals : diagonals_intersect_at A B C D M)
  (h_area : area_triangle A D M > area_triangle B C M)
  (h_midpoint_P : is_midpoint P B C)
  (h_midpoint_Q : is_midpoint Q A D)
  (h_distance : distance A P + distance A Q = Real.sqrt 2) :
  area_quadrilateral A B C D < 1 := by sorry

end area_ABCD_less_than_one_l2662_266222


namespace carol_weight_l2662_266248

/-- Given two people's weights satisfying certain conditions, prove that one person's weight is 165 pounds. -/
theorem carol_weight (alice_weight carol_weight : ℝ) 
  (sum_condition : alice_weight + carol_weight = 220)
  (difference_condition : carol_weight - alice_weight = (2/3) * carol_weight) :
  carol_weight = 165 := by
  sorry

end carol_weight_l2662_266248


namespace linear_equation_mn_l2662_266234

theorem linear_equation_mn (m n : ℝ) : 
  (∀ x y : ℝ, ∃ a b c : ℝ, x^(4-3*|m|) + y^(3*|n|) = a*x + b*y + c) →
  m * n < 0 →
  0 < m + n →
  m + n ≤ 3 →
  m - n = 4/3 := by
sorry

end linear_equation_mn_l2662_266234


namespace find_x_value_l2662_266261

theorem find_x_value (A B : Set ℝ) (x : ℝ) : 
  A = {-1, 1} → 
  B = {0, 1, x-1} → 
  A ⊆ B → 
  x = 0 := by
sorry

end find_x_value_l2662_266261


namespace parallelogram_center_not_axis_symmetric_l2662_266231

-- Define the shape types
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | Rectangle
  | Rhombus

-- Define the symmetry properties
def isAxisSymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.Parallelogram => false
  | Shape.Rectangle => true
  | Shape.Rhombus => true

def isCenterSymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => false
  | Shape.Parallelogram => true
  | Shape.Rectangle => true
  | Shape.Rhombus => true

-- Theorem statement
theorem parallelogram_center_not_axis_symmetric :
  ∃ (s : Shape), isCenterSymmetric s ∧ ¬isAxisSymmetric s ∧
  ∀ (t : Shape), t ≠ s → ¬(isCenterSymmetric t ∧ ¬isAxisSymmetric t) :=
sorry

end parallelogram_center_not_axis_symmetric_l2662_266231


namespace product_repeating_third_and_nine_l2662_266262

/-- The repeating decimal 0.3̄ -/
def repeating_third : ℚ := 1 / 3

/-- The theorem stating that 0.3̄ * 9 = 3 -/
theorem product_repeating_third_and_nine : repeating_third * 9 = 3 := by
  sorry

end product_repeating_third_and_nine_l2662_266262


namespace second_replaced_man_age_l2662_266243

theorem second_replaced_man_age 
  (n : ℕ) 
  (age_increase : ℝ) 
  (first_replaced_age : ℕ) 
  (new_men_avg_age : ℝ) 
  (h1 : n = 15)
  (h2 : age_increase = 2)
  (h3 : first_replaced_age = 21)
  (h4 : new_men_avg_age = 37) :
  ∃ (second_replaced_age : ℕ),
    (n : ℝ) * age_increase = 
      2 * new_men_avg_age - (first_replaced_age : ℝ) - (second_replaced_age : ℝ) ∧
    second_replaced_age = 23 :=
by sorry

end second_replaced_man_age_l2662_266243


namespace expand_expression_l2662_266203

theorem expand_expression (x y : ℝ) : (x + 3) * (4 * x - 5 * y) = 4 * x^2 - 5 * x * y + 12 * x - 15 * y := by
  sorry

end expand_expression_l2662_266203


namespace monkey_count_l2662_266246

theorem monkey_count : ∃! x : ℕ, x > 0 ∧ (x / 8)^2 + 12 = x := by
  sorry

end monkey_count_l2662_266246


namespace book_shelf_average_width_l2662_266254

theorem book_shelf_average_width :
  let book_widths : List ℝ := [5, 3/4, 1.5, 3.25, 4, 3, 7/2, 12]
  (book_widths.sum / book_widths.length : ℝ) = 4.125 := by
  sorry

end book_shelf_average_width_l2662_266254


namespace sum_of_cube_ratios_l2662_266269

open BigOperators

/-- Given a finite sequence of rational numbers x_t = i/101 for i = 0 to 101,
    the sum T = ∑(i=0 to 101) [x_i^3 / (3x_t^2 - 3x_t + 1)] is equal to 51. -/
theorem sum_of_cube_ratios (x : Fin 102 → ℚ) 
  (h : ∀ i : Fin 102, x i = (i : ℚ) / 101) : 
  ∑ i : Fin 102, (x i)^3 / (3 * (x i)^2 - 3 * (x i) + 1) = 51 := by
  sorry

end sum_of_cube_ratios_l2662_266269


namespace savings_calculation_l2662_266211

theorem savings_calculation (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  furniture_fraction = 3 / 4 →
  tv_cost = 230 →
  (1 - furniture_fraction) * savings = tv_cost →
  savings = 920 := by
sorry

end savings_calculation_l2662_266211


namespace angle_B_measure_l2662_266223

-- Define the hexagon NUMBERS
structure Hexagon :=
  (N U M B E S : ℝ)

-- Define the properties of the hexagon
def is_valid_hexagon (h : Hexagon) : Prop :=
  h.N + h.U + h.M + h.B + h.E + h.S = 720 ∧ 
  h.N = h.M ∧ h.M = h.B ∧
  h.U + h.S = 180

-- Theorem statement
theorem angle_B_measure (h : Hexagon) (hvalid : is_valid_hexagon h) : h.B = 135 := by
  sorry


end angle_B_measure_l2662_266223


namespace least_subtraction_for_divisibility_l2662_266249

theorem least_subtraction_for_divisibility :
  ∃ (n : ℕ), n = 4 ∧ 
  (15 ∣ (9679 - n)) ∧ 
  ∀ (m : ℕ), m < n → ¬(15 ∣ (9679 - m)) := by
  sorry

end least_subtraction_for_divisibility_l2662_266249


namespace leg_length_theorem_l2662_266210

/-- An isosceles triangle with a median on one leg dividing the perimeter -/
structure IsoscelesTriangleWithMedian where
  leg : ℝ
  base : ℝ
  median_divides_perimeter : leg + leg + base = 12 + 18
  isosceles : leg > 0
  base_positive : base > 0

/-- The theorem stating the possible lengths of the leg -/
theorem leg_length_theorem (triangle : IsoscelesTriangleWithMedian) :
  triangle.leg = 8 ∨ triangle.leg = 12 := by
  sorry

#check leg_length_theorem

end leg_length_theorem_l2662_266210


namespace mortgage_payment_months_l2662_266206

theorem mortgage_payment_months (a : ℝ) (r : ℝ) (total : ℝ) (n : ℕ) 
  (h1 : a = 100)
  (h2 : r = 3)
  (h3 : total = 12100)
  (h4 : total = a * (1 - r^n) / (1 - r)) :
  n = 5 := by
  sorry

end mortgage_payment_months_l2662_266206


namespace min_value_of_function_l2662_266229

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x + 2 / (2 * x + 1) - 3 / 2 ≥ 0 ∧ ∃ y > 0, y + 2 / (2 * y + 1) - 3 / 2 = 0 :=
sorry

end min_value_of_function_l2662_266229


namespace expression_evaluation_l2662_266298

theorem expression_evaluation : 12 - (-18) + (-7) - 15 = 8 := by
  sorry

end expression_evaluation_l2662_266298


namespace greenhill_soccer_kicks_l2662_266233

/-- Given a soccer team with total players and goalies, calculate the number of penalty kicks required --/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  goalies * (total_players - 1)

/-- Theorem: For a team with 25 players including 4 goalies, 96 penalty kicks are required --/
theorem greenhill_soccer_kicks : penalty_kicks 25 4 = 96 := by
  sorry

end greenhill_soccer_kicks_l2662_266233


namespace max_sum_of_digits_of_sum_l2662_266240

/-- Represents a three-digit positive integer with distinct digits from 1 to 9 -/
structure ThreeDigitNumber :=
  (value : ℕ)
  (is_three_digit : 100 ≤ value ∧ value ≤ 999)
  (distinct_digits : ∀ d₁ d₂ d₃, value = 100 * d₁ + 10 * d₂ + d₃ → d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃)
  (digits_range : ∀ d₁ d₂ d₃, value = 100 * d₁ + 10 * d₂ + d₃ → 1 ≤ d₁ ∧ d₁ ≤ 9 ∧ 1 ≤ d₂ ∧ d₂ ≤ 9 ∧ 1 ≤ d₃ ∧ d₃ ≤ 9)

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The main theorem -/
theorem max_sum_of_digits_of_sum (a b : ThreeDigitNumber) :
  let S := a.value + b.value
  100 ≤ S ∧ S ≤ 999 →
  sum_of_digits S ≤ 12 :=
sorry

end max_sum_of_digits_of_sum_l2662_266240


namespace witnesses_same_type_l2662_266205

-- Define the possible types of witnesses
inductive WitnessType
| Truthful
| Liar

-- Define the structure for a witness
structure Witness where
  name : String
  type : WitnessType

-- Define the theorem
theorem witnesses_same_type (A B C : Witness) 
  (h1 : C.name ≠ A.name ∧ C.name ≠ B.name)
  (h2 : A.name ≠ B.name)
  (h3 : ¬(A.type ≠ B.type)) :
  A.type = B.type := by sorry

end witnesses_same_type_l2662_266205


namespace negation_of_universal_proposition_l2662_266251

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by
  sorry

end negation_of_universal_proposition_l2662_266251


namespace picnic_basket_theorem_l2662_266266

/-- Calculate the total cost of a picnic basket given the number of people and item prices -/
def picnic_basket_cost (num_people : ℕ) (sandwich_price fruit_salad_price soda_price snack_price : ℚ) : ℚ :=
  let sandwich_cost := num_people * sandwich_price
  let fruit_salad_cost := num_people * fruit_salad_price
  let soda_cost := num_people * 2 * soda_price
  let snack_cost := 3 * snack_price
  sandwich_cost + fruit_salad_cost + soda_cost + snack_cost

/-- The total cost of the picnic basket is $60 -/
theorem picnic_basket_theorem :
  picnic_basket_cost 4 5 3 2 4 = 60 := by
  sorry

end picnic_basket_theorem_l2662_266266


namespace leadership_selection_count_l2662_266299

/-- The number of people in the group -/
def n : ℕ := 5

/-- The number of positions to be filled (leader and deputy) -/
def k : ℕ := 2

/-- The number of ways to select a leader and deputy with no restrictions -/
def total_selections : ℕ := n * (n - 1)

/-- The number of invalid selections (when the restricted person is deputy) -/
def invalid_selections : ℕ := n - 1

/-- The number of valid selections -/
def valid_selections : ℕ := total_selections - invalid_selections

theorem leadership_selection_count :
  valid_selections = 16 :=
sorry

end leadership_selection_count_l2662_266299


namespace set_equality_l2662_266293

def S : Set (ℕ × ℕ) := {p | p.1 + p.2 = 3}

theorem set_equality : S = {(1, 2), (2, 1)} := by
  sorry

end set_equality_l2662_266293


namespace average_daily_allowance_l2662_266201

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The weekly calorie allowance -/
def weekly_allowance : ℕ := 10500

/-- The average daily calorie allowance -/
def daily_allowance : ℕ := weekly_allowance / days_in_week

theorem average_daily_allowance :
  daily_allowance = 1500 :=
sorry

end average_daily_allowance_l2662_266201


namespace lacy_correct_percentage_l2662_266270

theorem lacy_correct_percentage (x : ℝ) (h : x > 0) : 
  let total := 6 * x
  let missed := 2 * x
  let correct := total - missed
  (correct / total) * 100 = 200 / 3 := by
sorry

end lacy_correct_percentage_l2662_266270


namespace carriage_sharing_problem_l2662_266256

theorem carriage_sharing_problem (x : ℕ) : 
  (x / 3 : ℚ) + 2 = (x - 9 : ℚ) / 2 ↔ 
  (∃ (total_carriages : ℕ), 
    (x / 3 : ℚ) + 2 = total_carriages ∧ 
    (x - 9 : ℚ) / 2 = total_carriages) :=
sorry

end carriage_sharing_problem_l2662_266256


namespace base_seven_43210_equals_10738_l2662_266253

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_43210_equals_10738 :
  base_seven_to_ten [0, 1, 2, 3, 4] = 10738 := by
  sorry

end base_seven_43210_equals_10738_l2662_266253


namespace cube_difference_l2662_266288

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) :
  a^3 - b^3 = 108 := by
sorry

end cube_difference_l2662_266288


namespace tickets_spent_on_glow_bracelets_l2662_266220

/-- Given Connie's ticket redemption scenario, prove the number of tickets spent on glow bracelets. -/
theorem tickets_spent_on_glow_bracelets 
  (total_tickets : ℕ) 
  (koala_tickets : ℕ) 
  (earbud_tickets : ℕ) : 
  total_tickets = 50 → 
  koala_tickets = total_tickets / 2 → 
  earbud_tickets = 10 → 
  total_tickets - (koala_tickets + earbud_tickets) = 15 := by
  sorry


end tickets_spent_on_glow_bracelets_l2662_266220


namespace power_of_three_mod_eight_l2662_266279

theorem power_of_three_mod_eight : 3^2007 % 8 = 3 := by sorry

end power_of_three_mod_eight_l2662_266279


namespace stratified_sampling_sample_size_l2662_266230

theorem stratified_sampling_sample_size
  (ratio_10 : ℕ)
  (ratio_11 : ℕ)
  (ratio_12 : ℕ)
  (sample_12 : ℕ)
  (h_ratio : ratio_10 = 2 ∧ ratio_11 = 3 ∧ ratio_12 = 5)
  (h_sample_12 : sample_12 = 150)
  : ∃ (n : ℕ), n = 300 ∧ (ratio_12 : ℚ) / (ratio_10 + ratio_11 + ratio_12 : ℚ) = sample_12 / n :=
by
  sorry

end stratified_sampling_sample_size_l2662_266230


namespace even_digits_529_base9_l2662_266236

/-- Converts a natural number from base 10 to base 9 -/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of even digits in the base-9 representation of 529₁₀ is 2 -/
theorem even_digits_529_base9 : 
  countEvenDigits (toBase9 529) = 2 :=
sorry

end even_digits_529_base9_l2662_266236


namespace max_carlson_jars_l2662_266204

/-- Represents the initial state of jam jars --/
structure JamState where
  carlson_weight : ℕ  -- Total weight of Carlson's jars
  baby_weight : ℕ     -- Total weight of Baby's jars
  carlson_jars : ℕ    -- Number of Carlson's jars

/-- Represents the state after Carlson gives his smallest jar to Baby --/
structure NewJamState where
  carlson_weight : ℕ  -- New total weight of Carlson's jars
  baby_weight : ℕ     -- New total weight of Baby's jars

/-- Conditions of the problem --/
def jam_problem (initial : JamState) (final : NewJamState) : Prop :=
  initial.carlson_weight = 13 * initial.baby_weight ∧
  final.carlson_weight = 8 * final.baby_weight ∧
  initial.carlson_weight = final.carlson_weight + (final.baby_weight - initial.baby_weight) ∧
  initial.carlson_jars > 0

/-- The theorem to be proved --/
theorem max_carlson_jars :
  ∀ (initial : JamState) (final : NewJamState),
    jam_problem initial final →
    initial.carlson_jars ≤ 23 :=
sorry

end max_carlson_jars_l2662_266204


namespace sqrt_sum_simplification_l2662_266214

theorem sqrt_sum_simplification :
  ∃ (a b c : ℕ), c > 0 ∧ 
  (∀ (d : ℕ), d > 0 → (∃ (x y : ℕ), Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) = (x * Real.sqrt 6 + y * Real.sqrt 8) / d) → c ≤ d) ∧
  Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) = (a * Real.sqrt 6 + b * Real.sqrt 8) / c ∧
  a + b + c = 280 := by
  sorry

end sqrt_sum_simplification_l2662_266214


namespace odd_most_likely_l2662_266213

def box_size : Nat := 30

def is_multiple_of_10 (n : Nat) : Bool :=
  n % 10 = 0

def is_odd (n : Nat) : Bool :=
  n % 2 ≠ 0

def contains_digit_3 (n : Nat) : Bool :=
  ∃ d, d ∈ n.digits 10 ∧ d = 3

def is_multiple_of_5 (n : Nat) : Bool :=
  n % 5 = 0

def contains_digit_2 (n : Nat) : Bool :=
  ∃ d, d ∈ n.digits 10 ∧ d = 2

def count_satisfying (p : Nat → Bool) : Nat :=
  (List.range box_size).filter p |>.length

theorem odd_most_likely :
  count_satisfying is_odd >
  max
    (count_satisfying is_multiple_of_10)
    (max
      (count_satisfying contains_digit_3)
      (max
        (count_satisfying is_multiple_of_5)
        (count_satisfying contains_digit_2))) :=
by sorry

end odd_most_likely_l2662_266213


namespace x_value_l2662_266287

theorem x_value (x : ℝ) (h : (x / 3) / 3 = 9 / (x / 3)) : x = 9 * Real.sqrt 3 ∨ x = -9 * Real.sqrt 3 := by
  sorry

end x_value_l2662_266287


namespace roots_equation_value_l2662_266281

theorem roots_equation_value (α β : ℝ) : 
  α^2 - 2*α - 4 = 0 → β^2 - 2*β - 4 = 0 → α^3 + 8*β + 6 = 30 := by
  sorry

end roots_equation_value_l2662_266281


namespace solve_watermelon_problem_l2662_266289

def watermelon_problem (n : ℕ) (initial_avg : ℝ) (new_weight : ℝ) (new_avg : ℝ) : Prop :=
  let total_initial := n * initial_avg
  let replaced_weight := total_initial + new_weight - n * new_avg
  replaced_weight = 3

theorem solve_watermelon_problem :
  watermelon_problem 10 4.2 5 4.4 := by sorry

end solve_watermelon_problem_l2662_266289


namespace one_pair_percentage_l2662_266247

def five_digit_numbers : ℕ := 90000

def numbers_with_one_pair : ℕ := 10 * 10 * 9 * 8 * 7

theorem one_pair_percentage : 
  (numbers_with_one_pair : ℚ) / five_digit_numbers * 100 = 56 :=
by sorry

end one_pair_percentage_l2662_266247


namespace solution_l2662_266263

-- Define the equation
def equation (A B x : ℝ) : Prop :=
  A / (x + 3) + B / (x^2 - 9*x) = (x^2 - 3*x + 15) / (x^3 + x^2 - 27*x)

-- State the theorem
theorem solution (A B : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 9 → equation A B x) →
  (B : ℝ) / (A : ℝ) = 7.5 := by
sorry

end solution_l2662_266263


namespace sum_product_implies_difference_l2662_266202

theorem sum_product_implies_difference (x y : ℝ) : 
  x + y = 42 → x * y = 437 → |x - y| = 4 := by sorry

end sum_product_implies_difference_l2662_266202


namespace henry_twice_jills_age_l2662_266290

/-- Proves that Henry was twice Jill's age 6 years ago given their present ages and sum. -/
theorem henry_twice_jills_age (henry_age jill_age : ℕ) (sum_ages : ℕ) : 
  henry_age = 20 → 
  jill_age = 13 → 
  sum_ages = henry_age + jill_age → 
  sum_ages = 33 → 
  ∃ (years_ago : ℕ), years_ago = 6 ∧ henry_age - years_ago = 2 * (jill_age - years_ago) := by
  sorry

end henry_twice_jills_age_l2662_266290


namespace det_special_matrix_l2662_266265

/-- The determinant of the matrix [[2x + 2, 2x, 2x], [2x, 2x + 2, 2x], [2x, 2x, 2x + 2]] is equal to 20x + 8 -/
theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![2*x + 2, 2*x, 2*x; 
                2*x, 2*x + 2, 2*x; 
                2*x, 2*x, 2*x + 2] = 20*x + 8 := by
  sorry

end det_special_matrix_l2662_266265


namespace circus_show_acrobats_l2662_266268

/-- Represents the number of acrobats in the circus show. -/
def numAcrobats : ℕ := 2

/-- Represents the number of elephants in the circus show. -/
def numElephants : ℕ := 14

/-- Represents the number of clowns in the circus show. -/
def numClowns : ℕ := 14

/-- The total number of legs observed in the circus show. -/
def totalLegs : ℕ := 88

/-- The total number of heads observed in the circus show. -/
def totalHeads : ℕ := 30

theorem circus_show_acrobats :
  (2 * numAcrobats + 4 * numElephants + 2 * numClowns = totalLegs) ∧
  (numAcrobats + numElephants + numClowns = totalHeads) ∧
  (numAcrobats = 2) := by sorry

end circus_show_acrobats_l2662_266268


namespace invalid_external_diagonals_l2662_266259

theorem invalid_external_diagonals : ¬ ∃ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (a^2 + b^2 = 5^2 ∧ b^2 + c^2 = 6^2 ∧ a^2 + c^2 = 8^2) :=
sorry

end invalid_external_diagonals_l2662_266259


namespace ratio_extended_points_l2662_266252

-- Define the triangle ABC
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b ∧ b < c ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the points B₁, A₁, C₂, B₂
def ExtendedPoints (a b c : ℝ) : Prop :=
  ∃ (A B C A₁ B₁ C₂ B₂ : ℝ × ℝ),
    Triangle a b c ∧
    dist B C = a ∧
    dist C A = b ∧
    dist A B = c ∧
    dist B B₁ = c ∧
    dist A A₁ = c ∧
    dist C C₂ = a ∧
    dist B B₂ = a

-- State the theorem
theorem ratio_extended_points (a b c : ℝ) :
  Triangle a b c → ExtendedPoints a b c →
  ∃ (A₁ B₁ C₂ B₂ : ℝ × ℝ), dist A₁ B₁ / dist C₂ B₂ = c / a :=
sorry

end ratio_extended_points_l2662_266252


namespace sales_increase_price_reduction_for_target_profit_l2662_266232

/-- Represents the Asian Games mascot badge sales scenario -/
structure BadgeSales where
  originalProfit : ℝ  -- Original profit per set
  originalSold : ℝ    -- Original number of sets sold per day
  profitReduction : ℝ -- Reduction in profit per set
  saleIncrease : ℝ    -- Increase in sales per $1 reduction

/-- Calculates the increase in sets sold given a profit reduction -/
def increasedSales (s : BadgeSales) : ℝ :=
  s.profitReduction * s.saleIncrease

/-- Calculates the daily profit given a price reduction -/
def dailyProfit (s : BadgeSales) (priceReduction : ℝ) : ℝ :=
  (s.originalProfit - priceReduction) * (s.originalSold + priceReduction * s.saleIncrease)

/-- Theorem stating the increase in sales when profit is reduced by $2 -/
theorem sales_increase (s : BadgeSales) (h : s.originalProfit = 40 ∧ s.originalSold = 20 ∧ s.profitReduction = 2 ∧ s.saleIncrease = 2) :
  increasedSales s = 4 := by sorry

/-- Theorem stating the price reduction needed for a daily profit of $1200 -/
theorem price_reduction_for_target_profit (s : BadgeSales) (h : s.originalProfit = 40 ∧ s.originalSold = 20 ∧ s.saleIncrease = 2) :
  ∃ x : ℝ, x = 20 ∧ dailyProfit s x = 1200 := by sorry

end sales_increase_price_reduction_for_target_profit_l2662_266232


namespace triangle_abc_properties_l2662_266207

/-- Theorem about a specific triangle ABC -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B →
  S = 5 * Real.sqrt 3 →
  a = 5 →
  (1/2) * a * c * Real.sin B = S →
  B = π / 3 ∧ b = Real.sqrt 21 := by
  sorry

end triangle_abc_properties_l2662_266207


namespace trigonometric_problem_l2662_266275

theorem trigonometric_problem (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (Real.pi / 4 + α) = 1 / 3)
  (h4 : Real.cos (Real.pi / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 ∧
  Real.sin β = -1 / 3 ∧
  α - β = Real.pi / 4 := by
sorry

end trigonometric_problem_l2662_266275


namespace not_monotonic_implies_a_in_open_unit_interval_l2662_266218

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x - a

-- Define the derivative of g(x)
def g' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

-- Theorem statement
theorem not_monotonic_implies_a_in_open_unit_interval :
  ∀ a : ℝ, (∃ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 ∧ (g a x - g a y) * (x - y) > 0) →
  0 < a ∧ a < 1 := by
  sorry

end not_monotonic_implies_a_in_open_unit_interval_l2662_266218


namespace maddie_tshirt_cost_l2662_266217

-- Define the number of packs of white and blue T-shirts
def white_packs : ℕ := 2
def blue_packs : ℕ := 4

-- Define the number of T-shirts per pack for white and blue
def white_per_pack : ℕ := 5
def blue_per_pack : ℕ := 3

-- Define the cost per T-shirt
def cost_per_shirt : ℕ := 3

-- Define the total number of T-shirts
def total_shirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

-- Define the total cost
def total_cost : ℕ := total_shirts * cost_per_shirt

-- Theorem to prove
theorem maddie_tshirt_cost : total_cost = 66 := by
  sorry

end maddie_tshirt_cost_l2662_266217


namespace workshop_salary_problem_l2662_266283

theorem workshop_salary_problem (total_workers : ℕ) (avg_salary : ℕ) 
  (num_technicians : ℕ) (avg_salary_technicians : ℕ) :
  total_workers = 21 →
  avg_salary = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 12000 →
  let remaining_workers := total_workers - num_technicians
  let total_salary := total_workers * avg_salary
  let technicians_salary := num_technicians * avg_salary_technicians
  let remaining_salary := total_salary - technicians_salary
  (remaining_salary / remaining_workers : ℚ) = 6000 := by
  sorry

end workshop_salary_problem_l2662_266283


namespace geometric_sequence_ratio_l2662_266216

/-- A geometric sequence with common ratio q satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  is_geometric : ∀ n, a (n + 1) = a n * q
  condition1 : a 5 - a 1 = 15
  condition2 : a 4 - a 2 = 6

/-- The common ratio of a geometric sequence satisfying the given conditions is either 1/2 or 2 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : 
  seq.q = 1/2 ∨ seq.q = 2 :=
sorry

end geometric_sequence_ratio_l2662_266216


namespace one_square_covered_l2662_266260

/-- Represents a square on the checkerboard -/
structure Square where
  x : ℕ
  y : ℕ

/-- Represents the circular disc -/
structure Disc where
  center : Square
  diameter : ℝ

/-- Determines if a square is completely covered by the disc -/
def is_covered (s : Square) (d : Disc) : Prop :=
  (s.x - d.center.x)^2 + (s.y - d.center.y)^2 ≤ (d.diameter / 2)^2

/-- The checkerboard -/
def checkerboard : Set Square :=
  {s | s.x ≤ 8 ∧ s.y ≤ 8}

theorem one_square_covered (d : Disc) :
  d.diameter = Real.sqrt 2 →
  d.center ∈ checkerboard →
  ∃! s : Square, s ∈ checkerboard ∧ is_covered s d :=
sorry

end one_square_covered_l2662_266260


namespace perpendicular_condition_l2662_266237

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line y = ax + 1 -/
def slope1 (a : ℝ) : ℝ := a

/-- The slope of the second line y = (a-2)x + 3 -/
def slope2 (a : ℝ) : ℝ := a - 2

/-- The theorem stating that a = 1 is the necessary and sufficient condition for perpendicularity -/
theorem perpendicular_condition (a : ℝ) : 
  perpendicular (slope1 a) (slope2 a) ↔ a = 1 := by sorry

end perpendicular_condition_l2662_266237


namespace xiao_ming_book_price_l2662_266235

/-- The price of Xiao Ming's book satisfies 15 < x < 20, given that:
    1. Classmate A guessed the price is at least 20.
    2. Classmate B guessed the price is at most 15.
    3. Xiao Ming said both classmates are wrong. -/
theorem xiao_ming_book_price (x : ℝ) 
  (hA : x < 20)  -- Xiao Ming said A is wrong, so price is less than 20
  (hB : x > 15)  -- Xiao Ming said B is wrong, so price is greater than 15
  : 15 < x ∧ x < 20 := by
  sorry

end xiao_ming_book_price_l2662_266235


namespace largest_a_for_integer_solution_l2662_266296

theorem largest_a_for_integer_solution : 
  ∃ (a : ℝ), ∀ (b : ℝ), 
    (∃ (x y : ℤ), x - 4*y = 1 ∧ a*x + 3*y = 1) ∧
    (∀ (x y : ℤ), b*x + 3*y = 1 → x - 4*y = 1 → b ≤ a) ∧
    a = 1 :=
by sorry

end largest_a_for_integer_solution_l2662_266296


namespace product_congruence_l2662_266274

theorem product_congruence : 45 * 68 * 99 ≡ 15 [ZMOD 25] := by
  sorry

end product_congruence_l2662_266274


namespace initial_percent_problem_l2662_266264

theorem initial_percent_problem (x : ℝ) :
  (3 : ℝ) / 100 = (60 : ℝ) / 100 * x → x = (5 : ℝ) / 100 := by
  sorry

end initial_percent_problem_l2662_266264


namespace solution_set_f_min_value_fraction_equality_condition_l2662_266224

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + 2 * |x - 1|

-- Theorem for the solution set of f(x) ≤ 4
theorem solution_set_f :
  {x : ℝ | f x ≤ 4} = {x : ℝ | -1 ≤ x ∧ x ≤ 5/3} :=
sorry

-- Theorem for the minimum value of 2/a + 1/b
theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 4) :
  2/a + 1/b ≥ 2 :=
sorry

-- Theorem for the equality condition
theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 4) :
  2/a + 1/b = 2 ↔ a = 2 ∧ b = 1 :=
sorry

end solution_set_f_min_value_fraction_equality_condition_l2662_266224


namespace existence_of_common_root_l2662_266272

-- Define the structure of a quadratic polynomial
structure QuadraticPolynomial (R : Type*) [Ring R] where
  a : R
  b : R
  c : R

-- Define the evaluation of a quadratic polynomial
def evaluate {R : Type*} [Ring R] (p : QuadraticPolynomial R) (x : R) : R :=
  p.a * x * x + p.b * x + p.c

-- Theorem statement
theorem existence_of_common_root 
  {R : Type*} [Field R] 
  (f g h : QuadraticPolynomial R)
  (no_roots : ∀ x, evaluate f x ≠ 0 ∧ evaluate g x ≠ 0 ∧ evaluate h x ≠ 0)
  (same_leading_coeff : f.a = g.a ∧ f.a = h.a)
  (diff_x_coeff : f.b ≠ g.b ∧ f.b ≠ h.b ∧ g.b ≠ h.b) :
  ∃ c x, evaluate f x + c * evaluate g x = 0 ∧ evaluate f x + c * evaluate h x = 0 :=
sorry

end existence_of_common_root_l2662_266272


namespace solution_value_l2662_266227

/-- Represents a 2x3 augmented matrix --/
def AugmentedMatrix := Matrix (Fin 2) (Fin 3) ℝ

/-- Given augmented matrix --/
def givenMatrix : AugmentedMatrix := !![1, 0, 3; 1, 1, 4]

/-- Theorem: For the system of linear equations represented by the given augmented matrix,
    the value of x + 2y is equal to 5 --/
theorem solution_value (x y : ℝ) 
  (hx : givenMatrix 0 0 * x + givenMatrix 0 1 * y = givenMatrix 0 2)
  (hy : givenMatrix 1 0 * x + givenMatrix 1 1 * y = givenMatrix 1 2) :
  x + 2 * y = 5 := by
  sorry

end solution_value_l2662_266227


namespace tiling_pattern_ratio_l2662_266291

/-- The ratio of the area covered by triangles to the total area in a specific tiling pattern -/
theorem tiling_pattern_ratio : ∀ s : ℝ,
  s > 0 →
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let triangle_area := (Real.sqrt 3 / 16) * s^2
  let total_area := hexagon_area + 2 * triangle_area
  triangle_area / total_area = 1 / 13 :=
by sorry

end tiling_pattern_ratio_l2662_266291


namespace points_are_coplanar_l2662_266226

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define the points
variable (A B C P O : V)

-- Define the non-collinearity condition
def not_collinear (A B C : V) : Prop :=
  ∀ (t : ℝ), B - A ≠ t • (C - A)

-- Define the vector equation
def vector_equation (O A B C P : V) : Prop :=
  P - O = (3/4) • (A - O) + (1/8) • (B - O) + (1/8) • (C - O)

-- Define coplanarity
def coplanar (A B C P : V) : Prop :=
  ∃ (a b c : ℝ), P - A = a • (B - A) + b • (C - A)

-- State the theorem
theorem points_are_coplanar
  (h1 : not_collinear A B C)
  (h2 : ∀ O, vector_equation O A B C P) :
  coplanar A B C P :=
sorry

end points_are_coplanar_l2662_266226


namespace max_value_cos_sin_linear_combination_l2662_266242

theorem max_value_cos_sin_linear_combination (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos (θ - φ) + b * Real.sin (θ - φ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (θ - φ) + b * Real.sin (θ - φ) = Real.sqrt (a^2 + b^2)) := by
  sorry

end max_value_cos_sin_linear_combination_l2662_266242


namespace mateo_grape_bottles_l2662_266219

/-- Represents the number of bottles of a specific soda type. -/
structure SodaCount where
  orange : ℕ
  grape : ℕ

/-- Represents a person's soda inventory. -/
structure SodaInventory where
  count : SodaCount
  litersPerBottle : ℕ

def julio : SodaInventory :=
  { count := { orange := 4, grape := 7 },
    litersPerBottle := 2 }

def mateo (grapeBottles : ℕ) : SodaInventory :=
  { count := { orange := 1, grape := grapeBottles },
    litersPerBottle := 2 }

def totalLiters (inventory : SodaInventory) : ℕ :=
  (inventory.count.orange + inventory.count.grape) * inventory.litersPerBottle

theorem mateo_grape_bottles :
  ∃ g : ℕ, totalLiters julio = totalLiters (mateo g) + 14 ∧ g = 3 := by
  sorry

end mateo_grape_bottles_l2662_266219


namespace marble_count_l2662_266292

theorem marble_count (r : ℝ) (b g y : ℝ) : 
  r > 0 →
  b = r / 1.3 →
  g = 1.5 * r →
  y = 1.2 * g →
  r + b + g + y = 5.069 * r := by
sorry

end marble_count_l2662_266292


namespace purely_imaginary_complex_number_l2662_266276

theorem purely_imaginary_complex_number (a : ℝ) :
  (Complex.I * Complex.im (a * (1 + Complex.I) - 2) = a * (1 + Complex.I) - 2) →
  a = 2 := by
  sorry

end purely_imaginary_complex_number_l2662_266276


namespace chopping_percentage_difference_l2662_266285

/-- Represents the chopping rates and total amount for Tom and Tammy -/
structure ChoppingData where
  tom_rate : ℚ  -- Tom's chopping rate in lb/min
  tammy_rate : ℚ  -- Tammy's chopping rate in lb/min
  total_amount : ℚ  -- Total amount of salad chopped in lb

/-- Calculates the percentage difference between Tammy's and Tom's chopped quantities -/
def percentage_difference (data : ChoppingData) : ℚ :=
  let combined_rate := data.tom_rate + data.tammy_rate
  let tom_share := (data.tom_rate / combined_rate) * data.total_amount
  let tammy_share := (data.tammy_rate / combined_rate) * data.total_amount
  ((tammy_share - tom_share) / tom_share) * 100

/-- Theorem stating that the percentage difference is 125% for the given data -/
theorem chopping_percentage_difference :
  let data : ChoppingData := {
    tom_rate := 2 / 3,  -- 2 lb in 3 minutes
    tammy_rate := 3 / 2,  -- 3 lb in 2 minutes
    total_amount := 65
  }
  percentage_difference data = 125 := by sorry


end chopping_percentage_difference_l2662_266285


namespace fraction_simplification_l2662_266238

theorem fraction_simplification :
  5 / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end fraction_simplification_l2662_266238


namespace evaluate_expression_l2662_266244

theorem evaluate_expression : 3 * 403 + 5 * 403 + 2 * 403 + 401 = 4431 := by
  sorry

end evaluate_expression_l2662_266244


namespace min_socks_in_box_min_socks_even_black_l2662_266258

/-- Represents a box of socks -/
structure SockBox where
  red : ℕ
  black : ℕ

/-- The probability of drawing two red socks from the box -/
def prob_two_red (box : SockBox) : ℚ :=
  (box.red / (box.red + box.black)) * ((box.red - 1) / (box.red + box.black - 1))

/-- The total number of socks in the box -/
def total_socks (box : SockBox) : ℕ := box.red + box.black

theorem min_socks_in_box :
  ∃ (box : SockBox), prob_two_red box = 1/2 ∧
    ∀ (other : SockBox), prob_two_red other = 1/2 → total_socks box ≤ total_socks other :=
sorry

theorem min_socks_even_black :
  ∃ (box : SockBox), prob_two_red box = 1/2 ∧ box.black % 2 = 0 ∧
    ∀ (other : SockBox), prob_two_red other = 1/2 ∧ other.black % 2 = 0 →
      total_socks box ≤ total_socks other :=
sorry

end min_socks_in_box_min_socks_even_black_l2662_266258


namespace x_value_theorem_l2662_266282

theorem x_value_theorem (x y : ℝ) (h : (x - 1) / x = (y^3 + 3*y^2 - 4) / (y^3 + 3*y^2 - 5)) :
  x = y^3 + 3*y^2 - 5 := by
  sorry

end x_value_theorem_l2662_266282


namespace system_has_solution_solutions_for_a_nonpositive_or_one_solutions_for_a_between_zero_and_two_solutions_for_a_geq_two_l2662_266257

/-- The system of equations has at least one solution for all real a -/
theorem system_has_solution (a : ℝ) : ∃ x y : ℝ, 
  (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x) := by sorry

/-- Solutions for a ≤ 0 or a = 1 -/
theorem solutions_for_a_nonpositive_or_one (a : ℝ) (h : a ≤ 0 ∨ a = 1) : 
  (∃ x y : ℝ, x = 0 ∧ y = 1 - a / 2 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = 1 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) := by sorry

/-- Solutions for 0 < a < 2, a ≠ 1 -/
theorem solutions_for_a_between_zero_and_two (a : ℝ) (h : 0 < a ∧ a < 2 ∧ a ≠ 1) : 
  (∃ x y : ℝ, x = 0 ∧ y = 1 - a / 2 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = 1 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) ∧
  (∃ x y : ℝ, x = ((2 - a) / a)^2 ∧ y = (2 - a) / a ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) := by sorry

/-- Solutions for a ≥ 2 -/
theorem solutions_for_a_geq_two (a : ℝ) (h : a ≥ 2) : 
  ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x) := by sorry

end system_has_solution_solutions_for_a_nonpositive_or_one_solutions_for_a_between_zero_and_two_solutions_for_a_geq_two_l2662_266257


namespace sum_f_neg1_0_1_l2662_266295

-- Define the function f
variable (f : ℝ → ℝ)

-- State the condition
axiom f_add (x y : ℝ) : f x + f y = f (x + y)

-- State the theorem to be proved
theorem sum_f_neg1_0_1 : f (-1) + f 0 + f 1 = 0 := by
  sorry

end sum_f_neg1_0_1_l2662_266295


namespace units_digit_of_sequence_sum_l2662_266284

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sequence_term (n : ℕ) : ℕ := factorial n + n

def sum_sequence (n : ℕ) : ℕ := (List.range n).map sequence_term |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sequence_sum :
  units_digit (sum_sequence 12) = 1 := by sorry

end units_digit_of_sequence_sum_l2662_266284


namespace boat_speed_in_still_water_l2662_266208

/-- Proves that the speed of a boat in still water is 24 km/hr, given the speed of the stream and the downstream travel details. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 196)
  (h3 : downstream_time = 7)
  : ∃ (boat_speed : ℝ), boat_speed = 24 := by
  sorry

#check boat_speed_in_still_water

end boat_speed_in_still_water_l2662_266208


namespace box_volume_formula_l2662_266271

/-- The volume of a box formed by cutting squares from corners of a sheet -/
def boxVolume (x : ℝ) : ℝ := (16 - 2*x) * (12 - 2*x) * x

/-- The constraint on the side length of the cut squares -/
def sideConstraint (x : ℝ) : Prop := x ≤ 12/5

theorem box_volume_formula (x : ℝ) (h : sideConstraint x) :
  boxVolume x = 192*x - 56*x^2 + 4*x^3 := by
  sorry

end box_volume_formula_l2662_266271


namespace problem_solution_l2662_266212

theorem problem_solution (x y z : ℚ) : 
  x / y = 7 / 3 → y = 21 → z = 3 * y → x = 49 ∧ z = 63 := by
  sorry


end problem_solution_l2662_266212


namespace sqrt_over_thirteen_equals_four_l2662_266245

theorem sqrt_over_thirteen_equals_four :
  Real.sqrt 2704 / 13 = 4 := by
  sorry

end sqrt_over_thirteen_equals_four_l2662_266245


namespace intersection_point_correct_l2662_266286

/-- Represents a 2D point or vector -/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Represents a line in 2D space defined by a point and a direction vector -/
structure Line2D where
  point : Point2D
  direction : Point2D

/-- The first line -/
def line1 : Line2D := {
  point := { x := 3, y := 0 },
  direction := { x := 1, y := 2 }
}

/-- The second line -/
def line2 : Line2D := {
  point := { x := -1, y := 4 },
  direction := { x := 3, y := -1 }
}

/-- The proposed intersection point -/
def intersectionPoint : Point2D := {
  x := 30 / 7,
  y := 18 / 7
}

/-- Function to check if a point lies on a line -/
def isPointOnLine (p : Point2D) (l : Line2D) : Prop :=
  ∃ t : ℚ, p.x = l.point.x + t * l.direction.x ∧ p.y = l.point.y + t * l.direction.y

/-- Theorem stating that the proposed intersection point lies on both lines -/
theorem intersection_point_correct :
  isPointOnLine intersectionPoint line1 ∧ isPointOnLine intersectionPoint line2 := by
  sorry

end intersection_point_correct_l2662_266286


namespace hyperbola_real_axis_length_l2662_266278

/-- The length of the real axis of a hyperbola with equation x^2 - y^2/9 = 1 is 2 -/
theorem hyperbola_real_axis_length :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - y^2/9 = 1}
  ∃ (a : ℝ), a > 0 ∧ (∀ (p : ℝ × ℝ), p ∈ hyperbola → p.1 ≤ a) ∧ 2 * a = 2 :=
sorry

end hyperbola_real_axis_length_l2662_266278


namespace smallest_dual_representation_l2662_266225

/-- Represents a number in a given base with repeated digits -/
def repeatedDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a digit is valid in a given base -/
def isValidDigit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_representation : ∃ (n : Nat),
  (∃ (A : Nat), isValidDigit A 5 ∧ n = repeatedDigitNumber A 5) ∧
  (∃ (B : Nat), isValidDigit B 7 ∧ n = repeatedDigitNumber B 7) ∧
  (∀ (m : Nat),
    ((∃ (A : Nat), isValidDigit A 5 ∧ m = repeatedDigitNumber A 5) ∧
     (∃ (B : Nat), isValidDigit B 7 ∧ m = repeatedDigitNumber B 7))
    → m ≥ n) ∧
  n = 24 :=
sorry

end smallest_dual_representation_l2662_266225


namespace handshakes_count_l2662_266221

/-- Represents the number of people in the gathering -/
def total_people : ℕ := 40

/-- Represents the number of people in Group A who all know each other -/
def group_a_size : ℕ := 25

/-- Represents the number of people in Group B -/
def group_b_size : ℕ := 15

/-- Represents the number of people in Group B who know exactly 3 people from Group A -/
def group_b_connected : ℕ := 5

/-- Represents the number of people in Group B who know no one -/
def group_b_isolated : ℕ := 10

/-- Represents the number of people each connected person in Group B knows in Group A -/
def connections_per_person : ℕ := 3

/-- Calculates the total number of handshakes in the gathering -/
def total_handshakes : ℕ := 
  (group_b_isolated * group_a_size) + 
  (group_b_connected * (group_a_size - connections_per_person)) + 
  (group_b_isolated.choose 2)

/-- Theorem stating that the total number of handshakes is 405 -/
theorem handshakes_count : total_handshakes = 405 := by
  sorry

end handshakes_count_l2662_266221


namespace expand_expression_l2662_266255

theorem expand_expression (x : ℝ) : 5 * (2 * x^3 - 3 * x^2 + 4 * x - 1) = 10 * x^3 - 15 * x^2 + 20 * x - 5 := by
  sorry

end expand_expression_l2662_266255


namespace point_below_line_l2662_266250

theorem point_below_line (m : ℝ) : 
  ((-2 : ℝ) + m * (-1 : ℝ) - 1 < 0) ↔ (m < -3 ∨ m > 0) :=
by sorry

end point_below_line_l2662_266250


namespace fraction_c_simplest_form_l2662_266273

/-- A fraction is in its simplest form if its numerator and denominator have no common factors other than 1 and -1. -/
def IsSimplestForm (num den : ℤ → ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, (∀ k : ℤ, k ≠ 1 ∧ k ≠ -1 → (k ∣ num a b ↔ k ∣ den a b) → False)

/-- The fraction (3a + b) / (a + b) is in its simplest form. -/
theorem fraction_c_simplest_form :
  IsSimplestForm (fun a b => 3*a + b) (fun a b => a + b) := by
  sorry

#check fraction_c_simplest_form

end fraction_c_simplest_form_l2662_266273


namespace binomial_1293_1_l2662_266215

theorem binomial_1293_1 : Nat.choose 1293 1 = 1293 := by
  sorry

end binomial_1293_1_l2662_266215


namespace roses_per_decoration_correct_l2662_266200

/-- The number of white roses in each table decoration -/
def roses_per_decoration : ℕ := 12

/-- The number of bouquets -/
def num_bouquets : ℕ := 5

/-- The number of table decorations -/
def num_decorations : ℕ := 7

/-- The number of white roses in each bouquet -/
def roses_per_bouquet : ℕ := 5

/-- The total number of white roses used -/
def total_roses : ℕ := 109

theorem roses_per_decoration_correct :
  roses_per_decoration * num_decorations + roses_per_bouquet * num_bouquets = total_roses :=
by sorry

end roses_per_decoration_correct_l2662_266200


namespace exists_idempotent_l2662_266209

/-- A custom binary operation on a finite set -/
class CustomOperation (α : Type*) [Fintype α] where
  op : α → α → α

/-- Axioms for the custom operation -/
class CustomOperationAxioms (α : Type*) [Fintype α] [CustomOperation α] where
  closure : ∀ (a b : α), CustomOperation.op a b ∈ (Finset.univ : Finset α)
  property : ∀ (a b : α), CustomOperation.op (CustomOperation.op a b) a = b

/-- Theorem: There exists an element that is idempotent under the custom operation -/
theorem exists_idempotent (α : Type*) [Fintype α] [CustomOperation α] [CustomOperationAxioms α] :
  ∃ (a : α), CustomOperation.op a a = a :=
sorry

end exists_idempotent_l2662_266209


namespace product_divisible_by_eight_l2662_266277

theorem product_divisible_by_eight (n : ℤ) (h : 1 ≤ n ∧ n ≤ 96) : 
  8 ∣ (n * (n + 1) * (n + 2)) := by
  sorry

end product_divisible_by_eight_l2662_266277


namespace total_loss_proof_l2662_266267

/-- Represents the capital and loss of an investor -/
structure Investor where
  capital : ℝ
  loss : ℝ

/-- Calculates the total loss given two investors -/
def totalLoss (investor1 investor2 : Investor) : ℝ :=
  investor1.loss + investor2.loss

/-- Theorem: Given two investors with capitals in ratio 1:9 and losses proportional to their investments,
    if one investor loses Rs 900, the total loss is Rs 1000 -/
theorem total_loss_proof (investor1 investor2 : Investor) 
    (h1 : investor1.capital = (1/9) * investor2.capital)
    (h2 : investor1.loss / investor2.loss = investor1.capital / investor2.capital)
    (h3 : investor2.loss = 900) :
    totalLoss investor1 investor2 = 1000 := by
  sorry

#eval totalLoss { capital := 1, loss := 100 } { capital := 9, loss := 900 }

end total_loss_proof_l2662_266267


namespace triangle_properties_l2662_266297

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The radius of the circumcircle of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

theorem triangle_properties (t : Triangle) 
  (h1 : t.c ≥ t.a ∧ t.c ≥ t.b) 
  (h2 : t.b = Real.sqrt 3 * circumradius t)
  (h3 : t.b * Real.sin t.B = (t.a + t.c) * Real.sin t.A)
  (h4 : 0 < t.A ∧ t.A < Real.pi / 2)
  (h5 : 0 < t.B ∧ t.B < Real.pi / 2)
  (h6 : 0 < t.C ∧ t.C < Real.pi / 2) :
  t.B = Real.pi / 3 ∧ t.A = Real.pi / 6 ∧ t.C = Real.pi / 2 := by sorry

end triangle_properties_l2662_266297


namespace segments_intersection_l2662_266241

-- Define the number of segments
def n : ℕ := 1977

-- Define the type for segments
def Segment : Type := ℕ → Set ℝ

-- Define the property of intersection
def intersects (s1 s2 : Set ℝ) : Prop := ∃ x, x ∈ s1 ∧ x ∈ s2

-- State the theorem
theorem segments_intersection 
  (A B : Segment) 
  (h1 : ∀ k ∈ Finset.range n, intersects (A k) (B ((k + n - 1) % n)))
  (h2 : ∀ k ∈ Finset.range n, intersects (A k) (B ((k + 1) % n)))
  (h3 : intersects (A (n - 1)) (B 0))
  (h4 : intersects (A 0) (B (n - 1)))
  : ∃ k ∈ Finset.range n, intersects (A k) (B k) :=
by sorry

end segments_intersection_l2662_266241


namespace thirty_three_million_scientific_notation_l2662_266228

/-- Proves that 33 million is equal to 3.3 × 10^7 in scientific notation -/
theorem thirty_three_million_scientific_notation :
  (33 : ℝ) * 1000000 = 3.3 * (10 : ℝ)^7 := by
  sorry

end thirty_three_million_scientific_notation_l2662_266228


namespace yans_distance_ratio_l2662_266294

/-- Yan's problem statement -/
theorem yans_distance_ratio :
  ∀ (w x y : ℝ),
  w > 0 →  -- walking speed is positive
  x > 0 →  -- distance from Yan to home is positive
  y > 0 →  -- distance from Yan to stadium is positive
  x + y > 0 →  -- Yan is between home and stadium
  y / w = (x / w + (x + y) / (9 * w)) →  -- both choices take the same time
  x / y = 4 / 5 := by
  sorry

end yans_distance_ratio_l2662_266294


namespace irrationality_of_pi_and_rationality_of_others_l2662_266239

-- Define irrational numbers
def IsIrrational (x : ℝ) : Prop :=
  ∀ a b : ℤ, b ≠ 0 → x ≠ a / b

-- State the theorem
theorem irrationality_of_pi_and_rationality_of_others :
  IsIrrational Real.pi ∧ ¬IsIrrational 0 ∧ ¬IsIrrational (-1/3) ∧ ¬IsIrrational (3/2) :=
sorry

end irrationality_of_pi_and_rationality_of_others_l2662_266239
