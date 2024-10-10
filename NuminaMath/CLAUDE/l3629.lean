import Mathlib

namespace blue_cards_count_l3629_362938

theorem blue_cards_count (total_cards : ℕ) (blue_cards : ℕ) : 
  (10 : ℕ) + blue_cards = total_cards →
  (blue_cards : ℚ) / total_cards = 4/5 →
  blue_cards = 40 := by
  sorry

end blue_cards_count_l3629_362938


namespace expense_increase_percentage_l3629_362904

def monthly_salary : ℝ := 6250
def initial_savings_rate : ℝ := 0.20
def final_savings : ℝ := 250

theorem expense_increase_percentage :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_savings - final_savings
  let percentage_increase := (expense_increase / initial_expenses) * 100
  percentage_increase = 20 := by sorry

end expense_increase_percentage_l3629_362904


namespace m_less_than_two_l3629_362941

open Real

/-- Proposition p -/
def p (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 + 1 > 0

/-- Proposition q -/
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 ≤ 0

/-- The main theorem -/
theorem m_less_than_two (m : ℝ) (h : ¬(p m) ∨ ¬(q m)) : m < 2 := by
  sorry

end m_less_than_two_l3629_362941


namespace consecutive_integers_average_l3629_362919

theorem consecutive_integers_average (a : ℕ) (c : ℕ) (h1 : c = 3 * a + 3) : 
  (c + (c + 1) + (c + 2)) / 3 = 3 * a + 4 := by
  sorry

end consecutive_integers_average_l3629_362919


namespace number_of_pupils_l3629_362983

/-- Represents the number of pupils in the class -/
def n : ℕ := sorry

/-- The correct first mark -/
def correct_first_mark : ℕ := 63

/-- The incorrect first mark -/
def incorrect_first_mark : ℕ := 83

/-- The correct second mark -/
def correct_second_mark : ℕ := 85

/-- The incorrect second mark -/
def incorrect_second_mark : ℕ := 75

/-- The weight for the first mark -/
def weight_first : ℕ := 3

/-- The weight for the second mark -/
def weight_second : ℕ := 2

/-- The increase in average marks due to the errors -/
def average_increase : ℚ := 1/2

theorem number_of_pupils : n = 80 := by
  sorry

end number_of_pupils_l3629_362983


namespace inscribed_triangle_condition_l3629_362910

/-- A rectangle with side lengths a and b. -/
structure Rectangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- An equilateral triangle inscribed in a rectangle such that one vertex is at A
    and the other two vertices lie on sides BC and CD respectively. -/
structure InscribedTriangle (rect : Rectangle) where
  vertex_on_BC : ℝ
  vertex_on_CD : ℝ
  vertex_on_BC_in_range : 0 ≤ vertex_on_BC ∧ vertex_on_BC ≤ rect.b
  vertex_on_CD_in_range : 0 ≤ vertex_on_CD ∧ vertex_on_CD ≤ rect.a
  is_equilateral : True  -- We assume this condition is met

/-- The theorem stating the condition for inscribing an equilateral triangle in a rectangle. -/
theorem inscribed_triangle_condition (rect : Rectangle) :
  (∃ t : InscribedTriangle rect, True) ↔ 
  (Real.sqrt 3 / 2 ≤ rect.a / rect.b ∧ rect.a / rect.b ≤ 2 / Real.sqrt 3) :=
sorry

end inscribed_triangle_condition_l3629_362910


namespace baker_cakes_l3629_362907

theorem baker_cakes (pastries_made : ℕ) (cakes_sold : ℕ) (pastries_sold : ℕ) 
  (h1 : pastries_made = 169)
  (h2 : cakes_sold = pastries_sold + 11)
  (h3 : cakes_sold = 158)
  (h4 : pastries_sold = 147) :
  pastries_made + 11 = 180 := by
  sorry

end baker_cakes_l3629_362907


namespace hyperbola_foci_on_x_axis_l3629_362912

/-- A curve C defined by mx^2 + (2-m)y^2 = 1 is a hyperbola with foci on the x-axis if and only if m ∈ (2, +∞) -/
theorem hyperbola_foci_on_x_axis (m : ℝ) :
  (∀ x y : ℝ, m * x^2 + (2 - m) * y^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, c > 0 ∧ ∀ x y : ℝ, x^2 / (a^2 + c^2) + y^2 / a^2 = 1) →
  m > 2 :=
by sorry

end hyperbola_foci_on_x_axis_l3629_362912


namespace quadratic_inequality_iff_abs_a_le_two_l3629_362957

theorem quadratic_inequality_iff_abs_a_le_two (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ |a| ≤ 2 :=
by sorry

end quadratic_inequality_iff_abs_a_le_two_l3629_362957


namespace cos_240_degrees_l3629_362955

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end cos_240_degrees_l3629_362955


namespace intersection_M_complement_N_l3629_362921

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x = 0}

def N : Set ℝ := {x | x - 1 > 0}

theorem intersection_M_complement_N : M ∩ (U \ N) = {0} := by
  sorry

end intersection_M_complement_N_l3629_362921


namespace milk_consumption_ratio_l3629_362958

/-- The ratio of Minyoung's milk consumption to Yuna's milk consumption -/
theorem milk_consumption_ratio (minyoung_milk yuna_milk : ℚ) 
  (h1 : minyoung_milk = 10)
  (h2 : yuna_milk = 2/3) :
  minyoung_milk / yuna_milk = 15 := by
sorry

end milk_consumption_ratio_l3629_362958


namespace largest_of_three_consecutive_multiples_of_three_l3629_362932

theorem largest_of_three_consecutive_multiples_of_three (a b c : ℕ) : 
  (∃ n : ℕ, a = 3 * n ∧ b = 3 * (n + 1) ∧ c = 3 * (n + 2)) → 
  a + b + c = 72 → 
  max a (max b c) = 27 :=
by
  sorry

end largest_of_three_consecutive_multiples_of_three_l3629_362932


namespace girls_in_chemistry_class_l3629_362924

theorem girls_in_chemistry_class (total : ℕ) (girls boys : ℕ) : 
  total = 70 →
  girls + boys = total →
  4 * boys = 3 * girls →
  girls = 40 :=
by sorry

end girls_in_chemistry_class_l3629_362924


namespace polynomial_evaluation_at_negative_two_l3629_362937

theorem polynomial_evaluation_at_negative_two :
  let f : ℝ → ℝ := λ x ↦ x^3 + x^2 + 2*x + 2
  f (-2) = -6 := by
  sorry

end polynomial_evaluation_at_negative_two_l3629_362937


namespace book_selection_problem_l3629_362902

theorem book_selection_problem (total_books math_books physics_books selected_books selected_math selected_physics : ℕ) :
  total_books = 20 →
  math_books = 6 →
  physics_books = 4 →
  selected_books = 8 →
  selected_math = 4 →
  selected_physics = 2 →
  (Nat.choose math_books selected_math) * (Nat.choose physics_books selected_physics) *
  (Nat.choose (total_books - math_books - physics_books) (selected_books - selected_math - selected_physics)) = 4050 := by
  sorry

end book_selection_problem_l3629_362902


namespace total_books_l3629_362954

theorem total_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 8)
  (h2 : mystery_shelves = 12)
  (h3 : picture_shelves = 9) :
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 168 :=
by
  sorry

end total_books_l3629_362954


namespace worm_domino_division_l3629_362911

/-- A worm is represented by a list of directions (Up or Right) -/
inductive Direction
| Up
| Right

def Worm := List Direction

/-- Count the number of cells in a worm -/
def cellCount (w : Worm) : Nat :=
  w.length + 1

/-- Predicate to check if a worm can be divided into n dominoes -/
def canDivideIntoDominoes (w : Worm) (n : Nat) : Prop :=
  ∃ (division : List (Worm × Worm)), 
    division.length = n ∧
    (division.map (λ (p : Worm × Worm) => cellCount p.1 + cellCount p.2)).sum = cellCount w

/-- The main theorem -/
theorem worm_domino_division (w : Worm) (n : Nat) :
  n > 2 → (canDivideIntoDominoes w n ↔ cellCount w = 2 * n) :=
by sorry

end worm_domino_division_l3629_362911


namespace fruits_in_red_basket_l3629_362967

theorem fruits_in_red_basket :
  let blue_bananas : ℕ := 12
  let blue_apples : ℕ := 4
  let blue_total : ℕ := blue_bananas + blue_apples
  let red_total : ℕ := blue_total / 2
  red_total = 8 := by sorry

end fruits_in_red_basket_l3629_362967


namespace spending_recording_l3629_362959

/-- Represents the recording of a financial transaction -/
def record (amount : ℤ) : ℤ := amount

/-- Axiom: Depositing is recorded as a positive amount -/
axiom deposit_positive (amount : ℕ) : record amount = amount

/-- The main theorem: If depositing 300 is recorded as +300, then spending 500 should be recorded as -500 -/
theorem spending_recording :
  record 300 = 300 → record (-500) = -500 := by
  sorry

end spending_recording_l3629_362959


namespace only_A_is_impossible_l3629_362940

-- Define the set of possible ball colors in the bag
inductive BallColor
| Red
| White

-- Define the set of possible outcomes for a dice roll
inductive DiceOutcome
| One | Two | Three | Four | Five | Six

-- Define the set of possible last digits for a license plate
inductive LicensePlateLastDigit
| Zero | One | Two | Three | Four | Five | Six | Seven | Eight | Nine

-- Define the events
def event_A : Prop := ∃ (ball : BallColor), ball = BallColor.Red ∨ ball = BallColor.White

def event_B : Prop := True  -- We can't model weather prediction precisely, so we assume it's always possible

def event_C : Prop := ∃ (outcome : DiceOutcome), outcome = DiceOutcome.Six

def event_D : Prop := ∃ (digit : LicensePlateLastDigit), 
  digit = LicensePlateLastDigit.Zero ∨ 
  digit = LicensePlateLastDigit.Two ∨ 
  digit = LicensePlateLastDigit.Four ∨ 
  digit = LicensePlateLastDigit.Six ∨ 
  digit = LicensePlateLastDigit.Eight

-- Theorem stating that only event A is impossible
theorem only_A_is_impossible :
  (¬ event_A) ∧ event_B ∧ event_C ∧ event_D :=
sorry

end only_A_is_impossible_l3629_362940


namespace line_and_volume_proof_l3629_362956

-- Define the line l
def line_l (x y : ℝ) := x + y - 4 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) := x + y - 1 = 0

-- Theorem statement
theorem line_and_volume_proof :
  -- Condition 1: Line l passes through (3,1)
  line_l 3 1 ∧
  -- Condition 2: Line l is parallel to x+y-1=0
  ∀ (x y : ℝ), line_l x y ↔ ∃ (k : ℝ), parallel_line (x + k) (y + k) →
  -- Conclusion 1: Equation of line l is x+y-4=0
  (∀ (x y : ℝ), line_l x y ↔ x + y - 4 = 0) ∧
  -- Conclusion 2: Volume of the geometric solid is (64/3)π
  (let volume := (64 / 3) * Real.pi
   volume = (1 / 3) * Real.pi * 4^2 * 4) :=
by sorry

end line_and_volume_proof_l3629_362956


namespace parallelogram_on_circle_l3629_362999

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is inside a circle -/
def isInside (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

/-- Check if a point is on a circle -/
def isOn (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if four points form a parallelogram -/
def isParallelogram (a b c d : ℝ × ℝ) : Prop :=
  (b.1 - a.1 = d.1 - c.1) ∧ (b.2 - a.2 = d.2 - c.2) ∧
  (c.1 - b.1 = a.1 - d.1) ∧ (c.2 - b.2 = a.2 - d.2)

theorem parallelogram_on_circle (ω : Circle) (A B : ℝ × ℝ) 
  (h_A : isInside ω A) (h_B : isOn ω B) :
  ∃ (C D : ℝ × ℝ), isOn ω C ∧ isOn ω D ∧ isParallelogram A B C D :=
sorry

end parallelogram_on_circle_l3629_362999


namespace square_root_equation_l3629_362947

theorem square_root_equation (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end square_root_equation_l3629_362947


namespace mary_promised_cards_l3629_362988

/-- The number of baseball cards Mary promised to give Fred -/
def promised_cards (initial : ℝ) (bought : ℝ) (left : ℝ) : ℝ :=
  initial + bought - left

theorem mary_promised_cards :
  promised_cards 18.0 40.0 32.0 = 26.0 := by
  sorry

end mary_promised_cards_l3629_362988


namespace quadratic_solution_l3629_362935

theorem quadratic_solution (x a : ℝ) : x = 3 ∧ x^2 = a → a = 9 := by sorry

end quadratic_solution_l3629_362935


namespace cat_food_bags_l3629_362949

theorem cat_food_bags (cat_food_weight : ℕ) (dog_food_bags : ℕ) (weight_difference : ℕ) (ounces_per_pound : ℕ) (total_ounces : ℕ) : 
  cat_food_weight = 3 →
  dog_food_bags = 2 →
  weight_difference = 2 →
  ounces_per_pound = 16 →
  total_ounces = 256 →
  ∃ (x : ℕ), x * cat_food_weight * ounces_per_pound + 
    dog_food_bags * (cat_food_weight + weight_difference) * ounces_per_pound = total_ounces ∧ 
    x = 2 :=
by sorry

end cat_food_bags_l3629_362949


namespace nested_radical_solution_l3629_362913

theorem nested_radical_solution :
  ∃! x : ℝ, x > 0 ∧ x = Real.sqrt (3 + x) :=
by
  use (1 + Real.sqrt 13) / 2
  sorry

end nested_radical_solution_l3629_362913


namespace expression_value_l3629_362936

theorem expression_value (x : ℝ) (h : x^2 - 4*x - 1 = 0) : 
  (x - 3) / (x - 4) - 1 / x = 5 := by
  sorry

end expression_value_l3629_362936


namespace fractional_factorial_max_experiments_l3629_362963

/-- The number of experimental points -/
def n : ℕ := 20

/-- The maximum number of experiments needed -/
def max_experiments : ℕ := 6

/-- Theorem stating that for 20 experimental points, 
    the maximum number of experiments needed is 6 
    when using the fractional factorial design method -/
theorem fractional_factorial_max_experiments :
  n = 2^max_experiments - 1 := by sorry

end fractional_factorial_max_experiments_l3629_362963


namespace percent_relation_l3629_362969

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : c = 0.50 * b) : b = 0.50 * a := by
  sorry

end percent_relation_l3629_362969


namespace equation_solution_l3629_362984

theorem equation_solution :
  ∃! x : ℚ, (4 * x - 2) / (5 * x - 5) = 3 / 4 ∧ x = -7 :=
by sorry

end equation_solution_l3629_362984


namespace total_income_calculation_l3629_362923

/-- Calculates the total income for a clothing store sale --/
def calculate_total_income (tshirt_price : ℚ) (pants_price : ℚ) (skirt_price : ℚ) 
                           (refurbished_tshirt_price : ℚ) (skirt_discount_rate : ℚ) 
                           (tshirt_discount_rate : ℚ) (sales_tax_rate : ℚ) 
                           (tshirts_sold : ℕ) (refurbished_tshirts_sold : ℕ) 
                           (pants_sold : ℕ) (skirts_sold : ℕ) : ℚ :=
  sorry

theorem total_income_calculation :
  let tshirt_price : ℚ := 5
  let pants_price : ℚ := 4
  let skirt_price : ℚ := 6
  let refurbished_tshirt_price : ℚ := tshirt_price / 2
  let skirt_discount_rate : ℚ := 1 / 10
  let tshirt_discount_rate : ℚ := 1 / 5
  let sales_tax_rate : ℚ := 2 / 25
  let tshirts_sold : ℕ := 15
  let refurbished_tshirts_sold : ℕ := 7
  let pants_sold : ℕ := 6
  let skirts_sold : ℕ := 12
  calculate_total_income tshirt_price pants_price skirt_price refurbished_tshirt_price 
                         skirt_discount_rate tshirt_discount_rate sales_tax_rate
                         tshirts_sold refurbished_tshirts_sold pants_sold skirts_sold = 1418 / 10 :=
by
  sorry

end total_income_calculation_l3629_362923


namespace quadrilateral_area_with_diagonal_and_offsets_l3629_362965

/-- The area of a quadrilateral with a diagonal and its offsets -/
theorem quadrilateral_area_with_diagonal_and_offsets 
  (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) :
  diagonal = 40 → offset1 = 9 → offset2 = 6 →
  (1/2 * diagonal * offset1) + (1/2 * diagonal * offset2) = 300 :=
by sorry

end quadrilateral_area_with_diagonal_and_offsets_l3629_362965


namespace intersection_A_complement_B_l3629_362927

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {1,3,5}
def B : Set Nat := {2,3}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1,5} := by
  sorry

end intersection_A_complement_B_l3629_362927


namespace f_property_l3629_362909

def property_P (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 ≤ k ∧ k + 1 < n ∧
  2 * Nat.choose n k = Nat.choose n (k - 1) + Nat.choose n (k + 1)

theorem f_property :
  (property_P 7) ∧
  (∀ n : ℕ, n ≤ 2016 → property_P n → n ≤ 1934) ∧
  (property_P 1934) :=
sorry

end f_property_l3629_362909


namespace odd_function_implies_a_value_f_is_increasing_f_range_l3629_362934

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 3^x / (3^x + 1) - a

theorem odd_function_implies_a_value (a : ℝ) :
  (∀ x, f x a = -f (-x) a) → a = 1/2 := by sorry

theorem f_is_increasing (a : ℝ) (h : a = 1/2) :
  Monotone (f · a) := by sorry

theorem f_range (a : ℝ) (h : a = 1/2) :
  Set.range (f · a) = Set.Ioo (-1/2 : ℝ) (1/2 : ℝ) := by sorry

end odd_function_implies_a_value_f_is_increasing_f_range_l3629_362934


namespace function_equality_l3629_362973

open Real

theorem function_equality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x = x^2 + 2 * (deriv f 1) * x + 3) : 
  f 0 = f 4 := by
  sorry

end function_equality_l3629_362973


namespace union_of_A_and_B_l3629_362908

-- Define the sets A and B
def A : Set ℝ := {x | |x| < 3}
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x < 3} := by sorry

end union_of_A_and_B_l3629_362908


namespace white_bread_loaves_l3629_362945

/-- Given that a restaurant served 0.2 loaf of wheat bread and 0.6 loaves in total,
    prove that the number of loaves of white bread served is 0.4. -/
theorem white_bread_loaves (wheat_bread : Real) (total_bread : Real)
    (h1 : wheat_bread = 0.2)
    (h2 : total_bread = 0.6) :
    total_bread - wheat_bread = 0.4 := by
  sorry

end white_bread_loaves_l3629_362945


namespace probability_at_least_nine_correct_l3629_362900

theorem probability_at_least_nine_correct (n : ℕ) (p : ℝ) : 
  n = 10 → 
  p = 1/4 → 
  let P := (n.choose 9) * p^9 * (1-p)^1 + (n.choose 10) * p^10
  ∃ ε > 0, abs (P - 3e-5) < ε := by sorry

end probability_at_least_nine_correct_l3629_362900


namespace number_fraction_problem_l3629_362918

theorem number_fraction_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 17 → (40/100 : ℝ) * N = 204 := by
  sorry

end number_fraction_problem_l3629_362918


namespace min_sum_squares_l3629_362905

theorem min_sum_squares (a b c d : ℝ) (h : a + 3*b + 5*c + 7*d = 14) :
  ∃ (m : ℝ), (∀ (x y z w : ℝ), x + 3*y + 5*z + 7*w = 14 → x^2 + y^2 + z^2 + w^2 ≥ m) ∧
             (a^2 + b^2 + c^2 + d^2 = m) ∧
             (m = 7/3) :=
by sorry

end min_sum_squares_l3629_362905


namespace solve_for_a_l3629_362994

theorem solve_for_a : ∃ a : ℝ, (1 : ℝ) - a * 2 = 3 ∧ a = -1 := by
  sorry

end solve_for_a_l3629_362994


namespace exists_touching_arrangement_l3629_362944

/-- Represents a coin as a circle in a 2D plane -/
structure Coin where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two coins are touching -/
def are_touching (c1 c2 : Coin) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Represents an arrangement of five coins -/
structure CoinArrangement where
  coins : Fin 5 → Coin
  all_same_size : ∀ i j, (coins i).radius = (coins j).radius

/-- Theorem stating that there exists an arrangement where each coin touches exactly four others -/
theorem exists_touching_arrangement :
  ∃ (arr : CoinArrangement), ∀ i : Fin 5, (∃! j : Fin 5, ¬(are_touching (arr.coins i) (arr.coins j))) :=
sorry

end exists_touching_arrangement_l3629_362944


namespace unique_k_for_prime_roots_l3629_362916

/-- A natural number is prime if it's greater than 1 and its only positive divisors are 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The roots of the quadratic equation x^2 - 73x + k = 0 -/
def roots (k : ℕ) : Set ℝ :=
  {x : ℝ | x^2 - 73*x + k = 0}

/-- The statement that both roots of x^2 - 73x + k = 0 are prime numbers -/
def both_roots_prime (k : ℕ) : Prop :=
  ∀ x ∈ roots k, ∃ n : ℕ, (x : ℝ) = n ∧ is_prime n

/-- There is exactly one value of k such that both roots of x^2 - 73x + k = 0 are prime numbers -/
theorem unique_k_for_prime_roots : ∃! k : ℕ, both_roots_prime k :=
  sorry

end unique_k_for_prime_roots_l3629_362916


namespace min_disks_for_problem_l3629_362961

/-- Represents a file with its size in MB -/
structure File where
  size : Float

/-- Represents a disk with its capacity in MB -/
structure Disk where
  capacity : Float

/-- Function to calculate the minimum number of disks needed -/
def min_disks_needed (files : List File) (disk_capacity : Float) : Nat :=
  sorry

/-- Theorem stating the minimum number of disks needed for the given problem -/
theorem min_disks_for_problem : 
  let files : List File := 
    (List.replicate 5 ⟨1.0⟩) ++ 
    (List.replicate 15 ⟨0.6⟩) ++ 
    (List.replicate 25 ⟨0.3⟩)
  let disk_capacity : Float := 1.44
  min_disks_needed files disk_capacity = 16 := by
  sorry

end min_disks_for_problem_l3629_362961


namespace fruit_stand_problem_l3629_362962

/-- Represents the fruit stand problem --/
structure FruitStand where
  apple_price : ℝ
  banana_price : ℝ
  orange_price : ℝ
  apple_discount : ℝ
  min_fruit_qty : ℕ
  emmy_budget : ℝ
  gerry_budget : ℝ

/-- Calculates the maximum number of apples that can be bought --/
def max_apples (fs : FruitStand) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem fruit_stand_problem :
  let fs : FruitStand := {
    apple_price := 2,
    banana_price := 1,
    orange_price := 3,
    apple_discount := 0.2,
    min_fruit_qty := 5,
    emmy_budget := 200,
    gerry_budget := 100
  }
  max_apples fs = 160 :=
sorry

end fruit_stand_problem_l3629_362962


namespace min_xyz_value_l3629_362980

theorem min_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y + z = (x + z) * (y + z)) :
  x * y * z ≥ (1 : ℝ) / 27 := by sorry

end min_xyz_value_l3629_362980


namespace line_intersects_circle_l3629_362901

/-- The line x - ky + 1 = 0 (k ∈ ℝ) always intersects the circle x^2 + y^2 + 4x - 2y + 2 = 0 -/
theorem line_intersects_circle (k : ℝ) : 
  ∃ (x y : ℝ), 
    (x - k*y + 1 = 0) ∧ 
    (x^2 + y^2 + 4*x - 2*y + 2 = 0) :=
by sorry


end line_intersects_circle_l3629_362901


namespace range_of_x_l3629_362946

theorem range_of_x (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -2) :
  x > 1 / 3 ∨ x < -1 / 2 := by
  sorry

end range_of_x_l3629_362946


namespace trip_theorem_l3629_362939

/-- Represents the ticket prices and group sizes for a school trip -/
structure TripData where
  adultPrice : ℕ
  studentDiscount : ℚ
  groupDiscount : ℚ
  totalPeople : ℕ
  totalCost : ℕ

/-- Calculates the number of adults and students in the group -/
def calculateGroup (data : TripData) : ℕ × ℕ :=
  sorry

/-- Calculates the cost of tickets for different purchasing strategies -/
def calculateCosts (data : TripData) (adults : ℕ) (students : ℕ) : ℕ × ℕ × ℕ :=
  sorry

/-- Theorem stating the correct number of adults and students, and the most cost-effective purchasing strategy -/
theorem trip_theorem (data : TripData) 
  (h1 : data.adultPrice = 120)
  (h2 : data.studentDiscount = 1/2)
  (h3 : data.groupDiscount = 3/5)
  (h4 : data.totalPeople = 130)
  (h5 : data.totalCost = 9600) :
  let (adults, students) := calculateGroup data
  let (regularCost, allGroupCost, mixedCost) := calculateCosts data adults students
  adults = 30 ∧ 
  students = 100 ∧ 
  mixedCost < allGroupCost ∧
  mixedCost < regularCost :=
sorry

end trip_theorem_l3629_362939


namespace solve_equation_l3629_362906

theorem solve_equation (x : ℝ) : 5 + 7 / x = 6 - 5 / x → x = 12 := by
  sorry

end solve_equation_l3629_362906


namespace payment_is_two_l3629_362976

/-- The amount Edmund needs to save -/
def saving_goal : ℕ := 75

/-- The number of chores Edmund normally does per week -/
def normal_chores_per_week : ℕ := 12

/-- The number of chores Edmund does per day during the saving period -/
def chores_per_day : ℕ := 4

/-- The number of days Edmund works during the saving period -/
def working_days : ℕ := 14

/-- The total amount Edmund earns for extra chores -/
def total_earned : ℕ := 64

/-- Calculates the number of extra chores Edmund does -/
def extra_chores : ℕ := chores_per_day * working_days - normal_chores_per_week * 2

/-- The payment per extra chore -/
def payment_per_extra_chore : ℚ := total_earned / extra_chores

theorem payment_is_two :
  payment_per_extra_chore = 2 := by sorry

end payment_is_two_l3629_362976


namespace shopkeeper_face_cards_l3629_362995

/-- The number of complete decks of playing cards the shopkeeper has -/
def num_decks : ℕ := 5

/-- The number of face cards in a standard deck of playing cards -/
def face_cards_per_deck : ℕ := 12

/-- The total number of face cards the shopkeeper has -/
def total_face_cards : ℕ := num_decks * face_cards_per_deck

theorem shopkeeper_face_cards : total_face_cards = 60 := by
  sorry

end shopkeeper_face_cards_l3629_362995


namespace middle_card_is_five_l3629_362948

/-- Represents a set of three cards with distinct positive integers. -/
structure CardSet where
  left : ℕ+
  middle : ℕ+
  right : ℕ+
  distinct : left ≠ middle ∧ middle ≠ right ∧ left ≠ right
  ascending : left < middle ∧ middle < right
  sum_15 : left + middle + right = 15

/-- Predicate for Ada's statement about the leftmost card -/
def ada_statement (cs : CardSet) : Prop :=
  ∃ cs' : CardSet, cs'.left = cs.left ∧ cs' ≠ cs

/-- Predicate for Bella's statement about the rightmost card -/
def bella_statement (cs : CardSet) : Prop :=
  ∃ cs' : CardSet, cs'.right = cs.right ∧ cs' ≠ cs

/-- The main theorem stating that the middle card must be 5 -/
theorem middle_card_is_five :
  ∀ cs : CardSet,
    ada_statement cs →
    bella_statement cs →
    cs.middle = 5 :=
sorry

end middle_card_is_five_l3629_362948


namespace john_yearly_expenses_l3629_362931

/-- Calculates the total amount John needs to pay for his EpiPens and additional medical expenses for a year. -/
def total_yearly_expenses (epipen_cost : ℚ) (first_epipen_coverage : ℚ) (second_epipen_coverage : ℚ) (yearly_medical_expenses : ℚ) (medical_expenses_coverage : ℚ) : ℚ :=
  let first_epipen_payment := epipen_cost * (1 - first_epipen_coverage)
  let second_epipen_payment := epipen_cost * (1 - second_epipen_coverage)
  let total_epipen_cost := first_epipen_payment + second_epipen_payment
  let medical_expenses_payment := yearly_medical_expenses * (1 - medical_expenses_coverage)
  total_epipen_cost + medical_expenses_payment

/-- Theorem stating that John's total yearly expenses are $725 given the problem conditions. -/
theorem john_yearly_expenses :
  total_yearly_expenses 500 0.75 0.6 2000 0.8 = 725 := by
  sorry

end john_yearly_expenses_l3629_362931


namespace max_y_over_x_l3629_362974

theorem max_y_over_x (x y : ℝ) 
  (h1 : x - 1 ≥ 0) 
  (h2 : x - y ≥ 0) 
  (h3 : x + y - 4 ≤ 0) : 
  ∃ (max : ℝ), max = 3 ∧ ∀ (x' y' : ℝ), 
    x' - 1 ≥ 0 → x' - y' ≥ 0 → x' + y' - 4 ≤ 0 → y' / x' ≤ max :=
by sorry

end max_y_over_x_l3629_362974


namespace expression_evaluation_l3629_362951

/-- Evaluates the given expression for x = 1.5 and y = -2 -/
theorem expression_evaluation :
  let x : ℝ := 1.5
  let y : ℝ := -2
  let expr := (1.2 * x^3 + 4 * y) * (0.86)^3 - (0.1)^3 / (0.86)^2 + 0.086 + (0.1)^2 * (2 * x^2 - 3 * y^2)
  ∃ ε > 0, |expr + 2.5027737774| < ε :=
by
  sorry

end expression_evaluation_l3629_362951


namespace marcy_cat_time_l3629_362996

def total_time (petting combing brushing playing feeding cleaning : ℚ) : ℚ :=
  petting + combing + brushing + playing + feeding + cleaning

theorem marcy_cat_time : 
  let petting : ℚ := 12
  let combing : ℚ := (1/3) * petting
  let brushing : ℚ := (1/4) * combing
  let playing : ℚ := (1/2) * petting
  let feeding : ℚ := 5
  let cleaning : ℚ := (2/5) * feeding
  total_time petting combing brushing playing feeding cleaning = 30 := by
sorry

end marcy_cat_time_l3629_362996


namespace at_least_one_passes_l3629_362926

theorem at_least_one_passes (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end at_least_one_passes_l3629_362926


namespace chip_credit_card_balance_l3629_362966

/-- Calculates the final balance on a credit card after two months with interest --/
def final_balance (initial_balance : ℝ) (interest_rate : ℝ) (additional_charge : ℝ) : ℝ :=
  let balance_after_first_month := initial_balance * (1 + interest_rate)
  let balance_before_second_interest := balance_after_first_month + additional_charge
  balance_before_second_interest * (1 + interest_rate)

/-- Theorem stating the final balance on Chip's credit card --/
theorem chip_credit_card_balance :
  final_balance 50 0.2 20 = 96 :=
by sorry

end chip_credit_card_balance_l3629_362966


namespace odd_function_sum_l3629_362979

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : is_odd f) (h_pos : ∀ x > 0, f x = 2*x - 3) :
  f (-2) + f 0 = -1 := by
  sorry

end odd_function_sum_l3629_362979


namespace acid_solution_mixing_l3629_362917

theorem acid_solution_mixing (y z : ℝ) (hy : y > 25) :
  (y * y / 100 + z * 40 / 100) / (y + z) * 100 = y + 10 →
  z = 10 * y / (y - 30) := by
sorry

end acid_solution_mixing_l3629_362917


namespace third_measurement_is_integer_meters_l3629_362920

def tape_length : ℕ := 100
def length1 : ℕ := 600
def length2 : ℕ := 500

theorem third_measurement_is_integer_meters :
  ∃ (k : ℕ), ∀ (third_length : ℕ),
    (tape_length ∣ length1) ∧
    (tape_length ∣ length2) ∧
    (tape_length ∣ third_length) →
    ∃ (n : ℕ), third_length = n * 100 := by
  sorry

end third_measurement_is_integer_meters_l3629_362920


namespace neg_a_cubed_times_a_squared_l3629_362991

theorem neg_a_cubed_times_a_squared (a : ℝ) : (-a)^3 * a^2 = -a^5 := by
  sorry

end neg_a_cubed_times_a_squared_l3629_362991


namespace sqrt_50_plus_sqrt_32_l3629_362925

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end sqrt_50_plus_sqrt_32_l3629_362925


namespace f_symmetric_about_x_eq_2_l3629_362914

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if -2 ≤ x ∧ x ≤ 0 then 2^x - 2^(-x) + x else 0  -- We define f on [-2,0] as given, and 0 elsewhere

-- State the theorem
theorem f_symmetric_about_x_eq_2 :
  (∀ x, x * f x = -x * f (-x)) →  -- y = xf(x) is even
  (∀ x, f (x - 1) + f (x + 3) = 0) →  -- given condition
  (∀ x, f (x - 2) = f (-x + 2)) :=  -- symmetry about x = 2
by sorry

end f_symmetric_about_x_eq_2_l3629_362914


namespace right_triangle_sets_l3629_362953

/-- A function that checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (6, 8, 11) cannot form a right-angled triangle --/
theorem right_triangle_sets :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 8 15 17 ∧
  is_right_triangle 7 24 25 ∧
  ¬(is_right_triangle 6 8 11) :=
by sorry

end right_triangle_sets_l3629_362953


namespace initial_amount_sufficient_l3629_362977

/-- Kanul's initial amount of money -/
def initial_amount : ℝ := 11058.82

/-- Raw materials cost -/
def raw_materials_cost : ℝ := 5000

/-- Machinery cost -/
def machinery_cost : ℝ := 200

/-- Employee wages -/
def employee_wages : ℝ := 1200

/-- Maintenance cost percentage -/
def maintenance_percentage : ℝ := 0.15

/-- Desired remaining balance -/
def desired_balance : ℝ := 3000

/-- Theorem: Given the expenses and conditions, the initial amount is sufficient -/
theorem initial_amount_sufficient :
  initial_amount - (raw_materials_cost + machinery_cost + employee_wages + maintenance_percentage * initial_amount) ≥ desired_balance := by
  sorry

#check initial_amount_sufficient

end initial_amount_sufficient_l3629_362977


namespace shortest_distance_circle_to_origin_l3629_362989

/-- The shortest distance between any point on the circle (x-2)^2+(y+m-4)^2=1 and the origin (0,0) is 1, where m is a real number. -/
theorem shortest_distance_circle_to_origin :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), (x - 2)^2 + (y + m - 4)^2 = 1) →
  (∃ (d : ℝ), d = 1 ∧ 
    ∀ (x y : ℝ), (x - 2)^2 + (y + m - 4)^2 = 1 → 
      Real.sqrt (x^2 + y^2) ≥ d) :=
by sorry

end shortest_distance_circle_to_origin_l3629_362989


namespace sin_sum_angles_l3629_362992

theorem sin_sum_angles (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1/4)
  (h2 : Real.cos α + Real.sin β = -8/5) : 
  Real.sin (α + β) = 249/800 := by
  sorry

end sin_sum_angles_l3629_362992


namespace parametric_to_ordinary_l3629_362987

theorem parametric_to_ordinary :
  ∀ θ : ℝ,
  let x : ℝ := 2 + Real.sin θ ^ 2
  let y : ℝ := -1 + Real.cos (2 * θ)
  2 * x + y - 4 = 0 ∧ x ∈ Set.Icc 2 3 := by
  sorry

end parametric_to_ordinary_l3629_362987


namespace ratio_equality_l3629_362952

theorem ratio_equality (a b : ℝ) (h1 : 4 * a^2 = 5 * b^3) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  a^2 / 5 = b^3 / 4 := by
sorry

end ratio_equality_l3629_362952


namespace division_problem_l3629_362943

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 997)
  (h2 : divisor = 23)
  (h3 : remainder = 8)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 43 := by
  sorry

end division_problem_l3629_362943


namespace inequality_proof_l3629_362971

theorem inequality_proof (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x^2 + y^2 + z^2 = 3) : 
  (x^2009 - 2008*(x-1))/(y+z) + (y^2009 - 2008*(y-1))/(x+z) + (z^2009 - 2008*(z-1))/(x+y) ≥ (1/2)*(x+y+z) := by
  sorry

end inequality_proof_l3629_362971


namespace repeating_decimal_sum_l3629_362985

/-- Represents a repeating decimal with a single digit repetend -/
def repeating_decimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with a two-digit repetend -/
def repeating_decimal_two_digits (n : ℕ) : ℚ := n / 99

theorem repeating_decimal_sum :
  repeating_decimal 6 + repeating_decimal_two_digits 12 - repeating_decimal 4 = 34 / 99 := by
  sorry

end repeating_decimal_sum_l3629_362985


namespace james_budget_theorem_l3629_362981

/-- James's budget and expenses --/
def budget : ℝ := 1000
def food_percentage : ℝ := 0.22
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.18
def transportation_percentage : ℝ := 0.12
def clothes_percentage : ℝ := 0.08
def miscellaneous_percentage : ℝ := 0.05

/-- Theorem: James's savings percentage and combined expenses --/
theorem james_budget_theorem :
  let food := budget * food_percentage
  let accommodation := budget * accommodation_percentage
  let entertainment := budget * entertainment_percentage
  let transportation := budget * transportation_percentage
  let clothes := budget * clothes_percentage
  let miscellaneous := budget * miscellaneous_percentage
  let total_spent := food + accommodation + entertainment + transportation + clothes + miscellaneous
  let savings := budget - total_spent
  let savings_percentage := (savings / budget) * 100
  let combined_expenses := entertainment + transportation + miscellaneous
  savings_percentage = 20 ∧ combined_expenses = 350 := by
  sorry

end james_budget_theorem_l3629_362981


namespace prob_last_is_one_l3629_362915

/-- Represents the set of possible digits Andrea can write. -/
def Digits : Finset ℕ := {1, 2, 3, 4}

/-- Represents whether a number is prime. -/
def isPrime (n : ℕ) : Prop := sorry

/-- Represents the process of writing digits until the sum of the last two is prime. -/
def StoppingProcess : Type := sorry

/-- The probability of the last digit being 1 given the first digit. -/
def probLastIsOne (first : ℕ) : ℚ := sorry

/-- The probability of the last digit being 1 for the entire process. -/
def totalProbLastIsOne : ℚ := sorry

/-- Theorem stating the probability of the last digit being 1 is 17/44. -/
theorem prob_last_is_one :
  totalProbLastIsOne = 17 / 44 := by sorry

end prob_last_is_one_l3629_362915


namespace sqrt_3x_minus_1_defined_l3629_362997

theorem sqrt_3x_minus_1_defined (x : ℝ) : Real.sqrt (3 * x - 1) ≥ 0 ↔ x ≥ 1/3 := by
  sorry

end sqrt_3x_minus_1_defined_l3629_362997


namespace last_digit_divisibility_l3629_362960

theorem last_digit_divisibility (n : ℕ) (h : n > 3) :
  let a := (2^n) % 10
  let b := 2^n - a
  6 ∣ (a * b) := by sorry

end last_digit_divisibility_l3629_362960


namespace table_height_proof_l3629_362928

/-- Given two configurations of stacked blocks on a table, prove that the table height is 34 inches -/
theorem table_height_proof (r s b : ℝ) (hr : r = 40) (hs : s = 34) (hb : b = 6) :
  ∃ (h l w : ℝ), h = 34 ∧ l + h - w = r ∧ w + h - l + b = s := by
  sorry

end table_height_proof_l3629_362928


namespace expand_product_l3629_362998

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_product_l3629_362998


namespace non_shaded_perimeter_l3629_362950

/-- Given a rectangle with dimensions 12 inches by 10 inches and an overlapping
    rectangle of 4 inches by 3 inches, if the shaded area is 130 square inches,
    then the perimeter of the non-shaded region is 7 1/3 inches. -/
theorem non_shaded_perimeter (shaded_area : ℝ) : 
  shaded_area = 130 → 
  (12 * 10 + 4 * 3 - shaded_area) / (12 - 4) * 2 + (12 - 4) * 2 = 22 / 3 :=
by sorry

end non_shaded_perimeter_l3629_362950


namespace east_to_north_ratio_l3629_362993

/-- Represents the number of tents in different areas of the campsite -/
structure Campsite where
  total : ℕ
  north : ℕ
  center : ℕ
  south : ℕ
  east : ℕ

/-- The conditions of the campsite as described in the problem -/
def campsite_conditions (c : Campsite) : Prop :=
  c.total = 900 ∧
  c.north = 100 ∧
  c.center = 4 * c.north ∧
  c.south = 200 ∧
  c.total = c.north + c.center + c.south + c.east

/-- The theorem stating the ratio of tents on the east side to the northernmost part -/
theorem east_to_north_ratio (c : Campsite) 
  (h : campsite_conditions c) : c.east = 2 * c.north :=
sorry

end east_to_north_ratio_l3629_362993


namespace number_equals_twenty_l3629_362922

theorem number_equals_twenty : ∃ x : ℝ, 5 * x = 100 ∧ x = 20 := by
  sorry

end number_equals_twenty_l3629_362922


namespace units_digit_problem_l3629_362903

theorem units_digit_problem : ∃ n : ℕ, n % 10 = 4 ∧ 8 * 14 * 1955 - 6^4 ≡ n [ZMOD 10] :=
by sorry

end units_digit_problem_l3629_362903


namespace average_weight_of_children_l3629_362968

def ages : List ℝ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

def regression_equation (x : ℝ) : ℝ := 2 * x + 7

theorem average_weight_of_children :
  let avg_age := (ages.sum) / (ages.length : ℝ)
  regression_equation avg_age = 15 := by sorry

end average_weight_of_children_l3629_362968


namespace mets_fans_count_l3629_362990

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The conditions of the problem -/
def fan_conditions (fc : FanCounts) : Prop :=
  -- Ratio of Yankees to Mets fans is 3:2
  3 * fc.mets = 2 * fc.yankees ∧
  -- Ratio of Mets to Red Sox fans is 4:5
  4 * fc.red_sox = 5 * fc.mets ∧
  -- Total number of fans is 330
  fc.yankees + fc.mets + fc.red_sox = 330

/-- The theorem to prove -/
theorem mets_fans_count (fc : FanCounts) : 
  fan_conditions fc → fc.mets = 88 := by
  sorry


end mets_fans_count_l3629_362990


namespace vector_parallel_condition_l3629_362964

/-- Given vectors a and b, if a + 3b is parallel to b, then the first component of a is 6. -/
theorem vector_parallel_condition (a b : ℝ × ℝ) (m : ℝ) :
  a = (m, 2) →
  b = (3, 1) →
  ∃ k : ℝ, a + 3 • b = k • b →
  m = 6 := by
  sorry

end vector_parallel_condition_l3629_362964


namespace max_weeks_correct_l3629_362930

/-- Represents a weekly ranking of 10 songs -/
def Ranking := Fin 10 → Fin 10

/-- The maximum number of weeks the same 10 songs can remain in the ranking -/
def max_weeks : ℕ := 46

/-- A function that represents the ranking change from one week to the next -/
def next_week (r : Ranking) : Ranking := sorry

/-- Predicate to check if a song's ranking has dropped -/
def has_dropped (r1 r2 : Ranking) (song : Fin 10) : Prop :=
  r2 song > r1 song

theorem max_weeks_correct (initial : Ranking) :
  ∀ (sequence : ℕ → Ranking),
    (∀ n, sequence (n + 1) = next_week (sequence n)) →
    (∀ n, sequence n ≠ sequence (n + 1)) →
    (∀ n m song, n < m → has_dropped (sequence n) (sequence m) song →
      ∀ k > m, has_dropped (sequence m) (sequence k) song ∨ sequence m song = sequence k song) →
    (∃ n ≤ max_weeks, ∃ song, sequence 0 song ≠ sequence n song) ∧
    (∀ n > max_weeks, ∃ song, sequence 0 song ≠ sequence n song) :=
  sorry


end max_weeks_correct_l3629_362930


namespace union_nonempty_iff_in_range_l3629_362942

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | x^2 + (2*a - 3)*x + 2*a^2 - a - 3 = 0}

-- Define the set A (inferred from the problem)
def A (a : ℝ) : Set ℝ := {x | x^2 - (a - 2)*x - 2*a + 4 = 0}

-- Define the range of a
def range_a : Set ℝ := {a | a ≤ -6 ∨ (-7/2 ≤ a ∧ a ≤ 3/2) ∨ a ≥ 2}

-- Theorem statement
theorem union_nonempty_iff_in_range (a : ℝ) :
  (A a ∪ B a).Nonempty ↔ a ∈ range_a :=
sorry

end union_nonempty_iff_in_range_l3629_362942


namespace intersection_P_Q_l3629_362978

def P : Set ℝ := {-3, 0, 2, 4}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by sorry

end intersection_P_Q_l3629_362978


namespace arithmetic_sequence_ratio_l3629_362972

/-- An arithmetic sequence defined by the given recurrence relation. -/
def ArithmeticSequence (x : ℕ → ℚ) : Prop :=
  ∀ n ≥ 3, x (n - 1) = (x n + x (n - 1) + x (n - 2)) / 3

/-- The main theorem stating the ratio of differences in the sequence. -/
theorem arithmetic_sequence_ratio 
  (x : ℕ → ℚ) 
  (h : ArithmeticSequence x) : 
  (x 300 - x 33) / (x 333 - x 3) = 89 / 110 := by
  sorry


end arithmetic_sequence_ratio_l3629_362972


namespace quadratic_root_zero_l3629_362986

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, (m - 3) * x^2 + x + m^2 - 9 = 0) ∧ 
  ((m - 3) * 0^2 + 0 + m^2 - 9 = 0) →
  m = -3 :=
sorry

end quadratic_root_zero_l3629_362986


namespace some_number_value_l3629_362933

theorem some_number_value (x : ℝ) : (50 + x / 90) * 90 = 4520 → x = 4470 := by
  sorry

end some_number_value_l3629_362933


namespace square_of_integer_l3629_362929

theorem square_of_integer (x y z : ℤ) (A : ℤ) 
  (h1 : A = x * y + y * z + z * x)
  (h2 : 4 * x + y + z = 0) : 
  ∃ (k : ℤ), (-1) * A = k^2 := by
  sorry

end square_of_integer_l3629_362929


namespace max_value_3a_plus_b_l3629_362975

theorem max_value_3a_plus_b (a b : ℝ) :
  (∀ x ∈ Set.Icc 1 2, |a * x^2 + b * x + a| ≤ x) →
  (∃ a₀ b₀ : ℝ, (∀ x ∈ Set.Icc 1 2, |a₀ * x^2 + b₀ * x + a₀| ≤ x) ∧ 3 * a₀ + b₀ = 3) ∧
  (∀ a₁ b₁ : ℝ, (∀ x ∈ Set.Icc 1 2, |a₁ * x^2 + b₁ * x + a₁| ≤ x) → 3 * a₁ + b₁ ≤ 3) :=
by sorry

#check max_value_3a_plus_b

end max_value_3a_plus_b_l3629_362975


namespace unique_value_2n_plus_m_l3629_362982

theorem unique_value_2n_plus_m :
  ∀ n m : ℤ,
  (3 * n - m < 5) →
  (n + m > 26) →
  (3 * m - 2 * n < 46) →
  (2 * n + m = 36) :=
by sorry

end unique_value_2n_plus_m_l3629_362982


namespace x_in_interval_l3629_362970

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define x as in the problem
noncomputable def x : ℝ := 1 / log (1/2) (1/3) + 1 / log (1/5) (1/3)

-- State the theorem
theorem x_in_interval : 2 < x ∧ x < 3 := by sorry

end x_in_interval_l3629_362970
