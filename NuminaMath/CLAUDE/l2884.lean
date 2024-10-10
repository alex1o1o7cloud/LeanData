import Mathlib

namespace grade_distribution_l2884_288477

theorem grade_distribution (n₂ n₃ n₄ n₅ : ℕ) : 
  n₂ + n₃ + n₄ + n₅ = 25 →
  n₄ = n₃ + 4 →
  2 * n₂ + 3 * n₃ + 4 * n₄ + 5 * n₅ = 121 →
  n₂ = 0 := by
sorry

end grade_distribution_l2884_288477


namespace multiply_33333_33334_l2884_288463

theorem multiply_33333_33334 : 33333 * 33334 = 1111122222 := by
  sorry

end multiply_33333_33334_l2884_288463


namespace sum_of_roots_eq_neg_one_l2884_288443

theorem sum_of_roots_eq_neg_one (m n : ℝ) : 
  m ≠ 0 → 
  n ≠ 0 → 
  (∀ x : ℝ, x ≠ 0 → 1 / x^2 + m / x + n = 0) → 
  m * n = 1 → 
  m + n = -1 := by
sorry

end sum_of_roots_eq_neg_one_l2884_288443


namespace trig_inequality_l2884_288430

theorem trig_inequality (α β γ : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : 0 < γ ∧ γ < Real.pi / 2)
  (h4 : Real.sin α ^ 3 + Real.sin β ^ 3 + Real.sin γ ^ 3 = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 * Real.sqrt 3 / 2 := by
sorry

end trig_inequality_l2884_288430


namespace guests_who_stayed_l2884_288461

-- Define the initial conditions
def total_guests : ℕ := 50
def men_guests : ℕ := 15

-- Define the number of guests who left
def men_left : ℕ := men_guests / 5
def children_left : ℕ := 4

-- Theorem to prove
theorem guests_who_stayed :
  let women_guests := total_guests / 2
  let children_guests := total_guests - women_guests - men_guests
  let guests_who_stayed := total_guests - men_left - children_left
  guests_who_stayed = 43 := by sorry

end guests_who_stayed_l2884_288461


namespace x_squared_in_set_l2884_288411

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({1, 0, x} : Set ℝ) → x = -1 := by
  sorry

end x_squared_in_set_l2884_288411


namespace dormitory_expenditure_l2884_288499

theorem dormitory_expenditure 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (cost_decrease : ℕ) 
  (expenditure_increase : ℕ) 
  (h1 : initial_students = 250)
  (h2 : new_students = 75)
  (h3 : cost_decrease = 20)
  (h4 : expenditure_increase = 10000) :
  (initial_students + new_students) * 
  ((initial_students + new_students) * expenditure_increase / initial_students - cost_decrease) = 65000 := by
  sorry

end dormitory_expenditure_l2884_288499


namespace horizontal_shift_l2884_288470

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the shift amount
variable (a : ℝ)

-- Define a point (x, y) on the original graph
variable (x y : ℝ)

-- Theorem statement
theorem horizontal_shift (h : y = f x) :
  y = f (x - a) ↔ y = f ((x + a) - a) :=
sorry

end horizontal_shift_l2884_288470


namespace second_class_size_l2884_288420

theorem second_class_size (n : ℕ) (avg_all : ℚ) : 
  n > 0 ∧ 
  (30 : ℚ) * 40 + n * 60 = (30 + n) * avg_all ∧ 
  avg_all = (105 : ℚ) / 2 → 
  n = 50 := by
sorry

end second_class_size_l2884_288420


namespace investment_final_value_l2884_288457

def investment_value (initial : ℝ) (w1 w2 w3 w4 w5 w6 : ℝ) : ℝ :=
  initial * (1 + w1) * (1 + w2) * (1 - w3) * (1 + w4) * (1 + w5) * (1 - w6)

theorem investment_final_value :
  let initial : ℝ := 400
  let week1_gain : ℝ := 0.25
  let week2_gain : ℝ := 0.50
  let week3_loss : ℝ := 0.10
  let week4_gain : ℝ := 0.20
  let week5_gain : ℝ := 0.05
  let week6_loss : ℝ := 0.15
  investment_value initial week1_gain week2_gain week3_loss week4_gain week5_gain week6_loss = 722.925 := by
  sorry

end investment_final_value_l2884_288457


namespace arithmetic_sequence_sum_condition_l2884_288476

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n * (a 1 + a n)) / 2

/-- The theorem stating that m = 5 for the given conditions -/
theorem arithmetic_sequence_sum_condition (seq : ArithmeticSequence) (m : ℕ) :
  m > 1 →
  seq.S (m - 1) = -2 →
  seq.S m = 0 →
  seq.S (m + 1) = 3 →
  m = 5 := by
  sorry

end arithmetic_sequence_sum_condition_l2884_288476


namespace largest_even_n_inequality_l2884_288483

theorem largest_even_n_inequality (n : ℕ) : 
  (n = 8 ∧ n % 2 = 0) ↔ 
  (∀ x : ℝ, (Real.sin x)^(2*n) + (Real.cos x)^(2*n) + (Real.tan x)^2 ≥ 1/n) ∧
  (∀ m : ℕ, m > n → m % 2 = 0 → 
    ∃ x : ℝ, (Real.sin x)^(2*m) + (Real.cos x)^(2*m) + (Real.tan x)^2 < 1/m) :=
by sorry

end largest_even_n_inequality_l2884_288483


namespace watch_loss_percentage_l2884_288448

/-- Calculates the loss percentage for a watch sale given specific conditions. -/
def loss_percentage (cost_price selling_price increased_price : ℚ) : ℚ :=
  let gain_percentage : ℚ := 2 / 100
  let price_difference : ℚ := increased_price - selling_price
  let loss : ℚ := cost_price - selling_price
  (loss / cost_price) * 100

/-- Theorem stating that the loss percentage is 10% under given conditions. -/
theorem watch_loss_percentage : 
  let cost_price : ℚ := 1166.67
  let selling_price : ℚ := cost_price - 116.67
  let increased_price : ℚ := selling_price + 140
  loss_percentage cost_price selling_price increased_price = 10 := by
  sorry

end watch_loss_percentage_l2884_288448


namespace days_to_pay_for_cash_register_l2884_288449

/-- Represents the daily sales and costs for Marie's bakery --/
structure BakeryFinances where
  breadPrice : ℝ
  breadQuantity : ℝ
  bagelPrice : ℝ
  bagelQuantity : ℝ
  cakePrice : ℝ
  cakeQuantity : ℝ
  muffinPrice : ℝ
  muffinQuantity : ℝ
  rent : ℝ
  electricity : ℝ
  wages : ℝ
  ingredientCosts : ℝ
  salesTax : ℝ

/-- Calculates the number of days needed to pay for the cash register --/
def daysToPayForCashRegister (finances : BakeryFinances) (cashRegisterCost : ℝ) : ℕ :=
  sorry

/-- Theorem stating that it takes 17 days to pay for the cash register --/
theorem days_to_pay_for_cash_register :
  ∃ (finances : BakeryFinances),
    finances.breadPrice = 2 ∧
    finances.breadQuantity = 40 ∧
    finances.bagelPrice = 1.5 ∧
    finances.bagelQuantity = 20 ∧
    finances.cakePrice = 12 ∧
    finances.cakeQuantity = 6 ∧
    finances.muffinPrice = 3 ∧
    finances.muffinQuantity = 10 ∧
    finances.rent = 20 ∧
    finances.electricity = 2 ∧
    finances.wages = 80 ∧
    finances.ingredientCosts = 30 ∧
    finances.salesTax = 0.08 ∧
    daysToPayForCashRegister finances 1040 = 17 :=
  sorry

end days_to_pay_for_cash_register_l2884_288449


namespace reading_rate_difference_l2884_288490

-- Define the given information
def songhee_pages : ℕ := 288
def songhee_days : ℕ := 12
def eunju_pages : ℕ := 243
def eunju_days : ℕ := 9

-- Define the daily reading rates
def songhee_rate : ℚ := songhee_pages / songhee_days
def eunju_rate : ℚ := eunju_pages / eunju_days

-- Theorem statement
theorem reading_rate_difference : eunju_rate - songhee_rate = 3 := by
  sorry

end reading_rate_difference_l2884_288490


namespace stating_solutions_eq_partitions_l2884_288402

/-- The number of solutions to the equation in positive integers -/
def numSolutions : ℕ := sorry

/-- The number of partitions of 7 -/
def numPartitions7 : ℕ := sorry

/-- 
Theorem stating that the number of solutions to the equation
a₁(b₁) + a₂(b₁+b₂) + ... + aₖ(b₁+b₂+...+bₖ) = 7
in positive integers (k; a₁, a₂, ..., aₖ; b₁, b₂, ..., bₖ)
is equal to the number of partitions of 7
-/
theorem solutions_eq_partitions : numSolutions = numPartitions7 := by sorry

end stating_solutions_eq_partitions_l2884_288402


namespace paper_stack_thickness_sheets_in_six_cm_stack_l2884_288467

/-- Calculates the number of sheets in a stack of paper given the thickness of the stack and the number of sheets per unit thickness. -/
def sheets_in_stack (stack_thickness : ℝ) (sheets_per_unit : ℝ) : ℝ :=
  stack_thickness * sheets_per_unit

theorem paper_stack_thickness (bundle_sheets : ℝ) (bundle_thickness : ℝ) (stack_thickness : ℝ) :
  bundle_sheets > 0 → bundle_thickness > 0 → stack_thickness > 0 →
  sheets_in_stack stack_thickness (bundle_sheets / bundle_thickness) = 
    (stack_thickness / bundle_thickness) * bundle_sheets := by
  sorry

/-- The main theorem that proves the number of sheets in a 6 cm stack given a 400-sheet bundle is 4 cm thick. -/
theorem sheets_in_six_cm_stack : 
  sheets_in_stack 6 (400 / 4) = 600 := by
  sorry

end paper_stack_thickness_sheets_in_six_cm_stack_l2884_288467


namespace bowl_glass_pairings_l2884_288405

theorem bowl_glass_pairings :
  let num_bowls : ℕ := 5
  let num_glasses : ℕ := 4
  num_bowls * num_glasses = 20 :=
by sorry

end bowl_glass_pairings_l2884_288405


namespace computer_price_decrease_l2884_288484

/-- The price of a computer after a certain number of years, given an initial price and a constant rate of decrease every two years. -/
def price_after_years (initial_price : ℝ) (decrease_rate : ℝ) (years : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ (years / 2)

/-- Theorem stating that a computer with an initial price of 8100 yuan, decreasing by one-third every two years, will cost 2400 yuan after 6 years. -/
theorem computer_price_decrease (initial_price : ℝ) (years : ℕ) :
  initial_price = 8100 →
  years = 6 →
  price_after_years initial_price (1/3) years = 2400 := by
  sorry

end computer_price_decrease_l2884_288484


namespace fraction_equivalence_l2884_288491

theorem fraction_equivalence : ∃ x : ℚ, (4 + x) / (7 + x) = 3 / 4 := by
  use 5
  sorry

end fraction_equivalence_l2884_288491


namespace two_scoop_sundaes_l2884_288444

theorem two_scoop_sundaes (n : ℕ) (h : n = 8) : Nat.choose n 2 = 28 := by
  sorry

end two_scoop_sundaes_l2884_288444


namespace mixed_fraction_product_l2884_288488

theorem mixed_fraction_product (X Y : ℕ) : 
  (5 + 1 / X) * (Y + 1 / 2) = 43 → X = 17 ∧ Y = 8 :=
sorry

end mixed_fraction_product_l2884_288488


namespace area_equals_scientific_notation_l2884_288417

-- Define the area of the radio telescope
def telescope_area : ℝ := 250000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.5 * (10 ^ 5)

-- Theorem stating that the area is equal to its scientific notation representation
theorem area_equals_scientific_notation : telescope_area = scientific_notation := by
  sorry

end area_equals_scientific_notation_l2884_288417


namespace power_inequality_l2884_288472

theorem power_inequality (a b m n : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) (hn : n > 0) :
  a^(m+n) + b^(m+n) ≥ a^m * b^n + a^n * b^m := by sorry

end power_inequality_l2884_288472


namespace frog_eyes_in_pond_l2884_288424

/-- The number of eyes a frog has -/
def eyes_per_frog : ℕ := 2

/-- The number of frogs in the pond -/
def frogs_in_pond : ℕ := 4

/-- The total number of frog eyes in the pond -/
def total_frog_eyes : ℕ := frogs_in_pond * eyes_per_frog

theorem frog_eyes_in_pond : total_frog_eyes = 8 := by
  sorry

end frog_eyes_in_pond_l2884_288424


namespace jerrys_age_l2884_288432

/-- Given that Mickey's age is 20 years old and 10 years more than 200% of Jerry's age, prove that Jerry is 5 years old. -/
theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 20 → 
  mickey_age = 2 * jerry_age + 10 → 
  jerry_age = 5 := by
  sorry

end jerrys_age_l2884_288432


namespace bananas_per_friend_l2884_288456

def virginia_bananas : ℕ := 40
def virginia_marbles : ℕ := 4
def number_of_friends : ℕ := 40

theorem bananas_per_friend :
  virginia_bananas / number_of_friends = 1 :=
sorry

end bananas_per_friend_l2884_288456


namespace solve_brownies_problem_l2884_288458

def brownies_problem (total : ℕ) (to_admin : ℕ) (to_simon : ℕ) (left : ℕ) : Prop :=
  let remaining_after_admin := total - to_admin
  let to_carl := remaining_after_admin - to_simon - left
  (to_carl : ℚ) / remaining_after_admin = 1 / 2

theorem solve_brownies_problem :
  brownies_problem 20 10 2 3 := by
  sorry

end solve_brownies_problem_l2884_288458


namespace cylinder_height_from_balls_l2884_288404

/-- The height of a cylinder formed by melting steel balls -/
theorem cylinder_height_from_balls (num_balls : ℕ) (ball_radius cylinder_radius : ℝ) :
  num_balls = 12 →
  ball_radius = 2 →
  cylinder_radius = 3 →
  (4 / 3 * π * num_balls * ball_radius ^ 3) / (π * cylinder_radius ^ 2) = 128 / 9 := by
  sorry

end cylinder_height_from_balls_l2884_288404


namespace square_sum_of_powers_l2884_288473

theorem square_sum_of_powers (a b : ℕ+) : 
  (∃ n : ℕ, 2^(a : ℕ) + 3^(b : ℕ) = n^2) ↔ a = 4 ∧ b = 2 := by
sorry

end square_sum_of_powers_l2884_288473


namespace ten_points_chords_l2884_288429

/-- The number of chords that can be drawn from n points on a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: There are 45 different chords that can be drawn from 10 points on a circle -/
theorem ten_points_chords : num_chords 10 = 45 := by
  sorry

end ten_points_chords_l2884_288429


namespace right_triangle_k_values_l2884_288418

/-- A right triangle ABC with vectors AB and AC -/
structure RightTriangle where
  AB : ℝ × ℝ
  AC : ℝ × ℝ
  is_right : Bool

/-- The possible k values for a right triangle with AB = (2, 3) and AC = (1, k) -/
def possible_k_values : Set ℝ :=
  {-2/3, 11/3, (3 + Real.sqrt 13)/2, (3 - Real.sqrt 13)/2}

/-- Theorem stating that k must be one of the possible values -/
theorem right_triangle_k_values (triangle : RightTriangle) 
  (h1 : triangle.AB = (2, 3)) 
  (h2 : triangle.AC = (1, triangle.AC.snd)) 
  (h3 : triangle.is_right = true) : 
  triangle.AC.snd ∈ possible_k_values := by
  sorry

end right_triangle_k_values_l2884_288418


namespace water_in_mixture_l2884_288400

theorem water_in_mixture (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (b * x) / (a + b) = x * (b / (a + b)) := by
  sorry

end water_in_mixture_l2884_288400


namespace coloring_existence_and_impossibility_l2884_288401

def is_monochromatic (color : ℕ → Bool) (x y z : ℕ) : Prop :=
  color x = color y ∧ color y = color z

theorem coloring_existence_and_impossibility :
  (∃ (color : ℕ → Bool),
    ∀ x y z, 1 ≤ x ∧ x ≤ 2017 ∧ 1 ≤ y ∧ y ≤ 2017 ∧ 1 ≤ z ∧ z ≤ 2017 →
      8 * (x + y) = z → ¬is_monochromatic color x y z) ∧
  (∀ n : ℕ, n ≥ 2056 →
    ¬∃ (color : ℕ → Bool),
      ∀ x y z, 1 ≤ x ∧ x ≤ n ∧ 1 ≤ y ∧ y ≤ n ∧ 1 ≤ z ∧ z ≤ n →
        8 * (x + y) = z → ¬is_monochromatic color x y z) :=
by sorry

end coloring_existence_and_impossibility_l2884_288401


namespace hospital_staff_count_l2884_288415

theorem hospital_staff_count (doctors nurses : ℕ) (h1 : doctors * 9 = nurses * 5) (h2 : nurses = 180) :
  doctors + nurses = 280 := by
  sorry

end hospital_staff_count_l2884_288415


namespace only_fatigued_drivers_accidents_correlative_l2884_288453

/-- Represents a pair of quantities -/
inductive QuantityPair
  | StudentGradesWeight
  | TimeDisplacement
  | WaterVolumeWeight
  | FatiguedDriversAccidents

/-- Describes the relationship between two quantities -/
inductive Relationship
  | Correlative
  | Functional
  | Independent

/-- Function that determines the relationship for a given pair of quantities -/
def determineRelationship (pair : QuantityPair) : Relationship :=
  match pair with
  | QuantityPair.StudentGradesWeight => Relationship.Independent
  | QuantityPair.TimeDisplacement => Relationship.Functional
  | QuantityPair.WaterVolumeWeight => Relationship.Functional
  | QuantityPair.FatiguedDriversAccidents => Relationship.Correlative

/-- Theorem stating that only the FatiguedDriversAccidents pair has a correlative relationship -/
theorem only_fatigued_drivers_accidents_correlative :
  ∀ (pair : QuantityPair),
    determineRelationship pair = Relationship.Correlative ↔ pair = QuantityPair.FatiguedDriversAccidents :=
by
  sorry


end only_fatigued_drivers_accidents_correlative_l2884_288453


namespace fifth_rectangle_is_square_l2884_288451

-- Define the structure of a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the structure of a square
structure Square where
  side : ℝ

-- Define the division of a square into rectangles
def squareDivision (s : Square) (r1 r2 r3 r4 r5 : Rectangle) : Prop :=
  -- The sum of widths and heights of corner rectangles equals the square's side
  r1.width + r2.width = s.side ∧
  r1.height + r3.height = s.side ∧
  -- The areas of the four corner rectangles are equal
  r1.width * r1.height = r2.width * r2.height ∧
  r2.width * r2.height = r3.width * r3.height ∧
  r3.width * r3.height = r4.width * r4.height ∧
  -- The fifth rectangle doesn't touch the sides of the square
  r5.width < s.side - r1.width ∧
  r5.height < s.side - r1.height

-- Theorem statement
theorem fifth_rectangle_is_square 
  (s : Square) (r1 r2 r3 r4 r5 : Rectangle) 
  (h : squareDivision s r1 r2 r3 r4 r5) : 
  r5.width = r5.height :=
sorry

end fifth_rectangle_is_square_l2884_288451


namespace expression_value_l2884_288496

theorem expression_value (x y : ℝ) (h : |x + 1| + (2 * y - 4)^2 = 0) :
  (2 * x^2 * y - 3 * x * y) - 2 * (x^2 * y - x * y + 1/2 * x * y^2) + x * y = 4 :=
by sorry

end expression_value_l2884_288496


namespace root_conditions_imply_relation_l2884_288492

/-- Given two equations with specific root conditions, prove a relation between constants c and d -/
theorem root_conditions_imply_relation (c d : ℝ) : 
  (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    ((r₁ + c) * (r₁ + d) * (r₁ - 7)) / ((r₁ + 4)^2) = 0 ∧
    ((r₂ + c) * (r₂ + d) * (r₂ - 7)) / ((r₂ + 4)^2) = 0 ∧
    ((r₃ + c) * (r₃ + d) * (r₃ - 7)) / ((r₃ + 4)^2) = 0) →
  (∃! (r : ℝ), ((r + 2*c) * (r + 5) * (r + 8)) / ((r + d) * (r - 7)) = 0) →
  100 * c + d = 408 := by
sorry

end root_conditions_imply_relation_l2884_288492


namespace reciprocal_square_sum_l2884_288494

theorem reciprocal_square_sum : (((1 : ℚ) / 4 + 1 / 6) ^ 2)⁻¹ = 144 / 25 := by
  sorry

end reciprocal_square_sum_l2884_288494


namespace polynomial_not_factorizable_l2884_288439

theorem polynomial_not_factorizable : 
  ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ), 
    x^4 + 3*x^3 + 6*x^2 + 9*x + 12 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end polynomial_not_factorizable_l2884_288439


namespace max_product_given_sum_l2884_288474

theorem max_product_given_sum (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 2 → ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a * b ≥ x * y :=
sorry

end max_product_given_sum_l2884_288474


namespace product_first_two_terms_l2884_288468

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ
  /-- The seventh term of the sequence is 20 -/
  seventh_term : a₁ + 6 * d = 20
  /-- The common difference is 2 -/
  common_diff : d = 2

/-- The product of the first two terms of the arithmetic sequence is 80 -/
theorem product_first_two_terms (seq : ArithmeticSequence) :
  seq.a₁ * (seq.a₁ + seq.d) = 80 := by
  sorry


end product_first_two_terms_l2884_288468


namespace inverse_sum_theorem_l2884_288481

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem inverse_sum_theorem : 
  ∃ (a b : ℝ), f a = 9 ∧ f b = -64 ∧ a + b = -5 := by sorry

end inverse_sum_theorem_l2884_288481


namespace f_decreasing_implies_a_range_l2884_288413

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * (a - 1) * x + 2

-- Define the property of f being decreasing on (-∞, 4]
def isDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 4 → f x > f y

-- State the theorem
theorem f_decreasing_implies_a_range :
  ∀ a : ℝ, isDecreasingOn (f a) ↔ 0 ≤ a ∧ a ≤ 1/5 := by sorry

end f_decreasing_implies_a_range_l2884_288413


namespace triangle_ABC_properties_l2884_288464

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- Define the line parallel to BC passing through A
def line_parallel_BC (x y : ℝ) : Prop :=
  x + 2*y - 8 = 0

-- Define the lines passing through B equidistant from A and C
def line_equidistant_1 (x y : ℝ) : Prop :=
  3*x - 2*y - 4 = 0

def line_equidistant_2 (x y : ℝ) : Prop :=
  3*x + 2*y - 44 = 0

theorem triangle_ABC_properties :
  (∀ x y, line_parallel_BC x y ↔ (x - A.1) * (C.2 - B.2) = (y - A.2) * (C.1 - B.1)) ∧
  (∀ x y, (line_equidistant_1 x y ∨ line_equidistant_2 x y) ↔
    ((x - A.1)^2 + (y - A.2)^2 = (x - C.1)^2 + (y - C.2)^2 ∧ x = B.1 ∧ y = B.2)) :=
by sorry

end triangle_ABC_properties_l2884_288464


namespace junior_score_l2884_288414

theorem junior_score (n : ℝ) (junior_ratio : ℝ) (senior_ratio : ℝ) 
  (class_avg : ℝ) (senior_avg : ℝ) (h1 : junior_ratio = 0.2) 
  (h2 : senior_ratio = 0.8) (h3 : junior_ratio + senior_ratio = 1) 
  (h4 : class_avg = 84) (h5 : senior_avg = 82) : 
  (class_avg * n - senior_avg * senior_ratio * n) / (junior_ratio * n) = 92 := by
sorry

end junior_score_l2884_288414


namespace team_average_weight_l2884_288407

theorem team_average_weight 
  (num_forwards : ℕ) 
  (num_defensemen : ℕ) 
  (avg_weight_forwards : ℝ) 
  (avg_weight_defensemen : ℝ) 
  (h1 : num_forwards = 8)
  (h2 : num_defensemen = 12)
  (h3 : avg_weight_forwards = 75)
  (h4 : avg_weight_defensemen = 82) :
  let total_players := num_forwards + num_defensemen
  let total_weight := num_forwards * avg_weight_forwards + num_defensemen * avg_weight_defensemen
  total_weight / total_players = 79.2 := by
  sorry

end team_average_weight_l2884_288407


namespace fraction_simplification_l2884_288485

theorem fraction_simplification (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (1 / y) / (1 / x) = 3 / 4 := by
  sorry

end fraction_simplification_l2884_288485


namespace at_most_one_super_plus_good_l2884_288416

/-- Represents an 8x8 chessboard with numbers 1 to 64 --/
def Chessboard := Fin 8 → Fin 8 → Fin 64

/-- A number is super-plus-good if it's the largest in its row and smallest in its column --/
def is_super_plus_good (board : Chessboard) (row col : Fin 8) : Prop :=
  (∀ c : Fin 8, board row c ≤ board row col) ∧
  (∀ r : Fin 8, board row col ≤ board r col)

/-- The arrangement is valid if each number appears exactly once --/
def is_valid_arrangement (board : Chessboard) : Prop :=
  ∀ n : Fin 64, ∃! (row col : Fin 8), board row col = n

theorem at_most_one_super_plus_good (board : Chessboard) 
  (h : is_valid_arrangement board) :
  ∃! (row col : Fin 8), is_super_plus_good board row col :=
sorry

end at_most_one_super_plus_good_l2884_288416


namespace gildas_marbles_theorem_l2884_288412

/-- The percentage of marbles Gilda has left after giving away to her friends and family -/
def gildas_remaining_marbles : ℝ :=
  let after_pedro := 1 - 0.30
  let after_ebony := after_pedro * (1 - 0.20)
  let after_jimmy := after_ebony * (1 - 0.15)
  let after_clara := after_jimmy * (1 - 0.10)
  after_clara * 100

/-- Theorem stating that Gilda has 42.84% of her original marbles left -/
theorem gildas_marbles_theorem : 
  ∃ ε > 0, |gildas_remaining_marbles - 42.84| < ε :=
sorry

end gildas_marbles_theorem_l2884_288412


namespace rectangle_width_l2884_288422

theorem rectangle_width (w : ℝ) (h1 : w > 0) : 
  (2 * w * w = 3 * 2 * (2 * w + w)) → w = 9 := by
  sorry

end rectangle_width_l2884_288422


namespace group_size_calculation_l2884_288438

theorem group_size_calculation (n : ℕ) : 
  (n * n = 5929) → n = 77 := by
  sorry

end group_size_calculation_l2884_288438


namespace granger_bread_loaves_l2884_288421

/-- Represents the grocery items and their prices --/
structure GroceryItems where
  spam_price : ℕ
  peanut_butter_price : ℕ
  bread_price : ℕ

/-- Represents the quantities of items bought --/
structure Quantities where
  spam_cans : ℕ
  peanut_butter_jars : ℕ

/-- Calculates the number of bread loaves bought given the total amount paid --/
def bread_loaves_bought (items : GroceryItems) (quantities : Quantities) (total_paid : ℕ) : ℕ :=
  (total_paid - (items.spam_price * quantities.spam_cans + items.peanut_butter_price * quantities.peanut_butter_jars)) / items.bread_price

/-- Theorem stating that Granger bought 4 loaves of bread --/
theorem granger_bread_loaves :
  let items := GroceryItems.mk 3 5 2
  let quantities := Quantities.mk 12 3
  let total_paid := 59
  bread_loaves_bought items quantities total_paid = 4 := by
  sorry


end granger_bread_loaves_l2884_288421


namespace ten_people_round_table_arrangements_l2884_288425

/-- The number of unique seating arrangements for n people around a round table,
    considering rotations as identical. -/
def uniqueRoundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem stating that the number of unique seating arrangements for 10 people
    around a round table, considering rotations as identical, is 362,880. -/
theorem ten_people_round_table_arrangements :
  uniqueRoundTableArrangements 10 = 362880 := by sorry

end ten_people_round_table_arrangements_l2884_288425


namespace square_side_length_l2884_288489

theorem square_side_length (perimeter : ℝ) (h : perimeter = 28) : 
  perimeter / 4 = 7 := by
  sorry

end square_side_length_l2884_288489


namespace y_intercept_of_parallel_line_l2884_288469

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if two lines are parallel. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Returns true if a point lies on a line. -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The given line y = -3x + 6 -/
def givenLine : Line :=
  { slope := -3, yIntercept := 6 }

theorem y_intercept_of_parallel_line :
  ∀ (b : Line),
    parallel b givenLine →
    pointOnLine b 3 (-4) →
    b.yIntercept = 5 := by
  sorry

end y_intercept_of_parallel_line_l2884_288469


namespace M_intersect_N_equals_target_l2884_288498

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (x - 1) < 0}
def N : Set ℝ := {x | 2 * x^2 - 3 * x ≤ 0}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the open-closed interval (1, 3/2]
def target_set : Set ℝ := Set.Ioc 1 (3/2)

-- Theorem statement
theorem M_intersect_N_equals_target : M_intersect_N = target_set := by
  sorry

end M_intersect_N_equals_target_l2884_288498


namespace lcm_problem_l2884_288427

theorem lcm_problem (a b c : ℕ+) (ha : a = 24) (hb : b = 36) (hlcm : Nat.lcm (Nat.lcm a b) c = 360) : c = 5 := by
  sorry

end lcm_problem_l2884_288427


namespace line_through_three_points_l2884_288446

/-- A line passes through three points: (2, 5), (-3, m), and (15, -1).
    This theorem proves that the value of m is 95/13. -/
theorem line_through_three_points (m : ℚ) : 
  (∃ (line : ℝ → ℝ), 
    line 2 = 5 ∧ 
    line (-3) = m ∧ 
    line 15 = -1) → 
  m = 95 / 13 :=
by sorry

end line_through_three_points_l2884_288446


namespace ceiling_sum_evaluation_l2884_288487

theorem ceiling_sum_evaluation : 
  ⌈Real.sqrt (16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)^2⌉ + ⌈(16/9 : ℝ)^(1/3)⌉ = 10 := by
  sorry

end ceiling_sum_evaluation_l2884_288487


namespace equal_distribution_of_stickers_l2884_288452

/-- The number of stickers Haley has -/
def total_stickers : ℕ := 72

/-- The number of Haley's friends -/
def num_friends : ℕ := 9

/-- The number of stickers each friend will receive -/
def stickers_per_friend : ℕ := total_stickers / num_friends

theorem equal_distribution_of_stickers :
  stickers_per_friend * num_friends = total_stickers :=
by sorry

end equal_distribution_of_stickers_l2884_288452


namespace proportional_segments_l2884_288445

theorem proportional_segments (a b c d : ℝ) :
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a / b = c / d) →
  a = 4 →
  b = 2 →
  c = 3 →
  d = 3 / 2 := by
sorry

end proportional_segments_l2884_288445


namespace line_intersects_cubic_at_two_points_l2884_288434

/-- The function representing the cubic curve y = x^3 -/
def cubic_curve (x : ℝ) : ℝ := x^3

/-- The function representing the line y = ax + 16 -/
def line (a x : ℝ) : ℝ := a * x + 16

/-- Predicate to check if the line intersects the curve at exactly two distinct points -/
def intersects_at_two_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    cubic_curve x₁ = line a x₁ ∧
    cubic_curve x₂ = line a x₂ ∧
    ∀ x : ℝ, cubic_curve x = line a x → x = x₁ ∨ x = x₂

theorem line_intersects_cubic_at_two_points (a : ℝ) :
  intersects_at_two_points a → a = 12 := by sorry

end line_intersects_cubic_at_two_points_l2884_288434


namespace calculate_second_solution_percentage_l2884_288419

/-- Given two solutions mixed to form a final solution, calculates the percentage of the second solution. -/
theorem calculate_second_solution_percentage
  (final_volume : ℝ)
  (final_percentage : ℝ)
  (first_volume : ℝ)
  (first_percentage : ℝ)
  (second_volume : ℝ)
  (h_final_volume : final_volume = 40)
  (h_final_percentage : final_percentage = 0.45)
  (h_first_volume : first_volume = 28)
  (h_first_percentage : first_percentage = 0.30)
  (h_second_volume : second_volume = 12)
  (h_volume_sum : first_volume + second_volume = final_volume)
  (h_substance_balance : first_volume * first_percentage + second_volume * (second_percentage / 100) = final_volume * final_percentage) :
  second_percentage = 80 := by
  sorry

#check calculate_second_solution_percentage

end calculate_second_solution_percentage_l2884_288419


namespace copy_pages_theorem_l2884_288454

/-- Given a cost per page in cents and a budget in dollars, 
    calculate the number of pages that can be copied. -/
def pages_copied (cost_per_page : ℕ) (budget : ℕ) : ℕ :=
  (budget * 100) / cost_per_page

/-- Theorem: With a cost of 3 cents per page and a budget of $15, 
    500 pages can be copied. -/
theorem copy_pages_theorem : pages_copied 3 15 = 500 := by
  sorry

end copy_pages_theorem_l2884_288454


namespace smallest_pair_sum_divisible_by_125_l2884_288406

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by 125 -/
def divisible_by_125 (n : ℕ) : Prop := n % 125 = 0

/-- The smallest pair of consecutive numbers with sum of digits divisible by 125 -/
def smallest_pair : ℕ × ℕ := (89999999999998, 89999999999999)

theorem smallest_pair_sum_divisible_by_125 :
  let (a, b) := smallest_pair
  divisible_by_125 (sum_of_digits a) ∧
  divisible_by_125 (sum_of_digits b) ∧
  b = a + 1 ∧
  ∀ (x y : ℕ), x < a → y = x + 1 →
    ¬(divisible_by_125 (sum_of_digits x) ∧ divisible_by_125 (sum_of_digits y)) :=
by sorry

end smallest_pair_sum_divisible_by_125_l2884_288406


namespace evenBlueFaceCubesFor6x3x2_l2884_288466

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of cubes with an even number of blue faces -/
def evenBlueFaceCubes (b : Block) : ℕ :=
  let edgeCubes := 4 * (b.length + b.width + b.height - 6)
  let internalCubes := (b.length - 2) * (b.width - 2) * (b.height - 2)
  edgeCubes + internalCubes

/-- The main theorem stating that a 6x3x2 block has 20 cubes with an even number of blue faces -/
theorem evenBlueFaceCubesFor6x3x2 : 
  evenBlueFaceCubes { length := 6, width := 3, height := 2 } = 20 := by
  sorry

end evenBlueFaceCubesFor6x3x2_l2884_288466


namespace integral_sqrt_minus_2x_l2884_288433

theorem integral_sqrt_minus_2x (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f x = Real.sqrt (1 - (x - 1)^2)) →
  (∀ x, g x = 2 * x) →
  ∫ x in (0 : ℝ)..1, (f x - g x) = π / 4 - 1 := by
  sorry

end integral_sqrt_minus_2x_l2884_288433


namespace min_value_quadratic_ratio_l2884_288479

/-- A quadratic function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The derivative of a quadratic function -/
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem min_value_quadratic_ratio 
  (a b c : ℝ) 
  (h1 : quadratic_derivative a b 0 > 0)
  (h2 : ∀ x, quadratic a b c x ≥ 0) :
  (quadratic a b c 1) / (quadratic_derivative a b 0) ≥ 1 := by
  sorry

end min_value_quadratic_ratio_l2884_288479


namespace final_black_fraction_is_512_729_l2884_288437

/-- Represents the fraction of black area remaining after one change -/
def remaining_black_fraction : ℚ := 8 / 9

/-- Represents the number of changes applied to the triangle -/
def num_changes : ℕ := 3

/-- Represents the fraction of the original area that remains black after the specified number of changes -/
def final_black_fraction : ℚ := remaining_black_fraction ^ num_changes

/-- Theorem stating that the final black fraction is equal to 512/729 -/
theorem final_black_fraction_is_512_729 : 
  final_black_fraction = 512 / 729 := by sorry

end final_black_fraction_is_512_729_l2884_288437


namespace baking_difference_l2884_288493

/-- Given a recipe and current state of baking, calculate the difference between
    remaining sugar and flour to be added. -/
def sugar_flour_difference (total_flour total_sugar added_flour : ℕ) : ℕ :=
  total_sugar - (total_flour - added_flour)

/-- Theorem stating the difference between remaining sugar and flour to be added
    for the given recipe and current state. -/
theorem baking_difference : sugar_flour_difference 9 11 4 = 6 := by
  sorry

end baking_difference_l2884_288493


namespace calculate_b_investment_l2884_288423

/-- Calculates B's investment in a partnership given the investments of A and C, 
    the total profit, and A's share of the profit. -/
theorem calculate_b_investment (a_investment c_investment total_profit a_profit : ℕ) : 
  a_investment = 6300 →
  c_investment = 10500 →
  total_profit = 14200 →
  a_profit = 4260 →
  ∃ b_investment : ℕ, 
    b_investment = 4220 ∧ 
    (a_investment : ℚ) / (a_investment + b_investment + c_investment) = 
    (a_profit : ℚ) / total_profit :=
by sorry

end calculate_b_investment_l2884_288423


namespace geometric_distribution_variance_l2884_288480

/-- A random variable following a geometric distribution with parameter p -/
def GeometricDistribution (p : ℝ) := { X : ℝ → ℝ // 0 < p ∧ p ≤ 1 }

/-- The variance of a random variable -/
def variance (X : ℝ → ℝ) : ℝ := sorry

/-- The theorem stating that the variance of a geometric distribution is (1-p)/p^2 -/
theorem geometric_distribution_variance (p : ℝ) (X : GeometricDistribution p) :
  variance X.val = (1 - p) / p^2 := by sorry

end geometric_distribution_variance_l2884_288480


namespace infinite_pairs_with_same_prime_factors_l2884_288455

theorem infinite_pairs_with_same_prime_factors :
  ∀ k : ℕ, k > 1 →
  ∃ m n : ℕ, m ≠ n ∧ m > 0 ∧ n > 0 ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)) ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ (m + 1) ↔ p ∣ (n + 1))) ∧
  m = 2^k - 2 ∧
  n = 2^k * (2^k - 2) :=
sorry

end infinite_pairs_with_same_prime_factors_l2884_288455


namespace intersection_y_coordinate_constant_l2884_288426

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x

-- Define the slope of the tangent at a point
def tangent_slope (p : PointOnParabola) : ℝ := 4 * p.x

-- Define perpendicular tangents
def perpendicular_tangents (p1 p2 : PointOnParabola) : Prop :=
  tangent_slope p1 * tangent_slope p2 = -1

-- Theorem statement
theorem intersection_y_coordinate_constant 
  (A B : PointOnParabola) 
  (h : perpendicular_tangents A B) : 
  ∃ (P : ℝ × ℝ), 
    (P.1 = (A.x + B.x) / 2) ∧ 
    (P.2 = -1/8) ∧
    (P.2 = 4 * A.x * (P.1 - A.x) + A.y) ∧
    (P.2 = 4 * B.x * (P.1 - B.x) + B.y) := by
  sorry

end intersection_y_coordinate_constant_l2884_288426


namespace hyperbola_eccentricity_l2884_288471

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity :
  ∀ (a : ℝ), a > 0 →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / 4 = 1) →
  (∀ (x y : ℝ), y^2 = 12*x) →
  (∃ (xf : ℝ), xf = 3 ∧ (∀ (y : ℝ), x^2 / a^2 - y^2 / 4 = 1 → (x - xf)^2 + y^2 = (3*a/5)^2)) →
  3 * Real.sqrt 5 / 5 = 3 / Real.sqrt (a^2) :=
by sorry

end hyperbola_eccentricity_l2884_288471


namespace zola_paityn_blue_hat_ratio_l2884_288440

/-- Proves the ratio of Zola's blue hats to Paityn's blue hats -/
theorem zola_paityn_blue_hat_ratio :
  let paityn_red : ℕ := 20
  let paityn_blue : ℕ := 24
  let zola_red : ℕ := (4 * paityn_red) / 5
  let total_hats : ℕ := 54 * 2
  let zola_blue : ℕ := total_hats - paityn_red - paityn_blue - zola_red
  (zola_blue : ℚ) / paityn_blue = 2 := by
  sorry

end zola_paityn_blue_hat_ratio_l2884_288440


namespace stock_price_decrease_l2884_288441

theorem stock_price_decrease (a : ℝ) (n : ℕ) (h1 : a > 0) : a * (0.99 ^ n) < a := by
  sorry

end stock_price_decrease_l2884_288441


namespace sum_precision_l2884_288428

theorem sum_precision (n : ℕ) (h : n ≤ 5) :
  ∃ (e : ℝ), e ≤ 0.001 ∧ n * e ≤ 0.01 :=
by sorry

end sum_precision_l2884_288428


namespace ant_walk_probability_l2884_288497

/-- A point on a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The ant's walk on the lattice -/
def AntWalk :=
  { p : LatticePoint // |p.x| + |p.y| ≥ 2 }

/-- Probability measure on the ant's walk -/
noncomputable def ProbMeasure : Type := AntWalk → ℝ

/-- The starting point of the ant -/
def start : LatticePoint := ⟨1, 0⟩

/-- The target end point -/
def target : LatticePoint := ⟨1, 1⟩

/-- Adjacent points are those that differ by 1 in exactly one coordinate -/
def adjacent (p q : LatticePoint) : Prop :=
  (abs (p.x - q.x) + abs (p.y - q.y) = 1)

/-- The probability measure satisfies the uniform distribution on adjacent points -/
axiom uniform_distribution (μ : ProbMeasure) (p : LatticePoint) :
  ∀ q : LatticePoint, adjacent p q → μ ⟨q, sorry⟩ = (1 : ℝ) / 4

/-- The main theorem: probability of reaching (1,1) from (1,0) is 7/24 -/
theorem ant_walk_probability (μ : ProbMeasure) :
  μ ⟨target, sorry⟩ = 7 / 24 := by sorry

end ant_walk_probability_l2884_288497


namespace quadratic_root_l2884_288465

theorem quadratic_root (a b c : ℝ) (h1 : 4 * a - 2 * b + c = 0) (h2 : a ≠ 0) :
  a * (-2)^2 + b * (-2) + c = 0 := by
  sorry

end quadratic_root_l2884_288465


namespace chord_equation_l2884_288462

/-- A chord of the hyperbola x^2 - y^2 = 1 with midpoint (2, 1) -/
structure Chord where
  /-- First endpoint of the chord -/
  p1 : ℝ × ℝ
  /-- Second endpoint of the chord -/
  p2 : ℝ × ℝ
  /-- The chord endpoints lie on the hyperbola -/
  h1 : p1.1^2 - p1.2^2 = 1
  h2 : p2.1^2 - p2.2^2 = 1
  /-- The midpoint of the chord is (2, 1) -/
  h3 : (p1.1 + p2.1) / 2 = 2
  h4 : (p1.2 + p2.2) / 2 = 1

/-- The equation of the line containing the chord is y = 2x - 3 -/
theorem chord_equation (c : Chord) : 
  ∃ (m b : ℝ), m = 2 ∧ b = -3 ∧ ∀ (x y : ℝ), y = m * x + b ↔ 
  ∃ (t : ℝ), x = (1 - t) * c.p1.1 + t * c.p2.1 ∧ y = (1 - t) * c.p1.2 + t * c.p2.2 :=
sorry

end chord_equation_l2884_288462


namespace solution_set_f_greater_than_two_range_of_t_l2884_288431

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 2/3 ∨ x < -6} :=
sorry

-- Theorem for the range of t
theorem range_of_t (t : ℝ) :
  (∃ x : ℝ, f x < 2 - 7/2 * t) ↔ (t < 3/2 ∨ t > 2) :=
sorry

end solution_set_f_greater_than_two_range_of_t_l2884_288431


namespace eight_squares_sharing_two_vertices_l2884_288403

/-- A square in a 2D plane -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ
  is_square : IsSquare vertices

/-- Two squares share two vertices -/
def SharesTwoVertices (s1 s2 : Square) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧ s1.vertices i = s2.vertices i ∧ s1.vertices j = s2.vertices j

/-- The main theorem -/
theorem eight_squares_sharing_two_vertices (s : Square) :
  ∃ (squares : Finset Square), squares.card = 8 ∧
    ∀ s' ∈ squares, SharesTwoVertices s s' ∧
    ∀ s', SharesTwoVertices s s' → s' ∈ squares :=
  sorry

end eight_squares_sharing_two_vertices_l2884_288403


namespace sum_of_possible_intersection_counts_l2884_288410

/-- A configuration of five distinct lines in a plane -/
structure LineConfiguration where
  lines : Finset (Set ℝ × ℝ)
  distinct : lines.card = 5

/-- The number of distinct intersection points in a configuration -/
def intersectionPoints (config : LineConfiguration) : ℕ :=
  sorry

/-- The set of all possible values for the number of intersection points -/
def possibleIntersectionCounts : Finset ℕ :=
  sorry

/-- Theorem: The sum of all possible values for the number of intersection points is 53 -/
theorem sum_of_possible_intersection_counts :
  (possibleIntersectionCounts.sum id) = 53 := by
  sorry

end sum_of_possible_intersection_counts_l2884_288410


namespace sqrt_one_plus_cos_alpha_l2884_288460

theorem sqrt_one_plus_cos_alpha (α : Real) (h : π < α ∧ α < 2*π) :
  Real.sqrt (1 + Real.cos α) = -Real.sqrt 2 * Real.cos (α/2) := by
  sorry

end sqrt_one_plus_cos_alpha_l2884_288460


namespace greatest_of_three_integers_l2884_288435

theorem greatest_of_three_integers (a b c : ℤ) : 
  a + b + c = 21 → 
  c = max a (max b c) →
  c = 8 →
  max a (max b c) = 8 := by
sorry

end greatest_of_three_integers_l2884_288435


namespace min_tiles_cover_floor_l2884_288409

/-- Represents the dimensions of a rectangle in inches -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle in square inches -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the tile dimensions -/
def tile : Rectangle := { length := 3, width := 4 }

/-- Represents the floor dimensions -/
def floor : Rectangle := { length := 36, width := 60 }

/-- Calculates the number of tiles needed to cover the floor -/
def tilesNeeded (t : Rectangle) (f : Rectangle) : ℕ :=
  (area f) / (area t)

theorem min_tiles_cover_floor :
  tilesNeeded tile floor = 180 := by sorry

end min_tiles_cover_floor_l2884_288409


namespace min_value_of_E_l2884_288475

theorem min_value_of_E :
  ∃ (E : ℝ), (∀ (x : ℝ), |x - 4| + |E| + |x - 5| ≥ 12) ∧
  (∀ (F : ℝ), (∀ (x : ℝ), |x - 4| + |F| + |x - 5| ≥ 12) → |F| ≥ |E|) ∧
  |E| = 11 :=
by
  sorry

end min_value_of_E_l2884_288475


namespace jump_rope_time_ratio_l2884_288447

/-- Given information about jump rope times for Cindy, Betsy, and Tina, 
    prove that the ratio of Tina's time to Betsy's time is 3. -/
theorem jump_rope_time_ratio :
  ∀ (cindy_time betsy_time tina_time : ℕ),
    cindy_time = 12 →
    betsy_time = cindy_time / 2 →
    tina_time = cindy_time + 6 →
    tina_time / betsy_time = 3 := by
  sorry

end jump_rope_time_ratio_l2884_288447


namespace max_gcd_consecutive_b_terms_l2884_288482

def b (n : ℕ) : ℕ := n.factorial + 2^n + n

theorem max_gcd_consecutive_b_terms :
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (b n) (b (n + 1)) ≤ k ∧
  ∃ (m : ℕ), Nat.gcd (b m) (b (m + 1)) = k ∧
  k = 2 :=
sorry

end max_gcd_consecutive_b_terms_l2884_288482


namespace total_hours_spent_l2884_288495

def project_time : ℕ := 300
def research_time : ℕ := 45
def presentation_time : ℕ := 75

def total_minutes : ℕ := project_time + research_time + presentation_time

theorem total_hours_spent : (total_minutes : ℚ) / 60 = 7 := by
  sorry

end total_hours_spent_l2884_288495


namespace circle_center_l2884_288459

/-- A circle passes through (0,1) and is tangent to y = (x-1)^2 at (3,4). Its center is (-2, 15/2). -/
theorem circle_center (c : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x - c.1)^2 + (y - c.2)^2 = (c.1 - 3)^2 + (c.2 - 4)^2 → 
    (x = 0 ∧ y = 1) ∨ (x = 3 ∧ y = 4)) →
  (∀ (x : ℝ), (x - 3)^2 + (((x - 1)^2 - 4)^2) / 16 = (c.1 - 3)^2 + (c.2 - 4)^2) →
  c = (-2, 15/2) := by
sorry

end circle_center_l2884_288459


namespace smallest_sum_of_reciprocals_l2884_288478

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24 → 
  (x : ℤ) + y ≤ (a : ℤ) + b → (x : ℤ) + y = 100 := by
  sorry

end smallest_sum_of_reciprocals_l2884_288478


namespace uncle_ben_eggs_l2884_288436

theorem uncle_ben_eggs (total_chickens : ℕ) (roosters : ℕ) (non_laying_hens : ℕ) (eggs_per_hen : ℕ) 
  (h1 : total_chickens = 440)
  (h2 : roosters = 39)
  (h3 : non_laying_hens = 15)
  (h4 : eggs_per_hen = 3) :
  total_chickens - roosters - non_laying_hens * eggs_per_hen = 1158 := by
  sorry

end uncle_ben_eggs_l2884_288436


namespace scientific_notation_equivalence_l2884_288408

/-- Proves that 2370000 is equal to 2.37 × 10^6 in scientific notation -/
theorem scientific_notation_equivalence :
  2370000 = 2.37 * (10 : ℝ)^6 := by
  sorry

end scientific_notation_equivalence_l2884_288408


namespace ball_distribution_after_199_students_l2884_288442

/-- Represents the state of the boxes -/
structure BoxState :=
  (a b c d e : ℕ)

/-- Simulates one student's action -/
def moveOneBall (state : BoxState) : BoxState :=
  let minBox := min state.a (min state.b (min state.c (min state.d state.e)))
  { a := if state.a > minBox then state.a - 1 else state.a + 4,
    b := if state.b > minBox then state.b - 1 else state.b + 4,
    c := if state.c > minBox then state.c - 1 else state.c + 4,
    d := if state.d > minBox then state.d - 1 else state.d + 4,
    e := if state.e > minBox then state.e - 1 else state.e + 4 }

/-- Simulates n students' actions -/
def simulateNStudents (n : ℕ) (initialState : BoxState) : BoxState :=
  match n with
  | 0 => initialState
  | n + 1 => moveOneBall (simulateNStudents n initialState)

/-- The main theorem to prove -/
theorem ball_distribution_after_199_students :
  let initialState : BoxState := ⟨9, 5, 3, 2, 1⟩
  let finalState := simulateNStudents 199 initialState
  finalState = ⟨5, 6, 4, 3, 2⟩ := by
  sorry


end ball_distribution_after_199_students_l2884_288442


namespace hyperbola_asymptote_l2884_288486

/-- Given a hyperbola (x^2 / a^2) - (y^2 / 81) = 1 with a > 0, if one of its asymptotes is y = 3x, then a = 3 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 81 = 1 → ∃ k : ℝ, y = k * x ∧ |k| = 9 / a) →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / 81 = 1 ∧ y = 3 * x) →
  a = 3 :=
by sorry

end hyperbola_asymptote_l2884_288486


namespace root_values_l2884_288450

-- Define the polynomial
def polynomial (a b c : ℝ) (x : ℂ) : ℂ := x^3 + a*x^2 + b*x - c

-- State the theorem
theorem root_values (a b c : ℝ) :
  (polynomial a b c (1 - 2*I) = 0) →
  (polynomial a b c (2 - I) = 0) →
  (a, b, c) = (-6, 21, -30) := by
  sorry

end root_values_l2884_288450
