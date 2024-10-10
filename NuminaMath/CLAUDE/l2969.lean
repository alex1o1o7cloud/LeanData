import Mathlib

namespace third_column_second_row_l2969_296921

/-- Represents a position in a classroom grid -/
structure Position :=
  (column : ℕ)
  (row : ℕ)

/-- The coordinate system for the classroom -/
def classroom_coordinate_system : Position → Bool
  | ⟨1, 2⟩ => true  -- This represents the condition that (1,2) is a valid position
  | _ => false

/-- Theorem: In the given coordinate system, (3,2) represents the 3rd column and 2nd row -/
theorem third_column_second_row :
  classroom_coordinate_system ⟨1, 2⟩ → 
  (∃ p : Position, p.column = 3 ∧ p.row = 2 ∧ classroom_coordinate_system p) :=
sorry

end third_column_second_row_l2969_296921


namespace imaginary_unit_calculation_l2969_296999

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_unit_calculation : i * (1 + i)^2 = -2 := by sorry

end imaginary_unit_calculation_l2969_296999


namespace six_digit_multiple_of_nine_l2969_296968

theorem six_digit_multiple_of_nine :
  ∀ (d : ℕ), d < 10 →
  (456780 + d) % 9 = 0 ↔ d = 6 := by
  sorry

end six_digit_multiple_of_nine_l2969_296968


namespace average_and_difference_l2969_296908

theorem average_and_difference (x : ℝ) : 
  (23 + x) / 2 = 27 → |x - 23| = 8 := by
sorry

end average_and_difference_l2969_296908


namespace rectangle_length_l2969_296951

/-- Given a rectangle where the length is three times the width, and decreasing the length by 5
    while increasing the width by 5 results in a square, prove that the original length is 15. -/
theorem rectangle_length (w : ℝ) (h1 : w > 0) : 
  (∃ l : ℝ, l = 3 * w ∧ l - 5 = w + 5) → 3 * w = 15 := by
  sorry

end rectangle_length_l2969_296951


namespace sarahs_bowling_score_l2969_296917

theorem sarahs_bowling_score (greg_score sarah_score : ℕ) : 
  sarah_score = greg_score + 50 → 
  (sarah_score + greg_score) / 2 = 110 → 
  sarah_score = 135 := by
  sorry

end sarahs_bowling_score_l2969_296917


namespace remainder_91_power_91_mod_100_l2969_296936

/-- The remainder when 91^91 is divided by 100 is 91. -/
theorem remainder_91_power_91_mod_100 : 91^91 % 100 = 91 := by
  sorry

end remainder_91_power_91_mod_100_l2969_296936


namespace intersection_area_zero_l2969_296997

-- Define the triangle vertices
def P : ℝ × ℝ := (3, -2)
def Q : ℝ × ℝ := (5, 4)
def R : ℝ × ℝ := (1, 1)

-- Define the reflection function across y = 0
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Define the triangle and its reflection
def triangle : Set (ℝ × ℝ) := {p | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ 
  p = (a * P.1 + b * Q.1 + c * R.1, a * P.2 + b * Q.2 + c * R.2)}

def reflectedTriangle : Set (ℝ × ℝ) := {p | ∃ q ∈ triangle, p = reflect q}

-- State the theorem
theorem intersection_area_zero : 
  MeasureTheory.volume (triangle ∩ reflectedTriangle) = 0 := by sorry

end intersection_area_zero_l2969_296997


namespace repeating_decimal_subtraction_l2969_296995

theorem repeating_decimal_subtraction : 
  ∃ (x y : ℚ), (∀ n : ℕ, (10 * x - x.floor) * 10^n % 10 = 4) ∧ 
               (∀ n : ℕ, (10 * y - y.floor) * 10^n % 10 = 6) ∧ 
               (x - y = -2/9) := by
  sorry

end repeating_decimal_subtraction_l2969_296995


namespace exam_students_count_l2969_296906

/-- The total number of students in an examination -/
def total_students : ℕ := 400

/-- The number of students who failed the examination -/
def failed_students : ℕ := 260

/-- The percentage of students who passed the examination -/
def pass_percentage : ℚ := 35 / 100

theorem exam_students_count :
  (1 - pass_percentage) * total_students = failed_students :=
sorry

end exam_students_count_l2969_296906


namespace gold_cube_profit_calculation_l2969_296954

/-- Calculates the profit from selling a gold cube -/
def goldCubeProfit (side : ℝ) (density : ℝ) (purchasePrice : ℝ) (markupFactor : ℝ) : ℝ :=
  let volume := side^3
  let mass := volume * density
  let cost := mass * purchasePrice
  let sellingPrice := cost * markupFactor
  sellingPrice - cost

/-- Theorem stating the profit from selling a specific gold cube -/
theorem gold_cube_profit_calculation :
  goldCubeProfit 6 19 60 1.5 = 123120 := by sorry

end gold_cube_profit_calculation_l2969_296954


namespace paco_cookies_theorem_l2969_296983

/-- Represents the number of cookies Paco has after all actions --/
def remaining_cookies (initial_salty initial_sweet initial_chocolate : ℕ)
  (eaten_sweet eaten_salty : ℕ) (given_chocolate received_chocolate : ℕ) :
  ℕ × ℕ × ℕ :=
  let remaining_sweet := initial_sweet - eaten_sweet
  let remaining_salty := initial_salty - eaten_salty
  let remaining_chocolate := initial_chocolate - given_chocolate + received_chocolate
  (remaining_sweet, remaining_salty, remaining_chocolate)

/-- Theorem stating the final number of cookies Paco has --/
theorem paco_cookies_theorem :
  remaining_cookies 97 34 45 15 56 22 7 = (19, 41, 30) := by
  sorry

end paco_cookies_theorem_l2969_296983


namespace jinas_mascots_l2969_296971

/-- The number of mascots Jina has -/
def total_mascots (initial_teddies : ℕ) (bunny_multiplier : ℕ) (koalas : ℕ) (additional_teddies_per_bunny : ℕ) : ℕ :=
  let bunnies := initial_teddies * bunny_multiplier
  let additional_teddies := bunnies * additional_teddies_per_bunny
  initial_teddies + bunnies + koalas + additional_teddies

/-- Theorem stating the total number of mascots Jina has -/
theorem jinas_mascots :
  total_mascots 5 3 1 2 = 51 := by
  sorry

end jinas_mascots_l2969_296971


namespace sin_plus_two_cos_l2969_296978

/-- Given a point P(-3, 4) on the terminal side of angle α, prove that sin α + 2cos α = -2/5 -/
theorem sin_plus_two_cos (α : Real) (P : ℝ × ℝ) (h : P = (-3, 4)) : 
  Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end sin_plus_two_cos_l2969_296978


namespace newspaper_cost_difference_l2969_296993

/-- Grant's yearly newspaper expenditure -/
def grant_yearly_cost : ℝ := 200

/-- Juanita's weekday newspaper cost -/
def juanita_weekday_cost : ℝ := 0.5

/-- Juanita's Sunday newspaper cost -/
def juanita_sunday_cost : ℝ := 2

/-- Number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Number of weekdays in a week -/
def weekdays_per_week : ℕ := 6

/-- Juanita's weekly newspaper cost -/
def juanita_weekly_cost : ℝ := juanita_weekday_cost * weekdays_per_week + juanita_sunday_cost

/-- Juanita's yearly newspaper cost -/
def juanita_yearly_cost : ℝ := juanita_weekly_cost * weeks_per_year

theorem newspaper_cost_difference : juanita_yearly_cost - grant_yearly_cost = 60 := by
  sorry

end newspaper_cost_difference_l2969_296993


namespace quadratic_solution_product_l2969_296962

theorem quadratic_solution_product (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) → 
  (3 * q^2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = 122 := by
sorry

end quadratic_solution_product_l2969_296962


namespace square_area_l2969_296930

theorem square_area (side_length : ℝ) (h : side_length = 7) : side_length ^ 2 = 49 := by
  sorry

end square_area_l2969_296930


namespace distance_walked_l2969_296965

/-- Proves that the distance walked is 18 miles given specific conditions on speed changes and time. -/
theorem distance_walked (speed : ℝ) (time : ℝ) : 
  speed > 0 → 
  time > 0 → 
  (speed + 1) * (3 * time / 4) = speed * time → 
  (speed - 1) * (time + 3) = speed * time → 
  speed * time = 18 := by
  sorry

end distance_walked_l2969_296965


namespace lauren_change_l2969_296992

/-- Represents the grocery items with their prices and discounts --/
structure GroceryItems where
  hamburger_meat_price : ℝ
  hamburger_meat_discount : ℝ
  hamburger_buns_price : ℝ
  lettuce_price : ℝ
  tomato_price : ℝ
  tomato_weight : ℝ
  onion_price : ℝ
  onion_weight : ℝ
  pickles_price : ℝ
  pickles_coupon : ℝ
  potatoes_price : ℝ
  soda_price : ℝ
  soda_discount : ℝ

/-- Calculates the total cost of the grocery items including tax --/
def calculateTotalCost (items : GroceryItems) (tax_rate : ℝ) : ℝ :=
  let hamburger_meat_cost := 2 * items.hamburger_meat_price * (1 - items.hamburger_meat_discount)
  let hamburger_buns_cost := items.hamburger_buns_price
  let tomato_cost := items.tomato_price * items.tomato_weight
  let onion_cost := items.onion_price * items.onion_weight
  let pickles_cost := items.pickles_price - items.pickles_coupon
  let soda_cost := items.soda_price * (1 - items.soda_discount)
  let subtotal := hamburger_meat_cost + hamburger_buns_cost + items.lettuce_price + 
                  tomato_cost + onion_cost + pickles_cost + items.potatoes_price + soda_cost
  subtotal * (1 + tax_rate)

/-- Proves that Lauren's change from a $50 bill is $24.67 --/
theorem lauren_change (items : GroceryItems) (tax_rate : ℝ) :
  items.hamburger_meat_price = 3.5 →
  items.hamburger_meat_discount = 0.15 →
  items.hamburger_buns_price = 1.5 →
  items.lettuce_price = 1 →
  items.tomato_price = 2 →
  items.tomato_weight = 1.5 →
  items.onion_price = 0.75 →
  items.onion_weight = 0.5 →
  items.pickles_price = 2.5 →
  items.pickles_coupon = 1 →
  items.potatoes_price = 4 →
  items.soda_price = 5.99 →
  items.soda_discount = 0.07 →
  tax_rate = 0.06 →
  50 - calculateTotalCost items tax_rate = 24.67 := by
  sorry

end lauren_change_l2969_296992


namespace fourth_root_squared_l2969_296975

theorem fourth_root_squared (x : ℝ) : (x^(1/4))^2 = 16 → x = 256 := by
  sorry

end fourth_root_squared_l2969_296975


namespace share_ratio_l2969_296958

theorem share_ratio (total : ℚ) (a b c : ℚ) : 
  total = 510 →
  b = (1/4) * c →
  a + b + c = total →
  a = 360 →
  a / b = 12 := by
sorry

end share_ratio_l2969_296958


namespace probability_no_consecutive_ones_l2969_296979

/-- Represents a binary sequence -/
def BinarySequence := List Bool

/-- Checks if a binary sequence contains two consecutive 1s -/
def hasConsecutiveOnes : BinarySequence → Bool :=
  fun seq => sorry

/-- Generates all valid 12-digit binary sequences starting with 1 -/
def generateSequences : List BinarySequence :=
  sorry

/-- Counts the number of sequences without consecutive 1s -/
def countValidSequences : Nat :=
  sorry

/-- The total number of possible 12-digit sequences starting with 1 -/
def totalSequences : Nat := 2^11

theorem probability_no_consecutive_ones :
  (countValidSequences : ℚ) / totalSequences = 233 / 2048 :=
sorry

end probability_no_consecutive_ones_l2969_296979


namespace sandwich_non_condiment_percentage_l2969_296940

theorem sandwich_non_condiment_percentage
  (total_weight : ℝ)
  (condiment_weight : ℝ)
  (h1 : total_weight = 150)
  (h2 : condiment_weight = 45) :
  (total_weight - condiment_weight) / total_weight * 100 = 70 := by
  sorry

end sandwich_non_condiment_percentage_l2969_296940


namespace max_absolute_value_of_z_l2969_296938

theorem max_absolute_value_of_z (z : ℂ) : 
  Complex.abs (z - (3 + 4*I)) ≤ 2 → Complex.abs z ≤ 7 ∧ ∃ w : ℂ, Complex.abs (w - (3 + 4*I)) ≤ 2 ∧ Complex.abs w = 7 :=
sorry

end max_absolute_value_of_z_l2969_296938


namespace solve_equation_l2969_296974

theorem solve_equation : ∃ x : ℝ, 3 * x + 36 = 48 ∧ x = 4 := by
  sorry

end solve_equation_l2969_296974


namespace range_of_m_l2969_296970

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - m| < 1 → x^2 - 8*x + 12 < 0) ∧ 
  (∃ x, x^2 - 8*x + 12 < 0 ∧ |x - m| ≥ 1)) →
  (3 ≤ m ∧ m ≤ 5) := by
sorry

end range_of_m_l2969_296970


namespace notebook_cost_l2969_296931

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : 
  total_students = 30 →
  total_cost = 1584 →
  ∃ (students_bought notebooks_per_student cost_per_notebook : Nat),
    students_bought = 20 ∧
    students_bought * notebooks_per_student * cost_per_notebook = total_cost ∧
    cost_per_notebook ≥ notebooks_per_student ∧
    cost_per_notebook = 11 := by
  sorry

end notebook_cost_l2969_296931


namespace root_existence_implies_n_range_l2969_296910

-- Define the function f
def f (m n x : ℝ) : ℝ := m * x^2 - (5 * m + n) * x + n

-- State the theorem
theorem root_existence_implies_n_range :
  (∀ m ∈ Set.Ioo (-2 : ℝ) (-1 : ℝ),
    ∃ x ∈ Set.Ioo (3 : ℝ) (5 : ℝ), f m n x = 0) →
  n ∈ Set.Ioo (0 : ℝ) (3 : ℝ) :=
by sorry

end root_existence_implies_n_range_l2969_296910


namespace inequality_property_l2969_296912

theorem inequality_property (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end inequality_property_l2969_296912


namespace circle_diameter_from_area_l2969_296914

theorem circle_diameter_from_area : 
  ∀ (A d : ℝ), A = 78.53981633974483 → d = 10 → A = π * (d / 2)^2 :=
by sorry

end circle_diameter_from_area_l2969_296914


namespace log_exponent_sum_l2969_296923

theorem log_exponent_sum (a : ℝ) (h : a = Real.log 3 / Real.log 4) : 
  2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 := by sorry

end log_exponent_sum_l2969_296923


namespace problem_solution_l2969_296900

def y (m x : ℝ) : ℝ := (m + 1) * x^2 - m * x + m - 1

theorem problem_solution :
  (∀ m : ℝ, (∀ x : ℝ, y m x ≥ 0) ↔ m ≥ 2 * Real.sqrt 3 / 3) ∧
  (∀ m : ℝ, m > -2 →
    (∀ x : ℝ, y m x ≥ m) ↔
      (m = -1 ∧ x ≥ 1) ∨
      (m > -1 ∧ (x ≤ -1 / (m + 1) ∨ x ≥ 1)) ∨
      (-2 < m ∧ m < -1 ∧ 1 ≤ x ∧ x ≤ -1 / (m + 1))) :=
by sorry

end problem_solution_l2969_296900


namespace more_cats_than_dogs_l2969_296959

theorem more_cats_than_dogs : 
  let num_dogs : ℕ := 9
  let num_cats : ℕ := 23
  num_cats - num_dogs = 14 := by sorry

end more_cats_than_dogs_l2969_296959


namespace correct_operation_l2969_296976

theorem correct_operation (a : ℝ) : 2 * a^2 * (3 * a) = 6 * a^3 := by
  sorry

end correct_operation_l2969_296976


namespace infinitely_many_solutions_l2969_296973

/-- There exist infinitely many ordered quadruples (x, y, z, w) of real numbers
    satisfying the given conditions. -/
theorem infinitely_many_solutions :
  ∃ (S : Set (ℝ × ℝ × ℝ × ℝ)), Set.Infinite S ∧
    ∀ (x y z w : ℝ), (x, y, z, w) ∈ S →
      (x + y = 3 ∧ x * y - z^2 = w ∧ w + z = 4) :=
by sorry

end infinitely_many_solutions_l2969_296973


namespace parabola_directrix_l2969_296957

/-- The equation of the directrix of the parabola y = -4x^2 - 16x + 1 -/
theorem parabola_directrix : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = -4 * x^2 - 16 * x + 1 ↔ y = a * (x - b)^2 + c) →
    (∃ d : ℝ, d = 273 / 16 ∧ 
      ∀ x y : ℝ, y = d → 
        (x - b)^2 + (y - c)^2 = (y - (c - 1 / (4 * |a|)))^2) :=
sorry

end parabola_directrix_l2969_296957


namespace greatest_divisor_with_remainders_l2969_296909

theorem greatest_divisor_with_remainders (d : ℕ) : d > 0 ∧ 
  d ∣ (4351 - 8) ∧ 
  d ∣ (5161 - 10) ∧ 
  (∀ k : ℕ, k > d → k ∣ (4351 - 8) → k ∣ (5161 - 10) → 
    (4351 % k ≠ 8 ∨ 5161 % k ≠ 10)) → 
  d = 1 :=
sorry

end greatest_divisor_with_remainders_l2969_296909


namespace harmonic_mean_counterexample_l2969_296945

theorem harmonic_mean_counterexample :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2 / (1/a + 1/b) < Real.sqrt (a * b)) := by
  sorry

end harmonic_mean_counterexample_l2969_296945


namespace butterfly_stickers_l2969_296920

/-- Given a collection of butterflies with the following properties:
  * There are 330 butterflies in total
  * They are numbered consecutively starting from 1
  * 21 butterflies have double-digit numbers
  * 4 butterflies have triple-digit numbers
  Prove that the total number of single-digit stickers needed is 63 -/
theorem butterfly_stickers (total : ℕ) (double_digit : ℕ) (triple_digit : ℕ)
  (h_total : total = 330)
  (h_double : double_digit = 21)
  (h_triple : triple_digit = 4)
  (h_consecutive : ∀ n : ℕ, n ≤ total → n ≥ 1)
  (h_double_range : ∀ n : ℕ, n ≥ 10 ∧ n < 100 → n ≤ 30)
  (h_triple_range : ∀ n : ℕ, n ≥ 100 ∧ n < 1000 → n ≤ 103) :
  (total - double_digit - triple_digit) +
  (double_digit * 2) +
  (triple_digit * 3) = 63 := by
sorry

end butterfly_stickers_l2969_296920


namespace train_bridge_crossing_time_l2969_296922

theorem train_bridge_crossing_time
  (train_length : ℝ)
  (bridge_length : ℝ)
  (train_speed_kmph : ℝ)
  (h1 : train_length = 165)
  (h2 : bridge_length = 660)
  (h3 : train_speed_kmph = 90) :
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 33 := by
  sorry

end train_bridge_crossing_time_l2969_296922


namespace complement_of_A_wrt_U_l2969_296988

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {3, 4, 5}

theorem complement_of_A_wrt_U : (U \ A) = {1, 2, 6} := by sorry

end complement_of_A_wrt_U_l2969_296988


namespace yarn_parts_count_l2969_296969

/-- Given a yarn of 10 meters cut into equal parts, where 3 parts equal 6 meters,
    prove that the yarn was cut into 5 parts. -/
theorem yarn_parts_count (total_length : ℝ) (used_parts : ℕ) (used_length : ℝ) :
  total_length = 10 →
  used_parts = 3 →
  used_length = 6 →
  (total_length / (used_length / used_parts : ℝ) : ℝ) = 5 := by
  sorry

end yarn_parts_count_l2969_296969


namespace smallest_covering_set_smallest_n_is_five_l2969_296967

theorem smallest_covering_set (n : ℕ) : Prop :=
  ∃ (k a : Fin n → ℕ),
    (∀ i j : Fin n, i < j → 1 < k i ∧ k i < k j) ∧
    (∀ N : ℤ, ∃ i : Fin n, (k i : ℤ) ∣ (N - (a i : ℤ)))

theorem smallest_n_is_five :
  (∃ n : ℕ, smallest_covering_set n) ∧
  (∀ m : ℕ, smallest_covering_set m → m ≥ 5) ∧
  smallest_covering_set 5 :=
sorry

end smallest_covering_set_smallest_n_is_five_l2969_296967


namespace april_coffee_cost_l2969_296944

/-- The number of coffees Jon buys per day -/
def coffees_per_day : ℕ := 2

/-- The cost of one coffee in dollars -/
def cost_per_coffee : ℕ := 2

/-- The number of days in April -/
def days_in_april : ℕ := 30

/-- The total cost of coffee for Jon in April -/
def total_cost : ℕ := coffees_per_day * cost_per_coffee * days_in_april

theorem april_coffee_cost : total_cost = 120 := by
  sorry

end april_coffee_cost_l2969_296944


namespace machine_quality_l2969_296949

/-- Represents a packaging machine --/
structure PackagingMachine where
  weight : Real → Real  -- Random variable representing packaging weight

/-- Defines the expected value of a random variable --/
def expectedValue (X : Real → Real) : Real :=
  sorry

/-- Defines the variance of a random variable --/
def variance (X : Real → Real) : Real :=
  sorry

/-- Determines if a packaging machine has better quality --/
def betterQuality (m1 m2 : PackagingMachine) : Prop :=
  expectedValue m1.weight = expectedValue m2.weight ∧
  variance m1.weight > variance m2.weight →
  sorry  -- This represents that m2 has better quality

/-- Theorem stating which machine has better quality --/
theorem machine_quality (A B : PackagingMachine) :
  betterQuality A B → sorry  -- This represents that B has better quality
:= by sorry

end machine_quality_l2969_296949


namespace expected_weight_of_disks_l2969_296955

/-- The expected weight of 100 disks with manufacturing errors -/
theorem expected_weight_of_disks (nominal_diameter : Real) (perfect_weight : Real) 
  (radius_std_dev : Real) (h1 : nominal_diameter = 1) (h2 : perfect_weight = 100) 
  (h3 : radius_std_dev = 0.01) : 
  ∃ (expected_weight : Real), 
    expected_weight = 10004 ∧ 
    expected_weight = 100 * perfect_weight * (1 + (radius_std_dev / (nominal_diameter / 2))^2) :=
by sorry

end expected_weight_of_disks_l2969_296955


namespace expression_equality_l2969_296961

theorem expression_equality : 40 + 5 * 12 / (180 / 3) = 41 := by
  sorry

end expression_equality_l2969_296961


namespace wage_comparison_l2969_296952

/-- Proves that given the wage relationships between Erica, Robin, and Charles,
    Charles earns approximately 170% more than Erica. -/
theorem wage_comparison (erica robin charles : ℝ) 
  (h1 : robin = erica * 1.30)
  (h2 : charles = robin * 1.3076923076923077) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0000001 ∧ 
  charles = erica * (2.70 + ε) :=
sorry

end wage_comparison_l2969_296952


namespace average_of_six_numbers_l2969_296977

theorem average_of_six_numbers 
  (total_average : ℝ) 
  (second_pair_average : ℝ) 
  (third_pair_average : ℝ) 
  (h1 : total_average = 3.9) 
  (h2 : second_pair_average = 3.85) 
  (h3 : third_pair_average = 4.45) : 
  ∃ first_pair_average : ℝ, first_pair_average = 3.4 ∧ 
  6 * total_average = 2 * first_pair_average + 2 * second_pair_average + 2 * third_pair_average :=
sorry

end average_of_six_numbers_l2969_296977


namespace camera_price_difference_l2969_296939

/-- The list price of Camera Y in dollars -/
def list_price : ℝ := 59.99

/-- The discount percentage at Budget Buys -/
def budget_buys_discount : ℝ := 0.15

/-- The discount amount at Frugal Finds in dollars -/
def frugal_finds_discount : ℝ := 20

/-- The sale price at Budget Buys in dollars -/
def budget_buys_price : ℝ := list_price * (1 - budget_buys_discount)

/-- The sale price at Frugal Finds in dollars -/
def frugal_finds_price : ℝ := list_price - frugal_finds_discount

/-- The price difference in cents -/
def price_difference_cents : ℝ := (budget_buys_price - frugal_finds_price) * 100

theorem camera_price_difference :
  price_difference_cents = 1099.15 := by
  sorry

end camera_price_difference_l2969_296939


namespace crazy_silly_school_series_l2969_296932

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 14

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 15

/-- The number of books read -/
def books_read : ℕ := 11

/-- The number of movies watched -/
def movies_watched : ℕ := 40

theorem crazy_silly_school_series :
  (num_books = num_movies + 1) ∧
  (num_books = 15) ∧
  (books_read = 11) ∧
  (movies_watched = 40) →
  num_movies = 14 := by
sorry

end crazy_silly_school_series_l2969_296932


namespace unique_number_exists_l2969_296905

/-- A function that checks if a natural number consists only of digits 2 and 5 -/
def only_2_and_5 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 5

/-- The theorem to be proved -/
theorem unique_number_exists : ∃! n : ℕ,
  only_2_and_5 n ∧
  n.digits 10 = List.replicate 2005 0 ∧
  n % (2^2005) = 0 :=
sorry

end unique_number_exists_l2969_296905


namespace relationship_abcd_l2969_296919

theorem relationship_abcd :
  let a : ℝ := 10 / 7
  let b : ℝ := Real.log 3
  let c : ℝ := 2 * Real.sqrt 3 / 3
  let d : ℝ := Real.exp 0.3
  a > d ∧ d > c ∧ c > b := by sorry

end relationship_abcd_l2969_296919


namespace john_running_distance_l2969_296941

def monday_distance : ℕ := 1700
def tuesday_distance : ℕ := monday_distance + 200
def wednesday_distance : ℕ := (7 * tuesday_distance) / 10
def thursday_distance : ℕ := 2 * wednesday_distance
def friday_distance : ℕ := 3500

def total_distance : ℕ := monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance

theorem john_running_distance : total_distance = 10090 := by
  sorry

end john_running_distance_l2969_296941


namespace min_value_K_l2969_296946

theorem min_value_K (α β γ : ℝ) (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  (α + 3*γ)/(α + 2*β + γ) + 4*β/(α + β + 2*γ) - 8*γ/(α + β + 3*γ) ≥ 2/5 := by
  sorry

end min_value_K_l2969_296946


namespace wand_cost_proof_l2969_296911

/-- The cost of each wand --/
def wand_cost : ℚ := 115 / 3

/-- The number of wands Kate bought --/
def num_wands : ℕ := 3

/-- The additional amount Kate charged when selling each wand --/
def additional_charge : ℚ := 5

/-- The total amount Kate collected after selling all wands --/
def total_collected : ℚ := 130

theorem wand_cost_proof : 
  num_wands * (wand_cost + additional_charge) = total_collected :=
sorry

end wand_cost_proof_l2969_296911


namespace vase_capacity_l2969_296943

/-- The number of flowers each vase can hold -/
def flowers_per_vase (carnations : ℕ) (roses : ℕ) (vases : ℕ) : ℕ :=
  (carnations + roses) / vases

/-- Proof that each vase can hold 6 flowers -/
theorem vase_capacity : flowers_per_vase 7 47 9 = 6 := by
  sorry

end vase_capacity_l2969_296943


namespace percentage_spent_l2969_296963

theorem percentage_spent (initial_amount remaining_amount : ℝ) 
  (h1 : initial_amount = 5000)
  (h2 : remaining_amount = 3500) :
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 := by
  sorry

end percentage_spent_l2969_296963


namespace estimate_city_standards_l2969_296980

/-- Estimates the number of students meeting standards in a population based on a sample. -/
def estimate_meeting_standards (sample_size : ℕ) (sample_meeting : ℕ) (total_population : ℕ) : ℕ :=
  (total_population * sample_meeting) / sample_size

/-- Theorem stating the estimated number of students meeting standards in the city -/
theorem estimate_city_standards : 
  let sample_size := 1000
  let sample_meeting := 950
  let total_population := 1200000
  estimate_meeting_standards sample_size sample_meeting total_population = 1140000 := by
  sorry

end estimate_city_standards_l2969_296980


namespace child_playing_time_l2969_296982

/-- Calculates the playing time for each child in a game where 6 children take turns playing for 120 minutes, with only two children playing at a time. -/
theorem child_playing_time (total_time : ℕ) (num_children : ℕ) (players_per_game : ℕ) :
  total_time = 120 ∧ num_children = 6 ∧ players_per_game = 2 →
  (total_time * players_per_game) / num_children = 40 := by
  sorry

end child_playing_time_l2969_296982


namespace exists_a_with_median_4_l2969_296913

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
  2 * (s.filter (λ x => x ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ x => x ≥ m)).card ≥ s.card

theorem exists_a_with_median_4 : 
  ∃ a : ℝ, is_median {a, 2, 4, 0, 5} 4 := by
sorry

end exists_a_with_median_4_l2969_296913


namespace area_trace_proportionality_specific_area_trace_l2969_296915

/-- Given two concentric spheres and a smaller sphere tracing areas on both, 
    the areas traced are proportional to the square of the radii ratio. -/
theorem area_trace_proportionality 
  (R1 R2 r A1 : ℝ) 
  (h1 : 0 < r) 
  (h2 : r < R1) 
  (h3 : R1 < R2) 
  (h4 : 0 < A1) : 
  ∃ A2 : ℝ, A2 = A1 * (R2 / R1)^2 := by
  sorry

/-- The specific case with given values -/
theorem specific_area_trace 
  (R1 R2 r A1 : ℝ) 
  (h1 : r = 1) 
  (h2 : R1 = 4) 
  (h3 : R2 = 6) 
  (h4 : A1 = 17) : 
  ∃ A2 : ℝ, A2 = 38.25 := by
  sorry

end area_trace_proportionality_specific_area_trace_l2969_296915


namespace custom_operation_theorem_l2969_296935

def custom_operation (M N : Set ℕ) : Set ℕ :=
  {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

def M : Set ℕ := {0, 2, 4, 6, 8, 10}
def N : Set ℕ := {0, 3, 6, 9, 12, 15}

theorem custom_operation_theorem :
  custom_operation (custom_operation M N) M = N := by
  sorry

end custom_operation_theorem_l2969_296935


namespace tim_bought_three_goats_l2969_296972

/-- Proves that Tim bought 3 goats given the conditions of the problem -/
theorem tim_bought_three_goats
  (goat_cost : ℕ)
  (llama_count : ℕ → ℕ)
  (llama_cost : ℕ → ℕ)
  (total_spent : ℕ)
  (h1 : goat_cost = 400)
  (h2 : ∀ g, llama_count g = 2 * g)
  (h3 : ∀ g, llama_cost g = goat_cost + goat_cost / 2)
  (h4 : total_spent = 4800)
  (h5 : ∀ g, total_spent = g * goat_cost + llama_count g * llama_cost g) :
  ∃ g : ℕ, g = 3 ∧ total_spent = g * goat_cost + llama_count g * llama_cost g :=
sorry

end tim_bought_three_goats_l2969_296972


namespace inequality_proof_l2969_296902

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) := by
  sorry

end inequality_proof_l2969_296902


namespace expansion_terms_count_l2969_296984

theorem expansion_terms_count (N : ℕ+) : 
  (Nat.choose N 5 = 2002) ↔ (N = 16) := by sorry

end expansion_terms_count_l2969_296984


namespace geometric_sequence_condition_l2969_296942

def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_sequence_condition (a b c : ℝ) :
  (is_geometric_sequence a b c → b^2 = a*c) ∧
  ¬(b^2 = a*c → is_geometric_sequence a b c) :=
sorry

end geometric_sequence_condition_l2969_296942


namespace right_triangle_base_length_l2969_296990

theorem right_triangle_base_length 
  (height : ℝ) 
  (area : ℝ) 
  (hypotenuse : ℝ) 
  (h1 : height = 8) 
  (h2 : area = 24) 
  (h3 : hypotenuse = 10) : 
  ∃ (base : ℝ), base = 6 ∧ area = (1/2) * base * height ∧ hypotenuse^2 = height^2 + base^2 := by
sorry

end right_triangle_base_length_l2969_296990


namespace petya_counterexample_l2969_296956

theorem petya_counterexample : ∃ (a b : ℕ), 
  (a^5 % b^2 = 0) ∧ (a^2 % b ≠ 0) := by
  sorry

end petya_counterexample_l2969_296956


namespace jack_journey_time_l2969_296918

/-- Represents the time spent in a country during Jack's journey --/
structure CountryTime where
  customs : ℕ
  quarantine_days : ℕ

/-- Represents a layover during Jack's journey --/
structure Layover where
  duration : ℕ

/-- Calculates the total time spent in a country in hours --/
def total_country_time (ct : CountryTime) : ℕ :=
  ct.customs + ct.quarantine_days * 24

/-- Calculates the total time of Jack's journey in hours --/
def total_journey_time (canada : CountryTime) (australia : CountryTime) (japan : CountryTime)
                       (to_australia : Layover) (to_japan : Layover) : ℕ :=
  total_country_time canada + total_country_time australia + total_country_time japan +
  to_australia.duration + to_japan.duration

theorem jack_journey_time :
  let canada : CountryTime := ⟨20, 14⟩
  let australia : CountryTime := ⟨15, 10⟩
  let japan : CountryTime := ⟨10, 7⟩
  let to_australia : Layover := ⟨12⟩
  let to_japan : Layover := ⟨5⟩
  total_journey_time canada australia japan to_australia to_japan = 806 :=
by sorry

end jack_journey_time_l2969_296918


namespace square_area_from_diagonal_l2969_296926

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 64 := by
  sorry

end square_area_from_diagonal_l2969_296926


namespace complex_modulus_equality_l2969_296929

theorem complex_modulus_equality (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) :
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end complex_modulus_equality_l2969_296929


namespace greatest_perimeter_of_special_triangle_l2969_296996

theorem greatest_perimeter_of_special_triangle :
  ∀ a b c : ℕ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (b = 4 * a ∨ a = 4 * b ∨ c = 4 * a ∨ a = 4 * c ∨ b = 4 * c ∨ c = 4 * b) →
  (a = 12 ∨ b = 12 ∨ c = 12) →
  (a + b > c ∧ b + c > a ∧ a + c > b) →
  a + b + c ≤ 27 :=
by sorry

end greatest_perimeter_of_special_triangle_l2969_296996


namespace nitrogen_atomic_weight_l2969_296966

/-- The atomic weight of nitrogen in a compound with given properties -/
theorem nitrogen_atomic_weight (molecular_weight : ℝ) (hydrogen_weight : ℝ) (bromine_weight : ℝ) :
  molecular_weight = 98 →
  hydrogen_weight = 1.008 →
  bromine_weight = 79.904 →
  molecular_weight = 4 * hydrogen_weight + bromine_weight + 14.064 :=
by sorry

end nitrogen_atomic_weight_l2969_296966


namespace sum_of_special_numbers_l2969_296986

theorem sum_of_special_numbers : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a % 10^8 = 0 ∧ 
  b % 10^8 = 0 ∧ 
  (Nat.divisors a).card = 90 ∧ 
  (Nat.divisors b).card = 90 ∧ 
  a + b = 700000000 := by
sorry

end sum_of_special_numbers_l2969_296986


namespace substitution_remainder_l2969_296953

/-- Represents the number of available players -/
def total_players : ℕ := 15

/-- Represents the number of starting players -/
def starting_players : ℕ := 5

/-- Represents the maximum number of substitutions allowed -/
def max_substitutions : ℕ := 4

/-- Calculates the number of ways to make substitutions for a given number of substitutions -/
def substitution_ways (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then starting_players * (total_players - starting_players)
  else starting_players * (total_players - starting_players - n + 2) * substitution_ways (n - 1)

/-- Calculates the total number of ways to make substitutions -/
def total_substitution_ways : ℕ :=
  (List.range (max_substitutions + 1)).map substitution_ways |>.sum

/-- The main theorem stating that the remainder of total substitution ways divided by 1000 is 301 -/
theorem substitution_remainder :
  total_substitution_ways % 1000 = 301 := by sorry

end substitution_remainder_l2969_296953


namespace sin_18_degrees_l2969_296916

theorem sin_18_degrees : Real.sin (18 * π / 180) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end sin_18_degrees_l2969_296916


namespace alyssa_allowance_proof_l2969_296985

/-- Alyssa's weekly allowance -/
def weekly_allowance : ℝ := 240

theorem alyssa_allowance_proof :
  ∃ (A : ℝ),
    A > 0 ∧
    A / 2 + A / 5 + A / 4 + 12 = A ∧
    A = weekly_allowance :=
by sorry

end alyssa_allowance_proof_l2969_296985


namespace sum_of_coefficients_l2969_296991

theorem sum_of_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (2 - x) * (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 →
  a₀ + a₆ = -30 := by
sorry

end sum_of_coefficients_l2969_296991


namespace equation_solution_l2969_296904

theorem equation_solution :
  ∃ x : ℚ, (5 + 3.5 * x = 2.1 * x - 25) ∧ (x = -150 / 7) := by
  sorry

end equation_solution_l2969_296904


namespace B_equals_set_l2969_296925

def A : Set ℤ := {-2, -1, 1, 2, 3, 4}

def B : Set ℕ := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_set : B = {1, 4, 9, 16} := by sorry

end B_equals_set_l2969_296925


namespace farmer_wheat_harvest_l2969_296994

theorem farmer_wheat_harvest (estimated_harvest additional_harvest : ℕ) 
  (h1 : estimated_harvest = 213489)
  (h2 : additional_harvest = 13257) :
  estimated_harvest + additional_harvest = 226746 := by
  sorry

end farmer_wheat_harvest_l2969_296994


namespace expected_bounces_l2969_296928

/-- The expected number of bounces for a ball on a rectangular billiard table -/
theorem expected_bounces (table_length table_width ball_travel : ℝ) 
  (h_length : table_length = 3)
  (h_width : table_width = 1)
  (h_travel : ball_travel = 2) :
  ∃ (E : ℝ), E = 1 + (2 / Real.pi) * (Real.arccos (3/4) + Real.arccos (1/4) - Real.arcsin (3/4)) :=
by sorry

end expected_bounces_l2969_296928


namespace triangle_angle_C_l2969_296964

theorem triangle_angle_C (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  3 * Real.sin A + 4 * Real.cos B = 6 →
  4 * Real.sin B + 3 * Real.cos A = 1 →
  C = Real.pi / 6 := by sorry

end triangle_angle_C_l2969_296964


namespace difference_of_squares_l2969_296927

def digits : List Nat := [9, 8, 7, 6, 4, 2, 1, 5]

def largest_number : Nat := 98765421

def smallest_number : Nat := 12456789

theorem difference_of_squares (d : List Nat) (largest smallest : Nat) :
  d = digits →
  largest = largest_number →
  smallest = smallest_number →
  largest * largest - smallest * smallest = 9599477756293120 := by
  sorry

end difference_of_squares_l2969_296927


namespace rectangle_x_value_l2969_296981

/-- A rectangular construction with specified side lengths -/
structure RectConstruction where
  top_left : ℝ
  top_middle : ℝ
  top_right : ℝ
  bottom_left : ℝ
  bottom_middle : ℝ
  bottom_right : ℝ

/-- The theorem stating that X = 5 in the given rectangular construction -/
theorem rectangle_x_value (r : RectConstruction) 
  (h1 : r.top_left = 2)
  (h2 : r.top_right = 3)
  (h3 : r.bottom_left = 4)
  (h4 : r.bottom_middle = 1)
  (h5 : r.bottom_right = 5)
  (h6 : r.top_left + r.top_middle + r.top_right = r.bottom_left + r.bottom_middle + r.bottom_right) :
  r.top_middle = 5 := by
  sorry

#check rectangle_x_value

end rectangle_x_value_l2969_296981


namespace initial_sodium_chloride_percentage_l2969_296998

/-- Proves that given a tank with 10,000 gallons of solution, if 5,500 gallons of water evaporate
    and the remaining solution is 11.11111111111111% sodium chloride, then the initial percentage
    of sodium chloride was 5%. -/
theorem initial_sodium_chloride_percentage
  (initial_volume : ℝ)
  (evaporated_volume : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 10000)
  (h2 : evaporated_volume = 5500)
  (h3 : final_percentage = 11.11111111111111)
  (h4 : final_percentage = (100 * initial_volume * (initial_percentage / 100)) /
                           (initial_volume - evaporated_volume)) :
  initial_percentage = 5 :=
by sorry

end initial_sodium_chloride_percentage_l2969_296998


namespace sequence_sum_2000_is_zero_l2969_296901

def sequence_sum (n : ℕ) : ℤ :=
  let group_sum (k : ℕ) : ℤ := (4*k + 1) - (4*k + 2) - (4*k + 3) + (4*k + 4)
  (Finset.range (n/4)).sum (λ k => group_sum k)

theorem sequence_sum_2000_is_zero : sequence_sum 500 = 0 := by
  sorry

end sequence_sum_2000_is_zero_l2969_296901


namespace professor_seating_count_l2969_296989

/-- The number of chairs in a row --/
def num_chairs : ℕ := 9

/-- The number of professors --/
def num_professors : ℕ := 3

/-- The number of students --/
def num_students : ℕ := 6

/-- Represents the possible seating arrangements for professors --/
def professor_seating_arrangements : ℕ := sorry

/-- Theorem stating the number of ways professors can choose their chairs --/
theorem professor_seating_count :
  professor_seating_arrangements = 238 :=
sorry

end professor_seating_count_l2969_296989


namespace integer_roots_of_polynomial_l2969_296924

def polynomial (x b₂ b₁ : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial x b₂ b₁ = 0} ⊆ possible_roots :=
sorry

end integer_roots_of_polynomial_l2969_296924


namespace angle_measure_problem_l2969_296987

theorem angle_measure_problem (x : ℝ) : 
  x = 2 * (180 - x) + 30 → x = 130 := by
  sorry

end angle_measure_problem_l2969_296987


namespace rowing_distance_l2969_296950

/-- The distance to a destination given rowing conditions and round trip time -/
theorem rowing_distance (v : ℝ) (w : ℝ) (c : ℝ) (t : ℝ) (h1 : v > 0) (h2 : w ≥ 0) (h3 : c ≥ 0) (h4 : t > 0) :
  let d := (t * (v + c) * (v + c - w)) / ((v + c) + (v + c - w))
  d = 45 / 11 ↔ v = 4 ∧ w = 1 ∧ c = 2 ∧ t = 3/2 := by
  sorry

#check rowing_distance

end rowing_distance_l2969_296950


namespace problem_solution_l2969_296903

theorem problem_solution : 
  (1/2 - 1/4 + 1/12) * (-12) = -4 ∧ 
  -(3^2) + (-5)^2 * (4/5) - |(-6)| = 5 := by
sorry

end problem_solution_l2969_296903


namespace greatest_integer_with_gcf_4_l2969_296937

theorem greatest_integer_with_gcf_4 : ∃ n : ℕ, 
  n < 200 ∧ 
  Nat.gcd n 24 = 4 ∧ 
  ∀ m : ℕ, m < 200 → Nat.gcd m 24 = 4 → m ≤ n :=
by
  -- The proof would go here
  sorry

end greatest_integer_with_gcf_4_l2969_296937


namespace complex_number_in_fourth_quadrant_l2969_296934

def i : ℂ := Complex.I

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 1 / (1 + i)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l2969_296934


namespace prove_average_marks_l2969_296947

def average_marks (M P C : ℝ) : Prop :=
  M + P = 40 ∧ C = P + 20 → (M + C) / 2 = 30

theorem prove_average_marks :
  ∀ M P C : ℝ, average_marks M P C :=
by
  sorry

end prove_average_marks_l2969_296947


namespace value_of_expression_l2969_296933

theorem value_of_expression (x : ℝ) (h : 7 * x + 6 = 3 * x - 18) : 
  3 * (2 * x + 4) = -24 := by
sorry

end value_of_expression_l2969_296933


namespace juan_reading_speed_l2969_296960

/-- Proves that Juan reads 250 pages per hour given the conditions of the problem -/
theorem juan_reading_speed (lunch_trip : ℝ) (book_pages : ℕ) (office_to_lunch : ℝ) 
  (h1 : lunch_trip = 2 * office_to_lunch)
  (h2 : book_pages = 4000)
  (h3 : office_to_lunch = 4)
  (h4 : lunch_trip = (book_pages : ℝ) / (250 : ℝ)) : 
  (book_pages : ℝ) / (2 * lunch_trip) = 250 := by
sorry

end juan_reading_speed_l2969_296960


namespace eggs_left_l2969_296948

theorem eggs_left (initial_eggs : ℕ) (taken_eggs : ℕ) (h1 : initial_eggs = 47) (h2 : taken_eggs = 5) :
  initial_eggs - taken_eggs = 42 := by
sorry

end eggs_left_l2969_296948


namespace store_a_cheaper_than_b_l2969_296907

/-- Represents the number of tennis rackets to be purchased -/
def num_rackets : ℕ := 30

/-- Represents the price of a tennis racket in yuan -/
def racket_price : ℕ := 100

/-- Represents the price of a can of tennis balls in yuan -/
def ball_price : ℕ := 20

/-- Represents the discount factor for Store B -/
def store_b_discount : ℚ := 9/10

/-- Theorem comparing costs of purchasing from Store A and Store B -/
theorem store_a_cheaper_than_b (x : ℕ) (h : x > num_rackets) :
  (20 : ℚ) * x + 2400 < (18 : ℚ) * x + 2700 ↔ x < 150 := by
  sorry

end store_a_cheaper_than_b_l2969_296907
