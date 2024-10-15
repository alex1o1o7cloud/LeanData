import Mathlib

namespace NUMINAMATH_CALUDE_minimum_dinner_cost_l3079_307990

/-- Represents an ingredient with its cost, quantity, and number of servings -/
structure Ingredient where
  name : String
  cost : ℚ
  quantity : ℚ
  servings : ℕ

/-- Calculates the minimum number of units needed to serve a given number of people -/
def minUnitsNeeded (servingsPerUnit : ℕ) (people : ℕ) : ℕ :=
  (people + servingsPerUnit - 1) / servingsPerUnit

/-- Calculates the total cost for an ingredient given the number of people to serve -/
def ingredientCost (i : Ingredient) (people : ℕ) : ℚ :=
  i.cost * (minUnitsNeeded i.servings people : ℚ)

/-- The list of ingredients for the dinner -/
def ingredients : List Ingredient := [
  ⟨"Pasta", 112/100, 500, 5⟩,
  ⟨"Meatballs", 524/100, 500, 4⟩,
  ⟨"Tomato sauce", 231/100, 400, 5⟩,
  ⟨"Tomatoes", 147/100, 400, 4⟩,
  ⟨"Lettuce", 97/100, 1, 6⟩,
  ⟨"Olives", 210/100, 1, 8⟩,
  ⟨"Cheese", 270/100, 1, 7⟩
]

/-- The number of people to serve -/
def numPeople : ℕ := 8

/-- The theorem stating the minimum total cost and cost per serving -/
theorem minimum_dinner_cost :
  let totalCost := (ingredients.map (ingredientCost · numPeople)).sum
  totalCost = 2972/100 ∧ totalCost / (numPeople : ℚ) = 3715/1000 := by
  sorry


end NUMINAMATH_CALUDE_minimum_dinner_cost_l3079_307990


namespace NUMINAMATH_CALUDE_borrowed_amount_correct_l3079_307944

/-- The amount of money borrowed, in Rupees -/
def borrowed_amount : ℝ := 5000

/-- The interest rate for borrowing, as a decimal -/
def borrow_rate : ℝ := 0.04

/-- The interest rate for lending, as a decimal -/
def lend_rate : ℝ := 0.07

/-- The duration of the loan in years -/
def duration : ℝ := 2

/-- The yearly gain from the transaction, in Rupees -/
def yearly_gain : ℝ := 150

/-- Theorem stating that the borrowed amount is correct given the conditions -/
theorem borrowed_amount_correct :
  borrowed_amount * borrow_rate * duration = 
  borrowed_amount * lend_rate * duration - yearly_gain * duration := by
  sorry

#check borrowed_amount_correct

end NUMINAMATH_CALUDE_borrowed_amount_correct_l3079_307944


namespace NUMINAMATH_CALUDE_PQ_length_is_correct_l3079_307960

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (7, 8, 9)

-- Define the altitude AH
def altitude (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

-- Define the angle bisectors BD and CE
def angle_bisector_BD (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry
def angle_bisector_CE (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

-- Define the intersection points P and Q
def P (t : Triangle) : ℝ × ℝ := sorry
def Q (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of PQ
def PQ_length (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem PQ_length_is_correct (t : Triangle) :
  PQ_length t = (8 / 15) * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_PQ_length_is_correct_l3079_307960


namespace NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l3079_307961

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram ABCD with point F -/
structure Parallelogram :=
  (A B C D F : Point)
  (isParallelogram : sorry) -- Condition that ABCD is a parallelogram
  (F_on_AD_extension : sorry) -- Condition that F is on the extension of AD

/-- Represents the intersection points E and G -/
structure Intersections (p : Parallelogram) :=
  (E : Point)
  (G : Point)
  (E_on_AC_BF : sorry) -- Condition that E is on both AC and BF
  (G_on_DC_BF : sorry) -- Condition that G is on both DC and BF

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- The main theorem -/
theorem parallelogram_intersection_theorem (p : Parallelogram) (i : Intersections p) :
  distance i.E p.F = 40 → distance i.G p.F = 18 → distance p.B i.E = 20 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l3079_307961


namespace NUMINAMATH_CALUDE_complex_in_third_quadrant_l3079_307904

theorem complex_in_third_quadrant (z : ℂ) : z * (1 + Complex.I) = 1 - 2 * Complex.I → 
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_third_quadrant_l3079_307904


namespace NUMINAMATH_CALUDE_division_theorem_l3079_307902

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 132 →
  divisor = 16 →
  quotient = 8 →
  dividend = divisor * quotient + remainder →
  remainder = 4 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l3079_307902


namespace NUMINAMATH_CALUDE_josh_remaining_money_l3079_307989

def initial_amount : ℚ := 9
def first_expense : ℚ := 1.75
def second_expense : ℚ := 1.25

theorem josh_remaining_money :
  initial_amount - first_expense - second_expense = 6 := by sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l3079_307989


namespace NUMINAMATH_CALUDE_minutes_ratio_to_hour_l3079_307978

theorem minutes_ratio_to_hour (minutes_in_hour : ℕ) (ratio : ℚ) (result : ℕ) : 
  minutes_in_hour = 60 →
  ratio = 1/5 →
  result = minutes_in_hour * ratio →
  result = 12 := by sorry

end NUMINAMATH_CALUDE_minutes_ratio_to_hour_l3079_307978


namespace NUMINAMATH_CALUDE_no_integer_roots_l3079_307932

theorem no_integer_roots (a b : ℤ) : ¬ ∃ x : ℤ, 2 * a * b * x^4 - a^2 * x^2 - b^2 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3079_307932


namespace NUMINAMATH_CALUDE_coefficient_abc_in_expansion_coefficient_of_ab2c3_l3079_307923

theorem coefficient_abc_in_expansion : ℕ → Prop :=
  fun n => (1 + 1 + 1)^6 = n + sorry

theorem coefficient_of_ab2c3 : coefficient_abc_in_expansion 60 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_abc_in_expansion_coefficient_of_ab2c3_l3079_307923


namespace NUMINAMATH_CALUDE_brooke_earnings_l3079_307931

/-- Represents Brooke's milk and butter business -/
structure MilkBusiness where
  milk_price : ℝ
  butter_cost : ℝ
  milk_to_butter : ℝ
  butter_price : ℝ
  num_cows : ℕ
  milk_per_cow : ℝ
  num_customers : ℕ
  min_demand : ℝ
  max_demand : ℝ

/-- Calculates the total milk produced -/
def total_milk (b : MilkBusiness) : ℝ :=
  b.num_cows * b.milk_per_cow

/-- Calculates Brooke's earnings -/
def earnings (b : MilkBusiness) : ℝ :=
  total_milk b * b.milk_price

/-- Theorem stating that Brooke's earnings are $144 -/
theorem brooke_earnings :
  ∀ b : MilkBusiness,
    b.milk_price = 3 ∧
    b.butter_cost = 0.5 ∧
    b.milk_to_butter = 2 ∧
    b.butter_price = 1.5 ∧
    b.num_cows = 12 ∧
    b.milk_per_cow = 4 ∧
    b.num_customers = 6 ∧
    b.min_demand = 4 ∧
    b.max_demand = 8 →
    earnings b = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_brooke_earnings_l3079_307931


namespace NUMINAMATH_CALUDE_coordinates_of_q_l3079_307942

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle in 2D space -/
structure Triangle where
  p : Point2D
  q : Point2D
  r : Point2D

/-- Predicate for a right-angled triangle at Q -/
def isRightAngledAtQ (t : Triangle) : Prop :=
  -- Definition of right angle at Q (placeholder)
  True

/-- Predicate for a horizontal line segment -/
def isHorizontal (p1 p2 : Point2D) : Prop :=
  p1.y = p2.y

/-- Predicate for a vertical line segment -/
def isVertical (p1 p2 : Point2D) : Prop :=
  p1.x = p2.x

theorem coordinates_of_q (t : Triangle) :
  isRightAngledAtQ t →
  isHorizontal t.p t.q →
  isVertical t.q t.r →
  t.p = Point2D.mk 1 1 →
  t.r = Point2D.mk 5 3 →
  t.q = Point2D.mk 5 1 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_q_l3079_307942


namespace NUMINAMATH_CALUDE_largest_712_triple_l3079_307985

/-- Converts a number from base 7 to base 10 --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 12 --/
def decimalToBase12 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 7-12 triple --/
def is712Triple (n : ℕ) : Prop :=
  decimalToBase12 n = 3 * base7ToDecimal n

/-- The largest 7-12 triple --/
def largestTriple : ℕ := 450

theorem largest_712_triple :
  is712Triple largestTriple ∧
  ∀ n : ℕ, n > largestTriple → ¬is712Triple n := by sorry

end NUMINAMATH_CALUDE_largest_712_triple_l3079_307985


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3079_307980

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |m - (7^3 + 9^3)^(1/3)| ≥ |n - (7^3 + 9^3)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3079_307980


namespace NUMINAMATH_CALUDE_larry_basketball_shots_l3079_307987

theorem larry_basketball_shots 
  (initial_shots : ℕ) 
  (initial_success_rate : ℚ) 
  (additional_shots : ℕ) 
  (new_success_rate : ℚ) 
  (h1 : initial_shots = 30)
  (h2 : initial_success_rate = 3/5)
  (h3 : additional_shots = 10)
  (h4 : new_success_rate = 13/20) :
  (new_success_rate * (initial_shots + additional_shots) - initial_success_rate * initial_shots : ℚ) = 8 := by
sorry

end NUMINAMATH_CALUDE_larry_basketball_shots_l3079_307987


namespace NUMINAMATH_CALUDE_triangle_area_l3079_307922

/-- Given a triangle ABC with sides a, b, c and circumradius R, 
    prove that its area is 2√3 / 3 under specific conditions -/
theorem triangle_area (a b c R : ℝ) (h1 : (a^2 - c^2) / (2*R) = (a - b) * Real.sin b)
                                    (h2 : Real.sin b = 2 * Real.sin a)
                                    (h3 : c = 2) :
  (1/2) * a * b * Real.sin ((1/3) * Real.pi) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3079_307922


namespace NUMINAMATH_CALUDE_thousand_worries_conforms_to_cognitive_movement_l3079_307914

-- Define cognitive movement
structure CognitiveMovement where
  repetitive : Bool
  infinite : Bool

-- Define a phrase
structure Phrase where
  text : String
  conformsToCognitiveMovement : Bool

-- Define the specific phrase
def thousandWorries : Phrase where
  text := "A thousand worries yield one insight"
  conformsToCognitiveMovement := true -- This is what we want to prove

-- Theorem statement
theorem thousand_worries_conforms_to_cognitive_movement 
  (cm : CognitiveMovement) 
  (h1 : cm.repetitive = true) 
  (h2 : cm.infinite = true) : 
  thousandWorries.conformsToCognitiveMovement = true := by
  sorry


end NUMINAMATH_CALUDE_thousand_worries_conforms_to_cognitive_movement_l3079_307914


namespace NUMINAMATH_CALUDE_sales_third_month_l3079_307981

def sales_problem (m1 m2 m4 m5 m6 avg : ℚ) : ℚ :=
  let total_sales := avg * 6
  let known_sales := m1 + m2 + m4 + m5 + m6
  total_sales - known_sales

theorem sales_third_month
  (m1 m2 m4 m5 m6 avg : ℚ)
  (h_avg : avg = 6600)
  (h_m1 : m1 = 6435)
  (h_m2 : m2 = 6927)
  (h_m4 : m4 = 7230)
  (h_m5 : m5 = 6562)
  (h_m6 : m6 = 5591) :
  sales_problem m1 m2 m4 m5 m6 avg = 14085 := by
  sorry

#eval sales_problem 6435 6927 7230 6562 5591 6600

end NUMINAMATH_CALUDE_sales_third_month_l3079_307981


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l3079_307994

/-- A circle is tangent to a line if and only if the distance from the center of the circle to the line is equal to the radius of the circle. -/
theorem circle_tangent_to_line (b : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + 2*x + y^2 - 4*y + 3 = 0}
  let line := {(x, y) : ℝ × ℝ | x + y + b = 0}
  let center := (-1, 2)
  let radius := Real.sqrt 2
  (∀ p ∈ circle, p ∈ line → (∀ q ∈ circle, q = p ∨ q ∉ line)) → 
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l3079_307994


namespace NUMINAMATH_CALUDE_triangle_area_comparison_l3079_307959

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

theorem triangle_area_comparison : 
  triangleArea 30 30 45 > triangleArea 30 30 55 := by sorry

end NUMINAMATH_CALUDE_triangle_area_comparison_l3079_307959


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l3079_307967

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - |x + a|

-- State the theorem
theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l3079_307967


namespace NUMINAMATH_CALUDE_product_of_sums_l3079_307969

theorem product_of_sums (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) :
  (x + 2) * (y + 2) = 16 := by sorry

end NUMINAMATH_CALUDE_product_of_sums_l3079_307969


namespace NUMINAMATH_CALUDE_general_laborer_pay_general_laborer_pay_is_90_l3079_307935

/-- The daily pay for general laborers given the following conditions:
  - There are 35 people hired in total
  - The total payroll is 3950 dollars
  - 19 of the hired people are general laborers
  - Heavy equipment operators are paid 140 dollars per day
-/
theorem general_laborer_pay (total_hired : ℕ) (total_payroll : ℕ) 
  (num_laborers : ℕ) (operator_pay : ℕ) : ℕ :=
  let num_operators := total_hired - num_laborers
  let operator_total_pay := num_operators * operator_pay
  let laborer_total_pay := total_payroll - operator_total_pay
  laborer_total_pay / num_laborers

/-- Proof that the daily pay for general laborers is 90 dollars -/
theorem general_laborer_pay_is_90 : 
  general_laborer_pay 35 3950 19 140 = 90 := by
  sorry

end NUMINAMATH_CALUDE_general_laborer_pay_general_laborer_pay_is_90_l3079_307935


namespace NUMINAMATH_CALUDE_maggie_earnings_l3079_307907

/-- The amount Maggie earns for each magazine subscription she sells -/
def earnings_per_subscription : ℝ := 5

/-- The number of subscriptions Maggie sold to her parents -/
def parents_subscriptions : ℕ := 4

/-- The number of subscriptions Maggie sold to her grandfather -/
def grandfather_subscriptions : ℕ := 1

/-- The number of subscriptions Maggie sold to the next-door neighbor -/
def neighbor_subscriptions : ℕ := 2

/-- The total amount Maggie earned from all subscriptions -/
def total_earnings : ℝ := 55

theorem maggie_earnings :
  earnings_per_subscription * (parents_subscriptions + grandfather_subscriptions + 
  neighbor_subscriptions + 2 * neighbor_subscriptions) = total_earnings :=
sorry

end NUMINAMATH_CALUDE_maggie_earnings_l3079_307907


namespace NUMINAMATH_CALUDE_find_number_l3079_307924

theorem find_number (x : ℝ) (h : 0.46 * x = 165.6) : x = 360 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3079_307924


namespace NUMINAMATH_CALUDE_basketball_scores_l3079_307920

def first_ten_games : List Nat := [9, 5, 7, 4, 8, 6, 2, 3, 5, 6]

theorem basketball_scores (game_11 game_12 : Nat) : 
  game_11 < 10 →
  game_12 < 10 →
  (List.sum first_ten_games + game_11) % 11 = 0 →
  (List.sum first_ten_games + game_11 + game_12) % 12 = 0 →
  game_11 * game_12 = 0 := by
sorry

end NUMINAMATH_CALUDE_basketball_scores_l3079_307920


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l3079_307992

theorem power_of_power_at_three : (3^(3^2))^(3^3) = 3^243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l3079_307992


namespace NUMINAMATH_CALUDE_mean_home_runs_l3079_307926

def total_players : ℕ := 12
def players_with_5 : ℕ := 3
def players_with_7 : ℕ := 5
def players_with_9 : ℕ := 3
def players_with_11 : ℕ := 1

def total_home_runs : ℕ := 
  5 * players_with_5 + 7 * players_with_7 + 9 * players_with_9 + 11 * players_with_11

theorem mean_home_runs : 
  (total_home_runs : ℚ) / total_players = 88 / 12 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l3079_307926


namespace NUMINAMATH_CALUDE_ratio_calculation_l3079_307975

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (5 * A + 3 * B) / (3 * C - 2 * A) = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l3079_307975


namespace NUMINAMATH_CALUDE_min_value_of_z_l3079_307995

/-- The objective function to be minimized -/
def z (x y : ℝ) : ℝ := y - 2 * x

/-- The feasible region defined by the given constraints -/
def feasible_region (x y : ℝ) : Prop :=
  3 * x + y - 6 ≥ 0 ∧ x - y - 2 ≤ 0 ∧ y - 3 ≤ 0

/-- Theorem stating that the minimum value of z in the feasible region is -7 -/
theorem min_value_of_z :
  ∃ (x y : ℝ), feasible_region x y ∧
  ∀ (x' y' : ℝ), feasible_region x' y' → z x' y' ≥ z x y ∧
  z x y = -7 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l3079_307995


namespace NUMINAMATH_CALUDE_shorter_side_length_l3079_307908

theorem shorter_side_length (a b : ℕ) : 
  a > b →                 -- Ensure a is the longer side
  2 * a + 2 * b = 48 →    -- Perimeter condition
  a * b = 140 →           -- Area condition
  b = 10 := by            -- Conclusion: shorter side is 10 feet
sorry

end NUMINAMATH_CALUDE_shorter_side_length_l3079_307908


namespace NUMINAMATH_CALUDE_unique_solution_natural_numbers_l3079_307965

theorem unique_solution_natural_numbers : 
  ∃! (a b : ℕ), a^b + a + b = b^a ∧ a = 5 ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_natural_numbers_l3079_307965


namespace NUMINAMATH_CALUDE_chemistry_physics_difference_l3079_307910

/-- Proves that the difference between chemistry and physics scores is 10 -/
theorem chemistry_physics_difference
  (M P C : ℕ)  -- Marks in Mathematics, Physics, and Chemistry
  (h1 : M + P = 60)  -- Sum of Mathematics and Physics scores
  (h2 : (M + C) / 2 = 35)  -- Average of Mathematics and Chemistry scores
  (h3 : C > P)  -- Chemistry score is higher than Physics score
  : C - P = 10 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_physics_difference_l3079_307910


namespace NUMINAMATH_CALUDE_quadratic_condition_l3079_307949

def is_quadratic (m : ℝ) : Prop :=
  (|m| = 2) ∧ (m - 2 ≠ 0)

theorem quadratic_condition (m : ℝ) :
  is_quadratic m ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l3079_307949


namespace NUMINAMATH_CALUDE_product_19_reciprocal_squares_sum_l3079_307996

theorem product_19_reciprocal_squares_sum :
  ∀ a b : ℕ+, 
  (a * b : ℕ+) = 19 → 
  (1 : ℚ) / (a * a : ℚ) + (1 : ℚ) / (b * b : ℚ) = 362 / 361 := by
  sorry

end NUMINAMATH_CALUDE_product_19_reciprocal_squares_sum_l3079_307996


namespace NUMINAMATH_CALUDE_knights_and_liars_l3079_307937

/-- Represents the two types of inhabitants in the country -/
inductive Inhabitant
  | Knight
  | Liar

/-- The statement made by A -/
def statement (a b : Inhabitant) : Prop :=
  a = Inhabitant.Liar ∧ b ≠ Inhabitant.Liar

/-- A function that determines if a given statement is true based on the speaker's type -/
def isTrueStatement (speaker : Inhabitant) (stmt : Prop) : Prop :=
  (speaker = Inhabitant.Knight ∧ stmt) ∨ (speaker = Inhabitant.Liar ∧ ¬stmt)

theorem knights_and_liars (a b : Inhabitant) :
  isTrueStatement a (statement a b) →
  a = Inhabitant.Liar ∧ b = Inhabitant.Liar :=
by sorry

end NUMINAMATH_CALUDE_knights_and_liars_l3079_307937


namespace NUMINAMATH_CALUDE_investment_total_calculation_l3079_307976

/-- Represents an investment split between two interest rates -/
structure Investment where
  total : ℝ
  rate1 : ℝ
  rate2 : ℝ
  amount1 : ℝ

/-- Calculates the total interest earned from an investment -/
def totalInterest (inv : Investment) : ℝ :=
  inv.rate1 * inv.amount1 + inv.rate2 * (inv.total - inv.amount1)

theorem investment_total_calculation (inv : Investment) 
  (h1 : inv.rate1 = 0.07)
  (h2 : inv.rate2 = 0.09)
  (h3 : inv.amount1 = 5500)
  (h4 : totalInterest inv = 970) :
  inv.total = 12000 := by
sorry

end NUMINAMATH_CALUDE_investment_total_calculation_l3079_307976


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3079_307936

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 15⁻¹) :
  (x : ℕ) + y ≥ 64 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3079_307936


namespace NUMINAMATH_CALUDE_room_width_calculation_l3079_307998

/-- Given a rectangular room with known length, flooring cost per square meter, 
    and total flooring cost, calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  cost_per_sqm = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_sqm) / length = 3.75 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l3079_307998


namespace NUMINAMATH_CALUDE_tomato_production_l3079_307968

/-- The number of tomatoes produced by the first plant -/
def plant1_tomatoes : ℕ := 24

/-- The number of tomatoes produced by the second plant -/
def plant2_tomatoes : ℕ := plant1_tomatoes / 2 + 5

/-- The number of tomatoes produced by the third plant -/
def plant3_tomatoes : ℕ := plant2_tomatoes + 2

/-- The total number of tomatoes produced by all three plants -/
def total_tomatoes : ℕ := plant1_tomatoes + plant2_tomatoes + plant3_tomatoes

theorem tomato_production : total_tomatoes = 60 := by
  sorry

end NUMINAMATH_CALUDE_tomato_production_l3079_307968


namespace NUMINAMATH_CALUDE_cube_of_negative_l3079_307929

theorem cube_of_negative (x : ℝ) (h : x^3 = 32.768) : (-x)^3 = -32.768 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_l3079_307929


namespace NUMINAMATH_CALUDE_max_discussions_left_l3079_307951

/-- Represents a group of politicians at a summit --/
structure PoliticianGroup where
  size : Nat
  has_talked : Fin size → Fin size → Bool
  all_pairs_plan_to_talk : ∀ i j, i ≠ j → has_talked i j = false → True
  four_politician_condition : ∀ a b c d, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    (has_talked a b ∧ has_talked a c ∧ has_talked a d) ∨
    (has_talked b a ∧ has_talked b c ∧ has_talked b d) ∨
    (has_talked c a ∧ has_talked c b ∧ has_talked c d) ∨
    (has_talked d a ∧ has_talked d b ∧ has_talked d c)

/-- The theorem stating the maximum number of discussions yet to be held --/
theorem max_discussions_left (g : PoliticianGroup) (h : g.size = 2018) :
  (∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ¬g.has_talked a b ∧ ¬g.has_talked b c ∧ ¬g.has_talked a c) ∧
  (∀ a b c d, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    g.has_talked a b ∨ g.has_talked b c ∨ g.has_talked a c ∨
    g.has_talked a d ∨ g.has_talked b d ∨ g.has_talked c d) :=
by sorry

end NUMINAMATH_CALUDE_max_discussions_left_l3079_307951


namespace NUMINAMATH_CALUDE_vacation_days_l3079_307943

theorem vacation_days (rainy_days clear_mornings clear_afternoons : ℕ) 
  (h1 : rainy_days = 13)
  (h2 : clear_mornings = 11)
  (h3 : clear_afternoons = 12)
  (h4 : rainy_days = clear_mornings + clear_afternoons) :
  clear_mornings + clear_afternoons = 23 := by
  sorry

end NUMINAMATH_CALUDE_vacation_days_l3079_307943


namespace NUMINAMATH_CALUDE_pasta_cost_is_one_dollar_l3079_307948

/-- The cost of pasta per box for Sam's spaghetti and meatballs dinner -/
def pasta_cost (total_cost sauce_cost meatballs_cost : ℚ) : ℚ :=
  total_cost - (sauce_cost + meatballs_cost)

/-- Theorem: The cost of pasta per box is $1.00 -/
theorem pasta_cost_is_one_dollar :
  pasta_cost 8 2 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_pasta_cost_is_one_dollar_l3079_307948


namespace NUMINAMATH_CALUDE_sandi_spending_ratio_l3079_307916

/-- Proves that the ratio of Sandi's spending to her initial amount is 1:2 --/
theorem sandi_spending_ratio :
  ∀ (sandi_initial sandi_spent gillian_spent : ℚ),
  sandi_initial = 600 →
  gillian_spent = 3 * sandi_spent + 150 →
  gillian_spent = 1050 →
  sandi_spent / sandi_initial = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sandi_spending_ratio_l3079_307916


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3079_307938

theorem other_root_of_quadratic (a c : ℝ) (h : a ≠ 0) :
  (∃ x, 4 * a * x^2 - 2 * a * x + c = 0 ∧ x = 0) →
  (∃ y, 4 * a * y^2 - 2 * a * y + c = 0 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3079_307938


namespace NUMINAMATH_CALUDE_sum_in_base5_l3079_307974

/-- Represents a number in base 5 --/
def Base5 : Type := ℕ

/-- Converts a base 5 number to its decimal representation --/
def to_decimal (n : Base5) : ℕ := sorry

/-- Converts a decimal number to its base 5 representation --/
def to_base5 (n : ℕ) : Base5 := sorry

/-- Addition operation for base 5 numbers --/
def base5_add (a b : Base5) : Base5 := to_base5 (to_decimal a + to_decimal b)

theorem sum_in_base5 :
  let a : Base5 := to_base5 231
  let b : Base5 := to_base5 414
  let c : Base5 := to_base5 123
  let result : Base5 := to_base5 1323
  base5_add (base5_add a b) c = result := by sorry

end NUMINAMATH_CALUDE_sum_in_base5_l3079_307974


namespace NUMINAMATH_CALUDE_unique_composite_with_sum_power_of_two_l3079_307915

theorem unique_composite_with_sum_power_of_two :
  ∃! m : ℕ+, 
    (1 < m) ∧ 
    (∀ a b : ℕ+, a * b = m → ∃ k : ℕ, a + b = 2^k) ∧
    m = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_composite_with_sum_power_of_two_l3079_307915


namespace NUMINAMATH_CALUDE_equation_solution_l3079_307979

theorem equation_solution (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  (x + y) / x = 2 * y / (x + y) + 1 ↔ x = y ∨ x = -3 * y :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3079_307979


namespace NUMINAMATH_CALUDE_linear_function_increasing_condition_l3079_307925

/-- A linear function y = (2m-1)x + 1 -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (2*m - 1)*x + 1

theorem linear_function_increasing_condition 
  (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁ < x₂) 
  (h2 : y₁ < y₂)
  (h3 : y₁ = linear_function m x₁)
  (h4 : y₂ = linear_function m x₂) :
  m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_increasing_condition_l3079_307925


namespace NUMINAMATH_CALUDE_distance_on_number_line_l3079_307958

theorem distance_on_number_line (A B C : ℝ) : 
  (|B - A| = 5) → (|C - B| = 3) → (|C - A| = 2 ∨ |C - A| = 8) :=
by sorry

end NUMINAMATH_CALUDE_distance_on_number_line_l3079_307958


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3079_307901

-- Define the quadratic function
variable (f : ℝ → ℝ)

-- Define the interval [a, b]
variable (a b : ℝ)

-- Define the axis of symmetry
def axis_of_symmetry (f : ℝ → ℝ) : Prop := ∀ x, f x = f x

-- Define the range condition
def range_condition (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ y ∈ Set.Icc (f b) (f a), ∃ x ∈ Set.Icc a b, f x = y

-- Theorem statement
theorem quadratic_function_property
  (h_axis : axis_of_symmetry f)
  (h_range : range_condition f a b) :
  ∀ x, x ∉ Set.Ioo a b :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3079_307901


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l3079_307909

/-- Represents a rhombus with given area and one diagonal -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ

/-- Theorem: In a rhombus with area 60 cm² and one diagonal 12 cm, the other diagonal is 10 cm -/
theorem rhombus_other_diagonal (r : Rhombus) (h1 : r.area = 60) (h2 : r.diagonal1 = 12) :
  ∃ (diagonal2 : ℝ), diagonal2 = 10 ∧ r.area = (r.diagonal1 * diagonal2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l3079_307909


namespace NUMINAMATH_CALUDE_prism_square_intersection_angle_l3079_307962

theorem prism_square_intersection_angle (d : ℝ) (h : d > 0) : 
  let rhombus_acute_angle : ℝ := 60 * π / 180
  let rhombus_diagonal : ℝ := d * Real.sqrt 3
  let intersection_angle : ℝ := Real.arccos (Real.sqrt 3 / 3)
  intersection_angle = Real.arccos (d / rhombus_diagonal) :=
by sorry

end NUMINAMATH_CALUDE_prism_square_intersection_angle_l3079_307962


namespace NUMINAMATH_CALUDE_equivalent_form_l3079_307941

theorem equivalent_form :
  (2 + 5) * (2^2 + 5^2) * (2^4 + 5^4) * (2^8 + 5^8) * 
  (2^16 + 5^16) * (2^32 + 5^32) * (2^64 + 5^64) = 5^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_form_l3079_307941


namespace NUMINAMATH_CALUDE_square_of_binomial_b_value_l3079_307940

/-- If 9x^2 + 27x + b is the square of a binomial, then b = 81/4 -/
theorem square_of_binomial_b_value (b : ℝ) : 
  (∃ (c : ℝ), ∀ (x : ℝ), 9*x^2 + 27*x + b = (3*x + c)^2) → 
  b = 81/4 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_b_value_l3079_307940


namespace NUMINAMATH_CALUDE_initial_stock_proof_l3079_307903

/-- The number of coloring books initially in stock at a store -/
def initial_stock : ℕ := 86

/-- The number of coloring books sold -/
def books_sold : ℕ := 37

/-- The number of shelves used for remaining books -/
def shelves_used : ℕ := 7

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 7

/-- Theorem stating that the initial stock equals 86 -/
theorem initial_stock_proof : 
  initial_stock = books_sold + (shelves_used * books_per_shelf) :=
by sorry

end NUMINAMATH_CALUDE_initial_stock_proof_l3079_307903


namespace NUMINAMATH_CALUDE_distance_problems_l3079_307917

def distance_point_to_line (p : Fin n → ℝ) (a b : Fin n → ℝ) : ℝ :=
  sorry

theorem distance_problems :
  let d1 := distance_point_to_line (![1, 0]) (![0, 0]) (![0, 1])
  let d2 := distance_point_to_line (![1, 0]) (![0, 0]) (![1, 1])
  let d3 := distance_point_to_line (![1, 0, 0]) (![0, 0, 0]) (![1, 1, 1])
  (d1 = 1) ∧ (d2 = Real.sqrt 2 / 2) ∧ (d3 = Real.sqrt 6 / 3) := by
  sorry

end NUMINAMATH_CALUDE_distance_problems_l3079_307917


namespace NUMINAMATH_CALUDE_point_m_location_l3079_307911

theorem point_m_location (L P M : ℚ) : 
  L = 1/6 → P = 1/12 → M - L = (P - L) / 3 → M = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_point_m_location_l3079_307911


namespace NUMINAMATH_CALUDE_tv_sale_effect_l3079_307997

-- Define the price reduction percentage
def price_reduction : ℝ := 0.18

-- Define the sales increase percentage
def sales_increase : ℝ := 0.88

-- Define the net effect on sale value
def net_effect : ℝ := 0.5416

-- Theorem statement
theorem tv_sale_effect :
  let new_price_factor := 1 - price_reduction
  let new_sales_factor := 1 + sales_increase
  (new_price_factor * new_sales_factor - 1) = net_effect := by sorry

end NUMINAMATH_CALUDE_tv_sale_effect_l3079_307997


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l3079_307912

/-- The area of a square with one side on y = 10 and endpoints on y = x^2 + 4x + 3 is 44 -/
theorem square_area_on_parabola : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 + 4*x₁ + 3 = 10) ∧ 
    (x₂^2 + 4*x₂ + 3 = 10) ∧ 
    (x₁ ≠ x₂) ∧
    ((x₂ - x₁)^2 = 44) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l3079_307912


namespace NUMINAMATH_CALUDE_definite_integral_x_squared_l3079_307913

theorem definite_integral_x_squared : ∫ x in (0 : ℝ)..1, x^2 = 1/3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_x_squared_l3079_307913


namespace NUMINAMATH_CALUDE_complex_to_exponential_form_l3079_307970

theorem complex_to_exponential_form (z : ℂ) :
  z = 2 - I →
  Real.arctan (1 / 2) = Real.arctan (Complex.abs z / Complex.im z) :=
by sorry

end NUMINAMATH_CALUDE_complex_to_exponential_form_l3079_307970


namespace NUMINAMATH_CALUDE_jungkook_points_l3079_307971

/-- Calculates the total points earned by Jungkook in a math test. -/
theorem jungkook_points (total_problems : ℕ) (correct_two_point : ℕ) (correct_one_point : ℕ) 
  (h1 : total_problems = 15)
  (h2 : correct_two_point = 8)
  (h3 : correct_one_point = 2) :
  correct_two_point * 2 + correct_one_point = 18 := by
  sorry

#check jungkook_points

end NUMINAMATH_CALUDE_jungkook_points_l3079_307971


namespace NUMINAMATH_CALUDE_exists_removable_piece_l3079_307991

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  pieces : Finset (Fin 8 × Fin 8)
  piece_count : pieces.card = 15
  row_coverage : ∀ r : Fin 8, ∃ c : Fin 8, (r, c) ∈ pieces
  col_coverage : ∀ c : Fin 8, ∃ r : Fin 8, (r, c) ∈ pieces

/-- Theorem stating that there always exists a removable piece -/
theorem exists_removable_piece (config : ChessboardConfig) :
  ∃ p ∈ config.pieces, 
    let remaining := config.pieces.erase p
    (∀ r : Fin 8, ∃ c : Fin 8, (r, c) ∈ remaining) ∧
    (∀ c : Fin 8, ∃ r : Fin 8, (r, c) ∈ remaining) :=
  sorry

end NUMINAMATH_CALUDE_exists_removable_piece_l3079_307991


namespace NUMINAMATH_CALUDE_election_winner_votes_l3079_307956

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (65 : ℚ) / 100 * total_votes - (35 : ℚ) / 100 * total_votes = 300) : 
  (65 : ℚ) / 100 * total_votes = 650 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3079_307956


namespace NUMINAMATH_CALUDE_unique_quadruple_l3079_307950

theorem unique_quadruple :
  ∃! (a b c d : ℝ), 
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a + b + c + d = 2 ∧
    a^2 + b^2 + c^2 + d^2 = 3 ∧
    (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 18 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadruple_l3079_307950


namespace NUMINAMATH_CALUDE_y_plus_z_value_l3079_307964

theorem y_plus_z_value (x y z : ℕ) (hx : x = 4) (hy : y = 3 * x) (hz : z = 2 * y) : 
  y + z = 36 := by
  sorry

end NUMINAMATH_CALUDE_y_plus_z_value_l3079_307964


namespace NUMINAMATH_CALUDE_binomial_five_one_l3079_307988

theorem binomial_five_one : (5 : ℕ).choose 1 = 5 := by sorry

end NUMINAMATH_CALUDE_binomial_five_one_l3079_307988


namespace NUMINAMATH_CALUDE_fraction_reciprocal_difference_l3079_307966

theorem fraction_reciprocal_difference (x : ℚ) : 
  0 < x → x < 1 → (1 / x - x = 9 / 20) → x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_reciprocal_difference_l3079_307966


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3079_307977

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3079_307977


namespace NUMINAMATH_CALUDE_complex_argument_and_reality_l3079_307945

noncomputable def arg (z : ℂ) : ℝ := Real.arctan (z.im / z.re)

theorem complex_argument_and_reality (θ : ℝ) (a : ℝ) :
  0 < θ ∧ θ < 2 * Real.pi →
  let z : ℂ := 1 - Real.cos θ + Complex.I * Real.sin θ
  let u : ℂ := a^2 + Complex.I * a
  (z * u).re = 0 →
  (
    (0 < θ ∧ θ < Real.pi → arg u = θ / 2) ∧
    (Real.pi < θ ∧ θ < 2 * Real.pi → arg u = Real.pi + θ / 2)
  ) ∧
  ∀ ω : ℂ, ω = z^2 + u^2 + 2 * z * u → ω.im ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_argument_and_reality_l3079_307945


namespace NUMINAMATH_CALUDE_point_on_line_l3079_307953

theorem point_on_line (m n k : ℝ) : 
  (m = 2 * n + 5) ∧ (m + 4 = 2 * (n + k) + 5) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l3079_307953


namespace NUMINAMATH_CALUDE_average_age_combined_l3079_307993

theorem average_age_combined (num_students : Nat) (num_parents : Nat) 
  (avg_age_students : ℝ) (avg_age_parents : ℝ) :
  num_students = 45 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 40 →
  (num_students * avg_age_students + num_parents * avg_age_parents) / (num_students + num_parents : ℝ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l3079_307993


namespace NUMINAMATH_CALUDE_smallest_value_in_range_l3079_307905

theorem smallest_value_in_range (x : ℝ) (h : 0 < x ∧ x < 2) :
  x^2 ≤ min x (min (3*x) (min (Real.sqrt x) (1/x))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_in_range_l3079_307905


namespace NUMINAMATH_CALUDE_even_function_property_l3079_307972

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_positive : ∀ x > 0, f x = 10^x) : 
  ∀ x < 0, f x = (1/10)^x := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l3079_307972


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l3079_307983

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 10) → cows = 5 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l3079_307983


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_l3079_307952

theorem smallest_n_for_candy (n : ℕ+) : 
  (∃ m : ℕ+, 16 ∣ m ∧ 18 ∣ m ∧ 20 ∣ m ∧ m = 30 * n) →
  (∀ k : ℕ+, k < n → ¬∃ m : ℕ+, 16 ∣ m ∧ 18 ∣ m ∧ 20 ∣ m ∧ m = 30 * k) →
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_l3079_307952


namespace NUMINAMATH_CALUDE_k_h_negative_three_equals_sixteen_l3079_307933

-- Define the function h
def h (x : ℝ) : ℝ := 4 * x^2 - 8

-- Define a variable k as a function from ℝ to ℝ
variable (k : ℝ → ℝ)

-- State the theorem
theorem k_h_negative_three_equals_sixteen 
  (h_def : ∀ x, h x = 4 * x^2 - 8)
  (k_h_three : k (h 3) = 16) :
  k (h (-3)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_k_h_negative_three_equals_sixteen_l3079_307933


namespace NUMINAMATH_CALUDE_hexacontagon_triangles_l3079_307986

/-- The number of sides in a regular hexacontagon -/
def n : ℕ := 60

/-- The number of triangles that can be formed using the vertices of a regular hexacontagon,
    without using any three consecutive vertices -/
def num_triangles : ℕ := Nat.choose n 3 - n

theorem hexacontagon_triangles : num_triangles = 34160 := by
  sorry

end NUMINAMATH_CALUDE_hexacontagon_triangles_l3079_307986


namespace NUMINAMATH_CALUDE_unique_n_for_integer_Sn_l3079_307954

theorem unique_n_for_integer_Sn : ∃! (n : ℕ+), ∃ (m : ℕ), 
  n.val > 0 ∧ m^2 = 17^2 + n.val^2 ∧ 
  ∀ (k : ℕ+), k ≠ n → ¬∃ (l : ℕ), l^2 = 17^2 + k.val^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_for_integer_Sn_l3079_307954


namespace NUMINAMATH_CALUDE_gift_cost_equation_l3079_307963

/-- Represents the cost equation for Xiaofen's gift purchase -/
theorem gift_cost_equation (x : ℝ) : 
  (15 : ℝ) * (x + 2 * 20) = 900 ↔ 
  (∃ (total_cost num_gifts num_lollipops_per_gift lollipop_cost : ℝ),
    total_cost = 900 ∧
    num_gifts = 15 ∧
    num_lollipops_per_gift = 2 ∧
    lollipop_cost = 20 ∧
    total_cost = num_gifts * (x + num_lollipops_per_gift * lollipop_cost)) :=
by sorry

end NUMINAMATH_CALUDE_gift_cost_equation_l3079_307963


namespace NUMINAMATH_CALUDE_box_height_l3079_307930

/-- Given a rectangular box with width 10 inches, length 20 inches, and height h inches,
    if the area of the triangle formed by the center points of three faces meeting at a corner
    is 40 square inches, then h = (24 * sqrt(21)) / 5 inches. -/
theorem box_height (h : ℝ) : 
  let width : ℝ := 10
  let length : ℝ := 20
  let triangle_area : ℝ := 40
  let diagonal := Real.sqrt (width ^ 2 + length ^ 2)
  let side1 := Real.sqrt (width ^ 2 + (h / 2) ^ 2)
  let side2 := Real.sqrt (length ^ 2 + (h / 2) ^ 2)
  triangle_area = Real.sqrt (
    (diagonal + side1 + side2) *
    (diagonal + side1 - side2) *
    (diagonal - side1 + side2) *
    (-diagonal + side1 + side2)
  ) / 4
  →
  h = 24 * Real.sqrt 21 / 5 := by
sorry

end NUMINAMATH_CALUDE_box_height_l3079_307930


namespace NUMINAMATH_CALUDE_alice_grading_papers_l3079_307928

/-- Given that Ms. Alice can grade 296 papers in 8 hours, prove that she can grade 407 papers in 11 hours. -/
theorem alice_grading_papers : 
  let papers_in_8_hours : ℕ := 296
  let hours_initial : ℕ := 8
  let hours_new : ℕ := 11
  let papers_in_11_hours : ℕ := 407
  (papers_in_8_hours : ℚ) / hours_initial * hours_new = papers_in_11_hours :=
by sorry

end NUMINAMATH_CALUDE_alice_grading_papers_l3079_307928


namespace NUMINAMATH_CALUDE_operation_probability_l3079_307984

/-- An operation that randomly changes a positive integer to a smaller nonnegative integer -/
def operation (n : ℕ+) : ℕ := sorry

/-- The probability of choosing any specific smaller number during the operation -/
def transition_prob (n k : ℕ) : ℝ := sorry

/-- The probability of encountering specific numbers during the operation process -/
def encounter_prob (start : ℕ+) (targets : List ℕ) : ℝ := sorry

theorem operation_probability :
  encounter_prob 2019 [10, 100, 1000] = 1 / 2019000000 := by sorry

end NUMINAMATH_CALUDE_operation_probability_l3079_307984


namespace NUMINAMATH_CALUDE_point_coordinates_l3079_307946

/-- Given a point P with coordinates (2m+4, m-1), prove that P has coordinates (-6, -6) 
    under the condition that it lies on the y-axis or its distance from the y-axis is 6, 
    and it lies in the third quadrant and is equidistant from both coordinate axes. -/
theorem point_coordinates (m : ℝ) : 
  (((2*m + 4 = 0) ∨ (|2*m + 4| = 6)) ∧ 
   (2*m + 4 < 0) ∧ (m - 1 < 0) ∧ 
   (|2*m + 4| = |m - 1|)) → 
  (2*m + 4 = -6 ∧ m - 1 = -6) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3079_307946


namespace NUMINAMATH_CALUDE_bench_placement_l3079_307900

theorem bench_placement (path_length : ℕ) (interval : ℕ) (bench_count : ℕ) : 
  path_length = 120 ∧ 
  interval = 10 ∧ 
  bench_count = (path_length / interval) + 1 →
  bench_count = 13 := by
  sorry

end NUMINAMATH_CALUDE_bench_placement_l3079_307900


namespace NUMINAMATH_CALUDE_xy_length_is_30_l3079_307919

/-- A right triangle XYZ with specific angle and side length properties -/
structure RightTriangleXYZ where
  /-- The length of side XZ -/
  xz : ℝ
  /-- The measure of angle Y in radians -/
  angle_y : ℝ
  /-- XZ equals 15 -/
  xz_eq : xz = 15
  /-- Angle Y equals 30 degrees (π/6 radians) -/
  angle_y_eq : angle_y = π / 6
  /-- The triangle is a right triangle (angle X is 90 degrees) -/
  right_angle : True

/-- The length of side XY in the right triangle XYZ -/
def length_xy (t : RightTriangleXYZ) : ℝ := 2 * t.xz

/-- Theorem stating that the length of XY is 30 in the given right triangle -/
theorem xy_length_is_30 (t : RightTriangleXYZ) : length_xy t = 30 := by
  sorry

end NUMINAMATH_CALUDE_xy_length_is_30_l3079_307919


namespace NUMINAMATH_CALUDE_roots_reciprocal_sum_l3079_307947

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 2 * x₁ - 1 = 0) → 
  (2 * x₂^2 - 2 * x₂ - 1 = 0) → 
  (x₁ ≠ x₂) →
  (1 / x₁ + 1 / x₂ = -2) := by
  sorry

end NUMINAMATH_CALUDE_roots_reciprocal_sum_l3079_307947


namespace NUMINAMATH_CALUDE_f_composition_equals_one_third_l3079_307982

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 4
  else 3^x

-- State the theorem
theorem f_composition_equals_one_third :
  f (f (1/4)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_one_third_l3079_307982


namespace NUMINAMATH_CALUDE_cards_per_page_l3079_307927

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 16)
  (h3 : pages = 8) :
  (new_cards + old_cards) / pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_page_l3079_307927


namespace NUMINAMATH_CALUDE_roses_given_l3079_307973

/-- The number of students in the class -/
def total_students : ℕ := 28

/-- The number of different types of flowers -/
def flower_types : ℕ := 3

/-- The relationship between daffodils and roses -/
def rose_daffodil_ratio : ℕ := 4

/-- The relationship between tulips and roses -/
def tulip_rose_ratio : ℕ := 3

/-- The number of boys in the class -/
def num_boys : ℕ := 11

/-- The number of girls in the class -/
def num_girls : ℕ := 17

/-- The number of daffodils given -/
def num_daffodils : ℕ := 11

/-- The number of roses given -/
def num_roses : ℕ := 44

/-- The number of tulips given -/
def num_tulips : ℕ := 132

theorem roses_given :
  num_roses = 44 ∧
  total_students = num_boys + num_girls ∧
  num_roses = rose_daffodil_ratio * num_daffodils ∧
  num_tulips = tulip_rose_ratio * num_roses ∧
  num_boys * num_girls = num_daffodils + num_roses + num_tulips :=
by sorry

end NUMINAMATH_CALUDE_roses_given_l3079_307973


namespace NUMINAMATH_CALUDE_derivative_x_plus_one_squared_times_x_minus_one_l3079_307957

theorem derivative_x_plus_one_squared_times_x_minus_one (x : ℝ) :
  deriv (λ x => (x + 1)^2 * (x - 1)) x = 3*x^2 + 2*x - 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_plus_one_squared_times_x_minus_one_l3079_307957


namespace NUMINAMATH_CALUDE_roots_sum_product_l3079_307921

theorem roots_sum_product (a b : ℝ) : 
  (a^4 - 4*a - 1 = 0) → 
  (b^4 - 4*b - 1 = 0) → 
  (∀ x : ℝ, x ≠ a ∧ x ≠ b → x^4 - 4*x - 1 ≠ 0) →
  a * b + a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_product_l3079_307921


namespace NUMINAMATH_CALUDE_square_root_meaningful_l3079_307939

theorem square_root_meaningful (x : ℝ) : 
  x ≥ 5 → (x = 6 ∧ x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_square_root_meaningful_l3079_307939


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l3079_307934

theorem point_on_terminal_side (m : ℝ) (α : ℝ) :
  (2 : ℝ) / Real.sqrt (m^2 + 4) = (1 : ℝ) / 3 →
  m = 4 * Real.sqrt 2 ∨ m = -4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l3079_307934


namespace NUMINAMATH_CALUDE_ticket_cost_l3079_307918

/-- Given the total amount collected and average daily ticket sales over three days,
    prove that the cost of one ticket is $4. -/
theorem ticket_cost (total_amount : ℚ) (avg_daily_sales : ℚ) 
  (h1 : total_amount = 960)
  (h2 : avg_daily_sales = 80) : 
  total_amount / (avg_daily_sales * 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_l3079_307918


namespace NUMINAMATH_CALUDE_square_position_2010_l3079_307906

-- Define the possible positions of the square
inductive SquarePosition
  | ABCD
  | DABC
  | BDAC
  | ACBD
  | CABD
  | DCBA
  | CDAB
  | BADC
  | DBCA

def next_position (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.BDAC
  | SquarePosition.DABC => SquarePosition.BDAC
  | SquarePosition.BDAC => SquarePosition.ACBD
  | SquarePosition.ACBD => SquarePosition.CABD
  | SquarePosition.CABD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.DBCA
  | SquarePosition.DBCA => SquarePosition.ABCD

def nth_position (n : Nat) : SquarePosition :=
  match n with
  | 0 => SquarePosition.ABCD
  | n + 1 => next_position (nth_position n)

theorem square_position_2010 :
  nth_position 2010 = SquarePosition.BDAC :=
by sorry

end NUMINAMATH_CALUDE_square_position_2010_l3079_307906


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3079_307955

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → (|m| < 1) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → (|m| < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3079_307955


namespace NUMINAMATH_CALUDE_black_balls_count_l3079_307999

theorem black_balls_count (total_balls : ℕ) (red_balls : ℕ) (prob_red : ℚ) : 
  red_balls = 10 →
  prob_red = 2/7 →
  (red_balls : ℚ) / total_balls = prob_red →
  total_balls - red_balls = 25 := by
sorry

end NUMINAMATH_CALUDE_black_balls_count_l3079_307999
