import Mathlib

namespace parallel_vectors_m_value_l2525_252532

/-- Given vectors a and b in ℝ², where a = (-1, 1) and b = (3, m),
    and a is parallel to (a + b), prove that m = -3. -/
theorem parallel_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![-1, 1]
  let b : Fin 2 → ℝ := ![3, m]
  (∃ (k : ℝ), k ≠ 0 ∧ (λ i => a i + b i) = λ i => k * a i) →
  m = -3 := by
sorry

end parallel_vectors_m_value_l2525_252532


namespace square_area_ratio_l2525_252524

/-- Given three squares where each square's side is the diagonal of the next,
    the ratio of the largest square's area to the smallest square's area is 4 -/
theorem square_area_ratio (s₁ s₂ s₃ : ℝ) (h₁ : s₁ = s₂ * Real.sqrt 2) (h₂ : s₂ = s₃ * Real.sqrt 2) :
  s₁^2 / s₃^2 = 4 := by
  sorry

end square_area_ratio_l2525_252524


namespace car_distance_l2525_252589

/-- Proves that given the ratio of Amar's speed to the car's speed and the distance Amar covers,
    we can calculate the distance the car covers in kilometers. -/
theorem car_distance (amar_speed : ℝ) (car_speed : ℝ) (amar_distance : ℝ) :
  amar_speed / car_speed = 15 / 40 →
  amar_distance = 712.5 →
  ∃ (car_distance : ℝ), car_distance = 1.9 ∧ car_distance * 1000 * (amar_speed / car_speed) = amar_distance :=
by
  sorry

end car_distance_l2525_252589


namespace diamond_three_five_l2525_252575

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 2 * y + x * y

-- Theorem statement
theorem diamond_three_five : diamond 3 5 = 37 := by
  sorry

end diamond_three_five_l2525_252575


namespace triangle_proof_l2525_252577

theorem triangle_proof (A B C : Real) (a b c : Real) (m n : Real × Real) :
  -- Given conditions
  (A + B + C = π) →
  (m = (Real.cos B, Real.sin B)) →
  (n = (Real.cos C, -Real.sin C)) →
  (m.1 * n.1 + m.2 * n.2 = 1/2) →
  (a = 2 * Real.sqrt 3) →
  (b + c = 4) →
  -- Conclusions
  (A = 2*π/3) ∧
  (1/2 * b * c * Real.sin A = Real.sqrt 3) := by
  sorry

end triangle_proof_l2525_252577


namespace rectangular_garden_area_l2525_252511

/-- The area of a rectangular garden with length 2.5 meters and width 0.48 meters is 1.2 square meters. -/
theorem rectangular_garden_area : 
  let length : ℝ := 2.5
  let width : ℝ := 0.48
  length * width = 1.2 := by sorry

end rectangular_garden_area_l2525_252511


namespace cube_sum_implies_sum_l2525_252552

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end cube_sum_implies_sum_l2525_252552


namespace sqrt_eight_and_nine_sixteenths_l2525_252566

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) :
  x = Real.sqrt (8 + 9 / 16) → x = Real.sqrt 137 / 4 := by
  sorry

end sqrt_eight_and_nine_sixteenths_l2525_252566


namespace ellipse_fixed_point_l2525_252571

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

-- Define the right vertex M
def right_vertex (M : ℝ × ℝ) : Prop := 
  M.1 = 2 ∧ M.2 = 0 ∧ ellipse_C M.1 M.2

-- Define points A and B on the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop := 
  ellipse_C P.1 P.2 ∧ P ≠ (2, 0)

-- Define the product of slopes condition
def slope_product (M A B : ℝ × ℝ) : Prop :=
  (A.2 / (A.1 - M.1)) * (B.2 / (B.1 - M.1)) = 1/4

-- Theorem statement
theorem ellipse_fixed_point 
  (M A B : ℝ × ℝ) 
  (hM : right_vertex M) 
  (hA : on_ellipse A) 
  (hB : on_ellipse B) 
  (hAB : A ≠ B) 
  (hSlope : slope_product M A B) :
  ∃ (k : ℝ), A.2 - B.2 = k * (A.1 - B.1) ∧ 
             A.2 = k * (A.1 + 4) ∧ 
             B.2 = k * (B.1 + 4) :=
sorry

end ellipse_fixed_point_l2525_252571


namespace larger_smaller_division_l2525_252551

theorem larger_smaller_division (L S Q : ℕ) : 
  L - S = 1311 → 
  L = 1430 → 
  L = S * Q + 11 → 
  Q = 11 := by
sorry

end larger_smaller_division_l2525_252551


namespace beth_winning_strategy_l2525_252550

/-- Represents a wall of bricks in the game --/
structure Wall :=
  (size : Nat)

/-- Represents a game state with multiple walls --/
structure GameState :=
  (walls : List Wall)

/-- Calculates the nim-value of a single wall --/
def nimValue (w : Wall) : Nat :=
  sorry

/-- Calculates the nim-value of a game state --/
def gameNimValue (state : GameState) : Nat :=
  sorry

/-- Checks if a game state is a losing position for the current player --/
def isLosingPosition (state : GameState) : Prop :=
  gameNimValue state = 0

/-- The main theorem to prove --/
theorem beth_winning_strategy (startState : GameState) :
  startState.walls = [Wall.mk 6, Wall.mk 2, Wall.mk 1] →
  isLosingPosition startState :=
sorry

end beth_winning_strategy_l2525_252550


namespace completing_square_equivalence_l2525_252599

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 8*x + 9 = 0) ↔ ((x - 4)^2 = 7) := by
  sorry

end completing_square_equivalence_l2525_252599


namespace ticket_cost_l2525_252560

/-- Given that 7 tickets were purchased for a total of $308, prove that each ticket costs $44. -/
theorem ticket_cost (num_tickets : ℕ) (total_cost : ℕ) (h1 : num_tickets = 7) (h2 : total_cost = 308) :
  total_cost / num_tickets = 44 := by
  sorry

end ticket_cost_l2525_252560


namespace sin_cos_equation_solution_set_l2525_252506

theorem sin_cos_equation_solution_set (x : ℝ) : 
  Real.sin (x / 2) - Real.cos (x / 2) = 1 ↔ 
  (∃ k : ℤ, x = π * (1 + 4 * k) ∨ x = 2 * π * (1 + 2 * k)) :=
by sorry

end sin_cos_equation_solution_set_l2525_252506


namespace ellipse_m_range_l2525_252562

/-- 
Given that the equation (x^2)/(5-m) + (y^2)/(m+3) = 1 represents an ellipse,
prove that the range of values for m is (-3, 1) ∪ (1, 5).
-/
theorem ellipse_m_range (x y m : ℝ) : 
  (∃ x y, x^2 / (5 - m) + y^2 / (m + 3) = 1 ∧ 5 - m ≠ m + 3) → 
  m ∈ Set.Ioo (-3 : ℝ) 1 ∪ Set.Ioo 1 5 :=
by sorry

end ellipse_m_range_l2525_252562


namespace discount_ratio_l2525_252573

/-- Calculates the total discount for a given number of gallons -/
def calculateDiscount (gallons : ℕ) : ℚ :=
  let firstTier := min gallons 10
  let secondTier := min (gallons - 10) 10
  let thirdTier := max (gallons - 20) 0
  (firstTier : ℚ) * (5 / 100) + (secondTier : ℚ) * (10 / 100) + (thirdTier : ℚ) * (15 / 100)

/-- The discount ratio theorem -/
theorem discount_ratio :
  let kimDiscount := calculateDiscount 20
  let isabellaDiscount := calculateDiscount 25
  let elijahDiscount := calculateDiscount 30
  (isabellaDiscount : ℚ) / kimDiscount = 3 / 2 ∧
  (elijahDiscount : ℚ) / kimDiscount = 4 / 2 :=
by sorry

end discount_ratio_l2525_252573


namespace geometric_seq_property_P_iff_q_range_l2525_252567

/-- Property P for a finite sequence -/
def has_property_P (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, 1 < i ∧ i < j ∧ j ≤ n → |a 1 - a i| ≤ |a 1 - a j|

/-- Geometric sequence with first term 1 and common ratio q -/
def geometric_seq (q : ℝ) (n : ℕ) : ℝ := q^(n-1)

theorem geometric_seq_property_P_iff_q_range :
  ∀ q : ℝ, has_property_P (geometric_seq q) 10 ↔ q ∈ Set.Iic (-2) ∪ Set.Ioi 0 := by
  sorry

end geometric_seq_property_P_iff_q_range_l2525_252567


namespace odd_function_condition_l2525_252527

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 2x^3 + ax^2 + b - 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  2 * x^3 + a * x^2 + b - 1

/-- If f(x) = 2x^3 + ax^2 + b - 1 is an odd function, then a - b = -1 -/
theorem odd_function_condition (a b : ℝ) :
  IsOdd (f a b) → a - b = -1 := by
  sorry

end odd_function_condition_l2525_252527


namespace shells_calculation_l2525_252538

/-- Given an initial amount of shells and an additional amount of shells,
    calculate the total amount of shells. -/
def total_shells (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that given 5 pounds of shells initially and 23 pounds added,
    the total is 28 pounds. -/
theorem shells_calculation :
  total_shells 5 23 = 28 := by
  sorry

end shells_calculation_l2525_252538


namespace A_not_always_in_second_quadrant_l2525_252574

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of being on the negative x-axis -/
def isOnNegativeXAxis (p : Point) : Prop :=
  p.x < 0 ∧ p.y = 0

/-- The point A(-a^2-1,|b|) -/
def A (a b : ℝ) : Point :=
  { x := -a^2 - 1, y := |b| }

/-- Theorem stating that A(-a^2-1,|b|) is not always in the second quadrant -/
theorem A_not_always_in_second_quadrant :
  ∃ a b : ℝ, ¬(isInSecondQuadrant (A a b)) ∧ (isInSecondQuadrant (A a b) ∨ isOnNegativeXAxis (A a b)) :=
sorry

end A_not_always_in_second_quadrant_l2525_252574


namespace square_roots_equality_l2525_252518

theorem square_roots_equality (m : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (m + 1)^2 = x ∧ (3*m - 1)^2 = x) → m = 1 := by
  sorry

end square_roots_equality_l2525_252518


namespace function_equality_l2525_252515

theorem function_equality (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x - 5) :
  2 * (f 3) - 10 = f (3 - 2) := by
  sorry

end function_equality_l2525_252515


namespace susans_bread_profit_l2525_252596

/-- Susan's bread selling problem -/
theorem susans_bread_profit :
  let total_loaves : ℕ := 60
  let cost_per_loaf : ℚ := 1
  let morning_price : ℚ := 3
  let afternoon_price : ℚ := 2
  let evening_price : ℚ := 3/2
  let morning_fraction : ℚ := 1/3
  let afternoon_fraction : ℚ := 1/2

  let morning_sales : ℚ := morning_fraction * total_loaves * morning_price
  let afternoon_sales : ℚ := afternoon_fraction * (total_loaves - morning_fraction * total_loaves) * afternoon_price
  let evening_sales : ℚ := (total_loaves - morning_fraction * total_loaves - afternoon_fraction * (total_loaves - morning_fraction * total_loaves)) * evening_price

  let total_revenue : ℚ := morning_sales + afternoon_sales + evening_sales
  let total_cost : ℚ := total_loaves * cost_per_loaf
  let profit : ℚ := total_revenue - total_cost

  profit = 70 := by sorry

end susans_bread_profit_l2525_252596


namespace salaries_degrees_l2525_252517

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  transportation : ℝ
  research_development : ℝ
  utilities : ℝ
  equipment : ℝ
  supplies : ℝ
  salaries : ℝ

/-- The total budget percentage should sum to 100% -/
axiom budget_sum (b : BudgetAllocation) : 
  b.transportation + b.research_development + b.utilities + b.equipment + b.supplies + b.salaries = 100

/-- The given budget allocation -/
def company_budget : BudgetAllocation where
  transportation := 20
  research_development := 9
  utilities := 5
  equipment := 4
  supplies := 2
  salaries := 100 - (20 + 9 + 5 + 4 + 2)

/-- The number of degrees in a full circle -/
def full_circle : ℝ := 360

/-- Theorem: The number of degrees representing salaries in the circle graph is 216 -/
theorem salaries_degrees : 
  (company_budget.salaries / 100) * full_circle = 216 := by sorry

end salaries_degrees_l2525_252517


namespace intersection_and_union_when_a_is_one_range_of_a_when_complement_A_subset_B_l2525_252530

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 > 0}
def B (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_one :
  (A ∩ B 1 = {x | x < -2}) ∧ (A ∪ B 1 = {x | x > 2 ∨ x ≤ 1}) := by sorry

-- Theorem for part (2)
theorem range_of_a_when_complement_A_subset_B :
  ∀ a : ℝ, (Set.univ \ A : Set ℝ) ⊆ B a → a ≥ 2 := by sorry

end intersection_and_union_when_a_is_one_range_of_a_when_complement_A_subset_B_l2525_252530


namespace polynomial_factorization_l2525_252529

theorem polynomial_factorization (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3)^2 := by
  sorry

end polynomial_factorization_l2525_252529


namespace cola_sales_count_l2525_252572

/-- Represents the number of bottles sold for each drink type -/
structure DrinkSales where
  cola : ℕ
  juice : ℕ
  water : ℕ

/-- Calculates the total earnings from drink sales -/
def totalEarnings (sales : DrinkSales) : ℚ :=
  3 * sales.cola + 1.5 * sales.juice + 1 * sales.water

/-- Theorem stating that the number of cola bottles sold is 15 -/
theorem cola_sales_count : ∃ (sales : DrinkSales), 
  sales.juice = 12 ∧ 
  sales.water = 25 ∧ 
  totalEarnings sales = 88 ∧ 
  sales.cola = 15 := by
  sorry

end cola_sales_count_l2525_252572


namespace identity_function_satisfies_equation_l2525_252522

theorem identity_function_satisfies_equation (f : ℕ → ℕ) :
  (∀ m n : ℕ, f (f m + f n) = m + n) → (∀ n : ℕ, f n = n) := by
  sorry

end identity_function_satisfies_equation_l2525_252522


namespace fixed_point_of_exponential_function_l2525_252537

/-- For any positive real number a, the function f(x) = a^(x-1) + 2 always passes through the point (1, 3). -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a^(x-1) + 2
  f 1 = 3 := by
  sorry

end fixed_point_of_exponential_function_l2525_252537


namespace e_is_largest_l2525_252598

-- Define the variables
variable (a b c d e : ℝ)

-- Define the given equation
def equation := (a - 2 = b + 3) ∧ (b + 3 = c - 4) ∧ (c - 4 = d + 5) ∧ (d + 5 = e - 6)

-- Theorem statement
theorem e_is_largest (h : equation a b c d e) : 
  e = max a (max b (max c d)) :=
sorry

end e_is_largest_l2525_252598


namespace unique_solution_fourth_root_equation_l2525_252516

theorem unique_solution_fourth_root_equation :
  ∃! x : ℝ, (((4 - x) ^ (1/4) : ℝ) + ((x - 2) ^ (1/2) : ℝ) = 2) := by
  sorry

end unique_solution_fourth_root_equation_l2525_252516


namespace circle_center_coordinates_l2525_252534

/-- The center of a circle that is tangent to two parallel lines and lies on a third line -/
theorem circle_center_coordinates (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    ((x - 20)^2 + (y - 10)^2 = r^2) ∧ 
    ((x - 40/3)^2 + y^2 = r^2) ∧ 
    x^2 + y^2 = r^2) → 
  (3*x - 4*y = 20 ∧ x - 2*y = 0) → 
  x = 20 ∧ y = 10 := by
sorry

end circle_center_coordinates_l2525_252534


namespace expression_evaluation_l2525_252549

theorem expression_evaluation (c d : ℝ) (hc : c = 3) (hd : d = 2) :
  (c^2 + d + 1)^2 - (c^2 - d - 1)^2 = 80 := by
  sorry

end expression_evaluation_l2525_252549


namespace sample_capacity_proof_l2525_252561

/-- The sample capacity for a population of 36 individuals -/
def sample_capacity : ℕ := 6

theorem sample_capacity_proof :
  let total_population : ℕ := 36
  (sample_capacity ∣ total_population) ∧
  (6 ∣ sample_capacity) ∧
  ((total_population - 1) % (sample_capacity + 1) = 0) →
  sample_capacity = 6 :=
by sorry

end sample_capacity_proof_l2525_252561


namespace factor_x_pow_10_minus_1296_l2525_252569

theorem factor_x_pow_10_minus_1296 (x : ℝ) : x^10 - 1296 = (x^5 + 36) * (x^5 - 36) := by
  sorry

end factor_x_pow_10_minus_1296_l2525_252569


namespace sum_of_digits_l2525_252556

def num1 : ℕ := 404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404
def num2 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

def product : ℕ := num1 * num2

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits : 
  thousands_digit product + units_digit product = 10 := by sorry

end sum_of_digits_l2525_252556


namespace john_mary_distance_difference_l2525_252504

/-- The width of the streets in feet -/
def street_width : ℕ := 15

/-- The side length of a block in feet -/
def block_side_length : ℕ := 300

/-- The perimeter of a square -/
def square_perimeter (side_length : ℕ) : ℕ := 4 * side_length

theorem john_mary_distance_difference :
  square_perimeter (block_side_length + 2 * street_width) - square_perimeter block_side_length = 120 := by
  sorry

end john_mary_distance_difference_l2525_252504


namespace paige_catfish_l2525_252565

/-- The number of goldfish Paige initially raised -/
def initial_goldfish : ℕ := 7

/-- The number of fish that disappeared -/
def disappeared_fish : ℕ := 4

/-- The number of fish left -/
def remaining_fish : ℕ := 15

/-- The number of catfish Paige initially raised -/
def initial_catfish : ℕ := initial_goldfish + disappeared_fish + remaining_fish - initial_goldfish

theorem paige_catfish : initial_catfish = 12 := by
  sorry

end paige_catfish_l2525_252565


namespace x_over_y_equals_negative_one_fourth_l2525_252541

theorem x_over_y_equals_negative_one_fourth (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 * x + y)^5 + x^5 + 4 * x + y = 0) : x / y = -1 / 4 := by
  sorry

end x_over_y_equals_negative_one_fourth_l2525_252541


namespace f_inequality_l2525_252586

noncomputable def f (x : ℝ) : ℝ := x * Real.log (Real.sqrt (x^2 + 1) + x) + x^2 - x * Real.sin x

theorem f_inequality (x : ℝ) : f x > f (2*x - 1) ↔ x ∈ Set.Ioo (1/3 : ℝ) 1 := by sorry

end f_inequality_l2525_252586


namespace limit_of_a_l2525_252521

def a (n : ℕ) : ℚ := (2 * n + 1) / (5 * n - 1)

theorem limit_of_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - 2/5| < ε :=
by
  sorry

end limit_of_a_l2525_252521


namespace min_value_expression_l2525_252585

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_prod : x * y * z = 3 / 4) (h_sum : x + y + z = 4) :
  x^3 + x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 + z^3 ≥ 21/2 ∧
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 3 / 4 ∧ x + y + z = 4 ∧
    x^3 + x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 + z^3 = 21/2 := by
  sorry

end min_value_expression_l2525_252585


namespace rectangle_area_l2525_252584

/-- The area of a rectangle with sides 5.9 cm and 3 cm is 17.7 square centimeters. -/
theorem rectangle_area : 
  let side1 : ℝ := 5.9
  let side2 : ℝ := 3
  side1 * side2 = 17.7 :=
by sorry

end rectangle_area_l2525_252584


namespace solution_exists_l2525_252555

theorem solution_exists (x y : ℝ) : (2*x - 3*y + 5)^2 + |x - y + 2| = 0 → x = -1 ∧ y = 1 := by
  sorry

end solution_exists_l2525_252555


namespace sqrt_equation_solution_l2525_252514

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 3) = 5 → x = 28 := by
  sorry

end sqrt_equation_solution_l2525_252514


namespace simplify_2A_minus_B_value_2A_minus_B_special_case_l2525_252540

/-- Definition of A in terms of a and b -/
def A (a b : ℝ) : ℝ := b^2 - a^2 + 5*a*b

/-- Definition of B in terms of a and b -/
def B (a b : ℝ) : ℝ := 3*a*b + 2*b^2 - a^2

/-- Theorem stating the simplified form of 2A - B -/
theorem simplify_2A_minus_B (a b : ℝ) : 2 * A a b - B a b = -a^2 + 7*a*b := by sorry

/-- Theorem stating the value of 2A - B when a = 1 and b = 2 -/
theorem value_2A_minus_B_special_case : 2 * A 1 2 - B 1 2 = 13 := by sorry

end simplify_2A_minus_B_value_2A_minus_B_special_case_l2525_252540


namespace line_circle_separation_l2525_252579

theorem line_circle_separation (a b : ℝ) 
  (h_inside : a^2 + b^2 < 1) : 
  ∃ (d : ℝ), d > 1 ∧ d = 1 / Real.sqrt (a^2 + b^2) := by
  sorry

end line_circle_separation_l2525_252579


namespace minimum_value_of_f_l2525_252581

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + a else x^2 - a*x

theorem minimum_value_of_f (a : ℝ) :
  (∀ x, f a x ≥ a) ∧ (∃ x, f a x = a) → a = -4 := by
  sorry

end minimum_value_of_f_l2525_252581


namespace min_distance_to_line_l2525_252513

theorem min_distance_to_line (x y : ℝ) (h1 : 8 * x + 15 * y = 120) (h2 : x ≥ 0) :
  ∃ (min_dist : ℝ), min_dist = 120 / 17 ∧ 
    ∀ (x' y' : ℝ), 8 * x' + 15 * y' = 120 → x' ≥ 0 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_dist :=
by sorry

end min_distance_to_line_l2525_252513


namespace gift_original_price_gift_price_calculation_l2525_252591

/-- The original price of a gift, given certain conditions --/
theorem gift_original_price (half_cost : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let full_cost := 2 * half_cost
  let discounted_price := (1 - discount_rate) * full_cost / ((1 - discount_rate) * (1 + tax_rate))
  discounted_price

/-- The original price of the gift is approximately $30.50 --/
theorem gift_price_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |gift_original_price 14 0.15 0.08 - 30.50| < ε :=
sorry

end gift_original_price_gift_price_calculation_l2525_252591


namespace inscribed_squares_ratio_l2525_252587

/-- Right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 5 ∧ b = 12 ∧ c = 13

/-- Square inscribed in the right triangle with vertex at right angle -/
def inscribed_square_vertex (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x ≤ t.b ∧ x / t.a = x / t.b

/-- Square inscribed in the right triangle with side on hypotenuse -/
def inscribed_square_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.c ∧ (t.b / t.a) * y + y + (t.a / t.b) * y = t.c

theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ)
  (h1 : inscribed_square_vertex t1 x)
  (h2 : inscribed_square_hypotenuse t2 y) :
  x / y = 39 / 51 := by
  sorry

end inscribed_squares_ratio_l2525_252587


namespace f_is_even_l2525_252525

-- Define g as an odd function
def g_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^3)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_odd g) : ∀ x, f g (-x) = f g x := by sorry

end f_is_even_l2525_252525


namespace intersection_of_A_and_B_l2525_252502

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ -2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l2525_252502


namespace complex_equation_solution_l2525_252536

theorem complex_equation_solution (z : ℂ) :
  (3 - 4 * Complex.I) * z = 5 → z = 3/5 + 4/5 * Complex.I :=
by
  sorry

end complex_equation_solution_l2525_252536


namespace quadratic_roots_complex_l2525_252505

theorem quadratic_roots_complex (x : ℂ) :
  x^2 - 6*x + 25 = 0 ↔ x = 3 + 4*I ∨ x = 3 - 4*I :=
by sorry

end quadratic_roots_complex_l2525_252505


namespace white_dandelions_on_saturday_l2525_252533

/-- Represents the day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the state of a dandelion -/
inductive DandelionState
  | Yellow
  | White
  | Dispersed

/-- Represents the count of dandelions in different states -/
structure DandelionCount where
  yellow : ℕ
  white : ℕ

/-- The life cycle of a dandelion -/
def dandelionLifeCycle (day : ℕ) : DandelionState :=
  match day with
  | 0 | 1 | 2 => DandelionState.Yellow
  | 3 | 4 => DandelionState.White
  | _ => DandelionState.Dispersed

/-- Count of dandelions on a given day -/
def dandelionCountOnDay (day : Day) : DandelionCount :=
  match day with
  | Day.Monday => { yellow := 20, white := 14 }
  | Day.Wednesday => { yellow := 15, white := 11 }
  | _ => { yellow := 0, white := 0 }  -- We don't have information for other days

/-- Days between two given days -/
def daysBetween (start finish : Day) : ℕ :=
  match start, finish with
  | Day.Monday, Day.Wednesday => 2
  | Day.Wednesday, Day.Saturday => 3
  | _, _ => 0  -- We don't need other cases for this problem

/-- The main theorem -/
theorem white_dandelions_on_saturday :
  ∃ (new_dandelions : ℕ),
    new_dandelions = (dandelionCountOnDay Day.Wednesday).yellow + (dandelionCountOnDay Day.Wednesday).white
                   - (dandelionCountOnDay Day.Monday).yellow
    ∧ new_dandelions = 6
    ∧ (dandelionLifeCycle (daysBetween Day.Tuesday Day.Saturday) = DandelionState.White
    ∧ dandelionLifeCycle (daysBetween Day.Wednesday Day.Saturday) = DandelionState.White)
    → new_dandelions = 6 := by sorry


end white_dandelions_on_saturday_l2525_252533


namespace f_equals_g_l2525_252508

theorem f_equals_g (f g : ℝ → ℝ) 
  (hf_cont : Continuous f)
  (hg_mono : Monotone g)
  (h_seq : ∀ a b c : ℝ, a < b → b < c → 
    ∃ (x : ℕ → ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - b| < ε) ∧ 
    (∃ L : ℝ, ∀ ε > 0, ∃ N, ∀ n ≥ N, |g (x n) - L| < ε) ∧
    f a < L ∧ L < f c) :
  f = g := by
sorry

end f_equals_g_l2525_252508


namespace a_in_M_necessary_not_sufficient_for_a_in_N_l2525_252588

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = Real.log x}
def N : Set ℝ := {x | x > 0}

-- Statement to prove
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by
  sorry

end a_in_M_necessary_not_sufficient_for_a_in_N_l2525_252588


namespace derivative_of_f_l2525_252546

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 1 + Real.cos x := by sorry

end derivative_of_f_l2525_252546


namespace mrs_hilt_travel_distance_l2525_252593

/-- Calculates the total miles traveled given the initial odometer reading and additional miles --/
def total_miles_traveled (initial_reading : ℝ) (additional_miles : ℝ) : ℝ :=
  initial_reading + additional_miles

/-- Theorem stating that the total miles traveled is 2,210.23 given the specific conditions --/
theorem mrs_hilt_travel_distance :
  total_miles_traveled 1498.76 711.47 = 2210.23 := by
  sorry

end mrs_hilt_travel_distance_l2525_252593


namespace triangle_angle_calculation_l2525_252563

/-- Theorem: In a triangle ABC where angle A is x degrees, angle B is 2x degrees, 
    and angle C is 45°, the value of x is 45°. -/
theorem triangle_angle_calculation (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ 2*x < 180 ∧ 
  x + 2*x + 45 = 180 → 
  x = 45 := by
  sorry

end triangle_angle_calculation_l2525_252563


namespace bread_sharing_theorem_l2525_252501

/-- Calculates the number of slices each friend eats when sharing bread equally -/
def slices_per_friend (slices_per_loaf : ℕ) (num_friends : ℕ) (num_loaves : ℕ) : ℕ :=
  (slices_per_loaf * num_loaves) / num_friends

/-- Proves that under the given conditions, each friend eats 6 slices of bread -/
theorem bread_sharing_theorem :
  let slices_per_loaf : ℕ := 15
  let num_friends : ℕ := 10
  let num_loaves : ℕ := 4
  slices_per_friend slices_per_loaf num_friends num_loaves = 6 := by
  sorry

end bread_sharing_theorem_l2525_252501


namespace cube_volume_from_surface_area_l2525_252539

theorem cube_volume_from_surface_area :
  ∀ s : ℝ, 6 * s^2 = 150 → s^3 = 125 := by sorry

end cube_volume_from_surface_area_l2525_252539


namespace multiple_choice_test_choices_l2525_252592

theorem multiple_choice_test_choices (n : ℕ) : 
  (n + 1)^4 = 625 → n = 4 := by
  sorry

end multiple_choice_test_choices_l2525_252592


namespace regular_octagon_interior_angle_measure_l2525_252509

/-- The measure of an interior angle of a regular octagon in degrees -/
def regular_octagon_interior_angle : ℝ := 135

/-- A regular octagon has 8 sides -/
def regular_octagon_sides : ℕ := 8

theorem regular_octagon_interior_angle_measure :
  regular_octagon_interior_angle = (((regular_octagon_sides - 2) * 180) : ℝ) / regular_octagon_sides :=
sorry

end regular_octagon_interior_angle_measure_l2525_252509


namespace division_problem_l2525_252528

theorem division_problem (x : ℝ) : 
  (1.5 * 1265) / x = 271.07142857142856 → x = 7 := by
  sorry

end division_problem_l2525_252528


namespace percentage_problem_l2525_252510

theorem percentage_problem (x : ℝ) (h1 : 0.2 * x = 400) : 
  (2400 / x) * 100 = 120 := by
  sorry

end percentage_problem_l2525_252510


namespace badminton_tournament_matches_l2525_252523

/-- Represents a single elimination tournament -/
structure Tournament :=
  (total_participants : ℕ)
  (auto_progressed : ℕ)
  (first_round_players : ℕ)
  (h_participants : total_participants = auto_progressed + first_round_players)

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : Tournament) : ℕ := t.total_participants - 1

theorem badminton_tournament_matches :
  ∀ t : Tournament,
  t.total_participants = 120 →
  t.auto_progressed = 16 →
  t.first_round_players = 104 →
  total_matches t = 119 :=
by sorry

end badminton_tournament_matches_l2525_252523


namespace base_comparison_l2525_252580

theorem base_comparison (a b n : ℕ) (A_n B_n A_n_minus_1 B_n_minus_1 : ℕ) 
  (ha : a > 1) (hb : b > 1) (hn : n > 1)
  (hA : A_n > 0) (hB : B_n > 0) (hA_minus_1 : A_n_minus_1 > 0) (hB_minus_1 : B_n_minus_1 > 0)
  (hA_def : A_n = a^n + A_n_minus_1) (hB_def : B_n = b^n + B_n_minus_1) :
  (a > b) ↔ (A_n_minus_1 / A_n : ℚ) < (B_n_minus_1 / B_n : ℚ) := by
sorry

end base_comparison_l2525_252580


namespace sin_seven_pi_sixths_l2525_252500

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end sin_seven_pi_sixths_l2525_252500


namespace union_complement_equals_set_l2525_252545

universe u

def U : Finset ℕ := {0,1,2,4,6,8}
def M : Finset ℕ := {0,4,6}
def N : Finset ℕ := {0,1,6}

theorem union_complement_equals_set : M ∪ (U \ N) = {0,2,4,6,8} := by sorry

end union_complement_equals_set_l2525_252545


namespace composite_sum_l2525_252582

theorem composite_sum (x y : ℕ) (h1 : x > 1) (h2 : y > 1) 
  (h3 : (x^2 + y^2 - 1) % (x + y - 1) = 0) : 
  ¬ Nat.Prime (x + y - 1) := by
sorry

end composite_sum_l2525_252582


namespace divisible_by_six_percentage_l2525_252553

theorem divisible_by_six_percentage (n : ℕ) : n = 150 →
  (((Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))).card : ℚ) / (n : ℚ)) * 100 = 50/3 := by
  sorry

end divisible_by_six_percentage_l2525_252553


namespace polynomial_product_equality_l2525_252564

theorem polynomial_product_equality (x : ℝ) : 
  (x^4 + 50*x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end polynomial_product_equality_l2525_252564


namespace reciprocal_of_negative_fraction_reciprocal_of_negative_one_thirteenth_l2525_252548

theorem reciprocal_of_negative_fraction (a b : ℤ) (hb : b ≠ 0) :
  ((-1 : ℚ) / (a : ℚ) / (b : ℚ))⁻¹ = -((b : ℚ) / (a : ℚ)) :=
by sorry

theorem reciprocal_of_negative_one_thirteenth :
  ((-1 : ℚ) / 13)⁻¹ = -13 :=
by sorry

end reciprocal_of_negative_fraction_reciprocal_of_negative_one_thirteenth_l2525_252548


namespace system_solution_existence_l2525_252558

/-- Given a system of equations:
    1. y = b - x²
    2. x² + y² + 2a² = 4 - 2a(x + y)
    This theorem states the condition on b for the existence of at least one solution (x, y)
    for some real number a. -/
theorem system_solution_existence (b : ℝ) : 
  (∃ (a x y : ℝ), y = b - x^2 ∧ x^2 + y^2 + 2*a^2 = 4 - 2*a*(x + y)) ↔ 
  b ≥ -2 * Real.sqrt 2 - 1/4 := by
sorry


end system_solution_existence_l2525_252558


namespace x_coordinate_of_first_point_l2525_252503

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x = 2 * y + 5

-- Define the two points
def point1 (m n : ℝ) : ℝ × ℝ := (m, n)
def point2 (m n : ℝ) : ℝ × ℝ := (m + 4, n + 2)

-- Theorem statement
theorem x_coordinate_of_first_point (m n : ℝ) :
  line_equation m n ∧ line_equation (m + 4) (n + 2) → m = 2 * n + 5 := by
  sorry

end x_coordinate_of_first_point_l2525_252503


namespace fraction_equality_l2525_252535

theorem fraction_equality : (2015 : ℚ) / (2015^2 - 2016 * 2014) = 2015 := by sorry

end fraction_equality_l2525_252535


namespace inequalities_hold_l2525_252557

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ab ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ 1/a + 1/b ≥ 2 := by
  sorry

end inequalities_hold_l2525_252557


namespace n_sticks_ge_n_plus_one_minos_l2525_252570

/-- An n-stick is a connected figure of n matches of length 1, placed horizontally or vertically, no two touching except at ends. -/
def NStick (n : ℕ) : Type := sorry

/-- An n-mino is a shape built by connecting n squares of side length 1 on their sides, with a path between each two squares. -/
def NMino (n : ℕ) : Type := sorry

/-- S_n is the number of n-sticks -/
def S (n : ℕ) : ℕ := sorry

/-- M_n is the number of n-minos -/
def M (n : ℕ) : ℕ := sorry

/-- For any natural number n, the number of n-sticks is greater than or equal to the number of (n+1)-minos. -/
theorem n_sticks_ge_n_plus_one_minos (n : ℕ) : S n ≥ M (n + 1) := by sorry

end n_sticks_ge_n_plus_one_minos_l2525_252570


namespace sufficient_not_necessary_l2525_252544

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x < -1 → 2 * x^2 + x - 1 > 0) ∧
  (∃ x, 2 * x^2 + x - 1 > 0 ∧ x ≥ -1) :=
by sorry

end sufficient_not_necessary_l2525_252544


namespace triangle_circle_radii_l2525_252559

theorem triangle_circle_radii (a b c : ℝ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 8) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * area)
  let r := area / s
  R = (7 * Real.sqrt 3) / 3 ∧ r = Real.sqrt 3 := by sorry

end triangle_circle_radii_l2525_252559


namespace cards_in_hospital_l2525_252507

/-- Proves that the number of get well cards Mariela received while in the hospital is 403 -/
theorem cards_in_hospital (total_cards : ℕ) (cards_after_home : ℕ) 
  (h1 : total_cards = 690) 
  (h2 : cards_after_home = 287) : 
  total_cards - cards_after_home = 403 := by
  sorry

end cards_in_hospital_l2525_252507


namespace milk_remaining_l2525_252554

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial = 5 →
  given_away = 17 / 4 →
  remaining = initial - given_away →
  remaining = 3 / 4 := by
  sorry

end milk_remaining_l2525_252554


namespace additive_inverse_of_negative_2023_l2525_252595

theorem additive_inverse_of_negative_2023 :
  ∃! x : ℝ, -2023 + x = 0 ∧ x = 2023 := by sorry

end additive_inverse_of_negative_2023_l2525_252595


namespace hike_weight_after_six_hours_l2525_252512

/-- Calculates the remaining weight after a hike given initial weights and consumption rates -/
def remaining_weight (initial_water : ℝ) (initial_food : ℝ) (initial_gear : ℝ) 
                     (water_rate : ℝ) (food_rate : ℝ) (hours : ℝ) : ℝ :=
  let remaining_water := initial_water - water_rate * hours
  let remaining_food := initial_food - food_rate * hours
  remaining_water + remaining_food + initial_gear

/-- Theorem: The remaining weight after 6 hours of hiking is 34 pounds -/
theorem hike_weight_after_six_hours :
  remaining_weight 20 10 20 2 (2/3) 6 = 34 := by
  sorry

end hike_weight_after_six_hours_l2525_252512


namespace power_product_equality_l2525_252519

theorem power_product_equality : (3^5 * 4^5) = 248832 := by sorry

end power_product_equality_l2525_252519


namespace cubic_sum_powers_l2525_252590

theorem cubic_sum_powers (a : ℝ) (h : a^3 + 3*a^2 + 3*a + 2 = 0) :
  (a + 1)^2008 + (a + 1)^2009 + (a + 1)^2010 = 1 := by
  sorry

end cubic_sum_powers_l2525_252590


namespace initial_cookies_correct_l2525_252578

/-- The number of cookies Paco had initially -/
def initial_cookies : ℕ := 36

/-- The number of cookies Paco gave to his friend -/
def given_cookies : ℕ := 14

/-- The number of cookies Paco ate -/
def eaten_cookies : ℕ := 10

/-- The number of cookies Paco had left -/
def remaining_cookies : ℕ := 12

/-- Theorem stating that the initial number of cookies is correct -/
theorem initial_cookies_correct : 
  initial_cookies = given_cookies + eaten_cookies + remaining_cookies :=
by sorry

end initial_cookies_correct_l2525_252578


namespace first_three_seeds_l2525_252547

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is a valid seed number --/
def isValidSeedNumber (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 850

/-- Extracts numbers from the random number table --/
def extractNumbers (table : RandomNumberTable) (startRow : Nat) (startCol : Nat) (count : Nat) : List Nat :=
  sorry

/-- Selects valid seed numbers from a list of numbers --/
def selectValidSeedNumbers (numbers : List Nat) (count : Nat) : List Nat :=
  sorry

theorem first_three_seeds (table : RandomNumberTable) :
  let extractedNumbers := extractNumbers table 8 7 10
  let selectedSeeds := selectValidSeedNumbers extractedNumbers 3
  selectedSeeds = [785, 567, 199] := by
  sorry

end first_three_seeds_l2525_252547


namespace ap_num_terms_l2525_252531

/-- The number of terms in an arithmetic progression -/
def num_terms_ap (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  (aₙ - a₁) / d + 1

/-- Theorem: In an arithmetic progression with first term 2, last term 62,
    and common difference 2, the number of terms is 31 -/
theorem ap_num_terms :
  num_terms_ap 2 62 2 = 31 := by
  sorry

#eval num_terms_ap 2 62 2

end ap_num_terms_l2525_252531


namespace largest_power_dividing_factorial_l2525_252543

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 2^5 * 3^2 * 7) :
  (∃ m : ℕ, n^m ∣ n! ∧ ∀ k > m, ¬(n^k ∣ n!)) →
  (∃ m : ℕ, n^m ∣ n! ∧ ∀ k > m, ¬(n^k ∣ n!) ∧ m = 334) :=
by sorry

end largest_power_dividing_factorial_l2525_252543


namespace complex_square_quadrant_l2525_252583

theorem complex_square_quadrant (z : ℂ) : 
  z = Complex.exp (Complex.I * Real.pi * (5/12)) → 
  (z^2).re < 0 ∧ (z^2).im > 0 :=
sorry

end complex_square_quadrant_l2525_252583


namespace color_film_fraction_l2525_252520

theorem color_film_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw := 20 * x
  let total_color := 4 * y
  let selected_bw := y / (5 * x) * total_bw
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  (selected_color / total_selected) = 20 / 21 := by
  sorry

end color_film_fraction_l2525_252520


namespace apples_left_l2525_252526

def initial_apples : ℕ := 150
def sold_percentage : ℚ := 30 / 100
def given_percentage : ℚ := 20 / 100
def donated_apples : ℕ := 2

theorem apples_left : 
  let remaining_after_sale := initial_apples - (↑initial_apples * sold_percentage).floor
  let remaining_after_given := remaining_after_sale - (↑remaining_after_sale * given_percentage).floor
  remaining_after_given - donated_apples = 82 := by sorry

end apples_left_l2525_252526


namespace tenth_pattern_stones_l2525_252597

def stone_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => stone_sequence n + 3 * (n + 2) - 2

theorem tenth_pattern_stones : stone_sequence 9 = 145 := by
  sorry

end tenth_pattern_stones_l2525_252597


namespace markup_rate_l2525_252542

theorem markup_rate (selling_price : ℝ) (profit_rate : ℝ) (expense_rate : ℝ) : 
  selling_price = 5 → 
  profit_rate = 0.1 → 
  expense_rate = 0.15 → 
  (selling_price / (selling_price * (1 - profit_rate - expense_rate)) - 1) = 1/3 := by
  sorry

end markup_rate_l2525_252542


namespace complex_sum_theorem_l2525_252576

theorem complex_sum_theorem (x y u v w z : ℂ) : 
  v = 2 → 
  w = -x - u → 
  (x + y * Complex.I) + (u + v * Complex.I) + (w + z * Complex.I) = 2 * Complex.I → 
  z + y = 0 := by sorry

end complex_sum_theorem_l2525_252576


namespace betty_oranges_purchase_l2525_252568

/-- Represents the problem of determining how many kg of oranges Betty bought. -/
theorem betty_oranges_purchase :
  ∀ (orange_kg : ℝ) (apple_kg : ℝ) (orange_cost : ℝ) (apple_price_per_kg : ℝ),
    apple_kg = 3 →
    orange_cost = 12 →
    apple_price_per_kg = 2 →
    apple_price_per_kg * 2 = orange_cost / orange_kg →
    orange_kg = 3 := by
  sorry

end betty_oranges_purchase_l2525_252568


namespace function_equality_condition_l2525_252594

theorem function_equality_condition (m n p q : ℝ) : 
  let f := λ x : ℝ => m * x^2 + n
  let g := λ x : ℝ => p * x + q
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p^2) = q * (1 - m) := by sorry

end function_equality_condition_l2525_252594
