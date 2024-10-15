import Mathlib

namespace NUMINAMATH_CALUDE_first_player_wins_l2985_298501

/-- Represents the state of a Kayles game -/
structure KaylesGame where
  pins : List Bool
  turn : Nat

/-- Knocks over one pin or two adjacent pins -/
def makeMove (game : KaylesGame) (start : Nat) (count : Nat) : KaylesGame :=
  sorry

/-- Checks if the game is over (no pins left standing) -/
def isGameOver (game : KaylesGame) : Bool :=
  sorry

/-- Represents a strategy for playing Kayles -/
def Strategy := KaylesGame → Nat × Nat

/-- Checks if a strategy is winning for the current player -/
def isWinningStrategy (strat : Strategy) (game : KaylesGame) : Bool :=
  sorry

/-- The main theorem: there exists a winning strategy for the first player -/
theorem first_player_wins :
  ∀ n : Nat, ∃ strat : Strategy, isWinningStrategy strat (KaylesGame.mk (List.replicate n true) 0) :=
  sorry

end NUMINAMATH_CALUDE_first_player_wins_l2985_298501


namespace NUMINAMATH_CALUDE_cube_net_theorem_l2985_298541

theorem cube_net_theorem (a b c : ℝ) 
  (eq1 : 3 * a + 2 = 17)
  (eq2 : 7 * b - 4 = 10)
  (eq3 : a + 3 * b - 2 * c = 11) :
  a - b * c = 5 := by
sorry

end NUMINAMATH_CALUDE_cube_net_theorem_l2985_298541


namespace NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l2985_298576

theorem equidistant_point_on_x_axis : ∃ x : ℝ, 
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (0, 5)
  let P : ℝ × ℝ := (x, 0)
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧ x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l2985_298576


namespace NUMINAMATH_CALUDE_incorrect_mark_is_83_l2985_298583

/-- The number of pupils in the class -/
def num_pupils : ℕ := 40

/-- The correct mark for the pupil -/
def correct_mark : ℕ := 63

/-- The incorrect mark entered for the pupil -/
def incorrect_mark : ℕ := 83

/-- Theorem stating that the incorrect mark is 83 -/
theorem incorrect_mark_is_83 :
  (incorrect_mark - correct_mark) * num_pupils = num_pupils * (num_pupils / 2) :=
sorry

end NUMINAMATH_CALUDE_incorrect_mark_is_83_l2985_298583


namespace NUMINAMATH_CALUDE_equation_solution_l2985_298590

theorem equation_solution :
  ∃ y : ℝ, y - 9 / (y - 4) = 2 - 9 / (y - 4) ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2985_298590


namespace NUMINAMATH_CALUDE_task_completion_probability_l2985_298511

theorem task_completion_probability 
  (task1_prob : ℝ) 
  (task1_not_task2_prob : ℝ) 
  (h1 : task1_prob = 2/3) 
  (h2 : task1_not_task2_prob = 4/15) 
  (h_independent : task1_not_task2_prob = task1_prob * (1 - task2_prob)) : 
  task2_prob = 3/5 :=
by
  sorry

#check task_completion_probability

end NUMINAMATH_CALUDE_task_completion_probability_l2985_298511


namespace NUMINAMATH_CALUDE_parabola_and_line_intersection_l2985_298575

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -(Real.sqrt 3 / 2) * y

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

theorem parabola_and_line_intersection :
  -- Conditions
  (∀ x y, parabola x y → parabola (-x) y) → -- Symmetry about y-axis
  parabola 0 0 → -- Vertex at origin
  parabola (Real.sqrt 3) (-2 * Real.sqrt 3) → -- Passes through (√3, -2√3)
  -- Conclusion
  (∀ m, (∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ 
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ 
    line x₁ y₁ m ∧ line x₂ y₂ m) ↔ 
    m < Real.sqrt 3 / 8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_line_intersection_l2985_298575


namespace NUMINAMATH_CALUDE_differentiable_implies_continuous_l2985_298555

theorem differentiable_implies_continuous (f : ℝ → ℝ) (x₀ : ℝ) :
  DifferentiableAt ℝ f x₀ → ContinuousAt f x₀ := by
  sorry

end NUMINAMATH_CALUDE_differentiable_implies_continuous_l2985_298555


namespace NUMINAMATH_CALUDE_max_m_inequality_l2985_298534

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 1 / 4) :
  (∀ m : ℝ, 2 * a + b ≥ 4 * m) → (∃ m : ℝ, m = 9 ∧ 2 * a + b = 4 * m) :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l2985_298534


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l2985_298542

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem intersection_points_theorem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_even f)
  (h2 : has_period f 2)
  (h3 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2)
  (h4 : ∃! (p q : ℝ), p ≠ q ∧ f p = p + a ∧ f q = q + a) :
  ∃ k : ℤ, a = 2 * k ∨ a = 2 * k - 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l2985_298542


namespace NUMINAMATH_CALUDE_total_interest_calculation_l2985_298550

/-- Calculates the total interest earned from two bank investments -/
theorem total_interest_calculation 
  (total_investment : ℝ) 
  (bank1_rate : ℝ) 
  (bank2_rate : ℝ) 
  (bank1_investment : ℝ) 
  (h1 : total_investment = 5000)
  (h2 : bank1_rate = 0.04)
  (h3 : bank2_rate = 0.065)
  (h4 : bank1_investment = 1700) :
  let bank2_investment := total_investment - bank1_investment
  let interest1 := bank1_investment * bank1_rate
  let interest2 := bank2_investment * bank2_rate
  interest1 + interest2 = 282.50 := by
sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l2985_298550


namespace NUMINAMATH_CALUDE_f_range_of_a_l2985_298599

/-- The function f(x) defined as |x-1| + |x-a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

/-- The theorem stating that if f(x) ≥ 2 for all real x, then a is in (-∞, -1] ∪ [3, +∞) -/
theorem f_range_of_a (a : ℝ) : (∀ x : ℝ, f a x ≥ 2) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_f_range_of_a_l2985_298599


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2985_298589

theorem inequality_system_solution :
  ∀ x : ℝ, (2 * x + 1 < 5 ∧ 3 - x > 2) ↔ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2985_298589


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l2985_298538

/-- Represents the budget allocation for Megatech Corporation's research and development --/
structure BudgetAllocation where
  microphotonics : Float
  home_electronics : Float
  food_additives : Float
  genetically_modified_microorganisms : Float
  industrial_lubricants : Float

/-- Calculates the degrees in a circle for a given percentage --/
def percentageToDegrees (percentage : Float) : Float :=
  percentage * 360 / 100

/-- Theorem stating that the degrees for basic astrophysics research is 43.2 --/
theorem basic_astrophysics_degrees 
  (budget : BudgetAllocation)
  (h1 : budget.microphotonics = 12)
  (h2 : budget.home_electronics = 24)
  (h3 : budget.food_additives = 15)
  (h4 : budget.genetically_modified_microorganisms = 29)
  (h5 : budget.industrial_lubricants = 8)
  : percentageToDegrees (100 - (budget.microphotonics + budget.home_electronics + 
    budget.food_additives + budget.genetically_modified_microorganisms + 
    budget.industrial_lubricants)) = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l2985_298538


namespace NUMINAMATH_CALUDE_binary_multiplication_l2985_298568

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits -/
def binary_1101 : List Bool := [true, false, true, true]
def binary_111 : List Bool := [true, true, true]
def binary_10010111 : List Bool := [true, true, true, false, true, false, false, true]

/-- Theorem stating that the product of 1101₂ and 111₂ is equal to 10010111₂ -/
theorem binary_multiplication :
  (binary_to_decimal binary_1101) * (binary_to_decimal binary_111) =
  binary_to_decimal binary_10010111 := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_l2985_298568


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2985_298561

theorem container_volume_ratio (volume_1 volume_2 : ℚ) : 
  volume_1 > 0 → volume_2 > 0 → 
  (3 / 4 : ℚ) * volume_1 = (5 / 8 : ℚ) * volume_2 → 
  volume_1 / volume_2 = (5 / 6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2985_298561


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2985_298551

theorem sum_of_coefficients (d : ℝ) (a b c : ℤ) (h : d ≠ 0) :
  (8 : ℝ) * d + 9 + 10 * d^2 + 4 * d + 3 = (a : ℝ) * d + b + (c : ℝ) * d^2 →
  a + b + c = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2985_298551


namespace NUMINAMATH_CALUDE_max_value_implies_m_l2985_298543

-- Define the function f(x)
def f (x m : ℝ) : ℝ := -x^3 + 6*x^2 + m

-- State the theorem
theorem max_value_implies_m (m : ℝ) :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x m ≤ f x_max m) ∧ (∃ (x_max : ℝ), f x_max m = 12) →
  m = -20 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l2985_298543


namespace NUMINAMATH_CALUDE_milk_parts_in_drink_A_l2985_298528

/-- Represents the composition of a drink mixture -/
structure DrinkMixture where
  milk : ℕ
  fruit_juice : ℕ

/-- Converts volume in parts to liters -/
def parts_to_liters (total_parts : ℕ) (volume_liters : ℕ) (parts : ℕ) : ℕ :=
  (volume_liters * parts) / total_parts

theorem milk_parts_in_drink_A (drink_A : DrinkMixture) (drink_B : DrinkMixture) : 
  drink_A.fruit_juice = 3 →
  drink_B.milk = 3 →
  drink_B.fruit_juice = 4 →
  parts_to_liters (drink_A.milk + drink_A.fruit_juice) 21 drink_A.fruit_juice +
    7 = parts_to_liters (drink_B.milk + drink_B.fruit_juice) 28 drink_B.fruit_juice →
  drink_A.milk = 12 := by
  sorry

end NUMINAMATH_CALUDE_milk_parts_in_drink_A_l2985_298528


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l2985_298592

def is_valid_number (n : ℕ) : Prop :=
  n % 5 = 184 ∧ n % 6 = 184 ∧ n % 9 = 184 ∧ n % 12 = 184

theorem least_number_with_remainder :
  is_valid_number 364 ∧ ∀ m : ℕ, m < 364 → ¬(is_valid_number m) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l2985_298592


namespace NUMINAMATH_CALUDE_min_value_problem_l2985_298565

theorem min_value_problem (x y : ℝ) 
  (h1 : (x - 3)^3 + 2014 * (x - 3) = 1)
  (h2 : (2 * y - 3)^3 + 2014 * (2 * y - 3) = -1) :
  ∀ z : ℝ, z = x^2 + 4 * y^2 + 4 * x → z ≥ 28 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l2985_298565


namespace NUMINAMATH_CALUDE_trash_outside_classrooms_l2985_298580

-- Define the number of classrooms
def num_classrooms : Nat := 8

-- Define the total number of trash pieces picked up
def total_trash : Nat := 1576

-- Define the number of trash pieces picked up in each classroom
def classroom_trash : Fin num_classrooms → Nat
  | ⟨0, _⟩ => 124  -- Classroom 1
  | ⟨1, _⟩ => 98   -- Classroom 2
  | ⟨2, _⟩ => 176  -- Classroom 3
  | ⟨3, _⟩ => 212  -- Classroom 4
  | ⟨4, _⟩ => 89   -- Classroom 5
  | ⟨5, _⟩ => 241  -- Classroom 6
  | ⟨6, _⟩ => 121  -- Classroom 7
  | ⟨7, _⟩ => 102  -- Classroom 8
  | ⟨n+8, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 8 n))

-- Theorem to prove
theorem trash_outside_classrooms :
  total_trash - (Finset.sum Finset.univ classroom_trash) = 413 := by
  sorry

end NUMINAMATH_CALUDE_trash_outside_classrooms_l2985_298580


namespace NUMINAMATH_CALUDE_a_2_equals_3_l2985_298572

-- Define the sequence a_n
def a : ℕ → ℕ
  | n => 3^(n-1)

-- State the theorem
theorem a_2_equals_3 : a 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_2_equals_3_l2985_298572


namespace NUMINAMATH_CALUDE_ball_in_cylinder_l2985_298510

/-- Given a horizontal cylindrical measuring cup with base radius √3 cm and a solid ball
    of radius R cm that is submerged and causes the water level to rise exactly R cm,
    prove that R = 3/2 cm. -/
theorem ball_in_cylinder (R : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * R^3 = Real.pi * 3 * R → R = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ball_in_cylinder_l2985_298510


namespace NUMINAMATH_CALUDE_pen_purchase_price_l2985_298585

/-- The purchase price of a pen. -/
def purchase_price : ℝ := sorry

/-- The profit from selling one pen for 10 rubles. -/
def profit_one_pen : ℝ := 10 - purchase_price

/-- The profit from selling three pens for 20 rubles. -/
def profit_three_pens : ℝ := 20 - 3 * purchase_price

/-- The theorem stating that the purchase price of each pen is 5 rubles. -/
theorem pen_purchase_price :
  (profit_one_pen = profit_three_pens) → purchase_price = 5 := by sorry

end NUMINAMATH_CALUDE_pen_purchase_price_l2985_298585


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2985_298505

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2985_298505


namespace NUMINAMATH_CALUDE_tangent_points_on_line_l2985_298519

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + (y - 1)^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define the line l
def l (x y : ℝ) : Prop := x = -2 ∨ 3*x - 4*y + 18 = 0

-- Define point A
def A : ℝ × ℝ := (-2, 3)

-- Define the chord length condition
def chord_length (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12 -- (2√3)^2 = 12

-- Define the tangent length equality condition
def equal_tangents (x y : ℝ) : Prop :=
  (x + 3)^2 + (y - 1)^2 - 4 = (x - 3)^2 + (y - 4)^2 - 1

-- Main theorem
theorem tangent_points_on_line :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    l x₁ y₁ ∧ l x₂ y₂ ∧
    equal_tangents x₁ y₁ ∧ equal_tangents x₂ y₂ ∧
    ((x₁ = -2 ∧ y₁ = 7) ∨ (x₁ = -6/11 ∧ y₁ = 45/11)) ∧
    ((x₂ = -2 ∧ y₂ = 7) ∨ (x₂ = -6/11 ∧ y₂ = 45/11)) ∧
    x₁ ≠ x₂ := by
  sorry

end NUMINAMATH_CALUDE_tangent_points_on_line_l2985_298519


namespace NUMINAMATH_CALUDE_basket_probability_l2985_298522

-- Define the total number of shots
def total_shots : ℕ := 5

-- Define the number of successful shots
def successful_shots : ℕ := 3

-- Define the number of unsuccessful shots
def unsuccessful_shots : ℕ := 2

-- Define the probability of second and third shots being successful
def prob_second_third_successful : ℚ := 3/10

-- Theorem statement
theorem basket_probability :
  (total_shots = successful_shots + unsuccessful_shots) →
  (prob_second_third_successful = 3/10) :=
by sorry

end NUMINAMATH_CALUDE_basket_probability_l2985_298522


namespace NUMINAMATH_CALUDE_restaurant_glasses_problem_l2985_298598

theorem restaurant_glasses_problem (x : ℕ) :
  let small_box_count : ℕ := 1
  let large_box_count : ℕ := 16 + small_box_count
  let total_boxes : ℕ := small_box_count + large_box_count
  let glasses_per_large_box : ℕ := 16
  let average_glasses_per_box : ℕ := 15
  let total_glasses : ℕ := 480
  (total_boxes * average_glasses_per_box + x + large_box_count * glasses_per_large_box = total_glasses) →
  x = 224 := by
sorry

end NUMINAMATH_CALUDE_restaurant_glasses_problem_l2985_298598


namespace NUMINAMATH_CALUDE_sin_675_degrees_l2985_298512

theorem sin_675_degrees : Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_675_degrees_l2985_298512


namespace NUMINAMATH_CALUDE_original_price_after_discounts_l2985_298508

/-- 
Given an article sold at $126 after two successive discounts of 10% and 20%,
prove that its original price was $175.
-/
theorem original_price_after_discounts (final_price : ℝ) 
  (h1 : final_price = 126) 
  (discount1 : ℝ) (h2 : discount1 = 0.1)
  (discount2 : ℝ) (h3 : discount2 = 0.2) : 
  ∃ (original_price : ℝ), 
    original_price = 175 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) := by
  sorry

end NUMINAMATH_CALUDE_original_price_after_discounts_l2985_298508


namespace NUMINAMATH_CALUDE_sequence_increasing_iff_last_term_l2985_298577

theorem sequence_increasing_iff_last_term (a : ℕ → ℝ) : 
  (∀ n : ℕ, n ≥ 1 ∧ n < 64 → |a (n + 1) - a n| = n) →
  a 1 = 2 →
  (∀ n : ℕ, n ≥ 1 ∧ n < 64 → a (n + 1) > a n) ↔ a 64 = 2018 :=
by sorry

end NUMINAMATH_CALUDE_sequence_increasing_iff_last_term_l2985_298577


namespace NUMINAMATH_CALUDE_arrangementsWithRestrictionsCorrect_l2985_298566

/-- The number of ways to arrange 5 distinct items in a row with restrictions -/
def arrangementsWithRestrictions : ℕ :=
  let n := 5  -- total number of items
  let mustBeAdjacent := 2  -- number of items that must be adjacent
  let cannotBeAdjacent := 2  -- number of items that cannot be adjacent
  24  -- the result we want to prove

theorem arrangementsWithRestrictionsCorrect :
  arrangementsWithRestrictions = 24 :=
by sorry

end NUMINAMATH_CALUDE_arrangementsWithRestrictionsCorrect_l2985_298566


namespace NUMINAMATH_CALUDE_abcd_product_magnitude_l2985_298530

theorem abcd_product_magnitude (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h_eq : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/d ∧ c + 1/d = d + 1/a) :
  |a * b * c * d| = 1 := by
sorry

end NUMINAMATH_CALUDE_abcd_product_magnitude_l2985_298530


namespace NUMINAMATH_CALUDE_richard_twice_scott_age_david_age_four_years_ago_future_years_correct_l2985_298553

/-- The number of years in the future when Richard will be twice as old as Scott -/
def future_years : ℕ := 8

/-- David's current age -/
def david_age : ℕ := 14

/-- Richard's current age -/
def richard_age : ℕ := david_age + 6

/-- Scott's current age -/
def scott_age : ℕ := david_age - 8

theorem richard_twice_scott_age : 
  richard_age + future_years = 2 * (scott_age + future_years) :=
by sorry

theorem david_age_four_years_ago : 
  david_age = 10 + 4 :=
by sorry

theorem future_years_correct : 
  ∃ (y : ℕ), y = future_years ∧ richard_age + y = 2 * (scott_age + y) :=
by sorry

end NUMINAMATH_CALUDE_richard_twice_scott_age_david_age_four_years_ago_future_years_correct_l2985_298553


namespace NUMINAMATH_CALUDE_race_distance_l2985_298595

/-- The race problem -/
theorem race_distance (t_a t_b : ℝ) (d_diff : ℝ) (h1 : t_a = 20) (h2 : t_b = 25) (h3 : d_diff = 16) :
  ∃ d : ℝ, d > 0 ∧ d / t_a * t_b = d + d_diff := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l2985_298595


namespace NUMINAMATH_CALUDE_raja_income_proof_l2985_298582

/-- Raja's monthly income in rupees -/
def monthly_income : ℝ := 37500

/-- Percentage spent on household items -/
def household_percentage : ℝ := 35

/-- Percentage spent on clothes -/
def clothes_percentage : ℝ := 20

/-- Percentage spent on medicines -/
def medicine_percentage : ℝ := 5

/-- Amount saved in rupees -/
def savings : ℝ := 15000

theorem raja_income_proof :
  monthly_income * (1 - (household_percentage + clothes_percentage + medicine_percentage) / 100) = savings := by
  sorry

#check raja_income_proof

end NUMINAMATH_CALUDE_raja_income_proof_l2985_298582


namespace NUMINAMATH_CALUDE_smallest_angle_range_l2985_298548

theorem smallest_angle_range (A B C : Real) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) 
  (h4 : A + B + C = 180) : 
  ∃ α : Real, (α = min A (min B C)) ∧ (0 < α) ∧ (α ≤ 60) := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_range_l2985_298548


namespace NUMINAMATH_CALUDE_smallest_x_value_l2985_298558

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = (y : ℚ) / ((250 : ℚ) + x)) :
  x ≥ 2 ∧ ∃ (y' : ℕ+), (3 : ℚ) / 4 = (y' : ℚ) / ((250 : ℚ) + 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2985_298558


namespace NUMINAMATH_CALUDE_certain_number_problem_l2985_298535

theorem certain_number_problem (p q : ℕ) (x : ℤ) : 
  p > 1 → q > 1 → p + q = 36 → x * (p + 1) = 21 * (q + 1) → x = 245 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2985_298535


namespace NUMINAMATH_CALUDE_max_min_x_plus_y_l2985_298547

theorem max_min_x_plus_y :
  ∀ x y : ℝ,
  x + y = Real.sqrt (2 * x - 1) + Real.sqrt (4 * y + 3) →
  (x + y ≤ 3 + Real.sqrt (21 / 2)) ∧
  (x + y ≥ 1 + Real.sqrt (3 / 2)) ∧
  (∃ x₁ y₁ : ℝ, x₁ + y₁ = Real.sqrt (2 * x₁ - 1) + Real.sqrt (4 * y₁ + 3) ∧ x₁ + y₁ = 3 + Real.sqrt (21 / 2)) ∧
  (∃ x₂ y₂ : ℝ, x₂ + y₂ = Real.sqrt (2 * x₂ - 1) + Real.sqrt (4 * y₂ + 3) ∧ x₂ + y₂ = 1 + Real.sqrt (3 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_max_min_x_plus_y_l2985_298547


namespace NUMINAMATH_CALUDE_pool_filling_problem_l2985_298549

/-- The pool filling problem -/
theorem pool_filling_problem (pool_capacity : ℝ) (both_valves_time : ℝ) (first_valve_time : ℝ) :
  pool_capacity = 12000 →
  both_valves_time = 48 →
  first_valve_time = 120 →
  let first_valve_rate := pool_capacity / first_valve_time
  let both_valves_rate := pool_capacity / both_valves_time
  let second_valve_rate := both_valves_rate - first_valve_rate
  second_valve_rate - first_valve_rate = 50 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_problem_l2985_298549


namespace NUMINAMATH_CALUDE_abigail_money_problem_l2985_298593

/-- Proves that Abigail had $11 at the start of the day given the conditions -/
theorem abigail_money_problem :
  ∀ (initial_money : ℕ),
    initial_money - 2 - 6 = 3 →
    initial_money = 11 := by
  sorry

end NUMINAMATH_CALUDE_abigail_money_problem_l2985_298593


namespace NUMINAMATH_CALUDE_product_inequality_l2985_298563

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum_prod : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l2985_298563


namespace NUMINAMATH_CALUDE_prob_sum_nine_l2985_298537

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The sum we're looking for -/
def targetSum : ℕ := 9

/-- The set of all possible outcomes when throwing two dice -/
def allOutcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range numSides) (Finset.range numSides)

/-- The set of favorable outcomes (pairs that sum to targetSum) -/
def favorableOutcomes : Finset (ℕ × ℕ) :=
  allOutcomes.filter (fun p => p.1 + p.2 = targetSum)

/-- The probability of obtaining the target sum -/
def probability : ℚ :=
  (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ)

theorem prob_sum_nine : probability = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_nine_l2985_298537


namespace NUMINAMATH_CALUDE_transistors_in_2010_l2985_298587

/-- The number of transistors in a typical CPU doubles every three years -/
def doubling_period : ℕ := 3

/-- The number of transistors in a typical CPU in 1992 -/
def initial_transistors : ℕ := 2000000

/-- The year from which we start counting -/
def initial_year : ℕ := 1992

/-- The year for which we want to calculate the number of transistors -/
def target_year : ℕ := 2010

/-- Calculates the number of transistors in a given year -/
def transistors_in_year (year : ℕ) : ℕ :=
  initial_transistors * 2^((year - initial_year) / doubling_period)

theorem transistors_in_2010 :
  transistors_in_year target_year = 128000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_2010_l2985_298587


namespace NUMINAMATH_CALUDE_cattle_purchase_cost_l2985_298509

theorem cattle_purchase_cost 
  (num_cattle : ℕ) 
  (feeding_cost_ratio : ℝ) 
  (weight_per_cattle : ℝ) 
  (selling_price_per_pound : ℝ) 
  (profit : ℝ) 
  (h1 : num_cattle = 100)
  (h2 : feeding_cost_ratio = 1.2)
  (h3 : weight_per_cattle = 1000)
  (h4 : selling_price_per_pound = 2)
  (h5 : profit = 112000) : 
  ∃ (purchase_cost : ℝ), purchase_cost = 40000 ∧ 
    num_cattle * weight_per_cattle * selling_price_per_pound - 
    (purchase_cost * (1 + (feeding_cost_ratio - 1))) = profit :=
by sorry

end NUMINAMATH_CALUDE_cattle_purchase_cost_l2985_298509


namespace NUMINAMATH_CALUDE_perpendicular_dot_product_zero_l2985_298503

/-- Given a point P on the curve y = x + 2/x for x > 0, prove that the dot product
    of PA and PB is zero, where A is the foot of the perpendicular from P to y = x,
    and B is the foot of the perpendicular from P to x = 0. -/
theorem perpendicular_dot_product_zero (x : ℝ) (hx : x > 0) :
  let P : ℝ × ℝ := (x, x + 2/x)
  let A : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
  let B : ℝ × ℝ := (0, x + 2/x)
  let PA : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)
  let PB : ℝ × ℝ := (B.1 - P.1, B.2 - P.2)
  PA.1 * PB.1 + PA.2 * PB.2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_dot_product_zero_l2985_298503


namespace NUMINAMATH_CALUDE_cake_calories_l2985_298533

/-- Proves that given a cake with 8 slices and a pan of 6 brownies where each brownie has 375 calories,
    if the cake has 526 more calories than the pan of brownies, then each slice of the cake has 347 calories. -/
theorem cake_calories (cake_slices : ℕ) (brownie_count : ℕ) (brownie_calories : ℕ) (extra_calories : ℕ) :
  cake_slices = 8 →
  brownie_count = 6 →
  brownie_calories = 375 →
  extra_calories = 526 →
  (cake_slices * (brownie_count * brownie_calories + extra_calories) / cake_slices : ℚ) = 347 := by
  sorry

end NUMINAMATH_CALUDE_cake_calories_l2985_298533


namespace NUMINAMATH_CALUDE_fraction_value_l2985_298557

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) 
  (h4 : b ≠ 0) 
  (h5 : d ≠ 0) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l2985_298557


namespace NUMINAMATH_CALUDE_xy_equals_five_l2985_298506

theorem xy_equals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdistinct : x ≠ y)
  (h : x + 5 / x = y + 5 / y) : x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_five_l2985_298506


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2985_298556

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 1 = 2 ∧
  a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2985_298556


namespace NUMINAMATH_CALUDE_function_equality_l2985_298526

-- Define the function type
def FunctionType := ℝ → ℝ → ℝ → ℝ

-- State the theorem
theorem function_equality (f : FunctionType) 
  (h : ∀ x y z : ℝ, f x y z = 2 * f z x y) : 
  ∀ x y z : ℝ, f x y z = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2985_298526


namespace NUMINAMATH_CALUDE_intersection_values_eq_two_eight_l2985_298513

/-- The set of positive real values k for which |z - 4| = 3|z + 4| and |z| = k have exactly one solution in ℂ -/
def intersection_values : Set ℝ :=
  {k : ℝ | k > 0 ∧ ∃! (z : ℂ), Complex.abs (z - 4) = 3 * Complex.abs (z + 4) ∧ Complex.abs z = k}

/-- Theorem stating that the intersection_values set contains exactly 2 and 8 -/
theorem intersection_values_eq_two_eight : intersection_values = {2, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_values_eq_two_eight_l2985_298513


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2985_298527

/-- Given an ellipse with equation x²/(k+8) + y²/9 = 1, foci on the y-axis, 
    and eccentricity 1/2, prove that k = -5/4 -/
theorem ellipse_eccentricity (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (k + 8) + y^2 / 9 = 1) →  -- ellipse equation
  (∃ c : ℝ, c > 0 ∧ ∀ x : ℝ, x^2 / (k + 8) + (y - c)^2 / 9 = 1 ∧ 
                              x^2 / (k + 8) + (y + c)^2 / 9 = 1) →  -- foci on y-axis
  (let a := 3; let c := a / 2; c / a = 1 / 2) →  -- eccentricity = 1/2
  k = -5/4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2985_298527


namespace NUMINAMATH_CALUDE_ratio_change_proof_l2985_298586

/-- Represents the ratio of bleach, detergent, and water in a solution -/
structure SolutionRatio where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- The original solution ratio -/
def original_ratio : SolutionRatio := ⟨2, 40, 100⟩

/-- The altered solution ratio -/
def altered_ratio : SolutionRatio := ⟨6, 60, 300⟩

/-- The factor by which the ratio of detergent to water changes -/
def ratio_change_factor : ℚ := 2

theorem ratio_change_proof : 
  (original_ratio.detergent / original_ratio.water) / 
  (altered_ratio.detergent / altered_ratio.water) = ratio_change_factor := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_proof_l2985_298586


namespace NUMINAMATH_CALUDE_set_equalities_l2985_298562

-- Definition for even numbers
def IsEven (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k

-- Set 1
def Set1 : Set ℤ := {x | -1 ≤ x ∧ x < 5 ∧ IsEven x}

-- Set 2
def Set2 : Set ℤ := {0, 1, 2, 3, 4, 5}

-- Set 3
def Set3 : Set ℝ := {x | |x| = 1}

theorem set_equalities :
  (Set1 = {0, 2, 4}) ∧
  (Set2 = {x : ℤ | 0 ≤ x ∧ x ≤ 5}) ∧
  (Set3 = {x : ℝ | |x| = 1}) := by
  sorry


end NUMINAMATH_CALUDE_set_equalities_l2985_298562


namespace NUMINAMATH_CALUDE_prime_order_existence_l2985_298524

theorem prime_order_existence (p : ℕ) (hp : Prime p) :
  ∃ k : ℤ, (∀ m : ℕ, m < p - 1 → k ^ m % p ≠ 1) ∧
            k ^ (p - 1) % p = 1 ∧
            (∀ m : ℕ, m < p * (p - 1) → k ^ m % (p ^ 2) ≠ 1) ∧
            k ^ (p * (p - 1)) % (p ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_order_existence_l2985_298524


namespace NUMINAMATH_CALUDE_turban_price_gopi_servant_l2985_298544

/-- Calculates the price of a turban given salary and payment conditions. -/
def turban_price (initial_yearly_salary : ℚ) (raise_percentage : ℚ) (months_worked : ℕ) (final_cash_payment : ℚ) : ℚ :=
  let initial_monthly_salary := initial_yearly_salary / 12
  let raised_monthly_salary := initial_monthly_salary * (1 + raise_percentage)
  let total_salary := 
    if months_worked ≤ 6 then
      initial_monthly_salary * months_worked
    else
      (initial_monthly_salary * 6) + (raised_monthly_salary * (months_worked - 6))
  total_salary - final_cash_payment

/-- The price of the turban in Gopi's servant scenario. -/
theorem turban_price_gopi_servant : 
  turban_price 90 (1/10) 9 65 = 475/100 := by sorry

end NUMINAMATH_CALUDE_turban_price_gopi_servant_l2985_298544


namespace NUMINAMATH_CALUDE_max_daily_revenue_l2985_298578

def sales_price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then -t + 100
  else if 25 ≤ t ∧ t ≤ 30 then t + 20
  else 0

def daily_sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def daily_revenue (t : ℕ) : ℝ := sales_price t * daily_sales_volume t

theorem max_daily_revenue :
  (∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_revenue t = 1125) ∧
  (∀ t : ℕ, 0 < t ∧ t ≤ 30 → daily_revenue t ≤ 1125) ∧
  (∀ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_revenue t = 1125 → t = 25) :=
sorry

end NUMINAMATH_CALUDE_max_daily_revenue_l2985_298578


namespace NUMINAMATH_CALUDE_tori_trash_total_l2985_298507

/-- The number of pieces of trash Tori picked up in the classrooms -/
def classroom_trash : ℕ := 344

/-- The number of pieces of trash Tori picked up outside the classrooms -/
def outside_trash : ℕ := 1232

/-- The total number of pieces of trash Tori picked up last week -/
def total_trash : ℕ := classroom_trash + outside_trash

/-- Theorem stating that the total number of pieces of trash Tori picked up is 1576 -/
theorem tori_trash_total : total_trash = 1576 := by
  sorry

end NUMINAMATH_CALUDE_tori_trash_total_l2985_298507


namespace NUMINAMATH_CALUDE_larger_cuboid_height_l2985_298594

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem larger_cuboid_height 
  (small : CuboidDimensions)
  (large_length : ℝ)
  (large_width : ℝ)
  (h_small_dims : small = { length := 5, width := 4, height := 3 })
  (h_large_dims : large_length = 16 ∧ large_width = 10)
  (h_count : 32 * cuboidVolume small = cuboidVolume { length := large_length, width := large_width, height := 12 }) :
  ∃ (large : CuboidDimensions), large.length = large_length ∧ large.width = large_width ∧ large.height = 12 :=
sorry

end NUMINAMATH_CALUDE_larger_cuboid_height_l2985_298594


namespace NUMINAMATH_CALUDE_anais_toy_difference_l2985_298518

/-- Given a total number of toys and the number of toys Kamari has,
    calculates how many more toys Anais has than Kamari. -/
def toyDifference (total : ℕ) (kamariToys : ℕ) : ℕ :=
  total - kamariToys - kamariToys

/-- Proves that given the specific conditions, Anais has 30 more toys than Kamari. -/
theorem anais_toy_difference :
  let total := 160
  let kamariToys := 65
  toyDifference total kamariToys = 30 := by
  sorry

#eval toyDifference 160 65  -- Should output 30

end NUMINAMATH_CALUDE_anais_toy_difference_l2985_298518


namespace NUMINAMATH_CALUDE_expression_value_at_12_l2985_298560

theorem expression_value_at_12 :
  let y : ℝ := 12
  (y^9 - 27*y^6 + 243*y^3 - 729) / (y^3 - 9) = 5082647079 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_12_l2985_298560


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2985_298532

/-- The asymptote equation of the hyperbola x² - y²/3 = -1 is y = ±√3x -/
theorem hyperbola_asymptote :
  let h : ℝ × ℝ → ℝ := λ (x, y) ↦ x^2 - y^2/3 + 1
  ∃ (k : ℝ), k = Real.sqrt 3 ∧
    (∀ ε > 0, ∃ M > 0, ∀ x y, 
      x^2 + y^2 > M^2 → h (x, y) = 0 → |y - k*x| < ε*|x| ∨ |y + k*x| < ε*|x|) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2985_298532


namespace NUMINAMATH_CALUDE_distinct_polygons_from_circle_points_l2985_298571

theorem distinct_polygons_from_circle_points (n : ℕ) (h : n = 12) : 
  (2^n : ℕ) - (1 + n + n*(n-1)/2) = 4017 := by
  sorry

end NUMINAMATH_CALUDE_distinct_polygons_from_circle_points_l2985_298571


namespace NUMINAMATH_CALUDE_inequality_proof_l2985_298584

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b * c / a + c * a / b + a * b / c ≥ a + b + c) ∧
  (a + b + c = 1 → (1 - a) / a + (1 - b) / b + (1 - c) / c ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2985_298584


namespace NUMINAMATH_CALUDE_point_A_satisfies_condition_l2985_298552

/-- The line on which point P moves -/
def line_P (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The line that always passes through point A -/
def line_A (a b x y : ℝ) : Prop := 3 * a * x + 4 * b * y = 12

/-- Point A -/
def point_A : ℝ × ℝ := (1, 1)

theorem point_A_satisfies_condition :
  ∀ a b : ℝ, line_P a b → line_A a b (point_A.1) (point_A.2) :=
sorry

end NUMINAMATH_CALUDE_point_A_satisfies_condition_l2985_298552


namespace NUMINAMATH_CALUDE_solution_sets_equality_l2985_298559

theorem solution_sets_equality (b c : ℝ) : 
  (∀ x : ℝ, |2*x - 3| < 5 ↔ -x^2 + b*x + c > 0) → b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_equality_l2985_298559


namespace NUMINAMATH_CALUDE_complete_residue_system_l2985_298564

theorem complete_residue_system (m : ℕ) (x : Fin m → ℤ) 
  (h_incongruent : ∀ i j : Fin m, i ≠ j → x i % m ≠ x j % m) :
  ∀ y : ℤ, ∃ i : Fin m, y % m = x i % m :=
by sorry

end NUMINAMATH_CALUDE_complete_residue_system_l2985_298564


namespace NUMINAMATH_CALUDE_tangency_point_l2985_298540

def parabola1 (x y : ℝ) : Prop := y = 2 * x^2 + 10 * x + 14

def parabola2 (x y : ℝ) : Prop := x = 4 * y^2 + 16 * y + 68

def point_of_tangency (x y : ℝ) : Prop :=
  parabola1 x y ∧ parabola2 x y

theorem tangency_point : 
  point_of_tangency (-9/4) (-15/8) := by sorry

end NUMINAMATH_CALUDE_tangency_point_l2985_298540


namespace NUMINAMATH_CALUDE_least_cars_per_day_l2985_298515

/-- Represents a mechanic's work details -/
structure Mechanic where
  rate : ℕ  -- cars per hour
  hours : ℕ  -- total work hours
  lunch_break : ℕ  -- lunch break in hours
  additional_break : ℕ  -- additional break in half-hours

/-- Calculates the number of cars a mechanic can service in a day -/
def cars_serviced (m : Mechanic) : ℕ :=
  m.rate * (m.hours - m.lunch_break - m.additional_break / 2)

/-- The three mechanics at the oil spot -/
def paul : Mechanic := { rate := 2, hours := 8, lunch_break := 1, additional_break := 1 }
def jack : Mechanic := { rate := 3, hours := 6, lunch_break := 1, additional_break := 1 }
def sam : Mechanic := { rate := 4, hours := 5, lunch_break := 1, additional_break := 0 }

/-- Theorem stating the least number of cars the mechanics can finish together per workday -/
theorem least_cars_per_day : 
  cars_serviced paul + cars_serviced jack + cars_serviced sam = 42 := by
  sorry

end NUMINAMATH_CALUDE_least_cars_per_day_l2985_298515


namespace NUMINAMATH_CALUDE_intersection_when_p_zero_union_equals_B_implies_p_range_l2985_298596

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B with parameter p
def B (p : ℝ) : Set ℝ := {x | |x - p| > 1}

-- Theorem for part (1)
theorem intersection_when_p_zero :
  A ∩ B 0 = {x | 1 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem union_equals_B_implies_p_range (p : ℝ) :
  A ∪ B p = B p → p ≤ -2 ∨ p ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_p_zero_union_equals_B_implies_p_range_l2985_298596


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2985_298502

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x < 2}
def N : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2985_298502


namespace NUMINAMATH_CALUDE_piano_lesson_cost_l2985_298520

theorem piano_lesson_cost (lesson_duration : Real) (total_hours : Real) (total_cost : Real) :
  lesson_duration = 1.5 →
  total_hours = 18 →
  total_cost = 360 →
  lesson_duration * (total_cost / total_hours) = 30 := by
sorry

end NUMINAMATH_CALUDE_piano_lesson_cost_l2985_298520


namespace NUMINAMATH_CALUDE_girls_in_selection_l2985_298514

theorem girls_in_selection (n : ℕ) : 
  (1 - (Nat.choose 3 3 : ℚ) / (Nat.choose (3 + n) 3 : ℚ) = 34 / 35) → n = 4 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_selection_l2985_298514


namespace NUMINAMATH_CALUDE_grocery_expense_l2985_298545

def monthly_expenses (rent milk education petrol misc groceries savings : ℕ) : Prop :=
  let total_salary := savings * 10
  rent + milk + education + petrol + misc + groceries + savings = total_salary

theorem grocery_expense : 
  ∃ (groceries : ℕ), monthly_expenses 5000 1500 2500 2000 5200 groceries 2300 ∧ groceries = 4500 :=
by sorry

end NUMINAMATH_CALUDE_grocery_expense_l2985_298545


namespace NUMINAMATH_CALUDE_first_number_proof_l2985_298569

theorem first_number_proof (x : ℝ) : 
  (((10 + 60 + 35) / 3) + 5 = (x + 40 + 60) / 3) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l2985_298569


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2985_298504

-- Define the line
def line (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0

-- Define the third quadrant
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem: The line does not pass through the third quadrant
theorem line_not_in_third_quadrant : 
  ¬ ∃ (x y : ℝ), line x y ∧ third_quadrant x y := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2985_298504


namespace NUMINAMATH_CALUDE_lotus_pollen_diameter_scientific_notation_l2985_298523

theorem lotus_pollen_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0025 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.5 ∧ n = -3 := by
  sorry

end NUMINAMATH_CALUDE_lotus_pollen_diameter_scientific_notation_l2985_298523


namespace NUMINAMATH_CALUDE_powerjet_45min_output_l2985_298574

/-- A pump that pumps water at a given rate. -/
structure Pump where
  rate : ℝ  -- Gallons per hour

/-- Calculates the amount of water pumped in a given time. -/
def water_pumped (p : Pump) (time : ℝ) : ℝ :=
  p.rate * time

/-- Theorem: A pump that pumps 420 gallons per hour will pump 315 gallons in 45 minutes. -/
theorem powerjet_45min_output (p : Pump) (h : p.rate = 420) : 
  water_pumped p (45 / 60) = 315 := by
  sorry

#check powerjet_45min_output

end NUMINAMATH_CALUDE_powerjet_45min_output_l2985_298574


namespace NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l2985_298525

/-- Given two plane vectors a and b that are perpendicular, prove that their difference has a magnitude of 2 -/
theorem perpendicular_vectors_difference_magnitude
  (x : ℝ)
  (a : ℝ × ℝ := (4^x, 2^x))
  (b : ℝ × ℝ := (1, (2^x - 2)/2^x))
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  ‖(a.1 - b.1, a.2 - b.2)‖ = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l2985_298525


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2985_298521

/-- Given vectors a and b in ℝ², prove that if a is perpendicular to b, then m = 2 -/
theorem perpendicular_vectors_m_value 
  (a b : ℝ × ℝ) 
  (h1 : a = (-2, 3)) 
  (h2 : b = (3, m)) 
  (h3 : a.fst * b.fst + a.snd * b.snd = 0) : 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2985_298521


namespace NUMINAMATH_CALUDE_equation_a_graph_l2985_298529

theorem equation_a_graph (x y : ℝ) :
  (x - 2) * (y + 3) = 0 ↔ (x = 2 ∨ y = -3) :=
sorry

end NUMINAMATH_CALUDE_equation_a_graph_l2985_298529


namespace NUMINAMATH_CALUDE_intersection_line_l2985_298567

-- Define the first circle
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

-- Define the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line
def line (x y : ℝ) : Prop := x - y - 3 = 0

-- Theorem statement
theorem intersection_line : 
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_l2985_298567


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2985_298517

theorem inequality_equivalence (x : ℝ) : (x - 1) / 2 + 1 < (4 * x - 5) / 3 ↔ x > 13 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2985_298517


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2985_298539

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ - 1 = 0 ∧ k * x₂^2 - 2 * x₂ - 1 = 0) ↔
  (k > -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2985_298539


namespace NUMINAMATH_CALUDE_complement_M_l2985_298573

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x > 2 ∨ x < 0}

theorem complement_M : U \ M = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_l2985_298573


namespace NUMINAMATH_CALUDE_volleyball_contributions_l2985_298554

theorem volleyball_contributions :
  ∀ (x y z : ℝ),
  -- Condition 1: Third boy contributed 6.4 rubles more than the first boy
  z = x + 6.4 →
  -- Condition 2: Half of first boy's contribution equals one-third of second boy's
  (1/2) * x = (1/3) * y →
  -- Condition 3: Half of first boy's contribution equals one-fourth of third boy's
  (1/2) * x = (1/4) * z →
  -- Conclusion: The contributions are 6.4, 9.6, and 12.8 rubles
  x = 6.4 ∧ y = 9.6 ∧ z = 12.8 :=
by
  sorry


end NUMINAMATH_CALUDE_volleyball_contributions_l2985_298554


namespace NUMINAMATH_CALUDE_eggs_from_martha_l2985_298570

/-- The number of chickens Trevor collects eggs from -/
def num_chickens : ℕ := 4

/-- The number of eggs Trevor collected from Gertrude -/
def eggs_from_gertrude : ℕ := 4

/-- The number of eggs Trevor collected from Blanche -/
def eggs_from_blanche : ℕ := 3

/-- The number of eggs Trevor collected from Nancy -/
def eggs_from_nancy : ℕ := 2

/-- The number of eggs Trevor dropped -/
def eggs_dropped : ℕ := 2

/-- The number of eggs Trevor had left after dropping some -/
def eggs_left : ℕ := 9

/-- Theorem stating that Trevor got 2 eggs from Martha -/
theorem eggs_from_martha : 
  eggs_from_gertrude + eggs_from_blanche + eggs_from_nancy + 2 = eggs_left + eggs_dropped :=
by sorry

end NUMINAMATH_CALUDE_eggs_from_martha_l2985_298570


namespace NUMINAMATH_CALUDE_parabola_symmetry_l2985_298546

/-- Given a parabola y = 2x^2 - 4x - 5 translated 3 units left and 2 units up to obtain parabola C,
    the equation of the parabola symmetric to C about the y-axis is y = 2x^2 - 8x + 3 -/
theorem parabola_symmetry (x y : ℝ) :
  let original := fun x => 2 * x^2 - 4 * x - 5
  let translated := fun x => original (x + 3) + 2
  let symmetric := fun x => translated (-x)
  symmetric x = 2 * x^2 - 8 * x + 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l2985_298546


namespace NUMINAMATH_CALUDE_overall_percentage_calculation_l2985_298500

theorem overall_percentage_calculation (grade1 grade2 grade3 : ℚ) :
  grade1 = 50 / 100 →
  grade2 = 60 / 100 →
  grade3 = 70 / 100 →
  (grade1 + grade2 + grade3) / 3 = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_overall_percentage_calculation_l2985_298500


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_equality_condition_l2985_298536

theorem sum_of_fractions_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c :=
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_equality_condition_l2985_298536


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2985_298597

/-- Given a geometric sequence where the first term is m and the fourth term is n,
    prove that the product of the second and fifth terms is equal to n⋅∛(m⋅n²). -/
theorem geometric_sequence_product (m n : ℝ) (h : m > 0) (h' : n > 0) : 
  let q := (n / m) ^ (1/3)
  let second_term := m * q
  let fifth_term := m * q^4
  second_term * fifth_term = n * (m * n^2)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2985_298597


namespace NUMINAMATH_CALUDE_farmer_children_count_l2985_298591

theorem farmer_children_count : ∃ n : ℕ,
  (n ≠ 0) ∧
  (15 * n - 8 - 7 = 60) ∧
  (n = 5) := by
  sorry

end NUMINAMATH_CALUDE_farmer_children_count_l2985_298591


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l2985_298588

theorem spencer_walk_distance (distance_house_to_library : ℝ) 
                               (distance_library_to_post_office : ℝ) 
                               (distance_post_office_to_home : ℝ) 
                               (h1 : distance_house_to_library = 0.3)
                               (h2 : distance_library_to_post_office = 0.1)
                               (h3 : distance_post_office_to_home = 0.4) :
  distance_house_to_library + distance_library_to_post_office + distance_post_office_to_home = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_spencer_walk_distance_l2985_298588


namespace NUMINAMATH_CALUDE_largest_power_2024_divides_factorial_l2985_298531

def largest_power_dividing_factorial (n : ℕ) : ℕ :=
  min (sum_floor_div n 11) (sum_floor_div n 23)
where
  sum_floor_div (n p : ℕ) : ℕ :=
    (n / p) + (n / (p * p))

theorem largest_power_2024_divides_factorial :
  largest_power_dividing_factorial 2024 = 91 := by
sorry

#eval largest_power_dividing_factorial 2024

end NUMINAMATH_CALUDE_largest_power_2024_divides_factorial_l2985_298531


namespace NUMINAMATH_CALUDE_haley_spent_32_l2985_298581

/-- The amount Haley spent on concert tickets -/
def haley_spent (ticket_price : ℕ) (self_and_friends : ℕ) (extra : ℕ) : ℕ :=
  (self_and_friends + extra) * ticket_price

/-- Proof that Haley spent $32 on concert tickets -/
theorem haley_spent_32 :
  haley_spent 4 3 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_haley_spent_32_l2985_298581


namespace NUMINAMATH_CALUDE_books_calculation_initial_books_count_l2985_298579

/-- The number of books initially on the shelf -/
def initial_books : ℕ := sorry

/-- The number of books Marta added to the shelf -/
def books_added : ℕ := 10

/-- The final number of books on the shelf -/
def final_books : ℕ := 48

/-- Theorem stating that the initial number of books plus the added books equals the final number of books -/
theorem books_calculation : initial_books + books_added = final_books := by sorry

/-- Theorem proving that the initial number of books is 38 -/
theorem initial_books_count : initial_books = 38 := by sorry

end NUMINAMATH_CALUDE_books_calculation_initial_books_count_l2985_298579


namespace NUMINAMATH_CALUDE_textbook_order_cost_l2985_298516

/-- The total cost of ordering English and geography textbooks -/
def total_cost (english_count : ℕ) (geography_count : ℕ) (english_price : ℚ) (geography_price : ℚ) : ℚ :=
  english_count * english_price + geography_count * geography_price

/-- Theorem stating that the total cost of the textbook order is $630.00 -/
theorem textbook_order_cost :
  total_cost 35 35 (7.5 : ℚ) (10.5 : ℚ) = 630 := by
  sorry

end NUMINAMATH_CALUDE_textbook_order_cost_l2985_298516
