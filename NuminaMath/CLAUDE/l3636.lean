import Mathlib

namespace eliminate_denominators_l3636_363653

theorem eliminate_denominators (x : ℝ) :
  (1 / 2 * (x + 1) = 1 - 1 / 3 * x) →
  (3 * (x + 1) = 6 - 2 * x) :=
by sorry

end eliminate_denominators_l3636_363653


namespace zeroes_elimination_theorem_l3636_363680

/-- A step in the digit replacement process. -/
structure Step where
  digits_removed : ℕ := 2

/-- The initial state of the blackboard. -/
structure Blackboard where
  zeroes : ℕ
  ones : ℕ

/-- The final state after all steps are completed. -/
structure FinalState where
  steps : ℕ
  remaining_ones : ℕ

/-- The theorem to be proved. -/
theorem zeroes_elimination_theorem (initial : Blackboard) (final : FinalState) :
  initial.zeroes = 150 ∧
  final.steps = 76 ∧
  final.remaining_ones = initial.ones - 2 →
  initial.ones = 78 :=
by sorry

end zeroes_elimination_theorem_l3636_363680


namespace trig_value_comparison_l3636_363662

theorem trig_value_comparison :
  let a : ℝ := Real.tan (-7 * π / 6)
  let b : ℝ := Real.cos (23 * π / 4)
  let c : ℝ := Real.sin (-33 * π / 4)
  b > a ∧ a > c := by sorry

end trig_value_comparison_l3636_363662


namespace tangent_line_x_intercept_l3636_363636

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

-- Theorem statement
theorem tangent_line_x_intercept :
  let tangent_slope : ℝ := f' 1
  let tangent_point : ℝ × ℝ := (1, f 1)
  let x_intercept : ℝ := tangent_point.1 - tangent_point.2 / tangent_slope
  x_intercept = -3/7 := by sorry

end tangent_line_x_intercept_l3636_363636


namespace equation_solution_l3636_363651

theorem equation_solution :
  ∃ (x : ℝ), x > 0 ∧ 6 * Real.sqrt (4 + x) + 6 * Real.sqrt (4 - x) = 8 * Real.sqrt 5 ∧ x = Real.sqrt (1280 / 81) := by
  sorry

end equation_solution_l3636_363651


namespace function_properties_l3636_363694

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (log x) / x - k / x

theorem function_properties (k : ℝ) :
  (∀ x ≥ 1, x^2 * f k x + 1 / (x + 1) ≥ 0) →
  (∀ x ≥ 1, k ≥ 1/2 * x^2 + (exp 2 - 2) * x - exp x - 7) →
  (∀ x > 0, deriv (f k) x = (1 - log x + k) / x^2) →
  (deriv (f k) 1 = 10) →
  (∃ x_max > 0, ∀ x > 0, f k x ≤ f k x_max ∧ f k x_max = 1 / (exp 10)) ∧
  (exp 2 - 9 ≤ k ∧ k ≤ 1/2) :=
sorry

end function_properties_l3636_363694


namespace trig_expression_equals_one_l3636_363676

theorem trig_expression_equals_one :
  let numerator := Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
                   Real.cos (162 * π / 180) * Real.cos (102 * π / 180)
  let denominator := Real.sin (22 * π / 180) * Real.cos (8 * π / 180) + 
                     Real.cos (158 * π / 180) * Real.cos (98 * π / 180)
  numerator / denominator = 1 := by
sorry

end trig_expression_equals_one_l3636_363676


namespace john_average_increase_l3636_363677

def john_scores : List ℝ := [90, 85, 92, 95]

theorem john_average_increase :
  let initial_average := (john_scores.take 3).sum / 3
  let new_average := john_scores.sum / 4
  new_average - initial_average = 1.5 := by sorry

end john_average_increase_l3636_363677


namespace open_box_volume_l3636_363686

/-- The volume of an open box constructed from a rectangular sheet -/
def box_volume (x : ℝ) : ℝ :=
  (16 - 2*x) * (12 - 2*x) * x

/-- Theorem stating the volume of the open box -/
theorem open_box_volume (x : ℝ) : 
  box_volume x = 4*x^3 - 56*x^2 + 192*x :=
by sorry

end open_box_volume_l3636_363686


namespace soccer_preference_and_goals_l3636_363643

/-- Chi-square test statistic for 2x2 contingency table -/
def chi_square (a b c d : ℕ) : ℚ :=
  (200 * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Critical value for chi-square test at α = 0.001 -/
def critical_value : ℚ := 10828 / 1000

/-- Probability of scoring a goal for male students -/
def p_male : ℚ := 2 / 3

/-- Probability of scoring a goal for female student -/
def p_female : ℚ := 1 / 2

/-- Expected value of goals scored by 2 male and 1 female student -/
def expected_goals : ℚ := 11 / 6

theorem soccer_preference_and_goals (a b c d : ℕ) 
  (h1 : a + b = 100) (h2 : c + d = 100) (h3 : a + c = 90) (h4 : b + d = 110) :
  chi_square a b c d > critical_value ∧ 
  2 * p_male + p_female = expected_goals :=
sorry

end soccer_preference_and_goals_l3636_363643


namespace tangent_ellipse_hyperbola_l3636_363668

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+1)^2 = 1

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse x y ∧ hyperbola x y m ∧
  ∀ x' y' : ℝ, ellipse x' y' ∧ hyperbola x' y' m → (x', y') = (x, y)

-- State the theorem
theorem tangent_ellipse_hyperbola :
  ∀ m : ℝ, are_tangent m → m = 72 :=
by sorry

end tangent_ellipse_hyperbola_l3636_363668


namespace unique_right_triangle_l3636_363648

-- Define a structure for a right-angled triangle with integer sides
structure RightTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  right_angle : a * a + b * b = c * c
  perimeter_80 : a + b + c = 80

-- Theorem statement
theorem unique_right_triangle : 
  ∃! (t : RightTriangle), t.a = 30 ∧ t.b = 16 ∧ t.c = 34 :=
by sorry

end unique_right_triangle_l3636_363648


namespace coordinates_of_C_l3636_363660

-- Define the points
def A : ℝ × ℝ := (2, 8)
def M : ℝ × ℝ := (4, 11)
def L : ℝ × ℝ := (6, 6)

-- Define the properties
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def on_angle_bisector (L B C : ℝ × ℝ) : Prop :=
  (L.1 - B.1) * (C.2 - B.2) = (L.2 - B.2) * (C.1 - B.1)

-- Theorem statement
theorem coordinates_of_C (B C : ℝ × ℝ) 
  (h1 : is_midpoint M A B)
  (h2 : on_angle_bisector L B C) :
  C = (14, 2) := by sorry

end coordinates_of_C_l3636_363660


namespace element_value_l3636_363659

theorem element_value (a : Nat) : 
  a ∈ ({0, 1, 2, 3} : Set Nat) → 
  a ∉ ({0, 1, 2} : Set Nat) → 
  a = 3 := by
  sorry

end element_value_l3636_363659


namespace M_intersect_N_l3636_363603

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem M_intersect_N : M ∩ N = {0} := by
  sorry

end M_intersect_N_l3636_363603


namespace partitions_divisible_by_2_pow_n_l3636_363671

/-- Represents a valid partition of a 1 × n strip -/
def StripPartition (n : ℕ) : Type := Unit

/-- The number of valid partitions for a 1 × n strip -/
def num_partitions (n : ℕ) : ℕ := sorry

/-- The main theorem: the number of valid partitions is divisible by 2^n -/
theorem partitions_divisible_by_2_pow_n (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, num_partitions n = 2^n * k :=
sorry

end partitions_divisible_by_2_pow_n_l3636_363671


namespace jogging_time_proportional_to_distance_l3636_363697

/-- Given a constant jogging speed, prove that if it takes 30 minutes to jog 4 miles,
    then it will take 15 minutes to jog 2 miles. -/
theorem jogging_time_proportional_to_distance
  (speed : ℝ) -- Constant jogging speed
  (h1 : speed > 0) -- Assumption that speed is positive
  (h2 : 4 / speed = 30) -- It takes 30 minutes to jog 4 miles
  : 2 / speed = 15 := by
  sorry

end jogging_time_proportional_to_distance_l3636_363697


namespace flowers_per_bouquet_l3636_363698

theorem flowers_per_bouquet 
  (initial_flowers : ℕ) 
  (wilted_flowers : ℕ) 
  (num_bouquets : ℕ) 
  (h1 : initial_flowers = 66) 
  (h2 : wilted_flowers = 10) 
  (h3 : num_bouquets = 7) : 
  (initial_flowers - wilted_flowers) / num_bouquets = 8 := by
  sorry

end flowers_per_bouquet_l3636_363698


namespace stratified_sample_theorem_l3636_363685

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  totalPopulation : ℕ
  totalSampleSize : ℕ
  stratumSize : ℕ
  stratumSampleSize : ℕ

/-- Checks if the stratified sample is proportional -/
def isProportional (s : StratifiedSample) : Prop :=
  s.stratumSampleSize * s.totalPopulation = s.totalSampleSize * s.stratumSize

theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.totalPopulation = 2048)
  (h2 : s.totalSampleSize = 128)
  (h3 : s.stratumSize = 256)
  (h4 : isProportional s) :
  s.stratumSampleSize = 16 := by
  sorry

#check stratified_sample_theorem

end stratified_sample_theorem_l3636_363685


namespace linear_equation_implies_a_squared_plus_a_minus_one_equals_one_l3636_363673

theorem linear_equation_implies_a_squared_plus_a_minus_one_equals_one (a : ℝ) :
  (∀ x, ∃ k, (a + 4) * x^(|a + 3|) + 8 = k * x + 8) →
  a^2 + a - 1 = 1 :=
by sorry

end linear_equation_implies_a_squared_plus_a_minus_one_equals_one_l3636_363673


namespace product_equals_fraction_fraction_is_simplified_l3636_363612

/-- The repeating decimal 0.256̄ as a rational number -/
def repeating_decimal : ℚ := 256 / 999

/-- The product of 0.256̄ and 12 -/
def product : ℚ := repeating_decimal * 12

/-- Theorem stating that the product of 0.256̄ and 12 is equal to 1024/333 -/
theorem product_equals_fraction : product = 1024 / 333 := by
  sorry

/-- Theorem stating that 1024/333 is in its simplest form -/
theorem fraction_is_simplified : Int.gcd 1024 333 = 1 := by
  sorry

end product_equals_fraction_fraction_is_simplified_l3636_363612


namespace roberto_skipping_rate_l3636_363655

/-- Roberto's skipping rate problem -/
theorem roberto_skipping_rate 
  (valerie_rate : ℕ) 
  (total_skips : ℕ) 
  (duration : ℕ) 
  (h1 : valerie_rate = 80)
  (h2 : total_skips = 2250)
  (h3 : duration = 15) :
  ∃ (roberto_hourly_rate : ℕ), 
    roberto_hourly_rate = 4200 ∧ 
    roberto_hourly_rate * duration = (total_skips - valerie_rate * duration) * 4 :=
sorry

end roberto_skipping_rate_l3636_363655


namespace committee_seating_arrangements_l3636_363625

/-- The number of Democrats on the committee -/
def num_democrats : ℕ := 6

/-- The number of Republicans on the committee -/
def num_republicans : ℕ := 4

/-- The total number of politicians on the committee -/
def total_politicians : ℕ := num_democrats + num_republicans

/-- Represents that all politicians are distinguishable -/
axiom politicians_distinguishable : True

/-- Represents the constraint that no two Republicans can sit next to each other -/
axiom no_adjacent_republicans : True

/-- The number of ways to arrange the politicians around a circular table -/
def arrangement_count : ℕ := 43200

/-- Theorem stating that the number of valid arrangements is 43,200 -/
theorem committee_seating_arrangements :
  arrangement_count = 43200 :=
sorry

end committee_seating_arrangements_l3636_363625


namespace decimal_111_to_base5_l3636_363602

/-- Converts a natural number to its base-5 representation as a list of digits -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Theorem: The base-5 representation of 111 (decimal) is [2, 3, 3] -/
theorem decimal_111_to_base5 :
  toBase5 111 = [2, 3, 3] := by
  sorry

#eval toBase5 111  -- This will output [2, 3, 3]

end decimal_111_to_base5_l3636_363602


namespace max_value_theorem_l3636_363649

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 3*x*y + 5*y^2 = 9) : 
  x^2 + 3*x*y + 5*y^2 ≤ (315 + 297 * Real.sqrt 5) / 55 :=
by sorry

end max_value_theorem_l3636_363649


namespace profit_percentage_calculation_l3636_363633

theorem profit_percentage_calculation
  (purchase_price : ℕ)
  (repair_cost : ℕ)
  (transportation_charges : ℕ)
  (selling_price : ℕ)
  (h1 : purchase_price = 12000)
  (h2 : repair_cost = 5000)
  (h3 : transportation_charges = 1000)
  (h4 : selling_price = 27000) :
  (selling_price - (purchase_price + repair_cost + transportation_charges)) * 100 /
  (purchase_price + repair_cost + transportation_charges) = 50 :=
by
  sorry

#check profit_percentage_calculation

end profit_percentage_calculation_l3636_363633


namespace fruit_store_profit_l3636_363679

-- Define the cost price
def cost_price : ℝ := 40

-- Define the linear function for weekly sales quantity
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_quantity x

-- Define the new profit function with increased cost
def new_profit (x m : ℝ) : ℝ := (x - cost_price - m) * sales_quantity x

theorem fruit_store_profit :
  -- 1. The selling price that maximizes profit is 70 yuan/kg
  (∀ x : ℝ, profit x ≤ profit 70) ∧
  -- 2. The maximum profit is 1800 yuan
  (profit 70 = 1800) ∧
  -- 3. When the cost price increases by m yuan/kg (m > 0), and the profit decreases
  --    for selling prices > 76 yuan/kg, then 0 < m ≤ 12
  (∀ m : ℝ, m > 0 →
    (∀ x : ℝ, x > 76 → (∀ y : ℝ, y > x → new_profit y m < new_profit x m)) →
    m ≤ 12) :=
by sorry

end fruit_store_profit_l3636_363679


namespace todd_ate_cupcakes_l3636_363682

theorem todd_ate_cupcakes (initial_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : initial_cupcakes = 20)
  (h2 : packages = 3)
  (h3 : cupcakes_per_package = 3) :
  initial_cupcakes - (packages * cupcakes_per_package) = 11 :=
by sorry

end todd_ate_cupcakes_l3636_363682


namespace distance_on_number_line_distance_negative_five_negative_one_l3636_363652

theorem distance_on_number_line : ∀ (a b : ℝ), abs (a - b) = abs (b - a) :=
by sorry

theorem distance_negative_five_negative_one : abs (-5 - (-1)) = 4 :=
by sorry

end distance_on_number_line_distance_negative_five_negative_one_l3636_363652


namespace sum_of_squares_l3636_363666

theorem sum_of_squares (a b c x y z : ℝ) 
  (h1 : x/a + y/b + z/c = 5)
  (h2 : a/x + b/y + c/z = 3) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 19 := by
  sorry

end sum_of_squares_l3636_363666


namespace circular_garden_area_increase_l3636_363642

theorem circular_garden_area_increase : 
  let r₁ : ℝ := 6  -- radius of larger garden
  let r₂ : ℝ := 4  -- radius of smaller garden
  let area₁ := π * r₁^2  -- area of larger garden
  let area₂ := π * r₂^2  -- area of smaller garden
  (area₁ - area₂) / area₂ * 100 = 125
  := by sorry

end circular_garden_area_increase_l3636_363642


namespace find_x_value_l3636_363637

theorem find_x_value (x : ℝ) : 
  (max 1 (max 2 (max 3 x)) = 1 + 2 + 3 + x) → x = -3 := by
  sorry

end find_x_value_l3636_363637


namespace distinct_numbers_probability_l3636_363657

def num_sides : ℕ := 5
def num_dice : ℕ := 5

theorem distinct_numbers_probability : 
  (Nat.factorial num_sides : ℚ) / (num_sides ^ num_dice : ℚ) = 24 / 625 := by
  sorry

end distinct_numbers_probability_l3636_363657


namespace males_not_listening_l3636_363647

/-- Radio station survey data -/
structure SurveyData where
  total_listeners : ℕ
  total_non_listeners : ℕ
  male_listeners : ℕ
  female_non_listeners : ℕ

/-- Theorem: Number of males who do not listen to the station -/
theorem males_not_listening (data : SurveyData)
  (h1 : data.total_listeners = 200)
  (h2 : data.total_non_listeners = 180)
  (h3 : data.male_listeners = 75)
  (h4 : data.female_non_listeners = 120) :
  data.total_listeners + data.total_non_listeners - data.male_listeners - data.female_non_listeners = 185 := by
  sorry

end males_not_listening_l3636_363647


namespace product_not_ending_1999_l3636_363670

theorem product_not_ending_1999 (a b c d e : ℕ) : 
  a + b + c + d + e = 200 → 
  ∃ k : ℕ, a * b * c * d * e = 1000 * k ∨ a * b * c * d * e = 1000 * k + 1 ∨ 
          a * b * c * d * e = 1000 * k + 2 ∨ a * b * c * d * e = 1000 * k + 3 ∨ 
          a * b * c * d * e = 1000 * k + 4 ∨ a * b * c * d * e = 1000 * k + 5 ∨ 
          a * b * c * d * e = 1000 * k + 6 ∨ a * b * c * d * e = 1000 * k + 7 ∨ 
          a * b * c * d * e = 1000 * k + 8 ∨ a * b * c * d * e = 1000 * k + 9 := by
  sorry

end product_not_ending_1999_l3636_363670


namespace first_question_percentage_l3636_363605

theorem first_question_percentage
  (second_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : second_correct = 55)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 40)
  : ℝ :=
by
  -- The percentage answering the first question correctly is 65%
  sorry

#check first_question_percentage

end first_question_percentage_l3636_363605


namespace shondas_kids_l3636_363688

theorem shondas_kids (friends : ℕ) (other_adults : ℕ) (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_person : ℕ) :
  friends = 10 →
  other_adults = 7 →
  baskets = 15 →
  eggs_per_basket = 12 →
  eggs_per_person = 9 →
  ∃ (shondas_kids : ℕ), shondas_kids = 2 ∧
    (shondas_kids + friends + other_adults + 1) * eggs_per_person = baskets * eggs_per_basket :=
by sorry

end shondas_kids_l3636_363688


namespace inequality_not_always_holds_l3636_363699

theorem inequality_not_always_holds (a b : ℝ) (h : a < b) :
  ¬ ∀ m : ℝ, a * m^2 < b * m^2 := by
  sorry

end inequality_not_always_holds_l3636_363699


namespace max_value_theorem_l3636_363645

theorem max_value_theorem (x y : ℝ) (h : x^2 + 4*y^2 ≤ 4) :
  ∃ (M : ℝ), M = 12 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 ≤ 4 → |a+2*b-4| + |3-a-b| ≤ M :=
sorry

end max_value_theorem_l3636_363645


namespace sum_geq_sqrt_products_l3636_363601

theorem sum_geq_sqrt_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

end sum_geq_sqrt_products_l3636_363601


namespace percentage_difference_l3636_363641

theorem percentage_difference (n : ℝ) (h : n = 160) : 0.5 * n - 0.35 * n = 24 := by
  sorry

end percentage_difference_l3636_363641


namespace arithmetic_sequence_problem_l3636_363631

/-- An arithmetic sequence. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h : ∀ n, a (n + 1) = a n + d

/-- The problem statement. -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 1 * seq.a 5 = 9)
  (h2 : seq.a 2 = 3) :
  seq.a 4 = 3 ∨ seq.a 4 = 7 := by
  sorry

end arithmetic_sequence_problem_l3636_363631


namespace cube_order_l3636_363629

theorem cube_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_order_l3636_363629


namespace point_translation_l3636_363690

/-- Given a point A with coordinates (1, -1), prove that moving it up by 2 units
    and then left by 3 units results in a point B with coordinates (-2, 1). -/
theorem point_translation (A B : ℝ × ℝ) :
  A = (1, -1) →
  B.1 = A.1 - 3 →
  B.2 = A.2 + 2 →
  B = (-2, 1) := by
sorry

end point_translation_l3636_363690


namespace regular_pentagon_perimeter_l3636_363646

/-- A pentagon with all sides of equal length -/
structure RegularPentagon where
  side_length : ℝ

/-- The sum of all side lengths of a regular pentagon -/
def total_perimeter (p : RegularPentagon) : ℝ := 5 * p.side_length

/-- Theorem: If one side of a regular pentagon is 15 cm long, 
    then the sum of all side lengths is 75 cm -/
theorem regular_pentagon_perimeter : 
  ∀ (p : RegularPentagon), p.side_length = 15 → total_perimeter p = 75 := by
  sorry

#check regular_pentagon_perimeter

end regular_pentagon_perimeter_l3636_363646


namespace angle_equivalence_l3636_363691

-- Define α in degrees
def α : ℝ := 2010

-- Theorem statement
theorem angle_equivalence (α : ℝ) : 
  -- Part 1: Rewrite α in the form θ + 2kπ
  (α * π / 180 = 7 * π / 6 + 10 * π) ∧
  -- Part 2: Find equivalent angles in [-5π, 0)
  (∀ β : ℝ, -5 * π ≤ β ∧ β < 0 ∧ 
    (∃ k : ℤ, β = 7 * π / 6 + 2 * k * π) ↔ 
    (β = -29 * π / 6 ∨ β = -17 * π / 6 ∨ β = -5 * π / 6)) :=
by sorry

end angle_equivalence_l3636_363691


namespace sequence_sum_theorem_l3636_363606

theorem sequence_sum_theorem (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ+, S n = n^2 * a n) :
  (∀ n : ℕ+, S n = 2 * n / (n + 1)) ∧
  (∀ n : ℕ+, a n = 2 / (n * (n + 1))) := by
sorry

end sequence_sum_theorem_l3636_363606


namespace abs_neg_three_times_two_l3636_363681

theorem abs_neg_three_times_two : |(-3)| * 2 = 6 := by
  sorry

end abs_neg_three_times_two_l3636_363681


namespace probability_sum_six_is_three_sixteenths_l3636_363639

/-- A uniform tetrahedral die with faces numbered 1, 2, 3, 4 -/
def TetrahedralDie : Finset ℕ := {1, 2, 3, 4}

/-- The sample space of throwing the die twice -/
def SampleSpace : Finset (ℕ × ℕ) := TetrahedralDie.product TetrahedralDie

/-- The event where the sum of two throws equals 6 -/
def SumSixEvent : Finset (ℕ × ℕ) := SampleSpace.filter (fun p => p.1 + p.2 = 6)

/-- The probability of the sum being 6 when throwing the die twice -/
def probability_sum_six : ℚ := (SumSixEvent.card : ℚ) / (SampleSpace.card : ℚ)

theorem probability_sum_six_is_three_sixteenths : 
  probability_sum_six = 3 / 16 := by
  sorry

end probability_sum_six_is_three_sixteenths_l3636_363639


namespace sum_of_fractions_inequality_l3636_363635

theorem sum_of_fractions_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ (3 : ℝ) / 2 := by sorry

end sum_of_fractions_inequality_l3636_363635


namespace consecutive_integers_sqrt_11_l3636_363658

theorem consecutive_integers_sqrt_11 (a b : ℤ) : 
  (b = a + 1) →  -- a and b are consecutive integers
  (a < Real.sqrt 11) → 
  (Real.sqrt 11 < b) → 
  a + b = 7 := by
sorry

end consecutive_integers_sqrt_11_l3636_363658


namespace largest_winning_start_is_correct_l3636_363661

/-- The largest starting integer that guarantees a win for Bernardo in the number game. -/
def largest_winning_start : ℕ := 40

/-- Checks if a given number is a valid starting number for Bernardo to win. -/
def is_valid_start (m : ℕ) : Prop :=
  m ≥ 1 ∧ m ≤ 500 ∧
  3 * m < 1500 ∧
  3 * m + 30 < 1500 ∧
  9 * m + 90 < 1500 ∧
  9 * m + 120 < 1500 ∧
  27 * m + 360 < 1500 ∧
  27 * m + 390 < 1500

theorem largest_winning_start_is_correct :
  is_valid_start largest_winning_start ∧
  ∀ n : ℕ, n > largest_winning_start → ¬ is_valid_start n :=
by sorry

end largest_winning_start_is_correct_l3636_363661


namespace complement_of_M_in_U_l3636_363650

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 2, 4}

theorem complement_of_M_in_U : 
  (U \ M) = {3, 5, 6} := by sorry

end complement_of_M_in_U_l3636_363650


namespace complement_of_A_in_U_l3636_363609

def U : Set Int := {-2, -1, 1, 3, 5}
def A : Set Int := {-1, 3}

theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {-2, 1, 5} := by sorry

end complement_of_A_in_U_l3636_363609


namespace product_of_five_integers_l3636_363678

theorem product_of_five_integers (E F G H I : ℕ) : 
  E > 0 → F > 0 → G > 0 → H > 0 → I > 0 →
  E + F + G + H + I = 80 →
  E + 2 = F - 2 →
  E + 2 = G * 2 →
  E + 2 = H * 3 →
  E + 2 = I / 2 →
  E * F * G * H * I = 5120000 / 81 := by
sorry

end product_of_five_integers_l3636_363678


namespace james_fish_tanks_l3636_363663

def fish_tank_problem (num_tanks : ℕ) (fish_in_first_tank : ℕ) (total_fish : ℕ) : Prop :=
  ∃ (num_double_tanks : ℕ),
    num_tanks = 1 + num_double_tanks ∧
    fish_in_first_tank = 20 ∧
    total_fish = fish_in_first_tank + num_double_tanks * (2 * fish_in_first_tank) ∧
    total_fish = 100

theorem james_fish_tanks :
  ∃ (num_tanks : ℕ), fish_tank_problem num_tanks 20 100 ∧ num_tanks = 3 :=
sorry

end james_fish_tanks_l3636_363663


namespace quadratic_problem_l3636_363618

-- Define the quadratic function y₁
def y₁ (x b c : ℝ) : ℝ := x^2 + b*x + c

-- Define the quadratic function y₂
def y₂ (x m : ℝ) : ℝ := 2*x^2 + x + m

theorem quadratic_problem :
  ∀ (b c m : ℝ),
  (y₁ 0 b c = 4) →                        -- y₁ passes through (0,4)
  (∀ x, y₁ (1 + x) b c = y₁ (1 - x) b c) →  -- symmetry axis x = 1
  (b^2 - c = 0) →                         -- condition b² - c = 0
  (∃ x₀, b - 3 ≤ x₀ ∧ x₀ ≤ b ∧ 
    (∀ x, b - 3 ≤ x ∧ x ≤ b → y₁ x₀ b c ≤ y₁ x b c) ∧
    y₁ x₀ b c = 21) →                     -- minimum value 21 when b-3 ≤ x ≤ b
  (∀ x, 0 ≤ x ∧ x ≤ 1 → y₂ x m ≥ y₁ x b c) →  -- y₂ ≥ y₁ for 0 ≤ x ≤ 1
  ((∀ x, y₁ x b c = x^2 - 2*x + 4) ∧      -- Part 1 result
   (b = -Real.sqrt 7 ∨ b = 4) ∧           -- Part 2 result
   (m = 4))                               -- Part 3 result
  := by sorry

end quadratic_problem_l3636_363618


namespace shaded_area_in_circle_l3636_363600

theorem shaded_area_in_circle (r : ℝ) (h1 : r > 0) : 
  let circle_area := π * r^2
  let sector_area := 2 * π
  let sector_fraction := 1 / 8
  let triangle_area := r^2 / 2
  sector_area = sector_fraction * circle_area → 
  sector_area - triangle_area = 2 * π - 4 := by
sorry

end shaded_area_in_circle_l3636_363600


namespace union_of_A_and_B_l3636_363667

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_of_A_and_B_l3636_363667


namespace eulers_formula_l3636_363608

/-- A planar graph structure -/
structure PlanarGraph where
  V : Type*  -- Set of vertices
  E : Type*  -- Set of edges
  F : Type*  -- Set of faces
  n : ℕ      -- Number of vertices
  m : ℕ      -- Number of edges
  ℓ : ℕ      -- Number of faces
  is_connected : Prop  -- Property that the graph is connected

/-- Euler's formula for planar graphs -/
theorem eulers_formula (G : PlanarGraph) : G.n - G.m + G.ℓ = 2 := by
  sorry

end eulers_formula_l3636_363608


namespace plums_for_oranges_l3636_363692

-- Define the cost of fruits as real numbers
variables (orange pear plum : ℝ)

-- Define the conditions
def condition1 : Prop := 5 * orange = 3 * pear
def condition2 : Prop := 4 * pear = 6 * plum

-- Theorem to prove
theorem plums_for_oranges 
  (h1 : condition1 orange pear) 
  (h2 : condition2 pear plum) : 
  20 * orange = 18 * plum :=
sorry

end plums_for_oranges_l3636_363692


namespace arithmetic_sequence_sum_l3636_363665

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement to prove -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 ^ 2 - 6 * a 3 + 5 = 0 →
  a 15 ^ 2 - 6 * a 15 + 5 = 0 →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by sorry

end arithmetic_sequence_sum_l3636_363665


namespace intersection_equality_necessary_not_sufficient_l3636_363683

theorem intersection_equality_necessary_not_sufficient :
  (∀ (M N P : Set α), M = N → M ∩ P = N ∩ P) ∧
  (∃ (M N P : Set α), M ∩ P = N ∩ P ∧ M ≠ N) :=
by sorry

end intersection_equality_necessary_not_sufficient_l3636_363683


namespace right_handed_players_count_l3636_363616

theorem right_handed_players_count (total_players throwers : ℕ) : 
  total_players = 150 →
  throwers = 60 →
  (total_players - throwers) % 2 = 0 →
  105 = throwers + (total_players - throwers) / 2 := by
  sorry

end right_handed_players_count_l3636_363616


namespace circle_area_theorem_l3636_363610

/-- Line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Circle with center at origin -/
structure Circle where
  radius : ℝ

/-- The perpendicular line to a given line passing through a point -/
def perpendicularLine (l : Line) (p : ℝ × ℝ) : Line :=
  sorry

/-- The length of the chord formed by the intersection of a line and a circle -/
def chordLength (l : Line) (c : Circle) : ℝ :=
  sorry

/-- The area of a circle -/
def circleArea (c : Circle) : ℝ :=
  sorry

theorem circle_area_theorem (l : Line) (c : Circle) :
  l.point1 = (2, 1) →
  l.point2 = (1, -1) →
  let m := perpendicularLine l (2, 1)
  chordLength m c = 6 * Real.sqrt 5 / 5 →
  circleArea c = 5 * Real.pi :=
sorry

end circle_area_theorem_l3636_363610


namespace monotonicity_indeterminate_l3636_363632

-- Define the concept of an increasing function on an open interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Define the theorem
theorem monotonicity_indeterminate
  (f : ℝ → ℝ) (a b c : ℝ) 
  (hab : a < b) (hbc : b < c)
  (h1 : IncreasingOn f a b)
  (h2 : IncreasingOn f b c) :
  ¬ (IncreasingOn f a c ∨ (∀ x y, a < x ∧ x < y ∧ y < c → f x > f y)) :=
sorry

end monotonicity_indeterminate_l3636_363632


namespace corner_cut_rectangle_l3636_363644

/-- Given a rectangle ABCD with dimensions AB = 18 m and AD = 12 m,
    and identical right-angled isosceles triangles cut off from the corners,
    leaving a smaller rectangle PQRS. The total area cut off is 180 m². -/
theorem corner_cut_rectangle (AB AD : ℝ) (area_cut : ℝ) (PR : ℝ) : AB = 18 → AD = 12 → area_cut = 180 → PR = 18 - 6 * Real.sqrt 10 := by
  sorry

end corner_cut_rectangle_l3636_363644


namespace total_weight_CaI2_is_1469_4_l3636_363695

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of moles of calcium iodide -/
def moles_CaI2 : ℝ := 5

/-- The molecular weight of calcium iodide (CaI2) in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_I

/-- The total weight of calcium iodide in grams -/
def total_weight_CaI2 : ℝ := moles_CaI2 * molecular_weight_CaI2

theorem total_weight_CaI2_is_1469_4 :
  total_weight_CaI2 = 1469.4 := by sorry

end total_weight_CaI2_is_1469_4_l3636_363695


namespace possible_values_of_a_l3636_363696

def P : Set ℝ := {x : ℝ | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x : ℝ | a * x + 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, Q a ⊆ P → a = 0 ∨ a = -1/2 ∨ a = 1/3 :=
by sorry

end possible_values_of_a_l3636_363696


namespace ratio_equality_l3636_363623

theorem ratio_equality (p q r u v w : ℝ) 
  (h_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ u > 0 ∧ v > 0 ∧ w > 0)
  (h_pqr : p^2 + q^2 + r^2 = 49)
  (h_uvw : u^2 + v^2 + w^2 = 64)
  (h_sum : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end ratio_equality_l3636_363623


namespace quadrilateral_area_l3636_363640

/-- The area of a quadrilateral with vertices at (4,0), (0,5), (3,4), and (10,10) is 22.5 square units. -/
theorem quadrilateral_area : 
  let vertices : List (ℝ × ℝ) := [(4,0), (0,5), (3,4), (10,10)]
  ∃ (area : ℝ), area = 22.5 ∧ 
  area = (1/2) * abs (
    (4 * 5 + 0 * 4 + 3 * 10 + 10 * 0) - 
    (0 * 0 + 5 * 3 + 4 * 10 + 10 * 4)
  ) := by sorry

end quadrilateral_area_l3636_363640


namespace work_completion_theorem_l3636_363627

/-- Given a work that could be finished in 12 days, and was actually finished in 9 days
    after 10 more men joined, prove that the original number of men employed was 30. -/
theorem work_completion_theorem (original_days : ℕ) (actual_days : ℕ) (additional_men : ℕ) :
  original_days = 12 →
  actual_days = 9 →
  additional_men = 10 →
  ∃ (original_men : ℕ), original_men * original_days = (original_men + additional_men) * actual_days ∧ original_men = 30 :=
by
  sorry

end work_completion_theorem_l3636_363627


namespace f_min_at_three_l3636_363617

/-- The quadratic function f(x) = x^2 - 6x + 8 -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- Theorem stating that f(x) attains its minimum value when x = 3 -/
theorem f_min_at_three : 
  ∀ x : ℝ, f x ≥ f 3 := by sorry

end f_min_at_three_l3636_363617


namespace zebra_stripes_l3636_363621

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes = white stripes + 1
  b = w + 7 →      -- White stripes = wide black stripes + 7
  n = 8            -- Number of narrow black stripes is 8
:= by sorry

end zebra_stripes_l3636_363621


namespace nonzero_terms_count_l3636_363687

-- Define the polynomials
def p (x : ℝ) : ℝ := 2*x + 3
def q (x : ℝ) : ℝ := x^2 + 4*x + 5
def r (x : ℝ) : ℝ := x^3 - x^2 + 2*x + 1

-- Define the expanded expression
def expanded_expr (x : ℝ) : ℝ := p x * q x - 4 * r x

-- Theorem statement
theorem nonzero_terms_count :
  ∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  ∀ x, expanded_expr x = a*x^3 + b*x^2 + c*x + d :=
by sorry

end nonzero_terms_count_l3636_363687


namespace distance_from_circle_center_to_line_l3636_363669

/-- The distance from the center of a circle to a line --/
theorem distance_from_circle_center_to_line :
  let circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 = 0}
  let center : ℝ × ℝ := (2, 0)
  let line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 = 0}
  center ∈ circle →
  (∀ p ∈ circle, (p.1 - 2)^2 + p.2^2 = 4) →
  (∀ p ∈ line, p.1 = p.2) →
  ∃ d : ℝ, d = Real.sqrt 2 ∧ ∀ p ∈ line, (center.1 - p.1)^2 + (center.2 - p.2)^2 = d^2 :=
by sorry

end distance_from_circle_center_to_line_l3636_363669


namespace jed_speeding_fine_l3636_363622

/-- Calculates the speeding fine in Zeoland -/
def speeding_fine (speed_limit : ℕ) (actual_speed : ℕ) (fine_per_mph : ℕ) : ℕ :=
  if actual_speed > speed_limit
  then (actual_speed - speed_limit) * fine_per_mph
  else 0

/-- Proves that Jed's speeding fine is $256 -/
theorem jed_speeding_fine :
  let speed_limit : ℕ := 50
  let actual_speed : ℕ := 66
  let fine_per_mph : ℕ := 16
  speeding_fine speed_limit actual_speed fine_per_mph = 256 :=
by
  sorry

end jed_speeding_fine_l3636_363622


namespace cubic_rational_roots_l3636_363628

/-- A cubic polynomial with rational coefficients -/
structure CubicPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The roots of a cubic polynomial -/
def roots (p : CubicPolynomial) : Set ℚ :=
  {x : ℚ | x^3 + p.a * x^2 + p.b * x + p.c = 0}

/-- The theorem stating the only possible sets of rational roots for a cubic polynomial -/
theorem cubic_rational_roots (p : CubicPolynomial) :
  roots p = {0, 1, -2} ∨ roots p = {1, -1, -1} := by
  sorry


end cubic_rational_roots_l3636_363628


namespace polynomial_factor_sum_l3636_363626

theorem polynomial_factor_sum (d M N K : ℝ) :
  (∃ a b : ℝ, (X^2 + 3*X + 1) * (X^2 + a*X + b) = X^4 - d*X^3 + M*X^2 + N*X + K) →
  M + N + K = 5*K - 4*d - 11 :=
by sorry

end polynomial_factor_sum_l3636_363626


namespace octagon_edge_length_l3636_363693

/-- The length of one edge of a regular octagon made from the same thread as a regular pentagon with one edge of 16 cm -/
theorem octagon_edge_length (pentagon_edge : ℝ) (thread_length : ℝ) : 
  pentagon_edge = 16 → thread_length = 5 * pentagon_edge → thread_length / 8 = 10 := by
  sorry

#check octagon_edge_length

end octagon_edge_length_l3636_363693


namespace ones_digit_of_largest_power_of_3_dividing_27_factorial_l3636_363620

/-- The largest power of 3 that divides n! -/
def largestPowerOf3DividingFactorial (n : ℕ) : ℕ :=
  (n / 3) + (n / 9) + (n / 27)

/-- The ones digit of 3^n -/
def onesDigitOf3ToPower (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0 -- This case is impossible, but Lean requires it

theorem ones_digit_of_largest_power_of_3_dividing_27_factorial :
  onesDigitOf3ToPower (largestPowerOf3DividingFactorial 27) = 3 := by
  sorry

#eval onesDigitOf3ToPower (largestPowerOf3DividingFactorial 27)

end ones_digit_of_largest_power_of_3_dividing_27_factorial_l3636_363620


namespace student_rank_from_right_l3636_363664

/-- Given a student ranked 8th from the left in a group of 20 students, 
    their rank from the right is 13th. -/
theorem student_rank_from_right 
  (total_students : ℕ) 
  (rank_from_left : ℕ) 
  (h1 : total_students = 20) 
  (h2 : rank_from_left = 8) : 
  total_students - (rank_from_left - 1) = 13 := by
sorry

end student_rank_from_right_l3636_363664


namespace quadratic_inequality_solution_l3636_363614

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  a < 0 →
  (∀ x, x^2 - a*x - 6*a^2 > 0 ↔ x < x₁ ∨ x > x₂) →
  x₂ - x₁ = 5 * Real.sqrt 2 →
  a = -Real.sqrt 2 :=
sorry

end quadratic_inequality_solution_l3636_363614


namespace workshop_sampling_theorem_l3636_363654

-- Define the total number of workers in each group
def total_workers_A : ℕ := 10
def total_workers_B : ℕ := 10

-- Define the number of female workers in each group
def female_workers_A : ℕ := 4
def female_workers_B : ℕ := 6

-- Define the total number of workers selected for assessment
def total_selected : ℕ := 4

-- Define the number of workers selected from each group
def selected_from_A : ℕ := 2
def selected_from_B : ℕ := 2

-- Define the probability of selecting exactly 1 female worker from Group A
def prob_one_female_A : ℚ := (Nat.choose female_workers_A 1 * Nat.choose (total_workers_A - female_workers_A) (selected_from_A - 1)) / Nat.choose total_workers_A selected_from_A

-- Define the probability of selecting exactly 2 male workers from both groups
def prob_two_males : ℚ :=
  (Nat.choose (total_workers_A - female_workers_A) 0 * Nat.choose female_workers_A 2 *
   Nat.choose (total_workers_B - female_workers_B) 2 * Nat.choose female_workers_B 0 +
   Nat.choose (total_workers_A - female_workers_A) 1 * Nat.choose female_workers_A 1 *
   Nat.choose (total_workers_B - female_workers_B) 1 * Nat.choose female_workers_B 1 +
   Nat.choose (total_workers_A - female_workers_A) 2 * Nat.choose female_workers_A 0 *
   Nat.choose (total_workers_B - female_workers_B) 0 * Nat.choose female_workers_B 2) /
  (Nat.choose total_workers_A selected_from_A * Nat.choose total_workers_B selected_from_B)

theorem workshop_sampling_theorem :
  (selected_from_A + selected_from_B = total_selected) ∧
  (prob_one_female_A = (Nat.choose female_workers_A 1 * Nat.choose (total_workers_A - female_workers_A) (selected_from_A - 1)) / Nat.choose total_workers_A selected_from_A) ∧
  (prob_two_males = (Nat.choose (total_workers_A - female_workers_A) 0 * Nat.choose female_workers_A 2 *
                     Nat.choose (total_workers_B - female_workers_B) 2 * Nat.choose female_workers_B 0 +
                     Nat.choose (total_workers_A - female_workers_A) 1 * Nat.choose female_workers_A 1 *
                     Nat.choose (total_workers_B - female_workers_B) 1 * Nat.choose female_workers_B 1 +
                     Nat.choose (total_workers_A - female_workers_A) 2 * Nat.choose female_workers_A 0 *
                     Nat.choose (total_workers_B - female_workers_B) 0 * Nat.choose female_workers_B 2) /
                    (Nat.choose total_workers_A selected_from_A * Nat.choose total_workers_B selected_from_B)) := by
  sorry


end workshop_sampling_theorem_l3636_363654


namespace g_sum_equals_two_l3636_363604

-- Define the function g
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 5

-- State the theorem
theorem g_sum_equals_two (a b c : ℝ) :
  g a b c 11 = 1 → g a b c 11 + g a b c (-11) = 2 := by
sorry

end g_sum_equals_two_l3636_363604


namespace a4b4_value_l3636_363674

theorem a4b4_value (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end a4b4_value_l3636_363674


namespace no_even_three_digit_sum_31_l3636_363656

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_even_three_digit_sum_31 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 31 ∧ Even n :=
sorry

end no_even_three_digit_sum_31_l3636_363656


namespace largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l3636_363638

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 24 < 0 → n ≤ 7 :=
by
  sorry

theorem seven_satisfies_inequality :
  (7 : ℤ)^2 - 11*7 + 24 < 0 :=
by
  sorry

theorem eight_does_not_satisfy_inequality :
  (8 : ℤ)^2 - 11*8 + 24 ≥ 0 :=
by
  sorry

end largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l3636_363638


namespace ladybug_count_l3636_363689

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem ladybug_count : total_ladybugs = 67082 := by
  sorry

end ladybug_count_l3636_363689


namespace short_answer_time_l3636_363672

/-- Represents the time in minutes for various writing assignments --/
structure WritingTimes where
  essay : ℕ        -- Time for one essay in minutes
  paragraph : ℕ    -- Time for one paragraph in minutes
  shortAnswer : ℕ  -- Time for one short-answer question in minutes

/-- Represents the number of each type of assignment --/
structure AssignmentCounts where
  essays : ℕ
  paragraphs : ℕ
  shortAnswers : ℕ

/-- Calculates the total time in minutes for all assignments --/
def totalTime (times : WritingTimes) (counts : AssignmentCounts) : ℕ :=
  times.essay * counts.essays +
  times.paragraph * counts.paragraphs +
  times.shortAnswer * counts.shortAnswers

/-- The main theorem to prove --/
theorem short_answer_time 
  (times : WritingTimes) 
  (counts : AssignmentCounts) 
  (h1 : times.essay = 60)           -- Each essay takes 1 hour (60 minutes)
  (h2 : times.paragraph = 15)       -- Each paragraph takes 15 minutes
  (h3 : counts.essays = 2)          -- Karen assigns 2 essays
  (h4 : counts.paragraphs = 5)      -- Karen assigns 5 paragraphs
  (h5 : counts.shortAnswers = 15)   -- Karen assigns 15 short-answer questions
  (h6 : totalTime times counts = 240) -- Total homework time is 4 hours (240 minutes)
  : times.shortAnswer = 3 :=
by
  sorry


end short_answer_time_l3636_363672


namespace count_solution_pairs_l3636_363619

/-- The number of ordered pairs (a, b) of complex numbers satisfying the given equations -/
def solution_count : ℕ := 24

/-- The predicate defining the condition for a pair of complex numbers -/
def satisfies_equations (a b : ℂ) : Prop :=
  a^4 * b^6 = 1 ∧ a^8 * b^3 = 1

theorem count_solution_pairs :
  (∃! (s : Finset (ℂ × ℂ)), s.card = solution_count ∧ 
   ∀ p ∈ s, satisfies_equations p.1 p.2 ∧
   ∀ a b, satisfies_equations a b → (a, b) ∈ s) :=
sorry

end count_solution_pairs_l3636_363619


namespace braiding_time_proof_l3636_363684

/-- Calculates the time in minutes required to braid dancers' hair -/
def braiding_time (num_dancers : ℕ) (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) : ℚ :=
  (num_dancers * braids_per_dancer * seconds_per_braid : ℚ) / 60

/-- Proves that given 15 dancers, 10 braids per dancer, and 45 seconds per braid,
    the total time required to braid all dancers' hair is 112.5 minutes -/
theorem braiding_time_proof :
  braiding_time 15 10 45 = 112.5 := by
  sorry

end braiding_time_proof_l3636_363684


namespace y2_less_than_y1_l3636_363607

def f (x : ℝ) := -4 * x - 3

theorem y2_less_than_y1 (y₁ y₂ : ℝ) 
  (h1 : f (-2) = y₁) 
  (h2 : f 5 = y₂) : 
  y₂ < y₁ := by
sorry

end y2_less_than_y1_l3636_363607


namespace inequality_solution_set_l3636_363624

theorem inequality_solution_set :
  {x : ℝ | (x + 1) * (2 - x) < 0} = {x : ℝ | x < -1 ∨ x > 2} := by
  sorry

end inequality_solution_set_l3636_363624


namespace sin_sixty_degrees_l3636_363630

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_sixty_degrees_l3636_363630


namespace elevator_max_additional_weight_l3636_363675

/-- The maximum weight a person can have to enter an elevator without overloading it,
    given the current occupants and the elevator's weight limit. -/
def max_additional_weight (adult_count : ℕ) (adult_avg_weight : ℝ)
                          (child_count : ℕ) (child_avg_weight : ℝ)
                          (max_elevator_weight : ℝ) : ℝ :=
  max_elevator_weight - (adult_count * adult_avg_weight + child_count * child_avg_weight)

/-- Theorem stating the maximum weight of the next person to enter the elevator
    without overloading it, given the specific conditions. -/
theorem elevator_max_additional_weight :
  max_additional_weight 3 140 2 64 600 = 52 := by
  sorry

end elevator_max_additional_weight_l3636_363675


namespace quadrilateral_diagonal_bounds_l3636_363611

/-- Given four points A, B, C, D in a plane, with distances AB = 2, BC = 7, CD = 5, and DA = 12,
    the minimum possible length of AC is 7 and the maximum possible length of AC is 9. -/
theorem quadrilateral_diagonal_bounds (A B C D : EuclideanSpace ℝ (Fin 2)) 
  (h1 : dist A B = 2)
  (h2 : dist B C = 7)
  (h3 : dist C D = 5)
  (h4 : dist D A = 12) :
  (∃ (m M : ℝ), m = 7 ∧ M = 9 ∧ 
    (∀ (AC : ℝ), AC = dist A C → m ≤ AC ∧ AC ≤ M)) := by
  sorry

end quadrilateral_diagonal_bounds_l3636_363611


namespace farmer_turkeys_l3636_363634

theorem farmer_turkeys (total_cost : ℝ) (kept_turkeys : ℕ) (sale_revenue : ℝ) (profit_per_bird : ℝ) :
  total_cost = 60 ∧
  kept_turkeys = 15 ∧
  sale_revenue = 54 ∧
  profit_per_bird = 0.1 →
  ∃ n : ℕ,
    n * (total_cost / n) = total_cost ∧
    ((total_cost / n) + profit_per_bird) * (n - kept_turkeys) = sale_revenue ∧
    n = 75 :=
by sorry

end farmer_turkeys_l3636_363634


namespace symmetrical_line_intersection_l3636_363615

/-- Given points A and B, and a circle, prove that if the line symmetrical to AB about y=a intersects the circle, then a is in the range [1/3, 3/2]. -/
theorem symmetrical_line_intersection (a : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (0, a)
  let circle (x y : ℝ) := (x + 3)^2 + (y + 2)^2 = 1
  let symmetrical_line (x y : ℝ) := (3 - a) * x - 2 * y + 2 * a = 0
  (∃ x y, circle x y ∧ symmetrical_line x y) → a ∈ Set.Icc (1/3) (3/2) :=
by sorry

end symmetrical_line_intersection_l3636_363615


namespace triangle_inequality_theorem_l3636_363613

theorem triangle_inequality_theorem (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
by sorry

end triangle_inequality_theorem_l3636_363613
