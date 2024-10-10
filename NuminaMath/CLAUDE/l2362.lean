import Mathlib

namespace complex_arithmetic_equalities_l2362_236282

theorem complex_arithmetic_equalities :
  (16 / (-2)^3 - (-1/2)^3 * (-4) + 2.5 = 0) ∧
  ((-1)^2022 + |(-2)^2 + 4| - (1/2 - 1/4 + 1/8) * (-24) = 10) := by
  sorry

end complex_arithmetic_equalities_l2362_236282


namespace quadrilateral_perimeter_sum_l2362_236263

/-- A quadrilateral with vertices at (1,2), (4,6), (5,4), and (2,0) has a perimeter that can be
    expressed as a√2 + b√5 + c√10 where a, b, and c are integers, and their sum is 2. -/
theorem quadrilateral_perimeter_sum (a b c : ℤ) : 
  let v1 : ℝ × ℝ := (1, 2)
  let v2 : ℝ × ℝ := (4, 6)
  let v3 : ℝ × ℝ := (5, 4)
  let v4 : ℝ × ℝ := (2, 0)
  let perimeter := dist v1 v2 + dist v2 v3 + dist v3 v4 + dist v4 v1
  perimeter = a * Real.sqrt 2 + b * Real.sqrt 5 + c * Real.sqrt 10 →
  a + b + c = 2 := by
  sorry

end quadrilateral_perimeter_sum_l2362_236263


namespace no_opposite_midpoints_l2362_236237

/-- Represents a rectangular billiard table -/
structure BilliardTable where
  length : ℝ
  width : ℝ
  corner_pockets : Bool

/-- Represents the trajectory of a ball on the billiard table -/
structure BallTrajectory where
  table : BilliardTable
  start_corner : Fin 4
  angle : ℝ

/-- Predicate to check if a point is on the midpoint of a side -/
def is_side_midpoint (table : BilliardTable) (x y : ℝ) : Prop :=
  (x = 0 ∧ y = table.width / 2) ∨
  (x = table.length ∧ y = table.width / 2) ∨
  (y = 0 ∧ x = table.length / 2) ∨
  (y = table.width ∧ x = table.length / 2)

/-- Theorem stating that a ball cannot visit midpoints of opposite sides -/
theorem no_opposite_midpoints (trajectory : BallTrajectory) 
  (h1 : trajectory.angle = π/4)
  (h2 : ∃ (x1 y1 : ℝ), is_side_midpoint trajectory.table x1 y1) :
  ¬ ∃ (x2 y2 : ℝ), 
    is_side_midpoint trajectory.table x2 y2 ∧ 
    ((x1 = 0 ∧ x2 = trajectory.table.length) ∨ 
     (x1 = trajectory.table.length ∧ x2 = 0) ∨
     (y1 = 0 ∧ y2 = trajectory.table.width) ∨
     (y1 = trajectory.table.width ∧ y2 = 0)) :=
by
  sorry

end no_opposite_midpoints_l2362_236237


namespace stage_8_area_l2362_236265

/-- The area of the rectangle at a given stage in the square-adding process -/
def rectangleArea (stage : ℕ) : ℕ :=
  stage * (4 * 4)

/-- Theorem: The area of the rectangle at Stage 8 is 128 square inches -/
theorem stage_8_area : rectangleArea 8 = 128 := by
  sorry

end stage_8_area_l2362_236265


namespace quadratic_equation_integer_roots_l2362_236219

theorem quadratic_equation_integer_roots (a : ℝ) : 
  (a > 0 ∧ ∃ x y : ℤ, x ≠ y ∧ 
    a^2 * (x : ℝ)^2 + a * (x : ℝ) + 1 - 7 * a^2 = 0 ∧
    a^2 * (y : ℝ)^2 + a * (y : ℝ) + 1 - 7 * a^2 = 0) ↔ 
  (a = 1 ∨ a = 1/2 ∨ a = 1/3) :=
sorry

end quadratic_equation_integer_roots_l2362_236219


namespace sqrt_450_simplification_l2362_236262

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l2362_236262


namespace seashells_given_to_sam_l2362_236287

theorem seashells_given_to_sam (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 70) 
  (h2 : remaining_seashells = 27) : 
  initial_seashells - remaining_seashells = 43 := by
  sorry

end seashells_given_to_sam_l2362_236287


namespace larger_number_theorem_l2362_236220

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem larger_number_theorem (a b : ℕ) 
  (h1 : Nat.gcd a b = 37)
  (h2 : is_prime 37)
  (h3 : ∃ (k : ℕ), Nat.lcm a b = k * 37 * 17 * 23 * 29 * 31) :
  max a b = 13007833 := by
sorry

end larger_number_theorem_l2362_236220


namespace restaurant_bill_change_l2362_236295

/-- Calculates the change received after a restaurant bill payment --/
theorem restaurant_bill_change
  (salmon_price truffled_mac_price chicken_katsu_price seafood_pasta_price black_burger_price wine_price : ℝ)
  (discount_rate service_charge_rate additional_tip_rate : ℝ)
  (payment : ℝ)
  (h_salmon : salmon_price = 40)
  (h_truffled_mac : truffled_mac_price = 20)
  (h_chicken_katsu : chicken_katsu_price = 25)
  (h_seafood_pasta : seafood_pasta_price = 30)
  (h_black_burger : black_burger_price = 15)
  (h_wine : wine_price = 50)
  (h_discount : discount_rate = 0.1)
  (h_service : service_charge_rate = 0.12)
  (h_tip : additional_tip_rate = 0.05)
  (h_payment : payment = 300) :
  let food_cost := salmon_price + truffled_mac_price + chicken_katsu_price + seafood_pasta_price + black_burger_price
  let total_cost := food_cost + wine_price
  let service_charge := service_charge_rate * total_cost
  let bill_before_discount := total_cost + service_charge
  let discount := discount_rate * food_cost
  let bill_after_discount := bill_before_discount - discount
  let additional_tip := additional_tip_rate * bill_after_discount
  let final_bill := bill_after_discount + additional_tip
  payment - final_bill = 101.97 := by sorry

end restaurant_bill_change_l2362_236295


namespace isosceles_triangle_area_l2362_236272

/-- An isosceles triangle with specific side lengths -/
structure IsoscelesTriangle where
  /-- Length of equal sides PQ and PR -/
  side : ℝ
  /-- Length of base QR -/
  base : ℝ
  /-- side is positive -/
  side_pos : side > 0
  /-- base is positive -/
  base_pos : base > 0

/-- The area of an isosceles triangle with side length 13 and base length 10 -/
theorem isosceles_triangle_area (t : IsoscelesTriangle)
  (h_side : t.side = 13)
  (h_base : t.base = 10) :
  let height := Real.sqrt (t.side ^ 2 - (t.base / 2) ^ 2)
  (1 / 2 : ℝ) * t.base * height = 60 := by
  sorry

end isosceles_triangle_area_l2362_236272


namespace internal_resistance_of_current_source_l2362_236243

/-- Given an electric circuit with resistors R₁ and R₂, and a current source
    with internal resistance r, prove that r = 30 Ω when R₁ = 10 Ω, R₂ = 30 Ω,
    and the current ratio I₂/I₁ = 1.5 when the polarity is reversed. -/
theorem internal_resistance_of_current_source
  (R₁ R₂ r : ℝ)
  (h₁ : R₁ = 10)
  (h₂ : R₂ = 30)
  (h₃ : (R₁ + r) / (R₂ + r) = 1.5) :
  r = 30 := by
  sorry

#check internal_resistance_of_current_source

end internal_resistance_of_current_source_l2362_236243


namespace greatest_integer_difference_l2362_236242

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 7) :
  (∀ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 7 → ⌊b - a⌋ ≤ 2) ∧
  (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 7 ∧ ⌊b - a⌋ = 2) :=
by sorry

end greatest_integer_difference_l2362_236242


namespace intersection_above_axis_implies_no_roots_l2362_236241

/-- 
Given that the graphs of y = ax², y = bx, and y = c intersect at a point above the x-axis,
prove that the equation ax² + bx + c = 0 has no real roots.
-/
theorem intersection_above_axis_implies_no_roots 
  (a b c : ℝ) 
  (ha : a > 0)
  (hc : c > 0)
  (h_intersect : ∃ (m : ℝ), a * m^2 = b * m ∧ b * m = c) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 :=
by sorry

end intersection_above_axis_implies_no_roots_l2362_236241


namespace max_sum_on_ellipse_l2362_236222

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the sum function S
def S (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem max_sum_on_ellipse :
  (∀ x y : ℝ, on_ellipse x y → S x y ≤ 2) ∧
  (∃ x y : ℝ, on_ellipse x y ∧ S x y = 2) := by
  sorry


end max_sum_on_ellipse_l2362_236222


namespace prime_sum_of_squares_l2362_236293

/-- A number is prime if it's greater than 1 and its only divisors are 1 and itself -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The expression we're interested in -/
def expression (p q : ℕ) : ℕ :=
  2^2 + p^2 + q^2

theorem prime_sum_of_squares : 
  ∀ p q : ℕ, isPrime p → isPrime q → isPrime (expression p q) → 
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
sorry

end prime_sum_of_squares_l2362_236293


namespace water_transfer_problem_l2362_236296

theorem water_transfer_problem (initial_volume : ℝ) (loss_percentage : ℝ) (hemisphere_volume : ℝ) : 
  initial_volume = 10936 →
  loss_percentage = 2.5 →
  hemisphere_volume = 4 →
  ⌈(initial_volume * (1 - loss_percentage / 100)) / hemisphere_volume⌉ = 2666 := by
  sorry

end water_transfer_problem_l2362_236296


namespace cube_volume_from_surface_area_l2362_236249

/-- Given a cube with surface area 294 square centimeters, prove its volume is 343 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 294 → s^3 = 343 :=
by
  sorry

end cube_volume_from_surface_area_l2362_236249


namespace product_of_sum_equals_three_times_product_l2362_236207

theorem product_of_sum_equals_three_times_product (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : x + y = 3 * x * y) (h4 : x + y ≠ 0) : x * y = (x + y) / 3 := by
  sorry

end product_of_sum_equals_three_times_product_l2362_236207


namespace dot_path_length_on_rolled_cube_l2362_236264

/-- The path length of a dot on a cube when rolled twice --/
theorem dot_path_length_on_rolled_cube : 
  let cube_edge_length : ℝ := 2
  let dot_distance_from_center : ℝ := cube_edge_length / 4
  let roll_count : ℕ := 2
  let radius : ℝ := Real.sqrt (1^2 + dot_distance_from_center^2)
  let path_length : ℝ := roll_count * 2 * π * radius
  path_length = 2.236 * π := by sorry

end dot_path_length_on_rolled_cube_l2362_236264


namespace alcohol_percentage_x_is_correct_l2362_236221

/-- The percentage of alcohol by volume in solution x -/
def alcohol_percentage_x : ℝ := 0.10

/-- The percentage of alcohol by volume in solution y -/
def alcohol_percentage_y : ℝ := 0.30

/-- The volume of solution y in milliliters -/
def volume_y : ℝ := 600

/-- The volume of solution x in milliliters -/
def volume_x : ℝ := 200

/-- The percentage of alcohol by volume in the final mixture -/
def alcohol_percentage_final : ℝ := 0.25

theorem alcohol_percentage_x_is_correct :
  alcohol_percentage_x * volume_x + alcohol_percentage_y * volume_y =
  alcohol_percentage_final * (volume_x + volume_y) :=
by sorry

end alcohol_percentage_x_is_correct_l2362_236221


namespace train_speed_l2362_236244

/-- The speed of a train given its length, time to pass a man, and the man's speed in the opposite direction -/
theorem train_speed (train_length : Real) (passing_time : Real) (man_speed : Real) :
  train_length = 240 ∧ 
  passing_time = 13.090909090909092 ∧ 
  man_speed = 6 →
  (train_length / 1000) / (passing_time / 3600) - man_speed = 60 := by
  sorry

#check train_speed

end train_speed_l2362_236244


namespace one_female_selection_count_l2362_236257

/-- The number of male students in Group A -/
def group_a_male : ℕ := 5

/-- The number of female students in Group A -/
def group_a_female : ℕ := 3

/-- The number of male students in Group B -/
def group_b_male : ℕ := 6

/-- The number of female students in Group B -/
def group_b_female : ℕ := 2

/-- The number of students to be selected from each group -/
def students_per_group : ℕ := 2

/-- The total number of selections with exactly one female student -/
def total_selections : ℕ := 345

theorem one_female_selection_count :
  (Nat.choose group_a_male 1 * Nat.choose group_a_female 1 * Nat.choose group_b_male 2) +
  (Nat.choose group_a_male 2 * Nat.choose group_b_male 1 * Nat.choose group_b_female 1) = total_selections :=
by sorry

end one_female_selection_count_l2362_236257


namespace quadratic_negative_root_l2362_236254

theorem quadratic_negative_root (a : ℝ) (h : a < 0) :
  ∃ (condition : Prop), condition → ∃ x : ℝ, x < 0 ∧ a * x^2 + 2*x + 1 = 0 :=
sorry

end quadratic_negative_root_l2362_236254


namespace smallest_other_integer_l2362_236212

theorem smallest_other_integer (m n x : ℕ) : 
  m > 0 → n > 0 → x > 0 →
  (m = 72 ∨ n = 72) →
  Nat.gcd m n = x + 7 →
  Nat.lcm m n = x^2 * (x + 7) →
  (m ≠ 72 → m ≥ 15309) ∧ (n ≠ 72 → n ≥ 15309) :=
sorry

end smallest_other_integer_l2362_236212


namespace seven_digit_integers_count_l2362_236284

/-- The number of different seven-digit integers that can be formed using the digits 1, 2, 2, 3, 3, 3, and 5 -/
def seven_digit_integers : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 3)

/-- Theorem stating that the number of different seven-digit integers
    formed using the digits 1, 2, 2, 3, 3, 3, and 5 is equal to 420 -/
theorem seven_digit_integers_count : seven_digit_integers = 420 := by
  sorry

end seven_digit_integers_count_l2362_236284


namespace watch_correction_l2362_236279

/-- Represents the time difference between two dates in hours -/
def timeDifference (startDate endDate : Nat) : Nat :=
  (endDate - startDate) * 24

/-- Represents the additional hours on the last day -/
def additionalHours (startHour endHour : Nat) : Nat :=
  endHour - startHour

/-- Calculates the total hours elapsed -/
def totalHours (daysDifference additionalHours : Nat) : Nat :=
  daysDifference + additionalHours

/-- Converts daily time loss to hourly time loss -/
def hourlyLoss (dailyLoss : Rat) : Rat :=
  dailyLoss / 24

/-- Calculates the total time loss -/
def totalLoss (hourlyLoss : Rat) (totalHours : Nat) : Rat :=
  hourlyLoss * totalHours

theorem watch_correction (watchLoss : Rat) (startDate endDate startHour endHour : Nat) :
  watchLoss = 3.75 →
  startDate = 15 →
  endDate = 24 →
  startHour = 10 →
  endHour = 16 →
  totalLoss (hourlyLoss watchLoss) (totalHours (timeDifference startDate endDate) (additionalHours startHour endHour)) = 34.6875 := by
  sorry

#check watch_correction

end watch_correction_l2362_236279


namespace isosceles_triangle_arctan_sum_l2362_236201

/-- In an isosceles triangle ABC where AB = AC, 
    arctan(c/(a+b)) + arctan(a/(b+c)) = π/4 -/
theorem isosceles_triangle_arctan_sum (a b c : ℝ) (α : ℝ) :
  b = c →  -- AB = AC implies b = c
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  α > 0 ∧ α < π →  -- Valid angle measure
  Real.arctan (c / (a + b)) + Real.arctan (a / (b + c)) = π / 4 := by
  sorry

end isosceles_triangle_arctan_sum_l2362_236201


namespace tree_spacing_l2362_236240

theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 800 →
  num_trees = 26 →
  num_trees ≥ 2 →
  (yard_length / (num_trees - 1 : ℝ)) = 32 :=
by
  sorry

end tree_spacing_l2362_236240


namespace quadratic_equation_with_roots_as_coefficients_l2362_236205

/-- A quadratic equation with coefficients a, b, and c, represented as ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a quadratic equation -/
structure Roots where
  x₁ : ℝ
  x₂ : ℝ

/-- Checks if the given roots satisfy the quadratic equation -/
def satisfiesEquation (eq : QuadraticEquation) (roots : Roots) : Prop :=
  eq.a * roots.x₁^2 + eq.b * roots.x₁ + eq.c = 0 ∧
  eq.a * roots.x₂^2 + eq.b * roots.x₂ + eq.c = 0

/-- The theorem stating that given a quadratic equation with its roots as coefficients,
    only two specific equations are valid -/
theorem quadratic_equation_with_roots_as_coefficients
  (eq : QuadraticEquation)
  (roots : Roots)
  (h : satisfiesEquation eq roots)
  (h_coeff : eq.a = 1 ∧ eq.b = roots.x₁ ∧ eq.c = roots.x₂) :
  (eq.a = 1 ∧ eq.b = 0 ∧ eq.c = 0) ∨
  (eq.a = 1 ∧ eq.b = 1 ∧ eq.c = -2) :=
sorry

end quadratic_equation_with_roots_as_coefficients_l2362_236205


namespace new_boarders_correct_l2362_236234

-- Define the initial conditions
def initial_boarders : ℕ := 120
def initial_ratio_boarders : ℕ := 2
def initial_ratio_day : ℕ := 5
def new_ratio_boarders : ℕ := 1
def new_ratio_day : ℕ := 2

-- Define the function to calculate the number of new boarders
def new_boarders : ℕ := 30

-- Theorem statement
theorem new_boarders_correct :
  let initial_day_students := (initial_boarders * initial_ratio_day) / initial_ratio_boarders
  (new_ratio_boarders * (initial_boarders + new_boarders)) = (new_ratio_day * initial_day_students) :=
by sorry

end new_boarders_correct_l2362_236234


namespace product_of_solutions_l2362_236213

theorem product_of_solutions (y : ℝ) : 
  (∃ y₁ y₂ : ℝ, (|3 * y₁| = 2 * (|3 * y₁| - 1) ∧ 
                 |3 * y₂| = 2 * (|3 * y₂| - 1) ∧ 
                 y₁ ≠ y₂ ∧
                 (∀ y₃ : ℝ, |3 * y₃| = 2 * (|3 * y₃| - 1) → y₃ = y₁ ∨ y₃ = y₂)) →
                 y₁ * y₂ = -4/9) :=
by sorry

end product_of_solutions_l2362_236213


namespace investment_problem_l2362_236232

/-- Investment problem -/
theorem investment_problem (x y : ℝ) 
  (h1 : 0.06 * x = 0.05 * y + 160)  -- Income difference condition
  (h2 : 0.05 * y = 6000)            -- Income from 5% part
  : x + y = 222666.67 := by          -- Total investment
sorry

end investment_problem_l2362_236232


namespace arithmetic_sequence_formula_l2362_236223

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_term : a 1 = 3
  is_arithmetic : ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d
  is_geometric : ∃ r ≠ 0, (a 4) ^ 2 = (a 1) * (a 13)

/-- The theorem stating the general formula for the sequence -/
theorem arithmetic_sequence_formula (seq : ArithmeticSequence) : 
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, seq.a n = 2 * n + 1 := by
  sorry

end arithmetic_sequence_formula_l2362_236223


namespace x4_plus_y4_equals_7_l2362_236275

theorem x4_plus_y4_equals_7 (x y : ℝ) 
  (hx : x^4 + x^2 = 3) 
  (hy : y^4 - y^2 = 3) : 
  x^4 + y^4 = 7 := by
  sorry

end x4_plus_y4_equals_7_l2362_236275


namespace sqrt_product_simplification_l2362_236216

theorem sqrt_product_simplification : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end sqrt_product_simplification_l2362_236216


namespace minor_arc_intercept_l2362_236226

/-- Given a circle x^2 + y^2 = 4 and a line y = -√3x + b, if the minor arc intercepted
    by the line on the circle corresponds to a central angle of 120°, then b = ±2 -/
theorem minor_arc_intercept (b : ℝ) : 
  (∀ x y, x^2 + y^2 = 4 → y = -Real.sqrt 3 * x + b) →
  (∃ θ, θ = 2 * Real.pi / 3) →
  (b = 2 ∨ b = -2) := by
  sorry

end minor_arc_intercept_l2362_236226


namespace power_division_rule_l2362_236266

theorem power_division_rule (x : ℝ) : x^6 / x^3 = x^3 := by
  sorry

end power_division_rule_l2362_236266


namespace shaded_to_unshaded_ratio_is_five_thirds_l2362_236245

/-- A structure representing a nested square figure -/
structure NestedSquareFigure where
  /-- The number of nested squares in the figure -/
  num_squares : ℕ
  /-- Predicate ensuring inner squares have vertices at midpoints of outer squares -/
  midpoint_property : num_squares > 1 → True

/-- The ratio of shaded to unshaded area in a nested square figure -/
def shaded_to_unshaded_ratio (figure : NestedSquareFigure) : Rat :=
  5 / 3

/-- Theorem stating the ratio of shaded to unshaded area is 5:3 -/
theorem shaded_to_unshaded_ratio_is_five_thirds (figure : NestedSquareFigure) :
  shaded_to_unshaded_ratio figure = 5 / 3 := by
  sorry

end shaded_to_unshaded_ratio_is_five_thirds_l2362_236245


namespace joan_football_games_l2362_236246

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan went to -/
def total_games : ℕ := games_this_year + games_last_year

theorem joan_football_games : total_games = 13 := by
  sorry

end joan_football_games_l2362_236246


namespace solution_characterization_l2362_236256

theorem solution_characterization (x y z : ℝ) :
  (x - y + z)^2 = x^2 - y^2 + z^2 ↔ (x = y ∧ z = 0) ∨ (x = 0 ∧ y = z) := by
  sorry

end solution_characterization_l2362_236256


namespace largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l2362_236217

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 6 = 4 → n ≤ 94 :=
by sorry

theorem ninety_four_satisfies_conditions : 94 < 100 ∧ 94 % 6 = 4 :=
by sorry

theorem ninety_four_is_largest : ∃ (n : ℕ), n = 94 ∧ n < 100 ∧ n % 6 = 4 ∧ ∀ (m : ℕ), m < 100 ∧ m % 6 = 4 → m ≤ n :=
by sorry

end largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l2362_236217


namespace equation_solution_sum_l2362_236294

theorem equation_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 15 = 25) → 
  (d^2 - 6*d + 15 = 25) → 
  c ≥ d → 
  3*c + 2*d = 15 + Real.sqrt 19 := by
sorry

end equation_solution_sum_l2362_236294


namespace exists_right_triangle_with_perpendicular_medians_l2362_236230

/-- A right-angled triangle with one given leg and perpendicular medians to the other two sides -/
structure RightTriangleWithPerpendicularMedians where
  /-- The length of the given leg -/
  a : ℝ
  /-- The length of the second leg -/
  b : ℝ
  /-- The length of the hypotenuse -/
  c : ℝ
  /-- The given leg is positive -/
  a_pos : 0 < a
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagoras : a^2 + b^2 = c^2
  /-- The medians to the other two sides are perpendicular -/
  medians_perpendicular : (2*c^2 + 2*b^2 - a^2) * (2*c^2 + 2*a^2 - b^2) = 9*a^2*b^2

/-- There exists a right-angled triangle with one given leg and perpendicular medians to the other two sides -/
theorem exists_right_triangle_with_perpendicular_medians (a : ℝ) (ha : 0 < a) : 
  ∃ t : RightTriangleWithPerpendicularMedians, t.a = a :=
sorry

end exists_right_triangle_with_perpendicular_medians_l2362_236230


namespace elsa_lost_marbles_l2362_236274

/-- The number of marbles Elsa lost at breakfast -/
def x : ℕ := sorry

/-- Elsa's initial number of marbles -/
def initial_marbles : ℕ := 40

/-- Number of marbles Elsa gave to Susie -/
def marbles_given_to_susie : ℕ := 5

/-- Number of new marbles Elsa's mom bought -/
def new_marbles : ℕ := 12

/-- Elsa's final number of marbles -/
def final_marbles : ℕ := 54

theorem elsa_lost_marbles : 
  initial_marbles - x - marbles_given_to_susie + new_marbles + 2 * marbles_given_to_susie = final_marbles ∧
  x = 3 := by sorry

end elsa_lost_marbles_l2362_236274


namespace sin_20_cos_10_minus_cos_200_sin_10_l2362_236235

open Real

theorem sin_20_cos_10_minus_cos_200_sin_10 :
  sin (20 * π / 180) * cos (10 * π / 180) - cos (200 * π / 180) * sin (10 * π / 180) = 1/2 := by
  sorry

end sin_20_cos_10_minus_cos_200_sin_10_l2362_236235


namespace select_specific_boy_and_girl_probability_l2362_236258

/-- The probability of selecting both boy A and girl B when randomly choosing 1 boy and 2 girls from a group of 8 boys and 3 girls -/
theorem select_specific_boy_and_girl_probability :
  let total_boys : ℕ := 8
  let total_girls : ℕ := 3
  let boys_to_select : ℕ := 1
  let girls_to_select : ℕ := 2
  let total_events : ℕ := (total_boys.choose boys_to_select) * (total_girls.choose girls_to_select)
  let favorable_events : ℕ := 2  -- Only 2 ways to select the other girl
  (favorable_events : ℚ) / total_events = 1 / 12 := by
sorry

end select_specific_boy_and_girl_probability_l2362_236258


namespace cannot_finish_fourth_l2362_236208

-- Define the set of runners
inductive Runner : Type
  | A | B | C | D | E | F | G

-- Define the race result as a function from Runner to Nat (position)
def RaceResult := Runner → Nat

-- Define the conditions of the race
def ValidRaceResult (result : RaceResult) : Prop :=
  (result Runner.A < result Runner.B) ∧
  (result Runner.A < result Runner.C) ∧
  (result Runner.B < result Runner.D) ∧
  (result Runner.C < result Runner.E) ∧
  (result Runner.A < result Runner.F) ∧ (result Runner.F < result Runner.B) ∧
  (result Runner.B < result Runner.G) ∧ (result Runner.G < result Runner.C)

-- Theorem to prove
theorem cannot_finish_fourth (result : RaceResult) 
  (h : ValidRaceResult result) : 
  result Runner.A ≠ 4 ∧ result Runner.F ≠ 4 ∧ result Runner.G ≠ 4 := by
  sorry

end cannot_finish_fourth_l2362_236208


namespace soccer_balls_added_l2362_236273

/-- Given the initial number of soccer balls, the number removed, and the final number of balls,
    prove that the number of soccer balls added is 21. -/
theorem soccer_balls_added 
  (initial : ℕ) 
  (removed : ℕ) 
  (final : ℕ) 
  (h1 : initial = 6) 
  (h2 : removed = 3) 
  (h3 : final = 24) : 
  final - (initial - removed) = 21 := by
  sorry

end soccer_balls_added_l2362_236273


namespace smallest_multiple_of_nine_l2362_236278

theorem smallest_multiple_of_nine (x : ℕ) : x = 18 ↔ 
  (∃ k : ℕ, x = 9 * k) ∧ 
  (x^2 > 200) ∧ 
  (x < Real.sqrt (x^2 - 144) * 5) ∧
  (∀ y : ℕ, y < x → (∃ k : ℕ, y = 9 * k) → 
    (y^2 ≤ 200 ∨ y ≥ Real.sqrt (y^2 - 144) * 5)) :=
by sorry

end smallest_multiple_of_nine_l2362_236278


namespace cannot_form_triangle_5_6_11_l2362_236238

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem cannot_form_triangle_5_6_11 :
  ¬ can_form_triangle 5 6 11 := by
  sorry

end cannot_form_triangle_5_6_11_l2362_236238


namespace cubic_equation_solutions_l2362_236211

theorem cubic_equation_solutions :
  let z₁ : ℂ := -3
  let z₂ : ℂ := (3/2) + (3/2) * Complex.I * Real.sqrt 3
  let z₃ : ℂ := (3/2) - (3/2) * Complex.I * Real.sqrt 3
  (∀ z : ℂ, z^3 = -27 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) := by
  sorry

end cubic_equation_solutions_l2362_236211


namespace consecutive_integers_product_210_l2362_236250

theorem consecutive_integers_product_210 (n : ℤ) :
  n * (n + 1) * (n + 2) = 210 → n + (n + 1) = 11 := by
  sorry

end consecutive_integers_product_210_l2362_236250


namespace even_function_negative_domain_l2362_236285

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_negative_domain
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_positive : ∀ x > 0, f x = x) :
  ∀ x < 0, f x = -x :=
by sorry

end even_function_negative_domain_l2362_236285


namespace area_of_triangle_ABC_l2362_236269

-- Define the triangle ABC and related points
variable (A B C D E F : ℝ × ℝ)
variable (α : ℝ)

-- Define the conditions
axiom right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
axiom parallel_line : (D.2 - E.2) / (D.1 - E.1) = (B.2 - A.2) / (B.1 - A.1)
axiom DE_length : Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = 2
axiom BE_length : Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2) = 1
axiom BF_length : Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 1
axiom F_on_hypotenuse : (F.1 - A.1) / (B.1 - A.1) = (F.2 - A.2) / (B.2 - A.2)
axiom angle_FCB : Real.cos α = (F.1 - C.1) / Real.sqrt ((F.1 - C.1)^2 + (F.2 - C.2)^2)

-- Define the theorem
theorem area_of_triangle_ABC :
  (1/2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) =
  (1/2) * (2 * Real.cos (2*α) + 1)^2 * Real.tan (2*α) := by sorry

end area_of_triangle_ABC_l2362_236269


namespace pencil_case_notebook_prices_l2362_236210

theorem pencil_case_notebook_prices :
  ∀ (notebook_price pencil_case_price : ℚ),
    pencil_case_price = notebook_price + 3 →
    (200 : ℚ) / notebook_price = (350 : ℚ) / pencil_case_price →
    notebook_price = 4 ∧ pencil_case_price = 7 := by
  sorry

end pencil_case_notebook_prices_l2362_236210


namespace all_dihedral_angles_equal_all_polyhedral_angles_equal_l2362_236252

/-- A nearly regular polyhedron -/
structure NearlyRegularPolyhedron where
  /-- The polyhedron has a high degree of symmetry -/
  high_symmetry : Prop
  /-- Each face is a regular polygon -/
  regular_faces : Prop
  /-- Faces are arranged symmetrically around each vertex -/
  symmetric_face_arrangement : Prop
  /-- The polyhedron has vertex-transitivity property -/
  vertex_transitivity : Prop

/-- Dihedral angle of a polyhedron -/
def dihedral_angle (P : NearlyRegularPolyhedron) : Type := sorry

/-- Polyhedral angle of a polyhedron -/
def polyhedral_angle (P : NearlyRegularPolyhedron) : Type := sorry

/-- Theorem stating that all dihedral angles of a nearly regular polyhedron are equal -/
theorem all_dihedral_angles_equal (P : NearlyRegularPolyhedron) :
  ∀ a b : dihedral_angle P, a = b :=
sorry

/-- Theorem stating that all polyhedral angles of a nearly regular polyhedron are equal -/
theorem all_polyhedral_angles_equal (P : NearlyRegularPolyhedron) :
  ∀ a b : polyhedral_angle P, a = b :=
sorry

end all_dihedral_angles_equal_all_polyhedral_angles_equal_l2362_236252


namespace sharp_2_5_3_equals_1_l2362_236202

-- Define the # operation for real numbers
def sharp (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Theorem statement
theorem sharp_2_5_3_equals_1 : sharp 2 5 3 = 1 := by sorry

end sharp_2_5_3_equals_1_l2362_236202


namespace no_five_naturals_product_equals_sum_l2362_236277

theorem no_five_naturals_product_equals_sum :
  ¬ ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ d * e = a + b + c + d + e := by
sorry

end no_five_naturals_product_equals_sum_l2362_236277


namespace modified_lucas_105_mod_9_l2362_236255

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => modifiedLucas n + modifiedLucas (n + 1)

theorem modified_lucas_105_mod_9 : modifiedLucas 104 % 9 = 8 := by
  sorry

end modified_lucas_105_mod_9_l2362_236255


namespace triangle_sine_inequality_l2362_236209

theorem triangle_sine_inequality (A B C : Real) :
  A + B + C = 180 →
  0 < A ∧ A ≤ 180 →
  0 < B ∧ B ≤ 180 →
  0 < C ∧ C ≤ 180 →
  Real.sin ((A - 30) * π / 180) + Real.sin ((B - 30) * π / 180) + Real.sin ((C - 30) * π / 180) ≤ 3/2 :=
by sorry

end triangle_sine_inequality_l2362_236209


namespace solution_range_l2362_236214

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1/4)^x + (1/2)^(x-1) + a = 0) → 
  -3 < a ∧ a < 0 :=
by sorry

end solution_range_l2362_236214


namespace z_in_second_quadrant_l2362_236218

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation that z satisfies
def equation (z : ℂ) : Prop := (1 + i) * z = 1 - 2 * i^3

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end z_in_second_quadrant_l2362_236218


namespace tom_distance_covered_l2362_236224

theorem tom_distance_covered (swim_time : ℝ) (swim_speed : ℝ) : 
  swim_time = 2 →
  swim_speed = 2 →
  let run_time := swim_time / 2
  let run_speed := 4 * swim_speed
  let swim_distance := swim_time * swim_speed
  let run_distance := run_time * run_speed
  swim_distance + run_distance = 12 := by
  sorry

end tom_distance_covered_l2362_236224


namespace no_geometric_sequence_satisfies_conditions_l2362_236247

theorem no_geometric_sequence_satisfies_conditions :
  ¬ ∃ (a : ℕ → ℝ) (q : ℝ),
    (∀ n : ℕ, a (n + 1) = q * a n) ∧  -- geometric sequence
    (a 1 + a 6 = 11) ∧  -- condition 1
    (a 3 * a 4 = 32 / 9) ∧  -- condition 1
    (∀ n : ℕ, a (n + 1) > a n) ∧  -- condition 2
    (∃ m : ℕ, m > 4 ∧ 
      2 * (a m)^2 = 2/3 * a (m - 1) + (a (m + 1) + 4/9)) :=  -- condition 3
by sorry

end no_geometric_sequence_satisfies_conditions_l2362_236247


namespace alan_cd_purchase_cost_l2362_236297

theorem alan_cd_purchase_cost :
  let avnPrice : ℝ := 12
  let darkPrice : ℝ := 2 * avnPrice
  let darkTotal : ℝ := 2 * darkPrice
  let otherTotal : ℝ := darkTotal + avnPrice
  let ninetyPrice : ℝ := 0.4 * otherTotal
  darkTotal + avnPrice + ninetyPrice = 84 := by
  sorry

end alan_cd_purchase_cost_l2362_236297


namespace isosceles_triangle_l2362_236261

/-- A triangle is isosceles if it has at least two equal sides -/
def IsIsosceles (A B C : ℝ × ℝ) : Prop :=
  (dist A B = dist B C) ∨ (dist B C = dist A C) ∨ (dist A C = dist A B)

/-- The perimeter of a triangle is the sum of the lengths of its sides -/
def Perimeter (A B C : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist A C

theorem isosceles_triangle (A B C M N : ℝ × ℝ) :
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = (1 - t) • A + t • B) →  -- M is on AB
  (∃ s : ℝ, 0 < s ∧ s < 1 ∧ N = (1 - s) • B + s • C) →  -- N is on BC
  Perimeter A M C = Perimeter C N A →                  -- Perimeter condition 1
  Perimeter A N B = Perimeter C M B →                  -- Perimeter condition 2
  IsIsosceles A B C :=                                 -- Conclusion
by sorry

end isosceles_triangle_l2362_236261


namespace cube_color_probability_l2362_236239

-- Define the colors
inductive Color
| Black
| White
| Gray

-- Define a cube as a list of 6 colors
def Cube := List Color

-- Function to check if a cube meets the conditions
def meetsConditions (cube : Cube) : Bool :=
  sorry

-- Probability of a specific color
def colorProb : ℚ := 1 / 3

-- Total number of possible cube colorings
def totalColorings : ℕ := 729

-- Number of colorings that meet the conditions
def validColorings : ℕ := 39

theorem cube_color_probability :
  (validColorings : ℚ) / totalColorings = 13 / 243 := by
  sorry

end cube_color_probability_l2362_236239


namespace symmetric_line_theorem_l2362_236231

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two points are symmetric with respect to a vertical line -/
def symmetric_points (x₁ y₁ x₂ y₂ x_sym : ℝ) : Prop :=
  x₁ + x₂ = 2 * x_sym ∧ y₁ = y₂

/-- Check if a point (x, y) lies on a given line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are symmetric with respect to a vertical line -/
def symmetric_lines (l₁ l₂ : Line) (x_sym : ℝ) : Prop :=
  ∀ x₁ y₁, point_on_line x₁ y₁ l₁ →
    ∃ x₂ y₂, point_on_line x₂ y₂ l₂ ∧ symmetric_points x₁ y₁ x₂ y₂ x_sym

theorem symmetric_line_theorem :
  let l₁ : Line := ⟨2, -1, 1⟩
  let l₂ : Line := ⟨2, 1, -5⟩
  let x_sym : ℝ := 1
  symmetric_lines l₁ l₂ x_sym := by sorry

end symmetric_line_theorem_l2362_236231


namespace even_sum_theorem_l2362_236288

theorem even_sum_theorem (n : ℕ) (h1 : Odd n) 
  (h2 : (Finset.sum (Finset.filter Even (Finset.range n)) id) = 95 * 96) : 
  n = 191 := by
  sorry

end even_sum_theorem_l2362_236288


namespace sandwich_theorem_l2362_236229

/-- The number of sandwiches Samson ate on different days and meals --/
def sandwich_count : Prop :=
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let monday_total := monday_lunch + monday_dinner
  let tuesday_total := tuesday_lunch + tuesday_dinner
  let wednesday_total := wednesday_lunch + wednesday_dinner
  wednesday_total - (monday_total + tuesday_total) = 5

theorem sandwich_theorem : sandwich_count := by
  sorry

end sandwich_theorem_l2362_236229


namespace boat_travel_l2362_236236

theorem boat_travel (boat_speed : ℝ) (time_against : ℝ) (time_with : ℝ)
  (h1 : boat_speed = 12)
  (h2 : time_against = 10)
  (h3 : time_with = 6) :
  ∃ (current_speed : ℝ) (distance : ℝ),
    current_speed = 3 ∧
    distance = 90 ∧
    (boat_speed - current_speed) * time_against = (boat_speed + current_speed) * time_with :=
by sorry

end boat_travel_l2362_236236


namespace find_x1_l2362_236290

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h_order : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h_eq1 : (1-x1)^2 + (x1-x2)^2 + (x2-x3)^2 + (x3-x4)^2 + x4^2 = 1/3)
  (h_sum : x1 + x2 + x3 + x4 = 2) : 
  x1 = 4/5 := by
sorry

end find_x1_l2362_236290


namespace rectangle_length_breadth_difference_l2362_236271

/-- Given a rectangular plot with breadth 11 metres and area 21 times its breadth,
    the difference between its length and breadth is 10 metres. -/
theorem rectangle_length_breadth_difference : ℝ → Prop :=
  fun difference =>
    ∀ (length breadth area : ℝ),
      breadth = 11 →
      area = 21 * breadth →
      area = length * breadth →
      difference = length - breadth →
      difference = 10

/-- Proof of the theorem -/
lemma prove_rectangle_length_breadth_difference :
  rectangle_length_breadth_difference 10 := by
  sorry

end rectangle_length_breadth_difference_l2362_236271


namespace tony_running_distance_tony_running_distance_proof_l2362_236267

/-- Proves that Tony runs 10 miles without the backpack each morning given his exercise routine. -/
theorem tony_running_distance : ℝ → Prop :=
  fun x =>
    let walk_distance : ℝ := 3
    let walk_speed : ℝ := 3
    let run_speed : ℝ := 5
    let total_exercise_time : ℝ := 21
    let days_per_week : ℝ := 7
    
    let daily_walk_time : ℝ := walk_distance / walk_speed
    let daily_run_time : ℝ := x / run_speed
    let weekly_exercise_time : ℝ := days_per_week * (daily_walk_time + daily_run_time)
    
    weekly_exercise_time = total_exercise_time → x = 10

/-- The proof of the theorem. -/
theorem tony_running_distance_proof : tony_running_distance 10 := by
  sorry

end tony_running_distance_tony_running_distance_proof_l2362_236267


namespace pedestrian_cyclist_speeds_l2362_236215

theorem pedestrian_cyclist_speeds
  (distance : ℝ)
  (pedestrian_start : ℝ)
  (cyclist1_start : ℝ)
  (cyclist2_start : ℝ)
  (pedestrian_speed : ℝ)
  (cyclist_speed : ℝ)
  (h1 : distance = 40)
  (h2 : cyclist1_start - pedestrian_start = 10/3)
  (h3 : cyclist2_start - pedestrian_start = 4.5)
  (h4 : pedestrian_speed * ((cyclist1_start - pedestrian_start) + (distance/2 - pedestrian_speed * (cyclist1_start - pedestrian_start)) / (cyclist_speed - pedestrian_speed)) = distance/2)
  (h5 : pedestrian_speed * ((cyclist1_start - pedestrian_start) + (distance/2 - pedestrian_speed * (cyclist1_start - pedestrian_start)) / (cyclist_speed - pedestrian_speed) + 1) + cyclist_speed * ((cyclist2_start - pedestrian_start) - ((cyclist1_start - pedestrian_start) + (distance/2 - pedestrian_speed * (cyclist1_start - pedestrian_start)) / (cyclist_speed - pedestrian_speed) + 1)) = distance)
  : pedestrian_speed = 5 ∧ cyclist_speed = 30 := by
  sorry

end pedestrian_cyclist_speeds_l2362_236215


namespace sum_of_integers_l2362_236253

theorem sum_of_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4 →
  m + n + p + q = 24 := by
sorry

end sum_of_integers_l2362_236253


namespace pascal_triangle_53_l2362_236299

theorem pascal_triangle_53 (p : ℕ) (h_prime : Prime p) (h_p : p = 53) :
  (∃! n : ℕ, ∃ k : ℕ, Nat.choose n k = p) :=
sorry

end pascal_triangle_53_l2362_236299


namespace expression_simplification_l2362_236259

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 3) / x / ((x^2 - 6*x + 9) / (x^2 - 9)) - (x + 1) / x = Real.sqrt 2 :=
by sorry

end expression_simplification_l2362_236259


namespace batch_size_proof_l2362_236203

theorem batch_size_proof :
  ∃! n : ℕ, 500 ≤ n ∧ n ≤ 600 ∧ n % 20 = 13 ∧ n % 27 = 20 ∧ n = 533 := by
  sorry

end batch_size_proof_l2362_236203


namespace eleven_students_in_line_l2362_236233

/-- The number of students in a line, given Yoonjung's position -/
def total_students (students_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  students_in_front + 1 + (position_from_back - 1)

/-- Theorem: There are 11 students in the line -/
theorem eleven_students_in_line : 
  total_students 6 5 = 11 := by
  sorry

end eleven_students_in_line_l2362_236233


namespace max_salary_in_semipro_league_l2362_236286

/-- Represents a baseball team -/
structure Team where
  players : Nat
  minSalary : Nat
  maxTotalSalary : Nat

/-- Calculates the maximum possible salary for a single player in a team -/
def maxSinglePlayerSalary (team : Team) : Nat :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player in the given conditions -/
theorem max_salary_in_semipro_league :
  let team : Team := {
    players := 21,
    minSalary := 15000,
    maxTotalSalary := 700000
  }
  maxSinglePlayerSalary team = 400000 := by sorry

end max_salary_in_semipro_league_l2362_236286


namespace triangle_abc_properties_l2362_236260

open Real

theorem triangle_abc_properties (a b c A B C : ℝ) (k : ℤ) :
  -- Conditions
  (2 * Real.sqrt 3 * a * Real.sin C * Real.sin B = a * Real.sin A + b * Real.sin B - c * Real.sin C) →
  (a * Real.cos (π / 2 - B) = b * Real.cos (2 * ↑k * π + A)) →
  (a = 2) →
  -- Conclusions
  (C = π / 6) ∧
  (1 / 2 * a * c * Real.sin B = (1 + Real.sqrt 3) / 2) :=
by sorry

end triangle_abc_properties_l2362_236260


namespace ratio_sum_theorem_l2362_236227

theorem ratio_sum_theorem (a b c : ℝ) (h : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  (a + b + c) / c = 12 / 5 := by sorry

end ratio_sum_theorem_l2362_236227


namespace lines_parallel_iff_a_eq_neg_three_l2362_236283

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 n1 : ℝ) (m2 n2 : ℝ) : Prop := m1 * n2 = m2 * n1

/-- The line ax+3y+1=0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 1 = 0

/-- The line 2x+(a+1)y+1=0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := 2 * x + (a + 1) * y + 1 = 0

/-- The main theorem: the lines are parallel if and only if a = -3 -/
theorem lines_parallel_iff_a_eq_neg_three (a : ℝ) : 
  parallel a 3 2 (a + 1) ↔ a = -3 := by sorry

end lines_parallel_iff_a_eq_neg_three_l2362_236283


namespace two_numbers_subtracted_from_32_l2362_236289

theorem two_numbers_subtracted_from_32 : ∃ (A B : ℤ), 
  A ≠ B ∧
  ((32 - A = 23 ∧ 32 - B = 13) ∨ (32 - A = 13 ∧ 32 - B = 23)) ∧
  ¬ (∃ (k : ℤ), |A - B| = 11 * k) ∧
  A = 9 ∧ B = 19 := by
sorry

end two_numbers_subtracted_from_32_l2362_236289


namespace cylinder_from_equation_l2362_236298

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = d -/
def CylindricalSet (d : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = d}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ d : ℝ, d > 0 ∧ S = CylindricalSet d

/-- Theorem: The set of points satisfying r = d forms a cylinder -/
theorem cylinder_from_equation (d : ℝ) (h : d > 0) : 
  IsCylinder (CylindricalSet d) := by
  sorry

end cylinder_from_equation_l2362_236298


namespace ten_factorial_divided_by_nine_factorial_l2362_236268

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem ten_factorial_divided_by_nine_factorial : 
  factorial 10 / factorial 9 = 10 := by
  sorry

end ten_factorial_divided_by_nine_factorial_l2362_236268


namespace solution_x_proportion_l2362_236200

/-- Represents a solution with a certain percentage of material a -/
structure Solution :=
  (a_percent : ℝ)

/-- Represents the mixture of solutions -/
structure Mixture :=
  (x y z : ℝ)
  (x_sol y_sol z_sol : Solution)

/-- The conditions of the problem -/
def problem_conditions (m : Mixture) : Prop :=
  m.x_sol.a_percent = 0.2 ∧
  m.y_sol.a_percent = 0.3 ∧
  m.z_sol.a_percent = 0.4 ∧
  m.x_sol.a_percent * m.x + m.y_sol.a_percent * m.y + m.z_sol.a_percent * m.z = 0.25 * (m.x + m.y + m.z) ∧
  m.y = 1.5 * m.z ∧
  m.x > 0 ∧ m.y > 0 ∧ m.z > 0

/-- The theorem to be proved -/
theorem solution_x_proportion (m : Mixture) : 
  problem_conditions m → m.x / (m.x + m.y + m.z) = 9 / 14 := by
  sorry

end solution_x_proportion_l2362_236200


namespace jake_work_hours_l2362_236291

/-- Calculates the number of hours needed to work off a debt -/
def hoursToWorkOff (initialDebt : ℚ) (amountPaid : ℚ) (hourlyRate : ℚ) : ℚ :=
  (initialDebt - amountPaid) / hourlyRate

/-- Proves that Jake needs to work 4 hours to pay off his debt -/
theorem jake_work_hours :
  let initialDebt : ℚ := 100
  let amountPaid : ℚ := 40
  let hourlyRate : ℚ := 15
  hoursToWorkOff initialDebt amountPaid hourlyRate = 4 := by
  sorry

end jake_work_hours_l2362_236291


namespace largest_non_sum_30_and_composite_l2362_236292

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a number can be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
def isSum30AndComposite (n : ℕ) : Prop :=
  ∃ k c, 0 < k ∧ isComposite c ∧ n = 30 * k + c

/-- Theorem stating that 210 is the largest positive integer that cannot be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
theorem largest_non_sum_30_and_composite :
  (∀ n : ℕ, 210 < n → isSum30AndComposite n) ∧
  ¬isSum30AndComposite 210 :=
sorry

end largest_non_sum_30_and_composite_l2362_236292


namespace lcm_18_27_l2362_236206

theorem lcm_18_27 : Nat.lcm 18 27 = 54 := by
  sorry

end lcm_18_27_l2362_236206


namespace complex_modulus_l2362_236225

theorem complex_modulus (z : ℂ) (h : z * (1 - Complex.I) = 2 - 3 * Complex.I) :
  Complex.abs z = Real.sqrt 26 / 2 := by
  sorry

end complex_modulus_l2362_236225


namespace proposition_relationship_l2362_236276

theorem proposition_relationship :
  (∀ x : ℝ, (x - 3) * (x + 1) > 0 → x^2 - 2*x + 1 > 0) ∧
  (∃ x : ℝ, x^2 - 2*x + 1 > 0 ∧ (x - 3) * (x + 1) ≤ 0) :=
by sorry

end proposition_relationship_l2362_236276


namespace initial_marbles_relationship_l2362_236251

/-- Represents the marble collection problem --/
structure MarbleCollection where
  initial : ℕ  -- Initial number of marbles
  lost : ℕ     -- Number of marbles lost
  found : ℕ    -- Number of marbles found
  current : ℕ  -- Current number of marbles after losses and finds

/-- The marble collection satisfies the problem conditions --/
def validCollection (m : MarbleCollection) : Prop :=
  m.lost = 16 ∧ m.found = 8 ∧ m.lost - m.found = 8 ∧ m.current = m.initial - m.lost + m.found

/-- Theorem stating the relationship between initial and current marbles --/
theorem initial_marbles_relationship (m : MarbleCollection) 
  (h : validCollection m) : m.initial = m.current + 8 := by
  sorry

#check initial_marbles_relationship

end initial_marbles_relationship_l2362_236251


namespace tameka_cracker_sales_l2362_236281

/-- Proves that given the conditions in the problem, Tameka sold 30 more boxes on Saturday than on Friday --/
theorem tameka_cracker_sales : ∀ (saturday_sales : ℕ),
  (40 + saturday_sales + saturday_sales / 2 = 145) →
  (saturday_sales = 40 + 30) := by
  sorry

end tameka_cracker_sales_l2362_236281


namespace smallest_dual_base_palindrome_l2362_236270

/-- A function that checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- A function that converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ n : ℕ,
    n > 10 →
    (isPalindrome n 2 ∧ isPalindrome n 4) →
    n ≥ 15 :=
by sorry

end smallest_dual_base_palindrome_l2362_236270


namespace probability_three_white_balls_l2362_236204

def total_balls : ℕ := 11
def white_balls : ℕ := 4
def black_balls : ℕ := 7
def drawn_balls : ℕ := 3

def probability_all_white : ℚ :=
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ)

theorem probability_three_white_balls :
  probability_all_white = 4 / 165 := by
  sorry

end probability_three_white_balls_l2362_236204


namespace cube_root_equation_sum_l2362_236248

theorem cube_root_equation_sum (x y z : ℕ+) :
  (4 : ℝ) * ((7 : ℝ)^(1/3) - (6 : ℝ)^(1/3))^(1/2) = x.val^(1/3) + y.val^(1/3) - z.val^(1/3) →
  x.val + y.val + z.val = 51 := by
sorry

end cube_root_equation_sum_l2362_236248


namespace fixed_cost_satisfies_break_even_equation_l2362_236280

/-- The one-time fixed cost for a book publishing project -/
def fixed_cost : ℝ := 35678

/-- The variable cost per book -/
def variable_cost_per_book : ℝ := 11.50

/-- The selling price per book -/
def selling_price_per_book : ℝ := 20.25

/-- The number of books needed to break even -/
def break_even_quantity : ℕ := 4072

/-- Theorem stating that the fixed cost satisfies the break-even equation -/
theorem fixed_cost_satisfies_break_even_equation : 
  fixed_cost + (break_even_quantity : ℝ) * variable_cost_per_book = 
  (break_even_quantity : ℝ) * selling_price_per_book :=
by sorry

end fixed_cost_satisfies_break_even_equation_l2362_236280


namespace range_of_expression_l2362_236228

theorem range_of_expression (α β : ℝ) 
  (h_α : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h_β : β ∈ Set.Icc 0 (Real.pi / 2)) :
  ∃ (x : ℝ), x ∈ Set.Ioo (-Real.pi / 6) Real.pi ∧
  ∃ (α' β' : ℝ), α' ∈ Set.Ioo 0 (Real.pi / 2) ∧
                 β' ∈ Set.Icc 0 (Real.pi / 2) ∧
                 x = 2 * α' - β' / 3 :=
by sorry

end range_of_expression_l2362_236228
