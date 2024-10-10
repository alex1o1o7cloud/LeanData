import Mathlib

namespace value_of_a_l1649_164955

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + 2 = 0}

-- Theorem statement
theorem value_of_a (a : ℝ) : B a ⊆ A ∧ B a ≠ ∅ → a = 3 := by
  sorry

end value_of_a_l1649_164955


namespace correct_fraction_proof_l1649_164936

theorem correct_fraction_proof (x y : ℚ) : 
  (5 / 6 : ℚ) * 288 = x / y * 288 + 150 → x / y = 5 / 32 := by
sorry

end correct_fraction_proof_l1649_164936


namespace power_function_through_point_l1649_164929

/-- A power function passing through a specific point -/
def isPowerFunctionThroughPoint (m : ℝ) : Prop :=
  ∃ (y : ℝ → ℝ), (∀ x, y x = (m^2 - 3*m + 3) * x^m) ∧ y 2 = 4

/-- The value of m for which the power function passes through (2, 4) -/
theorem power_function_through_point :
  ∃ (m : ℝ), isPowerFunctionThroughPoint m ∧ m = 2 := by
  sorry

end power_function_through_point_l1649_164929


namespace longest_piece_length_l1649_164907

theorem longest_piece_length (rope1 rope2 rope3 : ℕ) 
  (h1 : rope1 = 75)
  (h2 : rope2 = 90)
  (h3 : rope3 = 135) : 
  Nat.gcd rope1 (Nat.gcd rope2 rope3) = 15 := by
  sorry

end longest_piece_length_l1649_164907


namespace greatest_digit_sum_base5_max_digit_sum_attainable_l1649_164988

/-- Given a positive integer n, returns the sum of its digits in base 5 representation -/
def sumOfDigitsBase5 (n : ℕ) : ℕ := sorry

/-- The greatest possible sum of digits in base 5 for integers less than 3139 -/
def maxDigitSum : ℕ := 16

theorem greatest_digit_sum_base5 :
  ∀ n : ℕ, n > 0 ∧ n < 3139 → sumOfDigitsBase5 n ≤ maxDigitSum :=
by sorry

theorem max_digit_sum_attainable :
  ∃ n : ℕ, n > 0 ∧ n < 3139 ∧ sumOfDigitsBase5 n = maxDigitSum :=
by sorry

end greatest_digit_sum_base5_max_digit_sum_attainable_l1649_164988


namespace magnets_ratio_l1649_164951

theorem magnets_ratio (adam_initial : ℕ) (peter : ℕ) (adam_final : ℕ) : 
  adam_initial = 18 →
  peter = 24 →
  adam_final = peter / 2 →
  adam_final = adam_initial - (adam_initial - adam_final) →
  (adam_initial - adam_final : ℚ) / adam_initial = 1 / 3 := by
  sorry

end magnets_ratio_l1649_164951


namespace car_journey_distance_l1649_164947

theorem car_journey_distance :
  ∀ (v : ℝ),
  v > 0 →
  v * 7 = (v + 12) * 5 →
  v * 7 = 210 :=
by
  sorry

end car_journey_distance_l1649_164947


namespace mutually_exclusive_events_l1649_164926

/-- The set of integers from which we select numbers -/
def S : Set ℕ := {1, 2, 3, 4, 5, 6}

/-- A pair of numbers selected from S -/
def Selection := (ℕ × ℕ)

/-- Predicate for a number being even -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Predicate for a number being odd -/
def is_odd (n : ℕ) : Prop := n % 2 ≠ 0

/-- Event: Exactly one is even and exactly one is odd -/
def event1 (s : Selection) : Prop :=
  (is_even s.1 ∧ is_odd s.2) ∨ (is_odd s.1 ∧ is_even s.2)

/-- Event: At least one is odd and both are odd -/
def event2 (s : Selection) : Prop :=
  is_odd s.1 ∧ is_odd s.2

/-- Event: At least one is odd and both are even -/
def event3 (s : Selection) : Prop :=
  (is_odd s.1 ∨ is_odd s.2) ∧ (is_even s.1 ∧ is_even s.2)

/-- Event: At least one is odd and at least one is even -/
def event4 (s : Selection) : Prop :=
  (is_odd s.1 ∨ is_odd s.2) ∧ (is_even s.1 ∨ is_even s.2)

theorem mutually_exclusive_events :
  ∀ (s : Selection), s.1 ∈ S ∧ s.2 ∈ S →
    (¬(event1 s ∧ event2 s) ∧
     ¬(event1 s ∧ event3 s) ∧
     ¬(event1 s ∧ event4 s) ∧
     ¬(event2 s ∧ event3 s) ∧
     ¬(event2 s ∧ event4 s) ∧
     ¬(event3 s ∧ event4 s)) ∧
    (event3 s → ¬event1 s ∧ ¬event2 s ∧ ¬event4 s) :=
by sorry

end mutually_exclusive_events_l1649_164926


namespace quadratic_completion_square_constant_term_value_l1649_164981

theorem quadratic_completion_square (x : ℝ) : 
  x^2 - 8*x + 3 = (x - 4)^2 - 13 :=
by sorry

theorem constant_term_value : 
  ∃ (a h : ℝ), ∀ (x : ℝ), x^2 - 8*x + 3 = a*(x - h)^2 - 13 :=
by sorry

end quadratic_completion_square_constant_term_value_l1649_164981


namespace remaining_cube_volume_l1649_164991

/-- The remaining volume of a cube after removing two perpendicular cylindrical sections. -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) 
  (h_cube_side : cube_side = 6)
  (h_cylinder_radius : cylinder_radius = 1) :
  cube_side ^ 3 - 2 * π * cylinder_radius ^ 2 * cube_side = 216 - 12 * π := by
  sorry

#check remaining_cube_volume

end remaining_cube_volume_l1649_164991


namespace smallest_number_l1649_164924

def number_set : Set ℤ := {1, 0, -2, -6}

theorem smallest_number :
  ∀ x ∈ number_set, -6 ≤ x :=
by sorry

end smallest_number_l1649_164924


namespace road_signs_count_l1649_164964

/-- The number of road signs at the first intersection -/
def first_intersection : ℕ := 40

/-- The number of road signs at the second intersection -/
def second_intersection : ℕ := first_intersection + (first_intersection / 4)

/-- The number of road signs at the third intersection -/
def third_intersection : ℕ := 2 * second_intersection

/-- The number of road signs at the fourth intersection -/
def fourth_intersection : ℕ := third_intersection - 20

/-- The total number of road signs at all four intersections -/
def total_road_signs : ℕ := first_intersection + second_intersection + third_intersection + fourth_intersection

theorem road_signs_count : total_road_signs = 270 := by
  sorry

end road_signs_count_l1649_164964


namespace platform_length_l1649_164904

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and a signal pole in 26 seconds, the length of the platform is 150 meters. -/
theorem platform_length (train_length : ℝ) (platform_cross_time : ℝ) (pole_cross_time : ℝ) :
  train_length = 300 →
  platform_cross_time = 39 →
  pole_cross_time = 26 →
  ∃ platform_length : ℝ,
    platform_length = 150 ∧
    (train_length / pole_cross_time) * platform_cross_time = train_length + platform_length :=
by sorry

end platform_length_l1649_164904


namespace swimmer_speed_in_still_water_l1649_164969

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  swimmer : ℝ
  stream : ℝ

/-- Calculates the effective speed of the swimmer given the direction. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Represents the conditions of the swimming problem. -/
structure SwimmingProblem where
  downstreamDistance : ℝ
  upstreamDistance : ℝ
  time : ℝ

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 6 km/h. -/
theorem swimmer_speed_in_still_water (p : SwimmingProblem)
  (h1 : p.downstreamDistance = 72)
  (h2 : p.upstreamDistance = 36)
  (h3 : p.time = 9)
  : ∃ (s : SwimmerSpeeds),
    effectiveSpeed s true * p.time = p.downstreamDistance ∧
    effectiveSpeed s false * p.time = p.upstreamDistance ∧
    s.swimmer = 6 := by
  sorry

end swimmer_speed_in_still_water_l1649_164969


namespace stratified_sample_size_l1649_164952

/-- Represents the quantity of each product model in a sample -/
structure ProductSample where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Defines the ratio of quantities for products A, B, and C -/
def quantity_ratio : ProductSample := ⟨2, 3, 5⟩

/-- Calculates the total sample size -/
def total_sample_size (s : ProductSample) : ℕ := s.a + s.b + s.c

/-- Theorem: If a stratified sample with the given ratio contains 16 units of model A, 
    then the total sample size is 80 -/
theorem stratified_sample_size 
  (sample : ProductSample)
  (h_ratio : ∃ k : ℕ, sample.a = k * quantity_ratio.a ∧ 
                      sample.b = k * quantity_ratio.b ∧ 
                      sample.c = k * quantity_ratio.c)
  (h_model_a : sample.a = 16) :
  total_sample_size sample = 80 := by
sorry

end stratified_sample_size_l1649_164952


namespace count_theorem_l1649_164979

/-- The count of positive integers less than 3000 with at most three different digits -/
def count_numbers_with_at_most_three_digits : ℕ :=
  -- Definition placeholder
  0

/-- The upper bound of the considered range -/
def upper_bound : ℕ := 3000

/-- Predicate to check if a number has at most three different digits -/
def has_at_most_three_digits (n : ℕ) : Prop :=
  -- Definition placeholder
  True

theorem count_theorem :
  count_numbers_with_at_most_three_digits = 891 :=
sorry


end count_theorem_l1649_164979


namespace bottle_caps_per_visit_l1649_164921

def store_visits : ℕ := 5
def total_bottle_caps : ℕ := 25

theorem bottle_caps_per_visit :
  total_bottle_caps / store_visits = 5 :=
by sorry

end bottle_caps_per_visit_l1649_164921


namespace sqrt_neg_two_squared_l1649_164915

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by
  sorry

end sqrt_neg_two_squared_l1649_164915


namespace least_n_with_gcd_conditions_l1649_164968

theorem least_n_with_gcd_conditions : 
  ∃ n : ℕ, n > 500 ∧ 
    Nat.gcd 42 (n + 80) = 14 ∧ 
    Nat.gcd (n + 42) 80 = 40 ∧
    (∀ m : ℕ, m > 500 → 
      Nat.gcd 42 (m + 80) = 14 → 
      Nat.gcd (m + 42) 80 = 40 → 
      n ≤ m) ∧
    n = 638 :=
by sorry

end least_n_with_gcd_conditions_l1649_164968


namespace chocolate_box_count_l1649_164990

def chocolate_problem (total_bars : ℕ) : Prop :=
  let bar_cost : ℕ := 3
  let unsold_bars : ℕ := 4
  let revenue : ℕ := 9
  (total_bars - unsold_bars) * bar_cost = revenue

theorem chocolate_box_count : ∃ (n : ℕ), chocolate_problem n ∧ n = 7 := by
  sorry

end chocolate_box_count_l1649_164990


namespace initials_count_l1649_164903

/-- The number of letters available for initials -/
def num_letters : ℕ := 10

/-- The number of initials in each set -/
def initials_per_set : ℕ := 3

/-- The total number of possible three-letter sets of initials -/
def total_sets : ℕ := num_letters ^ initials_per_set

theorem initials_count : total_sets = 1000 := by
  sorry

end initials_count_l1649_164903


namespace function_expression_l1649_164901

-- Define the function f
noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x => (2 * b * x) / (a * x - 1)

-- State the theorem
theorem function_expression (a b : ℝ) 
  (h1 : a ≠ 0)
  (h2 : f a b 1 = 1)
  (h3 : ∃! x : ℝ, f a b x = 2 * x) :
  ∃ g : ℝ → ℝ, (∀ x, f a b x = g x) ∧ (∀ x, g x = (2 * x) / (x + 1)) :=
sorry

end function_expression_l1649_164901


namespace shirley_trefoil_boxes_l1649_164944

/-- The number of cases of boxes Shirley needs to deliver -/
def num_cases : ℕ := 5

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 2

/-- The total number of boxes Shirley sold -/
def total_boxes : ℕ := num_cases * boxes_per_case

theorem shirley_trefoil_boxes : total_boxes = 10 := by
  sorry

end shirley_trefoil_boxes_l1649_164944


namespace max_value_f_one_l1649_164987

/-- Given a function f(x) = x^2 + abx + a + 2b where f(0) = 4,
    the maximum value of f(1) is 7. -/
theorem max_value_f_one (a b : ℝ) :
  let f := fun x : ℝ => x^2 + a*b*x + a + 2*b
  f 0 = 4 →
  (∀ x : ℝ, f 1 ≤ 7) ∧ (∃ x : ℝ, f 1 = 7) :=
by sorry

end max_value_f_one_l1649_164987


namespace root_product_sum_l1649_164980

theorem root_product_sum (a b c : ℝ) : 
  (3 * a^3 - 3 * a^2 + 11 * a - 8 = 0) →
  (3 * b^3 - 3 * b^2 + 11 * b - 8 = 0) →
  (3 * c^3 - 3 * c^2 + 11 * c - 8 = 0) →
  a * b + a * c + b * c = 11/3 := by
sorry

end root_product_sum_l1649_164980


namespace equation_solution_l1649_164965

theorem equation_solution (y : ℝ) (x : ℝ) :
  (x + 1) / (x - 2) = (y^2 + 4*y + 1) / (y^2 + 4*y - 3) →
  x = -(3*y^2 + 12*y - 1) / 2 := by
sorry

end equation_solution_l1649_164965


namespace binomial_15_choose_3_l1649_164963

theorem binomial_15_choose_3 : Nat.choose 15 3 = 455 := by
  sorry

end binomial_15_choose_3_l1649_164963


namespace three_distinct_values_l1649_164913

-- Define a type for the expression
inductive Expr
  | const : ℕ → Expr
  | power : Expr → Expr → Expr

-- Define a function to evaluate the expression
def eval : Expr → ℕ
  | Expr.const n => n
  | Expr.power a b => (eval a) ^ (eval b)

-- Define the base expression
def baseExpr : Expr := Expr.const 3

-- Define a function to generate all possible parenthesizations
def allParenthesizations : Expr → List Expr
  | e => sorry  -- Implementation omitted

-- Theorem statement
theorem three_distinct_values :
  let allExpr := allParenthesizations (Expr.power (Expr.power (Expr.power baseExpr baseExpr) baseExpr) baseExpr)
  (allExpr.map eval).toFinset.card = 3 := by sorry


end three_distinct_values_l1649_164913


namespace smallest_number_of_rectangles_l1649_164931

/-- The side length of the rectangle along one dimension -/
def rectangle_side1 : ℕ := 3

/-- The side length of the rectangle along the other dimension -/
def rectangle_side2 : ℕ := 4

/-- The area of a single rectangle -/
def rectangle_area : ℕ := rectangle_side1 * rectangle_side2

/-- The side length of the square that can be covered exactly by the rectangles -/
def square_side : ℕ := lcm rectangle_side1 rectangle_side2

/-- The area of the square -/
def square_area : ℕ := square_side ^ 2

/-- The number of rectangles needed to cover the square -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem smallest_number_of_rectangles :
  num_rectangles = 12 ∧
  ∀ n : ℕ, n < num_rectangles →
    ¬(∃ s : ℕ, s ^ 2 = n * rectangle_area ∧ s % rectangle_side1 = 0 ∧ s % rectangle_side2 = 0) :=
by sorry

end smallest_number_of_rectangles_l1649_164931


namespace original_number_proof_l1649_164906

theorem original_number_proof (x : ℝ) : 2 - (1 / x) = 5/2 → x = -2 := by
  sorry

end original_number_proof_l1649_164906


namespace second_reduction_percentage_l1649_164986

/-- Given two successive price reductions, where the first is 25% and the combined effect
    is equivalent to a single 47.5% reduction, proves that the second reduction is 30%. -/
theorem second_reduction_percentage (P : ℝ) (x : ℝ) 
  (h1 : P > 0)  -- Assume positive initial price
  (h2 : 0 ≤ x ∧ x ≤ 1)  -- Second reduction percentage is between 0 and 1
  (h3 : (1 - x) * (P - 0.25 * P) = P - 0.475 * P)  -- Combined reduction equation
  : x = 0.3 := by
  sorry

end second_reduction_percentage_l1649_164986


namespace perpendicular_bisector_of_intersection_points_l1649_164975

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- Perpendicular bisector equation -/
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

/-- Theorem stating that the perpendicular bisector of AB is 3x - y - 9 = 0 -/
theorem perpendicular_bisector_of_intersection_points :
  ∃ (A B : ℝ × ℝ), 
    (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧ 
    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧ 
    (∀ (x y : ℝ), perp_bisector x y ↔ 
      (x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = 
      ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4) :=
sorry

end perpendicular_bisector_of_intersection_points_l1649_164975


namespace min_value_sum_l1649_164959

theorem min_value_sum (a b : ℝ) (h : a^2 + 2*b^2 = 6) : 
  ∃ (m : ℝ), m = -3 ∧ ∀ (x y : ℝ), x^2 + 2*y^2 = 6 → m ≤ x + y :=
by sorry

end min_value_sum_l1649_164959


namespace least_common_period_is_36_l1649_164930

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 6) + f (x - 6) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

/-- The least positive period of a function -/
def IsLeastPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ IsPeriod f p ∧ ∀ q, 0 < q ∧ q < p → ¬IsPeriod f q

theorem least_common_period_is_36 :
  ∃ p, p = 36 ∧ 
    (∀ f : ℝ → ℝ, FunctionalEquation f → IsLeastPeriod f p) ∧
    (∀ q, q > 0 → 
      (∀ f : ℝ → ℝ, FunctionalEquation f → IsPeriod f q) → 
      q ≥ p) :=
sorry

end least_common_period_is_36_l1649_164930


namespace keychainSavings_l1649_164974

/-- Represents the cost and quantity of a pack of key chains -/
structure KeyChainPack where
  quantity : ℕ
  cost : ℚ

/-- Calculates the cost per key chain for a given pack -/
def costPerKeyChain (pack : KeyChainPack) : ℚ :=
  pack.cost / pack.quantity

/-- Calculates the total cost for a given number of key chains using a specific pack -/
def totalCost (pack : KeyChainPack) (totalKeyChains : ℕ) : ℚ :=
  (totalKeyChains / pack.quantity) * pack.cost

theorem keychainSavings :
  let pack1 : KeyChainPack := { quantity := 10, cost := 20 }
  let pack2 : KeyChainPack := { quantity := 4, cost := 12 }
  let totalKeyChains : ℕ := 20
  let savings := totalCost pack2 totalKeyChains - totalCost pack1 totalKeyChains
  savings = 20 := by sorry

end keychainSavings_l1649_164974


namespace calculation_proof_l1649_164939

theorem calculation_proof : (15200 * 3^2) / 12 / (6^3 * 5) = 10.5555555556 := by
  sorry

end calculation_proof_l1649_164939


namespace power_negative_two_a_squared_cubed_l1649_164998

theorem power_negative_two_a_squared_cubed (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end power_negative_two_a_squared_cubed_l1649_164998


namespace sum_of_solutions_quadratic_l1649_164934

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (81 - 27*x - 3*x^2 = 0) → 
  (∃ r s : ℝ, (81 - 27*r - 3*r^2 = 0) ∧ (81 - 27*s - 3*s^2 = 0) ∧ (r + s = -9)) := by
  sorry

end sum_of_solutions_quadratic_l1649_164934


namespace shaded_area_circle_with_inscribed_square_l1649_164983

/-- The area of the shaded region in a circle with radius 2, where the unshaded region forms an inscribed square -/
theorem shaded_area_circle_with_inscribed_square :
  let circle_radius : ℝ := 2
  let circle_area := π * circle_radius^2
  let inscribed_square_side := 2 * circle_radius
  let inscribed_square_area := inscribed_square_side^2
  let unshaded_area := inscribed_square_area / 2
  let shaded_area := circle_area - unshaded_area
  shaded_area = 4 * π - 8 := by
  sorry

end shaded_area_circle_with_inscribed_square_l1649_164983


namespace function_properties_l1649_164945

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / 4 + a / 4 - Real.log x - 3 / 2

theorem function_properties (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (deriv (f a)) x = -2) →
  (a = 5 / 4 ∧
   ∀ x : ℝ, x > 0 → x < 5 → (deriv (f a)) x < 0) ∧
  (∀ x : ℝ, x > 5 → (deriv (f a)) x > 0) :=
by sorry

end function_properties_l1649_164945


namespace square_sum_geq_product_sum_l1649_164927

theorem square_sum_geq_product_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a ∧
  (a^2 + b^2 + c^2 = a*b + b*c + c*a ↔ a = b ∧ b = c) :=
by sorry

end square_sum_geq_product_sum_l1649_164927


namespace sum_of_abs_roots_l1649_164961

/-- Given a polynomial x^3 - 2023x + n with integer roots p, q, and r, 
    prove that the sum of their absolute values is 84 -/
theorem sum_of_abs_roots (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) → 
  |p| + |q| + |r| = 84 := by
  sorry

end sum_of_abs_roots_l1649_164961


namespace gcd_of_powers_of_two_l1649_164956

theorem gcd_of_powers_of_two : Nat.gcd (2^1502 - 1) (2^1513 - 1) = 2^11 - 1 := by
  sorry

end gcd_of_powers_of_two_l1649_164956


namespace always_integer_solution_l1649_164911

theorem always_integer_solution (a : ℕ+) : ∃ x y : ℤ, x^2 - y^2 = (a : ℤ)^3 := by
  sorry

end always_integer_solution_l1649_164911


namespace complement_B_equals_M_intersection_A_B_l1649_164923

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A (a : ℝ) : Set ℝ := {x | x^2 + (a-1)*x - a > 0}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | (x+a)*(x+b) > 0}

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem 1
theorem complement_B_equals_M (a b : ℝ) (h : a ≠ b) :
  (U \ B a b = M) ↔ ((a = -3 ∧ b = 1) ∨ (a = 1 ∧ b = -3)) :=
sorry

-- Theorem 2
theorem intersection_A_B (a b : ℝ) (h : -1 < b ∧ b < a ∧ a < 1) :
  A a ∩ B a b = {x | x < -a ∨ x > 1} :=
sorry

end complement_B_equals_M_intersection_A_B_l1649_164923


namespace children_ages_exist_l1649_164932

theorem children_ages_exist :
  ∃ (a b c d : ℕ), 
    a + b + c + d = 33 ∧
    (a - 3) + (b - 3) + (c - 3) + (d - 3) = 22 ∧
    (a - 7) + (b - 7) + (c - 7) + (d - 7) = 11 ∧
    (a - 13) + (b - 13) + (c - 13) + (d - 13) = 1 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
by sorry

end children_ages_exist_l1649_164932


namespace circle_circumference_l1649_164953

theorem circle_circumference (r : ℝ) (h : r = 4) : 
  2 * Real.pi * r = 8 * Real.pi := by
  sorry

end circle_circumference_l1649_164953


namespace count_special_numbers_l1649_164982

def is_odd (n : Nat) : Bool := n % 2 = 1

def is_even (n : Nat) : Bool := n % 2 = 0

def digits : List Nat := [1, 2, 3, 4, 5]

def is_valid_number (n : List Nat) : Bool :=
  n.length = 5 ∧ 
  n.toFinset.card = 5 ∧
  n.all (λ d => d ∈ digits) ∧
  (∃ i, i ∈ [1, 2, 3] ∧ 
    is_odd (n.get! i) ∧ 
    is_even (n.get! (i-1)) ∧ 
    is_even (n.get! (i+1)))

theorem count_special_numbers :
  (List.filter is_valid_number (List.permutations digits)).length = 36 := by
  sorry

end count_special_numbers_l1649_164982


namespace separation_theorem_l1649_164950

-- Define a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a line in a plane
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define the set of white points and black points
def WhitePoints : Set Point := sorry
def BlackPoints : Set Point := sorry

-- Define a function to check if a point is on one side of a line
def onSide (p : Point) (l : Line) : Bool := sorry

-- Define a function to check if a line separates two sets of points
def separates (l : Line) (s1 s2 : Set Point) : Prop :=
  ∀ p1 ∈ s1, ∀ p2 ∈ s2, onSide p1 l ≠ onSide p2 l

-- State the theorem
theorem separation_theorem :
  (∀ p1 p2 p3 p4 : Point, p1 ∈ WhitePoints ∪ BlackPoints →
    p2 ∈ WhitePoints ∪ BlackPoints → p3 ∈ WhitePoints ∪ BlackPoints →
    p4 ∈ WhitePoints ∪ BlackPoints →
    ∃ l : Line, separates l (WhitePoints ∩ {p1, p2, p3, p4}) (BlackPoints ∩ {p1, p2, p3, p4})) →
  ∃ l : Line, separates l WhitePoints BlackPoints :=
sorry

end separation_theorem_l1649_164950


namespace hyperbola_point_distance_l1649_164928

/-- A point on a hyperbola with a specific distance to a line has a specific distance to another line --/
theorem hyperbola_point_distance (m n : ℝ) : 
  m^2 - n^2 = 9 →                        -- P(m, n) is on the hyperbola x^2 - y^2 = 9
  (|m + n| / Real.sqrt 2) = 2016 →       -- Distance from P to y = -x is 2016
  (|m - n| / Real.sqrt 2) = 448 :=       -- Distance from P to y = x is 448
by sorry

end hyperbola_point_distance_l1649_164928


namespace jenny_jellybeans_l1649_164971

/-- The fraction of jellybeans remaining after eating 25% -/
def remainingFraction : ℝ := 0.75

/-- The number of days that passed -/
def days : ℕ := 3

/-- The number of jellybeans remaining after 3 days -/
def remainingJellybeans : ℕ := 27

/-- The original number of jellybeans in Jenny's jar -/
def originalJellybeans : ℕ := 64

theorem jenny_jellybeans :
  (remainingFraction ^ days) * (originalJellybeans : ℝ) = remainingJellybeans := by
  sorry

end jenny_jellybeans_l1649_164971


namespace no_solution_iff_m_equals_seven_l1649_164978

theorem no_solution_iff_m_equals_seven (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 → x ≠ 8 → (x - 3) / (x - 4) ≠ (x - m) / (x - 8)) ↔ m = 7 :=
by sorry

end no_solution_iff_m_equals_seven_l1649_164978


namespace annie_cookies_total_l1649_164922

/-- The number of cookies Annie ate on Monday -/
def monday_cookies : ℕ := 5

/-- The number of cookies Annie ate on Tuesday -/
def tuesday_cookies : ℕ := 2 * monday_cookies

/-- The number of cookies Annie ate on Wednesday -/
def wednesday_cookies : ℕ := tuesday_cookies + (tuesday_cookies * 40 / 100)

/-- The total number of cookies Annie ate over three days -/
def total_cookies : ℕ := monday_cookies + tuesday_cookies + wednesday_cookies

/-- Theorem stating that Annie ate 29 cookies in total over three days -/
theorem annie_cookies_total : total_cookies = 29 := by
  sorry

end annie_cookies_total_l1649_164922


namespace max_investment_at_lower_rate_l1649_164960

theorem max_investment_at_lower_rate
  (total_investment : ℝ)
  (lower_rate : ℝ)
  (higher_rate : ℝ)
  (min_interest : ℝ)
  (h1 : total_investment = 25000)
  (h2 : lower_rate = 0.07)
  (h3 : higher_rate = 0.12)
  (h4 : min_interest = 2450)
  : ∃ (x : ℝ), x ≤ 11000 ∧
    x + (total_investment - x) = total_investment ∧
    lower_rate * x + higher_rate * (total_investment - x) ≥ min_interest ∧
    ∀ (y : ℝ), y > x →
      lower_rate * y + higher_rate * (total_investment - y) < min_interest :=
by sorry


end max_investment_at_lower_rate_l1649_164960


namespace household_expense_sharing_l1649_164925

theorem household_expense_sharing (X Y : ℝ) (h : X > Y) :
  (X - Y) / 2 = (X + Y) / 2 - Y := by
  sorry

end household_expense_sharing_l1649_164925


namespace stream_speed_l1649_164957

/-- Proves that the speed of a stream is 3 kmph given certain conditions about a boat's travel -/
theorem stream_speed (boat_speed : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : downstream_time = 1)
  (h3 : upstream_time = 1.5) :
  let stream_speed := (boat_speed * (upstream_time - downstream_time)) / (upstream_time + downstream_time)
  stream_speed = 3 := by
  sorry


end stream_speed_l1649_164957


namespace farmer_land_calculation_l1649_164912

theorem farmer_land_calculation (total_land : ℝ) : 
  (0.9 * total_land * 0.2 + 0.9 * total_land * 0.7 + 630 = 0.9 * total_land) →
  total_land = 7000 := by
  sorry

end farmer_land_calculation_l1649_164912


namespace not_necessarily_right_triangle_l1649_164949

theorem not_necessarily_right_triangle (a b c : ℝ) (h1 : a = 3^2) (h2 : b = 4^2) (h3 : c = 5^2) :
  ¬ (a^2 + b^2 = c^2) :=
sorry

end not_necessarily_right_triangle_l1649_164949


namespace base_seven_subtraction_l1649_164908

/-- Represents a number in base 7 as a list of digits (least significant first) -/
def BaseSevenNum := List Nat

/-- Converts a base 7 number to its decimal representation -/
def to_decimal (n : BaseSevenNum) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Subtracts two base 7 numbers -/
def base_seven_sub (a b : BaseSevenNum) : BaseSevenNum :=
  sorry -- Implementation details omitted

theorem base_seven_subtraction :
  let a : BaseSevenNum := [3, 3, 3, 2]  -- 2333 in base 7
  let b : BaseSevenNum := [1, 1, 1, 1]  -- 1111 in base 7
  let result : BaseSevenNum := [2, 2, 2, 1]  -- 1222 in base 7
  base_seven_sub a b = result :=
by sorry

end base_seven_subtraction_l1649_164908


namespace specific_polyhedron_space_diagonals_l1649_164985

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (P : ConvexPolyhedron) : ℕ :=
  (P.vertices.choose 2) - P.edges - (2 * P.quadrilateral_faces)

/-- Theorem stating the number of space diagonals in the specific polyhedron -/
theorem specific_polyhedron_space_diagonals :
  ∃ (P : ConvexPolyhedron),
    P.vertices = 26 ∧
    P.edges = 60 ∧
    P.faces = 36 ∧
    P.triangular_faces = 24 ∧
    P.quadrilateral_faces = 12 ∧
    space_diagonals P = 241 := by
  sorry

end specific_polyhedron_space_diagonals_l1649_164985


namespace return_probability_eight_reflections_l1649_164937

/-- A square with a point at its center -/
structure CenteredSquare where
  /-- The square -/
  square : Set (ℝ × ℝ)
  /-- The center point -/
  center : ℝ × ℝ
  /-- The center point is in the square -/
  center_in_square : center ∈ square

/-- A reflection over a line in a square -/
def reflect (s : CenteredSquare) (line : Set (ℝ × ℝ)) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The sequence of points generated by reflections -/
def reflection_sequence (s : CenteredSquare) (n : ℕ) : ℝ × ℝ := sorry

/-- The probability of returning to the center after n reflections -/
def return_probability (s : CenteredSquare) (n : ℕ) : ℚ := sorry

theorem return_probability_eight_reflections (s : CenteredSquare) :
  return_probability s 8 = 1225 / 16384 := by sorry

end return_probability_eight_reflections_l1649_164937


namespace absolute_value_equals_negation_implies_nonpositive_l1649_164989

theorem absolute_value_equals_negation_implies_nonpositive (a : ℝ) :
  |a| = -a → a ≤ 0 := by
  sorry

end absolute_value_equals_negation_implies_nonpositive_l1649_164989


namespace participation_schemes_with_restriction_l1649_164941

-- Define the number of students and competitions
def num_students : ℕ := 4
def num_competitions : ℕ := 4

-- Define a function to calculate the number of participation schemes
def participation_schemes (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else Nat.factorial n / Nat.factorial (n - k)

-- Theorem statement
theorem participation_schemes_with_restriction :
  participation_schemes (num_students - 1) (num_competitions - 1) *
  (num_students - 1) = 18 :=
sorry

end participation_schemes_with_restriction_l1649_164941


namespace fraction_equality_l1649_164942

theorem fraction_equality (a b : ℝ) (x : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : x = a / b) :
  (a + 2*b) / (a - 2*b) = (x + 2) / (x - 2) := by
sorry

end fraction_equality_l1649_164942


namespace income_expenditure_ratio_l1649_164976

/-- Proves that given an income of 20000 and savings of 5000, the ratio of income to expenditure is 4:3 -/
theorem income_expenditure_ratio (income : ℕ) (savings : ℕ) (expenditure : ℕ) :
  income = 20000 →
  savings = 5000 →
  expenditure = income - savings →
  (income : ℚ) / expenditure = 4 / 3 := by
sorry

end income_expenditure_ratio_l1649_164976


namespace shaded_fraction_of_rectangle_l1649_164977

theorem shaded_fraction_of_rectangle (length width : ℝ) (shaded_quarter : ℝ) :
  length = 15 →
  width = 20 →
  shaded_quarter = (1 / 4) * (length * width) →
  shaded_quarter = (1 / 5) * (length * width) →
  shaded_quarter / (length * width) = 1 / 5 := by
  sorry

end shaded_fraction_of_rectangle_l1649_164977


namespace f_properties_l1649_164948

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1) ∧
  (∀ (α : ℝ), α ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → f α = 6 / 5 → Real.cos (2 * α) = (3 - 4 * Real.sqrt 3) / 10) :=
by sorry


end f_properties_l1649_164948


namespace circle_radius_equals_one_l1649_164938

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

-- Define a right triangle
def RightTriangle (A B C : ℝ × ℝ) : Prop :=
  Triangle A B C ∧ 
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define an equilateral triangle
def EquilateralTriangle (A D E : ℝ × ℝ) : Prop :=
  Triangle A D E ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = (E.1 - A.1)^2 + (E.2 - A.2)^2 ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = (E.1 - D.1)^2 + (E.2 - D.2)^2

-- Define a point on a line segment
def PointOnSegment (P X Y : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
  P.1 = X.1 + t * (Y.1 - X.1) ∧
  P.2 = X.2 + t * (Y.2 - X.2)

-- Main theorem
theorem circle_radius_equals_one 
  (A B C D E : ℝ × ℝ) :
  RightTriangle A B C →
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1 →
  PointOnSegment D B C →
  PointOnSegment E A C →
  EquilateralTriangle A D E →
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = 1 :=
sorry

end circle_radius_equals_one_l1649_164938


namespace fraction_simplification_l1649_164999

theorem fraction_simplification (a : ℝ) (h : a ≠ 0) : (a^2 - 1) / a + 1 / a = a := by
  sorry

end fraction_simplification_l1649_164999


namespace engineer_designer_ratio_l1649_164995

/-- Represents the ratio of engineers to designers in a team -/
structure TeamRatio where
  engineers : ℕ
  designers : ℕ

/-- Proves that the ratio of engineers to designers is 2:1 given the average ages -/
theorem engineer_designer_ratio (team_avg : ℝ) (engineer_avg : ℝ) (designer_avg : ℝ) 
    (h1 : team_avg = 52) (h2 : engineer_avg = 48) (h3 : designer_avg = 60) : 
    ∃ (ratio : TeamRatio), ratio.engineers = 2 ∧ ratio.designers = 1 := by
  sorry

#check engineer_designer_ratio

end engineer_designer_ratio_l1649_164995


namespace sasha_purchase_l1649_164933

/-- The total number of items (pencils and pens) purchased by Sasha -/
def total_items : ℕ := 23

/-- The cost of a single pencil in rubles -/
def pencil_cost : ℕ := 13

/-- The cost of a single pen in rubles -/
def pen_cost : ℕ := 20

/-- The total amount spent in rubles -/
def total_spent : ℕ := 350

/-- Theorem stating that given the costs and total spent, the total number of items purchased is 23 -/
theorem sasha_purchase :
  ∃ (pencils pens : ℕ),
    pencils * pencil_cost + pens * pen_cost = total_spent ∧
    pencils + pens = total_items :=
by sorry

end sasha_purchase_l1649_164933


namespace non_integer_factors_integer_products_l1649_164996

theorem non_integer_factors_integer_products :
  ∃ (a b c : ℝ),
    (¬ ∃ (n : ℤ), a = n) ∧
    (¬ ∃ (n : ℤ), b = n) ∧
    (¬ ∃ (n : ℤ), c = n) ∧
    (∃ (m : ℤ), a * b = m) ∧
    (∃ (m : ℤ), b * c = m) ∧
    (∃ (m : ℤ), c * a = m) ∧
    (∃ (m : ℤ), a * b * c = m) :=
by sorry

end non_integer_factors_integer_products_l1649_164996


namespace advertisement_revenue_l1649_164905

/-- Calculates the revenue from advertisements for a college football program -/
theorem advertisement_revenue
  (production_cost : ℚ)
  (num_programs : ℕ)
  (selling_price : ℚ)
  (desired_profit : ℚ)
  (h1 : production_cost = 70/100)
  (h2 : num_programs = 35000)
  (h3 : selling_price = 50/100)
  (h4 : desired_profit = 8000) :
  production_cost * num_programs + desired_profit - selling_price * num_programs = 15000 :=
by sorry

end advertisement_revenue_l1649_164905


namespace planes_parallel_if_perpendicular_to_parallel_lines_l1649_164962

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_parallel_lines 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  α ≠ β →
  parallel m n →
  perpendicular m α →
  perpendicular n β →
  planeParallel α β :=
sorry

end planes_parallel_if_perpendicular_to_parallel_lines_l1649_164962


namespace eliana_fuel_cost_l1649_164984

/-- The amount Eliana spent on fuel in a week -/
def fuel_cost (refill_cost : ℕ) (refill_count : ℕ) : ℕ :=
  refill_cost * refill_count

/-- Proof that Eliana spent $63 on fuel this week -/
theorem eliana_fuel_cost :
  fuel_cost 21 3 = 63 := by
  sorry

end eliana_fuel_cost_l1649_164984


namespace probability_at_least_one_red_l1649_164943

-- Define the contents of each box
def box_contents : ℕ := 3

-- Define the number of red balls in each box
def red_balls : ℕ := 2

-- Define the number of white balls in each box
def white_balls : ℕ := 1

-- Define the total number of possible outcomes
def total_outcomes : ℕ := box_contents * box_contents

-- Define the number of outcomes with no red balls
def no_red_outcomes : ℕ := white_balls * white_balls

-- State the theorem
theorem probability_at_least_one_red :
  (1 : ℚ) - (no_red_outcomes : ℚ) / (total_outcomes : ℚ) = 8 / 9 :=
sorry

end probability_at_least_one_red_l1649_164943


namespace arithmetic_simplifications_l1649_164973

theorem arithmetic_simplifications :
  (∀ (a b c : Rat), a / 16 - b / 16 + c / 16 = (a - b + c) / 16) ∧
  (∀ (d e f : Rat), d / 12 - e / 12 + f / 12 = (d - e + f) / 12) ∧
  (∀ (g h i j k l m : Nat), g + h + i + j + k + l + m = 736) ∧
  (∀ (n p q r : Rat), n - p / 9 - q / 9 + (1 + r / 99) = 2 + r / 99) →
  5 / 16 - 3 / 16 + 7 / 16 = 9 / 16 ∧
  3 / 12 - 4 / 12 + 6 / 12 = 5 / 12 ∧
  64 + 27 + 81 + 36 + 173 + 219 + 136 = 736 ∧
  2 - 8 / 9 - 1 / 9 + (1 + 98 / 99) = 2 + 98 / 99 :=
by sorry

end arithmetic_simplifications_l1649_164973


namespace dividend_calculation_l1649_164994

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 20)
  (h2 : quotient = 8)
  (h3 : remainder = 6) :
  divisor * quotient + remainder = 166 := by
sorry

end dividend_calculation_l1649_164994


namespace fitness_center_membership_ratio_l1649_164954

theorem fitness_center_membership_ratio 
  (f m : ℕ) -- f: number of female members, m: number of male members
  (hf : f > 0) -- assume there's at least one female member
  (hm : m > 0) -- assume there's at least one male member
  (h_avg_female : (50 * f) / (f + m) = 50) -- average age of female members is 50
  (h_avg_male : (30 * m) / (f + m) = 30)   -- average age of male members is 30
  (h_avg_total : (50 * f + 30 * m) / (f + m) = 35) -- average age of all members is 35
  : f / m = 1 / 3 :=
by sorry

end fitness_center_membership_ratio_l1649_164954


namespace noodles_left_proof_l1649_164902

-- Define the initial number of noodles
def initial_noodles : Float := 54.0

-- Define the number of noodles given away
def noodles_given : Float := 12.0

-- Theorem to prove
theorem noodles_left_proof : initial_noodles - noodles_given = 42.0 := by
  sorry

end noodles_left_proof_l1649_164902


namespace not_always_equal_l1649_164966

theorem not_always_equal : ∃ (a b : ℝ), 3 * (a + b) ≠ 3 * a + b := by
  sorry

end not_always_equal_l1649_164966


namespace inequality_solution_l1649_164972

theorem inequality_solution (x : ℝ) : 
  (9*x^2 + 27*x - 64) / ((3*x - 5)*(x + 3)) < 2 ↔ 
  x < -3 ∨ (-17/3 < x ∧ x < 5/3) ∨ 2 < x := by sorry

end inequality_solution_l1649_164972


namespace least_multiple_24_above_450_l1649_164997

theorem least_multiple_24_above_450 : 
  ∀ n : ℕ, n > 0 ∧ 24 ∣ n ∧ n > 450 → n ≥ 456 := by sorry

end least_multiple_24_above_450_l1649_164997


namespace square_ends_in_001_l1649_164970

theorem square_ends_in_001 (x : ℤ) : 
  x^2 ≡ 1 [ZMOD 1000] → 
  (x ≡ 1 [ZMOD 500] ∨ x ≡ -1 [ZMOD 500] ∨ x ≡ 249 [ZMOD 500] ∨ x ≡ -249 [ZMOD 500]) :=
by sorry

end square_ends_in_001_l1649_164970


namespace nantucket_meeting_l1649_164958

/-- The number of females attending the meeting in Nantucket --/
def females_attending : ℕ := sorry

/-- The total population of Nantucket --/
def total_population : ℕ := 300

/-- The total number of people attending the meeting --/
def total_attending : ℕ := total_population / 2

/-- The number of males attending the meeting --/
def males_attending : ℕ := 2 * females_attending

theorem nantucket_meeting :
  females_attending = 50 ∧
  females_attending + males_attending = total_attending :=
by sorry

end nantucket_meeting_l1649_164958


namespace car_not_speeding_l1649_164967

/-- Braking distance function -/
def braking_distance (x : ℝ) : ℝ := 0.01 * x + 0.002 * x^2

/-- Speed limit in km/h -/
def speed_limit : ℝ := 120

/-- Measured braking distance in meters -/
def measured_distance : ℝ := 26.5

/-- Theorem: There exists a speed less than the speed limit that results in the measured braking distance -/
theorem car_not_speeding : ∃ x : ℝ, x < speed_limit ∧ braking_distance x = measured_distance := by
  sorry


end car_not_speeding_l1649_164967


namespace smallest_prime_divisor_of_sum_l1649_164909

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  n = 3^19 + 6^21 → (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → p ≤ q) ∧
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → p ≤ q) → p = 3 :=
by sorry

end smallest_prime_divisor_of_sum_l1649_164909


namespace opposite_of_negative_five_l1649_164916

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_negative_five : opposite (-5) = 5 := by
  sorry

end opposite_of_negative_five_l1649_164916


namespace polynomial_division_remainder_l1649_164918

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  2 * X^2 - 21 * X + 55 = (X + 3) * q + 136 := by
  sorry

end polynomial_division_remainder_l1649_164918


namespace f_range_l1649_164993

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 4 * Real.sin x ^ 2 + 2 * Real.sin x + 3 * Real.cos x ^ 2 - 6) / (Real.sin x - 1)

theorem f_range : 
  ∀ (y : ℝ), (∃ (x : ℝ), Real.sin x ≠ 1 ∧ f x = y) ↔ (0 ≤ y ∧ y < 8) := by
  sorry

end f_range_l1649_164993


namespace cube_surface_coverage_l1649_164935

/-- Represents a cube -/
structure Cube where
  vertices : ℕ
  angle_sum_at_vertex : ℕ

/-- Represents a triangle -/
structure Triangle where
  angle_sum : ℕ

/-- The problem statement -/
theorem cube_surface_coverage (c : Cube) (t : Triangle) : 
  c.vertices = 8 → 
  c.angle_sum_at_vertex = 270 → 
  t.angle_sum = 180 → 
  ¬ (3 * t.angle_sum ≥ c.vertices * 90) :=
by sorry

end cube_surface_coverage_l1649_164935


namespace crew_members_count_l1649_164900

/-- Represents the number of passengers at each stage of the flight --/
structure PassengerCount where
  initial : ℕ
  after_texas : ℕ
  after_north_carolina : ℕ

/-- Calculates the final number of passengers --/
def final_passengers (p : PassengerCount) : ℕ :=
  p.initial - 58 + 24 - 47 + 14

/-- Represents the flight data --/
structure FlightData where
  passenger_count : PassengerCount
  total_landed : ℕ

/-- Calculates the number of crew members --/
def crew_members (f : FlightData) : ℕ :=
  f.total_landed - final_passengers f.passenger_count

/-- Theorem stating the number of crew members --/
theorem crew_members_count (f : FlightData) 
  (h1 : f.passenger_count.initial = 124)
  (h2 : f.total_landed = 67) : 
  crew_members f = 10 := by
  sorry

#check crew_members_count

end crew_members_count_l1649_164900


namespace number_of_factors_of_n_l1649_164940

def n : ℕ := 2^5 * 3^4 * 5^6 * 6^3

theorem number_of_factors_of_n : (Finset.card (Nat.divisors n)) = 504 := by
  sorry

end number_of_factors_of_n_l1649_164940


namespace age_difference_theorem_l1649_164914

theorem age_difference_theorem (n : ℕ) 
  (ages : Fin n → ℕ) 
  (h1 : n = 5) 
  (h2 : ∃ (i j : Fin n), ages i = ages j + 1)
  (h3 : ∃ (i j : Fin n), ages i = ages j + 2)
  (h4 : ∃ (i j : Fin n), ages i = ages j + 3)
  (h5 : ∃ (i j : Fin n), ages i = ages j + 4) :
  ∃ (i j : Fin n), ages i = ages j + 10 :=
sorry

end age_difference_theorem_l1649_164914


namespace harriet_speed_back_l1649_164910

/-- Harriet's round trip between A-ville and B-town -/
def harriet_trip (speed_to_b : ℝ) (total_time : ℝ) (time_to_b_minutes : ℝ) : Prop :=
  let time_to_b : ℝ := time_to_b_minutes / 60
  let distance : ℝ := speed_to_b * time_to_b
  let time_from_b : ℝ := total_time - time_to_b
  let speed_from_b : ℝ := distance / time_from_b
  speed_from_b = 140

theorem harriet_speed_back :
  harriet_trip 110 5 168 := by sorry

end harriet_speed_back_l1649_164910


namespace pizza_both_toppings_l1649_164992

/-- Represents a pizza with cheese and olive toppings -/
structure Pizza where
  total_slices : ℕ
  cheese_slices : ℕ
  olive_slices : ℕ
  both_toppings : ℕ

/-- Theorem: Given the conditions, prove that 7 slices have both cheese and olives -/
theorem pizza_both_toppings (p : Pizza) 
  (h1 : p.total_slices = 24)
  (h2 : p.cheese_slices = 15)
  (h3 : p.olive_slices = 16)
  (h4 : p.total_slices = p.both_toppings + (p.cheese_slices - p.both_toppings) + (p.olive_slices - p.both_toppings)) :
  p.both_toppings = 7 := by
  sorry

#check pizza_both_toppings

end pizza_both_toppings_l1649_164992


namespace complex_power_difference_l1649_164920

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^24 - (1 - i)^24 = 0 := by
  sorry

end complex_power_difference_l1649_164920


namespace makeup_set_cost_l1649_164946

theorem makeup_set_cost (initial_money mom_contribution additional_needed : ℕ) :
  initial_money = 35 →
  mom_contribution = 20 →
  additional_needed = 10 →
  initial_money + mom_contribution + additional_needed = 65 := by
  sorry

end makeup_set_cost_l1649_164946


namespace geometric_sequence_150th_term_l1649_164917

def geometricSequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem geometric_sequence_150th_term :
  let a₁ := 8
  let a₂ := -16
  let r := a₂ / a₁
  geometricSequence a₁ r 150 = 8 * (-2)^149 := by
  sorry

end geometric_sequence_150th_term_l1649_164917


namespace contestant_A_score_l1649_164919

def speech_contest_score (content_score : ℕ) (skills_score : ℕ) (effects_score : ℕ) : ℚ :=
  (4 * content_score + 2 * skills_score + 4 * effects_score) / 10

theorem contestant_A_score :
  speech_contest_score 90 80 90 = 88 := by
  sorry

end contestant_A_score_l1649_164919
