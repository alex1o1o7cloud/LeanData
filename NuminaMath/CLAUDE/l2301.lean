import Mathlib

namespace downstream_distance_l2301_230103

/-- Calculates the distance traveled downstream given boat speed, stream rate, and time. -/
theorem downstream_distance
  (boat_speed : ℝ)
  (stream_rate : ℝ)
  (time : ℝ)
  (h1 : boat_speed = 16)
  (h2 : stream_rate = 5)
  (h3 : time = 6) :
  boat_speed + stream_rate * time = 126 :=
by sorry

end downstream_distance_l2301_230103


namespace d_neither_sufficient_nor_necessary_for_a_l2301_230146

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : (A → B) ∧ ¬(B → A))  -- A is sufficient but not necessary for B
variable (h2 : (B ↔ C))             -- B is necessary and sufficient for C
variable (h3 : (D → C) ∧ ¬(C → D))  -- C is necessary but not sufficient for D

-- Theorem to prove
theorem d_neither_sufficient_nor_necessary_for_a :
  ¬((D → A) ∧ (A → D)) :=
sorry

end d_neither_sufficient_nor_necessary_for_a_l2301_230146


namespace sum_of_possible_values_l2301_230180

theorem sum_of_possible_values (x y : ℝ) 
  (h : 2 * x * y - 2 * x / (y^2) - 2 * y / (x^2) = 4) : 
  ∃ (v₁ v₂ : ℝ), (x - 2) * (y - 2) = v₁ ∨ (x - 2) * (y - 2) = v₂ ∧ v₁ + v₂ = 10 := by
sorry

end sum_of_possible_values_l2301_230180


namespace wario_field_goals_l2301_230153

/-- Given the conditions of Wario's field goal attempts, prove the number of wide right misses. -/
theorem wario_field_goals (total_attempts : ℕ) (miss_ratio : ℚ) (wide_right_ratio : ℚ) 
  (h1 : total_attempts = 60)
  (h2 : miss_ratio = 1 / 4)
  (h3 : wide_right_ratio = 1 / 5) : 
  ⌊(total_attempts : ℚ) * miss_ratio * wide_right_ratio⌋ = 3 := by
  sorry

#check wario_field_goals

end wario_field_goals_l2301_230153


namespace sqrt_a_squared_plus_a_equals_two_thirds_l2301_230164

theorem sqrt_a_squared_plus_a_equals_two_thirds (a : ℝ) :
  a > 0 ∧ Real.sqrt (a^2 + a) = 2/3 ↔ a = 1/3 := by sorry

end sqrt_a_squared_plus_a_equals_two_thirds_l2301_230164


namespace ellipse_product_l2301_230134

-- Define the ellipse C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 = 1}

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (2, 0)
def F₂ : ℝ × ℝ := (-2, 0)

-- State the theorem
theorem ellipse_product (P : ℝ × ℝ) 
  (h_on_ellipse : P ∈ C) 
  (h_perpendicular : (P.1 - F₁.1, P.2 - F₁.2) • (P.1 - F₂.1, P.2 - F₂.2) = 0) :
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 := by
  sorry

end ellipse_product_l2301_230134


namespace salary_problem_l2301_230195

/-- Proves that given the conditions of the problem, A's salary is Rs. 3000 --/
theorem salary_problem (total : ℝ) (a_salary : ℝ) (b_salary : ℝ) 
  (h1 : total = 4000)
  (h2 : a_salary + b_salary = total)
  (h3 : 0.05 * a_salary = 0.15 * b_salary) :
  a_salary = 3000 := by
  sorry

#check salary_problem

end salary_problem_l2301_230195


namespace equation_solution_l2301_230135

theorem equation_solution (x : ℝ) : 9 / (1 + 4 / x) = 1 → x = 1/2 := by
  sorry

end equation_solution_l2301_230135


namespace nine_people_four_consecutive_l2301_230138

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def total_arrangements (n : ℕ) : ℕ := factorial n

def consecutive_arrangements (n : ℕ) (k : ℕ) : ℕ := 
  factorial (n - k + 1) * factorial k

def valid_arrangements (n : ℕ) (k : ℕ) : ℕ := 
  total_arrangements n - consecutive_arrangements n k

theorem nine_people_four_consecutive (n : ℕ) (k : ℕ) :
  n = 9 ∧ k = 4 → valid_arrangements n k = 345600 := by
  sorry

end nine_people_four_consecutive_l2301_230138


namespace second_day_visitors_count_l2301_230197

/-- Represents the food bank scenario --/
structure FoodBank where
  initial_stock : ℕ
  first_day_visitors : ℕ
  first_day_cans_per_person : ℕ
  first_restock : ℕ
  second_day_cans_per_person : ℕ
  second_restock : ℕ
  second_day_cans_given : ℕ

/-- Calculates the number of people who showed up on the second day --/
def second_day_visitors (fb : FoodBank) : ℕ :=
  fb.second_day_cans_given / fb.second_day_cans_per_person

/-- Theorem stating that given the conditions, 1250 people showed up on the second day --/
theorem second_day_visitors_count (fb : FoodBank) 
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.first_day_visitors = 500)
  (h3 : fb.first_day_cans_per_person = 1)
  (h4 : fb.first_restock = 1500)
  (h5 : fb.second_day_cans_per_person = 2)
  (h6 : fb.second_restock = 3000)
  (h7 : fb.second_day_cans_given = 2500) :
  second_day_visitors fb = 1250 := by
  sorry

#eval second_day_visitors {
  initial_stock := 2000,
  first_day_visitors := 500,
  first_day_cans_per_person := 1,
  first_restock := 1500,
  second_day_cans_per_person := 2,
  second_restock := 3000,
  second_day_cans_given := 2500
}

end second_day_visitors_count_l2301_230197


namespace quadratic_sum_l2301_230158

/-- Given a quadratic polynomial 20x^2 + 160x + 800, when expressed in the form a(x+b)^2 + c,
    the sum a + b + c equals 504. -/
theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (20 * x^2 + 160 * x + 800 = a * (x + b)^2 + c) ∧
  (a + b + c = 504) := by
  sorry

end quadratic_sum_l2301_230158


namespace coal_shoveling_time_l2301_230106

/-- Given that 10 people can shovel 10,000 pounds of coal in 10 days,
    prove that 5 people will take 80 days to shovel 40,000 pounds of coal. -/
theorem coal_shoveling_time 
  (people : ℕ) 
  (days : ℕ) 
  (coal : ℕ) 
  (h1 : people = 10) 
  (h2 : days = 10) 
  (h3 : coal = 10000) :
  (people / 2) * (coal * 4 / (people * days)) = 80 := by
  sorry

#check coal_shoveling_time

end coal_shoveling_time_l2301_230106


namespace stream_rate_proof_l2301_230162

/-- The speed of the man rowing in still water -/
def still_water_speed : ℝ := 24

/-- The rate of the stream -/
def stream_rate : ℝ := 12

/-- The ratio of time taken to row upstream vs downstream -/
def time_ratio : ℝ := 3

theorem stream_rate_proof :
  (1 / (still_water_speed - stream_rate) = time_ratio * (1 / (still_water_speed + stream_rate))) →
  stream_rate = 12 := by
  sorry

end stream_rate_proof_l2301_230162


namespace imaginary_number_condition_l2301_230117

theorem imaginary_number_condition (x : ℝ) : 
  let z : ℂ := Complex.mk (x^2 - 1) (x - 1)
  (z.re = 0 ∧ z.im ≠ 0) → x = -1 := by
sorry

end imaginary_number_condition_l2301_230117


namespace product_and_sum_of_squares_l2301_230139

theorem product_and_sum_of_squares (x y : ℝ) : 
  x * y = 120 → x^2 + y^2 = 289 → x + y = 22 := by
sorry

end product_and_sum_of_squares_l2301_230139


namespace delta_five_three_l2301_230175

def delta (a b : ℤ) : ℤ := 4 * a - 6 * b

theorem delta_five_three : delta 5 3 = 2 := by sorry

end delta_five_three_l2301_230175


namespace empty_board_prob_2013_l2301_230137

/-- Represents the state of the blackboard -/
inductive BoardState
| Empty : BoardState
| NonEmpty : Nat → BoardState

/-- The rules for updating the blackboard based on a coin flip -/
def updateBoard (state : BoardState) (n : Nat) (isHeads : Bool) : BoardState :=
  match state, isHeads with
  | BoardState.Empty, true => BoardState.NonEmpty n
  | BoardState.NonEmpty m, true => 
      if (m^2 + 2*n^2) % 3 = 0 then BoardState.Empty else BoardState.NonEmpty n
  | _, false => state

/-- The probability of an empty blackboard after n flips -/
def emptyBoardProb (n : Nat) : ℚ :=
  sorry  -- Definition omitted for brevity

theorem empty_board_prob_2013 :
  ∃ (u v : ℕ), emptyBoardProb 2013 = (2 * u + 1) / (2^1336 * (2 * v + 1)) :=
sorry

#check empty_board_prob_2013

end empty_board_prob_2013_l2301_230137


namespace min_value_of_sum_of_fractions_l2301_230190

theorem min_value_of_sum_of_fractions (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) (hy : -1 < y ∧ y < 0) (hz : -1 < z ∧ z < 0) :
  ∃ (m : ℝ), m = 2 ∧ ∀ (w : ℝ), w = 1/((1-x)*(1-y)*(1-z)) + 1/((1+x)*(1+y)*(1+z)) → m ≤ w :=
by sorry

end min_value_of_sum_of_fractions_l2301_230190


namespace cistern_water_depth_l2301_230167

/-- Proves that for a cistern with given dimensions and wet surface area, the water depth is 1.25 m -/
theorem cistern_water_depth
  (length : ℝ)
  (width : ℝ)
  (total_wet_area : ℝ)
  (h_length : length = 12)
  (h_width : width = 14)
  (h_total_wet_area : total_wet_area = 233) :
  let bottom_area := length * width
  let perimeter := 2 * (length + width)
  let water_depth := (total_wet_area - bottom_area) / perimeter
  water_depth = 1.25 := by
sorry

end cistern_water_depth_l2301_230167


namespace apple_pie_count_l2301_230179

/-- Represents the number of pies of each type --/
structure PieOrder where
  peach : ℕ
  apple : ℕ
  blueberry : ℕ

/-- Represents the cost of fruit per pound for each type of pie --/
structure FruitCosts where
  peach : ℚ
  apple : ℚ
  blueberry : ℚ

/-- Calculates the total cost of fruit for a given pie order --/
def totalCost (order : PieOrder) (costs : FruitCosts) (poundsPerPie : ℕ) : ℚ :=
  (order.peach * costs.peach + order.apple * costs.apple + order.blueberry * costs.blueberry) * poundsPerPie

theorem apple_pie_count (order : PieOrder) (costs : FruitCosts) (poundsPerPie totalSpent : ℕ) :
  order.peach = 5 →
  order.blueberry = 3 →
  poundsPerPie = 3 →
  costs.peach = 2 →
  costs.apple = 1 →
  costs.blueberry = 1 →
  totalCost order costs poundsPerPie = totalSpent →
  order.apple = 4 := by
  sorry

end apple_pie_count_l2301_230179


namespace max_leftover_grapes_l2301_230124

theorem max_leftover_grapes (n : ℕ) : ∃ k : ℕ, n = 7 * k + (n % 7) ∧ n % 7 ≤ 6 :=
sorry

end max_leftover_grapes_l2301_230124


namespace sqrt_sum_equals_eight_l2301_230161

theorem sqrt_sum_equals_eight :
  Real.sqrt (18 - 8 * Real.sqrt 2) + Real.sqrt (18 + 8 * Real.sqrt 2) = 8 := by
  sorry

end sqrt_sum_equals_eight_l2301_230161


namespace bananas_left_in_jar_l2301_230136

theorem bananas_left_in_jar (original : ℕ) (removed : ℕ) (h1 : original = 46) (h2 : removed = 5) :
  original - removed = 41 := by
  sorry

end bananas_left_in_jar_l2301_230136


namespace present_age_of_A_l2301_230141

/-- Given two people A and B, their ages, and future age ratios, 
    prove that A's present age is 15 years. -/
theorem present_age_of_A (a b : ℕ) : 
  a * 3 = b * 5 →  -- Present age ratio
  (a + 6) * 5 = (b + 6) * 7 →  -- Future age ratio
  a = 15 := by
sorry

end present_age_of_A_l2301_230141


namespace downstream_speed_calculation_l2301_230148

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- 
Given a man's upstream speed and still water speed, 
calculates and proves his downstream speed
-/
theorem downstream_speed_calculation (speed : RowingSpeed) 
  (h1 : speed.upstream = 30)
  (h2 : speed.stillWater = 45) :
  speed.downstream = 60 := by
  sorry

end downstream_speed_calculation_l2301_230148


namespace cake_piece_volume_and_icing_area_sum_l2301_230171

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the volume of a triangular prism -/
def triangularPrismVolume (base : ℝ) (height : ℝ) : ℝ := sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (width : ℝ) (height : ℝ) : ℝ := sorry

/-- Main theorem: The sum of the volume and icing area of the cake piece is 19.8 -/
theorem cake_piece_volume_and_icing_area_sum :
  let a : ℝ := 3  -- edge length of the cube
  let p : Point3D := ⟨0, 0, 0⟩
  let q : Point3D := ⟨a, 0, 0⟩
  let r : Point3D := ⟨0, a, 0⟩
  let m : Point3D := ⟨a/3, a, 0⟩
  let triangleQMR_area : ℝ := triangleArea q m r
  let volume : ℝ := triangularPrismVolume triangleQMR_area a
  let icingArea : ℝ := triangleQMR_area + rectangleArea a a
  volume + icingArea = 19.8 := by sorry

end cake_piece_volume_and_icing_area_sum_l2301_230171


namespace min_swaps_100_l2301_230133

/-- The type representing a permutation of the first 100 natural numbers. -/
def Perm100 := Fin 100 → Fin 100

/-- The identity permutation. -/
def id_perm : Perm100 := fun i => i

/-- The target permutation we want to achieve. -/
def target_perm : Perm100 := fun i =>
  if i = 99 then 0 else i + 1

/-- A swap operation on a permutation. -/
def swap (p : Perm100) (i j : Fin 100) : Perm100 := fun k =>
  if k = i then p j
  else if k = j then p i
  else p k

/-- The number of swaps needed to transform one permutation into another. -/
def num_swaps (p q : Perm100) : ℕ := sorry

theorem min_swaps_100 :
  num_swaps id_perm target_perm = 99 := by sorry

end min_swaps_100_l2301_230133


namespace f_symmetry_l2301_230156

/-- Given a function f(x) = x^5 + ax^3 + bx, if f(-2) = 10, then f(2) = -10 -/
theorem f_symmetry (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^5 + a*x^3 + b*x
  f (-2) = 10 → f 2 = -10 := by sorry

end f_symmetry_l2301_230156


namespace smallest_with_sum_2011_has_224_digits_l2301_230181

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The number of digits in a natural number -/
def numberOfDigits (n : ℕ) : ℕ := sorry

/-- The smallest natural number with a given sum of digits -/
def smallestWithSumOfDigits (s : ℕ) : ℕ := sorry

theorem smallest_with_sum_2011_has_224_digits :
  numberOfDigits (smallestWithSumOfDigits 2011) = 224 := by sorry

end smallest_with_sum_2011_has_224_digits_l2301_230181


namespace discriminant_positive_roots_when_k_zero_l2301_230140

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x - 1

-- Define the discriminant of the quadratic equation f(x) = 0
def discriminant (k : ℝ) : ℝ := (2*k)^2 - 4*1*(-1)

-- Theorem 1: The discriminant is always positive
theorem discriminant_positive (k : ℝ) : discriminant k > 0 := by
  sorry

-- Theorem 2: When k = 0, the roots are 1 and -1
theorem roots_when_k_zero :
  ∃ (x1 x2 : ℝ), x1 = 1 ∧ x2 = -1 ∧ f 0 x1 = 0 ∧ f 0 x2 = 0 := by
  sorry

end discriminant_positive_roots_when_k_zero_l2301_230140


namespace cut_triangles_perimeter_sum_l2301_230109

theorem cut_triangles_perimeter_sum (large_perimeter hexagon_perimeter : ℝ) :
  large_perimeter = 60 →
  hexagon_perimeter = 40 →
  ∃ (x y z : ℝ),
    x + y + z = large_perimeter / 3 - hexagon_perimeter / 3 ∧
    3 * (x + y + z) = 60 :=
by
  sorry

end cut_triangles_perimeter_sum_l2301_230109


namespace cubic_polynomial_property_l2301_230102

theorem cubic_polynomial_property (x : ℂ) (h : x^3 + x^2 + x + 1 = 0) :
  x^4 + 2*x^3 + 2*x^2 + 2*x + 1 = -2 := by
  sorry

end cubic_polynomial_property_l2301_230102


namespace oil_price_reduction_oil_price_reduction_result_l2301_230176

/-- Calculates the percentage reduction in oil price given the conditions -/
theorem oil_price_reduction (additional_oil : ℝ) (total_cost : ℝ) (reduced_price : ℝ) : ℝ :=
  let original_amount := (total_cost / reduced_price) - additional_oil
  let original_price := total_cost / original_amount
  let price_difference := original_price - reduced_price
  (price_difference / original_price) * 100

/-- The percentage reduction in oil price is approximately 24.99% -/
theorem oil_price_reduction_result : 
  ∃ ε > 0, |oil_price_reduction 5 500 25 - 24.99| < ε :=
sorry

end oil_price_reduction_oil_price_reduction_result_l2301_230176


namespace part_one_part_two_l2301_230192

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one :
  {x : ℝ | f 1 x ≥ 4 - |x - 1|} = {x : ℝ | x ≤ -1 ∨ x ≥ 3} :=
sorry

-- Part 2
theorem part_two (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  ({x : ℝ | f ((1/m) + 1/(2*n)) x ≤ 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2}) →
  (∀ k l : ℝ, k > 0 → l > 0 → k * l ≥ m * n) →
  m * n = 2 :=
sorry

end part_one_part_two_l2301_230192


namespace sum_of_squared_coefficients_l2301_230149

def original_expression (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 8 * (x^3 - 2*x^2 + 4*x - 1)

theorem sum_of_squared_coefficients :
  ∃ a b c d : ℝ, 
    (∀ x : ℝ, original_expression x = a * x^3 + b * x^2 + c * x + d) ∧
    a^2 + b^2 + c^2 + d^2 = 2395 := by
sorry

end sum_of_squared_coefficients_l2301_230149


namespace no_consecutive_product_l2301_230165

theorem no_consecutive_product (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 7*n + 8 = k * (k + 1) := by
  sorry

end no_consecutive_product_l2301_230165


namespace x_varies_as_z_power_l2301_230128

-- Define the relationships between x, y, and z
def varies_as (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ t, f t = k * g t

-- State the theorem
theorem x_varies_as_z_power (x y z : ℝ → ℝ) :
  varies_as x (λ t => (y t)^4) →
  varies_as y (λ t => (z t)^(1/3)) →
  varies_as x (λ t => (z t)^(4/3)) :=
sorry

end x_varies_as_z_power_l2301_230128


namespace pigeon_problem_l2301_230188

/-- The number of pigeons in a group with the following properties:
  1. When each pigeonhole houses 6 pigeons, 3 pigeons are left without a pigeonhole.
  2. When 5 more pigeons arrive, each pigeonhole fits exactly 8 pigeons. -/
def original_pigeons : ℕ := 27

/-- The number of pigeonholes available. -/
def pigeonholes : ℕ := 3

theorem pigeon_problem :
  (6 * pigeonholes + 3 = original_pigeons) ∧
  (8 * pigeonholes = original_pigeons + 5) := by
  sorry

end pigeon_problem_l2301_230188


namespace circle_inequality_l2301_230159

theorem circle_inequality (a b c d : ℝ) (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a * b + c * d = 1)
  (h1 : x1^2 + y1^2 = 1) (h2 : x2^2 + y2^2 = 1) 
  (h3 : x3^2 + y3^2 = 1) (h4 : x4^2 + y4^2 = 1) :
  (a*y1 + b*y2 + c*y3 + d*y4)^2 + (a*x4 + b*x3 + c*x2 + d*x1)^2 
  ≤ 2 * ((a^2 + b^2)/(a*b) + (c^2 + d^2)/(c*d)) := by
  sorry

end circle_inequality_l2301_230159


namespace ace_then_king_probability_l2301_230185

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of Aces in a standard deck -/
def numAces : ℕ := 4

/-- The number of Kings in a standard deck -/
def numKings : ℕ := 4

/-- The probability of drawing an Ace followed by a King from a standard deck -/
theorem ace_then_king_probability :
  (numAces / deckSize) * (numKings / (deckSize - 1)) = 4 / 663 := by
  sorry


end ace_then_king_probability_l2301_230185


namespace percentage_to_pass_l2301_230198

/-- Given a test with maximum marks, a student's score, and the amount by which they failed,
    calculate the percentage of marks needed to pass the test. -/
theorem percentage_to_pass (max_marks student_score fail_by : ℕ) :
  max_marks = 300 →
  student_score = 80 →
  fail_by = 100 →
  (((student_score + fail_by : ℚ) / max_marks) * 100 : ℚ) = 60 := by
  sorry

end percentage_to_pass_l2301_230198


namespace shortest_path_length_on_cube_l2301_230118

/-- Represents a cube with a given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents a path on the surface of a cube -/
structure SurfacePath (c : Cube) where
  length : ℝ
  isOnSurface : Bool

/-- The shortest path on the surface of a cube from the center of one face to the center of the opposite face -/
def shortestPath (c : Cube) : SurfacePath c :=
  sorry

/-- Theorem stating that the shortest path on a cube with edge length 2 has length 3 -/
theorem shortest_path_length_on_cube :
  let c : Cube := { edgeLength := 2 }
  (shortestPath c).length = 3 := by
  sorry

end shortest_path_length_on_cube_l2301_230118


namespace rectangle_area_l2301_230119

/-- Given a rectangle with length four times its width and perimeter 200 cm, 
    its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h₁ : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 → l * w = 1600 := by
sorry

end rectangle_area_l2301_230119


namespace spider_total_distance_l2301_230120

def spider_crawl (start end_1 end_2 end_3 : Int) : Int :=
  |end_1 - start| + |end_2 - end_1| + |end_3 - end_2|

theorem spider_total_distance :
  spider_crawl (-3) (-8) 0 7 = 20 := by
  sorry

end spider_total_distance_l2301_230120


namespace sum_of_factors_72_l2301_230130

theorem sum_of_factors_72 : (Finset.filter (· ∣ 72) (Finset.range 73)).sum id = 195 := by
  sorry

end sum_of_factors_72_l2301_230130


namespace combined_cost_theorem_l2301_230111

-- Define the cost prices of the two articles
def cost_price_1 : ℝ := sorry
def cost_price_2 : ℝ := sorry

-- Define the conditions
def condition_1 : Prop :=
  (350 - cost_price_1) = 1.12 * (280 - cost_price_1)

def condition_2 : Prop :=
  (420 - cost_price_2) = 1.08 * (380 - cost_price_2)

-- Theorem to prove
theorem combined_cost_theorem :
  condition_1 ∧ condition_2 → cost_price_1 + cost_price_2 = 423.33 :=
by sorry

end combined_cost_theorem_l2301_230111


namespace circle_diameter_endpoint_l2301_230174

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ

/-- A diameter of a circle -/
structure Diameter where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Given a circle with center (1, 2) and one endpoint of a diameter at (4, 6),
    the other endpoint of the diameter is at (-2, -2) -/
theorem circle_diameter_endpoint (P : Circle) (d : Diameter) :
  P.center = (1, 2) →
  d.endpoint1 = (4, 6) →
  d.endpoint2 = (-2, -2) := by
  sorry

end circle_diameter_endpoint_l2301_230174


namespace fill_tank_theorem_l2301_230151

/-- Calculates the remaining water needed to fill a tank -/
def remaining_water (tank_capacity : ℕ) (pour_rate : ℚ) (pour_time : ℕ) : ℕ :=
  tank_capacity - (pour_time / 15 : ℕ)

/-- Theorem: Given a 150-gallon tank, pouring water at 1 gallon per 15 seconds for 525 seconds,
    the remaining water needed to fill the tank is 115 gallons -/
theorem fill_tank_theorem :
  remaining_water 150 (1/15 : ℚ) 525 = 115 := by
  sorry

end fill_tank_theorem_l2301_230151


namespace quadratic_factorization_l2301_230125

theorem quadratic_factorization (b c d e f : ℤ) : 
  (∀ x : ℚ, 24 * x^2 + b * x + 24 = (c * x + d) * (e * x + f)) →
  c + d = 10 →
  c * e = 24 →
  d * f = 24 →
  b = 52 :=
by sorry

end quadratic_factorization_l2301_230125


namespace mans_speed_against_stream_l2301_230160

theorem mans_speed_against_stream 
  (rate : ℝ) 
  (speed_with_stream : ℝ) 
  (h1 : rate = 4) 
  (h2 : speed_with_stream = 12) : 
  abs (rate - (speed_with_stream - rate)) = 4 :=
by sorry

end mans_speed_against_stream_l2301_230160


namespace wednesday_tips_calculation_l2301_230196

/-- Represents Hallie's work data for a day -/
structure WorkDay where
  hours : ℕ
  tips : ℕ

/-- Calculates the total earnings for a given work day with an hourly rate -/
def dailyEarnings (day : WorkDay) (hourlyRate : ℕ) : ℕ :=
  day.hours * hourlyRate + day.tips

theorem wednesday_tips_calculation (hourlyRate : ℕ) (monday tuesday wednesday : WorkDay) 
    (totalEarnings : ℕ) : 
    hourlyRate = 10 →
    monday.hours = 7 →
    monday.tips = 18 →
    tuesday.hours = 5 →
    tuesday.tips = 12 →
    wednesday.hours = 7 →
    totalEarnings = 240 →
    totalEarnings = dailyEarnings monday hourlyRate + 
                    dailyEarnings tuesday hourlyRate + 
                    dailyEarnings wednesday hourlyRate →
    wednesday.tips = 20 := by
  sorry

#check wednesday_tips_calculation

end wednesday_tips_calculation_l2301_230196


namespace heart_diamond_inequality_l2301_230123

def heart (x y : ℝ) : ℝ := |x - y|

def diamond (z w : ℝ) : ℝ := (z + w)^2

theorem heart_diamond_inequality : ∃ x y : ℝ, (heart x y)^2 ≠ diamond x y := by sorry

end heart_diamond_inequality_l2301_230123


namespace intersection_segment_length_l2301_230132

/-- The length of the line segment formed by the intersection of a line and an ellipse -/
theorem intersection_segment_length 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_ecc : (a^2 - b^2) / a^2 = 1/2) 
  (h_focal : 2 * Real.sqrt (a^2 - b^2) = 2) : 
  ∃ (A B : ℝ × ℝ), 
    (A.2 = -A.1 + 1 ∧ A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧ 
    (B.2 = -B.1 + 1 ∧ B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧ 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4 * Real.sqrt 2 / 3 := by
  sorry

end intersection_segment_length_l2301_230132


namespace ellipse_tangent_product_l2301_230107

/-- An ellipse with its key points -/
structure Ellipse where
  A : ℝ × ℝ  -- Major axis endpoint
  B : ℝ × ℝ  -- Minor axis endpoint
  F₁ : ℝ × ℝ  -- Focus 1
  F₂ : ℝ × ℝ  -- Focus 2

/-- Vector dot product -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Vector from point to point -/
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Tangent of angle between three points -/
noncomputable def tan_angle (p q r : ℝ × ℝ) : ℝ :=
  let v1 := vector q p
  let v2 := vector q r
  (v1.2 * v2.1 - v1.1 * v2.2) / (v1.1 * v2.1 + v1.2 * v2.2)

/-- Main theorem -/
theorem ellipse_tangent_product (Γ : Ellipse) 
  (h : dot_product (vector Γ.A Γ.F₁) (vector Γ.A Γ.F₂) + 
       dot_product (vector Γ.B Γ.F₁) (vector Γ.B Γ.F₂) = 0) : 
  tan_angle Γ.A Γ.B Γ.F₁ * tan_angle Γ.A Γ.B Γ.F₂ = -1/5 := by
  sorry

end ellipse_tangent_product_l2301_230107


namespace power_bank_sales_theorem_l2301_230189

/-- Represents the sales scenario of mobile power banks -/
structure PowerBankSales where
  m : ℝ  -- Wholesale price per power bank
  n : ℝ  -- Markup per power bank
  total_count : ℕ := 100  -- Total number of power banks
  full_price_sold : ℕ := 60  -- Number of power banks sold at full price
  discount_rate : ℝ := 0.2  -- Discount rate for remaining power banks

/-- Calculates the total selling price of all power banks -/
def total_selling_price (s : PowerBankSales) : ℝ :=
  s.total_count * (s.m + s.n)

/-- Calculates the actual total revenue -/
def actual_revenue (s : PowerBankSales) : ℝ :=
  s.full_price_sold * (s.m + s.n) + 
  (s.total_count - s.full_price_sold) * (1 - s.discount_rate) * (s.m + s.n)

/-- Calculates the additional profit without discount -/
def additional_profit (s : PowerBankSales) : ℝ :=
  s.total_count * s.n - (actual_revenue s - s.total_count * s.m)

theorem power_bank_sales_theorem (s : PowerBankSales) :
  total_selling_price s = 100 * (s.m + s.n) ∧
  actual_revenue s = 92 * (s.m + s.n) ∧
  additional_profit s = 8 * (s.m + s.n) := by
  sorry

#check power_bank_sales_theorem

end power_bank_sales_theorem_l2301_230189


namespace three_not_in_range_of_g_l2301_230193

/-- The quadratic function g(x) = x^2 + 3x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

/-- 3 is not in the range of g(x) if and only if c > 21/4 -/
theorem three_not_in_range_of_g (c : ℝ) :
  (∀ x, g c x ≠ 3) ↔ c > 21/4 := by sorry

end three_not_in_range_of_g_l2301_230193


namespace f_of_three_equals_nine_sevenths_l2301_230184

/-- Given f(x) = (2x + 3) / (4x - 5), prove that f(3) = 9/7 -/
theorem f_of_three_equals_nine_sevenths :
  let f : ℝ → ℝ := λ x ↦ (2*x + 3) / (4*x - 5)
  f 3 = 9/7 := by
  sorry

end f_of_three_equals_nine_sevenths_l2301_230184


namespace time_until_sunset_l2301_230116

-- Define the initial sunset time in minutes past midnight
def initial_sunset : ℕ := 18 * 60

-- Define the daily sunset delay in minutes
def daily_delay : ℚ := 1.2

-- Define the number of days since March 1st
def days_passed : ℕ := 40

-- Define the current time in minutes past midnight
def current_time : ℕ := 18 * 60 + 10

-- Theorem statement
theorem time_until_sunset :
  let total_delay : ℚ := daily_delay * days_passed
  let new_sunset : ℚ := initial_sunset + total_delay
  ⌊new_sunset⌋ - current_time = 38 := by sorry

end time_until_sunset_l2301_230116


namespace exams_fourth_year_l2301_230115

theorem exams_fourth_year 
  (a b c d e : ℕ) 
  (h_sum : a + b + c + d + e = 31)
  (h_order : a < b ∧ b < c ∧ c < d ∧ d < e)
  (h_fifth : e = 3 * a)
  : d = 8 := by
  sorry

end exams_fourth_year_l2301_230115


namespace bob_age_proof_l2301_230186

theorem bob_age_proof :
  ∃! x : ℕ, 
    x > 0 ∧ 
    ∃ y : ℕ, x - 2 = y^2 ∧
    ∃ z : ℕ, x + 2 = z^3 ∧
    x = 123 := by
  sorry

end bob_age_proof_l2301_230186


namespace mentorship_arrangements_count_l2301_230122

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates the number of permutations of k items from n items --/
def permutations (n k : ℕ) : ℕ := sorry

/-- Calculates the number of mentorship arrangements for 5 students and 3 teachers --/
def mentorshipArrangements : ℕ :=
  let studentGroups := choose 5 2 * choose 3 2 * choose 1 1 / 2
  studentGroups * permutations 3 3

theorem mentorship_arrangements_count :
  mentorshipArrangements = 90 := by sorry

end mentorship_arrangements_count_l2301_230122


namespace lines_equivalence_l2301_230144

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A cylinder in 3D space -/
structure Cylinder3D where
  axis : Line3D
  radius : ℝ

/-- The set of lines passing through a point and at a given distance from another line -/
def linesAtDistanceFromLine (M : Point3D) (d : ℝ) (AB : Line3D) : Set Line3D :=
  sorry

/-- The set of lines lying in two planes tangent to a cylinder passing through a point -/
def linesInTangentPlanes (M : Point3D) (cylinder : Cylinder3D) : Set Line3D :=
  sorry

/-- Theorem stating the equivalence of the two sets of lines -/
theorem lines_equivalence (M : Point3D) (d : ℝ) (AB : Line3D) :
  let cylinder := Cylinder3D.mk AB d
  linesAtDistanceFromLine M d AB = linesInTangentPlanes M cylinder :=
sorry

end lines_equivalence_l2301_230144


namespace largest_n_for_negative_quadratic_five_satisfies_condition_six_does_not_satisfy_l2301_230199

theorem largest_n_for_negative_quadratic : 
  ∀ n : ℤ, n^2 - 9*n + 18 < 0 → n ≤ 5 :=
by sorry

theorem five_satisfies_condition : 
  (5 : ℤ)^2 - 9*5 + 18 < 0 :=
by sorry

theorem six_does_not_satisfy : 
  ¬((6 : ℤ)^2 - 9*6 + 18 < 0) :=
by sorry

end largest_n_for_negative_quadratic_five_satisfies_condition_six_does_not_satisfy_l2301_230199


namespace next_joint_work_day_is_360_l2301_230145

/-- Represents the work schedule of a staff member -/
structure WorkSchedule where
  cycle : Nat
  deriving Repr

/-- Represents the community center with its staff members -/
structure CommunityCenter where
  alan : WorkSchedule
  berta : WorkSchedule
  carlos : WorkSchedule
  dora : WorkSchedule

/-- Calculates the next day when all staff members work together -/
def nextJointWorkDay (center : CommunityCenter) : Nat :=
  Nat.lcm center.alan.cycle (Nat.lcm center.berta.cycle (Nat.lcm center.carlos.cycle center.dora.cycle))

/-- The main theorem: proving that the next joint work day is 360 days from today -/
theorem next_joint_work_day_is_360 (center : CommunityCenter) 
  (h1 : center.alan.cycle = 5)
  (h2 : center.berta.cycle = 6)
  (h3 : center.carlos.cycle = 8)
  (h4 : center.dora.cycle = 9) :
  nextJointWorkDay center = 360 := by
  sorry

end next_joint_work_day_is_360_l2301_230145


namespace field_trip_groups_l2301_230178

/-- Given the conditions for a field trip lunch preparation, prove the number of groups. -/
theorem field_trip_groups (
  sandwiches_per_student : ℕ)
  (bread_per_sandwich : ℕ)
  (students_per_group : ℕ)
  (total_bread : ℕ)
  (h1 : sandwiches_per_student = 2)
  (h2 : bread_per_sandwich = 2)
  (h3 : students_per_group = 6)
  (h4 : total_bread = 120) :
  total_bread / (bread_per_sandwich * sandwiches_per_student * students_per_group) = 5 := by
  sorry

end field_trip_groups_l2301_230178


namespace principal_calculation_l2301_230150

theorem principal_calculation (P r : ℝ) 
  (h1 : P * (1 + 2 * r) = 720)
  (h2 : P * (1 + 7 * r) = 1020) : 
  P = 600 := by
sorry

end principal_calculation_l2301_230150


namespace no_real_solutions_l2301_230168

theorem no_real_solutions :
  ¬∃ (x : ℝ), (6 * x) / (x^2 + 2 * x + 5) + (7 * x) / (x^2 - 7 * x + 5) = -3/2 :=
by sorry

end no_real_solutions_l2301_230168


namespace profit_increase_approx_l2301_230101

/-- Represents the monthly profit changes as factors -/
def march_to_april : ℝ := 1.35
def april_to_may : ℝ := 0.80
def may_to_june : ℝ := 1.50
def june_to_july : ℝ := 0.75
def july_to_august : ℝ := 1.45

/-- The overall factor of profit change from March to August -/
def overall_factor : ℝ :=
  march_to_april * april_to_may * may_to_june * june_to_july * july_to_august

/-- The overall percentage increase from March to August -/
def overall_percentage_increase : ℝ := (overall_factor - 1) * 100

/-- Theorem stating the overall percentage increase is approximately 21.95% -/
theorem profit_increase_approx :
  ∃ ε > 0, abs (overall_percentage_increase - 21.95) < ε :=
sorry

end profit_increase_approx_l2301_230101


namespace candy_packing_problem_l2301_230173

theorem candy_packing_problem :
  ∃! n : ℕ, 11 ≤ n ∧ n ≤ 100 ∧ 
    6 ∣ n ∧ 9 ∣ n ∧ n % 7 = 1 ∧
    n = 36 := by sorry

end candy_packing_problem_l2301_230173


namespace half_abs_diff_squares_18_16_l2301_230155

theorem half_abs_diff_squares_18_16 : (1 / 2 : ℝ) * |18^2 - 16^2| = 34 := by
  sorry

end half_abs_diff_squares_18_16_l2301_230155


namespace image_difference_l2301_230154

/-- Define the mapping f -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.1 + p.2)

/-- Theorem statement -/
theorem image_difference (m n : ℝ) (h : (m, n) = f (2, 1)) :
  m - n = -1 := by
  sorry

end image_difference_l2301_230154


namespace train_speed_l2301_230105

theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_l2301_230105


namespace inverse_sum_zero_l2301_230183

theorem inverse_sum_zero (a b : ℝ) (h : a * b = 1) :
  a^2015 * b^2016 + a^2016 * b^2017 + a^2017 * b^2016 + a^2016 * b^2015 = 0 := by
  sorry

end inverse_sum_zero_l2301_230183


namespace intersection_complement_equals_set_l2301_230112

def U : Set Int := Set.univ
def M : Set Int := {1, 2}
def P : Set Int := {-2, -1, 0, 1, 2}

theorem intersection_complement_equals_set : P ∩ (U \ M) = {-2, -1, 0} := by
  sorry

end intersection_complement_equals_set_l2301_230112


namespace cone_lateral_surface_area_l2301_230110

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 3
  let h : ℝ := 4
  let l : ℝ := Real.sqrt (r^2 + h^2)
  π * r * l = 15 * π := by
  sorry

end cone_lateral_surface_area_l2301_230110


namespace parabola_segment_sum_l2301_230104

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 12*y

-- Define the focus F (we don't know its exact coordinates, so we leave it abstract)
variable (F : ℝ × ℝ)

-- Define points A, B, and P
variable (A B : ℝ × ℝ)
def P : ℝ × ℝ := (2, 1)

-- State that A and B are on the parabola
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2

-- State that P is the midpoint of AB
axiom P_is_midpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- State the theorem
theorem parabola_segment_sum : 
  ∀ (F A B : ℝ × ℝ), 
  parabola A.1 A.2 → 
  parabola B.1 B.2 → 
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  dist A F + dist B F = 8 := by sorry

end parabola_segment_sum_l2301_230104


namespace proposition_relationship_l2301_230182

theorem proposition_relationship (x y : ℤ) :
  (∀ x y, x + y ≠ 2010 → (x ≠ 1010 ∨ y ≠ 1000)) ∧
  (∃ x y, (x ≠ 1010 ∨ y ≠ 1000) ∧ x + y = 2010) :=
by sorry

end proposition_relationship_l2301_230182


namespace brick_width_calculation_l2301_230121

theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  total_bricks = 20000 →
  ∃ brick_width : ℝ,
    brick_width = 0.1 ∧
    courtyard_length * courtyard_width * 10000 = brick_length * brick_width * total_bricks :=
by sorry

end brick_width_calculation_l2301_230121


namespace kermit_final_positions_l2301_230143

/-- The number of integer coordinate pairs (x, y) satisfying |x| + |y| = n -/
def count_coordinate_pairs (n : ℕ) : ℕ :=
  2 * (n + 1) * (n + 1) - 2 * n * (n + 1) + 1

/-- Kermit's energy in Joules -/
def kermit_energy : ℕ := 100

theorem kermit_final_positions : 
  count_coordinate_pairs kermit_energy = 10201 :=
sorry

end kermit_final_positions_l2301_230143


namespace remainder_problem_l2301_230152

theorem remainder_problem (g : ℕ) (h1 : g = 101) (h2 : 4351 % g = 8) :
  5161 % g = 10 := by
  sorry

end remainder_problem_l2301_230152


namespace find_A_l2301_230142

theorem find_A : ∃ A : ℕ, A % 5 = 4 ∧ A / 5 = 6 ∧ A = 34 := by
  sorry

end find_A_l2301_230142


namespace triangle_similarity_theorem_l2301_230147

-- Define the properties of the first triangle
def first_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ a = 15 ∧ c = 34

-- Define the similarity ratio between the two triangles
def similarity_ratio (r : ℝ) : Prop :=
  r = 102 / 34

-- Define the shortest side of the second triangle
def shortest_side (x : ℝ) : Prop :=
  x = 3 * Real.sqrt 931

-- Theorem statement
theorem triangle_similarity_theorem :
  ∀ a b c r x : ℝ,
  first_triangle a b c →
  similarity_ratio r →
  shortest_side x →
  x = r * a :=
by sorry

end triangle_similarity_theorem_l2301_230147


namespace ginger_water_usage_l2301_230194

/-- Calculates the total water used by Ginger for drinking and watering plants -/
def total_water_used (work_hours : ℕ) (bottle_capacity : ℚ) 
  (first_hour_drink : ℚ) (second_hour_drink : ℚ) (third_hour_drink : ℚ) 
  (hourly_increase : ℚ) (plant_type1_water : ℚ) (plant_type2_water : ℚ) 
  (plant_type3_water : ℚ) (plant_type1_count : ℕ) (plant_type2_count : ℕ) 
  (plant_type3_count : ℕ) : ℚ :=
  sorry

theorem ginger_water_usage :
  total_water_used 8 2 1 (3/2) 2 (1/2) 3 4 5 2 3 4 = 60 := by
  sorry

end ginger_water_usage_l2301_230194


namespace ratio_equality_l2301_230113

theorem ratio_equality (x y z : ℝ) (h : x / 3 = y / 5 ∧ y / 5 = z / 7) :
  (x - y + z) / (x + y - z) = 5 := by sorry

end ratio_equality_l2301_230113


namespace foldPointSetArea_l2301_230127

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Definition of a right triangle ABC with AB = 45, AC = 90 -/
def rightTriangle : Triangle :=
  { A := { x := 0, y := 0 }
  , B := { x := 45, y := 0 }
  , C := { x := 0, y := 90 }
  }

/-- A point P is a fold point if creases formed when A, B, and C are folded onto P do not intersect inside the triangle -/
def isFoldPoint (P : Point) (T : Triangle) : Prop := sorry

/-- The set of all fold points for a given triangle -/
def foldPointSet (T : Triangle) : Set Point :=
  {P | isFoldPoint P T}

/-- The area of a set of points -/
def areaOfSet (S : Set Point) : ℝ := sorry

/-- Theorem: The area of the fold point set for the right triangle is 506.25π - 607.5√3 -/
theorem foldPointSetArea :
  areaOfSet (foldPointSet rightTriangle) = 506.25 * Real.pi - 607.5 * Real.sqrt 3 := by
  sorry

end foldPointSetArea_l2301_230127


namespace survey_result_l2301_230191

theorem survey_result (U : Finset Int) (A B : Finset Int) 
  (h1 : Finset.card U = 70)
  (h2 : Finset.card A = 37)
  (h3 : Finset.card B = 49)
  (h4 : Finset.card (A ∩ B) = 20) :
  Finset.card (U \ (A ∪ B)) = 4 := by
  sorry

end survey_result_l2301_230191


namespace surface_area_unchanged_l2301_230187

/-- Represents a cube with given side length -/
structure Cube where
  side : ℝ
  side_pos : side > 0

/-- Calculates the surface area of a cube -/
def surfaceArea (c : Cube) : ℝ := 6 * c.side^2

/-- Represents the original cube -/
def originalCube : Cube := ⟨4, by norm_num⟩

/-- Represents a corner cube -/
def cornerCube : Cube := ⟨2, by norm_num⟩

/-- The number of corners in a cube -/
def numCorners : ℕ := 8

theorem surface_area_unchanged : 
  surfaceArea originalCube = surfaceArea originalCube - numCorners * (
    3 * cornerCube.side^2 - 3 * cornerCube.side^2
  ) := by sorry

end surface_area_unchanged_l2301_230187


namespace min_reciprocal_sum_of_roots_l2301_230172

/-- A quadratic function f(x) = 2x² + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + c

theorem min_reciprocal_sum_of_roots (b c : ℝ) :
  (f b c (-10) = f b c 12) →
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ f b c x₁ = 0 ∧ f b c x₂ = 0) →
  (∃ m : ℝ, m = 2 ∧ ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f b c x₁ = 0 → f b c x₂ = 0 → 1/x₁ + 1/x₂ ≥ m) :=
by sorry

end min_reciprocal_sum_of_roots_l2301_230172


namespace circle_center_l2301_230170

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius. -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_center (x y : ℝ) :
  CircleEquation 3 0 3 x y → (3, 0) = (3, 0) := by
  sorry

end circle_center_l2301_230170


namespace modulo_six_equality_l2301_230100

theorem modulo_six_equality : 47^1860 - 25^1860 ≡ 0 [ZMOD 6] := by
  sorry

end modulo_six_equality_l2301_230100


namespace rectangular_prism_parallel_edges_l2301_230126

/-- A rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ :=
  3 -- Each dimension contributes 2 pairs, so 2 + 2 + 2 = 6

/-- Theorem stating that a rectangular prism with dimensions 8, 4, and 2 has 6 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges :
  let prism : RectangularPrism := { length := 8, width := 4, height := 2 }
  parallel_edge_pairs prism = 6 := by
  sorry


end rectangular_prism_parallel_edges_l2301_230126


namespace ellipse_focus_distance_l2301_230163

/-- An ellipse in the first quadrant tangent to both axes with foci at (4,8) and (d,8) -/
structure Ellipse where
  d : ℝ
  tangent_to_axes : Bool
  in_first_quadrant : Bool
  focus1 : ℝ × ℝ := (4, 8)
  focus2 : ℝ × ℝ := (d, 8)

/-- The theorem stating that d = 30 for the given ellipse -/
theorem ellipse_focus_distance (e : Ellipse) (h1 : e.tangent_to_axes) (h2 : e.in_first_quadrant) :
  e.d = 30 := by
  sorry

end ellipse_focus_distance_l2301_230163


namespace basic_computer_price_l2301_230129

/-- Given the price of a basic computer and printer, prove the price of the basic computer. -/
theorem basic_computer_price (basic_price printer_price enhanced_price : ℝ) : 
  basic_price + printer_price = 2500 →
  enhanced_price = basic_price + 500 →
  enhanced_price + printer_price = 6 * printer_price →
  basic_price = 2000 := by
  sorry

end basic_computer_price_l2301_230129


namespace discount_difference_l2301_230166

theorem discount_difference (bill : ℝ) (single_discount : ℝ) 
  (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) : 
  bill = 12000 ∧ 
  single_discount = 0.35 ∧ 
  discount1 = 0.25 ∧ 
  discount2 = 0.08 ∧ 
  discount3 = 0.02 → 
  bill * (1 - (1 - discount1) * (1 - discount2) * (1 - discount3)) - 
  bill * single_discount = 314.40 := by
  sorry

end discount_difference_l2301_230166


namespace remainder_thirteen_plus_x_l2301_230169

theorem remainder_thirteen_plus_x (x : ℕ+) (h : 8 * x.val ≡ 1 [MOD 29]) :
  (13 + x.val) % 29 = 18 := by
  sorry

end remainder_thirteen_plus_x_l2301_230169


namespace probability_not_adjacent_l2301_230177

-- Define the total number of chairs
def total_chairs : ℕ := 10

-- Define the number of available chairs (excluding first and last)
def available_chairs : ℕ := total_chairs - 2

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the number of adjacent pairs in the available chairs
def adjacent_pairs : ℕ := available_chairs - 1

-- Theorem statement
theorem probability_not_adjacent :
  (1 : ℚ) - (adjacent_pairs : ℚ) / (choose available_chairs 2) = 3/4 :=
sorry

end probability_not_adjacent_l2301_230177


namespace least_number_for_divisibility_l2301_230157

theorem least_number_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((246835 + y) % 169 = 0 ∧ (246835 + y) % 289 = 0)) ∧ 
  ((246835 + x) % 169 = 0 ∧ (246835 + x) % 289 = 0) :=
by sorry

end least_number_for_divisibility_l2301_230157


namespace two_tangent_or_parallel_lines_l2301_230131

/-- A parabola in the x-y plane defined by y^2 = -8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = -8 * p.1}

/-- The point P through which the lines must pass -/
def P : ℝ × ℝ := (-2, -4)

/-- A line that passes through point P and has only one common point with the parabola -/
def TangentOrParallelLine (l : Set (ℝ × ℝ)) : Prop :=
  P ∈ l ∧ (∃! p, p ∈ l ∩ Parabola)

/-- There are exactly two lines that pass through P and have only one common point with the parabola -/
theorem two_tangent_or_parallel_lines : 
  ∃! (l1 l2 : Set (ℝ × ℝ)), l1 ≠ l2 ∧ TangentOrParallelLine l1 ∧ TangentOrParallelLine l2 ∧ 
  (∀ l, TangentOrParallelLine l → l = l1 ∨ l = l2) :=
sorry

end two_tangent_or_parallel_lines_l2301_230131


namespace prob_same_group_is_one_third_l2301_230114

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The set of all possible outcomes when two students choose interest groups -/
def total_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_groups) (Finset.range num_groups)

/-- The set of outcomes where both students choose the same group -/
def same_group_outcomes : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 = p.2) total_outcomes

/-- The probability of two students choosing the same interest group -/
def prob_same_group : ℚ :=
  (same_group_outcomes.card : ℚ) / (total_outcomes.card : ℚ)

theorem prob_same_group_is_one_third :
  prob_same_group = 1 / 3 := by sorry

end prob_same_group_is_one_third_l2301_230114


namespace range_of_k_for_inequality_l2301_230108

theorem range_of_k_for_inequality (k : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - 3| > |k - 1|) → k ∈ Set.Ioo 0 2 := by
  sorry

end range_of_k_for_inequality_l2301_230108
