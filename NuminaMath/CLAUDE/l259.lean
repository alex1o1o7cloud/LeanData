import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l259_25996

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l259_25996


namespace NUMINAMATH_CALUDE_inequality_proof_l259_25942

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_condition : a + b + c < 2) : 
  Real.sqrt (a^2 + b*c) + Real.sqrt (b^2 + c*a) + Real.sqrt (c^2 + a*b) < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l259_25942


namespace NUMINAMATH_CALUDE_three_letter_initials_count_l259_25960

theorem three_letter_initials_count (n : ℕ) (h : n = 7) : n^3 = 343 := by
  sorry

end NUMINAMATH_CALUDE_three_letter_initials_count_l259_25960


namespace NUMINAMATH_CALUDE_tangent_line_equation_l259_25916

-- Define the function f
def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x

-- State the theorem
theorem tangent_line_equation :
  (∀ x, f (x / 2) = x^3 - 3 * x) →
  ∃ m b, m * 1 - f 1 + b = 0 ∧
         ∀ x, m * x - f x + b = 0 → x = 1 ∧
         m = 18 ∧ b = -16 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l259_25916


namespace NUMINAMATH_CALUDE_classroom_capacity_l259_25981

/-- The number of rows of desks in the classroom -/
def num_rows : ℕ := 8

/-- The number of desks in the first row -/
def first_row_desks : ℕ := 10

/-- The increase in number of desks for each subsequent row -/
def desk_increase : ℕ := 2

/-- The total number of desks in the classroom -/
def total_desks : ℕ := (num_rows * (2 * first_row_desks + (num_rows - 1) * desk_increase)) / 2

theorem classroom_capacity :
  total_desks = 136 := by sorry

end NUMINAMATH_CALUDE_classroom_capacity_l259_25981


namespace NUMINAMATH_CALUDE_midpoints_collinear_l259_25959

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Two mutually perpendicular lines passing through a point -/
structure PerpendicularLines where
  origin : ℝ × ℝ
  direction1 : ℝ × ℝ
  direction2 : ℝ × ℝ
  perpendicular : direction1.1 * direction2.1 + direction1.2 * direction2.2 = 0

/-- Intersection points of lines with triangle sides -/
def intersectionPoints (t : Triangle) (l : PerpendicularLines) : List (ℝ × ℝ) := sorry

/-- Midpoints of segments -/
def midpoints (points : List (ℝ × ℝ)) : List (ℝ × ℝ) := sorry

/-- Check if points are collinear -/
def areCollinear (points : List (ℝ × ℝ)) : Prop := sorry

/-- Main theorem -/
theorem midpoints_collinear (t : Triangle) :
  let o := orthocenter t
  let l := PerpendicularLines.mk o (1, 0) (0, 1) (by simp)
  let intersections := intersectionPoints t l
  let mids := midpoints intersections
  areCollinear mids := by sorry

end NUMINAMATH_CALUDE_midpoints_collinear_l259_25959


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l259_25914

/-- Represents the cost structure and consumption of a pizza --/
structure PizzaOrder where
  total_slices : Nat
  plain_cost : Int
  cheese_slices : Nat
  veggie_slices : Nat
  topping_cost : Int
  jerry_plain_slices : Nat

/-- Calculates the difference in payment between Jerry and Tom --/
def payment_difference (order : PizzaOrder) : Int :=
  let total_cost := order.plain_cost + 2 * order.topping_cost
  let slice_cost := total_cost / order.total_slices
  let jerry_slices := order.cheese_slices + order.veggie_slices + order.jerry_plain_slices
  let tom_slices := order.total_slices - jerry_slices
  slice_cost * (jerry_slices - tom_slices)

/-- Theorem stating the difference in payment between Jerry and Tom --/
theorem pizza_payment_difference :
  ∃ (order : PizzaOrder),
    order.total_slices = 12 ∧
    order.plain_cost = 12 ∧
    order.cheese_slices = 4 ∧
    order.veggie_slices = 4 ∧
    order.topping_cost = 3 ∧
    order.jerry_plain_slices = 2 ∧
    payment_difference order = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l259_25914


namespace NUMINAMATH_CALUDE_fence_cost_for_square_plot_l259_25976

theorem fence_cost_for_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 81) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 2088 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_for_square_plot_l259_25976


namespace NUMINAMATH_CALUDE_g_of_eight_l259_25962

/-- Given a function g : ℝ → ℝ satisfying the equation
    g(x) + g(3x+y) + 7xy = g(4x - y) + 3x^2 + 2 for all real x and y,
    prove that g(8) = -30. -/
theorem g_of_eight (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, g x + g (3*x + y) + 7*x*y = g (4*x - y) + 3*x^2 + 2) : 
  g 8 = -30 := by
  sorry

end NUMINAMATH_CALUDE_g_of_eight_l259_25962


namespace NUMINAMATH_CALUDE_min_garden_cost_l259_25953

/-- Represents the dimensions of a rectangular region -/
structure Region where
  length : ℝ
  width : ℝ

/-- Represents a type of flower with its cost -/
structure Flower where
  name : String
  cost : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Calculates the cost of planting a flower in a region -/
def plantingCost (f : Flower) (r : Region) : ℝ := f.cost * area r

/-- The main theorem stating the minimum cost of the garden -/
theorem min_garden_cost (regions : List Region) (flowers : List Flower) : 
  regions.length = 5 →
  flowers.length = 5 →
  regions = [
    ⟨5, 2⟩, 
    ⟨7, 3⟩, 
    ⟨5, 5⟩, 
    ⟨2, 4⟩, 
    ⟨5, 4⟩
  ] →
  flowers = [
    ⟨"Marigold", 1⟩,
    ⟨"Sunflower", 1.75⟩,
    ⟨"Tulip", 1.25⟩,
    ⟨"Orchid", 2.75⟩,
    ⟨"Iris", 3.25⟩
  ] →
  ∃ (assignment : List (Flower × Region)), 
    assignment.length = 5 ∧ 
    (∀ f r, (f, r) ∈ assignment → f ∈ flowers ∧ r ∈ regions) ∧
    (∀ f, f ∈ flowers → ∃! r, (f, r) ∈ assignment) ∧
    (∀ r, r ∈ regions → ∃! f, (f, r) ∈ assignment) ∧
    (assignment.map (λ (f, r) => plantingCost f r)).sum = 140.75 ∧
    ∀ (other_assignment : List (Flower × Region)),
      other_assignment.length = 5 →
      (∀ f r, (f, r) ∈ other_assignment → f ∈ flowers ∧ r ∈ regions) →
      (∀ f, f ∈ flowers → ∃! r, (f, r) ∈ other_assignment) →
      (∀ r, r ∈ regions → ∃! f, (f, r) ∈ other_assignment) →
      (other_assignment.map (λ (f, r) => plantingCost f r)).sum ≥ 140.75 :=
by sorry

end NUMINAMATH_CALUDE_min_garden_cost_l259_25953


namespace NUMINAMATH_CALUDE_not_p_or_not_q_false_implies_p_and_q_and_p_or_q_true_l259_25929

theorem not_p_or_not_q_false_implies_p_and_q_and_p_or_q_true (p q : Prop) :
  (¬(¬p ∨ ¬q)) → ((p ∧ q) ∧ (p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_false_implies_p_and_q_and_p_or_q_true_l259_25929


namespace NUMINAMATH_CALUDE_power_of_two_equality_l259_25980

theorem power_of_two_equality (n : ℕ) : 2^n = 2 * 16^2 * 4^3 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l259_25980


namespace NUMINAMATH_CALUDE_circle_plus_minus_balance_l259_25901

theorem circle_plus_minus_balance (a b p q : ℕ) : a - b = p - q :=
  sorry

end NUMINAMATH_CALUDE_circle_plus_minus_balance_l259_25901


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_perfect_square_l259_25961

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_12_less_than_perfect_square :
  ∃ (n : ℕ), is_perfect_square n ∧ 
             is_prime (n - 12) ∧ 
             (n - 12 = 13) ∧
             (∀ m : ℕ, is_perfect_square m → is_prime (m - 12) → m - 12 ≥ 13) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_perfect_square_l259_25961


namespace NUMINAMATH_CALUDE_computer_repair_cost_l259_25948

theorem computer_repair_cost (phone_cost laptop_cost : ℕ) 
  (phone_repairs laptop_repairs computer_repairs : ℕ) (total_earnings : ℕ) :
  phone_cost = 11 →
  laptop_cost = 15 →
  phone_repairs = 5 →
  laptop_repairs = 2 →
  computer_repairs = 2 →
  total_earnings = 121 →
  ∃ (computer_cost : ℕ), 
    phone_cost * phone_repairs + laptop_cost * laptop_repairs + computer_cost * computer_repairs = total_earnings ∧
    computer_cost = 18 :=
by sorry

end NUMINAMATH_CALUDE_computer_repair_cost_l259_25948


namespace NUMINAMATH_CALUDE_zoo_layout_problem_l259_25952

/-- The number of tiger enclosures in a zoo -/
def tigerEnclosures : ℕ := sorry

/-- The number of zebra enclosures in the zoo -/
def zebraEnclosures : ℕ := 2 * tigerEnclosures

/-- The number of giraffe enclosures in the zoo -/
def giraffeEnclosures : ℕ := 3 * zebraEnclosures

/-- The number of tigers per tiger enclosure -/
def tigersPerEnclosure : ℕ := 4

/-- The number of zebras per zebra enclosure -/
def zebrasPerEnclosure : ℕ := 10

/-- The number of giraffes per giraffe enclosure -/
def giraffesPerEnclosure : ℕ := 2

/-- The total number of animals in the zoo -/
def totalAnimals : ℕ := 144

theorem zoo_layout_problem :
  tigerEnclosures * tigersPerEnclosure +
  zebraEnclosures * zebrasPerEnclosure +
  giraffeEnclosures * giraffesPerEnclosure = totalAnimals ∧
  tigerEnclosures = 4 := by sorry

end NUMINAMATH_CALUDE_zoo_layout_problem_l259_25952


namespace NUMINAMATH_CALUDE_alex_lorin_marble_ratio_l259_25919

/-- Given the following conditions:
  - Lorin has 4 black marbles
  - Jimmy has 22 yellow marbles
  - Alex has a certain ratio of black marbles as Lorin
  - Alex has one half as many yellow marbles as Jimmy
  - Alex has 19 marbles in total

  Prove that the ratio of Alex's black marbles to Lorin's black marbles is 2:1
-/
theorem alex_lorin_marble_ratio :
  ∀ (alex_black alex_yellow : ℕ),
  let lorin_black : ℕ := 4
  let jimmy_yellow : ℕ := 22
  let alex_total : ℕ := 19
  alex_yellow = jimmy_yellow / 2 →
  alex_black + alex_yellow = alex_total →
  ∃ (r : ℚ),
    alex_black = r * lorin_black ∧
    r = 2 := by
  sorry

#check alex_lorin_marble_ratio

end NUMINAMATH_CALUDE_alex_lorin_marble_ratio_l259_25919


namespace NUMINAMATH_CALUDE_compound_interest_rate_l259_25951

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r)^10 = 9000) 
  (h2 : P * (1 + r)^11 = 9990) : 
  r = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l259_25951


namespace NUMINAMATH_CALUDE_prob_even_sum_is_seven_sixteenths_l259_25957

/-- Represents the dartboard with inner and outer circles and point values -/
structure Dartboard where
  inner_radius : ℝ
  outer_radius : ℝ
  inner_values : Fin 3 → ℕ
  outer_values : Fin 3 → ℕ

/-- Calculates the probability of getting an even sum with two darts -/
def prob_even_sum (d : Dartboard) : ℚ :=
  sorry

/-- The specific dartboard described in the problem -/
def problem_dartboard : Dartboard where
  inner_radius := 4
  outer_radius := 8
  inner_values := ![3, 5, 5]
  outer_values := ![4, 3, 3]

theorem prob_even_sum_is_seven_sixteenths :
  prob_even_sum problem_dartboard = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_seven_sixteenths_l259_25957


namespace NUMINAMATH_CALUDE_solution_set_of_decreasing_function_l259_25924

/-- A decreasing function on ℝ -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem solution_set_of_decreasing_function
  (f : ℝ → ℝ) (h_decreasing : DecreasingFunction f) (h_f_1 : f 1 = 0) :
  {x : ℝ | f (x - 1) < 0} = {x : ℝ | x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_decreasing_function_l259_25924


namespace NUMINAMATH_CALUDE_defective_units_percentage_l259_25925

/-- The percentage of defective units that are shipped for sale -/
def defective_shipped_percent : ℝ := 4

/-- The percentage of all units that are defective and shipped for sale -/
def total_defective_shipped_percent : ℝ := 0.2

/-- The percentage of all units that are defective -/
def defective_percent : ℝ := 5

theorem defective_units_percentage : 
  defective_shipped_percent * defective_percent / 100 = total_defective_shipped_percent := by
  sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l259_25925


namespace NUMINAMATH_CALUDE_cash_percentage_is_ten_percent_l259_25986

def total_amount : ℝ := 1000
def raw_materials_cost : ℝ := 500
def machinery_cost : ℝ := 400

theorem cash_percentage_is_ten_percent :
  (total_amount - (raw_materials_cost + machinery_cost)) / total_amount * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cash_percentage_is_ten_percent_l259_25986


namespace NUMINAMATH_CALUDE_michaels_initial_money_l259_25900

/-- 
Given:
- Michael's brother initially had $17
- Michael gave half of his money to his brother
- His brother spent $3 on candy
- His brother had $35 left after buying candy

Prove that Michael's initial amount of money was $42
-/
theorem michaels_initial_money (brother_initial : ℕ) (candy_cost : ℕ) (brother_final : ℕ) :
  brother_initial = 17 →
  candy_cost = 3 →
  brother_final = 35 →
  ∃ (michael_initial : ℕ), 
    brother_initial + michael_initial / 2 = brother_final + candy_cost ∧
    michael_initial = 42 := by
  sorry

end NUMINAMATH_CALUDE_michaels_initial_money_l259_25900


namespace NUMINAMATH_CALUDE_sum_of_roots_l259_25926

theorem sum_of_roots (h b : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂)
  (h2 : 3 * x₁^2 - h * x₁ = b)
  (h3 : 3 * x₂^2 - h * x₂ = b) : 
  x₁ + x₂ = h / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l259_25926


namespace NUMINAMATH_CALUDE_polynomial_factorization_l259_25950

theorem polynomial_factorization (x : ℝ) : 
  x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l259_25950


namespace NUMINAMATH_CALUDE_inequality_proof_l259_25928

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / (x + 2*y + 3*z)) + (y / (y + 2*z + 3*x)) + (z / (z + 2*x + 3*y)) ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l259_25928


namespace NUMINAMATH_CALUDE_equation_solution_l259_25969

theorem equation_solution : 
  let S : Set ℝ := {x | (x^4 + 4*x^3*Real.sqrt 3 + 12*x^2 + 8*x*Real.sqrt 3 + 4) + (x^2 + 2*x*Real.sqrt 3 + 3) = 0}
  S = {-Real.sqrt 3, -Real.sqrt 3 + 1, -Real.sqrt 3 - 1} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l259_25969


namespace NUMINAMATH_CALUDE_cos_triple_angle_l259_25944

theorem cos_triple_angle (α : ℝ) : Real.cos (3 * α) = 4 * (Real.cos α)^3 - 3 * Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_cos_triple_angle_l259_25944


namespace NUMINAMATH_CALUDE_debby_water_bottles_l259_25906

/-- The number of water bottles Debby drank per day -/
def bottles_per_day : ℕ := 109

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := 74

/-- The total number of bottles Debby bought -/
def total_bottles : ℕ := bottles_per_day * days_lasted

theorem debby_water_bottles : total_bottles = 8066 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l259_25906


namespace NUMINAMATH_CALUDE_meetings_percentage_of_workday_l259_25967

def workday_hours : ℕ := 10
def first_meeting_minutes : ℕ := 40

def second_meeting_minutes : ℕ := 2 * first_meeting_minutes
def third_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes + third_meeting_minutes
def workday_minutes : ℕ := workday_hours * 60

theorem meetings_percentage_of_workday :
  (total_meeting_minutes : ℚ) / (workday_minutes : ℚ) = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_meetings_percentage_of_workday_l259_25967


namespace NUMINAMATH_CALUDE_largest_class_size_l259_25971

/-- Represents the number of students in each class of a school --/
structure School :=
  (largest_class : ℕ)

/-- Calculates the total number of students in the school --/
def total_students (s : School) : ℕ :=
  s.largest_class + (s.largest_class - 2) + (s.largest_class - 4) + (s.largest_class - 6) + (s.largest_class - 8)

/-- Theorem stating that a school with 5 classes, where each class has 2 students less than the previous class, 
    and a total of 105 students, has 25 students in the largest class --/
theorem largest_class_size :
  ∃ (s : School), total_students s = 105 ∧ s.largest_class = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_class_size_l259_25971


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l259_25943

theorem linear_function_decreasing (a b y₁ y₂ : ℝ) :
  a < 0 →
  y₁ = 2 * a * (-1) - b →
  y₂ = 2 * a * 2 - b →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l259_25943


namespace NUMINAMATH_CALUDE_norris_remaining_money_l259_25937

/-- Calculates the remaining money for Norris after savings and spending --/
theorem norris_remaining_money 
  (september_savings : ℕ) 
  (october_savings : ℕ) 
  (november_savings : ℕ) 
  (game_cost : ℕ) : 
  september_savings = 29 →
  october_savings = 25 →
  november_savings = 31 →
  game_cost = 75 →
  (september_savings + october_savings + november_savings) - game_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_norris_remaining_money_l259_25937


namespace NUMINAMATH_CALUDE_dog_age_is_twelve_l259_25910

def cat_age : ℕ := 8

def rabbit_age (cat_age : ℕ) : ℕ := cat_age / 2

def dog_age (rabbit_age : ℕ) : ℕ := 3 * rabbit_age

theorem dog_age_is_twelve : dog_age (rabbit_age cat_age) = 12 := by
  sorry

end NUMINAMATH_CALUDE_dog_age_is_twelve_l259_25910


namespace NUMINAMATH_CALUDE_range_of_f_l259_25903

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l259_25903


namespace NUMINAMATH_CALUDE_strawberry_sales_formula_l259_25984

/-- The relationship between strawberry sales volume and total sales price -/
theorem strawberry_sales_formula (n : ℕ+) :
  let price_increase : ℝ := 40.5
  let total_price : ℕ+ → ℝ := λ k => k.val * price_increase
  total_price n = n.val * price_increase :=
by sorry

end NUMINAMATH_CALUDE_strawberry_sales_formula_l259_25984


namespace NUMINAMATH_CALUDE_inequality_implication_l259_25927

theorem inequality_implication (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l259_25927


namespace NUMINAMATH_CALUDE_room_occupancy_l259_25993

theorem room_occupancy (x : ℕ) : 
  (3 * x / 8 : ℚ) - 6 = 18 → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_room_occupancy_l259_25993


namespace NUMINAMATH_CALUDE_product_difference_difference_of_products_l259_25999

theorem product_difference (a b c d : ℝ) (h : a * b = c) : 
  a * d - a * b = a * (d - b) :=
by sorry

theorem difference_of_products : 
  (16.47 * 34) - (16.47 * 24) = 164.7 :=
by sorry

end NUMINAMATH_CALUDE_product_difference_difference_of_products_l259_25999


namespace NUMINAMATH_CALUDE_triangle_sides_ratio_bound_l259_25941

theorem triangle_sides_ratio_bound (a b c : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →  -- Positive sides
  (a + b > c) → (a + c > b) → (b + c > a) →  -- Triangle inequality
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →  -- Arithmetic progression
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) ≥ 1 := by
  sorry

#check triangle_sides_ratio_bound

end NUMINAMATH_CALUDE_triangle_sides_ratio_bound_l259_25941


namespace NUMINAMATH_CALUDE_ninas_pet_insects_eyes_l259_25911

/-- The total number of eyes among Nina's pet insects -/
def total_eyes (num_spiders num_ants spider_eyes ant_eyes : ℕ) : ℕ :=
  num_spiders * spider_eyes + num_ants * ant_eyes

/-- Theorem stating that the total number of eyes among Nina's pet insects is 124 -/
theorem ninas_pet_insects_eyes :
  total_eyes 3 50 8 2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_ninas_pet_insects_eyes_l259_25911


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l259_25938

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l259_25938


namespace NUMINAMATH_CALUDE_alice_bob_meet_l259_25994

/-- The number of points on the circle -/
def n : ℕ := 24

/-- Alice's starting position -/
def alice_start : ℕ := 1

/-- Bob's starting position -/
def bob_start : ℕ := 12

/-- Alice's movement per turn (clockwise) -/
def alice_move : ℕ := 7

/-- Bob's movement per turn (counterclockwise) -/
def bob_move : ℕ := 17

/-- The number of turns it takes for Alice and Bob to meet -/
def meeting_turns : ℕ := 5

/-- Theorem stating that Alice and Bob meet after the specified number of turns -/
theorem alice_bob_meet :
  (alice_start + meeting_turns * alice_move) % n = 
  (bob_start - meeting_turns * bob_move + n * meeting_turns) % n :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l259_25994


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l259_25912

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l259_25912


namespace NUMINAMATH_CALUDE_linear_functions_coefficient_difference_l259_25977

/-- Given linear functions f and g, and their composition h with a known inverse,
    prove that the difference of coefficients of f is 5. -/
theorem linear_functions_coefficient_difference (a b : ℝ) : 
  (∃ (f g h : ℝ → ℝ), 
    (∀ x, f x = a * x + b) ∧ 
    (∀ x, g x = -2 * x + 7) ∧ 
    (∀ x, h x = f (g x)) ∧ 
    (∀ x, Function.invFun h x = x + 9)) → 
  a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_linear_functions_coefficient_difference_l259_25977


namespace NUMINAMATH_CALUDE_total_area_equals_total_frequency_l259_25909

/-- A frequency distribution histogram -/
structure FrequencyHistogram where
  /-- The list of frequencies for each bin -/
  frequencies : List ℝ
  /-- All frequencies are non-negative -/
  all_nonneg : ∀ f ∈ frequencies, f ≥ 0

/-- The total frequency of a histogram -/
def totalFrequency (h : FrequencyHistogram) : ℝ :=
  h.frequencies.sum

/-- The total area of small rectangles in a histogram -/
def totalArea (h : FrequencyHistogram) : ℝ :=
  h.frequencies.sum

/-- Theorem: The total area of small rectangles in a frequency distribution histogram
    is equal to the total frequency -/
theorem total_area_equals_total_frequency (h : FrequencyHistogram) :
  totalArea h = totalFrequency h := by
  sorry


end NUMINAMATH_CALUDE_total_area_equals_total_frequency_l259_25909


namespace NUMINAMATH_CALUDE_trajectory_intersection_slope_ratio_l259_25936

-- Define the curve E: y² = 2x
def E : Set (ℝ × ℝ) := {p | p.2^2 = 2 * p.1}

-- Define points S and Q
def S : ℝ × ℝ := (2, 0)
def Q : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem trajectory_intersection_slope_ratio 
  (k₁ : ℝ) 
  (A B C D : ℝ × ℝ) 
  (hA : A ∈ E) 
  (hB : B ∈ E) 
  (hC : C ∈ E) 
  (hD : D ∈ E) 
  (hAB : (B.2 - A.2) = k₁ * (B.1 - A.1)) 
  (hABS : (A.2 - S.2) = k₁ * (A.1 - S.1)) 
  (hAC : (C.2 - A.2) * (Q.1 - A.1) = (Q.2 - A.2) * (C.1 - A.1)) 
  (hBD : (D.2 - B.2) * (Q.1 - B.1) = (Q.2 - B.2) * (D.1 - B.1)) :
  ∃ (k₂ : ℝ), (D.2 - C.2) = k₂ * (D.1 - C.1) ∧ k₂ / k₁ = 2 := by
sorry

end NUMINAMATH_CALUDE_trajectory_intersection_slope_ratio_l259_25936


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l259_25983

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 - 1840*m + 2009 = 0 → 
  n^2 - 1840*n + 2009 = 0 → 
  (m^2 - 1841*m + 2009) * (n^2 - 1841*n + 2009) = 2009 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l259_25983


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l259_25974

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, n^2 = 900 ∧ 
  (∀ m : ℕ, m > 0 → m^2 < 900 → ¬(2 ∣ m^2 ∧ 3 ∣ m^2 ∧ 5 ∣ m^2)) ∧
  2 ∣ 900 ∧ 3 ∣ 900 ∧ 5 ∣ 900 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l259_25974


namespace NUMINAMATH_CALUDE_unique_solution_system_l259_25975

theorem unique_solution_system (x y z : ℝ) :
  (x + y = 2 ∧ x * y - z^2 = 1) ↔ (x = 1 ∧ y = 1 ∧ z = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l259_25975


namespace NUMINAMATH_CALUDE_quadratic_min_max_l259_25945

/-- The quadratic function f(x) = 2x^2 - 8x + 3 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 3

theorem quadratic_min_max :
  (∀ x : ℝ, f x ≥ -5) ∧
  (f 2 = -5) ∧
  (∀ M : ℝ, ∃ x : ℝ, f x > M) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_min_max_l259_25945


namespace NUMINAMATH_CALUDE_fred_current_money_l259_25972

/-- Fred's money situation --/
def fred_money_problem (initial_amount earned_amount : ℕ) : Prop :=
  initial_amount + earned_amount = 40

/-- Theorem: Fred now has 40 dollars --/
theorem fred_current_money :
  fred_money_problem 19 21 :=
by
  sorry

end NUMINAMATH_CALUDE_fred_current_money_l259_25972


namespace NUMINAMATH_CALUDE_min_modulus_m_for_real_root_l259_25989

theorem min_modulus_m_for_real_root (m : ℂ) : 
  (∃ x : ℝ, (1 + 2*I)*x^2 + m*x + (1 - 2*I) = 0) → 
  ∀ m' : ℂ, (∃ x : ℝ, (1 + 2*I)*x^2 + m'*x + (1 - 2*I) = 0) → 
  Complex.abs m ≥ 2 ∧ (Complex.abs m = 2 → Complex.abs m' ≥ Complex.abs m) :=
by sorry

end NUMINAMATH_CALUDE_min_modulus_m_for_real_root_l259_25989


namespace NUMINAMATH_CALUDE_f_properties_l259_25940

noncomputable def f (x φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + φ) + Real.cos (2 * x + φ)

theorem f_properties (φ : ℝ) :
  (∀ x, f x φ = f (-x) φ) →  -- f is an even function
  (∀ x ∈ Set.Icc 0 (π / 4), ∀ y ∈ Set.Icc 0 (π / 4), x < y → f x φ < f y φ) →  -- f is increasing in [0, π/4]
  φ = 4 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_f_properties_l259_25940


namespace NUMINAMATH_CALUDE_max_sum_of_unknown_pairs_l259_25973

def pairwise_sums (a b c d : ℕ) : Finset ℕ :=
  {a + b, a + c, a + d, b + c, b + d, c + d}

theorem max_sum_of_unknown_pairs (a b c d : ℕ) :
  let sums := pairwise_sums a b c d
  ∀ x y, x ∈ sums → y ∈ sums →
    {210, 335, 296, 245, x, y} = sums →
    x + y ≤ 717 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_unknown_pairs_l259_25973


namespace NUMINAMATH_CALUDE_roller_coaster_tickets_l259_25988

/-- The number of friends going on the roller coaster ride -/
def num_friends : ℕ := 8

/-- The total number of tickets needed for all friends -/
def total_tickets : ℕ := 48

/-- The number of tickets required per ride -/
def tickets_per_ride : ℕ := total_tickets / num_friends

theorem roller_coaster_tickets : tickets_per_ride = 6 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_tickets_l259_25988


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l259_25933

theorem students_in_both_clubs 
  (total_students : ℕ) 
  (drama_students : ℕ) 
  (science_students : ℕ) 
  (students_in_either : ℕ) 
  (h1 : total_students = 320)
  (h2 : drama_students = 90)
  (h3 : science_students = 140)
  (h4 : students_in_either = 200) :
  drama_students + science_students - students_in_either = 30 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l259_25933


namespace NUMINAMATH_CALUDE_percent_of_y_l259_25963

theorem percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 10 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l259_25963


namespace NUMINAMATH_CALUDE_profit_ratio_calculation_l259_25982

theorem profit_ratio_calculation (p q : ℕ) (investment_ratio_p investment_ratio_q : ℕ) 
  (investment_duration_p investment_duration_q : ℕ) :
  investment_ratio_p = 7 →
  investment_ratio_q = 5 →
  investment_duration_p = 2 →
  investment_duration_q = 4 →
  (investment_ratio_p * investment_duration_p) / (investment_ratio_q * investment_duration_q) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_profit_ratio_calculation_l259_25982


namespace NUMINAMATH_CALUDE_floor_length_proof_l259_25990

/-- Represents a rectangular tile with length and width -/
structure Tile where
  length : ℕ
  width : ℕ

/-- Represents a rectangular floor with length and width -/
structure Floor where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit on a floor -/
def maxTiles (t : Tile) (f : Floor) : ℕ :=
  let tilesAcross := f.width / t.width
  let tilesDown := f.length / t.length
  tilesAcross * tilesDown

theorem floor_length_proof (t : Tile) (f : Floor) (h1 : t.length = 25) (h2 : t.width = 16) 
    (h3 : f.width = 120) (h4 : maxTiles t f = 54) : f.length = 175 := by
  sorry

end NUMINAMATH_CALUDE_floor_length_proof_l259_25990


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l259_25904

theorem complex_in_second_quadrant (m : ℝ) :
  let z : ℂ := (2 + m * I) / (4 - 5 * I)
  (z.re < 0 ∧ z.im > 0) ↔ m > 8/5 := by sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l259_25904


namespace NUMINAMATH_CALUDE_product_division_theorem_l259_25991

theorem product_division_theorem (x y : ℝ) (hx : x = 1.6666666666666667) (hx_nonzero : x ≠ 0) :
  Real.sqrt ((5 * x) / y) = x → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_division_theorem_l259_25991


namespace NUMINAMATH_CALUDE_servant_worked_nine_months_l259_25931

/-- Represents the salary and employment duration of a servant --/
structure ServantSalary where
  yearly_cash : ℕ  -- Yearly cash salary in Rupees
  turban_price : ℕ  -- Price of the turban in Rupees
  leaving_cash : ℕ  -- Cash received when leaving in Rupees
  months_worked : ℚ  -- Number of months worked

/-- Calculates the number of months a servant worked based on the given salary structure --/
def calculate_months_worked (s : ServantSalary) : ℚ :=
  let total_yearly_salary : ℚ := s.yearly_cash + s.turban_price
  let monthly_salary : ℚ := total_yearly_salary / 12
  let total_received : ℚ := s.leaving_cash + s.turban_price
  total_received / monthly_salary

/-- Theorem stating that the servant worked for approximately 9 months --/
theorem servant_worked_nine_months (s : ServantSalary) 
  (h1 : s.yearly_cash = 90)
  (h2 : s.turban_price = 70)
  (h3 : s.leaving_cash = 50) :
  ∃ ε > 0, |calculate_months_worked s - 9| < ε := by
  sorry

#eval calculate_months_worked { yearly_cash := 90, turban_price := 70, leaving_cash := 50, months_worked := 0 }

end NUMINAMATH_CALUDE_servant_worked_nine_months_l259_25931


namespace NUMINAMATH_CALUDE_equation_solution_l259_25979

theorem equation_solution : ∃ (x y : ℝ), x + y + x*y = 4 ∧ 3*x*y = 4 ∧ x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l259_25979


namespace NUMINAMATH_CALUDE_no_triangle_with_geometric_angles_l259_25902

theorem no_triangle_with_geometric_angles : ¬∃ (a r : ℕ), 
  a ≥ 10 ∧ 
  a < a * r ∧ 
  a * r < a * r * r ∧ 
  a + a * r + a * r * r = 180 := by
  sorry

end NUMINAMATH_CALUDE_no_triangle_with_geometric_angles_l259_25902


namespace NUMINAMATH_CALUDE_farm_animals_l259_25907

theorem farm_animals (goats chickens ducks pigs : ℕ) : 
  chickens = 2 * goats →
  ducks = (goats + chickens) / 2 →
  pigs = ducks / 3 →
  goats = pigs + 33 →
  goats = 66 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l259_25907


namespace NUMINAMATH_CALUDE_inequality_proof_l259_25922

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : a * b + b * c + c * a = a + b + c) : 
  a^2 + b^2 + c^2 + 2*a*b*c ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l259_25922


namespace NUMINAMATH_CALUDE_proportion_result_l259_25978

/-- Given a proportion x : 474 :: 537 : 26, prove that x rounded to the nearest integer is 9795 -/
theorem proportion_result : 
  ∃ x : ℝ, (x / 474 = 537 / 26) ∧ (round x = 9795) :=
by sorry

end NUMINAMATH_CALUDE_proportion_result_l259_25978


namespace NUMINAMATH_CALUDE_trapezoid_area_maximization_l259_25921

/-- Given a triangle ABC with sides a, b, c, altitude h, and a point G on the altitude
    at distance x from A, the area of the trapezoid formed by drawing a line parallel
    to the base through G and extending the sides is maximized when
    x = ((b + c) * h) / (2 * (a + b + c)). -/
theorem trapezoid_area_maximization (a b c h x : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ h > 0 ∧ x > 0 ∧ x < h →
  let t := (1/2) * (a + ((a + b + c) * x / h)) * (h - x)
  ∃ (max_x : ℝ), max_x = ((b + c) * h) / (2 * (a + b + c)) ∧
    ∀ y, 0 < y ∧ y < h → t ≤ (1/2) * (a + ((a + b + c) * max_x / h)) * (h - max_x) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_maximization_l259_25921


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l259_25995

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (1 - 2*I) / (1 + I^3)
  Complex.im z = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l259_25995


namespace NUMINAMATH_CALUDE_students_looking_up_fraction_l259_25966

theorem students_looking_up_fraction : 
  ∀ (total_students : ℕ) (eyes_saw_plane : ℕ) (eyes_per_student : ℕ),
    total_students = 200 →
    eyes_saw_plane = 300 →
    eyes_per_student = 2 →
    (eyes_saw_plane / eyes_per_student : ℚ) / total_students = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_students_looking_up_fraction_l259_25966


namespace NUMINAMATH_CALUDE_set_inclusion_equivalence_l259_25968

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 5/4}

def B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1 - 1| + 2*|p.2 - 2| ≤ a}

-- State the theorem
theorem set_inclusion_equivalence (a : ℝ) : A ⊆ B a ↔ a ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_equivalence_l259_25968


namespace NUMINAMATH_CALUDE_equation_c_is_quadratic_l259_25998

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation (4x-3)(3x+1)=0 -/
def f (x : ℝ) : ℝ := (4*x - 3) * (3*x + 1)

/-- Theorem: The equation (4x-3)(3x+1)=0 is a quadratic equation -/
theorem equation_c_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_c_is_quadratic_l259_25998


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l259_25918

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge := 1.1 * L
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 0.21 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l259_25918


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l259_25915

theorem quadratic_equation_two_distinct_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * x₁^2 - m * x₁ - 1 = 0 ∧ 2 * x₂^2 - m * x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l259_25915


namespace NUMINAMATH_CALUDE_distance_to_line_l259_25956

/-- Represents a square with side length 2 inches -/
structure Square where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- The configuration of three squares, where the middle one is rotated -/
structure SquareConfiguration where
  left : Square
  middle : Square
  right : Square
  middle_rotated : middle.side_length = left.side_length ∧ middle.side_length = right.side_length

/-- The theorem stating the distance of point B from the original line -/
theorem distance_to_line (config : SquareConfiguration) :
  let diagonal := config.middle.side_length * Real.sqrt 2
  let height_increase := diagonal / 2
  let original_height := config.middle.side_length / 2
  height_increase + original_height = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_line_l259_25956


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l259_25965

theorem arithmetic_mean_of_fractions : 
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = (67 / 144 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l259_25965


namespace NUMINAMATH_CALUDE_no_convex_polygon_partition_into_non_convex_quadrilaterals_l259_25913

/-- A polygon is a closed planar figure bounded by straight line segments. -/
structure Polygon where
  vertices : Set (ℝ × ℝ)
  is_closed : Bool
  is_planar : Bool

/-- A polygon is convex if all its interior angles are less than or equal to 180 degrees. -/
def is_convex (p : Polygon) : Prop :=
  sorry

/-- A quadrilateral is a polygon with exactly four sides. -/
def is_quadrilateral (p : Polygon) : Prop :=
  sorry

/-- A quadrilateral is non-convex if at least one of its interior angles is greater than 180 degrees. -/
def is_non_convex_quadrilateral (q : Polygon) : Prop :=
  is_quadrilateral q ∧ ¬(is_convex q)

/-- A partition of a polygon is a set of smaller polygons that completely cover the original polygon without overlapping. -/
def is_partition (p : Polygon) (parts : Set Polygon) : Prop :=
  sorry

/-- The main theorem: It is impossible to partition a convex polygon into non-convex quadrilaterals. -/
theorem no_convex_polygon_partition_into_non_convex_quadrilaterals :
  ∀ (p : Polygon) (parts : Set Polygon),
    is_convex p →
    is_partition p parts →
    (∀ q ∈ parts, is_non_convex_quadrilateral q) →
    False :=
  sorry

end NUMINAMATH_CALUDE_no_convex_polygon_partition_into_non_convex_quadrilaterals_l259_25913


namespace NUMINAMATH_CALUDE_range_of_a_l259_25934

theorem range_of_a (a : ℝ) : 
  let A := {x : ℝ | x > a}
  let B := {x : ℝ | x > 6}
  A ⊆ B ↔ a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l259_25934


namespace NUMINAMATH_CALUDE_squirrel_count_l259_25947

theorem squirrel_count (first_count : ℕ) (second_count : ℕ) : 
  first_count = 12 →
  second_count = first_count + first_count / 3 →
  first_count + second_count = 28 := by
sorry

end NUMINAMATH_CALUDE_squirrel_count_l259_25947


namespace NUMINAMATH_CALUDE_perimeter_of_picture_area_l259_25985

/-- Given a sheet of paper and a margin, calculate the perimeter of the remaining area --/
def perimeter_of_remaining_area (paper_width paper_length margin : ℕ) : ℕ :=
  2 * ((paper_width - 2 * margin) + (paper_length - 2 * margin))

/-- Theorem: The perimeter of the remaining area for a 12x16 inch paper with 2-inch margins is 40 inches --/
theorem perimeter_of_picture_area : perimeter_of_remaining_area 12 16 2 = 40 := by
  sorry

#eval perimeter_of_remaining_area 12 16 2

end NUMINAMATH_CALUDE_perimeter_of_picture_area_l259_25985


namespace NUMINAMATH_CALUDE_trig_sum_zero_l259_25970

theorem trig_sum_zero : Real.sin (0 * π / 180) + Real.cos (90 * π / 180) + Real.tan (180 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_zero_l259_25970


namespace NUMINAMATH_CALUDE_amendment_effects_l259_25958

-- Define the administrative actions included in the amendment
def administrative_actions : Set String := 
  {"abuse of administrative power", "illegal fundraising", "apportionment of expenses", "failure to pay benefits"}

-- Define the amendment to the Administrative Litigation Law
def administrative_litigation_amendment (actions : Set String) : Prop :=
  ∀ action ∈ actions, action ∈ administrative_actions

-- Define the concept of standardizing government power exercise
def standardizes_government_power (amendment : Set String → Prop) : Prop :=
  amendment administrative_actions → 
    ∃ standard : String, standard = "improved government power exercise"

-- Define the concept of protecting citizens' rights
def protects_citizens_rights (amendment : Set String → Prop) : Prop :=
  amendment administrative_actions → 
    ∃ protection : String, protection = "better protection of citizens' rights"

-- Theorem statement
theorem amendment_effects 
  (h : administrative_litigation_amendment administrative_actions) :
  standardizes_government_power administrative_litigation_amendment ∧ 
  protects_citizens_rights administrative_litigation_amendment :=
by sorry

end NUMINAMATH_CALUDE_amendment_effects_l259_25958


namespace NUMINAMATH_CALUDE_haley_marbles_division_l259_25932

/-- Given a number of marbles and a number of boys, calculate the number of marbles each boy receives when divided equally. -/
def marblesPerBoy (totalMarbles : ℕ) (numBoys : ℕ) : ℕ :=
  totalMarbles / numBoys

/-- Theorem stating that when 35 marbles are divided equally among 5 boys, each boy receives 7 marbles. -/
theorem haley_marbles_division :
  marblesPerBoy 35 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_haley_marbles_division_l259_25932


namespace NUMINAMATH_CALUDE_max_value_ad_minus_bc_l259_25939

theorem max_value_ad_minus_bc :
  ∃ (a b c d : ℤ),
    a ∈ ({-1, 1, 2} : Set ℤ) ∧
    b ∈ ({-1, 1, 2} : Set ℤ) ∧
    c ∈ ({-1, 1, 2} : Set ℤ) ∧
    d ∈ ({-1, 1, 2} : Set ℤ) ∧
    a * d - b * c = 6 ∧
    ∀ (x y z w : ℤ),
      x ∈ ({-1, 1, 2} : Set ℤ) →
      y ∈ ({-1, 1, 2} : Set ℤ) →
      z ∈ ({-1, 1, 2} : Set ℤ) →
      w ∈ ({-1, 1, 2} : Set ℤ) →
      x * w - y * z ≤ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_ad_minus_bc_l259_25939


namespace NUMINAMATH_CALUDE_tan_150_degrees_l259_25955

theorem tan_150_degrees : Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l259_25955


namespace NUMINAMATH_CALUDE_intersection_A_B_l259_25920

def A : Set ℝ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l259_25920


namespace NUMINAMATH_CALUDE_percentage_both_correct_l259_25923

theorem percentage_both_correct (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) :
  p_first = 0.63 →
  p_second = 0.49 →
  p_neither = 0.20 →
  p_first + p_second - (1 - p_neither) = 0.32 := by
sorry

end NUMINAMATH_CALUDE_percentage_both_correct_l259_25923


namespace NUMINAMATH_CALUDE_apple_purchase_difference_l259_25908

theorem apple_purchase_difference : 
  ∀ (bonnie_apples samuel_apples : ℕ),
    bonnie_apples = 8 →
    samuel_apples > bonnie_apples →
    samuel_apples - (samuel_apples / 2) - (samuel_apples / 7) = 10 →
    samuel_apples - bonnie_apples = 20 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_difference_l259_25908


namespace NUMINAMATH_CALUDE_second_half_speed_l259_25930

/-- Given a trip with the following properties:
  * Total distance is 60 km
  * First half of the trip (30 km) is traveled at 48 km/h
  * Average speed of the entire trip is 32 km/h
  Then the speed of the second half of the trip is 24 km/h -/
theorem second_half_speed (total_distance : ℝ) (first_half_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 60 →
  first_half_distance = 30 →
  first_half_speed = 48 →
  average_speed = 32 →
  let second_half_distance := total_distance - first_half_distance
  let total_time := total_distance / average_speed
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := total_time - first_half_time
  let second_half_speed := second_half_distance / second_half_time
  second_half_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_second_half_speed_l259_25930


namespace NUMINAMATH_CALUDE_correct_sampling_pairing_l259_25935

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

-- Define the sampling scenarios
structure SamplingScenario where
  description : String
  populationSize : Nat
  sampleSize : Nat
  hasStrata : Bool

-- Define the correct pairing function
def correctPairing (scenario : SamplingScenario) : SamplingMethod :=
  if scenario.hasStrata then
    SamplingMethod.Stratified
  else if scenario.populationSize ≤ 100 then
    SamplingMethod.SimpleRandom
  else
    SamplingMethod.Systematic

-- Define the three scenarios
def universityScenario : SamplingScenario :=
  { description := "University student sampling"
  , populationSize := 300
  , sampleSize := 100
  , hasStrata := true }

def productScenario : SamplingScenario :=
  { description := "Product quality inspection"
  , populationSize := 20
  , sampleSize := 7
  , hasStrata := false }

def habitScenario : SamplingScenario :=
  { description := "Daily habits sampling"
  , populationSize := 2000
  , sampleSize := 10
  , hasStrata := false }

-- Theorem statement
theorem correct_sampling_pairing :
  (correctPairing universityScenario = SamplingMethod.Stratified) ∧
  (correctPairing productScenario = SamplingMethod.SimpleRandom) ∧
  (correctPairing habitScenario = SamplingMethod.Systematic) :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_pairing_l259_25935


namespace NUMINAMATH_CALUDE_map_scale_conversion_l259_25997

/-- Given a scale where 1 inch represents 500 feet, and a path measuring 6.5 inches on a map,
    the actual length of the path in feet is 3250. -/
theorem map_scale_conversion (scale : ℝ) (map_length : ℝ) (actual_length : ℝ) : 
  scale = 500 → map_length = 6.5 → actual_length = scale * map_length → actual_length = 3250 :=
by sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l259_25997


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l259_25954

theorem inequality_and_equality_condition (a b : ℝ) :
  a^2 + b^2 + 1 ≥ a + b + a*b ∧
  (a^2 + b^2 + 1 = a + b + a*b ↔ a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l259_25954


namespace NUMINAMATH_CALUDE_vector_subtraction_magnitude_l259_25949

theorem vector_subtraction_magnitude : ∃ (a b : ℝ × ℝ), 
  a = (2, 1) ∧ b = (-2, 4) ∧ 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_magnitude_l259_25949


namespace NUMINAMATH_CALUDE_range_of_m_l259_25964

-- Define P and q as functions of x and m
def P (x : ℝ) : Prop := |4 - x| / 3 ≤ 2

def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(P x) → ¬(q x m)) →
  (∃ x, P x ∧ ¬(q x m)) →
  m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l259_25964


namespace NUMINAMATH_CALUDE_stating_max_principals_in_period_l259_25987

/-- Represents the duration of the entire period in years -/
def total_period : ℕ := 10

/-- Represents the duration of each principal's term in years -/
def term_length : ℕ := 4

/-- Represents the maximum number of principals that can serve during the total period -/
def max_principals : ℕ := 3

/-- 
Theorem stating that given a total period of 10 years and principals serving 
exactly one 4-year term each, the maximum number of principals that can serve 
during this period is 3.
-/
theorem max_principals_in_period : 
  ∀ (num_principals : ℕ), 
  (num_principals * term_length ≥ total_period) → 
  (num_principals ≤ max_principals) :=
by sorry

end NUMINAMATH_CALUDE_stating_max_principals_in_period_l259_25987


namespace NUMINAMATH_CALUDE_oliver_ferris_wheel_rides_l259_25917

/-- The number of times Oliver rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride (ferris wheel or bumper car) -/
def ticket_cost : ℕ := 3

/-- The total number of tickets Oliver used -/
def total_tickets : ℕ := 30

/-- The number of times Oliver rode the ferris wheel -/
def ferris_wheel_rides : ℕ := (total_tickets - bumper_rides * ticket_cost) / ticket_cost

theorem oliver_ferris_wheel_rides :
  ferris_wheel_rides = 7 := by sorry

end NUMINAMATH_CALUDE_oliver_ferris_wheel_rides_l259_25917


namespace NUMINAMATH_CALUDE_exists_h_for_phi_l259_25946

-- Define the types for our functions
def φ : ℝ → ℝ → ℝ → ℝ := sorry
def f : ℝ → ℝ → ℝ := sorry
def g : ℝ → ℝ → ℝ := sorry

-- State the theorem
theorem exists_h_for_phi (hf : ∀ x y z, φ x y z = f (x + y) z)
                         (hg : ∀ x y z, φ x y z = g x (y + z)) :
  ∃ h : ℝ → ℝ, ∀ x y z, φ x y z = h (x + y + z) := by sorry

end NUMINAMATH_CALUDE_exists_h_for_phi_l259_25946


namespace NUMINAMATH_CALUDE_ellipse_m_value_l259_25992

/-- An ellipse with equation x^2 + my^2 = 1, where m is a positive real number -/
structure Ellipse (m : ℝ) where
  equation : ∀ x y : ℝ, x^2 + m*y^2 = 1

/-- The foci of the ellipse are on the x-axis -/
def foci_on_x_axis (e : Ellipse m) : Prop :=
  ∃ c : ℝ, c^2 = 1 - 1/m

/-- The length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (e : Ellipse m) : Prop :=
  1 = 2 * (1/m).sqrt

/-- Theorem: For an ellipse with equation x^2 + my^2 = 1, where the foci are on the x-axis
    and the length of the major axis is twice the length of the minor axis, m = 4 -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m) 
    (h1 : foci_on_x_axis e) (h2 : major_axis_twice_minor e) : m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l259_25992


namespace NUMINAMATH_CALUDE_expression_value_l259_25905

theorem expression_value (x : ℝ) (h : x^2 + 2*x = 2) :
  (x - 1)^2 + (x + 3)*(x - 3) - (x - 3)*(x - 1) = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l259_25905
