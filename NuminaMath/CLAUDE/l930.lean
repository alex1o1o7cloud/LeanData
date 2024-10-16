import Mathlib

namespace NUMINAMATH_CALUDE_sector_area_l930_93016

theorem sector_area (arc_length : Real) (central_angle : Real) (sector_area : Real) : 
  arc_length = π ∧ 
  central_angle = π / 4 →
  sector_area = 8 * π := by
sorry

end NUMINAMATH_CALUDE_sector_area_l930_93016


namespace NUMINAMATH_CALUDE_vector_addition_scalar_mult_l930_93092

/-- Given plane vectors a and b, prove that 3a + b equals (-2, 6) -/
theorem vector_addition_scalar_mult 
  (a b : ℝ × ℝ) 
  (ha : a = (-1, 2)) 
  (hb : b = (1, 0)) : 
  (3 : ℝ) • a + b = (-2, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_scalar_mult_l930_93092


namespace NUMINAMATH_CALUDE_cherry_pie_degrees_l930_93001

theorem cherry_pie_degrees (total : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ) 
  (h1 : total = 40)
  (h2 : chocolate = 15)
  (h3 : apple = 10)
  (h4 : blueberry = 7)
  (h5 : (total - (chocolate + apple + blueberry)) % 2 = 0) :
  let remaining := total - (chocolate + apple + blueberry)
  let cherry := remaining / 2
  (cherry : ℚ) / total * 360 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_degrees_l930_93001


namespace NUMINAMATH_CALUDE_number_1349_is_valid_l930_93033

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 100 % 10 = 3 * (n / 1000)) ∧
  (n % 10 = 3 * (n / 100 % 10))

theorem number_1349_is_valid : is_valid_number 1349 := by
  sorry

end NUMINAMATH_CALUDE_number_1349_is_valid_l930_93033


namespace NUMINAMATH_CALUDE_incircle_and_inscribed_circles_inequality_l930_93082

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the theorem
theorem incircle_and_inscribed_circles_inequality 
  (triangle : Triangle) 
  (incircle : Circle) 
  (inscribed_circle1 inscribed_circle2 inscribed_circle3 : Circle) :
  -- Conditions
  (incircle.radius > 0) →
  (inscribed_circle1.radius > 0) →
  (inscribed_circle2.radius > 0) →
  (inscribed_circle3.radius > 0) →
  (inscribed_circle1.radius < incircle.radius) →
  (inscribed_circle2.radius < incircle.radius) →
  (inscribed_circle3.radius < incircle.radius) →
  -- Theorem statement
  inscribed_circle1.radius + inscribed_circle2.radius + inscribed_circle3.radius ≥ incircle.radius :=
by
  sorry

end NUMINAMATH_CALUDE_incircle_and_inscribed_circles_inequality_l930_93082


namespace NUMINAMATH_CALUDE_divisibility_implies_equation_existence_l930_93048

theorem divisibility_implies_equation_existence (p x y : ℕ) (hp : Prime p) 
  (hp_form : ∃ k : ℕ, p = 4 * k + 3) (hx : x > 0) (hy : y > 0)
  (hdiv : p ∣ (x^2 - x*y + ((p+1)/4) * y^2)) :
  ∃ u v : ℤ, x^2 - x*y + ((p+1)/4) * y^2 = p * (u^2 - u*v + ((p+1)/4) * v^2) := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equation_existence_l930_93048


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l930_93090

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (x - 3) + Real.sqrt (x - 8) = 10 → x = 30.5625 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l930_93090


namespace NUMINAMATH_CALUDE_berkeley_class_as_l930_93011

theorem berkeley_class_as (abraham_total : ℕ) (abraham_as : ℕ) (berkeley_total : ℕ) :
  abraham_total = 20 →
  abraham_as = 12 →
  berkeley_total = 30 →
  (berkeley_total : ℚ) * (abraham_as : ℚ) / (abraham_total : ℚ) = 18 :=
by sorry

end NUMINAMATH_CALUDE_berkeley_class_as_l930_93011


namespace NUMINAMATH_CALUDE_range_of_f_l930_93004

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 7

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -2} :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l930_93004


namespace NUMINAMATH_CALUDE_small_parallelogram_area_l930_93084

/-- Given a parallelogram ABCD with area 1, where sides AB and CD are divided into n equal parts,
    and sides AD and BC are divided into m equal parts, the area of each smaller parallelogram
    formed by connecting the division points is 1 / (mn - 1). -/
theorem small_parallelogram_area (n m : ℕ) (h1 : n > 0) (h2 : m > 0) :
  let total_area : ℝ := 1
  let num_small_parallelograms : ℕ := n * m - 1
  let small_parallelogram_area : ℝ := total_area / num_small_parallelograms
  small_parallelogram_area = 1 / (n * m - 1) := by
  sorry

end NUMINAMATH_CALUDE_small_parallelogram_area_l930_93084


namespace NUMINAMATH_CALUDE_solution_set_l930_93096

theorem solution_set (x : ℝ) : 
  (1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < 1/4 ∧ x - 2 > 0 → x > 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l930_93096


namespace NUMINAMATH_CALUDE_beverly_bottle_caps_l930_93086

/-- The number of groups of bottle caps in Beverly's collection -/
def num_groups : ℕ := 7

/-- The number of bottle caps in each group -/
def caps_per_group : ℕ := 5

/-- The total number of bottle caps in Beverly's collection -/
def total_caps : ℕ := num_groups * caps_per_group

theorem beverly_bottle_caps : total_caps = 35 := by
  sorry

end NUMINAMATH_CALUDE_beverly_bottle_caps_l930_93086


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l930_93025

/-- Represents a triangle in the sequence -/
structure Triangle where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { a := t.a / 2 - 1,
    b := t.b / 2,
    c := t.c / 2 + 1 }

/-- Checks if a triangle is valid (satisfies triangle inequality) -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The initial triangle T₁ -/
def T₁ : Triangle :=
  { a := 1009, b := 1010, c := 1011 }

/-- Generates the sequence of triangles -/
def triangleSequence : ℕ → Triangle
  | 0 => T₁
  | n + 1 => nextTriangle (triangleSequence n)

/-- Finds the index of the last valid triangle in the sequence -/
def lastValidTriangleIndex : ℕ := sorry

/-- The last valid triangle in the sequence -/
def lastValidTriangle : Triangle :=
  triangleSequence lastValidTriangleIndex

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℚ :=
  t.a + t.b + t.c

theorem last_triangle_perimeter :
  perimeter lastValidTriangle = 71 / 8 := by sorry

end NUMINAMATH_CALUDE_last_triangle_perimeter_l930_93025


namespace NUMINAMATH_CALUDE_toy_store_shelves_l930_93087

/-- Calculates the number of shelves needed for a given number of items and shelf capacity -/
def shelves_needed (items : ℕ) (capacity : ℕ) : ℕ :=
  (items + capacity - 1) / capacity

/-- Proves that the total number of shelves needed for bears and rabbits is 6 -/
theorem toy_store_shelves : 
  let initial_bears : ℕ := 17
  let initial_rabbits : ℕ := 20
  let new_bears : ℕ := 10
  let new_rabbits : ℕ := 15
  let sold_bears : ℕ := 5
  let sold_rabbits : ℕ := 7
  let bear_shelf_capacity : ℕ := 9
  let rabbit_shelf_capacity : ℕ := 12
  let remaining_bears : ℕ := initial_bears + new_bears - sold_bears
  let remaining_rabbits : ℕ := initial_rabbits + new_rabbits - sold_rabbits
  let bear_shelves : ℕ := shelves_needed remaining_bears bear_shelf_capacity
  let rabbit_shelves : ℕ := shelves_needed remaining_rabbits rabbit_shelf_capacity
  bear_shelves + rabbit_shelves = 6 :=
by sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l930_93087


namespace NUMINAMATH_CALUDE_trig_simplification_l930_93019

theorem trig_simplification :
  (Real.sin (35 * π / 180))^2 / Real.sin (20 * π / 180) - 1 / (2 * Real.sin (20 * π / 180)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l930_93019


namespace NUMINAMATH_CALUDE_driver_net_rate_of_pay_l930_93054

/-- Calculates the net rate of pay for a driver given specific conditions --/
theorem driver_net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_rate = 0.60)
  (h5 : gas_price = 2.50)
  : (pay_rate * speed * travel_time - (speed * travel_time / fuel_efficiency) * gas_price) / travel_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_rate_of_pay_l930_93054


namespace NUMINAMATH_CALUDE_experiment_A_not_control_based_l930_93069

-- Define the type for experiments
inductive Experiment
| A
| B
| C
| D

-- Define a predicate for experiments designed based on the principle of control
def is_control_based (e : Experiment) : Prop :=
  match e with
  | Experiment.A => False
  | _ => True

-- Theorem statement
theorem experiment_A_not_control_based :
  is_control_based Experiment.B ∧
  is_control_based Experiment.C ∧
  is_control_based Experiment.D →
  ¬is_control_based Experiment.A :=
by
  sorry

end NUMINAMATH_CALUDE_experiment_A_not_control_based_l930_93069


namespace NUMINAMATH_CALUDE_crab_fishing_income_l930_93057

/-- Calculate weekly income from crab fishing --/
theorem crab_fishing_income 
  (num_buckets : ℕ) 
  (crabs_per_bucket : ℕ) 
  (price_per_crab : ℕ) 
  (days_per_week : ℕ) 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  num_buckets * crabs_per_bucket * price_per_crab * days_per_week = 3360 := by
  sorry

end NUMINAMATH_CALUDE_crab_fishing_income_l930_93057


namespace NUMINAMATH_CALUDE_solution_set_and_inequality_l930_93074

def f (x : ℝ) := -x + |2*x + 1|

def M : Set ℝ := {x | f x < 2}

theorem solution_set_and_inequality :
  (M = {x : ℝ | -1 < x ∧ x < 1}) ∧
  (∀ a b : ℝ, a ∈ M → b ∈ M → 2 * |a * b| + 1 > |a| + |b|) := by sorry

end NUMINAMATH_CALUDE_solution_set_and_inequality_l930_93074


namespace NUMINAMATH_CALUDE_max_white_pieces_correct_l930_93014

/-- Represents a game board with m rows and n columns -/
structure Board (m n : ℕ) where
  white_pieces : Finset (ℕ × ℕ)
  no_same_row_col : ∀ (i j k l : ℕ), (i, j) ∈ white_pieces → (k, l) ∈ white_pieces → i = k ∨ j = l → (i, j) = (k, l)

/-- The maximum number of white pieces that can be placed on the board -/
def max_white_pieces (m n : ℕ) : ℕ := m + n - 1

/-- Theorem stating that the maximum number of white pieces is m + n - 1 -/
theorem max_white_pieces_correct (m n : ℕ) :
  ∀ (b : Board m n), b.white_pieces.card ≤ max_white_pieces m n :=
by sorry

end NUMINAMATH_CALUDE_max_white_pieces_correct_l930_93014


namespace NUMINAMATH_CALUDE_area_enclosed_by_midpoints_l930_93070

/-- The area enclosed by midpoints of line segments with length 3 and endpoints on adjacent sides of a square with side length 3 -/
theorem area_enclosed_by_midpoints (square_side : ℝ) (segment_length : ℝ) : square_side = 3 → segment_length = 3 → 
  ∃ (area : ℝ), area = 9 - (9 * Real.pi / 16) := by
  sorry

end NUMINAMATH_CALUDE_area_enclosed_by_midpoints_l930_93070


namespace NUMINAMATH_CALUDE_contradiction_assumption_l930_93031

theorem contradiction_assumption (x y : ℝ) (h : x + y > 2) :
  ¬(x ≤ 1 ∧ y ≤ 1) → (x > 1 ∨ y > 1) := by
  sorry

#check contradiction_assumption

end NUMINAMATH_CALUDE_contradiction_assumption_l930_93031


namespace NUMINAMATH_CALUDE_megan_initial_markers_l930_93006

/-- The number of markers Megan initially had -/
def initial_markers : ℕ := sorry

/-- The number of markers Robert gave to Megan -/
def roberts_markers : ℕ := 109

/-- The total number of markers Megan has after receiving markers from Robert -/
def total_markers : ℕ := 326

/-- Theorem stating that the initial number of markers Megan had is 217 -/
theorem megan_initial_markers : initial_markers = 217 := by
  sorry

end NUMINAMATH_CALUDE_megan_initial_markers_l930_93006


namespace NUMINAMATH_CALUDE_race_time_proof_l930_93056

/-- 
Proves that in a 1000-meter race where runner A finishes 200 meters ahead of runner B, 
and the time difference between their finishes is 10 seconds, 
the time taken by runner A to complete the race is 50 seconds.
-/
theorem race_time_proof (race_length : ℝ) (distance_diff : ℝ) (time_diff : ℝ) 
  (h1 : race_length = 1000)
  (h2 : distance_diff = 200)
  (h3 : time_diff = 10) : 
  ∃ (time_A : ℝ), time_A = 50 ∧ 
    race_length / time_A = (race_length - distance_diff) / (time_A + time_diff) :=
by
  sorry

#check race_time_proof

end NUMINAMATH_CALUDE_race_time_proof_l930_93056


namespace NUMINAMATH_CALUDE_dot_product_sum_l930_93062

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -2)

theorem dot_product_sum : 
  (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) = -1 := by sorry

end NUMINAMATH_CALUDE_dot_product_sum_l930_93062


namespace NUMINAMATH_CALUDE_bicycle_distance_l930_93009

/-- Proves that a bicycle traveling 1/2 as fast as a motorcycle moving at 40 miles per hour
    will cover a distance of 10 miles in 30 minutes. -/
theorem bicycle_distance (motorcycle_speed : ℝ) (bicycle_speed_ratio : ℝ) (time : ℝ) :
  motorcycle_speed = 40 →
  bicycle_speed_ratio = (1 : ℝ) / 2 →
  time = 30 / 60 →
  (bicycle_speed_ratio * motorcycle_speed) * time = 10 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_distance_l930_93009


namespace NUMINAMATH_CALUDE_unique_b_c_solution_l930_93013

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

-- State the theorem
theorem unique_b_c_solution :
  ∃! (b c : ℝ), 
    (∃ a : ℝ, A a ≠ B b c) ∧ 
    (∃ a : ℝ, A a ∪ B b c = {-3, 4}) ∧
    (∃ a : ℝ, A a ∩ B b c = {-3}) ∧
    b = 6 ∧ c = 9 := by
  sorry


end NUMINAMATH_CALUDE_unique_b_c_solution_l930_93013


namespace NUMINAMATH_CALUDE_muffin_profit_l930_93044

/-- Bob's muffin business profit calculation -/
theorem muffin_profit : 
  ∀ (muffins_per_day : ℕ) 
    (cost_price selling_price : ℚ) 
    (days_in_week : ℕ),
  muffins_per_day = 12 →
  cost_price = 3/4 →
  selling_price = 3/2 →
  days_in_week = 7 →
  (selling_price - cost_price) * muffins_per_day * days_in_week = 63 := by
  sorry


end NUMINAMATH_CALUDE_muffin_profit_l930_93044


namespace NUMINAMATH_CALUDE_diane_has_27_cents_l930_93094

/-- The amount of money Diane has, given the cost of cookies and the additional amount needed. -/
def dianes_money (cookie_cost : ℕ) (additional_needed : ℕ) : ℕ :=
  cookie_cost - additional_needed

/-- Theorem stating that Diane has 27 cents given the problem conditions. -/
theorem diane_has_27_cents :
  dianes_money 65 38 = 27 := by
  sorry

end NUMINAMATH_CALUDE_diane_has_27_cents_l930_93094


namespace NUMINAMATH_CALUDE_no_k_satisfies_condition_l930_93068

-- Define a function to get the nth odd prime number
def nthOddPrime (n : ℕ) : ℕ := sorry

-- Define a function to calculate the product of the first k odd primes
def productOfFirstKOddPrimes (k : ℕ) : ℕ := sorry

-- Define a function to check if a number is a perfect power greater than 1
def isPerfectPowerGreaterThanOne (n : ℕ) : Prop := sorry

-- Theorem statement
theorem no_k_satisfies_condition :
  ∀ k : ℕ, k > 0 → ¬(isPerfectPowerGreaterThanOne (productOfFirstKOddPrimes k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_k_satisfies_condition_l930_93068


namespace NUMINAMATH_CALUDE_arithmetic_computation_l930_93066

theorem arithmetic_computation : (-12 * 6) - (-4 * -8) + (-15 * -3) - (36 / -2) = -77 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l930_93066


namespace NUMINAMATH_CALUDE_specific_polygon_area_l930_93000

/-- A polygon on a grid where dots are spaced one unit apart both horizontally and vertically -/
structure GridPolygon where
  vertices : List (ℤ × ℤ)

/-- Calculate the area of a GridPolygon -/
def area (p : GridPolygon) : ℚ :=
  sorry

/-- The specific polygon described in the problem -/
def specificPolygon : GridPolygon :=
  { vertices := [
    (0, 0), (10, 0), (20, 10), (30, 0), (40, 0),
    (30, 10), (30, 20), (20, 30), (20, 40), (10, 40),
    (0, 40), (0, 10)
  ] }

/-- Theorem stating that the area of the specific polygon is 31.5 square units -/
theorem specific_polygon_area :
  area specificPolygon = 31.5 := by sorry

end NUMINAMATH_CALUDE_specific_polygon_area_l930_93000


namespace NUMINAMATH_CALUDE_movie_profit_calculation_l930_93040

def actor_cost : ℕ := 1200
def num_people : ℕ := 50
def food_cost_per_person : ℕ := 3
def movie_selling_price : ℕ := 10000

def total_food_cost : ℕ := num_people * food_cost_per_person
def actors_and_food_cost : ℕ := actor_cost + total_food_cost
def equipment_rental_cost : ℕ := 2 * actors_and_food_cost
def total_cost : ℕ := actors_and_food_cost + equipment_rental_cost

theorem movie_profit_calculation :
  movie_selling_price - total_cost = 5950 := by
  sorry

end NUMINAMATH_CALUDE_movie_profit_calculation_l930_93040


namespace NUMINAMATH_CALUDE_recipe_flour_cups_l930_93078

/-- The number of cups of sugar required in the recipe -/
def sugar_cups : ℕ := 9

/-- The number of cups of flour Mary has already put in -/
def flour_cups_added : ℕ := 4

/-- The total number of cups of flour required in the recipe -/
def total_flour_cups : ℕ := sugar_cups + 1

theorem recipe_flour_cups : total_flour_cups = 10 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_cups_l930_93078


namespace NUMINAMATH_CALUDE_book_purchase_solution_l930_93097

/-- Represents the cost and purchase details of two types of books -/
structure BookPurchase where
  costA : ℕ  -- Cost of book A
  costB : ℕ  -- Cost of book B
  totalBooks : ℕ  -- Total number of books to purchase
  maxCost : ℕ  -- Maximum total cost

/-- Defines the conditions of the book purchase problem -/
def validBookPurchase (bp : BookPurchase) : Prop :=
  bp.costB = bp.costA + 20 ∧  -- Condition 1
  540 / bp.costA = 780 / bp.costB ∧  -- Condition 2
  bp.totalBooks = 70 ∧  -- Condition 3
  bp.maxCost = 3550  -- Condition 4

/-- Theorem stating the solution to the book purchase problem -/
theorem book_purchase_solution (bp : BookPurchase) 
  (h : validBookPurchase bp) : 
  bp.costA = 45 ∧ bp.costB = 65 ∧ 
  (∀ m : ℕ, m * bp.costA + (bp.totalBooks - m) * bp.costB ≤ bp.maxCost → m ≥ 50) :=
sorry

end NUMINAMATH_CALUDE_book_purchase_solution_l930_93097


namespace NUMINAMATH_CALUDE_work_completion_time_l930_93058

theorem work_completion_time (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 20)
  (hy : y_days = 16)
  (hw : y_worked_days = 12) : 
  (x_days : ℚ) * (1 - y_worked_days / y_days) = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l930_93058


namespace NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_roots_l930_93041

theorem unique_magnitude_of_quadratic_roots (w : ℂ) : 
  w^2 - 6*w + 40 = 0 → ∃! x : ℝ, ∃ w : ℂ, w^2 - 6*w + 40 = 0 ∧ Complex.abs w = x :=
sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_roots_l930_93041


namespace NUMINAMATH_CALUDE_inequality_proof_l930_93042

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < d) :
  a - c > b - d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l930_93042


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l930_93007

theorem algebraic_expression_equality (y : ℝ) : 
  2 * y^2 + 3 * y + 7 = 8 → 4 * y^2 + 6 * y - 9 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l930_93007


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l930_93036

/-- A quadratic function y = kx^2 - 7x - 7 intersects the x-axis if and only if k ≥ -7/4 and k ≠ 0 -/
theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ (k ≥ -7/4 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l930_93036


namespace NUMINAMATH_CALUDE_fifth_power_complex_equality_l930_93020

theorem fifth_power_complex_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + b * Complex.I) ^ 5 = (a - b * Complex.I) ^ 5) : 
  b / a = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_complex_equality_l930_93020


namespace NUMINAMATH_CALUDE_fourth_square_area_l930_93008

theorem fourth_square_area (PQ PR PS QR RS : ℝ) : 
  PQ^2 = 25 → QR^2 = 64 → RS^2 = 49 → 
  (PQ^2 + QR^2 = PR^2) → (PR^2 + RS^2 = PS^2) → 
  PS^2 = 138 := by sorry

end NUMINAMATH_CALUDE_fourth_square_area_l930_93008


namespace NUMINAMATH_CALUDE_additional_wax_is_22_l930_93091

/-- The amount of additional wax needed for painting feathers -/
def additional_wax_needed (total_wax : ℕ) (available_wax : ℕ) : ℕ :=
  total_wax - available_wax

/-- Theorem stating that the additional wax needed is 22 grams -/
theorem additional_wax_is_22 :
  additional_wax_needed 353 331 = 22 := by
  sorry

end NUMINAMATH_CALUDE_additional_wax_is_22_l930_93091


namespace NUMINAMATH_CALUDE_relationship_abc_l930_93053

theorem relationship_abc (a b c : ℝ) 
  (eq1 : b + c = 6 - 4*a + 3*a^2)
  (eq2 : c - b = 4 - 4*a + a^2) : 
  a < b ∧ b ≤ c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l930_93053


namespace NUMINAMATH_CALUDE_phone_answer_probability_l930_93080

theorem phone_answer_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 1/10)
  (h2 : p2 = 1/5)
  (h3 : p3 = 3/10)
  (h4 : p4 = 1/10) :
  p1 + p2 + p3 + p4 = 7/10 := by
  sorry

#check phone_answer_probability

end NUMINAMATH_CALUDE_phone_answer_probability_l930_93080


namespace NUMINAMATH_CALUDE_base_four_of_156_l930_93073

def base_four_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_four_of_156 :
  base_four_representation 156 = [2, 1, 3, 0] := by sorry

end NUMINAMATH_CALUDE_base_four_of_156_l930_93073


namespace NUMINAMATH_CALUDE_grandmas_red_bacon_bits_l930_93003

/-- Calculates the number of red bacon bits on Grandma's salad --/
def red_bacon_bits (mushrooms : ℕ) : ℕ :=
  let cherry_tomatoes := 2 * mushrooms
  let pickles := 4 * cherry_tomatoes
  let bacon_bits := 4 * pickles
  bacon_bits / 3

/-- Theorem stating that the number of red bacon bits on Grandma's salad is 32 --/
theorem grandmas_red_bacon_bits :
  red_bacon_bits 3 = 32 := by
  sorry

#eval red_bacon_bits 3

end NUMINAMATH_CALUDE_grandmas_red_bacon_bits_l930_93003


namespace NUMINAMATH_CALUDE_sphere_surface_area_l930_93010

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  A = 4 * Real.pi * r^2 → 
  A = 36 * Real.pi * (Real.rpow 2 (1/3))^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l930_93010


namespace NUMINAMATH_CALUDE_solution_set_theorem_l930_93012

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def f_def (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^2 - 2*x

theorem solution_set_theorem (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_def : f_def f) : 
  {x : ℝ | f (x + 1) < 3} = Set.Ioo (-4 : ℝ) 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l930_93012


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l930_93027

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_problem : 
  Nat.gcd (factorial 7) ((factorial 10) / (factorial 4)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l930_93027


namespace NUMINAMATH_CALUDE_prob_at_least_one_red_l930_93085

/-- Represents a box containing red and white balls -/
structure Box where
  red_balls : ℕ
  white_balls : ℕ

/-- Calculates the probability of drawing a specific color ball from a box -/
def prob_draw (b : Box) (color : String) : ℚ :=
  if color = "red" then
    b.red_balls / (b.red_balls + b.white_balls)
  else if color = "white" then
    b.white_balls / (b.red_balls + b.white_balls)
  else
    0

/-- Theorem: The probability of drawing at least one red ball from two boxes,
    each containing 2 red balls and 1 white ball, is equal to 8/9 -/
theorem prob_at_least_one_red (box_a box_b : Box) 
  (ha : box_a.red_balls = 2 ∧ box_a.white_balls = 1)
  (hb : box_b.red_balls = 2 ∧ box_b.white_balls = 1) : 
  1 - (prob_draw box_a "white" * prob_draw box_b "white") = 8/9 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_red_l930_93085


namespace NUMINAMATH_CALUDE_binary_sum_equals_116_l930_93077

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010101₂ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The binary representation of 11111₂ -/
def binary2 : List Bool := [true, true, true, true, true]

/-- Theorem stating that the sum of 1010101₂ and 11111₂ in decimal is 116 -/
theorem binary_sum_equals_116 : 
  binary_to_decimal binary1 + binary_to_decimal binary2 = 116 := by
  sorry


end NUMINAMATH_CALUDE_binary_sum_equals_116_l930_93077


namespace NUMINAMATH_CALUDE_binomial_coefficient_x4_in_x_plus_1_to_10_l930_93043

theorem binomial_coefficient_x4_in_x_plus_1_to_10 :
  (Finset.range 11).sum (fun k => (Nat.choose 10 k) * (1^(10 - k)) * (1^k)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x4_in_x_plus_1_to_10_l930_93043


namespace NUMINAMATH_CALUDE_odd_factors_of_x_squared_plus_one_l930_93072

theorem odd_factors_of_x_squared_plus_one (x : ℤ) (d : ℤ) :
  d > 0 → Odd d → (x^2 + 1) % d = 0 → ∃ h : ℤ, d = 4*h + 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_factors_of_x_squared_plus_one_l930_93072


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l930_93017

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  num_tiles : ℕ
  perimeter : ℕ

/-- Represents the result of adding tiles to a configuration -/
structure AddedTilesResult where
  new_perimeter : ℕ

/-- Function to add tiles to a configuration -/
def add_tiles (initial : TileConfiguration) (tiles_to_add : ℕ) : AddedTilesResult :=
  sorry

theorem perimeter_after_adding_tiles 
  (initial : TileConfiguration) 
  (tiles_to_add : ℕ) :
  initial.num_tiles = 10 →
  initial.perimeter = 14 →
  tiles_to_add = 3 →
  (add_tiles initial tiles_to_add).new_perimeter = 18 :=
sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l930_93017


namespace NUMINAMATH_CALUDE_cosine_sine_equation_l930_93083

theorem cosine_sine_equation (x : ℝ) :
  2 * Real.cos x - 3 * Real.sin x = 4 →
  3 * Real.sin x + 2 * Real.cos x = 0 ∨ 3 * Real.sin x + 2 * Real.cos x = 8/13 :=
by sorry

end NUMINAMATH_CALUDE_cosine_sine_equation_l930_93083


namespace NUMINAMATH_CALUDE_total_cement_is_54_4_l930_93061

/-- Amount of cement used for Lexi's street in tons -/
def lexis_cement : ℝ := 10

/-- Amount of cement used for Tess's street in tons -/
def tess_cement : ℝ := lexis_cement * 1.2

/-- Amount of cement used for Ben's street in tons -/
def bens_cement : ℝ := tess_cement * 0.9

/-- Amount of cement used for Olivia's street in tons -/
def olivias_cement : ℝ := bens_cement * 2

/-- Total amount of cement used for all four streets in tons -/
def total_cement : ℝ := lexis_cement + tess_cement + bens_cement + olivias_cement

theorem total_cement_is_54_4 : total_cement = 54.4 := by
  sorry

end NUMINAMATH_CALUDE_total_cement_is_54_4_l930_93061


namespace NUMINAMATH_CALUDE_complex_modulus_implies_real_value_l930_93021

theorem complex_modulus_implies_real_value (a : ℝ) : 
  Complex.abs ((a + 2 * Complex.I) * (1 + Complex.I)) = 4 → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_implies_real_value_l930_93021


namespace NUMINAMATH_CALUDE_fourth_group_frequency_count_l930_93045

theorem fourth_group_frequency_count 
  (f₁ f₂ f₃ : ℝ) 
  (n₁ : ℕ) 
  (h₁ : f₁ = 0.1) 
  (h₂ : f₂ = 0.3) 
  (h₃ : f₃ = 0.4) 
  (h₄ : n₁ = 5) 
  (h₅ : f₁ + f₂ + f₃ < 1) : 
  ∃ (N : ℕ) (n₄ : ℕ), 
    N > 0 ∧ 
    f₁ = n₁ / N ∧ 
    n₄ = N * (1 - (f₁ + f₂ + f₃)) ∧ 
    n₄ = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_frequency_count_l930_93045


namespace NUMINAMATH_CALUDE_walmart_shelving_problem_l930_93028

/-- Given a total number of pots and the capacity of each shelf,
    calculate the number of shelves needed to stock all pots. -/
def shelves_needed (total_pots : ℕ) (vertical_capacity : ℕ) (horizontal_capacity : ℕ) : ℕ :=
  (total_pots + vertical_capacity * horizontal_capacity - 1) / (vertical_capacity * horizontal_capacity)

/-- Proof that 4 shelves are needed to stock 60 pots when each shelf can hold 
    5 vertically stacked pots in 3 side-by-side sets. -/
theorem walmart_shelving_problem : shelves_needed 60 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_walmart_shelving_problem_l930_93028


namespace NUMINAMATH_CALUDE_base_twelve_square_l930_93059

theorem base_twelve_square (b : ℕ) : b > 0 → (3 * b + 2)^2 = b^3 + 2 * b^2 + 4 → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_base_twelve_square_l930_93059


namespace NUMINAMATH_CALUDE_pizza_pepperoni_count_l930_93037

theorem pizza_pepperoni_count :
  ∀ (pepperoni ham sausage : ℕ),
    ham = 2 * pepperoni →
    sausage = pepperoni + 12 →
    pepperoni + ham + sausage = 22 * 6 →
    pepperoni = 30 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pepperoni_count_l930_93037


namespace NUMINAMATH_CALUDE_combined_salaries_l930_93002

theorem combined_salaries (salary_C : ℕ) (average_salary : ℕ) (num_individuals : ℕ) :
  salary_C = 16000 →
  average_salary = 9000 →
  num_individuals = 5 →
  (average_salary * num_individuals) - salary_C = 29000 :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_l930_93002


namespace NUMINAMATH_CALUDE_quadratic_inequality_l930_93039

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 40 > 0 ↔ x < -5 ∨ x > 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l930_93039


namespace NUMINAMATH_CALUDE_triangular_prism_ribbon_length_specific_l930_93018

/-- The total length of ribbon required to form a triangular prism -/
def triangular_prism_ribbon_length (base_side_length : ℝ) (height : ℝ) : ℝ :=
  3 * base_side_length + 3 * base_side_length + 3 * height

/-- Theorem: The total length of ribbon required to form a triangular prism
    with an equilateral triangle base of side length 10 feet and height 15 feet
    is equal to 105 feet -/
theorem triangular_prism_ribbon_length_specific :
  triangular_prism_ribbon_length 10 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_ribbon_length_specific_l930_93018


namespace NUMINAMATH_CALUDE_cube_sum_symmetric_polynomials_l930_93050

theorem cube_sum_symmetric_polynomials (x y z : ℝ) :
  let σ₁ : ℝ := x + y + z
  let σ₂ : ℝ := x*y + y*z + z*x
  let σ₃ : ℝ := x*y*z
  x^3 + y^3 + z^3 = σ₁^3 - 3*σ₁*σ₂ + 3*σ₃ := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_symmetric_polynomials_l930_93050


namespace NUMINAMATH_CALUDE_rachel_day_visitor_count_l930_93035

/-- The number of visitors to Buckingham Palace over two days -/
def total_visitors : ℕ := 829

/-- The number of visitors to Buckingham Palace on the day before Rachel's visit -/
def previous_day_visitors : ℕ := 246

/-- The number of visitors to Buckingham Palace on the day of Rachel's visit -/
def rachel_day_visitors : ℕ := total_visitors - previous_day_visitors

theorem rachel_day_visitor_count : rachel_day_visitors = 583 := by
  sorry

end NUMINAMATH_CALUDE_rachel_day_visitor_count_l930_93035


namespace NUMINAMATH_CALUDE_arthurs_dinner_cost_l930_93024

def dinner_cost (appetizer steak wine_glass dessert : ℚ) (wine_glasses : ℕ) (discount_percent tip_percent : ℚ) : ℚ :=
  let full_cost := appetizer + steak + (wine_glass * wine_glasses) + dessert
  let discount := steak * discount_percent
  let discounted_cost := full_cost - discount
  let tip := full_cost * tip_percent
  discounted_cost + tip

theorem arthurs_dinner_cost :
  dinner_cost 8 20 3 6 2 (1/2) (1/5) = 38 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_dinner_cost_l930_93024


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l930_93015

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (f x - f y)^2 ≤ |x - y|^3) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l930_93015


namespace NUMINAMATH_CALUDE_seven_faced_prism_has_five_lateral_faces_l930_93032

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  total_faces : ℕ
  base_faces : ℕ := 2

/-- Define a function that calculates the number of lateral faces of a prism. -/
def lateral_faces (p : Prism) : ℕ :=
  p.total_faces - p.base_faces

/-- Theorem stating that a prism with 7 faces has 5 lateral faces. -/
theorem seven_faced_prism_has_five_lateral_faces (p : Prism) (h : p.total_faces = 7) :
  lateral_faces p = 5 := by
  sorry


end NUMINAMATH_CALUDE_seven_faced_prism_has_five_lateral_faces_l930_93032


namespace NUMINAMATH_CALUDE_kelly_games_left_l930_93005

/-- Given that Kelly has 106 Nintendo games initially and gives away 64 games,
    prove that she will have 42 games left. -/
theorem kelly_games_left (initial_games : ℕ) (games_given_away : ℕ) 
    (h1 : initial_games = 106) (h2 : games_given_away = 64) : 
    initial_games - games_given_away = 42 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_left_l930_93005


namespace NUMINAMATH_CALUDE_inequality_solution_range_l930_93030

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 + 2 * (a - 1) * x - 4 < 0) ↔ 
  (-3 < a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l930_93030


namespace NUMINAMATH_CALUDE_gcd_90_210_l930_93049

theorem gcd_90_210 : Nat.gcd 90 210 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_210_l930_93049


namespace NUMINAMATH_CALUDE_remainder_8927_mod_11_l930_93095

theorem remainder_8927_mod_11 : 8927 % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8927_mod_11_l930_93095


namespace NUMINAMATH_CALUDE_not_divisible_by_67_l930_93055

theorem not_divisible_by_67 (x y : ℕ) 
  (h1 : ¬ 67 ∣ x) 
  (h2 : ¬ 67 ∣ y) 
  (h3 : 67 ∣ (7 * x + 32 * y)) : 
  ¬ 67 ∣ (10 * x + 17 * y + 1) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_67_l930_93055


namespace NUMINAMATH_CALUDE_fruits_left_l930_93064

def fruits_problem (plums guavas apples given_away : ℕ) : ℕ :=
  (plums + guavas + apples) - given_away

theorem fruits_left (plums guavas apples given_away : ℕ) 
  (h : given_away ≤ plums + guavas + apples) : 
  fruits_problem plums guavas apples given_away = 
  (plums + guavas + apples) - given_away :=
by
  sorry

end NUMINAMATH_CALUDE_fruits_left_l930_93064


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l930_93089

/-- For a quadratic equation with two equal real roots, the value of k is ±6 --/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x + 9 = 0 ∧ 
   ∀ y : ℝ, y^2 - k*y + 9 = 0 → y = x) →
  k = 6 ∨ k = -6 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l930_93089


namespace NUMINAMATH_CALUDE_compound_interest_principal_is_5000_l930_93051

-- Define the simple interest rate
def simple_interest_rate : ℝ := 0.10

-- Define the compound interest rate
def compound_interest_rate : ℝ := 0.12

-- Define the simple interest time period
def simple_interest_time : ℕ := 5

-- Define the compound interest time period
def compound_interest_time : ℕ := 2

-- Define the simple interest principal
def simple_interest_principal : ℝ := 1272

-- Define the function to calculate simple interest
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * (time : ℝ)

-- Define the function to calculate compound interest
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

-- Theorem to prove
theorem compound_interest_principal_is_5000 :
  ∃ (compound_principal : ℝ),
    simple_interest simple_interest_principal simple_interest_rate simple_interest_time =
    (1/2) * compound_interest compound_principal compound_interest_rate compound_interest_time ∧
    compound_principal = 5000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_principal_is_5000_l930_93051


namespace NUMINAMATH_CALUDE_shopping_cost_calculation_l930_93071

-- Define the prices and quantities
def carrot_price : ℚ := 2
def carrot_quantity : ℕ := 7
def milk_price : ℚ := 3
def milk_quantity : ℕ := 4
def pineapple_price : ℚ := 5
def pineapple_quantity : ℕ := 3
def pineapple_discount : ℚ := 0.5
def flour_price : ℚ := 8
def flour_quantity : ℕ := 1
def cookie_price : ℚ := 10
def cookie_quantity : ℕ := 1

-- Define the store's discount conditions
def store_discount_threshold : ℚ := 40
def store_discount_rate : ℚ := 0.1

-- Define the coupon conditions
def coupon_value : ℚ := 5
def coupon_threshold : ℚ := 25

-- Calculate the total cost before discounts
def total_before_discounts : ℚ :=
  carrot_price * carrot_quantity +
  milk_price * milk_quantity +
  pineapple_price * pineapple_quantity * (1 - pineapple_discount) +
  flour_price * flour_quantity +
  cookie_price * cookie_quantity

-- Apply store discount if applicable
def after_store_discount : ℚ :=
  if total_before_discounts > store_discount_threshold then
    total_before_discounts * (1 - store_discount_rate)
  else
    total_before_discounts

-- Apply coupon if applicable
def final_cost : ℚ :=
  if after_store_discount > coupon_threshold then
    after_store_discount - coupon_value
  else
    after_store_discount

-- Theorem to prove
theorem shopping_cost_calculation :
  final_cost = 41.35 := by sorry

end NUMINAMATH_CALUDE_shopping_cost_calculation_l930_93071


namespace NUMINAMATH_CALUDE_isosceles_triangle_height_l930_93060

/-- Given an isosceles triangle and a rectangle with the same area, where the base of the triangle
    equals the width of the rectangle (10 units), and the length of the rectangle is twice its width,
    prove that the height of the triangle is 40 units. -/
theorem isosceles_triangle_height (triangle_area rectangle_area : ℝ) 
  (triangle_base rectangle_width rectangle_length : ℝ) (triangle_height : ℝ) : 
  triangle_area = rectangle_area →
  triangle_base = rectangle_width →
  triangle_base = 10 →
  rectangle_length = 2 * rectangle_width →
  triangle_area = 1/2 * triangle_base * triangle_height →
  rectangle_area = rectangle_width * rectangle_length →
  triangle_height = 40 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_height_l930_93060


namespace NUMINAMATH_CALUDE_john_total_height_climbed_l930_93034

/-- Calculates the total height climbed by John given the number of flights, 
    height per flight, and additional climbing information. -/
def totalHeightClimbed (numFlights : ℕ) (heightPerFlight : ℕ) : ℕ :=
  let stairsHeight := numFlights * heightPerFlight
  let ropeHeight := stairsHeight / 2
  let ladderHeight := ropeHeight + 10
  stairsHeight + ropeHeight + ladderHeight

/-- Theorem stating that the total height climbed by John is 70 feet. -/
theorem john_total_height_climbed :
  totalHeightClimbed 3 10 = 70 := by
  sorry

end NUMINAMATH_CALUDE_john_total_height_climbed_l930_93034


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_l930_93063

theorem largest_solution_quadratic (x : ℝ) : 
  (3 * (8 * x^2 + 10 * x + 8) = x * (8 * x - 34)) →
  x ≤ (-4 + Real.sqrt 10) / 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_l930_93063


namespace NUMINAMATH_CALUDE_suki_bag_weight_l930_93067

-- Define the given quantities
def suki_bags : ℝ := 6.5
def jimmy_bags : ℝ := 4.5
def jimmy_bag_weight : ℝ := 18
def container_weight : ℝ := 8
def total_containers : ℕ := 28

-- Define the theorem
theorem suki_bag_weight :
  let total_weight := container_weight * total_containers
  let jimmy_total_weight := jimmy_bags * jimmy_bag_weight
  let suki_total_weight := total_weight - jimmy_total_weight
  suki_total_weight / suki_bags = 22 := by
sorry


end NUMINAMATH_CALUDE_suki_bag_weight_l930_93067


namespace NUMINAMATH_CALUDE_stereo_trade_in_value_l930_93075

theorem stereo_trade_in_value (old_cost new_cost discount_percent out_of_pocket : ℚ) 
  (h1 : old_cost = 250)
  (h2 : new_cost = 600)
  (h3 : discount_percent = 25)
  (h4 : out_of_pocket = 250) :
  let discounted_price := new_cost * (1 - discount_percent / 100)
  let trade_in_value := discounted_price - out_of_pocket
  trade_in_value / old_cost * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_stereo_trade_in_value_l930_93075


namespace NUMINAMATH_CALUDE_student_age_problem_l930_93046

theorem student_age_problem (n : ℕ) : 
  n < 10 →
  (8 : ℝ) * n = (10 : ℝ) * (n + 1) - 28 →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_student_age_problem_l930_93046


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l930_93052

theorem triangle_angle_problem (A B C x : ℝ) : 
  A = 40 ∧ B = 3*x ∧ C = 2*x ∧ A + B + C = 180 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l930_93052


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l930_93026

theorem at_least_one_greater_than_one (x y : ℝ) (h : x + y > 2) : max x y > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l930_93026


namespace NUMINAMATH_CALUDE_city_mileage_problem_l930_93098

theorem city_mileage_problem (n : ℕ) : n * (n - 1) / 2 = 15 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_city_mileage_problem_l930_93098


namespace NUMINAMATH_CALUDE_jacket_price_restoration_l930_93047

theorem jacket_price_restoration :
  ∀ (original_price : ℝ),
  original_price > 0 →
  let price_after_first_reduction := original_price * (1 - 0.2)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.25)
  let required_increase := (original_price / price_after_second_reduction) - 1
  abs (required_increase - 0.6667) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_restoration_l930_93047


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l930_93088

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l930_93088


namespace NUMINAMATH_CALUDE_equivalence_theorem_l930_93093

theorem equivalence_theorem (x y z : ℝ) : 
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ 1/x + 1/y + 1/z ≤ 1) ↔ 
  (∀ (a b c d : ℝ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c > d) → a^2*x + b^2*y + c^2*z > d^2) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_theorem_l930_93093


namespace NUMINAMATH_CALUDE_empty_solution_set_has_solutions_l930_93029

-- Define the inequality
def inequality (x a : ℝ) : Prop := |x - 4| + |3 - x| < a

-- Theorem 1: The solution set is empty iff a ≤ 1
theorem empty_solution_set (a : ℝ) :
  (∀ x : ℝ, ¬ inequality x a) ↔ a ≤ 1 := by sorry

-- Theorem 2: The inequality has solutions iff a > 1
theorem has_solutions (a : ℝ) :
  (∃ x : ℝ, inequality x a) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_has_solutions_l930_93029


namespace NUMINAMATH_CALUDE_equation_roots_l930_93038

theorem equation_roots :
  let f (x : ℝ) := 20 / (x^2 - 9) - 3 / (x + 3) - 2
  ∀ x : ℝ, f x = 0 ↔ x = (-3 + Real.sqrt 385) / 4 ∨ x = (-3 - Real.sqrt 385) / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l930_93038


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l930_93065

theorem arccos_equation_solution :
  ∃ x : ℝ, x = Real.sqrt (1 / (64 - 36 * Real.sqrt 3)) ∧ 
    Real.arccos (3 * x) - Real.arccos x = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l930_93065


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_is_zero_l930_93081

theorem sum_of_four_numbers_is_zero 
  (x y s t : ℝ) 
  (h_distinct : x ≠ y ∧ x ≠ s ∧ x ≠ t ∧ y ≠ s ∧ y ≠ t ∧ s ≠ t) 
  (h_equality : (x + s) / (x + t) = (y + t) / (y + s)) : 
  x + y + s + t = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_is_zero_l930_93081


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l930_93076

def A : ℕ := 123456
def B : ℕ := 142857
def M : ℕ := 1000009
def N : ℕ := 750298

theorem multiplicative_inverse_modulo :
  (A * B * N) % M = 1 :=
sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l930_93076


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l930_93079

/-- Given an arithmetic sequence where the 20th term is 17 and the 21st term is 20,
    prove that the 5th term is -28. -/
theorem arithmetic_sequence_fifth_term
  (a : ℤ) -- First term of the sequence
  (d : ℤ) -- Common difference
  (h1 : a + 19 * d = 17) -- 20th term is 17
  (h2 : a + 20 * d = 20) -- 21st term is 20
  : a + 4 * d = -28 := by -- 5th term is -28
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l930_93079


namespace NUMINAMATH_CALUDE_chime_2500_date_l930_93099

/-- Represents a date --/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a time --/
structure Time where
  hour : ℕ
  minute : ℕ

/-- Represents a clock with a specific chiming pattern --/
structure ChimingClock where
  /-- Chimes once at 30 minutes past each hour --/
  chimeAtHalfHour : Bool
  /-- Chimes on the hour according to the hour number --/
  chimeOnHour : ℕ → ℕ

/-- Calculates the number of chimes between two dates and times --/
def countChimes (clock : ChimingClock) (startDate : Date) (startTime : Time) (endDate : Date) (endTime : Time) : ℕ := sorry

/-- The theorem to be proved --/
theorem chime_2500_date (clock : ChimingClock) : 
  let startDate := Date.mk 2003 2 26
  let startTime := Time.mk 10 45
  let endDate := Date.mk 2003 3 21
  countChimes clock startDate startTime endDate (Time.mk 23 59) = 2500 := by sorry

end NUMINAMATH_CALUDE_chime_2500_date_l930_93099


namespace NUMINAMATH_CALUDE_smallest_three_digit_solution_l930_93023

theorem smallest_three_digit_solution :
  ∃ (n : ℕ), 
    n ≥ 100 ∧ 
    n < 1000 ∧ 
    77 * n ≡ 231 [MOD 385] ∧ 
    (∀ m : ℕ, m ≥ 100 ∧ m < n ∧ 77 * m ≡ 231 [MOD 385] → false) ∧
    n = 113 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_solution_l930_93023


namespace NUMINAMATH_CALUDE_find_n_l930_93022

theorem find_n : ∃ n : ℤ, 3^3 - 7 = 4^2 + 2 + n ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l930_93022
