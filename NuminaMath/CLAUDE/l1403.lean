import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1403_140382

def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (2 * x - 1)}

theorem intersection_of_M_and_N : M ∩ N = {x | 1/2 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1403_140382


namespace NUMINAMATH_CALUDE_dish_washing_time_l1403_140303

-- Define the given constants
def sweep_time_per_room : ℕ := 3
def laundry_time_per_load : ℕ := 9
def anna_rooms_swept : ℕ := 10
def billy_laundry_loads : ℕ := 2
def billy_dishes_to_wash : ℕ := 6

-- Define the theorem
theorem dish_washing_time :
  ∃ (dish_time : ℕ),
    dish_time = 2 ∧
    sweep_time_per_room * anna_rooms_swept =
    laundry_time_per_load * billy_laundry_loads + billy_dishes_to_wash * dish_time :=
by sorry

end NUMINAMATH_CALUDE_dish_washing_time_l1403_140303


namespace NUMINAMATH_CALUDE_jimmy_passing_points_l1403_140308

/-- The minimum number of points required to pass to the next class -/
def min_points_to_pass : ℕ := 50

/-- The number of points earned per exam -/
def points_per_exam : ℕ := 20

/-- The number of exams taken -/
def num_exams : ℕ := 3

/-- The number of points lost for bad behavior -/
def points_lost_behavior : ℕ := 5

/-- The maximum number of additional points Jimmy can lose and still pass -/
def max_additional_points_to_lose : ℕ := 5

theorem jimmy_passing_points :
  max_additional_points_to_lose = 
    points_per_exam * num_exams - points_lost_behavior - min_points_to_pass := by
  sorry

end NUMINAMATH_CALUDE_jimmy_passing_points_l1403_140308


namespace NUMINAMATH_CALUDE_vector_computation_l1403_140318

theorem vector_computation :
  (4 : ℝ) • ![(-3 : ℝ), 5] - (3 : ℝ) • ![(-2 : ℝ), 6] = ![-6, 2] := by
  sorry

end NUMINAMATH_CALUDE_vector_computation_l1403_140318


namespace NUMINAMATH_CALUDE_tadd_250th_number_l1403_140360

/-- Represents the block size for a player in the n-th round -/
def blockSize (n : ℕ) : ℕ := 6 * n - 5

/-- Sum of numbers spoken up to the k-th block -/
def sumUpToBlock (k : ℕ) : ℕ := 3 * k * (k - 1)

/-- The counting game as described in the problem -/
def countingGame : Prop :=
  ∃ (k : ℕ),
    sumUpToBlock (k - 1) < 250 ∧
    250 ≤ sumUpToBlock k ∧
    250 = sumUpToBlock (k - 1) + (250 - sumUpToBlock (k - 1))

theorem tadd_250th_number :
  countingGame → (∃ (k : ℕ), 250 = sumUpToBlock (k - 1) + (250 - sumUpToBlock (k - 1))) :=
by sorry

end NUMINAMATH_CALUDE_tadd_250th_number_l1403_140360


namespace NUMINAMATH_CALUDE_union_complement_equal_l1403_140351

def U : Finset Nat := {0,1,2,4,6,8}
def M : Finset Nat := {0,4,6}
def N : Finset Nat := {0,1,6}

theorem union_complement_equal : M ∪ (U \ N) = {0,2,4,6,8} := by sorry

end NUMINAMATH_CALUDE_union_complement_equal_l1403_140351


namespace NUMINAMATH_CALUDE_sheila_attend_probability_l1403_140363

-- Define the probabilities
def prob_rain : ℝ := 0.5
def prob_sunny : ℝ := 1 - prob_rain
def prob_attend_if_rain : ℝ := 0.3
def prob_attend_if_sunny : ℝ := 0.9
def prob_remember : ℝ := 0.9

-- Define the theorem
theorem sheila_attend_probability :
  prob_rain * prob_attend_if_rain * prob_remember +
  prob_sunny * prob_attend_if_sunny * prob_remember = 0.54 := by
  sorry

end NUMINAMATH_CALUDE_sheila_attend_probability_l1403_140363


namespace NUMINAMATH_CALUDE_p_distance_is_300_l1403_140330

/-- A race between two runners p and q -/
structure Race where
  /-- The speed of runner q in meters per second -/
  q_speed : ℝ
  /-- The length of the race course in meters -/
  race_length : ℝ

/-- The result of the race -/
def race_result (r : Race) : ℝ := 
  let p_speed := 1.2 * r.q_speed
  let p_distance := r.race_length + 50
  p_distance

/-- Theorem: Under the given conditions, p runs 300 meters -/
theorem p_distance_is_300 (r : Race) : 
  r.race_length > 0 ∧ 
  r.q_speed > 0 ∧ 
  r.race_length / r.q_speed = (r.race_length + 50) / (1.2 * r.q_speed) → 
  race_result r = 300 := by
  sorry

end NUMINAMATH_CALUDE_p_distance_is_300_l1403_140330


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l1403_140301

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where b = √3 and (2c-a)/b * cos(B) = cos(A), prove that a+c is in the range (√3, 2√3]. -/
theorem triangle_side_sum_range (a b c A B C : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b = Real.sqrt 3 →
  (2*c - a)/b * Real.cos B = Real.cos A →
  ∃ (x : ℝ), Real.sqrt 3 < x ∧ x ≤ 2 * Real.sqrt 3 ∧ a + c = x :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l1403_140301


namespace NUMINAMATH_CALUDE_elena_max_flour_l1403_140397

/-- Represents the recipe and available ingredients for Elena's bread --/
structure BreadRecipe where
  butter_ratio : ℚ  -- Ratio of butter to flour (in ounces per cup)
  sugar_ratio : ℚ   -- Ratio of sugar to flour (in ounces per cup)
  available_butter : ℚ  -- Available butter in ounces
  available_sugar : ℚ   -- Available sugar in ounces

/-- Calculates the maximum cups of flour that can be used given the recipe and available ingredients --/
def max_flour (recipe : BreadRecipe) : ℚ :=
  min 
    (recipe.available_butter / recipe.butter_ratio)
    (recipe.available_sugar / recipe.sugar_ratio)

/-- Elena's specific bread recipe and available ingredients --/
def elena_recipe : BreadRecipe :=
  { butter_ratio := 3/4
  , sugar_ratio := 2/5
  , available_butter := 24
  , available_sugar := 30 }

/-- Theorem stating that the maximum number of cups of flour Elena can use is 32 --/
theorem elena_max_flour : 
  max_flour elena_recipe = 32 := by sorry

end NUMINAMATH_CALUDE_elena_max_flour_l1403_140397


namespace NUMINAMATH_CALUDE_perpendicular_unit_vectors_l1403_140325

def a : ℝ × ℝ := (2, -2)

theorem perpendicular_unit_vectors :
  let v₁ : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let v₂ : ℝ × ℝ := (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
  (v₁.1 * a.1 + v₁.2 * a.2 = 0 ∧ v₁.1^2 + v₁.2^2 = 1) ∧
  (v₂.1 * a.1 + v₂.2 * a.2 = 0 ∧ v₂.1^2 + v₂.2^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vectors_l1403_140325


namespace NUMINAMATH_CALUDE_infinitely_many_numbers_with_property_l1403_140332

/-- A function that returns the number of divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- A function that returns the product of prime factors of a natural number -/
def prodPrimeFactors (n : ℕ) : ℕ := sorry

/-- A function that returns the product of exponents in the prime factorization of a natural number -/
def prodExponents (n : ℕ) : ℕ := sorry

/-- The property that we want to prove holds for infinitely many natural numbers -/
def hasProperty (n : ℕ) : Prop :=
  numDivisors n = prodPrimeFactors n - prodExponents n

theorem infinitely_many_numbers_with_property :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, hasProperty n := by sorry

end NUMINAMATH_CALUDE_infinitely_many_numbers_with_property_l1403_140332


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l1403_140377

/-- Given a line with equation y = mx + b, where m = -2/3 and b = 3/2, prove that mb = -1 -/
theorem line_slope_intercept_product (m b : ℚ) : 
  m = -2/3 → b = 3/2 → m * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l1403_140377


namespace NUMINAMATH_CALUDE_heidi_painting_fraction_l1403_140333

/-- Represents the time in minutes it takes Heidi to paint a wall -/
def total_time : ℚ := 45

/-- Represents the time in minutes we want to calculate the painted fraction for -/
def given_time : ℚ := 9

/-- Represents the fraction of the wall painted in the given time -/
def painted_fraction : ℚ := given_time / total_time

theorem heidi_painting_fraction :
  painted_fraction = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_heidi_painting_fraction_l1403_140333


namespace NUMINAMATH_CALUDE_solution_concentration_change_l1403_140306

/-- Given an initial solution concentration, a replacement solution concentration,
    and the fraction of the solution replaced, calculate the new concentration. -/
def new_concentration (initial_conc replacement_conc fraction_replaced : ℝ) : ℝ :=
  (initial_conc * (1 - fraction_replaced)) + (replacement_conc * fraction_replaced)

/-- Theorem stating that replacing 0.7142857142857143 of a 60% solution with a 25% solution
    results in a new concentration of 0.21285714285714285 -/
theorem solution_concentration_change : 
  new_concentration 0.60 0.25 0.7142857142857143 = 0.21285714285714285 := by sorry

end NUMINAMATH_CALUDE_solution_concentration_change_l1403_140306


namespace NUMINAMATH_CALUDE_remainder_theorem_l1403_140315

theorem remainder_theorem : (439 * 319 * 2012 + 2013) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1403_140315


namespace NUMINAMATH_CALUDE_hostel_provisions_l1403_140394

/-- The number of days the provisions would last for the initial number of men -/
def initial_days : ℕ := 28

/-- The number of days the provisions would last if 50 men left -/
def extended_days : ℕ := 35

/-- The number of men that would leave -/
def men_leaving : ℕ := 50

/-- The initial number of men in the hostel -/
def initial_men : ℕ := 250

theorem hostel_provisions :
  initial_men * initial_days = (initial_men - men_leaving) * extended_days := by
  sorry

end NUMINAMATH_CALUDE_hostel_provisions_l1403_140394


namespace NUMINAMATH_CALUDE_traffic_light_probability_l1403_140347

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_time : ℕ
  change_times : List ℕ

/-- Calculates the probability of observing a color change in a given interval -/
def probability_of_change (cycle : TrafficLightCycle) (interval : ℕ) : ℚ :=
  let change_windows := cycle.change_times.map (λ t => if t ≤ cycle.total_time - interval then interval else t + interval - cycle.total_time)
  let total_change_time := change_windows.sum
  total_change_time / cycle.total_time

/-- The main theorem: probability of observing a color change is 1/7 -/
theorem traffic_light_probability :
  let cycle : TrafficLightCycle := { total_time := 63, change_times := [30, 33, 63] }
  probability_of_change cycle 3 = 1/7 := by
  sorry


end NUMINAMATH_CALUDE_traffic_light_probability_l1403_140347


namespace NUMINAMATH_CALUDE_absolute_value_of_3_plus_i_l1403_140358

theorem absolute_value_of_3_plus_i :
  let z : ℂ := 3 + Complex.I
  Complex.abs z = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_3_plus_i_l1403_140358


namespace NUMINAMATH_CALUDE_factors_of_M_l1403_140379

/-- The number of natural-number factors of M, where M = 2^4 * 3^3 * 5^2 * 7^1 -/
def num_factors (M : ℕ) : ℕ :=
  (5 : ℕ) * 4 * 3 * 2

/-- M is defined as 2^4 * 3^3 * 5^2 * 7^1 -/
def M : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem factors_of_M : num_factors M = 120 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_M_l1403_140379


namespace NUMINAMATH_CALUDE_beef_price_calculation_l1403_140339

/-- The price per pound of beef, given the conditions of John's food order --/
def beef_price_per_pound : ℝ := 8

theorem beef_price_calculation (beef_amount : ℝ) (chicken_price : ℝ) (total_cost : ℝ) :
  beef_amount = 1000 →
  chicken_price = 3 →
  total_cost = 14000 →
  beef_price_per_pound * beef_amount + chicken_price * (2 * beef_amount) = total_cost := by
  sorry

#check beef_price_calculation

end NUMINAMATH_CALUDE_beef_price_calculation_l1403_140339


namespace NUMINAMATH_CALUDE_rectangle_area_l1403_140396

theorem rectangle_area (b : ℝ) : 
  let square_area : ℝ := 1296
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := circle_radius / 6
  let rectangle_area : ℝ := rectangle_length * b
  rectangle_area = 6 * b :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1403_140396


namespace NUMINAMATH_CALUDE_max_area_OAP_l1403_140380

noncomputable section

/-- The maximum area of triangle OAP given the conditions --/
theorem max_area_OAP (a m : ℝ) (h1 : 0 < a) (h2 : a < 1/2) 
  (h3 : -a < m) (h4 : m ≤ (a^2 + 1)/2)
  (h5 : ∃! P : ℝ × ℝ, P.2 > 0 ∧ P.1^2 + a^2*P.2^2 = a^2 ∧ P.2^2 = 2*(P.1 + m)) :
  ∃ (S : ℝ), S = (1/54)*Real.sqrt 6 ∧ 
  (∀ A P : ℝ × ℝ, A.2 = 0 ∧ A.1^2 + a^2*A.2^2 = a^2 ∧ A.1 < 0 ∧
   P.2 > 0 ∧ P.1^2 + a^2*P.2^2 = a^2 ∧ P.2^2 = 2*(P.1 + m) →
   (1/2) * abs (A.1 * P.2) ≤ S) := by
  sorry

end

end NUMINAMATH_CALUDE_max_area_OAP_l1403_140380


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1403_140355

theorem polynomial_identity_sum_of_squares : 
  ∀ (a b c d e f : ℤ), 
  (∀ x : ℝ, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) → 
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 770 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l1403_140355


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1403_140390

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem: The man's speed against the current is 20 km/hr given the conditions -/
theorem mans_speed_against_current :
  speed_against_current 25 2.5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l1403_140390


namespace NUMINAMATH_CALUDE_triangle_area_in_nested_rectangles_l1403_140352

/-- Given a rectangle with dimensions a × b and a smaller rectangle inside with dimensions u × v,
    where the sides are parallel, the area of one of the four congruent right triangles formed by
    connecting the vertices of the smaller rectangle to the midpoints of the sides of the larger
    rectangle is (a-u)(b-v)/8. -/
theorem triangle_area_in_nested_rectangles (a b u v : ℝ) (ha : 0 < a) (hb : 0 < b)
    (hu : 0 < u) (hv : 0 < v) (hu_lt_a : u < a) (hv_lt_b : v < b) :
  (a - u) * (b - v) / 8 = (a - u) * (b - v) / 8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_in_nested_rectangles_l1403_140352


namespace NUMINAMATH_CALUDE_expression_equals_one_l1403_140334

theorem expression_equals_one :
  (121^2 - 11^2) / (91^2 - 13^2) * ((91-13)*(91+13)) / ((121-11)*(121+11)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1403_140334


namespace NUMINAMATH_CALUDE_box_difference_b_and_d_l1403_140314

/-- Represents the number of boxes of table tennis balls taken by each person. -/
structure BoxCount where
  a : ℕ  -- Number of boxes taken by A
  b : ℕ  -- Number of boxes taken by B
  c : ℕ  -- Number of boxes taken by C
  d : ℕ  -- Number of boxes taken by D

/-- Represents the money owed between individuals. -/
structure MoneyOwed where
  a_to_c : ℕ  -- Amount A owes to C
  b_to_d : ℕ  -- Amount B owes to D

/-- Theorem stating the difference in boxes between B and D is 18. -/
theorem box_difference_b_and_d (boxes : BoxCount) (money : MoneyOwed) : 
  boxes.b = boxes.a + 4 →  -- A took 4 boxes less than B
  boxes.d = boxes.c + 8 →  -- C took 8 boxes less than D
  money.a_to_c = 112 →     -- A owes C 112 yuan
  money.b_to_d = 72 →      -- B owes D 72 yuan
  boxes.b - boxes.d = 18 := by
  sorry

#check box_difference_b_and_d

end NUMINAMATH_CALUDE_box_difference_b_and_d_l1403_140314


namespace NUMINAMATH_CALUDE_equilibrium_constant_is_20_l1403_140391

/-- The equilibrium constant for the reaction NH₄I(s) ⇌ NH₃(g) + HI(g) -/
def equilibrium_constant (h2_conc : ℝ) (hi_conc : ℝ) : ℝ :=
  let hi_from_nh4i := hi_conc + 2 * h2_conc
  hi_from_nh4i * hi_conc

/-- Theorem stating that the equilibrium constant is 20 (mol/L)² under given conditions -/
theorem equilibrium_constant_is_20 (h2_conc : ℝ) (hi_conc : ℝ)
  (h2_eq : h2_conc = 0.5)
  (hi_eq : hi_conc = 4) :
  equilibrium_constant h2_conc hi_conc = 20 := by
  sorry

end NUMINAMATH_CALUDE_equilibrium_constant_is_20_l1403_140391


namespace NUMINAMATH_CALUDE_sequence_sum_l1403_140383

/-- Given a sequence {a_n} with sum of first n terms S_n, prove S_n = -3^(n-1) -/
theorem sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (a 1 = -1) → 
  (∀ n : ℕ, a (n + 1) = 2 * S n) → 
  (∀ n : ℕ, S n = -(3^(n - 1))) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1403_140383


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1403_140321

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (hr2 : r ≠ 0) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1403_140321


namespace NUMINAMATH_CALUDE_employee_salary_proof_l1403_140348

/-- The weekly salary of employee n -/
def salary_n : ℝ := 270

/-- The weekly salary of employee m -/
def salary_m : ℝ := 1.2 * salary_n

/-- The total weekly salary for both employees -/
def total_salary : ℝ := 594

theorem employee_salary_proof :
  salary_n + salary_m = total_salary :=
by sorry

end NUMINAMATH_CALUDE_employee_salary_proof_l1403_140348


namespace NUMINAMATH_CALUDE_wind_velocity_theorem_l1403_140359

-- Define the relationship between pressure, area, and velocity
def pressure_relation (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^2

-- Given initial condition
def initial_condition (k : ℝ) : Prop :=
  pressure_relation k 9 105 = 4

-- Theorem to prove
theorem wind_velocity_theorem (k : ℝ) (h : initial_condition k) :
  pressure_relation k 36 70 = 64 := by
  sorry

end NUMINAMATH_CALUDE_wind_velocity_theorem_l1403_140359


namespace NUMINAMATH_CALUDE_cannot_return_to_start_l1403_140392

-- Define the type for points on the plane
def Point := ℝ × ℝ

-- Define the allowed moves
def move_up (p : Point) : Point := (p.1, p.2 + 2*p.1)
def move_down (p : Point) : Point := (p.1, p.2 - 2*p.1)
def move_right (p : Point) : Point := (p.1 + 2*p.2, p.2)
def move_left (p : Point) : Point := (p.1 - 2*p.2, p.2)

-- Define a sequence of moves
inductive Move
| up : Move
| down : Move
| right : Move
| left : Move

def apply_move (p : Point) (m : Move) : Point :=
  match m with
  | Move.up => move_up p
  | Move.down => move_down p
  | Move.right => move_right p
  | Move.left => move_left p

def apply_moves (p : Point) (ms : List Move) : Point :=
  ms.foldl apply_move p

-- The main theorem
theorem cannot_return_to_start : 
  ∀ (ms : List Move), apply_moves (1, Real.sqrt 2) ms ≠ (1, Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_cannot_return_to_start_l1403_140392


namespace NUMINAMATH_CALUDE_second_month_sale_l1403_140338

def average_sale : ℕ := 5900
def first_month : ℕ := 5921
def third_month : ℕ := 5568
def fourth_month : ℕ := 6088
def fifth_month : ℕ := 6433
def sixth_month : ℕ := 5922

theorem second_month_sale :
  ∃ (second_month : ℕ),
    second_month = 
      6 * average_sale - (first_month + third_month + fourth_month + fifth_month + sixth_month) ∧
    second_month = 5468 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l1403_140338


namespace NUMINAMATH_CALUDE_part1_part2_l1403_140350

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + 3| - |2*x - a|

-- Part 1
theorem part1 (a : ℝ) : 
  (∃ x, f a x ≤ -5) → (a ≤ -8 ∨ a ≥ 2) := by sorry

-- Part 2
theorem part2 (a : ℝ) : 
  (∀ x, f a (x - 1/2) + f a (-x - 1/2) = 0) → a = 1 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1403_140350


namespace NUMINAMATH_CALUDE_last_page_cards_l1403_140373

/-- Represents the number of cards that can be placed on different page types -/
inductive PageType
| Four : PageType
| Six : PageType
| Eight : PageType

/-- Calculates the number of cards on the last partially-filled page -/
def cardsOnLastPage (totalCards : ℕ) (pageTypes : List PageType) : ℕ :=
  sorry

/-- Theorem stating that for 137 cards and the given page types, 
    the number of cards on the last partially-filled page is 1 -/
theorem last_page_cards : 
  cardsOnLastPage 137 [PageType.Four, PageType.Six, PageType.Eight] = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_page_cards_l1403_140373


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_range_of_a_l1403_140395

-- Part I
theorem sum_of_squares_inequality (a b c : ℝ) (h : a + b + c = 1) :
  (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16/3 := by sorry

-- Part II
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, |x - a| + |2*x - 1| ≥ 2) :
  a ≤ -3/2 ∨ a ≥ 5/2 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_range_of_a_l1403_140395


namespace NUMINAMATH_CALUDE_f_properties_l1403_140387

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2*x) + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 / 2

theorem f_properties :
  (∃ (t : ℝ), t > 0 ∧ ∀ (x : ℝ), f (x + t) = f x ∧ ∀ (s : ℝ), s > 0 ∧ (∀ (x : ℝ), f (x + s) = f x) → t ≤ s) ∧
  (∀ (x : ℝ), x ≥ -π/12 ∧ x ≤ 5*π/12 → f x ≥ -1/2 ∧ f x ≤ 1) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≥ -π/12 ∧ x₁ ≤ 5*π/12 ∧ x₂ ≥ -π/12 ∧ x₂ ≤ 5*π/12 ∧ f x₁ = -1/2 ∧ f x₂ = 1) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l1403_140387


namespace NUMINAMATH_CALUDE_kiwi_profit_optimization_l1403_140365

/-- Kiwi prices and profit optimization problem -/
theorem kiwi_profit_optimization 
  (green_price red_price : ℕ) 
  (green_cost red_cost : ℕ) 
  (total_boxes : ℕ) 
  (max_expenditure : ℕ) :
  green_cost = 80 →
  red_cost = 100 →
  total_boxes = 21 →
  max_expenditure = 2000 →
  red_price = green_price + 25 →
  6 * green_price = 5 * red_price - 25 →
  green_price = 100 ∧ 
  red_price = 125 ∧
  (∃ (green_boxes red_boxes : ℕ),
    green_boxes + red_boxes = total_boxes ∧
    green_boxes * green_cost + red_boxes * red_cost ≤ max_expenditure ∧
    green_boxes = 5 ∧ 
    red_boxes = 16 ∧
    (green_boxes * (green_price - green_cost) + red_boxes * (red_price - red_cost)) = 500 ∧
    ∀ (g r : ℕ), 
      g + r = total_boxes → 
      g * green_cost + r * red_cost ≤ max_expenditure →
      g * (green_price - green_cost) + r * (red_price - red_cost) ≤ 500) :=
by sorry

end NUMINAMATH_CALUDE_kiwi_profit_optimization_l1403_140365


namespace NUMINAMATH_CALUDE_quadratic_translation_l1403_140362

-- Define the original quadratic function
def f (x : ℝ) : ℝ := (x - 2009) * (x - 2008) + 4

-- Define the translated function
def g (x : ℝ) : ℝ := f x - 4

-- Theorem statement
theorem quadratic_translation :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧ |x₁ - x₂| = 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_translation_l1403_140362


namespace NUMINAMATH_CALUDE_triangle_area_l1403_140319

/-- Proves that a triangle with the given conditions has an original area of 4 square cm -/
theorem triangle_area (base : ℝ) (h : ℝ → ℝ) :
  h 0 = 2 →
  h 1 = h 0 + 6 →
  (1 / 2 : ℝ) * base * h 1 - (1 / 2 : ℝ) * base * h 0 = 12 →
  (1 / 2 : ℝ) * base * h 0 = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l1403_140319


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1403_140366

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x - 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | f a x > 0}

-- Theorem statement
theorem quadratic_inequality_solution (a : ℝ) (h : a < 0) :
  solution_set a = 
    if -1 < a then {x | 1 < x ∧ x < -1/a}
    else if a = -1 then ∅
    else {x | -1/a < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1403_140366


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1403_140326

noncomputable def α : ℝ := Real.arcsin (2/3 - Real.sqrt (5/9))
noncomputable def β : ℝ := Real.arctan 2

theorem trigonometric_identities 
  (h1 : Real.sin α + Real.cos α = 2/3)
  (h2 : π/2 < α ∧ α < π)
  (h3 : Real.tan β = 2) :
  (Real.sin (3*π/2 - α) * Real.cos (-π/2 - α) = -5/18) ∧
  ((1 / Real.sin (π - α)) - (1 / Real.cos (2*π - α)) + 
   (Real.sin β - Real.cos β) / (2*Real.sin β + Real.cos β) = (6*Real.sqrt 14 + 1)/5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1403_140326


namespace NUMINAMATH_CALUDE_union_equality_implies_a_values_l1403_140367

theorem union_equality_implies_a_values (a : ℝ) : 
  ({1, a} : Set ℝ) ∪ {a^2} = {1, a} → a = -1 ∨ a = 0 :=
by sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_values_l1403_140367


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1403_140322

theorem simplify_fraction_product : 
  (3 * 5 : ℚ) / (9 * 11) * (7 * 9 * 11) / (3 * 5 * 7) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1403_140322


namespace NUMINAMATH_CALUDE_base4_division_l1403_140361

/-- Represents a number in base 4 --/
def Base4 : Type := Nat

/-- Converts a natural number to its base 4 representation --/
def toBase4 (n : Nat) : Base4 := sorry

/-- Performs division in base 4 --/
def divBase4 (a b : Base4) : Base4 := sorry

/-- The theorem to be proved --/
theorem base4_division :
  divBase4 (toBase4 2023) (toBase4 13) = toBase4 155 := by sorry

end NUMINAMATH_CALUDE_base4_division_l1403_140361


namespace NUMINAMATH_CALUDE_flag_arrangement_count_l1403_140323

def total_arrangements (n m : ℕ) : ℕ := (n + m).factorial / (n.factorial * m.factorial)

def consecutive_arrangements (n m : ℕ) : ℕ := (m + 1).factorial / m.factorial

theorem flag_arrangement_count : 
  let total := total_arrangements 3 4
  let red_consecutive := consecutive_arrangements 1 4
  let blue_consecutive := consecutive_arrangements 3 1
  let both_consecutive := 2
  total - red_consecutive - blue_consecutive + both_consecutive = 28 := by sorry

end NUMINAMATH_CALUDE_flag_arrangement_count_l1403_140323


namespace NUMINAMATH_CALUDE_block_final_height_l1403_140328

/-- Given a block sliding down one ramp and up another, both at angle θ,
    with initial height h₁, mass m, and coefficient of kinetic friction μₖ,
    the final height h₂ is given by h₂ = h₁ / (1 + μₖ * √3) -/
theorem block_final_height
  (m : ℝ) (h₁ : ℝ) (μₖ : ℝ) (θ : ℝ) 
  (h₁_pos : h₁ > 0)
  (m_pos : m > 0)
  (μₖ_pos : μₖ > 0)
  (θ_val : θ = π/6) :
  let h₂ := h₁ / (1 + μₖ * Real.sqrt 3)
  ∀ ε > 0, abs (h₂ - h₁ / (1 + μₖ * Real.sqrt 3)) < ε :=
by
  sorry

#check block_final_height

end NUMINAMATH_CALUDE_block_final_height_l1403_140328


namespace NUMINAMATH_CALUDE_old_lamp_height_l1403_140346

theorem old_lamp_height (new_lamp_height : Real) (height_difference : Real) :
  new_lamp_height = 2.33 →
  height_difference = 1.33 →
  new_lamp_height - height_difference = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_old_lamp_height_l1403_140346


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_value_l1403_140349

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - (k^2 - 1) * x^2 - k^2 + 2

-- Define the derivative of f(x)
def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * (k^2 - 1) * x

-- Theorem statement
theorem tangent_perpendicular_implies_a_value (k : ℝ) (a : ℝ) (b : ℝ) :
  f k 1 = a →                   -- Point (1, a) is on the graph of f
  f_deriv k 1 = -1 →            -- Tangent line is perpendicular to x - y + b = 0
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_value_l1403_140349


namespace NUMINAMATH_CALUDE_angelinas_journey_l1403_140317

/-- Angelina's journey with varying speeds -/
theorem angelinas_journey (distance_home_grocery : ℝ) (distance_grocery_gym : ℝ) (time_difference : ℝ) :
  distance_home_grocery = 840 →
  distance_grocery_gym = 480 →
  time_difference = 40 →
  ∃ (speed_home_grocery : ℝ),
    speed_home_grocery > 0 ∧
    distance_home_grocery / speed_home_grocery - distance_grocery_gym / (2 * speed_home_grocery) = time_difference ∧
    2 * speed_home_grocery = 30 :=
by sorry

end NUMINAMATH_CALUDE_angelinas_journey_l1403_140317


namespace NUMINAMATH_CALUDE_six_people_arrangement_l1403_140320

/-- The number of ways to arrange n people in a row -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n people in a row, where two specific people must be adjacent in a fixed order -/
def arrangements_with_fixed_pair (n : ℕ) : ℕ := (n - 1).factorial

theorem six_people_arrangement : arrangements_with_fixed_pair 6 = 120 := by sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l1403_140320


namespace NUMINAMATH_CALUDE_delta_quotient_equals_two_plus_delta_x_l1403_140372

/-- The function f(x) = x^2 + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: For the function f(x) = x^2 + 1, given points (1, 2) and (1 + Δx, 2 + Δy) on the graph,
    Δy / Δx = 2 + Δx for any non-zero Δx -/
theorem delta_quotient_equals_two_plus_delta_x (Δx : ℝ) (Δy : ℝ) (h : Δx ≠ 0) :
  f (1 + Δx) = 2 + Δy →
  Δy / Δx = 2 + Δx :=
by sorry

end NUMINAMATH_CALUDE_delta_quotient_equals_two_plus_delta_x_l1403_140372


namespace NUMINAMATH_CALUDE_power_of_three_equation_l1403_140309

theorem power_of_three_equation (x : ℝ) : 
  4 * (3 : ℝ)^x = 243 → (x + 1) * (x - 1) = 16.696 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equation_l1403_140309


namespace NUMINAMATH_CALUDE_initial_bananas_count_l1403_140342

/-- The number of bananas Elizabeth bought initially -/
def initial_bananas : ℕ := sorry

/-- The number of bananas Elizabeth ate -/
def eaten_bananas : ℕ := 4

/-- The number of bananas Elizabeth has left -/
def remaining_bananas : ℕ := 8

/-- Theorem stating that the initial number of bananas is 12 -/
theorem initial_bananas_count : initial_bananas = 12 := by sorry

end NUMINAMATH_CALUDE_initial_bananas_count_l1403_140342


namespace NUMINAMATH_CALUDE_computer_price_increase_l1403_140329

theorem computer_price_increase (d : ℝ) (h1 : 2 * d = 520) (h2 : d > 0) : 
  (338 - d) / d * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1403_140329


namespace NUMINAMATH_CALUDE_min_box_height_is_19_l1403_140384

/-- Represents the specifications for packaging a fine arts collection --/
structure PackagingSpecs where
  totalVolume : ℝ  -- Total volume needed in cubic inches
  boxCost : ℝ      -- Cost per box in dollars
  totalCost : ℝ    -- Total cost spent on boxes in dollars

/-- Calculates the minimum height of cubic boxes needed to package a collection --/
def minBoxHeight (specs : PackagingSpecs) : ℕ :=
  sorry

/-- Theorem stating that the minimum box height for the given specifications is 19 inches --/
theorem min_box_height_is_19 :
  let specs : PackagingSpecs := {
    totalVolume := 3060000,  -- 3.06 million cubic inches
    boxCost := 0.5,          -- $0.50 per box
    totalCost := 255         -- $255 total cost
  }
  minBoxHeight specs = 19 := by sorry

end NUMINAMATH_CALUDE_min_box_height_is_19_l1403_140384


namespace NUMINAMATH_CALUDE_max_unique_sums_l1403_140313

def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25
def half_dollar : ℕ := 50

def coin_set : List ℕ := [nickel, nickel, nickel, dime, dime, dime, quarter, quarter, half_dollar, half_dollar]

def unique_sums (coins : List ℕ) : Finset ℕ :=
  (do
    let c1 <- coins
    let c2 <- coins
    pure (c1 + c2)
  ).toFinset

theorem max_unique_sums :
  Finset.card (unique_sums coin_set) = 10 := by sorry

end NUMINAMATH_CALUDE_max_unique_sums_l1403_140313


namespace NUMINAMATH_CALUDE_sector_area_l1403_140307

/-- The area of a sector with radius 6 and central angle 60° is 6π. -/
theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = 60 * π / 180) :
  (θ / (2 * π)) * π * r^2 = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1403_140307


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1403_140310

/-- Given a hyperbola with equation x²/144 - y²/81 = 1, prove that the slope of its asymptotes is 3/4 -/
theorem hyperbola_asymptote_slope :
  ∃ (m : ℚ), (∀ (x y : ℚ), x^2 / 144 - y^2 / 81 = 1 →
    (y = m * x ∨ y = -m * x) → m = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1403_140310


namespace NUMINAMATH_CALUDE_same_color_pairs_l1403_140371

def white_socks : ℕ := 5
def brown_socks : ℕ := 6
def blue_socks : ℕ := 3
def red_socks : ℕ := 2

def total_socks : ℕ := white_socks + brown_socks + blue_socks + red_socks

theorem same_color_pairs : 
  (Nat.choose white_socks 2) + (Nat.choose brown_socks 2) + 
  (Nat.choose blue_socks 2) + (Nat.choose red_socks 2) = 29 := by
sorry

end NUMINAMATH_CALUDE_same_color_pairs_l1403_140371


namespace NUMINAMATH_CALUDE_f_of_three_eq_seventeen_l1403_140385

/-- Given a function f(x) = ax + bx + c where c is a constant,
    if f(1) = 7 and f(2) = 12, then f(3) = 17 -/
theorem f_of_three_eq_seventeen
  (f : ℝ → ℝ)
  (a b c : ℝ)
  (h1 : ∀ x, f x = a * x + b * x + c)
  (h2 : f 1 = 7)
  (h3 : f 2 = 12) :
  f 3 = 17 := by
sorry

end NUMINAMATH_CALUDE_f_of_three_eq_seventeen_l1403_140385


namespace NUMINAMATH_CALUDE_X_equals_three_l1403_140337

/-- The length of the unknown segment X in the diagram --/
def X : ℝ := sorry

/-- The total length of the top side of the figure --/
def top_length : ℝ := 3 + 2 + X + 4

/-- The length of the bottom side of the figure --/
def bottom_length : ℝ := 12

/-- Theorem stating that X equals 3 --/
theorem X_equals_three : X = 3 := by
  sorry

end NUMINAMATH_CALUDE_X_equals_three_l1403_140337


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1403_140389

theorem quadratic_inequality_range (θ : Real) :
  (∀ m : Real, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) →
  (∀ m : Real, m ≥ 4 ∨ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1403_140389


namespace NUMINAMATH_CALUDE_hugo_prime_given_win_l1403_140312

/-- The number of players in the game -/
def num_players : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 8

/-- The set of prime numbers on the die -/
def prime_rolls : Set ℕ := {2, 3, 5, 7}

/-- The probability of rolling a prime number -/
def prob_prime : ℚ := 1/2

/-- The probability of Hugo winning the game -/
def prob_hugo_wins : ℚ := 1/num_players

/-- The probability that all other players roll non-prime or smaller prime -/
def prob_others_smaller : ℚ := (1/2)^(num_players - 1)

/-- The main theorem: probability of Hugo's first roll being prime given he won -/
theorem hugo_prime_given_win : 
  (prob_prime * prob_others_smaller) / prob_hugo_wins = 5/32 := by sorry

end NUMINAMATH_CALUDE_hugo_prime_given_win_l1403_140312


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l1403_140375

theorem trig_expression_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l1403_140375


namespace NUMINAMATH_CALUDE_sum_of_cubes_negative_l1403_140374

theorem sum_of_cubes_negative : 
  (Real.sqrt 2021 - Real.sqrt 2020)^3 + 
  (Real.sqrt 2020 - Real.sqrt 2019)^3 + 
  (Real.sqrt 2019 - Real.sqrt 2018)^3 < 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_negative_l1403_140374


namespace NUMINAMATH_CALUDE_problem_solution_l1403_140300

theorem problem_solution (a b : ℝ) : 
  |a + 1| + (b - 2)^2 = 0 → (a + b)^9 + a^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1403_140300


namespace NUMINAMATH_CALUDE_nine_b_equals_eighteen_l1403_140399

theorem nine_b_equals_eighteen (a b : ℤ) 
  (h1 : 6 * a + 3 * b = 0) 
  (h2 : b - 3 = a) : 
  9 * b = 18 := by
sorry

end NUMINAMATH_CALUDE_nine_b_equals_eighteen_l1403_140399


namespace NUMINAMATH_CALUDE_hexagon_perimeter_is_24_l1403_140302

/-- A hexagon with specific properties -/
structure Hexagon :=
  (AB EF BE AF CD DF : ℝ)
  (ab_ef_eq : AB = EF)
  (be_af_eq : BE = AF)
  (ab_length : AB = 3)
  (be_length : BE = 4)
  (cd_length : CD = 5)
  (df_length : DF = 5)

/-- The perimeter of the hexagon -/
def perimeter (h : Hexagon) : ℝ :=
  h.AB + h.BE + h.CD + h.DF + h.EF + h.AF

/-- Theorem stating that the perimeter of the hexagon is 24 units -/
theorem hexagon_perimeter_is_24 (h : Hexagon) : perimeter h = 24 := by
  sorry


end NUMINAMATH_CALUDE_hexagon_perimeter_is_24_l1403_140302


namespace NUMINAMATH_CALUDE_length_of_AE_l1403_140335

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (AF : ℝ)
  (CE : ℝ)
  (ED : ℝ)
  (area : ℝ)

/-- Theorem stating the length of AE in the given quadrilateral -/
theorem length_of_AE (q : Quadrilateral) 
  (h1 : q.AF = 30)
  (h2 : q.CE = 40)
  (h3 : q.ED = 50)
  (h4 : q.area = 7200) : 
  ∃ AE : ℝ, AE = 322.5 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AE_l1403_140335


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1403_140341

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1403_140341


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l1403_140324

def first_four_primes : List Nat := [2, 3, 5, 7]

theorem arithmetic_mean_reciprocals_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l1403_140324


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l1403_140398

theorem bottle_cap_distribution (total_caps : ℕ) (total_boxes : ℕ) (caps_per_box : ℕ) : 
  total_caps = 60 → 
  total_boxes = 60 → 
  total_caps = total_boxes * caps_per_box → 
  caps_per_box = 1 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l1403_140398


namespace NUMINAMATH_CALUDE_max_value_product_l1403_140376

theorem max_value_product (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + y + z = 2) : 
  x^4 * y^3 * z^2 ≤ (1 : ℝ) / 9765625 :=
sorry

end NUMINAMATH_CALUDE_max_value_product_l1403_140376


namespace NUMINAMATH_CALUDE_unique_solution_for_system_l1403_140386

theorem unique_solution_for_system : ∃! y : ℚ, 9 * y^2 + 8 * y - 2 = 0 ∧ 27 * y^2 + 62 * y - 8 = 0 :=
  by sorry

end NUMINAMATH_CALUDE_unique_solution_for_system_l1403_140386


namespace NUMINAMATH_CALUDE_range_of_a_l1403_140345

-- Define the conditions
def p (x : ℝ) : Prop := 1 / (x - 3) ≥ 1
def q (x a : ℝ) : Prop := |x - a| < 1

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- Theorem statement
theorem range_of_a :
  ∃ a_lower a_upper : ℝ, a_lower = 3 ∧ a_upper = 4 ∧
  ∀ a : ℝ, sufficient_not_necessary a ↔ a_lower < a ∧ a ≤ a_upper :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1403_140345


namespace NUMINAMATH_CALUDE_triangle_circle_relation_l1403_140388

theorem triangle_circle_relation (α β γ s R r : ℝ) :
  -- Triangle angles
  0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π →
  -- Perimeter is 2s
  s > 0 →
  -- R is the radius of the circumscribed circle
  R > 0 →
  -- r is the radius of the inscribed circle
  r > 0 →
  -- The theorem
  4 * R^2 * Real.cos α * Real.cos β * Real.cos γ = s^2 - (r + 2*R)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_relation_l1403_140388


namespace NUMINAMATH_CALUDE_diameters_sum_equals_legs_sum_l1403_140304

/-- A right-angled triangle with its circumscribed and inscribed circles -/
structure RightTriangle where
  /-- First leg of the right triangle -/
  a : ℝ
  /-- Second leg of the right triangle -/
  b : ℝ
  /-- Hypotenuse of the right triangle -/
  c : ℝ
  /-- Radius of the circumscribed circle -/
  R : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Condition: a and b are positive -/
  ha : 0 < a
  /-- Condition: a and b are positive -/
  hb : 0 < b
  /-- Pythagorean theorem -/
  pythagorean : a^2 + b^2 = c^2
  /-- Relation between hypotenuse and circumscribed circle diameter -/
  circum_diam : c = 2 * R
  /-- Relation for inscribed circle radius in a right triangle -/
  inscribed_radius : r = (a + b - c) / 2

/-- The sum of the diameters of the circumscribed and inscribed circles 
    is equal to the sum of the legs in a right-angled triangle -/
theorem diameters_sum_equals_legs_sum (t : RightTriangle) : 2 * t.R + 2 * t.r = t.a + t.b := by
  sorry

end NUMINAMATH_CALUDE_diameters_sum_equals_legs_sum_l1403_140304


namespace NUMINAMATH_CALUDE_no_quadratic_polynomial_satisfies_conditions_l1403_140305

theorem no_quadratic_polynomial_satisfies_conditions :
  ¬∃ (f : ℝ → ℝ), 
    (∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)) ∧ 
    (∀ x, f (x^2) = x^4) ∧
    (∀ x, f (f x) = (x^2 + 1)^4) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_polynomial_satisfies_conditions_l1403_140305


namespace NUMINAMATH_CALUDE_pet_store_ratios_l1403_140370

/-- Given the ratios of cats to dogs and dogs to parrots, and the number of cats,
    this theorem proves the number of dogs and parrots. -/
theorem pet_store_ratios (cats : ℕ) (dogs : ℕ) (parrots : ℕ) : 
  (3 : ℚ) / 4 = cats / dogs →  -- ratio of cats to dogs
  (2 : ℚ) / 5 = dogs / parrots →  -- ratio of dogs to parrots
  cats = 18 →  -- number of cats
  dogs = 24 ∧ parrots = 60 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_ratios_l1403_140370


namespace NUMINAMATH_CALUDE_function_max_value_l1403_140340

-- Define the function f
def f (x m : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + m

-- State the theorem
theorem function_max_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 20) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 20) →
  m = -2 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x (-2) ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_function_max_value_l1403_140340


namespace NUMINAMATH_CALUDE_magnitude_of_complex_reciprocal_l1403_140381

open Complex

theorem magnitude_of_complex_reciprocal (z : ℂ) : z = (1 : ℂ) / (1 - I) → abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_reciprocal_l1403_140381


namespace NUMINAMATH_CALUDE_binomial_square_constant_l1403_140327

theorem binomial_square_constant (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 60*x + k = (a*x + b)^2) → k = 900 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l1403_140327


namespace NUMINAMATH_CALUDE_k_range_l1403_140316

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the interval (5, 20)
def interval : Set ℝ := Set.Ioo 5 20

-- Define the property of having no maximum or minimum in the interval
def no_extremum (g : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x ∈ S, ∃ y ∈ S, g y > g x ∧ ∃ z ∈ S, g z < g x

-- State the theorem
theorem k_range (k : ℝ) :
  no_extremum (f k) interval → k ∈ Set.Iic 40 ∪ Set.Ici 160 := by sorry

end NUMINAMATH_CALUDE_k_range_l1403_140316


namespace NUMINAMATH_CALUDE_transformed_roots_l1403_140393

-- Define the original quadratic equation
def original_quadratic (p q r x : ℝ) : Prop := p * x^2 + q * x + r = 0

-- Define the roots of the original quadratic equation
def has_roots (p q r u v : ℝ) : Prop := original_quadratic p q r u ∧ original_quadratic p q r v

-- Define the new quadratic equation
def new_quadratic (p q r x : ℝ) : Prop := x^2 - 4 * q * x + 4 * p * r + 3 * q^2 = 0

-- Theorem statement
theorem transformed_roots (p q r u v : ℝ) (hp : p ≠ 0) :
  has_roots p q r u v →
  new_quadratic p q r (2 * p * u + 3 * q) ∧ new_quadratic p q r (2 * p * v + 3 * q) :=
by sorry

end NUMINAMATH_CALUDE_transformed_roots_l1403_140393


namespace NUMINAMATH_CALUDE_investment_comparison_l1403_140354

/-- Represents the final value of an investment after two years --/
def final_value (initial : ℝ) (change1 : ℝ) (change2 : ℝ) (dividend_rate : ℝ) : ℝ :=
  let value1 := initial * (1 + change1)
  let value2 := value1 * (1 + change2)
  value2 + value1 * dividend_rate

/-- Theorem stating the relationship between final investment values --/
theorem investment_comparison : 
  let a := final_value 200 0.15 (-0.10) 0.05
  let b := final_value 150 (-0.20) 0.30 0
  let c := final_value 100 0 0 0
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_investment_comparison_l1403_140354


namespace NUMINAMATH_CALUDE_probability_one_ball_in_last_box_l1403_140344

/-- The probability of exactly one ball landing in the last box when 100 balls
    are randomly distributed among 100 boxes. -/
theorem probability_one_ball_in_last_box :
  let n : ℕ := 100
  let p : ℝ := 1 / n
  (n : ℝ) * p * (1 - p) ^ (n - 1) = (1 - 1 / n) ^ (n - 1) := by sorry

end NUMINAMATH_CALUDE_probability_one_ball_in_last_box_l1403_140344


namespace NUMINAMATH_CALUDE_tims_weekend_ride_distance_l1403_140311

/-- Tim's weekly biking schedule and distance calculation -/
theorem tims_weekend_ride_distance 
  (work_distance : ℝ) 
  (work_days : ℕ) 
  (speed : ℝ) 
  (total_biking_hours : ℝ) 
  (h1 : work_distance = 20)
  (h2 : work_days = 5)
  (h3 : speed = 25)
  (h4 : total_biking_hours = 16) :
  let workday_distance := 2 * work_distance * work_days
  let workday_hours := workday_distance / speed
  let weekend_hours := total_biking_hours - workday_hours
  weekend_hours * speed = 200 := by
sorry


end NUMINAMATH_CALUDE_tims_weekend_ride_distance_l1403_140311


namespace NUMINAMATH_CALUDE_lottery_probability_l1403_140336

theorem lottery_probability : 
  let powerball_count : ℕ := 30
  let luckyball_count : ℕ := 49
  let luckyball_draw : ℕ := 6
  1 / (powerball_count * (Nat.choose luckyball_count luckyball_draw)) = 1 / 419514480 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l1403_140336


namespace NUMINAMATH_CALUDE_survivor_quitters_probability_l1403_140356

/-- The probability of all three quitters being from the same tribe in a Survivor-like scenario -/
theorem survivor_quitters_probability (total_people : ℕ) (tribe_size : ℕ) (quitters : ℕ)
  (h1 : total_people = 20)
  (h2 : tribe_size = 10)
  (h3 : quitters = 3)
  (h4 : total_people = 2 * tribe_size) :
  (Nat.choose tribe_size quitters * 2 : ℚ) / Nat.choose total_people quitters = 20 / 95 := by
  sorry

end NUMINAMATH_CALUDE_survivor_quitters_probability_l1403_140356


namespace NUMINAMATH_CALUDE_min_intersection_points_l1403_140364

/-- Represents a configuration of circles on a plane -/
structure CircleConfiguration where
  num_circles : ℕ
  num_intersections : ℕ
  intersections_per_circle : ℕ → ℕ

/-- The minimum number of intersections for a valid configuration -/
def min_intersections (config : CircleConfiguration) : ℕ :=
  (config.num_circles * config.intersections_per_circle 0) / 2

/-- Predicate for a valid circle configuration -/
def valid_configuration (config : CircleConfiguration) : Prop :=
  config.num_circles = 2008 ∧
  (∀ i, config.intersections_per_circle i ≥ 3) ∧
  config.num_intersections ≥ min_intersections config

theorem min_intersection_points (config : CircleConfiguration) 
  (h : valid_configuration config) : 
  config.num_intersections ≥ 3012 :=
sorry

end NUMINAMATH_CALUDE_min_intersection_points_l1403_140364


namespace NUMINAMATH_CALUDE_value_added_to_fraction_l1403_140331

theorem value_added_to_fraction : ∀ (N V : ℝ),
  N = 8 →
  0.75 * N + V = 8 →
  V = 2 := by
sorry

end NUMINAMATH_CALUDE_value_added_to_fraction_l1403_140331


namespace NUMINAMATH_CALUDE_only_set_B_forms_triangle_l1403_140368

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def set_A : List ℝ := [2, 6, 8]
def set_B : List ℝ := [4, 6, 7]
def set_C : List ℝ := [5, 6, 12]
def set_D : List ℝ := [2, 3, 6]

theorem only_set_B_forms_triangle :
  (¬ triangle_inequality set_A[0] set_A[1] set_A[2]) ∧
  (triangle_inequality set_B[0] set_B[1] set_B[2]) ∧
  (¬ triangle_inequality set_C[0] set_C[1] set_C[2]) ∧
  (¬ triangle_inequality set_D[0] set_D[1] set_D[2]) :=
by sorry

end NUMINAMATH_CALUDE_only_set_B_forms_triangle_l1403_140368


namespace NUMINAMATH_CALUDE_right_triangle_area_l1403_140353

theorem right_triangle_area (leg : ℝ) (altitude : ℝ) (area : ℝ) : 
  leg = 15 → altitude = 9 → area = 84.375 → 
  ∃ (hypotenuse : ℝ), 
    hypotenuse * altitude / 2 = area ∧ 
    leg^2 + altitude^2 = (hypotenuse / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1403_140353


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l1403_140343

/-- Represents an ellipse -/
structure Ellipse where
  center : ℝ × ℝ
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  point : ℝ × ℝ

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 81 + y^2 / 45 = 1

/-- Given ellipse satisfies the conditions -/
def given_ellipse : Ellipse :=
  { center := (0, 0)
  , foci := ((-3, 0), (3, 0))
  , point := (3, 8) }

/-- Theorem: The equation of the given ellipse is x²/81 + y²/45 = 1 -/
theorem ellipse_equation_proof :
  ellipse_equation given_ellipse (given_ellipse.point.1) (given_ellipse.point.2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_proof_l1403_140343


namespace NUMINAMATH_CALUDE_total_annual_interest_l1403_140357

theorem total_annual_interest (total_amount first_part : ℕ) : 
  total_amount = 4000 →
  first_part = 2800 →
  (first_part * 3 + (total_amount - first_part) * 5) / 100 = 144 := by
sorry

end NUMINAMATH_CALUDE_total_annual_interest_l1403_140357


namespace NUMINAMATH_CALUDE_min_blocks_for_slotted_structure_l1403_140378

/-- A block with one hook and five slots -/
structure Block :=
  (hook : Fin 6)
  (slots : Finset (Fin 6))
  (hook_slot_distinct : hook ∉ slots)
  (slot_count : slots.card = 5)

/-- A structure made of blocks -/
structure Structure :=
  (blocks : Finset Block)
  (no_visible_hooks : ∀ b ∈ blocks, ∃ b' ∈ blocks, b.hook ∈ b'.slots)

/-- The theorem stating that the minimum number of blocks required is 4 -/
theorem min_blocks_for_slotted_structure :
  ∀ s : Structure, s.blocks.card ≥ 4 ∧ 
  ∃ s' : Structure, s'.blocks.card = 4 :=
sorry

end NUMINAMATH_CALUDE_min_blocks_for_slotted_structure_l1403_140378


namespace NUMINAMATH_CALUDE_fish_problem_l1403_140369

/-- The number of fish originally in the shop -/
def original_fish : ℕ := 36

/-- The number of fish remaining after lunch sale -/
def after_lunch (f : ℕ) : ℕ := f / 2

/-- The number of fish sold for dinner -/
def dinner_sale (f : ℕ) : ℕ := (after_lunch f) / 3

/-- The number of fish remaining after both sales -/
def remaining_fish (f : ℕ) : ℕ := (after_lunch f) - (dinner_sale f)

theorem fish_problem :
  remaining_fish original_fish = 12 :=
by sorry

end NUMINAMATH_CALUDE_fish_problem_l1403_140369
