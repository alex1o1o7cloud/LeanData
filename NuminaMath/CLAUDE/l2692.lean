import Mathlib

namespace NUMINAMATH_CALUDE_trust_fund_remaining_zero_l2692_269289

/-- Represents the ratio of distribution for each beneficiary -/
structure DistributionRatio :=
  (dina : Rat)
  (eva : Rat)
  (frank : Rat)

/-- Theorem stating that the remaining fraction of the fund is 0 -/
theorem trust_fund_remaining_zero (ratio : DistributionRatio) 
  (h1 : ratio.dina = 4/8)
  (h2 : ratio.eva = 3/8)
  (h3 : ratio.frank = 1/8)
  (h4 : ratio.dina + ratio.eva + ratio.frank = 1) :
  let remaining : Rat := 1 - (ratio.dina + (1 - ratio.dina) * ratio.eva + (1 - ratio.dina - (1 - ratio.dina) * ratio.eva) * ratio.frank)
  remaining = 0 := by sorry

end NUMINAMATH_CALUDE_trust_fund_remaining_zero_l2692_269289


namespace NUMINAMATH_CALUDE_quadratic_function_y_order_l2692_269297

/-- Given a quadratic function f(x) = -x² - 4x + m, where m is a constant,
    and three points A, B, C on its graph, prove that the y-coordinate of B
    is greater than that of A, which is greater than that of C. -/
theorem quadratic_function_y_order (m : ℝ) (y₁ y₂ y₃ : ℝ) : 
  ((-3)^2 + 4*(-3) + m = y₁) →
  ((-2)^2 + 4*(-2) + m = y₂) →
  (1^2 + 4*1 + m = y₃) →
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_y_order_l2692_269297


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2692_269270

theorem quadratic_inequality_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + m > 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2692_269270


namespace NUMINAMATH_CALUDE_roots_of_f_eq_x_none_or_infinite_l2692_269279

theorem roots_of_f_eq_x_none_or_infinite (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) = f x + 1) :
  (∀ x : ℝ, f x ≠ x) ∨ (∃ S : Set ℝ, Set.Infinite S ∧ ∀ x ∈ S, f x = x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_f_eq_x_none_or_infinite_l2692_269279


namespace NUMINAMATH_CALUDE_next_occurrence_theorem_l2692_269214

/-- Represents a date and time -/
structure DateTime where
  day : Nat
  month : Nat
  year : Nat
  hour : Nat
  minute : Nat

/-- Check if a DateTime is valid -/
def isValidDateTime (dt : DateTime) : Prop :=
  dt.day ≥ 1 ∧ dt.day ≤ 31 ∧
  dt.month ≥ 1 ∧ dt.month ≤ 12 ∧
  dt.year ≥ 1900 ∧ dt.year ≤ 2100 ∧
  dt.hour ≥ 0 ∧ dt.hour ≤ 23 ∧
  dt.minute ≥ 0 ∧ dt.minute ≤ 59

/-- Check if two DateTimes use the same set of digits -/
def hasSameDigits (dt1 dt2 : DateTime) : Prop :=
  sorry

/-- The initial DateTime -/
def initialDateTime : DateTime :=
  { day := 25, month := 5, year := 1994, hour := 2, minute := 45 }

/-- The next DateTime with the same digits -/
def nextDateTime : DateTime :=
  { day := 1, month := 8, year := 1994, hour := 2, minute := 45 }

theorem next_occurrence_theorem :
  isValidDateTime initialDateTime ∧
  isValidDateTime nextDateTime ∧
  hasSameDigits initialDateTime nextDateTime ∧
  ∀ dt, isValidDateTime dt →
        hasSameDigits initialDateTime dt →
        (dt.year < nextDateTime.year ∨
         (dt.year = nextDateTime.year ∧ dt.month < nextDateTime.month) ∨
         (dt.year = nextDateTime.year ∧ dt.month = nextDateTime.month ∧ dt.day < nextDateTime.day) ∨
         (dt.year = nextDateTime.year ∧ dt.month = nextDateTime.month ∧ dt.day = nextDateTime.day ∧ dt.hour < nextDateTime.hour) ∨
         (dt.year = nextDateTime.year ∧ dt.month = nextDateTime.month ∧ dt.day = nextDateTime.day ∧ dt.hour = nextDateTime.hour ∧ dt.minute ≤ nextDateTime.minute)) :=
  by sorry

end NUMINAMATH_CALUDE_next_occurrence_theorem_l2692_269214


namespace NUMINAMATH_CALUDE_harolds_leftover_money_l2692_269206

/-- Harold's financial situation --/
def harolds_finances (income rent car_payment groceries : ℚ) : Prop :=
  let utilities := car_payment / 2
  let total_expenses := rent + car_payment + utilities + groceries
  let remaining := income - total_expenses
  let retirement := remaining / 2
  let left_after_retirement := remaining - retirement
  income = 2500 ∧ 
  rent = 700 ∧ 
  car_payment = 300 ∧ 
  groceries = 50 ∧ 
  left_after_retirement = 650

theorem harolds_leftover_money :
  ∃ (income rent car_payment groceries : ℚ),
    harolds_finances income rent car_payment groceries :=
sorry

end NUMINAMATH_CALUDE_harolds_leftover_money_l2692_269206


namespace NUMINAMATH_CALUDE_triangle_centers_l2692_269268

/-- Triangle XYZ with side lengths x, y, z -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Incenter coordinates (a, b, c) -/
structure Incenter where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_one : a + b + c = 1

/-- Centroid coordinates (p, q, r) -/
structure Centroid where
  p : ℝ
  q : ℝ
  r : ℝ
  sum_one : p + q + r = 1

/-- The theorem to be proved -/
theorem triangle_centers (t : Triangle) (i : Incenter) (c : Centroid) :
  t.x = 13 ∧ t.y = 15 ∧ t.z = 6 →
  i.a = 13/34 ∧ i.b = 15/34 ∧ i.c = 6/34 ∧
  c.p = 1/3 ∧ c.q = 1/3 ∧ c.r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_centers_l2692_269268


namespace NUMINAMATH_CALUDE_student_union_selections_l2692_269255

/-- Represents the number of students in each grade of the student union -/
structure StudentUnion where
  freshmen : Nat
  sophomores : Nat
  juniors : Nat

/-- Calculates the number of ways to select one person as president -/
def selectPresident (su : StudentUnion) : Nat :=
  su.freshmen + su.sophomores + su.juniors

/-- Calculates the number of ways to select one person from each grade for the standing committee -/
def selectStandingCommittee (su : StudentUnion) : Nat :=
  su.freshmen * su.sophomores * su.juniors

/-- Calculates the number of ways to select two people from different grades for a city activity -/
def selectCityActivity (su : StudentUnion) : Nat :=
  su.freshmen * su.sophomores + su.sophomores * su.juniors + su.juniors * su.freshmen

theorem student_union_selections (su : StudentUnion) 
  (h1 : su.freshmen = 5) 
  (h2 : su.sophomores = 6) 
  (h3 : su.juniors = 4) : 
  selectPresident su = 15 ∧ 
  selectStandingCommittee su = 120 ∧ 
  selectCityActivity su = 74 := by
  sorry

#eval selectPresident ⟨5, 6, 4⟩
#eval selectStandingCommittee ⟨5, 6, 4⟩
#eval selectCityActivity ⟨5, 6, 4⟩

end NUMINAMATH_CALUDE_student_union_selections_l2692_269255


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l2692_269211

-- Define the function
def f (x : ℝ) : ℝ := x^2 * (x - 3)

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem decreasing_interval_of_f :
  ∀ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioo 0 2, x < y → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l2692_269211


namespace NUMINAMATH_CALUDE_sphere_radius_in_tetrahedron_l2692_269262

/-- A regular tetrahedron with side length 1 containing four spheres --/
structure TetrahedronWithSpheres where
  /-- The side length of the regular tetrahedron --/
  sideLength : ℝ
  /-- The radius of each sphere --/
  sphereRadius : ℝ
  /-- The number of spheres --/
  numSpheres : ℕ
  /-- Each sphere is tangent to three faces of the tetrahedron --/
  tangentToFaces : Prop
  /-- Each sphere is tangent to the other three spheres --/
  tangentToOtherSpheres : Prop
  /-- The side length of the tetrahedron is 1 --/
  sideLength_eq_one : sideLength = 1
  /-- There are exactly four spheres --/
  numSpheres_eq_four : numSpheres = 4

/-- The theorem stating the radius of the spheres in the tetrahedron --/
theorem sphere_radius_in_tetrahedron (t : TetrahedronWithSpheres) :
  t.sphereRadius = (Real.sqrt 6 - 1) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_tetrahedron_l2692_269262


namespace NUMINAMATH_CALUDE_unique_base_number_l2692_269234

theorem unique_base_number : ∃! (x : ℕ), x < 6 ∧ x^23 % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_number_l2692_269234


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l2692_269287

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 400 ∧ 
  total_time = 30 ∧ 
  second_half_speed = 10 →
  (total_distance / 2) / (total_time - (total_distance / 2) / second_half_speed) = 20 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l2692_269287


namespace NUMINAMATH_CALUDE_nathan_harvest_earnings_is_186_l2692_269273

/-- Calculates the total earnings from Nathan's harvest --/
def nathan_harvest_earnings : ℕ :=
  let strawberry_plants : ℕ := 5
  let tomato_plants : ℕ := 7
  let strawberries_per_plant : ℕ := 14
  let tomatoes_per_plant : ℕ := 16
  let fruits_per_basket : ℕ := 7
  let price_strawberry_basket : ℕ := 9
  let price_tomato_basket : ℕ := 6

  let total_strawberries : ℕ := strawberry_plants * strawberries_per_plant
  let total_tomatoes : ℕ := tomato_plants * tomatoes_per_plant

  let strawberry_baskets : ℕ := total_strawberries / fruits_per_basket
  let tomato_baskets : ℕ := total_tomatoes / fruits_per_basket

  let earnings_strawberries : ℕ := strawberry_baskets * price_strawberry_basket
  let earnings_tomatoes : ℕ := tomato_baskets * price_tomato_basket

  earnings_strawberries + earnings_tomatoes

theorem nathan_harvest_earnings_is_186 : nathan_harvest_earnings = 186 := by
  sorry

end NUMINAMATH_CALUDE_nathan_harvest_earnings_is_186_l2692_269273


namespace NUMINAMATH_CALUDE_point_c_coordinates_l2692_269257

/-- Point with x and y coordinates -/
structure Point where
  x : ℚ
  y : ℚ

/-- Distance between two points -/
def distance (p q : Point) : ℚ :=
  ((p.x - q.x)^2 + (p.y - q.y)^2).sqrt

/-- Check if a point is on a line segment -/
def isOnSegment (p q r : Point) : Prop :=
  distance p r + distance r q = distance p q

theorem point_c_coordinates :
  let a : Point := ⟨-3, 2⟩
  let b : Point := ⟨5, 10⟩
  ∀ c : Point,
    isOnSegment a c b →
    distance a c = 2 * distance c b →
    c = ⟨7/3, 22/3⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l2692_269257


namespace NUMINAMATH_CALUDE_product_A_sample_size_l2692_269228

/-- Represents the ratio of quantities for products A, B, and C -/
def productRatio : Fin 3 → ℕ
| 0 => 2  -- Product A
| 1 => 3  -- Product B
| 2 => 5  -- Product C
| _ => 0  -- Unreachable case

/-- The total sample size -/
def sampleSize : ℕ := 80

/-- Calculates the number of items for a given product in the sample -/
def itemsInSample (product : Fin 3) : ℕ :=
  (sampleSize * productRatio product) / (productRatio 0 + productRatio 1 + productRatio 2)

theorem product_A_sample_size :
  itemsInSample 0 = 16 := by sorry

end NUMINAMATH_CALUDE_product_A_sample_size_l2692_269228


namespace NUMINAMATH_CALUDE_quadratic_function_ratio_bound_l2692_269285

/-- A quadratic function f(x) = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The function value at x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The derivative of the quadratic function -/
def QuadraticFunction.derivative (f : QuadraticFunction) (x : ℝ) : ℝ :=
  2 * f.a * x + f.b

theorem quadratic_function_ratio_bound (f : QuadraticFunction) 
    (h1 : f.derivative 0 > 0)
    (h2 : ∀ x : ℝ, f.eval x ≥ 0) :
    f.eval 1 / f.derivative 0 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_ratio_bound_l2692_269285


namespace NUMINAMATH_CALUDE_expression_value_approx_l2692_269291

def x : Real := 2.2
def a : Real := 3.6
def b : Real := 0.48
def c : Real := 2.50
def d : Real := 0.12
def e : Real := 0.09
def f : Real := 0.5

theorem expression_value_approx :
  ∃ (ε : Real), ε > 0 ∧ ε < 0.01 ∧ 
  |3 * x * ((a^2 * b * Real.log c) / (Real.sqrt d * Real.sin e * Real.log f)) + 720.72| < ε :=
by sorry

end NUMINAMATH_CALUDE_expression_value_approx_l2692_269291


namespace NUMINAMATH_CALUDE_worker_a_time_l2692_269203

theorem worker_a_time (worker_b_time worker_ab_time : ℝ) 
  (hb : worker_b_time = 15)
  (hab : worker_ab_time = 20 / 3) : 
  ∃ worker_a_time : ℝ, 
    worker_a_time = 12 ∧ 
    1 / worker_a_time + 1 / worker_b_time = 1 / worker_ab_time :=
by sorry

end NUMINAMATH_CALUDE_worker_a_time_l2692_269203


namespace NUMINAMATH_CALUDE_loan_duration_for_b_l2692_269271

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem loan_duration_for_b (principal_b principal_c duration_c total_interest : ℝ) 
  (rate : ℝ) (h1 : principal_b = 5000)
  (h2 : principal_c = 3000)
  (h3 : duration_c = 4)
  (h4 : rate = 0.1)
  (h5 : simple_interest principal_b rate (duration_b) + 
        simple_interest principal_c rate duration_c = total_interest)
  (h6 : total_interest = 2200) :
  duration_b = 2 := by
  sorry

#check loan_duration_for_b

end NUMINAMATH_CALUDE_loan_duration_for_b_l2692_269271


namespace NUMINAMATH_CALUDE_more_students_than_rabbits_l2692_269299

theorem more_students_than_rabbits : 
  let num_classrooms : ℕ := 6
  let students_per_classroom : ℕ := 22
  let rabbits_per_classroom : ℕ := 4
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_rabbits : ℕ := num_classrooms * rabbits_per_classroom
  total_students - total_rabbits = 108 := by
sorry

end NUMINAMATH_CALUDE_more_students_than_rabbits_l2692_269299


namespace NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l2692_269207

/-- The average speed of a car traveling different distances in two hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : 
  d1 > 0 → d2 > 0 → (d1 + d2) / 2 = (d1 + d2) / 2 := by
  sorry

/-- The average speed of a car traveling 50 km in the first hour and 60 km in the second hour is 55 km/h -/
theorem car_average_speed : (50 + 60) / 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l2692_269207


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2692_269201

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a * r^4 / (1 - r)) = (a / (1 - r)) / 81 → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2692_269201


namespace NUMINAMATH_CALUDE_complement_of_A_l2692_269200

def A : Set ℝ := {x | |x - 1| > 2}

theorem complement_of_A : 
  (Set.univ \ A) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2692_269200


namespace NUMINAMATH_CALUDE_one_slice_left_l2692_269236

/-- Represents the number of slices in a whole pizza after cutting -/
def total_slices : ℕ := 8

/-- Represents the number of friends who receive 1 slice each -/
def friends_one_slice : ℕ := 3

/-- Represents the number of friends who receive 2 slices each -/
def friends_two_slices : ℕ := 2

/-- Represents the number of slices given to friends -/
def slices_given : ℕ := friends_one_slice * 1 + friends_two_slices * 2

/-- Theorem stating that there is 1 slice left after distribution -/
theorem one_slice_left : total_slices - slices_given = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_slice_left_l2692_269236


namespace NUMINAMATH_CALUDE_cricket_target_runs_l2692_269242

/-- Calculates the target number of runs in a cricket game given specific conditions -/
theorem cricket_target_runs (total_overs run_rate_first_12 run_rate_remaining_38 : ℝ) 
  (h1 : total_overs = 50)
  (h2 : run_rate_first_12 = 4.5)
  (h3 : run_rate_remaining_38 = 8.052631578947368) : 
  ∃ (target : ℕ), target = 360 ∧ 
  target = ⌊run_rate_first_12 * 12 + run_rate_remaining_38 * (total_overs - 12)⌋ := by
  sorry

#check cricket_target_runs

end NUMINAMATH_CALUDE_cricket_target_runs_l2692_269242


namespace NUMINAMATH_CALUDE_min_n_greater_than_T10_plus_1013_l2692_269276

def T (n : ℕ) : ℚ := n + 1 - (1 / 2^n)

theorem min_n_greater_than_T10_plus_1013 :
  (∀ n : ℕ, n > T 10 + 1013 → n ≥ 1024) ∧
  (∃ n : ℕ, n > T 10 + 1013 ∧ n = 1024) :=
sorry

end NUMINAMATH_CALUDE_min_n_greater_than_T10_plus_1013_l2692_269276


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l2692_269238

theorem power_zero_eq_one (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l2692_269238


namespace NUMINAMATH_CALUDE_bike_race_distance_difference_l2692_269294

/-- Represents a cyclist with their distance traveled and time taken -/
structure Cyclist where
  distance : ℝ
  time : ℝ

/-- The difference in distance traveled between two cyclists -/
def distanceDifference (c1 c2 : Cyclist) : ℝ :=
  c1.distance - c2.distance

theorem bike_race_distance_difference :
  let carlos : Cyclist := { distance := 70, time := 5 }
  let dana : Cyclist := { distance := 50, time := 5 }
  distanceDifference carlos dana = 20 := by
  sorry

end NUMINAMATH_CALUDE_bike_race_distance_difference_l2692_269294


namespace NUMINAMATH_CALUDE_cost_effective_flower_purchase_l2692_269254

/-- Represents the cost-effective flower purchasing problem --/
theorem cost_effective_flower_purchase
  (total_flowers : ℕ)
  (carnation_price lily_price : ℚ)
  (h_total : total_flowers = 300)
  (h_carnation_price : carnation_price = 5)
  (h_lily_price : lily_price = 10)
  : ∃ (carnations lilies : ℕ),
    carnations + lilies = total_flowers ∧
    carnations ≤ 2 * lilies ∧
    ∀ (c l : ℕ),
      c + l = total_flowers →
      c ≤ 2 * l →
      carnation_price * carnations + lily_price * lilies ≤
      carnation_price * c + lily_price * l ∧
    carnations = 200 ∧
    lilies = 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_effective_flower_purchase_l2692_269254


namespace NUMINAMATH_CALUDE_share_distribution_l2692_269274

theorem share_distribution (x y z : ℚ) (a : ℚ) : 
  (x + y + z = 156) →  -- total amount
  (y = 36) →           -- y's share
  (z = x * (1/2)) →    -- z gets 50 paisa for each rupee x gets
  (y = x * a) →        -- y gets 'a' for each rupee x gets
  (a = 9/20) := by
    sorry

end NUMINAMATH_CALUDE_share_distribution_l2692_269274


namespace NUMINAMATH_CALUDE_intersection_complement_proof_l2692_269295

def U : Set Nat := {1, 2, 3, 4}

theorem intersection_complement_proof
  (A B : Set Nat)
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : (U \ (A ∪ B)) = {4})
  (h4 : B = {1, 2}) :
  A ∩ (U \ B) = {3} :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_proof_l2692_269295


namespace NUMINAMATH_CALUDE_base_8_4513_equals_2379_l2692_269259

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4513_equals_2379 :
  base_8_to_10 [3, 1, 5, 4] = 2379 := by
  sorry

end NUMINAMATH_CALUDE_base_8_4513_equals_2379_l2692_269259


namespace NUMINAMATH_CALUDE_arrangements_with_space_theorem_l2692_269248

/-- The number of arrangements of 6 people in a row where person A and person B
    have at least one person between them. -/
def arrangements_with_space_between (total_arrangements : ℕ) 
                                    (adjacent_arrangements : ℕ) : ℕ :=
  total_arrangements - adjacent_arrangements

/-- Theorem stating that the number of arrangements of 6 people in a row
    where person A and person B have at least one person between them is 480. -/
theorem arrangements_with_space_theorem :
  arrangements_with_space_between 720 240 = 480 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_space_theorem_l2692_269248


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l2692_269230

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial : 
  ∃ (k : ℕ), k = 287 ∧ 
  (∀ (m : ℕ), 1729^m ∣ factorial 1729 → m ≤ k) ∧
  (1729^k ∣ factorial 1729) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l2692_269230


namespace NUMINAMATH_CALUDE_gcd_problem_l2692_269288

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) :
  Int.gcd (4 * b ^ 2 + 35 * b + 72) (3 * b + 8) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2692_269288


namespace NUMINAMATH_CALUDE_expression_evaluation_l2692_269222

theorem expression_evaluation (a b c d : ℝ) :
  d = c + 5 →
  c = b - 8 →
  b = a + 3 →
  a = 3 →
  a - 1 ≠ 0 →
  d - 6 ≠ 0 →
  c + 4 ≠ 0 →
  ((a + 3) / (a - 1)) * ((d - 3) / (d - 6)) * ((c + 9) / (c + 4)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2692_269222


namespace NUMINAMATH_CALUDE_janes_cans_l2692_269227

theorem janes_cans (total_seeds : ℕ) (seeds_per_can : ℕ) (h1 : total_seeds = 54) (h2 : seeds_per_can = 6) :
  total_seeds / seeds_per_can = 9 :=
by sorry

end NUMINAMATH_CALUDE_janes_cans_l2692_269227


namespace NUMINAMATH_CALUDE_largest_prime_factor_l2692_269290

/-- The binary to decimal conversion for 10010000 -/
def binary_to_decimal_1 : Nat := 144

/-- The binary to decimal conversion for 100100000 -/
def binary_to_decimal_2 : Nat := 288

/-- The function to check if a number is prime -/
def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem stating that 3 is the largest prime factor of both numbers -/
theorem largest_prime_factor :
  ∃ (p : Nat), is_prime p ∧ 
    p ∣ binary_to_decimal_1 ∧ 
    p ∣ binary_to_decimal_2 ∧
    ∀ (q : Nat), is_prime q → 
      q ∣ binary_to_decimal_1 → 
      q ∣ binary_to_decimal_2 → 
      q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l2692_269290


namespace NUMINAMATH_CALUDE_cubic_polynomial_coefficient_expression_l2692_269258

/-- Represents a cubic polynomial of the form ax^3 + bx^2 + cx + d -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluates the cubic polynomial at a given x -/
def CubicPolynomial.evaluate (p : CubicPolynomial) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The specific cubic polynomial f(x) = 2x^3 - 3x^2 + 5x - 7 -/
def f : CubicPolynomial :=
  { a := 2, b := -3, c := 5, d := -7 }

theorem cubic_polynomial_coefficient_expression :
  16 * f.a - 9 * f.b + 3 * f.c - 2 * f.d = 88 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_coefficient_expression_l2692_269258


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l2692_269246

/-- The number of letters in the word "BANANA" -/
def total_letters : Nat := 6

/-- The number of occurrences of 'A' in "BANANA" -/
def count_A : Nat := 3

/-- The number of occurrences of 'N' in "BANANA" -/
def count_N : Nat := 2

/-- The number of occurrences of 'B' in "BANANA" -/
def count_B : Nat := 1

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : Nat := Nat.factorial total_letters / (Nat.factorial count_A * Nat.factorial count_N)

theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l2692_269246


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l2692_269252

/-- A rectangular prism is a three-dimensional shape with rectangular faces. -/
structure RectangularPrism where
  edges : Nat
  corners : Nat
  faces : Nat

/-- The properties of a standard rectangular prism. -/
def standardPrism : RectangularPrism :=
  { edges := 12
  , corners := 8
  , faces := 6 }

/-- The sum of edges, corners, and faces of a rectangular prism is 26. -/
theorem rectangular_prism_sum :
  standardPrism.edges + standardPrism.corners + standardPrism.faces = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l2692_269252


namespace NUMINAMATH_CALUDE_same_conclusion_from_true_and_false_l2692_269277

theorem same_conclusion_from_true_and_false :
  ∃ (A : Prop) (T F : Prop), T ∧ ¬F ∧ (T → A) ∧ (F → A) := by
  sorry

end NUMINAMATH_CALUDE_same_conclusion_from_true_and_false_l2692_269277


namespace NUMINAMATH_CALUDE_chicken_egg_problem_l2692_269250

theorem chicken_egg_problem (initial_eggs : ℕ) (used_eggs : ℕ) (eggs_per_chicken : ℕ) (final_eggs : ℕ) : 
  initial_eggs = 10 →
  used_eggs = 5 →
  eggs_per_chicken = 3 →
  final_eggs = 11 →
  (final_eggs - (initial_eggs - used_eggs)) / eggs_per_chicken = 2 :=
by sorry

end NUMINAMATH_CALUDE_chicken_egg_problem_l2692_269250


namespace NUMINAMATH_CALUDE_octagon_area_is_225_l2692_269237

-- Define the triangle and circle
structure Triangle :=
  (P Q R : ℝ × ℝ)

def circumradius : ℝ := 10

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ := 45

-- Define the points P', Q', R' as intersections of perpendicular bisectors with circumcircle
def P' (t : Triangle) : ℝ × ℝ := sorry
def Q' (t : Triangle) : ℝ × ℝ := sorry
def R' (t : Triangle) : ℝ × ℝ := sorry

-- Define S as reflection of circumcenter over PQ
def S (t : Triangle) : ℝ × ℝ := sorry

-- Define the area of the octagon
def octagon_area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem octagon_area_is_225 (t : Triangle) :
  octagon_area t = 225 := by sorry

end NUMINAMATH_CALUDE_octagon_area_is_225_l2692_269237


namespace NUMINAMATH_CALUDE_vector_problem_l2692_269231

/-- Given two vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![2, -2]

/-- Define vector c as a linear combination of a and b -/
def c : Fin 2 → ℝ := λ i ↦ 4 * a i + b i

/-- The dot product of two vectors -/
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

theorem vector_problem :
  (dot_product b c • a = 0) ∧
  (dot_product a (a + (5/2 • b)) = 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2692_269231


namespace NUMINAMATH_CALUDE_equation_positive_roots_l2692_269224

theorem equation_positive_roots (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ a * |x| + |x + a| = 0) ↔ -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_roots_l2692_269224


namespace NUMINAMATH_CALUDE_apple_trees_count_l2692_269235

/-- The number of apple trees in the orchard -/
def num_apple_trees : ℕ := 30

/-- The yield of apples per apple tree in kg -/
def apple_yield : ℕ := 150

/-- The number of peach trees in the orchard -/
def num_peach_trees : ℕ := 45

/-- The average yield of peaches per peach tree in kg -/
def peach_yield : ℕ := 65

/-- The total mass of fruit harvested in kg -/
def total_harvest : ℕ := 7425

/-- Theorem stating that the number of apple trees is correct given the conditions -/
theorem apple_trees_count :
  num_apple_trees * apple_yield + num_peach_trees * peach_yield = total_harvest :=
by sorry


end NUMINAMATH_CALUDE_apple_trees_count_l2692_269235


namespace NUMINAMATH_CALUDE_ferry_problem_l2692_269281

/-- Represents the ferry problem and proves the speed of the current and distance between docks. -/
theorem ferry_problem (still_water_speed time_against time_with : ℝ) 
  (h1 : still_water_speed = 12)
  (h2 : time_against = 10)
  (h3 : time_with = 6) :
  ∃ (current_speed distance : ℝ),
    current_speed = 3 ∧
    distance = 90 ∧
    time_with * (still_water_speed + current_speed) = time_against * (still_water_speed - current_speed) ∧
    distance = (still_water_speed + current_speed) * time_with :=
by
  sorry


end NUMINAMATH_CALUDE_ferry_problem_l2692_269281


namespace NUMINAMATH_CALUDE_exist_good_coloring_l2692_269215

/-- The set of colors --/
inductive Color
| red
| white

/-- The type of coloring functions --/
def Coloring := Fin 2017 → Color

/-- Checks if a sequence is an arithmetic progression --/
def isArithmeticSequence (s : Fin n → Fin 2017) : Prop :=
  ∃ a d : ℕ, ∀ i : Fin n, s i = a + i.val * d

/-- The main theorem --/
theorem exist_good_coloring (n : ℕ) (h : n ≥ 18) :
  ∃ f : Coloring, ∀ s : Fin n → Fin 2017, 
    isArithmeticSequence s → 
    ∃ i j : Fin n, f (s i) ≠ f (s j) :=
sorry

end NUMINAMATH_CALUDE_exist_good_coloring_l2692_269215


namespace NUMINAMATH_CALUDE_parity_of_solutions_l2692_269284

theorem parity_of_solutions (n m p q : ℤ) : 
  (∃ k : ℤ, n = 2 * k) →  -- n is even
  (∃ k : ℤ, m = 2 * k + 1) →  -- m is odd
  p - 1988 * q = n →  -- first equation
  11 * p + 27 * q = m →  -- second equation
  (∃ k : ℤ, p = 2 * k) ∧ (∃ k : ℤ, q = 2 * k + 1) :=  -- p is even and q is odd
by sorry

end NUMINAMATH_CALUDE_parity_of_solutions_l2692_269284


namespace NUMINAMATH_CALUDE_a_range_proof_l2692_269272

-- Define the propositions p and q
def p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4

def q (x : ℝ) : Prop := (x - 1) * (x - 3) < 0

-- Define the range of a
def a_range (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 5

-- State the theorem
theorem a_range_proof :
  (∀ x a : ℝ, q x → p x a) ∧  -- q is sufficient for p
  (∃ x a : ℝ, p x a ∧ ¬(q x)) ∧  -- q is not necessary for p
  (∀ a : ℝ, a_range a ↔ ∀ x : ℝ, q x → p x a) :=
by sorry

end NUMINAMATH_CALUDE_a_range_proof_l2692_269272


namespace NUMINAMATH_CALUDE_apples_remaining_l2692_269205

/-- The number of apples left after picking and eating -/
def applesLeft (mikeApples nancyApples keithApples : Float) : Float :=
  mikeApples + nancyApples - keithApples

theorem apples_remaining :
  applesLeft 7.0 3.0 6.0 = 4.0 := by
  sorry

#eval applesLeft 7.0 3.0 6.0

end NUMINAMATH_CALUDE_apples_remaining_l2692_269205


namespace NUMINAMATH_CALUDE_largest_number_with_constraints_l2692_269251

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 2

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_constraints :
  ∀ n : ℕ, 
    is_valid_number n ∧ 
    digit_sum n = 20 →
    n ≤ 44444 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_constraints_l2692_269251


namespace NUMINAMATH_CALUDE_points_earned_in_level_l2692_269267

/-- Calculates the points earned in a video game level -/
theorem points_earned_in_level 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (enemies_not_destroyed : ℕ) : 
  points_per_enemy = 9 →
  total_enemies = 11 →
  enemies_not_destroyed = 3 →
  (total_enemies - enemies_not_destroyed) * points_per_enemy = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_points_earned_in_level_l2692_269267


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l2692_269240

theorem square_garden_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 450 →
  area = side^2 →
  perimeter = 4 * side →
  perimeter = 60 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l2692_269240


namespace NUMINAMATH_CALUDE_probability_two_aces_l2692_269219

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Represents the total number of cards in the mixed deck -/
def TotalCards : ℕ := 2 * StandardDeck

/-- Represents the total number of aces in the mixed deck -/
def TotalAces : ℕ := 2 * AcesInDeck

/-- The probability of drawing two aces consecutively from a mixed deck of 104 cards -/
theorem probability_two_aces (StandardDeck AcesInDeck TotalCards TotalAces : ℕ) 
  (h1 : TotalCards = 2 * StandardDeck)
  (h2 : TotalAces = 2 * AcesInDeck)
  (h3 : StandardDeck = 52)
  (h4 : AcesInDeck = 4) :
  (TotalAces : ℚ) / TotalCards * (TotalAces - 1) / (TotalCards - 1) = 7 / 1339 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_aces_l2692_269219


namespace NUMINAMATH_CALUDE_remaining_money_theorem_l2692_269269

def calculate_remaining_money (initial_amount : ℚ) : ℚ :=
  let day1_remaining := initial_amount * (1 - 3/5)
  let day2_remaining := day1_remaining * (1 - 7/12)
  let day3_remaining := day2_remaining * (1 - 2/3)
  let day4_remaining := day3_remaining * (1 - 1/6)
  let day5_remaining := day4_remaining * (1 - 5/8)
  let day6_remaining := day5_remaining * (1 - 3/5)
  day6_remaining

theorem remaining_money_theorem :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  abs (calculate_remaining_money 500 - 347/100) < ε := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_theorem_l2692_269269


namespace NUMINAMATH_CALUDE_winter_carnival_participants_l2692_269226

theorem winter_carnival_participants (total_students : ℕ) (total_participants : ℕ) 
  (girls : ℕ) (boys : ℕ) (h1 : total_students = 1500) 
  (h2 : total_participants = 900) (h3 : girls + boys = total_students) 
  (h4 : (3 * girls / 4 : ℚ) + (2 * boys / 3 : ℚ) = total_participants) : 
  3 * girls / 4 = 900 := by
  sorry

end NUMINAMATH_CALUDE_winter_carnival_participants_l2692_269226


namespace NUMINAMATH_CALUDE_pet_store_cages_l2692_269296

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : puppies_per_cage = 5) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2692_269296


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l2692_269278

/-- The jumping contest between a grasshopper and a frog -/
theorem grasshopper_jump_distance 
  (frog_jump : ℕ) 
  (frog_grasshopper_difference : ℕ) 
  (h1 : frog_jump = 12)
  (h2 : frog_jump = frog_grasshopper_difference + grasshopper_jump) :
  grasshopper_jump = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l2692_269278


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l2692_269264

theorem largest_solution_of_equation (y : ℝ) :
  (3 * y^2 + 18 * y - 90 = y * (y + 17)) →
  y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l2692_269264


namespace NUMINAMATH_CALUDE_ebook_reader_difference_l2692_269253

theorem ebook_reader_difference (anna_count john_original_count : ℕ) : 
  anna_count = 50 →
  john_original_count < anna_count →
  john_original_count + anna_count = 82 + 3 →
  anna_count - john_original_count = 15 := by
sorry

end NUMINAMATH_CALUDE_ebook_reader_difference_l2692_269253


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2692_269232

theorem quadratic_minimum (x : ℝ) (h : x ≥ 0) : x^2 + 13*x + 4 ≥ 4 ∧ ∃ y ≥ 0, y^2 + 13*y + 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2692_269232


namespace NUMINAMATH_CALUDE_angle_property_equivalence_l2692_269210

theorem angle_property_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by sorry

end NUMINAMATH_CALUDE_angle_property_equivalence_l2692_269210


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2692_269256

theorem trigonometric_equation_solution (x : ℝ) :
  8.471 * (3 * Real.tan x - Real.tan x ^ 3) / (2 - 1 / Real.cos x ^ 2) = 
  (4 + 2 * Real.cos (6 * x / 5)) / (Real.cos (3 * x) + Real.cos x) ↔
  ∃ k : ℤ, x = 5 * π / 6 + 10 * π * k / 3 ∧ ¬∃ t : ℤ, k = 2 + 3 * t :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2692_269256


namespace NUMINAMATH_CALUDE_hannah_total_cost_l2692_269265

/-- The total cost of Hannah's purchase of sweatshirts and T-shirts -/
def total_cost (num_sweatshirts num_tshirts sweatshirt_price tshirt_price : ℕ) : ℕ :=
  num_sweatshirts * sweatshirt_price + num_tshirts * tshirt_price

/-- Theorem stating that Hannah's total cost is $65 -/
theorem hannah_total_cost :
  total_cost 3 2 15 10 = 65 := by
  sorry

end NUMINAMATH_CALUDE_hannah_total_cost_l2692_269265


namespace NUMINAMATH_CALUDE_cubic_equation_root_b_value_l2692_269266

theorem cubic_equation_root_b_value :
  ∀ (a b : ℚ),
  (∃ (x : ℝ), x = 2 + Real.sqrt 3 ∧ x^3 + a*x^2 + b*x + 10 = 0) →
  b = -39 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_root_b_value_l2692_269266


namespace NUMINAMATH_CALUDE_max_value_of_f_l2692_269283

-- Define the function
def f (x : ℝ) : ℝ := (x + 1)^2 - 4

-- State the theorem
theorem max_value_of_f : 
  ∃ (M : ℝ), M = 5 ∧ ∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2692_269283


namespace NUMINAMATH_CALUDE_corn_height_after_three_weeks_l2692_269208

def corn_growth (first_week_growth : ℝ) (second_week_multiplier : ℝ) (third_week_multiplier : ℝ) : ℝ :=
  let second_week_growth := first_week_growth * second_week_multiplier
  let third_week_growth := second_week_growth * third_week_multiplier
  first_week_growth + second_week_growth + third_week_growth

theorem corn_height_after_three_weeks :
  corn_growth 2 2 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_corn_height_after_three_weeks_l2692_269208


namespace NUMINAMATH_CALUDE_max_nonmanagers_for_nine_managers_l2692_269298

/-- Represents a department in the corporation -/
structure Department where
  managers : ℕ
  nonManagers : ℕ

/-- The conditions for a valid department -/
def isValidDepartment (d : Department) : Prop :=
  d.managers > 0 ∧
  d.managers * 37 > 7 * d.nonManagers ∧
  d.managers ≥ 5 ∧
  d.managers + d.nonManagers ≤ 300 ∧
  d.managers = (d.managers + d.nonManagers) * 12 / 100

theorem max_nonmanagers_for_nine_managers :
  ∀ d : Department,
    isValidDepartment d →
    d.managers = 9 →
    d.nonManagers ≤ 66 :=
by sorry

end NUMINAMATH_CALUDE_max_nonmanagers_for_nine_managers_l2692_269298


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2692_269280

theorem quadratic_equation_solution (k : ℝ) (x : ℝ) :
  k * x^2 - (3 * k + 3) * x + 2 * k + 6 = 0 →
  (k = 0 → x = 2) ∧
  (k ≠ 0 → (x = 2 ∨ x = 1 + 3 / k)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2692_269280


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2692_269212

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) * a m = a n * a (m + 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36 →
  a 5 + a 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2692_269212


namespace NUMINAMATH_CALUDE_christmas_monday_implies_jan25_thursday_l2692_269244

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in a year -/
structure Date where
  month : Nat
  day : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (d : Date) : DayOfWeek :=
  sorry

/-- Function to get the date of the next year -/
def nextYearDate (d : Date) : Date :=
  sorry

theorem christmas_monday_implies_jan25_thursday
  (h : dayOfWeek ⟨12, 25⟩ = DayOfWeek.Monday) :
  dayOfWeek (nextYearDate ⟨1, 25⟩) = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_christmas_monday_implies_jan25_thursday_l2692_269244


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2692_269245

theorem polynomial_factorization (a : ℝ) : 
  -3*a + 12*a^2 - 12*a^3 = -3*a*(1 - 2*a)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2692_269245


namespace NUMINAMATH_CALUDE_percentage_of_muslim_boys_l2692_269261

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 400 →
  hindu_percentage = 28 / 100 →
  sikh_percentage = 10 / 100 →
  other_boys = 72 →
  (total_boys - (hindu_percentage * total_boys + sikh_percentage * total_boys + other_boys : ℚ)) / total_boys = 44 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_muslim_boys_l2692_269261


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_l2692_269233

/-- The area of a regular octagon inscribed in a circle with area 400π square units is 800√2 square units. -/
theorem inscribed_octagon_area (circle_area : ℝ) (octagon_area : ℝ) :
  circle_area = 400 * Real.pi →
  octagon_area = 8 * (1 / 2 * (20^2) * Real.sin (π / 4)) →
  octagon_area = 800 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_l2692_269233


namespace NUMINAMATH_CALUDE_contestant_selection_probabilities_l2692_269292

/-- Represents the probability of selecting two females from a group of contestants. -/
def prob_two_females (total : ℕ) (females : ℕ) : ℚ :=
  (females.choose 2 : ℚ) / (total.choose 2 : ℚ)

/-- Represents the probability of selecting at least one male from a group of contestants. -/
def prob_at_least_one_male (total : ℕ) (females : ℕ) : ℚ :=
  1 - prob_two_females total females

/-- Theorem stating the probabilities for selecting contestants from a group of 8 with 5 females and 3 males. -/
theorem contestant_selection_probabilities :
  let total := 8
  let females := 5
  prob_two_females total females = 5 / 14 ∧
  prob_at_least_one_male total females = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_contestant_selection_probabilities_l2692_269292


namespace NUMINAMATH_CALUDE_scientific_notation_of_58_billion_l2692_269213

theorem scientific_notation_of_58_billion :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 58000000000 = a * (10 : ℝ) ^ n ∧ a = 5.8 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_58_billion_l2692_269213


namespace NUMINAMATH_CALUDE_total_grapes_is_83_l2692_269282

/-- The number of grapes in Rob's bowl -/
def rob_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allie_grapes : ℕ := rob_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyn_grapes : ℕ := allie_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := rob_grapes + allie_grapes + allyn_grapes

theorem total_grapes_is_83 : total_grapes = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_grapes_is_83_l2692_269282


namespace NUMINAMATH_CALUDE_line_equation_l2692_269286

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

/-- The point through which the line passes -/
def point : ℝ × ℝ := (2, 1)

/-- Predicate to check if a point is on the line -/
def on_line (m b x y : ℝ) : Prop := y = m * x + b

/-- Predicate to check if a point bisects a chord -/
def bisects_chord (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    on_line 8 (-15) x₁ y₁ ∧
    on_line 8 (-15) x₂ y₂ ∧
    bisects_chord point.1 point.2 x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2692_269286


namespace NUMINAMATH_CALUDE_complement_union_problem_l2692_269249

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 2}
def B : Set Int := {-1, 0, 1}

theorem complement_union_problem : (U \ B) ∪ A = {-2, -1, 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l2692_269249


namespace NUMINAMATH_CALUDE_prob_first_class_correct_l2692_269204

/-- Represents the two types of items -/
inductive ItemClass
| First
| Second

/-- Represents the two trucks -/
inductive Truck
| A
| B

/-- The total number of items -/
def totalItems : Nat := 10

/-- The number of items in each truck -/
def truckItems : Truck → ItemClass → Nat
| Truck.A, ItemClass.First => 2
| Truck.A, ItemClass.Second => 2
| Truck.B, ItemClass.First => 4
| Truck.B, ItemClass.Second => 2

/-- The number of broken items per truck -/
def brokenItemsPerTruck : Nat := 1

/-- The number of remaining items after breakage -/
def remainingItems : Nat := totalItems - 2 * brokenItemsPerTruck

/-- The probability of selecting a first-class item from the remaining items -/
def probFirstClass : Rat := 29 / 48

theorem prob_first_class_correct :
  probFirstClass = 29 / 48 := by sorry

end NUMINAMATH_CALUDE_prob_first_class_correct_l2692_269204


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l2692_269209

theorem sqrt_sum_squares_eq_sum (a b : ℝ) :
  Real.sqrt (a^2 + b^2) = a + b ↔ a * b = 0 ∧ a + b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l2692_269209


namespace NUMINAMATH_CALUDE_prime_sum_85_product_166_l2692_269216

theorem prime_sum_85_product_166 (p q : ℕ) (hp : Prime p) (hq : Prime q) (hsum : p + q = 85) :
  p * q = 166 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_85_product_166_l2692_269216


namespace NUMINAMATH_CALUDE_mud_efficacy_ratio_l2692_269243

/-- Represents the number of sprigs of mint in the original mud mixture -/
def original_mint_sprigs : ℕ := 3

/-- Represents the number of green tea leaves per sprig of mint -/
def tea_leaves_per_sprig : ℕ := 2

/-- Represents the number of green tea leaves needed in the new mud for the same efficacy -/
def new_mud_tea_leaves : ℕ := 12

/-- Calculates the ratio of efficacy of new mud to original mud -/
def efficacy_ratio : ℚ := 1 / 2

theorem mud_efficacy_ratio :
  efficacy_ratio = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_mud_efficacy_ratio_l2692_269243


namespace NUMINAMATH_CALUDE_total_pokemon_cards_l2692_269239

/-- The number of cards in one dozen -/
def cards_per_dozen : ℕ := 12

/-- The number of dozens each person has -/
def dozens_per_person : ℕ := 9

/-- The number of people -/
def number_of_people : ℕ := 4

/-- Theorem: The total number of Pokemon cards owned by 4 people, each having 9 dozen cards, is equal to 432 -/
theorem total_pokemon_cards : 
  (cards_per_dozen * dozens_per_person * number_of_people) = 432 := by
  sorry

end NUMINAMATH_CALUDE_total_pokemon_cards_l2692_269239


namespace NUMINAMATH_CALUDE_range_of_function_l2692_269217

theorem range_of_function (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) 
  (h4 : b = (1 + Real.sqrt 5) / 2 * a) : 
  ∃ (x : ℝ), (9 - 9 * Real.sqrt 5) / 32 < a * (b - 3/2) ∧ 
             a * (b - 3/2) < (Real.sqrt 5 - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l2692_269217


namespace NUMINAMATH_CALUDE_peter_winning_strategy_l2692_269202

open Set

/-- Represents a point on a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents a color (red or blue) -/
inductive Color
  | Red
  | Blue

/-- Function to check if two triangles are similar -/
def are_similar (t1 t2 : Triangle) : Prop :=
  sorry

/-- Function to check if all points in a set have the same color -/
def all_same_color (points : Set Point) (coloring : Point → Color) : Prop :=
  sorry

/-- Theorem stating that two points are sufficient for Peter's winning strategy -/
theorem peter_winning_strategy (original : Triangle) :
  ∃ (p1 p2 : Point), ∀ (coloring : Point → Color),
    ∃ (t : Triangle), are_similar t original ∧
      all_same_color {t.a, t.b, t.c} coloring :=
sorry

end NUMINAMATH_CALUDE_peter_winning_strategy_l2692_269202


namespace NUMINAMATH_CALUDE_B_is_smallest_l2692_269225

def A : ℕ := 32 + 7
def B : ℕ := 3 * 10 + 3
def C : ℕ := 50 - 9

theorem B_is_smallest : B ≤ A ∧ B ≤ C := by
  sorry

end NUMINAMATH_CALUDE_B_is_smallest_l2692_269225


namespace NUMINAMATH_CALUDE_cheapest_candle_combination_l2692_269218

/-- Represents a candle with its burning time and cost -/
structure Candle where
  burn_time : ℕ
  cost : ℕ

/-- Finds the minimum cost to measure exactly one minute using given candles -/
def min_cost_to_measure_one_minute (candles : List Candle) : ℕ :=
  sorry

/-- The problem statement -/
theorem cheapest_candle_combination :
  let big_candle : Candle := { burn_time := 16, cost := 16 }
  let small_candle : Candle := { burn_time := 7, cost := 7 }
  let candles : List Candle := [big_candle, small_candle]
  min_cost_to_measure_one_minute candles = 97 :=
sorry

end NUMINAMATH_CALUDE_cheapest_candle_combination_l2692_269218


namespace NUMINAMATH_CALUDE_arithmetic_progression_tenth_term_zero_l2692_269247

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.a + (n - 1 : ℝ) * ap.d

theorem arithmetic_progression_tenth_term_zero
  (ap : ArithmeticProgression)
  (h : ap.nthTerm 5 + ap.nthTerm 21 = ap.nthTerm 8 + ap.nthTerm 15 + ap.nthTerm 13) :
  ap.nthTerm 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_tenth_term_zero_l2692_269247


namespace NUMINAMATH_CALUDE_intersection_property_l2692_269241

/-- The curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 2*ρ*Real.cos θ - 3 = 0

/-- The line l in polar coordinates -/
def line_l (ρ θ : ℝ) (m : ℝ) : Prop :=
  ρ * (Real.cos θ + Real.sin θ) = m

/-- The theorem statement -/
theorem intersection_property (m : ℝ) :
  m > 0 →
  ∃ (ρ_A ρ_M ρ_N : ℝ),
    line_l ρ_A (π/4) m ∧
    curve_C ρ_M (π/4) ∧
    curve_C ρ_N (π/4) ∧
    ρ_A * ρ_M * ρ_N = 6 →
  m = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_property_l2692_269241


namespace NUMINAMATH_CALUDE_coefficient_sum_equality_l2692_269223

theorem coefficient_sum_equality (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_equality_l2692_269223


namespace NUMINAMATH_CALUDE_parabola_contradiction_l2692_269263

theorem parabola_contradiction (a b c : ℝ) : 
  ¬(((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c > 0)) ∧
    ((a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c < 0))) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_contradiction_l2692_269263


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l2692_269293

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_solution (a : ℕ → ℝ) (q : ℝ) 
  (h1 : geometric_sequence a q)
  (h2 : a 5 - a 1 = 15)
  (h3 : a 4 - a 2 = 6) :
  (q = 2 ∧ a 3 = 4) ∨ (q = 1/2 ∧ a 3 = -4) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l2692_269293


namespace NUMINAMATH_CALUDE_exists_line_and_circle_through_origin_l2692_269220

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define a line passing through (0, -2)
def line_through_point (k : ℝ) (x y : ℝ) : Prop := y = k * x - 2

-- Define two points on the intersection of the line and the ellipse
def intersection_points (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂ ∧
  line_through_point k x₁ y₁ ∧ line_through_point k x₂ y₂ ∧
  x₁ ≠ x₂

-- Define the condition for a circle with diameter AB passing through the origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- The main theorem
theorem exists_line_and_circle_through_origin :
  ∃ (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    intersection_points k x₁ y₁ x₂ y₂ ∧
    circle_through_origin x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_exists_line_and_circle_through_origin_l2692_269220


namespace NUMINAMATH_CALUDE_eight_digit_divisible_by_eleven_l2692_269275

theorem eight_digit_divisible_by_eleven (n : ℕ) : 
  n < 10 →
  (965 * 10^7 + n * 10^6 + 8 * 10^5 + 4 * 10^4 + 3 * 10^3 + 2 * 10^2) % 11 = 0 →
  n = 1 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_divisible_by_eleven_l2692_269275


namespace NUMINAMATH_CALUDE_office_employees_l2692_269221

theorem office_employees (total_employees : ℕ) 
  (h1 : (45 : ℚ) / 100 * total_employees = total_males)
  (h2 : (50 : ℚ) / 100 * total_males = males_50_and_above)
  (h3 : 1170 = total_males - males_50_and_above) :
  total_employees = 5200 :=
by sorry

end NUMINAMATH_CALUDE_office_employees_l2692_269221


namespace NUMINAMATH_CALUDE_f_inequality_solution_l2692_269229

noncomputable def f (a x : ℝ) : ℝ := |x - a| - |x + 3|

theorem f_inequality_solution (a : ℝ) :
  (a = -1 → {x : ℝ | f a x ≤ 1} = {x : ℝ | x ≥ -5/2}) ∧
  ({a : ℝ | ∀ x ∈ Set.Icc 0 3, f a x ≤ 4} = Set.Icc (-7) 7) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_l2692_269229


namespace NUMINAMATH_CALUDE_crayon_ratio_l2692_269260

def karen_crayons : ℕ := 128
def judah_crayons : ℕ := 8

def gilbert_crayons : ℕ := 4 * judah_crayons
def beatrice_crayons : ℕ := 2 * gilbert_crayons

theorem crayon_ratio :
  karen_crayons / beatrice_crayons = 2 := by
  sorry

end NUMINAMATH_CALUDE_crayon_ratio_l2692_269260
