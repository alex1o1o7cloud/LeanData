import Mathlib

namespace cyclist_speed_north_cyclist_speed_north_proof_l3626_362648

/-- The speed of the cyclist going north, given two cyclists starting from the same place
    in opposite directions, with one going south at 25 km/h, and they take 1.4285714285714286 hours
    to be 50 km apart. -/
theorem cyclist_speed_north : ℝ → Prop :=
  fun v : ℝ =>
    let south_speed : ℝ := 25
    let time : ℝ := 1.4285714285714286
    let distance : ℝ := 50
    v > 0 ∧ distance = (v + south_speed) * time → v = 10

/-- Proof of the cyclist_speed_north theorem -/
theorem cyclist_speed_north_proof : cyclist_speed_north 10 := by
  sorry

end cyclist_speed_north_cyclist_speed_north_proof_l3626_362648


namespace rhombus_area_l3626_362693

/-- The area of a rhombus with diagonals of 14 cm and 20 cm is 140 square centimeters. -/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 140 := by
sorry

end rhombus_area_l3626_362693


namespace son_age_l3626_362641

/-- Represents the ages of a father and son -/
structure Ages where
  father : ℕ
  son : ℕ

/-- The conditions of the age problem -/
def AgeConditions (ages : Ages) : Prop :=
  (ages.father + ages.son = 75) ∧
  (∃ (x : ℕ), ages.father = 8 * (ages.son - x) ∧ ages.father - x = ages.son)

/-- The theorem stating that under the given conditions, the son's age is 27 -/
theorem son_age (ages : Ages) (h : AgeConditions ages) : ages.son = 27 := by
  sorry

end son_age_l3626_362641


namespace rectangles_in_5x5_grid_l3626_362614

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of rectangles in a square grid -/
def rectanglesInGrid (n : ℕ) : ℕ := (choose n 2) ^ 2

theorem rectangles_in_5x5_grid :
  rectanglesInGrid 5 = 100 := by
  sorry

end rectangles_in_5x5_grid_l3626_362614


namespace buoy_distance_is_24_l3626_362670

/-- The distance between two consecutive buoys in the ocean -/
def buoy_distance (d1 d2 : ℝ) : ℝ := d2 - d1

/-- Theorem: The distance between two consecutive buoys is 24 meters -/
theorem buoy_distance_is_24 :
  let d1 := 72 -- distance of first buoy from beach
  let d2 := 96 -- distance of second buoy from beach
  buoy_distance d1 d2 = 24 := by sorry

end buoy_distance_is_24_l3626_362670


namespace coin_flip_probability_l3626_362626

theorem coin_flip_probability (n : ℕ) : 
  (n.choose 2 : ℚ) / 2^n = 1/8 → n = 5 := by
  sorry

end coin_flip_probability_l3626_362626


namespace optimal_plan_is_most_cost_effective_l3626_362688

/-- Represents a sewage treatment equipment model -/
structure EquipmentModel where
  price : ℕ  -- Price in million yuan
  capacity : ℕ  -- Capacity in tons/month

/-- Represents a purchasing plan -/
structure PurchasePlan where
  modelA : ℕ  -- Number of Model A units
  modelB : ℕ  -- Number of Model B units

def modelA : EquipmentModel := { price := 12, capacity := 240 }
def modelB : EquipmentModel := { price := 10, capacity := 200 }

def totalEquipment : ℕ := 10
def budgetConstraint : ℕ := 105
def minTreatmentCapacity : ℕ := 2040

def totalCost (plan : PurchasePlan) : ℕ :=
  plan.modelA * modelA.price + plan.modelB * modelB.price

def totalCapacity (plan : PurchasePlan) : ℕ :=
  plan.modelA * modelA.capacity + plan.modelB * modelB.capacity

def isValidPlan (plan : PurchasePlan) : Prop :=
  plan.modelA + plan.modelB = totalEquipment ∧
  totalCost plan ≤ budgetConstraint ∧
  totalCapacity plan ≥ minTreatmentCapacity

def optimalPlan : PurchasePlan := { modelA := 1, modelB := 9 }

theorem optimal_plan_is_most_cost_effective :
  isValidPlan optimalPlan ∧
  ∀ plan, isValidPlan plan → totalCost plan ≥ totalCost optimalPlan :=
by sorry

end optimal_plan_is_most_cost_effective_l3626_362688


namespace common_factor_implies_a_values_l3626_362652

theorem common_factor_implies_a_values (a : ℝ) :
  (∃ (p : ℝ) (A B : ℝ → ℝ), p ≠ 0 ∧
    (∀ x, x^3 - x - a = A x * (x + p)) ∧
    (∀ x, x^2 + x - a = B x * (x + p))) →
  (a = 0 ∨ a = 10 ∨ a = -2) :=
by sorry

end common_factor_implies_a_values_l3626_362652


namespace travel_equations_correct_l3626_362636

/-- Represents the travel scenario with bike riding and walking -/
structure TravelScenario where
  total_time : ℝ
  total_distance : ℝ
  bike_speed : ℝ
  walk_speed : ℝ
  bike_time : ℝ
  walk_time : ℝ

/-- The given travel scenario matches the system of equations -/
def scenario_matches_equations (s : TravelScenario) : Prop :=
  s.total_time = 1.5 ∧
  s.total_distance = 20 ∧
  s.bike_speed = 15 ∧
  s.walk_speed = 5 ∧
  s.bike_time + s.walk_time = s.total_time ∧
  s.bike_speed * s.bike_time + s.walk_speed * s.walk_time = s.total_distance

/-- The system of equations correctly represents the travel scenario -/
theorem travel_equations_correct (s : TravelScenario) :
  scenario_matches_equations s →
  s.bike_time + s.walk_time = 1.5 ∧
  15 * s.bike_time + 5 * s.walk_time = 20 :=
by sorry

end travel_equations_correct_l3626_362636


namespace bob_finishes_24_minutes_after_alice_l3626_362683

/-- Represents the race scenario -/
structure RaceScenario where
  distance : ℕ  -- Race distance in miles
  alice_speed : ℕ  -- Alice's speed in minutes per mile
  bob_speed : ℕ  -- Bob's speed in minutes per mile

/-- Calculates the time difference between Alice and Bob finishing the race -/
def finish_time_difference (race : RaceScenario) : ℕ :=
  race.distance * race.bob_speed - race.distance * race.alice_speed

/-- Theorem stating that in the given race scenario, Bob finishes 24 minutes after Alice -/
theorem bob_finishes_24_minutes_after_alice :
  let race := RaceScenario.mk 12 7 9
  finish_time_difference race = 24 := by
  sorry

end bob_finishes_24_minutes_after_alice_l3626_362683


namespace trapezium_marked_length_l3626_362610

/-- Represents an isosceles triangle ABC with base AC and equal sides AB and BC -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ

/-- Represents a trapezium AMNC formed from an isosceles triangle ABC -/
structure Trapezium (triangle : IsoscelesTriangle) where
  markedLength : ℝ
  perimeter : ℝ

/-- Theorem: In an isosceles triangle with base 12 and side 18, 
    if a trapezium is formed with perimeter 40, 
    then the marked length on each side is 6 -/
theorem trapezium_marked_length 
  (triangle : IsoscelesTriangle) 
  (trap : Trapezium triangle) : 
  triangle.base = 12 → 
  triangle.side = 18 → 
  trap.perimeter = 40 → 
  trap.markedLength = 6 := by
  sorry

end trapezium_marked_length_l3626_362610


namespace distance_to_CD_l3626_362630

/-- A square with semi-circle arcs -/
structure SquareWithArcs (s : ℝ) where
  -- Square ABCD
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Ensure it's a square with side length s
  square_side : dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ dist D A = s
  -- Semi-circle arcs
  arc_A : Set (ℝ × ℝ)
  arc_B : Set (ℝ × ℝ)
  -- Ensure arcs have correct radii and centers
  arc_A_def : arc_A = {p : ℝ × ℝ | dist p A = s/2 ∧ p.1 ≥ A.1 ∧ p.2 ≤ C.2}
  arc_B_def : arc_B = {p : ℝ × ℝ | dist p B = s/2 ∧ p.1 ≤ B.1 ∧ p.2 ≤ C.2}
  -- Intersection point X
  X : ℝ × ℝ
  X_def : X ∈ arc_A ∧ X ∈ arc_B

/-- The main theorem -/
theorem distance_to_CD (s : ℝ) (h : s > 0) (sq : SquareWithArcs s) :
  dist sq.X (sq.C.1, sq.X.2) = s :=
sorry

end distance_to_CD_l3626_362630


namespace least_k_for_error_bound_l3626_362684

-- Define the sequence u_k
def u : ℕ → ℚ
  | 0 => 1/3
  | k+1 => 2.5 * u k - 3 * (u k)^2

-- Define the limit L
noncomputable def L : ℚ := 2/5

-- Define the error bound
def error_bound : ℚ := 1 / 2^500

-- Theorem statement
theorem least_k_for_error_bound :
  ∃ k : ℕ, (∀ j : ℕ, j < k → |u j - L| > error_bound) ∧
           |u k - L| ≤ error_bound ∧
           k = 5 := by sorry

end least_k_for_error_bound_l3626_362684


namespace car_truck_difference_l3626_362621

theorem car_truck_difference (total_vehicles : ℕ) (trucks : ℕ) 
  (h1 : total_vehicles = 69) 
  (h2 : trucks = 21) : 
  total_vehicles - trucks - trucks = 27 := by
  sorry

end car_truck_difference_l3626_362621


namespace cube_as_difference_of_squares_l3626_362646

theorem cube_as_difference_of_squares (n : ℤ) (h : n > 1) :
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ n^3 = a^2 - b^2 := by
  sorry

end cube_as_difference_of_squares_l3626_362646


namespace base4_calculation_l3626_362627

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Multiplies two base 4 numbers --/
def multiplyBase4 (a b : ℕ) : ℕ := sorry

/-- Divides two base 4 numbers --/
def divideBase4 (a b : ℕ) : ℕ := sorry

/-- Subtracts two base 4 numbers --/
def subtractBase4 (a b : ℕ) : ℕ := sorry

theorem base4_calculation : 
  let a := 230
  let b := 21
  let c := 2
  let d := 12
  let e := 3
  subtractBase4 (divideBase4 (multiplyBase4 a b) c) (multiplyBase4 d e) = 3222 := by
  sorry

end base4_calculation_l3626_362627


namespace min_jugs_proof_l3626_362669

/-- The capacity of each jug in ounces -/
def jug_capacity : ℕ := 16

/-- The capacity of the container to be filled in ounces -/
def container_capacity : ℕ := 200

/-- The minimum number of jugs needed to fill or exceed the container capacity -/
def min_jugs : ℕ := 13

theorem min_jugs_proof :
  (∀ n : ℕ, n < min_jugs → n * jug_capacity < container_capacity) ∧
  min_jugs * jug_capacity ≥ container_capacity :=
sorry

end min_jugs_proof_l3626_362669


namespace constant_term_expansion_constant_term_is_fifteen_l3626_362698

/-- The constant term in the expansion of (x - 1/x^2)^6 -/
theorem constant_term_expansion : ℕ :=
  let n : ℕ := 6
  let k : ℕ := 2
  Nat.choose n k

/-- The constant term in the expansion of (x - 1/x^2)^6 is 15 -/
theorem constant_term_is_fifteen : constant_term_expansion = 15 := by
  sorry

end constant_term_expansion_constant_term_is_fifteen_l3626_362698


namespace max_distance_between_generatrices_l3626_362639

/-- The maximum distance between two generatrices of two cones with a common base -/
theorem max_distance_between_generatrices (r h H : ℝ) (h_pos : 0 < h) (H_pos : 0 < H) (h_le_H : h ≤ H) :
  ∃ (d : ℝ), d = (h + H) * r / Real.sqrt (r^2 + H^2) ∧
  ∀ (d' : ℝ), d' ≤ d :=
sorry

end max_distance_between_generatrices_l3626_362639


namespace congruence_problem_l3626_362680

theorem congruence_problem (y : ℤ) 
  (h1 : (4 + y) % (4^3) = 3^2 % (4^3))
  (h2 : (6 + y) % (6^3) = 4^2 % (6^3))
  (h3 : (8 + y) % (8^3) = 6^2 % (8^3)) :
  y % 168 = 4 := by
sorry

end congruence_problem_l3626_362680


namespace road_repair_hours_l3626_362689

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 57)
  (h2 : days1 = 12)
  (h3 : people2 = 30)
  (h4 : days2 = 19)
  (h5 : hours2 = 6)
  (h6 : people1 * days1 * (people2 * days2 * hours2) = people2 * days2 * (people1 * days1 * hours2)) :
  ∃ hours1 : ℕ, hours1 = 5 ∧ people1 * days1 * hours1 = people2 * days2 * hours2 := by
  sorry

end road_repair_hours_l3626_362689


namespace digit_extraction_l3626_362600

theorem digit_extraction (a b c : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) :
  let S := 100 * a + 10 * b + c
  (S / 100 = a) ∧ ((S / 10) % 10 = b) ∧ (S % 10 = c) := by
  sorry

end digit_extraction_l3626_362600


namespace solve_equation_l3626_362634

theorem solve_equation (k l x : ℝ) : 
  (2 : ℝ) / 3 = k / 54 ∧ 
  (2 : ℝ) / 3 = (k + l) / 90 ∧ 
  (2 : ℝ) / 3 = (x - l) / 150 → 
  x = 106 := by sorry

end solve_equation_l3626_362634


namespace prob_select_dime_l3626_362681

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The total value of quarters in the container in dollars -/
def total_quarters_value : ℚ := 12.50

/-- The total value of nickels in the container in dollars -/
def total_nickels_value : ℚ := 15.00

/-- The total value of dimes in the container in dollars -/
def total_dimes_value : ℚ := 5.00

/-- The probability of randomly selecting a dime from the container -/
theorem prob_select_dime : 
  (total_dimes_value / dime_value) / 
  ((total_quarters_value / quarter_value) + 
   (total_nickels_value / nickel_value) + 
   (total_dimes_value / dime_value)) = 1 / 8 := by
sorry

end prob_select_dime_l3626_362681


namespace correct_ordering_l3626_362612

/-- Represents the labels of the conjectures -/
inductive ConjLabel
  | A | C | G | P | R | E | S

/-- The smallest counterexample for each conjecture -/
def smallest_counterexample : ConjLabel → ℕ
  | ConjLabel.A => 44
  | ConjLabel.C => 105
  | ConjLabel.G => 5777
  | ConjLabel.P => 906150257
  | ConjLabel.R => 23338590792
  | ConjLabel.E => 31858749840007945920321
  | ConjLabel.S => 8424432925592889329288197322308900672459420460792433

/-- Checks if a list of ConjLabels is in ascending order based on their smallest counterexamples -/
def is_ascending (labels : List ConjLabel) : Prop :=
  labels.Pairwise (λ l1 l2 => smallest_counterexample l1 < smallest_counterexample l2)

/-- The theorem to be proved -/
theorem correct_ordering :
  is_ascending [ConjLabel.A, ConjLabel.C, ConjLabel.G, ConjLabel.P, ConjLabel.R, ConjLabel.E, ConjLabel.S] :=
by sorry

end correct_ordering_l3626_362612


namespace forgot_homework_percentage_l3626_362664

/-- Represents the percentage of students who forgot their homework in group B -/
def percentage_forgot_B : ℝ := 15

theorem forgot_homework_percentage :
  let total_students : ℕ := 100
  let group_A_students : ℕ := 20
  let group_B_students : ℕ := 80
  let percentage_forgot_A : ℝ := 20
  let percentage_forgot_total : ℝ := 16
  percentage_forgot_B = ((percentage_forgot_total * total_students) - 
                         (percentage_forgot_A * group_A_students)) / group_B_students * 100 :=
by sorry

end forgot_homework_percentage_l3626_362664


namespace min_value_reciprocal_sum_l3626_362661

theorem min_value_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 9) :
  2/a + 2/b + 2/c ≥ 2 ∧ 
  (2/a + 2/b + 2/c = 2 ↔ a = 3 ∧ b = 3 ∧ c = 3) := by
  sorry

#check min_value_reciprocal_sum

end min_value_reciprocal_sum_l3626_362661


namespace opposite_of_fraction_l3626_362666

theorem opposite_of_fraction (n : ℕ) (hn : n ≠ 0) :
  ∃ x : ℚ, (1 : ℚ) / n + x = 0 → x = -(1 : ℚ) / n := by
  sorry

end opposite_of_fraction_l3626_362666


namespace inequality_equivalence_l3626_362673

theorem inequality_equivalence (x y : ℝ) : y - x < Real.sqrt (x^2) ↔ y < 0 ∨ y < 2*x := by sorry

end inequality_equivalence_l3626_362673


namespace power_tower_mod_500_l3626_362657

theorem power_tower_mod_500 : 2^(2^(2^2)) ≡ 36 [ZMOD 500] := by
  sorry

end power_tower_mod_500_l3626_362657


namespace rational_inequality_l3626_362678

theorem rational_inequality (a b c d : ℚ) 
  (h : a^3 - 2005 = b^3 + 2027 ∧ 
       b^3 + 2027 = c^3 - 2822 ∧ 
       c^3 - 2822 = d^3 + 2820) : 
  c > a ∧ a > b ∧ b > d := by
sorry

end rational_inequality_l3626_362678


namespace adults_in_sleeper_class_l3626_362667

def total_passengers : ℕ := 320
def adult_percentage : ℚ := 75 / 100
def sleeper_adult_percentage : ℚ := 15 / 100

theorem adults_in_sleeper_class : 
  ⌊(total_passengers : ℚ) * adult_percentage * sleeper_adult_percentage⌋ = 36 := by
  sorry

end adults_in_sleeper_class_l3626_362667


namespace square_equality_l3626_362672

theorem square_equality (a b : ℝ) : a = b → a^2 = b^2 := by
  sorry

end square_equality_l3626_362672


namespace congruence_problem_l3626_362622

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 20 = 3 → (3 * x + 14) % 20 = 12 := by
  sorry

end congruence_problem_l3626_362622


namespace complex_power_simplification_l3626_362653

theorem complex_power_simplification :
  ((1 + 2 * Complex.I) / (1 - 2 * Complex.I)) ^ 1012 = 1 := by
  sorry

end complex_power_simplification_l3626_362653


namespace inequality_proof_l3626_362696

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y / z) + (y^2 * z / x) + (z^2 * x / y) ≥ x^2 + y^2 + z^2 := by
  sorry

end inequality_proof_l3626_362696


namespace line_slope_through_origin_and_one_neg_one_l3626_362613

/-- The slope of a line passing through points (0,0) and (1,-1) is -1. -/
theorem line_slope_through_origin_and_one_neg_one : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, -1)
  (B.2 - A.2) / (B.1 - A.1) = -1 :=
by sorry

end line_slope_through_origin_and_one_neg_one_l3626_362613


namespace square_sum_product_l3626_362660

theorem square_sum_product (x : ℝ) :
  (Real.sqrt (8 + x) + Real.sqrt (27 - x) = 9) →
  (8 + x) * (27 - x) = 529 := by
  sorry

end square_sum_product_l3626_362660


namespace cylinder_min_circumscribed_sphere_l3626_362625

/-- For a cylinder with surface area 16π and base radius r, 
    the surface area of its circumscribed sphere is minimized when r² = 8√5/5 -/
theorem cylinder_min_circumscribed_sphere (r : ℝ) : 
  (2 * π * r^2 + 2 * π * r * ((8 : ℝ) / r - r) = 16 * π) →
  (∃ (R : ℝ), R^2 = r^2 + ((8 : ℝ) / r - r)^2 / 4 ∧ 
    ∀ (R' : ℝ), R'^2 = r^2 + ((8 : ℝ) / r' - r')^2 / 4 → R'^2 ≥ R^2) →
  r^2 = 8 * Real.sqrt 5 / 5 := by
sorry

end cylinder_min_circumscribed_sphere_l3626_362625


namespace jordan_danielle_roses_l3626_362601

def roses_remaining (initial : ℕ) (additional : ℕ) : ℕ :=
  let total := initial + additional
  let after_first_day := total / 2
  let after_second_day := after_first_day / 2
  after_second_day

theorem jordan_danielle_roses : roses_remaining 24 12 = 9 := by
  sorry

end jordan_danielle_roses_l3626_362601


namespace shift_sine_graph_l3626_362609

/-- The problem statement as a theorem -/
theorem shift_sine_graph (φ : ℝ) (h₁ : 0 < φ) (h₂ : φ < π) : 
  let f : ℝ → ℝ := λ x => 2 * Real.sin (2 * x)
  let g : ℝ → ℝ := λ x => 2 * Real.sin (2 * x - 2 * φ)
  (∃ x₁ x₂ : ℝ, |f x₁ - g x₂| = 4 ∧ 
    (∀ y₁ y₂ : ℝ, |f y₁ - g y₂| = 4 → |x₁ - x₂| ≤ |y₁ - y₂|) ∧
    |x₁ - x₂| = π / 6) →
  φ = π / 3 ∨ φ = 2 * π / 3 := by
sorry

end shift_sine_graph_l3626_362609


namespace business_profit_l3626_362608

theorem business_profit (total_profit : ℝ) : 
  (0.25 * total_profit) + (2 * (0.25 * (0.75 * total_profit))) = 50000 → 
  total_profit = 80000 := by
sorry

end business_profit_l3626_362608


namespace muffin_cost_calculation_l3626_362645

/-- Given a purchase of 3 items of equal cost and one item of known cost,
    with a discount applied, prove the original cost of each equal-cost item. -/
theorem muffin_cost_calculation (M : ℝ) : 
  (∃ (M : ℝ), 
    (0.85 * (3 * M + 1.45) = 3.70) ∧ 
    (abs (M - 0.97) < 0.01)) := by
  sorry

end muffin_cost_calculation_l3626_362645


namespace supplementary_angles_ratio_l3626_362671

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 4 / 5 →  -- angles are in ratio 4:5
  a = 80 :=  -- smaller angle is 80°
by sorry

end supplementary_angles_ratio_l3626_362671


namespace triangle_area_l3626_362603

/-- The area of a triangle with vertices at (-2,3), (7,-3), and (4,6) is 31.5 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (7, -3)
  let C : ℝ × ℝ := (4, 6)
  let area := (1/2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|
  area = 31.5 := by
sorry


end triangle_area_l3626_362603


namespace not_p_necessary_not_sufficient_for_not_q_l3626_362602

theorem not_p_necessary_not_sufficient_for_not_q 
  (h1 : p → q) 
  (h2 : ¬(q → p)) : 
  (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
sorry

end not_p_necessary_not_sufficient_for_not_q_l3626_362602


namespace intersection_points_concyclic_l3626_362663

/-- A circle in which quadrilateral ABCD is inscribed -/
structure CircumCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- A convex quadrilateral ABCD inscribed in a circle -/
structure InscribedQuadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  circle : CircumCircle

/-- Circles drawn with each side of ABCD as a chord -/
structure SideCircles where
  AB : CircumCircle
  BC : CircumCircle
  CD : CircumCircle
  DA : CircumCircle

/-- Intersection points of circles drawn over adjacent sides -/
structure IntersectionPoints where
  A1 : ℝ × ℝ
  B1 : ℝ × ℝ
  C1 : ℝ × ℝ
  D1 : ℝ × ℝ

/-- Function to check if four points are concyclic -/
def areConcyclic (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

theorem intersection_points_concyclic 
  (quad : InscribedQuadrilateral) 
  (sides : SideCircles) 
  (points : IntersectionPoints) : 
  areConcyclic points.A1 points.B1 points.C1 points.D1 := by
  sorry

end intersection_points_concyclic_l3626_362663


namespace trip_distance_calculation_l3626_362677

theorem trip_distance_calculation (total_distance : ℝ) (speed1 speed2 avg_speed : ℝ) 
  (h1 : total_distance = 70)
  (h2 : speed1 = 48)
  (h3 : speed2 = 24)
  (h4 : avg_speed = 32) :
  ∃ (first_part : ℝ),
    first_part = 35 ∧
    first_part / speed1 + (total_distance - first_part) / speed2 = total_distance / avg_speed :=
by sorry

end trip_distance_calculation_l3626_362677


namespace car_price_calculation_l3626_362650

/-- Represents the price of a car given loan terms and payments -/
def carPrice (loanYears : ℕ) (downPayment : ℚ) (monthlyPayment : ℚ) : ℚ :=
  downPayment + (loanYears * 12 : ℕ) * monthlyPayment

/-- Theorem stating that given the specific loan terms, the car price is $20,000 -/
theorem car_price_calculation :
  let loanYears : ℕ := 5
  let downPayment : ℚ := 5000
  let monthlyPayment : ℚ := 250
  carPrice loanYears downPayment monthlyPayment = 20000 := by
  sorry

#eval carPrice 5 5000 250

end car_price_calculation_l3626_362650


namespace cost_of_skirt_l3626_362631

/-- Proves that the cost of each skirt is $15 --/
theorem cost_of_skirt (total_spent art_supplies_cost number_of_skirts : ℕ) 
  (h1 : total_spent = 50)
  (h2 : art_supplies_cost = 20)
  (h3 : number_of_skirts = 2) :
  (total_spent - art_supplies_cost) / number_of_skirts = 15 := by
  sorry

end cost_of_skirt_l3626_362631


namespace muffin_banana_price_ratio_l3626_362697

theorem muffin_banana_price_ratio : 
  ∀ (muffin_price banana_price : ℝ),
  (5 * muffin_price + 4 * banana_price > 0) →
  (3 * (5 * muffin_price + 4 * banana_price) = 3 * muffin_price + 20 * banana_price) →
  (muffin_price / banana_price = 3 / 2) :=
by sorry

end muffin_banana_price_ratio_l3626_362697


namespace books_vs_figures_difference_l3626_362651

theorem books_vs_figures_difference :
  ∀ (initial_figures initial_books added_figures : ℕ),
    initial_figures = 2 →
    initial_books = 10 →
    added_figures = 4 →
    initial_books - (initial_figures + added_figures) = 4 :=
by
  sorry

end books_vs_figures_difference_l3626_362651


namespace sphere_radius_from_hemisphere_volume_l3626_362679

/-- Given a sphere whose hemisphere has a volume of 36π cm³, prove that the radius of the sphere is 3 cm. -/
theorem sphere_radius_from_hemisphere_volume :
  ∀ r : ℝ, (2 / 3 * π * r^3 = 36 * π) → r = 3 := by
  sorry

end sphere_radius_from_hemisphere_volume_l3626_362679


namespace expr_D_not_complete_square_expr_A_is_complete_square_expr_B_is_complete_square_expr_C_is_complete_square_l3626_362685

-- Define the expressions
def expr_A (x : ℝ) := x^2 - 2*x + 1
def expr_B (x : ℝ) := 1 - 2*x + x^2
def expr_C (a b : ℝ) := a^2 + b^2 - 2*a*b
def expr_D (x : ℝ) := 4*x^2 + 4*x - 1

-- Define what it means for an expression to be factored as a complete square
def is_complete_square (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * (x - b)^2

-- Theorem stating that expr_D cannot be factored as a complete square
theorem expr_D_not_complete_square :
  ¬ is_complete_square expr_D :=
sorry

-- Theorems stating that the other expressions can be factored as complete squares
theorem expr_A_is_complete_square :
  is_complete_square expr_A :=
sorry

theorem expr_B_is_complete_square :
  is_complete_square expr_B :=
sorry

theorem expr_C_is_complete_square :
  ∃ (f : ℝ → ℝ → ℝ), ∀ a b, expr_C a b = f a b ∧ is_complete_square (f a) :=
sorry

end expr_D_not_complete_square_expr_A_is_complete_square_expr_B_is_complete_square_expr_C_is_complete_square_l3626_362685


namespace expression_evaluation_l3626_362635

theorem expression_evaluation : (1/3)⁻¹ - 2 * Real.cos (30 * π / 180) - |2 - Real.sqrt 3| - (4 - Real.pi)^0 = 0 := by
  sorry

end expression_evaluation_l3626_362635


namespace rectangle_width_decrease_l3626_362640

theorem rectangle_width_decrease (L W : ℝ) (h1 : L > 0) (h2 : W > 0) : 
  let new_length := 1.5 * L
  let new_width := W * (L / new_length)
  let percent_decrease := (W - new_width) / W * 100
  percent_decrease = 100/3 := by
sorry

end rectangle_width_decrease_l3626_362640


namespace elective_courses_schemes_l3626_362674

theorem elective_courses_schemes (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 10 → k = 3 → m = 3 →
  (Nat.choose (n - m) k + m * Nat.choose (n - m) (k - 1) = 98) :=
by sorry

end elective_courses_schemes_l3626_362674


namespace intersection_value_l3626_362617

theorem intersection_value (m n : ℝ) (h1 : n = 2 / m) (h2 : n = m + 3) :
  1 / m - 1 / n = 3 / 2 := by
  sorry

end intersection_value_l3626_362617


namespace mrs_hilt_bugs_l3626_362611

theorem mrs_hilt_bugs (total_flowers : ℝ) (flowers_per_bug : ℝ) (h1 : total_flowers = 3.0) (h2 : flowers_per_bug = 1.5) :
  total_flowers / flowers_per_bug = 2 := by
  sorry

end mrs_hilt_bugs_l3626_362611


namespace min_value_of_expression_l3626_362647

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_chord : 4 * a + 2 * b = 2) : 
  (1 / a + 2 / b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4 * a₀ + 2 * b₀ = 2 ∧ 1 / a₀ + 2 / b₀ = 8 :=
by sorry

end min_value_of_expression_l3626_362647


namespace distance_run_in_two_hours_l3626_362676

/-- Given a person's running capabilities, calculate the distance they can run in 2 hours -/
theorem distance_run_in_two_hours 
  (distance : ℝ) -- The unknown distance the person can run in 2 hours
  (time_for_distance : ℝ) -- Time taken to run the unknown distance
  (track_length : ℝ) -- Length of the track
  (time_for_track : ℝ) -- Time taken to run the track
  (h1 : time_for_distance = 2) -- The person can run the unknown distance in 2 hours
  (h2 : track_length = 10000) -- The track is 10000 meters long
  (h3 : time_for_track = 10) -- It takes 10 hours to run the track
  (h4 : distance / time_for_distance = track_length / time_for_track) -- The speed is constant
  : distance = 2000 := by
  sorry

end distance_run_in_two_hours_l3626_362676


namespace unique_triangle_exists_l3626_362605

/-- Triangle ABC with given side lengths and angle -/
structure Triangle where
  a : ℝ
  b : ℝ
  A : ℝ

/-- Predicate for a valid triangle satisfying the given conditions -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.a = Real.sqrt 3 ∧ t.b = 1 ∧ t.A = 130 * (Real.pi / 180)

/-- Theorem stating that there exists exactly one valid triangle -/
theorem unique_triangle_exists : ∃! t : Triangle, is_valid_triangle t :=
sorry

end unique_triangle_exists_l3626_362605


namespace cost_of_paints_paint_cost_is_five_l3626_362691

theorem cost_of_paints (classes : ℕ) (folders_per_class : ℕ) (pencils_per_class : ℕ) 
  (pencils_per_eraser : ℕ) (folder_cost : ℕ) (pencil_cost : ℕ) (eraser_cost : ℕ) 
  (total_spent : ℕ) : ℕ :=
  let total_folders := classes * folders_per_class
  let total_pencils := classes * pencils_per_class
  let total_erasers := total_pencils / pencils_per_eraser
  let folder_expense := total_folders * folder_cost
  let pencil_expense := total_pencils * pencil_cost
  let eraser_expense := total_erasers * eraser_cost
  let total_expense := folder_expense + pencil_expense + eraser_expense
  total_spent - total_expense

theorem paint_cost_is_five :
  cost_of_paints 6 1 3 6 6 2 1 80 = 5 := by
  sorry

end cost_of_paints_paint_cost_is_five_l3626_362691


namespace number_sum_proof_l3626_362694

theorem number_sum_proof : ∃ x : ℤ, x + 15 = 96 ∧ x = 81 := by
  sorry

end number_sum_proof_l3626_362694


namespace opposite_pairs_l3626_362637

-- Define the pairs of numbers
def pair_A : ℚ × ℚ := (-5, 1/5)
def pair_B : ℤ × ℤ := (8, 8)
def pair_C : ℤ × ℤ := (-3, 3)
def pair_D : ℚ × ℚ := (7/2, 7/2)

-- Define what it means for two numbers to be opposite
def are_opposite (a b : ℚ) : Prop := a = -b

-- Theorem stating that pair C contains opposite numbers, while others do not
theorem opposite_pairs :
  (¬ are_opposite pair_A.1 pair_A.2) ∧
  (¬ are_opposite pair_B.1 pair_B.2) ∧
  (are_opposite pair_C.1 pair_C.2) ∧
  (¬ are_opposite pair_D.1 pair_D.2) :=
sorry

end opposite_pairs_l3626_362637


namespace petya_win_probability_l3626_362624

/-- Represents the number of stones a player can take in one turn -/
inductive StonesPerTurn
  | one
  | two
  | three
  | four

/-- Represents a player in the game -/
inductive Player
  | petya
  | computer

/-- Represents the state of the game -/
structure GameState where
  stones : Nat
  turn : Player

/-- The initial state of the game -/
def initialState : GameState :=
  { stones := 16, turn := Player.petya }

/-- Represents the strategy of a player -/
def Strategy := GameState → StonesPerTurn

/-- Petya's random strategy -/
def petyaStrategy : Strategy :=
  fun _ => sorry -- Randomly choose between 1 and 4 stones

/-- Computer's optimal strategy -/
def computerStrategy : Strategy :=
  fun _ => sorry -- Always choose the optimal number of stones

/-- The probability of Petya winning the game -/
def petyaWinProbability : ℚ :=
  1 / 256

/-- Theorem stating that Petya's win probability is 1/256 -/
theorem petya_win_probability :
  petyaWinProbability = 1 / 256 := by sorry


end petya_win_probability_l3626_362624


namespace third_term_is_five_l3626_362662

-- Define the sequence a_n
def a (n : ℕ) : ℕ := sorry

-- Define the sum function S_n
def S (n : ℕ) : ℕ := n^2

-- State the theorem
theorem third_term_is_five :
  a 3 = 5 :=
by
  sorry

end third_term_is_five_l3626_362662


namespace bankers_discount_equation_l3626_362675

/-- The banker's discount (BD) for a certain sum of money. -/
def BD : ℚ := 80

/-- The true discount (TD) for the same sum of money. -/
def TD : ℚ := 70

/-- The present value (PV) of the sum due. -/
def PV : ℚ := 490

/-- Theorem stating that the given BD, TD, and PV satisfy the banker's discount equation. -/
theorem bankers_discount_equation : BD = TD + TD^2 / PV := by sorry

end bankers_discount_equation_l3626_362675


namespace central_angle_approx_longitude_diff_l3626_362659

/-- Represents a point on Earth's surface --/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Calculates the central angle between two points on Earth's surface,
    assuming Earth is a perfect sphere --/
def centralAngle (p1 p2 : EarthPoint) : Real :=
  sorry

theorem central_angle_approx_longitude_diff
  (L M : EarthPoint)
  (h1 : L.latitude = 0)
  (h2 : L.longitude = 45)
  (h3 : M.latitude = 23.5)
  (h4 : M.longitude = -90)
  (h5 : abs M.latitude < 30) :
  abs (centralAngle L M - 135) < 5 := by
  sorry

end central_angle_approx_longitude_diff_l3626_362659


namespace initial_sand_calculation_l3626_362687

/-- The amount of sand lost during the trip in pounds -/
def sand_lost : ℝ := 2.4

/-- The amount of sand remaining at arrival in pounds -/
def sand_remaining : ℝ := 1.7

/-- The initial amount of sand on the truck in pounds -/
def initial_sand : ℝ := sand_lost + sand_remaining

theorem initial_sand_calculation : initial_sand = 4.1 := by
  sorry

end initial_sand_calculation_l3626_362687


namespace function_equality_l3626_362665

theorem function_equality (x : ℝ) (h : x ≠ 0) : x^0 = x/x := by
  sorry

end function_equality_l3626_362665


namespace no_inscribed_triangle_with_sine_roots_l3626_362654

theorem no_inscribed_triangle_with_sine_roots :
  ¬ ∃ (a b c : ℝ) (A B C : ℝ),
    0 < a ∧ 0 < b ∧ 0 < c ∧
    0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
    A + B + C = π ∧
    a = 2 * Real.sin (A / 2) ∧
    b = 2 * Real.sin (B / 2) ∧
    c = 2 * Real.sin (C / 2) ∧
    ∃ (p : ℝ),
      (Real.sin A)^3 - 2 * a * (Real.sin A)^2 + b * c * Real.sin A = p ∧
      (Real.sin B)^3 - 2 * a * (Real.sin B)^2 + b * c * Real.sin B = p ∧
      (Real.sin C)^3 - 2 * a * (Real.sin C)^2 + b * c * Real.sin C = p :=
by sorry

end no_inscribed_triangle_with_sine_roots_l3626_362654


namespace triangle_angle_side_ratio_l3626_362655

/-- In a triangle ABC, if the ratio of angles A:B:C is 3:1:2, then the ratio of sides a:b:c is 2:1:√3 -/
theorem triangle_angle_side_ratio (A B C a b c : ℝ) (h_triangle : A + B + C = π) 
  (h_angle_ratio : A = 3 * B ∧ C = 2 * B) : 
  ∃ (k : ℝ), a = 2 * k ∧ b = k ∧ c = Real.sqrt 3 * k := by
  sorry

end triangle_angle_side_ratio_l3626_362655


namespace investment_change_investment_change_specific_l3626_362620

theorem investment_change (initial_investment : ℝ) 
                          (first_year_loss_percent : ℝ) 
                          (second_year_gain_percent : ℝ) : ℝ :=
  let first_year_amount := initial_investment * (1 - first_year_loss_percent / 100)
  let second_year_amount := first_year_amount * (1 + second_year_gain_percent / 100)
  let total_change_percent := (second_year_amount - initial_investment) / initial_investment * 100
  total_change_percent

theorem investment_change_specific : 
  investment_change 200 10 25 = 12.5 := by
  sorry

end investment_change_investment_change_specific_l3626_362620


namespace y₁_y₂_friendly_l3626_362628

/-- Two functions are friendly if their difference is between -1 and 1 for all x in (0,1) -/
def friendly (f g : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x → x < 1 → -1 < f x - g x ∧ f x - g x < 1

/-- The function y₁(x) = x² - 1 -/
def y₁ (x : ℝ) : ℝ := x^2 - 1

/-- The function y₂(x) = 2x - 1 -/
def y₂ (x : ℝ) : ℝ := 2*x - 1

/-- Theorem: y₁ and y₂ are friendly functions -/
theorem y₁_y₂_friendly : friendly y₁ y₂ := by
  sorry

end y₁_y₂_friendly_l3626_362628


namespace connected_graphs_lower_bound_l3626_362682

/-- The number of connected labeled graphs on n vertices -/
def g (n : ℕ) : ℕ := sorry

/-- The total number of labeled graphs on n vertices -/
def total_graphs (n : ℕ) : ℕ := 2^(n * (n - 1) / 2)

/-- Theorem: The number of connected labeled graphs on n vertices is at least half of the total number of labeled graphs on n vertices -/
theorem connected_graphs_lower_bound (n : ℕ) : g n ≥ total_graphs n / 2 := by sorry

end connected_graphs_lower_bound_l3626_362682


namespace peach_count_correct_l3626_362632

/-- The number of baskets -/
def total_baskets : ℕ := 150

/-- The number of peaches in each odd-numbered basket -/
def peaches_odd : ℕ := 14

/-- The number of peaches in each even-numbered basket -/
def peaches_even : ℕ := 12

/-- The total number of peaches -/
def total_peaches : ℕ := 1950

theorem peach_count_correct : 
  (total_baskets / 2) * peaches_odd + (total_baskets / 2) * peaches_even = total_peaches := by
  sorry

end peach_count_correct_l3626_362632


namespace isosceles_max_perimeter_l3626_362644

-- Define a triangle
structure Triangle where
  base : ℝ
  angle : ℝ
  side1 : ℝ
  side2 : ℝ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.base + t.side1 + t.side2

-- Define an isosceles triangle
def isIsosceles (t : Triangle) : Prop := t.side1 = t.side2

-- Theorem statement
theorem isosceles_max_perimeter (b a : ℝ) :
  ∀ (t : Triangle), t.base = b → t.angle = a →
    ∃ (t_iso : Triangle), t_iso.base = b ∧ t_iso.angle = a ∧ isIsosceles t_iso ∧
      perimeter t_iso ≥ perimeter t :=
sorry

end isosceles_max_perimeter_l3626_362644


namespace part1_part2_l3626_362656

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) * (x - 1)

-- Part 1: Prove that a = 1/2 given the conditions
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 2) → a = 1/2 := by sorry

-- Part 2: Characterize the solution set for f(x) < 0 when a > 0
theorem part2 (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, f a x < 0 ↔ 
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨
     (a = 1 ∧ False) ∨
     (a > 1 ∧ 1/a < x ∧ x < 1))) := by sorry

end part1_part2_l3626_362656


namespace stations_visited_l3626_362695

theorem stations_visited (total_nails : ℕ) (nails_per_station : ℕ) (h1 : total_nails = 140) (h2 : nails_per_station = 7) :
  total_nails / nails_per_station = 20 := by
sorry

end stations_visited_l3626_362695


namespace distance_point_to_line_l3626_362616

def point : ℝ × ℝ × ℝ := (0, 3, -1)
def linePoint1 : ℝ × ℝ × ℝ := (1, -2, 0)
def linePoint2 : ℝ × ℝ × ℝ := (3, 1, 4)

def distancePointToLine (p : ℝ × ℝ × ℝ) (l1 l2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_point_to_line :
  distancePointToLine point linePoint1 linePoint2 = Real.sqrt 22058 / 29 := by
  sorry

end distance_point_to_line_l3626_362616


namespace number_difference_proof_l3626_362633

theorem number_difference_proof (x : ℚ) : x - (3/5) * x = 58 → x = 145 := by
  sorry

end number_difference_proof_l3626_362633


namespace advertising_cost_proof_l3626_362643

/-- Proves that the advertising cost is $1000 given the problem conditions -/
theorem advertising_cost_proof 
  (total_customers : ℕ) 
  (purchase_rate : ℚ) 
  (item_cost : ℕ) 
  (profit : ℕ) :
  total_customers = 100 →
  purchase_rate = 4/5 →
  item_cost = 25 →
  profit = 1000 →
  (total_customers : ℚ) * purchase_rate * item_cost - profit = 1000 :=
by sorry

end advertising_cost_proof_l3626_362643


namespace product_of_binary_and_ternary_l3626_362699

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (λ acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert ternary to decimal
def ternary_to_decimal (ternary : List ℕ) : ℕ :=
  ternary.enum.foldl (λ acc (i, d) => acc + d * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary := [false, true, false, true]  -- 1010 in binary
  let ternary := [2, 0, 1]  -- 102 in ternary
  (binary_to_decimal binary) * (ternary_to_decimal ternary) = 110 := by
  sorry

end product_of_binary_and_ternary_l3626_362699


namespace common_chord_length_l3626_362642

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 2*y - 40 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Theorem statement
theorem common_chord_length :
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 :=
sorry

end common_chord_length_l3626_362642


namespace quadratic_function_minimum_l3626_362690

-- Define the quadratic function
def y (x m : ℝ) : ℝ := x^2 + 2*m*x - 3*m + 1

-- Define the conditions
def condition1 (p q : ℝ) : Prop := 4*p^2 + 9*q^2 = 2
def condition2 (x p q : ℝ) : Prop := (1/2)*x + 3*p*q = 1

-- State the theorem
theorem quadratic_function_minimum (x p q m : ℝ) :
  condition1 p q →
  condition2 x p q →
  (∀ x', y x' m ≥ 1) →
  (∃ x'', y x'' m = 1) →
  (m = -3 ∨ m = 1) :=
sorry

end quadratic_function_minimum_l3626_362690


namespace blue_ridge_elementary_calculation_l3626_362604

theorem blue_ridge_elementary_calculation (num_classrooms : ℕ) 
  (students_per_classroom : ℕ) (turtles_per_classroom : ℕ) (teachers_per_classroom : ℕ) : 
  num_classrooms = 6 →
  students_per_classroom = 22 →
  turtles_per_classroom = 2 →
  teachers_per_classroom = 1 →
  num_classrooms * students_per_classroom - 
  (num_classrooms * turtles_per_classroom + num_classrooms * teachers_per_classroom) = 114 := by
  sorry

#check blue_ridge_elementary_calculation

end blue_ridge_elementary_calculation_l3626_362604


namespace eleven_subtractions_to_zero_l3626_362649

def digit_sum (n : ℕ) : ℕ := sorry

def subtract_digit_sum (n : ℕ) : ℕ := n - digit_sum n

def repeat_subtract_digit_sum (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => repeat_subtract_digit_sum (subtract_digit_sum n) k

theorem eleven_subtractions_to_zero (n : ℕ) (h : 100 ≤ n ∧ n ≤ 109) :
  repeat_subtract_digit_sum n 11 = 0 := by sorry

end eleven_subtractions_to_zero_l3626_362649


namespace min_operations_for_2006_l3626_362686

/-- The minimal number of operations needed to calculate x^2006 -/
def min_operations : ℕ := 17

/-- A function that represents the number of operations needed to calculate x^n given x -/
noncomputable def operations (n : ℕ) : ℕ := sorry

/-- The theorem stating that the minimal number of operations to calculate x^2006 is 17 -/
theorem min_operations_for_2006 : operations 2006 = min_operations := by sorry

end min_operations_for_2006_l3626_362686


namespace unique_solution_implies_negative_a_l3626_362623

theorem unique_solution_implies_negative_a :
  ∀ a : ℝ,
  (∃! x : ℝ, |x^2 - 1| = a * |x - 1|) →
  a < 0 := by
sorry

end unique_solution_implies_negative_a_l3626_362623


namespace exists_N_average_fifteen_l3626_362619

theorem exists_N_average_fifteen : 
  ∃ N : ℝ, 15 < N ∧ N < 25 ∧ (8 + 14 + N) / 3 = 15 := by
sorry

end exists_N_average_fifteen_l3626_362619


namespace tetra_edge_is_2sqrt3_l3626_362692

/-- Configuration of five mutually tangent spheres with a circumscribed tetrahedron -/
structure SphereTetConfig where
  /-- Radius of each sphere -/
  r : ℝ
  /-- Centers of the four bottom spheres -/
  bottom_centers : Fin 4 → ℝ × ℝ × ℝ
  /-- Center of the top sphere -/
  top_center : ℝ × ℝ × ℝ
  /-- Vertices of the tetrahedron -/
  tetra_vertices : Fin 4 → ℝ × ℝ × ℝ

/-- The spheres are mutually tangent and properly configured -/
def is_valid_config (cfg : SphereTetConfig) : Prop :=
  cfg.r = 2 ∧
  ∀ i j, i ≠ j → dist (cfg.bottom_centers i) (cfg.bottom_centers j) = 4 ∧
  ∀ i, dist (cfg.bottom_centers i) cfg.top_center = 4 ∧
  cfg.top_center.2 = 2 ∧
  cfg.tetra_vertices 0 = cfg.top_center ∧
  ∀ i : Fin 3, cfg.tetra_vertices (i + 1) = cfg.bottom_centers i

/-- The edge length of the tetrahedron -/
def tetra_edge_length (cfg : SphereTetConfig) : ℝ :=
  dist (cfg.tetra_vertices 0) (cfg.tetra_vertices 1)

/-- Main theorem: The edge length of the tetrahedron is 2√3 -/
theorem tetra_edge_is_2sqrt3 (cfg : SphereTetConfig) (h : is_valid_config cfg) :
  tetra_edge_length cfg = 2 * Real.sqrt 3 := by sorry

end tetra_edge_is_2sqrt3_l3626_362692


namespace min_x_prime_factorization_l3626_362607

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x ^ 7 = 13 * y ^ 11) :
  ∃ (a b c d : ℕ),
    x = a ^ c * b ^ d ∧
    a.Prime ∧ b.Prime ∧
    (∀ (a' b' c' d' : ℕ), x = a' ^ c' * b' ^ d' → a' ^ c' * b' ^ d' ≥ a ^ c * b ^ d) ∧
    a + b + c + d = 25 :=
by sorry

end min_x_prime_factorization_l3626_362607


namespace almond_walnut_ratio_is_five_to_two_l3626_362606

/-- Represents a mixture of almonds and walnuts -/
structure NutMixture where
  total_weight : ℝ
  almond_weight : ℝ
  almond_parts : ℝ
  walnut_parts : ℝ

/-- The ratio of almonds to walnuts in a nut mixture -/
def almond_to_walnut_ratio (mix : NutMixture) : ℝ × ℝ :=
  (mix.almond_parts, mix.walnut_parts)

/-- Theorem stating the ratio of almonds to walnuts in the specific mixture -/
theorem almond_walnut_ratio_is_five_to_two 
  (mix : NutMixture)
  (h1 : mix.total_weight = 210)
  (h2 : mix.almond_weight = 150)
  (h3 : mix.almond_parts = 5)
  (h4 : mix.almond_parts + mix.walnut_parts = mix.total_weight / (mix.almond_weight / mix.almond_parts)) :
  almond_to_walnut_ratio mix = (5, 2) := by
  sorry


end almond_walnut_ratio_is_five_to_two_l3626_362606


namespace rectangle_area_diagonal_relation_l3626_362658

theorem rectangle_area_diagonal_relation :
  ∀ (length width diagonal : ℝ),
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 5 / 2 →
  length ^ 2 + width ^ 2 = diagonal ^ 2 →
  diagonal = 13 →
  ∃ (k : ℝ), length * width = k * diagonal ^ 2 ∧ k = 10 / 29 := by
  sorry

end rectangle_area_diagonal_relation_l3626_362658


namespace linear_function_property_l3626_362629

-- Define a linear function
def LinearFunction (g : ℝ → ℝ) : Prop :=
  ∀ x y t : ℝ, g (x + t * (y - x)) = g x + t * (g y - g x)

-- State the theorem
theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g)
  (h1 : g 8 - g 3 = 15)
  (h2 : g 4 - g 1 = 9) :
  g 10 - g 1 = 27 := by
  sorry

end linear_function_property_l3626_362629


namespace sum_of_cubes_of_roots_sum_of_reciprocals_of_roots_l3626_362615

-- Define the coefficients of the first equation: 2x^2 - 5x + 1 = 0
def a₁ : ℚ := 2
def b₁ : ℚ := -5
def c₁ : ℚ := 1

-- Define the coefficients of the second equation: 2x^2 - 11x + 13 = 0
def a₂ : ℚ := 2
def b₂ : ℚ := -11
def c₂ : ℚ := 13

-- Theorem for the sum of cubes of roots
theorem sum_of_cubes_of_roots :
  let x₁ := (-b₁ + Real.sqrt (b₁^2 - 4*a₁*c₁)) / (2*a₁)
  let x₂ := (-b₁ - Real.sqrt (b₁^2 - 4*a₁*c₁)) / (2*a₁)
  x₁^3 + x₂^3 = 95/8 := by sorry

-- Theorem for the sum of reciprocals of roots
theorem sum_of_reciprocals_of_roots :
  let y₁ := (-b₂ + Real.sqrt (b₂^2 - 4*a₂*c₂)) / (2*a₂)
  let y₂ := (-b₂ - Real.sqrt (b₂^2 - 4*a₂*c₂)) / (2*a₂)
  y₁/y₂ + y₂/y₁ = 69/26 := by sorry

end sum_of_cubes_of_roots_sum_of_reciprocals_of_roots_l3626_362615


namespace vector_sum_magnitude_l3626_362668

/-- Given vectors a and b, where b and b-a are collinear, prove |a+b| = 3√5/2 -/
theorem vector_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![1, 2]
  (∃ (k : ℝ), b = k • (b - a)) →
  ‖a + b‖ = 3 * Real.sqrt 5 / 2 := by
sorry

end vector_sum_magnitude_l3626_362668


namespace f_max_min_on_interval_l3626_362638

-- Define the function f(x) = x^3 - 3x + 2
def f (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Define the closed interval [0, 3]
def interval : Set ℝ := Set.Icc 0 3

-- Theorem statement
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), max = 20 ∧ min = 0 ∧
  (∀ x ∈ interval, f x ≤ max) ∧
  (∃ x ∈ interval, f x = max) ∧
  (∀ x ∈ interval, min ≤ f x) ∧
  (∃ x ∈ interval, f x = min) := by
  sorry

end f_max_min_on_interval_l3626_362638


namespace max_d_value_l3626_362618

def a (n : ℕ+) : ℕ := 103 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (N : ℕ+), d N = 13 ∧ ∀ (n : ℕ+), d n ≤ 13 :=
sorry

end max_d_value_l3626_362618
