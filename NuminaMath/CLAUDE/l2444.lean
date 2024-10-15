import Mathlib

namespace NUMINAMATH_CALUDE_systematic_sampling_survey_c_count_l2444_244432

theorem systematic_sampling_survey_c_count 
  (total_population : Nat) 
  (sample_size : Nat) 
  (first_number : Nat) 
  (survey_c_lower_bound : Nat) 
  (survey_c_upper_bound : Nat) 
  (h1 : total_population = 1000)
  (h2 : sample_size = 50)
  (h3 : first_number = 8)
  (h4 : survey_c_lower_bound = 751)
  (h5 : survey_c_upper_bound = 1000) :
  (Finset.filter (fun n => 
    let term := first_number + (n - 1) * (total_population / sample_size)
    term ≥ survey_c_lower_bound ∧ term ≤ survey_c_upper_bound
  ) (Finset.range sample_size)).card = 12 := by
  sorry

#check systematic_sampling_survey_c_count

end NUMINAMATH_CALUDE_systematic_sampling_survey_c_count_l2444_244432


namespace NUMINAMATH_CALUDE_cuboid_length_calculation_l2444_244482

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboid_surface_area (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: A cuboid with surface area 720, breadth 6, and height 10 has length 18.75 -/
theorem cuboid_length_calculation (l : ℝ) :
  cuboid_surface_area l 6 10 = 720 → l = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_length_calculation_l2444_244482


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2444_244414

theorem unique_prime_solution :
  ∃! (p q : ℕ) (n : ℕ), 
    Prime p ∧ Prime q ∧ n > 1 ∧
    (p^(2*n+1) - 1) / (p - 1) = (q^3 - 1) / (q - 1) ∧
    p = 2 ∧ q = 5 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2444_244414


namespace NUMINAMATH_CALUDE_number_fraction_proof_l2444_244418

theorem number_fraction_proof (N : ℝ) (h : (3/10) * N - 8 = 12) : (1/5) * N = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_proof_l2444_244418


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2444_244454

/-- Simple interest calculation -/
theorem simple_interest_rate
  (principal : ℝ)
  (time : ℝ)
  (interest : ℝ)
  (h1 : principal = 10000)
  (h2 : time = 1)
  (h3 : interest = 900) :
  (interest / (principal * time)) * 100 = 9 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2444_244454


namespace NUMINAMATH_CALUDE_xy_equation_solutions_l2444_244441

theorem xy_equation_solutions (x y : ℤ) : x + y = x * y ↔ (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_xy_equation_solutions_l2444_244441


namespace NUMINAMATH_CALUDE_geometry_theorem_l2444_244448

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane)
  (h_distinct_lines : m ≠ n)
  (h_distinct_planes : α ≠ β) :
  (∀ (m : Line) (β α : Plane), 
    subset m β → plane_parallel α β → parallel m α) ∧
  (∀ (m n : Line) (α β : Plane),
    perpendicular m α → perpendicular n β → plane_parallel α β → line_parallel m n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorem_l2444_244448


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2444_244415

theorem polynomial_evaluation :
  let f : ℝ → ℝ := λ x ↦ 2*x^4 + 3*x^3 - x^2 + 2*x + 5
  f (-2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2444_244415


namespace NUMINAMATH_CALUDE_chinese_paper_probability_l2444_244468

/-- The number of Chinese exam papers in the bag -/
def chinese_papers : ℕ := 2

/-- The number of Tibetan exam papers in the bag -/
def tibetan_papers : ℕ := 3

/-- The number of English exam papers in the bag -/
def english_papers : ℕ := 1

/-- The total number of exam papers in the bag -/
def total_papers : ℕ := chinese_papers + tibetan_papers + english_papers

/-- The probability of drawing a Chinese exam paper -/
def prob_chinese : ℚ := chinese_papers / total_papers

theorem chinese_paper_probability : prob_chinese = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_chinese_paper_probability_l2444_244468


namespace NUMINAMATH_CALUDE_optimal_promotional_expense_l2444_244409

noncomputable section

-- Define the sales volume function
def P (x : ℝ) : ℝ := 3 - 2 / (x + 1)

-- Define the profit function
def profit (x : ℝ) : ℝ := 16 - (4 / (x + 1) + x)

-- Define the theorem
theorem optimal_promotional_expense (a : ℝ) (h : a > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ a → profit x ≤ profit (min 1 a)) ∧
  (a ≥ 1 → profit 1 = (profit ∘ min 1) a) ∧
  (a < 1 → profit a = (profit ∘ min 1) a) := by
  sorry

end

end NUMINAMATH_CALUDE_optimal_promotional_expense_l2444_244409


namespace NUMINAMATH_CALUDE_smallest_n_squared_plus_n_divisibility_l2444_244406

theorem smallest_n_squared_plus_n_divisibility : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), (1 ≤ k) ∧ (k ≤ n) ∧ ((n^2 + n) % k = 0)) ∧
  (∃ (k : ℕ), (1 ≤ k) ∧ (k ≤ n) ∧ ((n^2 + n) % k ≠ 0)) ∧
  (∀ (m : ℕ), (m > 0) ∧ (m < n) → 
    (∀ (k : ℕ), (1 ≤ k) ∧ (k ≤ m) → ((m^2 + m) % k = 0)) ∨
    (∀ (k : ℕ), (1 ≤ k) ∧ (k ≤ m) → ((m^2 + m) % k ≠ 0))) ∧
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_squared_plus_n_divisibility_l2444_244406


namespace NUMINAMATH_CALUDE_smallest_x_value_l2444_244431

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (240 + x)) : 
  ∀ z : ℕ+, z < x → ¬∃ w : ℕ+, (3 : ℚ) / 4 = w / (240 + z) :=
by sorry

#check smallest_x_value

end NUMINAMATH_CALUDE_smallest_x_value_l2444_244431


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2444_244475

def U : Set Nat := {0, 1, 2, 3, 5, 6, 8}
def A : Set Nat := {1, 5, 8}
def B : Set Nat := {2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2444_244475


namespace NUMINAMATH_CALUDE_souvenir_purchase_theorem_l2444_244492

/-- Represents a purchasing plan for souvenirs -/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a purchase plan is valid according to the given constraints -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.typeA + p.typeB = 60 ∧
  p.typeB ≤ 2 * p.typeA ∧
  100 * p.typeA + 60 * p.typeB ≤ 4500

/-- Calculates the cost of a purchase plan -/
def planCost (p : PurchasePlan) : ℕ :=
  100 * p.typeA + 60 * p.typeB

/-- The main theorem encompassing all parts of the problem -/
theorem souvenir_purchase_theorem :
  (∃! p : PurchasePlan, p.typeA + p.typeB = 60 ∧ planCost p = 4600) ∧
  (∃! plans : List PurchasePlan, plans.length = 3 ∧ 
    ∀ p ∈ plans, isValidPlan p ∧
    ∀ p, isValidPlan p → p ∈ plans) ∧
  (∃ p : PurchasePlan, isValidPlan p ∧
    ∀ q, isValidPlan q → planCost p ≤ planCost q ∧
    planCost p = 4400) := by
  sorry

#check souvenir_purchase_theorem

end NUMINAMATH_CALUDE_souvenir_purchase_theorem_l2444_244492


namespace NUMINAMATH_CALUDE_raffle_prize_calculation_l2444_244460

theorem raffle_prize_calculation (kept_amount : ℝ) (kept_percentage : ℝ) (total_prize : ℝ) : 
  kept_amount = 80 → kept_percentage = 0.80 → kept_amount = kept_percentage * total_prize → 
  total_prize = 100 := by
sorry

end NUMINAMATH_CALUDE_raffle_prize_calculation_l2444_244460


namespace NUMINAMATH_CALUDE_short_trees_after_planting_l2444_244424

/-- The number of short trees in a park after planting new trees -/
def total_short_trees (initial_short_trees newly_planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + newly_planted_short_trees

/-- Theorem stating that the total number of short trees after planting
    is equal to the sum of initial short trees and newly planted short trees -/
theorem short_trees_after_planting
  (initial_short_trees : ℕ)
  (initial_tall_trees : ℕ)
  (newly_planted_short_trees : ℕ) :
  total_short_trees initial_short_trees newly_planted_short_trees =
  initial_short_trees + newly_planted_short_trees :=
by
  sorry

/-- Example calculation for the specific problem -/
def park_short_trees : ℕ :=
  total_short_trees 3 9

#eval park_short_trees

end NUMINAMATH_CALUDE_short_trees_after_planting_l2444_244424


namespace NUMINAMATH_CALUDE_wall_width_l2444_244405

/-- Given a rectangular wall with specific proportions and volume, prove its width is 4 meters. -/
theorem wall_width (w h l : ℝ) (h_height : h = 6 * w) (h_length : l = 7 * h) 
  (h_volume : w * h * l = 16128) : w = 4 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_l2444_244405


namespace NUMINAMATH_CALUDE_circular_table_arrangements_l2444_244488

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem circular_table_arrangements (num_mathletes : ℕ) (num_coaches : ℕ) : 
  num_mathletes = 4 → num_coaches = 2 → 
  (factorial num_mathletes * 2) / 2 = 24 := by
  sorry

#check circular_table_arrangements

end NUMINAMATH_CALUDE_circular_table_arrangements_l2444_244488


namespace NUMINAMATH_CALUDE_handshakes_for_seven_people_l2444_244490

/-- The number of handshakes in a group of n people, where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- Theorem stating that the number of handshakes for 7 people is 21. -/
theorem handshakes_for_seven_people : handshakes 7 = 21 := by sorry

end NUMINAMATH_CALUDE_handshakes_for_seven_people_l2444_244490


namespace NUMINAMATH_CALUDE_pots_per_vertical_stack_l2444_244472

theorem pots_per_vertical_stack (total_pots : ℕ) (num_shelves : ℕ) (sets_per_shelf : ℕ) : 
  total_pots = 60 → num_shelves = 4 → sets_per_shelf = 3 → 
  (total_pots / (num_shelves * sets_per_shelf) : ℕ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_pots_per_vertical_stack_l2444_244472


namespace NUMINAMATH_CALUDE_milk_price_calculation_l2444_244400

/-- Calculates the price per gallon of milk given the daily production, 
    number of days, and total income. -/
def price_per_gallon (daily_production : ℕ) (days : ℕ) (total_income : ℚ) : ℚ :=
  total_income / (daily_production * days)

/-- Theorem stating that the price per gallon of milk is $3.05 given the conditions. -/
theorem milk_price_calculation : 
  price_per_gallon 200 30 18300 = 305/100 := by
  sorry

#eval price_per_gallon 200 30 18300

end NUMINAMATH_CALUDE_milk_price_calculation_l2444_244400


namespace NUMINAMATH_CALUDE_matthew_ate_six_l2444_244480

/-- The number of egg rolls eaten by Matthew, Patrick, and Alvin. -/
structure EggRolls where
  matthew : ℕ
  patrick : ℕ
  alvin : ℕ

/-- The conditions of the egg roll problem. -/
def egg_roll_conditions (e : EggRolls) : Prop :=
  e.matthew = 3 * e.patrick ∧
  e.patrick = e.alvin / 2 ∧
  e.alvin = 4

/-- The theorem stating that Matthew ate 6 egg rolls. -/
theorem matthew_ate_six (e : EggRolls) (h : egg_roll_conditions e) : e.matthew = 6 := by
  sorry

end NUMINAMATH_CALUDE_matthew_ate_six_l2444_244480


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2444_244476

theorem chocolate_distribution (total_bars : ℕ) (num_people : ℕ) 
  (h1 : total_bars = 60) 
  (h2 : num_people = 5) : 
  let bars_per_person := total_bars / num_people
  let person1_final := bars_per_person - bars_per_person / 2
  let person2_final := bars_per_person + 2
  let person3_final := bars_per_person - 2
  let person4_final := bars_per_person
  person2_final + person3_final + person4_final = 36 := by
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2444_244476


namespace NUMINAMATH_CALUDE_heejin_has_most_volleyballs_l2444_244416

/-- The number of basketballs Heejin has -/
def basketballs : ℕ := 3

/-- The number of volleyballs Heejin has -/
def volleyballs : ℕ := 5

/-- The number of baseballs Heejin has -/
def baseball : ℕ := 1

/-- Theorem stating that Heejin has more volleyballs than any other type of ball -/
theorem heejin_has_most_volleyballs : 
  volleyballs > basketballs ∧ volleyballs > baseball :=
sorry

end NUMINAMATH_CALUDE_heejin_has_most_volleyballs_l2444_244416


namespace NUMINAMATH_CALUDE_stating_bus_passenger_count_l2444_244469

/-- 
Calculates the final number of passengers on a bus given the initial number
and the changes at various stops.
-/
def final_passengers (initial : ℕ) (first_stop_on : ℕ) (other_stops_off : ℕ) (other_stops_on : ℕ) : ℕ :=
  initial + first_stop_on - other_stops_off + other_stops_on

/-- 
Theorem stating that given the specific passenger changes described in the problem,
the final number of passengers on the bus is 49.
-/
theorem bus_passenger_count : final_passengers 50 16 22 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_stating_bus_passenger_count_l2444_244469


namespace NUMINAMATH_CALUDE_original_number_of_people_l2444_244426

theorem original_number_of_people (x : ℕ) : 
  (x / 3 : ℚ) - (x / 3 : ℚ) / 4 = 15 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_original_number_of_people_l2444_244426


namespace NUMINAMATH_CALUDE_exponent_division_l2444_244463

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^5 / a^3 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2444_244463


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2444_244434

/-- Given point A(-3, 1, 4), prove that its symmetric point B with respect to the origin has coordinates (3, -1, -4). -/
theorem symmetric_point_coordinates :
  let A : ℝ × ℝ × ℝ := (-3, 1, 4)
  let B : ℝ × ℝ × ℝ := (3, -1, -4)
  (∀ (x y z : ℝ), (x, y, z) = A → (-x, -y, -z) = B) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2444_244434


namespace NUMINAMATH_CALUDE_pine_boys_count_l2444_244422

/-- Represents a middle school in the winter program. -/
inductive School
| Maple
| Pine
| Oak

/-- Represents the gender of a student. -/
inductive Gender
| Boy
| Girl

/-- Represents the distribution of students in the winter program. -/
structure WinterProgram where
  total_students : Nat
  total_boys : Nat
  total_girls : Nat
  maple_students : Nat
  pine_students : Nat
  oak_students : Nat
  maple_girls : Nat

/-- Theorem stating that the number of boys from Pine Middle School is 20. -/
theorem pine_boys_count (wp : WinterProgram) 
  (h1 : wp.total_students = 120)
  (h2 : wp.total_boys = 68)
  (h3 : wp.total_girls = 52)
  (h4 : wp.maple_students = 50)
  (h5 : wp.pine_students = 40)
  (h6 : wp.oak_students = 30)
  (h7 : wp.maple_girls = 22)
  (h8 : wp.total_students = wp.total_boys + wp.total_girls)
  (h9 : wp.total_students = wp.maple_students + wp.pine_students + wp.oak_students) :
  ∃ (pine_boys : Nat), pine_boys = 20 ∧ 
    pine_boys + (wp.pine_students - pine_boys) = wp.pine_students :=
  sorry


end NUMINAMATH_CALUDE_pine_boys_count_l2444_244422


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2444_244407

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the property of being twice differentiable with continuous second derivative
def TwiceDifferentiableContinuous (f : RealFunction) : Prop :=
  Differentiable ℝ f ∧ 
  Differentiable ℝ (deriv f) ∧ 
  Continuous (deriv (deriv f))

-- Define the functional equation
def SatisfiesFunctionalEquation (f : RealFunction) : Prop :=
  ∀ t : ℝ, f t ^ 2 = f (t * Real.sqrt 2)

-- Main theorem
theorem functional_equation_solution 
  (f : RealFunction) 
  (h1 : TwiceDifferentiableContinuous f) 
  (h2 : SatisfiesFunctionalEquation f) : 
  (∃ c : ℝ, ∀ x : ℝ, f x = Real.exp (c * x^2)) ∨ 
  (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2444_244407


namespace NUMINAMATH_CALUDE_jims_investment_l2444_244401

/-- 
Given an investment scenario with three investors and a total investment,
calculate the investment amount for one specific investor.
-/
theorem jims_investment
  (total_investment : ℕ) 
  (john_ratio : ℕ) 
  (james_ratio : ℕ) 
  (jim_ratio : ℕ) 
  (h1 : total_investment = 80000)
  (h2 : john_ratio = 4)
  (h3 : james_ratio = 7)
  (h4 : jim_ratio = 9) : 
  jim_ratio * (total_investment / (john_ratio + james_ratio + jim_ratio)) = 36000 := by
  sorry

end NUMINAMATH_CALUDE_jims_investment_l2444_244401


namespace NUMINAMATH_CALUDE_brothers_book_pages_l2444_244452

/-- Represents the number of pages read by a person in a week --/
structure WeeklyReading where
  total_pages : ℕ
  books_per_week : ℕ
  days_to_finish : ℕ

/-- Calculates the average pages read per day --/
def average_pages_per_day (r : WeeklyReading) : ℕ :=
  r.total_pages / r.days_to_finish

theorem brothers_book_pages 
  (ryan : WeeklyReading)
  (ryan_brother : WeeklyReading)
  (h1 : ryan.total_pages = 2100)
  (h2 : ryan.books_per_week = 5)
  (h3 : ryan.days_to_finish = 7)
  (h4 : ryan_brother.books_per_week = 7)
  (h5 : ryan_brother.days_to_finish = 7)
  (h6 : average_pages_per_day ryan = average_pages_per_day ryan_brother + 100) :
  ryan_brother.total_pages / ryan_brother.books_per_week = 200 :=
by sorry

end NUMINAMATH_CALUDE_brothers_book_pages_l2444_244452


namespace NUMINAMATH_CALUDE_chosen_number_proof_l2444_244470

theorem chosen_number_proof (x : ℝ) : (x / 4) - 175 = 10 → x = 740 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l2444_244470


namespace NUMINAMATH_CALUDE_cashier_miscount_adjustment_l2444_244451

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Calculates the error when miscounting one coin as another -/
def miscount_error (actual : String) (counted_as : String) : ℤ :=
  (coin_value counted_as : ℤ) - (coin_value actual : ℤ)

/-- Theorem: The net error and correct adjustment for x miscounted coins -/
theorem cashier_miscount_adjustment (x : ℕ) :
  let penny_as_nickel_error := miscount_error "penny" "nickel"
  let quarter_as_dime_error := miscount_error "quarter" "dime"
  let net_error := x * penny_as_nickel_error + x * quarter_as_dime_error
  let adjustment := -net_error
  (net_error = -11 * x) ∧ (adjustment = 11 * x) := by
  sorry

end NUMINAMATH_CALUDE_cashier_miscount_adjustment_l2444_244451


namespace NUMINAMATH_CALUDE_sum_first_105_remainder_l2444_244466

theorem sum_first_105_remainder (n : Nat) (sum : Nat → Nat) : 
  n = 105 → 
  (∀ k, sum k = k * (k + 1) / 2) → 
  sum n % 1000 = 565 := by
sorry

end NUMINAMATH_CALUDE_sum_first_105_remainder_l2444_244466


namespace NUMINAMATH_CALUDE_circle_equation_and_tangent_lines_l2444_244435

def circle_C (a b : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + (y - b)^2 = 2}

theorem circle_equation_and_tangent_lines :
  ∀ (a b : ℝ),
    b = a + 1 →
    (5 - a)^2 + (4 - b)^2 = 2 →
    (3 - a)^2 + (6 - b)^2 = 2 →
    (∃ (x y : ℝ), circle_C a b (x, y)) →
    (circle_C 4 5 = circle_C a b) ∧
    (∀ (k : ℝ),
      (k = 1 ∨ k = 23/7) ↔
      (∃ (x : ℝ), x ≠ 1 ∧ circle_C 4 5 (x, k*(x-1)) ∧
        ∀ (y : ℝ), y ≠ k*(x-1) → ¬ circle_C 4 5 (x, y))) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_and_tangent_lines_l2444_244435


namespace NUMINAMATH_CALUDE_inequality_proof_l2444_244471

theorem inequality_proof (x : ℝ) : 
  -7 < x ∧ x < -0.775 → (x + Real.sqrt 3) / (x + 10) > (3*x + 2*Real.sqrt 3) / (2*x + 14) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2444_244471


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l2444_244458

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

-- State the theorem
theorem even_increasing_inequality (h1 : is_even f) (h2 : increasing_on f 0 1) :
  f 0 < f (-0.5) ∧ f (-0.5) < f (-1) :=
sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l2444_244458


namespace NUMINAMATH_CALUDE_number_exceeding_half_by_80_l2444_244411

theorem number_exceeding_half_by_80 (x : ℝ) : x = 0.5 * x + 80 → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_half_by_80_l2444_244411


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l2444_244461

theorem max_value_on_ellipse :
  ∀ x y : ℝ, (x^2 / 4 + y^2 / 9 = 1) → (2*x - y ≤ 5) ∧ ∃ x₀ y₀ : ℝ, (x₀^2 / 4 + y₀^2 / 9 = 1) ∧ (2*x₀ - y₀ = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l2444_244461


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2444_244421

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

theorem symmetric_points_sum (a b : ℝ) :
  let p : Point := ⟨-2, 3⟩
  let q : Point := ⟨a, b⟩
  symmetricYAxis p q → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2444_244421


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_negation_l2444_244484

theorem necessary_not_sufficient_negation (p q : Prop) :
  (q → p) ∧ ¬(p → q) → (¬p → ¬q) ∧ ¬(¬q → ¬p) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_negation_l2444_244484


namespace NUMINAMATH_CALUDE_four_digit_equal_digits_l2444_244465

theorem four_digit_equal_digits (n : ℤ) : 12 * n^2 + 12 * n + 11 = 5555 ↔ n = 21 ∨ n = -22 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_equal_digits_l2444_244465


namespace NUMINAMATH_CALUDE_radical_axis_existence_l2444_244403

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def power (p : ℝ × ℝ) (c : Circle) : ℝ :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 - c.radius^2

def radical_axis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | power p c1 = power p c2}

def intersect (c1 c2 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, power p c1 = 0 ∧ power p c2 = 0

def line_of_centers (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (t * c1.center.1 + (1 - t) * c2.center.1, 
                           t * c1.center.2 + (1 - t) * c2.center.2)}

def perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l1 ∧ p ∈ l2 ∧
    ∀ q r : ℝ × ℝ, q ∈ l1 → r ∈ l2 → 
      (q.1 - p.1) * (r.1 - p.1) + (q.2 - p.2) * (r.2 - p.2) = 0

theorem radical_axis_existence (c1 c2 : Circle) :
  (intersect c1 c2 → 
    ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ power p1 c1 = 0 ∧ power p1 c2 = 0 ∧
                     power p2 c1 = 0 ∧ power p2 c2 = 0 ∧
                     radical_axis c1 c2 = {p : ℝ × ℝ | ∃ t : ℝ, p = (t * p1.1 + (1 - t) * p2.1, 
                                                                   t * p1.2 + (1 - t) * p2.2)}) ∧
  (¬intersect c1 c2 → 
    ∃ c3 : Circle, intersect c1 c3 ∧ intersect c2 c3 ∧
    ∃ p : ℝ × ℝ, power p c1 = power p c2 ∧ power p c2 = power p c3 ∧
    perpendicular (radical_axis c1 c2) (line_of_centers c1 c2) ∧
    p ∈ radical_axis c1 c2) :=
sorry

end NUMINAMATH_CALUDE_radical_axis_existence_l2444_244403


namespace NUMINAMATH_CALUDE_jasmine_carry_weight_l2444_244420

/-- The weight of a bag of chips in ounces -/
def chipBagWeight : ℕ := 20

/-- The weight of a tin of cookies in ounces -/
def cookieTinWeight : ℕ := 9

/-- The number of bags of chips Jasmine buys -/
def numChipBags : ℕ := 6

/-- The ratio of tins of cookies to bags of chips Jasmine buys -/
def cookieToChipRatio : ℕ := 4

/-- The number of ounces in a pound -/
def ouncesPerPound : ℕ := 16

/-- Theorem: Given the conditions, Jasmine has to carry 21 pounds -/
theorem jasmine_carry_weight :
  (numChipBags * chipBagWeight +
   numChipBags * cookieToChipRatio * cookieTinWeight) / ouncesPerPound = 21 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_carry_weight_l2444_244420


namespace NUMINAMATH_CALUDE_entertainment_expense_calculation_l2444_244440

def entertainment_expense (initial_amount : ℝ) (food_percentage : ℝ) (phone_percentage : ℝ) (final_amount : ℝ) : ℝ :=
  let food_expense := initial_amount * food_percentage
  let after_food := initial_amount - food_expense
  let phone_expense := after_food * phone_percentage
  let after_phone := after_food - phone_expense
  after_phone - final_amount

theorem entertainment_expense_calculation :
  entertainment_expense 200 0.60 0.25 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_entertainment_expense_calculation_l2444_244440


namespace NUMINAMATH_CALUDE_volume_of_rotated_region_l2444_244483

/-- A region composed of unit squares resting along the x-axis and y-axis -/
structure Region :=
  (squares : ℕ)
  (along_x_axis : Bool)
  (along_y_axis : Bool)

/-- The volume of a solid formed by rotating a region about the y-axis -/
noncomputable def rotated_volume (r : Region) : ℝ :=
  sorry

/-- The problem statement -/
theorem volume_of_rotated_region :
  ∃ (r : Region),
    r.squares = 16 ∧
    r.along_x_axis = true ∧
    r.along_y_axis = true ∧
    rotated_volume r = 37 * Real.pi :=
  sorry

end NUMINAMATH_CALUDE_volume_of_rotated_region_l2444_244483


namespace NUMINAMATH_CALUDE_sum_M_N_equals_two_l2444_244498

/-- Definition of M -/
def M : ℚ := 1^5 + 2^4 * 3^3 - 4^2 / 5^1

/-- Definition of N -/
def N : ℚ := 1^5 - 2^4 * 3^3 + 4^2 / 5^1

/-- Theorem: The sum of M and N is equal to 2 -/
theorem sum_M_N_equals_two : M + N = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_M_N_equals_two_l2444_244498


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2444_244486

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo 0 2 : Set ℝ) = {x | |2*x - 1| < |x| + 1} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2444_244486


namespace NUMINAMATH_CALUDE_function_bounds_bounds_achievable_l2444_244427

theorem function_bounds (x : ℝ) : 
  6 ≤ 7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 ∧ 
  7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 ≤ 10 :=
by sorry

theorem bounds_achievable : 
  (∃ x : ℝ, 7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 = 6) ∧
  (∃ x : ℝ, 7 - 4 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4 = 10) :=
by sorry

end NUMINAMATH_CALUDE_function_bounds_bounds_achievable_l2444_244427


namespace NUMINAMATH_CALUDE_center_is_eight_l2444_244473

-- Define the type for our 3x3 grid
def Grid := Fin 3 → Fin 3 → Nat

-- Define what it means for two positions to share an edge
def sharesEdge (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

-- Define the property of consecutive numbers sharing an edge
def consecutiveShareEdge (g : Grid) : Prop :=
  ∀ (i j : Fin 3 × Fin 3), 
    g i.1 i.2 + 1 = g j.1 j.2 → sharesEdge i j

-- Define the sum of corner numbers
def cornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

-- Define the theorem
theorem center_is_eight (g : Grid) 
  (all_numbers : ∀ n : Fin 9, ∃ (i j : Fin 3), g i j = n.val + 1)
  (consec_edge : consecutiveShareEdge g)
  (corner_sum_20 : cornerSum g = 20) :
  g 1 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_center_is_eight_l2444_244473


namespace NUMINAMATH_CALUDE_equation_solution_count_l2444_244449

theorem equation_solution_count : ∃ (s : Finset ℕ),
  (∀ c ∈ s, c ≤ 1000) ∧ 
  (∀ c ∈ s, ∃ x : ℝ, 7 * ⌊x⌋ + 2 * ⌈x⌉ = c) ∧
  (∀ c ≤ 1000, c ∉ s → ¬∃ x : ℝ, 7 * ⌊x⌋ + 2 * ⌈x⌉ = c) ∧
  s.card = 223 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_count_l2444_244449


namespace NUMINAMATH_CALUDE_polynomial_identity_l2444_244497

/-- The polynomial p(x) = x^2 - x + 1 -/
def p (x : ℂ) : ℂ := x^2 - x + 1

/-- α is a root of p(p(p(p(x)))) -/
def α : ℂ := sorry

theorem polynomial_identity :
  (p α - 1) * p α * p (p α) * p (p (p α)) = -1 := by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2444_244497


namespace NUMINAMATH_CALUDE_group_size_correct_l2444_244446

/-- The number of members in the group -/
def n : ℕ := 93

/-- The total collection in paise -/
def total_paise : ℕ := 8649

/-- Theorem stating that n is the correct number of members -/
theorem group_size_correct : n * n = total_paise := by sorry

end NUMINAMATH_CALUDE_group_size_correct_l2444_244446


namespace NUMINAMATH_CALUDE_comic_cost_theorem_l2444_244429

/-- Calculates the final cost of each comic book type after discount --/
def final_comic_cost (common_cards : ℕ) (uncommon_cards : ℕ) (rare_cards : ℕ)
  (common_value : ℚ) (uncommon_value : ℚ) (rare_value : ℚ)
  (standard_price : ℚ) (deluxe_price : ℚ) (limited_price : ℚ)
  (discount_threshold_low : ℚ) (discount_threshold_high : ℚ)
  (discount_low : ℚ) (discount_high : ℚ)
  (ratio_standard : ℕ) (ratio_deluxe : ℕ) (ratio_limited : ℕ) : ℚ :=
  sorry

theorem comic_cost_theorem (common_cards : ℕ) (uncommon_cards : ℕ) (rare_cards : ℕ)
  (common_value : ℚ) (uncommon_value : ℚ) (rare_value : ℚ)
  (standard_price : ℚ) (deluxe_price : ℚ) (limited_price : ℚ)
  (discount_threshold_low : ℚ) (discount_threshold_high : ℚ)
  (discount_low : ℚ) (discount_high : ℚ)
  (ratio_standard : ℕ) (ratio_deluxe : ℕ) (ratio_limited : ℕ) :
  common_cards = 1000 ∧ uncommon_cards = 750 ∧ rare_cards = 250 ∧
  common_value = 5/100 ∧ uncommon_value = 1/10 ∧ rare_value = 1/5 ∧
  standard_price = 4 ∧ deluxe_price = 8 ∧ limited_price = 12 ∧
  discount_threshold_low = 100 ∧ discount_threshold_high = 150 ∧
  discount_low = 5/100 ∧ discount_high = 1/10 ∧
  ratio_standard = 3 ∧ ratio_deluxe = 2 ∧ ratio_limited = 1 →
  final_comic_cost common_cards uncommon_cards rare_cards
    common_value uncommon_value rare_value
    standard_price deluxe_price limited_price
    discount_threshold_low discount_threshold_high
    discount_low discount_high
    ratio_standard ratio_deluxe ratio_limited = 6 :=
by sorry

end NUMINAMATH_CALUDE_comic_cost_theorem_l2444_244429


namespace NUMINAMATH_CALUDE_face_covers_are_squares_and_rectangles_l2444_244413

/-- A parallelogram covering a face of a unit cube -/
structure FaceCover where
  -- The parallelogram's area
  area : ℝ
  -- The parallelogram is a square
  is_square : Prop
  -- The parallelogram is a rectangle
  is_rectangle : Prop

/-- A cube with edge length 1 covered by six identical parallelograms -/
structure CoveredCube where
  -- The edge length of the cube
  edge_length : ℝ
  -- The six identical parallelograms covering the cube
  face_covers : Fin 6 → FaceCover
  -- All face covers are identical
  covers_identical : ∀ (i j : Fin 6), face_covers i = face_covers j
  -- The edge length is 1
  edge_is_unit : edge_length = 1
  -- Each face cover has an area of 1
  cover_area_is_unit : ∀ (i : Fin 6), (face_covers i).area = 1

/-- Theorem: All face covers of a unit cube are squares and rectangles -/
theorem face_covers_are_squares_and_rectangles (cube : CoveredCube) :
  (∀ (i : Fin 6), (cube.face_covers i).is_square) ∧
  (∀ (i : Fin 6), (cube.face_covers i).is_rectangle) := by
  sorry


end NUMINAMATH_CALUDE_face_covers_are_squares_and_rectangles_l2444_244413


namespace NUMINAMATH_CALUDE_train_length_calculation_l2444_244447

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length_calculation (speed : Real) (time : Real) : 
  speed = 144 ∧ time = 1.24990000799936 → 
  ∃ (length : Real), abs (length - 50) < 0.01 ∧ length = speed * time * (5 / 18) := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2444_244447


namespace NUMINAMATH_CALUDE_expression_bounds_l2444_244477

theorem expression_bounds (x y : ℝ) 
  (hx : -3 ≤ x ∧ x ≤ 3) (hy : -2 ≤ y ∧ y ≤ 2) : 
  -6 ≤ x * Real.sqrt (4 - y^2) + y * Real.sqrt (9 - x^2) ∧ 
  x * Real.sqrt (4 - y^2) + y * Real.sqrt (9 - x^2) ≤ 6 := by
  sorry

#check expression_bounds

end NUMINAMATH_CALUDE_expression_bounds_l2444_244477


namespace NUMINAMATH_CALUDE_lucy_earnings_l2444_244445

/-- Calculates the earnings for a single 6-hour cycle -/
def cycle_earnings : ℕ := 1 + 2 + 3 + 4 + 5 + 6

/-- Calculates the earnings for the remaining hours after complete cycles -/
def remaining_earnings (hours : ℕ) : ℕ :=
  match hours with
  | 0 => 0
  | 1 => 1
  | 2 => 1 + 2
  | _ => 1 + 2 + 3

/-- Calculates the total earnings for a given number of hours -/
def total_earnings (hours : ℕ) : ℕ :=
  let complete_cycles := hours / 6
  let remaining_hours := hours % 6
  complete_cycles * cycle_earnings + remaining_earnings remaining_hours

/-- The theorem stating that Lucy's earnings for 45 hours of work is $153 -/
theorem lucy_earnings : total_earnings 45 = 153 := by
  sorry

end NUMINAMATH_CALUDE_lucy_earnings_l2444_244445


namespace NUMINAMATH_CALUDE_age_difference_james_jessica_prove_age_difference_l2444_244493

/-- Given the ages and relationships of Justin, Jessica, and James, prove that James is 7 years older than Jessica. -/
theorem age_difference_james_jessica : ℕ → Prop :=
  fun age_difference =>
    ∀ (justin_age jessica_age james_age : ℕ),
      justin_age = 26 →
      jessica_age = justin_age + 6 →
      james_age > jessica_age →
      james_age + 5 = 44 →
      james_age - jessica_age = age_difference →
      age_difference = 7

/-- Proof of the theorem -/
theorem prove_age_difference : ∃ (age_difference : ℕ), age_difference_james_jessica age_difference := by
  sorry

end NUMINAMATH_CALUDE_age_difference_james_jessica_prove_age_difference_l2444_244493


namespace NUMINAMATH_CALUDE_triangle_area_l2444_244478

theorem triangle_area (a b c : ℝ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) :
  (1 / 2) * a * b = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2444_244478


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2444_244494

theorem sphere_surface_area (V : ℝ) (r : ℝ) (S : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  S = 4 * Real.pi * r^2 → 
  S = 36 * Real.pi * (4^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2444_244494


namespace NUMINAMATH_CALUDE_interior_angles_integral_count_l2444_244485

theorem interior_angles_integral_count : 
  (Finset.filter (fun n : ℕ => n > 2 ∧ (n - 2) * 180 % n = 0) (Finset.range 361)).card = 22 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_integral_count_l2444_244485


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l2444_244410

def geometric_series (n : ℕ) : ℚ :=
  match n with
  | 0 => 7/8
  | 1 => -14/27
  | 2 => 56/81
  | _ => 0  -- We only define the first three terms explicitly

theorem common_ratio_of_geometric_series :
  ∃ r : ℚ, r = -2/3 ∧
    ∀ n : ℕ, n > 0 → geometric_series n = geometric_series (n-1) * r :=
sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l2444_244410


namespace NUMINAMATH_CALUDE_unique_prime_satisfying_condition_l2444_244412

theorem unique_prime_satisfying_condition : 
  ∃! p : ℕ, Prime p ∧ Prime (4 * p^2 + 1) ∧ Prime (6 * p^2 + 1) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_satisfying_condition_l2444_244412


namespace NUMINAMATH_CALUDE_roots_vs_ellipse_l2444_244423

def has_two_positive_roots (m n : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ x₁^2 - m*x₁ + n = 0 ∧ x₂^2 - m*x₂ + n = 0

def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

theorem roots_vs_ellipse (m n : ℝ) :
  ¬(has_two_positive_roots m n → is_ellipse m n) ∧
  ¬(is_ellipse m n → has_two_positive_roots m n) :=
sorry

end NUMINAMATH_CALUDE_roots_vs_ellipse_l2444_244423


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l2444_244459

/-- The number of fruit options Joe has -/
def num_fruits : ℕ := 4

/-- The number of meals Joe has in a day -/
def num_meals : ℕ := 3

/-- The probability of choosing any specific fruit for a meal -/
def prob_single_fruit : ℚ := 1 / num_fruits

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := num_fruits * prob_single_fruit ^ num_meals

/-- The probability of eating at least two different kinds of fruit in a day -/
def prob_different_fruits : ℚ := 1 - prob_same_fruit

theorem joe_fruit_probability :
  prob_different_fruits = 15 / 16 :=
sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l2444_244459


namespace NUMINAMATH_CALUDE_geometric_sum_of_root_l2444_244437

theorem geometric_sum_of_root (x : ℝ) : 
  x^10 - 3*x + 2 = 0 → x ≠ 1 → x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_of_root_l2444_244437


namespace NUMINAMATH_CALUDE_x_value_l2444_244425

theorem x_value (x y z : ℚ) : x = y / 3 → y = z / 4 → z = 80 → x = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2444_244425


namespace NUMINAMATH_CALUDE_prob_at_least_8_stay_correct_l2444_244474

def total_people : ℕ := 10
def certain_people : ℕ := 5
def uncertain_people : ℕ := 5
def uncertain_stay_prob : ℚ := 3/7

def prob_at_least_8_stay : ℚ := 4563/16807

theorem prob_at_least_8_stay_correct :
  let prob_8_stay := (uncertain_people.choose 3) * (uncertain_stay_prob^3 * (1 - uncertain_stay_prob)^2)
  let prob_10_stay := uncertain_stay_prob^uncertain_people
  prob_at_least_8_stay = prob_8_stay + prob_10_stay :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_8_stay_correct_l2444_244474


namespace NUMINAMATH_CALUDE_dogs_food_average_l2444_244442

theorem dogs_food_average (num_dogs : ℕ) (dog1_food : ℝ) (dog2_food : ℝ) (dog3_food : ℝ) :
  num_dogs = 3 →
  dog1_food = 13 →
  dog2_food = 2 * dog1_food →
  dog3_food = 6 →
  (dog1_food + dog2_food + dog3_food) / num_dogs = 15 := by
sorry

end NUMINAMATH_CALUDE_dogs_food_average_l2444_244442


namespace NUMINAMATH_CALUDE_two_digit_number_square_difference_l2444_244428

theorem two_digit_number_square_difference (a b : ℤ) 
  (h1 : a > b) (h2 : a + b = 10) : 
  ∃ k : ℤ, (9*a + 10)^2 - (100 - 9*a)^2 = 20 * k := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_square_difference_l2444_244428


namespace NUMINAMATH_CALUDE_cloth_trimming_l2444_244464

/-- Given a square piece of cloth with side length 22 feet, prove that after trimming 6 feet from two opposite edges and 5 feet from the other two edges, the remaining area is 272 square feet. -/
theorem cloth_trimming (original_length : ℕ) (trim_1 : ℕ) (trim_2 : ℕ) : 
  original_length = 22 → 
  trim_1 = 6 → 
  trim_2 = 5 → 
  (original_length - trim_1) * (original_length - trim_2) = 272 := by
sorry

end NUMINAMATH_CALUDE_cloth_trimming_l2444_244464


namespace NUMINAMATH_CALUDE_final_alloy_mass_l2444_244462

/-- Given two alloys with different copper percentages and their masses,
    prove that the total mass of the final alloy is the sum of the masses of the component alloys. -/
theorem final_alloy_mass
  (alloy1_copper_percent : ℚ)
  (alloy2_copper_percent : ℚ)
  (final_alloy_copper_percent : ℚ)
  (alloy1_mass : ℚ)
  (alloy2_mass : ℚ)
  (h1 : alloy1_copper_percent = 25 / 100)
  (h2 : alloy2_copper_percent = 50 / 100)
  (h3 : final_alloy_copper_percent = 45 / 100)
  (h4 : alloy1_mass = 200)
  (h5 : alloy2_mass = 800) :
  alloy1_mass + alloy2_mass = 1000 := by
  sorry

end NUMINAMATH_CALUDE_final_alloy_mass_l2444_244462


namespace NUMINAMATH_CALUDE_marks_quiz_goal_l2444_244404

theorem marks_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (as_earned : ℕ) (h1 : total_quizzes = 60) 
  (h2 : goal_percentage = 85 / 100) (h3 : completed_quizzes = 40) 
  (h4 : as_earned = 30) : 
  Nat.ceil (↑total_quizzes * goal_percentage) - as_earned ≥ total_quizzes - completed_quizzes := by
  sorry

end NUMINAMATH_CALUDE_marks_quiz_goal_l2444_244404


namespace NUMINAMATH_CALUDE_fraction_is_one_ninth_l2444_244457

/-- Represents a taxi trip with given parameters -/
structure TaxiTrip where
  initialFee : ℚ
  additionalChargePerFraction : ℚ
  totalDistance : ℚ
  totalCharge : ℚ

/-- Calculates the fraction of a mile for which the additional charge applies -/
def fractionOfMile (trip : TaxiTrip) : ℚ :=
  let additionalCharge := trip.totalCharge - trip.initialFee
  let numberOfFractions := additionalCharge / trip.additionalChargePerFraction
  trip.totalDistance / numberOfFractions

/-- Theorem stating that for the given trip parameters, the fraction of a mile
    for which the additional charge applies is 1/9 -/
theorem fraction_is_one_ninth :
  let trip := TaxiTrip.mk 2.25 0.15 3.6 3.60
  fractionOfMile trip = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_one_ninth_l2444_244457


namespace NUMINAMATH_CALUDE_root_sum_squared_plus_three_times_root_l2444_244456

theorem root_sum_squared_plus_three_times_root : ∀ (α β : ℝ), 
  (α^2 + 2*α - 2025 = 0) → 
  (β^2 + 2*β - 2025 = 0) → 
  (α^2 + 3*α + β = 2023) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_squared_plus_three_times_root_l2444_244456


namespace NUMINAMATH_CALUDE_function_relation_l2444_244450

/-- Given functions f, g, and h from ℝ to ℝ satisfying certain conditions,
    prove that h can be expressed in terms of f and g. -/
theorem function_relation (f g h : ℝ → ℝ) 
    (hf : ∀ x, f x = (h (x + 1) + h (x - 1)) / 2)
    (hg : ∀ x, g x = (h (x + 4) + h (x - 4)) / 2) :
    ∀ x, h x = g x - f (x - 3) + f (x - 1) + f (x + 1) - f (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_function_relation_l2444_244450


namespace NUMINAMATH_CALUDE_ellipse_symmetric_point_range_l2444_244438

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 2 = 1

/-- Definition of symmetry with respect to y = 2x -/
def symmetric_points (x₀ y₀ x₁ y₁ : ℝ) : Prop :=
  (y₀ - y₁) / (x₀ - x₁) = -1/2 ∧ (y₀ + y₁) / 2 = 2 * ((x₀ + x₁) / 2)

/-- The main theorem -/
theorem ellipse_symmetric_point_range :
  ∀ x₀ y₀ x₁ y₁ : ℝ,
  ellipse_C x₀ y₀ →
  symmetric_points x₀ y₀ x₁ y₁ →
  -10 ≤ 3 * x₁ - 4 * y₁ ∧ 3 * x₁ - 4 * y₁ ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_point_range_l2444_244438


namespace NUMINAMATH_CALUDE_complex_fraction_sum_complex_product_imaginary_l2444_244402

-- Problem 1
theorem complex_fraction_sum : (1 / (1 - Complex.I)) + (1 / (2 + 3 * Complex.I)) = Complex.mk (17/26) (7/26) := by sorry

-- Problem 2
theorem complex_product_imaginary (z₁ z₂ : ℂ) :
  z₁ = Complex.mk 3 4 →
  Complex.abs z₂ = 5 →
  (Complex.re (z₁ * z₂) = 0 ∧ Complex.im (z₁ * z₂) ≠ 0) →
  z₂ = Complex.mk 4 3 ∨ z₂ = Complex.mk (-4) (-3) := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_complex_product_imaginary_l2444_244402


namespace NUMINAMATH_CALUDE_no_integer_square_diff_222_l2444_244467

theorem no_integer_square_diff_222 : ¬ ∃ (a b : ℤ), a^2 - b^2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_diff_222_l2444_244467


namespace NUMINAMATH_CALUDE_point_outside_circle_l2444_244499

/-- Given a circle with center O and radius 3, and a point P outside the circle,
    prove that the distance between O and P is greater than 3. -/
theorem point_outside_circle (O P : ℝ × ℝ) (r : ℝ) : 
  r = 3 →  -- The radius of the circle is 3
  (∀ Q : ℝ × ℝ, dist O Q = r → dist O P > dist O Q) →  -- P is outside the circle
  dist O P > 3  -- The distance between O and P is greater than 3
:= by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2444_244499


namespace NUMINAMATH_CALUDE_range_of_x_minus_cosy_l2444_244436

theorem range_of_x_minus_cosy (x y : ℝ) (h : x^2 + 2 * Real.cos y = 1) :
  -1 ≤ x - Real.cos y ∧ x - Real.cos y ≤ 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_minus_cosy_l2444_244436


namespace NUMINAMATH_CALUDE_determinant_relations_l2444_244455

theorem determinant_relations (a b c : ℤ) : ∃ (p₁ q₁ r₁ p₂ q₂ r₂ : ℤ),
  a = q₁ * r₂ - q₂ * r₁ ∧
  b = r₁ * p₂ - r₂ * p₁ ∧
  c = p₁ * q₂ - p₂ * q₁ := by
  sorry

end NUMINAMATH_CALUDE_determinant_relations_l2444_244455


namespace NUMINAMATH_CALUDE_point_quadrant_theorem_l2444_244489

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Definition of the third quadrant -/
def in_third_quadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A(x, y-2) is in the fourth quadrant, then B(y-2, -x) is in the third quadrant -/
theorem point_quadrant_theorem (x y : ℝ) :
  let A : Point2D := ⟨x, y - 2⟩
  let B : Point2D := ⟨y - 2, -x⟩
  in_fourth_quadrant A → in_third_quadrant B := by
  sorry

end NUMINAMATH_CALUDE_point_quadrant_theorem_l2444_244489


namespace NUMINAMATH_CALUDE_M_equals_N_set_order_irrelevant_l2444_244430

-- Define the sets M and N
def M : Set ℕ := {3, 2}
def N : Set ℕ := {2, 3}

-- Theorem stating that M and N are equal
theorem M_equals_N : M = N := by
  sorry

-- Additional theorem to emphasize that order doesn't matter in sets
theorem set_order_irrelevant (A B : Set α) : 
  (∀ x, x ∈ A ↔ x ∈ B) → A = B := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_set_order_irrelevant_l2444_244430


namespace NUMINAMATH_CALUDE_roses_planted_is_difference_l2444_244408

/-- The number of rose bushes planted in a park --/
def rosesBushesPlanted (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that the number of rose bushes planted is the difference between final and initial counts --/
theorem roses_planted_is_difference (initial final : ℕ) (h : final ≥ initial) :
  rosesBushesPlanted initial final = final - initial :=
by
  sorry

/-- Specific instance for the given problem --/
example : rosesBushesPlanted 2 6 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_roses_planted_is_difference_l2444_244408


namespace NUMINAMATH_CALUDE_west_notation_l2444_244453

-- Define a type for distance with direction
inductive DirectedDistance
  | east (km : ℝ)
  | west (km : ℝ)

-- Define a function to convert DirectedDistance to a signed real number
def directedDistanceToSigned : DirectedDistance → ℝ
  | DirectedDistance.east km => km
  | DirectedDistance.west km => -km

-- State the theorem
theorem west_notation (d : ℝ) :
  directedDistanceToSigned (DirectedDistance.east 3) = 3 →
  directedDistanceToSigned (DirectedDistance.west 2) = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_west_notation_l2444_244453


namespace NUMINAMATH_CALUDE_gcd_factorial_ratio_l2444_244439

theorem gcd_factorial_ratio : Nat.gcd (Nat.factorial 6) ((Nat.factorial 9) / (Nat.factorial 4)) = 480 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_ratio_l2444_244439


namespace NUMINAMATH_CALUDE_disjunction_is_true_l2444_244443

-- Define the propositions p and q
def p : Prop := ∀ a b : ℝ, a > |b| → a^2 > b^2
def q : Prop := ∀ x : ℝ, x^2 = 4 → x = 2

-- State the theorem
theorem disjunction_is_true (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_is_true_l2444_244443


namespace NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l2444_244487

/-- Pascal's triangle contains every positive integer -/
axiom pascal_contains_all_positive : ∀ n : ℕ, n > 0 → ∃ (row k : ℕ), Nat.choose row k = n

/-- The binomial coefficient function -/
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

/-- The smallest four-digit number -/
def smallest_four_digit : ℕ := 1000

/-- Theorem: 1000 is the smallest four-digit number in Pascal's triangle -/
theorem smallest_four_digit_in_pascal : 
  ∃ (row k : ℕ), binomial_coeff row k = smallest_four_digit ∧ 
  (∀ (r s : ℕ), binomial_coeff r s < smallest_four_digit → binomial_coeff r s < 1000) :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l2444_244487


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l2444_244419

theorem sum_and_ratio_to_difference (m n : ℝ) 
  (sum_eq : m + n = 490)
  (ratio_eq : m / n = 1.2) : 
  ∃ (diff : ℝ), abs (m - n - diff) < 0.5 ∧ diff = 45 :=
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l2444_244419


namespace NUMINAMATH_CALUDE_sequence_term_expression_l2444_244481

def S (n : ℕ) : ℤ := 2 * n^2 - n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 2 else 4 * n - 3

theorem sequence_term_expression (n : ℕ) :
  n ≥ 1 → a n = S n - S (n - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_term_expression_l2444_244481


namespace NUMINAMATH_CALUDE_inequality_proof_l2444_244491

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2444_244491


namespace NUMINAMATH_CALUDE_product_of_solutions_l2444_244496

theorem product_of_solutions (x : ℝ) : 
  (45 = -x^2 - 4*x) → (∃ α β : ℝ, α * β = -45 ∧ 45 = -α^2 - 4*α ∧ 45 = -β^2 - 4*β) :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2444_244496


namespace NUMINAMATH_CALUDE_opposite_face_is_D_l2444_244495

/-- Represents the labels of the faces of a cube --/
inductive FaceLabel
  | A | B | C | D | E | F

/-- Represents the positions of faces on a cube --/
inductive Position
  | Top | Bottom | Left | Right | Front | Back

/-- Represents a cube with labeled faces --/
structure Cube where
  faces : Position → FaceLabel

/-- Defines the opposite position for each position on the cube --/
def oppositePosition : Position → Position
  | Position.Top => Position.Bottom
  | Position.Bottom => Position.Top
  | Position.Left => Position.Right
  | Position.Right => Position.Left
  | Position.Front => Position.Back
  | Position.Back => Position.Front

/-- Theorem stating that in a cube where C is on top and B is to its right, 
    the face opposite to A is labeled D --/
theorem opposite_face_is_D (cube : Cube) 
  (h1 : cube.faces Position.Top = FaceLabel.C) 
  (h2 : cube.faces Position.Right = FaceLabel.B) : 
  ∃ p : Position, cube.faces p = FaceLabel.A ∧ 
  cube.faces (oppositePosition p) = FaceLabel.D := by
  sorry

end NUMINAMATH_CALUDE_opposite_face_is_D_l2444_244495


namespace NUMINAMATH_CALUDE_correct_remaining_leaves_l2444_244417

/-- Calculates the number of remaining leaves on a tree at the end of summer --/
def remaining_leaves (branches : ℕ) (twigs_per_branch : ℕ) 
  (spring_3_leaf_percent : ℚ) (spring_4_leaf_percent : ℚ) (spring_5_leaf_percent : ℚ)
  (summer_leaf_increase : ℕ) (caterpillar_eaten_percent : ℚ) : ℕ :=
  sorry

/-- Theorem stating the correct number of remaining leaves --/
theorem correct_remaining_leaves :
  remaining_leaves 100 150 (20/100) (30/100) (50/100) 2 (10/100) = 85050 :=
sorry

end NUMINAMATH_CALUDE_correct_remaining_leaves_l2444_244417


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l2444_244444

def num_apples : ℕ := 4
def num_oranges : ℕ := 2
def num_bananas : ℕ := 2
def num_pears : ℕ := 1

def total_fruits : ℕ := num_apples + num_oranges + num_bananas + num_pears

theorem fruit_arrangement_count :
  (total_fruits.factorial) / (num_apples.factorial * num_oranges.factorial * num_bananas.factorial * num_pears.factorial) = 3780 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_count_l2444_244444


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l2444_244433

theorem subtraction_of_fractions : 
  (5 : ℚ) / 6 - 1 / 6 - 1 / 4 = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l2444_244433


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2444_244479

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

-- Define monotonicity on an interval
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

-- State the theorem
theorem sufficient_not_necessary (a : ℝ) :
  (a ≥ 2 → monotonic_on (f a) 1 2) ∧
  (∃ b : ℝ, b < 2 ∧ monotonic_on (f b) 1 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2444_244479
