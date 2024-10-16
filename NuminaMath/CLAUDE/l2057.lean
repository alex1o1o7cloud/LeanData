import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l2057_205745

theorem rectangular_prism_sum (a b c : ℕ+) : 
  a * b * c = 21 → a ≠ b → b ≠ c → a ≠ c → a + b + c = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l2057_205745


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2057_205732

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 4) :
  (1 + 1 / (x + 1)) * ((x + 1) / (x^2 + 4*x + 4)) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2057_205732


namespace NUMINAMATH_CALUDE_f_2017_value_l2057_205702

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2017_value (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_value : f (-1) = 6) :
  f 2017 = -6 := by
sorry

end NUMINAMATH_CALUDE_f_2017_value_l2057_205702


namespace NUMINAMATH_CALUDE_printer_equation_l2057_205797

/-- The equation for determining the time of the second printer to print 1000 flyers -/
theorem printer_equation (x : ℝ) : 
  (1000 : ℝ) > 0 → x > 0 → (
    (1000 / 10 + 1000 / x = 1000 / 4) ↔ 
    (1 / 10 + 1 / x = 1 / 4)
  ) := by sorry

end NUMINAMATH_CALUDE_printer_equation_l2057_205797


namespace NUMINAMATH_CALUDE_beacon_school_earnings_l2057_205759

/-- Represents a school's participation in the community project -/
structure School where
  name : String
  students : ℕ
  weekdays : ℕ
  weekendDays : ℕ

/-- Calculates the total earnings for a school given the daily rates -/
def schoolEarnings (s : School) (weekdayRate weekendRate : ℚ) : ℚ :=
  s.students * (s.weekdays * weekdayRate + s.weekendDays * weekendRate)

/-- The main theorem stating that Beacon school's earnings are $336.00 -/
theorem beacon_school_earnings :
  let apex : School := ⟨"Apex", 9, 4, 2⟩
  let beacon : School := ⟨"Beacon", 6, 6, 1⟩
  let citadel : School := ⟨"Citadel", 7, 8, 3⟩
  let schools : List School := [apex, beacon, citadel]
  let totalPaid : ℚ := 1470
  ∃ (weekdayRate : ℚ),
    weekdayRate > 0 ∧
    (schools.map (fun s => schoolEarnings s weekdayRate (2 * weekdayRate))).sum = totalPaid ∧
    schoolEarnings beacon weekdayRate (2 * weekdayRate) = 336 := by
  sorry

end NUMINAMATH_CALUDE_beacon_school_earnings_l2057_205759


namespace NUMINAMATH_CALUDE_square_area_is_25_l2057_205755

/-- A square in the coordinate plane with specific y-coordinates -/
structure SquareWithYCoords where
  -- The y-coordinates of the vertices
  y1 : ℝ
  y2 : ℝ
  y3 : ℝ
  y4 : ℝ
  -- Ensure the y-coordinates are distinct and in ascending order
  h1 : y1 < y2
  h2 : y2 < y3
  h3 : y3 < y4
  -- Ensure the square property (opposite sides are parallel and equal)
  h4 : y4 - y3 = y2 - y1

/-- The area of a square with specific y-coordinates is 25 -/
theorem square_area_is_25 (s : SquareWithYCoords) (h5 : s.y1 = 2) (h6 : s.y2 = 3) (h7 : s.y3 = 7) (h8 : s.y4 = 8) : 
  (s.y3 - s.y2) * (s.y3 - s.y2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_25_l2057_205755


namespace NUMINAMATH_CALUDE_mooncake_problem_l2057_205756

-- Define the types and variables
variable (type_a_cost type_b_cost : ℝ)
variable (total_cost_per_pair : ℝ)
variable (type_a_quantity type_b_quantity : ℕ)
variable (m : ℝ)

-- Define the conditions
def conditions (type_a_cost type_b_cost total_cost_per_pair type_a_quantity type_b_quantity m : ℝ) : Prop :=
  type_a_cost = 1200 ∧
  type_b_cost = 600 ∧
  total_cost_per_pair = 9 ∧
  type_a_quantity = 4 * type_b_quantity ∧
  m ≠ 0 ∧
  (type_a_cost / type_a_quantity + type_b_cost / type_b_quantity = total_cost_per_pair) ∧
  (2 * (type_a_quantity - 15 / 2 * m) + (6 - m / 5) * (type_b_quantity + 15 / 2 * m) = 1400 - 2 * m)

-- State the theorem
theorem mooncake_problem (type_a_cost type_b_cost total_cost_per_pair type_a_quantity type_b_quantity m : ℝ) :
  conditions type_a_cost type_b_cost total_cost_per_pair type_a_quantity type_b_quantity m →
  type_a_quantity = 400 ∧ type_b_quantity = 100 ∧ m = 8 :=
by sorry

end NUMINAMATH_CALUDE_mooncake_problem_l2057_205756


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2057_205764

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + k * x = -5) → 
  (3 : ℝ) ∈ {x : ℝ | 3 * x^2 + k * x = -5} → 
  (5/9 : ℝ) ∈ {x : ℝ | 3 * x^2 + k * x = -5} :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2057_205764


namespace NUMINAMATH_CALUDE_intersection_circle_origin_implies_a_plusminus_one_no_symmetric_intersection_l2057_205729

/-- The line equation y = ax + 1 -/
def line_equation (a x y : ℝ) : Prop := y = a * x + 1

/-- The hyperbola equation 3x^2 - y^2 = 1 -/
def hyperbola_equation (x y : ℝ) : Prop := 3 * x^2 - y^2 = 1

/-- Two points A(x₁, y₁) and B(x₂, y₂) are the intersection of the line and hyperbola -/
def intersection_points (a x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_equation a x₁ y₁ ∧ hyperbola_equation x₁ y₁ ∧
  line_equation a x₂ y₂ ∧ hyperbola_equation x₂ y₂

/-- The circle with diameter AB passes through the origin -/
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

/-- Two points are symmetric about the line y = (1/2)x -/
def symmetric_about_line (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ + y₂) / 2 = (1 / 2) * ((x₁ + x₂) / 2) ∧
  (y₁ - y₂) / (x₁ - x₂) = -2

theorem intersection_circle_origin_implies_a_plusminus_one :
  ∀ (a x₁ y₁ x₂ y₂ : ℝ),
  intersection_points a x₁ y₁ x₂ y₂ →
  circle_through_origin x₁ y₁ x₂ y₂ →
  a = 1 ∨ a = -1 :=
sorry

theorem no_symmetric_intersection :
  ¬ ∃ (a x₁ y₁ x₂ y₂ : ℝ),
  intersection_points a x₁ y₁ x₂ y₂ ∧
  symmetric_about_line x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_intersection_circle_origin_implies_a_plusminus_one_no_symmetric_intersection_l2057_205729


namespace NUMINAMATH_CALUDE_cone_base_diameter_l2057_205709

/-- A cone with surface area 3π and lateral surface unfolding to a semicircle has base diameter 2 -/
theorem cone_base_diameter (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  (π * r^2 + π * r * l = 3 * π) → 
  (π * l = 2 * π * r) → 
  (2 * r = 2) := by
  sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l2057_205709


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l2057_205780

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ y => 6 * y^2 - 29 * y + 24
  ∃ y : ℝ, f y = 0 ∧ ∀ z : ℝ, f z = 0 → y ≤ z ∧ y = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l2057_205780


namespace NUMINAMATH_CALUDE_correct_average_l2057_205753

theorem correct_average (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 18 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n * initial_avg - wrong_num + correct_num) / n = 19 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l2057_205753


namespace NUMINAMATH_CALUDE_best_athlete_is_A_l2057_205737

structure Athlete where
  name : String
  average : ℝ
  variance : ℝ

def betterPerformance (a b : Athlete) : Prop :=
  a.average > b.average ∨ (a.average = b.average ∧ a.variance < b.variance)

theorem best_athlete_is_A :
  let A := Athlete.mk "A" 185 3.6
  let B := Athlete.mk "B" 180 3.6
  let C := Athlete.mk "C" 185 7.4
  let D := Athlete.mk "D" 180 8.1
  ∀ x ∈ [B, C, D], betterPerformance A x := by
  sorry

end NUMINAMATH_CALUDE_best_athlete_is_A_l2057_205737


namespace NUMINAMATH_CALUDE_division_remainder_division_remainder_is_200000_l2057_205747

theorem division_remainder : ℤ → Prop :=
  fun r => ((8 * 10^9) / (4 * 10^4)) % (10^6) = r

theorem division_remainder_is_200000 : division_remainder 200000 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_division_remainder_is_200000_l2057_205747


namespace NUMINAMATH_CALUDE_simplify_fraction_1_l2057_205739

theorem simplify_fraction_1 (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1) :
  1 / x + 1 / (x * (x - 1)) = 1 / (x - 1) := by
  sorry


end NUMINAMATH_CALUDE_simplify_fraction_1_l2057_205739


namespace NUMINAMATH_CALUDE_birthday_celebration_attendance_l2057_205733

theorem birthday_celebration_attendance (total_guests : ℕ) 
  (women_ratio : ℚ) (men_count : ℕ) (men_left_ratio : ℚ) (children_left : ℕ) : 
  total_guests = 60 →
  women_ratio = 1/2 →
  men_count = 15 →
  men_left_ratio = 1/3 →
  children_left = 5 →
  ∃ (stayed : ℕ), stayed = 50 := by
  sorry

end NUMINAMATH_CALUDE_birthday_celebration_attendance_l2057_205733


namespace NUMINAMATH_CALUDE_combined_degrees_theorem_l2057_205700

/-- Represents the budget allocation for Megatech Corporation's research and development --/
structure BudgetAllocation where
  microphotonics : Float
  homeElectronics : Float
  foodAdditives : Float
  geneticallyModifiedMicroorganisms : Float
  industrialLubricants : Float
  nanotechnology : Float

/-- Calculates the degrees in a circle graph for a given percentage --/
def percentageToDegrees (percentage : Float) : Float :=
  percentage * 360 / 100

/-- Calculates the combined degrees for basic astrophysics and nanotechnology --/
def combinedDegrees (budget : BudgetAllocation) : Float :=
  let basicAstrophysics := 100 - (budget.microphotonics + budget.homeElectronics + 
                                  budget.foodAdditives + budget.geneticallyModifiedMicroorganisms + 
                                  budget.industrialLubricants + budget.nanotechnology)
  percentageToDegrees (basicAstrophysics + budget.nanotechnology)

/-- Theorem: The combined degrees for basic astrophysics and nanotechnology is 50.4 --/
theorem combined_degrees_theorem (budget : BudgetAllocation) 
  (h1 : budget.microphotonics = 10)
  (h2 : budget.homeElectronics = 24)
  (h3 : budget.foodAdditives = 15)
  (h4 : budget.geneticallyModifiedMicroorganisms = 29)
  (h5 : budget.industrialLubricants = 8)
  (h6 : budget.nanotechnology = 7) :
  combinedDegrees budget = 50.4 := by
  sorry


end NUMINAMATH_CALUDE_combined_degrees_theorem_l2057_205700


namespace NUMINAMATH_CALUDE_course_duration_l2057_205725

theorem course_duration (total_hours : ℕ) (class_hours_1 : ℕ) (class_hours_2 : ℕ) (homework_hours : ℕ) :
  total_hours = 336 →
  class_hours_1 = 3 →
  class_hours_2 = 4 →
  homework_hours = 4 →
  (2 * class_hours_1 + class_hours_2 + homework_hours) * 24 = total_hours :=
by
  sorry

end NUMINAMATH_CALUDE_course_duration_l2057_205725


namespace NUMINAMATH_CALUDE_minimum_guests_l2057_205784

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (min_guests : ℕ) :
  total_food = 411 →
  max_per_guest = 2.5 →
  min_guests = ⌈total_food / max_per_guest⌉ →
  min_guests = 165 := by
  sorry

end NUMINAMATH_CALUDE_minimum_guests_l2057_205784


namespace NUMINAMATH_CALUDE_jack_and_jill_probability_l2057_205704

/-- The probability of selecting both Jack and Jill when choosing 2 workers at random -/
def probability : ℚ := 1/6

/-- The number of other workers besides Jack and Jill -/
def other_workers : ℕ := 2

/-- The total number of workers -/
def total_workers : ℕ := other_workers + 2

theorem jack_and_jill_probability :
  (1 : ℚ) / (total_workers.choose 2) = probability → other_workers = 2 := by
  sorry

end NUMINAMATH_CALUDE_jack_and_jill_probability_l2057_205704


namespace NUMINAMATH_CALUDE_inequality_range_l2057_205740

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-5 : ℝ) 0, x^2 + 2*x - 3 + a ≤ 0) ↔ a ∈ Set.Iic (-12 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l2057_205740


namespace NUMINAMATH_CALUDE_committee_selection_ways_l2057_205782

/-- The number of ways to choose two committees from a club -/
def choose_committees (total_members : ℕ) (exec_size : ℕ) (aux_size : ℕ) : ℕ :=
  Nat.choose total_members exec_size * Nat.choose (total_members - exec_size) aux_size

/-- Theorem stating the number of ways to choose committees from a 30-member club -/
theorem committee_selection_ways :
  choose_committees 30 5 3 = 327764800 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l2057_205782


namespace NUMINAMATH_CALUDE_closest_to_target_l2057_205795

def options : List ℝ := [-4, -3, 0, 3, 4]

def target : ℝ := -3.4

def distance (x y : ℝ) : ℝ := |x - y|

theorem closest_to_target :
  ∃ (closest : ℝ), closest ∈ options ∧
    (∀ x ∈ options, distance target closest ≤ distance target x) ∧
    closest = -3 := by
  sorry

end NUMINAMATH_CALUDE_closest_to_target_l2057_205795


namespace NUMINAMATH_CALUDE_morgan_lunch_change_l2057_205721

/-- Calculates the change Morgan receives from his lunch order --/
theorem morgan_lunch_change : 
  let hamburger : ℚ := 5.75
  let onion_rings : ℚ := 2.50
  let smoothie : ℚ := 3.25
  let side_salad : ℚ := 3.75
  let chocolate_cake : ℚ := 4.20
  let discount_rate : ℚ := 0.10
  let tax_rate : ℚ := 0.06
  let payment : ℚ := 50

  let total_before_discount : ℚ := hamburger + onion_rings + smoothie + side_salad + chocolate_cake
  let discount : ℚ := (side_salad + chocolate_cake) * discount_rate
  let total_after_discount : ℚ := total_before_discount - discount
  let tax : ℚ := total_after_discount * tax_rate
  let final_total : ℚ := total_after_discount + tax
  let change : ℚ := payment - final_total

  change = 30.34 := by sorry

end NUMINAMATH_CALUDE_morgan_lunch_change_l2057_205721


namespace NUMINAMATH_CALUDE_monic_polynomial_divisibility_l2057_205749

open Polynomial

theorem monic_polynomial_divisibility (n k : ℕ) (h_pos_n : n > 0) (h_pos_k : k > 0) :
  ∀ (f : Polynomial ℤ),
    Monic f →
    (Polynomial.degree f = n) →
    (∀ (a : ℤ), f.eval a ≠ 0 → (f.eval a ∣ f.eval (2 * a ^ k))) →
    f = X ^ n :=
by sorry

end NUMINAMATH_CALUDE_monic_polynomial_divisibility_l2057_205749


namespace NUMINAMATH_CALUDE_inequality_proof_l2057_205796

theorem inequality_proof (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2057_205796


namespace NUMINAMATH_CALUDE_expand_expression_l2057_205703

-- Statement of the theorem
theorem expand_expression (x : ℝ) : (x + 3) * (6 * x - 12) = 6 * x^2 + 6 * x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2057_205703


namespace NUMINAMATH_CALUDE_power_relation_l2057_205735

theorem power_relation (a : ℝ) (m n : ℤ) (hm : a ^ m = 4) (hn : a ^ n = 2) :
  a ^ (m - 2 * n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l2057_205735


namespace NUMINAMATH_CALUDE_physics_score_l2057_205783

/-- Represents the scores in physics, chemistry, and mathematics -/
structure Scores where
  physics : ℕ
  chemistry : ℕ
  mathematics : ℕ

/-- The average score of all three subjects is 60 -/
def average_all (s : Scores) : Prop :=
  (s.physics + s.chemistry + s.mathematics) / 3 = 60

/-- The average score of physics and mathematics is 90 -/
def average_physics_math (s : Scores) : Prop :=
  (s.physics + s.mathematics) / 2 = 90

/-- The average score of physics and chemistry is 70 -/
def average_physics_chem (s : Scores) : Prop :=
  (s.physics + s.chemistry) / 2 = 70

/-- Theorem stating that given the conditions, the physics score is 140 -/
theorem physics_score (s : Scores) 
  (h1 : average_all s)
  (h2 : average_physics_math s)
  (h3 : average_physics_chem s) :
  s.physics = 140 := by
  sorry

end NUMINAMATH_CALUDE_physics_score_l2057_205783


namespace NUMINAMATH_CALUDE_rectangle_with_perpendicular_diagonals_is_square_l2057_205765

-- Define a rectangle
structure Rectangle :=
  (a b : ℝ)
  (a_positive : a > 0)
  (b_positive : b > 0)

-- Define a property for perpendicular diagonals
def has_perpendicular_diagonals (r : Rectangle) : Prop :=
  r.a^2 = r.b^2

-- Define a square as a special case of rectangle
def is_square (r : Rectangle) : Prop :=
  r.a = r.b

-- Theorem statement
theorem rectangle_with_perpendicular_diagonals_is_square 
  (r : Rectangle) (h : has_perpendicular_diagonals r) : 
  is_square r :=
sorry

end NUMINAMATH_CALUDE_rectangle_with_perpendicular_diagonals_is_square_l2057_205765


namespace NUMINAMATH_CALUDE_complex_number_representation_l2057_205728

theorem complex_number_representation : ∃ (z : ℂ), z = 1 + 2*I := by sorry

end NUMINAMATH_CALUDE_complex_number_representation_l2057_205728


namespace NUMINAMATH_CALUDE_no_solutions_to_inequality_system_l2057_205712

theorem no_solutions_to_inequality_system :
  ¬ ∃ (x y : ℝ), 11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧ 5 * x + y ≤ -10 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_to_inequality_system_l2057_205712


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_factorial_l2057_205761

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_product (n : Nat) : Nat :=
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

def is_five_digit (n : Nat) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_with_product_factorial :
  ∃ (n : Nat), is_five_digit n ∧
               digit_product n = factorial 8 ∧
               ∀ (m : Nat), is_five_digit m ∧ digit_product m = factorial 8 → m ≤ n :=
by
  use 98752
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_factorial_l2057_205761


namespace NUMINAMATH_CALUDE_light_bulb_configurations_l2057_205770

/-- The number of light bulbs -/
def num_bulbs : ℕ := 5

/-- The number of states each bulb can have (on or off) -/
def states_per_bulb : ℕ := 2

/-- The total number of possible lighting configurations -/
def total_configurations : ℕ := states_per_bulb ^ num_bulbs

theorem light_bulb_configurations :
  total_configurations = 32 :=
by sorry

end NUMINAMATH_CALUDE_light_bulb_configurations_l2057_205770


namespace NUMINAMATH_CALUDE_functional_equation_problem_l2057_205760

/-- The functional equation problem -/
theorem functional_equation_problem (α : ℝ) (hα : α ≠ 0) :
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, f (f (x + y)) = f (x + y) + f x * f y + α * x * y) ↔
  (α = -1 ∧ ∃! f : ℝ → ℝ, ∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l2057_205760


namespace NUMINAMATH_CALUDE_coin_grid_probability_l2057_205708

/-- Represents a square grid -/
structure Grid where
  size : ℕ  -- number of squares on each side
  square_size : ℝ  -- side length of each square
  
/-- Represents a circular coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of a coin landing in a winning position on a grid -/
def winning_probability (g : Grid) (c : Coin) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem coin_grid_probability :
  let g : Grid := { size := 5, square_size := 10 }
  let c : Coin := { diameter := 8 }
  winning_probability g c = 25 / 441 := by
  sorry

end NUMINAMATH_CALUDE_coin_grid_probability_l2057_205708


namespace NUMINAMATH_CALUDE_managers_salary_proof_l2057_205779

def prove_managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) : Prop :=
  let total_salary := num_employees * avg_salary
  let new_avg := avg_salary + avg_increase
  let new_total := (num_employees + 1) * new_avg
  new_total - total_salary = 3800

theorem managers_salary_proof :
  prove_managers_salary 20 1700 100 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_proof_l2057_205779


namespace NUMINAMATH_CALUDE_shooting_score_proof_l2057_205781

theorem shooting_score_proof (total_shots : ℕ) (total_score : ℕ) (ten_point_shots : ℕ) (remaining_shots : ℕ) :
  total_shots = 10 →
  total_score = 90 →
  ten_point_shots = 4 →
  remaining_shots = total_shots - ten_point_shots →
  (∃ (seven_point_shots eight_point_shots nine_point_shots : ℕ),
    seven_point_shots + eight_point_shots + nine_point_shots = remaining_shots ∧
    7 * seven_point_shots + 8 * eight_point_shots + 9 * nine_point_shots = total_score - 10 * ten_point_shots) →
  (∃ (nine_point_shots : ℕ), nine_point_shots = 3) :=
by sorry

end NUMINAMATH_CALUDE_shooting_score_proof_l2057_205781


namespace NUMINAMATH_CALUDE_julians_initial_debt_l2057_205790

/-- Given that Julian will owe Jenny 28 dollars if he borrows 8 dollars more,
    prove that Julian's initial debt to Jenny is 20 dollars. -/
theorem julians_initial_debt (current_debt additional_borrow total_after_borrow : ℕ) :
  additional_borrow = 8 →
  total_after_borrow = 28 →
  total_after_borrow = current_debt + additional_borrow →
  current_debt = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_julians_initial_debt_l2057_205790


namespace NUMINAMATH_CALUDE_rationalize_sum_l2057_205719

/-- Represents a fraction with a cube root in the denominator -/
structure CubeRootFraction where
  numerator : ℚ
  denominator : ℚ
  root : ℕ

/-- Represents a rationalized fraction with a cube root in the numerator -/
structure RationalizedFraction where
  A : ℤ
  B : ℕ
  C : ℕ

/-- Checks if a number is not divisible by the cube of any prime -/
def not_divisible_by_cube_of_prime (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^3 ∣ n)

/-- Rationalizes a fraction with a cube root in the denominator -/
def rationalize (f : CubeRootFraction) : RationalizedFraction :=
  sorry

theorem rationalize_sum (f : CubeRootFraction) 
  (h : f = { numerator := 2, denominator := 3, root := 7 }) :
  let r := rationalize f
  r.A + r.B + r.C = 72 ∧ 
  r.C > 0 ∧
  not_divisible_by_cube_of_prime r.B :=
sorry

end NUMINAMATH_CALUDE_rationalize_sum_l2057_205719


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2057_205714

/-- An ellipse with equation x^2 + 9y^2 = 9 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + 9 * p.2^2 = 9}

/-- An isosceles triangle inscribed in the ellipse -/
structure IsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_in_ellipse : A ∈ Ellipse ∧ B ∈ Ellipse ∧ C ∈ Ellipse
  h_isosceles : dist A B = dist A C
  h_vertex_at_origin : A = (0, 1)
  h_altitude_on_y_axis : B.1 + C.1 = 0 ∧ B.2 = C.2

/-- The square of the length of the equal sides of the isosceles triangle -/
def squareLengthEqualSides (t : IsoscelesTriangle) : ℝ :=
  (dist t.A t.B)^2

/-- The main theorem -/
theorem isosceles_triangle_side_length 
  (t : IsoscelesTriangle) : squareLengthEqualSides t = 108/25 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2057_205714


namespace NUMINAMATH_CALUDE_linda_furniture_spending_l2057_205722

theorem linda_furniture_spending (original_savings : ℝ) (tv_cost : ℝ) 
  (h1 : original_savings = 1800)
  (h2 : tv_cost = 450) :
  (original_savings - tv_cost) / original_savings = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_linda_furniture_spending_l2057_205722


namespace NUMINAMATH_CALUDE_increase_percentage_theorem_l2057_205798

theorem increase_percentage_theorem (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hM : M > 0) (hpq : q < p) :
  M * (1 + p / 100) * (1 + q / 100) > M ↔ (p > 0 ∧ q > 0) :=
by sorry

end NUMINAMATH_CALUDE_increase_percentage_theorem_l2057_205798


namespace NUMINAMATH_CALUDE_parabola_equation_l2057_205701

/-- A parabola with vertex at the origin and axis at x = 3/2 has the equation y² = -6x -/
theorem parabola_equation (y x : ℝ) : 
  (∀ (p : ℝ), p > 0 → y^2 = -2*p*x) → -- General equation of parabola with vertex at origin
  (3/2 : ℝ) = p/2 →                   -- Axis of parabola is at x = 3/2
  y^2 = -6*x :=                       -- Equation to be proved
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2057_205701


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2057_205726

theorem inequality_solution_set (x : ℝ) : 
  (x - 1)^2023 - 2^2023 * x^2023 ≤ x + 1 ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2057_205726


namespace NUMINAMATH_CALUDE_prob_three_unused_correct_expected_hits_nine_targets_correct_l2057_205786

/-- Rocket artillery system model -/
structure RocketSystem where
  total_rockets : ℕ
  hit_probability : ℝ

/-- Probability of exactly three unused rockets after firing at five targets -/
def prob_three_unused (system : RocketSystem) : ℝ :=
  10 * system.hit_probability^3 * (1 - system.hit_probability)^2

/-- Expected number of targets hit when firing at nine targets -/
def expected_hits_nine_targets (system : RocketSystem) : ℝ :=
  10 * system.hit_probability - system.hit_probability^10

/-- Theorem: Probability of exactly three unused rockets after firing at five targets -/
theorem prob_three_unused_correct (system : RocketSystem) :
  prob_three_unused system = 10 * system.hit_probability^3 * (1 - system.hit_probability)^2 := by
  sorry

/-- Theorem: Expected number of targets hit when firing at nine targets -/
theorem expected_hits_nine_targets_correct (system : RocketSystem) :
  expected_hits_nine_targets system = 10 * system.hit_probability - system.hit_probability^10 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_unused_correct_expected_hits_nine_targets_correct_l2057_205786


namespace NUMINAMATH_CALUDE_digit_57_is_5_l2057_205716

/-- The decimal expansion of 21/22 has a repeating pattern of "54" -/
def repeating_pattern : ℕ → ℕ
  | n => if n % 2 = 0 then 4 else 5

/-- The 57th digit after the decimal point in the expansion of 21/22 -/
def digit_57 : ℕ := repeating_pattern 56

theorem digit_57_is_5 : digit_57 = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_57_is_5_l2057_205716


namespace NUMINAMATH_CALUDE_remainder_theorem_l2057_205758

theorem remainder_theorem (n : ℤ) : 
  (2 * n) % 11 = 2 → n % 22 = 1 :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2057_205758


namespace NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l2057_205715

/-- Given two planes α and β with normal vectors, prove that if they are parallel, then k = 4 -/
theorem parallel_planes_normal_vectors (k : ℝ) : 
  let n_alpha : ℝ × ℝ × ℝ := (1, 2, -2)
  let n_beta : ℝ × ℝ × ℝ := (-2, -4, k)
  (∃ (c : ℝ), c ≠ 0 ∧ n_alpha = c • n_beta) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l2057_205715


namespace NUMINAMATH_CALUDE_tetromino_tiling_divisibility_l2057_205793

/-- Represents a T-tetromino tile -/
structure TTetromino :=
  (size : Nat)
  (shape : Unit)
  (h_size : size = 4)

/-- Represents a rectangle that can be tiled with T-tetrominoes -/
structure TileableRectangle :=
  (m n : Nat)
  (tiles : List TTetromino)
  (h_tiling : tiles.length * 4 = m * n)  -- Complete tiling without gaps or overlaps

/-- 
If a rectangle can be tiled with T-tetrominoes, then its dimensions are divisible by 4 
-/
theorem tetromino_tiling_divisibility (rect : TileableRectangle) : 
  4 ∣ rect.m ∧ 4 ∣ rect.n :=
sorry

end NUMINAMATH_CALUDE_tetromino_tiling_divisibility_l2057_205793


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l2057_205741

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  abs (l.a * x₀ + l.b * y₀ + l.c) / Real.sqrt (l.a^2 + l.b^2) = c.radius

/-- The main theorem -/
theorem tangent_lines_to_circle 
  (c : Circle) 
  (p : ℝ × ℝ) 
  (h_circle : c.center = (1, 1) ∧ c.radius = 1) 
  (h_point : p = (2, 3)) :
  ∃ (l₁ l₂ : Line),
    isTangent l₁ c ∧ isTangent l₂ c ∧
    (l₁.a = 1 ∧ l₁.b = 0 ∧ l₁.c = -2) ∧
    (l₂.a = 3 ∧ l₂.b = -4 ∧ l₂.c = 6) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l2057_205741


namespace NUMINAMATH_CALUDE_computer_employee_savings_l2057_205738

/-- Calculates the employee savings on a computer purchase given the initial cost,
    markup percentage, and employee discount percentage. -/
def employeeSavings (initialCost : ℝ) (markupPercentage : ℝ) (discountPercentage : ℝ) : ℝ :=
  let retailPrice := initialCost * (1 + markupPercentage)
  retailPrice * discountPercentage

/-- Theorem stating that an employee saves $86.25 when buying a computer
    with a 15% markup and 15% employee discount, given an initial cost of $500. -/
theorem computer_employee_savings :
  employeeSavings 500 0.15 0.15 = 86.25 := by
  sorry


end NUMINAMATH_CALUDE_computer_employee_savings_l2057_205738


namespace NUMINAMATH_CALUDE_problem_solution_l2057_205717

theorem problem_solution : 
  (∃ n : ℕ, 25 = 5 * n) ∧ 
  (∃ m : ℕ, 209 = 19 * m) ∧ ¬(∃ k : ℕ, 63 = 19 * k) ∧
  (∃ p : ℕ, 180 = 9 * p) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2057_205717


namespace NUMINAMATH_CALUDE_negation_of_existence_real_roots_l2057_205711

theorem negation_of_existence_real_roots :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_real_roots_l2057_205711


namespace NUMINAMATH_CALUDE_person_age_in_1900_l2057_205791

theorem person_age_in_1900 (birth_year : ℕ) (death_year : ℕ) (age_at_death : ℕ) :
  (age_at_death = birth_year / 29) →
  (birth_year < 1900) →
  (1901 ≤ death_year) →
  (death_year ≤ 1930) →
  (death_year = birth_year + age_at_death) →
  (1900 - birth_year = 44) :=
by sorry

end NUMINAMATH_CALUDE_person_age_in_1900_l2057_205791


namespace NUMINAMATH_CALUDE_dividend_remainder_proof_l2057_205778

theorem dividend_remainder_proof (D d q r : ℕ) : 
  D = 18972 → d = 526 → q = 36 → D = d * q + r → r = 36 := by
  sorry

end NUMINAMATH_CALUDE_dividend_remainder_proof_l2057_205778


namespace NUMINAMATH_CALUDE_squirrel_walnuts_l2057_205752

/-- The number of walnuts left in the squirrels' burrow after their gathering and eating activities. -/
def walnuts_left (initial : ℕ) (boy_gathered : ℕ) (boy_dropped : ℕ) (girl_brought : ℕ) (girl_ate : ℕ) : ℕ :=
  initial + (boy_gathered - boy_dropped) + girl_brought - girl_ate

/-- Theorem stating that given the specific conditions of the problem, the number of walnuts left is 20. -/
theorem squirrel_walnuts : walnuts_left 12 6 1 5 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_walnuts_l2057_205752


namespace NUMINAMATH_CALUDE_m_range_l2057_205730

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 ≠ 0

def q (m : ℝ) : Prop := m > 2

-- Define the condition that either p or q is true, but not both
def condition (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬(p m ∧ q m)

-- State the theorem
theorem m_range (m : ℝ) : condition m → ((-2 < m ∧ m < 2) ∨ m > 2) := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2057_205730


namespace NUMINAMATH_CALUDE_exactly_two_numbers_satisfy_l2057_205775

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

def satisfies_condition (n : ℕ) : Prop :=
  n < 500 ∧ n = 7 * sum_of_digits n ∧ is_prime (sum_of_digits n)

theorem exactly_two_numbers_satisfy :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_condition n) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_numbers_satisfy_l2057_205775


namespace NUMINAMATH_CALUDE_max_cables_cut_l2057_205750

/-- Represents a computer network -/
structure ComputerNetwork where
  numComputers : ℕ
  numCables : ℕ
  numClusters : ℕ

/-- The initial state of the computer network -/
def initialNetwork : ComputerNetwork :=
  { numComputers := 200
  , numCables := 345
  , numClusters := 1 }

/-- The final state of the computer network after cutting cables -/
def finalNetwork : ComputerNetwork :=
  { numComputers := 200
  , numCables := 345 - 153
  , numClusters := 8 }

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut (initial : ComputerNetwork) (final : ComputerNetwork) :
  initial.numComputers = 200 →
  initial.numCables = 345 →
  initial.numClusters = 1 →
  final.numComputers = initial.numComputers →
  final.numClusters = 8 →
  final.numCables = initial.numCables - 153 →
  ∀ n : ℕ, n > 153 → 
    ¬∃ (network : ComputerNetwork), 
      network.numComputers = initial.numComputers ∧
      network.numClusters = final.numClusters ∧
      network.numCables = initial.numCables - n :=
by sorry


end NUMINAMATH_CALUDE_max_cables_cut_l2057_205750


namespace NUMINAMATH_CALUDE_probability_theorem_l2057_205754

def total_marbles : ℕ := 8
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 2
def selected_marbles : ℕ := 4

def probability_one_each_plus_red : ℚ :=
  (red_marbles.choose 2 * blue_marbles.choose 1 * green_marbles.choose 1) /
  total_marbles.choose selected_marbles

theorem probability_theorem :
  probability_one_each_plus_red = 9 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2057_205754


namespace NUMINAMATH_CALUDE_laundry_cost_theorem_l2057_205762

/-- Represents the cost per load of laundry in EUR cents -/
def cost_per_load (loads_per_bottle : ℕ) (regular_price : ℚ) (sale_price : ℚ) 
  (tax_rate : ℚ) (coupon : ℚ) (conversion_rate : ℚ) : ℚ :=
  let total_loads := 2 * loads_per_bottle
  let pre_tax_cost := 2 * sale_price - coupon
  let total_cost := pre_tax_cost * (1 + tax_rate)
  let cost_in_eur := total_cost * conversion_rate
  (cost_in_eur * 100) / total_loads

theorem laundry_cost_theorem (loads_per_bottle : ℕ) (regular_price : ℚ) 
  (sale_price : ℚ) (tax_rate : ℚ) (coupon : ℚ) (conversion_rate : ℚ) :
  loads_per_bottle = 80 →
  regular_price = 25 →
  sale_price = 20 →
  tax_rate = 0.05 →
  coupon = 5 →
  conversion_rate = 0.85 →
  ∃ (n : ℕ), n ≤ 20 ∧ 20 < n + 1 ∧ 
    cost_per_load loads_per_bottle regular_price sale_price tax_rate coupon conversion_rate < n + 1 ∧
    n < cost_per_load loads_per_bottle regular_price sale_price tax_rate coupon conversion_rate := by
  sorry

end NUMINAMATH_CALUDE_laundry_cost_theorem_l2057_205762


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2057_205713

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0) → 0 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2057_205713


namespace NUMINAMATH_CALUDE_negation_equivalence_l2057_205771

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 < 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2057_205771


namespace NUMINAMATH_CALUDE_cyclist_return_speed_l2057_205774

/-- Calculates the average speed for the return trip of a cyclist -/
theorem cyclist_return_speed (total_distance : ℝ) (first_half_distance : ℝ) (first_speed : ℝ) (second_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 32)
  (h2 : first_half_distance = 16)
  (h3 : first_speed = 8)
  (h4 : second_speed = 10)
  (h5 : total_time = 6.8)
  : (total_distance / (total_time - (first_half_distance / first_speed + (total_distance - first_half_distance) / second_speed))) = 10 := by
  sorry

#check cyclist_return_speed

end NUMINAMATH_CALUDE_cyclist_return_speed_l2057_205774


namespace NUMINAMATH_CALUDE_youth_gathering_count_l2057_205785

/-- The number of youths at a gathering, given the conditions from the problem. -/
def total_youths (male_youths : ℕ) : ℕ := 2 * male_youths + 12

/-- The theorem stating the total number of youths at the gathering. -/
theorem youth_gathering_count : 
  ∃ (male_youths : ℕ), 
    (male_youths : ℚ) / (total_youths male_youths : ℚ) = 9 / 20 ∧ 
    total_youths male_youths = 120 := by
  sorry


end NUMINAMATH_CALUDE_youth_gathering_count_l2057_205785


namespace NUMINAMATH_CALUDE_morning_campers_l2057_205706

theorem morning_campers (afternoon evening total : ℕ) 
  (h1 : afternoon = 13)
  (h2 : evening = 49)
  (h3 : total = 98)
  : total - afternoon - evening = 36 := by
  sorry

end NUMINAMATH_CALUDE_morning_campers_l2057_205706


namespace NUMINAMATH_CALUDE_cos_five_pi_thirds_l2057_205727

theorem cos_five_pi_thirds : Real.cos (5 * π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_thirds_l2057_205727


namespace NUMINAMATH_CALUDE_problem_statement_l2057_205768

theorem problem_statement (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2057_205768


namespace NUMINAMATH_CALUDE_value_calculation_l2057_205767

theorem value_calculation (x : ℝ) (y : ℝ) (h1 : x = 50.0) (h2 : y = 0.20 * x - 4) : y = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l2057_205767


namespace NUMINAMATH_CALUDE_option_D_not_suitable_for_comprehensive_survey_l2057_205743

-- Define the type for survey options
inductive SurveyOption
| A -- Security check for passengers before boarding a plane
| B -- School recruiting teachers and conducting interviews for applicants
| C -- Understanding the extracurricular reading time of seventh-grade students in a school
| D -- Understanding the service life of a batch of light bulbs

-- Define a function to check if an option is suitable for a comprehensive survey
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => True
  | SurveyOption.B => True
  | SurveyOption.C => True
  | SurveyOption.D => False

-- Theorem stating that option D is not suitable for a comprehensive survey
theorem option_D_not_suitable_for_comprehensive_survey :
  ¬(isSuitableForComprehensiveSurvey SurveyOption.D) :=
by sorry

end NUMINAMATH_CALUDE_option_D_not_suitable_for_comprehensive_survey_l2057_205743


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l2057_205744

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

/-- The equation of the line -/
def line (m x y : ℝ) : Prop := (m+2)*x - (m+4)*y + 2-m = 0

/-- Theorem stating that the line always intersects the ellipse -/
theorem line_intersects_ellipse :
  ∀ m : ℝ, ∃ x y : ℝ, ellipse x y ∧ line m x y := by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l2057_205744


namespace NUMINAMATH_CALUDE_bob_baked_36_more_l2057_205799

/-- The number of additional peanut butter cookies Bob baked after the accident -/
def bob_additional_cookies (alice_initial : ℕ) (bob_initial : ℕ) (lost : ℕ) (alice_additional : ℕ) (final_total : ℕ) : ℕ :=
  final_total - ((alice_initial + bob_initial - lost) + alice_additional)

/-- Theorem stating that Bob baked 36 additional cookies given the problem conditions -/
theorem bob_baked_36_more (alice_initial bob_initial lost alice_additional final_total : ℕ) 
  (h1 : alice_initial = 74)
  (h2 : bob_initial = 7)
  (h3 : lost = 29)
  (h4 : alice_additional = 5)
  (h5 : final_total = 93) :
  bob_additional_cookies alice_initial bob_initial lost alice_additional final_total = 36 := by
  sorry

end NUMINAMATH_CALUDE_bob_baked_36_more_l2057_205799


namespace NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l2057_205751

theorem cube_sum_minus_product_eq_2003 :
  ∀ x y z : ℤ, x^3 + y^3 + z^3 - 3*x*y*z = 2003 ↔ 
  ((x = 668 ∧ y = 668 ∧ z = 667) ∨ 
   (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
   (x = 667 ∧ y = 668 ∧ z = 668)) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l2057_205751


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2057_205748

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 3 * x - y^2

-- State the theorem
theorem diamond_equation_solution :
  ∀ x : ℝ, diamond x 7 = 20 → x = 23 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2057_205748


namespace NUMINAMATH_CALUDE_recorder_price_problem_l2057_205776

theorem recorder_price_problem (a b : ℕ) : 
  a < 10 → b < 10 →  -- Ensure a and b are single digits
  10 * b + a < 50 →  -- Old price less than 50
  (10 * a + b : ℚ) = 1.2 * (10 * b + a) →  -- 20% price increase
  a = 5 ∧ b = 4 := by sorry

end NUMINAMATH_CALUDE_recorder_price_problem_l2057_205776


namespace NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l2057_205794

theorem largest_factorial_as_consecutive_product : 
  ∀ n : ℕ, n > 0 → 
  (∃ k : ℕ, k > 0 ∧ n.factorial = (List.range (n - 5)).prod.succ) → 
  n ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l2057_205794


namespace NUMINAMATH_CALUDE_completing_square_sum_l2057_205718

theorem completing_square_sum (a b : ℝ) : 
  (∀ x, x^2 - 4*x = 5 ↔ (x + a)^2 = b) → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_sum_l2057_205718


namespace NUMINAMATH_CALUDE_redistribution_contribution_l2057_205723

theorem redistribution_contribution (earnings : Fin 5 → ℕ) 
  (h1 : earnings 0 = 18)
  (h2 : earnings 1 = 23)
  (h3 : earnings 2 = 30)
  (h4 : earnings 3 = 35)
  (h5 : earnings 4 = 50)
  (min_amount : ℕ := 30)
  : (earnings 4 - min_amount : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_redistribution_contribution_l2057_205723


namespace NUMINAMATH_CALUDE_range_of_m_l2057_205773

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : B m ⊆ A → m < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2057_205773


namespace NUMINAMATH_CALUDE_solve_equation_l2057_205757

theorem solve_equation (y : ℚ) (h : 1/3 - 1/4 = 1/y) : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2057_205757


namespace NUMINAMATH_CALUDE_inequality_solution_set_quadratic_inequality_solution_set_l2057_205720

-- Part 1
theorem inequality_solution_set (x : ℝ) :
  (2 - x) / (x + 3) > 0 ↔ x ∈ Set.Ioo (-3 : ℝ) 2 :=
sorry

-- Part 2
theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_quadratic_inequality_solution_set_l2057_205720


namespace NUMINAMATH_CALUDE_car_selling_problem_l2057_205742

/-- Represents the selling price and profit information for two types of cars -/
structure CarInfo where
  price_a : ℕ  -- Selling price of type A car in yuan
  price_b : ℕ  -- Selling price of type B car in yuan
  profit_a : ℕ  -- Profit from selling one type A car in yuan
  profit_b : ℕ  -- Profit from selling one type B car in yuan

/-- Represents a purchasing plan -/
structure PurchasePlan where
  count_a : ℕ  -- Number of type A cars purchased
  count_b : ℕ  -- Number of type B cars purchased

/-- Theorem stating the properties of the car selling problem -/
theorem car_selling_problem (info : CarInfo) 
  (h1 : 2 * info.price_a + 3 * info.price_b = 800000)
  (h2 : 3 * info.price_a + 2 * info.price_b = 950000)
  (h3 : info.profit_a = 8000)
  (h4 : info.profit_b = 5000) :
  info.price_a = 250000 ∧ 
  info.price_b = 100000 ∧ 
  (∃ (plans : Finset PurchasePlan), 
    (∀ plan ∈ plans, plan.count_a * info.price_a + plan.count_b * info.price_b = 2000000) ∧
    plans.card = 3 ∧
    (∀ plan ∈ plans, ∀ other_plan : PurchasePlan, 
      other_plan.count_a * info.price_a + other_plan.count_b * info.price_b = 2000000 →
      other_plan ∈ plans)) ∧
  (∃ (max_profit : ℕ), 
    max_profit = 91000 ∧
    ∀ plan : PurchasePlan, 
      plan.count_a * info.price_a + plan.count_b * info.price_b = 2000000 →
      plan.count_a * info.profit_a + plan.count_b * info.profit_b ≤ max_profit) := by
  sorry


end NUMINAMATH_CALUDE_car_selling_problem_l2057_205742


namespace NUMINAMATH_CALUDE_cubic_real_root_l2057_205724

theorem cubic_real_root (a b : ℝ) :
  (∃ x : ℂ, a * x^3 + 4 * x^2 + b * x - 35 = 0 ∧ x = -1 - 2*I) →
  (∃ x : ℝ, a * x^3 + 4 * x^2 + b * x - 35 = 0 ∧ x = 21/5) :=
by sorry

end NUMINAMATH_CALUDE_cubic_real_root_l2057_205724


namespace NUMINAMATH_CALUDE_optimal_journey_solution_l2057_205705

/-- Represents the problem setup for the journey from M to N --/
structure JourneySetup where
  total_distance : ℝ
  walking_speed : ℝ
  cycling_speed : ℝ

/-- Represents the optimal solution for the journey --/
structure OptimalSolution where
  c_departure_time : ℝ
  walking_distance : ℝ
  cycling_distance : ℝ

/-- Theorem stating the optimal solution for the journey --/
theorem optimal_journey_solution (setup : JourneySetup) 
  (h1 : setup.total_distance = 15)
  (h2 : setup.walking_speed = 6)
  (h3 : setup.cycling_speed = 15) :
  ∃ (sol : OptimalSolution), 
    sol.c_departure_time = 3 / 11 ∧
    sol.walking_distance = 60 / 11 ∧
    sol.cycling_distance = 105 / 11 ∧
    (sol.walking_distance / setup.walking_speed + 
     sol.cycling_distance / setup.cycling_speed = 
     setup.total_distance / setup.cycling_speed + 
     sol.walking_distance / setup.walking_speed) ∧
    ∀ (other : OptimalSolution), 
      (other.walking_distance / setup.walking_speed + 
       other.cycling_distance / setup.cycling_speed ≥
       sol.walking_distance / setup.walking_speed + 
       sol.cycling_distance / setup.cycling_speed) :=
by sorry


end NUMINAMATH_CALUDE_optimal_journey_solution_l2057_205705


namespace NUMINAMATH_CALUDE_surface_area_of_modified_cube_l2057_205789

/-- Represents the structure of the cube after removals -/
structure ModifiedCube where
  initial_size : Nat
  small_cube_size : Nat
  removed_cubes : Nat
  remaining_cubes : Nat

/-- Calculates the surface area of the modified cube structure -/
def surface_area (cube : ModifiedCube) : Nat :=
  sorry

/-- Theorem stating the surface area of the specific cube structure -/
theorem surface_area_of_modified_cube :
  let cube : ModifiedCube := {
    initial_size := 12,
    small_cube_size := 3,
    removed_cubes := 12,
    remaining_cubes := 52
  }
  surface_area cube = 4020 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_modified_cube_l2057_205789


namespace NUMINAMATH_CALUDE_shirts_not_all_on_sale_l2057_205787

-- Define the universe of discourse
variable (Shirt : Type)
-- Define the property of being on sale
variable (on_sale : Shirt → Prop)
-- Define the property of being in the store
variable (in_store : Shirt → Prop)

-- Theorem statement
theorem shirts_not_all_on_sale 
  (h : ¬ (∀ s : Shirt, in_store s → on_sale s)) : 
  (∃ s : Shirt, in_store s ∧ ¬ on_sale s) ∧ 
  (¬ (∀ s : Shirt, in_store s → on_sale s)) := by
  sorry


end NUMINAMATH_CALUDE_shirts_not_all_on_sale_l2057_205787


namespace NUMINAMATH_CALUDE_incorrect_expression_l2057_205788

/-- Represents a repeating decimal with a 3-digit non-repeating part and a 2-digit repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℕ  -- Represents P (3-digit non-repeating part)
  repeating : ℕ     -- Represents Q (2-digit repeating part)
  nonRepeating_three_digits : nonRepeating < 1000
  repeating_two_digits : repeating < 100

/-- Converts a RepeatingDecimal to its decimal representation -/
def toDecimal (d : RepeatingDecimal) : ℚ :=
  (d.nonRepeating : ℚ) / 1000 + (d.repeating : ℚ) / 99900

/-- The statement that the given expression is incorrect -/
theorem incorrect_expression (d : RepeatingDecimal) :
  ¬(10^3 * (10^2 - 1) * toDecimal d = (d.repeating : ℚ) * (100 * d.nonRepeating - 1)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l2057_205788


namespace NUMINAMATH_CALUDE_cubic_inequality_l2057_205746

theorem cubic_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2057_205746


namespace NUMINAMATH_CALUDE_saturday_attendance_l2057_205766

theorem saturday_attendance (price : ℝ) (total_earnings : ℝ) : 
  price = 10 →
  total_earnings = 300 →
  ∃ (saturday : ℕ),
    saturday * price + (saturday / 2) * price = total_earnings ∧
    saturday = 20 := by
  sorry

end NUMINAMATH_CALUDE_saturday_attendance_l2057_205766


namespace NUMINAMATH_CALUDE_g_inverse_equals_g_l2057_205707

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (3 * x + 4) / (k * x - 3)

theorem g_inverse_equals_g (k : ℝ) :
  k ≠ -4/3 →
  ∀ x : ℝ, g k (g k x) = x :=
sorry

end NUMINAMATH_CALUDE_g_inverse_equals_g_l2057_205707


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l2057_205792

-- Define the points
def start_point : ℝ × ℝ := (-1, 3)
def end_point : ℝ × ℝ := (4, 6)

-- Define the reflection surface (x-axis)
def reflection_axis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the reflected ray
def reflected_ray : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (1 - t) • start_point + t • end_point}

-- Theorem statement
theorem reflected_ray_equation :
  ∀ p ∈ reflected_ray, 9 * p.1 - 5 * p.2 - 6 = 0 :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l2057_205792


namespace NUMINAMATH_CALUDE_ray_walks_11_blocks_home_l2057_205769

/-- Represents Ray's dog walking routine -/
structure DogWalk where
  trips_per_day : ℕ
  total_blocks_per_day : ℕ
  blocks_to_park : ℕ
  blocks_to_school : ℕ

/-- Calculates the number of blocks Ray walks to get back home -/
def blocks_to_home (dw : DogWalk) : ℕ :=
  (dw.total_blocks_per_day / dw.trips_per_day) - (dw.blocks_to_park + dw.blocks_to_school)

/-- Theorem stating that Ray walks 11 blocks to get back home -/
theorem ray_walks_11_blocks_home :
  ∃ (dw : DogWalk),
    dw.trips_per_day = 3 ∧
    dw.total_blocks_per_day = 66 ∧
    dw.blocks_to_park = 4 ∧
    dw.blocks_to_school = 7 ∧
    blocks_to_home dw = 11 := by
  sorry

end NUMINAMATH_CALUDE_ray_walks_11_blocks_home_l2057_205769


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2057_205736

theorem sin_cos_identity : 
  Real.sin (110 * π / 180) * Real.cos (40 * π / 180) - 
  Real.cos (70 * π / 180) * Real.sin (40 * π / 180) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2057_205736


namespace NUMINAMATH_CALUDE_parallel_to_y_axis_l2057_205734

/-- Given two points P and Q in a Cartesian coordinate system,
    where P has coordinates (m, 3) and Q has coordinates (2-2m, m-3),
    and PQ is parallel to the y-axis, prove that m = 2/3. -/
theorem parallel_to_y_axis (m : ℚ) : 
  let P : ℚ × ℚ := (m, 3)
  let Q : ℚ × ℚ := (2 - 2*m, m - 3)
  (P.1 = Q.1) → m = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_to_y_axis_l2057_205734


namespace NUMINAMATH_CALUDE_parabola_translation_l2057_205731

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation amount
def translation_amount : ℝ := 3

-- Define the translated parabola function
def translated_parabola (x : ℝ) : ℝ := original_parabola x + translation_amount

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = -2 * x^2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2057_205731


namespace NUMINAMATH_CALUDE_trains_meeting_problem_l2057_205772

/-- Theorem: Two trains meeting problem
    Given two trains starting 450 miles apart and traveling towards each other
    at 50 miles per hour each, the distance traveled by one train when they meet
    is 225 miles. -/
theorem trains_meeting_problem (distance_between_stations : ℝ) 
                                (speed_train_a : ℝ) 
                                (speed_train_b : ℝ) : ℝ :=
  by
  have h1 : distance_between_stations = 450 := by sorry
  have h2 : speed_train_a = 50 := by sorry
  have h3 : speed_train_b = 50 := by sorry
  
  -- Calculate the combined speed of the trains
  let combined_speed := speed_train_a + speed_train_b
  
  -- Calculate the time until the trains meet
  let time_to_meet := distance_between_stations / combined_speed
  
  -- Calculate the distance traveled by Train A
  let distance_traveled_by_a := speed_train_a * time_to_meet
  
  -- Prove that the distance traveled by Train A is 225 miles
  have h4 : distance_traveled_by_a = 225 := by sorry
  
  exact distance_traveled_by_a


end NUMINAMATH_CALUDE_trains_meeting_problem_l2057_205772


namespace NUMINAMATH_CALUDE_existence_of_large_n_with_same_digit_occurrences_l2057_205763

open Nat

-- Define a function to check if two numbers have the same digit occurrences
def sameDigitOccurrences (a b : ℕ) : Prop := sorry

-- Define the theorem
theorem existence_of_large_n_with_same_digit_occurrences :
  ∃ n : ℕ, n > 10^100 ∧
    sameDigitOccurrences (n^2) ((n+1)^2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_large_n_with_same_digit_occurrences_l2057_205763


namespace NUMINAMATH_CALUDE_remainder_theorem_application_l2057_205777

theorem remainder_theorem_application (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D * x^4 + E * x^2 + F * x - 2
  (q 2 = 10) → (q (-2) = -2) := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_application_l2057_205777


namespace NUMINAMATH_CALUDE_min_pumps_needed_l2057_205710

/-- Represents the rate at which water flows into the well (units per minute) -/
def M : ℝ := sorry

/-- Represents the rate at which one pump removes water (units per minute) -/
def A : ℝ := sorry

/-- Represents the initial amount of water in the well (units) -/
def W : ℝ := sorry

/-- The time it takes to empty the well with 4 pumps (minutes) -/
def time_4_pumps : ℝ := 40

/-- The time it takes to empty the well with 5 pumps (minutes) -/
def time_5_pumps : ℝ := 30

/-- The target time to empty the well (minutes) -/
def target_time : ℝ := 24

/-- Condition: 4 pumps take 40 minutes to empty the well -/
axiom condition_4_pumps : 4 * A * time_4_pumps = W + M * time_4_pumps

/-- Condition: 5 pumps take 30 minutes to empty the well -/
axiom condition_5_pumps : 5 * A * time_5_pumps = W + M * time_5_pumps

/-- Theorem: The minimum number of pumps needed to empty the well in 24 minutes is 6 -/
theorem min_pumps_needed : ∃ (n : ℕ), n * A * target_time = W + M * target_time ∧ n = 6 :=
  sorry

end NUMINAMATH_CALUDE_min_pumps_needed_l2057_205710
