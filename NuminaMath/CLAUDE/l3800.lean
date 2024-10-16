import Mathlib

namespace NUMINAMATH_CALUDE_servant_served_nine_months_l3800_380076

/-- Represents the employment contract and service details of a servant -/
structure ServantContract where
  fullYearSalary : ℕ  -- Salary for a full year in rupees
  uniformPrice : ℕ    -- Price of the uniform in rupees
  receivedSalary : ℕ  -- Salary actually received in rupees
  fullYearMonths : ℕ  -- Number of months in a full year

/-- Calculates the number of months served by the servant -/
def monthsServed (contract : ServantContract) : ℕ :=
  (contract.receivedSalary + contract.uniformPrice) * contract.fullYearMonths 
    / (contract.fullYearSalary + contract.uniformPrice)

/-- Theorem stating that the servant served for 9 months -/
theorem servant_served_nine_months :
  let contract : ServantContract := {
    fullYearSalary := 500,
    uniformPrice := 500,
    receivedSalary := 250,
    fullYearMonths := 12
  }
  monthsServed contract = 9 := by sorry

end NUMINAMATH_CALUDE_servant_served_nine_months_l3800_380076


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3800_380055

theorem quadratic_solution_sum (x y : ℝ) : 
  x + y = 5 → 2 * x * y = 5 → 
  ∃ (a b c d : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d) ∧
    a + b + c + d = 23 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3800_380055


namespace NUMINAMATH_CALUDE_painting_price_change_l3800_380044

theorem painting_price_change (P : ℝ) (x : ℝ) 
  (h1 : P > 0) 
  (h2 : (1.1 * P) * (1 - x / 100) = 0.935 * P) : 
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_change_l3800_380044


namespace NUMINAMATH_CALUDE_soccer_tournament_properties_l3800_380048

/-- Represents a round-robin soccer tournament. -/
structure SoccerTournament where
  num_teams : Nat
  top_scores : Fin 4 → Nat
  fifth_score_different : top_scores 3 ≠ (top_scores 3).succ

/-- The total number of matches in a round-robin tournament. -/
def total_matches (k : Nat) : Nat :=
  k * (k - 1) / 2

/-- The total points scored in all matches. -/
def total_points (k : Nat) : Nat :=
  k * (k - 1)

/-- Theorem stating the properties of the soccer tournament. -/
theorem soccer_tournament_properties (t : SoccerTournament) :
  t.num_teams = 7 ∧
  total_points t.num_teams = 42 ∧
  t.top_scores 0 = 11 ∧
  t.top_scores 1 = 9 ∧
  t.top_scores 2 = 7 ∧
  t.top_scores 3 = 5 :=
by sorry

end NUMINAMATH_CALUDE_soccer_tournament_properties_l3800_380048


namespace NUMINAMATH_CALUDE_y_derivative_f_monotonicity_l3800_380016

-- Part 1
noncomputable def y (x : ℝ) : ℝ := (2 * x^2 - 3) * Real.sqrt (1 + x^2)

theorem y_derivative (x : ℝ) :
  deriv y x = 4 * x * Real.sqrt (1 + x^2) + (2 * x^3 - 3 * x) / Real.sqrt (1 + x^2) :=
sorry

-- Part 2
noncomputable def f (x : ℝ) : ℝ := (x * Real.log x)⁻¹

theorem f_monotonicity (x : ℝ) (hx : x > 0 ∧ x ≠ 1) :
  (StrictMonoOn f (Set.Ioo 0 (Real.exp (-1)))) ∧
  (StrictAntiOn f (Set.Ioi (Real.exp (-1)))) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_f_monotonicity_l3800_380016


namespace NUMINAMATH_CALUDE_mode_of_data_set_l3800_380089

def data_set : List ℕ := [5, 4, 4, 3, 6, 2]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_data_set :
  mode data_set = 4 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_data_set_l3800_380089


namespace NUMINAMATH_CALUDE_class_average_increase_l3800_380004

/-- Proves that adding a 50-year-old student to a class of 19 students with an average age of 10 years
    increases the overall average age by 2 years. -/
theorem class_average_increase (n : ℕ) (original_avg : ℝ) (new_student_age : ℝ) :
  n = 19 →
  original_avg = 10 →
  new_student_age = 50 →
  (n * original_avg + new_student_age) / (n + 1) - original_avg = 2 := by
  sorry

#check class_average_increase

end NUMINAMATH_CALUDE_class_average_increase_l3800_380004


namespace NUMINAMATH_CALUDE_sum_floor_equality_l3800_380052

theorem sum_floor_equality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + b^2 = 2008) (h2 : c^2 + d^2 = 2008) (h3 : a * c = 1000) (h4 : b * d = 1000) :
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end NUMINAMATH_CALUDE_sum_floor_equality_l3800_380052


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3800_380073

theorem quadratic_roots_properties (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (h_roots : ∃ (r₁ r₂ : ℕ), r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₁^2 - p*r₁ + q = 0 ∧ r₂^2 - p*r₂ + q = 0) :
  (∃ (r₁ r₂ : ℕ), r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₁^2 - p*r₁ + q = 0 ∧ r₂^2 - p*r₂ + q = 0 ∧ 
    (∃ (k : ℕ), r₁ - r₂ = 2*k + 1 ∨ r₂ - r₁ = 2*k + 1)) ∧ 
  (∃ (r : ℕ), (r^2 - p*r + q = 0) ∧ Prime r) ∧
  Prime (p^2 - q) ∧
  Prime (p + q) := by
  sorry

#check quadratic_roots_properties

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3800_380073


namespace NUMINAMATH_CALUDE_furniture_by_design_salary_l3800_380081

/-- The monthly salary from Furniture by Design -/
def monthly_salary : ℝ := 1800

/-- The base salary -/
def base_salary : ℝ := 1600

/-- The commission rate -/
def commission_rate : ℝ := 0.04

/-- The sales amount at which both compensation options are equal -/
def equal_sales : ℝ := 5000

theorem furniture_by_design_salary :
  monthly_salary = base_salary + commission_rate * equal_sales := by
  sorry

end NUMINAMATH_CALUDE_furniture_by_design_salary_l3800_380081


namespace NUMINAMATH_CALUDE_modular_home_cost_l3800_380034

-- Define the parameters of the modular home
def total_area : ℝ := 3500
def kitchen_area : ℝ := 500
def kitchen_cost : ℝ := 35000
def bathroom_area : ℝ := 250
def bathroom_cost : ℝ := 15000
def bedroom_area : ℝ := 350
def bedroom_cost : ℝ := 21000
def living_area : ℝ := 600
def living_area_cost_per_sqft : ℝ := 100
def upgraded_cost_per_sqft : ℝ := 150

def num_kitchens : ℕ := 1
def num_bathrooms : ℕ := 3
def num_bedrooms : ℕ := 4
def num_living_areas : ℕ := 1

-- Define the theorem
theorem modular_home_cost :
  let total_module_area := kitchen_area * num_kitchens + bathroom_area * num_bathrooms +
                           bedroom_area * num_bedrooms + living_area * num_living_areas
  let remaining_area := total_area - total_module_area
  let upgraded_area := remaining_area / 2
  let total_cost := kitchen_cost * num_kitchens + bathroom_cost * num_bathrooms +
                    bedroom_cost * num_bedrooms + living_area * living_area_cost_per_sqft +
                    upgraded_area * upgraded_cost_per_sqft * 2
  total_cost = 261500 := by sorry

end NUMINAMATH_CALUDE_modular_home_cost_l3800_380034


namespace NUMINAMATH_CALUDE_count_problems_requiring_selection_l3800_380030

-- Define a structure to represent a problem
structure Problem where
  id : Nat
  requires_selection : Bool

-- Define our set of problems
def problems : List Problem := [
  { id := 1, requires_selection := true },  -- absolute value
  { id := 2, requires_selection := false }, -- square perimeter
  { id := 3, requires_selection := true },  -- maximum of three numbers
  { id := 4, requires_selection := true }   -- function value
]

-- Theorem statement
theorem count_problems_requiring_selection :
  (problems.filter Problem.requires_selection).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_problems_requiring_selection_l3800_380030


namespace NUMINAMATH_CALUDE_marbles_problem_l3800_380013

theorem marbles_problem (total : ℕ) (marc_initial : ℕ) (jon_initial : ℕ) (bag : ℕ) : 
  total = 66 →
  marc_initial = 2 * jon_initial →
  marc_initial + jon_initial = total →
  jon_initial + bag = 3 * marc_initial →
  bag = 110 := by
sorry

end NUMINAMATH_CALUDE_marbles_problem_l3800_380013


namespace NUMINAMATH_CALUDE_square_inequality_for_negatives_l3800_380086

theorem square_inequality_for_negatives (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_for_negatives_l3800_380086


namespace NUMINAMATH_CALUDE_twelve_students_pairs_l3800_380061

/-- The number of unique pairs in a group of n elements -/
def uniquePairs (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of unique pairs in a group of 12 students is 66 -/
theorem twelve_students_pairs : uniquePairs 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_twelve_students_pairs_l3800_380061


namespace NUMINAMATH_CALUDE_power_division_equality_l3800_380051

theorem power_division_equality (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l3800_380051


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3800_380028

/-- The total cost of buying mineral water and yogurt -/
def total_cost (m n : ℕ) : ℚ :=
  2.5 * m + 4 * n

/-- Theorem stating the total cost calculation -/
theorem total_cost_calculation (m n : ℕ) :
  total_cost m n = 2.5 * m + 4 * n := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3800_380028


namespace NUMINAMATH_CALUDE_number_of_girls_l3800_380007

-- Define the total number of polished nails
def total_nails : ℕ := 40

-- Define the number of nails per girl
def nails_per_girl : ℕ := 20

-- Theorem to prove the number of girls
theorem number_of_girls : total_nails / nails_per_girl = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_l3800_380007


namespace NUMINAMATH_CALUDE_parallel_vectors_l3800_380091

/-- Two vectors in R² are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- Given vectors a and b, if they are parallel, then the x-component of a is 1/2 -/
theorem parallel_vectors (a b : ℝ × ℝ) (h : a.2 = 1 ∧ b = (2, 4)) :
  parallel a b → a.1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l3800_380091


namespace NUMINAMATH_CALUDE_science_to_novel_ratio_l3800_380031

/-- Given the page counts of different books, prove the ratio of science to novel pages --/
theorem science_to_novel_ratio :
  let history_pages : ℕ := 300
  let science_pages : ℕ := 600
  let novel_pages : ℕ := history_pages / 2
  science_pages / novel_pages = 4 := by
  sorry


end NUMINAMATH_CALUDE_science_to_novel_ratio_l3800_380031


namespace NUMINAMATH_CALUDE_sequence_general_term_l3800_380032

/-- Given a sequence {a_n} where the sum of the first n terms S_n = n^2 + 3n,
    prove that the general term a_n = 2n + 2 for all n ∈ ℕ*. -/
theorem sequence_general_term (n : ℕ+) (S : ℕ+ → ℕ) (a : ℕ+ → ℕ) 
    (h_S : ∀ k : ℕ+, S k = k^2 + 3*k) :
  a n = 2*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3800_380032


namespace NUMINAMATH_CALUDE_equation_solution_l3800_380001

theorem equation_solution :
  ∃ x : ℝ, (5 + 3.4 * x = 2.8 * x - 35) ∧ (x = -200/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3800_380001


namespace NUMINAMATH_CALUDE_floor_of_4_7_l3800_380056

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l3800_380056


namespace NUMINAMATH_CALUDE_inequality_coverage_l3800_380094

theorem inequality_coverage (a : ℝ) : 
  (∀ x : ℝ, (2 * a - x > 1 ∧ 2 * x + 5 > 3 * a) → (1 ≤ x ∧ x ≤ 6)) →
  (7/3 ≤ a ∧ a ≤ 7/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_coverage_l3800_380094


namespace NUMINAMATH_CALUDE_total_vehicles_l3800_380077

-- Define the number of trucks
def num_trucks : ℕ := 20

-- Define the number of tanks as a function of the number of trucks
def num_tanks : ℕ := 5 * num_trucks

-- Theorem to prove
theorem total_vehicles : num_tanks + num_trucks = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_l3800_380077


namespace NUMINAMATH_CALUDE_train_overtake_time_l3800_380023

/-- The time it takes for a train to overtake a motorbike -/
theorem train_overtake_time (train_speed motorbike_speed train_length : ℝ) 
  (h1 : train_speed = 100)
  (h2 : motorbike_speed = 64)
  (h3 : train_length = 850.068) : 
  (train_length / ((train_speed - motorbike_speed) * (1000 / 3600))) = 85.0068 := by
  sorry

end NUMINAMATH_CALUDE_train_overtake_time_l3800_380023


namespace NUMINAMATH_CALUDE_removed_triangles_area_l3800_380041

/-- Given a square with side length s, from which isosceles right triangles
    with equal sides of length x are removed from each corner to form a rectangle
    with longer side 16 units, the total area of the four removed triangles is 512 square units. -/
theorem removed_triangles_area (s x : ℝ) : 
  s > 0 ∧ x > 0 ∧ s - x = 16 ∧ 2 * x^2 = (s - 2*x)^2 → 4 * (1/2 * x^2) = 512 := by
  sorry

#check removed_triangles_area

end NUMINAMATH_CALUDE_removed_triangles_area_l3800_380041


namespace NUMINAMATH_CALUDE_area_circle_radius_5_l3800_380071

/-- The area of a circle with radius 5 meters is 25π square meters. -/
theorem area_circle_radius_5 : 
  ∀ (π : ℝ), π > 0 → (5 : ℝ) ^ 2 * π = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_area_circle_radius_5_l3800_380071


namespace NUMINAMATH_CALUDE_owl_wings_area_l3800_380083

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  bottomLeft : Point
  topRight : Point

/-- Calculates the area of a triangle given three points using the shoelace formula -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p1.x * p2.y + p2.x * p3.y + p3.x * p1.y) - (p1.y * p2.x + p2.y * p3.x + p3.y * p1.x))

/-- Theorem: The area of the shaded region in the specified rectangle is 4 -/
theorem owl_wings_area (rect : Rectangle) 
    (h1 : rect.topRight.x - rect.bottomLeft.x = 4) 
    (h2 : rect.topRight.y - rect.bottomLeft.y = 5) 
    (h3 : rect.topRight.x - rect.bottomLeft.x = rect.topRight.y - rect.bottomLeft.y - 1) :
    ∃ (p1 p2 p3 : Point), triangleArea p1 p2 p3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_owl_wings_area_l3800_380083


namespace NUMINAMATH_CALUDE_complex_power_problem_l3800_380038

theorem complex_power_problem (z : ℂ) (h : z = (1 + Complex.I)^2 / 2) : z^2023 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l3800_380038


namespace NUMINAMATH_CALUDE_sine_cosine_sum_l3800_380058

theorem sine_cosine_sum (α : Real) : 
  (∃ (x y : Real), x = 3/5 ∧ y = 4/5 ∧ x^2 + y^2 = 1 ∧ 
    Real.cos α = x ∧ Real.sin α = y) → 
  Real.sin α + 2 * Real.cos α = 2 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_l3800_380058


namespace NUMINAMATH_CALUDE_exponential_function_property_l3800_380067

theorem exponential_function_property (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x - 2
  f 0 = -1 := by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l3800_380067


namespace NUMINAMATH_CALUDE_sum_in_base7_l3800_380036

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a base 10 number to base 7 --/
def base10ToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The statement to prove --/
theorem sum_in_base7 :
  let a := [2, 1]  -- 12 in base 7
  let b := [5, 4, 2]  -- 245 in base 7
  let sum := base7ToBase10 a + base7ToBase10 b
  base10ToBase7 sum = [0, 6, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base7_l3800_380036


namespace NUMINAMATH_CALUDE_polygon_sides_l3800_380084

/-- A polygon has n sides if its interior angles sum is 4 times its exterior angles sum -/
theorem polygon_sides (n : ℕ) : n = 10 :=
  by
  -- Define the sum of interior angles
  let interior_sum := (n - 2) * 180
  -- Define the sum of exterior angles
  let exterior_sum := 360
  -- State the condition that interior sum is 4 times exterior sum
  have h : interior_sum = 4 * exterior_sum := by sorry
  -- Prove that n = 10
  sorry


end NUMINAMATH_CALUDE_polygon_sides_l3800_380084


namespace NUMINAMATH_CALUDE_sixth_point_equals_initial_l3800_380025

/-- Triangle in a plane --/
structure Triangle where
  A₀ : ℝ × ℝ
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ

/-- Symmetric point with respect to a given point --/
def symmetric_point (P : ℝ × ℝ) (A : ℝ × ℝ) : ℝ × ℝ :=
  (2 * A.1 - P.1, 2 * A.2 - P.2)

/-- Generate the next point in the sequence --/
def next_point (P : ℝ × ℝ) (i : ℕ) (T : Triangle) : ℝ × ℝ :=
  match i % 3 with
  | 0 => symmetric_point P T.A₀
  | 1 => symmetric_point P T.A₁
  | _ => symmetric_point P T.A₂

/-- Generate the i-th point in the sequence --/
def P (i : ℕ) (P₀ : ℝ × ℝ) (T : Triangle) : ℝ × ℝ :=
  match i with
  | 0 => P₀
  | n + 1 => next_point (P n P₀ T) (n + 1) T

theorem sixth_point_equals_initial (P₀ : ℝ × ℝ) (T : Triangle) :
  P 6 P₀ T = P₀ := by
  sorry

end NUMINAMATH_CALUDE_sixth_point_equals_initial_l3800_380025


namespace NUMINAMATH_CALUDE_cesar_watched_fraction_l3800_380065

theorem cesar_watched_fraction (total_seasons : ℕ) (episodes_per_season : ℕ) (remaining_episodes : ℕ) :
  total_seasons = 12 →
  episodes_per_season = 20 →
  remaining_episodes = 160 →
  (total_seasons * episodes_per_season - remaining_episodes) / (total_seasons * episodes_per_season) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cesar_watched_fraction_l3800_380065


namespace NUMINAMATH_CALUDE_rectangle_width_range_l3800_380002

/-- Given a wire of length 20 cm shaped into a rectangle with length at least 6 cm,
    prove that the width x satisfies 0 < x ≤ 20/3 -/
theorem rectangle_width_range :
  ∀ x : ℝ,
  (∃ l : ℝ, l ≥ 6 ∧ 2 * (x + l) = 20) →
  (0 < x ∧ x ≤ 20 / 3) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_range_l3800_380002


namespace NUMINAMATH_CALUDE_fabric_delivery_problem_l3800_380082

/-- Represents the fabric delivery problem for Daniel's textile company -/
theorem fabric_delivery_problem (monday_delivery : ℝ) : 
  monday_delivery * 2 * 3.5 = 140 → monday_delivery = 20 := by
  sorry

#check fabric_delivery_problem

end NUMINAMATH_CALUDE_fabric_delivery_problem_l3800_380082


namespace NUMINAMATH_CALUDE_four_balls_three_boxes_l3800_380078

/-- The number of ways to put n different balls into k boxes -/
def ways_to_put_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: Putting 4 different balls into 3 boxes results in 81 different ways -/
theorem four_balls_three_boxes : ways_to_put_balls 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_four_balls_three_boxes_l3800_380078


namespace NUMINAMATH_CALUDE_prob_at_least_one_head_correct_l3800_380098

/-- The probability of getting at least one head when tossing 3 coins simultaneously -/
def prob_at_least_one_head : ℚ :=
  7/8

/-- The number of coins being tossed simultaneously -/
def num_coins : ℕ := 3

/-- The probability of getting heads on a single coin toss -/
def prob_heads : ℚ := 1/2

theorem prob_at_least_one_head_correct :
  prob_at_least_one_head = 1 - (1 - prob_heads) ^ num_coins :=
by sorry


end NUMINAMATH_CALUDE_prob_at_least_one_head_correct_l3800_380098


namespace NUMINAMATH_CALUDE_fraction_simplification_l3800_380011

theorem fraction_simplification :
  (1 / 5 + 1 / 3) / (3 / 4 - 1 / 8) = 64 / 75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3800_380011


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3800_380072

theorem quadratic_equation_roots (m : ℤ) : 
  (∃ x y : ℤ, x > 0 ∧ y > 0 ∧ 
   m * x^2 + (3 - m) * x - 3 = 0 ∧
   m * y^2 + (3 - m) * y - 3 = 0 ∧
   x ≠ y) →
  m = -1 ∨ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3800_380072


namespace NUMINAMATH_CALUDE_average_pastry_sales_l3800_380009

def pastry_sales : List Nat := [2, 3, 4, 5, 6, 7, 8]

theorem average_pastry_sales : 
  (List.sum pastry_sales) / pastry_sales.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_pastry_sales_l3800_380009


namespace NUMINAMATH_CALUDE_floor_tiles_1517_902_l3800_380043

/-- The least number of square tiles required to pave a rectangular floor -/
def leastSquareTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  ((length + tileSize - 1) / tileSize) * ((width + tileSize - 1) / tileSize)

/-- Proof that 814 square tiles are required for a 1517 cm x 902 cm floor -/
theorem floor_tiles_1517_902 :
  leastSquareTiles 1517 902 = 814 := by
  sorry

end NUMINAMATH_CALUDE_floor_tiles_1517_902_l3800_380043


namespace NUMINAMATH_CALUDE_div_point_one_eq_mul_ten_l3800_380070

theorem div_point_one_eq_mul_ten (a : ℝ) : a / 0.1 = a * 10 := by sorry

end NUMINAMATH_CALUDE_div_point_one_eq_mul_ten_l3800_380070


namespace NUMINAMATH_CALUDE_museum_artifacts_per_wing_l3800_380050

/-- Represents a museum with paintings and artifacts -/
structure Museum where
  total_wings : ℕ
  painting_wings : ℕ
  large_paintings : ℕ
  small_painting_wings : ℕ
  small_paintings_per_wing : ℕ
  artifact_multiplier : ℕ

/-- Calculates the number of artifacts in each artifact wing -/
def artifacts_per_wing (m : Museum) : ℕ :=
  let total_paintings := m.large_paintings + m.small_painting_wings * m.small_paintings_per_wing
  let total_artifacts := total_paintings * m.artifact_multiplier
  let artifact_wings := m.total_wings - m.painting_wings
  (total_artifacts + artifact_wings - 1) / artifact_wings

/-- Theorem stating the number of artifacts in each artifact wing for the given museum -/
theorem museum_artifacts_per_wing :
  let m : Museum := {
    total_wings := 16,
    painting_wings := 6,
    large_paintings := 2,
    small_painting_wings := 4,
    small_paintings_per_wing := 20,
    artifact_multiplier := 8
  }
  artifacts_per_wing m = 66 := by sorry

end NUMINAMATH_CALUDE_museum_artifacts_per_wing_l3800_380050


namespace NUMINAMATH_CALUDE_reflected_ray_slope_l3800_380075

theorem reflected_ray_slope (emissionPoint : ℝ × ℝ) (circleCenter : ℝ × ℝ) (circleRadius : ℝ) :
  emissionPoint = (-2, -3) →
  circleCenter = (-3, 2) →
  circleRadius = 1 →
  ∃ k : ℝ, (k = -4/3 ∨ k = -3/4) ∧
    (∀ x y : ℝ, y + 3 = k * (x - 2) →
      ((x + 3)^2 + (y - 2)^2 = 1 →
        abs (-3*k - 2 - 2*k - 3) / Real.sqrt (k^2 + 1) = 1)) :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_slope_l3800_380075


namespace NUMINAMATH_CALUDE_sequence_problem_l3800_380074

/-- Given a sequence a and a geometric sequence b, prove that a_2016 = 1 -/
theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  a 1 = 1 →
  (∀ n, b n = a (n + 1) / a n) →
  (∀ n, b (n + 1) / b n = b 2 / b 1) →
  b 1008 = 1 →
  a 2016 = 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3800_380074


namespace NUMINAMATH_CALUDE_min_overlap_percentage_l3800_380080

theorem min_overlap_percentage (math_pref science_pref : ℝ) 
  (h1 : math_pref = 0.90)
  (h2 : science_pref = 0.85) :
  let overlap := math_pref + science_pref - 1
  overlap ≥ 0.75 ∧ 
  ∀ x, x ≥ 0 ∧ x < overlap → 
    ∃ total_pref, total_pref ≤ 1 ∧ 
      total_pref = math_pref + science_pref - x :=
by sorry

end NUMINAMATH_CALUDE_min_overlap_percentage_l3800_380080


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3800_380021

theorem exponent_multiplication (a : ℝ) (m n : ℕ) : a ^ m * a ^ n = a ^ (m + n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3800_380021


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3800_380059

/-- An arithmetic sequence with a_6 = 1 -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ a1 d : ℚ, ∀ n : ℕ, a n = a1 + (n - 1) * d) ∧ a 6 = 1

/-- For any arithmetic sequence with a_6 = 1, a_2 + a_10 = 2 -/
theorem arithmetic_sequence_sum (a : ℕ → ℚ) (h : ArithmeticSequence a) :
  a 2 + a 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3800_380059


namespace NUMINAMATH_CALUDE_solution_system_equations_l3800_380095

theorem solution_system_equations (x y z : ℝ) : 
  (x^2 - y^2 + z = 27 / (x * y) ∧
   y^2 - z^2 + x = 27 / (y * z) ∧
   z^2 - x^2 + y = 27 / (x * z)) →
  ((x = 3 ∧ y = 3 ∧ z = 3) ∨
   (x = -3 ∧ y = -3 ∧ z = 3) ∨
   (x = -3 ∧ y = 3 ∧ z = -3) ∨
   (x = 3 ∧ y = -3 ∧ z = -3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l3800_380095


namespace NUMINAMATH_CALUDE_simplify_fraction_l3800_380042

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (75 * b^3) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3800_380042


namespace NUMINAMATH_CALUDE_tangent_lines_theorem_l3800_380090

noncomputable def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

def tangent_line_at_2 (x y : ℝ) : Prop := y = x - 4

def tangent_lines_through_A (x y : ℝ) : Prop := y = x - 4 ∨ y = -2

theorem tangent_lines_theorem :
  (∀ x y : ℝ, y = f x → tangent_line_at_2 x y ↔ x = 2) ∧
  (∀ x y : ℝ, y = f x → tangent_lines_through_A x y ↔ (x = 2 ∧ y = -2) ∨ (x = 1 ∧ y = -2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_theorem_l3800_380090


namespace NUMINAMATH_CALUDE_henrys_initial_games_henrys_initial_games_is_58_l3800_380005

/-- Proves the number of games Henry had at first -/
theorem henrys_initial_games : ℕ → Prop := fun h =>
  let neil_initial := 7
  let henry_to_neil := 6
  let neil_final := neil_initial + henry_to_neil
  let henry_final := h - henry_to_neil
  (henry_final = 4 * neil_final) → h = 58

/-- The theorem holds for 58 -/
theorem henrys_initial_games_is_58 : henrys_initial_games 58 := by
  sorry

end NUMINAMATH_CALUDE_henrys_initial_games_henrys_initial_games_is_58_l3800_380005


namespace NUMINAMATH_CALUDE_triangle_property_l3800_380027

theorem triangle_property (A B C : ℝ) (hABC : A + B + C = π) 
  (hDot : (Real.cos A * Real.cos C + Real.sin A * Real.sin C) * 
          (Real.cos A * Real.cos B + Real.sin A * Real.sin B) = 
          3 * (Real.cos B * Real.cos A + Real.sin B * Real.sin A) * 
             (Real.cos B * Real.cos C + Real.sin B * Real.sin C)) :
  (Real.tan B = 3 * Real.tan A) ∧ 
  (Real.cos C = Real.sqrt 5 / 5 → A = π / 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3800_380027


namespace NUMINAMATH_CALUDE_square_area_reduction_l3800_380060

theorem square_area_reduction (S1_area : ℝ) (S1_area_eq : S1_area = 25) : 
  let S1_side := Real.sqrt S1_area
  let S2_side := S1_side / Real.sqrt 2
  let S3_side := S2_side / Real.sqrt 2
  S3_side ^ 2 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_reduction_l3800_380060


namespace NUMINAMATH_CALUDE_even_function_domain_symmetric_l3800_380015

/-- An even function from ℝ to ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The domain of the function -/
def Domain (f : ℝ → ℝ) (t : ℝ) : Set ℝ :=
  {x | t - 4 ≤ x ∧ x ≤ t}

/-- Theorem: For an even function with domain [t-4, t], t = 2 -/
theorem even_function_domain_symmetric (f : ℝ → ℝ) (t : ℝ) 
    (h1 : EvenFunction f) (h2 : Domain f t = Set.Icc (t - 4) t) : t = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_domain_symmetric_l3800_380015


namespace NUMINAMATH_CALUDE_fifth_root_of_161051_l3800_380053

theorem fifth_root_of_161051 : ∃ n : ℕ, n^5 = 161051 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_161051_l3800_380053


namespace NUMINAMATH_CALUDE_example_polygon_area_l3800_380047

/-- A polygon on a unit grid with specified vertices -/
structure GridPolygon where
  vertices : List (Int × Int)

/-- Calculate the area of a GridPolygon -/
def area (p : GridPolygon) : ℕ :=
  sorry

/-- The specific polygon from the problem -/
def examplePolygon : GridPolygon :=
  { vertices := [(0,0), (20,0), (20,20), (10,20), (10,10), (0,10)] }

/-- Theorem stating that the area of the example polygon is 250 square units -/
theorem example_polygon_area : area examplePolygon = 250 :=
  sorry

end NUMINAMATH_CALUDE_example_polygon_area_l3800_380047


namespace NUMINAMATH_CALUDE_female_democrats_count_l3800_380068

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 660 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = (total / 3 : ℚ) →
  female / 2 = 110 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l3800_380068


namespace NUMINAMATH_CALUDE_xyz_fraction_l3800_380000

theorem xyz_fraction (x y z : ℝ) 
  (h1 : x * y / (x + y) = 1 / 3)
  (h2 : y * z / (y + z) = 1 / 5)
  (h3 : z * x / (z + x) = 1 / 6) :
  x * y * z / (x * y + y * z + z * x) = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_fraction_l3800_380000


namespace NUMINAMATH_CALUDE_num_divisors_not_div_by_5_eq_4_l3800_380040

/-- The number of positive divisors of 150 that are not divisible by 5 -/
def num_divisors_not_div_by_5 : ℕ :=
  (Finset.filter (fun d => d ∣ 150 ∧ ¬(5 ∣ d)) (Finset.range 151)).card

/-- 150 has the prime factorization 2 * 3 * 5^2 -/
axiom prime_factorization : 150 = 2 * 3 * 5^2

theorem num_divisors_not_div_by_5_eq_4 : num_divisors_not_div_by_5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_not_div_by_5_eq_4_l3800_380040


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3800_380085

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 3/8) 
  (h2 : x - y = 1/4) : 
  x^2 - y^2 = 3/32 := by sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3800_380085


namespace NUMINAMATH_CALUDE_function_inequality_l3800_380054

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) (h2 : ∀ x, deriv f x > 1) :
  f 3 > f 1 + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3800_380054


namespace NUMINAMATH_CALUDE_horner_method_for_f_l3800_380020

/-- Horner's method representation of a polynomial -/
def horner_rep (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

/-- The original polynomial function -/
def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

theorem horner_method_for_f :
  f 5 = horner_rep [1, 2, 1, -1, 3, -5] 5 ∧ 
  horner_rep [1, 2, 1, -1, 3, -5] 5 = 4485 :=
sorry

end NUMINAMATH_CALUDE_horner_method_for_f_l3800_380020


namespace NUMINAMATH_CALUDE_correct_average_marks_l3800_380039

theorem correct_average_marks (n : ℕ) (incorrect_avg : ℝ) (wrong_mark correct_mark : ℝ) :
  n = 10 ∧ incorrect_avg = 100 ∧ wrong_mark = 60 ∧ correct_mark = 10 →
  (n * incorrect_avg - (wrong_mark - correct_mark)) / n = 95 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l3800_380039


namespace NUMINAMATH_CALUDE_special_function_at_zero_l3800_380019

/-- A function satisfying f(x + y) = f(x) + f(y) - xy for all real x and y, with f(1) = 1 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y - x * y) ∧ (f 1 = 1)

/-- Theorem: For a special function f, f(0) = 0 -/
theorem special_function_at_zero {f : ℝ → ℝ} (hf : special_function f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_zero_l3800_380019


namespace NUMINAMATH_CALUDE_smallest_1755_more_than_sum_of_digits_l3800_380022

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Property that a number is 1755 more than the sum of its digits -/
def is1755MoreThanSumOfDigits (n : ℕ) : Prop :=
  n = sumOfDigits n + 1755

/-- Theorem stating that 1770 is the smallest natural number that is 1755 more than the sum of its digits -/
theorem smallest_1755_more_than_sum_of_digits :
  (1770 = sumOfDigits 1770 + 1755) ∧
  ∀ m : ℕ, m < 1770 → m ≠ sumOfDigits m + 1755 :=
by sorry

end NUMINAMATH_CALUDE_smallest_1755_more_than_sum_of_digits_l3800_380022


namespace NUMINAMATH_CALUDE_problem_17_l3800_380092

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

def p (a : ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → log_a a (x + 3) > log_a a (y + 3)

def q (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 + (2*a - 3)*x₁ + 1 = 0 ∧ x₂^2 + (2*a - 3)*x₂ + 1 = 0

theorem problem_17 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (a ∈ Set.Icc (1/2) 1 ∪ Set.Ioi (5/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_17_l3800_380092


namespace NUMINAMATH_CALUDE_vector_magnitude_l3800_380099

/-- Given vectors a and b, if c is parallel to a and perpendicular to b + c, 
    then the magnitude of c is 3√2. -/
theorem vector_magnitude (a b c : ℝ × ℝ) : 
  a = (-1, 1) → 
  b = (-2, 4) → 
  (∃ k : ℝ, c = k • a) →  -- parallel condition
  (a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2) = 0) →  -- perpendicular condition
  ‖c‖ = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3800_380099


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3800_380003

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) → (a < 1 ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3800_380003


namespace NUMINAMATH_CALUDE_xiao_wang_exam_scores_xiao_wang_final_results_l3800_380012

theorem xiao_wang_exam_scores :
  ∀ (x y : ℝ),
  (x * y + 98) / (x + 1) = y + 1 →
  (x * y + 98 + 70) / (x + 2) = y - 1 →
  x = 8 ∧ y = 89 :=
by
  sorry

theorem xiao_wang_final_results (x y : ℝ) 
  (h : x = 8 ∧ y = 89) :
  (x + 2 : ℝ) = 10 ∧ (y - 1 : ℝ) = 88 :=
by
  sorry

end NUMINAMATH_CALUDE_xiao_wang_exam_scores_xiao_wang_final_results_l3800_380012


namespace NUMINAMATH_CALUDE_dice_probability_l3800_380018

/-- The number of possible outcomes for a single die roll -/
def die_outcomes : ℕ := 6

/-- The number of favorable outcomes for a single die roll (not equal to 2) -/
def favorable_outcomes : ℕ := 5

/-- The probability that (a-2)(b-2)(c-2) ≠ 0 when three standard dice are tossed -/
theorem dice_probability : 
  (favorable_outcomes ^ 3 : ℚ) / (die_outcomes ^ 3 : ℚ) = 125 / 216 := by
sorry

end NUMINAMATH_CALUDE_dice_probability_l3800_380018


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3800_380045

theorem max_sum_of_factors (x y : ℕ+) : 
  x.val * y.val = 48 → 
  4 ∣ x.val → 
  ∀ (a b : ℕ+), a.val * b.val = 48 → 4 ∣ a.val → a + b ≤ x + y → 
  x + y = 49 := by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3800_380045


namespace NUMINAMATH_CALUDE_vertical_line_distance_l3800_380049

/-- The distance between two points on a vertical line with y-coordinates differing by 2 is 2 -/
theorem vertical_line_distance (a : ℝ) : 
  abs ((2 - a) - (-a)) = 2 := by sorry

end NUMINAMATH_CALUDE_vertical_line_distance_l3800_380049


namespace NUMINAMATH_CALUDE_max_value_on_interval_solution_set_inequality_l3800_380097

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2) * |x - 2|

-- Theorem for the maximum value of f on [-3, 1]
theorem max_value_on_interval :
  ∃ (m : ℝ), m = 4 ∧ ∀ x ∈ Set.Icc (-3) 1, f x ≤ m :=
sorry

-- Theorem for the solution set of f(x) > 3x
theorem solution_set_inequality :
  {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4 ∨ (-4 < x ∧ x < 1)} :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_solution_set_inequality_l3800_380097


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_two_l3800_380046

-- Define the complex number (2+i)i
def z : ℂ := (2 + Complex.I) * Complex.I

-- Theorem statement
theorem imaginary_part_of_z_is_two : Complex.im z = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_two_l3800_380046


namespace NUMINAMATH_CALUDE_angle_around_point_l3800_380035

theorem angle_around_point (x : ℝ) : x = 110 :=
  let total_angle : ℝ := 360
  let given_angle : ℝ := 140
  have h1 : x + x + given_angle = total_angle := by sorry
  sorry

end NUMINAMATH_CALUDE_angle_around_point_l3800_380035


namespace NUMINAMATH_CALUDE_cheries_whistlers_l3800_380079

/-- Represents the number of boxes of fireworks --/
def koby_boxes : ℕ := 2

/-- Represents the number of boxes of fireworks --/
def cherie_boxes : ℕ := 1

/-- Represents the number of sparklers in each of Koby's boxes --/
def koby_sparklers_per_box : ℕ := 3

/-- Represents the number of whistlers in each of Koby's boxes --/
def koby_whistlers_per_box : ℕ := 5

/-- Represents the number of sparklers in Cherie's box --/
def cherie_sparklers : ℕ := 8

/-- Represents the total number of fireworks Koby and Cherie have --/
def total_fireworks : ℕ := 33

/-- Theorem stating that Cherie's box contains 9 whistlers --/
theorem cheries_whistlers :
  (koby_boxes * koby_sparklers_per_box + koby_boxes * koby_whistlers_per_box +
   cherie_sparklers + (total_fireworks - (koby_boxes * koby_sparklers_per_box +
   koby_boxes * koby_whistlers_per_box + cherie_sparklers))) = total_fireworks ∧
  (total_fireworks - (koby_boxes * koby_sparklers_per_box +
   koby_boxes * koby_whistlers_per_box + cherie_sparklers)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_cheries_whistlers_l3800_380079


namespace NUMINAMATH_CALUDE_correct_calculation_l3800_380063

theorem correct_calculation (x : ℝ) (h : x + 20 = 180) : x / 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3800_380063


namespace NUMINAMATH_CALUDE_intersection_A_B_l3800_380064

-- Define the sets A and B
def A : Set ℝ := {x | x ≠ 3 ∧ x ≥ 2}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 5}

-- Define the interval (3,5]
def interval_3_5 : Set ℝ := {x | 3 < x ∧ x ≤ 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = interval_3_5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3800_380064


namespace NUMINAMATH_CALUDE_subset_sum_partition_l3800_380057

theorem subset_sum_partition (n : ℕ) (S : Finset ℝ) (h_pos : ∀ x ∈ S, 0 < x) (h_card : S.card = n) :
  ∃ (P : Finset (Finset ℝ)), 
    P.card = n ∧ 
    (∀ X ∈ P, ∃ (min max : ℝ), 
      (∀ y ∈ X, min ≤ y ∧ y ≤ max) ∧ 
      max < 2 * min) ∧
    (∀ A : Finset ℝ, A.Nonempty → A ⊆ S → ∃ X ∈ P, (A.sum id) ∈ X) :=
sorry

end NUMINAMATH_CALUDE_subset_sum_partition_l3800_380057


namespace NUMINAMATH_CALUDE_average_annual_growth_rate_l3800_380024

/-- Proves that the average annual growth rate is 20% given the initial and final revenues --/
theorem average_annual_growth_rate 
  (initial_revenue : ℝ) 
  (final_revenue : ℝ) 
  (years : ℕ) 
  (h1 : initial_revenue = 280)
  (h2 : final_revenue = 403.2)
  (h3 : years = 2) :
  ∃ (growth_rate : ℝ), 
    growth_rate = 0.2 ∧ 
    final_revenue = initial_revenue * (1 + growth_rate) ^ years :=
by sorry


end NUMINAMATH_CALUDE_average_annual_growth_rate_l3800_380024


namespace NUMINAMATH_CALUDE_boys_to_total_ratio_l3800_380096

/-- Represents a classroom with boys and girls -/
structure Classroom where
  total_students : ℕ
  num_boys : ℕ
  num_girls : ℕ
  boys_plus_girls : num_boys + num_girls = total_students

/-- The probability of choosing a student from a group -/
def prob_choose (group : ℕ) (total : ℕ) : ℚ :=
  group / total

theorem boys_to_total_ratio (c : Classroom) 
  (h1 : c.total_students > 0)
  (h2 : prob_choose c.num_boys c.total_students = 
        (3 / 4) * prob_choose c.num_girls c.total_students) :
  (c.num_boys : ℚ) / c.total_students = 3 / 7 := by
  sorry

#check boys_to_total_ratio

end NUMINAMATH_CALUDE_boys_to_total_ratio_l3800_380096


namespace NUMINAMATH_CALUDE_line_through_intersection_and_origin_l3800_380066

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 2*x - y + 7 = 0
def l2 (x y : ℝ) : Prop := y = 1 - x

-- Define the intersection point of l1 and l2
def intersection : ℝ × ℝ := ((-2 : ℝ), (3 : ℝ))

-- Define the origin
def origin : ℝ × ℝ := ((0 : ℝ), (0 : ℝ))

-- Theorem statement
theorem line_through_intersection_and_origin :
  ∀ (x y : ℝ), l1 (intersection.1) (intersection.2) ∧ 
               l2 (intersection.1) (intersection.2) ∧ 
               (3*x + 2*y = 0 ↔ ∃ t : ℝ, x = t * (intersection.1 - origin.1) ∧ 
                                        y = t * (intersection.2 - origin.2)) :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_origin_l3800_380066


namespace NUMINAMATH_CALUDE_bobby_total_pieces_l3800_380026

/-- The total number of candy and chocolate pieces Bobby ate -/
def total_pieces (initial_candy : ℕ) (additional_candy : ℕ) (chocolate : ℕ) : ℕ :=
  initial_candy + additional_candy + chocolate

/-- Theorem stating that Bobby ate 51 pieces of candy and chocolate in total -/
theorem bobby_total_pieces :
  total_pieces 33 4 14 = 51 := by
  sorry

end NUMINAMATH_CALUDE_bobby_total_pieces_l3800_380026


namespace NUMINAMATH_CALUDE_lisa_additional_marbles_l3800_380087

/-- The minimum number of additional marbles Lisa needs -/
def minimum_additional_marbles (num_friends : ℕ) (current_marbles : ℕ) : ℕ :=
  let min_marbles_per_friend := 3
  let max_marbles_per_friend := min_marbles_per_friend + num_friends - 1
  let total_marbles_needed := num_friends * (min_marbles_per_friend + max_marbles_per_friend) / 2
  max (total_marbles_needed - current_marbles) 0

/-- Theorem stating the minimum number of additional marbles Lisa needs -/
theorem lisa_additional_marbles :
  minimum_additional_marbles 12 50 = 52 := by
  sorry

#eval minimum_additional_marbles 12 50

end NUMINAMATH_CALUDE_lisa_additional_marbles_l3800_380087


namespace NUMINAMATH_CALUDE_matrix_equation_holds_l3800_380014

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 2, 1; 2, 0, 2; 1, 2, 0]

theorem matrix_equation_holds :
  B^3 + (-8 : ℤ) • B^2 + (-12 : ℤ) • B + (-28 : ℤ) • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_holds_l3800_380014


namespace NUMINAMATH_CALUDE_amelia_win_probability_l3800_380029

/-- Represents the outcome of a single coin toss -/
inductive CoinToss
| Heads
| Tails

/-- Represents a player in the game -/
inductive Player
| Amelia
| Blaine

/-- The state of the game after each round -/
structure GameState :=
  (round : Nat)
  (currentPlayer : Player)

/-- The result of the game -/
inductive GameResult
| AmeliaWins
| BlaineWins
| Tie

/-- The probability of getting heads for each player -/
def headsProbability (player : Player) : ℚ :=
  match player with
  | Player.Amelia => 1/4
  | Player.Blaine => 1/3

/-- The probability of the game ending in a specific result -/
noncomputable def gameResultProbability (result : GameResult) : ℚ :=
  sorry

/-- The main theorem stating the probability of Amelia winning -/
theorem amelia_win_probability :
  gameResultProbability GameResult.AmeliaWins = 15/32 :=
sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l3800_380029


namespace NUMINAMATH_CALUDE_average_first_50_naturals_l3800_380006

theorem average_first_50_naturals : 
  let n : ℕ := 50
  let sum : ℕ := n * (n + 1) / 2
  (sum : ℚ) / n = 25.5 := by sorry

end NUMINAMATH_CALUDE_average_first_50_naturals_l3800_380006


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3800_380069

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (20/7) * x^2 - (60/7) * x - 360/7

/-- Theorem stating that q satisfies the required conditions -/
theorem q_satisfies_conditions :
  q (-3) = 0 ∧ q 6 = 0 ∧ q (-1) = -40 := by
  sorry


end NUMINAMATH_CALUDE_q_satisfies_conditions_l3800_380069


namespace NUMINAMATH_CALUDE_right_triangle_sin_A_l3800_380010

theorem right_triangle_sin_A (A B C : Real) :
  -- Right triangle ABC with ∠B = 90°
  0 < A ∧ A < Real.pi / 2 →
  0 < C ∧ C < Real.pi / 2 →
  A + C = Real.pi / 2 →
  -- 3 tan A = 4
  3 * Real.tan A = 4 →
  -- Conclusion: sin A = 4/5
  Real.sin A = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_A_l3800_380010


namespace NUMINAMATH_CALUDE_simplify_power_l3800_380062

theorem simplify_power (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by sorry

end NUMINAMATH_CALUDE_simplify_power_l3800_380062


namespace NUMINAMATH_CALUDE_complex_simplification_l3800_380033

theorem complex_simplification :
  (7 * (4 - 2 * Complex.I) + 4 * Complex.I * (7 - 3 * Complex.I)) = (40 : ℂ) + 14 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l3800_380033


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3800_380017

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → 2^x > x^2) ↔ (∃ x : ℝ, x ≥ 0 ∧ 2^x ≤ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3800_380017


namespace NUMINAMATH_CALUDE_farm_fencing_cost_l3800_380008

/-- Proves that the cost of fencing per meter is 15 for a rectangular farm with given conditions -/
theorem farm_fencing_cost (area : ℝ) (short_side : ℝ) (total_cost : ℝ) :
  area = 1200 →
  short_side = 30 →
  total_cost = 1800 →
  let long_side := area / short_side
  let diagonal := Real.sqrt (long_side ^ 2 + short_side ^ 2)
  let total_length := long_side + short_side + diagonal
  total_cost / total_length = 15 := by
  sorry

end NUMINAMATH_CALUDE_farm_fencing_cost_l3800_380008


namespace NUMINAMATH_CALUDE_square_of_sum_31_3_l3800_380088

theorem square_of_sum_31_3 : 31^2 + 2*(31*3) + 3^2 = 1156 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_31_3_l3800_380088


namespace NUMINAMATH_CALUDE_b_received_15_pencils_l3800_380037

/-- The number of pencils each student received -/
structure PencilDistribution where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- The conditions of the pencil distribution problem -/
def ValidDistribution (p : PencilDistribution) : Prop :=
  p.a + p.b + p.c + p.d = 53 ∧
  (max p.a (max p.b (max p.c p.d))) - (min p.a (min p.b (min p.c p.d))) ≤ 5 ∧
  p.a + p.b = 2 * p.c ∧
  p.c + p.b = 2 * p.d

/-- The theorem stating that B received 15 pencils -/
theorem b_received_15_pencils (p : PencilDistribution) (h : ValidDistribution p) : p.b = 15 := by
  sorry

end NUMINAMATH_CALUDE_b_received_15_pencils_l3800_380037


namespace NUMINAMATH_CALUDE_twelve_bushes_for_sixty_zucchinis_l3800_380093

/-- The number of blueberry bushes needed to obtain a given number of zucchinis -/
def bushes_needed (zucchinis : ℕ) (containers_per_bush : ℕ) (containers_per_trade : ℕ) (zucchinis_per_trade : ℕ) : ℕ :=
  (zucchinis * containers_per_trade) / (zucchinis_per_trade * containers_per_bush)

/-- Theorem: 12 bushes are needed to obtain 60 zucchinis -/
theorem twelve_bushes_for_sixty_zucchinis :
  bushes_needed 60 10 6 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_bushes_for_sixty_zucchinis_l3800_380093
