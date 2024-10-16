import Mathlib

namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l3875_387511

theorem chicken_wings_distribution (total_friends : ℕ) (initial_wings : ℕ) (cooked_wings : ℕ) (non_eating_friends : ℕ) :
  total_friends = 15 →
  initial_wings = 7 →
  cooked_wings = 45 →
  non_eating_friends = 2 →
  (initial_wings + cooked_wings) / (total_friends - non_eating_friends) = 4 :=
by sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l3875_387511


namespace NUMINAMATH_CALUDE_right_angled_triangle_set_l3875_387508

theorem right_angled_triangle_set : ∃! (a b c : ℝ), 
  ((a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3) ∨
   (a = 2 ∧ b = 3 ∧ c = 4) ∨
   (a = 4 ∧ b = 6 ∧ c = 8) ∨
   (a = 5 ∧ b = 12 ∧ c = 15)) ∧
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_set_l3875_387508


namespace NUMINAMATH_CALUDE_optimal_plan_l3875_387525

/-- Represents a sewage treatment equipment purchasing plan -/
structure Plan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a plan is valid according to the given constraints -/
def isValidPlan (p : Plan) : Prop :=
  p.typeA + p.typeB = 20 ∧
  120000 * p.typeA + 100000 * p.typeB ≤ 2300000 ∧
  240 * p.typeA + 200 * p.typeB ≥ 4500

/-- Calculates the total cost of a plan -/
def planCost (p : Plan) : ℕ :=
  120000 * p.typeA + 100000 * p.typeB

/-- Theorem stating that the optimal plan is 13 units of A and 7 units of B -/
theorem optimal_plan :
  ∃ (optimalPlan : Plan),
    isValidPlan optimalPlan ∧
    optimalPlan.typeA = 13 ∧
    optimalPlan.typeB = 7 ∧
    planCost optimalPlan = 2260000 ∧
    ∀ (p : Plan), isValidPlan p → planCost p ≥ planCost optimalPlan :=
  sorry


end NUMINAMATH_CALUDE_optimal_plan_l3875_387525


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3875_387545

/-- For a regular polygon with n sides, if each exterior angle measures 45°, then n = 8. -/
theorem regular_polygon_exterior_angle (n : ℕ) : n > 2 → (360 : ℝ) / n = 45 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3875_387545


namespace NUMINAMATH_CALUDE_fraction_equality_l3875_387571

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3875_387571


namespace NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l3875_387555

/-- Given a point P in a 3D Cartesian coordinate system, 
    return its symmetric point P' with respect to the y-axis -/
def symmetric_point_y_axis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := P
  (-x, y, -z)

theorem symmetry_wrt_y_axis :
  let P : ℝ × ℝ × ℝ := (2, -4, 6)
  symmetric_point_y_axis P = (-2, -4, -6) := by
sorry

end NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l3875_387555


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l3875_387504

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt x}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -x^2}

-- Theorem statement
theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | x > 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l3875_387504


namespace NUMINAMATH_CALUDE_no_real_roots_l3875_387588

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 4) - Real.sqrt (x - 3) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3875_387588


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3875_387598

/-- A rectangle with perimeter 60 cm and area 225 cm² has a diagonal of 15√2 cm. -/
theorem rectangle_diagonal (x y : ℝ) (h_perimeter : x + y = 30) (h_area : x * y = 225) :
  Real.sqrt (x^2 + y^2) = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3875_387598


namespace NUMINAMATH_CALUDE_sugar_solution_concentration_l3875_387549

theorem sugar_solution_concentration (W : ℝ) (X : ℝ) : 
  W > 0 → -- W is positive (total weight of solution)
  0.08 * W = 0.08 * W - 0.02 * W + X * W / 400 → -- Sugar balance equation
  0.16 * W = 0.06 * W + X * W / 400 → -- Final concentration equation
  X = 40 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_concentration_l3875_387549


namespace NUMINAMATH_CALUDE_multiply_fractions_of_numbers_l3875_387560

theorem multiply_fractions_of_numbers : 
  (1/4 : ℚ) * 15 * ((1/3 : ℚ) * 10) = 25/2 := by sorry

end NUMINAMATH_CALUDE_multiply_fractions_of_numbers_l3875_387560


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3875_387547

theorem solution_set_quadratic_inequality :
  Set.Ioo 0 3 = {x : ℝ | x^2 - 3*x < 0} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3875_387547


namespace NUMINAMATH_CALUDE_common_root_theorem_l3875_387538

theorem common_root_theorem (a b c d : ℝ) (h1 : a + d = 2017) (h2 : b + c = 2017) :
  ∃ x : ℝ, x = 2017 / 2 ∧ (x - a) * (x - b) = (x - c) * (x - d) :=
by sorry

end NUMINAMATH_CALUDE_common_root_theorem_l3875_387538


namespace NUMINAMATH_CALUDE_cody_ticket_bill_l3875_387513

/-- Calculates the total bill for game tickets given the number of adult and children tickets -/
def total_bill (adult_tickets : ℕ) (children_tickets : ℕ) : ℚ :=
  12 * adult_tickets + (15/2) * children_tickets

/-- Proves that Cody's total bill for game tickets is $99.00 -/
theorem cody_ticket_bill : ∃ (adult_tickets children_tickets : ℕ),
  adult_tickets + children_tickets = 12 ∧
  children_tickets = adult_tickets + 8 ∧
  total_bill adult_tickets children_tickets = 99 := by
sorry


end NUMINAMATH_CALUDE_cody_ticket_bill_l3875_387513


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3875_387521

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a*b + b*c + c*a = 72) : 
  a + b + c = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3875_387521


namespace NUMINAMATH_CALUDE_football_field_area_is_9600_l3875_387590

/-- The total area of a football field in square yards -/
def football_field_area : ℝ := 9600

/-- The total amount of fertilizer used on the entire field in pounds -/
def total_fertilizer : ℝ := 1200

/-- The amount of fertilizer used on a part of the field in pounds -/
def partial_fertilizer : ℝ := 700

/-- The area covered by the partial fertilizer in square yards -/
def partial_area : ℝ := 5600

/-- Theorem stating that the football field area is 9600 square yards -/
theorem football_field_area_is_9600 : 
  football_field_area = (total_fertilizer * partial_area) / partial_fertilizer := by
  sorry

end NUMINAMATH_CALUDE_football_field_area_is_9600_l3875_387590


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3875_387565

-- Define the complex number z
def z : ℂ := (1 - Complex.I) * (2 + Complex.I)

-- Theorem statement
theorem z_in_fourth_quadrant :
  Complex.re z > 0 ∧ Complex.im z < 0 :=
by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3875_387565


namespace NUMINAMATH_CALUDE_ticket_cost_proof_l3875_387575

def initial_amount : ℕ := 760
def remaining_amount : ℕ := 310
def ticket_cost : ℕ := 300

theorem ticket_cost_proof :
  (initial_amount - remaining_amount = ticket_cost + ticket_cost / 2) →
  ticket_cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_proof_l3875_387575


namespace NUMINAMATH_CALUDE_tournament_games_count_l3875_387534

/-- Calculates the number of games in a single-elimination tournament. -/
def singleEliminationGames (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

/-- Calculates the total number of games in the tournament. -/
def totalGames (initialTeams : ℕ) : ℕ :=
  let preliminaryGames := initialTeams / 2
  let remainingTeams := initialTeams - preliminaryGames
  preliminaryGames + singleEliminationGames remainingTeams

/-- Theorem stating that the total number of games in the described tournament is 23. -/
theorem tournament_games_count :
  totalGames 24 = 23 := by sorry

end NUMINAMATH_CALUDE_tournament_games_count_l3875_387534


namespace NUMINAMATH_CALUDE_watermelon_cost_l3875_387543

/-- The problem of determining the cost of a watermelon --/
theorem watermelon_cost (total_fruits : ℕ) (total_value : ℕ) 
  (melon_capacity : ℕ) (watermelon_capacity : ℕ) :
  total_fruits = 150 →
  total_value = 24000 →
  melon_capacity = 120 →
  watermelon_capacity = 160 →
  ∃ (num_watermelons num_melons : ℕ) (watermelon_cost melon_cost : ℚ),
    num_watermelons + num_melons = total_fruits ∧
    num_watermelons * watermelon_cost = num_melons * melon_cost ∧
    num_watermelons * watermelon_cost + num_melons * melon_cost = total_value ∧
    (num_watermelons : ℚ) / watermelon_capacity + (num_melons : ℚ) / melon_capacity = 1 ∧
    watermelon_cost = 100 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_cost_l3875_387543


namespace NUMINAMATH_CALUDE_sheet_width_calculation_l3875_387528

/-- Proves that a sheet with given dimensions and margins has a width of 20 cm when 64% is used for typing -/
theorem sheet_width_calculation (w : ℝ) : 
  w > 0 ∧ 
  (w - 4) * 24 = 0.64 * w * 30 → 
  w = 20 := by
  sorry

end NUMINAMATH_CALUDE_sheet_width_calculation_l3875_387528


namespace NUMINAMATH_CALUDE_birds_total_distance_l3875_387505

def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30
def flight_time : ℕ := 2

def total_distance : ℕ := eagle_speed * flight_time + falcon_speed * flight_time + 
                           pelican_speed * flight_time + hummingbird_speed * flight_time

theorem birds_total_distance : total_distance = 248 := by
  sorry

end NUMINAMATH_CALUDE_birds_total_distance_l3875_387505


namespace NUMINAMATH_CALUDE_min_difference_of_product_l3875_387500

theorem min_difference_of_product (a b : ℤ) (h : a * b = 156) :
  ∀ x y : ℤ, x * y = 156 → a - b ≤ x - y :=
by sorry

end NUMINAMATH_CALUDE_min_difference_of_product_l3875_387500


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3875_387529

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 5, 7]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_equation_solution (x y z w : ℝ) 
  (h1 : A * B x y z w = B x y z w * A)
  (h2 : 2 * z ≠ 5 * y) :
  ∃ x y z w, (x - w) / (z - 2 * y) = 0 :=
by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3875_387529


namespace NUMINAMATH_CALUDE_christen_peeled_23_potatoes_l3875_387594

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  totalPotatoes : ℕ
  homerRate : ℕ
  christenInitialRate : ℕ
  christenFinalRate : ℕ
  homerAloneTime : ℕ
  workTogetherTime : ℕ
  christenBreakTime : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledPotatoes (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- The theorem stating that Christen peeled 23 potatoes -/
theorem christen_peeled_23_potatoes :
  let scenario := PotatoPeeling.mk 60 4 6 4 5 3 2
  christenPeeledPotatoes scenario = 23 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_23_potatoes_l3875_387594


namespace NUMINAMATH_CALUDE_cubic_root_equation_l3875_387569

theorem cubic_root_equation : 2 / (2 - Real.rpow 3 (1/3)) = 2 * (2 + Real.rpow 3 (1/3)) * (4 + Real.rpow 9 (1/3)) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_l3875_387569


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l3875_387580

/-- Represents a person in the arrangement -/
inductive Person
| Man (n : Fin 4)
| Woman (n : Fin 4)

/-- A circular arrangement of people -/
def CircularArrangement := List Person

/-- Checks if two people can be adjacent in the arrangement -/
def canBeAdjacent (p1 p2 : Person) : Bool :=
  match p1, p2 with
  | Person.Man _, Person.Woman _ => true
  | Person.Woman _, Person.Man _ => true
  | _, _ => false

/-- Checks if a circular arrangement is valid -/
def isValidArrangement (arr : CircularArrangement) : Bool :=
  arr.length = 8 ∧
  arr.all (fun p => match p with
    | Person.Man n => n.val < 4
    | Person.Woman n => n.val < 4) ∧
  (List.zip arr (arr.rotateLeft 1)).all (fun (p1, p2) => canBeAdjacent p1 p2) ∧
  (List.zip arr (arr.rotateLeft 1)).all (fun (p1, p2) =>
    match p1, p2 with
    | Person.Man n1, Person.Woman n2 => n1 ≠ n2
    | Person.Woman n1, Person.Man n2 => n1 ≠ n2
    | _, _ => true)

/-- Counts the number of valid circular arrangements -/
def countValidArrangements : Nat :=
  (List.filter isValidArrangement (List.permutations (List.map Person.Man (List.range 4) ++ List.map Person.Woman (List.range 4)))).length / 8

theorem valid_arrangements_count :
  countValidArrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l3875_387580


namespace NUMINAMATH_CALUDE_intersection_distance_l3875_387501

/-- The distance between the intersection points of a parabola and a circle -/
theorem intersection_distance (x1 y1 x2 y2 : ℝ) : 
  (y1^2 = 12*x1) →
  (x1^2 + y1^2 - 4*x1 - 6*y1 = 0) →
  (y2^2 = 12*x2) →
  (x2^2 + y2^2 - 4*x2 - 6*y2 = 0) →
  x1 ≠ x2 ∨ y1 ≠ y2 →
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2) = 3 * 13^(1/2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l3875_387501


namespace NUMINAMATH_CALUDE_sara_trout_count_l3875_387550

theorem sara_trout_count (melanie_trout : ℕ) (sara_trout : ℕ) : 
  melanie_trout = 10 → 
  melanie_trout = 2 * sara_trout → 
  sara_trout = 5 := by
sorry

end NUMINAMATH_CALUDE_sara_trout_count_l3875_387550


namespace NUMINAMATH_CALUDE_delta_phi_composition_l3875_387586

/-- Given two functions δ and φ, prove that δ(φ(x)) = 3 if and only if x = -19/20 -/
theorem delta_phi_composition (δ φ : ℝ → ℝ) (h1 : ∀ x, δ x = 4 * x + 6) (h2 : ∀ x, φ x = 5 * x + 4) :
  (∃ x, δ (φ x) = 3) ↔ (∃ x, x = -19/20) :=
by sorry

end NUMINAMATH_CALUDE_delta_phi_composition_l3875_387586


namespace NUMINAMATH_CALUDE_employee_selection_distribution_l3875_387544

theorem employee_selection_distribution 
  (total_employees : ℕ) 
  (under_35 : ℕ) 
  (between_35_49 : ℕ) 
  (over_50 : ℕ) 
  (selected : ℕ) 
  (h1 : total_employees = 500) 
  (h2 : under_35 = 125) 
  (h3 : between_35_49 = 280) 
  (h4 : over_50 = 95) 
  (h5 : selected = 100) 
  (h6 : total_employees = under_35 + between_35_49 + over_50) :
  let select_under_35 := (under_35 * selected) / total_employees
  let select_between_35_49 := (between_35_49 * selected) / total_employees
  let select_over_50 := (over_50 * selected) / total_employees
  select_under_35 = 25 ∧ select_between_35_49 = 56 ∧ select_over_50 = 19 := by
  sorry

end NUMINAMATH_CALUDE_employee_selection_distribution_l3875_387544


namespace NUMINAMATH_CALUDE_line_equation_from_circle_intersection_l3875_387561

/-- Given a circle and a line intersecting it, prove the equation of the line. -/
theorem line_equation_from_circle_intersection (a : ℝ) (h_a : a < 3) :
  let circle := fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y + a = 0
  let midpoint := (-2, 3)
  ∃ (A B : ℝ × ℝ),
    circle A.1 A.2 ∧
    circle B.1 B.2 ∧
    (A.1 + B.1) / 2 = midpoint.1 ∧
    (A.2 + B.2) / 2 = midpoint.2 →
    ∃ (m b : ℝ), ∀ (x y : ℝ), y = m*x + b ↔ x - y + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_circle_intersection_l3875_387561


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3875_387522

theorem polynomial_divisibility (c : ℤ) : 
  (∃ q : Polynomial ℤ, (X^2 + X + c) * q = X^13 - X + 106) ↔ c = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3875_387522


namespace NUMINAMATH_CALUDE_symmetric_points_imply_a_equals_two_l3875_387591

/-- Two points are symmetric about the x-axis if their x-coordinates are equal and their y-coordinates are negatives of each other -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_imply_a_equals_two (a b : ℝ) :
  let M : ℝ × ℝ := (2*a + b, a - 2*b)
  let N : ℝ × ℝ := (1 - 2*b, -2*a - b - 1)
  symmetric_about_x_axis M N → a = 2 := by
  sorry

#check symmetric_points_imply_a_equals_two

end NUMINAMATH_CALUDE_symmetric_points_imply_a_equals_two_l3875_387591


namespace NUMINAMATH_CALUDE_visitor_difference_l3875_387551

def visitors_current_day : ℕ := 317
def visitors_previous_day : ℕ := 295

theorem visitor_difference : visitors_current_day - visitors_previous_day = 22 := by
  sorry

end NUMINAMATH_CALUDE_visitor_difference_l3875_387551


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3875_387577

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 + I) :
  (z - 2) / z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3875_387577


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3875_387592

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3875_387592


namespace NUMINAMATH_CALUDE_linear_function_composition_l3875_387574

/-- A linear function f: ℝ → ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b ∧ a ≠ 0

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x : ℝ, f (f x) = x - 2) → ∀ x : ℝ, f x = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_composition_l3875_387574


namespace NUMINAMATH_CALUDE_max_value_7a_minus_9b_l3875_387597

theorem max_value_7a_minus_9b (r₁ r₂ r₃ a b : ℝ) : 
  (r₁ * r₁ * r₁ - r₁ * r₁ + a * r₁ - b = 0) →
  (r₂ * r₂ * r₂ - r₂ * r₂ + a * r₂ - b = 0) →
  (r₃ * r₃ * r₃ - r₃ * r₃ + a * r₃ - b = 0) →
  (0 < r₁ ∧ r₁ < 1) →
  (0 < r₂ ∧ r₂ < 1) →
  (0 < r₃ ∧ r₃ < 1) →
  7 * a - 9 * b ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_7a_minus_9b_l3875_387597


namespace NUMINAMATH_CALUDE_time_per_furniture_piece_l3875_387557

theorem time_per_furniture_piece (chairs tables total_time : ℕ) 
  (h1 : chairs = 4)
  (h2 : tables = 2)
  (h3 : total_time = 48) : 
  total_time / (chairs + tables) = 8 := by
  sorry

end NUMINAMATH_CALUDE_time_per_furniture_piece_l3875_387557


namespace NUMINAMATH_CALUDE_goods_transportable_l3875_387570

-- Define the problem parameters
def total_weight : ℝ := 13.5
def max_package_weight : ℝ := 0.35
def num_trucks : ℕ := 11
def truck_capacity : ℝ := 1.5

-- Theorem statement
theorem goods_transportable :
  total_weight ≤ (num_trucks : ℝ) * truck_capacity ∧
  ∃ (num_packages : ℕ), (num_packages : ℝ) * max_package_weight ≥ total_weight :=
by sorry

end NUMINAMATH_CALUDE_goods_transportable_l3875_387570


namespace NUMINAMATH_CALUDE_equation_holds_for_all_n_l3875_387537

-- Define α as the positive root of x^2 - 1989x - 1 = 0
noncomputable def α : ℝ := (1989 + Real.sqrt (1989^2 + 4)) / 2

-- Define the equation to be proven
def equation (n : ℕ) : Prop :=
  ⌊α * n + 1989 * α * ⌊α * n⌋⌋ = 1989 * n + (1989^2 + 1) * ⌊α * n⌋

-- Theorem statement
theorem equation_holds_for_all_n : ∀ n : ℕ, equation n := by sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_n_l3875_387537


namespace NUMINAMATH_CALUDE_least_integer_square_condition_l3875_387532

theorem least_integer_square_condition (x : ℤ) : x^2 = 3*(2*x) + 50 → x ≥ -4 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_condition_l3875_387532


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3875_387539

theorem trig_expression_equality : 
  1 / Real.sin (40 * π / 180) - Real.sqrt 2 / Real.cos (40 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3875_387539


namespace NUMINAMATH_CALUDE_tan_cos_expression_equals_negative_one_l3875_387585

theorem tan_cos_expression_equals_negative_one :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_cos_expression_equals_negative_one_l3875_387585


namespace NUMINAMATH_CALUDE_cubic_value_given_quadratic_l3875_387587

theorem cubic_value_given_quadratic (x : ℝ) : 
  x^2 + x - 1 = 0 → x^3 + 2*x^2 + 2005 = 2006 := by
  sorry

end NUMINAMATH_CALUDE_cubic_value_given_quadratic_l3875_387587


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3875_387559

/-- The center of the circle (x-1)^2 + (y+1)^2 = 2 -/
def circle_center : ℝ × ℝ := (1, -1)

/-- The slope of the given line 2x + y = 0 -/
def given_line_slope : ℝ := -2

/-- The perpendicular line passing through the circle center -/
def perpendicular_line (x y : ℝ) : Prop :=
  x - given_line_slope * y - (circle_center.1 - given_line_slope * circle_center.2) = 0

theorem perpendicular_line_equation :
  perpendicular_line = fun x y ↦ x - 2 * y - 3 = 0 := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3875_387559


namespace NUMINAMATH_CALUDE_log2_order_relation_l3875_387523

-- Define the logarithm function with base 2
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log2_order_relation :
  (∀ a b : ℝ, f a > f b → a > b) ∧
  ¬(∀ a b : ℝ, a > b → f a > f b) :=
sorry

end NUMINAMATH_CALUDE_log2_order_relation_l3875_387523


namespace NUMINAMATH_CALUDE_cube_root_function_l3875_387558

theorem cube_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (64 : ℝ)^(1/3) ∧ y = 8) →
  k * (27 : ℝ)^(1/3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_function_l3875_387558


namespace NUMINAMATH_CALUDE_bead_purchase_cost_l3875_387530

/-- Calculate the total cost of bead sets after discounts and taxes --/
theorem bead_purchase_cost (crystal_price metal_price glass_price : ℚ)
  (crystal_sets metal_sets glass_sets : ℕ)
  (crystal_discount metal_tax glass_discount : ℚ) :
  let crystal_cost := crystal_price * crystal_sets * (1 - crystal_discount)
  let metal_cost := metal_price * metal_sets * (1 + metal_tax)
  let glass_cost := glass_price * glass_sets * (1 - glass_discount)
  crystal_cost + metal_cost + glass_cost = 11028 / 100 →
  crystal_price = 12 →
  metal_price = 15 →
  glass_price = 8 →
  crystal_sets = 3 →
  metal_sets = 4 →
  glass_sets = 2 →
  crystal_discount = 1 / 10 →
  metal_tax = 1 / 20 →
  glass_discount = 7 / 100 →
  true := by sorry

end NUMINAMATH_CALUDE_bead_purchase_cost_l3875_387530


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3875_387503

theorem polynomial_evaluation (x : ℕ) (h : x = 4) :
  x^4 + x^3 + x^2 + x + 1 = 341 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3875_387503


namespace NUMINAMATH_CALUDE_brocard_angle_inequalities_l3875_387564

theorem brocard_angle_inequalities (α β γ φ : Real) 
  (triangle : α + β + γ = Real.pi)
  (brocard_condition : φ ≤ Real.pi / 6)
  (sin_relation : Real.sin (α - φ) * Real.sin (β - φ) * Real.sin (γ - φ) = Real.sin φ ^ 3) :
  φ ^ 3 ≤ (α - φ) * (β - φ) * (γ - φ) ∧ 8 * φ ^ 3 ≤ α * β * γ := by
  sorry

end NUMINAMATH_CALUDE_brocard_angle_inequalities_l3875_387564


namespace NUMINAMATH_CALUDE_art_museum_survey_l3875_387546

theorem art_museum_survey (total : ℕ) (not_enjoyed_not_understood : ℕ) (enjoyed : ℕ) (understood : ℕ) :
  total = 400 →
  not_enjoyed_not_understood = 100 →
  enjoyed = understood →
  (enjoyed : ℚ) / total = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_art_museum_survey_l3875_387546


namespace NUMINAMATH_CALUDE_tangent_parallel_and_inequality_l3875_387576

/-- The function f(x) = x³ - ax² + 3x + b -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + 3*x + b

/-- The derivative of f(x) -/
def f' (a x : ℝ) : ℝ := 3*x^2 - 2*a*x + 3

theorem tangent_parallel_and_inequality (a b : ℝ) :
  (f' a 1 = 0) →
  (∀ x ∈ Set.Icc (-1 : ℝ) 4, f a b x > f' a x) →
  (a = 3 ∧ b > 19) := by
  sorry


end NUMINAMATH_CALUDE_tangent_parallel_and_inequality_l3875_387576


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l3875_387512

theorem fraction_zero_solution (x : ℝ) :
  x ≠ 0 →
  (x^2 - 4*x + 3) / (5*x) = 0 ↔ x = 1 ∨ x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_solution_l3875_387512


namespace NUMINAMATH_CALUDE_evaluate_expression_l3875_387542

theorem evaluate_expression : 
  Real.sqrt 8 * 2^(3/2) + 18 / 3 * 3 - 6^(5/2) = 26 - 36 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3875_387542


namespace NUMINAMATH_CALUDE_peaches_picked_l3875_387589

def initial_peaches : ℝ := 34.0
def total_peaches : ℕ := 120

theorem peaches_picked (picked : ℕ) : 
  picked = total_peaches - Int.floor initial_peaches := by sorry

end NUMINAMATH_CALUDE_peaches_picked_l3875_387589


namespace NUMINAMATH_CALUDE_defective_from_factory1_l3875_387548

/-- The probability of a product coming from the first factory -/
def p_factory1 : ℝ := 0.20

/-- The probability of a product coming from the second factory -/
def p_factory2 : ℝ := 0.46

/-- The probability of a product coming from the third factory -/
def p_factory3 : ℝ := 0.34

/-- The probability of a defective item from the first factory -/
def p_defective1 : ℝ := 0.03

/-- The probability of a defective item from the second factory -/
def p_defective2 : ℝ := 0.02

/-- The probability of a defective item from the third factory -/
def p_defective3 : ℝ := 0.01

/-- The probability that a randomly selected defective item was produced at the first factory -/
theorem defective_from_factory1 : 
  (p_defective1 * p_factory1) / (p_defective1 * p_factory1 + p_defective2 * p_factory2 + p_defective3 * p_factory3) = 0.322 := by
sorry

end NUMINAMATH_CALUDE_defective_from_factory1_l3875_387548


namespace NUMINAMATH_CALUDE_students_behind_minyoung_l3875_387518

/-- Given a line of students with Minyoung in it, this theorem proves
    that the number of students behind Minyoung is equal to the total
    number of students minus the number of students in front of Minyoung
    minus 1 (Minyoung herself). -/
theorem students_behind_minyoung
  (total_students : ℕ)
  (students_in_front : ℕ)
  (h1 : total_students = 35)
  (h2 : students_in_front = 27) :
  total_students - students_in_front - 1 = 7 :=
by sorry

end NUMINAMATH_CALUDE_students_behind_minyoung_l3875_387518


namespace NUMINAMATH_CALUDE_sum_reciprocals_l3875_387507

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -2) (hb : b ≠ -2) (hc : c ≠ -2) (hd : d ≠ -2)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1/(a + ω) + 1/(b + ω) + 1/(c + ω) + 1/(d + ω) = 4/ω) :
  1/(a + 2) + 1/(b + 2) + 1/(c + 2) + 1/(d + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l3875_387507


namespace NUMINAMATH_CALUDE_trajectory_of_equidistant_complex_l3875_387541

theorem trajectory_of_equidistant_complex (z : ℂ) :
  Complex.abs (z + 1 - Complex.I) = Complex.abs (z - 1 + Complex.I) →
  z.re = z.im :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_equidistant_complex_l3875_387541


namespace NUMINAMATH_CALUDE_complement_A_in_U_is_correct_l3875_387578

-- Define the universal set U
def U : Set Int := {x | -2 ≤ x ∧ x ≤ 6}

-- Define set A
def A : Set Int := {x | ∃ n : Nat, x = 2 * n ∧ n ≤ 3}

-- Define the complement of A in U
def complement_A_in_U : Set Int := U \ A

-- Theorem to prove
theorem complement_A_in_U_is_correct :
  complement_A_in_U = {-2, -1, 1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_is_correct_l3875_387578


namespace NUMINAMATH_CALUDE_opposite_of_three_minus_one_l3875_387566

theorem opposite_of_three_minus_one :
  -(3 - 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_minus_one_l3875_387566


namespace NUMINAMATH_CALUDE_committee_selection_ways_l3875_387519

/-- The number of ways to choose a k-person committee from a group of n people -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the club -/
def club_size : ℕ := 11

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- Theorem stating that the number of ways to choose a 5-person committee from a club of 11 people is 462 -/
theorem committee_selection_ways : choose club_size committee_size = 462 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l3875_387519


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3875_387535

theorem triangle_area_proof (square_side : ℝ) (overlap_ratio_square : ℝ) (overlap_ratio_triangle : ℝ) : 
  square_side = 8 →
  overlap_ratio_square = 3/4 →
  overlap_ratio_triangle = 1/2 →
  let square_area := square_side * square_side
  let overlap_area := square_area * overlap_ratio_square
  let triangle_area := overlap_area / overlap_ratio_triangle
  triangle_area = 96 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3875_387535


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3875_387526

/-- An arithmetic sequence with first four terms a, x, b, 2x has the property that a/b = 1/3 -/
theorem arithmetic_sequence_ratio (a x b : ℝ) : 
  (∃ d : ℝ, x - a = d ∧ b - x = d ∧ 2*x - b = d) → a/b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3875_387526


namespace NUMINAMATH_CALUDE_function_properties_l3875_387584

/-- A function f(x) = x^2 + bx + c where b and c are real numbers -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The theorem statement -/
theorem function_properties (b c : ℝ) 
  (h : ∀ x : ℝ, 2*x + b ≤ f b c x) : 
  (∀ x : ℝ, x ≥ 0 → f b c x ≤ (x + c)^2) ∧ 
  (∃ m : ℝ, m = 3/2 ∧ ∀ b' c' : ℝ, (∀ x : ℝ, 2*x + b' ≤ f b' c' x) → 
    f b' c' c' - f b' c' b' ≤ m*(c'^2 - b'^2) ∧
    ∀ m' : ℝ, (∀ b' c' : ℝ, (∀ x : ℝ, 2*x + b' ≤ f b' c' x) → 
      f b' c' c' - f b' c' b' ≤ m'*(c'^2 - b'^2)) → m ≤ m') := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3875_387584


namespace NUMINAMATH_CALUDE_area_APRQ_is_6_25_l3875_387579

/-- A rectangle with points P, Q, and R located on its sides. -/
structure RectangleWithPoints where
  /-- The area of rectangle ABCD -/
  area : ℝ
  /-- Point P is located one-fourth the length of side AD from vertex A -/
  p_location : ℝ
  /-- Point Q is located one-fourth the length of side CD from vertex C -/
  q_location : ℝ
  /-- Point R is located one-fourth the length of side BC from vertex B -/
  r_location : ℝ

/-- The area of quadrilateral APRQ in a rectangle with given properties -/
def area_APRQ (rect : RectangleWithPoints) : ℝ := sorry

/-- Theorem stating that the area of APRQ is 6.25 square meters -/
theorem area_APRQ_is_6_25 (rect : RectangleWithPoints) 
  (h1 : rect.area = 100)
  (h2 : rect.p_location = 1/4)
  (h3 : rect.q_location = 1/4)
  (h4 : rect.r_location = 1/4) : 
  area_APRQ rect = 6.25 := by sorry

end NUMINAMATH_CALUDE_area_APRQ_is_6_25_l3875_387579


namespace NUMINAMATH_CALUDE_calc_complex_fraction_l3875_387536

theorem calc_complex_fraction : (3^2 * 5^4 * 7^2) / 7 = 39375 := by
  sorry

end NUMINAMATH_CALUDE_calc_complex_fraction_l3875_387536


namespace NUMINAMATH_CALUDE_sector_max_area_l3875_387517

/-- Given a sector with circumference 36, the radian measure of the central angle
    that maximizes the area of the sector is 2. -/
theorem sector_max_area (r : ℝ) (α : ℝ) (h1 : r > 0) (h2 : α > 0) 
  (h3 : 2 * r + r * α = 36) : 
  (∀ β : ℝ, β > 0 → 2 * r + r * β = 36 → r * r * α / 2 ≤ r * r * β / 2) → α = 2 := 
sorry

end NUMINAMATH_CALUDE_sector_max_area_l3875_387517


namespace NUMINAMATH_CALUDE_investment_growth_l3875_387583

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.08

/-- The time period in years -/
def time_period : ℕ := 28

/-- The initial investment amount in dollars -/
def initial_investment : ℝ := 3500

/-- The final value after the investment period in dollars -/
def final_value : ℝ := 31500

/-- Compound interest formula: A = P(1 + r)^t 
    Where A is the final amount, P is the principal (initial investment),
    r is the annual interest rate, and t is the time in years -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_growth :
  compound_interest initial_investment interest_rate time_period = final_value := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l3875_387583


namespace NUMINAMATH_CALUDE_simplify_expression_l3875_387581

theorem simplify_expression (p : ℝ) (h1 : 1 < p) (h2 : p < 2) :
  Real.sqrt ((1 - p)^2) + (Real.sqrt (2 - p))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3875_387581


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3875_387582

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 5 ∧ b = 12 ∧ c = 13

/-- A square inscribed in a right triangle with a vertex at the right angle -/
def squareAtRightAngle (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x < t.a ∧ x < t.b ∧ x / t.a = x / t.b

/-- A square inscribed in a right triangle with a side along the hypotenuse -/
def squareAlongHypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y < t.c ∧ (t.a - y)^2 + (t.b - y)^2 = y^2

theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ)
  (h1 : squareAtRightAngle t1 x) (h2 : squareAlongHypotenuse t2 y) :
  x / y = 144 / 85 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3875_387582


namespace NUMINAMATH_CALUDE_probability_of_pirate_letter_l3875_387573

def probability_letters : Finset Char := {'P', 'R', 'O', 'B', 'A', 'I', 'L', 'T', 'Y'}
def pirate_letters : Finset Char := {'P', 'I', 'R', 'A', 'T', 'E'}

def total_tiles : ℕ := 11

theorem probability_of_pirate_letter :
  (probability_letters ∩ pirate_letters).card / total_tiles = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_pirate_letter_l3875_387573


namespace NUMINAMATH_CALUDE_expression_simplification_l3875_387514

variables (a b : ℝ)

theorem expression_simplification :
  (3*a + 2*b - 5*a - b = -2*a + b) ∧
  (5*(3*a^2*b - a*b^2) - (a*b^2 + 3*a^2*b) = 12*a^2*b - 6*a*b^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3875_387514


namespace NUMINAMATH_CALUDE_water_added_for_nine_percent_solution_l3875_387596

/-- Represents a solution of alcohol and water -/
structure Solution where
  volume : ℝ
  alcohol_percentage : ℝ

/-- Calculates the amount of alcohol in a solution -/
def alcohol_amount (s : Solution) : ℝ :=
  s.volume * s.alcohol_percentage

/-- The initial solution -/
def initial_solution : Solution :=
  { volume := 40, alcohol_percentage := 0.05 }

/-- The amount of alcohol added -/
def added_alcohol : ℝ := 2.5

/-- Theorem stating the condition for the final solution to be 9% alcohol -/
theorem water_added_for_nine_percent_solution (x : ℝ) : 
  let final_solution : Solution := 
    { volume := initial_solution.volume + added_alcohol + x,
      alcohol_percentage := 0.09 }
  alcohol_amount final_solution = alcohol_amount initial_solution + added_alcohol ↔ x = 7.5 :=
sorry

end NUMINAMATH_CALUDE_water_added_for_nine_percent_solution_l3875_387596


namespace NUMINAMATH_CALUDE_walter_zoo_time_l3875_387531

def time_at_zoo (seal_time penguin_factor elephant_time : ℕ) : ℕ :=
  seal_time + (seal_time * penguin_factor) + elephant_time

theorem walter_zoo_time :
  time_at_zoo 13 8 13 = 130 :=
by sorry

end NUMINAMATH_CALUDE_walter_zoo_time_l3875_387531


namespace NUMINAMATH_CALUDE_debt_equality_time_l3875_387527

/-- The number of days for two debts to become equal -/
def daysUntilEqualDebt (initialDebt1 initialDebt2 interestRate1 interestRate2 : ℚ) : ℚ :=
  (initialDebt2 - initialDebt1) / (initialDebt1 * interestRate1 - initialDebt2 * interestRate2)

/-- Theorem: Darren and Fergie's debts will be equal after 25 days -/
theorem debt_equality_time : 
  daysUntilEqualDebt 200 300 (8/100) (4/100) = 25 := by sorry

end NUMINAMATH_CALUDE_debt_equality_time_l3875_387527


namespace NUMINAMATH_CALUDE_estimated_probability_is_two_fifths_l3875_387509

/-- Represents a set of three-digit numbers -/
def RandomSet : Type := List Nat

/-- Checks if a number represents a rainy day (1-6) -/
def isRainyDay (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 6

/-- Counts the number of rainy days in a three-digit number -/
def countRainyDays (n : Nat) : Nat :=
  (if isRainyDay (n / 100) then 1 else 0) +
  (if isRainyDay ((n / 10) % 10) then 1 else 0) +
  (if isRainyDay (n % 10) then 1 else 0)

/-- Checks if a number represents exactly two rainy days -/
def hasTwoRainyDays (n : Nat) : Bool :=
  countRainyDays n = 2

/-- The given set of random numbers -/
def givenSet : RandomSet :=
  [180, 792, 454, 417, 165, 809, 798, 386, 196, 206]

/-- Theorem: The estimated probability of exactly two rainy days is 2/5 -/
theorem estimated_probability_is_two_fifths :
  (givenSet.filter hasTwoRainyDays).length / givenSet.length = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_estimated_probability_is_two_fifths_l3875_387509


namespace NUMINAMATH_CALUDE_jorge_simon_age_difference_l3875_387595

/-- Represents a person's age at a given year -/
structure AgeAtYear where
  age : ℕ
  year : ℕ

/-- Calculates the age difference between two people -/
def ageDifference (person1 : AgeAtYear) (person2 : AgeAtYear) : ℕ :=
  if person1.year = person2.year then
    if person1.age ≥ person2.age then person1.age - person2.age else person2.age - person1.age
  else
    sorry -- We don't handle different years in this simplified version

theorem jorge_simon_age_difference :
  let jorge2005 : AgeAtYear := { age := 16, year := 2005 }
  let simon2010 : AgeAtYear := { age := 45, year := 2010 }
  let yearDiff : ℕ := simon2010.year - jorge2005.year
  let jorgeAge2010 : ℕ := jorge2005.age + yearDiff
  ageDifference { age := simon2010.age, year := simon2010.year } { age := jorgeAge2010, year := simon2010.year } = 24 := by
  sorry


end NUMINAMATH_CALUDE_jorge_simon_age_difference_l3875_387595


namespace NUMINAMATH_CALUDE_tangent_problem_l3875_387502

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 1/2) 
  (h2 : Real.tan β = 1/3) : 
  Real.tan (α - π/4) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l3875_387502


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l3875_387506

theorem quadratic_two_real_roots 
  (a b c : ℝ) 
  (h : a * (a + b + c) < 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l3875_387506


namespace NUMINAMATH_CALUDE_art_club_committee_probability_l3875_387554

def art_club_size : ℕ := 24
def boys_count : ℕ := 12
def girls_count : ℕ := 12
def committee_size : ℕ := 5

theorem art_club_committee_probability :
  let total_combinations := Nat.choose art_club_size committee_size
  let all_boys_or_all_girls := 2 * Nat.choose boys_count committee_size
  (total_combinations - all_boys_or_all_girls : ℚ) / total_combinations = 3427 / 3542 := by
  sorry

end NUMINAMATH_CALUDE_art_club_committee_probability_l3875_387554


namespace NUMINAMATH_CALUDE_johnny_earnings_l3875_387567

def calculate_earnings (hourly_wage : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) 
  (overtime_rate : ℝ) (tax_rate : ℝ) (insurance_rate : ℝ) : ℝ :=
  let regular_pay := hourly_wage * regular_hours
  let overtime_pay := hourly_wage * overtime_rate * overtime_hours
  let total_earnings := regular_pay + overtime_pay
  let tax_deduction := total_earnings * tax_rate
  let insurance_deduction := total_earnings * insurance_rate
  total_earnings - tax_deduction - insurance_deduction

theorem johnny_earnings :
  let hourly_wage : ℝ := 8.25
  let regular_hours : ℝ := 40
  let overtime_hours : ℝ := 7
  let overtime_rate : ℝ := 1.5
  let tax_rate : ℝ := 0.08
  let insurance_rate : ℝ := 0.05
  abs (calculate_earnings hourly_wage regular_hours overtime_hours overtime_rate tax_rate insurance_rate - 362.47) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_johnny_earnings_l3875_387567


namespace NUMINAMATH_CALUDE_inequality_problem_l3875_387524

/-- Given real numbers a, b, c satisfying c < b < a and ac < 0,
    prove that cb² < ca² is not necessarily true,
    while ab > ac, c(b-a) > 0, and ac(a-c) < 0 are always true. -/
theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (∀ x y z : ℝ, x < y ∧ y < z ∧ x * z < 0 → x * y^2 < x * z^2 → False) ∧
  (a * b > a * c) ∧
  (c * (b - a) > 0) ∧
  (a * c * (a - c) < 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_problem_l3875_387524


namespace NUMINAMATH_CALUDE_dog_catches_rabbit_l3875_387556

/-- Represents the chase scenario between a dog and a rabbit -/
structure ChaseScenario where
  rabbit_head_start : ℕ
  rabbit_distance_ratio : ℕ
  dog_distance_ratio : ℕ
  rabbit_time_ratio : ℕ
  dog_time_ratio : ℕ

/-- Calculates the minimum number of steps the dog must run to catch the rabbit -/
def min_steps_to_catch (scenario : ChaseScenario) : ℕ :=
  sorry

/-- Theorem stating that given the specific chase scenario, the dog needs 240 steps to catch the rabbit -/
theorem dog_catches_rabbit :
  let scenario : ChaseScenario := {
    rabbit_head_start := 100,
    rabbit_distance_ratio := 8,
    dog_distance_ratio := 3,
    rabbit_time_ratio := 9,
    dog_time_ratio := 4
  }
  min_steps_to_catch scenario = 240 := by
  sorry

end NUMINAMATH_CALUDE_dog_catches_rabbit_l3875_387556


namespace NUMINAMATH_CALUDE_opposite_abs_sum_l3875_387599

theorem opposite_abs_sum (a m n : ℝ) : 
  (|a - 2| + |m + n + 3| = 0) → (a + m + n = -1) := by
sorry

end NUMINAMATH_CALUDE_opposite_abs_sum_l3875_387599


namespace NUMINAMATH_CALUDE_lg_ratio_theorem_l3875_387516

theorem lg_ratio_theorem (m n : ℝ) (hm : Real.log 2 = m) (hn : Real.log 3 = n) :
  (Real.log 12) / (Real.log 15) = (2*m + n) / (1 - m + n) := by
  sorry

end NUMINAMATH_CALUDE_lg_ratio_theorem_l3875_387516


namespace NUMINAMATH_CALUDE_smallest_perimeter_consecutive_integer_triangle_l3875_387533

/-- A triangle with consecutive integer side lengths greater than 1 -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 1 ∧ c = b + 1
  greater_than_one : a > 1

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ :=
  t.a + t.b + t.c

/-- The triangle inequality -/
def satisfies_triangle_inequality (t : ConsecutiveIntegerTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

theorem smallest_perimeter_consecutive_integer_triangle :
  ∀ t : ConsecutiveIntegerTriangle,
  satisfies_triangle_inequality t →
  perimeter t ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_consecutive_integer_triangle_l3875_387533


namespace NUMINAMATH_CALUDE_exists_power_two_minus_one_divisible_by_n_l3875_387562

theorem exists_power_two_minus_one_divisible_by_n (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ (n ∣ 2^k - 1) :=
sorry

end NUMINAMATH_CALUDE_exists_power_two_minus_one_divisible_by_n_l3875_387562


namespace NUMINAMATH_CALUDE_proportion_fourth_term_l3875_387515

theorem proportion_fourth_term (x y : ℝ) : 
  (0.75 : ℝ) / x = 5 / y ∧ x = 1.2 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportion_fourth_term_l3875_387515


namespace NUMINAMATH_CALUDE_roden_fish_purchase_cost_l3875_387593

/-- Calculate the total cost of Roden's fish purchase -/
theorem roden_fish_purchase_cost : 
  let goldfish_cost : ℕ := 15 * 3
  let blue_fish_cost : ℕ := 7 * 6
  let neon_tetra_cost : ℕ := 10 * 2
  let angelfish_cost : ℕ := 5 * 8
  let total_cost : ℕ := goldfish_cost + blue_fish_cost + neon_tetra_cost + angelfish_cost
  total_cost = 147 := by
  sorry

end NUMINAMATH_CALUDE_roden_fish_purchase_cost_l3875_387593


namespace NUMINAMATH_CALUDE_power_tower_mod_1000_l3875_387568

theorem power_tower_mod_1000 : 5^(5^(5^5)) ≡ 125 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_1000_l3875_387568


namespace NUMINAMATH_CALUDE_initial_time_is_six_hours_l3875_387520

/-- Proves that the initial time to cover 288 km is 6 hours -/
theorem initial_time_is_six_hours 
  (distance : ℝ) 
  (new_speed : ℝ) 
  (time_ratio : ℝ) :
  distance = 288 →
  new_speed = 32 →
  time_ratio = 3 / 2 →
  ∃ (initial_time : ℝ), 
    initial_time = 6 ∧ 
    distance = new_speed * (time_ratio * initial_time) :=
by sorry

end NUMINAMATH_CALUDE_initial_time_is_six_hours_l3875_387520


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l3875_387510

/-- Given two speeds and an additional distance, proves that the actual distance traveled is 50 km -/
theorem actual_distance_traveled (speed1 speed2 additional_distance : ℝ) 
  (h1 : speed1 = 10)
  (h2 : speed2 = 14)
  (h3 : additional_distance = 20)
  (h4 : ∀ D : ℝ, D / speed1 = (D + additional_distance) / speed2) :
  ∃ D : ℝ, D = 50 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l3875_387510


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3875_387540

theorem ratio_x_to_y (x y : ℚ) (h : (12 * x - 7 * y) / (17 * x - 3 * y) = 4 / 7) :
  x / y = 37 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3875_387540


namespace NUMINAMATH_CALUDE_birds_on_fence_l3875_387563

/-- Given an initial number of birds on a fence and an additional number of birds that land on the fence,
    calculate the total number of birds on the fence. -/
def total_birds (initial : Nat) (additional : Nat) : Nat :=
  initial + additional

/-- Theorem stating that with 12 initial birds and 8 additional birds, the total is 20 -/
theorem birds_on_fence : total_birds 12 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3875_387563


namespace NUMINAMATH_CALUDE_sequence_property_l3875_387572

/-- The sum of the first n terms of the sequence {a_n} -/
def S (a : ℕ+ → ℕ) (n : ℕ+) : ℕ := (Finset.range n.val).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem stating that if S_n = 2a_n - 2 for all n, then a_n = 2^n for all n -/
theorem sequence_property (a : ℕ+ → ℕ) 
    (h : ∀ n : ℕ+, S a n = 2 * a n - 2) : 
    ∀ n : ℕ+, a n = 2^n.val := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3875_387572


namespace NUMINAMATH_CALUDE_perry_vs_phil_l3875_387552

-- Define the number of games won by each player
def phil_games : ℕ := 12
def charlie_games : ℕ := phil_games - 3
def dana_games : ℕ := charlie_games + 2
def perry_games : ℕ := dana_games + 5

-- Theorem statement
theorem perry_vs_phil : perry_games = phil_games + 4 := by
  sorry

end NUMINAMATH_CALUDE_perry_vs_phil_l3875_387552


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3875_387553

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: For a geometric sequence where a_4 + a_8 = -2, 
    the value of a_6(a_2 + 2a_6 + a_10) is equal to 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : geometric_sequence a) 
    (h_sum : a 4 + a 8 = -2) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3875_387553
