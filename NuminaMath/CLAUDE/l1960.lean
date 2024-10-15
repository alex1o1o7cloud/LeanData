import Mathlib

namespace NUMINAMATH_CALUDE_gondor_earnings_l1960_196081

/-- Calculates the total earnings of a technician named Gondor based on his repair work --/
theorem gondor_earnings :
  let phone_repair_fee : ℕ := 10
  let laptop_repair_fee : ℕ := 20
  let phones_monday : ℕ := 3
  let phones_tuesday : ℕ := 5
  let laptops_wednesday : ℕ := 2
  let laptops_thursday : ℕ := 4
  
  let total_phones : ℕ := phones_monday + phones_tuesday
  let total_laptops : ℕ := laptops_wednesday + laptops_thursday
  
  let phone_earnings : ℕ := phone_repair_fee * total_phones
  let laptop_earnings : ℕ := laptop_repair_fee * total_laptops
  
  let total_earnings : ℕ := phone_earnings + laptop_earnings
  
  total_earnings = 200 :=
by
  sorry


end NUMINAMATH_CALUDE_gondor_earnings_l1960_196081


namespace NUMINAMATH_CALUDE_min_sum_geometric_sequence_l1960_196038

/-- Given a positive geometric sequence {a_n} where a₅ * a₄ * a₂ * a₁ = 16,
    the minimum value of a₁ + a₅ is 4. -/
theorem min_sum_geometric_sequence (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)
  (h_prod : a 5 * a 4 * a 2 * a 1 = 16) :
  ∀ x y, x > 0 ∧ y > 0 ∧ x * y = a 1 * a 5 → x + y ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_geometric_sequence_l1960_196038


namespace NUMINAMATH_CALUDE_not_term_of_sequence_l1960_196023

theorem not_term_of_sequence (n : ℕ+) : 25 - 2 * (n : ℤ) ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_term_of_sequence_l1960_196023


namespace NUMINAMATH_CALUDE_blackboard_area_difference_l1960_196060

/-- The difference between the area of a square with side length 8 cm
    and the area of a rectangle with sides 10 cm and 5 cm is 14 cm². -/
theorem blackboard_area_difference : 
  (8 : ℝ) * 8 - (10 : ℝ) * 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_blackboard_area_difference_l1960_196060


namespace NUMINAMATH_CALUDE_students_walking_home_l1960_196072

theorem students_walking_home (car_pickup : ℚ) (bus_ride : ℚ) (cycle_home : ℚ) 
  (h1 : car_pickup = 1/3)
  (h2 : bus_ride = 1/5)
  (h3 : cycle_home = 1/8)
  (h4 : car_pickup + bus_ride + cycle_home + (walk_home : ℚ) = 1) :
  walk_home = 41/120 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_l1960_196072


namespace NUMINAMATH_CALUDE_equation_solution_l1960_196057

theorem equation_solution : ∃ x : ℚ, x * (-1/2) = 1 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1960_196057


namespace NUMINAMATH_CALUDE_cube_of_sqrt_three_l1960_196024

theorem cube_of_sqrt_three (x : ℝ) (h : Real.sqrt (x - 3) = 3) : (x - 3)^3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_sqrt_three_l1960_196024


namespace NUMINAMATH_CALUDE_multiplication_scheme_solution_l1960_196067

def is_valid_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem multiplication_scheme_solution :
  ∀ (A B C D E F G H I K L M N P : ℕ),
    is_valid_digit A →
    is_valid_digit B →
    is_valid_digit C →
    is_valid_digit D →
    is_valid_digit E →
    is_valid_digit G →
    is_valid_digit H →
    is_valid_digit I →
    is_valid_digit K →
    is_valid_digit L →
    is_valid_digit N →
    is_valid_digit P →
    C = D →
    A = B →
    K = L →
    F = 0 →
    M = 0 →
    I = E →
    H = E →
    P = A →
    N = A →
    (A * 10 + B) * (C * 10 + D) = E * 100 + F * 10 + G →
    (C * 10 + G) * (K * 10 + L) = A * 100 + M * 10 + C →
    A = 7 ∧ B = 7 ∧ C = 4 ∧ D = 4 ∧ E = 3 ∧ G = 8 ∧ K = 8 ∧ L = 8 :=
by sorry

#check multiplication_scheme_solution

end NUMINAMATH_CALUDE_multiplication_scheme_solution_l1960_196067


namespace NUMINAMATH_CALUDE_recycling_program_earnings_l1960_196021

/-- Recycling program earnings calculation -/
theorem recycling_program_earnings 
  (initial_signup_bonus : ℕ)
  (referral_bonus : ℕ)
  (friend_signup_bonus : ℕ)
  (day_one_friends : ℕ)
  (week_end_friends : ℕ) :
  initial_signup_bonus = 5 →
  referral_bonus = 5 →
  friend_signup_bonus = 5 →
  day_one_friends = 5 →
  week_end_friends = 7 →
  (initial_signup_bonus + 
   (day_one_friends + week_end_friends) * (referral_bonus + friend_signup_bonus)) = 125 :=
by sorry

end NUMINAMATH_CALUDE_recycling_program_earnings_l1960_196021


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1960_196093

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 48) : 
  1 / x + 1 / y = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1960_196093


namespace NUMINAMATH_CALUDE_probability_theorem_l1960_196026

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

def probability_no_more_than_five_girls_between_first_last_boys : ℚ :=
  (Nat.choose 14 9 + 6 * Nat.choose 13 8) / Nat.choose 20 9

theorem probability_theorem :
  probability_no_more_than_five_girls_between_first_last_boys =
  (Nat.choose 14 9 + 6 * Nat.choose 13 8) / Nat.choose 20 9 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l1960_196026


namespace NUMINAMATH_CALUDE_persistent_is_two_l1960_196004

/-- A number T is persistent if for any a, b, c, d ≠ 0, 1:
    a + b + c + d = T and 1/a + 1/b + 1/c + 1/d = T implies 1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T -/
def IsPersistent (T : ℝ) : Prop :=
  ∀ a b c d : ℝ, a ≠ 0 → a ≠ 1 → b ≠ 0 → b ≠ 1 → c ≠ 0 → c ≠ 1 → d ≠ 0 → d ≠ 1 →
    (a + b + c + d = T ∧ 1/a + 1/b + 1/c + 1/d = T) →
    1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T

theorem persistent_is_two (T : ℝ) : IsPersistent T → T = 2 := by
  sorry

end NUMINAMATH_CALUDE_persistent_is_two_l1960_196004


namespace NUMINAMATH_CALUDE_line_through_points_l1960_196084

/-- Given a line with equation x = 5y + 5 passing through points (m, n) and (m + 2, n + p),
    prove that p = 2/5 -/
theorem line_through_points (m n : ℝ) : 
  let p : ℝ := 2/5
  m = 5*n + 5 ∧ (m + 2) = 5*(n + p) + 5 → p = 2/5 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_l1960_196084


namespace NUMINAMATH_CALUDE_min_value_of_cubic_l1960_196008

/-- The function f(x) = 2x³ + 3x² - 12x has a minimum value of -7. -/
theorem min_value_of_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 2 * x^3 + 3 * x^2 - 12 * x
  ∃ (min_x : ℝ), f min_x = -7 ∧ ∀ y, f y ≥ f min_x :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_cubic_l1960_196008


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1960_196039

theorem square_plus_reciprocal_square (x : ℝ) (h : x^4 + 1/x^4 = 23) : 
  x^2 + 1/x^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1960_196039


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1960_196069

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1960_196069


namespace NUMINAMATH_CALUDE_ends_with_1994_l1960_196047

theorem ends_with_1994 : ∃ n : ℕ+, 1994 * 1993^(n : ℕ) ≡ 1994 [MOD 10000] := by
  sorry

end NUMINAMATH_CALUDE_ends_with_1994_l1960_196047


namespace NUMINAMATH_CALUDE_fuel_consumption_l1960_196042

/-- Represents the fuel consumption of a heating plant -/
structure HeatingPlant where
  hours : ℝ
  fuel : ℝ
  rate : ℝ
  hRate : rate = fuel / hours

/-- Given a heating plant that consumes 7 liters of fuel in 21 hours,
    prove that it consumes 30 liters of fuel in 90 hours -/
theorem fuel_consumption (plant : HeatingPlant) 
  (h1 : plant.hours = 21) 
  (h2 : plant.fuel = 7) : 
  plant.rate * 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fuel_consumption_l1960_196042


namespace NUMINAMATH_CALUDE_loss_equals_sixteen_pencils_l1960_196027

/-- Represents a pencil transaction with a loss -/
structure PencilTransaction where
  quantity : ℕ
  costMultiplier : ℝ
  sellingPrice : ℝ

/-- Calculates the number of pencils whose selling price equals the total loss -/
def lossInPencils (t : PencilTransaction) : ℝ :=
  t.quantity * (t.costMultiplier - 1)

/-- Theorem stating that for the given transaction, the loss equals the selling price of 16 pencils -/
theorem loss_equals_sixteen_pencils (t : PencilTransaction) 
  (h1 : t.quantity = 80)
  (h2 : t.costMultiplier = 1.2) : 
  lossInPencils t = 16 := by
  sorry

#eval lossInPencils { quantity := 80, costMultiplier := 1.2, sellingPrice := 1 }

end NUMINAMATH_CALUDE_loss_equals_sixteen_pencils_l1960_196027


namespace NUMINAMATH_CALUDE_mother_age_now_is_70_l1960_196041

/-- Jessica's age now -/
def jessica_age_now : ℕ := 40

/-- Years passed since mother's death -/
def years_passed : ℕ := 10

/-- Jessica's age when her mother died -/
def jessica_age_then : ℕ := jessica_age_now - years_passed

/-- Mother's age when she died -/
def mother_age_then : ℕ := 2 * jessica_age_then

/-- Mother's age now if she were alive -/
def mother_age_now : ℕ := mother_age_then + years_passed

theorem mother_age_now_is_70 : mother_age_now = 70 := by
  sorry

end NUMINAMATH_CALUDE_mother_age_now_is_70_l1960_196041


namespace NUMINAMATH_CALUDE_xy_value_l1960_196089

theorem xy_value (x y : ℝ) (h : (x - 2)^2 + Real.sqrt (y + 2) = 0) : x * y = -4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1960_196089


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1960_196012

theorem polynomial_coefficient_sum : 
  ∀ (A B C D : ℝ), 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1960_196012


namespace NUMINAMATH_CALUDE_continuous_diff_function_properties_l1960_196075

/-- A function with a continuous derivative on ℝ -/
structure ContinuousDiffFunction where
  f : ℝ → ℝ
  f_continuous : Continuous f
  f_deriv : ℝ → ℝ
  f_deriv_continuous : Continuous f_deriv
  f_has_deriv : ∀ x, HasDerivAt f (f_deriv x) x

/-- The theorem statement -/
theorem continuous_diff_function_properties
  (f : ContinuousDiffFunction) (a b : ℝ) (hab : a < b)
  (h_deriv_a : f.f_deriv a > 0) (h_deriv_b : f.f_deriv b < 0) :
  (∃ x ∈ Set.Icc a b, f.f x > f.f b) ∧
  (∃ x ∈ Set.Icc a b, f.f a - f.f b > f.f_deriv x * (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_continuous_diff_function_properties_l1960_196075


namespace NUMINAMATH_CALUDE_missing_fraction_proof_l1960_196020

theorem missing_fraction_proof (total_sum : ℚ) (f1 f2 f3 f4 f5 f6 : ℚ) :
  total_sum = 0.13333333333333333 ∧
  f1 = 1/3 ∧ f2 = 1/2 ∧ f3 = -5/6 ∧ f4 = 1/5 ∧ f5 = -9/20 ∧ f6 = -2/15 →
  ∃ x : ℚ, x + f1 + f2 + f3 + f4 + f5 + f6 = total_sum ∧ x = 31/60 :=
by sorry

end NUMINAMATH_CALUDE_missing_fraction_proof_l1960_196020


namespace NUMINAMATH_CALUDE_find_unknown_number_l1960_196091

theorem find_unknown_number (known_numbers : List ℕ) (average : ℕ) : 
  known_numbers = [55, 507, 2, 684, 42] → 
  average = 223 → 
  ∃ x : ℕ, x = 48 ∧ (List.sum known_numbers + x) / 6 = average := by
sorry


end NUMINAMATH_CALUDE_find_unknown_number_l1960_196091


namespace NUMINAMATH_CALUDE_factorization_x4_minus_y4_l1960_196007

theorem factorization_x4_minus_y4 (x y : ℝ) : x^4 - y^4 = (x^2 + y^2) * (x^2 - y^2) := by sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_y4_l1960_196007


namespace NUMINAMATH_CALUDE_fraction_calculation_l1960_196030

theorem fraction_calculation : (3 / 10 : ℚ) + (5 / 100 : ℚ) - (2 / 1000 : ℚ) * (5 / 1 : ℚ) = (34 / 100 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1960_196030


namespace NUMINAMATH_CALUDE_k_range_for_equation_solution_l1960_196015

-- Define the custom operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem k_range_for_equation_solution :
  ∀ k : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    otimes 1 (2 * k - 3 - k * x₁) = 1 + Real.sqrt (4 - x₁^2) ∧
    otimes 1 (2 * k - 3 - k * x₂) = 1 + Real.sqrt (4 - x₂^2)) →
  k > 5/12 ∧ k ≤ 3/4 :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_equation_solution_l1960_196015


namespace NUMINAMATH_CALUDE_increasing_interval_ln_minus_x_l1960_196082

/-- The function f(x) = ln x - x is increasing on the interval (0,1] -/
theorem increasing_interval_ln_minus_x : 
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ ≤ 1 → 
  (Real.log x₁ - x₁) < (Real.log x₂ - x₂) := by
  sorry

end NUMINAMATH_CALUDE_increasing_interval_ln_minus_x_l1960_196082


namespace NUMINAMATH_CALUDE_jane_visited_six_more_l1960_196097

/-- The number of rehabilitation centers visited by each person --/
structure RehabCenters where
  lisa : ℕ
  jude : ℕ
  han : ℕ
  jane : ℕ

/-- The conditions of the problem --/
def problem_conditions (rc : RehabCenters) : Prop :=
  rc.lisa = 6 ∧
  rc.jude = rc.lisa / 2 ∧
  rc.han = 2 * rc.jude - 2 ∧
  rc.jane > 2 * rc.han ∧
  rc.lisa + rc.jude + rc.han + rc.jane = 27

/-- The theorem to be proved --/
theorem jane_visited_six_more (rc : RehabCenters) : 
  problem_conditions rc → rc.jane = 2 * rc.han + 6 := by
  sorry


end NUMINAMATH_CALUDE_jane_visited_six_more_l1960_196097


namespace NUMINAMATH_CALUDE_scouts_hike_car_occupancy_l1960_196049

theorem scouts_hike_car_occupancy (cars : ℕ) (taxis : ℕ) (vans : ℕ) 
  (people_per_taxi : ℕ) (people_per_van : ℕ) (total_people : ℕ) :
  cars = 3 →
  taxis = 6 →
  vans = 2 →
  people_per_taxi = 6 →
  people_per_van = 5 →
  total_people = 58 →
  ∃ (people_per_car : ℕ), 
    people_per_car * cars + people_per_taxi * taxis + people_per_van * vans = total_people ∧
    people_per_car = 4 :=
by sorry

end NUMINAMATH_CALUDE_scouts_hike_car_occupancy_l1960_196049


namespace NUMINAMATH_CALUDE_common_point_intersection_l1960_196005

/-- The common point of intersection for a family of lines -/
def common_point : ℝ × ℝ := (-1, 1)

/-- The equation of lines in the family -/
def line_equation (a b c x y : ℝ) : Prop := a * x + b * y = c

/-- The arithmetic progression condition -/
def arithmetic_progression (a b c d : ℝ) : Prop := b = a - d ∧ c = a - 2 * d

theorem common_point_intersection :
  ∀ (a b c d x y : ℝ),
    arithmetic_progression a b c d →
    (x, y) = common_point ↔ line_equation a b c x y :=
by sorry

end NUMINAMATH_CALUDE_common_point_intersection_l1960_196005


namespace NUMINAMATH_CALUDE_parabola_equation_l1960_196033

/-- A parabola with vertex at the origin, symmetric about the x-axis, 
    and a chord of length 8 passing through the focus and perpendicular 
    to the axis of symmetry has the equation y² = ±8x -/
theorem parabola_equation (p : Set (ℝ × ℝ)) 
  (vertex_at_origin : (0, 0) ∈ p)
  (symmetric_x_axis : ∀ (x y : ℝ), (x, y) ∈ p ↔ (x, -y) ∈ p)
  (focus_chord_length : ∃ (a : ℝ), a ≠ 0 ∧ 
    (Set.Icc (-a) a).image (λ y => (a/2, y)) ⊆ p ∧
    Set.Icc (-a) a = Set.Icc (-4) 4) :
  p = {(x, y) | y^2 = 8*x ∨ y^2 = -8*x} :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1960_196033


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1960_196079

theorem smallest_number_with_remainders : ∃! x : ℕ, 
  x ≥ 0 ∧ 
  x % 5 = 2 ∧ 
  x % 7 = 3 ∧ 
  x % 11 = 7 ∧
  ∀ y : ℕ, y ≥ 0 ∧ y % 5 = 2 ∧ y % 7 = 3 ∧ y % 11 = 7 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1960_196079


namespace NUMINAMATH_CALUDE_tape_winding_turns_l1960_196035

/-- Represents the parameters of the tape winding problem -/
structure TapeWindingParams where
  tape_length : ℝ  -- in mm
  tape_thickness : ℝ  -- in mm
  reel_diameter : ℝ  -- in mm

/-- Calculates the minimum number of turns needed to wind a tape onto a reel -/
def min_turns (params : TapeWindingParams) : ℕ :=
  sorry

/-- Theorem stating that for the given parameters, the minimum number of turns is 791 -/
theorem tape_winding_turns :
  let params : TapeWindingParams := {
    tape_length := 90000,  -- 90 m converted to mm
    tape_thickness := 0.018,
    reel_diameter := 22
  }
  min_turns params = 791 := by
  sorry

end NUMINAMATH_CALUDE_tape_winding_turns_l1960_196035


namespace NUMINAMATH_CALUDE_total_marks_math_physics_l1960_196080

/-- Given a student's marks in mathematics, physics, and chemistry, prove that
    the total marks in mathematics and physics is 60, under the given conditions. -/
theorem total_marks_math_physics (M P C : ℕ) : 
  C = P + 10 →  -- Chemistry score is 10 more than Physics
  (M + C) / 2 = 35 →  -- Average of Mathematics and Chemistry is 35
  M + P = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_math_physics_l1960_196080


namespace NUMINAMATH_CALUDE_total_bananas_l1960_196061

def banana_problem (dawn_bananas lydia_bananas donna_bananas : ℕ) : Prop :=
  lydia_bananas = 60 ∧
  dawn_bananas = lydia_bananas + 40 ∧
  donna_bananas = 40 ∧
  dawn_bananas + lydia_bananas + donna_bananas = 200

theorem total_bananas : ∃ dawn_bananas lydia_bananas donna_bananas : ℕ,
  banana_problem dawn_bananas lydia_bananas donna_bananas :=
by
  sorry

end NUMINAMATH_CALUDE_total_bananas_l1960_196061


namespace NUMINAMATH_CALUDE_middle_to_tallest_tree_ratio_l1960_196073

/-- Given three trees in a town square, prove the ratio of the middle height tree to the tallest tree -/
theorem middle_to_tallest_tree_ratio 
  (tallest_height : ℝ) 
  (shortest_height : ℝ) 
  (h_tallest : tallest_height = 150) 
  (h_shortest : shortest_height = 50) 
  (h_middle_relation : ∃ middle_height : ℝ, middle_height = 2 * shortest_height) :
  ∃ (middle_height : ℝ), middle_height / tallest_height = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_middle_to_tallest_tree_ratio_l1960_196073


namespace NUMINAMATH_CALUDE_four_r_applications_l1960_196063

def r (θ : ℚ) : ℚ := 1 / (1 - θ)

theorem four_r_applications : r (r (r (r 15))) = -1/14 := by
  sorry

end NUMINAMATH_CALUDE_four_r_applications_l1960_196063


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1960_196085

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum1 : a 2 + a 5 + a 8 = 27)
  (h_sum2 : a 3 + a 6 + a 9 = 33) :
  a 4 = 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1960_196085


namespace NUMINAMATH_CALUDE_product_equals_one_l1960_196036

theorem product_equals_one :
  (∀ a b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) →
  6 * 15 * 11 = 1 := by
sorry

end NUMINAMATH_CALUDE_product_equals_one_l1960_196036


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1960_196043

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1960_196043


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1005_l1960_196053

theorem largest_gcd_of_sum_1005 :
  ∃ (a b : ℕ+), a + b = 1005 ∧
  ∀ (c d : ℕ+), c + d = 1005 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 335 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1005_l1960_196053


namespace NUMINAMATH_CALUDE_equation_solution_l1960_196094

theorem equation_solution (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 - b*d) / (b + 2*c + d) + (b^2 - c*a) / (c + 2*d + a) + 
  (c^2 - d*b) / (d + 2*a + b) + (d^2 - a*c) / (a + 2*b + c) = 0 ↔ 
  a = c ∧ b = d := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1960_196094


namespace NUMINAMATH_CALUDE_prob_two_red_cards_l1960_196095

/-- Probability of drawing two red cards in succession from a special deck -/
theorem prob_two_red_cards (total_cards : Nat) (red_cards : Nat) 
  (h1 : total_cards = 60)
  (h2 : red_cards = 36) : 
  (red_cards * (red_cards - 1)) / (total_cards * (total_cards - 1)) = 21 / 59 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_cards_l1960_196095


namespace NUMINAMATH_CALUDE_four_digit_perfect_square_prefix_l1960_196001

theorem four_digit_perfect_square_prefix : ∃ (N : ℕ), 
  (1000 ≤ N ∧ N < 10000) ∧ 
  (∃ (k : ℕ), 4000000 + N = k^2) ∧
  (N = 4001 ∨ N = 8004) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_perfect_square_prefix_l1960_196001


namespace NUMINAMATH_CALUDE_quadratic_maximum_l1960_196083

theorem quadratic_maximum (s : ℝ) : -3 * s^2 + 24 * s - 8 ≤ 40 ∧ ∃ s, -3 * s^2 + 24 * s - 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l1960_196083


namespace NUMINAMATH_CALUDE_range_of_m_l1960_196058

/-- Given an increasing function f on ℝ, if f(2m) < f(9-m), then m < 3 -/
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inequality : f (2 * m) < f (9 - m)) : 
  m < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1960_196058


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l1960_196077

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ),
    (x^2 + a*x + b = 0 ↔ x = 2 - 3*I ∨ x = 2 + 3*I) ∧
    (a = -4 ∧ b = 13) := by
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l1960_196077


namespace NUMINAMATH_CALUDE_square_root_meaningful_l1960_196031

theorem square_root_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_meaningful_l1960_196031


namespace NUMINAMATH_CALUDE_entry_fee_reduction_l1960_196090

theorem entry_fee_reduction (original_fee : ℝ) (sale_increase : ℝ) (visitor_increase : ℝ) :
  original_fee = 1 ∧ 
  sale_increase = 0.2 ∧ 
  visitor_increase = 0.6 →
  ∃ (reduced_fee : ℝ),
    reduced_fee = 1 - 0.375 ∧
    (1 + visitor_increase) * reduced_fee * original_fee = (1 + sale_increase) * original_fee :=
by sorry

end NUMINAMATH_CALUDE_entry_fee_reduction_l1960_196090


namespace NUMINAMATH_CALUDE_common_ratio_values_l1960_196071

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * q
  third_term : a 3 = 2
  sum_second_fourth : a 2 + a 4 = 20 / 3
  q : ℝ

/-- The common ratio of the geometric sequence is either 3 or 1/3 -/
theorem common_ratio_values (seq : GeometricSequence) : seq.q = 3 ∨ seq.q = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_values_l1960_196071


namespace NUMINAMATH_CALUDE_composition_of_f_and_g_l1960_196048

-- Define the functions f and g
def f (A B : ℝ) (x : ℝ) : ℝ := A * x^2 - B^2
def g (B : ℝ) (x : ℝ) : ℝ := B * x + B^2

-- State the theorem
theorem composition_of_f_and_g (A B : ℝ) (h : B ≠ 0) :
  g B (f A B 1) = B * A - B^3 + B^2 := by
  sorry

end NUMINAMATH_CALUDE_composition_of_f_and_g_l1960_196048


namespace NUMINAMATH_CALUDE_round_trip_time_l1960_196098

/-- Calculates the total time for a round trip boat journey given the boat's speed in standing water,
    the stream's speed, and the distance to travel. -/
theorem round_trip_time (boat_speed : ℝ) (stream_speed : ℝ) (distance : ℝ) :
  boat_speed = 9 →
  stream_speed = 6 →
  distance = 210 →
  (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = 84 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_time_l1960_196098


namespace NUMINAMATH_CALUDE_christmas_presents_l1960_196034

theorem christmas_presents (birthday_presents christmas_presents : ℕ) : 
  christmas_presents = 2 * birthday_presents →
  christmas_presents + birthday_presents = 90 →
  christmas_presents = 60 := by
sorry

end NUMINAMATH_CALUDE_christmas_presents_l1960_196034


namespace NUMINAMATH_CALUDE_street_light_ratio_l1960_196050

theorem street_light_ratio (first_month : ℕ) (second_month : ℕ) (remaining : ℕ) :
  first_month = 1200 →
  second_month = 1300 →
  remaining = 500 →
  (first_month + second_month) / remaining = 5 := by
  sorry

end NUMINAMATH_CALUDE_street_light_ratio_l1960_196050


namespace NUMINAMATH_CALUDE_helmet_discount_percentage_l1960_196099

def original_price : ℝ := 40
def amount_saved : ℝ := 8
def amount_spent : ℝ := 32

theorem helmet_discount_percentage :
  (amount_saved / original_price) * 100 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_helmet_discount_percentage_l1960_196099


namespace NUMINAMATH_CALUDE_base9_perfect_square_last_digit_l1960_196025

/-- Represents a number in base 9 of the form ab5d -/
structure Base9Number where
  a : ℕ
  b : ℕ
  d : ℕ
  a_nonzero : a ≠ 0
  b_range : b < 9
  d_range : d < 9

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 45 + n.d

/-- Predicate to check if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem base9_perfect_square_last_digit
  (n : Base9Number)
  (h : isPerfectSquare (toDecimal n)) :
  n.d = 0 := by
  sorry

#check base9_perfect_square_last_digit

end NUMINAMATH_CALUDE_base9_perfect_square_last_digit_l1960_196025


namespace NUMINAMATH_CALUDE_kanul_total_amount_l1960_196019

theorem kanul_total_amount 
  (raw_materials : ℝ) 
  (machinery : ℝ) 
  (cash_percentage : ℝ) 
  (total : ℝ) :
  raw_materials = 80000 →
  machinery = 30000 →
  cash_percentage = 0.20 →
  raw_materials + machinery + cash_percentage * total = total →
  total = 137500 :=
by sorry

end NUMINAMATH_CALUDE_kanul_total_amount_l1960_196019


namespace NUMINAMATH_CALUDE_messenger_speed_l1960_196062

/-- Proves that the messenger's speed is 25 km/h given the problem conditions -/
theorem messenger_speed (team_length : ℝ) (team_speed : ℝ) (journey_time : ℝ)
  (h1 : team_length = 6)
  (h2 : team_speed = 5)
  (h3 : journey_time = 0.5)
  (h4 : ∀ x : ℝ, x > team_speed → team_length / (x + team_speed) + team_length / (x - team_speed) = journey_time → x = 25) :
  ∃ x : ℝ, x > team_speed ∧ team_length / (x + team_speed) + team_length / (x - team_speed) = journey_time ∧ x = 25 :=
by sorry

end NUMINAMATH_CALUDE_messenger_speed_l1960_196062


namespace NUMINAMATH_CALUDE_dans_marbles_l1960_196018

theorem dans_marbles (dans_marbles mary_marbles : ℕ) 
  (h1 : mary_marbles = 2 * dans_marbles)
  (h2 : mary_marbles = 10) : 
  dans_marbles = 5 := by
sorry

end NUMINAMATH_CALUDE_dans_marbles_l1960_196018


namespace NUMINAMATH_CALUDE_geometric_series_example_l1960_196065

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  (a - a * r^n) / (1 - r)

theorem geometric_series_example : 
  let a : ℚ := 1/5
  let r : ℚ := -1/5
  let n : ℕ := 6
  geometric_series_sum a r n = 1562/9375 := by sorry

end NUMINAMATH_CALUDE_geometric_series_example_l1960_196065


namespace NUMINAMATH_CALUDE_hit_at_least_once_miss_both_times_mutually_exclusive_hit_at_least_once_miss_both_times_complementary_l1960_196045

-- Define the sample space
def Ω : Type := Unit

-- Define the event of hitting the target at least once
def hit_at_least_once : Set Ω := sorry

-- Define the event of missing the target both times
def miss_both_times : Set Ω := sorry

-- Theorem: hit_at_least_once and miss_both_times are mutually exclusive
theorem hit_at_least_once_miss_both_times_mutually_exclusive :
  hit_at_least_once ∩ miss_both_times = ∅ :=
sorry

-- Theorem: hit_at_least_once and miss_both_times are complementary
theorem hit_at_least_once_miss_both_times_complementary :
  hit_at_least_once ∪ miss_both_times = Set.univ :=
sorry

end NUMINAMATH_CALUDE_hit_at_least_once_miss_both_times_mutually_exclusive_hit_at_least_once_miss_both_times_complementary_l1960_196045


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1960_196086

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1960_196086


namespace NUMINAMATH_CALUDE_bookstore_travel_options_l1960_196009

theorem bookstore_travel_options (bus_ways subway_ways : ℕ) 
  (h1 : bus_ways = 3) 
  (h2 : subway_ways = 4) : 
  bus_ways + subway_ways = 7 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_travel_options_l1960_196009


namespace NUMINAMATH_CALUDE_star_value_of_a_l1960_196052

-- Define the star operation
def star (a b : ℝ) : ℝ := 3 * a - b^3

-- Theorem statement
theorem star_value_of_a :
  ∀ a : ℝ, star a 3 = 18 → a = 15 := by
  sorry

end NUMINAMATH_CALUDE_star_value_of_a_l1960_196052


namespace NUMINAMATH_CALUDE_tiller_swath_width_l1960_196076

/-- Calculates the swath width of a tiller given plot dimensions, tilling rate, and total tilling time -/
theorem tiller_swath_width
  (plot_width : ℝ)
  (plot_length : ℝ)
  (tilling_rate : ℝ)
  (total_time : ℝ)
  (h1 : plot_width = 110)
  (h2 : plot_length = 120)
  (h3 : tilling_rate = 2)  -- 2 seconds per foot
  (h4 : total_time = 220 * 60)  -- 220 minutes in seconds
  : (plot_width * plot_length) / (total_time / tilling_rate) = 2 := by
  sorry

#check tiller_swath_width

end NUMINAMATH_CALUDE_tiller_swath_width_l1960_196076


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1960_196044

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property :
  ∀ a : ℕ → ℝ,
  (is_geometric_sequence a → ∀ n : ℕ, a n ^ 2 = a (n - 1) * a (n + 1)) ∧
  (∃ a : ℕ → ℝ, (∀ n : ℕ, a n ^ 2 = a (n - 1) * a (n + 1)) ∧ ¬is_geometric_sequence a) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1960_196044


namespace NUMINAMATH_CALUDE_solution_x_l1960_196092

theorem solution_x (x y : ℝ) 
  (h1 : (2010 + x)^2 = x^2) 
  (h2 : x = 5*y + 2) : 
  x = -1005 := by
sorry

end NUMINAMATH_CALUDE_solution_x_l1960_196092


namespace NUMINAMATH_CALUDE_cubic_root_difference_l1960_196046

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + a

/-- The derivative of the cubic function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

theorem cubic_root_difference (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  (∃! (x₁ x₂ : ℝ), f a x₁ = 0 ∧ f a x₂ = 0) →
  x₂ - x₁ = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_difference_l1960_196046


namespace NUMINAMATH_CALUDE_same_terminal_side_l1960_196014

theorem same_terminal_side (k : ℤ) : 
  (2 * k * π + π / 5 : ℝ) = 21 * π / 5 → 
  ∃ n : ℤ, (21 * π / 5 : ℝ) = 2 * n * π + π / 5 :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_l1960_196014


namespace NUMINAMATH_CALUDE_positive_root_of_cubic_l1960_196013

theorem positive_root_of_cubic (x : ℝ) : 
  x = 3 - Real.sqrt 3 → x > 0 ∧ x^3 - 4*x^2 - 2*x - Real.sqrt 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_root_of_cubic_l1960_196013


namespace NUMINAMATH_CALUDE_equation_solution_l1960_196002

theorem equation_solution (y : ℝ) : (24 / 36 : ℝ) = Real.sqrt (y / 36) → y = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1960_196002


namespace NUMINAMATH_CALUDE_julia_shortage_l1960_196006

def rock_price : ℕ := 5
def pop_price : ℕ := 10
def dance_price : ℕ := 3
def country_price : ℕ := 7
def quantity : ℕ := 4
def budget : ℕ := 75

def total_cost : ℕ := rock_price * quantity + pop_price * quantity + 
                      dance_price * quantity + country_price * quantity

theorem julia_shortage : total_cost - budget = 25 := by
  sorry

end NUMINAMATH_CALUDE_julia_shortage_l1960_196006


namespace NUMINAMATH_CALUDE_min_a_for_no_zeros_l1960_196010

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * log x

noncomputable def g (x : ℝ) : ℝ := x * exp (1 - x)

theorem min_a_for_no_zeros (a : ℝ) :
  (∀ x, 0 < x ∧ x < 1/2 → f a x > 0) ↔ a ≥ 2 - 4 * log 2 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_no_zeros_l1960_196010


namespace NUMINAMATH_CALUDE_min_words_for_certification_l1960_196074

theorem min_words_for_certification (total_words : ℕ) (min_score : ℚ) : 
  total_words = 800 → 
  min_score = 9/10 → 
  ∃ (words_to_learn : ℕ), 
    (words_to_learn : ℚ) / total_words ≥ min_score ∧ 
    ∀ (w : ℕ), (w : ℚ) / total_words ≥ min_score → w ≥ words_to_learn ∧
    words_to_learn = 720 := by
  sorry

end NUMINAMATH_CALUDE_min_words_for_certification_l1960_196074


namespace NUMINAMATH_CALUDE_function_bounds_l1960_196068

/-- Given a function f(θ) = 1 - a cos θ - b sin θ - A cos 2θ - B sin 2θ that is non-negative for all real θ,
    prove that a² + b² ≤ 2 and A² + B² ≤ 1 -/
theorem function_bounds (a b A B : ℝ) 
  (h : ∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_bounds_l1960_196068


namespace NUMINAMATH_CALUDE_vector_expression_equality_l1960_196055

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_expression_equality (a b : V) :
  (1 / 3 : ℝ) • ((1 / 2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b)) = 2 • b - a :=
sorry

end NUMINAMATH_CALUDE_vector_expression_equality_l1960_196055


namespace NUMINAMATH_CALUDE_bob_cycling_wins_l1960_196032

/-- The number of weeks Bob has already won -/
def initial_wins : ℕ := 2

/-- The cost of the puppy in dollars -/
def puppy_cost : ℕ := 1000

/-- The additional number of wins Bob needs to afford the puppy -/
def additional_wins_needed : ℕ := 8

/-- The prize money Bob wins each week -/
def weekly_prize : ℚ := puppy_cost / (initial_wins + additional_wins_needed)

theorem bob_cycling_wins :
  ∀ (weeks : ℕ),
    weekly_prize * (initial_wins + weeks) ≥ puppy_cost →
    weeks ≥ additional_wins_needed :=
by
  sorry

end NUMINAMATH_CALUDE_bob_cycling_wins_l1960_196032


namespace NUMINAMATH_CALUDE_man_son_age_difference_man_son_age_difference_proof_l1960_196059

theorem man_son_age_difference : ℕ → ℕ → Prop :=
  fun man_age son_age =>
    son_age = 22 →
    man_age + 2 = 2 * (son_age + 2) →
    man_age - son_age = 24

-- Proof
theorem man_son_age_difference_proof :
  ∃ (man_age son_age : ℕ), man_son_age_difference man_age son_age := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_difference_man_son_age_difference_proof_l1960_196059


namespace NUMINAMATH_CALUDE_monotonic_intervals_range_of_a_l1960_196054

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (1/2) * a * x^2

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x

-- Theorem for Part I
theorem monotonic_intervals (a : ℝ) (h : a ≤ 1) :
  (∀ x < 0, a ≤ 0 → (f' a x < 0)) ∧
  (∀ x > 0, a ≤ 0 → (f' a x > 0)) ∧
  (∀ x < Real.log a, 0 < a → a < 1 → (f' a x > 0)) ∧
  (∀ x ∈ Set.Ioo (Real.log a) 0, 0 < a → a < 1 → (f' a x < 0)) ∧
  (∀ x > 0, 0 < a → a < 1 → (f' a x > 0)) ∧
  (∀ x : ℝ, a = 1 → (f' a x ≥ 0)) :=
sorry

-- Theorem for Part II
theorem range_of_a (a : ℝ) :
  (∀ x > 0, f' a x > a * x^3 + x^2 - (a - 1) * x) ↔ a ∈ Set.Iic (1/2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_intervals_range_of_a_l1960_196054


namespace NUMINAMATH_CALUDE_divisibility_by_five_l1960_196051

theorem divisibility_by_five (n : ℤ) : 
  ∃ (m : ℤ), 3 * (n^2 + n) + 7 = 5 * m ↔ ∃ (k : ℤ), n = 5 * k + 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l1960_196051


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1960_196078

theorem absolute_value_equality (a b : ℝ) : 
  |a| = |b| → (a = b ∨ a = -b) := by sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1960_196078


namespace NUMINAMATH_CALUDE_product_equals_143_l1960_196029

-- Define a function to convert binary to decimal
def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

-- Define a function to convert ternary to decimal
def ternary_to_decimal (t : List Nat) : Nat :=
  t.enum.foldr (fun (i, digit) acc => acc + digit * 3^i) 0

-- Define the binary number 1101₂
def binary_1101 : List Bool := [true, false, true, true]

-- Define the ternary number 102₃
def ternary_102 : List Nat := [2, 0, 1]

theorem product_equals_143 : 
  (binary_to_decimal binary_1101) * (ternary_to_decimal ternary_102) = 143 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_143_l1960_196029


namespace NUMINAMATH_CALUDE_x_range_l1960_196003

theorem x_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ x : ℝ, x^2 + 2*x < a/b + 16*b/a → -4 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_x_range_l1960_196003


namespace NUMINAMATH_CALUDE_power_three_twenty_mod_five_l1960_196028

theorem power_three_twenty_mod_five : 3^20 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_twenty_mod_five_l1960_196028


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l1960_196016

theorem least_sum_of_bases (c d : ℕ+) (h : 3 * c.val + 8 = 8 * d.val + 3) :
  ∃ (c' d' : ℕ+), 3 * c'.val + 8 = 8 * d'.val + 3 ∧ c'.val + d'.val ≤ c.val + d.val ∧ c'.val + d'.val = 13 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l1960_196016


namespace NUMINAMATH_CALUDE_polynomial_factor_problem_l1960_196000

theorem polynomial_factor_problem (b c : ℤ) :
  let p : ℝ → ℝ := fun x ↦ x^2 + b*x + c
  (∃ q : ℝ → ℝ, (fun x ↦ x^4 + 8*x^2 + 49) = fun x ↦ p x * q x) ∧
  (∃ r : ℝ → ℝ, (fun x ↦ 2*x^4 + 5*x^2 + 32*x + 8) = fun x ↦ p x * r x) →
  p 1 = 24 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_problem_l1960_196000


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1960_196022

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1960_196022


namespace NUMINAMATH_CALUDE_no_valid_labeling_exists_l1960_196070

/-- Represents a simple undirected graph with 6 vertices -/
structure Graph :=
  (edges : Set (Fin 6 × Fin 6))
  (symmetric : ∀ (a b : Fin 6), (a, b) ∈ edges → (b, a) ∈ edges)
  (irreflexive : ∀ (a : Fin 6), (a, a) ∉ edges)

/-- A function assigning natural numbers to vertices -/
def VertexLabeling := Fin 6 → ℕ+

/-- Checks if the labeling satisfies the divisibility condition for the given graph -/
def ValidLabeling (g : Graph) (f : VertexLabeling) : Prop :=
  (∀ (a b : Fin 6), (a, b) ∈ g.edges → (f a ∣ f b) ∨ (f b ∣ f a)) ∧
  (∀ (a b : Fin 6), a ≠ b → (a, b) ∉ g.edges → ¬(f a ∣ f b) ∧ ¬(f b ∣ f a))

/-- The main theorem stating that no valid labeling exists for any graph with 6 vertices -/
theorem no_valid_labeling_exists : ∀ (g : Graph), ¬∃ (f : VertexLabeling), ValidLabeling g f := by
  sorry

end NUMINAMATH_CALUDE_no_valid_labeling_exists_l1960_196070


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l1960_196017

theorem simplify_sqrt_difference : (Real.sqrt 882 / Real.sqrt 98) - (Real.sqrt 108 / Real.sqrt 12) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l1960_196017


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1960_196037

theorem sum_of_a_and_b (a b : ℕ+) (h : 143 * a + 500 * b = 2001) : a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1960_196037


namespace NUMINAMATH_CALUDE_smallest_n_value_l1960_196066

theorem smallest_n_value (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ (21 * N)) : 80 ≤ N := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1960_196066


namespace NUMINAMATH_CALUDE_inequalities_hold_l1960_196064

theorem inequalities_hold (x y z a b c : ℕ+) 
  (hx : x ≤ a) (hy : y ≤ b) (hz : z ≤ c) : 
  (x^2 * y^2 + y^2 * z^2 + z^2 * x^2 ≤ a^2 * b^2 + b^2 * c^2 + c^2 * a^2) ∧ 
  (x^3 + y^3 + z^3 ≤ a^3 + b^3 + c^3) ∧ 
  (x^2 * y * z + y^2 * z * x + z^2 * x * y ≤ a^2 * b * c + b^2 * c * a + c^2 * a * b) :=
by sorry

#check inequalities_hold

end NUMINAMATH_CALUDE_inequalities_hold_l1960_196064


namespace NUMINAMATH_CALUDE_pentagon_area_difference_l1960_196011

/-- Given a rectangle with dimensions 48 mm and 64 mm, when folded along its diagonal
    to form a pentagon, the area difference between the original rectangle and
    the resulting pentagon is 1200 mm². -/
theorem pentagon_area_difference (a b : ℝ) (ha : a = 48) (hb : b = 64) :
  let rect_area := a * b
  let diag := Real.sqrt (a^2 + b^2)
  let overlap_height := Real.sqrt ((diag/2)^2 - ((b - (b^2 - a^2) / (2 * b))^2))
  let overlap_area := (1/2) * diag * overlap_height
  rect_area - (rect_area - overlap_area) = 1200 :=
by sorry


end NUMINAMATH_CALUDE_pentagon_area_difference_l1960_196011


namespace NUMINAMATH_CALUDE_f_positivity_and_extrema_l1960_196056

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (2 * x^2 - 3 * x)

theorem f_positivity_and_extrema :
  (∀ x : ℝ, f x > 0 ↔ x < 0 ∨ x > 3/2) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f x ≤ 2 * Real.exp 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f x ≥ -Real.exp 1) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x = 2 * Real.exp 2) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x = -Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_f_positivity_and_extrema_l1960_196056


namespace NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l1960_196088

/-- The shortest distance between the point (3,6) and the parabola x = y^2/4 is √5. -/
theorem shortest_distance_point_to_parabola :
  let point := (3, 6)
  let parabola := {(x, y) : ℝ × ℝ | x = y^2 / 4}
  (∃ (d : ℝ), d = Real.sqrt 5 ∧
    ∀ (p : ℝ × ℝ), p ∈ parabola →
      d ≤ Real.sqrt ((p.1 - point.1)^2 + (p.2 - point.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l1960_196088


namespace NUMINAMATH_CALUDE_bela_has_winning_strategy_l1960_196096

/-- Represents a stick with a certain length -/
structure Stick :=
  (length : ℕ)

/-- Represents the game state -/
structure GameState :=
  (sticks : List Stick)

/-- Checks if three sticks can form a triangle -/
def canFormTriangle (s1 s2 s3 : Stick) : Prop :=
  s1.length + s2.length > s3.length ∧
  s1.length + s3.length > s2.length ∧
  s2.length + s3.length > s1.length

/-- Represents a player's strategy -/
def Strategy := GameState → Stick

/-- Represents the initial game state with 99 sticks -/
def initialGameState : GameState :=
  { sticks := List.range 99 |>.map (fun n => ⟨n + 1⟩) }

/-- Represents Béla's winning strategy -/
noncomputable def belasStrategy : Strategy :=
  sorry

/-- Theorem stating that Béla has a winning strategy -/
theorem bela_has_winning_strategy :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      game.sticks.length = 3 →
      ¬(∃ (s1 s2 s3 : Stick), s1 ∈ game.sticks ∧ s2 ∈ game.sticks ∧ s3 ∈ game.sticks ∧ canFormTriangle s1 s2 s3) :=
by
  sorry

end NUMINAMATH_CALUDE_bela_has_winning_strategy_l1960_196096


namespace NUMINAMATH_CALUDE_power_of_power_equals_six_l1960_196040

theorem power_of_power_equals_six (m : ℝ) : (m^2)^3 = m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_equals_six_l1960_196040


namespace NUMINAMATH_CALUDE_new_average_weight_l1960_196087

theorem new_average_weight (weight_A weight_D : ℝ) : 
  weight_A = 73 →
  (weight_A + (150 - weight_A)) / 3 = 50 →
  ((150 - weight_A) + weight_D + (weight_D + 3)) / 4 = 51 →
  (weight_A + (150 - weight_A) + weight_D) / 4 = 53 :=
by sorry

end NUMINAMATH_CALUDE_new_average_weight_l1960_196087
