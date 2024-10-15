import Mathlib

namespace NUMINAMATH_CALUDE_line_slope_equation_l1362_136218

/-- Given a line passing through points (-1, -4) and (3, k), where the slope
    of the line is equal to k, prove that k = 4/3 -/
theorem line_slope_equation (k : ℝ) : 
  (let x₁ : ℝ := -1
   let y₁ : ℝ := -4
   let x₂ : ℝ := 3
   let y₂ : ℝ := k
   let slope : ℝ := (y₂ - y₁) / (x₂ - x₁)
   slope = k) → k = 4/3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_equation_l1362_136218


namespace NUMINAMATH_CALUDE_emma_money_theorem_l1362_136297

def emma_money_problem (initial_amount furniture_cost fraction_to_anna : ℚ) : Prop :=
  let remaining_after_furniture := initial_amount - furniture_cost
  let amount_to_anna := fraction_to_anna * remaining_after_furniture
  let final_amount := remaining_after_furniture - amount_to_anna
  final_amount = 400

theorem emma_money_theorem :
  emma_money_problem 2000 400 (3/4) := by
  sorry

end NUMINAMATH_CALUDE_emma_money_theorem_l1362_136297


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l1362_136211

/-- Proves that the ratio of upstream to downstream rowing time is 2:1 given specific speeds -/
theorem upstream_downstream_time_ratio 
  (man_speed : ℝ) 
  (current_speed : ℝ) 
  (h1 : man_speed = 4.5)
  (h2 : current_speed = 1.5) : 
  (man_speed - current_speed) / (man_speed + current_speed) = 1 / 2 := by
  sorry

#check upstream_downstream_time_ratio

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l1362_136211


namespace NUMINAMATH_CALUDE_total_tylenol_grams_l1362_136221

-- Define the parameters
def tablet_count : ℕ := 2
def tablet_mg : ℕ := 500
def hours_between_doses : ℕ := 4
def total_hours : ℕ := 12
def mg_per_gram : ℕ := 1000

-- Theorem statement
theorem total_tylenol_grams : 
  (total_hours / hours_between_doses) * tablet_count * tablet_mg / mg_per_gram = 3 := by
  sorry

end NUMINAMATH_CALUDE_total_tylenol_grams_l1362_136221


namespace NUMINAMATH_CALUDE_subtraction_from_percentage_l1362_136227

theorem subtraction_from_percentage (x : ℝ) : x = 100 → (0.7 * x - 40 = 30) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_from_percentage_l1362_136227


namespace NUMINAMATH_CALUDE_simplify_fraction_l1362_136278

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (4 * x) / (x^2 - 4) - 2 / (x - 2) - 1 = -x / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1362_136278


namespace NUMINAMATH_CALUDE_rook_removal_theorem_l1362_136250

/-- Represents a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Checks if a rook at position (x, y) attacks a square (i, j) -/
def attacks (x y i j : Fin 8) : Bool :=
  x = i ∨ y = j

/-- A configuration of rooks on a chessboard -/
def RookConfiguration := Fin 20 → Fin 8 × Fin 8

/-- Checks if a configuration of rooks attacks the entire board -/
def attacks_all_squares (config : RookConfiguration) : Prop :=
  ∀ i j, ∃ k, attacks (config k).1 (config k).2 i j

/-- Represents a subset of 8 rooks from the original 20 -/
def Subset := Fin 8 → Fin 20

theorem rook_removal_theorem (initial_config : RookConfiguration) 
  (h : attacks_all_squares initial_config) :
  ∃ (subset : Subset), attacks_all_squares (λ i => initial_config (subset i)) :=
sorry

end NUMINAMATH_CALUDE_rook_removal_theorem_l1362_136250


namespace NUMINAMATH_CALUDE_evaluate_expression_at_negative_one_l1362_136208

-- Define the expression as a function of x
def f (x : ℚ) : ℚ := (4 + x * (4 + x) - 4^2) / (x - 4 + x^3)

-- State the theorem
theorem evaluate_expression_at_negative_one :
  f (-1) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_at_negative_one_l1362_136208


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1362_136276

/-- Prove that given two trains of equal length, where the faster train travels at a given speed
    and passes the slower train in a given time, the speed of the slower train can be calculated. -/
theorem train_speed_calculation (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 65 →
  faster_speed = 49 →
  passing_time = 36 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    2 * train_length = (faster_speed - slower_speed) * (5 / 18) * passing_time :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1362_136276


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1362_136256

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) 
  (h2 : x * y = -8) : 
  x^2 + y^2 = 33 := by sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1362_136256


namespace NUMINAMATH_CALUDE_fish_cost_per_kg_proof_l1362_136263

-- Define the constants
def total_cost_case1 : ℕ := 530
def fish_kg_case1 : ℕ := 4
def pork_kg_case1 : ℕ := 2
def pork_kg_case2 : ℕ := 3
def total_cost_case2 : ℕ := 875
def fish_cost_per_kg : ℕ := 80

-- Define the theorem
theorem fish_cost_per_kg_proof :
  let pork_cost_case1 := total_cost_case1 - fish_cost_per_kg * fish_kg_case1
  let pork_cost_per_kg := pork_cost_case1 / pork_kg_case1
  let pork_cost_case2 := pork_cost_per_kg * pork_kg_case2
  let fish_cost_case2 := total_cost_case2 - pork_cost_case2
  fish_cost_case2 / (fish_cost_case2 / fish_cost_per_kg) = fish_cost_per_kg :=
by
  sorry

#check fish_cost_per_kg_proof

end NUMINAMATH_CALUDE_fish_cost_per_kg_proof_l1362_136263


namespace NUMINAMATH_CALUDE_hotdog_price_l1362_136247

/-- The cost of a hamburger -/
def hamburger_cost : ℝ := sorry

/-- The cost of a hot dog -/
def hotdog_cost : ℝ := sorry

/-- First day's purchase equation -/
axiom first_day : 3 * hamburger_cost + 4 * hotdog_cost = 10

/-- Second day's purchase equation -/
axiom second_day : 2 * hamburger_cost + 3 * hotdog_cost = 7

/-- Theorem stating that a hot dog costs 1 dollar -/
theorem hotdog_price : hotdog_cost = 1 := by sorry

end NUMINAMATH_CALUDE_hotdog_price_l1362_136247


namespace NUMINAMATH_CALUDE_min_route_length_5x5_city_l1362_136269

/-- Represents a square grid city -/
structure SquareGridCity where
  size : Nat

/-- Represents a route in the city -/
structure CityRoute where
  length : Nat
  covers_all_streets : Bool
  returns_to_start : Bool

/-- The minimum length of a route that covers all streets and returns to the starting point -/
def min_route_length (city : SquareGridCity) : Nat :=
  sorry

theorem min_route_length_5x5_city :
  ∀ (city : SquareGridCity) (route : CityRoute),
    city.size = 5 →
    route.covers_all_streets = true →
    route.returns_to_start = true →
    route.length ≥ min_route_length city →
    min_route_length city = 68 :=
by sorry

end NUMINAMATH_CALUDE_min_route_length_5x5_city_l1362_136269


namespace NUMINAMATH_CALUDE_fraction_problem_l1362_136285

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4) * F * (2/5) * N = 15)
  (h2 : (40/100) * N = 180) : 
  F = 2/3 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1362_136285


namespace NUMINAMATH_CALUDE_pages_left_to_read_l1362_136243

theorem pages_left_to_read 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (pages_to_skip : ℕ) 
  (h1 : total_pages = 372) 
  (h2 : pages_read = 125) 
  (h3 : pages_to_skip = 16) :
  total_pages - (pages_read + pages_to_skip) = 231 :=
by sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l1362_136243


namespace NUMINAMATH_CALUDE_power_subtraction_equals_6444_l1362_136224

theorem power_subtraction_equals_6444 : 3^(1+3+4) - (3^1 * 3 + 3^3 + 3^4) = 6444 := by
  sorry

end NUMINAMATH_CALUDE_power_subtraction_equals_6444_l1362_136224


namespace NUMINAMATH_CALUDE_distance_to_SFL_is_81_l1362_136242

/-- The distance to Super Fun-tastic Land -/
def distance_to_SFL (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that the distance to Super Fun-tastic Land is 81 miles -/
theorem distance_to_SFL_is_81 :
  distance_to_SFL 27 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_SFL_is_81_l1362_136242


namespace NUMINAMATH_CALUDE_certain_number_equation_l1362_136261

theorem certain_number_equation (x : ℝ) : 5 * 1.6 - (2 * 1.4) / x = 4 ↔ x = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l1362_136261


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1362_136298

theorem quadratic_root_relation (p : ℝ) : 
  (∃ a : ℝ, a ≠ 0 ∧ (a^2 + p*a + 18 = 0) ∧ ((2*a)^2 + p*(2*a) + 18 = 0)) ↔ 
  (p = 9 ∨ p = -9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1362_136298


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l1362_136262

theorem cube_sum_magnitude (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 15) :
  Complex.abs (w^3 + z^3) = 41 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l1362_136262


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1362_136232

theorem right_rectangular_prism_volume
  (x y z : ℝ)
  (h_side : x * y = 15)
  (h_front : y * z = 10)
  (h_bottom : x * z = 6) :
  x * y * z = 30 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1362_136232


namespace NUMINAMATH_CALUDE_f_extrema_l1362_136290

def f (x : ℝ) := 3 * x^4 - 6 * x^2 + 4

theorem f_extrema :
  (∀ x ∈ Set.Icc (-1) 3, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 1) ∧
  (∀ x ∈ Set.Icc (-1) 3, f x ≤ 193) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 193) := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l1362_136290


namespace NUMINAMATH_CALUDE_number_of_observations_l1362_136259

/-- Given a set of observations with an initial mean, a correction to one observation,
    and a new mean, prove the number of observations. -/
theorem number_of_observations
  (initial_mean : ℝ)
  (wrong_value : ℝ)
  (correct_value : ℝ)
  (new_mean : ℝ)
  (h1 : initial_mean = 36)
  (h2 : wrong_value = 23)
  (h3 : correct_value = 45)
  (h4 : new_mean = 36.5) :
  ∃ n : ℕ, n * initial_mean + (correct_value - wrong_value) = n * new_mean ∧ n = 44 := by
  sorry

end NUMINAMATH_CALUDE_number_of_observations_l1362_136259


namespace NUMINAMATH_CALUDE_sum_remainder_thirteen_l1362_136282

theorem sum_remainder_thirteen : ∃ k : ℕ, (5000 + 5001 + 5002 + 5003 + 5004 + 5005 + 5006) = 13 * k + 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_thirteen_l1362_136282


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1362_136299

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (5 * x₁^2 + 20 * x₁ - 25 = 0) →
  (5 * x₂^2 + 20 * x₂ - 25 = 0) →
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 26 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1362_136299


namespace NUMINAMATH_CALUDE_function_transformation_l1362_136266

/-- Given a function f such that f(1/x) = x/(1-x) for all x ≠ 0 and x ≠ 1,
    prove that f(x) = 1/(x-1) for all x ≠ 0 and x ≠ 1. -/
theorem function_transformation (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f (1/x) = x / (1 - x)) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f x = 1 / (x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_transformation_l1362_136266


namespace NUMINAMATH_CALUDE_intersection_tangents_perpendicular_l1362_136229

/-- Two circles in the plane -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0

def circle2 (a x y : ℝ) : Prop := x^2 + y^2 + 2*(a-1)*x + 2*y + a^2 = 0

/-- The common chord of the two circles -/
def common_chord (a x y : ℝ) : Prop := 2*(a-1)*x - 2*y + a^2 = 0

/-- The condition for perpendicular tangents at intersection points -/
def perpendicular_tangents (a x y : ℝ) : Prop :=
  (y + 2) / x * (y + 1) / (x - (1 - a)) = -1

/-- The main theorem -/
theorem intersection_tangents_perpendicular (a : ℝ) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 a x y ∧ common_chord a x y ∧ perpendicular_tangents a x y) →
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_intersection_tangents_perpendicular_l1362_136229


namespace NUMINAMATH_CALUDE_two_solutions_sine_equation_l1362_136226

theorem two_solutions_sine_equation (x : ℝ) (a : ℝ) : 
  (x ∈ Set.Icc 0 Real.pi) →
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁ ∈ Set.Icc 0 Real.pi ∧ 
    x₂ ∈ Set.Icc 0 Real.pi ∧
    2 * Real.sin (x₁ + Real.pi / 3) = a ∧ 
    2 * Real.sin (x₂ + Real.pi / 3) = a) ↔
  (a > Real.sqrt 3 ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_two_solutions_sine_equation_l1362_136226


namespace NUMINAMATH_CALUDE_dress_designs_count_l1362_136213

/-- The number of fabric colors available -/
def num_colors : ℕ := 5

/-- The number of fabric materials available for each color -/
def num_materials : ℕ := 2

/-- The number of patterns available -/
def num_patterns : ℕ := 4

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_materials * num_patterns

theorem dress_designs_count : total_designs = 40 := by
  sorry

end NUMINAMATH_CALUDE_dress_designs_count_l1362_136213


namespace NUMINAMATH_CALUDE_annual_percentage_increase_l1362_136255

theorem annual_percentage_increase (initial_population final_population : ℕ) 
  (h1 : initial_population = 10000)
  (h2 : final_population = 12000) :
  (((final_population - initial_population) : ℚ) / initial_population) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_annual_percentage_increase_l1362_136255


namespace NUMINAMATH_CALUDE_M_remainder_1000_l1362_136267

/-- M is the greatest integer multiple of 9 with no two digits being the same -/
def M : ℕ :=
  sorry

/-- The remainder when M is divided by 1000 is 621 -/
theorem M_remainder_1000 : M % 1000 = 621 :=
  sorry

end NUMINAMATH_CALUDE_M_remainder_1000_l1362_136267


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1362_136252

theorem gcd_of_three_numbers : Nat.gcd 9242 (Nat.gcd 13863 34657) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1362_136252


namespace NUMINAMATH_CALUDE_light_wash_count_l1362_136214

/-- Represents the number of gallons of water used per load for different wash types -/
structure WaterUsage where
  heavy : ℕ
  regular : ℕ
  light : ℕ

/-- Represents the number of loads for each wash type -/
structure Loads where
  heavy : ℕ
  regular : ℕ
  light : ℕ
  bleached : ℕ

def totalWaterUsage (usage : WaterUsage) (loads : Loads) : ℕ :=
  usage.heavy * loads.heavy +
  usage.regular * loads.regular +
  usage.light * (loads.light + loads.bleached)

theorem light_wash_count (usage : WaterUsage) (loads : Loads) :
  usage.heavy = 20 →
  usage.regular = 10 →
  usage.light = 2 →
  loads.heavy = 2 →
  loads.regular = 3 →
  loads.bleached = 2 →
  totalWaterUsage usage loads = 76 →
  loads.light = 1 :=
sorry

end NUMINAMATH_CALUDE_light_wash_count_l1362_136214


namespace NUMINAMATH_CALUDE_distance_P_to_AB_l1362_136289

-- Define the rectangle ABCD
def rectangle_ABCD (A B C D : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 8 ∧
  B.1 = 6 ∧ B.2 = 8 ∧
  C.1 = 6 ∧ C.2 = 0 ∧
  D.1 = 0 ∧ D.2 = 0

-- Define point M as the midpoint of CD
def point_M (C D M : ℝ × ℝ) : Prop :=
  M.1 = (C.1 + D.1) / 2 ∧ M.2 = (C.2 + D.2) / 2

-- Define the circle with center M and radius 3
def circle_M (M P : ℝ × ℝ) : Prop :=
  (P.1 - M.1)^2 + (P.2 - M.2)^2 = 3^2

-- Define the circle with center B and radius 5
def circle_B (B P : ℝ × ℝ) : Prop :=
  (P.1 - B.1)^2 + (P.2 - B.2)^2 = 5^2

-- Theorem statement
theorem distance_P_to_AB (A B C D M P : ℝ × ℝ) :
  rectangle_ABCD A B C D →
  point_M C D M →
  circle_M M P →
  circle_B B P →
  P.1 = 18/5 := by sorry

end NUMINAMATH_CALUDE_distance_P_to_AB_l1362_136289


namespace NUMINAMATH_CALUDE_boots_sold_l1362_136223

theorem boots_sold (sneakers sandals total : ℕ) 
  (h1 : sneakers = 2)
  (h2 : sandals = 4)
  (h3 : total = 17) :
  total - (sneakers + sandals) = 11 := by
  sorry

end NUMINAMATH_CALUDE_boots_sold_l1362_136223


namespace NUMINAMATH_CALUDE_congruence_condition_l1362_136249

/-- A triangle specified by two sides and an angle --/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (angle : ℝ)

/-- Predicate to check if a triangle specification guarantees congruence --/
def guarantees_congruence (t : Triangle) : Prop :=
  t.side1 > 0 ∧ t.side2 > 0 ∧ t.angle > 0 ∧ t.angle < 180 ∧
  (t.angle > 90 ∨ (t.angle = 90 ∧ t.side1 ≠ t.side2))

/-- The triangles from the problem options --/
def triangle_A : Triangle := { side1 := 2, side2 := 0, angle := 60 }
def triangle_B : Triangle := { side1 := 2, side2 := 3, angle := 0 }
def triangle_C : Triangle := { side1 := 3, side2 := 5, angle := 150 }
def triangle_D : Triangle := { side1 := 3, side2 := 2, angle := 30 }

theorem congruence_condition :
  guarantees_congruence triangle_C ∧
  ¬guarantees_congruence triangle_A ∧
  ¬guarantees_congruence triangle_B ∧
  ¬guarantees_congruence triangle_D :=
sorry

end NUMINAMATH_CALUDE_congruence_condition_l1362_136249


namespace NUMINAMATH_CALUDE_tan_product_seventh_pi_l1362_136237

theorem tan_product_seventh_pi : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_seventh_pi_l1362_136237


namespace NUMINAMATH_CALUDE_matrix_transformation_l1362_136251

/-- Given a 2nd-order matrix M satisfying the condition, prove M and the transformed curve equation -/
theorem matrix_transformation (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M * !![1, 2; 3, 4] = !![7, 10; 4, 6]) : 
  (M = !![1, 2; 1, 1]) ∧ 
  (∀ x' y' : ℝ, (∃ x y : ℝ, 3*x^2 + 8*x*y + 6*y^2 = 1 ∧ 
                            x' = x + 2*y ∧ 
                            y' = x + y) ↔ 
                x'^2 + 2*y'^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_matrix_transformation_l1362_136251


namespace NUMINAMATH_CALUDE_hotel_rooms_count_l1362_136254

/-- Calculates the total number of rooms in a hotel with three wings. -/
def total_rooms_in_hotel (
  wing1_floors : ℕ) (wing1_halls_per_floor : ℕ) (wing1_rooms_per_hall : ℕ)
  (wing2_floors : ℕ) (wing2_halls_per_floor : ℕ) (wing2_rooms_per_hall : ℕ)
  (wing3_floors : ℕ) (wing3_halls_per_floor : ℕ) (wing3_rooms_per_hall : ℕ) : ℕ :=
  wing1_floors * wing1_halls_per_floor * wing1_rooms_per_hall +
  wing2_floors * wing2_halls_per_floor * wing2_rooms_per_hall +
  wing3_floors * wing3_halls_per_floor * wing3_rooms_per_hall

/-- Theorem stating that the total number of rooms in the hotel is 6648. -/
theorem hotel_rooms_count : 
  total_rooms_in_hotel 9 6 32 7 9 40 12 4 50 = 6648 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_count_l1362_136254


namespace NUMINAMATH_CALUDE_range_of_a_for_intersection_equality_l1362_136244

/-- The set A defined by the quadratic equation x^2 - 3x + 2 = 0 -/
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

/-- The set B defined by the quadratic equation x^2 - ax + 3a - 5 = 0, parameterized by a -/
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + 3*a - 5 = 0}

/-- The theorem stating the range of a for which A ∩ B = B -/
theorem range_of_a_for_intersection_equality :
  ∀ a : ℝ, (A ∩ B a = B a) → (a ∈ Set.Icc 2 10 ∪ {1}) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_intersection_equality_l1362_136244


namespace NUMINAMATH_CALUDE_locus_of_P_l1362_136202

-- Define the circle
def Circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the property for point P
def P_property (P : ℝ × ℝ) : Prop :=
  ∃ (B : ℝ × ℝ),
    B ∈ Circle ∧
    (P.1 - A.1) * B.1 = P.2 * B.2 ∧  -- AP || OB
    (P.1 - A.1) * (B.1 - A.1) + P.2 * B.2 = 1  -- AP · AB = 1

-- The theorem to prove
theorem locus_of_P :
  ∀ (P : ℝ × ℝ), P_property P ↔ P.2^2 = 2 * P.1 - 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_P_l1362_136202


namespace NUMINAMATH_CALUDE_peter_investment_duration_l1362_136248

/-- Calculates the final amount after simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem peter_investment_duration :
  let principal : ℝ := 710
  let peterFinalAmount : ℝ := 815
  let davidFinalAmount : ℝ := 850
  let davidTime : ℝ := 4
  ∃ (rate : ℝ), 
    (simpleInterest principal rate davidTime = davidFinalAmount) ∧
    (simpleInterest principal rate 3 = peterFinalAmount) := by
  sorry

end NUMINAMATH_CALUDE_peter_investment_duration_l1362_136248


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1362_136201

theorem ratio_of_sum_and_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 7 * (x - y)) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1362_136201


namespace NUMINAMATH_CALUDE_factor_implies_absolute_value_l1362_136231

/-- Given a polynomial 3x^4 - mx^2 + nx + p with factors (x-3), (x+1), and (x-2), 
    prove that |3m - 2n| = 25 -/
theorem factor_implies_absolute_value (m n p : ℝ) : 
  (∀ x : ℝ, (x - 3) * (x + 1) * (x - 2) ∣ (3 * x^4 - m * x^2 + n * x + p)) → 
  |3 * m - 2 * n| = 25 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_absolute_value_l1362_136231


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1362_136225

theorem smallest_multiple_of_6_and_15 :
  ∃ b : ℕ+, (∀ n : ℕ+, 6 ∣ n ∧ 15 ∣ n → b ≤ n) ∧ 6 ∣ b ∧ 15 ∣ b ∧ b = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1362_136225


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l1362_136274

/-- The set of digits to be used -/
def digits : Finset Nat := {1, 3, 4, 7, 9}

/-- A function to check if a number is a multiple of 11 -/
def is_multiple_of_11 (n : Nat) : Bool :=
  n % 11 = 0

/-- A function to generate all 5-digit numbers from the given digits -/
def generate_numbers (digits : Finset Nat) : Finset Nat :=
  sorry

/-- A function to sum all numbers in a set that are not multiples of 11 -/
def sum_non_multiples_of_11 (numbers : Finset Nat) : Nat :=
  sorry

/-- The main theorem -/
theorem sum_of_special_numbers :
  sum_non_multiples_of_11 (generate_numbers digits) = 5842368 :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l1362_136274


namespace NUMINAMATH_CALUDE_base_8_units_digit_l1362_136270

theorem base_8_units_digit : (((324 + 73) * 27) % 8 = 7) := by
  sorry

end NUMINAMATH_CALUDE_base_8_units_digit_l1362_136270


namespace NUMINAMATH_CALUDE_tangent_ellipse_d_value_l1362_136241

/-- An ellipse in the first quadrant tangent to the x-axis and y-axis with foci at (5,9) and (d,9) -/
structure TangentEllipse where
  d : ℝ
  focus1 : ℝ × ℝ := (5, 9)
  focus2 : ℝ × ℝ := (d, 9)
  first_quadrant : d > 5
  tangent_to_axes : True  -- We assume this property without formally defining it

/-- The value of d for the given ellipse is 29.9 -/
theorem tangent_ellipse_d_value (e : TangentEllipse) : e.d = 29.9 := by
  sorry

#check tangent_ellipse_d_value

end NUMINAMATH_CALUDE_tangent_ellipse_d_value_l1362_136241


namespace NUMINAMATH_CALUDE_principal_amount_proof_l1362_136215

/-- Proves that given the specified conditions, the principal amount is 4000 (rs.) --/
theorem principal_amount_proof (rate : ℚ) (amount : ℚ) (time : ℚ) : 
  rate = 8 / 100 → amount = 640 → time = 2 → 
  (amount * 100) / (rate * time) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l1362_136215


namespace NUMINAMATH_CALUDE_parabola_range_l1362_136200

-- Define the function f(x) = x^2 - 4x + 5
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem parabola_range :
  ∀ y : ℝ, (∃ x : ℝ, 0 < x ∧ x < 3 ∧ f x = y) ↔ 1 ≤ y ∧ y < 5 := by sorry

end NUMINAMATH_CALUDE_parabola_range_l1362_136200


namespace NUMINAMATH_CALUDE_parallelogram_sides_l1362_136286

theorem parallelogram_sides (a b : ℝ) (h1 : a = 3 * b) (h2 : 2 * a + 2 * b = 24) :
  (a = 9 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sides_l1362_136286


namespace NUMINAMATH_CALUDE_harold_remaining_amount_l1362_136238

def calculate_remaining_amount (primary_income : ℚ) (freelance_income : ℚ) 
  (rent : ℚ) (car_payment : ℚ) (car_insurance : ℚ) (internet : ℚ) 
  (groceries : ℚ) (miscellaneous : ℚ) : ℚ :=
  let total_income := primary_income + freelance_income
  let electricity := 0.25 * car_payment
  let water_sewage := 0.15 * rent
  let total_expenses := rent + car_payment + car_insurance + electricity + water_sewage + internet + groceries + miscellaneous
  let amount_before_savings := total_income - total_expenses
  let savings := (2 / 3) * amount_before_savings
  amount_before_savings - savings

theorem harold_remaining_amount :
  calculate_remaining_amount 2500 500 700 300 125 75 200 150 = 423.34 := by
  sorry

end NUMINAMATH_CALUDE_harold_remaining_amount_l1362_136238


namespace NUMINAMATH_CALUDE_jiyoung_pocket_money_l1362_136204

theorem jiyoung_pocket_money (total : ℕ) (difference : ℕ) (jiyoung : ℕ) :
  total = 12000 →
  difference = 1000 →
  total = jiyoung + (jiyoung - difference) →
  jiyoung = 6500 := by
  sorry

end NUMINAMATH_CALUDE_jiyoung_pocket_money_l1362_136204


namespace NUMINAMATH_CALUDE_smallest_sum_of_primes_with_all_digits_l1362_136203

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the digits of a number -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if a list contains all digits from 0 to 9 exactly once -/
def hasAllDigitsOnce (l : List ℕ) : Prop := sorry

/-- The theorem stating the smallest sum of primes using all digits once -/
theorem smallest_sum_of_primes_with_all_digits : 
  ∃ (s : List ℕ), 
    (∀ n ∈ s, isPrime n) ∧ 
    hasAllDigitsOnce (s.bind digits) ∧
    (s.sum = 208) ∧
    (∀ (t : List ℕ), 
      (∀ m ∈ t, isPrime m) → 
      hasAllDigitsOnce (t.bind digits) → 
      t.sum ≥ 208) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_primes_with_all_digits_l1362_136203


namespace NUMINAMATH_CALUDE_exam_probabilities_l1362_136207

def prob_above_90 : ℝ := 0.18
def prob_80_to_89 : ℝ := 0.51
def prob_70_to_79 : ℝ := 0.15
def prob_60_to_69 : ℝ := 0.09

theorem exam_probabilities :
  (prob_above_90 + prob_80_to_89 = 0.69) ∧
  (prob_above_90 + prob_80_to_89 + prob_70_to_79 + prob_60_to_69 = 0.93) := by
sorry

end NUMINAMATH_CALUDE_exam_probabilities_l1362_136207


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1362_136277

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1362_136277


namespace NUMINAMATH_CALUDE_find_x_value_l1362_136292

def is_ascending (l : List ℝ) : Prop :=
  ∀ i j, i < j → i < l.length → j < l.length → l[i]! ≤ l[j]!

def median (l : List ℝ) : ℝ :=
  l[l.length / 2]!

theorem find_x_value (l : List ℝ) (h_length : l.length = 5) 
  (h_ascending : is_ascending l) (h_median : median l = 22) 
  (h_elements : l = [14, 19, x, 23, 27]) : x = 22 :=
sorry

end NUMINAMATH_CALUDE_find_x_value_l1362_136292


namespace NUMINAMATH_CALUDE_hair_cut_length_l1362_136230

def initial_length : ℕ := 14
def current_length : ℕ := 1

theorem hair_cut_length : initial_length - current_length = 13 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_length_l1362_136230


namespace NUMINAMATH_CALUDE_problem_solution_l1362_136293

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

def g (x : ℝ) : ℝ := |2*x - 3|

theorem problem_solution :
  (∀ x : ℝ, f 3 x ≤ 6 ↔ 0 ≤ x ∧ x ≤ 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x + g x ≥ 5) ↔ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1362_136293


namespace NUMINAMATH_CALUDE_gym_time_is_two_hours_l1362_136258

/-- Represents the daily schedule of a working mom --/
structure DailySchedule where
  wakeTime : Nat
  sleepTime : Nat
  workHours : Nat
  cookingTime : Real
  bathTime : Real
  homeworkTime : Real
  lunchPackingTime : Real
  cleaningTime : Real
  leisureTime : Real

/-- Calculates the total awake hours in a day --/
def awakeHours (schedule : DailySchedule) : Nat :=
  schedule.sleepTime - schedule.wakeTime

/-- Calculates the total time spent on activities excluding work and gym --/
def otherActivitiesTime (schedule : DailySchedule) : Real :=
  schedule.cookingTime + schedule.bathTime + schedule.homeworkTime +
  schedule.lunchPackingTime + schedule.cleaningTime + schedule.leisureTime

/-- Theorem: The working mom spends 2 hours at the gym --/
theorem gym_time_is_two_hours (schedule : DailySchedule) 
    (h1 : schedule.wakeTime = 7)
    (h2 : schedule.sleepTime = 23)
    (h3 : schedule.workHours = 8)
    (h4 : schedule.workHours = awakeHours schedule / 2)
    (h5 : schedule.cookingTime = 1.5)
    (h6 : schedule.bathTime = 0.5)
    (h7 : schedule.homeworkTime = 1)
    (h8 : schedule.lunchPackingTime = 0.5)
    (h9 : schedule.cleaningTime = 0.5)
    (h10 : schedule.leisureTime = 2) :
    awakeHours schedule - schedule.workHours - otherActivitiesTime schedule = 2 := by
  sorry


end NUMINAMATH_CALUDE_gym_time_is_two_hours_l1362_136258


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1362_136294

/-- The value of 'a' for a circle with equation x^2 + y^2 - 2ax + 2y + 1 = 0,
    where the line y = -x + 1 passes through its center. -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 2*a*x + 2*y + 1 = 0 ∧ 
               y = -x + 1 ∧ 
               ∀ x' y' : ℝ, x'^2 + y'^2 - 2*a*x' + 2*y' + 1 = 0 → 
                 (x - x')^2 + (y - y')^2 ≤ (x' - a)^2 + (y' + 1)^2) → 
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1362_136294


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1362_136257

/-- Given an arithmetic sequence with common difference 2, if a₁, a₃, a₄ form a geometric sequence, then a₂ = -6 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →  -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1362_136257


namespace NUMINAMATH_CALUDE_license_plate_count_l1362_136235

/-- The number of vowels in the alphabet, considering Y as a vowel -/
def num_vowels : ℕ := 6

/-- The number of consonants in the alphabet -/
def num_consonants : ℕ := 20

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_consonants * num_vowels * num_vowels * num_consonants * num_vowels * num_digits

theorem license_plate_count : total_license_plates = 403200 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1362_136235


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_lawn_mowing_solution_l1362_136216

theorem lawn_mowing_problem (original_people : ℕ) (original_time : ℝ) 
  (new_time : ℝ) (efficiency : ℝ) (new_people : ℕ) : Prop :=
  original_people = 8 →
  original_time = 3 →
  new_time = 2 →
  efficiency = 0.9 →
  (original_people : ℝ) * original_time = (new_people : ℝ) * new_time * efficiency →
  new_people = 14

-- The proof of the theorem
theorem lawn_mowing_solution : lawn_mowing_problem 8 3 2 0.9 14 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_lawn_mowing_solution_l1362_136216


namespace NUMINAMATH_CALUDE_student_venue_arrangements_l1362_136260

theorem student_venue_arrangements (n : Nat) (a b c : Nat) 
  (h1 : n = 6)
  (h2 : a = 3)
  (h3 : b = 1)
  (h4 : c = 2)
  (h5 : a + b + c = n) :
  Nat.choose n a * Nat.choose (n - a) b * Nat.choose (n - a - b) c = 60 :=
by sorry

end NUMINAMATH_CALUDE_student_venue_arrangements_l1362_136260


namespace NUMINAMATH_CALUDE_equation_system_solution_l1362_136275

theorem equation_system_solution (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l1362_136275


namespace NUMINAMATH_CALUDE_reading_time_proof_l1362_136264

def total_chapters : Nat := 31
def reading_time_per_chapter : Nat := 20

def chapters_read (n : Nat) : Nat :=
  n - (n / 3)

def total_reading_time_minutes (n : Nat) (t : Nat) : Nat :=
  (chapters_read n) * t

def total_reading_time_hours (n : Nat) (t : Nat) : Nat :=
  (total_reading_time_minutes n t) / 60

theorem reading_time_proof :
  total_reading_time_hours total_chapters reading_time_per_chapter = 7 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_proof_l1362_136264


namespace NUMINAMATH_CALUDE_three_dozen_cost_l1362_136272

/-- The cost of apples in dollars -/
def apple_cost (dozens : ℚ) : ℚ := 15.60 * dozens / 2

/-- Theorem: The cost of three dozen apples is $23.40 -/
theorem three_dozen_cost : apple_cost 3 = 23.40 := by
  sorry

end NUMINAMATH_CALUDE_three_dozen_cost_l1362_136272


namespace NUMINAMATH_CALUDE_old_clock_slow_12_minutes_l1362_136288

/-- Represents the time interval between hand overlaps on the old clock -/
def old_clock_overlap_interval : ℚ := 66

/-- Represents the standard time interval between hand overlaps -/
def standard_overlap_interval : ℚ := 720 / 11

/-- Represents the number of minutes in a standard day -/
def standard_day_minutes : ℕ := 24 * 60

/-- Theorem stating that the old clock is 12 minutes slow over a 24-hour period -/
theorem old_clock_slow_12_minutes :
  (standard_day_minutes : ℚ) / standard_overlap_interval * old_clock_overlap_interval
  - standard_day_minutes = 12 := by sorry

end NUMINAMATH_CALUDE_old_clock_slow_12_minutes_l1362_136288


namespace NUMINAMATH_CALUDE_complex_symmetric_division_l1362_136219

/-- Two complex numbers are symmetric about the origin if their sum is zero -/
def symmetric_about_origin (z₁ z₂ : ℂ) : Prop := z₁ + z₂ = 0

theorem complex_symmetric_division (z₁ z₂ : ℂ) 
  (h_sym : symmetric_about_origin z₁ z₂) (h_z₁ : z₁ = 2 - I) : 
  z₁ / z₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_symmetric_division_l1362_136219


namespace NUMINAMATH_CALUDE_ceiling_sqrt_200_l1362_136279

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_200_l1362_136279


namespace NUMINAMATH_CALUDE_area_bisecting_line_sum_l1362_136265

/-- Triangle ABC with vertices A(0, 10), B(3, 0), C(9, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- Predicate to check if a line bisects the area of a triangle through a specific vertex -/
def bisects_area (t : Triangle) (l : Line) (vertex : ℝ × ℝ) : Prop :=
  sorry

/-- The triangle ABC with given vertices -/
def triangle_ABC : Triangle :=
  { A := (0, 10),
    B := (3, 0),
    C := (9, 0) }

theorem area_bisecting_line_sum :
  ∃ l : Line, bisects_area triangle_ABC l triangle_ABC.B ∧ l.slope + l.y_intercept = -20/3 := by
  sorry

end NUMINAMATH_CALUDE_area_bisecting_line_sum_l1362_136265


namespace NUMINAMATH_CALUDE_chessboard_pythagorean_triple_exists_l1362_136253

/-- Represents a point on an infinite chessboard --/
structure ChessboardPoint where
  x : Int
  y : Int

/-- Distance function between two ChessboardPoints --/
def distance (p q : ChessboardPoint) : Nat :=
  ((p.x - q.x)^2 + (p.y - q.y)^2).natAbs

/-- Predicate to check if three points are non-collinear --/
def nonCollinear (p q r : ChessboardPoint) : Prop :=
  (q.x - p.x) * (r.y - p.y) ≠ (r.x - p.x) * (q.y - p.y)

/-- Theorem stating the existence of points satisfying the given conditions --/
theorem chessboard_pythagorean_triple_exists : 
  ∃ (A B C : ChessboardPoint), 
    nonCollinear A B C ∧ 
    (distance A C)^2 + (distance B C)^2 = (distance A B)^2 := by
  sorry


end NUMINAMATH_CALUDE_chessboard_pythagorean_triple_exists_l1362_136253


namespace NUMINAMATH_CALUDE_real_part_of_z_l1362_136206

theorem real_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) :
  z.re = (Real.sqrt 2 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l1362_136206


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l1362_136273

theorem largest_n_binomial_equality : 
  (∃ n : ℕ, (Nat.choose 8 3 + Nat.choose 8 4 = Nat.choose 9 n)) ∧ 
  (∀ m : ℕ, Nat.choose 8 3 + Nat.choose 8 4 = Nat.choose 9 m → m ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l1362_136273


namespace NUMINAMATH_CALUDE_largest_difference_l1362_136280

def S : Set Int := {-20, -5, 1, 5, 7, 19}

theorem largest_difference (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x - y = 39 ∧ ∀ (c d : Int), c ∈ S → d ∈ S → c - d ≤ 39 := by
  sorry

end NUMINAMATH_CALUDE_largest_difference_l1362_136280


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1362_136240

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = {1, 5, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1362_136240


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_point_l1362_136268

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the left focus F
def left_focus : ℝ × ℝ := (-1, 0)

-- Define line l (implicitly through its properties)
def line_l (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x + 1)

-- Define points A and B
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the perpendicular point H
def point_H : ℝ × ℝ := sorry

-- State the theorem
theorem ellipse_perpendicular_point :
  ∀ (A B : ℝ × ℝ),
    ellipse A.1 A.2 →
    ellipse B.1 B.2 →
    line_l A.1 A.2 →
    line_l B.1 B.2 →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    (point_H = (-2/3, Real.sqrt 2/3) ∨ point_H = (-2/3, -Real.sqrt 2/3)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_point_l1362_136268


namespace NUMINAMATH_CALUDE_simplify_fraction_multiplication_l1362_136205

theorem simplify_fraction_multiplication :
  (180 : ℚ) / 1620 * 20 = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_multiplication_l1362_136205


namespace NUMINAMATH_CALUDE_als_original_portion_l1362_136296

theorem als_original_portion (total_initial : ℕ) (total_final : ℕ) 
  (al_loss : ℕ) (al betty clare : ℕ) :
  total_initial = 1500 →
  total_final = 2250 →
  al_loss = 150 →
  al + betty + clare = total_initial →
  (al - al_loss) + 3 * betty + 3 * clare = total_final →
  al = 1050 :=
by sorry

end NUMINAMATH_CALUDE_als_original_portion_l1362_136296


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_111_l1362_136212

theorem alpha_plus_beta_equals_111 :
  ∀ α β : ℝ, (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 72*x + 1343) / (x^2 + 63*x - 3360)) →
  α + β = 111 :=
by
  sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_111_l1362_136212


namespace NUMINAMATH_CALUDE_triangle_segment_calculation_l1362_136245

/-- Given a triangle ABC with point D on AB and point E on AD, prove that FC has the specified value. -/
theorem triangle_segment_calculation (DC CB AD : ℝ) (h1 : DC = 9) (h2 : CB = 10) 
  (h3 : (1 : ℝ)/5 * AD = AD - DC - CB) (h4 : (3 : ℝ)/4 * AD = ED) : 
  let CA := CB + AD - DC - CB
  let FC := ED * CA / AD
  FC = 11.025 := by sorry

end NUMINAMATH_CALUDE_triangle_segment_calculation_l1362_136245


namespace NUMINAMATH_CALUDE_susan_pencil_purchase_l1362_136236

/-- The number of pencils Susan bought -/
def num_pencils : ℕ := 16

/-- The number of pens Susan bought -/
def num_pens : ℕ := 36 - num_pencils

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := 25

/-- The cost of a pen in cents -/
def pen_cost : ℕ := 80

/-- The total amount Susan spent in cents -/
def total_spent : ℕ := 2000

theorem susan_pencil_purchase :
  num_pencils + num_pens = 36 ∧
  pencil_cost * num_pencils + pen_cost * num_pens = total_spent :=
by sorry

#check susan_pencil_purchase

end NUMINAMATH_CALUDE_susan_pencil_purchase_l1362_136236


namespace NUMINAMATH_CALUDE_overlapping_triangles_sum_l1362_136291

/-- Represents a triangle with angles a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_180 : a + b + c = 180

/-- Configuration of two pairs of overlapping triangles -/
structure OverlappingTriangles where
  t1 : Triangle
  t2 : Triangle

/-- The sum of all distinct angles in a configuration of two pairs of overlapping triangles is 360° -/
theorem overlapping_triangles_sum (ot : OverlappingTriangles) : 
  ot.t1.a + ot.t1.b + ot.t1.c + ot.t2.a + ot.t2.b + ot.t2.c = 360 := by
  sorry


end NUMINAMATH_CALUDE_overlapping_triangles_sum_l1362_136291


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l1362_136222

/-- The line passing through points (2, 9) and (4, 15) intersects the y-axis at (0, 3) -/
theorem line_intersection_y_axis :
  let p₁ : ℝ × ℝ := (2, 9)
  let p₂ : ℝ × ℝ := (4, 15)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  let line (x : ℝ) : ℝ := m * x + b
  (0, line 0) = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l1362_136222


namespace NUMINAMATH_CALUDE_cost_per_bag_is_seven_l1362_136295

/-- Calculates the cost per bag given the number of bags, selling price, and desired profit --/
def cost_per_bag (num_bags : ℕ) (selling_price : ℚ) (desired_profit : ℚ) : ℚ :=
  (num_bags * selling_price - desired_profit) / num_bags

/-- Theorem: Given 100 bags sold at $10 each with a $300 profit, the cost per bag is $7 --/
theorem cost_per_bag_is_seven :
  cost_per_bag 100 10 300 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_bag_is_seven_l1362_136295


namespace NUMINAMATH_CALUDE_dress_making_hours_l1362_136209

/-- Calculates the total hours required to make dresses given the available fabric, fabric per dress, and time per dress. -/
def total_hours (total_fabric : ℕ) (fabric_per_dress : ℕ) (time_per_dress : ℕ) : ℕ :=
  (total_fabric / fabric_per_dress) * time_per_dress

/-- Proves that given 56 square meters of fabric, where each dress requires 4 square meters of fabric
    and 3 hours to make, the total number of hours required to make all possible dresses is 42 hours. -/
theorem dress_making_hours : total_hours 56 4 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_dress_making_hours_l1362_136209


namespace NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l1362_136281

theorem sum_of_four_primes_divisible_by_60 
  (p q r s : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hr : Nat.Prime r) 
  (hs : Nat.Prime s) 
  (h_order : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) : 
  60 ∣ (p + q + r + s) := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l1362_136281


namespace NUMINAMATH_CALUDE_a_value_l1362_136246

def set_A : Set ℝ := {1, -2}

def set_B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem a_value (b : ℝ) : set_A = set_B 1 b → 1 ∈ set_A ∧ -2 ∈ set_A := by sorry

end NUMINAMATH_CALUDE_a_value_l1362_136246


namespace NUMINAMATH_CALUDE_sum_of_A_and_D_is_six_l1362_136234

-- Define single-digit numbers
def SingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

-- Define three-digit number ABX
def ThreeDigitABX (A B X : ℕ) : ℕ := 100 * A + 10 * B + X

-- Define three-digit number CDY
def ThreeDigitCDY (C D Y : ℕ) : ℕ := 100 * C + 10 * D + Y

-- Define four-digit number XYXY
def FourDigitXYXY (X Y : ℕ) : ℕ := 1000 * X + 100 * Y + 10 * X + Y

-- Theorem statement
theorem sum_of_A_and_D_is_six 
  (A B C D X Y : ℕ) 
  (hA : SingleDigit A) (hB : SingleDigit B) (hC : SingleDigit C) 
  (hD : SingleDigit D) (hX : SingleDigit X) (hY : SingleDigit Y)
  (h_sum : ThreeDigitABX A B X + ThreeDigitCDY C D Y = FourDigitXYXY X Y) :
  A + D = 6 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_A_and_D_is_six_l1362_136234


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l1362_136271

theorem quadratic_rewrite_ratio : 
  ∃ (c p q : ℚ), 
    (∀ j, 8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ 
    q / p = -77 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l1362_136271


namespace NUMINAMATH_CALUDE_f_at_three_l1362_136217

/-- Horner's method representation of the polynomial f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := (((2 * x + 3) * x + 0) * x + 5) * x - 4

/-- Theorem stating that f(3) = 254 -/
theorem f_at_three : f 3 = 254 := by sorry

end NUMINAMATH_CALUDE_f_at_three_l1362_136217


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1362_136210

theorem quadratic_inequality (x : ℝ) : x^2 + 7*x + 6 < 0 ↔ -6 < x ∧ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1362_136210


namespace NUMINAMATH_CALUDE_octagon_game_areas_l1362_136239

/-- A regular octagon inscribed in a circle of radius 2 -/
structure RegularOctagon :=
  (radius : ℝ)
  (vertices : Fin 8 → ℝ × ℝ)
  (is_regular : ∀ i : Fin 8, (vertices i).1^2 + (vertices i).2^2 = radius^2)

/-- The set of vertices selected by a player -/
def PlayerSelection := Finset (Fin 8)

/-- Predicate for optimal play -/
def OptimalPlay (octagon : RegularOctagon) (alice_selection : PlayerSelection) (bob_selection : PlayerSelection) : Prop :=
  sorry

/-- The area of the convex polygon formed by a player's selection -/
def PolygonArea (octagon : RegularOctagon) (selection : PlayerSelection) : ℝ :=
  sorry

/-- The main theorem -/
theorem octagon_game_areas (octagon : RegularOctagon) (alice_selection : PlayerSelection) (bob_selection : PlayerSelection) :
  octagon.radius = 2 →
  OptimalPlay octagon alice_selection bob_selection →
  alice_selection.card = 4 →
  bob_selection.card = 4 →
  (PolygonArea octagon alice_selection = 2 * Real.sqrt 2 ∨
   PolygonArea octagon alice_selection = 4 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_octagon_game_areas_l1362_136239


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_pi_12_l1362_136220

theorem sin_2alpha_plus_pi_12 (α : ℝ) 
  (h1 : -π/6 < α ∧ α < π/6) 
  (h2 : Real.cos (α + π/6) = 4/5) : 
  Real.sin (2*α + π/12) = 17 * Real.sqrt 2 / 50 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_pi_12_l1362_136220


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_l1362_136284

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℚ) : ℚ → ℚ := fun x ↦ a * x^2 + b * x + c

/-- The condition that [q(x)]^3 - x is divisible by (x - 2)(x + 2)(x - 5) -/
def DivisibilityCondition (q : ℚ → ℚ) : Prop :=
  ∀ x, x = 2 ∨ x = -2 ∨ x = 5 → q x ^ 3 = x

theorem quadratic_polynomial_value (a b c : ℚ) :
  (∃ q : ℚ → ℚ, q = QuadraticPolynomial a b c ∧ DivisibilityCondition q) →
  QuadraticPolynomial a b c 10 = -58/7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_value_l1362_136284


namespace NUMINAMATH_CALUDE_P_less_than_Q_l1362_136283

theorem P_less_than_Q (a : ℝ) (ha : a ≥ 0) :
  Real.sqrt (a + 3) + Real.sqrt (a + 5) < Real.sqrt (a + 1) + Real.sqrt (a + 7) := by
  sorry

end NUMINAMATH_CALUDE_P_less_than_Q_l1362_136283


namespace NUMINAMATH_CALUDE_hyperbola_center_l1362_136233

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f₁ f₂ c : ℝ × ℝ) : 
  f₁ = (3, 2) → f₂ = (11, 6) → c = (7, 4) → 
  c = ((f₁.1 + f₂.1) / 2, (f₁.2 + f₂.2) / 2) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l1362_136233


namespace NUMINAMATH_CALUDE_immediate_sale_more_profitable_l1362_136287

/-- Proves that selling flowers immediately is more profitable than selling after dehydration --/
theorem immediate_sale_more_profitable (initial_weight : ℝ) (initial_price : ℝ) (price_increase : ℝ) 
  (weight_loss_fraction : ℝ) (hw : initial_weight = 49) (hp : initial_price = 1.25) 
  (hpi : price_increase = 2) (hwl : weight_loss_fraction = 5/7) :
  initial_weight * initial_price > 
  (initial_weight * (1 - weight_loss_fraction)) * (initial_price + price_increase) :=
by sorry

end NUMINAMATH_CALUDE_immediate_sale_more_profitable_l1362_136287


namespace NUMINAMATH_CALUDE_equation_solution_l1362_136228

theorem equation_solution : ∃ x : ℝ, ((x * 0.85) / 2.5) - (8 * 2.25) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1362_136228
