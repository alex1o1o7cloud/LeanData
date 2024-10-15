import Mathlib

namespace NUMINAMATH_CALUDE_min_packs_for_144_cans_l3694_369436

/-- Represents the number of cans in each pack size --/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans for a given pack size --/
def cansInPack (s : PackSize) : Nat :=
  match s with
  | PackSize.small => 8
  | PackSize.medium => 18
  | PackSize.large => 30

/-- Represents a combination of packs --/
structure PackCombination where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of cans in a pack combination --/
def totalCans (c : PackCombination) : Nat :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a combination --/
def totalPacks (c : PackCombination) : Nat :=
  c.small + c.medium + c.large

/-- Defines what it means for a pack combination to be valid --/
def isValidCombination (c : PackCombination) : Prop :=
  totalCans c = 144

/-- Theorem: The minimum number of packs to buy 144 cans is 6 --/
theorem min_packs_for_144_cans :
  ∃ (c : PackCombination),
    isValidCombination c ∧
    totalPacks c = 6 ∧
    (∀ (c' : PackCombination), isValidCombination c' → totalPacks c' ≥ 6) :=
  sorry

end NUMINAMATH_CALUDE_min_packs_for_144_cans_l3694_369436


namespace NUMINAMATH_CALUDE_clean_city_people_l3694_369474

/-- The number of people in group A -/
def group_A : ℕ := 54

/-- The number of people in group B -/
def group_B : ℕ := group_A - 17

/-- The number of people in group C -/
def group_C : ℕ := 2 * group_B

/-- The number of people in group D -/
def group_D : ℕ := group_A / 3

/-- The total number of people working together to clean the city -/
def total_people : ℕ := group_A + group_B + group_C + group_D

theorem clean_city_people : total_people = 183 := by
  sorry

end NUMINAMATH_CALUDE_clean_city_people_l3694_369474


namespace NUMINAMATH_CALUDE_smaller_bedroom_size_l3694_369426

/-- Given two bedrooms with a total area of 300 square feet, where one bedroom
    is 60 square feet larger than the other, prove that the smaller bedroom
    is 120 square feet. -/
theorem smaller_bedroom_size (total_area : ℝ) (difference : ℝ) (smaller : ℝ) :
  total_area = 300 →
  difference = 60 →
  total_area = smaller + (smaller + difference) →
  smaller = 120 := by
  sorry

end NUMINAMATH_CALUDE_smaller_bedroom_size_l3694_369426


namespace NUMINAMATH_CALUDE_discount_profit_calculation_l3694_369473

/-- Given a discount percentage and an original profit percentage,
    calculate the new profit percentage after applying the discount. -/
def profit_after_discount (discount : ℝ) (original_profit : ℝ) : ℝ :=
  let original_price := 1 + original_profit
  let discounted_price := original_price * (1 - discount)
  (discounted_price - 1) * 100

/-- Theorem stating that a 5% discount on an item with 50% original profit
    results in a 42.5% profit. -/
theorem discount_profit_calculation :
  profit_after_discount 0.05 0.5 = 42.5 := by sorry

end NUMINAMATH_CALUDE_discount_profit_calculation_l3694_369473


namespace NUMINAMATH_CALUDE_solve_determinant_equation_l3694_369425

-- Define the determinant operation
def det (a b c d : ℚ) : ℚ := a * d - b * c

-- Theorem statement
theorem solve_determinant_equation :
  ∀ x : ℚ, det 2 4 (1 - x) 5 = 18 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_determinant_equation_l3694_369425


namespace NUMINAMATH_CALUDE_range_of_a_for_meaningful_sqrt_l3694_369417

theorem range_of_a_for_meaningful_sqrt (a : ℝ) : 
  (∃ x : ℝ, x^2 = 4 - a) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_meaningful_sqrt_l3694_369417


namespace NUMINAMATH_CALUDE_digit_2023_is_7_l3694_369429

/-- The sequence of digits obtained by writing integers 1 through 9999 in ascending order -/
def digit_sequence : ℕ → ℕ := sorry

/-- The real number x defined as .123456789101112...99989999 -/
noncomputable def x : ℝ := sorry

/-- The nth digit to the right of the decimal point in x -/
def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_2023_is_7 : nth_digit 2023 = 7 := by sorry

end NUMINAMATH_CALUDE_digit_2023_is_7_l3694_369429


namespace NUMINAMATH_CALUDE_function_property_l3694_369416

theorem function_property (f : ℕ+ → ℕ) 
  (h1 : ∀ (x y : ℕ+), f (x * y) = f x + f y)
  (h2 : f 30 = 21)
  (h3 : f 90 = 27) :
  f 270 = 33 := by
sorry

end NUMINAMATH_CALUDE_function_property_l3694_369416


namespace NUMINAMATH_CALUDE_water_boiling_point_l3694_369421

/-- The temperature in Fahrenheit at which water boils -/
def boiling_point_f : ℝ := 212

/-- The temperature in Fahrenheit at which water melts -/
def melting_point_f : ℝ := 32

/-- The temperature in Celsius at which water melts -/
def melting_point_c : ℝ := 0

/-- A function to convert Celsius to Fahrenheit -/
def celsius_to_fahrenheit (c : ℝ) : ℝ := sorry

/-- A function to convert Fahrenheit to Celsius -/
def fahrenheit_to_celsius (f : ℝ) : ℝ := sorry

/-- The boiling point of water in Celsius -/
def boiling_point_c : ℝ := 100

theorem water_boiling_point :
  ∃ (temp_c temp_f : ℝ),
    celsius_to_fahrenheit temp_c = temp_f ∧
    temp_c = 35 ∧
    temp_f = 95 →
  fahrenheit_to_celsius boiling_point_f = boiling_point_c :=
sorry

end NUMINAMATH_CALUDE_water_boiling_point_l3694_369421


namespace NUMINAMATH_CALUDE_only_1_and_4_perpendicular_l3694_369415

-- Define the slopes of the lines
def m1 : ℚ := 2/3
def m2 : ℚ := -2/3
def m3 : ℚ := -2/3
def m4 : ℚ := -3/2

-- Define a function to check if two lines are perpendicular
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem only_1_and_4_perpendicular :
  (are_perpendicular m1 m4) ∧
  ¬(are_perpendicular m1 m2) ∧
  ¬(are_perpendicular m1 m3) ∧
  ¬(are_perpendicular m2 m3) ∧
  ¬(are_perpendicular m2 m4) ∧
  ¬(are_perpendicular m3 m4) :=
by sorry

end NUMINAMATH_CALUDE_only_1_and_4_perpendicular_l3694_369415


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3694_369444

theorem arithmetic_calculations :
  ((-15) + 4 + (-6) - (-11) = -6) ∧
  (-1^2024 + (-3)^2 * |(-1/18)| - 1 / (-2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3694_369444


namespace NUMINAMATH_CALUDE_compound_composition_l3694_369464

/-- The number of Aluminium atoms in the compound -/
def n : ℕ := 1

/-- Atomic weight of Aluminium in g/mol -/
def Al_weight : ℚ := 26.98

/-- Atomic weight of Phosphorus in g/mol -/
def P_weight : ℚ := 30.97

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℚ := 16.00

/-- Total molecular weight of the compound in g/mol -/
def total_weight : ℚ := 122

/-- Number of Phosphorus atoms in the compound -/
def P_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 4

theorem compound_composition :
  n * Al_weight + P_count * P_weight + O_count * O_weight = total_weight :=
sorry

end NUMINAMATH_CALUDE_compound_composition_l3694_369464


namespace NUMINAMATH_CALUDE_divisible_by_fifteen_l3694_369410

theorem divisible_by_fifteen (a : ℤ) : ∃ k : ℤ, 9 * a^5 - 5 * a^3 - 4 * a = 15 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_fifteen_l3694_369410


namespace NUMINAMATH_CALUDE_mixture_problem_l3694_369494

theorem mixture_problem (initial_ratio_A B : ℚ) (drawn_off filled_B : ℚ) (final_ratio_A B : ℚ) :
  initial_ratio_A = 7 →
  initial_ratio_B = 5 →
  drawn_off = 9 →
  filled_B = 9 →
  final_ratio_A = 7 →
  final_ratio_B = 9 →
  ∃ x : ℚ,
    let initial_A := initial_ratio_A * x
    let initial_B := initial_ratio_B * x
    let removed_A := (initial_ratio_A / (initial_ratio_A + initial_ratio_B)) * drawn_off
    let removed_B := (initial_ratio_B / (initial_ratio_A + initial_ratio_B)) * drawn_off
    let remaining_A := initial_A - removed_A
    let remaining_B := initial_B - removed_B + filled_B
    remaining_A / remaining_B = final_ratio_A / final_ratio_B ∧
    initial_A = 23.625 :=
by sorry

end NUMINAMATH_CALUDE_mixture_problem_l3694_369494


namespace NUMINAMATH_CALUDE_arc_length_for_specific_circle_l3694_369414

theorem arc_length_for_specific_circle (r : ℝ) (α : ℝ) (l : ℝ) : 
  r = π → α = 2 * π / 3 → l = r * α → l = 2 * π^2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_arc_length_for_specific_circle_l3694_369414


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l3694_369434

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the intersection points A and B
def intersectionPoints (A B : ℝ × ℝ) : Prop :=
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B

-- Define the perpendicular bisector
def perpendicularBisector (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersecting_circles
  (A B : ℝ × ℝ) (h : intersectionPoints A B) :
  ∀ x y, perpendicularBisector x y ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l3694_369434


namespace NUMINAMATH_CALUDE_max_candies_in_25_days_l3694_369495

/-- Represents the dentist's instructions for candy consumption --/
structure CandyRules :=
  (max_daily : ℕ)
  (threshold : ℕ)
  (reduced_max : ℕ)
  (reduced_days : ℕ)

/-- Calculates the maximum number of candies that can be eaten in a given number of days --/
def max_candies (rules : CandyRules) (days : ℕ) : ℕ :=
  sorry

/-- The dentist's specific instructions --/
def dentist_rules : CandyRules :=
  { max_daily := 10
  , threshold := 7
  , reduced_max := 5
  , reduced_days := 2 }

/-- Theorem stating the maximum number of candies Sonia can eat in 25 days --/
theorem max_candies_in_25_days :
  max_candies dentist_rules 25 = 178 :=
sorry

end NUMINAMATH_CALUDE_max_candies_in_25_days_l3694_369495


namespace NUMINAMATH_CALUDE_root_product_equality_l3694_369492

-- Define the quadratic equations
def quadratic1 (x p c : ℝ) : ℝ := x^2 + p*x + c
def quadratic2 (x q c : ℝ) : ℝ := x^2 + q*x + c

-- Define the theorem
theorem root_product_equality (p q c : ℝ) (α β γ δ : ℝ) :
  quadratic1 α p c = 0 →
  quadratic1 β p c = 0 →
  quadratic2 γ q c = 0 →
  quadratic2 δ q c = 0 →
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = (p^2 - q^2) * c + c^2 - p*c - q*c :=
by sorry

end NUMINAMATH_CALUDE_root_product_equality_l3694_369492


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3694_369454

theorem chess_tournament_games (P : ℕ) (total_games : ℕ) (h1 : P = 21) (h2 : total_games = 210) :
  (P * (P - 1)) / 2 = total_games ∧ P - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3694_369454


namespace NUMINAMATH_CALUDE_suraya_vs_mia_l3694_369476

/-- The number of apples picked by each person -/
structure ApplePickers where
  kayla : ℕ
  caleb : ℕ
  suraya : ℕ
  mia : ℕ

/-- The conditions of the apple-picking scenario -/
def apple_picking_conditions (a : ApplePickers) : Prop :=
  a.kayla = 20 ∧
  a.caleb = a.kayla / 2 - 5 ∧
  a.suraya = 3 * a.caleb ∧
  a.mia = 2 * a.caleb

/-- The theorem stating that Suraya picked 5 more apples than Mia -/
theorem suraya_vs_mia (a : ApplePickers) 
  (h : apple_picking_conditions a) : a.suraya = a.mia + 5 := by
  sorry


end NUMINAMATH_CALUDE_suraya_vs_mia_l3694_369476


namespace NUMINAMATH_CALUDE_fractional_decomposition_sum_l3694_369483

theorem fractional_decomposition_sum (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_fractional_decomposition_sum_l3694_369483


namespace NUMINAMATH_CALUDE_wall_length_calculation_l3694_369487

/-- Given a square mirror and a rectangular wall, if the mirror's area is half the wall's area,
    prove that the wall's length is approximately 27 inches. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 24 →
  wall_width = 42 →
  (mirror_side * mirror_side) * 2 = wall_width * (27 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l3694_369487


namespace NUMINAMATH_CALUDE_wheel_speed_l3694_369488

/-- The speed of the wheel in miles per hour -/
def r : ℝ := sorry

/-- The circumference of the wheel in feet -/
def circumference : ℝ := 11

/-- The time for one rotation in hours -/
def t : ℝ := sorry

/-- Conversion factor from feet to miles -/
def feet_per_mile : ℝ := 5280

/-- Conversion factor from hours to seconds -/
def seconds_per_hour : ℝ := 3600

/-- The relationship between speed, time, and distance -/
axiom speed_time_distance : r * t = circumference / feet_per_mile

/-- The relationship when time is decreased and speed is increased -/
axiom increased_speed_decreased_time : 
  (r + 5) * (t - 1 / (4 * seconds_per_hour)) = circumference / feet_per_mile

theorem wheel_speed : r = 10 := by sorry

end NUMINAMATH_CALUDE_wheel_speed_l3694_369488


namespace NUMINAMATH_CALUDE_pet_insurance_coverage_calculation_l3694_369480

/-- Calculates the amount covered by pet insurance for a cat's visit -/
def pet_insurance_coverage (
  doctor_visit_cost : ℝ
  ) (health_insurance_rate : ℝ
  ) (cat_visit_cost : ℝ
  ) (total_out_of_pocket : ℝ
  ) : ℝ :=
  cat_visit_cost - (total_out_of_pocket - (doctor_visit_cost * (1 - health_insurance_rate)))

theorem pet_insurance_coverage_calculation :
  pet_insurance_coverage 300 0.75 120 135 = 60 := by
  sorry

end NUMINAMATH_CALUDE_pet_insurance_coverage_calculation_l3694_369480


namespace NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l3694_369455

theorem polygon_sides_and_diagonals (n : ℕ) : 
  n + (n * (n - 3)) / 2 = 77 → n = 14 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l3694_369455


namespace NUMINAMATH_CALUDE_asterisk_value_l3694_369475

theorem asterisk_value : ∃ x : ℚ, (x / 21) * (42 / 84) = 1 ∧ x = 21 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_value_l3694_369475


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_five_fourths_l3694_369401

theorem fraction_of_powers_equals_five_fourths :
  (3^1007 + 3^1005) / (3^1007 - 3^1005) = 5/4 := by sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_five_fourths_l3694_369401


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l3694_369418

/-- Given a 2x2 matrix N, prove that its inverse can be expressed as c * N + d * I -/
theorem inverse_as_linear_combination (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h₁ : N 0 0 = 3) (h₂ : N 0 1 = 1) (h₃ : N 1 0 = -2) (h₄ : N 1 1 = 4) :
  ∃ (c d : ℝ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ 
  c = -1/14 ∧ d = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l3694_369418


namespace NUMINAMATH_CALUDE_matrix_computation_l3694_369481

theorem matrix_computation (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec ![1, 3] = ![2, 5])
  (h2 : N.mulVec ![-2, 4] = ![3, 1]) :
  N.mulVec ![3, 11] = ![7.4, 17.2] := by
sorry

end NUMINAMATH_CALUDE_matrix_computation_l3694_369481


namespace NUMINAMATH_CALUDE_first_quadrant_is_well_defined_set_l3694_369452

-- Define the first quadrant
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 > 0}

-- Theorem stating that the FirstQuadrant is a well-defined set
theorem first_quadrant_is_well_defined_set : 
  ∀ p : ℝ × ℝ, Decidable (p ∈ FirstQuadrant) :=
by
  sorry


end NUMINAMATH_CALUDE_first_quadrant_is_well_defined_set_l3694_369452


namespace NUMINAMATH_CALUDE_circle_equation_tangent_line_equation_l3694_369463

-- Define the circle
def circle_center : ℝ × ℝ := (-1, 2)
def line_m (x y : ℝ) : ℝ := x + 2*y + 7

-- Define the point Q
def point_Q : ℝ × ℝ := (1, 6)

-- Theorem for the circle equation
theorem circle_equation : 
  ∃ (r : ℝ), ∀ (x y : ℝ), 
  (x + 1)^2 + (y - 2)^2 = r^2 ∧ 
  (∃ (x₀ y₀ : ℝ), line_m x₀ y₀ = 0 ∧ ((x₀ + 1)^2 + (y₀ - 2)^2 = r^2)) :=
sorry

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∀ (x y : ℝ),
  ((x + 1)^2 + (y - 2)^2 = 20) →
  (point_Q.1 + 1)^2 + (point_Q.2 - 2)^2 = 20 →
  (y - point_Q.2 = -(x - point_Q.1) / 2) ↔ (x + 2*y - 13 = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_line_equation_l3694_369463


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3694_369400

theorem contrapositive_equivalence (p q : Prop) :
  (p → q) → (¬q → ¬p) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3694_369400


namespace NUMINAMATH_CALUDE_apple_plum_ratio_l3694_369457

theorem apple_plum_ratio :
  ∀ (apples plums : ℕ),
    apples = 180 →
    apples + plums = 240 →
    (2 : ℚ) / 5 * (apples + plums) = 96 →
    (apples : ℚ) / plums = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_apple_plum_ratio_l3694_369457


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3694_369404

def U : Set Nat := {0, 1, 2, 3, 4, 5}
def M : Set Nat := {0, 1}

theorem complement_of_M_in_U : 
  (U \ M) = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3694_369404


namespace NUMINAMATH_CALUDE_variance_scaling_l3694_369489

-- Define a function to calculate the variance of a list of numbers
noncomputable def variance (data : List ℝ) : ℝ := sorry

-- Define our theorem
theorem variance_scaling (data : List ℝ) :
  variance data = 4 → variance (List.map (· * 2) data) = 16 := by
  sorry

end NUMINAMATH_CALUDE_variance_scaling_l3694_369489


namespace NUMINAMATH_CALUDE_quadratic_sum_l3694_369430

/-- Given a quadratic expression 4x^2 - 8x - 3, when expressed in the form a(x - h)^2 + k,
    the sum a + h + k equals -2. -/
theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 4*x^2 - 8*x - 3 = a*(x - h)^2 + k) → a + h + k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3694_369430


namespace NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l3694_369442

/-- The amount of money Adam spent on the ferris wheel ride -/
def ferris_wheel_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * ticket_price

/-- Theorem: Adam spent 81 dollars on the ferris wheel ride -/
theorem adam_ferris_wheel_cost :
  ferris_wheel_cost 13 4 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l3694_369442


namespace NUMINAMATH_CALUDE_gcd_of_390_455_546_l3694_369447

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_390_455_546_l3694_369447


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l3694_369438

/-- A square garden with an area of 9 square meters has a perimeter of 12 meters. -/
theorem square_garden_perimeter : 
  ∀ (side : ℝ), side > 0 → side^2 = 9 → 4 * side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l3694_369438


namespace NUMINAMATH_CALUDE_sphere_to_hemisphere_radius_l3694_369422

/-- Given a sphere that transforms into a hemisphere, this theorem relates the radius of the 
    hemisphere to the radius of the original sphere. -/
theorem sphere_to_hemisphere_radius (r : ℝ) (h : r = 5 * Real.rpow 2 (1/3)) : 
  ∃ R : ℝ, R = 5 ∧ (4/3) * Real.pi * R^3 = (2/3) * Real.pi * r^3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_to_hemisphere_radius_l3694_369422


namespace NUMINAMATH_CALUDE_jed_card_collection_l3694_369420

def cards_after_weeks (initial_cards : ℕ) (cards_per_week : ℕ) (cards_given_biweekly : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + weeks * cards_per_week - (weeks / 2) * cards_given_biweekly

theorem jed_card_collection (target_cards : ℕ) : 
  cards_after_weeks 20 6 2 4 = target_cards ∧ target_cards = 40 :=
by sorry

end NUMINAMATH_CALUDE_jed_card_collection_l3694_369420


namespace NUMINAMATH_CALUDE_average_problem_l3694_369413

theorem average_problem (t b c d e : ℝ) : 
  (t + b + c + 14 + 15) / 5 = 12 →
  t = 2 * b →
  (t + b + c + d + e + 14 + 15) / 7 = 12 →
  (t + b + c + 29) / 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l3694_369413


namespace NUMINAMATH_CALUDE_factorial_fraction_is_integer_l3694_369428

/-- Given that m and n are non-negative integers and 0! = 1, 
    prove that (2m)!(2n)! / (m!n!(m+n)!) is an integer. -/
theorem factorial_fraction_is_integer (m n : ℕ) : 
  ∃ k : ℤ, (↑((2*m).factorial * (2*n).factorial) : ℚ) / 
    (↑(m.factorial * n.factorial * (m+n).factorial)) = ↑k :=
sorry

end NUMINAMATH_CALUDE_factorial_fraction_is_integer_l3694_369428


namespace NUMINAMATH_CALUDE_geometric_difference_ratio_l3694_369407

def geometric_difference (a : ℕ+ → ℝ) (d : ℝ) :=
  ∀ n : ℕ+, a (n + 2) / a (n + 1) - a (n + 1) / a n = d

theorem geometric_difference_ratio 
  (a : ℕ+ → ℝ) 
  (h1 : geometric_difference a 2)
  (h2 : a 1 = 1)
  (h3 : a 2 = 1)
  (h4 : a 3 = 3) :
  a 12 / a 10 = 399 := by
sorry

end NUMINAMATH_CALUDE_geometric_difference_ratio_l3694_369407


namespace NUMINAMATH_CALUDE_milburg_population_l3694_369427

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := 5256

/-- The number of children in Milburg -/
def children : ℕ := 2987

/-- The total population of Milburg -/
def total_population : ℕ := grown_ups + children

/-- Theorem stating that the total population of Milburg is 8243 -/
theorem milburg_population : total_population = 8243 := by
  sorry

end NUMINAMATH_CALUDE_milburg_population_l3694_369427


namespace NUMINAMATH_CALUDE_textbook_distribution_is_four_l3694_369482

/-- The number of ways to distribute 8 identical textbooks between the classroom and students,
    given that at least 2 books must be in the classroom and at least 3 books must be with students. -/
def textbook_distribution : ℕ :=
  let total_books : ℕ := 8
  let min_classroom : ℕ := 2
  let min_students : ℕ := 3
  let valid_distributions := List.range (total_books + 1)
    |>.filter (λ classroom_books => 
      classroom_books ≥ min_classroom ∧ 
      (total_books - classroom_books) ≥ min_students)
  valid_distributions.length

/-- Proof that the number of valid distributions is 4 -/
theorem textbook_distribution_is_four : textbook_distribution = 4 := by
  sorry

end NUMINAMATH_CALUDE_textbook_distribution_is_four_l3694_369482


namespace NUMINAMATH_CALUDE_root_of_two_equations_l3694_369405

theorem root_of_two_equations (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^3 + b * k^2 - c * k + d = 0)
  (h2 : -b * k^3 + c * k^2 - d * k + a = 0) :
  k^4 = -1 := by
sorry

end NUMINAMATH_CALUDE_root_of_two_equations_l3694_369405


namespace NUMINAMATH_CALUDE_train_length_l3694_369453

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 9 → ∃ length : ℝ, 
  (length ≥ 74.96 ∧ length ≤ 74.98) ∧ length = speed * (5/18) * time := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3694_369453


namespace NUMINAMATH_CALUDE_trail_mix_nuts_l3694_369424

theorem trail_mix_nuts (walnuts almonds : ℚ) 
  (h1 : walnuts = 0.25)
  (h2 : almonds = 0.25) : 
  walnuts + almonds = 0.50 := by
sorry

end NUMINAMATH_CALUDE_trail_mix_nuts_l3694_369424


namespace NUMINAMATH_CALUDE_kaylee_biscuit_sales_l3694_369490

/-- The number of boxes Kaylee needs to sell -/
def total_boxes : ℕ := 33

/-- The number of lemon biscuit boxes sold -/
def lemon_boxes : ℕ := 12

/-- The number of chocolate biscuit boxes sold -/
def chocolate_boxes : ℕ := 5

/-- The number of oatmeal biscuit boxes sold -/
def oatmeal_boxes : ℕ := 4

/-- The number of additional boxes Kaylee needs to sell -/
def additional_boxes : ℕ := total_boxes - (lemon_boxes + chocolate_boxes + oatmeal_boxes)

theorem kaylee_biscuit_sales : additional_boxes = 12 := by
  sorry

end NUMINAMATH_CALUDE_kaylee_biscuit_sales_l3694_369490


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l3694_369460

/-- The angle in the lateral surface unfolding of a cone, given that its lateral surface area is twice the area of its base. -/
theorem cone_lateral_surface_angle (r : ℝ) (h : r > 0) : 
  let l := 2 * r
  let base_area := π * r^2
  let lateral_area := π * r * l
  lateral_area = 2 * base_area →
  (lateral_area / (π * l^2)) * 360 = 180 :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l3694_369460


namespace NUMINAMATH_CALUDE_integer_fractional_parts_theorem_l3694_369409

theorem integer_fractional_parts_theorem : ∃ (x y : ℝ), 
  (x = ⌊8 - Real.sqrt 11⌋) ∧ 
  (y = 8 - Real.sqrt 11 - ⌊8 - Real.sqrt 11⌋) ∧ 
  (2 * x * y - y^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_integer_fractional_parts_theorem_l3694_369409


namespace NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l3694_369471

theorem gcd_8_factorial_10_factorial :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l3694_369471


namespace NUMINAMATH_CALUDE_triangle_shape_l3694_369486

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

theorem triangle_shape (t : Triangle) 
  (p : Vector2D) 
  (q : Vector2D) 
  (hp : p = ⟨t.c^2, t.a^2⟩) 
  (hq : q = ⟨Real.tan t.C, Real.tan t.A⟩) 
  (hpq : parallel p q) : 
  (t.a = t.c) ∨ (t.b^2 = t.a^2 + t.c^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l3694_369486


namespace NUMINAMATH_CALUDE_license_plate_count_l3694_369419

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of odd digits -/
def num_odd_digits : ℕ := 5

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of prime digits under 10 -/
def num_prime_digits : ℕ := 4

/-- The total number of license plates -/
def total_license_plates : ℕ := num_letters ^ 3 * num_odd_digits * num_digits * num_prime_digits

theorem license_plate_count :
  total_license_plates = 351520 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l3694_369419


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3694_369499

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point on a hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Left focus of a hyperbola -/
def left_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Right focus of a hyperbola -/
def right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_min : ∀ p : HyperbolaPoint h, 
    (distance (p.x, p.y) (right_focus h))^2 / distance (p.x, p.y) (left_focus h) ≥ 9 * h.a) :
  eccentricity h = 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3694_369499


namespace NUMINAMATH_CALUDE_max_stamps_is_125_l3694_369432

/-- The maximum number of stamps that can be purchased with a given budget. -/
def max_stamps (budget : ℕ) (price_low : ℕ) (price_high : ℕ) (threshold : ℕ) : ℕ :=
  max (min (budget / price_high) threshold) (budget / price_low)

/-- Proof that 125 stamps is the maximum number that can be purchased with 5000 cents,
    given the pricing conditions. -/
theorem max_stamps_is_125 :
  max_stamps 5000 40 45 100 = 125 := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_is_125_l3694_369432


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_522_l3694_369470

theorem sin_n_equals_cos_522 :
  ∃ n : ℤ, -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (522 * π / 180) :=
by
  use -72
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_522_l3694_369470


namespace NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l3694_369450

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 2 / x ≥ 5 ∧
  (3 * Real.sqrt x + 2 / x = 5 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l3694_369450


namespace NUMINAMATH_CALUDE_smallest_k_for_negative_three_in_range_l3694_369408

-- Define the function g
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + k

-- State the theorem
theorem smallest_k_for_negative_three_in_range :
  (∃ k₀ : ℝ, (∀ k : ℝ, (∃ x : ℝ, g k x = -3) → k ≥ k₀) ∧
             (∃ x : ℝ, g k₀ x = -3) ∧
             k₀ = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_negative_three_in_range_l3694_369408


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3694_369412

theorem inequality_solution_set (x : ℝ) : (-2 * x - 1 < -1) ↔ (x > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3694_369412


namespace NUMINAMATH_CALUDE_percentage_error_calculation_l3694_369459

theorem percentage_error_calculation : 
  let correct_operation (x : ℝ) := 3 * x
  let incorrect_operation (x : ℝ) := x / 5
  let error (x : ℝ) := correct_operation x - incorrect_operation x
  let percentage_error (x : ℝ) := (error x / correct_operation x) * 100
  ∀ x : ℝ, x ≠ 0 → percentage_error x = (14 / 15) * 100 := by
sorry

end NUMINAMATH_CALUDE_percentage_error_calculation_l3694_369459


namespace NUMINAMATH_CALUDE_probability_of_three_positive_answers_l3694_369484

/-- The probability of getting exactly k successes in n trials,
    where the probability of success on each trial is p. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of questions asked -/
def total_questions : ℕ := 7

/-- The number of positive answers we're interested in -/
def positive_answers : ℕ := 3

/-- The probability of a positive answer for each question -/
def positive_probability : ℚ := 3/7

theorem probability_of_three_positive_answers :
  binomial_probability total_questions positive_answers positive_probability = 242112/823543 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_positive_answers_l3694_369484


namespace NUMINAMATH_CALUDE_remainder_theorem_l3694_369446

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^4 - 4*x^2 + 7*x - 8

-- State the theorem
theorem remainder_theorem :
  ∃ (Q : ℝ → ℝ), P = λ x => (x - 3) * Q x + 50 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3694_369446


namespace NUMINAMATH_CALUDE_starship_age_conversion_l3694_369478

/-- Converts an octal digit to decimal --/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- Converts an octal number to decimal --/
def octal_to_decimal_number (octal : List Nat) : Nat :=
  octal.enum.foldr (fun (i, digit) acc => acc + octal_to_decimal digit * (8^i)) 0

theorem starship_age_conversion :
  octal_to_decimal_number [6, 7, 2, 4] = 3540 := by
  sorry

end NUMINAMATH_CALUDE_starship_age_conversion_l3694_369478


namespace NUMINAMATH_CALUDE_second_quadrant_complex_l3694_369433

theorem second_quadrant_complex (a b : ℝ) : 
  (Complex.ofReal a + Complex.I * Complex.ofReal b).im > 0 ∧ 
  (Complex.ofReal a + Complex.I * Complex.ofReal b).re < 0 → 
  a < 0 ∧ b > 0 := by sorry

end NUMINAMATH_CALUDE_second_quadrant_complex_l3694_369433


namespace NUMINAMATH_CALUDE_complex_product_negative_l3694_369468

theorem complex_product_negative (a : ℝ) :
  let z : ℂ := (a + Complex.I) * (-3 + a * Complex.I)
  (z.re < 0) → a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_product_negative_l3694_369468


namespace NUMINAMATH_CALUDE_ampersand_eight_two_squared_l3694_369477

def ampersand (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem ampersand_eight_two_squared :
  (ampersand 8 2)^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_eight_two_squared_l3694_369477


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l3694_369435

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 2*x - 4*y + 1 = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l3694_369435


namespace NUMINAMATH_CALUDE_not_divisible_seven_digit_numbers_l3694_369479

def is_seven_digit (n : ℕ) : Prop := 1000000 ≤ n ∧ n ≤ 9999999

def uses_digits_1_to_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 7 → ∃ k : ℕ, n / (10^k) % 10 = d

theorem not_divisible_seven_digit_numbers (A B : ℕ) :
  is_seven_digit A ∧ is_seven_digit B ∧
  uses_digits_1_to_7 A ∧ uses_digits_1_to_7 B ∧
  A ≠ B →
  ¬(∃ k : ℕ, A = k * B) :=
sorry

end NUMINAMATH_CALUDE_not_divisible_seven_digit_numbers_l3694_369479


namespace NUMINAMATH_CALUDE_stream_speed_relationship_l3694_369451

-- Define the boat speeds and distances
def low_speed : ℝ := 20
def high_speed : ℝ := 40
def downstream_distance : ℝ := 26
def upstream_distance : ℝ := 14

-- Define the stream speeds as variables
variable (x y : ℝ)

-- Define the theorem
theorem stream_speed_relationship :
  (downstream_distance / (low_speed + x) = upstream_distance / (high_speed - y)) →
  380 = 7 * x + 13 * y :=
by
  sorry


end NUMINAMATH_CALUDE_stream_speed_relationship_l3694_369451


namespace NUMINAMATH_CALUDE_base7_to_base10_65432_l3694_369449

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (init := 0) fun acc (i, d) => acc + d * (7 ^ (digits.length - 1 - i))

/-- The base-7 representation of the number --/
def base7Number : List Nat := [6, 5, 4, 3, 2]

/-- Theorem: The base-10 equivalent of 65432 in base-7 is 16340 --/
theorem base7_to_base10_65432 : base7ToBase10 base7Number = 16340 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_65432_l3694_369449


namespace NUMINAMATH_CALUDE_third_factorial_is_seven_l3694_369411

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def gcd_of_three (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

theorem third_factorial_is_seven (b : ℕ) (x : ℕ) 
  (h1 : b = 9) 
  (h2 : gcd_of_three (factorial (b - 2)) (factorial (b + 1)) (factorial x) = 5040) : 
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_factorial_is_seven_l3694_369411


namespace NUMINAMATH_CALUDE_candy_duration_l3694_369485

theorem candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) :
  neighbors_candy = 66 →
  sister_candy = 15 →
  daily_consumption = 9 →
  (neighbors_candy + sister_candy) / daily_consumption = 9 :=
by sorry

end NUMINAMATH_CALUDE_candy_duration_l3694_369485


namespace NUMINAMATH_CALUDE_workers_count_l3694_369491

-- Define the work function
def work (workers : ℕ) (hours : ℕ) : ℕ := workers * hours

-- Define the problem parameters
def initial_hours : ℕ := 8
def initial_depth : ℕ := 30
def second_hours : ℕ := 6
def second_depth : ℕ := 55
def extra_workers : ℕ := 65

theorem workers_count :
  ∃ (W : ℕ), 
    (work W initial_hours) * second_depth = 
    (work (W + extra_workers) second_hours) * initial_depth ∧
    W = 45 := by
  sorry

end NUMINAMATH_CALUDE_workers_count_l3694_369491


namespace NUMINAMATH_CALUDE_probability_one_second_class_l3694_369493

def total_products : ℕ := 12
def first_class_products : ℕ := 10
def second_class_products : ℕ := 2
def selected_products : ℕ := 4

theorem probability_one_second_class :
  (Nat.choose second_class_products 1 * Nat.choose first_class_products (selected_products - 1)) /
  (Nat.choose total_products selected_products) = 16 / 33 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_second_class_l3694_369493


namespace NUMINAMATH_CALUDE_initial_carrots_count_l3694_369439

/-- Proves that the initial number of carrots is 300 given the problem conditions --/
theorem initial_carrots_count : ℕ :=
  let initial_carrots : ℕ := 300
  let before_lunch_fraction : ℚ := 2/5
  let after_lunch_fraction : ℚ := 3/5
  let unused_carrots : ℕ := 72

  have h1 : (1 - before_lunch_fraction) * (1 - after_lunch_fraction) * initial_carrots = unused_carrots := by sorry

  initial_carrots


end NUMINAMATH_CALUDE_initial_carrots_count_l3694_369439


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_l3694_369461

theorem recurring_decimal_fraction (a b : ℚ) :
  a = 36 * (1 / 99) ∧ b = 12 * (1 / 99) → a / b = 3 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_l3694_369461


namespace NUMINAMATH_CALUDE_A_greater_than_B_l3694_369440

def A : ℕ → ℕ
  | 0 => 3
  | n+1 => 3^(A n)

def B : ℕ → ℕ
  | 0 => 8
  | n+1 => 8^(B n)

theorem A_greater_than_B (n : ℕ) : A (n + 1) > B n := by
  sorry

end NUMINAMATH_CALUDE_A_greater_than_B_l3694_369440


namespace NUMINAMATH_CALUDE_max_homework_time_l3694_369441

/-- The time Max spent on biology homework -/
def biology_time : ℕ := 20

/-- The time Max spent on history homework -/
def history_time : ℕ := 2 * biology_time

/-- The time Max spent on geography homework -/
def geography_time : ℕ := 3 * history_time

/-- The total time Max spent on homework -/
def total_time : ℕ := 180

theorem max_homework_time : 
  biology_time + history_time + geography_time = total_time ∧ 
  biology_time = 20 := by sorry

end NUMINAMATH_CALUDE_max_homework_time_l3694_369441


namespace NUMINAMATH_CALUDE_book_cost_problem_l3694_369465

/-- Proves that given two books with a total cost of 480, where one is sold at a 15% loss 
and the other at a 19% gain, and both are sold at the same price, 
the cost of the book sold at a loss is 280. -/
theorem book_cost_problem (c1 c2 : ℝ) : 
  c1 + c2 = 480 →
  c1 * 0.85 = c2 * 1.19 →
  c1 = 280 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l3694_369465


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3694_369445

/-- A regular polygon with side length 7 and exterior angle 90 degrees has perimeter 28. -/
theorem regular_polygon_perimeter :
  ∀ (n : ℕ) (s : ℝ) (θ : ℝ),
    n > 0 →
    s = 7 →
    θ = 90 →
    (360 : ℝ) / n = θ →
    n * s = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3694_369445


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l3694_369402

def a (n : ℕ) : ℕ := n.factorial + 3 * n

theorem max_gcd_consecutive_terms :
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ 3 ∧
  (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = 3) :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l3694_369402


namespace NUMINAMATH_CALUDE_constant_difference_of_equal_derivatives_l3694_369458

theorem constant_difference_of_equal_derivatives 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h : ∀ x, deriv f x = deriv g x) : 
  ∃ C, ∀ x, f x - g x = C :=
sorry

end NUMINAMATH_CALUDE_constant_difference_of_equal_derivatives_l3694_369458


namespace NUMINAMATH_CALUDE_spinner_probability_theorem_l3694_369466

/-- Represents the probability of landing on each part of a circular spinner -/
structure SpinnerProbabilities where
  A : ℚ
  B : ℚ
  C : ℚ
  D : ℚ

/-- Theorem: If a circular spinner has probabilities 1/4 for A, 1/3 for B, and 1/6 for D,
    then the probability for C is 1/4 -/
theorem spinner_probability_theorem (sp : SpinnerProbabilities) 
  (hA : sp.A = 1/4)
  (hB : sp.B = 1/3)
  (hD : sp.D = 1/6)
  (hSum : sp.A + sp.B + sp.C + sp.D = 1) :
  sp.C = 1/4 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_theorem_l3694_369466


namespace NUMINAMATH_CALUDE_S_intersect_T_l3694_369496

def S : Set ℝ := {x | (x + 5) / (5 - x) > 0}
def T : Set ℝ := {x | x^2 + 4*x - 21 < 0}

theorem S_intersect_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_l3694_369496


namespace NUMINAMATH_CALUDE_watches_synchronize_after_1600_days_l3694_369431

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The rate at which Glafira's watch gains time (in seconds per day) -/
def glafira_gain : ℕ := 36

/-- The rate at which Gavrila's watch loses time (in seconds per day) -/
def gavrila_loss : ℕ := 18

/-- The theorem stating that the watches will display the correct time simultaneously after 1600 days -/
theorem watches_synchronize_after_1600_days :
  (seconds_per_day * 1600) % (glafira_gain + gavrila_loss) = 0 := by
  sorry

end NUMINAMATH_CALUDE_watches_synchronize_after_1600_days_l3694_369431


namespace NUMINAMATH_CALUDE_horse_journey_l3694_369467

theorem horse_journey (a₁ : ℚ) : 
  (a₁ * (1 - (1/2)^7) / (1 - 1/2) = 700) → 
  (a₁ * (1/2)^6 = 700/127) := by
sorry

end NUMINAMATH_CALUDE_horse_journey_l3694_369467


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l3694_369423

theorem sum_of_x_and_y_equals_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l3694_369423


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l3694_369498

/-- Given a boat's upstream and downstream travel times, 
    prove the ratio of current speed to boat speed in still water -/
theorem boat_speed_ratio 
  (distance : ℝ) 
  (upstream_time downstream_time : ℝ) 
  (h1 : distance = 15)
  (h2 : upstream_time = 5)
  (h3 : downstream_time = 3) :
  ∃ (boat_speed current_speed : ℝ),
    boat_speed > 0 ∧
    current_speed > 0 ∧
    distance / upstream_time = boat_speed - current_speed ∧
    distance / downstream_time = boat_speed + current_speed ∧
    current_speed / boat_speed = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l3694_369498


namespace NUMINAMATH_CALUDE_equation_equivalence_l3694_369403

theorem equation_equivalence (x : ℝ) : 
  (x - 1) / 2 - (2 * x + 3) / 3 = 1 ↔ 3 * (x - 1) - 2 * (2 * x + 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3694_369403


namespace NUMINAMATH_CALUDE_a_greater_than_b_l3694_369462

theorem a_greater_than_b (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_a_eq : a^n = a + 1) 
  (h_b_eq : b^(2*n) = b + 3*a) : 
  a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l3694_369462


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3694_369437

/-- Represents the number of animals in the farm -/
structure FarmAnimals where
  pigs : ℕ
  hens : ℕ

/-- The total number of legs in the farm -/
def totalLegs (animals : FarmAnimals) : ℕ :=
  4 * animals.pigs + 2 * animals.hens

/-- The total number of heads in the farm -/
def totalHeads (animals : FarmAnimals) : ℕ :=
  animals.pigs + animals.hens

/-- The condition given in the problem -/
def satisfiesCondition (animals : FarmAnimals) : Prop :=
  totalLegs animals = 3 * totalHeads animals + 36

theorem infinitely_many_solutions : 
  ∀ n : ℕ, ∃ animals : FarmAnimals, satisfiesCondition animals ∧ animals.pigs = n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3694_369437


namespace NUMINAMATH_CALUDE_expression_value_l3694_369472

theorem expression_value :
  let x : ℚ := 2
  let y : ℚ := 3
  let z : ℚ := 4
  (4 * x^2 - 6 * y^3 + z^2) / (5 * x + 7 * z - 3 * y^2) = -130 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3694_369472


namespace NUMINAMATH_CALUDE_circle_center_on_line_l3694_369448

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_center_on_line (x y : ℚ) : 
  (5 * x - 4 * y = 40) ∧ 
  (5 * x - 4 * y = -20) ∧ 
  (3 * x - y = 0) →
  x = -10/7 ∧ y = -30/7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_on_line_l3694_369448


namespace NUMINAMATH_CALUDE_problem_solution_l3694_369443

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem problem_solution :
  (∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3)) ∧
  (∀ a : ℝ, (a > 0 ∧ (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x)) ↔ (1 ≤ a ∧ a ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3694_369443


namespace NUMINAMATH_CALUDE_field_reduction_l3694_369406

theorem field_reduction (L W : ℝ) (h : L > 0 ∧ W > 0) :
  ∃ x : ℝ, x > 0 ∧ x < 100 ∧ 
  (1 - x / 100) * (1 - x / 100) * (L * W) = (1 - 0.64) * (L * W) →
  x = 40 := by
sorry

end NUMINAMATH_CALUDE_field_reduction_l3694_369406


namespace NUMINAMATH_CALUDE_largest_rational_satisfying_equation_l3694_369456

theorem largest_rational_satisfying_equation :
  ∀ x : ℚ, |x - 7/2| = 25/2 → x ≤ 16 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_rational_satisfying_equation_l3694_369456


namespace NUMINAMATH_CALUDE_queens_attack_probability_l3694_369497

/-- The size of the chessboard -/
def boardSize : Nat := 8

/-- The total number of squares on the chessboard -/
def totalSquares : Nat := boardSize * boardSize

/-- The number of ways to choose two different squares -/
def totalChoices : Nat := totalSquares * (totalSquares - 1) / 2

/-- The number of ways two queens can attack each other -/
def attackingChoices : Nat := 
  -- Same row
  boardSize * (boardSize * (boardSize - 1) / 2) +
  -- Same column
  boardSize * (boardSize * (boardSize - 1) / 2) +
  -- Same diagonal (main and anti-diagonals)
  (2 * (1 + 3 + 6 + 10 + 15 + 21) + 28)

/-- The probability of two queens attacking each other -/
def attackProbability : Rat := attackingChoices / totalChoices

theorem queens_attack_probability : 
  attackProbability = 7 / 24 := by sorry

end NUMINAMATH_CALUDE_queens_attack_probability_l3694_369497


namespace NUMINAMATH_CALUDE_unique_cube_difference_nineteen_l3694_369469

theorem unique_cube_difference_nineteen :
  ∀ x y : ℕ, x^3 - y^3 = 19 → x = 3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_difference_nineteen_l3694_369469
