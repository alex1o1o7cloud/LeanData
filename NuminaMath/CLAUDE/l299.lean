import Mathlib

namespace NUMINAMATH_CALUDE_correct_num_cows_l299_29904

/-- Represents the number of dairy cows owned by the breeder -/
def num_cows : ℕ := 52

/-- Represents the amount of milk (in oz) one cow produces per day -/
def milk_per_cow_per_day : ℕ := 1000

/-- Represents the total amount of milk (in oz) produced in a week -/
def total_milk_per_week : ℕ := 364000

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem stating that the number of cows is correct given the milk production data -/
theorem correct_num_cows : 
  num_cows * milk_per_cow_per_day * days_in_week = total_milk_per_week :=
sorry

end NUMINAMATH_CALUDE_correct_num_cows_l299_29904


namespace NUMINAMATH_CALUDE_projects_for_30_points_l299_29931

/-- Calculates the minimum number of projects required to earn a given number of study points -/
def min_projects (total_points : ℕ) : ℕ :=
  let block_size := 6
  let num_blocks := (total_points + block_size - 1) / block_size
  (num_blocks * (num_blocks + 1) * block_size) / 2

/-- Theorem stating that 90 projects are required to earn 30 study points -/
theorem projects_for_30_points :
  min_projects 30 = 90 := by
  sorry


end NUMINAMATH_CALUDE_projects_for_30_points_l299_29931


namespace NUMINAMATH_CALUDE_radius_of_special_polygon_l299_29972

/-- A regular polygon with the given properties -/
structure RegularPolygon where
  side_length : ℝ
  interior_angle_sum : ℝ
  exterior_angle_sum : ℝ

/-- The radius of a regular polygon -/
def radius (p : RegularPolygon) : ℝ := sorry

/-- The theorem to be proved -/
theorem radius_of_special_polygon :
  ∀ (p : RegularPolygon),
    p.side_length = 2 →
    p.interior_angle_sum = 2 * p.exterior_angle_sum →
    radius p = 2 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_special_polygon_l299_29972


namespace NUMINAMATH_CALUDE_log_equality_l299_29909

theorem log_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 4*y^2 = 12*x*y) :
  Real.log (x + 2*y) - 2 * Real.log 2 = 0.5 * (Real.log x + Real.log y) := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l299_29909


namespace NUMINAMATH_CALUDE_prime_square_plus_two_prime_l299_29935

theorem prime_square_plus_two_prime (P : ℕ) : 
  Nat.Prime P → Nat.Prime (P^2 + 2) → P^4 + 1921 = 2002 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_plus_two_prime_l299_29935


namespace NUMINAMATH_CALUDE_checkerboard_probability_l299_29940

/-- The size of one side of the square checkerboard -/
def boardSize : ℕ := 10

/-- The total number of squares on the checkerboard -/
def totalSquares : ℕ := boardSize * boardSize

/-- The number of squares on the perimeter of the checkerboard -/
def perimeterSquares : ℕ := 4 * boardSize - 4

/-- The number of squares not touching the outer edge -/
def innerSquares : ℕ := totalSquares - perimeterSquares

/-- The probability of choosing a square not touching the outer edge -/
def innerProbability : ℚ := innerSquares / totalSquares

theorem checkerboard_probability :
  innerProbability = 16 / 25 := by sorry

end NUMINAMATH_CALUDE_checkerboard_probability_l299_29940


namespace NUMINAMATH_CALUDE_min_value_on_interval_l299_29996

/-- A function f(x) = x^2 + x + a with a maximum value of 2 on [-1, 1] -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 + x + a

/-- The maximum value of f on [-1, 1] is 2 -/
axiom max_value_2 (a : ℝ) : ∃ x ∈ Set.Icc (-1) 1, ∀ y ∈ Set.Icc (-1) 1, f a x ≥ f a y ∧ f a x = 2

/-- The theorem to prove -/
theorem min_value_on_interval (a : ℝ) : 
  (∃ x ∈ Set.Icc (-1) 1, ∀ y ∈ Set.Icc (-1) 1, f a x ≥ f a y ∧ f a x = 2) →
  ∃ z ∈ Set.Icc (-1) 1, ∀ w ∈ Set.Icc (-1) 1, f a z ≤ f a w ∧ f a z = -1/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l299_29996


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l299_29915

theorem complex_magnitude_problem (z : ℂ) : 
  z = (Complex.I : ℂ) / (1 + Complex.I) → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l299_29915


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l299_29900

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10 % 10) + (n % 10)

theorem sum_of_digits_of_power : sum_of_digits ((3 + 4) ^ 11) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l299_29900


namespace NUMINAMATH_CALUDE_pepperoni_coverage_fraction_l299_29943

-- Define the pizza and pepperoni properties
def pizza_diameter : ℝ := 12
def pepperoni_across_diameter : ℕ := 6
def total_pepperoni : ℕ := 24

-- Theorem statement
theorem pepperoni_coverage_fraction :
  let pepperoni_diameter : ℝ := pizza_diameter / pepperoni_across_diameter
  let pepperoni_radius : ℝ := pepperoni_diameter / 2
  let pepperoni_area : ℝ := π * pepperoni_radius ^ 2
  let total_pepperoni_area : ℝ := pepperoni_area * total_pepperoni
  let pizza_radius : ℝ := pizza_diameter / 2
  let pizza_area : ℝ := π * pizza_radius ^ 2
  total_pepperoni_area / pizza_area = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_fraction_l299_29943


namespace NUMINAMATH_CALUDE_total_seeds_eaten_l299_29981

def player1_seeds : ℕ := 78
def player2_seeds : ℕ := 53
def extra_seeds : ℕ := 30

def player3_seeds : ℕ := player2_seeds + extra_seeds

theorem total_seeds_eaten :
  player1_seeds + player2_seeds + player3_seeds = 214 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_eaten_l299_29981


namespace NUMINAMATH_CALUDE_expenditure_for_specific_hall_l299_29989

/-- Calculates the total expenditure for covering the interior of a rectangular hall with mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := 2 * floor_area + wall_area
  total_area * cost_per_sqm

/-- Proves that the total expenditure for covering the interior of a specific rectangular hall with mat is Rs. 9500 -/
theorem expenditure_for_specific_hall :
  total_expenditure 20 15 5 10 = 9500 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_for_specific_hall_l299_29989


namespace NUMINAMATH_CALUDE_locus_is_ray_l299_29991

/-- The locus of point P satisfying |PM| - |PN| = 4 is a ray -/
theorem locus_is_ray (M N P : ℝ × ℝ) :
  M = (-2, 0) →
  N = (2, 0) →
  abs (P.1 - M.1) + abs (P.2 - M.2) - (abs (P.1 - N.1) + abs (P.2 - N.2)) = 4 →
  P.1 ≥ 2 ∧ P.2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_is_ray_l299_29991


namespace NUMINAMATH_CALUDE_paula_shopping_remaining_l299_29914

/-- Given an initial amount, cost of shirts, number of shirts, and cost of pants,
    calculate the remaining amount after purchases. -/
def remaining_amount (initial : ℕ) (shirt_cost : ℕ) (num_shirts : ℕ) (pants_cost : ℕ) : ℕ :=
  initial - (shirt_cost * num_shirts + pants_cost)

/-- Theorem stating that given the specific values from the problem,
    the remaining amount is 74. -/
theorem paula_shopping_remaining : remaining_amount 109 11 2 13 = 74 := by
  sorry

end NUMINAMATH_CALUDE_paula_shopping_remaining_l299_29914


namespace NUMINAMATH_CALUDE_one_thirds_in_eight_halves_l299_29977

theorem one_thirds_in_eight_halves : (8 / 2) / (1 / 3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_eight_halves_l299_29977


namespace NUMINAMATH_CALUDE_cosine_symmetric_minimum_l299_29984

open Real

theorem cosine_symmetric_minimum (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, 0 < x → x < 2 → f x = cos (π * x)) →
  0 < a → a < 2 →
  0 < b → b < 2 →
  a ≠ b →
  f a = f b →
  (∀ x y, 0 < x → x < 2 → 0 < y → y < 2 → x ≠ y → f x = f y → 1/x + 4/y ≥ 9/2) →
  ∃ x y, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x ≠ y ∧ f x = f y ∧ 1/x + 4/y = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_cosine_symmetric_minimum_l299_29984


namespace NUMINAMATH_CALUDE_common_tangent_sum_l299_29903

theorem common_tangent_sum (a b c : ℕ+) : 
  let P₁ : ℝ → ℝ := λ x => x^2 + 12/5
  let P₂ : ℝ → ℝ := λ y => y^2 + 99/10
  let L : ℝ → ℝ → Prop := λ x y => a*x + b*y = c
  Nat.gcd a (Nat.gcd b c) = 1 →
  (∃ x₀ y₀ : ℝ, L x₀ y₀ ∧ y₀ = P₁ x₀ ∧ 
    ∀ x y, L x y → y ≥ P₁ x) →
  (∃ x₁ y₁ : ℝ, L x₁ y₁ ∧ x₁ = P₂ y₁ ∧ 
    ∀ x y, L x y → x ≥ P₂ y) →
  (∃ q : ℚ, (b : ℝ) / (a : ℝ) = q) →
  a + b + c = 11 := by
sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l299_29903


namespace NUMINAMATH_CALUDE_total_pears_is_fifteen_l299_29936

/-- The number of pears Mike picked -/
def mike_pears : ℕ := 8

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 7

/-- The total number of pears picked -/
def total_pears : ℕ := mike_pears + jason_pears

theorem total_pears_is_fifteen : total_pears = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_is_fifteen_l299_29936


namespace NUMINAMATH_CALUDE_modular_congruence_l299_29908

theorem modular_congruence (x : ℤ) :
  (5 * x + 9) % 16 = 3 → (3 * x + 8) % 16 = 14 := by
sorry

end NUMINAMATH_CALUDE_modular_congruence_l299_29908


namespace NUMINAMATH_CALUDE_monday_appointment_duration_l299_29971

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℝ := 20

/-- Number of appointments on Monday -/
def monday_appointments : ℕ := 5

/-- Hours worked on Tuesday -/
def tuesday_hours : ℝ := 3

/-- Hours worked on Thursday -/
def thursday_hours : ℝ := 4

/-- Hours worked on Saturday -/
def saturday_hours : ℝ := 6

/-- Total earnings for the week in dollars -/
def total_earnings : ℝ := 410

/-- Theorem: Given Amanda's schedule and earnings, each of her Monday appointments lasts 1.5 hours -/
theorem monday_appointment_duration :
  (total_earnings - hourly_rate * (tuesday_hours + thursday_hours + saturday_hours)) / 
  (hourly_rate * monday_appointments) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_monday_appointment_duration_l299_29971


namespace NUMINAMATH_CALUDE_polynomial_expansion_l299_29953

theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 6 * z^2 - 5 * z - 4) * (4 * z^4 - 3 * z^2 + 7) =
  12 * z^7 + 24 * z^6 - 29 * z^5 - 34 * z^4 + 36 * z^3 + 54 * z^2 + 35 * z - 28 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l299_29953


namespace NUMINAMATH_CALUDE_complement_of_intersection_equals_universe_l299_29913

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | x < -1}

-- State the theorem
theorem complement_of_intersection_equals_universe :
  (A ∩ B)ᶜ = U := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_equals_universe_l299_29913


namespace NUMINAMATH_CALUDE_solve_for_a_l299_29926

theorem solve_for_a : ∀ a : ℝ, (3 * 2 - a = -2 + 7) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l299_29926


namespace NUMINAMATH_CALUDE_intersection_sum_l299_29995

/-- Given two lines that intersect at (3,6), prove that a + b = 6 -/
theorem intersection_sum (a b : ℝ) : 
  (3 = (1/3) * 6 + a) → 
  (6 = (1/3) * 3 + b) → 
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l299_29995


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l299_29960

theorem sum_of_a_and_b (a b : ℝ) : (Real.sqrt (a + 3) + abs (b - 5) = 0) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l299_29960


namespace NUMINAMATH_CALUDE_school_boys_count_l299_29919

theorem school_boys_count (muslim_percent : ℝ) (hindu_percent : ℝ) (sikh_percent : ℝ) (other_count : ℕ) : 
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  other_count = 126 →
  ∃ (total : ℕ), 
    (muslim_percent + hindu_percent + sikh_percent + (other_count : ℝ) / total = 1) ∧
    total = 700 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l299_29919


namespace NUMINAMATH_CALUDE_special_operation_result_l299_29933

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := a + b - 1

-- Define the odot operation
def odot (a b : ℚ) : ℚ := a * b - a^2

-- Theorem statement
theorem special_operation_result :
  odot (-2) (triangle 8 (-3)) = -12 := by sorry

end NUMINAMATH_CALUDE_special_operation_result_l299_29933


namespace NUMINAMATH_CALUDE_square_circle_perimeter_l299_29951

/-- Given a square with perimeter 28 cm, the perimeter of a circle whose radius is equal to the side of the square is 14π cm. -/
theorem square_circle_perimeter (square_perimeter : ℝ) (h : square_perimeter = 28) :
  let square_side := square_perimeter / 4
  let circle_radius := square_side
  let circle_perimeter := 2 * Real.pi * circle_radius
  circle_perimeter = 14 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_square_circle_perimeter_l299_29951


namespace NUMINAMATH_CALUDE_percentage_difference_l299_29997

theorem percentage_difference (x y : ℝ) (h : x = 7 * y) :
  (1 - y / x) * 100 = (1 - 1 / 7) * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l299_29997


namespace NUMINAMATH_CALUDE_area_circle_radius_5_l299_29987

/-- The area of a circle with radius 5 meters is 25π square meters. -/
theorem area_circle_radius_5 : 
  ∀ (π : ℝ), π > 0 → (5 : ℝ) ^ 2 * π = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_area_circle_radius_5_l299_29987


namespace NUMINAMATH_CALUDE_trip_participants_l299_29954

theorem trip_participants :
  ∃ (men women children : ℕ),
    men + women + children = 150 ∧
    17 * men + 14 * women + 9 * children = 1530 ∧
    children < 120 ∧
    men = 5 ∧
    women = 28 ∧
    children = 117 := by
  sorry

end NUMINAMATH_CALUDE_trip_participants_l299_29954


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l299_29956

theorem pure_imaginary_complex_number (m : ℝ) : 
  (m^2 - 9 : ℂ) + (m + 3 : ℂ) * Complex.I = Complex.I * (m + 3 : ℂ) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l299_29956


namespace NUMINAMATH_CALUDE_garden_feet_count_l299_29947

/-- The number of feet for a dog -/
def dog_feet : ℕ := 4

/-- The number of feet for a duck -/
def duck_feet : ℕ := 2

/-- The number of dogs in the garden -/
def num_dogs : ℕ := 6

/-- The number of ducks in the garden -/
def num_ducks : ℕ := 2

/-- The total number of feet in the garden -/
def total_feet : ℕ := num_dogs * dog_feet + num_ducks * duck_feet

theorem garden_feet_count : total_feet = 28 := by
  sorry

end NUMINAMATH_CALUDE_garden_feet_count_l299_29947


namespace NUMINAMATH_CALUDE_cave_depth_l299_29974

theorem cave_depth (total_depth remaining_distance current_depth : ℕ) 
  (h1 : total_depth = 1218)
  (h2 : remaining_distance = 369)
  (h3 : current_depth = total_depth - remaining_distance) :
  current_depth = 849 := by
  sorry

end NUMINAMATH_CALUDE_cave_depth_l299_29974


namespace NUMINAMATH_CALUDE_smallest_third_term_of_gp_l299_29923

theorem smallest_third_term_of_gp (a b c : ℝ) : 
  (∃ d : ℝ, a = 9 ∧ b = 9 + d ∧ c = 9 + 2*d) →  -- arithmetic progression
  (∃ r : ℝ, 9 * (c + 20) = (b + 2)^2) →  -- geometric progression after modification
  (∃ x : ℝ, x ≥ c + 20 ∧ 
    ∀ y : ℝ, (∃ d : ℝ, 9 = 9 ∧ 9 + d + 2 = (9 * (9 + 2*d + 20))^(1/2) ∧ 9 + 2*d + 20 = y) 
    → x ≤ y) →
  1 ≤ c + 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_gp_l299_29923


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l299_29925

/-- Given a complex shape composed of rectangles, prove that the perimeter of the non-shaded region is 28 inches. -/
theorem non_shaded_perimeter (total_area shaded_area : ℝ) (h1 : total_area = 160) (h2 : shaded_area = 120) : ∃ (length width : ℝ), length * width = total_area - shaded_area ∧ 2 * (length + width) = 28 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l299_29925


namespace NUMINAMATH_CALUDE_cone_to_cylinder_volume_ratio_l299_29968

/-- The ratio of the volume of a cone to the volume of a cylinder with shared base radius -/
theorem cone_to_cylinder_volume_ratio 
  (r : ℝ) (h_c h_n : ℝ) 
  (hr : r = 5) 
  (hh_c : h_c = 20) 
  (hh_n : h_n = 10) : 
  (1 / 3 * π * r^2 * h_n) / (π * r^2 * h_c) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_to_cylinder_volume_ratio_l299_29968


namespace NUMINAMATH_CALUDE_smallest_nonfactor_product_of_48_l299_29929

theorem smallest_nonfactor_product_of_48 (a b : ℕ) : 
  a ≠ b ∧ 
  a > 0 ∧ 
  b > 0 ∧ 
  48 % a = 0 ∧ 
  48 % b = 0 ∧ 
  48 % (a * b) ≠ 0 →
  ∀ x y : ℕ, 
    x ≠ y ∧ 
    x > 0 ∧ 
    y > 0 ∧ 
    48 % x = 0 ∧ 
    48 % y = 0 ∧ 
    48 % (x * y) ≠ 0 →
    a * b ≤ x * y ∧
    a * b = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonfactor_product_of_48_l299_29929


namespace NUMINAMATH_CALUDE_x_value_when_y_is_one_l299_29980

theorem x_value_when_y_is_one (x y : ℝ) :
  y = 1 / (3 * x + 1) → y = 1 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_one_l299_29980


namespace NUMINAMATH_CALUDE_distance_specific_point_to_line_l299_29985

/-- The distance from a point to a line in 3D space -/
def distance_point_to_line (point : ℝ × ℝ × ℝ) (line_point : ℝ × ℝ × ℝ) (line_direction : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem: The distance from (2, -1, 4) to the line (4, 3, 9) + t(1, -1, 3) is 65/11 -/
theorem distance_specific_point_to_line :
  let point : ℝ × ℝ × ℝ := (2, -1, 4)
  let line_point : ℝ × ℝ × ℝ := (4, 3, 9)
  let line_direction : ℝ × ℝ × ℝ := (1, -1, 3)
  distance_point_to_line point line_point line_direction = 65 / 11 :=
by sorry

end NUMINAMATH_CALUDE_distance_specific_point_to_line_l299_29985


namespace NUMINAMATH_CALUDE_eliana_steps_l299_29975

/-- The number of steps Eliana walked on the first day before adding more steps -/
def initial_steps : ℕ := 200

/-- The number of additional steps Eliana walked on the first day -/
def additional_steps : ℕ := 300

/-- The number of extra steps Eliana walked on the third day compared to the second day -/
def extra_steps : ℕ := 100

/-- The total number of steps Eliana walked over the three days -/
def total_steps : ℕ := 2600

theorem eliana_steps :
  let first_day := initial_steps + additional_steps
  let second_day := 2 * first_day
  let third_day := second_day + extra_steps
  first_day + second_day + third_day = total_steps := by sorry

end NUMINAMATH_CALUDE_eliana_steps_l299_29975


namespace NUMINAMATH_CALUDE_julie_school_year_hours_l299_29927

/-- Calculates the number of hours Julie needs to work per week during the school year
    to maintain the same rate of pay as her summer job. -/
theorem julie_school_year_hours
  (summer_weeks : ℕ)
  (summer_hours_per_week : ℕ)
  (summer_earnings : ℕ)
  (school_year_weeks : ℕ)
  (school_year_earnings : ℕ)
  (h1 : summer_weeks = 15)
  (h2 : summer_hours_per_week = 40)
  (h3 : summer_earnings = 6000)
  (h4 : school_year_weeks = 30)
  (h5 : school_year_earnings = 7500)
  : (school_year_earnings * summer_weeks * summer_hours_per_week) / 
    (summer_earnings * school_year_weeks) = 25 := by
  sorry


end NUMINAMATH_CALUDE_julie_school_year_hours_l299_29927


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l299_29937

/-- Arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0  -- Arithmetic property
  h_sum : ∀ n, S n = n * (a 0 + a (n-1)) / 2  -- Sum formula

/-- The main theorem -/
theorem arithmetic_sequence_difference
  (seq : ArithmeticSequence)
  (h : seq.S 10 / 10 - seq.S 9 / 9 = 1) :
  seq.a 1 - seq.a 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l299_29937


namespace NUMINAMATH_CALUDE_rational_root_implies_even_coefficient_l299_29992

theorem rational_root_implies_even_coefficient 
  (a b c : ℤ) 
  (h : ∃ (p q : ℤ), q ≠ 0 ∧ a * (p / q)^2 + b * (p / q) + c = 0) : 
  a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_root_implies_even_coefficient_l299_29992


namespace NUMINAMATH_CALUDE_sum_fraction_problem_l299_29921

theorem sum_fraction_problem (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h1 : a₁ / 2 + a₂ / 3 + a₃ / 4 + a₄ / 5 + a₅ / 6 = 1)
  (h2 : a₁ / 5 + a₂ / 6 + a₃ / 7 + a₄ / 8 + a₅ / 9 = 1 / 4)
  (h3 : a₁ / 10 + a₂ / 11 + a₃ / 12 + a₄ / 13 + a₅ / 14 = 1 / 9)
  (h4 : a₁ / 17 + a₂ / 18 + a₃ / 19 + a₄ / 20 + a₅ / 21 = 1 / 16)
  (h5 : a₁ / 26 + a₂ / 27 + a₃ / 28 + a₄ / 29 + a₅ / 30 = 1 / 25) :
  a₁ / 37 + a₂ / 38 + a₃ / 39 + a₄ / 40 + a₅ / 41 = 187465 / 6744582 := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_problem_l299_29921


namespace NUMINAMATH_CALUDE_max_value_on_interval_l299_29979

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + a

theorem max_value_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a x ≤ f a y) →
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, 3 ≤ f a x) →
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a y ≤ f a x ∧ f a x = 57) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l299_29979


namespace NUMINAMATH_CALUDE_daves_shirts_l299_29955

theorem daves_shirts (short_sleeve : ℕ) (washed : ℕ) (unwashed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : washed = 20)
  (h3 : unwashed = 16) :
  washed + unwashed - short_sleeve = 27 := by
  sorry

end NUMINAMATH_CALUDE_daves_shirts_l299_29955


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l299_29970

theorem fractional_equation_solution :
  ∃ x : ℝ, (1 / (x - 1) = 2 / (x - 2)) ∧ (x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l299_29970


namespace NUMINAMATH_CALUDE_gcd_180_450_l299_29930

theorem gcd_180_450 : Nat.gcd 180 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_450_l299_29930


namespace NUMINAMATH_CALUDE_sin_cos_identity_l299_29945

theorem sin_cos_identity : 
  Real.sin (43 * π / 180) * Real.sin (17 * π / 180) - 
  Real.cos (43 * π / 180) * Real.cos (17 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l299_29945


namespace NUMINAMATH_CALUDE_angle_C_equals_140_l299_29932

/-- A special quadrilateral ABCD where ∠A + ∠B = 180° and ∠C = ∠A -/
structure SpecialQuadrilateral where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_AB : A + B = 180
  C_eq_A : C = A

/-- Theorem: In a special quadrilateral ABCD where ∠A : ∠B = 7 : 2, ∠C = 140° -/
theorem angle_C_equals_140 (ABCD : SpecialQuadrilateral) (h : ABCD.A / ABCD.B = 7 / 2) : 
  ABCD.C = 140 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_equals_140_l299_29932


namespace NUMINAMATH_CALUDE_jared_popcorn_order_l299_29973

/-- Calculates the minimum number of popcorn servings needed for a group -/
def min_popcorn_servings (pieces_per_serving : ℕ) (jared_consumption : ℕ) 
  (friend_group1_size : ℕ) (friend_group1_consumption : ℕ)
  (friend_group2_size : ℕ) (friend_group2_consumption : ℕ) : ℕ :=
  let total_consumption := jared_consumption + 
    friend_group1_size * friend_group1_consumption +
    friend_group2_size * friend_group2_consumption
  (total_consumption + pieces_per_serving - 1) / pieces_per_serving

/-- The minimum number of popcorn servings needed for Jared and his friends is 21 -/
theorem jared_popcorn_order : 
  min_popcorn_servings 50 120 5 90 3 150 = 21 := by
  sorry

end NUMINAMATH_CALUDE_jared_popcorn_order_l299_29973


namespace NUMINAMATH_CALUDE_apples_fallen_count_l299_29911

/-- Represents the number of apples that fell out of Carla's backpack -/
def apples_fallen (initial : ℕ) (stolen : ℕ) (remaining : ℕ) : ℕ :=
  initial - stolen - remaining

/-- Theorem stating that 26 apples fell out of Carla's backpack -/
theorem apples_fallen_count : apples_fallen 79 45 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_apples_fallen_count_l299_29911


namespace NUMINAMATH_CALUDE_polynomial_roots_l299_29912

theorem polynomial_roots (x : ℝ) : x^4 - 3*x^3 + x^2 - 3*x = 0 ↔ x = 0 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l299_29912


namespace NUMINAMATH_CALUDE_voldemort_shopping_l299_29999

theorem voldemort_shopping (book_price : ℝ) (journal_price : ℝ) : 
  book_price = 8 ∧ 
  book_price = (1/8) * (book_price * 8) ∧ 
  journal_price = 2 * book_price →
  (book_price * 8 = 64) ∧ 
  (book_price + journal_price = 24) := by
sorry

end NUMINAMATH_CALUDE_voldemort_shopping_l299_29999


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l299_29990

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 2) (h₂ : a₂ = 5) (h₃ : a₃ = 8) :
  arithmeticSequence a₁ (a₂ - a₁) 30 = 89 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l299_29990


namespace NUMINAMATH_CALUDE_evaluate_expression_l299_29901

theorem evaluate_expression : (0.5^2 + 0.05^3) / (0.005^3) = 2000100 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l299_29901


namespace NUMINAMATH_CALUDE_inverse_proportion_quadrants_l299_29934

theorem inverse_proportion_quadrants (k : ℝ) (h1 : k ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ k / x
  (f 1 = 1) →
  (∀ x : ℝ, x ≠ 0 → (x > 0 ∧ f x > 0) ∨ (x < 0 ∧ f x < 0)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_quadrants_l299_29934


namespace NUMINAMATH_CALUDE_tom_shirt_purchase_l299_29922

def shirts_per_fandom : ℕ := 5
def num_fandoms : ℕ := 4
def original_price : ℚ := 15
def discount_percentage : ℚ := 20
def tax_percentage : ℚ := 10

def discounted_price : ℚ := original_price * (1 - discount_percentage / 100)

def total_shirts : ℕ := shirts_per_fandom * num_fandoms

def pre_tax_total : ℚ := (total_shirts : ℚ) * discounted_price

def tax_amount : ℚ := pre_tax_total * (tax_percentage / 100)

def total_cost : ℚ := pre_tax_total + tax_amount

theorem tom_shirt_purchase :
  total_cost = 264 := by sorry

end NUMINAMATH_CALUDE_tom_shirt_purchase_l299_29922


namespace NUMINAMATH_CALUDE_cubic_function_coefficient_l299_29961

/-- Given a cubic function f(x) = ax³ + 3x² + 2, 
    prove that if its second derivative at x = -1 is 4, 
    then the coefficient a must be 10/3. -/
theorem cubic_function_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = a * x^3 + 3 * x^2 + 2) →
  (deriv (deriv f)) (-1) = 4 →
  a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_coefficient_l299_29961


namespace NUMINAMATH_CALUDE_gecko_ratio_l299_29983

-- Define the number of geckos sold last year
def geckos_last_year : ℕ := 86

-- Define the total number of geckos sold in the last two years
def total_geckos : ℕ := 258

-- Define the number of geckos sold the year before
def geckos_year_before : ℕ := total_geckos - geckos_last_year

-- Theorem to prove the ratio
theorem gecko_ratio : 
  geckos_year_before = 2 * geckos_last_year := by
  sorry

#check gecko_ratio

end NUMINAMATH_CALUDE_gecko_ratio_l299_29983


namespace NUMINAMATH_CALUDE_parallelepiped_with_extensions_volume_l299_29938

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a parallelepiped -/
def volume (p : Parallelepiped) : ℝ :=
  p.length * p.width * p.height

/-- Calculates the volume of extensions from all faces -/
def extension_volume (p : Parallelepiped) : ℝ :=
  2 * (p.length * p.width + p.width * p.height + p.length * p.height)

/-- The main theorem to prove -/
theorem parallelepiped_with_extensions_volume 
  (p : Parallelepiped) 
  (h1 : p.length = 2) 
  (h2 : p.width = 3) 
  (h3 : p.height = 4) :
  volume p + extension_volume p = 76 := by
  sorry

#check parallelepiped_with_extensions_volume

end NUMINAMATH_CALUDE_parallelepiped_with_extensions_volume_l299_29938


namespace NUMINAMATH_CALUDE_inequality_proof_l299_29905

theorem inequality_proof (a b c d p q : ℝ) 
  (h1 : a * b + c * d = 2 * p * q) 
  (h2 : a * c ≥ p^2) 
  (h3 : p > 0) : 
  b * d ≤ q^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l299_29905


namespace NUMINAMATH_CALUDE_infinite_subset_with_common_gcd_l299_29946

-- Define the set A
def A : Set ℕ := {n : ℕ | ∃ (primes : Finset ℕ), primes.card ≤ 1987 ∧ (∀ p ∈ primes, Nat.Prime p) ∧ n = primes.prod id}

-- State the theorem
theorem infinite_subset_with_common_gcd (h : Set.Infinite A) :
  ∃ (B : Set ℕ) (b : ℕ), Set.Infinite B ∧ B ⊆ A ∧ ∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → Nat.gcd x y = b :=
sorry

end NUMINAMATH_CALUDE_infinite_subset_with_common_gcd_l299_29946


namespace NUMINAMATH_CALUDE_car_rental_cost_l299_29952

/-- The daily rental cost of a car, given a daily budget, maximum mileage, and per-mile rate. -/
theorem car_rental_cost 
  (daily_budget : ℝ) 
  (max_miles : ℝ) 
  (per_mile_rate : ℝ) 
  (h1 : daily_budget = 88) 
  (h2 : max_miles = 190) 
  (h3 : per_mile_rate = 0.2) : 
  daily_budget - max_miles * per_mile_rate = 50 := by
  sorry

#check car_rental_cost

end NUMINAMATH_CALUDE_car_rental_cost_l299_29952


namespace NUMINAMATH_CALUDE_point_on_line_product_of_y_coords_l299_29950

theorem point_on_line_product_of_y_coords :
  ∀ y₁ y₂ : ℝ,
  ((-3 - 7)^2 + (y₁ - (-3))^2 = 15^2) →
  ((-3 - 7)^2 + (y₂ - (-3))^2 = 15^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -116 :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_product_of_y_coords_l299_29950


namespace NUMINAMATH_CALUDE_diana_earnings_l299_29994

theorem diana_earnings (x : ℝ) 
  (july : x > 0)
  (august : x > 0)
  (september : x > 0)
  (total : x + 3*x + 6*x = 1500) : x = 150 := by
  sorry

end NUMINAMATH_CALUDE_diana_earnings_l299_29994


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l299_29966

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 4 + (1/2) * a 7 + a 10 = 10) →
  (a 3 + a 11 = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l299_29966


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l299_29967

theorem triangle_angle_calculation (a b c A B C : Real) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  -- Given conditions
  (a = Real.sqrt 2) →
  (b = Real.sqrt 3) →
  (B = π / 3) →
  -- Law of Sines
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion
  A = π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l299_29967


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_of_squares_l299_29976

def is_consecutive_odd (a b c : ℤ) : Prop :=
  b = a + 2 ∧ c = b + 2 ∧ ∃ k : ℤ, a = 2 * k + 1

theorem consecutive_odd_sum_of_squares (a b c : ℤ) :
  is_consecutive_odd a b c → a^2 + b^2 + c^2 = 251 →
  ((a = 7 ∧ b = 9 ∧ c = 11) ∨ (a = -11 ∧ b = -9 ∧ c = -7)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_of_squares_l299_29976


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l299_29910

theorem quadratic_expression_value (x y : ℚ) 
  (eq1 : 3 * x + 2 * y = 10) 
  (eq2 : x + 3 * y = 11) : 
  9 * x^2 + 15 * x * y + 9 * y^2 = 8097 / 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l299_29910


namespace NUMINAMATH_CALUDE_max_inradii_difference_l299_29939

noncomputable def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

def Q₁ (P : ℝ × ℝ) : ℝ × ℝ := sorry
def Q₂ (P : ℝ × ℝ) : ℝ × ℝ := sorry

def r₁ (P : ℝ × ℝ) : ℝ := sorry
def r₂ (P : ℝ × ℝ) : ℝ := sorry

theorem max_inradii_difference :
  ∃ (max : ℝ), max = 1/3 ∧
  ∀ (P : ℝ × ℝ), on_ellipse P → first_quadrant P.1 P.2 →
  r₁ P - r₂ P ≤ max ∧
  ∃ (P' : ℝ × ℝ), on_ellipse P' ∧ first_quadrant P'.1 P'.2 ∧ r₁ P' - r₂ P' = max :=
sorry

end NUMINAMATH_CALUDE_max_inradii_difference_l299_29939


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l299_29941

-- Define the repeating decimals
def repeating_decimal_1 : ℚ := 8/9
def repeating_decimal_2 : ℚ := 15/11

-- State the theorem
theorem repeating_decimal_fraction : 
  repeating_decimal_1 / repeating_decimal_2 = 88 / 135 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l299_29941


namespace NUMINAMATH_CALUDE_race_finish_time_difference_l299_29958

/-- Race finish time difference problem -/
theorem race_finish_time_difference 
  (malcolm_speed : ℕ) 
  (joshua_speed : ℕ) 
  (race_distance : ℕ) 
  (h1 : malcolm_speed = 5)
  (h2 : joshua_speed = 7)
  (h3 : race_distance = 12) :
  joshua_speed * race_distance - malcolm_speed * race_distance = 24 := by
sorry

end NUMINAMATH_CALUDE_race_finish_time_difference_l299_29958


namespace NUMINAMATH_CALUDE_sum_fifth_sixth_l299_29920

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q
  sum_first_two : a 1 + a 2 = 3
  sum_third_fourth : a 3 + a 4 = 12

/-- The sum of the fifth and sixth terms of the geometric sequence is 48 -/
theorem sum_fifth_sixth (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_fifth_sixth_l299_29920


namespace NUMINAMATH_CALUDE_arithmetic_sequence_collinearity_geometric_sequence_characterization_l299_29963

def isArithmeticSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

def isGeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ+, a (n + 1) = q * a n

def S (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

def areCollinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

theorem arithmetic_sequence_collinearity (a : ℕ+ → ℝ) :
  isArithmeticSequence a →
  areCollinear (10, S a 10 / 10) (100, S a 100 / 100) (110, S a 110 / 110) :=
sorry

theorem geometric_sequence_characterization (a : ℕ+ → ℝ) (a₁ q : ℝ) :
  (∀ n : ℕ+, S a (n + 1) = a₁ + q * S a n) ∧ q ≠ 0 →
  isGeometricSequence a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_collinearity_geometric_sequence_characterization_l299_29963


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l299_29928

-- Define the two fixed circles
def Q₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def Q₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Define the property of being externally tangent
def externally_tangent (x y R : ℝ) : Prop :=
  ∀ (x₁ y₁ : ℝ), Q₁ x₁ y₁ → (x - x₁)^2 + (y - y₁)^2 = (1 + R)^2

-- Define the property of being internally tangent
def internally_tangent (x y R : ℝ) : Prop :=
  ∀ (x₂ y₂ : ℝ), Q₂ x₂ y₂ → (x - x₂)^2 + (y - y₂)^2 = (9 - R)^2

-- Define the trajectory of the center of the moving circle
def trajectory (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

-- State the theorem
theorem moving_circle_trajectory :
  ∀ (x y R : ℝ), externally_tangent x y R → internally_tangent x y R → trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l299_29928


namespace NUMINAMATH_CALUDE_trajectory_equation_l299_29942

/-- The trajectory of point M satisfying the distance ratio condition -/
theorem trajectory_equation (x y : ℝ) :
  (((x - 5)^2 + y^2).sqrt / |x - 9/5| = 5/3) →
  (x^2 / 9 - y^2 / 16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l299_29942


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l299_29965

theorem fraction_to_zero_power : 
  (17381294 : ℚ) / (-43945723904 : ℚ) ^ (0 : ℕ) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l299_29965


namespace NUMINAMATH_CALUDE_bonnie_cupcakes_l299_29993

/-- Represents the problem of calculating cupcakes to give to Bonnie -/
def cupcakes_to_give (total_goal : ℕ) (days : ℕ) (daily_goal : ℕ) : ℕ :=
  days * daily_goal - total_goal

theorem bonnie_cupcakes :
  cupcakes_to_give 96 2 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_bonnie_cupcakes_l299_29993


namespace NUMINAMATH_CALUDE_ratio_problem_l299_29902

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 4) : 
  (a + b) / (b + c) = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l299_29902


namespace NUMINAMATH_CALUDE_london_trip_train_time_l299_29969

/-- Calculates the train ride time given the total trip time and other components of the journey. -/
def train_ride_time (total_trip_time : ℕ) (bus_ride_time : ℕ) (walking_time : ℕ) : ℕ :=
  let waiting_time := 2 * walking_time
  let total_trip_minutes := total_trip_time * 60
  let non_train_time := bus_ride_time + walking_time + waiting_time
  (total_trip_minutes - non_train_time) / 60

/-- Theorem stating that given the specific journey times, the train ride takes 6 hours. -/
theorem london_trip_train_time :
  train_ride_time 8 75 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_london_trip_train_time_l299_29969


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l299_29949

theorem square_difference_of_sum_and_diff (x y : ℝ) 
  (h1 : x + y = 5) (h2 : x - y = 10) : x^2 - y^2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l299_29949


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_l299_29957

theorem smallest_multiple_of_seven (x y : ℤ) 
  (hx : (x - 2) % 7 = 0) 
  (hy : (y + 2) % 7 = 0) : 
  (∃ n : ℕ+, (x^2 + x*y + y^2 + n) % 7 = 0 ∧ 
    ∀ m : ℕ+, m < n → (x^2 + x*y + y^2 + m) % 7 ≠ 0) → 
  (∃ n : ℕ+, n = 3 ∧ (x^2 + x*y + y^2 + n) % 7 = 0 ∧ 
    ∀ m : ℕ+, m < n → (x^2 + x*y + y^2 + m) % 7 ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_l299_29957


namespace NUMINAMATH_CALUDE_inequality_equiv_interval_l299_29906

theorem inequality_equiv_interval (x : ℝ) (h : x + 3 ≠ 0) :
  (x + 1) / (x + 3) ≤ 3 ↔ x ∈ Set.Ici (-4) ∩ Set.Iio (-3) :=
sorry

end NUMINAMATH_CALUDE_inequality_equiv_interval_l299_29906


namespace NUMINAMATH_CALUDE_area_difference_square_inscribed_triangle_l299_29916

/-- Given an isosceles right triangle inscribed in a square, where the hypotenuse
    of the triangle is the diagonal of the square and has length 8√2 cm,
    prove that the area difference between the square and the triangle is 32 cm². -/
theorem area_difference_square_inscribed_triangle :
  ∀ (square_side triangle_side : ℝ),
  square_side * square_side * 2 = (8 * Real.sqrt 2) ^ 2 →
  triangle_side * triangle_side * 2 = (8 * Real.sqrt 2) ^ 2 →
  square_side ^ 2 - (triangle_side ^ 2 / 2) = 32 :=
by sorry

end NUMINAMATH_CALUDE_area_difference_square_inscribed_triangle_l299_29916


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l299_29988

theorem quadratic_equation_roots (m : ℤ) : 
  (∃ x y : ℤ, x > 0 ∧ y > 0 ∧ 
   m * x^2 + (3 - m) * x - 3 = 0 ∧
   m * y^2 + (3 - m) * y - 3 = 0 ∧
   x ≠ y) →
  m = -1 ∨ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l299_29988


namespace NUMINAMATH_CALUDE_reflected_ray_slope_l299_29948

theorem reflected_ray_slope (emissionPoint : ℝ × ℝ) (circleCenter : ℝ × ℝ) (circleRadius : ℝ) :
  emissionPoint = (-2, -3) →
  circleCenter = (-3, 2) →
  circleRadius = 1 →
  ∃ k : ℝ, (k = -4/3 ∨ k = -3/4) ∧
    (∀ x y : ℝ, y + 3 = k * (x - 2) →
      ((x + 3)^2 + (y - 2)^2 = 1 →
        abs (-3*k - 2 - 2*k - 3) / Real.sqrt (k^2 + 1) = 1)) :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_slope_l299_29948


namespace NUMINAMATH_CALUDE_hexagon_side_length_l299_29907

/-- The side length of a regular hexagon given the distance between opposite sides -/
theorem hexagon_side_length (d : ℝ) (h : d = 10) : 
  let s := d * 2 / (3 : ℝ).sqrt
  s = 40 / 3 := by sorry

#check hexagon_side_length

end NUMINAMATH_CALUDE_hexagon_side_length_l299_29907


namespace NUMINAMATH_CALUDE_fifth_term_equals_eight_l299_29944

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = 2ⁿ⁻¹, prove that a₅ = 8 -/
theorem fifth_term_equals_eight (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = 2^(n - 1)) : a 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_equals_eight_l299_29944


namespace NUMINAMATH_CALUDE_solution_set_for_a_3_range_of_a_l299_29964

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Part 1: Solution set for a = 3
theorem solution_set_for_a_3 :
  {x : ℝ | f 3 x ≥ 2} = {x : ℝ | x ≤ 2/3 ∨ x ≥ 2} := by sorry

-- Part 2: Range of a
theorem range_of_a :
  (∀ x : ℝ, f a x ≥ 5 - x) → a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_3_range_of_a_l299_29964


namespace NUMINAMATH_CALUDE_boat_license_count_l299_29986

def boat_license_options : ℕ :=
  let letter_options := 3  -- A, M, or S
  let digit_options := 10  -- 0 to 9
  let number_of_digits := 6
  letter_options * digit_options ^ number_of_digits

theorem boat_license_count : boat_license_options = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_count_l299_29986


namespace NUMINAMATH_CALUDE_tomatoes_picked_yesterday_l299_29982

theorem tomatoes_picked_yesterday (initial : Nat) (picked_today : Nat) (left : Nat) :
  initial = 171 →
  picked_today = 30 →
  left = 7 →
  initial - (initial - picked_today - left) - picked_today = 134 := by
sorry

end NUMINAMATH_CALUDE_tomatoes_picked_yesterday_l299_29982


namespace NUMINAMATH_CALUDE_remainder_four_eleven_mod_five_l299_29917

theorem remainder_four_eleven_mod_five : 4^11 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_four_eleven_mod_five_l299_29917


namespace NUMINAMATH_CALUDE_complex_imaginary_condition_l299_29959

theorem complex_imaginary_condition (a : ℝ) : 
  (Complex.I.re = 0 ∧ Complex.I.im = 1) →
  ((1 - 2 * Complex.I) * (a + Complex.I)).re = 0 →
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_complex_imaginary_condition_l299_29959


namespace NUMINAMATH_CALUDE_cylindrical_to_rectangular_l299_29962

theorem cylindrical_to_rectangular :
  let r : ℝ := 6
  let θ : ℝ := 5 * π / 3
  let z : ℝ := 7
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y, z) = (3, 3 * Real.sqrt 3, 7) := by sorry

end NUMINAMATH_CALUDE_cylindrical_to_rectangular_l299_29962


namespace NUMINAMATH_CALUDE_oil_leak_during_work_l299_29924

/-- The amount of oil leaked while engineers were working is equal to the difference between the total oil leak and the initial oil leak. -/
theorem oil_leak_during_work (initial_leak total_leak : ℕ) (h : initial_leak = 6522 ∧ total_leak = 11687) :
  total_leak - initial_leak = 5165 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_during_work_l299_29924


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_negative_two_l299_29978

theorem sum_of_x_and_y_is_negative_two (x y : ℝ) 
  (hx : (x + 1) ^ (3/5 : ℝ) + 2023 * (x + 1) = -2023)
  (hy : (y + 1) ^ (3/5 : ℝ) + 2023 * (y + 1) = 2023) :
  x + y = -2 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_negative_two_l299_29978


namespace NUMINAMATH_CALUDE_thirteen_fourth_mod_eight_l299_29998

theorem thirteen_fourth_mod_eight : 13^4 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_fourth_mod_eight_l299_29998


namespace NUMINAMATH_CALUDE_boxes_left_is_three_l299_29918

/-- The number of boxes of apples Merry had on Saturday -/
def saturday_boxes : ℕ := 50

/-- The number of boxes of apples Merry had on Sunday -/
def sunday_boxes : ℕ := 25

/-- The number of apples in each box -/
def apples_per_box : ℕ := 10

/-- The total number of apples Merry sold on Saturday and Sunday -/
def apples_sold : ℕ := 720

/-- Calculate the number of boxes of apples left -/
def boxes_left : ℕ := 
  (saturday_boxes * apples_per_box + sunday_boxes * apples_per_box - apples_sold) / apples_per_box

/-- Theorem stating that the number of boxes left is 3 -/
theorem boxes_left_is_three : boxes_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_boxes_left_is_three_l299_29918
