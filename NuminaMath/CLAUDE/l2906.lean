import Mathlib

namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l2906_290638

theorem abs_inequality_solution_set :
  {x : ℝ | |2*x + 1| < 3} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l2906_290638


namespace NUMINAMATH_CALUDE_flour_needed_l2906_290626

/-- The amount of flour Katie needs in pounds -/
def katie_flour : ℕ := 3

/-- The additional amount of flour Sheila needs compared to Katie in pounds -/
def sheila_extra : ℕ := 2

/-- The total amount of flour needed by Katie and Sheila -/
def total_flour : ℕ := katie_flour + (katie_flour + sheila_extra)

theorem flour_needed : total_flour = 8 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_l2906_290626


namespace NUMINAMATH_CALUDE_salt_mixture_price_l2906_290689

theorem salt_mixture_price (salt_price_1 : ℚ) (salt_weight_1 : ℚ)
  (salt_weight_2 : ℚ) (selling_price : ℚ) (profit_percentage : ℚ) :
  salt_price_1 = 50 / 100 →
  salt_weight_1 = 8 →
  salt_weight_2 = 40 →
  selling_price = 48 / 100 →
  profit_percentage = 20 / 100 →
  ∃ (salt_price_2 : ℚ),
    salt_price_2 * salt_weight_2 + salt_price_1 * salt_weight_1 =
      (selling_price * (salt_weight_1 + salt_weight_2)) / (1 + profit_percentage) ∧
    salt_price_2 = 38 / 100 :=
by sorry

end NUMINAMATH_CALUDE_salt_mixture_price_l2906_290689


namespace NUMINAMATH_CALUDE_pt_length_in_special_quadrilateral_l2906_290631

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (quad : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def intersection_point (p1 p2 p3 p4 : Point) : Point := sorry

theorem pt_length_in_special_quadrilateral 
  (PQRS : Quadrilateral) 
  (T : Point)
  (h_convex : is_convex PQRS)
  (h_PQ : distance PQRS.P PQRS.Q = 10)
  (h_RS : distance PQRS.R PQRS.S = 15)
  (h_PR : distance PQRS.P PQRS.R = 18)
  (h_T : T = intersection_point PQRS.P PQRS.R PQRS.Q PQRS.S)
  (h_equal_areas : triangle_area PQRS.P T PQRS.S = triangle_area PQRS.Q T PQRS.R) :
  distance PQRS.P T = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_pt_length_in_special_quadrilateral_l2906_290631


namespace NUMINAMATH_CALUDE_floor_equality_iff_interval_l2906_290691

theorem floor_equality_iff_interval (x : ℝ) :
  ⌊⌊2 * x⌋ - (1/2 : ℝ)⌋ = ⌊x + 3⌋ ↔ 3 ≤ x ∧ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_floor_equality_iff_interval_l2906_290691


namespace NUMINAMATH_CALUDE_target_probability_l2906_290655

theorem target_probability (p : ℝ) : 
  (1 - (1 - p)^3 = 0.875) → p = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_target_probability_l2906_290655


namespace NUMINAMATH_CALUDE_toms_running_days_l2906_290694

/-- Proves that Tom runs 5 days a week given his running schedule and total distance covered -/
theorem toms_running_days 
  (hours_per_day : ℝ) 
  (speed : ℝ) 
  (total_miles_per_week : ℝ) 
  (h1 : hours_per_day = 1.5)
  (h2 : speed = 8)
  (h3 : total_miles_per_week = 60) :
  (total_miles_per_week / (speed * hours_per_day)) = 5 := by
  sorry


end NUMINAMATH_CALUDE_toms_running_days_l2906_290694


namespace NUMINAMATH_CALUDE_oil_production_fraction_l2906_290656

/-- Represents the fraction of oil sent for production -/
def x : ℝ := sorry

/-- Initial sulfur concentration -/
def initial_conc : ℝ := 0.015

/-- Sulfur concentration of first replacement oil -/
def first_repl_conc : ℝ := 0.005

/-- Sulfur concentration of second replacement oil -/
def second_repl_conc : ℝ := 0.02

/-- Theorem stating that the fraction of oil sent for production is 1/2 -/
theorem oil_production_fraction :
  (initial_conc - initial_conc * x + first_repl_conc * x - 
   (initial_conc - initial_conc * x + first_repl_conc * x) * x + 
   second_repl_conc * x = initial_conc) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_oil_production_fraction_l2906_290656


namespace NUMINAMATH_CALUDE_number_problem_l2906_290682

theorem number_problem (N : ℝ) 
  (h1 : (1/4) * (1/3) * (2/5) * N = 17) 
  (h2 : Real.sqrt (0.6 * N) = (N^(1/3)) / 2) : 
  0.4 * N = 204 := by sorry

end NUMINAMATH_CALUDE_number_problem_l2906_290682


namespace NUMINAMATH_CALUDE_bill_and_caroline_ages_l2906_290650

/-- Given that Bill is 17 years old and 1 year less than twice as old as his sister Caroline,
    prove that the sum of their ages is 26. -/
theorem bill_and_caroline_ages : ∀ (caroline_age : ℕ),
  17 = 2 * caroline_age - 1 →
  17 + caroline_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_bill_and_caroline_ages_l2906_290650


namespace NUMINAMATH_CALUDE_square_sum_value_l2906_290677

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 9) : a^2 + b^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l2906_290677


namespace NUMINAMATH_CALUDE_art_supplies_cost_l2906_290664

def total_spent : ℕ := 50
def num_skirts : ℕ := 2
def skirt_cost : ℕ := 15

theorem art_supplies_cost : total_spent - (num_skirts * skirt_cost) = 20 := by
  sorry

end NUMINAMATH_CALUDE_art_supplies_cost_l2906_290664


namespace NUMINAMATH_CALUDE_angle_between_planes_l2906_290667

def plane1 : ℝ → ℝ → ℝ → ℝ := fun x y z ↦ 3 * x - 4 * y + z - 8
def plane2 : ℝ → ℝ → ℝ → ℝ := fun x y z ↦ 9 * x - 12 * y - 4 * z + 6

def normal1 : Fin 3 → ℝ := fun i ↦ match i with
  | 0 => 3
  | 1 => -4
  | 2 => 1

def normal2 : Fin 3 → ℝ := fun i ↦ match i with
  | 0 => 9
  | 1 => -12
  | 2 => -4

theorem angle_between_planes :
  let dot_product := (normal1 0 * normal2 0 + normal1 1 * normal2 1 + normal1 2 * normal2 2)
  let magnitude1 := Real.sqrt (normal1 0 ^ 2 + normal1 1 ^ 2 + normal1 2 ^ 2)
  let magnitude2 := Real.sqrt (normal2 0 ^ 2 + normal2 1 ^ 2 + normal2 2 ^ 2)
  dot_product / (magnitude1 * magnitude2) = 71 / (Real.sqrt 26 * Real.sqrt 241) := by
sorry

end NUMINAMATH_CALUDE_angle_between_planes_l2906_290667


namespace NUMINAMATH_CALUDE_cos_negative_seventy_nine_sixths_pi_l2906_290645

theorem cos_negative_seventy_nine_sixths_pi :
  Real.cos (-79/6 * Real.pi) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_seventy_nine_sixths_pi_l2906_290645


namespace NUMINAMATH_CALUDE_cube_not_always_positive_l2906_290607

theorem cube_not_always_positive : ¬ (∀ x : ℝ, x^3 > 0) := by
  sorry

end NUMINAMATH_CALUDE_cube_not_always_positive_l2906_290607


namespace NUMINAMATH_CALUDE_different_plant_choice_probability_l2906_290649

theorem different_plant_choice_probability :
  let num_plant_types : ℕ := 4
  let num_employees : ℕ := 2
  let total_combinations : ℕ := num_plant_types ^ num_employees
  let same_choice_combinations : ℕ := num_plant_types
  let different_choice_combinations : ℕ := total_combinations - same_choice_combinations
  (different_choice_combinations : ℚ) / total_combinations = 13 / 16 :=
by sorry

end NUMINAMATH_CALUDE_different_plant_choice_probability_l2906_290649


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_2_8_plus_5_5_l2906_290681

theorem greatest_prime_factor_of_2_8_plus_5_5 :
  ∃ p : ℕ, p.Prime ∧ p ∣ (2^8 + 5^5) ∧ ∀ q : ℕ, q.Prime → q ∣ (2^8 + 5^5) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_2_8_plus_5_5_l2906_290681


namespace NUMINAMATH_CALUDE_scientific_notation_138000_l2906_290698

theorem scientific_notation_138000 :
  138000 = 1.38 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_138000_l2906_290698


namespace NUMINAMATH_CALUDE_largest_space_diagonal_squared_of_box_l2906_290678

/-- The square of the largest possible length of the space diagonal of a smaller box -/
def largest_space_diagonal_squared (a b c : ℕ) : ℕ :=
  max
    (a * a + (b / 2) * (b / 2) + c * c)
    (max
      (a * a + b * b + (c / 2) * (c / 2))
      ((a / 2) * (a / 2) + b * b + c * c))

/-- Theorem stating the largest possible space diagonal squared for the given box -/
theorem largest_space_diagonal_squared_of_box :
  largest_space_diagonal_squared 1 2 16 = 258 := by
  sorry

end NUMINAMATH_CALUDE_largest_space_diagonal_squared_of_box_l2906_290678


namespace NUMINAMATH_CALUDE_cosine_sum_zero_l2906_290651

theorem cosine_sum_zero (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos (y + 2 * Real.pi / 3) + Real.cos (z + 4 * Real.pi / 3) = 0)
  (h2 : Real.sin x + Real.sin (y + 2 * Real.pi / 3) + Real.sin (z + 4 * Real.pi / 3) = 0) :
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_zero_l2906_290651


namespace NUMINAMATH_CALUDE_circle_x_plus_y_bounds_l2906_290625

-- Define the circle in polar form
def polar_circle (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi/4) + 6 = 0

-- Define a point on the circle in Cartesian coordinates
def point_on_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 4

-- Theorem statement
theorem circle_x_plus_y_bounds :
  ∀ x y : ℝ, point_on_circle x y →
  (∃ θ : ℝ, polar_circle (Real.sqrt (x^2 + y^2)) θ) →
  2 ≤ x + y ∧ x + y ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_x_plus_y_bounds_l2906_290625


namespace NUMINAMATH_CALUDE_total_tickets_sold_l2906_290675

theorem total_tickets_sold (student_price general_price total_amount general_tickets : ℕ)
  (h1 : student_price = 4)
  (h2 : general_price = 6)
  (h3 : total_amount = 2876)
  (h4 : general_tickets = 388)
  (h5 : ∃ student_tickets : ℕ, student_price * student_tickets + general_price * general_tickets = total_amount) :
  ∃ total_tickets : ℕ, total_tickets = general_tickets + (total_amount - general_price * general_tickets) / student_price ∧ total_tickets = 525 :=
by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l2906_290675


namespace NUMINAMATH_CALUDE_circumcenter_coincidence_l2906_290663

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  distance : ℝ

/-- The circumcenter of a tetrahedron -/
def circumcenter (t : Tetrahedron) : Point3D :=
  sorry

/-- The inscribed sphere of a tetrahedron -/
def inscribedSphere (t : Tetrahedron) : Sphere :=
  sorry

/-- Points where the inscribed sphere touches the faces of the tetrahedron -/
def touchPoints (t : Tetrahedron) (s : Sphere) : (Point3D × Point3D × Point3D × Point3D) :=
  sorry

/-- Plane equidistant from a point and another plane -/
def equidistantPlane (p : Point3D) (pl : Plane) : Plane :=
  sorry

/-- Tetrahedron formed by four planes -/
def tetrahedronFromPlanes (p1 p2 p3 p4 : Plane) : Tetrahedron :=
  sorry

/-- Main theorem statement -/
theorem circumcenter_coincidence (t : Tetrahedron) : 
  let s := inscribedSphere t
  let (A₁, B₁, C₁, D₁) := touchPoints t s
  let p1 := equidistantPlane t.A (Plane.mk B₁ 0)
  let p2 := equidistantPlane t.B (Plane.mk C₁ 0)
  let p3 := equidistantPlane t.C (Plane.mk D₁ 0)
  let p4 := equidistantPlane t.D (Plane.mk A₁ 0)
  let t' := tetrahedronFromPlanes p1 p2 p3 p4
  circumcenter t = circumcenter t' :=
by
  sorry

end NUMINAMATH_CALUDE_circumcenter_coincidence_l2906_290663


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l2906_290684

theorem cosine_sine_identity : 
  (Real.cos (10 * π / 180)) / (2 * Real.sin (10 * π / 180)) - 2 * Real.cos (10 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l2906_290684


namespace NUMINAMATH_CALUDE_solve_for_y_l2906_290623

/-- Custom operation € defined for real numbers -/
def custom_op (x y : ℝ) : ℝ := 2 * x * y

/-- Theorem stating that under given conditions, y must equal 5 -/
theorem solve_for_y (y : ℝ) :
  (custom_op 7 (custom_op 4 y) = 560) → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2906_290623


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2906_290611

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ r₁ r₂ : ℝ, r₁ + r₂ = 4 ∧ r₁ * r₂ = 6 ∧ 
      3 * r₁^2 - p * r₁ + q = 0 ∧ 
      3 * r₂^2 - p * r₂ + q = 0)) → 
  p = 12 ∧ q = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2906_290611


namespace NUMINAMATH_CALUDE_cream_cheese_cost_l2906_290671

/-- Cost of items for staff meetings -/
theorem cream_cheese_cost (bagel_cost cream_cheese_cost : ℝ) : 
  2 * bagel_cost + 3 * cream_cheese_cost = 12 →
  4 * bagel_cost + 2 * cream_cheese_cost = 14 →
  cream_cheese_cost = 2.5 := by
sorry

end NUMINAMATH_CALUDE_cream_cheese_cost_l2906_290671


namespace NUMINAMATH_CALUDE_stable_journey_population_l2906_290642

/-- Represents the interstellar vehicle Gibraltar --/
structure Gibraltar where
  full_capacity : ℕ
  family_units : ℕ
  members_per_family : ℕ

/-- Calculates the starting population for a stable journey --/
def starting_population (ship : Gibraltar) : ℕ :=
  ship.full_capacity / 3 - 100

/-- Theorem: The starting population for a stable journey is 300 people --/
theorem stable_journey_population (ship : Gibraltar) 
  (h1 : ship.family_units = 300)
  (h2 : ship.members_per_family = 4)
  (h3 : ship.full_capacity = ship.family_units * ship.members_per_family) :
  starting_population ship = 300 := by
  sorry

#eval starting_population { full_capacity := 1200, family_units := 300, members_per_family := 4 }

end NUMINAMATH_CALUDE_stable_journey_population_l2906_290642


namespace NUMINAMATH_CALUDE_mona_monday_distance_l2906_290601

/-- Represents the distance biked on a given day -/
structure DailyBike where
  distance : ℝ
  time : ℝ
  speed : ℝ

/-- Represents Mona's weekly biking schedule -/
structure WeeklyBike where
  monday : DailyBike
  wednesday : DailyBike
  saturday : DailyBike
  total_distance : ℝ

theorem mona_monday_distance (w : WeeklyBike) :
  w.total_distance = 30 ∧
  w.wednesday.distance = 12 ∧
  w.wednesday.time = 2 ∧
  w.saturday.distance = 2 * w.monday.distance ∧
  w.monday.speed = 15 ∧
  w.monday.time = 1.5 ∧
  w.saturday.speed = 0.8 * w.monday.speed →
  w.monday.distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_mona_monday_distance_l2906_290601


namespace NUMINAMATH_CALUDE_x_value_proof_l2906_290600

theorem x_value_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 3)
  (h2 : y^2 / z = 4)
  (h3 : z^2 / x = 5) :
  x = (6480 : ℝ)^(1/7) := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l2906_290600


namespace NUMINAMATH_CALUDE_percentage_B_of_D_l2906_290632

theorem percentage_B_of_D (A B C D : ℝ) 
  (h1 : B = 1.71 * A)
  (h2 : C = 1.80 * A)
  (h3 : D = 1.90 * B)
  (h4 : B = 1.62 * C)
  (h5 : A = 0.65 * D)
  (h6 : C = 0.55 * D) :
  B = 1.1115 * D := by
sorry

end NUMINAMATH_CALUDE_percentage_B_of_D_l2906_290632


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l2906_290629

/-- The displacement function for the object's motion -/
def displacement (t : ℝ) : ℝ := 2 * t^3

/-- The velocity function, which is the derivative of the displacement function -/
def velocity (t : ℝ) : ℝ := 6 * t^2

theorem instantaneous_velocity_at_3 : velocity 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l2906_290629


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2906_290669

theorem function_passes_through_point (a : ℝ) (h : 0 < a ∧ a < 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a * x - 1
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2906_290669


namespace NUMINAMATH_CALUDE_boxtimes_self_not_always_zero_l2906_290603

-- Define the ⊠ operation
def boxtimes (x y : ℝ) : ℝ := |x + y|

-- Statement to be proven false
theorem boxtimes_self_not_always_zero :
  ¬ (∀ x : ℝ, boxtimes x x = 0) := by
sorry

end NUMINAMATH_CALUDE_boxtimes_self_not_always_zero_l2906_290603


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_average_side_length_l2906_290605

/-- Given a triangle with three sides where the average length of the sides is 12,
    prove that the perimeter of the triangle is 36. -/
theorem triangle_perimeter_from_average_side_length :
  ∀ (a b c : ℝ), (a + b + c) / 3 = 12 → a + b + c = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_average_side_length_l2906_290605


namespace NUMINAMATH_CALUDE_max_value_wxyz_l2906_290639

theorem max_value_wxyz (w x y z : ℝ) 
  (nonneg_w : w ≥ 0) (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_100 : w + x + y + z = 100) : 
  w * x + x * y + y * z ≤ 2500 := by
sorry

end NUMINAMATH_CALUDE_max_value_wxyz_l2906_290639


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2906_290654

theorem arithmetic_sequence_sum_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h1 : ∀ n, S n = n / 2 * (a 1 + a n))
  (h2 : a 7 / a 4 = 2) :
  S 13 / S 7 = 26 / 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2906_290654


namespace NUMINAMATH_CALUDE_feasible_measures_correct_l2906_290661

-- Define the set of all proposed measures
def AllMeasures : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the set of infeasible measures
def InfeasibleMeasures : Set ℕ := {4, 5, 6, 8}

-- Define a predicate for feasible measures
def IsFeasibleMeasure (m : ℕ) : Prop :=
  m ∈ AllMeasures ∧ m ∉ InfeasibleMeasures

-- Define the set of feasible measures
def FeasibleMeasures : Set ℕ := {m ∈ AllMeasures | IsFeasibleMeasure m}

-- Theorem statement
theorem feasible_measures_correct :
  FeasibleMeasures = AllMeasures \ InfeasibleMeasures :=
sorry

end NUMINAMATH_CALUDE_feasible_measures_correct_l2906_290661


namespace NUMINAMATH_CALUDE_infinite_nested_sqrt_three_l2906_290621

theorem infinite_nested_sqrt_three : ∃ x > 0, x^2 = 3 + 2*x ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_infinite_nested_sqrt_three_l2906_290621


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2906_290633

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  Real.sqrt (a + 1) + Real.sqrt (b + 2) ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2906_290633


namespace NUMINAMATH_CALUDE_ellipse_to_hyperbola_l2906_290622

/-- Given an ellipse with equation x²/4 + y²/2 = 1, 
    prove that the equation of the hyperbola with its vertices at the foci of the ellipse 
    and its foci at the vertices of the ellipse is x²/2 - y²/2 = 1 -/
theorem ellipse_to_hyperbola (x y : ℝ) :
  (x^2 / 4 + y^2 / 2 = 1) →
  ∃ (a b : ℝ), (a^2 = 2 ∧ b^2 = 2) ∧
  (x^2 / a^2 - y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_to_hyperbola_l2906_290622


namespace NUMINAMATH_CALUDE_second_year_cost_difference_l2906_290686

/-- Proves that the difference between second and first year payments is 2 --/
theorem second_year_cost_difference (total_payments : ℕ) (first_year : ℕ) (x : ℕ) :
  total_payments = 96 →
  first_year = 20 →
  total_payments = first_year + (first_year + x) + (first_year + x + 3) + (first_year + x + 3 + 4) →
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_year_cost_difference_l2906_290686


namespace NUMINAMATH_CALUDE_optimal_price_l2906_290687

/-- Represents the selling price and corresponding daily sales volume -/
structure PriceSales where
  price : ℝ
  sales : ℝ

/-- The cost price of the fruit in yuan per kilogram -/
def costPrice : ℝ := 22

/-- The initial selling price and sales volume -/
def initialSale : PriceSales :=
  { price := 38, sales := 160 }

/-- The change in sales volume per yuan price reduction -/
def salesIncrease : ℝ := 40

/-- The required daily profit in yuan -/
def requiredProfit : ℝ := 3640

/-- Calculates the daily profit given a selling price -/
def calculateProfit (sellingPrice : ℝ) : ℝ :=
  let priceReduction := initialSale.price - sellingPrice
  let salesVolume := initialSale.sales + salesIncrease * priceReduction
  (sellingPrice - costPrice) * salesVolume

/-- The theorem to be proved -/
theorem optimal_price :
  ∃ (optimalPrice : ℝ),
    calculateProfit optimalPrice = requiredProfit ∧
    optimalPrice = 29 ∧
    ∀ (price : ℝ),
      calculateProfit price = requiredProfit →
      price ≥ optimalPrice :=
sorry

end NUMINAMATH_CALUDE_optimal_price_l2906_290687


namespace NUMINAMATH_CALUDE_max_product_constraint_l2906_290674

theorem max_product_constraint (x y : ℝ) : 
  x > 0 → y > 0 → x + 2 * y = 1 → x * y ≤ 1/8 := by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l2906_290674


namespace NUMINAMATH_CALUDE_expression_evaluation_l2906_290624

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 2
  ((x + 2*y)^2 + (3*x + y)*(3*x - y) - 3*y*(y - x)) / (2*x) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2906_290624


namespace NUMINAMATH_CALUDE_max_sum_abc_l2906_290662

theorem max_sum_abc (a b c : ℤ) 
  (h1 : a + b = 2006) 
  (h2 : c - a = 2005) 
  (h3 : a < b) : 
  (∀ x y z : ℤ, x + y = 2006 → z - x = 2005 → x < y → x + y + z ≤ a + b + c) ∧ 
  a + b + c = 5013 :=
sorry

end NUMINAMATH_CALUDE_max_sum_abc_l2906_290662


namespace NUMINAMATH_CALUDE_constant_function_derivative_l2906_290612

theorem constant_function_derivative (f : ℝ → ℝ) (h : ∀ x, f x = 7) :
  ∀ x, deriv f x = 0 := by sorry

end NUMINAMATH_CALUDE_constant_function_derivative_l2906_290612


namespace NUMINAMATH_CALUDE_minimize_square_root_difference_l2906_290619

theorem minimize_square_root_difference (p : ℕ) (h_p : Nat.Prime p) (h_p_odd : Odd p) :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ x ≤ y ∧
    (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 0) ∧
    (∀ (a b : ℕ), a > 0 → b > 0 → a ≤ b → 
      Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b ≥ 0 →
      Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≤ Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) ∧
    x = (p - 1) / 2 ∧ y = (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_minimize_square_root_difference_l2906_290619


namespace NUMINAMATH_CALUDE_total_gum_packages_l2906_290627

theorem total_gum_packages : ∀ (robin_pieces_per_package : ℕ) 
                               (robin_extra_pieces : ℕ) 
                               (robin_total_pieces : ℕ)
                               (alex_pieces_per_package : ℕ) 
                               (alex_extra_pieces : ℕ) 
                               (alex_total_pieces : ℕ),
  robin_pieces_per_package = 7 →
  robin_extra_pieces = 6 →
  robin_total_pieces = 41 →
  alex_pieces_per_package = 5 →
  alex_extra_pieces = 3 →
  alex_total_pieces = 23 →
  (robin_total_pieces - robin_extra_pieces) / robin_pieces_per_package +
  (alex_total_pieces - alex_extra_pieces) / alex_pieces_per_package = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_total_gum_packages_l2906_290627


namespace NUMINAMATH_CALUDE_probability_of_two_distinct_roots_l2906_290696

/-- Represents the outcome of rolling two dice where at least one die shows 4 -/
inductive DiceRoll
  | first_four (second : Nat)
  | second_four (first : Nat)
  | both_four

/-- The set of all possible outcomes when rolling two dice and at least one is 4 -/
def all_outcomes : Finset DiceRoll :=
  sorry

/-- Checks if the quadratic equation x^2 + mx + n = 0 has two distinct real roots -/
def has_two_distinct_roots (roll : DiceRoll) : Bool :=
  sorry

/-- The set of outcomes where the equation has two distinct real roots -/
def favorable_outcomes : Finset DiceRoll :=
  sorry

theorem probability_of_two_distinct_roots :
  (Finset.card favorable_outcomes) / (Finset.card all_outcomes) = 5 / 11 :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_distinct_roots_l2906_290696


namespace NUMINAMATH_CALUDE_cousins_distribution_l2906_290635

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 4 rooms available -/
def num_rooms : ℕ := 4

/-- There are 5 cousins to accommodate -/
def num_cousins : ℕ := 5

/-- The number of ways to distribute the cousins is 51 -/
theorem cousins_distribution :
  distribute num_cousins num_rooms = 51 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l2906_290635


namespace NUMINAMATH_CALUDE_stone_slab_length_l2906_290659

theorem stone_slab_length (total_area : ℝ) (num_slabs : ℕ) (slab_length : ℝ) :
  total_area = 72 →
  num_slabs = 50 →
  (slab_length ^ 2) * num_slabs = total_area →
  slab_length = 1.2 := by
sorry

end NUMINAMATH_CALUDE_stone_slab_length_l2906_290659


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_not_q_l2906_290616

/-- Given conditions p and q, prove that p is a sufficient but not necessary condition for ¬q -/
theorem p_sufficient_not_necessary_for_not_q :
  ∀ x : ℝ,
  (0 < x ∧ x ≤ 1) →  -- condition p
  ((1 / x < 1) → False) →  -- ¬q
  ∃ y : ℝ, ((1 / y < 1) → False) ∧ ¬(0 < y ∧ y ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_not_q_l2906_290616


namespace NUMINAMATH_CALUDE_range_of_a_l2906_290608

def has_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - a*x + 4 = 0

def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x ≥ 3, (4*x + a) ≥ 0

theorem range_of_a (a : ℝ) :
  has_real_roots a ∧ is_increasing_on_interval a →
  a ∈ Set.Icc (-12) (-4) ∪ Set.Ioi 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2906_290608


namespace NUMINAMATH_CALUDE_vector_linear_combination_l2906_290634

/-- Given vectors a, b, and c in R^2, prove that if c = x*a + y*b,
    then x + y = 8/3 -/
theorem vector_linear_combination (a b c : ℝ × ℝ) (x y : ℝ) 
    (h1 : a = (2, 3))
    (h2 : b = (3, 3))
    (h3 : c = (7, 8))
    (h4 : c = x • a + y • b) :
  x + y = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l2906_290634


namespace NUMINAMATH_CALUDE_sin_585_degrees_l2906_290660

theorem sin_585_degrees :
  Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l2906_290660


namespace NUMINAMATH_CALUDE_evaluate_expression_l2906_290637

theorem evaluate_expression (x z : ℝ) (hx : x = 2) (hz : z = 1) :
  z * (z - 4 * x) = -7 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2906_290637


namespace NUMINAMATH_CALUDE_card_game_unfair_l2906_290676

/-- Represents a playing card with a rank and suit -/
structure Card :=
  (rank : Nat)
  (suit : Nat)

/-- Represents the deck of cards -/
def Deck : Finset Card := sorry

/-- The number of cards in the deck -/
def deckSize : Nat := Finset.card Deck

/-- Volodya's draw -/
def volodyaDraw : Deck → Card := sorry

/-- Masha's draw -/
def mashaDraw : Deck → Card → Card := sorry

/-- Masha wins if her card rank is higher than Volodya's -/
def mashaWins (vCard mCard : Card) : Prop := mCard.rank > vCard.rank

/-- The probability of Masha winning -/
def probMashaWins : ℝ := sorry

/-- Theorem: The card game is unfair (biased against Masha) -/
theorem card_game_unfair : probMashaWins < 1/2 := by sorry

end NUMINAMATH_CALUDE_card_game_unfair_l2906_290676


namespace NUMINAMATH_CALUDE_consecutive_primes_sum_composite_l2906_290666

theorem consecutive_primes_sum_composite (p₁ p₂ q : ℕ) : 
  Nat.Prime p₁ → Nat.Prime p₂ → 
  Odd p₁ → Odd p₂ → 
  p₁ < p₂ → 
  ¬∃k, Nat.Prime k ∧ p₁ < k ∧ k < p₂ →
  p₁ + p₂ = 2 * q → 
  ¬(Nat.Prime q) := by
sorry

end NUMINAMATH_CALUDE_consecutive_primes_sum_composite_l2906_290666


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2906_290699

theorem complex_fraction_evaluation :
  let expr := (0.128 / 3.2 + 0.86) / ((5/6) * 1.2 + 0.8) * ((1 + 32/63 - 13/21) * 3.6) / (0.505 * 2/5 - 0.002)
  expr = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2906_290699


namespace NUMINAMATH_CALUDE_cylinder_volume_relationship_l2906_290695

/-- Theorem about the volumes of two cylinders with specific relationships -/
theorem cylinder_volume_relationship (h r_C r_D h_C h_D : ℝ) 
  (h_positive : h > 0)
  (cylinder_C : r_C = h ∧ h_C = 3 * r_D)
  (cylinder_D : r_D = h / 3 ∧ h_D = h)
  (volume_ratio : π * r_D^2 * h_D = 3 * (π * r_C^2 * h_C)) :
  π * r_D^2 * h_D = 3 * π * h^3 :=
sorry

end NUMINAMATH_CALUDE_cylinder_volume_relationship_l2906_290695


namespace NUMINAMATH_CALUDE_perpendicular_sum_difference_l2906_290610

/-- Given unit vectors a and b in the plane, prove that (a + b) is perpendicular to (a - b) -/
theorem perpendicular_sum_difference (a b : ℝ × ℝ) 
  (ha : a = (5/13, 12/13)) 
  (hb : b = (4/5, 3/5)) 
  (unit_a : a.1^2 + a.2^2 = 1) 
  (unit_b : b.1^2 + b.2^2 = 1) : 
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2) = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_sum_difference_l2906_290610


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l2906_290672

theorem quadratic_root_implies_k (k : ℝ) : 
  (1 : ℝ)^2 + k*(1 : ℝ) - 3 = 0 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l2906_290672


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l2906_290602

theorem product_of_sum_and_difference : 
  let a : ℝ := 4.93
  let b : ℝ := 3.78
  (a + b) * (a - b) = 10.0165 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l2906_290602


namespace NUMINAMATH_CALUDE_geometric_mean_of_45_and_80_l2906_290617

theorem geometric_mean_of_45_and_80 : 
  ∃ x : ℝ, (x ^ 2 = 45 * 80) ∧ (x = 60 ∨ x = -60) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_45_and_80_l2906_290617


namespace NUMINAMATH_CALUDE_cyclist_stump_problem_l2906_290613

/-- Represents the problem of cyclists on a road with stumps -/
theorem cyclist_stump_problem 
  (road_length : ℝ)
  (speed_1 speed_2 : ℝ)
  (rest_time : ℕ)
  (num_stumps : ℕ) :
  road_length = 37 →
  speed_1 = 15 →
  speed_2 = 20 →
  rest_time > 0 →
  num_stumps > 1 →
  (road_length / speed_1 + num_stumps * rest_time / 60) =
  (road_length / speed_2 + num_stumps * (2 * rest_time) / 60) →
  num_stumps = 37 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_stump_problem_l2906_290613


namespace NUMINAMATH_CALUDE_solution_part_i_solution_part_ii_l2906_290636

/-- The function f(x) defined as |x - a| + |x - 1| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 1|

/-- Theorem for part (I) of the problem -/
theorem solution_part_i :
  let a : ℝ := 2
  {x : ℝ | f a x < 4} = {x : ℝ | -1/2 < x ∧ x < 7/2} := by sorry

/-- Theorem for part (II) of the problem -/
theorem solution_part_ii :
  {a : ℝ | ∀ x, f a x ≥ 2} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_part_i_solution_part_ii_l2906_290636


namespace NUMINAMATH_CALUDE_range_of_a_l2906_290618

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

theorem range_of_a (a : ℝ) :
  (a ≠ 0) →
  (∀ x, ¬q x → ¬p x a) →
  (a ≥ 2 ∨ a ≤ -4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2906_290618


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2906_290628

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∃ (x y : ℝ), x + y = r ∧ x^2 + y^2 = r) → 
  (∀ (x y : ℝ), x + y = r → x^2 + y^2 ≥ r) → 
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2906_290628


namespace NUMINAMATH_CALUDE_intersection_M_N_l2906_290606

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def N : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2906_290606


namespace NUMINAMATH_CALUDE_min_sum_positive_reals_l2906_290641

theorem min_sum_positive_reals (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2 * a + 8 * b - a * b = 0 → x + y ≤ a + b ∧ x + y = 6 :=
sorry

end NUMINAMATH_CALUDE_min_sum_positive_reals_l2906_290641


namespace NUMINAMATH_CALUDE_well_diameter_is_six_l2906_290697

def well_depth : ℝ := 24
def well_volume : ℝ := 678.5840131753953

theorem well_diameter_is_six :
  ∃ (d : ℝ), d = 6 ∧ well_volume = π * (d / 2)^2 * well_depth := by sorry

end NUMINAMATH_CALUDE_well_diameter_is_six_l2906_290697


namespace NUMINAMATH_CALUDE_inequality_proof_l2906_290692

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2906_290692


namespace NUMINAMATH_CALUDE_milk_container_problem_l2906_290668

/-- The initial quantity of milk in container A --/
def initial_quantity : ℝ := 1232

/-- The fraction of container A's capacity that goes into container B --/
def fraction_in_B : ℝ := 0.375

/-- The amount transferred from C to B to equalize the quantities --/
def transfer_amount : ℝ := 154

theorem milk_container_problem :
  -- Container A is filled to its brim
  -- All milk from A is poured into B and C
  -- Quantity in B is 62.5% less than A (which means it's 37.5% of A)
  -- If 154L is transferred from C to B, they become equal
  (fraction_in_B * initial_quantity + transfer_amount = 
   (1 - fraction_in_B) * initial_quantity - transfer_amount) →
  -- Then the initial quantity in A is 1232 liters
  initial_quantity = 1232 := by
  sorry

end NUMINAMATH_CALUDE_milk_container_problem_l2906_290668


namespace NUMINAMATH_CALUDE_power_addition_equality_l2906_290615

theorem power_addition_equality : 2^345 + 9^4 / 9^2 = 2^345 + 81 := by
  sorry

end NUMINAMATH_CALUDE_power_addition_equality_l2906_290615


namespace NUMINAMATH_CALUDE_negation_equivalence_l2906_290658

-- Define the set [-1, 3]
def interval : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- Define the original proposition
def original_prop (a : ℝ) : Prop := ∀ x ∈ interval, x^2 - a ≥ 0

-- Define the negation of the proposition
def negation_prop (a : ℝ) : Prop := ∃ x ∈ interval, x^2 - a < 0

-- Theorem stating that the negation of the original proposition is equivalent to negation_prop
theorem negation_equivalence (a : ℝ) : ¬(original_prop a) ↔ negation_prop a := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2906_290658


namespace NUMINAMATH_CALUDE_min_red_pieces_l2906_290679

theorem min_red_pieces (w b r : ℕ) : 
  b ≥ w / 2 →
  b ≤ r / 3 →
  w + b ≥ 55 →
  r ≥ 57 ∧ ∀ r', (∃ w' b', b' ≥ w' / 2 ∧ b' ≤ r' / 3 ∧ w' + b' ≥ 55) → r' ≥ r :=
by sorry

end NUMINAMATH_CALUDE_min_red_pieces_l2906_290679


namespace NUMINAMATH_CALUDE_book_sale_price_l2906_290640

theorem book_sale_price (total_books : ℕ) (sold_books : ℕ) (unsold_books : ℕ) (total_amount : ℕ) : 
  sold_books = (2 : ℕ) * total_books / 3 →
  unsold_books = 36 →
  sold_books + unsold_books = total_books →
  total_amount = 288 →
  total_amount / sold_books = 4 := by
sorry

end NUMINAMATH_CALUDE_book_sale_price_l2906_290640


namespace NUMINAMATH_CALUDE_exponential_inequality_l2906_290644

theorem exponential_inequality (x : ℝ) : 
  Real.exp (2 * x - 1) < 1 ↔ x < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2906_290644


namespace NUMINAMATH_CALUDE_lillian_initial_candies_l2906_290685

/-- The number of candies Lillian's father gave her -/
def candies_from_father : ℕ := 5

/-- The total number of candies Lillian has after receiving candies from her father -/
def total_candies : ℕ := 93

/-- The number of candies Lillian collected initially -/
def initial_candies : ℕ := total_candies - candies_from_father

theorem lillian_initial_candies :
  initial_candies = 88 :=
by sorry

end NUMINAMATH_CALUDE_lillian_initial_candies_l2906_290685


namespace NUMINAMATH_CALUDE_contemporary_probability_correct_l2906_290647

/-- The duration in years of the period considered -/
def period : ℕ := 800

/-- The lifespan of each mathematician in years -/
def lifespan : ℕ := 150

/-- The probability that two mathematicians born within a given period
    are contemporaries, given their lifespans and assuming uniform distribution
    of birth years -/
def contemporaryProbability (p : ℕ) (l : ℕ) : ℚ :=
  let totalArea := p * p
  let nonOverlapArea := 2 * (p - l) * l / 2
  let overlapArea := totalArea - nonOverlapArea
  overlapArea / totalArea

theorem contemporary_probability_correct :
  contemporaryProbability period lifespan = 27125 / 32000 := by
  sorry

end NUMINAMATH_CALUDE_contemporary_probability_correct_l2906_290647


namespace NUMINAMATH_CALUDE_sector_area_l2906_290643

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 5) (h2 : θ = 2) :
  (1 / 2) * r^2 * θ = 25 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2906_290643


namespace NUMINAMATH_CALUDE_broken_seashells_l2906_290657

theorem broken_seashells (total : ℕ) (unbroken : ℕ) (broken : ℕ) : 
  total = 7 → unbroken = 3 → broken = total - unbroken → broken = 4 := by
sorry

end NUMINAMATH_CALUDE_broken_seashells_l2906_290657


namespace NUMINAMATH_CALUDE_jim_travels_two_miles_l2906_290688

/-- The distance John travels in miles -/
def john_distance : ℝ := 15

/-- The difference between John's and Jill's travel distances in miles -/
def distance_difference : ℝ := 5

/-- The percentage of Jill's distance that Jim travels -/
def jim_percentage : ℝ := 0.20

/-- Jill's travel distance in miles -/
def jill_distance : ℝ := john_distance - distance_difference

/-- Jim's travel distance in miles -/
def jim_distance : ℝ := jill_distance * jim_percentage

theorem jim_travels_two_miles :
  jim_distance = 2 := by sorry

end NUMINAMATH_CALUDE_jim_travels_two_miles_l2906_290688


namespace NUMINAMATH_CALUDE_simplify_expression_l2906_290646

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^2)^11 = 5368709120 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2906_290646


namespace NUMINAMATH_CALUDE_three_sevenths_minus_forty_percent_l2906_290665

theorem three_sevenths_minus_forty_percent (x : ℝ) : 
  (0.3 * x = 63.0000000000001) → 
  ((3/7) * x - 0.4 * x = 6.00000000000006) := by
sorry

end NUMINAMATH_CALUDE_three_sevenths_minus_forty_percent_l2906_290665


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l2906_290648

/-- Theorem: Volume of cylinder X in terms of cylinder Y's height -/
theorem cylinder_volume_relation (h : ℝ) (h_pos : h > 0) : ∃ (r_x r_y h_x : ℝ),
  r_y = 2 * h_x ∧ 
  h_x = 3 * r_y ∧ 
  h_x = 3 * h ∧
  r_x = 6 * h ∧
  π * r_x^2 * h_x = 3 * (π * r_y^2 * h) ∧
  π * r_x^2 * h_x = 108 * π * h^3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l2906_290648


namespace NUMINAMATH_CALUDE_probability_at_least_one_head_l2906_290690

theorem probability_at_least_one_head (p : ℝ) (n : ℕ) : 
  p = 1/2 → n = 3 → 1 - (1 - p)^n = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_head_l2906_290690


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_square_solution_power_of_two_equals_square_plus_one_solution_l2906_290652

theorem power_of_two_plus_one_square_solution (n x : ℕ+) :
  2^(n:ℕ) + 1 = (x:ℕ)^2 ↔ n = 3 ∧ x = 3 :=
sorry

theorem power_of_two_equals_square_plus_one_solution (n x : ℕ+) :
  2^(n:ℕ) = (x:ℕ)^2 + 1 ↔ n = 1 ∧ x = 1 :=
sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_square_solution_power_of_two_equals_square_plus_one_solution_l2906_290652


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_xy_l2906_290620

theorem factorization_x_squared_minus_xy (x y : ℝ) : x^2 - x*y = x*(x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_xy_l2906_290620


namespace NUMINAMATH_CALUDE_square_of_complex_2_minus_i_l2906_290693

theorem square_of_complex_2_minus_i :
  let z : ℂ := 2 - I
  z^2 = 3 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_square_of_complex_2_minus_i_l2906_290693


namespace NUMINAMATH_CALUDE_total_topping_combinations_l2906_290614

/-- Represents the number of cheese options -/
def cheese_options : ℕ := 3

/-- Represents the number of meat options -/
def meat_options : ℕ := 4

/-- Represents the number of vegetable options -/
def vegetable_options : ℕ := 5

/-- Represents whether pepperoni is a meat option -/
def pepperoni_is_meat_option : Prop := True

/-- Represents whether peppers is a vegetable option -/
def peppers_is_vegetable_option : Prop := True

/-- Represents the restriction that pepperoni and peppers cannot be chosen together -/
def pepperoni_peppers_restriction : Prop := True

/-- Theorem stating the total number of pizza topping combinations -/
theorem total_topping_combinations : 
  cheese_options * meat_options * vegetable_options - 
  cheese_options * (meat_options - 1) = 57 := by
  sorry


end NUMINAMATH_CALUDE_total_topping_combinations_l2906_290614


namespace NUMINAMATH_CALUDE_sphere_expansion_l2906_290609

/-- Given a sphere with initial radius 1 and final radius m, 
    if the volume expansion rate is 28π/3, then m = 2 -/
theorem sphere_expansion (m : ℝ) : 
  m > 0 →  -- Ensure m is positive (as it's a radius)
  (4 * π / 3 * (m^3 - 1)) / (m - 1) = 28 * π / 3 →
  m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_sphere_expansion_l2906_290609


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2906_290673

theorem simplify_square_roots : 
  (Real.sqrt 338 / Real.sqrt 288) + (Real.sqrt 150 / Real.sqrt 96) = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2906_290673


namespace NUMINAMATH_CALUDE_block_rotation_theorem_l2906_290653

/-- Represents a rectangular block with three dimensions -/
structure Block where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Represents a square board -/
structure Board where
  size : ℕ

/-- Represents a face of the block -/
inductive Face
  | X
  | Y
  | Z

/-- Calculates the area of a given face of the block -/
def faceArea (b : Block) (f : Face) : ℕ :=
  match f with
  | Face.X => b.x * b.y
  | Face.Y => b.x * b.z
  | Face.Z => b.y * b.z

/-- Represents a sequence of rotations -/
def Rotations := List Face

/-- Calculates the number of unique squares contacted after a series of rotations -/
def uniqueSquaresContacted (block : Block) (board : Board) (rotations : Rotations) : ℕ :=
  sorry  -- Implementation details omitted

theorem block_rotation_theorem (block : Block) (board : Board) (rotations : Rotations) :
  block.x = 1 ∧ block.y = 2 ∧ block.z = 3 ∧
  board.size = 8 ∧
  rotations = [Face.X, Face.Y, Face.Z, Face.X, Face.Y, Face.Z] →
  uniqueSquaresContacted block board rotations = 19 :=
by sorry

end NUMINAMATH_CALUDE_block_rotation_theorem_l2906_290653


namespace NUMINAMATH_CALUDE_incorrect_calculation_l2906_290683

theorem incorrect_calculation : 3 * Real.sqrt 3 - Real.sqrt 3 ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l2906_290683


namespace NUMINAMATH_CALUDE_mean_of_four_integers_l2906_290604

theorem mean_of_four_integers (x : ℤ) : 
  (78 + 83 + 82 + x) / 4 = 80 → x = 77 ∧ x = 80 - 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_four_integers_l2906_290604


namespace NUMINAMATH_CALUDE_equation_solutions_l2906_290670

theorem equation_solutions :
  (∀ x : ℝ, x * (x + 2) - (x + 2) = 0 ↔ x = -2 ∨ x = 1) ∧
  (∀ x : ℝ, x^2 - 4*x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2906_290670


namespace NUMINAMATH_CALUDE_competition_score_l2906_290630

theorem competition_score (correct_points incorrect_points total_questions final_score : ℕ) 
  (h1 : correct_points = 6)
  (h2 : incorrect_points = 3)
  (h3 : total_questions = 15)
  (h4 : final_score = 36) :
  ∃ (correct_answers : ℕ),
    correct_answers * correct_points - (total_questions - correct_answers) * incorrect_points = final_score ∧
    correct_answers = 9 := by
  sorry

#check competition_score

end NUMINAMATH_CALUDE_competition_score_l2906_290630


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2906_290680

/-- The equation of the tangent line to the circle x^2 + y^2 - 4x = 0 at the point (1, √3) is x - √3y + 2 = 0. -/
theorem tangent_line_equation (x y : ℝ) :
  let circle_equation := (x^2 + y^2 - 4*x = 0)
  let point_on_circle := (1, Real.sqrt 3)
  let tangent_line := (x - Real.sqrt 3 * y + 2 = 0)
  circle_equation ∧ (x, y) = point_on_circle → tangent_line := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2906_290680
