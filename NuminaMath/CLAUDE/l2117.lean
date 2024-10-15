import Mathlib

namespace NUMINAMATH_CALUDE_total_rose_bushes_l2117_211732

theorem total_rose_bushes (rose_cost : ℕ) (aloe_cost : ℕ) (friend_roses : ℕ) (total_spent : ℕ) (aloe_count : ℕ) : 
  rose_cost = 75 → 
  friend_roses = 2 → 
  aloe_cost = 100 → 
  aloe_count = 2 → 
  total_spent = 500 → 
  (total_spent - aloe_count * aloe_cost) / rose_cost + friend_roses = 6 := by
sorry

end NUMINAMATH_CALUDE_total_rose_bushes_l2117_211732


namespace NUMINAMATH_CALUDE_diagonal_intersects_n_rhombuses_l2117_211774

/-- A regular hexagon with side length n -/
structure RegularHexagon (n : ℕ) where
  side_length : ℕ
  is_positive : 0 < side_length
  eq_n : side_length = n

/-- A rhombus with internal angles 60° and 120° -/
structure Rhombus where
  internal_angles : Fin 2 → ℝ
  angle_sum : internal_angles 0 + internal_angles 1 = 180
  angles_correct : (internal_angles 0 = 60 ∧ internal_angles 1 = 120) ∨ 
                   (internal_angles 0 = 120 ∧ internal_angles 1 = 60)

/-- Theorem: The diagonal of a regular hexagon intersects n rhombuses -/
theorem diagonal_intersects_n_rhombuses (n : ℕ) (h : RegularHexagon n) :
  ∃ (rhombuses : Finset Rhombus),
    (Finset.card rhombuses = 3 * n^2) ∧
    (∃ (intersected : Finset Rhombus),
      Finset.card intersected = n ∧
      intersected ⊆ rhombuses) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersects_n_rhombuses_l2117_211774


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2117_211726

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 40) (h3 : x - y = 10) :
  (7 : ℝ) * (375 / 7) = k := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2117_211726


namespace NUMINAMATH_CALUDE_divisible_by_seven_l2117_211779

theorem divisible_by_seven : ∃ k : ℤ, (1 + 5)^4 - 1 = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l2117_211779


namespace NUMINAMATH_CALUDE_paper_boat_travel_time_l2117_211794

/-- Represents the problem of calculating the time for a paper boat to travel along an embankment --/
theorem paper_boat_travel_time 
  (embankment_length : ℝ)
  (boat_length : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (h1 : embankment_length = 50)
  (h2 : boat_length = 10)
  (h3 : downstream_time = 5)
  (h4 : upstream_time = 4) :
  let downstream_speed := embankment_length / downstream_time
  let upstream_speed := embankment_length / upstream_time
  let boat_speed := (downstream_speed + upstream_speed) / 2
  let current_speed := (downstream_speed - upstream_speed) / 2
  (embankment_length / current_speed) = 40 := by
  sorry

end NUMINAMATH_CALUDE_paper_boat_travel_time_l2117_211794


namespace NUMINAMATH_CALUDE_james_rainwater_profit_l2117_211785

/-- Calculates the money James made from selling rainwater collected over two days -/
theorem james_rainwater_profit : 
  let gallons_per_inch : ℝ := 15
  let monday_rain : ℝ := 4
  let tuesday_rain : ℝ := 3
  let price_per_gallon : ℝ := 1.2
  let total_gallons := gallons_per_inch * (monday_rain + tuesday_rain)
  total_gallons * price_per_gallon = 126 := by
sorry


end NUMINAMATH_CALUDE_james_rainwater_profit_l2117_211785


namespace NUMINAMATH_CALUDE_election_votes_l2117_211722

theorem election_votes (marcy barry joey : ℕ) : 
  marcy = 3 * barry → 
  barry = 2 * (joey + 3) → 
  marcy = 66 → 
  joey = 8 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l2117_211722


namespace NUMINAMATH_CALUDE_farthest_corner_distance_l2117_211737

/-- Represents a rectangular pool with given dimensions -/
structure Pool :=
  (length : ℝ)
  (width : ℝ)

/-- Calculates the perimeter of a rectangular pool -/
def perimeter (p : Pool) : ℝ := 2 * (p.length + p.width)

/-- Theorem: In a 10m × 25m pool, if three children walk 50m total,
    the distance to the farthest corner is 20m -/
theorem farthest_corner_distance (p : Pool) 
  (h1 : p.length = 25)
  (h2 : p.width = 10)
  (h3 : ∃ (x : ℝ), x ≥ 0 ∧ x ≤ perimeter p ∧ perimeter p - x = 50) :
  ∃ (y : ℝ), y = 20 ∧ y = perimeter p - 50 := by
  sorry

end NUMINAMATH_CALUDE_farthest_corner_distance_l2117_211737


namespace NUMINAMATH_CALUDE_intersection_slope_l2117_211724

/-- Given two lines p and q that intersect at (-3, -9), prove that the slope of line q is 0 -/
theorem intersection_slope (k : ℝ) : 
  (∀ x y : ℝ, y = 4*x + 3 → y = k*x - 9 → x = -3 ∧ y = -9) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l2117_211724


namespace NUMINAMATH_CALUDE_third_participant_score_l2117_211767

/-- Represents the score of a participant -/
structure ParticipantScore where
  score : ℕ

/-- Represents the total number of competitions -/
def totalCompetitions : ℕ := 10

/-- Represents the total points awarded in each competition -/
def pointsPerCompetition : ℕ := 4

/-- Theorem: Given the conditions of the competition and two participants' scores,
    the third participant's score is determined -/
theorem third_participant_score 
  (dima misha : ParticipantScore)
  (h1 : dima.score = 22)
  (h2 : misha.score = 8) :
  ∃ yura : ParticipantScore, yura.score = 10 := by
  sorry

end NUMINAMATH_CALUDE_third_participant_score_l2117_211767


namespace NUMINAMATH_CALUDE_intersection_area_of_specific_circles_l2117_211733

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of intersection of two circles -/
def intersectionArea (c1 c2 : Circle) : ℝ := sorry

/-- The first circle centered at (3,0) with radius 3 -/
def circle1 : Circle := { center := (3, 0), radius := 3 }

/-- The second circle centered at (0,3) with radius 3 -/
def circle2 : Circle := { center := (0, 3), radius := 3 }

/-- Theorem stating the area of intersection of the two given circles -/
theorem intersection_area_of_specific_circles :
  intersectionArea circle1 circle2 = (9 * Real.pi - 18) / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_area_of_specific_circles_l2117_211733


namespace NUMINAMATH_CALUDE_solution_system_equations_l2117_211738

theorem solution_system_equations (x y z : ℝ) : 
  (x^2 - y^2 + z = 27 / (x * y) ∧
   y^2 - z^2 + x = 27 / (y * z) ∧
   z^2 - x^2 + y = 27 / (x * z)) →
  ((x = 3 ∧ y = 3 ∧ z = 3) ∨
   (x = -3 ∧ y = -3 ∧ z = 3) ∨
   (x = -3 ∧ y = 3 ∧ z = -3) ∨
   (x = 3 ∧ y = -3 ∧ z = -3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l2117_211738


namespace NUMINAMATH_CALUDE_boys_in_class_l2117_211783

theorem boys_in_class (total : ℕ) (girls_ratio boys_ratio others_ratio : ℕ) 
  (h1 : total = 63)
  (h2 : girls_ratio = 4)
  (h3 : boys_ratio = 3)
  (h4 : others_ratio = 2)
  (h5 : ∃ k : ℕ, total = k * (girls_ratio + boys_ratio + others_ratio)) :
  ∃ num_boys : ℕ, num_boys = 21 ∧ num_boys * (girls_ratio + boys_ratio + others_ratio) = boys_ratio * total :=
by
  sorry

#check boys_in_class

end NUMINAMATH_CALUDE_boys_in_class_l2117_211783


namespace NUMINAMATH_CALUDE_phone_not_answered_probability_l2117_211789

theorem phone_not_answered_probability 
  (p1 p2 p3 p4 : ℝ) 
  (h1 : p1 = 0.1) 
  (h2 : p2 = 0.3) 
  (h3 : p3 = 0.4) 
  (h4 : p4 = 0.1) : 
  1 - (p1 + p2 + p3 + p4) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_phone_not_answered_probability_l2117_211789


namespace NUMINAMATH_CALUDE_f_composition_l2117_211795

def f (x : ℝ) : ℝ := 2 * x + 1

theorem f_composition : ∀ x : ℝ, f (f x) = 4 * x + 3 := by sorry

end NUMINAMATH_CALUDE_f_composition_l2117_211795


namespace NUMINAMATH_CALUDE_monotonic_interval_implies_a_bound_l2117_211740

open Real

theorem monotonic_interval_implies_a_bound (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, (fun x => 1/x + 2*a*x - 2) > 0) →
  a > -1/2 := by
sorry

end NUMINAMATH_CALUDE_monotonic_interval_implies_a_bound_l2117_211740


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2117_211797

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ a c : ℝ, c^2 * g a = a^2 * g c

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h1 : FunctionalEquation g) (h2 : g 3 ≠ 0) : 
  (g 6 - g 2) / g 3 = 32 / 9 := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2117_211797


namespace NUMINAMATH_CALUDE_faye_initial_apps_l2117_211736

/-- Represents the number of apps Faye had initially -/
def initial_apps : ℕ := sorry

/-- Represents the number of apps Faye deleted -/
def deleted_apps : ℕ := 8

/-- Represents the number of apps Faye had left after deleting -/
def remaining_apps : ℕ := 4

/-- Theorem stating that the initial number of apps was 12 -/
theorem faye_initial_apps : initial_apps = 12 := by
  sorry

end NUMINAMATH_CALUDE_faye_initial_apps_l2117_211736


namespace NUMINAMATH_CALUDE_bus_tour_tickets_l2117_211787

/-- Represents the total number of tickets sold in a local bus tour. -/
def total_tickets (senior_tickets : ℕ) (regular_tickets : ℕ) : ℕ :=
  senior_tickets + regular_tickets

/-- Represents the total sales from tickets. -/
def total_sales (senior_tickets : ℕ) (regular_tickets : ℕ) : ℕ :=
  10 * senior_tickets + 15 * regular_tickets

theorem bus_tour_tickets :
  ∃ (senior_tickets : ℕ),
    total_tickets senior_tickets 41 = 65 ∧
    total_sales senior_tickets 41 = 855 :=
by sorry

end NUMINAMATH_CALUDE_bus_tour_tickets_l2117_211787


namespace NUMINAMATH_CALUDE_factor_calculation_l2117_211781

theorem factor_calculation (x : ℝ) (f : ℝ) : 
  x = 22.142857142857142 → 
  ((x + 5) * f / 5) - 5 = 66 / 2 → 
  f = 7 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l2117_211781


namespace NUMINAMATH_CALUDE_truncated_cube_volume_ratio_l2117_211720

/-- A convex polyhedron with specific properties -/
structure TruncatedCube where
  /-- The polyhedron has 6 square faces -/
  square_faces : Nat
  /-- The polyhedron has 8 equilateral triangle faces -/
  triangle_faces : Nat
  /-- Each edge is shared between one triangle and one square -/
  shared_edges : Bool
  /-- All dihedral angles between triangles and squares are equal -/
  equal_dihedral_angles : Bool
  /-- The polyhedron can be circumscribed by a sphere -/
  circumscribable : Bool
  /-- Properties of the truncated cube -/
  h_square_faces : square_faces = 6
  h_triangle_faces : triangle_faces = 8
  h_shared_edges : shared_edges = true
  h_equal_dihedral_angles : equal_dihedral_angles = true
  h_circumscribable : circumscribable = true

/-- The theorem stating the ratio of squared volumes -/
theorem truncated_cube_volume_ratio (tc : TruncatedCube) :
  ∃ (v_polyhedron v_sphere : ℝ),
    v_polyhedron > 0 ∧ v_sphere > 0 ∧
    (v_polyhedron / v_sphere)^2 = 25 / (8 * Real.pi^2) :=
sorry

end NUMINAMATH_CALUDE_truncated_cube_volume_ratio_l2117_211720


namespace NUMINAMATH_CALUDE_circular_film_radius_l2117_211711

/-- The radius of a circular film formed by pouring a cylindrical container of liquid onto water -/
theorem circular_film_radius 
  (h : ℝ) -- height of the cylindrical container
  (d : ℝ) -- diameter of the cylindrical container
  (t : ℝ) -- thickness of the resulting circular film
  (h_pos : h > 0)
  (d_pos : d > 0)
  (t_pos : t > 0)
  (h_val : h = 10)
  (d_val : d = 5)
  (t_val : t = 0.2) :
  ∃ (r : ℝ), r^2 = 312.5 ∧ π * (d/2)^2 * h = π * r^2 * t :=
by sorry

end NUMINAMATH_CALUDE_circular_film_radius_l2117_211711


namespace NUMINAMATH_CALUDE_total_liquid_consumed_l2117_211706

/-- The amount of cups in one pint -/
def cups_per_pint : ℝ := 2

/-- The amount of cups in one liter -/
def cups_per_liter : ℝ := 4.22675

/-- The amount of pints Elijah drank -/
def elijah_pints : ℝ := 8.5

/-- The amount of pints Emilio drank -/
def emilio_pints : ℝ := 9.5

/-- The amount of liters Isabella drank -/
def isabella_liters : ℝ := 3

/-- The total cups of liquid consumed by Elijah, Emilio, and Isabella -/
def total_cups : ℝ := elijah_pints * cups_per_pint + emilio_pints * cups_per_pint + isabella_liters * cups_per_liter

theorem total_liquid_consumed :
  total_cups = 48.68025 := by sorry

end NUMINAMATH_CALUDE_total_liquid_consumed_l2117_211706


namespace NUMINAMATH_CALUDE_negation_equivalence_l2117_211714

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2117_211714


namespace NUMINAMATH_CALUDE_bus_stop_time_l2117_211764

/-- The time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 54)
  (h2 : speed_with_stops = 45) :
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_time_l2117_211764


namespace NUMINAMATH_CALUDE_g_expression_l2117_211755

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the relationship between f and g
def g_relation (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) = f x

-- Theorem statement
theorem g_expression (g : ℝ → ℝ) (h : g_relation g) :
  ∀ x, g x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l2117_211755


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_eight_l2117_211717

theorem sum_of_solutions_eq_eight : 
  ∃ (N₁ N₂ : ℝ), N₁ * (N₁ - 8) = -7 ∧ N₂ * (N₂ - 8) = -7 ∧ N₁ + N₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_eight_l2117_211717


namespace NUMINAMATH_CALUDE_square_plot_area_l2117_211742

theorem square_plot_area (perimeter : ℝ) (h1 : perimeter * 55 = 3740) : 
  (perimeter / 4) ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_area_l2117_211742


namespace NUMINAMATH_CALUDE_all_figures_on_page_20_only_figures_in_figure5_on_page_20_l2117_211748

/-- Represents a geometric figure in the book --/
structure GeometricFigure where
  page : Nat

/-- Represents the collection of figures shown in Figure 5 --/
def Figure5 : Set GeometricFigure := sorry

/-- The property that distinguishes the figures in Figure 5 --/
def DistinguishingProperty (f : GeometricFigure) : Prop :=
  f.page = 20

/-- Theorem stating that all figures in Figure 5 have the distinguishing property --/
theorem all_figures_on_page_20 :
  ∀ f ∈ Figure5, DistinguishingProperty f :=
sorry

/-- Theorem stating that no other figures have this property --/
theorem only_figures_in_figure5_on_page_20 :
  ∀ f : GeometricFigure, DistinguishingProperty f → f ∈ Figure5 :=
sorry

end NUMINAMATH_CALUDE_all_figures_on_page_20_only_figures_in_figure5_on_page_20_l2117_211748


namespace NUMINAMATH_CALUDE_tom_balloons_l2117_211700

theorem tom_balloons (initial given left : ℕ) : 
  given = 16 → left = 14 → initial = given + left :=
by sorry

end NUMINAMATH_CALUDE_tom_balloons_l2117_211700


namespace NUMINAMATH_CALUDE_fifth_term_is_67_l2117_211725

def sequence_condition (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n + 1) = (s n + s (n + 2)) / 3

theorem fifth_term_is_67 (s : ℕ → ℕ) :
  sequence_condition s →
  s 1 = 3 →
  s 4 = 27 →
  s 5 = 67 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_67_l2117_211725


namespace NUMINAMATH_CALUDE_complex_fraction_inequality_l2117_211705

theorem complex_fraction_inequality (a b c : ℂ) 
  (h1 : a * b + a * c - b * c ≠ 0) 
  (h2 : b * a + b * c - a * c ≠ 0) 
  (h3 : c * a + c * b - a * b ≠ 0) : 
  Complex.abs (a^2 / (a * b + a * c - b * c)) + 
  Complex.abs (b^2 / (b * a + b * c - a * c)) + 
  Complex.abs (c^2 / (c * a + c * b - a * b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_inequality_l2117_211705


namespace NUMINAMATH_CALUDE_price_reduction_for_1750_profit_max_profit_1800_at_20_l2117_211745

-- Define the initial conditions
def initial_sales : ℕ := 40
def initial_profit_per_shirt : ℕ := 40
def sales_increase_rate : ℚ := 2  -- 1 shirt per 0.5 yuan decrease

-- Define the profit function
def profit_function (price_reduction : ℚ) : ℚ :=
  (initial_profit_per_shirt - price_reduction) * (initial_sales + sales_increase_rate * price_reduction)

-- Theorem 1: The price reduction for 1750 yuan profit is 15 yuan
theorem price_reduction_for_1750_profit :
  ∃ (x : ℚ), profit_function x = 1750 ∧ x = 15 := by sorry

-- Theorem 2: The maximum profit is 1800 yuan at 20 yuan price reduction
theorem max_profit_1800_at_20 :
  ∃ (max_profit : ℚ) (optimal_reduction : ℚ),
    max_profit = 1800 ∧
    optimal_reduction = 20 ∧
    (∀ x, profit_function x ≤ max_profit) ∧
    profit_function optimal_reduction = max_profit := by sorry

end NUMINAMATH_CALUDE_price_reduction_for_1750_profit_max_profit_1800_at_20_l2117_211745


namespace NUMINAMATH_CALUDE_julie_monthly_salary_l2117_211741

/-- Calculates the monthly salary for a worker given their hourly rate, hours per day,
    days per week, and number of missed days in a month. -/
def monthly_salary (hourly_rate : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) (missed_days : ℕ) : ℚ :=
  let daily_earnings := hourly_rate * hours_per_day
  let weekly_earnings := daily_earnings * days_per_week
  let monthly_earnings := weekly_earnings * 4
  monthly_earnings - (daily_earnings * missed_days)

/-- Proves that Julie's monthly salary after missing a day of work is $920. -/
theorem julie_monthly_salary :
  monthly_salary 5 8 6 1 = 920 := by
  sorry

end NUMINAMATH_CALUDE_julie_monthly_salary_l2117_211741


namespace NUMINAMATH_CALUDE_problem_solution_l2117_211765

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 10}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - a^2 ≥ 0}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

theorem problem_solution (a : ℝ) (h : a > 0) :
  ((A ∩ B a = ∅) → a ≥ 9) ∧
  ((∀ x, (¬p x → q a x) ∧ (∃ y, q a y ∧ p y)) → (a ≤ 3)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2117_211765


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_l2117_211718

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 8)

-- Define the altitude equation
def altitude_eq (x y : ℝ) : Prop := 6 * x - y - 24 = 0

-- Define the median equation
def median_eq (x y : ℝ) : Prop := y = -15/2 * x + 30

-- Theorem statement
theorem triangle_altitude_and_median :
  (∀ x y : ℝ, altitude_eq x y ↔ 
    (x - A.1) * (B.2 - C.2) = (y - A.2) * (B.1 - C.1) ∧ 
    (x - A.1) * (B.1 - C.1) + (y - A.2) * (B.2 - C.2) = 0) ∧
  (∀ x y : ℝ, median_eq x y ↔ 
    2 * (y - A.2) * (B.1 - C.1) = (x - A.1) * (B.2 + C.2 - 2 * A.2)) :=
sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_l2117_211718


namespace NUMINAMATH_CALUDE_prob_even_sum_three_dice_l2117_211762

/-- The number of faces on each die -/
def num_faces : ℕ := 9

/-- The probability of rolling an even number on one die -/
def p_even : ℚ := 5/9

/-- The probability of rolling an odd number on one die -/
def p_odd : ℚ := 4/9

/-- The probability of getting an even sum when rolling three 9-sided dice -/
theorem prob_even_sum_three_dice : 
  (p_even^3) + 3 * (p_odd^2 * p_even) + 3 * (p_odd * p_even^2) = 665/729 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_three_dice_l2117_211762


namespace NUMINAMATH_CALUDE_new_average_production_l2117_211707

/-- Given the following conditions:
    1. The average daily production for the past n days was 50 units.
    2. Today's production is 90 units.
    3. The value of n is 19 days.
    Prove that the new average daily production is 52 units per day. -/
theorem new_average_production (n : ℕ) (prev_avg : ℝ) (today_prod : ℝ) :
  n = 19 ∧ prev_avg = 50 ∧ today_prod = 90 →
  (n * prev_avg + today_prod) / (n + 1) = 52 := by
  sorry

#check new_average_production

end NUMINAMATH_CALUDE_new_average_production_l2117_211707


namespace NUMINAMATH_CALUDE_probability_A_and_B_selected_l2117_211721

/-- The number of students in the group -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B when choosing 3 students from 5 -/
def prob_select_A_and_B : ℚ := 3 / 10

/-- Theorem stating the probability of selecting both A and B -/
theorem probability_A_and_B_selected :
  (Nat.choose (total_students - 2) (selected_students - 2)) / 
  (Nat.choose total_students selected_students) = prob_select_A_and_B := by
  sorry


end NUMINAMATH_CALUDE_probability_A_and_B_selected_l2117_211721


namespace NUMINAMATH_CALUDE_inequality_contradiction_l2117_211788

theorem inequality_contradiction (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ 
    (a + b) * (c + d) < a * b + c * d ∧ 
    (a + b) * c * d < a * b * (c + d)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l2117_211788


namespace NUMINAMATH_CALUDE_lizzies_garbage_collection_l2117_211780

theorem lizzies_garbage_collection (x : ℝ) 
  (h1 : x + (x - 39) = 735) : x = 387 := by
  sorry

end NUMINAMATH_CALUDE_lizzies_garbage_collection_l2117_211780


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2117_211777

-- Define propositions p and q
variable (p q : Prop)

-- Define the given conditions
variable (h1 : p → q)
variable (h2 : ¬(¬p → ¬q))

-- State the theorem
theorem p_sufficient_not_necessary :
  (∃ (r : Prop), r → q) ∧ ¬(q → p) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2117_211777


namespace NUMINAMATH_CALUDE_cube_root_of_sqrt_64_l2117_211756

theorem cube_root_of_sqrt_64 : (64 : ℝ) ^ (1/2 : ℝ) ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_sqrt_64_l2117_211756


namespace NUMINAMATH_CALUDE_frustum_radius_l2117_211782

theorem frustum_radius (r : ℝ) 
  (h1 : (2 * π * (3 * r)) / (2 * π * r) = 3)
  (h2 : 3 = 3)  -- slant height
  (h3 : π * (r + 3 * r) * 3 = 84 * π) : r = 7 := by
  sorry

end NUMINAMATH_CALUDE_frustum_radius_l2117_211782


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l2117_211790

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l2117_211790


namespace NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_nine_l2117_211747

theorem difference_of_cubes_divisible_by_nine (a b : ℤ) :
  ∃ k : ℤ, (3*a + 2)^3 - (3*b + 2)^3 = 9*k :=
by sorry

end NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_nine_l2117_211747


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2117_211754

theorem fraction_sum_equality : 
  (-3 : ℚ) / 20 + 5 / 200 - 7 / 2000 * 2 = -132 / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2117_211754


namespace NUMINAMATH_CALUDE_equation_solution_l2117_211769

theorem equation_solution (x : ℝ) : 
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 4) ↔ 
  (x = 5 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2117_211769


namespace NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l2117_211713

/-- Given an article with cost C, selling price S, profit factor p, and margin M,
    prove that M can be expressed in terms of S as (p+n)S / (n(2n + p)). -/
theorem margin_in_terms_of_selling_price
  (C S : ℝ) (p n : ℝ) (h_pos : n > 0)
  (h_margin : ∀ M, M = p * (1/n) * C + C)
  (h_selling : S = C + M) :
  ∃ M, M = (p + n) * S / (n * (2 * n + p)) :=
sorry

end NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l2117_211713


namespace NUMINAMATH_CALUDE_union_of_sets_l2117_211799

theorem union_of_sets : 
  let M : Set ℤ := {4, -3}
  let N : Set ℤ := {0, -3}
  M ∪ N = {0, -3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l2117_211799


namespace NUMINAMATH_CALUDE_james_age_when_thomas_grows_l2117_211786

/-- Given the ages and relationships of Thomas, Shay, and James, prove James' age when Thomas reaches his current age. -/
theorem james_age_when_thomas_grows (thomas_age : ℕ) (shay_thomas_diff : ℕ) (james_shay_diff : ℕ) : 
  thomas_age = 6 →
  shay_thomas_diff = 13 →
  james_shay_diff = 5 →
  thomas_age + shay_thomas_diff + james_shay_diff + shay_thomas_diff = 37 :=
by sorry

end NUMINAMATH_CALUDE_james_age_when_thomas_grows_l2117_211786


namespace NUMINAMATH_CALUDE_fraction_simplification_l2117_211766

theorem fraction_simplification : 
  (1 / 4 - 1 / 5) / (1 / 3 - 1 / 4) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2117_211766


namespace NUMINAMATH_CALUDE_coffee_blend_cost_calculation_l2117_211798

/-- The cost of the coffee blend given the prices and amounts of two types of coffee. -/
def coffee_blend_cost (price_a price_b : ℝ) (amount_a : ℝ) : ℝ :=
  amount_a * price_a + 2 * amount_a * price_b

/-- Theorem stating the total cost of the coffee blend under given conditions. -/
theorem coffee_blend_cost_calculation :
  coffee_blend_cost 4.60 5.95 67.52 = 1114.08 := by
  sorry

end NUMINAMATH_CALUDE_coffee_blend_cost_calculation_l2117_211798


namespace NUMINAMATH_CALUDE_novels_difference_l2117_211751

def jordan_novels : ℕ := 120

def alexandre_novels : ℕ := jordan_novels / 10

theorem novels_difference : jordan_novels - alexandre_novels = 108 := by
  sorry

end NUMINAMATH_CALUDE_novels_difference_l2117_211751


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l2117_211734

theorem existence_of_counterexample (x y : ℝ) (h : x > y) :
  ∃ (x y : ℝ), x > y ∧ x^2 - 3 ≤ y^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l2117_211734


namespace NUMINAMATH_CALUDE_base_conversion_sum_l2117_211773

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- Represents the digit C in base 13 -/
def C : Nat := 12

theorem base_conversion_sum : 
  let base_5_num := to_base_10 [2, 4, 3] 5
  let base_13_num := to_base_10 [9, C, 2] 13
  base_5_num + base_13_num = 600 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l2117_211773


namespace NUMINAMATH_CALUDE_function_properties_l2117_211744

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin x + Real.cos x

theorem function_properties (m : ℝ) :
  f m (π / 2) = 1 →
  (∃ T : ℝ, ∀ x : ℝ, f m x = f m (x + T) ∧ T > 0 ∧ ∀ S : ℝ, (∀ x : ℝ, f m x = f m (x + S) ∧ S > 0) → T ≤ S) →
  (∃ M : ℝ, ∀ x : ℝ, f m x ≤ M ∧ ∃ y : ℝ, f m y = M) →
  m = 1 ∧
  (∀ x : ℝ, f m x = Real.sqrt 2 * Real.sin (x + π / 4)) ∧
  (∃ T : ℝ, T = 2 * π ∧ ∀ x : ℝ, f m x = f m (x + T) ∧ T > 0 ∧ ∀ S : ℝ, (∀ x : ℝ, f m x = f m (x + S) ∧ S > 0) → T ≤ S) ∧
  (∃ M : ℝ, M = Real.sqrt 2 ∧ ∀ x : ℝ, f m x ≤ M ∧ ∃ y : ℝ, f m y = M) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2117_211744


namespace NUMINAMATH_CALUDE_emily_necklaces_l2117_211727

def beads_per_necklace : ℕ := 28
def total_beads : ℕ := 308

theorem emily_necklaces :
  (total_beads / beads_per_necklace : ℕ) = 11 :=
by sorry

end NUMINAMATH_CALUDE_emily_necklaces_l2117_211727


namespace NUMINAMATH_CALUDE_original_cost_price_l2117_211719

/-- Given an article with a 15% markup and 20% discount, 
    prove that the original cost is 540 when sold at 496.80 --/
theorem original_cost_price (marked_up_price : ℝ) (selling_price : ℝ) : 
  marked_up_price = 1.15 * 540 ∧ 
  selling_price = 0.8 * marked_up_price ∧
  selling_price = 496.80 → 
  540 = (496.80 : ℝ) / 0.92 := by
sorry

#eval (496.80 : Float) / 0.92

end NUMINAMATH_CALUDE_original_cost_price_l2117_211719


namespace NUMINAMATH_CALUDE_factorization_proof_l2117_211791

theorem factorization_proof (a : ℝ) : (2*a + 1)*a - 4*a - 2 = (2*a + 1)*(a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2117_211791


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2117_211784

theorem root_sum_theorem (a b c : ℝ) : 
  a^3 - 15*a^2 + 22*a - 8 = 0 →
  b^3 - 15*b^2 + 22*b - 8 = 0 →
  c^3 - 15*c^2 + 22*c - 8 = 0 →
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b) = 181/9 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2117_211784


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2117_211772

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2117_211772


namespace NUMINAMATH_CALUDE_inverse_g_inverse_14_l2117_211758

def g (x : ℝ) : ℝ := 3 * x - 4

theorem inverse_g_inverse_14 : 
  (Function.invFun g) ((Function.invFun g) 14) = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_inverse_14_l2117_211758


namespace NUMINAMATH_CALUDE_complex_function_minimum_on_unit_circle_l2117_211770

/-- For a ∈ (0,1) and f(z) = z^2 - z + a, for any complex number z with |z| ≥ 1,
    there exists a complex number z₀ with |z₀| = 1 such that |f(z₀)| ≤ |f(z)| -/
theorem complex_function_minimum_on_unit_circle (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∀ z : ℂ, Complex.abs z ≥ 1 →
    ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧
      Complex.abs (z₀^2 - z₀ + a) ≤ Complex.abs (z^2 - z + a) :=
by sorry

end NUMINAMATH_CALUDE_complex_function_minimum_on_unit_circle_l2117_211770


namespace NUMINAMATH_CALUDE_quadratic_even_function_sum_l2117_211771

/-- A quadratic function of the form f(x) = x^2 + (a-1)x + a + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + (a - 1) * x + a + b

/-- f is an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem quadratic_even_function_sum (a b : ℝ) :
  is_even_function (f a b) → f a b 2 = 0 → a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_even_function_sum_l2117_211771


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_cubed_plus_one_l2117_211793

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_part_of_i_cubed_plus_one (h : i^2 = -1) :
  (i * (i^3 + 1)).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_cubed_plus_one_l2117_211793


namespace NUMINAMATH_CALUDE_total_weekly_batches_l2117_211712

/-- Represents the types of flour --/
inductive FlourType
| Regular
| GlutenFree
| WholeWheat

/-- Represents a day of the week --/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents the flour usage for a single day --/
structure DailyUsage where
  regular : ℕ
  glutenFree : ℕ
  wholeWheat : ℕ
  regularToWholeWheat : ℕ

/-- The number of batches that can be made from one sack of flour --/
def batchesPerSack (t : FlourType) : ℕ :=
  match t with
  | FlourType.Regular => 15
  | FlourType.GlutenFree => 10
  | FlourType.WholeWheat => 12

/-- The conversion rate from regular flour to whole-wheat flour --/
def regularToWholeWheatRate : ℚ := 3/2

/-- The daily flour usage for the week --/
def weekUsage : Day → DailyUsage
| Day.Monday => ⟨4, 3, 2, 0⟩
| Day.Tuesday => ⟨6, 2, 0, 1⟩
| Day.Wednesday => ⟨5, 1, 2, 0⟩
| Day.Thursday => ⟨3, 4, 3, 0⟩
| Day.Friday => ⟨7, 1, 0, 2⟩
| Day.Saturday => ⟨5, 3, 1, 0⟩
| Day.Sunday => ⟨2, 4, 0, 2⟩

/-- Calculates the total number of batches for a given flour type in a week --/
def totalBatches (t : FlourType) : ℕ := sorry

/-- The main theorem: Bruce can make 846 batches of pizza dough in a week --/
theorem total_weekly_batches : (totalBatches FlourType.Regular) + 
                               (totalBatches FlourType.GlutenFree) + 
                               (totalBatches FlourType.WholeWheat) = 846 := sorry

end NUMINAMATH_CALUDE_total_weekly_batches_l2117_211712


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2117_211731

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Define the smallest composite number
def smallest_composite : ℕ := 4

theorem reciprocal_problem :
  (reciprocal 0.8 = 5/4) ∧
  (reciprocal (1/4) = smallest_composite) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2117_211731


namespace NUMINAMATH_CALUDE_curve_C_properties_l2117_211778

-- Define the curve C
def C (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (4 - t) + p.2^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) : Prop :=
  t < 1 ∨ t > 4

-- Define what it means for C to be an ellipse with foci on the X-axis
def is_ellipse_x_axis (t : ℝ) : Prop :=
  1 < t ∧ t < 5/2

-- State the theorem
theorem curve_C_properties :
  ∀ t : ℝ,
    (is_hyperbola t ↔ ∃ a b : ℝ, C t = {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1}) ∧
    (is_ellipse_x_axis t ↔ ∃ a b : ℝ, C t = {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧ a > b) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_properties_l2117_211778


namespace NUMINAMATH_CALUDE_at_least_one_nonnegative_l2117_211792

theorem at_least_one_nonnegative (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n > 1) :
  let f := fun x => x^2 - x
  f m ≥ 0 ∨ f n ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_nonnegative_l2117_211792


namespace NUMINAMATH_CALUDE_rachel_furniture_time_l2117_211739

def chairs : ℕ := 7
def tables : ℕ := 3
def time_per_piece : ℕ := 4

theorem rachel_furniture_time :
  chairs * time_per_piece + tables * time_per_piece = 40 := by
  sorry

end NUMINAMATH_CALUDE_rachel_furniture_time_l2117_211739


namespace NUMINAMATH_CALUDE_two_digit_number_between_30_and_40_with_units_digit_2_l2117_211750

theorem two_digit_number_between_30_and_40_with_units_digit_2 (n : ℕ) :
  (n ≥ 30 ∧ n < 40) →  -- two-digit number between 30 and 40
  (n % 10 = 2) →       -- units digit is 2
  n = 32 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_between_30_and_40_with_units_digit_2_l2117_211750


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l2117_211709

theorem partial_fraction_decomposition_product (x : ℝ) 
  (A B C : ℝ) : 
  (x^2 + 5*x - 14) / (x^3 - 3*x^2 - x + 3) = 
    A / (x - 1) + B / (x - 3) + C / (x + 1) →
  A * B * C = -25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l2117_211709


namespace NUMINAMATH_CALUDE_painting_price_change_l2117_211730

theorem painting_price_change (P : ℝ) (x : ℝ) 
  (h1 : P > 0) 
  (h2 : (1.1 * P) * (1 - x / 100) = 0.935 * P) : 
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_change_l2117_211730


namespace NUMINAMATH_CALUDE_cosine_identity_l2117_211728

theorem cosine_identity (θ : ℝ) 
  (h : Real.cos (π / 4 - θ) = Real.sqrt 3 / 3) : 
  Real.cos (3 * π / 4 + θ) - Real.sin (θ - π / 4) ^ 2 = -(2 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l2117_211728


namespace NUMINAMATH_CALUDE_maciek_purchase_l2117_211796

/-- The cost of a pack of pretzels in dollars -/
def pretzel_cost : ℚ := 4

/-- The cost of a pack of chips in dollars -/
def chip_cost : ℚ := pretzel_cost * (1 + 3/4)

/-- The total amount Maciek spent in dollars -/
def total_spent : ℚ := 22

/-- The number of packets of each type (chips and pretzels) that Maciek bought -/
def num_packets : ℚ := total_spent / (pretzel_cost + chip_cost)

theorem maciek_purchase : num_packets = 2 := by
  sorry

end NUMINAMATH_CALUDE_maciek_purchase_l2117_211796


namespace NUMINAMATH_CALUDE_x1_value_l2117_211776

theorem x1_value (x1 x2 x3 x4 : Real) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1/3) :
  x1 = 4/5 := by sorry

end NUMINAMATH_CALUDE_x1_value_l2117_211776


namespace NUMINAMATH_CALUDE_equation_to_lines_l2117_211743

theorem equation_to_lines (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) :=
sorry

end NUMINAMATH_CALUDE_equation_to_lines_l2117_211743


namespace NUMINAMATH_CALUDE_solar_panel_distribution_l2117_211703

theorem solar_panel_distribution (total_homes : ℕ) (installed_homes : ℕ) (panel_shortage : ℕ) :
  total_homes = 20 →
  installed_homes = 15 →
  panel_shortage = 50 →
  ∃ (panels_per_home : ℕ),
    panels_per_home = 10 ∧
    panels_per_home * total_homes = panels_per_home * installed_homes + panel_shortage :=
by sorry

end NUMINAMATH_CALUDE_solar_panel_distribution_l2117_211703


namespace NUMINAMATH_CALUDE_burger_expenditure_l2117_211752

theorem burger_expenditure (total : ℚ) (movies music ice_cream : ℚ) 
  (h1 : total = 30)
  (h2 : movies = 1/3 * total)
  (h3 : music = 3/10 * total)
  (h4 : ice_cream = 1/5 * total) :
  total - (movies + music + ice_cream) = 5 := by
  sorry

end NUMINAMATH_CALUDE_burger_expenditure_l2117_211752


namespace NUMINAMATH_CALUDE_molar_mass_calculation_l2117_211749

/-- Given a chemical compound where 3 moles weigh 168 grams, prove that its molar mass is 56 grams per mole. -/
theorem molar_mass_calculation (mass : ℝ) (moles : ℝ) (h1 : mass = 168) (h2 : moles = 3) :
  mass / moles = 56 := by
  sorry

end NUMINAMATH_CALUDE_molar_mass_calculation_l2117_211749


namespace NUMINAMATH_CALUDE_negation_equivalence_l2117_211701

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 + Real.sin x < 0)) ↔ (∀ x : ℝ, x > 0 → x^2 + Real.sin x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2117_211701


namespace NUMINAMATH_CALUDE_inequality_proof_l2117_211723

theorem inequality_proof (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2117_211723


namespace NUMINAMATH_CALUDE_min_quotient_is_53_5_l2117_211715

/-- A three-digit number with distinct non-zero digits -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  a_lt_ten : a < 10
  b_lt_ten : b < 10
  c_lt_ten : c < 10
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.a + n.b + n.c

/-- The quotient of a three-digit number divided by the sum of its digits -/
def quotient (n : ThreeDigitNumber) : Rat :=
  (value n : Rat) / (digitSum n : Rat)

theorem min_quotient_is_53_5 :
  ∃ (min : Rat), ∀ (n : ThreeDigitNumber), quotient n ≥ min ∧ (∃ (m : ThreeDigitNumber), quotient m = min) ∧ min = 53.5 := by
  sorry

end NUMINAMATH_CALUDE_min_quotient_is_53_5_l2117_211715


namespace NUMINAMATH_CALUDE_system_solution_l2117_211757

theorem system_solution (x y : ℝ) : 
  (16 * x^3 + 4*x = 16*y + 5) ∧ 
  (16 * y^3 + 4*y = 16*x + 5) → 
  (x = y) ∧ (16 * x^3 - 12*x - 5 = 0) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2117_211757


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2117_211768

/-- Definition of the hyperbola with given foci and passing point -/
def Hyperbola (f : ℝ × ℝ) (p : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | 
    |((x - f.1)^2 + (y - f.2)^2).sqrt - ((x - f.1)^2 + (y + f.2)^2).sqrt| = 
    |((p.1 - f.1)^2 + (p.2 - f.2)^2).sqrt - ((p.1 - f.1)^2 + (p.2 + f.2)^2).sqrt|}

/-- The theorem stating the equation of the hyperbola -/
theorem hyperbola_equation :
  let f : ℝ × ℝ := (0, 3)
  let p : ℝ × ℝ := (Real.sqrt 15, 4)
  ∀ (x y : ℝ), (x, y) ∈ Hyperbola f p ↔ y^2 / 4 - x^2 / 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2117_211768


namespace NUMINAMATH_CALUDE_square_sum_eq_18_l2117_211708

theorem square_sum_eq_18 (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x^2 + y^2 = 18) : 
  x^2 + y^2 = 18 := by
sorry

end NUMINAMATH_CALUDE_square_sum_eq_18_l2117_211708


namespace NUMINAMATH_CALUDE_lunchroom_students_l2117_211753

theorem lunchroom_students (tables : ℕ) (seated_per_table : ℕ) (standing : ℕ) : 
  tables = 34 → seated_per_table = 6 → standing = 15 →
  tables * seated_per_table + standing = 219 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l2117_211753


namespace NUMINAMATH_CALUDE_lotto_game_minimum_draws_l2117_211710

theorem lotto_game_minimum_draws (n : ℕ) (h : n = 90) : 
  ∃ k : ℕ, k = 49 ∧ 
  (∀ S : Finset ℕ, S.card = k → S ⊆ Finset.range n → 
    ∃ x ∈ S, x % 3 = 0 ∨ x % 5 = 0) ∧
  (∀ m : ℕ, m < k → 
    ∃ T : Finset ℕ, T.card = m ∧ T ⊆ Finset.range n ∧ 
    ∀ x ∈ T, x % 3 ≠ 0 ∧ x % 5 ≠ 0) :=
by sorry


end NUMINAMATH_CALUDE_lotto_game_minimum_draws_l2117_211710


namespace NUMINAMATH_CALUDE_josie_shopping_shortfall_l2117_211760

def gift_amount : ℕ := 80
def cassette_price : ℕ := 15
def num_cassettes : ℕ := 3
def headphone_price : ℕ := 40
def vinyl_price : ℕ := 12

theorem josie_shopping_shortfall :
  gift_amount < cassette_price * num_cassettes + headphone_price + vinyl_price ∧
  cassette_price * num_cassettes + headphone_price + vinyl_price - gift_amount = 17 :=
by sorry

end NUMINAMATH_CALUDE_josie_shopping_shortfall_l2117_211760


namespace NUMINAMATH_CALUDE_length_width_relation_l2117_211763

/-- A rectangle enclosed by a wire -/
structure WireRectangle where
  wireLength : ℝ
  width : ℝ
  length : ℝ
  wireLength_positive : 0 < wireLength
  width_positive : 0 < width
  length_positive : 0 < length
  perimeter_eq_wireLength : 2 * (width + length) = wireLength

/-- The relationship between length and width for a 20-meter wire rectangle -/
theorem length_width_relation (rect : WireRectangle) 
    (h : rect.wireLength = 20) : 
    rect.length = -rect.width + 10 := by
  sorry

end NUMINAMATH_CALUDE_length_width_relation_l2117_211763


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_for_two_l2117_211735

theorem arithmetic_geometric_mean_inequality_for_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt x + Real.sqrt y) / 2 ≤ Real.sqrt ((x + y) / 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_for_two_l2117_211735


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2117_211716

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q)
    (h_sum1 : a 3 + a 6 = 6) (h_sum2 : a 5 + a 8 = 9) :
  a 7 + a 10 = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2117_211716


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2117_211759

/-- Pick's Theorem for quadrilaterals -/
def area_by_picks_theorem (interior_points : ℕ) (boundary_points : ℕ) : ℚ :=
  interior_points + boundary_points / 2 - 1

/-- The quadrilateral in the problem -/
structure Quadrilateral where
  interior_points : ℕ
  boundary_points : ℕ

/-- The specific quadrilateral from the problem -/
def problem_quadrilateral : Quadrilateral where
  interior_points := 12
  boundary_points := 6

theorem quadrilateral_area :
  area_by_picks_theorem problem_quadrilateral.interior_points problem_quadrilateral.boundary_points = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2117_211759


namespace NUMINAMATH_CALUDE_milk_problem_l2117_211746

theorem milk_problem (initial_milk : ℚ) (given_milk : ℚ) (result : ℚ) : 
  initial_milk = 4 →
  given_milk = 16/3 →
  result = initial_milk - given_milk →
  result = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_milk_problem_l2117_211746


namespace NUMINAMATH_CALUDE_serenas_mother_age_l2117_211702

/-- Serena's current age -/
def serena_age : ℕ := 9

/-- Years into the future when the age comparison is made -/
def years_future : ℕ := 6

/-- Serena's mother's age now -/
def mother_age : ℕ := 39

/-- Theorem stating that Serena's mother's current age is 39 -/
theorem serenas_mother_age : 
  (mother_age + years_future) = 3 * (serena_age + years_future) → 
  mother_age = 39 := by
  sorry

end NUMINAMATH_CALUDE_serenas_mother_age_l2117_211702


namespace NUMINAMATH_CALUDE_floor_tiles_1517_902_l2117_211729

/-- The least number of square tiles required to pave a rectangular floor -/
def leastSquareTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  ((length + tileSize - 1) / tileSize) * ((width + tileSize - 1) / tileSize)

/-- Proof that 814 square tiles are required for a 1517 cm x 902 cm floor -/
theorem floor_tiles_1517_902 :
  leastSquareTiles 1517 902 = 814 := by
  sorry

end NUMINAMATH_CALUDE_floor_tiles_1517_902_l2117_211729


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2117_211775

theorem smallest_number_with_remainders : ∃! a : ℕ+, 
  (a : ℤ) % 4 = 1 ∧ 
  (a : ℤ) % 3 = 1 ∧ 
  (a : ℤ) % 5 = 2 ∧ 
  (∀ n : ℕ+, n < a → ((n : ℤ) % 4 ≠ 1 ∨ (n : ℤ) % 3 ≠ 1 ∨ (n : ℤ) % 5 ≠ 2)) ∧
  a = 37 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2117_211775


namespace NUMINAMATH_CALUDE_expression_equality_l2117_211704

theorem expression_equality : (50 + 20 / 90) * 90 = 4520 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2117_211704


namespace NUMINAMATH_CALUDE_calculation_proof_l2117_211761

theorem calculation_proof : (-1) * (-4) + 2^2 / (7 - 5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2117_211761
