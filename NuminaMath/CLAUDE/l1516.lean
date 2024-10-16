import Mathlib

namespace NUMINAMATH_CALUDE_average_speed_calculation_l1516_151610

theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := first_half_time * second_half_time_factor
  let total_time := first_half_time + second_half_time
  total_distance / total_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1516_151610


namespace NUMINAMATH_CALUDE_min_triangle_area_in_cube_l1516_151675

/-- Given a cube with edge length a, the minimum area of triangles formed by 
    intersections of a plane parallel to the base with specific lines is 7a²/32 -/
theorem min_triangle_area_in_cube (a : ℝ) (ha : a > 0) : 
  ∃ (S : ℝ), S = (7 * a^2) / 32 ∧ 
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ a → 
    S ≤ (1/4) * |2*x^2 - 3*a*x + 2*a^2| := by
  sorry

end NUMINAMATH_CALUDE_min_triangle_area_in_cube_l1516_151675


namespace NUMINAMATH_CALUDE_carnival_ride_wait_time_l1516_151699

/-- Proves that the wait time for the giant slide is 15 minutes given the carnival ride conditions. -/
theorem carnival_ride_wait_time 
  (total_time : ℕ) 
  (roller_coaster_wait : ℕ) 
  (tilt_a_whirl_wait : ℕ) 
  (roller_coaster_rides : ℕ) 
  (tilt_a_whirl_rides : ℕ) 
  (giant_slide_rides : ℕ) 
  (h1 : total_time = 4 * 60)  -- 4 hours in minutes
  (h2 : roller_coaster_wait = 30)
  (h3 : tilt_a_whirl_wait = 60)
  (h4 : roller_coaster_rides = 4)
  (h5 : tilt_a_whirl_rides = 1)
  (h6 : giant_slide_rides = 4)
  : ∃ (giant_slide_wait : ℕ), 
    giant_slide_wait * giant_slide_rides = 
      total_time - (roller_coaster_wait * roller_coaster_rides + tilt_a_whirl_wait * tilt_a_whirl_rides) ∧
    giant_slide_wait = 15 :=
by sorry

end NUMINAMATH_CALUDE_carnival_ride_wait_time_l1516_151699


namespace NUMINAMATH_CALUDE_range_of_a_l1516_151654

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → a > 2 * x - 1) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1516_151654


namespace NUMINAMATH_CALUDE_vector_equality_implies_m_value_l1516_151604

theorem vector_equality_implies_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![m, 2]
  let b : Fin 2 → ℝ := ![2, -3]
  (‖a + b‖ = ‖a - b‖) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_implies_m_value_l1516_151604


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1516_151609

/-- Given a geometric sequence {a_n}, if a_2 and a_6 are roots of x^2 - 34x + 81 = 0, then a_4 = 9 -/
theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1))
  (h_roots : a 2 * a 6 = 81 ∧ a 2 + a 6 = 34) :
  a 4 = 9 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1516_151609


namespace NUMINAMATH_CALUDE_john_notebook_duration_l1516_151671

/-- The number of days notebooks last given specific conditions -/
def notebook_duration (
  num_notebooks : ℕ
  ) (pages_per_notebook : ℕ
  ) (pages_per_weekday : ℕ
  ) (pages_per_weekend_day : ℕ
  ) : ℕ :=
  let total_pages := num_notebooks * pages_per_notebook
  let pages_per_week := 5 * pages_per_weekday + 2 * pages_per_weekend_day
  let full_weeks := total_pages / pages_per_week
  let remaining_pages := total_pages % pages_per_week
  let full_days := full_weeks * 7
  let extra_days := 
    if remaining_pages ≤ 5 * pages_per_weekday
    then remaining_pages / pages_per_weekday
    else 5 + (remaining_pages - 5 * pages_per_weekday + pages_per_weekend_day - 1) / pages_per_weekend_day
  full_days + extra_days

theorem john_notebook_duration :
  notebook_duration 5 40 4 6 = 43 := by
  sorry

end NUMINAMATH_CALUDE_john_notebook_duration_l1516_151671


namespace NUMINAMATH_CALUDE_hole_movable_to_any_corner_l1516_151686

/-- Represents a rectangular box with dominoes -/
structure DominoBox where
  m : Nat
  n : Nat
  isOddM : Odd m
  isOddN : Odd n

/-- Represents a position in the box -/
structure Position where
  row : Nat
  col : Nat

/-- Checks if a position is a corner of the box -/
def isCorner (box : DominoBox) (pos : Position) : Prop :=
  (pos.row = 1 ∨ pos.row = box.m) ∧ (pos.col = 1 ∨ pos.col = box.n)

/-- Represents the state of the box with the hole position -/
structure BoxState where
  box : DominoBox
  holePos : Position

/-- Represents a single move of sliding a domino -/
inductive Move where
  | slideUp : BoxState → Move
  | slideDown : BoxState → Move
  | slideLeft : BoxState → Move
  | slideRight : BoxState → Move

/-- Represents a sequence of moves -/
def MoveSequence := List Move

/-- Checks if a move sequence is valid and moves the hole to the target position -/
def isValidMoveSequence (start : BoxState) (target : Position) (moves : MoveSequence) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem hole_movable_to_any_corner (box : DominoBox) (start : Position) (target : Position)
    (h_start_corner : isCorner box start) (h_target_corner : isCorner box target) :
    ∃ (moves : MoveSequence), isValidMoveSequence ⟨box, start⟩ target moves :=
  sorry

end NUMINAMATH_CALUDE_hole_movable_to_any_corner_l1516_151686


namespace NUMINAMATH_CALUDE_barbaras_coin_collection_l1516_151685

/-- The total number of coins Barbara has -/
def total_coins : ℕ := 18

/-- The number of type A coins Barbara has -/
def type_A_coins : ℕ := 12

/-- The value of 8 type A coins in dollars -/
def value_8_type_A : ℕ := 24

/-- The value of 6 type B coins in dollars -/
def value_6_type_B : ℕ := 21

/-- The total worth of Barbara's entire collection in dollars -/
def total_worth : ℕ := 57

theorem barbaras_coin_collection :
  total_coins = type_A_coins + (total_coins - type_A_coins) ∧
  value_8_type_A / 8 * type_A_coins + value_6_type_B / 6 * (total_coins - type_A_coins) = total_worth :=
by sorry

end NUMINAMATH_CALUDE_barbaras_coin_collection_l1516_151685


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l1516_151614

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem tangent_point_coordinates (a : ℝ) :
  (∀ x, f_deriv a (-x) = -f_deriv a x) →
  ∃ x₀ y₀, f a x₀ = y₀ ∧ f_deriv a x₀ = 3/2 →
  x₀ = Real.log 2 ∧ y₀ = 5/2 := by sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l1516_151614


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1516_151660

-- Define the universe as the set of real numbers
def U : Set ℝ := Set.univ

-- Define sets A and B as open intervals
def A : Set ℝ := Set.Ioo 1 3
def B : Set ℝ := Set.Ioo 2 4

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = Set.Ioo 1 4 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1516_151660


namespace NUMINAMATH_CALUDE_expression_value_l1516_151676

theorem expression_value : 65 + (120 / 15) + (15 * 18) - 250 - (405 / 9) + 3^3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1516_151676


namespace NUMINAMATH_CALUDE_vacation_miles_driven_l1516_151658

theorem vacation_miles_driven (vacation_days : Float) (miles_per_day : Float) : 
  vacation_days = 5.0 → miles_per_day = 250 → vacation_days * miles_per_day = 1250 := by
sorry

end NUMINAMATH_CALUDE_vacation_miles_driven_l1516_151658


namespace NUMINAMATH_CALUDE_second_project_depth_l1516_151639

/-- Represents a digging project with its dimensions and duration -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ
  days : ℝ

/-- Calculates the volume of a digging project -/
def volume (p : DiggingProject) : ℝ := p.depth * p.length * p.breadth

/-- The first digging project -/
def project1 : DiggingProject := {
  depth := 100,
  length := 25,
  breadth := 30,
  days := 12
}

/-- The second digging project with unknown depth -/
def project2 (depth : ℝ) : DiggingProject := {
  depth := depth,
  length := 20,
  breadth := 50,
  days := 12
}

/-- Theorem stating that the depth of the second project is 75 meters -/
theorem second_project_depth : 
  ∃ (depth : ℝ), volume project1 = volume (project2 depth) ∧ depth = 75 := by
  sorry


end NUMINAMATH_CALUDE_second_project_depth_l1516_151639


namespace NUMINAMATH_CALUDE_box_has_four_balls_l1516_151670

/-- A color of a ball -/
inductive Color
| Red
| Blue
| Other

/-- A box containing balls of different colors -/
structure Box where
  balls : List Color

/-- Checks if a list of colors contains at least one red and one blue -/
def hasRedAndBlue (colors : List Color) : Prop :=
  Color.Red ∈ colors ∧ Color.Blue ∈ colors

/-- The main theorem stating that the box must contain exactly 4 balls -/
theorem box_has_four_balls (box : Box) : 
  (∀ (a b c : Color), a ∈ box.balls → b ∈ box.balls → c ∈ box.balls → 
    a ≠ b → b ≠ c → a ≠ c → hasRedAndBlue [a, b, c]) →
  (3 < box.balls.length) →
  box.balls.length = 4 := by
  sorry


end NUMINAMATH_CALUDE_box_has_four_balls_l1516_151670


namespace NUMINAMATH_CALUDE_arctan_sum_problem_l1516_151615

theorem arctan_sum_problem (a b : ℝ) : 
  a = 1/3 → 
  (a + 2) * (b + 2) = 15 → 
  Real.arctan a + Real.arctan b = 5 * π / 6 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_problem_l1516_151615


namespace NUMINAMATH_CALUDE_min_value_expression_l1516_151608

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1516_151608


namespace NUMINAMATH_CALUDE_four_parts_of_400_l1516_151633

theorem four_parts_of_400 (a b c d : ℝ) 
  (sum_eq_400 : a + b + c + d = 400)
  (parts_equal : a + 1 = b - 2 ∧ b - 2 = 3 * c ∧ 3 * c = d / 4)
  (positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  a = 62 ∧ b = 65 ∧ c = 21 ∧ d = 252 := by
sorry

end NUMINAMATH_CALUDE_four_parts_of_400_l1516_151633


namespace NUMINAMATH_CALUDE_exam_average_l1516_151645

theorem exam_average (n1 n2 : ℕ) (avg1 avg2 : ℚ) : 
  n1 = 15 → 
  n2 = 10 → 
  avg1 = 75 / 100 → 
  avg2 = 90 / 100 → 
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 81 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l1516_151645


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l1516_151613

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 ∧ 4 ∣ x ∧ x^3 < 1728 → x ≤ 8 ∧ ∃ y : ℕ, y > 0 ∧ 4 ∣ y ∧ y^3 < 1728 ∧ y = 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l1516_151613


namespace NUMINAMATH_CALUDE_cake_mix_buyers_l1516_151620

theorem cake_mix_buyers (total : ℕ) (muffin : ℕ) (both : ℕ) (neither_prob : ℚ) :
  total = 100 →
  muffin = 40 →
  both = 18 →
  neither_prob = 28/100 →
  ∃ cake : ℕ, cake = 50 ∧ cake + muffin - both = (1 - neither_prob) * total := by
  sorry

end NUMINAMATH_CALUDE_cake_mix_buyers_l1516_151620


namespace NUMINAMATH_CALUDE_negation_equivalence_l1516_151694

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Teenager : U → Prop)
variable (Responsible : U → Prop)

-- Theorem statement
theorem negation_equivalence :
  (∃ x, Teenager x ∧ ¬Responsible x) ↔ ¬(∀ x, Teenager x → Responsible x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1516_151694


namespace NUMINAMATH_CALUDE_percentage_difference_l1516_151697

theorem percentage_difference : (0.8 * 40) - ((4 / 5) * 15) = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1516_151697


namespace NUMINAMATH_CALUDE_family_celebration_attendees_l1516_151667

theorem family_celebration_attendees :
  ∀ (n : ℕ) (s : ℕ),
    s / n = n →
    (s - 29) / (n - 1) = n - 1 →
    n = 15 := by
  sorry

end NUMINAMATH_CALUDE_family_celebration_attendees_l1516_151667


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l1516_151621

theorem percentage_of_sikh_boys (total_boys : ℕ) (muslim_percentage : ℚ) (hindu_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  muslim_percentage = 44 / 100 →
  hindu_percentage = 14 / 100 →
  other_boys = 272 →
  (total_boys - (muslim_percentage * total_boys + hindu_percentage * total_boys + other_boys)) / total_boys = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l1516_151621


namespace NUMINAMATH_CALUDE_prob_three_students_same_group_l1516_151634

/-- The total number of students -/
def total_students : ℕ := 800

/-- The number of lunch groups -/
def num_groups : ℕ := 4

/-- The size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- The probability of a student being assigned to a specific group -/
def prob_one_group : ℚ := 1 / num_groups

/-- The probability that three specific students are assigned to the same lunch group -/
theorem prob_three_students_same_group :
  (prob_one_group * prob_one_group : ℚ) = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_prob_three_students_same_group_l1516_151634


namespace NUMINAMATH_CALUDE_routes_between_plains_cities_l1516_151677

theorem routes_between_plains_cities
  (total_cities : Nat)
  (mountainous_cities : Nat)
  (plains_cities : Nat)
  (total_routes : Nat)
  (mountainous_routes : Nat)
  (h1 : total_cities = 100)
  (h2 : mountainous_cities = 30)
  (h3 : plains_cities = 70)
  (h4 : mountainous_cities + plains_cities = total_cities)
  (h5 : total_routes = 150)
  (h6 : mountainous_routes = 21) :
  ∃ (plains_routes : Nat),
    plains_routes = 81 ∧
    plains_routes + mountainous_routes + (total_routes - plains_routes - mountainous_routes) = total_routes :=
by sorry

end NUMINAMATH_CALUDE_routes_between_plains_cities_l1516_151677


namespace NUMINAMATH_CALUDE_equation_equivalence_l1516_151692

theorem equation_equivalence (x : ℝ) : x^2 + 4*x + 2 = 0 ↔ (x + 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1516_151692


namespace NUMINAMATH_CALUDE_optimal_order_l1516_151622

variable (p1 p2 p3 : ℝ)

-- Probabilities are between 0 and 1
axiom prob_range1 : 0 ≤ p1 ∧ p1 ≤ 1
axiom prob_range2 : 0 ≤ p2 ∧ p2 ≤ 1
axiom prob_range3 : 0 ≤ p3 ∧ p3 ≤ 1

-- Ordering of probabilities
axiom prob_order : p3 < p1 ∧ p1 < p2

-- Function to calculate probability of winning two games in a row
def win_probability (p_first p_second p_third : ℝ) : ℝ :=
  p_first * p_second + (1 - p_first) * p_second * p_third

-- Theorem stating that playing against p2 (highest probability) second is optimal
theorem optimal_order :
  win_probability p1 p2 p3 > win_probability p2 p1 p3 ∧
  win_probability p3 p2 p1 > win_probability p2 p3 p1 :=
sorry

end NUMINAMATH_CALUDE_optimal_order_l1516_151622


namespace NUMINAMATH_CALUDE_cylinder_properties_l1516_151681

/-- Properties of a cylinder with height 15 and radius 5 -/
theorem cylinder_properties :
  let h : ℝ := 15
  let r : ℝ := 5
  let total_surface_area : ℝ := 2 * Real.pi * r * r + 2 * Real.pi * r * h
  let volume : ℝ := Real.pi * r * r * h
  (total_surface_area = 200 * Real.pi) ∧ (volume = 375 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_properties_l1516_151681


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equation_l1516_151662

/-- Given a curve C with polar coordinate equation ρ sin (θ - π/4) = √2,
    where the origin is at the pole and the polar axis lies on the x-axis
    in a rectangular coordinate system, prove that the rectangular
    coordinate equation of C is x - y + 2 = 0. -/
theorem polar_to_rectangular_equation :
  ∀ (ρ θ x y : ℝ),
  (ρ * Real.sin (θ - π/4) = Real.sqrt 2) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x - y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equation_l1516_151662


namespace NUMINAMATH_CALUDE_abs_neg_six_l1516_151619

theorem abs_neg_six : |(-6 : ℤ)| = 6 := by sorry

end NUMINAMATH_CALUDE_abs_neg_six_l1516_151619


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l1516_151666

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_sec : ℝ) : ℝ :=
  speed_km_per_sec * (60 * 60)

theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 2 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l1516_151666


namespace NUMINAMATH_CALUDE_circle_line_distances_l1516_151618

/-- The maximum and minimum distances from a point on the circle x^2 + y^2 = 1 to the line x - 2y - 12 = 0 -/
theorem circle_line_distances :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | x - 2*y - 12 = 0}
  ∃ (max_dist min_dist : ℝ),
    (∀ p ∈ circle, ∀ q ∈ line, dist p q ≤ max_dist) ∧
    (∃ p ∈ circle, ∃ q ∈ line, dist p q = max_dist) ∧
    (∀ p ∈ circle, ∀ q ∈ line, dist p q ≥ min_dist) ∧
    (∃ p ∈ circle, ∃ q ∈ line, dist p q = min_dist) ∧
    max_dist = (12 * Real.sqrt 5) / 5 + 1 ∧
    min_dist = (12 * Real.sqrt 5) / 5 - 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distances_l1516_151618


namespace NUMINAMATH_CALUDE_at_least_two_first_class_products_l1516_151635

def total_products : ℕ := 9
def first_class : ℕ := 4
def second_class : ℕ := 3
def third_class : ℕ := 2
def products_to_draw : ℕ := 4

theorem at_least_two_first_class_products :
  (Nat.choose first_class 2 * Nat.choose (total_products - first_class) 2 +
   Nat.choose first_class 3 * Nat.choose (total_products - first_class) 1 +
   Nat.choose first_class 4 * Nat.choose (total_products - first_class) 0) =
  (Nat.choose total_products products_to_draw -
   Nat.choose (second_class + third_class) products_to_draw -
   (Nat.choose first_class 1 * Nat.choose (second_class + third_class) 3)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_first_class_products_l1516_151635


namespace NUMINAMATH_CALUDE_pen_count_l1516_151695

theorem pen_count (num_pencils : ℕ) (max_students : ℕ) (h1 : num_pencils = 910) (h2 : max_students = 91) 
  (h3 : max_students ∣ num_pencils) : 
  ∃ num_pens : ℕ, num_pens = num_pencils :=
by sorry

end NUMINAMATH_CALUDE_pen_count_l1516_151695


namespace NUMINAMATH_CALUDE_complex_difference_magnitude_l1516_151661

def i : ℂ := Complex.I

theorem complex_difference_magnitude : Complex.abs ((1 + i)^13 - (1 - i)^13) = 128 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_magnitude_l1516_151661


namespace NUMINAMATH_CALUDE_outfit_combinations_l1516_151648

def num_shirts : ℕ := 5
def num_pants : ℕ := 6
def num_hats : ℕ := 2

theorem outfit_combinations : num_shirts * num_pants * num_hats = 60 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1516_151648


namespace NUMINAMATH_CALUDE_calculation_proof_l1516_151659

theorem calculation_proof : 
  (3.2 - 2.95) / (0.25 * 2 + 1/4) + (2 * 0.3) / (2.3 - (1 + 2/5)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1516_151659


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_35_l1516_151684

def sum_of_consecutive_integers (start : ℕ) (count : ℕ) : ℕ :=
  count * (2 * start + count - 1) / 2

theorem largest_consecutive_sum_35 :
  (∃ (start : ℕ), sum_of_consecutive_integers start 7 = 35) ∧
  (∀ (start count : ℕ), count > 7 → sum_of_consecutive_integers start count ≠ 35) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_35_l1516_151684


namespace NUMINAMATH_CALUDE_square_area_problem_l1516_151688

theorem square_area_problem : 
  ∀ x : ℚ, 
  (5 * x - 20 : ℚ) = (25 - 2 * x : ℚ) → 
  (5 * x - 20 : ℚ) > 0 →
  (5 * x - 20 : ℚ)^2 = 7225 / 49 := by
sorry

end NUMINAMATH_CALUDE_square_area_problem_l1516_151688


namespace NUMINAMATH_CALUDE_letter_distribution_l1516_151647

/-- Represents the number of letters in each pocket -/
def pocket1_letters : ℕ := 5
def pocket2_letters : ℕ := 4

/-- Represents the number of mailboxes -/
def num_mailboxes : ℕ := 4

/-- The number of ways to take one letter from two pockets -/
def ways_to_take_one_letter : ℕ := pocket1_letters + pocket2_letters

/-- The number of ways to take one letter from each pocket -/
def ways_to_take_one_from_each : ℕ := pocket1_letters * pocket2_letters

/-- The number of ways to put all letters into mailboxes -/
def ways_to_put_in_mailboxes : ℕ := num_mailboxes ^ (pocket1_letters + pocket2_letters)

theorem letter_distribution :
  (ways_to_take_one_letter = 9) ∧
  (ways_to_take_one_from_each = 20) ∧
  (ways_to_put_in_mailboxes = 262144) := by
  sorry

end NUMINAMATH_CALUDE_letter_distribution_l1516_151647


namespace NUMINAMATH_CALUDE_square_of_binomial_l1516_151643

theorem square_of_binomial (b : ℚ) : 
  (∃ (c : ℚ), ∀ (x : ℚ), 9*x^2 + 21*x + b = (3*x + c)^2) → b = 49/4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l1516_151643


namespace NUMINAMATH_CALUDE_unique_digit_sum_count_l1516_151607

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Count of numbers in [1, 1000] with a given digit sum -/
def count_numbers_with_digit_sum (sum : ℕ) : ℕ := sorry

theorem unique_digit_sum_count :
  ∃! n : ℕ, n ∈ Finset.range 28 ∧ count_numbers_with_digit_sum n = 10 :=
sorry

end NUMINAMATH_CALUDE_unique_digit_sum_count_l1516_151607


namespace NUMINAMATH_CALUDE_money_distribution_l1516_151668

theorem money_distribution (total : ℝ) (a b c d : ℝ) :
  a + b + c + d = total →
  a = (5 / 14) * total →
  b = (2 / 14) * total →
  c = (4 / 14) * total →
  d = (3 / 14) * total →
  c = d + 500 →
  d = 1500 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1516_151668


namespace NUMINAMATH_CALUDE_students_without_glasses_l1516_151687

theorem students_without_glasses (total_students : ℕ) (percent_with_glasses : ℚ) 
  (h1 : total_students = 325)
  (h2 : percent_with_glasses = 40/100) : 
  ↑total_students * (1 - percent_with_glasses) = 195 := by
  sorry

end NUMINAMATH_CALUDE_students_without_glasses_l1516_151687


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1516_151652

theorem trigonometric_problem (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.sin α = 4 / 5)
  (h4 : Real.cos (α + β) = 5 / 13) : 
  (Real.cos β = 63 / 65) ∧ 
  ((Real.sin α)^2 + Real.sin (2 * α)) / (Real.cos (2 * α) - 1) = -5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1516_151652


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l1516_151644

theorem x_fourth_plus_inverse_fourth (x : ℝ) (h : x^2 + 1/x^2 = 6) : x^4 + 1/x^4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l1516_151644


namespace NUMINAMATH_CALUDE_complex_magnitude_l1516_151690

theorem complex_magnitude (z : ℂ) (h : z + (Real.exp 1) / z + Real.pi = 0) :
  Complex.abs z = Real.sqrt (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1516_151690


namespace NUMINAMATH_CALUDE_family_income_problem_l1516_151632

/-- Proves that in a family of 4 members with an average income of 10000,
    if three members earn 8000, 6000, and 11000 respectively,
    then the income of the fourth member is 15000. -/
theorem family_income_problem (family_size : ℕ) (average_income : ℕ) 
  (member1_income : ℕ) (member2_income : ℕ) (member3_income : ℕ) :
  family_size = 4 →
  average_income = 10000 →
  member1_income = 8000 →
  member2_income = 6000 →
  member3_income = 11000 →
  average_income * family_size - (member1_income + member2_income + member3_income) = 15000 :=
by sorry

end NUMINAMATH_CALUDE_family_income_problem_l1516_151632


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l1516_151640

/-- Proves that given a distance that takes 2 hours 45 minutes to walk and 40 minutes to run at 16.5 kmph, the walking speed is 4 kmph. -/
theorem walking_speed_calculation (distance : ℝ) : 
  distance / (2 + 45 / 60) = 4 → distance / (40 / 60) = 16.5 → distance / (2 + 45 / 60) = 4 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l1516_151640


namespace NUMINAMATH_CALUDE_magazine_fraction_l1516_151627

theorem magazine_fraction (initial_amount : ℚ) (grocery_fraction : ℚ) (remaining_amount : ℚ) :
  initial_amount = 600 →
  grocery_fraction = 1/5 →
  remaining_amount = 360 →
  let amount_after_groceries := initial_amount - grocery_fraction * initial_amount
  (amount_after_groceries - remaining_amount) / amount_after_groceries = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_magazine_fraction_l1516_151627


namespace NUMINAMATH_CALUDE_twelfth_root_of_unity_l1516_151616

theorem twelfth_root_of_unity : 
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 11 ∧ 
  (Complex.tan (π / 6) + Complex.I) / (Complex.tan (π / 6) - Complex.I) = 
  Complex.exp (Complex.I * (2 * n * π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_twelfth_root_of_unity_l1516_151616


namespace NUMINAMATH_CALUDE_fraction_modification_l1516_151691

theorem fraction_modification (a b c d x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a^2 + x) / (b^2 + x) = c / d) 
  (h4 : c ≠ d) : 
  x = (a^2 * d - b^2 * c) / (c - d) := by
sorry

end NUMINAMATH_CALUDE_fraction_modification_l1516_151691


namespace NUMINAMATH_CALUDE_common_solution_y_values_l1516_151663

theorem common_solution_y_values : 
  ∃ y₁ y₂ : ℝ, 
    (∀ x y : ℝ, x^2 + y^2 - 9 = 0 ∧ x^2 - 4*y + 8 = 0 → y = y₁ ∨ y = y₂) ∧
    y₁ = -2 + Real.sqrt 21 ∧
    y₂ = -2 - Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_common_solution_y_values_l1516_151663


namespace NUMINAMATH_CALUDE_rachel_age_when_emily_half_is_eight_l1516_151628

/-- Rachel's age when Emily's age is half of Rachel's --/
def rachels_age_when_emily_half (emily_current_age rachel_current_age : ℕ) : ℕ :=
  let age_difference := rachel_current_age - emily_current_age
  let emily_half_age := age_difference
  emily_half_age + age_difference

theorem rachel_age_when_emily_half_is_eight :
  rachels_age_when_emily_half 20 24 = 8 := by sorry

end NUMINAMATH_CALUDE_rachel_age_when_emily_half_is_eight_l1516_151628


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l1516_151682

/-- The set of points P satisfying |PF₁| + |PF₂| = 10, where F₁ and F₂ are fixed points, forms a line segment. -/
theorem trajectory_is_line_segment (F₁ F₂ : ℝ × ℝ) (h₁ : F₁ = (-5, 0)) (h₂ : F₂ = (5, 0)) :
  {P : ℝ × ℝ | dist P F₁ + dist P F₂ = 10} = {P : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂} :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l1516_151682


namespace NUMINAMATH_CALUDE_original_people_count_l1516_151693

theorem original_people_count (initial_count : ℕ) : 
  (initial_count / 3 : ℚ) = 18 →
  initial_count = 54 := by
  sorry

end NUMINAMATH_CALUDE_original_people_count_l1516_151693


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1516_151617

theorem trigonometric_inequality (x : ℝ) :
  (-1/4 : ℝ) ≤ 5 * (Real.cos x)^2 - 5 * (Real.cos x)^4 + 5 * Real.sin x * Real.cos x + 1 ∧
  5 * (Real.cos x)^2 - 5 * (Real.cos x)^4 + 5 * Real.sin x * Real.cos x + 1 ≤ (19/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1516_151617


namespace NUMINAMATH_CALUDE_ellipse_constants_correct_l1516_151651

def ellipse_constants (f₁ f₂ p : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

theorem ellipse_constants_correct :
  let f₁ : ℝ × ℝ := (3, 3)
  let f₂ : ℝ × ℝ := (3, 9)
  let p : ℝ × ℝ := (16, -2)
  let (a, b, h, k) := ellipse_constants f₁ f₂ p
  (h = 3 ∧
   k = 6 ∧
   a = (Real.sqrt 194 + Real.sqrt 290) / 2 ∧
   b = Real.sqrt ((Real.sqrt 194 + Real.sqrt 290)^2 / 4 - 9)) ∧
  (a > 0 ∧ b > 0) ∧
  ∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ 
    Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) + Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2) = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_ellipse_constants_correct_l1516_151651


namespace NUMINAMATH_CALUDE_prob_odd_sum_is_correct_l1516_151696

/-- Represents the dartboard with given radii and scoring regions -/
structure Dartboard where
  inner_radius : ℝ
  intermediate_radius : ℝ
  outer_radius : ℝ
  inner_scores : Fin 3 → ℕ
  intermediate_scores : Fin 3 → ℕ
  outer_scores : Fin 3 → ℕ

/-- Calculates the probability of getting an odd sum when throwing two darts -/
def prob_odd_sum (d : Dartboard) : ℚ :=
  265 / 855

/-- The specific dartboard described in the problem -/
def problem_dartboard : Dartboard where
  inner_radius := 4.5
  intermediate_radius := 6.75
  outer_radius := 9
  inner_scores := ![3, 2, 2]
  intermediate_scores := ![2, 1, 1]
  outer_scores := ![1, 1, 3]

theorem prob_odd_sum_is_correct :
  prob_odd_sum problem_dartboard = 265 / 855 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_sum_is_correct_l1516_151696


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1516_151601

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1516_151601


namespace NUMINAMATH_CALUDE_function_nature_l1516_151612

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem function_nature (f : ℝ → ℝ) 
  (h1 : f 0 ≠ 0)
  (h2 : ∀ x₁ x₂ : ℝ, f x₁ + f x₂ = 2 * f ((x₁ + x₂) / 2) * f ((x₁ - x₂) / 2)) :
  is_even f ∧ ¬ is_odd f := by
sorry

end NUMINAMATH_CALUDE_function_nature_l1516_151612


namespace NUMINAMATH_CALUDE_triangle_solutions_l1516_151653

def is_valid_triangle (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ a + b > c

def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

def has_100 (a c : ℕ) : Prop :=
  a = 100 ∨ c = 100

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(49,70,100), (64,80,100), (81,90,100), (100,100,100), (100,110,121),
   (100,120,144), (100,130,169), (100,140,196), (100,150,225), (100,160,256)}

theorem triangle_solutions :
  ∀ a b c : ℕ,
    is_valid_triangle a b c ∧
    is_geometric_sequence a b c ∧
    has_100 a c ↔
    (a, b, c) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_triangle_solutions_l1516_151653


namespace NUMINAMATH_CALUDE_unique_number_doubled_plus_thirteen_l1516_151664

theorem unique_number_doubled_plus_thirteen : ∃! x : ℝ, 2 * x + 13 = 89 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_doubled_plus_thirteen_l1516_151664


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1516_151656

theorem rectangular_box_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 18) 
  (h3 : c * a = 10) : 
  a * b * c = 30 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1516_151656


namespace NUMINAMATH_CALUDE_max_value_expression_tightness_of_bound_l1516_151626

theorem max_value_expression (x y : ℝ) :
  (x + 3*y + 2) / Real.sqrt (2*x^2 + y^2 + 1) ≤ Real.sqrt 14 :=
sorry

theorem tightness_of_bound : 
  ∀ ε > 0, ∃ x y : ℝ, Real.sqrt 14 - (x + 3*y + 2) / Real.sqrt (2*x^2 + y^2 + 1) < ε :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_tightness_of_bound_l1516_151626


namespace NUMINAMATH_CALUDE_right_triangle_medians_count_l1516_151674

/-- A right triangle with legs parallel to the coordinate axes -/
structure RightTriangle where
  /-- The slope of one median -/
  slope1 : ℝ
  /-- The slope of the other median -/
  slope2 : ℝ
  /-- One median lies on the line y = 5x + 1 -/
  median1_eq : slope1 = 5
  /-- The other median lies on the line y = mx + 2 -/
  median2_eq : slope2 = m
  /-- The slopes satisfy the right triangle condition -/
  slope_condition : slope1 = 4 * slope2 ∨ slope2 = 4 * slope1

/-- The theorem stating that there are exactly two values of m for which a right triangle
    with the given conditions can be constructed -/
theorem right_triangle_medians_count :
  ∃ (m1 m2 : ℝ), m1 ≠ m2 ∧
  (∀ m : ℝ, (∃ t : RightTriangle, t.slope2 = m) ↔ (m = m1 ∨ m = m2)) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_medians_count_l1516_151674


namespace NUMINAMATH_CALUDE_sandwich_count_l1516_151646

/-- Represents the number of items bought in a week -/
structure WeeklyPurchase where
  sandwiches : ℕ
  cookies : ℕ
  total_items : sandwiches + cookies = 7

/-- Represents the cost in cents -/
def cost (p : WeeklyPurchase) : ℕ := 60 * p.sandwiches + 90 * p.cookies

theorem sandwich_count : 
  ∃ (p : WeeklyPurchase), 
    500 ≤ cost p ∧ 
    cost p ≤ 700 ∧ 
    cost p % 100 = 0 ∧
    p.sandwiches = 11 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_count_l1516_151646


namespace NUMINAMATH_CALUDE_punger_baseball_cards_l1516_151630

/-- The number of packs of baseball cards Punger bought -/
def number_of_packs : ℕ := sorry

/-- The number of cards in each pack -/
def cards_per_pack : ℕ := 7

/-- The number of cards each page can hold -/
def cards_per_page : ℕ := 10

/-- The number of pages Punger needs -/
def number_of_pages : ℕ := 42

theorem punger_baseball_cards : number_of_packs = 60 := by sorry

end NUMINAMATH_CALUDE_punger_baseball_cards_l1516_151630


namespace NUMINAMATH_CALUDE_gcd_294_84_l1516_151679

theorem gcd_294_84 : Nat.gcd 294 84 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_294_84_l1516_151679


namespace NUMINAMATH_CALUDE_employment_calculation_l1516_151683

/-- The percentage of employed people in the population of town X -/
def employed_percentage : ℝ := sorry

/-- The percentage of the population that are employed males -/
def employed_males_percentage : ℝ := 15

/-- The percentage of employed people who are females -/
def employed_females_percentage : ℝ := 75

theorem employment_calculation :
  employed_percentage = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_employment_calculation_l1516_151683


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1516_151642

-- Define the conditions
def p (x : ℝ) : Prop := |x - 1| < 2
def q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- Theorem statement
theorem p_sufficient_not_necessary : 
  (∀ x : ℝ, p x → q x) ∧ 
  (∃ x : ℝ, q x ∧ ¬(p x)) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1516_151642


namespace NUMINAMATH_CALUDE_inverse_f_at_4_l1516_151600

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the inverse function of f
def f_inv : ℝ → ℝ := sorry

-- State the symmetry condition
axiom symmetry_condition (x : ℝ) : f (x + 1) + f (1 - x) = 4

-- State that f has an inverse
axiom has_inverse : Function.Bijective f

-- State that f(4) = 0
axiom f_at_4 : f 4 = 0

-- Theorem to prove
theorem inverse_f_at_4 : f_inv 4 = -2 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_4_l1516_151600


namespace NUMINAMATH_CALUDE_max_intersections_12_6_l1516_151678

/-- The maximum number of intersection points in the first quadrant 
    given the number of points on x and y axes -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersections for 12 x-axis points
    and 6 y-axis points -/
theorem max_intersections_12_6 :
  max_intersections 12 6 = 990 := by
  sorry

#eval max_intersections 12 6

end NUMINAMATH_CALUDE_max_intersections_12_6_l1516_151678


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l1516_151689

theorem smallest_four_digit_mod_8 : 
  ∀ n : ℕ, 
    1000 ≤ n ∧ n ≡ 3 [MOD 8] → 
    1003 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l1516_151689


namespace NUMINAMATH_CALUDE_min_value_theorem_l1516_151665

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 14 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 2 / a₀ + 3 / b₀ = 14 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1516_151665


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1516_151650

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1516_151650


namespace NUMINAMATH_CALUDE_equidistant_function_c_squared_l1516_151680

/-- A complex function that is equidistant from its input and the origin -/
def EquidistantFunction (f : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs (f z - z) = Complex.abs (f z)

theorem equidistant_function_c_squared
  (a c : ℝ)
  (f : ℂ → ℂ)
  (h1 : f = fun z ↦ (a + c * Complex.I) * z)
  (h2 : EquidistantFunction f)
  (h3 : Complex.abs (a + c * Complex.I) = 5) :
  c^2 = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_function_c_squared_l1516_151680


namespace NUMINAMATH_CALUDE_number_properties_l1516_151649

theorem number_properties (n : ℕ) (h : n > 0) :
  (∃ (factors : Set ℕ), Finite factors ∧ ∀ k ∈ factors, n % k = 0) ∧
  (∃ (multiples : Set ℕ), ¬Finite multiples ∧ ∀ m ∈ multiples, m % n = 0) ∧
  (∀ k : ℕ, k ∣ n → k ≥ 1) ∧
  (∀ k : ℕ, k ∣ n → k ≤ n) ∧
  (∀ m : ℕ, n ∣ m → m ≥ n) := by
sorry

end NUMINAMATH_CALUDE_number_properties_l1516_151649


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1516_151636

/-- A quadratic polynomial with no real roots -/
structure QuadraticNoRoots where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  no_roots : ∀ x, a * x^2 + b * x + c > 0

/-- The quadratic function -/
def f (q : QuadraticNoRoots) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- The main theorem -/
theorem quadratic_inequality (q : QuadraticNoRoots) :
  ∀ x, f q x + f q (x - 1) - f q (x + 1) > -4 * q.a := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1516_151636


namespace NUMINAMATH_CALUDE_product_of_primes_factors_l1516_151606

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def nth_prime (n : ℕ) : ℕ := sorry

def sum_of_products (k : ℕ) : ℕ := sorry

theorem product_of_primes_factors (k : ℕ) (h : k > 4) :
  ∃ (factors : List ℕ), 
    (∀ f ∈ factors, is_prime f) ∧ 
    (factors.length ≥ 2 * k) ∧
    (factors.prod = sum_of_products k + 1) :=
sorry

end NUMINAMATH_CALUDE_product_of_primes_factors_l1516_151606


namespace NUMINAMATH_CALUDE_cara_speed_l1516_151625

/-- 
Proves that given a distance of 120 miles between two cities, 
if a person (Dan) leaving 60 minutes after another person (Cara) 
must exceed 40 mph to arrive first, then the first person's (Cara's) 
constant speed is 30 mph.
-/
theorem cara_speed (distance : ℝ) (dan_delay : ℝ) (dan_min_speed : ℝ) : 
  distance = 120 → 
  dan_delay = 1 → 
  dan_min_speed = 40 → 
  ∃ (cara_speed : ℝ), 
    cara_speed * (distance / dan_min_speed + dan_delay) = distance ∧ 
    cara_speed = 30 := by
  sorry


end NUMINAMATH_CALUDE_cara_speed_l1516_151625


namespace NUMINAMATH_CALUDE_remainder_proof_l1516_151673

theorem remainder_proof (k : ℤ) : (k * 1127 * 1129) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l1516_151673


namespace NUMINAMATH_CALUDE_sum_of_primes_divisible_by_60_l1516_151605

theorem sum_of_primes_divisible_by_60 (p q r s : ℕ) : 
  Prime p → Prime q → Prime r → Prime s →
  5 < p → p < q → q < r → r < s → s < p + 10 →
  60 ∣ (p + q + r + s) := by
sorry

end NUMINAMATH_CALUDE_sum_of_primes_divisible_by_60_l1516_151605


namespace NUMINAMATH_CALUDE_man_mass_l1516_151603

-- Define the boat's dimensions
def boat_length : Real := 3
def boat_breadth : Real := 2
def sinking_depth : Real := 0.01  -- 1 cm in meters

-- Define water density
def water_density : Real := 1000  -- kg/m³

-- Define the theorem
theorem man_mass (volume : Real) (h1 : volume = boat_length * boat_breadth * sinking_depth)
  (mass : Real) (h2 : mass = water_density * volume) : mass = 60 := by
  sorry

end NUMINAMATH_CALUDE_man_mass_l1516_151603


namespace NUMINAMATH_CALUDE_system_solution_l1516_151624

theorem system_solution : ∃ (x y : ℚ), 
  (7 * x = -10 - 3 * y) ∧ 
  (4 * x = 5 * y - 35) ∧ 
  (x = -155 / 47) ∧ 
  (y = 205 / 47) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1516_151624


namespace NUMINAMATH_CALUDE_jim_journey_remaining_distance_l1516_151657

/-- Given a total journey distance and the distance already driven, 
    calculate the remaining distance to be driven. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem stating that for a 1200-mile journey with 642 miles driven,
    the remaining distance is 558 miles. -/
theorem jim_journey_remaining_distance :
  remaining_distance 1200 642 = 558 := by
  sorry

end NUMINAMATH_CALUDE_jim_journey_remaining_distance_l1516_151657


namespace NUMINAMATH_CALUDE_gift_spending_theorem_l1516_151672

def num_siblings : ℕ := 3
def num_parents : ℕ := 2
def cost_per_sibling : ℕ := 30
def cost_per_parent : ℕ := 30

def total_cost : ℕ := num_siblings * cost_per_sibling + num_parents * cost_per_parent

theorem gift_spending_theorem : total_cost = 150 := by
  sorry

end NUMINAMATH_CALUDE_gift_spending_theorem_l1516_151672


namespace NUMINAMATH_CALUDE_salary_change_percentage_l1516_151602

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.75 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l1516_151602


namespace NUMINAMATH_CALUDE_first_podcast_length_l1516_151669

/-- Given a 6-hour drive and podcast lengths, prove the first podcast is 0.75 hours long -/
theorem first_podcast_length (total_time : ℝ) (podcast1 : ℝ) (podcast2 : ℝ) (podcast3 : ℝ) (podcast4 : ℝ) (podcast5 : ℝ) :
  total_time = 6 →
  podcast2 = 2 * podcast1 →
  podcast3 = 1.75 →
  podcast4 = 1 →
  podcast5 = 1 →
  podcast1 + podcast2 + podcast3 + podcast4 + podcast5 = total_time →
  podcast1 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_first_podcast_length_l1516_151669


namespace NUMINAMATH_CALUDE_division_of_fractions_l1516_151698

theorem division_of_fractions : (3 : ℚ) / 7 / 4 = 3 / 28 := by sorry

end NUMINAMATH_CALUDE_division_of_fractions_l1516_151698


namespace NUMINAMATH_CALUDE_modulus_of_z_l1516_151637

theorem modulus_of_z (z : ℂ) : z = 5 / (1 - 2*I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1516_151637


namespace NUMINAMATH_CALUDE_parabola_fixed_point_l1516_151611

/-- The parabola passes through the point (3, 36) for all real t -/
theorem parabola_fixed_point :
  ∀ t : ℝ, 36 = 4 * (3 : ℝ)^2 + t * 3 - t^2 - 3 * t := by
  sorry

end NUMINAMATH_CALUDE_parabola_fixed_point_l1516_151611


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l1516_151623

/-- Given line segments a and b, x is their geometric mean if x^2 = ab -/
def is_geometric_mean (a b x : ℝ) : Prop := x^2 = a * b

/-- Proof that for line segments a = 4 and b = 9, their geometric mean x equals 6 -/
theorem geometric_mean_of_4_and_9 :
  ∀ x : ℝ, is_geometric_mean 4 9 x → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l1516_151623


namespace NUMINAMATH_CALUDE_cross_section_area_less_than_half_face_l1516_151641

/-- A cube with an inscribed sphere and a triangular cross-section touching the sphere -/
structure CubeWithSphereAndCrossSection where
  /-- Side length of the cube -/
  a : ℝ
  /-- Assumption that the cube has positive side length -/
  a_pos : 0 < a
  /-- The triangular cross-section touches the inscribed sphere -/
  touches_sphere : Bool

/-- The area of the triangular cross-section is less than half the area of the cube face -/
theorem cross_section_area_less_than_half_face (cube : CubeWithSphereAndCrossSection) :
  ∃ (area : ℝ), area < (1/2) * cube.a^2 ∧ 
  (∀ (cross_section_area : ℝ), cross_section_area ≤ area) :=
sorry

end NUMINAMATH_CALUDE_cross_section_area_less_than_half_face_l1516_151641


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1516_151655

theorem product_of_sum_and_sum_of_cubes (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (sum_cubes_eq : x^3 + y^3 = 370) : 
  x * y = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l1516_151655


namespace NUMINAMATH_CALUDE_circle_equation_l1516_151631

/-- The circle C with center (a, b) in the second quadrant -/
structure Circle where
  a : ℝ
  b : ℝ
  h1 : a < 0
  h2 : b > 0

/-- The line 3x+y-5=0 -/
def line1 (x y : ℝ) : Prop := 3*x + y - 5 = 0

/-- The line 2x-3y+4=0 -/
def line2 (x y : ℝ) : Prop := 2*x - 3*y + 4 = 0

/-- The line 3x-4y+5=0 -/
def line3 (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

/-- The x-axis -/
def xAxis (y : ℝ) : Prop := y = 0

/-- The intersection point of line1 and line2 -/
def intersectionPoint : ℝ × ℝ := (1, 2)

/-- Circle C passes through the intersection point -/
def passesThroughIntersection (C : Circle) : Prop :=
  let (x, y) := intersectionPoint
  (x - C.a)^2 + (y - C.b)^2 = C.b^2

/-- Circle C is tangent to line3 -/
def tangentToLine3 (C : Circle) : Prop :=
  |3*C.a - 4*C.b + 5| / 5 = C.b

/-- Circle C is tangent to x-axis -/
def tangentToXAxis (C : Circle) : Prop :=
  C.b = C.b

theorem circle_equation (C : Circle) 
  (h1 : passesThroughIntersection C)
  (h2 : tangentToLine3 C)
  (h3 : tangentToXAxis C) :
  ∀ (x y : ℝ), (x + 5)^2 + (y - 10)^2 = 100 ↔ (x - C.a)^2 + (y - C.b)^2 = C.b^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1516_151631


namespace NUMINAMATH_CALUDE_optimal_selling_price_l1516_151629

/-- Represents the store's pricing problem --/
structure StorePricing where
  purchasePrice : ℝ
  demandSlope : ℝ
  demandIntercept : ℝ
  maxPriceIncrease : ℝ
  desiredProfit : ℝ

/-- Calculates the profit for a given selling price --/
def profit (sp : StorePricing) (sellingPrice : ℝ) : ℝ :=
  (sellingPrice - sp.purchasePrice) * (sp.demandIntercept - sp.demandSlope * sellingPrice)

/-- Checks if the selling price satisfies the government restriction --/
def satisfiesRestriction (sp : StorePricing) (sellingPrice : ℝ) : Prop :=
  (sellingPrice - sp.purchasePrice) / sp.purchasePrice ≤ sp.maxPriceIncrease

/-- Theorem stating that 41 yuan is the optimal selling price --/
theorem optimal_selling_price (sp : StorePricing) 
  (h_purchase : sp.purchasePrice = 30)
  (h_demand : sp.demandSlope = 2 ∧ sp.demandIntercept = 112)
  (h_restriction : sp.maxPriceIncrease = 0.4)
  (h_profit : sp.desiredProfit = 330) :
  ∃ (optimalPrice : ℝ), 
    optimalPrice = 41 ∧ 
    satisfiesRestriction sp optimalPrice ∧ 
    profit sp optimalPrice = sp.desiredProfit ∧
    ∀ (price : ℝ), satisfiesRestriction sp price → profit sp price ≤ profit sp optimalPrice :=
sorry


end NUMINAMATH_CALUDE_optimal_selling_price_l1516_151629


namespace NUMINAMATH_CALUDE_minimum_rows_needed_l1516_151638

structure School where
  students : ℕ
  h1 : 1 ≤ students
  h2 : students ≤ 39

def City := List School

def totalStudents (city : City) : ℕ :=
  city.map (λ s => s.students) |>.sum

theorem minimum_rows_needed (city : City) 
  (h_total : totalStudents city = 1990) 
  (h_seats_per_row : ℕ := 199) : ℕ :=
  12

#check minimum_rows_needed

end NUMINAMATH_CALUDE_minimum_rows_needed_l1516_151638
