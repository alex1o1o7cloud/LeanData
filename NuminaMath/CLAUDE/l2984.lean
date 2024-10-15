import Mathlib

namespace NUMINAMATH_CALUDE_intersection_k_range_l2984_298494

-- Define the line equation
def line_eq (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the condition for intersection at two distinct points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧
  hyperbola_eq x₁ (line_eq k x₁) ∧
  hyperbola_eq x₂ (line_eq k x₂)

-- State the theorem
theorem intersection_k_range :
  ∀ k : ℝ, intersects_at_two_points k ↔ 1 < k ∧ k < Real.sqrt 15 / 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_k_range_l2984_298494


namespace NUMINAMATH_CALUDE_distance_between_intersection_points_l2984_298458

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 9

-- Define the line that intersects the circle
def intersecting_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ circle_C x y ∧ intersecting_line x y}

-- State the theorem
theorem distance_between_intersection_points :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_intersection_points_l2984_298458


namespace NUMINAMATH_CALUDE_different_smallest_angles_l2984_298464

/-- A type representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A type representing a set of 6 points in a plane -/
structure SixPoints :=
  (points : Fin 6 → Point)

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Predicate to check if no three points in a set of six points are collinear -/
def no_three_collinear (s : SixPoints) : Prop :=
  ∀ (i j k : Fin 6), i ≠ j → j ≠ k → i ≠ k →
    ¬collinear (s.points i) (s.points j) (s.points k)

/-- Function to calculate the angle between three points -/
noncomputable def angle (p q r : Point) : ℝ := sorry

/-- Function to find the smallest angle in a triangle -/
noncomputable def smallest_angle (p q r : Point) : ℝ :=
  min (angle p q r) (min (angle q r p) (angle r p q))

/-- The main theorem -/
theorem different_smallest_angles (s : SixPoints) (h : no_three_collinear s) :
  ∃ (i₁ j₁ k₁ i₂ j₂ k₂ : Fin 6),
    smallest_angle (s.points i₁) (s.points j₁) (s.points k₁) ≠
    smallest_angle (s.points i₂) (s.points j₂) (s.points k₂) :=
  sorry

end NUMINAMATH_CALUDE_different_smallest_angles_l2984_298464


namespace NUMINAMATH_CALUDE_dress_savings_l2984_298472

/-- Given a dress with an original cost of $180, if someone buys it for 10 dollars less than half the price, they save $100. -/
theorem dress_savings (original_cost : ℕ) (purchase_price : ℕ) : 
  original_cost = 180 → 
  purchase_price = original_cost / 2 - 10 → 
  original_cost - purchase_price = 100 := by
sorry

end NUMINAMATH_CALUDE_dress_savings_l2984_298472


namespace NUMINAMATH_CALUDE_negative_distribution_l2984_298440

theorem negative_distribution (a b c : ℝ) : -(a - b + c) = -a + b - c := by
  sorry

end NUMINAMATH_CALUDE_negative_distribution_l2984_298440


namespace NUMINAMATH_CALUDE_probability_two_black_balls_l2984_298477

def total_balls : ℕ := 15
def black_balls : ℕ := 10
def white_balls : ℕ := 5

theorem probability_two_black_balls :
  let p_first_black : ℚ := black_balls / total_balls
  let p_second_black : ℚ := (black_balls - 1) / (total_balls - 1)
  p_first_black * p_second_black = 3 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_two_black_balls_l2984_298477


namespace NUMINAMATH_CALUDE_water_saving_calculation_l2984_298471

/-- The amount of water Hyunwoo's family uses daily in liters -/
def daily_water_usage : ℝ := 215

/-- The fraction of water saved when adjusting the water pressure valve weakly -/
def water_saving_fraction : ℝ := 0.32

/-- The amount of water saved when adjusting the water pressure valve weakly -/
def water_saved : ℝ := daily_water_usage * water_saving_fraction

theorem water_saving_calculation :
  water_saved = 68.8 := by sorry

end NUMINAMATH_CALUDE_water_saving_calculation_l2984_298471


namespace NUMINAMATH_CALUDE_circles_cover_quadrilateral_l2984_298452

-- Define a convex quadrilateral
def ConvexQuadrilateral (A B C D : Real × Real) : Prop :=
  -- Add conditions for convexity
  sorry

-- Define a circle with diameter as a side of the quadrilateral
def CircleOnSide (A B : Real × Real) : Set (Real × Real) :=
  {P | ∃ (t : Real), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B ∧ 
    (P.1 - A.1)^2 + (P.2 - A.2)^2 ≤ ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 4}

-- Define the union of four circles on the sides of the quadrilateral
def UnionOfCircles (A B C D : Real × Real) : Set (Real × Real) :=
  CircleOnSide A B ∪ CircleOnSide B C ∪ CircleOnSide C D ∪ CircleOnSide D A

-- Define the interior of the quadrilateral
def QuadrilateralInterior (A B C D : Real × Real) : Set (Real × Real) :=
  -- Add definition for the interior of the quadrilateral
  sorry

-- Theorem statement
theorem circles_cover_quadrilateral (A B C D : Real × Real) :
  ConvexQuadrilateral A B C D →
  QuadrilateralInterior A B C D ⊆ UnionOfCircles A B C D :=
sorry

end NUMINAMATH_CALUDE_circles_cover_quadrilateral_l2984_298452


namespace NUMINAMATH_CALUDE_solution_value_l2984_298499

theorem solution_value (t : ℝ) : 
  (let y := -(t - 1)
   2 * y - 4 = 3 * (y - 2)) → 
  t = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2984_298499


namespace NUMINAMATH_CALUDE_equal_payment_payment_difference_l2984_298400

/-- Represents the pizza scenario with given conditions -/
structure PizzaScenario where
  total_slices : ℕ
  meat_slices : ℕ
  plain_cost : ℚ
  meat_cost : ℚ
  joe_meat_slices : ℕ
  joe_veg_slices : ℕ

/-- Calculate the total cost of the pizza -/
def total_cost (p : PizzaScenario) : ℚ :=
  p.plain_cost + p.meat_cost

/-- Calculate the cost per slice -/
def cost_per_slice (p : PizzaScenario) : ℚ :=
  total_cost p / p.total_slices

/-- Calculate Joe's payment -/
def joe_payment (p : PizzaScenario) : ℚ :=
  cost_per_slice p * (p.joe_meat_slices + p.joe_veg_slices)

/-- Calculate Karen's payment -/
def karen_payment (p : PizzaScenario) : ℚ :=
  cost_per_slice p * (p.total_slices - p.joe_meat_slices - p.joe_veg_slices)

/-- The main theorem stating that Joe and Karen paid the same amount -/
theorem equal_payment (p : PizzaScenario) 
  (h1 : p.total_slices = 12)
  (h2 : p.meat_slices = 4)
  (h3 : p.plain_cost = 12)
  (h4 : p.meat_cost = 4)
  (h5 : p.joe_meat_slices = 4)
  (h6 : p.joe_veg_slices = 2) :
  joe_payment p = karen_payment p :=
by sorry

/-- The difference in payment is zero -/
theorem payment_difference (p : PizzaScenario) 
  (h1 : p.total_slices = 12)
  (h2 : p.meat_slices = 4)
  (h3 : p.plain_cost = 12)
  (h4 : p.meat_cost = 4)
  (h5 : p.joe_meat_slices = 4)
  (h6 : p.joe_veg_slices = 2) :
  joe_payment p - karen_payment p = 0 :=
by sorry

end NUMINAMATH_CALUDE_equal_payment_payment_difference_l2984_298400


namespace NUMINAMATH_CALUDE_instrument_players_fraction_l2984_298495

theorem instrument_players_fraction 
  (total_people : ℕ) 
  (two_or_more : ℕ) 
  (prob_exactly_one : ℚ) 
  (h1 : total_people = 800) 
  (h2 : two_or_more = 128) 
  (h3 : prob_exactly_one = 1/25) : 
  (↑two_or_more + ↑total_people * prob_exactly_one) / ↑total_people = 1/5 := by
sorry

end NUMINAMATH_CALUDE_instrument_players_fraction_l2984_298495


namespace NUMINAMATH_CALUDE_unique_solution_l2984_298488

-- Define the two equations
def equation1 (x y : ℝ) : Prop := (x + y - 5) * (2 * x - 3 * y + 5) = 0
def equation2 (x y : ℝ) : Prop := (x - y + 1) * (3 * x + 2 * y - 12) = 0

-- Define a solution as a point satisfying both equations
def is_solution (p : ℝ × ℝ) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

-- Theorem stating that there is exactly one solution
theorem unique_solution : ∃! p : ℝ × ℝ, is_solution p :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2984_298488


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l2984_298489

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 2) + 2

theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = 2 ∧ f a 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l2984_298489


namespace NUMINAMATH_CALUDE_ellipse_point_inside_circle_l2984_298485

theorem ellipse_point_inside_circle 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (he : c / a = 1 / 2) 
  (hf : c > 0) 
  (x₁ x₂ : ℝ) 
  (hroots : x₁ * x₂ = -c / a ∧ x₁ + x₂ = -b / a) :
  x₁^2 + x₂^2 < 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_point_inside_circle_l2984_298485


namespace NUMINAMATH_CALUDE_min_value_theorem_l2984_298419

theorem min_value_theorem (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  (1 / (2 * x)) + (x / (y + 1)) ≥ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2984_298419


namespace NUMINAMATH_CALUDE_prime_pairs_satisfying_equation_l2984_298462

theorem prime_pairs_satisfying_equation :
  ∀ p q n : ℕ,
    Prime p → Prime q → n > 0 →
    p * (p + 1) + q * (q + 1) = n * (n + 1) →
    ((p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) ∨ (p = 2 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_satisfying_equation_l2984_298462


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2984_298417

theorem intersection_implies_a_value (a : ℝ) : 
  let M : Set ℝ := {1, 2, a^2 - 3*a - 1}
  let N : Set ℝ := {-1, a, 3}
  (M ∩ N = {3}) → a = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2984_298417


namespace NUMINAMATH_CALUDE_proportional_function_decreases_l2984_298442

/-- Proves that for a proportional function y = kx passing through the point (4, -1),
    where k is a non-zero constant, y decreases as x increases. -/
theorem proportional_function_decreases (k : ℝ) (h1 : k ≠ 0) (h2 : k * 4 = -1) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ > k * x₂ := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_decreases_l2984_298442


namespace NUMINAMATH_CALUDE_max_inscribed_cylinder_volume_l2984_298408

/-- 
Given a right circular cone with base radius R and height M, 
prove that the maximum volume of an inscribed right circular cylinder 
is 4πMR²/27, and this volume is 4/9 of the cone's volume.
-/
theorem max_inscribed_cylinder_volume (R M : ℝ) (hR : R > 0) (hM : M > 0) :
  let cone_volume := (1/3) * π * R^2 * M
  let max_cylinder_volume := (4/27) * π * M * R^2
  max_cylinder_volume = (4/9) * cone_volume := by
  sorry


end NUMINAMATH_CALUDE_max_inscribed_cylinder_volume_l2984_298408


namespace NUMINAMATH_CALUDE_viewers_scientific_notation_equality_l2984_298411

-- Define the number of viewers
def viewers : ℕ := 16300000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.63 * (10 ^ 10)

-- Theorem to prove the equality
theorem viewers_scientific_notation_equality :
  (viewers : ℝ) = scientific_notation := by sorry

end NUMINAMATH_CALUDE_viewers_scientific_notation_equality_l2984_298411


namespace NUMINAMATH_CALUDE_third_derivative_y_l2984_298491

noncomputable def y (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = 4 / (1 + x^2)^2 := by sorry

end NUMINAMATH_CALUDE_third_derivative_y_l2984_298491


namespace NUMINAMATH_CALUDE_multiple_properties_l2984_298434

theorem multiple_properties (a b : ℤ) 
  (h1 : ∃ k : ℤ, a = 5 * k)
  (h2 : ∃ m : ℤ, a = 2 * m + 1)
  (h3 : ∃ n : ℤ, b = 10 * n) :
  (∃ p : ℤ, b = 5 * p) ∧ (∃ q : ℤ, a - b = 5 * q) := by
sorry

end NUMINAMATH_CALUDE_multiple_properties_l2984_298434


namespace NUMINAMATH_CALUDE_negation_equivalence_l2984_298483

-- Define the universe of switches and lights
variable (Switch Light : Type)

-- Define the state of switches and lights
variable (is_off : Switch → Prop)
variable (is_on : Light → Prop)

-- Define the main switch
variable (main_switch : Switch)

-- Define the conditions
variable (h1 : ∀ s : Switch, is_off s → ∀ l : Light, ¬(is_on l))
variable (h2 : is_off main_switch → ∀ s : Switch, is_off s)

-- The theorem to prove
theorem negation_equivalence :
  ¬(is_off main_switch → ∀ l : Light, ¬(is_on l)) ↔
  (is_off main_switch ∧ ∃ l : Light, is_on l) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2984_298483


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2984_298436

theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * Real.cos y) :
  ∀ t : ℝ, f t = f 0 * Real.cos t + f (Real.pi / 2) * Real.sin t :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2984_298436


namespace NUMINAMATH_CALUDE_initial_seashell_count_l2984_298445

theorem initial_seashell_count (henry paul leo : ℕ) : 
  henry = 11 →
  paul = 24 →
  henry + paul + (3/4 * leo) = 53 →
  henry + paul + leo = 59 :=
by sorry

end NUMINAMATH_CALUDE_initial_seashell_count_l2984_298445


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2984_298482

/-- A rectangular solid with prime edge lengths and volume 399 has surface area 422. -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 399 →
  2 * (a * b + b * c + c * a) = 422 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2984_298482


namespace NUMINAMATH_CALUDE_estimate_theorem_l2984_298407

/-- Represents a company with employees and their distance from workplace -/
structure Company where
  total_employees : ℕ
  sample_size : ℕ
  within_1000m : ℕ
  within_2000m : ℕ

/-- Calculates the estimated number of employees living between 1000 and 2000 meters -/
def estimate_between_1000_2000 (c : Company) : ℕ :=
  let sample_between := c.within_2000m - c.within_1000m
  (sample_between * c.total_employees) / c.sample_size

/-- Theorem stating the estimated number of employees living between 1000 and 2000 meters -/
theorem estimate_theorem (c : Company) 
  (h1 : c.total_employees = 2000)
  (h2 : c.sample_size = 200)
  (h3 : c.within_1000m = 10)
  (h4 : c.within_2000m = 30) :
  estimate_between_1000_2000 c = 200 := by
  sorry

#eval estimate_between_1000_2000 { total_employees := 2000, sample_size := 200, within_1000m := 10, within_2000m := 30 }

end NUMINAMATH_CALUDE_estimate_theorem_l2984_298407


namespace NUMINAMATH_CALUDE_trig_expression_equality_l2984_298460

theorem trig_expression_equality : 
  (Real.sin (20 * π / 180) * Real.sqrt (1 + Real.cos (40 * π / 180))) / Real.cos (50 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l2984_298460


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l2984_298405

theorem solve_equation_and_evaluate (x : ℚ) : 
  (4 * x - 8 = 12 * x + 4) → (5 * (x - 3) = -45 / 2) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l2984_298405


namespace NUMINAMATH_CALUDE_earl_floor_problem_l2984_298446

theorem earl_floor_problem (total_floors : ℕ) (initial_floor : ℕ) (first_up : ℕ) (second_up : ℕ) (floors_from_top : ℕ) (floors_down : ℕ) :
  total_floors = 20 →
  initial_floor = 1 →
  first_up = 5 →
  second_up = 7 →
  floors_from_top = 9 →
  initial_floor + first_up - floors_down + second_up = total_floors - floors_from_top →
  floors_down = 2 := by
sorry

end NUMINAMATH_CALUDE_earl_floor_problem_l2984_298446


namespace NUMINAMATH_CALUDE_problem_statement_l2984_298438

theorem problem_statement (p q : Prop) 
  (hp : p ↔ 3 % 2 = 1) 
  (hq : q ↔ 5 % 2 = 0) : 
  p ∨ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2984_298438


namespace NUMINAMATH_CALUDE_viggo_age_ratio_l2984_298481

theorem viggo_age_ratio :
  ∀ (viggo_current_age brother_current_age M Y : ℕ),
    viggo_current_age + brother_current_age = 32 →
    brother_current_age = 10 →
    viggo_current_age - brother_current_age = M * 2 + Y - 2 →
    (M * 2 + Y) / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_viggo_age_ratio_l2984_298481


namespace NUMINAMATH_CALUDE_comprehensive_score_example_l2984_298427

/-- Calculates the comprehensive score given regular assessment and final exam scores and their weightings -/
def comprehensive_score (regular_score : ℝ) (final_score : ℝ) (regular_weight : ℝ) (final_weight : ℝ) : ℝ :=
  regular_score * regular_weight + final_score * final_weight

/-- Proves that the comprehensive score is 91 given the specified scores and weightings -/
theorem comprehensive_score_example : 
  comprehensive_score 95 90 0.2 0.8 = 91 := by
  sorry

end NUMINAMATH_CALUDE_comprehensive_score_example_l2984_298427


namespace NUMINAMATH_CALUDE_allans_balloons_prove_allans_balloons_l2984_298444

theorem allans_balloons (jake_balloons : ℕ) (difference : ℕ) : ℕ :=
  jake_balloons + difference

theorem prove_allans_balloons :
  allans_balloons 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_allans_balloons_prove_allans_balloons_l2984_298444


namespace NUMINAMATH_CALUDE_power_seven_mod_nine_l2984_298404

theorem power_seven_mod_nine : 7^145 % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_nine_l2984_298404


namespace NUMINAMATH_CALUDE_total_blocks_l2984_298457

theorem total_blocks (initial_blocks additional_blocks : ℕ) :
  initial_blocks = 86 →
  additional_blocks = 9 →
  initial_blocks + additional_blocks = 95 :=
by sorry

end NUMINAMATH_CALUDE_total_blocks_l2984_298457


namespace NUMINAMATH_CALUDE_hyperbola_from_ellipse_l2984_298420

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define the hyperbola equations
def hyperbola1 (x y : ℝ) : Prop := x^2/16 - y^2/48 = 1
def hyperbola2 (x y : ℝ) : Prop := y^2/9 - x^2/27 = 1

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_from_ellipse :
  ∀ x y : ℝ, ellipse x y →
  (∃ a b : ℝ, (hyperbola1 a b ∧ (a = x ∨ a = -x) ∧ (b = y ∨ b = -y)) ∨
              (hyperbola2 a b ∧ (a = x ∨ a = -x) ∧ (b = y ∨ b = -y))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_from_ellipse_l2984_298420


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l2984_298409

theorem solve_system_of_equations (x y : ℤ) 
  (h1 : x + y = 250) 
  (h2 : x - y = 200) : 
  y = 25 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l2984_298409


namespace NUMINAMATH_CALUDE_texas_passengers_on_l2984_298453

/-- Represents the number of passengers at different stages of the flight --/
structure PassengerCount where
  initial : ℕ
  texas_off : ℕ
  texas_on : ℕ
  nc_off : ℕ
  nc_on : ℕ
  crew : ℕ
  final : ℕ

/-- Theorem stating that given the flight conditions, 24 passengers got on in Texas --/
theorem texas_passengers_on (p : PassengerCount) 
  (h1 : p.initial = 124)
  (h2 : p.texas_off = 58)
  (h3 : p.nc_off = 47)
  (h4 : p.nc_on = 14)
  (h5 : p.crew = 10)
  (h6 : p.final = 67)
  (h7 : p.final = p.initial - p.texas_off + p.texas_on - p.nc_off + p.nc_on + p.crew) :
  p.texas_on = 24 := by
  sorry

end NUMINAMATH_CALUDE_texas_passengers_on_l2984_298453


namespace NUMINAMATH_CALUDE_min_value_of_f_l2984_298435

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

theorem min_value_of_f (a : ℝ) :
  (∃ (h : ℝ), ∀ x, f a x ≥ f a (-2)) →
  (∃ (m : ℝ), ∀ x, f a x ≥ m ∧ ∃ y, f a y = m) →
  (∃ (m : ℝ), ∀ x, f a x ≥ m ∧ ∃ y, f a y = m ∧ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2984_298435


namespace NUMINAMATH_CALUDE_min_value_theorem_l2984_298474

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x + 2*y + 3*z = 12) : 
  9/x + 4/y + 1/z ≥ 49/12 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2984_298474


namespace NUMINAMATH_CALUDE_dice_probability_l2984_298433

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice being rolled -/
def num_dice : ℕ := 6

/-- The probability of rolling all the same numbers -/
def prob_all_same : ℚ := 1 / (sides ^ (num_dice - 1))

/-- The probability of not rolling all the same numbers -/
def prob_not_all_same : ℚ := 1 - prob_all_same

theorem dice_probability :
  prob_not_all_same = 7775 / 7776 :=
sorry

end NUMINAMATH_CALUDE_dice_probability_l2984_298433


namespace NUMINAMATH_CALUDE_largest_t_value_l2984_298414

theorem largest_t_value (t : ℚ) : 
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  t ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_largest_t_value_l2984_298414


namespace NUMINAMATH_CALUDE_stratified_sampling_equal_allocation_l2984_298447

/-- Represents a group of workers -/
structure WorkerGroup where
  total : Nat
  female : Nat

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  groupA : WorkerGroup
  groupB : WorkerGroup
  totalSamples : Nat

/-- Theorem: In a stratified sampling scenario with two equal-sized strata,
    the number of samples drawn from each stratum is equal to half of the total sample size -/
theorem stratified_sampling_equal_allocation 
  (sample : StratifiedSample) 
  (h1 : sample.groupA.total = sample.groupB.total)
  (h2 : sample.totalSamples % 2 = 0) :
  ∃ (n : Nat), n = sample.totalSamples / 2 ∧ 
               n = sample.totalSamples - n :=
sorry

#check stratified_sampling_equal_allocation

end NUMINAMATH_CALUDE_stratified_sampling_equal_allocation_l2984_298447


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l2984_298493

def M : Matrix (Fin 2) (Fin 2) ℚ :=
  !![3, 1; 0, -2]

theorem inverse_as_linear_combination :
  ∃ (a b : ℚ), a = 1/6 ∧ b = -1/6 ∧ M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l2984_298493


namespace NUMINAMATH_CALUDE_task_completion_time_relation_l2984_298421

/-- 
Theorem: Given three individuals A, B, and C working on a task, where:
- A's time = m * (B and C's time together)
- B's time = n * (A and C's time together)
- C's time = k * (A and B's time together)
Then k can be expressed in terms of m and n as: k = (m + n + 2) / (mn - 1)
-/
theorem task_completion_time_relation (m n k : ℝ) (hm : m > 0) (hn : n > 0) (hk : k > 0) :
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    (1 / x = m / (y + z)) ∧
    (1 / y = n / (x + z)) ∧
    (1 / z = k / (x + y))) →
  k = (m + n + 2) / (m * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_task_completion_time_relation_l2984_298421


namespace NUMINAMATH_CALUDE_range_of_expression_l2984_298401

theorem range_of_expression (x y : ℝ) (h1 : x + 2*y - 6 = 0) (h2 : 0 < x) (h3 : x < 3) :
  1 < (x + 2) / (y - 1) ∧ (x + 2) / (y - 1) < 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l2984_298401


namespace NUMINAMATH_CALUDE_average_age_decrease_l2984_298461

/-- Proves that replacing a 46-year-old person with a 16-year-old person in a group of 10 decreases the average age by 3 years -/
theorem average_age_decrease (initial_avg : ℝ) : 
  let total_age := 10 * initial_avg
  let new_total_age := total_age - 46 + 16
  let new_avg := new_total_age / 10
  initial_avg - new_avg = 3 := by sorry

end NUMINAMATH_CALUDE_average_age_decrease_l2984_298461


namespace NUMINAMATH_CALUDE_average_fish_caught_l2984_298490

def fish_caught (person : String) : ℕ :=
  match person with
  | "Aang" => 7
  | "Sokka" => 5
  | "Toph" => 12
  | _ => 0

def people : List String := ["Aang", "Sokka", "Toph"]

theorem average_fish_caught :
  (people.map fish_caught).sum / people.length = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_fish_caught_l2984_298490


namespace NUMINAMATH_CALUDE_work_completion_time_l2984_298486

/-- The time it takes for A to finish the remaining work after B has worked for 10 days -/
def remaining_time_for_A (a_time b_time b_work_days : ℚ) : ℚ :=
  (1 - b_work_days / b_time) / (1 / a_time)

theorem work_completion_time :
  remaining_time_for_A 9 15 10 = 3 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2984_298486


namespace NUMINAMATH_CALUDE_quadratic_roots_and_m_value_l2984_298473

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-3)*x - m

-- Theorem statement
theorem quadratic_roots_and_m_value (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0) ∧
  (∀ x₁ x₂ : ℝ, quadratic m x₁ = 0 → quadratic m x₂ = 0 → x₁^2 + x₂^2 - x₁*x₂ = 7 → m = 1 ∨ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_m_value_l2984_298473


namespace NUMINAMATH_CALUDE_chase_travel_time_l2984_298480

/-- Represents the journey from Granville to Salisbury with intermediate stops -/
structure Journey where
  chase_speed : ℝ
  cameron_speed : ℝ
  danielle_speed : ℝ
  chase_scooter_speed : ℝ
  cameron_bike_speed : ℝ
  danielle_time : ℝ

/-- The conditions of the journey -/
def journey_conditions (j : Journey) : Prop :=
  j.cameron_speed = 2 * j.chase_speed ∧
  j.danielle_speed = 3 * j.cameron_speed ∧
  j.cameron_bike_speed = 0.75 * j.cameron_speed ∧
  j.chase_scooter_speed = 1.25 * j.chase_speed ∧
  j.danielle_time = 30

/-- The theorem stating that Chase's travel time is 180 minutes -/
theorem chase_travel_time (j : Journey) 
  (h : journey_conditions j) : 
  (180 : ℝ) * j.chase_speed = j.danielle_speed * j.danielle_time :=
sorry

end NUMINAMATH_CALUDE_chase_travel_time_l2984_298480


namespace NUMINAMATH_CALUDE_gcf_lcm_problem_l2984_298451

-- Define GCF (Greatest Common Factor)
def GCF (a b : ℕ) : ℕ := sorry

-- Define LCM (Least Common Multiple)
def LCM (c d : ℕ) : ℕ := sorry

-- Theorem statement
theorem gcf_lcm_problem : GCF (LCM 9 15) (LCM 10 21) = 15 := by sorry

end NUMINAMATH_CALUDE_gcf_lcm_problem_l2984_298451


namespace NUMINAMATH_CALUDE_milk_pouring_l2984_298492

theorem milk_pouring (initial_amount : ℚ) (pour_fraction : ℚ) : 
  initial_amount = 3/7 → pour_fraction = 5/8 → pour_fraction * initial_amount = 15/56 := by
  sorry

end NUMINAMATH_CALUDE_milk_pouring_l2984_298492


namespace NUMINAMATH_CALUDE_increasing_sequences_count_l2984_298467

theorem increasing_sequences_count :
  let n := 2013
  let k := 12
  let count := Nat.choose (((n - 1) / 2) + k - 1) k
  (count = Nat.choose 1017 12) ∧
  (1017 % 1000 = 17) := by sorry

end NUMINAMATH_CALUDE_increasing_sequences_count_l2984_298467


namespace NUMINAMATH_CALUDE_newspaper_delivery_start_l2984_298413

def building_floors : ℕ := 20

def start_floor : ℕ → Prop
| f => ∃ (current : ℕ), 
    current = f + 5 - 2 + 7 ∧ 
    current = building_floors - 9

theorem newspaper_delivery_start : start_floor 1 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_delivery_start_l2984_298413


namespace NUMINAMATH_CALUDE_sqrt_72_plus_sqrt_32_l2984_298465

theorem sqrt_72_plus_sqrt_32 : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_72_plus_sqrt_32_l2984_298465


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2984_298429

theorem gain_percent_calculation (MP : ℝ) (MP_pos : MP > 0) : 
  let CP := 0.64 * MP
  let SP := 0.82 * MP
  let gain_percent := ((SP - CP) / CP) * 100
  gain_percent = 28.125 := by sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2984_298429


namespace NUMINAMATH_CALUDE_smallest_a_for_polynomial_l2984_298424

theorem smallest_a_for_polynomial (a b : ℤ) (r₁ r₂ r₃ : ℕ+) : 
  r₁ * r₂ * r₃ = 1806 →
  r₁ + r₂ + r₃ = a →
  ∀ a' : ℤ, (∃ b' r₁' r₂' r₃' : ℕ+, 
    r₁' * r₂' * r₃' = 1806 ∧ 
    r₁' + r₂' + r₃' = a') → 
  a ≤ a' →
  a = 76 := by
sorry

end NUMINAMATH_CALUDE_smallest_a_for_polynomial_l2984_298424


namespace NUMINAMATH_CALUDE_chris_parents_gift_l2984_298422

/-- The amount of money Chris had before his birthday -/
def before_birthday : ℕ := 159

/-- The amount Chris received from his grandmother -/
def from_grandmother : ℕ := 25

/-- The amount Chris received from his aunt and uncle -/
def from_aunt_uncle : ℕ := 20

/-- The total amount Chris had after his birthday -/
def total_after_birthday : ℕ := 279

/-- The amount Chris's parents gave him -/
def from_parents : ℕ := total_after_birthday - before_birthday - from_grandmother - from_aunt_uncle

theorem chris_parents_gift : from_parents = 75 := by
  sorry

end NUMINAMATH_CALUDE_chris_parents_gift_l2984_298422


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l2984_298426

theorem largest_integer_with_remainder : ∃ n : ℕ, 
  n < 200 ∧ 
  n % 7 = 4 ∧ 
  ∀ m : ℕ, m < 200 → m % 7 = 4 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l2984_298426


namespace NUMINAMATH_CALUDE_min_value_expression_l2984_298476

theorem min_value_expression (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + 3*y = 2) :
  1/x + 3/y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 2 ∧ 1/x₀ + 3/y₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2984_298476


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2984_298428

/-- Given a geometric sequence {aₙ} where a₁ = 1 and a₄ = 27, prove that a₃ = 9 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- The geometric sequence
  (h1 : a 1 = 1)  -- First term is 1
  (h4 : a 4 = 27)  -- Fourth term is 27
  (h_geom : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a n = a 1 * q^(n-1))  -- Definition of geometric sequence
  : a 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2984_298428


namespace NUMINAMATH_CALUDE_line_intersection_canonical_form_l2984_298437

/-- Given two planes in 3D space, this theorem proves that their line of intersection
    can be represented by specific canonical equations. -/
theorem line_intersection_canonical_form :
  ∀ (x y z : ℝ),
  (x + y - 2*z - 2 = 0 ∧ x - y + z + 2 = 0) →
  ∃ (t : ℝ), x = -t ∧ y = -3*t + 2 ∧ z = -2*t := by sorry

end NUMINAMATH_CALUDE_line_intersection_canonical_form_l2984_298437


namespace NUMINAMATH_CALUDE_five_Y_three_equals_two_l2984_298463

-- Define the Y operation
def Y (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2 - x + y

-- Theorem statement
theorem five_Y_three_equals_two : Y 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_five_Y_three_equals_two_l2984_298463


namespace NUMINAMATH_CALUDE_one_sixths_in_eleven_thirds_l2984_298478

theorem one_sixths_in_eleven_thirds : (11 / 3) / (1 / 6) = 22 := by
  sorry

end NUMINAMATH_CALUDE_one_sixths_in_eleven_thirds_l2984_298478


namespace NUMINAMATH_CALUDE_flood_damage_conversion_l2984_298455

/-- Calculates the equivalent amount in USD given an amount in AUD and the exchange rate -/
def convert_aud_to_usd (amount_aud : ℝ) (exchange_rate : ℝ) : ℝ :=
  amount_aud * exchange_rate

/-- Theorem stating the conversion of flood damage from AUD to USD -/
theorem flood_damage_conversion :
  let damage_aud : ℝ := 45000000
  let exchange_rate : ℝ := 0.75
  convert_aud_to_usd damage_aud exchange_rate = 33750000 := by
  sorry

#check flood_damage_conversion

end NUMINAMATH_CALUDE_flood_damage_conversion_l2984_298455


namespace NUMINAMATH_CALUDE_car_speed_problem_l2984_298415

theorem car_speed_problem (V : ℝ) (x : ℝ) : 
  let V1 := V * (1 - x / 100)
  let V2 := V1 * (1 + 0.5 * x / 100)
  V2 = V * (1 - 0.6 * x / 100) →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2984_298415


namespace NUMINAMATH_CALUDE_total_mail_delivered_l2984_298448

/-- Represents the types of mail --/
inductive MailType
  | JunkMail
  | Magazine
  | Newspaper
  | Bill
  | Postcard

/-- Represents the mail distribution for a single house --/
structure HouseMailDistribution where
  junkMail : Nat
  magazines : Nat
  newspapers : Nat
  bills : Nat
  postcards : Nat

/-- Calculates the total pieces of mail for a single house --/
def totalMailForHouse (dist : HouseMailDistribution) : Nat :=
  dist.junkMail + dist.magazines + dist.newspapers + dist.bills + dist.postcards

/-- The mail distribution for the first house --/
def house1 : HouseMailDistribution :=
  { junkMail := 6, magazines := 5, newspapers := 3, bills := 4, postcards := 2 }

/-- The mail distribution for the second house --/
def house2 : HouseMailDistribution :=
  { junkMail := 4, magazines := 7, newspapers := 2, bills := 5, postcards := 3 }

/-- The mail distribution for the third house --/
def house3 : HouseMailDistribution :=
  { junkMail := 8, magazines := 3, newspapers := 4, bills := 6, postcards := 1 }

/-- Theorem stating that the total pieces of mail delivered to all three houses is 63 --/
theorem total_mail_delivered :
  totalMailForHouse house1 + totalMailForHouse house2 + totalMailForHouse house3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_mail_delivered_l2984_298448


namespace NUMINAMATH_CALUDE_sum_of_three_odd_squares_l2984_298496

theorem sum_of_three_odd_squares (a b c : ℕ) : 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →  -- pairwise different
  (∃ k l m : ℕ, a = 2*k + 1 ∧ b = 2*l + 1 ∧ c = 2*m + 1) →  -- odd integers
  (∃ x₁ x₂ x₃ x₄ x₅ x₆ : ℕ, a^2 + b^2 + c^2 = x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 + x₆^2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_odd_squares_l2984_298496


namespace NUMINAMATH_CALUDE_max_identical_bathrooms_l2984_298430

theorem max_identical_bathrooms (toilet_paper soap towels shower_gel shampoo toothpaste : ℕ) 
  (h1 : toilet_paper = 45)
  (h2 : soap = 30)
  (h3 : towels = 36)
  (h4 : shower_gel = 18)
  (h5 : shampoo = 27)
  (h6 : toothpaste = 24) :
  ∃ (max_bathrooms : ℕ), 
    max_bathrooms = 3 ∧ 
    (toilet_paper % max_bathrooms = 0) ∧
    (soap % max_bathrooms = 0) ∧
    (towels % max_bathrooms = 0) ∧
    (shower_gel % max_bathrooms = 0) ∧
    (shampoo % max_bathrooms = 0) ∧
    (toothpaste % max_bathrooms = 0) ∧
    ∀ (n : ℕ), n > max_bathrooms → 
      (toilet_paper % n ≠ 0) ∨
      (soap % n ≠ 0) ∨
      (towels % n ≠ 0) ∨
      (shower_gel % n ≠ 0) ∨
      (shampoo % n ≠ 0) ∨
      (toothpaste % n ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_max_identical_bathrooms_l2984_298430


namespace NUMINAMATH_CALUDE_triangle_formation_l2984_298454

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def can_form_triangle (sides : List ℝ) : Prop :=
  sides.length = 3 ∧ 
  let a := sides[0]!
  let b := sides[1]!
  let c := sides[2]!
  triangle_inequality a b c

theorem triangle_formation :
  ¬(can_form_triangle [1, 2, 4]) ∧
  ¬(can_form_triangle [2, 3, 6]) ∧
  ¬(can_form_triangle [12, 5, 6]) ∧
  can_form_triangle [8, 6, 4] :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l2984_298454


namespace NUMINAMATH_CALUDE_ellipse_chord_properties_l2984_298423

/-- Given an ellipse with equation x²/2 + y² = 1, this theorem proves various properties
    related to chords and their midpoints. -/
theorem ellipse_chord_properties :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/2 + y^2 = 1}
  let P := (1/2, 1/2)
  let A := (2, 1)
  ∀ (x y : ℝ), (x, y) ∈ ellipse →
    (∃ (m b : ℝ), 2*x + 4*y - 3 = 0 ∧ 
      ∀ (x' y' : ℝ), (x', y') ∈ ellipse → 
        (y' - P.2 = m*(x' - P.1) + b ↔ y - P.2 = m*(x - P.1) + b)) ∧
    (∃ (x₀ y₀ : ℝ), x₀ + 4*y₀ = 0 ∧ -Real.sqrt 2 < x₀ ∧ x₀ < Real.sqrt 2 ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ ellipse ∧ (x₂, y₂) ∈ ellipse ∧
        (y₂ - y₁)/(x₂ - x₁) = 2 ∧ x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2) ∧
    (∃ (x₀ y₀ : ℝ), x₀^2 - 2*x₀ + 2*y₀^2 - 2*y₀ = 0 ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ ellipse ∧ (x₂, y₂) ∈ ellipse ∧
        (y₁ - A.2)/(x₁ - A.1) = (y₂ - A.2)/(x₂ - A.1) ∧
        x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2) ∧
    (∃ (x₀ y₀ : ℝ), x₀^2 + 2*y₀^2 = 1 ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ ellipse ∧ (x₂, y₂) ∈ ellipse ∧
        (y₁/x₁) * (y₂/x₂) = -1/2 ∧ x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_properties_l2984_298423


namespace NUMINAMATH_CALUDE_speed_conversion_l2984_298410

/-- Conversion factor from km/h to m/s -/
def km_h_to_m_s : ℝ := 0.27777777777778

/-- Given speed in km/h -/
def speed_km_h : ℝ := 0.8666666666666666

/-- Calculated speed in m/s -/
def speed_m_s : ℝ := 0.24074074074074

theorem speed_conversion : speed_km_h * km_h_to_m_s = speed_m_s := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l2984_298410


namespace NUMINAMATH_CALUDE_pizza_slices_theorem_l2984_298469

/-- Represents the number of slices in different pizza sizes -/
structure PizzaSlices where
  small : ℕ
  medium : ℕ
  large : ℕ
  extraLarge : ℕ

/-- Represents the ratio of different pizza sizes ordered -/
structure PizzaRatio where
  small : ℕ
  medium : ℕ
  large : ℕ
  extraLarge : ℕ

/-- Calculates the total number of slices from all pizzas -/
def totalSlices (slices : PizzaSlices) (ratio : PizzaRatio) (totalPizzas : ℕ) : ℕ :=
  let ratioSum := ratio.small + ratio.medium + ratio.large + ratio.extraLarge
  let pizzasPerRatio := totalPizzas / ratioSum
  (slices.small * ratio.small * pizzasPerRatio) +
  (slices.medium * ratio.medium * pizzasPerRatio) +
  (slices.large * ratio.large * pizzasPerRatio) +
  (slices.extraLarge * ratio.extraLarge * pizzasPerRatio)

theorem pizza_slices_theorem (slices : PizzaSlices) (ratio : PizzaRatio) (totalPizzas : ℕ) :
  slices.small = 6 →
  slices.medium = 8 →
  slices.large = 12 →
  slices.extraLarge = 16 →
  ratio.small = 3 →
  ratio.medium = 2 →
  ratio.large = 4 →
  ratio.extraLarge = 1 →
  totalPizzas = 20 →
  totalSlices slices ratio totalPizzas = 196 := by
  sorry

#eval totalSlices ⟨6, 8, 12, 16⟩ ⟨3, 2, 4, 1⟩ 20

end NUMINAMATH_CALUDE_pizza_slices_theorem_l2984_298469


namespace NUMINAMATH_CALUDE_min_value_theorem_l2984_298418

theorem min_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 2) :
  (1/3 : ℝ) * x^3 + y^2 + z ≥ 13/12 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2984_298418


namespace NUMINAMATH_CALUDE_donation_ratio_l2984_298406

theorem donation_ratio : 
  ∀ (total parents teachers students : ℝ),
  parents = 0.25 * total →
  teachers + students = 0.75 * total →
  teachers = (2/5) * (teachers + students) →
  students = (3/5) * (teachers + students) →
  parents / students = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_donation_ratio_l2984_298406


namespace NUMINAMATH_CALUDE_residue_of_neg1237_mod37_l2984_298487

theorem residue_of_neg1237_mod37 : ∃ (k : ℤ), -1237 = 37 * k + 21 ∧ (0 ≤ 21 ∧ 21 < 37) := by
  sorry

end NUMINAMATH_CALUDE_residue_of_neg1237_mod37_l2984_298487


namespace NUMINAMATH_CALUDE_seeds_per_pack_l2984_298475

def desired_flowers : ℕ := 20
def survival_rate : ℚ := 1/2
def pack_cost : ℕ := 5
def total_spent : ℕ := 10

theorem seeds_per_pack : 
  ∃ (seeds_per_pack : ℕ), 
    (total_spent / pack_cost) * seeds_per_pack = desired_flowers / survival_rate :=
by sorry

end NUMINAMATH_CALUDE_seeds_per_pack_l2984_298475


namespace NUMINAMATH_CALUDE_channels_taken_away_proof_l2984_298432

/-- Calculates the number of channels initially taken away --/
def channels_taken_away (initial_channels : ℕ) 
  (replaced_channels : ℕ) (reduced_channels : ℕ) 
  (sports_package : ℕ) (supreme_sports : ℕ) (final_channels : ℕ) : ℕ :=
  initial_channels + replaced_channels - reduced_channels + sports_package + supreme_sports - final_channels

/-- Proves that 20 channels were initially taken away --/
theorem channels_taken_away_proof : 
  channels_taken_away 150 12 10 8 7 147 = 20 := by sorry

end NUMINAMATH_CALUDE_channels_taken_away_proof_l2984_298432


namespace NUMINAMATH_CALUDE_rod_weight_10m_l2984_298470

/-- Represents the weight of a rod given its length -/
def rod_weight (length : ℝ) : ℝ := sorry

/-- The constant of proportionality between rod length and weight -/
def weight_per_meter : ℝ := sorry

theorem rod_weight_10m (h1 : rod_weight 6 = 14.04) 
  (h2 : ∀ l : ℝ, rod_weight l = weight_per_meter * l) : 
  rod_weight 10 = 23.4 := by sorry

end NUMINAMATH_CALUDE_rod_weight_10m_l2984_298470


namespace NUMINAMATH_CALUDE_solve_equations_l2984_298497

theorem solve_equations :
  (∃ x : ℝ, 5 * x - 2.9 = 12) ∧
  (∃ x : ℝ, 10.5 * x + 0.6 * x = 44) ∧
  (∃ x : ℝ, 8 * x / 2 = 1.5) :=
by
  constructor
  · use 1.82
    sorry
  constructor
  · use 3
    sorry
  · use 0.375
    sorry

end NUMINAMATH_CALUDE_solve_equations_l2984_298497


namespace NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_lowest_terms_l2984_298412

/-- The repeating decimal 0.4̄37 as a real number -/
def repeating_decimal : ℚ := 433 / 990

theorem repeating_decimal_equiv : repeating_decimal = 0.4 + (37 / 990) := by sorry

theorem fraction_lowest_terms : ∀ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 → (433 * a = 990 * b) → (a = 990 ∧ b = 433) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_lowest_terms_l2984_298412


namespace NUMINAMATH_CALUDE_tourist_attraction_arrangements_l2984_298431

def total_attractions : ℕ := 10
def daytime_attractions : ℕ := 8
def nighttime_attractions : ℕ := 2
def selected_attractions : ℕ := 5
def day1_slots : ℕ := 3
def day2_slots : ℕ := 2

theorem tourist_attraction_arrangements :
  (∃ (arrangements_with_A_or_B : ℕ) 
      (arrangements_A_and_B_same_day : ℕ) 
      (arrangements_without_A_and_B_together : ℕ),
    arrangements_with_A_or_B = 2352 ∧
    arrangements_A_and_B_same_day = 28560 ∧
    arrangements_without_A_and_B_together = 2352) := by
  sorry

end NUMINAMATH_CALUDE_tourist_attraction_arrangements_l2984_298431


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2984_298443

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 2x^2 + (2 + 1/2)x + 1/2 has discriminant 2.25 -/
theorem quadratic_discriminant : discriminant 2 (2 + 1/2) (1/2) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2984_298443


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l2984_298468

/-- Given two rectangles of equal area, where one rectangle measures 15 inches by 20 inches
    and the other rectangle is 6 inches long, prove that the width of the second rectangle
    is 50 inches. -/
theorem equal_area_rectangles_width (carol_length carol_width jordan_length jordan_width : ℝ)
  (h1 : carol_length = 15)
  (h2 : carol_width = 20)
  (h3 : jordan_length = 6)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 50 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l2984_298468


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l2984_298498

theorem quadratic_function_m_range (a c m : ℝ) :
  let f := fun x : ℝ => a * x^2 - 2 * a * x + c
  (∀ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, x < y → f x > f y) →
  f m ≤ f 0 →
  m ∈ Set.Icc 0 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l2984_298498


namespace NUMINAMATH_CALUDE_shortest_median_le_longest_angle_bisector_l2984_298450

/-- Represents a triangle with side lengths a ≤ b ≤ c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hab : a ≤ b
  hbc : b ≤ c

/-- The length of the shortest median in a triangle -/
def shortestMedian (t : Triangle) : ℝ := sorry

/-- The length of the longest angle bisector in a triangle -/
def longestAngleBisector (t : Triangle) : ℝ := sorry

/-- Theorem: The shortest median is always shorter than or equal to the longest angle bisector -/
theorem shortest_median_le_longest_angle_bisector (t : Triangle) :
  shortestMedian t ≤ longestAngleBisector t := by sorry

end NUMINAMATH_CALUDE_shortest_median_le_longest_angle_bisector_l2984_298450


namespace NUMINAMATH_CALUDE_square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l2984_298441

theorem square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three :
  ∀ x : ℝ, x = Real.sqrt 2 + 1 → x^2 - 2*x + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l2984_298441


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2984_298479

/-- Represents a geometric sequence -/
structure GeometricSequence (α : Type*) [Ring α] where
  a : ℕ → α
  r : α
  h : ∀ n, a (n + 1) = r * a n

/-- Sum of the first n terms of a geometric sequence -/
def sum_n {α : Type*} [Ring α] (seq : GeometricSequence α) (n : ℕ) : α :=
  sorry

/-- The main theorem stating that for any geometric sequence, 
    a_{2016}(S_{2016}-S_{2015}) ≠ 0 -/
theorem geometric_sequence_property {α : Type*} [Field α] (seq : GeometricSequence α) :
  seq.a 2016 * (sum_n seq 2016 - sum_n seq 2015) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2984_298479


namespace NUMINAMATH_CALUDE_same_terminal_side_diff_multiple_360_l2984_298466

/-- Two angles have the same terminal side -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

/-- Theorem: If two angles have the same terminal side, their difference is a multiple of 360° -/
theorem same_terminal_side_diff_multiple_360 (α β : ℝ) :
  same_terminal_side α β → ∃ k : ℤ, α - β = k * 360 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_diff_multiple_360_l2984_298466


namespace NUMINAMATH_CALUDE_no_opposite_divisibility_l2984_298425

theorem no_opposite_divisibility (k n a : ℕ) : 
  k ≥ 3 → n ≥ 3 → Odd k → Odd n → a ≥ 1 → 
  k ∣ (2^a + 1) → n ∣ (2^a - 1) → 
  ¬∃ b : ℕ, b ≥ 1 ∧ k ∣ (2^b - 1) ∧ n ∣ (2^b + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_opposite_divisibility_l2984_298425


namespace NUMINAMATH_CALUDE_complex_number_problem_l2984_298439

open Complex

theorem complex_number_problem (z : ℂ) (a b : ℝ) : 
  z = ((1 + I)^2 + 2*(5 - I)) / (3 + I) →
  abs z = Real.sqrt 10 ∧
  (z * (z + a) = b + I → a = -7 ∧ b = -13) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2984_298439


namespace NUMINAMATH_CALUDE_five_hour_charge_l2984_298403

/-- Represents the pricing structure and total charge calculation for a psychologist's therapy sessions. -/
structure TherapyPricing where
  /-- The charge for the first hour of therapy -/
  first_hour : ℕ
  /-- The charge for each additional hour of therapy -/
  additional_hour : ℕ
  /-- The first hour costs $30 more than each additional hour -/
  first_hour_premium : first_hour = additional_hour + 30
  /-- The total charge for 3 hours of therapy is $252 -/
  three_hour_charge : first_hour + 2 * additional_hour = 252

/-- Theorem stating that given the pricing structure, the total charge for 5 hours of therapy is $400 -/
theorem five_hour_charge (p : TherapyPricing) : p.first_hour + 4 * p.additional_hour = 400 := by
  sorry

end NUMINAMATH_CALUDE_five_hour_charge_l2984_298403


namespace NUMINAMATH_CALUDE_trig_range_equivalence_l2984_298402

theorem trig_range_equivalence (α : Real) :
  (0 < α ∧ α < 2 * Real.pi) →
  (Real.sin α < Real.sqrt 3 / 2 ∧ Real.cos α > 1 / 2) ↔
  ((0 < α ∧ α < Real.pi / 3) ∨ (5 * Real.pi / 3 < α ∧ α < 2 * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_trig_range_equivalence_l2984_298402


namespace NUMINAMATH_CALUDE_woman_work_days_l2984_298484

/-- A woman's work and pay scenario -/
theorem woman_work_days (total_days : ℕ) (pay_per_day : ℕ) (forfeit_per_day : ℕ) (net_earnings : ℕ) 
    (h1 : total_days = 25)
    (h2 : pay_per_day = 20)
    (h3 : forfeit_per_day = 5)
    (h4 : net_earnings = 450) :
  ∃ (work_days : ℕ), 
    work_days ≤ total_days ∧ 
    (pay_per_day * work_days - forfeit_per_day * (total_days - work_days) = net_earnings) ∧
    work_days = 23 := by
  sorry

end NUMINAMATH_CALUDE_woman_work_days_l2984_298484


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2984_298449

/-- The weight of the replaced person in a group of 6 people -/
def weight_of_replaced_person (initial_count : ℕ) (new_person_weight : ℝ) (average_increase : ℝ) : ℝ :=
  new_person_weight - initial_count * average_increase

/-- Theorem stating the weight of the replaced person -/
theorem replaced_person_weight :
  weight_of_replaced_person 6 68 3.5 = 47 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2984_298449


namespace NUMINAMATH_CALUDE_sin_increasing_on_interval_l2984_298456

-- Define the sine function (already defined in Mathlib)
-- def sin : ℝ → ℝ := Real.sin

-- Define the interval
def interval : Set ℝ := Set.Icc (-Real.pi / 2) (Real.pi / 2)

-- State the theorem
theorem sin_increasing_on_interval :
  StrictMonoOn Real.sin interval :=
sorry

end NUMINAMATH_CALUDE_sin_increasing_on_interval_l2984_298456


namespace NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l2984_298459

/-- Represents the ratio of boarders to day students -/
structure Ratio where
  boarders : ℕ
  day_students : ℕ

/-- Represents the school's student composition -/
structure School where
  initial_boarders : ℕ
  initial_ratio : Ratio
  new_boarders : ℕ

/-- Calculates the new ratio of boarders to day students after new boarders join -/
def new_ratio (school : School) : Ratio :=
  sorry

/-- Theorem stating that the new ratio is 1:2 given the initial conditions -/
theorem new_ratio_is_one_to_two (school : School) 
  (h1 : school.initial_boarders = 120)
  (h2 : school.initial_ratio = Ratio.mk 2 5)
  (h3 : school.new_boarders = 30) :
  new_ratio school = Ratio.mk 1 2 :=
sorry

end NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l2984_298459


namespace NUMINAMATH_CALUDE_range_of_f_l2984_298416

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2984_298416
