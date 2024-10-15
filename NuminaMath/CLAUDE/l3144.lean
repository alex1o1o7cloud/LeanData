import Mathlib

namespace NUMINAMATH_CALUDE_max_visibility_score_l3144_314409

/-- Represents a configuration of towers --/
structure TowerConfig where
  height1 : ℕ  -- Number of towers with height 1
  height2 : ℕ  -- Number of towers with height 2

/-- The total height of all towers is 30 --/
def validConfig (config : TowerConfig) : Prop :=
  config.height1 + 2 * config.height2 = 30

/-- Calculate the visibility score for a given configuration --/
def visibilityScore (config : TowerConfig) : ℕ :=
  config.height1 * config.height2

/-- Theorem: The maximum visibility score is 112 and is achieved
    when all towers are either height 1 or 2 --/
theorem max_visibility_score :
  ∃ (config : TowerConfig), validConfig config ∧
    visibilityScore config = 112 ∧
    (∀ (other : TowerConfig), validConfig other →
      visibilityScore other ≤ visibilityScore config) :=
by sorry

end NUMINAMATH_CALUDE_max_visibility_score_l3144_314409


namespace NUMINAMATH_CALUDE_factor_implies_p_value_l3144_314462

theorem factor_implies_p_value (p : ℚ) : 
  (∀ x : ℚ, (3 * x + 4 = 0) → (4 * x^3 + p * x^2 + 17 * x + 24 = 0)) → 
  p = 13/4 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_p_value_l3144_314462


namespace NUMINAMATH_CALUDE_complex_equation_proof_l3144_314455

theorem complex_equation_proof (z : ℂ) (h : z = 1 + I) : z^2 - 2*z + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l3144_314455


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3144_314498

/-- The line l with equation x - ky - 1 = 0 intersects the circle C with equation x^2 + y^2 = 2 for any real k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), (x - k*y - 1 = 0) ∧ (x^2 + y^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3144_314498


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l3144_314463

theorem quadratic_solution_range (a b c : ℝ) (h_a : a ≠ 0) :
  let f := fun x => a * x^2 + b * x + c
  (f 3.24 = -0.02) → (f 3.25 = 0.01) → (f 3.26 = 0.03) →
  ∃ x, f x = 0 ∧ 3.24 < x ∧ x < 3.25 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l3144_314463


namespace NUMINAMATH_CALUDE_sqrt_sum_quotient_l3144_314466

theorem sqrt_sum_quotient : (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 48 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_quotient_l3144_314466


namespace NUMINAMATH_CALUDE_arrangement_count_l3144_314488

-- Define the total number of people
def total_people : ℕ := 7

-- Define the number of people to be selected
def selected_people : ℕ := 5

-- Define a function to calculate the number of arrangements
def arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

-- Theorem statement
theorem arrangement_count :
  arrangements total_people selected_people = 600 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l3144_314488


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l3144_314412

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the symmetric point relation
def symmetric_point (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- Midpoint of (x₁, y₁) and (x₂, y₂) lies on the line of symmetry
  line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) ∧
  -- The line connecting (x₁, y₁) and (x₂, y₂) is perpendicular to the line of symmetry
  (y₂ - y₁) = (x₂ - x₁)

-- Theorem statement
theorem symmetric_point_theorem :
  symmetric_point 2 1 (-2) (-3) :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l3144_314412


namespace NUMINAMATH_CALUDE_smaller_sphere_radius_l3144_314499

/-- The radius of a smaller sphere when a sphere of radius R is cast into two smaller spheres -/
theorem smaller_sphere_radius (R : ℝ) (R_pos : R > 0) : ℝ :=
  let smaller_radius := R / 3
  let larger_radius := 2 * smaller_radius
  have volume_conservation : (4 / 3) * Real.pi * R^3 = (4 / 3) * Real.pi * smaller_radius^3 + (4 / 3) * Real.pi * larger_radius^3 := by sorry
  have radius_ratio : larger_radius = 2 * smaller_radius := by sorry
  smaller_radius

#check smaller_sphere_radius

end NUMINAMATH_CALUDE_smaller_sphere_radius_l3144_314499


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3144_314421

theorem sufficient_not_necessary_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, a > 0 → b > 0 → a > b → a^3 + b^3 > a^2*b + a*b^2) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a^3 + b^3 > a^2*b + a*b^2 ∧ a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3144_314421


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_72_l3144_314461

theorem sqrt_nine_factorial_over_72 : Real.sqrt (Nat.factorial 9 / 72) = 12 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_72_l3144_314461


namespace NUMINAMATH_CALUDE_group_shopping_popularity_justified_l3144_314435

/-- Represents the practice of group shopping -/
structure GroupShopping where
  risks : ℕ  -- Number of risks associated with group shopping
  countries : ℕ  -- Number of countries where group shopping is practiced

/-- Factors contributing to group shopping popularity -/
structure PopularityFactors where
  cost_savings : ℝ  -- Percentage of cost savings
  quality_assessment : ℝ  -- Measure of quality assessment improvement
  trust_dynamics : ℝ  -- Measure of trust within the community

/-- Theorem stating that group shopping popularity is justified -/
theorem group_shopping_popularity_justified 
  (gs : GroupShopping) 
  (pf : PopularityFactors) : 
  gs.risks > 0 → 
  gs.countries > 10 → 
  pf.cost_savings > 20 → 
  pf.quality_assessment > 0.5 → 
  pf.trust_dynamics > 0.7 → 
  True := by
  sorry


end NUMINAMATH_CALUDE_group_shopping_popularity_justified_l3144_314435


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3144_314444

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 8) : 
  ∀ z, z = x + 2*y → z ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3144_314444


namespace NUMINAMATH_CALUDE_bisected_parallelogram_perimeter_l3144_314443

/-- Represents a parallelogram with a bisected angle -/
structure BisectedParallelogram where
  -- The length of one segment created by the angle bisector
  segment1 : ℝ
  -- The length of the other segment created by the angle bisector
  segment2 : ℝ
  -- Assumption that the segments are 7 and 14 (in either order)
  h_segments : (segment1 = 7 ∧ segment2 = 14) ∨ (segment1 = 14 ∧ segment2 = 7)

/-- The perimeter of the parallelogram is either 56 or 70 -/
theorem bisected_parallelogram_perimeter (p : BisectedParallelogram) :
  let perimeter := 2 * (p.segment1 + p.segment2)
  perimeter = 56 ∨ perimeter = 70 := by
  sorry


end NUMINAMATH_CALUDE_bisected_parallelogram_perimeter_l3144_314443


namespace NUMINAMATH_CALUDE_opposite_expressions_solution_l3144_314453

theorem opposite_expressions_solution (x : ℚ) : (8*x - 7 = -(6 - 2*x)) → x = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_expressions_solution_l3144_314453


namespace NUMINAMATH_CALUDE_average_of_dataset_l3144_314418

def dataset : List ℝ := [5, 9, 9, 3, 4]

theorem average_of_dataset : 
  (dataset.sum / dataset.length : ℝ) = 6 := by sorry

end NUMINAMATH_CALUDE_average_of_dataset_l3144_314418


namespace NUMINAMATH_CALUDE_optimal_chair_removal_l3144_314497

def chairs_per_row : ℕ := 13
def initial_chairs : ℕ := 169
def expected_attendees : ℕ := 95
def max_removable_chairs : ℕ := 26

theorem optimal_chair_removal :
  ∀ n : ℕ,
  n ≤ max_removable_chairs →
  (initial_chairs - n) % chairs_per_row = 0 →
  initial_chairs - max_removable_chairs ≤ initial_chairs - n →
  (initial_chairs - n) - expected_attendees ≥
    (initial_chairs - max_removable_chairs) - expected_attendees :=
by sorry

end NUMINAMATH_CALUDE_optimal_chair_removal_l3144_314497


namespace NUMINAMATH_CALUDE_stadium_entrance_exit_ways_l3144_314400

/-- The number of gates on the south side of the stadium -/
def south_gates : ℕ := 4

/-- The number of gates on the north side of the stadium -/
def north_gates : ℕ := 3

/-- The total number of gates in the stadium -/
def total_gates : ℕ := south_gates + north_gates

/-- The number of different ways to enter and exit the stadium -/
def entrance_exit_ways : ℕ := total_gates * total_gates

theorem stadium_entrance_exit_ways :
  entrance_exit_ways = 49 := by sorry

end NUMINAMATH_CALUDE_stadium_entrance_exit_ways_l3144_314400


namespace NUMINAMATH_CALUDE_village_distance_l3144_314468

def round_trip_time : ℝ := 4
def uphill_speed : ℝ := 15
def downhill_speed : ℝ := 30

theorem village_distance (d : ℝ) (h : d > 0) :
  d / uphill_speed + d / downhill_speed = round_trip_time →
  d = 40 := by
sorry

end NUMINAMATH_CALUDE_village_distance_l3144_314468


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l3144_314442

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the upstream speed given the rowing speeds in still water and downstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the specific conditions, the upstream speed is 15 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 25) 
  (h2 : s.downstream = 35) : 
  upstreamSpeed s = 15 := by
  sorry

#eval upstreamSpeed { stillWater := 25, downstream := 35 }

end NUMINAMATH_CALUDE_upstream_speed_calculation_l3144_314442


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l3144_314469

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 6 * (8 * x^2 + 7 * x + 11) - x * (8 * x - 45)
  ∃ x : ℝ, (f x = 0) ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ (x = -11/8) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l3144_314469


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3144_314404

/-- Represents a repeating decimal with a single digit repeating infinitely. -/
def RepeatingDecimal (n : Nat) : ℚ := n / 9

theorem sum_of_repeating_decimals :
  RepeatingDecimal 6 + RepeatingDecimal 4 = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3144_314404


namespace NUMINAMATH_CALUDE_age_difference_l3144_314452

theorem age_difference (son_age father_age : ℕ) 
  (h1 : son_age = 9)
  (h2 : father_age = 36) :
  father_age - son_age = 27 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3144_314452


namespace NUMINAMATH_CALUDE_kendras_cookies_l3144_314473

/-- Kendra's cookie baking problem -/
theorem kendras_cookies :
  ∀ (cookies_per_batch : ℕ)
    (family_size : ℕ)
    (chips_per_cookie : ℕ)
    (chips_per_person : ℕ),
  cookies_per_batch = 12 →
  family_size = 4 →
  chips_per_cookie = 2 →
  chips_per_person = 18 →
  (chips_per_person / chips_per_cookie * family_size) / cookies_per_batch = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_kendras_cookies_l3144_314473


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l3144_314447

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x - 1| - a

-- Theorem 1
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f a x - 2 * |x - 7| ≤ 0) → a ≥ -12 := by sorry

-- Theorem 2
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f 1 x + |x + 7| ≥ m) → m ≤ 7 := by sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l3144_314447


namespace NUMINAMATH_CALUDE_expression_value_l3144_314493

theorem expression_value (x y z : ℝ) (hx : x = 1) (hy : y = 1) (hz : z = 3) :
  x^2 * y * z - x * y * z^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3144_314493


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3144_314411

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 > 0 →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3144_314411


namespace NUMINAMATH_CALUDE_count_even_perfect_square_factors_l3144_314446

/-- The number of even perfect square factors of 2^6 * 7^3 * 3^8 -/
def evenPerfectSquareFactors : ℕ := 30

/-- The original number -/
def originalNumber : ℕ := 2^6 * 7^3 * 3^8

/-- A function that counts the number of even perfect square factors of originalNumber -/
def countEvenPerfectSquareFactors : ℕ := sorry

theorem count_even_perfect_square_factors :
  countEvenPerfectSquareFactors = evenPerfectSquareFactors := by sorry

end NUMINAMATH_CALUDE_count_even_perfect_square_factors_l3144_314446


namespace NUMINAMATH_CALUDE_square_difference_l3144_314470

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) :
  (x - y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3144_314470


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3144_314494

theorem regular_polygon_sides (central_angle : ℝ) : 
  central_angle = 40 → (360 : ℝ) / central_angle = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3144_314494


namespace NUMINAMATH_CALUDE_frame_width_is_five_l3144_314402

/-- A rectangular frame containing three square photograph openings with uniform width. -/
structure PhotoFrame where
  /-- The side length of each square opening -/
  opening_side : ℝ
  /-- The width of the frame -/
  frame_width : ℝ

/-- The perimeter of one square opening -/
def opening_perimeter (frame : PhotoFrame) : ℝ :=
  4 * frame.opening_side

/-- The perimeter of the entire rectangular frame -/
def frame_perimeter (frame : PhotoFrame) : ℝ :=
  2 * (frame.opening_side + 2 * frame.frame_width) + 2 * (3 * frame.opening_side + 2 * frame.frame_width)

/-- Theorem stating that if the perimeter of one opening is 60 cm and the perimeter of the entire frame is 180 cm, 
    then the width of the frame is 5 cm -/
theorem frame_width_is_five (frame : PhotoFrame) 
  (h1 : opening_perimeter frame = 60) 
  (h2 : frame_perimeter frame = 180) : 
  frame.frame_width = 5 := by
  sorry

end NUMINAMATH_CALUDE_frame_width_is_five_l3144_314402


namespace NUMINAMATH_CALUDE_isosceles_triangle_relationship_l3144_314433

-- Define the isosceles triangle
structure IsoscelesTriangle where
  x : ℝ  -- leg length
  y : ℝ  -- base length

-- Define the properties of the isosceles triangle
def validIsoscelesTriangle (t : IsoscelesTriangle) : Prop :=
  t.x > 0 ∧ t.y > 0 ∧ 2 * t.x > t.y ∧ t.x + t.y > t.x

-- Define the perimeter constraint
def hasPerimeter30 (t : IsoscelesTriangle) : Prop :=
  2 * t.x + t.y = 30

-- Define the relationship between x and y
def relationshipXY (t : IsoscelesTriangle) : Prop :=
  t.y = 30 - 2 * t.x

-- Define the constraints on x
def xConstraints (t : IsoscelesTriangle) : Prop :=
  15 / 2 < t.x ∧ t.x < 15

-- Theorem stating the relationship between x and y for the isosceles triangle
theorem isosceles_triangle_relationship (t : IsoscelesTriangle) :
  validIsoscelesTriangle t → hasPerimeter30 t → relationshipXY t ∧ xConstraints t :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_relationship_l3144_314433


namespace NUMINAMATH_CALUDE_alices_favorite_number_l3144_314440

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem alices_favorite_number :
  ∃! n : ℕ,
    100 < n ∧ n < 200 ∧
    n % 13 = 0 ∧
    n % 3 ≠ 0 ∧
    (sum_of_digits n) % 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_alices_favorite_number_l3144_314440


namespace NUMINAMATH_CALUDE_skyscraper_anniversary_l3144_314457

theorem skyscraper_anniversary (current_year : ℕ) : 
  let years_since_built := 100
  let years_to_event := 95
  let event_year := current_year + years_to_event
  let years_at_event := years_since_built + years_to_event
  ∃ (anniversary : ℕ), anniversary > years_at_event ∧ anniversary - years_at_event = 5 :=
by sorry

end NUMINAMATH_CALUDE_skyscraper_anniversary_l3144_314457


namespace NUMINAMATH_CALUDE_symmetric_points_m_value_l3144_314430

/-- Two points are symmetric about the origin if their coordinates are negatives of each other -/
def symmetric_about_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

/-- Given that point A(2, -1) is symmetric with point B(-2, m) about the origin, prove that m = 1 -/
theorem symmetric_points_m_value :
  let A : ℝ × ℝ := (2, -1)
  let B : ℝ × ℝ := (-2, m)
  symmetric_about_origin A B → m = 1 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_m_value_l3144_314430


namespace NUMINAMATH_CALUDE_mothers_carrots_count_l3144_314422

/-- The number of carrots Nancy picked -/
def nancys_carrots : ℕ := 38

/-- The number of good carrots -/
def good_carrots : ℕ := 71

/-- The number of bad carrots -/
def bad_carrots : ℕ := 14

/-- The number of carrots Nancy's mother picked -/
def mothers_carrots : ℕ := (good_carrots + bad_carrots) - nancys_carrots

theorem mothers_carrots_count : mothers_carrots = 47 := by
  sorry

end NUMINAMATH_CALUDE_mothers_carrots_count_l3144_314422


namespace NUMINAMATH_CALUDE_smallest_packages_for_more_envelopes_l3144_314486

theorem smallest_packages_for_more_envelopes (n : ℕ) : 
  (∀ k : ℕ, k < n → 10 * k ≤ 8 * k + 7) ∧ 
  (10 * n > 8 * n + 7) → 
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_packages_for_more_envelopes_l3144_314486


namespace NUMINAMATH_CALUDE_min_value_x_plus_two_over_x_l3144_314423

theorem min_value_x_plus_two_over_x (x : ℝ) (h : x > 0) :
  x + 2 / x ≥ 2 * Real.sqrt 2 ∧
  (x + 2 / x = 2 * Real.sqrt 2 ↔ x = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_two_over_x_l3144_314423


namespace NUMINAMATH_CALUDE_benny_candy_bars_l3144_314456

/-- The number of candy bars Benny bought -/
def candy_bars : ℕ := 5

/-- The cost of each soft drink -/
def soft_drink_cost : ℕ := 4

/-- The number of soft drinks Benny bought -/
def soft_drinks : ℕ := 2

/-- The total amount Benny spent -/
def total_spent : ℕ := 28

/-- The cost of each candy bar -/
def candy_bar_cost : ℕ := 4

theorem benny_candy_bars : 
  candy_bars * candy_bar_cost + soft_drinks * soft_drink_cost = total_spent := by
  sorry

end NUMINAMATH_CALUDE_benny_candy_bars_l3144_314456


namespace NUMINAMATH_CALUDE_unique_number_existence_l3144_314487

theorem unique_number_existence : ∃! N : ℕ, N / 1000 = 220 ∧ N % 1000 = 40 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_existence_l3144_314487


namespace NUMINAMATH_CALUDE_selection_methods_count_l3144_314432

/-- The number of teachers in each department -/
def teachers_per_dept : ℕ := 4

/-- The total number of departments -/
def total_depts : ℕ := 4

/-- The number of leaders to be selected -/
def leaders_to_select : ℕ := 3

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to select leaders satisfying the given conditions -/
def selection_methods : ℕ :=
  -- One from admin, two from same other dept
  choose teachers_per_dept 1 * choose (total_depts - 1) 1 * choose teachers_per_dept 2 +
  -- One from admin, two from different other depts
  choose teachers_per_dept 1 * choose (total_depts - 1) 2 * choose teachers_per_dept 1 * choose teachers_per_dept 1 +
  -- Two from admin, one from any other dept
  choose teachers_per_dept 2 * choose (total_depts - 1) 1 * choose teachers_per_dept 1

theorem selection_methods_count :
  selection_methods = 336 :=
by sorry

end NUMINAMATH_CALUDE_selection_methods_count_l3144_314432


namespace NUMINAMATH_CALUDE_square_plus_linear_equals_square_l3144_314481

theorem square_plus_linear_equals_square (x y : ℕ+) 
  (h : x^2 + 84*x + 2016 = y^2) : 
  x^3 + y^2 = 12096 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_linear_equals_square_l3144_314481


namespace NUMINAMATH_CALUDE_rent_increase_group_size_l3144_314438

theorem rent_increase_group_size :
  ∀ (n : ℕ) (initial_average rent_increase new_average original_rent : ℚ),
    initial_average = 800 →
    new_average = 880 →
    original_rent = 1600 →
    rent_increase = 0.2 * original_rent →
    n * new_average = n * initial_average + rent_increase →
    n = 4 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_group_size_l3144_314438


namespace NUMINAMATH_CALUDE_ball_contact_height_l3144_314434

theorem ball_contact_height (horizontal_distance : ℝ) (hypotenuse : ℝ) (height : ℝ) : 
  horizontal_distance = 7 → hypotenuse = 53 → height ^ 2 + horizontal_distance ^ 2 = hypotenuse ^ 2 → height = 2 := by
  sorry

end NUMINAMATH_CALUDE_ball_contact_height_l3144_314434


namespace NUMINAMATH_CALUDE_cubic_eq_given_quadratic_l3144_314420

theorem cubic_eq_given_quadratic (x : ℝ) :
  x^2 + 5*x - 990 = 0 → x^3 + 6*x^2 - 985*x + 1012 = 2002 := by
  sorry

end NUMINAMATH_CALUDE_cubic_eq_given_quadratic_l3144_314420


namespace NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l3144_314436

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Main theorem: If a line is perpendicular to two different planes, then the planes are parallel -/
theorem line_perp_two_planes_implies_parallel (l : Line3D) (α β : Plane3D) :
  α ≠ β → perpendicular l α → perpendicular l β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l3144_314436


namespace NUMINAMATH_CALUDE_milk_per_milkshake_l3144_314475

/-- The amount of milk needed for each milkshake, given:
  * Blake has 72 ounces of milk initially
  * Blake has 192 ounces of ice cream
  * Each milkshake needs 12 ounces of ice cream
  * After making milkshakes, Blake has 8 ounces of milk left
-/
theorem milk_per_milkshake (initial_milk : ℕ) (ice_cream : ℕ) (ice_cream_per_shake : ℕ) (milk_left : ℕ)
  (h1 : initial_milk = 72)
  (h2 : ice_cream = 192)
  (h3 : ice_cream_per_shake = 12)
  (h4 : milk_left = 8) :
  (initial_milk - milk_left) / (ice_cream / ice_cream_per_shake) = 4 := by
  sorry

end NUMINAMATH_CALUDE_milk_per_milkshake_l3144_314475


namespace NUMINAMATH_CALUDE_bee_hive_problem_l3144_314407

theorem bee_hive_problem (B : ℕ) : 
  (B / 5 : ℚ) + (B / 3 : ℚ) + (3 * ((B / 3 : ℚ) - (B / 5 : ℚ))) + 1 = B → B = 15 := by
  sorry

end NUMINAMATH_CALUDE_bee_hive_problem_l3144_314407


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l3144_314460

/-- Given a survey of pasta preferences among students, this theorem proves
    that the ratio of students preferring spaghetti to those preferring manicotti is 2. -/
theorem pasta_preference_ratio :
  let total_students : ℕ := 800
  let spaghetti_preference : ℕ := 320
  let manicotti_preference : ℕ := 160
  (spaghetti_preference : ℚ) / manicotti_preference = 2 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l3144_314460


namespace NUMINAMATH_CALUDE_jacob_test_score_l3144_314450

theorem jacob_test_score (x : ℝ) : 
  (x + 79 + 92 + 84 + 85) / 5 = 85 → x = 85 := by
sorry

end NUMINAMATH_CALUDE_jacob_test_score_l3144_314450


namespace NUMINAMATH_CALUDE_geometric_increasing_condition_l3144_314413

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The condition "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem geometric_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  ¬(((q > 1) → increasing_sequence a) ∧ (increasing_sequence a → (q > 1))) :=
sorry

end NUMINAMATH_CALUDE_geometric_increasing_condition_l3144_314413


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3144_314454

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = 3 * x - 10) → x = 25 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3144_314454


namespace NUMINAMATH_CALUDE_root_sum_l3144_314483

theorem root_sum (p q : ℝ) (h1 : q ≠ 0) (h2 : q^2 + p*q + q = 0) : p + q = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_l3144_314483


namespace NUMINAMATH_CALUDE_expected_girls_left_10_7_l3144_314415

/-- The expected number of girls standing to the left of all boys in a random lineup -/
def expected_girls_left (num_boys num_girls : ℕ) : ℚ :=
  (num_girls : ℚ) / (num_boys + num_girls + 1 : ℚ)

/-- Theorem: In a random lineup of 10 boys and 7 girls, the expected number of girls 
    standing to the left of all boys is 7/11 -/
theorem expected_girls_left_10_7 :
  expected_girls_left 10 7 = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expected_girls_left_10_7_l3144_314415


namespace NUMINAMATH_CALUDE_max_value_of_f_l3144_314480

/-- The function f(x) = -2x^2 + 4x + 10 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 10

/-- The maximum value of f(x) for x ≥ 0 is 12 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 12 ∧ ∀ (x : ℝ), x ≥ 0 → f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3144_314480


namespace NUMINAMATH_CALUDE_tens_digit_of_1047_pow_1024_minus_1049_l3144_314424

theorem tens_digit_of_1047_pow_1024_minus_1049 : ∃ n : ℕ, (1047^1024 - 1049) % 100 = 32 ∧ n * 10 + 3 = (1047^1024 - 1049) / 10 % 10 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_1047_pow_1024_minus_1049_l3144_314424


namespace NUMINAMATH_CALUDE_heavy_equipment_operator_pay_is_140_l3144_314406

/-- Calculates the daily pay for heavy equipment operators given the total number of people hired,
    total payroll, number of laborers, and daily pay for laborers. -/
def heavy_equipment_operator_pay (total_hired : ℕ) (total_payroll : ℕ) (laborers : ℕ) (laborer_pay : ℕ) : ℕ :=
  (total_payroll - laborers * laborer_pay) / (total_hired - laborers)

/-- Proves that given the specified conditions, the daily pay for heavy equipment operators is $140. -/
theorem heavy_equipment_operator_pay_is_140 :
  heavy_equipment_operator_pay 35 3950 19 90 = 140 := by
  sorry

end NUMINAMATH_CALUDE_heavy_equipment_operator_pay_is_140_l3144_314406


namespace NUMINAMATH_CALUDE_peanut_seed_germination_probability_l3144_314439

/-- The probability of exactly k successes in n independent trials,
    each with probability p of success. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability that exactly 2 out of 4 planted seeds will germinate,
    given that the probability of each seed germinating is 4/5. -/
theorem peanut_seed_germination_probability :
  binomial_probability 4 2 (4/5) = 96/625 := by
  sorry

end NUMINAMATH_CALUDE_peanut_seed_germination_probability_l3144_314439


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3144_314429

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 9 →
  a 3 = 15 →
  a 7 = 33 →
  a 4 + a 5 + a 6 = 81 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3144_314429


namespace NUMINAMATH_CALUDE_apple_products_cost_l3144_314445

/-- Calculate the final cost of Apple products after discounts, taxes, and cashback -/
theorem apple_products_cost (iphone_price iwatch_price ipad_price : ℝ)
  (iphone_discount iwatch_discount ipad_discount : ℝ)
  (iphone_tax iwatch_tax ipad_tax : ℝ)
  (cashback : ℝ)
  (h1 : iphone_price = 800)
  (h2 : iwatch_price = 300)
  (h3 : ipad_price = 500)
  (h4 : iphone_discount = 0.15)
  (h5 : iwatch_discount = 0.10)
  (h6 : ipad_discount = 0.05)
  (h7 : iphone_tax = 0.07)
  (h8 : iwatch_tax = 0.05)
  (h9 : ipad_tax = 0.06)
  (h10 : cashback = 0.02) :
  ∃ (total_cost : ℝ), 
    abs (total_cost - 1484.31) < 0.01 ∧
    total_cost = 
      (1 - cashback) * 
      ((iphone_price * (1 - iphone_discount) * (1 + iphone_tax)) +
       (iwatch_price * (1 - iwatch_discount) * (1 + iwatch_tax)) +
       (ipad_price * (1 - ipad_discount) * (1 + ipad_tax))) :=
by sorry


end NUMINAMATH_CALUDE_apple_products_cost_l3144_314445


namespace NUMINAMATH_CALUDE_savings_ratio_proof_l3144_314471

def husband_contribution : ℕ := 335
def wife_contribution : ℕ := 225
def savings_period_months : ℕ := 6
def weeks_per_month : ℕ := 4
def num_children : ℕ := 4
def amount_per_child : ℕ := 1680

def total_weekly_contribution : ℕ := husband_contribution + wife_contribution
def total_weeks : ℕ := savings_period_months * weeks_per_month
def total_savings : ℕ := total_weekly_contribution * total_weeks
def total_divided : ℕ := amount_per_child * num_children

theorem savings_ratio_proof : 
  (total_divided : ℚ) / total_savings = 1/2 := by sorry

end NUMINAMATH_CALUDE_savings_ratio_proof_l3144_314471


namespace NUMINAMATH_CALUDE_probability_two_students_together_l3144_314408

/-- The probability of two specific students standing together in a row of 4 students -/
theorem probability_two_students_together (n : ℕ) (h : n = 4) : 
  (2 * 3 * 2 * 1 : ℚ) / (n * (n - 1) * (n - 2) * (n - 3)) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_students_together_l3144_314408


namespace NUMINAMATH_CALUDE_gcf_72_90_l3144_314431

theorem gcf_72_90 : Nat.gcd 72 90 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_72_90_l3144_314431


namespace NUMINAMATH_CALUDE_rowing_round_trip_time_l3144_314465

/-- The time taken for a round trip rowing journey given the rowing speed, current velocity, and distance. -/
theorem rowing_round_trip_time
  (rowing_speed : ℝ)
  (current_velocity : ℝ)
  (distance : ℝ)
  (h1 : rowing_speed = 10)
  (h2 : current_velocity = 2)
  (h3 : distance = 72)
  : (distance / (rowing_speed - current_velocity) + distance / (rowing_speed + current_velocity)) = 15 :=
by sorry

end NUMINAMATH_CALUDE_rowing_round_trip_time_l3144_314465


namespace NUMINAMATH_CALUDE_paige_folders_proof_l3144_314417

def number_of_folders (initial_files deleted_files files_per_folder : ℕ) : ℕ :=
  (initial_files - deleted_files) / files_per_folder

theorem paige_folders_proof (initial_files deleted_files files_per_folder : ℕ) 
  (h1 : initial_files = 27)
  (h2 : deleted_files = 9)
  (h3 : files_per_folder = 6)
  : number_of_folders initial_files deleted_files files_per_folder = 3 := by
  sorry

#eval number_of_folders 27 9 6

end NUMINAMATH_CALUDE_paige_folders_proof_l3144_314417


namespace NUMINAMATH_CALUDE_circle_equation_through_pole_l3144_314464

/-- A circle in a polar coordinate system --/
structure PolarCircle where
  center : (ℝ × ℝ)
  passes_through_pole : Bool

/-- The polar coordinate equation of a circle --/
def polar_equation (c : PolarCircle) : ℝ → ℝ := sorry

theorem circle_equation_through_pole (c : PolarCircle) 
  (h1 : c.center = (Real.sqrt 2, Real.pi))
  (h2 : c.passes_through_pole = true) :
  polar_equation c = λ θ => -2 * Real.sqrt 2 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_through_pole_l3144_314464


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3144_314427

theorem sufficient_not_necessary_condition : 
  (∃ x : ℝ, x ≠ 5 ∧ x^2 - 4*x - 5 = 0) ∧
  (∀ x : ℝ, x = 5 → x^2 - 4*x - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3144_314427


namespace NUMINAMATH_CALUDE_binary_1101_is_13_l3144_314485

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_is_13 : 
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_is_13_l3144_314485


namespace NUMINAMATH_CALUDE_signUpWaysCorrect_l3144_314410

/-- The number of ways four students can sign up for three sports -/
def signUpWays : ℕ := 81

/-- The number of students -/
def numStudents : ℕ := 4

/-- The number of sports -/
def numSports : ℕ := 3

theorem signUpWaysCorrect : signUpWays = numSports ^ numStudents := by
  sorry

end NUMINAMATH_CALUDE_signUpWaysCorrect_l3144_314410


namespace NUMINAMATH_CALUDE_max_value_implies_a_values_exactly_two_a_values_l3144_314416

/-- The function f for a given real number a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

/-- The theorem stating the possible values of a -/
theorem max_value_implies_a_values (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f a x = 2) → 
  a = -1 ∨ a = 2 := by
sorry

/-- The main theorem stating that there are exactly two possible values for a -/
theorem exactly_two_a_values : 
  ∃! s : Set ℝ, s = {-1, 2} ∧ 
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧ 
           (∃ x ∈ Set.Icc 0 1, f a x = 2) → 
           a ∈ s := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_values_exactly_two_a_values_l3144_314416


namespace NUMINAMATH_CALUDE_units_digit_of_division_l3144_314492

theorem units_digit_of_division : 
  (30 * 31 * 32 * 33 * 34 * 35) / 14000 % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_division_l3144_314492


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3144_314449

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 23 * n ≡ 789 [ZMOD 11] ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬(23 * m ≡ 789 [ZMOD 11])) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3144_314449


namespace NUMINAMATH_CALUDE_number_accurate_to_hundreds_l3144_314472

def number : ℝ := 1.45 * 10^4

def accurate_to_hundreds_place (x : ℝ) : Prop :=
  ∃ n : ℤ, x = (n * 100 : ℝ) ∧ n ≥ 0 ∧ n < 1000

theorem number_accurate_to_hundreds : accurate_to_hundreds_place number := by
  sorry

end NUMINAMATH_CALUDE_number_accurate_to_hundreds_l3144_314472


namespace NUMINAMATH_CALUDE_joan_family_distance_l3144_314474

/-- Calculates the distance traveled given the total time, driving speed, and break times. -/
def distance_traveled (total_time : ℝ) (speed : ℝ) (lunch_break : ℝ) (bathroom_break : ℝ) : ℝ :=
  (total_time - (lunch_break + 2 * bathroom_break)) * speed

/-- Theorem: Given Joan's travel conditions, her family lives 480 miles away. -/
theorem joan_family_distance :
  let total_time : ℝ := 9  -- 9 hours total trip time
  let speed : ℝ := 60      -- 60 mph driving speed
  let lunch_break : ℝ := 0.5  -- 30 minutes = 0.5 hours
  let bathroom_break : ℝ := 0.25  -- 15 minutes = 0.25 hours
  distance_traveled total_time speed lunch_break bathroom_break = 480 := by
  sorry

#eval distance_traveled 9 60 0.5 0.25

end NUMINAMATH_CALUDE_joan_family_distance_l3144_314474


namespace NUMINAMATH_CALUDE_shopping_trip_remainder_l3144_314478

/-- Calculates the remaining amount after a shopping trip --/
theorem shopping_trip_remainder
  (initial_amount : ℝ)
  (peach_price peach_quantity : ℝ)
  (cherry_price cherry_quantity : ℝ)
  (baguette_price baguette_quantity : ℝ)
  (discount_threshold discount_rate : ℝ)
  (tax_rate : ℝ)
  (h1 : initial_amount = 20)
  (h2 : peach_price = 2)
  (h3 : peach_quantity = 3)
  (h4 : cherry_price = 3.5)
  (h5 : cherry_quantity = 2)
  (h6 : baguette_price = 1.25)
  (h7 : baguette_quantity = 4)
  (h8 : discount_threshold = 10)
  (h9 : discount_rate = 0.1)
  (h10 : tax_rate = 0.05) :
  let subtotal := peach_price * peach_quantity + cherry_price * cherry_quantity + baguette_price * baguette_quantity
  let discounted_total := if subtotal > discount_threshold then subtotal * (1 - discount_rate) else subtotal
  let final_total := discounted_total * (1 + tax_rate)
  let remainder := initial_amount - final_total
  remainder = 2.99 := by sorry

end NUMINAMATH_CALUDE_shopping_trip_remainder_l3144_314478


namespace NUMINAMATH_CALUDE_fruit_basket_difference_l3144_314490

/-- Proof that the difference between oranges and apples is 2 in a fruit basket -/
theorem fruit_basket_difference : ∀ (apples bananas peaches : ℕ),
  apples + bananas + peaches + 6 = 28 →
  bananas = 3 * apples →
  peaches = bananas / 2 →
  6 - apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_difference_l3144_314490


namespace NUMINAMATH_CALUDE_positive_number_square_root_l3144_314405

theorem positive_number_square_root (x : ℝ) : 
  x > 0 → Real.sqrt ((7 * x) / 3) = x → x = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_positive_number_square_root_l3144_314405


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l3144_314479

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 18 % 31 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 18 % 31 → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l3144_314479


namespace NUMINAMATH_CALUDE_barry_votes_difference_l3144_314414

def election_votes (marcy_votes barry_votes joey_votes : ℕ) : Prop :=
  marcy_votes = 3 * barry_votes ∧
  ∃ x, barry_votes = 2 * (joey_votes + x) ∧
  marcy_votes = 66 ∧
  joey_votes = 8

theorem barry_votes_difference :
  ∀ marcy_votes barry_votes joey_votes,
  election_votes marcy_votes barry_votes joey_votes →
  barry_votes - joey_votes = 14 := by
sorry

end NUMINAMATH_CALUDE_barry_votes_difference_l3144_314414


namespace NUMINAMATH_CALUDE_product_87_93_l3144_314425

theorem product_87_93 : 87 * 93 = 8091 := by
  sorry

end NUMINAMATH_CALUDE_product_87_93_l3144_314425


namespace NUMINAMATH_CALUDE_final_position_of_E_l3144_314495

-- Define the position of E as a pair of axes (base_axis, top_axis)
inductive Axis
  | PositiveX
  | NegativeX
  | PositiveY
  | NegativeY

def Position := Axis × Axis

-- Define the transformations
def rotateClockwise270 (p : Position) : Position :=
  match p with
  | (Axis.NegativeX, Axis.PositiveY) => (Axis.PositiveY, Axis.NegativeX)
  | _ => p  -- For completeness, though we only care about the initial position

def reflectXAxis (p : Position) : Position :=
  match p with
  | (base, top) => (
      match base with
      | Axis.PositiveY => Axis.NegativeY
      | Axis.NegativeY => Axis.PositiveY
      | _ => base,
      top
    )

def reflectYAxis (p : Position) : Position :=
  match p with
  | (base, top) => (
      base,
      match top with
      | Axis.PositiveX => Axis.NegativeX
      | Axis.NegativeX => Axis.PositiveX
      | _ => top
    )

def halfTurn (p : Position) : Position :=
  match p with
  | (base, top) => (
      match base with
      | Axis.PositiveY => Axis.NegativeY
      | Axis.NegativeY => Axis.PositiveY
      | Axis.PositiveX => Axis.NegativeX
      | Axis.NegativeX => Axis.PositiveX,
      match top with
      | Axis.PositiveY => Axis.NegativeY
      | Axis.NegativeY => Axis.PositiveY
      | Axis.PositiveX => Axis.NegativeX
      | Axis.NegativeX => Axis.PositiveX
    )

-- Theorem statement
theorem final_position_of_E :
  let initial_position : Position := (Axis.NegativeX, Axis.PositiveY)
  let final_position := halfTurn (reflectYAxis (reflectXAxis (rotateClockwise270 initial_position)))
  final_position = (Axis.NegativeY, Axis.NegativeX) :=
by
  sorry

end NUMINAMATH_CALUDE_final_position_of_E_l3144_314495


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3144_314426

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := 3 * i / (1 + i)
  Complex.im z = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3144_314426


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3144_314496

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (6, x)
  parallel a b → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3144_314496


namespace NUMINAMATH_CALUDE_fold_coincide_points_l3144_314458

/-- Given a number line where folding causes -2 to coincide with 8, and two points A and B
    with a distance of 2024 between them (A to the left of B) that coincide after folding,
    the coordinate of point A is -1009. -/
theorem fold_coincide_points (A B : ℝ) : 
  (A < B) →  -- A is to the left of B
  (B - A = 2024) →  -- Distance between A and B is 2024
  (A + B) / 2 = (-2 + 8) / 2 →  -- Midpoint of A and B is the same as midpoint of -2 and 8
  A = -1009 := by
sorry

end NUMINAMATH_CALUDE_fold_coincide_points_l3144_314458


namespace NUMINAMATH_CALUDE_volleyball_substitutions_remainder_l3144_314482

/-- Number of players in a volleyball team -/
def total_players : ℕ := 18

/-- Number of starting players -/
def starting_players : ℕ := 6

/-- Number of substitute players -/
def substitute_players : ℕ := total_players - starting_players

/-- Maximum number of substitutions allowed -/
def max_substitutions : ℕ := 5

/-- Calculate the number of ways to make k substitutions -/
def substitution_ways (k : ℕ) : ℕ :=
  if k = 0 then 1
  else starting_players * (substitute_players - k + 1) * substitution_ways (k - 1)

/-- Total number of ways to execute substitutions -/
def total_substitution_ways : ℕ :=
  List.sum (List.map substitution_ways (List.range (max_substitutions + 1)))

/-- The main theorem to prove -/
theorem volleyball_substitutions_remainder :
  total_substitution_ways % 1000 = 271 := by sorry

end NUMINAMATH_CALUDE_volleyball_substitutions_remainder_l3144_314482


namespace NUMINAMATH_CALUDE_mathborough_rainfall_2007_l3144_314484

/-- Calculates the total rainfall in Mathborough for the year 2007 given the rainfall data from 2005 to 2007. -/
theorem mathborough_rainfall_2007 (rainfall_2005 : ℝ) (increase_2006 : ℝ) (increase_2007 : ℝ) :
  rainfall_2005 = 40.5 →
  increase_2006 = 3 →
  increase_2007 = 4 →
  (rainfall_2005 + increase_2006 + increase_2007) * 12 = 570 := by
  sorry

end NUMINAMATH_CALUDE_mathborough_rainfall_2007_l3144_314484


namespace NUMINAMATH_CALUDE_percentage_markup_approx_l3144_314467

def selling_price : ℝ := 8337
def cost_price : ℝ := 6947.5

theorem percentage_markup_approx (ε : ℝ) (h : ε > 0) :
  ∃ (markup_percentage : ℝ),
    abs (markup_percentage - 19.99) < ε ∧
    markup_percentage = (selling_price - cost_price) / cost_price * 100 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_markup_approx_l3144_314467


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3144_314459

theorem quadratic_roots_condition (c : ℚ) : 
  (∀ x : ℚ, x^2 - 7*x + c = 0 ↔ ∃ s : ℤ, s^2 = 9*c ∧ x = (7 + s) / 2 ∨ x = (7 - s) / 2) →
  c = 49 / 13 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3144_314459


namespace NUMINAMATH_CALUDE_malcom_cards_left_l3144_314428

theorem malcom_cards_left (brandon_cards : ℕ) (malcom_extra_cards : ℕ) : 
  brandon_cards = 20 →
  malcom_extra_cards = 8 →
  (brandon_cards + malcom_extra_cards) / 2 = 14 := by
sorry

end NUMINAMATH_CALUDE_malcom_cards_left_l3144_314428


namespace NUMINAMATH_CALUDE_robs_money_total_l3144_314491

/-- Represents the value of coins in dollars -/
def coin_value (coin : String) : ℚ :=
  match coin with
  | "quarter" => 25 / 100
  | "dime" => 10 / 100
  | "nickel" => 5 / 100
  | "penny" => 1 / 100
  | _ => 0

/-- Calculates the total value of a given number of coins -/
def coin_total (coin : String) (count : ℕ) : ℚ :=
  (coin_value coin) * count

/-- Theorem: Rob's total money is $2.42 -/
theorem robs_money_total :
  let quarters := coin_total "quarter" 7
  let dimes := coin_total "dime" 3
  let nickels := coin_total "nickel" 5
  let pennies := coin_total "penny" 12
  quarters + dimes + nickels + pennies = 242 / 100 := by
  sorry

end NUMINAMATH_CALUDE_robs_money_total_l3144_314491


namespace NUMINAMATH_CALUDE_experimental_primary_school_students_l3144_314489

/-- The total number of students in Experimental Primary School -/
def total_students (num_classes : ℕ) (boys_in_class1 : ℕ) (girls_in_class1 : ℕ) : ℕ :=
  num_classes * (boys_in_class1 + girls_in_class1)

/-- Theorem: The total number of students in Experimental Primary School is 896 -/
theorem experimental_primary_school_students :
  total_students 28 21 11 = 896 := by
  sorry

end NUMINAMATH_CALUDE_experimental_primary_school_students_l3144_314489


namespace NUMINAMATH_CALUDE_substance_volume_l3144_314403

/-- Given a substance where 1 gram occupies 10 cubic centimeters, 
    prove that 100 kg of this substance occupies 1 cubic meter. -/
theorem substance_volume (substance : Type) 
  (volume : substance → ℝ) 
  (mass : substance → ℝ) 
  (s : substance) 
  (h1 : volume s = mass s * 10 * 1000000⁻¹) 
  (h2 : mass s = 100) : 
  volume s = 1 := by
sorry

end NUMINAMATH_CALUDE_substance_volume_l3144_314403


namespace NUMINAMATH_CALUDE_negative_cube_squared_l3144_314448

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l3144_314448


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l3144_314441

theorem largest_n_divisibility : ∃ (n : ℕ), n = 20 ∧ 
  (∀ m : ℕ, m > n → ¬((m^3 + 150) % (m + 5) = 0)) ∧ 
  ((n^3 + 150) % (n + 5) = 0) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l3144_314441


namespace NUMINAMATH_CALUDE_extreme_points_when_a_neg_one_max_value_on_interval_l3144_314476

/-- The function f(x) = x³ + 3ax² -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2

/-- Theorem for extreme points and values when a = -1 -/
theorem extreme_points_when_a_neg_one :
  let f_neg_one := f (-1)
  ∃ (local_max local_min : ℝ),
    (local_max = 0 ∧ f_neg_one local_max = 0) ∧
    (local_min = 2 ∧ f_neg_one local_min = -4) ∧
    ∀ x, f_neg_one x ≤ f_neg_one local_max ∨ f_neg_one x ≥ f_neg_one local_min :=
sorry

/-- Theorem for maximum value on [0,2] -/
theorem max_value_on_interval (a : ℝ) :
  let max_value := if a ≥ 0 then f a 2
                   else if a > -1 then max (f a 0) (f a 2)
                   else f a 0
  ∀ x ∈ Set.Icc 0 2, f a x ≤ max_value :=
sorry

end NUMINAMATH_CALUDE_extreme_points_when_a_neg_one_max_value_on_interval_l3144_314476


namespace NUMINAMATH_CALUDE_triangle_angle_determinant_l3144_314477

theorem triangle_angle_determinant (A B C : ℝ) (h : A + B + C = Real.pi) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
    if i = j then Real.sin (2 * (if i = 0 then A else if i = 1 then B else C))
    else 1
  Matrix.det M = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_determinant_l3144_314477


namespace NUMINAMATH_CALUDE_certain_amount_added_l3144_314401

theorem certain_amount_added (x y : ℝ) : 
  x = 18 → 3 * (2 * x + y) = 123 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_added_l3144_314401


namespace NUMINAMATH_CALUDE_circle_line_intersection_min_value_l3144_314451

/-- Given a circle with center (m,n) in the first quadrant and radius 3,
    intersected by a line to form a chord of length 4,
    the minimum value of (m+2n)/(mn) is 8/3 -/
theorem circle_line_intersection_min_value (m n : ℝ) :
  m > 0 →
  n > 0 →
  m + 2*n = 3 →
  (∀ x y : ℝ, (x - m)^2 + (y - n)^2 = 9 → x + 2*y + 2 = 0 → 
    ∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ (x' - m)^2 + (y' - n)^2 = 9 ∧ 
    x' + 2*y' + 2 = 0 ∧ (x - x')^2 + (y - y')^2 = 16) →
  (m + 2*n) / (m * n) ≥ 8/3 :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_min_value_l3144_314451


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l3144_314437

/-- The gain percent on a cycle sale --/
theorem cycle_gain_percent (cost_price selling_price : ℝ) 
  (h1 : cost_price = 900)
  (h2 : selling_price = 1180) : 
  (selling_price - cost_price) / cost_price * 100 = 31.11 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l3144_314437


namespace NUMINAMATH_CALUDE_trapezoid_long_side_length_l3144_314419

/-- Represents a square divided into two trapezoids and a quadrilateral -/
structure DividedSquare where
  side_length : ℝ
  segment_length : ℝ
  trapezoid_long_side : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : DividedSquare) : Prop :=
  s.side_length = 2 ∧
  s.segment_length = 1 ∧
  (s.trapezoid_long_side + s.segment_length) * s.segment_length / 2 = s.side_length^2 / 3

/-- The theorem to be proved -/
theorem trapezoid_long_side_length (s : DividedSquare) :
  problem_conditions s → s.trapezoid_long_side = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_long_side_length_l3144_314419
