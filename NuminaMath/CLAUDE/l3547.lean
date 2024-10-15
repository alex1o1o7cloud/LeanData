import Mathlib

namespace NUMINAMATH_CALUDE_largest_view_angle_point_l3547_354711

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle -/
structure Angle where
  vertex : Point
  side1 : Point
  side2 : Point

/-- Checks if an angle is acute -/
def isAcute (α : Angle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a point is on one side of an angle -/
def isOnSide (p : Point) (α : Angle) : Prop := sorry

/-- Checks if a point is on the other side of an angle -/
def isOnOtherSide (p : Point) (α : Angle) : Prop := sorry

/-- Calculates the angle at which a segment is seen from a point -/
def viewAngle (p : Point) (a b : Point) : ℝ := sorry

/-- States that a point maximizes the view angle of a segment -/
def maximizesViewAngle (c : Point) (a b : Point) (α : Angle) : Prop :=
  ∀ p, isOnOtherSide p α → viewAngle c a b ≥ viewAngle p a b

theorem largest_view_angle_point (α : Angle) (a b c : Point) :
  isAcute α →
  isOnSide a α →
  isOnSide b α →
  isOnOtherSide c α →
  maximizesViewAngle c a b α →
  (distance α.vertex c)^2 = distance α.vertex a * distance α.vertex b := by
  sorry

end NUMINAMATH_CALUDE_largest_view_angle_point_l3547_354711


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3547_354727

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_incr : is_increasing_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_diff : a 4 - a 3 = 4) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3547_354727


namespace NUMINAMATH_CALUDE_susan_strawberry_picking_l3547_354724

theorem susan_strawberry_picking (basket_capacity : ℕ) (total_picked : ℕ) (eaten_per_handful : ℕ) :
  basket_capacity = 60 →
  total_picked = 75 →
  eaten_per_handful = 1 →
  ∃ (strawberries_per_handful : ℕ),
    strawberries_per_handful * (total_picked / strawberries_per_handful) = total_picked ∧
    (strawberries_per_handful - eaten_per_handful) * (total_picked / strawberries_per_handful) = basket_capacity ∧
    strawberries_per_handful = 5 :=
by sorry

end NUMINAMATH_CALUDE_susan_strawberry_picking_l3547_354724


namespace NUMINAMATH_CALUDE_tasty_candy_identification_l3547_354767

/-- Represents a strategy for identifying tasty candies -/
structure TastyStrategy where
  query : (ℕ → Bool) → Finset ℕ → ℕ
  interpret : (Finset ℕ → ℕ) → Finset ℕ

/-- The total number of candies -/
def total_candies : ℕ := 28

/-- A function that determines if a candy is tasty -/
def is_tasty : ℕ → Bool := sorry

/-- The maximum number of queries allowed -/
def max_queries : ℕ := 20

theorem tasty_candy_identification :
  ∃ (s : TastyStrategy),
    (∀ (f : ℕ → Bool),
      let query_count := (Finset.range total_candies).card
      s.interpret (λ subset => s.query f subset) =
        {i | i ∈ Finset.range total_candies ∧ f i}) ∧
    (∀ (f : ℕ → Bool),
      (Finset.range total_candies).card ≤ max_queries) :=
sorry

end NUMINAMATH_CALUDE_tasty_candy_identification_l3547_354767


namespace NUMINAMATH_CALUDE_valid_paths_count_l3547_354764

/-- Represents a point on a 2D grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points on a grid --/
def numPaths (start finish : Point) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- Calculates the number of paths between two points passing through an intermediate point --/
def numPathsThrough (start mid finish : Point) : ℕ :=
  (numPaths start mid) * (numPaths mid finish)

/-- The main theorem stating the number of valid paths --/
theorem valid_paths_count :
  let start := Point.mk 0 0
  let finish := Point.mk 5 3
  let risky := Point.mk 2 2
  (numPaths start finish) - (numPathsThrough start risky finish) = 32 := by
  sorry

end NUMINAMATH_CALUDE_valid_paths_count_l3547_354764


namespace NUMINAMATH_CALUDE_product_equals_square_l3547_354714

theorem product_equals_square : 50 * 24.96 * 2.496 * 500 = (1248 : ℝ)^2 := by sorry

end NUMINAMATH_CALUDE_product_equals_square_l3547_354714


namespace NUMINAMATH_CALUDE_absolute_value_relation_l3547_354768

theorem absolute_value_relation :
  let p : ℝ → Prop := λ x ↦ |x| = 2
  let q : ℝ → Prop := λ x ↦ x = 2
  (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_relation_l3547_354768


namespace NUMINAMATH_CALUDE_julio_lime_cost_l3547_354786

/-- Represents the cost of limes for Julio's mocktails over 30 days -/
def lime_cost (mocktails_per_day : ℕ) (lime_juice_per_mocktail : ℚ) (juice_per_lime : ℚ) (days : ℕ) (limes_per_dollar : ℕ) : ℚ :=
  let limes_needed := (mocktails_per_day * lime_juice_per_mocktail * days) / juice_per_lime
  let lime_sets := (limes_needed / limes_per_dollar).ceil
  lime_sets

theorem julio_lime_cost :
  lime_cost 1 (1/2) 2 30 3 = 5 := by
  sorry

#eval lime_cost 1 (1/2) 2 30 3

end NUMINAMATH_CALUDE_julio_lime_cost_l3547_354786


namespace NUMINAMATH_CALUDE_intersection_subset_l3547_354701

def P : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {1, 2, 3}

theorem intersection_subset : P ∩ Q ⊆ Q := by
  sorry

end NUMINAMATH_CALUDE_intersection_subset_l3547_354701


namespace NUMINAMATH_CALUDE_equation_solution_l3547_354754

theorem equation_solution : ∃! x : ℚ, 2 * x - 5/6 = 7/18 + 1/2 ∧ x = 31/36 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3547_354754


namespace NUMINAMATH_CALUDE_divisors_of_ten_factorial_greater_than_nine_factorial_l3547_354732

theorem divisors_of_ten_factorial_greater_than_nine_factorial : 
  (Finset.filter (fun d => d > Nat.factorial 9 ∧ Nat.factorial 10 % d = 0) 
    (Finset.range (Nat.factorial 10 + 1))).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_ten_factorial_greater_than_nine_factorial_l3547_354732


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l3547_354723

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Theorem stating the volume and height of a specific tetrahedron -/
theorem tetrahedron_volume_and_height :
  let A₁ : Point3D := ⟨2, 3, 1⟩
  let A₂ : Point3D := ⟨4, 1, -2⟩
  let A₃ : Point3D := ⟨6, 3, 7⟩
  let A₄ : Point3D := ⟨7, 5, -3⟩
  (tetrahedronVolume A₁ A₂ A₃ A₄ = 70/3) ∧
  (tetrahedronHeight A₁ A₂ A₃ A₄ = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l3547_354723


namespace NUMINAMATH_CALUDE_female_student_stats_l3547_354752

/-- Represents the class statistics -/
structure ClassStats where
  total_students : ℕ
  male_students : ℕ
  overall_avg_score : ℚ
  male_algebra_avg : ℚ
  male_geometry_avg : ℚ
  male_calculus_avg : ℚ
  female_algebra_avg : ℚ
  female_geometry_avg : ℚ
  female_calculus_avg : ℚ
  algebra_geometry_attendance : ℚ
  calculus_attendance_increase : ℚ

/-- Theorem stating the proportion and number of female students -/
theorem female_student_stats (stats : ClassStats)
  (h_total : stats.total_students = 30)
  (h_male : stats.male_students = 8)
  (h_overall_avg : stats.overall_avg_score = 90)
  (h_male_algebra : stats.male_algebra_avg = 87)
  (h_male_geometry : stats.male_geometry_avg = 95)
  (h_male_calculus : stats.male_calculus_avg = 89)
  (h_female_algebra : stats.female_algebra_avg = 92)
  (h_female_geometry : stats.female_geometry_avg = 94)
  (h_female_calculus : stats.female_calculus_avg = 91)
  (h_alg_geo_attendance : stats.algebra_geometry_attendance = 85)
  (h_calc_attendance : stats.calculus_attendance_increase = 4) :
  (stats.total_students - stats.male_students : ℚ) / stats.total_students = 11 / 15 ∧
  stats.total_students - stats.male_students = 22 := by
    sorry


end NUMINAMATH_CALUDE_female_student_stats_l3547_354752


namespace NUMINAMATH_CALUDE_circle_M_equation_l3547_354757

/-- The equation of a line passing through the center of circle M -/
def center_line (x y : ℝ) : Prop := x - y - 4 = 0

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

/-- The equation of circle M -/
def circle_M (x y : ℝ) : Prop := (x - 1/2)^2 + (y + 7/2)^2 = 89/2

/-- Theorem stating that the given conditions imply the equation of circle M -/
theorem circle_M_equation (x y : ℝ) :
  (∃ (xc yc : ℝ), center_line xc yc ∧ 
    (∀ (xi yi : ℝ), (circle1 xi yi ∧ circle2 xi yi) → 
      (x - xc)^2 + (y - yc)^2 = (xi - xc)^2 + (yi - yc)^2)) →
  circle_M x y :=
sorry

end NUMINAMATH_CALUDE_circle_M_equation_l3547_354757


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3547_354784

theorem quadratic_inequality_solution (x : ℝ) : 
  (2 * x^2 + x < 6) ↔ (-2 < x ∧ x < 3/2) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3547_354784


namespace NUMINAMATH_CALUDE_min_people_liking_both_l3547_354774

theorem min_people_liking_both (total : ℕ) (chopin : ℕ) (beethoven : ℕ) 
  (h1 : total = 120) (h2 : chopin = 95) (h3 : beethoven = 80) :
  ∃ both : ℕ, both ≥ 55 ∧ chopin + beethoven - both ≤ total := by
  sorry

end NUMINAMATH_CALUDE_min_people_liking_both_l3547_354774


namespace NUMINAMATH_CALUDE_existence_of_critical_point_and_upper_bound_l3547_354734

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * x^2 - 2 * x - 1

theorem existence_of_critical_point_and_upper_bound (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo (-1/a) (-1/4) ∧ 
    (deriv (f a)) x₀ = 0 ∧ 
    f a x₀ < 15/16 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_critical_point_and_upper_bound_l3547_354734


namespace NUMINAMATH_CALUDE_commute_time_difference_l3547_354777

theorem commute_time_difference (x y : ℝ) : 
  (x + y + 10 + 11 + 9) / 5 = 10 →
  ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2 →
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_commute_time_difference_l3547_354777


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l3547_354792

/-- The distance from a point on the line y = 2x + 1 to the x-axis -/
theorem distance_to_x_axis (k : ℝ) : 
  let M : ℝ × ℝ := (-2, k)
  let line_eq : ℝ → ℝ := λ x => 2 * x + 1
  k = line_eq (-2) →
  |k| = 3 := by
sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l3547_354792


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3547_354708

/-- A hyperbola with focal length 2√5 and asymptote x - 2y = 0 has equation x^2/4 - y^2 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Given hyperbola equation
  (a^2 + b^2 = 5) →                         -- Focal length condition
  (a = 2 * b) →                             -- Asymptote condition
  (∀ x y : ℝ, x^2 / 4 - y^2 = 1) :=         -- Conclusion: specific hyperbola equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3547_354708


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3547_354797

theorem geometric_sequence_problem (a b c : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ -2 = -2 * r ∧ a = -2 * r^2 ∧ b = -2 * r^3 ∧ c = -2 * r^4 ∧ -8 = -2 * r^5) →
  b = -4 ∧ a * c = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3547_354797


namespace NUMINAMATH_CALUDE_triangle_inequality_l3547_354730

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3547_354730


namespace NUMINAMATH_CALUDE_max_bouquets_sara_l3547_354794

def red_flowers : ℕ := 47
def yellow_flowers : ℕ := 63
def blue_flowers : ℕ := 54
def orange_flowers : ℕ := 29
def pink_flowers : ℕ := 36

theorem max_bouquets_sara :
  ∀ n : ℕ,
    n ≤ red_flowers ∧
    n ≤ yellow_flowers ∧
    n ≤ blue_flowers ∧
    n ≤ orange_flowers ∧
    n ≤ pink_flowers →
    n ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_bouquets_sara_l3547_354794


namespace NUMINAMATH_CALUDE_production_days_calculation_l3547_354702

/-- Given the average production and a new day's production, find the number of previous days. -/
theorem production_days_calculation (avg_n : ℝ) (new_prod : ℝ) (avg_n_plus_1 : ℝ) :
  avg_n = 50 →
  new_prod = 100 →
  avg_n_plus_1 = 55 →
  (avg_n * n + new_prod) / (n + 1) = avg_n_plus_1 →
  n = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l3547_354702


namespace NUMINAMATH_CALUDE_polygon_side_length_theorem_l3547_354720

/-- A convex polygon that can be divided into unit equilateral triangles and unit squares -/
structure ConvexPolygon where
  sides : List ℕ
  is_convex : Bool

/-- The number of ways to divide a ConvexPolygon into unit equilateral triangles and unit squares -/
def divisionWays (M : ConvexPolygon) : ℕ := sorry

theorem polygon_side_length_theorem (M : ConvexPolygon) (p : ℕ) (h_prime : Nat.Prime p) :
  divisionWays M = p → ∃ (side : ℕ), side ∈ M.sides ∧ side = p - 1 := by sorry

end NUMINAMATH_CALUDE_polygon_side_length_theorem_l3547_354720


namespace NUMINAMATH_CALUDE_polynomial_never_33_l3547_354782

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_never_33_l3547_354782


namespace NUMINAMATH_CALUDE_coin_value_calculation_l3547_354771

theorem coin_value_calculation (num_quarters num_nickels : ℕ) 
  (quarter_value nickel_value : ℚ) : 
  num_quarters = 8 → 
  num_nickels = 13 → 
  quarter_value = 25 / 100 → 
  nickel_value = 5 / 100 → 
  num_quarters * quarter_value + num_nickels * nickel_value = 265 / 100 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_calculation_l3547_354771


namespace NUMINAMATH_CALUDE_triangle_properties_l3547_354763

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) 
  (h2 : t.a = 5)
  (h3 : Real.cos t.A = 25 / 31) :
  (2 * t.a^2 = t.b^2 + t.c^2) ∧ 
  (t.a + t.b + t.c = 14) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3547_354763


namespace NUMINAMATH_CALUDE_josh_marbles_l3547_354762

/-- The number of marbles Josh had earlier -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Josh lost -/
def lost_marbles : ℕ := 11

/-- The number of marbles Josh has now -/
def current_marbles : ℕ := 8

/-- Theorem stating that the initial number of marbles is 19 -/
theorem josh_marbles : initial_marbles = lost_marbles + current_marbles := by sorry

end NUMINAMATH_CALUDE_josh_marbles_l3547_354762


namespace NUMINAMATH_CALUDE_dog_speed_is_16_l3547_354783

/-- Represents the scenario of a man and a dog walking on a path -/
structure WalkingScenario where
  path_length : Real
  man_speed : Real
  dog_trips : Nat
  remaining_distance : Real
  dog_speed : Real

/-- Checks if the given scenario is valid -/
def is_valid_scenario (s : WalkingScenario) : Prop :=
  s.path_length = 0.625 ∧
  s.man_speed = 4 ∧
  s.dog_trips = 4 ∧
  s.remaining_distance = 0.081 ∧
  s.dog_speed > s.man_speed

/-- Theorem: Given the conditions, the dog's speed is 16 km/h -/
theorem dog_speed_is_16 (s : WalkingScenario) 
  (h : is_valid_scenario s) : s.dog_speed = 16 := by
  sorry

#check dog_speed_is_16

end NUMINAMATH_CALUDE_dog_speed_is_16_l3547_354783


namespace NUMINAMATH_CALUDE_peggy_stamps_to_add_l3547_354741

/-- Given the number of stamps each person has, calculates how many stamps Peggy needs to add to have as many as Bert. -/
def stamps_to_add (peggy_stamps : ℕ) (ernie_multiplier : ℕ) (bert_multiplier : ℕ) : ℕ :=
  bert_multiplier * (ernie_multiplier * peggy_stamps) - peggy_stamps

/-- Proves that Peggy needs to add 825 stamps to have as many as Bert. -/
theorem peggy_stamps_to_add : 
  stamps_to_add 75 3 4 = 825 := by sorry

end NUMINAMATH_CALUDE_peggy_stamps_to_add_l3547_354741


namespace NUMINAMATH_CALUDE_trig_equation_solution_l3547_354715

theorem trig_equation_solution (x : ℝ) : 
  12 * Real.sin x - 5 * Real.cos x = 13 ↔ 
  ∃ k : ℤ, x = π / 2 + Real.arctan (5 / 12) + 2 * k * π :=
sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l3547_354715


namespace NUMINAMATH_CALUDE_range_of_a_l3547_354705

/-- A monotonically decreasing function on [-2, 2] -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → x < y → f x > f y

theorem range_of_a (f : ℝ → ℝ) (h1 : MonoDecreasing f) 
    (h2 : ∀ a, f (a + 1) < f (2 * a)) :
    Set.Icc (-1) 1 \ {1} = {a | a + 1 ∈ Set.Icc (-2) 2 ∧ 2 * a ∈ Set.Icc (-2) 2 ∧ f (a + 1) < f (2 * a)} :=
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3547_354705


namespace NUMINAMATH_CALUDE_touching_sphere_surface_area_l3547_354788

/-- A sphere touching a rectangle and additional segments -/
structure TouchingSphere where
  -- The rectangle ABCD
  ab : ℝ
  bc : ℝ
  -- The segment EF
  ef : ℝ
  -- EF is parallel to the plane of ABCD
  ef_parallel : True
  -- All sides of ABCD and segments AE, BE, CF, DF, EF touch the sphere
  all_touch : True
  -- Given conditions
  ef_length : ef = 3
  bc_length : bc = 5

/-- The surface area of the sphere is 180π/7 -/
theorem touching_sphere_surface_area (s : TouchingSphere) : 
  ∃ (r : ℝ), 4 * Real.pi * r^2 = (180 * Real.pi) / 7 :=
sorry

end NUMINAMATH_CALUDE_touching_sphere_surface_area_l3547_354788


namespace NUMINAMATH_CALUDE_evaluate_expression_l3547_354731

theorem evaluate_expression : 225 + 2 * 15 * 8 + 64 = 529 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3547_354731


namespace NUMINAMATH_CALUDE_f_properties_l3547_354704

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 * Real.exp x

theorem f_properties :
  (∃ x, f x = 0) ∧
  (∃ x₁ x₂, IsLocalMax f x₁ ∧ IsLocalMin f x₂) ∧
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = 1 ∧ f x₂ = 1 ∧ f x₃ = 1 ∧
    ∀ x, f x = 1 → x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l3547_354704


namespace NUMINAMATH_CALUDE_sin_cos_value_l3547_354773

theorem sin_cos_value (θ : Real) (h : Real.tan θ = 2) : Real.sin θ * Real.cos θ = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_value_l3547_354773


namespace NUMINAMATH_CALUDE_y_percent_of_y_is_9_l3547_354779

theorem y_percent_of_y_is_9 (y : ℝ) (h1 : y > 0) (h2 : y / 100 * y = 9) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_y_percent_of_y_is_9_l3547_354779


namespace NUMINAMATH_CALUDE_travis_potato_probability_l3547_354785

/-- Represents a player in the hot potato game -/
inductive Player : Type
  | George : Player
  | Jeff : Player
  | Brian : Player
  | Travis : Player

/-- The game state after each turn -/
structure GameState :=
  (george_potatoes : Nat)
  (jeff_potatoes : Nat)
  (brian_potatoes : Nat)
  (travis_potatoes : Nat)

/-- The initial game state -/
def initial_state : GameState :=
  ⟨1, 1, 0, 0⟩

/-- The probability of passing a potato to a specific player -/
def pass_probability : ℚ := 1 / 3

/-- The probability of Travis having at least one hot potato after one round -/
def travis_has_potato_probability : ℚ := 5 / 27

/-- Theorem stating the probability of Travis having at least one hot potato after one round -/
theorem travis_potato_probability :
  travis_has_potato_probability = 5 / 27 :=
by sorry


end NUMINAMATH_CALUDE_travis_potato_probability_l3547_354785


namespace NUMINAMATH_CALUDE_min_sum_squares_l3547_354750

theorem min_sum_squares (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x + 2*y + 3*z = 2) : 
  x^2 + y^2 + z^2 ≥ 2/7 ∧ 
  (x^2 + y^2 + z^2 = 2/7 ↔ x = 1/7 ∧ y = 2/7 ∧ z = 3/7) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3547_354750


namespace NUMINAMATH_CALUDE_probability_green_jellybean_l3547_354798

def total_jellybeans : ℕ := 7 + 9 + 8 + 10 + 6
def green_jellybeans : ℕ := 9

theorem probability_green_jellybean :
  (green_jellybeans : ℚ) / (total_jellybeans : ℚ) = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_jellybean_l3547_354798


namespace NUMINAMATH_CALUDE_square_perimeter_l3547_354780

theorem square_perimeter (s : Real) : 
  s > 0 → 
  (5 * s / 2 = 32) → 
  (4 * s = 51.2) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3547_354780


namespace NUMINAMATH_CALUDE_smallest_marked_cells_for_unique_tiling_l3547_354793

/-- Represents a grid of size 2n × 2n -/
structure Grid (n : ℕ) where
  size : ℕ := 2 * n

/-- Represents a set of marked cells in the grid -/
def MarkedCells (n : ℕ) := Finset (Fin (2 * n) × Fin (2 * n))

/-- Represents a domino tiling of the grid -/
def Tiling (n : ℕ) := Finset (Fin (2 * n) × Fin (2 * n) × Bool)

/-- Checks if a tiling is valid for a given set of marked cells -/
def isValidTiling (n : ℕ) (marked : MarkedCells n) (tiling : Tiling n) : Prop :=
  sorry

/-- Checks if there exists a unique valid tiling for a given set of marked cells -/
def hasUniqueTiling (n : ℕ) (marked : MarkedCells n) : Prop :=
  sorry

/-- The main theorem: The smallest number of marked cells that ensures a unique tiling is 2n -/
theorem smallest_marked_cells_for_unique_tiling (n : ℕ) (h : 0 < n) :
  ∃ (marked : MarkedCells n),
    marked.card = 2 * n ∧
    hasUniqueTiling n marked ∧
    ∀ (marked' : MarkedCells n),
      marked'.card < 2 * n → ¬(hasUniqueTiling n marked') :=
  sorry

end NUMINAMATH_CALUDE_smallest_marked_cells_for_unique_tiling_l3547_354793


namespace NUMINAMATH_CALUDE_puppy_price_calculation_l3547_354761

/-- Calculates the price per puppy in John's puppy selling scenario -/
theorem puppy_price_calculation (initial_puppies : ℕ) (stud_fee profit : ℚ) : 
  initial_puppies = 8 →
  stud_fee = 300 →
  profit = 1500 →
  (initial_puppies / 2 - 1) > 0 →
  (profit + stud_fee) / (initial_puppies / 2 - 1) = 600 := by
sorry

end NUMINAMATH_CALUDE_puppy_price_calculation_l3547_354761


namespace NUMINAMATH_CALUDE_range_of_a_l3547_354740

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  (prop_p a ∧ prop_q a) → (a ≤ -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3547_354740


namespace NUMINAMATH_CALUDE_min_running_time_l3547_354729

/-- Proves the minimum running time to cover a given distance within a time limit -/
theorem min_running_time 
  (total_distance : ℝ) 
  (time_limit : ℝ) 
  (walking_speed : ℝ) 
  (running_speed : ℝ) 
  (h1 : total_distance = 2.1) 
  (h2 : time_limit = 18) 
  (h3 : walking_speed = 90) 
  (h4 : running_speed = 210) :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ time_limit ∧ 
  running_speed * x + walking_speed * (time_limit - x) ≥ total_distance * 1000 :=
sorry

end NUMINAMATH_CALUDE_min_running_time_l3547_354729


namespace NUMINAMATH_CALUDE_cos_product_equality_l3547_354756

theorem cos_product_equality : 
  Real.cos (π / 9) * Real.cos (2 * π / 9) * Real.cos (-23 * π / 9) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_equality_l3547_354756


namespace NUMINAMATH_CALUDE_inequality_implies_a_value_l3547_354747

theorem inequality_implies_a_value (a : ℝ) 
  (h : ∀ x : ℝ, x > 0 → (x^2 + a*x - 5)*(a*x - 1) ≥ 0) : 
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_value_l3547_354747


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_396_l3547_354795

def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_of_distinct_prime_factors_396 :
  sum_of_distinct_prime_factors 396 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_396_l3547_354795


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3547_354718

theorem polynomial_simplification (x : ℝ) : 
  (6 * x^10 + 8 * x^9 + 3 * x^7) + (2 * x^12 + 3 * x^10 + x^9 + 5 * x^7 + 4 * x^4 + 7 * x + 6) = 
  2 * x^12 + 9 * x^10 + 9 * x^9 + 8 * x^7 + 4 * x^4 + 7 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3547_354718


namespace NUMINAMATH_CALUDE_right_triangle_trig_sum_l3547_354722

theorem right_triangle_trig_sum (A B C : Real) : 
  -- Conditions
  A = π / 2 →  -- A = 90° in radians
  0 < B → B < π / 2 →  -- B is acute angle
  C = π / 2 - B →  -- C is complementary to B in right triangle
  -- Theorem
  Real.sin A + Real.sin B ^ 2 + Real.sin C ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_sum_l3547_354722


namespace NUMINAMATH_CALUDE_odd_function_sum_l3547_354735

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_property : ∀ x, f (2 + x) + f (2 - x) = 0)
  (h_f1 : f 1 = 9) :
  f 2010 + f 2011 + f 2012 = -9 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l3547_354735


namespace NUMINAMATH_CALUDE_plant_species_numbering_impossibility_l3547_354710

theorem plant_species_numbering_impossibility :
  ∃ (a b : ℕ), 2 ≤ a ∧ a < b ∧ b ≤ 20000 ∧
  (∀ (x : ℕ), 2 ≤ x ∧ x ≤ 20000 →
    (Nat.gcd a x = Nat.gcd b x)) :=
sorry

end NUMINAMATH_CALUDE_plant_species_numbering_impossibility_l3547_354710


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3547_354769

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set M
def M : Set Int := {-1, 0, 1, 3}

-- Define set N
def N : Set Int := {-2, 0, 2, 3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3547_354769


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3547_354739

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_negative_two :
  reciprocal (-2) = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3547_354739


namespace NUMINAMATH_CALUDE_geometric_sequence_iff_k_eq_neg_one_l3547_354772

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def S (n : ℕ) (k : ℝ) : ℝ := 3^n + k

def a (n : ℕ) (k : ℝ) : ℝ :=
  if n = 1 then S 1 k else S n k - S (n-1) k

theorem geometric_sequence_iff_k_eq_neg_one (k : ℝ) :
  is_geometric_sequence (a · k) ↔ k = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_iff_k_eq_neg_one_l3547_354772


namespace NUMINAMATH_CALUDE_optimal_position_C_l3547_354787

/-- The optimal position of point C on segment AB to maximize the length of CD -/
theorem optimal_position_C (t : ℝ) : 
  (0 ≤ t) → (t < 1) → 
  (∀ s, (0 ≤ s ∧ s < 1) → (t * (1 - t^2) / 4 ≥ s * (1 - s^2) / 4)) → 
  t = 1 / Real.sqrt 3 := by
  sorry

#check optimal_position_C

end NUMINAMATH_CALUDE_optimal_position_C_l3547_354787


namespace NUMINAMATH_CALUDE_expression_simplification_l3547_354706

theorem expression_simplification (a b : ℝ) (h : a / b = 1 / 3) :
  1 - (a - b) / (a - 2 * b) / ((a^2 - b^2) / (a^2 - 4 * a * b + 4 * b^2)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3547_354706


namespace NUMINAMATH_CALUDE_all_subjects_identified_l3547_354742

theorem all_subjects_identified (num_colors : ℕ) (num_subjects : ℕ) : 
  num_colors = 5 → num_subjects = 16 → num_colors ^ 2 ≥ num_subjects := by
  sorry

#check all_subjects_identified

end NUMINAMATH_CALUDE_all_subjects_identified_l3547_354742


namespace NUMINAMATH_CALUDE_rhombus_area_from_square_midpoints_l3547_354719

/-- The area of a rhombus formed by connecting the midpoints of a square with side length 4 is 8 -/
theorem rhombus_area_from_square_midpoints (s : ℝ) (h : s = 4) : 
  let rhombus_area := s^2 / 2
  rhombus_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_from_square_midpoints_l3547_354719


namespace NUMINAMATH_CALUDE_total_dolls_count_l3547_354737

def grandmother_dolls : ℕ := 50

def sister_dolls (grandmother_dolls : ℕ) : ℕ := grandmother_dolls + 2

def rene_dolls (sister_dolls : ℕ) : ℕ := 3 * sister_dolls

def total_dolls (grandmother_dolls sister_dolls rene_dolls : ℕ) : ℕ :=
  grandmother_dolls + sister_dolls + rene_dolls

theorem total_dolls_count :
  total_dolls grandmother_dolls (sister_dolls grandmother_dolls) (rene_dolls (sister_dolls grandmother_dolls)) = 258 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_count_l3547_354737


namespace NUMINAMATH_CALUDE_late_average_speed_l3547_354713

/-- Proves that the late average speed is 50 kmph given the problem conditions -/
theorem late_average_speed (journey_length : ℝ) (on_time_speed : ℝ) (late_time : ℝ) :
  journey_length = 225 →
  on_time_speed = 60 →
  late_time = 0.75 →
  ∃ v : ℝ, journey_length / on_time_speed + late_time = journey_length / v ∧ v = 50 :=
by sorry

end NUMINAMATH_CALUDE_late_average_speed_l3547_354713


namespace NUMINAMATH_CALUDE_divisibility_condition_solutions_l3547_354796

theorem divisibility_condition_solutions (a b : ℕ+) :
  (a ^ 2017 + b : ℤ) % (a * b : ℤ) = 0 → (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 2 ^ 2017) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_solutions_l3547_354796


namespace NUMINAMATH_CALUDE_max_balls_in_cube_l3547_354712

theorem max_balls_in_cube (cube_side : ℝ) (ball_radius : ℝ) :
  cube_side = 9 →
  ball_radius = 3 →
  ⌊(cube_side^3) / ((4/3) * π * ball_radius^3)⌋ = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_balls_in_cube_l3547_354712


namespace NUMINAMATH_CALUDE_fraction_simplification_l3547_354733

theorem fraction_simplification :
  (30 : ℚ) / 35 * 21 / 45 * 70 / 63 - 2 / 3 = -8 / 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3547_354733


namespace NUMINAMATH_CALUDE_smallest_number_of_guesses_l3547_354703

def is_determinable (guesses : List Nat) : Prop :=
  ∀ N : Nat, 1 < N → N < 100 → 
    ∃! N', 1 < N' → N' < 100 → 
      ∀ g ∈ guesses, g % N = g % N'

theorem smallest_number_of_guesses :
  ∃ guesses : List Nat,
    guesses.length = 6 ∧
    is_determinable guesses ∧
    ∀ guesses' : List Nat, guesses'.length < 6 → ¬is_determinable guesses' :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_guesses_l3547_354703


namespace NUMINAMATH_CALUDE_birdseed_mix_proportion_l3547_354776

/-- Proves that the proportion of Brand A in a birdseed mix is 60% when the mix is 50% sunflower -/
theorem birdseed_mix_proportion :
  ∀ (x : ℝ), 
  x ≥ 0 ∧ x ≤ 1 →  -- x represents the proportion of Brand A in the mix
  0.60 * x + 0.35 * (1 - x) = 0.50 →  -- The mix is 50% sunflower
  x = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_birdseed_mix_proportion_l3547_354776


namespace NUMINAMATH_CALUDE_charles_whistle_count_l3547_354759

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference between Sean's and Charles' whistles -/
def whistle_difference : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - whistle_difference

theorem charles_whistle_count : charles_whistles = 13 := by
  sorry

end NUMINAMATH_CALUDE_charles_whistle_count_l3547_354759


namespace NUMINAMATH_CALUDE_carol_rectangle_length_l3547_354770

theorem carol_rectangle_length 
  (carol_width jordan_length jordan_width : ℕ) 
  (h1 : carol_width = 24)
  (h2 : jordan_length = 8)
  (h3 : jordan_width = 15)
  (h4 : carol_width * carol_length = jordan_length * jordan_width) :
  carol_length = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_carol_rectangle_length_l3547_354770


namespace NUMINAMATH_CALUDE_fourth_day_jumps_l3547_354736

def jump_count (day : ℕ) : ℕ :=
  match day with
  | 0 => 0  -- day 0 is not defined in the problem, so we set it to 0
  | 1 => 15 -- first day
  | n + 1 => 2 * jump_count n -- subsequent days

theorem fourth_day_jumps :
  jump_count 4 = 120 :=
by sorry

end NUMINAMATH_CALUDE_fourth_day_jumps_l3547_354736


namespace NUMINAMATH_CALUDE_function_range_iff_a_ge_one_l3547_354791

/-- Given a real number a, the function f(x) = √((a-1)x² + ax + 1) has range [0, +∞) if and only if a ≥ 1 -/
theorem function_range_iff_a_ge_one (a : ℝ) :
  (Set.range (fun x => Real.sqrt ((a - 1) * x^2 + a * x + 1)) = Set.Ici 0) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_iff_a_ge_one_l3547_354791


namespace NUMINAMATH_CALUDE_janabel_widget_sales_l3547_354781

def arithmeticSequenceSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem janabel_widget_sales : 
  let a₁ : ℕ := 2  -- First day sales
  let d : ℕ := 3   -- Daily increase
  let n : ℕ := 15  -- Number of days
  let bonus : ℕ := 1  -- Bonus widget on last day
  arithmeticSequenceSum a₁ d n + bonus = 346 :=
by
  sorry

#check janabel_widget_sales

end NUMINAMATH_CALUDE_janabel_widget_sales_l3547_354781


namespace NUMINAMATH_CALUDE_sqrt_seven_minus_fraction_greater_than_reciprocal_l3547_354765

theorem sqrt_seven_minus_fraction_greater_than_reciprocal 
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : Real.sqrt 7 - m / n > 0) : 
  Real.sqrt 7 - m / n > 1 / (m * n) := by
sorry

end NUMINAMATH_CALUDE_sqrt_seven_minus_fraction_greater_than_reciprocal_l3547_354765


namespace NUMINAMATH_CALUDE_total_pepper_pieces_l3547_354726

-- Define the number of bell peppers
def num_peppers : ℕ := 5

-- Define the number of large slices per pepper
def slices_per_pepper : ℕ := 20

-- Define the number of smaller pieces each half-slice is cut into
def smaller_pieces_per_slice : ℕ := 3

-- Theorem to prove
theorem total_pepper_pieces :
  let total_large_slices := num_peppers * slices_per_pepper
  let half_large_slices := total_large_slices / 2
  let smaller_pieces := half_large_slices * smaller_pieces_per_slice
  let remaining_large_slices := total_large_slices - half_large_slices
  remaining_large_slices + smaller_pieces = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_pepper_pieces_l3547_354726


namespace NUMINAMATH_CALUDE_cubic_fraction_factorization_l3547_354758

theorem cubic_fraction_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) 
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_factorization_l3547_354758


namespace NUMINAMATH_CALUDE_exists_easy_a_difficult_b_l3547_354738

structure TestConfiguration where
  variants : Type
  students : Type
  problems : Type
  solved : variants → students → problems → Prop

def easy_a (tc : TestConfiguration) : Prop :=
  ∀ v : tc.variants, ∀ p : tc.problems, ∃ s : tc.students, tc.solved v s p

def difficult_b (tc : TestConfiguration) : Prop :=
  ∀ v : tc.variants, ¬∃ s : tc.students, ∀ p : tc.problems, tc.solved v s p

theorem exists_easy_a_difficult_b :
  ∃ tc : TestConfiguration, easy_a tc ∧ difficult_b tc := by
  sorry

end NUMINAMATH_CALUDE_exists_easy_a_difficult_b_l3547_354738


namespace NUMINAMATH_CALUDE_square_greater_than_negative_l3547_354766

theorem square_greater_than_negative (x : ℝ) : x < 0 → x^2 > x := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_negative_l3547_354766


namespace NUMINAMATH_CALUDE_tips_fraction_l3547_354749

/-- Given a worker who works for 7 months, with one month's tips being twice
    the average of the other 6 months, the fraction of total tips from that
    one month is 1/4. -/
theorem tips_fraction (total_months : ℕ) (special_month_tips : ℝ) 
    (other_months_tips : ℝ) : 
    total_months = 7 →
    special_month_tips = 2 * (other_months_tips / 6) →
    special_month_tips / (special_month_tips + other_months_tips) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_tips_fraction_l3547_354749


namespace NUMINAMATH_CALUDE_set_with_gcd_property_has_power_of_two_elements_l3547_354744

theorem set_with_gcd_property_has_power_of_two_elements (S : Finset ℕ+) 
  (h : ∀ (s : ℕ+) (d : ℕ+), s ∈ S → d ∣ s → ∃! (t : ℕ+), t ∈ S ∧ Nat.gcd s.val t.val = d.val) :
  ∃ (k : ℕ), S.card = 2^k :=
sorry

end NUMINAMATH_CALUDE_set_with_gcd_property_has_power_of_two_elements_l3547_354744


namespace NUMINAMATH_CALUDE_danny_bottle_caps_count_l3547_354717

/-- The number of bottle caps Danny has in his collection now -/
def danny_bottle_caps : ℕ := 56

/-- The number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 46

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 50

/-- The number of wrappers Danny has in his collection now -/
def wrappers_in_collection : ℕ := 52

theorem danny_bottle_caps_count :
  danny_bottle_caps = wrappers_in_collection + (bottle_caps_found - wrappers_found) :=
by sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_count_l3547_354717


namespace NUMINAMATH_CALUDE_overlap_length_l3547_354700

/-- Given a set of overlapping red segments, this theorem proves the length of each overlap. -/
theorem overlap_length (total_length : ℝ) (end_to_end : ℝ) (num_overlaps : ℕ) : 
  total_length = 98 →
  end_to_end = 83 →
  num_overlaps = 6 →
  (total_length - end_to_end) / num_overlaps = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_length_l3547_354700


namespace NUMINAMATH_CALUDE_cubic_equation_with_double_root_l3547_354751

theorem cubic_equation_with_double_root (k : ℝ) : 
  (∃ a b : ℝ, (3 * a^3 - 9 * a^2 - 81 * a + k = 0) ∧ 
               (3 * (2*a)^3 - 9 * (2*a)^2 - 81 * (2*a) + k = 0) ∧ 
               (3 * b^3 - 9 * b^2 - 81 * b + k = 0) ∧ 
               (a ≠ b) ∧ (k > 0)) →
  k = -6 * ((9 + Real.sqrt 837) / 14)^2 * (3 - 3 * ((9 + Real.sqrt 837) / 14)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_with_double_root_l3547_354751


namespace NUMINAMATH_CALUDE_bar_chart_clarity_l3547_354748

/-- Represents a bar chart --/
structure BarChart where
  data : List (String × ℝ)

/-- Represents the clarity of quantity representation in a chart --/
def ClearQuantityRepresentation : Prop := True

/-- Theorem: A bar chart clearly shows the amount of each quantity it represents --/
theorem bar_chart_clarity (chart : BarChart) : ClearQuantityRepresentation := by
  sorry

end NUMINAMATH_CALUDE_bar_chart_clarity_l3547_354748


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l3547_354743

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a3 : a 3 = 2)
  (h_d : ∃ d : ℚ, d = -1/2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l3547_354743


namespace NUMINAMATH_CALUDE_least_period_scaled_least_period_sum_sine_cosine_least_period_sin_cos_least_period_cos_sin_l3547_354755

-- Definition of periodic function
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Definition of least period
def least_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  is_periodic f T ∧ ∀ T', 0 < T' ∧ T' < T → ¬ is_periodic f T'

-- Theorem 1
theorem least_period_scaled (g : ℝ → ℝ) :
  least_period g π → least_period (fun x ↦ g (x / 3)) (3 * π) := by sorry

-- Theorem 2
theorem least_period_sum_sine_cosine :
  least_period (fun x ↦ Real.sin (8 * x) + Real.cos (4 * x)) (π / 2) := by sorry

-- Theorem 3
theorem least_period_sin_cos :
  least_period (fun x ↦ Real.sin (Real.cos x)) (2 * π) := by sorry

-- Theorem 4
theorem least_period_cos_sin :
  least_period (fun x ↦ Real.cos (Real.sin x)) π := by sorry

end NUMINAMATH_CALUDE_least_period_scaled_least_period_sum_sine_cosine_least_period_sin_cos_least_period_cos_sin_l3547_354755


namespace NUMINAMATH_CALUDE_negation_of_squared_nonnegative_l3547_354760

theorem negation_of_squared_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_squared_nonnegative_l3547_354760


namespace NUMINAMATH_CALUDE_value_of_A_l3547_354775

-- Define the letter values as variables
variable (M A T H E : ℤ)

-- State the theorem
theorem value_of_A 
  (h_H : H = 8)
  (h_MATH : M + A + T + H = 32)
  (h_TEAM : T + E + A + M = 40)
  (h_MEET : M + E + E + T = 36) :
  A = 20 := by
sorry

end NUMINAMATH_CALUDE_value_of_A_l3547_354775


namespace NUMINAMATH_CALUDE_circles_intersect_l3547_354746

theorem circles_intersect : ∃ (x y : ℝ),
  (x^2 + y^2 - 8*x + 6*y - 11 = 0) ∧ (x^2 + y^2 = 100) := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l3547_354746


namespace NUMINAMATH_CALUDE_geometric_sequence_theorem_l3547_354778

/-- A geometric sequence with positive common ratio -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_theorem (a : ℕ → ℝ) :
  GeometricSequence a →
  a 2 * a 10 = 2 * (a 5)^2 →
  a 2 = 1 →
  ∀ n : ℕ, a n = 2^((n - 2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_theorem_l3547_354778


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_l3547_354721

theorem triangle_inequality_expression (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : 
  (a - b)^2 - c^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_l3547_354721


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3547_354725

/-- A line defined by the equation ax + (2-a)y + 1 = 0 -/
def line (a : ℝ) (x y : ℝ) : Prop := a * x + (2 - a) * y + 1 = 0

/-- The theorem states that for any real number a, 
    the line ax + (2-a)y + 1 = 0 passes through the point (-1/2, -1/2) -/
theorem line_passes_through_fixed_point :
  ∀ a : ℝ, line a (-1/2) (-1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3547_354725


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l3547_354716

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, abs x > 0) ↔ (∀ x : ℝ, ¬(abs x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l3547_354716


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l3547_354709

/-- Given a person's income and savings, calculate the ratio of income to expenditure -/
theorem income_expenditure_ratio 
  (income : ℕ) 
  (savings : ℕ) 
  (h1 : income = 10000) 
  (h2 : savings = 4000) : 
  (income : ℚ) / (income - savings) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l3547_354709


namespace NUMINAMATH_CALUDE_complement_of_P_intersection_P_M_range_of_a_l3547_354790

-- Define the sets P and M
def P : Set ℝ := {x | x * (x - 2) ≥ 0}
def M (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 3}

-- Theorem for the complement of P
theorem complement_of_P : 
  (Set.univ : Set ℝ) \ P = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem for the intersection of P and M when a = 1
theorem intersection_P_M : 
  P ∩ M 1 = {x | 2 ≤ x ∧ x < 4} := by sorry

-- Theorem for the range of a when ∁ₗP ⊆ M
theorem range_of_a (a : ℝ) : 
  ((Set.univ : Set ℝ) \ P) ⊆ M a ↔ -1 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_complement_of_P_intersection_P_M_range_of_a_l3547_354790


namespace NUMINAMATH_CALUDE_cylinder_base_area_l3547_354789

/-- Represents a container with a base area and height increase when a stone is submerged -/
structure Container where
  base_area : ℝ
  height_increase : ℝ

/-- Proves that the base area of the cylinder is 42 square centimeters -/
theorem cylinder_base_area
  (cylinder : Container)
  (prism : Container)
  (h1 : cylinder.height_increase = 8)
  (h2 : prism.height_increase = 6)
  (h3 : cylinder.base_area + prism.base_area = 98)
  : cylinder.base_area = 42 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_base_area_l3547_354789


namespace NUMINAMATH_CALUDE_inequality_proof_l3547_354728

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3547_354728


namespace NUMINAMATH_CALUDE_journey_satisfies_equations_l3547_354753

/-- Represents Li Hai's journey from point A to point B -/
structure Journey where
  totalDistance : ℝ
  uphillSpeed : ℝ
  downhillSpeed : ℝ
  totalTime : ℝ
  uphillTime : ℝ
  downhillTime : ℝ

/-- Checks if the given journey satisfies the system of equations -/
def satisfiesEquations (j : Journey) : Prop :=
  j.uphillTime + j.downhillTime = j.totalTime ∧
  (j.uphillSpeed * j.uphillTime / 60 + j.downhillSpeed * j.downhillTime / 60) * 1000 = j.totalDistance

/-- Theorem stating that Li Hai's journey satisfies the system of equations -/
theorem journey_satisfies_equations :
  ∀ (j : Journey),
    j.totalDistance = 1200 ∧
    j.uphillSpeed = 3 ∧
    j.downhillSpeed = 5 ∧
    j.totalTime = 16 →
    satisfiesEquations j :=
  sorry

#check journey_satisfies_equations

end NUMINAMATH_CALUDE_journey_satisfies_equations_l3547_354753


namespace NUMINAMATH_CALUDE_shoe_probability_l3547_354707

/-- Represents the total number of shoe pairs -/
def total_pairs : ℕ := 16

/-- Represents the number of black shoe pairs -/
def black_pairs : ℕ := 8

/-- Represents the number of brown shoe pairs -/
def brown_pairs : ℕ := 5

/-- Represents the number of white shoe pairs -/
def white_pairs : ℕ := 3

/-- The probability of picking two shoes of the same color with one being left and the other right -/
theorem shoe_probability : 
  (black_pairs * black_pairs + brown_pairs * brown_pairs + white_pairs * white_pairs) / 
  (total_pairs * (2 * total_pairs - 1)) = 49 / 248 := by
  sorry

end NUMINAMATH_CALUDE_shoe_probability_l3547_354707


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l3547_354799

theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) 
  (h1 : girls = 135) (h2 : total_students = 351) :
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l3547_354799


namespace NUMINAMATH_CALUDE_not_p_and_not_q_range_l3547_354745

-- Define proposition p
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (1 - 2*m) + y^2 / (m + 2) = 1

-- Define proposition q
def q (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 - m = 0

-- Theorem statement
theorem not_p_and_not_q_range (m : ℝ) :
  (¬p m ∧ ¬q m) ↔ (m > -2 ∧ m ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_range_l3547_354745
