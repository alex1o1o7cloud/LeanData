import Mathlib

namespace NUMINAMATH_CALUDE_factory_reorganization_l1378_137818

theorem factory_reorganization (workshop1 workshop2 : ℕ) : 
  (workshop1 / 2 + workshop2 / 3 = (workshop1 / 3 + workshop2 / 2) * 8 / 7) →
  (workshop1 + workshop2 - (workshop1 / 2 + workshop2 / 3 + workshop1 / 3 + workshop2 / 2) = 120) →
  (workshop1 = 480 ∧ workshop2 = 240) := by
  sorry

end NUMINAMATH_CALUDE_factory_reorganization_l1378_137818


namespace NUMINAMATH_CALUDE_condition_analysis_l1378_137827

theorem condition_analysis (a : ℝ) : 
  (∀ a, a > 1 → 1/a < 1) ∧ 
  (∃ a, 1/a < 1 ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_analysis_l1378_137827


namespace NUMINAMATH_CALUDE_ratio_of_divisors_sums_l1378_137802

def M : ℕ := 36 * 36 * 98 * 210

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 62 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisors_sums_l1378_137802


namespace NUMINAMATH_CALUDE_sean_charles_whistle_difference_l1378_137821

theorem sean_charles_whistle_difference : 
  ∀ (sean_whistles charles_whistles : ℕ),
    sean_whistles = 223 →
    charles_whistles = 128 →
    sean_whistles - charles_whistles = 95 := by
  sorry

end NUMINAMATH_CALUDE_sean_charles_whistle_difference_l1378_137821


namespace NUMINAMATH_CALUDE_hyperbola_equivalence_l1378_137805

theorem hyperbola_equivalence (x y : ℝ) :
  (4 * x^2 * y^2 = 4 * x * y + 3) ↔ (x * y = 3/2 ∨ x * y = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equivalence_l1378_137805


namespace NUMINAMATH_CALUDE_juice_cost_is_50_l1378_137816

/-- The cost of a candy bar in cents -/
def candy_cost : ℕ := 25

/-- The cost of a piece of chocolate in cents -/
def chocolate_cost : ℕ := 75

/-- The total cost in cents for the purchase -/
def total_cost : ℕ := 11 * 25

/-- The number of candy bars purchased -/
def num_candy : ℕ := 3

/-- The number of chocolate pieces purchased -/
def num_chocolate : ℕ := 2

/-- The number of juice packs purchased -/
def num_juice : ℕ := 1

theorem juice_cost_is_50 :
  ∃ (juice_cost : ℕ),
    juice_cost = 50 ∧
    total_cost = num_candy * candy_cost + num_chocolate * chocolate_cost + num_juice * juice_cost :=
by sorry

end NUMINAMATH_CALUDE_juice_cost_is_50_l1378_137816


namespace NUMINAMATH_CALUDE_simplify_fraction_and_multiply_l1378_137866

theorem simplify_fraction_and_multiply :
  (144 : ℚ) / 1296 * 36 = 4 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_and_multiply_l1378_137866


namespace NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l1378_137861

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the containment relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- Main theorem
theorem perp_planes_necessary_not_sufficient 
  (α β : Plane) (m : Line) 
  (h_different : α ≠ β)
  (h_contained : contained_in m α) :
  (∀ m, contained_in m α → perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m, contained_in m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l1378_137861


namespace NUMINAMATH_CALUDE_green_hat_cost_l1378_137831

/-- Proves that the cost of each green hat is $7 given the conditions of the problem -/
theorem green_hat_cost (total_hats : ℕ) (blue_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  blue_hat_cost = 6 →
  total_price = 550 →
  green_hats = 40 →
  (total_hats - green_hats) * blue_hat_cost + green_hats * 7 = total_price :=
by sorry

end NUMINAMATH_CALUDE_green_hat_cost_l1378_137831


namespace NUMINAMATH_CALUDE_f_lower_bound_l1378_137803

noncomputable section

variables (a x : ℝ)

def f (a x : ℝ) : ℝ := (1/2) * a * x^2 + (2*a - 1) * x - 2 * Real.log x

theorem f_lower_bound (ha : a > 0) (hx : x > 0) :
  f a x ≥ 4 - (5/(2*a)) := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_l1378_137803


namespace NUMINAMATH_CALUDE_bricks_needed_for_wall_l1378_137849

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℝ
  height : ℝ
  thickness : ℝ

/-- Represents the dimensions of the wall -/
structure WallDimensions where
  baseLength : ℝ
  topLength : ℝ
  height : ℝ
  thickness : ℝ

/-- Calculates the number of bricks needed to build the wall -/
def calculateBricksNeeded (brickDim : BrickDimensions) (wallDim : WallDimensions) (mortarThickness : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of bricks needed for the given wall -/
theorem bricks_needed_for_wall 
  (brickDim : BrickDimensions)
  (wallDim : WallDimensions)
  (mortarThickness : ℝ)
  (h1 : brickDim.length = 125)
  (h2 : brickDim.height = 11.25)
  (h3 : brickDim.thickness = 6)
  (h4 : wallDim.baseLength = 800)
  (h5 : wallDim.topLength = 650)
  (h6 : wallDim.height = 600)
  (h7 : wallDim.thickness = 22.5)
  (h8 : mortarThickness = 1.25) :
  calculateBricksNeeded brickDim wallDim mortarThickness = 1036 :=
sorry

end NUMINAMATH_CALUDE_bricks_needed_for_wall_l1378_137849


namespace NUMINAMATH_CALUDE_sum_of_squares_l1378_137895

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1378_137895


namespace NUMINAMATH_CALUDE_unique_tangent_circle_existence_l1378_137856

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given elements
variable (M : Point) -- Given point
variable (O : Point) -- Center of the given circle
variable (r : ℝ) -- Radius of the given circle
variable (N : Point) -- Point on the given circle

-- Define the condition that N is on the given circle
def is_on_circle (P : Point) (C : Circle) : Prop :=
  (P.x - C.center.x)^2 + (P.y - C.center.y)^2 = C.radius^2

-- Define tangency between two circles
def are_tangent (C1 C2 : Circle) : Prop :=
  (C1.center.x - C2.center.x)^2 + (C1.center.y - C2.center.y)^2 = (C1.radius + C2.radius)^2

-- State the theorem
theorem unique_tangent_circle_existence 
  (h_N_on_circle : is_on_circle N { center := O, radius := r }) :
  ∃! C : Circle, (is_on_circle M C) ∧ 
                 (are_tangent C { center := O, radius := r }) ∧ 
                 (is_on_circle N C) := by
  sorry

end NUMINAMATH_CALUDE_unique_tangent_circle_existence_l1378_137856


namespace NUMINAMATH_CALUDE_convex_quad_probability_l1378_137829

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords between n points -/
def total_chords : ℕ := n.choose 2

/-- The probability of forming a convex quadrilateral -/
def prob_convex_quad : ℚ := (n.choose k : ℚ) / (total_chords.choose k : ℚ)

/-- Theorem stating the probability of forming a convex quadrilateral -/
theorem convex_quad_probability : prob_convex_quad = 2 / 585 := by
  sorry

end NUMINAMATH_CALUDE_convex_quad_probability_l1378_137829


namespace NUMINAMATH_CALUDE_project_work_time_l1378_137884

/-- Calculates the time spent working on a project given the total days and nap information -/
def timeSpentWorking (totalDays : ℕ) (numberOfNaps : ℕ) (hoursPerNap : ℕ) : ℕ :=
  totalDays * 24 - numberOfNaps * hoursPerNap

/-- Theorem: Given 4 days and 6 seven-hour naps, the time spent working is 54 hours -/
theorem project_work_time :
  timeSpentWorking 4 6 7 = 54 := by
  sorry

end NUMINAMATH_CALUDE_project_work_time_l1378_137884


namespace NUMINAMATH_CALUDE_function_equation_implies_identity_l1378_137872

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^4 + 4*y^4) = (f (x^2))^2 + 4*y^3 * f y) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_identity_l1378_137872


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1378_137854

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_line_equation :
  ∃ (x0 y0 : ℝ), intersection_point x0 y0 ∧
  ∃ (m : ℝ), perpendicular m (-2) ∧
  ∀ (x y : ℝ), y - y0 = m * (x - x0) ↔ x - 2 * y + 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1378_137854


namespace NUMINAMATH_CALUDE_greatest_area_difference_l1378_137813

/-- A rectangle with integer dimensions and perimeter 200 cm -/
structure Rectangle where
  width : ℕ
  height : ℕ
  perimeter_eq : width * 2 + height * 2 = 200

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- A rectangle with one side of length 80 cm -/
structure DoorRectangle extends Rectangle where
  door_side : width = 80 ∨ height = 80

theorem greatest_area_difference :
  ∃ (r : Rectangle) (d : DoorRectangle),
    ∀ (r' : Rectangle) (d' : DoorRectangle),
      d.area - r.area ≥ d'.area - r'.area ∧
      d.area - r.area = 2300 := by
  sorry

end NUMINAMATH_CALUDE_greatest_area_difference_l1378_137813


namespace NUMINAMATH_CALUDE_triangle_possibilities_l1378_137824

-- Define a matchstick as a unit length
def matchstick_length : ℝ := 1

-- Define the total number of matchsticks
def total_matchsticks : ℕ := 12

-- Define a function to check if three lengths can form a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the types of triangles
def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (c = a ∧ c ≠ b)

def is_equilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

def is_right_angled (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

-- Theorem statement
theorem triangle_possibilities :
  ∃ (a b c : ℝ),
    a + b + c = total_matchsticks * matchstick_length ∧
    is_triangle a b c ∧
    (is_isosceles a b c ∧
     ∃ (d e f : ℝ), d + e + f = total_matchsticks * matchstick_length ∧
       is_triangle d e f ∧ is_equilateral d e f ∧
     ∃ (g h i : ℝ), g + h + i = total_matchsticks * matchstick_length ∧
       is_triangle g h i ∧ is_right_angled g h i) :=
by sorry

end NUMINAMATH_CALUDE_triangle_possibilities_l1378_137824


namespace NUMINAMATH_CALUDE_intersection_points_form_convex_polygon_l1378_137857

/-- Represents a point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an L-shaped figure -/
structure LShape where
  A : Point
  longSegment : List Point
  shortSegment : Point

/-- Represents the problem setup -/
structure ProblemSetup where
  L1 : LShape
  L2 : LShape
  n : ℕ
  intersectionPoints : List Point

/-- Predicate to check if a list of points forms a convex polygon -/
def IsConvexPolygon (points : List Point) : Prop := sorry

/-- Main theorem statement -/
theorem intersection_points_form_convex_polygon (setup : ProblemSetup) :
  IsConvexPolygon setup.intersectionPoints :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_form_convex_polygon_l1378_137857


namespace NUMINAMATH_CALUDE_unique_perpendicular_projection_l1378_137838

-- Define the types for projections and points
def Projection : Type := ℝ → ℝ → ℝ
def Point : Type := ℝ × ℝ × ℝ

-- Define the given projections and intersection points
variable (g' g'' d'' : Projection)
variable (A' A'' : Point)

-- Define the perpendicularity condition
def perpendicular (l1 l2 : Projection) : Prop := sorry

-- Define the intersection condition
def intersect (l1 l2 : Projection) (p : Point) : Prop := sorry

-- Theorem statement
theorem unique_perpendicular_projection :
  ∃! d' : Projection,
    intersect g' d' A' ∧
    intersect g'' d'' A'' ∧
    perpendicular g' d' ∧
    perpendicular g'' d'' :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_projection_l1378_137838


namespace NUMINAMATH_CALUDE_triangle_max_area_l1378_137871

/-- Given a triangle ABC with sides a, b, c, where S = a² - (b-c)² and b + c = 8,
    the maximum value of S is 64/17 -/
theorem triangle_max_area (a b c : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
    (h2 : b + c = 8) (h3 : ∀ S : ℝ, S = a^2 - (b-c)^2) : 
    ∃ (S : ℝ), S ≤ 64/17 ∧ ∃ (a' b' c' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ b' + c' = 8 ∧ 
    64/17 = a'^2 - (b'-c')^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1378_137871


namespace NUMINAMATH_CALUDE_lucas_150_mod_9_l1378_137865

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

theorem lucas_150_mod_9 : lucas 149 % 9 = 3 := by sorry

end NUMINAMATH_CALUDE_lucas_150_mod_9_l1378_137865


namespace NUMINAMATH_CALUDE_speed_increase_time_reduction_l1378_137880

theorem speed_increase_time_reduction 
  (initial_speed : ℝ) 
  (speed_increase : ℝ) 
  (distance : ℝ) 
  (h1 : initial_speed = 30)
  (h2 : speed_increase = 10)
  (h3 : distance > 0) :
  let final_speed := initial_speed + speed_increase
  let initial_time := distance / initial_speed
  let final_time := distance / final_speed
  final_time / initial_time = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_speed_increase_time_reduction_l1378_137880


namespace NUMINAMATH_CALUDE_two_members_absent_l1378_137800

/-- Represents a trivia team with its total members and game performance -/
structure TriviaTeam where
  totalMembers : Float
  totalPoints : Float
  pointsPerMember : Float

/-- Calculates the number of members who didn't show up for a trivia game -/
def membersAbsent (team : TriviaTeam) : Float :=
  team.totalMembers - (team.totalPoints / team.pointsPerMember)

/-- Theorem stating that for the given trivia team, 2 members didn't show up -/
theorem two_members_absent (team : TriviaTeam) 
  (h1 : team.totalMembers = 5.0)
  (h2 : team.totalPoints = 6.0)
  (h3 : team.pointsPerMember = 2.0) : 
  membersAbsent team = 2 := by
  sorry

#eval membersAbsent { totalMembers := 5.0, totalPoints := 6.0, pointsPerMember := 2.0 }

end NUMINAMATH_CALUDE_two_members_absent_l1378_137800


namespace NUMINAMATH_CALUDE_twelve_pointed_stars_count_l1378_137874

/-- Counts the number of non-similar regular n-pointed stars -/
def count_non_similar_stars (n : ℕ) : ℕ :=
  let valid_m := (Finset.range (n - 1)).filter (λ m => m > 1 ∧ m < n - 1 ∧ Nat.gcd m n = 1)
  (valid_m.card + 1) / 2

/-- The number of non-similar regular 12-pointed stars is 1 -/
theorem twelve_pointed_stars_count :
  count_non_similar_stars 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_twelve_pointed_stars_count_l1378_137874


namespace NUMINAMATH_CALUDE_right_triangle_area_l1378_137819

/-- The area of a right triangle with legs 18 and 80 is 720 -/
theorem right_triangle_area : 
  ∀ (a b c : ℝ), 
  a = 18 → b = 80 → c = 82 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 720 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1378_137819


namespace NUMINAMATH_CALUDE_find_c_l1378_137883

theorem find_c (a b c : ℝ) 
  (eq1 : a + b = 3) 
  (eq2 : a * c + b = 18) 
  (eq3 : b * c + a = 6) : 
  c = 7 := by sorry

end NUMINAMATH_CALUDE_find_c_l1378_137883


namespace NUMINAMATH_CALUDE_january_bill_is_120_l1378_137881

/-- Represents the oil bill for a month -/
structure OilBill where
  amount : ℚ

/-- Represents the oil bills for three months -/
structure ThreeMonthBills where
  january : OilBill
  february : OilBill
  march : OilBill

/-- The conditions given in the problem -/
def satisfiesConditions (bills : ThreeMonthBills) : Prop :=
  let j := bills.january.amount
  let f := bills.february.amount
  let m := bills.march.amount
  f / j = 3 / 2 ∧
  f / m = 4 / 5 ∧
  (f + 20) / j = 5 / 3 ∧
  (f + 20) / m = 2 / 3

/-- The theorem to be proved -/
theorem january_bill_is_120 (bills : ThreeMonthBills) :
  satisfiesConditions bills → bills.january.amount = 120 := by
  sorry

end NUMINAMATH_CALUDE_january_bill_is_120_l1378_137881


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1378_137878

theorem right_triangle_side_length 
  (X Y Z : ℝ) 
  (hypotenuse : ℝ) 
  (right_angle : X = 90) 
  (hyp_length : Y - Z = hypotenuse) 
  (hyp_value : hypotenuse = 13) 
  (tan_cos_relation : Real.tan Z = 3 * Real.cos Y) : 
  X - Y = (2 * Real.sqrt 338) / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1378_137878


namespace NUMINAMATH_CALUDE_chloe_trivia_game_score_l1378_137839

/-- Chloe's trivia game score calculation -/
theorem chloe_trivia_game_score (first_round : ℕ) (second_round : ℕ) (final_score : ℕ) 
  (h1 : first_round = 40)
  (h2 : second_round = 50)
  (h3 : final_score = 86) :
  (first_round + second_round) - final_score = 4 := by
  sorry

end NUMINAMATH_CALUDE_chloe_trivia_game_score_l1378_137839


namespace NUMINAMATH_CALUDE_inequality_product_l1378_137828

theorem inequality_product (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a * c > b * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_product_l1378_137828


namespace NUMINAMATH_CALUDE_initial_bottle_caps_l1378_137804

/-- Given the number of bottle caps lost and the final number of bottle caps,
    calculate the initial number of bottle caps. -/
theorem initial_bottle_caps (lost final : ℝ) : 
  lost = 18.0 → final = 45 → lost + final = 63.0 := by sorry

end NUMINAMATH_CALUDE_initial_bottle_caps_l1378_137804


namespace NUMINAMATH_CALUDE_triangle_inequality_l1378_137860

/-- Given a triangle ABC with sides a, b, c, heights h_a, h_b, h_c, area Δ, and a positive real number n,
    the inequality (ah_b)^n + (bh_c)^n + (ch_a)^n ≥ 3 * 2^n * Δ^n holds. -/
theorem triangle_inequality (a b c h_a h_b h_c Δ : ℝ) (n : ℝ) 
    (h_pos : n > 0)
    (h_heights : h_a = 2 * Δ / a ∧ h_b = 2 * Δ / b ∧ h_c = 2 * Δ / c)
    (h_area : Δ = a * h_a / 2 ∧ Δ = b * h_b / 2 ∧ Δ = c * h_c / 2) :
  (a * h_b)^n + (b * h_c)^n + (c * h_a)^n ≥ 3 * 2^n * Δ^n := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1378_137860


namespace NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_geometric_progression_l1378_137893

/-- Given a quadratic equation ax^2 + 2bx + c = 0 with discriminant zero,
    prove that a, b, and c form a geometric progression -/
theorem quadratic_discriminant_zero_implies_geometric_progression
  (a b c : ℝ) (h : a ≠ 0) :
  (2 * b)^2 - 4 * a * c = 0 →
  ∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ c = b * r :=
by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_geometric_progression_l1378_137893


namespace NUMINAMATH_CALUDE_num_divisors_23232_l1378_137841

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- 23232 as a positive integer -/
def n : ℕ+ := 23232

/-- Theorem stating that the number of positive divisors of 23232 is 42 -/
theorem num_divisors_23232 : num_divisors n = 42 := by sorry

end NUMINAMATH_CALUDE_num_divisors_23232_l1378_137841


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l1378_137889

theorem ratio_of_percentages (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hR : R = 0.6 * P)
  (hN : N = 0.75 * R)
  (hP : P ≠ 0) :
  M / N = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l1378_137889


namespace NUMINAMATH_CALUDE_joshua_justin_ratio_l1378_137836

def total_amount : ℝ := 40
def joshua_share : ℝ := 30

theorem joshua_justin_ratio :
  ∃ (k : ℝ), k > 0 ∧ joshua_share = k * (total_amount - joshua_share) →
  joshua_share / (total_amount - joshua_share) = 3 := by
  sorry

end NUMINAMATH_CALUDE_joshua_justin_ratio_l1378_137836


namespace NUMINAMATH_CALUDE_tangent_properties_l1378_137811

/-- The function f(x) = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x^2 + a, where a is a parameter -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The derivative of g(x) -/
def g' (x : ℝ) : ℝ := 2 * x

/-- The tangent line of f at x₁ is also the tangent line of g -/
def tangent_condition (a : ℝ) (x₁ : ℝ) : Prop :=
  ∃ x₂ : ℝ, f' x₁ = g' x₂ ∧ f x₁ + f' x₁ * (x₂ - x₁) = g a x₂

theorem tangent_properties :
  (tangent_condition 3 (-1)) ∧
  (∀ a : ℝ, (∃ x₁ : ℝ, tangent_condition a x₁) → a ≥ -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_properties_l1378_137811


namespace NUMINAMATH_CALUDE_games_within_division_is_48_l1378_137858

/-- Represents a baseball league with two divisions -/
structure BaseballLeague where
  /-- Number of games played against each team in the same division -/
  N : ℕ
  /-- Number of games played against each team in the other division -/
  M : ℕ
  /-- N is greater than 2M -/
  h1 : N > 2 * M
  /-- M is greater than 4 -/
  h2 : M > 4
  /-- Total number of games in the schedule is 76 -/
  h3 : 3 * N + 4 * M = 76

/-- The number of games a team plays within its own division -/
def gamesWithinDivision (league : BaseballLeague) : ℕ := 3 * league.N

/-- Theorem stating that the number of games within division is 48 -/
theorem games_within_division_is_48 (league : BaseballLeague) :
  gamesWithinDivision league = 48 := by
  sorry


end NUMINAMATH_CALUDE_games_within_division_is_48_l1378_137858


namespace NUMINAMATH_CALUDE_train_journey_time_l1378_137882

/-- Proves that if a train moving at 6/7 of its usual speed is 10 minutes late, then its usual journey time is 1 hour -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 → usual_time > 0 →
  (6 / 7 * usual_speed) * (usual_time + 1 / 6) = usual_speed * usual_time →
  usual_time = 1 := by
sorry

end NUMINAMATH_CALUDE_train_journey_time_l1378_137882


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_seven_l1378_137850

/-- The coefficient of x^2 in the expansion of (1 - 3x)^7 -/
def coefficient_x_squared : ℕ :=
  Nat.choose 7 6

/-- Theorem: The coefficient of x^2 in the expansion of (1 - 3x)^7 is 7 -/
theorem coefficient_x_squared_is_seven : coefficient_x_squared = 7 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_seven_l1378_137850


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l1378_137899

theorem square_of_real_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l1378_137899


namespace NUMINAMATH_CALUDE_class_mean_score_l1378_137862

theorem class_mean_score (total_students : ℕ) (first_day_students : ℕ) (second_day_students : ℕ)
  (first_day_mean : ℚ) (second_day_mean : ℚ) :
  total_students = first_day_students + second_day_students →
  first_day_students = 54 →
  second_day_students = 6 →
  first_day_mean = 76 / 100 →
  second_day_mean = 82 / 100 →
  let new_class_mean := (first_day_students * first_day_mean + second_day_students * second_day_mean) / total_students
  new_class_mean = 766 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_class_mean_score_l1378_137862


namespace NUMINAMATH_CALUDE_system_solution_l1378_137864

theorem system_solution :
  ∃ (x y : ℝ), x + y = 5 ∧ 2 * x - y = 1 ∧ x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1378_137864


namespace NUMINAMATH_CALUDE_more_24_than_32_placements_l1378_137870

/-- Represents a chessboard configuration --/
structure Chessboard :=
  (size : Nat)
  (dominoes : Nat)

/-- Represents the number of ways to place dominoes on a chessboard --/
def PlacementCount (board : Chessboard) : Nat := sorry

/-- The 8x8 chessboard with 32 dominoes --/
def board32 : Chessboard :=
  { size := 8, dominoes := 32 }

/-- The 8x8 chessboard with 24 dominoes --/
def board24 : Chessboard :=
  { size := 8, dominoes := 24 }

/-- Theorem stating that there are more ways to place 24 dominoes than 32 dominoes --/
theorem more_24_than_32_placements : PlacementCount board24 > PlacementCount board32 := by
  sorry

end NUMINAMATH_CALUDE_more_24_than_32_placements_l1378_137870


namespace NUMINAMATH_CALUDE_smaller_cuboid_width_l1378_137825

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem smaller_cuboid_width :
  let large_cuboid := CuboidDimensions.mk 12 14 10
  let num_smaller_cuboids : ℕ := 56
  let smaller_cuboid_length : ℝ := 5
  let smaller_cuboid_height : ℝ := 2
  let large_volume := cuboidVolume large_cuboid
  let smaller_volume := large_volume / num_smaller_cuboids
  smaller_volume / (smaller_cuboid_length * smaller_cuboid_height) = 3 := by
  sorry

#check smaller_cuboid_width

end NUMINAMATH_CALUDE_smaller_cuboid_width_l1378_137825


namespace NUMINAMATH_CALUDE_speed_time_relationship_l1378_137851

theorem speed_time_relationship (t v : ℝ) : t = 5 * v^2 ∧ t = 20 → v = 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_time_relationship_l1378_137851


namespace NUMINAMATH_CALUDE_expansion_no_constant_term_l1378_137817

def has_no_constant_term (n : ℕ+) : Prop :=
  ∀ k : ℕ, k ≤ n → (1 + k - 4 * (k / 4) ≠ 0 ∧ 2 + k - 4 * (k / 4) ≠ 0)

theorem expansion_no_constant_term (n : ℕ+) (h : 2 ≤ n ∧ n ≤ 7) :
  has_no_constant_term n ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_no_constant_term_l1378_137817


namespace NUMINAMATH_CALUDE_no_polynomial_satisfies_conditions_exists_polynomial_satisfies_modified_conditions_l1378_137877

-- Part 1
theorem no_polynomial_satisfies_conditions :
  ¬(∃ P : ℝ → ℝ, (∀ x : ℝ, Differentiable ℝ P ∧ Differentiable ℝ (deriv P)) ∧
    (∀ x : ℝ, (deriv P) x > (deriv (deriv P)) x ∧ P x > (deriv (deriv P)) x)) :=
sorry

-- Part 2
theorem exists_polynomial_satisfies_modified_conditions :
  ∃ P : ℝ → ℝ, (∀ x : ℝ, Differentiable ℝ P ∧ Differentiable ℝ (deriv P)) ∧
    (∀ x : ℝ, P x > (deriv P) x ∧ P x > (deriv (deriv P)) x) :=
sorry

end NUMINAMATH_CALUDE_no_polynomial_satisfies_conditions_exists_polynomial_satisfies_modified_conditions_l1378_137877


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1378_137897

theorem smallest_x_absolute_value_equation : 
  ∃ x : ℝ, (∀ y : ℝ, |4*y + 9| = 37 → x ≤ y) ∧ |4*x + 9| = 37 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1378_137897


namespace NUMINAMATH_CALUDE_lunch_probability_l1378_137834

def total_school_days : ℕ := 5

def ham_sandwich_days : ℕ := 3
def cake_days : ℕ := 1
def carrot_sticks_days : ℕ := 3

def prob_ham_sandwich : ℚ := ham_sandwich_days / total_school_days
def prob_cake : ℚ := cake_days / total_school_days
def prob_carrot_sticks : ℚ := carrot_sticks_days / total_school_days

theorem lunch_probability : 
  prob_ham_sandwich * prob_cake * prob_carrot_sticks = 3 / 125 := by
  sorry

end NUMINAMATH_CALUDE_lunch_probability_l1378_137834


namespace NUMINAMATH_CALUDE_cos_BHD_value_l1378_137845

/-- A rectangular solid with specific angle conditions -/
structure RectangularSolid where
  /-- Angle DHG is 30 degrees -/
  angle_DHG : ℝ
  angle_DHG_eq : angle_DHG = 30 * π / 180
  /-- Angle FHB is 45 degrees -/
  angle_FHB : ℝ
  angle_FHB_eq : angle_FHB = 45 * π / 180

/-- The cosine of angle BHD in the rectangular solid -/
def cos_BHD (solid : RectangularSolid) : ℝ := sorry

/-- Theorem stating that the cosine of angle BHD is 5√2/12 -/
theorem cos_BHD_value (solid : RectangularSolid) : 
  cos_BHD solid = 5 * Real.sqrt 2 / 12 := by sorry

end NUMINAMATH_CALUDE_cos_BHD_value_l1378_137845


namespace NUMINAMATH_CALUDE_tan_product_squared_l1378_137888

theorem tan_product_squared (a b : ℝ) :
  3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 6 / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_squared_l1378_137888


namespace NUMINAMATH_CALUDE_otimes_nested_l1378_137876

/-- Definition of the ⊗ operation -/
def otimes (g y : ℝ) : ℝ := g^2 + 2*y

/-- Theorem stating the result of g ⊗ (g ⊗ g) -/
theorem otimes_nested (g : ℝ) : otimes g (otimes g g) = g^4 + 4*g^3 + 6*g^2 + 4*g := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_l1378_137876


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1378_137873

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

def num_history_books : ℕ := 4
def num_science_books : ℕ := 6

theorem book_arrangement_theorem :
  factorial 2 * factorial num_history_books * factorial num_science_books = 34560 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1378_137873


namespace NUMINAMATH_CALUDE_x_intercepts_count_l1378_137810

theorem x_intercepts_count : 
  let f (x : ℝ) := (x - 3) * (x^2 + 4*x + 4)
  ∃ (a b : ℝ), a ≠ b ∧ 
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l1378_137810


namespace NUMINAMATH_CALUDE_hidden_dots_count_l1378_137801

def standard_die_sum : ℕ := 21

def visible_faces : List ℕ := [1, 2, 3, 3, 4, 5, 6, 6, 6]

def total_dice : ℕ := 4

theorem hidden_dots_count :
  (total_dice * standard_die_sum) - (visible_faces.sum) = 48 := by
  sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l1378_137801


namespace NUMINAMATH_CALUDE_opposite_numbers_pairs_l1378_137891

theorem opposite_numbers_pairs (a b : ℝ) (h1 : a + b = 0) (h2 : a ≠ 0) :
  ((-a) + b ≠ 0) ∧
  ((-a) + (-b) = 0) ∧
  (|a| + |b| ≠ 0) ∧
  (a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_pairs_l1378_137891


namespace NUMINAMATH_CALUDE_inequality_proof_l1378_137867

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x^3 / ((1+y)*(1+z))) + (y^3 / ((1+z)*(1+x))) + (z^3 / ((1+x)*(1+y))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1378_137867


namespace NUMINAMATH_CALUDE_sequence_properties_l1378_137885

def S (n : ℕ) : ℤ := n^2 - 9*n

def a (n : ℕ) : ℤ := 2*n - 10

theorem sequence_properties :
  (∀ n, S (n+1) - S n = a (n+1)) ∧
  (∃! k : ℕ, k > 0 ∧ 5 < a k ∧ a k < 8) ∧
  (∀ k : ℕ, k > 0 ∧ 5 < a k ∧ a k < 8 → k = 8) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1378_137885


namespace NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l1378_137898

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 7

-- Theorem statement
theorem g_zero_at_seven_fifths : g (7 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l1378_137898


namespace NUMINAMATH_CALUDE_factor_w4_minus_81_l1378_137846

theorem factor_w4_minus_81 (w : ℝ) : 
  w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) ∧ 
  (∀ (p q : ℝ → ℝ) (a b c : ℝ), (w^4 - 81 = p w * q w ∧ p a = 0 ∧ q b = 0) → 
    (c = 3 ∨ c = -3 ∨ (c^2 = -9 ∧ (∀ x : ℝ, x^2 ≠ -9)))) := by
  sorry

end NUMINAMATH_CALUDE_factor_w4_minus_81_l1378_137846


namespace NUMINAMATH_CALUDE_price_change_theorem_l1378_137820

theorem price_change_theorem (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_increase := initial_price * (1 + 0.34)
  let price_after_first_discount := price_after_increase * (1 - 0.10)
  let final_price := price_after_first_discount * (1 - 0.15)
  let percentage_change := (final_price - initial_price) / initial_price * 100
  percentage_change = 2.51 := by sorry

end NUMINAMATH_CALUDE_price_change_theorem_l1378_137820


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l1378_137835

/-- The range of c for which there are four points on the circle x^2 + y^2 = 4
    at a distance of 1 from the line 12x - 5y + c = 0 is (-13, 13) -/
theorem circle_line_distance_range :
  ∀ c : ℝ,
  (∃ (points : Finset (ℝ × ℝ)),
    points.card = 4 ∧
    (∀ (x y : ℝ), (x, y) ∈ points →
      x^2 + y^2 = 4 ∧
      (|12*x - 5*y + c| / Real.sqrt (12^2 + (-5)^2) = 1))) ↔
  -13 < c ∧ c < 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l1378_137835


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1378_137887

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g (x : ℚ) := c * x^3 - 8 * x^2 + d * x - 7
  (g 2 = -7) → (g (-3) = -80) → (c = -47/15 ∧ d = 428/15) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1378_137887


namespace NUMINAMATH_CALUDE_sum_of_negatives_l1378_137894

theorem sum_of_negatives : (-4) + (-6) = -10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_negatives_l1378_137894


namespace NUMINAMATH_CALUDE_oscars_voting_problem_l1378_137848

/-- Represents a film critic's vote --/
structure Vote where
  actor : Nat
  actress : Nat

/-- The problem statement --/
theorem oscars_voting_problem 
  (critics : Finset Vote) 
  (h_count : critics.card = 3366)
  (h_unique : ∀ n : Nat, 1 ≤ n ∧ n ≤ 100 → ∃ v : Vote, (critics.filter (λ x => x.actor = v.actor ∨ x.actress = v.actress)).card = n) :
  ∃ v1 v2 : Vote, v1 ∈ critics ∧ v2 ∈ critics ∧ v1 ≠ v2 ∧ v1.actor = v2.actor ∧ v1.actress = v2.actress :=
sorry

end NUMINAMATH_CALUDE_oscars_voting_problem_l1378_137848


namespace NUMINAMATH_CALUDE_quadratic_no_solution_l1378_137812

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_no_solution (a b c : ℝ) (h_a : a ≠ 0) :
  f a b c 0 = 2 →
  f a b c 1 = 1 →
  f a b c 2 = 2 →
  f a b c 3 = 5 →
  f a b c 4 = 10 →
  ∀ x, f a b c x ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_solution_l1378_137812


namespace NUMINAMATH_CALUDE_production_theorem_l1378_137814

-- Define production lines
structure ProductionLine where
  process_rate : ℝ → ℝ
  inv_process_rate : ℝ → ℝ

-- Define the company
structure Company where
  line_A : ProductionLine
  line_B : ProductionLine

-- Define the problem
def production_problem (c : Company) : Prop :=
  -- Line A processes a tons in (4a+1) hours
  (c.line_A.process_rate = fun a => 4 * a + 1) ∧
  (c.line_A.inv_process_rate = fun t => (t - 1) / 4) ∧
  -- Line B processes b tons in (2b+3) hours
  (c.line_B.process_rate = fun b => 2 * b + 3) ∧
  (c.line_B.inv_process_rate = fun t => (t - 3) / 2) ∧
  -- Day 1: 5 tons allocated with equal processing time
  ∃ (x : ℝ), 0 < x ∧ x < 5 ∧ c.line_A.process_rate x = c.line_B.process_rate (5 - x) ∧
  -- Day 2: 5 tons allocated based on day 1 results, plus m to A and n to B
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧
    c.line_A.process_rate (x + m) = c.line_B.process_rate (5 - x + n) ∧
    c.line_A.process_rate (x + m) ≤ 24 ∧ c.line_B.process_rate (5 - x + n) ≤ 24

-- Theorem to prove
theorem production_theorem (c : Company) :
  production_problem c →
  (∃ (x : ℝ), x = 2 ∧ 5 - x = 3) ∧
  (∃ (m n : ℝ), m / n = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_production_theorem_l1378_137814


namespace NUMINAMATH_CALUDE_bus_wheel_radius_l1378_137840

/-- The radius of a bus wheel given its speed and revolutions per minute -/
theorem bus_wheel_radius 
  (speed_kmh : ℝ) 
  (rpm : ℝ) 
  (h1 : speed_kmh = 66) 
  (h2 : rpm = 70.06369426751593) : 
  ∃ (r : ℝ), abs (r - 2500.57) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_bus_wheel_radius_l1378_137840


namespace NUMINAMATH_CALUDE_fraction_problem_l1378_137853

theorem fraction_problem (x : ℝ) (f : ℝ) (h1 : x = 140) (h2 : 0.65 * x = f * x - 21) : f = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1378_137853


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_equality_l1378_137823

theorem quadratic_roots_sum_equality (b₁ b₂ b₃ : ℝ) : ∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ),
  (x₁ = (-b₁ + 1) / 2 ∧ y₁ = (-b₁ - 1) / 2) ∧
  (x₂ = (-b₂ + 2) / 2 ∧ y₂ = (-b₂ - 2) / 2) ∧
  (x₃ = (-b₃ + 3) / 2 ∧ y₃ = (-b₃ - 3) / 2) ∧
  x₁ + x₂ + x₃ = y₁ + y₂ + y₃ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_equality_l1378_137823


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l1378_137855

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ (y : ℝ), y^2 = x - 2) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l1378_137855


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1378_137847

/-- Given an arithmetic sequence {aₙ} with S₁ = 10 and S₂ = 20, prove that S₁₀ = 100 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  S 1 = 10 →                            -- given S₁ = 10
  S 2 = 20 →                            -- given S₂ = 20
  S 10 = 100 := by                      -- prove S₁₀ = 100
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1378_137847


namespace NUMINAMATH_CALUDE_larger_number_problem_l1378_137896

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 1375)
  (h2 : L = 6 * S + 15) : 
  L = 1647 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1378_137896


namespace NUMINAMATH_CALUDE_max_guaranteed_amount_100_cards_l1378_137808

/-- Represents a set of bank cards with amounts from 1 to n rubles -/
def BankCards (n : ℕ) := Finset (Fin n)

/-- The strategy of requesting a fixed amount from each card -/
def Strategy (n : ℕ) := ℕ

/-- The amount guaranteed to be collected given a strategy -/
def guaranteedAmount (n : ℕ) (s : Strategy n) : ℕ := sorry

/-- The maximum guaranteed amount that can be collected -/
def maxGuaranteedAmount (n : ℕ) : ℕ := sorry

theorem max_guaranteed_amount_100_cards :
  maxGuaranteedAmount 100 = 2550 := by sorry

end NUMINAMATH_CALUDE_max_guaranteed_amount_100_cards_l1378_137808


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_k_is_1_k_range_when_intersection_nonempty_l1378_137852

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B (k : ℝ) : Set ℝ := {x | x ≤ k}

-- Part 1
theorem intersection_A_complement_B_when_k_is_1 :
  A ∩ (Set.univ \ B 1) = {x | 1 < x ∧ x < 3} := by sorry

-- Part 2
theorem k_range_when_intersection_nonempty :
  ∀ k : ℝ, (A ∩ B k).Nonempty → k ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_k_is_1_k_range_when_intersection_nonempty_l1378_137852


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l1378_137859

theorem intersection_point_of_lines (x y : ℝ) :
  (x - 2*y + 7 = 0) ∧ (2*x + y - 1 = 0) ↔ (x = -1 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l1378_137859


namespace NUMINAMATH_CALUDE_M_elements_l1378_137886

def M : Set (ℕ × ℕ) := {p | p.1 + p.2 ≤ 1}

theorem M_elements : M = {(0, 0), (0, 1), (1, 0)} := by
  sorry

end NUMINAMATH_CALUDE_M_elements_l1378_137886


namespace NUMINAMATH_CALUDE_bird_cage_problem_l1378_137826

theorem bird_cage_problem (initial_birds : ℕ) (final_birds : ℕ) : 
  initial_birds = 60 → 
  final_birds = 8 → 
  ∃ (remaining_after_second : ℕ),
    remaining_after_second = initial_birds * (2/3) * (3/5) ∧
    (2/3 : ℚ) = (remaining_after_second - final_birds) / remaining_after_second :=
by sorry

end NUMINAMATH_CALUDE_bird_cage_problem_l1378_137826


namespace NUMINAMATH_CALUDE_new_person_weight_l1378_137822

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 98.6 kg -/
theorem new_person_weight :
  weight_of_new_person 8 4.2 65 = 98.6 := by
  sorry

#eval weight_of_new_person 8 4.2 65

end NUMINAMATH_CALUDE_new_person_weight_l1378_137822


namespace NUMINAMATH_CALUDE_unique_arithmetic_triangle_l1378_137806

/-- A triangle with integer angles in arithmetic progression -/
structure ArithmeticTriangle where
  a : ℕ
  d : ℕ
  sum_180 : a + (a + d) + (a + 2*d) = 180
  distinct : a ≠ a + d ∧ a ≠ a + 2*d ∧ a + d ≠ a + 2*d

/-- Theorem stating there's exactly one valid arithmetic triangle with possibly zero angle -/
theorem unique_arithmetic_triangle : 
  ∃! t : ArithmeticTriangle, t.a = 0 ∨ t.a + t.d = 0 ∨ t.a + 2*t.d = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_arithmetic_triangle_l1378_137806


namespace NUMINAMATH_CALUDE_total_score_l1378_137844

-- Define the players and their scores
def Alex : ℕ := 18
def Sam : ℕ := Alex / 2
def Jon : ℕ := 2 * Sam + 3
def Jack : ℕ := Jon + 5
def Tom : ℕ := Jon + Jack - 4

-- State the theorem
theorem total_score : Alex + Sam + Jon + Jack + Tom = 117 := by
  sorry

end NUMINAMATH_CALUDE_total_score_l1378_137844


namespace NUMINAMATH_CALUDE_abc_value_l1378_137837

theorem abc_value (a b c : ℂ) 
  (eq1 : a * b + 4 * b = -16)
  (eq2 : b * c + 4 * c = -16)
  (eq3 : c * a + 4 * a = -16) :
  a * b * c = 64 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l1378_137837


namespace NUMINAMATH_CALUDE_valid_colorings_l1378_137842

-- Define a color type
inductive Color
| A
| B
| C

-- Define a coloring function type
def Coloring := ℕ → Color

-- Define the condition for a valid coloring
def ValidColoring (f : Coloring) : Prop :=
  ∀ a b c : ℕ, 2000 * (a + b) = c →
    (f a = f b ∧ f b = f c) ∨
    (f a ≠ f b ∧ f b ≠ f c ∧ f a ≠ f c)

-- Define the two valid colorings
def AllSameColor : Coloring :=
  λ _ => Color.A

def ModuloThreeColoring : Coloring :=
  λ n => match n % 3 with
    | 1 => Color.A
    | 2 => Color.B
    | 0 => Color.C
    | _ => Color.A  -- This case is unreachable, but needed for exhaustiveness

-- State the theorem
theorem valid_colorings (f : Coloring) :
  ValidColoring f ↔ (f = AllSameColor ∨ f = ModuloThreeColoring) :=
sorry

end NUMINAMATH_CALUDE_valid_colorings_l1378_137842


namespace NUMINAMATH_CALUDE_student_factor_problem_l1378_137809

theorem student_factor_problem (n : ℝ) (f : ℝ) : n = 124 → n * f - 138 = 110 → f = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_factor_problem_l1378_137809


namespace NUMINAMATH_CALUDE_complex_number_real_imag_equal_l1378_137868

theorem complex_number_real_imag_equal (a : ℝ) : 
  let z : ℂ := a + (Complex.I - 1) / (1 + Complex.I)
  (z.re = z.im) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_real_imag_equal_l1378_137868


namespace NUMINAMATH_CALUDE_school_gender_ratio_l1378_137843

theorem school_gender_ratio (num_boys : ℕ) (num_girls : ℕ) : 
  num_boys = 80 →
  num_boys * 13 = num_girls * 5 →
  num_girls > num_boys →
  num_girls - num_boys = 128 :=
by
  sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l1378_137843


namespace NUMINAMATH_CALUDE_simplify_expression_l1378_137875

theorem simplify_expression (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) + 1 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1378_137875


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1378_137879

-- Define the triangle ABC
theorem triangle_angle_measure (A B C : Real) (a b c : Real) :
  -- Conditions
  (a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = (Real.sqrt 3 / 2) * b) →
  (c > b) →
  -- Conclusion
  B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1378_137879


namespace NUMINAMATH_CALUDE_floss_leftover_result_l1378_137869

/-- Calculates the amount of floss left over after distributing to students --/
def floss_leftover (class1_size class2_size class3_size : ℕ) 
                   (floss_per_student1 floss_per_student2 floss_per_student3 : ℚ) 
                   (yards_per_packet : ℚ) : ℚ :=
  let total_floss_needed := class1_size * floss_per_student1 + 
                            class2_size * floss_per_student2 + 
                            class3_size * floss_per_student3
  let packets_needed := (total_floss_needed / yards_per_packet).ceil
  let total_floss_bought := packets_needed * yards_per_packet
  total_floss_bought - total_floss_needed

/-- Theorem stating the amount of floss left over --/
theorem floss_leftover_result : 
  floss_leftover 20 25 30 (3/2) (7/4) 2 35 = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_floss_leftover_result_l1378_137869


namespace NUMINAMATH_CALUDE_minimum_race_distance_minimum_race_distance_rounded_l1378_137830

/-- The minimum distance a runner must travel in a race with specific conditions -/
theorem minimum_race_distance : ℝ :=
  let wall_distance : ℝ := 1500
  let a_to_first_wall : ℝ := 400
  let b_to_second_wall : ℝ := 600
  let total_vertical_distance : ℝ := a_to_first_wall + wall_distance + b_to_second_wall
  let minimum_distance : ℝ := (wall_distance ^ 2 + total_vertical_distance ^ 2).sqrt
  ⌊minimum_distance + 0.5⌋

/-- The minimum distance rounded to the nearest meter is 2915 -/
theorem minimum_race_distance_rounded : 
  ⌊minimum_race_distance + 0.5⌋ = 2915 := by sorry

end NUMINAMATH_CALUDE_minimum_race_distance_minimum_race_distance_rounded_l1378_137830


namespace NUMINAMATH_CALUDE_mason_savings_l1378_137863

theorem mason_savings (total_savings : ℚ) (days : ℕ) (dime_value : ℚ) : 
  total_savings = 3 → days = 30 → dime_value = 0.1 → 
  (total_savings / days) * dime_value = 0.01 := by
sorry

end NUMINAMATH_CALUDE_mason_savings_l1378_137863


namespace NUMINAMATH_CALUDE_simplified_irrational_expression_l1378_137892

theorem simplified_irrational_expression :
  ∃ (a b c : ℤ), 
    (c > 0) ∧ 
    (∀ (a' b' c' : ℤ), c' > 0 → 
      Real.sqrt 11 + 2 / Real.sqrt 11 + Real.sqrt 2 + 3 / Real.sqrt 2 = (a' * Real.sqrt 11 + b' * Real.sqrt 2) / c' → 
      c ≤ c') ∧
    Real.sqrt 11 + 2 / Real.sqrt 11 + Real.sqrt 2 + 3 / Real.sqrt 2 = (a * Real.sqrt 11 + b * Real.sqrt 2) / c ∧
    a = 11 ∧ b = 44 ∧ c = 22 := by
  sorry

end NUMINAMATH_CALUDE_simplified_irrational_expression_l1378_137892


namespace NUMINAMATH_CALUDE_sqrt_negative_a_squared_plus_one_undefined_l1378_137815

theorem sqrt_negative_a_squared_plus_one_undefined (a : ℝ) : ¬ ∃ (x : ℝ), x^2 = -a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_negative_a_squared_plus_one_undefined_l1378_137815


namespace NUMINAMATH_CALUDE_player_arrangement_count_l1378_137833

def num_players_alpha : ℕ := 4
def num_players_beta : ℕ := 4
def num_players_gamma : ℕ := 2
def total_players : ℕ := num_players_alpha + num_players_beta + num_players_gamma

theorem player_arrangement_count :
  (Nat.factorial 3) * (Nat.factorial num_players_alpha) * (Nat.factorial num_players_beta) * (Nat.factorial num_players_gamma) = 6912 :=
by sorry

end NUMINAMATH_CALUDE_player_arrangement_count_l1378_137833


namespace NUMINAMATH_CALUDE_remainder_problem_l1378_137890

theorem remainder_problem (N : ℤ) (h : N % 350 = 37) : (2 * N) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1378_137890


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l1378_137807

/-- Given vectors a and b in R², if a + b is perpendicular to b, then the second component of a is 8. -/
theorem vector_perpendicular_condition (m : ℝ) : 
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (3, -2)
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2 = 0 → m = 8 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l1378_137807


namespace NUMINAMATH_CALUDE_min_sum_of_coefficients_l1378_137832

theorem min_sum_of_coefficients (a b : ℕ+) (h : 2 * a * 2 + b * 1 = 13) : 
  ∃ (m n : ℕ+), 2 * m * 2 + n * 1 = 13 ∧ m + n ≤ a + b ∧ m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_coefficients_l1378_137832
