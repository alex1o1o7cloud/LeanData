import Mathlib

namespace NUMINAMATH_CALUDE_impossible_all_positive_l3651_365136

/-- Represents a 4x4 grid of integers -/
def Grid := Fin 4 → Fin 4 → Int

/-- The initial grid configuration -/
def initial_grid : Grid :=
  fun i j => if i = 2 ∧ j = 3 then -1 else 1

/-- Represents an operation on the grid -/
inductive Operation
  | row (i : Fin 4)
  | col (j : Fin 4)
  | diag (d : Fin 7)

/-- Applies an operation to a grid -/
def apply_operation (g : Grid) (op : Operation) : Grid :=
  match op with
  | Operation.row i => fun x y => if x = i then -g x y else g x y
  | Operation.col j => fun x y => if y = j then -g x y else g x y
  | Operation.diag d => fun x y => if x + y = d then -g x y else g x y

/-- Applies a sequence of operations to a grid -/
def apply_operations (g : Grid) (ops : List Operation) : Grid :=
  ops.foldl apply_operation g

/-- Predicate to check if all cells in a grid are positive -/
def all_positive (g : Grid) : Prop :=
  ∀ i j, g i j > 0

/-- The main theorem -/
theorem impossible_all_positive (ops : List Operation) :
  ¬(all_positive (apply_operations initial_grid ops)) :=
sorry

end NUMINAMATH_CALUDE_impossible_all_positive_l3651_365136


namespace NUMINAMATH_CALUDE_a_equals_two_l3651_365165

/-- Given that b = 1 and the equation a(3x-2)+b(2x-3)=8x-7 has infinitely many solutions, prove that a = 2 -/
theorem a_equals_two (b : ℝ) (a : ℝ) (h1 : b = 1) 
  (h2 : ∀ x : ℝ, a * (3 * x - 2) + b * (2 * x - 3) = 8 * x - 7) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_two_l3651_365165


namespace NUMINAMATH_CALUDE_worker_weekly_pay_l3651_365186

/-- Worker's weekly pay calculation --/
theorem worker_weekly_pay (regular_rate : ℝ) (total_surveys : ℕ) (cellphone_rate_increase : ℝ) (cellphone_surveys : ℕ) :
  regular_rate = 10 →
  total_surveys = 100 →
  cellphone_rate_increase = 0.3 →
  cellphone_surveys = 60 →
  let non_cellphone_surveys := total_surveys - cellphone_surveys
  let cellphone_rate := regular_rate * (1 + cellphone_rate_increase)
  let non_cellphone_pay := non_cellphone_surveys * regular_rate
  let cellphone_pay := cellphone_surveys * cellphone_rate
  let total_pay := non_cellphone_pay + cellphone_pay
  total_pay = 1180 := by
sorry

end NUMINAMATH_CALUDE_worker_weekly_pay_l3651_365186


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3651_365164

theorem arithmetic_calculation : 10 - 9 + 8 * 7 + 6 - 5 * 4 / 2 + 3 - 1 = 55 := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3651_365164


namespace NUMINAMATH_CALUDE_largest_common_divisor_525_385_l3651_365181

theorem largest_common_divisor_525_385 : Nat.gcd 525 385 = 35 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_525_385_l3651_365181


namespace NUMINAMATH_CALUDE_bug_population_zero_l3651_365194

/-- Represents the bug population and predator actions in Bill's garden --/
structure GardenState where
  initial_bugs : ℕ
  spiders : ℕ
  ladybugs : ℕ
  mantises : ℕ
  spider_eat_rate : ℕ
  ladybug_eat_rate : ℕ
  mantis_eat_rate : ℕ
  first_spray_rate : ℚ
  second_spray_rate : ℚ

/-- Calculates the final bug population after all actions --/
def final_bug_population (state : GardenState) : ℕ :=
  sorry

/-- Theorem stating that the final bug population is 0 --/
theorem bug_population_zero (state : GardenState) 
  (h1 : state.initial_bugs = 400)
  (h2 : state.spiders = 12)
  (h3 : state.ladybugs = 5)
  (h4 : state.mantises = 8)
  (h5 : state.spider_eat_rate = 7)
  (h6 : state.ladybug_eat_rate = 6)
  (h7 : state.mantis_eat_rate = 4)
  (h8 : state.first_spray_rate = 4/5)
  (h9 : state.second_spray_rate = 7/10) :
  final_bug_population state = 0 :=
sorry

end NUMINAMATH_CALUDE_bug_population_zero_l3651_365194


namespace NUMINAMATH_CALUDE_max_same_count_2011_grid_max_same_count_2011_grid_achievable_l3651_365153

/-- Represents a configuration of napkins on a grid -/
structure NapkinConfiguration where
  grid_size : Nat
  napkin_size : Nat
  napkins : List (Nat × Nat)  -- List of (row, column) positions of napkin top-left corners

/-- Calculates the maximum number of cells with the same nonzero napkin count -/
def max_same_count (config : NapkinConfiguration) : Nat :=
  sorry

/-- The main theorem stating the maximum number of cells with the same nonzero napkin count -/
theorem max_same_count_2011_grid (config : NapkinConfiguration) 
  (h1 : config.grid_size = 2011)
  (h2 : config.napkin_size = 52) :
  max_same_count config ≤ 1994^2 + 37 * 17^2 :=
sorry

/-- The theorem stating that the upper bound is achievable -/
theorem max_same_count_2011_grid_achievable : 
  ∃ (config : NapkinConfiguration), 
    config.grid_size = 2011 ∧ 
    config.napkin_size = 52 ∧
    max_same_count config = 1994^2 + 37 * 17^2 :=
sorry

end NUMINAMATH_CALUDE_max_same_count_2011_grid_max_same_count_2011_grid_achievable_l3651_365153


namespace NUMINAMATH_CALUDE_taran_number_puzzle_l3651_365154

theorem taran_number_puzzle : ∃ x : ℕ, 
  ((x * 5 + 5 - 5 = 73) ∨ (x * 5 + 5 - 6 = 73) ∨ (x * 5 + 6 - 5 = 73) ∨ (x * 5 + 6 - 6 = 73) ∨
   (x * 6 + 5 - 5 = 73) ∨ (x * 6 + 5 - 6 = 73) ∨ (x * 6 + 6 - 5 = 73) ∨ (x * 6 + 6 - 6 = 73)) ∧
  x = 12 := by
  sorry

end NUMINAMATH_CALUDE_taran_number_puzzle_l3651_365154


namespace NUMINAMATH_CALUDE_isosceles_area_sum_l3651_365105

/-- Represents a right isosceles triangle constructed on a side of a right triangle -/
structure RightIsoscelesTriangle where
  side : ℝ
  area : ℝ

/-- Represents a 5-12-13 right triangle with right isosceles triangles on its sides -/
structure TriangleWithIsosceles where
  short_side1 : RightIsoscelesTriangle
  short_side2 : RightIsoscelesTriangle
  hypotenuse : RightIsoscelesTriangle

/-- The theorem to be proved -/
theorem isosceles_area_sum (t : TriangleWithIsosceles) : 
  t.short_side1.side = 5 ∧ 
  t.short_side2.side = 12 ∧ 
  t.hypotenuse.side = 13 ∧
  t.short_side1.area = (1/2) * t.short_side1.side * t.short_side1.side ∧
  t.short_side2.area = (1/2) * t.short_side2.side * t.short_side2.side ∧
  t.hypotenuse.area = (1/2) * t.hypotenuse.side * t.hypotenuse.side →
  t.short_side1.area + t.short_side2.area = t.hypotenuse.area := by
  sorry

end NUMINAMATH_CALUDE_isosceles_area_sum_l3651_365105


namespace NUMINAMATH_CALUDE_max_value_trigonometric_function_l3651_365139

theorem max_value_trigonometric_function :
  ∀ θ : ℝ, 0 < θ → θ < π / 2 →
  (∀ φ : ℝ, 0 < φ → φ < π / 2 →
    (1 / Real.sin θ - 1) * (1 / Real.cos θ - 1) ≥ (1 / Real.sin φ - 1) * (1 / Real.cos φ - 1)) →
  (1 / Real.sin θ - 1) * (1 / Real.cos θ - 1) = 3 - 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_max_value_trigonometric_function_l3651_365139


namespace NUMINAMATH_CALUDE_right_triangle_check_other_sets_not_right_triangle_l3651_365123

theorem right_triangle_check (a b c : ℝ) : 
  (a = 5 ∧ b = 12 ∧ c = 13) → a^2 + b^2 = c^2 :=
by sorry

theorem other_sets_not_right_triangle :
  ¬(∃ a b c : ℝ, 
    ((a = Real.sqrt 3 ∧ b = Real.sqrt 4 ∧ c = Real.sqrt 5) ∨
     (a = 4 ∧ b = 9 ∧ c = Real.sqrt 13) ∨
     (a = 0.8 ∧ b = 0.15 ∧ c = 0.17)) ∧
    a^2 + b^2 = c^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_check_other_sets_not_right_triangle_l3651_365123


namespace NUMINAMATH_CALUDE_marcus_baseball_cards_l3651_365190

theorem marcus_baseball_cards 
  (initial_cards : ℝ) 
  (cards_from_carter : ℝ) 
  (h1 : initial_cards = 210.0) 
  (h2 : cards_from_carter = 58.0) : 
  initial_cards + cards_from_carter = 268.0 := by
sorry

end NUMINAMATH_CALUDE_marcus_baseball_cards_l3651_365190


namespace NUMINAMATH_CALUDE_ethan_candles_l3651_365175

theorem ethan_candles (beeswax_per_candle : ℕ) (coconut_oil_per_candle : ℕ) (total_weight : ℕ) :
  beeswax_per_candle = 8 →
  coconut_oil_per_candle = 1 →
  total_weight = 63 →
  (total_weight / (beeswax_per_candle + coconut_oil_per_candle) : ℕ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_ethan_candles_l3651_365175


namespace NUMINAMATH_CALUDE_original_number_property_l3651_365141

theorem original_number_property (k : ℕ) : ∃ (N : ℕ), N = 23 * k + 22 ∧ (N + 1) % 23 = 0 := by
  sorry

end NUMINAMATH_CALUDE_original_number_property_l3651_365141


namespace NUMINAMATH_CALUDE_brick_height_l3651_365173

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The height of a rectangular prism with given dimensions -/
theorem brick_height (l w sa : ℝ) (hl : l = 8) (hw : w = 4) (hsa : sa = 112) :
  ∃ h : ℝ, surface_area l w h = sa ∧ h = 2 := by
sorry

end NUMINAMATH_CALUDE_brick_height_l3651_365173


namespace NUMINAMATH_CALUDE_dee_has_least_money_l3651_365120

-- Define the people
inductive Person : Type
  | Ada : Person
  | Ben : Person
  | Cal : Person
  | Dee : Person
  | Eve : Person

-- Define a function to represent the amount of money each person has
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q
axiom cal_more_than_ada_ben : money Person.Cal > money Person.Ada ∧ money Person.Cal > money Person.Ben
axiom ada_eve_more_than_dee : money Person.Ada > money Person.Dee ∧ money Person.Eve > money Person.Dee
axiom ben_between_ada_dee : money Person.Ben > money Person.Dee ∧ money Person.Ben < money Person.Ada

-- Theorem to prove
theorem dee_has_least_money :
  ∀ (p : Person), p ≠ Person.Dee → money Person.Dee < money p :=
sorry

end NUMINAMATH_CALUDE_dee_has_least_money_l3651_365120


namespace NUMINAMATH_CALUDE_negation_of_all_students_punctual_l3651_365169

namespace NegationOfUniversalStatement

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (student : U → Prop)
variable (punctual : U → Prop)

-- State the theorem
theorem negation_of_all_students_punctual :
  (¬ ∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) :=
sorry

end NegationOfUniversalStatement

end NUMINAMATH_CALUDE_negation_of_all_students_punctual_l3651_365169


namespace NUMINAMATH_CALUDE_coupon_one_best_l3651_365122

/-- Represents the discount offered by a coupon given a price --/
def discount (price : ℝ) : ℕ → ℝ
  | 1 => 0.1 * price
  | 2 => 20
  | 3 => 0.18 * (price - 100)
  | _ => 0  -- Default case for invalid coupon numbers

theorem coupon_one_best (price : ℝ) (h : price > 100) :
  (discount price 1 > discount price 2 ∧ discount price 1 > discount price 3) ↔ 
  (200 < price ∧ price < 225) := by
sorry

end NUMINAMATH_CALUDE_coupon_one_best_l3651_365122


namespace NUMINAMATH_CALUDE_sum_of_xyz_is_negative_one_l3651_365109

theorem sum_of_xyz_is_negative_one 
  (x y z : ℝ) 
  (h1 : x*y + x*z + y*z + x + y + z = -3) 
  (h2 : x^2 + y^2 + z^2 = 5) : 
  x + y + z = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_is_negative_one_l3651_365109


namespace NUMINAMATH_CALUDE_no_real_roots_l3651_365126

theorem no_real_roots : ∀ x : ℝ, x^2 - x + 9 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3651_365126


namespace NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l3651_365142

theorem at_least_one_fraction_less_than_two (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l3651_365142


namespace NUMINAMATH_CALUDE_rice_bags_sold_l3651_365178

/-- A trader sells rice bags and restocks. This theorem proves the number of bags sold. -/
theorem rice_bags_sold (initial_stock : ℕ) (restocked : ℕ) (final_stock : ℕ) 
  (h1 : initial_stock = 55)
  (h2 : restocked = 132)
  (h3 : final_stock = 164) :
  initial_stock + restocked - final_stock = 23 := by
  sorry

end NUMINAMATH_CALUDE_rice_bags_sold_l3651_365178


namespace NUMINAMATH_CALUDE_simplify_expression_l3651_365182

theorem simplify_expression (a : ℝ) (h : a ≠ -1) :
  a - 1 + 1 / (a + 1) = a^2 / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3651_365182


namespace NUMINAMATH_CALUDE_restaurant_tip_calculation_l3651_365151

theorem restaurant_tip_calculation 
  (food_cost : ℝ) 
  (service_fee_percentage : ℝ) 
  (total_spent : ℝ) 
  (h1 : food_cost = 50) 
  (h2 : service_fee_percentage = 0.12) 
  (h3 : total_spent = 61) : 
  total_spent - (food_cost + food_cost * service_fee_percentage) = 5 := by
sorry

end NUMINAMATH_CALUDE_restaurant_tip_calculation_l3651_365151


namespace NUMINAMATH_CALUDE_angle_GDA_measure_l3651_365172

-- Define the points
variable (A B C D E F G : Point)

-- Define the shapes
def is_regular_pentagon (C D E : Point) : Prop := sorry

def is_square (A B C D : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (G D A : Point) : ℝ := sorry

-- State the theorem
theorem angle_GDA_measure 
  (h1 : is_regular_pentagon C D E)
  (h2 : is_square A B C D)
  (h3 : is_square D E F G) :
  angle_measure G D A = 72 := by sorry

end NUMINAMATH_CALUDE_angle_GDA_measure_l3651_365172


namespace NUMINAMATH_CALUDE_unique_even_square_Q_l3651_365127

/-- Definition of the polynomial Q --/
def Q (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 3*x + 25

/-- Predicate for x being even --/
def is_even (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k

/-- Theorem stating that there exists exactly one even integer x such that Q(x) is a perfect square --/
theorem unique_even_square_Q : ∃! x : ℤ, is_even x ∧ ∃ y : ℤ, Q x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_even_square_Q_l3651_365127


namespace NUMINAMATH_CALUDE_compute_expression_l3651_365140

theorem compute_expression : 12 + 4 * (5 - 10)^3 = -488 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3651_365140


namespace NUMINAMATH_CALUDE_square_binomial_coefficient_l3651_365102

theorem square_binomial_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 20 * x + 16 = (r * x + s)^2) → 
  a = 25 / 4 :=
by sorry

end NUMINAMATH_CALUDE_square_binomial_coefficient_l3651_365102


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_equality_l3651_365180

/-- Represents the gross domestic product in billions of yuan -/
def gdp : ℝ := 2502.7

/-- The scientific notation representation of the GDP -/
def scientific_notation : ℝ := 2.5027 * (10 ^ 11)

/-- Theorem stating that the GDP in billions of yuan is equal to its scientific notation representation -/
theorem gdp_scientific_notation_equality : gdp * 10^9 = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_equality_l3651_365180


namespace NUMINAMATH_CALUDE_circle_passes_through_P_with_center_C_l3651_365170

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 29

-- Define the center point
def center : ℝ × ℝ := (3, 0)

-- Define the point P
def point_P : ℝ × ℝ := (-2, 2)

-- Theorem statement
theorem circle_passes_through_P_with_center_C :
  circle_equation point_P.1 point_P.2 ∧
  ∀ (x y : ℝ), circle_equation x y → 
    (x - center.1)^2 + (y - center.2)^2 = 
    (point_P.1 - center.1)^2 + (point_P.2 - center.2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_P_with_center_C_l3651_365170


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3651_365108

theorem inequality_equivalence (x y : ℝ) :
  (2 * y - 3 * x > Real.sqrt (9 * x^2)) ↔ ((y > 3 * x ∧ x ≥ 0) ∨ (y > 0 ∧ x < 0)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3651_365108


namespace NUMINAMATH_CALUDE_cyclist_speeds_l3651_365128

-- Define the distance between A and B
def total_distance : ℝ := 240

-- Define the time difference between starts
def start_time_diff : ℝ := 0.5

-- Define the speed difference between cyclists
def speed_diff : ℝ := 3

-- Define the time taken to fix the bike
def fix_time : ℝ := 1.5

-- Define the speeds of cyclists A and B
def speed_A : ℝ := 12
def speed_B : ℝ := speed_A + speed_diff

-- Theorem to prove
theorem cyclist_speeds :
  -- Person B reaches midpoint when bike breaks down
  (total_distance / 2) / speed_B = total_distance / speed_A - start_time_diff - fix_time :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speeds_l3651_365128


namespace NUMINAMATH_CALUDE_central_angle_approx_longitude_diff_l3651_365197

/-- Represents a point on Earth's surface --/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Calculates the central angle between two points on Earth's surface,
    assuming Earth is a perfect sphere --/
def centralAngle (p1 p2 : EarthPoint) : Real :=
  sorry

theorem central_angle_approx_longitude_diff
  (L M : EarthPoint)
  (h1 : L.latitude = 0)
  (h2 : L.longitude = 45)
  (h3 : M.latitude = 23.5)
  (h4 : M.longitude = -90)
  (h5 : abs M.latitude < 30) :
  abs (centralAngle L M - 135) < 5 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_approx_longitude_diff_l3651_365197


namespace NUMINAMATH_CALUDE_max_product_constraint_l3651_365130

theorem max_product_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = 1) :
  x * y ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3651_365130


namespace NUMINAMATH_CALUDE_constant_pace_time_ratio_l3651_365110

/-- Represents a runner with a constant pace -/
structure Runner where
  pace : ℝ  -- pace in minutes per mile

/-- Calculates the time taken to run a given distance -/
def time_to_run (r : Runner) (distance : ℝ) : ℝ :=
  r.pace * distance

theorem constant_pace_time_ratio 
  (r : Runner) 
  (store_distance : ℝ) 
  (store_time : ℝ) 
  (cousin_distance : ℝ) :
  store_distance = 5 →
  store_time = 30 →
  cousin_distance = 2.5 →
  time_to_run r store_distance = store_time →
  time_to_run r cousin_distance = 15 :=
by sorry

end NUMINAMATH_CALUDE_constant_pace_time_ratio_l3651_365110


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3651_365174

theorem quadratic_solution_difference_squared : 
  ∀ f g : ℝ, (4 * f^2 + 8 * f - 48 = 0) → (4 * g^2 + 8 * g - 48 = 0) → (f - g)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3651_365174


namespace NUMINAMATH_CALUDE_triangulation_count_l3651_365179

/-- A convex polygon with interior points and triangulation -/
structure ConvexPolygonWithTriangulation where
  n : ℕ  -- number of vertices in the polygon
  m : ℕ  -- number of interior points
  is_convex : Bool  -- the polygon is convex
  interior_points_are_vertices : Bool  -- each interior point is a vertex of at least one triangle
  vertices_among_given_points : Bool  -- vertices of triangles are among the n+m given points

/-- The number of triangles in the triangulation -/
def num_triangles (p : ConvexPolygonWithTriangulation) : ℕ := p.n + 2 * p.m - 2

/-- Theorem: The number of triangles in the triangulation is n + 2m - 2 -/
theorem triangulation_count (p : ConvexPolygonWithTriangulation) : 
  p.is_convex ∧ p.interior_points_are_vertices ∧ p.vertices_among_given_points →
  num_triangles p = p.n + 2 * p.m - 2 := by
  sorry

end NUMINAMATH_CALUDE_triangulation_count_l3651_365179


namespace NUMINAMATH_CALUDE_line_plane_intersection_l3651_365152

/-- Given a plane α and two intersecting planes that form a line l, 
    prove the direction vector of l and the sine of the angle between l and α. -/
theorem line_plane_intersection (x y z : ℝ) : 
  let α : ℝ → ℝ → ℝ → Prop := λ x y z => x + 2*y - 2*z + 1 = 0
  let plane1 : ℝ → ℝ → ℝ → Prop := λ x y z => x - y + 3 = 0
  let plane2 : ℝ → ℝ → ℝ → Prop := λ x y z => x - 2*z - 1 = 0
  let l : Set (ℝ × ℝ × ℝ) := {p | plane1 p.1 p.2.1 p.2.2 ∧ plane2 p.1 p.2.1 p.2.2}
  let direction_vector : ℝ × ℝ × ℝ := (2, 2, 1)
  let normal_vector : ℝ × ℝ × ℝ := (1, 2, -2)
  let angle_sine : ℝ := 4/9
  (∀ p ∈ l, ∃ t : ℝ, p = (t * direction_vector.1, t * direction_vector.2.1, t * direction_vector.2.2)) ∧
  (|normal_vector.1 * direction_vector.1 + normal_vector.2.1 * direction_vector.2.1 + normal_vector.2.2 * direction_vector.2.2| / 
   (Real.sqrt (normal_vector.1^2 + normal_vector.2.1^2 + normal_vector.2.2^2) * 
    Real.sqrt (direction_vector.1^2 + direction_vector.2.1^2 + direction_vector.2.2^2)) = angle_sine) :=
by sorry

end NUMINAMATH_CALUDE_line_plane_intersection_l3651_365152


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l3651_365107

theorem max_value_of_sum_of_square_roots (a b c : ℝ) : 
  0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 8 → 
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ 9 ∧ 
  ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 8 ∧
    Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l3651_365107


namespace NUMINAMATH_CALUDE_vegetarian_count_l3651_365145

theorem vegetarian_count (non_veg_only : ℕ) (both : ℕ) (total_veg : ℕ) 
  (h1 : non_veg_only = 9)
  (h2 : both = 12)
  (h3 : total_veg = 28) :
  total_veg - both = 16 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_count_l3651_365145


namespace NUMINAMATH_CALUDE_cubic_polynomial_special_case_l3651_365156

-- Define a cubic polynomial
def cubic_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d

-- Define the theorem
theorem cubic_polynomial_special_case (p : ℝ → ℝ) 
  (h_cubic : cubic_polynomial p)
  (h1 : p 1 = 1)
  (h2 : p 2 = 1/8)
  (h3 : p 3 = 1/27) :
  p 4 = 1/576 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_special_case_l3651_365156


namespace NUMINAMATH_CALUDE_optimal_plan_is_most_cost_effective_l3651_365192

/-- Represents a sewage treatment equipment model -/
structure EquipmentModel where
  price : ℕ  -- Price in million yuan
  capacity : ℕ  -- Capacity in tons/month

/-- Represents a purchasing plan -/
structure PurchasePlan where
  modelA : ℕ  -- Number of Model A units
  modelB : ℕ  -- Number of Model B units

def modelA : EquipmentModel := { price := 12, capacity := 240 }
def modelB : EquipmentModel := { price := 10, capacity := 200 }

def totalEquipment : ℕ := 10
def budgetConstraint : ℕ := 105
def minTreatmentCapacity : ℕ := 2040

def totalCost (plan : PurchasePlan) : ℕ :=
  plan.modelA * modelA.price + plan.modelB * modelB.price

def totalCapacity (plan : PurchasePlan) : ℕ :=
  plan.modelA * modelA.capacity + plan.modelB * modelB.capacity

def isValidPlan (plan : PurchasePlan) : Prop :=
  plan.modelA + plan.modelB = totalEquipment ∧
  totalCost plan ≤ budgetConstraint ∧
  totalCapacity plan ≥ minTreatmentCapacity

def optimalPlan : PurchasePlan := { modelA := 1, modelB := 9 }

theorem optimal_plan_is_most_cost_effective :
  isValidPlan optimalPlan ∧
  ∀ plan, isValidPlan plan → totalCost plan ≥ totalCost optimalPlan :=
by sorry

end NUMINAMATH_CALUDE_optimal_plan_is_most_cost_effective_l3651_365192


namespace NUMINAMATH_CALUDE_smallest_x_value_l3651_365131

theorem smallest_x_value (x : ℝ) : 
  ((((5 * x - 20) / (4 * x - 5)) ^ 2 + ((5 * x - 20) / (4 * x - 5))) = 6) →
  x ≥ 35 / 17 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3651_365131


namespace NUMINAMATH_CALUDE_complex_zero_of_polynomial_l3651_365157

def is_valid_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℤ), ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + (P 0)

theorem complex_zero_of_polynomial 
  (P : ℝ → ℝ) 
  (h_valid : is_valid_polynomial P) 
  (h_zero1 : P 3 = 0) 
  (h_zero2 : P (-1) = 0) : 
  P (3/2) = (15 : ℝ)/4 :=
sorry

end NUMINAMATH_CALUDE_complex_zero_of_polynomial_l3651_365157


namespace NUMINAMATH_CALUDE_cycling_equation_correct_l3651_365125

/-- Represents the cycling speeds and time difference between two cyclists A and B. -/
structure CyclingProblem where
  distance : ℝ  -- Distance between points A and B
  speed_diff : ℝ  -- Speed difference between A and B
  time_diff : ℝ  -- Time difference of arrival (in hours)

/-- Checks if the given equation correctly represents the cycling problem. -/
def is_correct_equation (prob : CyclingProblem) (x : ℝ) : Prop :=
  prob.distance / x - prob.distance / (x + prob.speed_diff) = prob.time_diff

/-- The main theorem stating that the given equation correctly represents the cycling problem. -/
theorem cycling_equation_correct : 
  let prob : CyclingProblem := { distance := 30, speed_diff := 3, time_diff := 2/3 }
  ∀ x > 0, is_correct_equation prob x := by
  sorry

end NUMINAMATH_CALUDE_cycling_equation_correct_l3651_365125


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3651_365163

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 2) (h2 : b = 7) (h3 : Odd c) : a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3651_365163


namespace NUMINAMATH_CALUDE_min_days_to_solve_100_problems_l3651_365146

/-- The number of problems solved on day n -/
def problems_solved (n : ℕ) : ℕ := 3^(n-1)

/-- The total number of problems solved up to day n -/
def total_problems (n : ℕ) : ℕ := (3^n - 1) / 2

theorem min_days_to_solve_100_problems :
  ∀ n : ℕ, n > 0 → (total_problems n ≥ 100 ↔ n ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_min_days_to_solve_100_problems_l3651_365146


namespace NUMINAMATH_CALUDE_initial_rate_is_36_l3651_365184

/-- Represents the production of cogs on an assembly line with two phases -/
def cog_production (initial_rate : ℝ) : Prop :=
  let initial_order := 60
  let second_order := 60
  let increased_rate := 60
  let total_cogs := initial_order + second_order
  let initial_time := initial_order / initial_rate
  let second_time := second_order / increased_rate
  let total_time := initial_time + second_time
  let average_output := 45
  (total_cogs / total_time) = average_output

/-- The theorem stating that the initial production rate is 36 cogs per hour -/
theorem initial_rate_is_36 : 
  ∃ (rate : ℝ), cog_production rate ∧ rate = 36 :=
sorry

end NUMINAMATH_CALUDE_initial_rate_is_36_l3651_365184


namespace NUMINAMATH_CALUDE_exercise_band_resistance_l3651_365167

/-- The resistance added by each exercise band -/
def band_resistance : ℝ := sorry

/-- The number of exercise bands -/
def num_bands : ℕ := 2

/-- The weight of the dumbbell in pounds -/
def dumbbell_weight : ℝ := 10

/-- The total squat weight with both sets of bands doubled and the dumbbell -/
def total_squat_weight : ℝ := 30

/-- Theorem stating that each band adds 10 pounds of resistance -/
theorem exercise_band_resistance :
  band_resistance = 10 :=
by sorry

end NUMINAMATH_CALUDE_exercise_band_resistance_l3651_365167


namespace NUMINAMATH_CALUDE_smallest_leftover_four_boxes_l3651_365101

/-- The number of kids among whom the Snackies are distributed -/
def num_kids : ℕ := 8

/-- The number of Snackies left over when one box is divided among the kids -/
def leftover_one_box : ℕ := 5

/-- The number of boxes used in the final distribution -/
def num_boxes : ℕ := 4

/-- Represents the number of Snackies in one box -/
def snackies_per_box : ℕ := num_kids * leftover_one_box + leftover_one_box

theorem smallest_leftover_four_boxes :
  ∃ (leftover : ℕ), leftover < num_kids ∧
  ∃ (pieces_per_kid : ℕ),
    num_boxes * snackies_per_box = num_kids * pieces_per_kid + leftover ∧
    ∀ (smaller_leftover : ℕ),
      smaller_leftover < leftover →
      ¬∃ (alt_pieces_per_kid : ℕ),
        num_boxes * snackies_per_box = num_kids * alt_pieces_per_kid + smaller_leftover :=
by sorry

end NUMINAMATH_CALUDE_smallest_leftover_four_boxes_l3651_365101


namespace NUMINAMATH_CALUDE_pasture_rent_problem_l3651_365119

/-- Represents the rent share of a person -/
structure RentShare where
  oxen : ℕ
  months : ℕ

/-- Calculates the total ox-months for a given rent share -/
def oxMonths (share : RentShare) : ℕ := share.oxen * share.months

/-- The problem statement -/
theorem pasture_rent_problem (a b c : RentShare) (c_rent : ℚ) 
  (h1 : a.oxen = 10 ∧ a.months = 7)
  (h2 : b.oxen = 12 ∧ b.months = 5)
  (h3 : c.oxen = 15 ∧ c.months = 3)
  (h4 : c_rent = 53.99999999999999)
  : ∃ (total_rent : ℚ), total_rent = 210 := by
  sorry

#check pasture_rent_problem

end NUMINAMATH_CALUDE_pasture_rent_problem_l3651_365119


namespace NUMINAMATH_CALUDE_cost_of_paper_towel_package_l3651_365133

/-- The cost of a 12-roll package of paper towels given specific conditions -/
theorem cost_of_paper_towel_package (individual_cost : ℝ) (num_rolls : ℕ) (savings_percent : ℝ) :
  individual_cost = 1 →
  num_rolls = 12 →
  savings_percent = 25 →
  let package_cost := num_rolls * (individual_cost * (1 - savings_percent / 100))
  package_cost = 9 := by
sorry

end NUMINAMATH_CALUDE_cost_of_paper_towel_package_l3651_365133


namespace NUMINAMATH_CALUDE_sum_difference_multiples_l3651_365148

theorem sum_difference_multiples (m n : ℕ+) : 
  (∃ x : ℕ+, m = 101 * x) → 
  (∃ y : ℕ+, n = 63 * y) → 
  m + n = 2018 → 
  m - n = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_difference_multiples_l3651_365148


namespace NUMINAMATH_CALUDE_triangle_vertices_from_midpoints_l3651_365196

/-- Given a triangle with midpoints, prove that its vertices have specific coordinates. -/
theorem triangle_vertices_from_midpoints :
  let m₁ : ℚ × ℚ := (1/4, 13/4)
  let m₂ : ℚ × ℚ := (-1/2, 1)
  let m₃ : ℚ × ℚ := (-5/4, 5/4)
  let v₁ : ℚ × ℚ := (-2, -1)
  let v₂ : ℚ × ℚ := (-1/2, 13/4)
  let v₃ : ℚ × ℚ := (1, 7/2)
  (m₁.1 = (v₂.1 + v₃.1) / 2 ∧ m₁.2 = (v₂.2 + v₃.2) / 2) ∧
  (m₂.1 = (v₁.1 + v₃.1) / 2 ∧ m₂.2 = (v₁.2 + v₃.2) / 2) ∧
  (m₃.1 = (v₁.1 + v₂.1) / 2 ∧ m₃.2 = (v₁.2 + v₂.2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_vertices_from_midpoints_l3651_365196


namespace NUMINAMATH_CALUDE_perimeter_approx_40_l3651_365189

/-- Represents a figure composed of three squares and one rectangle -/
structure CompositeFigure where
  square_side : ℝ
  total_area : ℝ

/-- Checks if the CompositeFigure satisfies the given conditions -/
def is_valid_figure (f : CompositeFigure) : Prop :=
  f.total_area = 150 ∧ 
  3 * f.square_side^2 + 2 * f.square_side^2 = f.total_area

/-- Calculates the perimeter of the CompositeFigure -/
def perimeter (f : CompositeFigure) : ℝ :=
  8 * f.square_side

/-- Theorem stating that the perimeter of a valid CompositeFigure is approximately 40 -/
theorem perimeter_approx_40 (f : CompositeFigure) (h : is_valid_figure f) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ abs (perimeter f - 40) < ε :=
sorry

end NUMINAMATH_CALUDE_perimeter_approx_40_l3651_365189


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3651_365144

theorem geometric_sequence_product (a : ℝ) (r : ℝ) (h1 : a = 8/3) (h2 : a * r^4 = 27/2) :
  (a * r) * (a * r^2) * (a * r^3) = 216 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3651_365144


namespace NUMINAMATH_CALUDE_triangle_with_given_altitudes_is_obtuse_l3651_365113

/-- A triangle with given altitudes --/
structure Triangle where
  alt1 : ℝ
  alt2 : ℝ
  alt3 : ℝ

/-- Definition of an obtuse triangle --/
def isObtuse (t : Triangle) : Prop :=
  ∃ θ : ℝ, θ > Real.pi / 2 ∧ θ < Real.pi ∧
    (Real.cos θ = -(5 : ℝ) / 16)

/-- Theorem: A triangle with altitudes 1/2, 1, and 2/5 is obtuse --/
theorem triangle_with_given_altitudes_is_obtuse :
  let t : Triangle := { alt1 := 1/2, alt2 := 1, alt3 := 2/5 }
  isObtuse t := by
  sorry


end NUMINAMATH_CALUDE_triangle_with_given_altitudes_is_obtuse_l3651_365113


namespace NUMINAMATH_CALUDE_product_reciprocal_sum_l3651_365137

theorem product_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_product : a * b = 16) (h_reciprocal : 1 / a = 3 * (1 / b)) : 
  a + b = 16 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_product_reciprocal_sum_l3651_365137


namespace NUMINAMATH_CALUDE_constant_term_implies_a_value_l3651_365111

/-- 
Given that the constant term in the expansion of (x + a/x)(2x-1)^5 is 30, 
prove that a = 3.
-/
theorem constant_term_implies_a_value (a : ℝ) : 
  (∃ (f : ℝ → ℝ), 
    (∀ x, f x = (x + a/x) * (2*x - 1)^5) ∧ 
    (∃ c, ∀ x, f x = c + x * (f x - c) ∧ c = 30)) → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_constant_term_implies_a_value_l3651_365111


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3651_365116

theorem negation_of_proposition (a b c : ℝ) :
  (¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3)) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3651_365116


namespace NUMINAMATH_CALUDE_initial_sand_calculation_l3651_365191

/-- The amount of sand lost during the trip in pounds -/
def sand_lost : ℝ := 2.4

/-- The amount of sand remaining at arrival in pounds -/
def sand_remaining : ℝ := 1.7

/-- The initial amount of sand on the truck in pounds -/
def initial_sand : ℝ := sand_lost + sand_remaining

theorem initial_sand_calculation : initial_sand = 4.1 := by
  sorry

end NUMINAMATH_CALUDE_initial_sand_calculation_l3651_365191


namespace NUMINAMATH_CALUDE_band_sections_sum_l3651_365106

theorem band_sections_sum (total : ℕ) (trumpet_frac trombone_frac clarinet_frac flute_frac : ℚ) : 
  total = 500 →
  trumpet_frac = 1/2 →
  trombone_frac = 3/25 →
  clarinet_frac = 23/100 →
  flute_frac = 2/25 →
  ⌊total * trumpet_frac⌋ + ⌊total * trombone_frac⌋ + ⌊total * clarinet_frac⌋ + ⌊total * flute_frac⌋ = 465 :=
by sorry

end NUMINAMATH_CALUDE_band_sections_sum_l3651_365106


namespace NUMINAMATH_CALUDE_gcd_45_75_l3651_365155

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_75_l3651_365155


namespace NUMINAMATH_CALUDE_max_divisor_of_prime_sum_l3651_365149

theorem max_divisor_of_prime_sum (a b c : ℕ) : 
  Prime a → Prime b → Prime c → 
  a > 3 → b > 3 → c > 3 →
  2 * a + 5 * b = c →
  (∃ (n : ℕ), n ∣ (a + b + c) ∧ ∀ (m : ℕ), m ∣ (a + b + c) → m ≤ n) →
  (∃ (n : ℕ), n ∣ (a + b + c) ∧ ∀ (m : ℕ), m ∣ (a + b + c) → m ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_max_divisor_of_prime_sum_l3651_365149


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3651_365114

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) ∧ 
  (Real.sqrt ((a + c) * (b + d)) = Real.sqrt (a * b) + Real.sqrt (c * d) ↔ a * d = b * c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3651_365114


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l3651_365138

/-- A quadrilateral inscribed in a circle with given side lengths --/
structure InscribedQuadrilateral where
  radius : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- The theorem stating the relationship between the sides of the inscribed quadrilateral --/
theorem inscribed_quadrilateral_fourth_side 
  (q : InscribedQuadrilateral) 
  (h1 : q.radius = 100 * Real.sqrt 3)
  (h2 : q.side1 = 100)
  (h3 : q.side2 = 150)
  (h4 : q.side3 = 200) :
  q.side4^2 = 35800 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l3651_365138


namespace NUMINAMATH_CALUDE_initial_stock_theorem_l3651_365103

/-- Represents the stock management of a bicycle shop over 3 months -/
structure BikeShop :=
  (mountain_weekly_add : Fin 3 → ℕ)
  (road_weekly_add : ℕ)
  (hybrid_weekly_add : Fin 3 → ℕ)
  (mountain_monthly_sell : ℕ)
  (road_monthly_sell : Fin 3 → ℕ)
  (hybrid_monthly_sell : ℕ)
  (helmet_initial : ℕ)
  (helmet_weekly_add : ℕ)
  (helmet_weekly_sell : ℕ)
  (lock_initial : ℕ)
  (lock_weekly_add : ℕ)
  (lock_weekly_sell : ℕ)
  (final_mountain : ℕ)
  (final_road : ℕ)
  (final_hybrid : ℕ)
  (final_helmet : ℕ)
  (final_lock : ℕ)

/-- The theorem stating the initial stock of bicycles -/
theorem initial_stock_theorem (shop : BikeShop) 
  (h_mountain : shop.mountain_weekly_add = ![6, 4, 3])
  (h_road : shop.road_weekly_add = 4)
  (h_hybrid : shop.hybrid_weekly_add = ![2, 2, 3])
  (h_mountain_sell : shop.mountain_monthly_sell = 12)
  (h_road_sell : shop.road_monthly_sell = ![16, 16, 24])
  (h_hybrid_sell : shop.hybrid_monthly_sell = 10)
  (h_helmet : shop.helmet_initial = 100 ∧ shop.helmet_weekly_add = 10 ∧ shop.helmet_weekly_sell = 15)
  (h_lock : shop.lock_initial = 50 ∧ shop.lock_weekly_add = 5 ∧ shop.lock_weekly_sell = 3)
  (h_final : shop.final_mountain = 75 ∧ shop.final_road = 80 ∧ shop.final_hybrid = 45 ∧ 
             shop.final_helmet = 115 ∧ shop.final_lock = 62) :
  ∃ (initial_mountain initial_road initial_hybrid : ℕ),
    initial_mountain = 59 ∧ 
    initial_road = 88 ∧ 
    initial_hybrid = 47 :=
by sorry

end NUMINAMATH_CALUDE_initial_stock_theorem_l3651_365103


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3651_365162

theorem complex_number_in_third_quadrant : 
  let i : ℂ := Complex.I
  let z : ℂ := i + 2 * i^2 + 3 * i^3
  (z.re < 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3651_365162


namespace NUMINAMATH_CALUDE_flag_distribution_l3651_365195

theorem flag_distribution (total_flags : ℕ) (blue_flags red_flags : ℕ) :
  total_flags % 2 = 0 →
  blue_flags + red_flags = total_flags →
  (3 * total_flags / 10 : ℚ) = blue_flags →
  (3 * total_flags / 10 : ℚ) = red_flags →
  (total_flags / 10 : ℚ) = (blue_flags + red_flags - total_flags / 2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_flag_distribution_l3651_365195


namespace NUMINAMATH_CALUDE_max_distance_ellipse_circle_l3651_365177

/-- The maximum distance between points on an ellipse and a moving circle --/
theorem max_distance_ellipse_circle (a b R : ℝ) (ha : 0 < b) (hab : b < a) (hR : b < R) (hRa : R < a) :
  let ellipse := {p : ℝ × ℝ | (p.1 / a)^2 + (p.2 / b)^2 = 1}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = R^2}
  ∃ (A B : ℝ × ℝ), A ∈ ellipse ∧ B ∈ circle ∧
    (∀ (C : ℝ × ℝ), C ∈ ellipse → (A.1 - C.1) * (B.2 - A.2) = (A.2 - C.2) * (B.1 - A.1)) ∧
    (∀ (D : ℝ × ℝ), D ∈ circle → (B.1 - D.1) * (A.2 - B.2) = (B.2 - D.2) * (A.1 - B.1)) ∧
    ∀ (A' B' : ℝ × ℝ), A' ∈ ellipse → B' ∈ circle →
      (∀ (C : ℝ × ℝ), C ∈ ellipse → (A'.1 - C.1) * (B'.2 - A'.2) = (A'.2 - C.2) * (B'.1 - A'.1)) →
      (∀ (D : ℝ × ℝ), D ∈ circle → (B'.1 - D.1) * (A'.2 - B'.2) = (B'.2 - D.2) * (A'.1 - B'.1)) →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) :=
by
  sorry

#check max_distance_ellipse_circle

end NUMINAMATH_CALUDE_max_distance_ellipse_circle_l3651_365177


namespace NUMINAMATH_CALUDE_prism_volume_l3651_365117

/-- The volume of a triangular prism inscribed in a cylinder -/
theorem prism_volume (H α β : Real) (h_α : 0 < α ∧ α < π / 2) (h_β : 0 < β ∧ β < π / 2) :
  ∃ V : Real,
    V = (H^3 * Real.cos β) / (2 * Real.sin α ^ 2) * Real.sqrt (Real.sin (β + α) * Real.sin (β - α)) ∧
    V > 0 :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_l3651_365117


namespace NUMINAMATH_CALUDE_modular_equivalence_123456_l3651_365199

theorem modular_equivalence_123456 :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_equivalence_123456_l3651_365199


namespace NUMINAMATH_CALUDE_complement_of_M_in_S_l3651_365185

def S : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_S :
  S \ M = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_S_l3651_365185


namespace NUMINAMATH_CALUDE_first_year_interest_l3651_365134

/-- Proves that the interest accrued in the first year is $100 --/
theorem first_year_interest
  (initial_deposit : ℝ)
  (first_year_balance : ℝ)
  (second_year_increase_rate : ℝ)
  (total_increase_rate : ℝ)
  (h1 : initial_deposit = 1000)
  (h2 : first_year_balance = 1100)
  (h3 : second_year_increase_rate = 0.2)
  (h4 : total_increase_rate = 0.32)
  (h5 : first_year_balance * (1 + second_year_increase_rate) = initial_deposit * (1 + total_increase_rate)) :
  first_year_balance - initial_deposit = 100 := by
sorry


end NUMINAMATH_CALUDE_first_year_interest_l3651_365134


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3651_365118

/-- Given that (10, -6) is the midpoint of a line segment with one endpoint at (12, 4),
    prove that the sum of coordinates of the other endpoint is -8. -/
theorem midpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (10 : ℝ) = (x + 12) / 2 →
  (-6 : ℝ) = (y + 4) / 2 →
  x + y = -8 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3651_365118


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3651_365188

/-- A point in a 2D Cartesian plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant in a Cartesian plane. -/
def FourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem stating that the point (1, -1) lies in the fourth quadrant. -/
theorem point_in_fourth_quadrant :
  let A : Point := ⟨1, -1⟩
  FourthQuadrant A := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3651_365188


namespace NUMINAMATH_CALUDE_repeating_decimal_length_seven_thirteenths_l3651_365143

/-- The length of the repeating block in the decimal expansion of 7/13 is 6. -/
theorem repeating_decimal_length_seven_thirteenths : ∃ (d : ℕ+) (n : ℕ),
  (7 : ℚ) / 13 = (n : ℚ) / (10 ^ d.val - 1 : ℚ) ∧ d = 6 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_length_seven_thirteenths_l3651_365143


namespace NUMINAMATH_CALUDE_road_repair_hours_l3651_365193

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 57)
  (h2 : days1 = 12)
  (h3 : people2 = 30)
  (h4 : days2 = 19)
  (h5 : hours2 = 6)
  (h6 : people1 * days1 * (people2 * days2 * hours2) = people2 * days2 * (people1 * days1 * hours2)) :
  ∃ hours1 : ℕ, hours1 = 5 ∧ people1 * days1 * hours1 = people2 * days2 * hours2 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_hours_l3651_365193


namespace NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l3651_365159

/-- Represents a chessboard configuration -/
structure ChessBoard where
  size : Nat
  rooks : Nat

/-- Calculates the minimum number of attacking rook pairs on a chessboard -/
def minAttackingPairs (board : ChessBoard) : Nat :=
  sorry

/-- Theorem: The minimum number of attacking rook pairs on an 8x8 board with 16 rooks is 16 -/
theorem min_attacking_pairs_8x8_16rooks :
  ∀ (board : ChessBoard),
    board.size = 8 ∧ board.rooks = 16 →
    minAttackingPairs board = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l3651_365159


namespace NUMINAMATH_CALUDE_expression_value_l3651_365187

theorem expression_value (x : ℝ) (h : 2 * x^2 - x - 1 = 5) : 6 * x^2 - 3 * x - 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3651_365187


namespace NUMINAMATH_CALUDE_indeterminate_product_sign_l3651_365135

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def continuous_on_interval : Prop :=
  ContinuousOn f (Set.Icc (-2) 2)

def has_root_in_open_interval : Prop :=
  ∃ x, x ∈ Set.Ioo (-2) 2 ∧ f x = 0

-- State the theorem
theorem indeterminate_product_sign
  (h_continuous : continuous_on_interval f)
  (h_root : has_root_in_open_interval f) :
  ¬∃ (sign : ℝ → Prop), ∀ (f : ℝ → ℝ),
    continuous_on_interval f →
    has_root_in_open_interval f →
    sign (f (-2) * f 2) :=
by sorry

end NUMINAMATH_CALUDE_indeterminate_product_sign_l3651_365135


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l3651_365150

-- Define the function f
def f (x : ℝ) : ℝ := (2 + x)^2 - 3*x

-- State the theorem
theorem derivative_f_at_1 :
  deriv f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l3651_365150


namespace NUMINAMATH_CALUDE_product_one_to_six_l3651_365112

theorem product_one_to_six : (List.range 6).foldl (· * ·) 1 = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_one_to_six_l3651_365112


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3651_365168

def in_quadrant_I_or_II (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∨ x < 0 ∧ y > 0

theorem points_in_quadrants_I_and_II (x y : ℝ) :
  y > 3 * x → y > 6 - x^2 → in_quadrant_I_or_II x y := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3651_365168


namespace NUMINAMATH_CALUDE_square_sum_product_l3651_365198

theorem square_sum_product (x : ℝ) :
  (Real.sqrt (8 + x) + Real.sqrt (27 - x) = 9) →
  (8 + x) * (27 - x) = 529 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l3651_365198


namespace NUMINAMATH_CALUDE_count_fours_to_1000_l3651_365129

/-- Count of digit 4 in a single number -/
def count_fours (n : ℕ) : ℕ := sorry

/-- Sum of count_fours for all numbers from 1 to n -/
def total_fours (n : ℕ) : ℕ := sorry

/-- The count of the digit 4 appearing in the integers from 1 to 1000 is equal to 300 -/
theorem count_fours_to_1000 : total_fours 1000 = 300 := by sorry

end NUMINAMATH_CALUDE_count_fours_to_1000_l3651_365129


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l3651_365160

theorem chicken_rabbit_problem (total_animals total_feet : ℕ) 
  (h1 : total_animals = 35)
  (h2 : total_feet = 94) :
  ∃ (chickens rabbits : ℕ), 
    chickens + rabbits = total_animals ∧
    2 * chickens + 4 * rabbits = total_feet ∧
    chickens = 23 := by
  sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l3651_365160


namespace NUMINAMATH_CALUDE_only_valid_k_values_l3651_365124

/-- Represents a line in the form y = kx + b --/
structure Line where
  k : ℤ
  b : ℤ

/-- Represents a parabola in the form y = a(x - c)² --/
structure Parabola where
  a : ℤ
  c : ℤ

/-- Checks if a given k value satisfies all conditions --/
def is_valid_k (k : ℤ) : Prop :=
  ∃ (b : ℤ) (a c : ℤ),
    -- Line passes through (-1, 2020)
    2020 = -k + b ∧
    -- Parabola vertex is on the line
    c = -1 - 2020 / k ∧
    -- a is an integer
    a = k^2 / (2020 + k) ∧
    -- k is negative
    k < 0

/-- The main theorem stating that only -404 and -1010 are valid k values --/
theorem only_valid_k_values :
  ∀ k : ℤ, is_valid_k k ↔ k = -404 ∨ k = -1010 := by sorry

end NUMINAMATH_CALUDE_only_valid_k_values_l3651_365124


namespace NUMINAMATH_CALUDE_unique_function_property_l3651_365104

-- Define the property for the function
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y = f (x - y)

-- Define that the function is not identically zero
def not_zero_function (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x ≠ 0

-- Theorem statement
theorem unique_function_property :
  ∀ f : ℝ → ℝ, satisfies_property f → not_zero_function f →
  ∀ x : ℝ, f x = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_function_property_l3651_365104


namespace NUMINAMATH_CALUDE_semi_circle_perimeter_l3651_365132

/-- The perimeter of a semi-circle with radius 38.50946843518593 cm is 198.03029487037186 cm. -/
theorem semi_circle_perimeter :
  let r : ℝ := 38.50946843518593
  let π : ℝ := Real.pi
  let perimeter : ℝ := π * r + 2 * r
  perimeter = 198.03029487037186 := by sorry

end NUMINAMATH_CALUDE_semi_circle_perimeter_l3651_365132


namespace NUMINAMATH_CALUDE_max_volume_of_prism_l3651_365166

/-- A right prism with a rectangular base -/
structure RectPrism where
  height : ℝ
  base_length : ℝ
  base_width : ℝ

/-- The surface area constraint for the prism -/
def surface_area_constraint (p : RectPrism) : Prop :=
  p.height * p.base_length + p.height * p.base_width + p.base_length * p.base_width = 36

/-- The constraint that base sides are twice the height -/
def base_height_constraint (p : RectPrism) : Prop :=
  p.base_length = 2 * p.height ∧ p.base_width = 2 * p.height

/-- The volume of the prism -/
def volume (p : RectPrism) : ℝ :=
  p.height * p.base_length * p.base_width

/-- The theorem stating the maximum volume of the prism -/
theorem max_volume_of_prism :
  ∃ (p : RectPrism), surface_area_constraint p ∧ base_height_constraint p ∧
    (∀ (q : RectPrism), surface_area_constraint q → base_height_constraint q →
      volume q ≤ volume p) ∧
    volume p = 27 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_max_volume_of_prism_l3651_365166


namespace NUMINAMATH_CALUDE_inequality_holds_in_interval_l3651_365176

-- Define the inequality function
def inequality (p q : ℝ) : Prop :=
  (5 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 4 * p * q)) / (p + 2 * q) > 3 * p^2 * q

-- State the theorem
theorem inequality_holds_in_interval :
  ∀ p : ℝ, 0 ≤ p → p < 4 → ∀ q : ℝ, q > 0 → inequality p q :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_in_interval_l3651_365176


namespace NUMINAMATH_CALUDE_max_sum_of_solutions_l3651_365161

def is_solution (x y : ℤ) : Prop := x^2 + y^2 = 100

theorem max_sum_of_solutions :
  ∃ (a b : ℤ), is_solution a b ∧ 
  (∀ (x y : ℤ), is_solution x y → x + y ≤ a + b) ∧
  a + b = 14 := by sorry

end NUMINAMATH_CALUDE_max_sum_of_solutions_l3651_365161


namespace NUMINAMATH_CALUDE_roots_order_l3651_365158

variables (a b m n : ℝ)

-- Define the equation
def f (x : ℝ) : ℝ := 1 - (x - a) * (x - b)

theorem roots_order (h1 : f m = 0) (h2 : f n = 0) (h3 : m < n) (h4 : a < b) :
  m < a ∧ a < b ∧ b < n := by
  sorry

end NUMINAMATH_CALUDE_roots_order_l3651_365158


namespace NUMINAMATH_CALUDE_unique_number_exists_l3651_365115

theorem unique_number_exists : ∃! n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  n / sum = 2 * diff ∧ n % sum = 80 ∧ n = 220080 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_exists_l3651_365115


namespace NUMINAMATH_CALUDE_problem_statement_l3651_365147

theorem problem_statement (a b : ℝ) : (2*a + b)^2 + |b - 2| = 0 → (-a - b)^2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3651_365147


namespace NUMINAMATH_CALUDE_pet_food_cost_l3651_365121

theorem pet_food_cost (total_cost rabbit_toy_cost cage_cost found_money : ℚ)
  (h1 : total_cost = 24.81)
  (h2 : rabbit_toy_cost = 6.51)
  (h3 : cage_cost = 12.51)
  (h4 : found_money = 1.00) :
  total_cost - (rabbit_toy_cost + cage_cost) + found_money = 6.79 := by
  sorry

end NUMINAMATH_CALUDE_pet_food_cost_l3651_365121


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3651_365183

theorem fixed_point_on_line (m : ℝ) : 
  (m - 1) * (7/2 : ℝ) - (m + 3) * (5/2 : ℝ) - (m - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3651_365183


namespace NUMINAMATH_CALUDE_chocolate_theorem_l3651_365171

def chocolate_problem (initial_bars : ℕ) (friends : ℕ) (returned_bars : ℕ) (piper_difference : ℕ) (remaining_bars : ℕ) : Prop :=
  ∃ (x : ℚ),
    -- Thomas and friends take x bars initially
    x > 0 ∧
    -- One friend returns 5 bars
    (x - returned_bars) > 0 ∧
    -- Piper takes 5 fewer bars than Thomas and friends
    (x - returned_bars - piper_difference) > 0 ∧
    -- Total bars taken plus remaining bars equals initial bars
    (x - returned_bars) + (x - returned_bars - piper_difference) + remaining_bars = initial_bars ∧
    -- The fraction of bars Thomas and friends took initially
    x / initial_bars = 21 / 80

theorem chocolate_theorem :
  chocolate_problem 200 5 5 5 110 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l3651_365171


namespace NUMINAMATH_CALUDE_max_distance_to_line_l3651_365100

noncomputable section

-- Define the curve C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 / 3 + y^2 = 1}

-- Define the line l
def l : Set (ℝ × ℝ) := {(x, y) | x - y - 4 = 0}

-- Define point M
def M : ℝ × ℝ := (-2, 2)

-- Define the midpoint P of MN
def P (N : ℝ × ℝ) : ℝ × ℝ := ((N.1 + M.1) / 2, (N.2 + M.2) / 2)

-- Define the distance function from a point to a line
def dist_to_line (P : ℝ × ℝ) : ℝ :=
  |P.1 - P.2 - 4| / Real.sqrt 2

-- Theorem statement
theorem max_distance_to_line :
  ∃ (max_dist : ℝ), max_dist = 7 * Real.sqrt 2 / 2 ∧
  ∀ (N : ℝ × ℝ), N ∈ C → dist_to_line (P N) ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l3651_365100
