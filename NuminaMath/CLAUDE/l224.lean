import Mathlib

namespace NUMINAMATH_CALUDE_mark_cans_proof_l224_22452

/-- The number of cans Mark bought -/
def mark_cans : ℕ := 27

/-- The number of cans Jennifer initially bought -/
def jennifer_initial : ℕ := 40

/-- The total number of cans Jennifer brought home -/
def jennifer_total : ℕ := 100

/-- For every 5 cans Mark bought, Jennifer bought 11 cans -/
def jennifer_to_mark_ratio : ℚ := 11 / 5

theorem mark_cans_proof :
  (jennifer_total - jennifer_initial : ℚ) / jennifer_to_mark_ratio = mark_cans := by
  sorry

end NUMINAMATH_CALUDE_mark_cans_proof_l224_22452


namespace NUMINAMATH_CALUDE_value_range_of_function_l224_22450

theorem value_range_of_function : 
  ∀ (y : ℝ), (∃ (x : ℝ), y = (x^2 - 1) / (x^2 + 1)) ↔ -1 ≤ y ∧ y < 1 := by
  sorry

end NUMINAMATH_CALUDE_value_range_of_function_l224_22450


namespace NUMINAMATH_CALUDE_meal_sales_tax_percentage_l224_22440

/-- The maximum total spending allowed for the meal -/
def total_limit : ℝ := 50

/-- The maximum cost of food allowed -/
def max_food_cost : ℝ := 40.98

/-- The tip percentage as a decimal -/
def tip_percentage : ℝ := 0.15

/-- The maximum sales tax percentage that satisfies the conditions -/
def max_sales_tax_percentage : ℝ := 6.1

/-- Theorem stating that the maximum sales tax percentage is approximately 6.1% -/
theorem meal_sales_tax_percentage :
  ∀ (sales_tax_percentage : ℝ),
    sales_tax_percentage ≤ max_sales_tax_percentage →
    max_food_cost + (sales_tax_percentage / 100 * max_food_cost) +
    (tip_percentage * (max_food_cost + (sales_tax_percentage / 100 * max_food_cost))) ≤ total_limit :=
by sorry

end NUMINAMATH_CALUDE_meal_sales_tax_percentage_l224_22440


namespace NUMINAMATH_CALUDE_quadratic_function_property_l224_22428

/-- A quadratic function y = x^2 + bx + c -/
def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_function_property (b c m n : ℝ) :
  (∀ x ≤ 2, quadratic_function b c (x + 0.01) < quadratic_function b c x) →
  quadratic_function b c m = n →
  quadratic_function b c (m + 1) = n →
  m ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l224_22428


namespace NUMINAMATH_CALUDE_geometric_sequence_decreasing_condition_l224_22483

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- A decreasing sequence -/
def DecreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ a n

/-- The condition "0 < q < 1" is neither sufficient nor necessary for a geometric sequence to be decreasing -/
theorem geometric_sequence_decreasing_condition (a : ℕ → ℝ) (q : ℝ) :
  ¬(((0 < q ∧ q < 1) → DecreasingSequence a) ∧ (DecreasingSequence a → (0 < q ∧ q < 1))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_decreasing_condition_l224_22483


namespace NUMINAMATH_CALUDE_hyperbola_condition_l224_22486

/-- A conic section represented by the equation ax² + by² = c -/
structure ConicSection where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to determine if a conic section is a hyperbola -/
def is_hyperbola (conic : ConicSection) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ conic.a * k > 0 ∧ conic.b * k < 0 ∧ conic.c * k ≠ 0

/-- The condition ab < 0 is sufficient but not necessary for a hyperbola -/
theorem hyperbola_condition (conic : ConicSection) :
  (conic.a * conic.b < 0 → is_hyperbola conic) ∧
  ¬(is_hyperbola conic → conic.a * conic.b < 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l224_22486


namespace NUMINAMATH_CALUDE_equation_solution_l224_22445

theorem equation_solution : ∃ x : ℚ, (x^2 + 4*x + 7) / (x + 5) = x + 6 ∧ x = -23/7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l224_22445


namespace NUMINAMATH_CALUDE_solution_set_ln_inequality_l224_22409

theorem solution_set_ln_inequality :
  {x : ℝ | Real.log (x - Real.exp 1) < 1} = {x | Real.exp 1 < x ∧ x < 2 * Real.exp 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_ln_inequality_l224_22409


namespace NUMINAMATH_CALUDE_fixed_point_quadratic_function_l224_22416

theorem fixed_point_quadratic_function (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - (2-m)*x + m
  f (-1) = 3 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_quadratic_function_l224_22416


namespace NUMINAMATH_CALUDE_sum_mod_ten_zero_l224_22447

theorem sum_mod_ten_zero : (5000 + 5001 + 5002 + 5003 + 5004) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_ten_zero_l224_22447


namespace NUMINAMATH_CALUDE_train_speed_calculation_l224_22410

/-- Given a train and tunnel with specified lengths and crossing time, calculate the train's speed in km/hr -/
theorem train_speed_calculation (train_length tunnel_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 415)
  (h2 : tunnel_length = 285)
  (h3 : crossing_time = 40) :
  (train_length + tunnel_length) / crossing_time * 3.6 = 63 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l224_22410


namespace NUMINAMATH_CALUDE_triangle_side_length_l224_22479

theorem triangle_side_length (a b c : ℝ) (h1 : a + b + c = 55) (h2 : b = 20) (h3 : c = 30) : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l224_22479


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l224_22430

/-- Two circles with equations x^2 + y^2 + 2ax + a^2 - 4 = 0 and x^2 + y^2 - 4by - 1 + 4b^2 = 0 -/
def circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 4 = 0
def circle2 (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The circles have exactly three common tangents -/
def have_three_common_tangents (a b : ℝ) : Prop := sorry

theorem min_value_sum_of_reciprocals (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : have_three_common_tangents a b) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), (1 / a^2 + 1 / b^2) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l224_22430


namespace NUMINAMATH_CALUDE_pyramid_angles_theorem_l224_22413

/-- Represents the angles formed by the lateral faces of a pyramid with its square base -/
structure PyramidAngles where
  α : Real
  β : Real
  γ : Real
  δ : Real

/-- Theorem: Given a pyramid with a square base, if the angles formed by the lateral faces 
    with the base are in the ratio 1:2:4:2, then these angles are π/6, π/3, 2π/3, and π/3. -/
theorem pyramid_angles_theorem (angles : PyramidAngles) : 
  (angles.α : Real) / (angles.β : Real) = 1 / 2 ∧
  (angles.α : Real) / (angles.γ : Real) = 1 / 4 ∧
  (angles.α : Real) / (angles.δ : Real) = 1 / 2 ∧
  angles.α + angles.β + angles.γ + angles.δ = 2 * Real.pi →
  angles.α = Real.pi / 6 ∧
  angles.β = Real.pi / 3 ∧
  angles.γ = 2 * Real.pi / 3 ∧
  angles.δ = Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_angles_theorem_l224_22413


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l224_22417

def f (x : ℝ) : ℝ := x^3 + x^2 - 4*x - 4

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = -1 ∨ x = 2 ∨ x = -2) := by
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l224_22417


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l224_22429

theorem subtraction_preserves_inequality (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l224_22429


namespace NUMINAMATH_CALUDE_marble_count_l224_22499

theorem marble_count (n : ℕ) (left_pos right_pos : ℕ) 
  (h1 : left_pos = 5)
  (h2 : right_pos = 3)
  (h3 : n = left_pos + right_pos - 1) :
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l224_22499


namespace NUMINAMATH_CALUDE_perpendicular_implies_intersects_parallel_perpendicular_transitive_perpendicular_implies_parallel_l224_22458

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)

-- Statement 1
theorem perpendicular_implies_intersects (l : Line) (a : Plane) :
  perpendicular l a → intersects l a :=
sorry

-- Statement 3
theorem parallel_perpendicular_transitive (l m n : Line) (a : Plane) :
  parallel l m → parallel m n → perpendicular l a → perpendicular n a :=
sorry

-- Statement 4
theorem perpendicular_implies_parallel (l m n : Line) (a : Plane) :
  parallel l m → perpendicular m a → perpendicular n a → parallel l n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_intersects_parallel_perpendicular_transitive_perpendicular_implies_parallel_l224_22458


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l224_22404

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l224_22404


namespace NUMINAMATH_CALUDE_marble_collection_total_l224_22446

/-- Given a collection of orange, purple, and yellow marbles, where:
    - The number of orange marbles is o
    - There are 30% more orange marbles than purple marbles
    - There are 50% more yellow marbles than orange marbles
    Prove that the total number of marbles is 3.269o -/
theorem marble_collection_total (o : ℝ) (o_positive : o > 0) : ∃ (p y : ℝ),
  p > 0 ∧ y > 0 ∧
  o = 1.3 * p ∧
  y = 1.5 * o ∧
  o + p + y = 3.269 * o :=
sorry

end NUMINAMATH_CALUDE_marble_collection_total_l224_22446


namespace NUMINAMATH_CALUDE_toby_speed_proof_l224_22460

/-- Represents the speed of Toby when pulling the unloaded sled -/
def unloaded_speed : ℝ := 20

/-- Represents the speed of Toby when pulling the loaded sled -/
def loaded_speed : ℝ := 10

/-- Represents the total journey time in hours -/
def total_time : ℝ := 39

/-- Represents the distance of the first part of the journey (loaded sled) -/
def distance1 : ℝ := 180

/-- Represents the distance of the second part of the journey (unloaded sled) -/
def distance2 : ℝ := 120

/-- Represents the distance of the third part of the journey (loaded sled) -/
def distance3 : ℝ := 80

/-- Represents the distance of the fourth part of the journey (unloaded sled) -/
def distance4 : ℝ := 140

theorem toby_speed_proof :
  (distance1 / loaded_speed) + (distance2 / unloaded_speed) +
  (distance3 / loaded_speed) + (distance4 / unloaded_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_toby_speed_proof_l224_22460


namespace NUMINAMATH_CALUDE_shell_ratio_l224_22478

/-- The number of shells each person has -/
structure ShellCounts where
  david : ℕ
  mia : ℕ
  ava : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def shell_problem (sc : ShellCounts) : Prop :=
  sc.david = 15 ∧
  sc.mia = 4 * sc.david ∧
  sc.ava = sc.mia + 20 ∧
  sc.david + sc.mia + sc.ava + sc.alice = 195

/-- The theorem to prove -/
theorem shell_ratio (sc : ShellCounts) 
  (h : shell_problem sc) : sc.alice * 2 = sc.ava := by
  sorry

end NUMINAMATH_CALUDE_shell_ratio_l224_22478


namespace NUMINAMATH_CALUDE_defective_film_probability_l224_22412

/-- The probability of selecting a defective X-ray film from a warehouse with
    specified conditions. -/
theorem defective_film_probability :
  let total_boxes : ℕ := 10
  let boxes_a : ℕ := 5
  let boxes_b : ℕ := 3
  let boxes_c : ℕ := 2
  let defective_rate_a : ℚ := 1 / 10
  let defective_rate_b : ℚ := 1 / 15
  let defective_rate_c : ℚ := 1 / 20
  let prob_a : ℚ := boxes_a / total_boxes
  let prob_b : ℚ := boxes_b / total_boxes
  let prob_c : ℚ := boxes_c / total_boxes
  let total_prob : ℚ := prob_a * defective_rate_a + prob_b * defective_rate_b + prob_c * defective_rate_c
  total_prob = 8 / 100 :=
by sorry

end NUMINAMATH_CALUDE_defective_film_probability_l224_22412


namespace NUMINAMATH_CALUDE_initial_players_l224_22449

theorem initial_players (initial_players new_players lives_per_player total_lives : ℕ) :
  new_players = 5 →
  lives_per_player = 3 →
  total_lives = 27 →
  (initial_players + new_players) * lives_per_player = total_lives →
  initial_players = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_players_l224_22449


namespace NUMINAMATH_CALUDE_equation_solution_l224_22471

theorem equation_solution : 
  ∃ x : ℝ, (5 * 0.85) / x - (8 * 2.25) = 5.5 ∧ x = 4.25 / 23.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l224_22471


namespace NUMINAMATH_CALUDE_inequality_system_solution_l224_22435

theorem inequality_system_solution (x : ℝ) : 
  (abs (x - 1) > 1 ∧ 1 / (4 - x) ≤ 1) ↔ (x < 0 ∨ (2 < x ∧ x ≤ 3) ∨ x > 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l224_22435


namespace NUMINAMATH_CALUDE_one_sixth_percent_of_180_l224_22425

theorem one_sixth_percent_of_180 : (1 / 6 : ℚ) / 100 * 180 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_one_sixth_percent_of_180_l224_22425


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_inequality_l224_22492

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x ∈ Set.Ioo 0 1, f a x > x} = Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_inequality_l224_22492


namespace NUMINAMATH_CALUDE_cyrus_remaining_pages_l224_22454

/-- Calculates the remaining pages Cyrus needs to write -/
def remaining_pages (total_pages first_day second_day third_day fourth_day : ℕ) : ℕ :=
  total_pages - (first_day + second_day + third_day + fourth_day)

/-- Theorem stating that Cyrus needs to write 315 more pages -/
theorem cyrus_remaining_pages :
  remaining_pages 500 25 (2 * 25) (2 * (2 * 25)) 10 = 315 := by
  sorry

end NUMINAMATH_CALUDE_cyrus_remaining_pages_l224_22454


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_eight_l224_22442

theorem consecutive_integers_sqrt_eight (a b : ℤ) : 
  (a < Real.sqrt 8 ∧ Real.sqrt 8 < b) → 
  (b = a + 1) → 
  (b ^ a : ℝ) = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_eight_l224_22442


namespace NUMINAMATH_CALUDE_sand_art_calculation_l224_22474

/-- The amount of sand needed to fill shapes given their dimensions and sand density. -/
theorem sand_art_calculation (rectangle_length : ℝ) (rectangle_area : ℝ) (square_side : ℝ) (sand_density : ℝ) : 
  rectangle_length = 7 →
  rectangle_area = 42 →
  square_side = 5 →
  sand_density = 3 →
  rectangle_area * sand_density + square_side * square_side * sand_density = 201 := by
  sorry

#check sand_art_calculation

end NUMINAMATH_CALUDE_sand_art_calculation_l224_22474


namespace NUMINAMATH_CALUDE_double_cone_is_cone_l224_22400

/-- Represents a point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Defines the set of points satisfying the given equations -/
def DoubleConeSurface (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.θ = c ∧ p.r = |p.z|}

/-- Defines a cone surface in cylindrical coordinates -/
def ConeSurface : Set CylindricalPoint :=
  {p : CylindricalPoint | ∃ (k : ℝ), p.r = k * |p.z|}

/-- Theorem stating that the surface defined by the equations is a cone -/
theorem double_cone_is_cone (c : ℝ) :
  ∃ (k : ℝ), DoubleConeSurface c ⊆ ConeSurface :=
sorry

end NUMINAMATH_CALUDE_double_cone_is_cone_l224_22400


namespace NUMINAMATH_CALUDE_abs_z_minus_i_equals_sqrt2_over_2_l224_22495

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z based on the given condition
def z : ℂ := by
  sorry

-- Theorem statement
theorem abs_z_minus_i_equals_sqrt2_over_2 :
  Complex.abs (z - i) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_minus_i_equals_sqrt2_over_2_l224_22495


namespace NUMINAMATH_CALUDE_extracurricular_activity_selection_l224_22407

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem extracurricular_activity_selection 
  (total_people : ℕ) 
  (boys : ℕ) 
  (girls : ℕ) 
  (leaders : ℕ) 
  (to_select : ℕ) :
  total_people = 13 →
  boys = 8 →
  girls = 5 →
  leaders = 2 →
  to_select = 5 →
  (choose girls 1 * choose boys 4 = 350) ∧
  (choose (total_people - leaders) 3 = 165) ∧
  (choose total_people to_select - choose (total_people - leaders) to_select = 825) :=
by sorry

end NUMINAMATH_CALUDE_extracurricular_activity_selection_l224_22407


namespace NUMINAMATH_CALUDE_equation_solution_l224_22422

theorem equation_solution : ∃ y : ℚ, y - 1/2 = 1/6 - 2/3 + 1/4 ∧ y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l224_22422


namespace NUMINAMATH_CALUDE_value_of_b_l224_22461

theorem value_of_b (a b : ℝ) (h1 : 4 * a^2 + 1 = 1) (h2 : b - a = 3) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l224_22461


namespace NUMINAMATH_CALUDE_median_sum_squares_l224_22432

theorem median_sum_squares (a b c : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) :
  let m₁ := (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2)
  let m₂ := (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2)
  let m₃ := (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)
  m₁^2 + m₂^2 + m₃^2 = 442.5 := by
sorry

end NUMINAMATH_CALUDE_median_sum_squares_l224_22432


namespace NUMINAMATH_CALUDE_discounted_biographies_count_l224_22488

theorem discounted_biographies_count (biography_price mystery_price total_savings mystery_count total_discount_rate mystery_discount_rate : ℝ) 
  (h1 : biography_price = 20)
  (h2 : mystery_price = 12)
  (h3 : total_savings = 19)
  (h4 : mystery_count = 3)
  (h5 : total_discount_rate = 0.43)
  (h6 : mystery_discount_rate = 0.375) :
  ∃ (biography_count : ℕ), 
    biography_count = 5 ∧ 
    biography_count * (biography_price * (total_discount_rate - mystery_discount_rate)) + 
    mystery_count * (mystery_price * mystery_discount_rate) = total_savings :=
by sorry

end NUMINAMATH_CALUDE_discounted_biographies_count_l224_22488


namespace NUMINAMATH_CALUDE_simplify_expression_l224_22424

theorem simplify_expression (b : ℝ) : (1:ℝ)*(2*b)*(3*b^2)*(4*b^3)*(5*b^4)*(6*b^5) = 720 * b^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l224_22424


namespace NUMINAMATH_CALUDE_system_solution_l224_22494

/-- Given a system of equations, prove that the only solution is (a, b, c) = (t, t, t) for any t ∈ ℝ -/
theorem system_solution (a b c : ℝ) : 
  (a * (b^2 + c) = c * (c + a*b)) ∧ 
  (b * (c^2 + a) = a * (a + b*c)) ∧ 
  (c * (a^2 + b) = b * (b + c*a)) → 
  ∃ t : ℝ, a = t ∧ b = t ∧ c = t := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l224_22494


namespace NUMINAMATH_CALUDE_total_presents_equals_58_l224_22491

/-- The number of presents Ethan has -/
def ethan_presents : ℕ := 31

/-- The number of presents Alissa has -/
def alissa_presents : ℕ := ethan_presents - 22

/-- The number of presents Bella has -/
def bella_presents : ℕ := 2 * alissa_presents

/-- The total number of presents Bella, Ethan, and Alissa have -/
def total_presents : ℕ := ethan_presents + alissa_presents + bella_presents

theorem total_presents_equals_58 : total_presents = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_presents_equals_58_l224_22491


namespace NUMINAMATH_CALUDE_supplementary_angles_equal_l224_22489

/-- Two angles that are supplementary to the same angle are equal. -/
theorem supplementary_angles_equal (α β γ : Real) (h1 : α + γ = 180) (h2 : β + γ = 180) : α = β := by
  sorry

end NUMINAMATH_CALUDE_supplementary_angles_equal_l224_22489


namespace NUMINAMATH_CALUDE_birthday_age_proof_l224_22485

theorem birthday_age_proof (A : ℕ) : A = 4 * (A - 10) - 5 ↔ A = 15 := by
  sorry

end NUMINAMATH_CALUDE_birthday_age_proof_l224_22485


namespace NUMINAMATH_CALUDE_trout_division_l224_22455

theorem trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) :
  total_trout = 18 →
  num_people = 2 →
  trout_per_person = total_trout / num_people →
  trout_per_person = 9 :=
by sorry

end NUMINAMATH_CALUDE_trout_division_l224_22455


namespace NUMINAMATH_CALUDE_square_root_pattern_l224_22418

theorem square_root_pattern (n : ℕ) (hn : n > 0) : 
  ∃ (f : ℕ → ℝ), 
    (f 1 = Real.sqrt (1 + 1 / 1^2 + 1 / 2^2)) ∧ 
    (f 2 = Real.sqrt (1 + 1 / 2^2 + 1 / 3^2)) ∧ 
    (f 3 = Real.sqrt (1 + 1 / 3^2 + 1 / 4^2)) ∧ 
    (f 1 = 3/2) ∧ 
    (f 2 = 7/6) ∧ 
    (f 3 = 13/12) ∧ 
    (∀ k : ℕ, k > 0 → f k = Real.sqrt (1 + 1 / k^2 + 1 / (k+1)^2)) ∧
    (f n = 1 + 1 / (n * (n+1))) :=
sorry

end NUMINAMATH_CALUDE_square_root_pattern_l224_22418


namespace NUMINAMATH_CALUDE_quadratic_factorization_l224_22434

theorem quadratic_factorization (x : ℝ) : -2 * x^2 + 2 * x - (1/2) = -2 * (x - 1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l224_22434


namespace NUMINAMATH_CALUDE_son_age_l224_22473

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end NUMINAMATH_CALUDE_son_age_l224_22473


namespace NUMINAMATH_CALUDE_reciprocal_of_three_halves_l224_22470

-- Define the concept of reciprocal
def is_reciprocal (a b : ℚ) : Prop := a * b = 1

-- State the theorem
theorem reciprocal_of_three_halves : 
  is_reciprocal (3/2 : ℚ) (2/3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_three_halves_l224_22470


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l224_22414

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ ((x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l224_22414


namespace NUMINAMATH_CALUDE_problem_solution_l224_22498

theorem problem_solution (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1) 
  (h3 : Real.rpow a (2 * Real.log a / Real.log 3) = 81 * Real.sqrt 3) : 
  1 / a^2 + Real.log a / Real.log 9 = 105/4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l224_22498


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l224_22481

theorem greatest_integer_with_gcd_six : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 → Nat.gcd m 18 = 6 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l224_22481


namespace NUMINAMATH_CALUDE_leonardo_chocolate_purchase_l224_22467

theorem leonardo_chocolate_purchase (chocolate_cost : ℕ) (leonardo_money : ℕ) (borrowed_money : ℕ) : 
  chocolate_cost = 500 ∧ leonardo_money = 400 ∧ borrowed_money = 59 →
  chocolate_cost - (leonardo_money + borrowed_money) = 41 :=
by sorry

end NUMINAMATH_CALUDE_leonardo_chocolate_purchase_l224_22467


namespace NUMINAMATH_CALUDE_chinese_character_number_puzzle_l224_22477

theorem chinese_character_number_puzzle :
  ∃! (A B C D : Nat),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
    A * 10 + B = 19 ∧
    C * 10 + D = 62 ∧
    (A * 1000 + B * 100 + C * 10 + D) - (A * 1000 + A * 100 + B * 10 + B) = 124 := by
  sorry

end NUMINAMATH_CALUDE_chinese_character_number_puzzle_l224_22477


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l224_22415

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = y
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 = x

-- Define the line
def intersection_line (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ (x y : ℝ), circle_O1 x y ∧ circle_O2 x y → intersection_line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l224_22415


namespace NUMINAMATH_CALUDE_subtraction_problem_l224_22423

theorem subtraction_problem : 888.88 - 444.44 = 444.44 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l224_22423


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l224_22480

/-- Number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute 6 4 = 72 := by sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l224_22480


namespace NUMINAMATH_CALUDE_rodrigos_classroom_chairs_l224_22464

/-- The number of chairs left in Rodrigo's classroom after Lisa borrows some chairs -/
def chairs_left (red_chairs yellow_chairs blue_chairs borrowed : ℕ) : ℕ :=
  red_chairs + yellow_chairs + blue_chairs - borrowed

/-- Theorem stating the number of chairs left in Rodrigo's classroom -/
theorem rodrigos_classroom_chairs :
  ∀ (red_chairs : ℕ),
  red_chairs = 4 →
  ∀ (yellow_chairs : ℕ),
  yellow_chairs = 2 * red_chairs →
  ∀ (blue_chairs : ℕ),
  blue_chairs = yellow_chairs - 2 →
  chairs_left red_chairs yellow_chairs blue_chairs 3 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_rodrigos_classroom_chairs_l224_22464


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l224_22433

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 ∧ b = 36 ∧ c^2 = a^2 + b^2 → c = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l224_22433


namespace NUMINAMATH_CALUDE_evaluate_x_squared_minus_y_squared_l224_22497

theorem evaluate_x_squared_minus_y_squared : 
  ∀ x y : ℝ, x + y = 10 → 2 * x + y = 13 → x^2 - y^2 = -40 := by
sorry

end NUMINAMATH_CALUDE_evaluate_x_squared_minus_y_squared_l224_22497


namespace NUMINAMATH_CALUDE_polynomial_simplification_l224_22437

theorem polynomial_simplification (x : ℝ) :
  (5 * x^12 - 3 * x^9 + 6 * x^8 - 2 * x^7) + 
  (7 * x^12 + 2 * x^11 - x^9 + 4 * x^7 + 2 * x^5 - x + 3) = 
  12 * x^12 + 2 * x^11 - 4 * x^9 + 6 * x^8 + 2 * x^7 + 2 * x^5 - x + 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l224_22437


namespace NUMINAMATH_CALUDE_ball_box_problem_l224_22451

/-- Given an opaque box with balls of three colors: red, yellow, and blue. -/
structure BallBox where
  total : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ
  total_eq : total = red + yellow + blue
  yellow_eq : yellow = 2 * blue

/-- The probability of drawing a blue ball from the box -/
def blue_probability (box : BallBox) : ℚ :=
  box.blue / box.total

/-- The number of additional blue balls needed to make the probability 1/2 -/
def additional_blue_balls (box : BallBox) : ℕ :=
  let new_total := box.total + 14
  let new_blue := box.blue + 14
  14

/-- Theorem stating the properties of the specific box in the problem -/
theorem ball_box_problem :
  ∃ (box : BallBox),
    box.total = 30 ∧
    box.red = 6 ∧
    blue_probability box = 4 / 15 ∧
    additional_blue_balls box = 14 ∧
    blue_probability ⟨box.total + 14, box.red, box.blue + 14, box.yellow,
      by sorry, by sorry⟩ = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ball_box_problem_l224_22451


namespace NUMINAMATH_CALUDE_absolute_value_equality_l224_22411

theorem absolute_value_equality (x : ℝ) (h : x > 3) : 
  |x - Real.sqrt ((x - 3)^2)| = 3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l224_22411


namespace NUMINAMATH_CALUDE_function_composition_l224_22402

/-- Given a function f such that f(3x) = 5 / (3 + x) for all x > 0,
    prove that 3f(x) = 45 / (9 + x) --/
theorem function_composition (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 5 / (3 + x)) :
  ∀ x > 0, 3 * f x = 45 / (9 + x) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l224_22402


namespace NUMINAMATH_CALUDE_greenleaf_academy_history_class_l224_22401

/-- The number of students in the history class at Greenleaf Academy -/
def history_class_size : ℕ := by sorry

theorem greenleaf_academy_history_class :
  let total_students : ℕ := 70
  let both_subjects : ℕ := 10
  let geography_only : ℕ := 16  -- Derived from the solution, but mathematically necessary
  let history_only : ℕ := total_students - both_subjects - geography_only
  let history_class_size : ℕ := history_only + both_subjects
  let geography_class_size : ℕ := geography_only + both_subjects
  (total_students = geography_only + history_only + both_subjects) ∧
  (history_class_size = 2 * geography_class_size) →
  history_class_size = 52 :=
by sorry

end NUMINAMATH_CALUDE_greenleaf_academy_history_class_l224_22401


namespace NUMINAMATH_CALUDE_swim_team_total_l224_22475

theorem swim_team_total (girls : ℕ) (boys : ℕ) : 
  girls = 80 → girls = 5 * boys → girls + boys = 96 := by
  sorry

end NUMINAMATH_CALUDE_swim_team_total_l224_22475


namespace NUMINAMATH_CALUDE_rams_weight_increase_percentage_l224_22431

theorem rams_weight_increase_percentage
  (weight_ratio : ℚ) -- Ratio of Ram's weight to Shyam's weight
  (total_weight_after : ℝ) -- Total weight after increase
  (total_increase_percentage : ℝ) -- Total weight increase percentage
  (shyam_increase_percentage : ℝ) -- Shyam's weight increase percentage
  (h1 : weight_ratio = 4 / 5) -- Condition: weight ratio is 4:5
  (h2 : total_weight_after = 82.8) -- Condition: total weight after increase is 82.8 kg
  (h3 : total_increase_percentage = 15) -- Condition: total weight increase is 15%
  (h4 : shyam_increase_percentage = 19) -- Condition: Shyam's weight increased by 19%
  : ∃ (ram_increase_percentage : ℝ), ram_increase_percentage = 10 :=
by sorry

end NUMINAMATH_CALUDE_rams_weight_increase_percentage_l224_22431


namespace NUMINAMATH_CALUDE_curve_fixed_point_l224_22406

/-- The curve C passes through a fixed point for all k ≠ -1 -/
theorem curve_fixed_point (k : ℝ) (hk : k ≠ -1) :
  ∃ (x y : ℝ), ∀ (k : ℝ), k ≠ -1 →
    x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0 ∧ x = 1 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_curve_fixed_point_l224_22406


namespace NUMINAMATH_CALUDE_inequality_chain_l224_22487

theorem inequality_chain (a b d m : ℝ) 
  (h1 : a > b) (h2 : b > d) (h3 : d ≥ m) : a > m := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l224_22487


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l224_22490

/-- A partnership business where one partner's investment and time are multiples of the other's -/
structure Partnership where
  investment_ratio : ℕ  -- Ratio of A's investment to B's
  time_ratio : ℕ        -- Ratio of A's investment time to B's
  b_profit : ℕ          -- B's profit in Rs

/-- Calculate the total profit of a partnership given B's profit -/
def total_profit (p : Partnership) : ℕ :=
  p.b_profit * (p.investment_ratio * p.time_ratio + 1)

/-- Theorem stating the total profit for the given partnership conditions -/
theorem partnership_profit_calculation (p : Partnership) 
  (h1 : p.investment_ratio = 3)
  (h2 : p.time_ratio = 2)
  (h3 : p.b_profit = 3000) :
  total_profit p = 21000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l224_22490


namespace NUMINAMATH_CALUDE_greatest_multiple_of_five_cubed_less_than_8000_l224_22466

theorem greatest_multiple_of_five_cubed_less_than_8000 :
  ∃ (y : ℕ), y > 0 ∧ 5 ∣ y ∧ y^3 < 8000 ∧ ∀ (z : ℕ), z > 0 → 5 ∣ z → z^3 < 8000 → z ≤ y :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_five_cubed_less_than_8000_l224_22466


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l224_22405

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2 else x^2 - 1

theorem f_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l224_22405


namespace NUMINAMATH_CALUDE_trinomial_square_difference_l224_22472

theorem trinomial_square_difference : (23 + 15 + 7)^2 - (23^2 + 15^2 + 7^2) = 1222 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_square_difference_l224_22472


namespace NUMINAMATH_CALUDE_trajectory_and_tangent_line_l224_22403

-- Define points A, B, and P
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)
def P : ℝ → ℝ → ℝ × ℝ := λ x y => (x, y)

-- Define the condition |PA| = 2|PB|
def condition (x y : ℝ) : Prop :=
  (x + 3)^2 + y^2 = 4 * ((x - 3)^2 + y^2)

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 16

-- Define the line l
def line_l (c : ℝ) (x y : ℝ) : Prop :=
  x + y + c = 0

-- Define the tangent condition
def is_tangent (c : ℝ) : Prop :=
  ∃ x y : ℝ, trajectory_C x y ∧ line_l c x y ∧
  ∀ x' y' : ℝ, trajectory_C x' y' → line_l c x' y' → (x', y') = (x, y)

-- State the theorem
theorem trajectory_and_tangent_line :
  (∀ x y : ℝ, condition x y ↔ trajectory_C x y) ∧
  (∃ c : ℝ, is_tangent c ∧ c = -5 + 4 * Real.sqrt 2 ∨ c = -5 - 4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_tangent_line_l224_22403


namespace NUMINAMATH_CALUDE_base_ten_to_base_seven_l224_22463

theorem base_ten_to_base_seven : 
  ∃ (a b c d : ℕ), 
    1357 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
    a = 3 ∧ b = 6 ∧ c = 4 ∧ d = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_ten_to_base_seven_l224_22463


namespace NUMINAMATH_CALUDE_bicycle_speed_l224_22420

/-- Proves that given a 400 km trip where the first 100 km is traveled at speed v km/h
    and the remaining 300 km at 15 km/h, if the average speed for the entire trip is 16 km/h,
    then v = 20 km/h. -/
theorem bicycle_speed (v : ℝ) :
  v > 0 →
  (100 / v + 300 / 15 = 400 / 16) →
  v = 20 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_speed_l224_22420


namespace NUMINAMATH_CALUDE_least_common_multiple_of_primes_l224_22436

theorem least_common_multiple_of_primes : ∃ n : ℕ,
  (n > 0) ∧
  (n % 7 = 0) ∧ (n % 11 = 0) ∧ (n % 13 = 0) ∧
  (∀ m : ℕ, m > 0 ∧ m % 7 = 0 ∧ m % 11 = 0 ∧ m % 13 = 0 → m ≥ n) ∧
  n = 1001 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_of_primes_l224_22436


namespace NUMINAMATH_CALUDE_min_value_expression_l224_22456

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((2*a + 2*a*b - b*(b+1))^2 + (b - 4*a^2 + 2*a*(b+1))^2) / (4*a^2 + b^2) ≥ 1 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    ((2*a₀ + 2*a₀*b₀ - b₀*(b₀+1))^2 + (b₀ - 4*a₀^2 + 2*a₀*(b₀+1))^2) / (4*a₀^2 + b₀^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l224_22456


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_l224_22426

theorem purely_imaginary_complex (m : ℝ) : 
  let z : ℂ := (m + Complex.I) / (1 + Complex.I)
  (∃ (y : ℝ), z = Complex.I * y) ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_l224_22426


namespace NUMINAMATH_CALUDE_fraction_value_zero_l224_22465

theorem fraction_value_zero (B A P E H b K p J C O : ℕ) :
  (B ≠ A ∧ B ≠ P ∧ B ≠ E ∧ B ≠ H ∧ B ≠ b ∧ B ≠ K ∧ B ≠ p ∧ B ≠ J ∧ B ≠ C ∧ B ≠ O) ∧
  (A ≠ P ∧ A ≠ E ∧ A ≠ H ∧ A ≠ b ∧ A ≠ K ∧ A ≠ p ∧ A ≠ J ∧ A ≠ C ∧ A ≠ O) ∧
  (P ≠ E ∧ P ≠ H ∧ P ≠ b ∧ P ≠ K ∧ P ≠ p ∧ P ≠ J ∧ P ≠ C ∧ P ≠ O) ∧
  (E ≠ H ∧ E ≠ b ∧ E ≠ K ∧ E ≠ p ∧ E ≠ J ∧ E ≠ C ∧ E ≠ O) ∧
  (H ≠ b ∧ H ≠ K ∧ H ≠ p ∧ H ≠ J ∧ H ≠ C ∧ H ≠ O) ∧
  (b ≠ K ∧ b ≠ p ∧ b ≠ J ∧ b ≠ C ∧ b ≠ O) ∧
  (K ≠ p ∧ K ≠ J ∧ K ≠ C ∧ K ≠ O) ∧
  (p ≠ J ∧ p ≠ C ∧ p ≠ O) ∧
  (J ≠ C ∧ J ≠ O) ∧
  (C ≠ O) ∧
  (B < 10 ∧ A < 10 ∧ P < 10 ∧ E < 10 ∧ H < 10 ∧ b < 10 ∧ K < 10 ∧ p < 10 ∧ J < 10 ∧ C < 10 ∧ O < 10) →
  (B * A * P * E * H * b * E : ℚ) / (K * A * p * J * C * O * H : ℕ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_value_zero_l224_22465


namespace NUMINAMATH_CALUDE_log_inequality_solution_set_l224_22469

-- Define the logarithm function with base 0.1
noncomputable def log_base_point_one (x : ℝ) := Real.log x / Real.log 0.1

-- State the theorem
theorem log_inequality_solution_set :
  ∀ x : ℝ, log_base_point_one (2^x - 1) < 0 ↔ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_solution_set_l224_22469


namespace NUMINAMATH_CALUDE_sin_pi_over_six_l224_22441

theorem sin_pi_over_six : Real.sin (π / 6) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_pi_over_six_l224_22441


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l224_22408

/-- Represents the pricing strategy of a merchant -/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  selling_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price based on the list price and purchase discount -/
def purchase_price (m : MerchantPricing) : ℝ :=
  m.list_price * (1 - m.purchase_discount)

/-- Calculates the selling price based on the marked price and selling discount -/
def selling_price (m : MerchantPricing) : ℝ :=
  m.marked_price * (1 - m.selling_discount)

/-- Calculates the profit based on the selling price and purchase price -/
def profit (m : MerchantPricing) : ℝ :=
  selling_price m - purchase_price m

/-- Theorem: The merchant must mark the goods at 125% of the list price -/
theorem merchant_pricing_strategy (m : MerchantPricing) 
  (h1 : m.purchase_discount = 0.3)
  (h2 : m.selling_discount = 0.2)
  (h3 : m.profit_margin = 0.3)
  (h4 : profit m = m.profit_margin * selling_price m) :
  m.marked_price = 1.25 * m.list_price := by
  sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l224_22408


namespace NUMINAMATH_CALUDE_matrix_product_equality_l224_22482

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 3, 1; 7, -1, 0; 0, 4, -2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -5, 2; 0, 4, 3; 1, 0, -1]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![3, 2, 12; 7, -39, 11; -2, 16, 14]

theorem matrix_product_equality : A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l224_22482


namespace NUMINAMATH_CALUDE_arun_weight_theorem_l224_22448

def arun_weight_conditions (w : ℕ) : Prop :=
  64 < w ∧ w < 72 ∧ w % 3 = 0 ∧  -- Arun's condition
  60 < w ∧ w < 70 ∧ w % 2 = 0 ∧  -- Brother's condition
  w ≤ 67 ∧ Nat.Prime w ∧         -- Mother's condition
  63 ≤ w ∧ w ≤ 71 ∧ w % 5 = 0 ∧  -- Sister's condition
  62 < w ∧ w ≤ 73 ∧ w % 4 = 0    -- Father's condition

theorem arun_weight_theorem :
  ∃! w : ℕ, arun_weight_conditions w ∧ w = 66 := by
  sorry

end NUMINAMATH_CALUDE_arun_weight_theorem_l224_22448


namespace NUMINAMATH_CALUDE_circle_radius_is_three_l224_22438

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y - 7 = 0

-- State the theorem
theorem circle_radius_is_three :
  ∃ (h k r : ℝ), r = 3 ∧ ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_three_l224_22438


namespace NUMINAMATH_CALUDE_set_S_bounds_l224_22493

def S : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (2*x + 3)/(x + 2)}

theorem set_S_bounds :
  (∃ M : ℝ, IsLUB S M ∧ M ∉ S) ∧
  (∃ m : ℝ, IsGLB S m ∧ m ∈ S) := by sorry

end NUMINAMATH_CALUDE_set_S_bounds_l224_22493


namespace NUMINAMATH_CALUDE_parallel_segment_sum_l224_22419

/-- Given two points A(a,-2) and B(1,b) in a plane rectangular coordinate system,
    if AB is parallel to the x-axis and AB = 3, then a + b = 2 or a + b = -4 -/
theorem parallel_segment_sum (a b : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (a, -2) ∧ B = (1, b) ∧ 
   (A.2 = B.2) ∧  -- AB is parallel to x-axis
   ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 3^2))  -- AB = 3
  → (a + b = 2 ∨ a + b = -4) := by
sorry

end NUMINAMATH_CALUDE_parallel_segment_sum_l224_22419


namespace NUMINAMATH_CALUDE_kids_left_playing_l224_22468

theorem kids_left_playing (initial_kids : ℝ) (kids_going_home : ℝ) :
  initial_kids = 22.0 →
  kids_going_home = 14.0 →
  initial_kids - kids_going_home = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_kids_left_playing_l224_22468


namespace NUMINAMATH_CALUDE_books_per_shelf_l224_22462

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) 
  (h1 : total_books = 315) (h2 : num_shelves = 7) : 
  total_books / num_shelves = 45 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l224_22462


namespace NUMINAMATH_CALUDE_largest_bundle_size_correct_l224_22476

def largest_bundle_size (john_notebooks emily_notebooks min_bundle_size : ℕ) : ℕ :=
  Nat.gcd john_notebooks emily_notebooks

theorem largest_bundle_size_correct 
  (john_notebooks : ℕ) 
  (emily_notebooks : ℕ) 
  (min_bundle_size : ℕ) 
  (h1 : john_notebooks = 36) 
  (h2 : emily_notebooks = 45) 
  (h3 : min_bundle_size = 5) :
  largest_bundle_size john_notebooks emily_notebooks min_bundle_size = 9 ∧ 
  largest_bundle_size john_notebooks emily_notebooks min_bundle_size > min_bundle_size := by
  sorry

#eval largest_bundle_size 36 45 5

end NUMINAMATH_CALUDE_largest_bundle_size_correct_l224_22476


namespace NUMINAMATH_CALUDE_coupon_discount_percentage_l224_22443

theorem coupon_discount_percentage (original_price increased_price final_price : ℝ) 
  (h1 : original_price = 200)
  (h2 : increased_price = original_price * 1.3)
  (h3 : final_price = 182) : 
  (increased_price - final_price) / increased_price = 0.3 := by
sorry

end NUMINAMATH_CALUDE_coupon_discount_percentage_l224_22443


namespace NUMINAMATH_CALUDE_no_snow_probability_l224_22421

theorem no_snow_probability (p : ℝ) (h : p = 3/4) :
  (1 - p)^3 = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l224_22421


namespace NUMINAMATH_CALUDE_total_ear_muffs_l224_22484

/-- The number of ear muffs bought before December -/
def before_december : ℕ := 1346

/-- The number of ear muffs bought during December -/
def during_december : ℕ := 6444

/-- The total number of ear muffs bought -/
def total : ℕ := before_december + during_december

theorem total_ear_muffs : total = 7790 := by sorry

end NUMINAMATH_CALUDE_total_ear_muffs_l224_22484


namespace NUMINAMATH_CALUDE_product_expansion_l224_22444

theorem product_expansion (x : ℝ) (h : x ≠ 0) :
  (3 / 7) * ((7 / x^3) - 14 * x^4) = 3 / x^3 - 6 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l224_22444


namespace NUMINAMATH_CALUDE_three_students_left_l224_22427

/-- Calculates the number of students who left during the year. -/
def students_left (initial : ℕ) (new : ℕ) (final : ℕ) : ℕ :=
  initial + new - final

/-- Proves that 3 students left during the year given the initial, new, and final student counts. -/
theorem three_students_left : students_left 4 42 43 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_students_left_l224_22427


namespace NUMINAMATH_CALUDE_quadratic_factorization_l224_22439

theorem quadratic_factorization (a b c : ℤ) :
  (∀ x : ℚ, x^2 + 16*x + 63 = (x + a) * (x + b)) →
  (∀ x : ℚ, x^2 + 6*x - 72 = (x + b) * (x - c)) →
  a + b + c = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l224_22439


namespace NUMINAMATH_CALUDE_expense_increase_l224_22459

theorem expense_increase (december_salary : ℝ) (h1 : december_salary > 0) : 
  let december_mortgage := 0.4 * december_salary
  let december_expenses := december_salary - december_mortgage
  let january_salary := 1.09 * december_salary
  let january_expenses := january_salary - december_mortgage
  (january_expenses - december_expenses) / december_expenses = 0.15
  := by sorry

end NUMINAMATH_CALUDE_expense_increase_l224_22459


namespace NUMINAMATH_CALUDE_min_value_theorem_l224_22453

theorem min_value_theorem (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 2*m + n = 1) :
  1/m + 2/n ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), 0 < m₀ ∧ 0 < n₀ ∧ 2*m₀ + n₀ = 1 ∧ 1/m₀ + 2/n₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l224_22453


namespace NUMINAMATH_CALUDE_functional_inequality_l224_22457

theorem functional_inequality (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  let f : ℝ → ℝ := λ x => (x^3 - x^2 - 1) / (2*x*(x-1))
  f x + f ((x-1)/x) ≥ 1 + x :=
by sorry

end NUMINAMATH_CALUDE_functional_inequality_l224_22457


namespace NUMINAMATH_CALUDE_product_from_gcd_lcm_l224_22496

theorem product_from_gcd_lcm (a b : ℕ+) : 
  Nat.gcd a b = 8 → Nat.lcm a b = 72 → a * b = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_from_gcd_lcm_l224_22496
