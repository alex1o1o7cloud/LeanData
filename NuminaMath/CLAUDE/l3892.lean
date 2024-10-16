import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3892_389250

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 - a 5 + a 15 = 20 →
  a 3 + a 19 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3892_389250


namespace NUMINAMATH_CALUDE_determine_sequence_from_final_state_l3892_389290

/-- Represents the state of the cards at each step -/
structure CardState where
  red : ℤ
  blue : ℤ

/-- Applies the transformation to the cards given k -/
def transform (state : CardState) (k : ℕ+) : CardState :=
  { red := k * state.red + state.blue
  , blue := state.red }

/-- Applies n transformations to the initial state using the sequence ks -/
def apply_transformations (initial : CardState) (ks : List ℕ+) : CardState :=
  ks.foldl transform initial

/-- States that it's possible to determine the sequence from the final state -/
theorem determine_sequence_from_final_state 
  (n : ℕ) 
  (ks : List ℕ+) 
  (h_length : ks.length = n) 
  (initial : CardState) 
  (h_initial : initial.red > initial.blue) :
  ∃ (f : CardState → List ℕ+), 
    f (apply_transformations initial ks) = ks :=
sorry

end NUMINAMATH_CALUDE_determine_sequence_from_final_state_l3892_389290


namespace NUMINAMATH_CALUDE_total_cost_is_21_l3892_389278

/-- The cost of a single carnation in dollars -/
def single_carnation_cost : ℚ := 0.50

/-- The cost of a dozen carnations in dollars -/
def dozen_carnation_cost : ℚ := 4.00

/-- The number of teachers Georgia is sending carnations to -/
def number_of_teachers : ℕ := 5

/-- The number of friends Georgia is buying carnations for -/
def number_of_friends : ℕ := 14

/-- The total cost of Georgia's carnation purchases -/
def total_cost : ℚ := dozen_carnation_cost * number_of_teachers + 
  single_carnation_cost * (number_of_friends % 12)

/-- Theorem stating that the total cost is $21.00 -/
theorem total_cost_is_21 : total_cost = 21 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_21_l3892_389278


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3892_389273

/-- Given a circle and a line, if their intersection forms a chord of length 4, then the parameter 'a' in the circle equation equals -4 -/
theorem circle_line_intersection (x y : ℝ) (a : ℝ) : 
  (x^2 + y^2 + 2*x - 2*y + a = 0) →  -- Circle equation
  (x + y + 2 = 0) →                  -- Line equation
  (∃ p q : ℝ × ℝ, p ≠ q ∧            -- Existence of two distinct intersection points
    (p.1^2 + p.2^2 + 2*p.1 - 2*p.2 + a = 0) ∧
    (p.1 + p.2 + 2 = 0) ∧
    (q.1^2 + q.2^2 + 2*q.1 - 2*q.2 + a = 0) ∧
    (q.1 + q.2 + 2 = 0) ∧
    ((p.1 - q.1)^2 + (p.2 - q.2)^2 = 16)) → -- Chord length is 4
  a = -4 := by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3892_389273


namespace NUMINAMATH_CALUDE_factor_expression_l3892_389280

theorem factor_expression (y : ℝ) : 4 * y * (y + 2) + 6 * (y + 2) = (y + 2) * (2 * (2 * y + 3)) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3892_389280


namespace NUMINAMATH_CALUDE_choose_four_from_thirteen_l3892_389252

theorem choose_four_from_thirteen : Nat.choose 13 4 = 715 := by sorry

end NUMINAMATH_CALUDE_choose_four_from_thirteen_l3892_389252


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3892_389259

/-- The line y = x + 1 intersects the circle x² + y² = 1 -/
theorem line_intersects_circle : ∃ (x y : ℝ), y = x + 1 ∧ x^2 + y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3892_389259


namespace NUMINAMATH_CALUDE_log3_of_9_cubed_l3892_389227

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log3_of_9_cubed : log3 (9^3) = 6 := by sorry

end NUMINAMATH_CALUDE_log3_of_9_cubed_l3892_389227


namespace NUMINAMATH_CALUDE_inverse_function_problem_l3892_389200

theorem inverse_function_problem (f : ℝ → ℝ) (hf : Function.Bijective f) :
  f 6 = 5 → f 5 = 1 → f 1 = 4 →
  (Function.invFun f) ((Function.invFun f 5) * (Function.invFun f 4)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l3892_389200


namespace NUMINAMATH_CALUDE_geometric_sequence_complex_l3892_389215

def z₁ (a : ℝ) : ℂ := a + Complex.I
def z₂ (a : ℝ) : ℂ := 2*a + 2*Complex.I
def z₃ (a : ℝ) : ℂ := 3*a + 4*Complex.I

theorem geometric_sequence_complex (a : ℝ) :
  (Complex.abs (z₂ a))^2 = (Complex.abs (z₁ a)) * (Complex.abs (z₃ a)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_complex_l3892_389215


namespace NUMINAMATH_CALUDE_gas_cost_proof_l3892_389279

/-- The total cost of gas for a trip to New York City -/
def total_cost : ℝ := 82.50

/-- The number of friends initially splitting the cost -/
def initial_friends : ℕ := 3

/-- The number of friends who joined later -/
def additional_friends : ℕ := 2

/-- The total number of friends after more joined -/
def total_friends : ℕ := initial_friends + additional_friends

/-- The amount by which each original friend's cost decreased -/
def cost_decrease : ℝ := 11

theorem gas_cost_proof :
  (total_cost / initial_friends) - (total_cost / total_friends) = cost_decrease :=
sorry

end NUMINAMATH_CALUDE_gas_cost_proof_l3892_389279


namespace NUMINAMATH_CALUDE_max_non_intersecting_points_l3892_389270

/-- A type representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A function that checks if a broken line formed by a list of points intersects itself -/
def is_self_intersecting (points : List Point) : Prop :=
  sorry

/-- The property that any permutation of points forms a non-self-intersecting broken line -/
def all_permutations_non_intersecting (points : List Point) : Prop :=
  ∀ perm : List Point, perm.Perm points → ¬(is_self_intersecting perm)

theorem max_non_intersecting_points :
  ∃ (points : List Point),
    points.length = 4 ∧
    all_permutations_non_intersecting points ∧
    ∀ (larger_set : List Point),
      larger_set.length > 4 →
      ¬(all_permutations_non_intersecting larger_set) :=
sorry

end NUMINAMATH_CALUDE_max_non_intersecting_points_l3892_389270


namespace NUMINAMATH_CALUDE_K_idempotent_l3892_389263

/-- The set of all 2013 × 2013 arrays with entries 0 and 1 -/
def F : Type := Fin 2013 → Fin 2013 → Fin 2

/-- The sum of all entries sharing a row or column with a[i,j] -/
def S (A : F) (i j : Fin 2013) : ℕ :=
  (Finset.sum (Finset.range 2013) (fun k => A i k)) +
  (Finset.sum (Finset.range 2013) (fun k => A k j)) -
  A i j

/-- The transformation K -/
def K (A : F) : F :=
  fun i j => (S A i j) % 2

/-- The main theorem: K(K(A)) = K(A) for all A in F -/
theorem K_idempotent (A : F) : K (K A) = K A := by sorry

end NUMINAMATH_CALUDE_K_idempotent_l3892_389263


namespace NUMINAMATH_CALUDE_seven_lines_intersection_impossibility_l3892_389228

/-- The maximum number of intersections for n lines in a Euclidean plane -/
def max_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of intersections required for a given number of triple and double intersections -/
def required_intersections (triple_points double_points : ℕ) : ℕ :=
  triple_points * 3 + double_points

theorem seven_lines_intersection_impossibility :
  let n_lines : ℕ := 7
  let min_triple_points : ℕ := 6
  let min_double_points : ℕ := 4
  required_intersections min_triple_points min_double_points > max_intersections n_lines := by
  sorry


end NUMINAMATH_CALUDE_seven_lines_intersection_impossibility_l3892_389228


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3892_389222

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 is √3/2 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let s : ℝ := 6
  let area : ℝ := (s^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * s
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3892_389222


namespace NUMINAMATH_CALUDE_tim_placed_three_pencils_l3892_389236

/-- Given that there were initially 2 pencils in a drawer and after Tim placed some pencils
    there are now 5 pencils in total, prove that Tim placed 3 pencils in the drawer. -/
theorem tim_placed_three_pencils (initial_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : initial_pencils = 2) 
  (h2 : total_pencils = 5) :
  total_pencils - initial_pencils = 3 := by
  sorry

end NUMINAMATH_CALUDE_tim_placed_three_pencils_l3892_389236


namespace NUMINAMATH_CALUDE_wenlock_years_ago_l3892_389235

/-- The year when the Wenlock Olympian Games were first held -/
def wenlock_first_year : ℕ := 1850

/-- The reference year (when the Olympic Games mascot 'Wenlock' was named) -/
def reference_year : ℕ := 2012

/-- The number of years between the first Wenlock Olympian Games and the reference year -/
def years_difference : ℕ := reference_year - wenlock_first_year

theorem wenlock_years_ago : years_difference = 162 := by
  sorry

end NUMINAMATH_CALUDE_wenlock_years_ago_l3892_389235


namespace NUMINAMATH_CALUDE_elliptical_lines_l3892_389262

-- Define the points M and N
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (-1, 0)

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the condition for a point to be on a line
def is_on_line (x y : ℝ) (a b c : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Define what it means for a line to be an "elliptical line"
def is_elliptical_line (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, is_on_line x y a b c ∧ is_on_ellipse x y

theorem elliptical_lines :
  is_elliptical_line 1 (-1) 0 ∧ 
  is_elliptical_line 2 (-1) 1 ∧ 
  ¬is_elliptical_line 1 (-2) 6 ∧ 
  ¬is_elliptical_line 1 1 (-3) :=
sorry

end NUMINAMATH_CALUDE_elliptical_lines_l3892_389262


namespace NUMINAMATH_CALUDE_cards_distribution_l3892_389274

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 52) (h2 : num_people = 9) :
  let base_cards := total_cards / num_people
  let remainder := total_cards % num_people
  let people_with_extra := remainder
  let people_with_base := num_people - people_with_extra
  people_with_base = 2 ∧ base_cards + 1 < 7 := by sorry

end NUMINAMATH_CALUDE_cards_distribution_l3892_389274


namespace NUMINAMATH_CALUDE_max_area_quadrilateral_ellipse_l3892_389287

/-- Given an ellipse with equation x²/a² + y²/b² = 1, where a > 0 and b > 0,
    the maximum area of quadrilateral OAPB is √2/2 * a * b,
    where A is the point on the positive x-axis,
    B is the point on the positive y-axis,
    and P is any point on the ellipse within the first quadrant. -/
theorem max_area_quadrilateral_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let ellipse := {P : ℝ × ℝ | (P.1^2 / a^2) + (P.2^2 / b^2) = 1}
  let A := (a, 0)
  let B := (0, b)
  let valid_P := {P ∈ ellipse | P.1 ≥ 0 ∧ P.2 ≥ 0}
  let area (P : ℝ × ℝ) := (P.1 * P.2 / 2) + ((a - P.1) * b + (0 - P.2) * a) / 2
  (⨆ P ∈ valid_P, area P) = Real.sqrt 2 / 2 * a * b := by
  sorry

end NUMINAMATH_CALUDE_max_area_quadrilateral_ellipse_l3892_389287


namespace NUMINAMATH_CALUDE_equation_solution_l3892_389295

theorem equation_solution : ∃ x : ℝ, (23 - 5 = 3 + x) ∧ (x = 15) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3892_389295


namespace NUMINAMATH_CALUDE_expression_simplification_l3892_389231

theorem expression_simplification (x : ℝ) : (2*x + 1)*(2*x - 1) - x*(4*x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3892_389231


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_one_l3892_389224

theorem sum_of_squares_equals_one 
  (a b c p q r : ℝ) 
  (h1 : a * b = p) 
  (h2 : b * c = q) 
  (h3 : c * a = r) 
  (hp : p ≠ 0) 
  (hq : q ≠ 0) 
  (hr : r ≠ 0) : 
  a^2 + b^2 + c^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_one_l3892_389224


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3892_389288

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt ((x-6)^2 + (y-4)^2) = 14

-- Define what it means for a point to be on the conic
def point_on_conic (x y : ℝ) : Prop :=
  conic_equation x y

-- Define the foci of the conic
def focus1 : ℝ × ℝ := (0, -2)
def focus2 : ℝ × ℝ := (6, 4)

-- Theorem stating that the conic is an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : ℝ), point_on_conic x y ↔
    (x - (focus1.1 + focus2.1) / 2)^2 / a^2 +
    (y - (focus1.2 + focus2.2) / 2)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3892_389288


namespace NUMINAMATH_CALUDE_cos_2α_in_second_quadrant_l3892_389258

theorem cos_2α_in_second_quadrant (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) → 
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) := by
sorry

end NUMINAMATH_CALUDE_cos_2α_in_second_quadrant_l3892_389258


namespace NUMINAMATH_CALUDE_egg_cost_l3892_389239

/-- The cost of breakfast items and breakfasts for Dale and Andrew -/
structure BreakfastCosts where
  toast : ℝ  -- Cost of a slice of toast
  egg : ℝ    -- Cost of an egg
  dale : ℝ   -- Cost of Dale's breakfast
  andrew : ℝ  -- Cost of Andrew's breakfast
  total : ℝ  -- Total cost of both breakfasts

/-- Theorem stating the cost of an egg given the breakfast costs -/
theorem egg_cost (b : BreakfastCosts) 
  (h_toast : b.toast = 1)
  (h_dale : b.dale = 2 * b.toast + 2 * b.egg)
  (h_andrew : b.andrew = b.toast + 2 * b.egg)
  (h_total : b.total = b.dale + b.andrew)
  (h_total_value : b.total = 15) :
  b.egg = 3 := by
  sorry

end NUMINAMATH_CALUDE_egg_cost_l3892_389239


namespace NUMINAMATH_CALUDE_first_sign_distance_l3892_389260

theorem first_sign_distance 
  (total_distance : ℕ) 
  (between_signs : ℕ) 
  (after_second_sign : ℕ) 
  (h1 : total_distance = 1000)
  (h2 : between_signs = 375)
  (h3 : after_second_sign = 275) :
  total_distance - after_second_sign - between_signs = 350 :=
by sorry

end NUMINAMATH_CALUDE_first_sign_distance_l3892_389260


namespace NUMINAMATH_CALUDE_total_fruits_bought_l3892_389284

/-- The total number of fruits bought given the cost and quantity constraints -/
theorem total_fruits_bought
  (total_cost : ℕ)
  (plum_cost peach_cost : ℕ)
  (plum_quantity : ℕ)
  (h1 : total_cost = 52)
  (h2 : plum_cost = 2)
  (h3 : peach_cost = 1)
  (h4 : plum_quantity = 20)
  (h5 : plum_cost * plum_quantity + peach_cost * (total_cost - plum_cost * plum_quantity) = total_cost) :
  plum_quantity + (total_cost - plum_cost * plum_quantity) = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_bought_l3892_389284


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3892_389292

/-- Given a boat that travels 13 km/hr downstream and 4 km/hr upstream,
    prove that its speed in still water is 8.5 km/hr. -/
theorem boat_speed_in_still_water 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : downstream_speed = 13)
  (h2 : upstream_speed = 4) :
  (downstream_speed + upstream_speed) / 2 = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3892_389292


namespace NUMINAMATH_CALUDE_museum_artifact_distribution_l3892_389217

theorem museum_artifact_distribution (total_wings : Nat) 
  (painting_wings : Nat) (large_painting_wing : Nat) 
  (small_painting_wings : Nat) (paintings_per_small_wing : Nat) 
  (artifact_ratio : Nat) :
  total_wings = 8 →
  painting_wings = 3 →
  large_painting_wing = 1 →
  small_painting_wings = 2 →
  paintings_per_small_wing = 12 →
  artifact_ratio = 4 →
  (total_wings - painting_wings) * 
    ((large_painting_wing + small_painting_wings * paintings_per_small_wing) * artifact_ratio / (total_wings - painting_wings)) = 
  (total_wings - painting_wings) * 20 := by
sorry

end NUMINAMATH_CALUDE_museum_artifact_distribution_l3892_389217


namespace NUMINAMATH_CALUDE_tv_sale_value_increase_l3892_389256

theorem tv_sale_value_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_reduction_rate : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : price_reduction_rate = 0.2) 
  (h2 : sales_increase_rate = 0.8) : 
  let new_price := original_price * (1 - price_reduction_rate)
  let new_quantity := original_quantity * (1 + sales_increase_rate)
  let original_value := original_price * original_quantity
  let new_value := new_price * new_quantity
  (new_value - original_value) / original_value = 0.44 := by
sorry

end NUMINAMATH_CALUDE_tv_sale_value_increase_l3892_389256


namespace NUMINAMATH_CALUDE_complex_magnitude_l3892_389216

theorem complex_magnitude (z : ℂ) (h : z * (1 - 2*I) = 3 + I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3892_389216


namespace NUMINAMATH_CALUDE_simplify_trig_ratio_l3892_389297

theorem simplify_trig_ratio : 
  (Real.sin (30 * π / 180) + Real.sin (40 * π / 180)) / 
  (Real.cos (30 * π / 180) + Real.cos (40 * π / 180)) = 
  Real.tan (35 * π / 180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_ratio_l3892_389297


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l3892_389242

theorem shaded_area_between_circles (r₁ r₂ r₃ R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) (h₃ : r₃ = 2)
  (h_external : R = (r₁ + r₂ + r₁ + r₂) / 2)
  (h_tangent : r₁ + r₂ = R - r₁ - r₂) :
  π * R^2 - π * r₁^2 - π * r₂^2 - π * r₃^2 = 36 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l3892_389242


namespace NUMINAMATH_CALUDE_sin_bounds_l3892_389283

theorem sin_bounds :
  (∀ x : ℝ, -5 ≤ 2 * Real.sin x - 3 ∧ 2 * Real.sin x - 3 ≤ -1) ∧
  (∃ x y : ℝ, 2 * Real.sin x - 3 = -5 ∧ 2 * Real.sin y - 3 = -1) := by sorry

end NUMINAMATH_CALUDE_sin_bounds_l3892_389283


namespace NUMINAMATH_CALUDE_fresh_corn_processing_capacity_l3892_389210

/-- The daily processing capacity of fresh corn before technological improvement -/
def daily_capacity : ℕ := 2400

/-- The annual processing capacity before technological improvement -/
def annual_capacity : ℕ := 260000

/-- The improvement factor for daily processing capacity -/
def improvement_factor : ℚ := 13/10

/-- The reduction in processing time after improvement (in days) -/
def time_reduction : ℕ := 25

theorem fresh_corn_processing_capacity :
  daily_capacity = 2400 ∧
  annual_capacity = 260000 ∧
  (annual_capacity : ℚ) / daily_capacity - 
    (annual_capacity : ℚ) / (improvement_factor * daily_capacity) = time_reduction := by
  sorry

#check fresh_corn_processing_capacity

end NUMINAMATH_CALUDE_fresh_corn_processing_capacity_l3892_389210


namespace NUMINAMATH_CALUDE_average_of_rst_l3892_389238

theorem average_of_rst (r s t : ℝ) (h : (5 / 4) * (r + s + t) = 20) :
  (r + s + t) / 3 = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_rst_l3892_389238


namespace NUMINAMATH_CALUDE_luke_bought_twelve_stickers_l3892_389261

/-- The number of stickers Luke bought from the store -/
def stickers_bought (initial : ℕ) (birthday : ℕ) (given_away : ℕ) (used : ℕ) (remaining : ℕ) : ℕ :=
  remaining + given_away + used - initial - birthday

/-- Theorem stating that Luke bought 12 stickers from the store -/
theorem luke_bought_twelve_stickers :
  stickers_bought 20 20 5 8 39 = 12 := by
  sorry

end NUMINAMATH_CALUDE_luke_bought_twelve_stickers_l3892_389261


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3892_389208

theorem no_integer_solutions : ¬∃ (x y z : ℤ), 
  (x^6 + x^3 + x^3*y + y = 147^157) ∧ 
  (x^3 + x^3*y + y^2 + y + z^9 = 157^147) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3892_389208


namespace NUMINAMATH_CALUDE_cars_meeting_time_l3892_389206

/-- Two cars meeting on a highway -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (meeting_time : ℝ) : 
  highway_length = 60 →
  speed1 = 13 →
  speed2 = 17 →
  meeting_time * (speed1 + speed2) = highway_length →
  meeting_time = 2 := by
sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l3892_389206


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_train_passing_jogger_time_approx_l3892_389277

/-- Time for a train to pass a jogger -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The time for the train to pass the jogger is approximately 38.75 seconds -/
theorem train_passing_jogger_time_approx :
  ∃ ε > 0, abs (train_passing_jogger_time 8 60 200 360 - 38.75) < ε :=
sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_train_passing_jogger_time_approx_l3892_389277


namespace NUMINAMATH_CALUDE_spring_fills_sixty_barrels_per_day_l3892_389294

/-- A spring fills barrels of water -/
structure Spring where
  fill_time : ℕ  -- Time to fill one barrel in minutes

/-- A day has a certain number of hours and minutes per hour -/
structure Day where
  hours : ℕ
  minutes_per_hour : ℕ

def barrels_filled_per_day (s : Spring) (d : Day) : ℕ :=
  (d.hours * d.minutes_per_hour) / s.fill_time

/-- Theorem: A spring that fills a barrel in 24 minutes will fill 60 barrels in a day -/
theorem spring_fills_sixty_barrels_per_day (s : Spring) (d : Day) :
  s.fill_time = 24 → d.hours = 24 → d.minutes_per_hour = 60 →
  barrels_filled_per_day s d = 60 := by
  sorry

end NUMINAMATH_CALUDE_spring_fills_sixty_barrels_per_day_l3892_389294


namespace NUMINAMATH_CALUDE_church_full_capacity_l3892_389211

/-- Calculates the number of people needed to fill a church given the number of rows, chairs per row, and people per chair. -/
def church_capacity (rows : ℕ) (chairs_per_row : ℕ) (people_per_chair : ℕ) : ℕ :=
  rows * chairs_per_row * people_per_chair

/-- Theorem stating that a church with 20 rows, 6 chairs per row, and 5 people per chair can hold 600 people. -/
theorem church_full_capacity : church_capacity 20 6 5 = 600 := by
  sorry

end NUMINAMATH_CALUDE_church_full_capacity_l3892_389211


namespace NUMINAMATH_CALUDE_triangle_side_length_l3892_389203

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC is oblique (implied by the other conditions)
  -- Side lengths opposite to angles A, B, C are a, b, c respectively
  A = π / 4 →
  Real.sin A + Real.sin (B - C) = 2 * Real.sqrt 2 * Real.sin (2 * C) →
  (1 / 2) * b * c * Real.sin A = 1 →
  a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3892_389203


namespace NUMINAMATH_CALUDE_buses_passed_count_l3892_389281

/-- Represents the schedule and trip details of buses between Austin and San Antonio -/
structure BusSchedule where
  austin_to_san_antonio_interval : ℕ -- in minutes
  san_antonio_to_austin_interval : ℕ -- in minutes
  san_antonio_to_austin_offset : ℕ -- in minutes
  trip_duration : ℕ -- in minutes

/-- Calculates the number of Austin-bound buses a San Antonio-bound bus passes on the highway -/
def buses_passed (schedule : BusSchedule) : ℕ :=
  sorry

/-- Theorem stating that given the specific bus schedule, a San Antonio-bound bus passes 10 Austin-bound buses -/
theorem buses_passed_count :
  let schedule : BusSchedule := {
    austin_to_san_antonio_interval := 30,
    san_antonio_to_austin_interval := 45,
    san_antonio_to_austin_offset := 15,
    trip_duration := 240
  }
  buses_passed schedule = 10 := by sorry

end NUMINAMATH_CALUDE_buses_passed_count_l3892_389281


namespace NUMINAMATH_CALUDE_largest_four_digit_multiple_of_three_l3892_389207

theorem largest_four_digit_multiple_of_three : ∃ n : ℕ, 
  n = 9999 ∧ 
  n % 3 = 0 ∧ 
  ∀ m : ℕ, m < 10000 ∧ m % 3 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_multiple_of_three_l3892_389207


namespace NUMINAMATH_CALUDE_orange_juice_fraction_is_467_2400_l3892_389232

/-- Represents a pitcher with a specific volume and content --/
structure Pitcher :=
  (volume : ℚ)
  (content : ℚ)

/-- Calculates the fraction of orange juice in the final mixture --/
def orangeJuiceFraction (p1 p2 p3 : Pitcher) : ℚ :=
  let totalVolume := p1.volume + p2.volume + p3.volume
  let orangeJuiceVolume := p1.content + p2.content
  orangeJuiceVolume / totalVolume

/-- Theorem stating the fraction of orange juice in the final mixture --/
theorem orange_juice_fraction_is_467_2400 :
  let p1 : Pitcher := ⟨800, 800 * (1/4)⟩
  let p2 : Pitcher := ⟨800, 800 * (1/3)⟩
  let p3 : Pitcher := ⟨800, 0⟩  -- Third pitcher doesn't contribute to orange juice
  orangeJuiceFraction p1 p2 p3 = 467 / 2400 := by
  sorry

#eval orangeJuiceFraction ⟨800, 800 * (1/4)⟩ ⟨800, 800 * (1/3)⟩ ⟨800, 0⟩

end NUMINAMATH_CALUDE_orange_juice_fraction_is_467_2400_l3892_389232


namespace NUMINAMATH_CALUDE_negation_equivalence_l3892_389254

theorem negation_equivalence :
  ¬(∀ x : ℝ, x^3 - x^2 + 2 < 0) ↔ ∃ x : ℝ, x^3 - x^2 + 2 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3892_389254


namespace NUMINAMATH_CALUDE_min_value_of_function_l3892_389230

theorem min_value_of_function (x : ℝ) (h : x > 5/4) :
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 4*x + 1/(4*x - 5) → y ≥ y_min ∧ y_min = 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3892_389230


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l3892_389296

-- Define the circle with center (8, -3) passing through (5, 1)
def circle1 (x y : ℝ) : Prop := (x - 8)^2 + (y + 3)^2 = 25

-- Define the circle x^2 + y^2 = 25
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the two tangent lines
def tangentLine1 (x y : ℝ) : Prop := y = (4/3) * x - 25/3
def tangentLine2 (x y : ℝ) : Prop := y = (-3/4) * x - 25/4

theorem circle_and_tangent_lines :
  (∀ x y, circle1 x y ↔ ((x = 8 ∧ y = -3) ∨ (x = 5 ∧ y = 1))) ∧
  (∀ x y, tangentLine1 x y → circle2 x y → x = 1 ∧ y = -7) ∧
  (∀ x y, tangentLine2 x y → circle2 x y → x = 1 ∧ y = -7) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l3892_389296


namespace NUMINAMATH_CALUDE_abscissa_range_theorem_l3892_389267

/-- The range of the abscissa of the center of circle M -/
def abscissa_range : Set ℝ := {a | a < 0 ∨ a > 12/5}

/-- The line on which the center of circle M lies -/
def center_line (x y : ℝ) : Prop := 2*x - y - 4 = 0

/-- The equation of circle M with center (a, 2a-4) and radius 1 -/
def circle_M (x y a : ℝ) : Prop := (x - a)^2 + (y - (2*a - 4))^2 = 1

/-- The condition that no point on circle M satisfies NO = 1/2 NA -/
def no_point_condition (x y : ℝ) : Prop := ¬(x^2 + y^2 = 1/4 * (x^2 + (y - 3)^2))

/-- The main theorem statement -/
theorem abscissa_range_theorem (a : ℝ) :
  (∀ x y : ℝ, center_line x y → circle_M x y a → no_point_condition x y) ↔ a ∈ abscissa_range :=
sorry

end NUMINAMATH_CALUDE_abscissa_range_theorem_l3892_389267


namespace NUMINAMATH_CALUDE_vertical_throw_meeting_conditions_l3892_389275

/-- Two objects thrown vertically upwards meet under specific conditions -/
theorem vertical_throw_meeting_conditions 
  (g a b τ : ℝ) (τ' : ℝ) (h_g_pos : g > 0) (h_a_pos : a > 0) (h_τ_pos : τ > 0) (h_τ'_pos : τ' > 0) :
  (b > a - g * τ) ∧ 
  (b > a + (g * τ^2 / 2) / (a/g - τ)) ∧ 
  (b ≥ a / Real.sqrt 2) ∧
  (b ≥ -g * τ' / 2 + Real.sqrt (2 * a^2 - g^2 * τ'^2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_vertical_throw_meeting_conditions_l3892_389275


namespace NUMINAMATH_CALUDE_june_birth_percentage_l3892_389248

/-- The total number of scientists -/
def total_scientists : ℕ := 150

/-- The number of scientists born in June -/
def june_scientists : ℕ := 15

/-- The percentage of scientists born in June -/
def june_percentage : ℚ := (june_scientists : ℚ) / (total_scientists : ℚ) * 100

theorem june_birth_percentage :
  june_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_june_birth_percentage_l3892_389248


namespace NUMINAMATH_CALUDE_committee_rearrangements_count_l3892_389233

/-- The number of distinguishable rearrangements of the letters in "COMMITTEE" with all vowels at the beginning of the sequence -/
def committee_rearrangements : ℕ := sorry

/-- The number of vowels in "COMMITTEE" -/
def num_vowels : ℕ := 4

/-- The number of consonants in "COMMITTEE" -/
def num_consonants : ℕ := 5

/-- The number of repeated vowels (E) in "COMMITTEE" -/
def num_repeated_vowels : ℕ := 2

/-- The number of repeated consonants (M and T) in "COMMITTEE" -/
def num_repeated_consonants : ℕ := 2

theorem committee_rearrangements_count :
  committee_rearrangements = (Nat.factorial num_vowels / Nat.factorial num_repeated_vowels) *
                             (Nat.factorial num_consonants / (Nat.factorial num_repeated_consonants * Nat.factorial num_repeated_consonants)) :=
by sorry

end NUMINAMATH_CALUDE_committee_rearrangements_count_l3892_389233


namespace NUMINAMATH_CALUDE_symmetric_x_axis_coords_symmetric_y_axis_coords_l3892_389218

/-- Given a point M with coordinates (x, y) in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to X-axis -/
def symmetricXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Definition of symmetry with respect to Y-axis -/
def symmetricYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- Theorem: The coordinates of the point symmetric to M(x, y) with respect to the X-axis are (x, -y) -/
theorem symmetric_x_axis_coords (M : Point2D) :
  symmetricXAxis M = { x := M.x, y := -M.y } := by sorry

/-- Theorem: The coordinates of the point symmetric to M(x, y) with respect to the Y-axis are (-x, y) -/
theorem symmetric_y_axis_coords (M : Point2D) :
  symmetricYAxis M = { x := -M.x, y := M.y } := by sorry

end NUMINAMATH_CALUDE_symmetric_x_axis_coords_symmetric_y_axis_coords_l3892_389218


namespace NUMINAMATH_CALUDE_field_dimensions_l3892_389249

theorem field_dimensions (m : ℝ) : (3*m + 11) * m = 100 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_field_dimensions_l3892_389249


namespace NUMINAMATH_CALUDE_log_equality_implies_product_l3892_389212

theorem log_equality_implies_product (x y : ℝ) :
  x > 1 →
  y > 1 →
  (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5) →
  x^2 * y^2 = 225^Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_product_l3892_389212


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_20_l3892_389241

/-- A regular polygon with exterior angles measuring 20 degrees has 18 sides. -/
theorem regular_polygon_exterior_angle_20 : 
  ∀ n : ℕ, 
  n > 0 → 
  (360 : ℝ) / n = 20 → 
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_20_l3892_389241


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l3892_389253

theorem right_triangle_hypotenuse_and_perimeter : 
  ∀ (a b h : ℝ), 
    a = 24 → 
    b = 25 → 
    h^2 = a^2 + b^2 → 
    h = Real.sqrt 1201 ∧ 
    a + b + h = 49 + Real.sqrt 1201 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l3892_389253


namespace NUMINAMATH_CALUDE_original_savings_l3892_389272

/-- Proves that if a person spends 4/5 of their savings on furniture and the remaining 1/5 on a TV that costs $100, their original savings were $500. -/
theorem original_savings (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  furniture_fraction = 4/5 → 
  tv_cost = 100 → 
  (1 - furniture_fraction) * savings = tv_cost → 
  savings = 500 := by
sorry

end NUMINAMATH_CALUDE_original_savings_l3892_389272


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3892_389202

theorem z_in_fourth_quadrant : 
  ∀ z : ℂ, (1 - I) / (z - 2) = 1 + I → 
  (z.re > 0 ∧ z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3892_389202


namespace NUMINAMATH_CALUDE_gcf_72_120_l3892_389226

theorem gcf_72_120 : Nat.gcd 72 120 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcf_72_120_l3892_389226


namespace NUMINAMATH_CALUDE_rectangle_area_l3892_389299

/-- Given a rectangle where the length is thrice the breadth and the perimeter is 120,
    prove that its area is 675. -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let p := 2 * (l + b)
  p = 120 → l * b = 675 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3892_389299


namespace NUMINAMATH_CALUDE_space_between_apple_trees_is_12_l3892_389237

/-- The space needed between apple trees in Quinton's backyard --/
def space_between_apple_trees : ℝ :=
  let total_space : ℝ := 71
  let apple_tree_width : ℝ := 10
  let peach_tree_width : ℝ := 12
  let space_between_peach_trees : ℝ := 15
  let num_apple_trees : ℕ := 2
  let num_peach_trees : ℕ := 2
  let peach_trees_space : ℝ := num_peach_trees * peach_tree_width + space_between_peach_trees
  let apple_trees_space : ℝ := total_space - peach_trees_space
  apple_trees_space - (num_apple_trees * apple_tree_width)

theorem space_between_apple_trees_is_12 :
  space_between_apple_trees = 12 := by
  sorry

end NUMINAMATH_CALUDE_space_between_apple_trees_is_12_l3892_389237


namespace NUMINAMATH_CALUDE_intersection_sum_l3892_389229

/-- Two lines intersect at a point (3,1). -/
def intersection_point : ℝ × ℝ := (3, 1)

/-- The first line equation: x = (1/3)y + a -/
def line1 (a : ℝ) (x y : ℝ) : Prop := x = (1/3) * y + a

/-- The second line equation: y = (1/3)x + b -/
def line2 (b : ℝ) (x y : ℝ) : Prop := y = (1/3) * x + b

/-- The theorem states that if two lines intersect at (3,1), then a + b = 8/3 -/
theorem intersection_sum (a b : ℝ) : 
  line1 a (intersection_point.1) (intersection_point.2) ∧ 
  line2 b (intersection_point.1) (intersection_point.2) → 
  a + b = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_l3892_389229


namespace NUMINAMATH_CALUDE_both_sports_fans_l3892_389213

/-- The number of students who like basketball -/
def basketball_fans : ℕ := 9

/-- The number of students who like cricket -/
def cricket_fans : ℕ := 8

/-- The number of students who like basketball or cricket or both -/
def total_fans : ℕ := 11

/-- The number of students who like both basketball and cricket -/
def both_fans : ℕ := basketball_fans + cricket_fans - total_fans

theorem both_sports_fans : both_fans = 6 := by
  sorry

end NUMINAMATH_CALUDE_both_sports_fans_l3892_389213


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l3892_389214

theorem cubic_sum_theorem (x y : ℝ) 
  (h1 : y + 3 = (x - 3)^2)
  (h2 : x + 3 = (y - 3)^2)
  (h3 : x ≠ y) : 
  x^3 + y^3 = 217 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l3892_389214


namespace NUMINAMATH_CALUDE_total_payment_is_195_l3892_389269

def monthly_rate : ℝ := 50

def discount_rate (month : ℕ) : ℝ :=
  match month with
  | 1 => 0.05
  | 2 => 0.07
  | 3 => 0.10
  | 4 => 0.12
  | _ => 0

def late_fee_rate (month : ℕ) : ℝ :=
  match month with
  | 1 => 0.03
  | 2 => 0.02
  | 3 => 0.04
  | 4 => 0.03
  | _ => 0

def payment_amount (month : ℕ) (on_time : Bool) : ℝ :=
  if on_time then
    monthly_rate * (1 - discount_rate month)
  else
    monthly_rate * (1 + late_fee_rate month)

def total_payment : ℝ :=
  payment_amount 1 true +
  payment_amount 2 false +
  payment_amount 3 true +
  payment_amount 4 false

theorem total_payment_is_195 : total_payment = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_is_195_l3892_389269


namespace NUMINAMATH_CALUDE_square_side_length_l3892_389240

-- Define the rectangle's dimensions
def rectangle_length : ℝ := 7
def rectangle_width : ℝ := 5

-- Define the theorem
theorem square_side_length : 
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  let square_side := rectangle_perimeter / 4
  square_side = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3892_389240


namespace NUMINAMATH_CALUDE_paper_tray_height_l3892_389286

theorem paper_tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 120 →
  cut_distance = Real.sqrt 20 →
  cut_angle = π / 4 →
  ∃ (height : ℝ), height = (800 : ℝ) ^ (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_paper_tray_height_l3892_389286


namespace NUMINAMATH_CALUDE_parabola_transformation_l3892_389257

def original_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

def shift_left (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f (x + k)

def shift_up (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

def transformed_parabola : ℝ → ℝ :=
  shift_up (shift_left original_parabola 3) 2

theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = 2 * x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l3892_389257


namespace NUMINAMATH_CALUDE_total_strings_is_72_l3892_389255

/-- Calculates the total number of strings John needs to restring his instruments. -/
def total_strings : ℕ :=
  let num_basses : ℕ := 3
  let strings_per_bass : ℕ := 4
  let num_guitars : ℕ := 2 * num_basses
  let strings_per_guitar : ℕ := 6
  let num_eight_string_guitars : ℕ := num_guitars - 3
  let strings_per_eight_string_guitar : ℕ := 8
  
  (num_basses * strings_per_bass) +
  (num_guitars * strings_per_guitar) +
  (num_eight_string_guitars * strings_per_eight_string_guitar)

theorem total_strings_is_72 : total_strings = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_strings_is_72_l3892_389255


namespace NUMINAMATH_CALUDE_hair_cut_total_l3892_389251

theorem hair_cut_total (first_cut second_cut : ℝ) 
  (h1 : first_cut = 0.375)
  (h2 : second_cut = 0.5) : 
  first_cut + second_cut = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_total_l3892_389251


namespace NUMINAMATH_CALUDE_inequalities_properties_l3892_389220

theorem inequalities_properties (a b : ℝ) (h : a < b ∧ b < 0) : 
  abs a > abs b ∧ 
  1 / a > 1 / b ∧ 
  a / b + b / a > 2 ∧ 
  a ^ 2 > b ^ 2 := by
sorry

end NUMINAMATH_CALUDE_inequalities_properties_l3892_389220


namespace NUMINAMATH_CALUDE_gold_pucks_count_gold_pucks_theorem_l3892_389298

theorem gold_pucks_count : ℕ → Prop :=
  fun total_gold : ℕ =>
    ∃ (pucks_per_box : ℕ),
      -- Each box has the same number of pucks
      3 * pucks_per_box = 40 + total_gold ∧
      -- One box contains all black pucks and 1/7 of gold pucks
      pucks_per_box = 40 + total_gold / 7 ∧
      -- The number of gold pucks is 140
      total_gold = 140

-- The proof of the theorem
theorem gold_pucks_theorem : gold_pucks_count 140 := by
  sorry

end NUMINAMATH_CALUDE_gold_pucks_count_gold_pucks_theorem_l3892_389298


namespace NUMINAMATH_CALUDE_simplify_expression_l3892_389205

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  12 * x^5 * y / (6 * x * y) = 2 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3892_389205


namespace NUMINAMATH_CALUDE_final_price_is_20_70_l3892_389265

/-- The price of one kilogram of cucumbers in dollars -/
def cucumber_price : ℝ := 5

/-- The price of one kilogram of tomatoes in dollars -/
def tomato_price : ℝ := cucumber_price * (1 - 0.2)

/-- The number of kilograms of tomatoes bought -/
def tomato_kg : ℝ := 2

/-- The number of kilograms of cucumbers bought -/
def cucumber_kg : ℝ := 3

/-- The discount rate applied to the total cost -/
def discount_rate : ℝ := 0.1

/-- The final price paid for the items after discount -/
def final_price : ℝ := (tomato_price * tomato_kg + cucumber_price * cucumber_kg) * (1 - discount_rate)

theorem final_price_is_20_70 : final_price = 20.70 := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_20_70_l3892_389265


namespace NUMINAMATH_CALUDE_sum_of_max_min_values_l3892_389223

theorem sum_of_max_min_values (x y z : ℝ) (h : x^2 + y^2 + z^2 = x + y + z) :
  ∃ (min_val max_val : ℝ),
    (∀ a b c : ℝ, a^2 + b^2 + c^2 = a + b + c → min_val ≤ a + b + c ∧ a + b + c ≤ max_val) ∧
    min_val + max_val = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_values_l3892_389223


namespace NUMINAMATH_CALUDE_price_decrease_revenue_unchanged_l3892_389246

theorem price_decrease_revenue_unchanged (P U : ℝ) (h_positive : P > 0 ∧ U > 0) :
  let new_price := 0.8 * P
  let new_units := U / 0.8
  let percent_decrease_price := 20
  let percent_increase_units := (new_units - U) / U * 100
  P * U = new_price * new_units →
  percent_increase_units / percent_decrease_price = 1.25 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_revenue_unchanged_l3892_389246


namespace NUMINAMATH_CALUDE_watermelon_problem_l3892_389244

theorem watermelon_problem (initial_watermelons : ℕ) (total_watermelons : ℕ) 
  (h1 : initial_watermelons = 4)
  (h2 : total_watermelons = 7) :
  total_watermelons - initial_watermelons = 3 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_problem_l3892_389244


namespace NUMINAMATH_CALUDE_set_operations_l3892_389209

def A : Set ℝ := {x | -x^2 + 2*x + 15 ≤ 0}
def B : Set ℝ := {x | |x - 5| < 1}

theorem set_operations :
  (A ∪ B = {x | x ≤ -3 ∨ x > 4}) ∧
  ((Set.univ \ A) ∩ B = {x | 4 < x ∧ x < 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3892_389209


namespace NUMINAMATH_CALUDE_floor_sqrt_30_squared_l3892_389276

theorem floor_sqrt_30_squared : ⌊Real.sqrt 30⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_30_squared_l3892_389276


namespace NUMINAMATH_CALUDE_intersection_M_N_l3892_389221

def M : Set ℕ := {1, 2, 3, 5, 7}

def N : Set ℕ := {x | ∃ k ∈ M, x = 2 * k - 1}

theorem intersection_M_N : M ∩ N = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3892_389221


namespace NUMINAMATH_CALUDE_triangle_inequality_and_sqrt_sides_l3892_389204

/-- Given a triangle with side lengths a, b, c, prove the existence of a triangle
    with side lengths √a, √b, √c and the inequality involving these lengths. -/
theorem triangle_inequality_and_sqrt_sides {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b + c) (hbc : b ≤ a + c) (hca : c ≤ a + b) :
  (∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ 
    a = v + w ∧ b = u + w ∧ c = u + v) ∧
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c) ∧
  (a + b + c ≤ 2 * Real.sqrt (a * b) + 2 * Real.sqrt (b * c) + 2 * Real.sqrt (c * a)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_sqrt_sides_l3892_389204


namespace NUMINAMATH_CALUDE_number_of_factors_60_l3892_389285

/-- The number of positive factors of 60 is 12. -/
theorem number_of_factors_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_60_l3892_389285


namespace NUMINAMATH_CALUDE_cell_population_growth_l3892_389245

/-- The number of cells in a population after n hours, given the specified conditions -/
def cell_count (n : ℕ) : ℕ :=
  2^(n-1) + 4

/-- Theorem stating that the cell_count function correctly models the cell population growth -/
theorem cell_population_growth (n : ℕ) :
  let initial_cells := 5
  let cells_lost_per_hour := 2
  let division_factor := 2
  cell_count n = (initial_cells - cells_lost_per_hour) * division_factor^n + cells_lost_per_hour :=
by sorry

end NUMINAMATH_CALUDE_cell_population_growth_l3892_389245


namespace NUMINAMATH_CALUDE_even_function_property_l3892_389282

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property 
  (f : ℝ → ℝ) 
  (h_even : even_function f)
  (h_increasing : ∀ x y, x < y → x < 0 → y < 0 → f x < f y)
  (x₁ x₂ : ℝ)
  (h_x₁_neg : x₁ < 0)
  (h_x₂_pos : x₂ > 0)
  (h_abs : abs x₁ < abs x₂) :
  f (-x₁) > f (-x₂) := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l3892_389282


namespace NUMINAMATH_CALUDE_snowboard_discount_proof_l3892_389264

theorem snowboard_discount_proof (original_price : ℝ) (friday_discount : ℝ) (monday_discount : ℝ) :
  original_price = 120 →
  friday_discount = 0.4 →
  monday_discount = 0.2 →
  let friday_price := original_price * (1 - friday_discount)
  let final_price := friday_price * (1 - monday_discount)
  final_price = 57.6 := by
sorry

end NUMINAMATH_CALUDE_snowboard_discount_proof_l3892_389264


namespace NUMINAMATH_CALUDE_binomial_max_remainder_l3892_389234

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem binomial_max_remainder (k : ℕ) (h1 : 30 ≤ k) (h2 : k ≤ 70) :
  ∃ M : ℕ, 
    (∀ j : ℕ, 30 ≤ j → j ≤ 70 → 
      (binomial 100 j) / Nat.gcd (binomial 100 j) (binomial 100 (j+3)) ≤ M) ∧
    M % 1000 = 664 := by
  sorry

end NUMINAMATH_CALUDE_binomial_max_remainder_l3892_389234


namespace NUMINAMATH_CALUDE_three_lines_divide_plane_l3892_389291

/-- A line in the plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at different points -/
def intersect_differently (l1 l2 l3 : Line) : Prop :=
  ¬ parallel l1 l2 ∧ ¬ parallel l1 l3 ∧ ¬ parallel l2 l3 ∧
  (l1.a * l2.b - l1.b * l2.a ≠ 0) ∧
  (l1.a * l3.b - l1.b * l3.a ≠ 0) ∧
  (l2.a * l3.b - l2.b * l3.a ≠ 0)

/-- The three given lines -/
def line1 : Line := ⟨1, 2, -1⟩
def line2 : Line := ⟨1, 0, 1⟩
def line3 (k : ℝ) : Line := ⟨1, k, 0⟩

theorem three_lines_divide_plane (k : ℝ) : 
  intersect_differently line1 line2 (line3 k) ↔ (k = 0 ∨ k = 1 ∨ k = 2) :=
sorry

end NUMINAMATH_CALUDE_three_lines_divide_plane_l3892_389291


namespace NUMINAMATH_CALUDE_initial_stock_calculation_l3892_389243

theorem initial_stock_calculation (sold : ℕ) (unsold_percentage : ℚ) 
  (h1 : sold = 402)
  (h2 : unsold_percentage = 665/1000) : 
  ∃ initial_stock : ℕ, 
    initial_stock = 1200 ∧ 
    (1 - unsold_percentage) * initial_stock = sold :=
by sorry

end NUMINAMATH_CALUDE_initial_stock_calculation_l3892_389243


namespace NUMINAMATH_CALUDE_sum_of_squares_l3892_389289

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3)
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3892_389289


namespace NUMINAMATH_CALUDE_parallel_transitivity_l3892_389271

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define a relation for parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (a b c : Line) :
  parallel a c → parallel b c → parallel a b := by sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l3892_389271


namespace NUMINAMATH_CALUDE_product_of_sums_evaluate_specific_product_l3892_389201

theorem product_of_sums (a b : ℕ) : (a + 1) * (a^2 + 1^2) * (a^4 + 1^4) = ((a^2 - 1^2) * (a^2 + 1^2) * (a^4 - 1^4) * (a^4 + 1^4)) / (a - 1) / 2 := by
  sorry

theorem evaluate_specific_product : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_evaluate_specific_product_l3892_389201


namespace NUMINAMATH_CALUDE_max_regions_40_parabolas_l3892_389247

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure VerticalParabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure HorizontalParabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the maximum number of regions created by a set of vertical and horizontal parabolas -/
def max_regions (vertical_parabolas : Finset VerticalParabola) (horizontal_parabolas : Finset HorizontalParabola) : ℕ :=
  sorry

/-- Theorem stating the maximum number of regions created by 20 vertical and 20 horizontal parabolas -/
theorem max_regions_40_parabolas :
  ∀ (v : Finset VerticalParabola) (h : Finset HorizontalParabola),
  v.card = 20 → h.card = 20 →
  max_regions v h = 2422 :=
by sorry

end NUMINAMATH_CALUDE_max_regions_40_parabolas_l3892_389247


namespace NUMINAMATH_CALUDE_all_positive_integers_are_valid_l3892_389266

-- Define a coloring of the infinite grid
def Coloring := ℤ → ℤ → Bool

-- Define a rectangle on the grid
structure Rectangle where
  x : ℤ
  y : ℤ
  width : ℕ+
  height : ℕ+

-- Count the number of red cells in a rectangle
def countRedCells (c : Coloring) (r : Rectangle) : ℕ :=
  sorry

-- Define the property that all n-cell rectangles have an odd number of red cells
def validColoring (n : ℕ+) (c : Coloring) : Prop :=
  ∀ r : Rectangle, r.width * r.height = n → Odd (countRedCells c r)

-- The main theorem
theorem all_positive_integers_are_valid :
  ∀ n : ℕ+, ∃ c : Coloring, validColoring n c :=
sorry

end NUMINAMATH_CALUDE_all_positive_integers_are_valid_l3892_389266


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3892_389268

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3892_389268


namespace NUMINAMATH_CALUDE_power_equation_solution_l3892_389219

theorem power_equation_solution : 2^90 * 8^90 = 64^(90 - 30) := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3892_389219


namespace NUMINAMATH_CALUDE_card_count_proof_l3892_389225

/-- The number of cards Sasha added to the box -/
def cards_added : ℕ := 48

/-- The fraction of cards Karen removed from what Sasha added -/
def removal_fraction : ℚ := 1 / 6

/-- The number of cards in the box after Sasha's and Karen's actions -/
def final_card_count : ℕ := 83

/-- The original number of cards in the box -/
def original_card_count : ℕ := 75

theorem card_count_proof :
  (cards_added : ℚ) - removal_fraction * cards_added + original_card_count = final_card_count :=
sorry

end NUMINAMATH_CALUDE_card_count_proof_l3892_389225


namespace NUMINAMATH_CALUDE_fraction_simplification_l3892_389293

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 200 + 3 * Real.sqrt 50 + 5) = (5 * Real.sqrt 2 - 1) / 49 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3892_389293
