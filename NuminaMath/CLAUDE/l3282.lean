import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_possible_intersection_counts_l3282_328216

/-- A configuration of five distinct lines in a plane -/
structure LineConfiguration where
  lines : Finset (Set ℝ × ℝ)
  distinct : lines.card = 5

/-- The number of distinct intersection points in a configuration -/
def intersectionPoints (config : LineConfiguration) : ℕ :=
  sorry

/-- The set of all possible values for the number of intersection points -/
def possibleIntersectionCounts : Finset ℕ :=
  sorry

/-- Theorem: The sum of all possible values for the number of intersection points is 53 -/
theorem sum_of_possible_intersection_counts :
  (possibleIntersectionCounts.sum id) = 53 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_intersection_counts_l3282_328216


namespace NUMINAMATH_CALUDE_only_fatigued_drivers_accidents_correlative_l3282_328247

/-- Represents a pair of quantities -/
inductive QuantityPair
  | StudentGradesWeight
  | TimeDisplacement
  | WaterVolumeWeight
  | FatiguedDriversAccidents

/-- Describes the relationship between two quantities -/
inductive Relationship
  | Correlative
  | Functional
  | Independent

/-- Function that determines the relationship for a given pair of quantities -/
def determineRelationship (pair : QuantityPair) : Relationship :=
  match pair with
  | QuantityPair.StudentGradesWeight => Relationship.Independent
  | QuantityPair.TimeDisplacement => Relationship.Functional
  | QuantityPair.WaterVolumeWeight => Relationship.Functional
  | QuantityPair.FatiguedDriversAccidents => Relationship.Correlative

/-- Theorem stating that only the FatiguedDriversAccidents pair has a correlative relationship -/
theorem only_fatigued_drivers_accidents_correlative :
  ∀ (pair : QuantityPair),
    determineRelationship pair = Relationship.Correlative ↔ pair = QuantityPair.FatiguedDriversAccidents :=
by
  sorry


end NUMINAMATH_CALUDE_only_fatigued_drivers_accidents_correlative_l3282_328247


namespace NUMINAMATH_CALUDE_circular_permutations_2a2b2c_l3282_328219

/-- The number of first-type circular permutations for a multiset with given element counts -/
def circularPermutations (counts : List Nat) : Nat :=
  sorry

/-- Theorem: The number of first-type circular permutations for 2 a's, 2 b's, and 2 c's is 16 -/
theorem circular_permutations_2a2b2c :
  circularPermutations [2, 2, 2] = 16 := by
  sorry

end NUMINAMATH_CALUDE_circular_permutations_2a2b2c_l3282_328219


namespace NUMINAMATH_CALUDE_polynomial_not_factorizable_l3282_328249

theorem polynomial_not_factorizable : 
  ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ), 
    x^4 + 3*x^3 + 6*x^2 + 9*x + 12 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_factorizable_l3282_328249


namespace NUMINAMATH_CALUDE_triangle_angle_bounds_l3282_328275

theorem triangle_angle_bounds (y : ℝ) : 
  y > 0 → 
  y + 10 > y + 5 → 
  y + 10 > 4 * y →
  y + 5 + 4 * y > y + 10 →
  y + 5 + y + 10 > 4 * y →
  4 * y + y + 10 > y + 5 →
  (∃ (p q : ℝ), p < y ∧ y < q ∧ 
    (∀ (p' q' : ℝ), p' < y ∧ y < q' → q' - p' ≥ q - p) ∧
    q - p = 25 / 12) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_bounds_l3282_328275


namespace NUMINAMATH_CALUDE_area_equals_scientific_notation_l3282_328272

-- Define the area of the radio telescope
def telescope_area : ℝ := 250000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.5 * (10 ^ 5)

-- Theorem stating that the area is equal to its scientific notation representation
theorem area_equals_scientific_notation : telescope_area = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_area_equals_scientific_notation_l3282_328272


namespace NUMINAMATH_CALUDE_junior_score_l3282_328224

theorem junior_score (n : ℝ) (junior_ratio : ℝ) (senior_ratio : ℝ) 
  (class_avg : ℝ) (senior_avg : ℝ) (h1 : junior_ratio = 0.2) 
  (h2 : senior_ratio = 0.8) (h3 : junior_ratio + senior_ratio = 1) 
  (h4 : class_avg = 84) (h5 : senior_avg = 82) : 
  (class_avg * n - senior_avg * senior_ratio * n) / (junior_ratio * n) = 92 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l3282_328224


namespace NUMINAMATH_CALUDE_jerrys_age_l3282_328298

/-- Given that Mickey's age is 20 years old and 10 years more than 200% of Jerry's age, prove that Jerry is 5 years old. -/
theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 20 → 
  mickey_age = 2 * jerry_age + 10 → 
  jerry_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_age_l3282_328298


namespace NUMINAMATH_CALUDE_y_order_on_quadratic_l3282_328215

/-- A quadratic function of the form y = x² + 4x + k -/
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + k

/-- Theorem stating the order of y-coordinates for specific x-values on the quadratic function -/
theorem y_order_on_quadratic (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : quadratic_function k (-4) = y₁)
  (h₂ : quadratic_function k (-1) = y₂)
  (h₃ : quadratic_function k 1 = y₃) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry


end NUMINAMATH_CALUDE_y_order_on_quadratic_l3282_328215


namespace NUMINAMATH_CALUDE_bowl_glass_pairings_l3282_328232

theorem bowl_glass_pairings :
  let num_bowls : ℕ := 5
  let num_glasses : ℕ := 4
  num_bowls * num_glasses = 20 :=
by sorry

end NUMINAMATH_CALUDE_bowl_glass_pairings_l3282_328232


namespace NUMINAMATH_CALUDE_right_triangle_k_values_l3282_328264

/-- A right triangle ABC with vectors AB and AC -/
structure RightTriangle where
  AB : ℝ × ℝ
  AC : ℝ × ℝ
  is_right : Bool

/-- The possible k values for a right triangle with AB = (2, 3) and AC = (1, k) -/
def possible_k_values : Set ℝ :=
  {-2/3, 11/3, (3 + Real.sqrt 13)/2, (3 - Real.sqrt 13)/2}

/-- Theorem stating that k must be one of the possible values -/
theorem right_triangle_k_values (triangle : RightTriangle) 
  (h1 : triangle.AB = (2, 3)) 
  (h2 : triangle.AC = (1, triangle.AC.snd)) 
  (h3 : triangle.is_right = true) : 
  triangle.AC.snd ∈ possible_k_values := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_k_values_l3282_328264


namespace NUMINAMATH_CALUDE_coloring_existence_and_impossibility_l3282_328208

def is_monochromatic (color : ℕ → Bool) (x y z : ℕ) : Prop :=
  color x = color y ∧ color y = color z

theorem coloring_existence_and_impossibility :
  (∃ (color : ℕ → Bool),
    ∀ x y z, 1 ≤ x ∧ x ≤ 2017 ∧ 1 ≤ y ∧ y ≤ 2017 ∧ 1 ≤ z ∧ z ≤ 2017 →
      8 * (x + y) = z → ¬is_monochromatic color x y z) ∧
  (∀ n : ℕ, n ≥ 2056 →
    ¬∃ (color : ℕ → Bool),
      ∀ x y z, 1 ≤ x ∧ x ≤ n ∧ 1 ≤ y ∧ y ≤ n ∧ 1 ≤ z ∧ z ≤ n →
        8 * (x + y) = z → ¬is_monochromatic color x y z) :=
by sorry

end NUMINAMATH_CALUDE_coloring_existence_and_impossibility_l3282_328208


namespace NUMINAMATH_CALUDE_sum_precision_l3282_328213

theorem sum_precision (n : ℕ) (h : n ≤ 5) :
  ∃ (e : ℝ), e ≤ 0.001 ∧ n * e ≤ 0.01 :=
by sorry

end NUMINAMATH_CALUDE_sum_precision_l3282_328213


namespace NUMINAMATH_CALUDE_line_through_three_points_l3282_328237

/-- A line passes through three points: (2, 5), (-3, m), and (15, -1).
    This theorem proves that the value of m is 95/13. -/
theorem line_through_three_points (m : ℚ) : 
  (∃ (line : ℝ → ℝ), 
    line 2 = 5 ∧ 
    line (-3) = m ∧ 
    line 15 = -1) → 
  m = 95 / 13 :=
by sorry

end NUMINAMATH_CALUDE_line_through_three_points_l3282_328237


namespace NUMINAMATH_CALUDE_ninth_root_unity_sum_l3282_328228

theorem ninth_root_unity_sum (z : ℂ) : 
  z = Complex.exp (2 * Real.pi * I / 9) →
  z^9 = 1 →
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ninth_root_unity_sum_l3282_328228


namespace NUMINAMATH_CALUDE_trig_inequality_l3282_328286

theorem trig_inequality (α β γ : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : 0 < γ ∧ γ < Real.pi / 2)
  (h4 : Real.sin α ^ 3 + Real.sin β ^ 3 + Real.sin γ ^ 3 = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 * Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_inequality_l3282_328286


namespace NUMINAMATH_CALUDE_solve_brownies_problem_l3282_328297

def brownies_problem (total : ℕ) (to_admin : ℕ) (to_simon : ℕ) (left : ℕ) : Prop :=
  let remaining_after_admin := total - to_admin
  let to_carl := remaining_after_admin - to_simon - left
  (to_carl : ℚ) / remaining_after_admin = 1 / 2

theorem solve_brownies_problem :
  brownies_problem 20 10 2 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_brownies_problem_l3282_328297


namespace NUMINAMATH_CALUDE_watch_loss_percentage_l3282_328267

/-- Calculates the loss percentage for a watch sale given specific conditions. -/
def loss_percentage (cost_price selling_price increased_price : ℚ) : ℚ :=
  let gain_percentage : ℚ := 2 / 100
  let price_difference : ℚ := increased_price - selling_price
  let loss : ℚ := cost_price - selling_price
  (loss / cost_price) * 100

/-- Theorem stating that the loss percentage is 10% under given conditions. -/
theorem watch_loss_percentage : 
  let cost_price : ℚ := 1166.67
  let selling_price : ℚ := cost_price - 116.67
  let increased_price : ℚ := selling_price + 140
  loss_percentage cost_price selling_price increased_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_watch_loss_percentage_l3282_328267


namespace NUMINAMATH_CALUDE_zola_paityn_blue_hat_ratio_l3282_328250

/-- Proves the ratio of Zola's blue hats to Paityn's blue hats -/
theorem zola_paityn_blue_hat_ratio :
  let paityn_red : ℕ := 20
  let paityn_blue : ℕ := 24
  let zola_red : ℕ := (4 * paityn_red) / 5
  let total_hats : ℕ := 54 * 2
  let zola_blue : ℕ := total_hats - paityn_red - paityn_blue - zola_red
  (zola_blue : ℚ) / paityn_blue = 2 := by
  sorry

end NUMINAMATH_CALUDE_zola_paityn_blue_hat_ratio_l3282_328250


namespace NUMINAMATH_CALUDE_harrison_croissant_expenditure_l3282_328211

/-- The cost of a regular croissant in dollars -/
def regular_croissant_cost : ℚ := 7/2

/-- The cost of an almond croissant in dollars -/
def almond_croissant_cost : ℚ := 11/2

/-- The number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- The total amount Harrison spends on croissants in a year -/
def total_spent_on_croissants : ℚ := 
  (regular_croissant_cost * weeks_in_year) + (almond_croissant_cost * weeks_in_year)

theorem harrison_croissant_expenditure : 
  total_spent_on_croissants = 468 := by sorry

end NUMINAMATH_CALUDE_harrison_croissant_expenditure_l3282_328211


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l3282_328273

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- The longer base of the trapezoid -/
  longerBase : ℝ
  /-- One of the base angles of the trapezoid -/
  baseAngle : ℝ
  /-- The height of the trapezoid -/
  height : ℝ

/-- The area of the isosceles trapezoid -/
def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  ∃ (t : IsoscelesTrapezoid),
    t.longerBase = 20 ∧
    t.baseAngle = Real.arcsin 0.6 ∧
    t.height = 9 ∧
    trapezoidArea t = 100 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l3282_328273


namespace NUMINAMATH_CALUDE_illuminated_cube_surface_area_l3282_328280

/-- The area of the illuminated part of a cube's surface when a cylindrical beam of light is directed along its main diagonal -/
theorem illuminated_cube_surface_area 
  (a : ℝ) 
  (ρ : ℝ) 
  (h_a : a = 1 / Real.sqrt 2) 
  (h_ρ : ρ = Real.sqrt (2 - Real.sqrt 3)) : 
  ∃ (area : ℝ), area = (Real.sqrt 3 - 3/2) * (Real.pi + 3) := by
  sorry

end NUMINAMATH_CALUDE_illuminated_cube_surface_area_l3282_328280


namespace NUMINAMATH_CALUDE_bananas_per_friend_l3282_328222

def virginia_bananas : ℕ := 40
def virginia_marbles : ℕ := 4
def number_of_friends : ℕ := 40

theorem bananas_per_friend :
  virginia_bananas / number_of_friends = 1 :=
sorry

end NUMINAMATH_CALUDE_bananas_per_friend_l3282_328222


namespace NUMINAMATH_CALUDE_ball_distribution_after_199_students_l3282_328287

/-- Represents the state of the boxes -/
structure BoxState :=
  (a b c d e : ℕ)

/-- Simulates one student's action -/
def moveOneBall (state : BoxState) : BoxState :=
  let minBox := min state.a (min state.b (min state.c (min state.d state.e)))
  { a := if state.a > minBox then state.a - 1 else state.a + 4,
    b := if state.b > minBox then state.b - 1 else state.b + 4,
    c := if state.c > minBox then state.c - 1 else state.c + 4,
    d := if state.d > minBox then state.d - 1 else state.d + 4,
    e := if state.e > minBox then state.e - 1 else state.e + 4 }

/-- Simulates n students' actions -/
def simulateNStudents (n : ℕ) (initialState : BoxState) : BoxState :=
  match n with
  | 0 => initialState
  | n + 1 => moveOneBall (simulateNStudents n initialState)

/-- The main theorem to prove -/
theorem ball_distribution_after_199_students :
  let initialState : BoxState := ⟨9, 5, 3, 2, 1⟩
  let finalState := simulateNStudents 199 initialState
  finalState = ⟨5, 6, 4, 3, 2⟩ := by
  sorry


end NUMINAMATH_CALUDE_ball_distribution_after_199_students_l3282_328287


namespace NUMINAMATH_CALUDE_proportional_segments_l3282_328236

theorem proportional_segments (a b c d : ℝ) :
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a / b = c / d) →
  a = 4 →
  b = 2 →
  c = 3 →
  d = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_proportional_segments_l3282_328236


namespace NUMINAMATH_CALUDE_lcm_problem_l3282_328212

theorem lcm_problem (a b c : ℕ+) (ha : a = 24) (hb : b = 36) (hlcm : Nat.lcm (Nat.lcm a b) c = 360) : c = 5 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3282_328212


namespace NUMINAMATH_CALUDE_team_average_weight_l3282_328288

theorem team_average_weight 
  (num_forwards : ℕ) 
  (num_defensemen : ℕ) 
  (avg_weight_forwards : ℝ) 
  (avg_weight_defensemen : ℝ) 
  (h1 : num_forwards = 8)
  (h2 : num_defensemen = 12)
  (h3 : avg_weight_forwards = 75)
  (h4 : avg_weight_defensemen = 82) :
  let total_players := num_forwards + num_defensemen
  let total_weight := num_forwards * avg_weight_forwards + num_defensemen * avg_weight_defensemen
  total_weight / total_players = 79.2 := by
  sorry

end NUMINAMATH_CALUDE_team_average_weight_l3282_328288


namespace NUMINAMATH_CALUDE_ten_people_round_table_arrangements_l3282_328269

/-- The number of unique seating arrangements for n people around a round table,
    considering rotations as identical. -/
def uniqueRoundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem stating that the number of unique seating arrangements for 10 people
    around a round table, considering rotations as identical, is 362,880. -/
theorem ten_people_round_table_arrangements :
  uniqueRoundTableArrangements 10 = 362880 := by sorry

end NUMINAMATH_CALUDE_ten_people_round_table_arrangements_l3282_328269


namespace NUMINAMATH_CALUDE_markus_marbles_l3282_328292

theorem markus_marbles (mara_bags : ℕ) (mara_marbles_per_bag : ℕ) 
  (markus_bags : ℕ) (markus_extra_marbles : ℕ) :
  mara_bags = 12 →
  mara_marbles_per_bag = 2 →
  markus_bags = 2 →
  markus_extra_marbles = 2 →
  (mara_bags * mara_marbles_per_bag + markus_extra_marbles) / markus_bags = 13 := by
  sorry

end NUMINAMATH_CALUDE_markus_marbles_l3282_328292


namespace NUMINAMATH_CALUDE_final_black_fraction_is_512_729_l3282_328241

/-- Represents the fraction of black area remaining after one change -/
def remaining_black_fraction : ℚ := 8 / 9

/-- Represents the number of changes applied to the triangle -/
def num_changes : ℕ := 3

/-- Represents the fraction of the original area that remains black after the specified number of changes -/
def final_black_fraction : ℚ := remaining_black_fraction ^ num_changes

/-- Theorem stating that the final black fraction is equal to 512/729 -/
theorem final_black_fraction_is_512_729 : 
  final_black_fraction = 512 / 729 := by sorry

end NUMINAMATH_CALUDE_final_black_fraction_is_512_729_l3282_328241


namespace NUMINAMATH_CALUDE_new_pet_ratio_l3282_328299

/-- Represents the number of pets of each type -/
structure PetCount where
  dogs : ℕ
  cats : ℕ
  birds : ℕ

/-- Calculates the new pet count after changes -/
def newPetCount (initial : PetCount) : PetCount :=
  { dogs := initial.dogs - 15,
    cats := initial.cats + 4 - 12,
    birds := initial.birds + 7 - 5 }

/-- Theorem stating the new ratio of pets after changes -/
theorem new_pet_ratio (initial : PetCount) :
  initial.dogs + initial.cats + initial.birds = 315 →
  initial.dogs * 35 = 315 * 10 →
  initial.cats * 35 = 315 * 17 →
  initial.birds * 35 = 315 * 8 →
  let final := newPetCount initial
  (final.dogs, final.cats, final.birds) = (75, 145, 74) :=
by sorry

end NUMINAMATH_CALUDE_new_pet_ratio_l3282_328299


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_neg_one_l3282_328259

theorem sum_of_roots_eq_neg_one (m n : ℝ) : 
  m ≠ 0 → 
  n ≠ 0 → 
  (∀ x : ℝ, x ≠ 0 → 1 / x^2 + m / x + n = 0) → 
  m * n = 1 → 
  m + n = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_neg_one_l3282_328259


namespace NUMINAMATH_CALUDE_calculate_b_investment_l3282_328271

/-- Calculates B's investment in a partnership given the investments of A and C, 
    the total profit, and A's share of the profit. -/
theorem calculate_b_investment (a_investment c_investment total_profit a_profit : ℕ) : 
  a_investment = 6300 →
  c_investment = 10500 →
  total_profit = 14200 →
  a_profit = 4260 →
  ∃ b_investment : ℕ, 
    b_investment = 4220 ∧ 
    (a_investment : ℚ) / (a_investment + b_investment + c_investment) = 
    (a_profit : ℚ) / total_profit :=
by sorry

end NUMINAMATH_CALUDE_calculate_b_investment_l3282_328271


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3282_328246

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 + 16 * x + 29

-- Define the y-coordinate of the vertex
def vertex_y : ℝ := -3

-- Theorem statement
theorem parabola_vertex_y_coordinate :
  ∃ x : ℝ, ∀ t : ℝ, f t ≥ f x ∧ f x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3282_328246


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3282_328231

theorem complex_number_quadrant (z : ℂ) (h : (1 - Complex.I)^2 / z = 1 + Complex.I) :
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3282_328231


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3282_328289

/-- Proves that 2370000 is equal to 2.37 × 10^6 in scientific notation -/
theorem scientific_notation_equivalence :
  2370000 = 2.37 * (10 : ℝ)^6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3282_328289


namespace NUMINAMATH_CALUDE_g_difference_theorem_l3282_328293

/-- The function g(x) = 3x^2 + x - 4 -/
def g (x : ℝ) : ℝ := 3 * x^2 + x - 4

/-- Theorem stating that [g(x+h) - g(x)] - [g(x) - g(x-h)] = 6h^2 for all real x and h -/
theorem g_difference_theorem (x h : ℝ) : 
  (g (x + h) - g x) - (g x - g (x - h)) = 6 * h^2 := by
  sorry

end NUMINAMATH_CALUDE_g_difference_theorem_l3282_328293


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l3282_328263

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 2/3 ∨ x < -6} :=
sorry

-- Theorem for the range of t
theorem range_of_t (t : ℝ) :
  (∃ x : ℝ, f x < 2 - 7/2 * t) ↔ (t < 3/2 ∨ t > 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l3282_328263


namespace NUMINAMATH_CALUDE_group_size_calculation_l3282_328252

theorem group_size_calculation (n : ℕ) : 
  (n * n = 5929) → n = 77 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l3282_328252


namespace NUMINAMATH_CALUDE_jump_rope_time_ratio_l3282_328266

/-- Given information about jump rope times for Cindy, Betsy, and Tina, 
    prove that the ratio of Tina's time to Betsy's time is 3. -/
theorem jump_rope_time_ratio :
  ∀ (cindy_time betsy_time tina_time : ℕ),
    cindy_time = 12 →
    betsy_time = cindy_time / 2 →
    tina_time = cindy_time + 6 →
    tina_time / betsy_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_jump_rope_time_ratio_l3282_328266


namespace NUMINAMATH_CALUDE_smallest_pair_sum_divisible_by_125_l3282_328233

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by 125 -/
def divisible_by_125 (n : ℕ) : Prop := n % 125 = 0

/-- The smallest pair of consecutive numbers with sum of digits divisible by 125 -/
def smallest_pair : ℕ × ℕ := (89999999999998, 89999999999999)

theorem smallest_pair_sum_divisible_by_125 :
  let (a, b) := smallest_pair
  divisible_by_125 (sum_of_digits a) ∧
  divisible_by_125 (sum_of_digits b) ∧
  b = a + 1 ∧
  ∀ (x y : ℕ), x < a → y = x + 1 →
    ¬(divisible_by_125 (sum_of_digits x) ∧ divisible_by_125 (sum_of_digits y)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_pair_sum_divisible_by_125_l3282_328233


namespace NUMINAMATH_CALUDE_stock_price_decrease_l3282_328277

theorem stock_price_decrease (a : ℝ) (n : ℕ) (h1 : a > 0) : a * (0.99 ^ n) < a := by
  sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l3282_328277


namespace NUMINAMATH_CALUDE_days_to_pay_for_cash_register_l3282_328291

/-- Represents the daily sales and costs for Marie's bakery --/
structure BakeryFinances where
  breadPrice : ℝ
  breadQuantity : ℝ
  bagelPrice : ℝ
  bagelQuantity : ℝ
  cakePrice : ℝ
  cakeQuantity : ℝ
  muffinPrice : ℝ
  muffinQuantity : ℝ
  rent : ℝ
  electricity : ℝ
  wages : ℝ
  ingredientCosts : ℝ
  salesTax : ℝ

/-- Calculates the number of days needed to pay for the cash register --/
def daysToPayForCashRegister (finances : BakeryFinances) (cashRegisterCost : ℝ) : ℕ :=
  sorry

/-- Theorem stating that it takes 17 days to pay for the cash register --/
theorem days_to_pay_for_cash_register :
  ∃ (finances : BakeryFinances),
    finances.breadPrice = 2 ∧
    finances.breadQuantity = 40 ∧
    finances.bagelPrice = 1.5 ∧
    finances.bagelQuantity = 20 ∧
    finances.cakePrice = 12 ∧
    finances.cakeQuantity = 6 ∧
    finances.muffinPrice = 3 ∧
    finances.muffinQuantity = 10 ∧
    finances.rent = 20 ∧
    finances.electricity = 2 ∧
    finances.wages = 80 ∧
    finances.ingredientCosts = 30 ∧
    finances.salesTax = 0.08 ∧
    daysToPayForCashRegister finances 1040 = 17 :=
  sorry

end NUMINAMATH_CALUDE_days_to_pay_for_cash_register_l3282_328291


namespace NUMINAMATH_CALUDE_triangle_side_range_l3282_328279

theorem triangle_side_range :
  ∀ x : ℝ, 
    (∃ t : Set (ℝ × ℝ × ℝ), 
      t.Nonempty ∧ 
      (∀ s ∈ t, s.1 = 3 ∧ s.2.1 = 6 ∧ s.2.2 = x) ∧
      (∀ s ∈ t, s.1 + s.2.1 > s.2.2 ∧ s.1 + s.2.2 > s.2.1 ∧ s.2.1 + s.2.2 > s.1)) →
    3 < x ∧ x < 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l3282_328279


namespace NUMINAMATH_CALUDE_potato_bag_weight_l3282_328226

theorem potato_bag_weight (morning_bags : ℕ) (afternoon_bags : ℕ) (total_weight : ℕ) :
  morning_bags = 29 →
  afternoon_bags = 17 →
  total_weight = 322 →
  total_weight / (morning_bags + afternoon_bags) = 7 :=
by sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l3282_328226


namespace NUMINAMATH_CALUDE_granger_bread_loaves_l3282_328204

/-- Represents the grocery items and their prices --/
structure GroceryItems where
  spam_price : ℕ
  peanut_butter_price : ℕ
  bread_price : ℕ

/-- Represents the quantities of items bought --/
structure Quantities where
  spam_cans : ℕ
  peanut_butter_jars : ℕ

/-- Calculates the number of bread loaves bought given the total amount paid --/
def bread_loaves_bought (items : GroceryItems) (quantities : Quantities) (total_paid : ℕ) : ℕ :=
  (total_paid - (items.spam_price * quantities.spam_cans + items.peanut_butter_price * quantities.peanut_butter_jars)) / items.bread_price

/-- Theorem stating that Granger bought 4 loaves of bread --/
theorem granger_bread_loaves :
  let items := GroceryItems.mk 3 5 2
  let quantities := Quantities.mk 12 3
  let total_paid := 59
  bread_loaves_bought items quantities total_paid = 4 := by
  sorry


end NUMINAMATH_CALUDE_granger_bread_loaves_l3282_328204


namespace NUMINAMATH_CALUDE_rectangle_width_l3282_328265

theorem rectangle_width (w : ℝ) (h1 : w > 0) : 
  (2 * w * w = 3 * 2 * (2 * w + w)) → w = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3282_328265


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l3282_328221

theorem average_of_a_and_b (a b : ℝ) : 
  (4 + 6 + 8 + a + b) / 5 = 17 → 
  b = 2 * a → 
  (a + b) / 2 = 33.5 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l3282_328221


namespace NUMINAMATH_CALUDE_village_population_l3282_328229

theorem village_population (P : ℝ) : 
  (P * 1.25 * 0.75 = 18750) → P = 20000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3282_328229


namespace NUMINAMATH_CALUDE_prepaid_card_cost_l3282_328239

/-- The cost of a prepaid phone card given call cost, call duration, and remaining balance -/
theorem prepaid_card_cost 
  (cost_per_minute : ℚ) 
  (call_duration : ℕ) 
  (remaining_balance : ℚ) : 
  cost_per_minute = 16/100 →
  call_duration = 22 →
  remaining_balance = 2648/100 →
  remaining_balance + cost_per_minute * call_duration = 30 := by
sorry

end NUMINAMATH_CALUDE_prepaid_card_cost_l3282_328239


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3282_328281

theorem cone_lateral_surface_area 
  (r : ℝ) 
  (V : ℝ) 
  (h : ℝ) 
  (l : ℝ) 
  (A : ℝ) :
  r = 3 →
  V = 12 * Real.pi →
  V = (1/3) * Real.pi * r^2 * h →
  l^2 = r^2 + h^2 →
  A = Real.pi * r * l →
  A = 15 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3282_328281


namespace NUMINAMATH_CALUDE_x_squared_in_set_l3282_328209

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({1, 0, x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_in_set_l3282_328209


namespace NUMINAMATH_CALUDE_suwy_unique_product_l3282_328244

/-- Represents a letter with its corresponding value -/
structure Letter where
  value : Nat
  h : value ≥ 1 ∧ value ≤ 26

/-- Represents a four-letter list -/
structure FourLetterList where
  letters : Fin 4 → Letter

/-- Calculates the product of a four-letter list -/
def product (list : FourLetterList) : Nat :=
  (list.letters 0).value * (list.letters 1).value * (list.letters 2).value * (list.letters 3).value

theorem suwy_unique_product :
  ∀ (list : FourLetterList),
    product list = 19 * 21 * 23 * 25 →
    (list.letters 0).value = 19 ∧
    (list.letters 1).value = 21 ∧
    (list.letters 2).value = 23 ∧
    (list.letters 3).value = 25 :=
by sorry

end NUMINAMATH_CALUDE_suwy_unique_product_l3282_328244


namespace NUMINAMATH_CALUDE_simplify_fraction_l3282_328278

theorem simplify_fraction : (210 : ℚ) / 315 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3282_328278


namespace NUMINAMATH_CALUDE_gildas_marbles_theorem_l3282_328202

/-- The percentage of marbles Gilda has left after giving away to her friends and family -/
def gildas_remaining_marbles : ℝ :=
  let after_pedro := 1 - 0.30
  let after_ebony := after_pedro * (1 - 0.20)
  let after_jimmy := after_ebony * (1 - 0.15)
  let after_clara := after_jimmy * (1 - 0.10)
  after_clara * 100

/-- Theorem stating that Gilda has 42.84% of her original marbles left -/
theorem gildas_marbles_theorem : 
  ∃ ε > 0, |gildas_remaining_marbles - 42.84| < ε :=
sorry

end NUMINAMATH_CALUDE_gildas_marbles_theorem_l3282_328202


namespace NUMINAMATH_CALUDE_min_tiles_cover_floor_l3282_328214

/-- Represents the dimensions of a rectangle in inches -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle in square inches -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the tile dimensions -/
def tile : Rectangle := { length := 3, width := 4 }

/-- Represents the floor dimensions -/
def floor : Rectangle := { length := 36, width := 60 }

/-- Calculates the number of tiles needed to cover the floor -/
def tilesNeeded (t : Rectangle) (f : Rectangle) : ℕ :=
  (area f) / (area t)

theorem min_tiles_cover_floor :
  tilesNeeded tile floor = 180 := by sorry

end NUMINAMATH_CALUDE_min_tiles_cover_floor_l3282_328214


namespace NUMINAMATH_CALUDE_billy_sandwiches_l3282_328295

theorem billy_sandwiches (billy katelyn chloe : ℕ) : 
  katelyn = billy + 47 →
  chloe = (katelyn : ℚ) / 4 →
  billy + katelyn + chloe = 169 →
  billy = 49 := by
sorry

end NUMINAMATH_CALUDE_billy_sandwiches_l3282_328295


namespace NUMINAMATH_CALUDE_questionnaire_C_count_l3282_328234

def population : ℕ := 960
def sample_size : ℕ := 32
def first_number : ℕ := 9
def questionnaire_A_upper : ℕ := 450
def questionnaire_B_upper : ℕ := 750

theorem questionnaire_C_count :
  let group_size := population / sample_size
  let groups_AB := questionnaire_B_upper / group_size
  sample_size - groups_AB = 7 := by sorry

end NUMINAMATH_CALUDE_questionnaire_C_count_l3282_328234


namespace NUMINAMATH_CALUDE_stating_solutions_eq_partitions_l3282_328218

/-- The number of solutions to the equation in positive integers -/
def numSolutions : ℕ := sorry

/-- The number of partitions of 7 -/
def numPartitions7 : ℕ := sorry

/-- 
Theorem stating that the number of solutions to the equation
a₁(b₁) + a₂(b₁+b₂) + ... + aₖ(b₁+b₂+...+bₖ) = 7
in positive integers (k; a₁, a₂, ..., aₖ; b₁, b₂, ..., bₖ)
is equal to the number of partitions of 7
-/
theorem solutions_eq_partitions : numSolutions = numPartitions7 := by sorry

end NUMINAMATH_CALUDE_stating_solutions_eq_partitions_l3282_328218


namespace NUMINAMATH_CALUDE_altitude_of_equal_area_triangle_trapezoid_l3282_328294

/-- The altitude of a triangle and trapezoid with equal areas -/
theorem altitude_of_equal_area_triangle_trapezoid
  (h : ℝ) -- altitude
  (b : ℝ) -- base of the triangle
  (m : ℝ) -- median of the trapezoid
  (h_pos : h > 0) -- altitude is positive
  (b_val : b = 24) -- base of triangle is 24 inches
  (m_val : m = b / 2) -- median of trapezoid is half of triangle base
  (area_eq : 1/2 * b * h = m * h) -- areas are equal
  : h ∈ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_altitude_of_equal_area_triangle_trapezoid_l3282_328294


namespace NUMINAMATH_CALUDE_inequality_proof_l3282_328235

theorem inequality_proof (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  a * b > a * c ∧ c * b^2 < a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3282_328235


namespace NUMINAMATH_CALUDE_infinitely_many_triples_divisible_by_p_cubed_l3282_328207

theorem infinitely_many_triples_divisible_by_p_cubed :
  ∀ n : ℕ, ∃ p a b : ℕ,
    p > n ∧
    Nat.Prime p ∧
    a < p ∧
    b < p ∧
    (p^3 : ℕ) ∣ ((a + b)^p - a^p - b^p) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_triples_divisible_by_p_cubed_l3282_328207


namespace NUMINAMATH_CALUDE_tricycle_count_l3282_328276

/-- Represents the number of wheels on a scooter -/
def scooter_wheels : ℕ := 2

/-- Represents the number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- Represents the total number of vehicles -/
def total_vehicles : ℕ := 10

/-- Represents the total number of wheels -/
def total_wheels : ℕ := 26

/-- Theorem stating that the number of tricycles must be 6 given the conditions -/
theorem tricycle_count :
  ∃ (scooters tricycles : ℕ),
    scooters + tricycles = total_vehicles ∧
    scooters * scooter_wheels + tricycles * tricycle_wheels = total_wheels ∧
    tricycles = 6 :=
by sorry

end NUMINAMATH_CALUDE_tricycle_count_l3282_328276


namespace NUMINAMATH_CALUDE_intersection_y_coordinate_constant_l3282_328242

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x

-- Define the slope of the tangent at a point
def tangent_slope (p : PointOnParabola) : ℝ := 4 * p.x

-- Define perpendicular tangents
def perpendicular_tangents (p1 p2 : PointOnParabola) : Prop :=
  tangent_slope p1 * tangent_slope p2 = -1

-- Theorem statement
theorem intersection_y_coordinate_constant 
  (A B : PointOnParabola) 
  (h : perpendicular_tangents A B) : 
  ∃ (P : ℝ × ℝ), 
    (P.1 = (A.x + B.x) / 2) ∧ 
    (P.2 = -1/8) ∧
    (P.2 = 4 * A.x * (P.1 - A.x) + A.y) ∧
    (P.2 = 4 * B.x * (P.1 - B.x) + B.y) := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_coordinate_constant_l3282_328242


namespace NUMINAMATH_CALUDE_calculate_second_solution_percentage_l3282_328270

/-- Given two solutions mixed to form a final solution, calculates the percentage of the second solution. -/
theorem calculate_second_solution_percentage
  (final_volume : ℝ)
  (final_percentage : ℝ)
  (first_volume : ℝ)
  (first_percentage : ℝ)
  (second_volume : ℝ)
  (h_final_volume : final_volume = 40)
  (h_final_percentage : final_percentage = 0.45)
  (h_first_volume : first_volume = 28)
  (h_first_percentage : first_percentage = 0.30)
  (h_second_volume : second_volume = 12)
  (h_volume_sum : first_volume + second_volume = final_volume)
  (h_substance_balance : first_volume * first_percentage + second_volume * (second_percentage / 100) = final_volume * final_percentage) :
  second_percentage = 80 := by
  sorry

#check calculate_second_solution_percentage

end NUMINAMATH_CALUDE_calculate_second_solution_percentage_l3282_328270


namespace NUMINAMATH_CALUDE_frog_eyes_in_pond_l3282_328268

/-- The number of eyes a frog has -/
def eyes_per_frog : ℕ := 2

/-- The number of frogs in the pond -/
def frogs_in_pond : ℕ := 4

/-- The total number of frog eyes in the pond -/
def total_frog_eyes : ℕ := frogs_in_pond * eyes_per_frog

theorem frog_eyes_in_pond : total_frog_eyes = 8 := by
  sorry

end NUMINAMATH_CALUDE_frog_eyes_in_pond_l3282_328268


namespace NUMINAMATH_CALUDE_fayes_rows_l3282_328230

/-- Given that Faye has 210 crayons in total and places 30 crayons in each row,
    prove that she created 7 rows. -/
theorem fayes_rows (total_crayons : ℕ) (crayons_per_row : ℕ) (h1 : total_crayons = 210) (h2 : crayons_per_row = 30) :
  total_crayons / crayons_per_row = 7 := by
  sorry

end NUMINAMATH_CALUDE_fayes_rows_l3282_328230


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_2x_l3282_328253

theorem integral_sqrt_minus_2x (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f x = Real.sqrt (1 - (x - 1)^2)) →
  (∀ x, g x = 2 * x) →
  ∫ x in (0 : ℝ)..1, (f x - g x) = π / 4 - 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_2x_l3282_328253


namespace NUMINAMATH_CALUDE_june_election_win_l3282_328201

theorem june_election_win (total_students : ℕ) (boy_percentage : ℚ) (male_vote_percentage : ℚ) :
  total_students = 200 →
  boy_percentage = 60 / 100 →
  male_vote_percentage = 675 / 1000 →
  ∃ (female_vote_percentage : ℚ),
    female_vote_percentage = 25 / 100 ∧
    (⌊total_students * boy_percentage⌋ : ℚ) * male_vote_percentage +
    (total_students - ⌊total_students * boy_percentage⌋ : ℚ) * female_vote_percentage >
    (total_students : ℚ) / 2 ∧
    ∀ (x : ℚ), x < female_vote_percentage →
      (⌊total_students * boy_percentage⌋ : ℚ) * male_vote_percentage +
      (total_students - ⌊total_students * boy_percentage⌋ : ℚ) * x ≤
      (total_students : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_june_election_win_l3282_328201


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l3282_328290

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * (a - 1) * x + 2

-- Define the property of f being decreasing on (-∞, 4]
def isDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 4 → f x > f y

-- State the theorem
theorem f_decreasing_implies_a_range :
  ∀ a : ℝ, isDecreasingOn (f a) ↔ 0 ≤ a ∧ a ≤ 1/5 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l3282_328290


namespace NUMINAMATH_CALUDE_water_in_mixture_l3282_328206

theorem water_in_mixture (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (b * x) / (a + b) = x * (b / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_water_in_mixture_l3282_328206


namespace NUMINAMATH_CALUDE_ten_points_chords_l3282_328285

/-- The number of chords that can be drawn from n points on a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: There are 45 different chords that can be drawn from 10 points on a circle -/
theorem ten_points_chords : num_chords 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_points_chords_l3282_328285


namespace NUMINAMATH_CALUDE_infinite_pairs_with_same_prime_factors_l3282_328258

theorem infinite_pairs_with_same_prime_factors :
  ∀ k : ℕ, k > 1 →
  ∃ m n : ℕ, m ≠ n ∧ m > 0 ∧ n > 0 ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)) ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ (m + 1) ↔ p ∣ (n + 1))) ∧
  m = 2^k - 2 ∧
  n = 2^k * (2^k - 2) :=
sorry

end NUMINAMATH_CALUDE_infinite_pairs_with_same_prime_factors_l3282_328258


namespace NUMINAMATH_CALUDE_orange_eating_contest_l3282_328210

theorem orange_eating_contest (num_students : ℕ) (max_oranges min_oranges : ℕ) :
  num_students = 8 →
  max_oranges = 8 →
  min_oranges = 1 →
  max_oranges - min_oranges = 7 := by
sorry

end NUMINAMATH_CALUDE_orange_eating_contest_l3282_328210


namespace NUMINAMATH_CALUDE_mean_of_six_numbers_with_sum_two_thirds_l3282_328296

theorem mean_of_six_numbers_with_sum_two_thirds :
  ∀ (a b c d e f : ℚ),
  a + b + c + d + e + f = 2/3 →
  (a + b + c + d + e + f) / 6 = 1/9 := by
sorry

end NUMINAMATH_CALUDE_mean_of_six_numbers_with_sum_two_thirds_l3282_328296


namespace NUMINAMATH_CALUDE_at_most_one_super_plus_good_l3282_328282

/-- Represents an 8x8 chessboard with numbers 1 to 64 --/
def Chessboard := Fin 8 → Fin 8 → Fin 64

/-- A number is super-plus-good if it's the largest in its row and smallest in its column --/
def is_super_plus_good (board : Chessboard) (row col : Fin 8) : Prop :=
  (∀ c : Fin 8, board row c ≤ board row col) ∧
  (∀ r : Fin 8, board row col ≤ board r col)

/-- The arrangement is valid if each number appears exactly once --/
def is_valid_arrangement (board : Chessboard) : Prop :=
  ∀ n : Fin 64, ∃! (row col : Fin 8), board row col = n

theorem at_most_one_super_plus_good (board : Chessboard) 
  (h : is_valid_arrangement board) :
  ∃! (row col : Fin 8), is_super_plus_good board row col :=
sorry

end NUMINAMATH_CALUDE_at_most_one_super_plus_good_l3282_328282


namespace NUMINAMATH_CALUDE_union_condition_implies_m_leq_4_l3282_328261

/-- Given sets A and B, if their union equals A, then m ≤ 4 -/
theorem union_condition_implies_m_leq_4 (m : ℝ) : 
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ 7}
  let B := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  (A ∪ B = A) → m ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_union_condition_implies_m_leq_4_l3282_328261


namespace NUMINAMATH_CALUDE_day_150_of_previous_year_is_wednesday_l3282_328274

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Function to determine the day of the week for a given day number in a year -/
def dayOfWeek (year : Year) (dayNumber : ℕ) : DayOfWeek :=
  sorry

/-- Function to determine if a year is a leap year -/
def isLeapYear (year : Year) : Bool :=
  sorry

theorem day_150_of_previous_year_is_wednesday 
  (P : Year)
  (h1 : dayOfWeek P 250 = DayOfWeek.Friday)
  (h2 : dayOfWeek (Year.mk (P.value + 1)) 300 = DayOfWeek.Friday)
  : dayOfWeek (Year.mk (P.value - 1)) 150 = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_day_150_of_previous_year_is_wednesday_l3282_328274


namespace NUMINAMATH_CALUDE_jeff_calculation_correction_l3282_328251

theorem jeff_calculation_correction (incorrect_input : ℕ × ℕ) (incorrect_result : ℕ) 
  (h1 : incorrect_input.1 = 52) 
  (h2 : incorrect_input.2 = 735) 
  (h3 : incorrect_input.1 * incorrect_input.2 = incorrect_result) 
  (h4 : incorrect_result = 38220) : 
  (0.52 : ℝ) * 7.35 = 3.822 := by
sorry

end NUMINAMATH_CALUDE_jeff_calculation_correction_l3282_328251


namespace NUMINAMATH_CALUDE_line_intersects_cubic_at_two_points_l3282_328254

/-- The function representing the cubic curve y = x^3 -/
def cubic_curve (x : ℝ) : ℝ := x^3

/-- The function representing the line y = ax + 16 -/
def line (a x : ℝ) : ℝ := a * x + 16

/-- Predicate to check if the line intersects the curve at exactly two distinct points -/
def intersects_at_two_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    cubic_curve x₁ = line a x₁ ∧
    cubic_curve x₂ = line a x₂ ∧
    ∀ x : ℝ, cubic_curve x = line a x → x = x₁ ∨ x = x₂

theorem line_intersects_cubic_at_two_points (a : ℝ) :
  intersects_at_two_points a → a = 12 := by sorry

end NUMINAMATH_CALUDE_line_intersects_cubic_at_two_points_l3282_328254


namespace NUMINAMATH_CALUDE_eight_squares_sharing_two_vertices_l3282_328256

/-- A square in a 2D plane -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ
  is_square : IsSquare vertices

/-- Two squares share two vertices -/
def SharesTwoVertices (s1 s2 : Square) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧ s1.vertices i = s2.vertices i ∧ s1.vertices j = s2.vertices j

/-- The main theorem -/
theorem eight_squares_sharing_two_vertices (s : Square) :
  ∃ (squares : Finset Square), squares.card = 8 ∧
    ∀ s' ∈ squares, SharesTwoVertices s s' ∧
    ∀ s', SharesTwoVertices s s' → s' ∈ squares :=
  sorry

end NUMINAMATH_CALUDE_eight_squares_sharing_two_vertices_l3282_328256


namespace NUMINAMATH_CALUDE_morios_current_age_l3282_328243

/-- Calculates Morio's current age given the ages of Teresa and Morio at different points in time. -/
theorem morios_current_age
  (teresa_current_age : ℕ)
  (morio_age_at_michikos_birth : ℕ)
  (teresa_age_at_michikos_birth : ℕ)
  (h1 : teresa_current_age = 59)
  (h2 : morio_age_at_michikos_birth = 38)
  (h3 : teresa_age_at_michikos_birth = 26) :
  morio_age_at_michikos_birth + (teresa_current_age - teresa_age_at_michikos_birth) = 71 :=
by sorry

end NUMINAMATH_CALUDE_morios_current_age_l3282_328243


namespace NUMINAMATH_CALUDE_second_class_size_l3282_328203

theorem second_class_size (n : ℕ) (avg_all : ℚ) : 
  n > 0 ∧ 
  (30 : ℚ) * 40 + n * 60 = (30 + n) * avg_all ∧ 
  avg_all = (105 : ℚ) / 2 → 
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_second_class_size_l3282_328203


namespace NUMINAMATH_CALUDE_john_foundation_homes_l3282_328220

/-- Represents the dimensions of a concrete slab for a home foundation -/
structure SlabDimensions where
  length : Float
  width : Float
  height : Float

/-- Calculates the number of homes given foundation parameters -/
def calculateHomes (slab : SlabDimensions) (concreteDensity : Float) (concreteCostPerPound : Float) (totalFoundationCost : Float) : Float :=
  let slabVolume := slab.length * slab.width * slab.height
  let concreteWeight := slabVolume * concreteDensity
  let costPerHome := concreteWeight * concreteCostPerPound
  totalFoundationCost / costPerHome

/-- Proves that John is laying the foundation for 3 homes -/
theorem john_foundation_homes :
  let slab : SlabDimensions := { length := 100, width := 100, height := 0.5 }
  let concreteDensity : Float := 150
  let concreteCostPerPound : Float := 0.02
  let totalFoundationCost : Float := 45000
  calculateHomes slab concreteDensity concreteCostPerPound totalFoundationCost = 3 := by
  sorry


end NUMINAMATH_CALUDE_john_foundation_homes_l3282_328220


namespace NUMINAMATH_CALUDE_tetrahedron_volume_bound_l3282_328227

/-- A tetrahedron is represented by its six edge lengths -/
structure Tetrahedron where
  edges : Fin 6 → ℝ
  edge_positive : ∀ i, edges i > 0

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: For any tetrahedron with only one edge length greater than 1, its volume is at most 1/8 -/
theorem tetrahedron_volume_bound (t : Tetrahedron) 
  (h : ∃! i, t.edges i > 1) : volume t ≤ 1/8 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_bound_l3282_328227


namespace NUMINAMATH_CALUDE_investment_final_value_l3282_328223

def investment_value (initial : ℝ) (w1 w2 w3 w4 w5 w6 : ℝ) : ℝ :=
  initial * (1 + w1) * (1 + w2) * (1 - w3) * (1 + w4) * (1 + w5) * (1 - w6)

theorem investment_final_value :
  let initial : ℝ := 400
  let week1_gain : ℝ := 0.25
  let week2_gain : ℝ := 0.50
  let week3_loss : ℝ := 0.10
  let week4_gain : ℝ := 0.20
  let week5_gain : ℝ := 0.05
  let week6_loss : ℝ := 0.15
  investment_value initial week1_gain week2_gain week3_loss week4_gain week5_gain week6_loss = 722.925 := by
  sorry

end NUMINAMATH_CALUDE_investment_final_value_l3282_328223


namespace NUMINAMATH_CALUDE_apples_in_box_l3282_328217

theorem apples_in_box (initial_apples : ℕ) : 
  (initial_apples / 2 - 25 = 6) → initial_apples = 62 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_box_l3282_328217


namespace NUMINAMATH_CALUDE_fifth_rectangle_is_square_l3282_328283

-- Define the structure of a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the structure of a square
structure Square where
  side : ℝ

-- Define the division of a square into rectangles
def squareDivision (s : Square) (r1 r2 r3 r4 r5 : Rectangle) : Prop :=
  -- The sum of widths and heights of corner rectangles equals the square's side
  r1.width + r2.width = s.side ∧
  r1.height + r3.height = s.side ∧
  -- The areas of the four corner rectangles are equal
  r1.width * r1.height = r2.width * r2.height ∧
  r2.width * r2.height = r3.width * r3.height ∧
  r3.width * r3.height = r4.width * r4.height ∧
  -- The fifth rectangle doesn't touch the sides of the square
  r5.width < s.side - r1.width ∧
  r5.height < s.side - r1.height

-- Theorem statement
theorem fifth_rectangle_is_square 
  (s : Square) (r1 r2 r3 r4 r5 : Rectangle) 
  (h : squareDivision s r1 r2 r3 r4 r5) : 
  r5.width = r5.height :=
sorry

end NUMINAMATH_CALUDE_fifth_rectangle_is_square_l3282_328283


namespace NUMINAMATH_CALUDE_cylinder_height_from_balls_l3282_328257

/-- The height of a cylinder formed by melting steel balls -/
theorem cylinder_height_from_balls (num_balls : ℕ) (ball_radius cylinder_radius : ℝ) :
  num_balls = 12 →
  ball_radius = 2 →
  cylinder_radius = 3 →
  (4 / 3 * π * num_balls * ball_radius ^ 3) / (π * cylinder_radius ^ 2) = 128 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_from_balls_l3282_328257


namespace NUMINAMATH_CALUDE_equal_distribution_of_stickers_l3282_328284

/-- The number of stickers Haley has -/
def total_stickers : ℕ := 72

/-- The number of Haley's friends -/
def num_friends : ℕ := 9

/-- The number of stickers each friend will receive -/
def stickers_per_friend : ℕ := total_stickers / num_friends

theorem equal_distribution_of_stickers :
  stickers_per_friend * num_friends = total_stickers :=
by sorry

end NUMINAMATH_CALUDE_equal_distribution_of_stickers_l3282_328284


namespace NUMINAMATH_CALUDE_uncle_ben_eggs_l3282_328240

theorem uncle_ben_eggs (total_chickens : ℕ) (roosters : ℕ) (non_laying_hens : ℕ) (eggs_per_hen : ℕ) 
  (h1 : total_chickens = 440)
  (h2 : roosters = 39)
  (h3 : non_laying_hens = 15)
  (h4 : eggs_per_hen = 3) :
  total_chickens - roosters - non_laying_hens * eggs_per_hen = 1158 := by
  sorry

end NUMINAMATH_CALUDE_uncle_ben_eggs_l3282_328240


namespace NUMINAMATH_CALUDE_copy_pages_theorem_l3282_328248

/-- Given a cost per page in cents and a budget in dollars, 
    calculate the number of pages that can be copied. -/
def pages_copied (cost_per_page : ℕ) (budget : ℕ) : ℕ :=
  (budget * 100) / cost_per_page

/-- Theorem: With a cost of 3 cents per page and a budget of $15, 
    500 pages can be copied. -/
theorem copy_pages_theorem : pages_copied 3 15 = 500 := by
  sorry

end NUMINAMATH_CALUDE_copy_pages_theorem_l3282_328248


namespace NUMINAMATH_CALUDE_at_least_one_third_l3282_328200

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  a ≥ 1/3 ∨ b ≥ 1/3 ∨ c ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_third_l3282_328200


namespace NUMINAMATH_CALUDE_root_values_l3282_328262

-- Define the polynomial
def polynomial (a b c : ℝ) (x : ℂ) : ℂ := x^3 + a*x^2 + b*x - c

-- State the theorem
theorem root_values (a b c : ℝ) :
  (polynomial a b c (1 - 2*I) = 0) →
  (polynomial a b c (2 - I) = 0) →
  (a, b, c) = (-6, 21, -30) := by
  sorry

end NUMINAMATH_CALUDE_root_values_l3282_328262


namespace NUMINAMATH_CALUDE_hospital_staff_count_l3282_328225

theorem hospital_staff_count (doctors nurses : ℕ) (h1 : doctors * 9 = nurses * 5) (h2 : nurses = 180) :
  doctors + nurses = 280 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l3282_328225


namespace NUMINAMATH_CALUDE_train_meeting_distance_l3282_328205

/-- Proves that when two trains starting 200 miles apart and traveling towards each other
    at 20 miles per hour each meet, one train will have traveled 100 miles. -/
theorem train_meeting_distance (total_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) 
  (h1 : total_distance = 200)
  (h2 : speed_a = 20)
  (h3 : speed_b = 20) :
  speed_a * (total_distance / (speed_a + speed_b)) = 100 :=
by sorry

end NUMINAMATH_CALUDE_train_meeting_distance_l3282_328205


namespace NUMINAMATH_CALUDE_greatest_of_three_integers_l3282_328255

theorem greatest_of_three_integers (a b c : ℤ) : 
  a + b + c = 21 → 
  c = max a (max b c) →
  c = 8 →
  max a (max b c) = 8 := by
sorry

end NUMINAMATH_CALUDE_greatest_of_three_integers_l3282_328255


namespace NUMINAMATH_CALUDE_two_parts_of_ten_l3282_328238

theorem two_parts_of_ten (x y : ℝ) : 
  x + y = 10 ∧ |x - y| = 5 → 
  (x = 7.5 ∧ y = 2.5) ∨ (x = 2.5 ∧ y = 7.5) := by
sorry

end NUMINAMATH_CALUDE_two_parts_of_ten_l3282_328238


namespace NUMINAMATH_CALUDE_problem_solution_l3282_328245

theorem problem_solution (x y : ℝ) (h1 : 3 * x + 2 = 11) (h2 : y = x - 1) : 6 * y - 3 * x = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3282_328245


namespace NUMINAMATH_CALUDE_two_scoop_sundaes_l3282_328260

theorem two_scoop_sundaes (n : ℕ) (h : n = 8) : Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_scoop_sundaes_l3282_328260
