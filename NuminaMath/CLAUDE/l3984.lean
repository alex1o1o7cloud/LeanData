import Mathlib

namespace NUMINAMATH_CALUDE_f_derivative_and_extrema_l3984_398458

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

theorem f_derivative_and_extrema (a : ℝ) :
  (∀ x, deriv (f a) x = 3 * x^2 - 2 * a * x - 4) ∧
  (deriv (f a) (-1) = 0 → a = 1/2) ∧
  (a = 1/2 → ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2) 2, f a x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2) 2, f a x = max) ∧
    (∀ x ∈ Set.Icc (-2) 2, f a x ≥ min) ∧
    (∃ x ∈ Set.Icc (-2) 2, f a x = min) ∧
    max = 9/2 ∧ min = -50/27) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_and_extrema_l3984_398458


namespace NUMINAMATH_CALUDE_inequality_proof_l3984_398440

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ 3 * Real.sqrt 2) ∧
  (2 * (a^3 + b^3 + c^3) ≥ a*b + b*c + c*a - 3*a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3984_398440


namespace NUMINAMATH_CALUDE_smallest_floor_sum_l3984_398434

theorem smallest_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y + z) / x⌋ + ⌊(x + y + z) / y⌋ + ⌊(x + y + z) / z⌋ ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_floor_sum_l3984_398434


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l3984_398449

theorem roof_dimension_difference (area : ℝ) (length_width_ratio : ℝ) :
  area = 676 ∧ length_width_ratio = 4 →
  ∃ (length width : ℝ),
    length = length_width_ratio * width ∧
    area = length * width ∧
    length - width = 39 :=
by sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l3984_398449


namespace NUMINAMATH_CALUDE_wrong_value_correction_l3984_398425

theorem wrong_value_correction (n : ℕ) (initial_mean correct_mean wrong_value : ℚ) 
  (h1 : n = 25)
  (h2 : initial_mean = 190)
  (h3 : wrong_value = 130)
  (h4 : correct_mean = 191.4) :
  let initial_sum := n * initial_mean
  let sum_without_wrong := initial_sum - wrong_value
  let correct_sum := n * correct_mean
  correct_sum - sum_without_wrong + wrong_value = 295 := by
sorry

end NUMINAMATH_CALUDE_wrong_value_correction_l3984_398425


namespace NUMINAMATH_CALUDE_smallest_class_number_l3984_398445

theorem smallest_class_number (total_classes : Nat) (selected_classes : Nat) (sum_selected : Nat) : 
  total_classes = 24 →
  selected_classes = 4 →
  sum_selected = 48 →
  ∃ x : Nat, 
    x > 0 ∧ 
    x ≤ total_classes ∧
    x + (x + (total_classes / selected_classes)) + 
    (x + 2 * (total_classes / selected_classes)) + 
    (x + 3 * (total_classes / selected_classes)) = sum_selected ∧
    x = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_class_number_l3984_398445


namespace NUMINAMATH_CALUDE_unique_zero_implies_a_range_l3984_398428

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 6 * x^2 + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 12 * x

theorem unique_zero_implies_a_range (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) → a < -4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_implies_a_range_l3984_398428


namespace NUMINAMATH_CALUDE_flowers_remaining_after_picking_l3984_398424

/-- The number of flowers remaining after Neznaika's picking --/
def remaining_flowers (total_flowers total_tulips watered_tulips picked_tulips unwatered_flowers : ℕ) : ℕ :=
  total_flowers - unwatered_flowers - picked_tulips

/-- Theorem stating the number of remaining flowers --/
theorem flowers_remaining_after_picking 
  (total_flowers : ℕ) 
  (total_tulips : ℕ)
  (total_peonies : ℕ)
  (watered_tulips : ℕ)
  (picked_tulips : ℕ)
  (unwatered_flowers : ℕ)
  (h1 : total_flowers = 30)
  (h2 : total_tulips = 15)
  (h3 : total_peonies = 15)
  (h4 : total_flowers = total_tulips + total_peonies)
  (h5 : watered_tulips = 10)
  (h6 : unwatered_flowers = 10)
  (h7 : picked_tulips = 6)
  : remaining_flowers total_flowers total_tulips watered_tulips picked_tulips unwatered_flowers = 19 :=
by
  sorry


end NUMINAMATH_CALUDE_flowers_remaining_after_picking_l3984_398424


namespace NUMINAMATH_CALUDE_volume_of_CO2_released_l3984_398466

/-- The volume of CO₂ gas released in a chemical reaction --/
theorem volume_of_CO2_released (n : ℝ) (Vₘ : ℝ) (h1 : n = 2.4) (h2 : Vₘ = 22.4) :
  n * Vₘ = 53.76 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_CO2_released_l3984_398466


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l3984_398421

-- Define the vertices of the tetrahedron
def A1 : ℝ × ℝ × ℝ := (1, 2, 0)
def A2 : ℝ × ℝ × ℝ := (3, 0, -3)
def A3 : ℝ × ℝ × ℝ := (5, 2, 6)
def A4 : ℝ × ℝ × ℝ := (8, 4, -9)

-- Function to calculate the volume of a tetrahedron
def tetrahedron_volume (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Function to calculate the height of a tetrahedron
def tetrahedron_height (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem stating the volume and height of the specific tetrahedron
theorem tetrahedron_properties :
  tetrahedron_volume A1 A2 A3 A4 = 34 ∧
  tetrahedron_height A1 A2 A3 A4 = 7 + 2/7 :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l3984_398421


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3984_398426

theorem marble_fraction_after_tripling (total : ℝ) (h_total_pos : total > 0) :
  let initial_blue := (2/3) * total
  let initial_red := total - initial_blue
  let new_red := 3 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 3/5 := by
sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3984_398426


namespace NUMINAMATH_CALUDE_percent_relation_l3984_398436

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.3 * a) 
  (h2 : c = 0.25 * b) : 
  b = 1.2 * a := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l3984_398436


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3984_398438

theorem quadratic_transformation (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*(m+1)*x + 16 = (x-4)^2) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3984_398438


namespace NUMINAMATH_CALUDE_smallest_a_l3984_398493

/-- The polynomial x³ - ax² + bx - 2010 with three positive integer zeros -/
def polynomial (a b x : ℤ) : ℤ := x^3 - a*x^2 + b*x - 2010

/-- The polynomial has three positive integer zeros -/
def has_three_positive_integer_zeros (a b : ℤ) : Prop :=
  ∃ (x y z : ℤ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    polynomial a b x = 0 ∧ polynomial a b y = 0 ∧ polynomial a b z = 0

/-- The smallest possible value of a is 78 -/
theorem smallest_a (a b : ℤ) :
  has_three_positive_integer_zeros a b → a ≥ 78 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_l3984_398493


namespace NUMINAMATH_CALUDE_non_perfect_power_probability_l3984_398476

/-- A function that determines if a natural number is a perfect power --/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ x y : ℕ, y > 1 ∧ n = x^y

/-- The count of numbers from 1 to 200 that are not perfect powers --/
def nonPerfectPowerCount : ℕ := 178

/-- The total count of numbers from 1 to 200 --/
def totalCount : ℕ := 200

/-- The probability of selecting a non-perfect power from 1 to 200 --/
def probabilityNonPerfectPower : ℚ := 89 / 100

theorem non_perfect_power_probability :
  (nonPerfectPowerCount : ℚ) / (totalCount : ℚ) = probabilityNonPerfectPower :=
sorry

end NUMINAMATH_CALUDE_non_perfect_power_probability_l3984_398476


namespace NUMINAMATH_CALUDE_axis_of_symmetry_cos_minus_sin_l3984_398407

/-- The axis of symmetry for the function y = cos(2x) - sin(2x) is x = -π/8 -/
theorem axis_of_symmetry_cos_minus_sin (x : ℝ) : 
  (∀ y, y = Real.cos (2 * x) - Real.sin (2 * x)) → 
  (∃ k : ℤ, x = (k : ℝ) * π / 2 - π / 8) :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_cos_minus_sin_l3984_398407


namespace NUMINAMATH_CALUDE_set_equality_proof_all_sets_satisfying_condition_l3984_398416

def solution_set : Set (Set Nat) :=
  {{3}, {1, 3}, {2, 3}, {1, 2, 3}}

theorem set_equality_proof (B : Set Nat) :
  ({1, 2} ∪ B = {1, 2, 3}) ↔ (B ∈ solution_set) := by
  sorry

theorem all_sets_satisfying_condition :
  {B : Set Nat | {1, 2} ∪ B = {1, 2, 3}} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_set_equality_proof_all_sets_satisfying_condition_l3984_398416


namespace NUMINAMATH_CALUDE_average_weight_problem_l3984_398417

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions and the average of a and b is 40. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 41 →
  b = 27 →
  (a + b) / 2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l3984_398417


namespace NUMINAMATH_CALUDE_sample_first_year_300_l3984_398471

/-- Represents the ratio of students in each grade -/
structure GradeRatio :=
  (first second third fourth : ℕ)

/-- Calculates the number of first-year students to be sampled given the total sample size and grade ratio -/
def sampleFirstYear (totalSample : ℕ) (ratio : GradeRatio) : ℕ :=
  let totalRatio := ratio.first + ratio.second + ratio.third + ratio.fourth
  (totalSample * ratio.first) / totalRatio

/-- Theorem stating that for a sample size of 300 and ratio 4:5:5:6, the number of first-year students sampled is 60 -/
theorem sample_first_year_300 :
  sampleFirstYear 300 ⟨4, 5, 5, 6⟩ = 60 := by
  sorry

#eval sampleFirstYear 300 ⟨4, 5, 5, 6⟩

end NUMINAMATH_CALUDE_sample_first_year_300_l3984_398471


namespace NUMINAMATH_CALUDE_grid_sum_theorem_l3984_398413

/-- Represents a 3x3 grid of numbers -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if all numbers in the grid are unique and between 1 and 9 -/
def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 9) ∧
  (∀ i j k l, g i j = g k l → (i = k ∧ j = l))

/-- Sum of the right column -/
def right_column_sum (g : Grid) : Nat :=
  g 0 2 + g 1 2 + g 2 2

/-- Sum of the bottom row -/
def bottom_row_sum (g : Grid) : Nat :=
  g 2 0 + g 2 1 + g 2 2

theorem grid_sum_theorem (g : Grid) 
  (h_valid : valid_grid g) 
  (h_right_sum : right_column_sum g = 32) 
  (h_corner : g 2 2 = 7) : 
  bottom_row_sum g = 18 :=
sorry

end NUMINAMATH_CALUDE_grid_sum_theorem_l3984_398413


namespace NUMINAMATH_CALUDE_class_test_problem_l3984_398497

theorem class_test_problem (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ)
  (h1 : p_first = 0.7)
  (h2 : p_second = 0.55)
  (h3 : p_neither = 0.2) :
  p_first + p_second - (1 - p_neither) = 0.45 := by
sorry

end NUMINAMATH_CALUDE_class_test_problem_l3984_398497


namespace NUMINAMATH_CALUDE_bricks_to_fill_road_l3984_398446

/-- Calculates the number of bricks needed to fill a rectangular road without overlapping -/
theorem bricks_to_fill_road (road_width road_length brick_width brick_height : ℝ) :
  road_width = 6 →
  road_length = 4 →
  brick_width = 0.6 →
  brick_height = 0.2 →
  (road_width * road_length) / (brick_width * brick_height) = 200 := by
  sorry

end NUMINAMATH_CALUDE_bricks_to_fill_road_l3984_398446


namespace NUMINAMATH_CALUDE_carly_running_ratio_l3984_398411

def week1_distance : ℝ := 2
def week2_distance : ℝ := 2 * week1_distance + 3
def week4_distance : ℝ := 4
def week3_distance : ℝ := week4_distance + 5

theorem carly_running_ratio :
  week3_distance / week2_distance = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_carly_running_ratio_l3984_398411


namespace NUMINAMATH_CALUDE_quadratic_translation_l3984_398402

/-- Given a quadratic function f(x) = 2x^2, translating its graph upwards by 2 units
    results in the function g(x) = 2x^2 + 2. -/
theorem quadratic_translation (x : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * x^2
  let g : ℝ → ℝ := λ x => 2 * x^2 + 2
  g x = f x + 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_translation_l3984_398402


namespace NUMINAMATH_CALUDE_current_rate_calculation_l3984_398468

/-- Given a boat with speed in still water and its downstream travel details, 
    calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 42)
  (h2 : downstream_distance = 33)
  (h3 : downstream_time = 44 / 60) : 
  (downstream_distance / downstream_time) - boat_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l3984_398468


namespace NUMINAMATH_CALUDE_sets_and_conditions_l3984_398410

def A : Set ℝ := {x | -2 < x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | 2 - a < x ∧ x < 1 + 2*a}

theorem sets_and_conditions :
  (∀ x, x ∈ (A ∪ B 3) ↔ -2 < x ∧ x < 7) ∧
  (∀ x, x ∈ (A ∩ B 3) ↔ -1 < x ∧ x < 5) ∧
  (∀ a, (∀ x, x ∈ B a → x ∈ A) ↔ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_sets_and_conditions_l3984_398410


namespace NUMINAMATH_CALUDE_theater_ticket_pricing_l3984_398452

theorem theater_ticket_pricing (adult_price : ℝ) 
  (h1 : 4 * adult_price + 3 * (adult_price / 2) + 2 * (0.75 * adult_price) = 35) :
  10 * adult_price + 8 * (adult_price / 2) + 5 * (0.75 * adult_price) = 88.75 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_pricing_l3984_398452


namespace NUMINAMATH_CALUDE_abs_neg_four_minus_six_l3984_398470

theorem abs_neg_four_minus_six : |-4 - 6| = 10 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_four_minus_six_l3984_398470


namespace NUMINAMATH_CALUDE_decimal_between_996_998_l3984_398499

theorem decimal_between_996_998 :
  ∃ x y : ℝ, x ≠ y ∧ 0.996 < x ∧ x < 0.998 ∧ 0.996 < y ∧ y < 0.998 :=
sorry

end NUMINAMATH_CALUDE_decimal_between_996_998_l3984_398499


namespace NUMINAMATH_CALUDE_charlie_original_price_l3984_398451

-- Define the given quantities
def alice_acorns : ℕ := 3600
def bob_acorns : ℕ := 2400
def charlie_acorns : ℕ := 4500
def bob_total_price : ℚ := 6000
def discount_rate : ℚ := 0.1

-- Define the relationships
def bob_price_per_acorn : ℚ := bob_total_price / bob_acorns
def alice_price_per_acorn : ℚ := 9 * bob_price_per_acorn
def average_price_per_acorn : ℚ := (alice_price_per_acorn * alice_acorns + bob_price_per_acorn * bob_acorns) / (alice_acorns + bob_acorns)
def charlie_discounted_price_per_acorn : ℚ := average_price_per_acorn * (1 - discount_rate)

-- State the theorem
theorem charlie_original_price : 
  charlie_acorns * average_price_per_acorn = 65250 := by sorry

end NUMINAMATH_CALUDE_charlie_original_price_l3984_398451


namespace NUMINAMATH_CALUDE_tangent_length_equals_hypotenuse_leg_l3984_398494

-- Define the triangle DEF
structure RightTriangle where
  DE : ℝ
  DF : ℝ
  rightAngleAtE : True

-- Define the circle
structure TangentCircle where
  centerOnDE : True
  tangentToDF : True
  tangentToEF : True

-- Define the theorem
theorem tangent_length_equals_hypotenuse_leg 
  (triangle : RightTriangle) 
  (circle : TangentCircle) 
  (h1 : triangle.DE = 7) 
  (h2 : triangle.DF = Real.sqrt 85) : 
  ∃ Q : ℝ × ℝ, ∃ FQ : ℝ, FQ = 6 :=
sorry

end NUMINAMATH_CALUDE_tangent_length_equals_hypotenuse_leg_l3984_398494


namespace NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l3984_398437

theorem rectangular_solid_edge_sum :
  ∀ (a b c r : ℝ),
    a * b * c = 512 →
    2 * (a * b + b * c + a * c) = 352 →
    b = a * r →
    c = a * r^2 →
    a = 4 →
    4 * (a + b + c) = 112 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l3984_398437


namespace NUMINAMATH_CALUDE_number_of_ways_to_choose_officials_l3984_398474

-- Define the number of people in the group
def group_size : ℕ := 8

-- Define the number of positions to be filled
def num_positions : ℕ := 3

-- Theorem stating the number of ways to choose the officials
theorem number_of_ways_to_choose_officials :
  (group_size * (group_size - 1) * (group_size - 2)) = 336 := by
  sorry

end NUMINAMATH_CALUDE_number_of_ways_to_choose_officials_l3984_398474


namespace NUMINAMATH_CALUDE_toms_floor_replacement_cost_l3984_398492

/-- The total cost to replace a floor given room dimensions, removal cost, and new floor cost per square foot. -/
def total_floor_replacement_cost (length width removal_cost cost_per_sqft : ℝ) : ℝ :=
  removal_cost + length * width * cost_per_sqft

/-- Theorem stating that the total cost to replace the floor in Tom's room is $120. -/
theorem toms_floor_replacement_cost :
  total_floor_replacement_cost 8 7 50 1.25 = 120 := by
  sorry

#eval total_floor_replacement_cost 8 7 50 1.25

end NUMINAMATH_CALUDE_toms_floor_replacement_cost_l3984_398492


namespace NUMINAMATH_CALUDE_norma_cards_l3984_398478

/-- The number of cards Norma has after losing some -/
def cards_remaining (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem: Norma has 18 cards remaining -/
theorem norma_cards : cards_remaining 88 70 = 18 := by
  sorry

end NUMINAMATH_CALUDE_norma_cards_l3984_398478


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l3984_398412

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  ∃ w : ℕ, w > 0 ∧ w ∣ n ∧ ∀ k : ℕ, k > 0 ∧ k ∣ n → k ≤ w ∧ w = 12 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l3984_398412


namespace NUMINAMATH_CALUDE_original_number_l3984_398469

theorem original_number (x : ℚ) : (1 / x) - 2 = 5 / 4 → x = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l3984_398469


namespace NUMINAMATH_CALUDE_convex_polyhedron_properties_l3984_398408

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  convex : Bool

/-- The number of sides of a face in a polyhedron. -/
def face_sides (p : ConvexPolyhedron) (f : Nat) : Nat :=
  sorry

/-- The number of edges meeting at a vertex in a polyhedron. -/
def vertex_edges (p : ConvexPolyhedron) (v : Nat) : Nat :=
  sorry

/-- The set of all faces in a polyhedron. -/
def faces (p : ConvexPolyhedron) : Set Nat :=
  sorry

/-- The set of all vertices in a polyhedron. -/
def vertices (p : ConvexPolyhedron) : Set Nat :=
  sorry

theorem convex_polyhedron_properties (p : ConvexPolyhedron) (h : p.convex) :
  (∃ f ∈ faces p, face_sides p f ≤ 5) ∧
  (∃ v ∈ vertices p, vertex_edges p v ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_convex_polyhedron_properties_l3984_398408


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3984_398439

/-- 
Given two points A and B that are symmetric with respect to the origin,
prove that the sum of their x and y coordinates is -2.
-/
theorem symmetric_points_sum (m n : ℝ) : 
  (3 : ℝ) = -(-m) → n = -(5 : ℝ) → m + n = -2 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3984_398439


namespace NUMINAMATH_CALUDE_no_special_polynomials_l3984_398427

/-- Represents a polynomial of the form x^4 + ax^3 + bx^2 + cx + 2048 -/
def SpecialPolynomial (a b c : ℝ) (x : ℂ) : ℂ :=
  x^4 + a*x^3 + b*x^2 + c*x + 2048

/-- Predicate to check if a complex number is a root of the polynomial -/
def IsRoot (a b c : ℝ) (s : ℂ) : Prop :=
  SpecialPolynomial a b c s = 0

/-- Predicate to check if the polynomial satisfies the special root property -/
def HasSpecialRootProperty (a b c : ℝ) : Prop :=
  ∀ s : ℂ, IsRoot a b c s → IsRoot a b c (s^2) ∧ IsRoot a b c (s⁻¹)

theorem no_special_polynomials :
  ¬∃ a b c : ℝ, HasSpecialRootProperty a b c :=
sorry

end NUMINAMATH_CALUDE_no_special_polynomials_l3984_398427


namespace NUMINAMATH_CALUDE_bens_gross_income_l3984_398443

theorem bens_gross_income (car_payment insurance maintenance fuel : ℝ)
  (h1 : car_payment = 400)
  (h2 : insurance = 150)
  (h3 : maintenance = 75)
  (h4 : fuel = 50)
  (h5 : ∀ after_tax_income : ℝ, 
    0.2 * after_tax_income = car_payment + insurance + maintenance + fuel)
  (h6 : ∀ gross_income : ℝ, 
    (2/3) * gross_income = after_tax_income) :
  ∃ gross_income : ℝ, gross_income = 5062.50 := by
sorry

end NUMINAMATH_CALUDE_bens_gross_income_l3984_398443


namespace NUMINAMATH_CALUDE_equation_solution_l3984_398415

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (1 / x + (3 / x) / (6 / x) - 5 / x = 0.5) ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3984_398415


namespace NUMINAMATH_CALUDE_unique_complex_solution_l3984_398418

theorem unique_complex_solution :
  ∃! z : ℂ, Complex.abs z < 20 ∧ Complex.exp z = 1 - z / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_complex_solution_l3984_398418


namespace NUMINAMATH_CALUDE_x_range_l3984_398455

theorem x_range (x : ℝ) (h1 : 1/x < 4) (h2 : 1/x > -6) (h3 : x < 0) :
  -1/6 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l3984_398455


namespace NUMINAMATH_CALUDE_log_sin_cos_theorem_l3984_398477

theorem log_sin_cos_theorem (x n : ℝ) 
  (h1 : Real.log (Real.sin x) + Real.log (Real.cos x) = -2)
  (h2 : Real.log (Real.sin x + Real.cos x) = (Real.log n - 2) / 2) : 
  n = Real.exp 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sin_cos_theorem_l3984_398477


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3984_398491

theorem smallest_number_with_given_remainders : ∃! n : ℕ, 
  (n % 6 = 2) ∧ (n % 5 = 3) ∧ (n % 7 = 1) ∧
  (∀ m : ℕ, m < n → ¬((m % 6 = 2) ∧ (m % 5 = 3) ∧ (m % 7 = 1))) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3984_398491


namespace NUMINAMATH_CALUDE_wilsons_theorem_l3984_398481

theorem wilsons_theorem (p : ℕ) (h : p ≥ 2) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) : ℤ) ≡ -1 [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l3984_398481


namespace NUMINAMATH_CALUDE_jeans_jail_sentence_l3984_398488

/-- Calculates the total jail sentence for Jean based on various charges --/
def total_jail_sentence (arson_counts : ℕ) (burglary_charges : ℕ) (arson_sentence : ℕ) (burglary_sentence : ℕ) : ℕ :=
  let petty_larceny_charges := 6 * burglary_charges
  let petty_larceny_sentence := burglary_sentence / 3
  arson_counts * arson_sentence +
  burglary_charges * burglary_sentence +
  petty_larceny_charges * petty_larceny_sentence

/-- Theorem stating that Jean's total jail sentence is 216 months --/
theorem jeans_jail_sentence :
  total_jail_sentence 3 2 36 18 = 216 := by
  sorry


end NUMINAMATH_CALUDE_jeans_jail_sentence_l3984_398488


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3984_398444

theorem complex_magnitude_product : Complex.abs (3 - 5 * Complex.I) * Complex.abs (3 + 5 * Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3984_398444


namespace NUMINAMATH_CALUDE_height_comparison_equivalences_l3984_398401

-- Define the classes A and B
variable (A B : Type)

-- Define a height function for students
variable (height : A ⊕ B → ℝ)

-- Define the propositions for each question
def tallest_A_taller_than_tallest_B : Prop :=
  ∀ b : B, ∃ a : A, height (Sum.inl a) > height (Sum.inr b)

def every_B_shorter_than_some_A : Prop :=
  ∀ b : B, ∃ a : A, height (Sum.inl a) > height (Sum.inr b)

def for_any_A_exists_shorter_B : Prop :=
  ∀ a : A, ∃ b : B, height (Sum.inl a) > height (Sum.inr b)

def shortest_B_shorter_than_shortest_A : Prop :=
  ∃ a : A, ∀ b : B, height (Sum.inl a) > height (Sum.inr b)

-- State the theorem
theorem height_comparison_equivalences
  (A B : Type) (height : A ⊕ B → ℝ) :
  (tallest_A_taller_than_tallest_B A B height ↔ every_B_shorter_than_some_A A B height) ∧
  (for_any_A_exists_shorter_B A B height ↔ shortest_B_shorter_than_shortest_A A B height) :=
sorry

end NUMINAMATH_CALUDE_height_comparison_equivalences_l3984_398401


namespace NUMINAMATH_CALUDE_three_color_theorem_min_three_colors_min_colors_is_three_l3984_398448

/-- Represents a 3D coordinate in the 3x3x3 grid --/
structure Coord where
  x : Fin 3
  y : Fin 3
  z : Fin 3

/-- Represents a coloring of the 3x3x3 grid --/
def Coloring := Coord → Fin 3

/-- Two coordinates are adjacent if they differ by 1 in exactly one dimension --/
def adjacent (c1 c2 : Coord) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ c1.z.val + 1 = c2.z.val) ∨
  (c1.x = c2.x ∧ c1.y = c2.y ∧ c1.z.val = c2.z.val + 1) ∨
  (c1.x = c2.x ∧ c1.y.val + 1 = c2.y.val ∧ c1.z = c2.z) ∨
  (c1.x = c2.x ∧ c1.y.val = c2.y.val + 1 ∧ c1.z = c2.z) ∨
  (c1.x.val + 1 = c2.x.val ∧ c1.y = c2.y ∧ c1.z = c2.z) ∨
  (c1.x.val = c2.x.val + 1 ∧ c1.y = c2.y ∧ c1.z = c2.z)

/-- A coloring is valid if no adjacent cubes have the same color --/
def validColoring (c : Coloring) : Prop :=
  ∀ c1 c2 : Coord, adjacent c1 c2 → c c1 ≠ c c2

/-- There exists a valid coloring using only 3 colors --/
theorem three_color_theorem : ∃ c : Coloring, validColoring c :=
  sorry

/-- Any valid coloring must use at least 3 colors --/
theorem min_three_colors (c : Coloring) (h : validColoring c) :
  ∃ c1 c2 c3 : Coord, c c1 ≠ c c2 ∧ c c2 ≠ c c3 ∧ c c1 ≠ c c3 :=
  sorry

/-- The minimum number of colors needed is exactly 3 --/
theorem min_colors_is_three :
  (∃ c : Coloring, validColoring c) ∧
  (∀ c : Coloring, validColoring c →
    ∃ c1 c2 c3 : Coord, c c1 ≠ c c2 ∧ c c2 ≠ c c3 ∧ c c1 ≠ c c3) :=
  sorry

end NUMINAMATH_CALUDE_three_color_theorem_min_three_colors_min_colors_is_three_l3984_398448


namespace NUMINAMATH_CALUDE_angle_d_measure_l3984_398454

/-- A scalene triangle with specific angle relationships -/
structure ScaleneTriangle where
  /-- Measure of angle D in degrees -/
  angleD : ℝ
  /-- Measure of angle E in degrees -/
  angleE : ℝ
  /-- Measure of angle F in degrees -/
  angleF : ℝ
  /-- Triangle is scalene -/
  scalene : angleD ≠ angleE ∧ angleE ≠ angleF ∧ angleD ≠ angleF
  /-- Angle E is twice angle D -/
  e_twice_d : angleE = 2 * angleD
  /-- Angle F is 40 degrees -/
  f_is_40 : angleF = 40
  /-- Sum of angles in a triangle is 180 degrees -/
  angle_sum : angleD + angleE + angleF = 180

/-- Theorem: In a scalene triangle DEF with the given conditions, angle D measures 140/3 degrees -/
theorem angle_d_measure (t : ScaleneTriangle) : t.angleD = 140 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_d_measure_l3984_398454


namespace NUMINAMATH_CALUDE_intersection_count_l3984_398431

/-- Represents a lattice point in the coordinate plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a circle centered at a lattice point -/
structure Circle where
  center : LatticePoint
  radius : ℚ

/-- Represents a square centered at a lattice point -/
structure Square where
  center : LatticePoint
  sideLength : ℚ

/-- Represents a line segment from (0,0) to (703, 299) -/
def lineSegment : Set (ℚ × ℚ) :=
  {p | ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧ p = (703 * t, 299 * t)}

/-- Counts the number of intersections with squares and circles -/
def countIntersections (line : Set (ℚ × ℚ)) (squares : Set Square) (circles : Set Circle) : ℕ :=
  sorry

/-- Main theorem statement -/
theorem intersection_count :
  ∀ (squares : Set Square) (circles : Set Circle),
    (∀ p : LatticePoint, ∃ s ∈ squares, s.center = p ∧ s.sideLength = 2/5) →
    (∀ p : LatticePoint, ∃ c ∈ circles, c.center = p ∧ c.radius = 1/5) →
    countIntersections lineSegment squares circles = 2109 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l3984_398431


namespace NUMINAMATH_CALUDE_garden_area_increase_l3984_398460

theorem garden_area_increase : 
  let rectangle_length : ℝ := 60
  let rectangle_width : ℝ := 20
  let rectangle_perimeter : ℝ := 2 * (rectangle_length + rectangle_width)
  let square_side : ℝ := rectangle_perimeter / 4
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let square_area : ℝ := square_side * square_side
  square_area - rectangle_area = 400 := by
sorry

end NUMINAMATH_CALUDE_garden_area_increase_l3984_398460


namespace NUMINAMATH_CALUDE_greg_original_seat_l3984_398487

/-- Represents a seat in the theater --/
inductive Seat
| one
| two
| three
| four
| five

/-- Represents a friend --/
inductive Friend
| Greg
| Iris
| Jamal
| Kim
| Leo

/-- Represents the seating arrangement --/
def Arrangement := Friend → Seat

/-- Represents a movement of a friend --/
def Movement := Friend → Int

theorem greg_original_seat 
  (initial_arrangement : Arrangement)
  (final_arrangement : Arrangement)
  (movements : Movement) :
  (movements Friend.Iris = 1) →
  (movements Friend.Jamal = -2) →
  (movements Friend.Kim + movements Friend.Leo = 0) →
  (final_arrangement Friend.Greg = Seat.one) →
  (initial_arrangement Friend.Greg = Seat.two) :=
sorry

end NUMINAMATH_CALUDE_greg_original_seat_l3984_398487


namespace NUMINAMATH_CALUDE_remainder_problem_l3984_398457

theorem remainder_problem (N : ℤ) (h : N % 899 = 63) : N % 29 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3984_398457


namespace NUMINAMATH_CALUDE_oil_drop_probability_l3984_398453

theorem oil_drop_probability (c : ℝ) (h : c > 0) : 
  (0.5 * c)^2 / (π * (c/2)^2) = 0.25 / π := by
  sorry

end NUMINAMATH_CALUDE_oil_drop_probability_l3984_398453


namespace NUMINAMATH_CALUDE_sin_equation_condition_l3984_398495

theorem sin_equation_condition (α β : Real) :
  (7 * 15 * Real.sin α + Real.sin β = Real.sin (α + β)) ↔
  (∃ k : ℤ, α = 2 * k * Real.pi ∨ β = 2 * k * Real.pi ∨ α + β = 2 * k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_sin_equation_condition_l3984_398495


namespace NUMINAMATH_CALUDE_certain_number_exists_and_unique_l3984_398404

theorem certain_number_exists_and_unique : 
  ∃! x : ℕ, 220050 = (x + 445) * (2 * (x - 445)) + 50 :=
sorry

end NUMINAMATH_CALUDE_certain_number_exists_and_unique_l3984_398404


namespace NUMINAMATH_CALUDE_pythagorean_numbers_l3984_398484

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_numbers : 
  (is_pythagorean_triple 9 12 15) ∧ 
  (¬ is_pythagorean_triple 3 4 5) ∧ 
  (¬ is_pythagorean_triple 1 1 2) :=
by
  sorry

end NUMINAMATH_CALUDE_pythagorean_numbers_l3984_398484


namespace NUMINAMATH_CALUDE_cards_distribution_l3984_398485

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 48) (h2 : num_people = 7) :
  let cards_per_person := total_cards / num_people
  let remaining_cards := total_cards % num_people
  num_people - remaining_cards = 1 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l3984_398485


namespace NUMINAMATH_CALUDE_max_sum_diff_unit_vectors_l3984_398420

theorem max_sum_diff_unit_vectors (a b : EuclideanSpace ℝ (Fin 2)) :
  ‖a‖ = 1 → ‖b‖ = 1 → ‖a + b‖ + ‖a - b‖ ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_diff_unit_vectors_l3984_398420


namespace NUMINAMATH_CALUDE_dot_product_range_l3984_398472

/-- The range of the dot product OP · BA -/
theorem dot_product_range (O A B P : ℝ × ℝ) : 
  O = (0, 0) →
  A = (2, 0) →
  B = (1, -2 * Real.sqrt 3) →
  (∃ (x : ℝ), P.1 = x ∧ P.2 = Real.sqrt (1 - x^2 / 4)) →
  -2 ≤ (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) ∧
  (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l3984_398472


namespace NUMINAMATH_CALUDE_chen_trigonometric_problem_l3984_398473

theorem chen_trigonometric_problem :
  ∃ (N : ℕ) (α β γ θ : ℝ),
    0.1 = Real.sin γ * Real.cos θ * Real.sin α ∧
    0.2 = Real.sin γ * Real.sin θ * Real.cos α ∧
    0.3 = Real.cos γ * Real.cos θ * Real.sin β ∧
    0.4 = Real.cos γ * Real.sin θ * Real.cos β ∧
    0.5 ≥ |N - 100 * Real.cos (2 * θ)| ∧
    N = 79 := by
  sorry

end NUMINAMATH_CALUDE_chen_trigonometric_problem_l3984_398473


namespace NUMINAMATH_CALUDE_first_fabulous_friday_is_oct31_l3984_398405

/-- Represents a date with a day, month, and day of the week -/
structure Date where
  day : Nat
  month : Nat
  dayOfWeek : Nat
  deriving Repr

/-- Represents a school calendar -/
structure SchoolCalendar where
  startDate : Date
  deriving Repr

/-- Determines if a given date is a Fabulous Friday -/
def isFabulousFriday (d : Date) : Bool :=
  sorry

/-- Finds the first Fabulous Friday after the school start date -/
def firstFabulousFriday (sc : SchoolCalendar) : Date :=
  sorry

/-- Theorem stating that the first Fabulous Friday after school starts on Tuesday, October 3 is October 31 -/
theorem first_fabulous_friday_is_oct31 (sc : SchoolCalendar) :
  sc.startDate = Date.mk 3 10 2 →  -- October 3 is a Tuesday (day 2 of the week)
  firstFabulousFriday sc = Date.mk 31 10 5 :=  -- October 31 is a Friday (day 5 of the week)
  sorry

end NUMINAMATH_CALUDE_first_fabulous_friday_is_oct31_l3984_398405


namespace NUMINAMATH_CALUDE_decimal_place_150_of_5_11_l3984_398465

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The period of the decimal representation of a rational number -/
def decimal_period (q : ℚ) : ℕ := sorry

theorem decimal_place_150_of_5_11 :
  decimal_representation (5/11) 150 = 5 := by sorry

end NUMINAMATH_CALUDE_decimal_place_150_of_5_11_l3984_398465


namespace NUMINAMATH_CALUDE_singing_competition_average_age_l3984_398482

theorem singing_competition_average_age 
  (num_females : Nat) 
  (num_males : Nat)
  (avg_age_females : ℝ) 
  (avg_age_males : ℝ) :
  num_females = 12 →
  num_males = 18 →
  avg_age_females = 25 →
  avg_age_males = 40 →
  (num_females * avg_age_females + num_males * avg_age_males) / (num_females + num_males) = 34 := by
sorry

end NUMINAMATH_CALUDE_singing_competition_average_age_l3984_398482


namespace NUMINAMATH_CALUDE_remainder_of_expression_l3984_398422

theorem remainder_of_expression (n : ℕ) : (1 - 90)^10 % 88 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_expression_l3984_398422


namespace NUMINAMATH_CALUDE_find_decrease_rate_village_x_decrease_rate_l3984_398461

/-- Represents the population change in two villages over time -/
def village_population_equality (x_initial : ℕ) (y_initial : ℕ) (y_growth_rate : ℕ) (years : ℕ) (x_decrease_rate : ℕ) : Prop :=
  x_initial - years * x_decrease_rate = y_initial + years * y_growth_rate

/-- Theorem stating the condition for equal populations after a given time -/
theorem find_decrease_rate (x_initial y_initial y_growth_rate years : ℕ) :
  ∃ (x_decrease_rate : ℕ),
    village_population_equality x_initial y_initial y_growth_rate years x_decrease_rate ∧
    x_decrease_rate = (x_initial - y_initial - years * y_growth_rate) / years :=
by
  sorry

/-- Application of the theorem to the specific problem -/
theorem village_x_decrease_rate :
  ∃ (x_decrease_rate : ℕ),
    village_population_equality 76000 42000 800 17 x_decrease_rate ∧
    x_decrease_rate = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_find_decrease_rate_village_x_decrease_rate_l3984_398461


namespace NUMINAMATH_CALUDE_radio_show_music_commercial_ratio_l3984_398406

/-- Represents a segment of a radio show -/
structure Segment where
  total_time : ℕ
  commercial_time : ℕ

/-- Calculates the greatest common divisor of two natural numbers -/
def gcd (a b : ℕ) : ℕ := sorry

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplify_ratio (a b : ℕ) : ℕ × ℕ := sorry

theorem radio_show_music_commercial_ratio 
  (segment1 : Segment)
  (segment2 : Segment)
  (segment3 : Segment)
  (h1 : segment1.total_time = 56 ∧ segment1.commercial_time = 22)
  (h2 : segment2.total_time = 84 ∧ segment2.commercial_time = 28)
  (h3 : segment3.total_time = 128 ∧ segment3.commercial_time = 34) :
  simplify_ratio 
    ((segment1.total_time - segment1.commercial_time) + 
     (segment2.total_time - segment2.commercial_time) + 
     (segment3.total_time - segment3.commercial_time))
    (segment1.commercial_time + segment2.commercial_time + segment3.commercial_time) = (46, 21) := by
  sorry

end NUMINAMATH_CALUDE_radio_show_music_commercial_ratio_l3984_398406


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3984_398480

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3984_398480


namespace NUMINAMATH_CALUDE_arithmetic_contains_geometric_l3984_398496

/-- Given positive integers a and d, there exist positive integers b and q such that 
    the geometric progression b, bq, bq^2, ... is a subset of the arithmetic progression a, a+d, a+2d, ... -/
theorem arithmetic_contains_geometric (a d : ℕ+) : 
  ∃ (b q : ℕ+), ∀ (n : ℕ), ∃ (k : ℕ), b * q ^ n = a + k * d := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_contains_geometric_l3984_398496


namespace NUMINAMATH_CALUDE_function_value_solution_l3984_398442

theorem function_value_solution (x : ℝ) :
  (x^2 + x - 1 = 5) ↔ (x = 2 ∨ x = -3) := by sorry

end NUMINAMATH_CALUDE_function_value_solution_l3984_398442


namespace NUMINAMATH_CALUDE_proposition_implications_l3984_398419

def p (a : ℝ) : Prop := 1 ∈ {x : ℝ | x^2 < a}
def q (a : ℝ) : Prop := 2 ∈ {x : ℝ | x^2 < a}

theorem proposition_implications (a : ℝ) :
  ((p a ∨ q a) → a > 1) ∧ ((p a ∧ q a) → a > 4) := by sorry

end NUMINAMATH_CALUDE_proposition_implications_l3984_398419


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l3984_398463

theorem compare_negative_fractions : -4/3 < -5/4 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l3984_398463


namespace NUMINAMATH_CALUDE_pie_sugar_percentage_l3984_398429

/-- Given a pie weighing 200 grams with 50 grams of sugar, 
    prove that 75% of the pie is not sugar. -/
theorem pie_sugar_percentage 
  (total_weight : ℝ) 
  (sugar_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : sugar_weight = 50) : 
  (total_weight - sugar_weight) / total_weight * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_pie_sugar_percentage_l3984_398429


namespace NUMINAMATH_CALUDE_complete_set_is_reals_l3984_398479

def is_complete (A : Set ℝ) : Prop :=
  A.Nonempty ∧ ∀ a b : ℝ, (a + b) ∈ A → (a * b) ∈ A

theorem complete_set_is_reals (A : Set ℝ) : is_complete A → A = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_complete_set_is_reals_l3984_398479


namespace NUMINAMATH_CALUDE_S_n_min_l3984_398464

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = -11
  sum_5_6 : a 5 + a 6 = -4

/-- The sum of the first n terms of the arithmetic sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n^2 - 12*n

/-- The theorem stating that S_n reaches its minimum when n = 6 -/
theorem S_n_min (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≠ 0 → S_n seq 6 ≤ S_n seq n :=
sorry

end NUMINAMATH_CALUDE_S_n_min_l3984_398464


namespace NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l3984_398423

/-- A cube with side length s contains a regular tetrahedron with vertices
    (0,0,0), (s,s,0), (s,0,s), and (0,s,s). The ratio of the surface area of
    the cube to the surface area of the tetrahedron is √3. -/
theorem cube_tetrahedron_surface_area_ratio (s : ℝ) (h : s > 0) :
  let cube_vertices : Fin 8 → ℝ × ℝ × ℝ := fun i =>
    ((i : ℕ) % 2 * s, ((i : ℕ) / 2) % 2 * s, ((i : ℕ) / 4) * s)
  let tetra_vertices : Fin 4 → ℝ × ℝ × ℝ := fun i =>
    match i with
    | 0 => (0, 0, 0)
    | 1 => (s, s, 0)
    | 2 => (s, 0, s)
    | 3 => (0, s, s)
  let cube_surface_area := 6 * s^2
  let tetra_surface_area := 2 * Real.sqrt 3 * s^2
  cube_surface_area / tetra_surface_area = Real.sqrt 3 := by
  sorry

#check cube_tetrahedron_surface_area_ratio

end NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l3984_398423


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3984_398430

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point through which the tangent line passes
def point : ℝ × ℝ := (1, 1)

-- Define the two possible tangent line equations
def tangent1 (x y : ℝ) : Prop := 3*x - y - 2 = 0
def tangent2 (x y : ℝ) : Prop := 3*x - 4*y + 1 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x₀ y₀ : ℝ), 
    (y₀ = f x₀) ∧ 
    ((tangent1 x₀ y₀ ∧ (∀ x : ℝ, tangent1 x (f x) → x = x₀)) ∨
     (tangent2 x₀ y₀ ∧ (∀ x : ℝ, tangent2 x (f x) → x = x₀))) ∧
    (point.1 = 1 ∧ point.2 = 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3984_398430


namespace NUMINAMATH_CALUDE_chessboard_probability_l3984_398459

theorem chessboard_probability (k : ℕ) : k ≥ 5 →
  (((k - 4)^2 - 1) / (2 * (k - 4)^2 : ℚ) = 48 / 100) ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_probability_l3984_398459


namespace NUMINAMATH_CALUDE_sphere_in_dihedral_angle_l3984_398462

/-- Given a sphere of unit radius with its center on the edge of a dihedral angle α,
    the radius r of a new sphere whose volume equals the volume of the part of the given sphere
    that lies inside the dihedral angle is r = ∛(α / (2π)). -/
theorem sphere_in_dihedral_angle (α : Real) (h : 0 < α ∧ α < 2 * Real.pi) :
  ∃ (r : Real), r = (α / (2 * Real.pi)) ^ (1/3) ∧
  (4/3 * Real.pi * r^3) = (α / (2 * Real.pi)) * (4/3 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_dihedral_angle_l3984_398462


namespace NUMINAMATH_CALUDE_medicine_types_count_l3984_398435

/-- The number of medical boxes -/
def num_boxes : ℕ := 5

/-- The number of boxes each medicine appears in -/
def boxes_per_medicine : ℕ := 2

/-- Calculates the binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of types of medicine -/
def num_medicine_types : ℕ := binomial num_boxes boxes_per_medicine

theorem medicine_types_count : num_medicine_types = 10 := by
  sorry

end NUMINAMATH_CALUDE_medicine_types_count_l3984_398435


namespace NUMINAMATH_CALUDE_initial_water_percentage_l3984_398400

theorem initial_water_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) :
  capacity = 80 →
  added_water = 36 →
  final_fraction = 3/4 →
  ∃ initial_percentage : ℝ,
    initial_percentage = 30 ∧
    (initial_percentage / 100) * capacity + added_water = final_fraction * capacity :=
by sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l3984_398400


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l3984_398414

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  945 = 21 * q + r ∧ 
  q > 0 ∧ 
  r > 0 ∧ 
  ∀ (q' r' : ℕ), 945 = 21 * q' + r' ∧ q' > 0 ∧ r' > 0 → q - r ≥ q' - r' :=
by sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l3984_398414


namespace NUMINAMATH_CALUDE_representation_625_ends_with_1_l3984_398432

def base_count : ℕ := 4

theorem representation_625_ends_with_1 :
  (∃ (S : Finset ℕ), (∀ b ∈ S, 3 ≤ b ∧ b ≤ 10) ∧
   (∀ b ∈ S, (625 : ℕ) % b = 1) ∧
   S.card = base_count) :=
by sorry

end NUMINAMATH_CALUDE_representation_625_ends_with_1_l3984_398432


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3984_398456

/-- Proves that the original stock price is 100 given the conditions --/
theorem stock_price_calculation (X : ℝ) : 
  X * 0.95 + 0.001 * (X * 0.95) = 95.2 → X = 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l3984_398456


namespace NUMINAMATH_CALUDE_park_to_restaurant_time_l3984_398489

/-- Represents the time in minutes for various segments of Dante's walk -/
structure WalkTimes where
  parkToHiddenLake : ℕ
  hiddenLakeToPark : ℕ
  parkToRestaurant : ℕ
  totalTime : ℕ

/-- Proves that the time to walk from Park Office to Lake Park restaurant is 10 minutes -/
theorem park_to_restaurant_time (w : WalkTimes) 
  (h1 : w.parkToHiddenLake = 15)
  (h2 : w.hiddenLakeToPark = 7)
  (h3 : w.totalTime = 32)
  (h4 : w.totalTime = w.parkToHiddenLake + w.hiddenLakeToPark + w.parkToRestaurant) :
  w.parkToRestaurant = 10 := by
  sorry

end NUMINAMATH_CALUDE_park_to_restaurant_time_l3984_398489


namespace NUMINAMATH_CALUDE_temple_shop_cost_l3984_398409

/-- The cost per object at the shop --/
def cost_per_object : ℕ := 11

/-- The number of people in Nathan's group --/
def number_of_people : ℕ := 3

/-- The number of shoes per person --/
def shoes_per_person : ℕ := 2

/-- The number of socks per person --/
def socks_per_person : ℕ := 2

/-- The number of mobiles per person --/
def mobiles_per_person : ℕ := 1

/-- The total cost for Nathan and his parents to store their belongings --/
def total_cost : ℕ := number_of_people * (shoes_per_person + socks_per_person + mobiles_per_person) * cost_per_object

theorem temple_shop_cost : total_cost = 165 := by
  sorry

end NUMINAMATH_CALUDE_temple_shop_cost_l3984_398409


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_l3984_398450

theorem binomial_expansion_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₄ = 80 ∧ a₁ + a₂ + a₃ = 130) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_l3984_398450


namespace NUMINAMATH_CALUDE_percent_relation_l3984_398475

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.30 * a) 
  (h2 : c = 0.25 * b) : 
  b = 1.2 * a := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l3984_398475


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_4_minus_5_2_l3984_398486

theorem least_prime_factor_of_5_4_minus_5_2 :
  Nat.minFac (5^4 - 5^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_4_minus_5_2_l3984_398486


namespace NUMINAMATH_CALUDE_percentage_problem_l3984_398403

theorem percentage_problem (x : ℝ) : x * 0.0005 = 6.178 → x = 12356 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3984_398403


namespace NUMINAMATH_CALUDE_voting_theorem_l3984_398498

/-- Represents the number of students voting for each issue and against all issues -/
structure VotingData where
  total : ℕ
  issueA : ℕ
  issueB : ℕ
  issueC : ℕ
  againstAll : ℕ

/-- Calculates the number of students voting for all three issues -/
def studentsVotingForAll (data : VotingData) : ℕ :=
  data.issueA + data.issueB + data.issueC - data.total + data.againstAll

/-- Theorem stating the number of students voting for all three issues -/
theorem voting_theorem (data : VotingData) 
    (h1 : data.total = 300)
    (h2 : data.issueA = 210)
    (h3 : data.issueB = 190)
    (h4 : data.issueC = 160)
    (h5 : data.againstAll = 40) :
  studentsVotingForAll data = 80 := by
  sorry

#eval studentsVotingForAll { total := 300, issueA := 210, issueB := 190, issueC := 160, againstAll := 40 }

end NUMINAMATH_CALUDE_voting_theorem_l3984_398498


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3984_398467

/-- The asymptotes of the hyperbola x²/4 - y² = 1 are y = ±(1/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 / 4 - y^2 = 1) → 
  (∃ (k : ℝ), k = 1/2 ∧ (y = k*x ∨ y = -k*x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3984_398467


namespace NUMINAMATH_CALUDE_piglets_count_l3984_398433

/-- Calculates the number of piglets given the total number of straws and straws per piglet -/
def number_of_piglets (total_straws : ℕ) (straws_per_piglet : ℕ) : ℕ :=
  let straws_for_adult_pigs := (3 * total_straws) / 5
  let straws_for_piglets := straws_for_adult_pigs
  straws_for_piglets / straws_per_piglet

/-- Proves that the number of piglets is 30 given the problem conditions -/
theorem piglets_count : number_of_piglets 300 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_piglets_count_l3984_398433


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3984_398490

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, x^2 + 2*k*x + 1 = 0) ↔ k = 1 ∨ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3984_398490


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l3984_398447

theorem magical_red_knights_fraction :
  ∀ (total_knights : ℕ) (red_knights blue_knights magical_knights : ℕ) 
    (red_magical blue_magical : ℚ),
    red_knights = total_knights / 3 →
    blue_knights = total_knights - red_knights →
    magical_knights = total_knights / 5 →
    blue_magical = (2/3) * red_magical →
    red_knights * red_magical + blue_knights * blue_magical = magical_knights →
    red_magical = 9/35 := by
  sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l3984_398447


namespace NUMINAMATH_CALUDE_debt_payment_calculation_l3984_398483

theorem debt_payment_calculation (total_installments : Nat) 
  (first_payments : Nat) (remaining_payments : Nat) (average_payment : ℚ) :
  total_installments = 52 →
  first_payments = 25 →
  remaining_payments = 27 →
  average_payment = 551.9230769230769 →
  ∃ (x : ℚ), 
    (x * first_payments + (x + 100) * remaining_payments) / total_installments = average_payment ∧
    x = 500 := by
  sorry

end NUMINAMATH_CALUDE_debt_payment_calculation_l3984_398483


namespace NUMINAMATH_CALUDE_mikis_sandcastle_height_l3984_398441

/-- The height of Miki's sister's sandcastle in feet -/
def sisters_height : ℝ := 0.5

/-- The difference in height between Miki's and her sister's sandcastles in feet -/
def height_difference : ℝ := 0.33

/-- The height of Miki's sandcastle in feet -/
def mikis_height : ℝ := sisters_height + height_difference

theorem mikis_sandcastle_height : mikis_height = 0.83 := by
  sorry

end NUMINAMATH_CALUDE_mikis_sandcastle_height_l3984_398441
