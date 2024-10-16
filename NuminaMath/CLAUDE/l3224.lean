import Mathlib

namespace NUMINAMATH_CALUDE_parabola_directrix_p_value_l3224_322447

/-- Given a parabola with equation x² = 2py where p > 0,
    if its directrix has equation y = -3, then p = 6 -/
theorem parabola_directrix_p_value (p : ℝ) :
  p > 0 →
  (∀ x y : ℝ, x^2 = 2*p*y) →
  (∀ y : ℝ, y = -3 → (∀ x : ℝ, x^2 ≠ 2*p*y)) →
  p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_p_value_l3224_322447


namespace NUMINAMATH_CALUDE_line_perp_from_plane_perp_l3224_322408

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perpLine : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perpPlane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpLineLine : Line → Line → Prop)

-- Theorem statement
theorem line_perp_from_plane_perp 
  (a b : Line) (α β : Plane) 
  (h1 : perpLine a α) 
  (h2 : perpLine b β) 
  (h3 : perpPlane α β) : 
  perpLineLine a b :=
sorry

end NUMINAMATH_CALUDE_line_perp_from_plane_perp_l3224_322408


namespace NUMINAMATH_CALUDE_parallelogram_centers_coincide_l3224_322463

-- Define a parallelogram
structure Parallelogram (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

-- Define a point on a line segment
def PointOnSegment (V : Type*) [AddCommGroup V] [Module ℝ V] (A B P : V) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

-- Define the center of a parallelogram
def CenterOfParallelogram (V : Type*) [AddCommGroup V] [Module ℝ V] (p : Parallelogram V) : V :=
  (1/2) • (p.A + p.C)

-- State the theorem
theorem parallelogram_centers_coincide
  (V : Type*) [AddCommGroup V] [Module ℝ V]
  (p₁ p₂ : Parallelogram V)
  (h₁ : PointOnSegment V p₁.A p₁.B p₂.A)
  (h₂ : PointOnSegment V p₁.B p₁.C p₂.B)
  (h₃ : PointOnSegment V p₁.C p₁.D p₂.C)
  (h₄ : PointOnSegment V p₁.D p₁.A p₂.D) :
  CenterOfParallelogram V p₁ = CenterOfParallelogram V p₂ :=
sorry

end NUMINAMATH_CALUDE_parallelogram_centers_coincide_l3224_322463


namespace NUMINAMATH_CALUDE_digital_earth_characteristics_l3224_322481

structure DigitalEarth where
  geographicInfoTech : Bool
  simulateReality : Bool
  centralizedStorage : Bool
  digitalInfoManagement : Bool

def is_correct_digital_earth (de : DigitalEarth) : Prop :=
  de.geographicInfoTech ∧ de.simulateReality ∧ ¬de.centralizedStorage ∧ de.digitalInfoManagement

theorem digital_earth_characteristics :
  ∃ (de : DigitalEarth), is_correct_digital_earth de :=
sorry

end NUMINAMATH_CALUDE_digital_earth_characteristics_l3224_322481


namespace NUMINAMATH_CALUDE_shaded_area_is_three_point_five_l3224_322488

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  L : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  O : ℝ × ℝ
  Q : ℝ × ℝ
  P : ℝ × ℝ
  h_dimensions : M.1 - L.1 = 4 ∧ O.2 - M.2 = 5
  h_equal_segments : 
    (M.1 - L.1 = 1) ∧ 
    (Q.2 - M.2 = 1) ∧ 
    (P.1 - Q.1 = 1) ∧ 
    (O.2 - P.2 = 1)

/-- The area of the shaded region in the special rectangle -/
def shadedArea (r : SpecialRectangle) : ℝ := sorry

/-- Theorem stating that the shaded area is 3.5 -/
theorem shaded_area_is_three_point_five (r : SpecialRectangle) : 
  shadedArea r = 3.5 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_three_point_five_l3224_322488


namespace NUMINAMATH_CALUDE_fruit_basket_count_l3224_322411

/-- The number of ways to choose k items from n identical items -/
def choose_with_repetition (n : ℕ) (k : ℕ) : ℕ := (n + k - 1).choose k

/-- The number of possible fruit baskets given the number of bananas and pears -/
def fruit_baskets (bananas : ℕ) (pears : ℕ) : ℕ :=
  (choose_with_repetition (bananas + 1) 1) * (choose_with_repetition (pears + 1) 1) - 1

theorem fruit_basket_count :
  fruit_baskets 6 9 = 69 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l3224_322411


namespace NUMINAMATH_CALUDE_train_distance_theorem_l3224_322442

/-- Calculates the distance traveled by a train given its average speed and travel time. -/
def train_distance (average_speed : ℝ) (travel_time : ℝ) : ℝ :=
  average_speed * travel_time

/-- Represents the train journey details -/
structure TrainJourney where
  average_speed : ℝ
  start_time : ℝ
  end_time : ℝ
  halt_time : ℝ

/-- Theorem stating the distance traveled by the train -/
theorem train_distance_theorem (journey : TrainJourney) 
  (h1 : journey.average_speed = 87)
  (h2 : journey.start_time = 9)
  (h3 : journey.end_time = 13.75)
  (h4 : journey.halt_time = 0.75) :
  train_distance journey.average_speed (journey.end_time - journey.start_time - journey.halt_time) = 348 := by
  sorry


end NUMINAMATH_CALUDE_train_distance_theorem_l3224_322442


namespace NUMINAMATH_CALUDE_sin_2alpha_equals_3_5_l3224_322465

theorem sin_2alpha_equals_3_5 (α : ℝ) (h : Real.tan (π/4 + α) = 2) : 
  Real.sin (2*α) = 3/5 := by sorry

end NUMINAMATH_CALUDE_sin_2alpha_equals_3_5_l3224_322465


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l3224_322468

theorem triangle_determinant_zero (A B C : ℝ) (h : A + B + C = π) :
  let f := fun x => (Real.cos x)^2
  let g := Real.tan
  Matrix.det !![f A, g A, 1; f B, g B, 1; f C, g C, 1] = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l3224_322468


namespace NUMINAMATH_CALUDE_negative_abs_neg_three_gt_negative_pi_l3224_322484

theorem negative_abs_neg_three_gt_negative_pi : -|-3| > -π := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_neg_three_gt_negative_pi_l3224_322484


namespace NUMINAMATH_CALUDE_exclusive_or_implications_l3224_322491

theorem exclusive_or_implications (p q : Prop) 
  (h_or : p ∨ q) (h_not_and : ¬(p ∧ q)) : 
  (q ↔ ¬p) ∧ (p ↔ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_exclusive_or_implications_l3224_322491


namespace NUMINAMATH_CALUDE_exists_x_sin_minus_x_negative_l3224_322418

open Real

theorem exists_x_sin_minus_x_negative :
  ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ sin x - x < 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_sin_minus_x_negative_l3224_322418


namespace NUMINAMATH_CALUDE_least_subtraction_l3224_322402

theorem least_subtraction (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((997 - y) % 5 = 3 ∧ (997 - y) % 9 = 3 ∧ (997 - y) % 11 = 3)) →
  (997 - x) % 5 = 3 ∧ (997 - x) % 9 = 3 ∧ (997 - x) % 11 = 3 →
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_l3224_322402


namespace NUMINAMATH_CALUDE_triangle_max_side_sum_l3224_322459

theorem triangle_max_side_sum (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a = Real.sqrt 3 ∧ 
  A = 2 * π / 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  (∀ b' c' : ℝ, b' + c' ≤ b + c → b' + c' ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_side_sum_l3224_322459


namespace NUMINAMATH_CALUDE_fraction_equality_l3224_322480

theorem fraction_equality (a b : ℝ) (h1 : a = (2/3) * b) (h2 : b ≠ 0) : 
  (9*a + 8*b) / (6*a) = 7/2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3224_322480


namespace NUMINAMATH_CALUDE_nh4cl_formation_l3224_322429

-- Define the chemical species
inductive ChemicalSpecies
| NH3
| HCl
| NH4Cl

-- Define a type for chemical reactions
structure Reaction where
  reactants : List (ChemicalSpecies × ℚ)
  products : List (ChemicalSpecies × ℚ)

-- Define the specific reaction
def nh3_hcl_reaction : Reaction :=
  { reactants := [(ChemicalSpecies.NH3, 1), (ChemicalSpecies.HCl, 1)],
    products := [(ChemicalSpecies.NH4Cl, 1)] }

-- Define the available amounts of reactants
def available_nh3 : ℚ := 1
def available_hcl : ℚ := 1

-- Theorem statement
theorem nh4cl_formation :
  let reaction := nh3_hcl_reaction
  let nh3_amount := available_nh3
  let hcl_amount := available_hcl
  let nh4cl_formed := 1
  nh4cl_formed = min nh3_amount hcl_amount := by sorry

end NUMINAMATH_CALUDE_nh4cl_formation_l3224_322429


namespace NUMINAMATH_CALUDE_range_of_g_l3224_322457

-- Define the function f(x) = |x|
def f (x : ℝ) : ℝ := |x|

-- Define the domain
def domain : Set ℝ := {x | -4 ≤ x ∧ x ≤ 4}

-- Define the function g(x) = f(x) - x
def g (x : ℝ) : ℝ := f x - x

-- Theorem statement
theorem range_of_g :
  {y | ∃ x ∈ domain, g x = y} = {y | 0 ≤ y ∧ y ≤ 8} := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l3224_322457


namespace NUMINAMATH_CALUDE_red_books_probability_l3224_322489

-- Define the number of red books and total books
def num_red_books : ℕ := 4
def total_books : ℕ := 8

-- Define the number of books to be selected
def books_selected : ℕ := 2

-- Define the probability function
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem red_books_probability :
  probability (combination num_red_books books_selected) (combination total_books books_selected) = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_red_books_probability_l3224_322489


namespace NUMINAMATH_CALUDE_virginia_adrienne_teaching_difference_l3224_322420

theorem virginia_adrienne_teaching_difference :
  ∀ (V A D : ℕ),
  V + A + D = 102 →
  D = 43 →
  V = D - 9 →
  V > A →
  V - A = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_virginia_adrienne_teaching_difference_l3224_322420


namespace NUMINAMATH_CALUDE_units_digit_sum_base8_l3224_322444

/-- The units digit of a number in a given base -/
def unitsDigit (n : ℕ) (base : ℕ) : ℕ := n % base

/-- Addition in a given base -/
def addInBase (a b base : ℕ) : ℕ := (a + b) % base

theorem units_digit_sum_base8 :
  unitsDigit (addInBase 45 37 8) 8 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_base8_l3224_322444


namespace NUMINAMATH_CALUDE_range_of_m_for_two_distinct_zeros_l3224_322441

/-- A quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + (m + 3)

/-- The discriminant of the quadratic function -/
def discriminant (m : ℝ) : ℝ := m^2 - 4*(m + 3)

/-- The theorem stating the range of m for which the quadratic function has two distinct zeros -/
theorem range_of_m_for_two_distinct_zeros :
  ∀ m : ℝ, (∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ m ∈ Set.Ioi 6 ∪ Set.Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_two_distinct_zeros_l3224_322441


namespace NUMINAMATH_CALUDE_remaining_length_is_23_l3224_322434

/-- Represents a figure with perpendicular sides -/
structure PerpendicularFigure where
  left_perimeter : ℝ
  right_perimeter : ℝ
  top_side : ℝ
  bottom_left : ℝ
  bottom_right : ℝ

/-- Calculates the total length of remaining segments after removal -/
def remaining_length (fig : PerpendicularFigure) : ℝ :=
  fig.left_perimeter + fig.right_perimeter + fig.bottom_left + fig.bottom_right

/-- Theorem stating the total length of remaining segments is 23 units -/
theorem remaining_length_is_23 (fig : PerpendicularFigure)
  (h1 : fig.left_perimeter = 10)
  (h2 : fig.right_perimeter = 7)
  (h3 : fig.top_side = 3)
  (h4 : fig.bottom_left = 2)
  (h5 : fig.bottom_right = 1) :
  remaining_length fig = 23 := by
  sorry

#eval remaining_length { left_perimeter := 10, right_perimeter := 7, top_side := 3, bottom_left := 2, bottom_right := 1 }

end NUMINAMATH_CALUDE_remaining_length_is_23_l3224_322434


namespace NUMINAMATH_CALUDE_no_real_roots_l3224_322409

theorem no_real_roots : ∀ x : ℝ, 2 * Real.cos (x / 2) ≠ 10^x + 10^(-x) + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3224_322409


namespace NUMINAMATH_CALUDE_melissa_points_per_game_l3224_322440

/-- The number of points Melissa scored in total -/
def total_points : ℕ := 91

/-- The number of games Melissa played -/
def num_games : ℕ := 13

/-- The number of points Melissa scored in each game -/
def points_per_game : ℕ := total_points / num_games

/-- Theorem stating that Melissa scored 7 points in each game -/
theorem melissa_points_per_game : points_per_game = 7 := by
  sorry

end NUMINAMATH_CALUDE_melissa_points_per_game_l3224_322440


namespace NUMINAMATH_CALUDE_complement_of_P_in_U_l3224_322415

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 - 5*x - 6 ≥ 0}

-- State the theorem
theorem complement_of_P_in_U : 
  Set.compl P = Set.Ioo (-1 : ℝ) (6 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_complement_of_P_in_U_l3224_322415


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3224_322466

theorem polynomial_simplification (p : ℝ) :
  (5 * p^4 + 2 * p^3 - 7 * p^2 + 3 * p - 2) + (-3 * p^4 + 4 * p^3 + 8 * p^2 - 2 * p + 6) =
  2 * p^4 + 6 * p^3 + p^2 + p + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3224_322466


namespace NUMINAMATH_CALUDE_total_oranges_picked_l3224_322443

/-- Represents the number of oranges picked by Jeremy on Monday -/
def monday_pick : ℕ := 100

/-- Represents the number of oranges picked by Jeremy and his brother on Tuesday -/
def tuesday_pick : ℕ := 3 * monday_pick

/-- Represents the number of oranges picked by all three on Wednesday -/
def wednesday_pick : ℕ := 2 * tuesday_pick

/-- Represents the number of oranges picked by the cousin on Wednesday -/
def cousin_wednesday_pick : ℕ := tuesday_pick - (tuesday_pick / 5)

/-- Represents the number of oranges picked by Jeremy on Thursday -/
def jeremy_thursday_pick : ℕ := (7 * monday_pick) / 10

/-- Represents the number of oranges picked by the brother on Thursday -/
def brother_thursday_pick : ℕ := tuesday_pick - monday_pick

/-- Represents the number of oranges picked by the cousin on Thursday -/
def cousin_thursday_pick : ℕ := cousin_wednesday_pick + (3 * cousin_wednesday_pick) / 10

/-- Represents the total number of oranges picked over the four days -/
def total_picked : ℕ := monday_pick + tuesday_pick + wednesday_pick + 
  (jeremy_thursday_pick + brother_thursday_pick + cousin_thursday_pick)

theorem total_oranges_picked : total_picked = 1642 := by sorry

end NUMINAMATH_CALUDE_total_oranges_picked_l3224_322443


namespace NUMINAMATH_CALUDE_no_two_cubes_between_squares_l3224_322496

theorem no_two_cubes_between_squares : ¬∃ (a b n : ℕ+), n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_two_cubes_between_squares_l3224_322496


namespace NUMINAMATH_CALUDE_grapes_boxes_count_l3224_322433

def asparagus_bundles : ℕ := 60
def asparagus_price : ℚ := 3
def grape_price : ℚ := 2.5
def apple_count : ℕ := 700
def apple_price : ℚ := 0.5
def total_worth : ℚ := 630

theorem grapes_boxes_count :
  ∃ (grape_boxes : ℕ),
    grape_boxes * grape_price +
    asparagus_bundles * asparagus_price +
    apple_count * apple_price = total_worth ∧
    grape_boxes = 40 := by sorry

end NUMINAMATH_CALUDE_grapes_boxes_count_l3224_322433


namespace NUMINAMATH_CALUDE_pet_shelter_adoption_time_l3224_322460

/-- Given an initial number of puppies, additional puppies, and a daily adoption rate,
    calculate the number of days required to adopt all puppies. -/
def days_to_adopt (initial : ℕ) (additional : ℕ) (adoption_rate : ℕ) : ℕ :=
  (initial + additional) / adoption_rate

/-- Theorem: For the given problem, it takes 2 days to adopt all puppies. -/
theorem pet_shelter_adoption_time : days_to_adopt 3 3 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_shelter_adoption_time_l3224_322460


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l3224_322470

/-- 
Given a rectangular plot where:
- The area is 18 times the breadth
- The length is 10 meters more than the breadth
Prove that the breadth is 8 meters
-/
theorem rectangular_plot_breadth (b : ℝ) (l : ℝ) (A : ℝ) : 
  A = 18 * b →
  l = b + 10 →
  A = l * b →
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l3224_322470


namespace NUMINAMATH_CALUDE_bus_exit_ways_10_5_l3224_322432

/-- The number of possible ways for passengers to get off a bus -/
def bus_exit_ways (num_passengers : ℕ) (num_stops : ℕ) : ℕ :=
  num_stops ^ num_passengers

/-- Theorem: Given 10 passengers and 5 stops, the number of possible ways
    for passengers to get off the bus is 5^10 -/
theorem bus_exit_ways_10_5 :
  bus_exit_ways 10 5 = 5^10 := by
  sorry

end NUMINAMATH_CALUDE_bus_exit_ways_10_5_l3224_322432


namespace NUMINAMATH_CALUDE_problem_statement_l3224_322419

theorem problem_statement (a b : ℝ) 
  (h1 : a / b + b / a = 5 / 2)
  (h2 : a - b = 3 / 2) :
  (a^2 + 2*a*b + b^2 + 2*a^2*b + 2*a*b^2 + a^2*b^2 = 0) ∨ 
  (a^2 + 2*a*b + b^2 + 2*a^2*b + 2*a*b^2 + a^2*b^2 = 81) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3224_322419


namespace NUMINAMATH_CALUDE_stationery_cost_theorem_l3224_322421

/-- Calculates the total cost of stationery given the number of pencil boxes, pencils per box,
    pencil cost, pen cost, and additional pens ordered. -/
def total_stationery_cost (pencil_boxes : ℕ) (pencils_per_box : ℕ) (pencil_cost : ℕ) 
                          (pen_cost : ℕ) (additional_pens : ℕ) : ℕ :=
  let total_pencils := pencil_boxes * pencils_per_box
  let total_pens := 2 * total_pencils + additional_pens
  let pencil_total_cost := total_pencils * pencil_cost
  let pen_total_cost := total_pens * pen_cost
  pencil_total_cost + pen_total_cost

/-- Theorem stating that the total cost of stationery for the given conditions is $18300. -/
theorem stationery_cost_theorem : 
  total_stationery_cost 15 80 4 5 300 = 18300 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_theorem_l3224_322421


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_and_equation_l3224_322412

/-- The tangent line to y = x^4 at (1, 1) is perpendicular to x + 4y - 8 = 0 and has equation 4x - y - 3 = 0 -/
theorem tangent_line_perpendicular_and_equation (x y : ℝ) : 
  let f : ℝ → ℝ := fun x => x^4
  let tangent_slope : ℝ := (deriv f) 1
  let perpendicular_line_slope : ℝ := -1/4
  let tangent_equation : ℝ → ℝ → Prop := fun x y => 4*x - y - 3 = 0
  tangent_slope * perpendicular_line_slope = -1 ∧
  tangent_equation 1 1 ∧
  (∀ x y, tangent_equation x y ↔ y - 1 = tangent_slope * (x - 1)) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_and_equation_l3224_322412


namespace NUMINAMATH_CALUDE_abc_inequality_l3224_322472

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1)
  (eq_a : a - 2 = Real.log (a / 2))
  (eq_b : b - 3 = Real.log (b / 3))
  (eq_c : c - 3 = Real.log (c / 2)) :
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3224_322472


namespace NUMINAMATH_CALUDE_carls_lawn_area_l3224_322430

/-- Represents a rectangular lawn with fence posts -/
structure FencedLawn where
  short_side : ℕ  -- Number of posts on shorter side
  long_side : ℕ   -- Number of posts on longer side
  post_spacing : ℕ -- Distance between posts in yards

/-- The total number of fence posts -/
def total_posts (lawn : FencedLawn) : ℕ :=
  2 * (lawn.short_side + lawn.long_side) - 4

/-- The area of the lawn in square yards -/
def lawn_area (lawn : FencedLawn) : ℕ :=
  (lawn.short_side - 1) * lawn.post_spacing * ((lawn.long_side - 1) * lawn.post_spacing)

/-- Theorem stating the area of Carl's lawn -/
theorem carls_lawn_area : 
  ∃ (lawn : FencedLawn), 
    lawn.short_side = 4 ∧ 
    lawn.long_side = 12 ∧ 
    lawn.post_spacing = 3 ∧ 
    total_posts lawn = 24 ∧ 
    lawn_area lawn = 243 := by
  sorry


end NUMINAMATH_CALUDE_carls_lawn_area_l3224_322430


namespace NUMINAMATH_CALUDE_sams_eatery_meal_cost_l3224_322487

/-- Calculates the cost of a meal at Sam's Eatery with a discount --/
def meal_cost (hamburger_price : ℚ) (fries_price : ℚ) (drink_price : ℚ) 
              (num_hamburgers : ℕ) (num_fries : ℕ) (num_drinks : ℕ) 
              (discount_percent : ℚ) : ℕ :=
  let total_before_discount := hamburger_price * num_hamburgers + 
                               fries_price * num_fries + 
                               drink_price * num_drinks
  let discount_amount := total_before_discount * (discount_percent / 100)
  let total_after_discount := total_before_discount - discount_amount
  (total_after_discount + 1/2).floor.toNat

/-- The cost of the meal at Sam's Eatery is 35 dollars --/
theorem sams_eatery_meal_cost : 
  meal_cost 5 3 2 3 4 6 10 = 35 := by
  sorry


end NUMINAMATH_CALUDE_sams_eatery_meal_cost_l3224_322487


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l3224_322413

theorem square_perimeter_problem (perimeter_I perimeter_II : ℝ) 
  (h1 : perimeter_I = 16)
  (h2 : perimeter_II = 36)
  (side_I : ℝ) (side_II : ℝ) (side_III : ℝ)
  (h3 : side_I = perimeter_I / 4)
  (h4 : side_II = perimeter_II / 4)
  (h5 : side_III = Real.sqrt (side_I * side_II))
  (perimeter_III : ℝ)
  (h6 : perimeter_III = 4 * side_III) :
  perimeter_III = 24 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l3224_322413


namespace NUMINAMATH_CALUDE_indeterminateNatureAndSanity_l3224_322499

-- Define the types for Transylvanians
inductive Transylvanian
| Human
| Vampire

-- Define the mental state
inductive MentalState
| Sane
| Insane

-- Define reliability
def isReliable (t : Transylvanian) (m : MentalState) : Prop :=
  (t = Transylvanian.Human ∧ m = MentalState.Sane) ∨
  (t = Transylvanian.Vampire ∧ m = MentalState.Insane)

-- Define unreliability
def isUnreliable (t : Transylvanian) (m : MentalState) : Prop :=
  (t = Transylvanian.Human ∧ m = MentalState.Insane) ∨
  (t = Transylvanian.Vampire ∧ m = MentalState.Sane)

-- Define the statement function
def statesTrue (t : Transylvanian) (m : MentalState) : Prop :=
  isReliable t m

-- Define the answer to the question "Are you reliable?"
def answersYes (t : Transylvanian) (m : MentalState) : Prop :=
  (isReliable t m ∧ statesTrue t m) ∨ (isUnreliable t m ∧ ¬statesTrue t m)

-- Theorem: It's impossible to determine the nature or sanity of a Transylvanian
-- based on their answer to the question "Are you reliable?"
theorem indeterminateNatureAndSanity (t : Transylvanian) (m : MentalState) :
  answersYes t m → 
  (∃ (t' : Transylvanian) (m' : MentalState), t' ≠ t ∨ m' ≠ m) ∧ answersYes t' m' :=
sorry


end NUMINAMATH_CALUDE_indeterminateNatureAndSanity_l3224_322499


namespace NUMINAMATH_CALUDE_distribute_six_students_two_activities_l3224_322416

/-- The number of ways to distribute n students between 2 activities,
    where each activity can have at most k students. -/
def distribute_students (n k : ℕ) : ℕ :=
  Nat.choose n k + Nat.choose n (n / 2)

/-- Theorem stating that the number of ways to distribute 6 students
    between 2 activities, where each activity can have at most 4 students,
    is equal to 35. -/
theorem distribute_six_students_two_activities :
  distribute_students 6 4 = 35 := by
  sorry

#eval distribute_students 6 4

end NUMINAMATH_CALUDE_distribute_six_students_two_activities_l3224_322416


namespace NUMINAMATH_CALUDE_uncool_students_l3224_322483

theorem uncool_students (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (cool_both : ℕ) (cool_siblings : ℕ) 
  (h1 : total = 50)
  (h2 : cool_dads = 25)
  (h3 : cool_moms = 30)
  (h4 : cool_both = 15)
  (h5 : cool_siblings = 10) :
  total - (cool_dads + cool_moms - cool_both) - cool_siblings = 10 := by
  sorry

end NUMINAMATH_CALUDE_uncool_students_l3224_322483


namespace NUMINAMATH_CALUDE_square_fence_poles_l3224_322461

/-- Given a square fence with a total of 104 poles, prove that the number of poles on each side is 26. -/
theorem square_fence_poles (total_poles : ℕ) (h1 : total_poles = 104) :
  ∃ (side_poles : ℕ), side_poles * 4 = total_poles ∧ side_poles = 26 := by
  sorry

end NUMINAMATH_CALUDE_square_fence_poles_l3224_322461


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_b_l3224_322485

theorem gcd_of_polynomial_and_b (b : ℤ) (h : 792 ∣ b) :
  Int.gcd (5 * b^3 + 2 * b^2 + 6 * b + 99) b = 99 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_b_l3224_322485


namespace NUMINAMATH_CALUDE_simplify_fraction_l3224_322417

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((x + 1) / (x - 1)) = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3224_322417


namespace NUMINAMATH_CALUDE_beads_per_necklace_l3224_322446

/-- Given that Emily made 6 necklaces and used a total of 18 beads,
    prove that each necklace needs 3 beads. -/
theorem beads_per_necklace :
  let total_necklaces : ℕ := 6
  let total_beads : ℕ := 18
  total_beads / total_necklaces = 3 := by sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l3224_322446


namespace NUMINAMATH_CALUDE_hyperbola_condition_l3224_322494

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 3) - y^2 / (k + 3) = 1

-- Define the sufficient condition
def sufficient_condition (k : ℝ) : Prop :=
  k > 3 → is_hyperbola k

-- Define the necessary condition
def necessary_condition (k : ℝ) : Prop :=
  is_hyperbola k → k > 3

-- Theorem statement
theorem hyperbola_condition :
  (∀ k : ℝ, sufficient_condition k) ∧ ¬(∀ k : ℝ, necessary_condition k) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l3224_322494


namespace NUMINAMATH_CALUDE_program_output_l3224_322476

def S : ℕ → ℚ
  | 0 => 2
  | n + 1 => 1 / (1 - S n)

theorem program_output :
  S 2017 = 2 ∧ S 2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_program_output_l3224_322476


namespace NUMINAMATH_CALUDE_min_sum_distances_squared_l3224_322452

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of the center of the ellipse -/
def center : ℝ × ℝ := (0, 0)

/-- Definition of the left focus of the ellipse -/
def left_focus : ℝ × ℝ := (-1, 0)

/-- Square of the distance between two points -/
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The minimum value of |OP|^2 + |PF|^2 is 2 -/
theorem min_sum_distances_squared :
  ∀ (x y : ℝ), is_on_ellipse x y →
  ∃ (min : ℝ), min = 2 ∧
  ∀ (p : ℝ × ℝ), is_on_ellipse p.1 p.2 →
  distance_squared center p + distance_squared p left_focus ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_squared_l3224_322452


namespace NUMINAMATH_CALUDE_square_diagonal_and_area_l3224_322469

/-- Given a square with side length 30√3 cm, this theorem proves the length of its diagonal and its area. -/
theorem square_diagonal_and_area :
  let side_length : ℝ := 30 * Real.sqrt 3
  let diagonal : ℝ := side_length * Real.sqrt 2
  let area : ℝ := side_length ^ 2
  diagonal = 30 * Real.sqrt 6 ∧ area = 2700 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_and_area_l3224_322469


namespace NUMINAMATH_CALUDE_least_phrases_to_learn_l3224_322400

theorem least_phrases_to_learn (total_phrases : ℕ) (min_grade : ℚ) : 
  total_phrases = 600 → min_grade = 90 / 100 → 
  ∃ (least_phrases : ℕ), 
    (least_phrases : ℚ) / total_phrases ≥ min_grade ∧
    ∀ (n : ℕ), (n : ℚ) / total_phrases ≥ min_grade → n ≥ least_phrases ∧
    least_phrases = 540 :=
by sorry

end NUMINAMATH_CALUDE_least_phrases_to_learn_l3224_322400


namespace NUMINAMATH_CALUDE_apple_box_problem_l3224_322431

theorem apple_box_problem (apples oranges : ℕ) : 
  oranges = 12 ∧ 
  (apples : ℝ) / (apples + (oranges - 6 : ℕ) : ℝ) = 0.7 → 
  apples = 14 := by
sorry

end NUMINAMATH_CALUDE_apple_box_problem_l3224_322431


namespace NUMINAMATH_CALUDE_percentage_decrease_l3224_322456

theorem percentage_decrease (w : ℝ) (x : ℝ) (h1 : w = 80) (h2 : w * (1 + 0.125) - w * (1 - x / 100) = 30) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_l3224_322456


namespace NUMINAMATH_CALUDE_xyz_value_l3224_322437

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 10 := by sorry

end NUMINAMATH_CALUDE_xyz_value_l3224_322437


namespace NUMINAMATH_CALUDE_complex_division_negative_l3224_322401

theorem complex_division_negative (m : ℝ) : 
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := m - I
  (z₁ / z₂).re < 0 ∧ (z₁ / z₂).im = 0 → m = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_complex_division_negative_l3224_322401


namespace NUMINAMATH_CALUDE_a_15_value_l3224_322477

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem a_15_value (a : ℕ → ℝ) (h : arithmetic_sequence a) (h5 : a 5 = 5) (h10 : a 10 = 15) :
  a 15 = 25 := by
  sorry

end NUMINAMATH_CALUDE_a_15_value_l3224_322477


namespace NUMINAMATH_CALUDE_units_digit_of_29_power_8_7_l3224_322428

theorem units_digit_of_29_power_8_7 : 29^(8^7) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_29_power_8_7_l3224_322428


namespace NUMINAMATH_CALUDE_min_value_theorem_l3224_322464

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a + c) * (a + b) = 6 - 2 * Real.sqrt 5) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3224_322464


namespace NUMINAMATH_CALUDE_quadrilateral_area_theorem_l3224_322475

/-- Represents a triangle divided into four smaller triangles and a quadrilateral --/
structure DividedTriangle where
  /-- Area of the first small triangle --/
  area1 : ℝ
  /-- Area of the second small triangle --/
  area2 : ℝ
  /-- Area of the third small triangle --/
  area3 : ℝ
  /-- Area of the fourth small triangle --/
  area4 : ℝ
  /-- Area of the central quadrilateral --/
  quadArea : ℝ

/-- Theorem stating that if the areas of the four triangles are 5, 10, 10, and 8,
    then the area of the quadrilateral is 15 --/
theorem quadrilateral_area_theorem (t : DividedTriangle)
    (h1 : t.area1 = 5)
    (h2 : t.area2 = 10)
    (h3 : t.area3 = 10)
    (h4 : t.area4 = 8) :
    t.quadArea = 15 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_theorem_l3224_322475


namespace NUMINAMATH_CALUDE_exterior_angles_hexagon_pentagon_l3224_322467

/-- The sum of exterior angles of a polygon -/
def sum_exterior_angles (n : ℕ) : ℝ := 360

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- A pentagon has 5 sides -/
def pentagon_sides : ℕ := 5

theorem exterior_angles_hexagon_pentagon : 
  sum_exterior_angles hexagon_sides = sum_exterior_angles pentagon_sides := by
  sorry

end NUMINAMATH_CALUDE_exterior_angles_hexagon_pentagon_l3224_322467


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3224_322478

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_increasing : ∀ x y, x < y → f x < f y)
variable (h_f_0 : f 0 = -1)
variable (h_f_3 : f 3 = 1)

-- Define the solution set
def solution_set := {x : ℝ | |f (x + 1)| < 1}

-- State the theorem
theorem solution_set_equivalence : 
  solution_set f = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3224_322478


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3224_322406

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  x + 2*y ≥ 8 ∧ (x + 2*y = 8 ↔ x = 2*y) :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3224_322406


namespace NUMINAMATH_CALUDE_parabolic_archway_height_l3224_322473

/-- Represents a parabolic function of the form f(x) = ax² + 20 -/
def parabolic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 20

theorem parabolic_archway_height :
  ∃ a : ℝ, 
    (parabolic_function a 25 = 0) ∧ 
    (parabolic_function a 0 = 20) ∧
    (parabolic_function a 10 = 16.8) := by
  sorry

end NUMINAMATH_CALUDE_parabolic_archway_height_l3224_322473


namespace NUMINAMATH_CALUDE_meaningful_expression_l3224_322495

theorem meaningful_expression (x : ℝ) : 
  (10 - x ≥ 0 ∧ x ≠ 4) ↔ x = 8 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3224_322495


namespace NUMINAMATH_CALUDE_abs_neg_three_equals_three_l3224_322497

theorem abs_neg_three_equals_three :
  abs (-3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_equals_three_l3224_322497


namespace NUMINAMATH_CALUDE_stimulus_check_amount_l3224_322453

theorem stimulus_check_amount : ∃ T : ℚ, 
  (27 / 125 : ℚ) * T = 432 ∧ T = 2000 := by
  sorry

end NUMINAMATH_CALUDE_stimulus_check_amount_l3224_322453


namespace NUMINAMATH_CALUDE_quadratic_vertex_l3224_322482

/-- The quadratic function f(x) = -3(x+2)^2 + 1 --/
def f (x : ℝ) : ℝ := -3 * (x + 2)^2 + 1

/-- The vertex of the quadratic function f --/
def vertex : ℝ × ℝ := (-2, 1)

theorem quadratic_vertex :
  (∀ x : ℝ, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l3224_322482


namespace NUMINAMATH_CALUDE_donut_distribution_proof_l3224_322471

/-- The number of ways to distribute donuts satisfying the given conditions -/
def donut_combinations : ℕ := 126

/-- The number of donut types -/
def num_types : ℕ := 5

/-- The total number of donuts to be purchased -/
def total_donuts : ℕ := 10

/-- The number of remaining donuts after selecting one of each type -/
def remaining_donuts : ℕ := total_donuts - num_types

/-- Binomial coefficient calculation -/
def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else (List.range k).foldl (fun m i => m * (n - i) / (i + 1)) 1

theorem donut_distribution_proof :
  binom (remaining_donuts + num_types - 1) (num_types - 1) = donut_combinations :=
by sorry

end NUMINAMATH_CALUDE_donut_distribution_proof_l3224_322471


namespace NUMINAMATH_CALUDE_student_goldfish_difference_l3224_322492

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 20

/-- The number of goldfish in each fourth-grade classroom -/
def goldfish_per_classroom : ℕ := 3

/-- The theorem stating the difference between the total number of students and goldfish -/
theorem student_goldfish_difference :
  num_classrooms * students_per_classroom - num_classrooms * goldfish_per_classroom = 85 := by
  sorry

end NUMINAMATH_CALUDE_student_goldfish_difference_l3224_322492


namespace NUMINAMATH_CALUDE_pin_sequence_solution_l3224_322493

def pin_sequence (k : ℕ) (n : ℕ) : ℕ := 2 + k * (n - 1)

theorem pin_sequence_solution :
  ∀ k : ℕ, (pin_sequence k 10 > 45 ∧ pin_sequence k 15 < 90) ↔ (k = 5 ∨ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_pin_sequence_solution_l3224_322493


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3224_322403

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) (h_sum : a 1 + a 19 = -18) :
  a 10 = -9 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3224_322403


namespace NUMINAMATH_CALUDE_total_skittles_calculation_l3224_322422

/-- The number of Skittles each friend receives -/
def skittles_per_friend : ℝ := 40.0

/-- The number of friends -/
def number_of_friends : ℝ := 5.0

/-- The total number of Skittles given to all friends -/
def total_skittles : ℝ := skittles_per_friend * number_of_friends

theorem total_skittles_calculation : total_skittles = 200.0 := by
  sorry

end NUMINAMATH_CALUDE_total_skittles_calculation_l3224_322422


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3224_322462

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Last term of an arithmetic sequence -/
def last_term (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_sum :
  ∃ n : ℕ, n > 0 ∧ last_term 1 2 n = 21 ∧ arithmetic_sum 1 2 n = 121 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3224_322462


namespace NUMINAMATH_CALUDE_todd_spending_proof_l3224_322451

/-- Calculates the total amount Todd spent given the prices of items, discount rate, and tax rate -/
def todd_spending (candy_price cookies_price soda_price : ℚ) (discount_rate tax_rate : ℚ) : ℚ :=
  let discounted_candy := candy_price * (1 - discount_rate)
  let subtotal := discounted_candy + cookies_price + soda_price
  let total := subtotal * (1 + tax_rate)
  total

/-- Proves that Todd's total spending is $5.53 given the problem conditions -/
theorem todd_spending_proof :
  todd_spending 1.14 2.39 1.75 0.1 0.07 = 5.53 := by
  sorry

end NUMINAMATH_CALUDE_todd_spending_proof_l3224_322451


namespace NUMINAMATH_CALUDE_time_after_2023_hours_l3224_322454

def hours_later (current_time : Nat) (hours_passed : Nat) : Nat :=
  (current_time + hours_passed) % 12

theorem time_after_2023_hours :
  let current_time := 9
  let hours_passed := 2023
  hours_later current_time hours_passed = 8 := by
sorry

end NUMINAMATH_CALUDE_time_after_2023_hours_l3224_322454


namespace NUMINAMATH_CALUDE_multiple_remainder_l3224_322439

theorem multiple_remainder (n m : ℤ) (h1 : n % 7 = 1) (h2 : ∃ k, (k * n) % 7 = 3) :
  m % 7 = 3 → (m * n) % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_multiple_remainder_l3224_322439


namespace NUMINAMATH_CALUDE_min_value_expression_l3224_322414

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 9*a*b + 9*b^2 + 3*c^2 ≥ 60 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 27 ∧ 
    a₀^2 + 9*a₀*b₀ + 9*b₀^2 + 3*c₀^2 = 60 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3224_322414


namespace NUMINAMATH_CALUDE_sector_max_area_l3224_322438

/-- Given a sector with circumference 12 cm, its maximum area is 9 cm². -/
theorem sector_max_area (r l : ℝ) (h_circumference : 2 * r + l = 12) :
  (1/2 : ℝ) * l * r ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l3224_322438


namespace NUMINAMATH_CALUDE_triangle_ratio_sum_l3224_322424

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that if ∠C = 60° and (a / (b+c)) + (b / (a+c)) = P, then P = 1 -/
theorem triangle_ratio_sum (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  let P := a / (b + c) + b / (a + c)
  (c^2 = a^2 + b^2 - a * b) → P = 1 := by
  sorry


end NUMINAMATH_CALUDE_triangle_ratio_sum_l3224_322424


namespace NUMINAMATH_CALUDE_evening_temperature_l3224_322474

def initial_temp : Int := -7
def temp_rise : Int := 11
def temp_drop : Int := 9

theorem evening_temperature :
  initial_temp + temp_rise - temp_drop = -5 :=
by sorry

end NUMINAMATH_CALUDE_evening_temperature_l3224_322474


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3224_322458

/-- Conversion of polar coordinates to rectangular coordinates -/
theorem polar_to_rectangular_conversion
  (r : ℝ) (θ : ℝ) 
  (h : r = 10 ∧ θ = 3 * π / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (-5 * Real.sqrt 2, 5 * Real.sqrt 2) := by
  sorry

#check polar_to_rectangular_conversion

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3224_322458


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3224_322448

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 55,
    where one side of the equilateral triangle is also a side of the isosceles triangle,
    the base of the isosceles triangle is 15 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 60)
  (h_isosceles_perimeter : isosceles_perimeter = 55)
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) :
  isosceles_base = 15 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3224_322448


namespace NUMINAMATH_CALUDE_tank_filling_l3224_322426

theorem tank_filling (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 25 →
  capacity_ratio = 2 / 5 →
  ∃ (new_buckets : ℕ), 
    (↑new_buckets : ℚ) > (↑original_buckets / capacity_ratio) ∧ 
    (↑new_buckets : ℚ) ≤ (↑original_buckets / capacity_ratio + 1) ∧
    new_buckets = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_filling_l3224_322426


namespace NUMINAMATH_CALUDE_expenditure_ratio_l3224_322427

theorem expenditure_ratio (rajan_income balan_income rajan_expenditure balan_expenditure : ℚ) : 
  (rajan_income / balan_income = 7 / 6) →
  (rajan_income = 7000) →
  (rajan_income - rajan_expenditure = 1000) →
  (balan_income - balan_expenditure = 1000) →
  (rajan_expenditure / balan_expenditure = 6 / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l3224_322427


namespace NUMINAMATH_CALUDE_eulersRelationHoldsForNewPolyhedron_l3224_322498

/-- Represents a polyhedron formed from a cube by marking midpoints of edges,
    connecting them on each face, and cutting off 8 pyramids around each vertex. -/
structure NewPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's Relation for convex polyhedra -/
def eulersRelation (p : NewPolyhedron) : Prop :=
  p.vertices + p.faces = p.edges + 2

/-- The specific new polyhedron formed from the cube -/
def cubeTransformedPolyhedron : NewPolyhedron :=
  { vertices := 12
  , edges := 24
  , faces := 14 }

/-- Theorem stating that Euler's Relation holds for the new polyhedron -/
theorem eulersRelationHoldsForNewPolyhedron :
  eulersRelation cubeTransformedPolyhedron := by
  sorry

end NUMINAMATH_CALUDE_eulersRelationHoldsForNewPolyhedron_l3224_322498


namespace NUMINAMATH_CALUDE_expression_simplification_l3224_322410

theorem expression_simplification (x y : ℝ) (hx : x ≥ 0) :
  (3 / 5) * Real.sqrt (x * y^2) / (-(4 / 15) * Real.sqrt (y / x)) * (-(5 / 6) * Real.sqrt (x^3 * y)) =
  (15 * x^2 * y * Real.sqrt x) / 8 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3224_322410


namespace NUMINAMATH_CALUDE_units_digit_of_2189_power_1242_l3224_322490

theorem units_digit_of_2189_power_1242 : ∃ n : ℕ, 2189^1242 ≡ 1 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_2189_power_1242_l3224_322490


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3224_322407

theorem complex_sum_theorem (A O P S : ℂ) 
  (hA : A = 2 + I) 
  (hO : O = 3 - 2*I) 
  (hP : P = 1 + I) 
  (hS : S = 4 + 3*I) : 
  A - O + P + S = 4 + 7*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3224_322407


namespace NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3224_322425

theorem square_ratio_side_length_sum (area_ratio : ℚ) :
  area_ratio = 135 / 45 →
  ∃ (a b c : ℕ), 
    (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧
    (Real.sqrt (area_ratio) = (a * Real.sqrt b) / c) ∧
    (a + b + c = 5) := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3224_322425


namespace NUMINAMATH_CALUDE_common_chord_and_length_l3224_322405

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the equation of the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- State the theorem
theorem common_chord_and_length :
  -- The equation of the common chord
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord x y) ∧
  -- The length of the common chord
  (∃ a b c d : ℝ,
    circle1 a b ∧ circle2 a b ∧ circle1 c d ∧ circle2 c d ∧
    common_chord a b ∧ common_chord c d ∧
    ((a - c)^2 + (b - d)^2) = 20) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_and_length_l3224_322405


namespace NUMINAMATH_CALUDE_wedge_volume_l3224_322404

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (θ : ℝ) : 
  d = 12 →  -- diameter of the log
  θ = π/4 →  -- angle between the two cuts (45° in radians)
  (1/2) * π * (d/2)^2 * d = 216 * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l3224_322404


namespace NUMINAMATH_CALUDE_choir_third_group_members_l3224_322450

theorem choir_third_group_members (total_members : ℕ) (group1_members : ℕ) (group2_members : ℕ) 
  (h1 : total_members = 70)
  (h2 : group1_members = 25)
  (h3 : group2_members = 30) :
  total_members - (group1_members + group2_members) = 15 := by
sorry

end NUMINAMATH_CALUDE_choir_third_group_members_l3224_322450


namespace NUMINAMATH_CALUDE_problem_statement_l3224_322486

theorem problem_statement (a b : ℝ) (h : a + b = 1) :
  (a^3 + b^3 ≥ 1/4) ∧
  (∃ x : ℝ, |x - a| + |x - b| ≤ 5 → 0 ≤ 2*a + 3*b ∧ 2*a + 3*b ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3224_322486


namespace NUMINAMATH_CALUDE_existence_of_fractions_l3224_322423

theorem existence_of_fractions (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  ∃ (p q r s : ℕ+), 
    (a < (p : ℝ) / q) ∧ 
    ((p : ℝ) / q < (r : ℝ) / s) ∧ 
    ((r : ℝ) / s < b) ∧ 
    (p^2 + q^2 : ℕ) = (r^2 + s^2 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_fractions_l3224_322423


namespace NUMINAMATH_CALUDE_base_3_minus_base_8_digits_of_2048_l3224_322455

theorem base_3_minus_base_8_digits_of_2048 : 
  (Nat.log 3 2048 + 1) - (Nat.log 8 2048 + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_3_minus_base_8_digits_of_2048_l3224_322455


namespace NUMINAMATH_CALUDE_angle_ADB_is_270_degrees_l3224_322435

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the properties of the triangle
def isRightTriangle (t : Triangle) : Prop := sorry

def angleAIs45 (t : Triangle) : Prop := sorry

def angleBIs45 (t : Triangle) : Prop := sorry

-- Define the angle bisectors and their intersection
def angleBisectorA (t : Triangle) : Line := sorry

def angleBisectorB (t : Triangle) : Line := sorry

def D (t : Triangle) : Point := sorry

-- Define the measure of angle ADB
def measureAngleADB (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_ADB_is_270_degrees (t : Triangle) :
  isRightTriangle t → angleAIs45 t → angleBIs45 t →
  measureAngleADB t = 270 :=
sorry

end NUMINAMATH_CALUDE_angle_ADB_is_270_degrees_l3224_322435


namespace NUMINAMATH_CALUDE_water_cup_fills_l3224_322436

theorem water_cup_fills (container_volume : ℚ) (cup_volume : ℚ) : 
  container_volume = 13/3 → cup_volume = 1/6 → 
  (container_volume / cup_volume : ℚ) = 26 := by
  sorry

end NUMINAMATH_CALUDE_water_cup_fills_l3224_322436


namespace NUMINAMATH_CALUDE_gcd_876543_765432_l3224_322479

theorem gcd_876543_765432 : Nat.gcd 876543 765432 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_876543_765432_l3224_322479


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_500_l3224_322449

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_factorials (n : ℕ) : ℕ :=
  (List.range n).map factorial |> List.sum

theorem units_digit_sum_factorials_500 :
  units_digit (sum_factorials 500) = units_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_500_l3224_322449


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_with_geometric_sides_is_square_l3224_322445

/-- A quadrilateral inscribed around a circle with sides in geometric progression is a square -/
theorem inscribed_quadrilateral_with_geometric_sides_is_square
  (R : ℝ) -- radius of the inscribed circle
  (a : ℝ) -- first term of the geometric progression
  (r : ℝ) -- common ratio of the geometric progression
  (h1 : R > 0)
  (h2 : a > 0)
  (h3 : r > 0)
  (h4 : a + a * r^3 = a * r + a * r^2) -- Pitot's theorem
  : 
  r = 1 ∧ -- all sides are equal
  R = a / 2 ∧ -- radius is half the side length
  a^2 = 4 * R^2 -- area of the quadrilateral
  := by sorry

#check inscribed_quadrilateral_with_geometric_sides_is_square

end NUMINAMATH_CALUDE_inscribed_quadrilateral_with_geometric_sides_is_square_l3224_322445
