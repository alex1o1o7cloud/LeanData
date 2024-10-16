import Mathlib

namespace NUMINAMATH_CALUDE_water_pouring_problem_l4005_400533

def water_remaining (n : ℕ) : ℚ :=
  1 / (n + 1)

theorem water_pouring_problem :
  ∃ n : ℕ, water_remaining n = 1 / 10 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_water_pouring_problem_l4005_400533


namespace NUMINAMATH_CALUDE_problem_solution_l4005_400518

def A : Set ℝ := {-1, 0}
def B (x : ℝ) : Set ℝ := {0, 1, x+2}

theorem problem_solution (x : ℝ) (h : A ⊆ B x) : x = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4005_400518


namespace NUMINAMATH_CALUDE_inverse_of_ln_l4005_400521

theorem inverse_of_ln (x : ℝ) : 
  (fun y ↦ Real.exp y) ∘ (fun x ↦ Real.log x) = id ∧ 
  (fun x ↦ Real.log x) ∘ (fun y ↦ Real.exp y) = id :=
sorry

end NUMINAMATH_CALUDE_inverse_of_ln_l4005_400521


namespace NUMINAMATH_CALUDE_max_trig_sum_value_l4005_400544

theorem max_trig_sum_value (θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ) :
  (∀ φ₁ φ₂ φ₃ φ₄ φ₅ φ₆ : ℝ,
    (Real.cos φ₁ * Real.sin φ₂ + Real.cos φ₂ * Real.sin φ₃ + 
     Real.cos φ₃ * Real.sin φ₄ + Real.cos φ₄ * Real.sin φ₅ + 
     Real.cos φ₅ * Real.sin φ₆ + Real.cos φ₆ * Real.sin φ₁) ≤
    (Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + 
     Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₅ + 
     Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁)) ∧
  (Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + 
   Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₅ + 
   Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁) = 3 + 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_trig_sum_value_l4005_400544


namespace NUMINAMATH_CALUDE_female_managers_count_l4005_400525

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  female_employees : ℕ
  total_managers : ℕ
  male_managers : ℕ

/-- Conditions for the company -/
def company_conditions (c : Company) : Prop :=
  c.female_employees = 500 ∧
  c.total_managers = (2 * c.total_employees) / 5 ∧
  c.male_managers = (2 * (c.total_employees - c.female_employees)) / 5

/-- Theorem stating the number of female managers -/
theorem female_managers_count (c : Company) 
  (h : company_conditions c) : 
  c.total_managers - c.male_managers = 200 := by
  sorry

#check female_managers_count

end NUMINAMATH_CALUDE_female_managers_count_l4005_400525


namespace NUMINAMATH_CALUDE_marbles_problem_l4005_400584

/-- Represents the number of marbles left in a box after removing some marbles. -/
def marblesLeft (total white : ℕ) : ℕ :=
  let red := (total - white) / 2
  let blue := (total - white) / 2
  let removed := 2 * (white - blue)
  total - removed

/-- Theorem stating that given the conditions of the problem, 40 marbles are left. -/
theorem marbles_problem : marblesLeft 50 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_marbles_problem_l4005_400584


namespace NUMINAMATH_CALUDE_square_sum_equals_21_l4005_400591

theorem square_sum_equals_21 (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -6) :
  x^2 + y^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_21_l4005_400591


namespace NUMINAMATH_CALUDE_log_2_3_in_terms_of_a_b_l4005_400557

theorem log_2_3_in_terms_of_a_b (a b : ℝ) (ha : a = Real.log 6) (hb : b = Real.log 20) :
  Real.log 3 / Real.log 2 = (a - b + 1) / (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_log_2_3_in_terms_of_a_b_l4005_400557


namespace NUMINAMATH_CALUDE_smallest_possible_value_l4005_400570

theorem smallest_possible_value (x : ℕ+) (a b : ℕ+) : 
  (Nat.gcd a b = x + 7) →
  (Nat.lcm a b = x * (x + 7)) →
  (a = 56) →
  (∀ y : ℕ+, y < x → ¬(∃ c : ℕ+, (Nat.gcd 56 c = y + 7) ∧ (Nat.lcm 56 c = y * (y + 7)))) →
  b = 294 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_value_l4005_400570


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l4005_400550

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 12*x^3 + 20*x^2 - 19*x - 24

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 5) * q x + 1012 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l4005_400550


namespace NUMINAMATH_CALUDE_prime_cube_sum_of_squares_l4005_400508

theorem prime_cube_sum_of_squares (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p^3 = p^2 + q^2 + r^2 → 
  p = 3 ∧ q = 3 ∧ r = 3 := by
sorry

end NUMINAMATH_CALUDE_prime_cube_sum_of_squares_l4005_400508


namespace NUMINAMATH_CALUDE_tim_stored_26_bales_l4005_400547

/-- The number of bales Tim stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Proof that Tim stored 26 bales in the barn -/
theorem tim_stored_26_bales : bales_stored 28 54 = 26 := by
  sorry

end NUMINAMATH_CALUDE_tim_stored_26_bales_l4005_400547


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l4005_400577

theorem cos_squared_minus_sin_squared_15_deg :
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l4005_400577


namespace NUMINAMATH_CALUDE_rationalize_denominator_l4005_400563

theorem rationalize_denominator : 
  1 / (Real.sqrt 3 - 2) = -(Real.sqrt 3) - 2 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l4005_400563


namespace NUMINAMATH_CALUDE_gina_netflix_minutes_l4005_400593

/-- The number of shows Gina's sister watches per week -/
def sister_shows : ℕ := 24

/-- The length of each show in minutes -/
def show_length : ℕ := 50

/-- The ratio of shows Gina chooses compared to her sister -/
def gina_ratio : ℕ := 3

/-- The total number of shows watched by both Gina and her sister -/
def total_shows : ℕ := sister_shows * (gina_ratio + 1)

/-- The number of shows Gina chooses -/
def gina_shows : ℕ := total_shows * gina_ratio / (gina_ratio + 1)

theorem gina_netflix_minutes : gina_shows * show_length = 900 :=
by sorry

end NUMINAMATH_CALUDE_gina_netflix_minutes_l4005_400593


namespace NUMINAMATH_CALUDE_sector_max_area_l4005_400575

/-- A sector is defined by its radius and central angle. -/
structure Sector where
  radius : ℝ
  angle : ℝ

/-- The perimeter of a sector. -/
def perimeter (s : Sector) : ℝ := s.radius * s.angle + 2 * s.radius

/-- The area of a sector. -/
def area (s : Sector) : ℝ := 0.5 * s.radius^2 * s.angle

/-- Theorem: For a sector with perimeter 4, the area is maximized when the central angle is 2 radians. -/
theorem sector_max_area (s : Sector) (h : perimeter s = 4) :
  area s ≤ area { radius := 1, angle := 2 } := by
  sorry

#check sector_max_area

end NUMINAMATH_CALUDE_sector_max_area_l4005_400575


namespace NUMINAMATH_CALUDE_cylinder_radius_l4005_400553

/-- 
Theorem: For a cylinder with an original height of 3 inches, 
if increasing either the radius or the height by 7 inches results in the same volume, 
then the original radius must be 7 inches.
-/
theorem cylinder_radius (r : ℝ) : 
  r > 0 →  -- radius must be positive
  3 * π * (r + 7)^2 = 10 * π * r^2 → -- volumes are equal
  r = 7 := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_l4005_400553


namespace NUMINAMATH_CALUDE_venkis_trip_speed_l4005_400578

/-- Venki's trip between towns X, Y, and Z -/
def venkis_trip (speed_xz speed_zy : ℝ) (time_xz time_zy : ℝ) : Prop :=
  let distance_xz := speed_xz * time_xz
  let distance_zy := distance_xz / 2
  speed_zy = distance_zy / time_zy

/-- The theorem statement for Venki's trip -/
theorem venkis_trip_speed :
  ∃ (speed_zy : ℝ),
    venkis_trip 80 speed_zy 5 (4 + 4/9) ∧
    abs (speed_zy - 42.86) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_venkis_trip_speed_l4005_400578


namespace NUMINAMATH_CALUDE_davids_biology_marks_l4005_400530

/-- Given David's marks in four subjects and his average across five subjects,
    prove that his marks in the fifth subject (Biology) must be 65. -/
theorem davids_biology_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (average : ℚ)
  (h1 : english = 70)
  (h2 : mathematics = 63)
  (h3 : physics = 80)
  (h4 : chemistry = 63)
  (h5 : average = 68.2)
  (h6 : (english + mathematics + physics + chemistry + biology : ℚ) / 5 = average) :
  biology = 65 := by
  sorry


end NUMINAMATH_CALUDE_davids_biology_marks_l4005_400530


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4005_400571

/-- A geometric sequence with its sum sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = q * a n

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (seq : GeometricSequence) : ℝ :=
  Classical.choose (seq.is_geometric 1 (by norm_num))

theorem geometric_sequence_ratio 
  (seq : GeometricSequence) 
  (h1 : seq.a 5 = 2 * seq.S 4 + 3)
  (h2 : seq.a 6 = 2 * seq.S 5 + 3) :
  common_ratio seq = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4005_400571


namespace NUMINAMATH_CALUDE_preimage_of_two_three_l4005_400552

/-- Given a mapping f : ℝ × ℝ → ℝ × ℝ defined by f(x, y) = (x+y, x-y),
    prove that f(5/2, -1/2) = (2, 3) -/
theorem preimage_of_two_three (f : ℝ × ℝ → ℝ × ℝ) 
    (h : ∀ x y : ℝ, f (x, y) = (x + y, x - y)) : 
    f (5/2, -1/2) = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_two_three_l4005_400552


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l4005_400576

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l4005_400576


namespace NUMINAMATH_CALUDE_kola_solution_water_percentage_l4005_400581

/-- Proves that the initial water percentage in a kola solution was 64% -/
theorem kola_solution_water_percentage :
  let initial_volume : ℝ := 340
  let initial_kola_percentage : ℝ := 9
  let added_sugar : ℝ := 3.2
  let added_water : ℝ := 8
  let added_kola : ℝ := 6.8
  let final_sugar_percentage : ℝ := 26.536312849162012
  let initial_water_percentage : ℝ := 64
  let initial_sugar_percentage : ℝ := 91 - initial_water_percentage - initial_kola_percentage
  let final_volume : ℝ := initial_volume + added_sugar + added_water + added_kola
  let final_sugar_volume : ℝ := (initial_sugar_percentage / 100) * initial_volume + added_sugar
  final_sugar_volume / final_volume * 100 = final_sugar_percentage :=
by
  sorry


end NUMINAMATH_CALUDE_kola_solution_water_percentage_l4005_400581


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l4005_400506

theorem sufficient_condition_for_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : a^2 + b^2 < 1) : 
  a * b + 1 > a + b := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l4005_400506


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l4005_400527

/-- Calculates the number of additional boxes needed to store chocolates --/
def additional_boxes_needed (total_chocolates : ℕ) (chocolates_not_in_box : ℕ) (existing_boxes : ℕ) (friend_chocolates : ℕ) : ℕ :=
  let chocolates_in_boxes := total_chocolates - chocolates_not_in_box
  let total_chocolates_to_box := chocolates_in_boxes + friend_chocolates
  let chocolates_per_box := chocolates_in_boxes / existing_boxes
  let total_boxes_needed := (total_chocolates_to_box + chocolates_per_box - 1) / chocolates_per_box
  total_boxes_needed - existing_boxes

theorem chocolate_box_problem :
  additional_boxes_needed 50 5 3 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l4005_400527


namespace NUMINAMATH_CALUDE_segment_length_unit_circle_l4005_400554

/-- The length of the segment cut by a unit circle from the line y - x = 1 is √2. -/
theorem segment_length_unit_circle : ∃ (L : ℝ), L = Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), x^2 + y^2 = 1 ∧ y - x = 1 → 
  ∃ (x' y' : ℝ), x'^2 + y'^2 = 1 ∧ y' - x' = 1 ∧ 
  Real.sqrt ((x - x')^2 + (y - y')^2) = L :=
sorry

end NUMINAMATH_CALUDE_segment_length_unit_circle_l4005_400554


namespace NUMINAMATH_CALUDE_exactly_four_separators_l4005_400542

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of five points in a plane -/
def FivePointSet := Fin 5 → Point

/-- A circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if three points are collinear -/
def are_collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if four points are concyclic -/
def are_concyclic (p q r s : Point) : Prop := sorry

/-- Predicate to check if a point is inside a circle -/
def is_inside (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a point is on a circle -/
def is_on_circle (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a circle is a separator for a set of five points -/
def is_separator (c : Circle) (s : FivePointSet) : Prop :=
  ∃ (i j k l m : Fin 5),
    i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ m ≠ i ∧ m ≠ j ∧ m ≠ k ∧ m ≠ l ∧
    is_on_circle (s i) c ∧ is_on_circle (s j) c ∧ is_on_circle (s k) c ∧
    is_inside (s l) c ∧ is_outside (s m) c

/-- The main theorem -/
theorem exactly_four_separators (s : FivePointSet) :
  (∀ (i j k : Fin 5), i ≠ j → j ≠ k → k ≠ i → ¬are_collinear (s i) (s j) (s k)) →
  (∀ (i j k l : Fin 5), i ≠ j → j ≠ k → k ≠ l → l ≠ i → ¬are_concyclic (s i) (s j) (s k) (s l)) →
  ∃! (separators : Finset Circle), (∀ c ∈ separators, is_separator c s) ∧ separators.card = 4 :=
sorry

end NUMINAMATH_CALUDE_exactly_four_separators_l4005_400542


namespace NUMINAMATH_CALUDE_value_of_a_l4005_400514

theorem value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 16 - 6 * a) : a = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l4005_400514


namespace NUMINAMATH_CALUDE_exists_field_trip_with_frequent_participants_l4005_400565

/-- Represents a field trip -/
structure FieldTrip where
  participants : Finset (Fin 20)
  at_least_four : participants.card ≥ 4

/-- Represents the collection of all field trips -/
structure FieldTrips where
  trips : Finset FieldTrip
  nonempty : trips.Nonempty

theorem exists_field_trip_with_frequent_participants (ft : FieldTrips) :
  ∃ (trip : FieldTrip), trip ∈ ft.trips ∧
    ∀ (student : Fin 20), student ∈ trip.participants →
      (ft.trips.filter (λ t : FieldTrip => student ∈ t.participants)).card ≥ ft.trips.card / 17 :=
sorry

end NUMINAMATH_CALUDE_exists_field_trip_with_frequent_participants_l4005_400565


namespace NUMINAMATH_CALUDE_probability_drawn_first_in_simple_random_sampling_l4005_400531

/-- Simple random sampling without replacement -/
def SimpleRandomSampling (populationSize sampleSize : ℕ) : Prop :=
  populationSize > 0 ∧ sampleSize ≤ populationSize

/-- Probability of being drawn first in a simple random sampling -/
def ProbabilityDrawnFirst (populationSize : ℕ) : ℚ :=
  1 / populationSize

theorem probability_drawn_first_in_simple_random_sampling
  (populationSize sampleSize : ℕ) (h : SimpleRandomSampling populationSize sampleSize) :
  populationSize = 10 → sampleSize = 3 → ProbabilityDrawnFirst populationSize = 1/10 := by
  sorry

#check probability_drawn_first_in_simple_random_sampling

end NUMINAMATH_CALUDE_probability_drawn_first_in_simple_random_sampling_l4005_400531


namespace NUMINAMATH_CALUDE_sequence_terms_equal_twenty_l4005_400541

def a (n : ℕ) : ℤ := n^2 - 14*n + 65

theorem sequence_terms_equal_twenty :
  (∀ n : ℕ, a n = 20 ↔ n = 5 ∨ n = 9) :=
sorry

end NUMINAMATH_CALUDE_sequence_terms_equal_twenty_l4005_400541


namespace NUMINAMATH_CALUDE_rectangle_problem_l4005_400523

/-- Given three rectangles with equal areas and integer sides, where one side is 31,
    the length of a side perpendicular to the side of length 31 is 992. -/
theorem rectangle_problem (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  (a * 31 = b * (992 : ℕ)) ∧ (∃ k l : ℕ, k * l = 31 * (k + l) ∧ k = 992) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_problem_l4005_400523


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l4005_400564

theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  b = 2 * a →
  c = 3 * a →
  c = 90 :=
sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l4005_400564


namespace NUMINAMATH_CALUDE_grid_line_count_l4005_400587

/-- Represents a point in the grid -/
structure Point where
  x : Fin 50
  y : Fin 50

/-- Represents the color of a point -/
inductive Color
  | Blue
  | Red

/-- Represents the color of a line segment -/
inductive LineColor
  | Blue
  | Red
  | Black

/-- The coloring of the grid -/
def grid_coloring : Point → Color := sorry

/-- The number of blue points in the grid -/
def num_blue_points : Nat := 1510

/-- The number of blue points on the edge of the grid -/
def num_blue_edge_points : Nat := 110

/-- The number of red line segments in the grid -/
def num_red_lines : Nat := 947

/-- Checks if a point is on the edge of the grid -/
def is_edge_point (p : Point) : Bool := 
  p.x = 0 || p.x = 49 || p.y = 0 || p.y = 49

/-- Checks if a point is at a corner of the grid -/
def is_corner_point (p : Point) : Bool :=
  (p.x = 0 && p.y = 0) || (p.x = 0 && p.y = 49) || 
  (p.x = 49 && p.y = 0) || (p.x = 49 && p.y = 49)

/-- The main theorem to prove -/
theorem grid_line_count : 
  (∀ p : Point, is_corner_point p → grid_coloring p = Color.Red) →
  (∃ edge_blue_points : Finset Point, 
    edge_blue_points.card = num_blue_edge_points ∧
    ∀ p ∈ edge_blue_points, is_edge_point p ∧ grid_coloring p = Color.Blue) →
  (∃ black_lines blue_lines : Nat, 
    black_lines = 1972 ∧ 
    blue_lines = 1981 ∧
    black_lines + blue_lines + num_red_lines = 50 * 49 * 2) :=
by sorry

end NUMINAMATH_CALUDE_grid_line_count_l4005_400587


namespace NUMINAMATH_CALUDE_cubic_expression_value_l4005_400562

theorem cubic_expression_value (m : ℝ) (h : m^2 - m - 1 = 0) :
  2 * m^3 - 3 * m^2 - m + 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l4005_400562


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l4005_400532

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - x₁ - 6 = 0) → (x₂^2 - x₂ - 6 = 0) → x₁ * x₂ = -6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l4005_400532


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l4005_400536

theorem smallest_solution_floor_equation :
  (∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 12) ∧
  (∀ (y : ℝ), y > 0 ∧ ⌊y^2⌋ - y * ⌊y⌋ = 12 → y ≥ 169/13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l4005_400536


namespace NUMINAMATH_CALUDE_sons_age_next_year_l4005_400580

/-- Given a father who is 35 years old and whose age is five times that of his son,
    prove that the son's age next year will be 8 years. -/
theorem sons_age_next_year (father_age : ℕ) (son_age : ℕ) : 
  father_age = 35 → father_age = 5 * son_age → son_age + 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_next_year_l4005_400580


namespace NUMINAMATH_CALUDE_upstairs_vacuuming_time_l4005_400511

/-- Represents the vacuuming problem with given conditions -/
def VacuumingProblem (downstairs upstairs total : ℕ) : Prop :=
  upstairs = 2 * downstairs + 5 ∧ 
  downstairs + upstairs = total ∧
  total = 38

/-- Proves that given the conditions, the time to vacuum upstairs is 27 minutes -/
theorem upstairs_vacuuming_time :
  ∀ downstairs upstairs total, 
  VacuumingProblem downstairs upstairs total → 
  upstairs = 27 := by
  sorry

end NUMINAMATH_CALUDE_upstairs_vacuuming_time_l4005_400511


namespace NUMINAMATH_CALUDE_milena_grandfather_age_difference_l4005_400595

/-- Calculates the age difference between a child and their grandfather given the child's age,
    the ratio of grandmother's age to child's age, and the age difference between grandparents. -/
def age_difference_child_grandfather (child_age : ℕ) (grandmother_ratio : ℕ) (grandparents_diff : ℕ) : ℕ :=
  (child_age * grandmother_ratio + grandparents_diff) - child_age

/-- The age difference between Milena and her grandfather is 58 years. -/
theorem milena_grandfather_age_difference :
  age_difference_child_grandfather 7 9 2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_milena_grandfather_age_difference_l4005_400595


namespace NUMINAMATH_CALUDE_largest_of_four_consecutive_odd_integers_l4005_400519

theorem largest_of_four_consecutive_odd_integers (a b c d : ℤ) : 
  (∀ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5 ∧ d = 2*n + 7) → 
  (a + b + c + d = 200) → 
  d = 53 := by
sorry

end NUMINAMATH_CALUDE_largest_of_four_consecutive_odd_integers_l4005_400519


namespace NUMINAMATH_CALUDE_adjacent_sum_of_six_l4005_400510

/-- Represents a 3x3 table filled with numbers 1 to 9 --/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Returns the list of adjacent positions for a given position --/
def adjacent_positions (row col : Fin 3) : List (Fin 3 × Fin 3) := sorry

/-- Returns the sum of adjacent numbers for a given number in the table --/
def adjacent_sum (t : Table) (n : Fin 9) : ℕ := sorry

/-- Checks if the table satisfies the given conditions --/
def valid_table (t : Table) : Prop :=
  (t 0 0 = 0) ∧ (t 2 0 = 1) ∧ (t 0 2 = 2) ∧ (t 2 2 = 3) ∧
  (adjacent_sum t 4 = 9) ∧
  (∀ i j : Fin 3, ∀ k : Fin 9, (t i j = k) → (∀ i' j' : Fin 3, (i ≠ i' ∨ j ≠ j') → t i' j' ≠ k))

theorem adjacent_sum_of_six (t : Table) (h : valid_table t) : adjacent_sum t 5 = 29 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_sum_of_six_l4005_400510


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l4005_400512

/-- Theorem: Area of a rectangle with length-to-width ratio 3:2 and diagonal d --/
theorem rectangle_area_diagonal (length width diagonal : ℝ) 
  (h_ratio : length / width = 3 / 2)
  (h_diagonal : length^2 + width^2 = diagonal^2) :
  length * width = (6/13) * diagonal^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l4005_400512


namespace NUMINAMATH_CALUDE_binary_search_upper_bound_l4005_400504

theorem binary_search_upper_bound (n : ℕ) (h : n ≤ 100) :
  ∃ (k : ℕ), k ≤ 7 ∧ 2^k > n :=
sorry

end NUMINAMATH_CALUDE_binary_search_upper_bound_l4005_400504


namespace NUMINAMATH_CALUDE_chord_length_theorem_main_theorem_l4005_400526

/-- The chord length formula for a circle and a line --/
def chord_length_formula (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    ((x - a)^2 + y^2 = 4) ∧  -- Circle equation
    (x - y - 2 = 0) ∧       -- Line equation
    ((x - a)^2 + y^2 = 4)   -- Circle equation (repeated for clarity)

/-- The theorem stating the possible values of a --/
theorem chord_length_theorem : 
  ∀ a : ℝ, chord_length_formula a → (a = 0 ∨ a = 4) :=
by
  sorry

/-- The main theorem combining the formula and the result --/
theorem main_theorem : 
  ∃ a : ℝ, chord_length_formula a ∧ (a = 0 ∨ a = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_chord_length_theorem_main_theorem_l4005_400526


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l4005_400524

/-- A square surrounded by four identical regular triangles -/
structure SquareWithTriangles where
  /-- Side length of the square -/
  squareSide : ℝ
  /-- The square has side length 2 -/
  squareSideIs2 : squareSide = 2
  /-- Side length of the surrounding triangles that touches the square -/
  triangleSide : ℝ
  /-- The triangle side that touches the square is equal to the square side -/
  triangleSideEqSquareSide : triangleSide = squareSide
  /-- The surrounding triangles are regular -/
  trianglesAreRegular : True
  /-- The surrounding triangles are symmetrically placed -/
  trianglesAreSymmetric : True

/-- Triangle ABC formed by connecting midpoints of outer sides of surrounding triangles -/
def TriangleABC (swt : SquareWithTriangles) : Set (ℝ × ℝ) := sorry

/-- The area of Triangle ABC -/
def areaOfTriangleABC (swt : SquareWithTriangles) : ℝ := sorry

/-- Theorem stating that the area of Triangle ABC is √3/2 -/
theorem area_of_triangle_ABC (swt : SquareWithTriangles) : 
  areaOfTriangleABC swt = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l4005_400524


namespace NUMINAMATH_CALUDE_sqrt_less_than_linear_approx_l4005_400558

theorem sqrt_less_than_linear_approx (x : ℝ) (hx : x > 0) : 
  Real.sqrt (1 + x) < 1 + x / 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_less_than_linear_approx_l4005_400558


namespace NUMINAMATH_CALUDE_accidents_occur_in_four_minutes_l4005_400549

/-- Represents the time between car collisions in seconds -/
def car_collision_interval : ℕ := 10

/-- Represents the time between big crashes in seconds -/
def big_crash_interval : ℕ := 20

/-- Represents the total number of accidents -/
def total_accidents : ℕ := 36

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Theorem stating that given the conditions, the accidents occur over 4 minutes -/
theorem accidents_occur_in_four_minutes :
  let collisions_per_minute := seconds_per_minute / car_collision_interval
  let crashes_per_minute := seconds_per_minute / big_crash_interval
  let accidents_per_minute := collisions_per_minute + crashes_per_minute
  total_accidents / accidents_per_minute = 4 := by
sorry

end NUMINAMATH_CALUDE_accidents_occur_in_four_minutes_l4005_400549


namespace NUMINAMATH_CALUDE_small_s_conference_teams_l4005_400592

-- Define the number of games in the tournament
def num_games : ℕ := 36

-- Define the function to calculate the number of games for n teams
def games_for_teams (n : ℕ) : ℕ := n * (n - 1) / 2

-- Theorem statement
theorem small_s_conference_teams :
  ∃ (n : ℕ), n > 0 ∧ games_for_teams n = num_games ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_small_s_conference_teams_l4005_400592


namespace NUMINAMATH_CALUDE_inverse_proportion_point_relation_l4005_400566

/-- Given two points A(x₁, 2) and B(x₂, 4) on the graph of y = k/x where k > 0,
    prove that x₁ > x₂ > 0 -/
theorem inverse_proportion_point_relation (k x₁ x₂ : ℝ) 
  (h_k : k > 0)
  (h_A : 2 = k / x₁)
  (h_B : 4 = k / x₂) :
  x₁ > x₂ ∧ x₂ > 0 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_relation_l4005_400566


namespace NUMINAMATH_CALUDE_circle_equation_tangent_to_x_axis_l4005_400535

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is on a circle --/
def Circle.isOn (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a circle is tangent to the x-axis --/
def Circle.tangentToXAxis (c : Circle) : Prop :=
  c.center.2 = c.radius

theorem circle_equation_tangent_to_x_axis (x y : ℝ) :
  (x - 5)^2 + (y - 4)^2 = 16 ↔
  ∃ (c : Circle), c.center = (5, 4) ∧ c.radius = 4 ∧
  c.isOn (x, y) ∧ c.tangentToXAxis :=
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_to_x_axis_l4005_400535


namespace NUMINAMATH_CALUDE_solve_equation_l4005_400596

theorem solve_equation : ∃ x : ℚ, (3 * x - 4) / 7 = 15 ∧ x = 109 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4005_400596


namespace NUMINAMATH_CALUDE_prob_odd_sum_coin_dice_prob_odd_sum_coin_dice_is_seven_sixteenths_l4005_400503

def coin_toss : Type := Bool
def die_roll : Type := Fin 6

def is_head (c : coin_toss) : Prop := c = true
def is_tail (c : coin_toss) : Prop := c = false

def sum_is_odd (rolls : List ℕ) : Prop := (rolls.sum % 2 = 1)

def prob_head : ℚ := 1/2
def prob_tail : ℚ := 1/2

def prob_odd_sum_two_dice : ℚ := 1/2

theorem prob_odd_sum_coin_dice : ℚ :=
  let p_0_head := prob_tail^3
  let p_1_head := 3 * prob_head * prob_tail^2
  let p_2_head := 3 * prob_head^2 * prob_tail
  let p_3_head := prob_head^3

  let p_odd_0_dice := 0
  let p_odd_2_dice := prob_odd_sum_two_dice
  let p_odd_4_dice := 1/2
  let p_odd_6_dice := 1/2

  p_0_head * p_odd_0_dice +
  p_1_head * p_odd_2_dice +
  p_2_head * p_odd_4_dice +
  p_3_head * p_odd_6_dice

theorem prob_odd_sum_coin_dice_is_seven_sixteenths :
  prob_odd_sum_coin_dice = 7/16 := by sorry

end NUMINAMATH_CALUDE_prob_odd_sum_coin_dice_prob_odd_sum_coin_dice_is_seven_sixteenths_l4005_400503


namespace NUMINAMATH_CALUDE_claire_crafting_time_l4005_400501

/-- Represents Claire's daily schedule --/
structure ClairesSchedule where
  clean : ℝ
  cook : ℝ
  errands : ℝ
  craft : ℝ
  tailor : ℝ

/-- Conditions for Claire's schedule --/
def validSchedule (s : ClairesSchedule) : Prop :=
  s.clean = 2 * s.cook ∧
  s.errands = s.cook - 1 ∧
  s.craft = s.tailor ∧
  s.clean + s.cook + s.errands + s.craft + s.tailor = 16 ∧
  s.craft + s.tailor = 9

/-- Theorem stating that in a valid schedule, Claire spends 4.5 hours crafting --/
theorem claire_crafting_time (s : ClairesSchedule) (h : validSchedule s) : s.craft = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_claire_crafting_time_l4005_400501


namespace NUMINAMATH_CALUDE_divisible_by_six_l4005_400559

theorem divisible_by_six (a : ℤ) : ∃ k : ℤ, a^3 + 11*a = 6*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l4005_400559


namespace NUMINAMATH_CALUDE_at_most_one_perfect_square_l4005_400579

theorem at_most_one_perfect_square (a : ℕ → ℤ) 
  (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) : 
  ∃! k, ∃ m : ℤ, a k = m ^ 2 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_perfect_square_l4005_400579


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4005_400546

theorem sufficient_but_not_necessary
  (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1)
  (f : ℝ → ℝ) (hf : f = λ x => a^x)
  (g : ℝ → ℝ) (hg : g = λ x => (2-a)*x^3) :
  (∀ x y : ℝ, x < y → f x > f y) →
  (∀ x y : ℝ, x < y → g x < g y) ∧
  ¬(∀ x y : ℝ, x < y → g x < g y → ∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4005_400546


namespace NUMINAMATH_CALUDE_two_pair_probability_l4005_400528

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of ranks in a standard deck -/
def NumRanks : ℕ := 13

/-- Number of cards per rank -/
def CardsPerRank : ℕ := 4

/-- Number of cards in a poker hand -/
def HandSize : ℕ := 5

/-- Number of ways to choose 5 cards from 52 -/
def TotalOutcomes : ℕ := Nat.choose StandardDeck HandSize

/-- Number of ways to form a two pair -/
def TwoPairOutcomes : ℕ := NumRanks * (Nat.choose CardsPerRank 2) * (NumRanks - 1) * (Nat.choose CardsPerRank 2) * (NumRanks - 2) * CardsPerRank

/-- Probability of forming a two pair -/
def TwoPairProbability : ℚ := TwoPairOutcomes / TotalOutcomes

theorem two_pair_probability : TwoPairProbability = 108 / 1005 := by
  sorry

end NUMINAMATH_CALUDE_two_pair_probability_l4005_400528


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l4005_400543

theorem unique_solution_power_equation :
  ∃! (x y z t : ℕ+), 2^y.val + 2^z.val * 5^t.val - 5^x.val = 1 ∧
    x = 2 ∧ y = 4 ∧ z = 1 ∧ t = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l4005_400543


namespace NUMINAMATH_CALUDE_quadratic_function_value_l4005_400502

/-- A quadratic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - m * x + 5

/-- The derivative of f with respect to x -/
def f_deriv (m : ℝ) (x : ℝ) : ℝ := 4 * x - m

theorem quadratic_function_value (m : ℝ) :
  (∀ x ≥ -2, f_deriv m x ≥ 0) → f m 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l4005_400502


namespace NUMINAMATH_CALUDE_train_length_calculation_l4005_400585

-- Define the walking speed in meters per second
def walking_speed : ℝ := 1

-- Define the time taken for the train to pass Xiao Ming
def time_ming : ℝ := 22

-- Define the time taken for the train to pass Xiao Hong
def time_hong : ℝ := 24

-- Define the train's speed (to be solved)
def train_speed : ℝ := 23

-- Define the train's length (to be proved)
def train_length : ℝ := 528

-- Theorem statement
theorem train_length_calculation :
  train_length = time_ming * (train_speed + walking_speed) ∧
  train_length = time_hong * (train_speed - walking_speed) := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l4005_400585


namespace NUMINAMATH_CALUDE_tobias_change_l4005_400505

def shoe_cost : ℕ := 95
def saving_months : ℕ := 3
def monthly_allowance : ℕ := 5
def lawn_mowing_charge : ℕ := 15
def driveway_shoveling_charge : ℕ := 7
def lawns_mowed : ℕ := 4
def driveways_shoveled : ℕ := 5

def total_savings : ℕ := 
  saving_months * monthly_allowance + 
  lawns_mowed * lawn_mowing_charge + 
  driveways_shoveled * driveway_shoveling_charge

theorem tobias_change : total_savings - shoe_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_tobias_change_l4005_400505


namespace NUMINAMATH_CALUDE_negation_of_square_positive_l4005_400520

theorem negation_of_square_positive :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, ¬(x^2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_square_positive_l4005_400520


namespace NUMINAMATH_CALUDE_condition_type_1_condition_type_2_condition_type_3_l4005_400597

-- Statement 1
theorem condition_type_1 :
  (∀ x : ℝ, 0 < x ∧ x < 3 → |x - 1| < 2) ∧
  ¬(∀ x : ℝ, |x - 1| < 2 → 0 < x ∧ x < 3) := by sorry

-- Statement 2
theorem condition_type_2 :
  (∀ x : ℝ, x = 2 → (x - 2) * (x - 3) = 0) ∧
  ¬(∀ x : ℝ, (x - 2) * (x - 3) = 0 → x = 2) := by sorry

-- Statement 3
theorem condition_type_3 :
  ∀ (a b c : ℝ), c = 0 ↔ a * 0^2 + b * 0 + c = 0 := by sorry

end NUMINAMATH_CALUDE_condition_type_1_condition_type_2_condition_type_3_l4005_400597


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l4005_400538

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, m ≠ 0 ∧ n ≠ 0 →
  (m^2 + n) * (m + n^2) = (m - n)^3 ↔
  (m = -1 ∧ n = -1) ∨ (m = 8 ∧ n = -10) ∨ (m = 9 ∧ n = -6) ∨ (m = 9 ∧ n = -21) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l4005_400538


namespace NUMINAMATH_CALUDE_system_solution_l4005_400573

theorem system_solution :
  let solutions : List (ℤ × ℤ) := [(-5, -3), (-3, -5), (3, 5), (5, 3)]
  ∀ x y : ℤ, (x^2 - x*y + y^2 = 19 ∧ x^4 + x^2*y^2 + y^4 = 931) ↔ (x, y) ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4005_400573


namespace NUMINAMATH_CALUDE_total_savings_l4005_400545

/-- The total savings from buying discounted milk and cereal with promotions -/
theorem total_savings (M C : ℝ) : ℝ := by
  -- M: original price of a gallon of milk
  -- C: original price of a box of cereal
  -- Milk discount: 25%
  -- Cereal promotion: buy two, get one 50% off
  -- Buying 3 gallons of milk and 6 boxes of cereal

  /- Define the milk discount -/
  let milk_discount_percent : ℝ := 0.25

  /- Define the cereal promotion discount -/
  let cereal_promotion_discount : ℝ := 0.5

  /- Calculate the savings on milk -/
  let milk_savings : ℝ := 3 * M * milk_discount_percent

  /- Calculate the savings on cereal -/
  let cereal_savings : ℝ := 2 * C * cereal_promotion_discount

  /- Calculate the total savings -/
  let total_savings : ℝ := milk_savings + cereal_savings

  /- Prove that the total savings equals (0.75 * M) + C -/
  have : total_savings = (0.75 * M) + C := by sorry

  /- Return the total savings -/
  exact total_savings

end NUMINAMATH_CALUDE_total_savings_l4005_400545


namespace NUMINAMATH_CALUDE_mountain_valley_trail_length_l4005_400522

/-- Represents the length of the Mountain Valley Trail hike --/
def MountainValleyTrail : Type := { trail : ℕ // trail > 0 }

/-- Represents the daily hike distances --/
def DailyHikes : Type := Fin 5 → ℕ

theorem mountain_valley_trail_length 
  (hikes : DailyHikes) 
  (day1_2 : hikes 0 + hikes 1 = 30)
  (day2_4_avg : (hikes 1 + hikes 3) / 2 = 15)
  (day3_4_5 : hikes 2 + hikes 3 + hikes 4 = 45)
  (day1_3 : hikes 0 + hikes 2 = 33) :
  ∃ (trail : MountainValleyTrail), (hikes 0 + hikes 1 + hikes 2 + hikes 3 + hikes 4 : ℕ) = trail.val ∧ trail.val = 75 := by
  sorry

end NUMINAMATH_CALUDE_mountain_valley_trail_length_l4005_400522


namespace NUMINAMATH_CALUDE_hexagon_rounding_exists_l4005_400590

/-- Represents a hexagon with numbers on its vertices and sums on its sides. -/
structure Hexagon where
  -- Vertex numbers
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  a₄ : ℝ
  a₅ : ℝ
  a₆ : ℝ
  -- Side sums
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  s₄ : ℝ
  s₅ : ℝ
  s₆ : ℝ
  -- Ensure side sums match vertex sums
  h₁ : s₁ = a₁ + a₂
  h₂ : s₂ = a₂ + a₃
  h₃ : s₃ = a₃ + a₄
  h₄ : s₄ = a₄ + a₅
  h₅ : s₅ = a₅ + a₆
  h₆ : s₆ = a₆ + a₁

/-- Represents a rounding strategy for the hexagon. -/
structure RoundedHexagon where
  -- Rounded vertex numbers
  r₁ : ℤ
  r₂ : ℤ
  r₃ : ℤ
  r₄ : ℤ
  r₅ : ℤ
  r₆ : ℤ
  -- Rounded side sums
  t₁ : ℤ
  t₂ : ℤ
  t₃ : ℤ
  t₄ : ℤ
  t₅ : ℤ
  t₆ : ℤ

/-- Theorem: For any hexagon, there exists a rounding strategy that maintains the sum property. -/
theorem hexagon_rounding_exists (h : Hexagon) : 
  ∃ (r : RoundedHexagon), 
    (r.t₁ = r.r₁ + r.r₂) ∧
    (r.t₂ = r.r₂ + r.r₃) ∧
    (r.t₃ = r.r₃ + r.r₄) ∧
    (r.t₄ = r.r₄ + r.r₅) ∧
    (r.t₅ = r.r₅ + r.r₆) ∧
    (r.t₆ = r.r₆ + r.r₁) :=
  sorry

end NUMINAMATH_CALUDE_hexagon_rounding_exists_l4005_400590


namespace NUMINAMATH_CALUDE_marks_deposit_l4005_400568

theorem marks_deposit (mark_deposit : ℝ) (bryan_deposit : ℝ) : 
  bryan_deposit = 5 * mark_deposit - 40 →
  mark_deposit + bryan_deposit = 400 →
  mark_deposit = 400 / 6 := by
sorry

end NUMINAMATH_CALUDE_marks_deposit_l4005_400568


namespace NUMINAMATH_CALUDE_mr_martin_purchase_cost_l4005_400572

/-- The cost of Mrs. Martin's purchase -/
def mrs_martin_cost : ℝ := 12.75

/-- The number of coffee cups Mrs. Martin bought -/
def mrs_martin_coffee : ℕ := 3

/-- The number of bagels Mrs. Martin bought -/
def mrs_martin_bagels : ℕ := 2

/-- The cost of one bagel -/
def bagel_cost : ℝ := 1.5

/-- The number of coffee cups Mr. Martin bought -/
def mr_martin_coffee : ℕ := 2

/-- The number of bagels Mr. Martin bought -/
def mr_martin_bagels : ℕ := 5

/-- Theorem stating that Mr. Martin's purchase costs $14.00 -/
theorem mr_martin_purchase_cost : 
  ∃ (coffee_cost : ℝ), 
    mrs_martin_cost = mrs_martin_coffee * coffee_cost + mrs_martin_bagels * bagel_cost ∧
    mr_martin_coffee * coffee_cost + mr_martin_bagels * bagel_cost = 14 :=
by sorry

end NUMINAMATH_CALUDE_mr_martin_purchase_cost_l4005_400572


namespace NUMINAMATH_CALUDE_commute_time_difference_l4005_400515

/-- Given a set of 5 commute times with known average and variance, prove the absolute difference between two unknown times. -/
theorem commute_time_difference (x y : ℝ) : 
  (x + y + 10 + 11 + 9) / 5 = 10 →
  ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2 →
  |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_commute_time_difference_l4005_400515


namespace NUMINAMATH_CALUDE_parabola_rotation_l4005_400556

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Rotates a parabola by 180° around its vertex --/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := p.h, k := p.k }

theorem parabola_rotation (p : Parabola) (hp : p = { a := 2, h := 3, k := -2 }) :
  rotate180 p = { a := -2, h := 3, k := -2 } := by
  sorry

#check parabola_rotation

end NUMINAMATH_CALUDE_parabola_rotation_l4005_400556


namespace NUMINAMATH_CALUDE_wrapping_and_ribbons_fractions_l4005_400574

/-- Given a roll of wrapping paper, prove the fractions used for wrapping and ribbons on each present -/
theorem wrapping_and_ribbons_fractions
  (total_wrap : ℚ) -- Total fraction of roll used for wrapping
  (total_ribbon : ℚ) -- Total fraction of roll used for ribbons
  (num_presents : ℕ) -- Number of presents
  (h1 : total_wrap = 2/5) -- Condition: 2/5 of roll used for wrapping
  (h2 : total_ribbon = 1/5) -- Condition: 1/5 of roll used for ribbons
  (h3 : num_presents = 5) -- Condition: 5 presents
  : (total_wrap / num_presents = 2/25) ∧ (total_ribbon / num_presents = 1/25) := by
  sorry


end NUMINAMATH_CALUDE_wrapping_and_ribbons_fractions_l4005_400574


namespace NUMINAMATH_CALUDE_extreme_value_condition_negative_interval_condition_l4005_400560

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Part I
theorem extreme_value_condition (a b : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = 10 →
  a = 4 ∧ b = -11 :=
sorry

-- Part II
theorem negative_interval_condition (b : ℝ) :
  (∀ (x : ℝ), x ∈ Set.Icc 1 2 → f (-1) b x < 0) →
  b < -5/2 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_negative_interval_condition_l4005_400560


namespace NUMINAMATH_CALUDE_amanda_savings_l4005_400551

/-- The cost of a single lighter at the gas station in dollars -/
def gas_station_price : ℚ := 175 / 100

/-- The cost of a pack of 12 lighters online in dollars -/
def online_pack_price : ℚ := 5

/-- The number of lighters in each online pack -/
def lighters_per_pack : ℕ := 12

/-- The number of lighters Amanda wants to buy -/
def lighters_to_buy : ℕ := 24

/-- The savings Amanda would have by buying online instead of at the gas station -/
theorem amanda_savings : 
  (lighters_to_buy : ℚ) * gas_station_price - 
  (lighters_to_buy / lighters_per_pack : ℚ) * online_pack_price = 32 := by
  sorry

end NUMINAMATH_CALUDE_amanda_savings_l4005_400551


namespace NUMINAMATH_CALUDE_product_equals_32_l4005_400513

theorem product_equals_32 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_l4005_400513


namespace NUMINAMATH_CALUDE_tangent_line_equation_l4005_400537

/-- A point on a cubic curve with a specific tangent slope -/
structure TangentPoint where
  x : ℝ
  y : ℝ
  on_curve : y = x^3 - 10*x + 3
  in_second_quadrant : x < 0 ∧ y > 0
  tangent_slope : 3*x^2 - 10 = 2

/-- The equation of the tangent line -/
def tangent_line (p : TangentPoint) : ℝ → ℝ := λ x => 2*x + 19

theorem tangent_line_equation (p : TangentPoint) :
  tangent_line p p.x = p.y ∧
  (λ x => tangent_line p x - p.y) = (λ x => 2*(x - p.x)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l4005_400537


namespace NUMINAMATH_CALUDE_closest_multiple_of_15_to_2028_l4005_400517

def closest_multiple (n : ℕ) (m : ℕ) : ℕ :=
  m * ((n + m / 2) / m)

theorem closest_multiple_of_15_to_2028 :
  closest_multiple 2028 15 = 2025 :=
sorry

end NUMINAMATH_CALUDE_closest_multiple_of_15_to_2028_l4005_400517


namespace NUMINAMATH_CALUDE_square_area_increase_l4005_400500

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.15 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.3225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l4005_400500


namespace NUMINAMATH_CALUDE_student_a_test_questions_l4005_400569

/-- Represents the grading system and test results for Student A -/
structure TestResults where
  correct_responses : ℕ
  incorrect_responses : ℕ
  score : ℤ
  score_calculation : score = correct_responses - 2 * incorrect_responses

/-- The total number of questions on the test -/
def total_questions (t : TestResults) : ℕ :=
  t.correct_responses + t.incorrect_responses

/-- Theorem stating that the total number of questions on Student A's test is 100 -/
theorem student_a_test_questions :
  ∃ t : TestResults, t.correct_responses = 90 ∧ t.score = 70 ∧ total_questions t = 100 := by
  sorry


end NUMINAMATH_CALUDE_student_a_test_questions_l4005_400569


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l4005_400529

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.b + t.c - t.a) = 3 * t.b * t.c

def condition2 (t : Triangle) : Prop :=
  Real.sin t.A = 2 * Real.sin t.B * Real.cos t.C

-- Define what it means for a triangle to be equilateral
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c ∧ t.A = t.B ∧ t.B = t.C ∧ t.A = Real.pi / 3

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : is_equilateral t := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l4005_400529


namespace NUMINAMATH_CALUDE_chocolate_bar_distribution_l4005_400589

theorem chocolate_bar_distribution (total_bars : ℕ) (total_boxes : ℕ) (bars_per_box : ℕ) :
  total_bars = 640 →
  total_boxes = 20 →
  total_bars = total_boxes * bars_per_box →
  bars_per_box = 32 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_distribution_l4005_400589


namespace NUMINAMATH_CALUDE_gcd_5670_9800_l4005_400516

theorem gcd_5670_9800 : Nat.gcd 5670 9800 = 70 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5670_9800_l4005_400516


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4005_400599

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 6 + a 7 + a 8 = 20) →
  (a 1 + a 12 = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4005_400599


namespace NUMINAMATH_CALUDE_cube_roll_no_90_degree_rotation_l4005_400540

/-- Represents a cube on a plane -/
structure Cube where
  position : ℝ × ℝ × ℝ
  top_face : Fin 6
  orientation : ℕ

/-- Represents a sequence of cube rolls -/
def RollSequence := List (Fin 4)

/-- Applies a sequence of rolls to a cube -/
def apply_rolls (c : Cube) (rolls : RollSequence) : Cube :=
  sorry

/-- Checks if the cube is in its initial position -/
def is_initial_position (initial : Cube) (final : Cube) : Prop :=
  initial.position = final.position ∧ initial.top_face = final.top_face

/-- Theorem: A cube rolled back to its initial position cannot have its top face rotated by 90 degrees -/
theorem cube_roll_no_90_degree_rotation 
  (c : Cube) (rolls : RollSequence) : 
  let c' := apply_rolls c rolls
  is_initial_position c c' → c.orientation ≠ (c'.orientation + 1) % 4 :=
sorry

end NUMINAMATH_CALUDE_cube_roll_no_90_degree_rotation_l4005_400540


namespace NUMINAMATH_CALUDE_final_elevation_proof_l4005_400582

def elevation_problem (start_elevation : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  start_elevation - rate * time

theorem final_elevation_proof (start_elevation : ℝ) (rate : ℝ) (time : ℝ)
  (h1 : start_elevation = 400)
  (h2 : rate = 10)
  (h3 : time = 5) :
  elevation_problem start_elevation rate time = 350 := by
  sorry

end NUMINAMATH_CALUDE_final_elevation_proof_l4005_400582


namespace NUMINAMATH_CALUDE_f_properties_l4005_400594

noncomputable def f (x : ℝ) : ℝ := (1 / (2^x - 1) + 1/2) * x^3

theorem f_properties :
  (∀ x, x ≠ 0 → f x ≠ 0) ∧
  (∀ x, x ≠ 0 → f (-x) = f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4005_400594


namespace NUMINAMATH_CALUDE_equation_roots_existence_and_location_l4005_400561

theorem equation_roots_existence_and_location 
  {a b c : ℝ} (hab : a < b) (hbc : b < c) :
  ∃ x₁ x₂ : ℝ, 
    (1 / (x₁ - a) + 1 / (x₁ - b) + 1 / (x₁ - c) = 0) ∧
    (1 / (x₂ - a) + 1 / (x₂ - b) + 1 / (x₂ - c) = 0) ∧
    a < x₁ ∧ x₁ < b ∧ b < x₂ ∧ x₂ < c ∧
    ∀ x : ℝ, (1 / (x - a) + 1 / (x - b) + 1 / (x - c) = 0) → (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_existence_and_location_l4005_400561


namespace NUMINAMATH_CALUDE_complement_of_M_l4005_400548

def M : Set ℝ := {x | x ≥ 2}

theorem complement_of_M : 
  {x : ℝ | x < 2} = (Set.univ : Set ℝ) \ M := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l4005_400548


namespace NUMINAMATH_CALUDE_greatest_possible_average_speed_l4005_400567

/-- A number is a palindrome if it reads the same backward as forward -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The next palindrome after a given number -/
def nextPalindrome (n : ℕ) : ℕ := sorry

theorem greatest_possible_average_speed 
  (initial_reading : ℕ) 
  (drive_duration : ℝ) 
  (speed_limit : ℝ) 
  (h1 : isPalindrome initial_reading)
  (h2 : drive_duration = 2)
  (h3 : speed_limit = 65)
  (h4 : initial_reading = 12321) :
  let final_reading := nextPalindrome initial_reading
  let distance := final_reading - initial_reading
  let max_distance := drive_duration * speed_limit
  let average_speed := distance / drive_duration
  (distance ≤ max_distance ∧ isPalindrome final_reading) →
  average_speed ≤ 50 ∧ 
  ∃ (s : ℝ), s > 50 → 
    ¬∃ (d : ℕ), d > distance ∧ 
      d ≤ max_distance ∧ 
      isPalindrome (initial_reading + d) ∧
      s = d / drive_duration :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_average_speed_l4005_400567


namespace NUMINAMATH_CALUDE_john_sleep_week_total_l4005_400586

/-- The amount of sleep John got during a week with varying sleep patterns. -/
def johnSleepWeek (recommendedSleep : ℝ) : ℝ :=
  let mondayTuesday := 2 * 3
  let wednesday := 0.8 * recommendedSleep
  let thursdayFriday := 2 * (0.5 * recommendedSleep)
  let saturday := 0.7 * recommendedSleep + 2
  let sunday := 0.4 * recommendedSleep
  mondayTuesday + wednesday + thursdayFriday + saturday + sunday

/-- Theorem stating that John's total sleep for the week is 31.2 hours. -/
theorem john_sleep_week_total : johnSleepWeek 8 = 31.2 := by
  sorry

#eval johnSleepWeek 8

end NUMINAMATH_CALUDE_john_sleep_week_total_l4005_400586


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4005_400539

theorem sum_of_squares_of_roots (a b c : ℂ) : 
  (3 * a^3 - 2 * a^2 + 5 * a + 15 = 0) ∧ 
  (3 * b^3 - 2 * b^2 + 5 * b + 15 = 0) ∧ 
  (3 * c^3 - 2 * c^2 + 5 * c + 15 = 0) →
  a^2 + b^2 + c^2 = -26/9 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4005_400539


namespace NUMINAMATH_CALUDE_probability_intersection_independent_events_l4005_400534

theorem probability_intersection_independent_events 
  (p : Set α → ℝ) (a b : Set α) 
  (ha : p a = 4/5) 
  (hb : p b = 2/5) 
  (hab_indep : p (a ∩ b) = p a * p b) : 
  p (a ∩ b) = 8/25 := by
sorry

end NUMINAMATH_CALUDE_probability_intersection_independent_events_l4005_400534


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l4005_400598

/-- The quadratic equation kx^2 + 18x + 2k = 0 has rational solutions if and only if k = 4, where k is a positive integer. -/
theorem quadratic_rational_solutions (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 18 * x + 2 * k = 0) ↔ k = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l4005_400598


namespace NUMINAMATH_CALUDE_rogers_candy_problem_l4005_400555

/-- Roger's candy problem -/
theorem rogers_candy_problem (initial_candies given_candies remaining_candies : ℕ) :
  given_candies = 3 →
  remaining_candies = 92 →
  initial_candies = remaining_candies + given_candies →
  initial_candies = 95 :=
by sorry

end NUMINAMATH_CALUDE_rogers_candy_problem_l4005_400555


namespace NUMINAMATH_CALUDE_max_area_inscribed_quadrilateral_l4005_400509

/-- The maximum area of an inscribed quadrilateral within a circle -/
def max_area_circle (r : ℝ) : ℝ := 2 * r^2

/-- The equation of an ellipse -/
def is_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- The maximum area of an inscribed quadrilateral within an ellipse -/
def max_area_ellipse (a b : ℝ) : ℝ := 2 * a * b

theorem max_area_inscribed_quadrilateral 
  (r a b : ℝ) 
  (hr : r > 0) 
  (hab : a > b) 
  (hb : b > 0) : 
  max_area_ellipse a b = 2 * a * b :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_quadrilateral_l4005_400509


namespace NUMINAMATH_CALUDE_vector_minimization_and_angle_l4005_400507

/-- Given vectors OP, OA, OB, and a point C on line OP, prove that OC minimizes CA · CB and calculate cos ∠ACB -/
theorem vector_minimization_and_angle (O P A B C : ℝ × ℝ) : 
  O = (0, 0) →
  P = (2, 1) →
  A = (1, 7) →
  B = (5, 1) →
  (∃ t : ℝ, C = (t * 2, t * 1)) →
  (∀ D : ℝ × ℝ, (∃ s : ℝ, D = (s * 2, s * 1)) → 
    (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) ≤ 
    (A.1 - D.1) * (B.1 - D.1) + (A.2 - D.2) * (B.2 - D.2)) →
  C = (4, 2) ∧ 
  (((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2)) / 
   (((A.1 - C.1)^2 + (A.2 - C.2)^2) * ((B.1 - C.1)^2 + (B.2 - C.2)^2))^(1/2) = -4 * 17^(1/2) / 17) :=
by sorry


end NUMINAMATH_CALUDE_vector_minimization_and_angle_l4005_400507


namespace NUMINAMATH_CALUDE_unique_solution_system_l4005_400583

/-- The system of equations has a unique solution (67/9, 1254/171) -/
theorem unique_solution_system :
  ∃! (x y : ℚ), (3 * x - 4 * y = -7) ∧ (6 * x - 5 * y = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l4005_400583


namespace NUMINAMATH_CALUDE_divisors_of_600_l4005_400588

theorem divisors_of_600 : Nat.card {d : ℕ | d ∣ 600} = 24 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_600_l4005_400588
