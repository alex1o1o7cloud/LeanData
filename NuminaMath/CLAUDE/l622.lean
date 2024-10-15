import Mathlib

namespace NUMINAMATH_CALUDE_hockey_puck_price_comparison_l622_62212

theorem hockey_puck_price_comparison (P : ℝ) (h : P > 0) : P > 0.99 * P := by
  sorry

end NUMINAMATH_CALUDE_hockey_puck_price_comparison_l622_62212


namespace NUMINAMATH_CALUDE_ink_needed_per_whiteboard_l622_62234

-- Define the given conditions
def num_classes : ℕ := 5
def whiteboards_per_class : ℕ := 2
def ink_cost_per_ml : ℚ := 50 / 100  -- 50 cents = 0.5 dollars
def total_daily_cost : ℚ := 100

-- Define the function to calculate ink needed per whiteboard
def ink_per_whiteboard : ℚ :=
  let total_whiteboards : ℕ := num_classes * whiteboards_per_class
  let total_ink_ml : ℚ := total_daily_cost / ink_cost_per_ml
  total_ink_ml / total_whiteboards

-- Theorem to prove
theorem ink_needed_per_whiteboard : ink_per_whiteboard = 20 := by
  sorry

end NUMINAMATH_CALUDE_ink_needed_per_whiteboard_l622_62234


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l622_62265

-- Define the point P as the intersection of two lines
def P : ℝ × ℝ := (2, 1)

-- Define line l1
def l1 (x y : ℝ) : Prop := 4 * x - y + 1 = 0

-- Define the condition that a line passes through point P
def passes_through_P (a b c : ℝ) : Prop := a * P.1 + b * P.2 + c = 0

-- Theorem for case I
theorem parallel_line_equation :
  ∀ (a b c : ℝ), passes_through_P a b c → (∀ x y, a * x + b * y + c = 0 ↔ 4 * x - y - 7 = 0) → 
  ∃ k, a = 4 * k ∧ b = -k ∧ c = -7 * k := by sorry

-- Theorem for case II
theorem perpendicular_line_equation :
  ∀ (a b c : ℝ), passes_through_P a b c → (∀ x y, a * x + b * y + c = 0 ↔ x + 4 * y - 6 = 0) → 
  ∃ k, a = k ∧ b = 4 * k ∧ c = -6 * k := by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l622_62265


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l622_62282

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧
    radius = 2 ∧
    ∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l622_62282


namespace NUMINAMATH_CALUDE_cubic_function_properties_monotonic_cubic_function_range_l622_62283

/-- A cubic function with specified properties -/
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- Theorem for part I -/
theorem cubic_function_properties (a b c d : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b c d x = -f a b c d (-x)) →  -- Symmetry about origin
  (f a b c d (1/2) = -1) →                -- Minimum value at x = 1/2
  (f' a b c (1/2) = 0) →                  -- Critical point at x = 1/2
  (f a b c d = f 4 0 (-3) 0) :=
sorry

/-- Theorem for part II -/
theorem monotonic_cubic_function_range (c : ℝ) :
  (∀ x y, x < y → (f 1 1 c 1 x < f 1 1 c 1 y) ∨ (∀ x y, x < y → f 1 1 c 1 x > f 1 1 c 1 y)) →
  c ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_monotonic_cubic_function_range_l622_62283


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l622_62281

-- Define the sets A and B
def A : Set ℝ := {x | x < 3/2}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l622_62281


namespace NUMINAMATH_CALUDE_tan_sixty_degrees_l622_62266

theorem tan_sixty_degrees : Real.tan (60 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sixty_degrees_l622_62266


namespace NUMINAMATH_CALUDE_stones_placement_theorem_l622_62274

/-- Represents the state of the strip and bag -/
structure GameState where
  stones_in_bag : Nat
  stones_on_strip : List Nat
  deriving Repr

/-- Allowed operations in the game -/
inductive Move
  | PlaceInFirst : Move
  | RemoveFromFirst : Move
  | PlaceInNext (i : Nat) : Move
  | RemoveFromNext (i : Nat) : Move

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.PlaceInFirst => 
      { state with 
        stones_in_bag := state.stones_in_bag - 1,
        stones_on_strip := 1 :: state.stones_on_strip }
  | Move.RemoveFromFirst =>
      { state with 
        stones_in_bag := state.stones_in_bag + 1,
        stones_on_strip := state.stones_on_strip.tail }
  | Move.PlaceInNext i =>
      if i ∈ state.stones_on_strip then
        { state with 
          stones_in_bag := state.stones_in_bag - 1,
          stones_on_strip := (i + 1) :: state.stones_on_strip }
      else state
  | Move.RemoveFromNext i =>
      if i ∈ state.stones_on_strip ∧ (i + 1) ∈ state.stones_on_strip then
        { state with 
          stones_in_bag := state.stones_in_bag + 1,
          stones_on_strip := state.stones_on_strip.filter (· ≠ i + 1) }
      else state

/-- Checks if it's possible to reach a certain cell number -/
def canReachCell (n : Nat) : Prop :=
  ∃ (moves : List Move), 
    let finalState := moves.foldl applyMove { stones_in_bag := 10, stones_on_strip := [] }
    n ∈ finalState.stones_on_strip

theorem stones_placement_theorem : 
  ∀ n : Nat, n ≤ 1023 → canReachCell n :=
by sorry

end NUMINAMATH_CALUDE_stones_placement_theorem_l622_62274


namespace NUMINAMATH_CALUDE_teachers_survey_l622_62226

theorem teachers_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_high_bp : high_bp = 90)
  (h_heart_trouble : heart_trouble = 60)
  (h_both : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_teachers_survey_l622_62226


namespace NUMINAMATH_CALUDE_first_box_weight_l622_62242

theorem first_box_weight (total_weight second_weight third_weight : ℕ) 
  (h1 : total_weight = 18)
  (h2 : second_weight = 11)
  (h3 : third_weight = 5)
  : ∃ first_weight : ℕ, first_weight + second_weight + third_weight = total_weight ∧ first_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_box_weight_l622_62242


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l622_62254

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≤ 1}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l622_62254


namespace NUMINAMATH_CALUDE_orange_removal_theorem_l622_62220

/-- Represents the number of oranges Mary must put back to achieve the desired average price -/
def oranges_to_remove (apple_price orange_price : ℚ) (total_fruits : ℕ) (initial_avg_price desired_avg_price : ℚ) : ℚ :=
  (total_fruits * initial_avg_price - total_fruits * desired_avg_price) / (orange_price - desired_avg_price)

theorem orange_removal_theorem (apple_price orange_price : ℚ) (total_fruits : ℕ) (initial_avg_price desired_avg_price : ℚ) :
  apple_price = 40/100 ∧ 
  orange_price = 60/100 ∧ 
  total_fruits = 10 ∧ 
  initial_avg_price = 54/100 ∧ 
  desired_avg_price = 50/100 → 
  oranges_to_remove apple_price orange_price total_fruits initial_avg_price desired_avg_price = 4 := by
  sorry

#eval oranges_to_remove (40/100) (60/100) 10 (54/100) (50/100)

end NUMINAMATH_CALUDE_orange_removal_theorem_l622_62220


namespace NUMINAMATH_CALUDE_max_product_arithmetic_sequence_l622_62260

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem max_product_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a6 : a 6 = 4) :
  (∃ x : ℝ, a 4 * a 7 ≤ x) ∧ a 4 * a 7 ≤ 18 ∧ (∃ d : ℝ, a 4 * a 7 = 18) :=
sorry

end NUMINAMATH_CALUDE_max_product_arithmetic_sequence_l622_62260


namespace NUMINAMATH_CALUDE_ellipse_and_triangle_properties_l622_62245

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about the ellipse and triangle properties -/
theorem ellipse_and_triangle_properties
  (e : Ellipse)
  (focus : Point)
  (pass_through : Point)
  (p : Point)
  (h_focus : focus = ⟨2 * Real.sqrt 2, 0⟩)
  (h_pass : pass_through = ⟨3, 1⟩)
  (h_p : p = ⟨-3, 2⟩)
  (h_on_ellipse : pass_through.x^2 / e.a^2 + pass_through.y^2 / e.b^2 = 1)
  (h_focus_prop : e.a^2 - e.b^2 = 8)
  (h_intersect : ∃ (a b : Point), a ≠ b ∧
    a.x^2 / e.a^2 + a.y^2 / e.b^2 = 1 ∧
    b.x^2 / e.a^2 + b.y^2 / e.b^2 = 1 ∧
    a.y - b.y = a.x - b.x)
  (h_isosceles : ∃ (a b : Point), 
    (a.x - p.x)^2 + (a.y - p.y)^2 = (b.x - p.x)^2 + (b.y - p.y)^2) :
  e.a^2 = 12 ∧ e.b^2 = 4 ∧
  (∃ (a b : Point), 
    (1/2) * Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2) * 
    (3 / Real.sqrt 2) = 9/2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_triangle_properties_l622_62245


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l622_62213

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l622_62213


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l622_62264

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (c x y : ℝ),
    (x^2 / a^2 - y^2 / b^2 = 1) ∧  -- P is on the hyperbola
    (x = c) ∧  -- PF is perpendicular to x-axis
    ((c - b) / (c + b) = 1/3) →  -- ratio of distances to asymptotes
    c^2 / a^2 = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l622_62264


namespace NUMINAMATH_CALUDE_terminating_decimals_count_l622_62241

theorem terminating_decimals_count : 
  let n_count := Finset.filter (fun n => Nat.gcd n 420 % 3 = 0 ∧ Nat.gcd n 420 % 7 = 0) (Finset.range 419)
  Finset.card n_count = 19 := by
  sorry

end NUMINAMATH_CALUDE_terminating_decimals_count_l622_62241


namespace NUMINAMATH_CALUDE_equation_solution_l622_62275

theorem equation_solution : ∃ x : ℝ, (6000 - (105 / x) = 5995) ∧ x = 21 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l622_62275


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l622_62221

theorem mary_baseball_cards :
  ∀ (initial_cards torn_cards fred_cards bought_cards : ℕ),
    initial_cards = 18 →
    torn_cards = 8 →
    fred_cards = 26 →
    bought_cards = 40 →
    initial_cards - torn_cards + fred_cards + bought_cards = 76 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l622_62221


namespace NUMINAMATH_CALUDE_apple_box_weight_proof_l622_62261

/-- The number of apple boxes -/
def num_boxes : ℕ := 7

/-- The number of boxes whose initial weight equals the final weight of all boxes -/
def num_equal_boxes : ℕ := 3

/-- The amount of apples removed from each box (in kg) -/
def removed_weight : ℕ := 20

/-- The initial weight of apples in each box (in kg) -/
def initial_weight : ℕ := 35

theorem apple_box_weight_proof :
  initial_weight * num_boxes - removed_weight * num_boxes = initial_weight * num_equal_boxes :=
by sorry

end NUMINAMATH_CALUDE_apple_box_weight_proof_l622_62261


namespace NUMINAMATH_CALUDE_min_tetrahedron_volume_l622_62201

/-- Given a point P(1, 4, 5) in 3D Cartesian coordinate system O-xyz,
    and a plane passing through P intersecting positive axes at points A, B, and C,
    prove that the minimum volume V of tetrahedron O-ABC is 15. -/
theorem min_tetrahedron_volume (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_plane : 1 / a + 4 / b + 5 / c = 1) :
  (1 / 6 : ℝ) * a * b * c ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_tetrahedron_volume_l622_62201


namespace NUMINAMATH_CALUDE_line_intercept_form_l622_62299

/-- Given a line with equation 3x - 2y = 4, its intercept form is x/(4/3) + y/(-2) = 1 -/
theorem line_intercept_form :
  ∀ (x y : ℝ), 3*x - 2*y = 4 → x/(4/3) + y/(-2) = 1 := by sorry

end NUMINAMATH_CALUDE_line_intercept_form_l622_62299


namespace NUMINAMATH_CALUDE_jo_age_proof_l622_62258

theorem jo_age_proof (j d g : ℕ) : 
  (∃ (x y z : ℕ), j = 2 * x ∧ d = 2 * y ∧ g = 2 * z) →  -- ages are even
  j * d * g = 2024 →                                   -- product of ages is 2024
  j ≥ d ∧ j ≥ g →                                      -- Jo's age is the largest
  j = 46 :=                                            -- Jo's age is 46
by sorry

end NUMINAMATH_CALUDE_jo_age_proof_l622_62258


namespace NUMINAMATH_CALUDE_curve_and_line_properties_l622_62286

-- Define the unit circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the stretched curve C₂
def C₂ (x y : ℝ) : Prop := (x / Real.sqrt 3)^2 + (y / 2)^2 = 1

-- Define the line l
def l (x y : ℝ) : Prop := 2 * x - y - 6 = 0

-- Theorem statement
theorem curve_and_line_properties :
  -- 1. Parametric equations of C₂
  (∀ φ : ℝ, C₂ (Real.sqrt 3 * Real.cos φ) (2 * Real.sin φ)) ∧
  -- 2. Point P(-3/2, 1) on C₂ has maximum distance to l
  (C₂ (-3/2) 1 ∧
   ∀ x y : ℝ, C₂ x y →
     (x + 3/2)^2 + (y - 1)^2 ≤ (2 * Real.sqrt 5)^2) ∧
  -- 3. Maximum distance from C₂ to l is 2√5
  (∃ x y : ℝ, C₂ x y ∧
    |2*x - y - 6| / Real.sqrt 5 = 2 * Real.sqrt 5 ∧
    ∀ x' y' : ℝ, C₂ x' y' →
      |2*x' - y' - 6| / Real.sqrt 5 ≤ 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_curve_and_line_properties_l622_62286


namespace NUMINAMATH_CALUDE_rope_length_l622_62210

/-- Calculates the length of a rope in centimeters given specific conditions -/
theorem rope_length : 
  let total_pieces : ℕ := 154
  let equal_pieces : ℕ := 150
  let equal_piece_length : ℕ := 75  -- in millimeters
  let remaining_piece_length : ℕ := 100  -- in millimeters
  let total_length : ℕ := equal_pieces * equal_piece_length + 
                          (total_pieces - equal_pieces) * remaining_piece_length
  total_length / 10 = 1165  -- length in centimeters
  := by sorry

end NUMINAMATH_CALUDE_rope_length_l622_62210


namespace NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l622_62224

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_A_in_fourth_quadrant :
  is_in_fourth_quadrant 2 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l622_62224


namespace NUMINAMATH_CALUDE_unique_solution_for_difference_of_squares_l622_62297

theorem unique_solution_for_difference_of_squares : 
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = 204 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_difference_of_squares_l622_62297


namespace NUMINAMATH_CALUDE_correct_plates_removed_l622_62256

/-- The number of plates that need to be removed to reach the acceptable weight -/
def plates_to_remove : ℕ :=
  let initial_plates : ℕ := 38
  let plate_weight : ℕ := 10  -- in ounces
  let max_weight_lbs : ℕ := 20
  let max_weight_oz : ℕ := max_weight_lbs * 16
  let total_weight : ℕ := initial_plates * plate_weight
  let excess_weight : ℕ := total_weight - max_weight_oz
  excess_weight / plate_weight

theorem correct_plates_removed : plates_to_remove = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_plates_removed_l622_62256


namespace NUMINAMATH_CALUDE_same_color_probability_l622_62228

/-- The probability that two remaining chairs are of the same color -/
theorem same_color_probability 
  (black_chairs : ℕ) 
  (brown_chairs : ℕ) 
  (h1 : black_chairs = 15) 
  (h2 : brown_chairs = 18) :
  (black_chairs * (black_chairs - 1) + brown_chairs * (brown_chairs - 1)) / 
  ((black_chairs + brown_chairs) * (black_chairs + brown_chairs - 1)) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_same_color_probability_l622_62228


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l622_62251

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_a2 : a 2 = 4) :
  ∀ n : ℕ, a n = 2^n := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l622_62251


namespace NUMINAMATH_CALUDE_smallest_number_remainder_l622_62231

theorem smallest_number_remainder (N : ℕ) : 
  N = 184 → N % 13 = 2 → N % 15 = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_remainder_l622_62231


namespace NUMINAMATH_CALUDE_expression_simplification_l622_62250

theorem expression_simplification (a b : ℝ) 
  (h1 : b ≠ 0) (h2 : b ≠ -3*a) (h3 : b ≠ a) (h4 : b ≠ -a) :
  ((2*b + a - (4*a^2 - b^2)/a) / (b^3 + 2*a*b^2 - 3*a^2*b)) * 
  ((a^3*b - 2*a^2*b^2 + a*b^3) / (a^2 - b^2)) = (a - b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l622_62250


namespace NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l622_62287

-- Define the set M
def M : Set ℝ := {x : ℝ | |2*x - 1| < 1}

-- Theorem 1: Characterization of M
theorem characterization_of_M : M = {x : ℝ | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_for_M_elements (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  a * b + 1 > a + b := by sorry

end NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l622_62287


namespace NUMINAMATH_CALUDE_chris_dana_distance_difference_l622_62205

/-- The difference in distance traveled between two bikers after a given time -/
def distance_difference (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 * time) - (speed2 * time)

/-- Theorem stating the difference in distance traveled between Chris and Dana -/
theorem chris_dana_distance_difference :
  distance_difference 17 12 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_chris_dana_distance_difference_l622_62205


namespace NUMINAMATH_CALUDE_birds_beetles_per_day_l622_62206

-- Define the constants
def birds_per_snake : ℕ := 3
def snakes_per_jaguar : ℕ := 5
def num_jaguars : ℕ := 6
def total_beetles : ℕ := 1080

-- Define the theorem
theorem birds_beetles_per_day :
  ∀ (beetles_per_bird : ℕ),
    beetles_per_bird * (birds_per_snake * snakes_per_jaguar * num_jaguars) = total_beetles →
    beetles_per_bird = 12 := by
  sorry

end NUMINAMATH_CALUDE_birds_beetles_per_day_l622_62206


namespace NUMINAMATH_CALUDE_second_company_daily_rate_l622_62277

/-- Represents the daily rate and per-mile rate for a car rental company -/
structure RentalRate where
  daily : ℝ
  perMile : ℝ

/-- Calculates the total cost for a rental given the rate and miles driven -/
def totalCost (rate : RentalRate) (miles : ℝ) : ℝ :=
  rate.daily + rate.perMile * miles

theorem second_company_daily_rate :
  let sunshine := RentalRate.mk 17.99 0.18
  let other := RentalRate.mk x 0.16
  let miles := 48.0
  totalCost sunshine miles = totalCost other miles →
  x = 18.95 := by
  sorry

end NUMINAMATH_CALUDE_second_company_daily_rate_l622_62277


namespace NUMINAMATH_CALUDE_snail_max_distance_l622_62222

/-- Represents the movement of a snail over time -/
structure SnailMovement where
  /-- The total observation time in hours -/
  total_time : ℝ
  /-- The observation duration of each observer in hours -/
  observer_duration : ℝ
  /-- The distance traveled during each observation in meters -/
  distance_per_observation : ℝ
  /-- Ensures there is always at least one observer -/
  always_observed : Prop

/-- The maximum distance the snail can travel given the conditions -/
def max_distance (sm : SnailMovement) : ℝ :=
  18

/-- Theorem stating the maximum distance the snail can travel is 18 meters -/
theorem snail_max_distance (sm : SnailMovement) 
    (h1 : sm.total_time = 10)
    (h2 : sm.observer_duration = 1)
    (h3 : sm.distance_per_observation = 1)
    (h4 : sm.always_observed) : 
  max_distance sm = 18 := by
  sorry

end NUMINAMATH_CALUDE_snail_max_distance_l622_62222


namespace NUMINAMATH_CALUDE_marias_salary_l622_62217

theorem marias_salary (S : ℝ) : 
  (S * 0.2 + S * 0.05 + (S - S * 0.2 - S * 0.05) * 0.25 + 1125 = S) → S = 2000 := by
sorry

end NUMINAMATH_CALUDE_marias_salary_l622_62217


namespace NUMINAMATH_CALUDE_bryden_receives_20_dollars_l622_62278

/-- The amount a collector pays for state quarters as a percentage of face value -/
def collector_rate : ℚ := 2000

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 4

/-- The face value of a single state quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The amount Bryden will receive for his quarters in dollars -/
def bryden_receives : ℚ := (collector_rate / 100) * (bryden_quarters : ℚ) * quarter_value

theorem bryden_receives_20_dollars : bryden_receives = 20 := by
  sorry

end NUMINAMATH_CALUDE_bryden_receives_20_dollars_l622_62278


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l622_62269

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove the measure of angle A
    and the perimeter of the triangle under specific conditions. -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a * Real.sin B = -Real.sqrt 3 * b * Real.cos A →
  b = 4 →
  S = 2 * Real.sqrt 3 →
  S = (1/2) * b * c * Real.sin A →
  (A = (2/3) * Real.pi ∧ a + b + c = 6 + 2 * Real.sqrt 7) := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l622_62269


namespace NUMINAMATH_CALUDE_quadratic_solution_square_l622_62219

theorem quadratic_solution_square (x : ℝ) :
  7 * x^2 + 6 = 5 * x + 11 →
  (8 * x - 5)^2 = (2865 - 120 * Real.sqrt 165) / 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_square_l622_62219


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l622_62209

/-- The common ratio of the geometric series 7/8 - 35/72 + 175/432 - ... is -5/9 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -35/72
  let a₃ : ℚ := 175/432
  let r := a₂ / a₁
  r = -5/9 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l622_62209


namespace NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l622_62223

/-- Given a sinusoidal function y = a * sin(b * x + c) + d with positive constants a, b, c, and d,
    if the maximum value of y is 3 and the minimum value is -1, then d = 1. -/
theorem sinusoidal_vertical_shift 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * Real.sin (b * x + c) + d)
  (hmax : ∀ x, f x ≤ 3)
  (hmin : ∀ x, f x ≥ -1)
  (hex_max : ∃ x, f x = 3)
  (hex_min : ∃ x, f x = -1) :
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l622_62223


namespace NUMINAMATH_CALUDE_no_solution_equation1_solutions_equation2_l622_62238

-- Define the equations
def equation1 (x : ℝ) : Prop := 1 + (3 * x) / (x - 2) = 6 / (x - 2)
def equation2 (x : ℝ) : Prop := x^2 + x - 6 = 0

-- Theorem for the first equation
theorem no_solution_equation1 : ¬ ∃ x : ℝ, equation1 x := by sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  (equation2 (-3) ∧ equation2 2) ∧ 
  (∀ x : ℝ, equation2 x → (x = -3 ∨ x = 2)) := by sorry

end NUMINAMATH_CALUDE_no_solution_equation1_solutions_equation2_l622_62238


namespace NUMINAMATH_CALUDE_solution_is_one_l622_62235

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  (7 / (x^2 + x)) - (3 / (x - x^2)) = 1 + ((7 - x^2) / (x^2 - 1))

/-- Theorem stating that x = 1 is the solution to the equation -/
theorem solution_is_one : equation 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_one_l622_62235


namespace NUMINAMATH_CALUDE_assembly_rate_after_transformation_l622_62225

/-- Represents the factory's car assembly rate before and after transformation -/
structure AssemblyRate where
  before : ℝ
  after : ℝ

/-- The conditions of the problem -/
def problem_conditions (r : AssemblyRate) : Prop :=
  r.after = (5/3) * r.before ∧
  (40 / r.after) = (30 / r.before) - 2

/-- The theorem to prove -/
theorem assembly_rate_after_transformation (r : AssemblyRate) :
  problem_conditions r → r.after = 5 := by sorry

end NUMINAMATH_CALUDE_assembly_rate_after_transformation_l622_62225


namespace NUMINAMATH_CALUDE_seulgi_winning_score_l622_62268

/-- Represents a player's scores in a two-round darts game -/
structure PlayerScores where
  round1 : ℕ
  round2 : ℕ

/-- Calculates the total score for a player -/
def totalScore (scores : PlayerScores) : ℕ :=
  scores.round1 + scores.round2

/-- Theorem: Seulgi needs at least 25 points in the second round to win -/
theorem seulgi_winning_score 
  (hohyeon : PlayerScores) 
  (hyunjeong : PlayerScores)
  (seulgi_round1 : ℕ) :
  hohyeon.round1 = 23 →
  hohyeon.round2 = 28 →
  hyunjeong.round1 = 32 →
  hyunjeong.round2 = 17 →
  seulgi_round1 = 27 →
  ∀ seulgi_round2 : ℕ,
    (totalScore ⟨seulgi_round1, seulgi_round2⟩ > totalScore hohyeon ∧
     totalScore ⟨seulgi_round1, seulgi_round2⟩ > totalScore hyunjeong) →
    seulgi_round2 ≥ 25 :=
by
  sorry


end NUMINAMATH_CALUDE_seulgi_winning_score_l622_62268


namespace NUMINAMATH_CALUDE_work_efficiency_ratio_l622_62272

/-- The work efficiency of a worker is defined as the fraction of the total work they can complete in one day -/
def work_efficiency (days : ℚ) : ℚ := 1 / days

theorem work_efficiency_ratio 
  (a_and_b_days : ℚ) 
  (b_alone_days : ℚ) 
  (h1 : a_and_b_days = 11) 
  (h2 : b_alone_days = 33) : 
  (work_efficiency a_and_b_days - work_efficiency b_alone_days) / work_efficiency b_alone_days = 2 := by
  sorry

#check work_efficiency_ratio

end NUMINAMATH_CALUDE_work_efficiency_ratio_l622_62272


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l622_62208

-- Define the first four prime numbers
def first_four_primes : List ℕ := [2, 3, 5, 7]

-- Define the function to calculate the arithmetic mean of reciprocals
def arithmetic_mean_of_reciprocals (numbers : List ℕ) : ℚ :=
  let reciprocals := numbers.map (fun n => (1 : ℚ) / n)
  reciprocals.sum / numbers.length

-- Theorem statement
theorem arithmetic_mean_of_first_four_primes_reciprocals :
  arithmetic_mean_of_reciprocals first_four_primes = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l622_62208


namespace NUMINAMATH_CALUDE_max_roses_for_680_l622_62294

/-- Represents the pricing options for roses -/
structure RosePrices where
  individual : ℝ  -- Price of an individual rose
  dozen : ℝ       -- Price of a dozen roses
  twoDozen : ℝ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def maxRoses (budget : ℝ) (prices : RosePrices) : ℕ :=
  sorry

/-- Theorem stating that given specific pricing options and a budget of $680, the maximum number of roses that can be purchased is 325 -/
theorem max_roses_for_680 :
  let prices : RosePrices := { individual := 2.30, dozen := 36, twoDozen := 50 }
  maxRoses 680 prices = 325 := by
  sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l622_62294


namespace NUMINAMATH_CALUDE_elastic_band_radius_increase_l622_62218

theorem elastic_band_radius_increase (r₁ r₂ : ℝ) : 
  2 * π * r₁ = 40 →  -- Initial circumference
  2 * π * r₂ = 80 →  -- Final circumference
  r₂ - r₁ = 20 / π := by
  sorry

end NUMINAMATH_CALUDE_elastic_band_radius_increase_l622_62218


namespace NUMINAMATH_CALUDE_library_book_loans_l622_62240

theorem library_book_loans (initial_A initial_B initial_C final_A final_B final_C : ℕ)
  (return_rate_A return_rate_B return_rate_C : ℚ) :
  initial_A = 75 →
  initial_B = 100 →
  initial_C = 150 →
  final_A = 54 →
  final_B = 82 →
  final_C = 121 →
  return_rate_A = 65/100 →
  return_rate_B = 1/2 →
  return_rate_C = 7/10 →
  ∃ (loaned_A loaned_B loaned_C : ℕ),
    loaned_A + loaned_B + loaned_C = 420 ∧
    loaned_A ≤ loaned_B ∧
    loaned_B ≤ loaned_C ∧
    (↑loaned_A : ℚ) * return_rate_A = final_A ∧
    (↑loaned_B : ℚ) * return_rate_B = final_B ∧
    (↑loaned_C : ℚ) * return_rate_C = final_C :=
by sorry

end NUMINAMATH_CALUDE_library_book_loans_l622_62240


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l622_62211

-- Part 1
theorem simplify_expression (x y : ℝ) : x - (2*x - y) + (3*x - 2*y) = 2*x - y := by
  sorry

-- Part 2
theorem evaluate_expression : 
  let x : ℚ := -2/3
  let y : ℚ := 3/2
  2*x*y + (-3*x^3 + 5*x*y + 2) - 3*(2*x*y - x^3 + 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l622_62211


namespace NUMINAMATH_CALUDE_unique_prime_six_digit_number_l622_62216

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def six_digit_number (B : ℕ) : ℕ := 303700 + B

theorem unique_prime_six_digit_number :
  ∃! B : ℕ, B < 10 ∧ is_prime (six_digit_number B) ∧ six_digit_number B = 303703 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_six_digit_number_l622_62216


namespace NUMINAMATH_CALUDE_time_equation_l622_62248

-- Define variables
variable (g V V₀ c S t : ℝ)

-- State the theorem
theorem time_equation (eq1 : V = g * t + V₀ + c) (eq2 : S = (1/2) * g * t^2 + V₀ * t + c * t^2) :
  t = 2 * S / (V + V₀ - c) := by
  sorry

end NUMINAMATH_CALUDE_time_equation_l622_62248


namespace NUMINAMATH_CALUDE_necklace_arrangement_count_l622_62207

/-- The number of distinct circular arrangements of balls in a necklace -/
def necklace_arrangements (red : ℕ) (green : ℕ) (yellow : ℕ) : ℕ :=
  let total := red + green + yellow
  let linear_arrangements := Nat.choose (total - 1) red * Nat.choose (total - 1 - red) yellow
  (linear_arrangements - Nat.choose (total / 2) (red / 2)) / 2 + Nat.choose (total / 2) (red / 2)

/-- Theorem stating the number of distinct arrangements for the given problem -/
theorem necklace_arrangement_count :
  necklace_arrangements 6 1 8 = 1519 := by
  sorry

#eval necklace_arrangements 6 1 8

end NUMINAMATH_CALUDE_necklace_arrangement_count_l622_62207


namespace NUMINAMATH_CALUDE_x_value_proof_l622_62293

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem x_value_proof (x y : ℕ) : 
  x = sum_integers 10 20 → 
  y = count_even_integers 10 20 → 
  x + y = 171 → 
  x = 165 := by sorry

end NUMINAMATH_CALUDE_x_value_proof_l622_62293


namespace NUMINAMATH_CALUDE_visited_neither_country_l622_62230

theorem visited_neither_country (total : ℕ) (visited_iceland : ℕ) (visited_norway : ℕ) (visited_both : ℕ)
  (h1 : total = 50)
  (h2 : visited_iceland = 25)
  (h3 : visited_norway = 23)
  (h4 : visited_both = 21) :
  total - (visited_iceland + visited_norway - visited_both) = 23 :=
by sorry

end NUMINAMATH_CALUDE_visited_neither_country_l622_62230


namespace NUMINAMATH_CALUDE_log_equality_l622_62215

theorem log_equality (a b : ℝ) (h1 : a = Real.log 484 / Real.log 4) (h2 : b = Real.log 22 / Real.log 2) : a = b := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l622_62215


namespace NUMINAMATH_CALUDE_positive_number_property_l622_62284

theorem positive_number_property (x : ℝ) (h1 : x > 0) (h2 : 0.01 * x^2 + 16 = 36) : x = 20 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_property_l622_62284


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l622_62200

theorem opposite_of_negative_two : (- (-2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l622_62200


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l622_62255

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 7

theorem volleyball_team_selection :
  (Nat.choose (total_players - quadruplets) starters) +
  (quadruplets * Nat.choose (total_players - quadruplets) (starters - 1)) = 4488 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l622_62255


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l622_62280

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

-- State the theorem
theorem arithmetic_sequence_problem (a₁ d : ℤ) (h_d : d ≠ 0) :
  (∃ r : ℚ, (arithmetic_sequence a₁ d 2 + 1) ^ 2 = (arithmetic_sequence a₁ d 1 + 1) * (arithmetic_sequence a₁ d 4 + 1)) →
  arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 = -12 →
  ∀ n : ℕ, arithmetic_sequence a₁ d n = -2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l622_62280


namespace NUMINAMATH_CALUDE_perfect_div_by_three_perfect_div_by_seven_l622_62291

/-- Definition of a perfect number -/
def isPerfect (n : ℕ) : Prop :=
  n > 0 ∧ n = (Finset.filter (· < n) (Finset.range (n + 1))).sum id

/-- Theorem for perfect numbers divisible by 3 -/
theorem perfect_div_by_three (n : ℕ) (h1 : isPerfect n) (h2 : n > 6) (h3 : 3 ∣ n) : 9 ∣ n := by
  sorry

/-- Theorem for perfect numbers divisible by 7 -/
theorem perfect_div_by_seven (n : ℕ) (h1 : isPerfect n) (h2 : n > 28) (h3 : 7 ∣ n) : 49 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_perfect_div_by_three_perfect_div_by_seven_l622_62291


namespace NUMINAMATH_CALUDE_isabellas_travel_l622_62271

/-- Proves that given the conditions of Isabella's travel and currency exchange, 
    the initial amount d is 120 U.S. dollars. -/
theorem isabellas_travel (d : ℚ) : 
  (8/5 * d - 72 = d) → d = 120 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_travel_l622_62271


namespace NUMINAMATH_CALUDE_matrix_product_equals_A_l622_62267

variable {R : Type*} [Field R]
variable (d e f x y z : R)

def A : Matrix (Fin 3) (Fin 3) R :=
  ![![0, d, -e],
    ![-d, 0, f],
    ![e, -f, 0]]

def B : Matrix (Fin 3) (Fin 3) R :=
  ![![x^2 + 1, x*y, x*z],
    ![x*y, y^2 + 1, y*z],
    ![x*z, y*z, z^2 + 1]]

theorem matrix_product_equals_A :
  A d e f * B x y z = A d e f := by sorry

end NUMINAMATH_CALUDE_matrix_product_equals_A_l622_62267


namespace NUMINAMATH_CALUDE_last_digit_of_2_to_20_l622_62263

theorem last_digit_of_2_to_20 (n : ℕ) :
  n ≥ 1 → (2^n : ℕ) % 10 = ((2^(n % 4)) : ℕ) % 10 →
  (2^20 : ℕ) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_2_to_20_l622_62263


namespace NUMINAMATH_CALUDE_sean_has_45_whistles_l622_62239

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := 13

/-- The number of additional whistles Sean has compared to Charles -/
def sean_additional_whistles : ℕ := 32

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := charles_whistles + sean_additional_whistles

theorem sean_has_45_whistles : sean_whistles = 45 := by
  sorry

end NUMINAMATH_CALUDE_sean_has_45_whistles_l622_62239


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_8_ending_4_l622_62227

theorem greatest_three_digit_divisible_by_8_ending_4 : ∃ n : ℕ, 
  n = 984 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  n % 8 = 0 ∧ 
  n % 10 = 4 ∧ 
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 8 = 0 ∧ m % 10 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_8_ending_4_l622_62227


namespace NUMINAMATH_CALUDE_remaining_digits_count_l622_62249

theorem remaining_digits_count (total_count : ℕ) (total_avg : ℚ) (subset_count : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total_count = 8 →
  total_avg = 20 →
  subset_count = 5 →
  subset_avg = 12 →
  remaining_avg = 33333333333333336 / 1000000000000000 →
  total_count - subset_count = 3 :=
by sorry

end NUMINAMATH_CALUDE_remaining_digits_count_l622_62249


namespace NUMINAMATH_CALUDE_columbus_discovery_year_l622_62262

def is_valid_year (year : ℕ) : Prop :=
  1000 ≤ year ∧ year < 2000 ∧
  (year / 1000 = 1) ∧
  (year / 100 % 10 ≠ year / 10 % 10) ∧
  (year / 100 % 10 ≠ year % 10) ∧
  (year / 10 % 10 ≠ year % 10) ∧
  (year / 1000 + year / 100 % 10 + year / 10 % 10 + year % 10 = 16) ∧
  (year / 10 % 10 + 1 = 5 * (year % 10))

theorem columbus_discovery_year :
  ∀ year : ℕ, is_valid_year year ↔ year = 1492 :=
by sorry

end NUMINAMATH_CALUDE_columbus_discovery_year_l622_62262


namespace NUMINAMATH_CALUDE_delta_phi_solution_l622_62296

def δ (x : ℝ) : ℝ := 4 * x + 5

def φ (x : ℝ) : ℝ := 5 * x + 4

theorem delta_phi_solution :
  ∃ x : ℝ, δ (φ x) = 4 ∧ x = -17/20 := by
  sorry

end NUMINAMATH_CALUDE_delta_phi_solution_l622_62296


namespace NUMINAMATH_CALUDE_sum_of_y_values_l622_62285

theorem sum_of_y_values (y : ℝ) : 
  (∃ (y₁ y₂ : ℝ), 
    (Real.sqrt ((y₁ - 2)^2) = 9 ∧ 
     Real.sqrt ((y₂ - 2)^2) = 9 ∧ 
     y₁ ≠ y₂ ∧
     (∀ y', Real.sqrt ((y' - 2)^2) = 9 → y' = y₁ ∨ y' = y₂)) →
    y₁ + y₂ = 4) :=
sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l622_62285


namespace NUMINAMATH_CALUDE_smallest_number_with_digit_sum_10_l622_62292

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → (d = 1 ∨ d = 2)

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_number_with_digit_sum_10 :
  ∀ n : ℕ,
    is_valid_number n ∧ digit_sum n = 10 →
    111111112 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_digit_sum_10_l622_62292


namespace NUMINAMATH_CALUDE_max_product_at_12_l622_62295

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

def product_of_terms (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  (a₁^n) * (q^(n * (n - 1) / 2))

theorem max_product_at_12 (a₁ q : ℝ) (h₁ : a₁ = 1536) (h₂ : q = -1/2) :
  product_of_terms a₁ q 12 > product_of_terms a₁ q 9 ∧
  product_of_terms a₁ q 12 > product_of_terms a₁ q 13 := by
  sorry

end NUMINAMATH_CALUDE_max_product_at_12_l622_62295


namespace NUMINAMATH_CALUDE_divisibility_of_quadratic_form_l622_62259

theorem divisibility_of_quadratic_form (n : ℕ) (h : 0 < n) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (n ∣ 4 * a^2 + 9 * b^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_quadratic_form_l622_62259


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l622_62247

open Set Real

theorem intersection_of_M_and_N :
  let M : Set ℝ := {x | x^2 < 3*x}
  let N : Set ℝ := {x | log x < 0}
  M ∩ N = Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l622_62247


namespace NUMINAMATH_CALUDE_equation_root_l622_62288

theorem equation_root (a b c d x : ℝ) 
  (h1 : a + d = 2015)
  (h2 : b + c = 2015)
  (h3 : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) ↔ x = 1007.5 := by
sorry

end NUMINAMATH_CALUDE_equation_root_l622_62288


namespace NUMINAMATH_CALUDE_x_plus_y_equals_negative_eight_l622_62229

theorem x_plus_y_equals_negative_eight (x y : ℝ) 
  (h1 : (5 : ℝ)^x = 25^(y+2)) 
  (h2 : (16 : ℝ)^y = 4^(x+4)) : 
  x + y = -8 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_negative_eight_l622_62229


namespace NUMINAMATH_CALUDE_sqrt_greater_than_cube_root_l622_62233

theorem sqrt_greater_than_cube_root (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 + y^2) > (x^3 + y^3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_greater_than_cube_root_l622_62233


namespace NUMINAMATH_CALUDE_chess_tournament_games_l622_62204

theorem chess_tournament_games (total_games : ℕ) (participants : ℕ) 
  (h1 : total_games = 120) (h2 : participants = 16) :
  (participants - 1 : ℕ) = 15 ∧ total_games = participants * (participants - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l622_62204


namespace NUMINAMATH_CALUDE_total_jeans_purchased_l622_62289

-- Define the regular prices and quantities
def fox_price : ℝ := 15
def pony_price : ℝ := 18
def fox_quantity : ℕ := 3
def pony_quantity : ℕ := 2

-- Define the total savings and discount rates
def total_savings : ℝ := 8.55
def total_discount_rate : ℝ := 0.22
def pony_discount_rate : ℝ := 0.15

-- Define the theorem
theorem total_jeans_purchased :
  fox_quantity + pony_quantity = 5 := by sorry

end NUMINAMATH_CALUDE_total_jeans_purchased_l622_62289


namespace NUMINAMATH_CALUDE_debate_team_boys_l622_62243

theorem debate_team_boys (girls : ℕ) (groups : ℕ) (group_size : ℕ) (boys : ℕ) : 
  girls = 46 → 
  groups = 8 → 
  group_size = 9 → 
  boys + girls = groups * group_size → 
  boys = 26 := by sorry

end NUMINAMATH_CALUDE_debate_team_boys_l622_62243


namespace NUMINAMATH_CALUDE_tina_remaining_money_l622_62253

def monthly_income : ℝ := 1000

def june_bonus_rate : ℝ := 0.1
def investment_return_rate : ℝ := 0.05
def tax_rate : ℝ := 0.1

def june_savings_rate : ℝ := 0.25
def july_savings_rate : ℝ := 0.2
def august_savings_rate : ℝ := 0.3

def june_rent : ℝ := 200
def june_groceries : ℝ := 100
def june_book_rate : ℝ := 0.05

def july_rent : ℝ := 250
def july_groceries : ℝ := 150
def july_shoes_rate : ℝ := 0.15

def august_rent : ℝ := 300
def august_groceries : ℝ := 175
def august_misc_rate : ℝ := 0.1

theorem tina_remaining_money :
  let june_income := monthly_income * (1 + june_bonus_rate)
  let june_expenses := june_rent + june_groceries + (june_income * june_book_rate)
  let june_savings := june_income * june_savings_rate
  let june_remaining := june_income - june_savings - june_expenses

  let july_investment_return := june_savings * investment_return_rate
  let july_income := monthly_income + july_investment_return
  let july_expenses := july_rent + july_groceries + (monthly_income * july_shoes_rate)
  let july_savings := july_income * july_savings_rate
  let july_remaining := july_income - july_savings - july_expenses

  let august_investment_return := july_savings * investment_return_rate
  let august_income := monthly_income + august_investment_return
  let august_expenses := august_rent + august_groceries + (monthly_income * august_misc_rate)
  let august_savings := august_income * august_savings_rate
  let august_remaining := august_income - august_savings - august_expenses

  let total_investment_return := july_investment_return + august_investment_return
  let total_tax := total_investment_return * tax_rate
  let total_remaining := june_remaining + july_remaining + august_remaining - total_tax

  total_remaining = 860.7075 := by sorry

end NUMINAMATH_CALUDE_tina_remaining_money_l622_62253


namespace NUMINAMATH_CALUDE_cafe_cake_division_l622_62273

theorem cafe_cake_division (total_cake : ℚ) (tom_portion bob_portion jerry_portion : ℚ) :
  total_cake = 8/9 →
  tom_portion = 2 * bob_portion →
  tom_portion = 2 * jerry_portion →
  total_cake = tom_portion + bob_portion + jerry_portion →
  bob_portion = 2/9 :=
by sorry

end NUMINAMATH_CALUDE_cafe_cake_division_l622_62273


namespace NUMINAMATH_CALUDE_fruit_drink_total_volume_l622_62279

/-- A fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_ounces : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_total_volume (drink : FruitDrink) 
  (h1 : drink.orange_percent = 0.15)
  (h2 : drink.watermelon_percent = 0.60)
  (h3 : drink.grape_ounces = 30) :
  (drink.grape_ounces / (1 - drink.orange_percent - drink.watermelon_percent)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_total_volume_l622_62279


namespace NUMINAMATH_CALUDE_max_books_borrowed_l622_62214

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 20)
  (h2 : zero_books = 2)
  (h3 : one_book = 8)
  (h4 : two_books = 3)
  (h5 : (total_students - (zero_books + one_book + two_books)) * 3 ≤ 
        total_students * 2 - (one_book * 1 + two_books * 2)) :
  ∃ (max_books : ℕ), max_books = 8 ∧ 
    ∀ (student_books : ℕ), student_books ≤ max_books := by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l622_62214


namespace NUMINAMATH_CALUDE_expansion_properties_l622_62246

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The expansion of (x^4 + 1/x)^n -/
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

theorem expansion_properties (n : ℕ) :
  (binomial n 2 - binomial n 1 = 35) →
  (n = 10 ∧ 
   ∃ (c : ℝ), c = (expansion n 1) ∧ c = 45) := by sorry

end NUMINAMATH_CALUDE_expansion_properties_l622_62246


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l622_62244

theorem triangle_ratio_theorem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  b * Real.cos C + c * Real.sin B = a →
  b = 6 →
  -- Theorem statement
  (a + 2*b) / (Real.sin A + 2 * Real.sin B) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l622_62244


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_smallest_pair_l622_62203

/-- Predicate defining the conditions for x and y -/
def satisfies_conditions (x y : ℕ+) : Prop :=
  (x * (x + 1) ∣ y * (y + 1)) ∧
  ¬(x ∣ y) ∧
  ¬((x + 1) ∣ y) ∧
  ¬(x ∣ (y + 1)) ∧
  ¬((x + 1) ∣ (y + 1))

/-- There exist infinitely many pairs of positive integers satisfying the conditions -/
theorem infinitely_many_pairs :
  ∀ n : ℕ, ∃ x y : ℕ+, x > n ∧ y > n ∧ satisfies_conditions x y :=
sorry

/-- The smallest pair satisfying the conditions is (14, 20) -/
theorem smallest_pair :
  satisfies_conditions 14 20 ∧
  ∀ x y : ℕ+, satisfies_conditions x y → x ≥ 14 ∧ y ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_smallest_pair_l622_62203


namespace NUMINAMATH_CALUDE_ben_baseball_cards_l622_62236

/-- The number of baseball cards in each box given to Ben by his mother -/
def baseball_cards_per_box : ℕ := sorry

theorem ben_baseball_cards :
  let basketball_boxes : ℕ := 4
  let basketball_cards_per_box : ℕ := 10
  let baseball_boxes : ℕ := 5
  let cards_given_away : ℕ := 58
  let cards_remaining : ℕ := 22
  
  basketball_boxes * basketball_cards_per_box + 
  baseball_boxes * baseball_cards_per_box = 
  cards_given_away + cards_remaining →
  
  baseball_cards_per_box = 8 := by sorry

end NUMINAMATH_CALUDE_ben_baseball_cards_l622_62236


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l622_62237

/-- Diamond operation -/
def diamond (a b : ℝ) : ℝ := 3 * a - 2 * b^2

/-- Theorem stating that if a ◇ 4 = 21, then a = 53/3 -/
theorem diamond_equation_solution :
  ∀ a : ℝ, diamond a 4 = 21 → a = 53/3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l622_62237


namespace NUMINAMATH_CALUDE_adult_admission_fee_if_all_receipts_from_adults_l622_62290

/-- Proves that if all receipts came from adult tickets, the adult admission fee would be the total receipts divided by the number of adults -/
theorem adult_admission_fee_if_all_receipts_from_adults 
  (total_attendees : ℕ) 
  (total_receipts : ℚ) 
  (num_adults : ℕ) 
  (h1 : total_attendees = 578)
  (h2 : total_receipts = 985)
  (h3 : num_adults = 342)
  (h4 : num_adults ≤ total_attendees)
  (h5 : num_adults > 0) :
  let adult_fee := total_receipts / num_adults
  adult_fee * num_adults = total_receipts :=
by sorry

#eval (985 : ℚ) / 342

end NUMINAMATH_CALUDE_adult_admission_fee_if_all_receipts_from_adults_l622_62290


namespace NUMINAMATH_CALUDE_area_of_R_l622_62232

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The region R -/
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | floor (p.1 ^ 2) = floor p.2 ∧ floor (p.2 ^ 2) = floor p.1}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating the area of region R -/
theorem area_of_R : area R = 4 - 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_area_of_R_l622_62232


namespace NUMINAMATH_CALUDE_sam_speed_l622_62202

/-- Given the biking speeds of Eugene, Clara, and Sam, prove Sam's speed --/
theorem sam_speed (eugene_speed : ℚ) (clara_ratio : ℚ) (sam_ratio : ℚ) :
  eugene_speed = 5 →
  clara_ratio = 3 / 4 →
  sam_ratio = 4 / 3 →
  sam_ratio * (clara_ratio * eugene_speed) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sam_speed_l622_62202


namespace NUMINAMATH_CALUDE_complex_addition_simplification_l622_62257

theorem complex_addition_simplification :
  (7 - 4 * Complex.I) + (3 + 9 * Complex.I) = 10 + 5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_addition_simplification_l622_62257


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l622_62252

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (∃ S : ℝ, S = a / (1 - r) ∧ S = 81 * (a * r^4 / (1 - r))) → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l622_62252


namespace NUMINAMATH_CALUDE_select_computers_l622_62276

theorem select_computers (type_a : ℕ) (type_b : ℕ) : 
  type_a = 4 → type_b = 5 → 
  (Nat.choose type_a 2 * Nat.choose type_b 1) + (Nat.choose type_a 1 * Nat.choose type_b 2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_select_computers_l622_62276


namespace NUMINAMATH_CALUDE_simplify_fraction_l622_62270

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l622_62270


namespace NUMINAMATH_CALUDE_solution_product_l622_62298

theorem solution_product (p q : ℝ) : 
  (p - 6) * (3 * p + 8) = p^2 - 15 * p + 54 →
  (q - 6) * (3 * q + 8) = q^2 - 15 * q + 54 →
  p ≠ q →
  (p + 4) * (q + 4) = 130 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l622_62298
