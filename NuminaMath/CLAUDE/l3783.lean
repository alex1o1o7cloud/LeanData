import Mathlib

namespace NUMINAMATH_CALUDE_sugar_in_recipe_l3783_378375

/-- Given a cake recipe and Mary's baking progress, calculate the amount of sugar required. -/
theorem sugar_in_recipe (total_flour sugar remaining_flour : ℕ) : 
  total_flour = 10 →
  remaining_flour = total_flour - 7 →
  remaining_flour = sugar + 1 →
  sugar = 2 := by sorry

end NUMINAMATH_CALUDE_sugar_in_recipe_l3783_378375


namespace NUMINAMATH_CALUDE_solution_m_l3783_378332

theorem solution_m (x y m : ℝ) 
  (hx : x = 1) 
  (hy : y = 3) 
  (heq : 3 * m * x - 2 * y = 9) : m = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_m_l3783_378332


namespace NUMINAMATH_CALUDE_first_number_is_24_l3783_378328

theorem first_number_is_24 (x : ℝ) : 
  (x + 35 + 58) / 3 = (19 + 51 + 29) / 3 + 6 → x = 24 := by
sorry

end NUMINAMATH_CALUDE_first_number_is_24_l3783_378328


namespace NUMINAMATH_CALUDE_rubiks_cube_return_to_original_state_l3783_378373

theorem rubiks_cube_return_to_original_state 
  {S : Type} [Finite S] (f : S → S) : 
  ∃ n : ℕ+, ∀ x : S, (f^[n] x = x) := by
  sorry

end NUMINAMATH_CALUDE_rubiks_cube_return_to_original_state_l3783_378373


namespace NUMINAMATH_CALUDE_line_proof_l3783_378330

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x - y + 2 = 0
def line3 (x y : ℝ) : Prop := x + y = 0
def line4 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_proof :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    line4 x y ∧
    perpendicular (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_line_proof_l3783_378330


namespace NUMINAMATH_CALUDE_smallest_c_value_l3783_378301

theorem smallest_c_value (a b c : ℤ) : 
  a < b → b < c → 
  (c - b = b - a) →  -- arithmetic progression
  (b * b = a * c) →  -- geometric progression
  b = 3 * a → 
  (∀ x : ℤ, (x < a ∨ x = a) → 
    ¬(x < 3*x → 3*x < 9*x → 
      (9*x - 3*x = 3*x - x) → 
      ((3*x) * (3*x) = x * (9*x)))) → 
  c = 9 * a :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l3783_378301


namespace NUMINAMATH_CALUDE_correct_notation_of_expression_l3783_378315

/-- Predicate to check if an expression is correctly written in standard algebraic notation -/
def is_correct_notation : Set ℝ → Prop :=
  sorry

/-- The specific expression we're checking -/
def expression : Set ℝ := {x | ∃ y, y = |4| / 3 ∧ x = y}

/-- Theorem stating that the given expression is correctly notated -/
theorem correct_notation_of_expression : is_correct_notation expression :=
  sorry

end NUMINAMATH_CALUDE_correct_notation_of_expression_l3783_378315


namespace NUMINAMATH_CALUDE_opposite_vectors_properties_l3783_378304

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def are_opposite (a b : V) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ b = -a

theorem opposite_vectors_properties {a b : V} (h : are_opposite a b) :
  (∃ (k : ℝ), b = k • a) ∧  -- a is parallel to b
  a ≠ b ∧                   -- a ≠ b
  ‖a‖ = ‖b‖ ∧              -- |a| = |b|
  b = -a :=                 -- b = -a
by sorry

end NUMINAMATH_CALUDE_opposite_vectors_properties_l3783_378304


namespace NUMINAMATH_CALUDE_line_obtuse_angle_range_l3783_378370

/-- Given a line passing through points P(1-a, 1+a) and Q(3, 2a) with an obtuse angle of inclination,
    prove that the range of the real number a is (-2, 1). -/
theorem line_obtuse_angle_range (a : ℝ) : 
  let P : ℝ × ℝ := (1 - a, 1 + a)
  let Q : ℝ × ℝ := (3, 2 * a)
  let slope : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  (slope < 0) → -2 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_line_obtuse_angle_range_l3783_378370


namespace NUMINAMATH_CALUDE_functions_identical_functions_not_identical_l3783_378329

-- Part 1
theorem functions_identical (x : ℝ) (h : x ≠ 0) : x / x^2 = 1 / x := by sorry

-- Part 2
theorem functions_not_identical : ∃ x : ℝ, x ≠ Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_functions_identical_functions_not_identical_l3783_378329


namespace NUMINAMATH_CALUDE_salary_change_percentage_l3783_378306

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 84 / 100 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l3783_378306


namespace NUMINAMATH_CALUDE_triangle_area_l3783_378395

/-- The area of a triangle with vertices at (3, -3), (8, 4), and (3, 4) is 17.5 square units. -/
theorem triangle_area : Real := by
  -- Define the vertices of the triangle
  let v1 : (Real × Real) := (3, -3)
  let v2 : (Real × Real) := (8, 4)
  let v3 : (Real × Real) := (3, 4)

  -- Calculate the area of the triangle
  let area : Real := 17.5

  sorry -- The proof is omitted

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l3783_378395


namespace NUMINAMATH_CALUDE_only_event1_is_random_l3783_378307

-- Define the possible types of events
inductive EventType
  | Random
  | Certain
  | Impossible

-- Define the events
def event1 : EventType := EventType.Random
def event2 : EventType := EventType.Certain
def event3 : EventType := EventType.Impossible

-- Define a function to check if an event is random
def isRandomEvent (e : EventType) : Prop :=
  e = EventType.Random

-- Theorem statement
theorem only_event1_is_random :
  isRandomEvent event1 ∧ ¬isRandomEvent event2 ∧ ¬isRandomEvent event3 :=
by
  sorry


end NUMINAMATH_CALUDE_only_event1_is_random_l3783_378307


namespace NUMINAMATH_CALUDE_A_equals_B_l3783_378339

/-- Number of partitions of n where even parts are distinct -/
def A (n : ℕ) : ℕ := sorry

/-- Number of partitions of n where each part appears at most 3 times -/
def B (n : ℕ) : ℕ := sorry

/-- Theorem stating that A_n equals B_n for all natural numbers n -/
theorem A_equals_B : ∀ n : ℕ, A n = B n := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l3783_378339


namespace NUMINAMATH_CALUDE_combination_sum_permutation_ratio_l3783_378381

-- Define combination function
def C (n : ℕ) (r : ℕ) : ℕ := 
  if r ≤ n then (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))
  else 0

-- Define permutation function
def A (n : ℕ) (r : ℕ) : ℕ := 
  if r ≤ n then (Nat.factorial n) / (Nat.factorial (n - r))
  else 0

-- Theorem 1: Combination sum
theorem combination_sum : C 9 2 + C 9 3 = 120 := by sorry

-- Theorem 2: Permutation ratio
theorem permutation_ratio (n m : ℕ) (h : m < n) : 
  (A n m) / (A (n-1) (m-1)) = n := by sorry

end NUMINAMATH_CALUDE_combination_sum_permutation_ratio_l3783_378381


namespace NUMINAMATH_CALUDE_shelter_ratio_l3783_378353

theorem shelter_ratio (initial_cats : ℕ) (initial_dogs : ℕ) (additional_dogs : ℕ) :
  initial_cats = 45 →
  (initial_cats : ℚ) / initial_dogs = 15 / 7 →
  additional_dogs = 12 →
  (initial_cats : ℚ) / (initial_dogs + additional_dogs) = 15 / 11 :=
by sorry

end NUMINAMATH_CALUDE_shelter_ratio_l3783_378353


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3783_378343

/-- The minimum distance between two points on different curves with the same y-coordinate -/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  (∀ (a x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 →
    a = 2 * (x₁ + 1) →
    a = x₂ + Real.log x₂ →
    |x₂ - x₁| ≥ min_dist) ∧
  (∃ (a x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
    a = 2 * (x₁ + 1) ∧
    a = x₂ + Real.log x₂ ∧
    |x₂ - x₁| = min_dist) ∧
  min_dist = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3783_378343


namespace NUMINAMATH_CALUDE_tan_roots_sum_angles_l3783_378388

theorem tan_roots_sum_angles (α β : Real) : 
  (∃ (x y : Real), x^2 + Real.sqrt 3 * x - 2 = 0 ∧ y^2 + Real.sqrt 3 * y - 2 = 0 ∧ 
   x = Real.tan α ∧ y = Real.tan β) →
  -π/2 < α ∧ α < π/2 →
  -π/2 < β ∧ β < π/2 →
  α + β = π/6 ∨ α + β = -5*π/6 :=
by sorry

end NUMINAMATH_CALUDE_tan_roots_sum_angles_l3783_378388


namespace NUMINAMATH_CALUDE_bird_tree_stone_ratio_l3783_378316

theorem bird_tree_stone_ratio :
  let num_stones : ℕ := 40
  let num_trees : ℕ := 3 * num_stones
  let num_birds : ℕ := 400
  let combined_trees_stones : ℕ := num_trees + num_stones
  (num_birds : ℚ) / combined_trees_stones = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bird_tree_stone_ratio_l3783_378316


namespace NUMINAMATH_CALUDE_digit_150_is_5_l3783_378352

/-- The decimal representation of 31/198 -/
def decimal_rep : ℚ := 31 / 198

/-- The period of the decimal representation -/
def period : ℕ := 6

/-- The nth digit after the decimal point in the decimal representation -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

/-- The 150th digit after the decimal point in the decimal representation of 31/198 is 5 -/
theorem digit_150_is_5 : nth_digit 150 = 5 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_5_l3783_378352


namespace NUMINAMATH_CALUDE_batsman_innings_count_l3783_378389

theorem batsman_innings_count
  (avg : ℝ)
  (score_diff : ℕ)
  (avg_excluding : ℝ)
  (highest_score : ℕ)
  (h_avg : avg = 60)
  (h_score_diff : score_diff = 150)
  (h_avg_excluding : avg_excluding = 58)
  (h_highest_score : highest_score = 179)
  : ∃ n : ℕ, n = 46 ∧ 
    avg * n = avg_excluding * (n - 2) + highest_score + (highest_score - score_diff) :=
by sorry

end NUMINAMATH_CALUDE_batsman_innings_count_l3783_378389


namespace NUMINAMATH_CALUDE_probability_same_group_l3783_378310

def total_items : ℕ := 6
def group_size : ℕ := 2
def num_groups : ℕ := 3
def items_to_choose : ℕ := 2

theorem probability_same_group :
  (num_groups * (group_size.choose items_to_choose)) / (total_items.choose items_to_choose) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_group_l3783_378310


namespace NUMINAMATH_CALUDE_complex_number_range_l3783_378365

theorem complex_number_range (x y : ℝ) : 
  let z : ℂ := x + y * Complex.I
  (Complex.abs (z - (3 + 4 * Complex.I)) = 1) →
  (16 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 36) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_range_l3783_378365


namespace NUMINAMATH_CALUDE_expression_simplification_l3783_378337

theorem expression_simplification
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hbc : b - 2 / c ≠ 0) :
  (a - 2 / b) / (b - 2 / c) = c / b :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3783_378337


namespace NUMINAMATH_CALUDE_opposite_of_three_l3783_378309

theorem opposite_of_three : 
  (∃ x : ℤ, x + 3 = 0 ∧ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3783_378309


namespace NUMINAMATH_CALUDE_equation_solution_l3783_378383

theorem equation_solution : ∃ x : ℚ, (2 * x + 1 = 0) ∧ (x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3783_378383


namespace NUMINAMATH_CALUDE_parallelogram_height_l3783_378344

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) (h1 : area = 704) (h2 : base = 32) 
  (h3 : area = base * height) : height = 22 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3783_378344


namespace NUMINAMATH_CALUDE_smallest_number_l3783_378322

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

theorem smallest_number : 
  let base_9 := to_decimal [5, 8] 9
  let base_4 := to_decimal [0, 0, 0, 1] 4
  let base_2 := to_decimal [1, 1, 1, 1, 1, 1] 2
  base_2 < base_4 ∧ base_2 < base_9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3783_378322


namespace NUMINAMATH_CALUDE_rod_system_equilibrium_l3783_378340

/-- Represents the equilibrium state of a rod system -/
structure RodSystem where
  l : Real          -- Length of the rod in meters
  m₂ : Real         -- Mass of the rod in kg
  s : Real          -- Distance of left thread attachment from right end in meters
  m₁ : Real         -- Mass of the load in kg

/-- Checks if the rod system is in equilibrium -/
def is_equilibrium (sys : RodSystem) : Prop :=
  sys.m₁ * sys.s = sys.m₂ * (sys.l / 2)

/-- Theorem stating the equilibrium condition for the given rod system -/
theorem rod_system_equilibrium :
  ∀ (sys : RodSystem),
    sys.l = 0.5 ∧ 
    sys.m₂ = 2 ∧ 
    sys.s = 0.1 ∧ 
    sys.m₁ = 5 →
    is_equilibrium sys := by
  sorry

end NUMINAMATH_CALUDE_rod_system_equilibrium_l3783_378340


namespace NUMINAMATH_CALUDE_circle_center_and_chord_length_l3783_378378

/-- Definition of the circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- Definition of the line y = x -/
def line_y_eq_x (x y : ℝ) : Prop := y = x

theorem circle_center_and_chord_length :
  ∃ (center_x center_y : ℝ) (chord_length : ℝ),
    (∀ x y, circle_C x y ↔ (x - center_x)^2 + (y - center_y)^2 = 1) ∧
    center_x = 1 ∧
    center_y = 0 ∧
    chord_length = Real.sqrt 2 ∧
    chord_length^2 = 2 * (1 - (1 / Real.sqrt 2)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_chord_length_l3783_378378


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3783_378312

theorem absolute_value_equality (x : ℝ) : |x + 2| = |x - 3| → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3783_378312


namespace NUMINAMATH_CALUDE_percent_relation_l3783_378359

theorem percent_relation (x y : ℝ) (h : (1/2) * (x - y) = (1/5) * (x + y)) : 
  y = (3/7) * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l3783_378359


namespace NUMINAMATH_CALUDE_clownfish_ratio_l3783_378335

/-- The aquarium scenario -/
structure Aquarium where
  total_fish : ℕ
  clownfish : ℕ
  blowfish : ℕ
  blowfish_in_own_tank : ℕ
  clownfish_in_display : ℕ
  (equal_fish : clownfish = blowfish)
  (total_sum : clownfish + blowfish = total_fish)
  (blowfish_display : blowfish - blowfish_in_own_tank = clownfish - clownfish_in_display)

/-- The theorem to prove -/
theorem clownfish_ratio (aq : Aquarium) 
  (h1 : aq.total_fish = 100)
  (h2 : aq.blowfish_in_own_tank = 26)
  (h3 : aq.clownfish_in_display = 16) :
  (aq.clownfish - aq.clownfish_in_display) / (aq.clownfish - aq.blowfish_in_own_tank) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_clownfish_ratio_l3783_378335


namespace NUMINAMATH_CALUDE_absolute_value_symmetry_axis_of_symmetry_is_three_l3783_378341

/-- The axis of symmetry for the absolute value function y = |x-a| --/
def axisOfSymmetry (a : ℝ) : ℝ := a

/-- A function is symmetric about a vertical line if it remains unchanged when reflected about that line --/
def isSymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem absolute_value_symmetry (a : ℝ) :
  isSymmetricAbout (fun x ↦ |x - a|) (axisOfSymmetry a) := by sorry

theorem axis_of_symmetry_is_three (a : ℝ) :
  axisOfSymmetry a = 3 → a = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_symmetry_axis_of_symmetry_is_three_l3783_378341


namespace NUMINAMATH_CALUDE_no_integer_roots_l3783_378350

theorem no_integer_roots (a b : ℤ) : ¬ ∃ x : ℤ, x^2 + 3*a*x + 3*(2 - b^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3783_378350


namespace NUMINAMATH_CALUDE_initial_girls_count_initial_girls_count_proof_l3783_378331

theorem initial_girls_count : ℕ → ℕ → Prop :=
  fun b g =>
    (3 * (g - 20) = b) →
    (4 * (b - 60) = g - 20) →
    g = 42

-- The proof is omitted
theorem initial_girls_count_proof : ∃ b g : ℕ, initial_girls_count b g := by sorry

end NUMINAMATH_CALUDE_initial_girls_count_initial_girls_count_proof_l3783_378331


namespace NUMINAMATH_CALUDE_wally_bear_cost_l3783_378399

/-- Calculates the total cost of bears given the number of bears, initial price, and discount per bear. -/
def total_cost (num_bears : ℕ) (initial_price : ℚ) (discount : ℚ) : ℚ :=
  initial_price + (num_bears - 1 : ℚ) * (initial_price - discount)

/-- Theorem stating that the total cost for 101 bears is $354, given the specified pricing scheme. -/
theorem wally_bear_cost :
  total_cost 101 4 0.5 = 354 := by
  sorry

end NUMINAMATH_CALUDE_wally_bear_cost_l3783_378399


namespace NUMINAMATH_CALUDE_ratio_proof_l3783_378368

theorem ratio_proof (N : ℝ) 
  (h1 : (1 / 1) * (1 / 3) * (2 / 5) * N = 20)
  (h2 : 0.4 * N = 240) :
  20 / ((1 / 3) * (2 / 5) * N) = 2 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_proof_l3783_378368


namespace NUMINAMATH_CALUDE_turtle_arrangement_l3783_378366

/-- The number of grid intersections in a rectangular arrangement of square tiles -/
def grid_intersections (width : ℕ) (height : ℕ) : ℕ :=
  (width + 1) * height

/-- Theorem: The number of grid intersections in a 20 × 21 rectangular arrangement of square tiles is 420 -/
theorem turtle_arrangement : grid_intersections 20 21 = 420 := by
  sorry

end NUMINAMATH_CALUDE_turtle_arrangement_l3783_378366


namespace NUMINAMATH_CALUDE_larger_integer_problem_l3783_378305

theorem larger_integer_problem (x y : ℤ) : 
  y - x = 8 → x * y = 272 → max x y = 17 := by sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l3783_378305


namespace NUMINAMATH_CALUDE_square_of_binomial_l3783_378327

theorem square_of_binomial (d : ℝ) : 
  (∃ a b : ℝ, ∀ x, 8*x^2 + 24*x + d = (a*x + b)^2) → d = 18 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3783_378327


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_l3783_378336

theorem right_triangle_leg_sum : 
  ∀ (a b : ℕ), 
  (∃ k : ℕ, a = 2 * k ∧ b = 2 * k + 2) → -- legs are consecutive even whole numbers
  a^2 + b^2 = 50^2 → -- Pythagorean theorem with hypotenuse 50
  a + b = 80 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_l3783_378336


namespace NUMINAMATH_CALUDE_sin_five_pi_sixths_minus_two_alpha_l3783_378394

theorem sin_five_pi_sixths_minus_two_alpha 
  (h : Real.cos (π / 6 - α) = 1 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_sixths_minus_two_alpha_l3783_378394


namespace NUMINAMATH_CALUDE_neighborhood_to_gina_litter_ratio_l3783_378397

/-- Given the following conditions:
  * Gina collected 2 bags of litter
  * Each bag of litter weighs 4 pounds
  * Total litter collected by everyone is 664 pounds
  Prove that the ratio of litter collected by the rest of the neighborhood
  to the amount collected by Gina is 82:1 -/
theorem neighborhood_to_gina_litter_ratio :
  let gina_bags : ℕ := 2
  let bag_weight : ℕ := 4
  let total_litter : ℕ := 664
  let gina_litter := gina_bags * bag_weight
  let neighborhood_litter := total_litter - gina_litter
  neighborhood_litter / gina_litter = 82 ∧ gina_litter ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_neighborhood_to_gina_litter_ratio_l3783_378397


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3783_378386

theorem sufficient_not_necessary (p q : Prop) : 
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ 
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3783_378386


namespace NUMINAMATH_CALUDE_tile_coverage_proof_l3783_378333

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the ceiling of a fraction represented as a numerator and denominator -/
def ceilingDiv (n d : ℕ) : ℕ := (n + d - 1) / d

theorem tile_coverage_proof (tile : Dimensions) (room : Dimensions) : 
  tile.length = 2 → 
  tile.width = 5 → 
  room.length = feetToInches 3 → 
  room.width = feetToInches 8 → 
  ceilingDiv (area room) (area tile) = 346 := by
  sorry

end NUMINAMATH_CALUDE_tile_coverage_proof_l3783_378333


namespace NUMINAMATH_CALUDE_expression_evaluation_l3783_378358

theorem expression_evaluation (x z : ℝ) (hz : z ≠ 0) (hx : x = 1 / z^2) :
  (x + 1/x) * (z^2 - 1/z^2) = z^4 - 1/z^4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3783_378358


namespace NUMINAMATH_CALUDE_standard_deck_probability_l3783_378382

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = black_cards + red_cards)

/-- The probability of drawing a black card, then a red card, then a black card -/
def draw_probability (d : Deck) : ℚ :=
  (d.black_cards : ℚ) * d.red_cards * (d.black_cards - 1) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- Theorem stating the probability for a standard 52-card deck -/
theorem standard_deck_probability :
  let d : Deck := ⟨52, 26, 26, rfl⟩
  draw_probability d = 13 / 102 := by
  sorry

end NUMINAMATH_CALUDE_standard_deck_probability_l3783_378382


namespace NUMINAMATH_CALUDE_unique_dissection_solution_l3783_378392

/-- Represents a square dissection into four-cell and five-cell figures -/
structure SquareDissection where
  size : ℕ
  four_cell_count : ℕ
  five_cell_count : ℕ

/-- Checks if a given dissection is valid for a square of size 6 -/
def is_valid_dissection (d : SquareDissection) : Prop :=
  d.size = 6 ∧ 
  d.four_cell_count > 0 ∧ 
  d.five_cell_count > 0 ∧
  d.size * d.size = 4 * d.four_cell_count + 5 * d.five_cell_count

/-- The unique solution to the square dissection problem -/
def unique_solution : SquareDissection :=
  { size := 6
    four_cell_count := 4
    five_cell_count := 4 }

/-- Theorem stating that the unique solution is the only valid dissection -/
theorem unique_dissection_solution :
  ∀ d : SquareDissection, is_valid_dissection d ↔ d = unique_solution :=
by sorry


end NUMINAMATH_CALUDE_unique_dissection_solution_l3783_378392


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3783_378364

def num_black_balls : ℕ := 6
def num_white_balls : ℕ := 5

def total_balls : ℕ := num_black_balls + num_white_balls

theorem probability_of_white_ball :
  (num_white_balls : ℚ) / (total_balls : ℚ) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3783_378364


namespace NUMINAMATH_CALUDE_candy_price_increase_l3783_378374

theorem candy_price_increase (W : ℝ) (P : ℝ) (h1 : W > 0) (h2 : P > 0) :
  let new_weight := 0.6 * W
  let old_price_per_unit := P / W
  let new_price_per_unit := P / new_weight
  (new_price_per_unit - old_price_per_unit) / old_price_per_unit * 100 = (5/3 - 1) * 100 :=
by sorry

end NUMINAMATH_CALUDE_candy_price_increase_l3783_378374


namespace NUMINAMATH_CALUDE_min_value_is_zero_l3783_378334

/-- The quadratic function we're minimizing -/
def f (x y : ℝ) : ℝ := 9*x^2 - 24*x*y + 19*y^2 - 6*x - 9*y + 12

/-- The minimum value of f over all real x and y is 0 -/
theorem min_value_is_zero : 
  ∀ x y : ℝ, f x y ≥ 0 ∧ ∃ x₀ y₀ : ℝ, f x₀ y₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_is_zero_l3783_378334


namespace NUMINAMATH_CALUDE_spinner_prime_sum_probability_l3783_378311

-- Define the spinners
def spinner1 : List ℕ := [1, 2, 3, 4]
def spinner2 : List ℕ := [3, 4, 5, 6]

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Bool := sorry

-- Define a function to calculate all possible sums
def allSums (s1 s2 : List ℕ) : List ℕ := sorry

-- Define a function to count prime sums
def countPrimeSums (sums : List ℕ) : ℕ := sorry

-- Theorem to prove
theorem spinner_prime_sum_probability :
  let sums := allSums spinner1 spinner2
  let primeCount := countPrimeSums sums
  let totalCount := spinner1.length * spinner2.length
  (primeCount : ℚ) / totalCount = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_spinner_prime_sum_probability_l3783_378311


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3783_378349

theorem complex_equation_solution (z : ℂ) : (1 - I)^2 * z = 3 + 2*I → z = -1 + (3/2)*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3783_378349


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l3783_378398

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 11) (h2 : Nat.lcm a b = 181) :
  a * b = 1991 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l3783_378398


namespace NUMINAMATH_CALUDE_trig_expression_evaluation_l3783_378355

theorem trig_expression_evaluation :
  (Real.sqrt (1 - Real.cos (10 * π / 180))) / (Real.cos (85 * π / 180)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_evaluation_l3783_378355


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3783_378345

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 5| + 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3783_378345


namespace NUMINAMATH_CALUDE_complete_residue_system_product_l3783_378325

theorem complete_residue_system_product (m n : ℕ) (a : Fin m → ℤ) (b : Fin n → ℤ) :
  (∀ k : Fin (m * n), ∃ i : Fin m, ∃ j : Fin n, (a i * b j) % (m * n) = k) →
  ((∀ k : Fin m, ∃ i : Fin m, a i % m = k) ∧
   (∀ k : Fin n, ∃ j : Fin n, b j % n = k)) :=
by sorry

end NUMINAMATH_CALUDE_complete_residue_system_product_l3783_378325


namespace NUMINAMATH_CALUDE_exists_valid_assignment_l3783_378302

/-- Represents a 7x7 square table with four corner squares deleted -/
def Table := Fin 7 → Fin 7 → Option ℤ

/-- Checks if a position is a valid square on the table -/
def isValidSquare (row col : Fin 7) : Prop :=
  ¬((row = 0 ∧ col = 0) ∨ (row = 0 ∧ col = 6) ∨ (row = 6 ∧ col = 0) ∨ (row = 6 ∧ col = 6))

/-- Represents a Greek cross on the table -/
structure GreekCross (t : Table) where
  center_row : Fin 7
  center_col : Fin 7
  valid : isValidSquare center_row center_col ∧
          isValidSquare center_row (center_col - 1) ∧
          isValidSquare center_row (center_col + 1) ∧
          isValidSquare (center_row - 1) center_col ∧
          isValidSquare (center_row + 1) center_col

/-- Calculates the sum of integers in a Greek cross -/
def sumGreekCross (t : Table) (cross : GreekCross t) : ℤ :=
  sorry

/-- Calculates the sum of all integers in the table -/
def sumTable (t : Table) : ℤ :=
  sorry

/-- Main theorem to prove -/
theorem exists_valid_assignment :
  ∃ (t : Table), (∀ (cross : GreekCross t), sumGreekCross t cross < 0) ∧ sumTable t > 0 :=
sorry

end NUMINAMATH_CALUDE_exists_valid_assignment_l3783_378302


namespace NUMINAMATH_CALUDE_jamie_speed_equals_alex_speed_l3783_378313

/-- Given the cycling speeds of Alex, Sam, and Jamie, prove that Jamie's speed equals Alex's speed. -/
theorem jamie_speed_equals_alex_speed (alex_speed : ℝ) (sam_speed : ℝ) (jamie_speed : ℝ)
  (h1 : alex_speed = 6)
  (h2 : sam_speed = 3/4 * alex_speed)
  (h3 : jamie_speed = 4/3 * sam_speed) :
  jamie_speed = alex_speed :=
by sorry

end NUMINAMATH_CALUDE_jamie_speed_equals_alex_speed_l3783_378313


namespace NUMINAMATH_CALUDE_triangular_weight_is_60_l3783_378317

/-- The weight of a rectangular weight in grams -/
def rectangular_weight : ℝ := 90

/-- The weight of a round weight in grams -/
def round_weight : ℝ := 30

/-- The weight of a triangular weight in grams -/
def triangular_weight : ℝ := 60

/-- Theorem stating that the weight of a triangular weight is 60 grams -/
theorem triangular_weight_is_60 :
  (1 * round_weight + 1 * triangular_weight = 3 * round_weight) ∧
  (4 * round_weight + 1 * triangular_weight = 1 * triangular_weight + 1 * round_weight + rectangular_weight) →
  triangular_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangular_weight_is_60_l3783_378317


namespace NUMINAMATH_CALUDE_daniel_noodles_remaining_l3783_378348

/-- The number of noodles Daniel has now, given his initial count and the number he gave away. -/
def noodles_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Daniel has 54 noodles remaining. -/
theorem daniel_noodles_remaining :
  noodles_remaining 66 12 = 54 := by
  sorry

end NUMINAMATH_CALUDE_daniel_noodles_remaining_l3783_378348


namespace NUMINAMATH_CALUDE_perpendicular_point_sets_l3783_378338

-- Define the concept of a "perpendicular point set"
def isPerpendicular (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (x₁ y₁ : ℝ), (x₁, y₁) ∈ M → 
    ∃ (x₂ y₂ : ℝ), (x₂, y₂) ∈ M ∧ x₁ * x₂ + y₁ * y₂ = 0

-- Define the sets
def M₁ : Set (ℝ × ℝ) := {(x, y) | y = 1 / x^2 ∧ x ≠ 0}
def M₂ : Set (ℝ × ℝ) := {(x, y) | y = Real.log x / Real.log 2 ∧ x > 0}
def M₃ : Set (ℝ × ℝ) := {(x, y) | y = 2^x - 2}
def M₄ : Set (ℝ × ℝ) := {(x, y) | y = Real.sin x + 1}

-- State the theorem
theorem perpendicular_point_sets :
  isPerpendicular M₁ ∧ 
  ¬(isPerpendicular M₂) ∧ 
  isPerpendicular M₃ ∧ 
  isPerpendicular M₄ := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_point_sets_l3783_378338


namespace NUMINAMATH_CALUDE_average_weight_B_and_C_l3783_378362

theorem average_weight_B_and_C (A B C : ℝ) : 
  (A + B + C) / 3 = 45 →
  (A + B) / 2 = 40 →
  B = 31 →
  (B + C) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_B_and_C_l3783_378362


namespace NUMINAMATH_CALUDE_shape_cell_count_l3783_378351

theorem shape_cell_count (n : ℕ) : 
  n < 16 ∧ 
  n % 4 = 0 ∧ 
  n % 3 = 0 → 
  n = 12 := by sorry

end NUMINAMATH_CALUDE_shape_cell_count_l3783_378351


namespace NUMINAMATH_CALUDE_f_value_at_2_l3783_378321

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8) (h2 : f (-2) = 10) :
  f 2 = -26 := by sorry

end NUMINAMATH_CALUDE_f_value_at_2_l3783_378321


namespace NUMINAMATH_CALUDE_square_side_length_l3783_378393

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 144 → side ^ 2 = area → side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3783_378393


namespace NUMINAMATH_CALUDE_triangle_side_length_l3783_378342

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  c = 2 → b = 2 * a → Real.cos C = (1 : ℝ) / 4 → 
  (a^2 + b^2 - c^2) / (2 * a * b) = Real.cos C → a = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3783_378342


namespace NUMINAMATH_CALUDE_function_properties_l3783_378371

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : isEven (fun x ↦ f (x - 3)))
  (h2 : isOdd (fun x ↦ f (2 * x - 1))) :
  (f (-1) = 0) ∧ 
  (∀ x, f x = f (-x - 6)) ∧ 
  (f 7 = 0) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l3783_378371


namespace NUMINAMATH_CALUDE_tangent_line_at_one_upper_bound_condition_inequality_holds_l3783_378308

noncomputable section

def f (x : ℝ) : ℝ := (2 - Real.log x) / (x + 1)

theorem tangent_line_at_one (x y : ℝ) :
  x > 0 → (x = 1 ∧ y = f 1) → x + y = 2 := by sorry

theorem upper_bound_condition (m : ℝ) :
  (∀ x > 0, x * f x < m) ↔ m > 1 := by sorry

theorem inequality_holds (x : ℝ) :
  x > 0 → 3 - (x + 1) * f x > 1 / Real.exp x - 2 / (Real.exp 1 * x) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_upper_bound_condition_inequality_holds_l3783_378308


namespace NUMINAMATH_CALUDE_howard_earnings_l3783_378326

/-- Calculates the money earned from washing windows --/
def money_earned (initial_amount current_amount : ℕ) : ℕ :=
  current_amount - initial_amount

theorem howard_earnings :
  let initial_amount : ℕ := 26
  let current_amount : ℕ := 52
  money_earned initial_amount current_amount = 26 := by
  sorry

end NUMINAMATH_CALUDE_howard_earnings_l3783_378326


namespace NUMINAMATH_CALUDE_one_third_of_one_fourth_l3783_378361

theorem one_third_of_one_fourth (n : ℝ) : (3 / 10 : ℝ) * n = 54 → (1 / 3 : ℝ) * ((1 / 4 : ℝ) * n) = 15 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_one_fourth_l3783_378361


namespace NUMINAMATH_CALUDE_james_missed_two_questions_l3783_378363

/-- Represents the quiz bowl scoring system and James' performance -/
structure QuizBowl where
  points_per_correct : ℕ := 2
  bonus_points : ℕ := 4
  num_rounds : ℕ := 5
  questions_per_round : ℕ := 5
  james_points : ℕ := 66

/-- Calculates the number of questions James missed based on his score -/
def questions_missed (qb : QuizBowl) : ℕ :=
  let max_points := qb.num_rounds * (qb.questions_per_round * qb.points_per_correct + qb.bonus_points)
  (max_points - qb.james_points) / qb.points_per_correct

/-- Theorem stating that James missed exactly 2 questions -/
theorem james_missed_two_questions (qb : QuizBowl) : questions_missed qb = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_missed_two_questions_l3783_378363


namespace NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l3783_378346

/-- Given a parabola y^2 = 2px bounded by x = a, the maximum area of an inscribed rectangle 
    with its midline on the parabola's axis is (4a/3) * sqrt(2ap/3) -/
theorem max_area_inscribed_rectangle (p a : ℝ) (hp : p > 0) (ha : a > 0) :
  let parabola := fun y : ℝ => y^2 / (2*p)
  let bound := a
  let inscribed_rectangle_area := fun x : ℝ => 2 * (a - x) * Real.sqrt (2*p*x)
  ∃ max_area : ℝ, max_area = (4*a/3) * Real.sqrt (2*a*p/3) ∧
    ∀ x, 0 < x ∧ x < a → inscribed_rectangle_area x ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l3783_378346


namespace NUMINAMATH_CALUDE_min_value_quadratic_roots_l3783_378385

theorem min_value_quadratic_roots (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + a*x + b = 0) →
  (∃ x : ℝ, x^2 + b*x + a = 0) →
  3*a + 2*b ≥ 20 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_roots_l3783_378385


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_15_l3783_378320

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem infinitely_many_divisible_by_15 :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 15 ∣ v n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_15_l3783_378320


namespace NUMINAMATH_CALUDE_complex_product_theorem_l3783_378314

theorem complex_product_theorem (z₁ z₂ : ℂ) : 
  z₁.re = 2 ∧ z₁.im = 1 ∧ z₂.re = 0 ∧ z₂.im = -1 → z₁ * z₂ = 1 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l3783_378314


namespace NUMINAMATH_CALUDE_integer_pairs_sum_product_l3783_378319

theorem integer_pairs_sum_product (m n : ℤ) : m + n + m * n = 6 ↔ (m = 0 ∧ n = 6) ∨ (m = 6 ∧ n = 0) := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_sum_product_l3783_378319


namespace NUMINAMATH_CALUDE_largest_square_size_l3783_378300

theorem largest_square_size (board_length board_width : ℕ) 
  (h1 : board_length = 77) (h2 : board_width = 93) :
  Nat.gcd board_length board_width = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_size_l3783_378300


namespace NUMINAMATH_CALUDE_sum_zero_from_abs_inequalities_l3783_378354

theorem sum_zero_from_abs_inequalities (a b c : ℝ) 
  (ha : |a| ≥ |b+c|) (hb : |b| ≥ |c+a|) (hc : |c| ≥ |a+b|) : 
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_from_abs_inequalities_l3783_378354


namespace NUMINAMATH_CALUDE_unique_solution_triple_l3783_378396

theorem unique_solution_triple (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2 * x^3 = 2 * y * (x^2 + 1) - (z^2 + 1)) ∧
  (2 * y^4 = 3 * z * (y^2 + 1) - 2 * (x^2 + 1)) ∧
  (2 * z^5 = 4 * x * (z^2 + 1) - 3 * (y^2 + 1)) →
  x = 1 ∧ y = 1 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_triple_l3783_378396


namespace NUMINAMATH_CALUDE_a_positive_if_f_decreasing_l3783_378377

/-- A function that represents a(x³ - x) --/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x^3 - x)

/-- The theorem stating that if f is decreasing on (-√3/3, √3/3), then a > 0 --/
theorem a_positive_if_f_decreasing (a : ℝ) :
  (∀ x₁ x₂ : ℝ, -Real.sqrt 3 / 3 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.sqrt 3 / 3 → f a x₁ > f a x₂) →
  a > 0 := by
  sorry


end NUMINAMATH_CALUDE_a_positive_if_f_decreasing_l3783_378377


namespace NUMINAMATH_CALUDE_andreys_stamps_l3783_378357

theorem andreys_stamps :
  ∃ (x : ℕ), x > 0 ∧ x % 3 = 1 ∧ x % 5 = 3 ∧ x % 7 = 5 ∧ x = 208 := by
  sorry

end NUMINAMATH_CALUDE_andreys_stamps_l3783_378357


namespace NUMINAMATH_CALUDE_ham_to_pepperoni_ratio_l3783_378376

/-- Represents the number of pieces of each type of meat on a pizza -/
structure PizzaToppings where
  pepperoni : ℕ
  ham : ℕ
  sausage : ℕ

/-- Represents the properties of the pizza -/
structure Pizza where
  toppings : PizzaToppings
  slices : ℕ
  meat_per_slice : ℕ

/-- The ratio of ham to pepperoni is 2:1 given the specified conditions -/
theorem ham_to_pepperoni_ratio (pizza : Pizza) : 
  pizza.toppings.pepperoni = 30 ∧ 
  pizza.toppings.sausage = pizza.toppings.pepperoni + 12 ∧
  pizza.slices = 6 ∧
  pizza.meat_per_slice = 22 →
  pizza.toppings.ham = 2 * pizza.toppings.pepperoni := by
  sorry

#check ham_to_pepperoni_ratio

end NUMINAMATH_CALUDE_ham_to_pepperoni_ratio_l3783_378376


namespace NUMINAMATH_CALUDE_family_ages_solution_l3783_378387

def family_ages (w h s d : ℕ) : Prop :=
  -- Woman's age reversed equals husband's age
  w = 10 * (h % 10) + (h / 10) ∧
  -- Husband is older than woman
  h > w ∧
  -- Difference between ages is one-eleventh of their sum
  h - w = (h + w) / 11 ∧
  -- Son's age is difference between parents' ages
  s = h - w ∧
  -- Daughter's age is average of all ages
  d = (w + h + s) / 3 ∧
  -- Sum of digits of each age is the same
  (w % 10 + w / 10) = (h % 10 + h / 10) ∧
  (w % 10 + w / 10) = s ∧
  (w % 10 + w / 10) = (d % 10 + d / 10)

theorem family_ages_solution :
  ∃ (w h s d : ℕ), family_ages w h s d ∧ w = 45 ∧ h = 54 ∧ s = 9 ∧ d = 36 :=
by sorry

end NUMINAMATH_CALUDE_family_ages_solution_l3783_378387


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l3783_378303

theorem tan_theta_minus_pi_fourth (θ : Real) (h : Real.tan θ = 3) : 
  Real.tan (θ - π/4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l3783_378303


namespace NUMINAMATH_CALUDE_remainder_product_l3783_378372

theorem remainder_product (n : ℕ) (d : ℕ) (m : ℕ) (h : d ≠ 0) :
  (n % d) * m = 33 ↔ n = 2345678 ∧ d = 128 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_product_l3783_378372


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3783_378323

theorem rectangular_field_area (L W : ℝ) (h1 : L = 30) (h2 : 2 * W + L = 84) : L * W = 810 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3783_378323


namespace NUMINAMATH_CALUDE_complex_number_validity_one_plus_i_is_valid_l3783_378391

theorem complex_number_validity : Complex → Prop :=
  fun z => ∃ (a b : ℝ), z = Complex.mk a b

theorem one_plus_i_is_valid : complex_number_validity (1 + Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_validity_one_plus_i_is_valid_l3783_378391


namespace NUMINAMATH_CALUDE_vacation_animals_l3783_378324

/-- The total number of animals bought on the last vacation --/
def total_animals (rainbowfish clowns tetras guppies angelfish cichlids : ℕ) : ℕ :=
  rainbowfish + clowns + tetras + guppies + angelfish + cichlids

/-- Theorem stating the total number of animals bought on the last vacation --/
theorem vacation_animals :
  ∃ (rainbowfish clowns tetras guppies angelfish cichlids : ℕ),
    rainbowfish = 40 ∧
    cichlids = rainbowfish / 2 ∧
    angelfish = cichlids + 10 ∧
    guppies = 3 * angelfish ∧
    clowns = 2 * guppies ∧
    tetras = 5 * clowns ∧
    total_animals rainbowfish clowns tetras guppies angelfish cichlids = 1260 := by
  sorry


end NUMINAMATH_CALUDE_vacation_animals_l3783_378324


namespace NUMINAMATH_CALUDE_point_coordinates_l3783_378384

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the second quadrant
def second_quadrant (p : Point) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Define distance to x-axis
def distance_to_x_axis (p : Point) : ℝ :=
  |p.2|

-- Define distance to y-axis
def distance_to_y_axis (p : Point) : ℝ :=
  |p.1|

theorem point_coordinates :
  ∀ p : Point,
    second_quadrant p →
    distance_to_x_axis p = 4 →
    distance_to_y_axis p = 3 →
    p = (-3, 4) :=
by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3783_378384


namespace NUMINAMATH_CALUDE_ellipse_parameter_range_l3783_378379

-- Define the ellipse equation
def ellipse_equation (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0

-- Define what it means for a point to be inside the ellipse
def point_inside_ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 < 0

-- Theorem statement
theorem ellipse_parameter_range :
  ∀ k : ℝ, (∃ x y : ℝ, point_inside_ellipse k x y) → 0 < |k| ∧ |k| < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_parameter_range_l3783_378379


namespace NUMINAMATH_CALUDE_largest_number_in_set_l3783_378318

theorem largest_number_in_set (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- a, b, c are in ascending order
  (a + b + c) / 3 = 6 ∧  -- mean is 6
  b = 6 ∧  -- median is 6
  a = 2  -- smallest number is 2
  → c = 10 := by sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l3783_378318


namespace NUMINAMATH_CALUDE_inequality_proof_l3783_378360

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + b^2 - Real.sqrt 2 * a * b) + Real.sqrt (b^2 + c^2 - Real.sqrt 2 * b * c) ≥ Real.sqrt (a^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3783_378360


namespace NUMINAMATH_CALUDE_text_files_deleted_l3783_378390

theorem text_files_deleted (pictures_deleted : ℕ) (songs_deleted : ℕ) (total_deleted : ℕ) :
  pictures_deleted = 2 →
  songs_deleted = 8 →
  total_deleted = 17 →
  total_deleted = pictures_deleted + songs_deleted + (total_deleted - pictures_deleted - songs_deleted) →
  total_deleted - pictures_deleted - songs_deleted = 7 :=
by sorry

end NUMINAMATH_CALUDE_text_files_deleted_l3783_378390


namespace NUMINAMATH_CALUDE_existence_of_special_number_l3783_378380

theorem existence_of_special_number : ∃ N : ℕ, 
  (∃ k : ℕ, k < 150 ∧ k + 1 ≤ 150 ∧ ¬(k ∣ N) ∧ ¬((k + 1) ∣ N)) ∧ 
  (∀ m : ℕ, m ≤ 150 → (∃ k : ℕ, k < 150 ∧ k + 1 ≤ 150 ∧ m ≠ k ∧ m ≠ k + 1) → m ∣ N) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l3783_378380


namespace NUMINAMATH_CALUDE_average_age_combined_l3783_378367

theorem average_age_combined (n_students : ℕ) (avg_age_students : ℝ)
                              (n_parents : ℕ) (avg_age_parents : ℝ)
                              (n_teachers : ℕ) (avg_age_teachers : ℝ) :
  n_students = 40 →
  avg_age_students = 10 →
  n_parents = 60 →
  avg_age_parents = 35 →
  n_teachers = 5 →
  avg_age_teachers = 45 →
  (n_students * avg_age_students + n_parents * avg_age_parents + n_teachers * avg_age_teachers) /
  (n_students + n_parents + n_teachers : ℝ) = 26 := by
  sorry

#check average_age_combined

end NUMINAMATH_CALUDE_average_age_combined_l3783_378367


namespace NUMINAMATH_CALUDE_glycerin_mixture_problem_l3783_378347

theorem glycerin_mixture_problem :
  let total_volume : ℝ := 100
  let final_concentration : ℝ := 0.75
  let solution1_volume : ℝ := 75
  let solution1_concentration : ℝ := 0.30
  let solution2_volume : ℝ := 75
  let solution2_concentration : ℝ := x
  (solution1_volume * solution1_concentration + solution2_volume * solution2_concentration = total_volume * final_concentration) →
  x = 0.70 :=
by sorry

end NUMINAMATH_CALUDE_glycerin_mixture_problem_l3783_378347


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_l3783_378369

theorem least_seven_digit_binary : ∀ n : ℕ, n > 0 →
  (64 ≤ n ↔ (Nat.log 2 n).succ ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_l3783_378369


namespace NUMINAMATH_CALUDE_integral_sqrt_plus_x_equals_pi_over_two_l3783_378356

open Set
open MeasureTheory
open Interval

/-- The definite integral of √(1-x²) + x from -1 to 1 equals π/2 -/
theorem integral_sqrt_plus_x_equals_pi_over_two :
  ∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_plus_x_equals_pi_over_two_l3783_378356
