import Mathlib

namespace NUMINAMATH_CALUDE_remainder_2468135790_div_101_l2815_281511

theorem remainder_2468135790_div_101 : 
  2468135790 % 101 = 50 := by sorry

end NUMINAMATH_CALUDE_remainder_2468135790_div_101_l2815_281511


namespace NUMINAMATH_CALUDE_common_tangent_implies_a_value_l2815_281526

/-- Two curves with a common tangent line at their common point imply a specific value for a parameter -/
theorem common_tangent_implies_a_value (e : ℝ) (a s t : ℝ) : 
  (t = (1 / (2 * Real.exp 1)) * s^2) →  -- Point P(s,t) is on the first curve
  (t = a * Real.log s) →                -- Point P(s,t) is on the second curve
  ((s / Real.exp 1) = (a / s)) →        -- Slopes are equal at point P(s,t)
  (a = 1) := by
sorry

end NUMINAMATH_CALUDE_common_tangent_implies_a_value_l2815_281526


namespace NUMINAMATH_CALUDE_cube_preserves_order_l2815_281571

theorem cube_preserves_order (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l2815_281571


namespace NUMINAMATH_CALUDE_oliver_seashell_collection_l2815_281525

/-- Represents the number of seashells Oliver collected on a given day -/
structure DailyCollection where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Calculates the total number of seashells Oliver has after Thursday -/
def totalAfterThursday (c : DailyCollection) : ℕ :=
  c.monday + c.tuesday / 2 + c.wednesday + 5

/-- Theorem stating that Oliver collected 71 seashells on Monday, Tuesday, and Wednesday -/
theorem oliver_seashell_collection (c : DailyCollection) :
  totalAfterThursday c = 76 →
  c.monday + c.tuesday + c.wednesday = 71 := by
  sorry

end NUMINAMATH_CALUDE_oliver_seashell_collection_l2815_281525


namespace NUMINAMATH_CALUDE_cuboid_diagonal_squared_l2815_281517

/-- The square of the diagonal of a cuboid equals the sum of the squares of its length, width, and height. -/
theorem cuboid_diagonal_squared (l w h d : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  d^2 = l^2 + w^2 + h^2 :=
by sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_squared_l2815_281517


namespace NUMINAMATH_CALUDE_smallest_value_of_fraction_sum_l2815_281536

theorem smallest_value_of_fraction_sum (a b : ℤ) (h : a > b) :
  (((a - b : ℚ) / (a + b)) + ((a + b : ℚ) / (a - b))) ≥ 2 ∧
  ∃ (a' b' : ℤ), a' > b' ∧ (((a' - b' : ℚ) / (a' + b')) + ((a' + b' : ℚ) / (a' - b'))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_fraction_sum_l2815_281536


namespace NUMINAMATH_CALUDE_dog_bones_total_dog_bones_example_l2815_281503

/-- Given a dog with an initial number of bones and a number of bones dug up,
    the total number of bones is equal to the sum of the initial bones and dug up bones. -/
theorem dog_bones_total (initial_bones dug_up_bones : ℕ) :
  initial_bones + dug_up_bones = initial_bones + dug_up_bones := by
  sorry

/-- The specific case from the problem -/
theorem dog_bones_example : 
  493 + 367 = 860 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_total_dog_bones_example_l2815_281503


namespace NUMINAMATH_CALUDE_extraneous_root_value_l2815_281522

theorem extraneous_root_value (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (1 / (x^2 - x) + (k - 5) / (x^2 + x) = (k - 1) / (x^2 - 1))) ∧
  (1 / (1^2 - 1) + (k - 5) / (1^2 + 1) ≠ (k - 1) / (1^2 - 1)) →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_extraneous_root_value_l2815_281522


namespace NUMINAMATH_CALUDE_regression_equation_equivalence_l2815_281564

/-- Conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Conversion factor from pounds to kilograms -/
def pound_to_kg : ℝ := 0.45

/-- Slope of the regression equation in imperial units (pounds per inch) -/
def imperial_slope : ℝ := 4

/-- Intercept of the regression equation in imperial units (pounds) -/
def imperial_intercept : ℝ := -130

/-- Predicted weight in imperial units (pounds) given height in inches -/
def predicted_weight_imperial (height : ℝ) : ℝ :=
  imperial_slope * height + imperial_intercept

/-- Predicted weight in metric units (kg) given height in cm -/
def predicted_weight_metric (height : ℝ) : ℝ :=
  0.72 * height - 58.5

theorem regression_equation_equivalence :
  ∀ height_inch : ℝ,
  let height_cm := height_inch * inch_to_cm
  predicted_weight_metric height_cm =
    predicted_weight_imperial height_inch * pound_to_kg := by
  sorry

end NUMINAMATH_CALUDE_regression_equation_equivalence_l2815_281564


namespace NUMINAMATH_CALUDE_intersection_M_N_l2815_281591

def M : Set ℝ := {x : ℝ | (x + 2) * (x - 2) > 0}

def N : Set ℝ := {-3, -2, 2, 3, 4}

theorem intersection_M_N : M ∩ N = {-3, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2815_281591


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l2815_281546

/-- The number of walnut trees in the park before planting -/
def trees_before : ℕ := 22

/-- The number of walnut trees in the park after planting -/
def trees_after : ℕ := 55

/-- The number of walnut trees planted -/
def trees_planted : ℕ := trees_after - trees_before

theorem walnut_trees_planted : trees_planted = 33 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l2815_281546


namespace NUMINAMATH_CALUDE_door_opening_probability_l2815_281580

/-- The probability of opening a door on the second try given specific conditions -/
theorem door_opening_probability (total_keys : ℕ) (opening_keys : ℕ) : 
  total_keys = 4 →
  opening_keys = 2 →
  (opening_keys : ℚ) / (total_keys : ℚ) * (opening_keys : ℚ) / ((total_keys - 1) : ℚ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_door_opening_probability_l2815_281580


namespace NUMINAMATH_CALUDE_S_n_min_l2815_281577

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

end NUMINAMATH_CALUDE_S_n_min_l2815_281577


namespace NUMINAMATH_CALUDE_book_purchase_with_discount_l2815_281549

/-- Calculates the total cost of books with a discount applied -/
theorem book_purchase_with_discount 
  (book_price : ℝ) 
  (quantity : ℕ) 
  (discount_per_book : ℝ) 
  (h1 : book_price = 5) 
  (h2 : quantity = 10) 
  (h3 : discount_per_book = 0.5) : 
  (book_price - discount_per_book) * quantity = 45 := by
sorry

end NUMINAMATH_CALUDE_book_purchase_with_discount_l2815_281549


namespace NUMINAMATH_CALUDE_calendar_puzzle_l2815_281506

def date_behind (letter : Char) (base : ℕ) : ℕ :=
  match letter with
  | 'A' => base
  | 'B' => base + 1
  | 'C' => base + 2
  | 'D' => base + 3
  | 'E' => base + 4
  | 'F' => base + 5
  | 'G' => base + 6
  | _ => base

theorem calendar_puzzle (base : ℕ) :
  ∃ (x : Char), (date_behind 'B' base + date_behind x base = 2 * date_behind 'A' base + 6) ∧ x = 'F' :=
by sorry

end NUMINAMATH_CALUDE_calendar_puzzle_l2815_281506


namespace NUMINAMATH_CALUDE_sequence_sign_change_l2815_281588

theorem sequence_sign_change (a₀ c : ℝ) (h₁ : a₀ > 0) (h₂ : c > 0) : 
  ∃ (a : ℕ → ℝ), a 0 = a₀ ∧ 
  (∀ n, a (n + 1) = (a n + c) / (1 - a n * c)) ∧
  (∀ n, n < 1990 → a n > 0) ∧
  a 1990 < 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_sign_change_l2815_281588


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l2815_281531

theorem quadratic_roots_difference (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 - m*x₁ + 8 = 0 ∧ 
   x₂^2 - m*x₂ + 8 = 0 ∧ 
   |x₁ - x₂| = Real.sqrt 84) →
  m ≤ 2 * Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l2815_281531


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_with_property_l2815_281573

-- Define the property for the sequence
def isPrimeSequenceWithProperty (p : ℕ → ℕ) : Prop :=
  (∀ n, Nat.Prime (p n)) ∧ 
  (∀ n, (p (n + 1) : ℤ) - 2 * (p n : ℤ) = 1 ∨ (p (n + 1) : ℤ) - 2 * (p n : ℤ) = -1)

-- State the theorem
theorem no_infinite_prime_sequence_with_property :
  ¬ ∃ p : ℕ → ℕ, isPrimeSequenceWithProperty p :=
sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_with_property_l2815_281573


namespace NUMINAMATH_CALUDE_lowest_number_of_students_twenty_four_is_lowest_l2815_281593

theorem lowest_number_of_students (n : ℕ) : n > 0 ∧ 12 ∣ n ∧ 24 ∣ n → n ≥ 24 := by
  sorry

theorem twenty_four_is_lowest : ∃ (n : ℕ), n > 0 ∧ 12 ∣ n ∧ 24 ∣ n ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_lowest_number_of_students_twenty_four_is_lowest_l2815_281593


namespace NUMINAMATH_CALUDE_all_vertices_integer_l2815_281504

/-- A cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℤ × ℤ × ℤ

/-- Predicate to check if four vertices form a valid cube face -/
def is_valid_face (v₁ v₂ v₃ v₄ : ℤ × ℤ × ℤ) : Prop := sorry

/-- Predicate to check if four vertices are non-coplanar -/
def are_non_coplanar (v₁ v₂ v₃ v₄ : ℤ × ℤ × ℤ) : Prop := sorry

/-- Theorem: If four non-coplanar vertices of a cube have integer coordinates, 
    then all vertices of the cube have integer coordinates -/
theorem all_vertices_integer (c : Cube) 
  (h₁ : is_valid_face (c.vertices 0) (c.vertices 1) (c.vertices 2) (c.vertices 3))
  (h₂ : are_non_coplanar (c.vertices 0) (c.vertices 1) (c.vertices 2) (c.vertices 3)) :
  ∀ i, ∃ (x y z : ℤ), c.vertices i = (x, y, z) := by
  sorry


end NUMINAMATH_CALUDE_all_vertices_integer_l2815_281504


namespace NUMINAMATH_CALUDE_probability_exactly_one_instrument_l2815_281561

theorem probability_exactly_one_instrument (total : ℕ) (at_least_one_fraction : ℚ) (two_or_more : ℕ) :
  total = 800 →
  at_least_one_fraction = 2 / 5 →
  two_or_more = 96 →
  (((at_least_one_fraction * total) - two_or_more) / total : ℚ) = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_probability_exactly_one_instrument_l2815_281561


namespace NUMINAMATH_CALUDE_octahedron_colorings_l2815_281555

/-- The number of symmetries in a regular octahedron -/
def octahedronSymmetries : ℕ := 24

/-- The number of vertices in a regular octahedron -/
def octahedronVertices : ℕ := 6

/-- The number of faces in a regular octahedron -/
def octahedronFaces : ℕ := 8

/-- The number of distinct colorings of vertices in a regular octahedron -/
def vertexColorings (m : ℕ) : ℚ :=
  (m^octahedronVertices + 3*m^4 + 12*m^3 + 8*m^2) / octahedronSymmetries

/-- The number of distinct colorings of faces in a regular octahedron -/
def faceColorings (m : ℕ) : ℚ :=
  (m^octahedronFaces + 17*m^6 + 6*m^2) / octahedronSymmetries

/-- Theorem stating the correct number of distinct colorings for vertices and faces -/
theorem octahedron_colorings (m : ℕ) :
  (vertexColorings m = (m^6 + 3*m^4 + 12*m^3 + 8*m^2) / 24) ∧
  (faceColorings m = (m^8 + 17*m^6 + 6*m^2) / 24) := by
  sorry

end NUMINAMATH_CALUDE_octahedron_colorings_l2815_281555


namespace NUMINAMATH_CALUDE_f_6_l2815_281592

/-- A function satisfying f(x) = f(x - 2) + 3 for all real x, with f(2) = 4 -/
def f : ℝ → ℝ :=
  sorry

/-- The functional equation for f -/
axiom f_eq (x : ℝ) : f x = f (x - 2) + 3

/-- The initial condition for f -/
axiom f_2 : f 2 = 4

/-- Theorem: f(6) = 10 -/
theorem f_6 : f 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_f_6_l2815_281592


namespace NUMINAMATH_CALUDE_pizza_combinations_l2815_281562

/-- The number of available toppings -/
def num_toppings : ℕ := 8

/-- The number of forbidden topping combinations -/
def num_forbidden : ℕ := 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of possible one- and two-topping pizzas -/
def total_pizzas : ℕ := num_toppings + choose num_toppings 2 - num_forbidden

theorem pizza_combinations :
  total_pizzas = 35 :=
sorry

end NUMINAMATH_CALUDE_pizza_combinations_l2815_281562


namespace NUMINAMATH_CALUDE_relay_race_distance_ratio_l2815_281587

theorem relay_race_distance_ratio :
  ∀ (last_year_distance : ℕ) (table_count : ℕ) (distance_1_to_3 : ℕ),
    last_year_distance = 300 →
    table_count = 6 →
    distance_1_to_3 = 400 →
    ∃ (this_year_distance : ℕ),
      this_year_distance % last_year_distance = 0 ∧
      (this_year_distance : ℚ) / last_year_distance = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_distance_ratio_l2815_281587


namespace NUMINAMATH_CALUDE_songs_added_l2815_281518

theorem songs_added (initial_songs deleted_songs final_songs : ℕ) : 
  initial_songs = 8 → deleted_songs = 5 → final_songs = 33 →
  final_songs - (initial_songs - deleted_songs) = 30 :=
by sorry

end NUMINAMATH_CALUDE_songs_added_l2815_281518


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l2815_281589

/-- A line with equal intercepts on both axes passing through (2, -3) -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (2, -3) -/
  point_condition : -3 = m * 2 + b
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b = -b / m

/-- The equation of a line with equal intercepts passing through (2, -3) is either x + y + 1 = 0 or 3x + 2y = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = -1 ∧ l.b = 1) ∨ (l.m = -3/2 ∧ l.b = 0) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l2815_281589


namespace NUMINAMATH_CALUDE_pebble_distribution_theorem_l2815_281574

/-- Represents a point on a 2D integer grid -/
structure Point where
  x : Int
  y : Int

/-- Represents the state of the pebble distribution -/
def PebbleState := Point → Nat

/-- Represents an operation on the pebble distribution -/
def Operation := PebbleState → Option PebbleState

/-- The initial state has 2009 pebbles distributed on integer coordinate points -/
def initial_state : PebbleState := sorry

/-- An operation is valid if it removes 4 pebbles from a point with at least 4 pebbles
    and adds 1 pebble to each of its four adjacent points -/
def valid_operation (op : Operation) : Prop := sorry

/-- A sequence of operations is valid if each operation in the sequence is valid -/
def valid_sequence (seq : List Operation) : Prop := sorry

/-- The final state after applying a sequence of operations -/
def final_state (init : PebbleState) (seq : List Operation) : PebbleState := sorry

/-- A state is stable if no point has more than 3 pebbles -/
def is_stable (state : PebbleState) : Prop := sorry

theorem pebble_distribution_theorem :
  ∀ (seq : List Operation),
    valid_sequence seq →
    ∃ (n : Nat),
      (is_stable (final_state initial_state (seq.take n))) ∧
      (∀ (seq' : List Operation),
        valid_sequence seq' →
        is_stable (final_state initial_state seq') →
        final_state initial_state seq = final_state initial_state seq') :=
sorry

end NUMINAMATH_CALUDE_pebble_distribution_theorem_l2815_281574


namespace NUMINAMATH_CALUDE_angle_identities_l2815_281541

theorem angle_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.cos α = 1 / 3) : 
  Real.tan α = 2 * Real.sqrt 2 ∧ 
  (Real.sqrt 2 * Real.sin (Real.pi + α) + 2 * Real.cos α) / 
  (Real.cos α - Real.sqrt 2 * Real.cos (Real.pi / 2 + α)) = -2 / 5 := by
sorry

end NUMINAMATH_CALUDE_angle_identities_l2815_281541


namespace NUMINAMATH_CALUDE_total_purchase_ways_l2815_281501

def oreo_flavors : ℕ := 7
def milk_types : ℕ := 4
def total_items : ℕ := 5

def ways_to_purchase : ℕ := sorry

theorem total_purchase_ways :
  ways_to_purchase = 13279 := by sorry

end NUMINAMATH_CALUDE_total_purchase_ways_l2815_281501


namespace NUMINAMATH_CALUDE_smallest_n_with_9_and_terminating_l2815_281535

def has_digit_9 (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ d = 9 ∧ ∃ k m : ℕ, n = k * 10 + d + m * 100

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ m k : ℕ, n = 2^m * 5^k

theorem smallest_n_with_9_and_terminating : 
  (∀ n : ℕ, n > 0 ∧ n < 4096 → ¬(is_terminating_decimal n ∧ has_digit_9 n)) ∧ 
  (is_terminating_decimal 4096 ∧ has_digit_9 4096) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_9_and_terminating_l2815_281535


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2815_281581

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≥ 5) → (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) → a ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2815_281581


namespace NUMINAMATH_CALUDE_coat_drive_l2815_281575

theorem coat_drive (total_coats high_school_coats : ℕ) 
  (h1 : total_coats = 9437)
  (h2 : high_school_coats = 6922) :
  total_coats - high_school_coats = 2515 := by
  sorry

end NUMINAMATH_CALUDE_coat_drive_l2815_281575


namespace NUMINAMATH_CALUDE_dave_tickets_l2815_281505

/-- The number of tickets Dave has at the end of the scenario -/
def final_tickets (initial_win : ℕ) (spent : ℕ) (later_win : ℕ) : ℕ :=
  initial_win - spent + later_win

/-- Theorem stating that Dave ends up with 16 tickets -/
theorem dave_tickets : final_tickets 11 5 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l2815_281505


namespace NUMINAMATH_CALUDE_combined_price_is_3105_l2815_281523

/-- Calculate the selling price of an item given its cost and profit percentage -/
def selling_price (cost : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost + cost * profit_percentage / 100

/-- Combined selling price of three items -/
def combined_selling_price (cost_A cost_B cost_C : ℕ) (profit_A profit_B profit_C : ℕ) : ℕ :=
  selling_price cost_A profit_A + selling_price cost_B profit_B + selling_price cost_C profit_C

/-- Theorem stating the combined selling price of the three items -/
theorem combined_price_is_3105 :
  combined_selling_price 500 800 1200 25 30 20 = 3105 := by
  sorry


end NUMINAMATH_CALUDE_combined_price_is_3105_l2815_281523


namespace NUMINAMATH_CALUDE_min_cost_water_tank_l2815_281520

/-- Represents the dimensions and cost of a rectangular water tank. -/
structure WaterTank where
  length : ℝ
  width : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the total cost of constructing the water tank. -/
def totalCost (tank : WaterTank) : ℝ :=
  tank.bottomCost * tank.length * tank.width +
  tank.wallCost * 2 * (tank.length + tank.width) * tank.depth

/-- Theorem stating the minimum cost configuration for the water tank. -/
theorem min_cost_water_tank :
  ∃ (tank : WaterTank),
    tank.depth = 3 ∧
    tank.length * tank.width * tank.depth = 48 ∧
    tank.bottomCost = 40 ∧
    tank.wallCost = 20 ∧
    tank.length = 4 ∧
    tank.width = 4 ∧
    totalCost tank = 1600 ∧
    (∀ (other : WaterTank),
      other.depth = 3 →
      other.length * other.width * other.depth = 48 →
      other.bottomCost = 40 →
      other.wallCost = 20 →
      totalCost other ≥ totalCost tank) := by
  sorry

end NUMINAMATH_CALUDE_min_cost_water_tank_l2815_281520


namespace NUMINAMATH_CALUDE_cannot_afford_both_phones_l2815_281539

/-- Represents the financial situation of Alexander and Natalia --/
structure FinancialSituation where
  alexander_salary : ℕ
  natalia_salary : ℕ
  utilities_expenses : ℕ
  loan_expenses : ℕ
  cultural_expenses : ℕ
  vacation_savings : ℕ
  dining_expenses : ℕ
  phone_a_cost : ℕ
  phone_b_cost : ℕ

/-- Theorem stating that Alexander and Natalia cannot afford both phones --/
theorem cannot_afford_both_phones (fs : FinancialSituation) 
  (h1 : fs.alexander_salary = 125000)
  (h2 : fs.natalia_salary = 61000)
  (h3 : fs.utilities_expenses = 17000)
  (h4 : fs.loan_expenses = 15000)
  (h5 : fs.cultural_expenses = 7000)
  (h6 : fs.vacation_savings = 20000)
  (h7 : fs.dining_expenses = 60000)
  (h8 : fs.phone_a_cost = 57000)
  (h9 : fs.phone_b_cost = 37000) :
  fs.alexander_salary + fs.natalia_salary - 
  (fs.utilities_expenses + fs.loan_expenses + fs.cultural_expenses + 
   fs.vacation_savings + fs.dining_expenses) < 
  fs.phone_a_cost + fs.phone_b_cost :=
by sorry

end NUMINAMATH_CALUDE_cannot_afford_both_phones_l2815_281539


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2815_281557

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*I)*z = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2815_281557


namespace NUMINAMATH_CALUDE_scientific_notation_of_44300000_l2815_281559

theorem scientific_notation_of_44300000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 44300000 = a * (10 : ℝ) ^ n ∧ a = 4.43 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_44300000_l2815_281559


namespace NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l2815_281599

theorem rectangular_solid_edge_sum :
  ∀ (a b c r : ℝ),
    a * b * c = 512 →
    2 * (a * b + b * c + a * c) = 352 →
    b = a * r →
    c = a * r^2 →
    a = 4 →
    4 * (a + b + c) = 112 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l2815_281599


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_perpendicular_lines_from_perpendicular_parallel_l2815_281529

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_lines_from_perpendicular_planes
  (m n : Line) (α β : Plane)
  (h1 : perpendicular_line_plane m α)
  (h2 : perpendicular_line_plane n β)
  (h3 : perpendicular_plane_plane α β) :
  perpendicular_line_line m n :=
sorry

-- Theorem 2
theorem perpendicular_lines_from_perpendicular_parallel
  (m n : Line) (α β : Plane)
  (h1 : perpendicular_line_plane m α)
  (h2 : parallel_line_plane n β)
  (h3 : parallel_plane_plane α β) :
  perpendicular_line_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_perpendicular_lines_from_perpendicular_parallel_l2815_281529


namespace NUMINAMATH_CALUDE_sum_of_digits_0_to_99_l2815_281532

/-- The sum of all digits of integers from 0 to 99 inclusive -/
def sum_of_digits : ℕ := 900

/-- Theorem stating that the sum of all digits of integers from 0 to 99 inclusive is 900 -/
theorem sum_of_digits_0_to_99 : sum_of_digits = 900 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_0_to_99_l2815_281532


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l2815_281521

/-- Given a square with perimeter 64 inches, cutting out an equilateral triangle
    with side length equal to the square's side and translating it to form a new figure
    results in a figure with perimeter 80 inches. -/
theorem perimeter_of_modified_square (square_perimeter : ℝ) (new_figure_perimeter : ℝ) :
  square_perimeter = 64 →
  new_figure_perimeter = square_perimeter + 2 * (square_perimeter / 4) - (square_perimeter / 4) →
  new_figure_perimeter = 80 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_modified_square_l2815_281521


namespace NUMINAMATH_CALUDE_last_number_is_25_l2815_281500

theorem last_number_is_25 (numbers : Fin 7 → ℝ) : 
  (((numbers 0) + (numbers 1) + (numbers 2) + (numbers 3)) / 4 = 13) →
  (((numbers 3) + (numbers 4) + (numbers 5) + (numbers 6)) / 4 = 15) →
  ((numbers 4) + (numbers 5) + (numbers 6) = 55) →
  ((numbers 3) ^ 2 = numbers 6) →
  (numbers 6 = 25) := by
  sorry

end NUMINAMATH_CALUDE_last_number_is_25_l2815_281500


namespace NUMINAMATH_CALUDE_binomial_expansion_equal_coefficients_l2815_281543

theorem binomial_expansion_equal_coefficients (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_equal_coefficients_l2815_281543


namespace NUMINAMATH_CALUDE_f_2_eq_0_f_positive_solution_set_l2815_281524

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - (3 - a)*x + 2*(1 - a)

-- Theorem for f(2) = 0
theorem f_2_eq_0 (a : ℝ) : f a 2 = 0 := by sorry

-- Define the solution set for f(x) > 0
def solution_set (a : ℝ) : Set ℝ :=
  if a < -1 then {x | x < 2 ∨ x > 1 - a}
  else if a = -1 then {x | x < 2 ∨ x > 2}
  else {x | x < 1 - a ∨ x > 2}

-- Theorem for the solution set of f(x) > 0
theorem f_positive_solution_set (a : ℝ) :
  {x : ℝ | f a x > 0} = solution_set a := by sorry

end NUMINAMATH_CALUDE_f_2_eq_0_f_positive_solution_set_l2815_281524


namespace NUMINAMATH_CALUDE_horse_grazing_width_l2815_281516

/-- Represents a rectangular field with a horse tethered to one corner. -/
structure GrazingField where
  length : ℝ
  width : ℝ
  rope_length : ℝ
  grazing_area : ℝ

/-- Theorem stating the width of the field that the horse can graze. -/
theorem horse_grazing_width (field : GrazingField)
  (h_length : field.length = 45)
  (h_rope : field.rope_length = 22)
  (h_area : field.grazing_area = 380.132711084365)
  : field.width = 22 := by
  sorry

end NUMINAMATH_CALUDE_horse_grazing_width_l2815_281516


namespace NUMINAMATH_CALUDE_teal_survey_result_l2815_281548

/-- Represents the survey results about teal color perception -/
structure TealSurvey where
  total : ℕ
  more_blue : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of people who believe teal is "more green" -/
def more_green (survey : TealSurvey) : ℕ :=
  survey.total - (survey.more_blue - survey.both) - survey.both - survey.neither

/-- Theorem stating the result of the teal color survey -/
theorem teal_survey_result : 
  let survey : TealSurvey := {
    total := 150,
    more_blue := 90,
    both := 40,
    neither := 20
  }
  more_green survey = 80 := by sorry

end NUMINAMATH_CALUDE_teal_survey_result_l2815_281548


namespace NUMINAMATH_CALUDE_expression_simplification_l2815_281513

theorem expression_simplification (y : ℝ) : 
  4 * y - 2 * y^2 + 3 - (8 - 4 * y + y^2) = 8 * y - 3 * y^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2815_281513


namespace NUMINAMATH_CALUDE_mikes_weekly_exercises_l2815_281586

/-- Represents the number of repetitions for each exercise -/
structure ExerciseReps where
  pullUps : ℕ
  pushUps : ℕ
  squats : ℕ

/-- Represents the number of daily visits to each room -/
structure RoomVisits where
  office : ℕ
  kitchen : ℕ
  livingRoom : ℕ

/-- Calculates the total number of exercises performed in a week -/
def weeklyExercises (reps : ExerciseReps) (visits : RoomVisits) : ExerciseReps :=
  { pullUps := reps.pullUps * visits.office * 7,
    pushUps := reps.pushUps * visits.kitchen * 7,
    squats := reps.squats * visits.livingRoom * 7 }

/-- Mike's exercise routine -/
def mikesRoutine : ExerciseReps :=
  { pullUps := 2, pushUps := 5, squats := 10 }

/-- Mike's daily room visits -/
def mikesVisits : RoomVisits :=
  { office := 5, kitchen := 8, livingRoom := 7 }

theorem mikes_weekly_exercises :
  weeklyExercises mikesRoutine mikesVisits = { pullUps := 70, pushUps := 280, squats := 490 } := by
  sorry

end NUMINAMATH_CALUDE_mikes_weekly_exercises_l2815_281586


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2815_281533

theorem min_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / y + y / z + z / x + x / z ≥ 4 ∧
  (x / y + y / z + z / x + x / z = 4 ↔ x = y ∧ y = z) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2815_281533


namespace NUMINAMATH_CALUDE_enough_money_for_jump_ropes_l2815_281527

/-- The cost of a single jump rope in yuan -/
def jump_rope_cost : ℕ := 8

/-- The number of jump ropes to be purchased -/
def num_jump_ropes : ℕ := 31

/-- The amount of money available in yuan -/
def available_money : ℕ := 250

/-- Theorem stating that the class has enough money to buy the jump ropes -/
theorem enough_money_for_jump_ropes :
  jump_rope_cost * num_jump_ropes ≤ available_money := by
  sorry

end NUMINAMATH_CALUDE_enough_money_for_jump_ropes_l2815_281527


namespace NUMINAMATH_CALUDE_triangle_side_length_l2815_281510

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Given conditions
  c = Real.sqrt 3 →
  A = π / 4 →  -- 45° in radians
  C = π / 3 →  -- 60° in radians
  -- Law of Sines
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  -- Conclusion
  a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2815_281510


namespace NUMINAMATH_CALUDE_find_set_M_l2815_281538

def U : Set Nat := {0, 1, 2, 3}

theorem find_set_M (M : Set Nat) (h : Set.compl M = {2}) : M = {0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_find_set_M_l2815_281538


namespace NUMINAMATH_CALUDE_major_premise_incorrect_l2815_281514

-- Define a differentiable function
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define a point x₀
variable (x₀ : ℝ)

-- State that f'(x₀) = 0
variable (h : deriv f x₀ = 0)

-- Define what it means for x₀ to be an extremum point
def is_extremum_point (f : ℝ → ℝ) (x₀ : ℝ) :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

-- Theorem stating that the major premise is false
theorem major_premise_incorrect :
  ¬(∀ f : ℝ → ℝ, ∀ x₀ : ℝ, Differentiable ℝ f → deriv f x₀ = 0 → is_extremum_point f x₀) :=
sorry

end NUMINAMATH_CALUDE_major_premise_incorrect_l2815_281514


namespace NUMINAMATH_CALUDE_badminton_players_count_l2815_281530

/-- A sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  tennis : ℕ
  both : ℕ
  neither : ℕ
  tennis_le_total : tennis ≤ total
  both_le_tennis : both ≤ tennis
  neither_le_total : neither ≤ total

/-- The number of members who play badminton in the sports club -/
def badminton_players (club : SportsClub) : ℕ :=
  club.total - club.tennis + club.both - club.neither

/-- Theorem stating the number of badminton players in the specific club scenario -/
theorem badminton_players_count (club : SportsClub) 
  (h_total : club.total = 30)
  (h_tennis : club.tennis = 19)
  (h_both : club.both = 8)
  (h_neither : club.neither = 2) :
  badminton_players club = 17 := by
  sorry

end NUMINAMATH_CALUDE_badminton_players_count_l2815_281530


namespace NUMINAMATH_CALUDE_scissors_cost_l2815_281569

theorem scissors_cost (initial_amount : ℕ) (num_scissors : ℕ) (num_erasers : ℕ) 
  (eraser_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 100 → 
  num_scissors = 8 → 
  num_erasers = 10 → 
  eraser_cost = 4 → 
  remaining_amount = 20 → 
  ∃ (scissor_cost : ℕ), 
    scissor_cost = 5 ∧ 
    initial_amount = num_scissors * scissor_cost + num_erasers * eraser_cost + remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_scissors_cost_l2815_281569


namespace NUMINAMATH_CALUDE_integer_between_sqrt27_and_7_l2815_281554

theorem integer_between_sqrt27_and_7 (x : ℤ) :
  (Real.sqrt 27 < x) ∧ (x < 7) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt27_and_7_l2815_281554


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l2815_281552

-- Define the circles and line
def C₁ (x y : ℝ) := (x + 3)^2 + y^2 = 4
def C₂ (x y : ℝ) := (x + 1)^2 + (y + 2)^2 = 4
def symmetry_line (x y : ℝ) := x - y + 1 = 0

-- Define points
def A : ℝ × ℝ := (0, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the theorem
theorem circle_and_line_problem :
  -- Given conditions
  (∀ x y : ℝ, C₁ x y ↔ C₂ (y - 1) (-x - 1)) →  -- Symmetry condition
  (∃ k : ℝ, ∀ x : ℝ, C₁ x (k * x + 3)) →      -- Line l intersects C₁
  -- Conclusion
  ((∀ x y : ℝ, C₁ x y ↔ (x + 3)^2 + y^2 = 4) ∧
   (∃ M N : ℝ × ℝ, 
     (C₁ M.1 M.2 ∧ C₁ N.1 N.2) ∧
     (M.2 = 2 * M.1 + 3 ∨ M.2 = 3 * M.1 + 3) ∧
     (N.2 = 2 * N.1 + 3 ∨ N.2 = 3 * N.1 + 3) ∧
     (M.1 * N.1 + M.2 * N.2 = 7/5))) :=
by sorry


end NUMINAMATH_CALUDE_circle_and_line_problem_l2815_281552


namespace NUMINAMATH_CALUDE_purely_imaginary_z_implies_tan_theta_minus_pi_4_l2815_281502

theorem purely_imaginary_z_implies_tan_theta_minus_pi_4 (θ : ℝ) :
  let z : ℂ := (Real.cos θ - 4/5) + (Real.sin θ - 3/5) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → Real.tan (θ - π/4) = -7 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_implies_tan_theta_minus_pi_4_l2815_281502


namespace NUMINAMATH_CALUDE_rectangle_area_l2815_281553

/-- A rectangle divided into four identical squares with a given perimeter has a specific area -/
theorem rectangle_area (perimeter : ℝ) (h_perimeter : perimeter = 160) :
  let side_length := perimeter / 10
  let length := 4 * side_length
  let width := side_length
  let area := length * width
  area = 1024 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2815_281553


namespace NUMINAMATH_CALUDE_gasoline_price_increase_percentage_l2815_281547

def lowest_price : ℝ := 15
def highest_price : ℝ := 24

theorem gasoline_price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_percentage_l2815_281547


namespace NUMINAMATH_CALUDE_simplify_fraction_l2815_281512

theorem simplify_fraction (x y z : ℚ) (hx : x = 5) (hz : z = 2) :
  (10 * x * y * z) / (15 * x^2 * z) = (2 * y) / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2815_281512


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l2815_281576

theorem compare_negative_fractions : -4/3 < -5/4 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l2815_281576


namespace NUMINAMATH_CALUDE_new_barbell_cost_l2815_281515

def old_barbell_cost : ℝ := 250
def price_increase_percentage : ℝ := 30

theorem new_barbell_cost : 
  old_barbell_cost * (1 + price_increase_percentage / 100) = 325 := by
  sorry

end NUMINAMATH_CALUDE_new_barbell_cost_l2815_281515


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l2815_281540

theorem quadratic_roots_real_and_equal : ∃ x : ℝ, 
  x^2 - 4*x*Real.sqrt 5 + 20 = 0 ∧ 
  (∀ y : ℝ, y^2 - 4*y*Real.sqrt 5 + 20 = 0 → y = x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l2815_281540


namespace NUMINAMATH_CALUDE_store_clearance_sale_profit_store_profit_is_3000_l2815_281560

/-- Calculates the money left after a store's clearance sale and paying creditors -/
theorem store_clearance_sale_profit (total_items : ℕ) (original_price : ℝ) 
  (discount_percent : ℝ) (sold_percent : ℝ) (owed_to_creditors : ℝ) : ℝ :=
  let sale_price := original_price * (1 - discount_percent)
  let items_sold := total_items * sold_percent
  let total_revenue := items_sold * sale_price
  let money_left := total_revenue - owed_to_creditors
  money_left

/-- Proves that the store has $3000 left after the clearance sale and paying creditors -/
theorem store_profit_is_3000 :
  store_clearance_sale_profit 2000 50 0.8 0.9 15000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_store_clearance_sale_profit_store_profit_is_3000_l2815_281560


namespace NUMINAMATH_CALUDE_divisible_by_1998_digit_sum_l2815_281542

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, if n is divisible by 1998, 
    then the sum of its digits is greater than or equal to 27 -/
theorem divisible_by_1998_digit_sum (n : ℕ) : 
  n % 1998 = 0 → sum_of_digits n ≥ 27 := by sorry

end NUMINAMATH_CALUDE_divisible_by_1998_digit_sum_l2815_281542


namespace NUMINAMATH_CALUDE_acute_triangle_angles_l2815_281590

-- Define an acute triangle
def is_acute_triangle (a b c : ℝ) : Prop :=
  0 < a ∧ a < 90 ∧
  0 < b ∧ b < 90 ∧
  0 < c ∧ c < 90 ∧
  a + b + c = 180

-- Theorem statement
theorem acute_triangle_angles (a b c : ℝ) :
  is_acute_triangle a b c →
  ∃ (x y z : ℝ), is_acute_triangle x y z ∧ x > 45 ∧ y > 45 ∧ z > 45 :=
by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_angles_l2815_281590


namespace NUMINAMATH_CALUDE_inequality_implies_x_equals_one_l2815_281585

theorem inequality_implies_x_equals_one (x : ℝ) : 
  (∀ m : ℝ, m > 0 → (m * x - 1) * (3 * m^2 - (x + 1) * m - 1) ≥ 0) → 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_x_equals_one_l2815_281585


namespace NUMINAMATH_CALUDE_toby_monday_steps_l2815_281567

/-- Represents the number of steps walked on each day of the week -/
structure WeekSteps where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total steps walked in a week -/
def totalSteps (w : WeekSteps) : ℕ :=
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday

/-- Calculates the average steps per day in a week -/
def averageSteps (w : WeekSteps) : ℚ :=
  (totalSteps w : ℚ) / 7

theorem toby_monday_steps (w : WeekSteps) 
  (h1 : averageSteps w = 9000)
  (h2 : w.sunday = 9400)
  (h3 : w.tuesday = 8300)
  (h4 : w.wednesday = 9200)
  (h5 : w.thursday = 8900)
  (h6 : (w.friday + w.saturday : ℚ) / 2 = 9050) :
  w.monday = 9100 := by
  sorry


end NUMINAMATH_CALUDE_toby_monday_steps_l2815_281567


namespace NUMINAMATH_CALUDE_octal_376_equals_decimal_254_l2815_281572

def octal_to_decimal (octal : ℕ) : ℕ :=
  (octal / 100) * 8^2 + ((octal / 10) % 10) * 8^1 + (octal % 10) * 8^0

theorem octal_376_equals_decimal_254 : octal_to_decimal 376 = 254 := by
  sorry

end NUMINAMATH_CALUDE_octal_376_equals_decimal_254_l2815_281572


namespace NUMINAMATH_CALUDE_length_AB_squared_l2815_281565

/-- The parabola function y = 3x^2 + 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 2

/-- The square of the distance between two points -/
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x2 - x1)^2 + (y2 - y1)^2

theorem length_AB_squared :
  ∀ (x1 y1 x2 y2 : ℝ),
  f x1 = y1 →  -- Point A is on the parabola
  f x2 = y2 →  -- Point B is on the parabola
  (x1 + x2) / 2 = 1 →  -- x-coordinate of midpoint C
  (y1 + y2) / 2 = 1 →  -- y-coordinate of midpoint C
  distance_squared x1 y1 x2 y2 = 17 := by
    sorry

end NUMINAMATH_CALUDE_length_AB_squared_l2815_281565


namespace NUMINAMATH_CALUDE_rectangle_area_error_percent_l2815_281568

/-- Theorem: Error percent in rectangle area calculation --/
theorem rectangle_area_error_percent
  (L W : ℝ)  -- L and W represent the actual length and width of the rectangle
  (h_positive_L : L > 0)
  (h_positive_W : W > 0)
  (measured_length : ℝ := 1.05 * L)  -- Length measured 5% in excess
  (measured_width : ℝ := 0.96 * W)   -- Width measured 4% in deficit
  (actual_area : ℝ := L * W)
  (calculated_area : ℝ := measured_length * measured_width)
  : (calculated_area - actual_area) / actual_area * 100 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percent_l2815_281568


namespace NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l2815_281534

theorem largest_of_eight_consecutive_integers (n : ℕ) 
  (h1 : n > 0) 
  (h2 : n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 2024) : 
  n + 7 = 256 := by
  sorry

#check largest_of_eight_consecutive_integers

end NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l2815_281534


namespace NUMINAMATH_CALUDE_rounding_comparison_l2815_281563

theorem rounding_comparison (a b : ℝ) : 
  (2.35 ≤ a ∧ a ≤ 2.44) → 
  (2.395 ≤ b ∧ b ≤ 2.404) → 
  (∃ x y : ℝ, (2.35 ≤ x ∧ x ≤ 2.44) ∧ (2.395 ≤ y ∧ y ≤ 2.404) ∧ x = y) ∧
  (∃ x y : ℝ, (2.35 ≤ x ∧ x ≤ 2.44) ∧ (2.395 ≤ y ∧ y ≤ 2.404) ∧ x > y) ∧
  (∃ x y : ℝ, (2.35 ≤ x ∧ x ≤ 2.44) ∧ (2.395 ≤ y ∧ y ≤ 2.404) ∧ x < y) :=
by sorry

end NUMINAMATH_CALUDE_rounding_comparison_l2815_281563


namespace NUMINAMATH_CALUDE_centipede_sock_shoe_arrangements_l2815_281544

/-- Represents the number of legs of the centipede -/
def num_legs : ℕ := 10

/-- Represents the total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- Represents the number of valid arrangements for putting on socks and shoes -/
def valid_arrangements : ℕ := Nat.factorial total_items / (2^num_legs)

/-- Theorem stating the number of valid arrangements for a centipede to put on its socks and shoes -/
theorem centipede_sock_shoe_arrangements :
  valid_arrangements = Nat.factorial total_items / (2^num_legs) := by sorry

end NUMINAMATH_CALUDE_centipede_sock_shoe_arrangements_l2815_281544


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l2815_281583

theorem no_positive_integer_solution :
  ¬ ∃ (a b c d : ℕ+), (a^2 + b^2 = c^2 - d^2) ∧ (a * b = c * d) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l2815_281583


namespace NUMINAMATH_CALUDE_ellipse_properties_l2815_281578

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  passes_through : ℝ × ℝ
  eccentricity : ℝ

/-- A line in 2D space -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem ellipse_properties (P : Ellipse) 
    (h1 : P.center = (0, 0))
    (h2 : P.foci_on_x_axis = true)
    (h3 : P.passes_through = (0, 2 * Real.sqrt 3))
    (h4 : P.eccentricity = 1/2) :
  (∃ (x y : ℝ), x^2/16 + y^2/12 = 1) ∧
  (∃ (l : Line) (R T : ℝ × ℝ),
    l.y_intercept = -4 ∧
    (l.slope = 1 ∨ l.slope = -1) ∧
    R.1^2/16 + R.2^2/12 = 1 ∧
    T.1^2/16 + T.2^2/12 = 1 ∧
    dot_product R T = 16/7) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2815_281578


namespace NUMINAMATH_CALUDE_characterization_of_brazilian_triples_l2815_281550

def is_brazilian (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (∃ k : ℕ, b * c + 1 = k * a) ∧
  (∃ k : ℕ, a * c + 1 = k * b) ∧
  (∃ k : ℕ, a * b + 1 = k * c)

def brazilian_triples : Set (ℕ × ℕ × ℕ) :=
  {(3, 2, 1), (2, 3, 1), (1, 3, 2), (2, 1, 3), (1, 2, 3), (3, 1, 2),
   (7, 3, 2), (3, 7, 2), (2, 7, 3), (3, 2, 7), (2, 3, 7), (7, 2, 3),
   (2, 1, 1), (1, 2, 1), (1, 1, 2),
   (1, 1, 1)}

theorem characterization_of_brazilian_triples :
  ∀ a b c : ℕ, is_brazilian a b c ↔ (a, b, c) ∈ brazilian_triples :=
sorry

end NUMINAMATH_CALUDE_characterization_of_brazilian_triples_l2815_281550


namespace NUMINAMATH_CALUDE_angle_difference_range_l2815_281584

theorem angle_difference_range (α β : Real) (h1 : -π < α) (h2 : α < β) (h3 : β < π) :
  -2*π < α - β ∧ α - β < 0 :=
by sorry

end NUMINAMATH_CALUDE_angle_difference_range_l2815_281584


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_perp_line_to_parallel_planes_l2815_281507

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem lines_perp_to_plane_are_parallel 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular n α) :
  parallel_lines m n := by sorry

-- Theorem 2: If two planes are parallel, and a line is perpendicular to one of them,
-- then it is perpendicular to the other
theorem perp_line_to_parallel_planes 
  (m : Line) (α β γ : Plane)
  (h1 : parallel_planes α β) (h2 : parallel_planes β γ) (h3 : perpendicular m α) :
  perpendicular m γ := by sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_perp_line_to_parallel_planes_l2815_281507


namespace NUMINAMATH_CALUDE_square_diff_roots_l2815_281551

theorem square_diff_roots : 
  (Real.sqrt (625681 + 1000) - Real.sqrt 1000)^2 = 
    626681 - 2 * Real.sqrt 626681 * 31.622776601683793 + 1000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_roots_l2815_281551


namespace NUMINAMATH_CALUDE_fraction_equality_implication_l2815_281566

theorem fraction_equality_implication (a b c d : ℝ) :
  (a + b) / (b + c) = (c + d) / (d + a) →
  (a = c ∨ a + b + c + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implication_l2815_281566


namespace NUMINAMATH_CALUDE_square_side_length_relation_l2815_281597

theorem square_side_length_relation (s L : ℝ) (h1 : s > 0) (h2 : L > 0) : 
  (4 * L) / (4 * s) = 2.5 → (L * Real.sqrt 2) / (s * Real.sqrt 2) = 2.5 → L = 2.5 * s := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_relation_l2815_281597


namespace NUMINAMATH_CALUDE_equal_division_of_cards_l2815_281582

theorem equal_division_of_cards (total_cards : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) : 
  total_cards = 455 → num_friends = 5 → cards_per_friend = total_cards / num_friends → cards_per_friend = 91 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_cards_l2815_281582


namespace NUMINAMATH_CALUDE_physics_marks_calculation_l2815_281508

def english_marks : ℕ := 74
def math_marks : ℕ := 65
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 90
def average_marks : ℚ := 75.6
def num_subjects : ℕ := 5

theorem physics_marks_calculation :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    physics_marks = 82 :=
by sorry

end NUMINAMATH_CALUDE_physics_marks_calculation_l2815_281508


namespace NUMINAMATH_CALUDE_power_equality_l2815_281556

theorem power_equality (n : ℕ) : 4^6 = 8^n → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2815_281556


namespace NUMINAMATH_CALUDE_calculate_3Z5_l2815_281537

-- Define the Z operation
def Z (a b : ℝ) : ℝ := b + 15 * a - a^3

-- Theorem statement
theorem calculate_3Z5 : Z 3 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_calculate_3Z5_l2815_281537


namespace NUMINAMATH_CALUDE_octal_addition_and_conversion_l2815_281596

/-- Converts a base-8 number to base-10 --/
def octalToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-16 --/
def decimalToHex (n : ℕ) : String := sorry

/-- Adds two base-8 numbers --/
def addOctal (a b : ℕ) : ℕ := sorry

theorem octal_addition_and_conversion :
  let a := 5214
  let b := 1742
  decimalToHex (octalToDecimal (addOctal a b)) = "E6E" := by sorry

end NUMINAMATH_CALUDE_octal_addition_and_conversion_l2815_281596


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2815_281570

theorem fifteenth_student_age 
  (total_students : Nat) 
  (group1_students : Nat) 
  (group2_students : Nat) 
  (total_average_age : ℝ) 
  (group1_average_age : ℝ) 
  (group2_average_age : ℝ) 
  (h1 : total_students = 15)
  (h2 : group1_students = 8)
  (h3 : group2_students = 6)
  (h4 : total_average_age = 15)
  (h5 : group1_average_age = 14)
  (h6 : group2_average_age = 16)
  (h7 : group1_students + group2_students + 1 = total_students) :
  (total_students : ℝ) * total_average_age - 
  ((group1_students : ℝ) * group1_average_age + (group2_students : ℝ) * group2_average_age) = 17 := by
  sorry


end NUMINAMATH_CALUDE_fifteenth_student_age_l2815_281570


namespace NUMINAMATH_CALUDE_inequality_proof_l2815_281528

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  Real.sqrt (x + (y - z)^2 / 12) + Real.sqrt (y + (x - z)^2 / 12) + Real.sqrt (z + (x - y)^2 / 12) ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2815_281528


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_zero_A_subset_B_iff_m_leq_neg_three_l2815_281519

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | m < x ∧ x < 5}

-- Theorem for part (1)
theorem complement_A_intersect_B_when_m_zero :
  (Set.univ \ A) ∩ B 0 = Set.Icc 2 5 := by sorry

-- Theorem for part (2)
theorem A_subset_B_iff_m_leq_neg_three (m : ℝ) :
  A ⊆ B m ↔ m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_zero_A_subset_B_iff_m_leq_neg_three_l2815_281519


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2815_281579

/-- An isosceles triangle with sides 4 and 6 has a perimeter of either 14 or 16 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  (a = 4 ∨ a = 6) →
  (b = 4 ∨ b = 6) →
  (c = 4 ∨ c = 6) →
  (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  a + b + c = 14 ∨ a + b + c = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2815_281579


namespace NUMINAMATH_CALUDE_percent_relation_l2815_281598

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.3 * a) 
  (h2 : c = 0.25 * b) : 
  b = 1.2 * a := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l2815_281598


namespace NUMINAMATH_CALUDE_complex_cube_eq_minus_26_plus_di_l2815_281594

-- Define a complex number z with positive integer real and imaginary parts
def z (x y : ℕ+) : ℂ := (x : ℂ) + (y : ℂ) * Complex.I

-- Theorem statement
theorem complex_cube_eq_minus_26_plus_di (x y : ℕ+) (d : ℤ) :
  (z x y)^3 = -26 + d * Complex.I → z x y = 1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_eq_minus_26_plus_di_l2815_281594


namespace NUMINAMATH_CALUDE_total_ear_muffs_bought_l2815_281509

theorem total_ear_muffs_bought (before_december : ℕ) (during_december : ℕ)
  (h1 : before_december = 1346)
  (h2 : during_december = 6444) :
  before_december + during_december = 7790 := by
  sorry

end NUMINAMATH_CALUDE_total_ear_muffs_bought_l2815_281509


namespace NUMINAMATH_CALUDE_increasing_subsequence_exists_l2815_281558

/-- Given a sequence of 2^n positive integers where each element is at most its index,
    there exists a monotonically increasing subsequence of length n+1. -/
theorem increasing_subsequence_exists (n : ℕ) (a : Fin (2^n) → ℕ)
  (h : ∀ k : Fin (2^n), a k ≤ k.val + 1) :
  ∃ (s : Fin (n + 1) → Fin (2^n)), Monotone (a ∘ s) :=
sorry

end NUMINAMATH_CALUDE_increasing_subsequence_exists_l2815_281558


namespace NUMINAMATH_CALUDE_sequence_appearance_l2815_281545

def sequence_digit (a b c : ℕ) : ℕ :=
  (a + b + c) % 10

def appears_in_sequence (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≥ 3 ∧
  (n / 1000 = sequence_digit 2 1 9) ∧
  (n / 100 % 10 = sequence_digit 1 9 (sequence_digit 2 1 9)) ∧
  (n / 10 % 10 = sequence_digit 9 (sequence_digit 2 1 9) (sequence_digit 1 9 (sequence_digit 2 1 9))) ∧
  (n % 10 = sequence_digit (sequence_digit 2 1 9) (sequence_digit 1 9 (sequence_digit 2 1 9)) (sequence_digit 9 (sequence_digit 2 1 9) (sequence_digit 1 9 (sequence_digit 2 1 9))))

theorem sequence_appearance :
  (¬ appears_in_sequence 1113 ∧ appears_in_sequence 2226 ∧ appears_in_sequence 2125 ∧ appears_in_sequence 2215) ∨
  (appears_in_sequence 1113 ∧ ¬ appears_in_sequence 2226 ∧ appears_in_sequence 2125 ∧ appears_in_sequence 2215) ∨
  (appears_in_sequence 1113 ∧ appears_in_sequence 2226 ∧ ¬ appears_in_sequence 2125 ∧ appears_in_sequence 2215) ∨
  (appears_in_sequence 1113 ∧ appears_in_sequence 2226 ∧ appears_in_sequence 2125 ∧ ¬ appears_in_sequence 2215) :=
by sorry

end NUMINAMATH_CALUDE_sequence_appearance_l2815_281545


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l2815_281595

/-- Proves that the initial markup percentage is 90% given the conditions of the coat pricing problem -/
theorem initial_markup_percentage (initial_price wholesale_price : ℚ) : 
  initial_price = 76 →
  initial_price + 4 = 2 * wholesale_price →
  wholesale_price = 40 →
  (initial_price - wholesale_price) / wholesale_price * 100 = 90 := by
sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l2815_281595
