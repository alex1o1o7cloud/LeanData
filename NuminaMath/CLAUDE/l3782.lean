import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_correct_sizes_l3782_378204

def total_population : ℕ := 300
def top_class_size : ℕ := 30
def experimental_class_size : ℕ := 90
def regular_class_size : ℕ := 180
def total_sample_size : ℕ := 30

def stratum_sample_size (stratum_size : ℕ) : ℕ :=
  (stratum_size * total_sample_size) / total_population

theorem stratified_sampling_correct_sizes :
  stratum_sample_size top_class_size = 3 ∧
  stratum_sample_size experimental_class_size = 9 ∧
  stratum_sample_size regular_class_size = 18 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_sizes_l3782_378204


namespace NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_five_l3782_378254

theorem no_real_roots_x_squared_plus_five :
  ∀ x : ℝ, x^2 + 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_x_squared_plus_five_l3782_378254


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3782_378257

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (-1, 2) and b = (2, m), then m = -4. -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (-1, 2) →
  b = (2, m) →
  (∃ (k : ℝ), b = k • a) →
  m = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3782_378257


namespace NUMINAMATH_CALUDE_plane_equation_proof_l3782_378252

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines a plane in 3D space using a point and a normal vector -/
structure Plane where
  point : Point3D
  normal : Vector3D

/-- Checks if a given equation represents the plane defined by a point and normal vector -/
def is_plane_equation (p : Plane) (a b c d : ℝ) : Prop :=
  ∀ (x y z : ℝ),
    (a * x + b * y + c * z + d = 0) ↔
    (x - p.point.x) * p.normal.x + (y - p.point.y) * p.normal.y + (z - p.point.z) * p.normal.z = 0

/-- The main theorem: proving that x + 2y - z - 2 = 0 is the equation of the specified plane -/
theorem plane_equation_proof :
  let A : Point3D := ⟨1, 2, 3⟩
  let n : Vector3D := ⟨-1, -2, 1⟩
  let p : Plane := ⟨A, n⟩
  is_plane_equation p 1 2 (-1) (-2) := by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l3782_378252


namespace NUMINAMATH_CALUDE_intersection_A_B_l3782_378205

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3782_378205


namespace NUMINAMATH_CALUDE_hugo_mountain_elevation_l3782_378294

/-- The elevation of Hugo's mountain in feet -/
def hugo_elevation : ℝ := 10000

/-- The elevation of Boris' mountain in feet -/
def boris_elevation : ℝ := hugo_elevation - 2500

theorem hugo_mountain_elevation :
  (3 * hugo_elevation = 4 * boris_elevation) ∧
  (boris_elevation = hugo_elevation - 2500) →
  hugo_elevation = 10000 := by
  sorry

end NUMINAMATH_CALUDE_hugo_mountain_elevation_l3782_378294


namespace NUMINAMATH_CALUDE_right_to_left_grouping_l3782_378251

/-- Evaluates an expression using right-to-left grouping -/
def evaluateRightToLeft (a b c d : ℤ) : ℚ :=
  a / (b - c * d^2)

/-- The original expression as a function -/
def originalExpression (a b c d : ℤ) : ℚ :=
  a / b - c * d^2

theorem right_to_left_grouping (a b c d : ℤ) :
  evaluateRightToLeft a b c d = originalExpression a b c d :=
sorry

end NUMINAMATH_CALUDE_right_to_left_grouping_l3782_378251


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3782_378289

theorem perfect_square_condition (a : ℕ) : a ≥ 1 → (∃ k : ℕ, 1 - 8 * 3^a + 2^(a+2) * (2^a - 1) = k^2) ↔ a = 3 ∨ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3782_378289


namespace NUMINAMATH_CALUDE_fraction_of_120_l3782_378288

theorem fraction_of_120 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 120 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_120_l3782_378288


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l3782_378226

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

-- Define monotonicity on an interval
def MonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

-- State the theorem
theorem monotonic_quadratic_function (a : ℝ) :
  MonotonicOn (f a) 1 2 → a ≤ -2 ∨ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l3782_378226


namespace NUMINAMATH_CALUDE_ellipse_intersection_ratio_l3782_378298

/-- First ellipse -/
def ellipse1 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Second ellipse -/
def ellipse2 (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- Point P on the first ellipse -/
def P : ℝ × ℝ := sorry

/-- Point O is the origin -/
def O : ℝ × ℝ := (0, 0)

/-- Q is the intersection of ray PO with the second ellipse -/
noncomputable def Q : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem ellipse_intersection_ratio :
  ellipse1 P.1 P.2 →
  ellipse2 Q.1 Q.2 →
  distance P Q / distance O P = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_ratio_l3782_378298


namespace NUMINAMATH_CALUDE_garment_fraction_theorem_l3782_378276

theorem garment_fraction_theorem (bikini_fraction trunks_fraction : ℝ) 
  (h1 : bikini_fraction = 0.38)
  (h2 : trunks_fraction = 0.25) : 
  bikini_fraction + trunks_fraction = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_garment_fraction_theorem_l3782_378276


namespace NUMINAMATH_CALUDE_tree_height_l3782_378230

/-- Given Jane's height, Jane's shadow length, and the tree's shadow length,
    prove that the tree's height is 30 meters. -/
theorem tree_height (jane_height jane_shadow tree_shadow : ℝ)
  (h1 : jane_height = 1.5)
  (h2 : jane_shadow = 0.5)
  (h3 : tree_shadow = 10)
  (h4 : ∀ (obj1 obj2 : ℝ), obj1 / jane_shadow = jane_height / jane_shadow → obj1 / obj2 = jane_height / jane_shadow) :
  jane_height / jane_shadow * tree_shadow = 30 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_l3782_378230


namespace NUMINAMATH_CALUDE_solvable_iff_edge_start_l3782_378235

/-- Represents a cell on the 4x4 board -/
inductive Cell
| corner : Cell
| edge : Cell
| center : Cell

/-- Represents the state of the board -/
structure Board :=
(empty_cell : Cell)
(stones : Nat)

/-- Defines a valid move on the board -/
def valid_move (b : Board) : Prop :=
  b.stones > 1 ∧ ∃ (new_empty : Cell), new_empty ≠ b.empty_cell

/-- Defines the final state with one stone -/
def final_state (b : Board) : Prop :=
  b.stones = 1

/-- The main theorem to prove -/
theorem solvable_iff_edge_start :
  ∀ (b : Board),
    (b.empty_cell = Cell.edge ∧ b.stones = 15) ↔
    (∃ (b_final : Board), 
      final_state b_final ∧
      (∃ (moves : Nat), ∀ (i : Nat), i < moves → valid_move (Board.mk b.empty_cell (b.stones - i)))) :=
sorry

end NUMINAMATH_CALUDE_solvable_iff_edge_start_l3782_378235


namespace NUMINAMATH_CALUDE_general_admission_ticket_cost_l3782_378282

theorem general_admission_ticket_cost
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (general_admission_tickets : ℕ)
  (student_ticket_cost : ℕ)
  (h1 : total_tickets = 525)
  (h2 : total_revenue = 2876)
  (h3 : general_admission_tickets = 388)
  (h4 : student_ticket_cost = 4) :
  ∃ (general_admission_cost : ℕ),
    general_admission_cost * general_admission_tickets +
    student_ticket_cost * (total_tickets - general_admission_tickets) =
    total_revenue ∧
    general_admission_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_general_admission_ticket_cost_l3782_378282


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l3782_378201

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem f_nonnegative_iff_a_le_e_plus_one (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
sorry

theorem zeros_product_lt_one (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → f a x₁ = 0 → f a x₂ = 0 → x₁ * x₂ < 1 :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l3782_378201


namespace NUMINAMATH_CALUDE_base_conversion_2546_to_base5_l3782_378256

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (a b c d : Nat) : Nat :=
  a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

/-- Theorem stating that 2546 (base 10) is equal to 4141 (base 5) --/
theorem base_conversion_2546_to_base5 :
  base5ToBase10 4 1 4 1 = 2546 := by
  sorry


end NUMINAMATH_CALUDE_base_conversion_2546_to_base5_l3782_378256


namespace NUMINAMATH_CALUDE_pats_calculation_l3782_378243

theorem pats_calculation (x : ℝ) : 
  (x / 4 - 18 = 12) → (400 < 4*x + 18 ∧ 4*x + 18 < 600) := by
  sorry

end NUMINAMATH_CALUDE_pats_calculation_l3782_378243


namespace NUMINAMATH_CALUDE_class_division_theorem_l3782_378255

theorem class_division_theorem :
  ∀ (x : ℕ),
  x ≤ 26 ∧ x ≤ 30 →
  x - (24 - (30 - x)) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_class_division_theorem_l3782_378255


namespace NUMINAMATH_CALUDE_larger_number_proof_l3782_378234

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1370) (h3 : L = 6 * S + 15) : L = 1641 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3782_378234


namespace NUMINAMATH_CALUDE_combined_height_is_twelve_l3782_378274

/-- The height of Chiquita in feet -/
def chiquita_height : ℝ := 5

/-- The height difference between Mr. Martinez and Chiquita in feet -/
def height_difference : ℝ := 2

/-- The height of Mr. Martinez in feet -/
def martinez_height : ℝ := chiquita_height + height_difference

/-- The combined height of Mr. Martinez and Chiquita in feet -/
def combined_height : ℝ := chiquita_height + martinez_height

theorem combined_height_is_twelve : combined_height = 12 := by
  sorry

end NUMINAMATH_CALUDE_combined_height_is_twelve_l3782_378274


namespace NUMINAMATH_CALUDE_train_crossing_time_l3782_378223

/-- Time taken for a train to cross a man running in the same direction --/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 450 →
  train_speed = 60 * 1000 / 3600 →
  man_speed = 6 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3782_378223


namespace NUMINAMATH_CALUDE_gcd_654327_543216_l3782_378213

theorem gcd_654327_543216 : Nat.gcd 654327 543216 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_654327_543216_l3782_378213


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_l3782_378285

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_b_terms :
  ∃ (m : ℕ), m = 3 ∧ 
  (∀ (n : ℕ), n ≥ 1 → Nat.gcd (b n) (b (n + 1)) ≤ m) ∧
  (∃ (k : ℕ), k ≥ 1 ∧ Nat.gcd (b k) (b (k + 1)) = m) :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_l3782_378285


namespace NUMINAMATH_CALUDE_cone_base_area_l3782_378211

-- Define the cone
structure Cone where
  lateral_surface_area : ℝ
  base_radius : ℝ

-- Define the properties of the cone
def is_valid_cone (c : Cone) : Prop :=
  c.lateral_surface_area = 2 * Real.pi ∧
  c.lateral_surface_area = Real.pi * c.base_radius * c.base_radius

-- Theorem statement
theorem cone_base_area (c : Cone) (h : is_valid_cone c) :
  Real.pi * c.base_radius^2 = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_base_area_l3782_378211


namespace NUMINAMATH_CALUDE_library_books_existence_l3782_378219

theorem library_books_existence :
  ∃ (r p c b : ℕ), 
    r > 3000 ∧
    r = p + c + b ∧
    3 * c = 2 * p ∧
    4 * b = 3 * c :=
by sorry

end NUMINAMATH_CALUDE_library_books_existence_l3782_378219


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3782_378284

theorem gcd_factorial_eight_and_factorial_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 1920 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3782_378284


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l3782_378203

/-- The length of each identical rectangle forming PQRS, rounded to the nearest integer -/
def rectangle_length : ℕ :=
  37

theorem rectangle_length_proof (area_PQRS : ℝ) (num_rectangles : ℕ) (PQ_ratio : ℝ) (RS_ratio : ℝ) :
  area_PQRS = 6000 →
  num_rectangles = 6 →
  PQ_ratio = 4 →
  RS_ratio = 3 →
  rectangle_length = 37 := by
  sorry

#check rectangle_length_proof

end NUMINAMATH_CALUDE_rectangle_length_proof_l3782_378203


namespace NUMINAMATH_CALUDE_sketch_finalization_orders_l3782_378259

/-- Represents the order of sketches in the stack -/
def sketchOrder : List Nat := [2, 4, 1, 3, 5, 7, 6, 10, 9, 8]

/-- Represents the sketches completed before lunch -/
def completedSketches : List Nat := [8, 4]

/-- Calculates the number of possible orders for finalizing remaining sketches -/
def possibleOrders (order : List Nat) (completed : List Nat) : Nat :=
  sorry

theorem sketch_finalization_orders :
  possibleOrders sketchOrder completedSketches = 64 := by
  sorry

end NUMINAMATH_CALUDE_sketch_finalization_orders_l3782_378259


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l3782_378238

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 25) :
  (train_length + bridge_length) / crossing_time = 16 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l3782_378238


namespace NUMINAMATH_CALUDE_total_spent_is_157_l3782_378272

-- Define the initial amount given to each person
def initial_amount : ℕ := 250

-- Define Pete's spending
def pete_spending : ℕ := 20 + 30 + 50 + 5

-- Define Raymond's remaining money
def raymond_remaining : ℕ := 70 + 100 + 25 + 3

-- Theorem to prove
theorem total_spent_is_157 : 
  pete_spending + (initial_amount - raymond_remaining) = 157 := by
  sorry


end NUMINAMATH_CALUDE_total_spent_is_157_l3782_378272


namespace NUMINAMATH_CALUDE_sofie_total_distance_l3782_378212

/-- Represents the side lengths of the pentagon-shaped track in meters -/
def track_sides : List ℕ := [25, 35, 20, 40, 30]

/-- Calculates the perimeter of the track in meters -/
def track_perimeter : ℕ := track_sides.sum

/-- The number of initial laps Sofie runs -/
def initial_laps : ℕ := 2

/-- The number of additional laps Sofie runs -/
def additional_laps : ℕ := 5

/-- Theorem stating the total distance Sofie runs -/
theorem sofie_total_distance :
  initial_laps * track_perimeter + additional_laps * track_perimeter = 1050 := by
  sorry

end NUMINAMATH_CALUDE_sofie_total_distance_l3782_378212


namespace NUMINAMATH_CALUDE_contrapositive_rhombus_diagonals_l3782_378200

-- Define a quadrilateral type
structure Quadrilateral where
  -- Add necessary fields here
  mk :: -- Constructor

-- Define the property of being a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  sorry

-- Define the property of having perpendicular diagonals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  sorry

-- State the theorem
theorem contrapositive_rhombus_diagonals :
  (∀ q : Quadrilateral, ¬(has_perpendicular_diagonals q) → ¬(is_rhombus q)) ↔
  (∀ q : Quadrilateral, is_rhombus q → has_perpendicular_diagonals q) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_rhombus_diagonals_l3782_378200


namespace NUMINAMATH_CALUDE_point_distance_product_l3782_378221

theorem point_distance_product : ∃ y₁ y₂ : ℝ,
  ((-1 - 4)^2 + (y₁ - 3)^2 = 8^2) ∧
  ((-1 - 4)^2 + (y₂ - 3)^2 = 8^2) ∧
  y₁ ≠ y₂ ∧
  y₁ * y₂ = -30 := by
sorry

end NUMINAMATH_CALUDE_point_distance_product_l3782_378221


namespace NUMINAMATH_CALUDE_calculation_proof_l3782_378275

theorem calculation_proof :
  ((-5/6 + 2/3) / (-7/12) * (7/2) = 1) ∧
  ((1 - 1/6) * (-3) - (-11/6) / (-22/3) = -11/4) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3782_378275


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l3782_378209

/-- The number of trailing zeros in n -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- 20 raised to the power of 50 -/
def a : ℕ := 20^50

/-- 50 raised to the power of 20 -/
def b : ℕ := 50^20

/-- The main theorem stating that the number of trailing zeros
    in the product of 20^50 and 50^20 is 90 -/
theorem product_trailing_zeros : trailingZeros (a * b) = 90 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l3782_378209


namespace NUMINAMATH_CALUDE_course_size_l3782_378283

theorem course_size (total : ℕ) 
  (h1 : (3 : ℚ) / 10 * total + (3 : ℚ) / 10 * total + (2 : ℚ) / 10 * total + 
        (1 : ℚ) / 10 * total + 12 + 5 = total) : 
  total = 170 := by
  sorry

end NUMINAMATH_CALUDE_course_size_l3782_378283


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l3782_378231

theorem square_area_with_four_circles (r : ℝ) (h : r = 8) : 
  (2 * (2 * r))^2 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l3782_378231


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3782_378250

theorem contrapositive_equivalence :
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔
  (∀ a b : ℝ, a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3782_378250


namespace NUMINAMATH_CALUDE_sqrt3_times_3_minus_sqrt3_range_l3782_378265

theorem sqrt3_times_3_minus_sqrt3_range :
  2 < Real.sqrt 3 * (3 - Real.sqrt 3) ∧ Real.sqrt 3 * (3 - Real.sqrt 3) < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_times_3_minus_sqrt3_range_l3782_378265


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3782_378296

theorem election_winner_percentage (total_votes : ℕ) (vote_majority : ℕ) (winning_percentage : ℚ) : 
  total_votes = 400 →
  vote_majority = 160 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = vote_majority :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3782_378296


namespace NUMINAMATH_CALUDE_jonathan_calorie_deficit_l3782_378246

/-- Jonathan's calorie consumption and burning schedule --/
structure CalorieSchedule where
  regularDailyIntake : ℕ
  saturdayIntake : ℕ
  dailyBurn : ℕ

/-- Calculate the weekly caloric deficit --/
def weeklyCalorieDeficit (schedule : CalorieSchedule) : ℕ :=
  7 * schedule.dailyBurn - (6 * schedule.regularDailyIntake + schedule.saturdayIntake)

/-- Theorem stating Jonathan's weekly caloric deficit --/
theorem jonathan_calorie_deficit :
  let schedule : CalorieSchedule := {
    regularDailyIntake := 2500,
    saturdayIntake := 3500,
    dailyBurn := 3000
  }
  weeklyCalorieDeficit schedule = 2500 := by
  sorry


end NUMINAMATH_CALUDE_jonathan_calorie_deficit_l3782_378246


namespace NUMINAMATH_CALUDE_johns_money_ratio_l3782_378269

theorem johns_money_ratio (days_in_april : Nat) (sundays : Nat) (daily_earnings : Nat) 
  (book_expense : Nat) (money_left : Nat) : 
  days_in_april = 30 →
  sundays = 4 →
  daily_earnings = 10 →
  book_expense = 50 →
  money_left = 160 →
  (days_in_april - sundays) * daily_earnings - book_expense - money_left = book_expense := by
  sorry

end NUMINAMATH_CALUDE_johns_money_ratio_l3782_378269


namespace NUMINAMATH_CALUDE_is_projection_matrix_l3782_378248

def projection_matrix (A : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * A = A

theorem is_projection_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![20/49, 20/49; 20/49, 29/49]
  projection_matrix A := by
  sorry

end NUMINAMATH_CALUDE_is_projection_matrix_l3782_378248


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3782_378264

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin (α + π/4) = Real.sqrt 3 / 3) : 
  Real.cos α = (-2 * Real.sqrt 3 + Real.sqrt 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3782_378264


namespace NUMINAMATH_CALUDE_water_for_reaction_l3782_378217

/-- Represents the balanced chemical reaction between Ammonium chloride and Water -/
structure BalancedReaction where
  nh4cl : ℕ  -- moles of Ammonium chloride
  h2o : ℕ    -- moles of Water
  hcl : ℕ    -- moles of Hydrochloric acid
  nh4oh : ℕ  -- moles of Ammonium hydroxide
  balanced : nh4cl = h2o ∧ nh4cl = hcl ∧ nh4cl = nh4oh

/-- The amount of water required for the given reaction -/
def water_required (r : BalancedReaction) : ℕ := r.h2o

/-- Theorem stating that 2 moles of water are required for the given reaction -/
theorem water_for_reaction (r : BalancedReaction) 
  (h1 : r.nh4cl = 2) 
  (h2 : r.hcl = 2) 
  (h3 : r.nh4oh = 2) : 
  water_required r = 2 := by
  sorry


end NUMINAMATH_CALUDE_water_for_reaction_l3782_378217


namespace NUMINAMATH_CALUDE_passing_percentage_l3782_378260

def max_marks : ℕ := 800
def obtained_marks : ℕ := 175
def failed_by : ℕ := 89

theorem passing_percentage :
  (((obtained_marks + failed_by : ℚ) / max_marks) * 100).floor = 33 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_l3782_378260


namespace NUMINAMATH_CALUDE_product_of_numbers_l3782_378280

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 21) (sum_squares_eq : x^2 + y^2 = 527) :
  x * y = -43 := by sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3782_378280


namespace NUMINAMATH_CALUDE_triangle_area_l3782_378214

/-- Given an oblique triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is (5 * √3) / 4 under certain conditions. -/
theorem triangle_area (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  c * Real.sin A = Real.sqrt 3 * a * Real.cos C ∧  -- Given condition
  c = Real.sqrt 21 ∧  -- Given condition
  Real.sin C + Real.sin (B - A) = 5 * Real.sin (2 * A) →  -- Given condition
  (1 / 2) * a * b * Real.sin C = (5 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3782_378214


namespace NUMINAMATH_CALUDE_overtime_rate_is_five_l3782_378237

/-- Calculates the overtime pay rate given daily wage, total earnings, days worked, and overtime hours. -/
def overtime_pay_rate (daily_wage : ℚ) (total_earnings : ℚ) (days_worked : ℕ) (overtime_hours : ℕ) : ℚ :=
  (total_earnings - daily_wage * days_worked) / overtime_hours

/-- Proves that given the conditions, the overtime pay rate is $5 per hour. -/
theorem overtime_rate_is_five :
  let daily_wage : ℚ := 150
  let total_earnings : ℚ := 770
  let days_worked : ℕ := 5
  let overtime_hours : ℕ := 4
  overtime_pay_rate daily_wage total_earnings days_worked overtime_hours = 5 := by
  sorry

#eval overtime_pay_rate 150 770 5 4

end NUMINAMATH_CALUDE_overtime_rate_is_five_l3782_378237


namespace NUMINAMATH_CALUDE_hair_color_cost_l3782_378268

/-- Calculates the cost of each box of hair color based on Maddie's beauty store purchase. -/
theorem hair_color_cost (palette_price : ℝ) (lipstick_price : ℝ) (total_paid : ℝ)
  (palette_count : ℕ) (lipstick_count : ℕ) (hair_color_count : ℕ) :
  palette_price = 15 →
  lipstick_price = 2.5 →
  palette_count = 3 →
  lipstick_count = 4 →
  hair_color_count = 3 →
  total_paid = 67 →
  (total_paid - (palette_price * palette_count + lipstick_price * lipstick_count)) / hair_color_count = 4 :=
by sorry

end NUMINAMATH_CALUDE_hair_color_cost_l3782_378268


namespace NUMINAMATH_CALUDE_distribute_5_4_l3782_378287

/-- The number of ways to distribute n distinct items into k identical bags, allowing empty bags. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 4 identical bags, allowing empty bags, is 36. -/
theorem distribute_5_4 : distribute 5 4 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_5_4_l3782_378287


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_negative_l3782_378207

/-- 
Given a real number m, prove that if the solution set of the inequality 
(mx-1)(x-2) > 0 is {x | 1/m < x < 2}, then m < 0.
-/
theorem inequality_solution_implies_m_negative (m : ℝ) : 
  (∀ x, (m * x - 1) * (x - 2) > 0 ↔ 1/m < x ∧ x < 2) → m < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_negative_l3782_378207


namespace NUMINAMATH_CALUDE_added_amount_after_doubling_and_tripling_l3782_378222

theorem added_amount_after_doubling_and_tripling (x y : ℝ) : x = 5 → 3 * (2 * x + y) = 75 → y = 15 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_after_doubling_and_tripling_l3782_378222


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_one_l3782_378206

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a}

-- State the theorem
theorem intersection_nonempty_implies_a_geq_neg_one (a : ℝ) :
  (A ∩ B a).Nonempty → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_one_l3782_378206


namespace NUMINAMATH_CALUDE_cubic_factorization_l3782_378290

theorem cubic_factorization (x : ℝ) : x^3 - 4*x^2 + 4*x = x*(x-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3782_378290


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_m_geq_neg_one_l3782_378218

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 6

-- State the theorem
theorem decreasing_quadratic_implies_m_geq_neg_one (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ -1 → f m x₁ > f m x₂) →
  m ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_m_geq_neg_one_l3782_378218


namespace NUMINAMATH_CALUDE_b_sequence_max_at_4_l3782_378216

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sequence_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

def b_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := (1 + arithmetic_sequence a₁ d n) / arithmetic_sequence a₁ d n

theorem b_sequence_max_at_4 (a₁ d : ℚ) (h₁ : a₁ = -5/2) (h₂ : sequence_sum a₁ d 4 = 2 * sequence_sum a₁ d 2 + 4) :
  ∀ n : ℕ, n ≥ 1 → b_sequence a₁ d 4 ≥ b_sequence a₁ d n :=
sorry

end NUMINAMATH_CALUDE_b_sequence_max_at_4_l3782_378216


namespace NUMINAMATH_CALUDE_root_count_relationship_l3782_378225

-- Define the number of real roots for each equation
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry

-- State the theorem
theorem root_count_relationship : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_root_count_relationship_l3782_378225


namespace NUMINAMATH_CALUDE_swap_result_l3782_378292

def swap_values (x y : ℕ) : ℕ × ℕ :=
  let t := x
  let x := y
  let y := t
  (x, y)

theorem swap_result : swap_values 5 6 = (6, 5) := by
  sorry

end NUMINAMATH_CALUDE_swap_result_l3782_378292


namespace NUMINAMATH_CALUDE_expanded_lattice_equilateral_triangles_l3782_378279

/-- Represents a point in the triangular lattice --/
structure LatticePoint where
  x : ℚ
  y : ℚ

/-- The set of all points in the expanded lattice --/
def ExpandedLattice : Set LatticePoint :=
  sorry

/-- Checks if three points form an equilateral triangle --/
def IsEquilateralTriangle (p1 p2 p3 : LatticePoint) : Prop :=
  sorry

/-- Counts the number of equilateral triangles in the expanded lattice --/
def CountEquilateralTriangles (lattice : Set LatticePoint) : ℕ :=
  sorry

/-- Main theorem: The number of equilateral triangles in the expanded lattice is 14 --/
theorem expanded_lattice_equilateral_triangles :
  CountEquilateralTriangles ExpandedLattice = 14 :=
sorry

end NUMINAMATH_CALUDE_expanded_lattice_equilateral_triangles_l3782_378279


namespace NUMINAMATH_CALUDE_quadratic_function_increasing_condition_l3782_378266

/-- Given a quadratic function y = x^2 - 2mx + 5, if y increases as x increases when x > -1, then m ≤ -1 -/
theorem quadratic_function_increasing_condition (m : ℝ) : 
  (∀ x > -1, ∀ y > x, (y^2 - 2*m*y + 5) > (x^2 - 2*m*x + 5)) → m ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_increasing_condition_l3782_378266


namespace NUMINAMATH_CALUDE_triangle_theorem_triangle_area_l3782_378273

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : Real.sin t.B * (Real.tan t.A + Real.tan t.C) = Real.tan t.A * Real.tan t.C) :
  t.b^2 = t.a * t.c :=
sorry

/-- The area theorem when a = 2c = 2 -/
theorem triangle_area (t : Triangle)
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : Real.sin t.B * (Real.tan t.A + Real.tan t.C) = Real.tan t.A * Real.tan t.C)
  (h3 : t.a = 2 * t.c)
  (h4 : t.a = 2) :
  (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 7 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_triangle_area_l3782_378273


namespace NUMINAMATH_CALUDE_tank_fill_time_with_leak_l3782_378261

/-- Given a tank and two processes:
    1. Pipe A that can fill the tank in 6 hours
    2. A leak that can empty the full tank in 15 hours
    This theorem proves that it takes 10 hours for Pipe A to fill the tank with the leak present. -/
theorem tank_fill_time_with_leak (tank : ℝ) (pipe_a_rate : ℝ) (leak_rate : ℝ) : 
  pipe_a_rate = 1 / 6 →
  leak_rate = 1 / 15 →
  (pipe_a_rate - leak_rate)⁻¹ = 10 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_with_leak_l3782_378261


namespace NUMINAMATH_CALUDE_wong_valentines_l3782_378229

/-- The number of Valentines Mrs. Wong initially had -/
def initial_valentines : ℕ := 30

/-- The number of Valentines Mrs. Wong gave to her children -/
def given_valentines : ℕ := 8

/-- The number of Valentines Mrs. Wong has left -/
def remaining_valentines : ℕ := initial_valentines - given_valentines

theorem wong_valentines : remaining_valentines = 22 := by
  sorry

end NUMINAMATH_CALUDE_wong_valentines_l3782_378229


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l3782_378267

theorem square_rectangle_area_relation :
  ∃ (x₁ x₂ : ℝ),
    (x₁ - 5) * (x₁ + 6) = 3 * (x₁ - 4)^2 ∧
    (x₂ - 5) * (x₂ + 6) = 3 * (x₂ - 4)^2 ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l3782_378267


namespace NUMINAMATH_CALUDE_cannot_compare_greening_areas_l3782_378253

-- Define the structure for a city
structure City where
  total_area : ℝ
  greening_coverage_rate : ℝ

-- Define the greening coverage area
def greening_coverage_area (city : City) : ℝ :=
  city.total_area * city.greening_coverage_rate

-- Theorem statement
theorem cannot_compare_greening_areas (city_a city_b : City)
  (h_a : city_a.greening_coverage_rate = 0.10)
  (h_b : city_b.greening_coverage_rate = 0.08) :
  ¬ (∀ (a b : City), a.greening_coverage_rate > b.greening_coverage_rate →
    greening_coverage_area a > greening_coverage_area b) :=
by sorry

end NUMINAMATH_CALUDE_cannot_compare_greening_areas_l3782_378253


namespace NUMINAMATH_CALUDE_solution_sum_l3782_378271

theorem solution_sum (x y : ℤ) : 
  (x : ℝ) * Real.log 27 * (Real.log 13)⁻¹ = 27 * Real.log y / Real.log 13 →
  y > 70 →
  ∀ z, z > 70 → z < y →
  x + y = 117 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l3782_378271


namespace NUMINAMATH_CALUDE_circle_equation_l3782_378241

-- Define the point P
def P : ℝ × ℝ := (-2, 1)

-- Define the line y = x + 1
def line1 (x y : ℝ) : Prop := y = x + 1

-- Define the line 3x + 4y - 11 = 0
def line2 (x y : ℝ) : Prop := 3*x + 4*y - 11 = 0

-- Define the circle C
def circle_C (c : ℝ × ℝ) (x y : ℝ) : Prop :=
  (x - c.1)^2 + (y - c.2)^2 = 18

-- Define the symmetry condition
def symmetric_point (p1 p2 : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), line1 x y ∧ 
  (p1.1 + p2.1) / 2 = x ∧ 
  (p1.2 + p2.2) / 2 = y

-- Define the intersection condition
def intersects (c : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), 
  circle_C c A.1 A.2 ∧ 
  circle_C c B.1 B.2 ∧
  line2 A.1 A.2 ∧ 
  line2 B.1 B.2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36

-- Theorem statement
theorem circle_equation (c : ℝ × ℝ) :
  symmetric_point P c ∧ intersects c →
  ∀ (x y : ℝ), circle_C c x y ↔ x^2 + (y+1)^2 = 18 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3782_378241


namespace NUMINAMATH_CALUDE_ln_inequality_l3782_378278

theorem ln_inequality (x y a b : ℝ) 
  (hx : 0 < x) (hy : x < y) (hy1 : y < 1)
  (hb : 1 < b) (ha : b < a) : 
  (Real.log x) / b < (Real.log y) / a :=
sorry

end NUMINAMATH_CALUDE_ln_inequality_l3782_378278


namespace NUMINAMATH_CALUDE_work_increase_with_absence_l3782_378236

theorem work_increase_with_absence (p : ℕ) (W : ℝ) (h : p > 0) :
  let original_work := W / p
  let remaining_workers := (3 : ℝ) / 4 * p
  let new_work := W / remaining_workers
  new_work - original_work = (1 : ℝ) / 3 * original_work :=
by sorry

end NUMINAMATH_CALUDE_work_increase_with_absence_l3782_378236


namespace NUMINAMATH_CALUDE_contractor_payment_l3782_378220

/-- A contractor's payment calculation --/
theorem contractor_payment
  (total_days : ℕ)
  (work_pay : ℚ)
  (fine : ℚ)
  (absent_days : ℕ)
  (h1 : total_days = 30)
  (h2 : work_pay = 25)
  (h3 : fine = 7.5)
  (h4 : absent_days = 4)
  : (total_days - absent_days : ℚ) * work_pay - (absent_days : ℚ) * fine = 620 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_l3782_378220


namespace NUMINAMATH_CALUDE_infinite_k_no_prime_sequence_l3782_378247

theorem infinite_k_no_prime_sequence :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ k ∈ S, ∃ (x : ℕ → ℕ),
      x 1 = 1 ∧
      x 2 = k + 2 ∧
      (∀ n, x (n + 2) = (k + 1) * x (n + 1) - x n) ∧
      ∀ n, ¬ Nat.Prime (x n) :=
sorry

end NUMINAMATH_CALUDE_infinite_k_no_prime_sequence_l3782_378247


namespace NUMINAMATH_CALUDE_class_size_problem_l3782_378228

theorem class_size_problem (total : ℕ) : 
  (total / 3 : ℚ) + 26 = total → total = 39 :=
by sorry

end NUMINAMATH_CALUDE_class_size_problem_l3782_378228


namespace NUMINAMATH_CALUDE_quadratic_no_intersection_l3782_378249

/-- A quadratic function that doesn't intersect the x-axis has c > 1 -/
theorem quadratic_no_intersection (c : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + c ≠ 0) → c > 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_intersection_l3782_378249


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l3782_378240

-- Define a decreasing function f
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- Define the function f passing through specific points
def f_passes_through (f : ℝ → ℝ) : Prop :=
  f 0 = 3 ∧ f 3 = -1

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | |f (x + 1) - 1| < 2}

theorem solution_set_is_open_interval
  (f : ℝ → ℝ)
  (h_decreasing : is_decreasing f)
  (h_passes_through : f_passes_through f) :
  solution_set f = Set.Ioo (-1 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l3782_378240


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l3782_378224

theorem butterflies_in_garden (initial : ℕ) (flew_away : ℕ) (remaining : ℕ) : 
  initial = 9 → 
  flew_away = initial / 3 → 
  remaining = initial - flew_away → 
  remaining = 6 := by
sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l3782_378224


namespace NUMINAMATH_CALUDE_percent_problem_l3782_378258

theorem percent_problem (x : ℝ) : (24 / x = 30 / 100) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l3782_378258


namespace NUMINAMATH_CALUDE_female_student_fraction_l3782_378270

theorem female_student_fraction :
  ∀ (f m : ℝ),
  f + m = 1 →
  (5/6 : ℝ) * f + (2/3 : ℝ) * m = 0.7333333333333333 →
  f = 0.4 := by
sorry

end NUMINAMATH_CALUDE_female_student_fraction_l3782_378270


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3782_378242

/-- The quadratic function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The vertex of f(x) -/
def vertex : ℝ × ℝ := (1, -1)

theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3782_378242


namespace NUMINAMATH_CALUDE_blue_faces_proportion_l3782_378297

/-- Given a cube of side length n, prove that if one-third of the faces of its unit cubes are blue, then n = 3 -/
theorem blue_faces_proportion (n : ℕ) : n ≥ 1 →
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by sorry

end NUMINAMATH_CALUDE_blue_faces_proportion_l3782_378297


namespace NUMINAMATH_CALUDE_min_value_ab_l3782_378281

theorem min_value_ab (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 4/b = Real.sqrt (a*b)) : 
  a * b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 4/b₀ = Real.sqrt (a₀*b₀) ∧ a₀ * b₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_ab_l3782_378281


namespace NUMINAMATH_CALUDE_meiosis_fertilization_importance_l3782_378293

structure ReproductiveProcess where
  meiosis : Bool
  fertilization : Bool

structure BiologicalImportance where
  chromosome_maintenance : Bool
  organism_biology : Bool

structure GenerationalEffect where
  somatic_cell_chromosomes : Bool
  heredity : Bool
  variation : Bool

/-- Given that meiosis and fertilization are important for maintaining constant
    chromosome numbers in species and crucial for the biology of organisms,
    prove that they are crucial for maintaining constant chromosome numbers in
    somatic cells of successive generations and are important for heredity and variation. -/
theorem meiosis_fertilization_importance
  (process : ReproductiveProcess)
  (importance : BiologicalImportance)
  (h1 : process.meiosis ∧ process.fertilization)
  (h2 : importance.chromosome_maintenance)
  (h3 : importance.organism_biology) :
  ∃ (effect : GenerationalEffect),
    effect.somatic_cell_chromosomes ∧
    effect.heredity ∧
    effect.variation :=
sorry

end NUMINAMATH_CALUDE_meiosis_fertilization_importance_l3782_378293


namespace NUMINAMATH_CALUDE_remaining_balloons_l3782_378291

def initial_balloons : ℕ := 30
def given_balloons : ℕ := 16

theorem remaining_balloons : initial_balloons - given_balloons = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balloons_l3782_378291


namespace NUMINAMATH_CALUDE_three_number_sum_l3782_378244

theorem three_number_sum (a b c : ℝ) 
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : (a + b + c) / 3 = a + 7)
  (h4 : (a + b + c) / 3 = c - 20)
  (h5 : b = 8) : 
  a + b + c = 63 := by sorry

end NUMINAMATH_CALUDE_three_number_sum_l3782_378244


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l3782_378239

theorem greatest_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) :
  (⌊y⌋ - ⌈x⌉ : ℤ) ≤ 6 ∧ ∃ (x' y' : ℝ), 4 < x' ∧ x' < 8 ∧ 8 < y' ∧ y' < 12 ∧ (⌊y'⌋ - ⌈x'⌉ : ℤ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l3782_378239


namespace NUMINAMATH_CALUDE_gcd_840_1764_l3782_378299

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l3782_378299


namespace NUMINAMATH_CALUDE_custom_mult_example_l3782_378232

/-- Custom multiplication operation for fractions -/
def custom_mult (m n p q : ℚ) : ℚ := m * p * (2 * q / n)

/-- Theorem stating that (6/5) * (3/4) = 144/5 under the custom multiplication -/
theorem custom_mult_example : custom_mult 6 5 3 4 = 144 / 5 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_example_l3782_378232


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3782_378208

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - 2*x)}
def N : Set ℝ := {y | ∃ x, y = Real.exp x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 0 < x ∧ x < 1/2} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3782_378208


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_is_two_l3782_378277

theorem sum_of_x_coordinates_is_two :
  let f (x : ℝ) := |x^2 - 4*x + 3|
  let g (x : ℝ) := 7 - 2*x
  ∃ (x₁ x₂ : ℝ), (f x₁ = g x₁) ∧ (f x₂ = g x₂) ∧ (x₁ + x₂ = 2) ∧
    (∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_is_two_l3782_378277


namespace NUMINAMATH_CALUDE_train_length_proof_l3782_378263

-- Define the given parameters
def train_speed : Real := 45 -- km/hr
def platform_length : Real := 180 -- meters
def time_to_pass : Real := 43.2 -- seconds

-- Define the theorem
theorem train_length_proof :
  let speed_ms : Real := train_speed * 1000 / 3600 -- Convert km/hr to m/s
  let total_distance : Real := speed_ms * time_to_pass
  let train_length : Real := total_distance - platform_length
  train_length = 360 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l3782_378263


namespace NUMINAMATH_CALUDE_product_closest_to_2500_l3782_378210

def options : List ℝ := [2500, 2600, 250, 260, 25000]

def product : ℝ := 0.0003125 * 8125312

theorem product_closest_to_2500 : 
  ∀ x ∈ options, |product - 2500| ≤ |product - x| :=
sorry

end NUMINAMATH_CALUDE_product_closest_to_2500_l3782_378210


namespace NUMINAMATH_CALUDE_range_of_a_l3782_378233

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (a : ℝ) : M ⊆ N a → a ∈ Set.Iic (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3782_378233


namespace NUMINAMATH_CALUDE_bill_pot_stacking_l3782_378262

/-- Calculates the total number of pots that can be stacked given the vertical stack size, 
    number of stacks per shelf, and number of shelves. -/
def total_pots (vertical_stack : ℕ) (stacks_per_shelf : ℕ) (num_shelves : ℕ) : ℕ :=
  vertical_stack * stacks_per_shelf * num_shelves

/-- Proves that given the specific conditions of Bill's pot stacking problem, 
    the total number of pots is 60. -/
theorem bill_pot_stacking : total_pots 5 3 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_bill_pot_stacking_l3782_378262


namespace NUMINAMATH_CALUDE_odd_and_even_implies_zero_range_even_function_abs_property_l3782_378295

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem 1: If a function is both odd and even, its range is {0}
theorem odd_and_even_implies_zero_range (f : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_even : IsEven f) : 
  ∀ x, f x = 0 := by sorry

-- Theorem 2: If a function is even, then f(|x|) = f(x)
theorem even_function_abs_property (f : ℝ → ℝ) 
  (h_even : IsEven f) : 
  ∀ x, f (|x|) = f x := by sorry

end NUMINAMATH_CALUDE_odd_and_even_implies_zero_range_even_function_abs_property_l3782_378295


namespace NUMINAMATH_CALUDE_increasing_g_implies_m_bound_l3782_378286

open Real

theorem increasing_g_implies_m_bound (m : ℝ) :
  (∀ x > 2, Monotone (fun x => (x - m) * (exp x - x) - exp x + x^2 + x)) →
  m ≤ (2 * exp 2 + 1) / (exp 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_increasing_g_implies_m_bound_l3782_378286


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3782_378227

-- Define the polynomial
def P (x r s t : ℝ) : ℝ := x^4 + r*x^2 + s*x + t

-- State the theorem
theorem root_sum_theorem (b r s t : ℝ) :
  (P (b - 6) r s t = 0) ∧ 
  (P (b - 5) r s t = 0) ∧ 
  (P (b - 4) r s t = 0) →
  r + t = -61 := by
  sorry


end NUMINAMATH_CALUDE_root_sum_theorem_l3782_378227


namespace NUMINAMATH_CALUDE_evaluate_expression_l3782_378202

theorem evaluate_expression : (4^150 * 9^152) / 6^301 = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3782_378202


namespace NUMINAMATH_CALUDE_a_minus_b_bounds_l3782_378245

theorem a_minus_b_bounds (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) :
  -2 < a - b ∧ a - b < 0 := by sorry

end NUMINAMATH_CALUDE_a_minus_b_bounds_l3782_378245


namespace NUMINAMATH_CALUDE_game_collection_proof_l3782_378215

theorem game_collection_proof (games_from_friend games_from_garage_sale total_good_games : ℕ) :
  let total_games := games_from_friend + games_from_garage_sale
  let non_working_games := total_games - total_good_games
  total_good_games = total_games - non_working_games :=
by
  sorry

end NUMINAMATH_CALUDE_game_collection_proof_l3782_378215
