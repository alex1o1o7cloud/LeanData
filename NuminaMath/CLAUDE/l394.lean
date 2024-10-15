import Mathlib

namespace NUMINAMATH_CALUDE_davids_age_l394_39443

/-- Given the ages of Uncle Bob, Emily, and David, prove David's age --/
theorem davids_age (uncle_bob_age : ℕ) (emily_age : ℕ) (david_age : ℕ) 
  (h1 : uncle_bob_age = 60)
  (h2 : emily_age = 2 * uncle_bob_age / 3)
  (h3 : david_age = emily_age - 10) : 
  david_age = 30 := by
  sorry

#check davids_age

end NUMINAMATH_CALUDE_davids_age_l394_39443


namespace NUMINAMATH_CALUDE_circle_tangent_probability_main_theorem_l394_39418

/-- The probability that two circles have exactly two common tangent lines -/
theorem circle_tangent_probability : Real → Prop := fun p =>
  let r_min : Real := 4
  let r_max : Real := 9
  let circle1_center : Real × Real := (2, -1)
  let circle2_center : Real × Real := (-1, 3)
  let circle1_radius : Real := 2
  let valid_r_min : Real := 3
  let valid_r_max : Real := 7
  p = (valid_r_max - valid_r_min) / (r_max - r_min)

/-- The main theorem -/
theorem main_theorem : circle_tangent_probability (4/5) := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_probability_main_theorem_l394_39418


namespace NUMINAMATH_CALUDE_rhino_weight_theorem_l394_39447

/-- The weight of a full-grown white rhino in pounds -/
def full_grown_white_rhino_weight : ℝ := 5100

/-- The weight of a newborn white rhino in pounds -/
def newborn_white_rhino_weight : ℝ := 150

/-- The weight of a full-grown black rhino in pounds -/
def full_grown_black_rhino_weight : ℝ := 2000

/-- The weight of a newborn black rhino in pounds -/
def newborn_black_rhino_weight : ℝ := 100

/-- The conversion factor from pounds to kilograms -/
def pounds_to_kg : ℝ := 0.453592

/-- The number of full-grown white rhinos -/
def num_full_grown_white : ℕ := 6

/-- The number of newborn white rhinos -/
def num_newborn_white : ℕ := 3

/-- The number of full-grown black rhinos -/
def num_full_grown_black : ℕ := 7

/-- The number of newborn black rhinos -/
def num_newborn_black : ℕ := 4

/-- The total weight of all rhinos in kilograms -/
def total_weight_kg : ℝ :=
  ((num_full_grown_white : ℝ) * full_grown_white_rhino_weight +
   (num_newborn_white : ℝ) * newborn_white_rhino_weight +
   (num_full_grown_black : ℝ) * full_grown_black_rhino_weight +
   (num_newborn_black : ℝ) * newborn_black_rhino_weight) * pounds_to_kg

theorem rhino_weight_theorem :
  total_weight_kg = 20616.436 := by
  sorry

end NUMINAMATH_CALUDE_rhino_weight_theorem_l394_39447


namespace NUMINAMATH_CALUDE_a_pow_b_gt_one_iff_a_minus_one_b_gt_zero_l394_39434

theorem a_pow_b_gt_one_iff_a_minus_one_b_gt_zero 
  (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) : 
  a^b > 1 ↔ (a - 1) * b > 0 := by sorry

end NUMINAMATH_CALUDE_a_pow_b_gt_one_iff_a_minus_one_b_gt_zero_l394_39434


namespace NUMINAMATH_CALUDE_invitation_methods_count_l394_39402

-- Define the total number of students
def total_students : ℕ := 10

-- Define the number of students to be invited
def invited_students : ℕ := 6

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem invitation_methods_count :
  combination total_students invited_students - combination (total_students - 2) (invited_students - 2) = 140 := by
  sorry

end NUMINAMATH_CALUDE_invitation_methods_count_l394_39402


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l394_39426

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) : 
  i ^ 2 = -1 →
  z * (2 - i) = i ^ 3 →
  z.im = -2/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l394_39426


namespace NUMINAMATH_CALUDE_point_D_in_fourth_quadrant_l394_39482

/-- Definition of a point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point D -/
def point_D : Point :=
  { x := 6, y := -7 }

/-- Theorem: point D is in the fourth quadrant -/
theorem point_D_in_fourth_quadrant : fourth_quadrant point_D := by
  sorry

end NUMINAMATH_CALUDE_point_D_in_fourth_quadrant_l394_39482


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l394_39489

/-- Calculates the perimeter of the non-shaded region in a composite figure --/
theorem non_shaded_perimeter (outer_length outer_width side_length side_width shaded_area : ℝ) : 
  outer_length = 10 →
  outer_width = 8 →
  side_length = 2 →
  side_width = 4 →
  shaded_area = 78 →
  let total_area := outer_length * outer_width + side_length * side_width
  let non_shaded_area := total_area - shaded_area
  let non_shaded_width := side_length
  let non_shaded_length := non_shaded_area / non_shaded_width
  2 * (non_shaded_length + non_shaded_width) = 14 :=
by sorry


end NUMINAMATH_CALUDE_non_shaded_perimeter_l394_39489


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l394_39421

/-- Given a rectangle formed by 2 rows and 3 columns of identical squares with a total area of 150 cm²,
    prove that its perimeter is 50 cm. -/
theorem rectangle_perimeter (total_area : ℝ) (num_squares : ℕ) (rows cols : ℕ) :
  total_area = 150 ∧ 
  num_squares = 6 ∧ 
  rows = 2 ∧ 
  cols = 3 →
  (2 * rows + 2 * cols) * Real.sqrt (total_area / num_squares) = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l394_39421


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l394_39473

-- Define the type for planes
variable (Plane : Type)

-- Define the relations for parallel and perpendicular planes
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (α β γ : Plane) 
  (h1 : parallel α β) 
  (h2 : perpendicular β γ) : 
  perpendicular α γ :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l394_39473


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l394_39490

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 4*i) / (1 + i) = -1/2 - 7/2*i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l394_39490


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l394_39465

theorem contrapositive_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, ¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l394_39465


namespace NUMINAMATH_CALUDE_distinct_students_count_l394_39492

/-- The number of distinct students taking the math contest at Euclid Middle School -/
def distinct_students : ℕ := by
  -- Define the number of students in each class
  let gauss_class : ℕ := 12
  let euler_class : ℕ := 10
  let fibonnaci_class : ℕ := 7
  
  -- Define the number of students counted twice
  let double_counted : ℕ := 1
  
  -- Calculate the total number of distinct students
  exact gauss_class + euler_class + fibonnaci_class - double_counted

/-- Theorem stating that the number of distinct students taking the contest is 28 -/
theorem distinct_students_count : distinct_students = 28 := by
  sorry

end NUMINAMATH_CALUDE_distinct_students_count_l394_39492


namespace NUMINAMATH_CALUDE_parabola_properties_l394_39440

-- Define the parabola function
def f (x : ℝ) : ℝ := (x + 2)^2 - 1

-- State the theorem
theorem parabola_properties :
  (∀ x y : ℝ, f x ≤ f y → (x + 2)^2 ≤ (y + 2)^2) ∧ -- Opens upwards
  (∀ x : ℝ, f ((-2) + x) = f ((-2) - x)) ∧ -- Axis of symmetry is x = -2
  (∀ x₁ x₂ : ℝ, x₁ > -2 ∧ x₂ > -2 ∧ x₁ < x₂ → f x₁ < f x₂) ∧ -- y increases as x increases when x > -2
  (∀ x : ℝ, f x ≥ f (-2)) ∧ -- Minimum value at x = -2
  (f (-2) = -1) -- Minimum value is -1
  := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l394_39440


namespace NUMINAMATH_CALUDE_square_area_ratio_l394_39444

theorem square_area_ratio (x : ℝ) (hx : x > 0) :
  (x^2) / ((3*x)^2) = 1/9 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l394_39444


namespace NUMINAMATH_CALUDE_bruce_grapes_purchase_l394_39494

theorem bruce_grapes_purchase (grape_price : ℝ) (mango_price : ℝ) (mango_quantity : ℝ) (total_paid : ℝ) :
  grape_price = 70 →
  mango_price = 55 →
  mango_quantity = 10 →
  total_paid = 1110 →
  (total_paid - mango_price * mango_quantity) / grape_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_bruce_grapes_purchase_l394_39494


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l394_39491

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |a * x + 2| < 4 ↔ -1 < x ∧ x < 3) →
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l394_39491


namespace NUMINAMATH_CALUDE_floor_sum_2017_l394_39456

theorem floor_sum_2017 : 
  let floor (x : ℚ) := ⌊x⌋
  ∀ (isPrime2017 : Nat.Prime 2017),
    (floor (2017 * 3 / 11) : ℤ) + 
    (floor (2017 * 4 / 11) : ℤ) + 
    (floor (2017 * 5 / 11) : ℤ) + 
    (floor (2017 * 6 / 11) : ℤ) + 
    (floor (2017 * 7 / 11) : ℤ) + 
    (floor (2017 * 8 / 11) : ℤ) = 6048 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_2017_l394_39456


namespace NUMINAMATH_CALUDE_percentage_decrease_l394_39416

/-- Given a percentage increase P in production value from one year to the next,
    calculate the percentage decrease from the latter year to the former year. -/
theorem percentage_decrease (P : ℝ) : 
  P > -100 → (100 * (1 - 1 / (1 + P / 100))) = P / (1 + P / 100) := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_l394_39416


namespace NUMINAMATH_CALUDE_interest_calculation_l394_39435

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_calculation :
  let principal : ℝ := 10000
  let rate : ℝ := 0.05
  let time : ℝ := 1
  simple_interest principal rate time = 500 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l394_39435


namespace NUMINAMATH_CALUDE_trig_identity_l394_39498

theorem trig_identity (α β : Real) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l394_39498


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l394_39468

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (edge_sum : 4 * a + 4 * b + 4 * c = 160) 
  (diagonal : a^2 + b^2 + c^2 = 25^2) : 
  2 * (a * b + b * c + c * a) = 975 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l394_39468


namespace NUMINAMATH_CALUDE_factorization_equality_l394_39475

theorem factorization_equality (a x y : ℝ) : a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l394_39475


namespace NUMINAMATH_CALUDE_paris_travel_distance_l394_39464

theorem paris_travel_distance (total_distance train_distance bus_distance cab_distance : ℝ) : 
  total_distance = 500 ∧
  bus_distance = train_distance / 2 ∧
  cab_distance = bus_distance / 3 ∧
  total_distance = train_distance + bus_distance + cab_distance →
  train_distance = 300 := by
sorry

end NUMINAMATH_CALUDE_paris_travel_distance_l394_39464


namespace NUMINAMATH_CALUDE_inequality_proof_l394_39476

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_eq_four : a + b + c + d = 4) : 
  (a*b + c*d) * (a*c + b*d) * (a*d + b*c) ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l394_39476


namespace NUMINAMATH_CALUDE_rectangle_width_l394_39420

/-- A rectangle with length twice its width and perimeter equal to its area has width 3. -/
theorem rectangle_width (w : ℝ) (h1 : w > 0) : 
  (6 * w = 2 * w ^ 2) → w = 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_l394_39420


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l394_39439

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | (x+2)*(4-x) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a+1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) : B ∪ C a = B → a ∈ Set.Icc (-2) 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l394_39439


namespace NUMINAMATH_CALUDE_one_third_of_product_l394_39425

theorem one_third_of_product : (1 / 3 : ℚ) * 7 * 9 * 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_product_l394_39425


namespace NUMINAMATH_CALUDE_blue_notes_under_red_l394_39480

theorem blue_notes_under_red (red_rows : Nat) (red_per_row : Nat) (additional_blue : Nat) (total_notes : Nat) : Nat :=
  let total_red := red_rows * red_per_row
  let total_blue := total_notes - total_red
  let blue_under_red := (total_blue - additional_blue) / total_red
  blue_under_red

#check blue_notes_under_red 5 6 10 100

end NUMINAMATH_CALUDE_blue_notes_under_red_l394_39480


namespace NUMINAMATH_CALUDE_marcella_shoes_theorem_l394_39408

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_pairs_remaining (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

/-- Theorem: Given 25 initial pairs of shoes and losing 9 individual shoes,
    the maximum number of complete pairs remaining is 16. -/
theorem marcella_shoes_theorem :
  max_pairs_remaining 25 9 = 16 := by
  sorry

#eval max_pairs_remaining 25 9

end NUMINAMATH_CALUDE_marcella_shoes_theorem_l394_39408


namespace NUMINAMATH_CALUDE_uninsured_employees_count_l394_39437

theorem uninsured_employees_count 
  (total : ℕ) 
  (part_time : ℕ) 
  (uninsured_part_time_ratio : ℚ) 
  (neither_uninsured_nor_part_time_prob : ℚ) 
  (h1 : total = 340)
  (h2 : part_time = 54)
  (h3 : uninsured_part_time_ratio = 125 / 1000)
  (h4 : neither_uninsured_nor_part_time_prob = 5735294117647058 / 10000000000000000) :
  ∃ uninsured : ℕ, uninsured = 104 := by
  sorry


end NUMINAMATH_CALUDE_uninsured_employees_count_l394_39437


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l394_39442

theorem min_coach_handshakes (total_handshakes : ℕ) (h : total_handshakes = 465) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_handshakes ∧
  (∀ (m₁ m₂ : ℕ), m₁ + m₂ = n → m₁ + m₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l394_39442


namespace NUMINAMATH_CALUDE_season_games_l394_39405

/-- The number of hockey games in a season -/
def total_games (games_per_month : ℕ) (season_length : ℕ) : ℕ :=
  games_per_month * season_length

/-- Proof that there are 450 hockey games in the season -/
theorem season_games : total_games 25 18 = 450 := by
  sorry

end NUMINAMATH_CALUDE_season_games_l394_39405


namespace NUMINAMATH_CALUDE_percent_of_a_is_4b_l394_39461

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.8 * b) : 
  (4 * b) / a * 100 = 222.22 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_4b_l394_39461


namespace NUMINAMATH_CALUDE_babysitting_hourly_rate_l394_39478

/-- Calculates the hourly rate for babysitting given total expenses, hours worked, and leftover money -/
def calculate_hourly_rate (total_expenses : ℕ) (hours_worked : ℕ) (leftover : ℕ) : ℚ :=
  (total_expenses + leftover) / hours_worked

/-- Theorem: Given the problem conditions, the hourly rate for babysitting is $8 -/
theorem babysitting_hourly_rate :
  let total_expenses := 65
  let hours_worked := 9
  let leftover := 7
  calculate_hourly_rate total_expenses hours_worked leftover = 8 := by
sorry

end NUMINAMATH_CALUDE_babysitting_hourly_rate_l394_39478


namespace NUMINAMATH_CALUDE_min_diagonal_rectangle_l394_39459

/-- Given a rectangle ABCD with perimeter 30 inches and width w ≥ 6 inches,
    the minimum length of diagonal AC is 7.5√2 inches. -/
theorem min_diagonal_rectangle (l w : ℝ) (h1 : l + w = 15) (h2 : w ≥ 6) :
  ∃ (AC : ℝ), AC = 7.5 * Real.sqrt 2 ∧ ∀ (AC' : ℝ), AC' ≥ AC := by
  sorry

end NUMINAMATH_CALUDE_min_diagonal_rectangle_l394_39459


namespace NUMINAMATH_CALUDE_walking_distance_l394_39454

/-- The distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that walking at 4 miles per hour for 2 hours results in 8 miles traveled -/
theorem walking_distance : distance 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_l394_39454


namespace NUMINAMATH_CALUDE_nabla_example_l394_39481

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 2 + b^a

-- Theorem statement
theorem nabla_example : nabla (nabla 1 2) 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_nabla_example_l394_39481


namespace NUMINAMATH_CALUDE_integer_condition_l394_39427

theorem integer_condition (m k n : ℕ) (h1 : 0 < m) (h2 : 0 < k) (h3 : 0 < n)
  (h4 : k < n - 1) (h5 : m ≤ n) :
  ∃ z : ℤ, (n - 3 * k + m : ℚ) / (k + m : ℚ) * (n.choose k : ℚ) = z ↔ 
  ∃ t : ℕ, 2 * m = t * (k + m) :=
by sorry

end NUMINAMATH_CALUDE_integer_condition_l394_39427


namespace NUMINAMATH_CALUDE_total_amount_proof_l394_39463

/-- Given that r has two-thirds of the total amount and r has Rs. 2400,
    prove that the total amount p, q, and r have among themselves is Rs. 3600. -/
theorem total_amount_proof (r p q : ℕ) (h1 : r = 2400) (h2 : r * 3 = (p + q + r) * 2) :
  p + q + r = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_proof_l394_39463


namespace NUMINAMATH_CALUDE_is_vertex_of_parabola_l394_39470

/-- The parabola equation -/
def f (x : ℝ) : ℝ := -4 * x^2 - 16 * x - 20

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-2, -4)

/-- Theorem stating that the given point is the vertex of the parabola -/
theorem is_vertex_of_parabola :
  ∀ x : ℝ, f x ≤ f (vertex.1) ∧ f (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_is_vertex_of_parabola_l394_39470


namespace NUMINAMATH_CALUDE_test_scores_mode_l394_39409

/-- Represents a stem-and-leaf plot entry -/
structure StemLeafEntry where
  stem : ℕ
  leaves : List ℕ

/-- The stem-and-leaf plot data -/
def testScores : List StemLeafEntry := [
  ⟨4, [5, 5, 5]⟩,
  ⟨5, [2, 6, 6]⟩,
  ⟨6, [1, 3, 3, 3, 3]⟩,
  ⟨7, [2, 4, 5, 5, 5, 5, 5]⟩,
  ⟨8, [0, 3, 6]⟩,
  ⟨9, [1, 1, 4, 7]⟩
]

/-- Convert a stem-leaf entry to a list of full scores -/
def toFullScores (entry : StemLeafEntry) : List ℕ :=
  entry.leaves.map (λ leaf => entry.stem * 10 + leaf)

/-- Find the mode of a list of numbers -/
def mode (numbers : List ℕ) : ℕ := sorry

/-- The main theorem stating that the mode of the test scores is 75 -/
theorem test_scores_mode :
  mode (testScores.bind toFullScores) = 75 := by sorry

end NUMINAMATH_CALUDE_test_scores_mode_l394_39409


namespace NUMINAMATH_CALUDE_complex_power_195_deg_60_l394_39485

theorem complex_power_195_deg_60 :
  (Complex.exp (195 * π / 180 * Complex.I)) ^ 60 = (1 / 2 : ℂ) - Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_195_deg_60_l394_39485


namespace NUMINAMATH_CALUDE_expression_value_l394_39499

theorem expression_value (a b c : ℝ) (ha : a ≠ 3) (hb : b ≠ 4) (hc : c ≠ 5) :
  (a - 3) / (5 - c) * (b - 4) / (3 - a) * (c - 5) / (4 - b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l394_39499


namespace NUMINAMATH_CALUDE_three_digit_squares_ending_with_self_l394_39401

theorem three_digit_squares_ending_with_self (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) ∧ (A^2 ≡ A [ZMOD 1000]) ↔ A = 376 ∨ A = 625 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_squares_ending_with_self_l394_39401


namespace NUMINAMATH_CALUDE_dans_age_l394_39438

/-- Given two people, Ben and Dan, where Ben is younger than Dan, 
    their ages sum to 53, and Ben is 25 years old, 
    prove that Dan is 28 years old. -/
theorem dans_age (ben_age dan_age : ℕ) : 
  ben_age < dan_age →
  ben_age + dan_age = 53 →
  ben_age = 25 →
  dan_age = 28 := by
  sorry

end NUMINAMATH_CALUDE_dans_age_l394_39438


namespace NUMINAMATH_CALUDE_total_profit_calculation_l394_39403

/-- The total profit of a business partnership given investments and one partner's profit share -/
theorem total_profit_calculation (p_investment q_investment : ℚ) (q_profit_share : ℚ) : 
  p_investment = 54000 →
  q_investment = 36000 →
  q_profit_share = 6001.89 →
  (p_investment + q_investment) / q_investment * q_profit_share = 15004.725 :=
by
  sorry

#eval (54000 + 36000) / 36000 * 6001.89

end NUMINAMATH_CALUDE_total_profit_calculation_l394_39403


namespace NUMINAMATH_CALUDE_total_percent_decrease_l394_39410

def year1_decrease : ℝ := 0.20
def year2_decrease : ℝ := 0.10
def year3_decrease : ℝ := 0.15

def compound_decrease (initial_value : ℝ) : ℝ :=
  initial_value * (1 - year1_decrease) * (1 - year2_decrease) * (1 - year3_decrease)

theorem total_percent_decrease (initial_value : ℝ) (h : initial_value > 0) :
  (initial_value - compound_decrease initial_value) / initial_value = 0.388 := by
  sorry

end NUMINAMATH_CALUDE_total_percent_decrease_l394_39410


namespace NUMINAMATH_CALUDE_partnership_contribution_l394_39497

theorem partnership_contribution 
  (a_capital : ℕ) 
  (a_time : ℕ) 
  (b_time : ℕ) 
  (total_profit : ℕ) 
  (a_profit : ℕ) 
  (h1 : a_capital = 5000)
  (h2 : a_time = 8)
  (h3 : b_time = 5)
  (h4 : total_profit = 8400)
  (h5 : a_profit = 4800) :
  ∃ b_capital : ℕ, 
    (a_capital * a_time : ℚ) / ((a_capital * a_time + b_capital * b_time) : ℚ) = 
    (a_profit : ℚ) / (total_profit : ℚ) ∧ 
    b_capital = 6000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_contribution_l394_39497


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l394_39432

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the interval [1,2)
def interval : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- State the theorem
theorem intersection_equals_interval : M ∩ N = interval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l394_39432


namespace NUMINAMATH_CALUDE_missing_number_proof_l394_39451

theorem missing_number_proof : ∃ x : ℤ, (4 + 3) + (8 - 3 - x) = 11 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l394_39451


namespace NUMINAMATH_CALUDE_iesha_book_count_l394_39428

/-- The number of school books Iesha has -/
def school_books : ℕ := 136

/-- The number of sports books Iesha has -/
def sports_books : ℕ := 208

/-- The total number of books Iesha has -/
def total_books : ℕ := school_books + sports_books

theorem iesha_book_count : total_books = 344 := by
  sorry

end NUMINAMATH_CALUDE_iesha_book_count_l394_39428


namespace NUMINAMATH_CALUDE_units_digit_power_plus_six_l394_39449

theorem units_digit_power_plus_six (x : ℕ) : 
  1 ≤ x → x ≤ 9 → (x^75 + 6) % 10 = 9 → x = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_power_plus_six_l394_39449


namespace NUMINAMATH_CALUDE_inequality_proof_l394_39417

theorem inequality_proof (a b c d : ℝ) 
  (non_neg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (sum_condition : a*b + a*c + a*d + b*c + b*d + c*d = 6) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1) + 1 / (d^2 + 1) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l394_39417


namespace NUMINAMATH_CALUDE_log_6_6_log_2_8_log_equation_l394_39493

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem statements
theorem log_6_6 : log 6 6 = 1 := by sorry

theorem log_2_8 : log 2 8 = 3 := by sorry

theorem log_equation (m : ℝ) : log 2 (m - 2) = 4 → m = 18 := by sorry

end NUMINAMATH_CALUDE_log_6_6_log_2_8_log_equation_l394_39493


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l394_39472

theorem angle_terminal_side_point (α : Real) (m : Real) :
  m > 0 →
  (2 : Real) / Real.sqrt (4 + m^2) = 2 * Real.sqrt 5 / 5 →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l394_39472


namespace NUMINAMATH_CALUDE_arithmetic_computation_l394_39466

theorem arithmetic_computation : (-9 * 5) - (-7 * -2) + (11 * -4) = -103 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l394_39466


namespace NUMINAMATH_CALUDE_polynomial_factorization_l394_39452

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x-1)^4 * (x+1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l394_39452


namespace NUMINAMATH_CALUDE_school_age_problem_l394_39414

theorem school_age_problem (num_students : ℕ) (num_teachers : ℕ) (avg_age_students : ℝ) 
  (avg_age_with_teachers : ℝ) (avg_age_with_principal : ℝ) :
  num_students = 30 →
  num_teachers = 3 →
  avg_age_students = 14 →
  avg_age_with_teachers = 16 →
  avg_age_with_principal = 17 →
  ∃ (total_age_teachers : ℝ) (age_principal : ℝ),
    total_age_teachers = 108 ∧ age_principal = 50 := by
  sorry

end NUMINAMATH_CALUDE_school_age_problem_l394_39414


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l394_39400

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_inequality
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h1 : a 1 = b 1)
  (h1_pos : a 1 > 0)
  (h11 : a 11 = b 11)
  (h11_pos : a 11 > 0) :
  a 6 ≥ b 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l394_39400


namespace NUMINAMATH_CALUDE_shower_frequency_l394_39448

/-- Represents the duration of each shower in minutes -/
def shower_duration : ℝ := 10

/-- Represents the water usage rate in gallons per minute -/
def water_usage_rate : ℝ := 2

/-- Represents the total water usage in 4 weeks in gallons -/
def total_water_usage : ℝ := 280

/-- Represents the number of weeks -/
def num_weeks : ℝ := 4

/-- Theorem stating the frequency of John's showers -/
theorem shower_frequency :
  (total_water_usage / (shower_duration * water_usage_rate)) / num_weeks = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_shower_frequency_l394_39448


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_not_prime_l394_39453

/-- 
Given integers a and b, if the quadratic equation x^2 + ax + b + 1 = 0 
has two positive integer roots, then a^2 + b^2 is not prime.
-/
theorem quadratic_roots_imply_not_prime (a b : ℤ) 
  (h : ∃ p q : ℕ+, p.val ≠ q.val ∧ p.val^2 + a * p.val + b + 1 = 0 ∧ q.val^2 + a * q.val + b + 1 = 0) : 
  ¬ Prime (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_not_prime_l394_39453


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l394_39469

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

theorem collinear_points_x_value :
  ∀ x : ℚ, collinear 2 7 10 x 25 (-2) → x = 89 / 23 :=
by
  sorry

#check collinear_points_x_value

end NUMINAMATH_CALUDE_collinear_points_x_value_l394_39469


namespace NUMINAMATH_CALUDE_carnival_friends_l394_39484

theorem carnival_friends (total_tickets : ℕ) (tickets_per_person : ℕ) (h1 : total_tickets = 234) (h2 : tickets_per_person = 39) :
  total_tickets / tickets_per_person = 6 := by
  sorry

end NUMINAMATH_CALUDE_carnival_friends_l394_39484


namespace NUMINAMATH_CALUDE_christels_initial_dolls_l394_39406

theorem christels_initial_dolls (debelyn_initial : ℕ) (debelyn_gave : ℕ) (christel_gave : ℕ) :
  debelyn_initial = 20 →
  debelyn_gave = 2 →
  christel_gave = 5 →
  ∃ (christel_initial : ℕ) (andrena_final : ℕ),
    andrena_final = debelyn_gave + christel_gave ∧
    andrena_final = (christel_initial - christel_gave) + 2 ∧
    andrena_final = (debelyn_initial - debelyn_gave) + 3 →
    christel_initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_christels_initial_dolls_l394_39406


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l394_39445

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 5 ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 5 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l394_39445


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l394_39457

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def axis_of_symmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) (h : is_even_function (fun x => f (x + 3))) :
  axis_of_symmetry f 3 :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l394_39457


namespace NUMINAMATH_CALUDE_possible_values_of_x_l394_39479

theorem possible_values_of_x (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 225)
  (h3 : x < 30) :
  x = 18 ∨ x = 27 :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_x_l394_39479


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l394_39495

/-- Given two squares with perimeters 20 and 28, prove that a third square with side length
    equal to the positive difference of the side lengths of the first two squares has a perimeter of 8. -/
theorem square_perimeter_problem (square_I square_II square_III : ℝ → ℝ) :
  (∀ s, square_I s = 4 * s) →
  (∀ s, square_II s = 4 * s) →
  (∀ s, square_III s = 4 * s) →
  (∃ s_I, square_I s_I = 20) →
  (∃ s_II, square_II s_II = 28) →
  (∃ s_III, s_III = |s_I - s_II| ∧ square_III s_III = 8) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l394_39495


namespace NUMINAMATH_CALUDE_quadratic_system_solution_l394_39486

theorem quadratic_system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℚ),
    (x₁ = 2/9 ∧ y₁ = 35/117) ∧
    (x₂ = -1 ∧ y₂ = -5/26) ∧
    (∀ x y : ℚ, 9*x^2 + 8*x - 2 = 0 ∧ 27*x^2 + 26*y + 8*x - 14 = 0 →
      (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_system_solution_l394_39486


namespace NUMINAMATH_CALUDE_balloon_difference_l394_39412

theorem balloon_difference (your_balloons friend_balloons : ℝ) 
  (h1 : your_balloons = -7)
  (h2 : friend_balloons = 4.5) :
  friend_balloons - your_balloons = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l394_39412


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l394_39483

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f := λ x : ℝ => a^(x - 2) + 2
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l394_39483


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l394_39441

theorem price_decrease_percentage (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := (4 / 5 : ℝ) * original_price
  let second_sale_price := (1 / 2 : ℝ) * original_price
  let price_difference := first_sale_price - second_sale_price
  let percentage_decrease := (price_difference / first_sale_price) * 100
  percentage_decrease = 37.5 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l394_39441


namespace NUMINAMATH_CALUDE_more_green_than_blue_l394_39458

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : Nat
  ratio : Fin 3 → Nat
  sum_ratio : ratio 0 + ratio 1 + ratio 2 = 18

theorem more_green_than_blue (bag : DiskBag) 
  (h_total : bag.total = 54)
  (h_ratio : bag.ratio = ![3, 7, 8]) :
  (bag.total * bag.ratio 2) / 18 - (bag.total * bag.ratio 0) / 18 = 15 := by
  sorry

#check more_green_than_blue

end NUMINAMATH_CALUDE_more_green_than_blue_l394_39458


namespace NUMINAMATH_CALUDE_complex_midpoint_and_distance_l394_39407

theorem complex_midpoint_and_distance (z₁ z₂ m : ℂ) (h₁ : z₁ = -7 + 5*I) (h₂ : z₂ = 9 - 11*I) 
  (h_m : m = (z₁ + z₂) / 2) : 
  m = 1 - 3*I ∧ Complex.abs (z₁ - m) = 8*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_midpoint_and_distance_l394_39407


namespace NUMINAMATH_CALUDE_even_periodic_function_l394_39467

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_period_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 2)

theorem even_periodic_function 
  (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : is_period_two f) 
  (h3 : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-1) 0, f x = 2 - x :=
sorry

end NUMINAMATH_CALUDE_even_periodic_function_l394_39467


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l394_39423

theorem arithmetic_calculation : 1323 + 150 / 50 * 3 - 223 = 1109 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l394_39423


namespace NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l394_39422

theorem longest_segment_in_quarter_circle (d : ℝ) (h : d = 10) :
  let r := d / 2
  let m := r * Real.sqrt 2
  m ^ 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l394_39422


namespace NUMINAMATH_CALUDE_six_by_six_checkerboard_half_shaded_l394_39429

/-- Represents a square grid with checkerboard shading -/
structure CheckerboardGrid :=
  (size : ℕ)
  (startUnshaded : Bool)

/-- Calculates the fraction of shaded squares in a checkerboard grid -/
def shadedFraction (grid : CheckerboardGrid) : ℚ :=
  1/2

/-- Theorem: In a 6x6 checkerboard grid starting with an unshaded square,
    half of the squares are shaded -/
theorem six_by_six_checkerboard_half_shaded :
  let grid : CheckerboardGrid := ⟨6, true⟩
  shadedFraction grid = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_six_by_six_checkerboard_half_shaded_l394_39429


namespace NUMINAMATH_CALUDE_rectangle_ratio_l394_39477

/-- Proves that for a rectangle with width 5 inches and area 50 square inches, 
    the ratio of its length to its width is 2. -/
theorem rectangle_ratio : 
  ∀ (length width : ℝ), 
    width = 5 → 
    length * width = 50 → 
    length / width = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l394_39477


namespace NUMINAMATH_CALUDE_line1_passes_through_points_line2_satisfies_conditions_l394_39446

-- Define the points
def point1 : ℝ × ℝ := (2, 1)
def point2 : ℝ × ℝ := (0, -3)
def point3 : ℝ × ℝ := (0, 5)

-- Define the sum of intercepts
def sum_of_intercepts : ℝ := 2

-- Define the equations of the lines
def line1_equation (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2_equation (x y : ℝ) : Prop := 5 * x - 3 * y + 15 = 0

-- Theorem for the first line
theorem line1_passes_through_points :
  line1_equation point1.1 point1.2 ∧ line1_equation point2.1 point2.2 :=
sorry

-- Theorem for the second line
theorem line2_satisfies_conditions :
  line2_equation point3.1 point3.2 ∧
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a + b = sum_of_intercepts ∧
  ∀ x y : ℝ, line2_equation x y ↔ x / a + y / b = 1) :=
sorry

end NUMINAMATH_CALUDE_line1_passes_through_points_line2_satisfies_conditions_l394_39446


namespace NUMINAMATH_CALUDE_constant_function_l394_39424

theorem constant_function (a : ℝ) (f : ℝ → ℝ) 
  (h1 : f 0 = (1 : ℝ) / 2)
  (h2 : ∀ x y : ℝ, f (x + y) = f x * f (a - y) + f y * f (a - x)) :
  ∀ x : ℝ, f x = (1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_constant_function_l394_39424


namespace NUMINAMATH_CALUDE_second_task_end_time_l394_39488

-- Define the start and end times in minutes since midnight
def start_time : Nat := 8 * 60  -- 8:00 AM
def end_time : Nat := 12 * 60 + 20  -- 12:20 PM

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem to prove
theorem second_task_end_time :
  let total_duration : Nat := end_time - start_time
  let task_duration : Nat := total_duration / num_tasks
  let second_task_end : Nat := start_time + 2 * task_duration
  second_task_end = 10 * 60 + 10  -- 10:10 AM
  := by sorry

end NUMINAMATH_CALUDE_second_task_end_time_l394_39488


namespace NUMINAMATH_CALUDE_line_passes_through_134_iff_a_gt_third_l394_39455

/-- A line passes through the first, third, and fourth quadrants if and only if its slope is positive -/
axiom passes_through_134_iff_positive_slope (m : ℝ) (b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) ↔ m > 0

/-- The main theorem: the line y = (3a-1)x - 1 passes through the first, third, and fourth quadrants
    if and only if a > 1/3 -/
theorem line_passes_through_134_iff_a_gt_third (a : ℝ) : 
  (∀ x y : ℝ, y = (3*a - 1) * x - 1 → 
    (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) ↔ a > 1/3 :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_134_iff_a_gt_third_l394_39455


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l394_39430

/-- The number of bouncy balls in each pack -/
def ballsPerPack : ℕ := 10

/-- The number of packs of red bouncy balls -/
def redPacks : ℕ := 4

/-- The number of packs of yellow bouncy balls -/
def yellowPacks : ℕ := 8

/-- The number of packs of green bouncy balls -/
def greenPacks : ℕ := 4

/-- The total number of bouncy balls Maggie bought -/
def totalBalls : ℕ := ballsPerPack * (redPacks + yellowPacks + greenPacks)

theorem maggie_bouncy_balls : totalBalls = 160 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l394_39430


namespace NUMINAMATH_CALUDE_rhombus_tangent_distance_l394_39404

/-- A rhombus with an inscribed circle -/
structure RhombusWithCircle where
  /-- Side length of the rhombus -/
  side : ℝ
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- Condition that the first diagonal is less than the second diagonal -/
  diag_condition : ℝ → ℝ → Prop

/-- The distance between tangent points on adjacent sides of the rhombus -/
def tangent_distance (r : RhombusWithCircle) : ℝ := sorry

/-- Theorem stating the distance between tangent points on adjacent sides -/
theorem rhombus_tangent_distance
  (r : RhombusWithCircle)
  (h1 : r.side = 5)
  (h2 : r.radius = 2.4)
  (h3 : r.diag_condition (2 * r.radius) (2 * r.side * (1 - r.radius / r.side))) :
  tangent_distance r = 3.84 := by sorry

end NUMINAMATH_CALUDE_rhombus_tangent_distance_l394_39404


namespace NUMINAMATH_CALUDE_mistake_percentage_l394_39411

theorem mistake_percentage (n : ℕ) (x : ℕ) : 
  n > 0 ∧ x > 0 ∧ x ≤ n ∧
  (x - 1 : ℚ) / n = 24 / 100 ∧
  (x - 1 : ℚ) / (n - 1) = 25 / 100 →
  (x : ℚ) / n = 28 / 100 :=
by sorry

end NUMINAMATH_CALUDE_mistake_percentage_l394_39411


namespace NUMINAMATH_CALUDE_value_of_N_l394_39431

theorem value_of_N : ∃ N : ℝ, (25 / 100) * (N + 100) = (35 / 100) * 1500 ∧ N = 2000 := by
  sorry

end NUMINAMATH_CALUDE_value_of_N_l394_39431


namespace NUMINAMATH_CALUDE_percentage_decrease_in_people_l394_39436

/-- Calculates the percentage decrease in the number of people to be fed given initial and new can counts. -/
theorem percentage_decrease_in_people (initial_cans initial_people new_cans : ℕ) : 
  initial_cans = 600 →
  initial_people = 40 →
  new_cans = 420 →
  (1 - (new_cans * initial_people : ℚ) / (initial_cans * initial_people)) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_decrease_in_people_l394_39436


namespace NUMINAMATH_CALUDE_matrix_self_inverse_l394_39415

theorem matrix_self_inverse (a b : ℚ) :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -2; a, b]
  A * A = 1 → a = 7.5 ∧ b = -4 := by
sorry

end NUMINAMATH_CALUDE_matrix_self_inverse_l394_39415


namespace NUMINAMATH_CALUDE_carpet_cost_per_meter_l394_39413

/-- Proves that the cost per meter of carpet is 30 paise given the room dimensions, carpet width, and total cost. -/
theorem carpet_cost_per_meter (room_length : ℝ) (room_width : ℝ) (carpet_width : ℝ) (total_cost : ℝ) :
  room_length = 15 →
  room_width = 6 →
  carpet_width = 0.75 →
  total_cost = 36 →
  (total_cost * 100) / (room_length * room_width / carpet_width) = 30 := by
  sorry

#check carpet_cost_per_meter

end NUMINAMATH_CALUDE_carpet_cost_per_meter_l394_39413


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l394_39474

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on the asymptote of a hyperbola -/
def onAsymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y = (h.b / h.a) * p.x ∨ p.y = -(h.b / h.a) * p.x

/-- Calculates the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Calculates the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem -/
theorem hyperbola_eccentricity_theorem (h : Hyperbola) (a1 a2 b : Point) :
  a1.x = -h.a ∧ a1.y = 0 ∧  -- A₁ is left vertex
  a2.x = h.a ∧ a2.y = 0 ∧   -- A₂ is right vertex
  onAsymptote h b ∧         -- B is on asymptote
  angle a1 b a2 = π/3 →     -- ∠A₁BA₂ = 60°
  eccentricity h = Real.sqrt 21 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l394_39474


namespace NUMINAMATH_CALUDE_sum_longest_altitudes_is_21_l394_39462

/-- A triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 9
  hb : b = 12
  hc : c = 15
  right_angle : a^2 + b^2 = c^2

/-- The sum of the lengths of the two longest altitudes in the right triangle -/
def sum_longest_altitudes (t : RightTriangle) : ℝ := t.a + t.b

/-- Theorem stating that the sum of the two longest altitudes is 21 -/
theorem sum_longest_altitudes_is_21 (t : RightTriangle) :
  sum_longest_altitudes t = 21 := by sorry

end NUMINAMATH_CALUDE_sum_longest_altitudes_is_21_l394_39462


namespace NUMINAMATH_CALUDE_rectangle_area_change_l394_39433

theorem rectangle_area_change (L W : ℝ) (h : L * W = 540) :
  (0.8 * L) * (1.2 * W) = 518.4 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l394_39433


namespace NUMINAMATH_CALUDE_second_box_price_l394_39450

/-- Represents a box of contacts with its quantity and price -/
structure ContactBox where
  quantity : ℕ
  price : ℚ

/-- Calculates the price per contact for a given box -/
def pricePerContact (box : ContactBox) : ℚ :=
  box.price / box.quantity

theorem second_box_price (box1 box2 : ContactBox)
  (h1 : box1.quantity = 50)
  (h2 : box1.price = 25)
  (h3 : box2.quantity = 99)
  (h4 : pricePerContact box2 < pricePerContact box1)
  (h5 : 3 * pricePerContact box2 = 1) :
  box2.price = 99/3 := by
  sorry

#eval (99 : ℚ) / 3  -- Should output 33

end NUMINAMATH_CALUDE_second_box_price_l394_39450


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l394_39487

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l394_39487


namespace NUMINAMATH_CALUDE_simplify_fraction_l394_39496

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) :
  (2 / (a^2 - 1)) * (1 / (a - 1)) = 2 / ((a - 1)^2 * (a + 1)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l394_39496


namespace NUMINAMATH_CALUDE_distance_to_left_focus_l394_39419

-- Define the ellipse
def is_on_ellipse (x y b : ℝ) : Prop :=
  x^2 / 25 + y^2 / b^2 = 1

-- Define the condition for b
def valid_b (b : ℝ) : Prop :=
  0 < b ∧ b < 5

-- Define a point P on the ellipse
def P_on_ellipse (P : ℝ × ℝ) (b : ℝ) : Prop :=
  is_on_ellipse P.1 P.2 b

-- Define the left focus F₁
def F₁ : ℝ × ℝ := sorry

-- Define the condition |OP⃗ + OF₁⃗| = 8
def vector_sum_condition (P : ℝ × ℝ) : Prop :=
  ‖P + F₁‖ = 8

-- Theorem statement
theorem distance_to_left_focus
  (b : ℝ)
  (P : ℝ × ℝ)
  (h_b : valid_b b)
  (h_P : P_on_ellipse P b)
  (h_sum : vector_sum_condition P) :
  ‖P - F₁‖ = 2 :=
sorry

end NUMINAMATH_CALUDE_distance_to_left_focus_l394_39419


namespace NUMINAMATH_CALUDE_some_ounce_size_is_eight_l394_39460

/-- The size of the some-ounce glasses -/
def some_ounce_size : ℕ := sorry

/-- The total amount of water available -/
def total_water : ℕ := 122

/-- The number of 5-ounce glasses filled -/
def five_ounce_glasses : ℕ := 6

/-- The number of some-ounce glasses filled -/
def some_ounce_glasses : ℕ := 4

/-- The number of 4-ounce glasses that can be filled with remaining water -/
def four_ounce_glasses : ℕ := 15

/-- Theorem stating that the size of the some-ounce glasses is 8 ounces -/
theorem some_ounce_size_is_eight :
  some_ounce_size = 8 ∧
  total_water = 
    five_ounce_glasses * 5 + 
    some_ounce_glasses * some_ounce_size + 
    four_ounce_glasses * 4 :=
by sorry

end NUMINAMATH_CALUDE_some_ounce_size_is_eight_l394_39460


namespace NUMINAMATH_CALUDE_trigonometric_values_l394_39471

theorem trigonometric_values : 
  (Real.sin (30 * π / 180) = 1 / 2) ∧ 
  (Real.cos (11 * π / 4) = -Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_values_l394_39471
