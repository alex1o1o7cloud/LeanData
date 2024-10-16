import Mathlib

namespace NUMINAMATH_CALUDE_joan_books_count_l1466_146661

/-- The number of books Joan sold in the yard sale -/
def books_sold : ℕ := 26

/-- The number of books Joan has left after the sale -/
def books_left : ℕ := 7

/-- The total number of books Joan gathered to sell -/
def total_books : ℕ := books_sold + books_left

theorem joan_books_count : total_books = 33 := by sorry

end NUMINAMATH_CALUDE_joan_books_count_l1466_146661


namespace NUMINAMATH_CALUDE_sandys_comic_books_l1466_146664

theorem sandys_comic_books (initial : ℕ) : 
  (initial / 2 + 6 = 13) → initial = 14 := by
  sorry

end NUMINAMATH_CALUDE_sandys_comic_books_l1466_146664


namespace NUMINAMATH_CALUDE_multiply_and_add_l1466_146600

theorem multiply_and_add : 45 * 28 + 45 * 72 + 45 = 4545 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_l1466_146600


namespace NUMINAMATH_CALUDE_constant_jump_returns_to_start_increasing_jump_returns_to_start_l1466_146644

-- Define the number of stones
def num_stones : ℕ := 10

-- Define the number of jumps
def num_jumps : ℕ := 100

-- Function to calculate the position after constant jumps
def constant_jump_position (jump_size : ℕ) : ℕ :=
  (1 + jump_size * num_jumps) % num_stones

-- Function to calculate the position after increasing jumps
def increasing_jump_position : ℕ :=
  (1 + (num_jumps * (num_jumps + 1) / 2)) % num_stones

-- Theorem for constant jump scenario
theorem constant_jump_returns_to_start :
  constant_jump_position 2 = 1 := by sorry

-- Theorem for increasing jump scenario
theorem increasing_jump_returns_to_start :
  increasing_jump_position = 1 := by sorry

end NUMINAMATH_CALUDE_constant_jump_returns_to_start_increasing_jump_returns_to_start_l1466_146644


namespace NUMINAMATH_CALUDE_range_of_a_l1466_146689

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x > 0, x + 1/x > a
def q (a : ℝ) : Prop := ∃ x, x^2 - 2*a*x + 1 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(¬(q a))) → 
  (¬(p a ∧ q a)) → 
  ((q a) → (a ≤ -1 ∨ a ≥ 1)) →
  ((¬(p a)) → a ≥ 2) →
  a ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1466_146689


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1466_146628

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def arrangements_with_four_consecutive (n : ℕ) : ℕ :=
  (Nat.factorial (n - 3)) * (Nat.factorial 4)

theorem seating_arrangements_with_restriction (n : ℕ) (k : ℕ) 
  (h1 : n = 10) (h2 : k = 4) : 
  total_arrangements n - arrangements_with_four_consecutive n = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1466_146628


namespace NUMINAMATH_CALUDE_prism_18_edges_8_faces_l1466_146640

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism. -/
def num_faces (p : Prism) : ℕ :=
  let lateral_faces := p.edges / 3
  lateral_faces + 2

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_8_faces :
  ∀ (p : Prism), p.edges = 18 → num_faces p = 8 := by
  sorry

#check prism_18_edges_8_faces

end NUMINAMATH_CALUDE_prism_18_edges_8_faces_l1466_146640


namespace NUMINAMATH_CALUDE_total_items_proof_l1466_146660

def days : ℕ := 10

def pebble_sequence (n : ℕ) : ℕ := n

def seashell_sequence (n : ℕ) : ℕ := 2 * n - 1

def total_items : ℕ := (days * (pebble_sequence 1 + pebble_sequence days)) / 2 +
                       (days * (seashell_sequence 1 + seashell_sequence days)) / 2

theorem total_items_proof : total_items = 155 := by
  sorry

end NUMINAMATH_CALUDE_total_items_proof_l1466_146660


namespace NUMINAMATH_CALUDE_syllogism_invalid_l1466_146694

-- Define the sets and properties
def Geese : Type := Unit
def Senators : Type := Unit
def eats_cabbage (α : Type) : α → Prop := fun _ => True

-- Define the syllogism
def invalid_syllogism (g : Geese) (s : Senators) : Prop :=
  eats_cabbage Geese g ∧ eats_cabbage Senators s → s = g

-- Theorem stating that the syllogism is invalid
theorem syllogism_invalid :
  ¬∀ (g : Geese) (s : Senators), invalid_syllogism g s :=
sorry

end NUMINAMATH_CALUDE_syllogism_invalid_l1466_146694


namespace NUMINAMATH_CALUDE_a_worked_six_days_l1466_146657

/-- Represents the number of days worked by person a -/
def days_a : ℕ := sorry

/-- Represents the daily wage of person a -/
def wage_a : ℕ := sorry

/-- Represents the daily wage of person b -/
def wage_b : ℕ := sorry

/-- Represents the daily wage of person c -/
def wage_c : ℕ := sorry

/-- The theorem stating that person a worked for 6 days given the conditions -/
theorem a_worked_six_days :
  wage_c = 105 ∧
  wage_a / wage_b = 3 / 4 ∧
  wage_b / wage_c = 4 / 5 ∧
  days_a * wage_a + 9 * wage_b + 4 * wage_c = 1554 →
  days_a = 6 := by sorry

end NUMINAMATH_CALUDE_a_worked_six_days_l1466_146657


namespace NUMINAMATH_CALUDE_number_of_factors_of_60_l1466_146629

theorem number_of_factors_of_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_60_l1466_146629


namespace NUMINAMATH_CALUDE_at_least_one_positive_solution_l1466_146674

def f (x : ℝ) : ℝ := x^10 + 4*x^9 + 7*x^8 + 2023*x^7 - 2024*x^6

theorem at_least_one_positive_solution :
  ∃ x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_positive_solution_l1466_146674


namespace NUMINAMATH_CALUDE_gears_can_rotate_l1466_146635

/-- A gear system with n identical gears arranged in a closed loop. -/
structure GearSystem where
  n : ℕ
  is_closed_loop : n ≥ 2

/-- Represents the rotation direction of a gear. -/
inductive RotationDirection
  | Clockwise
  | Counterclockwise

/-- Function to determine if adjacent gears have opposite rotation directions. -/
def opposite_rotation (d1 d2 : RotationDirection) : Prop :=
  (d1 = RotationDirection.Clockwise ∧ d2 = RotationDirection.Counterclockwise) ∨
  (d1 = RotationDirection.Counterclockwise ∧ d2 = RotationDirection.Clockwise)

/-- Theorem stating that the gears can rotate if and only if the number of gears is even. -/
theorem gears_can_rotate (system : GearSystem) :
  (∃ (rotation : ℕ → RotationDirection), 
    (∀ i : ℕ, i < system.n → opposite_rotation (rotation i) (rotation ((i + 1) % system.n))) ∧
    opposite_rotation (rotation 0) (rotation (system.n - 1)))
  ↔ 
  Even system.n :=
sorry

end NUMINAMATH_CALUDE_gears_can_rotate_l1466_146635


namespace NUMINAMATH_CALUDE_multiply_specific_numbers_l1466_146683

theorem multiply_specific_numbers : 469160 * 999999 = 469159530840 := by
  sorry

end NUMINAMATH_CALUDE_multiply_specific_numbers_l1466_146683


namespace NUMINAMATH_CALUDE_angle_c_in_triangle_l1466_146666

theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_in_triangle_l1466_146666


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1466_146617

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, x^2 + 2*x + a ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, x^2 + 2*x + a = 4) →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1466_146617


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1466_146632

theorem arithmetic_calculation : (((3.242 * (14 + 6)) - (7.234 * 7)) / 20) = 0.7101 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1466_146632


namespace NUMINAMATH_CALUDE_f_value_at_inverse_f_3_l1466_146603

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - 2 * x^2 else x^2 + 3 * x - 2

theorem f_value_at_inverse_f_3 : f (1 / f 3) = 127 / 128 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_inverse_f_3_l1466_146603


namespace NUMINAMATH_CALUDE_ashleys_age_l1466_146654

/-- Given that Ashley and Mary's ages are in the ratio 4:7 and their sum is 22, 
    prove that Ashley's age is 8 years. -/
theorem ashleys_age (ashley mary : ℕ) 
  (h_ratio : ashley * 7 = mary * 4)
  (h_sum : ashley + mary = 22) : 
  ashley = 8 := by
  sorry

end NUMINAMATH_CALUDE_ashleys_age_l1466_146654


namespace NUMINAMATH_CALUDE_speaking_orders_eq_264_l1466_146668

/-- The number of students in the group -/
def total_students : ℕ := 6

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of special students (A and B) -/
def special_students : ℕ := 2

/-- Function to calculate the number of different speaking orders -/
def speaking_orders : ℕ :=
  -- Case 1: Either A or B participates
  (special_students * (total_students - special_students).choose (selected_students - 1) * selected_students.factorial) +
  -- Case 2: Both A and B participate
  ((total_students - special_students).choose (selected_students - special_students) * special_students.factorial * (selected_students - 1).factorial)

/-- Theorem stating that the number of different speaking orders is 264 -/
theorem speaking_orders_eq_264 : speaking_orders = 264 := by
  sorry

end NUMINAMATH_CALUDE_speaking_orders_eq_264_l1466_146668


namespace NUMINAMATH_CALUDE_probability_3_or_more_babies_speak_l1466_146663

def probability_at_least_3_out_of_7 (p : ℝ) : ℝ :=
  1 - (Nat.choose 7 0 * p^0 * (1-p)^7 +
       Nat.choose 7 1 * p^1 * (1-p)^6 +
       Nat.choose 7 2 * p^2 * (1-p)^5)

theorem probability_3_or_more_babies_speak :
  probability_at_least_3_out_of_7 (1/3) = 939/2187 := by
  sorry

end NUMINAMATH_CALUDE_probability_3_or_more_babies_speak_l1466_146663


namespace NUMINAMATH_CALUDE_triangle_properties_l1466_146659

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define vector CM
variable (CM : ℝ × ℝ)

-- Given conditions
axiom side_angle_relation : 2 * b * Real.cos C = 2 * a - Real.sqrt 3 * c
axiom vector_relation : (0, 0) + CM + CM = (a, 0) + (b * Real.cos C, b * Real.sin C)
axiom cm_length : Real.sqrt (CM.1^2 + CM.2^2) = 1

-- Theorem to prove
theorem triangle_properties :
  B = π / 6 ∧
  (∃ (area : ℝ), area ≤ Real.sqrt 3 / 2 ∧
    ∀ (other_area : ℝ), other_area = 1/2 * a * b * Real.sin C → other_area ≤ area) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1466_146659


namespace NUMINAMATH_CALUDE_log_relationship_l1466_146646

theorem log_relationship (c d x : ℝ) (hc : c > 0) (hd : d > 0) (hx : x > 0 ∧ x ≠ 1) :
  6 * (Real.log x / Real.log c)^2 + 5 * (Real.log x / Real.log d)^2 = 12 * (Real.log x)^2 / (Real.log c * Real.log d) →
  d = c^(5 / (6 + Real.sqrt 6)) ∨ d = c^(5 / (6 - Real.sqrt 6)) :=
by sorry

end NUMINAMATH_CALUDE_log_relationship_l1466_146646


namespace NUMINAMATH_CALUDE_total_sleep_deficit_l1466_146625

/-- Calculates the total sleep deficit for three people over a week. -/
theorem total_sleep_deficit
  (ideal_sleep : ℕ)
  (tom_weeknight : ℕ)
  (tom_weekend : ℕ)
  (jane_weeknight : ℕ)
  (jane_weekend : ℕ)
  (mark_weeknight : ℕ)
  (mark_weekend : ℕ)
  (h1 : ideal_sleep = 8)
  (h2 : tom_weeknight = 5)
  (h3 : tom_weekend = 6)
  (h4 : jane_weeknight = 7)
  (h5 : jane_weekend = 9)
  (h6 : mark_weeknight = 6)
  (h7 : mark_weekend = 7) :
  (7 * ideal_sleep - (5 * tom_weeknight + 2 * tom_weekend)) +
  (7 * ideal_sleep - (5 * jane_weeknight + 2 * jane_weekend)) +
  (7 * ideal_sleep - (5 * mark_weeknight + 2 * mark_weekend)) = 34 := by
  sorry


end NUMINAMATH_CALUDE_total_sleep_deficit_l1466_146625


namespace NUMINAMATH_CALUDE_marathon_average_time_l1466_146648

theorem marathon_average_time (casey_time : ℝ) (zendaya_factor : ℝ) : 
  casey_time = 6 → 
  zendaya_factor = 1/3 → 
  let zendaya_time := casey_time + zendaya_factor * casey_time
  (casey_time + zendaya_time) / 2 = 7 := by
sorry

end NUMINAMATH_CALUDE_marathon_average_time_l1466_146648


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l1466_146620

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 8 * x + c = 0) →  -- Exactly one solution
  (a + c = 10) →                     -- Sum condition
  (a < c) →                          -- Inequality condition
  (a = 2 ∧ c = 8) :=                 -- Conclusion
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l1466_146620


namespace NUMINAMATH_CALUDE_cheese_slices_total_l1466_146676

/-- The number of cheese slices used for ham sandwiches -/
def ham_cheese_slices (num_ham_sandwiches : ℕ) (cheese_per_ham : ℕ) : ℕ :=
  num_ham_sandwiches * cheese_per_ham

/-- The number of cheese slices used for grilled cheese sandwiches -/
def grilled_cheese_slices (num_grilled_cheese : ℕ) (cheese_per_grilled : ℕ) : ℕ :=
  num_grilled_cheese * cheese_per_grilled

/-- The total number of cheese slices used for both types of sandwiches -/
def total_cheese_slices (ham_slices : ℕ) (grilled_slices : ℕ) : ℕ :=
  ham_slices + grilled_slices

/-- Theorem: The total number of cheese slices used for 10 ham sandwiches
    (each requiring 2 slices) and 10 grilled cheese sandwiches
    (each requiring 3 slices) is equal to 50. -/
theorem cheese_slices_total :
  total_cheese_slices
    (ham_cheese_slices 10 2)
    (grilled_cheese_slices 10 3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_cheese_slices_total_l1466_146676


namespace NUMINAMATH_CALUDE_speaking_orders_eq_720_l1466_146610

/-- The number of ways to select 4 students from 7 students (including A and B) to speak,
    where at least one of A and B must participate. -/
def speaking_orders : ℕ :=
  let n : ℕ := 7  -- Total number of students
  let k : ℕ := 4  -- Number of students to be selected
  let special : ℕ := 2  -- Number of special students (A and B)
  let others : ℕ := n - special  -- Number of other students

  -- Case 1: Exactly one of A and B participates
  let case1 : ℕ := special * (Nat.choose others (k - 1)) * (Nat.factorial k)

  -- Case 2: Both A and B participate
  let case2 : ℕ := (Nat.choose others (k - special)) * (Nat.factorial k)

  -- Total number of ways
  case1 + case2

/-- Theorem stating that the number of speaking orders is 720 -/
theorem speaking_orders_eq_720 : speaking_orders = 720 := by
  sorry

end NUMINAMATH_CALUDE_speaking_orders_eq_720_l1466_146610


namespace NUMINAMATH_CALUDE_floor_calculation_l1466_146687

/-- The floor of (2011^3 / (2009 * 2010)) - (2009^3 / (2010 * 2011)) is 8 -/
theorem floor_calculation : 
  ⌊(2011^3 : ℝ) / (2009 * 2010) - (2009^3 : ℝ) / (2010 * 2011)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_calculation_l1466_146687


namespace NUMINAMATH_CALUDE_age_difference_is_nine_l1466_146608

/-- The age difference between Bella's brother and Bella -/
def ageDifference (bellasAge : ℕ) (totalAge : ℕ) : ℕ :=
  totalAge - bellasAge - bellasAge

/-- Proof that the age difference is 9 years -/
theorem age_difference_is_nine :
  ageDifference 5 19 = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_nine_l1466_146608


namespace NUMINAMATH_CALUDE_jogging_distance_l1466_146642

/-- Calculates the distance traveled given a constant rate and time. -/
def distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Proves that jogging at 4 miles per hour for 2 hours results in a distance of 8 miles. -/
theorem jogging_distance : distance 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jogging_distance_l1466_146642


namespace NUMINAMATH_CALUDE_line_slope_l1466_146662

theorem line_slope (x y : ℝ) : 3 * y = 4 * x - 12 → (y - (-4)) / (x - 0) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l1466_146662


namespace NUMINAMATH_CALUDE_trajectory_equation_l1466_146614

theorem trajectory_equation (x y : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (y / (x + 1)) * (y / (x - 1)) = -2 → x^2 + y^2 / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1466_146614


namespace NUMINAMATH_CALUDE_angle_ABG_measure_l1466_146685

/-- A regular octagon is a polygon with 8 sides of equal length and 8 interior angles of equal measure. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- Angle ABG in a regular octagon ABCDEFGH -/
def angle_ABG (octagon : RegularOctagon) : ℝ := sorry

/-- The measure of angle ABG in a regular octagon is 22.5 degrees -/
theorem angle_ABG_measure (octagon : RegularOctagon) : angle_ABG octagon = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_ABG_measure_l1466_146685


namespace NUMINAMATH_CALUDE_min_value_a_plus_8b_l1466_146647

theorem min_value_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = 2 * a + b) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x * y = 2 * x + y → a + 8 * b ≤ x + 8 * y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x * y = 2 * x + y ∧ x + 8 * y = 25 / 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_8b_l1466_146647


namespace NUMINAMATH_CALUDE_expression_evaluation_l1466_146622

theorem expression_evaluation : 7500 + (1250 / 50) = 7525 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1466_146622


namespace NUMINAMATH_CALUDE_gift_box_volume_l1466_146667

/-- The volume of a rectangular box -/
def box_volume (width length height : ℝ) : ℝ := width * length * height

/-- Theorem: The volume of a box with dimensions 9 cm × 4 cm × 7 cm is 252 cm³ -/
theorem gift_box_volume :
  box_volume 9 4 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_gift_box_volume_l1466_146667


namespace NUMINAMATH_CALUDE_least_multiple_13_greater_than_418_l1466_146631

theorem least_multiple_13_greater_than_418 :
  ∀ n : ℕ, n > 0 ∧ 13 ∣ n ∧ n > 418 → n ≥ 429 :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_13_greater_than_418_l1466_146631


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1466_146615

theorem complex_number_in_fourth_quadrant (a b : ℝ) : 
  let z : ℂ := (a^2 - 6*a + 10) + (-b^2 + 4*b - 5)*I
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1466_146615


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1466_146636

theorem greatest_integer_satisfying_inequality :
  ∃ (x : ℕ), x > 0 ∧ (x^4 : ℚ) / (x^2 : ℚ) < 12 ∧
  ∀ (y : ℕ), y > x → (y^4 : ℚ) / (y^2 : ℚ) ≥ 12 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1466_146636


namespace NUMINAMATH_CALUDE_sqrt_0_1681_l1466_146652

theorem sqrt_0_1681 (h : Real.sqrt 16.81 = 4.1) : Real.sqrt 0.1681 = 0.41 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_0_1681_l1466_146652


namespace NUMINAMATH_CALUDE_platform_length_l1466_146686

/-- The length of a train platform given crossing times and lengths -/
theorem platform_length 
  (train_length : ℝ) 
  (first_time : ℝ) 
  (second_time : ℝ) 
  (second_platform : ℝ) 
  (h1 : train_length = 190) 
  (h2 : first_time = 15) 
  (h3 : second_time = 20) 
  (h4 : second_platform = 250) : 
  ∃ (first_platform : ℝ), 
    first_platform = 140 ∧ 
    (train_length + first_platform) / first_time = 
    (train_length + second_platform) / second_time :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1466_146686


namespace NUMINAMATH_CALUDE_a_squared_gt_b_squared_sufficient_not_necessary_l1466_146619

theorem a_squared_gt_b_squared_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a^2 > b^2 → abs a > b) ∧
  (∃ a b : ℝ, abs a > b ∧ a^2 ≤ b^2) :=
by sorry

end NUMINAMATH_CALUDE_a_squared_gt_b_squared_sufficient_not_necessary_l1466_146619


namespace NUMINAMATH_CALUDE_parallelogram_base_l1466_146645

/-- Given a parallelogram with area 308 square centimeters and height 14 cm, its base is 22 cm. -/
theorem parallelogram_base (area height base : ℝ) : 
  area = 308 ∧ height = 14 ∧ area = base * height → base = 22 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l1466_146645


namespace NUMINAMATH_CALUDE_center_numbers_l1466_146649

def numbers : List ℕ := [9, 12, 18, 24, 36, 48, 96]

def is_valid_center (x : ℕ) (nums : List ℕ) : Prop :=
  x ∈ nums ∧
  ∃ (a b c d e f : ℕ),
    a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧ e ∈ nums ∧ f ∈ nums ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
    a * x * d = b * x * e ∧ b * x * e = c * x * f

theorem center_numbers :
  ∀ x ∈ numbers, is_valid_center x numbers ↔ x = 12 ∨ x = 96 := by
  sorry

end NUMINAMATH_CALUDE_center_numbers_l1466_146649


namespace NUMINAMATH_CALUDE_inequality_proof_l1466_146612

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) ≥ Real.sqrt (a^2 + a*c + c^2) ∧
  (Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) = Real.sqrt (a^2 + a*c + c^2) ↔ 1/b = 1/a + 1/c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1466_146612


namespace NUMINAMATH_CALUDE_preferred_pets_combinations_l1466_146643

/-- The number of puppies available in the pet store -/
def num_puppies : Nat := 20

/-- The number of kittens available in the pet store -/
def num_kittens : Nat := 10

/-- The number of hamsters available in the pet store -/
def num_hamsters : Nat := 12

/-- The number of ways Alice, Bob, and Charlie can buy their preferred pets -/
def num_ways : Nat := num_puppies * num_kittens * num_hamsters

/-- Theorem stating that the number of ways to buy preferred pets is 2400 -/
theorem preferred_pets_combinations : num_ways = 2400 := by
  sorry

end NUMINAMATH_CALUDE_preferred_pets_combinations_l1466_146643


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l1466_146604

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 675

/-- The time in minutes for pipe A to fill the tank -/
def pipe_a_time : ℝ := 12

/-- The time in minutes for pipe B to fill the tank -/
def pipe_b_time : ℝ := 20

/-- The rate at which pipe C drains water in liters per minute -/
def pipe_c_rate : ℝ := 45

/-- The time in minutes to fill the tank when all pipes are opened -/
def all_pipes_time : ℝ := 15

/-- Theorem stating that the tank capacity is correct given the conditions -/
theorem tank_capacity_proof :
  tank_capacity = pipe_a_time * pipe_b_time * all_pipes_time * pipe_c_rate /
    (pipe_a_time * pipe_b_time - pipe_a_time * all_pipes_time - pipe_b_time * all_pipes_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l1466_146604


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1466_146691

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1466_146691


namespace NUMINAMATH_CALUDE_concurrent_lines_through_circumcenter_l1466_146602

-- Define the basic structures
structure Point :=
  (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Line :=
  (p1 p2 : Point)

-- Define the properties
def isAcuteAngled (t : Triangle) : Prop := sorry

def altitudeFoot (t : Triangle) (v : Point) : Point := sorry

def perpendicularFoot (p : Point) (l : Line) : Point := sorry

def isOn (p : Point) (l : Line) : Prop := sorry

def intersectionPoint (l1 l2 : Line) : Point := sorry

def circumcenter (t : Triangle) : Point := sorry

-- Main theorem
theorem concurrent_lines_through_circumcenter 
  (t : Triangle) 
  (hAcute : isAcuteAngled t)
  (D : Point) (hD : D = altitudeFoot t t.A)
  (E : Point) (hE : E = altitudeFoot t t.B)
  (F : Point) (hF : F = altitudeFoot t t.C)
  (P : Point) (hP : P = perpendicularFoot t.A (Line.mk E F))
  (Q : Point) (hQ : Q = perpendicularFoot t.B (Line.mk F D))
  (R : Point) (hR : R = perpendicularFoot t.C (Line.mk D E)) :
  ∃ O : Point, 
    isOn O (Line.mk t.A P) ∧ 
    isOn O (Line.mk t.B Q) ∧ 
    isOn O (Line.mk t.C R) ∧
    O = circumcenter t :=
sorry

end NUMINAMATH_CALUDE_concurrent_lines_through_circumcenter_l1466_146602


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l1466_146634

/-- Given an ellipse with equation x^2/16 + y^2/9 = 1, its foci are located at (√7, 0) and (-√7, 0) -/
theorem ellipse_foci_coordinates :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/16 + y^2/9 = 1}
  ∃ (f₁ f₂ : ℝ × ℝ), f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ 
    f₁ = (Real.sqrt 7, 0) ∧ f₂ = (-Real.sqrt 7, 0) ∧
    ∀ (p : ℝ × ℝ), p ∈ ellipse → 
      (dist p f₁) + (dist p f₂) = 2 * 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l1466_146634


namespace NUMINAMATH_CALUDE_total_amount_spent_l1466_146688

theorem total_amount_spent (num_pens num_pencils : ℕ) (avg_pen_price avg_pencil_price total_amount : ℚ) :
  num_pens = 30 →
  num_pencils = 75 →
  avg_pen_price = 18 →
  avg_pencil_price = 2 →
  total_amount = num_pens * avg_pen_price + num_pencils * avg_pencil_price →
  total_amount = 690 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_spent_l1466_146688


namespace NUMINAMATH_CALUDE_intersection_line_equation_l1466_146680

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define the line AB
def line_AB (x y : ℝ) : Prop := 3*x - 3*y - 10 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ A B : ℝ × ℝ,
  (circle_C1 A.1 A.2 ∧ circle_C2 A.1 A.2) →
  (circle_C1 B.1 B.2 ∧ circle_C2 B.1 B.2) →
  A ≠ B →
  line_AB A.1 A.2 ∧ line_AB B.1 B.2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l1466_146680


namespace NUMINAMATH_CALUDE_inequality_representation_l1466_146613

theorem inequality_representation (x : ℝ) : 
  (x + 4 < 10) ↔ (∃ y, y = x + 4 ∧ y < 10) :=
sorry

end NUMINAMATH_CALUDE_inequality_representation_l1466_146613


namespace NUMINAMATH_CALUDE_janes_numbers_l1466_146696

def is_valid_number (n : ℕ) : Prop :=
  n % 180 = 0 ∧ n % 42 = 0 ∧ 500 < n ∧ n < 4000

theorem janes_numbers :
  {n : ℕ | is_valid_number n} = {1260, 2520, 3780} :=
sorry

end NUMINAMATH_CALUDE_janes_numbers_l1466_146696


namespace NUMINAMATH_CALUDE_problem_solution_l1466_146633

theorem problem_solution (A B : ℝ) (hB : B ≠ 0) : 
  let f : ℝ → ℝ := λ x ↦ A * x^2 - 3 * B^2
  let g : ℝ → ℝ := λ x ↦ B * x^2
  f (g 1) = 0 → A = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1466_146633


namespace NUMINAMATH_CALUDE_cubic_is_constant_tangent_bounds_on_m_for_constant_tangent_function_l1466_146670

-- Definition of a "constant tangent function"
def is_constant_tangent_function (f : ℝ → ℝ) : Prop :=
  ∀ k b : ℝ, ∃ x₀ : ℝ, f x₀ + k * x₀ + b = k * x₀ + b ∧ 
  (deriv f) x₀ + k = k

-- Part 1: Prove that x^3 is a constant tangent function
theorem cubic_is_constant_tangent : is_constant_tangent_function (λ x : ℝ => x^3) := by
  sorry

-- Part 2: Prove the bounds on m for the given function
theorem bounds_on_m_for_constant_tangent_function :
  is_constant_tangent_function (λ x : ℝ => 1/2 * (Real.exp x - x - 1) * Real.exp x + m) →
  -1/8 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_is_constant_tangent_bounds_on_m_for_constant_tangent_function_l1466_146670


namespace NUMINAMATH_CALUDE_record_storage_cost_l1466_146611

def box_length : ℝ := 15
def box_width : ℝ := 12
def box_height : ℝ := 10
def total_volume : ℝ := 1080000
def cost_per_box : ℝ := 0.8

theorem record_storage_cost : 
  let box_volume := box_length * box_width * box_height
  let num_boxes := total_volume / box_volume
  num_boxes * cost_per_box = 480 := by sorry

end NUMINAMATH_CALUDE_record_storage_cost_l1466_146611


namespace NUMINAMATH_CALUDE_solution_set_of_f_less_than_two_range_of_m_for_f_geq_m_squared_l1466_146630

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| + |x|

-- Theorem 1
theorem solution_set_of_f_less_than_two (m : ℝ) (h : f 1 m = 1) :
  {x : ℝ | f x m < 2} = Set.Ioo (-1/2) (3/2) := by sorry

-- Theorem 2
theorem range_of_m_for_f_geq_m_squared :
  {m : ℝ | ∀ x, f x m ≥ m^2} = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_less_than_two_range_of_m_for_f_geq_m_squared_l1466_146630


namespace NUMINAMATH_CALUDE_container_filling_l1466_146637

theorem container_filling (capacity : ℝ) (initial_fraction : ℝ) (added_water : ℝ) :
  capacity = 80 →
  initial_fraction = 1/2 →
  added_water = 20 →
  (initial_fraction * capacity + added_water) / capacity = 3/4 := by
sorry

end NUMINAMATH_CALUDE_container_filling_l1466_146637


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_expression_l1466_146609

theorem largest_power_of_two_dividing_expression : ∃ k : ℕ, 
  (2^k : ℤ) ∣ (15^4 - 7^4 - 8) ∧ 
  ∀ m : ℕ, (2^m : ℤ) ∣ (15^4 - 7^4 - 8) → m ≤ k ∧
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_expression_l1466_146609


namespace NUMINAMATH_CALUDE_senior_mean_score_l1466_146624

theorem senior_mean_score 
  (total_students : ℕ) 
  (overall_mean : ℝ) 
  (senior_count : ℕ) 
  (non_senior_count : ℕ) 
  (h1 : total_students = 200)
  (h2 : overall_mean = 120)
  (h3 : non_senior_count = 2 * senior_count)
  (h4 : total_students = senior_count + non_senior_count)
  (h5 : senior_count > 0)
  (h6 : non_senior_count > 0) :
  ∃ (senior_mean non_senior_mean : ℝ),
    non_senior_mean = 0.8 * senior_mean ∧
    (senior_count : ℝ) * senior_mean + (non_senior_count : ℝ) * non_senior_mean = (total_students : ℝ) * overall_mean ∧
    senior_mean = 138 := by
  sorry


end NUMINAMATH_CALUDE_senior_mean_score_l1466_146624


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1466_146677

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/2)^2 + (1/3)^2 = ((1/3)^2 + (1/6)^2) * (13*x)/(47*y)) :
  Real.sqrt x / Real.sqrt y = Real.sqrt 47 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1466_146677


namespace NUMINAMATH_CALUDE_distance_to_park_l1466_146697

/-- Represents the scenario of Xiao Wu's bicycle ride and his father's car journey --/
structure JourneyScenario where
  wu_speed : ℝ
  father_speed : ℝ
  catch_up_distance : ℝ
  remaining_time : ℝ
  initial_delay : ℝ

/-- Calculates the total distance between Wu's home and the forest park --/
def calculate_total_distance (scenario : JourneyScenario) : ℝ :=
  sorry

/-- Theorem stating that the distance between Wu's home and the forest park is 11 km --/
theorem distance_to_park (scenario : JourneyScenario) 
  (h1 : scenario.father_speed = 5 * scenario.wu_speed)
  (h2 : scenario.catch_up_distance = 3.5)
  (h3 : scenario.remaining_time = 10)
  (h4 : scenario.initial_delay = 30) :
  calculate_total_distance scenario = 11 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_park_l1466_146697


namespace NUMINAMATH_CALUDE_fraction_value_l1466_146665

theorem fraction_value : 
  let a : ℕ := 2003
  let b : ℕ := 2002
  let four : ℕ := 2^2
  let six : ℕ := 2 * 3
  (four^a * 3^b) / (six^b * 2^a) = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l1466_146665


namespace NUMINAMATH_CALUDE_sum_min_period_length_l1466_146616

def min_period_length (x : ℚ) : ℕ :=
  sorry

theorem sum_min_period_length (A B : ℚ) :
  min_period_length A = 6 →
  min_period_length B = 12 →
  min_period_length (A + B) = 12 ∨ min_period_length (A + B) = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_min_period_length_l1466_146616


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_l1466_146653

/-- Three lines in a plane -/
structure PlaneLine where
  dir : ℝ × ℝ  -- Direction vector

/-- Perpendicular lines -/
def perpendicular (l1 l2 : PlaneLine) : Prop :=
  l1.dir.1 * l2.dir.1 + l1.dir.2 * l2.dir.2 = 0

/-- Parallel lines -/
def parallel (l1 l2 : PlaneLine) : Prop :=
  l1.dir.1 * l2.dir.2 = l1.dir.2 * l2.dir.1

theorem perpendicular_implies_parallel (a b c : PlaneLine) :
  perpendicular a b → perpendicular b c → parallel a c := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_l1466_146653


namespace NUMINAMATH_CALUDE_max_product_sum_2024_l1466_146638

theorem max_product_sum_2024 :
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144) ∧
  (∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2024_l1466_146638


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l1466_146671

/-- The number of herbs available for the wizard's elixir. -/
def num_herbs : ℕ := 4

/-- The number of crystals available for the wizard's elixir. -/
def num_crystals : ℕ := 6

/-- The number of incompatible combinations due to the first problematic crystal. -/
def incompatible_combinations_1 : ℕ := 2

/-- The number of incompatible combinations due to the second problematic crystal. -/
def incompatible_combinations_2 : ℕ := 1

/-- The total number of viable combinations for the wizard's elixir. -/
def viable_combinations : ℕ := num_herbs * num_crystals - (incompatible_combinations_1 + incompatible_combinations_2)

theorem wizard_elixir_combinations :
  viable_combinations = 21 :=
sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l1466_146671


namespace NUMINAMATH_CALUDE_polynomial_equality_l1466_146650

theorem polynomial_equality : 98^5 - 5 * 98^4 + 10 * 98^3 - 10 * 98^2 + 5 * 98 - 1 = 97^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1466_146650


namespace NUMINAMATH_CALUDE_simplify_expression_l1466_146623

theorem simplify_expression (x y : ℝ) : 7*x + 3 - 2*x + 15 + y = 5*x + y + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1466_146623


namespace NUMINAMATH_CALUDE_cards_given_to_friend_l1466_146655

/-- Given that Joseph had 16 baseball cards, gave 3/8 to his brother, and now has 50% left,
    prove that he gave 2 cards to his friend. -/
theorem cards_given_to_friend (initial_cards : ℕ) (cards_to_brother : ℚ) (cards_left : ℚ) :
  initial_cards = 16 →
  cards_to_brother = 3 / 8 →
  cards_left = 1 / 2 →
  initial_cards * (1 - cards_to_brother - cards_left) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_friend_l1466_146655


namespace NUMINAMATH_CALUDE_fraction_reduction_l1466_146651

theorem fraction_reduction (a : ℚ) : 
  (4 - a) / (5 - a) = 16 / 25 → 9 * a = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_reduction_l1466_146651


namespace NUMINAMATH_CALUDE_max_monthly_profit_l1466_146681

/-- Represents the monthly profit as a function of price increase --/
def monthly_profit (x : ℕ) : ℤ :=
  -10 * x^2 + 110 * x + 2100

/-- The maximum allowed price increase --/
def max_increase : ℕ := 15

/-- Theorem stating the maximum monthly profit and optimal selling prices --/
theorem max_monthly_profit :
  (∃ x : ℕ, x > 0 ∧ x ≤ max_increase ∧ monthly_profit x = 2400) ∧
  (∀ x : ℕ, x > 0 ∧ x ≤ max_increase → monthly_profit x ≤ 2400) ∧
  (monthly_profit 5 = 2400 ∧ monthly_profit 6 = 2400) :=
sorry

end NUMINAMATH_CALUDE_max_monthly_profit_l1466_146681


namespace NUMINAMATH_CALUDE_task_completion_probability_l1466_146695

theorem task_completion_probability
  (p_task2 : ℝ)
  (p_task1_not_task2 : ℝ)
  (h1 : p_task2 = 3 / 5)
  (h2 : p_task1_not_task2 = 0.15)
  (h_independent : True)  -- Representing the independence of tasks
  : ∃ (p_task1 : ℝ), p_task1 = 0.375 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_probability_l1466_146695


namespace NUMINAMATH_CALUDE_ratio_of_remaining_ingredients_l1466_146672

def total_sugar : ℕ := 13
def total_flour : ℕ := 25
def total_cocoa : ℕ := 60

def added_sugar : ℕ := 12
def added_flour : ℕ := 8
def added_cocoa : ℕ := 15

def remaining_flour : ℕ := total_flour - added_flour
def remaining_sugar : ℕ := total_sugar - added_sugar
def remaining_cocoa : ℕ := total_cocoa - added_cocoa

theorem ratio_of_remaining_ingredients :
  (remaining_flour : ℚ) / (remaining_sugar + remaining_cocoa) = 17 / 46 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_remaining_ingredients_l1466_146672


namespace NUMINAMATH_CALUDE_swallow_theorem_l1466_146639

/-- The number of swallows initially on the wire -/
def initial_swallows : ℕ := 9

/-- The distance between the first and last swallow in centimeters -/
def total_distance : ℕ := 720

/-- The number of additional swallows added between each pair -/
def additional_swallows : ℕ := 3

/-- Theorem stating the distance between neighboring swallows and the total number after adding more -/
theorem swallow_theorem :
  (let gaps := initial_swallows - 1
   let distance_between := total_distance / gaps
   let new_swallows := gaps * additional_swallows
   let total_swallows := initial_swallows + new_swallows
   (distance_between = 90 ∧ total_swallows = 33)) :=
by sorry

end NUMINAMATH_CALUDE_swallow_theorem_l1466_146639


namespace NUMINAMATH_CALUDE_mod_nine_equivalence_l1466_146678

theorem mod_nine_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2222 ≡ n [ZMOD 9] ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_mod_nine_equivalence_l1466_146678


namespace NUMINAMATH_CALUDE_suitcase_waiting_time_l1466_146641

/-- The number of suitcases loaded onto the plane -/
def total_suitcases : ℕ := 200

/-- The number of suitcases belonging to the businesspeople -/
def business_suitcases : ℕ := 10

/-- The time interval between placing suitcases on the conveyor belt (in seconds) -/
def placement_interval : ℕ := 2

/-- The probability of the businesspeople waiting exactly two minutes for their last suitcase -/
def exact_two_minutes_probability : ℚ :=
  (Nat.choose 59 9 : ℚ) / (Nat.choose total_suitcases business_suitcases : ℚ)

/-- The expected time (in seconds) the businesspeople will wait for their last suitcase -/
def expected_waiting_time : ℚ := 4020 / 11

theorem suitcase_waiting_time :
  (exact_two_minutes_probability = (Nat.choose 59 9 : ℚ) / (Nat.choose total_suitcases business_suitcases : ℚ)) ∧
  (expected_waiting_time = 4020 / 11) := by
  sorry

end NUMINAMATH_CALUDE_suitcase_waiting_time_l1466_146641


namespace NUMINAMATH_CALUDE_camping_hike_distance_l1466_146699

/-- The total distance hiked by Irwin's family on their camping trip -/
theorem camping_hike_distance 
  (car_to_stream : ℝ) 
  (stream_to_meadow : ℝ) 
  (meadow_to_campsite : ℝ) 
  (h1 : car_to_stream = 0.2)
  (h2 : stream_to_meadow = 0.4)
  (h3 : meadow_to_campsite = 0.1) :
  car_to_stream + stream_to_meadow + meadow_to_campsite = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_camping_hike_distance_l1466_146699


namespace NUMINAMATH_CALUDE_max_of_two_numbers_l1466_146693

theorem max_of_two_numbers (a b : ℕ) (ha : a = 2) (hb : b = 3) :
  max a b = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_of_two_numbers_l1466_146693


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l1466_146675

theorem final_sum_after_transformation (x y S : ℝ) : 
  x + y = S → 3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l1466_146675


namespace NUMINAMATH_CALUDE_history_not_statistics_l1466_146692

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ)
  (h_total : total = 90)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_or_statistics : history_or_statistics = 57) :
  history - (history + statistics - history_or_statistics) = 25 := by
  sorry

end NUMINAMATH_CALUDE_history_not_statistics_l1466_146692


namespace NUMINAMATH_CALUDE_athlete_arrangements_l1466_146679

def male_athletes : ℕ := 7
def female_athletes : ℕ := 3

theorem athlete_arrangements :
  let total_athletes := male_athletes + female_athletes
  let arrangements_case1 := (male_athletes.factorial) * (male_athletes - 1) * (male_athletes - 2) * (male_athletes - 3)
  let arrangements_case2 := 2 * (female_athletes.factorial) * (male_athletes.factorial)
  let arrangements_case3 := (total_athletes + 1).factorial * (female_athletes.factorial)
  (arrangements_case1 = 604800) ∧
  (arrangements_case2 = 60480) ∧
  (arrangements_case3 = 241920) := by
  sorry

#eval male_athletes.factorial * (male_athletes - 1) * (male_athletes - 2) * (male_athletes - 3)
#eval 2 * female_athletes.factorial * male_athletes.factorial
#eval (male_athletes + female_athletes + 1).factorial * female_athletes.factorial

end NUMINAMATH_CALUDE_athlete_arrangements_l1466_146679


namespace NUMINAMATH_CALUDE_composite_has_small_divisor_l1466_146601

/-- A number is composite if it's a natural number greater than 1 with a divisor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

/-- For any composite number, there exists a divisor greater than 1 but not greater than its square root. -/
theorem composite_has_small_divisor (n : ℕ) (h : IsComposite n) :
    ∃ d : ℕ, d ∣ n ∧ 1 < d ∧ d ≤ Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_composite_has_small_divisor_l1466_146601


namespace NUMINAMATH_CALUDE_hyperbola_parabola_focus_coincide_l1466_146606

theorem hyperbola_parabola_focus_coincide (a : ℝ) : 
  a > 0 → 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1) → 
  (∃ (x y : ℝ), y^2 = 8*x) → 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1 ∧ y^2 = 8*x ∧ x = 2 ∧ y = 0) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_focus_coincide_l1466_146606


namespace NUMINAMATH_CALUDE_inequality_proof_l1466_146682

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (1 / (a - b)) + (1 / (b - c)) ≥ 4 / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1466_146682


namespace NUMINAMATH_CALUDE_student_project_assignment_l1466_146658

/-- The number of ways to assign students to projects. -/
def assignmentCount (n : ℕ) (k : ℕ) : ℕ :=
  if k ≤ n then (n - k + 1).factorial * (n.choose k) else 0

/-- Theorem stating the number of ways to assign 6 students to 3 projects. -/
theorem student_project_assignment :
  assignmentCount 6 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_student_project_assignment_l1466_146658


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_right_triangle_l1466_146698

/-- In a right triangle with acute angles in the ratio 3:2, the smallest angle is 36 degrees. -/
theorem smallest_angle_in_special_right_triangle :
  ∀ (a b : ℝ), 
    a > 0 → b > 0 →  -- angles are positive
    a + b = 90 →     -- sum of acute angles in a right triangle
    a / b = 3 / 2 → -- ratio of angles is 3:2
    min a b = 36 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_right_triangle_l1466_146698


namespace NUMINAMATH_CALUDE_factorial_10_mod_13_l1466_146627

/-- Definition of factorial for natural numbers -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: The remainder when 10! is divided by 13 is 6 -/
theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_10_mod_13_l1466_146627


namespace NUMINAMATH_CALUDE_distance_is_seven_l1466_146673

def point : ℝ × ℝ × ℝ := (2, 4, 6)

def line_point : ℝ × ℝ × ℝ := (8, 9, 9)

def line_direction : ℝ × ℝ × ℝ := (5, 2, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_seven :
  distance_to_line point line_point line_direction = 7 :=
sorry

end NUMINAMATH_CALUDE_distance_is_seven_l1466_146673


namespace NUMINAMATH_CALUDE_player_one_wins_l1466_146607

/-- Represents the number of coins a player can take -/
def ValidMove (player : ℕ) (coins : ℕ) : Prop :=
  match player with
  | 1 => coins % 2 = 1 ∧ 1 ≤ coins ∧ coins ≤ 99
  | 2 => coins % 2 = 0 ∧ 2 ≤ coins ∧ coins ≤ 100
  | _ => False

/-- The game state -/
structure GameState where
  coins : ℕ
  currentPlayer : ℕ

/-- A winning strategy for a player -/
def WinningStrategy (player : ℕ) : Prop :=
  ∀ (state : GameState), state.currentPlayer = player →
    ∃ (move : ℕ), ValidMove player move ∧
      (state.coins < move ∨
       ¬∃ (opponentMove : ℕ), ValidMove (3 - player) opponentMove ∧
         state.coins - move - opponentMove ≥ 0)

/-- The main theorem: Player 1 has a winning strategy -/
theorem player_one_wins : WinningStrategy 1 := by
  sorry

#check player_one_wins

end NUMINAMATH_CALUDE_player_one_wins_l1466_146607


namespace NUMINAMATH_CALUDE_cube_side_ratio_l1466_146618

/-- Given two cubes of the same material, this theorem proves that if their weights are in the ratio of 32:4, then their side lengths are in the ratio of 2:1. -/
theorem cube_side_ratio (s₁ s₂ : ℝ) (w₁ w₂ : ℝ) (h₁ : w₁ = 4) (h₂ : w₂ = 32) :
  w₁ * s₂^3 = w₂ * s₁^3 → s₂ / s₁ = 2 := by sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l1466_146618


namespace NUMINAMATH_CALUDE_inequality_solution_l1466_146605

theorem inequality_solution (x : ℝ) (h1 : x ≠ 1) (h3 : x ≠ 3) (h4 : x ≠ 4) (h5 : x ≠ 5) :
  (2 / (x - 1) - 3 / (x - 3) + 2 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔
  (x < -1 ∨ (1 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (7 < x ∧ x < 8)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1466_146605


namespace NUMINAMATH_CALUDE_choir_arrangement_max_l1466_146684

theorem choir_arrangement_max (n : ℕ) : 
  (∃ k : ℕ, n = k^2 + 11) ∧ 
  (∃ x : ℕ, n = x * (x + 5)) →
  n ≤ 126 :=
sorry

end NUMINAMATH_CALUDE_choir_arrangement_max_l1466_146684


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1466_146656

theorem ratio_of_sum_to_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1466_146656


namespace NUMINAMATH_CALUDE_counterexample_exists_l1466_146690

-- Define the types for planes and lines in space
variable (Plane Line : Type)

-- Define the intersection operation for planes
variable (intersect_planes : Plane → Plane → Line)

-- Define the perpendicular relation between lines
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem counterexample_exists :
  ∃ (α β γ : Plane) (l m n : Line),
    intersect_planes α β = m ∧
    intersect_planes β γ = l ∧
    intersect_planes γ α = n ∧
    perpendicular l m ∧
    perpendicular l n ∧
    ¬ perpendicular m n :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1466_146690


namespace NUMINAMATH_CALUDE_jason_grass_cutting_time_l1466_146626

/-- The time it takes Jason to cut one lawn in minutes -/
def time_per_lawn : ℕ := 30

/-- The number of yards Jason cuts on Saturday -/
def yards_saturday : ℕ := 8

/-- The number of yards Jason cuts on Sunday -/
def yards_sunday : ℕ := 8

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem: Jason spends 8 hours cutting grass over the weekend -/
theorem jason_grass_cutting_time :
  (time_per_lawn * (yards_saturday + yards_sunday)) / minutes_per_hour = 8 := by
  sorry

end NUMINAMATH_CALUDE_jason_grass_cutting_time_l1466_146626


namespace NUMINAMATH_CALUDE_expression_evaluation_l1466_146669

theorem expression_evaluation (x : ℤ) (h : x = -2) : (x + 5)^2 - (x - 2) * (x + 2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1466_146669


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1466_146621

theorem quadratic_solution_difference_squared :
  ∀ f g : ℝ,
  (2 * f^2 + 8 * f - 42 = 0) →
  (2 * g^2 + 8 * g - 42 = 0) →
  (f ≠ g) →
  (f - g)^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1466_146621
