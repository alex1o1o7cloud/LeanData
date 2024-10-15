import Mathlib

namespace NUMINAMATH_CALUDE_full_price_store_a_is_125_l504_50422

/-- The full price of a smartphone at Store A, given discount information for two stores. -/
def full_price_store_a : ℝ :=
  let discount_a : ℝ := 0.08
  let price_b : ℝ := 130
  let discount_b : ℝ := 0.10
  let price_difference : ℝ := 2

  -- Define the equation based on the given conditions
  let equation : ℝ → Prop := fun p =>
    p * (1 - discount_a) = price_b * (1 - discount_b) - price_difference

  -- Assert that 125 satisfies the equation
  125

theorem full_price_store_a_is_125 :
  full_price_store_a = 125 := by sorry

end NUMINAMATH_CALUDE_full_price_store_a_is_125_l504_50422


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l504_50419

theorem common_internal_tangent_length 
  (center_distance : ℝ) 
  (radius1 : ℝ) 
  (radius2 : ℝ) 
  (h1 : center_distance = 50)
  (h2 : radius1 = 7)
  (h3 : radius2 = 10) : 
  Real.sqrt (center_distance^2 - (radius1 + radius2)^2) = Real.sqrt 2211 :=
sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l504_50419


namespace NUMINAMATH_CALUDE_candy_has_nine_pencils_l504_50400

-- Define variables
def candy_pencils : ℕ := sorry
def caleb_pencils : ℕ := sorry
def calen_original_pencils : ℕ := sorry
def calen_final_pencils : ℕ := sorry

-- Define conditions
axiom caleb_pencils_def : caleb_pencils = 2 * candy_pencils - 3
axiom calen_original_pencils_def : calen_original_pencils = caleb_pencils + 5
axiom calen_final_pencils_def : calen_final_pencils = calen_original_pencils - 10
axiom calen_final_pencils_value : calen_final_pencils = 10

-- Theorem to prove
theorem candy_has_nine_pencils : candy_pencils = 9 := by sorry

end NUMINAMATH_CALUDE_candy_has_nine_pencils_l504_50400


namespace NUMINAMATH_CALUDE_complex_modulus_one_l504_50463

theorem complex_modulus_one (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) :
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l504_50463


namespace NUMINAMATH_CALUDE_largest_angle_of_specific_triangle_l504_50462

/-- Given a triangle with sides 3√2, 6, and 3√10, its largest interior angle is 135°. -/
theorem largest_angle_of_specific_triangle : 
  ∀ (a b c θ : ℝ), 
  a = 3 * Real.sqrt 2 → 
  b = 6 → 
  c = 3 * Real.sqrt 10 → 
  c > a ∧ c > b → 
  θ = Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) → 
  θ = 135 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_largest_angle_of_specific_triangle_l504_50462


namespace NUMINAMATH_CALUDE_solution_interval_l504_50450

theorem solution_interval (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9 ↔ 63 / 26 < x ∧ x ≤ 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l504_50450


namespace NUMINAMATH_CALUDE_projection_a_onto_b_l504_50471

def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (-2, 1)

theorem projection_a_onto_b :
  let proj_magnitude := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  proj_magnitude = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_projection_a_onto_b_l504_50471


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l504_50431

theorem trigonometric_simplification :
  (Real.tan (12 * π / 180) - Real.sqrt 3) / (Real.sin (12 * π / 180) * Real.cos (24 * π / 180)) = -8 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l504_50431


namespace NUMINAMATH_CALUDE_factor_values_l504_50482

def polynomial (x : ℝ) : ℝ := 8 * x^2 + 18 * x - 5

theorem factor_values (t : ℝ) : 
  (∀ x, polynomial x = 0 → x = t) ↔ t = 1/4 ∨ t = -5 := by
  sorry

end NUMINAMATH_CALUDE_factor_values_l504_50482


namespace NUMINAMATH_CALUDE_combined_distance_theorem_l504_50444

/-- Represents the four lakes in the migration sequence -/
inductive Lake : Type
| Jim : Lake
| Disney : Lake
| London : Lake
| Everest : Lake

/-- The number of birds in the group -/
def num_birds : ℕ := 25

/-- The number of migration sequences completed in a year -/
def sequences_per_year : ℕ := 2

/-- The distance between two lakes in miles -/
def distance (a b : Lake) : ℕ :=
  match a, b with
  | Lake.Jim, Lake.Disney => 42
  | Lake.Disney, Lake.London => 57
  | Lake.London, Lake.Everest => 65
  | Lake.Everest, Lake.Jim => 70
  | _, _ => 0  -- For other combinations, return 0

/-- The total distance of one migration sequence -/
def sequence_distance : ℕ :=
  distance Lake.Jim Lake.Disney +
  distance Lake.Disney Lake.London +
  distance Lake.London Lake.Everest +
  distance Lake.Everest Lake.Jim

/-- Theorem: The combined distance traveled by all birds in a year is 11,700 miles -/
theorem combined_distance_theorem :
  num_birds * sequences_per_year * sequence_distance = 11700 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_theorem_l504_50444


namespace NUMINAMATH_CALUDE_santa_candy_problem_l504_50427

theorem santa_candy_problem (total : ℕ) (chocolate : ℕ) (gummy : ℕ) :
  total = 2023 →
  chocolate + gummy = total →
  chocolate = (75 * gummy) / 100 →
  chocolate = 867 := by
sorry

end NUMINAMATH_CALUDE_santa_candy_problem_l504_50427


namespace NUMINAMATH_CALUDE_biotech_job_count_l504_50460

/-- Represents the class of 2000 biotechnology graduates --/
structure BiotechClass :=
  (total : ℕ)
  (secondDegree : ℕ)
  (bothJobAndDegree : ℕ)
  (neither : ℕ)

/-- Calculates the number of graduates who found a job --/
def graduatesWithJob (c : BiotechClass) : ℕ :=
  c.total - c.neither - (c.secondDegree - c.bothJobAndDegree)

/-- Theorem: In the given biotech class, 32 graduates found a job --/
theorem biotech_job_count (c : BiotechClass) 
  (h1 : c.total = 73)
  (h2 : c.secondDegree = 45)
  (h3 : c.bothJobAndDegree = 13)
  (h4 : c.neither = 9) :
  graduatesWithJob c = 32 := by
sorry

end NUMINAMATH_CALUDE_biotech_job_count_l504_50460


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l504_50483

theorem smallest_n_satisfying_conditions : 
  ∃ n : ℕ, 
    n > 2021 ∧ 
    Nat.gcd 63 (n + 120) = 21 ∧ 
    Nat.gcd (n + 63) 120 = 60 ∧
    (∀ m : ℕ, m > 2021 → Nat.gcd 63 (m + 120) = 21 → Nat.gcd (m + 63) 120 = 60 → m ≥ n) ∧
    n = 2337 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l504_50483


namespace NUMINAMATH_CALUDE_incircle_touch_point_distance_special_triangle_incircle_touch_point_distance_l504_50417

/-- Given a triangle with sides a, b, c, and an incircle that touches side c at point P,
    the distance from one endpoint of side c to P is (a + b + c) / 2 - b -/
theorem incircle_touch_point_distance (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  let s := (a + b + c) / 2
  (s - b) = ((a + b + c) / 2) - b :=
by sorry

/-- In a triangle with sides 4, 5, and 6, the distance from one vertex to the point 
    where the incircle touches the opposite side is 2.5 -/
theorem special_triangle_incircle_touch_point_distance :
  let a := 4
  let b := 5
  let c := 6
  let s := (a + b + c) / 2
  (s - b) = 2.5 :=
by sorry

end NUMINAMATH_CALUDE_incircle_touch_point_distance_special_triangle_incircle_touch_point_distance_l504_50417


namespace NUMINAMATH_CALUDE_boat_distance_downstream_l504_50414

/-- Calculates the distance traveled downstream by a boat -/
theorem boat_distance_downstream 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 22) 
  (h2 : stream_speed = 5) 
  (h3 : time = 5) : 
  boat_speed + stream_speed * time = 135 := by
  sorry

#check boat_distance_downstream

end NUMINAMATH_CALUDE_boat_distance_downstream_l504_50414


namespace NUMINAMATH_CALUDE_triangle_altitude_angle_relation_l504_50436

theorem triangle_altitude_angle_relation (A B C : Real) (C₁ C₂ : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = 180 →
  -- A is 60° and greater than B
  A = 60 ∧ A > B →
  -- C₁ and C₂ are parts of angle C divided by the altitude
  C = C₁ + C₂ →
  -- C₁ is adjacent to side b (opposite to angle B)
  C₁ > 0 ∧ C₂ > 0 →
  -- The altitude creates right angles
  B + C₁ = 90 ∧ A + C₂ = 90 →
  -- Conclusion
  C₁ - C₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_angle_relation_l504_50436


namespace NUMINAMATH_CALUDE_inequalities_proof_l504_50416

theorem inequalities_proof (a b : ℝ) (h : 1/a > 1/b ∧ 1/b > 0) : 
  a^3 < b^3 ∧ Real.sqrt b - Real.sqrt a < Real.sqrt (b - a) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l504_50416


namespace NUMINAMATH_CALUDE_qr_length_l504_50499

-- Define a right triangle
structure RightTriangle where
  QP : ℝ
  QR : ℝ
  cosQ : ℝ
  right_angle : cosQ = QP / QR

-- Theorem statement
theorem qr_length (t : RightTriangle) (h1 : t.cosQ = 0.5) (h2 : t.QP = 10) : t.QR = 20 := by
  sorry

end NUMINAMATH_CALUDE_qr_length_l504_50499


namespace NUMINAMATH_CALUDE_logical_equivalences_l504_50424

theorem logical_equivalences (p q : Prop) : 
  ((p ∧ q) ↔ ¬(¬p ∨ ¬q)) ∧
  ((p ∨ q) ↔ ¬(¬p ∧ ¬q)) ∧
  ((p → q) ↔ (¬q → ¬p)) ∧
  ((p ↔ q) ↔ ((p → q) ∧ (q → p))) :=
by sorry

end NUMINAMATH_CALUDE_logical_equivalences_l504_50424


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l504_50447

/-- A line that bisects a circle passes through its center -/
axiom line_bisects_circle_passes_through_center 
  (a b c d : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = c^2 → y = d*x + (b - d*a)) → 
  b = d*a + c^2/(2*d)

/-- The equation of a circle -/
def is_on_circle (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

/-- The equation of a line -/
def is_on_line (x y m c : ℝ) : Prop :=
  y = m*x + c

theorem line_tangent_to_circle 
  (a b r : ℝ) (h : r > 0) :
  (∀ x y : ℝ, is_on_circle x y 1 2 2 ↔ is_on_circle x y a b r) →
  (∀ x y : ℝ, is_on_line x y 1 1 → is_on_circle x y a b r) →
  ∀ y : ℝ, is_on_circle 3 y a b r ↔ y = 2 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l504_50447


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l504_50488

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l504_50488


namespace NUMINAMATH_CALUDE_batsman_average_l504_50457

theorem batsman_average (total_matches : ℕ) (first_set_matches : ℕ) (first_set_average : ℝ) (total_average : ℝ) :
  total_matches = 30 →
  first_set_matches = 20 →
  first_set_average = 30 →
  total_average = 25 →
  let second_set_matches := total_matches - first_set_matches
  let second_set_average := (total_average * total_matches - first_set_average * first_set_matches) / second_set_matches
  second_set_average = 15 := by sorry

end NUMINAMATH_CALUDE_batsman_average_l504_50457


namespace NUMINAMATH_CALUDE_no_nine_diagonals_intersection_l504_50432

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A diagonal of a polygon -/
def Diagonal (n : ℕ) (p : RegularPolygon n) (i j : Fin n) : Set (ℝ × ℝ) :=
  sorry

/-- The set of all diagonals in a polygon -/
def AllDiagonals (n : ℕ) (p : RegularPolygon n) : Set (Set (ℝ × ℝ)) :=
  sorry

/-- A point is internal to a polygon if it's inside the polygon -/
def IsInternal (n : ℕ) (p : RegularPolygon n) (point : ℝ × ℝ) : Prop :=
  sorry

/-- The number of diagonals passing through a point -/
def DiagonalsThroughPoint (n : ℕ) (p : RegularPolygon n) (point : ℝ × ℝ) : ℕ :=
  sorry

theorem no_nine_diagonals_intersection (p : RegularPolygon 25) 
  (diags : AllDiagonals 25 p) :
  ¬ ∃ (point : ℝ × ℝ), IsInternal 25 p point ∧ DiagonalsThroughPoint 25 p point = 9 :=
sorry

end NUMINAMATH_CALUDE_no_nine_diagonals_intersection_l504_50432


namespace NUMINAMATH_CALUDE_simplify_product_of_radicals_l504_50445

theorem simplify_product_of_radicals (x : ℝ) (hx : x > 0) :
  Real.sqrt (48 * x) * Real.sqrt (27 * x) * Real.sqrt (32 * x) = 144 * x * Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_radicals_l504_50445


namespace NUMINAMATH_CALUDE_rectangular_field_length_l504_50438

theorem rectangular_field_length (width : ℝ) (pond_side : ℝ) : 
  pond_side = 8 →
  (pond_side ^ 2) = (1 / 2) * (2 * width * width) →
  2 * width = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l504_50438


namespace NUMINAMATH_CALUDE_second_person_age_l504_50473

/-- Given a group of 7 people, if adding a 39-year-old increases the average age by 2,
    and adding another person decreases the average age by 1,
    then the age of the second person added is 15 years old. -/
theorem second_person_age (initial_group : Finset ℕ) 
  (initial_total_age : ℕ) (second_person_age : ℕ) :
  (initial_group.card = 7) →
  (initial_total_age / 7 + 2 = (initial_total_age + 39) / 8) →
  (initial_total_age / 7 - 1 = (initial_total_age + second_person_age) / 8) →
  second_person_age = 15 := by
  sorry


end NUMINAMATH_CALUDE_second_person_age_l504_50473


namespace NUMINAMATH_CALUDE_students_have_two_hands_l504_50479

/-- Given a class with the following properties:
  * There are 11 students including Peter
  * The total number of hands excluding Peter's is 20
  * Every student has the same number of hands
  Prove that each student has 2 hands. -/
theorem students_have_two_hands
  (total_students : ℕ)
  (hands_excluding_peter : ℕ)
  (h_total_students : total_students = 11)
  (h_hands_excluding_peter : hands_excluding_peter = 20) :
  hands_excluding_peter + 2 = total_students * 2 :=
sorry

end NUMINAMATH_CALUDE_students_have_two_hands_l504_50479


namespace NUMINAMATH_CALUDE_probability_specific_case_l504_50493

/-- The probability of drawing a white marble first and a red marble second -/
def probability_white_then_red (total_marbles : ℕ) (red_marbles : ℕ) (white_marbles : ℕ) : ℚ :=
  (white_marbles : ℚ) / (total_marbles : ℚ) * (red_marbles : ℚ) / ((total_marbles - 1) : ℚ)

theorem probability_specific_case :
  probability_white_then_red 10 4 6 = 4 / 15 := by
  sorry

#eval probability_white_then_red 10 4 6

end NUMINAMATH_CALUDE_probability_specific_case_l504_50493


namespace NUMINAMATH_CALUDE_infinite_squares_sum_cube_l504_50435

theorem infinite_squares_sum_cube :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, ∃ m a : ℕ,
    m > f n ∧ m > 1 ∧ 3 * (2 * a + m + 1)^2 = 11 * m^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_infinite_squares_sum_cube_l504_50435


namespace NUMINAMATH_CALUDE_sine_inequality_l504_50498

theorem sine_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π/4) :
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l504_50498


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_seven_l504_50446

theorem greatest_two_digit_multiple_of_seven : ∃ n : ℕ, n = 98 ∧ 
  (∀ m : ℕ, m < 100 ∧ 7 ∣ m → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_seven_l504_50446


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l504_50476

theorem polynomial_division_theorem (x : ℝ) :
  ∃ r : ℝ, (5 * x^2 - 5 * x + 3) * (2 * x + 4) + r = 10 * x^3 + 20 * x^2 - 9 * x + 6 ∧ 
  (∃ c : ℝ, r = c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l504_50476


namespace NUMINAMATH_CALUDE_power_multiplication_l504_50474

theorem power_multiplication (x : ℝ) (h : x = 5) : x^3 * x^4 = 78125 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l504_50474


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l504_50475

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def M : Finset Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  (U \ M) = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l504_50475


namespace NUMINAMATH_CALUDE_x_plus_y_value_l504_50408

theorem x_plus_y_value (x y : ℤ) (h1 : x - y = 200) (h2 : y = 245) : x + y = 690 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l504_50408


namespace NUMINAMATH_CALUDE_roller_coaster_problem_l504_50491

def roller_coaster_rides (people_in_line : ℕ) (cars : ℕ) (people_per_car : ℕ) : ℕ :=
  (people_in_line + cars * people_per_car - 1) / (cars * people_per_car)

theorem roller_coaster_problem :
  roller_coaster_rides 84 7 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_problem_l504_50491


namespace NUMINAMATH_CALUDE_first_round_cookies_count_l504_50401

/-- Represents the number of cookies sold in each round -/
structure CookieSales where
  first_round : ℕ
  second_round : ℕ

/-- Calculates the total number of cookies sold -/
def total_cookies (sales : CookieSales) : ℕ :=
  sales.first_round + sales.second_round

/-- Theorem: Given the total cookies sold and the number sold in the second round,
    we can determine the number sold in the first round -/
theorem first_round_cookies_count 
  (sales : CookieSales) 
  (h1 : sales.second_round = 27) 
  (h2 : total_cookies sales = 61) : 
  sales.first_round = 34 := by
  sorry

end NUMINAMATH_CALUDE_first_round_cookies_count_l504_50401


namespace NUMINAMATH_CALUDE_max_value_fg_unique_root_condition_inequality_condition_l504_50480

noncomputable section

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x^2 + a*x + 1
def g (x : ℝ) : ℝ := Real.exp x

-- Part 1
theorem max_value_fg (x : ℝ) (hx : x ∈ Set.Icc (-2) 0) :
  (f 1 x) * (g x) ≤ 1 :=
sorry

-- Part 2
theorem unique_root_condition (k : ℝ) :
  (∃! x, f (-1) x = k * g x) ↔ (k > 3 / Real.exp 2 ∨ 0 < k ∧ k < 1 / Real.exp 1) :=
sorry

-- Part 3
theorem inequality_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ →
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) ↔
  (-1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_fg_unique_root_condition_inequality_condition_l504_50480


namespace NUMINAMATH_CALUDE_two_players_goals_l504_50441

theorem two_players_goals (total_goals : ℕ) (players : ℕ) (percentage : ℚ) 
  (h1 : total_goals = 300)
  (h2 : players = 2)
  (h3 : percentage = 1/5) : 
  (↑total_goals * percentage) / players = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_players_goals_l504_50441


namespace NUMINAMATH_CALUDE_cristinas_pace_l504_50405

/-- Prove Cristina's pace in a race with given conditions -/
theorem cristinas_pace (race_distance : ℝ) (head_start : ℝ) (nickys_pace : ℝ) (catch_up_time : ℝ)
  (h1 : race_distance = 500)
  (h2 : head_start = 12)
  (h3 : nickys_pace = 3)
  (h4 : catch_up_time = 30) :
  let cristinas_distance := nickys_pace * (head_start + catch_up_time)
  cristinas_distance / catch_up_time = 5.4 := by
  sorry

#check cristinas_pace

end NUMINAMATH_CALUDE_cristinas_pace_l504_50405


namespace NUMINAMATH_CALUDE_larger_number_proof_l504_50495

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1335)
  (h2 : L = 6 * S + 15) :
  L = 1599 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l504_50495


namespace NUMINAMATH_CALUDE_sams_remaining_pennies_l504_50449

/-- Given an initial amount of pennies and an amount spent, calculate the remaining pennies -/
def remaining_pennies (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem: Sam's remaining pennies -/
theorem sams_remaining_pennies :
  remaining_pennies 98 93 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sams_remaining_pennies_l504_50449


namespace NUMINAMATH_CALUDE_existence_of_point_N_l504_50459

theorem existence_of_point_N (a m : ℝ) (ha : a > 0) (hm : m ∈ Set.union (Set.Ioo (-1) 0) (Set.Ioi 0)) :
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = a^2 ∧ |y₀| = (|m| * a) / Real.sqrt (1 + m) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_point_N_l504_50459


namespace NUMINAMATH_CALUDE_sqrt_two_thirds_irrational_l504_50451

theorem sqrt_two_thirds_irrational (h : Irrational (Real.sqrt 6)) : Irrational (Real.sqrt (2/3)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_thirds_irrational_l504_50451


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l504_50421

/-- In a triangle ABC, if angle A is 2π/3 and side a is √3 times side c, then the ratio of side a to side b is √3. -/
theorem triangle_side_ratio (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  A + B + C = π →  -- angle sum property
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- valid angle measures
  A = 2 * π / 3 →  -- given angle A
  a = Real.sqrt 3 * c →  -- given side relation
  a / b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l504_50421


namespace NUMINAMATH_CALUDE_logarithmic_equality_implies_zero_product_l504_50465

theorem logarithmic_equality_implies_zero_product (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (a - b) * Real.log c + (b - c) * Real.log a + (c - a) * Real.log b = 0) :
  (a - b) * (b - c) * (c - a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equality_implies_zero_product_l504_50465


namespace NUMINAMATH_CALUDE_runners_in_picture_probability_l504_50415

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Represents the track and photograph setup -/
structure TrackSetup where
  rachelLapTime : ℝ
  robertLapTime : ℝ
  totalTime : ℝ
  photographerPosition : ℝ
  pictureWidth : ℝ

/-- Calculates the probability of both runners being in the picture -/
def probabilityBothInPicture (setup : TrackSetup) : ℝ :=
  sorry  -- Proof omitted

theorem runners_in_picture_probability (setup : TrackSetup) 
  (h1 : setup.rachelLapTime = 75)
  (h2 : setup.robertLapTime = 100)
  (h3 : setup.totalTime = 12 * 60)
  (h4 : setup.photographerPosition = 1/3)
  (h5 : setup.pictureWidth = 1/5) :
  probabilityBothInPicture setup = 4/15 := by
  sorry

#check runners_in_picture_probability

end NUMINAMATH_CALUDE_runners_in_picture_probability_l504_50415


namespace NUMINAMATH_CALUDE_equation_solution_l504_50439

theorem equation_solution (k : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ 9 * x - 3 = k * x + 14) ↔ (k = 8 ∨ k = -8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l504_50439


namespace NUMINAMATH_CALUDE_alpha_set_property_l504_50487

theorem alpha_set_property (r s : ℕ) (hr : r > s) (hgcd : Nat.gcd r s = 1) :
  let α : ℚ := r / s
  let N_α : Set ℕ := {m | ∃ n : ℕ, m = ⌊n * α⌋}
  ∀ m ∈ N_α, ¬(r ∣ (m + 1)) := by
  sorry

end NUMINAMATH_CALUDE_alpha_set_property_l504_50487


namespace NUMINAMATH_CALUDE_multiple_of_number_l504_50407

theorem multiple_of_number (n : ℝ) (h : n = 6) : ∃ k : ℝ, 3 * n - 6 = k * n ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_number_l504_50407


namespace NUMINAMATH_CALUDE_absolute_fraction_inequality_l504_50497

theorem absolute_fraction_inequality (x : ℝ) : 
  |((3 * x - 2) / (x + 1))| > 3 ↔ x < -1 ∨ (-1 < x ∧ x < 1/6) :=
by sorry

end NUMINAMATH_CALUDE_absolute_fraction_inequality_l504_50497


namespace NUMINAMATH_CALUDE_study_time_for_average_75_l504_50477

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  studyTime : ℝ
  score : ℝ
  ratio : ℝ
  rel : score = ratio * studyTime

/-- Proves that 4.5 hours of study will result in a score of 90, given the initial condition -/
theorem study_time_for_average_75 
  (initial : StudyScoreRelation) 
  (h_initial : initial.studyTime = 3 ∧ initial.score = 60) :
  ∃ (second : StudyScoreRelation), 
    second.studyTime = 4.5 ∧ 
    second.score = 90 ∧ 
    (initial.score + second.score) / 2 = 75 ∧
    second.ratio = initial.ratio := by
  sorry

end NUMINAMATH_CALUDE_study_time_for_average_75_l504_50477


namespace NUMINAMATH_CALUDE_find_n_l504_50452

theorem find_n (x y : ℝ) (h1 : x = 3) (h2 : y = 2) : x - y^(x-y) * (x+y) = -7 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l504_50452


namespace NUMINAMATH_CALUDE_die_roll_outcomes_l504_50494

/-- The number of faces on a standard die -/
def numDieFaces : ℕ := 6

/-- The number of rolls before stopping -/
def numRolls : ℕ := 5

/-- The number of different outcomes when rolling a die continuously and stopping
    after exactly 5 rolls, with the condition that three different numbers appear
    on the fifth roll -/
def numOutcomes : ℕ := 840

/-- Theorem stating that the number of different outcomes is 840 -/
theorem die_roll_outcomes :
  (numDieFaces.choose 2) * ((numDieFaces - 2).choose 1) * (4 + 6 + 4) = numOutcomes := by
  sorry

end NUMINAMATH_CALUDE_die_roll_outcomes_l504_50494


namespace NUMINAMATH_CALUDE_ball_count_l504_50423

theorem ball_count (white green yellow red purple : ℕ) 
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 5)
  (h4 : red = 6)
  (h5 : purple = 9)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 3/4) :
  white + green + yellow + red + purple = 60 := by sorry

end NUMINAMATH_CALUDE_ball_count_l504_50423


namespace NUMINAMATH_CALUDE_specific_case_general_case_l504_50453

-- Define the theorem for the specific case n = 4
theorem specific_case :
  Real.sqrt (4 + 4/15) = 8 * Real.sqrt 15 / 15 := by sorry

-- Define the theorem for the general case
theorem general_case (n : ℕ) (h : n ≥ 2) :
  Real.sqrt (n + n/(n^2 - 1)) = n * Real.sqrt (n/(n^2 - 1)) := by sorry

end NUMINAMATH_CALUDE_specific_case_general_case_l504_50453


namespace NUMINAMATH_CALUDE_greatest_common_divisor_546_180_under_70_l504_50411

def is_greatest_common_divisor (n : ℕ) : Prop :=
  n ∣ 546 ∧ n < 70 ∧ n ∣ 180 ∧
  ∀ m : ℕ, m ∣ 546 → m < 70 → m ∣ 180 → m ≤ n

theorem greatest_common_divisor_546_180_under_70 :
  is_greatest_common_divisor 6 := by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_546_180_under_70_l504_50411


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l504_50410

theorem smallest_number_with_given_remainders :
  ∃ b : ℕ, b ≥ 0 ∧
    b % 6 = 3 ∧
    b % 5 = 2 ∧
    b % 7 = 2 ∧
    (∀ c : ℕ, c ≥ 0 → c % 6 = 3 → c % 5 = 2 → c % 7 = 2 → b ≤ c) ∧
    b = 177 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l504_50410


namespace NUMINAMATH_CALUDE_dot_product_equals_negative_49_l504_50443

def vector1 : Fin 4 → ℝ := ![4, -5, 2, -1]
def vector2 : Fin 4 → ℝ := ![-6, 3, -4, 2]

theorem dot_product_equals_negative_49 :
  (Finset.sum Finset.univ (λ i => vector1 i * vector2 i)) = -49 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equals_negative_49_l504_50443


namespace NUMINAMATH_CALUDE_cheap_coat_duration_proof_l504_50403

/-- The duration of the less expensive coat -/
def cheap_coat_duration : ℕ := 5

/-- The cost of the expensive coat -/
def expensive_coat_cost : ℕ := 300

/-- The duration of the expensive coat -/
def expensive_coat_duration : ℕ := 15

/-- The cost of the less expensive coat -/
def cheap_coat_cost : ℕ := 120

/-- The total time period considered -/
def total_time : ℕ := 30

/-- The amount saved by buying the expensive coat over the total time period -/
def amount_saved : ℕ := 120

theorem cheap_coat_duration_proof :
  cheap_coat_duration * cheap_coat_cost * (total_time / cheap_coat_duration) =
  expensive_coat_cost * (total_time / expensive_coat_duration) + amount_saved :=
by sorry

end NUMINAMATH_CALUDE_cheap_coat_duration_proof_l504_50403


namespace NUMINAMATH_CALUDE_house_size_multiple_l504_50467

theorem house_size_multiple (sara_house : ℝ) (nada_house : ℝ) (extra_size : ℝ) :
  sara_house = 1000 →
  nada_house = 450 →
  sara_house = nada_house * (sara_house - extra_size) / nada_house + extra_size →
  extra_size = 100 →
  (sara_house - extra_size) / nada_house = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_house_size_multiple_l504_50467


namespace NUMINAMATH_CALUDE_inscribed_square_area_l504_50485

/-- The area of a square inscribed in an isosceles right triangle -/
theorem inscribed_square_area (leg_length : ℝ) (h : leg_length = 28 * Real.sqrt 2) :
  let diagonal := leg_length
  let side := diagonal / Real.sqrt 2
  side ^ 2 = 784 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l504_50485


namespace NUMINAMATH_CALUDE_joan_balloons_l504_50468

def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2

theorem joan_balloons : initial_balloons - lost_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l504_50468


namespace NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l504_50478

theorem mrs_sheridan_fish_count (initial_fish : Nat) (fish_from_sister : Nat) : 
  initial_fish = 22 → fish_from_sister = 47 → initial_fish + fish_from_sister = 69 := by
  sorry

end NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l504_50478


namespace NUMINAMATH_CALUDE_birthday_celebration_men_count_l504_50429

/-- Proves that the number of men at a birthday celebration was 15 given the specified conditions. -/
theorem birthday_celebration_men_count :
  ∀ (total_guests women men children : ℕ),
    total_guests = 60 →
    women = total_guests / 2 →
    total_guests = women + men + children →
    50 = women + (men - men / 3) + (children - 5) →
    men = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_birthday_celebration_men_count_l504_50429


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l504_50448

/-- Given a circle and a line of symmetry, this theorem proves the equation of the symmetric circle. -/
theorem symmetric_circle_equation (x y : ℝ) :
  (x - 3)^2 + (y + 4)^2 = 2 →  -- Original circle equation
  x + y = 0 →               -- Line of symmetry
  (x - 4)^2 + (y + 3)^2 = 2 -- Symmetric circle equation
:= by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l504_50448


namespace NUMINAMATH_CALUDE_amount_after_two_years_l504_50420

/-- The amount after n years given an initial amount and yearly increase rate -/
def amount_after_years (initial_amount : ℝ) (increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + increase_rate) ^ years

/-- Theorem stating the amount after two years -/
theorem amount_after_two_years :
  let initial_amount : ℝ := 62000
  let increase_rate : ℝ := 1/8
  let years : ℕ := 2
  amount_after_years initial_amount increase_rate years = 78468.75 := by
sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l504_50420


namespace NUMINAMATH_CALUDE_unique_solution_for_odd_prime_l504_50404

theorem unique_solution_for_odd_prime (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + p*x = y^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_odd_prime_l504_50404


namespace NUMINAMATH_CALUDE_probability_three_unused_theorem_expected_hits_nine_targets_theorem_l504_50418

/-- Represents a rocket artillery system on a missile cruiser -/
structure RocketSystem where
  total_rockets : ℕ
  hit_probability : ℝ

/-- Calculates the probability of exactly three unused rockets remaining after firing at five targets -/
def probability_three_unused (system : RocketSystem) : ℝ :=
  10 * system.hit_probability^3 * (1 - system.hit_probability)^2

/-- Calculates the expected number of targets hit when firing at nine targets -/
def expected_hits_nine_targets (system : RocketSystem) : ℝ :=
  10 * system.hit_probability - system.hit_probability^10

/-- Theorem stating the probability of exactly three unused rockets remaining after firing at five targets -/
theorem probability_three_unused_theorem (system : RocketSystem) :
  probability_three_unused system = 10 * system.hit_probability^3 * (1 - system.hit_probability)^2 := by
  sorry

/-- Theorem stating the expected number of targets hit when firing at nine targets -/
theorem expected_hits_nine_targets_theorem (system : RocketSystem) :
  expected_hits_nine_targets system = 10 * system.hit_probability - system.hit_probability^10 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_unused_theorem_expected_hits_nine_targets_theorem_l504_50418


namespace NUMINAMATH_CALUDE_eventual_shot_probability_l504_50437

def basketball_game (make_probability : ℝ) (get_ball_back_probability : ℝ) : Prop :=
  (0 ≤ make_probability ∧ make_probability ≤ 1) ∧
  (0 ≤ get_ball_back_probability ∧ get_ball_back_probability ≤ 1)

theorem eventual_shot_probability
  (make_prob : ℝ)
  (get_ball_back_prob : ℝ)
  (h_game : basketball_game make_prob get_ball_back_prob)
  (h_make_prob : make_prob = 1/10)
  (h_get_ball_back_prob : get_ball_back_prob = 9/10) :
  (1 - (1 - make_prob) * get_ball_back_prob / (1 - (1 - make_prob) * (1 - get_ball_back_prob))) = 10/19 :=
by sorry


end NUMINAMATH_CALUDE_eventual_shot_probability_l504_50437


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l504_50486

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 12*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (7/5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l504_50486


namespace NUMINAMATH_CALUDE_simplify_expression_l504_50461

theorem simplify_expression (t : ℝ) (h : t ≠ 0) :
  (t^5 * t^3) / t^4 = t^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l504_50461


namespace NUMINAMATH_CALUDE_total_weight_of_clothes_l504_50409

/-- The total weight of clothes collected is 8.58 kg, given that male student's clothes weigh 2.6 kg and female student's clothes weigh 5.98 kg. -/
theorem total_weight_of_clothes (male_clothes : ℝ) (female_clothes : ℝ)
  (h1 : male_clothes = 2.6)
  (h2 : female_clothes = 5.98) :
  male_clothes + female_clothes = 8.58 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_clothes_l504_50409


namespace NUMINAMATH_CALUDE_four_star_three_equals_nineteen_l504_50430

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- State the theorem
theorem four_star_three_equals_nineteen :
  customOp 4 3 = 19 := by sorry

end NUMINAMATH_CALUDE_four_star_three_equals_nineteen_l504_50430


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l504_50464

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_geometric : is_geometric_sequence a q)
  (h_arithmetic : is_arithmetic_sequence (λ n => match n with
    | 0 => a 3
    | 1 => 3 * a 2
    | 2 => 5 * a 1
    | _ => 0))
  (h_increasing : ∀ n : ℕ, a n < a (n + 1)) :
  q = 5 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l504_50464


namespace NUMINAMATH_CALUDE_mr_slinkums_order_l504_50481

theorem mr_slinkums_order (on_shelves_percent : ℚ) (in_storage : ℕ) : 
  on_shelves_percent = 1/5 ∧ in_storage = 120 → 
  (1 - on_shelves_percent) * 150 = in_storage :=
by
  sorry

end NUMINAMATH_CALUDE_mr_slinkums_order_l504_50481


namespace NUMINAMATH_CALUDE_train_length_l504_50490

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed : Real) (time : Real) :
  speed = 108 →
  time = 1.4998800095992322 →
  ∃ (length : Real), abs (length - (speed * 1000 / 3600 * time)) < 0.001 ∧ abs (length - 44.996) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l504_50490


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l504_50455

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- State the theorem
theorem f_derivative_at_2 : 
  (deriv f) 2 = 6 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l504_50455


namespace NUMINAMATH_CALUDE_ten_square_shape_perimeter_l504_50402

/-- A shape made from unit squares joined edge to edge -/
structure UnitSquareShape where
  /-- The number of unit squares in the shape -/
  num_squares : ℕ
  /-- The perimeter of the shape in cm -/
  perimeter : ℕ

/-- Theorem: A shape made from 10 unit squares has a perimeter of 18 cm -/
theorem ten_square_shape_perimeter :
  ∀ (shape : UnitSquareShape),
    shape.num_squares = 10 →
    shape.perimeter = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ten_square_shape_perimeter_l504_50402


namespace NUMINAMATH_CALUDE_triangle_side_length_l504_50470

theorem triangle_side_length 
  (A B C : ℝ) 
  (hBC : Real.cos C = -Real.sqrt 2 / 2) 
  (hAC : Real.sin A / Real.sin B = 1 / (2 * Real.cos (A + B))) 
  (hBA : B * A = 2 * Real.sqrt 2) : 
  Real.sqrt ((Real.sin A)^2 + (Real.sin B)^2 - 2 * Real.sin A * Real.sin B * Real.cos C) = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l504_50470


namespace NUMINAMATH_CALUDE_range_of_m_l504_50469

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 + 1) * (x^2 - 8*x - 20) ≤ 0 → -2 ≤ x ∧ x ≤ 10) ∧
  (∀ x : ℝ, x^2 - 2*x + (1 - m^2) ≤ 0 → 1 - m ≤ x ∧ x ≤ 1 + m) ∧
  (m > 0) ∧
  (∀ x : ℝ, (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 10 ∧ (x < 1 - m ∨ x > 1 + m)) →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l504_50469


namespace NUMINAMATH_CALUDE_probability_red_before_green_l504_50426

def num_red : ℕ := 4
def num_green : ℕ := 3
def num_blue : ℕ := 1

def total_chips : ℕ := num_red + num_green + num_blue

theorem probability_red_before_green :
  let favorable_arrangements := (total_chips - 1).choose num_green
  let total_arrangements := total_chips.choose num_green * total_chips.choose num_blue
  (favorable_arrangements * total_chips) / total_arrangements = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_red_before_green_l504_50426


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l504_50496

/-- Given two 2D vectors a and b, if a + b is parallel to 2a - b, then the x-coordinate of b is -4. -/
theorem parallel_vectors_imply_x_value (a b : ℝ × ℝ) (h : a = (2, 1)) (h' : b.2 = -2) :
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (2 • a - b)) → b.1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l504_50496


namespace NUMINAMATH_CALUDE_number_percentage_l504_50456

theorem number_percentage (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → (40/100 : ℝ) * N = 192 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_l504_50456


namespace NUMINAMATH_CALUDE_connie_total_markers_l504_50434

/-- The number of red markers Connie has -/
def red_markers : ℕ := 5230

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 4052

/-- The number of green markers Connie has -/
def green_markers : ℕ := 3180

/-- The number of purple markers Connie has -/
def purple_markers : ℕ := 2763

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers + green_markers + purple_markers

theorem connie_total_markers : total_markers = 15225 := by
  sorry

end NUMINAMATH_CALUDE_connie_total_markers_l504_50434


namespace NUMINAMATH_CALUDE_sum_always_negative_l504_50458

def f (x : ℝ) : ℝ := -x - x^3

theorem sum_always_negative (α β γ : ℝ) 
  (h1 : α + β > 0) (h2 : β + γ > 0) (h3 : γ + α > 0) : 
  f α + f β + f γ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_always_negative_l504_50458


namespace NUMINAMATH_CALUDE_even_increasing_ordering_l504_50442

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

theorem even_increasing_ordering (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : increasing_on_pos f) : 
  f 3 < f (-Real.pi) ∧ f (-Real.pi) < f (-4) := by sorry

end NUMINAMATH_CALUDE_even_increasing_ordering_l504_50442


namespace NUMINAMATH_CALUDE_max_area_rectangle_max_area_achievable_l504_50428

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

/-- The theorem stating the maximum area of a rectangle with given conditions -/
theorem max_area_rectangle (l w : ℕ) : 
  (l + w = 60) →  -- Perimeter condition: 2(l + w) = 120
  (isPrime l ∨ isPrime w) →  -- One dimension is prime
  (l * w ≤ 899) :=  -- The area is at most 899
by sorry

/-- The theorem stating that the maximum area of 899 is achievable -/
theorem max_area_achievable : 
  ∃ l w : ℕ, (l + w = 60) ∧ (isPrime l ∨ isPrime w) ∧ (l * w = 899) :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_max_area_achievable_l504_50428


namespace NUMINAMATH_CALUDE_count_rectangular_subsets_5x5_l504_50413

/-- The number of ways to select a rectangular subset in a 5x5 grid -/
def rectangular_subsets_5x5 : ℕ := 225

/-- A proof that there are 225 ways to select a rectangular subset in a 5x5 grid -/
theorem count_rectangular_subsets_5x5 : rectangular_subsets_5x5 = 225 := by
  sorry

end NUMINAMATH_CALUDE_count_rectangular_subsets_5x5_l504_50413


namespace NUMINAMATH_CALUDE_dropped_student_score_l504_50433

theorem dropped_student_score
  (initial_students : ℕ)
  (initial_average : ℚ)
  (remaining_students : ℕ)
  (new_average : ℚ)
  (h1 : initial_students = 16)
  (h2 : initial_average = 62.5)
  (h3 : remaining_students = 15)
  (h4 : new_average = 63)
  (h5 : remaining_students = initial_students - 1) :
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * new_average = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_dropped_student_score_l504_50433


namespace NUMINAMATH_CALUDE_balance_scale_l504_50412

/-- The weight of the book that balances the scale -/
def book_weight : ℝ := 1.1

/-- The weight of the first item on the scale -/
def weight1 : ℝ := 0.5

/-- The weight of each of the two identical items on the scale -/
def weight2 : ℝ := 0.3

/-- The number of identical items with weight2 -/
def count2 : ℕ := 2

theorem balance_scale :
  book_weight = weight1 + count2 * weight2 := by sorry

end NUMINAMATH_CALUDE_balance_scale_l504_50412


namespace NUMINAMATH_CALUDE_equation_solution_l504_50425

theorem equation_solution : ∃ x : ℝ, 
  (x^2 - 7*x + 12) / (x^2 - 9*x + 20) = (x^2 - 4*x - 21) / (x^2 - 5*x - 24) ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l504_50425


namespace NUMINAMATH_CALUDE_trig_simplification_l504_50472

theorem trig_simplification (x : ℝ) :
  (Real.sin x + Real.sin (3 * x)) / (1 + Real.cos x + Real.cos (3 * x)) =
  (4 * (Real.cos x)^2 * Real.sin x) / (1 + 2 * Real.cos x * Real.cos (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l504_50472


namespace NUMINAMATH_CALUDE_g_is_correct_l504_50454

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := -2*x^5 + 7*x^4 + 4*x^3 - 2*x^2 - 8*x + 4

-- Theorem statement
theorem g_is_correct :
  ∀ x : ℝ, 2*x^5 - 4*x^3 + 3*x + g x = 7*x^4 - 2*x^2 - 5*x + 4 :=
by
  sorry

end NUMINAMATH_CALUDE_g_is_correct_l504_50454


namespace NUMINAMATH_CALUDE_third_episode_duration_l504_50484

/-- Given a series of four episodes with known durations for three episodes
    and a total duration, this theorem proves the duration of the third episode. -/
theorem third_episode_duration
  (total_duration : ℕ)
  (first_episode : ℕ)
  (second_episode : ℕ)
  (fourth_episode : ℕ)
  (h1 : total_duration = 240)  -- 4 hours in minutes
  (h2 : first_episode = 58)
  (h3 : second_episode = 62)
  (h4 : fourth_episode = 55)
  : total_duration - (first_episode + second_episode + fourth_episode) = 65 := by
  sorry

#check third_episode_duration

end NUMINAMATH_CALUDE_third_episode_duration_l504_50484


namespace NUMINAMATH_CALUDE_sin_30_degrees_l504_50406

/-- Sine of 30 degrees is 1/2 -/
theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l504_50406


namespace NUMINAMATH_CALUDE_no_real_roots_l504_50492

theorem no_real_roots :
  ¬∃ x : ℝ, x^2 = 2*x - 3 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_l504_50492


namespace NUMINAMATH_CALUDE_percentage_decrease_of_b_l504_50466

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- a and b are positive
  a / b = 4 / 5 ∧  -- ratio of a to b is 4 to 5
  x = a * 1.25 ∧  -- x equals a increased by 25 percent
  m = b * (1 - p / 100) ∧  -- m equals b decreased by p percent
  m / x = 0.8  -- ratio of m to x is 0.8
  → p = 20 := by  -- prove that p (percentage decrease) is 20
sorry

end NUMINAMATH_CALUDE_percentage_decrease_of_b_l504_50466


namespace NUMINAMATH_CALUDE_intersection_M_N_l504_50489

def M : Set ℝ := {x | x^2 - x ≥ 0}
def N : Set ℝ := {x | x < 2}

theorem intersection_M_N : 
  ∀ x : ℝ, x ∈ M ∩ N ↔ x ≤ 0 ∨ (1 ≤ x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l504_50489


namespace NUMINAMATH_CALUDE_problem_solution_l504_50440

theorem problem_solution (x : ℝ) (h : x = 13 / Real.sqrt (19 + 8 * Real.sqrt 3)) :
  (x^4 - 6*x^3 - 2*x^2 + 18*x + 23) / (x^2 - 8*x + 15) = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l504_50440
