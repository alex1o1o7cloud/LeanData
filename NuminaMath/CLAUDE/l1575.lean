import Mathlib

namespace NUMINAMATH_CALUDE_num_lizards_seen_l1575_157573

/-- The number of legs Borgnine wants to see at the zoo -/
def total_legs : ℕ := 1100

/-- The number of chimps Borgnine has seen -/
def num_chimps : ℕ := 12

/-- The number of lions Borgnine has seen -/
def num_lions : ℕ := 8

/-- The number of tarantulas Borgnine will see -/
def num_tarantulas : ℕ := 125

/-- The number of legs a chimp has -/
def chimp_legs : ℕ := 4

/-- The number of legs a lion has -/
def lion_legs : ℕ := 4

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

/-- The number of legs a lizard has -/
def lizard_legs : ℕ := 4

/-- The theorem stating the number of lizards Borgnine has seen -/
theorem num_lizards_seen : 
  (total_legs - (num_chimps * chimp_legs + num_lions * lion_legs + num_tarantulas * tarantula_legs)) / lizard_legs = 5 := by
  sorry

end NUMINAMATH_CALUDE_num_lizards_seen_l1575_157573


namespace NUMINAMATH_CALUDE_square_plus_one_l1575_157542

theorem square_plus_one (a : ℝ) (h : a^2 + 2*a - 2 = 0) : (a + 1)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_l1575_157542


namespace NUMINAMATH_CALUDE_aladdin_travel_l1575_157581

/-- A continuous function that takes all values in [0,1) -/
def equator_travel (φ : ℝ → ℝ) : Prop :=
  Continuous φ ∧ ∀ y : ℝ, 0 ≤ y ∧ y < 1 → ∃ t : ℝ, φ t = y

/-- The maximum difference between any two values of φ is at least 1 -/
theorem aladdin_travel (φ : ℝ → ℝ) (h : equator_travel φ) :
  ∃ t₁ t₂ : ℝ, |φ t₁ - φ t₂| ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_aladdin_travel_l1575_157581


namespace NUMINAMATH_CALUDE_sphere_volume_l1575_157567

theorem sphere_volume (R : ℝ) (h : R = 3) : (4 / 3 : ℝ) * Real.pi * R^3 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l1575_157567


namespace NUMINAMATH_CALUDE_intersection_point_in_circle_range_l1575_157517

theorem intersection_point_in_circle_range (m : ℝ) : 
  let M : ℝ × ℝ := (1, 1)
  let line1 : ℝ × ℝ → Prop := λ p => p.1 + p.2 - 2 = 0
  let line2 : ℝ × ℝ → Prop := λ p => 3 * p.1 - p.2 - 2 = 0
  let circle : ℝ × ℝ → Prop := λ p => (p.1 - m)^2 + p.2^2 < 5
  (line1 M ∧ line2 M ∧ circle M) ↔ -1 < m ∧ m < 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_in_circle_range_l1575_157517


namespace NUMINAMATH_CALUDE_hospital_employee_arrangements_l1575_157516

theorem hospital_employee_arrangements (n : ℕ) (h : n = 6) :
  (Nat.factorial n = 720) ∧
  (Nat.factorial (n - 1) = 120) ∧
  (n * (n - 1) * (n - 2) = 120) := by
  sorry

#check hospital_employee_arrangements

end NUMINAMATH_CALUDE_hospital_employee_arrangements_l1575_157516


namespace NUMINAMATH_CALUDE_square_roots_calculation_l1575_157561

theorem square_roots_calculation : (Real.sqrt 3 + Real.sqrt 2)^2 * (5 - 2 * Real.sqrt 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_calculation_l1575_157561


namespace NUMINAMATH_CALUDE_squirrel_climb_l1575_157556

theorem squirrel_climb (x : ℝ) : 
  (∀ n : ℕ, n > 0 → (2 * n - 1) * x - 2 * (n - 1) = 26) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_climb_l1575_157556


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1575_157588

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1575_157588


namespace NUMINAMATH_CALUDE_unique_function_property_l1575_157568

def last_digit (n : ℕ) : ℕ := n % 10

def is_constant_one (f : ℕ → ℕ) : Prop :=
  ∀ n, f n = 1

theorem unique_function_property (f : ℕ → ℕ) 
  (h1 : ∀ x y, f (x * y) = f x * f y)
  (h2 : f 30 = 1)
  (h3 : ∀ n, last_digit n = 7 → f n = 1) :
  is_constant_one f :=
sorry

end NUMINAMATH_CALUDE_unique_function_property_l1575_157568


namespace NUMINAMATH_CALUDE_fourth_power_nested_root_l1575_157547

theorem fourth_power_nested_root : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_nested_root_l1575_157547


namespace NUMINAMATH_CALUDE_unique_number_problem_l1575_157528

theorem unique_number_problem : ∃! (x : ℝ), x > 0 ∧ (((x^2 / 3)^3) / 9) = x :=
by sorry

end NUMINAMATH_CALUDE_unique_number_problem_l1575_157528


namespace NUMINAMATH_CALUDE_stream_speed_l1575_157540

/-- Given a boat with a speed in still water and its travel time and distance downstream,
    calculate the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  boat_speed = 24 →
  time = 4 →
  distance = 112 →
  distance = (boat_speed + (distance / time - boat_speed)) * time →
  distance / time - boat_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1575_157540


namespace NUMINAMATH_CALUDE_old_belt_time_correct_l1575_157592

/-- The time it takes for the old conveyor belt to move one day's coal output -/
def old_belt_time : ℝ := 21

/-- The time it takes for the new conveyor belt to move one day's coal output -/
def new_belt_time : ℝ := 15

/-- The time it takes for both belts together to move one day's coal output -/
def combined_time : ℝ := 8.75

/-- Theorem stating that the old conveyor belt time is correct given the conditions -/
theorem old_belt_time_correct :
  1 / old_belt_time + 1 / new_belt_time = 1 / combined_time :=
by sorry

end NUMINAMATH_CALUDE_old_belt_time_correct_l1575_157592


namespace NUMINAMATH_CALUDE_janice_stairs_l1575_157553

/-- The number of flights of stairs to reach Janice's office -/
def flights_per_staircase : ℕ := 3

/-- The number of times Janice goes up the stairs in a day -/
def times_up : ℕ := 5

/-- The number of times Janice goes down the stairs in a day -/
def times_down : ℕ := 3

/-- The total number of flights Janice walks up in a day -/
def flights_up : ℕ := flights_per_staircase * times_up

/-- The total number of flights Janice walks down in a day -/
def flights_down : ℕ := flights_per_staircase * times_down

/-- The total number of flights Janice walks in a day -/
def total_flights : ℕ := flights_up + flights_down

theorem janice_stairs : total_flights = 24 := by
  sorry

end NUMINAMATH_CALUDE_janice_stairs_l1575_157553


namespace NUMINAMATH_CALUDE_isabel_ds_games_left_l1575_157562

/-- Given that Isabel initially had 90 DS games and gave 87 to her friend,
    prove that she has 3 DS games left. -/
theorem isabel_ds_games_left (initial_games : ℕ) (games_given : ℕ) (games_left : ℕ) : 
  initial_games = 90 → games_given = 87 → games_left = initial_games - games_given → games_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_isabel_ds_games_left_l1575_157562


namespace NUMINAMATH_CALUDE_peach_distribution_theorem_l1575_157564

/-- Represents the number of peaches each child received -/
structure PeachDistribution where
  anya : Nat
  katya : Nat
  liza : Nat
  dasha : Nat
  kolya : Nat
  petya : Nat
  tolya : Nat
  vasya : Nat

/-- Represents the last names of the children -/
inductive LastName
  | Ivanov
  | Grishin
  | Andreyev
  | Sergeyev

/-- Represents a child with their name and last name -/
structure Child where
  name : String
  lastName : LastName

/-- The theorem stating the correct distribution of peaches and last names -/
theorem peach_distribution_theorem (d : PeachDistribution) 
  (h1 : d.anya = 1)
  (h2 : d.katya = 2)
  (h3 : d.liza = 3)
  (h4 : d.dasha = 4)
  (h5 : d.kolya = d.liza)
  (h6 : d.petya = 2 * d.dasha)
  (h7 : d.tolya = 3 * d.anya)
  (h8 : d.vasya = 4 * d.katya)
  (h9 : d.anya + d.katya + d.liza + d.dasha + d.kolya + d.petya + d.tolya + d.vasya = 32) :
  ∃ (c1 c2 c3 c4 : Child),
    c1 = { name := "Liza", lastName := LastName.Ivanov } ∧
    c2 = { name := "Dasha", lastName := LastName.Grishin } ∧
    c3 = { name := "Anya", lastName := LastName.Andreyev } ∧
    c4 = { name := "Katya", lastName := LastName.Sergeyev } := by
  sorry

end NUMINAMATH_CALUDE_peach_distribution_theorem_l1575_157564


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1575_157579

theorem complex_equation_solution (z : ℂ) :
  (2 * z - Complex.I) * (2 - Complex.I) = 5 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1575_157579


namespace NUMINAMATH_CALUDE_inverse_function_solution_l1575_157541

/-- Given a function g(x) = 1 / (cx + d) where c and d are nonzero constants,
    prove that the solution to g^(-1)(x) = 0 is x = 1/d -/
theorem inverse_function_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  let g : ℝ → ℝ := fun x ↦ 1 / (c * x + d)
  (Function.invFun g) 0 = 1 / d := by
  sorry


end NUMINAMATH_CALUDE_inverse_function_solution_l1575_157541


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l1575_157509

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def total_cost : ℚ := 33.56

theorem jacket_cost_calculation :
  total_cost - shorts_cost - shirt_cost = 7.43 := by sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l1575_157509


namespace NUMINAMATH_CALUDE_product_mod_thousand_l1575_157526

theorem product_mod_thousand : (1234 * 5678) % 1000 = 652 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_thousand_l1575_157526


namespace NUMINAMATH_CALUDE_stratified_sampling_class_c_l1575_157507

theorem stratified_sampling_class_c (total_students : ℕ) (class_a class_b class_c class_d sample_size : ℕ) : 
  total_students = class_a + class_b + class_c + class_d →
  class_a = 75 →
  class_b = 75 →
  class_c = 200 →
  class_d = 150 →
  sample_size = 20 →
  (class_c * sample_size) / total_students = 8 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_class_c_l1575_157507


namespace NUMINAMATH_CALUDE_subtract_negative_term_l1575_157545

theorem subtract_negative_term (a : ℝ) :
  (4 * a^2 - 3 * a + 7) - (-6 * a) = 4 * a^2 + 3 * a + 7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_term_l1575_157545


namespace NUMINAMATH_CALUDE_obtuse_triangle_count_l1575_157571

/-- A function that checks if a triangle with sides a, b, and c is obtuse -/
def is_obtuse (a b c : ℝ) : Prop :=
  (a^2 > b^2 + c^2) ∨ (b^2 > a^2 + c^2) ∨ (c^2 > a^2 + b^2)

/-- A function that checks if a triangle with sides a, b, and c is valid -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- The main theorem stating that there are exactly 13 positive integer values of k
    for which a triangle with sides 12, 16, and k is obtuse -/
theorem obtuse_triangle_count :
  (∃! (s : Finset ℕ), s.card = 13 ∧ 
    (∀ k, k ∈ s ↔ (is_valid_triangle 12 16 k ∧ is_obtuse 12 16 k))) := by
  sorry

end NUMINAMATH_CALUDE_obtuse_triangle_count_l1575_157571


namespace NUMINAMATH_CALUDE_largest_factor_is_large_barrel_capacity_l1575_157565

def total_oil : ℕ := 95
def small_barrel_capacity : ℕ := 5
def small_barrels_used : ℕ := 1

def remaining_oil : ℕ := total_oil - (small_barrel_capacity * small_barrels_used)

def is_valid_large_barrel_capacity (capacity : ℕ) : Prop :=
  capacity > small_barrel_capacity ∧ 
  remaining_oil % capacity = 0 ∧
  capacity ≤ remaining_oil

theorem largest_factor_is_large_barrel_capacity : 
  ∃ (large_barrel_capacity : ℕ), 
    is_valid_large_barrel_capacity large_barrel_capacity ∧
    ∀ (x : ℕ), is_valid_large_barrel_capacity x → x ≤ large_barrel_capacity := by
  sorry

end NUMINAMATH_CALUDE_largest_factor_is_large_barrel_capacity_l1575_157565


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_equality_l1575_157549

theorem sqrt_sum_squares_equality (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b - c ↔ a * b = c * (a + b) ∧ a + b - c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_equality_l1575_157549


namespace NUMINAMATH_CALUDE_museum_trip_l1575_157539

theorem museum_trip (people_first : ℕ) : 
  (people_first + 
   2 * people_first + 
   (2 * people_first - 6) + 
   (people_first + 9) = 75) → 
  people_first = 12 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_l1575_157539


namespace NUMINAMATH_CALUDE_inequality_proof_l1575_157536

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) :
  a^2 + b^2 + 1/a^2 + b/a ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1575_157536


namespace NUMINAMATH_CALUDE_office_meeting_reduction_l1575_157502

theorem office_meeting_reduction (total_people : ℕ) (women_in_meeting : ℕ) : 
  total_people = 60 → 
  women_in_meeting = 6 → 
  (women_in_meeting : ℚ) / (total_people / 2 : ℚ) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_office_meeting_reduction_l1575_157502


namespace NUMINAMATH_CALUDE_sugar_distribution_l1575_157512

/-- The number of sugar boxes -/
def num_boxes : ℕ := 21

/-- The weight of sugar per box in kilograms -/
def sugar_per_box : ℚ := 6

/-- The amount of sugar distributed to each neighbor in kilograms -/
def sugar_per_neighbor : ℚ := 32 / 41

/-- The maximum number of neighbors who can receive sugar -/
def max_neighbors : ℕ := 161

theorem sugar_distribution :
  ⌊(num_boxes * sugar_per_box) / sugar_per_neighbor⌋ = max_neighbors := by
  sorry

end NUMINAMATH_CALUDE_sugar_distribution_l1575_157512


namespace NUMINAMATH_CALUDE_anderson_trousers_count_l1575_157589

theorem anderson_trousers_count :
  let total_clothing : ℕ := 934
  let shirts : ℕ := 589
  let trousers : ℕ := total_clothing - shirts
  trousers = 345 := by sorry

end NUMINAMATH_CALUDE_anderson_trousers_count_l1575_157589


namespace NUMINAMATH_CALUDE_least_subtraction_l1575_157558

theorem least_subtraction (x : ℕ) : x = 22 ↔ 
  x ≠ 0 ∧
  (∀ y : ℕ, y < x → ¬(1398 - y) % 7 = 5 ∨ ¬(1398 - y) % 9 = 5 ∨ ¬(1398 - y) % 11 = 5) ∧
  (1398 - x) % 7 = 5 ∧
  (1398 - x) % 9 = 5 ∧
  (1398 - x) % 11 = 5 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_l1575_157558


namespace NUMINAMATH_CALUDE_grid_figure_boundary_theorem_l1575_157513

/-- A grid figure is a shape cut from grid paper along grid lines without holes. -/
structure GridFigure where
  -- Add necessary fields here
  no_holes : Bool

/-- Represents a set of straight cuts along grid lines. -/
structure GridCuts where
  total_length : ℕ
  divides_into_cells : Bool

/-- Checks if a grid figure has a straight boundary segment of at least given length. -/
def has_straight_boundary_segment (figure : GridFigure) (length : ℕ) : Prop :=
  sorry

theorem grid_figure_boundary_theorem (figure : GridFigure) (cuts : GridCuts) :
  figure.no_holes ∧ 
  cuts.total_length = 2017 ∧
  cuts.divides_into_cells →
  has_straight_boundary_segment figure 2 :=
by sorry

end NUMINAMATH_CALUDE_grid_figure_boundary_theorem_l1575_157513


namespace NUMINAMATH_CALUDE_operation_results_in_zero_in_quotient_l1575_157548

-- Define the arithmetic operation
def operation : ℕ → ℕ → ℕ := (·+·)

-- Define the property of having a zero in the middle of the quotient
def has_zero_in_middle_of_quotient (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * 10 + 0 * 1 + b ∧ 0 < b ∧ b < 10

-- Theorem statement
theorem operation_results_in_zero_in_quotient :
  has_zero_in_middle_of_quotient (operation 6 4 / 3) :=
sorry

end NUMINAMATH_CALUDE_operation_results_in_zero_in_quotient_l1575_157548


namespace NUMINAMATH_CALUDE_simplify_expression_l1575_157550

theorem simplify_expression (x : ℝ) : 2*x - 3*(2-x) + 4*(3+x) - 5*(2+3*x) = -6*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1575_157550


namespace NUMINAMATH_CALUDE_mean_median_difference_l1575_157506

def class_size : ℕ := 40

def score_distribution : List (ℕ × ℚ) := [
  (60, 15/100),
  (75, 35/100),
  (82, 10/100),
  (88, 20/100),
  (92, 20/100)
]

def mean_score : ℚ :=
  (score_distribution.map (λ (score, percentage) => score * (percentage * class_size))).sum / class_size

def median_score : ℕ := 75

theorem mean_median_difference : 
  ⌊mean_score - median_score⌋ = 4 :=
sorry

end NUMINAMATH_CALUDE_mean_median_difference_l1575_157506


namespace NUMINAMATH_CALUDE_set_empty_properties_l1575_157525

theorem set_empty_properties (A : Set α) :
  let p := A ∩ ∅ = ∅
  let q := A ∪ ∅ = A
  (p ∧ q) ∧ (¬p ∨ q) := by sorry

end NUMINAMATH_CALUDE_set_empty_properties_l1575_157525


namespace NUMINAMATH_CALUDE_expression_evaluation_l1575_157531

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^(y + 1) + 6 * y^(x + 1) = 2751 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1575_157531


namespace NUMINAMATH_CALUDE_geometric_mean_of_one_and_nine_l1575_157520

theorem geometric_mean_of_one_and_nine :
  ∃ (c : ℝ), c^2 = 1 * 9 ∧ (c = 3 ∨ c = -3) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_one_and_nine_l1575_157520


namespace NUMINAMATH_CALUDE_max_students_in_class_l1575_157504

theorem max_students_in_class (x : ℕ) : 
  x > 0 ∧ 
  2 ∣ x ∧ 
  4 ∣ x ∧ 
  7 ∣ x ∧ 
  x - (x / 2 + x / 4 + x / 7) < 6 →
  x ≤ 28 :=
by sorry

end NUMINAMATH_CALUDE_max_students_in_class_l1575_157504


namespace NUMINAMATH_CALUDE_power_sum_equality_l1575_157557

theorem power_sum_equality : (-2)^2004 + 3 * (-2)^2003 = -2^2003 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1575_157557


namespace NUMINAMATH_CALUDE_factor_polynomial_l1575_157532

theorem factor_polynomial (x : ℝ) : 80 * x^5 - 250 * x^9 = -10 * x^5 * (25 * x^4 - 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1575_157532


namespace NUMINAMATH_CALUDE_unit_digit_of_seven_power_ten_l1575_157586

theorem unit_digit_of_seven_power_ten (n : ℕ) : n = 10 → (7^n) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_seven_power_ten_l1575_157586


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1575_157596

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + (2*k - 1)*x₁ - k - 1 = 0) → 
  (x₂^2 + (2*k - 1)*x₂ - k - 1 = 0) → 
  (x₁ + x₂ - 4*x₁*x₂ = 2) → 
  (k = -3/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1575_157596


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1575_157503

theorem quadratic_equation_roots (x : ℝ) : 
  (∃! r : ℝ, x^2 - 2*x + 1 = 0) ↔ (x^2 - 2*x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1575_157503


namespace NUMINAMATH_CALUDE_tomatoes_left_l1575_157544

theorem tomatoes_left (initial : ℕ) (picked_yesterday : ℕ) (picked_today : ℕ) 
  (h1 : initial = 171)
  (h2 : picked_yesterday = 134)
  (h3 : picked_today = 30) : 
  initial - picked_yesterday - picked_today = 7 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_l1575_157544


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1575_157566

/-- Given a line segment from (0, 2) to (3, y) with length 10 and y > 0, prove y = 2 + √91 -/
theorem line_segment_endpoint (y : ℝ) (h1 : y > 0) : 
  (((3 - 0)^2 + (y - 2)^2 : ℝ) = 10^2) → y = 2 + Real.sqrt 91 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1575_157566


namespace NUMINAMATH_CALUDE_video_game_spending_is_correct_l1575_157580

def total_allowance : ℚ := 50

def movie_fraction : ℚ := 1/4
def burger_fraction : ℚ := 1/5
def ice_cream_fraction : ℚ := 1/10
def music_fraction : ℚ := 2/5

def video_game_spending : ℚ := total_allowance - (movie_fraction * total_allowance + burger_fraction * total_allowance + ice_cream_fraction * total_allowance + music_fraction * total_allowance)

theorem video_game_spending_is_correct : video_game_spending = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_video_game_spending_is_correct_l1575_157580


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1575_157554

theorem rationalize_denominator : 
  (Real.sqrt 18 - Real.sqrt 2 + Real.sqrt 27) / (Real.sqrt 3 + Real.sqrt 2) = 5 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1575_157554


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l1575_157577

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ k < 3, k > 0 → (Nat.gcd k 63 = 1 ∨ Nat.gcd k 66 = 1)) ∧ 
  Nat.gcd 3 63 > 1 ∧ 
  Nat.gcd 3 66 > 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l1575_157577


namespace NUMINAMATH_CALUDE_perfect_square_units_mod_16_l1575_157578

theorem perfect_square_units_mod_16 : 
  ∃ (S : Finset ℕ), (∀ n : ℕ, ∃ m : ℕ, n ^ 2 % 16 ∈ S) ∧ S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_units_mod_16_l1575_157578


namespace NUMINAMATH_CALUDE_roots_of_equation_l1575_157537

theorem roots_of_equation : 
  let f (x : ℝ) := 21 / (x^2 - 9) - 3 / (x - 3) - 1
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -7 := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1575_157537


namespace NUMINAMATH_CALUDE_divisible_by_72_sum_of_digits_l1575_157587

theorem divisible_by_72_sum_of_digits (A B : ℕ) : 
  A < 10 → B < 10 → 
  (100000 * A + 44610 + B) % 72 = 0 → 
  A + B = 12 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_72_sum_of_digits_l1575_157587


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1575_157559

theorem triangle_angle_measure (b c S_ABC : ℝ) (h1 : b = 8) (h2 : c = 8 * Real.sqrt 3) 
  (h3 : S_ABC = 16 * Real.sqrt 3) :
  ∃ A : ℝ, (A = π / 6 ∨ A = 5 * π / 6) ∧ 
    S_ABC = (1/2) * b * c * Real.sin A ∧ 0 < A ∧ A < π :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1575_157559


namespace NUMINAMATH_CALUDE_one_seventh_minus_one_eleventh_equals_100_l1575_157591

theorem one_seventh_minus_one_eleventh_equals_100 :
  let N : ℚ := 1925
  (N / 7) - (N / 11) = 100 := by
  sorry

end NUMINAMATH_CALUDE_one_seventh_minus_one_eleventh_equals_100_l1575_157591


namespace NUMINAMATH_CALUDE_last_to_first_points_l1575_157546

/-- Represents a chess tournament with initial and final states -/
structure ChessTournament where
  initial_players : ℕ
  disqualified_players : ℕ
  points_per_win : ℚ
  points_per_draw : ℚ
  points_per_loss : ℚ

/-- Calculates the total number of games in a round-robin tournament -/
def total_games (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating that a player who goes from last to first must have 4 points after disqualification -/
theorem last_to_first_points (t : ChessTournament) 
  (h1 : t.initial_players = 10)
  (h2 : t.disqualified_players = 2)
  (h3 : t.points_per_win = 1)
  (h4 : t.points_per_draw = 1/2)
  (h5 : t.points_per_loss = 0) :
  ∃ (initial_points final_points : ℚ),
    initial_points < (total_games t.initial_players : ℚ) / t.initial_players ∧
    final_points > (total_games (t.initial_players - t.disqualified_players) : ℚ) / (t.initial_players - t.disqualified_players) ∧
    final_points = 4 :=
  sorry

end NUMINAMATH_CALUDE_last_to_first_points_l1575_157546


namespace NUMINAMATH_CALUDE_identity_function_property_l1575_157570

theorem identity_function_property (f : ℕ → ℕ) : 
  (∀ m n : ℕ, (f m + f n) ∣ (m + n)) → 
  (∀ m : ℕ, f m = m) := by
  sorry

end NUMINAMATH_CALUDE_identity_function_property_l1575_157570


namespace NUMINAMATH_CALUDE_no_x_satisfying_conditions_l1575_157551

theorem no_x_satisfying_conditions : ¬∃ x : ℝ, 
  250 ≤ x ∧ x ≤ 350 ∧ 
  ⌊Real.sqrt (x - 50)⌋ = 14 ∧ 
  ⌊Real.sqrt (50 * x)⌋ = 256 := by
  sorry

#check no_x_satisfying_conditions

end NUMINAMATH_CALUDE_no_x_satisfying_conditions_l1575_157551


namespace NUMINAMATH_CALUDE_hexagon_extended_side_length_l1575_157594

/-- Regular hexagon with side length 3 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Point Y on the extension of side CD such that CY = 4CD -/
def extend_side (h : RegularHexagon) (CD : ℝ) (Y : ℝ) : Prop :=
  CD = h.side_length ∧ Y = 4 * CD

/-- The length of segment FY in the described configuration -/
def segment_FY_length (h : RegularHexagon) (Y : ℝ) : ℝ := sorry

/-- Theorem stating the length of FY is 5.5√3 -/
theorem hexagon_extended_side_length (h : RegularHexagon) (CD Y : ℝ) 
  (h_extend : extend_side h CD Y) : 
  segment_FY_length h Y = 5.5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_extended_side_length_l1575_157594


namespace NUMINAMATH_CALUDE_jake_arrives_later_l1575_157505

/-- Represents the building with elevators and stairs --/
structure Building where
  floors : ℕ
  steps_per_floor : ℕ
  elevator_b_time : ℕ

/-- Represents a person descending the building --/
structure Person where
  steps_per_second : ℕ

def time_to_descend (b : Building) (p : Person) : ℕ :=
  let total_steps := b.steps_per_floor * (b.floors - 1)
  (total_steps + p.steps_per_second - 1) / p.steps_per_second

theorem jake_arrives_later (b : Building) (jake : Person) :
  b.floors = 12 →
  b.steps_per_floor = 25 →
  b.elevator_b_time = 90 →
  jake.steps_per_second = 3 →
  time_to_descend b jake - b.elevator_b_time = 2 := by
  sorry

#eval time_to_descend { floors := 12, steps_per_floor := 25, elevator_b_time := 90 } { steps_per_second := 3 }

end NUMINAMATH_CALUDE_jake_arrives_later_l1575_157505


namespace NUMINAMATH_CALUDE_equal_expressions_count_l1575_157563

theorem equal_expressions_count (x : ℝ) (h : x > 0) : 
  (∃! (count : ℕ), count = 2 ∧ 
    count = (Bool.toNat (2 * x^x = x^x + x^x) + 
             Bool.toNat (x^(x+1) = x^x + x^x) + 
             Bool.toNat ((x+1)^x = x^x + x^x) + 
             Bool.toNat (x^(2*(x+1)) = x^x + x^x))) :=
by sorry

end NUMINAMATH_CALUDE_equal_expressions_count_l1575_157563


namespace NUMINAMATH_CALUDE_complex_vector_difference_l1575_157576

theorem complex_vector_difference (z : ℂ) (h : z = 1 - I) :
  z^2 - z = -1 - I := by sorry

end NUMINAMATH_CALUDE_complex_vector_difference_l1575_157576


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1575_157595

theorem diophantine_equation_solution :
  ∀ a b c : ℕ+,
  a + b = c - 1 →
  a^3 + b^3 = c^2 - 1 →
  ((a = 2 ∧ b = 3 ∧ c = 6) ∨ (a = 3 ∧ b = 2 ∧ c = 6)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1575_157595


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l1575_157510

theorem cubic_roots_relation (p q r : ℝ) (u v w : ℝ) : 
  (p^3 + 4*p^2 + 5*p - 13 = 0) →
  (q^3 + 4*q^2 + 5*q - 13 = 0) →
  (r^3 + 4*r^2 + 5*r - 13 = 0) →
  ((p+q)^3 + u*(p+q)^2 + v*(p+q) + w = 0) →
  ((q+r)^3 + u*(q+r)^2 + v*(q+r) + w = 0) →
  ((r+p)^3 + u*(r+p)^2 + v*(r+p) + w = 0) →
  w = 33 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l1575_157510


namespace NUMINAMATH_CALUDE_left_of_kolya_l1575_157599

/-- The number of people in a line-up -/
def total_people : ℕ := 29

/-- The number of people to the right of Kolya -/
def right_of_kolya : ℕ := 12

/-- The number of people to the left of Sasha -/
def left_of_sasha : ℕ := 20

/-- The number of people to the right of Sasha -/
def right_of_sasha : ℕ := 8

/-- Theorem: The number of people to the left of Kolya is 16 -/
theorem left_of_kolya : total_people - right_of_kolya - 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_left_of_kolya_l1575_157599


namespace NUMINAMATH_CALUDE_fraction_of_juniors_l1575_157593

theorem fraction_of_juniors (J S : ℕ) : 
  J > 0 → -- There is at least one junior
  S > 0 → -- There is at least one senior
  (J : ℚ) / 2 = (S : ℚ) * 2 / 3 → -- Half the number of juniors equals two-thirds the number of seniors
  (J : ℚ) / (J + S) = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_juniors_l1575_157593


namespace NUMINAMATH_CALUDE_flea_treatment_effectiveness_l1575_157590

theorem flea_treatment_effectiveness (F : ℕ) : 
  (F : ℝ) * 0.4 * 0.55 * 0.7 * 0.8 = 20 → F - 20 = 142 := by
  sorry

end NUMINAMATH_CALUDE_flea_treatment_effectiveness_l1575_157590


namespace NUMINAMATH_CALUDE_sine_sum_simplification_l1575_157534

theorem sine_sum_simplification (x y : ℝ) :
  Real.sin (x - y) * Real.cos y + Real.cos (x - y) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_simplification_l1575_157534


namespace NUMINAMATH_CALUDE_bianca_birthday_money_l1575_157584

/-- The amount of money Bianca received for her birthday -/
def birthday_money (num_friends : ℕ) (dollars_per_friend : ℕ) : ℕ :=
  num_friends * dollars_per_friend

/-- Theorem stating that Bianca received 30 dollars for her birthday -/
theorem bianca_birthday_money :
  birthday_money 5 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_bianca_birthday_money_l1575_157584


namespace NUMINAMATH_CALUDE_lucas_cycling_speed_l1575_157583

theorem lucas_cycling_speed 
  (philippe_speed : ℝ) 
  (marta_ratio : ℝ) 
  (lucas_ratio : ℝ) 
  (h1 : philippe_speed = 10)
  (h2 : marta_ratio = 3/4)
  (h3 : lucas_ratio = 4/3) : 
  lucas_ratio * (marta_ratio * philippe_speed) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_lucas_cycling_speed_l1575_157583


namespace NUMINAMATH_CALUDE_system_solution_exists_l1575_157501

theorem system_solution_exists (a b : ℤ) (h1 : 5 * a ≥ 7 * b) (h2 : 7 * b ≥ 0) :
  ∃ (x y z u : ℕ), x + 2 * y + 3 * z + 7 * u = a ∧ y + 2 * z + 5 * u = b := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_l1575_157501


namespace NUMINAMATH_CALUDE_triangle_side_range_l1575_157569

theorem triangle_side_range (A B C : Real) (AB AC BC : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Given equation
  (Real.sqrt 3 * Real.sin B - Real.cos B) * (Real.sqrt 3 * Real.sin C - Real.cos C) = 4 * Real.cos B * Real.cos C →
  -- Sum of two sides
  AB + AC = 4 →
  -- Triangle inequality
  AB > 0 ∧ AC > 0 ∧ BC > 0 →
  -- BC satisfies the triangle inequality
  BC < AB + AC ∧ AB < BC + AC ∧ AC < AB + BC →
  -- Conclusion: Range of BC
  2 ≤ BC ∧ BC < 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1575_157569


namespace NUMINAMATH_CALUDE_f_increasing_min_value_sum_tangent_line_l1575_157597

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 + 1/2

-- Statement 1: f(x) is monotonically increasing on [5π/6, π]
theorem f_increasing : ∀ x y, 5*Real.pi/6 ≤ x ∧ x < y ∧ y ≤ Real.pi → f x < f y := by sorry

-- Statement 2: The minimum value of f(x) + f(x + π/4) is -√2
theorem min_value_sum : ∃ m : ℝ, (∀ x, m ≤ f x + f (x + Real.pi/4)) ∧ m = -Real.sqrt 2 := by sorry

-- Statement 3: The line y = √3x - 1/2 is a tangent line to y = f(x)
theorem tangent_line : ∃ x₀ : ℝ, f x₀ = Real.sqrt 3 * x₀ - 1/2 ∧ 
  (∀ x, f x ≤ Real.sqrt 3 * x - 1/2) := by sorry

end NUMINAMATH_CALUDE_f_increasing_min_value_sum_tangent_line_l1575_157597


namespace NUMINAMATH_CALUDE_marble_density_l1575_157552

/-- Density of a rectangular prism made of marble -/
theorem marble_density (height : ℝ) (base_side : ℝ) (weight : ℝ) :
  height = 8 →
  base_side = 2 →
  weight = 86400 →
  weight / (base_side * base_side * height) = 2700 := by
  sorry

end NUMINAMATH_CALUDE_marble_density_l1575_157552


namespace NUMINAMATH_CALUDE_tom_speed_l1575_157598

theorem tom_speed (karen_speed : ℝ) (karen_delay : ℝ) (win_distance : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 ∧ 
  karen_delay = 4 / 60 ∧ 
  win_distance = 4 ∧ 
  tom_distance = 24 → 
  ∃ (tom_speed : ℝ), tom_speed = 60 :=
by sorry

end NUMINAMATH_CALUDE_tom_speed_l1575_157598


namespace NUMINAMATH_CALUDE_xaxaxa_divisible_by_seven_l1575_157524

theorem xaxaxa_divisible_by_seven (X A : ℕ) 
  (h_digits : X < 10 ∧ A < 10) 
  (h_distinct : X ≠ A) : 
  ∃ k : ℕ, 101010 * X + 10101 * A = 7 * k := by
sorry

end NUMINAMATH_CALUDE_xaxaxa_divisible_by_seven_l1575_157524


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_power_of_sum_l1575_157555

theorem sum_of_powers_equals_power_of_sum : 5^5 + 5^5 + 5^5 + 5^5 = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_power_of_sum_l1575_157555


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_product_l1575_157575

theorem quadratic_equation_sum_product (m p : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + p = 0 ∧ 3 * y^2 - m * y + p = 0 ∧ x + y = 9 ∧ x * y = 14) →
  m + p = 69 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_product_l1575_157575


namespace NUMINAMATH_CALUDE_snackies_leftover_l1575_157585

theorem snackies_leftover (m : ℕ) (h : m % 8 = 5) : (4 * m) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_snackies_leftover_l1575_157585


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l1575_157508

/-- Given a cubic function f(x) = (1/3)x³ - x + m with a maximum value of 1,
    prove that its minimum value is -1/3 -/
theorem cubic_function_extrema (f : ℝ → ℝ) (m : ℝ) 
    (h1 : ∀ x, f x = (1/3) * x^3 - x + m) 
    (h2 : ∃ x₀, ∀ x, f x ≤ f x₀ ∧ f x₀ = 1) : 
    ∃ x₁, ∀ x, f x ≥ f x₁ ∧ f x₁ = -(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l1575_157508


namespace NUMINAMATH_CALUDE_problem_solution_l1575_157522

theorem problem_solution (x y : ℕ) (h1 : x > y) (h2 : x + x * y = 391) : x + y = 39 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1575_157522


namespace NUMINAMATH_CALUDE_triangle_exradius_theorem_l1575_157572

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  r_a : ℝ
  r_b : ℝ
  r_c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_R : 0 < R
  pos_r_a : 0 < r_a
  pos_r_b : 0 < r_b
  pos_r_c : 0 < r_c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

theorem triangle_exradius_theorem (t : Triangle) (h : 2 * t.R ≤ t.r_a) :
  t.a > t.b ∧ t.a > t.c ∧ 2 * t.R > t.r_b ∧ 2 * t.R > t.r_c := by
  sorry

end NUMINAMATH_CALUDE_triangle_exradius_theorem_l1575_157572


namespace NUMINAMATH_CALUDE_series_sum_l1575_157560

/-- The sum of the infinite series 2 + ∑(k=1 to ∞) ((k+2)*(1/1000)^(k-1)) is equal to 3000000/998001 -/
theorem series_sum : 
  let S := 2 + ∑' k, (k + 2) * (1 / 1000) ^ (k - 1)
  S = 3000000 / 998001 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l1575_157560


namespace NUMINAMATH_CALUDE_ratio_problem_l1575_157519

theorem ratio_problem (p q : ℚ) (h : 25 / 7 + (2 * q - p) / (2 * q + p) = 4) : p / q = -1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1575_157519


namespace NUMINAMATH_CALUDE_not_always_same_digit_sum_l1575_157582

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem not_always_same_digit_sum :
  ∃ (N M : ℕ), 
    (sum_of_digits (N + M) = sum_of_digits N) ∧
    (∃ (k : ℕ), k > 1 ∧ sum_of_digits (N + k * M) ≠ sum_of_digits N) :=
sorry

end NUMINAMATH_CALUDE_not_always_same_digit_sum_l1575_157582


namespace NUMINAMATH_CALUDE_difference_of_expressions_l1575_157521

theorem difference_of_expressions : 
  ((0.85 * 250)^2 / 2.3) - ((3/5 * 175) / 2.3) = 19587.5 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_expressions_l1575_157521


namespace NUMINAMATH_CALUDE_prime_with_integer_roots_range_l1575_157543

theorem prime_with_integer_roots_range (p : ℕ) : 
  Nat.Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 500*p = 0 ∧ y^2 + p*y - 500*p = 0) → 
  1 < p ∧ p ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_prime_with_integer_roots_range_l1575_157543


namespace NUMINAMATH_CALUDE_fifteenth_term_ratio_l1575_157511

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℚ  -- first term
  d : ℚ  -- common difference

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a + (n - 1) * seq.d) / 2

theorem fifteenth_term_ratio
  (seq1 seq2 : ArithmeticSequence)
  (h : ∀ n : ℕ, (sum_n seq1 n) / (sum_n seq2 n) = (9 * n + 3) / (5 * n + 35)) :
  (seq1.a + 14 * seq1.d) / (seq2.a + 14 * seq2.d) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_ratio_l1575_157511


namespace NUMINAMATH_CALUDE_investmentPlansCount_l1575_157523

/-- The number of ways to distribute 3 distinct projects among 6 locations,
    with no more than 2 projects per location. -/
def investmentPlans : ℕ :=
  Nat.descFactorial 6 3 + (Nat.choose 3 2 * Nat.descFactorial 6 2)

/-- Theorem stating that the number of distinct investment plans is 210. -/
theorem investmentPlansCount : investmentPlans = 210 := by
  sorry

end NUMINAMATH_CALUDE_investmentPlansCount_l1575_157523


namespace NUMINAMATH_CALUDE_amy_work_hours_l1575_157533

theorem amy_work_hours (hourly_wage : ℝ) (tips : ℝ) (total_earnings : ℝ) : 
  hourly_wage = 2 → tips = 9 → total_earnings = 23 → 
  ∃ h : ℝ, h * hourly_wage + tips = total_earnings ∧ h = 7 := by
sorry

end NUMINAMATH_CALUDE_amy_work_hours_l1575_157533


namespace NUMINAMATH_CALUDE_total_elephants_count_l1575_157527

def elephants_we_preserve : ℕ := 70

def elephants_gestures_for_good : ℕ := 3 * elephants_we_preserve

def total_elephants : ℕ := elephants_we_preserve + elephants_gestures_for_good

theorem total_elephants_count : total_elephants = 280 := by
  sorry

end NUMINAMATH_CALUDE_total_elephants_count_l1575_157527


namespace NUMINAMATH_CALUDE_fish_pond_problem_l1575_157518

/-- Calculates the number of fish in the second catch given the total number of fish in the pond,
    the number of tagged fish, and the number of tagged fish caught in the second catch. -/
def second_catch_size (total_fish : ℕ) (tagged_fish : ℕ) (tagged_caught : ℕ) : ℕ :=
  tagged_fish * total_fish / tagged_caught

/-- Theorem stating that given a pond with approximately 1000 fish, where 40 fish were initially tagged
    and released, and 2 tagged fish were found in a subsequent catch, the number of fish in the
    subsequent catch is 50. -/
theorem fish_pond_problem (total_fish : ℕ) (tagged_fish : ℕ) (tagged_caught : ℕ) :
  total_fish = 1000 → tagged_fish = 40 → tagged_caught = 2 →
  second_catch_size total_fish tagged_fish tagged_caught = 50 := by
  sorry

#eval second_catch_size 1000 40 2

end NUMINAMATH_CALUDE_fish_pond_problem_l1575_157518


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l1575_157500

theorem gcd_digits_bound (a b : ℕ) : 
  (1000000 ≤ a ∧ a < 10000000) →
  (1000000 ≤ b ∧ b < 10000000) →
  (100000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000000) →
  Nat.gcd a b < 1000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l1575_157500


namespace NUMINAMATH_CALUDE_triangle_properties_l1575_157514

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (a + b) / Real.sin (A + B) = (a - c) / (Real.sin A - Real.sin B) →
  b = 3 →
  Real.cos A = Real.sqrt 6 / 3 →
  B = π / 3 ∧
  (1 / 2) * a * b * Real.sin C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1575_157514


namespace NUMINAMATH_CALUDE_fraction_comparison_l1575_157530

theorem fraction_comparison : (2 : ℚ) / 3 - 66666666 / 100000000 = 2 / (3 * 100000000) := by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1575_157530


namespace NUMINAMATH_CALUDE_johns_beef_purchase_l1575_157529

/-- Given that John uses all but 1 pound of beef in soup, uses twice as many pounds of vegetables 
    as beef, and uses 6 pounds of vegetables, prove that John bought 4 pounds of beef. -/
theorem johns_beef_purchase (beef_used : ℝ) (vegetables_used : ℝ) (beef_leftover : ℝ) : 
  beef_leftover = 1 →
  vegetables_used = 2 * beef_used →
  vegetables_used = 6 →
  beef_used + beef_leftover = 4 := by
  sorry

end NUMINAMATH_CALUDE_johns_beef_purchase_l1575_157529


namespace NUMINAMATH_CALUDE_ten_lines_intersections_l1575_157574

/-- The number of intersections formed by n straight lines where no two lines are parallel
    and no three lines intersect at a single point. -/
def intersections (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else (n - 1) * (n - 2) / 2

/-- Theorem stating that 10 straight lines under the given conditions form 45 intersections -/
theorem ten_lines_intersections :
  intersections 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_lines_intersections_l1575_157574


namespace NUMINAMATH_CALUDE_area_of_S3_l1575_157515

/-- Given a square S1 with area 16, S2 is constructed by connecting the midpoints of S1's sides,
    and S3 is constructed by connecting the midpoints of S2's sides. -/
def nested_squares (S1 S2 S3 : Real) : Prop :=
  S1 = 16 ∧ S2 = S1 / 2 ∧ S3 = S2 / 2

theorem area_of_S3 (S1 S2 S3 : Real) (h : nested_squares S1 S2 S3) : S3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_S3_l1575_157515


namespace NUMINAMATH_CALUDE_tan_pi_plus_theta_l1575_157538

theorem tan_pi_plus_theta (θ : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 3/5 ∧ y = 4/5 ∧ 
   x = Real.cos θ ∧ y = Real.sin θ) →
  Real.tan (π + θ) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_pi_plus_theta_l1575_157538


namespace NUMINAMATH_CALUDE_hunter_saw_twelve_ants_l1575_157535

/-- The number of ants Hunter saw in the playground -/
def ants_seen (spiders ladybugs_initial ladybugs_left total_insects : ℕ) : ℕ :=
  total_insects - spiders - ladybugs_left

/-- Theorem stating that Hunter saw 12 ants given the problem conditions -/
theorem hunter_saw_twelve_ants :
  let spiders : ℕ := 3
  let ladybugs_initial : ℕ := 8
  let ladybugs_left : ℕ := ladybugs_initial - 2
  let total_insects : ℕ := 21
  ants_seen spiders ladybugs_initial ladybugs_left total_insects = 12 := by
  sorry

end NUMINAMATH_CALUDE_hunter_saw_twelve_ants_l1575_157535
