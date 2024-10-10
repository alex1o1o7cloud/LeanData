import Mathlib

namespace valid_arrangement_has_four_rows_of_seven_l2294_229483

/-- Represents a seating arrangement -/
structure SeatingArrangement where
  rows_of_seven : ℕ
  rows_of_six : ℕ

/-- Checks if a seating arrangement is valid -/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.rows_of_seven * 7 + s.rows_of_six * 6 = 52

/-- Theorem stating that the valid arrangement has 4 rows of 7 people -/
theorem valid_arrangement_has_four_rows_of_seven :
  ∃ (s : SeatingArrangement), is_valid_arrangement s ∧ s.rows_of_seven = 4 := by
  sorry

end valid_arrangement_has_four_rows_of_seven_l2294_229483


namespace root_exists_in_interval_l2294_229400

theorem root_exists_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ Real.exp x = 1/x := by
  sorry

end root_exists_in_interval_l2294_229400


namespace scientific_notation_equivalence_l2294_229485

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.000000301 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.01 ∧ n = -7 := by
  sorry

end scientific_notation_equivalence_l2294_229485


namespace vacant_seats_l2294_229441

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 60/100) : 
  (1 - filled_percentage) * total_seats = 240 := by
sorry

end vacant_seats_l2294_229441


namespace worker_count_l2294_229416

theorem worker_count (total : ℕ) (extra_total : ℕ) (extra_per_worker : ℕ) : 
  (total = 300000) → 
  (extra_total = 375000) → 
  (extra_per_worker = 50) → 
  (∃ (w : ℕ), w * (extra_total / w - total / w) = extra_per_worker ∧ w = 1500) :=
by sorry

end worker_count_l2294_229416


namespace product_of_roots_l2294_229431

theorem product_of_roots (x : ℝ) : (x - 1) * (x + 4) = 22 → ∃ y : ℝ, (y - 1) * (y + 4) = 22 ∧ x * y = -26 := by
  sorry

end product_of_roots_l2294_229431


namespace new_girl_weight_l2294_229465

/-- Given a group of 10 girls, if replacing one girl weighing 50 kg with a new girl
    increases the average weight by 5 kg, then the new girl weighs 100 kg. -/
theorem new_girl_weight (initial_weight : ℝ) (new_weight : ℝ) :
  (initial_weight - 50 + new_weight) / 10 = initial_weight / 10 + 5 →
  new_weight = 100 := by
sorry

end new_girl_weight_l2294_229465


namespace A_intersect_B_equals_nonnegative_reals_l2294_229406

open Set

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = 2 * x}
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Define the intersection set
def intersection_set : Set ℝ := {y | y ≥ 0}

-- Theorem statement
theorem A_intersect_B_equals_nonnegative_reals : A ∩ B = intersection_set := by
  sorry

end A_intersect_B_equals_nonnegative_reals_l2294_229406


namespace ball_count_theorem_l2294_229455

theorem ball_count_theorem (B W : ℕ) (h1 : W = 3 * B) 
  (h2 : 5 * B + W = 2 * (B + W)) : 
  B + 5 * W = 4 * (B + W) := by
sorry

end ball_count_theorem_l2294_229455


namespace g_neg_two_l2294_229489

def g (x : ℝ) : ℝ := x^3 - 2*x + 1

theorem g_neg_two : g (-2) = -3 := by
  sorry

end g_neg_two_l2294_229489


namespace bd_squared_equals_36_l2294_229437

theorem bd_squared_equals_36 
  (a b c d : ℤ) 
  (h1 : a - b - c + d = 18) 
  (h2 : a + b - c - d = 6) : 
  (b - d)^2 = 36 := by
sorry

end bd_squared_equals_36_l2294_229437


namespace vincent_book_expenditure_l2294_229420

def animal_books : ℕ := 10
def space_books : ℕ := 1
def train_books : ℕ := 3
def book_cost : ℕ := 16

theorem vincent_book_expenditure :
  (animal_books + space_books + train_books) * book_cost = 224 := by
  sorry

end vincent_book_expenditure_l2294_229420


namespace smallest_c_for_quadratic_inequality_l2294_229486

theorem smallest_c_for_quadratic_inequality : 
  ∃ c : ℝ, c = 2 ∧ (∀ x : ℝ, -x^2 + 9*x - 14 ≥ 0 → x ≥ c) := by
  sorry

end smallest_c_for_quadratic_inequality_l2294_229486


namespace baseball_players_l2294_229478

/-- Given a club with the following properties:
  * There are 310 people in total
  * 138 people play tennis
  * 94 people play both tennis and baseball
  * 11 people do not play any sport
  Prove that 255 people play baseball -/
theorem baseball_players (total : ℕ) (tennis : ℕ) (both : ℕ) (none : ℕ) 
  (h1 : total = 310)
  (h2 : tennis = 138)
  (h3 : both = 94)
  (h4 : none = 11) :
  total - (tennis - both) - none = 255 := by
  sorry

#eval 310 - (138 - 94) - 11

end baseball_players_l2294_229478


namespace calculate_expression_l2294_229448

theorem calculate_expression : (-2 + 3) * 2 + (-2)^3 / 4 = 0 := by
  sorry

end calculate_expression_l2294_229448


namespace triangle_area_l2294_229458

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 9√3/14 when a = 3, b = 2c, and A = 2π/3 -/
theorem triangle_area (a b c : ℝ) (A : ℝ) :
  a = 3 →
  b = 2 * c →
  A = 2 * Real.pi / 3 →
  (1 / 2 : ℝ) * b * c * Real.sin A = 9 * Real.sqrt 3 / 14 := by
  sorry

end triangle_area_l2294_229458


namespace inequality_system_solution_l2294_229425

theorem inequality_system_solution :
  ∃ (x y : ℝ), 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧
    (2 * x - 4 * y ≤ -3) ∧
    (x = -1/3) ∧ (y = 2/3) := by
  sorry

end inequality_system_solution_l2294_229425


namespace power_of_power_eq_expanded_power_l2294_229417

theorem power_of_power_eq_expanded_power (x : ℝ) : (2 * x^2)^3 = 8 * x^6 := by
  sorry

end power_of_power_eq_expanded_power_l2294_229417


namespace expression_evaluation_l2294_229404

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -2
  (3*x + 2*y) * (3*x - 2*y) - 5*x*(x - y) - (2*x - y)^2 = -14 := by
  sorry

end expression_evaluation_l2294_229404


namespace equation_positive_root_l2294_229494

theorem equation_positive_root (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (x - a) / (x - 1) - 3 / x = 1) → a = 1 := by
sorry

end equation_positive_root_l2294_229494


namespace volume_of_given_prism_l2294_229438

/-- Represents the dimensions of a rectangular prism in centimeters -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism given its dimensions -/
def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- The dimensions of the specific rectangular prism in the problem -/
def givenPrism : PrismDimensions :=
  { length := 4
    width := 2
    height := 8 }

/-- Theorem stating that the volume of the given rectangular prism is 64 cubic centimeters -/
theorem volume_of_given_prism :
  prismVolume givenPrism = 64 := by
  sorry

#check volume_of_given_prism

end volume_of_given_prism_l2294_229438


namespace line_intersects_circle_l2294_229493

/-- Proves that a line intersects a circle given specific conditions -/
theorem line_intersects_circle (x₀ y₀ a : ℝ) (h1 : a > 0) (h2 : x₀^2 + y₀^2 > a^2) :
  ∃ (x y : ℝ), x^2 + y^2 = a^2 ∧ x₀*x + y₀*y = a^2 := by
  sorry

end line_intersects_circle_l2294_229493


namespace polynomial_less_than_factorial_l2294_229444

theorem polynomial_less_than_factorial (A B C : ℝ) : 
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (A * n^2 + B * n + C : ℝ) < n! :=
sorry

end polynomial_less_than_factorial_l2294_229444


namespace binary_representation_of_51_l2294_229469

/-- Represents a binary number as a list of bits (0 or 1) in little-endian order -/
def BinaryNumber := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : ℕ) : BinaryNumber :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Theorem: The binary representation of 51 is 110011 -/
theorem binary_representation_of_51 :
  toBinary 51 = [true, true, false, false, true, true] := by
  sorry

end binary_representation_of_51_l2294_229469


namespace alphabet_proof_main_theorem_l2294_229423

/-- Represents an alphabet with letters containing dots and/or straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  line_only : ℕ
  dot_only : ℕ
  h_total : total = both + line_only + dot_only
  h_all_types : total > 0

/-- The specific alphabet described in the problem -/
def problem_alphabet : Alphabet where
  total := 40
  both := 9
  line_only := 24
  dot_only := 7
  h_total := by rfl
  h_all_types := by norm_num

/-- Theorem stating that the problem_alphabet satisfies the given conditions -/
theorem alphabet_proof : 
  ∃ (a : Alphabet), 
    a.total = 40 ∧ 
    a.both = 9 ∧ 
    a.line_only = 24 ∧ 
    a.dot_only = 7 :=
by
  use problem_alphabet
  simp [problem_alphabet]

/-- Main theorem to prove -/
theorem main_theorem (a : Alphabet) 
  (h1 : a.total = 40)
  (h2 : a.both = 9)
  (h3 : a.line_only = 24) :
  a.dot_only = 7 :=
by
  sorry

end alphabet_proof_main_theorem_l2294_229423


namespace certain_number_solution_l2294_229426

theorem certain_number_solution : 
  ∃ x : ℝ, (5100 - (102 / x) = 5095) ∧ (x = 20.4) := by
  sorry

end certain_number_solution_l2294_229426


namespace log_relation_l2294_229407

theorem log_relation (y : ℝ) (m : ℝ) : 
  Real.log 5 / Real.log 9 = y → Real.log 125 / Real.log 3 = m * y → m = 6 := by
  sorry

end log_relation_l2294_229407


namespace no_integer_solutions_l2294_229446

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), 
    (x^6 + x^3 + x^3*y + y = 147^157) ∧ 
    (x^3 + x^3*y + y^2 + y + z^9 = 157^147) := by
  sorry

end no_integer_solutions_l2294_229446


namespace percentage_of_workday_in_meetings_l2294_229401

/-- Represents the duration of a workday in hours -/
def workday_hours : ℕ := 9

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 45

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 2 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Calculates the total workday time in minutes -/
def total_workday_minutes : ℕ := workday_hours * 60

/-- Theorem stating that the percentage of the workday spent in meetings is 25% -/
theorem percentage_of_workday_in_meetings : 
  (total_meeting_minutes : ℚ) / (total_workday_minutes : ℚ) * 100 = 25 := by
  sorry


end percentage_of_workday_in_meetings_l2294_229401


namespace complex_on_imaginary_axis_l2294_229460

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (a^2 - 2*a : ℝ) + (a^2 - a - 2 : ℝ) * I
  (z.re = 0) → (a = 0 ∨ a = 2) :=
by sorry

end complex_on_imaginary_axis_l2294_229460


namespace pet_shop_kittens_l2294_229439

theorem pet_shop_kittens (total : ℕ) (hamsters : ℕ) (birds : ℕ) (kittens : ℕ) : 
  total = 77 → hamsters = 15 → birds = 30 → kittens = total - hamsters - birds → kittens = 32 := by
  sorry

end pet_shop_kittens_l2294_229439


namespace sufficient_not_necessary_l2294_229445

theorem sufficient_not_necessary (a b : ℝ) : 
  (a > b ∧ b > 0) → (1 / a < 1 / b) ∧ 
  ¬(∀ a b : ℝ, (1 / a < 1 / b) → (a > b ∧ b > 0)) :=
by sorry

end sufficient_not_necessary_l2294_229445


namespace gravitational_force_on_space_station_l2294_229457

/-- Gravitational force calculation -/
theorem gravitational_force_on_space_station 
  (inverse_square_law : ∀ (d : ℝ) (f : ℝ), f * d^2 = (400 : ℝ) * 6000^2)
  (earth_surface_distance : ℝ := 6000)
  (earth_surface_force : ℝ := 400)
  (space_station_distance : ℝ := 360000) :
  (earth_surface_force * earth_surface_distance^2) / space_station_distance^2 = 1/9 := by
sorry

end gravitational_force_on_space_station_l2294_229457


namespace mn_gcd_lcm_equation_l2294_229428

theorem mn_gcd_lcm_equation (m n : ℕ+) :
  m * n = (Nat.gcd m n)^2 + Nat.lcm m n →
  (m = 2 ∧ n = 4) ∨ (m = 4 ∧ n = 2) := by
  sorry

end mn_gcd_lcm_equation_l2294_229428


namespace wendy_flowers_proof_l2294_229482

/-- The number of flowers in each bouquet -/
def flowers_per_bouquet : ℕ := 5

/-- The number of flowers that wilted -/
def wilted_flowers : ℕ := 35

/-- The number of bouquets that can be made after some flowers wilted -/
def remaining_bouquets : ℕ := 2

/-- The initial number of flowers Wendy picked -/
def initial_flowers : ℕ := wilted_flowers + remaining_bouquets * flowers_per_bouquet

theorem wendy_flowers_proof : initial_flowers = 45 := by
  sorry

end wendy_flowers_proof_l2294_229482


namespace floor_width_calculation_l2294_229409

/-- Given a rectangular floor of length 10 m, covered by a square carpet of side 4 m,
    with 64 square meters uncovered, the width of the floor is 8 m. -/
theorem floor_width_calculation (floor_length : ℝ) (carpet_side : ℝ) (uncovered_area : ℝ) :
  floor_length = 10 →
  carpet_side = 4 →
  uncovered_area = 64 →
  ∃ (width : ℝ), width = 8 ∧ floor_length * width = carpet_side^2 + uncovered_area :=
by sorry

end floor_width_calculation_l2294_229409


namespace right_handed_players_l2294_229468

/-- The number of right-handed players on a cricket team -/
theorem right_handed_players (total : ℕ) (throwers : ℕ) : 
  total = 55 →
  throwers = 37 →
  throwers ≤ total →
  (total - throwers) % 3 = 0 →  -- Ensures one-third of non-throwers can be left-handed
  49 = throwers + (total - throwers) - (total - throwers) / 3 := by
  sorry

end right_handed_players_l2294_229468


namespace sin_theta_value_l2294_229471

/-- Definition of determinant for 2x2 matrix -/
def det (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem: If the determinant of the given matrix is 1/2, then sin θ = ±√3/2 -/
theorem sin_theta_value (θ : ℝ) (h : det (Real.sin (θ/2)) (Real.cos (θ/2)) (Real.cos (3*θ/2)) (Real.sin (3*θ/2)) = 1/2) :
  Real.sin θ = Real.sqrt 3 / 2 ∨ Real.sin θ = -Real.sqrt 3 / 2 := by
  sorry

end sin_theta_value_l2294_229471


namespace complex_number_sum_of_parts_l2294_229452

theorem complex_number_sum_of_parts (m : ℝ) : 
  let z : ℂ := m / (1 - Complex.I) + (1 - Complex.I) / 2 * Complex.I
  (z.re + z.im = 1) → m = 1 := by sorry

end complex_number_sum_of_parts_l2294_229452


namespace club_contribution_proof_l2294_229474

/-- Proves that the initial contribution per member is $300 --/
theorem club_contribution_proof (n : ℕ) (x : ℝ) : 
  n = 10 → -- Initial number of members
  (n + 5) * (x - 100) = n * x → -- Total amount remains constant with 5 more members
  x = 300 := by
sorry

end club_contribution_proof_l2294_229474


namespace rhombus_from_equal_triangle_perimeters_l2294_229479

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The intersection point of the diagonals of a quadrilateral -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- The perimeter of a triangle defined by three points -/
def trianglePerimeter (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop :=
  sorry

/-- Theorem: If the perimeters of triangles ABO, BCO, CDO, and DAO are equal
    in a convex quadrilateral ABCD where O is the intersection of diagonals,
    then ABCD is a rhombus -/
theorem rhombus_from_equal_triangle_perimeters (q : Quadrilateral) 
  (h_convex : isConvex q) :
  let O := diagonalIntersection q
  (trianglePerimeter q.A q.B O = trianglePerimeter q.B q.C O) ∧
  (trianglePerimeter q.B q.C O = trianglePerimeter q.C q.D O) ∧
  (trianglePerimeter q.C q.D O = trianglePerimeter q.D q.A O) →
  (q.A.x - q.B.x)^2 + (q.A.y - q.B.y)^2 = 
  (q.B.x - q.C.x)^2 + (q.B.y - q.C.y)^2 ∧
  (q.B.x - q.C.x)^2 + (q.B.y - q.C.y)^2 = 
  (q.C.x - q.D.x)^2 + (q.C.y - q.D.y)^2 ∧
  (q.C.x - q.D.x)^2 + (q.C.y - q.D.y)^2 = 
  (q.D.x - q.A.x)^2 + (q.D.y - q.A.y)^2 :=
sorry

end rhombus_from_equal_triangle_perimeters_l2294_229479


namespace line_slope_and_point_l2294_229403

/-- Given two points P and Q in a coordinate plane, if the slope of the line
    through P and Q is -5/4, then the y-coordinate of Q is -2. Additionally,
    if R is a point on this line and is horizontally 6 units to the right of Q,
    then R has coordinates (11, -9.5). -/
theorem line_slope_and_point (P Q R : ℝ × ℝ) : 
  P = (-3, 8) →
  Q.1 = 5 →
  (Q.2 - P.2) / (Q.1 - P.1) = -5/4 →
  R.1 = Q.1 + 6 →
  (R.2 - Q.2) / (R.1 - Q.1) = -5/4 →
  Q.2 = -2 ∧ R = (11, -9.5) :=
by sorry

end line_slope_and_point_l2294_229403


namespace power_two_minus_one_div_by_seven_l2294_229492

theorem power_two_minus_one_div_by_seven (n : ℕ) : 
  7 ∣ (2^n - 1) ↔ 3 ∣ n :=
sorry

end power_two_minus_one_div_by_seven_l2294_229492


namespace sum_of_squares_l2294_229450

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 14)
  (eq2 : b^2 + 5*c = -13)
  (eq3 : c^2 + 7*a = -26) :
  a^2 + b^2 + c^2 = 20.75 := by
sorry

end sum_of_squares_l2294_229450


namespace farm_tax_collection_l2294_229435

/-- Represents the farm tax collection scenario in a village -/
structure FarmTaxScenario where
  /-- Total cultivated land in the village -/
  total_cultivated_land : ℝ
  /-- Tax rate applied to taxable land -/
  tax_rate : ℝ
  /-- Proportion of cultivated land that is taxable (60%) -/
  taxable_land_ratio : ℝ
  /-- Mr. William's tax payment -/
  william_tax_payment : ℝ
  /-- Proportion of Mr. William's taxable land to total taxable land (16%) -/
  william_land_ratio : ℝ

/-- Calculates the total farm tax collected from the village -/
def total_farm_tax (scenario : FarmTaxScenario) : ℝ :=
  scenario.total_cultivated_land * scenario.taxable_land_ratio * scenario.tax_rate

/-- Theorem stating that the total farm tax collected is $3000 -/
theorem farm_tax_collection (scenario : FarmTaxScenario) 
  (h1 : scenario.taxable_land_ratio = 0.6)
  (h2 : scenario.william_tax_payment = 480)
  (h3 : scenario.william_land_ratio = 0.16) :
  total_farm_tax scenario = 3000 := by
  sorry

#check farm_tax_collection

end farm_tax_collection_l2294_229435


namespace sin_300_degrees_l2294_229481

theorem sin_300_degrees : Real.sin (300 * π / 180) = -(1 / 2) := by
  sorry

end sin_300_degrees_l2294_229481


namespace function_composition_l2294_229470

-- Define the function f
def f : ℝ → ℝ := fun x => 3 * (x + 1) - 1

-- State the theorem
theorem function_composition (x : ℝ) : f x = 3 * x + 2 := by
  sorry

end function_composition_l2294_229470


namespace debby_hat_tickets_l2294_229497

/-- The number of tickets Debby spent on various items at the arcade -/
structure ArcadeTickets where
  total : ℕ
  stuffedAnimal : ℕ
  yoyo : ℕ
  hat : ℕ

/-- Theorem stating that given the conditions, Debby spent 2 tickets on the hat -/
theorem debby_hat_tickets (tickets : ArcadeTickets) 
    (h1 : tickets.total = 14)
    (h2 : tickets.stuffedAnimal = 10)
    (h3 : tickets.yoyo = 2)
    (h4 : tickets.total = tickets.stuffedAnimal + tickets.yoyo + tickets.hat) : 
  tickets.hat = 2 := by
  sorry

end debby_hat_tickets_l2294_229497


namespace bipin_twice_chandan_age_l2294_229496

/-- Proves that Bipin's age will be twice Chandan's age after 10 years -/
theorem bipin_twice_chandan_age (alok_age bipin_age chandan_age : ℕ) : 
  alok_age = 5 →
  bipin_age = 6 * alok_age →
  chandan_age = 10 →
  ∃ (years : ℕ), years = 10 ∧ bipin_age + years = 2 * (chandan_age + years) :=
by
  sorry

end bipin_twice_chandan_age_l2294_229496


namespace car_speed_first_hour_l2294_229480

/-- Proves that given a car's average speed over two hours and its speed in the second hour, 
    we can determine its speed in the first hour. -/
theorem car_speed_first_hour 
  (average_speed : ℝ) 
  (second_hour_speed : ℝ) 
  (h1 : average_speed = 90) 
  (h2 : second_hour_speed = 60) : 
  ∃ (first_hour_speed : ℝ), 
    first_hour_speed = 120 ∧ 
    average_speed = (first_hour_speed + second_hour_speed) / 2 := by
  sorry

end car_speed_first_hour_l2294_229480


namespace radius_of_Q_l2294_229405

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
axiom P : Circle
axiom Q : Circle
axiom R : Circle
axiom S : Circle

-- Define the conditions
axiom externally_tangent : P.radius + Q.radius = dist P.center Q.center ∧
                           P.radius + R.radius = dist P.center R.center ∧
                           Q.radius + R.radius = dist Q.center R.center

axiom internally_tangent : S.radius = P.radius + dist P.center S.center ∧
                           S.radius = Q.radius + dist Q.center S.center ∧
                           S.radius = R.radius + dist R.center S.center

axiom Q_R_congruent : Q.radius = R.radius

axiom P_radius : P.radius = 2

axiom P_through_S_center : dist P.center S.center = P.radius

-- Theorem to prove
theorem radius_of_Q : Q.radius = 16/9 := by sorry

end radius_of_Q_l2294_229405


namespace tank_plastering_cost_l2294_229442

/-- Calculates the total cost of plastering a rectangular tank's walls and bottom. -/
def plasteringCost (length width depth : ℝ) (costPerSqm : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * costPerSqm

/-- Proves that the plastering cost for a tank with given dimensions is 223.2 rupees. -/
theorem tank_plastering_cost :
  let length : ℝ := 25
  let width : ℝ := 12
  let depth : ℝ := 6
  let costPerSqm : ℝ := 0.30  -- 30 paise = 0.30 rupees
  plasteringCost length width depth costPerSqm = 223.2 := by
  sorry

#eval plasteringCost 25 12 6 0.30

end tank_plastering_cost_l2294_229442


namespace cylinder_equation_l2294_229419

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) (c : ℝ) : Prop :=
  ∀ p : CylindricalPoint, p ∈ S ↔ p.r = c

theorem cylinder_equation (c : ℝ) (h : c > 0) :
  IsCylinder {p : CylindricalPoint | p.r = c} c :=
by sorry

end cylinder_equation_l2294_229419


namespace f_composition_seven_l2294_229459

-- Define the function f
def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 5 * x + 1

-- State the theorem
theorem f_composition_seven : f (f (f (f (f (f 7))))) = 116 := by
  sorry

end f_composition_seven_l2294_229459


namespace fuel_consumption_problem_l2294_229421

/-- The fuel consumption problem for an aviation engineer --/
theorem fuel_consumption_problem 
  (fuel_per_person : ℝ) 
  (fuel_per_bag : ℝ) 
  (num_passengers : ℕ) 
  (num_crew : ℕ) 
  (bags_per_person : ℕ) 
  (total_fuel : ℝ) 
  (trip_distance : ℝ) 
  (h1 : fuel_per_person = 3)
  (h2 : fuel_per_bag = 2)
  (h3 : num_passengers = 30)
  (h4 : num_crew = 5)
  (h5 : bags_per_person = 2)
  (h6 : total_fuel = 106000)
  (h7 : trip_distance = 400) :
  let total_people := num_passengers + num_crew
  let total_bags := total_people * bags_per_person
  let additional_fuel_per_mile := total_people * fuel_per_person + total_bags * fuel_per_bag
  let total_fuel_per_mile := total_fuel / trip_distance
  total_fuel_per_mile - additional_fuel_per_mile = 20 := by
  sorry

end fuel_consumption_problem_l2294_229421


namespace geometric_sequence_logarithm_l2294_229411

theorem geometric_sequence_logarithm (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = -Real.sqrt 2 * a n) :
  Real.log (a 2017)^2 - Real.log (a 2016)^2 = Real.log 2 := by
  sorry

end geometric_sequence_logarithm_l2294_229411


namespace cubic_sum_over_product_l2294_229462

theorem cubic_sum_over_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) (sum_squares : a^2 + b^2 + c^2 = 3) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 := by
  sorry

end cubic_sum_over_product_l2294_229462


namespace max_value_theorem_l2294_229476

theorem max_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2*x*y + 3*y^2 = 12) :
  ∃ (M : ℝ), M = 24 + 24*Real.sqrt 3 ∧ x^2 + 2*x*y + 3*y^2 ≤ M ∧
  ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x'^2 - 2*x'*y' + 3*y'^2 = 12 ∧ x'^2 + 2*x'*y' + 3*y'^2 = M :=
by sorry

end max_value_theorem_l2294_229476


namespace star_three_four_l2294_229475

/-- The ⋆ operation defined on real numbers -/
def star (a b : ℝ) : ℝ := 4*a + 3*b - 2*a*b

/-- Theorem stating that 3 ⋆ 4 = 0 -/
theorem star_three_four : star 3 4 = 0 := by
  sorry

end star_three_four_l2294_229475


namespace fourth_number_value_l2294_229413

theorem fourth_number_value (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d) / 4 = 4)
  (h2 : (d + e + f + g) / 4 = 4)
  (h3 : (a + b + c + d + e + f + g) / 7 = 3) :
  d = 11 := by sorry

end fourth_number_value_l2294_229413


namespace equal_candies_after_sharing_l2294_229434

/-- The number of candies Minyoung and Taehyung should have to be equal -/
def target_candies (total_candies : ℕ) : ℕ :=
  total_candies / 2

/-- The number of candies Taehyung should take from Minyoung -/
def candies_to_take (minyoung_candies taehyung_candies : ℕ) : ℕ :=
  (minyoung_candies + taehyung_candies) / 2 - taehyung_candies

theorem equal_candies_after_sharing 
  (minyoung_initial : ℕ) 
  (taehyung_initial : ℕ) 
  (h1 : minyoung_initial = 9) 
  (h2 : taehyung_initial = 3) :
  let candies_taken := candies_to_take minyoung_initial taehyung_initial
  minyoung_initial - candies_taken = taehyung_initial + candies_taken ∧
  candies_taken = 3 :=
by sorry

#eval candies_to_take 9 3

end equal_candies_after_sharing_l2294_229434


namespace triangle_properties_l2294_229461

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A))
  (h2 : t.A = 2 * t.B)
  (h3 : t.A + t.B + t.C = Real.pi) :  -- Triangle angle sum property
  t.C = 5 * Real.pi / 8 ∧ 2 * t.a ^ 2 = t.b ^ 2 + t.c ^ 2 := by
  sorry

end triangle_properties_l2294_229461


namespace kira_away_time_l2294_229473

/-- Represents the eating rate of the cat in hours per pound of kibble -/
def eating_rate : ℝ := 4

/-- Represents the initial amount of kibble in pounds -/
def initial_kibble : ℝ := 3

/-- Represents the remaining amount of kibble in pounds -/
def remaining_kibble : ℝ := 1

/-- Calculates the time Kira was away based on the given conditions -/
def time_away : ℝ := (initial_kibble - remaining_kibble) * eating_rate

/-- Proves that the time Kira was away from home is 8 hours -/
theorem kira_away_time : time_away = 8 := by
  sorry

end kira_away_time_l2294_229473


namespace sqrt_equation_solution_l2294_229491

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end sqrt_equation_solution_l2294_229491


namespace siblings_have_extra_money_l2294_229433

def perfume_cost : ℚ := 100
def christian_savings : ℚ := 7
def sue_savings : ℚ := 9
def bob_savings : ℚ := 3
def christian_yards : ℕ := 7
def christian_yard_rate : ℚ := 7
def sue_dogs : ℕ := 10
def sue_dog_rate : ℚ := 4
def bob_families : ℕ := 5
def bob_family_rate : ℚ := 2
def discount_rate : ℚ := 20 / 100

def total_earnings : ℚ :=
  christian_savings + sue_savings + bob_savings +
  christian_yards * christian_yard_rate +
  sue_dogs * sue_dog_rate +
  bob_families * bob_family_rate

def discounted_price : ℚ :=
  perfume_cost * (1 - discount_rate)

theorem siblings_have_extra_money :
  total_earnings - discounted_price = 38 := by sorry

end siblings_have_extra_money_l2294_229433


namespace final_price_fraction_l2294_229487

/-- The final price of a dress for a staff member after discounts and tax -/
def final_price (d : ℝ) : ℝ :=
  let discount_price := d * (1 - 0.45)
  let staff_price := discount_price * (1 - 0.40)
  staff_price * (1 + 0.08)

/-- Theorem stating the final price as a fraction of the initial price -/
theorem final_price_fraction (d : ℝ) :
  final_price d = 0.3564 * d := by
  sorry

end final_price_fraction_l2294_229487


namespace log_equation_solution_l2294_229447

theorem log_equation_solution (x : ℝ) : Real.log (729 : ℝ) / Real.log (3 * x) = x → x = 3 := by
  sorry

end log_equation_solution_l2294_229447


namespace remaining_customers_l2294_229472

theorem remaining_customers (initial : ℕ) (left : ℕ) (remaining : ℕ) : 
  initial = 14 → left = 11 → remaining = initial - left → remaining = 3 := by
  sorry

end remaining_customers_l2294_229472


namespace binary_addition_subtraction_l2294_229484

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then
    []
  else
    (n % 2 = 1) :: nat_to_binary (n / 2)

def a : List Bool := [true, false, true, true, false, true]  -- 101101₂
def b : List Bool := [true, true, true]  -- 111₂
def c : List Bool := [false, true, true, false, false, true, true]  -- 1100110₂
def d : List Bool := [false, true, false, true]  -- 1010₂
def result : List Bool := [true, false, true, true, true, false, true, true]  -- 11011101₂

theorem binary_addition_subtraction :
  nat_to_binary ((binary_to_nat a + binary_to_nat b + binary_to_nat c) - binary_to_nat d) = result := by
  sorry

end binary_addition_subtraction_l2294_229484


namespace simplify_expression_l2294_229432

theorem simplify_expression : (2^5 + 4^3) * (2^2 - (-2)^3)^8 = 96 * 12^8 := by
  sorry

end simplify_expression_l2294_229432


namespace fraction_simplification_l2294_229451

theorem fraction_simplification {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a^(2*b) * b^(3*a)) / (b^(2*b) * a^(3*a)) = (a/b)^(2*b - 3*a) := by
  sorry

end fraction_simplification_l2294_229451


namespace fourth_cd_cost_l2294_229467

theorem fourth_cd_cost (initial_avg_cost : ℝ) (new_avg_cost : ℝ) (initial_cd_count : ℕ) :
  initial_avg_cost = 15 →
  new_avg_cost = 16 →
  initial_cd_count = 3 →
  (initial_cd_count * initial_avg_cost + (new_avg_cost * (initial_cd_count + 1) - initial_cd_count * initial_avg_cost)) = 19 := by
  sorry

end fourth_cd_cost_l2294_229467


namespace initial_production_rate_l2294_229456

/-- Proves that the initial production rate is 15 cogs per hour given the problem conditions --/
theorem initial_production_rate : 
  ∀ (initial_rate : ℝ),
  (∃ (initial_time : ℝ),
    initial_rate * initial_time = 60 ∧  -- Initial order production
    initial_time + 1 = 120 / 24 ∧       -- Total time equation
    (60 + 60) / (initial_time + 1) = 24 -- Average output equation
  ) → initial_rate = 15 := by
  sorry


end initial_production_rate_l2294_229456


namespace sparrow_population_decrease_l2294_229498

/-- The annual decrease rate of the sparrow population -/
def decrease_rate : ℝ := 0.3

/-- The threshold percentage of the initial population -/
def threshold : ℝ := 0.2

/-- The remaining population fraction after one year -/
def remaining_fraction : ℝ := 1 - decrease_rate

/-- The number of years it takes for the population to fall below the threshold -/
def years_to_threshold : ℕ := 5

theorem sparrow_population_decrease :
  (remaining_fraction ^ years_to_threshold) < threshold ∧
  ∀ n : ℕ, n < years_to_threshold → (remaining_fraction ^ n) ≥ threshold :=
by sorry

end sparrow_population_decrease_l2294_229498


namespace six_digit_square_numbers_l2294_229449

/-- Represents a 6-digit number as a tuple of its digits -/
def SixDigitNumber := (Nat × Nat × Nat × Nat × Nat × Nat)

/-- Converts a 6-digit number tuple to its numerical value -/
def toNumber (n : SixDigitNumber) : Nat :=
  match n with
  | (a, b, c, d, e, f) => 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

/-- Extracts the last three digits of a 6-digit number tuple -/
def lastThreeDigits (n : SixDigitNumber) : Nat :=
  match n with
  | (_, _, _, d, e, f) => 100 * d + 10 * e + f

/-- Checks if a given 6-digit number satisfies the condition (abcdef) = (def)^2 -/
def satisfiesCondition (n : SixDigitNumber) : Prop :=
  toNumber n = (lastThreeDigits n) ^ 2

theorem six_digit_square_numbers :
  ∀ n : SixDigitNumber,
    satisfiesCondition n →
    (toNumber n = 390625 ∨ toNumber n = 141376) :=
by sorry

end six_digit_square_numbers_l2294_229449


namespace intersection_points_on_circle_l2294_229453

/-- The parabolas y = (x - 1)^2 and x - 2 = (y + 1)^2 intersect at four points. 
    These points lie on a circle with radius squared equal to 1/4. -/
theorem intersection_points_on_circle : 
  ∃ (c : ℝ × ℝ) (r : ℝ),
    (∀ (p : ℝ × ℝ), 
      (p.2 = (p.1 - 1)^2 ∧ p.1 - 2 = (p.2 + 1)^2) → 
      (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    r^2 = (1/4 : ℝ) := by
  sorry

end intersection_points_on_circle_l2294_229453


namespace vintik_shpuntik_journey_l2294_229402

/-- The problem of Vintik and Shpuntik's journey to school -/
theorem vintik_shpuntik_journey 
  (distance : ℝ) 
  (vintik_scooter_speed : ℝ) 
  (walking_speed : ℝ) 
  (h_distance : distance = 6) 
  (h_vintik_scooter : vintik_scooter_speed = 10) 
  (h_walking : walking_speed = 5) :
  ∃ (shpuntik_bicycle_speed : ℝ),
    -- Vintik's journey
    ∃ (vintik_time : ℝ),
      vintik_time * (vintik_scooter_speed / 2 + walking_speed / 2) = distance ∧
    -- Shpuntik's journey
    (distance / 2) / shpuntik_bicycle_speed + (distance / 2) / walking_speed = vintik_time ∧
    -- Shpuntik's bicycle speed
    shpuntik_bicycle_speed = 15 := by
  sorry

end vintik_shpuntik_journey_l2294_229402


namespace units_digit_G_1000_l2294_229410

/-- Modified Fermat number -/
def G (n : ℕ) : ℕ := 5^(5^n) + 6

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_1000 : units_digit (G 1000) = 1 := by
  sorry

end units_digit_G_1000_l2294_229410


namespace walk_ratio_l2294_229427

def distance_first_hour : ℝ := 2
def total_distance : ℝ := 6

def distance_second_hour : ℝ := total_distance - distance_first_hour

theorem walk_ratio :
  distance_second_hour / distance_first_hour = 2 := by
  sorry

end walk_ratio_l2294_229427


namespace student_number_problem_l2294_229477

theorem student_number_problem (x y : ℤ) : 
  x = 121 → 2 * x - y = 102 → y = 140 := by
  sorry

end student_number_problem_l2294_229477


namespace parabola_through_point_2_4_l2294_229454

/-- A parabola passing through the point (2, 4) can be represented by either y² = 8x or x² = y -/
theorem parabola_through_point_2_4 :
  ∃ (f : ℝ → ℝ), (f 2 = 4 ∧ (∀ x y : ℝ, y = f x ↔ (y^2 = 8*x ∨ x^2 = y))) :=
by sorry

end parabola_through_point_2_4_l2294_229454


namespace prairie_total_area_l2294_229440

/-- The total area of a prairie given the area covered by a dust storm and the area left untouched. -/
theorem prairie_total_area (dust_covered : ℕ) (untouched : ℕ) 
  (h1 : dust_covered = 64535) 
  (h2 : untouched = 522) : 
  dust_covered + untouched = 65057 := by
  sorry

end prairie_total_area_l2294_229440


namespace sqrt_x_minus_one_real_l2294_229464

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end sqrt_x_minus_one_real_l2294_229464


namespace inequality_system_solution_l2294_229429

theorem inequality_system_solution (x : ℝ) :
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4 → -2 < x ∧ x ≤ 1 := by
  sorry

end inequality_system_solution_l2294_229429


namespace united_charge_per_minute_is_correct_l2294_229463

/-- Additional charge per minute for United Telephone -/
def united_charge_per_minute : ℚ := 25 / 100

/-- Base rate for United Telephone -/
def united_base_rate : ℚ := 7

/-- Base rate for Atlantic Call -/
def atlantic_base_rate : ℚ := 12

/-- Additional charge per minute for Atlantic Call -/
def atlantic_charge_per_minute : ℚ := 1 / 5

/-- Number of minutes for which the bills are equal -/
def equal_minutes : ℕ := 100

theorem united_charge_per_minute_is_correct :
  united_base_rate + equal_minutes * united_charge_per_minute =
  atlantic_base_rate + equal_minutes * atlantic_charge_per_minute :=
by sorry

end united_charge_per_minute_is_correct_l2294_229463


namespace probability_rain_at_least_one_day_l2294_229443

/-- The probability of rain on at least one day given independent probabilities for each day -/
theorem probability_rain_at_least_one_day 
  (p_friday p_saturday p_sunday : ℝ) 
  (h_friday : p_friday = 0.3)
  (h_saturday : p_saturday = 0.45)
  (h_sunday : p_sunday = 0.55)
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - p_friday) * (1 - p_saturday) * (1 - p_sunday) = 0.82675 := by
  sorry

end probability_rain_at_least_one_day_l2294_229443


namespace cafeteria_fruit_sale_l2294_229490

/-- Cafeteria fruit sale problem -/
theorem cafeteria_fruit_sale
  (initial_apples : ℕ) 
  (initial_oranges : ℕ) 
  (apple_price : ℚ) 
  (orange_price : ℚ) 
  (total_earnings : ℚ) 
  (apples_left : ℕ) 
  (h1 : initial_apples = 50)
  (h2 : initial_oranges = 40)
  (h3 : apple_price = 4/5)
  (h4 : orange_price = 1/2)
  (h5 : total_earnings = 49)
  (h6 : apples_left = 10) :
  initial_oranges - (total_earnings - (initial_apples - apples_left) * apple_price) / orange_price = 6 := by
  sorry

end cafeteria_fruit_sale_l2294_229490


namespace sum_of_a_and_b_l2294_229415

theorem sum_of_a_and_b (a b : ℝ) 
  (ha : |a| = 5)
  (hb : |b| = 3)
  (hab : |a - b| = b - a) :
  a + b = -2 ∨ a + b = -8 := by
sorry

end sum_of_a_and_b_l2294_229415


namespace circles_tangent_m_value_l2294_229499

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

-- Define external tangency condition
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y m ∧
  ∀ (x' y' : ℝ), C₁ x' y' → C₂ x' y' m → (x = x' ∧ y = y')

-- Theorem statement
theorem circles_tangent_m_value :
  ∀ m : ℝ, externally_tangent m → m = 9 :=
sorry

end circles_tangent_m_value_l2294_229499


namespace polynomial_coefficient_sum_l2294_229436

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 + (x + 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a₀| + |a₁| + |a₂| + |a₅| = 105 := by
sorry

end polynomial_coefficient_sum_l2294_229436


namespace line_segment_length_l2294_229488

/-- The length of a line segment with endpoints (1,4) and (8,16) is √193. -/
theorem line_segment_length : Real.sqrt 193 = Real.sqrt ((8 - 1)^2 + (16 - 4)^2) := by
  sorry

end line_segment_length_l2294_229488


namespace total_vegetarian_count_l2294_229408

/-- Represents the dietary preferences in a family -/
structure DietaryPreferences where
  only_vegetarian : ℕ
  only_non_vegetarian : ℕ
  both_veg_and_non_veg : ℕ
  vegan : ℕ
  vegan_and_vegetarian : ℕ
  gluten_free_from_both : ℕ

/-- Calculates the total number of people eating vegetarian food -/
def total_vegetarian (d : DietaryPreferences) : ℕ :=
  d.only_vegetarian + d.both_veg_and_non_veg + (d.vegan - d.vegan_and_vegetarian)

/-- Theorem stating the total number of people eating vegetarian food -/
theorem total_vegetarian_count (d : DietaryPreferences)
  (h1 : d.only_vegetarian = 15)
  (h2 : d.only_non_vegetarian = 8)
  (h3 : d.both_veg_and_non_veg = 11)
  (h4 : d.vegan = 5)
  (h5 : d.vegan_and_vegetarian = 3)
  (h6 : d.gluten_free_from_both = 2)
  : total_vegetarian d = 28 := by
  sorry

end total_vegetarian_count_l2294_229408


namespace complex_modulus_power_l2294_229418

theorem complex_modulus_power : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end complex_modulus_power_l2294_229418


namespace curve_classification_l2294_229430

structure Curve where
  m : ℝ
  n : ℝ

def isEllipse (c : Curve) : Prop :=
  c.m > c.n ∧ c.n > 0

def isHyperbola (c : Curve) : Prop :=
  c.m * c.n < 0

def isTwoLines (c : Curve) : Prop :=
  c.m = 0 ∧ c.n > 0

theorem curve_classification (c : Curve) :
  (isEllipse c → ∃ foci : ℝ × ℝ, foci.1 = 0) ∧
  (isHyperbola c → ∃ k : ℝ, k^2 = -c.m / c.n) ∧
  (isTwoLines c → ∃ y₁ y₂ : ℝ, y₁ = -y₂ ∧ y₁^2 = 1 / c.n) :=
sorry

end curve_classification_l2294_229430


namespace number_puzzle_l2294_229466

theorem number_puzzle : ∃! x : ℝ, (x / 5 + 4 = x / 4 - 4) := by
  sorry

end number_puzzle_l2294_229466


namespace second_share_interest_rate_l2294_229495

theorem second_share_interest_rate 
  (total_investment : ℝ)
  (first_share_yield : ℝ)
  (total_interest_rate : ℝ)
  (second_share_investment : ℝ)
  (h1 : total_investment = 100000)
  (h2 : first_share_yield = 0.09)
  (h3 : total_interest_rate = 0.0925)
  (h4 : second_share_investment = 12500) :
  ∃ (second_share_yield : ℝ),
    second_share_yield = 0.11 ∧
    total_investment * total_interest_rate = 
      (total_investment - second_share_investment) * first_share_yield +
      second_share_investment * second_share_yield :=
by
  sorry

end second_share_interest_rate_l2294_229495


namespace not_perfect_square_l2294_229422

theorem not_perfect_square : ∃ (n : ℕ), n = 6^2041 ∧
  (∀ (m : ℕ), m^2 ≠ n) ∧
  (∃ (a : ℕ), 3^2040 = a^2) ∧
  (∃ (b : ℕ), 7^2042 = b^2) ∧
  (∃ (c : ℕ), 8^2043 = c^2) ∧
  (∃ (d : ℕ), 9^2044 = d^2) :=
by sorry

end not_perfect_square_l2294_229422


namespace two_digit_product_sum_l2294_229412

theorem two_digit_product_sum : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 3024 ∧ 
  a + b = 120 := by
sorry

end two_digit_product_sum_l2294_229412


namespace consecutive_lucky_tickets_exist_l2294_229414

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is lucky if the sum of its digits is divisible by 7 -/
def is_lucky (n : ℕ) : Prop := sum_of_digits n % 7 = 0

/-- There exist two consecutive lucky bus ticket numbers -/
theorem consecutive_lucky_tickets_exist : ∃ n : ℕ, is_lucky n ∧ is_lucky (n + 1) := by sorry

end consecutive_lucky_tickets_exist_l2294_229414


namespace adas_original_seat_was_two_l2294_229424

/-- Represents the seats in the theater --/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends sitting in the theater --/
inductive Friend
| Ada
| Bea
| Ceci
| Dee
| Edie
| Faye

/-- Represents the state of the seating arrangement --/
structure SeatingArrangement where
  seats : Fin 6 → Option Friend

/-- Defines a valid initial seating arrangement --/
def validInitialArrangement (arr : SeatingArrangement) : Prop :=
  ∃ (emptySlot : Fin 6), 
    (∀ i : Fin 6, i ≠ emptySlot → arr.seats i ≠ none) ∧
    (arr.seats emptySlot = none) ∧
    (∃ (ada bea ceci dee edie faye : Fin 6), 
      ada ≠ bea ∧ ada ≠ ceci ∧ ada ≠ dee ∧ ada ≠ edie ∧ ada ≠ faye ∧
      bea ≠ ceci ∧ bea ≠ dee ∧ bea ≠ edie ∧ bea ≠ faye ∧
      ceci ≠ dee ∧ ceci ≠ edie ∧ ceci ≠ faye ∧
      dee ≠ edie ∧ dee ≠ faye ∧
      edie ≠ faye ∧
      arr.seats ada = some Friend.Ada ∧
      arr.seats bea = some Friend.Bea ∧
      arr.seats ceci = some Friend.Ceci ∧
      arr.seats dee = some Friend.Dee ∧
      arr.seats edie = some Friend.Edie ∧
      arr.seats faye = some Friend.Faye)

/-- Defines the final seating arrangement after movements --/
def finalArrangement (initial : SeatingArrangement) (final : SeatingArrangement) : Prop :=
  ∃ (bea bea' ceci ceci' dee dee' edie edie' : Fin 6),
    initial.seats bea = some Friend.Bea ∧
    initial.seats ceci = some Friend.Ceci ∧
    initial.seats dee = some Friend.Dee ∧
    initial.seats edie = some Friend.Edie ∧
    bea' = (bea + 3) % 6 ∧
    ceci' = (ceci + 2) % 6 ∧
    dee' ≠ dee ∧ edie' ≠ edie ∧
    final.seats bea' = some Friend.Bea ∧
    final.seats ceci' = some Friend.Ceci ∧
    final.seats dee' = some Friend.Dee ∧
    final.seats edie' = some Friend.Edie ∧
    (final.seats 0 = none ∨ final.seats 5 = none)

/-- Theorem: Ada's original seat was Seat 2 --/
theorem adas_original_seat_was_two 
  (initial final : SeatingArrangement)
  (h_initial : validInitialArrangement initial)
  (h_final : finalArrangement initial final) :
  ∃ (ada : Fin 6), initial.seats ada = some Friend.Ada ∧ ada = 1 := by
  sorry

end adas_original_seat_was_two_l2294_229424
