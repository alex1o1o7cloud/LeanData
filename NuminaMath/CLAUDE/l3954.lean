import Mathlib

namespace quadratic_roots_quadratic_function_l3954_395437

-- Part 1
theorem quadratic_roots (a b c : ℝ) 
  (h : Real.sqrt (a - 2) + abs (b + 1) + (c + 2)^2 = 0) :
  let f := fun x => a * x^2 + b * x + c
  ∃ x1 x2 : ℝ, x1 = (1 + Real.sqrt 17) / 4 ∧ 
              x2 = (1 - Real.sqrt 17) / 4 ∧
              f x1 = 0 ∧ f x2 = 0 :=
sorry

-- Part 2
theorem quadratic_function (a b c : ℝ) 
  (h1 : a * (-1)^2 + b * (-1) + c = 0)
  (h2 : a * 0^2 + b * 0 + c = -3)
  (h3 : a * 3^2 + b * 3 + c = 0) :
  ∀ x : ℝ, a * x^2 + b * x + c = x^2 - 2*x - 3 :=
sorry

end quadratic_roots_quadratic_function_l3954_395437


namespace rectangle_area_ratio_l3954_395485

/-- Given a rectangle ABCD with vertices A(0,0), B(0,2), C(3,2), and D(3,0),
    point E as the midpoint of diagonal BD, and point F on DA such that DF = 1/4 DA,
    prove that the ratio of the area of triangle DFE to the area of quadrilateral ABEF is 3/17. -/
theorem rectangle_area_ratio :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ × ℝ := (3, 2)
  let D : ℝ × ℝ := (3, 0)
  let E : ℝ × ℝ := ((D.1 + B.1) / 2, (D.2 + B.2) / 2)
  let F : ℝ × ℝ := (D.1 - (D.1 - A.1) / 4, A.2)
  let area_DFE := abs ((D.1 - F.1) * E.2) / 2
  let area_ABE := abs (B.1 * E.2 - E.1 * B.2) / 2
  let area_AEF := abs ((F.1 - A.1) * E.2) / 2
  let area_ABEF := area_ABE + area_AEF
  area_DFE / area_ABEF = 3 / 17 :=
by sorry

end rectangle_area_ratio_l3954_395485


namespace min_height_box_l3954_395422

theorem min_height_box (x : ℝ) (h : x > 0) : 
  (2*x^2 + 4*x*(x + 4) ≥ 120) → (x + 4 ≥ 8) :=
by
  sorry

#check min_height_box

end min_height_box_l3954_395422


namespace length_BC_l3954_395419

/-- Right triangles ABC and ABD with specific properties -/
structure RightTriangles where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- D is on the x-axis
  h_D_on_x : D.2 = 0
  -- C is directly below A on the x-axis
  h_C_below_A : C.1 = A.1
  -- Distances
  h_AD : dist A D = 26
  h_BD : dist B D = 10
  h_AC : dist A C = 24
  -- ABC is a right triangle
  h_ABC_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- ABD is a right triangle
  h_ABD_right : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0

/-- The length of BC in the given configuration of right triangles -/
theorem length_BC (t : RightTriangles) : dist t.B t.C = 24 * Real.sqrt 2 := by
  sorry

end length_BC_l3954_395419


namespace carla_school_distance_l3954_395401

theorem carla_school_distance (grocery_distance : ℝ) (soccer_distance : ℝ) 
  (mpg : ℝ) (gas_price : ℝ) (gas_spent : ℝ) :
  grocery_distance = 8 →
  soccer_distance = 12 →
  mpg = 25 →
  gas_price = 2.5 →
  gas_spent = 5 →
  ∃ (school_distance : ℝ),
    grocery_distance + school_distance + soccer_distance + 2 * school_distance = 
      (gas_spent / gas_price) * mpg ∧
    school_distance = 10 := by
  sorry

end carla_school_distance_l3954_395401


namespace revolver_game_probability_l3954_395475

/-- Represents a six-shot revolver with one bullet -/
structure Revolver :=
  (chambers : Fin 6)
  (bullet : Fin 6)

/-- Represents the state of the game -/
inductive GameState
  | A
  | B

/-- The probability of firing the bullet on a single shot -/
def fire_probability : ℚ := 1 / 6

/-- The probability of not firing the bullet on a single shot -/
def not_fire_probability : ℚ := 1 - fire_probability

/-- The probability that A fires the bullet -/
noncomputable def prob_A_fires : ℚ :=
  fire_probability / (1 - not_fire_probability * not_fire_probability)

theorem revolver_game_probability :
  prob_A_fires = 6 / 11 :=
sorry

end revolver_game_probability_l3954_395475


namespace probability_of_sum_three_is_one_over_216_l3954_395461

def standard_die := Finset.range 6

def roll_sum (a b c : ℕ) : ℕ := a + b + c

def probability_of_sum_three : ℚ :=
  (Finset.filter (λ (abc : ℕ × ℕ × ℕ) => roll_sum abc.1 abc.2.1 abc.2.2 = 3) 
    (standard_die.product (standard_die.product standard_die))).card / 
  (standard_die.card ^ 3 : ℚ)

theorem probability_of_sum_three_is_one_over_216 :
  probability_of_sum_three = 1 / 216 := by sorry

end probability_of_sum_three_is_one_over_216_l3954_395461


namespace first_day_of_month_l3954_395431

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

theorem first_day_of_month (d : DayOfWeek) :
  advanceDay d 29 = DayOfWeek.Monday → d = DayOfWeek.Sunday :=
by sorry


end first_day_of_month_l3954_395431


namespace seven_unit_disks_cover_radius_two_disk_l3954_395463

-- Define a disk as a pair (center, radius)
def Disk := ℝ × ℝ × ℝ

-- Define a function to check if a point is covered by a disk
def is_covered (point : ℝ × ℝ) (disk : Disk) : Prop :=
  let (cx, cy, r) := disk
  (point.1 - cx)^2 + (point.2 - cy)^2 ≤ r^2

-- Define a function to check if a point is covered by any disk in a list
def is_covered_by_any (point : ℝ × ℝ) (disks : List Disk) : Prop :=
  ∃ d ∈ disks, is_covered point d

-- Define the main theorem
theorem seven_unit_disks_cover_radius_two_disk :
  ∃ (arrangement : List Disk),
    (arrangement.length = 7) ∧
    (∀ d ∈ arrangement, d.2.2 = 1) ∧
    (∀ point : ℝ × ℝ, point.1^2 + point.2^2 ≤ 4 → is_covered_by_any point arrangement) :=
sorry

end seven_unit_disks_cover_radius_two_disk_l3954_395463


namespace student_average_grade_l3954_395476

theorem student_average_grade
  (courses_last_year : ℕ)
  (courses_year_before : ℕ)
  (avg_grade_year_before : ℚ)
  (avg_grade_two_years : ℚ)
  (h1 : courses_last_year = 6)
  (h2 : courses_year_before = 5)
  (h3 : avg_grade_year_before = 70)
  (h4 : avg_grade_two_years = 86)
  : ∃ x : ℚ, x = 596 / 6 ∧ 
    (courses_year_before * avg_grade_year_before + courses_last_year * x) / 
    (courses_year_before + courses_last_year) = avg_grade_two_years :=
by sorry

end student_average_grade_l3954_395476


namespace third_side_of_similar_altitude_triangle_l3954_395414

/-- A triangle with sides a, b, and c, where the triangle is similar to the triangle formed by its altitudes. -/
structure SimilarAltitudeTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  similar_to_altitude : a * b * c = 2 * (a^2 + b^2 + c^2)

/-- Theorem: In a triangle similar to its altitude triangle with two sides 9 and 4, the third side is 6. -/
theorem third_side_of_similar_altitude_triangle :
  ∀ (t : SimilarAltitudeTriangle), t.a = 9 → t.b = 4 → t.c = 6 := by
  sorry

#check third_side_of_similar_altitude_triangle

end third_side_of_similar_altitude_triangle_l3954_395414


namespace kola_age_is_16_l3954_395400

/-- Kola's current age -/
def kola_age : ℕ := sorry

/-- Ola's current age -/
def ola_age : ℕ := sorry

/-- Kola's age is twice Ola's age when Kola was Ola's current age -/
axiom condition1 : kola_age = 2 * (ola_age - (kola_age - ola_age))

/-- Sum of their ages when Ola reaches Kola's current age is 36 -/
axiom condition2 : kola_age + (kola_age + (kola_age - ola_age)) = 36

/-- Theorem stating Kola's current age is 16 -/
theorem kola_age_is_16 : kola_age = 16 := by sorry

end kola_age_is_16_l3954_395400


namespace outbound_speed_l3954_395488

/-- Proves that given a round trip of 2 hours, with an outbound journey of 70 minutes
    and a return journey at 105 km/h, the outbound journey speed is 75 km/h -/
theorem outbound_speed (total_time : Real) (outbound_time : Real) (return_speed : Real) :
  total_time = 2 →
  outbound_time = 70 / 60 →
  return_speed = 105 →
  (total_time - outbound_time) * return_speed = outbound_time * 75 := by
  sorry

#check outbound_speed

end outbound_speed_l3954_395488


namespace sphere_division_l3954_395468

theorem sphere_division (π : ℝ) (h_π : π > 0) : 
  ∃ (R : ℝ), R > 0 ∧ 
  (4 / 3 * π * R^3 = 125 * (4 / 3 * π * 1^3)) ∧ 
  R = 5 := by
sorry

end sphere_division_l3954_395468


namespace perpendicular_planes_from_perpendicular_line_l3954_395499

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations
def contained_in (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry

-- State the theorem
theorem perpendicular_planes_from_perpendicular_line 
  (α β : Plane) (l : Line) :
  contained_in l β → perpendicular_line_plane l α → perpendicular_plane_plane α β := by
  sorry

end perpendicular_planes_from_perpendicular_line_l3954_395499


namespace remaining_cards_l3954_395445

-- Define the initial number of baseball cards Mike has
def initial_cards : ℕ := 87

-- Define the number of cards Sam bought
def bought_cards : ℕ := 13

-- Theorem stating that Mike's remaining cards is the difference between initial and bought
theorem remaining_cards : initial_cards - bought_cards = 74 := by
  sorry

end remaining_cards_l3954_395445


namespace average_weight_of_children_l3954_395458

def regression_equation (x : ℝ) : ℝ := 2 * x + 7

def children_ages : List ℝ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

theorem average_weight_of_children :
  let weights := children_ages.map regression_equation
  (weights.sum / weights.length) = 15 := by
  sorry

end average_weight_of_children_l3954_395458


namespace perpendicular_line_x_intercept_l3954_395432

/-- Given a line L1: 2x + 3y = 9, prove that a line L2 perpendicular to L1 with y-intercept 5 has x-intercept -10/3 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := fun x y ↦ 2 * x + 3 * y = 9
  let m1 : ℝ := -2 / 3  -- slope of L1
  let m2 : ℝ := -1 / m1  -- slope of perpendicular line
  let L2 : ℝ → ℝ → Prop := fun x y ↦ y = m2 * x + 5  -- equation of perpendicular line
  let x_intercept : ℝ := -10 / 3
  (∀ x y, L2 x y → y = 0 → x = x_intercept) :=
by
  sorry


end perpendicular_line_x_intercept_l3954_395432


namespace yadav_expenditure_l3954_395452

theorem yadav_expenditure (monthly_salary : ℝ) : 
  monthly_salary > 0 →
  (0.6 * monthly_salary) + (0.5 * (0.4 * monthly_salary)) + (0.2 * monthly_salary) = monthly_salary →
  (0.2 * monthly_salary) * 12 = 24624 →
  0.5 * (0.4 * monthly_salary) = 2052 := by
sorry

end yadav_expenditure_l3954_395452


namespace modulus_of_complex_quotient_l3954_395471

/-- The modulus of the complex number (4+3i)/(1-2i) is √5 -/
theorem modulus_of_complex_quotient :
  Complex.abs ((4 : ℂ) + 3 * Complex.I) / ((1 : ℂ) - 2 * Complex.I) = Real.sqrt 5 := by
  sorry

end modulus_of_complex_quotient_l3954_395471


namespace convoy_vehicles_l3954_395442

theorem convoy_vehicles (bridge_length : ℝ) (convoy_speed : ℝ) (crossing_time : ℝ)
                        (vehicle_length : ℝ) (gap_length : ℝ) :
  bridge_length = 298 →
  convoy_speed = 4 →
  crossing_time = 115 →
  vehicle_length = 6 →
  gap_length = 20 →
  ∃ (n : ℕ), n * vehicle_length + (n - 1) * gap_length = convoy_speed * crossing_time - bridge_length ∧
             n = 7 :=
by sorry

end convoy_vehicles_l3954_395442


namespace marks_initial_money_l3954_395459

theorem marks_initial_money (x : ℝ) : 
  x / 2 + 14 + x / 3 + 16 = x → x = 180 :=
by sorry

end marks_initial_money_l3954_395459


namespace binomial_9_choose_5_l3954_395423

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by
  sorry

end binomial_9_choose_5_l3954_395423


namespace digit_1234_is_4_l3954_395451

def decimal_sequence : ℕ → ℕ
  | 0 => 0  -- represents the decimal point
  | n+1 => 
    let k := (n-1) / 3 + 100
    if k ≤ 500 then
      match (n-1) % 3 with
      | 0 => k / 100
      | 1 => (k / 10) % 10
      | _ => k % 10
    else 0

theorem digit_1234_is_4 : decimal_sequence 1234 = 4 := by
  sorry

end digit_1234_is_4_l3954_395451


namespace gcd_315_168_l3954_395473

theorem gcd_315_168 : Nat.gcd 315 168 = 21 := by
  sorry

end gcd_315_168_l3954_395473


namespace smaller_angle_measure_l3954_395469

/-- A parallelogram with one angle exceeding the other by 70 degrees -/
structure SpecialParallelogram where
  /-- The measure of the smaller angle in degrees -/
  smaller_angle : ℝ
  /-- The measure of the larger angle in degrees -/
  larger_angle : ℝ
  /-- The larger angle exceeds the smaller angle by 70 degrees -/
  angle_difference : larger_angle = smaller_angle + 70
  /-- The sum of adjacent angles is 180 degrees -/
  angle_sum : smaller_angle + larger_angle = 180

/-- The measure of the smaller angle in a special parallelogram is 55 degrees -/
theorem smaller_angle_measure (p : SpecialParallelogram) : p.smaller_angle = 55 := by
  sorry

end smaller_angle_measure_l3954_395469


namespace power_of_power_three_l3954_395495

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end power_of_power_three_l3954_395495


namespace hide_and_seek_time_l3954_395407

/-- Represents the square wall in the hide and seek game -/
structure Square :=
  (side_length : ℝ)

/-- Represents a player in the hide and seek game -/
structure Player :=
  (speed : ℝ)
  (corner_pause : ℝ)

/-- Calculates the time needed for a player to see the other player -/
def time_to_see (s : Square) (a b : Player) : ℝ :=
  sorry

/-- Theorem stating that the minimum time for A to see B is 8 minutes -/
theorem hide_and_seek_time (s : Square) (a b : Player) :
  s.side_length = 100 ∧
  a.speed = 50 ∧
  b.speed = 30 ∧
  a.corner_pause = 1 ∧
  b.corner_pause = 1 →
  time_to_see s a b = 8 :=
sorry

end hide_and_seek_time_l3954_395407


namespace series_sum_equals_n_l3954_395421

/-- The floor function, denoted as ⌊x⌋, returns the greatest integer less than or equal to x. -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The sum of the series for a given positive integer n -/
noncomputable def series_sum (n : ℕ+) : ℝ :=
  ∑' k : ℕ, (floor ((n : ℝ) + 2^k) / 2^(k+1))

/-- Theorem stating that the sum of the series equals n for every positive integer n -/
theorem series_sum_equals_n (n : ℕ+) : series_sum n = n :=
  sorry

end series_sum_equals_n_l3954_395421


namespace cars_added_during_play_l3954_395484

/-- The number of cars added during a play, given initial car counts and final total. -/
def cars_added (front_initial : ℕ) (back_multiplier : ℕ) (total_final : ℕ) : ℕ :=
  total_final - (front_initial + back_multiplier * front_initial)

/-- Theorem stating that 400 cars were added during the play. -/
theorem cars_added_during_play :
  cars_added 100 2 700 = 400 := by sorry

end cars_added_during_play_l3954_395484


namespace valid_license_plates_l3954_395497

/-- The number of valid English letters for the license plate. --/
def validLetters : Nat := 24

/-- The number of positions to choose from for placing the letters. --/
def positionsForLetters : Nat := 4

/-- The number of letter positions to fill. --/
def letterPositions : Nat := 2

/-- The number of digit positions to fill. --/
def digitPositions : Nat := 3

/-- The number of possible digits (0-9). --/
def possibleDigits : Nat := 10

/-- Calculates the number of ways to choose k items from n items. --/
def choose (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of valid license plate combinations. --/
def totalCombinations : Nat :=
  choose positionsForLetters letterPositions * 
  validLetters ^ letterPositions * 
  possibleDigits ^ digitPositions

/-- Theorem stating that the total number of valid license plate combinations is 3,456,000. --/
theorem valid_license_plates : totalCombinations = 3456000 := by
  sorry

end valid_license_plates_l3954_395497


namespace problem_statement_l3954_395491

theorem problem_statement :
  ∀ m n : ℤ,
  m = -(-6) →
  -n = -1 →
  m * n - 7 = -1 := by
sorry

end problem_statement_l3954_395491


namespace arithmetic_sequence_common_difference_l3954_395494

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) (d : ℚ) 
  (h1 : arithmetic_sequence a d)
  (h2 : a 7 * a 11 = 6)
  (h3 : a 4 + a 14 = 5) :
  d = 1/4 ∨ d = -1/4 := by
  sorry

end arithmetic_sequence_common_difference_l3954_395494


namespace number_equality_l3954_395493

theorem number_equality (x : ℚ) : 
  (35 / 100 : ℚ) * x = (20 / 100 : ℚ) * 50 → x = 200 / 7 := by
  sorry

end number_equality_l3954_395493


namespace raccoon_carrots_l3954_395430

theorem raccoon_carrots (raccoon_per_hole rabbit_per_hole : ℕ) 
  (hole_difference : ℕ) (total_carrots : ℕ) : 
  raccoon_per_hole = 5 →
  rabbit_per_hole = 8 →
  hole_difference = 3 →
  raccoon_per_hole * (hole_difference + total_carrots / rabbit_per_hole) = total_carrots →
  total_carrots = 40 :=
by
  sorry

#check raccoon_carrots

end raccoon_carrots_l3954_395430


namespace arithmetic_sequence_sum_l3954_395408

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 48 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 3 + a 10 + a 11 = 48

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : sum_condition a) : 
  a 6 + a 7 = 24 := by
sorry

end arithmetic_sequence_sum_l3954_395408


namespace multiply_eight_negative_half_l3954_395418

theorem multiply_eight_negative_half : 8 * (-1/2 : ℚ) = -4 := by
  sorry

end multiply_eight_negative_half_l3954_395418


namespace systematic_sampling_problem_l3954_395443

/-- Represents a systematic sampling method -/
def systematicSample (totalSize : Nat) (sampleSize : Nat) (startingNumber : Nat) : List Nat :=
  List.range sampleSize |>.map (fun i => startingNumber + i * (totalSize / sampleSize))

/-- The problem statement -/
theorem systematic_sampling_problem (totalSize sampleSize startingNumber : Nat) 
  (h1 : totalSize = 60)
  (h2 : sampleSize = 6)
  (h3 : startingNumber = 7) :
  systematicSample totalSize sampleSize startingNumber = [7, 17, 27, 37, 47, 57] := by
  sorry

#eval systematicSample 60 6 7

end systematic_sampling_problem_l3954_395443


namespace sum_of_15th_set_l3954_395449

/-- Represents the first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 3 + (n - 1) * n / 2

/-- Represents the last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- Represents the sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

theorem sum_of_15th_set : S 15 = 1725 := by sorry

end sum_of_15th_set_l3954_395449


namespace cylinder_no_triangular_cross_section_l3954_395472

-- Define the type for geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | TriangularPrism
  | Cube

-- Define a function that determines if a solid can have a triangular cross-section
def canHaveTriangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => False
  | _ => True

-- Theorem stating that only the Cylinder cannot have a triangular cross-section
theorem cylinder_no_triangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveTriangularCrossSection solid) ↔ solid = GeometricSolid.Cylinder :=
by sorry

end cylinder_no_triangular_cross_section_l3954_395472


namespace smallest_matching_set_size_l3954_395435

theorem smallest_matching_set_size : ∃ (N₁ N₂ : Nat), 
  (10000 ≤ N₁ ∧ N₁ < 100000) ∧ 
  (10000 ≤ N₂ ∧ N₂ < 100000) ∧ 
  ∀ (A : Nat), 
    (10000 ≤ A ∧ A < 100000) → 
    (∀ (i j : Fin 5), i ≤ j → (A / 10^(4 - i.val) % 10) ≤ (A / 10^(4 - j.val) % 10)) →
    ∃ (k : Fin 5), 
      ((N₁ / 10^(4 - k.val)) % 10 = (A / 10^(4 - k.val)) % 10) ∨ 
      ((N₂ / 10^(4 - k.val)) % 10 = (A / 10^(4 - k.val)) % 10) := by
  sorry

end smallest_matching_set_size_l3954_395435


namespace power_function_exponent_l3954_395448

/-- A power function passing through (1/4, 1/2) has exponent 1/2 -/
theorem power_function_exponent (m : ℝ) (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = m * x^a) ∧ f (1/4) = 1/2) →
  a = 1/2 := by
sorry

end power_function_exponent_l3954_395448


namespace pastries_cakes_difference_l3954_395424

/-- The number of cakes made by the baker -/
def cakes_made : ℕ := 105

/-- The number of pastries made by the baker -/
def pastries_made : ℕ := 275

/-- The number of pastries sold by the baker -/
def pastries_sold : ℕ := 214

/-- The number of cakes sold by the baker -/
def cakes_sold : ℕ := 163

/-- Theorem stating the difference between pastries and cakes sold -/
theorem pastries_cakes_difference :
  pastries_sold - cakes_sold = 51 := by sorry

end pastries_cakes_difference_l3954_395424


namespace min_value_theorem_l3954_395406

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_theorem (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n : ℕ, a m * a n = 8 * (a 1)^2) →
  (∃ m n : ℕ, 1 / m + 4 / n = 11 / 6 ∧
    ∀ k l : ℕ, 1 / k + 4 / l ≥ 11 / 6) :=
by sorry

end min_value_theorem_l3954_395406


namespace largest_prime_factor_of_sum_of_divisors_180_l3954_395426

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number -/
def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end largest_prime_factor_of_sum_of_divisors_180_l3954_395426


namespace seven_balls_three_boxes_l3954_395444

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 36 := by
  sorry

end seven_balls_three_boxes_l3954_395444


namespace complex_on_line_l3954_395479

/-- Given a complex number z = (2a-i)/i that corresponds to a point on the line x-y=0 in the complex plane, prove that a = 1/2 --/
theorem complex_on_line (a : ℝ) : 
  let z : ℂ := (2*a - Complex.I) / Complex.I
  (z.re - z.im = 0) → a = 1/2 := by
sorry

end complex_on_line_l3954_395479


namespace algebraic_expression_value_l3954_395409

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x * y = -2) 
  (h2 : y - 2 * x = 5) : 
  8 * x^3 * y - 8 * x^2 * y^2 + 2 * x * y^3 = -100 :=
by sorry

end algebraic_expression_value_l3954_395409


namespace divisible_by_33_pairs_count_l3954_395440

theorem divisible_by_33_pairs_count : 
  (Finset.filter (fun p : ℕ × ℕ => 
    1 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ 40 ∧ (p.1 * p.2) % 33 = 0) 
    (Finset.product (Finset.range 40) (Finset.range 41))).card = 64 := by
  sorry

end divisible_by_33_pairs_count_l3954_395440


namespace sum_of_three_numbers_l3954_395441

theorem sum_of_three_numbers : ∀ (a b c : ℕ),
  b = 72 →
  a = 2 * b →
  c = a / 3 →
  a + b + c = 264 :=
by
  sorry

end sum_of_three_numbers_l3954_395441


namespace sin_minus_cos_value_l3954_395446

theorem sin_minus_cos_value (α : Real) 
  (h : ∃ (r : Real), r * (Real.cos (α - π/4)) = -1 ∧ r * (Real.sin (α - π/4)) = Real.sqrt 2) : 
  Real.sin α - Real.cos α = 2 * Real.sqrt 3 / 3 := by
sorry

end sin_minus_cos_value_l3954_395446


namespace solve_for_i_l3954_395433

-- Define the equation as a function of x and i
def equation (x i : ℝ) : Prop :=
  (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / i

-- State the theorem
theorem solve_for_i :
  ∃ i : ℝ, equation 0.3 i ∧ abs (i - 2.9993) < 0.0001 := by
  sorry

end solve_for_i_l3954_395433


namespace cube_root_of_negative_eight_l3954_395466

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end cube_root_of_negative_eight_l3954_395466


namespace vector_operations_l3954_395436

/-- Given vectors a and b in ℝ², prove their sum and dot product -/
theorem vector_operations (a b : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (3, 1)) : 
  (a.1 + b.1, a.2 + b.2) = (4, 3) ∧ a.1 * b.1 + a.2 * b.2 = 5 := by
  sorry

end vector_operations_l3954_395436


namespace basketball_tryouts_l3954_395410

theorem basketball_tryouts (girls : ℕ) (called_back : ℕ) (not_selected : ℕ) :
  girls = 17 → called_back = 10 → not_selected = 39 →
  ∃ (boys : ℕ), girls + boys = called_back + not_selected ∧ boys = 32 := by
  sorry

end basketball_tryouts_l3954_395410


namespace arithmetic_equality_l3954_395481

theorem arithmetic_equality : 1357 + 3571 + 5713 - 7135 = 3506 := by
  sorry

end arithmetic_equality_l3954_395481


namespace special_triangle_ratio_constant_l3954_395498

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = c^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = a^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = b^2

-- Define the property AC^2 + BC^2 = 2 AB^2
def SpecialTriangleProperty (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 + (C.1 - B.1)^2 + (C.2 - B.2)^2 = 
  2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define point M as the midpoint of AB
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define the angle equality ∠ACD = ∠BCD
def EqualAngles (A B C D : ℝ × ℝ) : Prop :=
  let v1 := (A.1 - C.1, A.2 - C.2)
  let v2 := (D.1 - C.1, D.2 - C.2)
  let v3 := (B.1 - C.1, B.2 - C.2)
  (v1.1 * v2.1 + v1.2 * v2.2)^2 / ((v1.1^2 + v1.2^2) * (v2.1^2 + v2.2^2)) =
  (v3.1 * v2.1 + v3.2 * v2.2)^2 / ((v3.1^2 + v3.2^2) * (v2.1^2 + v2.2^2))

-- Define D as the incenter of triangle CEM
def Incenter (D C E M : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = r^2 ∧
  (D.1 - E.1)^2 + (D.2 - E.2)^2 = r^2 ∧
  (D.1 - M.1)^2 + (D.2 - M.2)^2 = r^2

-- Main theorem
theorem special_triangle_ratio_constant 
  (A B C M D E : ℝ × ℝ) :
  Triangle A B C →
  SpecialTriangleProperty A B C →
  Midpoint M A B →
  EqualAngles A B C D →
  Incenter D C E M →
  (E.1 - M.1)^2 + (E.2 - M.2)^2 = 
  (1/9) * ((M.1 - C.1)^2 + (M.2 - C.2)^2) :=
by sorry

end special_triangle_ratio_constant_l3954_395498


namespace square_root_of_25_l3954_395464

-- Define the concept of square root
def is_square_root (x y : ℝ) : Prop := y^2 = x

-- Theorem statement
theorem square_root_of_25 : 
  ∃ (a b : ℝ), a ≠ b ∧ is_square_root 25 a ∧ is_square_root 25 b :=
sorry

end square_root_of_25_l3954_395464


namespace construction_material_total_l3954_395438

theorem construction_material_total (gravel sand : ℝ) 
  (h1 : gravel = 5.91) (h2 : sand = 8.11) : 
  gravel + sand = 14.02 := by
  sorry

end construction_material_total_l3954_395438


namespace lollipops_eaten_by_children_l3954_395460

/-- The number of lollipops Sushi's father bought -/
def initial_lollipops : ℕ := 12

/-- The number of lollipops left -/
def remaining_lollipops : ℕ := 7

/-- The number of lollipops eaten by the children -/
def eaten_lollipops : ℕ := initial_lollipops - remaining_lollipops

theorem lollipops_eaten_by_children : eaten_lollipops = 5 := by
  sorry

end lollipops_eaten_by_children_l3954_395460


namespace josh_pencils_l3954_395467

theorem josh_pencils (initial : ℕ) (given_away : ℕ) (left : ℕ) : 
  given_away = 31 → left = 111 → initial = given_away + left →
  initial = 142 := by sorry

end josh_pencils_l3954_395467


namespace no_such_function_exists_l3954_395474

open Set
open Function
open Real

theorem no_such_function_exists :
  ¬∃ f : {x : ℝ | x > 0} → {x : ℝ | x > 0},
    ∀ (x y : ℝ) (hx : x > 0) (hy : y > 0),
      f ⟨x + y, add_pos hx hy⟩ ≥ f ⟨x, hx⟩ + y * f (f ⟨x, hx⟩) :=
by
  sorry


end no_such_function_exists_l3954_395474


namespace right_triangle_circles_radius_l3954_395404

/-- Represents a right triangle with a circle tangent to one side and another circle --/
structure RightTriangleWithCircles where
  -- The length of side AC
  ac : ℝ
  -- The length of side AB
  ab : ℝ
  -- The radius of circle C
  rc : ℝ
  -- The radius of circle A
  ra : ℝ
  -- Circle C is tangent to AB
  c_tangent_ab : True
  -- Circle A and circle C are tangent
  a_tangent_c : True
  -- Angle C is 90 degrees
  angle_c_90 : True

/-- The main theorem --/
theorem right_triangle_circles_radius 
  (t : RightTriangleWithCircles) 
  (h1 : t.ac = 6) 
  (h2 : t.ab = 10) : 
  t.ra = 1.2 ∨ t.ra = 10.8 := by
  sorry

end right_triangle_circles_radius_l3954_395404


namespace truck_wheels_count_l3954_395477

/-- The toll formula for a truck crossing a bridge -/
def toll_formula (x : ℕ) : ℚ :=
  1.50 + 1.50 * (x - 2)

/-- The number of wheels on the front axle of the truck -/
def front_axle_wheels : ℕ := 2

/-- The number of wheels on each of the other axles of the truck -/
def other_axle_wheels : ℕ := 4

/-- Theorem stating that a truck with the given wheel configuration has 18 wheels in total -/
theorem truck_wheels_count :
  ∀ (x : ℕ), 
  x > 0 →
  toll_formula x = 6 →
  front_axle_wheels + (x - 1) * other_axle_wheels = 18 :=
by
  sorry


end truck_wheels_count_l3954_395477


namespace quadratic_decreasing_implies_m_geq_3_l3954_395447

/-- A quadratic function of the form y = (x - m)^2 - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := (x - m)^2 - 1

/-- The function decreases as x increases when x ≤ 3 -/
def decreasing_for_x_leq_3 (m : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≤ x₂ → x₂ ≤ 3 → f m x₁ ≥ f m x₂

/-- If the quadratic function y = (x - m)^2 - 1 decreases as x increases when x ≤ 3,
    then m ≥ 3 -/
theorem quadratic_decreasing_implies_m_geq_3 (m : ℝ) :
  decreasing_for_x_leq_3 m → m ≥ 3 :=
by
  sorry

end quadratic_decreasing_implies_m_geq_3_l3954_395447


namespace integral_of_f_with_min_neg_one_l3954_395405

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

-- State the theorem
theorem integral_of_f_with_min_neg_one (m : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f m x ≤ f m y) ∧ 
  (∃ (x : ℝ), f m x = -1) →
  ∫ x in (1 : ℝ)..(2 : ℝ), f m x = 16/3 := by
  sorry

end integral_of_f_with_min_neg_one_l3954_395405


namespace triangle_area_l3954_395434

/-- The area of a triangle with base 15 cm and height 20 cm is 150 cm². -/
theorem triangle_area : 
  let base : ℝ := 15
  let height : ℝ := 20
  let area : ℝ := (base * height) / 2
  area = 150 := by sorry

end triangle_area_l3954_395434


namespace triple_base_exponent_l3954_395416

theorem triple_base_exponent (a b x : ℝ) (h1 : b ≠ 0) : 
  (3 * a) ^ (3 * b) = a ^ b * x ^ b → x = 27 * a ^ 2 := by
  sorry

end triple_base_exponent_l3954_395416


namespace loom_weaving_rate_l3954_395413

/-- The rate at which an industrial loom weaves cloth, given the total amount of cloth woven and the time taken. -/
theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) (h : total_cloth = 26 ∧ total_time = 203.125) :
  total_cloth / total_time = 0.128 := by
sorry

end loom_weaving_rate_l3954_395413


namespace square_diagonal_length_l3954_395465

theorem square_diagonal_length (perimeter : ℝ) (diagonal : ℝ) : 
  perimeter = 40 → diagonal = 10 * Real.sqrt 2 := by
  sorry

end square_diagonal_length_l3954_395465


namespace prank_combinations_l3954_395470

/-- The number of people available for the prank on each day of the week -/
def available_people : Fin 5 → ℕ
  | 0 => 2  -- Monday
  | 1 => 3  -- Tuesday
  | 2 => 6  -- Wednesday
  | 3 => 4  -- Thursday
  | 4 => 3  -- Friday

/-- The total number of different combinations of people for the prank across the week -/
def total_combinations : ℕ := (List.range 5).map available_people |>.prod

theorem prank_combinations :
  total_combinations = 432 := by
  sorry

end prank_combinations_l3954_395470


namespace quadratic_equivalence_l3954_395483

/-- The original quadratic function -/
def original_quadratic (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The converted quadratic function -/
def converted_quadratic (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating the equivalence of the two quadratic functions -/
theorem quadratic_equivalence :
  ∀ x : ℝ, original_quadratic x = converted_quadratic x :=
by
  sorry

end quadratic_equivalence_l3954_395483


namespace purchase_with_discounts_l3954_395496

/-- Calculates the final cost of a purchase with specific discounts -/
theorem purchase_with_discounts 
  (initial_total : ℝ) 
  (discounted_item_price : ℝ) 
  (item_discount_rate : ℝ) 
  (total_discount_rate : ℝ) 
  (h1 : initial_total = 54)
  (h2 : discounted_item_price = 20)
  (h3 : item_discount_rate = 0.2)
  (h4 : total_discount_rate = 0.1) :
  initial_total - 
  (discounted_item_price * item_discount_rate) - 
  ((initial_total - (discounted_item_price * item_discount_rate)) * total_discount_rate) = 45 := by
  sorry

end purchase_with_discounts_l3954_395496


namespace triangle_area_maximized_l3954_395411

theorem triangle_area_maximized (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  a / (Real.sin A) = c / (Real.sin C) ∧
  Real.tan A = 2 * Real.tan B ∧
  b = Real.sqrt 2 →
  (∀ A' B' C' a' b' c' : ℝ,
    0 < A' ∧ 0 < B' ∧ 0 < C' ∧
    A' + B' + C' = π ∧
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    a' / (Real.sin A') = b' / (Real.sin B') ∧
    a' / (Real.sin A') = c' / (Real.sin C') ∧
    Real.tan A' = 2 * Real.tan B' ∧
    b' = Real.sqrt 2 →
    1/2 * a * b * Real.sin C ≥ 1/2 * a' * b' * Real.sin C') →
  a = Real.sqrt 5 :=
sorry

end triangle_area_maximized_l3954_395411


namespace reflection_line_equation_l3954_395462

-- Define the points
def P : ℝ × ℝ := (3, 4)
def Q : ℝ × ℝ := (8, 9)
def R : ℝ × ℝ := (-5, 7)
def P' : ℝ × ℝ := (3, -6)
def Q' : ℝ × ℝ := (8, -11)
def R' : ℝ × ℝ := (-5, -9)

-- Define the line of reflection
def M : ℝ → ℝ := λ x => -1

-- Theorem statement
theorem reflection_line_equation :
  (∀ x y, M x = y ↔ y = -1) ∧
  (P.1 = P'.1 ∧ P.2 + P'.2 = 2 * (M P.1)) ∧
  (Q.1 = Q'.1 ∧ Q.2 + Q'.2 = 2 * (M Q.1)) ∧
  (R.1 = R'.1 ∧ R.2 + R'.2 = 2 * (M R.1)) :=
by sorry

end reflection_line_equation_l3954_395462


namespace dice_probability_l3954_395486

theorem dice_probability : 
  let n : ℕ := 5  -- number of dice
  let s : ℕ := 12  -- number of sides on each die
  let p_one_digit : ℚ := 3 / 4  -- probability of rolling a one-digit number
  let p_two_digit : ℚ := 1 / 4  -- probability of rolling a two-digit number
  Nat.choose n (n / 2) * p_two_digit ^ (n / 2) * p_one_digit ^ (n - n / 2) = 135 / 512 :=
by sorry

end dice_probability_l3954_395486


namespace quadrant_I_solution_l3954_395487

theorem quadrant_I_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 5 ∧ c * x + y = 7 ∧ x > 3 ∧ y > 1) ↔ c < 1 := by
  sorry

end quadrant_I_solution_l3954_395487


namespace average_age_after_leaving_l3954_395489

theorem average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age : ℕ) (final_people : ℕ) (final_average : ℚ) : 
  initial_people = 7 →
  initial_average = 28 →
  leaving_age = 20 →
  final_people = 6 →
  final_average = 29 →
  (initial_people : ℚ) * initial_average - leaving_age = final_people * final_average := by
  sorry

#check average_age_after_leaving

end average_age_after_leaving_l3954_395489


namespace b_value_l3954_395417

-- Define the functions p and q
def p (x : ℝ) : ℝ := 2 * x - 7
def q (x b : ℝ) : ℝ := 3 * x - b

-- State the theorem
theorem b_value (b : ℝ) : p (q 3 b) = 3 → b = 4 := by
  sorry

end b_value_l3954_395417


namespace triangle_area_l3954_395454

/-- The area of a triangle with base 10 cm and height 3 cm is 15 cm² -/
theorem triangle_area : 
  let base : ℝ := 10
  let height : ℝ := 3
  let area : ℝ := (1/2) * base * height
  area = 15 := by sorry

end triangle_area_l3954_395454


namespace sqrt_three_square_form_l3954_395415

theorem sqrt_three_square_form (a b m n : ℕ+) :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 →
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n :=
by sorry

end sqrt_three_square_form_l3954_395415


namespace negation_of_universal_is_existential_l3954_395428

def A : Set ℤ := {x | ∃ k, x = 2*k + 1}
def B : Set ℤ := {x | ∃ k, x = 2*k}

theorem negation_of_universal_is_existential :
  ¬(∀ x ∈ A, 2*x ∈ B) ↔ ∃ x ∈ A, 2*x ∉ B :=
sorry

end negation_of_universal_is_existential_l3954_395428


namespace f_properties_l3954_395478

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.log x - a

theorem f_properties (a : ℝ) (h : a > 0) :
  -- Part 1
  (∃ (x : ℝ), x > 0 ∧ f a x = 0 ∧ ∀ (y : ℝ), y > 0 → f a y ≥ f a x) ∧
  (¬∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f a y ≤ f a x) ∧
  -- Part 2
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 →
    1 / a < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < a) ∧
  -- Part 3
  (∀ (x : ℝ), x > 0 → Real.exp (2 * x - 2) - Real.exp (x - 1) * Real.log x - x ≥ 0) :=
by sorry


end f_properties_l3954_395478


namespace divisor_count_problem_l3954_395455

theorem divisor_count_problem (n : ℕ+) :
  (∃ (d : ℕ → ℕ), d (110 * n ^ 3) = 110) →
  (∃ (d : ℕ → ℕ), d (81 * n ^ 4) = 325) :=
by sorry

end divisor_count_problem_l3954_395455


namespace aldens_nephews_l3954_395412

theorem aldens_nephews (alden_now alden_past vihaan : ℕ) : 
  alden_now = 2 * alden_past →
  vihaan = alden_now + 60 →
  alden_now + vihaan = 260 →
  alden_past = 50 := by
sorry

end aldens_nephews_l3954_395412


namespace periodic_function_property_l3954_395429

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β), if f(4) = 3, then f(2017) = -3 -/
theorem periodic_function_property (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 4 = 3 → f 2017 = -3 := by
  sorry


end periodic_function_property_l3954_395429


namespace power_of_product_l3954_395420

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end power_of_product_l3954_395420


namespace fraction_to_decimal_l3954_395403

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end fraction_to_decimal_l3954_395403


namespace solve_for_a_l3954_395456

theorem solve_for_a : ∃ a : ℝ, (2 * (3 - 1) - a = 0) ∧ a = 4 := by sorry

end solve_for_a_l3954_395456


namespace max_ab_value_l3954_395480

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := x^2 - 2*x + 2
def g (a b x : ℝ) : ℝ := -x^2 + a*x + b

-- State the theorem
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x₀ : ℝ, f x₀ = g a b x₀ ∧ 
    (2*x₀ - 2) * (-2*x₀ + a) = -1) →
  ab ≤ 25/16 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀*b₀ = 25/16 :=
by
  sorry

end max_ab_value_l3954_395480


namespace intersection_x_coordinate_l3954_395427

theorem intersection_x_coordinate : 
  let line1 : ℝ → ℝ := λ x => 3 * x + 5
  let line2 : ℝ → ℝ := λ x => 35 - 5 * x
  ∃ x : ℝ, x = 15 / 4 ∧ line1 x = line2 x :=
by sorry

end intersection_x_coordinate_l3954_395427


namespace max_y_proof_unique_x_exists_no_greater_y_l3954_395482

/-- The maximum value of y such that there exists a unique x satisfying the given inequality -/
def max_y : ℕ := 112

theorem max_y_proof :
  ∀ y : ℕ, y > max_y →
    ¬(∃! x : ℕ, (9:ℚ)/17 < (x:ℚ)/(x + y) ∧ (x:ℚ)/(x + y) < 8/15) :=
by sorry

theorem unique_x_exists :
  ∃! x : ℕ, (9:ℚ)/17 < (x:ℚ)/(x + max_y) ∧ (x:ℚ)/(x + max_y) < 8/15 :=
by sorry

theorem no_greater_y :
  ∀ y : ℕ, y > max_y →
    ¬(∃! x : ℕ, (9:ℚ)/17 < (x:ℚ)/(x + y) ∧ (x:ℚ)/(x + y) < 8/15) :=
by sorry

end max_y_proof_unique_x_exists_no_greater_y_l3954_395482


namespace white_paint_calculation_l3954_395450

/-- Given the total amount of paint and the amounts of green and brown paint,
    calculate the amount of white paint needed. -/
theorem white_paint_calculation (total green brown : ℕ) (h1 : total = 69) 
    (h2 : green = 15) (h3 : brown = 34) : total - (green + brown) = 20 := by
  sorry

end white_paint_calculation_l3954_395450


namespace greatest_divisor_3815_4521_l3954_395453

def is_greatest_divisor (d n1 n2 r1 r2 : ℕ) : Prop :=
  d > 0 ∧
  n1 % d = r1 ∧
  n2 % d = r2 ∧
  ∀ k : ℕ, k > d → (n1 % k ≠ r1 ∨ n2 % k ≠ r2)

theorem greatest_divisor_3815_4521 :
  is_greatest_divisor 64 3815 4521 31 33 := by sorry

end greatest_divisor_3815_4521_l3954_395453


namespace data_set_average_l3954_395439

theorem data_set_average (x : ℝ) : 
  (2 + 3 + 4 + x + 6) / 5 = 4 → x = 5 := by
sorry

end data_set_average_l3954_395439


namespace factor_difference_of_squares_l3954_395457

theorem factor_difference_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4*y) * (5 + 4*y) := by
  sorry

end factor_difference_of_squares_l3954_395457


namespace existence_of_special_sequences_l3954_395492

theorem existence_of_special_sequences : ∃ (a b : ℕ → ℕ), 
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, b n < b (n + 1)) ∧ 
  (a 1 = 25) ∧ 
  (b 1 = 57) ∧ 
  (∀ n, (b n)^2 + 1 ≡ 0 [MOD (a n) * ((a n) + 1)]) := by
  sorry

end existence_of_special_sequences_l3954_395492


namespace sector_central_angle_l3954_395425

theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = (2/5) * Real.pi) :
  (2 * area) / (r^2) = Real.pi / 5 := by
  sorry

end sector_central_angle_l3954_395425


namespace ninth_term_of_arithmetic_sequence_l3954_395402

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + m) - a n = m * (a (n + 1) - a n)

theorem ninth_term_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 3/8)
  (h_seventeenth : a 17 = 2/3) :
  a 9 = 25/48 := by
sorry

end ninth_term_of_arithmetic_sequence_l3954_395402


namespace car_average_speed_l3954_395490

/-- The average speed of a car given its distance traveled in two hours -/
theorem car_average_speed (d1 d2 : ℝ) (h1 : d1 = 98) (h2 : d2 = 60) :
  (d1 + d2) / 2 = 79 := by
  sorry

end car_average_speed_l3954_395490
