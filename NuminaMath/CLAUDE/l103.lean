import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l103_10379

theorem expression_evaluation : 86 + (144 / 12) + (15 * 13) - 300 - (480 / 8) = -67 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l103_10379


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l103_10387

/-- Represents the repeating decimal 4.565656... -/
def repeating_decimal : ℚ := 4 + 56 / 99

/-- The fraction representation of 4.565656... -/
def fraction : ℚ := 452 / 99

theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l103_10387


namespace NUMINAMATH_CALUDE_foreign_stamps_count_l103_10375

theorem foreign_stamps_count (total : ℕ) (old : ℕ) (foreign_and_old : ℕ) (neither : ℕ) :
  total = 200 →
  old = 60 →
  foreign_and_old = 20 →
  neither = 70 →
  ∃ foreign : ℕ, foreign = 90 ∧ 
    foreign + old - foreign_and_old = total - neither :=
by sorry

end NUMINAMATH_CALUDE_foreign_stamps_count_l103_10375


namespace NUMINAMATH_CALUDE_blue_beads_count_l103_10310

theorem blue_beads_count (total_beads blue_beads yellow_beads : ℕ) : 
  yellow_beads = 16 →
  total_beads = blue_beads + yellow_beads →
  total_beads % 3 = 0 →
  (total_beads / 3 - 10) * 2 = 6 →
  blue_beads = 23 := by
sorry

end NUMINAMATH_CALUDE_blue_beads_count_l103_10310


namespace NUMINAMATH_CALUDE_postage_cost_theorem_l103_10329

/-- The floor function, representing the greatest integer less than or equal to x -/
def floor (x : ℝ) : ℤ := sorry

/-- The cost in cents for mailing a letter weighing W ounces -/
def postageCost (W : ℝ) : ℤ := sorry

theorem postage_cost_theorem (W : ℝ) : 
  postageCost W = -6 * floor (-W) :=
sorry

end NUMINAMATH_CALUDE_postage_cost_theorem_l103_10329


namespace NUMINAMATH_CALUDE_complex_division_sum_l103_10367

theorem complex_division_sum (a b : ℝ) : 
  (Complex.I - 2) / (1 + Complex.I) = Complex.ofReal a + Complex.I * Complex.ofReal b → 
  a + b = 1 := by sorry

end NUMINAMATH_CALUDE_complex_division_sum_l103_10367


namespace NUMINAMATH_CALUDE_mary_saturday_wage_l103_10355

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  weekday_hours : Nat
  saturday_hours : Nat
  regular_weekly_earnings : Nat
  saturday_weekly_earnings : Nat

/-- Calculates Mary's Saturday hourly wage --/
def saturday_hourly_wage (schedule : WorkSchedule) : Rat :=
  let regular_hourly_wage := schedule.regular_weekly_earnings / schedule.weekday_hours
  let saturday_earnings := schedule.saturday_weekly_earnings - schedule.regular_weekly_earnings
  saturday_earnings / schedule.saturday_hours

/-- Mary's actual work schedule --/
def mary_schedule : WorkSchedule :=
  { weekday_hours := 37
  , saturday_hours := 4
  , regular_weekly_earnings := 407
  , saturday_weekly_earnings := 483 }

/-- Theorem stating that Mary's Saturday hourly wage is $19 --/
theorem mary_saturday_wage :
  saturday_hourly_wage mary_schedule = 19 := by
  sorry

end NUMINAMATH_CALUDE_mary_saturday_wage_l103_10355


namespace NUMINAMATH_CALUDE_hamburger_combinations_eq_1024_l103_10336

/-- Represents the number of condiments available. -/
def num_condiments : ℕ := 8

/-- Represents the number of patty options available. -/
def num_patty_options : ℕ := 4

/-- Calculates the total number of hamburger combinations. -/
def total_hamburger_combinations : ℕ := num_patty_options * 2^num_condiments

/-- Proves that the total number of hamburger combinations is 1024. -/
theorem hamburger_combinations_eq_1024 : total_hamburger_combinations = 1024 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_eq_1024_l103_10336


namespace NUMINAMATH_CALUDE_no_intersection_and_constraint_l103_10344

theorem no_intersection_and_constraint (a b : ℝ) : 
  ¬(∃ (x : ℤ), a * (x : ℝ) + b = 3 * (x : ℝ)^2 + 15 ∧ a^2 + b^2 ≤ 144) :=
sorry

end NUMINAMATH_CALUDE_no_intersection_and_constraint_l103_10344


namespace NUMINAMATH_CALUDE_mixed_number_multiplication_l103_10358

theorem mixed_number_multiplication (a b c d e f : ℚ) :
  a + b / c = -3 ∧ b / c = 3 / 4 ∧ d / e = 5 / 7 →
  (a + b / c) * (d / e) = (a - b / c) * (d / e) := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_multiplication_l103_10358


namespace NUMINAMATH_CALUDE_sum_and_divide_l103_10395

theorem sum_and_divide : (40 + 5) / 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_divide_l103_10395


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l103_10360

/-- The number of ways to distribute n students among k groups, with each student choosing exactly one group -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose m items from a set of n items -/
def choose (n : ℕ) (m : ℕ) : ℕ := sorry

theorem student_distribution_theorem :
  let total_students : ℕ := 4
  let total_groups : ℕ := 4
  let groups_to_fill : ℕ := 3
  distribute_students total_students groups_to_fill = 36 :=
by sorry

end NUMINAMATH_CALUDE_student_distribution_theorem_l103_10360


namespace NUMINAMATH_CALUDE_polygon_area_bound_l103_10385

/-- A polygon with n vertices -/
structure Polygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ

/-- The area of a polygon -/
def area (P : Polygon) : ℝ := sorry

/-- The length of a line segment between two points -/
def distance (a b : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Area of a polygon with constrained sides and diagonals -/
theorem polygon_area_bound (P : Polygon) 
  (h1 : ∀ (i j : Fin P.n), distance (P.vertices i) (P.vertices j) ≤ 1) : 
  area P < Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_polygon_area_bound_l103_10385


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l103_10319

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 26 ∧ x - y = 8 → x * y = 153 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l103_10319


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l103_10361

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (l m : Line) (α : Plane) : 
  perpendicular l α → parallel l m → perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l103_10361


namespace NUMINAMATH_CALUDE_intersection_M_N_l103_10391

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l103_10391


namespace NUMINAMATH_CALUDE_unique_number_property_l103_10369

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Reverse of a natural number -/
def reverseNum (n : ℕ) : ℕ := sorry

/-- Prime factors of a natural number -/
def primeFactors (n : ℕ) : List ℕ := sorry

/-- Remove zeros from a natural number -/
def removeZeros (n : ℕ) : ℕ := sorry

theorem unique_number_property : ∃! n : ℕ, 
  n > 0 ∧ 
  n = sumOfDigits n * reverseNum (sumOfDigits n) ∧ 
  n = removeZeros ((List.sum (List.map (λ x => x^2) (primeFactors n))) / 2) ∧
  n = 1729 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_property_l103_10369


namespace NUMINAMATH_CALUDE_sequence_property_l103_10325

theorem sequence_property (a : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ a n ≥ n :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l103_10325


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l103_10371

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l103_10371


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l103_10320

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 8*x - 15) → (∃ y : ℝ, y^2 = 8*y - 15 ∧ x + y = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l103_10320


namespace NUMINAMATH_CALUDE_periodic_decimal_to_fraction_l103_10397

theorem periodic_decimal_to_fraction :
  (0.02 : ℚ) = 2 / 99 →
  (2.06 : ℚ) = 68 / 33 := by
sorry

end NUMINAMATH_CALUDE_periodic_decimal_to_fraction_l103_10397


namespace NUMINAMATH_CALUDE_cubic_factorization_l103_10356

theorem cubic_factorization (a : ℝ) : a^3 - 9*a = a*(a+3)*(a-3) := by sorry

end NUMINAMATH_CALUDE_cubic_factorization_l103_10356


namespace NUMINAMATH_CALUDE_cosine_sum_special_case_l103_10362

theorem cosine_sum_special_case : 
  Real.cos (π/12) * Real.cos (π/6) - Real.sin (π/12) * Real.sin (π/6) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_special_case_l103_10362


namespace NUMINAMATH_CALUDE_matrix_sum_proof_l103_10372

theorem matrix_sum_proof :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, 3; -2, 1]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-1, 5; 8, -3]
  A + B = !![3, 8; 6, -2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_proof_l103_10372


namespace NUMINAMATH_CALUDE_problem_solution_l103_10345

theorem problem_solution (a b : ℝ) (h : Real.sqrt (a - 3) + abs (4 - b) = 0) :
  (a - b) ^ 2023 = -1 ∧ ∀ x n : ℝ, x > 0 → Real.sqrt x = a + n → Real.sqrt x = b - 2*n → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l103_10345


namespace NUMINAMATH_CALUDE_area_of_region_s_l103_10323

/-- A square with side length 4 -/
structure Square :=
  (side_length : ℝ)
  (is_four : side_length = 4)

/-- A region S in the square -/
structure Region (sq : Square) :=
  (area : ℝ)
  (in_square : area ≤ sq.side_length^2)
  (closer_to_vertex : area > 0)

/-- Theorem: The area of region S is 2 -/
theorem area_of_region_s (sq : Square) (S : Region sq) : S.area = 2 :=
sorry

end NUMINAMATH_CALUDE_area_of_region_s_l103_10323


namespace NUMINAMATH_CALUDE_nine_students_in_front_of_hoseok_l103_10365

/-- The number of students standing in front of Hoseok in a line of 20 students, 
    where 11 students are behind Yoongi and Hoseok is right behind Yoongi. -/
def studentsInFrontOfHoseok (totalStudents : Nat) (studentsBehinYoongi : Nat) : Nat :=
  totalStudents - studentsBehinYoongi

/-- Theorem stating that 9 students are in front of Hoseok given the conditions -/
theorem nine_students_in_front_of_hoseok :
  studentsInFrontOfHoseok 20 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_students_in_front_of_hoseok_l103_10365


namespace NUMINAMATH_CALUDE_shopping_money_theorem_l103_10300

theorem shopping_money_theorem (initial_money : ℚ) : 
  (initial_money - 3/7 * initial_money - 2/5 * initial_money - 1/4 * initial_money = 24) →
  (initial_money - 1/2 * initial_money - 1/3 * initial_money = 36) →
  (initial_money + initial_money) / 2 = 458.18 := by
sorry

end NUMINAMATH_CALUDE_shopping_money_theorem_l103_10300


namespace NUMINAMATH_CALUDE_inequality_proof_l103_10317

theorem inequality_proof (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 3) : 
  2 * Real.sqrt (x + Real.sqrt y) + 2 * Real.sqrt (y + Real.sqrt z) + 2 * Real.sqrt (z + Real.sqrt x) 
  ≤ Real.sqrt (8 + x - y) + Real.sqrt (8 + y - z) + Real.sqrt (8 + z - x) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l103_10317


namespace NUMINAMATH_CALUDE_remaining_money_after_buying_folders_l103_10302

def remaining_money (initial_amount : ℕ) (folder_cost : ℕ) : ℕ :=
  initial_amount - (initial_amount / folder_cost) * folder_cost

theorem remaining_money_after_buying_folders :
  remaining_money 19 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_after_buying_folders_l103_10302


namespace NUMINAMATH_CALUDE_problem_solution_l103_10341

theorem problem_solution (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l103_10341


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l103_10348

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l103_10348


namespace NUMINAMATH_CALUDE_cone_volume_with_inscribed_square_l103_10351

/-- The volume of a cone with a square inscribed in its base --/
theorem cone_volume_with_inscribed_square (a α : ℝ) (h_a : a > 0) (h_α : 0 < α ∧ α < π) :
  let r := a * Real.sqrt 2 / 2
  let h := a * Real.sqrt (Real.cos α) / (2 * Real.sin (α/2) ^ 2)
  π * r^2 * h / 3 = π * a^3 * Real.sqrt (Real.cos α) / (12 * Real.sin (α/2) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_with_inscribed_square_l103_10351


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_difference_l103_10350

theorem square_difference_of_sum_and_difference (x y : ℝ) 
  (h_sum : x + y = 20) (h_diff : x - y = 10) : x^2 - y^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_difference_l103_10350


namespace NUMINAMATH_CALUDE_train_length_proof_l103_10339

/-- Proves that a train with given speed crossing a bridge of known length in a specific time has a particular length -/
theorem train_length_proof (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) (train_length : ℝ) : 
  bridge_length = 320 →
  crossing_time = 40 →
  train_speed_kmh = 42.3 →
  train_length = 150 →
  (train_length + bridge_length) = (train_speed_kmh * 1000 / 3600) * crossing_time := by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l103_10339


namespace NUMINAMATH_CALUDE_walking_problem_solution_l103_10342

def walking_problem (total_distance : ℝ) (speed_R : ℝ) (speed_S_initial : ℝ) (speed_S_second : ℝ) : Prop :=
  ∃ (k : ℕ) (x : ℝ),
    -- The total distance is 76 miles
    total_distance = 76 ∧
    -- Speed of person at R is 4.5 mph
    speed_R = 4.5 ∧
    -- Initial speed of person at S is 3.25 mph
    speed_S_initial = 3.25 ∧
    -- Second hour speed of person at S is 3.75 mph
    speed_S_second = 3.75 ∧
    -- They meet after k hours (k is a natural number)
    k > 0 ∧
    -- Distance traveled by person from R
    speed_R * k + x = total_distance / 2 ∧
    -- Distance traveled by person from S (arithmetic sequence sum)
    k * (speed_S_initial + (speed_S_second - speed_S_initial) * (k - 1) / 2) - x = total_distance / 2 ∧
    -- x is the difference in distances, and it equals 4
    x = 4

theorem walking_problem_solution :
  walking_problem 76 4.5 3.25 3.75 :=
sorry

end NUMINAMATH_CALUDE_walking_problem_solution_l103_10342


namespace NUMINAMATH_CALUDE_quadratic_transformation_l103_10337

theorem quadratic_transformation (y m n : ℝ) : 
  (2 * y^2 - 2 = 4 * y) → 
  ((y - m)^2 = n) → 
  (m - n)^2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l103_10337


namespace NUMINAMATH_CALUDE_school_sports_probabilities_l103_10370

/-- Represents a school with boys and girls, some of whom like sports -/
structure School where
  girls : ℕ
  boys : ℕ
  boys_like_sports : ℕ
  girls_like_sports : ℕ
  boys_ratio : boys = 3 * girls / 2
  boys_sports_ratio : boys_like_sports = 2 * boys / 5
  girls_sports_ratio : girls_like_sports = girls / 5

/-- The probability that a randomly selected student likes sports -/
def prob_likes_sports (s : School) : ℚ :=
  (s.boys_like_sports + s.girls_like_sports : ℚ) / (s.boys + s.girls)

/-- The probability that a randomly selected student who likes sports is a boy -/
def prob_boy_given_sports (s : School) : ℚ :=
  (s.boys_like_sports : ℚ) / (s.boys_like_sports + s.girls_like_sports)

theorem school_sports_probabilities (s : School) :
  prob_likes_sports s = 8/25 ∧ prob_boy_given_sports s = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_school_sports_probabilities_l103_10370


namespace NUMINAMATH_CALUDE_shares_distribution_l103_10322

/-- Proves that given the conditions, the shares of A, B, C, D, and E are 50, 100, 300, 150, and 600 respectively. -/
theorem shares_distribution (total : ℝ) (a b c d e : ℝ) 
  (h_total : total = 1200)
  (h_ab : a = (1/2) * b)
  (h_bc : b = (1/3) * c)
  (h_cd : c = 2 * d)
  (h_de : d = (1/4) * e)
  (h_sum : a + b + c + d + e = total) :
  a = 50 ∧ b = 100 ∧ c = 300 ∧ d = 150 ∧ e = 600 := by
  sorry

#check shares_distribution

end NUMINAMATH_CALUDE_shares_distribution_l103_10322


namespace NUMINAMATH_CALUDE_train_stoppage_time_l103_10314

/-- Given a train with speeds excluding and including stoppages, 
    calculate the number of minutes the train stops per hour. -/
theorem train_stoppage_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 48 → speed_with_stops = 36 → 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 15 := by
  sorry

#check train_stoppage_time

end NUMINAMATH_CALUDE_train_stoppage_time_l103_10314


namespace NUMINAMATH_CALUDE_min_value_problem_max_value_problem_min_sum_problem_l103_10338

-- Problem 1
theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y ≥ 8 := by sorry

-- Problem 2
theorem max_value_problem (x : ℝ) (h : x < 3) :
  4/(x - 3) + x ≤ -1 := by sorry

-- Problem 3
theorem min_sum_problem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = 2) :
  n/m + 1/(2*n) ≥ 5/4 := by sorry

end NUMINAMATH_CALUDE_min_value_problem_max_value_problem_min_sum_problem_l103_10338


namespace NUMINAMATH_CALUDE_smallest_n_is_360_l103_10311

def is_smallest_n (n : ℕ) : Prop :=
  n > 0 ∧
  n^2 % 54 = 0 ∧
  n^3 % 1280 = 0 ∧
  ∀ m : ℕ, m > 0 ∧ m < n → (m^2 % 54 ≠ 0 ∨ m^3 % 1280 ≠ 0)

theorem smallest_n_is_360 : is_smallest_n 360 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_is_360_l103_10311


namespace NUMINAMATH_CALUDE_other_roots_form_new_equation_l103_10386

theorem other_roots_form_new_equation (a₁ a₂ a₃ : ℝ) :
  let eq1 := fun x => x^2 + a₁*x + a₂*a₃
  let eq2 := fun x => x^2 + a₂*x + a₁*a₃
  let eq3 := fun x => x^2 + a₃*x + a₁*a₂
  (∃! α, eq1 α = 0 ∧ eq2 α = 0) →
  ∃ β γ, eq1 β = 0 ∧ eq2 γ = 0 ∧ β ≠ γ ∧ eq3 β = 0 ∧ eq3 γ = 0 :=
by sorry


end NUMINAMATH_CALUDE_other_roots_form_new_equation_l103_10386


namespace NUMINAMATH_CALUDE_fermat_point_theorem_l103_10306

/-- Represents a line in a plane --/
structure Line where
  -- Define a line using two points it passes through
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Represents a triangle --/
structure Triangle where
  -- Define a triangle using its three vertices
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  vertex3 : ℝ × ℝ

/-- Get the orthocenter of a triangle --/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Get the circumcenter of a triangle --/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Get the perpendicular bisector of a line segment --/
def perpendicularBisector (p1 p2 : ℝ × ℝ) : Line := sorry

/-- Get the triangles formed by four lines --/
def getTriangles (l1 l2 l3 l4 : Line) : List Triangle := sorry

/-- Check if a point lies on a line --/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- The Fermat point theorem --/
theorem fermat_point_theorem (l1 l2 l3 l4 : Line) : 
  ∃! fermatPoint : ℝ × ℝ, 
    ∀ t ∈ getTriangles l1 l2 l3 l4, 
      pointOnLine fermatPoint (perpendicularBisector (orthocenter t) (circumcenter t)) := by
  sorry

end NUMINAMATH_CALUDE_fermat_point_theorem_l103_10306


namespace NUMINAMATH_CALUDE_angle_bisector_inequality_l103_10392

/-- A triangle with sides a, b, c and angle bisectors fa, fb, fc -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  fa : ℝ
  fb : ℝ
  fc : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_fa : fa > 0
  pos_fb : fb > 0
  pos_fc : fc > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The inequality holds for any triangle -/
theorem angle_bisector_inequality (t : Triangle) :
  1 / t.fa + 1 / t.fb + 1 / t.fc > 1 / t.a + 1 / t.b + 1 / t.c := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_inequality_l103_10392


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_triangle_perimeter_l103_10349

theorem smallest_whole_number_above_triangle_perimeter : ∀ s : ℝ,
  s > 0 →
  s + 8 > 25 →
  s + 25 > 8 →
  8 + 25 > s →
  (∃ n : ℕ, n = 67 ∧ ∀ m : ℕ, (m : ℝ) > 8 + 25 + s → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_triangle_perimeter_l103_10349


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l103_10328

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y) ∧ f = fun x ↦ x - 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l103_10328


namespace NUMINAMATH_CALUDE_daily_increase_calculation_l103_10332

def squats_sequence (initial : ℕ) (increase : ℕ) (day : ℕ) : ℕ :=
  initial + (day - 1) * increase

theorem daily_increase_calculation (initial : ℕ) (increase : ℕ) :
  initial = 30 →
  squats_sequence initial increase 4 = 45 →
  increase = 5 := by
  sorry

end NUMINAMATH_CALUDE_daily_increase_calculation_l103_10332


namespace NUMINAMATH_CALUDE_ln_gt_one_sufficient_not_necessary_for_x_gt_one_l103_10343

theorem ln_gt_one_sufficient_not_necessary_for_x_gt_one :
  (∃ x : ℝ, x > 1 ∧ ¬(Real.log x > 1)) ∧
  (∀ x : ℝ, Real.log x > 1 → x > 1) :=
sorry

end NUMINAMATH_CALUDE_ln_gt_one_sufficient_not_necessary_for_x_gt_one_l103_10343


namespace NUMINAMATH_CALUDE_constant_d_value_l103_10398

theorem constant_d_value (e f d : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 2 * x + 4) * (e * x^2 + d * x + f) = 9 * x^4 - 8 * x^3 + 13 * x^2 + 12 * x - 16) →
  d = -2/3 := by
sorry

end NUMINAMATH_CALUDE_constant_d_value_l103_10398


namespace NUMINAMATH_CALUDE_racing_track_circumference_difference_l103_10331

theorem racing_track_circumference_difference
  (r : ℝ)
  (inner_radius : ℝ)
  (outer_radius : ℝ)
  (track_width : ℝ)
  (h1 : inner_radius = 2 * r)
  (h2 : outer_radius = inner_radius + track_width)
  (h3 : track_width = 15)
  : 2 * Real.pi * outer_radius - 2 * Real.pi * inner_radius = 30 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_racing_track_circumference_difference_l103_10331


namespace NUMINAMATH_CALUDE_inequality_proof_l103_10318

theorem inequality_proof (x y z : ℝ) (h1 : 0 < z) (h2 : z < y) (h3 : y < x) (h4 : x < π/2) :
  (π/2) + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z >
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l103_10318


namespace NUMINAMATH_CALUDE_team_incorrect_answers_is_17_l103_10335

/-- Represents a team in a math contest -/
structure MathTeam where
  total_questions : Nat
  riley_mistakes : Nat
  ofelia_correct_offset : Nat

/-- Calculates the total number of incorrect answers for a math team -/
def total_incorrect_answers (team : MathTeam) : Nat :=
  let riley_correct := team.total_questions - team.riley_mistakes
  let ofelia_correct := riley_correct / 2 + team.ofelia_correct_offset
  let ofelia_incorrect := team.total_questions - ofelia_correct
  team.riley_mistakes + ofelia_incorrect

/-- Theorem stating that for the given conditions, the team got 17 incorrect answers -/
theorem team_incorrect_answers_is_17 :
  ∃ (team : MathTeam),
    team.total_questions = 35 ∧
    team.riley_mistakes = 3 ∧
    team.ofelia_correct_offset = 5 ∧
    total_incorrect_answers team = 17 := by
  sorry

end NUMINAMATH_CALUDE_team_incorrect_answers_is_17_l103_10335


namespace NUMINAMATH_CALUDE_vector_parallel_implies_m_l103_10368

def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (-3, 1)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_parallel_implies_m (m : ℝ) :
  parallel ((2 * vector_a m).1 + vector_b.1, (2 * vector_a m).2 + vector_b.2) vector_b →
  m = -1/3 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_m_l103_10368


namespace NUMINAMATH_CALUDE_average_age_proof_l103_10394

/-- Given three people a, b, and c, prove that if their average age is 25 years
    and b's age is 17 years, then the average age of a and c is 29 years. -/
theorem average_age_proof (a b c : ℕ) : 
  (a + b + c) / 3 = 25 → b = 17 → (a + c) / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_age_proof_l103_10394


namespace NUMINAMATH_CALUDE_factorial_equality_l103_10384

theorem factorial_equality : 5 * 8 * 2 * 6 * 756 = Nat.factorial 9 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equality_l103_10384


namespace NUMINAMATH_CALUDE_N2O3_molecular_weight_l103_10374

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of N2O3 in atomic mass units (amu) -/
def N2O3_weight : ℝ := nitrogen_weight * nitrogen_count + oxygen_weight * oxygen_count

theorem N2O3_molecular_weight : N2O3_weight = 76.02 := by sorry

end NUMINAMATH_CALUDE_N2O3_molecular_weight_l103_10374


namespace NUMINAMATH_CALUDE_warden_citations_l103_10352

theorem warden_citations (total : ℕ) (littering off_leash parking : ℕ) : 
  total = 24 ∧ 
  littering = off_leash ∧ 
  parking = 2 * (littering + off_leash) ∧ 
  total = littering + off_leash + parking →
  littering = 4 := by
sorry

end NUMINAMATH_CALUDE_warden_citations_l103_10352


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l103_10378

/-- A quadratic function satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : ∀ x, a*(x-1)^2 + b*(x-1) = a*(3-x)^2 + b*(3-x)
  h3 : ∃! x, a*x^2 + b*x = 2*x

/-- The main theorem about the quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  (∀ x, f.a*x^2 + f.b*x = -x^2 + 2*x) ∧
  ∃ m n, m < n ∧
    (∀ x, f.a*x^2 + f.b*x ∈ Set.Icc m n ↔ x ∈ Set.Icc (-1) 0) ∧
    (∀ y, y ∈ Set.Icc (4*(-1)) (4*0) ↔ ∃ x, f.a*x^2 + f.b*x = y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l103_10378


namespace NUMINAMATH_CALUDE_sum_always_positive_l103_10380

/-- A monotonically increasing odd function -/
def MonoIncreasingOdd (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h1 : MonoIncreasingOdd f)
  (h2 : ArithmeticSequence a)
  (h3 : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_always_positive_l103_10380


namespace NUMINAMATH_CALUDE_equation_has_four_real_solutions_l103_10388

theorem equation_has_four_real_solutions :
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, (5 * x) / (x^2 + 2*x + 4) + (7 * x) / (x^2 - 7*x + 4) = -2) ∧ 
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_four_real_solutions_l103_10388


namespace NUMINAMATH_CALUDE_inequality_equivalence_l103_10354

theorem inequality_equivalence (x : ℝ) :
  (x + 1) * (1 / x - 1) > 0 ↔ x ∈ Set.Ioi (-1) ∪ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l103_10354


namespace NUMINAMATH_CALUDE_union_definition_l103_10312

theorem union_definition (A B : Set α) : 
  A ∪ B = {x | x ∈ A ∨ x ∈ B} := by
  sorry

end NUMINAMATH_CALUDE_union_definition_l103_10312


namespace NUMINAMATH_CALUDE_pretzel_ratio_is_three_to_one_l103_10324

/-- The number of pretzels Barry bought -/
def barry_pretzels : ℕ := 12

/-- The number of pretzels Angie bought -/
def angie_pretzels : ℕ := 18

/-- The number of pretzels Shelly bought -/
def shelly_pretzels : ℕ := barry_pretzels / 2

/-- The ratio of pretzels Angie bought to pretzels Shelly bought -/
def pretzel_ratio : ℚ := angie_pretzels / shelly_pretzels

theorem pretzel_ratio_is_three_to_one :
  pretzel_ratio = 3 := by sorry

end NUMINAMATH_CALUDE_pretzel_ratio_is_three_to_one_l103_10324


namespace NUMINAMATH_CALUDE_plot_length_is_64_l103_10305

/-- Proves that the length of a rectangular plot is 64 meters given the specified conditions -/
theorem plot_length_is_64 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 28 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  cost_per_meter * perimeter = total_cost →
  length = 64 := by
sorry

end NUMINAMATH_CALUDE_plot_length_is_64_l103_10305


namespace NUMINAMATH_CALUDE_janet_stickers_l103_10364

theorem janet_stickers (S : ℕ) : 
  S > 2 ∧ 
  S % 5 = 2 ∧ 
  S % 11 = 2 ∧ 
  S % 13 = 2 ∧ 
  (∀ T : ℕ, T > 2 ∧ T % 5 = 2 ∧ T % 11 = 2 ∧ T % 13 = 2 → S ≤ T) → 
  S = 717 := by
sorry

end NUMINAMATH_CALUDE_janet_stickers_l103_10364


namespace NUMINAMATH_CALUDE_mortgage_repayment_duration_l103_10381

theorem mortgage_repayment_duration (a : ℝ) (r : ℝ) (S : ℝ) (h1 : a = 400) (h2 : r = 2) (h3 : S = 819200) :
  ∃ n : ℕ, n = 11 ∧ S = a * (1 - r^n) / (1 - r) ∧ ∀ m : ℕ, m < n → S > a * (1 - r^m) / (1 - r) :=
sorry

end NUMINAMATH_CALUDE_mortgage_repayment_duration_l103_10381


namespace NUMINAMATH_CALUDE_product_of_roots_l103_10307

theorem product_of_roots (t : ℝ) : 
  let equation := fun t : ℝ => 18 * t^2 + 45 * t - 500
  let product_of_roots := -500 / 18
  product_of_roots = -250 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l103_10307


namespace NUMINAMATH_CALUDE_jerseys_sold_is_two_l103_10363

/-- The profit made from selling one jersey -/
def profit_per_jersey : ℕ := 76

/-- The total profit made from selling jerseys during the game -/
def total_profit : ℕ := 152

/-- The number of jerseys sold during the game -/
def jerseys_sold : ℕ := total_profit / profit_per_jersey

theorem jerseys_sold_is_two : jerseys_sold = 2 := by sorry

end NUMINAMATH_CALUDE_jerseys_sold_is_two_l103_10363


namespace NUMINAMATH_CALUDE_solution_sets_equal_l103_10373

/-- A strictly increasing bijective function from R to R -/
def StrictlyIncreasingBijection (f : ℝ → ℝ) : Prop :=
  Function.Bijective f ∧ StrictMono f

/-- The solution set of x = f(x) -/
def SolutionSetP (f : ℝ → ℝ) : Set ℝ :=
  {x | x = f x}

/-- The solution set of x = f(f(x)) -/
def SolutionSetQ (f : ℝ → ℝ) : Set ℝ :=
  {x | x = f (f x)}

/-- Theorem: For a strictly increasing bijective function f from R to R,
    the solution set P of x = f(x) is equal to the solution set Q of x = f(f(x)) -/
theorem solution_sets_equal (f : ℝ → ℝ) (h : StrictlyIncreasingBijection f) :
  SolutionSetP f = SolutionSetQ f := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_equal_l103_10373


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l103_10353

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the initial conditions. -/
theorem man_son_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 18 →
    man_age = son_age + 20 →
    ∃ (k : ℕ), (man_age + 2) = k * (son_age + 2) →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

#check man_son_age_ratio

end NUMINAMATH_CALUDE_man_son_age_ratio_l103_10353


namespace NUMINAMATH_CALUDE_fundraiser_result_l103_10377

/-- Represents the fundraiser scenario with students bringing brownies, cookies, and donuts. -/
structure Fundraiser where
  brownie_students : ℕ
  brownies_per_student : ℕ
  cookie_students : ℕ
  cookies_per_student : ℕ
  donut_students : ℕ
  donuts_per_student : ℕ
  price_per_item : ℚ

/-- Calculates the total amount of money raised in the fundraiser. -/
def total_money_raised (f : Fundraiser) : ℚ :=
  ((f.brownie_students * f.brownies_per_student +
    f.cookie_students * f.cookies_per_student +
    f.donut_students * f.donuts_per_student) : ℚ) * f.price_per_item

/-- Theorem stating that the fundraiser with given conditions raises $2040.00. -/
theorem fundraiser_result : 
  let f : Fundraiser := {
    brownie_students := 30,
    brownies_per_student := 12,
    cookie_students := 20,
    cookies_per_student := 24,
    donut_students := 15,
    donuts_per_student := 12,
    price_per_item := 2
  }
  total_money_raised f = 2040 := by
  sorry


end NUMINAMATH_CALUDE_fundraiser_result_l103_10377


namespace NUMINAMATH_CALUDE_temperature_difference_l103_10346

/-- The temperature difference problem -/
theorem temperature_difference
  (morning_temp : ℝ)
  (noon_rise : ℝ)
  (night_drop : ℝ)
  (h_morning : morning_temp = 7)
  (h_noon_rise : noon_rise = 9)
  (h_night_drop : night_drop = 13)
  (h_highest : morning_temp + noon_rise = max morning_temp (morning_temp + noon_rise))
  (h_lowest : morning_temp + noon_rise - night_drop = min (morning_temp + noon_rise) (morning_temp + noon_rise - night_drop)) :
  (morning_temp + noon_rise) - (morning_temp + noon_rise - night_drop) = 13 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l103_10346


namespace NUMINAMATH_CALUDE_line_points_property_l103_10321

theorem line_points_property (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) :
  y₁ = -2 * x₁ + 3 →
  y₂ = -2 * x₂ + 3 →
  y₃ = -2 * x₃ + 3 →
  x₁ < x₂ →
  x₂ < x₃ →
  x₂ * x₃ < 0 →
  y₁ * y₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_points_property_l103_10321


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_perimeter_l103_10303

theorem isosceles_trapezoid_perimeter (a b h : ℝ) (p q : ℕ+) : 
  a = Real.log 3 →
  b = Real.log 192 →
  h = Real.log 16 →
  (∃ (perimeter : ℝ), perimeter = Real.log (2^(p:ℝ) * 3^(q:ℝ)) ∧
    perimeter = 2 * Real.sqrt (h^2 + ((b - a)/2)^2) + a + b) →
  p + q = 18 := by
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_perimeter_l103_10303


namespace NUMINAMATH_CALUDE_a_stops_implies_smudged_l103_10376

-- Define the set of ladies
inductive Lady : Type
  | A
  | B
  | C

-- Define the state of a lady's face
inductive FaceState : Type
  | Clean
  | Smudged

-- Define the laughing state of a lady
inductive LaughState : Type
  | Laughing
  | NotLaughing

-- Function to get the face state of a lady
def faceState : Lady → FaceState
  | Lady.A => FaceState.Smudged
  | Lady.B => FaceState.Smudged
  | Lady.C => FaceState.Smudged

-- Function to get the initial laugh state of a lady
def initialLaughState : Lady → LaughState
  | _ => LaughState.Laughing

-- Function to determine if a lady can see another lady's smudged face
def canSeeSmugedFace (observer viewer : Lady) : Prop :=
  observer ≠ viewer ∧ faceState viewer = FaceState.Smudged

-- Theorem: If A stops laughing, it implies A must have a smudged face
theorem a_stops_implies_smudged :
  (initialLaughState Lady.A = LaughState.Laughing) →
  (∃ (newLaughState : Lady → LaughState),
    newLaughState Lady.A = LaughState.NotLaughing ∧
    (∀ l : Lady, l ≠ Lady.A → newLaughState l = LaughState.Laughing)) →
  faceState Lady.A = FaceState.Smudged :=
by
  sorry


end NUMINAMATH_CALUDE_a_stops_implies_smudged_l103_10376


namespace NUMINAMATH_CALUDE_multiplication_problem_l103_10330

theorem multiplication_problem : 10 * (3/27) * 36 = 40 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l103_10330


namespace NUMINAMATH_CALUDE_log_difference_cube_l103_10347

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the theorem
theorem log_difference_cube (x y a : ℝ) (h : lg x - lg y = a) :
  lg ((x/2)^3) - lg ((y/2)^3) = 3*a := by
  sorry

end NUMINAMATH_CALUDE_log_difference_cube_l103_10347


namespace NUMINAMATH_CALUDE_street_number_painting_cost_l103_10390

/-- Calculates the sum of digits for a given range of numbers in an arithmetic sequence -/
def sumDigits (start : ℕ) (diff : ℕ) (count : ℕ) : ℕ :=
  sorry

/-- Calculates the total cost of painting house numbers on a street -/
def totalCost (eastStart eastDiff westStart westDiff houseCount : ℕ) : ℕ :=
  sorry

theorem street_number_painting_cost :
  totalCost 5 5 2 4 25 = 88 :=
sorry

end NUMINAMATH_CALUDE_street_number_painting_cost_l103_10390


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l103_10359

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 4 * r - 10 = 0 ∧ 2 * s^2 - 4 * s - 10 = 0 ∧
   ∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = r - 3 ∨ x = s - 3)) →
  c = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l103_10359


namespace NUMINAMATH_CALUDE_right_triangle_area_l103_10383

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  angle = 45 * π / 180 →
  (1 / 2) * (hypotenuse / Real.sqrt 2) * (hypotenuse / Real.sqrt 2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l103_10383


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_three_l103_10308

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The first line equation: ax + 3y + 4 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 4 = 0

/-- The second line equation: x + (a-2)y + a^2 - 5 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 2) * y + a^2 - 5 = 0

/-- Theorem: If the two lines are parallel, then a = 3 -/
theorem parallel_lines_a_equals_three :
  ∀ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) → a = 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_three_l103_10308


namespace NUMINAMATH_CALUDE_job_completion_time_l103_10366

/-- Given two workers A and B who can complete a job in 15 and 30 days respectively,
    prove that they worked together for 4 days if 0.6 of the job is left unfinished. -/
theorem job_completion_time 
  (rate_A : ℝ) (rate_B : ℝ) (days_worked : ℝ) (fraction_left : ℝ) :
  rate_A = 1 / 15 →
  rate_B = 1 / 30 →
  fraction_left = 0.6 →
  (rate_A + rate_B) * days_worked = 1 - fraction_left →
  days_worked = 4 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l103_10366


namespace NUMINAMATH_CALUDE_estimate_overweight_students_l103_10333

def sample_size : ℕ := 100
def total_population : ℕ := 2000
def frequencies : List ℝ := [0.04, 0.035, 0.015]

theorem estimate_overweight_students :
  let total_frequency := (List.sum frequencies) * (total_population / sample_size)
  let estimated_students := total_population * total_frequency
  estimated_students = 360 := by sorry

end NUMINAMATH_CALUDE_estimate_overweight_students_l103_10333


namespace NUMINAMATH_CALUDE_cube_side_ratio_l103_10301

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 36 → a / b = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l103_10301


namespace NUMINAMATH_CALUDE_souvenir_cost_problem_l103_10313

theorem souvenir_cost_problem (total_souvenirs : ℕ) (total_cost : ℚ) 
  (cheap_souvenirs : ℕ) (cheap_cost : ℚ) (expensive_souvenirs : ℕ) :
  total_souvenirs = 1000 →
  total_cost = 220 →
  cheap_souvenirs = 400 →
  cheap_cost = 1/4 →
  expensive_souvenirs = total_souvenirs - cheap_souvenirs →
  (total_cost - cheap_souvenirs * cheap_cost) / expensive_souvenirs = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_cost_problem_l103_10313


namespace NUMINAMATH_CALUDE_email_count_proof_l103_10393

/-- Calculates the total number of emails received in a month with changing email rates -/
def total_emails (days_in_month : ℕ) (initial_rate : ℕ) (new_rate : ℕ) : ℕ :=
  let half_month := days_in_month / 2
  let first_half := initial_rate * half_month
  let second_half := new_rate * half_month
  first_half + second_half

/-- Proves that given the specified conditions, the total number of emails is 675 -/
theorem email_count_proof :
  let days_in_month : ℕ := 30
  let initial_rate : ℕ := 20
  let new_rate : ℕ := 25
  total_emails days_in_month initial_rate new_rate = 675 := by
  sorry


end NUMINAMATH_CALUDE_email_count_proof_l103_10393


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l103_10304

/-- A geometric sequence with a_1 = 3 and a_3 = 12 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ a 3 = 12 ∧ ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 5 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l103_10304


namespace NUMINAMATH_CALUDE_total_selected_in_survey_l103_10334

/-- The number of residents aged 21 to 35 -/
def residents_21_35 : ℕ := 840

/-- The number of residents aged 36 to 50 -/
def residents_36_50 : ℕ := 700

/-- The number of residents aged 51 to 65 -/
def residents_51_65 : ℕ := 560

/-- The number of people selected from the 36 to 50 age group -/
def selected_36_50 : ℕ := 100

/-- The total number of residents -/
def total_residents : ℕ := residents_21_35 + residents_36_50 + residents_51_65

/-- The theorem stating the total number of people selected in the survey -/
theorem total_selected_in_survey : 
  (selected_36_50 : ℚ) * (total_residents : ℚ) / (residents_36_50 : ℚ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_selected_in_survey_l103_10334


namespace NUMINAMATH_CALUDE_fred_weekend_earnings_l103_10382

/-- Fred's earnings from delivering newspapers -/
def newspaper_earnings : ℕ := 16

/-- Fred's earnings from washing cars -/
def car_washing_earnings : ℕ := 74

/-- Fred's total earnings over the weekend -/
def total_earnings : ℕ := newspaper_earnings + car_washing_earnings

/-- Theorem stating that Fred's total earnings over the weekend equal $90 -/
theorem fred_weekend_earnings : total_earnings = 90 := by sorry

end NUMINAMATH_CALUDE_fred_weekend_earnings_l103_10382


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l103_10315

theorem polynomial_product_expansion (x : ℝ) : 
  (7 * x^2 + 5 * x - 3) * (3 * x^3 + 2 * x^2 - x + 4) = 
  21 * x^5 + 29 * x^4 - 6 * x^3 + 17 * x^2 + 23 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l103_10315


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l103_10327

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
by sorry

theorem lower_bound_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) = -17 + 12 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l103_10327


namespace NUMINAMATH_CALUDE_rectangle_length_l103_10316

theorem rectangle_length (square_side : ℝ) (rectangle_area : ℝ) : 
  square_side = 15 →
  rectangle_area = 216 →
  ∃ (rectangle_length rectangle_width : ℝ),
    4 * square_side = 2 * (rectangle_length + rectangle_width) ∧
    rectangle_length * rectangle_width = rectangle_area ∧
    rectangle_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l103_10316


namespace NUMINAMATH_CALUDE_specific_wall_rows_l103_10389

/-- Represents a brick wall with a specific structure -/
structure BrickWall where
  totalBricks : ℕ
  bottomRowBricks : ℕ
  (total_positive : 0 < totalBricks)
  (bottom_positive : 0 < bottomRowBricks)
  (bottom_leq_total : bottomRowBricks ≤ totalBricks)

/-- Calculates the number of rows in a brick wall -/
def numberOfRows (wall : BrickWall) : ℕ :=
  sorry

/-- Theorem stating that a wall with 100 total bricks and 18 bricks in the bottom row has 8 rows -/
theorem specific_wall_rows :
  ∀ (wall : BrickWall),
    wall.totalBricks = 100 →
    wall.bottomRowBricks = 18 →
    numberOfRows wall = 8 :=
  sorry

end NUMINAMATH_CALUDE_specific_wall_rows_l103_10389


namespace NUMINAMATH_CALUDE_range_of_f_range_of_a_l103_10340

-- Define the function f
def f (x : ℝ) : ℝ := 2 * |x - 1| - |x - 4|

-- Theorem for the range of f
theorem range_of_f : Set.range f = Set.Ici (-3) := by sorry

-- Define the inequality function g
def g (x a : ℝ) : ℝ := 2 * |x - 1| - |x - a|

-- Theorem for the range of a
theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, g x a ≥ -1) ↔ a ∈ Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_a_l103_10340


namespace NUMINAMATH_CALUDE_inequality_solution_l103_10399

theorem inequality_solution (x : ℝ) : 
  -1 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ 
  (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 1 ↔ 
  x < 5/2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l103_10399


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l103_10326

theorem quadratic_is_perfect_square : ∃ (a b : ℝ), ∀ x : ℝ, x^2 - 18*x + 81 = (a*x + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l103_10326


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l103_10396

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l103_10396


namespace NUMINAMATH_CALUDE_remaining_oranges_l103_10357

-- Define the initial number of oranges Mildred collects
def initial_oranges : ℝ := 77.0

-- Define the number of oranges Mildred's father eats
def eaten_oranges : ℝ := 2.0

-- Theorem stating the number of oranges Mildred has after her father eats some
theorem remaining_oranges : initial_oranges - eaten_oranges = 75.0 := by
  sorry

end NUMINAMATH_CALUDE_remaining_oranges_l103_10357


namespace NUMINAMATH_CALUDE_high_card_value_l103_10309

structure CardGame where
  total_cards : Nat
  high_cards : Nat
  low_cards : Nat
  high_value : Nat
  low_value : Nat
  target_points : Nat
  target_low_cards : Nat
  ways_to_earn : Nat

def is_valid_game (game : CardGame) : Prop :=
  game.total_cards = 52 ∧
  game.high_cards = game.low_cards ∧
  game.high_cards + game.low_cards = game.total_cards ∧
  game.low_value = 1 ∧
  game.target_points = 5 ∧
  game.target_low_cards = 3 ∧
  game.ways_to_earn = 4

theorem high_card_value (game : CardGame) :
  is_valid_game game → game.high_value = 2 := by
  sorry

end NUMINAMATH_CALUDE_high_card_value_l103_10309
