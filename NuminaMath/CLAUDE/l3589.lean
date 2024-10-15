import Mathlib

namespace NUMINAMATH_CALUDE_triangle_properties_l3589_358915

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.b * Real.sin abc.A = Real.sqrt 3 * abc.a * Real.cos abc.B)
  (h2 : abc.b = 3)
  (h3 : Real.sin abc.C = 2 * Real.sin abc.A) :
  abc.B = π / 3 ∧ 
  abc.a = Real.sqrt 3 ∧ 
  abc.c = 2 * Real.sqrt 3 ∧ 
  (1 / 2 : ℝ) * abc.a * abc.c * Real.sin abc.B = 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3589_358915


namespace NUMINAMATH_CALUDE_certain_number_proof_l3589_358903

theorem certain_number_proof : ∃ x : ℕ, (3 * 16) + (3 * 17) + (3 * 20) + x = 170 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3589_358903


namespace NUMINAMATH_CALUDE_daily_savings_amount_l3589_358939

def total_savings : ℝ := 8760
def days_in_year : ℕ := 365

theorem daily_savings_amount :
  total_savings / days_in_year = 24 := by
  sorry

end NUMINAMATH_CALUDE_daily_savings_amount_l3589_358939


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_1_quadratic_inequality_theorem_2_quadratic_inequality_theorem_3_l3589_358964

-- Define the quadratic inequality
def quadratic_inequality (k x : ℝ) : Prop := k * x^2 - 2 * x + 6 * k < 0

-- Define the solution sets
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2
def solution_set_2 (k x : ℝ) : Prop := x ≠ 1 / k
def solution_set_3 : Set ℝ := ∅

-- Theorem statements
theorem quadratic_inequality_theorem_1 (k : ℝ) :
  k ≠ 0 →
  (∀ x, quadratic_inequality k x ↔ solution_set_1 x) →
  k = -2/5 :=
sorry

theorem quadratic_inequality_theorem_2 (k : ℝ) :
  k ≠ 0 →
  (∀ x, quadratic_inequality k x ↔ solution_set_2 k x) →
  k = -Real.sqrt 6 / 6 :=
sorry

theorem quadratic_inequality_theorem_3 (k : ℝ) :
  k ≠ 0 →
  (∀ x, ¬quadratic_inequality k x) →
  k ≥ Real.sqrt 6 / 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_1_quadratic_inequality_theorem_2_quadratic_inequality_theorem_3_l3589_358964


namespace NUMINAMATH_CALUDE_hostel_mess_expenditure_l3589_358929

/-- Given a hostel with students and mess expenses, calculate the original expenditure --/
theorem hostel_mess_expenditure 
  (initial_students : ℕ) 
  (student_increase : ℕ) 
  (expense_increase : ℕ) 
  (avg_expense_decrease : ℕ) 
  (h1 : initial_students = 35)
  (h2 : student_increase = 7)
  (h3 : expense_increase = 42)
  (h4 : avg_expense_decrease = 1) :
  ∃ (original_expenditure : ℕ), 
    original_expenditure = initial_students * 
      ((initial_students + student_increase) * 
        (original_expenditure / initial_students - avg_expense_decrease) - 
      original_expenditure) / student_increase ∧
    original_expenditure = 420 :=
by sorry

end NUMINAMATH_CALUDE_hostel_mess_expenditure_l3589_358929


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3589_358948

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 6 + a 11 = 100) : 
  2 * a 7 - a 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3589_358948


namespace NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_open_interval_l3589_358914

theorem no_fixed_points_iff_a_in_open_interval
  (f : ℝ → ℝ)
  (h : ∀ x, f x = x^2 + a*x + 1) :
  (∀ x, f x ≠ x) ↔ a ∈ Set.Ioo (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_open_interval_l3589_358914


namespace NUMINAMATH_CALUDE_other_x_intercept_is_negative_one_l3589_358970

/-- A quadratic function with vertex (h, k) and one x-intercept at (r, 0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  k : ℝ
  r : ℝ
  vertex_x : h = 2
  vertex_y : k = -3
  intercept : r = 5

/-- The x-coordinate of the other x-intercept of the quadratic function -/
def other_x_intercept (f : QuadraticFunction) : ℝ := 2 * f.h - f.r

theorem other_x_intercept_is_negative_one (f : QuadraticFunction) :
  other_x_intercept f = -1 := by
  sorry

end NUMINAMATH_CALUDE_other_x_intercept_is_negative_one_l3589_358970


namespace NUMINAMATH_CALUDE_nested_fraction_value_l3589_358935

theorem nested_fraction_value : 1 + 2 / (3 + 4/5) = 29/19 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_value_l3589_358935


namespace NUMINAMATH_CALUDE_isabels_pop_albums_l3589_358941

theorem isabels_pop_albums (total_songs : ℕ) (country_albums : ℕ) (songs_per_album : ℕ) 
  (h1 : total_songs = 72)
  (h2 : country_albums = 4)
  (h3 : songs_per_album = 8) :
  total_songs - country_albums * songs_per_album = 5 * songs_per_album :=
by sorry

end NUMINAMATH_CALUDE_isabels_pop_albums_l3589_358941


namespace NUMINAMATH_CALUDE_clothing_store_loss_l3589_358902

/-- Proves that selling two sets of clothes at 168 yuan each, with one set having a 20% profit
    and the other having a 20% loss, results in a total loss of 14 yuan. -/
theorem clothing_store_loss (selling_price : ℝ) (profit_percentage : ℝ) (loss_percentage : ℝ) :
  selling_price = 168 →
  profit_percentage = 0.2 →
  loss_percentage = 0.2 →
  let profit_cost := selling_price / (1 + profit_percentage)
  let loss_cost := selling_price / (1 - loss_percentage)
  (2 * selling_price) - (profit_cost + loss_cost) = -14 := by
sorry

end NUMINAMATH_CALUDE_clothing_store_loss_l3589_358902


namespace NUMINAMATH_CALUDE_tangent_point_abscissa_l3589_358983

/-- The curve function f(x) = x³ - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 1

/-- The slope of the line perpendicular to x + 2y - 1 = 0 -/
def m : ℝ := 2

theorem tangent_point_abscissa :
  ∃ (x : ℝ), (f' x = m) ∧ (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_abscissa_l3589_358983


namespace NUMINAMATH_CALUDE_investment_amount_from_interest_difference_l3589_358927

/-- Proves that given two equal investments with specific interest rates and time period, 
    the investment amount can be determined from the interest difference. -/
theorem investment_amount_from_interest_difference 
  (P : ℝ) -- The amount invested (same for both investments)
  (r1 : ℝ) -- Interest rate for first investment
  (r2 : ℝ) -- Interest rate for second investment
  (t : ℝ) -- Time period in years
  (diff : ℝ) -- Difference in interest earned
  (h1 : r1 = 0.04) -- First interest rate is 4%
  (h2 : r2 = 0.045) -- Second interest rate is 4.5%
  (h3 : t = 7) -- Time period is 7 years
  (h4 : P * r2 * t - P * r1 * t = diff) -- Difference in interest equation
  (h5 : diff = 31.5) -- Interest difference is $31.50
  : P = 900 := by
  sorry

#check investment_amount_from_interest_difference

end NUMINAMATH_CALUDE_investment_amount_from_interest_difference_l3589_358927


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3589_358925

/-- The area of a rectangular field with one side 15 meters and diagonal 17 meters is 120 square meters. -/
theorem rectangular_field_area (side : ℝ) (diagonal : ℝ) (area : ℝ) : 
  side = 15 → diagonal = 17 → area = side * Real.sqrt (diagonal^2 - side^2) → area = 120 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3589_358925


namespace NUMINAMATH_CALUDE_tournament_games_count_l3589_358907

/-- Calculates the total number of games played in a tournament given the ratio of outcomes and the number of games won. -/
def totalGamesPlayed (ratioWon ratioLost ratioTied : ℕ) (gamesWon : ℕ) : ℕ :=
  let partValue := gamesWon / ratioWon
  let gamesLost := ratioLost * partValue
  let gamesTied := ratioTied * partValue
  gamesWon + gamesLost + gamesTied

/-- Theorem stating that given the specified ratio and number of games won, the total games played is 96. -/
theorem tournament_games_count :
  totalGamesPlayed 7 4 5 42 = 96 := by
  sorry

#eval totalGamesPlayed 7 4 5 42

end NUMINAMATH_CALUDE_tournament_games_count_l3589_358907


namespace NUMINAMATH_CALUDE_jeromes_contact_list_l3589_358981

theorem jeromes_contact_list (classmates : ℕ) (out_of_school_friends : ℕ) (family_members : ℕ) : 
  classmates = 20 →
  out_of_school_friends = classmates / 2 →
  family_members = 3 →
  classmates + out_of_school_friends + family_members = 33 := by
  sorry

end NUMINAMATH_CALUDE_jeromes_contact_list_l3589_358981


namespace NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l3589_358973

theorem max_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  2*x + y ≤ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 2/y₀ = 1 ∧ 2*x₀ + y₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l3589_358973


namespace NUMINAMATH_CALUDE_exist_three_quadratic_polynomials_l3589_358998

theorem exist_three_quadratic_polynomials :
  ∃ (f g h : ℝ → ℝ),
    (∀ x, f x = (x - 3)^2 - 1) ∧
    (∀ x, g x = x^2 - 1) ∧
    (∀ x, h x = (x + 3)^2 - 1) ∧
    (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
    (∃ y₁ y₂, y₁ ≠ y₂ ∧ g y₁ = 0 ∧ g y₂ = 0) ∧
    (∃ z₁ z₂, z₁ ≠ z₂ ∧ h z₁ = 0 ∧ h z₂ = 0) ∧
    (∀ x, (f x + g x) ≠ 0) ∧
    (∀ x, (f x + h x) ≠ 0) ∧
    (∀ x, (g x + h x) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_exist_three_quadratic_polynomials_l3589_358998


namespace NUMINAMATH_CALUDE_pool_water_proof_l3589_358951

def initial_volume : ℝ := 300
def evaporation_rate_1 : ℝ := 1
def evaporation_rate_2 : ℝ := 2
def days_1 : ℝ := 15
def days_2 : ℝ := 15
def total_days : ℝ := days_1 + days_2

def remaining_volume : ℝ :=
  initial_volume - (evaporation_rate_1 * days_1 + evaporation_rate_2 * days_2)

theorem pool_water_proof :
  remaining_volume = 255 := by sorry

end NUMINAMATH_CALUDE_pool_water_proof_l3589_358951


namespace NUMINAMATH_CALUDE_new_student_weight_l3589_358928

theorem new_student_weight (n : ℕ) (original_avg replaced_weight new_avg : ℝ) :
  n = 5 →
  replaced_weight = 72 →
  new_avg = original_avg - 12 →
  n * original_avg - replaced_weight = n * new_avg - (n * original_avg - n * new_avg) →
  n * original_avg - replaced_weight = 12 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l3589_358928


namespace NUMINAMATH_CALUDE_square_root_divided_by_three_l3589_358979

theorem square_root_divided_by_three : Real.sqrt 81 / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_divided_by_three_l3589_358979


namespace NUMINAMATH_CALUDE_cookery_club_committee_probability_l3589_358900

def total_members : ℕ := 24
def num_boys : ℕ := 14
def num_girls : ℕ := 10
def committee_size : ℕ := 5

theorem cookery_club_committee_probability :
  let total_committees := Nat.choose total_members committee_size
  let committees_with_fewer_than_two_girls := 
    Nat.choose num_boys committee_size + 
    (num_girls * Nat.choose num_boys (committee_size - 1))
  let committees_with_at_least_two_girls := 
    total_committees - committees_with_fewer_than_two_girls
  (committees_with_at_least_two_girls : ℚ) / total_committees = 2541 / 3542 := by
  sorry

end NUMINAMATH_CALUDE_cookery_club_committee_probability_l3589_358900


namespace NUMINAMATH_CALUDE_racetrack_probability_l3589_358922

/-- Represents a circular racetrack -/
structure Racetrack where
  length : ℝ
  isCircular : Bool

/-- Represents a car on the racetrack -/
structure Car where
  position : ℝ
  travelDistance : ℝ

/-- Calculates the probability of the car ending within the specified range -/
def probabilityOfEndingInRange (track : Racetrack) (car : Car) (targetPosition : ℝ) (range : ℝ) : ℝ :=
  sorry

theorem racetrack_probability (track : Racetrack) (car : Car) : 
  track.length = 3 →
  track.isCircular = true →
  car.travelDistance = 0.5 →
  probabilityOfEndingInRange track car 2.5 0.5 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_racetrack_probability_l3589_358922


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l3589_358956

theorem geometric_progression_problem (b₁ q : ℝ) 
  (h_decreasing : |q| < 1)
  (h_sum_diff : b₁ / (1 - q^2) - (b₁ * q) / (1 - q^2) = 10)
  (h_sum_squares_diff : b₁^2 / (1 - q^4) - (b₁^2 * q^2) / (1 - q^4) = 20) :
  b₁ = 5 ∧ q = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l3589_358956


namespace NUMINAMATH_CALUDE_binary_101101_eq_45_l3589_358953

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101101₂ -/
def binary_101101 : List Bool := [true, false, true, true, false, true]

/-- Theorem stating that the decimal equivalent of 101101₂ is 45 -/
theorem binary_101101_eq_45 : binary_to_decimal binary_101101 = 45 := by
  sorry

#eval binary_to_decimal binary_101101

end NUMINAMATH_CALUDE_binary_101101_eq_45_l3589_358953


namespace NUMINAMATH_CALUDE_garden_area_l3589_358920

theorem garden_area (length width : ℝ) (h1 : length = 350) (h2 : width = 50) :
  (length * width) / 10000 = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l3589_358920


namespace NUMINAMATH_CALUDE_cube_face_sum_theorem_l3589_358976

/-- Represents a cube with numbers on its faces -/
structure NumberedCube where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- Calculates the sum of vertex products for a NumberedCube -/
def vertexProductSum (cube : NumberedCube) : ℕ :=
  cube.a * cube.b * cube.c +
  cube.a * cube.e * cube.c +
  cube.a * cube.b * cube.f +
  cube.a * cube.e * cube.f +
  cube.d * cube.b * cube.c +
  cube.d * cube.e * cube.c +
  cube.d * cube.b * cube.f +
  cube.d * cube.e * cube.f

/-- Calculates the sum of face numbers for a NumberedCube -/
def faceSum (cube : NumberedCube) : ℕ :=
  cube.a + cube.b + cube.c + cube.d + cube.e + cube.f

/-- Theorem: If the sum of vertex products is 357, then the sum of face numbers is 27 -/
theorem cube_face_sum_theorem (cube : NumberedCube) :
  vertexProductSum cube = 357 → faceSum cube = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_sum_theorem_l3589_358976


namespace NUMINAMATH_CALUDE_integral_comparison_l3589_358952

theorem integral_comparison : ∃ (a b c : ℝ),
  (a = ∫ x in (0:ℝ)..1, x) ∧
  (b = ∫ x in (0:ℝ)..1, x^2) ∧
  (c = ∫ x in (0:ℝ)..1, Real.sqrt x) ∧
  (b < a ∧ a < c) :=
by sorry

end NUMINAMATH_CALUDE_integral_comparison_l3589_358952


namespace NUMINAMATH_CALUDE_vector_expression_l3589_358932

-- Define vectors a, b, and c
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, -2)

-- Theorem statement
theorem vector_expression :
  c = (-3/2 : ℝ) • a + (1/2 : ℝ) • b := by sorry

end NUMINAMATH_CALUDE_vector_expression_l3589_358932


namespace NUMINAMATH_CALUDE_pet_shelter_problem_l3589_358977

theorem pet_shelter_problem (total dogs_watermelon dogs_salmon dogs_chicken 
  dogs_watermelon_salmon dogs_salmon_chicken dogs_watermelon_chicken dogs_all_three : ℕ) 
  (h_total : total = 150)
  (h_watermelon : dogs_watermelon = 30)
  (h_salmon : dogs_salmon = 70)
  (h_chicken : dogs_chicken = 15)
  (h_watermelon_salmon : dogs_watermelon_salmon = 10)
  (h_salmon_chicken : dogs_salmon_chicken = 7)
  (h_watermelon_chicken : dogs_watermelon_chicken = 5)
  (h_all_three : dogs_all_three = 3) :
  total - (dogs_watermelon + dogs_salmon + dogs_chicken 
    - dogs_watermelon_salmon - dogs_salmon_chicken - dogs_watermelon_chicken 
    + dogs_all_three) = 54 := by
  sorry


end NUMINAMATH_CALUDE_pet_shelter_problem_l3589_358977


namespace NUMINAMATH_CALUDE_tom_payment_l3589_358991

/-- The amount Tom paid to the shopkeeper for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Theorem stating that Tom paid 1145 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 65 = 1145 := by
  sorry

#eval total_amount 8 70 9 65

end NUMINAMATH_CALUDE_tom_payment_l3589_358991


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l3589_358978

/-- A piecewise function f defined on real numbers -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then x^2 - 2*x else a*x - 1

/-- Proposition: If f is monotonically increasing on ℝ, then 0 < a ≤ 1/2 -/
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (0 < a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l3589_358978


namespace NUMINAMATH_CALUDE_joined_right_triangles_fourth_square_l3589_358962

theorem joined_right_triangles_fourth_square 
  (PQ QR RS : ℝ) 
  (h1 : PQ^2 = 25) 
  (h2 : QR^2 = 49) 
  (h3 : RS^2 = 64) 
  (h4 : PQ > 0 ∧ QR > 0 ∧ RS > 0) : 
  (PQ^2 + QR^2) + RS^2 = 138 := by
  sorry

end NUMINAMATH_CALUDE_joined_right_triangles_fourth_square_l3589_358962


namespace NUMINAMATH_CALUDE_ink_remaining_proof_l3589_358923

/-- The total area a full marker can cover, in square inches -/
def full_marker_coverage : ℝ := 48

/-- The area covered by the rectangles, in square inches -/
def area_covered : ℝ := 24

/-- The percentage of ink remaining after covering the rectangles -/
def ink_remaining_percentage : ℝ := 50

theorem ink_remaining_proof :
  (full_marker_coverage - area_covered) / full_marker_coverage * 100 = ink_remaining_percentage :=
by sorry

end NUMINAMATH_CALUDE_ink_remaining_proof_l3589_358923


namespace NUMINAMATH_CALUDE_divisibility_constraint_l3589_358963

theorem divisibility_constraint (m n : ℕ) : 
  m ≥ 1 → n ≥ 1 → 
  (m * n ∣ 3^m + 1) → 
  (m * n ∣ 3^n + 1) → 
  ((m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_constraint_l3589_358963


namespace NUMINAMATH_CALUDE_sixth_rack_dvds_l3589_358949

def dvd_sequence : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * dvd_sequence n

theorem sixth_rack_dvds : dvd_sequence 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_sixth_rack_dvds_l3589_358949


namespace NUMINAMATH_CALUDE_log_sum_theorem_l3589_358975

theorem log_sum_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : y = 2016 * x) (h2 : x^y = y^x) : 
  Real.log x / Real.log 2016 + Real.log y / Real.log 2016 = 2017 / 2015 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_theorem_l3589_358975


namespace NUMINAMATH_CALUDE_right_triangle_third_vertex_l3589_358958

theorem right_triangle_third_vertex 
  (v1 : ℝ × ℝ) 
  (v2 : ℝ × ℝ) 
  (x : ℝ) :
  v1 = (4, 3) →
  v2 = (0, 0) →
  x > 0 →
  (1/2 : ℝ) * x * 3 = 24 →
  x = 16 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_vertex_l3589_358958


namespace NUMINAMATH_CALUDE_trajectory_and_slope_product_l3589_358960

-- Define the points and the trajectory
def A : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (1, 2)
def P : ℝ × ℝ := (0, -2)

def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ^ 2 = 4 * p.1 ∧ p.2 ≠ 0}

-- Define the conditions
structure Triangle (A B C : ℝ × ℝ) : Prop where
  b_on_x_axis : B.2 = 0
  equal_sides : (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2
  midpoint_on_y : B.1 + C.1 = 0

-- Define the theorem
theorem trajectory_and_slope_product 
  (B C : ℝ × ℝ) 
  (h : Triangle A B C) 
  (hC : C ∈ Γ) 
  (l : Set (ℝ × ℝ)) 
  (hl : P ∈ l) 
  (M N : ℝ × ℝ) 
  (hM : M ∈ l ∩ Γ) 
  (hN : N ∈ l ∩ Γ) 
  (hMN : M ≠ N) :
  -- Part I: C satisfies the equation of Γ
  C.2 ^ 2 = 4 * C.1 ∧ 
  -- Part II: Product of slopes is constant
  (M.2 - Q.2) / (M.1 - Q.1) * (N.2 - Q.2) / (N.1 - Q.1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_and_slope_product_l3589_358960


namespace NUMINAMATH_CALUDE_movie_ticket_change_l3589_358931

/-- Represents the movie ticket formats --/
inductive TicketFormat
  | Regular
  | ThreeD
  | IMAX

/-- Returns the price of a ticket based on its format --/
def ticketPrice (format : TicketFormat) : ℝ :=
  match format with
  | TicketFormat.Regular => 8
  | TicketFormat.ThreeD => 12
  | TicketFormat.IMAX => 15

/-- Calculates the discounted price of a ticket --/
def discountedPrice (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

theorem movie_ticket_change : 
  let format := TicketFormat.ThreeD
  let fullPrice := ticketPrice format
  let discountPercent := 0.25
  let discountedTicket := discountedPrice fullPrice discountPercent
  let totalCost := fullPrice + discountedTicket
  let moneyBrought := 25
  moneyBrought - totalCost = 4 := by sorry


end NUMINAMATH_CALUDE_movie_ticket_change_l3589_358931


namespace NUMINAMATH_CALUDE_prob_different_colors_is_two_thirds_l3589_358968

/-- Represents the possible colors for socks -/
inductive SockColor
| Red
| Blue

/-- Represents the possible colors for headbands -/
inductive HeadbandColor
| Red
| Blue
| Green

/-- The probability of choosing different colors for socks and headbands -/
def prob_different_colors : ℚ :=
  2 / 3

theorem prob_different_colors_is_two_thirds :
  prob_different_colors = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_prob_different_colors_is_two_thirds_l3589_358968


namespace NUMINAMATH_CALUDE_field_trip_girls_fraction_l3589_358984

theorem field_trip_girls_fraction (total_boys : ℕ) (total_girls : ℕ) 
  (boys_fraction : ℚ) (girls_fraction : ℚ) :
  total_boys = 200 →
  total_girls = 150 →
  boys_fraction = 3 / 5 →
  girls_fraction = 4 / 5 →
  let boys_on_trip := (boys_fraction * total_boys : ℚ)
  let girls_on_trip := (girls_fraction * total_girls : ℚ)
  let total_on_trip := boys_on_trip + girls_on_trip
  (girls_on_trip / total_on_trip : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_field_trip_girls_fraction_l3589_358984


namespace NUMINAMATH_CALUDE_student_team_repetition_l3589_358999

theorem student_team_repetition (n : ℕ) (h : n > 0) :
  ∀ (arrangement : ℕ → Fin (n^2) → Fin n),
  ∃ (week1 week2 : ℕ) (student1 student2 : Fin (n^2)),
    week1 < week2 ∧ week2 ≤ n + 2 ∧
    student1 ≠ student2 ∧
    arrangement week1 student1 = arrangement week1 student2 ∧
    arrangement week2 student1 = arrangement week2 student2 :=
by sorry

end NUMINAMATH_CALUDE_student_team_repetition_l3589_358999


namespace NUMINAMATH_CALUDE_line_points_k_value_l3589_358980

/-- 
Given two points (m, n) and (m + 5, n + k) on a line with equation x = 2y + 5,
prove that k = 2.5
-/
theorem line_points_k_value 
  (m n k : ℝ) 
  (point1_on_line : m = 2 * n + 5)
  (point2_on_line : m + 5 = 2 * (n + k) + 5) :
  k = 2.5 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l3589_358980


namespace NUMINAMATH_CALUDE_ratio_of_fractions_l3589_358933

theorem ratio_of_fractions (P Q : ℤ) : 
  (∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 → 
    P / (x + 7) + Q / (x^2 - 6*x) = (x^2 - x + 15) / (x^3 + x^2 - 42*x)) →
  Q / P = 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_fractions_l3589_358933


namespace NUMINAMATH_CALUDE_profit_in_toys_l3589_358985

/-- 
Given:
- A man sold 18 toys for Rs. 18900
- The cost price of a toy is Rs. 900
Prove that the number of toys' cost price gained as profit is 3
-/
theorem profit_in_toys (total_toys : ℕ) (selling_price : ℕ) (cost_per_toy : ℕ) :
  total_toys = 18 →
  selling_price = 18900 →
  cost_per_toy = 900 →
  (selling_price - total_toys * cost_per_toy) / cost_per_toy = 3 :=
by sorry

end NUMINAMATH_CALUDE_profit_in_toys_l3589_358985


namespace NUMINAMATH_CALUDE_course_selection_methods_l3589_358926

/-- The number of courses in Group A -/
def group_A_courses : ℕ := 3

/-- The number of courses in Group B -/
def group_B_courses : ℕ := 4

/-- The total number of courses that must be selected -/
def total_selected : ℕ := 3

/-- The function to calculate the number of ways to select courses -/
def select_courses (group_A : ℕ) (group_B : ℕ) (total : ℕ) : ℕ :=
  Nat.choose group_A 2 * Nat.choose group_B 1 +
  Nat.choose group_A 1 * Nat.choose group_B 2

/-- Theorem stating that the number of different selection methods is 30 -/
theorem course_selection_methods :
  select_courses group_A_courses group_B_courses total_selected = 30 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_methods_l3589_358926


namespace NUMINAMATH_CALUDE_meal_combinations_count_l3589_358943

/-- The number of main dishes available on the menu -/
def num_main_dishes : ℕ := 12

/-- The number of appetizers available to choose from -/
def num_appetizers : ℕ := 5

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- Calculates the number of different meal combinations -/
def meal_combinations : ℕ := num_main_dishes ^ num_people * num_appetizers

/-- Theorem stating that the number of meal combinations is 720 -/
theorem meal_combinations_count : meal_combinations = 720 := by sorry

end NUMINAMATH_CALUDE_meal_combinations_count_l3589_358943


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l3589_358934

def is_prime (p : ℕ) : Prop := sorry

theorem largest_three_digit_product (n x y : ℕ) :
  n ≥ 100 ∧ n < 1000 ∧
  is_prime x ∧ is_prime y ∧ is_prime (10 * y - x) ∧
  x < 10 ∧ y < 10 ∧
  n = x * y * (10 * y - x) ∧
  x ≠ y ∧ x ≠ (10 * y - x) ∧ y ≠ (10 * y - x) →
  n ≤ 705 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l3589_358934


namespace NUMINAMATH_CALUDE_football_club_balance_l3589_358954

def initial_balance : ℝ := 100
def players_sold : ℕ := 2
def selling_price : ℝ := 10
def players_bought : ℕ := 4
def buying_price : ℝ := 15

theorem football_club_balance :
  initial_balance + players_sold * selling_price - players_bought * buying_price = 60 :=
by sorry

end NUMINAMATH_CALUDE_football_club_balance_l3589_358954


namespace NUMINAMATH_CALUDE_real_part_of_fraction_l3589_358950

theorem real_part_of_fraction (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 2) : 
  (2 / (1 - z)).re = 2/5 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_fraction_l3589_358950


namespace NUMINAMATH_CALUDE_triangle_area_function_l3589_358918

theorem triangle_area_function (A B C : ℝ) (a b c : ℝ) (x y : ℝ) :
  -- Given conditions
  A = π / 6 →
  a = 2 →
  0 < x →
  x < 5 * π / 6 →
  B = x →
  C = 5 * π / 6 - x →
  -- Area function
  y = 4 * Real.sin x * Real.sin (5 * π / 6 - x) →
  -- Prove
  0 < y ∧ y ≤ 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_function_l3589_358918


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_neg_one_range_of_a_for_inequality_l3589_358969

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x + 2|

-- Part 1
theorem solution_set_for_a_eq_neg_one (x : ℝ) :
  (f (-1) x ≥ x + 5) ↔ (x ≤ -2 ∨ x ≥ 4) := by sorry

-- Part 2
theorem range_of_a_for_inequality (a : ℝ) (h : a < 2) :
  (∀ x ∈ Set.Ioo (-5) (-3), f a x > x^2 + 2*x - 5) ↔ (a ≤ -2) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_neg_one_range_of_a_for_inequality_l3589_358969


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3589_358905

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (m : ℕ), (1101 + m) % 24 = 0 → m ≥ n) ∧
  (1101 + n) % 24 = 0 := by
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3589_358905


namespace NUMINAMATH_CALUDE_jacoby_trip_savings_l3589_358908

/-- The amount Jacoby needs for his trip to Brickville --/
def trip_cost : ℝ := 8000

/-- Jacoby's hourly wage --/
def hourly_wage : ℝ := 25

/-- Hours Jacoby worked --/
def hours_worked : ℝ := 15

/-- Tax rate on Jacoby's salary --/
def tax_rate : ℝ := 0.1

/-- Price of each cookie --/
def cookie_price : ℝ := 5

/-- Number of cookies sold --/
def cookies_sold : ℕ := 30

/-- Weekly tutoring earnings --/
def tutoring_weekly : ℝ := 100

/-- Weeks of tutoring --/
def tutoring_weeks : ℕ := 4

/-- Cost of lottery ticket --/
def lottery_ticket_cost : ℝ := 20

/-- Lottery winnings --/
def lottery_winnings : ℝ := 700

/-- Percentage of lottery winnings given to friend --/
def lottery_share : ℝ := 0.3

/-- Gift amount from each sister --/
def sister_gift : ℝ := 700

/-- Number of sisters --/
def number_of_sisters : ℕ := 2

/-- Cost of keychain --/
def keychain_cost : ℝ := 3

/-- Cost of backpack --/
def backpack_cost : ℝ := 47

/-- The amount Jacoby still needs for his trip --/
def amount_needed : ℝ := 5286.50

theorem jacoby_trip_savings : 
  trip_cost - (
    (hourly_wage * hours_worked * (1 - tax_rate)) +
    (cookie_price * cookies_sold) +
    (tutoring_weekly * tutoring_weeks) +
    ((lottery_winnings - lottery_ticket_cost) * (1 - lottery_share)) +
    (sister_gift * number_of_sisters) -
    (keychain_cost + backpack_cost)
  ) = amount_needed := by sorry

end NUMINAMATH_CALUDE_jacoby_trip_savings_l3589_358908


namespace NUMINAMATH_CALUDE_min_computers_to_purchase_l3589_358947

/-- Represents the problem of finding the minimum number of computers to purchase --/
theorem min_computers_to_purchase (total_devices : ℕ) (computer_cost whiteboard_cost max_cost : ℚ) :
  total_devices = 30 →
  computer_cost = 1/2 →
  whiteboard_cost = 3/2 →
  max_cost = 30 →
  ∃ (min_computers : ℕ),
    min_computers = 15 ∧
    ∀ (x : ℕ),
      x < 15 →
      (x : ℚ) * computer_cost + (total_devices - x : ℚ) * whiteboard_cost > max_cost :=
by sorry

end NUMINAMATH_CALUDE_min_computers_to_purchase_l3589_358947


namespace NUMINAMATH_CALUDE_last_four_digits_l3589_358988

theorem last_four_digits : (301 * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 4 % 10000 = 5856 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_l3589_358988


namespace NUMINAMATH_CALUDE_intersection_implies_k_zero_l3589_358912

/-- Line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem intersection_implies_k_zero (m n : Line) (h1 : m.slope = 3) (h2 : m.intercept = 5)
    (h3 : n.intercept = -7) (h4 : m.equation (-4) (-7)) (h5 : n.equation (-4) (-7)) :
    n.slope = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_k_zero_l3589_358912


namespace NUMINAMATH_CALUDE_otimes_h_h_otimes_h_h_l3589_358971

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^2 + y^2

-- Theorem statement
theorem otimes_h_h_otimes_h_h (h : ℝ) : otimes (otimes h h) (otimes h h) = 8 * h^4 := by
  sorry

end NUMINAMATH_CALUDE_otimes_h_h_otimes_h_h_l3589_358971


namespace NUMINAMATH_CALUDE_granola_bar_distribution_l3589_358966

/-- Given that Monroe made x granola bars, she and her husband ate 2/3 of them,
    and the rest were divided equally among y children, with each child receiving z granola bars,
    prove that z = x / (3 * y) -/
theorem granola_bar_distribution (x y z : ℚ) (hx : x > 0) (hy : y > 0) : 
  (2 / 3 * x + y * z = x) → z = x / (3 * y) := by
  sorry

end NUMINAMATH_CALUDE_granola_bar_distribution_l3589_358966


namespace NUMINAMATH_CALUDE_jeremy_dosage_l3589_358959

/-- Represents the duration of Jeremy's medication course in weeks -/
def duration : ℕ := 2

/-- Represents the number of pills Jeremy takes in total -/
def total_pills : ℕ := 112

/-- Represents the dosage of each pill in milligrams -/
def pill_dosage : ℕ := 500

/-- Represents the interval between doses in hours -/
def dose_interval : ℕ := 6

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the total milligrams of medication taken over the entire course -/
def total_mg : ℕ := total_pills * pill_dosage

/-- Calculates the number of doses taken per day -/
def doses_per_day : ℕ := hours_per_day / dose_interval

/-- Calculates the total number of doses taken over the entire course -/
def total_doses : ℕ := duration * 7 * doses_per_day

/-- Theorem stating that Jeremy takes 1000 mg every 6 hours -/
theorem jeremy_dosage : total_mg / total_doses = 1000 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_dosage_l3589_358959


namespace NUMINAMATH_CALUDE_sock_shoe_permutations_l3589_358944

def num_legs : ℕ := 10

def total_items : ℕ := 2 * num_legs

def valid_permutations : ℕ := Nat.factorial total_items / (2^num_legs)

theorem sock_shoe_permutations :
  valid_permutations = Nat.factorial total_items / (2^num_legs) :=
by sorry

end NUMINAMATH_CALUDE_sock_shoe_permutations_l3589_358944


namespace NUMINAMATH_CALUDE_domain_of_g_l3589_358942

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the new function g
def g (x : ℝ) : ℝ := f (2 * x + 1)

-- State the theorem
theorem domain_of_g :
  (∀ x, f x ≠ 0 → x ∈ Set.Icc (-2) 3) →
  (∀ x, g x ≠ 0 → x ∈ Set.Icc (-3/2) 1) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l3589_358942


namespace NUMINAMATH_CALUDE_problem_statement_l3589_358913

theorem problem_statement : (-4 : ℝ)^2007 * (-0.25 : ℝ)^2008 = -0.25 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3589_358913


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l3589_358901

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a line is tangent to a circle -/
def Line.tangentTo (l : Line) (c : Circle) : Prop :=
  (l.a * c.center.1 + l.b * c.center.2 + l.c)^2 = 
    (l.a^2 + l.b^2) * c.radius^2

theorem circle_and_tangent_line 
  (c : Circle) 
  (l : Line) : 
  c.contains (0, 0) → 
  c.contains (4, 0) → 
  c.contains (0, 2) → 
  l.a = 2 → 
  l.b = -1 → 
  l.c = 2 → 
  l.tangentTo c → 
  c = { center := (2, 1), radius := Real.sqrt 5 } ∧ 
  l = { a := 2, b := -1, c := 2 } := by
  sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l3589_358901


namespace NUMINAMATH_CALUDE_min_surface_area_height_l3589_358909

/-- The height that minimizes the surface area of an open-top rectangular box with square base and volume 4 -/
theorem min_surface_area_height : ∃ (h : ℝ), h > 0 ∧ 
  (∀ (x : ℝ), x > 0 → x^2 * h = 4 → 
    ∀ (h' : ℝ), h' > 0 → x^2 * h' = 4 → 
      x^2 + 4*x*h ≤ x^2 + 4*x*h') ∧ 
  h = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_surface_area_height_l3589_358909


namespace NUMINAMATH_CALUDE_expression_value_l3589_358961

/-- Custom operation for real numbers -/
def custom_op (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating the value of the expression when x^2 - 3x + 1 = 0 -/
theorem expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  custom_op (x + 1) (x - 2) (3*x) (x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3589_358961


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3589_358955

theorem min_value_expression (a b c : ℝ) 
  (ha : -1 < a ∧ a < 1) 
  (hb : -1 < b ∧ b < 1) 
  (hc : -1 < c ∧ c < 1) :
  (1 / ((1 - a^2) * (1 - b^2) * (1 - c^2))) + 
  (1 / ((1 + a^2) * (1 + b^2) * (1 + c^2))) ≥ 2 :=
by sorry

theorem min_value_achieved (a b c : ℝ) 
  (ha : a = 0) 
  (hb : b = 0) 
  (hc : c = 0) :
  (1 / ((1 - a^2) * (1 - b^2) * (1 - c^2))) + 
  (1 / ((1 + a^2) * (1 + b^2) * (1 + c^2))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3589_358955


namespace NUMINAMATH_CALUDE_largest_expression_l3589_358990

theorem largest_expression : 
  let a := 15847
  let b := 3174
  let expr1 := a + 1 / b
  let expr2 := a - 1 / b
  let expr3 := a * (1 / b)
  let expr4 := a / (1 / b)
  let expr5 := a ^ 1.03174
  (expr4 > expr1) ∧ 
  (expr4 > expr2) ∧ 
  (expr4 > expr3) ∧ 
  (expr4 > expr5) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l3589_358990


namespace NUMINAMATH_CALUDE_bret_frog_count_l3589_358986

theorem bret_frog_count :
  ∀ (alster_frogs quinn_frogs bret_frogs : ℕ),
    alster_frogs = 2 →
    quinn_frogs = 2 * alster_frogs →
    bret_frogs = 3 * quinn_frogs →
    bret_frogs = 12 := by
  sorry

end NUMINAMATH_CALUDE_bret_frog_count_l3589_358986


namespace NUMINAMATH_CALUDE_fourth_term_value_l3589_358987

def S (n : ℕ) : ℤ := n^2 - 3*n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem fourth_term_value : a 4 = 4 := by sorry

end NUMINAMATH_CALUDE_fourth_term_value_l3589_358987


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3589_358992

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  (1 / (a + 2*b)) + (1 / (b + 2*c)) + (1 / (c + 2*a)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3589_358992


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_has_20_sides_l3589_358995

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_has_20_sides :
  ∀ n : ℕ,
  n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_has_20_sides_l3589_358995


namespace NUMINAMATH_CALUDE_kaydence_sister_age_l3589_358940

/-- Given the ages of family members, prove the age of Kaydence's sister -/
theorem kaydence_sister_age 
  (total_age : ℕ) 
  (father_age : ℕ) 
  (mother_age : ℕ) 
  (brother_age : ℕ) 
  (kaydence_age : ℕ) 
  (h1 : total_age = 200)
  (h2 : father_age = 60)
  (h3 : mother_age = father_age - 2)
  (h4 : brother_age = father_age / 2)
  (h5 : kaydence_age = 12) :
  total_age - (father_age + mother_age + brother_age + kaydence_age) = 40 := by
  sorry


end NUMINAMATH_CALUDE_kaydence_sister_age_l3589_358940


namespace NUMINAMATH_CALUDE_ae_length_l3589_358993

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of side AB -/
  ab : ℝ
  /-- The length of side CD -/
  cd : ℝ
  /-- The property that AD = BC -/
  ad_eq_bc : True
  /-- Point E such that BC = EC -/
  bc_eq_ec : True
  /-- AE is perpendicular to EC -/
  ae_perp_ec : True

/-- The main theorem stating the length of AE in the specific isosceles trapezoid -/
theorem ae_length (t : IsoscelesTrapezoid) (h1 : t.ab = 3) (h2 : t.cd = 8) : 
  ∃ ae : ℝ, ae = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ae_length_l3589_358993


namespace NUMINAMATH_CALUDE_line_plane_relationship_l3589_358945

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (intersect : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_or_intersect_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (β : Plane) 
  (h1 : intersect a b) 
  (h2 : parallel a β) : 
  line_parallel_or_intersect_plane b β :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l3589_358945


namespace NUMINAMATH_CALUDE_x_is_bounded_l3589_358989

/-- Product of all decimal digits of a natural number -/
def P (x : ℕ) : ℕ := sorry

/-- Sequence defined recursively by xₙ₊₁ = xₙ + P(xₙ) -/
def x : ℕ → ℕ
  | 0 => sorry  -- x₁ is some positive integer
  | n + 1 => x n + P (x n)

/-- The sequence (xₙ) is bounded -/
theorem x_is_bounded : ∃ (M : ℕ), ∀ (n : ℕ), x n ≤ M := by sorry

end NUMINAMATH_CALUDE_x_is_bounded_l3589_358989


namespace NUMINAMATH_CALUDE_arithmetic_square_root_l3589_358921

theorem arithmetic_square_root (a : ℝ) (h : a > 0) : Real.sqrt a = (a ^ (1/2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_l3589_358921


namespace NUMINAMATH_CALUDE_required_plane_satisfies_conditions_l3589_358924

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane equation in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The given plane equation -/
def givenPlane : PlaneEquation := { A := 2, B := -1, C := 4, D := -7 }

/-- The two points that the required plane passes through -/
def point1 : Point3D := { x := 2, y := -1, z := 0 }
def point2 : Point3D := { x := 0, y := 3, z := 1 }

/-- The equation of the required plane -/
def requiredPlane : PlaneEquation := { A := 17, B := 10, C := -6, D := -24 }

/-- Function to check if a point satisfies a plane equation -/
def satisfiesPlaneEquation (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- Function to check if two planes are perpendicular -/
def arePlanesPerp (eq1 eq2 : PlaneEquation) : Prop :=
  eq1.A * eq2.A + eq1.B * eq2.B + eq1.C * eq2.C = 0

/-- Theorem stating that the required plane satisfies all conditions -/
theorem required_plane_satisfies_conditions :
  satisfiesPlaneEquation point1 requiredPlane ∧
  satisfiesPlaneEquation point2 requiredPlane ∧
  arePlanesPerp requiredPlane givenPlane ∧
  requiredPlane.A > 0 ∧
  Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs requiredPlane.A) (Int.natAbs requiredPlane.B)) (Int.natAbs requiredPlane.C)) (Int.natAbs requiredPlane.D) = 1 :=
by sorry


end NUMINAMATH_CALUDE_required_plane_satisfies_conditions_l3589_358924


namespace NUMINAMATH_CALUDE_complex_distance_theorem_l3589_358967

theorem complex_distance_theorem (z z₁ z₂ : ℂ) :
  z₁ ≠ z₂ →
  z₁^2 = -2 - 2 * Complex.I * Real.sqrt 3 →
  z₂^2 = -2 - 2 * Complex.I * Real.sqrt 3 →
  Complex.abs (z - z₁) = 4 →
  Complex.abs (z - z₂) = 4 →
  Complex.abs z = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_theorem_l3589_358967


namespace NUMINAMATH_CALUDE_find_divisor_l3589_358911

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 132 →
  quotient = 8 →
  remainder = 4 →
  dividend = divisor * quotient + remainder →
  divisor = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3589_358911


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3589_358957

theorem gcd_lcm_sum : Nat.gcd 25 64 + Nat.lcm 15 20 = 61 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3589_358957


namespace NUMINAMATH_CALUDE_cubic_polynomial_third_root_l3589_358938

theorem cubic_polynomial_third_root 
  (a b : ℚ) 
  (h1 : a * (-1)^3 + (a + 3*b) * (-1)^2 + (b - 2*a) * (-1) + (10 - a) = 0)
  (h2 : a * 4^3 + (a + 3*b) * 4^2 + (b - 2*a) * 4 + (10 - a) = 0) :
  ∃ (r : ℚ), a * r^3 + (a + 3*b) * r^2 + (b - 2*a) * r + (10 - a) = 0 ∧ 
              r = -67/88 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_third_root_l3589_358938


namespace NUMINAMATH_CALUDE_poker_night_cards_l3589_358910

theorem poker_night_cards (half_decks full_decks thrown_away remaining : ℕ) : 
  half_decks = 3 →
  full_decks = 3 →
  thrown_away = 34 →
  remaining = 200 →
  ∃ (cards_per_full_deck cards_per_half_deck : ℕ),
    cards_per_half_deck = cards_per_full_deck / 2 ∧
    remaining + thrown_away = half_decks * cards_per_half_deck + full_decks * cards_per_full_deck ∧
    cards_per_full_deck = 52 :=
by sorry

end NUMINAMATH_CALUDE_poker_night_cards_l3589_358910


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l3589_358917

theorem greatest_integer_radius (r : ℝ) : r > 0 → r * r * Real.pi < 75 * Real.pi → ∃ n : ℕ, n = 8 ∧ (∀ m : ℕ, m * m * Real.pi < 75 * Real.pi → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l3589_358917


namespace NUMINAMATH_CALUDE_B_equals_interval_A_union_C_equals_A_l3589_358904

-- Define sets A, B, and C
def A : Set ℝ := {x | 2 * x^2 - 9 * x + 4 > 0}
def B : Set ℝ := {y | ∃ x ∈ (Set.univ \ A), y = -x^2 + 2 * x}
def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x ≤ 2 * m - 1}

-- Theorem 1: B is equal to the closed interval [-8, 1]
theorem B_equals_interval : B = Set.Icc (-8) 1 := by sorry

-- Theorem 2: A ∪ C = A if and only if m ≤ 2 or m ≥ 3
theorem A_union_C_equals_A (m : ℝ) : A ∪ C m = A ↔ m ≤ 2 ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_B_equals_interval_A_union_C_equals_A_l3589_358904


namespace NUMINAMATH_CALUDE_at_home_workforce_trend_l3589_358972

/-- Represents the percentage of working adults in Parkertown working at home for a given year -/
def AtHomeWorkforce : ℕ → ℚ
  | 1990 => 12/100
  | 1995 => 15/100
  | 2000 => 14/100
  | 2005 => 28/100
  | _ => 0

/-- The trend of the at-home workforce in Parkertown from 1990 to 2005 -/
theorem at_home_workforce_trend :
  AtHomeWorkforce 1995 > AtHomeWorkforce 1990 ∧
  AtHomeWorkforce 2000 < AtHomeWorkforce 1995 ∧
  AtHomeWorkforce 2005 > AtHomeWorkforce 2000 ∧
  (AtHomeWorkforce 2005 - AtHomeWorkforce 2000) > (AtHomeWorkforce 1995 - AtHomeWorkforce 1990) :=
by sorry

end NUMINAMATH_CALUDE_at_home_workforce_trend_l3589_358972


namespace NUMINAMATH_CALUDE_max_value_theorem_l3589_358936

theorem max_value_theorem (x : ℝ) :
  x^4 / (x^8 + 2*x^6 + 4*x^4 + 4*x^2 + 16) ≤ 1/16 ∧
  ∃ y : ℝ, y^4 / (y^8 + 2*y^6 + 4*y^4 + 4*y^2 + 16) = 1/16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3589_358936


namespace NUMINAMATH_CALUDE_adults_attending_play_l3589_358996

/-- Proves the number of adults attending a play given the total attendance,
    admission prices, and total receipts. -/
theorem adults_attending_play
  (total_people : ℕ)
  (adult_price child_price : ℕ)
  (total_receipts : ℕ)
  (h1 : total_people = 610)
  (h2 : adult_price = 2)
  (h3 : child_price = 1)
  (h4 : total_receipts = 960) :
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_receipts ∧
    adults = 350 :=
by sorry

end NUMINAMATH_CALUDE_adults_attending_play_l3589_358996


namespace NUMINAMATH_CALUDE_problem_statement_l3589_358919

open Real

theorem problem_statement (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ - a)^2 + (log (x₀^2) - 2*a)^2 ≤ 4/5) → a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3589_358919


namespace NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_ln_x_plus_one_negative_l3589_358997

theorem x_negative_necessary_not_sufficient_for_ln_x_plus_one_negative :
  (∀ x : ℝ, Real.log (x + 1) < 0 → x < 0) ∧
  (∃ x : ℝ, x < 0 ∧ Real.log (x + 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_ln_x_plus_one_negative_l3589_358997


namespace NUMINAMATH_CALUDE_intersection_slope_problem_l3589_358974

/-- Given two lines intersecting at (40, 30), where one line has a slope of 6
    and the distance between their x-intercepts is 10,
    prove that the slope of the other line is 2. -/
theorem intersection_slope_problem (m : ℝ) : 
  let line1 : ℝ → ℝ := λ x => m * x - 40 * m + 30
  let line2 : ℝ → ℝ := λ x => 6 * x - 210
  let x_intercept1 : ℝ := (40 * m - 30) / m
  let x_intercept2 : ℝ := 35
  (∃ x y, line1 x = line2 x ∧ x = 40 ∧ y = 30) →  -- Lines intersect at (40, 30)
  |x_intercept1 - x_intercept2| = 10 →           -- Distance between x-intercepts is 10
  m = 2                                          -- Slope of the first line is 2
  := by sorry

end NUMINAMATH_CALUDE_intersection_slope_problem_l3589_358974


namespace NUMINAMATH_CALUDE_game_correct_answers_l3589_358982

theorem game_correct_answers : 
  ∀ (total_questions : ℕ) 
    (correct_reward incorrect_penalty : ℕ) 
    (correct_answers : ℕ),
  total_questions = 50 →
  correct_reward = 7 →
  incorrect_penalty = 3 →
  correct_answers * correct_reward = 
    (total_questions - correct_answers) * incorrect_penalty →
  correct_answers = 15 := by
sorry

end NUMINAMATH_CALUDE_game_correct_answers_l3589_358982


namespace NUMINAMATH_CALUDE_science_project_cans_l3589_358994

def empty_cans_problem (alyssa_cans abigail_cans more_needed : ℕ) : Prop :=
  alyssa_cans + abigail_cans + more_needed = 100

theorem science_project_cans : empty_cans_problem 30 43 27 := by
  sorry

end NUMINAMATH_CALUDE_science_project_cans_l3589_358994


namespace NUMINAMATH_CALUDE_total_surveys_per_week_l3589_358937

/-- Proves that the total number of surveys completed per week is 50 given the problem conditions -/
theorem total_surveys_per_week 
  (regular_rate : ℝ)
  (cellphone_rate_increase : ℝ)
  (cellphone_surveys : ℕ)
  (total_earnings : ℝ)
  (h1 : regular_rate = 30)
  (h2 : cellphone_rate_increase = 0.2)
  (h3 : cellphone_surveys = 50)
  (h4 : total_earnings = 3300)
  (h5 : total_earnings = cellphone_surveys * (regular_rate * (1 + cellphone_rate_increase))) :
  cellphone_surveys = 50 := by
  sorry

#check total_surveys_per_week

end NUMINAMATH_CALUDE_total_surveys_per_week_l3589_358937


namespace NUMINAMATH_CALUDE_frigate_catches_smuggler_l3589_358946

/-- Represents the chase scenario between a frigate and a smuggler's ship -/
structure ChaseScenario where
  initial_distance : ℝ
  frigate_speed : ℝ
  smuggler_speed : ℝ
  chase_duration : ℝ

/-- Calculates the distance traveled by a ship given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that the frigate catches up to the smuggler's ship after 3 hours -/
theorem frigate_catches_smuggler (scenario : ChaseScenario) 
    (h1 : scenario.initial_distance = 12)
    (h2 : scenario.frigate_speed = 14)
    (h3 : scenario.smuggler_speed = 10)
    (h4 : scenario.chase_duration = 3) :
    distance_traveled scenario.frigate_speed scenario.chase_duration = 
    scenario.initial_distance + distance_traveled scenario.smuggler_speed scenario.chase_duration :=
  sorry

#check frigate_catches_smuggler

end NUMINAMATH_CALUDE_frigate_catches_smuggler_l3589_358946


namespace NUMINAMATH_CALUDE_tims_drive_distance_l3589_358965

/-- Represents the scenario of Tim's drive to work -/
def TimsDrive (totalDistance : ℝ) : Prop :=
  let normalTime : ℝ := 120
  let newTime : ℝ := 165
  let speedReduction : ℝ := 30 / 60 -- 30 mph converted to miles per minute
  let normalSpeed : ℝ := totalDistance / normalTime
  let newSpeed : ℝ := normalSpeed - speedReduction
  let halfDistance : ℝ := totalDistance / 2
  normalTime / 2 + halfDistance / newSpeed = newTime

/-- Theorem stating that the total distance of Tim's drive is 140 miles -/
theorem tims_drive_distance : ∃ (d : ℝ), TimsDrive d ∧ d = 140 :=
sorry

end NUMINAMATH_CALUDE_tims_drive_distance_l3589_358965


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3589_358906

theorem smallest_number_satisfying_conditions : ∃ n : ℕ,
  (n % 6 = 1) ∧ (n % 8 = 3) ∧ (n % 9 = 2) ∧
  (∀ m : ℕ, m < n → ¬((m % 6 = 1) ∧ (m % 8 = 3) ∧ (m % 9 = 2))) ∧
  n = 107 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3589_358906


namespace NUMINAMATH_CALUDE_chocolate_eggs_duration_l3589_358930

/-- Proves that given 40 chocolate eggs, eating 2 eggs per day for 5 days a week will result in the eggs lasting for 4 weeks. -/
theorem chocolate_eggs_duration (total_eggs : ℕ) (eggs_per_day : ℕ) (school_days_per_week : ℕ) : 
  total_eggs = 40 → 
  eggs_per_day = 2 → 
  school_days_per_week = 5 → 
  (total_eggs / (eggs_per_day * school_days_per_week) : ℚ) = 4 := by
sorry


end NUMINAMATH_CALUDE_chocolate_eggs_duration_l3589_358930


namespace NUMINAMATH_CALUDE_step_increase_proof_l3589_358916

def daily_steps (x : ℕ) (week : ℕ) : ℕ :=
  1000 + (week - 1) * x

def weekly_steps (x : ℕ) (week : ℕ) : ℕ :=
  7 * daily_steps x week

def total_steps (x : ℕ) : ℕ :=
  weekly_steps x 1 + weekly_steps x 2 + weekly_steps x 3 + weekly_steps x 4

theorem step_increase_proof :
  ∃ x : ℕ, total_steps x = 70000 ∧ x = 1000 :=
by sorry

end NUMINAMATH_CALUDE_step_increase_proof_l3589_358916
