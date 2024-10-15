import Mathlib

namespace NUMINAMATH_CALUDE_exactly_one_project_not_selected_l1320_132070

/-- The number of employees and projects -/
def n : ℕ := 4

/-- The probability of exactly one project not being selected -/
def probability : ℚ := 9/16

/-- Theorem stating the probability of exactly one project not being selected -/
theorem exactly_one_project_not_selected :
  (n : ℚ)^n * probability = (n.choose 2) * n! :=
sorry

end NUMINAMATH_CALUDE_exactly_one_project_not_selected_l1320_132070


namespace NUMINAMATH_CALUDE_walking_time_l1320_132064

/-- Given a walking speed of 10 km/hr and a distance of 4 km, the time taken is 24 minutes. -/
theorem walking_time (speed : ℝ) (distance : ℝ) : 
  speed = 10 → distance = 4 → (distance / speed) * 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_l1320_132064


namespace NUMINAMATH_CALUDE_total_chickens_and_ducks_l1320_132095

theorem total_chickens_and_ducks (num_chickens : ℕ) (duck_difference : ℕ) : 
  num_chickens = 45 → 
  duck_difference = 8 → 
  num_chickens + (num_chickens - duck_difference) = 82 :=
by sorry

end NUMINAMATH_CALUDE_total_chickens_and_ducks_l1320_132095


namespace NUMINAMATH_CALUDE_equation_holds_iff_m_equals_168_l1320_132003

theorem equation_holds_iff_m_equals_168 :
  ∀ m : ℤ, (4^4 : ℤ) - 7 = 9^2 + m ↔ m = 168 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_iff_m_equals_168_l1320_132003


namespace NUMINAMATH_CALUDE_davids_remaining_money_l1320_132073

theorem davids_remaining_money (initial : ℝ) (remaining : ℝ) (spent : ℝ) : 
  initial = 1500 →
  remaining + spent = initial →
  remaining < spent →
  remaining < 750 := by
sorry

end NUMINAMATH_CALUDE_davids_remaining_money_l1320_132073


namespace NUMINAMATH_CALUDE_remaining_distance_to_grandma_l1320_132058

theorem remaining_distance_to_grandma (total_distance : ℕ) 
  (distance1 distance2 distance3 distance4 distance5 distance6 : ℕ) : 
  total_distance = 78 ∧ 
  distance1 = 35 ∧ 
  distance2 = 7 ∧ 
  distance3 = 18 ∧ 
  distance4 = 3 ∧ 
  distance5 = 12 ∧ 
  distance6 = 2 → 
  total_distance - (distance1 + distance2 + distance3 + distance4 + distance5 + distance6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_to_grandma_l1320_132058


namespace NUMINAMATH_CALUDE_power_division_equality_l1320_132043

theorem power_division_equality (m : ℕ) (h : m = 32^500) : m / 8 = 2^2497 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l1320_132043


namespace NUMINAMATH_CALUDE_min_value_theorem_l1320_132018

theorem min_value_theorem (x y : ℝ) (h : x - 2*y - 4 = 0) :
  ∃ (min : ℝ), min = 8 ∧ ∀ z, z = 2^x + 1/(4^y) → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1320_132018


namespace NUMINAMATH_CALUDE_rotation_theorem_l1320_132001

/-- Represents a square board with side length 2^n -/
structure Board (n : Nat) where
  size : Nat := 2^n
  elements : Fin (size * size) → Nat

/-- Represents the state of the board after rotations -/
def rotatedBoard (n : Nat) : Board n → Board n :=
  sorry

/-- The main diagonal of a board -/
def mainDiagonal (n : Nat) (b : Board n) : List Nat :=
  sorry

/-- The other main diagonal (bottom-left to top-right) of a board -/
def otherMainDiagonal (n : Nat) (b : Board n) : List Nat :=
  sorry

/-- Initial board setup -/
def initialBoard : Board 5 :=
  { elements := λ i => i.val + 1 }

theorem rotation_theorem :
  mainDiagonal 5 (rotatedBoard 5 initialBoard) =
    (otherMainDiagonal 5 initialBoard).reverse := by
  sorry

end NUMINAMATH_CALUDE_rotation_theorem_l1320_132001


namespace NUMINAMATH_CALUDE_cubic_function_extremum_l1320_132032

/-- Given a cubic function f(x) = x³ + ax² + bx + a², 
    if f has an extremum at x = 1 and f(1) = 10, then a + b = -7 -/
theorem cubic_function_extremum (a b : ℝ) : 
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x + a^2
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  f 1 = 10 →
  a + b = -7 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_extremum_l1320_132032


namespace NUMINAMATH_CALUDE_discontinuous_function_l1320_132038

def M (f : ℝ → ℝ) (x : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of (λ i j => if i = j then 1 + f (x i) else f (x j))

theorem discontinuous_function
  (f : ℝ → ℝ)
  (f_nonzero : ∀ x, f x ≠ 0)
  (f_condition : f 2014 = 1 - f 2013)
  (det_zero : ∀ (n : ℕ) (x : Fin n → ℝ), Function.Injective x → Matrix.det (M f x) = 0) :
  ¬Continuous f :=
sorry

end NUMINAMATH_CALUDE_discontinuous_function_l1320_132038


namespace NUMINAMATH_CALUDE_intersection_equals_Q_l1320_132086

-- Define the sets P and Q
def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {x | x^2 ≤ 1}

-- Theorem statement
theorem intersection_equals_Q : P ∩ Q = Q := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_Q_l1320_132086


namespace NUMINAMATH_CALUDE_claire_photos_l1320_132027

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 24) :
  claire = 12 := by
  sorry

end NUMINAMATH_CALUDE_claire_photos_l1320_132027


namespace NUMINAMATH_CALUDE_lucille_house_height_difference_l1320_132044

theorem lucille_house_height_difference (h1 h2 h3 : ℝ) :
  h1 = 80 ∧ h2 = 70 ∧ h3 = 99 →
  ((h1 + h2 + h3) / 3) - h1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_lucille_house_height_difference_l1320_132044


namespace NUMINAMATH_CALUDE_cube_root_unity_product_l1320_132065

/-- A complex cube root of unity -/
def ω : ℂ :=
  sorry

/-- The property of ω being a complex cube root of unity -/
axiom ω_cube_root : ω^3 = 1

/-- The sum of powers of ω equals zero -/
axiom ω_sum_zero : 1 + ω + ω^2 = 0

/-- The main theorem -/
theorem cube_root_unity_product (a b c : ℂ) :
  (a + b*ω + c*ω^2) * (a + b*ω^2 + c*ω) = a^2 + b^2 + c^2 - a*b - a*c - b*c :=
sorry

end NUMINAMATH_CALUDE_cube_root_unity_product_l1320_132065


namespace NUMINAMATH_CALUDE_basketball_game_third_quarter_score_l1320_132026

/-- Represents the points scored by a team in each quarter -/
structure TeamScore :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if a TeamScore follows a geometric sequence -/
def isGeometric (s : TeamScore) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if a TeamScore follows an arithmetic sequence -/
def isArithmetic (s : TeamScore) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a TeamScore -/
def totalScore (s : TeamScore) : ℕ := s.q1 + s.q2 + s.q3 + s.q4

theorem basketball_game_third_quarter_score :
  ∀ (teamA teamB : TeamScore),
    teamA.q1 = teamB.q1 →                        -- Tied at the end of first quarter
    isGeometric teamA →                          -- Team A follows geometric sequence
    isArithmetic teamB →                         -- Team B follows arithmetic sequence
    totalScore teamA = totalScore teamB + 3 →    -- Team A wins by 3 points
    totalScore teamA ≤ 100 →                     -- Team A's total score ≤ 100
    totalScore teamB ≤ 100 →                     -- Team B's total score ≤ 100
    teamA.q3 + teamB.q3 = 60                     -- Total score in third quarter is 60
  := by sorry

end NUMINAMATH_CALUDE_basketball_game_third_quarter_score_l1320_132026


namespace NUMINAMATH_CALUDE_fraction_equality_l1320_132062

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - d) * (b - c) / ((a - b) * (c - d)) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1320_132062


namespace NUMINAMATH_CALUDE_inequality_proof_l1320_132021

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a * 2^(-b) > b * 2^(-a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1320_132021


namespace NUMINAMATH_CALUDE_circle_radius_squared_l1320_132075

-- Define the circle and points
variable (r : ℝ) -- radius of the circle
variable (A B C D P : ℝ × ℝ) -- points in 2D plane

-- Define the conditions
def AB : ℝ := 12 -- length of chord AB
def CD : ℝ := 8 -- length of chord CD
def BP : ℝ := 9 -- distance from B to P

-- Define the angle condition
def angle_APD_is_right : Prop := sorry

-- Define that P is outside the circle
def P_outside_circle : Prop := sorry

-- Define that AB and CD extended intersect at P
def chords_intersect_at_P : Prop := sorry

-- Theorem statement
theorem circle_radius_squared 
  (h1 : AB = 12)
  (h2 : CD = 8)
  (h3 : BP = 9)
  (h4 : angle_APD_is_right)
  (h5 : P_outside_circle)
  (h6 : chords_intersect_at_P) :
  r^2 = 97.361 := by sorry

end NUMINAMATH_CALUDE_circle_radius_squared_l1320_132075


namespace NUMINAMATH_CALUDE_max_xy_value_l1320_132068

theorem max_xy_value (x y : ℝ) (hx : x < 0) (hy : y < 0) (h_eq : 3 * x + y = -2) :
  ∃ (max_xy : ℝ), max_xy = 1/3 ∧ ∀ z, z = x * y → z ≤ max_xy :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l1320_132068


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l1320_132089

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_special_case (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4)^2 - 4*(a 4) - 1 = 0 →
  (a 8)^2 - 4*(a 8) - 1 = 0 →
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l1320_132089


namespace NUMINAMATH_CALUDE_fraction_sum_integer_implies_fractions_integer_l1320_132014

theorem fraction_sum_integer_implies_fractions_integer
  (a b c : ℤ) (h : ∃ (m : ℤ), (a * b) / c + (a * c) / b + (b * c) / a = m) :
  (∃ (k : ℤ), (a * b) / c = k) ∧
  (∃ (l : ℤ), (a * c) / b = l) ∧
  (∃ (n : ℤ), (b * c) / a = n) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_implies_fractions_integer_l1320_132014


namespace NUMINAMATH_CALUDE_wife_walking_speed_l1320_132041

/-- Proves that given a circular track of 726 m circumference, if two people walk in opposite
    directions starting from the same point, with one person walking at 4.5 km/hr and they
    meet after 5.28 minutes, then the other person's walking speed is 3.75 km/hr. -/
theorem wife_walking_speed
  (track_circumference : ℝ)
  (suresh_speed : ℝ)
  (meeting_time : ℝ)
  (h1 : track_circumference = 726 / 1000) -- Convert 726 m to km
  (h2 : suresh_speed = 4.5)
  (h3 : meeting_time = 5.28 / 60) -- Convert 5.28 minutes to hours
  : ∃ (wife_speed : ℝ), wife_speed = 3.75 := by
  sorry

#check wife_walking_speed

end NUMINAMATH_CALUDE_wife_walking_speed_l1320_132041


namespace NUMINAMATH_CALUDE_qr_length_l1320_132051

/-- Triangle DEF with given side lengths -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ

/-- Circle with center and two points it passes through -/
structure Circle where
  center : ℝ × ℝ
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The problem setup -/
def ProblemSetup (t : Triangle) (c1 c2 : Circle) : Prop :=
  t.DE = 7 ∧ t.EF = 24 ∧ t.DF = 25 ∧
  c1.center.1 = c1.point1.1 ∧ -- Q is on the same vertical line as D
  c1.point2 = (t.DF, 0) ∧ -- F is at (25, 0)
  c2.center.2 = c2.point1.2 ∧ -- R is on the same horizontal line as E
  c2.point2 = (0, 0) -- D is at (0, 0)

theorem qr_length (t : Triangle) (c1 c2 : Circle) 
  (h : ProblemSetup t c1 c2) : 
  ‖c1.center - c2.center‖ = 8075 / 84 := by
  sorry

end NUMINAMATH_CALUDE_qr_length_l1320_132051


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_l1320_132033

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| - a

-- Theorem 1: Solution set for f(x) > x + 1 when a = 1
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x > x + 1} = {x : ℝ | x > 3 ∨ x < -1/3} :=
sorry

-- Theorem 2: Range of a for which ∃x : f(x) < 0.5 * f(x + 1)
theorem range_of_a :
  {a : ℝ | ∃ x, f a x < 0.5 * f a (x + 1)} = {a : ℝ | a > -2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_l1320_132033


namespace NUMINAMATH_CALUDE_tory_cookie_sales_l1320_132099

theorem tory_cookie_sales (grandmother_packs uncle_packs neighbor_packs more_packs : ℕ) 
  (h1 : grandmother_packs = 12)
  (h2 : uncle_packs = 7)
  (h3 : neighbor_packs = 5)
  (h4 : more_packs = 26) :
  grandmother_packs + uncle_packs + neighbor_packs + more_packs = 50 := by
  sorry

end NUMINAMATH_CALUDE_tory_cookie_sales_l1320_132099


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_main_theorem_l1320_132081

theorem repeating_decimal_sum : ∀ (a b c : ℕ), a < 10 → b < 10 → c < 10 →
  (a : ℚ) / 9 + (b : ℚ) / 9 - (c : ℚ) / 9 = (a + b - c : ℚ) / 9 :=
by sorry

theorem main_theorem : (8 : ℚ) / 9 + (2 : ℚ) / 9 - (6 : ℚ) / 9 = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_main_theorem_l1320_132081


namespace NUMINAMATH_CALUDE_point_transformation_l1320_132071

/-- Reflect a point (x, y) across the line y = x -/
def reflect_across_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Rotate a point (x, y) by 180° around a center (h, k) -/
def rotate_180_around (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let Q : ℝ × ℝ := (a, b)
  let reflected := reflect_across_y_eq_x Q
  let rotated := rotate_180_around reflected (1, 5)
  rotated = (-8, 2) → a - b = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1320_132071


namespace NUMINAMATH_CALUDE_complex_product_equals_five_l1320_132092

theorem complex_product_equals_five (a : ℝ) : 
  let z₁ : ℂ := -1 + 2*I
  let z₂ : ℂ := a - 2*I
  z₁ * z₂ = 5 → a = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_product_equals_five_l1320_132092


namespace NUMINAMATH_CALUDE_largest_package_size_l1320_132029

theorem largest_package_size (john_markers alice_markers : ℕ) 
  (h1 : john_markers = 36) (h2 : alice_markers = 60) : 
  Nat.gcd john_markers alice_markers = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l1320_132029


namespace NUMINAMATH_CALUDE_divisors_of_eight_n_cubed_l1320_132052

theorem divisors_of_eight_n_cubed (n : ℕ) (h_odd : Odd n) (h_divisors : (Nat.divisors n).card = 17) :
  (Nat.divisors (8 * n^3)).card = 196 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_eight_n_cubed_l1320_132052


namespace NUMINAMATH_CALUDE_salary_change_l1320_132066

theorem salary_change (original : ℝ) (h : original > 0) :
  let decreased := original * 0.5
  let increased := decreased * 1.5
  (original - increased) / original = 0.25 := by
sorry

end NUMINAMATH_CALUDE_salary_change_l1320_132066


namespace NUMINAMATH_CALUDE_two_students_know_same_number_l1320_132023

/-- Represents the number of students a given student knows -/
def StudentsKnown := Fin 81

/-- The set of all students in the course -/
def Students := Fin 81

theorem two_students_know_same_number (f : Students → StudentsKnown) :
  ∃ (i j : Students), i ≠ j ∧ f i = f j :=
sorry

end NUMINAMATH_CALUDE_two_students_know_same_number_l1320_132023


namespace NUMINAMATH_CALUDE_prism_volume_l1320_132080

/-- A right rectangular prism with face areas 45, 49, and 56 square units has a volume of 1470 cubic units. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 45) (h2 : b * c = 49) (h3 : a * c = 56) :
  a * b * c = 1470 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1320_132080


namespace NUMINAMATH_CALUDE_fraction_simplification_l1320_132084

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x - 2) / (x^2 - 2*x + 1) / (x / (x - 1)) + 1 / (x^2 - x) = 1 / x := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1320_132084


namespace NUMINAMATH_CALUDE_composite_ratio_theorem_l1320_132061

/-- The nth positive composite number -/
def nthComposite (n : ℕ) : ℕ := sorry

/-- The product of the first n positive composite numbers -/
def productFirstNComposites (n : ℕ) : ℕ := sorry

theorem composite_ratio_theorem : 
  (productFirstNComposites 7) / (productFirstNComposites 14 / productFirstNComposites 7) = 1 / 110 := by
  sorry

end NUMINAMATH_CALUDE_composite_ratio_theorem_l1320_132061


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l1320_132015

/-- 
Given two lines in the xy-plane:
- Line 1 with equation y = mx + 1
- Line 2 with equation y = 4x - 8
If Line 1 is perpendicular to Line 2, then m = -1/4
-/
theorem perpendicular_lines_slope (m : ℝ) : 
  (∃ (x y : ℝ), y = m * x + 1) →  -- Line 1 exists
  (∃ (x y : ℝ), y = 4 * x - 8) →  -- Line 2 exists
  (∀ (x₁ y₁ x₂ y₂ : ℝ), y₁ = m * x₁ + 1 → y₂ = 4 * x₂ - 8 → 
    (y₂ - y₁) * (x₂ - x₁) = -(x₂ - x₁) * (x₂ - x₁)) →  -- Lines are perpendicular
  m = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l1320_132015


namespace NUMINAMATH_CALUDE_exists_prescription_with_four_potent_l1320_132022

/-- Represents a type of medicine -/
structure Medicine :=
  (isPotent : Bool)

/-- Represents a prescription -/
structure Prescription :=
  (medicines : Finset Medicine)

/-- The set of all available medicines -/
def AllMedicines : Finset Medicine := sorry

/-- The set of all prescriptions -/
def AllPrescriptions : Finset Prescription := sorry

/-- Conditions of the problem -/
axiom total_prescriptions : Finset.card AllPrescriptions = 68

axiom medicines_per_prescription : 
  ∀ p : Prescription, p ∈ AllPrescriptions → Finset.card p.medicines = 5

axiom at_least_one_potent : 
  ∀ p : Prescription, p ∈ AllPrescriptions → 
    ∃ m : Medicine, m ∈ p.medicines ∧ m.isPotent

axiom three_medicines_in_one_prescription : 
  ∀ m₁ m₂ m₃ : Medicine, m₁ ∈ AllMedicines → m₂ ∈ AllMedicines → m₃ ∈ AllMedicines →
    m₁ ≠ m₂ → m₂ ≠ m₃ → m₁ ≠ m₃ →
    ∃! p : Prescription, p ∈ AllPrescriptions ∧ m₁ ∈ p.medicines ∧ m₂ ∈ p.medicines ∧ m₃ ∈ p.medicines

/-- The main theorem to prove -/
theorem exists_prescription_with_four_potent : 
  ∃ p : Prescription, p ∈ AllPrescriptions ∧ 
    (Finset.filter (fun m => m.isPotent) p.medicines).card ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_prescription_with_four_potent_l1320_132022


namespace NUMINAMATH_CALUDE_complex_average_calculation_l1320_132035

def avg2 (a b : ℚ) : ℚ := (a + b) / 2

def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem complex_average_calculation :
  avg4 (avg4 2 2 (-1) (avg2 1 3)) 7 (avg2 4 (5 - 2)) = 27 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_average_calculation_l1320_132035


namespace NUMINAMATH_CALUDE_bowknot_equation_solution_l1320_132025

-- Define the bowknot operation
noncomputable def bowknot (c d : ℝ) : ℝ :=
  c + Real.sqrt (d + Real.sqrt (d + Real.sqrt (d + Real.sqrt d)))

-- Theorem statement
theorem bowknot_equation_solution :
  ∃ x : ℝ, bowknot 3 x = 12 → x = 72 := by sorry

end NUMINAMATH_CALUDE_bowknot_equation_solution_l1320_132025


namespace NUMINAMATH_CALUDE_remainder_three_power_twenty_mod_five_l1320_132037

theorem remainder_three_power_twenty_mod_five : 3^20 ≡ 1 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_power_twenty_mod_five_l1320_132037


namespace NUMINAMATH_CALUDE_union_subset_intersection_implies_a_equals_one_l1320_132002

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

-- State the theorem
theorem union_subset_intersection_implies_a_equals_one (a : ℝ) :
  (A ∪ B a) ⊆ (A ∩ B a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_union_subset_intersection_implies_a_equals_one_l1320_132002


namespace NUMINAMATH_CALUDE_oil_cylinder_capacity_l1320_132063

theorem oil_cylinder_capacity (C : ℚ) 
  (h1 : (4/5 : ℚ) * C - (3/4 : ℚ) * C = 4) : C = 80 := by
  sorry

end NUMINAMATH_CALUDE_oil_cylinder_capacity_l1320_132063


namespace NUMINAMATH_CALUDE_matrix_sum_squares_invertible_l1320_132082

open Matrix

variable {n : ℕ}

/-- Given real n×n matrices M and N satisfying the conditions, M² + N² is invertible iff M and N are invertible -/
theorem matrix_sum_squares_invertible (M N : Matrix (Fin n) (Fin n) ℝ)
  (h_neq : M ≠ N)
  (h_cube : M^3 = N^3)
  (h_comm : M^2 * N = N^2 * M) :
  IsUnit (M^2 + N^2) ↔ IsUnit M ∧ IsUnit N := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_squares_invertible_l1320_132082


namespace NUMINAMATH_CALUDE_range_of_m_l1320_132009

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -Real.sqrt (4 - p.2^2)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 6}

-- Define the condition for points P and Q
def existsPQ (m : ℝ) : Prop :=
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ C ∧ Q ∈ l ∧
    (P.1 - m, P.2) + (Q.1 - m, Q.2) = (0, 0)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, existsPQ m → 2 ≤ m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1320_132009


namespace NUMINAMATH_CALUDE_bobby_final_paycheck_l1320_132047

/-- Represents Bobby's weekly paycheck calculation -/
def bobby_paycheck (salary : ℝ) (federal_tax_rate : ℝ) (state_tax_rate : ℝ) 
                   (health_insurance : ℝ) (life_insurance : ℝ) (parking_fee : ℝ) : ℝ :=
  salary - (federal_tax_rate * salary) - (state_tax_rate * salary) - 
  health_insurance - life_insurance - parking_fee

/-- Theorem stating that Bobby's final paycheck amount is $184 -/
theorem bobby_final_paycheck : 
  bobby_paycheck 450 (1/3) 0.08 50 20 10 = 184 := by
  sorry

end NUMINAMATH_CALUDE_bobby_final_paycheck_l1320_132047


namespace NUMINAMATH_CALUDE_pizza_cheese_calories_pizza_cheese_calories_proof_l1320_132039

theorem pizza_cheese_calories : ℝ → Prop :=
  fun cheese_calories =>
    let lettuce_calories : ℝ := 50
    let carrot_calories : ℝ := 2 * lettuce_calories
    let dressing_calories : ℝ := 210
    let salad_calories : ℝ := lettuce_calories + carrot_calories + dressing_calories
    let crust_calories : ℝ := 600
    let pepperoni_calories : ℝ := (1 / 3) * crust_calories
    let pizza_calories : ℝ := crust_calories + pepperoni_calories + cheese_calories
    let jackson_salad_portion : ℝ := 1 / 4
    let jackson_pizza_portion : ℝ := 1 / 5
    let jackson_consumed_calories : ℝ := 330
    jackson_salad_portion * salad_calories + jackson_pizza_portion * pizza_calories = jackson_consumed_calories →
    cheese_calories = 400

-- Proof
theorem pizza_cheese_calories_proof : pizza_cheese_calories 400 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cheese_calories_pizza_cheese_calories_proof_l1320_132039


namespace NUMINAMATH_CALUDE_greatest_k_value_l1320_132010

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 73) →
  k ≤ Real.sqrt 105 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l1320_132010


namespace NUMINAMATH_CALUDE_kittens_given_to_jessica_l1320_132045

theorem kittens_given_to_jessica (initial_kittens : ℕ) (kittens_to_sara : ℕ) (kittens_left : ℕ) :
  initial_kittens = 18 → kittens_to_sara = 6 → kittens_left = 9 →
  initial_kittens - kittens_to_sara - kittens_left = 3 :=
by sorry

end NUMINAMATH_CALUDE_kittens_given_to_jessica_l1320_132045


namespace NUMINAMATH_CALUDE_fg_squared_value_l1320_132072

-- Define the functions g and f
def g (x : ℝ) : ℝ := 4 * x + 5
def f (x : ℝ) : ℝ := 6 * x - 11

-- State the theorem
theorem fg_squared_value : (f (g 6))^2 = 26569 := by sorry

end NUMINAMATH_CALUDE_fg_squared_value_l1320_132072


namespace NUMINAMATH_CALUDE_max_integer_difference_l1320_132034

theorem max_integer_difference (x y : ℤ) (hx : 5 < x ∧ x < 8) (hy : 8 < y ∧ y < 13) :
  (∀ (a b : ℤ), 5 < a ∧ a < 8 ∧ 8 < b ∧ b < 13 → y - x ≥ b - a) ∧ y - x ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_integer_difference_l1320_132034


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l1320_132097

/-- Represents a rectangular field -/
structure RectangularField where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular field -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Calculates the perimeter of a rectangular field -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.width + field.length)

theorem rectangular_field_perimeter 
  (field : RectangularField) 
  (h_area : area field = 50) 
  (h_width : field.width = 5) : 
  perimeter field = 30 := by
  sorry

#check rectangular_field_perimeter

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l1320_132097


namespace NUMINAMATH_CALUDE_no_three_fractions_product_one_l1320_132024

theorem no_three_fractions_product_one :
  ¬ ∃ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 100 ∧
    (a : ℚ) / (101 - a) * (b : ℚ) / (101 - b) * (c : ℚ) / (101 - c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_three_fractions_product_one_l1320_132024


namespace NUMINAMATH_CALUDE_cake_recipe_flour_l1320_132060

theorem cake_recipe_flour (sugar cups_of_sugar : ℕ) (flour initial_flour : ℕ) :
  cups_of_sugar = 6 →
  initial_flour = 2 →
  flour = cups_of_sugar + 1 →
  flour = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_recipe_flour_l1320_132060


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l1320_132030

/-- Represents the average percentage decrease in price per reduction -/
def average_decrease : ℝ := 0.25

/-- The original price of the medicine in yuan -/
def original_price : ℝ := 16

/-- The current price of the medicine in yuan -/
def current_price : ℝ := 9

/-- The number of successive price reductions -/
def num_reductions : ℕ := 2

theorem medicine_price_reduction :
  current_price = original_price * (1 - average_decrease) ^ num_reductions :=
by sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l1320_132030


namespace NUMINAMATH_CALUDE_log_equation_solution_l1320_132087

/-- Proves that 56 is the solution to the logarithmic equation log_7(x) - 3log_7(2) = 1 -/
theorem log_equation_solution :
  ∃ x : ℝ, (Real.log x / Real.log 7) - 3 * (Real.log 2 / Real.log 7) = 1 ∧ x = 56 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1320_132087


namespace NUMINAMATH_CALUDE_principal_amount_l1320_132069

/-- Proves that given the specified conditions, the principal amount is 1300 --/
theorem principal_amount (P : ℝ) : 
  P * ((1 + 0.1)^2 - 1) - P * (0.1 * 2) = 13 → P = 1300 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l1320_132069


namespace NUMINAMATH_CALUDE_least_three_digit_product_6_l1320_132074

/-- A function that returns the product of the digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is three-digit -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_product_6 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 6 → 116 ≤ n := by sorry

end NUMINAMATH_CALUDE_least_three_digit_product_6_l1320_132074


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_six_l1320_132090

theorem sqrt_difference_equals_six :
  Real.sqrt (21 + 12 * Real.sqrt 3) - Real.sqrt (21 - 12 * Real.sqrt 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_six_l1320_132090


namespace NUMINAMATH_CALUDE_fraction_sum_l1320_132005

theorem fraction_sum : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1320_132005


namespace NUMINAMATH_CALUDE_mobile_phone_cost_mobile_phone_cost_is_8000_l1320_132096

/-- Proves that the cost of the mobile phone is 8000, given the conditions of the problem -/
theorem mobile_phone_cost : ℕ → Prop :=
  fun cost_mobile =>
    let cost_refrigerator : ℕ := 15000
    let loss_rate_refrigerator : ℚ := 2 / 100
    let profit_rate_mobile : ℚ := 10 / 100
    let overall_profit : ℕ := 500
    let selling_price_refrigerator : ℚ := cost_refrigerator * (1 - loss_rate_refrigerator)
    let selling_price_mobile : ℚ := cost_mobile * (1 + profit_rate_mobile)
    selling_price_refrigerator + selling_price_mobile - (cost_refrigerator + cost_mobile) = overall_profit →
    cost_mobile = 8000

/-- The cost of the mobile phone is 8000 -/
theorem mobile_phone_cost_is_8000 : mobile_phone_cost 8000 := by
  sorry

end NUMINAMATH_CALUDE_mobile_phone_cost_mobile_phone_cost_is_8000_l1320_132096


namespace NUMINAMATH_CALUDE_power_of_five_l1320_132004

theorem power_of_five (m : ℕ) : 5^m = 5 * 25^4 * 625^3 → m = 21 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_l1320_132004


namespace NUMINAMATH_CALUDE_cube_painting_theorem_l1320_132055

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 7

/-- The number of symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- The number of ways to paint a cube with different colors on each face -/
def paint_cube_ways : ℕ := (num_colors.factorial) / (num_colors - num_faces).factorial

/-- The number of distinct ways to paint a cube considering symmetries -/
def distinct_paint_ways : ℕ := paint_cube_ways / cube_symmetries

theorem cube_painting_theorem : distinct_paint_ways = 210 := by
  sorry

end NUMINAMATH_CALUDE_cube_painting_theorem_l1320_132055


namespace NUMINAMATH_CALUDE_volunteer_arrangement_count_l1320_132046

theorem volunteer_arrangement_count (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 3) :
  Nat.choose n k * Nat.choose (n - k) k = 140 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_arrangement_count_l1320_132046


namespace NUMINAMATH_CALUDE_emily_age_l1320_132093

/-- Represents the ages of people in the problem -/
structure Ages where
  alan : ℕ
  bob : ℕ
  carl : ℕ
  donna : ℕ
  emily : ℕ

/-- The age relationships in the problem -/
def valid_ages (ages : Ages) : Prop :=
  ages.alan = ages.bob - 4 ∧
  ages.bob = ages.carl + 5 ∧
  ages.donna = ages.carl + 2 ∧
  ages.emily = ages.alan + ages.donna - ages.bob

theorem emily_age (ages : Ages) :
  valid_ages ages → ages.bob = 20 → ages.emily = 13 := by
  sorry

end NUMINAMATH_CALUDE_emily_age_l1320_132093


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1320_132053

/-- Given a cone with slant height 4 and angle between the slant height and axis of rotation 30°,
    the lateral surface area of the cone is 8π. -/
theorem cone_lateral_surface_area (l : ℝ) (θ : ℝ) (h1 : l = 4) (h2 : θ = 30 * π / 180) :
  π * l * (l * Real.sin θ) = 8 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1320_132053


namespace NUMINAMATH_CALUDE_a_most_stable_l1320_132056

/-- Represents a participant in the shooting test -/
inductive Participant
  | A
  | B
  | C
  | D

/-- The variance of a participant's scores -/
def variance : Participant → ℝ
  | Participant.A => 0.12
  | Participant.B => 0.25
  | Participant.C => 0.35
  | Participant.D => 0.46

/-- A participant has the most stable performance if their variance is the lowest -/
def hasMostStablePerformance (p : Participant) : Prop :=
  ∀ q : Participant, variance p ≤ variance q

/-- Theorem: Participant A has the most stable performance -/
theorem a_most_stable : hasMostStablePerformance Participant.A := by
  sorry

end NUMINAMATH_CALUDE_a_most_stable_l1320_132056


namespace NUMINAMATH_CALUDE_sum_of_consecutive_even_integers_l1320_132049

/-- Three consecutive even integers where the sum of the first and third is 128 -/
structure ConsecutiveEvenIntegers where
  a : ℤ
  b : ℤ
  c : ℤ
  consecutive : b = a + 2 ∧ c = b + 2
  even : Even a
  sum_first_third : a + c = 128

/-- The sum of three consecutive even integers is 192 when the sum of the first and third is 128 -/
theorem sum_of_consecutive_even_integers (x : ConsecutiveEvenIntegers) : x.a + x.b + x.c = 192 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_even_integers_l1320_132049


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l1320_132020

theorem gcd_of_squares_sum : Nat.gcd (12^2 + 23^2 + 34^2) (13^2 + 22^2 + 35^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l1320_132020


namespace NUMINAMATH_CALUDE_investment_difference_is_1000_l1320_132013

/-- Represents the investment problem with three persons --/
structure InvestmentProblem where
  total_investment : ℕ
  total_gain : ℕ
  third_person_gain : ℕ

/-- Calculates the investment difference between the second and first person --/
def investment_difference (problem : InvestmentProblem) : ℕ :=
  let first_investment := problem.total_investment / 3
  let second_investment := first_investment + (problem.total_investment / 3 - first_investment)
  second_investment - first_investment

/-- Theorem stating that the investment difference is 1000 for the given problem --/
theorem investment_difference_is_1000 (problem : InvestmentProblem) 
  (h1 : problem.total_investment = 9000)
  (h2 : problem.total_gain = 1800)
  (h3 : problem.third_person_gain = 800) :
  investment_difference problem = 1000 := by
  sorry

#eval investment_difference ⟨9000, 1800, 800⟩

end NUMINAMATH_CALUDE_investment_difference_is_1000_l1320_132013


namespace NUMINAMATH_CALUDE_negative_three_to_zero_power_l1320_132094

theorem negative_three_to_zero_power : (-3 : ℤ) ^ (0 : ℕ) = 1 := by sorry

end NUMINAMATH_CALUDE_negative_three_to_zero_power_l1320_132094


namespace NUMINAMATH_CALUDE_sequence_general_term_l1320_132088

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ := 2^(n-1)

theorem sequence_general_term (n : ℕ) (h : n ≥ 1) : 
  a n = S n - S (n-1) :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1320_132088


namespace NUMINAMATH_CALUDE_garden_perimeter_l1320_132083

/-- The perimeter of a rectangular garden with width 8 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 64 meters. -/
theorem garden_perimeter : 
  let playground_length : ℝ := 16
  let playground_width : ℝ := 12
  let playground_area : ℝ := playground_length * playground_width
  let garden_width : ℝ := 8
  let garden_length : ℝ := playground_area / garden_width
  let garden_perimeter : ℝ := 2 * (garden_length + garden_width)
  garden_perimeter = 64 := by sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1320_132083


namespace NUMINAMATH_CALUDE_unique_four_letter_product_l1320_132017

def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5
  | 'F' => 6 | 'G' => 7 | 'H' => 8 | 'I' => 9 | 'J' => 10
  | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14 | 'O' => 15
  | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25
  | 'Z' => 26
  | _ => 0

def list_product (s : String) : ℕ :=
  s.foldl (fun acc c => acc * letter_value c) 1

def is_valid_four_letter_string (s : String) : Prop :=
  s.length = 4 ∧ s.all (fun c => 'A' ≤ c ∧ c ≤ 'Z')

theorem unique_four_letter_product :
  ∀ s : String, is_valid_four_letter_string s →
    list_product s = list_product "TUVW" →
    s = "TUVW" := by sorry

#check unique_four_letter_product

end NUMINAMATH_CALUDE_unique_four_letter_product_l1320_132017


namespace NUMINAMATH_CALUDE_sum_of_proportional_values_l1320_132012

theorem sum_of_proportional_values (a b c d e f : ℝ) 
  (h1 : a / b = 4 / 3)
  (h2 : c / d = 4 / 3)
  (h3 : e / f = 4 / 3)
  (h4 : b + d + f = 15) :
  a + c + e = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_proportional_values_l1320_132012


namespace NUMINAMATH_CALUDE_area_of_second_square_l1320_132042

-- Define the circle
def Circle : Type := Unit

-- Define squares
structure Square where
  area : ℝ

-- Define the inscribed square
def inscribed_square (c : Circle) (s : Square) : Prop :=
  s.area = 16

-- Define the second square
def second_square (c : Circle) (s1 s2 : Square) : Prop :=
  -- Vertices E and F are on sides of s1, G and H are on the circle
  True

-- Theorem statement
theorem area_of_second_square 
  (c : Circle) 
  (s1 s2 : Square) 
  (h1 : inscribed_square c s1) 
  (h2 : second_square c s1 s2) : 
  s2.area = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_of_second_square_l1320_132042


namespace NUMINAMATH_CALUDE_max_cables_theorem_l1320_132098

/-- Represents a computer network with two brands of computers. -/
structure ComputerNetwork where
  brandA : ℕ  -- Number of brand A computers
  brandB : ℕ  -- Number of brand B computers

/-- Calculates the maximum number of cables that can be used in the network. -/
def maxCables (network : ComputerNetwork) : ℕ :=
  network.brandA * network.brandB

/-- Theorem: The maximum number of cables in a network with 25 brand A and 15 brand B computers is 361. -/
theorem max_cables_theorem (network : ComputerNetwork) 
  (h1 : network.brandA = 25) 
  (h2 : network.brandB = 15) : 
  maxCables network = 361 := by
  sorry

#eval maxCables { brandA := 25, brandB := 15 }

end NUMINAMATH_CALUDE_max_cables_theorem_l1320_132098


namespace NUMINAMATH_CALUDE_positive_integer_solution_exists_l1320_132078

theorem positive_integer_solution_exists : 
  ∃ (x y z t : ℕ+), x + y + z + t = 10 :=
by sorry

#check positive_integer_solution_exists

end NUMINAMATH_CALUDE_positive_integer_solution_exists_l1320_132078


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1320_132016

theorem necessary_but_not_sufficient_condition 
  (A B C : Set α) 
  (hAnonempty : A.Nonempty) 
  (hBnonempty : B.Nonempty) 
  (hCnonempty : C.Nonempty) 
  (hUnion : A ∪ B = C) 
  (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ x, x ∈ C ∧ x ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1320_132016


namespace NUMINAMATH_CALUDE_smaller_rss_better_fit_regression_line_passes_through_center_l1320_132028

/-- Represents a linear regression model -/
structure LinearRegression where
  x : List ℝ  -- Independent variable data
  y : List ℝ  -- Dependent variable data
  β : ℝ       -- Slope of the regression line
  α : ℝ       -- Intercept of the regression line

/-- Calculates the residual sum of squares for a linear regression model -/
def residualSumOfSquares (model : LinearRegression) : ℝ :=
  sorry

/-- Calculates the mean of a list of real numbers -/
def mean (data : List ℝ) : ℝ :=
  sorry

/-- Theorem stating that a smaller residual sum of squares indicates a better fitting effect -/
theorem smaller_rss_better_fit (model1 model2 : LinearRegression) :
  residualSumOfSquares model1 < residualSumOfSquares model2 →
  -- The fitting effect of model1 is better than model2
  sorry :=
sorry

/-- Theorem stating that the linear regression equation passes through the center point (x̄, ȳ) of the sample -/
theorem regression_line_passes_through_center (model : LinearRegression) :
  let x_mean := mean model.x
  let y_mean := mean model.y
  model.α + model.β * x_mean = y_mean :=
sorry

end NUMINAMATH_CALUDE_smaller_rss_better_fit_regression_line_passes_through_center_l1320_132028


namespace NUMINAMATH_CALUDE_smallest_brownie_pan_dimension_l1320_132036

def is_valid_brownie_pan (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ (m - 2) * (n - 2) = 2 * (2 * m + 2 * n - 4)

def smallest_dimension : ℕ := 12

theorem smallest_brownie_pan_dimension :
  (is_valid_brownie_pan smallest_dimension smallest_dimension) ∧
  (∀ k : ℕ, k < smallest_dimension → ¬(is_valid_brownie_pan k k) ∧ ¬(∃ l : ℕ, is_valid_brownie_pan k l ∨ is_valid_brownie_pan l k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_brownie_pan_dimension_l1320_132036


namespace NUMINAMATH_CALUDE_parallel_implies_alternate_interior_angles_vertical_angles_are_equal_right_triangle_acute_angles_complementary_supplements_of_same_angle_are_equal_inverse_of_vertical_angles_false_others_true_l1320_132006

-- Define the basic concepts
def Line : Type := sorry
def Angle : Type := sorry
def Triangle : Type := sorry

-- Define the properties
def parallel (l1 l2 : Line) : Prop := sorry
def alternateInteriorAngles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry
def verticalAngles (a1 a2 : Angle) : Prop := sorry
def rightTriangle (t : Triangle) : Prop := sorry
def acuteAngles (t : Triangle) (a1 a2 : Angle) : Prop := sorry
def complementaryAngles (a1 a2 : Angle) : Prop := sorry
def supplementaryAngles (a1 a2 : Angle) : Prop := sorry

-- Theorem A
theorem parallel_implies_alternate_interior_angles (l1 l2 : Line) (a1 a2 : Angle) :
  parallel l1 l2 → alternateInteriorAngles a1 a2 l1 l2 := sorry

-- Theorem B
theorem vertical_angles_are_equal (a1 a2 : Angle) :
  verticalAngles a1 a2 → a1 = a2 := sorry

-- Theorem C
theorem right_triangle_acute_angles_complementary (t : Triangle) (a1 a2 : Angle) :
  rightTriangle t → acuteAngles t a1 a2 → complementaryAngles a1 a2 := sorry

-- Theorem D
theorem supplements_of_same_angle_are_equal (a1 a2 a3 : Angle) :
  supplementaryAngles a1 a3 → supplementaryAngles a2 a3 → a1 = a2 := sorry

-- The main theorem: inverse of B is false, while inverses of A, C, and D are true
theorem inverse_of_vertical_angles_false_others_true :
  (∃ a1 a2 : Angle, a1 = a2 ∧ ¬verticalAngles a1 a2) ∧
  (∀ l1 l2 : Line, ∀ a1 a2 : Angle, alternateInteriorAngles a1 a2 l1 l2 → parallel l1 l2) ∧
  (∀ a1 a2 : Angle, complementaryAngles a1 a2 → ∃ t : Triangle, rightTriangle t ∧ acuteAngles t a1 a2) ∧
  (∀ a1 a2 a3 : Angle, a1 = a2 → supplementaryAngles a1 a3 → supplementaryAngles a2 a3) := sorry

end NUMINAMATH_CALUDE_parallel_implies_alternate_interior_angles_vertical_angles_are_equal_right_triangle_acute_angles_complementary_supplements_of_same_angle_are_equal_inverse_of_vertical_angles_false_others_true_l1320_132006


namespace NUMINAMATH_CALUDE_esme_school_non_pizza_eaters_l1320_132007

/-- The number of teachers at Esme's school -/
def num_teachers : ℕ := 30

/-- The number of staff members at Esme's school -/
def num_staff : ℕ := 45

/-- The fraction of teachers who ate pizza -/
def teacher_pizza_fraction : ℚ := 2/3

/-- The fraction of staff members who ate pizza -/
def staff_pizza_fraction : ℚ := 4/5

/-- The total number of non-pizza eaters at Esme's school -/
def non_pizza_eaters : ℕ := 19

theorem esme_school_non_pizza_eaters :
  (num_teachers - (num_teachers : ℚ) * teacher_pizza_fraction).floor +
  (num_staff - (num_staff : ℚ) * staff_pizza_fraction).floor = non_pizza_eaters := by
  sorry

end NUMINAMATH_CALUDE_esme_school_non_pizza_eaters_l1320_132007


namespace NUMINAMATH_CALUDE_triangle_not_right_angle_l1320_132076

theorem triangle_not_right_angle (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : A = 2 * B) (h3 : A = 3 * C) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_not_right_angle_l1320_132076


namespace NUMINAMATH_CALUDE_migraine_expectation_l1320_132067

/-- The fraction of Canadians suffering from migraines -/
def migraine_fraction : ℚ := 2 / 7

/-- The total number of Canadians in the sample -/
def sample_size : ℕ := 350

/-- The expected number of Canadians in the sample suffering from migraines -/
def expected_migraines : ℕ := 100

theorem migraine_expectation :
  (migraine_fraction * sample_size : ℚ) = expected_migraines := by sorry

end NUMINAMATH_CALUDE_migraine_expectation_l1320_132067


namespace NUMINAMATH_CALUDE_isosceles_triangle_special_angles_l1320_132079

/-- An isosceles triangle with vertex angle twice the base angle has a 90° vertex angle and 45° base angles. -/
theorem isosceles_triangle_special_angles :
  ∀ (vertex_angle base_angle : ℝ),
    vertex_angle > 0 →
    base_angle > 0 →
    vertex_angle = 2 * base_angle →
    vertex_angle + 2 * base_angle = 180 →
    vertex_angle = 90 ∧ base_angle = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_special_angles_l1320_132079


namespace NUMINAMATH_CALUDE_multiple_properties_l1320_132050

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a + b = 4 * p) ∧ 
  (∃ q : ℤ, a + b = 2 * q) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l1320_132050


namespace NUMINAMATH_CALUDE_min_filters_correct_l1320_132011

/-- The minimum number of filters required to reduce impurities -/
def min_filters : ℕ := 5

/-- The initial impurity concentration -/
def initial_impurity : ℝ := 0.2

/-- The fraction of impurities remaining after each filter -/
def filter_efficiency : ℝ := 0.2

/-- The maximum allowed final impurity concentration -/
def max_final_impurity : ℝ := 0.0001

/-- Theorem stating that min_filters is the minimum number of filters required -/
theorem min_filters_correct :
  (initial_impurity * filter_efficiency ^ min_filters ≤ max_final_impurity) ∧
  (∀ k : ℕ, k < min_filters → initial_impurity * filter_efficiency ^ k > max_final_impurity) :=
sorry

end NUMINAMATH_CALUDE_min_filters_correct_l1320_132011


namespace NUMINAMATH_CALUDE_quadratic_trinomial_decomposition_l1320_132059

/-- Any quadratic trinomial can be represented as the sum of two quadratic trinomials with zero discriminants -/
theorem quadratic_trinomial_decomposition (a b c : ℝ) :
  ∃ (p q r s t u : ℝ), 
    (∀ x, a * x^2 + b * x + c = (p * x^2 + q * x + r) + (s * x^2 + t * x + u)) ∧
    (q^2 - 4 * p * r = 0) ∧
    (t^2 - 4 * s * u = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_decomposition_l1320_132059


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l1320_132091

theorem largest_three_digit_congruence :
  ∀ n : ℕ,
  n ≤ 998 →
  100 ≤ n →
  n ≤ 999 →
  70 * n ≡ 210 [MOD 350] →
  ∃ m : ℕ,
  m = 998 ∧
  70 * m ≡ 210 [MOD 350] ∧
  ∀ k : ℕ,
  100 ≤ k →
  k ≤ 999 →
  70 * k ≡ 210 [MOD 350] →
  k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l1320_132091


namespace NUMINAMATH_CALUDE_systematic_sampling_last_id_l1320_132000

theorem systematic_sampling_last_id 
  (total_students : Nat) 
  (sample_size : Nat) 
  (first_id : Nat) 
  (h1 : total_students = 2000) 
  (h2 : sample_size = 50) 
  (h3 : first_id = 3) :
  let interval := total_students / sample_size
  let last_id := first_id + interval * (sample_size - 1)
  last_id = 1963 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_last_id_l1320_132000


namespace NUMINAMATH_CALUDE_jamie_water_bottle_limit_l1320_132085

/-- The maximum amount of liquid Jamie can consume before needing the bathroom -/
def bathroom_limit : ℕ := 32

/-- The amount of milk Jamie consumed -/
def milk_consumed : ℕ := 8

/-- The amount of grape juice Jamie consumed -/
def grape_juice_consumed : ℕ := 16

/-- The amount Jamie can drink from her water bottle during the test -/
def water_bottle_limit : ℕ := bathroom_limit - (milk_consumed + grape_juice_consumed)

theorem jamie_water_bottle_limit :
  water_bottle_limit = 8 :=
by sorry

end NUMINAMATH_CALUDE_jamie_water_bottle_limit_l1320_132085


namespace NUMINAMATH_CALUDE_distinct_bracelets_count_l1320_132040

/-- Represents a bracelet configuration -/
structure Bracelet :=
  (red : Nat)
  (blue : Nat)
  (green : Nat)

/-- Defines the specific bracelet configuration in the problem -/
def problem_bracelet : Bracelet :=
  { red := 1, blue := 2, green := 2 }

/-- Calculates the total number of beads in a bracelet -/
def total_beads (b : Bracelet) : Nat :=
  b.red + b.blue + b.green

/-- Represents the number of distinct bracelets -/
def distinct_bracelets (b : Bracelet) : Nat :=
  (Nat.factorial (total_beads b)) / 
  (Nat.factorial b.red * Nat.factorial b.blue * Nat.factorial b.green * 
   (total_beads b) * 2)

/-- Theorem stating that the number of distinct bracelets for the given configuration is 4 -/
theorem distinct_bracelets_count :
  distinct_bracelets problem_bracelet = 4 := by
  sorry

end NUMINAMATH_CALUDE_distinct_bracelets_count_l1320_132040


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1320_132057

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h_different_lines : m ≠ n)
  (h_different_planes : α ≠ β)
  (h_n_subset_β : subset n β)
  (h_m_parallel_n : parallel m n)
  (h_m_perp_α : perpendicular m α) :
  perpendicular_planes α β := by
  sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1320_132057


namespace NUMINAMATH_CALUDE_cricketer_wickets_after_match_l1320_132031

/-- Represents a cricketer's bowling statistics -/
structure Cricketer where
  initialAverage : ℝ
  initialWickets : ℕ
  matchWickets : ℕ
  matchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the total number of wickets taken by a cricketer after a match -/
def totalWicketsAfterMatch (c : Cricketer) : ℕ :=
  c.initialWickets + c.matchWickets

/-- Theorem stating that for a cricketer with given statistics, the total wickets after the match is 90 -/
theorem cricketer_wickets_after_match (c : Cricketer) 
  (h1 : c.initialAverage = 12.4)
  (h2 : c.matchWickets = 5)
  (h3 : c.matchRuns = 26)
  (h4 : c.averageDecrease = 0.4) :
  totalWicketsAfterMatch c = 90 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_wickets_after_match_l1320_132031


namespace NUMINAMATH_CALUDE_select_perfect_square_l1320_132054

theorem select_perfect_square (nums : Finset ℕ) (h_card : nums.card = 48) 
  (h_prime_factors : (nums.prod id).factors.toFinset.card = 10) :
  ∃ (a b c d : ℕ), a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ (m : ℕ), a * b * c * d = m ^ 2 := by
sorry


end NUMINAMATH_CALUDE_select_perfect_square_l1320_132054


namespace NUMINAMATH_CALUDE_prime_factorization_sum_l1320_132008

theorem prime_factorization_sum (a b c : ℕ) : 
  2^a * 3^b * 7^c = 432 → a + b + c = 5 → 3*a + 2*b + 4*c = 18 := by
sorry

end NUMINAMATH_CALUDE_prime_factorization_sum_l1320_132008


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l1320_132019

/-- A line defined by y = (m-2)x + m, where 0 < m < 2, does not pass through the third quadrant. -/
theorem line_not_in_third_quadrant (m : ℝ) (h : 0 < m ∧ m < 2) :
  ∃ (x y : ℝ), y = (m - 2) * x + m → ¬(x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l1320_132019


namespace NUMINAMATH_CALUDE_mike_buys_36_games_l1320_132077

/-- Represents the number of days Mike worked --/
def total_days : ℕ := 20

/-- Represents the earnings per lawn in dollars --/
def earnings_per_lawn : ℕ := 5

/-- Represents the number of lawns mowed on a weekday --/
def lawns_per_weekday : ℕ := 2

/-- Represents the number of lawns mowed on a weekend day --/
def lawns_per_weekend : ℕ := 3

/-- Represents the cost of new mower blades in dollars --/
def cost_of_blades : ℕ := 24

/-- Represents the cost of gasoline in dollars --/
def cost_of_gas : ℕ := 15

/-- Represents the cost of each game in dollars --/
def cost_per_game : ℕ := 5

/-- Calculates the number of games Mike can buy --/
def games_mike_can_buy : ℕ :=
  let weekdays := 16
  let weekend_days := 4
  let total_lawns := weekdays * lawns_per_weekday + weekend_days * lawns_per_weekend
  let total_earnings := total_lawns * earnings_per_lawn
  let total_expenses := cost_of_blades + cost_of_gas
  let money_left := total_earnings - total_expenses
  money_left / cost_per_game

/-- Theorem stating that Mike can buy 36 games --/
theorem mike_buys_36_games : games_mike_can_buy = 36 := by
  sorry

end NUMINAMATH_CALUDE_mike_buys_36_games_l1320_132077


namespace NUMINAMATH_CALUDE_sequence_product_l1320_132048

/-- An arithmetic sequence with first term -9 and last term -1 -/
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₁ = -9 + d ∧ a₂ = a₁ + d ∧ -1 = a₂ + d

/-- A geometric sequence with first term -9 and last term -1 -/
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ = -9 * r ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r ∧ -1 = b₃ * r

theorem sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) 
  (h₁ : arithmetic_sequence a₁ a₂)
  (h₂ : geometric_sequence b₁ b₂ b₃) :
  b₂ * (a₂ - a₁) = -8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l1320_132048
