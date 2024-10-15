import Mathlib

namespace NUMINAMATH_CALUDE_no_valid_arrangement_l2599_259944

-- Define the set of people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person

-- Define a seating arrangement as a function from Person to ℕ (seat number)
def SeatingArrangement := Person → Fin 5

-- Define the adjacency relation for a circular table
def adjacent (s : SeatingArrangement) (p1 p2 : Person) : Prop :=
  (s p1 - s p2 = 1) ∨ (s p2 - s p1 = 1) ∨ (s p1 = 4 ∧ s p2 = 0) ∨ (s p1 = 0 ∧ s p2 = 4)

-- Define the seating restrictions
def validArrangement (s : SeatingArrangement) : Prop :=
  (¬ adjacent s Person.Alice Person.Bob) ∧
  (¬ adjacent s Person.Alice Person.Carla) ∧
  (¬ adjacent s Person.Derek Person.Eric) ∧
  (¬ adjacent s Person.Carla Person.Derek) ∧
  Function.Injective s

-- Theorem stating that no valid seating arrangement exists
theorem no_valid_arrangement : ¬ ∃ s : SeatingArrangement, validArrangement s := by
  sorry


end NUMINAMATH_CALUDE_no_valid_arrangement_l2599_259944


namespace NUMINAMATH_CALUDE_hundred_with_five_twos_l2599_259903

theorem hundred_with_five_twos :
  (222 / 2) - (22 / 2) = 100 :=
by sorry

end NUMINAMATH_CALUDE_hundred_with_five_twos_l2599_259903


namespace NUMINAMATH_CALUDE_course_selection_theorem_l2599_259935

def total_course_selection_plans (n : ℕ) (k₁ k₂ : ℕ) : ℕ :=
  (n.choose k₁) * (n.choose k₂) * (n.choose k₂)

theorem course_selection_theorem :
  total_course_selection_plans 4 2 3 = 96 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l2599_259935


namespace NUMINAMATH_CALUDE_olaf_initial_cars_l2599_259968

/-- The number of toy cars Olaf's uncle gave him -/
def uncle_cars : ℕ := 5

/-- The number of toy cars Olaf's grandpa gave him -/
def grandpa_cars : ℕ := 2 * uncle_cars

/-- The number of toy cars Olaf's dad gave him -/
def dad_cars : ℕ := 10

/-- The number of toy cars Olaf's mum gave him -/
def mum_cars : ℕ := dad_cars + 5

/-- The number of toy cars Olaf's auntie gave him -/
def auntie_cars : ℕ := 6

/-- The total number of toy cars Olaf has after receiving gifts -/
def total_cars : ℕ := 196

/-- The number of toy cars Olaf had initially -/
def initial_cars : ℕ := total_cars - (grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars)

theorem olaf_initial_cars : initial_cars = 150 := by
  sorry

end NUMINAMATH_CALUDE_olaf_initial_cars_l2599_259968


namespace NUMINAMATH_CALUDE_hyperbola_iff_ab_neg_l2599_259995

/-- A curve in the xy-plane -/
structure Curve where
  equation : ℝ → ℝ → Prop

/-- Definition of a hyperbola -/
def is_hyperbola (c : Curve) : Prop := sorry

/-- The specific curve ax^2 + by^2 = 1 -/
def quadratic_curve (a b : ℝ) : Curve where
  equation := fun x y => a * x^2 + b * y^2 = 1

/-- Theorem stating that ab < 0 is both necessary and sufficient for the curve to be a hyperbola -/
theorem hyperbola_iff_ab_neg (a b : ℝ) :
  is_hyperbola (quadratic_curve a b) ↔ a * b < 0 := by sorry

end NUMINAMATH_CALUDE_hyperbola_iff_ab_neg_l2599_259995


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2599_259920

def A : Set ℕ := {x : ℕ | x^2 - 5*x ≤ 0}
def B : Set ℕ := {0, 2, 5, 7}

theorem intersection_of_A_and_B : A ∩ B = {0, 2, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2599_259920


namespace NUMINAMATH_CALUDE_student_pet_difference_l2599_259933

/-- Represents a fourth-grade classroom -/
structure Classroom where
  students : ℕ
  rabbits : ℕ
  birds : ℕ

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- A fourth-grade classroom at Green Park Elementary -/
def green_park_classroom : Classroom := {
  students := 22,
  rabbits := 3,
  birds := 2
}

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * green_park_classroom.students

/-- The total number of pets (rabbits and birds) in all classrooms -/
def total_pets : ℕ := num_classrooms * (green_park_classroom.rabbits + green_park_classroom.birds)

/-- Theorem: The difference between the total number of students and the total number of pets is 85 -/
theorem student_pet_difference : total_students - total_pets = 85 := by
  sorry

end NUMINAMATH_CALUDE_student_pet_difference_l2599_259933


namespace NUMINAMATH_CALUDE_exists_subset_with_common_gcd_l2599_259947

/-- A function that checks if a number is the product of at most 1987 prime factors -/
def is_valid_element (n : ℕ) : Prop := ∃ (factors : List ℕ), n = factors.prod ∧ factors.all Nat.Prime ∧ factors.length ≤ 1987

/-- The set A of integers, each being a product of at most 1987 prime factors -/
def A : Set ℕ := {n | is_valid_element n}

/-- The theorem to be proved -/
theorem exists_subset_with_common_gcd (h : Set.Infinite A) :
  ∃ (B : Set ℕ) (b : ℕ), Set.Infinite B ∧ B ⊆ A ∧ b > 0 ∧
  ∀ (x y : ℕ), x ∈ B → y ∈ B → Nat.gcd x y = b :=
sorry

end NUMINAMATH_CALUDE_exists_subset_with_common_gcd_l2599_259947


namespace NUMINAMATH_CALUDE_power_of_seven_mod_ten_thousand_l2599_259989

theorem power_of_seven_mod_ten_thousand :
  7^2045 % 10000 = 6807 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_ten_thousand_l2599_259989


namespace NUMINAMATH_CALUDE_ellipse_m_value_l2599_259945

/-- Represents an ellipse with equation x^2 + my^2 = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ x y : ℝ, x^2 + m * y^2 = 1

/-- Indicates that the foci of the ellipse are on the y-axis -/
def foci_on_y_axis (e : Ellipse m) : Prop :=
  ∃ c : ℝ, c^2 = 1/m - 1 ∧ c ≥ 0

/-- The length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (e : Ellipse m) : Prop :=
  2 * (1 : ℝ) = Real.sqrt (1/m)

/-- Theorem stating that m = 1/4 for the given ellipse properties -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m)
  (h1 : foci_on_y_axis e)
  (h2 : major_axis_twice_minor e) :
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l2599_259945


namespace NUMINAMATH_CALUDE_inequality_proof_l2599_259908

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2599_259908


namespace NUMINAMATH_CALUDE_initial_charge_is_3_5_l2599_259939

/-- A taxi company's pricing model -/
structure TaxiCompany where
  initialCharge : ℝ  -- Initial charge for the first 1/5 mile
  additionalCharge : ℝ  -- Charge for each additional 1/5 mile
  totalCharge : ℝ  -- Total charge for a specific ride
  rideLength : ℝ  -- Length of the ride in miles

/-- The initial charge for the first 1/5 mile is $3.5 -/
theorem initial_charge_is_3_5 (t : TaxiCompany) 
    (h1 : t.additionalCharge = 0.4)
    (h2 : t.totalCharge = 19.1)
    (h3 : t.rideLength = 8) : 
    t.initialCharge = 3.5 := by
  sorry

#check initial_charge_is_3_5

end NUMINAMATH_CALUDE_initial_charge_is_3_5_l2599_259939


namespace NUMINAMATH_CALUDE_solve_equation_l2599_259904

theorem solve_equation (y : ℚ) (h : 3 * y - 9 = -6 * y + 3) : y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2599_259904


namespace NUMINAMATH_CALUDE_angle_triple_complement_measure_l2599_259923

theorem angle_triple_complement_measure :
  ∀ x : ℝ, 
    (x = 3 * (90 - x)) → 
    x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_measure_l2599_259923


namespace NUMINAMATH_CALUDE_min_teams_non_negative_balance_l2599_259979

/-- Represents the number of wins in a series --/
inductive SeriesScore
| Four_Zero
| Four_One
| Four_Two
| Four_Three

/-- Represents a team's performance in the tournament --/
structure TeamPerformance where
  wins : ℕ
  losses : ℕ

/-- Represents the NHL playoff tournament --/
structure NHLPlayoffs where
  num_teams : ℕ
  num_rounds : ℕ
  series_scores : List SeriesScore

/-- Defines a non-negative balance of wins --/
def has_non_negative_balance (team : TeamPerformance) : Prop :=
  team.wins ≥ team.losses

/-- Theorem stating the minimum number of teams with non-negative balance --/
theorem min_teams_non_negative_balance (playoffs : NHLPlayoffs) 
  (h1 : playoffs.num_teams = 16)
  (h2 : playoffs.num_rounds = 4)
  (h3 : ∀ s ∈ playoffs.series_scores, s ∈ [SeriesScore.Four_Zero, SeriesScore.Four_One, SeriesScore.Four_Two, SeriesScore.Four_Three]) :
  ∃ (teams : List TeamPerformance), 
    (∀ team ∈ teams, has_non_negative_balance team) ∧ 
    (teams.length = 2) ∧
    (∀ (n : ℕ), n < 2 → ¬∃ (teams' : List TeamPerformance), 
      (∀ team ∈ teams', has_non_negative_balance team) ∧ 
      (teams'.length = n)) :=
by sorry

end NUMINAMATH_CALUDE_min_teams_non_negative_balance_l2599_259979


namespace NUMINAMATH_CALUDE_expression_evaluation_l2599_259983

theorem expression_evaluation (x : ℝ) (hx : x^2 - 2*x - 3 = 0) (hx_neq : x ≠ 3) :
  (2 / (x - 3) - 1 / x) * ((x^2 - 3*x) / (x^2 + 6*x + 9)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2599_259983


namespace NUMINAMATH_CALUDE_furniture_markup_l2599_259949

/-- Given a selling price and a cost price, calculate the percentage markup -/
def percentageMarkup (sellingPrice costPrice : ℕ) : ℚ :=
  ((sellingPrice - costPrice : ℚ) / costPrice) * 100

theorem furniture_markup :
  percentageMarkup 5750 5000 = 15 := by sorry

end NUMINAMATH_CALUDE_furniture_markup_l2599_259949


namespace NUMINAMATH_CALUDE_clock_correction_theorem_l2599_259946

/-- The number of days between March 1st at noon and March 10th at 6 P.M. -/
def days_passed : ℚ := 9 + 6/24

/-- The rate at which the clock loses time, in minutes per day -/
def loss_rate : ℚ := 15

/-- The function to calculate the positive correction in minutes -/
def correction (d : ℚ) (r : ℚ) : ℚ := d * r

/-- Theorem stating that the positive correction needed is 138.75 minutes -/
theorem clock_correction_theorem :
  correction days_passed loss_rate = 138.75 := by sorry

end NUMINAMATH_CALUDE_clock_correction_theorem_l2599_259946


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_5_pow_1993_l2599_259997

/-- The rightmost three digits of 5^1993 are 125 -/
theorem rightmost_three_digits_of_5_pow_1993 : 5^1993 % 1000 = 125 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_5_pow_1993_l2599_259997


namespace NUMINAMATH_CALUDE_jet_flight_time_l2599_259928

theorem jet_flight_time (distance : ℝ) (time_with_wind : ℝ) (wind_speed : ℝ) 
  (h1 : distance = 2000)
  (h2 : time_with_wind = 4)
  (h3 : wind_speed = 50)
  : ∃ (jet_speed : ℝ), 
    (jet_speed + wind_speed) * time_with_wind = distance ∧
    distance / (jet_speed - wind_speed) = 5 := by
  sorry

end NUMINAMATH_CALUDE_jet_flight_time_l2599_259928


namespace NUMINAMATH_CALUDE_village_chief_assistants_l2599_259917

theorem village_chief_assistants (n : ℕ) (k : ℕ) (a b c : Fin n) (h1 : n = 10) (h2 : k = 3) :
  let total_combinations := Nat.choose n k
  let combinations_without_ab := Nat.choose (n - 2) k
  total_combinations - combinations_without_ab = 49 :=
sorry

end NUMINAMATH_CALUDE_village_chief_assistants_l2599_259917


namespace NUMINAMATH_CALUDE_dividing_line_theorem_l2599_259982

/-- Represents a disk in 2D space -/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of five disks -/
structure DiskConfiguration where
  disks : Fin 5 → Disk
  square_vertices : Fin 4 → ℝ × ℝ
  aligned_centers : Fin 3 → ℝ × ℝ

/-- Represents a line in 2D space -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The center of a square given its vertices -/
def square_center (vertices : Fin 4 → ℝ × ℝ) : ℝ × ℝ := sorry

/-- Calculates the area of the figure formed by the disks on one side of a line -/
def area_on_side (config : DiskConfiguration) (line : Line) : ℝ := sorry

/-- States that the line passing through the square center and the fifth disk's center
    divides the total area of the five disks into two equal parts -/
theorem dividing_line_theorem (config : DiskConfiguration) :
  let square_center := square_center config.square_vertices
  let fifth_disk_center := (config.disks 4).center
  let dividing_line := Line.mk square_center fifth_disk_center
  area_on_side config dividing_line = (area_on_side config dividing_line) / 2 := by sorry

end NUMINAMATH_CALUDE_dividing_line_theorem_l2599_259982


namespace NUMINAMATH_CALUDE_shekars_social_studies_score_l2599_259980

/-- Given Shekar's scores in four subjects and his average marks across all five subjects,
    prove that his marks in social studies must be 82. -/
theorem shekars_social_studies_score
  (math_score : ℕ)
  (science_score : ℕ)
  (english_score : ℕ)
  (biology_score : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 47)
  (h4 : biology_score = 85)
  (h5 : average_marks = 71)
  (h6 : num_subjects = 5)
  : ∃ (social_studies_score : ℕ),
    social_studies_score = 82 ∧
    (math_score + science_score + english_score + biology_score + social_studies_score) / num_subjects = average_marks :=
by
  sorry


end NUMINAMATH_CALUDE_shekars_social_studies_score_l2599_259980


namespace NUMINAMATH_CALUDE_max_books_borrowed_l2599_259919

theorem max_books_borrowed (total_students : ℕ) 
  (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (three_plus_books : ℕ) (five_plus_books : ℕ) 
  (average_books : ℝ) :
  total_students = 100 ∧ 
  zero_books = 5 ∧ 
  one_book = 20 ∧ 
  two_books = 25 ∧ 
  three_plus_books = 30 ∧ 
  five_plus_books = 20 ∧ 
  average_books = 3 →
  ∃ (max_books : ℕ), 
    max_books = 50 ∧ 
    ∀ (student_books : ℕ), 
      student_books ≤ max_books :=
by
  sorry

#check max_books_borrowed

end NUMINAMATH_CALUDE_max_books_borrowed_l2599_259919


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l2599_259975

theorem max_digits_product_5_4 : ∀ a b : ℕ, 
  10000 ≤ a ∧ a < 100000 → 1000 ≤ b ∧ b < 10000 → 
  a * b < 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l2599_259975


namespace NUMINAMATH_CALUDE_delta_computation_l2599_259969

-- Define the new operation
def delta (a b : ℕ) : ℕ := a^3 - b

-- State the theorem
theorem delta_computation :
  delta (5^(delta 6 8)) (4^(delta 2 7)) = 5^624 - 4 := by
  sorry

end NUMINAMATH_CALUDE_delta_computation_l2599_259969


namespace NUMINAMATH_CALUDE_sum_of_ages_l2599_259943

theorem sum_of_ages (age1 age2 : ℕ) : 
  age2 = age1 + 1 → age1 = 13 → age2 = 14 → age1 + age2 = 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2599_259943


namespace NUMINAMATH_CALUDE_triangle_properties_l2599_259907

noncomputable section

-- Define the triangle ABC
variable (A B C : Real) -- Angles
variable (a b c : Real) -- Side lengths

-- Define the conditions
axiom angle_side_relation : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c
axiom c_value : c = Real.sqrt 7
axiom triangle_area : 1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2

-- Theorem to prove
theorem triangle_properties : C = π/3 ∧ a + b + c = 5 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2599_259907


namespace NUMINAMATH_CALUDE_unique_intersection_point_l2599_259967

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 5*x^2 + 12*x + 20

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-5, -5) := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l2599_259967


namespace NUMINAMATH_CALUDE_max_distance_to_upper_vertex_l2599_259994

def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

def upper_vertex (B : ℝ × ℝ) : Prop :=
  B.1 = 0 ∧ B.2 = 1 ∧ ellipse B.1 B.2

theorem max_distance_to_upper_vertex :
  ∃ (B : ℝ × ℝ), upper_vertex B ∧
  ∀ (P : ℝ × ℝ), ellipse P.1 P.2 →
  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≤ 5/2 :=
sorry

end NUMINAMATH_CALUDE_max_distance_to_upper_vertex_l2599_259994


namespace NUMINAMATH_CALUDE_inclination_angle_of_line_l2599_259963

/-- The inclination angle of a line with equation ax + by + c = 0 is the angle between the positive x-axis and the line. -/
def InclinationAngle (a b c : ℝ) : ℝ := sorry

/-- The line equation sqrt(3)x + y - 1 = 0 -/
def LineEquation (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 1 = 0

theorem inclination_angle_of_line :
  InclinationAngle (Real.sqrt 3) 1 (-1) = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_inclination_angle_of_line_l2599_259963


namespace NUMINAMATH_CALUDE_janet_savings_l2599_259951

theorem janet_savings (monthly_rent : ℕ) (advance_months : ℕ) (deposit : ℕ) (additional_needed : ℕ) : 
  monthly_rent = 1250 →
  advance_months = 2 →
  deposit = 500 →
  additional_needed = 775 →
  monthly_rent * advance_months + deposit - additional_needed = 2225 :=
by sorry

end NUMINAMATH_CALUDE_janet_savings_l2599_259951


namespace NUMINAMATH_CALUDE_smallest_square_l2599_259912

theorem smallest_square (a b : ℕ+) 
  (h1 : ∃ r : ℕ+, (15 * a + 16 * b : ℕ) = r ^ 2)
  (h2 : ∃ s : ℕ+, (16 * a - 15 * b : ℕ) = s ^ 2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 481 ^ 2 ∧
  ∃ (a₀ b₀ : ℕ+), (15 * a₀ + 16 * b₀ : ℕ) = 481 ^ 2 ∧ (16 * a₀ - 15 * b₀ : ℕ) = 481 ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_l2599_259912


namespace NUMINAMATH_CALUDE_camp_attendance_l2599_259911

/-- The total number of kids in Lawrence county -/
def total_kids : ℕ := 1363293

/-- The number of kids who stay home -/
def kids_at_home : ℕ := 907611

/-- The number of kids who go to camp -/
def kids_at_camp : ℕ := total_kids - kids_at_home

theorem camp_attendance : kids_at_camp = 455682 := by
  sorry

end NUMINAMATH_CALUDE_camp_attendance_l2599_259911


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l2599_259914

/-- The product of the given numbers -/
def product : ℕ := 101 * 103 * 105 * 107

/-- The set of prime factors of the product -/
def prime_factors : Finset ℕ := sorry

theorem distinct_prime_factors_count :
  Finset.card prime_factors = 6 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l2599_259914


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2599_259906

theorem complex_magnitude_problem (z : ℂ) : z = (3 + I) / (2 - I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2599_259906


namespace NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_l2599_259929

theorem complex_exp_thirteen_pi_over_two (z : ℂ) : z = Complex.exp (13 * Real.pi * Complex.I / 2) → z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_l2599_259929


namespace NUMINAMATH_CALUDE_number_of_gigs_played_l2599_259950

-- Define the earnings for each band member
def lead_singer_earnings : ℕ := 30
def guitarist_earnings : ℕ := 25
def bassist_earnings : ℕ := 20
def drummer_earnings : ℕ := 25
def keyboardist_earnings : ℕ := 20
def backup_singer_earnings : ℕ := 15

-- Define the total earnings per gig
def earnings_per_gig : ℕ := lead_singer_earnings + guitarist_earnings + bassist_earnings + 
                            drummer_earnings + keyboardist_earnings + backup_singer_earnings

-- Define the total earnings from all gigs
def total_earnings : ℕ := 2055

-- Theorem: The number of gigs played is 15
theorem number_of_gigs_played : 
  ⌊(total_earnings : ℚ) / (earnings_per_gig : ℚ)⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_number_of_gigs_played_l2599_259950


namespace NUMINAMATH_CALUDE_divisors_of_60_l2599_259900

/-- The number of positive divisors of 60 is 12. -/
theorem divisors_of_60 : Nat.card (Nat.divisors 60) = 12 := by sorry

end NUMINAMATH_CALUDE_divisors_of_60_l2599_259900


namespace NUMINAMATH_CALUDE_fifth_valid_number_is_443_l2599_259964

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is valid (less than or equal to 600) --/
def isValidNumber (n : Nat) : Bool :=
  n ≤ 600

/-- Finds the nth valid number in a list --/
def findNthValidNumber (numbers : List Nat) (n : Nat) : Option Nat :=
  let validNumbers := numbers.filter isValidNumber
  validNumbers.get? (n - 1)

/-- The given random number table (partial) --/
def givenTable : RandomNumberTable :=
  [[84, 42, 17, 53, 31, 57, 24, 55, 6, 88, 77, 4, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 6, 76, 63, 1, 63],
   [78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 7, 44, 39, 52, 38, 79, 33, 21, 12, 34, 29, 78],
   [64, 56, 7, 82, 52, 42, 7, 44, 38, 15, 51, 0, 13, 42, 99, 66, 2, 79, 54]]

/-- The main theorem --/
theorem fifth_valid_number_is_443 :
  let numbers := (givenTable.get! 1).drop 7 ++ (givenTable.get! 2) ++ (givenTable.get! 3)
  findNthValidNumber numbers 5 = some 443 := by
  sorry

end NUMINAMATH_CALUDE_fifth_valid_number_is_443_l2599_259964


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l2599_259972

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) :
  (x * x^(1/3))^(1/4) = x^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l2599_259972


namespace NUMINAMATH_CALUDE_fresh_fruit_amount_l2599_259958

-- Define the total amount of fruit sold
def total_fruit : ℕ := 9792

-- Define the amount of frozen fruit sold
def frozen_fruit : ℕ := 3513

-- Define the amount of fresh fruit sold
def fresh_fruit : ℕ := total_fruit - frozen_fruit

-- Theorem to prove
theorem fresh_fruit_amount : fresh_fruit = 6279 := by
  sorry

end NUMINAMATH_CALUDE_fresh_fruit_amount_l2599_259958


namespace NUMINAMATH_CALUDE_fair_coin_four_tosses_l2599_259953

/-- A fair coin is a coin with equal probability of landing on either side -/
def fairCoin (p : ℝ) : Prop := p = 1/2

/-- The probability of n consecutive tosses landing on the same side -/
def consecutiveSameSide (p : ℝ) (n : ℕ) : ℝ := p^(n-1)

/-- Theorem: The probability of a fair coin landing on the same side 4 times in a row is 1/8 -/
theorem fair_coin_four_tosses (p : ℝ) (h : fairCoin p) : consecutiveSameSide p 4 = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_fair_coin_four_tosses_l2599_259953


namespace NUMINAMATH_CALUDE_complex_product_equals_33_l2599_259996

theorem complex_product_equals_33 (x : ℂ) (h : x = Complex.exp (2 * π * I / 9)) :
  (2 * x + x^2) * (2 * x^2 + x^4) * (2 * x^3 + x^6) * (2 * x^4 + x^8) = 33 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_33_l2599_259996


namespace NUMINAMATH_CALUDE_mr_a_net_gain_l2599_259927

def initial_value : ℚ := 12000
def first_sale_profit : ℚ := 20 / 100
def second_sale_loss : ℚ := 15 / 100
def third_sale_profit : ℚ := 10 / 100

theorem mr_a_net_gain : 
  let first_sale := initial_value * (1 + first_sale_profit)
  let second_sale := first_sale * (1 - second_sale_loss)
  let third_sale := second_sale * (1 + third_sale_profit)
  first_sale - second_sale + third_sale - initial_value = 3384 := by
sorry

end NUMINAMATH_CALUDE_mr_a_net_gain_l2599_259927


namespace NUMINAMATH_CALUDE_min_rooks_correct_min_rooks_minimal_l2599_259971

/-- A function that returns the minimum number of rooks needed on an n × n board
    to guarantee k non-attacking rooks can be selected. -/
def min_rooks (n k : ℕ) : ℕ :=
  n * (k - 1) + 1

/-- Theorem stating that min_rooks gives the correct minimum number of rooks. -/
theorem min_rooks_correct (n k : ℕ) (h1 : 1 < k) (h2 : k ≤ n) :
  ∀ (m : ℕ), m ≥ min_rooks n k →
    ∀ (placement : Fin m → Fin n × Fin n),
      ∃ (selected : Fin k → Fin m),
        ∀ (i j : Fin k), i ≠ j →
          (placement (selected i)).1 ≠ (placement (selected j)).1 ∧
          (placement (selected i)).2 ≠ (placement (selected j)).2 :=
by
  sorry

/-- Theorem stating that min_rooks gives the smallest such number. -/
theorem min_rooks_minimal (n k : ℕ) (h1 : 1 < k) (h2 : k ≤ n) :
  ∀ (m : ℕ), m < min_rooks n k →
    ∃ (placement : Fin m → Fin n × Fin n),
      ∀ (selected : Fin k → Fin m),
        ∃ (i j : Fin k), i ≠ j ∧
          ((placement (selected i)).1 = (placement (selected j)).1 ∨
           (placement (selected i)).2 = (placement (selected j)).2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_rooks_correct_min_rooks_minimal_l2599_259971


namespace NUMINAMATH_CALUDE_function_intersects_x_axis_l2599_259974

/-- A function f(x) = kx² - 2x - 1 intersects the x-axis if and only if k ≥ -1 -/
theorem function_intersects_x_axis (k : ℝ) :
  (∃ x, k * x^2 - 2*x - 1 = 0) ↔ k ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_function_intersects_x_axis_l2599_259974


namespace NUMINAMATH_CALUDE_paint_remaining_l2599_259938

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint = 1 →
  let remaining_after_day1 := initial_paint - (3/8 * initial_paint)
  let remaining_after_day2 := remaining_after_day1 - (1/4 * remaining_after_day1)
  remaining_after_day2 = 15/32 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_l2599_259938


namespace NUMINAMATH_CALUDE_smallest_q_for_inequality_l2599_259977

theorem smallest_q_for_inequality : ∃ (q : ℕ+), 
  (q = 2015) ∧ 
  (∀ (q' : ℕ+), q' < q → 
    ∃ (m : ℕ), 1 ≤ m ∧ m ≤ 1006 ∧ 
      ∀ (n : ℤ), (↑m / 1007 : ℚ) * ↑q' ≥ ↑n ∨ ↑n ≥ (↑(m + 1) / 1008 : ℚ) * ↑q') ∧
  (∀ (m : ℕ), 1 ≤ m → m ≤ 1006 → 
    ∃ (n : ℤ), (↑m / 1007 : ℚ) * ↑q < ↑n ∧ ↑n < (↑(m + 1) / 1008 : ℚ) * ↑q) :=
by sorry

end NUMINAMATH_CALUDE_smallest_q_for_inequality_l2599_259977


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2599_259934

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + k^2 - 1 = 0) ↔ 
  (-2 / Real.sqrt 3 ≤ k ∧ k ≤ 2 / Real.sqrt 3 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2599_259934


namespace NUMINAMATH_CALUDE_murtha_pebbles_l2599_259925

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Murtha's pebble collection problem -/
theorem murtha_pebbles : arithmetic_sum 3 3 18 = 513 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebbles_l2599_259925


namespace NUMINAMATH_CALUDE_square_equation_solution_l2599_259981

theorem square_equation_solution :
  ∃ x : ℝ, (3000 + x)^2 = x^2 ∧ x = -1500 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2599_259981


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2599_259992

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x < -1 ∨ x > 3/2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2599_259992


namespace NUMINAMATH_CALUDE_lines_without_common_point_are_parallel_or_skew_l2599_259976

-- Define a type for straight lines in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- or any other suitable representation
  -- This is just a placeholder structure
  mk :: (dummy : Unit)

-- Define the property of two lines not having a common point
def noCommonPoint (a b : Line3D) : Prop :=
  -- The actual implementation would depend on how you define Line3D
  sorry

-- Define the property of two lines being parallel
def parallel (a b : Line3D) : Prop :=
  -- The actual implementation would depend on how you define Line3D
  sorry

-- Define the property of two lines being skew
def skew (a b : Line3D) : Prop :=
  -- The actual implementation would depend on how you define Line3D
  sorry

-- The theorem statement
theorem lines_without_common_point_are_parallel_or_skew 
  (a b : Line3D) (h : noCommonPoint a b) : 
  parallel a b ∨ skew a b :=
sorry

end NUMINAMATH_CALUDE_lines_without_common_point_are_parallel_or_skew_l2599_259976


namespace NUMINAMATH_CALUDE_initial_red_marbles_l2599_259954

/-- Represents the number of marbles in a bag -/
structure MarbleBag where
  red : ℚ
  green : ℚ

/-- The initial ratio of red to green marbles is 7:3 -/
def initial_ratio (bag : MarbleBag) : Prop :=
  bag.red / bag.green = 7 / 3

/-- After removing 14 red marbles and adding 30 green marbles, the new ratio is 1:4 -/
def new_ratio (bag : MarbleBag) : Prop :=
  (bag.red - 14) / (bag.green + 30) = 1 / 4

/-- Theorem stating that the initial number of red marbles is 24 -/
theorem initial_red_marbles (bag : MarbleBag) :
  initial_ratio bag → new_ratio bag → bag.red = 24 := by
  sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l2599_259954


namespace NUMINAMATH_CALUDE_percentage_of_120_to_40_l2599_259909

theorem percentage_of_120_to_40 : ∃ (p : ℝ), p = (120 : ℝ) / 40 * 100 ∧ p = 300 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_40_l2599_259909


namespace NUMINAMATH_CALUDE_integral_shift_reciprocal_l2599_259955

open MeasureTheory

-- Define the function f and the integral L
variable (f : ℝ → ℝ)
variable (L : ℝ)

-- State the theorem
theorem integral_shift_reciprocal (hf : Continuous f) 
  (hL : ∫ (x : ℝ), f x = L) :
  ∫ (x : ℝ), f (x - 1/x) = L := by
  sorry

end NUMINAMATH_CALUDE_integral_shift_reciprocal_l2599_259955


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2599_259913

theorem quadratic_equation_solution :
  ∃ x : ℝ, 4 * x^2 - 12 * x + 9 = 0 ∧ x = 3/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2599_259913


namespace NUMINAMATH_CALUDE_value_of_x_l2599_259942

theorem value_of_x (x y z : ℝ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2599_259942


namespace NUMINAMATH_CALUDE_kangaroo_jump_theorem_l2599_259916

theorem kangaroo_jump_theorem :
  ∃ (a b c d : ℕ),
    a + b + c + d = 30 ∧
    7 * a + 5 * b + 3 * c - 3 * d = 200 ∧
    (a = 25 ∧ c = 5 ∧ b = 0 ∧ d = 0) ∨
    (a = 26 ∧ b = 3 ∧ c = 1 ∧ d = 0) ∨
    (a = 27 ∧ b = 1 ∧ c = 2 ∧ d = 0) ∨
    (a = 29 ∧ d = 1 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_kangaroo_jump_theorem_l2599_259916


namespace NUMINAMATH_CALUDE_prob_at_most_one_white_ball_l2599_259984

/-- The number of black balls in the box -/
def black_balls : ℕ := 10

/-- The number of red balls in the box -/
def red_balls : ℕ := 12

/-- The number of white balls in the box -/
def white_balls : ℕ := 4

/-- The total number of balls in the box -/
def total_balls : ℕ := black_balls + red_balls + white_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 2

/-- X represents the number of white balls drawn -/
def X : Fin (drawn_balls + 1) → ℕ := sorry

/-- The probability of drawing at most one white ball -/
def P_X_le_1 : ℚ := sorry

/-- The main theorem to prove -/
theorem prob_at_most_one_white_ball :
  P_X_le_1 = (Nat.choose (total_balls - white_balls) 1 * Nat.choose white_balls 1 + 
              Nat.choose (total_balls - white_balls) 2) / 
             Nat.choose total_balls 2 :=
sorry

end NUMINAMATH_CALUDE_prob_at_most_one_white_ball_l2599_259984


namespace NUMINAMATH_CALUDE_pages_difference_l2599_259961

theorem pages_difference (beatrix_pages cristobal_pages : ℕ) : 
  beatrix_pages = 704 →
  cristobal_pages = 3 * beatrix_pages + 15 →
  cristobal_pages - beatrix_pages = 1423 := by
sorry

end NUMINAMATH_CALUDE_pages_difference_l2599_259961


namespace NUMINAMATH_CALUDE_f_minimum_and_tangents_l2599_259922

noncomputable def f (x : ℝ) : ℝ := x * (Real.log x + 1)

theorem f_minimum_and_tangents 
  (a b : ℝ) 
  (h1 : 0 < b) (h2 : b < a * Real.log a + a) :
  (∃ (min : ℝ), min = -Real.exp (-2) ∧ ∀ x > 0, f x ≥ min) ∧
  (∃ (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    x₁ > Real.exp (-2) ∧ 
    x₂ > Real.exp (-2) ∧
    b - f x₁ = (Real.log x₁ + 2) * (a - x₁) ∧
    b - f x₂ = (Real.log x₂ + 2) * (a - x₂)) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_and_tangents_l2599_259922


namespace NUMINAMATH_CALUDE_tangent_points_tangent_parallel_points_l2599_259957

/-- The function f(x) = x³ + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_points :
  ∀ x : ℝ, (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
by sorry

theorem tangent_parallel_points :
  ∀ x : ℝ, (f' x = 4) → (x = 1 ∧ f x = 0) ∨ (x = -1 ∧ f x = -4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_points_tangent_parallel_points_l2599_259957


namespace NUMINAMATH_CALUDE_school_trip_student_count_l2599_259959

theorem school_trip_student_count :
  let num_buses : ℕ := 95
  let max_seats_per_bus : ℕ := 118
  let bus_capacity_percentage : ℚ := 9/10
  let attendance_percentage : ℚ := 4/5
  let total_students : ℕ := 12588
  (↑num_buses * ↑max_seats_per_bus * bus_capacity_percentage).floor = 
    (↑total_students * attendance_percentage).floor := by
  sorry

end NUMINAMATH_CALUDE_school_trip_student_count_l2599_259959


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l2599_259962

theorem geometric_series_first_term 
  (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l2599_259962


namespace NUMINAMATH_CALUDE_red_ball_probability_l2599_259901

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  red : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (counts : BallCounts) : ℕ :=
  counts.red + counts.yellow + counts.white

/-- Calculates the probability of drawing a ball of a specific color -/
def drawProbability (counts : BallCounts) (color : ℕ) : ℚ :=
  color / (totalBalls counts)

/-- Theorem: The probability of drawing a red ball from a bag with 3 red, 5 yellow, and 2 white balls is 3/10 -/
theorem red_ball_probability :
  let bag := BallCounts.mk 3 5 2
  drawProbability bag bag.red = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_l2599_259901


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l2599_259937

theorem max_value_trig_expression :
  ∀ x y z : ℝ, 
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) * 
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 
  (9 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l2599_259937


namespace NUMINAMATH_CALUDE_parallelepiped_diagonal_squared_l2599_259973

/-- The square of the diagonal of a rectangular parallelepiped is equal to the sum of squares of its dimensions -/
theorem parallelepiped_diagonal_squared (p q r : ℝ) :
  let diagonal_squared := p^2 + q^2 + r^2
  diagonal_squared = p^2 + q^2 + r^2 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_diagonal_squared_l2599_259973


namespace NUMINAMATH_CALUDE_framed_picture_perimeter_is_six_feet_l2599_259970

/-- Calculates the perimeter of a framed picture given original dimensions and scaling factor. -/
def framedPicturePerimeter (width height scale border : ℚ) : ℚ :=
  2 * (width * scale + height * scale + 2 * border)

/-- Converts inches to feet -/
def inchesToFeet (inches : ℚ) : ℚ :=
  inches / 12

theorem framed_picture_perimeter_is_six_feet :
  let originalWidth : ℚ := 3
  let originalHeight : ℚ := 5
  let scaleFactor : ℚ := 3
  let borderWidth : ℚ := 3
  
  inchesToFeet (framedPicturePerimeter originalWidth originalHeight scaleFactor borderWidth) = 6 := by
  sorry

end NUMINAMATH_CALUDE_framed_picture_perimeter_is_six_feet_l2599_259970


namespace NUMINAMATH_CALUDE_coffee_payment_dimes_l2599_259924

/-- Represents the number of coins of each type used in the payment -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The total value of the coins in cents -/
def totalValue (c : CoinCount) : ℕ :=
  c.pennies + 5 * c.nickels + 10 * c.dimes

/-- The total number of coins -/
def totalCoins (c : CoinCount) : ℕ :=
  c.pennies + c.nickels + c.dimes

theorem coffee_payment_dimes :
  ∃ (c : CoinCount),
    totalValue c = 200 ∧
    totalCoins c = 50 ∧
    c.dimes = 14 :=
by sorry

end NUMINAMATH_CALUDE_coffee_payment_dimes_l2599_259924


namespace NUMINAMATH_CALUDE_f_difference_at_five_l2599_259987

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 3*x^3 + 5*x

-- Theorem statement
theorem f_difference_at_five : f 5 - f (-5) = 800 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_five_l2599_259987


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2599_259931

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 2 * x - 1
  ∀ x : ℝ, f x = 0 ↔ x = -1/3 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2599_259931


namespace NUMINAMATH_CALUDE_binomial_seven_one_l2599_259918

theorem binomial_seven_one : (7 : ℕ).choose 1 = 7 := by sorry

end NUMINAMATH_CALUDE_binomial_seven_one_l2599_259918


namespace NUMINAMATH_CALUDE_exists_b_for_234_quadrants_l2599_259986

-- Define the linear function
def f (b : ℝ) (x : ℝ) : ℝ := -2 * x + b

-- Define the property of passing through the second, third, and fourth quadrants
def passes_through_234_quadrants (b : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ,
    (x₁ < 0 ∧ f b x₁ > 0) ∧  -- Second quadrant
    (x₂ < 0 ∧ f b x₂ < 0) ∧  -- Third quadrant
    (x₃ > 0 ∧ f b x₃ < 0)    -- Fourth quadrant

-- Theorem statement
theorem exists_b_for_234_quadrants :
  ∃ b : ℝ, b < 0 ∧ passes_through_234_quadrants b :=
sorry

end NUMINAMATH_CALUDE_exists_b_for_234_quadrants_l2599_259986


namespace NUMINAMATH_CALUDE_correct_ages_l2599_259988

/-- Represents the ages of family members -/
structure FamilyAges where
  man : ℕ
  son : ℕ
  sibling : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  (ages.man = ages.son + 30) ∧
  (ages.man + 2 = 2 * (ages.son + 2)) ∧
  (ages.sibling + 2 = (ages.son + 2) / 2)

/-- Theorem stating that the ages 58, 28, and 13 satisfy the conditions -/
theorem correct_ages : 
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ ages.son = 28 ∧ ages.sibling = 13 :=
by
  sorry


end NUMINAMATH_CALUDE_correct_ages_l2599_259988


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_expression_l2599_259956

theorem smallest_prime_factor_of_expression : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (12^3 + 15^4 - 6^6) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (12^3 + 15^4 - 6^6) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_expression_l2599_259956


namespace NUMINAMATH_CALUDE_snowboard_final_price_l2599_259965

/-- Calculates the final price of an item after applying two discounts and a sales tax. -/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (salesTax : ℝ) : ℝ :=
  let priceAfterDiscount1 := originalPrice * (1 - discount1)
  let priceAfterDiscount2 := priceAfterDiscount1 * (1 - discount2)
  priceAfterDiscount2 * (1 + salesTax)

/-- Theorem stating that the final price of a $200 snowboard after 40% and 20% discounts
    and 5% sales tax is $100.80. -/
theorem snowboard_final_price :
  finalPrice 200 0.4 0.2 0.05 = 100.80 := by
  sorry

end NUMINAMATH_CALUDE_snowboard_final_price_l2599_259965


namespace NUMINAMATH_CALUDE_age_problem_solution_l2599_259948

/-- The ages of the king and queen satisfy the given conditions -/
def age_problem (king_age queen_age : ℕ) : Prop :=
  ∃ (t : ℕ),
    -- The king's current age is twice the queen's age when the king was as old as the queen is now
    king_age = 2 * (queen_age - t) ∧
    -- When the queen is as old as the king is now, their combined ages will be 63 years
    king_age + (king_age + t) = 63 ∧
    -- The age difference
    king_age - queen_age = t

/-- The solution to the age problem -/
theorem age_problem_solution :
  ∃ (king_age queen_age : ℕ), age_problem king_age queen_age ∧ king_age = 28 ∧ queen_age = 21 :=
sorry

end NUMINAMATH_CALUDE_age_problem_solution_l2599_259948


namespace NUMINAMATH_CALUDE_football_games_per_month_l2599_259978

theorem football_games_per_month 
  (total_games : ℕ) 
  (num_months : ℕ) 
  (h1 : total_games = 323) 
  (h2 : num_months = 17) 
  (h3 : total_games % num_months = 0) : 
  total_games / num_months = 19 := by
sorry

end NUMINAMATH_CALUDE_football_games_per_month_l2599_259978


namespace NUMINAMATH_CALUDE_circular_arrangement_theorem_l2599_259940

/-- Represents a circular seating arrangement of men and women -/
structure CircularArrangement where
  total_people : ℕ
  women : ℕ
  men : ℕ
  women_left_of_women : ℕ
  men_left_of_women : ℕ
  women_right_of_men_ratio : ℚ

/-- The properties of the circular arrangement in the problem -/
def problem_arrangement : CircularArrangement where
  total_people := 35
  women := 19
  men := 16
  women_left_of_women := 7
  men_left_of_women := 12
  women_right_of_men_ratio := 3/4

theorem circular_arrangement_theorem (arr : CircularArrangement) :
  arr.women_left_of_women = 7 ∧
  arr.men_left_of_women = 12 ∧
  arr.women_right_of_men_ratio = 3/4 →
  arr.total_people = 35 ∧
  arr.women = 19 ∧
  arr.men = 16 := by
  sorry

#check circular_arrangement_theorem problem_arrangement

end NUMINAMATH_CALUDE_circular_arrangement_theorem_l2599_259940


namespace NUMINAMATH_CALUDE_cubic_sum_implies_square_sum_less_than_one_l2599_259910

theorem cubic_sum_implies_square_sum_less_than_one 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x^3 + y^3 = x - y) : 
  x^2 + y^2 < 1 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_implies_square_sum_less_than_one_l2599_259910


namespace NUMINAMATH_CALUDE_toonies_count_l2599_259905

/-- Represents the number of toonies in a set of coins --/
def num_toonies (total_coins : ℕ) (total_value : ℕ) : ℕ :=
  total_coins - (2 * total_coins - total_value)

/-- Theorem stating that given 10 coins with a total value of $14, 
    the number of $2 coins (toonies) is 4 --/
theorem toonies_count : num_toonies 10 14 = 4 := by
  sorry

#eval num_toonies 10 14  -- Should output 4

end NUMINAMATH_CALUDE_toonies_count_l2599_259905


namespace NUMINAMATH_CALUDE_no_additional_savings_when_purchasing_together_l2599_259926

/-- Represents the store's window offer -/
structure WindowOffer where
  price : ℕ  -- Price per window
  buy : ℕ    -- Number of windows to buy
  free : ℕ   -- Number of free windows

/-- Calculates the cost for a given number of windows under the offer -/
def calculateCost (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let fullSets := windowsNeeded / (offer.buy + offer.free)
  let remainingWindows := windowsNeeded % (offer.buy + offer.free)
  fullSets * (offer.price * offer.buy) + min remainingWindows offer.buy * offer.price

/-- Theorem stating that there's no additional savings when purchasing together -/
theorem no_additional_savings_when_purchasing_together 
  (offer : WindowOffer)
  (daveWindows : ℕ)
  (dougWindows : ℕ) :
  offer.price = 150 ∧ 
  offer.buy = 6 ∧ 
  offer.free = 2 ∧
  daveWindows = 9 ∧
  dougWindows = 10 →
  (calculateCost offer daveWindows + calculateCost offer dougWindows) - 
  calculateCost offer (daveWindows + dougWindows) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_additional_savings_when_purchasing_together_l2599_259926


namespace NUMINAMATH_CALUDE_floor_of_expression_equals_32_l2599_259993

theorem floor_of_expression_equals_32 :
  ⌊(1 + (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 4) / 
     (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 8 + 4))^10⌋ = 32 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_expression_equals_32_l2599_259993


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_fifty_l2599_259991

theorem largest_multiple_of_seven_below_negative_fifty :
  ∀ n : ℤ, n * 7 < -50 → n * 7 ≤ -56 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_fifty_l2599_259991


namespace NUMINAMATH_CALUDE_line_equation_through_points_l2599_259930

/-- The equation of a line passing through two given points -/
theorem line_equation_through_points (x y : ℝ) : 
  (2 * x - y - 2 = 0) ↔ 
  ((x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = -2) ∨ 
   (∃ t : ℝ, x = 1 - t ∧ y = 0 + 2*t)) :=
sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l2599_259930


namespace NUMINAMATH_CALUDE_prob_at_least_three_babies_speak_l2599_259921

/-- The probability that at least 3 out of 6 babies will speak tomorrow, 
    given that each baby has a 1/3 probability of speaking. -/
theorem prob_at_least_three_babies_speak (n : ℕ) (p : ℝ) : 
  n = 6 → p = 1/3 → 
  (1 : ℝ) - (Nat.choose n 0 * (1 - p)^n + 
             Nat.choose n 1 * p * (1 - p)^(n-1) + 
             Nat.choose n 2 * p^2 * (1 - p)^(n-2)) = 353/729 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_babies_speak_l2599_259921


namespace NUMINAMATH_CALUDE_calculus_class_mean_l2599_259999

/-- Calculates the class mean given the number of students and average scores for three groups -/
def class_mean (total_students : ℕ) (group1_students : ℕ) (group1_avg : ℚ) 
               (group2_students : ℕ) (group2_avg : ℚ)
               (group3_students : ℕ) (group3_avg : ℚ) : ℚ :=
  (group1_students * group1_avg + group2_students * group2_avg + group3_students * group3_avg) / total_students

theorem calculus_class_mean :
  let total_students : ℕ := 60
  let group1_students : ℕ := 40
  let group1_avg : ℚ := 68 / 100
  let group2_students : ℕ := 15
  let group2_avg : ℚ := 74 / 100
  let group3_students : ℕ := 5
  let group3_avg : ℚ := 88 / 100
  class_mean total_students group1_students group1_avg group2_students group2_avg group3_students group3_avg = 4270 / 60 :=
by sorry

end NUMINAMATH_CALUDE_calculus_class_mean_l2599_259999


namespace NUMINAMATH_CALUDE_johns_allowance_l2599_259998

/-- John's weekly allowance problem -/
theorem johns_allowance :
  ∀ (A : ℚ),
  (A > 0) →
  (3 / 5 * A + 1 / 3 * (2 / 5 * A) + 0.4 = A) →
  A = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l2599_259998


namespace NUMINAMATH_CALUDE_min_disks_needed_l2599_259902

def total_files : ℕ := 40
def disk_capacity : ℚ := 1.44
def large_files : ℕ := 5
def medium_files : ℕ := 15
def small_files : ℕ := total_files - large_files - medium_files
def large_file_size : ℚ := 0.9
def medium_file_size : ℚ := 0.75
def small_file_size : ℚ := 0.5

theorem min_disks_needed :
  let total_size := large_files * large_file_size + medium_files * medium_file_size + small_files * small_file_size
  ∃ (n : ℕ), n * disk_capacity ≥ total_size ∧
             ∀ (m : ℕ), m * disk_capacity ≥ total_size → n ≤ m ∧
             n = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_disks_needed_l2599_259902


namespace NUMINAMATH_CALUDE_composite_rectangle_theorem_l2599_259985

/-- The side length of square S2 in the composite rectangle. -/
def side_length_S2 : ℕ := 775

/-- The width of the composite rectangle. -/
def total_width : ℕ := 4000

/-- The height of the composite rectangle. -/
def total_height : ℕ := 2450

/-- The shorter side length of rectangles R1 and R2. -/
def shorter_side_R : ℕ := (total_height - side_length_S2) / 2

theorem composite_rectangle_theorem :
  (2 * shorter_side_R + side_length_S2 = total_height) ∧
  (2 * shorter_side_R + 3 * side_length_S2 = total_width) := by
  sorry

#check composite_rectangle_theorem

end NUMINAMATH_CALUDE_composite_rectangle_theorem_l2599_259985


namespace NUMINAMATH_CALUDE_seven_power_minus_three_times_two_power_eq_one_solutions_l2599_259966

theorem seven_power_minus_three_times_two_power_eq_one_solutions :
  ∀ x y : ℕ, 7^x - 3 * 2^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_seven_power_minus_three_times_two_power_eq_one_solutions_l2599_259966


namespace NUMINAMATH_CALUDE_triangle_angle_C_l2599_259932

noncomputable def f (x φ : Real) : Real :=
  2 * Real.sin x * (Real.cos (φ / 2))^2 + Real.cos x * Real.sin φ - Real.sin x

theorem triangle_angle_C (φ A B C : Real) (a b c : Real) :
  0 < φ ∧ φ < Real.pi ∧
  (∀ x, f x φ ≥ f Real.pi φ) ∧
  Real.cos (2 * C) - Real.cos (2 * A) = 2 * Real.sin (Real.pi / 3 + C) * Real.sin (Real.pi / 3 - C) ∧
  a = 1 ∧
  b = Real.sqrt 2 ∧
  f A φ = Real.sqrt 3 / 2 ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C
  →
  C = 7 * Real.pi / 12 ∨ C = Real.pi / 12 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l2599_259932


namespace NUMINAMATH_CALUDE_son_age_proof_l2599_259952

theorem son_age_proof (father_age son_age : ℕ) : 
  father_age = son_age + 29 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 27 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l2599_259952


namespace NUMINAMATH_CALUDE_four_of_a_kind_probability_l2599_259990

-- Define a standard deck of cards
def standardDeck : ℕ := 52

-- Define the number of cards drawn
def cardsDrawn : ℕ := 6

-- Define the number of different card values (ranks)
def cardValues : ℕ := 13

-- Define the number of cards of each value
def cardsPerValue : ℕ := 4

-- Function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem four_of_a_kind_probability :
  (cardValues * binomial (standardDeck - cardsPerValue) (cardsDrawn - cardsPerValue)) /
  (binomial standardDeck cardsDrawn) = 3 / 4165 :=
sorry

end NUMINAMATH_CALUDE_four_of_a_kind_probability_l2599_259990


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l2599_259960

theorem loss_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1500 → 
  selling_price = 1275 → 
  (cost_price - selling_price) / cost_price * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l2599_259960


namespace NUMINAMATH_CALUDE_parry_prob_secretary_or_treasurer_l2599_259915

-- Define the number of club members
def total_members : ℕ := 10

-- Define the probability of being chosen as secretary
def prob_secretary : ℚ := 1 / 9

-- Define the probability of being chosen as treasurer
def prob_treasurer : ℚ := 1 / 10

-- Theorem statement
theorem parry_prob_secretary_or_treasurer :
  let prob_either := prob_secretary + prob_treasurer
  prob_either = 19 / 90 := by
  sorry

end NUMINAMATH_CALUDE_parry_prob_secretary_or_treasurer_l2599_259915


namespace NUMINAMATH_CALUDE_zeroes_of_f_range_of_a_l2599_259941

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + b - 1

-- Theorem for the zeroes of f(x) when a = 1 and b = -2
theorem zeroes_of_f : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f 1 (-2) x₁ = 0 ∧ f 1 (-2) x₂ = 0 ∧ x₁ = 3 ∧ x₂ = -1 :=
sorry

-- Theorem for the range of a when f(x) always has two distinct zeroes
theorem range_of_a (a : ℝ) : 
  (a ≠ 0 ∧ ∀ b : ℝ, ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0) ↔ 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_zeroes_of_f_range_of_a_l2599_259941


namespace NUMINAMATH_CALUDE_quadratic_equation_transform_l2599_259936

theorem quadratic_equation_transform (x : ℝ) :
  25 * x^2 - 10 * x - 1000 = 0 →
  ∃ (r : ℝ), (x + r)^2 = 40.04 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_transform_l2599_259936
